"""Aggregate existing deterministic navigation reports and raw traces; no training.

Missing/censored values stay null. Axes with different units are never averaged
into a dimensional RMSE. Ranking is a pilot diagnostic, not an acceptance test.
"""
from pathlib import Path
import argparse
import csv
import json

import numpy as np


def mean(values):
  values = [v for v in values if v is not None]
  return float(np.mean(values)) if values else None


def summarize(directory):
  report = json.loads((directory / 'report.json').read_text())
  errors, low_errors, roll, pitch = [], [], [], []
  stops, drifts, recoveries, turns, yaws, overshoots = [], [], [], [], [], []
  low_forward = {}
  stop_paths, stop_residuals, turn_translation = [], [], []
  nonzero_response, transition_attitude = [], []
  num_transitions = num_stops = 0
  for trial in report['trials']:
    trace = np.load(directory / f"{trial['scenario']}_{trial['env_index']}.npz")
    t, c, v, p, rpy = [trace[key] for key in ('time', 'command', 'velocity', 'position', 'rpy')]
    err = v-c
    errors.append(err)
    mask = (np.abs(c) <= [.2, .15, .4]).all(1) & (np.abs(c).max(1) > 0)
    low_errors.append(err[mask])
    roll.extend(rpy[:, 0]); pitch.extend(rpy[:, 1])
    for index, segment in enumerate(trial['segments']):
      start = segment['start_s']
      end = trial['segments'][index+1]['start_s'] if index+1 < len(trial['segments']) else t[-1]
      cmd = np.array(segment['command'])
      active = (t > start + 1e-6) & (t <= end + 1e-6)
      steady = active & (t > start + 1.)
      if np.any(cmd != 0) and steady.any():
        scale = np.array([.2, .15, .4])
        response = np.mean(v[steady] / scale, axis=0) @ (cmd / scale) / np.sum((cmd / scale)**2)
        nonzero_response.append(float(response))
      if index > 0:
        num_transitions += 1
        recoveries.append(segment['recovery_time'])
        overshoots.append(segment['velocity_overshoot'])
        transition_attitude.append(float(np.abs(rpy[active, :2]).max()))
      if np.all(cmd == 0) and index > 0:
        num_stops += 1
        stops.append(segment['stopping_distance']); drifts.append(segment['post_stop_drift'])
        before = max(0, int(np.searchsorted(t, start + 1e-6))-1)
        first_s = active & (t <= start+1.+1e-6)
        points = np.vstack([p[before, :2], p[first_s, :2]])
        stop_paths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
        if steady.any():
          stop_residuals.append(float(np.mean(np.linalg.norm(v[steady, :2], axis=1))))
      if cmd[2] != 0:
        turns.append(abs(segment['turning_error']))
        if np.all(cmd[:2] == 0):
          before = max(0, int(np.searchsorted(t, start + 1e-6))-1)
          turn_translation.append(float(np.linalg.norm(p[active, :2]-p[before, :2], axis=1).max()))
      elif segment['yaw_drift'] is not None:
        yaws.append(abs(segment['yaw_drift']))
      if trial['scenario'].startswith('low_') and cmd[0] > 0:
        record = low_forward.setdefault(trial['scenario'], {'vx_mean': [], 'vx_rmse': [], 'bias': []})
        if steady.any():
          record['vx_mean'].append(float(v[steady, 0].mean()))
          record['bias'].append(float(err[steady, 0].mean()))
        record['vx_rmse'].append(float(np.sqrt(np.mean(err[active, 0]**2))))
  rmse = np.sqrt(np.mean(np.concatenate(errors)**2, axis=0))
  low = np.concatenate(low_errors)
  low_rmse = np.sqrt(np.mean(low**2, axis=0)).tolist() if len(low) else None
  index = int(directory.name.removeprefix('eval_'))
  row = dict(checkpoint=f'model_{index}.pt', updates=index+1,
    vx_rmse=float(rmse[0]), vy_rmse=float(rmse[1]), wz_rmse=float(rmse[2]),
    low_speed_rmse=low_rmse, stopping_distance=mean(stops), post_stop_drift=mean(drifts),
    transition_recovery_time=mean(recoveries),
    transition_recovery_rate=sum(v is not None for v in recoveries)/num_transitions if num_transitions else None,
    recovered_transitions=sum(v is not None for v in recoveries), observed_transitions=num_transitions,
    confirmed_stops=sum(v is not None for v in stops), observed_stops=num_stops,
    turning_error=mean(turns), yaw_drift=mean(yaws),
    velocity_overshoot=np.max(overshoots, axis=0).tolist() if overshoots else None,
    roll_rms=float(np.sqrt(np.mean(np.square(roll)))), pitch_rms=float(np.sqrt(np.mean(np.square(pitch)))),
    fall_rate=report['fall_rate'], completed_rate=mean([r['completed'] for r in report['trials']]),
    trials=len(report['trials']), samples=sum(len(e) for e in errors),
    low_forward={k: {metric: mean(values) for metric, values in values.items()} for k, values in low_forward.items()},
    stop_first_second_path_mean=mean(stop_paths), stop_after_one_second_speed_mean=mean(stop_residuals),
    turn_translation_max=max(turn_translation) if turn_translation else None,
    nonzero_response_below_20pct_rate=mean([v < .2 for v in nonzero_response]),
    transition_peak_roll_pitch=max(transition_attitude) if transition_attitude else None,
    report=str(directory / 'report.json'))
  # Screening thresholds are transparent diagnostics, not a tuned pass score.
  low005 = row['low_forward'].get('low_005', {}).get('vx_mean')
  low010 = row['low_forward'].get('low_010', {}).get('vx_mean')
  row['screening'] = dict(
    ZERO_COMMAND_COLLAPSE=(row['nonzero_response_below_20pct_rate'] > .5) if nonzero_response else None,
    LOW_SPEED_DEAD_ZONE=(low005 < .025 or low010 < .05) if low005 is not None and low010 is not None else None,
    OVERSPEED_OBSERVED=any(v['bias'] is not None and v['bias'] > .03 for v in row['low_forward'].values()) if low_forward else None,
    STOP_OVERSHOOT_OBSERVED=(max(stop_paths) > .05 or max(stop_residuals, default=0) > .03) if stop_paths else None,
    TURN_TRANSLATION_OBSERVED=max(turn_translation) > .05 if turn_translation else None,
    TRANSITION_INSTABILITY=(row['fall_rate'] > 0 or max(transition_attitude) > .35) if transition_attitude else None)
  return row


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=Path, default=Path('logs/navigation_phase_a1'))
  parser.add_argument('--training-dir', type=Path, required=True)
  args = parser.parse_args()
  rows = [summarize(p.parent) for p in sorted(args.root.glob('eval_*/report.json'), key=lambda p: int(p.parent.name[5:]))]
  if not rows:
    raise RuntimeError('No evaluation reports')
  rank = lambda r: (r['fall_rate'], -r['completed_rate'], -(r['transition_recovery_rate'] or 0),
                    np.linalg.norm(np.array([r['vx_rmse'], r['vy_rmse'], r['wz_rmse']]) / [.03, .03, .08]))
  eligible = [r for r in rows if r["observed_transitions"] > 0 and r["completed_rate"] > 0]
  best = min(eligible, key=rank) if eligible else None
  output = dict(ranking='lexicographic: fall_rate, -completed_rate, -transition_recovery_rate, tolerance-normalized tracking RMSE norm',
                aggregation='sample-pooled per-axis RMSE; segment-equal means for confirmed stops/recoveries; nulls excluded with counts reported; initial stand excluded from transitions/stops',
                best_checkpoint=best['checkpoint'] if best else None, checkpoints=rows)
  (args.root/'comparison.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
  columns = ['checkpoint','updates','vx_rmse','vy_rmse','wz_rmse','low_speed_rmse','stopping_distance','post_stop_drift','transition_recovery_time','transition_recovery_rate','confirmed_stops','observed_stops','turning_error','fall_rate']
  with (args.root/'comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  events = EventAccumulator(str(args.training_dir), size_guidance={'scalars': 0}).Reload()
  tags = events.Tags()['scalars']
  with (args.root/'learning_scalars.csv').open('w') as f:
    writer = csv.writer(f); writer.writerow(['tag','step','wall_time','value'])
    for tag in tags:
      for event in events.Scalars(tag):
        writer.writerow([tag,event.step,event.wall_time,event.value])
  required = ['track_vx','track_vy','track_wz','velocity_error','stop_velocity','stand_still','foot_gait','foot_slip','action_rate_l2','is_terminated']
  assert all('Episode_Reward/'+name in tags for name in required)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(2, 2, figsize=(12, 8))
  for ax, names in zip(axes.flat, [required[:3], required[3:6], required[6:], []]):
    for name in names:
      series=events.Scalars('Episode_Reward/'+name)
      ax.plot([s.step+1 for s in series], [s.value for s in series], label=name)
    ax.set_xlabel('PPO updates'); ax.grid(alpha=.25)
    if names: ax.legend(fontsize=8)
  for key in ['vx_rmse','vy_rmse','wz_rmse']:
    axes[1,1].plot([r['updates'] for r in rows], [r[key] for r in rows], 'o-', label=key)
  axes[1,1].legend(); axes[1,1].set_title('Benchmark: m/s (vx/vy), rad/s (wz)')
  fig.tight_layout(); fig.savefig(args.root/'learning_and_navigation_curves.png', dpi=160); plt.close(fig)
  fig, axes = plt.subplots(2, 3, figsize=(14, 8))
  updates = [r['updates'] for r in rows]
  for key, target in [('low_005', .05), ('low_010', .1), ('low_020', .2)]:
    values = [r['low_forward'].get(key, {}).get('vx_mean') for r in rows]
    axes[0,0].plot(updates, [v if v is not None else np.nan for v in values], 'o-', label=key)
    axes[0,0].axhline(target, linestyle=':', alpha=.3)
  axes[0,0].set_title('Steady forward velocity (m/s)'); axes[0,0].legend()
  for ax, key, title in zip(list(axes.flat)[1:5], ['stopping_distance','post_stop_drift','turning_error','transition_recovery_rate'], ['Confirmed stopping distance (m)','Post-stop drift (m)','Absolute turning error (rad)','Transition recovery fraction']):
    ax.plot(updates, [r[key] if r[key] is not None else np.nan for r in rows], 'o-')
    ax.set_title(title)
  axes[1,2].plot(updates, [r['fall_rate'] for r in rows], 'o-', label='fall rate')
  axes[1,2].plot(updates, [r['completed_rate'] for r in rows], 'o-', label='completion')
  axes[1,2].legend()
  for ax in axes.flat:
    ax.set_xlabel('PPO updates'); ax.grid(alpha=.25)
  fig.tight_layout(); fig.savefig(args.root/'navigation_detail_curves.png', dpi=160); plt.close(fig)
  last_dir = args.root / ('eval_' + str(rows[-1]['updates']-1))
  fig, axes = plt.subplots(4, 3, figsize=(14, 11))
  for row_axes, scenario in zip(axes, ['low_005','low_010','turn_stop','forward_backward']):
    trace = np.load(last_dir / (scenario + '_0.npz'))
    for axis, ax in enumerate(row_axes):
      ax.plot(trace['time'], trace['velocity'][:, axis], label='actual')
      ax.plot(trace['time'], trace['command'][:, axis], '--', label='command')
      ax.set_title(scenario + ': ' + ['vx (m/s)','vy (m/s)','wz (rad/s)'][axis]); ax.grid(alpha=.25)
  axes[0,0].legend(); fig.tight_layout()
  fig.savefig(args.root/'final_checkpoint_velocity_traces.png', dpi=160); plt.close(fig)
  (args.root/'logger_audit.json').write_text(json.dumps(dict(tags=tags, required_reward_tags_present=True, finite=all(np.isfinite(e.value) for tag in tags for e in events.Scalars(tag))), indent=2)+'\n')
  print(json.dumps(output, indent=2))


if __name__ == '__main__':
  main()
