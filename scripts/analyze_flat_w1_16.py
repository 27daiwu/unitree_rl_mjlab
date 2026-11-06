"""Read saved W1.13/W1.15 artifacts only; never create an environment."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT13 = Path('logs/flat_w1_13_optimizer_restore_20261104')
ROOT15 = Path('logs/flat_w1_15_fixed_lr_optimizer_20261104_retry01')
OUT = Path('logs/flat_w1_16_multidirectional_attribution_20261104')
STAGES = (50, 100, 200, 300, 399)
COMMANDS = dict(forward_020=(.2, 0, 0), forward_040=(.4, 0, 0),
                backward_015=(-.15, 0, 0), left_020=(0, .2, 0),
                right_020=(0, -.2, 0), turn_left_050=(0, 0, .5),
                turn_right_050=(0, 0, -.5))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = {}
    def track(path):
        sources[str(path)] = digest(path)
        return path
    def read_json(path):
        return json.loads(track(path).read_text())
    aligned, training, configurations, transitions = [], {}, {}, []
    cached = read_json(ROOT15 / 'comparison_final.json')
    for experiment, root in (('w1_13', ROOT13), ('w1_15', ROOT15)):
        for label in ('control', 'experiment'):
            run = root / ('matched_reward_control/control' if experiment == 'w1_13' and label == 'control' else label)
            key = f'{experiment}/{label}'
            for name in ('env', 'agent'):
                p = track(run / 'params' / f'{name}.yaml')
                config = yaml.load(p.read_text(), Loader=yaml.BaseLoader)
                configurations[f'{key}/{name}'] = (dict(commands=config['commands'], curriculum=config['curriculum'], rewards=config['rewards'])
                                                        if name == 'env' else config)
            events = list(run.glob('events.out.tfevents.*'))
            for event in events:
                track(event)
            acc = EventAccumulator(str(run), size_guidance={'scalars': 0})
            acc.Reload()
            tags = acc.Tags()
            scalars = {tag: [dict(step=e.step, value=e.value) for e in acc.Scalars(tag)] for tag in tags['scalars']}
            windows = {}
            for stage in STAGES:
                # Training at iteration N produced model_N; use last 50 updates ending at N.
                windows[str(stage)] = {tag: float(np.mean([e['value'] for e in values if stage-49 <= e['step'] <= stage]))
                                       for tag, values in scalars.items()
                                       if any(stage-49 <= e['step'] <= stage for e in values)}
            training[key] = dict(tags=tags, scalars=scalars, trailing_50_iteration_means=windows)
            ledger = run / 'ppo_update_lr_audit.jsonl'
            if ledger.exists():
                entries = [json.loads(line) for line in track(ledger).read_text().splitlines()]
                assert len(entries) == 400
                assert [e['ppo_update'] for e in entries] == list(range(1, 401))
                assert all(e['learning_rate_before'] == e['learning_rate_after'] == .0002562890625
                           and e['param_group_lrs_before'] == e['param_group_lrs_after'] == [.0002562890625] for e in entries)
                training[key]['lr_audit'] = dict(count=400, violations=0)
            for stage in STAGES:
                checkpoint = track(run / f'model_{stage}.pt')
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                opt = state['optimizer_state_dict']
                optimizer = dict(iteration=state['iter'], entries=len(opt['state']),
                                 steps=sorted({float(s['step']) for s in opt['state'].values()}),
                                 lrs=[g['lr'] for g in opt['param_groups']],
                                 exp_avg_l2=float(sum(s['exp_avg'].double().square().sum().item() for s in opt['state'].values())**.5),
                                 exp_avg_sq_sum=float(sum(s['exp_avg_sq'].double().sum().item() for s in opt['state'].values())))
                assert state['iter'] == stage
                if experiment == 'w1_13':
                    directory = root / (f'w1_14_stage_{stage:03d}_formal' if stage != 399 else 'paired_diagnostics_matched')
                else:
                    directory = root / f'stage_{stage}_actor_diagnostics'
                manifest = read_json(directory / 'manifest.json')
                cfg = manifest['config']
                assert (cfg['seed'], cfg['steps'], cfg['num_envs'], cfg['device']) == (42, 600, 4, 'cpu')
                metrics = read_json(directory / 'metrics.json')
                for scenario, command in COMMANDS.items():
                    group = metrics['comparison'][scenario][label]
                    axis = int(np.flatnonzero(command)[0])
                    replica_stats = []
                    for replica in range(4):
                        with np.load(track(directory / f'{label}_{scenario}_{replica}.npz')) as z:
                            error = z['velocity'][z['time'] > 7., axis] - command[axis]
                            replica_stats.append(dict(rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()), std=float(error.std()),
                                                      mse=float(np.mean(error**2)), bias_squared=float(error.mean()**2), variance=float(error.var())))
                    row = dict(experiment=experiment, policy=label, checkpoint_index=stage, actual_updates=stage+1, scenario=scenario,
                               **{k: float(np.mean([r[k] for r in replica_stats])) for k in replica_stats[0]},
                               stride_mm={f: 1000*group['feet'][f]['step_length_mean_m'] for f in ('left', 'right')},
                               roll_pitch_rms_rad=group['roll_pitch_rms_rad'], completed=group['completed_trials'],
                               evaluation_reward=group['reward_component_weighted_mean_per_s'], optimizer=optimizer)
                    assert np.isclose(row['mse'], row['bias_squared']+row['variance'], atol=1e-8)
                    if experiment == 'w1_15':
                        assert np.isclose(row['rmse'], cached['stages'][str(stage)][scenario]['policies'][label]['rmse_m_s_or_rad_s'])
                    aligned.append(row)
            for first, last in zip(STAGES[:-1], STAGES[1:]):
                def get(stage, scenario):
                    return next(r for r in aligned if (r['experiment'], r['policy'], r['checkpoint_index'], r['scenario']) == (experiment, label, stage, scenario))
                delta = {s: 1000*(get(last,s)['rmse']-get(first,s)['rmse']) for s in COMMANDS}
                transitions.append(dict(experiment=experiment, policy=label, start=first, end=last, rmse_delta_x1000=delta))
    # All input files, including models, logs and trajectories, remain byte-identical.
    assert all(digest(Path(p)) == original for p, original in sources.items())
    OUT.mkdir()
    result = dict(stages=STAGES, aligned=aligned, transitions=transitions, training=training, configurations=configurations,
                  integrity=dict(files_checked=len(sources), unchanged=True, sha256=sources),
                  aggregation='Four per-replica means; MSE components averaged separately; t>7s. Turn errors are rad/s.',
                  training_direction_histograms_available=False)
    (OUT/'aligned_evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (OUT/'direction_metrics.csv').open('w', newline='') as f:
        fields=['experiment','policy','checkpoint_index','actual_updates','scenario','rmse','bias','std','mse','bias_squared','variance','stride_left_mm','stride_right_mm','roll_rms_rad','pitch_rms_rad','track_linear_evaluation','track_angular_evaluation']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in aligned:
            w.writerow({**{k:r[k] for k in fields if k in r},'stride_left_mm':r['stride_mm']['left'],'stride_right_mm':r['stride_mm']['right'],
                        'roll_rms_rad':r['roll_pitch_rms_rad'][0],'pitch_rms_rad':r['roll_pitch_rms_rad'][1],
                        'track_linear_evaluation':r['evaluation_reward']['track_linear_velocity'], 'track_angular_evaluation':r['evaluation_reward']['track_angular_velocity']})
    print(json.dumps(dict(output=str(OUT),rows=len(aligned),files_verified=len(sources))))


if __name__ == '__main__':
    main()
