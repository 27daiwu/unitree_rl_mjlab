"""Read-only cross-state action, depth sensitivity and runtime audit for S1-D1.

No optimizer, checkpoint save, persistent task configuration edits or training.
Action alternatives are evaluated offline; only the named driver's real-depth
deterministic actions reach the simulator. Terminal snapshots precede auto reset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from src.tasks.velocity.mdp.stairs_metrics import geometry_tensors
import train_g1_visual_stairs_v1a_s0 as s0

OUT = Path('logs/experiments/g1_visual_stairs_v1a_s1_d1_20261104')
S1_DIR = Path('logs/experiments/g1_visual_stairs_v1a_s1_20261104')
CHECKPOINTS = {
    'flat': s0.FLAT_SOURCE,
    's01': Path('logs/experiments/g1_visual_stairs_v1a_s01_20261104/v1a_s0_1_bc_model.pt'),
    's1': S1_DIR / 'v1a_s1_bc_model.pt',
    's1a': S1_DIR / 'v1a_s1a_dagger_bc_model.pt',
}
KEY_JOINTS = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
              'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_hip_roll_joint',
              'right_hip_roll_joint', 'left_knee_joint', 'right_knee_joint')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest_state(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(driver, seed, device):
    env, runner = s0.make_env(seed, 16, device)
    try:
        vec = runner.env
        models = {}
        modes = {}
        # All models see this same wrapper: no extra reset or env-state restore.
        for label, checkpoint in CHECKPOINTS.items():
            task = s0.FLAT if label == 'flat' else s0.TASK
            other = load_runner_cls(task)(vec, asdict(load_rl_cfg(task)), device=device)
            model = other.alg.get_policy()
            sd = torch.load(checkpoint, map_location=device, weights_only=False)['actor_state_dict']
            model.load_state_dict(sd, strict=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            models[label] = model
            modes[label] = {'class': type(model).__name__, 'training': model.training,
                            'normalizer_training': model.obs_normalizer.training,
                            'dropout_batchnorm': [type(m).__name__ for m in model.modules()
                              if isinstance(m, (torch.nn.modules.dropout._DropoutNd, torch.nn.modules.batchnorm._BatchNorm))],
                            'state_before': digest_state(model)}
        robot = env.scene['robot']
        feet = env.scene['feet_ground_contact']
        term = env.action_manager.get_term('joint_pos')
        joint_ids = term.target_ids
        names = list(term.target_names)
        scale = term.scale
        offset = term.offset
        bias = robot.data.encoder_bias[:, joint_ids]
        foot_ids, _ = robot.find_sites(('left_foot', 'right_foot'))
        geometry = geometry_tensors(env)
        riser_x = (env.scene.env_origins[:, 0] + geometry['staircase_start_x'] - geometry['spawn_x'])
        obs = vec.get_observations()
        s0.set_command(env, obs)
        initial = {key: s0.arr(value) for key, value in {
            'qpos': env.sim.data.qpos, 'qvel': env.sim.data.qvel,
            'origins': env.scene.env_origins, 'actor': obs['actor'],
            'depth': obs['depth'], 'encoder_bias': robot.data.encoder_bias,
            'friction': env.sim.model.geom_friction, 'body_ipos': env.sim.model.body_ipos}.items()}
        np.savez_compressed(OUT / f'initial_{seed}_{driver}.npz', **initial)
        x0 = robot.data.root_link_pos_w[:, 0].clone()
        initial_depth = obs['depth'].clone()
        initial_features = {k: m.cnns['depth'](initial_depth).detach() for k, m in models.items() if k != 'flat'}
        active = np.ones(16, bool)
        state_rows, error_rows, sensitivity_rows, feature_rows = [], [], [], []
        snapshots = {k: [] for k in ('time', 'env_id', 'actor', 'teacher_action',
                                    's01_action', 's1_action', 's1a_action', 'qpos')}
        step = 0
        pending = {}
        max_target_error = 0.0
        original_metrics = env.metrics_manager.compute

        def capture():
            nonlocal max_target_error
            original_metrics()
            q = s0.arr(robot.data.root_link_quat_w)
            w, x, y, z = q.T
            pitch = np.degrees(np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)))
            roll = np.degrees(np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)))
            tilt = s0.arr(torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))))
            vx = s0.arr(robot.data.root_link_lin_vel_w[:, 0])
            dx = s0.arr(robot.data.root_link_pos_w[:, 0] - x0)
            distance = s0.arr(riser_x - robot.data.site_pos_w[:, foot_ids, 0].max(dim=1).values)
            contacts = s0.arr(feet.data.found[:, :2])
            force = s0.arr(feet.data.force[:, :2].norm(dim=-1))
            targets = s0.arr(robot.data.joint_pos_target[:, joint_ids])
            positions = s0.arr(robot.data.joint_pos[:, joint_ids])
            actual_raw = s0.arr(term.raw_action)
            done = s0.arr(env.reset_buf).astype(bool)
            fall = s0.arr(env.termination_manager.get_term('fell_over')).astype(bool)
            expected = pending['targets'][driver]
            for i in np.flatnonzero(active):
                mismatch = float(np.max(np.abs(expected[i] - targets[i])))
                max_target_error = max(max_target_error, mismatch)
                row = {'seed': seed, 'driver': driver, 'env_id': int(i), 'step': step,
                       'input_time_s': round((step-1)*env.step_dt, 6),
                       'post_time_s': round(step*env.step_dt, 6),
                       'pitch_deg': float(pitch[i]), 'roll_deg': float(roll[i]), 'tilt_deg': float(tilt[i]),
                       'vx_mps': float(vx[i]), 'dx_m': float(dx[i]),
                       'front_site_riser_distance_m': float(distance[i]),
                       'left_contact': int(contacts[i, 0] > 0), 'right_contact': int(contacts[i, 1] > 0),
                       'left_force_n': float(force[i, 0]), 'right_force_n': float(force[i, 1]),
                       'phase_sin_input': float(pending['actor'][i, 9]),
                       'phase_cos_input': float(pending['actor'][i, 10]),
                       'done': int(done[i]), 'fell_over': int(fall[i]),
                       'executed_raw': json.dumps(actual_raw[i].tolist()),
                       'target_rad': json.dumps(targets[i].tolist()), 'joint_pos_rad': json.dumps(positions[i].tolist())}
                state_rows.append(row)
                snapshots['time'].append((step-1)*env.step_dt)
                snapshots['env_id'].append(i)
                snapshots['actor'].append(pending['actor'][i])
                snapshots['qpos'].append(s0.arr(env.sim.data.qpos)[i])
                for label in models:
                    snapshots['teacher_action' if label == 'flat' else f'{label}_action'].append(pending['raw'][label][i])
                for label in ('s01', 's1', 's1a'):
                    raw_error = pending['raw'][label][i] - pending['raw']['flat'][i]
                    target_error = pending['targets'][label][i] - pending['targets']['flat'][i]
                    e = {'seed': seed, 'driver': driver, 'student': label, 'env_id': int(i), 'step': step,
                         'input_time_s': round((step-1)*env.step_dt, 6),
                         'raw_rmse': float(np.sqrt(np.mean(raw_error**2))),
                         'target_rmse_rad': float(np.sqrt(np.mean(target_error**2))),
                         'target_max_abs_rad': float(np.abs(target_error).max()),
                         'error_rad': json.dumps(target_error.tolist()),
                         'teacher_target_rad': json.dumps(pending['targets']['flat'][i].tolist()),
                         'student_target_rad': json.dumps(pending['targets'][label][i].tolist())}
                    error_rows.append(e)
                if done[i]:
                    active[i] = False
        env.metrics_manager.compute = capture
        audit = {'seed': seed, 'driver': driver, 'joint_names': names,
                 'terms': list(env.observation_manager.active_terms['actor']),
                 'dims': [list(v) for v in env.observation_manager.group_obs_term_dim['actor']],
                 'clip_actions': vec.clip_actions, 'action_scale': s0.arr(scale).tolist(),
                 'action_offset': s0.arr(offset).tolist(), 'riser_x': s0.arr(riser_x).tolist(),
                 'counter_at_start': int(env.common_step_counter), 'modes': modes}
        probe = {'actor': obs['actor'].clone(), 'depth': obs['depth'].clone()}
        for label, model in models.items():
            eval_output = model(probe)
            model.train()
            model.obs_normalizer.eval()
            train_output = model(probe)
            model.eval()
            modes[label]['train_eval_output_max_abs'] = float((train_output-eval_output).abs().max())
        for step in range(1, 151):
            s0.set_command(env, obs)
            actor = obs['actor']
            depth = obs['depth']
            outputs = {k: m(obs) for k, m in models.items()}
            for label, model in models.items():
                if label == 'flat':
                    continue
                norm = model.obs_normalizer(actor)
                feature = model.cnns['depth'](depth)
                first_linear = model.mlp[0]
                depth_pre = torch.nn.functional.linear(feature, first_linear.weight[:, 98:], bias=None)
                prop_pre = torch.nn.functional.linear(norm, first_linear.weight[:, :98], bias=None)
                for i in np.flatnonzero(active):
                    feature_rows.append({'seed': seed, 'driver': driver, 'student': label,
                      'env_id': int(i), 'input_time_s': round((step-1)*env.step_dt, 6),
                      'normalizer_abs_max': float(norm[i].abs().max()),
                      'normalizer_rms': float(norm[i].square().mean().sqrt()),
                      'cnn_feature_rms': float(feature[i].square().mean().sqrt()),
                      'cnn_feature_drift_rms': float((feature[i]-initial_features[label][i]).square().mean().sqrt()),
                      'depth_first_linear_rms': float(depth_pre[i].square().mean().sqrt()),
                      'prop_first_linear_rms': float(prop_pre[i].square().mean().sqrt())})
                if (step-1) % 5 == 0:
                    variants = {'initial_depth': initial_depth, 'zero_depth': torch.zeros_like(depth),
                                'constant_2m': torch.full_like(depth, 2),
                                'spatial_flip': depth.flip(-1),
                                'batch_swap': depth.roll(1, 0), 'scale_plus_1pct': depth*1.01}
                    # Also isolate proprioceptive input subsets with their checkpoint means.
                    variants_out = {v: model({'actor': actor, 'depth': d}) for v, d in variants.items()}
                    for group, sl in {'phase_mean': slice(9,11), 'last_action_mean': slice(69,98)}.items():
                        altered = actor.clone()
                        altered[:, sl] = model.obs_normalizer._mean[:, sl]
                        variants_out[group] = model({'actor': altered, 'depth': depth})
                    for variant, alt in variants_out.items():
                        delta = (alt - outputs[label]) * scale
                        err_alt = (alt - outputs['flat']) * scale
                        for i in np.flatnonzero(active):
                            sensitivity_rows.append({'seed': seed, 'driver': driver, 'student': label,
                              'env_id': int(i), 'input_time_s': round((step-1)*env.step_dt, 6), 'variant': variant,
                              'target_delta_rmse_rad': float(delta[i].square().mean().sqrt()),
                              'target_delta_max_rad': float(delta[i].abs().max()),
                              'teacher_target_error_rmse_rad': float(err_alt[i].square().mean().sqrt())})
            pending = {'actor': s0.arr(actor), 'raw': {k: s0.arr(v) for k, v in outputs.items()},
                       'targets': {k: s0.arr(v*scale+offset-bias) for k,v in outputs.items()}}
            obs, _, _, _ = vec.step(outputs[driver])
            if not active.any():
                break
        audit['max_executed_target_mapping_error_rad'] = max_target_error
        for label, model in models.items():
            modes[label]['state_after'] = digest_state(model)
            assert modes[label]['state_after'] == modes[label]['state_before'], 'Model mutated during diagnosis'
        prefix = f'{seed}_{driver}'
        write_csv(OUT / f'states_{prefix}.csv', state_rows)
        write_csv(OUT / f'cross_actions_{prefix}.csv', error_rows)
        write_csv(OUT / f'sensitivity_{prefix}.csv', sensitivity_rows)
        write_csv(OUT / f'features_{prefix}.csv', feature_rows)
        np.savez_compressed(OUT / f'snapshots_{prefix}.npz', **{k: np.asarray(v) for k,v in snapshots.items()})
        (OUT / f'audit_{prefix}.json').write_text(json.dumps(audit, indent=2)+'\n')
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = {k: sha(v) for k,v in CHECKPOINTS.items()}
    with torch.inference_mode():
        for seed in (142, 314):
            for driver in CHECKPOINTS:
                print(f'RUN seed={seed} driver={driver}', flush=True)
                run(driver, seed, args.device)
    after = {k: sha(v) for k,v in CHECKPOINTS.items()}
    assert before == after
    pairs = {}
    for seed in (142,314):
        reference = np.load(OUT / f'initial_{seed}_flat.npz')
        for driver in CHECKPOINTS:
            other = np.load(OUT / f'initial_{seed}_{driver}.npz')
            pairs[f'{seed}_{driver}'] = {k: bool(np.array_equal(reference[k], other[k])) for k in reference.files}
    assert all(all(v.values()) for v in pairs.values()), 'Initial state mismatch'
    (OUT / 'integrity.json').write_text(json.dumps({'checkpoints_before': before, 'checkpoints_after': after,
                                                  'initial_pairs_bitwise_equal': pairs}, indent=2)+'\n')
    print('Read-only cross-state diagnosis complete.', flush=True)


if __name__ == '__main__':
    main()
