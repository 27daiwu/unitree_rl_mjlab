"""Drift diagnostic: how fast does a policy's global root pose drift from the reference over whole clips,
and how much of the world-frame velocity error is just that drift (yaw misalignment)?

  uv run python scripts/diag_drift.py --ckpt logs/e7/ckpt_0020000.pt --npz-dir data/npz/lafan1

Same deterministic protocol as eval_benchmark (first frame start, no noise/DR). Each clip plays once in
its own env. Reports per-clip and aggregate:
  * mean / final anchor xy error [m] and |yaw error| [deg], time until xy error > 0.3 m and |yaw| > 15 deg
  * body linear-velocity error: world frame (what the reward sees) vs after rotating the robot's velocities
    by the yaw error (drift removed) -> the residual is the genuine velocity-tracking error.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude, wrap_to_pi

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.rewards import reanchor_reference
from supermimic.eval.benchmark import load_policy_from_checkpoint


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@torch.no_grad()
def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    device: str = "cuda:0",
    out: Path | None = None,
) -> None:
    """Save summary JSON and a full-rate .npz sidecar (pre-step, excludes reset states).

    XY/yaw fields inherited from upstream describe the torso anchor. Additional
    pelvis and yaw/XY-aligned body errors distinguish drift from local pose error.
    Velocity reward fields are exp(-mean(error)), NOT mean(exp(-error)) benchmark
    rewards. Foot link kinematics are saved, but are NOT contact/slip measurements.
    """
    out = out or Path("reports") / f"drift_{ckpt.parent.name}_{ckpt.stem}.json"
    trace_path = out.with_suffix(".npz")
    if out.exists() or trace_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    if not files:
        raise ValueError("no motion clips selected")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    n = len(files)
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=n,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        cmd.forced_frame[:] = lib.clip_start[torch.arange(n, device=device)]
        obs, _ = env.reset()
        cmd.forced_frame[:] = -1
        dt = env.step_dt
        active = torch.ones(n, dtype=torch.bool, device=device)
        steps = torch.zeros(n, device=device)
        sum_xy = torch.zeros(n, device=device)
        sum_yaw = torch.zeros(n, device=device)
        last_xy = torch.zeros(n, device=device)
        last_yaw = torch.zeros(n, device=device)
        t_xy = torch.full((n,), -1.0, device=device)
        t_yaw = torch.full((n,), -1.0, device=device)
        sum_v_world = torch.zeros(n, device=device)
        sum_v_aligned = torch.zeros(n, device=device)
        sum_w_world = torch.zeros(n, device=device)
        sum_w_aligned = torch.zeros(n, device=device)
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        traces = []
        foot_ids = [
            cmd.cfg.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")
        ]
        columns = [
            "time_s",
            "active",
            "anchor_xy_m",
            "anchor_yaw_signed_deg",
            "pelvis_xy_m",
            "pelvis_yaw_signed_deg",
            "body_pos_local_rmse_m",
            "body_ori_local_rmse_deg",
            "lin_vel_world_mse",
            "lin_vel_aligned_mse",
            "ang_vel_world_mse",
            "ang_vel_aligned_mse",
        ]
        for side in ("left", "right"):
            columns += [
                f"{side}_foot_{k}" for k in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
            ]
        columns += ["joint_pos_rmse_rad", "joint_vel_rmse_rad_s"]
        while active.any():
            # measure before stepping: reference target frame vs current robot state
            xy = (cmd.anchor_pos_w[:, :2] - cmd.robot_anchor_pos_w[:, :2]).norm(dim=-1)
            yaw_err = wrap_to_pi(_yaw(cmd.anchor_quat_w) - _yaw(cmd.robot_anchor_quat_w))
            # rotate robot body velocities into the reference heading (remove yaw drift)
            # Pure difference of headings: unlike yaw(q_ref * inverse(q_robot)),
            # this does not mix roll/pitch mismatch into the yaw correction.
            q_fix = torch.zeros(n, 4, device=device)
            q_fix[:, 0], q_fix[:, 3] = torch.cos(yaw_err / 2), torch.sin(yaw_err / 2)
            nb = cmd.body_lin_vel_w.shape[1]
            qf = q_fix[:, None, :].expand(n, nb, 4).reshape(-1, 4)
            v_rob = cmd.robot_body_lin_vel_w.reshape(-1, 3)
            w_rob = cmd.robot_body_ang_vel_w.reshape(-1, 3)
            v_al = quat_apply(qf, v_rob).reshape(n, nb, 3)
            w_al = quat_apply(qf, w_rob).reshape(n, nb, 3)
            ev_w = ((cmd.body_lin_vel_w - cmd.robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
            ev_a = ((cmd.body_lin_vel_w - v_al) ** 2).sum(-1).mean(-1)
            ew_w = ((cmd.body_ang_vel_w - cmd.robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)
            ew_a = ((cmd.body_ang_vel_w - w_al) ** 2).sum(-1).mean(-1)
            local_pos, local_quat = reanchor_reference(
                cmd.body_pos_w,
                cmd.body_quat_w,
                cmd.anchor_pos_w,
                cmd.anchor_quat_w,
                cmd.robot_anchor_pos_w,
                cmd.robot_anchor_quat_w,
            )
            pelvis_xy = (cmd.root_pos_w[:, :2] - cmd.robot_root_pos_w[:, :2]).norm(dim=-1)
            pelvis_yaw = wrap_to_pi(_yaw(cmd.root_quat_w) - _yaw(cmd.robot_root_quat_w))
            local_pos_rmse = ((local_pos - cmd.robot_body_pos_w) ** 2).sum(-1).mean(-1).sqrt()
            local_ori_rmse = (
                quat_error_magnitude(local_quat, cmd.robot_body_quat_w).square().mean(-1).sqrt()
            )
            row = torch.stack(
                [
                    steps * dt,
                    active.float(),
                    xy,
                    yaw_err.rad2deg(),
                    pelvis_xy,
                    pelvis_yaw.rad2deg(),
                    local_pos_rmse,
                    local_ori_rmse.rad2deg(),
                    ev_w,
                    ev_a,
                    ew_w,
                    ew_a,
                ],
                dim=-1,
            )
            feet = torch.cat(
                [
                    cmd.robot_body_pos_w[:, foot_ids] - env.scene.env_origins[:, None, :],
                    cmd.robot_body_lin_vel_w[:, foot_ids],
                ],
                dim=-1,
            ).flatten(1)
            joint_errors = torch.stack(
                [
                    (cmd.robot_joint_pos - cmd.joint_pos).square().mean(-1).sqrt(),
                    (cmd.robot_joint_vel - cmd.joint_vel).square().mean(-1).sqrt(),
                ],
                dim=-1,
            )
            traces.append(torch.cat([row, feet, joint_errors], dim=-1).cpu().numpy())
            a = active.float()
            sum_xy += xy * a
            sum_yaw += yaw_err.abs() * a
            last_xy = torch.where(active, xy, last_xy)
            last_yaw = torch.where(active, yaw_err.abs(), last_yaw)
            t_xy = torch.where(active & (t_xy < 0) & (xy > 0.3), steps * dt, t_xy)
            t_yaw = torch.where(
                active & (t_yaw < 0) & (yaw_err.abs() > math.radians(15)), steps * dt, t_yaw
            )
            sum_v_world += ev_w * a
            sum_v_aligned += ev_a * a
            sum_w_world += ew_w * a
            sum_w_aligned += ew_a * a
            steps += a
            act = policy(obs["policy"], obs["goal"])
            obs, _, term, tout, _ = env.step(act)
            done = (term | tout) & active
            failed |= term & active
            active &= ~done
            if len(traces) % 2000 == 0:
                print(f"diagnostic step {len(traces)} | active {int(active.sum())}/{n}", flush=True)
        st = steps.clamp(min=1)
        trace = np.stack(traces)
        rows = []
        for i in range(n):
            series = trace[: int(steps[i]), i]
            rows.append(
                {
                    "name": lib.clips[i].name,
                    "steps": int(steps[i]),
                    "failed": bool(failed[i]),
                    "mean_xy": float(sum_xy[i] / st[i]),
                    "final_xy": float(last_xy[i]),
                    "mean_yaw_deg": math.degrees(float(sum_yaw[i] / st[i])),
                    "final_yaw_deg": math.degrees(float(last_yaw[i])),
                    "t_xy_gt_0p3": float(t_xy[i]),
                    "t_yaw_gt_15": float(t_yaw[i]),
                    "r_lin_world": float(torch.exp(-sum_v_world[i] / st[i] / 1.0)),
                    "r_lin_aligned": float(torch.exp(-sum_v_aligned[i] / st[i] / 1.0)),
                    "r_ang_world": float(torch.exp(-sum_w_world[i] / st[i] / 3.14**2)),
                    "r_ang_aligned": float(torch.exp(-sum_w_aligned[i] / st[i] / 3.14**2)),
                    "duration_s": float(steps[i] * dt),
                    "last_sample_s": float(series[-1, 0]),
                    "pelvis_mean_xy_m": float(series[:, 4].mean()),
                    "pelvis_final_xy_m": float(series[-1, 4]),
                    "pelvis_mean_abs_yaw_deg": float(np.abs(series[:, 5]).mean()),
                    "pelvis_final_abs_yaw_deg": float(abs(series[-1, 5])),
                    "body_pos_local_rmse_m": float(np.sqrt(np.mean(series[:, 6] ** 2))),
                    "body_ori_local_rmse_deg": float(np.sqrt(np.mean(series[:, 7] ** 2))),
                    "lin_vel_world_rmse_m_s": float(np.sqrt(series[:, 8].mean())),
                    "lin_vel_aligned_rmse_m_s": float(np.sqrt(series[:, 9].mean())),
                    "joint_pos_rmse_rad": float(np.sqrt(np.mean(series[:, 24] ** 2))),
                    "joint_vel_rmse_rad_s": float(np.sqrt(np.mean(series[:, 25] ** 2))),
                }
            )
        tot = steps.sum()
        agg = {
            "clips": n,
            "failed": int(failed.sum()),
            "mean_xy": float((sum_xy).sum() / tot),
            "mean_yaw_deg": math.degrees(float(sum_yaw.sum() / tot)),
            "median_t_xy_gt_0p3": float(torch.median(t_xy[t_xy >= 0])) if (t_xy >= 0).any() else -1,
            "median_t_yaw_gt_15": float(torch.median(t_yaw[t_yaw >= 0]))
            if (t_yaw >= 0).any()
            else -1,
            "lin_vel_reward_world": float(torch.exp(-sum_v_world.sum() / tot)),
            "lin_vel_reward_yaw_aligned": float(torch.exp(-sum_v_aligned.sum() / tot)),
            "ang_vel_reward_world": float(torch.exp(-sum_w_world.sum() / tot / 3.14**2)),
            "ang_vel_reward_yaw_aligned": float(torch.exp(-sum_w_aligned.sum() / tot / 3.14**2)),
        }
        print(json.dumps(agg, indent=1))
        print(
            f"{'clip':30s} {'steps':>6s} {'fail':>4s} {'xy_mean':>7s} {'xy_end':>6s} {'yaw_mean':>8s} {'yaw_end':>7s} {'t>0.3m':>7s} {'t>15deg':>7s} {'rlin_w':>6s} {'rlin_a':>6s}"
        )
        for r in rows:
            print(
                f"{r['name']:30s} {r['steps']:6d} {str(r['failed'])[0]:>4s} {r['mean_xy']:7.2f} {r['final_xy']:6.2f} {r['mean_yaw_deg']:8.1f} {r['final_yaw_deg']:7.1f} {r['t_xy_gt_0p3']:7.1f} {r['t_yaw_gt_15']:7.1f} {r['r_lin_world']:6.2f} {r['r_lin_aligned']:6.2f}"
            )
        np.savez_compressed(
            trace_path,
            values=trace,
            columns=np.array(columns),
            clips=np.array([c.name for c in lib.clips]),
            dt=np.array(dt),
        )
        Path(out).write_text(
            json.dumps(
                {
                    "protocol": {
                        "checkpoint": str(ckpt.resolve()),
                        "anchor_obs": cfg["env"].get("anchor_obs", False),
                        "npz_files": files,
                        "dt": dt,
                        "seed": 0,
                        "sample_phase": "pre_step",
                        "terminal_state_included": False,
                        "anchor_body": "torso_link",
                        "root_body": "pelvis",
                        "velocity_alignment": "difference_of_heading_yaws",
                        "velocity_reward_aggregation": "exp(-mean(squared_error)/std^2), not benchmark mean reward",
                        "foot_contact_measured": False,
                        "trace": str(trace_path),
                    },
                    "summary": agg,
                    "clips": rows,
                },
                indent=1,
            )
        )
        print("saved", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
