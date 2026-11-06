"""Read-only, full-rate terminal-state traces under the 40-env benchmark protocol."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
from supermimic.eval.benchmark import load_policy_from_checkpoint

COLUMNS = [
    "time_s", "active", "ref_pelvis_z_m", "robot_pelvis_z_m", "height_error_m",
    "ref_gravity_z", "robot_gravity_z", "tilt_error", "ref_vz_m_s", "robot_vz_m_s",
    "ref_vxy_m_s", "robot_vxy_m_s", "root_vel_error_m_s", "joint_pos_rmse_rad",
    "joint_vel_rmse_rad_s", "left_foot_z_m", "right_foot_z_m", "left_foot_vxy_m_s",
    "right_foot_vxy_m_s", "left_foot_contact", "right_foot_contact",
]


def main(ckpt: Path, manifest: Path, npz_dir: Path, out: Path, device: str = "cuda:0") -> None:
    if out.exists():
        raise FileExistsError(out)
    names = [x for x in manifest.read_text().splitlines() if x]
    files = [str(p) for p in discover_npz(npz_dir, names=names)]
    if [Path(p).stem for p in files] != names or len(names) != 40:
        raise ValueError("manifest and dataset must match in the fixed 40-clip order")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=40, future_frames=cfg["env"]["future_frames"],
        episode_length_s=1e5, action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False,
        seed=0, anchor_obs=cfg["env"]["anchor_obs"],
    ), device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        foot_ids = [cmd.cfg.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        robot = env.scene["robot"]
        local_foot_geom, _ = robot.find_geoms(FOOT_GEOM_REGEX)
        foot_geom = robot.indexing.geom_ids[local_foot_geom]
        contact = env.sim.data.contact
        contact_geom = contact.geom.reshape(-1, 2)
        contact_dist = contact.dist.reshape(-1)
        contact_world = contact.worldid.reshape(-1)
        contact_slot = torch.arange(contact_geom.shape[0], device=device)
        contact_nacon = env.sim.data.nacon.reshape(-1)
        if contact_nacon.numel() != 1:
            raise RuntimeError("unexpected MuJoCo-Warp contact layout")
        foot_geom_by_side = [foot_geom[:7], foot_geom[7:]]
        ground_geom = torch.as_tensor(
            np.flatnonzero(np.asarray(env.sim.mj_model.geom_type) == mujoco.mjtGeom.mjGEOM_PLANE),
            device=device,
        )
        if ground_geom.numel() == 0:
            raise RuntimeError("ground plane geometry is missing; contact status would be invalid")
        print("contact tensors", contact_geom.shape, contact_dist.shape, contact_world.shape,
              "foot geoms", foot_geom.tolist(), "ground geoms", ground_geom.tolist(), flush=True)

        def sample(t: torch.Tensor, active: torch.Tensor) -> np.ndarray:
            g = robot.data.gravity_vec_w
            ref_gz = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
            rob_gz = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
            rz = cmd.root_pos_w[:, 2]
            az = cmd.robot_root_pos_w[:, 2]
            rv = cmd._ref.root_lin_vel_w
            av = robot.data.body_link_lin_vel_w[:, cmd.robot_root_body_index]
            foot_p = cmd.robot_body_pos_w[:, foot_ids] - env.scene.env_origins[:, None, :]
            foot_v = cmd.robot_body_lin_vel_w[:, foot_ids]

            # MuJoCo-Warp contact slots expose candidate geom pairs. Penetrating foot
            # contacts are an observable proxy, not a measured contact force or slip.
            contact_flags = []
            valid_contact = contact_slot < contact_nacon[0]
            for side_geoms in foot_geom_by_side:
                pair = (
                    (torch.isin(contact_geom[:, 0], side_geoms) & torch.isin(contact_geom[:, 1], ground_geom))
                    | (torch.isin(contact_geom[:, 1], side_geoms) & torch.isin(contact_geom[:, 0], ground_geom))
                )
                selected = valid_contact & (contact_dist <= 0) & pair & (contact_world >= 0)
                counts = torch.bincount(contact_world[selected].long(), minlength=40)
                contact_flags.append((counts[:40] > 0).float())
            fields = [
                t, active.float(), rz, az, rz - az, ref_gz, rob_gz, ref_gz - rob_gz,
                rv[:, 2], av[:, 2], rv[:, :2].norm(dim=-1), av[:, :2].norm(dim=-1),
                (rv - av).norm(dim=-1),
                (cmd.joint_pos - cmd.robot_joint_pos).square().mean(-1).sqrt(),
                (cmd.joint_vel - cmd.robot_joint_vel).square().mean(-1).sqrt(),
                foot_p[:, 0, 2], foot_p[:, 1, 2],
                foot_v[:, 0, :2].norm(dim=-1), foot_v[:, 1, :2].norm(dim=-1),
                *contact_flags,
            ]
            return torch.stack(fields, dim=-1).cpu().numpy()

        cmd.forced_frame[:] = cmd.library.clip_start
        obs, _ = env.reset()
        cmd.forced_frame[:] = cmd.library.clip_start[-1]
        active = torch.ones(40, device=device, dtype=torch.bool)
        steps = torch.zeros(40, device=device, dtype=torch.long)
        rows = []
        terminal: dict[int, np.ndarray] = {}
        reasons: dict[int, list[str]] = {}
        original_reset = env._reset_idx
        armed = True

        def record_before_reset(env_ids: torch.Tensor) -> None:
            if armed:
                capture_ids = env_ids[active[env_ids]]
                if capture_ids.numel():
                    snapshot = sample((steps + 1) * env.step_dt, active)
                    for i in capture_ids.tolist():
                        terminal[i] = snapshot[i].copy()
                        reasons[i] = [name for name in ("root_height", "root_tilt")
                                      if bool(env.termination_manager.get_term(name)[i])]
            original_reset(env_ids)

        env._reset_idx = record_before_reset
        with torch.no_grad():
            while bool(active.any()):
                rows.append(sample(steps * env.step_dt, active))
                action = policy(obs["policy"], obs["goal"])
                obs, _, term, tout, _ = env.step(action)
                done = (term | tout) & active
                steps += active.long()
                active &= ~done
                if len(rows) % 2000 == 0:
                    print("step", len(rows), "active", int(active.sum()), flush=True)
        armed = False
        values = np.stack(rows)
        events = []
        for i, name in enumerate(names):
            if i not in terminal:
                raise RuntimeError(f"missing terminal sample: {name}")
            row = terminal[i]
            events.append({
                "name": name, "steps": int(steps[i]), "frames": cmd.library.clips[i].length,
                "failed": bool(reasons[i]), "reasons": reasons[i], "time_s": float(row[0]),
                "ref_z_m": float(row[2]), "robot_z_m": float(row[3]),
                "height_error_m": float(row[4]), "tilt_error": float(row[7]),
                "ref_gravity_z": float(row[5]), "robot_gravity_z": float(row[6]),
            })
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, columns=np.array(COLUMNS), clips=np.array(names), values=values,
                            terminal=np.stack([terminal[i] for i in range(40)]),
                            events_json=json.dumps(events), checkpoint=str(ckpt))
        print("saved", out, "failed", sum(e["failed"] for e in events), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
