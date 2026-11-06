"""Read-only 40-env playback with solver contact/actuator telemetry at 200 Hz."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
from supermimic.eval.benchmark import load_policy_from_checkpoint

SELECTED = ["fight1_subject2", "fightAndSports1_subject1", "fightAndSports1_subject4",
            "fallAndGetUp2_subject2", "fallAndGetUp2_subject3", "walk1_subject2", "run1_subject5"]
CONTACT_COLUMNS = ["left_fn_N", "right_fn_N", "other_ground_fn_N", "left_vt_weighted_m_s",
                   "right_vt_weighted_m_s", "left_slip_force_N", "right_slip_force_N",
                   "left_ft_N", "right_ft_N"]


@torch.no_grad()
def main(ckpt: Path, out: Path, max_steps: int = 0) -> None:
    if out.exists():
        raise FileExistsError(out)
    names = Path("reports/phase2a_baseline_20261030/clips40.txt").read_text().splitlines()
    policy, cfg = load_policy_from_checkpoint(ckpt, "cuda:0")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[f"data/npz/lafan1/{n}.npz" for n in names], num_envs=40,
        future_frames=cfg["env"]["future_frames"], episode_length_s=1e5,
        action_noise_std=0, obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False, seed=0, anchor_obs=True))
    try:
        cmd = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        sim, model = env.sim, env.sim.mj_model
        assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        ids = torch.tensor([names.index(n) for n in SELECTED], device=env.device)
        io = StateIO(env)
        ctx = SimContext(env, FOOT_GEOM_REGEX)
        ctrl_ids = robot.indexing.ctrl_ids.cpu().numpy()
        # Map actual actuator order to joint/action order, never assume they coincide.
        joint_ids = robot.indexing.joint_ids.cpu().numpy()
        actuator_order = [int(np.flatnonzero(model.actuator_trnid[:, 0] == j)[0]) for j in joint_ids]
        ai = torch.tensor(actuator_order, device=env.device)
        force_range = np.array(model.actuator_forcerange)[actuator_order]
        gear = np.array(model.actuator_gear)[actuator_order, 0]
        assert np.allclose(gear, 1) and np.all(model.actuator_forcelimited[actuator_order])
        assert action_term.target_names == list(robot.joint_names)
        local, foot_names = robot.find_geoms(FOOT_GEOM_REGEX)
        geomids = robot.indexing.geom_ids[local]
        feet = [geomids[[i for i, n in enumerate(foot_names) if n.startswith(side)]]
                for side in ("left", "right")]
        ground = torch.tensor(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE), device=env.device)
        geom_body = torch.tensor(model.geom_bodyid, device=env.device, dtype=torch.long)
        body_root = torch.tensor(model.body_rootid, device=env.device, dtype=torch.long)
        slots = torch.arange(sim.data.contact.geom.shape[0], device=env.device)
        traces = {k: [] for k in ("state", "next_state", "ref", "action", "target", "active",
                                  "contact", "torque", "torque_unclipped", "joint_substep")}
        errors = {"power_abs_max": 0., "point_velocity_abs_max": 0., "pd_clip_abs_max": 0.}
        cmd.forced_frame[:] = cmd.library.clip_start
        obs, _ = env.reset()
        cmd.forced_frame[:] = cmd.library.clip_start[-1]
        context = ctx.read()[ids].cpu().numpy()
        active = torch.ones(40, device=env.device, dtype=torch.bool)
        events = {}
        original_step = sim.step
        sub_contact, sub_torque, sub_demand, sub_joint = [], [], [], []

        def capture_contact():
            c = sim.data.contact
            world = c.worldid[:].long().clamp(0, 39)
            adr = c.efc_address[:].long()
            valid = (slots < sim.data.nacon[0]) & (adr[:, 0] >= 0)
            valid &= torch.isin(c.geom[:, 0], ground) | torch.isin(c.geom[:, 1], ground)
            ci = valid.nonzero().flatten()
            w, a = world[ci], adr[ci, :4]
            dim = c.dim[ci]
            if not bool((dim == 3).all()):
                raise RuntimeError("expected condim=3 ground contacts")
            f = sim.data.efc.force[w[:, None], a]
            v = sim.data.efc.Jqvel[w[:, None], a]
            mu = c.friction[ci, :2]
            fn = f.sum(-1)
            ft = (f[:, 0::2] - f[:, 1::2]) * mu
            vt = (v[:, 0::2] - v[:, 1::2]) / (2 * mu)
            vn = (v[:, 0] + v[:, 1]) / 2
            power_error = ((f * v).sum(-1) - fn * vn - (ft * vt).sum(-1)).abs()
            # Independent rigid-body point velocity using solver kinematics.
            pair = c.geom[ci].long()
            b = geom_body[pair]
            cv = sim.data.cvel[w[:, None], b]
            center = sim.data.subtree_com[w[:, None], body_root[b]]
            point_v = cv[:, :, 3:] + torch.cross(cv[:, :, :3], c.pos[ci, None, :] - center, dim=-1)
            relative = point_v[:, 1] - point_v[:, 0]
            tangent = torch.einsum("nij,nj->ni", c.frame[ci, 1:3, :], relative)
            if ci.numel():
                errors["power_abs_max"] = max(errors["power_abs_max"], float(power_error.max()))
                errors["point_velocity_abs_max"] = max(errors["point_velocity_abs_max"], float((vt - tangent).abs().max()))
            speed = vt.norm(dim=-1)
            output = torch.zeros(40, len(CONTACT_COLUMNS), device=env.device)
            isfoot = torch.zeros(ci.numel(), dtype=torch.bool, device=env.device)
            for side, geoms in enumerate(feet):
                mask = torch.isin(pair[:, 0], geoms) | torch.isin(pair[:, 1], geoms)
                isfoot |= mask
                index = w[mask]
                output[:, side].index_add_(0, index, fn[mask])
                output[:, 3 + side].index_add_(0, index, fn[mask] * speed[mask])
                output[:, 5 + side].index_add_(0, index, fn[mask] * (speed[mask] > .1))
                output[:, 7 + side].index_add_(0, index, ft[mask].norm(dim=-1))
            output[:, 2].index_add_(0, w[~isfoot], fn[~isfoot])
            output[:, 3:5] /= output[:, :2].clamp_min(1e-8)
            return output[ids].cpu().numpy()

        def measured_step():
            original_step()
            sub_contact.append(capture_contact())
            tau = sim.data.actuator_force[:, ai]
            gain = torch.tensor(model.actuator_gainprm[actuator_order, 0], device=env.device)
            bias = torch.tensor(model.actuator_biasprm[actuator_order, :3], device=env.device)
            demand = sim.data.ctrl[:, ai] * gain + bias[:, 0] + (
                sim.data.actuator_length[:, ai] * bias[:, 1] + sim.data.actuator_velocity[:, ai] * bias[:, 2])
            limits = torch.tensor(force_range, device=env.device)
            err = (tau - demand.clamp(limits[:, 0], limits[:, 1])).abs().max()
            errors["pd_clip_abs_max"] = max(errors["pd_clip_abs_max"], float(err))
            sub_torque.append(tau[ids].cpu().numpy())
            sub_demand.append(demand[ids].cpu().numpy())
            sub_joint.append(torch.stack([cmd.robot_joint_pos[ids], cmd.robot_joint_vel[ids]], -1).cpu().numpy())

        sim.step = measured_step
        last_state = None
        original_reset = env._reset_idx

        def before_reset(reset_ids):
            nonlocal last_state
            last_state = io.read() if last_state is None else last_state
            original_reset(reset_ids)

        env._reset_idx = before_reset
        for step in range(max_steps or int(cmd.library.clip_len.max())):
            if not bool(active[ids].any()):
                break
            sub_contact.clear(); sub_torque.clear(); sub_demand.clear(); sub_joint.clear()
            traces["state"].append(io.read()[ids].cpu().numpy())
            traces["ref"].append(cmd.library.ref71(cmd.frame)[ids].cpu().numpy())
            traces["active"].append(active[ids].cpu().numpy())
            action = policy(obs["policy"], obs["goal"])
            traces["action"].append(action[ids].cpu().numpy())
            traces["target"].append((action * action_term.scale + action_term.offset)[ids].cpu().numpy())
            last_state = None
            obs, _, term, tout, _ = env.step(action)
            nxt = last_state if last_state is not None else io.read()
            traces["next_state"].append(nxt[ids].cpu().numpy())
            for k, samples in (("contact", sub_contact), ("torque", sub_torque),
                               ("torque_unclipped", sub_demand), ("joint_substep", sub_joint)):
                traces[k].append(np.stack(samples))
            for i in ((term | tout) & active).nonzero().flatten().tolist():
                events[names[i]] = {"step": step + 1, "time_s": (step + 1) * .02,
                                   "failed": bool(term[i]), "reasons": [n for n in ("root_height", "root_tilt")
                                   if bool(env.termination_manager.get_term(n)[i])]}
            active &= ~(term | tout)
            if (step + 1) % 2000 == 0:
                print("step", step + 1, "selected active", int(active[ids].sum()), "validation", errors, flush=True)
        meta = {"checkpoint": str(ckpt), "events": events, "checks": errors,
                "dt": .02, "substep_dt": .005, "action_clip": action_term.cfg.clip,
                "weight_N": float(model.body_mass.sum() * 9.81), "actuator_order": actuator_order,
                "ctrl_ids": ctrl_ids.tolist(), "joint_names": list(robot.joint_names)}
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{k: np.stack(v) for k, v in traces.items()},
                            clips=np.array(SELECTED), contact_columns=np.array(CONTACT_COLUMNS),
                            force_range=force_range, soft_limits=robot.data.soft_joint_pos_limits[0].cpu().numpy(),
                            xi=context, meta_json=json.dumps(meta))
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print("saved", out, errors, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
