"""Offline D3: capture complete pre-failure branches and cross two frozen policies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.restore_validation import validate_branch_slots, validate_snapshot_frame
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
from supermimic.eval.benchmark import load_policy_from_checkpoint

HERE = Path("reports/phase2b_d3_20261030")
D2 = Path("reports/phase2b_d2_20261030")
CKPTS = {
    "38200": Path("logs/phase2a_anchor_4090_s0_20261030/ckpt_latest.pt"),
    "46200": Path("logs/phase2b_anchor_4090_s0_38200_to_46200/ckpt_0046200.pt"),
}
# Before the first D2 warning, with enough room to observe each terminal event.
STARTS = {
    "fight1_subject2": 202.60,
    "fightAndSports1_subject1": 42.90,
    "fightAndSports1_subject4": 28.60,
    "fallAndGetUp2_subject2": 13.40,
    "fallAndGetUp2_subject3": 12.20,
}
SOURCE_STARTS = {"38200": {**STARTS, "fight1_subject2": 78.80}, "46200": STARTS}
DT = .02
HORIZON = 100


def env_args(names):
    return SuperMimicEnvCfgArgs(
        npz_files=[f"data/npz/lafan1/{n}.npz" for n in names], num_envs=40,
        future_frames=10, episode_length_s=1e5, action_noise_std=0,
        obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False, seed=0, anchor_obs=True,
    )


def obs_history(env, i):
    group = env.observation_manager._group_obs_term_history_buffer["policy"]
    return {k: v.buffer[i].cpu().numpy().copy() for k, v in group.items()}


def capture(env, i, obs, step):
    d = env.sim.data
    term = env.action_manager.get_term("joint_pos")
    cmd = env.command_manager.get_term("motion")
    return {
        "step": step, "frame": int(cmd.frame[i]), "origin": env.scene.env_origins[i].cpu().numpy(),
        "qpos": d.qpos[i].cpu().numpy().copy(), "qvel": d.qvel[i].cpu().numpy().copy(),
        "qacc_warmstart": d.qacc_warmstart[i].cpu().numpy().copy(),
        "ctrl": d.ctrl[i].cpu().numpy().copy(), "time": float(d.time[i]),
        "raw_action": term.raw_action[i].cpu().numpy().copy(),
        "executed_action": term.executed_raw[i].cpu().numpy().copy(),
        "target": term._processed_actions[i].cpu().numpy().copy(),
        "manager_action": env.action_manager.action[i].cpu().numpy().copy(),
        "manager_prev_action": env.action_manager.prev_action[i].cpu().numpy().copy(),
        "manager_prev_prev_action": env.action_manager.prev_prev_action[i].cpu().numpy().copy(),
        "history": obs_history(env, i),
        "obs_policy": obs["policy"][i].cpu().numpy().copy(),
        "obs_goal": obs["goal"][i].cpu().numpy().copy(),
    }


def restore(env, snapshots, branches):
    d = env.sim.data
    cmd = env.command_manager.get_term("motion")
    term = env.action_manager.get_term("joint_pos")
    group = env.observation_manager._group_obs_term_history_buffer["policy"]
    n_envs = env.num_envs
    lib = cmd.library
    slots = [int(branch[0]) for branch in branches]
    validate_branch_slots(slots, n_envs)
    for slot, name, *_ in branches:
        if name not in snapshots:
            raise KeyError(f"missing snapshot {name!r} for branch slot {slot}")
        snap = snapshots[name]
        frame = int(snap["frame"])
        expected_clip = name.split("@", 1)[0]
        validate_snapshot_frame(frame, expected_clip, lib)
        expected_shapes = {
            "qpos": tuple(d.qpos.shape[1:]),
            "qvel": tuple(d.qvel.shape[1:]),
            "qacc_warmstart": tuple(d.qacc_warmstart.shape[1:]),
            "ctrl": tuple(d.ctrl.shape[1:]),
            "raw_action": tuple(term.raw_action.shape[1:]),
            "executed_action": tuple(term.executed_raw.shape[1:]),
            "target": tuple(term._processed_actions.shape[1:]),
        }
        for key, shape in expected_shapes.items():
            if tuple(np.shape(snap[key])) != shape:
                raise ValueError(f"snapshot {name} {key} shape {np.shape(snap[key])} != {shape}")
        if set(snap["history"]) != set(group):
            raise ValueError(f"snapshot {name} history terms do not match environment: "
                             f"{set(snap['history'])} != {set(group)}")
        for key, ring in group.items():
            if tuple(np.shape(snap["history"][key])) != (ring.max_length, ring._buffer.shape[-1]):
                raise ValueError(f"snapshot {name} history[{key}] shape "
                                 f"{np.shape(snap['history'][key])} incompatible with "
                                 f"({ring.max_length}, {ring._buffer.shape[-1]})")
    for slot, name, source, policy, repeat in branches:
        snap = snapshots[name]
        qpos = torch.as_tensor(snap["qpos"], device=env.device).clone()
        qpos[:3] += env.scene.env_origins[slot] - torch.as_tensor(snap["origin"], device=env.device)
        d.qpos[slot] = qpos
        for k in ("qvel", "qacc_warmstart", "ctrl"):
            getattr(d, k)[slot] = torch.as_tensor(snap[k], device=env.device)
        d.time[slot] = snap["time"]
        cmd.frame[slot] = snap["frame"]
        cmd.clip_end[slot] = False
        env.episode_length_buf[slot] = snap["step"]
        term._raw_actions[slot] = torch.as_tensor(snap["raw_action"], device=env.device)
        term._executed_raw[slot] = torch.as_tensor(snap["executed_action"], device=env.device)
        term._processed_actions[slot] = torch.as_tensor(snap["target"], device=env.device)
        for field, key in (("_action", "manager_action"), ("_prev_action", "manager_prev_action"),
                           ("_prev_prev_action", "manager_prev_prev_action")):
            lag = {"manager_action": 1, "manager_prev_action": 2,
                   "manager_prev_prev_action": 3}[key]
            value = snap.get(key, snap["history"]["actions"][-lag])
            getattr(env.action_manager, field)[slot] = torch.as_tensor(value, device=env.device)
        for k, hist in snap["history"].items():
            ring = group[k]
            order = (torch.arange(ring.max_length, device=env.device) + ring._pointer + 1) % ring.max_length
            ring._buffer[order, slot] = torch.as_tensor(hist, device=env.device)
            ring._num_pushes[slot] = ring.max_length
    env.sim.forward()
    # forward may refresh solver fields; restore the saved warm start afterwards.
    for slot, name, *_ in branches:
        d.qacc_warmstart[slot] = torch.as_tensor(snapshots[name]["qacc_warmstart"], device=env.device)
    cmd._refresh_ref()
    cmd.update_relative_body_poses()
    env.sim.sense()
    env.observation_manager._obs_buffer = None
    obs = env.observation_manager.compute(update_history=False)
    errors = {}
    for slot, name, *_ in branches:
        snap = snapshots[name]
        errors[slot] = {
            "qpos": float((d.qpos[slot] - torch.as_tensor(snap["qpos"], device=env.device)
                            - torch.cat([env.scene.env_origins[slot] - torch.as_tensor(snap["origin"], device=env.device),
                                         torch.zeros(33, device=env.device)])).abs().max()),
            "qvel": float((d.qvel[slot] - torch.as_tensor(snap["qvel"], device=env.device)).abs().max()),
            "warmstart": float((d.qacc_warmstart[slot] - torch.as_tensor(snap["qacc_warmstart"], device=env.device)).abs().max()),
            "obs_policy": float((obs["policy"][slot] - torch.as_tensor(snap["obs_policy"], device=env.device)).abs().max()),
            "obs_goal": float((obs["goal"][slot] - torch.as_tensor(snap["obs_goal"], device=env.device)).abs().max()),
        }
    return obs, errors


def contact_telemetry(env, slots, pre_qvel):
    import mujoco

    sim, model = env.sim, env.sim.mj_model
    data = sim.data
    c = data.contact
    n = data.nacon[:].cpu().numpy()
    geoms = c.geom[:].cpu().numpy()
    world = c.worldid[:].cpu().numpy()
    adr = c.efc_address[:].cpu().numpy()
    mu = c.friction[:].cpu().numpy()
    forces = data.efc.force[:].cpu().numpy()
    j = data.efc.J[:, 0, :].cpu().numpy()
    col = data.efc.J_colind[:, 0, :].cpu().numpy()
    rowadr = data.efc.J_rowadr[:].cpu().numpy()
    rownnz = data.efc.J_rownnz[:].cpu().numpy()
    pre = data.efc.Jqvel[:].cpu().numpy()
    qvel = data.qvel[:].cpu().numpy()
    result = np.zeros((len(slots), 7), dtype=np.float64)
    validation = 0.0
    plane = set(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
    foot = set()
    robot = env.scene["robot"]
    local, _ = robot.find_geoms(FOOT_GEOM_REGEX)
    foot.update(robot.indexing.geom_ids[local].cpu().numpy().tolist())
    lookup = {s: i for i, s in enumerate(slots)}
    for idx in range(len(world)):
        w = int(world[idx])
        if w not in lookup or idx >= n[0] or adr[idx, 0] < 0:
            continue
        pair = geoms[idx]
        if not (int(pair[0]) in plane or int(pair[1]) in plane):
            continue
        rows = adr[idx, :4]
        if any(int(r) < 0 for r in rows):
            continue
        f = forces[w, rows]
        fn = f.sum()
        if int(pair[0]) not in foot and int(pair[1]) not in foot:
            result[lookup[w], 2] += fn
            continue
        vpost = np.zeros(4)
        for k, r in enumerate(rows):
            a = rowadr[w, r]
            count = rownnz[w, r]
            vpost[k] = np.dot(j[w, a:a+count], qvel[w, col[w, a:a+count]])
        vtpre = (pre[w, rows[::2]] - pre[w, rows[1::2]]) / (2 * mu[idx, :2])
        vtpost = (vpost[::2] - vpost[1::2]) / (2 * mu[idx, :2])
        side = 0 if any(model.geom(int(g)).name.startswith("left_foot") for g in pair) else 1
        r = result[lookup[w]]
        r[side] += fn
        r[3] += fn * np.linalg.norm(vtpost)
        r[4] += fn * np.linalg.norm(vtpre)
        r[5] += fn * (np.linalg.norm(vtpost) > .1)
        r[6] += fn * (np.linalg.norm(vtpre) > .1)
        # The sparse contact Jacobian must reproduce the pre-solver Jqvel rows.
        for k, row in enumerate(rows):
            a, count = rowadr[w, row], rownnz[w, row]
            v = np.dot(j[w, a:a+count], pre_qvel[w, col[w, a:a+count]])
            validation = max(validation, abs(float(v - pre[w, row])))
    return result, validation


@torch.no_grad()
def branch_run(source_label):
    path = HERE / f"cross_{source_label}.npz"
    if path.exists():
        raise FileExistsError(path)
    saved = torch.load(HERE / f"source_{source_label}.pt", map_location="cpu", weights_only=False)
    names = saved["names"]
    policies = {k: load_policy_from_checkpoint(v, "cuda:0")[0] for k, v in CKPTS.items()}
    env = make_env(env_args(names))
    try:
        cmd = env.command_manager.get_term("motion")
        cmd.forced_frame[:] = cmd.library.clip_start
        env.reset()
        cmd.forced_frame[:] = cmd.library.clip_start[-1]
        # Put the native-policy first repeat in its original 40-env slot. This
        # makes restoration fidelity testable without changing GPU world index.
        starts = saved.get("starts", SOURCE_STARTS[source_label])
        assigned = {names.index(n): (n, source_label, 0) for n in starts}
        pending = [(n, p, r) for n in starts for p in CKPTS for r in range(3)
                   if not (p == source_label and r == 0)]
        for slot in range(30):
            if slot not in assigned:
                assigned[slot] = pending.pop(0)
        assert not pending
        branches = [(slot, *assigned[slot][:1], source_label, *assigned[slot][1:])
                    for slot in range(30)]
        slots = [b[0] for b in branches]
        obs, restore_errors = restore(env, saved["snapshots"], branches)
        if max(max(x.values()) for x in restore_errors.values()) > 1e-4:
            raise RuntimeError(f"state restore mismatch: {restore_errors}")
        ids = torch.as_tensor(slots, device=env.device)
        label_slots = {k: torch.as_tensor([b[0] for b in branches if b[3] == k], device=env.device)
                       for k in CKPTS}
        io = StateIO(env)
        force_range = np.asarray(env.sim.mj_model.actuator_forcerange)
        robot = env.scene["robot"]
        joint_ids = robot.indexing.joint_ids.cpu().numpy()
        order = [int(np.flatnonzero(env.sim.mj_model.actuator_trnid[:, 0] == j)[0]) for j in joint_ids]
        limits = np.max(np.abs(force_range[order]), axis=1)
        ai = torch.as_tensor(order, device=env.device)
        traces = {k: [] for k in ("state", "ref", "action", "target", "active", "torque", "demand", "contact")}
        term = env.action_manager.get_term("joint_pos")
        active = np.ones(len(slots), dtype=bool)
        events = {}
        original_step = env.sim.step
        sub_torque, sub_demand, sub_contact = [], [], []
        jac_error = 0.0

        def measured_step():
            nonlocal jac_error
            pre_qvel = env.sim.data.qvel[:].cpu().numpy().copy()
            original_step()
            c, err = contact_telemetry(env, slots, pre_qvel)
            jac_error = max(jac_error, err)
            sub_contact.append(c)
            sub_torque.append(env.sim.data.actuator_force[ids[:, None], ai].cpu().numpy())
            gain = torch.as_tensor(env.sim.mj_model.actuator_gainprm[order, 0], device=env.device)
            bias = torch.as_tensor(env.sim.mj_model.actuator_biasprm[order, :3], device=env.device)
            demand = env.sim.data.ctrl[:, ai] * gain + bias[:, 0] + (
                env.sim.data.actuator_length[:, ai] * bias[:, 1] +
                env.sim.data.actuator_velocity[:, ai] * bias[:, 2])
            sub_demand.append(demand[ids].cpu().numpy())

        env.sim.step = measured_step
        for step in range(HORIZON):
            for key in ("state", "ref", "active"):
                value = (io.read()[ids].cpu().numpy() if key == "state" else
                         cmd.library.ref71(cmd.frame)[ids].cpu().numpy() if key == "ref" else active.copy())
                traces[key].append(value)
            action = torch.zeros(40, 29, device=env.device)
            for label, these in label_slots.items():
                action[these] = policies[label](obs["policy"][these], obs["goal"][these])
            traces["action"].append(action[ids].cpu().numpy())
            traces["target"].append((action * term.scale + term.offset)[ids].cpu().numpy())
            sub_torque.clear(); sub_demand.clear(); sub_contact.clear()
            obs, _, terminated, timeout, _ = env.step(action)
            traces["torque"].append(np.stack(sub_torque))
            traces["demand"].append(np.stack(sub_demand))
            traces["contact"].append(np.stack(sub_contact))
            for slot, name, source, label, repeat in branches:
                if active[slot] and (bool(terminated[slot]) or bool(timeout[slot])):
                    events[str(slot)] = {"step": step + 1, "failed": bool(terminated[slot]),
                                         "reasons": [k for k in ("root_height", "root_tilt")
                                                     if bool(env.termination_manager.get_term(k)[slot])]}
                    active[slot] = False
        np.savez_compressed(path, **{k: np.stack(v) for k, v in traces.items()},
                            branches=np.asarray(branches, dtype=str), limits=limits,
                            meta=json.dumps({"events": events, "restore_errors": restore_errors,
                                             "source_delta": saved["d2_snapshot_max_abs_state_delta"],
                                             "checkpoint_hashes": {k: hashlib.sha256(v.read_bytes()).hexdigest()
                                                                   for k, v in CKPTS.items()},
                                             "jacobian_pre_velocity_max_abs_error": jac_error,
                                             "contact_fields": ["left_fn", "right_fn", "other_fn",
                                                                "post_vt_force", "pre_vt_force",
                                                                "post_motion_force", "pre_motion_force"]}))
        print("saved", path, "events", events, flush=True)
    finally:
        env.close()


def main(mode: str, source: str):
    if source not in CKPTS:
        raise ValueError(source)
    if mode == "source":
        from capture_policy_cross_sources import main as capture_sources

        capture_sources(source)
    elif mode == "branch":
        branch_run(source)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    tyro.cli(main)
