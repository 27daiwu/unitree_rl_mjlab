"""Replay A4 and W1.3 on the Flat assay and quantify foot/contact gait quality."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import re
import numpy as np
import torch
import tyro
from evaluate_g1_flat_control import COMMANDS

CHECKPOINTS = {
    "a4": "logs/rsl_rl/g1_velocity/2026-09-28_20-30-16_phase_a4_flat_control/model_499.pt",
    "w1_3": "logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt",
}
SCENARIOS = tuple(COMMANDS)
FEET = ("left", "right")
ANALYSIS_START = 7.0


@dataclass
class Config:
    output_dir: str = "logs/flat_w1_3_gait_diagnostics"
    checkpoint_a4: str = CHECKPOINTS["a4"]
    checkpoint_w1_3: str = CHECKPOINTS["w1_3"]
    checkpoint_control: str | None = None
    checkpoint_experiment: str | None = None
    task: str = "Unitree-G1-Flat"
    num_envs: int = 4
    seed: int = 42
    steps: int = 600
    device: str = "cuda:0"
    analyze_only: bool = False


def contact_labels(contact, force_z, time, threshold_n=5.0, clearance=None, foot_vz=None):
    """Classify force-supported contact onsets; never infer scuff from height."""
    force_contact = np.asarray(force_z) >= threshold_n
    time = np.asarray(time)
    onsets = force_contact & ~np.r_[False, force_contact[:-1]]
    events = []
    for idx in np.flatnonzero(onsets):
        prior_contact = np.flatnonzero(force_contact[:idx])
        air_start = prior_contact[-1] if len(prior_contact) else 0
        air_s = time[idx] - time[air_start] if idx > air_start else None
        if clearance is not None and clearance[idx] > 0.01:
            label = "swing_phase_contact_candidate"
        elif air_s is None or air_s >= 0.08:
            label = "touchdown_candidate"
        else:
            label = "brief_contact"
        events.append({"time_s": float(time[idx]), "normal_force_n": float(force_z[idx]),
                       "preceding_no_force_s": None if air_s is None else float(air_s),
                       "foot_collision_envelope_clearance_m": None if clearance is None else float(clearance[idx]),
                       "classification": label})
    return events


def _segments(mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def _normalize_quats(q):
    q = np.asarray(q)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return np.where(q[:, :1] < 0, -q, q)


def _rotate_quat(q, points):
    q = np.asarray(q)
    p = np.asarray(points)
    w = q[..., 0:1]
    v = q[..., 1:]
    return p + 2.0 * np.cross(v, np.cross(v, p) + w * p)


def summarize_trace(trace, command, geom=None, force_threshold_n=5.0):
    """Summarize the steady final five seconds from one 600-step trial."""
    t0 = np.asarray(trace["time"])
    keep = t0 > ANALYSIS_START
    t = t0[keep]
    command = np.asarray(command, dtype=float)
    actual = np.asarray(trace["velocity"])[keep]
    root = np.asarray(trace["position"])[keep]
    rpy = np.asarray(trace["rpy"])[keep]
    site = np.asarray(trace["foot_position"])[keep]
    quat = np.asarray(trace["foot_quaternion"])[keep]
    site_vel = np.asarray(trace["foot_velocity"])[keep]
    contact_time = np.asarray(trace["contact_time"])[keep]
    force_z = np.asarray(trace["normal_force"])[keep]
    force_substeps = np.asarray(trace["force_substeps"])[keep]
    action = np.asarray(trace["action"])[keep]
    joint_pos = np.asarray(trace["joint_pos"])[keep]
    joint_vel = np.asarray(trace["joint_vel"])[keep]
    joint_acc = np.asarray(trace["joint_acc"])[keep]
    actuator_force = np.asarray(trace["actuator_force"])[keep]
    actuator_limit = np.asarray(trace["actuator_limit"])[keep]
    joint_limits = np.asarray(trace["joint_limits"])[keep]
    sat = np.abs(actuator_force) >= (0.98 * np.maximum(actuator_limit, 1e-8))
    joint_margin = np.minimum(joint_pos - joint_limits[:, :, 0], joint_limits[:, :, 1] - joint_pos)
    reward_components = np.asarray(trace["reward_components"])[keep]
    reward_names = trace["reward_names"]
    contact = (contact_time > 0.0) | (force_z >= force_threshold_n)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    cmd_axis = int(np.flatnonzero(command)[0])
    velocity_error = actual - command
    orientation_rms = np.sqrt(np.mean(np.square(rpy[:, :2]), axis=0)) if len(t) else [None, None]
    result = {
        "command_velocity": command.tolist(),
        "actual_velocity_mean": np.mean(actual, axis=0).tolist() if len(t) else None,
        "velocity_tracking_error_mean": np.mean(velocity_error, axis=0).tolist() if len(t) else None,
        "velocity_tracking_rmse": np.sqrt(np.mean(velocity_error ** 2, axis=0)).tolist() if len(t) else None,
        "roll_pitch_rms_rad": np.asarray(orientation_rms).tolist(),
        "base_height_mean_m": float(np.mean(root[:, 2])) if len(t) else None,
        "double_support_fraction": float(np.mean(contact.all(axis=1))) if len(t) else None,
        "double_support_duration_total_s": float(np.sum(contact.all(axis=1)) * dt) if len(t) else None,
        "double_support_bout_duration_mean_s": _mean_or_none([
            float((b-a)*dt) for a,b in _segments(contact.all(axis=1)) if b-a >= 2
        ]),
        "single_support_fraction": float(np.mean(contact.sum(axis=1) == 1)) if len(t) else None,
        "normal_contact_force_peak_n": np.max(force_z, axis=0).tolist() if len(t) else None,
        "normal_contact_force_mean_n": [float(force_z[:, i][force_z[:, i] > 0].mean()) if np.any(force_z[:, i] > 0) else None for i in range(2)],
        "reward_component_weighted_mean_per_s": {n: float(reward_components[:, i].mean()) for i, n in enumerate(reward_names)},
        "action_delta_rms": np.sqrt(np.mean(np.diff(action, axis=0) ** 2, axis=0)).tolist() if len(t) > 1 else None,
        "joint_velocity_abs_peak_rad_s": float(np.max(np.abs(joint_vel))) if len(t) else None,
        "joint_acc_abs_rms_rad_s2": float(np.sqrt(np.mean(joint_acc ** 2))) if len(t) else None,
        "joint_position_abs_peak_rad": float(np.max(np.abs(joint_pos))) if len(t) else None,
        "actuator_force_abs_peak": float(np.max(np.abs(actuator_force))) if len(t) else None,
        "actuator_effort_limit_fraction_peak": float(np.max(np.abs(actuator_force) / np.maximum(actuator_limit, 1e-8))) if len(t) else None,
        "actuator_effort_saturation_sample_fraction": float(np.mean(sat)) if len(t) else None,
        "joint_soft_limit_min_margin_rad": float(np.min(joint_margin)) if len(t) else None,
        "feet": {},
        "toe_heel_clearance_status": "UNKNOWN" if geom is None else "computed_from_collision_capsule_envelope",
    }
    for i, foot in enumerate(FEET):
        in_contact = contact[:, i]
        swings = _segments(~in_contact)
        stances = _segments(in_contact)
        stance_durations = [float((b-a)*dt) for a,b in stances if b-a >= 2]
        swing_durations = [float((b-a)*dt) for a,b in swings if b-a >= 2]
        step_lengths = []
        swing_excursions = []
        swing_site_clearances = []
        toe_clearances = []
        heel_clearances = []
        swing_events = []
        world_vel = site_vel[:, i, :2]
        stance_vel = np.linalg.norm(world_vel[in_contact], axis=-1)
        sliding_distance = float(np.sum(np.linalg.norm(world_vel[in_contact], axis=-1)) * dt)
        foot_envelope_clearance = np.zeros(len(t))
        if geom is not None:
            points = geom[foot]
            rotated_all = _rotate_quat(quat[:, i, None, :], points[None, :, :])
            foot_envelope_clearance = site[:, i, None, 2] + rotated_all[:, :, 2]
            foot_envelope_clearance = np.min(foot_envelope_clearance, axis=1)
        substep_times = (t[:, None] - np.array([0.015, 0.010, 0.005, 0.0])[None, :]).reshape(-1)
        substep_force = force_substeps[:, i, ::-1].reshape(-1)
        substep_clearance = np.repeat(foot_envelope_clearance, 4)
        substep_vz = np.repeat(site_vel[:, i, 2], 4)
        swing_events = contact_labels(
            np.repeat(in_contact, 4), substep_force, substep_times, force_threshold_n,
            clearance=substep_clearance if geom is not None else None, foot_vz=substep_vz,
        )
        for a, b in swings:
            if b-a < 2:
                continue
            if a > 0 and b < len(t):
                yaw = float(rpy[a, 2])
                delta = site[b, i, :2] - site[a, i, :2]
                if cmd_axis == 0:
                    direction = np.array([math.cos(yaw), math.sin(yaw)]) * np.sign(command[0])
                    swing_excursions.append(float(abs(np.dot(delta, direction))))
                elif cmd_axis == 1:
                    direction = np.array([-math.sin(yaw), math.cos(yaw)]) * np.sign(command[1])
                    swing_excursions.append(float(abs(np.dot(delta, direction))))
                else:
                    swing_excursions.append(float(np.linalg.norm(delta)))
            swing_site_clearances.append(float(np.max(site[a:b, i, 2])))
            if geom is not None:
                points = geom[foot]
                rotated = _rotate_quat(quat[a:b, i, None, :], points[None, :, :])
                heights = site[a:b, i, None, 2] + rotated[:, :, 2]
                toe_clearances.append(float(np.max(np.min(heights[:, points[:, 0] > 0.05], axis=1))))
                heel_clearances.append(float(np.max(np.min(heights[:, points[:, 0] < -0.03], axis=1))))
        touchdown_indices = [a for a, b in stances if b-a >= 2]
        touchdown_periods = np.diff(touchdown_indices) * dt
        for first, second in zip(touchdown_indices, touchdown_indices[1:]):
            yaw = float(rpy[first, 2])
            root_delta = root[second, :2] - root[first, :2]
            if cmd_axis == 0:
                direction = np.array([math.cos(yaw), math.sin(yaw)]) * np.sign(command[0])
                step_lengths.append(float(abs(np.dot(root_delta, direction))))
            elif cmd_axis == 1:
                direction = np.array([-math.sin(yaw), math.cos(yaw)]) * np.sign(command[1])
                step_lengths.append(float(abs(np.dot(root_delta, direction))))
            else:
                step_lengths.append(float(np.linalg.norm(root_delta)))
        freq = 1.0 / _mean_or_none(touchdown_periods) if len(touchdown_periods) else None
        result["feet"][foot] = {
            "foot_site_pose_world_mean_xyz_m": np.mean(site[:, i], axis=0).tolist() if len(t) else None,
            "foot_orientation_world_quaternion_wxyz_mean": _normalize_quats(quat[:, i]).mean(axis=0).tolist() if len(t) else None,
            "foot_world_velocity_mean_xy_m_s": np.mean(world_vel, axis=0).tolist() if len(t) else None,
            "stance_duration_mean_s": _mean_or_none(stance_durations),
            "swing_duration_mean_s": _mean_or_none(swing_durations),
            "step_length_mean_m": _mean_or_none(step_lengths),
            "swing_excursion_mean_m": _mean_or_none(swing_excursions),
            "step_width_mean_m": float(np.mean(np.abs(site[:, 0, 1] - site[:, 1, 1]))) if len(t) else None,
            "step_frequency_hz": freq,
            "stance_foot_speed_mean_m_s": _mean_or_none(stance_vel),
            "stance_foot_sliding_distance_m": sliding_distance,
            "foot_site_swing_clearance_peak_mean_m": _mean_or_none(swing_site_clearances),
            "toe_clearance_peak_mean_m": _mean_or_none(toe_clearances),
            "heel_clearance_peak_mean_m": _mean_or_none(heel_clearances),
            "unexpected_swing_contact_candidates": [e for e in swing_events if e["classification"] == "swing_phase_contact_candidate"],
            "force_contact_onsets": contact_labels(contact[:, i], force_z[:, i], t, force_threshold_n,
                                                   clearance=foot_envelope_clearance if geom is not None else None,
                                                   foot_vz=site_vel[:, i, 2]),
        }
    freqs = [v["step_frequency_hz"] for v in result["feet"].values() if v["step_frequency_hz"] is not None]
    result["step_frequency_hz"] = float(np.mean(freqs)) if freqs else None
    result["contact_event_counts"] = {foot: len(result["feet"][foot]["force_contact_onsets"]) for foot in FEET}
    return result


def load_foot_collision_points(xml_path):
    """Read collision capsule endpoint geometry from the robot MJCF."""
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    radius = float(root.find(".//default[@class='foot_capsule']/geom").attrib["size"])
    geometry = {}
    for side in FEET:
        body = None
        site_pos = None
        for element in root.iter("body"):
            site = next((s for s in element.findall("site") if s.attrib.get("name") == f"{side}_foot"), None)
            if site is not None:
                body = element
                site_pos = np.fromstring(site.attrib["pos"], sep=" ")
                break
        points = []
        for geom in body.findall("geom"):
            if not geom.attrib.get("name", "").startswith(f"{side}_foot"):
                continue
            endpoints = np.fromstring(geom.attrib["fromto"], sep=" ").reshape(2, 3)
            for endpoint in endpoints:
                for axis in range(3):
                    for sign in (-1, 1):
                        offset = np.zeros(3)
                        offset[axis] = sign * radius
                        points.append(endpoint + offset - site_pos)
        geometry[side] = np.asarray(points)
    return geometry


def _capture(env):
    robot = env.scene["robot"]
    data = robot.data
    env.navigation_snapshot = torch.cat((
        data.root_link_lin_vel_b[:, :2], data.root_link_ang_vel_b[:, 2:3],
        data.root_link_pos_w, data.root_link_quat_w,
    ), dim=1).detach().clone()
    site_ids = [robot.site_names.index("left_foot"), robot.site_names.index("right_foot")]
    sensor = env.scene["feet_ground_contact"].data
    effort_limits = np.zeros(robot.num_actuators, dtype=np.float32)
    for actuator in robot.actuators:
        ctrl_ids = actuator.ctrl_ids.detach().cpu().numpy().astype(int)
        limit = actuator.cfg.effort_limit
        if limit is not None:
            effort_limits[ctrl_ids] = float(limit)
    env.gait_diag = {
        "site": robot.data.site_pos_w[:, site_ids].clone(),
        "quat": robot.data.site_quat_w[:, site_ids].clone(),
        "velocity": robot.data.site_lin_vel_w[:, site_ids, :].clone(),
        "contact_time": sensor.current_contact_time.clone(),
        "air_time": sensor.current_air_time.clone(),
        "force": sensor.force.clone(),
        "force_history": sensor.force_history.clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        "joint_acc": robot.data.joint_acc.clone(),
        "actuator_force": robot.data.actuator_force.clone(),
        "actuator_limit": torch.as_tensor(effort_limits, device=env.device).expand(env.num_envs, -1),
        "joint_limits": robot.data.soft_joint_pos_limits.clone(),
        "action": env.action_manager.action.clone(),
        "reward": env.reward_manager._step_reward.clone(),
        "reward_names": tuple(env.reward_manager.active_terms),
    }
    return torch.zeros(env.num_envs, device=env.device)


def run(cfg):
    if cfg.analyze_only:
        return analyze_saved(cfg)
    if cfg.steps != 600 or cfg.num_envs != 4 or cfg.seed != 42 or cfg.task != "Unitree-G1-Flat":
        raise ValueError("W1.3-D requires Flat, seed 42, four replicas, and exactly 600 steps")
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from src.tasks.velocity.mdp.navigation import evaluation_observations
    from evaluate_g1_navigation import snapshot, decode

    output = Path(cfg.output_dir)
    if (output / "manifest.json").exists() or any(output.glob("*.npz")):
        raise FileExistsError(f"Refusing to overwrite diagnostics: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = ({"control": cfg.checkpoint_control, "experiment": cfg.checkpoint_experiment}
                   if cfg.checkpoint_control and cfg.checkpoint_experiment else
                   {"a4": cfg.checkpoint_a4, "w1_3": cfg.checkpoint_w1_3})
    if bool(cfg.checkpoint_control) != bool(cfg.checkpoint_experiment):
        raise ValueError("Both W1.4 checkpoints must be provided")
    geom = load_foot_collision_points("src/assets/robots/unitree_g1/xmls/g1.xml")
    manifest = {
        "config": asdict(cfg), "checkpoints": checkpoints, "commands": COMMANDS,
        "scenarios": list(SCENARIOS), "evaluation_steps": 600, "step_dt_s": 0.02,
        "physics_dt_s": 0.005, "contact_force_history_substeps": 4,
        "fixed_command_overrides": "same as evaluate_g1_flat_control; no observation/action/reward configuration changes",
        "sensor_sampling": "contact force history every 5 ms, exported each 20 ms policy step",
        "foot_geometry": "toe/heel extrema derived from named MJCF collision capsule endpoints and radius; not inferred from foot site",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    trials = {}
    for label, checkpoint in checkpoints.items():
        env_cfg = load_env_cfg(cfg.task)
        env_cfg.seed = cfg.seed
        env_cfg.scene.num_envs = cfg.num_envs
        env_cfg.episode_length_s = 1000.0
        env_cfg.events["reset_base"].params["pose_range"] = {}
        env_cfg.events = {k: v for k, v in env_cfg.events.items() if k in ("reset_base", "reset_robot_joints")}
        env_cfg.curriculum = {}
        for group in env_cfg.observations.values():
            group.enable_corruption = False
        sensor_cfg = next(s for s in env_cfg.scene.sensors if s.name == "feet_ground_contact")
        sensor_cfg.history_length = 4
        env_cfg.metrics["gait_diagnostic_snapshot"] = MetricsTermCfg(func=_capture)
        env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
        try:
            rl = load_rl_cfg(cfg.task)
            vec = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
            runner = load_runner_cls(cfg.task)(vec, asdict(rl), device=cfg.device)
            runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=cfg.device)
            policy = runner.get_inference_policy(device=cfg.device)
            for scenario, command in COMMANDS.items():
                env.reset(seed=cfg.seed)
                term = env.command_manager.get_term("twist")
                term._update_command = lambda: None
                term.time_left.fill_(1e9)
                traces = [dict(time=[], command=[], velocity=[], position=[], rpy=[], action=[],
                               contact_time=[], air_time=[], foot_position=[], foot_quaternion=[],
                               foot_velocity=[], normal_force=[], force_substeps=[], joint_pos=[],
                               joint_vel=[], joint_acc=[], actuator_force=[], actuator_limit=[],
                               joint_limits=[], reward_components=[]) for _ in range(cfg.num_envs)]
                alive = np.ones(cfg.num_envs, dtype=bool)
                reward_names = None
                for step in range(600):
                    term.vel_command_b[:] = term.vel_command_b.new_tensor(command)
                    term.is_standing_env[:] = False
                    term.is_heading_env[:] = False
                    term.time_left.fill_(1e9)
                    obs = evaluation_observations(env)
                    with torch.no_grad():
                        action = policy(obs)
                        _, _, terminated, truncated, _ = env.step(action)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("non-finite policy action")
                    gait = env.gait_diag
                    reward_names = gait["reward_names"]
                    velocity, position, rpy = decode(env.navigation_snapshot)
                    site = gait["site"].cpu().numpy()
                    force_hist = gait["force_history"].cpu().numpy()
                    force_z = np.abs(gait["force"].cpu().numpy()[:, :, 2])
                    for i in np.flatnonzero(alive):
                        tr = traces[i]
                        values = {
                            "time": (step + 1) * env.step_dt, "command": command,
                            "velocity": velocity[i], "position": position[i], "rpy": rpy[i],
                            "action": gait["action"][i].cpu().numpy(),
                            "contact_time": gait["contact_time"][i].cpu().numpy(),
                            "air_time": gait["air_time"][i].cpu().numpy(),
                            "foot_position": site[i], "foot_quaternion": gait["quat"][i].cpu().numpy(),
                            "foot_velocity": gait["velocity"][i].cpu().numpy(),
                            "normal_force": force_z[i],
                            "force_substeps": np.abs(force_hist[i, :, :, 2]),
                            "joint_pos": gait["joint_pos"][i].cpu().numpy(),
                            "joint_vel": gait["joint_vel"][i].cpu().numpy(),
                            "joint_acc": gait["joint_acc"][i].cpu().numpy(),
                            "actuator_force": gait["actuator_force"][i].cpu().numpy(),
                            "actuator_limit": gait["actuator_limit"][i].cpu().numpy(),
                            "joint_limits": gait["joint_limits"][i].cpu().numpy(),
                            "reward_components": gait["reward"][i].cpu().numpy(),
                        }
                        for key, value in values.items():
                            tr[key].append(np.asarray(value).copy())
                    alive &= ~(terminated.cpu().numpy().astype(bool) | truncated.cpu().numpy().astype(bool))
                    if not alive.any():
                        break
                for i, tr in enumerate(traces):
                    tr["reward_names"] = reward_names
                    path = output / f"{label}_{scenario}_{i}.npz"
                    np.savez_compressed(path, **{k: np.asarray(v) for k, v in tr.items() if k != "reward_names"})
                    trials[(label, scenario, i)] = summarize_trace(tr, command, geom)
                    trials[(label, scenario, i)]["completed"] = len(tr["time"]) == 600 and bool(alive[i])
                    trials[(label, scenario, i)]["terminated_early"] = not bool(alive[i])
        finally:
            env.close()
    comparison = aggregate(trials, tuple(checkpoints))
    (output / "metrics.json").write_text(json.dumps({"trials": _json_trials(trials), "comparison": comparison}, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(render_summary(comparison, manifest))
    print(json.dumps({"output_dir": str(output), "cases": comparison}, indent=2))
    return comparison


def analyze_saved(cfg):
    """Recompute metrics from completed diagnostic traces without a simulator."""
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    output = Path(cfg.output_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    labels = tuple(manifest["checkpoints"])
    env_cfg = load_env_cfg(cfg.task)
    reward_names = tuple(env_cfg.rewards)
    robot_cfg = env_cfg.scene.entities["robot"]
    joint_names = []
    joint_limits = []
    import xml.etree.ElementTree as ET
    root = ET.parse("src/assets/robots/unitree_g1/xmls/g1.xml").getroot()
    for joint in root.iter("joint"):
        if joint.attrib.get("type") == "free":
            continue
        joint_names.append(joint.attrib["name"])
        joint_limits.append(np.fromstring(joint.attrib["range"], sep=" "))
    joint_limits = np.asarray(joint_limits, dtype=float)
    factor = robot_cfg.articulation.soft_joint_pos_limit_factor
    mid = joint_limits.mean(axis=1)
    half = (joint_limits[:, 1] - joint_limits[:, 0]) * factor / 2
    soft_limits = np.stack((mid - half, mid + half), axis=-1)
    actuator_limits = []
    for actuator_cfg in robot_cfg.articulation.actuators:
        matches = [name for name in joint_names
                   if any(re.fullmatch(pattern, name) for pattern in actuator_cfg.target_names_expr)]
        actuator_limits.extend([float(actuator_cfg.effort_limit)] * len(matches))
    if len(actuator_limits) != 29 or len(joint_names) != 29:
        raise RuntimeError("Could not map the 29 G1 joints/actuator effort limits from the registered Flat config")
    trials = {}
    for label in labels:
        for scenario in SCENARIOS:
            for i in range(4):
                path = output / f"{label}_{scenario}_{i}.npz"
                with np.load(path) as z:
                    trace = {k: z[k] for k in z.files}
                trace["reward_names"] = reward_names
                trace["actuator_limit"] = np.broadcast_to(actuator_limits, trace["actuator_force"].shape)
                trace["joint_limits"] = np.broadcast_to(soft_limits, trace["joint_pos"].shape + (2,))
                row = summarize_trace(trace, COMMANDS[scenario], load_foot_collision_points(
                    "src/assets/robots/unitree_g1/xmls/g1.xml"))
                row["completed"] = len(trace["time"]) == 600
                row["terminated_early"] = not row["completed"]
                trials[(label, scenario, i)] = row
    comparison = aggregate(trials, labels)
    (output / "metrics.json").write_text(json.dumps({"trials": _json_trials(trials), "comparison": comparison}, indent=2, allow_nan=False) + "\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    (output / "summary.md").write_text(render_summary(comparison, manifest))
    print(json.dumps({"analysis_only": True, "output_dir": str(output)}, indent=2))
    return comparison


def aggregate(trials, labels=("a4", "w1_3")):
    out = {}
    for scenario in SCENARIOS:
        out[scenario] = {}
        for label in labels:
            rows = [trials[(label, scenario, i)] for i in range(4)]
            out[scenario][label] = _aggregate_rows(rows)
        out[scenario][f"{labels[1]}_minus_{labels[0]}"] = _difference(out[scenario][labels[1]], out[scenario][labels[0]])
    return out


def _json_trials(trials):
    return {f"{label}/{scenario}/replica_{replica}": values
            for (label, scenario, replica), values in trials.items()}


def _aggregate_rows(rows):
    keys = ("actual_velocity_mean", "velocity_tracking_error_mean", "velocity_tracking_rmse",
            "roll_pitch_rms_rad", "base_height_mean_m", "double_support_fraction",
            "double_support_duration_total_s", "double_support_bout_duration_mean_s",
            "single_support_fraction", "step_frequency_hz", "actuator_effort_limit_fraction_peak",
            "actuator_effort_saturation_sample_fraction", "joint_soft_limit_min_margin_rad")
    out = {k: np.mean([r[k] for r in rows if r[k] is not None], axis=0).tolist()
           if rows and any(r[k] is not None for r in rows) else None for k in keys}
    out["completed_trials"] = sum(bool(r["completed"]) for r in rows)
    out["normal_contact_force_peak_n"] = np.mean([r["normal_contact_force_peak_n"] for r in rows], axis=0).tolist()
    out["normal_contact_force_mean_n"] = np.nanmean([
        [np.nan if x is None else x for x in r["normal_contact_force_mean_n"]] for r in rows
    ], axis=0).tolist()
    out["feet"] = {}
    for foot in FEET:
        frows = [r["feet"][foot] for r in rows]
        metrics = ("stance_duration_mean_s", "swing_duration_mean_s", "step_length_mean_m",
                   "swing_excursion_mean_m", "step_width_mean_m", "step_frequency_hz", "stance_foot_speed_mean_m_s",
                   "stance_foot_sliding_distance_m", "foot_site_swing_clearance_peak_mean_m",
                   "toe_clearance_peak_mean_m", "heel_clearance_peak_mean_m")
        out["feet"][foot] = {k: _mean_or_none([r[k] for r in frows if r[k] is not None]) for k in metrics}
        out["feet"][foot]["swing_contact_event_count"] = sum(len(r["unexpected_swing_contact_candidates"]) for r in frows)
    out["reward_component_weighted_mean_per_s"] = {
        name: float(np.mean([r["reward_component_weighted_mean_per_s"][name] for r in rows]))
        for name in rows[0]["reward_component_weighted_mean_per_s"]
    }
    out["contact_event_counts"] = {foot: sum(r["contact_event_counts"][foot] for r in rows) for foot in FEET}
    return out


def _difference(new, old):
    result = {}
    for key in ("actual_velocity_mean", "velocity_tracking_error_mean", "velocity_tracking_rmse", "roll_pitch_rms_rad"):
        result[key] = (np.asarray(new[key]) - np.asarray(old[key])).tolist() if new[key] is not None and old[key] is not None else None
    result["double_support_fraction"] = new["double_support_fraction"] - old["double_support_fraction"]
    result["step_frequency_hz"] = new["step_frequency_hz"] - old["step_frequency_hz"]
    result["feet"] = {
        foot: {key: new["feet"][foot][key] - old["feet"][foot][key]
               for key in new["feet"][foot]
               if isinstance(new["feet"][foot][key], (int, float)) and isinstance(old["feet"][foot][key], (int, float))}
        for foot in FEET
    }
    return result


def render_summary(comparison, manifest):
    lines = [
        "# Flat Gait Diagnostics", "",
        f"{' and '.join(manifest['checkpoints'])} use Flat, seed 42, four replicas, seven pinned commands across six directions, and 600 policy steps. Per-trial data is in metrics.json.", "",
        f"Contact sampling: {manifest['sensor_sampling']}.", "",
        "## Tracking and stability", "",
        "| Command | Policy | Command | Actual velocity | Velocity RMSE | Roll/pitch RMS (rad) | Base height (m) | Completed |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for scenario, values in comparison.items():
        for label in manifest["checkpoints"]:
            row = values[label]
            command = values[label].get("command_velocity")
            if command is None:
                command = manifest["commands"][scenario]
            lines.append(f"| {scenario} | {label} | {command} | {row['actual_velocity_mean']} | {row['velocity_tracking_rmse']} | {row['roll_pitch_rms_rad']} | {row['base_height_mean_m']:.3f} | {row['completed_trials']}/4 |")
    lines += ["", "## Gait geometry and contact", "",
              "| Command | Policy | DS fraction / total s | Mean DS bout s | Cycle Hz/foot | Step length L/R (m) | Swing excursion L/R (m) | Swing L/R (s) | Toe clearance L/R (m) | Heel clearance L/R (m) | Stance speed L/R (m/s) | Slip distance L/R (m) | Peak normal force L/R (N) | Swing contact candidates L/R | Effort limit peak / saturated fraction |"]
    lines += ["|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|"]
    for scenario, values in comparison.items():
        for label in manifest["checkpoints"]:
            row = values[label]
            lengths = [row["feet"][f]["step_length_mean_m"] for f in FEET]
            excursions = [row["feet"][f]["swing_excursion_mean_m"] for f in FEET]
            swings = [row["feet"][f]["swing_duration_mean_s"] for f in FEET]
            toes = [row["feet"][f]["toe_clearance_peak_mean_m"] for f in FEET]
            heels = [row["feet"][f]["heel_clearance_peak_mean_m"] for f in FEET]
            stance_speed = [row["feet"][f]["stance_foot_speed_mean_m_s"] for f in FEET]
            slide = [row["feet"][f]["stance_foot_sliding_distance_m"] for f in FEET]
            events = [row["feet"][f]["swing_contact_event_count"] for f in FEET]
            limits = [row["actuator_effort_limit_fraction_peak"], row["actuator_effort_saturation_sample_fraction"]]
            lines.append(f"| {scenario} | {label} | {row['double_support_fraction']:.3f} / {row['double_support_duration_total_s']:.3f} | {row['double_support_bout_duration_mean_s']:.3f} | {row['step_frequency_hz']:.2f} | {lengths} | {excursions} | {swings} | {toes} | {heels} | {stance_speed} | {slide} | {row['normal_contact_force_peak_n']} | {events} | {limits} |")
    lines += [
        "", "## Interpretation limits", "",
        "Foot-site height is not toe clearance. Toe and heel collision-envelope heights use the MJCF capsule endpoints and radius. Contact uses measured normal force and contact timing, not foot height. Sensor history samples 5 ms physics substeps and stores four samples per 20 ms policy step. Reward terms are read from the normal reward-manager update; reward functions are not called a second time. The six-direction set includes two forward speed commands.", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run(tyro.cli(Config))
