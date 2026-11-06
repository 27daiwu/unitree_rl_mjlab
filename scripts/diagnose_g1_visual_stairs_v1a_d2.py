"""Matched, read-only Flat/V1A startup and control-chain diagnosis."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # noqa: F401


TASK = "Unitree-G1-Visual-Stairs-V1A"
FLAT = "Unitree-G1-Flat"
OUT = Path("docs/assets/g1_visual_stairs_v1a/d2")
CHECKPOINTS = {
  "v1a": Path("logs/rsl_rl/g1_visual_stairs_v1a/2026-11-04_11-29-53_v1a_pilot_20261104/model_99.pt"),
  "flat": Path("logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt"),
}
FOOT_GEOMS = tuple(f"robot/{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))


def array(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy().copy()


def pitch_deg(quat: np.ndarray) -> np.ndarray:
  w, x, y, z = quat.T
  return np.degrees(np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)))


def make_cfg(label: str):
  cfg = load_env_cfg(TASK)
  cfg.seed = 142
  cfg.scene.num_envs = 16
  cfg.observations["actor"].enable_corruption = False
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0), "yaw": (-0.03, 0.03),
  }
  return cfg


def schema(env, label: str) -> dict:
  action = env.action_manager.get_term("joint_pos")
  robot = env.scene["robot"]
  model = env.sim.mj_model
  return {
    "actor_terms": env.observation_manager.active_terms["actor"],
    "actor_dims": [list(x) if isinstance(x, tuple) else x for x in env.observation_manager.group_obs_term_dim["actor"]],
    "obs_groups": {k: list(v) for k, v in load_rl_cfg(FLAT if label == "flat" else TASK).obs_groups.items()},
    "action_dim": action.action_dim,
    "target_names": action.target_names,
    "target_ids": action.target_ids.tolist(),
    "scale": array(action.scale[0]).tolist() if isinstance(action.scale, torch.Tensor) else action.scale,
    "offset": array(action.offset[0]).tolist() if isinstance(action.offset, torch.Tensor) else action.offset,
    "joint_names": list(robot.joint_names),
    "actuator_names": list(robot.actuator_names),
    "actuator_gainprm": model.actuator_gainprm.tolist(),
    "actuator_biasprm": model.actuator_biasprm.tolist(),
    "actuator_forcerange": model.actuator_forcerange.tolist(),
    "dof_armature": model.dof_armature.tolist(),
    "physics_dt_s": env.physics_dt,
    "control_dt_s": env.step_dt,
    "decimation": env.cfg.decimation,
  }


def cpu_geometry(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> dict:
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  foot_geom_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                   for name in FOOT_GEOMS}
  if -1 in foot_geom_ids:
    raise RuntimeError("A foot collision geom is missing from the compiled model")
  terrain_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  if terrain_body < 0:
    raise RuntimeError("Compiled model has no terrain body")
  heights = {"left": [], "right": []}
  contacts = {"left": 0, "right": 0}
  for name in FOOT_GEOMS:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    side = name.split("/")[1].split("_")[0]
    heights[side].append(float(data.geom_xpos[idx, 2] - model.geom_size[idx, 0]))
  for contact in data.contact:
    for geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
      if geom in foot_geom_ids and model.geom_bodyid[other] == terrain_body:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        contacts[name.split("/")[1].split("_")[0]] += 1
  return {"left_collision_bottom_m": min(heights["left"]),
          "right_collision_bottom_m": min(heights["right"]),
          "left_cpu_contact_count": contacts["left"],
          "right_cpu_contact_count": contacts["right"]}


def evaluate(label: str):
  cfg = make_cfg(label)
  checkpoint = CHECKPOINTS[label]
  saved_cfg = yaml.load((checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
  native_flat_cfg = yaml.load((CHECKPOINTS["flat"].parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
  saved_actor_terms = list(saved_cfg["observations"]["actor"]["terms"])
  current_actor_terms = list(cfg.observations["actor"].terms)
  if saved_actor_terms != current_actor_terms:
    raise RuntimeError(f"{label} saved/current actor term mismatch: {saved_actor_terms} != {current_actor_terms}")
  if saved_cfg["actions"]["joint_pos"] != native_flat_cfg["actions"]["joint_pos"]:
    raise RuntimeError(f"{label} saved action config differs from Flat checkpoint")
  if saved_cfg["observations"]["actor"] != native_flat_cfg["observations"]["actor"]:
    raise RuntimeError(f"{label} saved proprioceptive actor observation config differs from Flat checkpoint")
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    vec = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(FLAT if label == "flat" else TASK).clip_actions)
    info = schema(env, label)
    info["saved_actor_terms"] = saved_actor_terms
    info["saved_actor_group_matches_flat"] = True
    info["saved_action_config_matches_flat"] = True
    if label == "flat":
      native = load_env_cfg(FLAT)
      if list(cfg.observations["actor"].terms) != list(native.observations["actor"].terms):
        raise RuntimeError("Flat actor observation order does not match its native task")
      if asdict(cfg.actions["joint_pos"]) != asdict(native.actions["joint_pos"]):
        raise RuntimeError("Flat and V1A action configuration differs")
      if asdict(cfg.scene.entities["robot"]) != asdict(native.scene.entities["robot"]):
        raise RuntimeError("Flat and V1A robot configuration differs")
    runner_cls = load_runner_cls(FLAT if label == "flat" else TASK)
    assert runner_cls is not None
    runner = runner_cls(vec, asdict(load_rl_cfg(FLAT if label == "flat" else TASK)), device="cuda:0")
    # Strict loading checks network structure and normalization buffers.
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    info["checkpoint"] = str(checkpoint)
    info["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    info["strict_actor_load"] = True
    robot = env.scene["robot"]
    foot = env.scene["feet_ground_contact"]
    foot_ids, foot_names = robot.find_sites(("left_foot", "right_foot"))
    if list(foot_names) != ["left_foot", "right_foot"]:
      raise RuntimeError(f"Unexpected foot sites: {foot_names}")
    cpu_data = mujoco.MjData(env.sim.mj_model)
    origins = array(env.scene.env_origins)
    initial = {"qpos": array(env.sim.data.qpos), "qvel": array(env.sim.data.qvel),
               "origins": origins, "default_joint_pos": array(robot.data.default_joint_pos),
               "encoder_bias": array(robot.data.encoder_bias),
               "initial_command": array(env.command_manager.get_term("twist").vel_command_b)}
    for field in ("geom_friction", "body_ipos", "body_subtreemass"):
      initial[field] = array(getattr(env.sim.model, field))
    rows = []
    active = np.ones(16, dtype=bool)
    control_step = 0
    substep = 0

    def capture(phase: str, qpos: np.ndarray | None = None):
      pos = array(robot.data.root_link_pos_w)
      quat = array(robot.data.root_link_quat_w)
      world_vel = array(robot.data.root_link_lin_vel_w)
      site = array(robot.data.site_pos_w[:, foot_ids])
      found = array(foot.data.found[:, :2])
      force = array(foot.data.force[:, :2])
      target = array(robot.data.joint_pos_target)
      joint_pos = array(robot.data.joint_pos)
      raw_actions = array(env.action_manager.get_term("joint_pos").raw_action)
      current_command = array(env.command_manager.get_term("twist").vel_command_b)
      ctrl = array(env.sim.data.ctrl)
      effort = array(env.sim.data.qfrc_actuator)
      qpos = array(env.sim.data.qpos) if qpos is None else qpos
      done = array(env.reset_buf) if phase == "terminal" else np.zeros(16, dtype=bool)
      p = pitch_deg(quat)
      for i in range(16):
        if not active[i]:
          continue
        geometry = cpu_geometry(env.sim.mj_model, cpu_data, qpos[i]) if phase in ("reset", "substep", "terminal") and (control_step == 0 or substep <= 3 or phase == "terminal") else {}
        rows.append({
          "policy": label, "env_id": i,
          "phase": "terminal" if phase == "terminal" and done[i] else "control" if phase == "terminal" else phase,
          "control_step": control_step, "physics_substep": substep,
          "time_s": round(substep * env.physics_dt, 6),
          "pitch_deg": float(p[i]), "root_z_m": float(pos[i, 2] - origins[i, 2]),
          "world_x_velocity_mps": float(world_vel[i, 0]),
          "left_site_z_m": float(site[i, 0, 2] - origins[i, 2]),
          "right_site_z_m": float(site[i, 1, 2] - origins[i, 2]),
          "left_sensor_contact": int(found[i, 0] > 0),
          "right_sensor_contact": int(found[i, 1] > 0),
          "left_sensor_force_n": float(np.linalg.norm(force[i, 0])),
          "right_sensor_force_n": float(np.linalg.norm(force[i, 1])),
          "left_collision_bottom_m": geometry.get("left_collision_bottom_m"),
          "right_collision_bottom_m": geometry.get("right_collision_bottom_m"),
          "left_cpu_contact_count": geometry.get("left_cpu_contact_count"),
          "right_cpu_contact_count": geometry.get("right_cpu_contact_count"),
          "command_mps": json.dumps(current_command[i].tolist()),
          "raw_action": json.dumps(raw_actions[i].tolist()) if phase != "reset" else None,
          "joint_target_rad": json.dumps(target[i].tolist()),
          "joint_pos_rad": json.dumps(joint_pos[i].tolist()),
          "sim_ctrl": json.dumps(ctrl[i].tolist()),
          "actuator_effort": json.dumps(effort[i].tolist()),
          "done": int(done[i]),
        })
        if phase == "terminal" and done[i]:
          active[i] = False

    capture("reset")
    original_step = env.sim.step
    def capture_substep():
      nonlocal substep
      original_step()
      substep += 1
      if substep <= 3:
        capture("substep")
    env.sim.step = capture_substep
    original_metrics = env.metrics_manager.compute
    def capture_terminal():
      original_metrics()
      capture("terminal")
    env.metrics_manager.compute = capture_terminal
    command = env.command_manager.get_term("twist")
    obs = vec.get_observations()
    initial["actor_obs"] = array(obs["actor"])
    for control_step in range(1, 16):
      command.vel_command_b[:] = torch.tensor((0.4, 0.0, 0.0), device="cuda:0")
      command.time_left.fill_(1e9)
      with torch.no_grad():
        raw = policy(obs)
        obs, reward, _, _ = vec.step(raw)
      if not torch.isfinite(raw).all() or not torch.isfinite(reward).all():
        raise RuntimeError(f"Non-finite output in {label}")
      if not active.any():
        break
    return info, initial, rows
  finally:
    env.close()


def plot(rows, path):
  fig, axs = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
  for label, color in (("v1a", "#c7433b"), ("flat", "#176b87")):
    for i in range(16):
      ep = [r for r in rows if r["policy"] == label and r["env_id"] == i and r["phase"] in ("control", "terminal")]
      t = [r["time_s"] for r in ep]
      axs[0, 0].plot(t, [r["pitch_deg"] for r in ep], color=color, alpha=.23)
      axs[0, 1].plot(t, [r["world_x_velocity_mps"] for r in ep], color=color, alpha=.23)
      if i == 5:
        axs[1, 0].step(t, [r["left_sensor_contact"] for r in ep], where="post", color=color, label=f"{label} left")
        axs[1, 0].step(t, [r["right_sensor_contact"] for r in ep], where="post", color=color, linestyle="--", label=f"{label} right")
        axs[1, 1].plot(t, [np.linalg.norm(json.loads(r["actuator_effort"])) for r in ep], color=color, label=label)
        axs[2, 0].plot(t, [json.loads(r["joint_target_rad"])[14] for r in ep], color=color, label=label)
        axs[2, 1].plot(t, [json.loads(r["joint_pos_rad"])[14] for r in ep], color=color, label=label)
  for ax, title, unit in zip(axs.flat, ("Pitch", "World X velocity", "Foot contact, env 05", "Actuator effort norm, env 05", "Waist pitch target, env 05", "Waist pitch position, env 05"), ("deg", "m/s", "0/1", "N m", "rad", "rad")):
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(unit)
    ax.grid(alpha=.25)
  axs[0, 0].axhline(70, color="black", linestyle="--")
  axs[1, 0].legend(fontsize=8)
  axs[1, 1].legend()
  axs[2, 0].legend()
  axs[2, 1].legend()
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  results = {}
  for label in ("v1a", "flat"):
    results[label] = evaluate(label)
  schemas = {label: result[0] for label, result in results.items()}
  shared = ("action_dim", "target_names", "target_ids", "scale", "offset", "joint_names",
            "actuator_names", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
            "dof_armature", "physics_dt_s", "control_dt_s", "decimation")
  compatibility = {key: schemas["v1a"][key] == schemas["flat"][key] for key in shared}
  initial_max_abs = {key: float(np.nanmax(np.abs(results["v1a"][1][key] - results["flat"][1][key])))
                     for key in results["v1a"][1]}
  initial = {key: bool(np.allclose(results["v1a"][1][key], results["flat"][1][key],
                                   rtol=1e-6, atol=1e-7) if key == "body_subtreemass"
                      else np.array_equal(results["v1a"][1][key], results["flat"][1][key]))
             for key in results["v1a"][1]}
  if not all(compatibility.values()) or not all(initial.values()):
    differences = {key: {"shape": list(results["v1a"][1][key].shape),
                         "max_abs": initial_max_abs[key]}
                   for key, equal in initial.items() if not equal}
    raise RuntimeError(f"Control or initial-state mismatch: {compatibility}, {initial}, {differences}")
  rows = results["v1a"][2] + results["flat"][2]
  with (OUT / "trajectories.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  report = {"schemas": schemas, "shared_control_equal": compatibility,
            "initial_arrays_match": initial, "initial_max_abs_difference": initial_max_abs,
            "mass_comparison_tolerance": {"rtol": 1e-6, "atol": 1e-7},
            "episodes": {label: [{"env_id": i, "terminal_s": next((r["time_s"] for r in result[2] if r["env_id"] == i and r["phase"] == "terminal" and r["done"]), None)} for i in range(16)] for label, result in results.items()}}
  (OUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
  plot(rows, OUT / "comparison.png")
  print(json.dumps({"shared_control_equal": compatibility, "initial_arrays_match": initial,
                    "episodes": report["episodes"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
  main()
