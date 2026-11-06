"""Read-only A/B replay of the V1A first-action command observation timing."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from diagnose_g1_visual_stairs_v1a_d2 import CHECKPOINTS, TASK, array, make_cfg, pitch_deg


OUT = Path("docs/assets/g1_visual_stairs_v1a/d3")
COMMAND = (0.4, 0.0, 0.0)
LABELS = ("A_original", "B_command_first")


def command_slice(env: ManagerBasedRlEnv) -> slice:
  names = env.observation_manager.active_terms["actor"]
  dims = env.observation_manager.group_obs_term_dim["actor"]
  start = 0
  for name, dim in zip(names, dims, strict=True):
    width = int(np.prod(dim))
    if name == "command":
      if width != 3:
        raise RuntimeError(f"Unexpected command observation width: {dim}")
      return slice(start, start + width)
    start += width
  raise RuntimeError("Actor has no command observation term")


def replay(label: str) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
  cfg = make_cfg("v1a")
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    rl_cfg = load_rl_cfg(TASK)
    vec = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK)
    assert runner_cls is not None
    runner = runner_cls(vec, asdict(rl_cfg), device="cuda:0")
    checkpoint = CHECKPOINTS["v1a"]
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    feet = env.scene["feet_ground_contact"]
    action = env.action_manager.get_term("joint_pos")
    foot_ids, foot_names = robot.find_sites(("left_foot", "right_foot"))
    if tuple(foot_names) != ("left_foot", "right_foot"):
      raise RuntimeError(f"Unexpected foot site order: {foot_names}")
    command = env.command_manager.get_term("twist")
    cmd_slice = command_slice(env)
    origins = array(env.scene.env_origins)
    initial = {
      "qpos": array(env.sim.data.qpos), "qvel": array(env.sim.data.qvel),
      "origins": origins, "reset_command": array(command.vel_command_b),
      "default_joint_pos": array(robot.data.default_joint_pos),
      "encoder_bias": array(robot.data.encoder_bias),
    }
    for field in ("geom_friction", "body_ipos", "body_subtreemass"):
      initial[field] = array(getattr(env.sim.model, field))
    rows: list[dict] = []
    active = np.ones(env.num_envs, dtype=bool)
    policy_input_command: np.ndarray | None = None
    policy_input_actor: np.ndarray | None = None
    step_number = 0

    def capture() -> None:
      pos = array(robot.data.root_link_pos_w)
      quat = array(robot.data.root_link_quat_w)
      vx = array(robot.data.root_link_lin_vel_w)[:, 0]
      site = array(robot.data.site_pos_w[:, foot_ids])
      found = array(feet.data.found[:, :2])
      force = array(feet.data.force[:, :2])
      raw = array(action.raw_action)
      target = array(robot.data.joint_pos_target)
      joint_pos = array(robot.data.joint_pos)
      ctrl = array(env.sim.data.ctrl)
      effort = array(env.sim.data.qfrc_actuator)
      done = array(env.reset_buf)
      fallen = array(env.termination_manager.get_term("fell_over"))
      pitch = pitch_deg(quat)
      for i in range(env.num_envs):
        if not active[i]:
          continue
        rows.append({
          "arm": label, "env_id": i, "step": step_number,
          "time_s": round(step_number * env.step_dt, 6),
          "pitch_deg": float(pitch[i]),
          "world_x_velocity_mps": float(vx[i]),
          "root_height_m": float(pos[i, 2] - origins[i, 2]),
          "left_foot_site_z_m": float(site[i, 0, 2] - origins[i, 2]),
          "right_foot_site_z_m": float(site[i, 1, 2] - origins[i, 2]),
          "left_contact": int(found[i, 0] > 0),
          "right_contact": int(found[i, 1] > 0),
          "left_contact_force_n": float(np.linalg.norm(force[i, 0])),
          "right_contact_force_n": float(np.linalg.norm(force[i, 1])),
          "policy_input_command_mps": json.dumps(policy_input_command[i].tolist()),
          "manager_command_mps": json.dumps(array(command.vel_command_b)[i].tolist()),
          "raw_action": json.dumps(raw[i].tolist()),
          "joint_target_rad": json.dumps(target[i].tolist()),
          "joint_pos_rad": json.dumps(joint_pos[i].tolist()),
          "sim_ctrl": json.dumps(ctrl[i].tolist()),
          "actuator_effort": json.dumps(effort[i].tolist()),
          "done": int(done[i]), "fell_over": int(fallen[i]),
        })
        if done[i]:
          active[i] = False

    original_metrics = env.metrics_manager.compute

    def capture_before_reset() -> None:
      original_metrics()
      capture()

    env.metrics_manager.compute = capture_before_reset
    obs = vec.get_observations()
    if label == "B_command_first":
      command.vel_command_b[:] = torch.tensor(COMMAND, device="cuda:0")
      command.time_left.fill_(1e9)
      # The actor group has one-step history, so a plain compute_group() call
      # returns the reset sample. Replace only its command slice for this one
      # inference; subsequent observations come from normal env.step().
      obs["actor"][:, cmd_slice] = command.vel_command_b
      if not torch.allclose(obs["actor"][:, cmd_slice], command.vel_command_b):
        raise RuntimeError("B actor command does not match the fixed manager command")
    initial["policy_actor_obs"] = array(obs["actor"])
    initial["policy_depth_obs"] = array(obs["depth"])
    for step_number in range(1, 16):
      command.vel_command_b[:] = torch.tensor(COMMAND, device="cuda:0")
      command.time_left.fill_(1e9)
      policy_input_actor = array(obs["actor"])
      policy_input_command = policy_input_actor[:, cmd_slice]
      with torch.no_grad():
        raw_action = policy(obs)
        obs, reward, _, _ = vec.step(raw_action)
      if not torch.isfinite(raw_action).all() or not torch.isfinite(reward).all():
        raise RuntimeError(f"Non-finite action/reward in {label}")
      if not active.any():
        break
    info = {
      "checkpoint": str(checkpoint),
      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      "actor_command_slice": [cmd_slice.start, cmd_slice.stop],
      "joint_names": action.target_names,
      "step_dt_s": env.step_dt,
      "physical_dt_s": env.physics_dt,
      "first_input_command_mps": initial["policy_actor_obs"][:, cmd_slice].tolist(),
      "reset_command_mps": initial["reset_command"].tolist(),
    }
    return info, initial, rows
  finally:
    env.close()


def plot(rows: list[dict], joint_diff: list[dict], path: Path) -> None:
  fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
  for label, color in ((LABELS[0], "#bc3d33"), (LABELS[1], "#087e82")):
    for i in range(16):
      ep = [r for r in rows if r["arm"] == label and r["env_id"] == i]
      t = [r["time_s"] for r in ep]
      axes[0, 0].plot(t, [r["pitch_deg"] for r in ep], color=color, alpha=0.28)
      axes[0, 1].plot(t, [r["world_x_velocity_mps"] for r in ep], color=color, alpha=0.28)
      if i == 5:
        axes[1, 0].step(t, [r["left_contact"] for r in ep], where="post", color=color, label=f"{label} left")
        axes[1, 0].step(t, [r["right_contact"] for r in ep], where="post", color=color, linestyle="--", label=f"{label} right")
        axes[1, 1].plot(t, [json.loads(r["joint_target_rad"])[14] for r in ep], color=color, label=label)
  axes[0, 0].axhline(70, color="black", linestyle="--")
  axes[0, 0].set_title("Pitch, all 16")
  axes[0, 1].set_title("World X velocity, all 16")
  axes[1, 0].set_title("Foot contacts, env 05")
  axes[1, 1].set_title("Waist pitch target, env 05")
  axes[1, 0].legend(fontsize=7)
  axes[1, 1].legend(fontsize=7)
  names = [r["joint_name"] for r in joint_diff]
  delta = [r["mean_target_delta_B_minus_A_rad"] for r in joint_diff]
  axes[2, 0].bar(range(len(names)), delta, color="#4e7377")
  axes[2, 0].set_xticks(range(len(names)))
  axes[2, 0].set_xticklabels(names, rotation=90, fontsize=5)
  axes[2, 0].set_title("First-step target B - A, all 29 joints")
  key = [r for r in joint_diff if any(token in r["joint_name"] for token in ("waist_pitch", "hip_pitch", "knee"))]
  axes[2, 1].barh([r["joint_name"] for r in key], [r["mean_target_delta_B_minus_A_rad"] for r in key], color="#c26745")
  axes[2, 1].set_title("Waist, hip and knee target difference")
  axes[2, 1].tick_params(axis="y", labelsize=8)
  for ax in axes[:2].flat:
    ax.set_xlabel("time (s)")
    ax.grid(alpha=.25)
  axes[2, 0].grid(axis="y", alpha=.25)
  axes[2, 1].grid(axis="x", alpha=.25)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main() -> None:
  results = {label: replay(label) for label in LABELS}
  a_info, a_initial, a_rows = results[LABELS[0]]
  b_info, b_initial, b_rows = results[LABELS[1]]
  if a_info["checkpoint_sha256"] != b_info["checkpoint_sha256"]:
    raise RuntimeError("A/B checkpoint hashes differ")
  initial_match = {}
  initial_max_abs = {}
  for key in a_initial:
    if key in ("policy_actor_obs", "policy_depth_obs"):
      continue
    difference = np.abs(a_initial[key] - b_initial[key])
    initial_max_abs[key] = float(np.nanmax(difference))
    initial_match[key] = bool(np.allclose(a_initial[key], b_initial[key], rtol=1e-6, atol=1e-7)
                              if key == "body_subtreemass" else np.array_equal(a_initial[key], b_initial[key]))
  cmd_start, cmd_stop = a_info["actor_command_slice"]
  a_actor = a_initial["policy_actor_obs"]
  b_actor = b_initial["policy_actor_obs"]
  noncommand = np.ones(a_actor.shape[1], dtype=bool)
  noncommand[cmd_start:cmd_stop] = False
  noncommand_match = bool(np.array_equal(a_actor[:, noncommand], b_actor[:, noncommand]))
  depth_match = bool(np.array_equal(a_initial["policy_depth_obs"], b_initial["policy_depth_obs"]))
  a_cmd = a_actor[:, cmd_start:cmd_stop]
  b_cmd = b_actor[:, cmd_start:cmd_stop]
  b_command_correct = bool(np.allclose(b_cmd, np.asarray(COMMAND), atol=1e-6, rtol=0))
  if not all(initial_match.values()) or not noncommand_match or not depth_match or not b_command_correct:
    raise RuntimeError(f"A/B mismatch beyond command: {initial_match}, actor={noncommand_match}, depth={depth_match}, B_command={b_command_correct}, slice={cmd_start}:{cmd_stop}, A0={a_cmd[0].tolist()}, B0={b_cmd[0].tolist()}")
  rows = a_rows + b_rows
  first = {label: {r["env_id"]: r for r in results[label][2] if r["step"] == 1} for label in LABELS}
  if any(len(first[label]) != 16 for label in LABELS):
    raise RuntimeError("Missing first-step records")
  first_targets = []
  for i in range(16):
    a = first[LABELS[0]][i]
    b = first[LABELS[1]][i]
    a_raw, b_raw = np.asarray(json.loads(a["raw_action"])), np.asarray(json.loads(b["raw_action"]))
    a_target, b_target = np.asarray(json.loads(a["joint_target_rad"])), np.asarray(json.loads(b["joint_target_rad"]))
    for j, name in enumerate(a_info["joint_names"]):
      first_targets.append({"env_id": i, "joint_index": j, "joint_name": name,
                            "A_raw_action": float(a_raw[j]), "B_raw_action": float(b_raw[j]),
                            "A_target_rad": float(a_target[j]), "B_target_rad": float(b_target[j]),
                            "delta_B_minus_A_rad": float(b_target[j] - a_target[j])})
  joint_diff = []
  for j, name in enumerate(a_info["joint_names"]):
    subset = [r for r in first_targets if r["joint_index"] == j]
    joint_diff.append({"joint_index": j, "joint_name": name,
                       "A_mean_target_rad": float(np.mean([r["A_target_rad"] for r in subset])),
                       "B_mean_target_rad": float(np.mean([r["B_target_rad"] for r in subset])),
                       "mean_target_delta_B_minus_A_rad": float(np.mean([r["delta_B_minus_A_rad"] for r in subset])),
                       "max_abs_paired_delta_rad": float(max(abs(r["delta_B_minus_A_rad"]) for r in subset))})
  episodes = {}
  for label in LABELS:
    ep_rows = results[label][2]
    episodes[label] = []
    for i in range(16):
      ep = [r for r in ep_rows if r["env_id"] == i]
      last = ep[-1]
      episodes[label].append({"env_id": i, "last_observed_time_s": last["time_s"],
                              "fall_time_s": last["time_s"] if last["fell_over"] else None,
                              "survived_0_30_s": bool(len(ep) == 15 and not last["done"]),
                              "last_pitch_deg": last["pitch_deg"],
                              "last_world_x_velocity_mps": last["world_x_velocity_mps"]})
  OUT.mkdir(parents=True, exist_ok=True)
  with (OUT / "trajectories.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  with (OUT / "first_step_targets.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(first_targets[0]))
    writer.writeheader()
    writer.writerows(first_targets)
  summary = {
    "task": TASK, "checkpoint": a_info["checkpoint"], "checkpoint_sha256": a_info["checkpoint_sha256"],
    "seed": 142, "num_envs": 16, "command_after_first_observation_mps": COMMAND,
    "step_dt_s": a_info["step_dt_s"], "physics_dt_s": a_info["physical_dt_s"],
    "initial_match": initial_match, "initial_max_abs_difference": initial_max_abs,
    "noncommand_actor_observation_exact_match": noncommand_match,
    "initial_depth_exact_match": depth_match,
    "A_first_input_command_mps": a_cmd.tolist(), "B_first_input_command_mps": b_cmd.tolist(),
    "B_first_input_command_correct": b_command_correct,
    "joint_names": a_info["joint_names"], "first_step_joint_summary": joint_diff,
    "episodes": episodes,
    "A_survival_0_30_count": sum(e["survived_0_30_s"] for e in episodes[LABELS[0]]),
    "B_survival_0_30_count": sum(e["survived_0_30_s"] for e in episodes[LABELS[1]]),
    "kinematic_note": "Pre-reset derived kinematics and contact signals lag qpos by one physics substep, as in termination/reward computation.",
  }
  (OUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
  plot(rows, joint_diff, OUT / "comparison.png")
  print(json.dumps({"initial_match": initial_match, "noncommand_actor_match": noncommand_match,
                    "depth_match": depth_match, "A_survival_0_30_count": summary["A_survival_0_30_count"],
                    "B_survival_0_30_count": summary["B_survival_0_30_count"],
                    "fall_times": {label: [e["fall_time_s"] for e in episodes[label]] for label in LABELS}}, indent=2))


if __name__ == "__main__":
  main()
