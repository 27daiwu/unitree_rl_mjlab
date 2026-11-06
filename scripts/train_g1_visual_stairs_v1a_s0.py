"""Flat-teacher behavior cloning for a short V1A approach-stability phase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # noqa: F401


TASK = "Unitree-G1-Visual-Stairs-V1A"
FLAT = "Unitree-G1-Flat"
V1A_SOURCE = Path("logs/rsl_rl/g1_visual_stairs_v1a/2026-11-04_11-29-53_v1a_pilot_20261104/model_99.pt")
FLAT_SOURCE = Path("logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt")
D2_TRAJECTORIES = Path("docs/assets/g1_visual_stairs_v1a/d2/trajectories.csv")
OUT = Path("logs/experiments/g1_visual_stairs_v1a_s0_20261104")
COMMAND = (0.4, 0.0, 0.0)
TRAIN_ENVS = 96
TRAIN_STEPS = 15
EPOCHS = 10
MINIBATCH = 512
LR = 3e-4


def arr(x: torch.Tensor) -> np.ndarray:
  return x.detach().cpu().numpy().copy()


def make_env(seed: int, n: int, device: str = "cuda:0"):
  cfg = load_env_cfg(TASK)
  cfg.seed = seed
  cfg.scene.num_envs = n
  cfg.observations["actor"].enable_corruption = False
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0), "yaw": (-0.03, 0.03),
  }
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  runner_cls = load_runner_cls(TASK)
  assert runner_cls is not None
  v1_runner = runner_cls(RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions),
                         asdict(load_rl_cfg(TASK)), device=device)
  return env, v1_runner


def make_runner(env: ManagerBasedRlEnv, task: str, device: str):
  runner_cls = load_runner_cls(task)
  assert runner_cls is not None
  rl = load_rl_cfg(task)
  return runner_cls(RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions), asdict(rl), device=device)


def load_actor(runner, path: Path, device: str):
  runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def set_command(env, obs: dict[str, torch.Tensor]) -> None:
  command = env.command_manager.get_term("twist")
  command.vel_command_b[:] = torch.tensor(COMMAND, device=env.device)
  command.time_left.fill_(1e9)
  obs["actor"][:, 6:9] = command.vel_command_b


def collect_dataset(device: str) -> dict[str, torch.Tensor]:
  env, v1_runner = make_env(812, TRAIN_ENVS, device)
  try:
    flat_runner = make_runner(env, FLAT, device)
    teacher = load_actor(flat_runner, FLAT_SOURCE, device)
    # Native Flat policy consumes only proprioception, while V1A retains its depth input.
    obs = v1_runner.env.get_observations()
    actor_obs, depth_obs, actions = [], [], []
    for _ in range(TRAIN_STEPS):
      set_command(env, obs)
      actor_obs.append(obs["actor"].detach().clone())
      depth_obs.append(obs["depth"].detach().clone())
      with torch.no_grad():
        action = teacher({"actor": obs["actor"]})
      actions.append(action.detach().clone())
      obs, _, done, _ = v1_runner.env.step(action)
      if bool(done.any()):
        raise RuntimeError("Flat teacher failed during the 0.30 s training-data rollout")
    dataset = {
      "actor": torch.cat(actor_obs),
      "depth": torch.cat(depth_obs),
      "action": torch.cat(actions),
    }
    if not all(torch.isfinite(t).all() for t in dataset.values()):
      raise RuntimeError("Non-finite supervised data")
    return dataset
  finally:
    env.close()


def train_bc(dataset: dict[str, torch.Tensor], device: str) -> tuple[dict, dict]:
  env, runner = make_env(917, 1, device)
  try:
    model = runner.alg.get_policy()
    runner.load(str(V1A_SOURCE), load_cfg={"actor": True}, strict=True, map_location=device)
    model = runner.alg.get_policy()
    model.train()
    # Keep the checkpoint's proprioceptive normalization fixed during cloning.
    model.obs_normalizer.eval()
    tensors = {k: v.to(device) for k, v in dataset.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n = tensors["action"].shape[0]
    rng = torch.Generator(device=device).manual_seed(20261104)
    history = []
    for epoch in range(EPOCHS):
      order = torch.randperm(n, generator=rng, device=device)
      losses = []
      for ids in order.split(MINIBATCH):
        batch = {"actor": tensors["actor"][ids], "depth": tensors["depth"][ids]}
        pred = model(batch)
        loss = torch.nn.functional.mse_loss(pred, tensors["action"][ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
      history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
    model.eval()
    checkpoint = torch.load(V1A_SOURCE, map_location="cpu", weights_only=False)
    checkpoint["actor_state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    checkpoint["infos"] = {"s0_behavior_cloned_from": str(FLAT_SOURCE), "source_v1a": str(V1A_SOURCE),
                           "updates": EPOCHS, "training_samples": int(n)}
    bc_path = OUT / "v1a_s0_bc_model.pt"
    torch.save(checkpoint, bc_path)
    return {"epochs": history, "samples": int(n), "batch_size": MINIBATCH, "learning_rate": LR,
            "model": str(bc_path)}, model.state_dict()
  finally:
    env.close()


def evaluate(policy_name: str, checkpoint: Path, seed: int, device: str, max_steps: int = 50) -> list[dict]:
  env, runner = make_env(seed, 16, device)
  try:
    policy_runner = runner if policy_name != "flat_teacher" else make_runner(env, FLAT, device)
    policy = load_actor(policy_runner, checkpoint, device)
    obs = runner.env.get_observations()
    set_command(env, obs)
    feet = env.scene["feet_ground_contact"]
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_sites(("left_foot", "right_foot"))
    origins = arr(env.scene.env_origins)
    initial_x = arr(robot.data.root_link_pos_w)[:, 0].copy()
    done_rows: list[dict] = []
    active = np.ones(16, dtype=bool)
    step_id = 0

    def capture_terminal_step():
      pos = arr(robot.data.root_link_pos_w)
      quat = arr(robot.data.root_link_quat_w)
      # MuJoCo quaternion convention is w,x,y,z.
      w, x, y, z = quat.T
      pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
      roll = np.degrees(np.arcsin(np.clip(2 * (w * x + y * z), -1, 1)))
      tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
      vx = arr(robot.data.root_link_lin_vel_w)[:, 0]
      found = arr(feet.data.found[:, :2])
      foot_pos = arr(robot.data.site_pos_w[:, foot_ids])
      stair_distance = origins[:, 0] + 2.0 - foot_pos[:, :, 0].max(axis=1)
      target = arr(robot.data.joint_pos_target)
      joint_names = robot.joint_names
      done = arr(env.reset_buf).astype(bool)
      for i in range(16):
        if not active[i]:
          continue
        done_rows.append({"policy": policy_name, "env_id": i, "step": step_id,
                          "time_s": round(step_id * env.step_dt, 6),
                          "pitch_deg": float(pitch[i]),
                          "roll_deg": float(roll[i]), "tilt_deg": float(tilt[i]),
                          "root_x_m": float(pos[i, 0] - initial_x[i]),
                          "stair_distance_m": float(stair_distance[i]),
                          "world_x_velocity_mps": float(vx[i]),
                          "left_contact": int(found[i, 0] > 0), "right_contact": int(found[i, 1] > 0),
                          "raw_action": json.dumps(arr(env.action_manager.get_term("joint_pos").raw_action)[i].tolist()),
                          "joint_target_rad": json.dumps(target[i].tolist()),
                          "done": int(done[i]),
                          "fell_over": int(bool(env.reset_terminated[i]))})
        if done[i]:
          active[i] = False

    original_metrics = env.metrics_manager.compute
    def metric_hook():
      original_metrics()
      capture_terminal_step()
    env.metrics_manager.compute = metric_hook
    for step_id in range(1, max_steps + 1):
      set_command(env, obs)
      with torch.no_grad():
        action = policy(obs)
        obs, _, _, _ = runner.env.step(action)
      if not active.any():
        break
    return done_rows
  finally:
    env.close()


def save_csv(path: Path, rows: list[dict]) -> None:
  fields = list(rows[0])
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
  result = {}
  for policy in sorted({r["policy"] for r in rows}):
    subset = [r for r in rows if r["policy"] == policy]
    episodes = []
    for env_id in range(16):
      ep = [r for r in subset if r["env_id"] == env_id]
      at_030 = [r for r in ep if r["time_s"] >= 0.30 - 1e-6]
      survived = bool(at_030) and not any(r["fell_over"] for r in ep if r["time_s"] <= .30 + 1e-6)
      last = at_030[-1] if survived else ep[-1]
      episodes.append({"env_id": env_id, "survived_030s": survived,
                       "x_progress_030s_m": last["root_x_m"] if survived else None,
                       "survival_1s": len(ep) >= 50 and not any(r["fell_over"] for r in ep),
                       "duration_s": ep[-1]["time_s"],
                       "x_progress_end_m": ep[-1]["root_x_m"],
                       "pitch_rms_deg": float(np.sqrt(np.mean([r["pitch_deg"] ** 2 for r in ep]))),
                       "contact_fraction": float(np.mean([r["left_contact"] + r["right_contact"] for r in ep]))})
    result[policy] = {
      "episodes": episodes,
      "survivors_030s": sum(e["survived_030s"] for e in episodes),
      "positive_x_030s": sum(e["survived_030s"] and e["x_progress_030s_m"] > 0 for e in episodes),
      "survivors_1s": sum(e["survival_1s"] for e in episodes),
      "mean_x_progress_030s_m": float(np.mean([e["x_progress_030s_m"] for e in episodes if e["x_progress_030s_m"] is not None])) if any(e["survived_030s"] for e in episodes) else None,
      "mean_pitch_rms_deg": float(np.mean([e["pitch_rms_deg"] for e in episodes])),
      "mean_contact_count": float(np.mean([e["contact_fraction"] for e in episodes])),
    }
  return result


def plot(rows: list[dict], path: Path) -> None:
  fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
  colors = {"v1a_original": "#b6473e", "v1a_s0_bc": "#167a78", "flat_teacher": "#3168a8"}
  for policy, color in colors.items():
    data = [r for r in rows if r["policy"] == policy]
    for env_id in range(16):
      ep = [r for r in data if r["env_id"] == env_id]
      if not ep:
        continue
      alpha = .25
      t = [r["time_s"] for r in ep]
      axs[0, 0].plot(t, [r["pitch_deg"] for r in ep], color=color, alpha=alpha)
      axs[0, 1].plot(t, [r["root_x_m"] for r in ep], color=color, alpha=alpha)
      if env_id == 5:
        axs[1, 0].step(t, [r["left_contact"] for r in ep], where="post", color=color, label=f"{policy} L")
        axs[1, 0].step(t, [r["right_contact"] for r in ep], where="post", color=color, linestyle="--", label=f"{policy} R")
  axs[0, 0].set_title("Pitch, 16 held-out starts")
  axs[0, 1].set_title("World X progress")
  axs[1, 0].set_title("Foot contacts, env 05")
  axs[1, 1].axis("off")
  for ax in axs.flat[:3]:
    ax.set_xlabel("time (s)")
    ax.grid(alpha=.25)
  axs[1, 0].legend(fontsize=7, ncol=2)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)
  torch.manual_seed(20261104)
  np.random.seed(20261104)
  random.seed(20261104)
  src = torch.load(V1A_SOURCE, map_location="cpu", weights_only=False)
  actor = src["actor_state_dict"]
  std = actor["obs_normalizer._std"].flatten()
  zeros = torch.where(std == 0)[0].tolist()
  with D2_TRAJECTORIES.open(newline="") as f:
    d2 = [r for r in csv.DictReader(f) if r["policy"] == "v1a" and r["phase"] == "control" and int(r["control_step"]) == 1]
  first_actions = np.asarray([json.loads(r["raw_action"]) for r in d2], dtype=np.float32)
  first_targets = np.asarray([json.loads(r["joint_target_rad"]) for r in d2], dtype=np.float32)
  audit = {
    "v1a_checkpoint": str(V1A_SOURCE),
    "v1a_sha256": hashlib.sha256(V1A_SOURCE.read_bytes()).hexdigest(),
    "flat_checkpoint": str(FLAT_SOURCE),
    "flat_sha256": hashlib.sha256(FLAT_SOURCE.read_bytes()).hexdigest(),
    "actor_tensors_finite": all(torch.isfinite(v).all().item() for v in actor.values() if torch.is_tensor(v)),
    "actor_obs_normalizer_std_min": float(std.min()),
    "actor_obs_normalizer_zero_variance_indices": zeros,
    "pretraining_first_action_output_range_from_matched_d2_replay": {
      "samples": int(first_actions.shape[0]),
      "raw_action_min": float(first_actions.min()),
      "raw_action_max": float(first_actions.max()),
      "raw_action_abs_max": float(np.abs(first_actions).max()),
      "raw_action_mean_abs": float(np.abs(first_actions).mean()),
      "joint_target_abs_max_rad": float(np.abs(first_targets).max()),
      "joint_target_mean_abs_rad": float(np.abs(first_targets).mean()),
      "source_trajectories": str(D2_TRAJECTORIES),
    },
    "actor_parameter_shapes": {k: list(v.shape) for k, v in actor.items() if k.startswith(("mlp.", "cnns.")) and torch.is_tensor(v)},
    "strict_checkpoint_load": "validated during dataset collection and policy evaluation",
    "known_zero_variance_dims": "actor command slice [6:9], indices 7 and 8 are fixed lateral/yaw command channels",
  }
  if not audit["actor_tensors_finite"]:
    raise RuntimeError("V1A checkpoint contains non-finite actor state")
  (OUT / "pre_training_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
  dataset = collect_dataset(args.device)
  torch.save({k: v.cpu() for k, v in dataset.items()}, OUT / "flat_teacher_dataset.pt")
  train_info, _ = train_bc(dataset, args.device)
  rows = []
  rows += evaluate("v1a_original", V1A_SOURCE, 142, args.device)
  rows += evaluate("v1a_s0_bc", OUT / "v1a_s0_bc_model.pt", 142, args.device)
  rows += evaluate("flat_teacher", FLAT_SOURCE, 142, args.device)
  save_csv(OUT / "heldout_trajectories.csv", rows)
  summary = summarize(rows)
  gate = summary["v1a_s0_bc"]
  summary["acceptance"] = {
    "primary_pass": gate["survivors_030s"] == 16 and gate["positive_x_030s"] == 16,
    "criterion": "16/16 held-out starts survive 0.30 s and have positive world-X displacement",
    "continue_training_basis": gate["survivors_1s"],
  }
  result = {"audit": audit, "training": train_info, "comparison": summary}
  (OUT / "s0_result.json").write_text(json.dumps(result, indent=2) + "\n")
  plot(rows, OUT / "heldout_curves.png")
  print(json.dumps({"training": train_info, "comparison": summary}, indent=2))


if __name__ == "__main__":
  main()
