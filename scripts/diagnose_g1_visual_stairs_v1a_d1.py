"""Read-only, first-episode failure attribution for the V1A pilot checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # noqa: F401
from src.tasks.velocity.mdp.stairs_metrics import geometry_tensors


TASK = "Unitree-G1-Visual-Stairs-V1A"
RUN = Path("logs/rsl_rl/g1_visual_stairs_v1a/2026-11-04_11-29-53_v1a_pilot_20261104")
OUT = Path("docs/assets/g1_visual_stairs_v1a/d1")
KEY_REWARDS = (
  "track_linear_velocity", "body_orientation_l2", "body_ang_vel",
  "foot_clearance", "foot_slip", "pose",
)


def euler_deg(quat: np.ndarray) -> tuple[float, float, float]:
  w, x, y, z = quat.T
  roll = np.degrees(np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)))
  pitch = np.degrees(np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)))
  yaw = np.degrees(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
  return roll, pitch, yaw


def run(args: argparse.Namespace) -> None:
  checkpoint = args.run_dir / "model_99.pt"
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  cfg = load_env_cfg(TASK)
  cfg.seed = 142
  cfg.scene.num_envs = 16
  cfg.observations["actor"].enable_corruption = False
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0), "yaw": (-0.03, 0.03),
  }
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK)
    assert runner_cls is not None
    runner = runner_cls(vec, asdict(agent_cfg), device="cuda:0")
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    feet = env.scene["feet_ground_contact"]
    foot_ids, foot_names = robot.find_sites(("left_foot", "right_foot"))
    if tuple(foot_names) != ("left_foot", "right_foot"):
      raise RuntimeError(f"Unexpected foot site order: {foot_names}")
    geometry = geometry_tensors(env)
    origins = env.scene.env_origins.detach().cpu().numpy()
    spawn = geometry["spawn_x"].detach().cpu().numpy()
    stair_start = geometry["staircase_start_x"].detach().cpu().numpy()
    step_depth = geometry["step_depth"].detach().cpu().numpy()
    metric_names = env.metrics_manager.active_terms
    reward_names = env.reward_manager.active_terms
    term_names = env.termination_manager.active_terms
    rows: list[list[dict]] = [[] for _ in range(16)]
    qposes: list[list[np.ndarray]] = [[] for _ in range(16)]
    completed = np.zeros(16, dtype=bool)
    step_number = 0

    def sample() -> None:
      # Called after reward/metric calculation and before auto-reset. This is
      # the same one-physics-substep-lagged state used by termination/rewards.
      pos = robot.data.root_link_pos_w.detach().cpu().numpy()
      quat = robot.data.root_link_quat_w.detach().cpu().numpy()
      vel = robot.data.root_link_lin_vel_b.detach().cpu().numpy()
      world_vel = robot.data.root_link_lin_vel_w.detach().cpu().numpy()
      sites = robot.data.site_pos_w[:, foot_ids].detach().cpu().numpy()
      found = feet.data.found[:, :2].detach().cpu().numpy()
      force = feet.data.force[:, :2].detach().cpu().numpy()
      metrics = env.metrics_manager._step_values.detach().cpu().numpy()
      rewards = env.reward_manager._step_reward.detach().cpu().numpy()
      qpos = env.sim.data.qpos.detach().cpu().numpy()
      done = env.reset_buf.detach().cpu().numpy()
      reasons = {name: env.termination_manager.get_term(name).detach().cpu().numpy()
                 for name in term_names}
      roll, pitch, yaw = euler_deg(quat)
      for i in range(16):
        if completed[i]:
          continue
        root_patch = float(pos[i, 0] - origins[i, 0] + spawn[i])
        foot_patch = sites[i, :, 0] - origins[i, 0] + spawn[i]
        row = {
          "env_id": i, "step": step_number, "time_s": round(step_number * env.step_dt, 6),
          "root_patch_x_m": root_patch,
          "root_to_first_riser_m": float(stair_start[i] - root_patch),
          "front_foot_to_first_riser_m": float(stair_start[i] - foot_patch.max()),
          "left_foot_patch_x_m": float(foot_patch[0]),
          "right_foot_patch_x_m": float(foot_patch[1]),
          "root_height_m": float(pos[i, 2] - origins[i, 2]),
          "roll_deg": float(roll[i]), "pitch_deg": float(pitch[i]),
          "yaw_deg": float(yaw[i]), "forward_velocity_mps": float(vel[i, 0]),
          "world_x_velocity_mps": float(world_vel[i, 0]),
          "left_contact": int(found[i, 0] > 0),
          "right_contact": int(found[i, 1] > 0),
          "left_contact_force_n": float(np.linalg.norm(force[i, 0])),
          "right_contact_force_n": float(np.linalg.norm(force[i, 1])),
          "left_foot_height_m": float(sites[i, 0, 2] - origins[i, 2]),
          "right_foot_height_m": float(sites[i, 1, 2] - origins[i, 2]),
          "done": int(done[i]),
        }
        row.update({"metric/" + name: float(metrics[i, j]) for j, name in enumerate(metric_names)})
        row.update({"reward/" + name: float(rewards[i, j]) for j, name in enumerate(reward_names)})
        row.update({"termination/" + name: int(reasons[name][i]) for name in term_names})
        rows[i].append(row)
        qposes[i].append(qpos[i].copy())
        completed[i] = bool(done[i])

    original_compute = env.metrics_manager.compute

    def captured_compute() -> None:
      original_compute()
      sample()

    env.metrics_manager.compute = captured_compute
    command = env.command_manager.get_term("twist")
    obs = vec.get_observations()
    initial_qpos = env.sim.data.qpos.detach().cpu().numpy().copy()
    for i in range(16):
      qposes[i].append(initial_qpos[i])
    for step_number in range(1, args.max_steps + 1):
      command.vel_command_b[:] = torch.tensor((0.4, 0.0, 0.0), device="cuda:0")
      command.time_left.fill_(1e9)
      with torch.no_grad():
        obs, reward, done, _ = vec.step(policy(obs))
      if not torch.isfinite(reward).all() or not torch.isfinite(obs["depth"]).all():
        raise RuntimeError("Non-finite reward or depth")
      if completed.all():
        break
    if not completed.all():
      raise RuntimeError(f"Only {completed.sum()}/16 first episodes terminated")

    args.output.mkdir(parents=True, exist_ok=True)
    flat = [row for episode in rows for row in episode]
    with (args.output / "trajectories.csv").open("w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
      writer.writeheader()
      writer.writerows(flat)
    summary = []
    for i, episode in enumerate(rows):
      last = episode[-1]
      previous = episode[-2] if len(episode) > 1 else last
      cumulative = {name: float(sum(r["reward/" + name] * env.step_dt for r in episode))
                    for name in reward_names}
      summary.append({
        "env_id": i, "steps": len(episode), "time_s": last["time_s"],
        "root_patch_x_m": last["root_patch_x_m"],
        "root_height_m": last["root_height_m"],
        "root_to_first_riser_m": last["root_to_first_riser_m"],
        "front_foot_to_first_riser_m": last["front_foot_to_first_riser_m"],
        "min_front_foot_to_first_riser_m": min(r["front_foot_to_first_riser_m"] for r in episode),
        "roll_deg": last["roll_deg"], "pitch_deg": last["pitch_deg"],
        "forward_velocity_mps": last["forward_velocity_mps"],
        "world_x_velocity_mps": last["world_x_velocity_mps"],
        "previous_forward_velocity_mps": previous["forward_velocity_mps"],
        "left_contact": last["left_contact"], "right_contact": last["right_contact"],
        "left_contact_force_n": last["left_contact_force_n"],
        "right_contact_force_n": last["right_contact_force_n"],
        "left_foot_height_m": last["left_foot_height_m"],
        "right_foot_height_m": last["right_foot_height_m"],
        "max_physical_stair_index": max(r.get("metric/stairs/physical_stair_index", 0) for r in episode),
        "termination": {name: last["termination/" + name] for name in term_names},
        "reward_integrals": cumulative,
        "terminal_reward_rates": {name: last["reward/" + name] for name in reward_names},
      })
    (args.output / "summary.json").write_text(json.dumps({
      "task": TASK, "checkpoint": str(checkpoint), "seed": 142,
      "command_mps": [0.4, 0, 0], "step_dt_s": env.step_dt,
      "first_riser_patch_x_m": stair_start.tolist(), "step_depth_m": step_depth.tolist(),
      "kinematics_note": "Derived pose, velocity, sites and contacts are one physics substep behind qpos, as in env termination/reward computation.",
      "episodes": summary,
    }, indent=2) + "\n")

    # Select the median-duration episode for a representative, complete failure.
    representative = sorted(range(16), key=lambda i: len(rows[i]))[8]
    render_video(env.sim.mj_model, qposes[representative], origins[representative],
                 args.output / f"failure_env_{representative:02d}.mp4", env.step_dt)
    plot_diagnostics(rows, representative, args.output / "diagnostics.png")
    print(json.dumps({"output": str(args.output), "representative_env": representative,
                      "episodes": summary}, indent=2))
  finally:
    env.close()


def render_video(model: mujoco.MjModel, qposes: list[np.ndarray], origin: np.ndarray,
                 path: Path, step_dt: float) -> None:
  data = mujoco.MjData(model)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = (float(origin[0]) + 1.3, float(origin[1]), 0.65)
  camera.distance = 3.6
  camera.azimuth = 135
  camera.elevation = -18
  fps = round(1 / step_dt)
  with mujoco.Renderer(model, height=480, width=800) as renderer, imageio.get_writer(
      path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
    for qpos in qposes:
      data.qpos[:] = qpos
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, camera=camera)
      frame = renderer.render().copy()
      if not frame.any():
        raise RuntimeError("Rendered video frame is blank")
      writer.append_data(frame)


def plot_diagnostics(rows: list[list[dict]], representative: int, path: Path) -> None:
  selected = rows[representative]
  t = np.array([r["time_s"] for r in selected])
  fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
  for episode in rows:
    axes[0, 0].plot([r["time_s"] for r in episode],
                    [r["front_foot_to_first_riser_m"] for r in episode], alpha=0.25)
  axes[0, 0].axhline(0, color="black", linestyle="--")
  axes[0, 0].set_title("Front foot to first riser (all 16)")
  axes[0, 0].set_ylabel("m")
  for key, label in (("roll_deg", "roll"), ("pitch_deg", "pitch")):
    axes[0, 1].plot(t, [r[key] for r in selected], label=label)
  axes[0, 1].axhline(70, color="black", linestyle="--")
  axes[0, 1].axhline(-70, color="black", linestyle="--")
  axes[0, 1].set_title(f"Body orientation, env {representative}")
  axes[0, 1].set_ylabel("degrees")
  axes[0, 1].legend()
  axes[1, 0].plot(t, [r["forward_velocity_mps"] for r in selected], label="body +X")
  axes[1, 0].plot(t, [r["world_x_velocity_mps"] for r in selected], label="world +X")
  axes[1, 0].axhline(0.4, color="black", linestyle="--")
  axes[1, 0].set_title("Forward velocity")
  axes[1, 0].set_ylabel("m/s")
  axes[1, 0].legend()
  for side in ("left", "right"):
    axes[1, 1].plot(t, [r[f"{side}_contact_force_n"] for r in selected], label=side)
  axes[1, 1].set_title("Foot terrain contact-force magnitude")
  axes[1, 1].set_ylabel("N")
  axes[1, 1].legend()
  for name in KEY_REWARDS:
    key = "reward/" + name
    if key in selected[0]:
      axes[2, 0].plot(t, [r[key] for r in selected], label=name)
  axes[2, 0].set_title("Weighted reward rates")
  axes[2, 0].text(0.02, 0.04, "terminal penalty: -200/s (-4 per step)",
                  transform=axes[2, 0].transAxes, fontsize=8)
  axes[2, 0].legend(fontsize=7)
  axes[2, 1].step(t, [r["left_contact"] for r in selected], where="post", label="left")
  axes[2, 1].step(t, [r["right_contact"] for r in selected], where="post", label="right")
  axes[2, 1].set_title("Foot terrain contact")
  axes[2, 1].legend()
  for ax in axes.flat:
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-dir", type=Path, default=RUN)
  parser.add_argument("--output", type=Path, default=OUT)
  parser.add_argument("--max-steps", type=int, default=1000)
  run(parser.parse_args())


if __name__ == "__main__":
  main()
