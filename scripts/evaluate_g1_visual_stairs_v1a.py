"""Evaluate the V1A pilot checkpoint and record an external stair-view video."""

from __future__ import annotations

import argparse
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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # noqa: F401


TASK = "Unitree-G1-Visual-Stairs-V1A"
CURVE_TAGS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Perf/total_fps",
  "Loss/value",
  "Loss/surrogate",
  "Episode_Metrics/stairs/physical_stair_index",
  "Episode_Metrics/stairs/reached_top_platform",
)


def export_curves(run_dir: Path) -> None:
  events = list(run_dir.glob("events.out.tfevents.*"))
  if not events:
    raise FileNotFoundError(f"No TensorBoard events in {run_dir}")
  accumulator = EventAccumulator(str(events[0]))
  accumulator.Reload()
  available = set(accumulator.Tags()["scalars"])
  curves = {
    tag: [{"iteration": event.step, "value": event.value} for event in accumulator.Scalars(tag)]
    for tag in CURVE_TAGS if tag in available
  }
  (run_dir / "training_curves.json").write_text(json.dumps(curves, indent=2) + "\n")
  fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
  for ax, tag in zip(axes.flat, (
    "Train/mean_reward", "Train/mean_episode_length",
    "Episode_Metrics/stairs/physical_stair_index", "Episode_Metrics/stairs/reached_top_platform",
  ), strict=True):
    values = curves.get(tag, [])
    if values:
      ax.plot([p["iteration"] for p in values], [p["value"] for p in values])
    ax.set_title(tag.split("/")[-1].replace("_", " "))
    ax.set_xlabel("PPO iteration")
    ax.grid(alpha=0.3)
  fig.savefig(run_dir / "training_curves.png", dpi=160)
  plt.close(fig)


def evaluate(run_dir: Path, episodes: int, max_steps: int, video_frames: int) -> dict:
  checkpoint = run_dir / "model_99.pt"
  if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)
  cfg = load_env_cfg(TASK)
  cfg.seed = 142
  cfg.scene.num_envs = episodes
  cfg.observations["actor"].enable_corruption = False
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0), "yaw": (-0.03, 0.03)
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
    command = env.command_manager.get_term("twist")
    obs = vec.get_observations()
    completed = torch.zeros(episodes, dtype=torch.bool, device="cuda:0")
    success = completed.clone()
    fallen = completed.clone()
    timed_out = completed.clone()
    max_stair = torch.zeros(episodes, device="cuda:0")
    max_x = torch.full((episodes,), -torch.inf, device="cuda:0")
    metric_names = env.metrics_manager.active_terms
    stair_idx = metric_names.index("stairs/physical_stair_index")
    top_idx = metric_names.index("stairs/reached_top_platform")
    x_idx = metric_names.index("stairs/root_x_progress")
    video_path = run_dir / "evaluation_stairs.mp4"
    model = env.sim.mj_model
    cpu_data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (
      float(env.scene.env_origins[0, 0]) + 1.8,
      float(env.scene.env_origins[0, 1]),
      0.65,
    )
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -18.0
    written_frames = 0
    with mujoco.Renderer(model, height=360, width=640) as renderer, imageio.get_writer(
      video_path, fps=50, codec="libx264", quality=8, macro_block_size=16
    ) as writer:
      for _ in range(max_steps):
        if written_frames < video_frames and not bool(completed[0]):
          cpu_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
          mujoco.mj_forward(model, cpu_data)
          renderer.update_scene(cpu_data, camera=camera)
          frame = renderer.render().copy()
          if not frame.any():
            raise RuntimeError("Evaluation video frame is blank")
          writer.append_data(frame)
          written_frames += 1
        command.vel_command_b[:] = torch.tensor((0.4, 0.0, 0.0), device="cuda:0")
        command.time_left.fill_(1e9)
        with torch.no_grad():
          obs, reward, done, extras = vec.step(policy(obs))
        if not torch.isfinite(reward).all() or not torch.isfinite(obs["depth"]).all():
          raise RuntimeError("Non-finite evaluation reward or depth")
        active = ~completed
        metrics = env.metrics_manager._step_values
        max_stair = torch.where(active, torch.maximum(max_stair, metrics[:, stair_idx]), max_stair)
        max_x = torch.where(active, torch.maximum(max_x, metrics[:, x_idx]), max_x)
        success |= active & (metrics[:, top_idx] > 0.5)
        newly_done = active & done.bool()
        fallen |= newly_done & env.reset_terminated
        timed_out |= newly_done & env.reset_time_outs
        completed |= newly_done
        if bool(completed.all()):
          break
    result = {
      "task": TASK,
      "checkpoint": str(checkpoint),
      "episodes": episodes,
      "completed_episodes": int(completed.sum()),
      "success_count": int(success.sum()),
      "success_rate": float(success.float().mean()),
      "fall_rate": float(fallen.float().mean()),
      "timeout_rate": float(timed_out.float().mean()),
      "mean_max_physical_stair_index": float(max_stair.mean()),
      "max_physical_stair_index": float(max_stair.max()),
      "mean_max_root_x_progress_m": float(max_x.mean()),
      "video_frames": written_frames,
      "video": str(video_path),
      "command_mps": [0.4, 0.0, 0.0],
      "seed": 142,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=16)
  parser.add_argument("--max-steps", type=int, default=1000)
  parser.add_argument("--video-frames", type=int, default=300)
  args = parser.parse_args()
  export_curves(args.run_dir)
  print(json.dumps(evaluate(args.run_dir, args.episodes, args.max_steps, args.video_frames), indent=2))


if __name__ == "__main__":
  main()
