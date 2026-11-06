"""Check visual-stairs observations, CNN inference, and a few environment steps.

This does not train a policy. Run after inspecting the static camera artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # noqa: F401
from src.tasks.velocity.config.g1.visual_stairs_v1a_env_cfg import raw_camera_depth


TASK_ID = "Unitree-G1-Visual-Stairs-V1A"


def check(device: str, num_envs: int, steps: int, artifact_dir: Path) -> dict[str, object]:
  cfg = load_env_cfg(TASK_ID)
  agent_cfg = load_rl_cfg(TASK_ID)
  agent_cfg.num_steps_per_env = 4
  agent_cfg.algorithm.num_learning_epochs = 1
  agent_cfg.algorithm.num_mini_batches = 1
  cfg.scene.num_envs = num_envs
  cfg.seed = 42
  assert "height_scan" not in cfg.observations["actor"].terms
  assert agent_cfg.obs_groups["actor"] == ("actor", "depth")

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    assert vec.num_actions == 29
    obs = vec.get_observations()
    assert obs["depth"].shape == (num_envs, 1, 90, 160)
    assert torch.isfinite(obs["depth"]).all()
    sensor_depth = env.scene["d435_depth"].data.depth
    assert sensor_depth is not None
    torch.testing.assert_close(obs["depth"], raw_camera_depth(env, "d435_depth"))
    valid = obs["depth"][obs["depth"] > 0]
    assert valid.numel() > 0, "Camera has no visible geometry"
    initial_valid_fraction = float((obs["depth"] > 0).float().mean())
    rgb = env.scene["d435_depth"].data.rgb
    assert rgb is not None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    depth_image = obs["depth"][0, 0].cpu().numpy()
    Image.fromarray(rgb[0].cpu().numpy()).save(artifact_dir / "runtime_rgb.png")
    np.save(artifact_dir / "runtime_depth_m.npy", depth_image)
    depth_vis = np.zeros_like(depth_image, dtype=np.uint8)
    hit = depth_image > 0
    depth_vis[hit] = np.clip(
      255 * (1 - (depth_image[hit] - 0.5) / 4.5), 0, 255
    ).astype(np.uint8)
    Image.fromarray(depth_vis).save(artifact_dir / "runtime_depth_visual.png")
    runner_cls = load_runner_cls(TASK_ID)
    assert runner_cls is not None
    runner = runner_cls(vec, asdict(agent_cfg), device=device)
    policy = runner.get_inference_policy(device=device)
    for _ in range(steps):
      with torch.no_grad():
        actions = policy(obs)
        obs, reward, dones, _ = vec.step(actions)
      assert torch.isfinite(actions).all()
      assert torch.isfinite(reward).all()
      assert torch.isfinite(obs["depth"]).all()
    if device.startswith("cuda"):
      torch.cuda.synchronize()
      torch.cuda.reset_peak_memory_stats()
    sense_start = time.perf_counter()
    for _ in range(10):
      env.sim.sense()
    if device.startswith("cuda"):
      torch.cuda.synchronize()
    sense_seconds = (time.perf_counter() - sense_start) / 10
    runner.alg.train_mode()
    update_start = time.perf_counter()
    with torch.inference_mode():
      for _ in range(agent_cfg.num_steps_per_env):
        actions = runner.alg.act(obs)
        obs, reward, dones, extras = vec.step(actions)
        assert torch.isfinite(actions).all() and torch.isfinite(reward).all()
        runner.alg.process_env_step(obs, reward, dones, extras)
      runner.alg.compute_returns(obs)
    losses = runner.alg.update()
    if device.startswith("cuda"):
      torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_start
    assert all(math.isfinite(value) for value in losses.values())
    return {
      "task": TASK_ID,
      "status": "passed",
      "device": device,
      "num_envs": num_envs,
      "action_dim": vec.num_actions,
      "steps": steps,
      "ppo_rollout_steps": agent_cfg.num_steps_per_env,
      "total_env_steps": steps + agent_cfg.num_steps_per_env,
      "depth_shape": list(obs["depth"].shape),
      "depth_definition": "optical Z metres, zero for no hit",
      "depth_min_valid_m": float(valid.min()),
      "depth_max_valid_m": float(valid.max()),
      "initial_depth_valid_fraction": initial_valid_fraction,
      "final_depth_valid_fraction": float((obs["depth"] > 0).float().mean()),
      "rgb_unique_colors": int(np.unique(rgb[0].cpu().numpy().reshape(-1, 3), axis=0).shape[0]),
      "cnn_actor": type(policy).__name__,
      "ppo_smoke_update": {key: float(value) for key, value in losses.items()},
      "sense_ms_per_batch": sense_seconds * 1000,
      "depth_frames_per_second": num_envs / sense_seconds,
      "ppo_smoke_seconds": update_seconds,
      "ppo_smoke_samples_per_second": num_envs * agent_cfg.num_steps_per_env / update_seconds,
      "cuda_peak_allocated_mib": (
        torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
      ),
      "cuda_peak_reserved_mib": (
        torch.cuda.max_memory_reserved() / 2**20 if device.startswith("cuda") else None
      ),
      "formal_training_started": False,
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("docs/assets/g1_visual_stairs_v1a/env_check.json"),
  )
  args = parser.parse_args()
  try:
    result = check(args.device, args.num_envs, args.steps, args.output.parent)
  except Exception as exc:
    result = {
      "task": TASK_ID,
      "status": "failed",
      "device": args.device,
      "error_type": type(exc).__name__,
      "error": str(exc),
      "formal_training_started": False,
    }
    raise
  finally:
    if "result" in locals():
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(result, indent=2) + "\n")
      print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
