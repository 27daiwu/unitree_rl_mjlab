"""Deterministic checkpoint evaluation for the Phase 2A low-stairs task.

This script is intentionally an evaluator only; it never updates policy weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.mdp.stairs_metrics import geometry_tensors


@dataclass(frozen=True)
class EvaluateConfig:
  checkpoint: str
  episodes: int = 100
  seed: int = 42
  device: str | None = None
  output_file: str | None = None


def run(task_id: str, cfg: EvaluateConfig) -> dict[str, object]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.episodes
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
  runner.load(
    cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  obs = vec_env.get_observations()
  done = torch.zeros(cfg.episodes, dtype=torch.bool, device=device)
  returns = torch.zeros_like(done, dtype=torch.float32)
  max_index = torch.zeros_like(returns)
  max_x_index = torch.zeros_like(returns)
  max_x_progress = torch.full_like(returns, -torch.inf)
  final_x_progress = torch.zeros_like(returns)
  max_root_height = torch.full_like(returns, -torch.inf)
  max_left_foot_height = torch.full_like(returns, -torch.inf)
  max_right_foot_height = torch.full_like(returns, -torch.inf)
  min_first_riser_distance = torch.full_like(returns, torch.inf)
  final_first_riser_distance = torch.zeros_like(returns)
  max_abs_root_y = torch.zeros_like(returns)
  forward_velocity_sum = torch.zeros_like(returns)
  episode_steps = torch.zeros_like(returns)
  success = torch.zeros_like(done)
  first_riser_reached = torch.zeros_like(done)
  fallen = torch.zeros_like(done)
  timeout = torch.zeros_like(done)
  geometry = geometry_tensors(env)
  robot = env.scene["robot"]
  left_foot_id = robot.site_names.index("left_foot")
  right_foot_id = robot.site_names.index("right_foot")
  while not bool(done.all()):
    with torch.no_grad():
      action = policy(obs)  # MLPModel defaults to deterministic mean output.
    obs, reward, dones, _ = vec_env.step(action)
    active = ~done
    returns += torch.where(active, reward, torch.zeros_like(reward))
    metrics = env.metrics_manager._step_values
    names = env.metrics_manager.active_terms

    def metric(name: str) -> torch.Tensor:
      return metrics[:, names.index(name)]

    max_index = torch.where(
      active,
      torch.maximum(max_index, metric("stairs/physical_stair_index")),
      max_index,
    )
    max_x_index = torch.where(
      active,
      torch.maximum(max_x_index, metric("stairs/x_progress_index")),
      max_x_index,
    )
    local_root = robot.data.root_link_pos_w - env.scene.env_origins
    feet = robot.data.site_pos_w[:, (left_foot_id, right_foot_id)]
    local_feet = feet - env.scene.env_origins.unsqueeze(1)
    root_x_progress = local_root[:, 0]
    front_foot_patch_x = local_feet[..., 0].max(dim=1).values + geometry["spawn_x"]
    first_riser_distance = geometry["staircase_start_x"] - front_foot_patch_x
    max_x_progress = torch.where(
      active, torch.maximum(max_x_progress, root_x_progress), max_x_progress
    )
    final_x_progress = torch.where(active, root_x_progress, final_x_progress)
    max_root_height = torch.where(
      active, torch.maximum(max_root_height, local_root[:, 2]), max_root_height
    )
    max_left_foot_height = torch.where(
      active,
      torch.maximum(max_left_foot_height, local_feet[:, 0, 2]),
      max_left_foot_height,
    )
    max_right_foot_height = torch.where(
      active,
      torch.maximum(max_right_foot_height, local_feet[:, 1, 2]),
      max_right_foot_height,
    )
    min_first_riser_distance = torch.where(
      active,
      torch.minimum(min_first_riser_distance, first_riser_distance),
      min_first_riser_distance,
    )
    final_first_riser_distance = torch.where(
      active, first_riser_distance, final_first_riser_distance
    )
    max_abs_root_y = torch.where(
      active, torch.maximum(max_abs_root_y, local_root[:, 1].abs()), max_abs_root_y
    )
    # A foot within 0.15 m of the first riser has reached a genuine interaction
    # region.  This remains separate from verified support on the first tread.
    first_riser_reached |= active & (first_riser_distance <= 0.15)
    forward_velocity_sum += torch.where(
      active, metric("stairs/forward_velocity"), torch.zeros_like(returns)
    )
    episode_steps += active.float()
    success |= active & (metric("stairs/reached_top_platform") > 0.5)
    newly_done = dones.bool() & ~done
    if newly_done.any():
      fallen |= newly_done & (metric("stairs/fell_over") > 0.5)
      timeout |= newly_done & (metric("stairs/episode_timeout") > 0.5)
      done |= newly_done

  success &= ~fallen

  first_stair_support = max_index >= 1.0
  threshold_rates = {
    f"physical_index_ge{level}_rate": (max_index >= level).float().mean().item()
    for level in range(1, 9)
  }
  no_support = ~first_stair_support
  reached_no_support = first_riser_reached & no_support
  episode_summaries = []
  for index in range(cfg.episodes):
    episode_summaries.append(
      {
        "env_index": index,
        "success": bool(success[index].item()),
        "fell_over": bool(fallen[index].item()),
        "timeout": bool(timeout[index].item()),
        "first_riser_reached": bool(first_riser_reached[index].item()),
        "first_stair_support": bool(first_stair_support[index].item()),
        "max_physical_stair_index": max_index[index].item(),
        "max_x_progress_index": max_x_index[index].item(),
        "max_root_x_progress": max_x_progress[index].item(),
        "final_root_x_progress": final_x_progress[index].item(),
        "max_root_terrain_relative_height": max_root_height[index].item(),
        "max_left_foot_height": max_left_foot_height[index].item(),
        "max_right_foot_height": max_right_foot_height[index].item(),
        "min_first_riser_distance": min_first_riser_distance[index].item(),
        "final_first_riser_distance": final_first_riser_distance[index].item(),
        "max_abs_root_y": max_abs_root_y[index].item(),
        "episode_return": returns[index].item(),
      }
    )

  result: dict[str, object] = {
    "task": task_id,
    "checkpoint": str(Path(cfg.checkpoint).resolve()),
    "episodes": cfg.episodes,
    "seed": cfg.seed,
    "success_rate": success.float().mean().item(),
    "fall_rate": fallen.float().mean().item(),
    "timeout_rate": timeout.float().mean().item(),
    "mean_max_physical_stair_index": max_index.mean().item(),
    "max_physical_stair_index": max_index.max().item(),
    "mean_max_x_progress_index": max_x_index.mean().item(),
    "max_x_progress_index": max_x_index.max().item(),
    **threshold_rates,
    "first_riser_reached_rate": first_riser_reached.float().mean().item(),
    "first_stair_support_rate": first_stair_support.float().mean().item(),
    "reached_first_riser_without_support_rate": reached_no_support.float().mean().item(),
    "mean_max_root_x_progress": max_x_progress.mean().item(),
    "max_root_x_progress": max_x_progress.max().item(),
    "mean_final_root_x_progress": final_x_progress.mean().item(),
    "mean_max_root_terrain_relative_height": max_root_height.mean().item(),
    "max_root_terrain_relative_height": max_root_height.max().item(),
    "mean_max_left_foot_height": max_left_foot_height.mean().item(),
    "max_left_foot_height": max_left_foot_height.max().item(),
    "mean_max_right_foot_height": max_right_foot_height.mean().item(),
    "max_right_foot_height": max_right_foot_height.max().item(),
    "mean_min_first_riser_distance": min_first_riser_distance.mean().item(),
    "mean_min_first_riser_distance_without_support": (
      min_first_riser_distance[no_support].mean().item() if no_support.any() else None
    ),
    "mean_max_foot_height_without_support": (
      torch.maximum(max_left_foot_height, max_right_foot_height)[no_support].mean().item()
      if no_support.any()
      else None
    ),
    "mean_max_root_height_without_support": (
      max_root_height[no_support].mean().item() if no_support.any() else None
    ),
    "lateral_boundary_cross_rate": (max_abs_root_y > 2.15).float().mean().item(),
    "max_abs_root_y": max_abs_root_y.max().item(),
    "top_platform_support_rate": success.float().mean().item(),
    "mean_episode_return": returns.mean().item(),
    "mean_forward_velocity": (
      forward_velocity_sum / episode_steps.clamp(min=1.0)
    ).mean().item(),
    "episode_summaries": episode_summaries,
  }
  if cfg.output_file:
    output = Path(cfg.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
  env.close()
  return result


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  task_id = "Unitree-G1-Stairs-Clearance-Ablation"
  cfg = tyro.cli(EvaluateConfig)
  print(json.dumps(run(task_id, cfg), indent=2))


if __name__ == "__main__":
  main()
