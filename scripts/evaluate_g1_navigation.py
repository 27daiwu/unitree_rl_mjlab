"""Deterministic navigation evaluation; checkpoint inference or zero-action smoke.

No optimizer/training calls. Example: python scripts/evaluate_g1_navigation.py
--checkpoint PATH --output-dir logs/navigation_eval
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import torch
import tyro

from src.tasks.velocity.mdp.navigation import evaluation_observations
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity.mdp.navigation_eval import SCENARIOS, RecoveryTolerance, summarize_trace


@dataclass
class EvaluateConfig:
  checkpoint: str | None = None
  smoke: bool = False
  num_envs: int = 4
  seed: int = 42
  device: str = "cuda:0"
  output_dir: str = "logs/navigation_eval"
  scenario: str | None = None
  max_steps: int | None = None


def snapshot(env):
  """Metrics run before automatic reset: preserve terminal states, never reset poses."""
  d = env.scene["robot"].data
  env.navigation_snapshot = torch.cat((
    d.root_link_lin_vel_b[:, :2], d.root_link_ang_vel_b[:, 2:3],
    d.root_link_pos_w, d.root_link_quat_w,
  ), dim=1).detach().clone()
  return torch.zeros(env.num_envs, device=env.device)


def decode(sample):
  a = sample.cpu().numpy()
  w, x, y, z = a[:, 6:10].T
  rpy = np.column_stack((np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                        np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                        np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))
  return a[:, :3], a[:, 3:6], rpy


def run(cfg: EvaluateConfig, disturbance=None):
  if cfg.smoke == (cfg.checkpoint is not None):
    raise ValueError("Provide either --checkpoint or --smoke (exclusively)")
  if cfg.num_envs < 1 or (cfg.max_steps is not None and cfg.max_steps < 2):
    raise ValueError("num_envs >= 1 and max_steps >= 2 required")
  if cfg.max_steps is not None and not cfg.smoke:
    raise ValueError("max_steps truncation is reserved for smoke runs")
  import src.tasks  # noqa: F401
  task = "Unitree-G1-Navigation"
  env_cfg = load_env_cfg(task)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.episode_length_s = 1000.
  env_cfg.events["reset_base"].params["pose_range"] = {}
  env_cfg.metrics["navigation_snapshot"] = MetricsTermCfg(func=snapshot)
  if disturbance is not None:
    disturbance.configure(env_cfg)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  output = Path(cfg.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  try:
    agent_cfg = load_rl_cfg(task)
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = None
    if cfg.checkpoint:
      runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
      runner = runner_cls(vec, asdict(agent_cfg), device=cfg.device)
      runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location=cfg.device)
      policy = runner.get_inference_policy(device=cfg.device)
    scenarios = {cfg.scenario: SCENARIOS[cfg.scenario]} if cfg.scenario else SCENARIOS
    results = []
    for name, segments in scenarios.items():
      env.reset(seed=cfg.seed)
      term = env.command_manager.get_term("twist")
      snapshot(env)
      _, initial_pos, initial_rpy = decode(env.navigation_snapshot)
      alive = np.ones(cfg.num_envs, dtype=bool)
      fallen = np.zeros(cfg.num_envs, dtype=bool)
      traces = [dict(time=[], command=[], velocity=[], position=[], rpy=[],
                     initial_position=initial_pos[i].copy(), initial_yaw=float(initial_rpy[i, 2])) for i in range(cfg.num_envs)]
      step = 0
      for duration, command in segments:
        for _ in range(round(duration / env.step_dt)):
          term.set_evaluation_command(command)
          if disturbance is not None:
            disturbance.before_step(env, step * env.step_dt)
          # Refresh command observations BEFORE policy inference (no stale command step).
          obs = evaluation_observations(env)
          with torch.no_grad():
            action = policy(obs) if policy else torch.zeros((cfg.num_envs, env.action_manager.total_action_dim), device=cfg.device)
            _, reward, terminated, truncated, _ = env.step(action)
          if not torch.isfinite(reward).all() or not torch.isfinite(env.navigation_snapshot).all():
            raise RuntimeError("Non-finite simulation output")
          velocity, position, rpy = decode(env.navigation_snapshot)
          step += 1
          for i in np.flatnonzero(alive):
            for key, value in (("time", step*env.step_dt), ("command", command),
                               ("velocity", velocity[i]), ("position", position[i]), ("rpy", rpy[i])):
              traces[i][key].append(np.array(value).copy())
          falls = terminated.cpu().numpy().astype(bool)
          ends = falls | truncated.cpu().numpy().astype(bool)
          fallen |= alive & falls
          alive &= ~ends
          if not alive.any() or (cfg.max_steps and step >= cfg.max_steps):
            break
        if not alive.any() or (cfg.max_steps and step >= cfg.max_steps):
          break
      for i, trace in enumerate(traces):
        np.savez_compressed(output / f"{name}_{i}.npz", **trace)
        result = summarize_trace(trace, bool(fallen[i]), () if disturbance is None else disturbance.windows)
        results.append(dict(scenario=name, env_index=i, completed=bool(alive[i] and not (cfg.max_steps and step >= cfg.max_steps)), **result))
    report = dict(task=task, config=asdict(cfg), tolerance=asdict(RecoveryTolerance()),
                  policy_quality_evaluation=not cfg.smoke,
                  fall_rate=float(np.mean([r["fall_rate"] for r in results])), trials=results)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
  finally:
    env.close()


if __name__ == "__main__":
  result = run(tyro.cli(EvaluateConfig))
  print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
