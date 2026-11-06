"""Matched W1.6 continuations from the original W1.3 checkpoint."""

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import os

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401
from src.tasks.velocity.mdp.rewards import feet_clearance, flat_toe_clearance


TASKS = {"control": "Unitree-G1-Flat", "experiment": "Unitree-G1-Flat-W1.6-Toe-Rectangle"}
SOURCE = Path("logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt")
UPDATES = 400
SEED = 42
NUM_ENVS = 4096


def prepare(group):
  task = TASKS[group]
  env_cfg = load_env_cfg(task)
  rl_cfg = load_rl_cfg(task)
  env_cfg.seed = rl_cfg.seed = SEED
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.scene.terrain.num_envs = NUM_ENVS
  rl_cfg.max_iterations = UPDATES
  rl_cfg.run_name = f"flat_w1_6_{group}"
  rl_cfg.logger = "tensorboard"
  rl_cfg.resume = False
  term = env_cfg.rewards["foot_clearance"]
  assert term.weight == -1.0
  if group == "control":
    assert term.func is feet_clearance and term.params["target_height"] == 0.10
  else:
    assert term.func is flat_toe_clearance and term.params["target_height"] == 0.03
  return env_cfg, rl_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("group", choices=("control", "experiment"))
  parser.add_argument("--output-root", default="logs/flat_w1_6_toe_clearance_20261030")
  args = parser.parse_args()
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  configure_torch_backends()
  if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
  output = Path(args.output_root) / args.group
  output.mkdir(parents=True, exist_ok=False)
  with SOURCE.open("rb") as source_file:
    source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()

  env_cfg, rl_cfg = prepare(args.group)
  manifest = {
    "source": str(SOURCE), "source_sha256": source_hash, "group": args.group,
    "seed": SEED, "num_envs": NUM_ENVS, "updates": UPDATES,
    "foot_clearance": "legacy_site" if args.group == "control" else "toe_mid_swing_30mm",
    "warm_start": {"actor": True, "critic": True, "optimizer": False,
                   "iteration": False, "simulator_state": False, "rng_state": False},
  }
  (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  dump_yaml(output / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(output / "params" / "agent.yaml", asdict(rl_cfg))

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    vec = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = load_runner_cls(TASKS[args.group])(vec, asdict(rl_cfg), str(output), device="cuda:0")
    checkpoint = torch.load(SOURCE, map_location="cuda:0", weights_only=False)
    runner.alg.load(checkpoint, {"actor": True, "critic": True}, strict=True)
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
      raise RuntimeError("Warm start unexpectedly restored iteration or optimizer")
    runner.learn(num_learning_iterations=UPDATES, init_at_random_ep_len=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
