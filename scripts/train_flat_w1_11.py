"""Matched W1.11 rightward velocity-reward continuations from W1.3."""

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
from src.tasks.velocity.mdp.rewards import feet_clearance, track_linear_velocity


TASKS = {"control": "Unitree-G1-Flat", "experiment": "Unitree-G1-Flat-W1.11-Right-Track"}
SOURCE = Path("logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt")
SOURCE_SHA256 = "a89a76cdfd9079e7a9ccfb1c9b343ef90c30164527f0eed84a4805f0ee7b3054"
UPDATES = 400
SEED = 42
NUM_ENVS = 4096


def prepare(group):
  env_cfg = load_env_cfg(TASKS[group])
  rl_cfg = load_rl_cfg(TASKS[group])
  env_cfg.seed = rl_cfg.seed = SEED
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.scene.terrain.num_envs = NUM_ENVS
  rl_cfg.max_iterations = UPDATES
  rl_cfg.run_name = f"flat_w1_11_{group}"
  rl_cfg.logger = "tensorboard"
  rl_cfg.resume = False
  term = env_cfg.rewards["track_linear_velocity"]
  assert term.func is track_linear_velocity and term.weight == 1.0
  assert term.params["std"] == 0.5
  assert term.params.get("right_y_error_scale", 1.0) == (1.0 if group == "control" else 2.0)
  toe = env_cfg.rewards["foot_clearance"]
  assert toe.func is feet_clearance and toe.weight == -1.0 and toe.params["target_height"] == 0.1
  return env_cfg, rl_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("group", choices=tuple(TASKS))
  parser.add_argument("--output-root", default="logs/flat_w1_11_right_track_20261030")
  args = parser.parse_args()
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  configure_torch_backends()
  if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
  with SOURCE.open("rb") as source_file:
    source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
  if source_hash != SOURCE_SHA256:
    raise ValueError("W1.3 source checkpoint hash changed")
  output = Path(args.output_root) / args.group
  output.mkdir(parents=True, exist_ok=False)

  env_cfg, rl_cfg = prepare(args.group)
  manifest = {
    "source": str(SOURCE), "source_sha256": source_hash, "group": args.group,
    "task": TASKS[args.group], "seed": SEED, "num_envs": NUM_ENVS, "updates": UPDATES,
    "right_y_error_scale": 1.0 if args.group == "control" else 2.0,
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
