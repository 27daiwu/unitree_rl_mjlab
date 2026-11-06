"""Matched W1.13 optimizer-state continuations from the original W1.3 checkpoint."""

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


TASKS = {"control": "Unitree-G1-Flat-W1.11-Right-Track",
         "experiment": "Unitree-G1-Flat-W1.11-Right-Track"}
SOURCE = Path("logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt")
SOURCE_SHA256 = "a89a76cdfd9079e7a9ccfb1c9b343ef90c30164527f0eed84a4805f0ee7b3054"
UPDATES = 400
SEED = 42
NUM_ENVS = 4096
OUTPUT_ROOT = Path("logs/flat_w1_13_optimizer_restore_20261104")


def prepare(group):
  env_cfg = load_env_cfg(TASKS[group])
  rl_cfg = load_rl_cfg(TASKS[group])
  env_cfg.seed = rl_cfg.seed = SEED
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.scene.terrain.num_envs = NUM_ENVS
  rl_cfg.max_iterations = UPDATES
  rl_cfg.run_name = f"flat_w1_13_{group}"
  rl_cfg.logger = "tensorboard"
  rl_cfg.resume = False
  term = env_cfg.rewards["track_linear_velocity"]
  assert term.func is track_linear_velocity and term.weight == 1.0
  assert term.params["std"] == 0.5
  assert term.params.get("right_y_error_scale", 1.0) == 2.0
  toe = env_cfg.rewards["foot_clearance"]
  assert toe.func is feet_clearance and toe.weight == -1.0 and toe.params["target_height"] == 0.1
  return env_cfg, rl_cfg


def digest_state(state):
  digest = hashlib.sha256()
  for section in ("actor_state_dict", "critic_state_dict"):
    for key, value in sorted(state[section].items()):
      digest.update(section.encode())
      digest.update(key.encode())
      digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("group", choices=tuple(TASKS))
  parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
  parser.add_argument("--preflight-only", action="store_true")
  args = parser.parse_args()

  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  configure_torch_backends()
  if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
  source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
  if source_hash != SOURCE_SHA256:
    raise ValueError("W1.3 source checkpoint hash changed")

  output = args.output_root / args.group
  if args.preflight_only:
    output.mkdir(parents=True, exist_ok=True)
  elif not (output / "pre_training_state_audit.json").is_file():
    raise FileNotFoundError(f"Run the optimizer preflight first: {output}")
  env_cfg, rl_cfg = prepare(args.group)
  rl_cfg.save_interval = 50
  if args.preflight_only:
    (output / "source_checkpoint.pt").write_bytes(SOURCE.read_bytes())
    dump_yaml(output / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(output / "params" / "agent.yaml", asdict(rl_cfg))

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    vec = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = load_runner_cls(TASKS[args.group])(vec, asdict(rl_cfg), str(output), device="cuda:0")
    checkpoint = torch.load(SOURCE, map_location="cuda:0", weights_only=False)
    initial_iteration = runner.current_learning_iteration
    initial_env_step = env.common_step_counter
    runner.alg.load(checkpoint, {"actor": True, "critic": True}, strict=True)
    actor_critic_loaded = digest_state({
      "actor_state_dict": getattr(runner.alg, "_raw_actor", runner.alg.actor).state_dict(),
      "critic_state_dict": getattr(runner.alg, "_raw_critic", runner.alg.critic).state_dict(),
    }) == digest_state(checkpoint)
    if not actor_critic_loaded:
      raise RuntimeError("Actor/critic (including normalizers) did not match source checkpoint")

    optimizer_restored = args.group == "experiment"
    if optimizer_restored:
      runner.alg.load(checkpoint, {"optimizer": True}, strict=True)
      checkpoint_lr = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
      runner.alg.learning_rate = checkpoint_lr
    if runner.current_learning_iteration != 0 or initial_iteration != 0:
      raise RuntimeError("Iteration must start at zero")
    if env.common_step_counter != initial_env_step or initial_env_step != 0:
      raise RuntimeError("Environment curriculum counter was unexpectedly restored")
    if optimizer_restored:
      if runner.alg.optimizer.state_dict()["state"].keys() != checkpoint["optimizer_state_dict"]["state"].keys():
        raise RuntimeError("Adam state entries do not match source checkpoint")
      checkpoint_lr = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
      effective_lrs = {group["lr"] for group in runner.alg.optimizer.param_groups}
      if runner.alg.learning_rate != checkpoint_lr or effective_lrs != {checkpoint_lr}:
        raise RuntimeError("Checkpoint learning rate was not restored")
      source_steps = sorted({int(state["step"]) for state in checkpoint["optimizer_state_dict"]["state"].values()})
      restored_steps = sorted({int(state["step"]) for state in runner.alg.optimizer.state.values()})
      if restored_steps != source_steps:
        raise RuntimeError("Adam moment step counters do not match source checkpoint")
    elif runner.alg.optimizer.state:
      raise RuntimeError("Control optimizer must be fresh")

    audit = {
      "group": args.group,
      "source_sha256": source_hash,
      "actor_critic_and_normalizer_loaded": actor_critic_loaded,
      "optimizer_restored": optimizer_restored,
      "source_optimizer_lr": checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"],
      "effective_optimizer_lr_before_training": runner.alg.learning_rate,
      "source_adam_state_entries": len(checkpoint["optimizer_state_dict"]["state"]),
      "effective_adam_state_entries_before_training": len(runner.alg.optimizer.state),
      "iteration_before": initial_iteration,
      "iteration_after_load": runner.current_learning_iteration,
      "environment_common_step_counter_before": initial_env_step,
      "environment_common_step_counter_after": env.common_step_counter,
      "simulator_state_loaded": False,
      "rng_state_loaded": False,
      "checkpoint_keys_ignored": ["iter", "infos", "simulator_state", "rng_state"],
    }
    (output / "pre_training_state_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.preflight_only:
      print(json.dumps(audit, indent=2))
      return
    runner.learn(num_learning_iterations=UPDATES, init_at_random_ep_len=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
