"""Fixed-learning-rate matched optimizer continuations for W1.15."""

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


TASK = "Unitree-G1-Flat-W1.11-Right-Track"
SOURCE = Path("logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt")
SOURCE_SHA256 = "a89a76cdfd9079e7a9ccfb1c9b343ef90c30164527f0eed84a4805f0ee7b3054"
OUTPUT_ROOT = Path("logs/flat_w1_15_fixed_lr_optimizer_20261104")
UPDATES = 400
SEED = 42
NUM_ENVS = 4096
FIXED_LR = 2.562890625e-4
CHECKPOINT_ITERATIONS = {50, 100, 200, 300}


def prepare():
  env_cfg = load_env_cfg(TASK)
  rl_cfg = load_rl_cfg(TASK)
  env_cfg.seed = rl_cfg.seed = SEED
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.scene.terrain.num_envs = NUM_ENVS
  rl_cfg.max_iterations = UPDATES
  rl_cfg.run_name = "flat_w1_15_optimizer"
  rl_cfg.logger = "tensorboard"
  rl_cfg.resume = False
  rl_cfg.save_interval = UPDATES + 1
  rl_cfg.algorithm.learning_rate = FIXED_LR
  rl_cfg.algorithm.schedule = "fixed"

  linear = env_cfg.rewards["track_linear_velocity"]
  assert linear.func is track_linear_velocity and linear.weight == 1.0
  assert linear.params["std"] == 0.5
  assert linear.params.get("right_y_error_scale", 1.0) == 2.0
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


def reward_signature(env_cfg):
  rows = []
  for name, term in env_cfg.rewards.items():
    rows.append({
      "name": name,
      "function": f"{term.func.__module__}.{term.func.__name__}",
      "weight": float(term.weight),
      "params": term.params,
    })
  return json.loads(json.dumps(rows, sort_keys=True, default=str))


def optimizer_steps(optimizer):
  return sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})


def optimizer_state_matches(source_state, restored_state):
  source_groups = source_state["param_groups"]
  restored_groups = restored_state["param_groups"]
  if len(source_groups) != len(restored_groups):
    return False
  for source_group, restored_group in zip(source_groups, restored_groups):
    source_params = source_group["params"]
    restored_params = restored_group["params"]
    if len(source_params) != len(restored_params):
      return False
    for source_id, restored_id in zip(source_params, restored_params):
      source_values = source_state["state"].get(source_id)
      restored_values = restored_state["state"].get(restored_id)
      if source_values is None or restored_values is None or source_values.keys() != restored_values.keys():
        return False
      for key, source_value in source_values.items():
        restored_value = restored_values[key]
        if torch.is_tensor(source_value):
          if not torch.equal(source_value.cpu(), restored_value.detach().cpu()):
            return False
        elif source_value != restored_value:
          return False
  return True


def run_group(group, output_root, preflight_only):
  output = output_root / group
  if preflight_only:
    if output.exists():
      if (output / "pre_training_state_audit.json").exists():
        raise FileExistsError(output / "pre_training_state_audit.json")
    else:
      output.mkdir(parents=True)
  elif not (output / "pre_training_state_audit.json").is_file():
    raise FileNotFoundError(f"Run preflight first: {output}")
  pair_marker = output_root / "pair_preflight_verified.json"
  if not preflight_only and not pair_marker.is_file():
    raise FileNotFoundError(f"Run --verify-pair before training: {pair_marker}")
  if not preflight_only and (output / "ppo_update_lr_audit.jsonl").exists():
    raise RuntimeError(
      f"Refusing to reuse an output directory with an existing PPO LR ledger: {output}. "
      "Use a new --output-root to preserve any partial run."
    )

  env_cfg, rl_cfg = prepare()
  if rl_cfg.algorithm.schedule != "fixed" or rl_cfg.algorithm.learning_rate != FIXED_LR:
    raise RuntimeError("PPO must use the registered fixed W1.15 learning rate")
  if preflight_only:
    dump_yaml(output / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(output / "params" / "agent.yaml", asdict(rl_cfg))
    (output / "source_checkpoint.pt").write_bytes(SOURCE.read_bytes())

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    vec = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = load_runner_cls(TASK)(vec, asdict(rl_cfg), str(output), device="cuda:0")
    checkpoint = torch.load(SOURCE, map_location="cuda:0", weights_only=False)
    initial_iteration = runner.current_learning_iteration
    initial_env_step = int(env.common_step_counter)
    runner.alg.load(checkpoint, {"actor": True, "critic": True}, strict=True)
    actor_critic_digest = digest_state({
      "actor_state_dict": getattr(runner.alg, "_raw_actor", runner.alg.actor).state_dict(),
      "critic_state_dict": getattr(runner.alg, "_raw_critic", runner.alg.critic).state_dict(),
    })
    source_digest = digest_state(checkpoint)
    if actor_critic_digest != source_digest:
      raise RuntimeError("Actor, critic, or normalizer differs from W1.3 source")

    source_optimizer = checkpoint["optimizer_state_dict"]
    source_steps = sorted({int(s["step"].item()) for s in source_optimizer["state"].values()})
    restored = group == "experiment"
    if restored:
      runner.alg.load(checkpoint, {"optimizer": True}, strict=True)
    runner.alg.learning_rate = FIXED_LR
    for param_group in runner.alg.optimizer.param_groups:
      param_group["lr"] = FIXED_LR

    if runner.current_learning_iteration != 0 or initial_iteration != 0:
      raise RuntimeError("Runner iteration must start at zero")
    if initial_env_step != 0 or int(env.common_step_counter) != 0:
      raise RuntimeError("Environment common_step_counter must start at zero")
    if restored:
      restored_optimizer = runner.alg.optimizer.state_dict()
      if len(restored_optimizer["state"]) != len(source_optimizer["state"]):
        raise RuntimeError("Experiment Adam state entry count differs from W1.3")
      if not optimizer_state_matches(source_optimizer, restored_optimizer):
        raise RuntimeError("Experiment Adam moments/steps differ from W1.3 after parameter remapping")
      if optimizer_steps(runner.alg.optimizer) != source_steps:
        raise RuntimeError("Experiment Adam step counters differ from W1.3")
    elif runner.alg.optimizer.state:
      raise RuntimeError("Control Adam must be fresh")
    if runner.alg.learning_rate != FIXED_LR or {p["lr"] for p in runner.alg.optimizer.param_groups} != {FIXED_LR}:
      raise RuntimeError("Initial effective LR differs from the fixed W1.15 LR")

    agent_cfg = asdict(rl_cfg)
    pair_invariants = {
      "source_sha256": SOURCE_SHA256,
      "actor_critic_normalizer_sha256": actor_critic_digest,
      "task": TASK,
      "seed": SEED,
      "num_envs": NUM_ENVS,
      "updates": UPDATES,
      "fixed_learning_rate": FIXED_LR,
      "schedule": rl_cfg.algorithm.schedule,
      "algorithm": agent_cfg["algorithm"],
      "reward_signature": reward_signature(env_cfg),
    }
    audit = {
      "group": group,
      "source_sha256": SOURCE_SHA256,
      "actor_critic_and_normalizer_loaded": True,
      "actor_critic_normalizer_sha256": actor_critic_digest,
      "optimizer_mode": "restored_full_adam_state" if restored else "fresh_adam",
      "optimizer_state_entries": len(runner.alg.optimizer.state),
      "optimizer_step_values": optimizer_steps(runner.alg.optimizer),
      "source_optimizer_state_entries": len(source_optimizer["state"]),
      "source_optimizer_step_values": source_steps,
      "fixed_learning_rate": FIXED_LR,
      "effective_learning_rate_before_training": runner.alg.learning_rate,
      "optimizer_param_group_lrs_before_training": [p["lr"] for p in runner.alg.optimizer.param_groups],
      "schedule": rl_cfg.algorithm.schedule,
      "iteration_before": initial_iteration,
      "iteration_after_load": runner.current_learning_iteration,
      "environment_common_step_counter_before": initial_env_step,
      "environment_common_step_counter_after": int(env.common_step_counter),
      "reward_signature": pair_invariants["reward_signature"],
      "pair_invariants": pair_invariants,
      "simulator_state_loaded": False,
      "rng_state_loaded": False,
      "curriculum_state_loaded": False,
    }
    if preflight_only:
      (output / "pre_training_state_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
      print(json.dumps(audit, indent=2))
      return

    with (output / "ppo_update_lr_audit.jsonl").open("x") as ledger:
      original_update = runner.alg.update
      update_count = 0

      def audited_update(*args, **kwargs):
        nonlocal update_count
        update_count += 1
        before_groups = [group["lr"] for group in runner.alg.optimizer.param_groups]
        before_scalar = runner.alg.learning_rate
        if before_scalar != FIXED_LR or before_groups != [FIXED_LR] * len(before_groups):
          raise RuntimeError(f"LR mismatch before PPO update {update_count}")
        result = original_update(*args, **kwargs)
        after_groups = [group["lr"] for group in runner.alg.optimizer.param_groups]
        after_scalar = runner.alg.learning_rate
        if after_scalar != FIXED_LR or after_groups != [FIXED_LR] * len(after_groups):
          raise RuntimeError(f"LR changed during PPO update {update_count}")
        ledger.write(json.dumps({
          "ppo_update": update_count,
          "runner_iteration": update_count - 1,
          "learning_rate_before": before_scalar,
          "param_group_lrs_before": before_groups,
          "learning_rate_after": after_scalar,
          "param_group_lrs_after": after_groups,
        }) + "\n")
        ledger.flush()
        if update_count in {iteration + 1 for iteration in CHECKPOINT_ITERATIONS}:
          payload = runner.alg.save()
          payload["iter"] = update_count - 1
          torch.save(payload, output / f"model_{update_count - 1}.pt")
        return result

      runner.alg.update = audited_update
      runner.learn(num_learning_iterations=UPDATES, init_at_random_ep_len=True)
      if update_count != UPDATES:
        raise RuntimeError(f"Expected {UPDATES} PPO updates, observed {update_count}")
  finally:
    env.close()


def verify_pair(output_root):
  audits = []
  for group in ("control", "experiment"):
    path = output_root / group / "pre_training_state_audit.json"
    if not path.is_file():
      raise FileNotFoundError(path)
    audits.append(json.loads(path.read_text()))
  control, experiment = audits
  if control["pair_invariants"] != experiment["pair_invariants"]:
    raise RuntimeError("Control and Experiment reward/PPO/source invariants differ")
  if control["optimizer_mode"] != "fresh_adam" or control["optimizer_state_entries"] != 0:
    raise RuntimeError("Control preflight did not produce a fresh Adam optimizer")
  if experiment["optimizer_mode"] != "restored_full_adam_state":
    raise RuntimeError("Experiment preflight did not restore the full Adam optimizer")
  marker = {
    "pair_invariants_match": True,
    "control_optimizer_state_entries": control["optimizer_state_entries"],
    "experiment_optimizer_state_entries": experiment["optimizer_state_entries"],
    "fixed_learning_rate": FIXED_LR,
    "per_update_lr_audit_required": True,
  }
  path = output_root / "pair_preflight_verified.json"
  if path.exists():
    raise FileExistsError(path)
  path.write_text(json.dumps(marker, indent=2) + "\n")
  print(json.dumps(marker, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("group", nargs="?", choices=("control", "experiment"))
  parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
  parser.add_argument("--preflight-only", action="store_true")
  parser.add_argument("--verify-pair", action="store_true")
  args = parser.parse_args()
  if args.verify_pair == (args.group is not None):
    parser.error("provide a group for preflight/training, or use --verify-pair alone")

  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  configure_torch_backends()
  if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
  source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
  if source_hash != SOURCE_SHA256:
    raise ValueError("W1.3 source checkpoint hash changed")
  if args.verify_pair:
    verify_pair(args.output_root)
  else:
    run_group(args.group, args.output_root, args.preflight_only)


if __name__ == "__main__":
  main()
