"""Deterministic full-motion evaluation for tracking policies."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends

from src.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


@dataclass(frozen=True)
class EvalTrackingConfig:
  checkpoint_file: str
  motion_file: str
  device: str | None = None
  output_file: str | None = None
  start_from_fixstand: bool = True
  max_steps: int | None = None


def _read_motion_start_frame(motion_file: Path) -> int:
  with np.load(motion_file) as data:
    if "motion_start_frame" in data:
      return int(np.asarray(data["motion_start_frame"]).reshape(-1)[0])
    if "transition_frames" in data:
      return int(np.asarray(data["transition_frames"]).reshape(-1)[0])
  return 0


def _tracking_errors(env: ManagerBasedRlEnv, motion_start_frame: int) -> dict[str, Any]:
  command = cast(MotionCommand, env.command_manager.get_term("motion"))
  time_step = int(command.time_steps[0].item())
  total_steps = int(command.motion.time_step_total)
  motion_steps = max(total_steps - motion_start_frame - 1, 1)
  motion_step = max(time_step - motion_start_frame, 0)

  root_pos_error = command.anchor_pos_w[0] - command.robot_anchor_pos_w[0]
  body_pos_error = torch.norm(
    command.body_pos_w[0] - command.robot_body_pos_w[0], dim=-1
  )
  body_ori_error = quat_error_magnitude(
    command.body_quat_w[0], command.robot_body_quat_w[0]
  )

  return {
    "time_step": time_step,
    "time_s": time_step * env.step_dt,
    "phase_total": time_step / max(total_steps - 1, 1),
    "motion_phase": min(motion_step / motion_steps, 1.0),
    "root_pos_error_m": float(torch.norm(root_pos_error).item()),
    "root_xy_error_m": float(torch.norm(root_pos_error[:2]).item()),
    "root_z_error_m": float(root_pos_error[2].item()),
    "root_ori_error_rad": float(
      quat_error_magnitude(
        command.anchor_quat_w[0:1], command.robot_anchor_quat_w[0:1]
      )[0].item()
    ),
    "body_pos_error_mean_m": float(body_pos_error.mean().item()),
    "body_pos_error_max_m": float(body_pos_error.max().item()),
    "body_ori_error_mean_rad": float(body_ori_error.mean().item()),
    "body_ori_error_max_rad": float(body_ori_error.max().item()),
    "joint_pos_error_rad": float(
      torch.norm(command.joint_pos[0] - command.robot_joint_pos[0]).item()
    ),
    "joint_vel_error_rad_s": float(
      torch.norm(command.joint_vel[0] - command.robot_joint_vel[0]).item()
    ),
  }


def _termination_reasons(env: ManagerBasedRlEnv) -> list[str]:
  term_dones = getattr(env.termination_manager, "_term_dones", {})
  reasons: list[str] = []
  for name, value in term_dones.items():
    if bool(value[0].item()):
      reasons.append(name)
  return reasons


def run_eval(task_id: str, cfg: EvalTrackingConfig) -> dict[str, Any]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  motion_file = Path(cfg.motion_file).expanduser().resolve()
  checkpoint_file = Path(cfg.checkpoint_file).expanduser().resolve()
  if not motion_file.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_file}")
  if not checkpoint_file.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  env_cfg.events = {}
  env_cfg.observations["actor"].enable_corruption = False

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(motion_file)
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)
  motion_cmd.sampling_mode = "start"
  motion_cmd.reset_robot_to_motion_state = not cfg.start_from_fixstand
  env_cfg.episode_length_s = int(1e9)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  command = cast(MotionCommand, raw_env.command_manager.get_term("motion"))
  total_motion_steps = int(command.motion.time_step_total)
  eval_steps = max(total_motion_steps - 1, 1)
  if cfg.max_steps is not None:
    eval_steps = min(eval_steps, cfg.max_steps)
  motion_start_frame = _read_motion_start_frame(motion_file)

  obs = env.get_observations()
  last_errors = _tracking_errors(raw_env, motion_start_frame)
  first_failure: dict[str, Any] | None = None

  for step in range(eval_steps):
    with torch.no_grad():
      action = policy(obs)
    obs, _, dones, _ = env.step(action)

    if bool(dones[0].item()):
      first_failure = {
        "eval_step": step + 1,
        "eval_time_s": (step + 1) * raw_env.step_dt,
        "termination_reasons": _termination_reasons(raw_env),
        "last_pre_failure_errors": last_errors,
      }
      break

    last_errors = _tracking_errors(raw_env, motion_start_frame)

  result = {
    "task_id": task_id,
    "checkpoint_file": str(checkpoint_file),
    "motion_file": str(motion_file),
    "start_from_fixstand": cfg.start_from_fixstand,
    "observation_noise": False,
    "domain_randomization": False,
    "push_robot": False,
    "motion_steps": total_motion_steps,
    "motion_start_frame": motion_start_frame,
    "evaluated_steps": eval_steps if first_failure is None else first_failure["eval_step"],
    "success": first_failure is None,
    "first_failure": first_failure,
    "final_errors": last_errors,
  }

  env.close()

  if cfg.output_file is not None:
    output_file = Path(cfg.output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps(result, indent=2))
  return result


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    EvalTrackingConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_eval(chosen_task, args)


if __name__ == "__main__":
  main()
