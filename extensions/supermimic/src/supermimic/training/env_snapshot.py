"""Snapshot the live mjlab state needed to resume a SuperMimic environment step."""
from __future__ import annotations

import torch


SIM_DATA_FIELDS = (
    "time", "qpos", "qvel", "act", "qacc_warmstart", "cqacc_warmstart",
    "ctrl", "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
)
COMMAND_FIELDS = (
    "frame", "clip_end", "time_left", "command_counter", "forced_frame",
    "ep_rewards", "ep_bins", "ep_len", "bin_return", "bin_memory",
)
ACTION_MANAGER_FIELDS = ("_action", "_prev_action", "_prev_prev_action")
ACTION_TERM_FIELDS = ("_raw_actions", "_processed_actions", "_executed_raw")
ROBOT_FIELDS = (
    "joint_pos_target", "joint_vel_target", "joint_effort_target", "encoder_bias",
)
ENV_TENSOR_FIELDS = (
    "episode_length_buf", "_manual_reset_pending", "_command_dt",
    "reset_buf", "reset_terminated", "reset_time_outs", "reward_buf",
)


def _copy(t):
    return t.detach().cpu().clone()


def _take_fields(obj, names, *, required=True):
    out = {}
    for name in names:
        if not hasattr(obj, name):
            if required:
                raise AttributeError(f"checkpoint field unavailable: {type(obj).__name__}.{name}")
            continue
        out[name] = _copy(getattr(obj, name))
    return out


def _restore_fields(obj, saved):
    for name, value in saved.items():
        target = getattr(obj, name)
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f"checkpoint field shape/dtype mismatch: {name}")
        target[:] = value.to(device=target.device)


def capture_env(env) -> dict:
    cmd = env.command_manager.get_term("motion")
    action = env.action_manager.get_term("joint_pos")
    histories = {}
    for group, terms in env.observation_manager._group_obs_term_history_buffer.items():
        histories[group] = {
            name: {
                "pointer": buf._pointer,
                "buffer": None if buf._buffer is None else _copy(buf._buffer),
                "num_pushes": _copy(buf._num_pushes),
            }
            for name, buf in terms.items()
        }
    obs = env.observation_manager._obs_buffer
    return {
        "version": 1,
        "num_envs": env.num_envs,
        "sim_data": _take_fields(env.sim.data, SIM_DATA_FIELDS),
        "model_randomization": _take_fields(env.sim.model, sorted(env.sim.expanded_fields)),
        "command": _take_fields(cmd, COMMAND_FIELDS),
        "command_metrics": {name: _copy(t) for name, t in cmd.metrics.items()},
        "action_manager": _take_fields(env.action_manager, ACTION_MANAGER_FIELDS),
        "action_term": _take_fields(action, ACTION_TERM_FIELDS),
        "robot": _take_fields(env.scene["robot"].data, ROBOT_FIELDS),
        "env_tensors": _take_fields(env, ENV_TENSOR_FIELDS, required=False),
        "obs_history": histories,
        "obs_buffer": None if obs is None else {name: _copy(t) for name, t in obs.items()},
        "sim_step_counter": env._sim_step_counter,
        "common_step_counter": env.common_step_counter,
    }


def restore_env(env, state: dict) -> None:
    if state.get("version") != 1 or state.get("num_envs") != env.num_envs:
        raise ValueError("environment checkpoint version or environment count mismatch")
    if set(state["model_randomization"]) != set(env.sim.expanded_fields):
        raise ValueError("domain-randomization field set mismatch")
    _restore_fields(env.sim.model, state["model_randomization"])
    _restore_fields(env.sim.data, state["sim_data"])
    _restore_fields(env, state["env_tensors"])
    _restore_fields(env.scene["robot"].data, state["robot"])
    cmd = env.command_manager.get_term("motion")
    _restore_fields(cmd, state["command"])
    for name, value in state["command_metrics"].items():
        cmd.metrics[name].copy_(value.to(device=env.device))
    _restore_fields(env.action_manager, state["action_manager"])
    _restore_fields(env.action_manager.get_term("joint_pos"), state["action_term"])
    for group, terms in state["obs_history"].items():
        for name, saved in terms.items():
            buf = env.observation_manager._group_obs_term_history_buffer[group][name]
            buf._pointer = int(saved["pointer"])
            buf._num_pushes.copy_(saved["num_pushes"].to(device=env.device))
            if saved["buffer"] is None:
                buf._buffer = None
            elif buf._buffer is None:
                buf._buffer = saved["buffer"].to(device=env.device).clone()
            else:
                buf._buffer.copy_(saved["buffer"].to(device=env.device))
    # Forward refreshes derived MuJoCo arrays after restoring the primary state.
    env.sim.forward()
    cmd._refresh_ref()
    cmd.update_relative_body_poses()
    obs = state["obs_buffer"]
    env.observation_manager._obs_buffer = None if obs is None else {
        name: value.to(device=env.device).clone() for name, value in obs.items()
    }
    env.obs_buf = env.observation_manager._obs_buffer
    env._sim_step_counter = int(state["sim_step_counter"])
    env.common_step_counter = int(state["common_step_counter"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
