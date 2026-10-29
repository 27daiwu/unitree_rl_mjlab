"""Phase A: clean flat-ground navigation velocity control."""
from mjlab.managers.reward_manager import RewardTermCfg

from .env_cfgs import unitree_g1_flat_env_cfg
from .rl_cfg import unitree_g1_ppo_runner_cfg
from src.tasks.velocity.mdp.navigation import (
  NavigationVelocityCommandCfg, axis_tracking, navigation_phase, velocity_error, vertical_velocity,
)


def unitree_g1_navigation_env_cfg(play=False):
  cfg = unitree_g1_flat_env_cfg(play=False)
  cfg.commands["twist"] = NavigationVelocityCommandCfg(
    entity_name="robot", resampling_time_range=(1.5, 3.0),
    heading_command=False, init_velocity_prob=0.0,
    ranges=NavigationVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.2, 0.4), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.8, 0.8)),
  )
  cfg.curriculum = {}
  cfg.events = {key: value for key, value in cfg.events.items() if key in ("reset_base", "reset_robot_joints")}
  for group in cfg.observations.values():
    group.enable_corruption = False
    group.terms["phase"].func = navigation_phase
    for term in group.terms.values():
      term.delay_min_lag = term.delay_max_lag = 0
  cfg.rewards.pop("track_linear_velocity")
  cfg.rewards.pop("track_angular_velocity")
  for axis, name, std, weight in ((0, "vx", 0.10, 2.0), (1, "vy", 0.08, 2.0), (2, "wz", 0.20, 1.5)):
    cfg.rewards[f"track_{name}"] = RewardTermCfg(func=axis_tracking, weight=weight, params={"axis": axis, "std": std})
  cfg.rewards["vertical_velocity"] = RewardTermCfg(func=vertical_velocity, weight=-0.5)
  cfg.rewards["velocity_error"] = RewardTermCfg(func=velocity_error, weight=-0.15)
  cfg.rewards["stop_velocity"] = RewardTermCfg(func=velocity_error, weight=-0.5, params={"zero_only": True})
  cfg.rewards["turn_translation"] = RewardTermCfg(func=velocity_error, weight=-0.3, params={"turn_only": True})
  for reward in cfg.rewards.values():
    if "command_threshold" in reward.params:
      reward.params["command_threshold"] = 1e-6
  # Contact slip must also be penalized while stopped.
  cfg.rewards["foot_slip"].params["command_threshold"] = -1.0
  cfg.rewards["pose"].params["walking_threshold"] = 1e-6
  cfg.rewards["pose"].weight = 0.3
  cfg.rewards["foot_gait"].weight = 0.2
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.episode_length_s = 20.0 if not play else 1e9
  return cfg


def unitree_g1_navigation_ppo_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_navigation"
  return cfg
