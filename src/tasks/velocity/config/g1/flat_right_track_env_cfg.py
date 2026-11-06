"""Explicit W1.11 rightward velocity-tracking reward experiment."""

from mjlab.envs import ManagerBasedRlEnvCfg

from .env_cfgs import unitree_g1_flat_env_cfg


def unitree_g1_flat_right_track_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.rewards["track_linear_velocity"].params["right_y_error_scale"] = 2.0
  return cfg
