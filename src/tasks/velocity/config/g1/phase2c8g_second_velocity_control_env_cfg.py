"""Phase 2C.8-G isolated second velocity-control intervention config."""

from mjlab.envs import ManagerBasedRlEnvCfg

from .phase2c8d_overspeed_penalty_env_cfg import (
  unitree_g1_phase2c8d_overspeed_penalty_env_cfg,
)


OVERSPEED_RAW_WEIGHT = -12.0


def unitree_g1_phase2c8g_second_velocity_control_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c8d_overspeed_penalty_env_cfg(play=play)
  cfg.rewards["command_direction_overspeed_l2"].weight = OVERSPEED_RAW_WEIGHT
  return cfg
