"""Phase 2C.8-J reverse-motion safeguard pilot configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

import src.tasks.velocity.mdp as mdp

from .phase2c8g_second_velocity_control_env_cfg import (
  unitree_g1_phase2c8g_second_velocity_control_env_cfg,
)


REVERSE_TOLERANCE = 0.05


def unitree_g1_phase2c8j_reverse_safeguard_env_cfg(
  play: bool = False,
  reverse_weight: float = -1.0,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c8g_second_velocity_control_env_cfg(play=play)
  cfg.rewards["command_direction_reverse_velocity_l2"] = RewardTermCfg(
    func=mdp.command_direction_reverse_velocity_l2,
    weight=reverse_weight,
    params={"command_name": "twist", "tolerance": REVERSE_TOLERANCE,
            "min_command_speed": 0.05},
  )
  return cfg
