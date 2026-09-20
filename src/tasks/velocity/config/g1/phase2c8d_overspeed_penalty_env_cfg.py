"""Phase 2C.8-D isolated command-aware overspeed intervention config."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

import src.tasks.velocity.mdp as mdp

from .phase2c8a_mixed_tread_env_cfg import unitree_g1_phase2c8a_mixed_tread_env_cfg


OVERSPEED_TOLERANCE = 0.10
OVERSPEED_RAW_WEIGHT = -6.0
MIN_COMMAND_SPEED = 0.05


def unitree_g1_phase2c8d_overspeed_penalty_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c8a_mixed_tread_env_cfg(play=play)
  cfg.rewards["command_direction_overspeed_l2"] = RewardTermCfg(
    func=mdp.command_direction_overspeed_l2,
    weight=OVERSPEED_RAW_WEIGHT,
    params={
      "command_name": "twist",
      "tolerance": OVERSPEED_TOLERANCE,
      "min_command_speed": MIN_COMMAND_SPEED,
    },
  )
  return cfg
