"""Phase 2C.4-D single-variable reward-intervention experiment config."""
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.velocity.mdp as mdp
from .phase2c4_16cm_single_riser_env_cfg import unitree_g1_phase2c4_16cm_single_riser_env_cfg


def unitree_g1_phase2c4_16cm_reward_intervention_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c4_16cm_single_riser_env_cfg(play=play)
  cfg.rewards["one_time_bilateral_upper_platform_acquisition"] = RewardTermCfg(
    func=mdp.one_time_bilateral_upper_platform_acquisition,
    weight=300.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
      "contact_sensor_name": "feet_ground_contact",
      "tolerance": 0.05,
    },
  )
  return cfg
