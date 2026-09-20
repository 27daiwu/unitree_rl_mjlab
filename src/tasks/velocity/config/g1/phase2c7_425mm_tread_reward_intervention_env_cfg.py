"""Phase 2C.7-D frozen-design config for a 0.425 m tread intervention."""
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.velocity.mdp as mdp
from .phase2c6_two_riser_env_cfg import unitree_g1_phase2c6_two_riser_env_cfg


def unitree_g1_phase2c7_425mm_tread_reward_intervention_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c6_two_riser_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  terrain_cfg = deepcopy(cfg.scene.terrain.terrain_generator)
  terrain_cfg.sub_terrains["two_fixed_riser"].intermediate_tread_depth = 0.425
  terrain_cfg.sub_terrains["two_fixed_riser"].generated_geometries.clear()
  cfg.scene.terrain.terrain_generator = terrain_cfg
  cfg.rewards["one_time_valid_intermediate_support_acquisition"] = RewardTermCfg(
    func=mdp.one_time_valid_intermediate_support_acquisition,
    weight=200.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
      "contact_sensor_name": "feet_ground_contact",
      "tolerance": 0.05,
      "footprint_margin": 0.03,
    },
  )
  return cfg
