"""Phase 2C.7-H fixed 0.280 m near-frontier continuation config."""
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from .phase2c7_425mm_tread_reward_intervention_env_cfg import (
  unitree_g1_phase2c7_425mm_tread_reward_intervention_env_cfg,
)


def unitree_g1_phase2c7h_280mm_near_frontier_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c7_425mm_tread_reward_intervention_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  terrain_cfg = deepcopy(cfg.scene.terrain.terrain_generator)
  terrain_cfg.sub_terrains["two_fixed_riser"].intermediate_tread_depth = 0.280
  terrain_cfg.sub_terrains["two_fixed_riser"].generated_geometries.clear()
  cfg.scene.terrain.terrain_generator = terrain_cfg
  return cfg
