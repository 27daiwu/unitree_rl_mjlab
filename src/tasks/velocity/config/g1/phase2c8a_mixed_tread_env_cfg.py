"""Phase 2C.8-A mixed-tread robustness pilot configuration."""
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from .phase2c7_425mm_tread_reward_intervention_env_cfg import (
  unitree_g1_phase2c7_425mm_tread_reward_intervention_env_cfg,
)
from .phase2c6_two_riser_terrain_cfg import TwoFixedRiserTerrainCfg


MIXED_TREAD_ANCHORS = (0.260, 0.280, 0.320, 0.425, 0.600)


def unitree_g1_phase2c8a_mixed_tread_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c7_425mm_tread_reward_intervention_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  terrain_cfg = deepcopy(cfg.scene.terrain.terrain_generator)
  terrain_cfg.curriculum = True
  terrain_cfg.num_rows = 1
  terrain_cfg.num_cols = len(MIXED_TREAD_ANCHORS)
  terrain_cfg.sub_terrains = {
    f"two_fixed_riser_{int(anchor * 1000):03d}mm": TwoFixedRiserTerrainCfg(
      proportion=1.0 / len(MIXED_TREAD_ANCHORS),
      size=terrain_cfg.size,
      riser_1_height=0.175,
      riser_2_increment=0.175,
      intermediate_tread_depth=anchor,
    )
    for anchor in MIXED_TREAD_ANCHORS
  }
  cfg.scene.terrain.terrain_generator = terrain_cfg
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg
