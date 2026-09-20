"""Phase 2C.4-B fixed 16 cm single-riser continuation environment."""
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from .phase2c3_7cm_single_riser_env_cfg import unitree_g1_phase2c3_7cm_single_riser_env_cfg
from .phase2c4_16cm_single_riser_terrain_cfg import PHASE2C4_16CM_SINGLE_RISER_TERRAIN_CFG


def unitree_g1_phase2c4_16cm_single_riser_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c3_7cm_single_riser_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = deepcopy(PHASE2C4_16CM_SINGLE_RISER_TERRAIN_CFG)
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg
