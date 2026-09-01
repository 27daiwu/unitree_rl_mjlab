"""Phase 2C.1 continuation environment: mild terrain only."""
from copy import deepcopy
from mjlab.envs import ManagerBasedRlEnvCfg
from .phase2c0_flat_sanity_env_cfg import unitree_g1_phase2c0_flat_sanity_env_cfg
from .phase2c1_mild_terrain_cfg import PHASE2C1_MILD_TERRAIN_CFG


def unitree_g1_phase2c1_mild_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = deepcopy(unitree_g1_phase2c0_flat_sanity_env_cfg(play=play))
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = deepcopy(PHASE2C1_MILD_TERRAIN_CFG)
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg
