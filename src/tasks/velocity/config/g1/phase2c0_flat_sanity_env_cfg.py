"""Phase 2C.0 flat-ground sanity environment.

This is a deep copy of the Phase 2A.7 environment.  Terrain geometry is the
only training-side change; the existing plane terrain implementation is used.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from .stairs_baseline_env_cfg import unitree_g1_stairs_baseline_env_cfg
from .flat_sanity_terrain_cfg import PHASE2C0_FLAT_TERRAIN_CFG


STAIRS_METRIC_PREFIX = "stairs/"


def unitree_g1_phase2c0_flat_sanity_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = deepcopy(unitree_g1_stairs_baseline_env_cfg(play=play))
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = deepcopy(PHASE2C0_FLAT_TERRAIN_CFG)
  cfg.scene.terrain.max_init_terrain_level = None

  # Compatibility exception: stairs metrics require generated staircase
  # metadata and are evaluation-only, so they have no defined value on a plane.
  cfg.metrics = {
    name: term for name, term in cfg.metrics.items()
    if not name.startswith(STAIRS_METRIC_PREFIX)
  }
  return cfg
