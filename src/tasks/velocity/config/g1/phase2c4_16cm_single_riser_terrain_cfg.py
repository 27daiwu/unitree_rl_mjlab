"""Fixed 16 cm single-upward-riser terrain for Phase 2C.4-B continuation."""
from __future__ import annotations

from copy import deepcopy

from .phase2c3_7cm_single_riser_terrain_cfg import PHASE2C3_7CM_SINGLE_RISER_TERRAIN_CFG


PHASE2C4_16CM_SINGLE_RISER_TERRAIN_CFG = deepcopy(PHASE2C3_7CM_SINGLE_RISER_TERRAIN_CFG)
PHASE2C4_16CM_SINGLE_RISER_TERRAIN_CFG.sub_terrains["single_fixed_riser"].riser_height = 0.160
