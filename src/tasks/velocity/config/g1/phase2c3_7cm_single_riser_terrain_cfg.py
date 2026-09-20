"""Fixed 7 cm single-upward-riser terrain for Phase 2C.3-B continuation."""
from __future__ import annotations

from copy import deepcopy

from .phase2c2_single_riser_terrain_cfg import PHASE2C2_SINGLE_RISER_TERRAIN_CFG


# Keep the Phase 2C.2 geometry generator byte-for-byte equivalent except for
# the one authorized terrain parameter change: riser/platform height.
PHASE2C3_7CM_SINGLE_RISER_TERRAIN_CFG = deepcopy(PHASE2C2_SINGLE_RISER_TERRAIN_CFG)
PHASE2C3_7CM_SINGLE_RISER_TERRAIN_CFG.sub_terrains["single_fixed_riser"].riser_height = 0.070
