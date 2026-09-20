"""Phase 2C.3-B fixed 7 cm single-riser continuation environment."""
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensorCfg

import src.tasks.velocity.mdp as mdp
from .phase2c1_mild_env_cfg import unitree_g1_phase2c1_mild_env_cfg
from .phase2c2_single_riser_env_cfg import ForwardGridPatternCfg
from .phase2c3_7cm_single_riser_terrain_cfg import PHASE2C3_7CM_SINGLE_RISER_TERRAIN_CFG


def unitree_g1_phase2c3_7cm_single_riser_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # Inherit the closed C1-R objective and C2 sensor/metric contract. The only
  # terrain difference is riser_height=upper_platform_height=0.070 m.
  cfg = unitree_g1_phase2c1_mild_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = deepcopy(PHASE2C3_7CM_SINGLE_RISER_TERRAIN_CFG)
  cfg.scene.terrain.max_init_terrain_level = None
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern = ForwardGridPatternCfg(size=(1.5, 1.0), resolution=0.1, x_offset=0.45)
  cfg.metrics.update({
    "stairs/first_riser_interaction": MetricsTermCfg(func=mdp.first_riser_interaction),
    "stairs/upper_platform_foot_support": MetricsTermCfg(
      func=mdp.upper_platform_foot_support,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
    ),
    "stairs/both_feet_upper_platform": MetricsTermCfg(
      func=mdp.both_feet_upper_platform,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
    ),
    "stairs/single_riser_success": MetricsTermCfg(
      func=mdp.single_riser_success,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
    ),
  })
  return cfg
