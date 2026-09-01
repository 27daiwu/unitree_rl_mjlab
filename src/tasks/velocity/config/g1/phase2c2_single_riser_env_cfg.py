"""Phase 2C.2 fixed single-riser environment (no training changes)."""
from copy import deepcopy
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
import src.tasks.velocity.mdp as mdp
from .phase2c1_mild_env_cfg import unitree_g1_phase2c1_mild_env_cfg
from .stairs_env_cfg import ForwardGridPatternCfg
from mjlab.sensor import RayCastSensorCfg
from .phase2c2_single_riser_terrain_cfg import PHASE2C2_SINGLE_RISER_TERRAIN_CFG

def unitree_g1_phase2c2_single_riser_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # Start from the closed C1-R environment so command, DR, reset, termination,
  # reward, actuator and simulation contracts remain identical. Only terrain
  # geometry, terrain-aware scan placement, and read-only metrics are added.
  cfg = unitree_g1_phase2c1_mild_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = deepcopy(PHASE2C2_SINGLE_RISER_TERRAIN_CFG)
  cfg.scene.terrain.max_init_terrain_level = None
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern = ForwardGridPatternCfg(size=(1.5, 1.0), resolution=0.1, x_offset=0.45)
  cfg.metrics.update({
    "stairs/first_riser_interaction": MetricsTermCfg(func=mdp.first_riser_interaction),
    "stairs/upper_platform_foot_support": MetricsTermCfg(func=mdp.upper_platform_foot_support,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))}),
    "stairs/both_feet_upper_platform": MetricsTermCfg(func=mdp.both_feet_upper_platform,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))}),
    "stairs/single_riser_success": MetricsTermCfg(func=mdp.single_riser_success,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))}),
  })
  return cfg
