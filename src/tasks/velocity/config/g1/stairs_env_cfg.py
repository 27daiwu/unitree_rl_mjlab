"""Unitree G1 straight-stairs velocity environment configuration."""

from dataclasses import dataclass

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg

import src.tasks.velocity.mdp as stairs_mdp

from .env_cfgs import unitree_g1_rough_env_cfg
from .stairs_terrain_cfg import make_stairs_terrain_cfg


@dataclass
class ForwardGridPatternCfg(GridPatternCfg):
  """Grid pattern shifted forward relative to its attached body frame."""

  x_offset: float = 0.45

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 0] += self.x_offset
    return offsets, directions


def unitree_g1_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Derive a G1 stairs task from the official G1 rough baseline."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = make_stairs_terrain_cfg(play=play)
  # MJWarp reports/omits contacts above this per-world capacity.  The G1
  # staircase task needs the tested 96-contact budget; Rough remains 48.
  cfg.sim.nconmax = 96

  cfg.metrics.update(
    {
      "stairs/x_progress_index": MetricsTermCfg(func=stairs_mdp.current_stair_index),
      "stairs/physical_stair_index": MetricsTermCfg(
        func=stairs_mdp.physical_stair_index,
        params={
          "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
          "contact_sensor_name": "feet_ground_contact",
          "height_tolerance": 0.05,
          "footprint_margin": 0.03,
        },
      ),
      "stairs/left_support_step": MetricsTermCfg(
        func=stairs_mdp.left_support_step,
        params={
          "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot",)),
          "contact_sensor_name": "feet_ground_contact",
          "height_tolerance": 0.05,
          "footprint_margin": 0.03,
        },
      ),
      "stairs/right_support_step": MetricsTermCfg(
        func=stairs_mdp.right_support_step,
        params={
          "asset_cfg": SceneEntityCfg("robot", site_names=("right_foot",)),
          "contact_sensor_name": "feet_ground_contact",
          "height_tolerance": 0.05,
          "footprint_margin": 0.03,
        },
      ),
    }
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern = ForwardGridPatternCfg(
        size=(1.5, 1.0),
        resolution=0.1,
        x_offset=0.45,
      )

  # The terrain origin is on the flat approach. Keep reset jitter away from the stairs.
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (-0.15, 0.15),
    "y": (-0.25, 0.25),
    "z": (0.0, 0.0),
    "yaw": (-0.10, 0.10),
  }

  return cfg
