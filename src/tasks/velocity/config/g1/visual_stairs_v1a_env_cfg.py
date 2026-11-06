"""G1 29-DoF indoor straight-stairs vision task, V1.0-A."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.terrains import TerrainGeneratorCfg

from .rl_cfg import unitree_g1_ppo_runner_cfg
from .stairs_baseline_env_cfg import unitree_g1_stairs_baseline_env_cfg
from .stairs_terrain_cfg import StraightUphillStairsTerrainCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


DEPTH_CAMERA_NAME = "d435_depth"
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 90
DEPTH_FOVY_DEG = 58.0


@lru_cache(maxsize=4)
def _optical_z_scale(device: torch.device, height: int, width: int) -> torch.Tensor:
  """Cosine of each pixel ray to the optical axis, matching MJWarp rays."""
  half_h = math.tan(math.radians(DEPTH_FOVY_DEG) / 2)
  x = ((torch.arange(width, device=device) + 0.5) / width * 2 - 1) * (
    half_h * width / height
  )
  y = (1 - (torch.arange(height, device=device) + 0.5) / height * 2) * half_h
  return torch.rsqrt(1 + y[:, None].square() + x[None, :].square())[None, None]


def raw_camera_depth(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Return D435-style optical-Z depth in metres, channel first."""
  sensor: CameraSensor = env.scene[sensor_name]
  depth = sensor.data.depth
  if depth is None:
    raise RuntimeError(f"Camera '{sensor_name}' has no depth data")
  ray_range = depth.permute(0, 3, 1, 2)
  return ray_range * _optical_z_scale(ray_range.device, depth.shape[1], depth.shape[2])


def unitree_g1_visual_stairs_v1a_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Use the fixed straight-stairs baseline with a head-mounted depth camera."""
  cfg = unitree_g1_stairs_baseline_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=0,
    curriculum=True,
    size=(8.0, 4.3),
    border_width=5.0,
    num_rows=1,
    num_cols=1,
    sub_terrains={
      "indoor_straight_uphill": StraightUphillStairsTerrainCfg(
        proportion=1.0,
        size=(8.0, 4.3),
        approach_length=2.0,
        step_height_bands=((0.16, 0.16),),
        initial_step_depth=0.30,
        final_step_depth=0.30,
        num_steps=8,
      )
    },
    difficulty_range=(0.0, 1.0),
    add_lights=True,
  )

  # The mounting transform follows the official G1 Rev 1.0 d435_link pose.
  # MuJoCo cameras look along local -Z with local +Y as image up.
  camera = CameraSensorCfg(
    name=DEPTH_CAMERA_NAME,
    parent_body="robot/torso_link",
    pos=(0.0576235, 0.01753, 0.42987),
    quat=(0.65925248, 0.25570719, -0.25570719, -0.65925248),
    fovy=DEPTH_FOVY_DEG,
    width=DEPTH_IMAGE_WIDTH,
    height=DEPTH_IMAGE_HEIGHT,
    data_types=("rgb", "depth"),
    use_textures=True,
    use_shadows=False,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera,)

  # The actor receives only proprioception, command, and optical-Z depth.
  # The critic intentionally retains the baseline's privileged height scan.
  del cfg.observations["actor"].terms["height_scan"]
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "image": ObservationTermCfg(
        func=raw_camera_depth,
        params={"sensor_name": DEPTH_CAMERA_NAME},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg


def unitree_g1_visual_stairs_v1a_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Fuse raw depth through a CNN with proprioception in the PPO actor."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor.class_name = "CNNModel"
  cfg.actor.cnn_cfg = {
    "output_channels": [16, 32, 32, 64],
    "kernel_size": [5, 3, 3, 3],
    "stride": [2, 2, 2, 2],
    "padding": "zeros",
    "activation": "elu",
    "global_pool": "none",
  }
  cfg.obs_groups = {"actor": ("actor", "depth"), "critic": ("critic",)}
  cfg.experiment_name = "g1_visual_stairs_v1a"
  return cfg
