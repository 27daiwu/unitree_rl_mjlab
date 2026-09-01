"""Phase 2A low-stairs baseline environment configuration."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainGeneratorCfg

import src.tasks.velocity.mdp as mdp

from .stairs_env_cfg import unitree_g1_stairs_env_cfg
from .stairs_terrain_cfg import StraightUphillStairsTerrainCfg


STAIRS_BASELINE_TERRAIN_CFG = TerrainGeneratorCfg(
  seed=0,
  # Curriculum layout gives an exact row/column-to-metadata mapping. The env
  # curriculum manager is disabled, so terrain levels never upgrade.
  curriculum=True,
  size=(8.0, 4.3),
  border_width=5.0,
  num_rows=5,
  num_cols=5,
  sub_terrains={
    "straight_uphill_low": StraightUphillStairsTerrainCfg(
      proportion=1.0,
      size=(8.0, 4.3),
      step_height_bands=((0.04, 0.06),),
      initial_step_depth=0.35,
      final_step_depth=0.30,
      num_steps=8,
    )
  },
  difficulty_range=(0.0, 1.0),
  add_lights=True,
)


def make_stairs_baseline_terrain_cfg() -> TerrainGeneratorCfg:
  return deepcopy(STAIRS_BASELINE_TERRAIN_CFG)


def unitree_g1_stairs_baseline_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Derive the isolated Phase 2A baseline from Unitree-G1-Stairs."""
  cfg = unitree_g1_stairs_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = make_stairs_baseline_terrain_cfg()
  cfg.scene.terrain.max_init_terrain_level = None

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.ranges.lin_vel_x = (0.30, 0.50)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  command.ranges.heading = None
  command.heading_command = False
  command.rel_standing_envs = 0.0

  # Both terrain and command distributions are deliberately fixed in Phase 2A.
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)
  cfg.events.pop("randomize_terrain", None)

  cfg.metrics.update(
    {
      "stairs/root_x_progress": MetricsTermCfg(func=mdp.root_x_progress),
      "stairs/root_z": MetricsTermCfg(func=mdp.root_z),
      "stairs/reached_top_platform": MetricsTermCfg(
        func=mdp.reached_top_platform,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_foot", "right_foot")
          ),
          "contact_sensor_name": "feet_ground_contact",
          "foot_height_tolerance": 0.05,
          "footprint_margin": 0.03,
        },
      ),
      "stairs/fell_over": MetricsTermCfg(func=mdp.fell_over),
      "stairs/episode_timeout": MetricsTermCfg(func=mdp.episode_timeout),
      "stairs/forward_velocity": MetricsTermCfg(func=mdp.forward_velocity),
      "stairs/minimum_foot_height": MetricsTermCfg(
        func=mdp.minimum_foot_height,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_foot", "right_foot")
          )
        },
      ),
      "stairs/maximum_foot_height": MetricsTermCfg(
        func=mdp.maximum_foot_height,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_foot", "right_foot")
          )
        },
      ),
      "stairs/left_foot_contact": MetricsTermCfg(
        func=mdp.single_foot_contact,
        params={"sensor_name": "feet_ground_contact", "foot_index": 0},
      ),
      "stairs/right_foot_contact": MetricsTermCfg(
        func=mdp.single_foot_contact,
        params={"sensor_name": "feet_ground_contact", "foot_index": 1},
      ),
    }
  )

  return cfg
