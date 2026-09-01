"""Read-only metrics for straight uphill staircase evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from src.tasks.velocity.config.g1.stairs_terrain_cfg import StraightStairsGeometry


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _geometry_for_envs(env: ManagerBasedRlEnv) -> list[Any]:
  """Return the exact compiled geometry for every environment's terrain patch."""
  terrain = env.scene.terrain
  assert terrain is not None
  generator_cfg = terrain.cfg.terrain_generator
  if generator_cfg is None or not generator_cfg.curriculum:
    raise RuntimeError("Stairs metrics require a curriculum terrain grid.")

  stairs_cfgs = list(generator_cfg.sub_terrains.values())
  proportions = [cfg.proportion for cfg in stairs_cfgs]
  proportion_sum = sum(proportions)
  cumulative = []
  running = 0.0
  for proportion in proportions:
    running += proportion / proportion_sum
    cumulative.append(running)

  # Match mjlab TerrainGenerator._generate_curriculum_terrains exactly, while
  # also tracking each config's column ordinal within its metadata list.
  column_lookup: list[tuple[Any, int]] = []
  config_column_counts = [0] * len(stairs_cfgs)
  for column in range(generator_cfg.num_cols):
    fraction = column / generator_cfg.num_cols + 0.001
    config_index = next(
      index for index, threshold in enumerate(cumulative) if fraction < threshold
    )
    stairs_cfg = stairs_cfgs[config_index]
    column_lookup.append((stairs_cfg, config_column_counts[config_index]))
    config_column_counts[config_index] += 1

  for stairs_cfg in stairs_cfgs:
    if not hasattr(stairs_cfg, "generated_geometries"):
      raise TypeError("Stairs sub-terrain does not expose compiled geometry metadata.")
    if not stairs_cfg.generated_geometries:
      raise RuntimeError("Stairs terrain geometry has not been compiled yet.")

  num_rows = generator_cfg.num_rows
  levels = terrain.terrain_levels.detach().cpu().tolist()
  columns = terrain.terrain_types.detach().cpu().tolist()
  geometries = []
  for level, column in zip(levels, columns, strict=True):
    stairs_cfg, config_column = column_lookup[column]
    geometries.append(
      stairs_cfg.generated_geometries[config_column * num_rows + level]
    )
  return geometries


def geometry_tensors(
  env: ManagerBasedRlEnv,
) -> dict[str, torch.Tensor]:
  """Convert exact per-patch staircase metadata to device tensors."""
  geometries = _geometry_for_envs(env)
  names = (
    "staircase_start_x",
    "staircase_end_x",
    "step_height",
    "step_depth",
    "num_steps",
    "top_height",
    "top_platform_length",
    "spawn_x",
    "corridor_inner_y_min",
    "corridor_inner_y_max",
  )
  return {
    name: torch.tensor(
      [getattr(geometry, name) for geometry in geometries],
      device=env.device,
      dtype=torch.float32,
    )
    for name in names
  }


def root_x_progress(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]


def root_z(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def current_stair_index(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  geometry = geometry_tensors(env)
  patch_x = root_x_progress(env, asset_cfg) + geometry["spawn_x"]
  index = torch.floor(
    (patch_x - geometry["staircase_start_x"]) / geometry["step_depth"]
  ) + 1.0
  return torch.clamp(index, min=0.0).minimum(geometry["num_steps"])


def _support_steps(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  contact_sensor_name: str,
  height_tolerance: float,
  footprint_margin: float,
) -> torch.Tensor:
  """Return current physically supported step for left and right feet.

  A foot is valid only when its site is inside one tread's XY footprint, its
  world z is within the tread-height tolerance, and the terrain contact sensor
  reports supporting contact.  The geometry is resolved per terrain patch.
  """
  asset: Entity = env.scene[asset_cfg.name]
  geometry = geometry_tensors(env)
  feet_sensor: ContactSensor = env.scene[contact_sensor_name]
  assert feet_sensor.data.found is not None
  positions = asset.data.site_pos_w[:, asset_cfg.site_ids]
  local = positions - env.scene.env_origins.unsqueeze(1)
  patch_x = local[..., 0] + geometry["spawn_x"].unsqueeze(1)
  patch_y = local[..., 1]
  step_height = geometry["step_height"].unsqueeze(1)
  step_depth = geometry["step_depth"].unsqueeze(1)
  start_x = geometry["staircase_start_x"].unsqueeze(1)
  raw = torch.floor((patch_x - start_x) / step_depth) + 1.0
  candidate = torch.clamp(raw, min=1.0).minimum(geometry["num_steps"].unsqueeze(1))
  tread_z = candidate * step_height
  inside_xy = (
    (patch_x >= start_x + footprint_margin)
    & (patch_x <= geometry["staircase_end_x"].unsqueeze(1) - footprint_margin)
    & ((patch_x - start_x) % step_depth >= footprint_margin)
    & ((patch_x - start_x) % step_depth <= step_depth - footprint_margin)
    & (patch_y >= geometry["corridor_inner_y_min"].unsqueeze(1) + footprint_margin)
    & (patch_y <= geometry["corridor_inner_y_max"].unsqueeze(1) - footprint_margin)
  )
  if asset_cfg.site_names == ("right_foot",):
    contact = feet_sensor.data.found[:, 1:2]
  elif asset_cfg.site_names == ("left_foot", "right_foot"):
    contact = feet_sensor.data.found[:, :2]
  else:
    contact = feet_sensor.data.found[:, :1]
  supported = (
    inside_xy
    & (torch.abs(positions[..., 2] - env.scene.env_origins[:, 2].unsqueeze(1) - tread_z) <= height_tolerance)
    & (contact > 0)
  )
  return torch.where(supported, candidate, torch.zeros_like(candidate))


def left_support_step(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  contact_sensor_name: str = "feet_ground_contact",
  height_tolerance: float = 0.05,
  footprint_margin: float = 0.03,
) -> torch.Tensor:
  return _support_steps(
    env, asset_cfg, contact_sensor_name, height_tolerance, footprint_margin
  )[:, 0]


def right_support_step(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  contact_sensor_name: str = "feet_ground_contact",
  height_tolerance: float = 0.05,
  footprint_margin: float = 0.03,
) -> torch.Tensor:
  return _support_steps(
    env, asset_cfg, contact_sensor_name, height_tolerance, footprint_margin
  )[:, 0]


class physical_stair_index:
  """Highest stair with verified foot support during the episode."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.maximum = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_name: str = "feet_ground_contact",
    height_tolerance: float = 0.05,
    footprint_margin: float = 0.03,
  ) -> torch.Tensor:
    support = _support_steps(
      env, asset_cfg, contact_sensor_name, height_tolerance, footprint_margin
    )
    self.maximum = torch.maximum(self.maximum, support.max(dim=1).values)
    return self.maximum

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.maximum[env_ids if env_ids is not None else slice(None)] = 0.0


class maximum_stair_index:
  """Track the maximum stair index reached during the current episode."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.maximum = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    self.maximum = torch.maximum(self.maximum, current_stair_index(env, asset_cfg))
    return self.maximum

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.maximum[env_ids if env_ids is not None else slice(None)] = 0.0


def reached_top_platform(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  contact_sensor_name: str = "feet_ground_contact",
  platform_margin: float = 0.15,
  root_height_tolerance: float = 0.20,
  foot_height_tolerance: float = 0.05,
  footprint_margin: float = 0.03,
) -> torch.Tensor:
  """Detect entry onto the top platform with at least one supported foot."""
  asset: Entity = env.scene[asset_cfg.name]
  geometry = geometry_tensors(env)
  local_root = asset.data.root_link_pos_w - env.scene.env_origins
  root_patch_x = local_root[:, 0] + geometry["spawn_x"]
  platform_end_x = geometry["staircase_end_x"] + geometry["top_platform_length"]
  nominal_root_height = float(asset.cfg.init_state.pos[2])
  expected_root_z = geometry["top_height"] + nominal_root_height
  height_ok = (
    torch.abs(root_z(env, asset_cfg) - expected_root_z) <= root_height_tolerance
  )

  foot_positions = asset.data.site_pos_w[:, asset_cfg.site_ids]
  local_feet = foot_positions - env.scene.env_origins.unsqueeze(1)
  foot_patch_x = local_feet[..., 0] + geometry["spawn_x"].unsqueeze(1)
  foot_inside = (
    (foot_patch_x >= geometry["staircase_end_x"].unsqueeze(1) + footprint_margin)
    & (foot_patch_x <= platform_end_x.unsqueeze(1) - footprint_margin)
    & (
      local_feet[..., 1]
      >= geometry["corridor_inner_y_min"].unsqueeze(1) + footprint_margin
    )
    & (
      local_feet[..., 1]
      <= geometry["corridor_inner_y_max"].unsqueeze(1) - footprint_margin
    )
    & (
      torch.abs(local_feet[..., 2] - geometry["top_height"].unsqueeze(1))
      <= foot_height_tolerance
    )
  )
  feet_sensor: ContactSensor = env.scene[contact_sensor_name]
  assert feet_sensor.data.found is not None
  supported_foot = (foot_inside & (feet_sensor.data.found[:, :2] > 0)).any(dim=1)

  not_fallen = ~env.termination_manager.get_term("fell_over")
  return (
    (root_patch_x >= geometry["staircase_end_x"] + platform_margin)
    & (root_patch_x <= platform_end_x - platform_margin)
    & (local_root[:, 1] >= geometry["corridor_inner_y_min"] + platform_margin)
    & (local_root[:, 1] <= geometry["corridor_inner_y_max"] - platform_margin)
    & height_ok
    & supported_foot
    & not_fallen
  ).float()


def fell_over(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.get_term("fell_over").float()


def episode_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.get_term("time_out").float()


def forward_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 0]


class minimum_foot_height:
  """Track the minimum foot height seen during the current episode."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.minimum = torch.full((env.num_envs,), torch.inf, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    value = torch.min(asset.data.site_pos_w[:, asset_cfg.site_ids, 2], dim=1).values
    self.minimum = torch.minimum(self.minimum, value)
    return self.minimum

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.minimum[env_ids if env_ids is not None else slice(None)] = torch.inf


class maximum_foot_height:
  """Track the maximum foot height seen during the current episode."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.maximum = torch.full((env.num_envs,), -torch.inf, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    value = torch.max(asset.data.site_pos_w[:, asset_cfg.site_ids, 2], dim=1).values
    self.maximum = torch.maximum(self.maximum, value)
    return self.maximum

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.maximum[env_ids if env_ids is not None else slice(None)] = -torch.inf


def single_foot_contact(
  env: ManagerBasedRlEnv, sensor_name: str, foot_index: int
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found[:, foot_index] > 0).float()

def _single_riser_geometry(env):
  return geometry_tensors(env)

def first_riser_interaction(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
                            interaction_margin: float = 0.12) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  g = _single_riser_geometry(env)
  x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0] + g["spawn_x"]
  return ((x >= g["staircase_start_x"] - interaction_margin) &
          (x <= g["staircase_start_x"] + interaction_margin)).float()

def upper_platform_foot_support(env, asset_cfg: SceneEntityCfg,
                                contact_sensor_name: str = "feet_ground_contact",
                                tolerance: float = 0.05) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]; g = _single_riser_geometry(env)
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids] - env.scene.env_origins.unsqueeze(1)
  inside = ((feet[..., 0] + g["spawn_x"].unsqueeze(1) >= g["staircase_end_x"].unsqueeze(1) + 0.03) &
            (feet[..., 0] + g["spawn_x"].unsqueeze(1) <= (g["staircase_end_x"] + g["top_platform_length"]).unsqueeze(1) - 0.03) &
            (torch.abs(feet[..., 2] - g["top_height"].unsqueeze(1)) <= tolerance))
  sensor: ContactSensor = env.scene[contact_sensor_name]
  return (inside & (sensor.data.found[:, :2] > 0)).float().sum(dim=1).clamp(max=2)

def both_feet_upper_platform(env, asset_cfg: SceneEntityCfg,
                             contact_sensor_name: str = "feet_ground_contact") -> torch.Tensor:
  return (upper_platform_foot_support(env, asset_cfg, contact_sensor_name) >= 2).float()

def single_riser_success(env, asset_cfg: SceneEntityCfg,
                         contact_sensor_name: str = "feet_ground_contact") -> torch.Tensor:
  g = _single_riser_geometry(env)
  asset: Entity = env.scene[asset_cfg.name]
  root = asset.data.root_link_pos_w - env.scene.env_origins
  crossed = root[:, 0] + g["spawn_x"] >= g["staircase_end_x"] + 0.15
  support = both_feet_upper_platform(env, asset_cfg, contact_sensor_name) > 0
  height = torch.abs(root[:, 2] - (g["top_height"] + float(asset.cfg.init_state.pos[2]))) <= 0.20
  not_fallen = ~env.termination_manager.get_term("fell_over")
  return (crossed & support & height & not_fallen).float()
