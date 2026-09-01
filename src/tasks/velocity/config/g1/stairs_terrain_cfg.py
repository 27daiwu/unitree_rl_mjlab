"""Straight uphill staircase terrain for the Unitree G1 stairs task."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


STAIR_HEIGHT_BANDS: tuple[tuple[float, float], ...] = (
  (0.030, 0.050),
  (0.050, 0.070),
  (0.070, 0.090),
  (0.090, 0.110),
  (0.110, 0.130),
  (0.130, 0.145),
  (0.145, 0.160),
  (0.160, 0.175),
  (0.170, 0.185),
)
STAIR_FINAL_DEPTHS: tuple[float, ...] = (0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30)


@dataclass(frozen=True)
class StraightStairsGeometry:
  """Resolved geometry values for one curriculum difficulty."""

  staircase_start_x: float
  staircase_end_x: float
  step_height: float
  step_depth: float
  num_steps: int
  top_height: float
  top_platform_length: float
  spawn_x: float
  corridor_wall_thickness: float
  corridor_wall_height: float
  corridor_inner_y_min: float
  corridor_inner_y_max: float

  @property
  def corridor_outer_y_min(self) -> float:
    return self.corridor_inner_y_min - self.corridor_wall_thickness

  @property
  def corridor_outer_y_max(self) -> float:
    return self.corridor_inner_y_max + self.corridor_wall_thickness


@dataclass(kw_only=True)
class StraightUphillStairsTerrainCfg(SubTerrainCfg):
  """Flat approach followed by straight uphill stairs and a top platform."""

  approach_length: float = 2.0
  approach_spawn_x: float = 0.75
  step_height_bands: tuple[tuple[float, float], ...] = STAIR_HEIGHT_BANDS
  initial_step_depth: float = 0.35
  final_step_depth: float = 0.26
  num_steps: int = 8
  top_platform_min_length: float = 2.5
  floor_thickness: float = 0.10
  corridor_width: float = 4.0
  corridor_wall_thickness: float = 0.15
  corridor_wall_height: float = 0.80
  generated_geometries: list[StraightStairsGeometry] = field(
    default_factory=list, init=False, repr=False
  )

  def geometry_at(self, difficulty: float) -> StraightStairsGeometry:
    """Resolve staircase metadata without constructing MuJoCo geometry."""
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    scaled_band = difficulty * len(self.step_height_bands)
    band_index = min(int(scaled_band), len(self.step_height_bands) - 1)
    band_fraction = min(scaled_band - band_index, 1.0)
    height_min, height_max = self.step_height_bands[band_index]
    step_height = height_min + band_fraction * (height_max - height_min)
    step_depth = self.initial_step_depth + difficulty * (
      self.final_step_depth - self.initial_step_depth
    )
    staircase_end_x = self.approach_length + self.num_steps * step_depth
    top_platform_length = self.size[0] - staircase_end_x
    if top_platform_length < self.top_platform_min_length:
      raise ValueError(
        "Staircase does not leave the required top platform: "
        f"{top_platform_length:.3f} < {self.top_platform_min_length:.3f} m"
      )
    return StraightStairsGeometry(
      staircase_start_x=self.approach_length,
      staircase_end_x=staircase_end_x,
      step_height=step_height,
      step_depth=step_depth,
      num_steps=self.num_steps,
      top_height=self.num_steps * step_height,
      top_platform_length=top_platform_length,
      spawn_x=self.approach_spawn_x,
      corridor_wall_thickness=self.corridor_wall_thickness,
      corridor_wall_height=self.corridor_wall_height,
      corridor_inner_y_min=-self.corridor_width / 2,
      corridor_inner_y_max=self.corridor_width / 2,
    )

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # Geometry is deterministic for a given difficulty and column variant.
    geometry = self.geometry_at(difficulty)
    self.generated_geometries.append(geometry)
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    width = self.corridor_width
    patch_center_y = self.size[1] / 2
    required_width = width + 2 * self.corridor_wall_thickness
    if self.size[1] < required_width:
      raise ValueError(
        f"Terrain patch width {self.size[1]:.3f} is smaller than corridor plus walls {required_width:.3f}"
      )

    approach = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.approach_length / 2, width / 2, self.floor_thickness / 2),
      pos=(self.approach_length / 2, patch_center_y, -self.floor_thickness / 2),
    )
    geoms.append(TerrainGeometry(geom=approach))

    for step_index in range(geometry.num_steps):
      step_top = (step_index + 1) * geometry.step_height
      step_center_x = (
        geometry.staircase_start_x + (step_index + 0.5) * geometry.step_depth
      )
      step = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(geometry.step_depth / 2, width / 2, step_top / 2),
        pos=(step_center_x, patch_center_y, step_top / 2),
      )
      geoms.append(TerrainGeometry(geom=step))

    platform = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(geometry.top_platform_length / 2, width / 2, geometry.top_height / 2),
      pos=(
        geometry.staircase_end_x + geometry.top_platform_length / 2,
        patch_center_y,
        geometry.top_height / 2,
      ),
    )
    geoms.append(TerrainGeometry(geom=platform))

    # Keep the full 4 m usable corridor while preventing the learned policy
    # from escaping around the staircase.  Walls start at x=0 and continue
    # through the top platform, so there is no approach-side bypass.
    wall_length = self.size[0]
    wall_half = self.corridor_wall_thickness / 2
    wall_z = self.corridor_wall_height / 2
    for wall_y in (
      patch_center_y - width / 2 - wall_half,
      patch_center_y + width / 2 + wall_half,
    ):
      wall = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(wall_length / 2, wall_half, wall_z),
        pos=(wall_length / 2, wall_y, wall_z),
      )
      geoms.append(TerrainGeometry(geom=wall))

    return TerrainOutput(
      origin=np.array([geometry.spawn_x, patch_center_y, 0.0]),
      geometries=geoms,
    )


STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
  seed=0,
  curriculum=True,
  size=(8.0, 4.3),
  border_width=5.0,
  num_rows=len(STAIR_HEIGHT_BANDS),
  num_cols=len(STAIR_FINAL_DEPTHS),
  sub_terrains={
    f"straight_uphill_depth_{round(final_depth * 100):02d}": StraightUphillStairsTerrainCfg(
      proportion=1.0,
      size=(8.0, 4.3),
      final_step_depth=final_depth,
    )
    for final_depth in STAIR_FINAL_DEPTHS
  },
  difficulty_range=(0.0, 1.0),
  add_lights=True,
)


def make_stairs_terrain_cfg(play: bool = False) -> TerrainGeneratorCfg:
  """Return an isolated terrain config because mjlab mutates sub-terrain sizes."""
  cfg = copy.deepcopy(STAIRS_TERRAINS_CFG)
  if play:
    # Keep row/column metadata deterministic so stairs metrics can resolve the
    # exact compiled geometry in viewer rollouts as well as training.
    cfg.num_rows = 3
  return cfg
