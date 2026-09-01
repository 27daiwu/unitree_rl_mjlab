"""Fixed single-upward-riser terrain for Phase 2C.2 capability audit."""
from __future__ import annotations

from dataclasses import dataclass, field
import mujoco
import numpy as np

from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


@dataclass(frozen=True)
class SingleRiserGeometry:
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


@dataclass(kw_only=True)
class SingleFixedRiserTerrainCfg(SubTerrainCfg):
  """Flat approach, one 20 mm vertical riser, and a long upper platform."""
  approach_length: float = 1.9
  approach_spawn_x: float = 0.75
  riser_height: float = 0.020
  step_depth: float = 0.30
  corridor_width: float = 4.0
  corridor_wall_thickness: float = 0.15
  corridor_wall_height: float = 0.80
  floor_thickness: float = 0.10
  generated_geometries: list[SingleRiserGeometry] = field(default_factory=list, init=False, repr=False)

  def geometry_at(self, difficulty: float = 0.0) -> SingleRiserGeometry:
    del difficulty
    end = self.approach_length
    return SingleRiserGeometry(
      staircase_start_x=end, staircase_end_x=end,
      step_height=self.riser_height, step_depth=self.step_depth,
      num_steps=1, top_height=self.riser_height,
      top_platform_length=self.size[0] - end,
      spawn_x=self.approach_spawn_x,
      corridor_wall_thickness=self.corridor_wall_thickness,
      corridor_wall_height=self.corridor_wall_height,
      corridor_inner_y_min=-self.corridor_width / 2,
      corridor_inner_y_max=self.corridor_width / 2,
    )

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del rng
    geometry = self.geometry_at(difficulty)
    self.generated_geometries.append(geometry)
    body = spec.body("terrain")
    cy, width = self.size[1] / 2, self.corridor_width
    geoms: list[TerrainGeometry] = []
    lower = body.add_geom(name="single_riser_lower_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.approach_length / 2, width / 2, self.floor_thickness / 2),
      pos=(self.approach_length / 2, cy, -self.floor_thickness / 2))
    geoms.append(TerrainGeometry(geom=lower))
    upper_len = geometry.top_platform_length
    upper = body.add_geom(name="single_riser_upper_platform", type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(upper_len / 2, width / 2, self.floor_thickness / 2),
      pos=(geometry.staircase_end_x + upper_len / 2, cy,
           geometry.top_height - self.floor_thickness / 2))
    geoms.append(TerrainGeometry(geom=upper))
    half = self.corridor_wall_thickness / 2
    for side, wy in (("left", cy - width / 2 - half), ("right", cy + width / 2 + half)):
      wall_body = body.add_body(name=f"{side}_corridor_wall_single_riser")
      wall = wall_body.add_geom(name=f"{side}_corridor_wall_geom_single_riser", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.size[0] / 2, half, self.corridor_wall_height / 2),
        pos=(self.size[0] / 2, wy, self.corridor_wall_height / 2))
      geoms.append(TerrainGeometry(geom=wall))
    return TerrainOutput(origin=np.array([geometry.spawn_x, cy, 0.0]), geometries=geoms)


PHASE2C2_SINGLE_RISER_TERRAIN_CFG = TerrainGeneratorCfg(
  seed=0, curriculum=True, size=(8.0, 4.3), border_width=5.0,
  num_rows=1, num_cols=1,
  sub_terrains={"single_fixed_riser": SingleFixedRiserTerrainCfg(
    proportion=1.0, size=(8.0, 4.3), riser_height=0.020, step_depth=0.30,
  )}, difficulty_range=(0.0, 0.0), add_lights=True,
)
