"""Diagnostic two-riser terrain for Phase 2C.6 zero-shot repetition audit."""
from __future__ import annotations

from dataclasses import dataclass, field
import mujoco
import numpy as np
from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


@dataclass(frozen=True)
class TwoRiserGeometry:
  staircase_start_x: float
  staircase_end_x: float
  step_height: float
  step_depth: float
  num_steps: int
  top_height: float
  top_platform_length: float
  riser_1_x: float
  riser_2_x: float
  riser_1_height: float
  riser_2_increment: float
  intermediate_platform_z: float
  final_platform_z: float
  intermediate_tread_depth: float
  intermediate_platform_length: float
  final_platform_length: float
  spawn_x: float
  corridor_wall_thickness: float
  corridor_wall_height: float
  corridor_inner_y_min: float
  corridor_inner_y_max: float


@dataclass(kw_only=True)
class TwoFixedRiserTerrainCfg(SubTerrainCfg):
  approach_length: float = 1.9
  approach_spawn_x: float = 0.75
  riser_1_height: float = 0.175
  riser_2_increment: float = 0.175
  intermediate_tread_depth: float = 0.600
  corridor_width: float = 4.0
  corridor_wall_thickness: float = 0.15
  corridor_wall_height: float = 0.80
  floor_thickness: float = 0.10
  generated_geometries: list[TwoRiserGeometry] = field(default_factory=list, init=False, repr=False)

  def geometry_at(self, difficulty: float = 0.0) -> TwoRiserGeometry:
    del difficulty
    r1 = self.approach_length
    r2 = r1 + self.intermediate_tread_depth
    final_z = self.riser_1_height + self.riser_2_increment
    return TwoRiserGeometry(
      staircase_start_x=r1, staircase_end_x=r2, step_height=self.riser_1_height,
      step_depth=self.intermediate_tread_depth, num_steps=2, top_height=final_z,
      top_platform_length=self.size[0] - r2,
      riser_1_x=r1, riser_2_x=r2, riser_1_height=self.riser_1_height,
      riser_2_increment=self.riser_2_increment,
      intermediate_platform_z=self.riser_1_height, final_platform_z=final_z,
      intermediate_tread_depth=self.intermediate_tread_depth,
      intermediate_platform_length=self.intermediate_tread_depth,
      final_platform_length=self.size[0] - r2,
      spawn_x=self.approach_spawn_x,
      corridor_wall_thickness=self.corridor_wall_thickness,
      corridor_wall_height=self.corridor_wall_height,
      corridor_inner_y_min=-self.corridor_width / 2,
      corridor_inner_y_max=self.corridor_width / 2,
    )

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del rng
    g = self.geometry_at(difficulty); self.generated_geometries.append(g)
    body = spec.body("terrain"); cy, width = self.size[1] / 2, self.corridor_width
    geoms: list[TerrainGeometry] = []
    suffix = f"{int(round(self.intermediate_tread_depth * 1000)):03d}mm"

    def box(name, sx, cx, top):
      geom = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(sx / 2, width / 2, max(top, self.floor_thickness) / 2),
        pos=(cx, cy, top / 2 - (0 if top > 0 else self.floor_thickness / 2)))
      geoms.append(TerrainGeometry(geom=geom))
    box(f"two_riser_lower_floor_{suffix}", self.approach_length, self.approach_length / 2, 0.0)
    box(f"two_riser_intermediate_platform_{suffix}", g.intermediate_platform_length,
        g.riser_1_x + g.intermediate_platform_length / 2, g.intermediate_platform_z)
    box(f"two_riser_final_platform_{suffix}", g.final_platform_length,
        g.riser_2_x + g.final_platform_length / 2, g.final_platform_z)
    half = self.corridor_wall_thickness / 2
    for side, wy in (("left", cy - width / 2 - half), ("right", cy + width / 2 + half)):
      wb = body.add_body(name=f"{side}_corridor_wall_two_riser_{suffix}")
      geom = wb.add_geom(name=f"{side}_corridor_wall_geom_two_riser_{suffix}", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.size[0] / 2, half, self.corridor_wall_height / 2),
        pos=(self.size[0] / 2, wy, self.corridor_wall_height / 2))
      geoms.append(TerrainGeometry(geom=geom))
    return TerrainOutput(origin=np.array([g.spawn_x, cy, 0.0]), geometries=geoms)


PHASE2C6_TWO_RISER_TERRAIN_CFG = TerrainGeneratorCfg(
  seed=0, curriculum=True, size=(8.0, 4.3), border_width=5.0,
  num_rows=1, num_cols=1,
  sub_terrains={"two_fixed_riser": TwoFixedRiserTerrainCfg(
    proportion=1.0, size=(8.0, 4.3), riser_1_height=0.175,
    riser_2_increment=0.175, intermediate_tread_depth=0.600,
  )}, difficulty_range=(0.0, 0.0), add_lights=True,
)
