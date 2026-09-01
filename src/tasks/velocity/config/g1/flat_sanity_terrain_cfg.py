"""Flat corridor terrain for Phase 2C.0 sanity training."""

from dataclasses import dataclass
import itertools

import mujoco
import numpy as np

from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

_WALL_ID = itertools.count()


@dataclass(kw_only=True)
class FlatCorridorTerrainCfg(SubTerrainCfg):
  corridor_width: float = 4.0
  floor_thickness: float = 0.10
  corridor_wall_thickness: float = 0.15
  corridor_wall_height: float = 0.80
  spawn_x: float = 0.75

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    center_y = self.size[1] / 2
    geoms: list[TerrainGeometry] = []
    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.corridor_width / 2, self.floor_thickness / 2),
      pos=(self.size[0] / 2, center_y, -self.floor_thickness / 2),
    )
    geoms.append(TerrainGeometry(geom=floor))
    wall_half = self.corridor_wall_thickness / 2
    for side, wall_y in (("left", center_y - self.corridor_width / 2 - wall_half), ("right", center_y + self.corridor_width / 2 + wall_half)):
      wall_body = body.add_body(name=f"{side}_corridor_wall_{next(_WALL_ID)}")
      wall = wall_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.size[0] / 2, wall_half, self.corridor_wall_height / 2),
        pos=(self.size[0] / 2, wall_y, self.corridor_wall_height / 2),
      )
      geoms.append(TerrainGeometry(geom=wall))
    return TerrainOutput(origin=np.array([self.spawn_x, center_y, 0.0]), geometries=geoms)


PHASE2C0_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
  seed=0,
  curriculum=True,
  size=(8.0, 4.3),
  border_width=5.0,
  num_rows=1,
  num_cols=1,
  sub_terrains={"flat_corridor": FlatCorridorTerrainCfg(proportion=1.0, size=(8.0, 4.3))},
  difficulty_range=(0.0, 0.0),
  add_lights=True,
)
