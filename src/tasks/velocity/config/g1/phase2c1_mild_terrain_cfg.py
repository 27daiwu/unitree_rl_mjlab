"""Very-mild smooth terrain for Phase 2C.1."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import mujoco
import numpy as np

from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

_WALL_ID = itertools.count()


@dataclass(kw_only=True)
class MildWaveCorridorTerrainCfg(SubTerrainCfg):
  """Low-cost smooth slope <=1 cm peak-to-peak with corridor walls."""
  slope_range: tuple[float, float] = (0.0, 0.0)
  corridor_width: float = 4.0
  corridor_wall_thickness: float = 0.15
  corridor_wall_height: float = 0.80
  spawn_x: float = 0.75

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    center_y = self.size[1] / 2
    slope = self.slope_range[0] + float(np.clip(difficulty, 0.0, 1.0)) * (self.slope_range[1] - self.slope_range[0])
    # Keep the floor geom on the canonical `terrain` body so the existing
    # feet_ground_contact secondary matcher sees it.
    angle = float(np.arctan(slope))
    floor = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=(self.size[0] / 2, self.corridor_width / 2, 0.05),
                          pos=(self.size[0] / 2, center_y, slope * self.size[0] / 2 - 0.05),
                          quat=(float(np.cos(angle / 2)), 0.0, float(np.sin(angle / 2)), 0.0))
    output = TerrainOutput(origin=np.array([self.spawn_x, center_y, slope * self.spawn_x]), geometries=[TerrainGeometry(geom=floor)])
    wall_half = self.corridor_wall_thickness / 2
    walls: list[TerrainGeometry] = []
    for side, wall_y in (("left", center_y - self.corridor_width / 2 - wall_half),
                         ("right", center_y + self.corridor_width / 2 + wall_half)):
      wall_body = body.add_body(name=f"{side}_corridor_wall_{next(_WALL_ID)}")
      wall = wall_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=(self.size[0] / 2, wall_half, self.corridor_wall_height / 2),
                                pos=(self.size[0] / 2, wall_y, self.corridor_wall_height / 2))
      walls.append(TerrainGeometry(geom=wall))
    output.geometries.extend(walls)
    return output


PHASE2C1_MILD_TERRAIN_CFG = TerrainGeneratorCfg(
  seed=0,
  curriculum=True,
  size=(8.0, 4.3),
  border_width=5.0,
  num_rows=2,
  num_cols=2,
  sub_terrains={
    "flat_corridor": MildWaveCorridorTerrainCfg(
      proportion=0.5, size=(8.0, 4.3), slope_range=(0.0, 0.0),
    ),
    "mild_wave": MildWaveCorridorTerrainCfg(
      proportion=0.5, size=(8.0, 4.3), slope_range=(0.0, 0.0025),
    ),
  },
  difficulty_range=(0.0, 1.0),
  add_lights=True,
)
