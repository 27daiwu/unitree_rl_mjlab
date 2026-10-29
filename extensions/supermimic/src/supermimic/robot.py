"""G1 configuration using this repository's 29-DoF MJCF and upstream actuators."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    G1_ARTICULATION,
    KNEES_BENT_KEYFRAME,
)


G1_XML = (
    Path(__file__).resolve().parents[4]
    / "src/assets/robots/unitree_g1/xmls/g1.xml"
)


def _get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(G1_XML))


def get_g1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=_get_spec,
        articulation=G1_ARTICULATION,
    )
