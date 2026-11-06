"""Render a static G1 D435 stair view and audit its depth geometry.

This uses CPU MuJoCo rendering, independently of the training environment.
Run with MUJOCO_GL=egl for headless rendering.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from src.assets.robots.unitree_g1.g1_constants import get_spec
from src.tasks.velocity.config.g1.stairs_terrain_cfg import StraightUphillStairsTerrainCfg


OUTPUT_DIR = Path("docs/assets/g1_visual_stairs_v1a")
WIDTH, HEIGHT = 640, 360
FOVY_DEG = 58.0
CAMERA_POS = (0.0576235, 0.01753, 0.42987)
CAMERA_PITCH_RAD = 0.8307767239493009


def camera_quat() -> tuple[float, float, float, float]:
  # D435 link: +X forward, +Y left, +Z up. MuJoCo camera: -Z forward, +Y up.
  link_to_camera = Rotation.from_matrix(
    np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
  )
  rotation = Rotation.from_euler("y", CAMERA_PITCH_RAD) * link_to_camera
  return tuple(np.roll(rotation.as_quat(), 1))


def look_at_quat(position: np.ndarray, target: np.ndarray) -> tuple[float, ...]:
  forward = target - position
  forward /= np.linalg.norm(forward)
  right = np.cross(forward, (0, 0, 1))
  right /= np.linalg.norm(right)
  up = np.cross(right, forward)
  axes = np.column_stack((right, up, -forward))
  return tuple(np.roll(Rotation.from_matrix(axes).as_quat(), 1))


def build_scene() -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
  spec = get_spec()
  spec.worldbody.add_body(name="terrain")
  terrain_cfg = StraightUphillStairsTerrainCfg(
    size=(8.0, 4.3),
    step_height_bands=((0.16, 0.16),),
    initial_step_depth=0.30,
    final_step_depth=0.30,
    num_steps=8,
  )
  geometry = terrain_cfg.geometry_at(0.0)
  output = terrain_cfg.function(0.0, spec, np.random.default_rng(0))
  names = ["approach", *(f"stair_{i:02d}" for i in range(1, 9)), "top_platform", "wall_left", "wall_right"]
  for item, name in zip(output.geometries, names, strict=True):
    item.geom.name = name
    if name.startswith("stair_"):
      item.geom.rgba = (0.65, 0.70, 0.72, 1.0)
    elif name == "approach":
      item.geom.rgba = (0.48, 0.53, 0.55, 1.0)
    elif name.startswith("wall_"):
      item.geom.rgba = (0.88, 0.87, 0.83, 1.0)
  spec.body("torso_link").add_camera(
    name="d435_depth", pos=CAMERA_POS, quat=camera_quat(), fovy=FOVY_DEG
  )
  overview_pos = np.array((3.0, -4.0, 3.0))
  spec.worldbody.add_camera(
    name="overview",
    pos=overview_pos,
    quat=look_at_quat(overview_pos, np.array((2.5, 2.15, 0.45))),
    fovy=60.0,
  )
  spec.worldbody.add_light(pos=(2, 1, 5), dir=(0, 0, -1))
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[:3] = (geometry.spawn_x, 2.15, 0.8)
  data.qpos[3:7] = (1, 0, 0, 0)
  joint_positions = {
    r".*_hip_pitch_joint": -0.1,
    r".*_knee_joint": 0.3,
    r".*_ankle_pitch_joint": -0.2,
    r".*_shoulder_pitch_joint": 0.35,
    r".*_elbow_joint": 0.87,
    r"left_shoulder_roll_joint": 0.18,
    r"right_shoulder_roll_joint": -0.18,
  }
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for pattern, value in joint_positions.items():
      if re.fullmatch(pattern, name):
        data.qpos[model.jnt_qposadr[i]] = value
  mujoco.mj_forward(model, data)
  return model, data, vars(geometry)


def render_images(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
    renderer.update_scene(data, camera="d435_depth")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    optical_depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera="overview")
    overview = renderer.render().copy()
  return rgb, optical_depth, segmentation, overview


def audit(model: mujoco.MjModel, data: mujoco.MjData, optical_depth: np.ndarray, segmentation: np.ndarray, geometry: dict) -> tuple[np.ndarray, dict]:
  cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435_depth")
  cam_pos = data.cam_xpos[cam_id].copy()
  cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
  forward = cam_rot @ np.array((0, 0, -1))
  fovy = math.radians(FOVY_DEG)
  hfov = 2 * math.atan(math.tan(fovy / 2) * WIDTH / HEIGHT)
  rows, cols = np.indices((HEIGHT, WIDTH))
  x = ((cols + 0.5) / WIDTH * 2 - 1) * math.tan(hfov / 2)
  y = (1 - (rows + 0.5) / HEIGHT * 2) * math.tan(fovy / 2)
  ray_scale = np.sqrt(1 + x * x + y * y)
  # MuJoCo OpenGL returns optical-axis Z; the actor converts MJWarp ray range to Z.
  # The background lies on the far clipping plane and is invalid for the actor.
  far = model.vis.map.zfar * model.stat.extent
  valid = np.isfinite(optical_depth) & (optical_depth > 0) & (optical_depth < 0.99 * far)
  depth = np.where(valid, optical_depth, 0.0).astype(np.float32)
  ray_checks = []
  for row in (10, 70, 140, 180, 280, 350):
    col = WIDTH // 2
    direction = cam_rot @ np.array((x[row, col], y[row, col], -1.0))
    direction /= np.linalg.norm(direction)
    geom_id = np.zeros(1, dtype=np.int32)
    expected = mujoco.mj_ray(model, data, cam_pos, direction, None, 1, -1, geom_id)
    measured = float(depth[row, col] * ray_scale[row, col])
    ray_checks.append({
      "pixel_yx": [row, col],
      "rendered_m": measured,
      "raycast_m": float(expected),
      "absolute_error_m": abs(measured - expected),
      "hit_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])),
    })
  geom_pixels = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
  stair_pixels = {}
  for i in range(1, 9):
    name = f"stair_{i:02d}"
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    stair_pixels[name] = int(np.count_nonzero(geom_pixels & (segmentation[:, :, 0] == geom_id)))
  center_depth = float(depth[HEIGHT // 2, WIDTH // 2])
  positive = depth[depth > 0]
  checks = {
    "camera_forward_x_positive": bool(forward[0] > 0.5),
    "camera_downward_z_negative": bool(forward[2] < -0.5),
    "fovy_58_degrees": bool(abs(math.degrees(fovy) - 58) < 1e-6),
    "depth_finite_positive_pixels": bool(len(positive) > 0 and np.isfinite(positive).all()),
    "depth_matches_geometry_raycast": bool(all(
      item["raycast_m"] > 0 and item["absolute_error_m"] < 0.03
      for item in ray_checks
    )),
    "staircase_visible": bool(sum(stair_pixels.values()) > 100),
    "first_stair_visible": bool(stair_pixels["stair_01"] > 0),
  }
  report = {
    "status": "pass" if all(checks.values()) else "fail",
    "checks": checks,
    "source": "official G1 29DoF Rev 1.0 URDF d435_joint, attached to torso_link",
    "camera": {
      "parent": "torso_link", "local_position_m": CAMERA_POS,
      "local_quaternion_wxyz": camera_quat(), "world_position_m": cam_pos.tolist(),
      "world_forward_xyz": forward.tolist(),
      "fovy_deg": FOVY_DEG, "hfov_deg": math.degrees(hfov),
      "resolution": [WIDTH, HEIGHT],
    },
    "geometry": geometry,
    "depth": {
      "units": "metres along optical Z axis", "shape": list(depth.shape),
      "invalid_value": 0.0, "valid_fraction": float(np.mean(valid)),
      "minimum_m": float(positive.min()) if len(positive) else None,
      "median_m": float(np.median(positive)) if len(positive) else None,
      "maximum_m": float(positive.max()) if len(positive) else None,
      "center_pixel_m": center_depth,
      "raycast_checks": ray_checks,
      "conversion": "MuJoCo OpenGL optical-Z depth; raycast comparison multiplies by the per-pixel ray scale",
      "preview_range_m": [1.0, 3.0],
    },
    "stair_visible_pixels": stair_pixels,
    "artifacts": ["camera_rgb.png", "camera_depth_m.npy", "camera_depth_preview.png", "overview.png"],
  }
  return depth, report


def main() -> None:
  model, data, geometry = build_scene()
  rgb, optical_depth, segmentation, overview = render_images(model, data)
  depth, report = audit(model, data, optical_depth, segmentation, geometry)
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  Image.fromarray(rgb).save(OUTPUT_DIR / "camera_rgb.png")
  Image.fromarray(overview).save(OUTPUT_DIR / "overview.png")
  np.save(OUTPUT_DIR / "camera_depth_m.npy", depth)
  preview = np.uint8(np.clip((depth - 1.0) / 2.0, 0, 1) * 255)
  Image.fromarray(preview).save(OUTPUT_DIR / "camera_depth_preview.png")
  (OUTPUT_DIR / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  if report["status"] != "pass":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
