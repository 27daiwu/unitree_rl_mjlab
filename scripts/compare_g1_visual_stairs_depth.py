"""Compare MuJoCo optical-Z depth with the Visual-Stairs actor at three poses."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401


TASK = "Unitree-G1-Visual-Stairs-V1A"
OUTPUT = Path("docs/assets/g1_visual_stairs_v1a/pose_comparison")
POSES = (
  ("start", 0.75, 0.80),
  ("first_riser_approach", 1.65, 0.80),
  ("middle_stair", 3.05, 1.44),
)


def preview(depth: np.ndarray) -> Image.Image:
  pixels = np.uint8(np.clip((depth - 0.4) / 3.6, 0, 1) * 255)
  return Image.fromarray(pixels)


def main() -> None:
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 1
  cfg.seed = 42
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  reports = []
  try:
    model = env.sim.mj_model
    cpu_data = mujoco.MjData(model)
    robot = env.scene["robot"]
    origin = env.scene.env_origins[0].clone()
    camera_id = model.camera("d435_depth").id
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env.reset()
    with mujoco.Renderer(model, height=90, width=160) as renderer:
      for name, patch_x, root_z in POSES:
        root_state = torch.zeros((1, 13), device=env.device)
        root_state[0, :3] = origin + torch.tensor(
          [patch_x - 0.75, 0.0, root_z], device=env.device
        )
        root_state[0, 3] = 1.0
        robot.write_root_state_to_sim(root_state)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        env.sim.sense()
        actor = env.observation_manager.compute_group("depth")[0, 0].cpu().numpy().copy()
        cpu_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        mujoco.mj_forward(model, cpu_data)
        renderer.update_scene(cpu_data, camera="d435_depth")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        reference = renderer.render().copy()
        renderer.disable_depth_rendering()
        valid = (actor > 0) & np.isfinite(reference) & (reference > 0)
        error = np.abs(actor[valid] - reference[valid])
        # Report both all pixels and smooth regions; raster and ray tracing
        # legitimately choose different surfaces at silhouette boundaries.
        edge = np.zeros_like(valid)
        edge[1:, :] |= np.abs(np.diff(reference, axis=0)) > 0.05
        edge[:-1, :] |= np.abs(np.diff(reference, axis=0)) > 0.05
        edge[:, 1:] |= np.abs(np.diff(reference, axis=1)) > 0.05
        edge[:, :-1] |= np.abs(np.diff(reference, axis=1)) > 0.05
        interior = valid & ~edge
        interior_error = np.abs(actor[interior] - reference[interior])
        camera_position_error = np.linalg.norm(
          env.sim.data.cam_xpos[0, camera_id].cpu().numpy()
          - cpu_data.cam_xpos[camera_id]
        )
        Image.fromarray(rgb).save(OUTPUT / f"{name}_rgb.png")
        preview(reference).save(OUTPUT / f"{name}_mujoco_depth.png")
        preview(actor).save(OUTPUT / f"{name}_actor_depth.png")
        np.save(OUTPUT / f"{name}_mujoco_depth_m.npy", reference)
        np.save(OUTPUT / f"{name}_actor_depth_m.npy", actor)
        reports.append({
          "pose": name,
          "patch_root_x_m": patch_x,
          "root_z_m": root_z,
          "qpos_identical": bool(np.array_equal(cpu_data.qpos, env.sim.data.qpos[0].cpu().numpy())),
          "camera_position_error_m": float(camera_position_error),
          "valid_fraction": float(valid.mean()),
          "actor_center_m": float(actor[45, 80]),
          "mujoco_center_m": float(reference[45, 80]),
          "mae_all_m": float(error.mean()),
          "p95_all_m": float(np.quantile(error, 0.95)),
          "mae_interior_m": float(interior_error.mean()),
          "p95_interior_m": float(np.quantile(interior_error, 0.95)),
          "interior_fraction": float(interior.mean()),
        })
    checks = {
      "same_qpos": all(p["qpos_identical"] for p in reports),
      "same_camera_position": all(p["camera_position_error_m"] < 1e-4 for p in reports),
      "depth_valid": all(p["valid_fraction"] > 0.8 for p in reports),
      "optical_z_agreement": all(p["p95_interior_m"] < 0.05 for p in reports),
    }
    result = {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "poses": reports}
    (OUTPUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
      raise SystemExit(1)
  finally:
    env.close()


if __name__ == "__main__":
  main()
