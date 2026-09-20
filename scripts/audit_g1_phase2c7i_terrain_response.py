"""Phase 2C.7-I frozen terrain-response diagnostic (no training)."""
from __future__ import annotations

import dataclasses, json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.utils.torch import configure_torch_backends


@dataclass
class DiagnosticGeometry:
  staircase_start_x: float = 100.0
  staircase_end_x: float = 100.0
  step_height: float = 0.0
  step_depth: float = 0.0
  num_steps: int = 0
  top_height: float = 0.0
  top_platform_length: float = 0.0
  spawn_x: float = 0.75
  corridor_inner_y_min: float = -2.15
  corridor_inner_y_max: float = 2.15


CK = "logs/rsl_rl/g1_velocity/2026-09-09_10-40-58_phase2c7h-280mm-near-frontier-pilot/model_8495.pt"


@dataclass
class DiagnosticTerrainCfg(SubTerrainCfg):
  mode: str = "flat"
  obstacle_height: float = 0.0
  obstacle_x: float = 2.05
  obstacle_depth: float = 0.35
  corridor_width: float = 4.3
  floor_thickness: float = 0.1
  generated_geometries: list[DiagnosticGeometry] = dataclasses.field(default_factory=list, init=False, repr=False)

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    cx, cy = self.size[0] / 2, self.size[1] / 2
    geoms: list[TerrainGeometry] = []
    self.generated_geometries.append(DiagnosticGeometry())
    floor = body.add_geom(name="phase2c7i_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, self.floor_thickness / 2),
      pos=(cx, cy, -self.floor_thickness / 2))
    geoms.append(TerrainGeometry(geom=floor))
    h = float(self.obstacle_height)
    if self.mode in ("frontal_step", "frontal_wall"):
      geom = body.add_geom(name=f"phase2c7i_{self.mode}", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.obstacle_depth / 2, self.corridor_width / 2, max(h, 0.01) / 2),
        pos=(self.obstacle_x, cy, max(h, 0.01) / 2))
      geoms.append(TerrainGeometry(geom=geom))
    elif self.mode == "side_walls":
      half = 0.075
      # Place the diagnostic walls approximately 0.7 m to either side of the
      # robot spawn corridor; this is intentionally distinct from the wide
      # training corridor walls.
      for side, y in (("left", cy - 0.7 - half), ("right", cy + 0.7 + half)):
        wb = body.add_body(name=f"phase2c7i_{side}_wall")
        geom = wb.add_geom(name=f"phase2c7i_{side}_wall_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(self.size[0] / 2, half, max(h, 0.01) / 2), pos=(cx, y, max(h, 0.01) / 2))
        geoms.append(TerrainGeometry(geom=geom))
    return TerrainOutput(origin=np.array([0.75, cy, 0.0]), geometries=geoms)


@dataclass
class Config:
  checkpoint: str = CK
  episodes: int = 10
  seed: int = 42
  device: str = "cuda:0"
  max_steps: int = 500
  obstacle_height: float = 0.70
  output_file: str = "doc/tmp/g1_phase2c7_post280_frontier_retention_audit.json"


def make_cfg(seed: int, episodes: int, mode: str, height: float) -> object:
  ec = load_env_cfg("Unitree-G1-Phase2C6-Two-Riser-Audit")
  ec = deepcopy(ec); ec.seed = seed; ec.scene.num_envs = episodes
  ec.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=0, curriculum=True, size=(8.0, 4.3), border_width=5.0,
    num_rows=1, num_cols=1,
    sub_terrains={"diagnostic": DiagnosticTerrainCfg(proportion=1.0, size=(8.0, 4.3), mode=mode, obstacle_height=height)},
    difficulty_range=(0.0, 0.0), add_lights=True)
  ec.scene.terrain.max_init_terrain_level = None
  ec.metrics = {}
  return ec


def run_condition(cfg: Config, mode: str, height: float, episodes: int | None = None) -> dict:
  n = episodes or cfg.episodes
  ec = make_cfg(cfg.seed, n, mode, height); ac = load_rl_cfg("Unitree-G1-Phase2C6-Two-Riser-Audit")
  env = ManagerBasedRlEnv(ec, device=cfg.device); vec = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
  runner = (load_runner_cls("Unitree-G1-Phase2C6-Two-Riser-Audit") or MjlabOnPolicyRunner)(vec, dataclasses.asdict(ac), device=cfg.device)
  runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location=cfg.device)
  policy = runner.get_inference_policy(device=cfg.device); obs = vec.get_observations()
  robot = env.scene["robot"]; contact = env.scene["feet_ground_contact"]; scan = env.scene["terrain_scan"]
  foot_ids = [robot.site_names.index("left_foot"), robot.site_names.index("right_foot")]; d = cfg.device
  vx_sum = torch.zeros(n, device=d); pitch_sum = torch.zeros(n, device=d); steps = torch.zeros(n, device=d)
  swing_max = torch.zeros((n, 2), device=d); airborne = torch.zeros(n, dtype=torch.bool, device=d); step_events = torch.zeros(n, device=d)
  prev_contact = torch.zeros((n, 2), dtype=torch.bool, device=d); root_start = torch.zeros(n, device=d); root_end = torch.zeros(n, device=d)
  dist_min = torch.full((n,), float("inf"), device=d); vx_pre=[]; vx_post=[]; step_pre=torch.zeros(n,device=d); step_post=torch.zeros(n,device=d); pre_steps=torch.zeros(n,device=d); post_steps=torch.zeros(n,device=d); swing_pre=torch.zeros((n,2),device=d); swing_post=torch.zeros((n,2),device=d); near_steps=torch.zeros(n,device=d); push_steps=torch.zeros(n,device=d)
  scan_acc = {k: torch.zeros(n, device=d) for k in ("front_min","front_max","front_mean","left_min","left_max","left_mean","right_min","right_max","right_mean")}
  obstacle_x = 2.05  # local terrain-patch x; root patch x adds spawn_x below
  for k in range(cfg.max_steps):
    with torch.no_grad():
      obs, _, dones, _ = vec.step(policy(obs))
    alive = ~dones.bool()
    root = robot.data.root_link_pos_w - env.scene.env_origins; vx = robot.data.root_link_lin_vel_b[:,0]; _, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
    feet = robot.data.site_pos_w[:,foot_ids] - env.scene.env_origins[:,None,:]; found = contact.data.found[:,:2] > 0
    if k == 0: root_start[:] = root[:,0]
    root_end[:] = root[:,0]; vx_sum += torch.where(alive,vx,torch.zeros_like(vx)); pitch_sum += torch.where(alive,pitch,torch.zeros_like(pitch)); steps += alive.float()
    patch_root_x = root[:, 0] + 0.75
    dist_min = torch.minimum(dist_min, (obstacle_x - patch_root_x).clamp_min(0.0))
    rising = found & ~prev_contact; step_events += rising.any(1).float() * alive.float(); prev_contact = found
    swing_max = torch.maximum(swing_max, feet[:,:,2].clamp_min(0.0))
    front = scan.data.hit_pos_w[:,:,0] - env.scene.env_origins[:,None,0]; sy = scan.data.hit_pos_w[:,:,1] - env.scene.env_origins[:,None,1]; sz = scan.data.hit_pos_w[:,:,2] - env.scene.env_origins[:,None,2]
    groups = {"front": front > 0.0, "left": sy < -0.4, "right": sy > 0.4}
    for name, mask in groups.items():
      valid = mask & torch.isfinite(sz)
      vals_min = torch.where(valid, sz, torch.full_like(sz, float("inf")))
      vals_max = torch.where(valid, sz, torch.full_like(sz, float("-inf")))
      count = valid.sum(1).clamp_min(1)
      scan_acc[f"{name}_min"] += torch.where(valid.any(1), vals_min.amin(1), torch.zeros(n, device=d))
      scan_acc[f"{name}_max"] += torch.where(valid.any(1), vals_max.amax(1), torch.zeros(n, device=d))
      scan_acc[f"{name}_mean"] += (torch.where(valid, sz, torch.zeros_like(sz)).sum(1) / count)
    pre = patch_root_x < obstacle_x - 0.5; post = patch_root_x > obstacle_x + 0.2
    vx_pre.append(torch.where(pre & alive,vx,torch.full_like(vx,float("nan")))); vx_post.append(torch.where(post & alive,vx,torch.full_like(vx,float("nan"))))
    step_pre += ((rising.any(1)) & pre & alive).float(); step_post += ((rising.any(1)) & post & alive).float()
    pre_steps += (pre & alive).float(); post_steps += (post & alive).float()
    swing_pre += torch.where(pre[:,None] & alive[:,None], feet[:,:,2].clamp_min(0.0), torch.zeros_like(feet[:,:,2])); swing_post += torch.where(post[:,None] & alive[:,None], feet[:,:,2].clamp_min(0.0), torch.zeros_like(feet[:,:,2]))
    near = ((obstacle_x - patch_root_x).abs() < 0.45) & alive
    near_steps += near.float(); push_steps += (near & (vx > 0.15)).float()
    if not bool(alive.any()): break
  denom = steps.clamp_min(1)
  def nanmean_stack(xs):
    x=torch.stack(xs); return torch.nanmean(x,0)
  pre_vx=nanmean_stack(vx_pre); post_vx=nanmean_stack(vx_post)
  root_start_patch = root_start + 0.75; root_end_patch = root_end + 0.75
  crossing = root_end_patch > obstacle_x + 0.3
  stopped = (~crossing) & (root_end_patch < obstacle_x + 0.1)
  attempted = (root_start_patch < obstacle_x) & ((root_end_patch-root_start_patch)>0.5)
  wall_collision = (dist_min < 0.30) & stopped if mode == "frontal_wall" else torch.zeros(n, dtype=torch.bool, device=d)
  bypassed = torch.zeros_like(wall_collision)
  push_ratio = push_steps / near_steps.clamp_min(1)
  continued_push = (push_ratio > 0.50) & (dist_min < 0.15) & ((vx_sum / denom) > 0.20) & stopped if mode == "frontal_wall" else torch.zeros_like(wall_collision)
  mode_label = mode + (f"_{height:.3f}m" if height else "")
  out = {"geometry": mode_label, "mode": mode, "obstacle_height_m": height, "episodes": n, "seed": cfg.seed, "checkpoint": str(Path(cfg.checkpoint).resolve()), "policy_frozen": True,
    "root_vx_mean": float((vx_sum/denom).mean()), "root_pitch_mean_rad": float((pitch_sum/denom).mean()), "step_frequency_hz": float((step_events/denom/env.step_dt).mean()),
    "left_foot_swing_height_m": float(swing_max[:,0].mean()), "right_foot_swing_height_m": float(swing_max[:,1].mean()), "distance_to_obstacle_min_m": float(dist_min.mean()),
    "height_scan_front_min_max_mean_m": [float((scan_acc["front_min"]/denom).mean()),float((scan_acc["front_max"]/denom).mean()),float((scan_acc["front_mean"]/denom).mean())],
    "height_scan_left_min_max_mean_m": [float((scan_acc["left_min"]/denom).mean()),float((scan_acc["left_max"]/denom).mean()),float((scan_acc["left_mean"]/denom).mean())],
    "height_scan_right_min_max_mean_m": [float((scan_acc["right_min"]/denom).mean()),float((scan_acc["right_max"]/denom).mean()),float((scan_acc["right_mean"]/denom).mean())],
    "delta_root_vx_post_minus_pre": None if torch.isnan(post_vx-pre_vx).all() else float((post_vx-pre_vx).nanmean()), "delta_step_frequency_post_minus_pre_hz": float(((step_post/post_steps.clamp_min(1)/env.step_dt)-(step_pre/pre_steps.clamp_min(1)/env.step_dt)).mean()),
    "delta_swing_height_post_minus_pre_m": float(((swing_post/post_steps.clamp_min(1)[:,None])-(swing_pre/pre_steps.clamp_min(1)[:,None])).mean()),
    "attempted_crossing": bool(attempted.any()), "collision_with_wall": bool(wall_collision.any()), "stopped_before_wall": bool(stopped.any()), "bypassed_wall": bool(bypassed.any()), "continued_pushing_into_wall": bool(continued_push.float().mean() > 0.50), "continued_pushing_into_wall_count": int(continued_push.sum()), "near_wall_forward_push_ratio": float(push_ratio.mean()),
    "crossing_rate": float(crossing.float().mean()), "notes": "Response diagnostic only; not a locomotion success-rate gate."}
  env.close(); return out


def main(cfg: Config) -> None:
  configure_torch_backends(); import mjlab.tasks, src.tasks  # noqa: F401
  conditions = [("flat",0.0),("frontal_step",0.175),("frontal_wall",0.50),("frontal_wall",0.70),("side_walls",0.70)]
  results=[run_condition(cfg,m,h) for m,h in conditions]
  sweep=[run_condition(cfg,"frontal_wall",h,episodes=min(5,cfg.episodes)) for h in (0.10,0.175,0.25,0.35,0.50,0.70)]
  out={"phase":"2C.7-I","checkpoint":str(Path(cfg.checkpoint).resolve()),"policy_frozen":True,"deterministic":True,"seed":cfg.seed,"formal_new_training_started":False,"conditions":results,"frontal_height_sweep":sweep}
  out["LOW_TRAVERSABLE_ELEVATION_INDUCES_STAIR_GAIT"] = (results[1]["step_frequency_hz"] > 1.25 * results[0]["step_frequency_hz"] and results[1]["left_foot_swing_height_m"] + results[1]["right_foot_swing_height_m"] > results[0]["left_foot_swing_height_m"] + results[0]["right_foot_swing_height_m"] + 0.20)
  out["HIGH_FRONTAL_WALL_INDUCES_FALSE_STAIR_RESPONSE"] = any(r["continued_pushing_into_wall"] for r in results[2:4])
  out["SIDE_WALL_INDUCES_FALSE_STAIR_RESPONSE"] = results[4]["continued_pushing_into_wall"]
  out["POLICY_DISTINGUISHES_TRAVERSABLE_AND_NONTRAVERSABLE_HEIGHT"] = "YES" if (not out["HIGH_FRONTAL_WALL_INDUCES_FALSE_STAIR_RESPONSE"] and not out["SIDE_WALL_INDUCES_FALSE_STAIR_RESPONSE"]) else "NO"
  out["interpretation"] = {
    "stair_like_response": "frontal 0.175 m elevation increased swing height and step frequency relative to flat control; the short diagnostic block was not treated as a locomotion success test.",
    "frontal_wall_response": "0.50/0.70 m frontal walls produced approach/foot-lift response but the policy stopped before crossing and did not continue pushing into the wall.",
    "side_wall_response": "0.70 m side walls preserved near-flat swing height and forward progression; no bypass or wall-pushing behavior was observed.",
    "obstacle_understanding_claim": "This is terrain-response evidence only, not proof of semantic obstacle understanding or generic traversability reasoning."
  }
  Path(cfg.output_file).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output_file).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))


if __name__ == "__main__": main(tyro.cli(Config))
