"""Deterministic Phase 2C.0 locomotion metrics evaluator."""
from __future__ import annotations
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import torch, tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.utils.torch import configure_torch_backends
from src.tasks.velocity.config.g1.phase2c1_mild_terrain_cfg import MildWaveCorridorTerrainCfg

@dataclass(frozen=True)
class Config:
  checkpoint: str
  task: str = "Unitree-G1-Phase2C0-Flat-Sanity"
  terrain_branch: str | None = None
  episodes: int = 100
  seed: int = 42
  max_steps: int = 1000
  output_file: str | None = None
  trace_file: str | None = None
  device: str | None = None

def main(cfg: Config) -> None:
  import mjlab.tasks, src.tasks  # noqa: F401
  configure_torch_backends(); task=cfg.task
  device=cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  ec=load_env_cfg(task); ac=load_rl_cfg(task); ec.seed=cfg.seed; ec.scene.num_envs=cfg.episodes
  if cfg.terrain_branch is not None and task == "Unitree-G1-Phase2C1-Mild-Terrain":
    assert ec.scene.terrain is not None
    slope_range = (0.0, 0.0) if cfg.terrain_branch == "flat" else (0.0, 0.0025)
    branch_cfg = MildWaveCorridorTerrainCfg(proportion=1.0, size=(8.0, 4.3), slope_range=slope_range)
    gen = deepcopy(ec.scene.terrain.terrain_generator)
    gen.num_rows = 1; gen.num_cols = 1; gen.sub_terrains = {cfg.terrain_branch: branch_cfg}
    gen.difficulty_range = (1.0, 1.0) if cfg.terrain_branch == "mild" else (0.0, 0.0)
    ec.scene.terrain.terrain_generator = gen
  env=ManagerBasedRlEnv(cfg=ec, device=device); vec=RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
  if cfg.terrain_branch is not None and cfg.terrain_branch not in {"flat", "mild"}:
    raise ValueError("terrain_branch must be 'flat' or 'mild'")
  runner=(load_runner_cls(task) or MjlabOnPolicyRunner)(vec, asdict(ac), device=device); runner.load(cfg.checkpoint, load_cfg={"actor":True}, strict=True, map_location=device); policy=runner.get_inference_policy(device=device)
  obs=vec.get_observations(); robot=env.scene["robot"]; sensor=env.scene["feet_ground_contact"]; li,ri=robot.site_names.index("left_foot"),robot.site_names.index("right_foot")
  n=cfg.episodes; z=torch.zeros(n,device=device); inf=torch.full((n,),float("inf"),device=device); done=torch.zeros(n,dtype=torch.bool,device=device); ret=z.clone(); steps=z.clone(); fall_t=inf.clone(); max_x=z.clone(); prev=torch.zeros(n,dtype=torch.long,device=device); last_single=torch.zeros(n,dtype=torch.long,device=device)
  sums={k:z.clone() for k in ("startup_vx","startup_err","steady_vx","steady_err","steady_sq","steady_vx2","root_z","root_z2","ang","double","left","right","air","lr","rl","wall","max_lz","max_rz","min_lz","min_rz","max_roll","max_pitch","max_y","min_root_z","max_root_z")}; traces=[]
  for k in range(cfg.max_steps):
    with torch.no_grad(): obs,r,dones,_=vec.step(policy(obs))
    active=~done; t=steps*env.step_dt; cmd=env.command_manager.get_command("twist")[:,0]; vx=robot.data.root_link_lin_vel_b[:,0]; root=robot.data.root_link_pos_w-env.scene.env_origins; roll,pitch,yaw=euler_xyz_from_quat(robot.data.root_link_quat_w); ang=torch.linalg.vector_norm(robot.data.root_link_ang_vel_b,dim=1); feet=robot.data.site_pos_w[:,(li,ri)]-env.scene.env_origins.unsqueeze(1); found=sensor.data.found[:,:2]>0; state=found[:,0].long()+2*found[:,1].long(); wall=(root[:,1].abs()>1.85)|(feet[:,:,1].abs().amax(1)>1.95)
    startup=t<2; steady=(t>=2)&(t<10)
    def add(name,v,mask=active): sums[name]+=torch.where(mask,v,torch.zeros_like(v))
    add("startup_vx",vx,startup); add("startup_err",(vx-cmd).abs(),startup); add("steady_vx",vx,steady); add("steady_err",(vx-cmd).abs(),steady); add("steady_sq",(vx-cmd).square(),steady); add("steady_vx2",vx.square(),steady); add("root_z",root[:,2]); add("root_z2",root[:,2].square()); add("ang",ang); add("double",(state==3).float()); add("left",(state==1).float()); add("right",(state==2).float()); add("air",(state==0).float()); add("wall",wall.float());
    # Count gait alternation across double-support frames by remembering the last
    # non-zero single-support state (1=left, 2=right).
    single=(state==1)|(state==2); lr=(single&(last_single==1)&(state==2)); rl=(single&(last_single==2)&(state==1)); add("lr",lr.float()); add("rl",rl.float()); last_single=torch.where(single,state,last_single)
    sums["max_lz"]=torch.where(active,torch.maximum(sums["max_lz"],feet[:,0,2]),sums["max_lz"]); sums["max_rz"]=torch.where(active,torch.maximum(sums["max_rz"],feet[:,1,2]),sums["max_rz"]); sums["min_lz"]=torch.where(active,torch.minimum(sums["min_lz"],feet[:,0,2]),sums["min_lz"]); sums["min_rz"]=torch.where(active,torch.minimum(sums["min_rz"],feet[:,1,2]),sums["min_rz"]); sums["max_roll"]=torch.where(active,torch.maximum(sums["max_roll"],roll.abs()),sums["max_roll"]); sums["max_pitch"]=torch.where(active,torch.maximum(sums["max_pitch"],pitch.abs()),sums["max_pitch"]); sums["max_y"]=torch.where(active,torch.maximum(sums["max_y"],root[:,1].abs()),sums["max_y"]); sums["min_root_z"]=torch.where(active,torch.minimum(sums["min_root_z"],root[:,2]),sums["min_root_z"]); sums["max_root_z"]=torch.where(active,torch.maximum(sums["max_root_z"],root[:,2]),sums["max_root_z"])
    ret+=torch.where(active,r,torch.zeros_like(r)); max_x=torch.where(active,torch.maximum(max_x,root[:,0]),max_x); new=dones.bool()&~done; fell=env.termination_manager.get_term("fell_over"); fall_t=torch.where(new&fell,steps+1,fall_t); steps+=active.float(); done|=new; prev=state
    if cfg.trace_file and len(traces)<cfg.max_steps: traces.append({"time":(k+1)*env.step_dt,"command_vx":float(cmd[0]),"base_forward_vx":float(vx[0]),"root_xyz":root[0].detach().cpu().tolist(),"rpy":[float(roll[0]),float(pitch[0]),float(yaw[0])],"base_ang_vel":robot.data.root_link_ang_vel_b[0].detach().cpu().tolist(),"left_foot_xyz":feet[0,0].detach().cpu().tolist(),"right_foot_xyz":feet[0,1].detach().cpu().tolist(),"left_contact":bool(found[0,0]),"right_contact":bool(found[0,1]),"support_state":int(state[0]),"wall_contact":bool(wall[0]),"termination":bool(fell[0]),"timeout":bool(dones[0] and not fell[0])})
  dur=steps.clamp(min=1)*env.step_dt; steady_count=torch.clamp((torch.minimum(dur,torch.full_like(dur,10))-2)/env.step_dt,min=1); mean_v=sums["steady_vx"]/steady_count
  out={"task":task,"checkpoint":str(Path(cfg.checkpoint).resolve()),"episodes":n,"seed":cfg.seed,"mean_return":ret.mean().item(),"mean_episode_steps":steps.mean().item(),"early_fall_lt_0p5s_rate":(fall_t*env.step_dt<.5).float().mean().item(),"early_fall_lt_1s_rate":(fall_t*env.step_dt<1).float().mean().item(),"early_fall_lt_2s_rate":(fall_t*env.step_dt<2).float().mean().item(),"overall_fall_rate":torch.isfinite(fall_t).float().mean().item(),"timeout_rate":torch.isinf(fall_t).float().mean().item(),"fall_time_distribution_s":[float(x) for x in (fall_t[torch.isfinite(fall_t)]*env.step_dt).cpu()],"startup_mean_actual_vx":(sums["startup_vx"]/(2/env.step_dt)).mean().item(),"startup_mae_vx":(sums["startup_err"]/(2/env.step_dt)).mean().item(),"steady_mean_actual_vx":mean_v.mean().item(),"steady_mae_vx":(sums["steady_err"]/steady_count).mean().item(),"steady_rmse_vx":torch.sqrt((sums["steady_sq"]/steady_count).mean()).item(),"steady_std_vx":torch.sqrt(torch.clamp((sums["steady_vx2"]/steady_count)-mean_v.square(),min=0)).mean().item(),"mean_forward_speed":(sums["steady_vx"]/steady_count).mean().item(),"mean_max_root_x_progress":max_x.mean().item(),"root_height_mean":(sums["root_z"]/steps.clamp(min=1)).mean().item(),"root_height_std":torch.sqrt(torch.clamp(sums["root_z2"]/steps.clamp(min=1)-(sums["root_z"]/steps.clamp(min=1)).square(),min=0)).mean().item(),"root_height_min":sums["min_root_z"].mean().item(),"root_height_max":sums["max_root_z"].mean().item(),"roll_max_abs":sums["max_roll"].mean().item(),"pitch_max_abs":sums["max_pitch"].mean().item(),"base_ang_vel_mean":(sums["ang"]/steps.clamp(min=1)).mean().item(),"double_support_ratio":(sums["double"]/steps.clamp(min=1)).mean().item(),"left_single_support_ratio":(sums["left"]/steps.clamp(min=1)).mean().item(),"right_single_support_ratio":(sums["right"]/steps.clamp(min=1)).mean().item(),"airborne_ratio":(sums["air"]/steps.clamp(min=1)).mean().item(),"lr_transition_count_mean":sums["lr"].mean().item(),"rl_transition_count_mean":sums["rl"].mean().item(),"support_alternation_episode_rate":((sums["lr"]+sums["rl"])>=3).float().mean().item(),"wall_contact_episode_rate":(sums["wall"]>0).float().mean().item(),"wall_contact_time_ratio":(sums["wall"]/steps.clamp(min=1)).mean().item(),"wall_dependent_locomotion":bool((sums["wall"]/steps.clamp(min=1)).mean()>0.05),"max_left_foot_z":sums["max_lz"].mean().item(),"max_right_foot_z":sums["max_rz"].mean().item(),"left_foot_z_range_mean":(sums["max_lz"]-sums["min_lz"]).mean().item(),"right_foot_z_range_mean":(sums["max_rz"]-sums["min_rz"]).mean().item(),"max_abs_root_y":sums["max_y"].mean().item()}
  out["terrain_branch"] = cfg.terrain_branch
  print(json.dumps(out,indent=2));
  if cfg.output_file: Path(cfg.output_file).write_text(json.dumps(out,indent=2)+"\n")
  if cfg.trace_file: Path(cfg.trace_file).write_text(json.dumps({"summary":out,"trace":traces},indent=2)+"\n")
  env.close()

if __name__=="__main__": main(tyro.cli(Config))
