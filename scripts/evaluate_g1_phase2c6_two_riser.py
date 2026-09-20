"""Phase 2C.6 two-riser frozen-policy evaluator."""
from __future__ import annotations
import dataclasses, json
from dataclasses import dataclass
from pathlib import Path
import torch, tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from src.tasks.velocity.mdp.stairs_metrics import geometry_tensors

@dataclass
class Config:
  checkpoint: str
  episodes: int = 20
  max_steps: int = 1000
  seed: int = 42
  device: str = "cuda:0"
  output_file: str = "doc/tmp/g1_phase2c6_two_riser.json"

def main(cfg: Config) -> None:
  configure_torch_backends(); import src.tasks, mjlab.tasks
  task = "Unitree-G1-Phase2C6-Two-Riser-Audit"
  ec=load_env_cfg(task); ac=load_rl_cfg(task); ec.seed=cfg.seed; ec.scene.num_envs=cfg.episodes
  env=ManagerBasedRlEnv(ec,device=cfg.device); vec=RslRlVecEnvWrapper(env,clip_actions=ac.clip_actions)
  runner=(load_runner_cls(task) or MjlabOnPolicyRunner)(vec,dataclasses.asdict(ac),device=cfg.device)
  runner.load(cfg.checkpoint,load_cfg={"actor":True},strict=True,map_location=cfg.device); policy=runner.get_inference_policy(device=cfg.device)
  obs=vec.get_observations(); robot=env.scene["robot"]; sensor=env.scene["feet_ground_contact"]; g=geometry_tensors(env)
  r1=g["staircase_start_x"]; r2=g["staircase_end_x"]; z1=g["step_height"]; z2=g["top_height"]
  ids=[robot.site_names.index("left_foot"),robot.site_names.index("right_foot")]; n=cfg.episodes; dev=cfg.device
  alive=torch.ones(n,dtype=torch.bool,device=dev); done=torch.zeros_like(alive); fell=torch.zeros_like(alive); timeout=torch.zeros_like(alive)
  r1l=torch.zeros_like(alive); r1r=torch.zeros_like(alive); r2l=torch.zeros_like(alive); r2r=torch.zeros_like(alive); r1cross=torch.zeros_like(alive); r2cross=torch.zeros_like(alive); r2reach=torch.zeros_like(alive); stable=torch.zeros(n,dtype=torch.int32,device=dev); success_event=torch.zeros_like(alive)
  maxx=torch.full((n,),-float("inf"),device=dev); vxsum=torch.zeros(n,device=dev); vxcount=torch.zeros(n,device=dev); ret=torch.zeros(n,device=dev); wallsteps=torch.zeros(n,device=dev)
  bonus_idx=env.reward_manager._term_names.index("one_time_bilateral_upper_platform_acquisition"); bonus_events=torch.zeros(n,dtype=torch.int32,device=dev); bonus_return=torch.zeros(n,device=dev)
  bonus_before_r2=torch.zeros(n,dtype=torch.int32,device=dev); bonus_after_r2=torch.zeros(n,dtype=torch.int32,device=dev)
  scan_r1=torch.zeros(n,dtype=torch.bool,device=dev); scan_r2=torch.zeros(n,dtype=torch.bool,device=dev)
  for k in range(cfg.max_steps):
    root=robot.data.root_link_pos_w-env.scene.env_origins; feet=robot.data.site_pos_w[:,ids]-env.scene.env_origins.unsqueeze(1); found=sensor.data.found[:,:2]>0
    fx=feet[:,:,0]+g["spawn_x"].unsqueeze(1); fz=feet[:,:,2]
    s1=(fx>=r1.unsqueeze(1)+.03)&(fx<=r2.unsqueeze(1)-.03)&((fz-z1.unsqueeze(1)).abs()<.05)&found
    s2=(fx>=r2.unsqueeze(1)+.03)&((fz-z2.unsqueeze(1)).abs()<.05)&found
    r1l|=s1[:,0]&alive; r1r|=s1[:,1]&alive; r2l|=s2[:,0]&alive; r2r|=s2[:,1]&alive
    r1cross|=(root[:,0]+g["spawn_x"]>=r1+.15)&alive; r2reach|=(root[:,0]+g["spawn_x"]>=r2-.12)&alive; r2cross|=(root[:,0]+g["spawn_x"]>=r2+.15)&alive
    hits=env.scene["terrain_scan"].data.hit_pos_w[:,:,2]-env.scene.env_origins[:,None,2]
    scan_r1 |= (((hits-z1[:,None]).abs()<0.01).any(1) & (root[:,0]+g["spawn_x"]<r1+.25) & alive)
    scan_r2 |= (((hits-z2[:,None]).abs()<0.01).any(1) & (root[:,0]+g["spawn_x"]>r2-.25) & alive)
    both2=s2.all(1); bilateral_ever=r2l&r2r; roll,pitch,_=euler_xyz_from_quat(robot.data.root_link_quat_w); expected_z=z2+float(robot.cfg.init_state.pos[2]); root_height_ok=(root[:,2]-expected_z).abs()<=.20; forward_ok=robot.data.root_link_lin_vel_b[:,0]>.05; wall_now=(root[:,1].abs()>1.85)|(feet[:,:,1].abs().amax(1)>1.95); stable_now=r2cross&bilateral_ever&root_height_ok&(roll.abs()<.6)&(pitch.abs()<.6)&forward_ok&~wall_now
    stable=torch.where(stable_now&alive,stable+1,torch.where(~alive,stable,torch.zeros_like(stable)))
    success_event|=(stable>=50)&alive
    with torch.no_grad(): obs,reward,dones,_=vec.step(policy(obs))
    stepbonus=env.reward_manager._step_reward[:,bonus_idx]*env.step_dt; fired=(stepbonus>0)&alive; bonus_events+=fired.int(); bonus_before_r2+=(fired&~both2).int(); bonus_after_r2+=(fired&both2).int(); bonus_return+=torch.where(alive,stepbonus,torch.zeros_like(stepbonus))
    root=robot.data.root_link_pos_w-env.scene.env_origins; wall=((root[:,1].abs()>1.85)|(feet[:,:,1].abs().amax(1)>1.95)); wallsteps+=wall&alive
    ret+=torch.where(alive,reward,torch.zeros_like(reward)); vxsum+=torch.where(alive,robot.data.root_link_lin_vel_b[:,0],torch.zeros_like(vxsum)); vxcount+=alive; maxx=torch.where(alive,torch.maximum(maxx,root[:,0]+g["spawn_x"]),maxx)
    fell_now=env.termination_manager.get_term("fell_over").bool(); timeout_now=env.termination_manager.get_term("time_out").bool(); new=dones.bool()&alive; fell|=new&fell_now; timeout|=new&timeout_now; done|=new; alive&=~new
    if not bool(alive.any()): break
  terminal=done; r1bil=r1l&r1r; r2bil=r2l&r2r; r1succ=r1cross&r1bil; success=r1succ&r2reach&r2cross&r2bil&success_event
  # Mutually exclusive terminal attribution, ordered by progression.
  buckets={"R0_PRE_RISER1_FAILURE":(~r1cross)&terminal,"R1_NO_FIRST_PLATFORM_ACQUISITION":r1cross&~r1bil&terminal,"R1_ACQUIRED_THEN_INTERMEDIATE_FAILURE":r1succ&~r2reach&terminal,"R2_REACHED_BUT_NO_FINAL_PLATFORM_SUPPORT":r2reach&~r2bil&terminal,"R2_ONE_FOOT_FINAL_SUPPORT_ONLY":r2reach&(r2l^r2r)&terminal,"R2_BILATERAL_THEN_STABILITY_FAILURE":r2cross&r2bil&~success&terminal,"R2_SUCCESS":success}
  assigned=torch.zeros_like(alive)
  counts={}
  for name,mask in buckets.items(): counts[name]=int((mask&~assigned).sum()); assigned|=mask
  out={"task":task,"checkpoint":str(Path(cfg.checkpoint).resolve()),"episodes":n,"seed":cfg.seed,"riser_1_height":float(z1[0]),"riser_2_increment":float(z2[0]-z1[0]),"intermediate_tread_depth":float(g["step_depth"][0]),"intermediate_platform_z":float(z1[0]),"final_platform_z":float(z2[0]),"height_scan_sees_riser_1":bool(scan_r1.all()),"height_scan_sees_riser_2":bool(scan_r2.all()),"two_riser_height_scan_signal_valid":bool(scan_r1.all() and scan_r2.all()),"r1_left_support_rate":r1l.float().mean().item(),"r1_right_support_rate":r1r.float().mean().item(),"r1_bilateral_rate":r1bil.float().mean().item(),"r1_success_rate":r1succ.float().mean().item(),"r2_reach_rate":r2reach.float().mean().item(),"r2_left_support_rate":r2l.float().mean().item(),"r2_right_support_rate":r2r.float().mean().item(),"r2_bilateral_rate":r2bil.float().mean().item(),"r2_success_rate":success.float().mean().item(),"two_riser_success_rate":success.float().mean().item(),"pre_riser1_fall_rate":((~r1cross)&fell).float().mean().item(),"intermediate_platform_fall_rate":(r1succ&~r2reach&fell).float().mean().item(),"riser2_post_contact_fall_rate":(r2reach&~r2bil&fell).float().mean().item(),"final_platform_late_fall_rate":(r2cross&r2bil&~success&fell).float().mean().item(),"mean_max_root_x":maxx.mean().item(),"mean_forward_velocity_alive":(vxsum/vxcount.clamp(min=1)).mean().item(),"mean_return":ret.mean().item(),"wall_dependent_locomotion":bool((wallsteps/vxcount.clamp(min=1)).mean()>0.05),"bonus_event_count":int(bonus_events.sum()),"bonus_event_max_per_episode":int(bonus_events.max()),"bonus_return_per_event":float(bonus_return.sum()/bonus_events.sum().clamp(min=1)),"bonus_events_before_riser2_crossing":int(bonus_before_r2.sum()),"bonus_events_after_riser2_crossing":int(bonus_after_r2.sum()),"bucket_counts":counts,"sum_repeated_riser_buckets":sum(counts.values()),"repeated_riser_bucket_partition_complete":bool(bool(torch.all(assigned==terminal)) and sum(counts.values())==n),"compiled_geometry":{"riser_1_x":float(r1[0]),"riser_2_x":float(r2[0]),"riser_1_height":float(z1[0]),"riser_2_increment":float((z2-z1)[0]),"intermediate_platform_z":float(z1[0]),"final_platform_z":float(z2[0]),"intermediate_tread_depth":float(g["step_depth"][0])}}
  Path(cfg.output_file).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output_file).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); env.close()
if __name__=="__main__": main(tyro.cli(Config))
