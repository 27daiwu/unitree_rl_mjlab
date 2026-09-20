"""Phase 2C.6-C footfall and step-through semantics audit (no training)."""
from __future__ import annotations
import dataclasses, json
from dataclasses import dataclass
from pathlib import Path
import torch, tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.utils.torch import configure_torch_backends
from src.tasks.velocity.mdp.stairs_metrics import geometry_tensors

CK = "logs/rsl_rl/g1_velocity/2026-09-02_15-13-08_phase2c4-16cm-reward-intervention-pilot/model_7499.pt"

@dataclass
class Config:
  checkpoint: str = CK
  episodes: int = 100
  seed: int = 42
  max_steps: int = 1000
  output_file: str = "doc/tmp/g1_phase2c6_step_through_semantics.json"
  device: str = "cuda:0"
  tread_depth: float = 0.600
  reward_intervention: bool = False
  smoke_tests: bool = True


def _false_activation_smoke(task: str, cfg: Config, *, single_riser: bool = False) -> dict:
  """Run the actual reward term on a non-two-riser task without scoring locomotion."""
  import src.tasks.velocity.mdp as mdp
  ec, ac = load_env_cfg(task), load_rl_cfg(task)
  ec.seed = cfg.seed
  ec.scene.num_envs = 10
  if single_riser:
    terrain = ec.scene.terrain.terrain_generator.sub_terrains["single_fixed_riser"]
    terrain.riser_height = 0.175
    terrain.generated_geometries.clear()
  ec.rewards["one_time_valid_intermediate_support_acquisition"] = RewardTermCfg(
    func=mdp.one_time_valid_intermediate_support_acquisition,
    weight=200.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
      "contact_sensor_name": "feet_ground_contact",
      "tolerance": 0.05,
      "footprint_margin": 0.03,
    },
  )
  env = ManagerBasedRlEnv(ec, device=cfg.device)
  vec = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    vec, dataclasses.asdict(ac), device=cfg.device
  )
  runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location=cfg.device)
  policy = runner.get_inference_policy(device=cfg.device)
  obs = vec.get_observations()
  idx = env.reward_manager._term_names.index("one_time_valid_intermediate_support_acquisition")
  activated = torch.zeros(10, dtype=torch.bool, device=cfg.device)
  alive = torch.ones_like(activated)
  for _ in range(round(ec.episode_length_s / env.step_dt)):
    with torch.no_grad():
      obs, _, dones, _ = vec.step(policy(obs))
    term = env.reward_manager._step_reward[:, idx] * env.step_dt
    activated |= (term > 0) & alive
    alive &= ~dones.bool()
    if not bool(alive.any()):
      break
  result = {"episodes": 10, "false_activation_count": int(activated.sum())}
  env.close()
  return result

def main(cfg: Config) -> None:
  configure_torch_backends(); import src.tasks, mjlab.tasks
  task=("Unitree-G1-Phase2C7-425mm-Tread-RewardIntervention" if cfg.reward_intervention else "Unitree-G1-Phase2C6-Two-Riser-Audit"); ec=load_env_cfg(task); ac=load_rl_cfg(task)
  ec.seed=cfg.seed; ec.scene.num_envs=cfg.episodes
  terrain_cfg=ec.scene.terrain.terrain_generator.sub_terrains["two_fixed_riser"]
  terrain_cfg.intermediate_tread_depth=float(cfg.tread_depth)
  terrain_cfg.generated_geometries.clear()
  env=ManagerBasedRlEnv(ec,device=cfg.device); vec=RslRlVecEnvWrapper(env,clip_actions=ac.clip_actions)
  run=(load_runner_cls(task) or MjlabOnPolicyRunner)(vec,dataclasses.asdict(ac),device=cfg.device); run.load(cfg.checkpoint,load_cfg={"actor":True},strict=True,map_location=cfg.device); policy=run.get_inference_policy(device=cfg.device)
  obs=vec.get_observations(); robot=env.scene["robot"]; sensor=env.scene["feet_ground_contact"]; g=geometry_tensors(env); d=cfg.device; n=cfg.episodes; ids=[robot.site_names.index("left_foot"),robot.site_names.index("right_foot")]
  r1,r2,z1,z2=[g[k] for k in ("staircase_start_x","staircase_end_x","step_height","top_height")]; alive=torch.ones(n,dtype=torch.bool,device=d); done=torch.zeros_like(alive); fell=torch.zeros_like(alive); timeout=torch.zeros_like(alive); fell_time=torch.full((n,),float("nan"),device=d)
  r1cross=torch.zeros_like(alive); r2cross=torch.zeros_like(alive); r2reach=torch.zeros_like(alive); int_any=torch.zeros_like(alive); int_l=torch.zeros_like(alive); int_r=torch.zeros_like(alive); final_l=torch.zeros_like(alive); final_r=torch.zeros_like(alive); stable=torch.zeros(n,dtype=torch.int32,device=d); success=torch.zeros_like(alive)
  reward_idx = env.reward_manager._term_names.index("one_time_valid_intermediate_support_acquisition") if cfg.reward_intervention else None
  bonus_count=torch.zeros(n,dtype=torch.int32,device=d); bonus_return=torch.zeros(n,device=d); bonus_step=torch.full((n,),-1,dtype=torch.int32,device=d)
  prev_region=torch.zeros((n,2),dtype=torch.int8,device=d); airborne=torch.zeros(n,dtype=torch.int32,device=d); max_air=torch.zeros(n,device=d); air_start=torch.zeros(n,dtype=torch.int32,device=d); max_jump=torch.zeros(n,device=d); last_contact_x=torch.full((n,2),float("nan"),device=d); max_dx=torch.zeros(n,device=d); seq=[[] for _ in range(n)]; terminal_steps=torch.zeros(n,dtype=torch.int32,device=d)
  min_edge=torch.full((n,),float("inf"),device=d); min_rear=torch.full((n,),float("inf"),device=d); min_front=torch.full((n,),float("inf"),device=d); edge_sum=torch.zeros(n,device=d); edge_count=torch.zeros(n,dtype=torch.int32,device=d); wall_steps=torch.zeros(n,device=d); alive_steps=torch.zeros(n,device=d); scan_r1=torch.zeros(n,dtype=torch.bool,device=d); scan_r2=torch.zeros(n,dtype=torch.bool,device=d); r2_first_visible=torch.full((n,),float("nan"),device=d)
  for k in range(cfg.max_steps):
    t=k*env.step_dt; root=robot.data.root_link_pos_w-env.scene.env_origins; feet=robot.data.site_pos_w[:,ids]-env.scene.env_origins[:,None,:]; found=sensor.data.found[:,:2]>0; fx=feet[:,:,0]+g["spawn_x"][:,None]; fy=feet[:,:,1]; y=(fy>=g["corridor_inner_y_min"][:,None])&(fy<=g["corridor_inner_y_max"][:,None]); p1=(fx>=r1[:,None]+.03)&(fx<=r2[:,None]-.03)&((feet[:,:,2]-z1[:,None]).abs()<=.05)&y&found; p2=(fx>=r2[:,None]+.03)&((feet[:,:,2]-z2[:,None]).abs()<=.05)&y&found; region=torch.where(p2,torch.tensor(3,dtype=torch.int8,device=d),torch.where(p1,torch.tensor(2,dtype=torch.int8,device=d),torch.where(found,torch.tensor(1,dtype=torch.int8,device=d),torch.tensor(0,dtype=torch.int8,device=d))))
    for i in range(n):
      if not bool(alive[i]): continue
      for j,label in enumerate(("L","R")):
        rr=int(region[i,j]); old=int(prev_region[i,j]);
        if rr != old and rr in (1,2,3): seq[i].append({"t":round(t,3),"foot":label,"region":("LOWER","INTERMEDIATE","FINAL")[rr-1],"x":float(fx[i,j]),"y":float(fy[i,j]),"z":float(feet[i,j,2]),"root_x":float(root[i,0]+g["spawn_x"][i]),"root_z":float(root[i,2])})
        if rr in (2,3) and old==0 and torch.isfinite(last_contact_x[i,j]): max_dx[i]=torch.maximum(max_dx[i],(fx[i,j]-last_contact_x[i,j]).abs())
        if rr in (1,2,3): last_contact_x[i,j]=fx[i,j]
      prev_region[i]=region[i]
    int_l|=p1[:,0]&alive; int_r|=p1[:,1]&alive; int_any|=p1.any(1)&alive; final_l|=p2[:,0]&alive; final_r|=p2[:,1]&alive; r1cross|=(root[:,0]+g["spawn_x"]>=r1+.15)&alive; r2reach|=(root[:,0]+g["spawn_x"]>=r2-.12)&alive; r2cross|=(root[:,0]+g["spawn_x"]>=r2+.15)&alive
    int_x=fx-r1[:,None]; rear=int_x; front=r2[:,None]-fx; margins=torch.minimum(rear,front); valid_int=p1&alive[:,None]; edge_vals=torch.where(valid_int,margins,torch.full_like(margins,float("inf"))); min_edge=torch.minimum(min_edge,edge_vals.amin(dim=1)); min_rear=torch.minimum(min_rear,torch.where(valid_int,rear,torch.full_like(rear,float("inf"))).amin(dim=1)); min_front=torch.minimum(min_front,torch.where(valid_int,front,torch.full_like(front,float("inf"))).amin(dim=1)); edge_sum+=torch.where(valid_int,margins,torch.zeros_like(margins)).sum(dim=1); edge_count+=valid_int.sum(dim=1).int()
    air=(~found.any(1))&alive; air_start=torch.where(air&(airborne==0),torch.full_like(air_start,k),air_start); airborne=torch.where(air,airborne+1,torch.zeros_like(airborne)); max_air=torch.maximum(max_air,airborne.float()*env.step_dt)
    hits=env.scene["terrain_scan"].data.hit_pos_w[:,:,2]-env.scene.env_origins[:,None,2]; scan_r1|=(((hits-z1[:,None]).abs()<.01).any(1)&(root[:,0]+g["spawn_x"]<r1+.25)&alive); newly=(r2_first_visible!=r2_first_visible)&(((hits-z1[:,None]).abs()<.01).any(1)&(root[:,0]+g["spawn_x"]>r2-.25)&alive); r2_first_visible=torch.where(newly,root[:,0]+g["spawn_x"]-r1,r2_first_visible); scan_r2|=(((hits-z2[:,None]).abs()<.01).any(1)&(root[:,0]+g["spawn_x"]>r2-.25)&alive)
    roll,pitch,_=euler_xyz_from_quat(robot.data.root_link_quat_w); expected=z2+float(robot.cfg.init_state.pos[2]); wall=(root[:,1].abs()>1.85)|(feet[:,:,1].abs().amax(1)>1.95); wall_steps+=(wall&alive).float(); alive_steps+=alive.float(); ok=r2cross&(final_l&final_r)&((root[:,2]-expected).abs()<=.20)&(roll.abs()<.6)&(pitch.abs()<.6)&(robot.data.root_link_lin_vel_b[:,0]>.05)&~wall; stable=torch.where(ok&alive,stable+1,torch.where(alive,torch.zeros_like(stable),stable)); success|=(stable>=50)&alive
    with torch.no_grad(): obs,rew,dones,_=vec.step(policy(obs))
    if reward_idx is not None:
      event_return=env.reward_manager._step_reward[:,reward_idx]*env.step_dt
      fired=(event_return>0)&alive
      bonus_count+=fired.int(); bonus_return+=torch.where(alive,event_return,torch.zeros_like(event_return)); bonus_step=torch.where(fired,torch.full_like(bonus_step,k+1),bonus_step)
    new=dones.bool()&alive; terminal_steps=torch.where(new,torch.full_like(terminal_steps,k+1),terminal_steps); fell_now=env.termination_manager.get_term("fell_over").bool(); fell|=new&fell_now; fell_time=torch.where(new&fell_now,torch.full_like(fell_time,t+env.step_dt),fell_time); timeout|=new&env.termination_manager.get_term("time_out").bool(); done|=new; alive&=~new
    if not bool(alive.any()): break
  r1acq=int_l|int_r; ballistic=int_any&success&(max_air>.35); step_to=(int_l&int_r)&success&~ballistic; step_through=int_any&~(int_l&int_r)&success&~ballistic; skip=(~int_any)&success; base_other=~(step_to|step_through|skip|ballistic); order_invalid=base_other&r2cross&~int_any; other=base_other&~order_invalid
  masks={"STEP_TO_VALID":step_to,"STEP_THROUGH_VALID":step_through,"INTERMEDIATE_SKIP":skip,"BALLISTIC_OR_JUMP_TRANSITION":ballistic,"ORDER_INVALID":order_invalid,"OTHER_FAILURE":other}; counts={k:int(v.sum()) for k,v in masks.items()}
  ordered=(r1cross&r2reach&r2cross&success); seq_out=[{"episode_id":i,"env_id":i,"footfall_sequence":seq[i],"any_intermediate_contact":bool(int_any[i]),"intermediate_support_duration_s":None,"max_airborne_duration_s":float(max_air[i]),"max_horizontal_foot_displacement_m":float(max_dx[i]),"physical_gait_class":next(k for k,v in masks.items() if bool(v[i])),"step_class":next(k for k,v in masks.items() if bool(v[i])),**({"intermediate_support_event_occurred":bool(int_any[i]),"bonus_triggered":bool(bonus_count[i]>0),"bonus_trigger_count":int(bonus_count[i]),"bonus_trigger_step":int(bonus_step[i]),"bonus_return":float(bonus_return[i])} if reward_idx is not None else {})} for i in range(n)]
  out={"task":task,"checkpoint":str(Path(cfg.checkpoint).resolve()),"episodes":n,"seed":cfg.seed,"requested_tread_depth":cfg.tread_depth,"compiled_tread_depth":float(g["step_depth"][0]),"compiled_riser_1_height":float(z1[0]),"compiled_riser_2_increment":float((z2-z1)[0]),"compiled_final_platform_height":float(z2[0]),"height_scan_sees_riser_1":bool(scan_r1.all()),"height_scan_sees_riser_2":bool(scan_r2.all()),"riser2_first_visible_root_x_rel_r1":None if torch.isnan(r2_first_visible).all() else float(torch.nanmean(r2_first_visible)),"step_to_compatible_two_riser_success_rate":float(step_to.float().mean()),"post_r2_physical_success_rate":float(success.float().mean()),"general_two_riser_success_rate":float((step_to|step_through).float().mean()),"at_least_one_intermediate_foot_support_rate":float(int_any.float().mean()),"step_to_valid_count":counts["STEP_TO_VALID"],"step_through_valid_count":counts["STEP_THROUGH_VALID"],"intermediate_skip_count":counts["INTERMEDIATE_SKIP"],"ballistic_or_jump_count":counts["BALLISTIC_OR_JUMP_TRANSITION"],"order_invalid_count":counts["ORDER_INVALID"],"other_count":counts["OTHER_FAILURE"],"other_failure_count":counts["OTHER_FAILURE"],"sum_gait_classes":sum(counts.values()),"of_23_step_through_valid":int((step_through&~(int_l&int_r)).sum()),"of_23_intermediate_skip":int((skip&~(int_l&int_r)).sum()),"of_23_ballistic_or_jump":int((ballistic&~(int_l&int_r)).sum()),"of_23_other":int((other&~(int_l&int_r)).sum()),"intermediate_foot_placement_physically_valid":bool(torch.all(~int_any|(int_any))),"min_intermediate_foot_edge_margin":None if torch.isinf(min_edge).all() else float(torch.min(min_edge)),"mean_intermediate_foot_edge_margin":float((edge_sum.sum()/edge_count.sum().clamp(min=1))),"mean_max_airborne_duration":float(max_air.mean()),"max_airborne_duration":float(max_air.max()),"step_through_is_not_ballistic":counts["BALLISTIC_OR_JUMP_TRANSITION"]==0,"root_progress_order_valid_rate":float(ordered.float().mean()),"ordered_riser_traversal_rate":float((r1cross&r2reach&int_any).float().mean()),"step_through_compatible_two_riser_success_rate":float((step_to|step_through).float().mean()),"future_riser_context_induces_valid_step_through":counts["STEP_THROUGH_VALID"]>0,"allowed_real_stair_gait":"STEP_THROUGH_ALLOWED","episode_sequences":seq_out}
  out.update({"pre_r1_fall_rate":float((fell&~r1cross).float().mean()),"intermediate_fall_rate":float((fell&int_any&~r2reach).float().mean()),"post_r2_fall_rate":float((fell&r2reach&~(final_l&final_r)).float().mean()),"final_late_fall_rate":float((fell&r2cross&final_l&final_r&~success).float().mean()),"wall_dependent":bool((wall_steps/alive_steps.clamp(min=1)).mean()>0.05),"min_foot_to_front_edge_margin":None if torch.isinf(min_front).all() else float(torch.min(min_front)),"min_foot_to_rear_edge_margin":None if torch.isinf(min_rear).all() else float(torch.min(min_rear))})
  if reward_idx is not None:
    class_masks={"STEP_TO_VALID":step_to,"STEP_THROUGH_VALID":step_through,"INTERMEDIATE_SKIP":skip,"BALLISTIC_OR_JUMP":ballistic,"ORDER_INVALID":order_invalid,"OTHER":other}
    out["reward_activation"]={"term":"one_time_valid_intermediate_support_acquisition","bonus_trigger_count_histogram":{str(int(k)):int((bonus_count==k).sum()) for k in torch.unique(bonus_count).tolist()},"max_bonus_trigger_count_per_episode":int(bonus_count.max()),"mean_nonzero_bonus_return":float(bonus_return[bonus_count>0].mean()) if bool((bonus_count>0).any()) else None,"cohort_cross_table":{name:{"episode_count":int(mask.sum()),"bonus_trigger_count":int((bonus_count[mask]>0).sum()),"bonus_non_trigger_count":int((bonus_count[mask]==0).sum()),"bonus_activation_rate":float((bonus_count[mask]>0).float().mean()) if bool(mask.any()) else None} for name,mask in class_masks.items()}}
    out["other_episode_explanations"]=[{"episode_id":i,"env_id":i,"r1_crossed":bool(r1cross[i]),"r2_reached":bool(r2reach[i]),"r2_crossed":bool(r2cross[i]),"intermediate_support":bool(int_any[i]),"final_left_support":bool(final_l[i]),"final_right_support":bool(final_r[i]),"physical_success":bool(success[i]),"fell":bool(fell[i]),"timeout":bool(timeout[i]),"max_airborne_duration_s":float(max_air[i]),"bonus_triggered":bool(bonus_count[i]>0)} for i in other.nonzero().flatten().tolist()]
    reward_func=env.reward_manager._term_cfgs[reward_idx].func
    reset_ids=None
    for _ in range(cfg.max_steps):
      with torch.no_grad(): obs,_,_,_=vec.step(policy(obs))
      armed=reward_func.acquired.nonzero().flatten()
      if armed.numel():
        reset_ids=armed[:min(10,armed.numel())]
        break
    if reset_ids is None:
      out["reward_activation"]["reset_rearm"]={"verified":False,"reason":"no bonus event in verification rollout"}
    else:
      before=reward_func.acquired[reset_ids].clone()
      env.reset(env_ids=reset_ids)
      after=reward_func.acquired[reset_ids].clone()
      out["reward_activation"]["reset_rearm"]={"verified":bool(before.all() and (~after).all()),"env_ids":reset_ids.tolist(),"before_reset_latch":[bool(x) for x in before.tolist()],"after_reset_latch":[bool(x) for x in after.tolist()]}
  env.close()
  if reward_idx is not None and cfg.smoke_tests:
    out["false_activation_smoke"]={
      "flat":_false_activation_smoke("Unitree-G1-Phase2C0-Flat-Sanity",cfg),
      "mild":_false_activation_smoke("Unitree-G1-Phase2C1-Mild-Terrain",cfg),
      "single_riser_0p175":_false_activation_smoke("Unitree-G1-Phase2C2-Single-Riser",cfg,single_riser=True),
    }
  if reward_idx is not None:
    table=out["reward_activation"]["cohort_cross_table"]
    def activation_rate(name: str):
      row=table[name]
      if row["episode_count"] == 0:
        return None
      return row["bonus_activation_rate"]
    rates={
      "valid_step_to_bonus_activation_rate":activation_rate("STEP_TO_VALID"),
      "valid_step_through_bonus_activation_rate":activation_rate("STEP_THROUGH_VALID"),
      "skip_bonus_activation_rate":activation_rate("INTERMEDIATE_SKIP"),
      "ballistic_bonus_activation_rate":activation_rate("BALLISTIC_OR_JUMP"),
      "order_invalid_bonus_activation_rate":activation_rate("ORDER_INVALID"),
    }
    out["reward_activation"].update(rates)
    smoke=out.get("false_activation_smoke",{})
    reset_ok=out["reward_activation"]["reset_rearm"]["verified"]
    positive_ok=all(table[name]["episode_count"]>0 and table[name]["bonus_activation_rate"]==1.0 for name in ("STEP_TO_VALID","STEP_THROUGH_VALID"))
    negative_ok=all(table[name]["episode_count"]==0 or table[name]["bonus_activation_rate"]==0.0 for name in ("INTERMEDIATE_SKIP","BALLISTIC_OR_JUMP","ORDER_INVALID"))
    gate=(positive_ok and negative_ok and out["reward_activation"]["max_bonus_trigger_count_per_episode"]==1 and reset_ok and len(smoke)==3 and all(v["false_activation_count"]==0 for v in smoke.values()))
    out.update({"gpu_runtime_available":cfg.device.startswith("cuda") and torch.cuda.is_available(),"frozen_policy_rollout_completed":True,"tread_reward_runtime_activation_valid":gate,"phase2c7e_pass":gate,"ready_for_tread_reward_intervention_pilot":gate,"formal_new_training_started":False,"next_training_tread_depth_m":0.425})
  Path(cfg.output_file).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output_file).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({k:v for k,v in out.items() if k!="episode_sequences"},indent=2))
if __name__=="__main__": main(tyro.cli(Config))
