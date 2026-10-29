"""Read-only rollout estimate of W1.1 effective-command gate occupancy."""
from dataclasses import asdict
from pathlib import Path
import json
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
import src.tasks  # noqa: F401

TASK='Unitree-G1-Flat'
CHECKPOINT='logs/rsl_rl/g1_velocity/2026-09-29_20-42-55_flat_w1_1_yaw_decoupling/model_399.pt'
OUT=Path('logs/flat_w1_2')


def main():
  cfg=load_env_cfg(TASK);cfg.scene.num_envs=4096;cfg.seed=42
  env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
  try:
    rl=load_rl_cfg(TASK)
    runner=load_runner_cls(TASK)(RslRlVecEnvWrapper(env,clip_actions=rl.clip_actions),asdict(rl),device='cuda:0')
    runner.load(CHECKPOINT,load_cfg={'actor':True},strict=True,map_location='cuda:0')
    policy=runner.get_inference_policy(device='cuda:0')
    obs,_=env.reset(seed=42)
    term=env.command_manager.get_term('twist')
    totals={name:torch.zeros(5,device='cuda:0',dtype=torch.float64) for name in
      ('zero','forward','backward','lateral','turning','multi_axis')}
    raw_turn_gate=torch.zeros(2,device='cuda:0',dtype=torch.float64)
    standing_gate=torch.zeros(2,device='cuda:0',dtype=torch.float64)
    heading_gate_nonstanding=torch.zeros(2,device='cuda:0',dtype=torch.float64)
    for _ in range(500):
      c=term.vel_command_b;w=env.scene['robot'].data.root_link_ang_vel_b[:,2]
      gate=c[:,2].abs()<.1
      x,y,z=c[:,0],c[:,1],c[:,2]
      zero=(x.abs()<.1)&(y.abs()<.1)&(z.abs()<.1)
      forward=(x>=.1)&(y.abs()<.1)&(z.abs()<.1)
      backward=(x<=-.1)&(y.abs()<.1)&(z.abs()<.1)
      lateral=(x.abs()<.1)&(y.abs()>=.1)&(z.abs()<.1)
      turning=(x.abs()<.1)&(y.abs()<.1)&(z.abs()>=.1)
      assigned=zero|forward|backward|lateral|turning
      masks=dict(zero=zero,forward=forward,backward=backward,lateral=lateral,turning=turning,multi_axis=~assigned)
      for name,m in masks.items():
        totals[name]+=torch.stack((m.sum(),(c[:,2].abs()*m).sum(),(gate&m).sum(),
          (-2*w.square()*gate*m).sum(),(w.abs()*m).sum())).double()
      heading_turn=term.is_heading_env & (term.heading_error.abs()>=.2)
      raw_turn_gate+=torch.stack((heading_turn.sum(),(heading_turn&gate).sum())).double()
      standing_gate+=torch.stack((term.is_standing_env.sum(),(term.is_standing_env&gate).sum())).double()
      heading_gate_nonstanding+=torch.stack(((heading_turn&~term.is_standing_env).sum(),
        (heading_turn&~term.is_standing_env&gate).sum())).double()
      with torch.no_grad():
        action=policy(obs)
        obs,_,_,_,_=env.step(action)
    total=sum(x[0].item() for x in totals.values())
    rows={}
    for name,s in totals.items():
      n=s[0].item()
      rows[name]=dict(samples=int(n),sample_fraction=n/total,
        mean_abs_wz_command=s[1].item()/n if n else None,
        yaw_decoupling_active_fraction=s[2].item()/n if n else None,
        mean_yaw_decoupling_contribution=s[3].item()/n if n else None,
        mean_actual_abs_wz=s[4].item()/n if n else None)
    result=dict(source='read-only checkpoint policy rollout, not historical training samples',
      checkpoint=CHECKPOINT,envs=4096,steps=500,rows=rows,
      heading_error_ge_0p2_samples=int(raw_turn_gate[0].item()),
      heading_error_ge_0p2_gate_samples=int(raw_turn_gate[1].item()),
      standing_samples=int(standing_gate[0].item()),
      standing_gate_samples=int(standing_gate[1].item()),
      heading_turn_nonstanding_samples=int(heading_gate_nonstanding[0].item()),
      heading_turn_nonstanding_gate_samples=int(heading_gate_nonstanding[1].item()))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'gating_rollout.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
  finally:env.close()

if __name__=='__main__':main()
