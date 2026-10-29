"""Frozen A.1 checkpoint replay: command-conditioned rewards, NOT historical logs.

Logs contain episode aggregate rewards only, so historic conditional means and
sample occupancy cannot be reconstructed. This supplemental replay freezes all
normalizers/weights and supports deterministic and stochastic actions explicitly.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import src.tasks  # noqa: F401


def capture(env):
  # Runs immediately after reward compute, before command resampling/reset.
  c = env.command_manager.get_command('twist').clone()
  d = env.scene['robot'].data
  v = torch.cat((d.root_link_lin_vel_b[:, :2], d.root_link_ang_vel_b[:, 2:3]), dim=1)
  env.attribution_sample = (c, v.clone(), env.reward_manager._step_reward.clone())
  return torch.zeros(env.num_envs, device=env.device)


def run(checkpoint, output, stochastic, steps, num_envs):
  task='Unitree-G1-Navigation'
  cfg=load_env_cfg(task); cfg.scene.num_envs=num_envs; cfg.seed=42
  cfg.metrics['conditional_capture']=MetricsTermCfg(func=capture)
  env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
  try:
    rl=load_rl_cfg(task); vec=RslRlVecEnvWrapper(env,clip_actions=rl.clip_actions)
    runner=load_runner_cls(task)(vec,asdict(rl),device='cuda:0')
    runner.load(checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0')
    policy=runner.get_inference_policy(device='cuda:0')
    env.reset(seed=42)
    commands=[]; velocities=[]; rewards=[]; falls=0
    for _ in range(steps):
      with torch.no_grad():
        action=policy(env.observation_manager.compute(),stochastic_output=stochastic)
        _, _, terminated, _, _=env.step(action)
      c,v,r=env.attribution_sample
      commands.append(c.cpu().numpy());velocities.append(v.cpu().numpy());rewards.append(r.cpu().numpy())
      falls+=int(terminated.sum())
    c=np.stack(commands);v=np.stack(velocities);r=np.stack(rewards)
    assert np.isfinite(r).all()
    zero=(np.abs(c).max(-1)<1e-6)
    low=(np.abs(c)<[.2,.15,.4]).all(-1)&~zero
    groups={'all':np.ones_like(zero),'zero':zero,'nonzero':~zero,'low_nonzero':low,'higher_nonzero':~zero&~low}
    names=env.reward_manager.active_terms
    result={}
    # Velocity-only algebraic reference, not a dynamically feasible controller.
    stationary=2*np.exp(-c[...,0]**2/.1**2)+2*np.exp(-c[...,1]**2/.08**2)+1.5*np.exp(-c[...,2]**2/.2**2)-.15*np.sum((c/[.2,.15,.4])**2,axis=-1)
    for name,mask in groups.items():
      result[name]=dict(samples=int(mask.sum()),fraction=float(mask.mean()),
        weighted_reward_rate={k:float(r[mask,i].mean()) for i,k in enumerate(names)},
        total_reward_rate=float(r[mask].sum(-1).mean()),
        velocity_rmse=np.sqrt(np.mean((v[mask]-c[mask])**2,axis=0)).tolist(),
        hypothetical_stationary_velocity_reward_rate=float(stationary[mask].mean()),
        hypothetical_perfect_tracking_velocity_reward_rate=5.5)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'samples.npz',command=c,velocity=v,weighted_reward_rate=r,reward_names=names)
    summary=dict(source='supplemental frozen-checkpoint replay, NOT recorded historical training occupancy',checkpoint=checkpoint,seed=42,steps=steps,num_envs=num_envs,dt=env.step_dt,stochastic=stochastic,fall_events=falls,groups=result,
      units='weighted reward per second; multiply by dt for actual step reward. No optimizer or normalizer updates.')
    (output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
  finally:env.close()


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stochastic',action='store_true');p.add_argument('--steps',type=int,default=2000);p.add_argument('--num-envs',type=int,default=128)
  a=p.parse_args();run(a.checkpoint,a.output,a.stochastic,a.steps,a.num_envs)
