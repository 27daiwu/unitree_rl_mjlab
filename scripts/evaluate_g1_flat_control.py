"""A.4 Flat clean deterministic assay with explicit alternating-support detection.

12 s commanded motion from reset (optional initial stand). Judge final five 1 s windows,
not the transient. Operational motion detection is NOT precision acceptance.
"""
from dataclasses import asdict,dataclass
from pathlib import Path
import json
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from src.tasks.velocity.mdp.navigation import evaluation_observations
from src.tasks.velocity.mdp.motion_formation_metrics import motion_formation_diagnostics
from evaluate_g1_navigation import snapshot,decode

COMMANDS={'forward_020':(.2,0.,0.),'forward_040':(.4,0.,0.),'backward_015':(-.15,0.,0.),
          'left_020':(0.,.2,0.),'right_020':(0.,-.2,0.),'turn_left_050':(0.,0.,.5),'turn_right_050':(0.,0.,-.5)}


@dataclass
class EvaluateConfig:
  checkpoint:str
  output_dir:str
  task:str='Unitree-G1-Flat'
  num_envs:int=4
  seed:int=42
  initial_stand_s:float=0.0
  device:str='cuda:0'
  record_foot_positions:bool=True


def capture(env):
  result=snapshot(env)
  robot=env.scene['robot'];ids=[robot.site_names.index(n) for n in ('left_foot','right_foot')]
  env.motion_diagnostics=(env.action_manager.action.clone(),
    env.scene['feet_ground_contact'].data.current_contact_time.clone(),
    robot.data.site_lin_vel_w[:,ids,:2].clone())
  if getattr(env, 'record_motion_foot_positions', False):
    env.motion_foot_positions = robot.data.site_pos_w[:,ids].clone()
  return result


def summarize(trace, command, fallen, initial_stand_s=0.):
  t=np.array(trace['time']);v=np.array(trace['velocity']);p=np.array(trace['position']);rpy=np.array(trace['rpy'])
  cmd=np.array(command);axis=int(np.flatnonzero(cmd)[0]);direction=np.sign(cmd[axis])
  motion=t>initial_stand_s+1e-6;stable=t>7.+1e-6
  complete=len(t)==600 and not fallen
  rms=lambda x:np.sqrt(np.mean(x*x,axis=0)).tolist() if len(x) else None
  windows=[]
  for start in range(7,12):
    mask=(t>start+1e-6)&(t<=start+1+1e-6)
    windows.append(v[mask].mean(0).tolist() if mask.sum()==50 else None)
  threshold=max(.02 if axis<2 else .05,.2*abs(cmd[axis]))
  progress=None
  if complete:
    begin=int(np.searchsorted(t,7.));delta=p[-1,:2]-p[begin,:2];yaw=rpy[begin,2]
    if axis==0:progress=float(direction*(delta@np.array([np.cos(yaw),np.sin(yaw)])))
    elif axis==1:progress=float(direction*(delta@np.array([-np.sin(yaw),np.cos(yaw)])))
    else:progress=float(direction*(np.unwrap(rpy[:,2])[-1]-np.unwrap(rpy[:,2])[begin]))
  acquired=bool(complete and all(w is not None and direction*w[axis]>=threshold for w in windows) and progress>=threshold*4.)
  actions=np.array(trace['action']);contact=np.array(trace['contact_time'])>0;feet=np.array(trace['foot_velocity'])
  return dict(motion_formation=motion_formation_diagnostics(trace, complete), commanded_velocity=list(command),completed=complete,fall_rate=float(fallen),duration=float(t[-1]),
    stable_samples=int(stable.sum()),stable_actual_velocity=v[stable].mean(0).tolist() if stable.any() else None,
    stable_velocity_rmse=rms((v-cmd)[stable]),motion_velocity_rmse=rms((v-cmd)[motion]),
    stable_one_second_velocity_means=windows,roll_pitch_rms=rms(rpy[motion,:2]),
    stable_signed_progress=progress,motion_detection_velocity_threshold=threshold,acquired=acquired,
    normalized_action_delta_rms=rms(np.diff(actions,axis=0)),
    foot_contact_transitions=np.count_nonzero(np.diff(contact.astype(int),axis=0),axis=0).tolist(),
    contact_foot_speed_mean=float(np.linalg.norm(feet,axis=-1)[contact].mean()) if contact.any() else None)


def support_switches(time, contact, completed):
  if not completed:return dict(alternating_support_acquired=False,left_bouts=0,right_bouts=0,alternations=0)
  c=contact[time>7.+1e-6]
  states=c[:,0].astype(int)+2*c[:,1].astype(int)
  starts=np.r_[0,np.flatnonzero(np.diff(states))+1];ends=np.r_[starts[1:],len(states)]
  bouts=[int(states[a]) for a,b in zip(starts,ends) if b-a>=3 and states[a] in (1,2)]
  left=bouts.count(1);right=bouts.count(2);alternations=sum(a!=b for a,b in zip(bouts,bouts[1:]))
  return dict(left_bouts=left,right_bouts=right,alternations=alternations,
              alternating_support_acquired=left>=2 and right>=2 and alternations>=3)


def run(cfg):
  import src.tasks  # noqa: F401
  if not 0 <= cfg.initial_stand_s <= 2.:raise ValueError('initial_stand_s must be 0..2')
  env_cfg=load_env_cfg(cfg.task);env_cfg.seed=cfg.seed;env_cfg.scene.num_envs=cfg.num_envs
  ranges=env_cfg.commands['twist'].ranges
  for command in COMMANDS.values():
    for value, bounds in zip(command,(ranges.lin_vel_x,ranges.lin_vel_y,ranges.ang_vel_z)):
      if not bounds[0]<=value<=bounds[1]:raise ValueError('Benchmark command outside Flat configured range')
  env_cfg.episode_length_s=1000.;env_cfg.events['reset_base'].params['pose_range']={}
  # Clean fixed-command assay only; the training task is never modified.
  env_cfg.events={k:v for k,v in env_cfg.events.items() if k in ('reset_base','reset_robot_joints')}
  env_cfg.curriculum={}
  for group in env_cfg.observations.values():group.enable_corruption=False
  env_cfg.metrics['motion_snapshot']=MetricsTermCfg(func=capture)
  env=ManagerBasedRlEnv(cfg=env_cfg,device=cfg.device)
  env.record_motion_foot_positions=cfg.record_foot_positions
  out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
  try:
    rl=load_rl_cfg(cfg.task);vec=RslRlVecEnvWrapper(env,clip_actions=rl.clip_actions)
    runner=load_runner_cls(cfg.task)(vec,asdict(rl),device=cfg.device)
    runner.load(cfg.checkpoint,load_cfg={'actor':True},strict=True,map_location=cfg.device)
    policy=runner.get_inference_policy(device=cfg.device)
    trials=[]
    for name,command in COMMANDS.items():
      env.reset(seed=cfg.seed);term=env.command_manager.get_term('twist')
      # Prevent heading/standing overrides and resampling without writing root state.
      term._update_command=lambda: None
      term.time_left.fill_(1e9)
      def pin(command):
        term.vel_command_b[:]=term.vel_command_b.new_tensor(command)
        term.is_standing_env[:]=not any(command)
        term.is_heading_env[:]=False
        term.time_left.fill_(1e9)
      alive=np.ones(cfg.num_envs,dtype=bool);fallen=np.zeros_like(alive)
      traces=[{k:[] for k in ('time','command','velocity','position','rpy','action','contact_time','foot_velocity')} for _ in range(cfg.num_envs)]
      if cfg.record_foot_positions:
        for trace in traces: trace['foot_position']=[]
      for step in range(600):
        cmd=(0.,0.,0.) if step*env.step_dt<cfg.initial_stand_s else command
        pin(cmd);obs=evaluation_observations(env)
        with torch.no_grad():
          action=policy(obs);_,reward,terminated,truncated,_=env.step(action)
        if not torch.isfinite(action).all() or not torch.isfinite(env.navigation_snapshot).all():raise RuntimeError('Non-finite evaluation state/action')
        velocity,position,rpy=decode(env.navigation_snapshot)
        actions,contact,feet=[a.cpu().numpy() for a in env.motion_diagnostics]
        foot_positions=env.motion_foot_positions.cpu().numpy() if cfg.record_foot_positions else None
        for i in np.flatnonzero(alive):
          for k,value in zip(traces[i],((step+1)*env.step_dt,cmd,velocity[i],position[i],rpy[i],actions[i],contact[i],feet[i])):
            traces[i][k].append(np.array(value).copy())
          if foot_positions is not None:
            traces[i]['foot_position'].append(foot_positions[i].copy())
        falls=terminated.cpu().numpy().astype(bool);fallen|=alive&falls
        alive&=~(falls|truncated.cpu().numpy().astype(bool))
        if not alive.any():break
      for i,trace in enumerate(traces):
        np.savez_compressed(out/f'{name}_{i}.npz',**trace)
        summary=summarize(trace,command,bool(fallen[i]),cfg.initial_stand_s)
        support=support_switches(np.asarray(trace['time']),np.asarray(trace['contact_time'])>0,summary['completed'])
        summary['motion_only_acquired']=summary['acquired']
        summary['support_switches']=support
        summary['acquired']=summary['acquired'] and support['alternating_support_acquired']
        trials.append(dict(scenario=name,env_index=i,**summary))
    cases={}
    for name,cmd in COMMANDS.items():
      group=[r for r in trials if r['scenario']==name]
      # Stable means are only aggregated over completed trials; retain coverage.
      full=[r for r in group if r['completed']]
      def avg(key):return np.mean([r[key] for r in full],axis=0).tolist() if full else None
      cases[name]=dict(commanded_velocity=cmd,completed_trials=len(full),trials=len(group),
        stable_actual_velocity=avg('stable_actual_velocity'),stable_velocity_rmse=avg('stable_velocity_rmse'),
        roll_pitch_rms=avg('roll_pitch_rms'),fall_rate=float(np.mean([r['fall_rate'] for r in group])),
        acquired_trials=sum(r['acquired'] for r in group),acquired=bool(sum(r['acquired'] for r in group)>=.75*len(group)))
    categories={'FORWARD_LOCOMOTION_ACQUIRED':['forward_020','forward_040'],'BACKWARD_LOCOMOTION_ACQUIRED':['backward_015'],
                'LATERAL_LOCOMOTION_ACQUIRED':['left_020','right_020'],'TURN_IN_PLACE_ACQUIRED':['turn_left_050','turn_right_050']}
    report=dict(config=asdict(cfg),duration_s=12.,stable_window_s=[7.,12.],
      evaluation_overrides='Clean fixed-command assay: no push/startup DR/noise/curriculum, root pose fixed, command pinned; Flat phase/action/reward/termination retained. No root velocity injection.',
      criterion='Also require >=2 left-only and >=2 right-only support bouts of >=60 ms with >=3 alternations in final5s. Survive full trial; every final 1s window direction-correct mean speed >= max(20% target,.02 m/s or .05 rad/s); signed world progress >=4*threshold over final5s; >=75% replicas for each case. Motion detection only, not navigation precision acceptance.',
      categories={k:all(cases[n]['acquired'] for n in names) for k,names in categories.items()},cases=cases,trials=trials)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'categories':report['categories'],'cases':cases},indent=2))
    return report
  finally:env.close()


if __name__=='__main__':run(tyro.cli(EvaluateConfig))
