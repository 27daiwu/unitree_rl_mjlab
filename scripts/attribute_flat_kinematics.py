"""Read-only W1/W2 attribution from the saved Flat model_499 assay traces."""
from pathlib import Path
import json
import numpy as np

SOURCE = Path('logs/navigation_phase_a4/eval_499')
OUT = Path('logs/navigation_phase_a4/flat_kinematic_attribution')
COMMANDS = ('forward_020', 'forward_040', 'left_020', 'right_020')


def bouts(mask):
  starts = np.r_[0, np.flatnonzero(np.diff(mask.astype(int))) + 1]
  ends = np.r_[starts[1:], len(mask)]
  return [(int(a), int(b)) for a, b in zip(starts, ends) if mask[a]]


def analyze(path):
  with np.load(path) as z:
    raw = {key:z[key].copy() for key in ('time','velocity','position','rpy','foot_position','contact_time')}
  mask = raw['time'] > 7. + 1e-6
  t, body, root, rpy, feet, contact = (raw[k][mask] for k in raw)
  contact = contact > 0
  yaw = np.unwrap(rpy[:, 2]); co, si = np.cos(yaw), np.sin(yaw)
  world_from_body = np.column_stack((co*body[:,0]-si*body[:,1], si*body[:,0]+co*body[:,1]))
  world_finite_difference = np.gradient(root[:,:2], t, axis=0)
  rel = feet-root[:,None,:]
  feet_root_yaw = np.empty_like(rel)
  feet_root_yaw[:,:,0] = co[:,None]*rel[:,:,0]+si[:,None]*rel[:,:,1]
  feet_root_yaw[:,:,1] = -si[:,None]*rel[:,:,0]+co[:,None]*rel[:,:,1]
  feet_root_yaw[:,:,2] = rel[:,:,2]
  landings=[]; swings=[]
  for foot in (0,1):
    intervals=bouts(contact[:,foot])
    for a,b in intervals:
      if a==0 or b-a<3: continue
      other=1-foot
      landings.append(dict(foot='left' if foot==0 else 'right', time_s=float(t[a]),
        world_xyz=feet[a,foot].tolist(), root_yaw_frame_xyz=feet_root_yaw[a,foot].tolist(),
        step_width_signed_m=float(feet_root_yaw[a,0,1]-feet_root_yaw[a,1,1]),
        cross_over=bool(feet_root_yaw[a,0,1] < feet_root_yaw[a,1,1]),
        other_foot_contact=bool(contact[a,other])))
    for (a,b),(next_a,_) in zip(intervals,intervals[1:]):
      if next_a-b<3: continue
      ground_height=float(np.mean((feet[b-1,foot,2],feet[next_a,foot,2])))
      airborne=feet[b:next_a,foot,2]-ground_height
      local_delta=feet_root_yaw[next_a,foot]-feet_root_yaw[b-1,foot]
      swings.append(dict(foot='left' if foot==0 else 'right', takeoff_s=float(t[b]),landing_s=float(t[next_a]),
        duration_s=float(t[next_a]-t[b]), clearance_m=float(np.max(airborne)),
        lateral_step_root_yaw_m=float(local_delta[1]), forward_step_root_yaw_m=float(local_delta[0]),
        world_step_xy_m=(feet[next_a,foot,:2]-feet[b-1,foot,:2]).tolist()))
  states=contact[:,0].astype(int)+2*contact[:,1].astype(int)
  starts=np.r_[0,np.flatnonzero(np.diff(states))+1];ends=np.r_[starts[1:],len(states)]
  seq=[int(states[a]) for a,b in zip(starts,ends) if b-a>=3 and states[a] in (1,2)]
  result=dict(trace=str(path),stable_window_s=[float(t[0]),float(t[-1])],
    body_velocity_mean=body.mean(0).tolist(), world_velocity_rotated_mean=world_from_body.mean(0).tolist(),
    world_velocity_finite_difference_mean=world_finite_difference.mean(0).tolist(),
    velocity_frame_residual_rms=float(np.sqrt(np.mean((world_from_body-world_finite_difference)**2))),
    integrated_rotated_body_displacement_xy=np.trapezoid(world_from_body,t,axis=0).tolist(),
    world_y_from_forward_m=float(np.trapezoid(si*body[:,0],t)),
    world_y_from_lateral_m=float(np.trapezoid(co*body[:,1],t)),
    root_displacement_xy=(root[-1,:2]-root[0,:2]).tolist(),
    yaw_initial=float(yaw[0]),yaw_final=float(yaw[-1]),yaw_change=float(yaw[-1]-yaw[0]),
    yaw_rate_body_mean=float(body[:,2].mean()), roll_rms=float(np.sqrt(np.mean(rpy[:,0]**2))),
    pitch_rms=float(np.sqrt(np.mean(rpy[:,1]**2))),
    single_support_fraction=float(np.mean(contact.sum(1)==1)),
    double_support_fraction=float(np.mean(contact.sum(1)==2)),
    support_alternations=int(sum(a!=b for a,b in zip(seq,seq[1:]))),
    contact_bouts={name:len(bouts(contact[:,i])) for i,name in enumerate(('left','right'))},
    landing_events=landings,swing_events=swings)
  OUT.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(OUT/path.name, time=t,body_velocity=body,world_velocity_rotated=world_from_body,
    world_velocity_finite_difference=world_finite_difference,root_world=root,rpy=rpy,
    foot_world=feet,foot_root_yaw_frame=feet_root_yaw,contact=contact)
  return result


def average(rows,key):
  return np.mean([r[key] for r in rows],axis=0).tolist()


def main():
  trials=[]
  for command in COMMANDS:
    for replica in range(4):
      trials.append(dict(command=command,replica=replica,**analyze(SOURCE/f'{command}_{replica}.npz')))
  (OUT/'trials.json').write_text(json.dumps(trials,indent=2)+'\n')
  summary={}
  for command in COMMANDS:
    rows=[r for r in trials if r['command']==command]
    x={key:average(rows,key) for key in ('body_velocity_mean','world_velocity_rotated_mean',
      'world_velocity_finite_difference_mean','integrated_rotated_body_displacement_xy',
      'root_displacement_xy','world_y_from_forward_m','world_y_from_lateral_m',
      'yaw_change','yaw_rate_body_mean','support_alternations',
      'single_support_fraction','roll_rms','pitch_rms','velocity_frame_residual_rms')}
    for foot in ('left','right'):
      land=[event for r in rows for event in r['landing_events'] if event['foot']==foot]
      swings=[event for r in rows for event in r['swing_events'] if event['foot']==foot]
      x[foot]=dict(landings=len(land),swings=len(swings),
        landing_root_yaw_frame_xyz=np.mean([e['root_yaw_frame_xyz'] for e in land],axis=0).tolist() if land else None,
        lateral_step_root_yaw_m=float(np.mean([e['lateral_step_root_yaw_m'] for e in swings])) if swings else None,
        forward_step_root_yaw_m=float(np.mean([e['forward_step_root_yaw_m'] for e in swings])) if swings else None,
        swing_clearance_m=float(np.mean([e['clearance_m'] for e in swings])) if swings else None)
    x['mean_step_width_m']=float(np.mean([e['step_width_signed_m'] for r in rows for e in r['landing_events']]))
    x['cross_over_landings']=sum(e['cross_over'] for r in rows for e in r['landing_events'])
    summary[command]=x
  (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
