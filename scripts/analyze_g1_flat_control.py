"""A.4 artifact aggregation; no training or task mutation."""
from pathlib import Path
import json,csv
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path('logs/navigation_phase_a4')
def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name):return json.loads((ROOT/name).read_text())

def differences(a,b,path=''):
  if isinstance(a,dict) and isinstance(b,dict):
    return [r for k in sorted(a.keys()|b.keys()) for r in differences(a.get(k),b.get(k),path+'.'+k)]
  if a!=b:return [dict(path=path,flat=a,a3=b)]
  return []


def main():
  flat=load('flat_preflight.json');a3=load('a3_runtime.json')
  # Both runtime probes use 4 envs; avoid classifying replication count as terrain difference.
  save('flat_vs_a3_full_diff.json',differences(flat['config'],a3['config']))
  sections={
    'command sampling':('commands',True), 'curriculum':('curriculum',True),
    'push / DR / reset':('events',True),'reward terms and weights':('rewards',True),
    'termination':('terminations',False),'action scale / mapping':('actions',False),
    'observations / phase / noise':('observations',True),'physics':('sim',False)}
  summary=[]
  for name,(key,relevant) in sections.items():
    same=flat['config'][key]==a3['config'][key]
    summary.append(dict(component=name,status='IDENTICAL' if same else 'DIFFERENT',potentially_relevant_to_acquisition=not same and relevant,flat=flat['config'][key],a3=a3['config'][key]))
  for name,key in [('robot / actuator / initial state','entities'),('terrain','terrain')]:
    x=flat['config']['scene'][key];y=a3['config']['scene'][key]
    summary.append(dict(component=name,status='IDENTICAL' if x==y else 'DIFFERENT',potentially_relevant_to_acquisition=x!=y,flat=x,a3=y))
  fp=load('flat_preflight_ppo.json');ap=load('a3_runtime_ppo.json')
  fp.pop('experiment_name');ap.pop('experiment_name')
  summary.append(dict(component='PPO / architecture / initial std',status='IDENTICAL' if fp==ap else 'DIFFERENT',potentially_relevant_to_acquisition=fp!=ap,flat=fp,a3=ap))
  save('flat_vs_a3_summary.json',summary)
  reports=[];cases=[]
  for path in sorted(ROOT.glob('eval_*/report.json'),key=lambda x:int(x.parent.name[5:])):
    r=json.loads(path.read_text());idx=int(path.parent.name[5:])
    reports.append(dict(checkpoint=f'model_{idx}.pt',updates=idx+1,categories=r['categories'],cases_acquired=sum(c['acquired'] for c in r['cases'].values()),fall_rate=float(np.mean([t['fall_rate'] for t in r['trials']])),completed_trials=sum(t['completed'] for t in r['trials'])))
    for scenario,c in r['cases'].items():
      ts=[t for t in r['trials'] if t['scenario']==scenario];full=[t for t in ts if t['completed']];axis=int(np.flatnonzero(c['commanded_velocity'])[0])
      avg=lambda key:np.mean([t['motion_formation']['stable'][key] for t in full],axis=0).tolist() if full else None
      cases.append(dict(checkpoint=f'model_{idx}.pt',scenario=scenario,command=c['commanded_velocity'],
        stable_axis_velocity=c['stable_actual_velocity'][axis] if full else None,
        stable_axis_rmse=c['stable_velocity_rmse'][axis] if full else None,
        root_displacement=avg('root_displacement_xyz'),accumulated_yaw=avg('accumulated_yaw'),
        single_support_fraction=avg('single_support_fraction'),action_rms=avg('action_rms'),action_temporal_variance=avg('action_temporal_variance'),
        foot_contact_transitions=avg('foot_contact_transitions'),foot_horizontal_path=avg('foot_horizontal_path'),
        observed_roll_pitch_rms=np.mean([t['roll_pitch_rms'] for t in ts],axis=0).tolist(),
        fall_rate=c['fall_rate'],completed_trials=len(full),motion_only_trials=sum(t['motion_only_acquired'] for t in ts),
        alternating_support_trials=sum(t['support_switches']['alternating_support_acquired'] for t in ts),acquired_trials=c['acquired_trials'],acquired=c['acquired']))
  best=max(reports,key=lambda r:(r['cases_acquired'],-r['fall_rate'],r['updates'])) if reports else None
  save('checkpoint_comparison.json',dict(checkpoints=reports,best_checkpoint=best['checkpoint'] if best else None,cases=cases))
  if cases:
    with (ROOT/'checkpoint_comparison.csv').open('w') as f:
      w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
  run=sorted(Path('logs/rsl_rl/g1_velocity').glob('*phase_a4_flat_control'))[-1]
  e=EventAccumulator(str(run),size_guidance={'scalars':0}).Reload();tags=e.Tags()['scalars']
  # Resume events supersede orphaned post-checkpoint updates by step number.
  series={tag:list({x.step:x for x in e.Scalars(tag)}.values()) for tag in tags}
  series={tag:sorted(values,key=lambda x:x.step) for tag,values in series.items()}
  stats={tag:dict(count=len(series[tag]),first20_mean=float(np.mean([x.value for x in series[tag][:20]])),last50_mean=float(np.mean([x.value for x in series[tag][-50:]])),last_step=series[tag][-1].step) for tag in tags}
  stats['finite']=bool(all(np.isfinite(x.value) for tag in tags for x in e.Scalars(tag)));save('training_summary.json',stats)
  with (ROOT/'learning_scalars.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['tag','step','value'])
    for tag in tags:w.writerows((tag,x.step,x.value) for x in series[tag])
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,2,figsize=(13,8))
  for tag in ['track_linear_velocity','track_angular_velocity','foot_gait','pose','action_rate_l2','is_terminated']:
    s=series['Episode_Reward/'+tag];axes[0,0].plot([x.step+1 for x in s],[x.value for x in s],label=tag)
  axes[0,0].legend(fontsize=7);axes[0,0].set_title('Weighted episode contributions / max episode seconds')
  for tag in ['Train/mean_episode_length']:
    s=series[tag];axes[0,1].plot([x.step+1 for x in s],[x.value*.02 for x in s]);axes[0,1].set_title('Mean training episode duration (s)')
  axes[1,0].plot([r['updates'] for r in reports],[r['cases_acquired'] for r in reports],'o-');axes[1,0].set_title('Motion + alternating support cases / 7')
  axes[1,1].plot([r['updates'] for r in reports],[r['fall_rate'] for r in reports],'o-');axes[1,1].set_title('Clean benchmark fall rate')
  for ax in axes.flat:ax.grid(alpha=.3);ax.set_xlabel('PPO updates')
  fig.tight_layout();fig.savefig(ROOT/'flat_learning_curves.png',dpi=160);plt.close(fig)
  if reports:
    names=list(dict.fromkeys(c['scenario'] for c in cases));last=ROOT/('eval_'+reports[-1]['checkpoint'][6:-3])
    fig,axes=plt.subplots(7,2,figsize=(14,16))
    for row,name in zip(axes,names):
      z=np.load(last/(name+'_0.npz'));cmd=z['command'];axis=int(np.argmax(np.abs(cmd).max(0)));t=z['time']
      row[0].plot(t,z['velocity'][:,axis],label='actual');row[0].plot(t,cmd[:,axis],'--',label='command');row[0].set_title(name)
      c=z['contact_time']>0;row[1].step(t,c[:,0],where='post',label='left');row[1].step(t,c[:,1]+1.2,where='post',label='right +1.2');row[1].set_title('Foot support')
      for ax in row:ax.axvspan(7,12,alpha=.08,color='green');ax.grid(alpha=.2)
    for ax in axes[0]:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'flat_motion_and_contacts.png',dpi=160);plt.close(fig)
  print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
