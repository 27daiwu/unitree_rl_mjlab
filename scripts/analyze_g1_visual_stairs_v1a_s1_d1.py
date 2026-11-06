"""Offline analysis only of S1-D1 read-only captures; never loads a simulator."""
from pathlib import Path
import csv
import json
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('logs/experiments/g1_visual_stairs_v1a_s1_d1_20261104')
SEEDS = (142, 314)
DRIVERS = ('flat', 's01', 's1', 's1a')
STUDENTS = DRIVERS[1:]
WINDOWS = ((0, .3), (.3, .5), (.5, 1), (1, 1.5), (1.5, 3))
COLORS = dict(flat='black', s01='#0072B2', s1='#D55E00', s1a='#009E73')

def rows(kind, seed, driver):
    with (OUT / f'{kind}_{seed}_{driver}.csv').open() as f:
        return list(csv.DictReader(f))

def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def avg(rs, key):
    return float(np.mean([float(r[key]) for r in rs])) if rs else None

def curve(rs, key, time='input_time_s'):
    groups = collections.defaultdict(list)
    for r in rs:
        groups[float(r[time])].append(float(r[key]))
    t = sorted(groups)
    return np.array(t), np.array([np.mean(groups[x]) for x in t])

def draw(ax, rs, key, label, color, time='input_time_s'):
    t, y = curve(rs, key, time)
    ax.plot(t, y, label=label, color=color)
    return t, y

states = {(s,d): rows('states',s,d) for s in SEEDS for d in DRIVERS}
cross = {(s,d): rows('cross_actions',s,d) for s in SEEDS for d in DRIVERS}
audit = {(s,d): json.loads((OUT/f'audit_{s}_{d}.json').read_text()) for s in SEEDS for d in DRIVERS}
names = audit[142,'flat']['joint_names']
summary = {'time_convention': 'action/input at t; integrated state at t+0.02s; terminal retained, no reset or padding',
           'windows': [], 'episodes': [], 'sensitivity': [], 'first_action': {}, 'divergence': {}, 'joint_windows': []}
for driver in DRIVERS:
    rs = sum((cross[s,driver] for s in SEEDS), [])
    for student in STUDENTS:
        for lo,hi in WINDOWS:
            selected=[r for r in rs if r['student']==student and lo<=float(r['input_time_s'])<hi]
            summary['windows'].append(dict(driver=driver,student=student,start=lo,end=hi,n=len(selected),
              target_rmse_rad=avg(selected,'target_rmse_rad'),raw_rmse=avg(selected,'raw_rmse')))
for (s,d),rs in states.items():
    for i in range(16):
        ep=[r for r in rs if int(r['env_id'])==i]
        last=ep[-1]
        summary['episodes'].append(dict(seed=s,driver=d,env_id=i,end_s=float(last['post_time_s']),
          fell_over=int(last['fell_over']),done=int(last['done']),final_dx_m=float(last['dx_m']),
          min_riser_site_distance_m=min(float(r['front_site_riser_distance_m']) for r in ep),
          pitch_rms_deg=float(np.sqrt(np.mean([float(r['pitch_deg'])**2 for r in ep]))),
          roll_rms_deg=float(np.sqrt(np.mean([float(r['roll_deg'])**2 for r in ep]))),
          vx_rmse_mps=float(np.sqrt(np.mean([(float(r['vx_mps'])-.4)**2 for r in ep])))))
for student in STUDENTS:
    rs=[r for s in SEEDS for r in cross[s,student] if r['student']==student]
    first=[r for r in rs if float(r['input_time_s'])==0]
    e=np.array([json.loads(r['error_rad']) for r in first])
    summary['first_action'][student]={'target_rmse_rad':avg(first,'target_rmse_rad'),
       'signed_joint_error_rad':dict(zip(names,np.mean(e,axis=0).tolist())),
       'mean_abs_joint_error_rad':dict(zip(names,np.mean(np.abs(e),axis=0).tolist()))}
    for lo,hi in WINDOWS:
        selected=[r for r in rs if lo<=float(r['input_time_s'])<hi]
        if not selected: continue
        e=np.array([json.loads(r['error_rad']) for r in selected])
        for j,name in enumerate(names):
            summary['joint_windows'].append(dict(student=student,start=lo,end=hi,joint=name,n=len(e),
              signed_error_rad=float(e[:,j].mean()),mae_rad=float(abs(e[:,j]).mean())))
    sen=[r for s in SEEDS for r in rows('sensitivity',s,student) if r['student']==student]
    base={(r['seed'],r['env_id'],r['input_time_s']):float(r['target_rmse_rad']) for r in rs}
    for variant in sorted({r['variant'] for r in sen}):
        ss=[r for r in sen if r['variant']==variant]
        summary['sensitivity'].append(dict(student=student,variant=variant,n=len(ss),
          delta_rmse_rad=avg(ss,'target_delta_rmse_rad'),delta_max_rad=max(float(r['target_delta_max_rad']) for r in ss),
          baseline_teacher_error_rad=float(np.mean([base[r['seed'],r['env_id'],r['input_time_s']] for r in ss])),
          alternative_teacher_error_rad=avg(ss,'teacher_target_error_rmse_rad')))
# Paired state divergences: diagnostic thresholds, not preregistered acceptance.
# Three consecutive controls (60ms); at least 8 matched environments must remain.
for s in SEEDS:
    teacher={(r['step'],r['env_id']):r for r in states[s,'flat']}
    rs=states[s,'s01']
    metrics=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rs:
        tr=teacher[r['step'],r['env_id']]; t=float(r['post_time_s'])
        for key in ('pitch_deg','roll_deg','tilt_deg','vx_mps','dx_m'):
            metrics[key][t].append(abs(float(r[key])-float(tr[key])))
        metrics['contact_mismatch'][t].append(float(any(r[k]!=tr[k] for k in ('left_contact','right_contact'))))
        metrics['phase_max_abs'][t].append(max(abs(float(r[k])-float(tr[k])) for k in ('phase_sin_input','phase_cos_input')))
    result={}
    for key,threshold in dict(pitch_deg=5,roll_deg=5,tilt_deg=5,vx_mps=.1,contact_mismatch=.5).items():
        t=sorted(metrics[key]); hit=None
        for j in range(len(t)-2):
            if all(len(metrics[key][tt])>=8 and np.mean(metrics[key][tt])>threshold for tt in t[j:j+3]):
                hit=t[j];break
        result[key]={'threshold':threshold,'first_sustained_post_s':hit}
    result['phase_max_abs']=max(max(x) for x in metrics['phase_max_abs'].values())
    summary['divergence'][str(s)]=result
    # Plot aligned paired means and action errors, no survivor padding.
    fig,axs=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    for student in STUDENTS:
        for source,ls in [('flat','--'),(student,'-')]:
            rr=[r for r in cross[s,source] if r['student']==student]
            t,y=curve(rr,'target_rmse_rad');axs[0,0].plot(t,y,ls,color=COLORS[student],label=f'{student} on {source}')
    axs[0,0].set(ylabel='29-joint target RMSE vs teacher (rad)',xlabel='Input time (s)')
    for d in DRIVERS:
        for ax,key,label in [(axs[0,1],'tilt_deg','Total tilt (deg)'),(axs[1,0],'pitch_deg','Pitch (deg)'),
             (axs[1,1],'roll_deg','Euler roll (deg)'),(axs[2,0],'vx_mps','World X velocity (m/s)')]:
            draw(ax,states[s,d],key,d,COLORS[d],'post_time_s');ax.set(ylabel=label,xlabel='Post-step time (s)')
        ep=[r for r in summary['episodes'] if r['seed']==s and r['driver']==d]
        t=np.arange(0,3.001,.02)
        axs[2,1].step(t,[sum(r['end_s']>tt+1e-6 or (r['end_s']>=tt-1e-6 and not r['done']) for r in ep) for tt in t],where='post',label=d,color=COLORS[d])
    axs[2,0].axhline(.4,color='gray',ls=':');axs[0,1].axhline(70,color='gray',ls=':')
    axs[2,1].set(xlabel='Time (s)',ylabel='Survivors / 16',ylim=(-.5,16.5))
    for ax in axs.flat: ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(f'S1-D1 paired read-only rollouts, seed {s}; terminal-only traces, no padding')
    fig.savefig(OUT/f'cross_state_seed{s}.png',dpi=160);plt.close(fig)
# Key targets of S0.1 on own states vs same-state teacher; actual positions alongside.
for s in SEEDS:
    rr=[r for r in cross[s,'s01'] if r['student']=='s01']
    fig,axs=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    selected=('waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_hip_pitch_joint','right_hip_pitch_joint','left_hip_roll_joint','right_hip_roll_joint','left_knee_joint','right_knee_joint')
    for ax,name in zip(axs.flat,selected):
        j=names.index(name)
        for key,label,color in [('teacher_target_rad','Teacher advice on student state','black'),('student_target_rad','Executed S0.1 target',COLORS['s01'])]:
            groups=collections.defaultdict(list)
            for r in rr:groups[float(r['input_time_s'])].append(json.loads(r[key])[j])
            ts=sorted(groups);ax.plot(ts,[np.mean(groups[t]) for t in ts],label=label,color=color)
        groups=collections.defaultdict(list)
        for r in states[s,'s01']: groups[float(r['post_time_s'])].append(json.loads(r['joint_pos_rad'])[j])
        ts=sorted(groups);ax.plot(ts,[np.mean(groups[t]) for t in ts],label='Measured joint after integration',color='gray',ls=':')
        ax.axvspan(.5,1.5,alpha=.06,color='orange');ax.set(title=name.replace('_joint',''),xlabel='Time (s)',ylabel='rad',xlim=(0,1.7));ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7);fig.suptitle(f'S0.1 waist/leg targets, seed {s}; action t vs measured q at t+20ms')
    fig.savefig(OUT/f'joint_targets_seed{s}.png',dpi=160);plt.close(fig)
# Representative env12 timeline with teacher paired driver and same-state action error.
s=314;i=12
fig,axs=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
for d in ('flat','s01'):
    rr=[r for r in states[s,d] if int(r['env_id'])==i]
    for ax,key in zip(axs.flat[:6],('pitch_deg','roll_deg','tilt_deg','vx_mps','left_force_n','right_force_n')):
        draw(ax,rr,key,d,COLORS[d],'post_time_s');ax.set(xlabel='Post-step time (s)',ylabel=key)
rr=[r for r in cross[s,'s01'] if r['student']=='s01' and int(r['env_id'])==i]
draw(axs[3,0],rr,'target_rmse_rad','same-state S0.1 vs Flat',COLORS['s01'])
axs[3,0].set(xlabel='Input time (s)',ylabel='Target RMSE (rad)')
for d in ('flat','s01'):
    rr=[r for r in states[s,d] if int(r['env_id'])==i]
    draw(axs[3,1],rr,'front_site_riser_distance_m',d,COLORS[d],'post_time_s')
axs[3,1].set(xlabel='Post-step time (s)',ylabel='Front-foot-site distance to riser (m)')
for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Seed 314 env 12: paired teacher / S0.1; contact force is sensor magnitude')
fig.savefig(OUT/'env12_seed314.png',dpi=160);plt.close(fig)
# Proprioceptive difference from paired teacher at identical input times.
import torch
pilot=Path('logs/rsl_rl/g1_visual_stairs_v1a/2026-11-04_11-29-53_v1a_pilot_20261104/model_99.pt')
ps=torch.load(pilot,map_location='cpu',weights_only=False)['actor_state_dict']
checkpoints={'s01':Path('logs/experiments/g1_visual_stairs_v1a_s01_20261104/v1a_s0_1_bc_model.pt'),
             's1':Path('logs/experiments/g1_visual_stairs_v1a_s1_20261104/v1a_s1_bc_model.pt'),
             's1a':Path('logs/experiments/g1_visual_stairs_v1a_s1_20261104/v1a_s1a_dagger_bc_model.pt')}
normkeys=[k for k in ps if k.startswith('obs_normalizer.')]
summary['normalizer_equal_to_pilot']={}
std=ps['obs_normalizer._std'].numpy().reshape(-1)
for d,path in checkpoints.items():
    ds=torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']
    summary['normalizer_equal_to_pilot'][d]=all(torch.equal(ps[k],ds[k]) for k in normkeys)
summary['normalizer_zero_std_indices']=np.flatnonzero(std==0).tolist()
summary['proprio_shift']=[]
for s in SEEDS:
    teacher=np.load(OUT/f'snapshots_{s}_flat.npz')
    lookup={(round(float(t),5),int(i)):a for t,i,a in zip(teacher['time'],teacher['env_id'],teacher['actor'])}
    for d in STUDENTS:
        data=np.load(OUT/f'snapshots_{s}_{d}.npz')
        delta=np.array([a-lookup[round(float(t),5),int(i)] for t,i,a in zip(data['time'],data['env_id'],data['actor'])])/(std+.01)
        for lo,hi in WINDOWS:
            keep=(data['time']>=lo)&(data['time']<hi)
            if not keep.any():continue
            for name,sl in dict(ang_vel=slice(0,3),gravity=slice(3,6),phase=slice(9,11),joint_pos=slice(11,40),joint_vel=slice(40,69),last_actions=slice(69,98)).items():
                summary['proprio_shift'].append(dict(seed=s,driver=d,start=lo,end=hi,term=name,n=int(keep.sum()),normalized_rms=float(np.sqrt(np.mean(delta[keep,sl]**2)))))
summary['runtime_audit']={'max_target_mapping_error_rad':max(a['max_executed_target_mapping_error_rad'] for a in audit.values()),
 'max_train_eval_output_difference':max(m['train_eval_output_max_abs'] for a in audit.values() for m in a['modes'].values()),
 'all_model_states_unchanged':all(m['state_before']==m['state_after'] for a in audit.values() for m in a['modes'].values()),
 'dropout_batchnorm':audit[142,'s01']['modes']['s01']['dropout_batchnorm'],
 'clip_actions':audit[142,'s01']['clip_actions']}
summary['feature_stats']=[]
for d in ('flat','s01'):
    rs=[r for s in SEEDS for r in rows('features',s,d) if r['student']=='s01']
    summary['feature_stats'].append(dict(driver=d,**{k:avg(rs,k) for k in ('normalizer_abs_max','normalizer_rms','cnn_feature_rms','cnn_feature_drift_rms','depth_first_linear_rms','prop_first_linear_rms')},maximum_normalized_abs=max(float(r['normalizer_abs_max']) for r in rs)))
# Joint first sustained MAE > 0.05rad for 3 controls, initial bias distinguished from later growth.
summary['joint_first_sustained_005rad']={}
for s in SEEDS:
    rr=[r for r in cross[s,'s01'] if r['student']=='s01']
    g=collections.defaultdict(list)
    for r in rr:g[float(r['input_time_s'])].append(json.loads(r['error_rad']))
    ts=sorted(g);mae=np.array([np.abs(g[t]).mean(axis=0) for t in ts])
    summary['joint_first_sustained_005rad'][str(s)]={name:next((ts[k] for k in range(len(ts)-2) if (mae[k:k+3,j]>.05).all()),None) for j,name in enumerate(names)}
write('analysis_summary.json',summary)
for key in ('windows','episodes','sensitivity','joint_windows','proprio_shift'):
    with (OUT/f'{key}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[key][0]));w.writeheader();w.writerows(summary[key])
print(json.dumps({'runtime':summary['runtime_audit'],'divergence':summary['divergence'],'normalizer':summary['normalizer_equal_to_pilot']},indent=2))
# Check saved supervision labels against the frozen Flat actor (CPU; no optimizer).
flat_sd=torch.load('logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt',map_location='cpu',weights_only=False)['actor_state_dict']
summary['saved_label_audit']=[]
with torch.inference_mode():
    for dataset in (Path('logs/experiments/g1_visual_stairs_v1a_s01_20261104/flat_teacher_1s_dataset.pt'),Path('logs/experiments/g1_visual_stairs_v1a_s1_20261104/flat_teacher_safe_3s_dataset.pt')):
        data=torch.load(dataset,map_location='cpu',weights_only=False)
        diffs=[]
        for x,y in zip(data['actor'].split(512),data['action'].split(512)):
            z=(x-flat_sd['obs_normalizer._mean'])/(flat_sd['obs_normalizer._std']+.01)
            for layer in (0,2,4,6):
                z=torch.nn.functional.linear(z,flat_sd[f'mlp.{layer}.weight'],flat_sd[f'mlp.{layer}.bias'])
                if layer!=6:z=torch.nn.functional.elu(z)
            diffs.append((z-y).numpy())
        err=np.concatenate(diffs)
        summary['saved_label_audit'].append(dict(dataset=str(dataset),n=len(err),all_finite=all(bool(torch.isfinite(v).all()) for v in data.values()),flat_label_rmse=float(np.sqrt(np.mean(err**2))),flat_label_max_abs=float(abs(err).max())))
        del data
# Numeric recording integrity, including JSON vector-valued columns.
finite=True
raw_max={}
for (s,d),rs in states.items():
    mx=0
    for r in rs:
        for key,value in r.items():
            if key=='driver':continue
            a=np.array(json.loads(value) if key in ('executed_raw','target_rad','joint_pos_rad') else float(value))
            finite=finite and bool(np.isfinite(a).all())
            if key=='executed_raw':mx=max(mx,float(abs(a).max()))
    raw_max[f'{s}_{d}']=mx
summary['recording_all_state_fields_finite']=finite
summary['executed_raw_max_abs']=raw_max
write('analysis_summary.json',summary)
# Time-resolved sensitivity, with real-depth error baseline on matched timestamps.
fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
for s in SEEDS:
    sen=[r for r in rows('sensitivity',s,'s01') if r['student']=='s01']
    ax=axs[list(SEEDS).index(s)]
    for v in ('initial_depth','batch_swap','spatial_flip','scale_plus_1pct','zero_depth','constant_2m'):
        draw(ax,[r for r in sen if r['variant']==v],'target_delta_rmse_rad',v,None)
    ax.set(title=f'S0.1 on own states, seed {s}',xlabel='Input time (s)',ylabel='Target change from real depth (rad)')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(OUT/'depth_sensitivity.png',dpi=160);plt.close(fig)
print('Saved label audit:',json.dumps(summary['saved_label_audit']))
print('All state fields finite:',finite)
fig,axs=plt.subplots(4,1,figsize=(12,9),sharex=True,constrained_layout=True)
for ax,d in zip(axs[:3],('flat','s01','s1a')):
    rr=sorted((r for r in states[314,d] if r['env_id']=='12'),key=lambda r:float(r['post_time_s']))
    t=[float(r['post_time_s']) for r in rr]
    ax.step(t,[int(r['left_contact']) for r in rr],where='post',label='Left contact')
    ax.step(t,[int(r['right_contact'])+1.5 for r in rr],where='post',label='Right contact + 1.5')
    ax.set(ylabel=d,yticks=(0,1,1.5,2.5),ylim=(-.2,2.7));ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.2)
rr=[r for r in states[314,'s01'] if r['env_id']=='12']
for key in ('phase_sin_input','phase_cos_input'):
    draw(axs[3],rr,key,key,None)
axs[3].set(xlabel='Time (s); contacts post-step, phase input time',ylabel='Clock phase',xlim=(0,1.7));axs[3].legend(fontsize=8);axs[3].grid(alpha=.2)
fig.suptitle('Seed 314 env 12 contact timing; phase input is identical across paired drivers')
fig.savefig(OUT/'contact_phase_seed314.png',dpi=160);plt.close(fig)
