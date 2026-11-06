"""Offline 2x2 actor-input attribution from lossless S1-D2 paired captures.

No environment creation, optimizer, model updates or checkpoint serialization.
Teacher reference follows proprioception, never follows the depth source.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import csv
import json
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import CNNModel, MLPModel
from mjlab.tasks.registry import load_rl_cfg
import train_g1_visual_stairs_v1a_s0 as s0
from diagnose_g1_visual_stairs_v1a_s1_d1 import digest_state, sha
from diagnose_g1_visual_stairs_v1a_s1_d2 import OUT, CHECKPOINTS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS=('TT','TS','ST','SS') # first letter proprio, second depth
COLORS=dict(TT='black',TS='#CC79A7',ST='#009E73',SS='#0072B2')
LABELS=dict(TT='Teacher proprio + teacher depth',TS='Teacher proprio + student depth',
            ST='Student proprio + teacher depth',SS='Student proprio + student depth')
WINDOWS={'0-.2':(0,10),'.2-.4':(10,20),'.4-.6':(20,30),'.6-.8':(30,40),'.8-1':(40,50),
         'focus_.4-1':(20,50),'1-1.5':(50,75),'all_paired':(0,150)}

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csv_write(path,rs):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def rms(x):return np.sqrt(np.mean(x*x,axis=-1))
def metrics(x):return dict(mean_sample_rmse_rad=float(rms(x).mean()),pooled_rms_rad=float(np.sqrt(np.mean(x*x))),max_abs_rad=float(abs(x).max()))

def model_from_checkpoint(task,path,device):
    cfg=asdict(load_rl_cfg(task));kwargs=cfg['actor'];kind=kwargs.pop('class_name')
    if kind=='MLPModel':kwargs.pop('cnn_cfg')
    fake=TensorDict({'actor':torch.zeros(1,98,device=device),'depth':torch.zeros(1,1,90,160,device=device)},batch_size=[1])
    cls=CNNModel if kind=='CNNModel' else MLPModel
    model=cls(fake,cfg['obs_groups'],'actor',29,**kwargs).to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['actor_state_dict'],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

def forward(model,p,d,device,batch=16):
    outputs=[]
    for start in range(0,len(p),batch):
        obs={'actor':torch.from_numpy(p[start:start+batch]).to(device)}
        if d is not None:
            depth=torch.from_numpy(d[start:start+batch]).to(device)
            # The camera uses NHWC -> NCHW with a singleton channel, yielding
            # (14400,1,160,1), while NPZ reload yields (14400,14400,160,1).
            # Both index identical pixels, but cuDNN can choose a different
            # numerical kernel. Restore the camera's stride without changing
            # a single value or manufacturing an observation.
            assert depth.shape[1:]==(1,90,160)
            obs['depth']=depth.as_strided(depth.shape,(14400,1,160,1))
        outputs.append(model(obs).cpu().numpy())
    return np.concatenate(outputs)

def line(ax,step,x,label,color=None,ls='-',spread=False):
    ts=np.unique(step)
    values=np.array([x[step==s].mean() for s in ts])
    ax.plot(ts*.02,values,label=label,color=color,ls=ls)
    if spread:
        lo=np.array([np.quantile(x[step==s],.25) for s in ts]);hi=np.array([np.quantile(x[step==s],.75) for s in ts])
        ax.fill_between(ts*.02,lo,hi,color=color,alpha=.08)

def decorate(ax,ylabel):
    ax.set(xlabel='Aligned input time (s)',ylabel=ylabel)
    ax.axvspan(.4,1,alpha=.06,color='orange');ax.grid(alpha=.2)

def plot_all(out,seed,data,names):
    step=data['step']
    fig,axs=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    for g in GROUPS:line(axs[0,0],step,rms(data[f'e_{g}']),g,COLORS[g],spread=True)
    decorate(axs[0,0],'29-joint target error vs corresponding Flat (rad)');axs[0,0].legend()
    for key,label,color in [('P_error','Proprio residual effect','#D55E00'),('V','Depth effect at teacher proprio','#009E73'),('I','Interaction','#CC79A7')]:
        line(axs[0,1],step,rms(data[key]),label,color)
    decorate(axs[0,1],'Factor residual effect RMS (rad)');axs[0,1].legend(fontsize=8)
    for key,label,color in [('P_raw','Student output proprio response','#D55E00'),('teacher_response','Flat proprio response','black'),('V','Student depth response','#009E73')]:
        line(axs[1,0],step,rms(data[key]),label,color)
    decorate(axs[1,0],'Raw target response RMS (rad)');axs[1,0].legend(fontsize=8)
    ts=np.unique(step);axs[1,1].step(ts*.02,[sum(step==s) for s in ts],where='post',color='black')
    decorate(axs[1,1],'Matched valid environment pairs / 16');axs[1,1].set_ylim(0,17)
    fig.suptitle(f'S1-D2 seed {seed}: offline 2x2 queries; TT/TS reference Flat(pT), ST/SS reference Flat(pS)')
    fig.savefig(out/f'four_errors_seed{seed}.png',dpi=160);plt.close(fig)
    for kind in ('all29_errors','key_targets','key_effects'):
        keynames=names if kind=='all29_errors' else [n for n in names if any(v in n for v in ('waist','hip','knee'))]
        nrows,ncols=(6,5) if kind=='all29_errors' else (4,3)
        fig,axs=plt.subplots(nrows,ncols,figsize=(17,18) if kind=='all29_errors' else (14,12),constrained_layout=True)
        for ax,name in zip(axs.flat,keynames):
            j=names.index(name)
            if kind=='all29_errors':
                for g in GROUPS:line(ax,step,abs(data[f'e_{g}'][:,j]),g,COLORS[g])
                ylabel='Target MAE (rad)'
            elif kind=='key_targets':
                for g in GROUPS:line(ax,step,data[f'q_{g}'][:,j],g,COLORS[g])
                line(ax,step,data['gT'][:,j],'Flat(pT)','gray','--')
                line(ax,step,data['gS'][:,j],'Flat(pS)','#D55E00','--')
                ylabel='Mean target (rad)'
            else:
                for key,label,color in [('P_error','P residual','#D55E00'),('V','V','#009E73'),('I','Interaction','#CC79A7')]:
                    line(ax,step,abs(data[key][:,j]),label,color)
                ylabel='Mean abs factor effect (rad)'
            ax.set_title(name.replace('_joint',''),fontsize=10);decorate(ax,ylabel)
        for ax in list(axs.flat)[len(keynames):]:ax.axis('off')
        axs.flat[0].legend(fontsize=7)
        fig.suptitle(f'S1-D2 seed {seed}: {kind.replace("_"," ")}; changing valid sample count shown separately')
        fig.savefig(out/f'{kind}_seed{seed}.png',dpi=150);plt.close(fig)

def analyze(out,device):
    before={k:sha(p) for k,p in CHECKPOINTS.items()}
    teacher=model_from_checkpoint(s0.FLAT,CHECKPOINTS['flat'],device)
    student=model_from_checkpoint(s0.TASK,CHECKPOINTS['s01'],device)
    digests=dict(teacher=digest_state(teacher),student=digest_state(student))
    audits={};arrays=[]
    for seed in (142,314):
        print(f'Offline query seed={seed}',flush=True)
        t=np.load(out/f'inputs_{seed}_flat.npz');s=np.load(out/f'inputs_{seed}_s01.npz')
        tmap={(int(k),int(e)):i for i,(k,e) in enumerate(zip(t['step'],t['env_id']))}
        smap={(int(k),int(e)):i for i,(k,e) in enumerate(zip(s['step'],s['env_id']))}
        assert len(tmap)==len(t['step']) and len(smap)==len(s['step'])
        keys=sorted(tmap.keys()&smap.keys())
        ti=np.array([tmap[k] for k in keys]);si=np.array([smap[k] for k in keys])
        for source in (t,s):
            for eid in range(16):
                idx=np.flatnonzero(source['env_id']==eid)
                assert np.array_equal(source['step'][idx],np.arange(len(idx)))
                assert not source['done_after'][idx[:-1]].any(),'Post-terminal input leaked'
        for key in ('step','env_id','input_time_s','sim_time_s','sense_physics_step','episode_length'):
            assert np.array_equal(t[key][ti],s[key][si]),f'Time alignment failed: {key}'
        assert np.all(t['episode_length'][ti]==t['step'][ti])
        audit=json.loads((out/f'audit_{seed}_s01.json').read_text())
        teacher_audit=json.loads((out/f'audit_{seed}_flat.json').read_text())
        for key in ('joint_names','scale','offset','bias'):
            assert audit[key]==teacher_audit[key],f'Action schema differs: {key}'
        names=audit['joint_names'];env_ids=s['env_id'][si]
        scale=np.broadcast_to(np.array(audit['scale'],np.float32),(16,29))[env_ids]
        offset=np.broadcast_to(np.array(audit['offset'],np.float32),(16,29))[env_ids]
        bias=np.array(audit['bias'],np.float32)[env_ids]
        pT=t['actor'][ti];pS=s['actor'][si];dT=t['depth'][ti];dS=s['depth'][si]
        outputs={g:forward(student,p,d,device) for g,p,d in [('TT',pT,dT),('TS',pT,dS),('ST',pS,dT),('SS',pS,dS)]}
        gt=forward(teacher,pT,None,device);gs=forward(teacher,pS,None,device)
        checks={'TT_raw_max_abs':float(abs(outputs['TT']-t['s01_raw'][ti]).max()),
                'SS_raw_max_abs':float(abs(outputs['SS']-s['s01_raw'][si]).max()),
                'gT_raw_max_abs':float(abs(gt-t['flat_raw'][ti]).max()),
                'gS_raw_max_abs':float(abs(gs-s['flat_raw'][si]).max())}
        assert max(checks.values())<2e-5,checks
        data={'seed':np.full(len(keys),seed,np.int32),'step':s['step'][si],
              'env_id':s['env_id'][si],'input_time_s':s['input_time_s'][si],
              'gT':gt*scale+offset-bias,'gS':gs*scale+offset-bias,
              'depth_pair_rms_m':rms((dS-dT).reshape(len(keys),-1))}
        for g,a in outputs.items():
            data[f'q_{g}']=a*scale+offset-bias
            data[f'e_{g}']=(a-(gt if g[0]=='T' else gs))*scale
        A,B,C,D=[data[f'q_{g}'] for g in GROUPS]
        data.update(P_raw=C-A,V=B-A,I=D-C-B+A,teacher_response=data['gS']-data['gT'])
        data['P_error']=data['e_ST']-data['e_TT']
        data['delta_error']=data['e_SS']-data['e_TT']
        data['P_shapley']=data['P_error']+data['I']/2
        data['V_shapley']=data['V']+data['I']/2
        data['P_raw_at_student_depth']=D-B
        data['V_at_student_proprio']=D-C
        data['P_raw_average']=data['P_raw']+data['I']/2
        closure=float(abs(data['P_error']+data['V']+data['I']-data['delta_error']).max())
        assert closure<2e-6
        assert all(np.isfinite(v).all() for v in data.values())
        np.savez_compressed(out/f'cross_queries_{seed}.npz',**data)
        audits[str(seed)]=dict(paired_samples=len(keys),all_source_samples_student_paired=len(keys)==len(s['step']),
          capture_counts=dict(flat=len(t['step']),s01=len(s['step'])),time_alignment_exact=True,
          no_post_terminal_inputs=True,offline_forward_reproduction=checks,algebra_closure_max_abs_rad=closure,
          phase_pair_max_abs=float(abs(pT[:,9:11]-pS[:,9:11]).max()))
        arrays.append(data)
        plot_all(out,seed,data,names)
        # Raw, aligned measured depth illustrations, no substitutions or clipping in inference.
        if seed==314:
            fig,axs=plt.subplots(2,4,figsize=(14,6),constrained_layout=True)
            for col,step in enumerate((0,20,40,50)):
                n=keys.index((step,12))
                for row,(d,label) in enumerate(((dT,'Teacher'),(dS,'Student'))):
                    im=axs[row,col].imshow(d[n,0],vmin=0,vmax=3,cmap='viridis')
                    axs[row,col].set_title(f'{label}, t={step*.02:.2f}s');axs[row,col].axis('off')
            fig.colorbar(im,ax=axs.ravel().tolist(),label='Optical-Z depth (m); display range 0-3m only',shrink=.7)
            fig.suptitle('Seed 314 env 12: actual depth inputs captured before each action')
            fig.savefig(out/'paired_depth_seed314_env12.png',dpi=160);plt.close(fig)
        del pT,pS,dT,dS,t,s
    combined={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    window_rows=[];factor_rows=[];joint_rows=[];factor_joint_rows=[];focus_env=[];attribution=[]
    rng=np.random.default_rng(20261104)
    for seed in (142,314,'pooled'):
        for window,(lo,hi) in WINDOWS.items():
            keep=(combined['step']>=lo)&(combined['step']<hi)
            if seed!='pooled':keep &= combined['seed']==seed
            if not keep.any():continue
            d={k:v[keep] for k,v in combined.items()};n=len(d['step'])
            for group in GROUPS:
                e=d[f'e_{group}']
                window_rows.append(dict(seed=seed,window=window,group=group,n=n,**metrics(e)))
                for j,name in enumerate(names):
                    joint_rows.append(dict(seed=seed,window=window,group=group,joint=name,n=n,
                      mae_rad=float(abs(e[:,j]).mean()),signed_error_rad=float(e[:,j].mean()),rmse_rad=float(np.sqrt(np.mean(e[:,j]**2)))))
            for factor in ('P_raw','P_raw_at_student_depth','P_raw_average','teacher_response','P_error',
                           'V','V_at_student_proprio','I','P_shapley','V_shapley','delta_error'):
                e=d[factor];factor_rows.append(dict(seed=seed,window=window,factor=factor,n=n,**metrics(e)))
                for j,name in enumerate(names):
                    factor_joint_rows.append(dict(seed=seed,window=window,factor=factor,joint=name,n=n,
                      mae_rad=float(abs(e[:,j]).mean()),signed_rad=float(e[:,j].mean())))
            # Signed attribution of residual-change energy, not independent variance or causal fractions.
            energy=float(np.mean(d['delta_error']**2))
            p_share=float(np.mean(d['P_shapley']*d['delta_error'])/energy) if energy>1e-15 else 0
            v_share=float(np.mean(d['V_shapley']*d['delta_error'])/energy) if energy>1e-15 else 0
            attribution.append(dict(seed=seed,window=window,n=n,
              teacher_depth_error_reduction=1-float(rms(d['e_ST']).mean()/rms(d['e_SS']).mean()),
              P_signed_residual_change_energy_share=p_share,V_signed_residual_change_energy_share=v_share,
              interaction_to_delta_pooled_rms=float(np.sqrt(np.mean(d['I']**2)/energy)) if energy>1e-15 else 0,
              student_teacher_proprio_response_cosine=float(np.sum(d['P_raw']*d['teacher_response'])/
                max(float(np.sqrt(np.sum(d['P_raw']**2)*np.sum(d['teacher_response']**2))),1e-15)),
              teacher_direction_projection_gain=float(np.sum(d['P_raw']*d['teacher_response'])/
                max(float(np.sum(d['teacher_response']**2)),1e-15))))
            if window=='focus_.4-1':
                for sid in np.unique(d['seed']):
                    for eid in np.unique(d['env_id']):
                        kk=(d['seed']==sid)&(d['env_id']==eid)
                        if not kk.any():continue
                        C=float(rms(d['e_ST'][kk]).mean());D=float(rms(d['e_SS'][kk]).mean())
                        if seed!='pooled':focus_env.append(dict(seed=int(sid),env_id=int(eid),n=int(kk.sum()),
                          ST_error=C,SS_error=D,reduction_fraction=1-C/D,delta_error=C-D))
    intervals={}
    for seed in (142,314,'pooled'):
        ee=[r for r in focus_env if seed=='pooled' or r['seed']==seed]
        mat=np.array([[r['ST_error'],r['SS_error']] for r in ee]);idx=rng.integers(0,len(mat),size=(5000,len(mat)))
        boot=1-mat[idx,0].mean(axis=1)/mat[idx,1].mean(axis=1)
        intervals[str(seed)]={'env_count':len(ee),'environments_improved':sum(r['delta_error']<0 for r in ee),
          'reduction_bootstrap_env_95pct':np.quantile(boot,[.025,.975]).tolist(),
          'point_reduction':float(1-mat[:,0].mean()/mat[:,1].mean())}
    long=[]
    for n in range(len(combined['step'])):
        for g in GROUPS:
            e=combined[f'e_{g}'][n]
            long.append(dict(seed=int(combined['seed'][n]),env_id=int(combined['env_id'][n]),step=int(combined['step'][n]),
              input_time_s=float(combined['input_time_s'][n]),group=g,teacher_reference='pT' if g[0]=='T' else 'pS',
              target_rmse_rad=float(rms(e)),target_rad=json.dumps(combined[f'q_{g}'][n].tolist()),error_rad=json.dumps(e.tolist())))
    for file,rs in [('window_errors.csv',window_rows),('factor_effects.csv',factor_rows),('joint_errors.csv',joint_rows),
      ('joint_factors.csv',factor_joint_rows),('focus_per_env.csv',focus_env),('attribution.csv',attribution),('four_queries.csv',long)]:csv_write(out/file,rs)
    assert digests==dict(teacher=digest_state(teacher),student=digest_state(student))
    assert before=={k:sha(p) for k,p in CHECKPOINTS.items()}
    summary=dict(groups=LABELS,reference='TT/TS vs Flat(pT), ST/SS vs Flat(pS)',
      focus_window='integer steps [20,50), input time [0.4,1.0) s; 30 samples/env',
      decomposition='Praw=C-A; V=B-A; I=D-C-B+A; Perr=(C-gS)-(A-gT); eD-eA=Perr+V+I',
      energy_share_warning='Signed Shapley allocation of paired residual-change energy, not independent variance or closed-loop causal probability; may be negative',
      offline_depth_stride=[14400,1,160,1],
      audits=audits,window_errors=window_rows,factor_effects=factor_rows,attribution=attribution,
      focus_teacher_depth_reduction=intervals,checkpoint_hashes=before,model_states_unchanged=True,
      no_training=True,cross_actions_executed=False)
    dump(out/'analysis_summary.json',summary)
    print(json.dumps({'audits':audits,'focus':intervals},indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    with torch.inference_mode():analyze(args.out,args.device)

if __name__=='__main__':main()
