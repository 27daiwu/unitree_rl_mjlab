"""S1-D2: lossless, time-stamped paired observations; no training or checkpoint writes.

Only natural Flat/S0.1 actions reach the simulator. Crossed inputs are queried by
an independent offline analyzer. Uses the existing S1-D1 evaluation protocol.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
import train_g1_visual_stairs_v1a_s0 as s0
from diagnose_g1_visual_stairs_v1a_s1_d1 import digest_state, sha
from src.tasks.velocity.config.g1.visual_stairs_v1a_env_cfg import raw_camera_depth, DEPTH_CAMERA_NAME

OUT = Path('logs/experiments/g1_visual_stairs_v1a_s1_d2_20261104')
D1 = Path('logs/experiments/g1_visual_stairs_v1a_s1_d1_20261104')
CHECKPOINTS = {'flat':s0.FLAT_SOURCE,
 's01':Path('logs/experiments/g1_visual_stairs_v1a_s01_20261104/v1a_s0_1_bc_model.pt')}
PROTECTED = list(CHECKPOINTS.values()) + [
 Path('src/tasks/velocity/config/g1/visual_stairs_v1a_env_cfg.py'),
 Path('src/tasks/velocity/config/g1/stairs_terrain_cfg.py'),
 Path('src/tasks/velocity/config/g1/stairs_baseline_env_cfg.py'),
 Path('src/tasks/velocity/config/g1/rl_cfg.py'),
 Path('src/tasks/velocity/mdp/rewards.py')]

def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def capture(seed, driver, out, device):
    env, runner = s0.make_env(seed,16,device)
    try:
        vec=runner.env
        models={}
        for label,checkpoint in CHECKPOINTS.items():
            task=s0.FLAT if label=='flat' else s0.TASK
            other=load_runner_cls(task)(vec,asdict(load_rl_cfg(task)),device=device)
            model=other.alg.get_policy()
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['actor_state_dict'],strict=True)
            model.eval()
            for param in model.parameters():param.requires_grad_(False)
            models[label]=model
        digests={k:digest_state(m) for k,m in models.items()}
        robot=env.scene['robot'];term=env.action_manager.get_term('joint_pos')
        joint_ids=term.target_ids
        scale=s0.arr(term.scale);offset=s0.arr(term.offset)
        bias=s0.arr(robot.data.encoder_bias[:,joint_ids])
        audit={'seed':seed,'driver':driver,'step_dt':env.step_dt,'physics_dt':env.physics_dt,
               'decimation':env.cfg.decimation,'joint_names':list(term.target_names),
               'scale':scale.tolist(),'offset':offset.tolist(),'bias':bias.tolist(),
               'clip_actions':vec.clip_actions,'model_before':digests,
               'training_modes':{k:m.training for k,m in models.items()},
               'normalizer_modes':{k:m.obs_normalizer.training for k,m in models.items()},
               'actor_terms':list(env.observation_manager.active_terms['actor'])}
        # Stamp actual sensing calls without changing their state or frequency after reset.
        original_sense=env.sim.sense
        stamp={};sense_count=0
        def sense():
            nonlocal sense_count
            original_sense();sense_count+=1
            stamp.update(sense_id=sense_count,physics_step=int(env._sim_step_counter),
              sim_time=s0.arr(env.sim.data.time),qpos=s0.arr(env.sim.data.qpos))
        env.sim.sense=sense
        env.sim.sense()  # Same reset state, no integration; establishes initial frame stamp.
        obs=vec.get_observations();s0.set_command(env,obs)
        initial={k:s0.arr(v) for k,v in dict(qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,
          origins=env.scene.env_origins,actor=obs['actor'],depth=obs['depth'],
          encoder_bias=robot.data.encoder_bias,friction=env.sim.model.geom_friction,
          body_ipos=env.sim.model.body_ipos).items()}
        np.savez_compressed(out/f'initial_{seed}_{driver}.npz',**initial)
        old=np.load(D1/f'initial_{seed}_{driver}.npz')
        audit['initial_equal_d1']={k:bool(np.array_equal(v,old[k])) for k,v in initial.items()}
        assert all(audit['initial_equal_d1'].values()),'D1 initial condition mismatch'
        active=np.ones(16,bool)
        buffers={k:[] for k in ('step','env_id','input_time_s','sim_time_s','sense_id','sense_physics_step',
          'episode_length','actor','depth','qpos_input','flat_raw','s01_raw','done_after','fell_after','post_tilt_deg')}
        original_metrics=env.metrics_manager.compute
        post={};expected=None
        def metrics():
            original_metrics()
            post.update(done=s0.arr(env.reset_buf).astype(bool),
              fell=s0.arr(env.termination_manager.get_term('fell_over')).astype(bool),
              tilt=s0.arr(torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1)))))
            actual=s0.arr(robot.data.joint_pos_target[:,joint_ids])
            err=float(abs(actual[active]-expected[active]).max())
            post['mapping_error']=err
        env.metrics_manager.compute=metrics
        max_depth_error=0.;max_mapping_error=0.;max_time_error=0.
        origin_time=s0.arr(env.sim.data.time).copy()
        for step in range(150):
            if not active.any():break
            s0.set_command(env,obs)
            ids=np.flatnonzero(active)
            ep=s0.arr(env.episode_length_buf)
            assert np.all(ep[ids]==step),'Episode clock mismatch or reset leakage'
            assert stamp['physics_step']==int(env._sim_step_counter)==step*env.cfg.decimation
            qpos=s0.arr(env.sim.data.qpos)
            assert np.array_equal(stamp['qpos'][ids],qpos[ids]),'Depth/proprio qpos stamp mismatch'
            sim_time=s0.arr(env.sim.data.time)
            assert np.array_equal(stamp['sim_time'][ids],sim_time[ids]),'Sensing time mismatch'
            max_time_error=max(max_time_error,float(abs(sim_time[ids]-origin_time[ids]-step*env.step_dt).max()))
            depth_error=float((obs['depth']-raw_camera_depth(env,DEPTH_CAMERA_NAME)).abs().max())
            max_depth_error=max(max_depth_error,depth_error)
            assert depth_error==0,'Observation depth differs from current camera frame'
            outputs={k:m(obs) for k,m in models.items()}
            actor=s0.arr(obs['actor']);depth=s0.arr(obs['depth'])
            assert np.isfinite(actor).all() and np.isfinite(depth).all()
            for k,v in {'step':np.full(len(ids),step,dtype=np.int32),'env_id':ids,
                'input_time_s':np.full(len(ids),step*env.step_dt,dtype=np.float64),
                'sim_time_s':sim_time[ids], 'sense_id':np.full(len(ids),stamp['sense_id'],dtype=np.int32),
                'sense_physics_step':np.full(len(ids),stamp['physics_step'],dtype=np.int32),
                'episode_length':ep[ids], 'actor':actor[ids], 'depth':depth[ids],
                'qpos_input':qpos[ids],'flat_raw':s0.arr(outputs['flat'])[ids],
                's01_raw':s0.arr(outputs['s01'])[ids]}.items():buffers[k].append(v)
            expected=s0.arr(outputs[driver])*scale+offset-bias
            obs,_,_,_=vec.step(outputs[driver])
            for k,v in {'done_after':post['done'][ids],'fell_after':post['fell'][ids],
                        'post_tilt_deg':post['tilt'][ids]}.items():buffers[k].append(v)
            max_mapping_error=max(max_mapping_error,post['mapping_error'])
            active &= ~post['done']
        data={k:np.concatenate(v) for k,v in buffers.items()}
        assert max_depth_error==0 and max_mapping_error==0
        audit.update(max_input_depth_difference=max_depth_error,max_action_mapping_error_rad=max_mapping_error,
          max_sim_clock_roundoff_s=max_time_error,samples=len(data['step']),sense_calls=sense_count,
          model_after={k:digest_state(m) for k,m in models.items()})
        assert audit['model_before']==audit['model_after']
        # Check the original D1 path, not just its initial condition.
        previous=np.load(D1/f'snapshots_{seed}_{driver}.npz')
        lookup={(round(float(t),6),int(i)):n for n,(t,i) in enumerate(zip(previous['time'],previous['env_id']))}
        old_ids=np.array([lookup[round(float(t),6),int(i)] for t,i in zip(data['input_time_s'],data['env_id'])])
        audit['d1_replay_check']={'sample_count_equal':len(previous['time'])==len(data['step']),
          'actor_max_abs':float(abs(previous['actor'][old_ids]-data['actor']).max()),
          'flat_raw_max_abs':float(abs(previous['teacher_action'][old_ids]-data['flat_raw']).max()),
          's01_raw_max_abs':float(abs(previous['s01_action'][old_ids]-data['s01_raw']).max())}
        np.savez_compressed(out/f'inputs_{seed}_{driver}.npz',**data)
        dump(out/f'audit_{seed}_{driver}.json',audit)
        print(f'CAPTURE seed={seed} driver={driver} samples={len(data["step"])} replay={audit["d1_replay_check"]}',flush=True)
    finally:
        env.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    before={str(p):sha(p) for p in PROTECTED}
    old=json.loads((D1/'integrity.json').read_text())['checkpoints_before']
    assert all(before[str(p)]==old[k] for k,p in CHECKPOINTS.items())
    with torch.inference_mode():
        for seed in (142,314):
            for driver in ('flat','s01'):capture(seed,driver,args.out,args.device)
    pairs={}
    for seed in (142,314):
        teacher=np.load(args.out/f'initial_{seed}_flat.npz');student=np.load(args.out/f'initial_{seed}_s01.npz')
        pairs[str(seed)]={k:bool(np.array_equal(teacher[k],student[k])) for k in teacher.files}
    assert all(all(v.values()) for v in pairs.values())
    after={str(p):sha(p) for p in PROTECTED};assert before==after
    dump(args.out/'integrity.json',{'protected_before':before,'protected_after':after,'initial_pairs_equal':pairs,
      'observation_contract':'actor/depth captured together before action; render qpos/time matches input; one integer step key; terminal successor/reset observations excluded',
      'cross_inputs_executed':False,'training_performed':False})
    print('S1-D2 lossless paired capture complete.',flush=True)

if __name__=='__main__':main()
