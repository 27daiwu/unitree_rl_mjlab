"""Independent R9 bounded torque-control worker. Importing never runs a Policy/simulator.
R9A prepares/mock-tests this module only. Future execution requires --execute.
"""
import argparse,csv,json,time,traceback,os,sys
from pathlib import Path
import numpy as np
THRESHOLDS={'physics_dt_s':.002,'control_dt_s':.02,'cycles':2,'substeps':10,'torque_atol':1e-8,'joint_atol':1e-8,'contact_dim':3,'max_penetration_m':.005,'normal_force_min_N':-1e-8,'root_height_error_m':.25,'root_tilt_error':.8,'scope':'Unmodified R8 robot XML without ground; no ground stability/contact claim','original_environment_oracle':'NOT_AVAILABLE'}

def joint_mapping(joint_names,model_joint_names,actuator_joint_ids):
    if len(joint_names)!=29 or len(set(joint_names))!=29 or len(set(model_joint_names))!=len(model_joint_names):
        raise ValueError('29 unique action joints required')
    ids=[]
    for name in joint_names:
        if name not in model_joint_names:raise ValueError('Missing joint '+name)
        joint=model_joint_names.index(name)+(0 if len(model_joint_names)==30 else 1)
        matches=np.flatnonzero(np.asarray(actuator_joint_ids)==joint)
        if len(matches)!=1:raise ValueError('Missing/duplicate actuator '+name)
        ids.append(int(matches[0]))
    return np.array(ids,dtype=int)

def torque_command(target,q,qd,kp,kd,effort,lower,upper):
    values=[np.asarray(v,dtype=float) for v in (target,q,qd,kp,kd,effort,lower,upper)]
    if any(v.shape!=(29,) or not np.isfinite(v).all() for v in values):raise ValueError('Nonfinite or invalid torque inputs')
    target,q,qd,kp,kd,effort,lower,upper=values
    if np.any(effort<=0) or np.any(lower>=upper):raise ValueError('Invalid effort/official limits')
    raw=kp*(target-q)-kd*qd
    train=np.clip(raw,-effort,effort)
    applied=np.clip(train,lower,upper)
    return raw,train,applied

def run_control(backend,deadline,thresholds=None,now=time.time):
    t=dict(THRESHOLDS if thresholds is None else thresholds)
    if t['cycles']!=2 or t['substeps']!=10 or t['physics_dt_s']!=.002 or t['control_dt_s']!=.02:raise ValueError('Frozen protocol changed')
    counts={'inference_attempts':0,'policy_inferences':0,'target_updates':0,'torque_writes':0,'physics_attempts':0,'physics_steps':0}
    result={'status':'RUNNING','counts':counts,'rows':[],'controls':[],'S9C':'BLOCKED'}
    def guard():
        if now()>=deadline:raise TimeoutError('Absolute check deadline')
    def finite(value):
        return bool(np.isfinite(np.asarray(value)).all())
    def validate(state,inspection,applied=None):
        if any(not finite(state[k]) for k in ['q','qd','qpos','qvel','root_pos','root_quat','time']):raise ValueError('Nonfinite state')
        if inspection['warnings']:raise ValueError('MuJoCo warning')
        if not finite(inspection['force']):raise ValueError('Nonfinite actuator force')
        if inspection['joint_violation']>t['joint_atol']:raise ValueError('Joint hard limit violation')
        if inspection['root_height_error']>t['root_height_error_m'] or inspection['root_tilt_error']>t['root_tilt_error']:raise ValueError('Root termination')
        force=np.asarray(inspection['force'])
        if np.any(force<backend.lower-t['torque_atol']) or np.any(force>backend.upper+t['torque_atol']):raise ValueError('Actuator force exceeds effective range')
        if applied is not None and np.max(abs(force-applied))>t['torque_atol']:raise ValueError('Actual torque differs from command')
        for c in inspection['contacts']:
            if not finite([c['dist'],c['normal_force']]) or c['dim']!=t['contact_dim'] or c['dist'] < -t['max_penetration_m'] or c['normal_force']<t['normal_force_min_N']:raise ValueError('Abnormal contact')
    try:
        guard();initial=backend.state();validate(initial,backend.inspect());initial_time=float(initial['time'])
        for cycle in range(2):
            guard();counts['inference_attempts']+=1;action=np.asarray(backend.infer(cycle));counts['policy_inferences']+=1
            if action.shape!=(29,) or not finite(action):raise ValueError('Invalid Policy action')
            target=action*backend.scale+backend.offset
            if not finite(target):raise ValueError('Nonfinite target')
            counts['target_updates']+=1;result['controls'].append({'cycle':cycle,'action':action.copy(),'target':target.copy()})
            for substep in range(10):
                guard();pre=backend.state();raw,train,applied=torque_command(target,pre['q'],pre['qd'],backend.kp,backend.kd,backend.effort,backend.lower,backend.upper)
                backend.update_ctrl(applied);counts['torque_writes']+=1;guard();counts['physics_attempts']+=1;backend.step();counts['physics_steps']+=1
                post=backend.state();inspection=backend.inspect()
                row={'cycle':cycle,'substep':substep,'target':target.copy(),'raw':raw,'training_clipped':train,'ctrl':applied,'normal_saturation':np.abs(raw-applied)>t['torque_atol'],'pre':pre,'post':post,'inspection':inspection,'finite':all(finite(post[k]) for k in ['qpos','qvel']),'fault':None}
                result['rows'].append(row)
                try:
                    if abs(float(post['time'])-float(pre['time'])-.002)>1e-12:raise ValueError('Physics clock mismatch')
                    validate(post,inspection,applied)
                except BaseException as exc:row['fault']=repr(exc);raise
        if abs(float(backend.state()['time'])-initial_time-.04)>1e-12:raise ValueError('Control clock mismatch')
        result['status']='PASS_BOUNDED_CONTROL_LOOP'
    except BaseException as exc:
        result['status']='TIMEOUT' if isinstance(exc,TimeoutError) else 'FAIL';result['error']=repr(exc)
    return result

def json_value(v):
    if isinstance(v,np.ndarray):return json_value(v.tolist())
    if isinstance(v,np.generic):return json_value(v.item())
    if isinstance(v,dict):return {k:json_value(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [json_value(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return {'nonfinite_float':repr(v)}
    return v

def absolute_deadline(out):
    clock=json.loads((Path(out)/'clock_start.json').read_text())
    return min(clock['start_epoch']+180,clock.get('deadline_epoch',clock['start_epoch']+180))

def supervise(out):
    """Use the existing session/group supervisor; never invoke a backend here."""
    from supermimic.validation.dance1_supervisor import run_process
    out=Path(out);deadline=absolute_deadline(out)
    if time.time()>=deadline:
        result={'status':'REFUSED_EXPIRED','launched':False,'returncode':None}
    else:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        command=[sys.executable,'-m','supermimic.validation.dance1_physics',
                 '--out',str(out),'--execute','--worker-internal']
        result=run_process(command,deadline,out/'physics_worker.log',env)
    (out/'supervisor_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main(out,execute=False,_supervised=False):
    out=Path(out)
    if not execute:raise SystemExit('Preparation only; real execution requires a separate authorized --execute run')
    if not _supervised:return supervise(out)
    from supermimic.validation.dance1_physics_backend import MujocoBackend
    deadline=absolute_deadline(out)
    if time.time()>=deadline:raise SystemExit('BLOCKED: absolute deadline expired before setup')
    (out/'thresholds.json').write_text(json.dumps(THRESHOLDS,indent=2)+'\n')
    try:
        backend=MujocoBackend(out)
        result=run_control(backend,deadline)
    except BaseException as exc:
        result={'status':'FAIL_SETUP','error':repr(exc),'traceback':traceback.format_exc(),'rows':[],'counts':{},'S9C':'BLOCKED'}
    # Keep the failing step and all previous rows, including every vector/contact.
    (out/'result.json').write_text(json.dumps(json_value(result),indent=2,allow_nan=False)+'\n')
    rows=result['rows']
    if rows:
        np.savez_compressed(out/'substeps.npz',qpos=np.array([r['post']['qpos'] for r in rows]),qvel=np.array([r['post']['qvel'] for r in rows]),target=np.array([r['target'] for r in rows]),ctrl=np.array([r['ctrl'] for r in rows]),raw=np.array([r['raw'] for r in rows]),force=np.array([r['inspection']['force'] for r in rows]),root_pos=np.array([r['post']['root_pos'] for r in rows]),root_quat=np.array([r['post']['root_quat'] for r in rows]))
        with (out/'substeps.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['cycle','substep','time','contacts','joint_violation','finite','fault']);w.writerows((r['cycle'],r['substep'],r['post']['time'],len(r['inspection']['contacts']),r['inspection']['joint_violation'],r['finite'],r['fault']) for r in rows)
    print(result['status'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--worker-internal',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    outcome=main(a.out,a.execute,a.worker_internal)
    if isinstance(outcome,dict):
        raise SystemExit(outcome['returncode'] if outcome['status']=='EXITED' else 1)
