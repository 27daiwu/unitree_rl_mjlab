"""Bounded dance38100 CPU interface acceptance. Never integrates physics.
Run with the fixed SuperMimic venv; --out is an existing archive directory.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,json,time,hashlib,re,traceback,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];EXT=ROOT/'extensions/supermimic'
OFFICIAL=Path('/home/unitree/workwpace/unitree_ws/unitree_mujoco/unitree_robots/g1/g1_29dof.xml')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def encode(v):
 if isinstance(v,Path):return {'type':type(v).__name__,'value':str(v)}
 if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [encode(x) for x in v]
 if v is None or type(v) in (int,float,str,bool):return v
 raise TypeError(type(v).__name__)
def config_audit(left,right,npz):
 rows=[]
 def visit(a,b,key):
  if isinstance(a,dict) and isinstance(b,dict):
   for k in sorted(set(a)|set(b)):
    if k not in a or k not in b:rows.append({'field':key+k,'status':'FAIL','reason':'missing field'})
    else:visit(a[k],b[k],key+k+'.')
   return
  field=key.rstrip('.');path=None
  if field=='data.npz_dir':
   pa=(EXT/Path(a)).resolve(strict=True);pb=(EXT/Path(b)).resolve(strict=True)
   ok=pa.is_dir() and pb.is_dir() and pa.samefile(pb) and pa==npz.resolve().parent
   path={'left_base':str(EXT),'right_base':str(EXT),'left_directory':str(pa),'right_directory':str(pb),'reference_sha256':sha(npz)}
  elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):ok=list(a)==list(b)
  else:ok=type(a) is type(b) and a==b
  metadata=field in {'log_root','run_name','device','seed','resume'}
  rows.append({'field':field,'embedded':encode(a),'embedded_type':type(a).__name__,'json':encode(b),'json_type':type(b).__name__,'status':'PASS' if ok or metadata else 'FAIL','metadata':metadata,'path_evidence':path})
 visit(left,right,'');return {'passed':all(x['status']=='PASS' for x in rows),'fields':rows}
def main(out):
 out=Path(out);start=json.loads((out/'clock_start.json').read_text())['start_epoch']
 counts={'checkpoint_load_attempts':0,'checkpoint_load_successes':0,'strict_bindings':0,'inferences':0,'mj_forward':0,'mj_step_attempts':0,'mj_step':0}
 result={'status':'RUNNING','counts':counts,'checks':{},'oracle':'NOT_AVAILABLE','S9C':'BLOCKED'}
 def write(n,v):(out/n).write_text(json.dumps(encode(v),indent=2,allow_nan=False)+'\n')
 def save():write('result.json',result)
 def check(name,error,tol=1e-5,**extra):
  result['checks'][name]={'status':'PASS' if error<=tol else 'FAIL','max_abs_error':float(error),'atol':tol,'rtol':0,**extra};save()
  if error>tol:raise AssertionError(name+' exceeds original tolerance')
 save()
 try:
  import numpy as np,torch,mujoco
  from scipy.spatial.transform import Rotation as R
  from supermimic.validation.cpu_interface_helpers import scipy_fk,history_pack,goal_scipy
  from supermimic.policy.nets import Policy,PolicyCfg
  from supermimic.envs.mdp.motion_library import RefState,goal_features
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as constants
  from mjlab.utils.lab_api.math import subtract_frame_transforms,matrix_from_quat
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  assert sys.executable==str(EXT/'.venv/bin/python')
  def forbidden(*args,**kwargs):
   counts['mj_step_attempts']+=1;raise RuntimeError('Physical integration forbidden')
  for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,forbidden)
  suite=unittest.defaultTestLoader.discover(str(EXT/'tests'),pattern='test_cpu_interface_helpers.py')
  with (out/'tests.log').open('w') as stream:tested=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
  result['tests']={'run':tested.testsRun,'failures':len(tested.failures),'errors':len(tested.errors),'passed':tested.wasSuccessful()};save();assert tested.wasSuccessful()
  ckpath=EXT/'logs/phase2_s5b_dance1_subject1_anchor_2h_20261105/ckpt_final_0038100.pt';npz=EXT/'data/npz/lafan1/dance1_subject1.npz';cfgpath=ckpath.parent/'config.json';trainxml=ROOT/'src/assets/robots/unitree_g1/xmls/g1.xml'
  expected={str(ckpath):'2f7859f591bc2a9b2b1c19f6f666bf2eea8cfc1ef913c0309e0c404f1cbd0472',str(npz):'d4dc0c57f9ac445a7910d8f9cf420fb0ad06d0e0e40c139ea43db4d2481304ce',str(OFFICIAL):'423e28bd718b19f7a65cda539b6f794ddbb268b4b9bdbd85f4bd982b30729617'}
  before={p:sha(p) for p in [ckpath,npz,cfgpath,OFFICIAL,trainxml,Path(constants.__file__)]};write('asset_hashes.json',{str(k):v for k,v in before.items()});assert all(before[Path(k)]==v for k,v in expected.items())
  cfg=json.loads(cfgpath.read_text());assert cfg['data']['clips']==['dance1_subject1'] and cfg['policy']['history']==10 and cfg['policy']['goal_dim']==689 and cfg['env']['future_frames']==10 and cfg['env']['anchor_obs'] and not cfg['env']['compact_library']
  counts['checkpoint_load_attempts']+=1;ck=torch.load(ckpath,map_location='cpu',weights_only=False);counts['checkpoint_load_successes']+=1;result['checkpoint_iteration']=ck.get('iteration');audit=config_audit(ck['config'],cfg,npz);write('config_audit.json',audit);assert audit['passed'];result['live_config']='PASS';save()
  pcfg=PolicyCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in cfg['policy'].items()});policy=Policy(pcfg).cpu().eval();policy.load_state_dict(ck['policy'],strict=True);counts['strict_bindings']+=1;del ck
  model=mujoco.MjModel.from_xml_path(str(OFFICIAL));training=mujoco.MjModel.from_xml_path(str(trainxml));data=mujoco.MjData(model)
  with np.load(npz) as z:
   names=z['joint_names'].astype(str).tolist();bn=z['body_names'].astype(str).tolist();pelvis,anchor=bn.index('pelvis'),bn.index('torso_link');fps=float(z['fps'][0]);assert fps==50.
   ref=np.concatenate([z['body_pos_w'][:,pelvis],z['body_quat_w'][:,pelvis],z['joint_pos'],z['body_lin_vel_w'][:,pelvis],z['body_ang_vel_w'][:,pelvis],z['joint_vel']],axis=1).astype(np.float32)
   apos=z['body_pos_w'][:,anchor].copy();aquat=z['body_quat_w'][:,anchor].copy();result['reference']={'fps':fps,'input_fps':z['input_fps'].tolist() if 'input_fps' in z else None,'frames':len(ref),'online_resampling':False,'control_dt':.02}
  actual=[model.joint(i).name for i in range(1,model.njnt)];assert names==actual==[training.joint(i).name for i in range(1,training.njnt)];result['joint_names']=names
  def match(mapping,default=0.):
   values=[]
   for n in names:
    hits=[v for pat,v in mapping.items() if re.fullmatch(pat,n)];assert len(hits)<=1;values.append(hits[0] if hits else default)
   return np.array(values,np.float32)
  q0=match(constants.KNEES_BENT_KEYFRAME.joint_pos);scale=match(constants.G1_ACTION_SCALE);assert (scale>0).all();offset=q0.copy();qadr=np.array([model.joint(n).qposadr[0] for n in names]);vadr=np.array([model.joint(n).dofadr[0] for n in names])
  lim=training.jnt_range[1:];mid=lim.mean(axis=1);half=.9*(lim[:,1]-lim[:,0])/2;data.qpos[:7]=ref[0,:7];data.qpos[qadr]=np.clip(ref[0,7:36],mid-half,mid+half);data.qvel[:3]=ref[0,36:39];data.qvel[3:6]=R.from_quat(ref[0,[4,5,6,3]]).inv().apply(ref[0,39:42]);data.qvel[vadr]=ref[0,42:71];mujoco.mj_forward(model,data);counts['mj_forward']+=1
  body=model.body('torso_link').id;pos,rot=scipy_fk(model,data.qpos);check('official_fk_position',np.max(abs(pos-data.xpos)));check('official_fk_rotation',np.max(abs(rot-data.xmat.reshape(-1,3,3))))
  split=RefState(*(torch.from_numpy(ref[:,a:b]) for a,b in [(0,3),(3,7),(7,36),(36,39),(39,42),(42,71)]));features=goal_features(split,torch.from_numpy(q0)).numpy();independent=goal_scipy(ref,q0);check('reference_future_features',np.max(abs(features-independent)))
  history=None;previous=np.zeros(29,np.float32);termhistory=None;arrays={'qpos':data.qpos.copy(),'qvel':data.qvel.copy(),'q0':q0,'scale':scale,'offset':offset,'fk_positions':pos,'fk_rotations':rot}
  for frame in [1,2]:
   rootrot=R.from_quat(data.qpos[[4,5,6,3]]);terms=[data.qpos[qadr].astype(np.float32)-q0,data.qvel[vadr].astype(np.float32),data.qvel[3:6].astype(np.float32),rootrot.inv().apply([0,0,-1.]).astype(np.float32),previous.copy()]
   history,obs=history_pack(history,terms);obs=obs.astype(np.float32)
   termhistory=[np.tile(v,(10,1)) for v in terms] if termhistory is None else [np.vstack([h[1:],v]) for h,v in zip(termhistory,terms)]
   check('history_frame'+str(frame),np.max(abs(obs-np.concatenate([h.ravel() for h in termhistory]))))
   t=lambda x:torch.tensor(np.asarray(x).copy(),dtype=torch.float32).reshape(1,-1)
   relp,relq=subtract_frame_transforms(t(data.xpos[body]),t(data.xquat[body]),t(apos[frame]),t(aquat[frame]));anch=np.r_[relp.numpy()[0],matrix_from_quat(relq).numpy()[0,:,:2].ravel()]
   refrot=R.from_quat(aquat[frame][[1,2,3,0]]).as_matrix();scipy_anchor=np.r_[rot[body].T@(apos[frame]-pos[body]),(rot[body].T@refrot)[:,:2].ravel()]
   idx=np.minimum(frame+np.arange(10),len(ref)-1);goal=np.r_[features[idx].ravel(),anch].astype(np.float32);assert obs.shape==(930,) and goal.shape==(689,)
   with torch.inference_mode():action=policy(torch.from_numpy(obs[None]),torch.from_numpy(goal[None])).numpy()[0]
   counts['inferences']+=1;assert action.shape==(29,);target=action*scale+offset
   for k,v in [('obs',obs),('goal',goal),('action',action),('target',target),('anchor',anch),('anchor_scipy',scipy_anchor)]:arrays[k+'_'+str(frame)]=v
   np.savez_compressed(out/'interface_values.npz',**arrays);save()
   assert all(np.isfinite(v).all() for v in [obs,goal,action,target]);check('anchor_frame'+str(frame),np.max(abs(anch-scipy_anchor)));check('future_goal_frame'+str(frame),np.max(abs(goal[:680]-independent[idx].ravel())))
   check('target_frame'+str(frame),np.max(abs(target-(action.astype(float)*scale.astype(float)+offset))),1e-4)
   expectedscale=np.array([.25*next(a.effort_limit/a.stiffness for a in constants.G1_ARTICULATION.actuators if any(re.fullmatch(p,n) for p in a.target_names_expr)) for n in names]);check('scale_formula',np.max(abs(scale-expectedscale)),1e-4)
   previous=action.copy()
  result['protected_asset_changes']=[str(p) for p,h in before.items() if sha(p)!=h];assert not result['protected_asset_changes'];result['cuda_initialized']=torch.cuda.is_initialized();assert not result['cuda_initialized']
  result['status']='PASS_INTERFACE_SELF_CONSISTENCY_ONLY';result['physical_acceptance']='NOT_RUN'
 except BaseException as exc:
  result['status']='FAIL';result['error']=repr(exc);result['traceback']=traceback.format_exc()
 finally:
  result['checks_elapsed_s']=time.time()-start;result['checks_budget_passed']=result['checks_elapsed_s']<=180
  result['inference_status']='PASS_EXECUTED' if counts['inferences'] else 'NOT_RUN';save();print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);main(parser.parse_args().out)
