"""No-policy diagnostic for the native G1 observation builder.

Run from repository root, in conda unitree_rl_mjlab:
  python -m simulate.policy_observation --dump logs/observation/flat.npz
No physics stepping, DDS, policy inference, or action output is performed.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import time
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAY_MODES = {'TRAINING_PARITY': 0, 'DEPLOYMENT_FILTERED': 1}
BLOCKS = {'base_ang_vel': (0,3), 'projected_gravity': (3,6), 'command': (6,9),
          'phase': (9,11), 'joint_pos': (11,40), 'joint_vel': (40,69),
          'previous_action': (69,98), 'height_scan': (98,274)}


def native_library():
    """Build against the SAME MuJoCo ABI as Python; never mix with bundled 3.3."""
    src = ROOT/'simulate/src'
    names = ('height_scan.h','policy_observation.h','policy_observation.cc','policy_observation_c_api.cc')
    mjdir = Path(mujoco.__file__).parent
    digest = hashlib.sha256((str(mjdir)+mujoco.__version__).encode())
    for name in names:
        digest.update((src/name).read_bytes())
    target = ROOT/'logs/observation/build'/digest.hexdigest()[:16]/'observation.so'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-shared','-fPIC',
                        str(src/'policy_observation.cc'),str(src/'policy_observation_c_api.cc'),
                        '-I'+str(mjdir/'include'),str(next(mjdir.glob('libmujoco.so*'))),
                        '-Wl,-rpath,'+str(mjdir),'-o',str(target)],check=True)
    lib = ct.CDLL(str(target))
    ptr = ct.c_void_p
    lib.observation_create.argtypes=[ptr,ct.c_char_p,ct.c_int];lib.observation_create.restype=ptr
    lib.observation_destroy.argtypes=[ptr];lib.observation_destroy.restype=None
    lib.observation_reset.argtypes=[ptr];lib.observation_reset.restype=None
    lib.observation_set_action.argtypes=[ptr,ptr];lib.observation_set_action.restype=ct.c_int
    lib.observation_build.argtypes=[ptr,ptr,ptr,ct.c_ulonglong,ct.c_double,ptr]
    lib.observation_build.restype=ct.c_int
    lib.observation_ray_hits.argtypes=[ptr,ptr,ptr,ptr]
    lib.observation_ray_hits.restype=ct.c_int
    lib.observation_error.restype=ct.c_char_p
    lib.observation_joint_name.argtypes=[ct.c_int];lib.observation_joint_name.restype=ct.c_char_p
    lib.observation_default_q.argtypes=[ct.c_int];lib.observation_default_q.restype=ct.c_float
    return lib


class Builder:
    def __init__(self, model, prefix='', ray_mode='DEPLOYMENT_FILTERED'):
        self.ray_mode=ray_mode
        mode=RAY_MODES[ray_mode]
        self.model=model  # retain model for lifetime of native builder
        self.lib=native_library()
        self.handle=self.lib.observation_create(model._address,prefix.encode(),mode)
        if not self.handle:
            raise ValueError(self.lib.observation_error().decode())
        self.previous_action=np.zeros(29,np.float32)
        self.prefix=prefix
        self.joint_names=[self.lib.observation_joint_name(i).decode() for i in range(29)]
        self.default_q=np.array([self.lib.observation_default_q(i) for i in range(29)])

    def close(self):
        if self.handle:
            self.lib.observation_destroy(self.handle);self.handle=None

    def _check(self, result):
        if result:
            raise ValueError(self.lib.observation_error().decode())

    def reset(self):
        self.lib.observation_reset(self.handle);self.previous_action.fill(0)

    def set_previous_action(self, action):
        a=np.ascontiguousarray(action,dtype=np.float32)
        if a.shape != (29,): raise ValueError('Expected 29 previous actions')
        self._check(self.lib.observation_set_action(self.handle,a.ctypes.data))
        self.previous_action=a.copy()

    def build(self, data, command=(0,0,0), episode_step=0, step_dt=.02):
        if not self.handle: raise ValueError('Builder is closed')
        if episode_step < 0: raise ValueError('episode_step must be nonnegative')
        cmd=np.ascontiguousarray(command,dtype=np.float32)
        if cmd.shape != (3,): raise ValueError('Expected vx, vy, yaw_rate')
        obs=np.empty(274,np.float32)
        self._check(self.lib.observation_build(self.handle,data._address,cmd.ctypes.data,
                                              episode_step,step_dt,obs.ctypes.data))
        return obs

    def ray_hits(self, data):
        geoms=np.empty(176,np.int32);heights=np.empty(176,np.float64)
        self._check(self.lib.observation_ray_hits(self.handle,data._address,geoms.ctypes.data,heights.ctypes.data))
        return geoms,heights

    def dump(self, path, data, obs, command, episode_step, step_dt):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        body=self.model.body(self.prefix+'pelvis').id
        np.savez(path,timestamp=time.time(),simulation_time=data.time,q=data.qpos.copy(),
                 dq=data.qvel.copy(),base_quaternion_wxyz=data.xquat[body].copy(),
                 base_ang_vel=obs[:3],command=np.asarray(command),
                 previous_action=self.previous_action,raw_height_scan=obs[98:]/np.float32(.2),
                 obs=obs,ray_mode=self.ray_mode,episode_step=episode_step,step_dt=step_dt)


def describe(obs):
    print('OBS_DIM =',len(obs))
    for name,(start,stop) in BLOCKS.items():
        a=obs[start:stop]
        print(f'BLOCK {name} [{start}:{stop}] min={a.min():.8g} max={a.max():.8g} mean={a.mean():.8g}')
    print('OBS_NAN_COUNT =',np.isnan(obs).sum())
    print('OBS_INF_COUNT =',np.isinf(obs).sum())
    print('HEIGHT_SCAN_DIM =',len(obs[98:]))
    print('POLICY_INFERENCE = OFF\nACTION_OUTPUT = OFF')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=ROOT/'src/assets/robots/unitree_g1/xmls/scene_g1.xml')
    p.add_argument('--ray-mode',choices=RAY_MODES,default='DEPLOYMENT_FILTERED')
    p.add_argument('--prefix',default='',help='Model object prefix; use robot/ for training snapshots')
    p.add_argument('--dump',type=Path,default=ROOT/'logs/observation/snapshot.npz')
    p.add_argument('--snapshot',type=Path,help='Replay q/dq, command, previous_action and phase from NPZ')
    p.add_argument('--command',type=float,nargs=3,default=[0,0,0])
    p.add_argument('--episode-step',type=int,default=0)
    p.add_argument('--step-dt',type=float,default=.02)
    args=p.parse_args()
    m=(mujoco.MjModel.from_binary_path(str(args.scene)) if args.scene.suffix=='.mjb'
       else mujoco.MjModel.from_xml_path(str(args.scene)))
    d=mujoco.MjData(m)
    builder=Builder(m,args.prefix,args.ray_mode)
    if args.snapshot:
        state=np.load(args.snapshot)
        d.qpos[:]=state['q'];d.qvel[:]=state['dq']
        args.command=state['command'];args.episode_step=int(state['episode_step']);args.step_dt=float(state['step_dt'])
        builder.set_previous_action(state['previous_action'])
    else:
        for name,q in zip(builder.joint_names,builder.default_q,strict=True):
            j=m.joint(args.prefix+name).id
            d.qpos[m.jnt_qposadr[j]]=q
        d.qpos[2]=.8
    mujoco.mj_forward(m,d)
    obs=builder.build(d,args.command,args.episode_step,args.step_dt)
    describe(obs)
    builder.dump(args.dump,d,obs,args.command,args.episode_step,args.step_dt)
    print('SNAPSHOT =',args.dump)
    builder.close()

if __name__=='__main__': main()
