"""Direct native-position-actuator control, with no viewer or DDS dependency."""
from __future__ import annotations
import numpy as np
import mujoco
from simulate.policy_observation import Builder
from simulate.policy.inference import Policy


class SafetyAbort(RuntimeError):
    pass


class Suspension:
    """Always-on linear band: existing simulator k=200, c=100, length=0.

    Anchor height is chosen for one robot weight of support at HOME root z.
    As in the existing simulator, root translation drives the spring/damper
    and its world force is applied to torso_link. No automatic release.
    """
    def __init__(self, model, data):
        self.body=model.body('robot/torso_link').id
        self.stiffness=200.;self.damping=100.
        self.weight=float(model.body_mass.sum()*abs(model.opt.gravity[2]))
        self.anchor=data.qpos[:3].copy()+np.array([0,0,self.weight/self.stiffness])
        self.force=np.zeros(3)

    def apply(self, data):
        delta=self.anchor-data.qpos[:3]
        distance=np.linalg.norm(delta)
        if not np.isfinite(distance): raise SafetyAbort('nonfinite suspension state')
        direction=delta/distance if distance>1e-9 else np.zeros(3)
        self.force=(self.stiffness*distance-self.damping*np.dot(data.qvel[:3],direction))*direction
        data.xfrc_applied[self.body,:3]=self.force
        data.xfrc_applied[self.body,3:]=0
        return self.force.copy()


class PolicyController:
    def __init__(self, model, data, artifact, action_scale, action_offset, decimation,
                 velocity_limits, raw_action_limit):
        self.model=model;self.data=data
        self.builder=Builder(model,'robot/','TRAINING_PARITY')
        self.policy=Policy(artifact)
        self.scale=np.asarray(action_scale,np.float64);self.offset=np.asarray(action_offset,np.float64)
        self.decimation=int(decimation);self.policy_dt=model.opt.timestep*self.decimation
        self.joints=np.array([model.joint('robot/'+n).id for n in self.builder.joint_names])
        self.qadr=model.jnt_qposadr[self.joints];self.vadr=model.jnt_dofadr[self.joints]
        self.actuators=[]
        for j in self.joints:
            ids=np.flatnonzero((model.actuator_trnid[:,0]==j)&(model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT))
            if len(ids)!=1:raise ValueError('Expected one training position actuator per joint')
            self.actuators.append(int(ids[0]))
        self.actuators=np.array(self.actuators)
        self.kp=model.actuator_gainprm[self.actuators,0]
        self.kd=-model.actuator_biasprm[self.actuators,2]
        self.effort=model.actuator_forcerange[self.actuators,1]
        self.velocity=np.asarray(velocity_limits)
        self.raw_limit=float(raw_action_limit)
        self.error_limit=self.effort/self.kp  # static deflection at rated torque
        self.target_step_limit=self.velocity*self.policy_dt
        self.limits=model.jnt_range[self.joints]
        self.previous=np.zeros(29,np.float32);self.target=self.offset.copy()
        self.episode_step=0;self.inferences=0;self.physics_steps=0;self.aborted=False
        self.suspension=Suspension(model,data)
        self.rows=[];self.pending={}

    def finite(self, name, values):
        if not np.isfinite(values).all():raise SafetyAbort('nonfinite '+name)

    def check_state(self):
        d=self.data
        for name,x in [('q',d.qpos),('dq',d.qvel),('ctrl',d.ctrl),('force',d.actuator_force)]:self.finite(name,x)
        q=d.qpos[self.qadr];v=d.qvel[self.vadr]
        if np.any((q<self.limits[:,0])|(q>self.limits[:,1])):raise SafetyAbort('joint hard position limit')
        if np.any(abs(v)>self.velocity):raise SafetyAbort('joint rated velocity limit')
        if np.any(abs(d.actuator_force[self.actuators])>self.effort+1e-8):raise SafetyAbort('actuator effort limit')
        if np.any(abs(self.target-q)>self.error_limit):raise SafetyAbort('live q_target-q exceeds effort/Kp')
        if np.any(d.warning.number):raise SafetyAbort('MuJoCo warning: '+str(d.warning.number))

    def validate_action(self, action, target):
        self.finite('raw_action',action);self.finite('q_target',target)
        if np.max(abs(action))>self.raw_limit:raise SafetyAbort('historical raw action diagnostic limit')
        error=abs(target-self.data.qpos[self.qadr])
        if np.any(error>self.error_limit):raise SafetyAbort('q_target-q exceeds effort/Kp: '+str(np.flatnonzero(error>self.error_limit).tolist()))
        if np.any(abs(target-self.target)>self.target_step_limit):raise SafetyAbort('q_target jump exceeds rated velocity * policy_dt')

    def safe_stop(self):
        # ctrl=0 would drive all position actuators toward zero, not zero torque.
        # Freeze simulation, disable actuator forces in this model, retain band.
        self.aborted=True
        self.data.ctrl[:]=0
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        if np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all():
            self.suspension.apply(self.data)
            mujoco.mj_forward(self.model,self.data)
        else:
            # Invalid state must not enter the solver. Freeze; retain the last
            # finite external suspension force and disable all actuator forces.
            self.data.actuator_force[:]=0

    def physics_step(self):
        self.check_state()
        self.suspension.apply(self.data)
        self.data.ctrl[self.actuators]=self.target
        self.finite('ctrl',self.data.ctrl)
        mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        self.check_state()

    def settle(self, count=20):
        self.target=self.offset.copy()
        for _ in range(count):self.physics_step()

    def reset_episode(self):
        self.builder.reset();self.previous.fill(0);self.episode_step=0
        self.inferences=0;self.physics_steps=0;self.rows=[]

    def step(self, command):
        try:
            return self._step(command)
        except Exception:
            self.safe_stop()
            raise

    def _step(self, command):
        if self.aborted:raise SafetyAbort('controller already aborted')
        self.check_state()
        obs=self.builder.build(self.data,command,self.episode_step,self.policy_dt)
        self.finite('obs',obs)
        if not np.array_equal(obs[69:98],self.previous):raise SafetyAbort('previous_action mismatch')
        if np.linalg.norm(command)<.1 and np.any(obs[9:11]):raise SafetyAbort('zero command phase mismatch')
        action=self.policy(obs);self.inferences+=1
        target=self.offset+self.scale*action
        self.pending={'obs':obs.copy(),'raw_action':action.copy(),'q_target':target.copy()}
        self.validate_action(action,target)
        step_change=float(np.max(abs(target-self.target)))
        self.builder.set_previous_action(action);self.previous=action.copy();self.target=target
        row={'policy_step':self.episode_step,'sim_time':self.data.time,'command':np.array(command),
             'obs':obs,'raw_action':action,'q_target':target.copy(),'q':self.data.qpos[self.qadr].copy(),
             'dq':self.data.qvel[self.vadr].copy(),'root_position':self.data.qpos[:3].copy(),
             'root_quaternion':self.data.qpos[3:7].copy(),'max_q_target_step':step_change,
             'suspension_force':self.suspension.force.copy()}
        row.update(obs_min=float(obs.min()),obs_max=float(obs.max()),obs_l2=float(np.linalg.norm(obs)),
                   height_min=float(obs[98:].min()),height_max=float(obs[98:].max()),
                   raw_action_min=float(action.min()),raw_action_max=float(action.max()),raw_action_l2=float(np.linalg.norm(action)),
                   q_target_min=float(target.min()),q_target_max=float(target.max()))
        max_error=0.;max_velocity=0.;max_torque=0.;forces=[];support=[]
        for _ in range(self.decimation):
            self.physics_step();self.physics_steps+=1
            np.testing.assert_array_equal(self.data.ctrl[self.actuators],target)
            forces.append(self.data.actuator_force[self.actuators].copy())
            support.append(self.suspension.force.copy())
            expected_torque=np.clip(self.kp*(target-self.data.qpos[self.qadr])-self.kd*self.data.qvel[self.vadr],-self.effort,self.effort)
            np.testing.assert_allclose(forces[-1],expected_torque,rtol=0,atol=1e-8)
            max_error=max(max_error,float(np.max(abs(target-self.data.qpos[self.qadr]))))
            max_velocity=max(max_velocity,float(np.max(abs(self.data.qvel[self.vadr]))))
            max_torque=max(max_torque,float(np.max(abs(forces[-1]))))
        row.update(ctrl=self.data.ctrl[self.actuators].copy(),substep_torques=np.asarray(forces),substep_suspension_forces=np.asarray(support),
                   ctrl_min=float(target.min()),ctrl_max=float(target.max()),
                   torque_min=float(np.min(forces)),torque_max=float(np.max(forces)),
                   max_position_error=max_error,max_velocity=max_velocity,max_torque=max_torque)
        self.rows.append(row);self.episode_step+=1
        assert self.physics_steps==self.episode_step*self.decimation
        return row
