"""Suspension-only load transfer and read-only actual-contact telemetry.

The frozen PolicyController is composed, not edited. All dynamics and policy
settings except suspension rest length are retained.
"""
from collections import deque
import numpy as np
import mujoco
from simulate.policy.controller import Suspension, SafetyAbort


class AdjustableBand(Suspension):
    def __init__(self, model, data):
        super().__init__(model,data)
        self.length=0.

    def apply(self,data):
        delta=self.anchor-data.qpos[:3];distance=np.linalg.norm(delta)
        if not np.isfinite(distance):raise SafetyAbort('nonfinite band distance')
        direction=delta/distance if distance>1e-9 else np.zeros(3)
        # Tension-only slack extension. At length=0 this is the previous band
        # law throughout the audited positive-tension operating range.
        extension=distance-self.length
        magnitude=max(0.,self.stiffness*extension-self.damping*np.dot(data.qvel[:3],direction)) if extension>0 else 0.
        self.force=magnitude*direction
        data.xfrc_applied[self.body,:3]=self.force
        data.xfrc_applied[self.body,3:]=0
        return self.force.copy()


def contact_loads(model,data):
    """World force ON each robot foot from external geoms; internal contacts excluded."""
    forces=np.zeros(2);contact=np.zeros(2,dtype=bool);nonfoot=0
    for i in range(data.ncon):
        con=data.contact[i];g1,g2=map(int,con.geom)
        if g1<0 or g2<0:continue
        names=[model.body(int(model.geom_bodyid[g])).name for g in (g1,g2)]
        robot=[n.startswith('robot/') for n in names]
        if robot[0]==robot[1]:continue
        local=np.zeros(6);mujoco.mj_contactForce(model,data,i,local)
        if np.linalg.norm(local[:3])<=1e-6:continue
        # MuJoCo contact normal points geom1 -> geom2; force acts on geom2.
        world=np.asarray(con.frame).reshape(3,3).T@local[:3]
        side=1 if robot[1] else 0
        world*=1 if side==1 else -1
        body=names[side]
        if body in ('robot/left_ankle_roll_link','robot/right_ankle_roll_link'):
            foot=0 if '/left_' in body else 1
            forces[foot]+=world[2];contact[foot]=True
        else:nonfoot+=1
    return forces,contact,nonfoot


class LoadMonitor:
    def __init__(self,c,tilt_limit):
        self.c=c;self.m=c.model;self.d=c.data
        c.suspension=AdjustableBand(self.m,self.d)
        self.band=c.suspension
        self.root0=float(self.d.qpos[2]);self.tilt_limit=tilt_limit
        # Additional geometric low-height guard: HOME pelvis-to-knee drop.
        knees=[self.d.xpos[self.m.body('robot/'+side+'_knee_link').id,2] for side in ('left','right')]
        self.height_min=max(self.root0-z for z in knees)
        self.samples=[];self.section='settle';self.goal_length=0.
        # Rest-length speed transfers at most 5% robot weight per second in
        # fixed geometry: k * dlength/dt = 0.05 * robot_weight.
        self.length_rate=.05*self.band.weight/self.band.stiffness
        self.original_check=c.check_state;self.original_step=c.physics_step
        c.check_state=self.check
        c.physics_step=self.physics_step

    def measure(self):
        d=self.d;R=d.xmat[self.m.body('robot/pelvis').id].reshape(3,3)
        roll=np.arctan2(R[2,1],R[2,2]);pitch=np.arcsin(np.clip(-R[2,0],-1,1))
        tilt=np.arccos(np.clip(R[2,2],-1,1))
        feet,contacts,nonfoot=contact_loads(self.m,d)
        return {'time':float(d.time),'section':self.section,'length':self.band.length,
                'root_height':float(d.qpos[2]),'root_position':d.qpos[:3].copy(),
                'roll':float(roll),'pitch':float(pitch),'tilt':float(tilt),
                'base_linear_velocity':d.qvel[:3].copy(),'base_angular_velocity':d.qvel[3:6].copy(),
                'foot_fz':feet,'foot_contact':contacts,'nonfoot_contacts':nonfoot,
                'suspension_force':self.band.force.copy(),
                'vertical_support_ratio':float(max(0,self.band.force[2])/self.band.weight),
                'ground_load_ratio':float(feet.sum()/self.band.weight),
                'joint_velocity_max':float(np.max(abs(d.qvel[self.c.vadr]))),
                'torque_max':float(np.max(abs(d.actuator_force[self.c.actuators])))}

    def check(self):
        x=self.measure()
        if x['root_height']<self.height_min:raise SafetyAbort('root below HOME pelvis-to-knee height bound')
        if max(x['tilt'],abs(x['roll']),abs(x['pitch']))>self.tilt_limit:raise SafetyAbort('training orientation termination (70 degrees)')
        if x['nonfoot_contacts']:raise SafetyAbort('non-foot robot body contacting ground')
        self.original_check()

    def physics_step(self):
        dt=self.m.opt.timestep
        # No stiffness/damping/anchor changes, and no discrete force release.
        delta=np.clip(self.goal_length-self.band.length,-self.length_rate*dt,self.length_rate*dt)
        self.band.length+=float(delta)
        try:self.original_step()
        finally:self.samples.append(self.measure())

    def stats(self,window=None):
        samples=self.samples if window is None else window
        if not samples:return {}
        a=lambda k:np.asarray([s[k] for s in samples])
        contacts=a('foot_contact');support=a('vertical_support_ratio');ground=a('ground_load_ratio')
        missing=0;longest=0
        for bilateral in contacts.all(axis=1):
            missing=0 if bilateral else missing+1;longest=max(longest,missing)
        return {'support_mean':float(support.mean()),'support_min':float(support.min()),'support_max':float(support.max()),
                'ground_load_mean':float(ground.mean()),'support_plus_ground_mean':float((support+ground).mean()),
                'left_foot_fz_mean':float(a('foot_fz')[:,0].mean()),'right_foot_fz_mean':float(a('foot_fz')[:,1].mean()),
                'bilateral_contact_ratio':float(contacts.all(axis=1).mean()),
                'left_contact_ratio':float(contacts[:,0].mean()),'right_contact_ratio':float(contacts[:,1].mean()),
                'longest_nonbilateral_seconds':longest*self.m.opt.timestep,
                'max_nonfoot_contact_count':int(a('nonfoot_contacts').max()),
                'max_root_height_drop':float(self.root0-a('root_height').min()),
                'min_root_height':float(a('root_height').min()),'max_abs_roll':float(abs(a('roll')).max()),
                'max_abs_pitch':float(abs(a('pitch')).max()),'root_speed_max':float(np.linalg.norm(a('base_linear_velocity'),axis=1).max()),
                'max_joint_velocity':float(a('joint_velocity_max').max()),'max_torque':float(a('torque_max').max())}
