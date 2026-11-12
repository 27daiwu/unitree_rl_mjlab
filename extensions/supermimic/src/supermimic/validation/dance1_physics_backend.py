"""R9 runtime-only backend. Importing this module loads no Policy or simulator.

R9A may compile this source but must not instantiate MujocoBackend. The model is
never edited: explicit PD is evaluated by the independent bounded worker and
its limited joint torque is sent to the frozen official unit-gear torque motors.
"""
from pathlib import Path
import json
import re


class MujocoBackend:
    def __init__(self, out):
        # Every heavyweight import, checkpoint load and forward solve is deferred
        # to explicit R9B construction. No R9A test calls this constructor.
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import numpy as np
        import torch
        import mujoco
        from scipy.spatial.transform import Rotation
        from supermimic.validation.dance1_cpu import EXT, ROOT, OFFICIAL, sha, config_audit
        from supermimic.validation.cpu_interface_helpers import history_pack, scipy_fk
        from supermimic.policy.nets import Policy, PolicyCfg
        from supermimic.envs.mdp.motion_library import RefState, goal_features
        from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as constants
        from mjlab.utils.lab_api.math import subtract_frame_transforms, matrix_from_quat
        self.np, self.torch, self.mj, self.R = np, torch, mujoco, Rotation
        self.history_pack, self.scipy_fk = history_pack, scipy_fk
        self.subtract, self.matrix = subtract_frame_transforms, matrix_from_quat
        self.out = Path(out)
        r8 = EXT / 'reports/phase2_s9b5r8_dance1_cpu_first_execution_20261110'
        manifest = json.loads((r8 / 'asset_hashes.json').read_text())
        self.asset_hashes = {p: sha(p) for p in manifest}
        if self.asset_hashes != manifest:
            raise ValueError('R8 frozen asset hash mismatch')
        self.initial_artifact_sha256 = sha(r8 / 'interface_values.npz')
        ckpath = EXT / 'logs/phase2_s5b_dance1_subject1_anchor_2h_20261105/ckpt_final_0038100.pt'
        refpath = EXT / 'data/npz/lafan1/dance1_subject1.npz'
        cfg = json.loads((ckpath.parent / 'config.json').read_text())
        self.counts = {'checkpoint_loads': 0, 'strict_bindings': 0, 'policy_inferences': 0,
                       'mj_forward': 0, 'physics_steps': 0, 'torque_writes': 0}
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        checkpoint = torch.load(ckpath, map_location='cpu', weights_only=False)
        self.counts['checkpoint_loads'] += 1
        audit = config_audit(checkpoint['config'], cfg, refpath)
        (self.out / 'config_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        if not audit['passed']:
            raise ValueError('Live critical config mismatch')
        if cfg['policy']['history'] != 10 or cfg['policy']['goal_dim'] != 689:
            raise ValueError('Frozen policy dimensions changed')
        pcfg = PolicyCfg(**{k: tuple(v) if isinstance(v, list) else v
                            for k, v in cfg['policy'].items()})
        self.policy = Policy(pcfg).cpu().eval()
        self.policy.load_state_dict(checkpoint['policy'], strict=True)
        self.counts['strict_bindings'] += 1
        del checkpoint
        self.model = mujoco.MjModel.from_xml_path(str(OFFICIAL))
        self.data = mujoco.MjData(self.model)
        self.diagnostic = mujoco.MjData(self.model)
        m = self.model
        if not (m.nq == 36 and m.nv == 35 and m.nu == 29 and m.opt.timestep == .002):
            raise ValueError('Official model dimensions/timestep mismatch')
        with np.load(refpath) as z:
            self.names = z['joint_names'].astype(str).tolist()
            bodies = z['body_names'].astype(str).tolist()
            root, anchor = bodies.index('pelvis'), bodies.index('torso_link')
            if float(z['fps'][0]) != 50.:
                raise ValueError('Frozen 50Hz reference required; no online resampling')
            self.ref = np.concatenate([z['body_pos_w'][:, root], z['body_quat_w'][:, root],
                z['joint_pos'], z['body_lin_vel_w'][:, root], z['body_ang_vel_w'][:, root],
                z['joint_vel']], axis=1).astype(np.float32)
            self.apos = z['body_pos_w'][:, anchor].copy()
            self.aquat = z['body_quat_w'][:, anchor].copy()
        if self.names != [m.joint(i).name for i in range(1, m.njnt)]:
            raise ValueError('Reference/official joint order mismatch')
        self.jids = np.array([m.joint(n).id for n in self.names], dtype=int)
        self.qadr = np.array([m.joint(n).qposadr[0] for n in self.names], dtype=int)
        self.vadr = np.array([m.joint(n).dofadr[0] for n in self.names], dtype=int)
        self.aids = []
        for jid in self.jids:
            hits = np.flatnonzero((m.actuator_trntype == 0) & (m.actuator_trnid[:, 0] == jid))
            if len(hits) != 1:
                raise ValueError('Each joint must have exactly one joint torque actuator')
            aid = int(hits[0]); self.aids.append(aid)
            if not (m.actuator_gaintype[aid] == 0 and m.actuator_biastype[aid] == 0
                    and m.actuator_dyntype[aid] == 0 and m.actuator_gainprm[aid, 0] == 1
                    and np.array_equal(m.actuator_gear[aid], [1, 0, 0, 0, 0, 0])):
                raise ValueError('Frozen official direct unit-gear torque motor required')
        self.aids = np.array(self.aids, dtype=int)
        if len(set(self.aids)) != 29:
            raise ValueError('Actuator mapping is not bijective')
        acts = []
        for name in self.names:
            hits = [a for a in constants.G1_ARTICULATION.actuators
                    if any(re.fullmatch(p, name) for p in a.target_names_expr)]
            if len(hits) != 1:
                raise ValueError('Ambiguous/missing frozen training actuator config')
            acts.append(hits[0])
        self.kp = np.array([a.stiffness for a in acts], dtype=float)
        self.kd = np.array([a.damping for a in acts], dtype=float)
        self.effort = np.array([a.effort_limit for a in acts], dtype=float)
        self.lower, self.upper = -self.effort.copy(), self.effort.copy()
        for k, (jid, aid) in enumerate(zip(self.jids, self.aids)):
            ranges = []
            if m.actuator_ctrllimited[aid]: ranges.append(m.actuator_ctrlrange[aid])
            if m.jnt_actfrclimited[jid]: ranges.append(m.jnt_actfrcrange[jid])
            if m.actuator_forcelimited[aid]: ranges.append(m.actuator_forcerange[aid])
            for lo, hi in ranges:
                self.lower[k] = max(self.lower[k], lo)
                self.upper[k] = min(self.upper[k], hi)
        if not np.all(self.lower < self.upper):
            raise ValueError('Invalid intersection of frozen torque limits')
        with np.load(r8 / 'interface_values.npz') as z:
            self.data.qpos[:] = z['qpos']; self.data.qvel[:] = z['qvel']
            self.q0 = z['q0'].copy(); self.scale = z['scale'].copy(); self.offset = z['offset'].copy()
        if not np.allclose(self.scale, .25 * self.effort / self.kp, atol=1e-4, rtol=0):
            raise ValueError('Frozen action scale does not implement training contract')
        if not np.array_equal(self.offset, self.q0):
            raise ValueError('Frozen action offset mismatch')
        split = RefState(*(torch.from_numpy(self.ref[:, a:b]) for a, b in
                           [(0,3),(3,7),(7,36),(36,39),(39,42),(42,71)]))
        self.features = goal_features(split, torch.from_numpy(self.q0)).numpy()
        self.history = None; self.previous = np.zeros(29, np.float32); self.frame = 1
        self.body = m.body('torso_link').id
        self.no_floor = not any(int(m.geom_type[g]) == 0 and int(m.geom_bodyid[g]) == 0
                                for g in range(m.ngeom))
        mujoco.mj_forward(m, self.data); self.counts['mj_forward'] += 1
        if torch.cuda.is_initialized():
            raise RuntimeError('GPU initialized unexpectedly')

    def state(self):
        d = self.data
        return {'q': d.qpos[self.qadr].copy(), 'qd': d.qvel[self.vadr].copy(),
                'qpos': d.qpos.copy(), 'qvel': d.qvel.copy(), 'time': float(d.time),
                'root_pos': d.qpos[:3].copy(), 'root_quat': d.qpos[3:7].copy()}

    def infer(self, cycle):
        np, torch, d = self.np, self.torch, self.data
        self.frame = cycle + 1
        root = self.R.from_quat(d.qpos[[4, 5, 6, 3]])
        terms = [d.qpos[self.qadr].astype(np.float32)-self.q0,
                 d.qvel[self.vadr].astype(np.float32), d.qvel[3:6].astype(np.float32),
                 root.inv().apply([0., 0., -1.]).astype(np.float32), self.previous.copy()]
        self.history, obs = self.history_pack(self.history, terms)
        tensor = lambda x: torch.tensor(np.asarray(x).copy(), dtype=torch.float32).reshape(1,-1)
        anchor_pos, anchor_quat = self.current_anchor_pose()
        rp, rq = self.subtract(tensor(anchor_pos), tensor(anchor_quat),
                               tensor(self.apos[self.frame]), tensor(self.aquat[self.frame]))
        anchor = np.r_[rp.numpy()[0], self.matrix(rq).numpy()[0,:,:2].ravel()]
        idx = np.minimum(self.frame+np.arange(10), len(self.ref)-1)
        goal = np.r_[self.features[idx].ravel(), anchor].astype(np.float32)
        obs = np.asarray(obs, np.float32)
        if obs.shape != (930,) or goal.shape != (689,) or not all(np.isfinite(x).all() for x in (obs, goal)):
            raise ValueError('Nonfinite/wrong shape policy interface')
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(obs[None]), torch.from_numpy(goal[None])).numpy()[0]
        self.counts['policy_inferences'] += 1
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError('Invalid Policy action')
        self.previous = action.copy()
        self.last_interface = {'obs': obs, 'goal': goal, 'action': action.copy()}
        return action

    def current_anchor_pose(self):
        """Refresh FK from current state without overwriting main-step forces."""
        self.diagnostic.qpos[:] = self.data.qpos
        self.diagnostic.qvel[:] = self.data.qvel
        self.mj.mj_forward(self.model, self.diagnostic)
        self.counts['mj_forward'] += 1
        return (self.diagnostic.xpos[self.body].copy(),
                self.diagnostic.xquat[self.body].copy())

    def update_ctrl(self, applied):
        self.data.ctrl[self.aids] = applied
        self.counts['torque_writes'] += 1

    def step(self):
        self.mj.mj_step(self.model, self.data)
        self.counts['physics_steps'] += 1

    def inspect(self):
        np, m, d = self.np, self.model, self.data
        contacts = []
        for i in range(d.ncon):
            c = d.contact[i]; force = np.zeros(6)
            self.mj.mj_contactForce(m, d, i, force)
            contacts.append({'dim': int(c.dim), 'dist': float(c.dist),
                             'normal_force': float(force[0]), 'geom1': int(c.geom1), 'geom2': int(c.geom2)})
        limited = m.jnt_limited[self.jids].astype(bool)
        ranges = m.jnt_range[self.jids]
        q = d.qpos[self.qadr]
        violation = np.maximum(ranges[:,0]-q, q-ranges[:,1])
        # Separate forward solve: keep actual main-step forces and cached pose intact.
        self.diagnostic.qpos[:] = d.qpos; self.diagnostic.qvel[:] = d.qvel
        self.mj.mj_forward(m, self.diagnostic); self.counts['mj_forward'] += 1
        ref = self.ref[self.frame]
        actual_gz = self.R.from_quat(d.qpos[[4,5,6,3]]).inv().apply([0.,0.,-1.])[2]
        ref_gz = self.R.from_quat(ref[[4,5,6,3]]).inv().apply([0.,0.,-1.])[2]
        return {'force': d.qfrc_actuator[self.vadr].copy(),
                'actuator_force': d.actuator_force[self.aids].copy(), 'ctrl': d.ctrl[self.aids].copy(),
                'contacts': contacts, 'warnings': int(sum(w.number for w in d.warning)),
                'joint_violation': float(max(0., np.max(violation[limited], initial=0.))),
                'root_height_error': float(abs(d.qpos[2]-ref[2])),
                'root_tilt_error': float(abs(actual_gz-ref_gz)),
                'clip_end': bool(self.frame >= len(self.ref)-1), 'no_floor': self.no_floor,
                'termination_evaluation': 'initial_and_each_physics_substep_safety; strict > source thresholds; adapted_control_not_training_equivalence'}
