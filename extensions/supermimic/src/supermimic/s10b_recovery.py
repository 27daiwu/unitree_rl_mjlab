"""Contact-aware model-based recovery. No motion reference or Policy calls."""

from __future__ import annotations

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation


def support(runtime):
    m, d = runtime.model, runtime.diag
    points, forces = [[], []], [0.0, 0.0]
    for i in range(d.ncon):
        c = d.contact[i]
        bodies = m.geom_bodyid[[c.geom1, c.geom2]]
        if not any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in (c.geom1, c.geom2)):
            continue
        for side, foot in enumerate(runtime.feet):
            if foot in bodies and c.dist < 0.003:
                points[side].append(c.pos.copy())
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                forces[side] += max(0.0, f[0])
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, d, jac, runtime.pelvis)
    com, vel = d.subtree_com[runtime.pelvis].copy(), jac @ d.qvel
    capture = com[:2] + vel[:2] / np.sqrt(9.81 / max(com[2], 0.2))
    pts = np.asarray(points[0] + points[1])
    margin = -1.0
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts[:, :2])
            margin = float(-np.max(hull.equations[:, :2] @ capture + hull.equations[:, 2]))
        except QhullError:
            pass  # Collinear contacts do not establish a support polygon.
    return {
        "contacts": [len(p) for p in points],
        "normal_N": forces,
        "points": points,
        "com": com,
        "com_velocity": vel,
        "capture": capture,
        "capture_margin": margin,
        "com_jac": jac,
    }


class Recovery:
    def __init__(self, runtime):
        self.r = runtime
        self.started = None
        self.rows = []
        self.previous_com_jac = None
        self.last_solution = None

    def ready(self):
        s = support(self.r)
        d = self.r.diag
        tilt = np.arccos(np.clip(d.xmat[self.r.pelvis].reshape(3, 3)[2, 2], -1, 1))
        double_support = (
            min(s["contacts"]) >= 2
            and s["capture_margin"] > 0.015
            and np.linalg.norm(s["com_velocity"]) < 0.15
            and tilt < 0.20
            and np.max(abs(d.qvel[6:])) < 0.5
        )
        brakeable_support = (
            max(s["contacts"]) >= 3
            and s["capture_margin"] > -0.02
            and np.linalg.norm(s["com_velocity"]) < 0.2
            and tilt < 0.25
        )
        return double_support or brakeable_support

    def start(self):
        d = self.r.diag
        self.started = float(self.r.data.time)
        self.start_q = d.qpos[7:].copy()
        self.yaw = Rotation.from_matrix(d.xmat[self.r.pelvis].reshape(3, 3)).as_euler("xyz")[2]
        self.heading = Rotation.from_euler("z", self.yaw)
        self.yaw_speed = float((self.heading.apply(d.qvel[3:6]))[2])
        self.anchors = d.xpos[self.r.feet].copy()
        self.rotations = [
            Rotation.from_euler(
                "z", Rotation.from_matrix(d.xmat[f].reshape(3, 3)).as_euler("xyz")[2]
            )
            for f in self.r.feet
        ]
        s = support(self.r)
        self.com_height = float(s["com"][2])
        self.contact = np.array(s["contacts"]) > 0
        for i in range(2):
            if not self.contact[i]:
                self.anchors[i, :2] = (
                    s["capture"] + self.heading.apply([0.0, 0.10 if i == 0 else -0.10, 0.0])[:2]
                )
                self.anchors[i, 2] = 0.035

    def torque(self, previous, rate):
        r, m, d = self.r, self.r.model, self.r.diag
        if self.started is None:
            self.start()
        dt = m.opt.timestep
        elapsed = r.data.time - self.started
        s = support(r)
        contact = np.asarray(s["contacts"]) > 0
        for i in range(2):
            if contact[i] and not self.contact[i]:
                self.anchors[i] = d.xpos[r.feet[i]]
                self.anchors[i, 2] = 0.035
        self.contact = contact
        active = np.flatnonzero(contact)
        jf = []
        normal_tasks, normal_goals = [], []
        for side in active:
            foot = r.feet[side]
            points = np.asarray(s["points"][side])
            ids = sorted(
                {
                    int(points[:, 0].argmin()),
                    int(points[:, 0].argmax()),
                    int(points[:, 1].argmin()),
                    int(points[:, 1].argmax()),
                }
            )
            for point in points[ids]:
                jp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, None, point, foot)
                jf.append(jp)
            point = points.mean(axis=0)
            jp, dp = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, point, foot)
            mujoco.mj_jacDot(m, d, dp, None, point, foot)
            normal_tasks.append(jp[2:])
            normal_goals.append(-dp[2:] @ d.qvel - 20 * (jp[2:] @ d.qvel))
        jf = np.concatenate(jf, axis=0) if jf else np.zeros((0, m.nv))
        nf, nv = len(jf), m.nv
        n = nv + nf
        mass = np.zeros((nv, nv))
        mujoco.mj_fullM(m, d, mass)
        dynamics = np.c_[mass, -jf.T]
        h = d.qfrc_bias.copy()
        tasks, goals = [], []

        def task(jac, desired, weight):
            tasks.append(np.c_[jac, np.zeros((len(jac), nf))] * weight)
            goals.append(np.asarray(desired) * weight)

        # Posture is a soft objective. Contact and floating-base tasks dominate.
        blend = min(1.0, elapsed / 2.0)
        qgoal = (1 - blend) * self.start_q + blend * r.q0
        task(np.eye(nv)[6:], 25 * (qgoal - d.qpos[7:]) - 12 * d.qvel[6:], 3.0)
        center = self.anchors[active, :2].mean(axis=0) if len(active) else s["com"][:2]
        center += self.heading.apply([0.025, 0.0, 0.0])[:2]
        height = self.com_height if len(active) < 2 else 0.65
        acc = np.r_[
            np.clip(35 * (center - s["com"][:2]) - 12 * s["com_velocity"][:2], -6, 6),
            np.clip(80 * (height - s["com"][2]) - 18 * s["com_velocity"][2], -8, 8),
        ]
        jdotv = (
            np.zeros(3)
            if self.previous_com_jac is None
            else ((s["com_jac"] - self.previous_com_jac) / dt) @ d.qvel
        )
        self.previous_com_jac = s["com_jac"].copy()
        task(s["com_jac"][:2], (acc - jdotv)[:2], 100.0)
        task(s["com_jac"][2:], (acc - jdotv)[2:], 120.0)
        jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBody(m, d, jp, jr, r.pelvis)
        orientation = Rotation.from_matrix(d.xmat[r.pelvis].reshape(3, 3))
        yaw_goal = self.yaw + 0.5 * self.yaw_speed * (1 - np.exp(-elapsed / 0.5))
        yaw_velocity = self.yaw_speed * np.exp(-elapsed / 0.5)
        rot_error = (Rotation.from_euler("z", yaw_goal) * orientation.inv()).as_rotvec()
        angular = 60 * rot_error - 14 * (jr @ d.qvel)
        angular[2] = 8 * rot_error[2] + 3 * (yaw_velocity - (jr @ d.qvel)[2])
        task(jr[:2], angular[:2], 40.0)
        task(jr[2:], angular[2:], 1.0)
        for side, foot in enumerate(r.feet):
            jp, jr, dp, dr = [np.zeros((3, nv)) for _ in range(4)]
            mujoco.mj_jacBody(m, d, jp, jr, foot)
            mujoco.mj_jacDot(m, d, dp, dr, d.xpos[foot], foot)
            pos_error = self.anchors[side] - d.xpos[foot]
            foot_rot = Rotation.from_matrix(d.xmat[foot].reshape(3, 3))
            error = (self.rotations[side] * foot_rot.inv()).as_rotvec()
            task(
                jp,
                180 * pos_error - 24 * (jp @ d.qvel) - dp @ d.qvel,
                100.0 if contact[side] else 160.0,
            )
            rotational_acc = 100 * error - 20 * (jr @ d.qvel) - dr @ d.qvel
            task(jr[:2], rotational_acc[:2], 100.0)
            task(jr[2:], rotational_acc[2:], 1.0)
        a, b = np.concatenate(tasks), np.concatenate(goals)
        p = a.T @ a + np.diag(np.r_[np.full(nv, 0.01), np.full(nf, 0.0001)])
        q = -a.T @ b
        constraints, lower, upper = [dynamics[:6]], [-h[:6]], [-h[:6]]
        if normal_tasks:
            constraints.append(
                np.c_[np.concatenate(normal_tasks), np.zeros((len(normal_tasks), nf))]
            )
            lower.append(np.concatenate(normal_goals))
            upper.append(np.concatenate(normal_goals))
        constraints.append(np.c_[np.eye(nv)[6:], np.zeros((nv - 6, nf))])
        lower.append(np.r_[np.full(12, -300.0), np.full(nv - 18, -80.0)])
        upper.append(np.r_[np.full(12, 300.0), np.full(nv - 18, 80.0)])
        constraints.append(dynamics[6:])
        lower.append(np.maximum(-r.effort, previous - rate * dt) - h[6:])
        upper.append(np.minimum(r.effort, previous + rate * dt) - h[6:])
        for k in range(nf // 3):
            friction = np.zeros((5, n))
            i = nv + 3 * k
            friction[0, i + 2] = 1.0
            for axis in range(2):
                friction[1 + 2 * axis, i + axis] = 1.0
                friction[2 + 2 * axis, i + axis] = -1.0
                friction[1 + 2 * axis : 3 + 2 * axis, i + 2] = -0.6
            constraints.append(friction)
            lower.append(np.array([0.0, -np.inf, -np.inf, -np.inf, -np.inf]))
            upper.append(np.array([800.0, 0.0, 0.0, 0.0, 0.0]))
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(np.triu(p)),
            q=q,
            A=sparse.csc_matrix(np.concatenate(constraints)),
            l=np.concatenate(lower),
            u=np.concatenate(upper),
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=4000,
            polishing=False,
        )
        if self.last_solution is not None and len(self.last_solution) == n:
            solver.warm_start(x=self.last_solution)
        solution = solver.solve(raise_error=False)
        ok = solution.info.status_val in (1, 2) and np.isfinite(solution.x).all()
        if ok:
            self.last_solution = solution.x
            torque = dynamics[6:] @ solution.x + h[6:]
        else:
            self.last_solution = None
            torque = r.kp * (r.stand(d.qpos[7:]) - d.qpos[7:]) - 3 * r.kd * d.qvel[6:]
        self.rows.append(
            (
                float(r.data.time),
                elapsed,
                s["contacts"],
                s["normal_N"],
                s["com"],
                s["com_velocity"],
                s["capture_margin"],
                d.xpos[r.feet].copy(),
                d.xmat[r.feet].copy(),
                solution.info.status,
                ok,
            )
        )
        return torque

    def save(self, out):
        if self.rows:
            keys = [
                "time",
                "elapsed",
                "foot_contacts",
                "normal_N",
                "com",
                "com_velocity",
                "capture_margin",
                "foot_position",
                "foot_rotation",
                "qp_status",
                "qp_ok",
            ]
            np.savez_compressed(
                out / "recovery.npz",
                **{key: np.asarray([row[i] for row in self.rows]) for i, key in enumerate(keys)},
            )
