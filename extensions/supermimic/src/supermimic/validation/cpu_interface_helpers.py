"""Independent CPU-only numerical helpers for policy interface validation.

No simulation integration occurs here. MuJoCo model arrays are read as data;
SciPy rotation operations independently check the engine's forward kinematics.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


TERM_SIZES = (29, 29, 3, 3, 29)


def _rotation_wxyz(value):
    q = np.asarray(value, dtype=np.float64)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def scipy_fk(model, qpos):
    """Return every MuJoCo body's world position and matrix without mj_forward.

    Fixed, hinge, slide and free joints are supported. Ball joints deliberately
    reject rather than silently use a partial convention. Hinge positions are
    measured relative to model.qpos0, exactly as MuJoCo's kinematics are.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("qpos must be finite and have model.nq entries")
    pos = np.zeros((model.nbody, 3), dtype=np.float64)
    mats = np.repeat(np.eye(3)[None], model.nbody, axis=0)
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        p = pos[parent] + mats[parent] @ model.body_pos[body]
        r = mats[parent] @ _rotation_wxyz(model.body_quat[body])
        for joint in range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body])):
            kind = int(model.jnt_type[joint])
            address = int(model.jnt_qposadr[joint])
            if kind == 0:  # free: global pose, overriding fixed body offset
                p = qpos[address:address + 3].copy()
                r = _rotation_wxyz(qpos[address + 3:address + 7])
            elif kind == 3:  # hinge
                pivot = np.asarray(model.jnt_pos[joint], dtype=np.float64)
                anchor = p + r @ pivot
                angle = qpos[address] - model.qpos0[address]
                r = r @ Rotation.from_rotvec(model.jnt_axis[joint] * angle).as_matrix()
                p = anchor - r @ pivot
            elif kind == 2:  # slide
                p = p + r @ model.jnt_axis[joint] * (qpos[address] - model.qpos0[address])
            else:
                raise ValueError(f"Unsupported joint type {kind} at joint {joint}")
        pos[body], mats[body] = p, r
    return pos, mats


def history_pack(history, terms):
    """Update ten-frame history and flatten each term oldest-to-newest.

    The first observation fills all history slots, matching the manager buffer's
    first-append initialization. The returned history owns its arrays.
    """
    terms = tuple(np.asarray(term).reshape(-1) for term in terms)
    if tuple(term.size for term in terms) != TERM_SIZES:
        raise ValueError("Observation term sizes must be (29,29,3,3,29)")
    frame = np.concatenate(terms)
    if not np.isfinite(frame).all():
        raise ValueError("Observation terms must be finite")
    if history is None or len(history) == 0:
        updated = [frame.copy() for _ in range(10)]
    else:
        old = np.asarray(history)
        if old.shape != (10, 93):
            raise ValueError("history must be (10,93)")
        updated = [row.copy() for row in old[1:]] + [frame.copy()]
    stack = np.asarray(updated)
    boundaries = np.cumsum((0,) + TERM_SIZES)
    flat = np.concatenate([stack[:, start:end].reshape(-1) for start, end in zip(boundaries[:-1], boundaries[1:])])
    return updated, flat


def goal_scipy(ref, q0):
    """Independent 68-feature goal construction from canonical N x 71 states."""
    ref = np.asarray(ref, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 71 or q0.shape != (29,):
        raise ValueError("Expected reference (N,71), default joint positions (29,)")
    if not np.isfinite(ref).all() or not np.isfinite(q0).all():
        raise ValueError("Goal inputs must be finite")
    roots = Rotation.from_quat(ref[:, [4, 5, 6, 3]])
    gravity = roots.inv().apply(np.broadcast_to([0., 0., -1.], (len(ref), 3)))
    angular = roots.inv().apply(ref[:, 39:42])
    # Extract yaw from the quaternion directly; avoid Euler gimbal conventions.
    w, x, y, z = ref[:, 3:7].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    heading = Rotation.from_rotvec(np.stack([yaw * 0, yaw * 0, yaw], axis=-1))
    linear = heading.inv().apply(ref[:, 36:39])
    return np.concatenate([ref[:, 7:36] - q0, ref[:, 42:71], gravity, angular, linear, ref[:, 2:3]], axis=-1)
