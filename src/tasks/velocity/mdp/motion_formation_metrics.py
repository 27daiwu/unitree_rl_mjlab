"""Read-only motion formation diagnostics from saved evaluation traces.

Action units are normalized policy outputs, not joint radians. Positions are
world coordinates; root path and net displacement are deliberately distinct.
"""
import numpy as np


def motion_formation_diagnostics(trace, completed):
  t = np.asarray(trace['time'])

  def window(mask):
    times = t[mask]
    p = np.asarray(trace['position'])[mask]
    rpy = np.asarray(trace['rpy'])[mask]
    a = np.asarray(trace['action'])[mask]
    contact = np.asarray(trace['contact_time'])[mask] > 0
    foot_velocity = np.asarray(trace['foot_velocity'])[mask]
    yaw = np.unwrap(rpy[:, 2])
    states = contact[:, 0].astype(int) + 2*contact[:, 1].astype(int)
    starts = np.r_[0, np.flatnonzero(np.diff(states)) + 1]
    ends = np.r_[starts[1:], len(states)]
    sequence = [dict(start_s=float(times[s]), end_s=float(times[e-1]),
                     left=bool(contact[s, 0]), right=bool(contact[s, 1]), samples=int(e-s))
                for s, e in zip(starts, ends)]
    out = dict(samples=len(times), start_s=float(times[0]), end_s=float(times[-1]),
      root_displacement_xyz=(p[-1]-p[0]).tolist(),
      root_horizontal_path=float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()),
      accumulated_yaw=float(yaw[-1]-yaw[0]),
      absolute_yaw_path=float(np.abs(np.diff(yaw)).sum()),
      action_rms=float(np.sqrt(np.mean(a*a))),
      action_temporal_variance=float(np.var(a, axis=0).mean()),
      action_rms_per_dimension=np.sqrt(np.mean(a*a, axis=0)).tolist(),
      action_temporal_variance_per_dimension=np.var(a, axis=0).tolist(),
      single_support_fraction=float((contact.sum(1) == 1).mean()),
      double_support_fraction=float((contact.sum(1) == 2).mean()),
      no_foot_contact_fraction=float((contact.sum(1) == 0).mean()),
      foot_contact_transitions=np.count_nonzero(np.diff(contact.astype(int), axis=0), axis=0).tolist(),
      contact_sequence=sequence,
      contact_foot_speed_mean=float(np.linalg.norm(foot_velocity, axis=-1)[contact].mean()) if contact.any() else None,
      foot_horizontal_displacement=None, foot_horizontal_path=None)
    if 'foot_position' in trace:
      feet = np.asarray(trace['foot_position'])[mask, :, :2]
      out['foot_horizontal_displacement'] = (feet[-1]-feet[0]).tolist()
      out['foot_horizontal_path'] = np.linalg.norm(np.diff(feet, axis=0), axis=-1).sum(0).tolist()
    return out

  return dict(observed=window(np.ones(len(t), dtype=bool)),
              stable=window(t > 7.+1e-6) if completed else None)
