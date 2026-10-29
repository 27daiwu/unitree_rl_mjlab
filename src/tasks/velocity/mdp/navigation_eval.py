"""Policy-independent navigation benchmark and Phase B disturbance contract.

Trace samples are interval endpoints in seconds, metres and radians. Commands
are body-frame vx/vy/wz; yaw/position are world-frame, relative to trial start.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RecoveryTolerance:
  velocity: tuple[float, float, float] = (0.03, 0.03, 0.08)
  attitude: float = 0.15
  hold_s: float = 0.5


@dataclass(frozen=True)
class DisturbanceWindow:
  kind: str
  start_s: float
  end_s: float

  def __post_init__(self):
    if not 0 <= self.start_s <= self.end_s:
      raise ValueError("Disturbance requires 0 <= start <= end")


class DisturbanceInjector(Protocol):
  """Phase B plugin: configure cfg before env creation, then inject before step.

  Implementations expose windows for scoring. Phase A supplies no injector.
  Supports push/friction/mass/COM/PD/motor/latency/noise/small terrain separately.
  """
  windows: tuple[DisturbanceWindow, ...]

  def configure(self, env_cfg): ...
  def before_step(self, env, time_s: float): ...


# Each trial starts from rest, preserving continuous state across transitions.
SCENARIOS = {
  "low_005": ((2., (0., 0., 0.)), (4., (.05, 0., 0.)), (3., (0., 0., 0.))),
  "low_010": ((2., (0., 0., 0.)), (4., (.10, 0., 0.)), (3., (0., 0., 0.))),
  "low_020": ((2., (0., 0., 0.)), (4., (.20, 0., 0.)), (3., (0., 0., 0.))),
  "forward_backward": ((2., (0., 0., 0.)), (3., (.3, 0., 0.)), (3., (-.15, 0., 0.)), (3., (0., 0., 0.))),
  "left_right": ((2., (0., 0., 0.)), (3., (0., .1, 0.)), (3., (0., -.1, 0.)), (3., (0., 0., 0.))),
  "turn_stop": ((2., (0., 0., 0.)), (3., (0., 0., .4)), (3., (0., 0., 0.)), (3., (0., 0., -.4)), (3., (0., 0., 0.))),
  "simultaneous": ((2., (0., 0., 0.)), (3., (.15, .1, .3)), (3., (.05, -.1, -.3)), (3., (0., 0., 0.))),
  "zero": ((8., (0., 0., 0.)),),
}


def recovery_time(time, error, attitude, start_s, tolerance=RecoveryTolerance()):
  """First entry after start followed by hold_s continuously inside tolerance.

  Returns entry delay, not confirmation delay; None means right-censored.
  """
  valid = (np.abs(error) <= tolerance.velocity).all(axis=1) & (np.abs(attitude) <= tolerance.attitude).all(axis=1)
  entry = None
  for i in np.flatnonzero(time >= start_s - 1e-9):
    if not valid[i]:
      entry = None
    elif entry is None:
      entry = i
    if entry is not None and time[i] - time[entry] >= tolerance.hold_s - 1e-9:
      return float(max(0., time[entry] - start_s))
  return None


def summarize_trace(trace, fallen=False, disturbances=(), tolerance=RecoveryTolerance()):
  t, cmd, vel, pos, rpy = (np.asarray(trace[k]) for k in ("time", "command", "velocity", "position", "rpy"))
  if len(t) < 2 or not np.all(np.diff(t) > 0):
    raise ValueError("Trace needs at least two strictly increasing samples")
  error = vel - cmd
  rms = lambda a: np.sqrt(np.mean(np.square(a), axis=0)).tolist() if len(a) else None
  yaw = np.unwrap(rpy[:, 2])
  dt = np.diff(t, prepend=0.)
  target_yaw = np.cumsum(cmd[:, 2] * dt)
  heading_error = yaw - trace["initial_yaw"] - target_yaw
  low = (np.abs(cmd) <= [.2, .15, .4]).all(1) & (np.abs(cmd).max(1) > 0)
  result = dict(velocity_rmse=rms(error), low_speed_tracking_error=rms(error[low]),
                roll_rms=rms(rpy[:, 0]), pitch_rms=rms(rpy[:, 1]),
                body_height_variance=float(np.var(pos[:, 2])), fall_rate=float(fallen),
                heading_deviation=float(np.max(np.abs(heading_error))), samples=len(t))
  starts = np.r_[0, np.flatnonzero(np.any(np.diff(cmd, axis=0) != 0, axis=1)) + 1]
  transitions = []
  for s, e in zip(starts, np.r_[starts[1:], len(t)]):
    start = float(t[s] - dt[s])
    rec = recovery_time(t[s:e], error[s:e], rpy[s:e, :2], start, tolerance)
    before = np.asarray(trace["initial_position"]) if s == 0 else pos[s-1]
    before_yaw = trace["initial_yaw"] if s == 0 else yaw[s-1]
    desired = cmd[s]
    direction = np.sign(desired - (cmd[s-1] if s else np.zeros(3)))
    overshoot = np.maximum(error[s:e] * direction, 0).max(0)
    row = dict(start_s=start, command=desired.tolist(), recovery_time=rec,
               recovery_censored=rec is None, velocity_rmse=rms(error[s:e]),
               velocity_overshoot=overshoot.tolist(),
               yaw_drift=float(yaw[e-1] - before_yaw) if desired[2] == 0 else None,
               turning_error=float(yaw[e-1] - before_yaw - np.sum(cmd[s:e, 2] * dt[s:e])),
               stopping_distance=None, post_stop_drift=None)
    if np.all(desired == 0):
      # Path length through first stable entry; null if stopping was not confirmed.
      if rec is not None:
        stop = min(e-1, s + int(np.searchsorted(t[s:e], start + rec)))
        points = np.vstack((before, pos[s:stop+1]))
        row["stopping_distance"] = float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())
        row["post_stop_drift"] = float(np.linalg.norm(pos[stop:e, :2] - pos[stop, :2], axis=1).max())
    transitions.append(row)
  result.update(zip(("vx_rmse", "vy_rmse", "wz_rmse"), result["velocity_rmse"]))
  result["segments"] = transitions
  robustness = []
  for window in disturbances:
    # Refuse to label command changes as disturbance recovery.
    mask = t >= window.end_s - 1e-9
    if not mask.any():
      robustness.append(dict(kind=window.kind, observed=False, recovery_time=None, recovery_censored=True, fall_rate=float(fallen)))
      continue
    indices = np.flatnonzero(mask)
    begin = indices[0]
    changed = np.flatnonzero(np.any(cmd[begin:] != cmd[begin], axis=1))
    end = begin + changed[0] if len(changed) else len(t)
    sl = slice(begin, end)
    during = (t >= window.start_s) & (t <= t[end-1])
    # Integrate commanded body velocity in actual heading: lateral residual,
    # expressed in heading at disturbance end, not raw world-y displacement.
    c, s = np.cos(yaw[sl]), np.sin(yaw[sl])
    target_world = np.column_stack((c * cmd[sl, 0] - s * cmd[sl, 1], s * cmd[sl, 0] + c * cmd[sl, 1]))
    # Anchor to the first observed state at/after disturbance end. Do not
    # integrate the interval preceding end_s into post-disturbance drift.
    post_dt = np.diff(t[sl], prepend=t[begin])
    residual = pos[sl, :2] - pos[begin, :2] - np.cumsum(target_world * post_dt[:, None], axis=0)
    lateral = residual @ np.array([-np.sin(yaw[begin]), np.cos(yaw[begin])])
    pre_disturbance = max(0, int(np.searchsorted(t, window.start_s)) - 1)
    heading_baseline = heading_error[pre_disturbance] if window.start_s > 0 else 0.
    robustness.append(dict(kind=window.kind, observed=True,
      peak_velocity_error=np.abs(error[during]).max(0).tolist(),
      peak_roll_pitch=np.abs(rpy[during, :2]).max(0).tolist(),
      heading_deviation=float(np.abs(heading_error[during] - heading_baseline).max()),
      recovery_time=recovery_time(t[sl], error[sl], rpy[sl, :2], window.end_s, tolerance),
      post_disturbance_velocity_rmse=rms(error[sl]),
      post_disturbance_lateral_drift=float(np.abs(lateral).max()), fall_rate=float(fallen)))
  for row in robustness:
    row["recovery_censored"] = row["recovery_time"] is None
  result["robustness"] = robustness
  return result
