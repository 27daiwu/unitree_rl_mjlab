"""Offline W1.7 attribution from saved W1.6 fixed-command NPZ traces."""

from pathlib import Path
import json

import numpy as np

from analyze_flat_w1_6 import capsule_points
from diagnose_flat_gait import COMMANDS, _rotate_quat, _segments


ROOT = Path("logs/flat_w1_6_toe_clearance_20261030/paired_diagnostics")
OUT = Path("logs/flat_w1_7_attribution_20261030")
FEET = ("left", "right")
LABELS = ("control", "experiment")
SCENARIOS = ("backward_015", "left_020", "right_020")
PLANNED_BINS = ((0.50, 0.60), (0.60, 0.66), (0.66, 0.70), (0.70, 0.80),
                (0.80, 0.90), (0.90, 1.00))
ACTUAL_BINS = ((0.00, 0.25), (0.25, 0.75), (0.75, 1.00))


def load_trace(label, scenario, replica, geometry, radius):
    with np.load(ROOT / f"{label}_{scenario}_{replica}.npz") as z:
        assert len(z["time"]) == 600
        keep = z["time"] > 7.0
        t = z["time"][keep]
        site = z["foot_position"][keep]
        quats = z["foot_quaternion"][keep]
        contact = ((z["contact_time"][keep] > 0) |
                   (z["normal_force"][keep] >= 5))
        toe = np.empty((len(t), 2), dtype=float)
        for index, foot in enumerate(FEET):
            points = geometry[foot][0]
            rotated = _rotate_quat(quats[:, index, None, :], points[None])
            toe[:, index] = (site[:, index, None, 2] + rotated[..., 2]).min(axis=1) - radius
        phase = (np.rint(t[:, None] / 0.02) / 30 + np.array([0.0, 0.5])[None]) % 1.0
        actual_progress = np.full_like(toe, np.nan)
        for index in range(2):
            for start, end in _segments(~contact[:, index]):
                if start == 0 or end == len(t) or end - start < 2:
                    continue
                actual_progress[start:end, index] = (np.arange(end - start) + 0.5) / (end - start)
        return {"time": t, "velocity": z["velocity"][keep], "toe": toe,
                "phase": phase, "contact": contact, "actual_progress": actual_progress}


def describe_toe(values, mask):
    selected = values[mask]
    if not len(selected):
        return {"n": 0, "mean_mm": None, "below_30_fraction": None}
    return {"n": int(len(selected)), "mean_mm": float(selected.mean() * 1000),
            "below_30_fraction": float(np.mean(selected < 0.03))}


def velocity_metrics(velocity, command):
    axis = int(np.flatnonzero(command)[0])
    error = velocity[:, axis] - command[axis]
    result = {"axis_mean_m_s": float(velocity[:, axis].mean()),
              "axis_bias_m_s": float(error.mean()),
              "axis_std_m_s": float(error.std()),
              "axis_rmse_m_s": float(np.sqrt(np.mean(error ** 2))),
              "axis_error_squared_m2_s2": float(np.mean(error ** 2))}
    if axis == 1:
        result.update(x_drift_mean_m_s=float(velocity[:, 0].mean()),
                      x_drift_rmse_m_s=float(np.sqrt(np.mean(velocity[:, 0] ** 2))),
                      yaw_drift_mean_rad_s=float(velocity[:, 2].mean()),
                      yaw_drift_rmse_rad_s=float(np.sqrt(np.mean(velocity[:, 2] ** 2))))
    return result


def correlation(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze():
    geometry, radius = capsule_points()
    traces = {(scenario, label, replica): load_trace(label, scenario, replica, geometry, radius)
              for scenario in SCENARIOS for label in LABELS for replica in range(4)}
    result = {"source": str(ROOT), "steady_window_s": [7.0, 12.0],
              "planned_phase_bins": PLANNED_BINS, "actual_swing_progress_bins": ACTUAL_BINS,
              "scenarios": {}}
    for scenario in SCENARIOS:
        command = np.asarray(COMMANDS[scenario])
        axis = int(np.flatnonzero(command)[0])
        row = {"command": command.tolist(), "feet": {}, "velocity": {}, "association": {}}
        for foot_index, foot in enumerate(FEET):
            row["feet"][foot] = {"planned": {}, "actual": {}}
            for lower, upper in PLANNED_BINS:
                key = f"{lower:.2f}-{upper:.2f}"
                for label in LABELS:
                    toe = np.concatenate([traces[scenario, label, i]["toe"][:, foot_index] for i in range(4)])
                    phase = np.concatenate([traces[scenario, label, i]["phase"][:, foot_index] for i in range(4)])
                    contact = np.concatenate([traces[scenario, label, i]["contact"][:, foot_index] for i in range(4)])
                    mask = (phase >= lower - 1e-6) & (phase < upper - 1e-6)
                    metrics = describe_toe(toe, mask)
                    metrics["no_contact_fraction"] = float(np.mean(~contact[mask])) if mask.any() else None
                    row["feet"][foot]["planned"].setdefault(key, {})[label] = metrics
            for lower, upper in ACTUAL_BINS:
                key = f"{lower:.2f}-{upper:.2f}"
                for label in LABELS:
                    toe = np.concatenate([traces[scenario, label, i]["toe"][:, foot_index] for i in range(4)])
                    progress = np.concatenate([traces[scenario, label, i]["actual_progress"][:, foot_index] for i in range(4)])
                    mask = (progress >= lower) & (progress < upper)
                    row["feet"][foot]["actual"].setdefault(key, {})[label] = describe_toe(toe, mask)
        for label in LABELS:
            velocity = np.concatenate([traces[scenario, label, i]["velocity"] for i in range(4)])
            row["velocity"][label] = velocity_metrics(velocity, command)
            row["velocity"][label]["replicas"] = [
                velocity_metrics(traces[scenario, label, i]["velocity"], command) for i in range(4)]

        toe_delta_replica = []
        rmse_delta_replica = []
        paired_windows = []
        for replica in range(4):
            control = traces[scenario, "control", replica]
            experiment = traces[scenario, "experiment", replica]
            assert np.array_equal(control["time"], experiment["time"])
            mid = (control["phase"] > 0.66 + 1e-6) & (control["phase"] < 0.90 - 1e-6)
            toe_change = float((experiment["toe"][mid] - control["toe"][mid]).mean() * 1000)
            c_rmse = row["velocity"]["control"]["replicas"][replica]["axis_rmse_m_s"]
            e_rmse = row["velocity"]["experiment"]["replicas"][replica]["axis_rmse_m_s"]
            toe_delta_replica.append(toe_change)
            rmse_delta_replica.append((e_rmse - c_rmse) * 1000)
            for second in range(7, 12):
                sample = (control["time"] > second) & (control["time"] <= second + 1)
                phase_sample = mid & sample[:, None]
                c_error = control["velocity"][sample, axis] - command[axis]
                e_error = experiment["velocity"][sample, axis] - command[axis]
                paired_windows.append({
                    "replica": replica, "second": second,
                    "toe_delta_mm": float((experiment["toe"][phase_sample] - control["toe"][phase_sample]).mean() * 1000),
                    "axis_rmse_delta_mm_s": float((np.sqrt(np.mean(e_error ** 2)) - np.sqrt(np.mean(c_error ** 2))) * 1000),
                })
        row["association"]["replica_toe_delta_mm"] = toe_delta_replica
        row["association"]["replica_rmse_delta_mm_s"] = rmse_delta_replica
        row["association"]["replica_correlation"] = correlation(toe_delta_replica, rmse_delta_replica)
        row["association"]["paired_one_second_windows"] = paired_windows
        row["association"]["windows_toe_up_rmse_worse"] = sum(
            window["toe_delta_mm"] > 0 and window["axis_rmse_delta_mm_s"] > 0 for window in paired_windows)
        row["association"]["window_correlation"] = correlation(
            [window["toe_delta_mm"] for window in paired_windows],
            [window["axis_rmse_delta_mm_s"] for window in paired_windows])
        row["association"]["planned_phase"] = {}
        for lower, upper in PLANNED_BINS:
            key = f"{lower:.2f}-{upper:.2f}"
            toe_changes = []
            squared_error_changes = []
            for replica in range(4):
                control = traces[scenario, "control", replica]
                experiment = traces[scenario, "experiment", replica]
                mask = (control["phase"] >= lower - 1e-6) & (control["phase"] < upper - 1e-6)
                toe_changes.extend((experiment["toe"][mask] - control["toe"][mask]).tolist())
                # Include a time sample once when either foot is in this phase bin.
                time_mask = mask.any(axis=1)
                c_error = control["velocity"][time_mask, axis] - command[axis]
                e_error = experiment["velocity"][time_mask, axis] - command[axis]
                squared_error_changes.extend((e_error ** 2 - c_error ** 2).tolist())
            row["association"]["planned_phase"][key] = {
                "toe_delta_mean_mm": float(np.mean(toe_changes) * 1000),
                "axis_squared_error_delta_mean_m2_s2": float(np.mean(squared_error_changes))}
        result["scenarios"][scenario] = row
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    result = analyze()
    OUT.mkdir(parents=True)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(OUT / "metrics.json")


if __name__ == "__main__":
    main()
