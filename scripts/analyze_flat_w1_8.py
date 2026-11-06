"""Summarize W1.8 fixed-command traces against saved W1.3/W1.6 results."""

from pathlib import Path
import json

import numpy as np

from analyze_flat_w1_6 import capsule_points
from diagnose_flat_gait import COMMANDS, _rotate_quat, _segments


ROOT = Path("logs/flat_w1_8_smooth_toe_20261030")
DATA = ROOT / "paired_diagnostics"
FEET = ("left", "right")
LABELS = ("control", "experiment")
PHASE_BINS = ((0.50, 0.60), (0.60, 0.66), (0.66, 0.70), (0.70, 0.80),
              (0.80, 0.90), (0.90, 1.00))


def trace_metrics(path, command, geometry, radius):
    with np.load(path) as z:
        assert len(z["time"]) == 600
        keep = z["time"] > 7.0
        time = z["time"][keep]
        velocity = z["velocity"][keep]
        site = z["foot_position"][keep]
        quat = z["foot_quaternion"][keep]
        contact = ((z["contact_time"][keep] > 0) |
                   (z["normal_force"][keep] >= 5))
        phase = (np.rint(time[:, None] / 0.02) / 30 + np.array([0.0, 0.5])[None]) % 1.0
        axis = int(np.flatnonzero(command)[0])
        error = velocity[:, axis] - command[axis]
        result = {
            "axis_mean_m_s": float(velocity[:, axis].mean()),
            "axis_bias_m_s": float(error.mean()),
            "axis_std_m_s": float(error.std()),
            "axis_rmse_m_s": float(np.sqrt(np.mean(error ** 2))),
            "cross_axis_mean_m_s": float(velocity[:, 1 - axis].mean()) if axis < 2 else None,
            "yaw_mean_rad_s": float(velocity[:, 2].mean()),
            "feet": {}, "phase": {},
        }
        for index, foot in enumerate(FEET):
            points = geometry[foot][0]
            toe = (site[:, index, None, 2] +
                   _rotate_quat(quat[:, index, None, :], points[None])[:, :, 2]).min(axis=1) - radius
            mid = (phase[:, index] > 0.66 + 1e-6) & (phase[:, index] < 0.90 - 1e-6)
            progress = np.full(len(time), np.nan)
            peaks = []
            for start, end in _segments(~contact[:, index]):
                if start == 0 or end == len(time) or end - start < 2:
                    continue
                progress[start:end] = (np.arange(end - start) + 0.5) / (end - start)
                peaks.append(float(toe[start:end].max()))
            actual_mid = (progress >= 0.25) & (progress < 0.75)
            result["feet"][foot] = {
                "planned_mid_toe_mean_m": float(toe[mid].mean()),
                "planned_mid_toe_ge_30mm_fraction": float(np.mean(toe[mid] >= 0.03)),
                "actual_mid_toe_mean_m": float(toe[actual_mid].mean()),
                "actual_swing_toe_peak_mean_m": float(np.mean(peaks)),
                "planned_mid_contact_fraction": float(contact[mid, index].mean()),
            }
        for lower, upper in PHASE_BINS:
            mask = ((phase >= lower - 1e-6) & (phase < upper - 1e-6)).any(axis=1)
            section = error[mask]
            result["phase"][f"{lower:.2f}-{upper:.2f}"] = {
                "n": int(mask.sum()), "axis_bias_m_s": float(section.mean()),
                "axis_variance_m2_s2": float(section.var()),
                "axis_mse_m2_s2": float(np.mean(section ** 2)),
            }
        return result


def mean_rows(rows):
    first = rows[0]
    if isinstance(first, dict):
        return {key: mean_rows([row[key] for row in rows]) for key in first}
    if first is None:
        return None
    return float(np.mean(rows))


def main():
    output = ROOT / "comparison_metrics.json"
    if output.exists():
        raise FileExistsError(output)
    geometry, radius = capsule_points()
    history = {
        "w1_3": (Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2"), "w1_3"),
        "w1_6_rectangle": (Path("logs/flat_w1_6_toe_clearance_20261030/paired_diagnostics"), "experiment"),
    }
    rows = {}
    for scenario, command in COMMANDS.items():
        rows[scenario] = {"command": list(command), "policies": {}}
        for label in LABELS:
            trials = [trace_metrics(DATA / f"{label}_{scenario}_{replica}.npz", command, geometry, radius)
                      for replica in range(4)]
            rows[scenario]["policies"][label] = mean_rows(trials)
            rows[scenario]["policies"][label]["replicas"] = trials
        rows[scenario]["history"] = {}
        for historical_label, (directory, trace_label) in history.items():
            trials = [trace_metrics(directory / f"{trace_label}_{scenario}_{replica}.npz",
                                    command, geometry, radius) for replica in range(4)]
            rows[scenario]["history"][historical_label] = mean_rows(trials)
    output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
