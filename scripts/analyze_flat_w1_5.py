"""Read-only gait attribution from saved W1.3-D and W1.4 traces."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from diagnose_flat_gait import _rotate_quat, _segments, load_foot_collision_points


ROOT = Path("logs/flat_w1_4_min_reward_20261030")
OUT = Path("logs/flat_w1_5_attribution_20261030")
SOURCE = Path("logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt")
DATA = {"w1_3": ROOT / "baseline_diagnostics_v2",
        "control": ROOT / "paired_diagnostics", "experiment": ROOT / "paired_diagnostics"}
SCENARIOS = ("forward_020", "forward_040", "backward_015", "left_020", "right_020")
FEET = ("left", "right")


def mean(values):
    return float(np.mean(values))


def trace_metrics(path, geom, joint_names):
    with np.load(path) as z:
        t = z["time"]
        assert len(t) == 600
        keep = t > 7.0
        q = z["joint_pos"][keep]
        margin = np.minimum(q - z["joint_limits"][keep, :, 0],
                            z["joint_limits"][keep, :, 1] - q)
        sites = z["foot_position"][keep]
        quats = z["foot_quaternion"][keep]
        velocities = z["foot_velocity"][keep]
        contact = (z["contact_time"][keep] > 0) | (z["normal_force"][keep] >= 5)
        joint_ranges = dict(zip(joint_names, np.ptp(q, axis=0).tolist()))
        result = {"joint_ranges_rad": joint_ranges, "min_soft_joint_margin_rad": float(np.min(margin)),
                  "feet": {}}
        for idx, foot in enumerate(FEET):
            points = geom[foot]
            rotated = _rotate_quat(quats[:, idx, None, :], points[None, :, :])
            heights = sites[:, idx, None, 2] + rotated[:, :, 2]
            toe = np.min(heights[:, points[:, 0] > 0.05], axis=1)
            heel = np.min(heights[:, points[:, 0] < -0.03], axis=1)
            events = []
            for start, end in _segments(~contact[:, idx]):
                if end - start < 2:
                    continue
                peak = start + int(np.argmax(toe[start:end]))
                events.append((float(toe[peak]), float(sites[peak, idx, 2]), float(heel[peak])))
            speed = np.linalg.norm(velocities[:, idx, :2], axis=-1)
            result["feet"][foot] = {
                "toe_peak_events": events,
                "site_swing_z_p95_m": float(np.percentile(sites[~contact[:, idx], idx, 2], 95)),
                "site_above_target_swing_fraction": float(np.mean(sites[~contact[:, idx], idx, 2] >= 0.1)),
                "swing_site_speed_mean_m_s": float(np.mean(speed[~contact[:, idx]])),
            }
        return result


def checkpoint_optimizer(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer = state["optimizer_state_dict"]
    steps = [int(s["step"]) for s in optimizer["state"].values() if "step" in s]
    return {"iteration": state["iter"], "learning_rate": optimizer["param_groups"][0]["lr"],
            "adam_state_entries": len(optimizer["state"]), "adam_step_min_max": [min(steps), max(steps)]}


def training_scalars(path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
    names = ("Metrics/twist/error_vel_xy", "Episode_Reward/track_linear_velocity",
             "Episode_Reward/pose", "Episode_Reward/foot_gait", "Episode_Reward/foot_clearance",
             "Episode_Termination/fell_over", "Loss/learning_rate", "Policy/mean_std")
    result = {}
    for name in names:
        values = np.asarray([entry.value for entry in accumulator.Scalars(name)])
        result[name] = {"first25": mean(values[:25]), "last25": mean(values[-25:]),
                        "first": float(values[0]), "last": float(values[-1])}
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geom = load_foot_collision_points("src/assets/robots/unitree_g1/xmls/g1.xml")
    xml = ET.parse("src/assets/robots/unitree_g1/xmls/g1.xml").getroot()
    joint_names = [joint.attrib["name"] for joint in xml.iter("joint") if joint.attrib.get("type") != "free"]
    assert len(joint_names) == 29 and joint_names[:2] == ["left_hip_pitch_joint", "left_hip_roll_joint"]
    saved = {"w1_3": json.loads((DATA["w1_3"] / "metrics.json").read_text()),
             "control": json.loads((DATA["control"] / "metrics.json").read_text()),
             "experiment": json.loads((DATA["experiment"] / "metrics.json").read_text())}
    rows = {}
    for scenario in SCENARIOS:
        rows[scenario] = {}
        for label, folder in DATA.items():
            aggregate = saved[label]["comparison"][scenario][label]
            trials = [trace_metrics(folder / f"{label}_{scenario}_{i}.npz", geom, joint_names) for i in range(4)]
            command = saved[label]["trials"][f"{label}/{scenario}/replica_0"]["command_velocity"]
            axis = int(np.flatnonzero(command)[0])
            period = 1.0 / aggregate["step_frequency_hz"]
            velocity = abs(aggregate["actual_velocity_mean"][axis])
            row = {"command_m_s": command[axis], "actual_axis_speed_m_s": velocity,
                   "axis_rmse_m_s": aggregate["velocity_tracking_rmse"][axis],
                   "touchdown_period_s": period,
                   "speed_times_period_m": velocity * period,
                   "min_soft_joint_margin_rad": min(t["min_soft_joint_margin_rad"] for t in trials),
                   "actuator_effort_limit_fraction_peak": aggregate["actuator_effort_limit_fraction_peak"],
                   "feet": {}}
            for foot in FEET:
                f = aggregate["feet"][foot]
                events = [event for trial in trials for event in trial["feet"][foot]["toe_peak_events"]]
                row["feet"][foot] = {
                    "stride_m": f["step_length_mean_m"],
                    "swing_excursion_m": f["swing_excursion_mean_m"],
                    "swing_duration_s": f["swing_duration_mean_s"],
                    "site_peak_m": f["foot_site_swing_clearance_peak_mean_m"],
                    "toe_peak_m": f["toe_clearance_peak_mean_m"],
                    "heel_peak_m": f["heel_clearance_peak_mean_m"],
                    "toe_peak_event_count": len(events),
                    "site_at_toe_peak_m": mean([e[1] for e in events]),
                    "heel_at_toe_peak_m": mean([e[2] for e in events]),
                    "site_swing_z_p95_m": mean([t["feet"][foot]["site_swing_z_p95_m"] for t in trials]),
                    "site_above_target_swing_fraction": mean([t["feet"][foot]["site_above_target_swing_fraction"] for t in trials]),
                    "swing_site_speed_mean_m_s": mean([t["feet"][foot]["swing_site_speed_mean_m_s"] for t in trials]),
                    "joint_range_rad": {name: mean([t["joint_ranges_rad"][name] for t in trials])
                                        for name in joint_names if name.startswith(foot + "_") and
                                        any(part in name for part in ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"))},
                }
            rows[scenario][label] = row
    optimizer = {"source": checkpoint_optimizer(SOURCE)}
    for label in ("control", "experiment"):
        optimizer[label] = {str(i): checkpoint_optimizer(ROOT / label / f"model_{i}.pt")
                            for i in (0, 100, 200, 300, 399)}
    scalars = {label: training_scalars(ROOT / label) for label in ("control", "experiment")}
    OUT.mkdir(parents=True)
    (OUT / "mechanism_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (OUT / "optimizer_history.json").write_text(json.dumps(optimizer, indent=2) + "\n")
    (OUT / "training_scalars.json").write_text(json.dumps(scalars, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "scenarios": list(rows), "traces_read": 60}, indent=2))


if __name__ == "__main__":
    main()
