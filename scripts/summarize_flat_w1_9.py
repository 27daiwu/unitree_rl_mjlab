"""Read-only comparison of saved A4, W1.3, W1.6, and W1.8 Flat traces."""

from pathlib import Path
import json

import numpy as np

from analyze_flat_w1_6 import capsule_points
from analyze_flat_w1_8 import trace_metrics
from diagnose_flat_gait import COMMANDS


OUT = Path("logs/flat_w1_9_baseline_20261030")
SOURCES = {
    "a4": (Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2"), "a4"),
    "w1_3": (Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2"), "w1_3"),
    "w1_6_rectangle": (Path("logs/flat_w1_6_toe_clearance_20261030/paired_diagnostics"), "experiment"),
    "w1_8_smooth": (Path("logs/flat_w1_8_smooth_toe_20261030/paired_diagnostics"), "experiment"),
}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geometry, radius = capsule_points()
    data = {"sources": {label: {"diagnostics": str(folder), "trace_label": trace_label}
                        for label, (folder, trace_label) in SOURCES.items()}, "commands": {}}
    for scenario, command in COMMANDS.items():
        data["commands"][scenario] = {}
        for label, (folder, trace_label) in SOURCES.items():
            trials = [trace_metrics(folder / f"{trace_label}_{scenario}_{replica}.npz",
                                    command, geometry, radius) for replica in range(4)]
            stored = json.loads((folder / "metrics.json").read_text())["comparison"][scenario][trace_label]
            data["commands"][scenario][label] = {
                "axis_rmse_m_s_or_rad_s": float(np.mean([trial["axis_rmse_m_s"] for trial in trials])),
                "axis_bias_m_s_or_rad_s": float(np.mean([trial["axis_bias_m_s"] for trial in trials])),
                "axis_std_m_s_or_rad_s": float(np.mean([trial["axis_std_m_s"] for trial in trials])),
                "toe_planned_mid_left_m": float(np.mean([trial["feet"]["left"]["planned_mid_toe_mean_m"] for trial in trials])),
                "toe_planned_mid_right_m": float(np.mean([trial["feet"]["right"]["planned_mid_toe_mean_m"] for trial in trials])),
                "toe_actual_mid_left_m": float(np.mean([trial["feet"]["left"]["actual_mid_toe_mean_m"] for trial in trials])),
                "toe_actual_mid_right_m": float(np.mean([trial["feet"]["right"]["actual_mid_toe_mean_m"] for trial in trials])),
                "step_length_left_m": stored["feet"]["left"]["step_length_mean_m"],
                "step_length_right_m": stored["feet"]["right"]["step_length_mean_m"],
                "roll_pitch_rms_rad": stored["roll_pitch_rms_rad"],
                "completed_trials": stored["completed_trials"],
            }
    OUT.mkdir(parents=True)
    path = OUT / "metrics.json"
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
