"""Evaluate saved W1.11 traces against the W1.10 acceptance gates."""

from pathlib import Path
import json

import numpy as np

from analyze_flat_w1_6 import capsule_points
from analyze_flat_w1_8 import trace_metrics
from diagnose_flat_gait import COMMANDS


ROOT = Path("logs/flat_w1_11_right_track_20261030")
DIAGNOSTICS = ROOT / "paired_diagnostics_cpu"
HISTORY = Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2")
OUTPUT = ROOT / "comparison_cpu_final.json"
FEET = ("left", "right")
GROUPS = ("control", "experiment")


def main():
  if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
  geometry, radius = capsule_points()
  stored = json.loads((DIAGNOSTICS / "metrics.json").read_text())["comparison"]
  historical = json.loads((HISTORY / "metrics.json").read_text())["comparison"]
  rows = {}
  for scenario, command in COMMANDS.items():
    rows[scenario] = {}
    for group in GROUPS:
      trials = [trace_metrics(DIAGNOSTICS / f"{group}_{scenario}_{i}.npz",
                              command, geometry, radius) for i in range(4)]
      summary = stored[scenario][group]
      rows[scenario][group] = {
        "axis_rmse_m_s_or_rad_s": float(np.mean([t["axis_rmse_m_s"] for t in trials])),
        "axis_bias_m_s_or_rad_s": float(np.mean([t["axis_bias_m_s"] for t in trials])),
        "axis_std_m_s_or_rad_s": float(np.mean([t["axis_std_m_s"] for t in trials])),
        "cross_axis_rmse_m_s": summary["velocity_tracking_rmse"][0] if scenario in
          ("left_020", "right_020") else None,
        "yaw_rmse_rad_s": summary["velocity_tracking_rmse"][2],
        "roll_pitch_rms_rad": summary["roll_pitch_rms_rad"],
        "completed_trials": summary["completed_trials"],
        "feet": {foot: {
          "step_length_m": summary["feet"][foot]["step_length_mean_m"],
          "swing_excursion_m": summary["feet"][foot]["swing_excursion_mean_m"],
          "planned_mid_toe_mean_m": float(np.mean([
            trial["feet"][foot]["planned_mid_toe_mean_m"] for trial in trials])),
        } for foot in FEET},
      }
  right = rows["right_020"]
  source_rmse = historical["right_020"]["w1_3"]["velocity_tracking_rmse"][1]
  gates = {}

  def gate(name, value, limit, relation="max"):
    passed = value <= limit + 1e-12 if relation == "max" else value >= limit - 1e-12
    gates[name] = {"value": value, "limit": limit, "relation": relation, "passed": bool(passed)}

  c, e = right["control"], right["experiment"]
  gate("right_rmse_gain", c["axis_rmse_m_s_or_rad_s"] - e["axis_rmse_m_s_or_rad_s"], .015, "min")
  gate("right_rmse_vs_w1_3", e["axis_rmse_m_s_or_rad_s"], source_rmse)
  gate("right_abs_bias_gain", abs(c["axis_bias_m_s_or_rad_s"]) -
       abs(e["axis_bias_m_s_or_rad_s"]), .025, "min")
  gate("right_std", e["axis_std_m_s_or_rad_s"], c["axis_std_m_s_or_rad_s"] + .010)
  gate("right_cross_axis_rmse", e["cross_axis_rmse_m_s"], c["cross_axis_rmse_m_s"] + .010)
  gate("right_yaw_rmse", e["yaw_rmse_rad_s"], c["yaw_rmse_rad_s"] + .030)
  for foot in FEET:
    gate(f"right_{foot}_stride_gain", e["feet"][foot]["step_length_m"] -
         c["feet"][foot]["step_length_m"], .010, "min")
    gate(f"right_{foot}_swing_excursion", e["feet"][foot]["swing_excursion_m"],
         c["feet"][foot]["swing_excursion_m"], "min")
  for scenario in COMMANDS:
    c, e = rows[scenario]["control"], rows[scenario]["experiment"]
    if scenario not in ("right_020",):
      tolerance = .030 if scenario.startswith("turn_") else .010
      gate(f"{scenario}_axis_rmse", e["axis_rmse_m_s_or_rad_s"],
           c["axis_rmse_m_s_or_rad_s"] + tolerance)
    if scenario in ("backward_015", "left_020"):
      for foot in FEET:
        gate(f"{scenario}_{foot}_stride", e["feet"][foot]["step_length_m"],
             c["feet"][foot]["step_length_m"] - .005, "min")
    if scenario in ("backward_015", "left_020", "right_020"):
      for foot in FEET:
        gate(f"{scenario}_{foot}_toe", e["feet"][foot]["planned_mid_toe_mean_m"],
             c["feet"][foot]["planned_mid_toe_mean_m"] - .005, "min")
    for axis, name in enumerate(("roll", "pitch")):
      gate(f"{scenario}_{name}_rms", e["roll_pitch_rms_rad"][axis],
           c["roll_pitch_rms_rad"][axis] + .010)
    gate(f"{scenario}_completed_trials", e["completed_trials"], 4, "min")
    gate(f"{scenario}_control_completed_trials", c["completed_trials"], 4, "min")
  result = {"source": str(DIAGNOSTICS), "historical_w1_3_right_rmse_m_s": source_rmse,
            "commands": rows, "gates": gates,
            "passed": all(item["passed"] for item in gates.values())}
  OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(OUTPUT, "PASS" if result["passed"] else "FAIL")


if __name__ == "__main__":
  main()
