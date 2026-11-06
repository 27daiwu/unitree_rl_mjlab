"""Analyze saved W1.13 fixed-command traces and frozen acceptance gates."""

from pathlib import Path
import json

import numpy as np

from analyze_flat_w1_6 import capsule_points
from analyze_flat_w1_8 import trace_metrics
from diagnose_flat_gait import COMMANDS


ROOT = Path("logs/flat_w1_13_optimizer_restore_20261104")
CURRENT = ROOT / "paired_diagnostics_matched"
W13 = Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2")
W11 = Path("logs/flat_w1_11_right_track_20261030/paired_diagnostics_cpu")
W11_FINAL = Path("logs/flat_w1_11_right_track_20261030/comparison_cpu_final.json")
OUTPUT = ROOT / "comparison_matched_final.json"
FEET = ("left", "right")


def summarize(directory, label, scenario, command, geometry, radius):
  traces = [trace_metrics(directory / f"{label}_{scenario}_{replica}.npz",
                          command, geometry, radius) for replica in range(4)]
  metrics = json.loads((directory / "metrics.json").read_text())
  group = metrics["comparison"][scenario][label]
  axis = int(np.flatnonzero(command)[0])
  row = {
    "rmse_m_s_or_rad_s": float(np.mean([trace["axis_rmse_m_s"] for trace in traces])),
    "bias_m_s_or_rad_s": float(np.mean([trace["axis_bias_m_s"] for trace in traces])),
    "std_m_s_or_rad_s": float(np.mean([trace["axis_std_m_s"] for trace in traces])),
    "completed_trials": group["completed_trials"],
    "roll_pitch_rms_rad": group["roll_pitch_rms_rad"],
    "step_length_mm": {foot: float(np.mean([
      group["feet"][foot]["step_length_mean_m"] * 1000
      for _ in range(1)])) for foot in FEET},
    "toe_mid_mean_mm": {foot: float(np.mean([
      trace["feet"][foot]["planned_mid_toe_mean_m"] * 1000 for trace in traces]))
      for foot in FEET},
  }
  return row


def main():
  if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
  geometry, radius = capsule_points()
  rows = {}
  for scenario, command in COMMANDS.items():
    rows[scenario] = {"command": list(command), "policies": {}}
    for label, directory, checkpoint_label in (
      ("control", ROOT / "matched_reward_control/paired_diagnostics", "control"),
      ("experiment", ROOT / "paired_diagnostics_matched", "experiment"),
      ("w1_3", W13, "w1_3"), ("w1_11_control", W11, "control"),
      ("w1_11_experiment", W11, "experiment"),
    ):
      rows[scenario]["policies"][label] = summarize(
        directory, checkpoint_label, scenario, command, geometry, radius)
  w11_gates = json.loads(W11_FINAL.read_text())["gates"]

  gates = {}
  def gate(name, value, limit, relation="max"):
    passed = value <= limit + 1e-12 if relation == "max" else value >= limit - 1e-12
    gates[name] = {"value": value, "limit": limit, "relation": relation, "passed": bool(passed)}

  backward = rows["backward_015"]["policies"]
  c, e, source = backward["control"], backward["experiment"], backward["w1_3"]
  gate("backward_rmse_gain_20pct", c["rmse_m_s_or_rad_s"] - e["rmse_m_s_or_rad_s"],
       .20 * c["rmse_m_s_or_rad_s"], "min")
  gate("backward_rmse_absolute_vs_w13", e["rmse_m_s_or_rad_s"],
       1.25 * source["rmse_m_s_or_rad_s"])
  for foot, source_min in zip(FEET, (66.6, 65.7)):
    gate(f"backward_{foot}_stride_vs_w13", e["step_length_mm"][foot], source_min, "min")

  right = rows["right_020"]["policies"]
  c, e, w11e = right["control"], right["experiment"], right["w1_11_experiment"]
  gate("right_rmse_gain_vs_control",
       (c["rmse_m_s_or_rad_s"] - e["rmse_m_s_or_rad_s"]) * 1000, 15.0, "min")
  gate("right_rmse_absolute_vs_w11", e["rmse_m_s_or_rad_s"],
       1.10 * w11e["rmse_m_s_or_rad_s"])
  gate("right_abs_bias_vs_control", abs(e["bias_m_s_or_rad_s"]) -
       abs(c["bias_m_s_or_rad_s"]), .010)

  for scenario in ("forward_020", "forward_040", "left_020", "turn_left_050", "turn_right_050"):
    c, e = rows[scenario]["policies"]["control"], rows[scenario]["policies"]["experiment"]
    gate(f"{scenario}_rmse_nonregression_10pct", e["rmse_m_s_or_rad_s"],
         c["rmse_m_s_or_rad_s"] * 1.10)
  for foot in FEET:
    c, e = rows["left_020"]["policies"]["control"], rows["left_020"]["policies"]["experiment"]
    gate(f"left_{foot}_stride_nonregression_10pct", e["step_length_mm"][foot],
         c["step_length_mm"][foot] * .90, "min")

  for scenario in COMMANDS:
    c, e = rows[scenario]["policies"]["control"], rows[scenario]["policies"]["experiment"]
    gate(f"{scenario}_completion_control", c["completed_trials"], 4, "min")
    gate(f"{scenario}_completion_experiment", e["completed_trials"], 4, "min")
    for axis, name in enumerate(("roll", "pitch")):
      gate(f"{scenario}_{name}_rms", e["roll_pitch_rms_rad"][axis],
           c["roll_pitch_rms_rad"][axis] + .010)
  for scenario in ("backward_015", "right_020"):
    c, e = rows[scenario]["policies"]["control"], rows[scenario]["policies"]["experiment"]
    for foot in FEET:
      gate(f"{scenario}_{foot}_toe", e["toe_mid_mean_mm"][foot],
           c["toe_mid_mean_mm"][foot] - 5.0, "min")

  result = {"rows": rows, "w1_11_original_gates_passed": all(
    item["passed"] for item in w11_gates.values()), "gates": gates,
    "passed": all(item["passed"] for item in gates.values())}
  OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(json.dumps({"output": str(OUTPUT), "passed": result["passed"],
                    "gates_passed": sum(v["passed"] for v in gates.values()),
                    "gates_total": len(gates)}, indent=2))


if __name__ == "__main__":
  main()
