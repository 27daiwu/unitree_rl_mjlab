"""Generate the W1.4 paired diagnostic tables and verification artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [row for k in sorted(a.keys() | b.keys())
                for row in differences(a.get(k), b.get(k), f"{prefix}/{k}")]
    return [] if a == b else [{"path": prefix, "control": a, "experiment": b}]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/flat_w1_4_min_reward_20261030"))
    args = parser.parse_args()
    root = args.root
    data = json.loads((root / "paired_diagnostics/metrics.json").read_text())
    baseline = json.loads((root / "baseline_diagnostics_v2/metrics.json").read_text())
    diffs = {}
    for name in ("env", "agent"):
        cfgs = [yaml.load((root / g / f"params/{name}.yaml").read_text(), Loader=yaml.BaseLoader)
                for g in ("control", "experiment")]
        diffs[name] = differences(*cfgs)
    assert [d["path"] for d in diffs["env"]] == ["/rewards/foot_gait/weight"]
    assert [d["path"] for d in diffs["agent"]] == ["/run_name"]
    (root / "paired_config_diff.json").write_text(json.dumps(diffs, indent=2) + "\n")
    original = json.loads((root / "preserved_assets_sha256.json").read_text())
    changed = [p for p, value in original.items() if not Path(p).is_file() or sha(Path(p)) != value]
    assert not changed, changed
    (root / "preservation_check.json").write_text(json.dumps({"checked_files": len(original), "changed": changed}, indent=2) + "\n")

    feet = ("left", "right")
    rows = []
    for scenario, groups in data["comparison"].items():
        c, e = groups["control"], groups["experiment"]
        command = data["trials"][f"control/{scenario}/replica_0"]["command_velocity"]
        axis = int(np.flatnonzero(command)[0])
        row = {"scenario": scenario, "command_axis": axis}
        for label, values in (("w1_3", baseline["comparison"][scenario]["w1_3"]),
                              ("control", c), ("experiment", e)):
            row[label] = {"velocity": values["actual_velocity_mean"][axis],
                          "axis_rmse": values["velocity_tracking_rmse"][axis],
                          "vector_rmse": values["velocity_tracking_rmse"],
                          "roll_pitch_rms": values["roll_pitch_rms_rad"],
                          "completed": values["completed_trials"],
                          "double_support": values["double_support_fraction"],
                          "frequency": values["step_frequency_hz"]}
            for key in ("step_length_mean_m", "swing_excursion_mean_m", "toe_clearance_peak_mean_m",
                        "heel_clearance_peak_mean_m", "stance_foot_speed_mean_m_s"):
                row[label][key] = [values["feet"][f][key] for f in feet]
        row["experiment_minus_control"] = {key: (np.asarray(row["experiment"][key]) - np.asarray(row["control"][key])).tolist()
                                            for key in row["control"]}
        rows.append(row)
    (root / "comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    lines = ["# W1.4 paired results", "", "All lengths are mm; same-foot touchdown root displacement is a stride proxy.", "",
             "| Command | Policy | Axis velocity | Axis RMSE | Stride L/R mm | Toe L/R mm | Swing excursion L/R mm | Roll/pitch RMS rad | Completed |",
             "|---|---|---:|---:|---|---|---|---|---:|"]
    for row in rows:
        for group in ("w1_3", "control", "experiment"):
            v = row[group]
            pair = lambda key: "/".join(f"{1000*x:.1f}" for x in v[key])
            lines.append(f"| {row['scenario']} | {group} | {v['velocity']:.4f} | {v['axis_rmse']:.4f} | {pair('step_length_mean_m')} | {pair('toe_clearance_peak_mean_m')} | {pair('swing_excursion_mean_m')} | {v['roll_pitch_rms']} | {v['completed']}/4 |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
