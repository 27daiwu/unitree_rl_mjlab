"""Read-only paired analysis of W1.6 fixed-command traces."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np

from diagnose_flat_gait import COMMANDS, _rotate_quat, _segments


ROOT = Path("logs/flat_w1_6_toe_clearance_20261030")
DATA = ROOT / "paired_diagnostics"
FEET = ("left", "right")


def capsule_points():
    root = ET.parse("src/assets/robots/unitree_g1/xmls/g1.xml").getroot()
    radius = float(root.find(".//default[@class='foot_capsule']/geom").get("size").split()[0])
    geometry = {}
    for side in FEET:
        body = next(b for b in root.iter("body")
                    if any(s.get("name") == f"{side}_foot" for s in b.findall("site")))
        site = np.fromstring(next(s.get("pos") for s in body.findall("site")
                                  if s.get("name") == f"{side}_foot"), sep=" ")
        toe, heel = [], []
        for geom in body.findall("geom"):
            if not geom.get("name", "").startswith(f"{side}_foot") or not geom.get("name", "").endswith("_collision"):
                continue
            for point in np.fromstring(geom.get("fromto"), sep=" ").reshape(2, 3) - site:
                if point[0] > 0.05:
                    toe.append(point)
                if point[0] < -0.03:
                    heel.append(point)
        geometry[side] = (np.asarray(toe), np.asarray(heel))
    return geometry, radius


def trace_row(path, geometry, radius, foot_index, foot):
    with np.load(path) as z:
        t = z["time"]
        assert len(t) == 600
        keep = t > 7.0
        t = t[keep]
        pos = z["foot_position"][keep, foot_index]
        quat = z["foot_quaternion"][keep, foot_index]
        contact = ((z["contact_time"][keep, foot_index] > 0) |
                   (z["normal_force"][keep, foot_index] >= 5))
        toe_points, heel_points = geometry[foot]
        toe = (pos[:, None, 2] + _rotate_quat(quat[:, None, :], toe_points[None])[:, :, 2]).min(axis=1) - radius
        heel = (pos[:, None, 2] + _rotate_quat(quat[:, None, :], heel_points[None])[:, :, 2]).min(axis=1) - radius
        phase = (np.rint(t / 0.02) / 30 + (0.0, 0.5)[foot_index]) % 1.0
        mid = (phase > 0.66 + 1e-6) & (phase < 0.90 - 1e-6)
        events = []
        for start, end in _segments(~contact):
            if end - start < 2:
                continue
            peak = start + int(np.argmax(toe[start:end]))
            events.append((toe[peak], heel[peak], pos[peak, 2]))
        return {
            "mid_swing_toe_mean_m": float(toe[mid].mean()),
            "mid_swing_toe_p10_m": float(np.percentile(toe[mid], 10)),
            "mid_swing_toe_target_fraction": float(np.mean(toe[mid] >= 0.03)),
            "mid_swing_contact_fraction": float(np.mean(contact[mid])),
            "swing_toe_peak_mean_m": float(np.mean([e[0] for e in events])),
            "heel_at_toe_peak_mean_m": float(np.mean([e[1] for e in events])),
            "site_at_toe_peak_mean_m": float(np.mean([e[2] for e in events])),
            "toe_peak_events": len(events),
            "stance_toe_p05_m": float(np.percentile(toe[contact], 5)),
        }


def main():
    geometry, radius = capsule_points()
    rows = {}
    for scenario in COMMANDS:
        rows[scenario] = {}
        for label in ("control", "experiment"):
            feet = {}
            for index, foot in enumerate(FEET):
                traces = [trace_row(DATA / f"{label}_{scenario}_{replica}.npz", geometry, radius, index, foot)
                          for replica in range(4)]
                feet[foot] = {key: float(np.mean([trace[key] for trace in traces]))
                              for key in traces[0] if key != "toe_peak_events"}
                feet[foot]["toe_peak_events"] = sum(trace["toe_peak_events"] for trace in traces)
            rows[scenario][label] = feet
    path = ROOT / "exact_toe_metrics.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
