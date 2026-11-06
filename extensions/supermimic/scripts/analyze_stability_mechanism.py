"""Offline paired telemetry/WM analysis; never creates or mutates an environment."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supermimic.kinematics.rotations import projected_gravity
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

HERE = Path("reports/phase2b_d2_20261030")
CKPTS = {
    "38200": "logs/phase2a_anchor_4090_s0_20261030/ckpt_latest.pt",
    "46200": "logs/phase2b_anchor_4090_s0_38200_to_46200/ckpt_0046200.pt",
}


def load(label):
    with np.load(HERE / f"trace_{label}.npz") as raw:
        d = {k: raw[k] for k in raw.files}
    d["meta"] = json.loads(str(d["meta_json"]))
    return d


def gz(q):
    return -(1 - 2 * (q[..., 1]**2 + q[..., 2]**2))


def features(d, i):
    s, ref = d["state"][:, i], d["ref"][:, i]
    c = d["contact"][:, :, i]
    tau, demand = d["torque"][:, :, i], d["torque_unclipped"][:, :, i]
    lim = np.max(np.abs(d["force_range"]), axis=-1)
    names = d["meta"]["joint_names"]
    legs = [j for j, name in enumerate(names) if any(p in name for p in ("hip", "knee", "ankle"))]
    target = d["target"][:, i]
    delta = np.concatenate([np.zeros((1, 29)), np.diff(target, axis=0)])
    force = c[:, :, :2].sum(-1)
    slip_force = c[:, :, 5:7].sum(-1)
    limits = d["soft_limits"]
    return {
        "z_error": np.abs(s[:, 2] - ref[:, 2]),
        "tilt_error": np.abs(gz(s[:, 3:7]) - gz(ref[:, 3:7])),
        "joint_rmse": np.sqrt(np.mean((s[:, 7:36] - ref[:, 7:36])**2, -1)),
        "target_tracking_rmse": np.sqrt(np.mean((target - s[:, 7:36])**2, -1)),
        "target_ref_rmse": np.sqrt(np.mean((target - ref[:, 7:36])**2, -1)),
        "target_step_rms": np.sqrt(np.mean(delta**2, -1)),
        "raw_action_step_rms": np.sqrt(np.mean(np.diff(d["action"][:, i], axis=0,
                                                     prepend=d["action"][:1, i])**2, -1)),
        "saturation_fraction": (np.abs(tau) >= .95 * lim).mean((1, 2)),
        "leg_saturation_fraction": (np.abs(tau[:, :, legs]) >= .95 * lim[legs]).mean((1, 2)),
        "clipped_demand_fraction": (np.abs(demand) > lim).mean((1, 2)),
        "target_outside_soft_limit_fraction": ((target < limits[:, 0]) | (target > limits[:, 1])).mean(-1),
        "foot_force_N": force.mean(-1),
        "other_ground_force_N": c[:, :, 2].mean(-1),
        "low_support_fraction": (force < .2 * d["meta"]["weight_N"]).mean(-1),
        "no_foot_load_fraction": (force < 20).mean(-1),
        "loaded_slip_fraction": ((slip_force > .2 * force) & (force > 20)).mean(-1),
        "weighted_slip_speed": ((c[:, :, 3:5] * c[:, :, :2]).sum(-1) / force.clip(1e-8)).mean(-1),
    }


def first_sustained(signal, threshold, steps=3):
    above = signal > threshold
    for j in range(max(0, len(above) - steps + 1)):
        if above[j:j+steps].all():
            return j * .02
    return None


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


@torch.no_grad()
def wm_errors(model_label, d, i, stop):
    ck = torch.load(CKPTS[model_label], map_location="cpu", weights_only=False)
    config = dict(ck["config"]["world_model"])
    config["state_spec"] = StateSpecCfg(**config["state_spec"])
    wm = WorldModel(WorldModelCfg(**config), None, "cuda:0")
    wm.load_state_dict(ck["wm"], strict=True)
    wm.eval()
    start = max(0, stop - 100)
    outputs = {}
    for horizon in (1, 8):
        starts = np.arange(start, stop - horizon + 1)
        if not len(starts):
            continue
        pred = torch.tensor(d["state"][starts, i], device="cuda:0")
        xi = torch.tensor(d["xi"][i], device="cuda:0").expand(len(starts), -1)
        for k in range(horizon):
            act = torch.tensor(d["action"][starts+k, i], device="cuda:0")
            pred = wm.step(pred, act, xi)
        truth = torch.tensor(d["next_state"][starts+horizon-1, i], device="cuda:0")
        result = torch.stack([
            pred[:, 2] - truth[:, 2],
            projected_gravity(pred[:, 3:7])[:, 2] - projected_gravity(truth[:, 3:7])[:, 2],
            (pred[:, 7:36] - truth[:, 7:36]).square().mean(-1).sqrt(),
            wm.spec.normalized_sq_error(pred, truth),
        ], -1).cpu().numpy()
        outputs[horizon] = {"z_mae_m": float(np.abs(result[:, 0]).mean()),
                            "z_bias_m": float(result[:, 0].mean()),
                            "gravity_z_mae": float(np.abs(result[:, 1]).mean()),
                            "joint_rmse_rad": float(result[:, 2].mean()), "nse": float(result[:, 3].mean())}
    del wm, ck
    return outputs


def render(name, datasets, start, stop, tag):
    fig, axes = plt.subplots(8, 2, figsize=(14, 17), sharex=True)
    for col, (label, d) in enumerate(datasets.items()):
        i = list(d["clips"]).index(name)
        end = min(stop, d["meta"]["events"][name]["step"])
        if end <= start:
            axes[0, col].set_title(f"{label}: already terminated at {end*.02:.2f}s")
            continue
        sl = slice(start, end)
        t = np.arange(start, end) * .02
        x = features(d, i)
        s, ref, target = d["state"][sl, i], d["ref"][sl, i], d["target"][sl, i]
        axes[0, col].set_title(f"{name} | {label} | {tag}\nSame absolute reference stage; masked after first termination")
        axes[0, col].plot(t, ref[:, 2], label="ref pelvis z")
        axes[0, col].plot(t, s[:, 2], label="robot pelvis z")
        axes[1, col].plot(t, x["tilt_error"][sl], label="gravity-z error")
        axes[1, col].axhline(.8, color="r", linestyle="--")
        axes[2, col].plot(t, x["target_step_rms"][sl], label="target increment RMS [rad]")
        axes[2, col].plot(t, x["target_ref_rmse"][sl], label="target vs ref [rad]")
        axes[2, col].plot(t, x["joint_rmse"][sl], label="actual vs ref [rad]")
        j = d["meta"]["joint_names"].index("left_knee_joint")
        axes[3, col].plot(t, target[:, j], label="left knee target")
        axes[3, col].plot(t, s[:, 7+j], label="left knee actual")
        axes[3, col].plot(t, ref[:, 7+j], label="left knee reference")
        axes[4, col].plot(t, x["leg_saturation_fraction"][sl], label="leg torque >=95% limit")
        axes[4, col].plot(t, x["clipped_demand_fraction"][sl], label="demand clipped (all joints)")
        axes[5, col].plot(t, x["foot_force_N"][sl], label="left + right normal force")
        axes[5, col].plot(t, x["other_ground_force_N"][sl], label="non-foot ground normal force")
        axes[5, col].axhline(d["meta"]["weight_N"], color="gray", linestyle="--", label="body weight")
        axes[6, col].plot(t, x["weighted_slip_speed"][sl], label="force-weighted tangential speed [m/s]")
        axes[6, col].plot(t, x["loaded_slip_fraction"][sl], label="loaded slip fraction (>0.1 m/s)")
        axes[7, col].plot(t, x["low_support_fraction"][sl], label="substeps foot load <20% BW")
        axes[7, col].plot(t, x["target_outside_soft_limit_fraction"][sl], label="targets outside soft limits")
        for ax in axes[:, col]:
            ax.grid(alpha=.25); ax.legend(fontsize=7)
        axes[-1, col].set_xlabel("reference time [s]")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / f"{name}_{tag}.png", dpi=130)
    plt.close(fig)


def main():
    primary = {k: load(k) for k in CKPTS}
    repeat = load("46200_repeat")
    (HERE / "figures").mkdir(exist_ok=True)
    rows, wm_rows, onset_rows, joint_rows = [], [], [], []
    fixed = [138., 44.2, 29.6, 14.9, 14.1]
    names = list(primary["46200"]["clips"])
    for i, name in enumerate(names[:5]):
        datasets = {"38200": primary["38200"],
                    "46200": repeat if name == "fightAndSports1_subject4" else primary["46200"]}
        e = datasets["46200"]["meta"]["events"][name]
        endpoints = {"d1_phase": round(fixed[i] / .02)}
        if e["failed"]:
            endpoints["d2_failure"] = e["step"]
        for tag, stop in endpoints.items():
            start = max(0, stop - 100)
            render(name, datasets, start, stop, tag)
            for label, d in datasets.items():
                control = "walk1_subject2" if stop <= datasets["38200"]["meta"]["events"]["walk1_subject2"]["step"] else "fightAndSports1_subject4"
                for subject in dict.fromkeys((name, control)):
                    idx = list(d["clips"]).index(subject)
                    available = d["meta"]["events"][subject]["step"]
                    if available < stop:
                        rows.append({"case": name, "phase": tag, "trace": label, "motion": subject,
                                     "end_s": stop*.02, "available": False,
                                     **{k: None for k in features(d, idx)}})
                        continue
                    f = features(d, idx)
                    rows.append({"case": name, "phase": tag, "trace": label, "motion": subject,
                                 "end_s": stop*.02, "available": True,
                                 **{k: float(v[start:stop].mean()) for k, v in f.items()}})
                    if tag == "d2_failure" or (tag == "d1_phase" and not e["failed"]):
                        for model in CKPTS:
                            for horizon, result in wm_errors(model, d, idx, stop).items():
                                wm_rows.append({"case": name, "trace": label, "motion": subject,
                                                "model": model, "horizon": horizon, "end_s": stop*.02, **result})
                    if subject != name:
                        continue
                    # Operational alarms, not causal labels. Use same-clip earlier baseline
                    # for output changes; fixed physical thresholds for load and root errors.
                    baseline = slice(max(0, start-300), max(1, start-100))
                    thresholds = {"target_step_rms": max(.05, float(np.quantile(f["target_step_rms"][baseline], .99))),
                                  "target_tracking_rmse": max(.3, float(np.quantile(f["target_tracking_rmse"][baseline], .99))),
                                  "leg_saturation_fraction": .1, "low_support_fraction": .75,
                                  "loaded_slip_fraction": .75, "z_error": .1, "tilt_error": .25}
                    for key, threshold in thresholds.items():
                        onset = first_sustained(f[key][start:stop], threshold)
                        onset_rows.append({"case": name, "phase": tag, "trace": label, "signal": key,
                                           "threshold": threshold, "onset_s": None if onset is None else start*.02+onset,
                                           "left_censored": onset == 0})
                    limit = np.max(np.abs(d["force_range"]), -1)
                    for j, joint in enumerate(d["meta"]["joint_names"]):
                        clipped = (np.abs(d["torque_unclipped"][start:stop, :, idx, j]) > limit[j]).reshape(-1)
                        sustained = next((k for k in range(len(clipped)-2) if clipped[k:k+3].all()), None)
                        joint_rows.append({"case": name, "phase": tag, "trace": label, "joint": joint,
                            "torque_peak_Nm": float(np.abs(d["torque"][start:stop, :, idx, j]).max()),
                            "limit_Nm": float(limit[j]),
                            "clipped_fraction": float((np.abs(d["torque_unclipped"][start:stop, :, idx, j])>limit[j]).mean()),
                            "first_15ms_clip_interval_start_s": None if sustained is None else start*.02+sustained*.005,
                            "target_actual_rmse_rad": float(np.sqrt(np.mean((d["target"][start:stop,idx,j]-d["state"][start:stop,idx,7+j])**2))),
                            "actual_ref_rmse_rad": float(np.sqrt(np.mean((d["ref"][start:stop,idx,7+j]-d["state"][start:stop,idx,7+j])**2)))})
    write_csv(HERE / "paired_windows.csv", rows)
    write_csv(HERE / "wm_crosscheck.csv", wm_rows)
    write_csv(HERE / "onsets.csv", onset_rows)
    write_csv(HERE / "joint_windows.csv", joint_rows)
    success_rows = []
    for case, label in (("fightAndSports1_subject4", "46200"), ("fightAndSports1_subject1", "46200_repeat")):
        d = primary["46200"] if label == "46200" else repeat
        idx = names.index(case)
        stop = round((29.6 if case.endswith("subject4") else 44.8)/.02)
        success_rows.append({"case": case, "trace": label, "end_s": stop*.02,
                             **{k: float(v[stop-100:stop].mean()) for k, v in features(d, idx).items()}})
    write_csv(HERE / "same_checkpoint_success_windows.csv", success_rows)
    print("saved paired windows", len(rows), "WM comparisons", len(wm_rows))


if __name__ == "__main__":
    main()
