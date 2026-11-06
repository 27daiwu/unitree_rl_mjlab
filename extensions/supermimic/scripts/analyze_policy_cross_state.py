"""Summarize D3 frozen-policy interventions on paired restored simulator states."""
from __future__ import annotations

import csv
import json

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diag_policy_cross_state import CKPTS, DT, HERE, SOURCE_STARTS, STARTS


def gz(q):
    return -(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2))


def load(source):
    with np.load(HERE / f"cross_{source}.npz") as raw:
        d = {k: raw[k] for k in raw.files}
    d["meta"] = json.loads(str(d["meta"]))
    d["branches"] = [(int(x[0]), *x[1:4], int(x[4])) for x in d["branches"]]
    return d


def features(d, slot):
    s, ref = d["state"][:, slot], d["ref"][:, slot]
    contact = d["contact"][:, :, slot]
    foot = contact[:, :, :2].sum(-1)
    limits = d["limits"]
    delta = np.diff(d["target"][:, slot], axis=0, prepend=d["target"][:1, slot])
    vt = np.divide(contact[:, :, 3], foot, out=np.zeros_like(foot), where=foot > 20)
    loaded_vt = np.where(foot > 20, vt, 0)
    return {
        "z_abs_m": np.abs(s[:, 2] - ref[:, 2]),
        "tilt_gz_abs": np.abs(gz(s[:, 3:7]) - gz(ref[:, 3:7])),
        "joint_rmse_rad": np.sqrt(np.mean((s[:, 7:36] - ref[:, 7:36]) ** 2, axis=-1)),
        "target_tracking_rmse_rad": np.sqrt(np.mean((d["target"][:, slot] - s[:, 7:36]) ** 2, axis=-1)),
        "target_step_rms_rad": np.sqrt(np.mean(delta ** 2, axis=-1)),
        "action_rms": np.sqrt(np.mean(d["action"][:, slot] ** 2, axis=-1)),
        "torque_clipped_fraction": (np.abs(d["demand"][:, :, slot]) > limits).mean((1, 2)),
        "foot_force_N": foot.mean(-1),
        "other_force_N": contact[:, :, 2].mean(-1),
        "foot_low_load_fraction": (foot < .2 * 327.076603).mean(-1),
        "post_contact_vt_m_s": loaded_vt.mean(-1),
        "pre_contact_vt_m_s": np.divide(contact[:, :, 4], foot,
                                         out=np.zeros_like(foot), where=foot > 20).mean(-1),
        "post_contact_motion_fraction": np.divide(contact[:, :, 5], foot,
                                                    out=np.zeros_like(foot), where=foot > 20).mean(-1),
        "loaded_contact_path_m": np.cumsum(loaded_vt.sum(-1) * .005),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    datasets = {source: load(source) for source in CKPTS}
    rows, consistency, pair_rows, common_rows, joint_rows = [], [], [], [], []
    for source, d in datasets.items():
        saved = torch.load(HERE / f"source_{source}.pt", map_location="cpu", weights_only=False)
        starts = saved.get("starts", SOURCE_STARTS[source])
        for slot, name, _, policy, rep in d["branches"]:
            f = features(d, slot)
            event = d["meta"]["events"].get(str(slot))
            valid = int(event["step"] if event else len(d["state"]))
            row = {"motion": name, "source": source, "policy": policy, "repeat": rep,
                   "start_s": starts[name], "valid_s": valid * DT,
                   "failed": bool(event and event["failed"]),
                   "fail_time_s": None if not event or not event["failed"] else starts[name] + valid * DT,
                   "fail_reasons": "|".join(event["reasons"]) if event else ""}
            for horizon, n in (("h05", min(25, valid)), ("h20", valid)):
                for key, x in f.items():
                    row[f"{horizon}_{key}"] = float(x[n-1] if key == "loaded_contact_path_m" else x[:n].mean())
            rows.append(row)
            if policy == source:
                native = saved["native"][name]
                size = min(len(native), valid)
                errs = np.asarray([np.abs(d["state"][k, slot] - native[k]["state"]) for k in range(size)])
                action_err = np.asarray([np.abs(d["action"][k, slot] - native[k]["action"]) for k in range(size)])
                consistency.append({"motion": name, "source": source, "repeat": rep,
                                    "initial_state_max": float(errs[0].max()),
                                    "initial_action_max": float(action_err[0].max()),
                                    "state_max_0p5s": float(errs[:min(25, size)].max()),
                                    "state_max_full": float(errs.max()),
                                    "action_max_0p5s": float(action_err[:min(25, size)].max()),
                                    "action_max_full": float(action_err.max()),
                                    "native_event_step": saved["events"].get(name, {}).get("step"),
                                    "branch_event_step": event["step"] + saved["snapshots"][name]["step"] if event else None})
        for name in STARTS:
            two = [x for x in rows if x["motion"] == name and x["source"] == source]
            motion_branches = [b for b in d["branches"] if b[1] == name]
            common = min(d["meta"]["events"].get(str(b[0]), {}).get("step", len(d["state"]))
                         for b in motion_branches)
            for policy in CKPTS:
                three = [x for x in two if x["policy"] == policy]
                assert len(three) == 3
                relevant = [b[0] for b in motion_branches if b[3] == policy]
                feature_list = [features(d, slot) for slot in relevant]
                aligned = {key: [float(f[key][common-1] if key == "loaded_contact_path_m"
                                        else f[key][:common].mean()) for f in feature_list]
                           for key in feature_list[0]}
                common_rows.append({"motion": name, "source": source, "policy": policy,
                                    "common_s": common * DT, "failures": sum(x["failed"] for x in three),
                                    **{key: float(np.mean(values)) for key, values in aligned.items()},
                                    "z_repeat_range": float(np.ptp(aligned["z_abs_m"])),
                                    "tilt_repeat_range": float(np.ptp(aligned["tilt_gz_abs"])),
                                    "post_vt_repeat_range": float(np.ptp(aligned["post_contact_vt_m_s"]))})
                with np.load(HERE / f"source_trace_{source}.npz") as raw:
                    joint_names = json.loads(str(raw["meta_json"]))["joint_names"]
                for joint, joint_name in enumerate(joint_names):
                    clipped, tau_peak, response = [], [], []
                    for slot in relevant:
                        clipped.append(float((np.abs(d["demand"][:common, :, slot, joint]) >
                                              d["limits"][joint]).mean()))
                        tau_peak.append(float(np.abs(d["torque"][:common, :, slot, joint]).max()))
                        response.append(float(np.sqrt(np.mean((d["target"][:common, slot, joint] -
                                                                 d["state"][:common, slot, 7+joint]) ** 2))))
                    joint_rows.append({"motion": name, "source": source, "policy": policy,
                                       "common_s": common * DT, "joint": joint_name,
                                       "clipped_fraction": float(np.mean(clipped)),
                                       "torque_peak_Nm": float(np.mean(tau_peak)),
                                       "limit_Nm": float(d["limits"][joint]),
                                       "target_response_rmse_rad": float(np.mean(response))})
                pair_rows.append({"motion": name, "source": source, "policy": policy,
                                  "failures": sum(x["failed"] for x in three),
                                  "earliest_fail_s": min((x["fail_time_s"] for x in three if x["failed"]), default=None),
                                  "h05_z_mean": float(np.mean([x["h05_z_abs_m"] for x in three])),
                                  "h20_z_mean": float(np.mean([x["h20_z_abs_m"] for x in three])),
                                  "h20_tilt_mean": float(np.mean([x["h20_tilt_gz_abs"] for x in three])),
                                  "h20_joint_mean": float(np.mean([x["h20_joint_rmse_rad"] for x in three])),
                                  "h20_clipped_mean": float(np.mean([x["h20_torque_clipped_fraction"] for x in three])),
                                  "h20_foot_N": float(np.mean([x["h20_foot_force_N"] for x in three])),
                                  "h20_other_N": float(np.mean([x["h20_other_force_N"] for x in three])),
                                  "h20_post_vt_m_s": float(np.mean([x["h20_post_contact_vt_m_s"] for x in three])),
                                  "h20_loaded_path_m": float(np.mean([x["h20_loaded_contact_path_m"] for x in three])),
                                  "repeat_z_range": float(np.ptp([x["h20_z_abs_m"] for x in three])),
                                  "repeat_path_range": float(np.ptp([x["h20_loaded_contact_path_m"] for x in three]))})
    write_csv(HERE / "branches.csv", rows)
    write_csv(HERE / "restoration_consistency.csv", consistency)
    write_csv(HERE / "cross_summary.csv", pair_rows)
    write_csv(HERE / "common_window.csv", common_rows)
    write_csv(HERE / "joint_common.csv", joint_rows)
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    keys = ["z_abs_m", "tilt_gz_abs", "joint_rmse_rad", "torque_clipped_fraction",
            "foot_force_N", "other_force_N", "post_contact_vt_m_s", "loaded_contact_path_m"]
    for name in STARTS:
        fig, axes = plt.subplots(len(keys), 2, figsize=(13, 17), sharex=True)
        for col, source in enumerate(CKPTS):
            d = datasets[source]
            saved = torch.load(HERE / f"source_{source}.pt", map_location="cpu", weights_only=False)
            starts = saved.get("starts", SOURCE_STARTS[source])
            for slot, motion, _, policy, rep in d["branches"]:
                if motion != name:
                    continue
                f = features(d, slot)
                event = d["meta"]["events"].get(str(slot))
                valid = event["step"] if event else len(d["state"])
                t = starts[name] + np.arange(valid) * DT
                for row, key in enumerate(keys):
                    axes[row, col].plot(t, f[key][:valid], color="tab:blue" if policy == "38200" else "tab:orange",
                                        alpha=.55 if rep else .9, linewidth=.8, label=policy if rep == 0 else None)
                    if row == 0 and event and event["failed"]:
                        axes[row, col].axvline(starts[name] + valid * DT,
                                               color="tab:blue" if policy == "38200" else "tab:orange", alpha=.3)
            axes[0, col].set_title(f"{name} | source {source}")
            axes[0, col].legend()
            axes[-1, col].set_xlabel("absolute reference time [s]")
            for row, key in enumerate(keys):
                axes[row, col].grid(alpha=.25)
                if col == 0:
                    axes[row, col].set_ylabel(key)
        fig.tight_layout()
        fig.savefig(figdir / f"{name}_cross.png", dpi=140)
        plt.close(fig)
    print("saved", len(rows), "branches, consistency", len(consistency), "summaries", len(pair_rows))


if __name__ == "__main__":
    main()
