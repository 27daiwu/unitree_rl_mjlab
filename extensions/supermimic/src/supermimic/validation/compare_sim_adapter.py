"""P2 read-only comparison against sealed legacy MuJoCo runs.

Thresholds are read from the pre-run plan. Physical FAIL and equivalence PASS
are independent outputs; this comparator never writes to the baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(folder, name):
    return json.loads((folder / name).read_text())


def arrays(folder, filename):
    with np.load(folder / filename, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def difference(a, b):
    count = min(len(a), len(b))
    if a.ndim != b.ndim or a.shape[1:] != b.shape[1:]:
        return {"equal": False, "shapes": [list(a.shape), list(b.shape)]}
    bad = a[:count] != b[:count]
    first = np.flatnonzero(bad.reshape(count, -1).any(axis=1)) if count else []
    return {
        "equal": bool(np.array_equal(a, b)),
        "shapes": [list(a.shape), list(b.shape)],
        "max_abs_common": float(
            np.max(np.abs(a[:count].astype(float) - b[:count].astype(float)), initial=0)
        ),
        "first_different_row": int(first[0])
        if len(first)
        else (count if len(a) != len(b) else None),
    }


def compare(baseline: Path, candidate: Path, plan: dict, fault=False):
    old, new = read(baseline, "result.json"), read(candidate, "result.json")
    z0, z1 = arrays(baseline, "trace.npz"), arrays(candidate, "trace.npz")
    p0, p1 = arrays(baseline, "policy.npz"), arrays(candidate, "policy.npz")
    metrics = {
        "policy": {k: difference(p0[k], p1[k]) for k in p0},
        "trace": {k: difference(z0[k], z1[k]) for k in z0},
    }
    for filename in ("contact_telemetry.npz", "recovery.npz"):
        if (baseline / filename).exists() and (candidate / filename).exists():
            a, b = arrays(baseline, filename), arrays(candidate, filename)
            metrics[filename] = {
                k: difference(a[k], b[k]) for k in a if a[k].dtype.kind not in "US"
            }
    e0, e1 = read(baseline, "events.json"), read(candidate, "events.json")
    checks = {
        "activation_exact": read(baseline, "activation.json") == read(candidate, "activation.json"),
        "asset_mapping_exact": read(baseline, "asset_audit.json")
        == read(candidate, "asset_audit.json"),
        "classification_same": all(
            old[k] == new[k]
            for k in (
                "initial_stand",
                "takeover",
                "dance",
                "release",
                "full_chain",
                "final_state",
                "bounds_ok",
                "dance_frames_executed",
                "policy_inferences",
            )
        ),
        "policy_exact": all(v["equal"] for v in metrics["policy"].values()),
        "events_exact": e0 == e1,
        "stop_events_exact": read(baseline, "stop_events.json")
        == read(candidate, "stop_events.json"),
        "trace_exact": all(v["equal"] for v in metrics["trace"].values()),
        "telemetry_exact": all(
            v["equal"]
            for k in ("contact_telemetry.npz", "recovery.npz")
            for v in metrics.get(k, {}).values()
        ),
    }
    w0, w1 = read(baseline, "native_watchdog.json"), read(candidate, "native_watchdog.json")
    checks["fault_isolation_same"] = (
        all(
            w0[k] == w1[k]
            for k in ("fault_kind", "recovery_policy_requests", "late_results_discarded")
        )
        and w1["recovery_policy_requests"] == 0
    )
    checks["process_cleanup"] = all(not v["alive"] for v in w1["cleanup"].values())
    session = read(candidate, "shared_session.json")
    accepted = [r for r in session["requests"] if r["accepted"]]
    dance = [r["frame"] for r in accepted if r["stage"] == "DANCE"]
    checks["shared_session_phase"] = (
        dance == list(range(new["dance_frames_executed"]))
        and len(accepted) == new["policy_inferences"]
        and session["warmup_count"] == 11
    )
    checks["original_physics_no_state_writes"] = bool(
        np.array_equal(z1["pre_qpos"][1:], z1["qpos"][:-1])
    )
    release = next(e for e in e1 if e["to"] == "RELEASE")
    requests = [json.loads(line) for line in (candidate / "control.jsonl").read_text().splitlines()]
    checks["no_recovery_policy_requests"] = not any(
        r["event"] == "inference_requested" and r["sim_time"] > release["time"] + 1e-9
        for r in requests
    )
    details = {}
    if fault:
        release0 = next(e for e in e0 if e["to"] == "RELEASE")
        delta = release["time"] - release0["time"]
        until = min(w0["fault_trigger"]["sim_time"], w1["fault_trigger"]["sim_time"])
        prefix0, prefix1 = z0["time"] <= until, z1["time"] <= until
        checks["fault_prefix_exact"] = all(
            np.array_equal(z0[k][prefix0], z1[k][prefix1]) for k in z0
        )
        checks["release_within_frozen_jitter"] = (
            abs(delta) <= plan["tolerances"]["fault_release_sim_time_difference_s"] + 1e-10
        )
        checks["same_event_identity"] = [
            {k: e[k] for k in ("from", "to", "reason", "phase_frame")} for e in e0
        ] == [{k: e[k] for k in ("from", "to", "reason", "phase_frame")} for e in e1]
        checks["pointwise_if_release_state_equal"] = abs(delta) > 1e-10 or checks["trace_exact"]
        lines = [
            json.loads(line) for line in (candidate / "watchdog.jsonl").read_text().splitlines()
        ]
        expired = next(line for line in lines if line["event"] == "deadline_expired")
        checks["independent_fault_supervision"] = (
            len({w1[k] for k in ("control_pid", "inference_pid", "watchdog_pid")}) == 3
            and expired["elapsed_s"] <= 0.12
            and w1["pending_brake"]["request_age_s"] <= 0.035
            and w1["physics_steps_after_watchdog"] > 0
            and w1["control_end"]["inference_alive"]
            and (
                w1["late_results_discarded"] >= 1
                if w1["fault_kind"] == "timeout"
                else w1["late_results_discarded"] == 0
            )
        )
        details = {
            "release_time_difference_s": delta,
            "watchdog_detector_s": expired["elapsed_s"],
            "pending_brake_s": w1["pending_brake"]["request_age_s"],
            "post_release_pointwise_equal": checks["trace_exact"],
            "physics_steps_after_watchdog": w1["physics_steps_after_watchdog"],
        }
        required = [
            k for k in checks if k not in ("events_exact", "trace_exact", "telemetry_exact")
        ]
    else:
        required = list(checks)
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "fault": fault,
        "equivalence": "PASS" if all(checks[k] for k in required) else "FAIL",
        "required_checks": required,
        "checks": checks,
        "differences": metrics,
        "fault_details": details,
        "physical": {
            k: new[k]
            for k in (
                "initial_stand",
                "takeover",
                "dance",
                "release",
                "full_chain",
                "final_state",
                "dance_frames_executed",
                "physics_steps",
            )
        },
        "max_release_servo_step_rad": new["stages"]["RELEASE"]["max_servo_ctrl_step_rad"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fault", action="store_true")
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, json.loads(args.plan.read_text()), args.fault)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "equivalence": result["equivalence"],
                "failed_checks": [k for k in result["required_checks"] if not result["checks"][k]],
                "physical": result["physical"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["equivalence"] == "PASS" else 1)


if __name__ == "__main__":
    main()
