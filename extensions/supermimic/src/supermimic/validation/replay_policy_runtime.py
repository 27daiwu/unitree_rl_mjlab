"""P1 offline golden replay. MuJoCo is a static FK oracle only in this validator.

No Scene, control loop, adapter or physical integration is constructed. The new
runtime itself is tested separately with simulator imports actively forbidden.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.policy.runtime import PolicyRuntime, RobotSnapshot

ROOT = Path(__file__).resolve().parents[5]
REPORTS = ROOT / "extensions/supermimic/reports"
CASES = ["phase2_s10a_handoff_20261110/final_full_01"] + [
    f"phase2_s10b_dynamic_recovery_20261110/formal_v1/{name}"
    for name in (
        "full_01",
        "full_02",
        "full_03",
        "abort_25",
        "abort_50",
        "abort_75",
        "watchdog_timeout",
        "watchdog_permanent",
    )
]


def check_equal(actual, expected, label, metrics):
    error = float(np.max(np.abs(np.asarray(actual) - expected)))
    metrics[label] = max(metrics.get(label, 0.0), error)
    if not np.array_equal(actual, expected):
        raise AssertionError(f"{label}: exact equality failed, max_abs={error}")


def replay(runtime, relative, limit=None):
    folder = REPORTS / relative
    with np.load(folder / "policy.npz") as saved, np.load(folder / "trace.npz") as trace:
        policy = {key: saved[key] for key in saved.files}
        states = {key: trace[key] for key in trace.files}
    activation = json.loads((folder / "activation.json").read_text())
    events = json.loads((folder / "events.json").read_text())
    dance_start = next(event["time"] for event in events if event["to"] == "DANCE")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "src/assets/robots/unitree_g1/xmls/g1.xml"))
    data = mujoco.MjData(model)
    anchor = model.body("torso_link").id
    session = runtime.session(relative, clock_domain="archived_sim_time", max_snapshot_age_s=0.005)
    metrics = {}
    count = len(policy["time"]) if limit is None else min(limit, len(policy["time"]))
    stages, frames = [], []
    first_snapshot = None
    started = time.monotonic()
    for i in range(count):
        now = float(policy["time"][i])
        row = int(np.searchsorted(states["time"], now))
        if row >= len(states["time"]) or abs(states["time"][row] - now) > 1e-10:
            raise AssertionError("No matching archived pre-inference state")
        data.qpos[:] = states["qpos"][row]
        mujoco.mj_kinematics(model, data)
        snapshot = RobotSnapshot(
            f"{relative}:{i}",
            "archived_sim_time",
            now,
            runtime.joint_names,
            states["qpos"][row, 7:],
            states["qvel"][row, 6:],
            states["qpos"][row, 3:7],
            states["qvel"][row, 3:6],
            data.xpos[anchor].copy(),
            data.xquat[anchor].copy(),
        )
        if i == 0:
            # Activation precedes the first infer by one archived 5 ms step;
            # reading the first inference target would include a later update.
            activation_row = row - 1
            activation_time = float(states["time"][activation_row])
            data.qpos[:] = states["qpos"][activation_row]
            mujoco.mj_kinematics(model, data)
            first_snapshot = RobotSnapshot(
                f"{relative}:activation",
                "archived_sim_time",
                activation_time,
                runtime.joint_names,
                states["qpos"][activation_row, 7:],
                states["qvel"][activation_row, 6:],
                states["qpos"][activation_row, 3:7],
                states["qvel"][activation_row, 3:6],
                data.xpos[anchor].copy(),
                data.xquat[anchor].copy(),
            )
            session.activate(
                first_snapshot, states["target"][activation_row], epoch=1, now=activation_time
            )
            check_equal(
                session.previous, activation["previous_action"], "activation_previous", metrics
            )
            check_equal(
                session.alignment["translation"],
                activation["translation"],
                "anchor_alignment",
                metrics,
            )
            check_equal(session.alignment["yaw_rad"], activation["yaw_rad"], "anchor_yaw", metrics)
        if now >= dance_start and session.stage == "WARMUP":
            session.begin_dance()
        request = session.build_request(snapshot, now=now)
        if request.frame != int(policy["frame"][i]):
            raise AssertionError("Warmup/DANCE frame mismatch")
        check_equal(request.obs, policy["obs"][i], "obs", metrics)
        check_equal(request.goal, policy["goal"][i], "goal", metrics)
        check_equal(request.goal[-9:], policy["goal"][i, -9:], "anchor", metrics)
        # A logical clock isolates numerical equivalence from host load. Real
        # elapsed time is recorded separately, without claiming a 20 ms deadline.
        result = runtime.infer(
            request, clock=lambda at=now: at + 0.001, clock_domain="archived_sim_time"
        )
        check_equal(result.raw_action, policy["action"][i], "raw_action", metrics)
        accepted = session.accept(result, now=now + 0.001)
        expected_q = runtime.q0 + runtime.action_scale * policy["action"][i]
        check_equal(accepted.q_policy, expected_q, "q_policy", metrics)
        if request.stage == "DANCE":
            # The next archived physical step carries the unblended policy
            # target. It is an independent recorded output, not just a formula.
            check_equal(
                accepted.q_policy,
                states["desired_target"][row + 1],
                "archived_desired_target",
                metrics,
            )
        check_equal(session.previous, policy["action"][i], "previous", metrics)
        # Validate the last slot of each term independently against source state.
        chunks = np.split(request.obs, [290, 580, 610, 640])
        check_equal(
            chunks[0].reshape(10, 29)[-1],
            (snapshot.q - runtime.q0).astype(np.float32),
            "joint_term",
            metrics,
        )
        check_equal(
            chunks[1].reshape(10, 29)[-1], snapshot.dq.astype(np.float32), "velocity_term", metrics
        )
        stages.append(request.stage)
        frames.append(request.frame)
    dance_frames = [frame for frame, stage in zip(frames, stages) if stage == "DANCE"]
    if dance_frames != list(range(len(dance_frames))):
        raise AssertionError("Missing/repeated DANCE phase")
    full = count == len(policy["time"]) and len(dance_frames) == 6574
    if session.clip_complete != full:
        raise AssertionError("Incorrect clip completion status")
    full_history = session.history
    intervals = np.diff(policy["time"][:count])
    return {
        "case": relative,
        "result": "PASS",
        "requests_compared": count,
        "warmup_count": session.warmup_count,
        "dance_frames": len(dance_frames),
        "clip_complete": session.clip_complete,
        "exact_max_abs_error": metrics,
        "wall_elapsed_s": time.monotonic() - started,
        "history_shape": list(full_history.shape),
        "first_snapshot_time": first_snapshot.measured_at,
        "scope": "offline_equivalence_only_not_physical_recovery",
        "physics_steps": 0,
        "policy_time_intervals_s_rounded": {
            str(value): int(np.sum(np.isclose(intervals, value, atol=1e-9, rtol=0)))
            for value in np.unique(np.round(intervals, 9))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", nargs="*", default=CASES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    runtime = PolicyRuntime(ROOT)
    results = []
    for case in args.cases:
        try:
            result = replay(runtime, case, args.limit)
        except Exception as error:
            result = {"case": case, "result": "FAIL", "error": f"{type(error).__name__}: {error}"}
            results.append(result)
            (args.out / "numerical_equivalence.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            raise
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.out / "numerical_equivalence.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
