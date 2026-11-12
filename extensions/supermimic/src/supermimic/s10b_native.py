"""Actual S10A entrypoint's independent inference and watchdog supervision."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from supermimic.s10b import (
    InferencePending,
    IsolatedPolicy,
    policy_worker,
    record,
    stop_process,
    watchdog_worker,
)
from supermimic.s10b_recovery import support


class NativePolicy(IsolatedPolicy):
    retired = False
    owner = "STAND"
    recovery_requests = 0

    def __call__(self, obs, goal):
        if self.owner in ("RELEASE", "SAFE_DAMP", "STAND"):
            self.recovery_requests += 1
            raise RuntimeError("Policy request from inactive owner")
        record(
            self.log,
            "inference_requested",
            seq=self.seq + 1,
            state=self.owner,
            frame=self.frame,
            sim_time=self.sim_time,
        )
        return super().__call__(obs, goal)

    def accept(self, action):
        if self.retired and not self.tripped.is_set():
            self.late_results += 1
            record(self.log, "retired_action_discarded", seq=self.seq)
            return None
        return super().accept(action)


def run_native(runtime, args):
    from supermimic.s10a import CHECKPOINT, ROOT, Handoff, State, sha

    out = args.out
    args.dynamic_recovery = True
    h = Handoff(runtime, args)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    status = ctx.Array("d", [0.0, 0.0, 0.0], lock=True)
    tripped, stop, ready = ctx.Event(), ctx.Event(), ctx.Event()
    kind = getattr(args, "policy_fault", "none")
    deadline, delay = 0.1, 0.5
    infer = ctx.Process(
        target=policy_worker,
        args=(child, CHECKPOINT, kind, delay, out / "inference.jsonl"),
        name="native_s3_policy",
    )
    watch = ctx.Process(
        target=watchdog_worker,
        args=(status, tripped, stop, ready, deadline, out / "watchdog.jsonl"),
        name="native_watchdog",
    )
    infer.start()
    child.close()
    watch.start()
    proxy = NativePolicy(parent, infer, status, tripped, out / "control.jsonl")
    runtime.policy = proxy
    trip = pending_context = trigger = pending_brake = None
    pacing_wall = pacing_sim = None
    telemetry = []
    start = time.monotonic()
    fault_frame = math.ceil(len(runtime.ref) * 0.5)
    result = None
    try:
        if not parent.poll(30.0) or parent.recv().get("kind") != "ready" or not ready.wait(5.0):
            raise RuntimeError("Native inference/watchdog startup failed")
        record(
            out / "control.jsonl",
            "native_ready",
            inference_pid=infer.pid,
            watchdog_pid=watch.pid,
            watchdog_deadline_s=deadline,
            pending_brake_after_s=0.02,
        )
        while (
            runtime.data.time < args.max_sim_seconds
            and time.monotonic() - start < args.wall_seconds
        ):
            if not infer.is_alive() and h.state in (State.TAKEOVER, State.DANCE):
                runtime.refresh()
                h.release("inference_process_dead", fault=True)
                proxy.retired = True
                pending_context = None
            if tripped.is_set() and trip is None:
                trip = record(
                    out / "control.jsonl",
                    "watchdog_observed",
                    sim_time=float(runtime.data.time),
                    physics_steps=len(h.rows),
                    state=h.state.name,
                    inference_alive=infer.is_alive(),
                )
                if h.state in (State.TAKEOVER, State.DANCE):
                    runtime.refresh()
                    h.release("policy_watchdog_timeout", fault=True)
                proxy.retired = True
                pending_context = None
            if proxy.pending is not None:
                packet = proxy.pending.copy()
                try:
                    action = proxy.poll()
                except (EOFError, OSError) as error:
                    record(
                        out / "control.jsonl",
                        "inference_channel_failed",
                        error=type(error).__name__,
                    )
                    proxy.pending = None
                    action = None
                    if h.state in (State.TAKEOVER, State.DANCE):
                        runtime.refresh()
                        h.release("inference_channel_failed", fault=True)
                        proxy.retired = True
                        pending_context = None
                if (
                    action is not None
                    and pending_context is not None
                    and h.state in (State.TAKEOVER, State.DANCE)
                ):
                    if action.shape != (1, 29) or not np.isfinite(action).all():
                        raise ValueError("Invalid native inference output")
                    action = action[0]
                    runtime.previous = action.copy()
                    runtime.inferences += 1
                    runtime.interfaces.append(
                        (
                            runtime.data.time,
                            packet["frame"],
                            packet["obs"][0].copy(),
                            packet["goal"][0].copy(),
                            action.copy(),
                        )
                    )
                    h.policy_target = runtime.q0 + runtime.scale * action
                    if h.state == State.DANCE:
                        h.phase += 1
                    h.next_infer = max(
                        pending_context["next_infer"] + 0.02, float(runtime.data.time)
                    )
                    pending_context = None
                if (
                    proxy.pending is not None
                    and not proxy.retired
                    and time.monotonic() - packet["sent_monotonic"] >= 0.02
                ):
                    runtime.refresh()
                    h.release("policy_response_deadline", fault=True)
                    proxy.retired = True
                    pending_context = None
                    pending_brake = record(
                        out / "control.jsonl",
                        "independent_pending_brake",
                        sim_time=float(runtime.data.time),
                        physics_steps=len(h.rows),
                        request_age_s=time.monotonic() - packet["sent_monotonic"],
                        inference_alive=infer.is_alive(),
                    )
            if (
                kind != "none"
                and trigger is None
                and h.state == State.DANCE
                and h.phase >= fault_frame
            ):
                proxy.inject = True
                trigger = record(
                    out / "control.jsonl",
                    "fault_armed",
                    frame=h.phase,
                    sim_time=float(runtime.data.time),
                )
                pacing_wall, pacing_sim = time.monotonic(), float(runtime.data.time)
            proxy.frame = h.phase if h.state == State.DANCE else 0
            proxy.sim_time = float(runtime.data.time)
            proxy.owner = h.state.name
            next_infer = h.next_infer
            try:
                keep_running = h.tick()
            except InferencePending:
                record(
                    out / "control.jsonl",
                    "inference_pending",
                    seq=proxy.seq,
                    frame=proxy.frame,
                    sim_time=float(runtime.data.time),
                )
                pending_context = {"next_infer": next_infer}
                h.next_infer = float("inf")
                if pacing_wall is None:
                    pacing_wall, pacing_sim = time.monotonic(), float(runtime.data.time)
                continue
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- fault must retain independent physics ownership
                if h.state in (State.RELEASE, State.SAFE_DAMP):
                    h.fault = f"unrecoverable:{type(error).__name__}:{error}"
                    break
                runtime.refresh()
                h.release(f"{type(error).__name__}:{error}", fault=True)
                proxy.retired = True
                continue
            if not keep_running:
                break
            s = support(runtime)
            foot_velocity = []
            for foot in runtime.feet:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    runtime.model, runtime.diag, mujoco.mjtObj.mjOBJ_BODY, foot, velocity, 0
                )
                foot_velocity.append(velocity)
            actual_normal, actual_count = [0.0, 0.0], [0, 0]
            for i in range(runtime.data.ncon):
                c = runtime.data.contact[i]
                bodies = runtime.model.geom_bodyid[[c.geom1, c.geom2]]
                if not any(
                    runtime.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
                    for g in (c.geom1, c.geom2)
                ):
                    continue
                for side, foot in enumerate(runtime.feet):
                    if foot in bodies:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(runtime.model, runtime.data, i, force)
                        actual_normal[side] += force[0]
                        actual_count[side] += 1
            telemetry.append(
                (
                    runtime.data.time - 0.005,
                    actual_count,
                    actual_normal,
                    runtime.diag.xpos[runtime.feet].copy(),
                    runtime.diag.xmat[runtime.feet].copy(),
                    np.asarray(foot_velocity),
                    s["com"],
                    s["com_velocity"],
                    s["capture_margin"],
                    h.use_dynamic,
                )
            )
            if pacing_wall is not None:
                until = pacing_wall + runtime.data.time - pacing_sim - time.monotonic()
                if until > 0:
                    time.sleep(min(until, 0.005))
        end = record(
            out / "control.jsonl",
            "control_finished",
            sim_time=float(runtime.data.time),
            state=h.state.name,
            physics_steps=len(h.rows),
            inference_alive=infer.is_alive(),
        )
        result = h.save(time.monotonic() - start)
    finally:
        stop.set()
        watch.join(0.1)
        cleanup = {"inference": stop_process(infer), "watchdog": stop_process(watch)}
        parent.close()
        (out / "process_cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    keys = [
        "pre_step_time",
        "foot_contacts",
        "actual_normal_N",
        "foot_position",
        "foot_rotation",
        "foot_world_angular_linear_velocity",
        "com",
        "com_velocity",
        "capture_margin",
        "dynamic_recovery",
    ]
    np.savez_compressed(
        out / "contact_telemetry.npz",
        **{key: np.asarray([row[i] for row in telemetry]) for i, key in enumerate(keys)},
    )
    evidence = {
        "native_entrypoint": "python -m supermimic.s10a",
        "control_pid": os.getpid(),
        "inference_pid": infer.pid,
        "watchdog_pid": watch.pid,
        "watchdog_deadline_s": deadline,
        "fault_kind": kind,
        "fault_trigger": trigger,
        "fault_ack": proxy.fault_ack,
        "pending_brake": pending_brake,
        "watchdog_observed": trip,
        "control_end": end,
        "physics_steps_after_watchdog": len(h.rows) - trip["physics_steps"] if trip else 0,
        "late_results_discarded": proxy.late_results,
        "cleanup": cleanup,
        "recovery_policy_requests": proxy.recovery_requests,
        "result": result["release"],
    }
    (out / "native_watchdog.json").write_text(json.dumps(evidence, indent=2) + "\n")
    paths = [
        Path(__file__),
        Path(__file__).with_name("s10b_recovery.py"),
        Path(__file__).with_name("s10b.py"),
    ]
    (out / "native_source_sha256.json").write_text(
        json.dumps({str(p.relative_to(ROOT)): sha(p) for p in paths}, indent=2) + "\n"
    )
    return result
