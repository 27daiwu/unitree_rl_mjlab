"""Two-process local DDS deployment for the existing S10 Handoff and S3 runtime."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.dds_runtime import DdsSharedDanceRuntime
from supermimic.dds_transport import DdsTransport
from supermimic.s10a import CHECKPOINT, ROOT, Handoff, State, sha
from supermimic.s10b import InferencePending, policy_worker, record, stop_process, watchdog_worker
from supermimic.s10b_native import NativePolicy


def _keyboard_listener(signal_file: Path, log_path: Path):
    import sys

    for line in sys.stdin:
        if line.strip().lower() in ("", "d", "dance"):
            signal_file.touch()
            record(log_path, "keyboard_dance_trigger", signal_file=str(signal_file))
            print(f"Dance trigger accepted: {signal_file}", flush=True)
            return


def run_dds(runtime, args, transport):
    out = args.out
    args.dynamic_recovery = False  # Contact points/forces are unavailable on this bridge.
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
        name="dds_s3_policy",
    )
    watch = ctx.Process(
        target=watchdog_worker,
        args=(status, tripped, stop, ready, deadline, out / "watchdog.jsonl"),
        name="dds_watchdog",
    )
    infer.start()
    child.close()
    watch.start()
    proxy = NativePolicy(parent, infer, status, tripped, out / "control.jsonl")
    runtime.policy = proxy
    trip = pending_context = trigger = pending_brake = None
    result = None
    last_initial_stable = False
    start = time.monotonic()
    start_sim_time = float(runtime.data.time)
    end_sim_time = start_sim_time + args.max_sim_seconds
    original_step = mujoco.mj_step
    mujoco.mj_step = runtime.external_step
    try:
        if not parent.poll(30.0) or parent.recv().get("kind") != "ready" or not ready.wait(5.0):
            raise RuntimeError("DDS inference/watchdog startup failed")
        record(
            out / "control.jsonl",
            "dds_ready",
            inference_pid=infer.pid,
            watchdog_pid=watch.pid,
            watchdog_deadline_s=deadline,
            pending_brake_after_s=0.02,
            backend="LOCAL_DDS_SIM_PHYSICS",
        )
        while runtime.data.time < end_sim_time and time.monotonic() - start < args.wall_seconds:
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
                and h.phase >= math.ceil(len(runtime.ref) * 0.5)
            ):
                proxy.inject = True
                trigger = record(
                    out / "control.jsonl",
                    "fault_armed",
                    frame=h.phase,
                    sim_time=float(runtime.data.time),
                )
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
                    sim_time=proxy.sim_time,
                )
                pending_context = {"next_infer": next_infer}
                h.next_infer = float("inf")
                continue
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- retain independent fault ownership
                if h.state in (State.RELEASE, State.SAFE_DAMP):
                    h.fault = f"unrecoverable:{type(error).__name__}:{error}"
                    break
                runtime.refresh()
                h.release(f"{type(error).__name__}:{error}", fault=True)
                proxy.retired = True
                continue
            if not keep_running:
                break
            if (
                args.auto_trigger_after_stable
                and args.arm_file is not None
                and args.arm_file.exists()
                and h.initial_stable
                and not args.signal_file.exists()
            ):
                args.signal_file.touch()
                record(
                    out / "control.jsonl",
                    "auto_dance_trigger",
                    sim_time=float(runtime.data.time),
                    arm_file=str(args.arm_file),
                )
                print("Dance trigger accepted after measured stable stand", flush=True)
            if h.initial_stable != last_initial_stable:
                record(
                    out / "control.jsonl",
                    "initial_stand_stable" if h.initial_stable else "initial_stand_unstable",
                    sim_time=float(runtime.data.time),
                )
                last_initial_stable = h.initial_stable
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
        mujoco.mj_step = original_step
        transport.close()
        (out / "process_cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    evidence = {
        "native_entrypoint": "python -m supermimic.dds_deploy",
        "backend": "LOCAL_DDS_SIM_PHYSICS",
        "network_interface": args.network,
        "domain_id": args.domain_id,
        "contact_available": False,
        "dynamic_qp_recovery": "NOT_RUN",
        "control_pid": os.getpid(),
        "start_sim_time": start_sim_time,
        "end_sim_time": end_sim_time,
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
        "result": result["release"] if result is not None else None,
    }
    (out / "native_watchdog.json").write_text(json.dumps(evidence, indent=2) + "\n")
    paths = [
        Path(__file__),
        Path(__file__).with_name("dds_runtime.py"),
        Path(__file__).with_name("dds_transport.py"),
        Path(__file__).with_name("s10a.py"),
        Path(__file__).with_name("s10b.py"),
        Path(__file__).with_name("s10b_native.py"),
    ]
    (out / "dds_source_sha256.json").write_text(
        json.dumps({str(path.relative_to(ROOT)): sha(path) for path in paths}, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--network", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--signal-file", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--keyboard-trigger", action="store_true")
    parser.add_argument("--auto-trigger-after-stable", action="store_true")
    parser.add_argument("--arm-file", type=Path)
    parser.add_argument("--trigger-at", type=float, default=3.0)
    parser.add_argument("--dance-seconds", type=float)
    parser.add_argument("--prepare-seconds", type=float, default=2.0)
    parser.add_argument("--blend-seconds", type=float, default=0.2)
    parser.add_argument("--release-seconds", type=float, default=2.0)
    parser.add_argument("--post-stand", type=float, default=3.0)
    parser.add_argument("--transition-target-rate", type=float, default=2.0)
    parser.add_argument("--dance-target-rate", type=float, default=20.0)
    parser.add_argument("--torque-rate", type=float, default=6000.0)
    parser.add_argument("--transition-torque-rate", type=float, default=1500.0)
    parser.add_argument("--max-sim-seconds", type=float, default=160.0)
    parser.add_argument("--wall-seconds", type=float, default=180.0)
    parser.add_argument("--fault", choices=["none", "nan", "exception", "stop"], default="none")
    parser.add_argument("--fault-at", type=float, default=0.30)
    parser.add_argument("--policy-fault", choices=["none", "timeout", "permanent"], default="none")
    args = parser.parse_args()
    if args.keyboard_trigger and args.signal_file is None:
        parser.error("--keyboard-trigger requires --signal-file")
    if args.auto_trigger_after_stable and args.signal_file is None:
        parser.error("--auto-trigger-after-stable requires --signal-file")
    if args.auto_trigger_after_stable and args.arm_file is None:
        parser.error("--auto-trigger-after-stable requires --arm-file")
    if args.signal_file is not None and args.signal_file.exists():
        parser.error(f"Refusing stale dance trigger: {args.signal_file}")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    transport = DdsTransport(interface=args.network, domain_id=args.domain_id)
    try:
        runtime = DdsSharedDanceRuntime(args.out, transport)
        initial = runtime.sync_initial_state()
        (args.out / "initial_dds_state.json").write_text(
            json.dumps(
                {
                    "tick": initial.tick,
                    "mode_machine": initial.mode_machine,
                    "root_pos_w": initial.root_pos_w.tolist(),
                    "root_lin_vel_w": initial.root_lin_vel_w.tolist(),
                    "root_quat_wxyz": initial.root_quat_wxyz.tolist(),
                    "received_monotonic": initial.received_monotonic,
                },
                indent=2,
            )
            + "\n"
        )
        if args.keyboard_trigger:
            threading.Thread(
                target=_keyboard_listener,
                args=(args.signal_file, args.out / "control.jsonl"),
                name="keyboard_dance_trigger",
                daemon=True,
            ).start()
        result = run_dds(runtime, args, transport)
    finally:
        transport.close()
    raise SystemExit(0 if result["full_chain"] == "PASS" else 1)


if __name__ == "__main__":
    main()
