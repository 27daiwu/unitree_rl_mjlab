"""S10B fault fixture: process-isolated inference around unchanged S10A control.

Nominal and progress-abort acceptance runs invoke the original S10A module.
This module tests timeout/block recovery using a separate inference process,
a separate monotonic watchdog process, and a physics owner that never waits
indefinitely for inference. No S10A source, control law or limits are modified.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def record(path, event, **values):
    item = {"event": event, "pid": os.getpid(), "monotonic": time.monotonic(), **values}
    with Path(path).open("a") as stream:
        stream.write(json.dumps(item) + "\n")
    return item


def policy_worker(conn, checkpoint, kind, delay, log):
    import torch

    from supermimic.policy.nets import Policy, PolicyCfg

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]["policy"]
    policy = (
        Policy(PolicyCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}))
        .cpu()
        .eval()
    )
    policy.load_state_dict(ckpt["policy"], strict=True)
    del ckpt
    with torch.inference_mode():
        policy(torch.zeros(1, 930), torch.zeros(1, 689))
    record(log, "ready", strict_binding=True, checkpoint=str(checkpoint), warmup_forwards=1)
    conn.send({"kind": "ready", "pid": os.getpid()})
    while True:
        request = conn.recv()
        if request["kind"] == "close":
            break
        seq = request["seq"]
        if request["inject"]:
            started = record(
                log,
                "fault_entered",
                seq=seq,
                fault=kind,
                frame=request["frame"],
                sim_time=request["sim_time"],
            )
            conn.send({"kind": "fault_entered", **started})
            if kind == "permanent":
                threading.Event().wait()  # No response or recovery in this process.
            else:
                time.sleep(delay)
                record(log, "delay_finished", seq=seq)
        with torch.inference_mode():
            action = (
                policy(torch.from_numpy(request["obs"]), torch.from_numpy(request["goal"]))
                .numpy()
                .copy()
            )
        conn.send({"kind": "action", "seq": seq, "action": action})
    record(log, "closed")


def watchdog_worker(status, tripped, stop, ready, timeout, log):
    record(log, "ready", deadline_s=timeout)
    ready.set()
    while not stop.wait(0.001):
        with status.get_lock():
            seq, sent, completed = status[:]
        if seq > completed and sent > 0 and time.monotonic() - sent >= timeout:
            record(
                log,
                "deadline_expired",
                seq=int(seq),
                sent_monotonic=sent,
                deadline_s=timeout,
                elapsed_s=time.monotonic() - sent,
            )
            tripped.set()
            return


class InferencePending(Exception):
    """IPC's short polling budget expired; the physics owner must keep stepping."""


class IsolatedPolicy:
    def __init__(self, conn, process, status, tripped, log, poll_budget=0.005):
        self.conn, self.process = conn, process
        self.status, self.tripped, self.log = status, tripped, log
        self.poll_budget = poll_budget
        self.seq = 0
        self.inject = False
        self.frame = 0
        self.sim_time = 0.0
        self.pending = None
        self.fault_ack = None
        self.late_results = 0

    def _receive(self):
        response = self.conn.recv()
        if response["kind"] == "fault_entered":
            self.fault_ack = response
            record(
                self.log,
                "worker_fault_ack",
                worker_pid=response["pid"],
                seq=response["seq"],
                worker_monotonic=response["monotonic"],
            )
            return None
        if response["kind"] != "action" or response["seq"] != self.seq:
            raise ValueError("Invalid or out-of-order inference reply")
        return response["action"]

    def accept(self, action):
        if self.tripped.is_set():
            self.late_results += 1
            record(self.log, "late_action_discarded", seq=self.seq)
            return None
        with self.status.get_lock():
            self.status[2] = self.seq
        self.pending = None
        return action

    def __call__(self, obs, goal):
        import torch

        if self.pending is not None or self.tripped.is_set():
            raise RuntimeError("Inference owner is unavailable")
        self.seq += 1
        request = {
            "kind": "infer",
            "seq": self.seq,
            "obs": obs.numpy().copy(),
            "goal": goal.numpy().copy(),
            "frame": self.frame,
            "sim_time": self.sim_time,
            "inject": self.inject,
        }
        sent = time.monotonic()
        self.pending = {**request, "sent_monotonic": sent}
        with self.status.get_lock():
            self.status[0], self.status[1] = self.seq, sent
        self.conn.send(request)
        if self.inject:
            record(
                self.log,
                "fault_request_sent",
                seq=self.seq,
                frame=self.frame,
                sim_time=self.sim_time,
                worker_pid=self.process.pid,
            )
        end = sent + self.poll_budget
        while self.conn.poll(max(0.0, end - time.monotonic())):
            action = self._receive()
            if action is not None:
                accepted = self.accept(action)
                if accepted is not None:
                    return torch.from_numpy(accepted)
                break
            if time.monotonic() >= end:
                break
        raise InferencePending

    def poll(self):
        while self.conn.poll():
            action = self._receive()
            if action is not None:
                return self.accept(action)
        return None


def stop_process(process):
    if process.is_alive():
        process.terminate()
    process.join(2.0)
    if process.is_alive():
        process.kill()
        process.join(2.0)
    return {"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()}


def run_watchdog(out, kind, fraction=0.5, deadline=0.1, delay=0.5):
    import numpy as np
    import torch

    from supermimic.s10a import CHECKPOINT, ROOT, DanceRuntime, Handoff, State, sha

    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frozen = ROOT / "extensions/supermimic/reports/phase2_s10a_handoff_20261110/final_full_01"
    values = json.loads((frozen / "arguments.json").read_text())
    values.update(out=out, signal_file=None, stop_file=None, fault="none")
    args = SimpleNamespace(**values)
    (out / "arguments.json").write_text(json.dumps(values, default=str, indent=2) + "\n")
    (out / "fixture_arguments.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "fraction": fraction,
                "deadline_s": deadline,
                "delay_s": delay,
                "short_ipc_poll_budget_s": 0.005,
                "paced_physics_during_pending_and_recovery": True,
            },
            indent=2,
        )
        + "\n"
    )
    (out / "fixture_sha256.json").write_text(
        json.dumps({str(Path(__file__).relative_to(ROOT)): sha(__file__)}, indent=2) + "\n"
    )
    runtime = DanceRuntime(out)
    handoff = Handoff(runtime, args)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    status = ctx.Array("d", [0.0, 0.0, 0.0], lock=True)
    tripped, stop, ready = ctx.Event(), ctx.Event(), ctx.Event()
    infer = ctx.Process(
        target=policy_worker,
        args=(child, CHECKPOINT, kind, delay, out / "inference.jsonl"),
        name="s10b_inference",
    )
    watch = ctx.Process(
        target=watchdog_worker,
        args=(status, tripped, stop, ready, deadline, out / "watchdog.jsonl"),
        name="s10b_watchdog",
    )
    infer.start()
    child.close()
    watch.start()
    if not parent.poll(30.0) or parent.recv().get("kind") != "ready" or not ready.wait(5.0):
        stop.set()
        cleanup = [stop_process(p) for p in (infer, watch)]
        (out / "setup_failure.json").write_text(json.dumps(cleanup, indent=2) + "\n")
        raise RuntimeError("Isolated worker/watchdog startup failed")
    proxy = IsolatedPolicy(parent, infer, status, tripped, out / "control.jsonl")
    runtime.policy = proxy
    fault_frame = math.ceil(len(runtime.ref) * fraction)
    trigger = None
    trip_record = None
    pending_context = None
    pacing_wall = None
    pacing_sim = None
    accepted_after_trip = 0
    blocked_physics_steps = 0
    min_time_step = float("inf")
    max_time_step = 0.0
    start = time.monotonic()
    record(
        out / "control.jsonl",
        "ready",
        inference_pid=infer.pid,
        watchdog_pid=watch.pid,
        fault_frame=fault_frame,
    )
    try:
        while (
            runtime.data.time < args.max_sim_seconds
            and time.monotonic() - start < args.wall_seconds
        ):
            if tripped.is_set() and trip_record is None:
                trip_record = record(
                    out / "control.jsonl",
                    "watchdog_observed",
                    sim_time=float(runtime.data.time),
                    physics_steps=len(handoff.rows),
                    inference_alive=infer.is_alive(),
                    frame=handoff.phase,
                    state=handoff.state.name,
                )
                if handoff.state in (State.TAKEOVER, State.DANCE):
                    handoff.release("policy_watchdog_timeout", fault=True)
                pending_context = None
            if proxy.pending is not None:
                packet = proxy.pending.copy()
                action = proxy.poll()
                if action is not None:
                    if trip_record is not None:
                        accepted_after_trip += 1
                    elif pending_context is not None:
                        if action.shape != (1, 29) or not np.isfinite(action).all():
                            raise ValueError("Invalid isolated Policy action")
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
                        handoff.policy_target = runtime.q0 + runtime.scale * action
                        if handoff.state == State.DANCE:
                            handoff.phase += 1
                        handoff.next_infer = max(
                            pending_context["next_infer"] + 0.02, float(runtime.data.time)
                        )
                        pending_context = None
            if trigger is None and handoff.state == State.DANCE and handoff.phase >= fault_frame:
                proxy.inject = True
                trigger = record(
                    out / "control.jsonl",
                    "fault_armed",
                    frame=handoff.phase,
                    progress=handoff.phase / len(runtime.ref),
                    sim_time=float(runtime.data.time),
                    physics_steps=len(handoff.rows),
                )
                pacing_wall, pacing_sim = time.monotonic(), float(runtime.data.time)
            proxy.frame = handoff.phase if handoff.state == State.DANCE else 0
            proxy.sim_time = float(runtime.data.time)
            previous_time = float(runtime.data.time)
            next_infer = handoff.next_infer
            try:
                keep_running = handoff.tick()
            except InferencePending:
                record(
                    out / "control.jsonl",
                    "inference_pending",
                    seq=proxy.seq,
                    frame=proxy.frame,
                    sim_time=float(runtime.data.time),
                    injected=bool(proxy.inject),
                )
                pending_context = {"next_infer": next_infer}
                handoff.next_infer = float("inf")
                if pacing_wall is None:
                    pacing_wall, pacing_sim = time.monotonic(), float(runtime.data.time)
                continue
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- preserve S10A fault handling
                if handoff.state in (State.RELEASE, State.SAFE_DAMP):
                    handoff.fault = f"unrecoverable:{type(error).__name__}:{error}"
                    break
                handoff.release(f"{type(error).__name__}:{error}", fault=True)
                continue
            if not keep_running:
                break
            step = float(runtime.data.time) - previous_time
            min_time_step, max_time_step = min(min_time_step, step), max(max_time_step, step)
            if trigger is not None and infer.is_alive():
                blocked_physics_steps += 1
            if pacing_wall is not None:
                # Fault intervals advance physical time at real wall time. A long
                # IPC wait cannot be hidden by simply pausing or accelerating sim.
                until = pacing_wall + float(runtime.data.time) - pacing_sim - time.monotonic()
                if until > 0:
                    time.sleep(min(until, 0.005))
        worker_alive_at_end = infer.is_alive()
        end_record = record(
            out / "control.jsonl",
            "control_finished",
            state=handoff.state.name,
            sim_time=float(runtime.data.time),
            inference_alive=worker_alive_at_end,
            physics_steps=len(handoff.rows),
            returned=handoff.returned,
        )
        result = handoff.save(time.monotonic() - start)
    finally:
        stop.set()
        cleanup = {"inference": stop_process(infer), "watchdog": stop_process(watch)}
        parent.close()
        (out / "process_cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    evidence = {
        "control_pid": os.getpid(),
        "inference_pid": infer.pid,
        "watchdog_pid": watch.pid,
        "fault_frame": fault_frame,
        "fault_trigger": trigger,
        "watchdog_release": trip_record,
        "fault_ack": proxy.fault_ack,
        "worker_alive_at_end": worker_alive_at_end,
        "control_end": end_record,
        "physics_steps_while_inference_process_alive_after_fault": blocked_physics_steps,
        "physics_steps_after_watchdog_release": (
            len(handoff.rows) - trip_record["physics_steps"] if trip_record else 0
        ),
        "physics_time_step_min_s": min_time_step,
        "physics_time_step_max_s": max_time_step,
        "late_results_discarded": proxy.late_results,
        "policy_actions_accepted_after_watchdog": accepted_after_trip,
        "cleanup": cleanup,
        "underlying_result": result["release"],
    }
    (out / "watchdog_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=["timeout", "permanent"], required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--deadline", type=float, default=0.1)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.fraction < 1 or args.deadline <= 0 or args.delay <= args.deadline:
        parser.error("Use fraction in (0,1), positive deadline, delay longer than deadline")
    result = run_watchdog(args.out, args.kind, args.fraction, args.deadline, args.delay)
    raise SystemExit(0 if result["release"] == "PASS" else 1)


if __name__ == "__main__":
    main()
