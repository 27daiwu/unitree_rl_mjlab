"""Capture D3 branch snapshots inside the exact D2 telemetry playback path."""
from __future__ import annotations

import json

import diag_stability_mechanism as d2
import numpy as np
import torch
import tyro
from diag_policy_cross_state import CKPTS, D2, DT, HERE, HORIZON, SOURCE_STARTS, capture


def main(source: str):
    if source not in CKPTS:
        raise ValueError(source)
    starts = SOURCE_STARTS[source]
    HERE.mkdir(parents=True, exist_ok=True)
    destination = HERE / f"source_{source}.pt"
    trace = HERE / f"source_trace_{source}.npz"
    if destination.exists() or trace.exists():
        raise FileExistsError(destination if destination.exists() else trace)
    original_make_env = d2.make_env
    snapshots = {}
    names = None

    def instrumented_make_env(*args, **kwargs):
        nonlocal names
        env = original_make_env(*args, **kwargs)
        names = [p.rsplit("/", 1)[-1].removesuffix(".npz") for p in args[0].npz_files]
        original_step = env.step
        count = 0

        def capture_then_step(action):
            nonlocal count
            for name, second in starts.items():
                if count == round(second / DT):
                    snapshots[name] = capture(env, names.index(name), env.obs_buf, count)
            count += 1
            return original_step(action)

        env.step = capture_then_step
        return env

    d2.make_env = instrumented_make_env
    try:
        d2.main(CKPTS[source], trace)
    finally:
        d2.make_env = original_make_env
    if names is None or len(snapshots) != len(starts):
        raise RuntimeError(f"missing snapshots: {set(starts) - set(snapshots)}")
    with np.load(trace) as raw:
        events = json.loads(str(raw["meta_json"]))["events"]
        clips = list(raw["clips"])
        native = {}
        for name, snap in snapshots.items():
            end = min(snap["step"] + HORIZON, events[name]["step"])
            if end <= snap["step"]:
                raise RuntimeError(f"snapshot after termination: {name}, {events[name]}")
            idx = clips.index(name)
            for key, lag in (("manager_action", 1), ("manager_prev_action", 2),
                             ("manager_prev_prev_action", 3)):
                if key not in snap:
                    snap[key] = raw["action"][max(0, snap["step"] - lag), idx].copy()
            native[name] = [{"state": raw["state"][k, idx], "action": raw["action"][k, idx]}
                            for k in range(snap["step"], end)]
        with np.load(D2 / f"trace_{source}.npz") as old:
            oldclips = list(old["clips"])
            deltas = {name: float(np.max(np.abs(
                snap["qpos"].copy()[:3] - snap["origin"] - old["state"][snap["step"], oldclips.index(name), :3]
            ))) for name, snap in snapshots.items()}
            state_deltas = {}
            for name, snap in snapshots.items():
                state = np.concatenate([snap["qpos"].copy(), snap["qvel"]])
                state[:3] -= snap["origin"]
                state_deltas[name] = float(np.max(np.abs(state - old["state"][snap["step"], oldclips.index(name)])))
    torch.save({"snapshots": snapshots, "native": native, "events": events,
                "d2_snapshot_max_abs_state_delta": state_deltas,
                "d2_snapshot_max_abs_xy_delta": deltas,
                "names": names, "starts": starts, "checkpoint": str(CKPTS[source]), "trace": str(trace)}, destination)
    print("saved", destination, "events", {k: events[k] for k in starts},
          "D2 deltas", state_deltas, flush=True)


if __name__ == "__main__":
    tyro.cli(main)
