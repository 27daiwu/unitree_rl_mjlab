"""D3 adjacent-start replay for fallAndGetUp2_subject3 (24 frozen-policy branches)."""
from __future__ import annotations

import json
from pathlib import Path

import diag_stability_mechanism as d2
import numpy as np
import torch
import tyro
from diag_policy_cross_state import CKPTS, DT, capture, restore

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint

OUT = Path("reports/phase2c_d3_adjacent_20261104")
CLIP = "fallAndGetUp2_subject3"
STARTS = (12.0, 12.4)
HORIZON = 100


@torch.no_grad()
def capture_source(source: str) -> None:
    out = OUT / f"source_trace_{source}.npz"
    if out.exists():
        raise FileExistsError(out)
    snapshots = {}
    library_names = None
    original_make_env = d2.make_env

    def instrumented_make_env(*args, **kwargs):
        nonlocal library_names
        env = original_make_env(*args, **kwargs)
        names = [p.rsplit("/", 1)[-1].removesuffix(".npz") for p in args[0].npz_files]
        library_names = names
        clip_slot = names.index(CLIP)
        original_step = env.step
        count = 0

        def capture_then_step(action):
            nonlocal count
            for second in STARTS:
                if count == round(second / DT):
                    snapshots[f"{second:.1f}"] = capture(env, clip_slot, env.obs_buf, count)
            count += 1
            return original_step(action)

        env.step = capture_then_step
        return env

    d2.make_env = instrumented_make_env
    try:
        d2.main(CKPTS[source], out, max_steps=round((max(STARTS) + 0.1) / DT))
    finally:
        d2.make_env = original_make_env
    if set(snapshots) != {f"{s:.1f}" for s in STARTS}:
        raise RuntimeError(f"missing source snapshots: {set(snapshots)}")
    torch.save({"snapshots": snapshots, "starts": STARTS, "clip": CLIP, "source": source,
                "library_names": library_names},
               OUT / f"source_{source}.pt")
    print("captured", source, sorted(snapshots), flush=True)


@torch.no_grad()
def branch_source(source: str, single: bool = False) -> None:
    path = OUT / (f"branch_smoke_{source}.npz" if single else f"branches_{source}.npz")
    if path.exists():
        raise FileExistsError(path)
    saved = torch.load(OUT / f"source_{source}.pt", map_location="cpu", weights_only=False)
    if "library_names" not in saved:
        saved["library_names"] = Path(
            "reports/phase2a_baseline_20261030/clips40.txt"
        ).read_text().splitlines()
    snapshots = {f"{CLIP}@{t}": saved["snapshots"][f"{t:.1f}"] for t in STARTS}
    branches = [(slot, f"{CLIP}@{t}", source, policy, repeat)
                for slot, (t, policy, repeat) in enumerate(
                    (t, p, r) for t in STARTS for p in CKPTS for r in range(3))]
    if single:
        branches = branches[:1]
    policies = {k: load_policy_from_checkpoint(v, "cuda:0")[0] for k, v in CKPTS.items()}
    names = Path("reports/phase2a_baseline_20261030/clips40.txt").read_text().splitlines()
    if saved["library_names"] != names:
        raise ValueError("source and branch clip manifest ordering differs")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[f"data/npz/lafan1/{name}.npz" for name in names], num_envs=40,
        future_frames=10, episode_length_s=1e5, action_noise_std=0,
        obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False, seed=0, anchor_obs=True,
    ))
    try:
        cmd = env.command_manager.get_term("motion")
        cmd.forced_frame[:] = cmd.library.clip_start
        obs, _ = env.reset()
        cmd.forced_frame[:] = cmd.library.clip_start[-1]
        obs, errors = restore(env, snapshots, branches)
        if max(max(v.values()) for v in errors.values()) > 1e-4:
            raise RuntimeError(f"state restore mismatch: {errors}")
        ids = torch.arange(len(branches), device=env.device)
        io = StateIO(env)
        active = np.ones(len(branches), dtype=bool)
        events = {}
        traces = {k: [] for k in ("state", "ref", "action", "active")}
        for step in range(HORIZON):
            traces["state"].append(io.read()[ids].cpu().numpy())
            traces["ref"].append(cmd.library.ref71(cmd.frame)[ids].cpu().numpy())
            traces["active"].append(active.copy())
            action = torch.zeros((env.num_envs, 29), device=env.device)
            for policy in CKPTS:
                slots = torch.as_tensor(
                    [b[0] for b in branches if b[3] == policy],
                    device=env.device,
                    dtype=torch.long,
                )
                action[slots] = policies[policy](obs["policy"][slots], obs["goal"][slots])
            traces["action"].append(action[ids].cpu().numpy())
            obs, _, terminated, timeout, _ = env.step(action)
            for b in branches:
                slot = b[0]
                if active[slot] and (bool(terminated[slot]) or bool(timeout[slot])):
                    events[str(slot)] = {"step": step + 1, "failed": bool(terminated[slot]),
                                         "reasons": [k for k in ("root_height", "root_tilt")
                                                     if bool(env.termination_manager.get_term(k)[slot])]}
                    active[slot] = False
        np.savez_compressed(path, **{k: np.stack(v) for k, v in traces.items()},
                            branches=np.asarray(branches, dtype=str),
                            meta=json.dumps({"events": events, "restore_errors": errors,
                                             "source_checkpoint": str(CKPTS[source]),
                                             "policy_checkpoints": {k: str(v) for k, v in CKPTS.items()}}))
        print("saved", path, "events", events, flush=True)
    finally:
        env.close()


def main(mode: str, source: str):
    if source not in CKPTS:
        raise ValueError(source)
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "source":
        capture_source(source)
    elif mode == "branch":
        branch_source(source)
    elif mode == "branch-single":
        branch_source(source, single=True)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    tyro.cli(main)
