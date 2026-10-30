"""Render deterministic baseline/Anchor replays with the blue reference ghost.

Video uses the same 40-world clip order as benchmark; quantitative acceptance
still comes from the saved benchmark and drift outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.eval.benchmark import load_policy_from_checkpoint


def record(
    ckpt: Path,
    clip: str,
    npz_dir: Path,
    out: Path,
    seconds: float,
    render_every: int,
    width: int,
    height: int,
) -> dict:
    policy, saved_cfg = load_policy_from_checkpoint(ckpt, "cuda:0")
    anchor = bool(saved_cfg["env"].get("anchor_obs", False))
    assert policy.cfg.goal_dim == saved_cfg["env"]["future_frames"] * 68 + 9 * anchor
    names = (Path(__file__).resolve().parents[1] / "reports/phase2a_baseline_20261030/clips40.txt").read_text().splitlines()
    files = discover_npz(npz_dir, names=names)
    if len(files) != 40 or [p.stem for p in files] != names:
        raise ValueError("40-clip manifest does not match discovered motion order")
    clip_index = names.index(clip)
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files],
        num_envs=40,
        future_frames=saved_cfg["env"]["future_frames"],
        episode_length_s=1e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor,
        debug_vis=True,
    )
    env_cfg = make_env_cfg(args)
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    env_cfg.viewer.max_extra_envs = 0
    env_cfg.viewer.env_idx = clip_index
    env_cfg.viewer.distance = 3.3
    env = ManagerBasedRlEnv(env_cfg, device="cuda:0", render_mode="rgb_array")
    try:
        cmd = env.command_manager.get_term("motion")
        cmd.forced_frame[:] = cmd.library.clip_start[torch.arange(40, device="cuda:0")]
        obs, _ = env.reset()
        cmd.forced_frame[:] = -1
        steps = min(int(seconds / env.step_dt), cmd.library.clips[clip_index].length)
        frames = 0
        failed = False
        fps = round(1 / (env.step_dt * render_every), 6)
        with imageio.get_writer(
            out, fps=fps, codec="libx264", quality=8, macro_block_size=16
        ) as writer:
            for step in range(steps):
                if step % render_every == 0:
                    frame = env.render()
                    if frame is None or frame.shape != (height, width, 3) or np.std(frame) < 3:
                        raise RuntimeError(f"blank/invalid render at {clip} step {step}")
                    writer.append_data(frame)
                    frames += 1
                    if step == 0:
                        imageio.imwrite(out.with_suffix(".png"), frame)
                with torch.no_grad():
                    act = policy(obs["policy"], obs["goal"])
                    obs, _, term, tout, _ = env.step(act)
                if bool(term[clip_index] or tout[clip_index]):
                    failed = bool(term[clip_index])
                    break
        return {
            "checkpoint": str(ckpt),
            "clip": clip,
            "anchor_obs": anchor,
            "video": str(out),
            "frames": frames,
            "fps": fps,
            "steps_played": step + 1,
            "failed": failed,
            "seed": 0,
            "num_envs": 40,
            "note": "camera tracks robot torso; blue translucent robot is global reference",
        }
    finally:
        env.close()


def main(
    anchor_ckpt: Path,
    baseline_ckpt: Path,
    out_dir: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: tuple[str, ...] = ("dance1_subject1", "walk1_subject2", "run1_subject5"),
    seconds: float = 30.0,
    render_every: int = 5,
    width: int = 640,
    height: int = 480,
) -> None:
    if seconds <= 0 or render_every <= 0:
        raise ValueError("seconds and render_every must be positive")
    paths = [
        out_dir / f"{clip}_{label}.mp4"
        for clip in clips
        for label in ("baseline", "anchor", "comparison")
    ]
    paths += [out_dir / "manifest.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("video output already exists; use a new directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in clips:
        for label, ckpt in (("baseline", baseline_ckpt), ("anchor", anchor_ckpt)):
            out = out_dir / f"{clip}_{label}.mp4"
            record_info = record(ckpt, clip, npz_dir, out, seconds, render_every, width, height)
            records.append(record_info)
            print(f"recorded {out}: {record_info['frames']} frames", flush=True)
        left, right = (out_dir / f"{clip}_{label}.mp4" for label in ("baseline", "anchor"))
        comparison = out_dir / f"{clip}_comparison.mp4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(left),
                "-i",
                str(right),
                "-filter_complex",
                "hstack=inputs=2:shortest=1",
                "-c:v",
                "libx264",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                str(comparison),
            ],
            check=True,
        )
    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    tyro.cli(main)
