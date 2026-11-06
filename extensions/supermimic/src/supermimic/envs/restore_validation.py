"""Index contracts for restoring captured simulator branches."""

from __future__ import annotations


def validate_branch_slots(slots: list[int], num_envs: int) -> None:
    if not slots:
        raise ValueError("restore requires at least one branch")
    if len(set(slots)) != len(slots) or min(slots) < 0 or max(slots) >= num_envs:
        raise IndexError(f"branch slots must be unique and in [0, {num_envs}); got {slots}")


def validate_snapshot_frame(frame: int, clip_name: str, library) -> None:
    if not 0 <= frame < library.num_frames:
        raise IndexError(f"snapshot frame {frame} for {clip_name} outside [0, {library.num_frames})")
    clip_idx = int(library.frame_clip[frame])
    actual_clip = library.clips[clip_idx].name
    if actual_clip != clip_name:
        raise ValueError(
            f"snapshot {clip_name} frame {frame} maps to clip {actual_clip!r}; "
            "source and branch motion-library ordering must match"
        )
