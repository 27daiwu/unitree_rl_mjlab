"""SuperMimic task registry, mirroring ``mjlab.tasks.registry`` for the PPO tasks.

SuperMimic is a plugin task of ``unitree_rl_mjlab``, but its trainer is the paper's
world-model + BPTT loop, not RSL-RL PPO.  It therefore cannot share
``mjlab.tasks.registry`` (whose entries are ``ManagerBasedRlEnvCfg`` +
``RslRlBaseRunnerCfg`` pairs consumed by ``scripts/train.py``).  This module gives
SuperMimic the same *usage*: a task id selects a complete, named configuration.

A task bundles the motion set plus the training defaults validated for that motion
(environment count, anchor observations, schedule).  Every field stays overridable
on the command line, exactly like ``scripts/train.py`` does for PPO tasks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from supermimic.config import TrainConfig


# Absolute extension root, so defaults resolve identically no matter which
# directory the training entry point is launched from (repo root or plugin root).
EXT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ_DIR = EXT_ROOT / "data/npz/lafan1"
DEFAULT_LOG_ROOT = EXT_ROOT / "logs"


@dataclass(frozen=True)
class TaskSpec:
    """A named SuperMimic task: motion selection + validated training defaults."""

    task_id: str
    clips: list[str] | None
    num_envs: int
    anchor_obs: bool
    max_iterations: int
    description: str


_TASKS: dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec) -> None:
    if spec.task_id in _TASKS:
        raise ValueError(f"task '{spec.task_id}' is already registered")
    _TASKS[spec.task_id] = spec


def list_tasks() -> list[str]:
    return sorted(_TASKS)


def get_task(task_id: str) -> TaskSpec:
    try:
        return _TASKS[task_id]
    except KeyError:
        raise KeyError(
            f"unknown SuperMimic task '{task_id}'; available: {', '.join(list_tasks())}"
        ) from None


def describe_tasks() -> list[tuple[str, str, str]]:
    """(task_id, clips, description) rows for the CLI listing."""
    rows = []
    for task_id in list_tasks():
        spec = _TASKS[task_id]
        clips = ",".join(spec.clips) if spec.clips else "all"
        rows.append((task_id, clips, spec.description))
    return rows


def build_config(task_id: str, npz_dir: Path | None = None) -> TrainConfig:
    """Default ``TrainConfig`` for a task, before CLI overrides."""
    spec = get_task(task_id)
    cfg = TrainConfig()
    cfg.run_name = task_id
    cfg.data.npz_dir = Path(npz_dir) if npz_dir is not None else DEFAULT_NPZ_DIR
    cfg.log_root = DEFAULT_LOG_ROOT
    cfg.data.clips = deepcopy(spec.clips)
    cfg.env.num_envs = spec.num_envs
    cfg.env.anchor_obs = spec.anchor_obs
    cfg.schedule.total_iterations = spec.max_iterations
    return cfg


# The verified single-motion reference: full-length replay passed 3/3 for the
# checkpoints produced by this configuration (see extensions/supermimic/README.md).
register_task(
    TaskSpec(
        task_id="Supermimic-G1-Dance1-Subject2",
        clips=["dance1_subject2"],
        num_envs=256,
        anchor_obs=True,
        max_iterations=320_000,
        description="dance1_subject2 anchor-relative tracking (verified reference motion)",
    )
)

register_task(
    TaskSpec(
        task_id="Supermimic-G1-Dance1-Subject1",
        clips=["dance1_subject1"],
        num_envs=256,
        anchor_obs=True,
        max_iterations=320_000,
        description="dance1_subject1 anchor-relative tracking",
    )
)

register_task(
    TaskSpec(
        task_id="Supermimic-G1-Dance2-Subject1",
        clips=["dance2_subject1"],
        num_envs=256,
        anchor_obs=True,
        max_iterations=320_000,
        description="dance2_subject1 anchor-relative tracking",
    )
)

register_task(
    TaskSpec(
        task_id="Supermimic-G1-Jumps1-Subject1",
        clips=["jumps1_subject1"],
        num_envs=256,
        anchor_obs=True,
        max_iterations=320_000,
        description="jumps1_subject1 anchor-relative tracking",
    )
)

register_task(
    TaskSpec(
        task_id="Supermimic-G1-Fight1-Subject2",
        clips=["fight1_subject2"],
        num_envs=256,
        anchor_obs=True,
        max_iterations=320_000,
        description="fight1_subject2 anchor-relative tracking",
    )
)

register_task(
    TaskSpec(
        task_id="Supermimic-G1-Lafan1-All",
        clips=None,
        num_envs=256,
        anchor_obs=True,
        max_iterations=320_000,
        description="every clip in the motion directory (multi-clip training)",
    )
)
