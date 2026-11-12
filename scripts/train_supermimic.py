"""Train a SuperMimic task, with the same CLI shape as ``scripts/train.py``.

  python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2
  python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2 --env.num-envs 512

The first positional argument is a task id from ``supermimic.tasks`` (list them
with ``--list-tasks``), and every remaining field of the SuperMimic
``TrainConfig`` is overridable as a tyro flag -- identical to how
``scripts/train.py <PPO task id> --agent.max-iterations N`` works.

This script must run under the extension environment
(``extensions/supermimic/.venv``), because SuperMimic pins a newer mjlab than the
root PPO environment.  Running it with any interpreter re-executes automatically
inside that environment, so ``python scripts/train_supermimic.py <TASK_ID>`` works
from either the repository root or the plugin directory.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXT_ROOT = _REPO_ROOT / "extensions/supermimic"
_EXT_VENV = _EXT_ROOT / ".venv/bin/python"

# Allow `python scripts/train_supermimic.py` from any working directory.
for _p in (_EXT_ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _reexec_in_extension_env() -> None:
    """Re-run under the extension interpreter if SuperMimic's dependencies are absent.

    The root PPO environment ships an older mjlab without ``managers.recorder_manager``,
    so importing SuperMimic there fails.  Delegating keeps a single entry point.
    """
    if os.environ.get("SM_REEXEC") == "1" or not _EXT_VENV.exists():
        return
    try:
        import mjlab.managers.recorder_manager  # noqa: F401
        return
    except Exception:
        pass
    os.environ["SM_REEXEC"] = "1"
    os.execv(str(_EXT_VENV), [str(_EXT_VENV), str(Path(__file__).resolve()), *sys.argv[1:]])


def main() -> None:
    _reexec_in_extension_env()

    import tyro

    import supermimic.tasks as tasks
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_usage(tasks)
        return
    if argv[0] in ("--list-tasks", "--list_tasks"):
        _print_usage(tasks)
        return

    task_id, rest = argv[0], argv[1:]
    if task_id not in tasks.list_tasks():
        sys.exit(
            f"[ERROR] unknown task '{task_id}'.\n"
            f"Available: {', '.join(tasks.list_tasks())}\n"
            f"Run with --list-tasks for details."
        )

    from mjlab.utils.gpu import select_gpus

    # --gpu-ids is consumed here (like scripts/train.py) rather than by TrainConfig,
    # so that a task id plus flags stays a flat, PPO-compatible CLI.
    gpu_ids, rest = _pop_gpu_ids(rest)
    selected_gpus, num_gpus = select_gpus(gpu_ids)
    if selected_gpus is None:
        sys.exit(
            "[ERROR] SuperMimic requires a CUDA device (MuJoCo Warp simulation); "
            "no GPU selected."
        )
    if num_gpus > 1:
        sys.exit(
            "[ERROR] multi-GPU SuperMimic launch is not wired here; select one GPU "
            "with --gpu-ids <n> and launch one process per GPU via torchrunx if needed."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ.setdefault("MUJOCO_GL", "egl")

    cfg = tyro.cli(
        TrainConfig,
        args=rest,
        default=tasks.build_config(task_id),
        prog=f"{sys.argv[0]} {task_id}",
    )
    # cfg.device is set by tyro default; align it with the selected physical GPU.
    if cfg.device.startswith("cuda"):
        cfg.device = "cuda:0"

    print(
        f"[INFO] task={task_id} device={cfg.device} gpu={selected_gpus} "
        f"clips={cfg.data.clips or 'all'} num_envs={cfg.env.num_envs} "
        f"seed={cfg.seed} iterations={cfg.schedule.total_iterations}"
    )
    Trainer(cfg).run()


def _pop_gpu_ids(args: list[str]) -> tuple[list[int] | str | None, list[str]]:
    """Extract --gpu-ids before tyro sees it (tyro has no such field in TrainConfig)."""
    out: list[str] = []
    gpu_ids: list[int] | str | None = [0]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--gpu-ids":
            if i + 1 >= len(args):
                sys.exit("[ERROR] --gpu-ids requires a value")
            val = args[i + 1]
            gpu_ids = "all" if val == "all" else [int(v) for v in val.split(",")]
            i += 2
            continue
        if a.startswith("--gpu-ids="):
            val = a.split("=", 1)[1]
            gpu_ids = "all" if val == "all" else [int(v) for v in val.split(",")]
            i += 1
            continue
        out.append(a)
        i += 1
    return gpu_ids, out


def _print_usage(tasks) -> None:
    print("Usage: python scripts/train_supermimic.py <TASK_ID> [OPTIONS]\n")
    print("Available SuperMimic tasks:\n")
    rows = tasks.describe_tasks()
    w = max(len(r[0]) for r in rows) if rows else 10
    for task_id, clips, desc in rows:
        print(f"  {task_id:<{w}}  clips={clips:<18}  {desc}")
    print(
        "\nCommon options:\n"
        "  --gpu-ids 0                 GPU index (default 0)\n"
        "  --env.num-envs 256          parallel environments\n"
        "  --seed 0                    random seed\n"
        "  --schedule.total-iterations 320000   training length (env steps)\n"
        "  --log-root logs             output root; run dir is <log-root>/<run-name>\n"
        "  --run-name my_run           run directory name\n"
        "  --resume logs/<run>/ckpt_latest.pt   resume from checkpoint\n"
        "  --data.clips dance1_subject2         override the motion clips\n"
        "  --data.npz-dir data/npz/lafan1       motion directory\n"
    )


if __name__ == "__main__":
    main()
