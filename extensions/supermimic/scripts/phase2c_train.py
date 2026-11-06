"""Launch one isolated Phase 2C resume from the immutable 38200 checkpoint."""
from __future__ import annotations

import types
import typing
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import torch
import tyro

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer

SOURCE = Path("logs/phase2a_anchor_4090_s0_20261030/ckpt_latest.pt")
STOP = 46_200


def construct(cls, values):
    hints = get_type_hints(cls)
    result = {}
    for field in fields(cls):
        if field.name not in values:
            continue
        value = values[field.name]
        kind = hints[field.name]
        origin = get_origin(kind)
        args = get_args(kind)
        if is_dataclass(kind):
            value = construct(kind, value)
        elif kind is Path:
            value = None if value is None else Path(value)
        elif origin is list:
            value = list(value)
        elif origin is tuple:
            value = tuple(value)
        elif origin in (typing.Union, types.UnionType) and value is not None:
            path_types = [t for t in args if t is Path]
            if path_types:
                value = Path(value)
        result[field.name] = value
    return cls(**result)


def changed_fields(before, after, prefix=""):
    changes = set()
    for key in before.keys() | after.keys():
        name = f"{prefix}.{key}" if prefix else key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changes.update(changed_fields(left, right, name))
        elif left != right:
            changes.add(name)
    return changes


def prepare_config(variant: str) -> TrainConfig:
    if variant not in ("control", "experiment"):
        raise ValueError(variant)
    run_name = f"phase2c_{variant}_38200_to_46200_20261104"
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    cfg = construct(TrainConfig, checkpoint["config"])
    original = asdict(cfg)
    cfg.run_name = run_name
    cfg.log_root = Path("logs")
    cfg.resume = SOURCE
    cfg.schedule.stop_iteration = STOP
    cfg.env.action_rate_weight = -0.1 if variant == "control" else -0.2
    allowed = {"run_name", "resume", "schedule.stop_iteration"}
    if variant == "experiment":
        allowed.add("env.action_rate_weight")
    changes = changed_fields(original, asdict(cfg))
    if changes != allowed:
        raise RuntimeError(f"unexpected config changes: {sorted(changes ^ allowed)}")
    return cfg


def main(variant: str):
    cfg = prepare_config(variant)
    destination = cfg.log_root / cfg.run_name
    if destination.exists():
        raise FileExistsError(destination)
    trainer = Trainer(cfg)
    try:
        if trainer.iteration != 38_200 or trainer.num_updates != 15_264:
            raise RuntimeError(f"unexpected resume point: {trainer.iteration}/{trainer.num_updates}")
        print(f"PHASE2C_CONFIG variant={variant} weight={cfg.env.action_rate_weight} "
              f"iteration={trainer.iteration} updates={trainer.num_updates} "
              f"stop={cfg.schedule.stop_iteration}", flush=True)
        trainer.run()
    finally:
        trainer.env.close()


if __name__ == "__main__":
    tyro.cli(main)
