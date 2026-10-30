"""Read-only preflight for a staged SuperMimic resume."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer
from supermimic.training.schedule import cosine_lr


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def main(cfg: TrainConfig, report: Path) -> None:
    if cfg.resume is None or report.exists():
        raise ValueError("resume is required and report must not exist")
    old = torch.load(cfg.resume, map_location="cpu", weights_only=False)
    trainer = Trainer(cfg)
    try:
        checks = {
            "wm_weights": _equal(trainer.wm.state_dict(), old["wm"]),
            "policy_weights": _equal(trainer.policy.state_dict(), old["policy"]),
            "critic_weights": _equal(trainer.critic.state_dict(), old["critic"]),
            "wm_optimizer": _equal(trainer.wm_trainer.opt.state_dict(), old["wm_opt"]),
            "policy_optimizer": _equal(trainer.policy_trainer.opt.state_dict(), old["policy_opt"]),
            "critic_optimizer": _equal(trainer.policy_trainer.critic_opt.state_dict(), old["critic_opt"]),
            "sampler_bin_return": _equal(trainer.command.bin_return, old["bin_return"]),
            "sampler_bin_memory": _equal(trainer.command.bin_memory, old["bin_memory"]),
            "iteration": trainer.iteration == old["iteration"],
            "updates": trainer.num_updates == old["num_updates"],
            "replay_empty": trainer.buffer.size == 0,
            "warmup_origin": trainer.start_iteration == old["iteration"],
            "goal_dim": trainer.policy.cfg.goal_dim == 689,
            "anchor_obs": cfg.env.anchor_obs and cfg.policy_trainer.anchor_obs,
        }
        saved = old["config"]
        current = asdict(cfg)
        allowed = {"run_name", "resume", "log_root"}
        changed = []
        for key in current:
            if key == "schedule":
                for field, value in current[key].items():
                    if field in {"stop_iteration", "max_hours"}:
                        continue
                    if value != saved[key][field]:
                        changed.append(f"schedule.{field}")
            elif key not in allowed and current[key] != saved[key]:
                changed.append(key)
        checks["training_config_unchanged"] = not changed
        old_lr = old["wm_opt"]["param_groups"][0]["lr"]
        new_lr = trainer.wm_trainer.opt.param_groups[0]["lr"]
        checks["optimizer_lr_restored"] = old_lr == new_lr
        checks["lr_schedule_unchanged"] = cfg.schedule.total_iterations == saved["schedule"]["total_iterations"]
        result = {
            "checks": checks,
            "changed_unexpected_fields": changed,
            "source_iteration": old["iteration"],
            "source_updates": old["num_updates"],
            "restored_optimizer_lr": new_lr,
            "schedule_lr_at_source": cosine_lr(old["iteration"], cfg.schedule.total_iterations, cfg.wm_trainer.lr,
                                                cfg.schedule.lr_final_ratio, cfg.schedule.lr_decay_fraction),
            "next_update_earliest_iteration": old["iteration"] + cfg.schedule.warmup_iterations,
            "lr_at_first_possible_update": cosine_lr(old["iteration"] + cfg.schedule.warmup_iterations,
                cfg.schedule.total_iterations, cfg.wm_trainer.lr,
                cfg.schedule.lr_final_ratio, cfg.schedule.lr_decay_fraction),
            "replay_capacity": trainer.buffer.T,
            "warmup_steps": cfg.schedule.warmup_iterations,
            "window": cfg.wm_trainer.window,
            "history": cfg.policy_trainer.history,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not all(checks.values()):
            raise RuntimeError("resume preflight failed")
    finally:
        trainer.env.close()


if __name__ == "__main__":
    tyro.cli(main)
