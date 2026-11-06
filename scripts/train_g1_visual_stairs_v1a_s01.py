"""Longer-horizon Flat-teacher cloning for V1A approach stabilization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import train_g1_visual_stairs_v1a_s0 as s0


OUT = Path("logs/experiments/g1_visual_stairs_v1a_s01_20261104")
S0_BC = Path("logs/experiments/g1_visual_stairs_v1a_s0_20261104/v1a_s0_bc_model.pt")
DATA_STEPS = 50
EPOCHS = 5
LR = 1e-4
BATCH = 512
EVAL_SEEDS = (142, 314)


def collect_long_dataset(device: str) -> dict[str, torch.Tensor]:
  s0.TRAIN_STEPS = DATA_STEPS
  return s0.collect_dataset(device)


def train(dataset: dict[str, torch.Tensor], device: str) -> dict:
  env, runner = s0.make_env(917, 1, device)
  try:
    runner.load(str(S0_BC), load_cfg={"actor": True}, strict=True, map_location=device)
    model = runner.alg.get_policy()
    model.train()
    model.obs_normalizer.eval()
    tensors = {k: v.to(device) for k, v in dataset.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator(device=device).manual_seed(20261105)
    history = []
    for epoch in range(EPOCHS):
      order = torch.randperm(tensors["action"].shape[0], generator=rng, device=device)
      losses = []
      for ids in order.split(BATCH):
        obs = {"actor": tensors["actor"][ids], "depth": tensors["depth"][ids]}
        pred = model(obs)
        loss = torch.nn.functional.mse_loss(pred, tensors["action"][ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
      history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
    model.eval()
    checkpoint = torch.load(S0_BC, map_location="cpu", weights_only=False)
    checkpoint["actor_state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    checkpoint["infos"] = {"s0_1_behavior_cloned_from": str(s0.FLAT_SOURCE),
                           "initialized_from": str(S0_BC), "updates": EPOCHS,
                           "training_samples": int(tensors["action"].shape[0]),
                           "teacher_horizon_s": DATA_STEPS * 0.02}
    out = OUT / "v1a_s0_1_bc_model.pt"
    torch.save(checkpoint, out)
    return {"epochs": history, "samples": int(tensors["action"].shape[0]),
            "batch_size": BATCH, "learning_rate": LR, "initialized_from": str(S0_BC),
            "model": str(out)}
  finally:
    env.close()


def summarize(rows: list[dict]) -> dict:
  summaries = {}
  for seed in EVAL_SEEDS:
    summaries[str(seed)] = {}
    for policy in sorted({r["policy"] for r in rows}):
      group = [r for r in rows if r["seed"] == seed and r["policy"] == policy]
      episodes = []
      for env_id in range(16):
        ep = [r for r in group if r["env_id"] == env_id]
        survived = len(ep) == 50 and not any(r["fell_over"] for r in ep)
        episodes.append({"env_id": env_id, "survived_1s": survived,
                         "positive_x": ep[-1]["root_x_m"] > 0,
                         "duration_s": ep[-1]["time_s"],
                         "x_displacement_m": ep[-1]["root_x_m"],
                         "pitch_rms_deg": float(np.sqrt(np.mean([r["pitch_deg"] ** 2 for r in ep]))),
                         "vx_rmse_mps": float(np.sqrt(np.mean([(r["world_x_velocity_mps"] - 0.4) ** 2 for r in ep]))),
                         "mean_speed_mps": float(np.mean([r["world_x_velocity_mps"] for r in ep])),
                         "peak_abs_pitch_deg": float(max(abs(r["pitch_deg"]) for r in ep)),
                         "mean_contact_count": float(np.mean([r["left_contact"] + r["right_contact"] for r in ep]))})
      summaries[str(seed)][policy] = {
        "episodes": episodes,
        "survivors_1s": sum(e["survived_1s"] for e in episodes),
        "positive_x_1s": sum(e["positive_x"] for e in episodes),
        "mean_x_displacement_m": float(np.mean([e["x_displacement_m"] for e in episodes])),
        "mean_pitch_rms_deg": float(np.mean([e["pitch_rms_deg"] for e in episodes])),
        "mean_peak_abs_pitch_deg": float(np.mean([e["peak_abs_pitch_deg"] for e in episodes])),
        "mean_vx_rmse_mps": float(np.mean([e["vx_rmse_mps"] for e in episodes])),
        "mean_world_x_speed_mps": float(np.mean([e["mean_speed_mps"] for e in episodes])),
        "mean_contact_count": float(np.mean([e["mean_contact_count"] for e in episodes])),
      }
  return summaries


def plot(rows: list[dict], path: Path) -> None:
  fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
  colors = {"s0_bc": "#c26a34", "s0_1_bc": "#177c78", "flat_teacher": "#3568a8"}
  for policy, color in colors.items():
    subset = [r for r in rows if r["seed"] == 142 and r["policy"] == policy]
    for env_id in range(16):
      ep = [r for r in subset if r["env_id"] == env_id]
      if not ep:
        continue
      t = [r["time_s"] for r in ep]
      axs[0, 0].plot(t, [r["pitch_deg"] for r in ep], color=color, alpha=.24)
      axs[0, 1].plot(t, [r["world_x_velocity_mps"] for r in ep], color=color, alpha=.24)
      axs[1, 0].plot(t, [r["root_x_m"] for r in ep], color=color, alpha=.24)
      if env_id == 12:
        axs[1, 1].step(t, [r["left_contact"] for r in ep], where="post", color=color, label=f"{policy} L")
        axs[1, 1].step(t, [r["right_contact"] for r in ep], where="post", color=color, linestyle="--", label=f"{policy} R")
  axs[0, 0].set_title("Pitch, fixed seed 142")
  axs[0, 1].set_title("World X speed; command = 0.4 m/s")
  axs[1, 0].set_title("World X displacement")
  axs[1, 1].set_title("Foot contacts, env 12")
  for ax in axs.flat:
    ax.set_xlabel("time (s)")
    ax.grid(alpha=.25)
  axs[1, 1].legend(fontsize=8, ncol=2)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)
  dataset = collect_long_dataset(args.device)
  torch.save({k: v.cpu() for k, v in dataset.items()}, OUT / "flat_teacher_1s_dataset.pt")
  training = train(dataset, args.device)
  rows = []
  for seed in EVAL_SEEDS:
    for policy, ckpt in (("s0_bc", S0_BC), ("s0_1_bc", OUT / "v1a_s0_1_bc_model.pt"),
                         ("flat_teacher", s0.FLAT_SOURCE)):
      for row in s0.evaluate(policy, ckpt, seed, args.device):
        row["seed"] = seed
        rows.append(row)
  with (OUT / "s0_s01_flat_trajectories.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  comparisons = summarize(rows)
  result = {"training": training, "data": {"teacher": str(s0.FLAT_SOURCE), "seed": 812,
            "envs": s0.TRAIN_ENVS, "steps_per_env": DATA_STEPS, "horizon_s": 1.0,
            "input_shapes": {k: list(v.shape) for k, v in dataset.items()}},
            "comparisons": comparisons,
            "acceptance": {str(seed): {
              "all_16_survive_1s": comparisons[str(seed)]["s0_1_bc"]["survivors_1s"] == 16,
              "all_16_positive_x": comparisons[str(seed)]["s0_1_bc"]["positive_x_1s"] == 16,
              "pitch_improved_vs_s0": comparisons[str(seed)]["s0_1_bc"]["mean_pitch_rms_deg"] < comparisons[str(seed)]["s0_bc"]["mean_pitch_rms_deg"],
              "vx_rmse_improved_vs_s0": comparisons[str(seed)]["s0_1_bc"]["mean_vx_rmse_mps"] < comparisons[str(seed)]["s0_bc"]["mean_vx_rmse_mps"],
            } for seed in EVAL_SEEDS}}
  (OUT / "s0_1_result.json").write_text(json.dumps(result, indent=2) + "\n")
  plot(rows, OUT / "s0_s01_flat_curves.png")
  print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
  main()
