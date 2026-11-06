"""Three-second flat-approach diagnostics and teacher cloning for V1A."""

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
import train_g1_visual_stairs_v1a_s01 as s01


OUT = Path("logs/experiments/g1_visual_stairs_v1a_s1_20261104")
S01_BC = Path("logs/experiments/g1_visual_stairs_v1a_s01_20261104/v1a_s0_1_bc_model.pt")
SEEDS = (142, 314, 628)
STEPS = 150
DATA_ENVS = 64
SAFE_STAIR_MARGIN_M = 0.15
TRAIN_EPOCHS = 4
TRAIN_LR = 5e-5
BATCH = 512


def rollout(policy_name: str, checkpoint: Path, seed: int, device: str,
            steps: int = STEPS, n: int = 16) -> list[dict]:
  env, runner = s0.make_env(seed, n, device)
  try:
    policy_runner = runner if policy_name != "flat_teacher" else s0.make_runner(env, s0.FLAT, device)
    policy = s0.load_actor(policy_runner, checkpoint, device)
    obs = runner.env.get_observations()
    s0.set_command(env, obs)
    robot = env.scene["robot"]
    feet = env.scene["feet_ground_contact"]
    foot_ids, _ = robot.find_sites(("left_foot", "right_foot"))
    origins = s0.arr(env.scene.env_origins)
    initial_x = s0.arr(robot.data.root_link_pos_w)[:, 0].copy()
    rows: list[dict] = []
    active = np.ones(n, dtype=bool)
    step = 0

    def capture():
      pos = s0.arr(robot.data.root_link_pos_w)
      quat = s0.arr(robot.data.root_link_quat_w)
      w, x, y, z = quat.T
      pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
      roll = np.degrees(np.arcsin(np.clip(2 * (w * x + y * z), -1, 1)))
      tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
      vx = s0.arr(robot.data.root_link_lin_vel_w)[:, 0]
      site = s0.arr(robot.data.site_pos_w[:, foot_ids])
      distance = origins[:, 0] + 2.0 - site[:, :, 0].max(axis=1)
      found = s0.arr(feet.data.found[:, :2])
      target = s0.arr(robot.data.joint_pos_target)
      action = s0.arr(env.action_manager.get_term("joint_pos").raw_action)
      done = s0.arr(env.reset_buf).astype(bool)
      fallen = s0.arr(env.termination_manager.get_term("fell_over")).astype(bool)
      for i in range(n):
        if not active[i]:
          continue
        rows.append({"policy": policy_name, "seed": seed, "env_id": i,
                     "step": step, "time_s": round(step * env.step_dt, 6),
                     "pitch_deg": float(pitch[i]), "roll_deg": float(roll[i]),
                     "tilt_deg": float(tilt[i]), "root_x_m": float(pos[i, 0] - initial_x[i]),
                     "world_x_velocity_mps": float(vx[i]),
                     "stair_distance_m": float(distance[i]),
                     "left_contact": int(found[i, 0] > 0), "right_contact": int(found[i, 1] > 0),
                     "done": int(done[i]), "fell_over": int(fallen[i]),
                     "raw_action": json.dumps(action[i].tolist()),
                     "joint_target_rad": json.dumps(target[i].tolist())})
        if done[i]:
          active[i] = False

    original_metrics = env.metrics_manager.compute
    def hook():
      original_metrics()
      capture()
    env.metrics_manager.compute = hook
    for step in range(1, steps + 1):
      s0.set_command(env, obs)
      with torch.no_grad():
        action = policy(obs)
        obs, _, _, _ = runner.env.step(action)
      if not active.any():
        break
    return rows
  finally:
    env.close()


def collect_safe_teacher(device: str) -> dict[str, torch.Tensor]:
  env, runner = s0.make_env(821, DATA_ENVS, device)
  try:
    teacher_runner = s0.make_runner(env, s0.FLAT, device)
    teacher = s0.load_actor(teacher_runner, s0.FLAT_SOURCE, device)
    obs = runner.env.get_observations()
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_sites(("left_foot", "right_foot"))
    origins = s0.arr(env.scene.env_origins)
    keep = np.ones(DATA_ENVS, dtype=bool)
    buffers: dict[str, list[torch.Tensor]] = {k: [] for k in ("actor", "depth", "action")}
    lengths = np.zeros(DATA_ENVS, dtype=np.int32)
    for _ in range(STEPS):
      s0.set_command(env, obs)
      site = s0.arr(robot.data.site_pos_w[:, foot_ids])
      distance = origins[:, 0] + 2.0 - site[:, :, 0].max(axis=1)
      valid = keep & (distance > SAFE_STAIR_MARGIN_M)
      with torch.no_grad():
        action = teacher({"actor": obs["actor"]})
      ids = torch.as_tensor(np.flatnonzero(valid), device=device)
      if ids.numel():
        buffers["actor"].append(obs["actor"][ids].detach().clone())
        buffers["depth"].append(obs["depth"][ids].detach().clone())
        buffers["action"].append(action[ids].detach().clone())
        lengths[valid] += 1
      obs, _, done, _ = runner.env.step(action)
      keep &= ~s0.arr(done).astype(bool)
      keep &= distance > SAFE_STAIR_MARGIN_M
    if not buffers["action"]:
      raise RuntimeError("No safe Flat-teacher samples were collected before the stair margin")
    data = {k: torch.cat(v) for k, v in buffers.items()}
    if not all(torch.isfinite(t).all() for t in data.values()):
      raise RuntimeError("Non-finite safe teacher data")
    metadata = {"env_count": DATA_ENVS, "seed": 821, "requested_horizon_s": 3.0,
                "safe_margin_m": SAFE_STAIR_MARGIN_M, "sample_count": int(data["action"].shape[0]),
                "per_env_samples_min": int(lengths.min()), "per_env_samples_median": float(np.median(lengths)),
                "per_env_samples_max": int(lengths.max()),
                "accepted_sample_min_stair_distance_m": float(distance[keep].min()) if keep.any() else SAFE_STAIR_MARGIN_M}
    (OUT / "teacher_data_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return data
  finally:
    env.close()


def train(data: dict[str, torch.Tensor], device: str) -> dict:
  env, runner = s0.make_env(917, 1, device)
  try:
    runner.load(str(S01_BC), load_cfg={"actor": True}, strict=True, map_location=device)
    model = runner.alg.get_policy()
    model.train()
    model.obs_normalizer.eval()
    tensors = {k: v.to(device) for k, v in data.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    rng = torch.Generator(device=device).manual_seed(20261106)
    history = []
    for epoch in range(TRAIN_EPOCHS):
      order = torch.randperm(tensors["action"].shape[0], generator=rng, device=device)
      losses = []
      for ids in order.split(BATCH):
        pred = model({"actor": tensors["actor"][ids], "depth": tensors["depth"][ids]})
        loss = torch.nn.functional.mse_loss(pred, tensors["action"][ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
      history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
    checkpoint = torch.load(S01_BC, map_location="cpu", weights_only=False)
    checkpoint["actor_state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    checkpoint["infos"] = {"s1_behavior_cloned_from": str(s0.FLAT_SOURCE),
                           "initialized_from": str(S01_BC), "teacher_horizon_s": 3.0,
                           "safe_stair_margin_m": SAFE_STAIR_MARGIN_M,
                           "training_samples": int(tensors["action"].shape[0]), "updates": TRAIN_EPOCHS}
    target = OUT / "v1a_s1_bc_model.pt"
    torch.save(checkpoint, target)
    return {"epochs": history, "samples": int(tensors["action"].shape[0]), "batch_size": BATCH,
            "learning_rate": TRAIN_LR, "initialized_from": str(S01_BC), "model": str(target)}
  finally:
    env.close()


def collect_dagger(device: str) -> dict[str, torch.Tensor]:
  env, runner = s0.make_env(839, DATA_ENVS, device)
  try:
    student = s0.load_actor(runner, OUT / "v1a_s1_bc_model.pt", device)
    teacher_runner = s0.make_runner(env, s0.FLAT, device)
    teacher = s0.load_actor(teacher_runner, s0.FLAT_SOURCE, device)
    obs = runner.env.get_observations()
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_sites(("left_foot", "right_foot"))
    origins = s0.arr(env.scene.env_origins)
    active = np.ones(DATA_ENVS, dtype=bool)
    override = np.zeros(DATA_ENVS, dtype=bool)
    buffers = {k: [] for k in ("actor", "depth", "action")}
    lengths = np.zeros(DATA_ENVS, dtype=np.int32)
    min_distance = float("inf")
    override_count = 0
    for _ in range(STEPS):
      s0.set_command(env, obs)
      site = s0.arr(robot.data.site_pos_w[:, foot_ids])
      distance = origins[:, 0] + 2.0 - site[:, :, 0].max(axis=1)
      quat = s0.arr(robot.data.root_link_quat_w)
      _, x, y, _ = quat.T
      tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
      vx = s0.arr(robot.data.root_link_lin_vel_w)[:, 0]
      override |= (tilt >= 20.0) | (np.abs(vx) >= 0.7)
      valid = active & (distance > SAFE_STAIR_MARGIN_M)
      min_distance = min(min_distance, float(distance[valid].min())) if valid.any() else min_distance
      with torch.no_grad():
        label = teacher({"actor": obs["actor"]})
        learner_action = student(obs)
      ids = torch.as_tensor(np.flatnonzero(valid), device=device)
      if ids.numel():
        buffers["actor"].append(obs["actor"][ids].detach().clone())
        buffers["depth"].append(obs["depth"][ids].detach().clone())
        buffers["action"].append(label[ids].detach().clone())
        lengths[valid] += 1
      use_teacher = torch.as_tensor(override, device=device).unsqueeze(-1)
      action = torch.where(use_teacher, label, learner_action)
      obs, _, done, _ = runner.env.step(action)
      done_np = s0.arr(done).astype(bool)
      active &= ~done_np
      active &= distance > SAFE_STAIR_MARGIN_M
      override_count += int(override.sum())
    data = {k: torch.cat(v) for k, v in buffers.items()}
    metadata = {"env_count": DATA_ENVS, "seed": 839, "samples": int(data["action"].shape[0]),
                "per_env_samples_min": int(lengths.min()),
                "per_env_samples_median": float(np.median(lengths)),
                "per_env_samples_max": int(lengths.max()),
                "teacher_override_env_steps": override_count,
                "override_rule": "total tilt >= 20 deg or abs(world X speed) >= 0.7 m/s",
                "minimum_sample_stair_distance_m": min_distance}
    (OUT / "dagger_data_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return data
  finally:
    env.close()


def train_dagger(data: dict[str, torch.Tensor], device: str) -> dict:
  env, runner = s0.make_env(917, 1, device)
  try:
    source = OUT / "v1a_s1_bc_model.pt"
    runner.load(str(source), load_cfg={"actor": True}, strict=True, map_location=device)
    model = runner.alg.get_policy()
    model.train()
    model.obs_normalizer.eval()
    tensors = {k: v.to(device) for k, v in data.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    rng = torch.Generator(device=device).manual_seed(20261107)
    history = []
    for epoch in range(3):
      order = torch.randperm(tensors["action"].shape[0], generator=rng, device=device)
      losses = []
      for ids in order.split(BATCH):
        pred = model({"actor": tensors["actor"][ids], "depth": tensors["depth"][ids]})
        loss = torch.nn.functional.mse_loss(pred, tensors["action"][ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
      history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["actor_state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    checkpoint["infos"] = {"s1_dagger_behavior_cloned_from": str(s0.FLAT_SOURCE),
                           "initialized_from": str(source), "training_samples": int(tensors["action"].shape[0]),
                           "updates": 3, "dagger_safety_override": True}
    target = OUT / "v1a_s1a_dagger_bc_model.pt"
    torch.save(checkpoint, target)
    return {"epochs": history, "samples": int(tensors["action"].shape[0]),
            "learning_rate": 2e-5, "initialized_from": str(source), "model": str(target)}
  finally:
    env.close()


def summarize(rows: list[dict]) -> dict:
  result = {}
  for seed in SEEDS:
    result[str(seed)] = {}
    for policy in sorted({r["policy"] for r in rows}):
      subset = [r for r in rows if r["seed"] == seed and r["policy"] == policy]
      result[str(seed)][policy] = {}
      for horizon in (1, 2, 3):
        cutoff = horizon + 1e-7
        episodes = []
        for env_id in range(16):
          ep = [r for r in subset if r["env_id"] == env_id and r["time_s"] <= cutoff]
          survived = len(ep) >= horizon * 50 and not any(r["fell_over"] for r in ep)
          last = ep[-1]
          episodes.append({"env_id": env_id, "survived": survived,
                           "positive_x": last["root_x_m"] > 0,
                           "x_displacement_m": last["root_x_m"],
                           "pitch_rms_deg": float(np.sqrt(np.mean([r["pitch_deg"] ** 2 for r in ep]))),
                           "roll_rms_deg": float(np.sqrt(np.mean([r["roll_deg"] ** 2 for r in ep]))),
                           "peak_tilt_deg": float(max(r["tilt_deg"] for r in ep)),
                           "vx_rmse_mps": float(np.sqrt(np.mean([(r["world_x_velocity_mps"] - .4) ** 2 for r in ep]))),
                           "mean_world_x_velocity_mps": float(np.mean([r["world_x_velocity_mps"] for r in ep])),
                           "mean_stair_distance_m": float(np.mean([r["stair_distance_m"] for r in ep])),
                           "minimum_stair_distance_m": float(min(r["stair_distance_m"] for r in ep)),
                           "mean_contact_count": float(np.mean([r["left_contact"] + r["right_contact"] for r in ep]))})
        result[str(seed)][policy][f"{horizon}s"] = {
          "episodes": episodes,
          "survivors": sum(e["survived"] for e in episodes),
          "positive_x": sum(e["positive_x"] for e in episodes),
          **{f"mean_{key}": float(np.mean([e[key] for e in episodes])) for key in
             ("x_displacement_m", "pitch_rms_deg", "roll_rms_deg", "peak_tilt_deg", "vx_rmse_mps",
              "mean_world_x_velocity_mps", "mean_stair_distance_m", "mean_contact_count")},
        }
  return result


def plot(rows: list[dict], path: Path) -> None:
  fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
  colors = {"s0_1_bc": "#c16b34", "s1_bc": "#177b78", "flat_teacher": "#3568a8"}
  for policy, color in colors.items():
    subset = [r for r in rows if r["seed"] == 314 and r["policy"] == policy]
    for env_id in range(16):
      ep = [r for r in subset if r["env_id"] == env_id]
      if not ep:
        continue
      t = [r["time_s"] for r in ep]
      axs[0, 0].plot(t, [r["pitch_deg"] for r in ep], color=color, alpha=.23)
      axs[0, 1].plot(t, [r["roll_deg"] for r in ep], color=color, alpha=.23)
      axs[1, 0].plot(t, [r["world_x_velocity_mps"] for r in ep], color=color, alpha=.23)
      if env_id == 5:
        axs[1, 1].plot(t, [r["stair_distance_m"] for r in ep], color=color, label=policy)
  axs[0, 0].set_title("Pitch, seed 314")
  axs[0, 1].set_title("Roll, seed 314")
  axs[1, 0].set_title("World X velocity; target 0.4 m/s")
  axs[1, 1].set_title("Front-foot distance to first riser, env 05")
  for ax in axs.flat:
    ax.set_xlabel("time (s)")
    ax.grid(alpha=.25)
  axs[1, 1].legend(fontsize=8)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def write_rows(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def diagnose(device: str) -> list[dict]:
  rows = []
  for seed in SEEDS:
    for policy, checkpoint in (("s0_1_bc", S01_BC), ("flat_teacher", s0.FLAT_SOURCE)):
      rows.extend(rollout(policy, checkpoint, seed, device))
  write_rows(OUT / "pretraining_3s_diagnostics.csv", rows)
  # Teacher data are valid only if every teacher run remains upright before the stair margin.
  teacher = [r for r in rows if r["policy"] == "flat_teacher"]
  failures = [r for r in teacher if r["fell_over"]]
  min_distance = min(r["stair_distance_m"] for r in teacher)
  if failures or min_distance <= SAFE_STAIR_MARGIN_M:
    raise RuntimeError(f"Flat teacher is not safe for the requested three-second data horizon: falls={len(failures)}, min stair distance={min_distance:.3f} m")
  summaries = summarize(rows)
  (OUT / "pretraining_3s_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
  plot(rows, OUT / "pretraining_3s_seed314_curves.png")
  return rows


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--diagnose-only", action="store_true")
  parser.add_argument("--dagger-refine", action="store_true")
  args = parser.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)
  if args.dagger_refine:
    dagger_data = collect_dagger(args.device)
    torch.save({k: v.cpu() for k, v in dagger_data.items()}, OUT / "flat_teacher_dagger_dataset.pt")
    base_data = torch.load(OUT / "flat_teacher_safe_3s_dataset.pt", map_location="cpu", weights_only=True)
    combined = {k: torch.cat((base_data[k], dagger_data[k].cpu())) for k in base_data}
    training = train_dagger(combined, args.device)
    rows = []
    for seed in SEEDS:
      for policy, checkpoint in (("s0_1_bc", S01_BC), ("s1_bc", OUT / "v1a_s1_bc_model.pt"),
                                 ("s1a_dagger_bc", OUT / "v1a_s1a_dagger_bc_model.pt"),
                                 ("flat_teacher", s0.FLAT_SOURCE)):
        rows.extend(rollout(policy, checkpoint, seed, args.device))
    write_rows(OUT / "s1a_trajectories.csv", rows)
    result = {"training": training, "dagger_data": json.loads((OUT / "dagger_data_audit.json").read_text()),
              "comparisons": summarize(rows)}
    (OUT / "s1a_result.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(rows, OUT / "s1a_seed314_curves.png")
    print(json.dumps({seed: {p: {h: result["comparisons"][seed][p][h]["survivors"]
                                   for h in ("1s", "2s", "3s")}
                               for p in ("s0_1_bc", "s1_bc", "s1a_dagger_bc", "flat_teacher")}
                      for seed in map(str, SEEDS)}, indent=2))
    return
  diagnostic_rows = diagnose(args.device)
  if args.diagnose_only:
    print(json.dumps({"diagnostic_rows": len(diagnostic_rows), "output": str(OUT / "pretraining_3s_diagnostics.csv")}, indent=2))
    return
  data = collect_safe_teacher(args.device)
  torch.save({k: v.cpu() for k, v in data.items()}, OUT / "flat_teacher_safe_3s_dataset.pt")
  training = train(data, args.device)
  rows = []
  for seed in SEEDS:
    for policy, checkpoint in (("s0_1_bc", S01_BC), ("s1_bc", OUT / "v1a_s1_bc_model.pt"),
                               ("flat_teacher", s0.FLAT_SOURCE)):
      rows.extend(rollout(policy, checkpoint, seed, args.device))
  write_rows(OUT / "s1_trajectories.csv", rows)
  comparisons = summarize(rows)
  result = {"training": training,
            "teacher_data": json.loads((OUT / "teacher_data_audit.json").read_text()),
            "comparisons": comparisons,
            "acceptance": {str(seed): {
              f"s1_survive_{h}s": comparisons[str(seed)]["s1_bc"][f"{h}s"]["survivors"] == 16,
              f"s1_positive_{h}s": comparisons[str(seed)]["s1_bc"][f"{h}s"]["positive_x"] == 16,
            } for seed in SEEDS for h in (1, 2, 3)}}
  (OUT / "s1_result.json").write_text(json.dumps(result, indent=2) + "\n")
  plot(rows, OUT / "s1_seed314_curves.png")
  print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
  main()
