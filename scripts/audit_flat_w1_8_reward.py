"""Preflight the W1.8 smooth toe target against saved traces and configs."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile

import numpy as np
import torch

from mjlab.utils.os import dump_yaml
from train_flat_w1_8 import SOURCE, prepare
from src.tasks.velocity.mdp.rewards import flat_toe_clearance


ROOT = Path("logs/flat_w1_8_smooth_toe_20261030")
SOURCES = {
    "w1_3": (Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2"), "w1_3"),
    "w1_6": (Path("logs/flat_w1_6_toe_clearance_20261030/paired_diagnostics"), "experiment"),
}
SCENARIOS = ("forward_020", "forward_040", "backward_015", "left_020", "right_020",
             "turn_left_050", "turn_right_050")


def run():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    control_env, control_rl = prepare("control")
    experiment_env, experiment_rl = prepare("experiment")
    control = asdict(control_env)
    experiment = asdict(experiment_env)
    assert experiment["rewards"]["foot_clearance"]["params"].pop("target_shape") == "smooth"
    assert control == experiment
    assert {k: v for k, v in asdict(control_rl).items() if k != "run_name"} == {
        k: v for k, v in asdict(experiment_rl).items() if k != "run_name"}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "env.yaml"
        dump_yaml(path, asdict(control_env))
        assert path.read_bytes() == Path(
            "logs/flat_w1_6_toe_clearance_20261030/experiment/params/env.yaml").read_bytes()

    cfg = SimpleNamespace(params={"asset_cfg": SimpleNamespace(name="robot", site_ids=[0, 1])})
    term = flat_toe_clearance(cfg, SimpleNamespace(device="cpu"))
    args = dict(target_height=0.03, period=0.6, offset=[0.0, 0.5],
                swing_phase_start=0.66, swing_phase_end=0.90,
                command_name="twist", command_threshold=0.1,
                asset_cfg=cfg.params["asset_cfg"])
    rows = {}
    for source, (directory, label) in SOURCES.items():
        rows[source] = {}
        for scenario in SCENARIOS:
            samples = []
            for replica in range(4):
                with np.load(directory / f"{label}_{scenario}_{replica}.npz") as z:
                    keep = z["time"] > 7.0
                    time = z["time"][keep]
                    steps = torch.from_numpy(np.rint(time / 0.02).astype(np.int64))
                    robot = SimpleNamespace(data=SimpleNamespace(
                        site_pos_w=torch.from_numpy(z["foot_position"][keep]),
                        site_quat_w=torch.from_numpy(z["foot_quaternion"][keep])))
                    env = SimpleNamespace(scene={"robot": robot}, device="cpu", step_dt=0.02,
                                          episode_length_buf=steps,
                                          command_manager=SimpleNamespace(
                                              get_command=lambda _, z=z, keep=keep: torch.from_numpy(z["command"][keep])))
                    rectangle = term(env, **args).detach().numpy()
                    smooth = term(env, **args, target_shape="smooth").detach().numpy()
                    phase = (np.rint(time[:, None] / 0.02) / 30 + np.array([0.0, 0.5])[None]) % 1.0
                    mid = (phase > 0.66 + 1e-6) & (phase < 0.90 - 1e-6)
                    contact = (z["contact_time"][keep] > 0) | (z["normal_force"][keep] >= 5)
                    assert np.all(smooth >= -1e-9) and np.all(smooth <= rectangle + 1e-7)
                    samples.append(dict(rectangle_mean=float(rectangle.mean()),
                                        smooth_mean=float(smooth.mean()),
                                        rectangle_max=float(rectangle.max()),
                                        smooth_max=float(smooth.max()),
                                        planned_samples=int(mid.sum()),
                                        contact_in_planned=int((mid & contact).sum())))
            rows[source][scenario] = {
                key: float(np.mean([sample[key] for sample in samples]))
                for key in ("rectangle_mean", "smooth_mean", "rectangle_max", "smooth_max")}
            rows[source][scenario]["planned_samples"] = sum(sample["planned_samples"] for sample in samples)
            rows[source][scenario]["contact_in_planned"] = sum(
                sample["contact_in_planned"] for sample in samples)
    with SOURCE.open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    output = {"source_sha256": source_hash, "control_matches_w1_6_env_yaml": True,
              "config_difference": "foot_clearance.params.target_shape only", "traces": rows}
    ROOT.mkdir(parents=True)
    (ROOT / "reward_preflight.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run()
