"""Offline W1.11 reward and matched-training-config preflight."""

from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import yaml

from train_flat_w1_11 import SOURCE, SOURCE_SHA256, prepare


OUT = Path("logs/flat_w1_11_right_track_20261030/reward_preflight.json")
DIAGNOSTICS = Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2")
HISTORICAL_ENV = Path("logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/params/env.yaml")


def main():
  if OUT.exists():
    raise FileExistsError(OUT)
  with SOURCE.open("rb") as source_file:
    source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
  assert source_hash == SOURCE_SHA256
  control_env, control_agent = prepare("control")
  experiment_env, experiment_agent = prepare("experiment")
  control_yaml = yaml.dump(asdict(control_env), sort_keys=False)
  assert control_yaml == HISTORICAL_ENV.read_text()
  experiment_data = asdict(experiment_env)
  assert experiment_data["rewards"]["track_linear_velocity"]["params"].pop("right_y_error_scale") == 2.0
  assert yaml.dump(experiment_data, sort_keys=False) == control_yaml
  control_agent_data, experiment_agent_data = asdict(control_agent), asdict(experiment_agent)
  experiment_agent_data["run_name"] = control_agent_data["run_name"]
  assert yaml.dump(experiment_agent_data, sort_keys=False) == yaml.dump(control_agent_data, sort_keys=False)

  cases = {}
  for name in ("forward_020", "backward_015", "left_020", "right_020"):
    control_rewards = []
    experiment_rewards = []
    for replica in range(4):
      with np.load(DIAGNOSTICS / f"w1_3_{name}_{replica}.npz") as data:
        keep = data["time"] > 7.0
        command = data["command"][keep]
        actual = data["velocity"][keep].astype(np.float64)
        error = command[:, :2] - actual[:, :2]
        # The trace stores yaw in velocity[:, 2]; use the recorded original reward.
        baseline = data["reward_components"][keep, 0].astype(np.float64)
        modified = baseline * np.exp(-np.where(command[:, 1] < 0, 1.0, 0.0) *
                                     error[:, 1] ** 2 / .25)
        assert np.all(np.isfinite(modified)) and np.all((modified >= 0) & (modified <= 1))
        assert np.all(modified <= baseline)
        if not np.any(command[:, 1] < 0):
          assert np.array_equal(modified, baseline)
        control_rewards.append(float(baseline.mean()))
        experiment_rewards.append(float(modified.mean()))
    cases[name] = {"recorded_control_reward_mean": float(np.mean(control_rewards)),
                   "counterfactual_experiment_reward_mean": float(np.mean(experiment_rewards))}
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps({"source_sha256": source_hash, "control_env_matches_w1_3": True,
                             "only_experiment_env_difference": "right_y_error_scale=2.0",
                             "only_agent_difference": "run_name", "historical_trace_rewards": cases},
                            indent=2) + "\n")
  print(OUT)


if __name__ == "__main__":
  main()
