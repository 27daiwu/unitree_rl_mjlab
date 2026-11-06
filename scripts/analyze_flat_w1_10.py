"""Read-only W1.3 left/right lateral tracking attribution from saved traces."""

from pathlib import Path
import json

import numpy as np


SOURCE = Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2")
OUTPUT = Path("logs/flat_w1_10_attribution_20261030/metrics.json")
COMMANDS = {"left_020": 0.2, "right_020": -0.2}
FEET = ("left", "right")


def trace_metrics(path: Path, target: float) -> dict:
    with np.load(path) as data:
        keep = data["time"] > 7.0
        time = data["time"][keep]
        velocity = data["velocity"][keep].astype(np.float64)
        command = data["command"][keep]
        if len(time) != 250 or not np.allclose(command[:, 1], target):
            raise ValueError(f"Unexpected W1.3 diagnostic trace: {path}")

        error = velocity[:, 1] - target
        phase_index = np.rint(time / 0.02).astype(int) % 30
        # Predict each sample from the same gait phase in other cycles only.
        prediction = np.empty_like(error)
        for phase in range(30):
            mask = phase_index == phase
            values = error[mask]
            prediction[mask] = (values.sum() - values) / (len(values) - 1)
        bias = float(error.mean())
        std = float(error.std())
        rmse = float(np.sqrt(np.mean(error**2)))
        assert np.isclose(rmse**2, bias**2 + std**2)
        return {
            "axis_mean_m_s": float(velocity[:, 1].mean()),
            "axis_bias_m_s": bias,
            "axis_std_m_s": std,
            "axis_rmse_m_s": rmse,
            "bias_squared_fraction_of_mse": bias**2 / rmse**2,
            "phase_held_out_r2": float(1 - np.mean((error - prediction)**2) / np.var(error)),
            "axis_lag_30_correlation": float(np.corrcoef(error[:-30], error[30:])[0, 1]),
            "cross_axis_mean_m_s": float(velocity[:, 0].mean()),
            "cross_axis_rmse_m_s": float(np.sqrt(np.mean(velocity[:, 0]**2))),
            "yaw_mean_rad_s": float(velocity[:, 2].mean()),
            "yaw_rmse_rad_s": float(np.sqrt(np.mean(velocity[:, 2]**2))),
        }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    stored = json.loads((SOURCE / "metrics.json").read_text())["comparison"]
    result = {"source": str(SOURCE), "steady_state_time_s": [7.0, 12.0], "commands": {}}
    for name, target in COMMANDS.items():
        replicas = [trace_metrics(SOURCE / f"w1_3_{name}_{index}.npz", target)
                    for index in range(4)]
        metrics = {key: float(np.mean([replica[key] for replica in replicas]))
                   for key in replicas[0]}
        feet = stored[name]["w1_3"]["feet"]
        metrics["feet"] = {
            foot: {
                "step_length_m": feet[foot]["step_length_mean_m"],
                "swing_excursion_m": feet[foot]["swing_excursion_mean_m"],
            } for foot in FEET
        }
        metrics["completed_trials"] = stored[name]["w1_3"]["completed_trials"]
        result["commands"][name] = {"mean": metrics, "replicas": replicas}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
