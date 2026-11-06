"""Audit the proposed Flat toe reward on saved W1.3-D traces."""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch

from src.tasks.velocity.mdp.rewards import flat_toe_clearance


SOURCE = Path("logs/flat_w1_4_min_reward_20261030/baseline_diagnostics_v2")
OUT = Path("logs/flat_w1_6_toe_clearance_20261030")
REPORT = OUT / "reward_preflight_v2.json"


def main():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    cfg = SimpleNamespace(params={"asset_cfg": SimpleNamespace(name="robot", site_ids=[0, 1])})
    init_env = SimpleNamespace(device="cpu")
    term = flat_toe_clearance(cfg, init_env)
    reward_names = list(json.loads((SOURCE / "metrics.json").read_text())["comparison"]["backward_015"]["w1_3"]["reward_component_weighted_mean_per_s"])
    old_index = reward_names.index("foot_clearance")
    rows = {}
    for scenario in ("forward_020", "forward_040", "backward_015", "left_020", "right_020", "turn_left_050", "turn_right_050"):
        samples = []
        for replica in range(4):
            with np.load(SOURCE / f"w1_3_{scenario}_{replica}.npz") as z:
                keep = z["time"] > 7.0
                pos = torch.from_numpy(z["foot_position"][keep])
                quat = torch.from_numpy(z["foot_quaternion"][keep])
                time = z["time"][keep]
                steps = torch.from_numpy(np.rint(time / 0.02).astype(np.int64))
                command = torch.from_numpy(z["command"][keep])
                contact = (z["contact_time"][keep] > 0) | (z["normal_force"][keep] >= 5)
                robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=pos, site_quat_w=quat))
                env = SimpleNamespace(scene={"robot": robot}, device="cpu", step_dt=0.02,
                                      episode_length_buf=steps,
                                      command_manager=SimpleNamespace(get_command=lambda _: command))
                args = dict(target_height=0.03, period=0.6, offset=[0.0, 0.5],
                            swing_phase_start=0.66, swing_phase_end=0.90,
                            command_name="twist", command_threshold=0.1,
                            asset_cfg=cfg.params["asset_cfg"])
                new = term(env, **args).detach().numpy()
                phase = (time[:, None] / 0.6 + np.array([0.0, 0.5])[None]) % 1.0
                mid = (phase > 0.66 + 1e-6) & (phase < 0.90 - 1e-6)
                samples.append({"old_mean": float(-z["reward_components"][keep, old_index].mean()),
                                "new_mean": float(new.mean()),
                                "new_max": float(new.max()),
                                "planned_mid_swing_samples": int(mid.sum()),
                                "contact_in_planned_mid_swing": int((mid & contact).sum())})
        rows[scenario] = {key: float(np.mean([row[key] for row in samples]))
                          for key in ("old_mean", "new_mean", "new_max")}
        rows[scenario]["planned_mid_swing_samples"] = sum(row["planned_mid_swing_samples"] for row in samples)
        rows[scenario]["contact_in_planned_mid_swing"] = sum(row["contact_in_planned_mid_swing"] for row in samples)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
