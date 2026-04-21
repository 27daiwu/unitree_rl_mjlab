# walk_env_cfg.py

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from .config.g1.env_cfgs import unitree_g1_flat_env_cfg
from . import curriculums as custom_curriculums
from . import observations as custom_observations
from . import rewards as custom_rewards


def _scale_dict_values(src: dict[str, float], scale: float) -> dict[str, float]:
    return {k: v * scale for k, v in src.items()}


def unitree_g1_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)

    # 1) 动作
    cfg.actions["joint_pos"].scale = 0.37

    # 2) 命令
    twist_cmd = cfg.commands["twist"]
    twist_cmd.resampling_time_range = (5.5, 8.5)
    twist_cmd.rel_standing_envs = 0.10
    twist_cmd.rel_heading_envs = 0.22
    twist_cmd.heading_control_stiffness = 0.50

    twist_cmd.ranges.lin_vel_x = (-0.40, 0.70)
    twist_cmd.ranges.lin_vel_y = (-0.18, 0.18)
    twist_cmd.ranges.ang_vel_z = (-0.55, 0.55)

    # 3) 观测
    feet_asset_cfg = cfg.rewards["foot_slip"].params["asset_cfg"]

    cfg.observations["policy"].terms["foot_contact"] = ObservationTermCfg(
        func=custom_observations.foot_contact,
        params={"sensor_name": "feet_ground_contact"},
    )
    cfg.observations["policy"].terms["foot_height"] = ObservationTermCfg(
        func=custom_observations.foot_height,
        params={"asset_cfg": feet_asset_cfg},
    )

    cfg.observations["policy"].history_length = 4

    # 4) 奖励
    cfg.rewards["track_linear_velocity"].weight = 1.00
    cfg.rewards["track_angular_velocity"].weight = 1.00

    cfg.rewards["flat_orientation_l2"].weight = -10.5

    cfg.rewards["action_rate_l2"].weight = -0.07
    cfg.rewards["joint_acc_l2"].weight = -7.0e-7

    cfg.rewards["soft_landing"].weight = -6.0e-3

    cfg.rewards["foot_clearance"].weight = -1.40
    cfg.rewards["foot_clearance"].params["target_height"] = 0.11
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.04

    cfg.rewards["foot_slip"].weight = -1.35
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.02

    cfg.rewards["feet_air_time"] = RewardTermCfg(
        func=custom_rewards.feet_air_time,
        weight=0.16,
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.07,
            "threshold_max": 0.28,
            "command_name": "twist",
            "command_threshold": 0.04,
        },
    )

    pose_cfg = cfg.rewards["pose"]
    pose_cfg.weight = 1.05
    pose_cfg.params["walking_threshold"] = 0.06
    pose_cfg.params["running_threshold"] = 0.90
    pose_cfg.params["std_standing"] = _scale_dict_values(
        pose_cfg.params["std_standing"], 0.85
    )
    pose_cfg.params["std_walking"] = _scale_dict_values(
        pose_cfg.params["std_walking"], 0.97
    )
    pose_cfg.params["std_running"] = _scale_dict_values(
        pose_cfg.params["std_running"], 1.00
    )

    # 5) 随机化
    if not play:
        if "body_friction" in cfg.events:
            cfg.events["body_friction"].params["ranges"] = (0.80, 1.25)

        if "encoder_bias" in cfg.events:
            cfg.events["encoder_bias"].params["bias_range"] = (-0.008, 0.008)

        if "push_robot" in cfg.events:
            cfg.events["push_robot"].interval_range_s = (5.0, 8.0)
            cfg.events["push_robot"].params["velocity_range"] = {
                "x": (-0.12, 0.12),
                "y": (-0.10, 0.10),
                "z": (-0.05, 0.05),
                "roll": (-0.12, 0.12),
                "pitch": (-0.12, 0.12),
                "yaw": (-0.18, 0.18),
            }

    # 6) Curriculum：命令
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=custom_curriculums.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (-0.20, 0.40),
                    "lin_vel_y": (-0.10, 0.10),
                    "ang_vel_z": (-0.25, 0.25),
                },
                {
                    "step": 4000 * 24,
                    "lin_vel_x": (-0.35, 0.55),
                    "lin_vel_y": (-0.15, 0.15),
                    "ang_vel_z": (-0.40, 0.40),
                },
                {
                    "step": 10000 * 24,
                    "lin_vel_x": (-0.40, 0.70),
                    "lin_vel_y": (-0.20, 0.20),
                    "ang_vel_z": (-0.55, 0.55),
                },
            ],
        },
    )

    # 7) Curriculum：脚滑惩罚
    cfg.curriculum["foot_slip_weight"] = CurriculumTermCfg(
        func=custom_curriculums.reward_weight,
        params={
            "reward_name": "foot_slip",
            "weight_stages": [
                {"step": 0, "weight": -1.00},
                {"step": 4000 * 24, "weight": -1.20},
                {"step": 10000 * 24, "weight": -1.35},
            ],
        },
    )

    # 8) Curriculum：角速度跟踪，后期更重视转向质量
    cfg.curriculum["track_angular_velocity_weight"] = CurriculumTermCfg(
        func=custom_curriculums.reward_weight,
        params={
            "reward_name": "track_angular_velocity",
            "weight_stages": [
                {"step": 0, "weight": 0.75},
                {"step": 4000 * 24, "weight": 0.90},
                {"step": 10000 * 24, "weight": 1.00},
            ],
        },
    )

    # 9) Curriculum：抗扰动逐步增强
    # 这两个需要你的 curriculum 函数支持 event 参数修改；如果你那边没有，
    # 先保留下面注释中的思路，后续我可以继续帮你补 custom_curriculums。
    # cfg.curriculum["push_strength"] = ...
    # cfg.curriculum["friction_range"] = ...

    if play:
        cfg.events.pop("push_robot", None)
        cfg.events.pop("body_friction", None)
        cfg.events.pop("encoder_bias", None)

    return cfg


# python scripts/train.py Mjlab-Walk-Unitree-G1 \
#   --env.scene.num-envs=4096 \
#   --agent.resume=True \
#   --agent.load_run=2026-04-21_10-08-05 \
#   --agent.load_checkpoint=model_10000.pt \
#   --agent.max_iterations=30000
