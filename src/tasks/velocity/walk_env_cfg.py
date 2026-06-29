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


def _set_reward_weight(cfg: ManagerBasedRlEnvCfg, name: str, weight: float) -> None:
    if name in cfg.rewards:
        cfg.rewards[name].weight = weight


def _set_reward_param(cfg: ManagerBasedRlEnvCfg, name: str, key: str, value) -> None:
    if name in cfg.rewards and hasattr(cfg.rewards[name], "params"):
        cfg.rewards[name].params[key] = value


def unitree_g1_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)

    # 目标：修复 60000 轮后仍存在的后退摇晃、转向僵硬/非原地、速度突变后持续晃动。
    # 核心思路：不要继续单纯加大脚滑惩罚；改为提高转向/后退样本密度、缩短命令重采样、加强停稳约束。

    # 1) 动作幅值：略小于上一版 0.36，降低速度突变时的髋/踝过冲，但不要压到 0.30 以下。
    cfg.actions["joint_pos"].scale = 0.35

    # 2) 命令分布：缩短重采样时间，让策略真正学习加减速和刹停过渡。
    twist_cmd = cfg.commands["twist"]
    twist_cmd.resampling_time_range = (2.5, 4.0)
    twist_cmd.rel_standing_envs = 0.30
    twist_cmd.rel_heading_envs = 0.10
    twist_cmd.heading_control_stiffness = 0.65

    # 比上一版放开后退和角速度，否则会学成“慢转 + 姿态补偿位移”。
    twist_cmd.ranges.lin_vel_x = (-0.35, 0.70)
    twist_cmd.ranges.lin_vel_y = (-0.18, 0.18)
    twist_cmd.ranges.ang_vel_z = (-0.75, 0.75)

    # 3) 观测：增加短历史，帮助策略根据 command / action / IMU 历史推断速度变化趋势。
    feet_asset_cfg = cfg.rewards["foot_slip"].params["asset_cfg"]

    for group_name in ("policy", "actor"):
        if group_name not in cfg.observations:
            continue
        cfg.observations[group_name].terms["foot_contact"] = ObservationTermCfg(
            func=custom_observations.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        )
        cfg.observations[group_name].terms["foot_height"] = ObservationTermCfg(
            func=custom_observations.foot_height,
            params={"asset_cfg": feet_asset_cfg},
        )
        cfg.observations[group_name].history_length = 6

    # 4) 奖励：转向和刹停分开处理。
    #    - 角速度跟踪更强，减少“转不起来”。
    #    - 姿态/角动量约束适中，减少速度突变后的上身晃动。
    #    - 动作平滑不要太重，否则转向响应会继续慢。
    _set_reward_weight(cfg, "track_linear_velocity", 1.08)
    _set_reward_weight(cfg, "track_angular_velocity", 1.15)
    _set_reward_param(cfg, "track_angular_velocity", "std", 0.55)

    _set_reward_weight(cfg, "flat_orientation_l2", -10.5)
    _set_reward_weight(cfg, "body_orientation_l2", -1.10)
    _set_reward_weight(cfg, "body_ang_vel", -0.07)
    _set_reward_weight(cfg, "angular_momentum", -0.035)

    _set_reward_weight(cfg, "action_rate_l2", -0.055)
    _set_reward_weight(cfg, "joint_acc_l2", -6.0e-7)
    _set_reward_weight(cfg, "soft_landing", -8.0e-3)

    # 后退脚尖/脚跟更容易蹭地，略提高目标脚高，但脚滑惩罚仍采用渐进增强。
    _set_reward_weight(cfg, "foot_clearance", -1.25)
    _set_reward_param(cfg, "foot_clearance", "target_height", 0.105)
    _set_reward_param(cfg, "foot_clearance", "command_threshold", 0.06)

    _set_reward_weight(cfg, "foot_slip", -1.70)
    _set_reward_param(cfg, "foot_slip", "command_threshold", 0.04)

    if "foot_gait" in cfg.rewards:
        cfg.rewards["foot_gait"].weight = 0.55
        cfg.rewards["foot_gait"].params["command_threshold"] = 0.06
        cfg.rewards["foot_gait"].params["threshold"] = 0.55

    # 摇杆归零后乱挪步，本质上需要 stand_still 与低速阈值更明确。
    _set_reward_weight(cfg, "stand_still", -1.45)
    _set_reward_param(cfg, "stand_still", "command_threshold", 0.07)

    cfg.rewards["feet_air_time"] = RewardTermCfg(
        func=custom_rewards.feet_air_time,
        weight=0.16,
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.08,
            "threshold_max": 0.24,
            "command_name": "twist",
            "command_threshold": 0.06,
        },
    )

    # 姿态先别压太死：过强 pose 会导致原地转向时靠髋/踝硬扭，产生 XY 漂移。
    pose_cfg = cfg.rewards["pose"]
    pose_cfg.weight = 1.15
    pose_cfg.params["walking_threshold"] = 0.08
    pose_cfg.params["running_threshold"] = 0.90
    pose_cfg.params["std_standing"] = _scale_dict_values(
        pose_cfg.params["std_standing"], 0.78
    )
    pose_cfg.params["std_walking"] = _scale_dict_values(
        pose_cfg.params["std_walking"], 1.06
    )
    pose_cfg.params["std_running"] = _scale_dict_values(
        pose_cfg.params["std_running"], 1.05
    )

    # 5) 随机化：保留温和扰动，不能完全关掉；否则刹停和后退到真机上会更脆。
    if not play:
        # 兼容不同版本命名：官方常见是 foot_friction，你当前代码里用了 body_friction。
        for friction_event_name in ("foot_friction", "body_friction"):
            if friction_event_name in cfg.events:
                cfg.events[friction_event_name].params["ranges"] = (0.80, 1.25)

        if "encoder_bias" in cfg.events:
            cfg.events["encoder_bias"].params["bias_range"] = (-0.006, 0.006)

        if "base_com" in cfg.events:
            cfg.events["base_com"].params["ranges"] = {
                0: (-0.025, 0.025),
                1: (-0.025, 0.025),
                2: (-0.020, 0.020),
            }

        if "push_robot" in cfg.events:
            cfg.events["push_robot"].interval_range_s = (5.5, 8.0)
            cfg.events["push_robot"].params["velocity_range"] = {
                "x": (-0.10, 0.10),
                "y": (-0.08, 0.08),
                "z": (-0.04, 0.04),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.16, 0.16),
            }

    # 6) Curriculum：从你的 60000 轮 checkpoint 续训时，先回到中等难度适应新奖励，
    #    然后再逐步放开后退和大角速度。这样比直接全范围续训更稳。
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=custom_curriculums.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (-0.18, 0.45),
                    "lin_vel_y": (-0.08, 0.08),
                    "ang_vel_z": (-0.55, 0.55),
                },
                {
                    "step": 3000 * 24,
                    "lin_vel_x": (-0.25, 0.60),
                    "lin_vel_y": (-0.12, 0.12),
                    "ang_vel_z": (-0.65, 0.65),
                },
                {
                    "step": 8000 * 24,
                    "lin_vel_x": (-0.32, 0.68),
                    "lin_vel_y": (-0.16, 0.16),
                    "ang_vel_z": (-0.75, 0.75),
                },
                {
                    "step": 16000 * 24,
                    "lin_vel_x": (-0.38, 0.75),
                    "lin_vel_y": (-0.20, 0.20),
                    "ang_vel_z": (-0.85, 0.85),
                },
            ],
        },
    )

    cfg.curriculum["foot_slip_weight"] = CurriculumTermCfg(
        func=custom_curriculums.reward_weight,
        params={
            "reward_name": "foot_slip",
            "weight_stages": [
                {"step": 0, "weight": -1.35},
                {"step": 3000 * 24, "weight": -1.50},
                {"step": 8000 * 24, "weight": -1.60},
                {"step": 16000 * 24, "weight": -1.70},
            ],
        },
    )

    cfg.curriculum["track_angular_velocity_weight"] = CurriculumTermCfg(
        func=custom_curriculums.reward_weight,
        params={
            "reward_name": "track_angular_velocity",
            "weight_stages": [
                {"step": 0, "weight": 0.95},
                {"step": 3000 * 24, "weight": 1.05},
                {"step": 8000 * 24, "weight": 1.15},
                {"step": 16000 * 24, "weight": 1.20},
            ],
        },
    )

    if "stand_still" in cfg.rewards:
        cfg.curriculum["stand_still_weight"] = CurriculumTermCfg(
            func=custom_curriculums.reward_weight,
            params={
                "reward_name": "stand_still",
                "weight_stages": [
                    {"step": 0, "weight": -1.15},
                    {"step": 3000 * 24, "weight": -1.30},
                    {"step": 8000 * 24, "weight": -1.40},
                    {"step": 16000 * 24, "weight": -1.45},
                ],
            },
        )

    if play:
        cfg.events.pop("push_robot", None)
        cfg.events.pop("foot_friction", None)
        cfg.events.pop("body_friction", None)
        cfg.events.pop("encoder_bias", None)
        cfg.events.pop("base_com", None)

    return cfg

# python scripts/train.py Mjlab-Walk-Unitree-G1 \
#   --env.scene.num-envs=4096 \
#   --agent.resume=True \
#   --agent.load_run=2026-05-04_23-18-28 \
#   --agent.load_checkpoint=model_79999.pt \
#   --agent.max_iterations=30000