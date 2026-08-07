#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np


# video2robot G1 23 DoF 关节顺序
G1_23_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]


# unitree_rl_mjlab G1 29 DoF 目标关节顺序
G1_29_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """归一化 xyzw 四元数。"""
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)

    bad_indices = np.where(norms[:, 0] < 1e-8)[0]
    if len(bad_indices) > 0:
        raise ValueError(
            "root_rot 中存在接近零模长的四元数，"
            f"前几个异常帧：{bad_indices[:10].tolist()}"
        )

    return quaternions / norms


def enforce_quaternion_continuity(
    quaternions: np.ndarray,
) -> np.ndarray:
    """
    修复相邻帧四元数符号跳变。

    q 和 -q 表示相同旋转，但直接计算速度时会产生异常尖峰。
    """
    result = quaternions.copy()

    for frame_index in range(1, len(result)):
        if np.dot(
            result[frame_index - 1],
            result[frame_index],
        ) < 0.0:
            result[frame_index] *= -1.0

    return result


def expand_23_to_29(dof_pos_23: np.ndarray) -> np.ndarray:
    """
    将 G1 23 DoF 数据映射为 G1 29 DoF。

    缺失的腰部和手腕关节使用 0。
    """
    if dof_pos_23.ndim != 2 or dof_pos_23.shape[1] != 23:
        raise ValueError(
            "expand_23_to_29 需要形状为 (N, 23) 的数组，"
            f"当前为 {dof_pos_23.shape}"
        )

    num_frames = dof_pos_23.shape[0]
    dof_pos_29 = np.zeros(
        (num_frames, 29),
        dtype=dof_pos_23.dtype,
    )

    source_indices = {
        joint_name: joint_index
        for joint_index, joint_name in enumerate(G1_23_DOF_NAMES)
    }

    for target_index, joint_name in enumerate(G1_29_DOF_NAMES):
        source_index = source_indices.get(joint_name)

        if source_index is not None:
            dof_pos_29[:, target_index] = dof_pos_23[:, source_index]

    return dof_pos_29


def validate_finite(name: str, array: np.ndarray) -> None:
    """检查数组是否包含 NaN 或 Inf。"""
    finite_mask = np.isfinite(array)

    if not finite_mask.all():
        bad_count = int(array.size - finite_mask.sum())
        raise ValueError(
            f"{name} 包含 {bad_count} 个 NaN 或 Inf"
        )


def load_motion(input_path: Path) -> dict:
    """读取 video2robot 输出的 PKL。"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    with input_path.open("rb") as file:
        motion = pickle.load(file)

    if not isinstance(motion, dict):
        raise TypeError(
            "当前脚本要求 PKL 顶层对象为 dict，"
            f"实际类型为 {type(motion)}"
        )

    required_keys = {
        "fps",
        "robot_type",
        "num_frames",
        "root_pos",
        "root_rot",
        "dof_pos",
    }

    missing_keys = required_keys - set(motion.keys())

    if missing_keys:
        raise KeyError(
            f"PKL 缺少必要字段：{sorted(missing_keys)}"
        )

    return motion


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "将 video2robot 输出的 Unitree G1 PKL "
            "转换为 unitree_rl_mjlab 使用的 CSV"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="输入 robot_motion_track_*.pkl 文件",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="输出 CSV 文件",
    )

    parser.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help=(
            "PKL 中 root_rot 的四元数顺序，"
            "video2robot 通常使用 xyzw，默认：xyzw"
        ),
    )

    args = parser.parse_args()

    motion = load_motion(args.input)

    robot_type = str(motion["robot_type"])
    fps = float(motion["fps"])
    declared_num_frames = int(motion["num_frames"])

    root_pos = np.asarray(
        motion["root_pos"],
        dtype=np.float64,
    )

    root_rot = np.asarray(
        motion["root_rot"],
        dtype=np.float64,
    )

    dof_pos = np.asarray(
        motion["dof_pos"],
        dtype=np.float64,
    )

    if robot_type != "unitree_g1":
        raise ValueError(
            "当前脚本只支持 robot_type=unitree_g1，"
            f"实际为 {robot_type}"
        )

    if fps <= 0:
        raise ValueError(f"fps 必须大于 0，当前为 {fps}")

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(
            "root_pos 应为 (N, 3)，"
            f"当前为 {root_pos.shape}"
        )

    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(
            "root_rot 应为 (N, 4)，"
            f"当前为 {root_rot.shape}"
        )

    if dof_pos.ndim != 2 or dof_pos.shape[1] not in (23, 29):
        raise ValueError(
            "dof_pos 应为 (N, 23) 或 (N, 29)，"
            f"当前为 {dof_pos.shape}"
        )

    actual_num_frames = root_pos.shape[0]

    if root_rot.shape[0] != actual_num_frames:
        raise ValueError(
            "root_pos 与 root_rot 帧数不一致："
            f"{root_pos.shape[0]} != {root_rot.shape[0]}"
        )

    if dof_pos.shape[0] != actual_num_frames:
        raise ValueError(
            "root_pos 与 dof_pos 帧数不一致："
            f"{root_pos.shape[0]} != {dof_pos.shape[0]}"
        )

    if declared_num_frames != actual_num_frames:
        raise ValueError(
            "PKL 中 num_frames 与数组实际帧数不一致："
            f"num_frames={declared_num_frames}，"
            f"实际={actual_num_frames}"
        )

    validate_finite("root_pos", root_pos)
    validate_finite("root_rot", root_rot)
    validate_finite("dof_pos", dof_pos)

    # 将输入四元数统一转成 CSV 需要的 xyzw 顺序
    if args.quat_order == "wxyz":
        root_rot_xyzw = root_rot[:, [1, 2, 3, 0]]
    else:
        root_rot_xyzw = root_rot.copy()

    root_rot_xyzw = normalize_quaternions(root_rot_xyzw)
    root_rot_xyzw = enforce_quaternion_continuity(
        root_rot_xyzw
    )

    if dof_pos.shape[1] == 29:
        dof_pos_29 = dof_pos.copy()
        conversion_mode = "原生 29 DoF，直接使用"
        inserted_zero_joints = []

    else:
        dof_pos_29 = expand_23_to_29(dof_pos)
        conversion_mode = "23 DoF 映射并补零为 29 DoF"

        inserted_zero_joints = [
            joint_name
            for joint_name in G1_29_DOF_NAMES
            if joint_name not in G1_23_DOF_NAMES
        ]

    # mjlab CSV：
    # root_pos(3) + root_quat_xyzw(4) + joint_pos(29)
    csv_data = np.concatenate(
        [
            root_pos,
            root_rot_xyzw,
            dof_pos_29,
        ],
        axis=1,
    )

    expected_shape = (actual_num_frames, 36)

    if csv_data.shape != expected_shape:
        raise RuntimeError(
            "CSV 数据形状异常："
            f"期望 {expected_shape}，实际 {csv_data.shape}"
        )

    validate_finite("csv_data", csv_data)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        args.output,
        csv_data,
        delimiter=",",
        fmt="%.10f",
    )

    duration = actual_num_frames / fps
    quaternion_norms = np.linalg.norm(
        root_rot_xyzw,
        axis=1,
    )

    print("转换完成")
    print(f"输入文件：{args.input}")
    print(f"输出文件：{args.output}")
    print(f"机器人类型：{robot_type}")
    print(f"帧数：{actual_num_frames}")
    print(f"输入帧率：{fps:g} Hz")
    print(f"动作时长：{duration:.3f} 秒")
    print(f"原始 dof_pos：{dof_pos.shape}")
    print(f"目标 dof_pos：{dof_pos_29.shape}")
    print(f"转换模式：{conversion_mode}")
    print(f"CSV shape：{csv_data.shape}")
    print("CSV 四元数顺序：xyzw")
    print(
        "四元数模长范围："
        f"{quaternion_norms.min():.8f} ~ "
        f"{quaternion_norms.max():.8f}"
    )

    if inserted_zero_joints:
        print("补零关节：")
        for joint_name in inserted_zero_joints:
            print(f"  - {joint_name}")
    else:
        print("补零关节：无")

    print("\nCSV 列格式：")
    print(
        "root_x, root_y, root_z, "
        "quat_x, quat_y, quat_z, quat_w, "
        "29 个 joint_pos"
    )

    print("\n下一步 csv_to_npz.py 参数：")
    print(f"--input-fps {fps:g}")


if __name__ == "__main__":
    main()