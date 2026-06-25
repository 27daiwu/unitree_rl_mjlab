#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NPZ 动作播放器 + 裁剪器（MuJoCo 版）

功能：
1. 读取并检查 NPZ 字段
2. 使用 MuJoCo 模型播放机器人动作
3. 暂停、逐帧、调速、循环播放
4. 设置裁剪起点/终点
5. 同步裁剪所有逐帧字段并保存新 NPZ
6. 可选：裁剪后重新计算常见速度字段

依赖：
    pip install numpy mujoco glfw

示例：
    python scripts/npz_player_cropper.py \
        --input mjlab/motions/g1/bencaogangmu_bencaogangmu.npz \
        --output mjlab/motions/g1/bencaogangmu_crop.npz \
        --model-xml mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml
        --no-loop

如果 NPZ 字段无法自动识别，可手动指定：
    python npz_player_cropper.py \
        --input motion.npz \
        --model-xml g1.xml \
        --joint-key joint_pos \
        --root-pos-key root_pos \
        --root-quat-key root_quat \
        --fps 30

快捷键：
    Space       播放/暂停
    Left/Right  前一帧/后一帧
    A / D       后退/前进 10 帧
    J / L       后退/前进 1 秒
    [ / ]       降低/提高播放速度
    I           将当前帧设为裁剪起点
    O           将当前帧设为裁剪终点（包含当前帧）
    C           清除裁剪区间
    S           保存裁剪后的 NPZ
    R           回到第 0 帧
    Home/End    跳到首帧/末帧
    H           打印帮助
    Esc         退出
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit(
        "未安装 mujoco。请执行：pip install mujoco glfw"
    ) from exc


JOINT_KEY_CANDIDATES = (
    "joint_pos",
    "joint_positions",
    "dof_pos",
    "qpos",
    "position",
    "positions",
)

ROOT_POS_KEY_CANDIDATES = (
    "root_pos",
    "root_position",
    "base_pos",
    "base_position",
    "root_trans",
    "trans",
)

ROOT_QUAT_KEY_CANDIDATES = (
    "root_quat",
    "root_quaternion",
    "base_quat",
    "base_quaternion",
    "root_rot",
)

FPS_KEY_CANDIDATES = (
    "fps",
    "frame_rate",
    "framerate",
    "motion_fps",
    "frequency",
)

TIME_KEY_CANDIDATES = (
    "time",
    "times",
    "timestamp",
    "timestamps",
)

JOINT_VEL_KEYS = (
    "joint_vel",
    "joint_velocity",
    "joint_velocities",
    "dof_vel",
)

ROOT_LIN_VEL_KEYS = (
    "root_lin_vel",
    "root_linear_velocity",
    "base_lin_vel",
)

ROOT_ANG_VEL_KEYS = (
    "root_ang_vel",
    "root_angular_velocity",
    "base_ang_vel",
)


def first_existing(data: dict[str, np.ndarray], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in data:
            return name
    return None


def scalar_value(value: np.ndarray) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("空数组无法转换为标量")
    return float(arr.reshape(-1)[0])


def normalize_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return q[[3, 0, 1, 2]]


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_delta_to_angular_velocity(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    q0 = normalize_quaternion_wxyz(q0)
    q1 = normalize_quaternion_wxyz(q1)

    # 防止四元数符号跳变。
    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    dq = quat_multiply_wxyz(q1, quat_conjugate_wxyz(q0))
    dq = normalize_quaternion_wxyz(dq)

    w = float(np.clip(dq[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))

    if s < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float64)

    axis = dq[1:] / s
    return axis * (angle / dt)


class MotionData:
    def __init__(self, path: Path, args: argparse.Namespace):
        self.path = path

        with np.load(path, allow_pickle=True) as src:
            self.data = {key: src[key] for key in src.files}

        if not self.data:
            raise ValueError(f"NPZ 文件为空：{path}")

        self.joint_key = args.joint_key or first_existing(
            self.data, JOINT_KEY_CANDIDATES
        )
        if self.joint_key is None:
            raise ValueError(
                "无法识别关节位置字段。请使用 --joint-key 指定。\n"
                f"现有字段：{list(self.data.keys())}"
            )

        self.joint_pos = np.asarray(self.data[self.joint_key])
        if self.joint_pos.ndim != 2:
            raise ValueError(
                f"{self.joint_key} 应为二维数组 (帧数, 关节数)，"
                f"实际 shape={self.joint_pos.shape}"
            )

        self.num_frames = int(self.joint_pos.shape[0])
        self.num_joints = int(self.joint_pos.shape[1])

        self.root_pos_key = args.root_pos_key or first_existing(
            self.data, ROOT_POS_KEY_CANDIDATES
        )
        self.root_quat_key = args.root_quat_key or first_existing(
            self.data, ROOT_QUAT_KEY_CANDIDATES
        )

        self.root_pos = (
            np.asarray(self.data[self.root_pos_key])
            if self.root_pos_key is not None
            else None
        )
        self.root_quat = (
            np.asarray(self.data[self.root_quat_key])
            if self.root_quat_key is not None
            else None
        )

        if self.root_pos is not None:
            if self.root_pos.ndim != 2 or self.root_pos.shape[1] < 3:
                raise ValueError(
                    f"{self.root_pos_key} 应为 (T,3)，实际 {self.root_pos.shape}"
                )
            if self.root_pos.shape[0] != self.num_frames:
                raise ValueError(
                    f"{self.root_pos_key} 帧数与 {self.joint_key} 不一致"
                )

        if self.root_quat is not None:
            if self.root_quat.ndim != 2 or self.root_quat.shape[1] < 4:
                raise ValueError(
                    f"{self.root_quat_key} 应为 (T,4)，实际 {self.root_quat.shape}"
                )
            if self.root_quat.shape[0] != self.num_frames:
                raise ValueError(
                    f"{self.root_quat_key} 帧数与 {self.joint_key} 不一致"
                )

        self.fps = self._detect_fps(args.fps)
        self.frame_dt = 1.0 / self.fps

    def _detect_fps(self, cli_fps: float | None) -> float:
        if cli_fps is not None:
            if cli_fps <= 0:
                raise ValueError("--fps 必须大于 0")
            return float(cli_fps)

        fps_key = first_existing(self.data, FPS_KEY_CANDIDATES)
        if fps_key is not None:
            value = scalar_value(self.data[fps_key])
            if value > 0:
                return value

        time_key = first_existing(self.data, TIME_KEY_CANDIDATES)
        if time_key is not None:
            t = np.asarray(self.data[time_key], dtype=np.float64).reshape(-1)
            if len(t) == self.num_frames and len(t) >= 2:
                dt = float(np.median(np.diff(t)))
                if dt > 0:
                    return 1.0 / dt

        raise ValueError(
            "无法识别帧率。请通过 --fps 指定，例如 --fps 30"
        )

    def print_summary(self) -> None:
        print("\n========== NPZ 信息 ==========")
        print(f"文件       : {self.path}")
        print(f"总帧数     : {self.num_frames}")
        print(f"关节数     : {self.num_joints}")
        print(f"帧率       : {self.fps:.6g} FPS")
        print(f"时长       : {self.num_frames / self.fps:.3f} s")
        print(f"关节字段   : {self.joint_key}")
        print(f"根位置字段 : {self.root_pos_key}")
        print(f"根姿态字段 : {self.root_quat_key}")
        print("\n全部字段：")
        for key, value in self.data.items():
            print(f"  {key:28s} shape={value.shape!s:18s} dtype={value.dtype}")
        print("==============================\n")

    def crop(
        self,
        start: int,
        end_inclusive: int,
        recalc_velocity: bool,
    ) -> dict[str, np.ndarray]:
        start = max(0, min(start, self.num_frames - 1))
        end_inclusive = max(start, min(end_inclusive, self.num_frames - 1))
        end_exclusive = end_inclusive + 1

        output: dict[str, np.ndarray] = {}

        for key, value in self.data.items():
            arr = np.asarray(value)

            if arr.ndim >= 1 and arr.shape[0] == self.num_frames:
                output[key] = arr[start:end_exclusive].copy()
            else:
                output[key] = arr.copy()

        if recalc_velocity:
            self._recalculate_velocity_fields(output)

        return output

    def _recalculate_velocity_fields(self, output: dict[str, np.ndarray]) -> None:
        dt = self.frame_dt

        joint_pos = np.asarray(output[self.joint_key], dtype=np.float64)
        if len(joint_pos) >= 2:
            joint_vel = np.gradient(joint_pos, dt, axis=0)
        else:
            joint_vel = np.zeros_like(joint_pos)

        for key in JOINT_VEL_KEYS:
            if key in output and np.asarray(output[key]).shape == joint_vel.shape:
                output[key] = joint_vel.astype(np.asarray(output[key]).dtype, copy=False)

        if self.root_pos_key and self.root_pos_key in output:
            root_pos = np.asarray(output[self.root_pos_key], dtype=np.float64)
            if len(root_pos) >= 2:
                root_lin_vel = np.gradient(root_pos[:, :3], dt, axis=0)
            else:
                root_lin_vel = np.zeros((len(root_pos), 3), dtype=np.float64)

            for key in ROOT_LIN_VEL_KEYS:
                if key in output and np.asarray(output[key]).shape == root_lin_vel.shape:
                    output[key] = root_lin_vel.astype(
                        np.asarray(output[key]).dtype, copy=False
                    )

        if self.root_quat_key and self.root_quat_key in output:
            quat = np.asarray(output[self.root_quat_key], dtype=np.float64)
            if len(quat) > 0:
                quat_wxyz = quat.copy()
                # 内部重算函数统一使用 wxyz；是否转换由调用方保证。
                ang_vel = np.zeros((len(quat_wxyz), 3), dtype=np.float64)
                for i in range(1, len(quat_wxyz)):
                    ang_vel[i] = quat_delta_to_angular_velocity(
                        quat_wxyz[i - 1, :4], quat_wxyz[i, :4], dt
                    )
                if len(ang_vel) >= 2:
                    ang_vel[0] = ang_vel[1]

                for key in ROOT_ANG_VEL_KEYS:
                    if key in output and np.asarray(output[key]).shape == ang_vel.shape:
                        output[key] = ang_vel.astype(
                            np.asarray(output[key]).dtype, copy=False
                        )


class MotionPlayer:
    def __init__(self, motion: MotionData, args: argparse.Namespace):
        self.motion = motion
        self.args = args

        self.model = mujoco.MjModel.from_xml_path(str(args.model_xml))
        self.data = mujoco.MjData(self.model)

        self.frame = 0
        self.playing = not args.start_paused
        self.speed = float(args.speed)
        self.loop = not args.no_loop
        self.crop_start: int | None = None
        self.crop_end: int | None = None
        self.exit_requested = False
        self.last_wall_time = time.perf_counter()
        self.accumulator = 0.0

        self.free_joint_qpos_adr = self._find_free_joint_qpos_address()
        self.joint_qpos_addresses = self._build_joint_qpos_mapping()

        self.output_path = (
            args.output
            if args.output is not None
            else motion.path.with_name(motion.path.stem + "_crop.npz")
        )

    def _find_free_joint_qpos_address(self) -> int | None:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[joint_id])
        return None

    def _build_joint_qpos_mapping(self) -> list[int]:
        if self.args.joint_names:
            names = [x.strip() for x in self.args.joint_names.split(",") if x.strip()]
            if len(names) != self.motion.num_joints:
                raise ValueError(
                    f"--joint-names 数量 {len(names)} 与 NPZ 关节数 "
                    f"{self.motion.num_joints} 不一致"
                )

            addresses: list[int] = []
            for name in names:
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                if joint_id < 0:
                    raise ValueError(f"MuJoCo 模型中不存在关节：{name}")
                jnt_type = self.model.jnt_type[joint_id]
                if jnt_type not in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ):
                    raise ValueError(f"关节 {name} 不是单自由度关节")
                addresses.append(int(self.model.jnt_qposadr[joint_id]))
            return addresses

        # 自动选择所有单自由度关节，按 XML 中的关节顺序。
        addresses = []
        names = []
        for joint_id in range(self.model.njnt):
            jnt_type = self.model.jnt_type[joint_id]
            if jnt_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                addresses.append(int(self.model.jnt_qposadr[joint_id]))
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                names.append(name or f"joint_{joint_id}")

        if len(addresses) != self.motion.num_joints:
            raise ValueError(
                "自动关节映射失败：\n"
                f"  NPZ 关节数          = {self.motion.num_joints}\n"
                f"  模型单自由度关节数  = {len(addresses)}\n"
                "请通过 --joint-names 按 NPZ 列顺序指定关节名称，"
                "多个名称使用逗号分隔。\n"
                f"模型中的单自由度关节：{names}"
            )

        print("自动关节顺序：")
        for index, name in enumerate(names):
            print(f"  [{index:02d}] {name}")
        return addresses

    def _set_frame(self, frame: int) -> None:
        self.frame = max(0, min(frame, self.motion.num_frames - 1))
        self._apply_frame_to_model()

    def _apply_frame_to_model(self) -> None:
        qpos = self.data.qpos

        if self.free_joint_qpos_adr is not None:
            adr = self.free_joint_qpos_adr

            if self.motion.root_pos is not None:
                qpos[adr : adr + 3] = self.motion.root_pos[self.frame, :3]

            if self.motion.root_quat is not None:
                quat = self.motion.root_quat[self.frame, :4]
                if self.args.quat_order == "xyzw":
                    quat = quat_xyzw_to_wxyz(quat)
                qpos[adr + 3 : adr + 7] = normalize_quaternion_wxyz(quat)

        joint_values = self.motion.joint_pos[self.frame]
        for qpos_adr, value in zip(self.joint_qpos_addresses, joint_values):
            qpos[qpos_adr] = float(value)

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _print_status(self) -> None:
        start = "-" if self.crop_start is None else str(self.crop_start)
        end = "-" if self.crop_end is None else str(self.crop_end)
        state = "播放" if self.playing else "暂停"
        print(
            f"\r[{state}] frame={self.frame:6d}/{self.motion.num_frames - 1:6d} "
            f"time={self.frame / self.motion.fps:8.3f}s "
            f"speed={self.speed:4.2f}x crop=[{start},{end}]      ",
            end="",
            flush=True,
        )

    def _save_crop(self) -> None:
        start = self.crop_start if self.crop_start is not None else 0
        end = (
            self.crop_end
            if self.crop_end is not None
            else self.motion.num_frames - 1
        )

        if start > end:
            start, end = end, start

        output = self.motion.crop(start, end, self.args.recalc_velocity)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, **output)

        print(
            f"\n已保存裁剪文件：{self.output_path}\n"
            f"帧范围：[ {start}, {end} ]，共 {end - start + 1} 帧，"
            f"时长 {(end - start + 1) / self.motion.fps:.3f} 秒"
        )

    def _print_help(self) -> None:
        print(
            """
快捷键：
  Space       播放/暂停
  Left/Right  前一帧/后一帧
  A / D       后退/前进 10 帧
  J / L       后退/前进 1 秒
  [ / ]       降低/提高播放速度
  I           设置裁剪起点
  O           设置裁剪终点
  C           清除裁剪区间
  S           保存裁剪 NPZ
  R           回到第 0 帧
  Home/End    跳到首帧/末帧
  H           显示帮助
  Esc         退出
"""
        )

    def key_callback(self, keycode: int) -> None:
        # GLFW 键码。
        KEY_SPACE = 32
        KEY_LEFT = 263
        KEY_RIGHT = 262
        KEY_HOME = 268
        KEY_END = 269
        KEY_ESCAPE = 256

        if keycode == KEY_SPACE:
            self.playing = not self.playing
            self.accumulator = 0.0

        elif keycode == KEY_LEFT:
            self.playing = False
            self._set_frame(self.frame - 1)

        elif keycode == KEY_RIGHT:
            self.playing = False
            self._set_frame(self.frame + 1)

        elif keycode in (ord("A"), ord("a")):
            self.playing = False
            self._set_frame(self.frame - 10)

        elif keycode in (ord("D"), ord("d")):
            self.playing = False
            self._set_frame(self.frame + 10)

        elif keycode in (ord("J"), ord("j")):
            self.playing = False
            self._set_frame(self.frame - round(self.motion.fps))

        elif keycode in (ord("L"), ord("l")):
            self.playing = False
            self._set_frame(self.frame + round(self.motion.fps))

        elif keycode in (ord("I"), ord("i")):
            self.crop_start = self.frame
            print(f"\n裁剪起点设为 frame={self.frame}")

        elif keycode in (ord("O"), ord("o")):
            self.crop_end = self.frame
            print(f"\n裁剪终点设为 frame={self.frame}")

        elif keycode in (ord("C"), ord("c")):
            self.crop_start = None
            self.crop_end = None
            print("\n已清除裁剪区间")

        elif keycode in (ord("S"), ord("s")):
            self._save_crop()

        elif keycode in (ord("R"), ord("r")):
            self.playing = False
            self._set_frame(0)

        elif keycode in (ord("H"), ord("h")):
            self._print_help()

        elif keycode in (ord("["),):
            self.speed = max(0.05, self.speed / 1.25)
            print(f"\n播放速度：{self.speed:.3f}x")

        elif keycode in (ord("]"),):
            self.speed = min(8.0, self.speed * 1.25)
            print(f"\n播放速度：{self.speed:.3f}x")

        elif keycode == KEY_HOME:
            self.playing = False
            self._set_frame(0)

        elif keycode == KEY_END:
            self.playing = False
            self._set_frame(self.motion.num_frames - 1)

        elif keycode == KEY_ESCAPE:
            self.exit_requested = True

    def run(self) -> None:
        self._set_frame(0)
        self._print_help()

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            while viewer.is_running() and not self.exit_requested:
                now = time.perf_counter()
                wall_dt = now - self.last_wall_time
                self.last_wall_time = now

                if self.playing:
                    self.accumulator += wall_dt * self.speed

                    while self.accumulator >= self.motion.frame_dt:
                        self.accumulator -= self.motion.frame_dt
                        next_frame = self.frame + 1

                        if next_frame >= self.motion.num_frames:
                            if self.loop:
                                next_frame = 0
                            else:
                                next_frame = self.motion.num_frames - 1
                                self.playing = False

                        self._set_frame(next_frame)

                        if not self.playing:
                            break

                self._apply_frame_to_model()
                viewer.sync()
                self._print_status()

                # 避免主循环占满 CPU。
                time.sleep(0.001)

        print("\n播放器已退出。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo NPZ 动作播放器与裁剪器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", type=Path, required=True, help="输入 NPZ 文件")
    parser.add_argument(
        "--model-xml",
        type=Path,
        required=True,
        help="MuJoCo 机器人 XML/MJCF 模型文件",
    )
    parser.add_argument("--output", type=Path, default=None, help="裁剪输出 NPZ")

    parser.add_argument("--fps", type=float, default=None, help="手动指定帧率")
    parser.add_argument("--joint-key", default=None, help="关节位置字段名")
    parser.add_argument("--root-pos-key", default=None, help="根节点位置字段名")
    parser.add_argument("--root-quat-key", default=None, help="根节点四元数字段名")

    parser.add_argument(
        "--quat-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="NPZ 根节点四元数排列方式",
    )
    parser.add_argument(
        "--joint-names",
        default=None,
        help="按 NPZ 列顺序指定模型关节名，使用逗号分隔",
    )

    parser.add_argument("--speed", type=float, default=1.0, help="初始播放倍率")
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="启动后保持暂停",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="播放到末尾后停止，不循环",
    )
    parser.add_argument(
        "--recalc-velocity",
        action="store_true",
        help="保存裁剪文件时重算已存在的常见速度字段",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"输入 NPZ 不存在：{args.input}")

    if not args.model_xml.is_file():
        parser.error(f"MuJoCo XML 不存在：{args.model_xml}")

    if args.speed <= 0:
        parser.error("--speed 必须大于 0")

    return args


def main() -> int:
    args = parse_args()

    try:
        motion = MotionData(args.input, args)
        motion.print_summary()

        player = MotionPlayer(motion, args)
        player.run()
        return 0

    except KeyboardInterrupt:
        print("\n用户中断。")
        return 130

    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())