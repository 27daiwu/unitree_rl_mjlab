"""S10A: frozen S3 Dance policy ownership transfer in CPU MuJoCo.

Run with the extension's Python: python -m supermimic.s10a --help.
The reference supplies policy goals and a single preparation pose. Only policy
outputs supply moving dance targets; there are no state writes after reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from enum import IntEnum
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from scipy.spatial.transform import Rotation

from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robot import get_g1_robot_cfg
from supermimic.validation.cpu_interface_helpers import goal_scipy, history_pack

ROOT = Path(__file__).resolve().parents[4]
EXT = ROOT / "extensions/supermimic"
S3 = EXT / "logs/phase2_s3_dance1_subject2_anchor_2h_20261105"
CHECKPOINT = S3 / "ckpt_final_0036800.pt"
MOTION = EXT / "data/npz/lafan1/dance1_subject2.npz"
MANIFEST = Path(__file__).with_name("s10a_assets.json")


class State(IntEnum):
    STAND = 0
    TAKEOVER = 1
    DANCE = 2
    RELEASE = 3
    SAFE_DAMP = 4


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 + x * (-15.0 + 6.0 * x))


def rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]])


def limited(previous, desired, rate, dt):
    if not np.isfinite(desired).all():
        raise ValueError("Nonfinite control command")
    return previous + np.clip(desired - previous, -rate * dt, rate * dt)


class DanceRuntime:
    def __init__(self, out: Path):
        self.out = out
        self.manifest = json.loads(MANIFEST.read_text())
        self.check_assets()
        (out / "source_sha256.json").write_text(
            json.dumps(
                {
                    str(Path(__file__).relative_to(ROOT)): sha(__file__),
                    str(MANIFEST.relative_to(ROOT)): sha(MANIFEST),
                },
                indent=2,
            )
            + "\n"
        )
        cfg = json.loads((S3 / "config.json").read_text())
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        live_cfg = json.loads(json.dumps(checkpoint["config"], default=str))
        if live_cfg != cfg or cfg["data"]["clips"] != ["dance1_subject2"]:
            raise ValueError("S3 checkpoint/config/clip mismatch")
        if (
            checkpoint["iteration"] != 36800
            or cfg["policy"]["history"] != 10
            or cfg["policy"]["goal_dim"] != 689
            or not cfg["env"]["anchor_obs"]
        ):
            raise ValueError("S3 policy contract mismatch")
        self.policy = (
            Policy(
                PolicyCfg(
                    **{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["policy"].items()}
                )
            )
            .cpu()
            .eval()
        )
        self.policy.load_state_dict(checkpoint["policy"], strict=True)
        del checkpoint
        self.scene = Scene(
            SceneCfg(
                terrain=TerrainEntityCfg(terrain_type="plane"),
                entities={"robot": get_g1_robot_cfg()},
            ),
            device="cpu",
        )
        self.model = self.scene.compile()
        # Same configuration as make_env_cfg, without constructing a Warp trainer.
        MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20).apply(self.model)
        m = self.model
        self.data = mujoco.MjData(m)
        self.diag = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, self.data, 0)
        mujoco.mj_forward(m, self.data)
        self.qadr = np.arange(7, 36)
        self.vadr = np.arange(6, 35)
        self.names = [m.joint(i).name.split("/")[-1] for i in range(1, m.njnt)]
        self.aids = np.array(
            [int(np.flatnonzero(m.actuator_trnid[:, 0] == j)[0]) for j in range(1, m.njnt)]
        )
        if (m.nq, m.nv, m.nu) != (36, 35, 29) or len(set(self.aids)) != 29:
            raise ValueError("G1 actuator map mismatch")
        self.kp = m.actuator_gainprm[self.aids, 0].copy()
        self.kd = -m.actuator_biasprm[self.aids, 2].copy()
        if not np.allclose(m.actuator_biasprm[self.aids, 1], -self.kp):
            raise ValueError("Expected native position actuators")
        self.effort = m.actuator_forcerange[self.aids, 1].copy()
        self.q0 = m.key_qpos[0, self.qadr].copy()
        self.scale = 0.25 * self.effort / self.kp
        self.qmin, self.qmax = m.jnt_range[1:].T.copy()
        self.anchor = m.body("robot/torso_link").id
        self.pelvis = m.body("robot/pelvis").id
        self.feet = [m.body("robot/" + n + "_ankle_roll_link").id for n in ("left", "right")]
        with np.load(MOTION) as z:
            if z["joint_names"].astype(str).tolist() != self.names or z["fps"][0] != 50:
                raise ValueError("S3 motion joint order/rate mismatch")
            b = z["body_names"].astype(str).tolist()
            root, anchor = b.index("pelvis"), b.index("torso_link")
            self.ref = np.concatenate(
                [
                    z["body_pos_w"][:, root],
                    z["body_quat_w"][:, root],
                    z["joint_pos"],
                    z["body_lin_vel_w"][:, root],
                    z["body_ang_vel_w"][:, root],
                    z["joint_vel"],
                ],
                axis=1,
            ).astype(np.float32)
            self.apos = z["body_pos_w"][:, anchor].copy()
            self.arot = Rotation.from_quat(z["body_quat_w"][:, anchor][:, [1, 2, 3, 0]])
        self.features = goal_scipy(self.ref, self.q0).astype(np.float32)
        self.history = None
        self.previous = np.zeros(29, np.float32)
        self.inferences = 0
        self.inject_output = None
        self.interfaces = []
        self.alignment = None
        self.joint_index = {name: i for i, name in enumerate(self.names)}
        self.refresh()
        self.stand_xy = self.data.subtree_com[self.pelvis, :2].copy()
        audit = {
            "checkpoint_iteration": 36800,
            "config_equal": True,
            "strict_policy_binding": True,
            "clip": "dance1_subject2",
            "frames": len(self.ref),
            "policy_dt_s": 0.02,
            "physics_dt_s": 0.005,
            "joint_names": self.names,
            "actuator_ids_in_joint_order": self.aids.tolist(),
            "kp": self.kp.tolist(),
            "kd": self.kd.tolist(),
            "effort_Nm": self.effort.tolist(),
            "action_scale": self.scale.tolist(),
            "action_offset": self.q0.tolist(),
            "model": "training G1 + training plane/config",
            "engine": "CPU native MuJoCo; no claim of Warp numerical equivalence",
            "physics_state_writes": "one initial keyframe reset only",
        }
        (out / "asset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    def check_assets(self):
        actual = {p: sha(ROOT / p) for p in self.manifest["sha256"]}
        if actual != self.manifest["sha256"]:
            raise ValueError("Frozen S3 asset changed")
        return actual

    def refresh(self):
        self.diag.qpos[:] = self.data.qpos
        self.diag.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, self.diag)

    def stand(self, base):
        """Static joint PD, gravity compensation and bounded ankle balance feedback."""
        d = self.diag
        target = base.copy()
        # Keep CoM above the current feet, with horizontal velocity damping.
        center = np.mean(d.xpos[self.feet, :2], axis=0)
        com = d.subtree_com[self.pelvis, :2]
        r = rotation(d.qpos[3:7])
        yaw = r.as_euler("xyz")[2]
        heading = Rotation.from_euler("z", yaw)
        error = heading.inv().apply(np.r_[com - center, 0.0])[:2]
        vel = heading.inv().apply(d.qvel[:3])[:2]
        pitch_adjust = np.clip(10.0 * error[0] + 3.0 * vel[0], -0.50, 0.50)
        roll_adjust = np.clip(-10.0 * error[1] - 3.0 * vel[1], -0.50, 0.50)
        roll, pitch = r.as_euler("xyz")[:2]
        for side in ("left", "right"):
            target[self.joint_index[side + "_ankle_pitch_joint"]] += pitch_adjust
            target[self.joint_index[side + "_ankle_roll_joint"]] += roll_adjust
            target[self.joint_index[side + "_hip_pitch_joint"]] += np.clip(
                pitch + 0.1 * d.qvel[4], -0.30, 0.30
            )
            target[self.joint_index[side + "_hip_roll_joint"]] += np.clip(
                roll + 0.1 * d.qvel[3], -0.20, 0.20
            )
        # Partial support-load compensation avoids treating the floating pelvis
        # as a fixed base. The residual load is handled by joint PD and balance.
        gravity = d.qfrc_bias[self.vadr].copy()
        for foot in self.feet:
            jac = np.zeros((3, self.model.nv))
            point = d.xpos[foot] + heading.apply([0.025, 0.0, -0.04])
            self_force = np.array([0.0, 0.0, self.model.body_mass.sum() * 9.81 / 4])
            mujoco.mj_jac(self.model, d, jac, None, point, foot)
            gravity -= jac[:, self.vadr].T @ self_force
        target += np.clip(gravity / self.kp, -0.50, 0.50)
        return target

    def activate(self, target):
        self.refresh()
        live = rotation(self.diag.xquat[self.anchor])
        ref = self.arot[0]
        yaw = live.as_euler("xyz")[2] - ref.as_euler("xyz")[2]
        self.align_rot = Rotation.from_euler("z", yaw)
        offset = self.diag.xpos[self.anchor] - self.align_rot.apply(self.apos[0])
        offset[2] = 0.0  # preserve the frozen reference height above the plane
        self.align_pos = offset
        self.history = None
        # Match the actual previous controller target; no stale policy history.
        self.previous = ((target - self.q0) / self.scale).astype(np.float32)
        self.alignment = {
            "yaw_rad": float(yaw),
            "translation": offset.tolist(),
            "frame": 0,
            "history": "repeat live frame x10, term-major",
            "previous_action": self.previous.tolist(),
        }
        (self.out / "activation.json").write_text(json.dumps(self.alignment, indent=2) + "\n")

    def infer(self, frame):
        d = self.diag
        root = rotation(d.qpos[3:7])
        terms = [
            d.qpos[self.qadr] - self.q0,
            d.qvel[self.vadr],
            d.qvel[3:6],
            root.inv().apply([0.0, 0.0, -1.0]),
            self.previous,
        ]
        self.history, obs = history_pack(self.history, terms)
        live = rotation(d.xquat[self.anchor])
        refpos = self.align_rot.apply(self.apos[frame]) + self.align_pos
        relpos = live.inv().apply(refpos - d.xpos[self.anchor])
        relrot = live.inv() * self.align_rot * self.arot[frame]
        idx = np.minimum(frame + np.arange(10), len(self.ref) - 1)
        goal = np.r_[self.features[idx].ravel(), relpos, relrot.as_matrix()[:, :2].ravel()]
        obs, goal = np.asarray(obs, np.float32), np.asarray(goal, np.float32)
        if obs.shape != (930,) or goal.shape != (689,):
            raise ValueError("Policy interface dimensions")
        with torch.inference_mode():
            if self.inject_output == "exception":
                self.inject_output = None
                raise RuntimeError("injected_policy_exception")
            action = self.policy(torch.from_numpy(obs[None]), torch.from_numpy(goal[None])).numpy()[
                0
            ]
        self.inferences += 1
        if self.inject_output == "nan":
            self.inject_output = None
            action = np.full(29, np.nan, np.float32)
        self.interfaces.append((self.data.time, frame, obs.copy(), goal.copy(), action.copy()))
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("Nonfinite/invalid policy output")
        self.previous = action.copy()
        return self.q0 + self.scale * action


class Handoff:
    def __init__(self, runtime, args):
        self.r, self.args = runtime, args
        self.state = State.STAND
        self.entered = 0.0
        self.events = []
        self.rows = []
        self.target = runtime.q0.copy()
        self.torque = np.zeros(29)
        self.source = self.target.copy()
        self.policy_target = self.target.copy()
        self.phase = 0
        self.active = False
        self.next_infer = 0.0
        self.fault = None
        self.initial_stable = False
        self.returned = False
        self.completed = False
        self.takeover_ok = False
        self.compatibility = None
        self.stable_since = None
        self.fault_injected = False
        self.guard_metrics = None
        self.faults = []
        self.recovery = None
        self.stop_requested = None
        self.stop_events = []
        self.use_dynamic = False
        if getattr(args, "dynamic_recovery", False):
            from supermimic.s10b_recovery import Recovery

            self.recovery = Recovery(runtime)
        self.limit_frames = (
            len(runtime.ref)
            if args.dance_seconds is None
            else min(len(runtime.ref), max(1, round(args.dance_seconds / 0.02)))
        )
        self.full_length = self.limit_frames == len(runtime.ref)

    def transition(self, state, reason):
        event = {
            "time": float(self.r.data.time),
            "from": self.state.name,
            "to": state.name,
            "reason": reason,
            "phase_frame": self.phase,
        }
        self.events.append(event)
        print(json.dumps(event), flush=True)
        self.state, self.entered = state, self.r.data.time
        self.source = self.target.copy()
        if state == State.RELEASE:
            # Begin recovery from measured joints, not a policy target that may
            # anticipate the next movement. The output slew preserves continuity.
            self.release_base = self.r.data.qpos[7:].copy()
        self.stable_since = None

    def stable(self):
        d = self.r.data
        tilt = np.arccos(np.clip(rotation(d.qpos[3:7]).as_matrix()[2, 2], -1, 1))
        return (
            d.qpos[2] > 0.65
            and tilt < 0.15
            and np.linalg.norm(d.qvel[:3]) < 0.10
            and np.linalg.norm(d.qvel[3:6]) < 0.20
            and np.max(np.abs(d.qvel[6:])) < 0.40
        )

    def release(self, reason, fault=False):
        if self.state in (State.RELEASE, State.SAFE_DAMP):
            return
        if fault:
            self.fault = self.fault or reason
            self.faults.append({"time": float(self.r.data.time), "reason": reason})
        self.active = False
        self.transition(State.RELEASE, reason)
        if self.recovery is not None and reason not in ("clip_end", "short_segment_end"):
            self.use_dynamic = True
            self.recovery.start()

    def request_stop(self, reason):
        if self.recovery is None:
            self.release(reason, fault=True)
        elif self.stop_requested is None:
            self.stop_requested = float(self.r.data.time)
            self.stop_events.append(
                {
                    "event": "normal_stop_requested",
                    "time": self.stop_requested,
                    "phase_frame": self.phase,
                    "reason": reason,
                }
            )

    def safety(self):
        d = self.r.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            return "nonfinite_physics"
        tilt = np.arccos(np.clip(rotation(d.qpos[3:7]).as_matrix()[2, 2], -1, 1))
        if self.state == State.DANCE:
            ref = self.r.ref[min(max(self.phase - 1, 0), len(self.r.ref) - 1)]
            ref_cos = rotation(ref[3:7]).as_matrix()[2, 2]
            height_error = abs(d.qpos[2] - ref[2])
            tilt_error = abs(np.cos(tilt) - ref_cos)
            self.guard_metrics = {
                "height_error_m": float(height_error),
                "gravity_z_error": float(tilt_error),
            }
            if height_error > 0.25 or tilt_error > 0.8:
                return "tracking_error_guard"
            fallen = d.qpos[2] < 0.20 or tilt > 1.4
        else:
            fallen = d.qpos[2] < 0.48 or tilt > 0.8
        if fallen:
            return "fall_guard"
        if np.max(np.abs(d.qvel[6:])) > 35:
            return "joint_speed_guard"
        if sum(w.number for w in d.warning):
            return "mujoco_warning"
        return None

    def tick(self):
        r, a = self.r, self.args
        d, dt = r.data, r.model.opt.timestep
        r.refresh()
        elapsed = d.time - self.entered
        reason = self.safety()
        if reason and self.state != State.SAFE_DAMP:
            self.fault = self.fault or reason
            self.faults.append({"time": float(d.time), "reason": reason})
            self.active = False
            self.transition(State.SAFE_DAMP, reason)
            elapsed = 0.0
        if self.stable():
            if self.stable_since is None:
                self.stable_since = d.time
        else:
            self.stable_since = None
        stable_long = self.stable_since is not None and d.time - self.stable_since >= 1.0
        if (
            a.stop_file is not None
            and a.stop_file.exists()
            and self.state in (State.TAKEOVER, State.DANCE)
        ):
            self.request_stop("explicit_abort_file")
        if self.stop_requested is not None and self.state == State.DANCE:
            waited = d.time - self.stop_requested
            if self.recovery.ready() or waited + 1e-8 >= 2.0:
                self.stop_events.append(
                    {
                        "event": "normal_stop_release",
                        "time": float(d.time),
                        "wait_s": waited,
                        "phase_frame": self.phase,
                    }
                )
                self.release("explicit_abort", fault=False)
                elapsed = 0.0
        desired = self.target.copy()
        if self.state == State.STAND:
            desired = r.stand(r.q0)
            if not self.returned:
                self.initial_stable = bool(stable_long)
                signal = (a.signal_file is not None and a.signal_file.exists()) or (
                    a.signal_file is None and d.time >= a.trigger_at
                )
                if signal and self.initial_stable:
                    self.transition(State.TAKEOVER, "explicit_takeover_signal")
            elif stable_long and elapsed >= a.post_stand:
                return False
        elif self.state == State.TAKEOVER:
            if not self.active:
                blend = smooth(elapsed / a.prepare_seconds)
                desired = r.stand((1 - blend) * r.q0 + blend * r.ref[0, 7:36])
                if elapsed >= a.prepare_seconds:
                    qerr = np.abs(d.qpos[7:] - r.ref[0, 7:36])
                    tilt = np.arccos(np.clip(rotation(d.qpos[3:7]).as_matrix()[2, 2], -1, 1))
                    self.compatibility = {
                        "max_joint_error_rad": float(qerr.max()),
                        "rms_joint_error_rad": float(np.sqrt(np.mean(qerr * qerr))),
                        "root_tilt_rad": float(tilt),
                        "root_height_m": float(d.qpos[2]),
                    }
                    compatible = (
                        qerr.max() <= 0.35
                        and tilt <= 0.20
                        and np.linalg.norm(d.qvel[:3]) <= 0.15
                        and np.max(np.abs(d.qvel[6:])) <= 0.5
                    )
                    if not compatible and elapsed > a.prepare_seconds + 4:
                        self.release("initial_pose_incompatible", fault=True)
                    elif compatible:
                        self.active = True
                        self.blend_start = d.time
                        self.source = self.target.copy()
                        r.activate(self.target)
                        self.next_infer = d.time
            else:
                if d.time + 1e-8 >= self.next_infer:
                    self.policy_target = r.infer(0)
                    self.next_infer += 0.02
                blend = smooth((d.time - self.blend_start) / a.blend_seconds)
                desired = (1 - blend) * self.source + blend * self.policy_target
                if d.time - self.blend_start >= a.blend_seconds:
                    self.takeover_ok = True
                    self.phase = 0
                    self.next_infer = d.time
                    self.transition(State.DANCE, "policy_blend_complete")
        elif self.state == State.DANCE:
            if a.fault != "none" and not self.fault_injected and elapsed >= a.fault_at:
                self.fault_injected = True
                if a.fault == "stop":
                    self.request_stop("explicit_abort")
                else:
                    r.inject_output = a.fault
            if self.state == State.DANCE:
                if d.time + 1e-8 >= self.next_infer:
                    if self.phase >= self.limit_frames:
                        self.completed = True
                        self.release("clip_end" if self.full_length else "short_segment_end")
                    else:
                        self.policy_target = r.infer(self.phase)
                        self.phase += 1
                        self.next_infer += 0.02
                desired = self.policy_target.copy()
        elif self.state == State.RELEASE:
            blend = smooth(elapsed / a.release_seconds)
            desired = r.stand((1 - blend) * self.release_base + blend * r.q0)
            if elapsed >= a.release_seconds and stable_long:
                self.returned = True
                self.stable_since = d.time
                self.transition(State.STAND, "release_settled")
            elif elapsed > a.release_seconds + 5:
                self.fault = self.fault or "release_settle_timeout"
                self.transition(State.SAFE_DAMP, "release_settle_timeout")
        else:
            desired = d.qpos[7:].copy()
            if elapsed >= 1.0:
                return False
        if self.state == State.RELEASE and not np.isfinite(desired).all():
            desired = self.source.copy()
        # Substep target slew and hard joint limits apply to every owner.
        target_rate = a.dance_target_rate if self.state == State.DANCE else a.transition_target_rate
        self.target = limited(self.target, np.clip(desired, r.qmin, r.qmax), target_rate, dt)
        wanted_tau = r.kp * (self.target - d.qpos[7:]) - r.kd * d.qvel[6:]
        if self.use_dynamic and self.state in (State.RELEASE, State.STAND):
            wanted_tau = self.recovery.torque(self.torque, a.transition_torque_rate)
        if self.state == State.SAFE_DAMP:
            wanted_tau = -2 * r.kd * d.qvel[6:]
        torque_rate = a.torque_rate if self.state == State.DANCE else a.transition_torque_rate
        self.torque = limited(
            self.torque, np.clip(wanted_tau, -r.effort, r.effort), torque_rate, dt
        )
        # Encode bounded torque in the unchanged native position actuator. MuJoCo
        # retains the native implicit PD derivatives and training armature.
        servo = d.qpos[7:] + (self.torque + r.kd * d.qvel[6:]) / r.kp
        d.ctrl[r.aids] = servo
        pre = d.qpos.copy()
        mujoco.mj_step(r.model, d)
        self.rows.append(
            (
                float(d.time),
                int(self.state),
                self.phase,
                pre,
                d.qpos.copy(),
                d.qvel.copy(),
                desired.copy(),
                self.target.copy(),
                servo.copy(),
                self.torque.copy(),
                d.qfrc_actuator[r.vadr].copy(),
                d.actuator_force[r.aids].copy(),
                int(d.ncon),
                bool(self.stable()),
            )
        )
        return True

    def run(self):
        start = time.monotonic()
        last_progress = start
        cleanup_started = None
        cleanup_wall = None
        while True:
            if cleanup_started is None and (
                self.r.data.time >= self.args.max_sim_seconds
                or time.monotonic() - start >= self.args.wall_seconds
            ):
                reason = (
                    "wall_timeout"
                    if time.monotonic() - start >= self.args.wall_seconds
                    else "simulation_timeout"
                )
                self.release(reason, fault=True)
                cleanup_started, cleanup_wall = self.r.data.time, time.monotonic()
            if cleanup_started is not None and (
                self.r.data.time - cleanup_started > 12 or time.monotonic() - cleanup_wall > 15
            ):
                break
            try:
                if not self.tick():
                    break
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- release on any inference failure
                if self.state in (State.RELEASE, State.SAFE_DAMP):
                    self.fault = f"unrecoverable:{type(error).__name__}:{error}"
                    break
                self.release(f"{type(error).__name__}:{error}", fault=True)
            if time.monotonic() - last_progress > 10:
                print(
                    f"progress sim={self.r.data.time:.2f}s state={self.state.name} frame={self.phase}",
                    flush=True,
                )
                last_progress = time.monotonic()
        return self.save(time.monotonic() - start)

    def save(self, wall):
        out, r = self.args.out, self.r
        keys = [
            "time",
            "state",
            "phase",
            "pre_qpos",
            "qpos",
            "qvel",
            "desired_target",
            "target",
            "servo_ctrl",
            "bounded_torque",
            "joint_torque",
            "actuator_force",
            "ncon",
            "stable",
        ]
        arrays = {key: np.asarray([row[i] for row in self.rows]) for i, key in enumerate(keys)}
        np.savez_compressed(out / "trace.npz", **arrays)
        if r.interfaces:
            np.savez_compressed(
                out / "policy.npz",
                time=[v[0] for v in r.interfaces],
                frame=[v[1] for v in r.interfaces],
                obs=[v[2] for v in r.interfaces],
                goal=[v[3] for v in r.interfaces],
                action=[v[4] for v in r.interfaces],
            )
        (out / "events.json").write_text(json.dumps(self.events, indent=2) + "\n")
        (out / "stop_events.json").write_text(json.dumps(self.stop_events, indent=2) + "\n")
        if self.recovery is not None:
            self.recovery.save(out)
        stages = {}
        dt = r.model.opt.timestep
        for state in State:
            mask = arrays["state"] == int(state)
            if not mask.any():
                stages[state.name] = None
                continue
            indexes = np.flatnonzero(mask)
            steps = indexes[indexes > 0]
            qstep = np.max(
                np.abs(arrays["target"][steps] - arrays["target"][steps - 1]), initial=0.0
            )
            tstep = np.max(
                np.abs(arrays["joint_torque"][steps] - arrays["joint_torque"][steps - 1]),
                initial=0.0,
            )
            servo_step = np.max(
                np.abs(arrays["servo_ctrl"][steps] - arrays["servo_ctrl"][steps - 1]), initial=0.0
            )
            rate = (
                self.args.dance_target_rate
                if state == State.DANCE
                else self.args.transition_target_rate
            )
            trate = (
                self.args.torque_rate if state == State.DANCE else self.args.transition_torque_rate
            )
            finite = all(
                np.isfinite(arrays[k][mask]).all()
                for k in ("qpos", "qvel", "target", "joint_torque")
            )
            magnitude_ok = np.all(np.abs(arrays["joint_torque"][mask]) <= r.effort + 1e-6)
            stages[state.name] = {
                "samples": int(mask.sum()),
                "max_target_step_rad": float(qstep),
                "max_servo_ctrl_step_rad": float(servo_step),
                "max_torque_step_Nm": float(tstep),
                "max_target_rate_rad_s": rate,
                "max_torque_rate_Nm_s": trate,
                "finite": bool(finite),
                "bounded": bool(
                    finite
                    and magnitude_ok
                    and qstep <= rate * dt + 1e-6
                    and tstep <= trate * dt + 1e-6
                ),
            }
        bounds_ok = all(v["bounded"] for v in stages.values() if v is not None)
        takeover_bounds = stages["TAKEOVER"] is not None and stages["TAKEOVER"]["bounded"]
        release_bounds = stages["RELEASE"] is not None and stages["RELEASE"]["bounded"]
        release_ok = (
            self.returned and self.state == State.STAND and self.stable() and release_bounds
        )
        dance_ok = self.completed and self.takeover_ok and stages["DANCE"]["bounded"]
        chain_ok = (
            self.initial_stable
            and self.takeover_ok
            and takeover_bounds
            and dance_ok
            and release_ok
            and bounds_ok
            and self.fault is None
        )
        result = {
            "initial_stand": "PASS" if self.initial_stable else "FAIL",
            "takeover": "PASS" if self.takeover_ok and takeover_bounds else "FAIL",
            "dance": "PASS" if dance_ok else "FAIL",
            "release": "PASS" if release_ok else "FAIL",
            "full_chain": "PASS" if chain_ok else "FAIL",
            "scope": "full_dance" if self.full_length else "short_segment",
            "full_dance_acceptance": "NOT_RUN"
            if not self.full_length
            else ("PASS" if chain_ok else "FAIL"),
            "fault": self.fault,
            "faults": self.faults,
            "final_state": self.state.name,
            "policy_inferences": r.inferences,
            "dance_frames_executed": self.phase,
            "required_frames": self.limit_frames,
            "total_clip_frames": len(r.ref),
            "physics_steps": len(self.rows),
            "sim_seconds": float(r.data.time),
            "wall_seconds": wall,
            "stages": stages,
            "bounds_ok": bool(bounds_ok),
            "compatibility": self.compatibility,
            "guard_metrics": self.guard_metrics,
            "state_codes": {s.name: int(s) for s in State},
            "sim_only": True,
        }
        r.check_assets()
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--policy-runtime",
        choices=["shared", "legacy"],
        default="shared",
        help="Shared PolicyRuntime with SimAdapter, or unchanged original DanceRuntime",
    )
    p.add_argument(
        "--dance-seconds", type=float, help="Short segment; omitted means all 6574 frames"
    )
    p.add_argument(
        "--trigger-at", type=float, default=3.0, help="Explicit simulated switch signal time"
    )
    p.add_argument(
        "--signal-file", type=Path, help="Presence of this file triggers takeover when stable"
    )
    p.add_argument("--stop-file", type=Path, help="Presence requests a bounded release")
    p.add_argument("--prepare-seconds", type=float, default=2.0)
    p.add_argument("--blend-seconds", type=float, default=0.20)
    p.add_argument("--release-seconds", type=float, default=2.0)
    p.add_argument("--post-stand", type=float, default=3.0)
    p.add_argument("--transition-target-rate", type=float, default=2.0)
    p.add_argument("--dance-target-rate", type=float, default=20.0)
    p.add_argument("--torque-rate", type=float, default=6000.0)
    p.add_argument("--transition-torque-rate", type=float, default=1500.0)
    p.add_argument("--max-sim-seconds", type=float, default=160.0)
    p.add_argument("--wall-seconds", type=float, default=180.0)
    p.add_argument("--fault", choices=["none", "nan", "exception", "stop"], default="none")
    p.add_argument("--fault-at", type=float, default=0.30)
    p.add_argument(
        "--policy-fault",
        choices=["none", "timeout", "permanent"],
        default="none",
        help="Inject inside isolated native Policy worker at 50%% Dance progress",
    )
    a = p.parse_args()
    if any(
        getattr(a, n) <= 0
        for n in (
            "prepare_seconds",
            "blend_seconds",
            "release_seconds",
            "post_stand",
            "transition_target_rate",
            "dance_target_rate",
            "torque_rate",
            "transition_torque_rate",
            "max_sim_seconds",
            "wall_seconds",
        )
    ) or (a.dance_seconds is not None and a.dance_seconds <= 0):
        p.error("Durations and rate limits must be positive")
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "arguments.json").write_text(json.dumps(vars(a), default=str, indent=2) + "\n")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from supermimic.s10b_native import run_native

    if a.policy_runtime == "shared":
        from supermimic.sim_adapter import SharedDanceRuntime

        runtime = SharedDanceRuntime(a.out)
    else:
        runtime = DanceRuntime(a.out)
    result = run_native(runtime, a)
    raise SystemExit(0 if result["full_chain"] == "PASS" else 1)


if __name__ == "__main__":
    main()
