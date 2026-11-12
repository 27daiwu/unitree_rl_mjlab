"""Shared S3 runtime boundary for the external two-process MuJoCo simulator."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from supermimic.dds_transport import DdsTimeoutError
from supermimic.policy.runtime import ContractError
from supermimic.s10a import ROOT
from supermimic.sim_adapter import SharedDanceRuntime


class _DdsAdapterView:
    def __init__(self, runtime):
        self.runtime = runtime

    def __getattr__(self, name):
        return getattr(self.runtime._sim_adapter, name)

    def refresh(self):
        self.runtime._sim_adapter.refresh()

    def snapshot(self, *, measured_at):
        frame = self.runtime.latest_frame
        source_time = frame.received_monotonic if frame is not None else measured_at
        return self.runtime._sim_adapter.snapshot(measured_at=source_time)


class DdsSharedDanceRuntime(SharedDanceRuntime):
    """Use real external states while retaining frozen inference and Handoff."""

    def __init__(self, out: Path, transport):
        super().__init__(out)
        self.transport = transport
        self.latest_frame = None
        self._next_tick = None
        # scene_g1 uses 2 ms direct-motor physics steps, not S10A's local 5 ms.
        self.model.opt.timestep = 0.002
        self._sim_adapter = self.adapter
        self.adapter = _DdsAdapterView(self)
        self.contact_available = False
        self.command_rows = []
        self._last_sent_tau = np.zeros(29)
        scene_path = ROOT / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
        self.external_model = mujoco.MjModel.from_xml_path(str(scene_path))
        if self.external_model.opt.timestep != 0.002:
            raise ContractError("Unexpected external simulator timestep")
        external_names = [
            self.external_model.actuator(i).name for i in range(self.external_model.nu)
        ]
        expected = [name.removesuffix("_joint") for name in self.names]
        if external_names != expected:
            raise ContractError("External actuator and S3 joint mapping mismatch")
        self.external_ctrlrange = np.abs(self.external_model.actuator_ctrlrange).max(axis=1)
        self.external_effort = np.minimum(self.effort, self.external_ctrlrange)
        self.backend_info = {
            "backend": "LOCAL_DDS_SIM_PHYSICS",
            "simulator": "simulate/build/unitree_mujoco",
            "simulator_model": "scene_g1.xml; original direct motor actuators",
            "control_dt_s": 0.002,
            "policy_dt_s": 0.02,
            "command_topic": "rt/lowcmd",
            "command_mapping": "direct torque: tau_ff=clip(kp_s3*(servo-q_meas)-kd_s3*dq_meas, external_ctrlrange), kp=kd=dq=0",
            "lowstate_topic": "rt/lowstate",
            "root_topic": "rt/sportmodestate",
            "joint_mapping": "SDK index 0..28 equals frozen S3 joint-name order",
            "contact_available": False,
            "ncon_sentinel": -1,
            "dynamic_qp_recovery": "NOT_RUN_CONTACT_POINTS_UNAVAILABLE",
            "s3_effort_Nm": self.effort.tolist(),
            "external_ctrlrange_Nm": self.external_ctrlrange.tolist(),
            "external_effort_min_Nm": self.external_effort.tolist(),
            "source_snapshot_time": "DDS receive monotonic time",
            "applied_frame_rule": "accept only tick>=target and |tau_est-sent_tau_ff|<=1e-4 Nm; timeout 100ms",
            "sent_tau_slew_margin_Nm": 0.001,
            "physics_integration": "unitree_mujoco only; no local mj_step in deploy",
        }
        (out / "dds_backend.json").write_text(json.dumps(self.backend_info, indent=2) + "\n")

    def sync_initial_state(self):
        frame = self.transport.read_state(timeout_s=5.0)
        if frame is None or not frame.root_valid:
            raise ContractError("No complete LowState/SportModeState from local simulator")
        self._apply_frame(frame)
        self._next_tick = frame.tick + 2
        return frame

    def _apply_frame(self, frame):
        if self.latest_frame is not None and frame.tick < self.latest_frame.tick:
            raise ContractError("Simulator tick moved backwards")
        if frame.mode_machine != 5:
            raise ContractError(f"Unexpected G1 simulation mode_machine={frame.mode_machine}")
        if abs(np.linalg.norm(frame.root_quat_wxyz) - 1.0) > 1e-5:
            raise ContractError("Invalid external root quaternion")
        data = self.data
        data.time = frame.tick * 0.001
        data.qpos[:3] = frame.root_pos_w
        data.qpos[3:7] = frame.root_quat_wxyz
        data.qpos[7:] = frame.q
        data.qvel[:3] = frame.root_lin_vel_w
        data.qvel[3:6] = frame.omega_b
        data.qvel[6:] = frame.dq
        data.qfrc_actuator[self.vadr] = frame.tau_est
        data.actuator_force[self.aids] = frame.tau_est
        # Contact points/forces are not published by the existing bridge.
        data.ncon = -1
        self.latest_frame = frame

    def _read_applied(self, target_tick, sent_tau):
        deadline = time.monotonic() + 0.10
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DdsTimeoutError(f"Timed out waiting for applied tau at tick {target_tick}")
            frame = self.transport.read_until(target_tick, timeout_s=remaining)
            error = float(np.max(np.abs(frame.tau_est - sent_tau)))
            if error <= 1e-4:
                return frame, error

    def external_step(self, model, data):
        if model is not self.model or data is not self.data or self._next_tick is None:
            raise ContractError("Unsynchronized external runtime step")
        servo = data.ctrl[self.aids].copy()
        predicted_tau = self.kp * (servo - data.qpos[7:]) - self.kd * data.qvel[6:]
        desired_tau = np.clip(predicted_tau, -self.external_effort, self.external_effort)
        owner = getattr(getattr(self, "policy", None), "owner", "STAND")
        torque_rate = 6000.0 if owner == "DANCE" else 1500.0
        step_limit = torque_rate * self.model.opt.timestep - 0.001
        sent_tau = np.clip(
            desired_tau, self._last_sent_tau - step_limit, self._last_sent_tau + step_limit
        )
        self._last_sent_tau = sent_tau.copy()
        self.command_rows.append(
            {
                "time": float(data.time),
                "servo": servo,
                "predicted_tau": predicted_tau,
                "sent_tau_ff": sent_tau,
            }
        )
        self.transport.write_command(
            q=servo,
            kp=np.zeros(29),
            kd=np.zeros(29),
            dq=np.zeros(29),
            tau=sent_tau,
            mode_machine=self.latest_frame.mode_machine,
        )
        frame, ack_error = self._read_applied(self._next_tick, sent_tau)
        self.command_rows[-1]["applied_tick"] = frame.tick
        self.command_rows[-1]["ack_max_abs_error_Nm"] = ack_error
        self._apply_frame(frame)
        self._next_tick = frame.tick + 2

    def check_assets(self):
        actual = super().check_assets()
        audit_path = self.out / "asset_audit.json"
        audit = json.loads(audit_path.read_text())
        audit.update(
            {
                "backend": "LOCAL_DDS_SIM_PHYSICS",
                "physics_dt_s": 0.002,
                "engine": "external unitree_mujoco direct motor physics; deploy does not call mj_step",
                "contact_available": False,
                "ncon_sentinel": -1,
                "dynamic_qp_recovery": "NOT_RUN",
            }
        )
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        session_path = self.out / "shared_session.json"
        session = json.loads(session_path.read_text())
        session["backend"] = "LOCAL_DDS_SIM_PHYSICS"
        session["contact_available"] = False
        session_path.write_text(json.dumps(session, indent=2) + "\n")
        self.backend_info["result"] = actual
        (self.out / "dds_backend.json").write_text(json.dumps(self.backend_info, indent=2) + "\n")
        arrays = {
            key: np.asarray([row[key] for row in self.command_rows])
            if self.command_rows
            else np.empty((0, 29))
            for key in ("servo", "predicted_tau", "sent_tau_ff")
        }
        arrays["time"] = np.asarray([row["time"] for row in self.command_rows])
        arrays["applied_tick"] = np.asarray(
            [row.get("applied_tick", -1) for row in self.command_rows]
        )
        arrays["ack_max_abs_error_Nm"] = np.asarray(
            [row.get("ack_max_abs_error_Nm", np.nan) for row in self.command_rows]
        )
        arrays["kp"] = np.zeros((len(self.command_rows), 29))
        arrays["kd"] = np.zeros((len(self.command_rows), 29))
        arrays["external_effort"] = self.external_effort
        np.savez_compressed(self.out / "dds_commands.npz", **arrays)
        return actual
