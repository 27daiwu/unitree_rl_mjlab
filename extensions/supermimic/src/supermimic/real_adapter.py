"""Offline-only Real state and motor-field contracts for the frozen S3 runtime.

This module contains no SDK/DDS import, transport, serializer, device discovery,
or motor publication.  Every input is an explicit Python data object and every
output remains in memory.  ``REAL_MOCK`` validity describes data-contract checks
only; U1--U5 and real actuator/ownership/physical recovery remain UNVERIFIED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.policy.runtime import AcceptedAction, ContractError, RobotSnapshot

REAL_MOCK = "REAL_MOCK"
UNVERIFIED = "UNVERIFIED"
NOT_RUN = "NOT_RUN"
COMMAND_ENCODING = "REAL_LOWCMD_FIELDS_OFFLINE_MOCK_V1"

CAPABILITIES = {
    "U1": "root_anchor_estimation_frames_floor_extrinsics_drift_and_relocalization",
    "U2": "factory_arbitration_prepublication_revocation_loss_and_late_rpc_semantics",
    "U3": "motor_calibration_inner_loop_effort_and_command_continuity",
    "U4": "factory_normal_and_fault_physical_recovery_domain",
    "U5": "real_timing_clock_scheduling_dds_and_rpc_deadlines",
}


def capability_report() -> dict[str, dict[str, str]]:
    """These statuses cannot be cleared by constructing a successful mock."""

    return {
        key: {
            "status": UNVERIFIED,
            "blocks": "REAL_ROBOT_AND_S10C",
            "capability": value,
        }
        for key, value in CAPABILITIES.items()
    }


def _array(value, shape, name, dtype=np.float64):
    try:
        result = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as error:
        raise ContractError(f"Missing/invalid {name}") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ContractError(f"Missing/invalid {name}: expected finite {shape}")
    result.flags.writeable = False
    return result


def _time(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
        raise ContractError(f"Missing/nonfinite {name}")
    return float(value)


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ContractError(f"{name} must be >= {minimum}")
    return result


def _rotation(value, name):
    quat = _array(value, (4,), name)
    if abs(np.linalg.norm(quat) - 1.0) > 1e-8:
        raise ContractError(f"{name} must be a unit wxyz quaternion")
    return Rotation.from_quat(quat[[1, 2, 3, 0]])


def _quat(rotation):
    xyzw = rotation.as_quat()
    return np.array(xyzw[[3, 0, 1, 2]], dtype=np.float64)


def _require_mock(domain):
    if domain != REAL_MOCK:
        raise ContractError("P3 accepts explicit REAL_MOCK data only")


@dataclass(frozen=True)
class JointCalibration:
    """Candidate calibration for one named joint; never an S3 q0 offset."""

    joint_name: str
    sdk_index: int
    sign: int
    zero_offset_rad: float
    status: str = UNVERIFIED


@dataclass(frozen=True)
class RealCalibration:
    joints: tuple[JointCalibration, ...]
    evidence_domain: str = REAL_MOCK


@dataclass(frozen=True)
class ImuExtrinsic:
    """Unit R_IB quaternion: rotate pelvis/B vectors into the IMU/I frame."""

    quat_ib_wxyz: np.ndarray
    status: str = UNVERIFIED
    evidence_domain: str = REAL_MOCK


@dataclass(frozen=True)
class RealTimingContract:
    """Explicit mock budgets; None is a blocker, never an infinite/default budget."""

    max_state_age_s: float | None = None
    max_interfield_skew_s: float | None = None
    max_clock_error_s: float | None = None
    evidence_domain: str = REAL_MOCK


@dataclass(frozen=True)
class LowStateFrame:
    """Pure-data subset of LowState, indexed by explicit SDK motor slot."""

    sequence: int
    clock_domain: str
    motor_measured_at: float
    imu_measured_at: float
    received_monotonic: float
    motor_indices: tuple[int, ...]
    q_sdk: np.ndarray
    dq_sdk: np.ndarray
    imu_quat_wi_wxyz: np.ndarray
    gyroscope_i: np.ndarray
    mode_machine: int | None
    motor_valid: bool = True
    imu_valid: bool = True
    tau_est_sdk: np.ndarray | None = None
    evidence_domain: str = REAL_MOCK


@dataclass(frozen=True)
class RealStateEstimate:
    """Required external root estimate plus frozen-FK relative Anchor pose.

    p_BA and R_BA are torso_link relative to pelvis at the same q.  They are
    composed with p_WB/R_WB; no global Anchor position is inferred from IMU/q.
    """

    estimate_id: str
    lowstate_sequence: int
    clock_domain: str
    measured_at: float
    root_pos_w: np.ndarray
    root_lin_vel_w: np.ndarray
    anchor_pos_b: np.ndarray
    anchor_quat_ba_wxyz: np.ndarray
    valid: bool = True
    estimator_reset: bool = False
    evidence_domain: str = REAL_MOCK
    status: str = UNVERIFIED


@dataclass(frozen=True)
class TorqueEvidence:
    """Telemetry semantics stay distinct from predicted/measured total torque."""

    kind: str
    values_s3: np.ndarray | None
    status: str = UNVERIFIED


@dataclass(frozen=True)
class DecodedLowState:
    sequence: int
    measured_monotonic: float
    motor_measured_monotonic: float
    imu_measured_monotonic: float
    received_monotonic: float
    joint_names: tuple[str, ...]
    q_s3: np.ndarray
    dq_s3: np.ndarray
    root_quat_wb_wxyz: np.ndarray
    omega_b: np.ndarray
    mode_machine: int
    torque_evidence: TorqueEvidence
    clock_error_s: float
    motor_clock_error_s: float
    imu_clock_error_s: float
    evidence_domain: str = REAL_MOCK


@dataclass(frozen=True)
class RealState:
    robot: RobotSnapshot
    root_pos_w: np.ndarray
    root_lin_vel_w: np.ndarray
    estimate_id: str
    lowstate: DecodedLowState
    estimate_status: str = UNVERIFIED
    evidence_domain: str = REAL_MOCK


class ClockConverter(Protocol):
    def to_monotonic(self, source_time: float, clock_domain: str) -> tuple[float, float]:
        """Return (converted monotonic time, conservative error bound seconds)."""


class StateEstimator(Protocol):
    def estimate(self, lowstate: DecodedLowState) -> RealStateEstimate:
        """Pure-data external estimator/FK interface; no implementation is assumed."""


class RealAdapter:
    """Validate and compose offline Real fields into the shared RobotSnapshot."""

    def __init__(
        self,
        runtime,
        calibration: RealCalibration,
        imu_extrinsic: ImuExtrinsic,
        timing: RealTimingContract,
        *,
        clock_converter: ClockConverter | None = None,
        estimator: StateEstimator | None = None,
    ):
        _require_mock(calibration.evidence_domain)
        _require_mock(imu_extrinsic.evidence_domain)
        _require_mock(timing.evidence_domain)
        self.runtime = runtime
        self.calibration = calibration
        self.extrinsic = _rotation(imu_extrinsic.quat_ib_wxyz, "R_IB quaternion")
        self.timing = timing
        self.clock_converter = clock_converter
        self.estimator = estimator
        self.joint_names = tuple(runtime.joint_names)
        if (
            len(calibration.joints) != 29
            or tuple(item.joint_name for item in calibration.joints) != self.joint_names
        ):
            raise ContractError("Real calibration must follow frozen S3 joint names exactly")
        self.sdk_indices = tuple(
            _integer(item.sdk_index, "SDK motor index", 0) for item in calibration.joints
        )
        if set(self.sdk_indices) != set(range(29)):
            raise ContractError("Real SDK motor map must be a one-to-one permutation of 0..28")
        signs = np.array([item.sign for item in calibration.joints])
        if not np.all(np.isin(signs, (-1, 1))):
            raise ContractError("Every Real motor sign must be -1 or +1")
        self.signs = _array(signs, (29,), "motor signs")
        self.offsets = _array(
            [item.zero_offset_rad for item in calibration.joints], (29,), "motor zero offsets"
        )
        if any(item.status != UNVERIFIED for item in calibration.joints):
            raise ContractError("P3 cannot promote motor calibration beyond UNVERIFIED")
        metadata = json.loads(
            (Path(__file__).with_name("policy") / "s3_runtime_metadata.json").read_text()
        )
        if tuple(metadata["joint_names"]) != self.joint_names:
            raise ContractError("Frozen Real joint metadata mismatch")
        self.q0 = _array(runtime.q0, (29,), "q0")
        self.action_scale = _array(runtime.action_scale, (29,), "action scale")
        if not np.array_equal(self.q0, metadata["q0"]) or not np.array_equal(
            self.action_scale, metadata["action_scale"]
        ):
            raise ContractError("Real adapter differs from frozen S3 q0/action scale")
        self.kp = _array(metadata["kp"], (29,), "kp")
        self.kd = _array(metadata["kd"], (29,), "kd")
        self.effort = _array(metadata["effort_Nm"], (29,), "effort")
        self._last_sequence = -1
        self._last_times = None
        self._last_estimate_sequence = -1

    def _timing(self):
        values = (
            self.timing.max_state_age_s,
            self.timing.max_interfield_skew_s,
            self.timing.max_clock_error_s,
        )
        if any(value is None for value in values):
            raise ContractError(
                "U5 UNVERIFIED: explicit state-age/skew/clock-error budgets required"
            )
        age, skew, clock = (_time(value, "Real timing budget") for value in values)
        if age <= 0 or skew < 0 or clock < 0:
            raise ContractError("Invalid Real timing budgets")
        return age, skew, clock

    def _source_time(self, value, domain, name):
        source = _time(value, name)
        if not domain:
            raise ContractError("Missing source clock domain")
        if domain == "monotonic":
            return source, 0.0
        if self.clock_converter is None:
            raise ContractError("U5 UNVERIFIED: cross-clock data requires an explicit converter")
        converted, error = self.clock_converter.to_monotonic(source, domain)
        return _time(converted, f"converted {name}"), _time(error, "clock error bound")

    def _motor_to_s3(self, values, sdk_indices, name):
        raw = _array(values, (29,), name)
        if tuple(_integer(index, "LowState motor index", 0) for index in sdk_indices) != tuple(
            sorted(sdk_indices)
        ) or set(sdk_indices) != set(range(29)):
            raise ContractError("LowState motor slots must be ordered unique indices 0..28")
        # Each S3 joint reads its explicitly named SDK slot.  argsort is only
        # for the inverse direction (emitting slots in SDK order), not here.
        return raw[np.asarray(self.sdk_indices)]

    def decode_low_state(self, frame: LowStateFrame, *, now: float) -> DecodedLowState:
        _require_mock(frame.evidence_domain)
        now = _time(now, "decode now")
        max_age, max_skew, max_clock_error = self._timing()
        sequence = _integer(frame.sequence, "LowState sequence", 0)
        if frame.mode_machine is None:
            raise ContractError("Missing mode_machine; simulation bypass is forbidden")
        mode_machine = _integer(frame.mode_machine, "mode_machine", 0)
        if mode_machine == 0:
            raise ContractError("mode_machine=0 is a simulation value, not a Real model match")
        if type(frame.motor_valid) is not bool or type(frame.imu_valid) is not bool:
            raise ContractError("LowState validity flags must be Boolean")
        if not frame.motor_valid or not frame.imu_valid:
            raise ContractError("Invalid motor or IMU source data")
        motor, motor_error = self._source_time(
            frame.motor_measured_at, frame.clock_domain, "motor measured_at"
        )
        imu, imu_error = self._source_time(
            frame.imu_measured_at, frame.clock_domain, "IMU measured_at"
        )
        received = _time(frame.received_monotonic, "received_monotonic")
        if received > now or motor > now or imu > now:
            raise ContractError("Future-dated LowState or receive time")
        if motor - received > motor_error or imu - received > imu_error:
            raise ContractError("Source measurement is later than its reception")
        clock_error = max(motor_error, imu_error)
        if clock_error > max_clock_error:
            raise ContractError("Clock conversion error exceeds the explicit mock bound")
        if now - min(motor, imu) > max_age:
            raise ContractError("Stale source measurement; fresh reception is insufficient")
        if max(motor, imu) - min(motor, imu) + motor_error + imu_error > max_skew:
            raise ContractError("LowState interfield timestamp skew exceeds budget")
        q_sdk = self._motor_to_s3(frame.q_sdk, frame.motor_indices, "motor q")
        dq_sdk = self._motor_to_s3(frame.dq_sdk, frame.motor_indices, "motor dq")
        # q_S3=s*(q_sdk-b), dq_S3=s*dq_sdk.  q0 is not b.
        q_s3 = _array(self.signs * (q_sdk - self.offsets), (29,), "S3 q")
        dq_s3 = _array(self.signs * dq_sdk, (29,), "S3 dq")
        r_wi = _rotation(frame.imu_quat_wi_wxyz, "IMU R_WI quaternion")
        r_wb = r_wi * self.extrinsic.inv()
        omega_b = _array(
            self.extrinsic.inv().apply(_array(frame.gyroscope_i, (3,), "gyroscope")),
            (3,),
            "omega_b",
        )
        torque_values = None
        if frame.tau_est_sdk is not None:
            tau_sdk = self._motor_to_s3(
                frame.tau_est_sdk, frame.motor_indices, "estimated motor torque"
            )
            torque_values = _array(self.signs * tau_sdk, (29,), "S3 estimated torque")
        return DecodedLowState(
            sequence,
            max(motor, imu),
            motor,
            imu,
            received,
            self.joint_names,
            q_s3,
            dq_s3,
            _quat(r_wb),
            omega_b,
            mode_machine,
            TorqueEvidence(
                "ESTIMATED" if torque_values is not None else "UNAVAILABLE", torque_values
            ),
            clock_error,
            motor_error,
            imu_error,
        )

    def _estimate(self, lowstate):
        if self.estimator is None:
            raise ContractError("U1 UNVERIFIED: no root/Anchor state estimator was supplied")
        estimate = self.estimator.estimate(lowstate)
        if not isinstance(estimate, RealStateEstimate):
            raise ContractError("Estimator did not return RealStateEstimate")
        return estimate

    def build_state(
        self,
        frame: LowStateFrame,
        estimate: RealStateEstimate | None = None,
        *,
        now: float,
    ) -> RealState:
        lowstate = self.decode_low_state(frame, now=now)
        estimate = self._estimate(lowstate) if estimate is None else estimate
        _require_mock(estimate.evidence_domain)
        if estimate.status != UNVERIFIED:
            raise ContractError("P3 cannot promote a root estimate beyond UNVERIFIED")
        if not estimate.estimate_id or estimate.lowstate_sequence != lowstate.sequence:
            raise ContractError("Estimate identity does not match LowState")
        if not estimate.valid or estimate.estimator_reset:
            raise ContractError("Invalid/reset root estimate must not be realigned or zero-filled")
        estimate_sequence = _integer(estimate.lowstate_sequence, "estimate sequence", 0)
        if (
            lowstate.sequence <= self._last_sequence
            or estimate_sequence <= self._last_estimate_sequence
        ):
            raise ContractError("Duplicate/out-of-order LowState or estimate")
        estimated, estimate_error = self._source_time(
            estimate.measured_at, estimate.clock_domain, "estimate measured_at"
        )
        max_age, max_skew, max_clock_error = self._timing()
        if estimated > now or now - estimated > max_age:
            raise ContractError("Future-dated or stale root estimate")
        if estimate_error > max_clock_error:
            raise ContractError("Estimate clock error exceeds bound")
        times = (
            (lowstate.motor_measured_monotonic, lowstate.motor_clock_error_s),
            (lowstate.imu_measured_monotonic, lowstate.imu_clock_error_s),
            (estimated, estimate_error),
        )
        if (
            max(value for value, _ in times)
            - min(value for value, _ in times)
            + sum(error for _, error in times)
            > max_skew
        ):
            raise ContractError("Motor/IMU/estimate interfield skew exceeds budget")
        source_times = (
            lowstate.motor_measured_monotonic,
            lowstate.imu_measured_monotonic,
            estimated,
        )
        if self._last_times is not None and any(
            current < previous
            for current, previous in zip(source_times, self._last_times, strict=True)
        ):
            raise ContractError("Source timestamps moved backwards")
        root_pos = _array(estimate.root_pos_w, (3,), "root position")
        root_vel = _array(estimate.root_lin_vel_w, (3,), "root linear velocity")
        anchor_b = _array(estimate.anchor_pos_b, (3,), "relative Anchor position")
        r_ba = _rotation(estimate.anchor_quat_ba_wxyz, "R_BA quaternion")
        r_wb = _rotation(lowstate.root_quat_wb_wxyz, "R_WB quaternion")
        anchor_w = _array(root_pos + r_wb.apply(anchor_b), (3,), "world Anchor position")
        anchor_quat = _quat(r_wb * r_ba)
        robot = RobotSnapshot(
            f"real_mock:{lowstate.sequence}:{estimate.estimate_id}",
            "monotonic",
            max(source_times),
            self.joint_names,
            lowstate.q_s3,
            lowstate.dq_s3,
            lowstate.root_quat_wb_wxyz,
            lowstate.omega_b,
            anchor_w,
            anchor_quat,
        )
        self._last_sequence = lowstate.sequence
        self._last_estimate_sequence = estimate_sequence
        self._last_times = source_times
        return RealState(robot, root_pos, root_vel, estimate.estimate_id, lowstate)

    def snapshot(
        self,
        frame: LowStateFrame,
        estimate: RealStateEstimate | None = None,
        *,
        now: float,
    ) -> RobotSnapshot:
        return self.build_state(frame, estimate, now=now).robot

    def command_encoder(self):
        return RealCommandEncoder(self)


@dataclass(frozen=True)
class ActuationIntent:
    run_id: str
    epoch: int
    tick: int
    frame: int
    created_at: float
    checkpoint_sha256: str
    raw_action: np.ndarray
    q_policy: np.ndarray

    @classmethod
    def from_accepted(cls, request, accepted: AcceptedAction, checkpoint_sha256, created_at):
        if accepted.frame != request.frame:
            raise ContractError("Accepted action/request frame mismatch")
        raw = _array(accepted.raw_action, (29,), "raw action", np.float32)
        target = _array(accepted.q_policy, (29,), "q_policy")
        return cls(
            request.run_id,
            _integer(request.epoch, "epoch", 0),
            _integer(request.tick, "tick", 1),
            _integer(request.frame, "frame", 0),
            _time(created_at, "intent created_at"),
            str(checkpoint_sha256),
            raw,
            target,
        )


@dataclass(frozen=True)
class MotorCommandSpec:
    """Explicit nominal q-PD candidate fields; zeros are never implicit defaults."""

    dq_cmd_s3: np.ndarray
    tau_ff_s3: np.ndarray
    mode_machine: int
    motor_mode: int
    evidence_domain: str = REAL_MOCK


@dataclass(frozen=True)
class MotorSlotCommand:
    joint_name: str
    sdk_index: int
    q_cmd: float
    dq_cmd: float
    kp: float
    kd: float
    tau_ff: float
    mode: int


@dataclass(frozen=True)
class OfflineLowCmd:
    run_id: str
    epoch: int
    tick: int
    frame: int
    sequence: int
    created_at: float
    lowstate_sequence: int
    checkpoint_sha256: str
    raw_action: np.ndarray
    q_policy_s3: np.ndarray
    q_cmd_s3: np.ndarray
    dq_cmd_s3: np.ndarray
    kp_s3: np.ndarray
    kd_s3: np.ndarray
    tau_ff_s3: np.ndarray
    measured_q_s3: np.ndarray
    measured_dq_s3: np.ndarray
    predicted_tau_s3: np.ndarray
    predicted_tau_sdk: np.ndarray
    estimated_tau_s3: np.ndarray | None
    slots: tuple[MotorSlotCommand, ...]
    mode_machine: int
    encoding: str = COMMAND_ENCODING
    crc: None = None
    crc_status: str = "UNVERIFIED_NOT_ENCODED"
    actual_torque_status: str = UNVERIFIED
    transport_status: str = "OFFLINE_MEMORY_ONLY_NOT_SERIALIZED_OR_PUBLISHED"
    evidence_domain: str = REAL_MOCK


@dataclass(frozen=True)
class CommandEnvelope:
    """Complete mock-only envelope; no component is optional or silently unlimited."""

    dt_s: float
    max_q_cmd_step_rad: np.ndarray
    max_dq_cmd_step_rad_s: np.ndarray
    max_kp_step: np.ndarray
    max_kd_step: np.ndarray
    max_tau_ff_step_nm: np.ndarray
    max_predicted_tau_step_nm: np.ndarray
    max_predicted_tau_rate_nm_s: np.ndarray
    evidence_domain: str = REAL_MOCK

    def validated(self):
        _require_mock(self.evidence_domain)
        dt = _time(self.dt_s, "command envelope dt")
        if dt <= 0:
            raise ContractError("Command envelope dt must be positive")
        names = (
            "max_q_cmd_step_rad",
            "max_dq_cmd_step_rad_s",
            "max_kp_step",
            "max_kd_step",
            "max_tau_ff_step_nm",
            "max_predicted_tau_step_nm",
            "max_predicted_tau_rate_nm_s",
        )
        limits = []
        for name in names:
            value = _array(getattr(self, name), (29,), name)
            if np.any(value < 0):
                raise ContractError(f"Negative {name}")
            limits.append(value)
        return dt, limits


@dataclass(frozen=True)
class CommandInspection:
    effort_status: str
    continuity_status: str
    offline_write_allowed: bool
    violations: tuple[str, ...]
    max_abs_changes: dict[str, float]
    predicted_tau_delta_at_current_state: np.ndarray | None
    real_actuator_status: str = UNVERIFIED
    actual_torque_comparison_status: str = UNVERIFIED
    evidence_domain: str = REAL_MOCK


class RealCommandEncoder:
    """Encode separate LowCmd fields and diagnose, never repair, q-PD commands."""

    def __init__(self, adapter: RealAdapter):
        self.adapter = adapter

    def encode(
        self,
        intent: ActuationIntent,
        measurement: DecodedLowState,
        *,
        q_cmd_s3,
        spec: MotorCommandSpec,
        sequence: int,
        created_at: float,
    ) -> OfflineLowCmd:
        _require_mock(spec.evidence_domain)
        if intent.checkpoint_sha256 != self.adapter.runtime.checkpoint_sha256:
            raise ContractError("Command checkpoint differs from frozen S3 runtime")
        if not intent.run_id:
            raise ContractError("Missing command run identity")
        _integer(intent.epoch, "command epoch", 0)
        _integer(intent.tick, "command tick", 1)
        _integer(intent.frame, "command frame", 0)
        sequence = _integer(sequence, "command sequence", 0)
        created = _time(created_at, "command created_at")
        if created < intent.created_at or created < measurement.received_monotonic:
            raise ContractError("Command time precedes intent or measurement")
        max_age, _, _ = self.adapter._timing()
        if (
            created < measurement.measured_monotonic
            or created
            - min(measurement.motor_measured_monotonic, measurement.imu_measured_monotonic)
            > max_age
        ):
            raise ContractError("Command uses a future-dated or stale measurement")
        q_policy = _array(intent.q_policy, (29,), "nominal q_policy")
        raw = _array(intent.raw_action, (29,), "raw action", np.float32)
        expected = self.adapter.q0 + self.adapter.action_scale * raw
        if not np.array_equal(q_policy, expected):
            raise ContractError("q_policy is not q0+scale*raw_action")
        q_cmd = _array(q_cmd_s3, (29,), "q_cmd")
        dq_cmd = _array(spec.dq_cmd_s3, (29,), "dq_cmd")
        tau_ff = _array(spec.tau_ff_s3, (29,), "tau_ff")
        mode_machine = _integer(spec.mode_machine, "command mode_machine", 0)
        motor_mode = _integer(spec.motor_mode, "motor mode", 0)
        if mode_machine == 0 or mode_machine != measurement.mode_machine:
            raise ContractError("Command/LowState mode_machine mismatch or simulation bypass")
        q_meas = _array(measurement.q_s3, (29,), "measured q")
        dq_meas = _array(measurement.dq_s3, (29,), "measured dq")
        predicted = (
            self.adapter.kp * (q_cmd - q_meas) + self.adapter.kd * (dq_cmd - dq_meas) + tau_ff
        )
        predicted = _array(predicted, (29,), "predicted total torque")
        q_sdk = self.adapter.signs * q_cmd + self.adapter.offsets
        dq_sdk = self.adapter.signs * dq_cmd
        tau_ff_sdk = self.adapter.signs * tau_ff
        predicted_sdk = self.adapter.signs * predicted
        by_sdk = np.argsort(np.asarray(self.adapter.sdk_indices))
        predicted_sdk_slots = predicted_sdk[by_sdk]
        slots = tuple(
            MotorSlotCommand(
                self.adapter.joint_names[s3_index],
                self.adapter.sdk_indices[s3_index],
                float(q_sdk[s3_index]),
                float(dq_sdk[s3_index]),
                float(self.adapter.kp[s3_index]),
                float(self.adapter.kd[s3_index]),
                float(tau_ff_sdk[s3_index]),
                motor_mode,
            )
            for s3_index in by_sdk
        )
        if len(slots) != 29 or tuple(slot.sdk_index for slot in slots) != tuple(range(29)):
            raise ContractError("Offline LowCmd must contain exactly approved slots 0..28")
        estimated = measurement.torque_evidence.values_s3
        return OfflineLowCmd(
            intent.run_id,
            intent.epoch,
            intent.tick,
            intent.frame,
            sequence,
            created,
            measurement.sequence,
            intent.checkpoint_sha256,
            raw,
            q_policy,
            q_cmd,
            dq_cmd,
            self.adapter.kp,
            self.adapter.kd,
            tau_ff,
            q_meas,
            dq_meas,
            predicted,
            _array(predicted_sdk_slots, (29,), "SDK-slot predicted torque"),
            None if estimated is None else _array(estimated, (29,), "estimated torque"),
            slots,
            mode_machine,
        )

    def inspect(
        self,
        current: OfflineLowCmd,
        *,
        previous: OfflineLowCmd | None,
        envelope: CommandEnvelope | None,
    ) -> CommandInspection:
        for name, packet in (("current", current), ("previous", previous)):
            if packet is not None and (
                packet.encoding != COMMAND_ENCODING
                or packet.evidence_domain != REAL_MOCK
                or packet.checkpoint_sha256 != self.adapter.runtime.checkpoint_sha256
                or len(packet.slots) != 29
            ):
                raise ContractError(f"Invalid {name} offline command identity/encoding")
        _array(current.predicted_tau_s3, (29,), "current predicted torque")
        violations = []
        effort_over = np.abs(current.predicted_tau_s3) > self.adapter.effort
        if effort_over.any():
            indices = np.flatnonzero(effort_over).tolist()
            violations.append(f"predicted_total_torque_exceeds_frozen_effort:{indices}")
        effort_status = "FAIL" if effort_over.any() else "PASS"
        changes = {}
        predicted_delta = None
        if previous is None:
            violations.append("previous_or_factory_last_command_unavailable")
        if envelope is None:
            violations.append("U3/U5 UNVERIFIED: complete command envelope unavailable")
        if previous is None or envelope is None:
            return CommandInspection(
                effort_status,
                UNVERIFIED,
                False,
                tuple(violations),
                changes,
                predicted_delta,
            )
        dt, limits = envelope.validated()
        fields = (
            "q_cmd_s3",
            "dq_cmd_s3",
            "kp_s3",
            "kd_s3",
            "tau_ff_s3",
        )
        names = ("q_cmd", "dq_cmd", "kp", "kd", "tau_ff")
        for field, name, limit in zip(fields, names, limits[:5], strict=True):
            delta = np.abs(getattr(current, field) - getattr(previous, field))
            changes[name] = float(np.max(delta))
            if np.any(delta > limit):
                violations.append(f"{name}_change_exceeds_explicit_mock_envelope")
        previous_at_current = (
            previous.kp_s3 * (previous.q_cmd_s3 - current.measured_q_s3)
            + previous.kd_s3 * (previous.dq_cmd_s3 - current.measured_dq_s3)
            + previous.tau_ff_s3
        )
        predicted_delta = _array(
            current.predicted_tau_s3 - previous_at_current,
            (29,),
            "same-state predicted torque delta",
        )
        changes["predicted_tau"] = float(np.max(np.abs(predicted_delta)))
        changes["predicted_tau_rate"] = float(np.max(np.abs(predicted_delta) / dt))
        if np.any(np.abs(predicted_delta) > limits[5]):
            violations.append("predicted_tau_change_exceeds_explicit_mock_envelope")
        if np.any(np.abs(predicted_delta) / dt > limits[6]):
            violations.append("predicted_tau_rate_exceeds_explicit_mock_envelope")
        continuity = "FAIL" if any("envelope" in item for item in violations) else "PASS"
        return CommandInspection(
            effort_status,
            continuity,
            effort_status == "PASS" and continuity == "PASS",
            tuple(violations),
            changes,
            predicted_delta,
        )


@dataclass(frozen=True)
class WriterLease:
    writer_id: str
    run_id: str
    epoch: int


class OfflineCommandWriter:
    """Single-lease, epoch-gated in-memory command recorder; no publish method."""

    def __init__(self, writer_id: str, encoder: RealCommandEncoder):
        if not writer_id:
            raise ContractError("A unique offline writer identity is required")
        self.writer_id = writer_id
        self.encoder = encoder
        self.lease = None
        self.retired_epochs = set()
        self.records = []
        self.rejections = []
        self._written_identities = set()
        self.inflight_offline = 0

    @property
    def closed(self):
        return self.lease is None

    def authorize(self, run_id, epoch):
        epoch = _integer(epoch, "writer epoch", 0)
        if not run_id or epoch in self.retired_epochs:
            raise ContractError("Cannot authorize a missing/retired writer epoch")
        if self.lease is not None:
            if self.lease == WriterLease(self.writer_id, run_id, epoch):
                return self.lease
            raise ContractError("A second writer or epoch cannot share the active lease")
        self.lease = WriterLease(self.writer_id, run_id, epoch)
        return self.lease

    def revoke(self, epoch, reason):
        self.retired_epochs.add(_integer(epoch, "revoked epoch", 0))
        if self.lease is not None and self.lease.epoch == epoch:
            self.lease = None
        self.rejections.append({"epoch": epoch, "reason": str(reason)})

    def write(self, command: OfflineLowCmd, *, previous, envelope):
        lease = self.lease
        identity = (command.run_id, command.epoch)
        if (
            lease is None
            or identity != (lease.run_id, lease.epoch)
            or command.epoch in self.retired_epochs
        ):
            self.rejections.append({"epoch": command.epoch, "reason": "unauthorized_or_retired"})
            raise ContractError("Offline command rejected by unique-writer/epoch gate")
        command_identities = {
            (command.run_id, command.epoch, "tick", command.tick),
            (command.run_id, command.epoch, "sequence", command.sequence),
        }
        if self._written_identities.intersection(command_identities):
            self.rejections.append({"epoch": command.epoch, "reason": "duplicate_command"})
            raise ContractError("Duplicate offline command tick/sequence rejected")
        inspection = self.encoder.inspect(command, previous=previous, envelope=envelope)
        if not inspection.offline_write_allowed:
            self.rejections.append(
                {
                    "epoch": command.epoch,
                    "reason": "command_contract",
                    "violations": inspection.violations,
                }
            )
            raise ContractError(f"Offline command contract rejected: {inspection.violations}")
        self.records.append((command, inspection))
        self._written_identities.update(command_identities)
        self.inflight_offline += 1
        return inspection

    def drain_offline(self, count=None):
        """Test harness acknowledgement only; not a DDS/robot delivery receipt."""

        if count is None:
            self.inflight_offline = 0
        else:
            count = _integer(count, "offline drain count", 0)
            if count > self.inflight_offline:
                raise ContractError("Cannot drain more offline records than are outstanding")
            self.inflight_offline -= count
