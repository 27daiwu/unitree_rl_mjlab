"""Backend-independent, frozen S3 inference and session contracts.

No control, scheduling, FK, simulator state, DDS or ownership RPC lives here.
Physical snapshots and a clock are supplied by the caller. S10A is deliberately
not wired to this module in P1.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ._s3_frozen_network import FrozenS3Network

METADATA_SHA256 = "83c52d8db8c12afa7a202136c8b6f1e0f701e933aee3fb4eb46bb75f2e8eac10"
RAW_CONTRACT = "S3_PHYSICAL_UNSCALED_V1"
TENSOR_CONTRACT = "S3_RAW_930_689_NETWORK_SCALES_INTERNAL_V1"
TERM_SIZES = (29, 29, 3, 3, 29)


class ContractError(ValueError):
    """Invalid data, assets, state, identity or time; never a control fallback."""


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array(value, shape, name, dtype=None):
    try:
        result = np.array(value, dtype=dtype, copy=True)
        valid = result.shape == shape and np.isfinite(result).all()
    except (TypeError, ValueError) as error:
        raise ContractError(f"Missing/invalid {name}") from error
    if not valid:
        raise ContractError(f"Missing/invalid {name}: expected {shape}")
    result.flags.writeable = False
    return result


def _rotation(value):
    q = _array(value, (4,), "wxyz quaternion", np.float64)
    if abs(np.linalg.norm(q) - 1.0) > 1e-5:
        raise ContractError("Quaternion must be unit wxyz")
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def _finite_time(value):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ContractError("Missing/nonfinite time")
    return float(value)


def _digest(obs, goal):
    return hashlib.sha256(obs.tobytes() + goal.tobytes()).hexdigest()


@dataclass(frozen=True)
class RobotSnapshot:
    snapshot_id: str
    clock_domain: str
    measured_at: float
    joint_names: tuple[str, ...]
    q: np.ndarray
    dq: np.ndarray
    root_quat_wxyz: np.ndarray
    omega_b: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_wxyz: np.ndarray
    encoding: str = RAW_CONTRACT
    valid: bool = True


@dataclass(frozen=True)
class InferenceRequest:
    run_id: str
    epoch: int
    tick: int
    stage: str
    frame: int
    snapshot_id: str
    clock_domain: str
    sent_at: float
    deadline: float
    obs: np.ndarray
    goal: np.ndarray
    digest: str
    encoding: str = TENSOR_CONTRACT


@dataclass(frozen=True)
class PolicyResult:
    run_id: str
    epoch: int
    tick: int
    stage: str
    frame: int
    digest: str
    clock_domain: str
    completed_at: float
    raw_action: np.ndarray
    checkpoint_sha256: str


@dataclass(frozen=True)
class AcceptedAction:
    frame: int
    stage: str
    raw_action: np.ndarray
    q_policy: np.ndarray


class PolicyRuntime:
    """Immutable frozen inputs/model; separate sessions own history and phase."""

    def __init__(self, root: Path | None = None):
        root = Path(root) if root is not None else Path(__file__).resolve().parents[5]
        metadata_path = Path(__file__).with_name("s3_runtime_metadata.json")
        if _sha(metadata_path) != METADATA_SHA256:
            raise ContractError("Frozen metadata mismatch")
        meta = json.loads(metadata_path.read_text())
        manifest_path = root / "extensions/supermimic/src/supermimic/s10a_assets.json"
        if _sha(manifest_path) != meta["manifest_sha256"]:
            raise ContractError("Frozen manifest mismatch")
        self.asset_hashes = json.loads(manifest_path.read_text())["sha256"]
        for path, expected in self.asset_hashes.items():
            try:
                actual = _sha(root / path)
            except OSError as error:
                raise ContractError(f"Missing frozen asset: {path}") from error
            if actual != expected:
                raise ContractError(f"Frozen asset mismatch: {path}")
        paths = {key: root / value for key, value in meta["primary_paths"].items()}
        cfg = json.loads(paths["config"].read_text())
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        saved_cfg = json.loads(json.dumps(checkpoint["config"], default=str))
        if (
            saved_cfg != cfg
            or cfg["data"]["clips"] != ["dance1_subject2"]
            or checkpoint["iteration"] != 36800
            or cfg["policy"]["history"] != 10
            or cfg["policy"]["goal_dim"] != 689
            or cfg["env"]["future_frames"] != 10
            or not cfg["env"]["anchor_obs"]
        ):
            raise ContractError("S3 checkpoint/config/clip contract mismatch")
        self.checkpoint_sha256 = self.asset_hashes[meta["primary_paths"]["checkpoint"]]
        self.joint_names = tuple(meta["joint_names"])
        self.q0 = _array(meta["q0"], (29,), "q0", np.float64)
        self.action_scale = _array(meta["action_scale"], (29,), "action scale", np.float64)
        if len(set(self.joint_names)) != 29 or np.any(self.action_scale <= 0):
            raise ContractError("Invalid frozen joint metadata")
        # Preserve the original module keys/operations without importing its
        # training/simulator dependencies. Every tensor is overwritten strictly.
        with torch.random.fork_rng(devices=[]):
            self._network = FrozenS3Network(cfg["policy"]).cpu().eval()
        expected_scales = {
            key: value.clone()
            for key, value in self._network.state_dict().items()
            if ".scale" in key
        }
        self._network.load_state_dict(checkpoint["policy"], strict=True)
        for key, value in expected_scales.items():
            if not torch.equal(self._network.state_dict()[key], value):
                raise ContractError("S3 internal scaling buffer mismatch")
        self._network.requires_grad_(False)
        del checkpoint
        with np.load(paths["motion"], allow_pickle=False) as motion:
            if tuple(motion["joint_names"].astype(str)) != self.joint_names:
                raise ContractError("S3 motion joint order mismatch")
            if motion["fps"].shape != (1,) or motion["fps"][0] != 50:
                raise ContractError("S3 motion frequency mismatch")
            bodies = motion["body_names"].astype(str).tolist()
            root_body, anchor = bodies.index("pelvis"), bodies.index("torso_link")
            ref = np.concatenate(
                [
                    motion["body_pos_w"][:, root_body],
                    motion["body_quat_w"][:, root_body],
                    motion["joint_pos"],
                    motion["body_lin_vel_w"][:, root_body],
                    motion["body_ang_vel_w"][:, root_body],
                    motion["joint_vel"],
                ],
                axis=1,
            ).astype(np.float32)
            self.anchor_ref_pos = _array(
                motion["body_pos_w"][:, anchor], (6574, 3), "motion Anchor"
            )
            self._anchor_ref_rot = Rotation.from_quat(
                motion["body_quat_w"][:, anchor][:, [1, 2, 3, 0]]
            )
        self.reference = _array(ref, (6574, 71), "reference")
        self.features = _array(self._features(ref), (6574, 68), "reference features", np.float32)

    def _features(self, ref):
        # Same precision and operation order as the sealed goal_scipy path.
        ref = np.asarray(ref, np.float64)
        roots = Rotation.from_quat(ref[:, [4, 5, 6, 3]])
        gravity = roots.inv().apply(np.broadcast_to([0.0, 0.0, -1.0], (len(ref), 3)))
        angular = roots.inv().apply(ref[:, 39:42])
        w, x, y, z = ref[:, 3:7].T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        heading = Rotation.from_rotvec(np.stack([yaw * 0, yaw * 0, yaw], axis=-1))
        linear = heading.inv().apply(ref[:, 36:39])
        return np.concatenate(
            [ref[:, 7:36] - self.q0, ref[:, 42:71], gravity, angular, linear, ref[:, 2:3]],
            axis=-1,
        )

    def session(self, run_id, *, clock_domain, max_snapshot_age_s, request_ttl_s=0.02):
        return PolicySession(self, run_id, clock_domain, max_snapshot_age_s, request_ttl_s)

    def infer(
        self,
        request: InferenceRequest,
        *,
        clock: Callable[[], float] = time.monotonic,
        clock_domain: str = "monotonic",
        executor: Callable | None = None,
    ) -> PolicyResult:
        """Forward only; Session.accept is the independent output gate.

        An optional tensor executor preserves an existing isolated transport.
        It receives the original unscaled FP32 tensors. Scheduling, IPC and
        supervision remain external; exceptions (including pending) propagate.
        """
        now = _finite_time(clock())
        if request.clock_domain != clock_domain or not request.sent_at <= now < request.deadline:
            raise ContractError("Wrong clock or expired inference request")
        if (
            np.asarray(request.obs).dtype != np.float32
            or np.asarray(request.goal).dtype != np.float32
        ):
            raise ContractError("Request tensors must be float32")
        obs = _array(request.obs, (930,), "obs", np.float32)
        goal = _array(request.goal, (689,), "goal", np.float32)
        if request.encoding != TENSOR_CONTRACT or _digest(obs, goal) != request.digest:
            raise ContractError("Changed/scaled request tensors or encoding")
        with torch.inference_mode():
            raw = (self._network if executor is None else executor)(
                torch.from_numpy(obs.copy()[None]), torch.from_numpy(goal.copy()[None])
            )
        completed = _finite_time(clock())
        if not now <= completed < request.deadline:
            raise ContractError("Inference completed after deadline or clock moved backwards")
        return PolicyResult(
            request.run_id,
            request.epoch,
            request.tick,
            request.stage,
            request.frame,
            request.digest,
            clock_domain,
            completed,
            _array(raw.numpy()[0], (29,), "raw action", np.float32),
            self.checkpoint_sha256,
        )


class PolicySession:
    """One authorized epoch, at most one request, caller-driven 50 Hz ticks."""

    def __init__(self, runtime, run_id, clock_domain, max_age, ttl):
        if not run_id or not clock_domain or _finite_time(max_age) <= 0 or _finite_time(ttl) <= 0:
            raise ContractError("Run, clock and finite positive age/TTL budgets are required")
        self.runtime, self.run_id, self.clock_domain = runtime, run_id, clock_domain
        self.max_age, self.ttl = float(max_age), float(ttl)
        self.epoch = -1
        self.stage = "IDLE"
        self.phase = self.warmup_count = self.tick = 0
        self.pending = None
        self._history = self._previous = None
        self._last_now = -np.inf
        self._last_snapshot_time = -np.inf
        self.retired_reason = None

    @property
    def history(self):
        return None if self._history is None else self._history.copy()

    @property
    def previous(self):
        return None if self._previous is None else self._previous.copy()

    @property
    def clip_complete(self):
        return self.stage == "COMPLETE" and self.phase == 6574

    def _now(self, value):
        value = _finite_time(value)
        if value < self._last_now:
            raise ContractError("Session clock moved backwards")
        self._last_now = value
        return value

    def _snapshot(self, snapshot, now):
        if snapshot.encoding != RAW_CONTRACT or not snapshot.valid:
            raise ContractError("Snapshot is invalid or already scaled")
        if not snapshot.snapshot_id or snapshot.clock_domain != self.clock_domain:
            raise ContractError("Missing snapshot identity or wrong clock domain")
        measured = _finite_time(snapshot.measured_at)
        if not 0 <= now - measured <= self.max_age or measured < self._last_snapshot_time:
            raise ContractError("Snapshot is stale, future dated or out of order")
        if snapshot.joint_names is None or tuple(snapshot.joint_names) != self.runtime.joint_names:
            raise ContractError("Snapshot joint order mismatch")
        values = (
            _array(snapshot.q, (29,), "joint q", np.float64),
            _array(snapshot.dq, (29,), "joint dq", np.float64),
            _rotation(snapshot.root_quat_wxyz),
            _array(snapshot.omega_b, (3,), "omega_b", np.float64),
            _array(snapshot.anchor_pos_w, (3,), "Anchor position", np.float64),
            _rotation(snapshot.anchor_quat_wxyz),
        )
        return measured, values

    def activate(self, snapshot, q_source, *, epoch: int, now: float):
        now = self._now(now)
        if (
            self.stage not in ("IDLE", "RETIRED", "COMPLETE")
            or type(epoch) is not int
            or epoch <= self.epoch
        ):
            raise ContractError(
                "Activation requires an inactive session and a new supervisor epoch"
            )
        measured, (_, _, _, _, position, live) = self._snapshot(snapshot, now)
        target = _array(q_source, (29,), "known source target", np.float64)
        ref = self.runtime._anchor_ref_rot[0]
        yaw = live.as_euler("xyz")[2] - ref.as_euler("xyz")[2]
        self._align_rot = Rotation.from_euler("z", yaw)
        offset = position - self._align_rot.apply(self.runtime.anchor_ref_pos[0])
        offset[2] = 0.0
        self._align_pos = offset
        self._previous = ((target - self.runtime.q0) / self.runtime.action_scale).astype(np.float32)
        self._history = None
        self.epoch, self.stage = epoch, "WARMUP"
        self.phase = self.warmup_count = self.tick = 0
        self.pending = self.retired_reason = None
        self._last_snapshot_time = measured
        self.alignment = {
            "yaw_rad": float(yaw),
            "translation": offset.tolist(),
            "frame": 0,
            "history": "repeat live frame x10, term-major",
            "previous_action": self._previous.tolist(),
        }
        return self.alignment.copy()

    def begin_dance(self):
        if self.stage != "WARMUP" or self.pending is not None or self.warmup_count < 1:
            raise ContractError("DANCE requires completed warmup and no pending request")
        self.stage, self.phase = "DANCE", 0

    def build_request(self, snapshot, *, now: float):
        now = self._now(now)
        if self.stage not in ("WARMUP", "DANCE"):
            raise ContractError("No inference allowed in inactive/complete/retired session")
        if self.pending is not None:
            if now >= self.pending.deadline:
                self.retire("request_expired")
            raise ContractError("Pending or expired request; no duplicate history advance")
        measured, (q, dq, root, omega, position, live) = self._snapshot(snapshot, now)
        terms = (q - self.runtime.q0, dq, omega, root.inv().apply([0.0, 0.0, -1.0]), self._previous)
        frame = np.concatenate(terms)
        if self._history is None:
            history = np.repeat(frame[None], 10, axis=0)
        else:
            history = np.concatenate((self._history[1:], frame[None]), axis=0)
        boundaries = np.cumsum((0,) + TERM_SIZES)
        obs = np.concatenate(
            [history[:, start:end].reshape(-1) for start, end in pairwise(boundaries)]
        ).astype(np.float32)
        k = self.phase if self.stage == "DANCE" else 0
        refpos = self._align_rot.apply(self.runtime.anchor_ref_pos[k]) + self._align_pos
        relpos = live.inv().apply(refpos - position)
        relrot = live.inv() * self._align_rot * self.runtime._anchor_ref_rot[k]
        indices = np.minimum(k + np.arange(10), 6573)
        goal = np.r_[
            self.runtime.features[indices].ravel(), relpos, relrot.as_matrix()[:, :2].ravel()
        ]
        obs = _array(obs, (930,), "obs", np.float32)
        goal = _array(goal, (689,), "goal", np.float32)
        self.tick += 1
        request = InferenceRequest(
            self.run_id,
            self.epoch,
            self.tick,
            self.stage,
            k,
            snapshot.snapshot_id,
            self.clock_domain,
            now,
            now + self.ttl,
            obs,
            goal,
            _digest(obs, goal),
        )
        self._history, self._last_snapshot_time, self.pending = history, measured, request
        return request

    def accept(self, result: PolicyResult, *, now: float) -> AcceptedAction:
        now = self._now(now)
        request = self.pending
        if request is None or self.stage not in ("WARMUP", "DANCE"):
            raise ContractError("Late/duplicate result for inactive session")
        identity = (
            result.run_id,
            result.epoch,
            result.tick,
            result.stage,
            result.frame,
            result.digest,
        )
        expected = (
            request.run_id,
            request.epoch,
            request.tick,
            request.stage,
            request.frame,
            request.digest,
        )
        if identity != expected or result.checkpoint_sha256 != self.runtime.checkpoint_sha256:
            raise ContractError("Late/foreign result identity or checkpoint mismatch")
        if (
            result.clock_domain != self.clock_domain
            or not request.sent_at <= result.completed_at <= now
        ):
            raise ContractError("Result clock mismatch")
        if now >= request.deadline:
            self.retire("result_expired")
            raise ContractError("Expired result; epoch retired")
        if np.asarray(result.raw_action).dtype != np.float32:
            raise ContractError("Raw action must be float32")
        action = _array(result.raw_action, (29,), "raw action", np.float32)
        target = _array(self.runtime.q0 + self.runtime.action_scale * action, (29,), "q_policy")
        self._previous, self.pending = action.copy(), None
        if self.stage == "WARMUP":
            self.warmup_count += 1
        else:
            self.phase += 1
            if self.phase == 6574:
                self.stage = "COMPLETE"
        return AcceptedAction(request.frame, request.stage, action, target)

    def retire(self, reason):
        self.pending = None
        self.stage, self.retired_reason = "RETIRED", str(reason)
