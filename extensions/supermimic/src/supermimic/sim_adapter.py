"""MuJoCo boundary and legacy field bridge for the frozen S3 Policy Runtime.

No control law, physics step, IPC worker, watchdog or transition lives here.
The native entrypoint's existing ``previous = action`` completion publication
is also the shared session's asynchronous acceptance boundary.
"""

from __future__ import annotations

import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg
from mjlab.terrains import TerrainEntityCfg

from supermimic.policy.runtime import ContractError, PolicyResult, PolicyRuntime, RobotSnapshot
from supermimic.robot import get_g1_robot_cfg
from supermimic.s10a import MANIFEST, ROOT, DanceRuntime, sha


@contextmanager
def inference_boundary():
    """Defer automatic cyclic GC within the existing 20 ms request budget.

    Restore the caller's GC mode on success, pending and error. Collection is
    otherwise unchanged; it may run in the surrounding non-request work.
    """
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class SimAdapter:
    """Read physical state/FK and bind model arrays to frozen policy joint order."""

    def __init__(self, model, data, diag, runtime):
        self.model, self.data, self.diag = model, data, diag
        m = model
        self.names = [m.joint(i).name.split("/")[-1] for i in range(1, m.njnt)]
        if (m.nq, m.nv, m.nu) != (36, 35, 29) or tuple(self.names) != runtime.joint_names:
            raise ContractError("Sim model dimensions or joint order mismatch")
        self.qadr = m.jnt_qposadr[1:].copy()
        self.vadr = m.jnt_dofadr[1:].copy()
        self.aids = np.array(
            [int(np.flatnonzero(m.actuator_trnid[:, 0] == j)[0]) for j in range(1, m.njnt)]
        )
        if len(set(self.aids)) != 29:
            raise ContractError("Sim actuator map is not one to one")
        self.kp = m.actuator_gainprm[self.aids, 0].copy()
        self.kd = -m.actuator_biasprm[self.aids, 2].copy()
        if not np.array_equal(m.actuator_biasprm[self.aids, 1], -self.kp):
            raise ContractError("Sim requires unchanged native position actuators")
        self.effort = m.actuator_forcerange[self.aids, 1].copy()
        self.q0 = m.key_qpos[0, self.qadr].copy()
        self.scale = 0.25 * self.effort / self.kp
        meta = json.loads(
            Path(__file__).with_name("policy").joinpath("s3_runtime_metadata.json").read_text()
        )
        for actual, expected in (
            (self.q0, runtime.q0),
            (self.scale, runtime.action_scale),
            (self.kp, meta["kp"]),
            (self.kd, meta["kd"]),
            (self.effort, meta["effort_Nm"]),
        ):
            if not np.array_equal(actual, expected):
                raise ContractError("Sim actuator parameters differ from frozen S3 metadata")
        self.qmin, self.qmax = m.jnt_range[1:].T.copy()
        self.anchor = m.body("robot/torso_link").id
        self.pelvis = m.body("robot/pelvis").id
        self.feet = [m.body("robot/" + n + "_ankle_roll_link").id for n in ("left", "right")]
        self.joint_names = runtime.joint_names
        self.snapshot_count = 0

    @classmethod
    def create(cls, runtime):
        scene = Scene(
            SceneCfg(
                terrain=TerrainEntityCfg(terrain_type="plane"),
                entities={"robot": get_g1_robot_cfg()},
            ),
            device="cpu",
        )
        model = scene.compile()
        MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20).apply(model)
        data, diag = mujoco.MjData(model), mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        adapter = cls(model, data, diag, runtime)
        adapter.scene = scene
        return adapter

    def refresh(self):
        self.diag.qpos[:] = self.data.qpos
        self.diag.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, self.diag)

    def snapshot(self, *, measured_at):
        # Caller refreshes once at the same boundary as legacy Handoff.tick.
        # MuJoCo free-joint angular qvel is local; no world-frame re-rotation.
        d = self.diag
        self.snapshot_count += 1
        return RobotSnapshot(
            f"sim:{self.snapshot_count}",
            "monotonic",
            measured_at,
            self.joint_names,
            d.qpos[self.qadr].copy(),
            d.qvel[self.vadr].copy(),
            d.qpos[3:7].copy(),
            d.qvel[3:6].copy(),
            d.xpos[self.anchor].copy(),
            d.xquat[self.anchor].copy(),
        )

    def actuator_commands(self, joint_servo):
        command = np.asarray(joint_servo)
        if command.shape != (29,) or not np.isfinite(command).all():
            raise ContractError("Invalid Sim servo command")
        mapped = np.empty(self.model.nu, dtype=command.dtype)
        mapped[self.aids] = command
        return mapped


class SharedDanceRuntime(DanceRuntime):
    """Legacy controller-facing fields; inference semantics owned by P1 runtime.

    ``policy`` remains the untouched native IPC proxy installed by run_native.
    PolicyRuntime.infer delegates the tensor forward to that existing executor.
    If it raises InferencePending, native supervision continues independently;
    only its existing accepted raw-action publication completes this session.
    """

    def __init__(self, out: Path):
        self.out = out
        self.common = PolicyRuntime()
        self.adapter = SimAdapter.create(self.common)
        for name in (
            "scene",
            "model",
            "data",
            "diag",
            "qadr",
            "vadr",
            "names",
            "aids",
            "kp",
            "kd",
            "effort",
            "q0",
            "scale",
            "qmin",
            "qmax",
            "anchor",
            "pelvis",
            "feet",
        ):
            setattr(self, name, getattr(self.adapter, name))
        self.manifest = json.loads(MANIFEST.read_text())
        self.ref, self.features = self.common.reference, self.common.features
        self.session = self.common.session(
            str(out.resolve()),
            clock_domain="monotonic",
            max_snapshot_age_s=0.005,
            request_ttl_s=0.02,
        )
        self.policy = None
        self.inferences = 0
        self.inject_output = None
        self.interfaces = []
        self.alignment = None
        self.joint_index = {name: i for i, name in enumerate(self.names)}
        self.session_events, self.requests = [], []
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
        paths = [
            Path(__file__),
            Path(__file__).with_name("policy") / "runtime.py",
            Path(__file__).with_name("s10a.py"),
            MANIFEST,
        ]
        (out / "source_sha256.json").write_text(
            json.dumps({str(p.relative_to(ROOT)): sha(p) for p in paths}, indent=2) + "\n"
        )

    @property
    def history(self):
        return self.session.history

    @property
    def previous(self):
        return self.session.previous

    @previous.setter
    def previous(self, action):
        # Called by the unchanged native loop only after IPC accepts a reply.
        with inference_boundary():
            self._accept_polled(action)

    def _accept_polled(self, action):
        request = self.session.pending
        if request is None or getattr(self.policy, "retired", False):
            raise ContractError("No authorized asynchronous completion")
        now = time.monotonic()
        result = PolicyResult(
            request.run_id,
            request.epoch,
            request.tick,
            request.stage,
            request.frame,
            request.digest,
            request.clock_domain,
            now,
            action,
            self.common.checkpoint_sha256,
        )
        self.session.accept(result, now=now)
        self.requests[-1]["accepted"] = True
        self.requests[-1]["completion"] = "native_poll"

    def refresh(self):
        self.adapter.refresh()
        owner = getattr(self.policy, "owner", None)
        if (
            getattr(self.policy, "retired", False) or owner in ("RELEASE", "SAFE_DAMP", "STAND")
        ) and self.session.stage in ("WARMUP", "DANCE"):
            self.session.retire("native_owner_retired")
            self.session_events.append(
                {"event": "retired", "sim_time": float(self.data.time), "owner": owner}
            )

    def activate(self, target):
        self.refresh()
        now = time.monotonic()
        self.alignment = self.session.activate(
            self.adapter.snapshot(measured_at=now),
            target,
            epoch=self.session.epoch + 1,
            now=now,
        )
        self.session_events.append({"event": "activated", "sim_time": float(self.data.time)})
        (self.out / "activation.json").write_text(json.dumps(self.alignment, indent=2) + "\n")

    def infer(self, frame):
        with inference_boundary():
            return self._infer(frame)

    def _infer(self, frame):
        owner = getattr(self.policy, "owner", None)
        if owner not in ("TAKEOVER", "DANCE") or getattr(self.policy, "retired", False):
            raise ContractError("Shared Sim inference requires active native owner")
        if owner == "DANCE" and self.session.stage == "WARMUP":
            self.session.begin_dance()
            self.session_events.append({"event": "begin_dance", "sim_time": float(self.data.time)})
        now = time.monotonic()
        request = self.session.build_request(self.adapter.snapshot(measured_at=now), now=now)
        if request.frame != frame:
            raise ContractError("Legacy scheduler and shared phase disagree")
        self.requests.append(
            {
                "tick": request.tick,
                "stage": request.stage,
                "frame": frame,
                "sim_time": float(self.data.time),
                "accepted": False,
            }
        )
        if self.inject_output == "exception":
            self.inject_output = None
            raise RuntimeError("injected_policy_exception")
        result = self.common.infer(request, executor=self.policy)
        self.inferences += 1
        if self.inject_output == "nan":
            self.inject_output = None
            # Exercise the same existing nonfinite safety branch.
            from dataclasses import replace

            result = replace(result, raw_action=np.full(29, np.nan, np.float32))
        self.interfaces.append(
            (
                self.data.time,
                frame,
                request.obs.copy(),
                request.goal.copy(),
                result.raw_action.copy(),
            )
        )
        accepted = self.session.accept(result, now=time.monotonic())
        self.requests[-1]["accepted"] = True
        self.requests[-1]["completion"] = "inline"
        return accepted.q_policy

    def check_assets(self):
        actual = super().check_assets()
        (self.out / "shared_session.json").write_text(
            json.dumps(
                {
                    "backend": "SIM_PHYSICS",
                    "phase": self.session.phase,
                    "stage": self.session.stage,
                    "warmup_count": self.session.warmup_count,
                    "clip_complete": self.session.clip_complete,
                    "retired_reason": self.session.retired_reason,
                    "events": self.session_events,
                    "requests": self.requests,
                },
                indent=2,
            )
            + "\n"
        )
        return actual
