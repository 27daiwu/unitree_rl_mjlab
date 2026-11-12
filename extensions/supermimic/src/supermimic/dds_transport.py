"""Minimal ctypes binding for the local-only SuperMimic DDS transport."""

from __future__ import annotations

import ctypes
import fcntl
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supermimic.policy.runtime import ContractError

MOTORS = 29
FloatVector = ctypes.c_float * MOTORS


class DdsTimeoutError(RuntimeError):
    pass


class _State(ctypes.Structure):
    _fields_ = [
        ("tick", ctypes.c_uint32),
        ("mode_machine", ctypes.c_uint8),
        ("q", FloatVector),
        ("dq", FloatVector),
        ("tau_est", FloatVector),
        ("root_quat_wxyz", ctypes.c_float * 4),
        ("omega_b", ctypes.c_float * 3),
        ("root_pos_w", ctypes.c_float * 3),
        ("root_lin_vel_w", ctypes.c_float * 3),
        ("received_monotonic", ctypes.c_double),
        ("sequence", ctypes.c_uint64),
        ("root_valid", ctypes.c_int32),
    ]


class _Command(ctypes.Structure):
    _fields_ = [
        ("q", FloatVector),
        ("dq", FloatVector),
        ("kp", FloatVector),
        ("kd", FloatVector),
        ("tau", FloatVector),
        ("motor_mode", ctypes.c_uint8),
        ("mode_machine", ctypes.c_uint8),
    ]


@dataclass(frozen=True)
class DdsFrame:
    tick: int
    mode_machine: int
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    root_quat_wxyz: np.ndarray
    omega_b: np.ndarray
    root_pos_w: np.ndarray
    root_lin_vel_w: np.ndarray
    received_monotonic: float
    sequence: int
    root_valid: bool


def _readonly(values, shape, name):
    result = np.array(values, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ContractError(f"Invalid DDS {name}")
    result.flags.writeable = False
    return result


class DdsTransport:
    """One local writer and one state reader; no ChannelFactory call at import."""

    def __init__(
        self,
        *,
        interface="lo",
        domain_id=0,
        library_path=None,
        writer_lock_path="/tmp/supermimic_dds_writer.lock",
        read_only=False,
    ):
        self.read_only = read_only
        self._lock_stream = None
        if not read_only:
            self._lock_stream = Path(writer_lock_path).open("a+")  # noqa: SIM115 -- writer lifetime.
            try:
                fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                self._lock_stream.close()
                raise ContractError(
                    "Another SuperMimic DDS writer holds the process lock"
                ) from error
        self.handle = None
        self._closed = False
        try:
            library = (
                Path(library_path)
                if library_path
                else Path(__file__).with_name("dds") / "build" / "libsupermimic_dds.so"
            )
            self._lib = ctypes.CDLL(str(library))
            self._lib.sm_dds_last_error.restype = ctypes.c_char_p
            self._lib.sm_dds_state_size.restype = ctypes.c_uint32
            self._lib.sm_dds_command_size.restype = ctypes.c_uint32
            self._lib.sm_dds_create.restype = ctypes.c_void_p
            self._lib.sm_dds_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            self._lib.sm_dds_open.restype = ctypes.c_int
            self._lib.sm_dds_open_read_only.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            self._lib.sm_dds_open_read_only.restype = ctypes.c_int
            self._lib.sm_dds_read_state.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_State),
                ctypes.c_uint32,
            ]
            self._lib.sm_dds_read_state.restype = ctypes.c_int
            self._lib.sm_dds_write_command.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Command)]
            self._lib.sm_dds_write_command.restype = ctypes.c_int
            self._lib.sm_dds_close.argtypes = [ctypes.c_void_p]
            self._lib.sm_dds_close.restype = ctypes.c_int
            self._lib.sm_dds_destroy.argtypes = [ctypes.c_void_p]
            self._lib.sm_dds_destroy.restype = None
            if self._lib.sm_dds_state_size() != ctypes.sizeof(_State):
                raise ContractError("DDS state ABI size mismatch")
            if self._lib.sm_dds_command_size() != ctypes.sizeof(_Command):
                raise ContractError("DDS command ABI size mismatch")
            self.handle = self._lib.sm_dds_create()
            if not self.handle:
                raise ContractError("Unable to allocate DDS handle")
            open_function = self._lib.sm_dds_open_read_only if read_only else self._lib.sm_dds_open
            result = open_function(self.handle, interface.encode(), int(domain_id))
            if result != 0:
                raise ContractError(f"DDS open failed ({result}): {self.last_error}")
        except Exception:
            self.close()
            raise

    @property
    def last_error(self):
        value = self._lib.sm_dds_last_error()
        return value.decode(errors="replace") if value else ""

    def read_state(self, timeout_s=0.02):
        if self._closed:
            raise ContractError("DDS transport is closed")
        raw = _State()
        result = self._lib.sm_dds_read_state(self.handle, raw, max(0, round(timeout_s * 1000)))
        if result == 0:
            return None
        if result < 0:
            raise ContractError(f"DDS read failed ({result}): {self.last_error}")
        return DdsFrame(
            raw.tick,
            raw.mode_machine,
            _readonly(raw.q, (MOTORS,), "q"),
            _readonly(raw.dq, (MOTORS,), "dq"),
            _readonly(raw.tau_est, (MOTORS,), "tau_est"),
            _readonly(raw.root_quat_wxyz, (4,), "root quaternion"),
            _readonly(raw.omega_b, (3,), "omega"),
            _readonly(raw.root_pos_w, (3,), "root position"),
            _readonly(raw.root_lin_vel_w, (3,), "root velocity"),
            float(raw.received_monotonic),
            int(raw.sequence),
            bool(raw.root_valid),
        )

    def read_until(self, target_tick, *, timeout_s=0.10):
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DdsTimeoutError(f"Timed out waiting for simulator tick {target_tick}")
            frame = self.read_state(min(0.02, remaining))
            if frame is not None and frame.root_valid and frame.tick >= target_tick:
                return frame

    def write_command(self, *, q, kp, kd, dq=None, tau=None, mode_machine, motor_mode=1):
        if self._closed:
            raise ContractError("DDS transport is closed")
        if self.read_only:
            raise ContractError("Read-only DDS transport cannot publish commands")
        q = _readonly(q, (MOTORS,), "command q")
        kp = _readonly(kp, (MOTORS,), "command kp")
        kd = _readonly(kd, (MOTORS,), "command kd")
        dq = _readonly(np.zeros(MOTORS) if dq is None else dq, (MOTORS,), "command dq")
        tau = _readonly(np.zeros(MOTORS) if tau is None else tau, (MOTORS,), "command tau_ff")
        command = _Command(
            FloatVector(*q),
            FloatVector(*dq),
            FloatVector(*kp),
            FloatVector(*kd),
            FloatVector(*tau),
            int(motor_mode),
            int(mode_machine),
        )
        result = self._lib.sm_dds_write_command(self.handle, command)
        if result != 1:
            raise ContractError(f"DDS command write failed ({result}): {self.last_error}")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.handle:
            self._lib.sm_dds_close(self.handle)
            self._lib.sm_dds_destroy(self.handle)
            self.handle = None
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()


def encode_offline_command(
    *, q, kp, kd, dq=None, tau=None, mode_machine, motor_mode=1, library_path=None
):
    """Encode a full LowCmd without initializing DDS or a writer lock."""

    q = _readonly(q, (MOTORS,), "command q")
    kp = _readonly(kp, (MOTORS,), "command kp")
    kd = _readonly(kd, (MOTORS,), "command kd")
    dq = _readonly(np.zeros(MOTORS) if dq is None else dq, (MOTORS,), "command dq")
    tau = _readonly(np.zeros(MOTORS) if tau is None else tau, (MOTORS,), "command tau_ff")
    command = _Command(
        FloatVector(*q),
        FloatVector(*dq),
        FloatVector(*kp),
        FloatVector(*kd),
        FloatVector(*tau),
        int(motor_mode),
        int(mode_machine),
    )
    library = (
        Path(library_path)
        if library_path
        else Path(__file__).with_name("dds") / "build" / "libsupermimic_dds.so"
    )
    lib = ctypes.CDLL(str(library))
    lib.sm_dds_last_error.restype = ctypes.c_char_p
    lib.sm_dds_lowcmd_size.restype = ctypes.c_uint32
    lib.sm_dds_encode_command.argtypes = [
        ctypes.POINTER(_Command),
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.sm_dds_encode_command.restype = ctypes.c_int
    size = lib.sm_dds_lowcmd_size()
    output = ctypes.create_string_buffer(size)
    result = lib.sm_dds_encode_command(command, output, size)
    if result < 0:
        error = lib.sm_dds_last_error()
        raise ContractError(f"Offline command encoding failed ({result}): {error}")
    if result != size:
        raise ContractError("Offline LowCmd size mismatch")
    return output.raw
