"""Deterministic UserCtrl ownership protocol mock with epoch and writer isolation.

No class in this module talks to Unitree, DDS, a network, or a robot.  The mock
vendor is a scripted in-memory message source.  ``mock_a/mock_b=PASS`` never
promotes ``real_a/real_b`` beyond UNVERIFIED and never produces physical C PASS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from supermimic.policy.runtime import ContractError
from supermimic.real_adapter import (
    NOT_RUN,
    REAL_MOCK,
    UNVERIFIED,
    OfflineCommandWriter,
    _integer,
    _require_mock,
    _time,
)


class Operation(str, Enum):
    SWITCH_TO_USER_CTRL = "SwitchToUserCtrl"
    SWITCH_TO_INTERNAL_CTRL = "SwitchToInternalCtrl"
    GET_FSM_ID = "GetFsmId"


class InternalTarget(str, Enum):
    PASSIVE = "PASSIVE"
    WALKRUN = "WALKRUN"


class Owner(str, Enum):
    USER = "USER"
    INTERNAL = "INTERNAL"
    UNKNOWN = "UNKNOWN"


class OwnerState(str, Enum):
    IDLE = "IDLE"
    TAKEOVER_PENDING = "TAKEOVER_PENDING"
    USER_CTRL = "USER_CTRL"
    RETURN_PENDING = "RETURN_PENDING"
    FACTORY_INTERNAL = "FACTORY_INTERNAL"
    OWNER_UNCONFIRMED = "OWNER_UNCONFIRMED"


@dataclass(frozen=True)
class MockOwnershipSemantics:
    """Explicit mock tokens, not a statement about firmware ownership semantics."""

    user_ctrl_fsm_id: int
    internal_fsm_ids: Mapping[InternalTarget, int]
    evidence_domain: str = REAL_MOCK
    status: str = UNVERIFIED

    def validated(self):
        _require_mock(self.evidence_domain)
        if self.status != UNVERIFIED:
            raise ContractError("P3 cannot promote FSM semantics beyond UNVERIFIED")
        user = _integer(self.user_ctrl_fsm_id, "mock UserCtrl FSM ID", 0)
        targets = {
            InternalTarget.PASSIVE: _integer(
                self.internal_fsm_ids[InternalTarget.PASSIVE], "mock PASSIVE FSM ID", 0
            ),
            InternalTarget.WALKRUN: _integer(
                self.internal_fsm_ids[InternalTarget.WALKRUN], "mock WALKRUN FSM ID", 0
            ),
        }
        if len(set(self.internal_fsm_ids)) != 2 or len({user, *targets.values()}) != 3:
            raise ContractError("Mock UserCtrl/PASSIVE/WALKRUN readback IDs must be distinct")
        return user, targets


@dataclass(frozen=True)
class OwnershipPreconditions:
    explicit_start_whitelist: bool
    factory_ready: bool
    state_estimate_ready: bool
    known_source_target: bool
    command_path_ready: bool


@dataclass(frozen=True)
class OwnershipRequest:
    request_id: int
    run_id: str
    epoch: int
    operation: Operation
    issued_at: float
    deadline: float
    target: InternalTarget | None = None
    selector: int | None = None


@dataclass(frozen=True)
class MockBehavior:
    delay_s: float
    ret: int
    apply: bool = True
    duplicates: int = 1


@dataclass(frozen=True)
class MockRpcResponse:
    request: OwnershipRequest
    completed_at: float
    ret: int
    fsm_id: int | None = None
    owner: Owner = Owner.UNKNOWN


@dataclass(frozen=True)
class FsmReadback:
    request_id: int
    run_id: str
    epoch: int
    completed_at: float
    ret: int
    fsm_id: int | None
    owner: Owner
    evidence_domain: str = REAL_MOCK
    status: str = UNVERIFIED


@dataclass(frozen=True)
class OwnershipEvidence:
    direction: str
    epoch: int
    mock_a: str
    mock_b: str
    reasons: tuple[str, ...] = ()
    evidence_domain: str = REAL_MOCK
    real_a: str = UNVERIFIED
    real_b: str = UNVERIFIED
    physical_c: str = NOT_RUN


class MockVendorTransport:
    """Scripted responses; vendor state may change even when a reply is late."""

    def __init__(self, semantics: MockOwnershipSemantics):
        self.semantics = semantics
        self.user_fsm_id, self.internal_fsm_ids = semantics.validated()
        self.vendor_fsm_id = self.internal_fsm_ids[InternalTarget.WALKRUN]
        self.vendor_owner = Owner.INTERNAL
        self._scripts = {operation: [] for operation in Operation}
        self._messages = []
        self.requests = []

    def script(self, operation: Operation, behavior: MockBehavior):
        _time(behavior.delay_s, "mock RPC delay")
        if behavior.delay_s < 0:
            raise ContractError("Mock RPC delay cannot be negative")
        _integer(behavior.ret, "mock RPC return code")
        _integer(behavior.duplicates, "mock response duplicates", 1)
        if type(behavior.apply) is not bool:
            raise ContractError("Mock apply flag must be Boolean")
        self._scripts[operation].append(behavior)

    def submit(self, request: OwnershipRequest):
        if request.operation not in Operation:
            raise ContractError("Unknown mock operation")
        if not self._scripts[request.operation]:
            raise ContractError(f"No scripted mock response for {request.operation.value}")
        behavior = self._scripts[request.operation].pop(0)
        self.requests.append(request)
        self._messages.extend(
            [(request.issued_at + behavior.delay_s, request, behavior)] * behavior.duplicates
        )

    def poll(self, now):
        now = _time(now, "mock transport poll time")
        due = [item for item in self._messages if item[0] <= now]
        self._messages = [item for item in self._messages if item[0] > now]
        responses = []
        for completed, request, behavior in sorted(due, key=lambda item: item[0]):
            if behavior.apply and behavior.ret == 0:
                if request.operation is Operation.SWITCH_TO_USER_CTRL:
                    self.vendor_owner, self.vendor_fsm_id = Owner.USER, self.user_fsm_id
                elif request.operation is Operation.SWITCH_TO_INTERNAL_CTRL:
                    self.vendor_owner = Owner.INTERNAL
                    self.vendor_fsm_id = self.internal_fsm_ids[request.target]
            if request.operation is Operation.GET_FSM_ID and behavior.ret == 0:
                fsm_id, owner = self.vendor_fsm_id, self.vendor_owner
            else:
                fsm_id, owner = None, Owner.UNKNOWN
            responses.append(MockRpcResponse(request, completed, behavior.ret, fsm_id, owner))
        return responses


class UserCtrlOwnershipMock:
    """Local supervisor for scripted RPC/FSM messages and one offline writer."""

    def __init__(
        self,
        transport: MockVendorTransport,
        writer: OfflineCommandWriter,
        *,
        session=None,
        evidence_domain: str = REAL_MOCK,
    ):
        _require_mock(evidence_domain)
        self.transport = transport
        self.writer = writer
        self.session = session
        self.evidence_domain = evidence_domain
        self.state = OwnerState.IDLE
        self.run_id = None
        self.epoch = -1
        self._next_request_id = 0
        self._pending = None
        self._active_switch = None
        self._return_target = None
        self._a = {}
        self._a_completed = {}
        self._b = {}
        self._reasons = {}
        self.readbacks = {}
        self.last_valid_readback = None
        self.events = []
        self.late_or_duplicate = []

    def _event(self, name, **values):
        item = {
            "event": name,
            "evidence_domain": REAL_MOCK,
            "run_id": self.run_id,
            "epoch": self.epoch,
            "physical_recovery": NOT_RUN,
            **values,
        }
        self.events.append(item)
        return item

    def _new_request(self, operation, now, ack_timeout_s, target=None):
        if self._pending is not None:
            raise ContractError("An RPC is already pending; duplicate request rejected")
        now = _time(now, "RPC issued_at")
        timeout = _time(ack_timeout_s, "explicit mock RPC timeout")
        if timeout <= 0:
            raise ContractError("RPC timeout must be positive and explicit")
        self._next_request_id += 1
        selector = None
        if operation is Operation.SWITCH_TO_INTERNAL_CTRL:
            if target not in (InternalTarget.PASSIVE, InternalTarget.WALKRUN):
                raise ContractError("Return target must be explicit PASSIVE or WALKRUN, never LAST")
            selector = 1 if target is InternalTarget.PASSIVE else 2
        request = OwnershipRequest(
            self._next_request_id,
            self.run_id,
            self.epoch,
            operation,
            now,
            now + timeout,
            target,
            selector,
        )
        self.transport.submit(request)
        self._pending = request
        self._event("rpc_submitted", request_id=request.request_id, operation=operation.value)
        return request

    def _retire_local(self, reason):
        if self.epoch >= 0:
            self.writer.revoke(self.epoch, reason)
            if self.session is not None and self.session.stage in ("WARMUP", "DANCE"):
                self.session.retire(reason)
        self._event("local_epoch_revoked", reason=reason, writer_closed=self.writer.closed)

    def request_takeover(
        self,
        run_id,
        epoch,
        preconditions: OwnershipPreconditions,
        *,
        now,
        ack_timeout_s,
    ):
        epoch = _integer(epoch, "takeover epoch", 0)
        if (
            not run_id
            or epoch <= self.epoch
            or self.state
            not in (
                OwnerState.IDLE,
                OwnerState.FACTORY_INTERNAL,
                OwnerState.OWNER_UNCONFIRMED,
            )
        ):
            raise ContractError("Takeover requires a new epoch and inactive local owner")
        if self._pending is not None:
            raise ContractError("Cannot take over while another RPC is pending")
        self._retire_local("takeover_precheck")
        self.run_id, self.epoch = run_id, epoch
        checks = (
            preconditions.explicit_start_whitelist,
            preconditions.factory_ready,
            preconditions.state_estimate_ready,
            preconditions.known_source_target,
            preconditions.command_path_ready,
        )
        if any(type(value) is not bool for value in checks):
            self.state = OwnerState.OWNER_UNCONFIRMED
            self._set_a("takeover", "FAIL", "non_boolean_start_precondition")
            self._retire_local("non_boolean_start_precondition")
            raise ContractError("Every takeover precondition must be explicitly Boolean")
        ready = all(checks)
        if not ready:
            self.state = OwnerState.OWNER_UNCONFIRMED
            self._set_a("takeover", "FAIL", "start_preconditions_not_satisfied")
            self._retire_local("takeover_preconditions_not_satisfied")
            raise ContractError("Explicit takeover preconditions are not satisfied")
        self.state = OwnerState.TAKEOVER_PENDING
        self._set_a("takeover", "NOT_RUN")
        self._set_b("takeover", UNVERIFIED)
        self._active_switch = self._new_request(Operation.SWITCH_TO_USER_CTRL, now, ack_timeout_s)
        return self._active_switch

    def request_fsm_readback(self, *, now, ack_timeout_s):
        if self.epoch < 0:
            raise ContractError("Cannot read FSM without an active request epoch")
        return self._new_request(Operation.GET_FSM_ID, now, ack_timeout_s)

    def request_return(self, target: InternalTarget, *, now, ack_timeout_s):
        if self.epoch < 0 or self.state in (OwnerState.IDLE, OwnerState.FACTORY_INTERNAL):
            raise ContractError("No active UserCtrl epoch to return")
        if self._pending is not None:
            raise ContractError("Cannot request return while another RPC is pending")
        # Local dance authority and Policy phase retire before the RPC.  This is
        # a local mock ordering choice, not a claim about required vendor ordering.
        self._retire_local("return_requested")
        self.state = OwnerState.RETURN_PENDING
        self._return_target = target
        self._set_a("return", "NOT_RUN")
        self._set_b("return", UNVERIFIED)
        self._active_switch = self._new_request(
            Operation.SWITCH_TO_INTERNAL_CTRL, now, ack_timeout_s, target
        )
        return self._active_switch

    def _set_a(self, direction, status, reason=None):
        key = (direction, self.epoch)
        self._a[key] = status
        self._reasons[(direction, "a", self.epoch)] = reason

    def _set_b(self, direction, status, reason=None):
        key = (direction, self.epoch)
        self._b[key] = status
        self._reasons[(direction, "b", self.epoch)] = reason

    def _timeout(self, request):
        operation = request.operation.value
        self._event("rpc_timeout", request_id=request.request_id, operation=operation)
        if request.operation is not Operation.GET_FSM_ID:
            direction = (
                "takeover" if request.operation is Operation.SWITCH_TO_USER_CTRL else "return"
            )
            self._set_a(direction, "FAIL", "rpc_timeout")
            self._set_b(direction, UNVERIFIED, "owner_unconfirmed")
            self.state = OwnerState.OWNER_UNCONFIRMED
            self._active_switch = None
            self._retire_local("rpc_timeout")
        else:
            self.last_valid_readback = None
        self._pending = None

    def poll(self, *, now):
        """Process due messages without sleeping; late/foreign replies are inert."""

        now = _time(now, "ownership poll time")
        if self._pending is not None and now >= self._pending.deadline:
            self._timeout(self._pending)
        responses = self.transport.poll(now)
        completed = []
        for response in responses:
            request = response.request
            if self._pending is None or request != self._pending:
                item = self._event(
                    "late_or_duplicate_rpc_ignored",
                    request_id=request.request_id,
                    operation=request.operation.value,
                )
                self.late_or_duplicate.append(item)
                continue
            self._pending = None
            if response.completed_at >= request.deadline or now >= request.deadline:
                self._timeout(request)
                continue
            completed.append(response)
            if request.operation is Operation.GET_FSM_ID:
                readback = FsmReadback(
                    request.request_id,
                    request.run_id,
                    request.epoch,
                    response.completed_at,
                    response.ret,
                    response.fsm_id,
                    response.owner,
                )
                self.readbacks[request.request_id] = readback
                # A failed poll invalidates the cache; it cannot reuse old FSM data.
                self.last_valid_readback = readback if response.ret == 0 else None
                self._event(
                    "fsm_readback",
                    request_id=request.request_id,
                    ret=response.ret,
                    fsm_id=response.fsm_id,
                )
                continue
            direction = (
                "takeover" if request.operation is Operation.SWITCH_TO_USER_CTRL else "return"
            )
            if response.ret == 0:
                self._set_a(direction, "PASS", "mock_rpc_success_only")
                self._a_completed[(direction, self.epoch)] = response.completed_at
                self._event(
                    "mock_A_pass",
                    request_id=request.request_id,
                    direction=direction,
                    real_A=UNVERIFIED,
                )
                # Deliberately no writer authorization or B PASS here.
            else:
                self._set_a(direction, "FAIL", f"rpc_rejected:{response.ret}")
                self._set_b(direction, UNVERIFIED, "rpc_rejected")
                self.state = OwnerState.OWNER_UNCONFIRMED
                self._active_switch = None
                self._retire_local("rpc_rejected")
        return completed

    def _readback_for_confirmation(self, switch_request, readback_request_id, now, max_age_s):
        if self._active_switch != switch_request:
            raise ContractError("Switch is no longer active; late confirmation rejected")
        readback = self.readbacks.get(readback_request_id)
        if readback is None:
            raise ContractError("Missing FSM readback")
        now = _time(now, "owner confirmation time")
        max_age = _time(max_age_s, "explicit mock readback max age")
        if max_age <= 0:
            raise ContractError("Readback max age must be positive")
        if (
            readback.run_id != switch_request.run_id
            or readback.epoch != switch_request.epoch
            or readback.completed_at < switch_request.issued_at
        ):
            raise ContractError("Foreign/stale-epoch FSM readback")
        if readback.ret != 0 or readback.fsm_id is None:
            raise ContractError("Failed FSM readback cannot reuse a cached value")
        if not 0 <= now - readback.completed_at <= max_age:
            raise ContractError("Expired or future-dated FSM readback")
        return readback

    def confirm_takeover(self, switch_request, readback_request_id, *, now, max_age_s):
        readback = self._readback_for_confirmation(
            switch_request, readback_request_id, now, max_age_s
        )
        key = ("takeover", self.epoch)
        if self.state is not OwnerState.TAKEOVER_PENDING or self._a.get(key) != "PASS":
            raise ContractError("Takeover B requires active epoch and Mock A")
        if readback.completed_at < self._a_completed[key]:
            raise ContractError("Takeover FSM readback predates Mock A")
        expected = self.transport.user_fsm_id
        if readback.owner is not Owner.USER or readback.fsm_id != expected:
            self._set_b("takeover", "FAIL", "explicit_user_fsm_mismatch")
            self.state = OwnerState.OWNER_UNCONFIRMED
            self._active_switch = None
            self._retire_local("takeover_fsm_mismatch")
            return self.evidence("takeover")
        self.writer.authorize(self.run_id, self.epoch)
        self.state = OwnerState.USER_CTRL
        self._active_switch = None
        self._set_b("takeover", "PASS", "fresh_mock_fsm_and_single_writer_lease")
        self._event(
            "mock_B_takeover_pass",
            fsm_id=readback.fsm_id,
            real_B=UNVERIFIED,
            physical_C=NOT_RUN,
        )
        return self.evidence("takeover")

    def confirm_return(self, switch_request, readback_request_id, *, now, max_age_s):
        readback = self._readback_for_confirmation(
            switch_request, readback_request_id, now, max_age_s
        )
        key = ("return", self.epoch)
        if self.state is not OwnerState.RETURN_PENDING or self._a.get(key) != "PASS":
            raise ContractError("Return B requires active return and Mock A")
        if readback.completed_at < self._a_completed[key]:
            raise ContractError("Return FSM readback predates Mock A")
        target = self._return_target
        expected = self.transport.internal_fsm_ids[target]
        if readback.owner is not Owner.INTERNAL or readback.fsm_id != expected:
            self._set_b("return", "FAIL", "explicit_target_fsm_mismatch_not_id_not_1000")
            self.state = OwnerState.OWNER_UNCONFIRMED
            self._active_switch = None
            self._retire_local("return_fsm_mismatch")
            return self.evidence("return")
        if not self.writer.closed or self.writer.inflight_offline != 0:
            self._set_b("return", UNVERIFIED, "writer_or_inflight_offline_records_not_closed")
            return self.evidence("return")
        self.state = OwnerState.FACTORY_INTERNAL
        self._active_switch = None
        self._set_b("return", "PASS", "fresh_explicit_mock_target_and_writer_gate_closed")
        self._event(
            "mock_B_return_pass",
            target=target.value,
            fsm_id=readback.fsm_id,
            real_B=UNVERIFIED,
            physical_C=NOT_RUN,
        )
        return self.evidence("return")

    def revoke(self, reason):
        """Revoke local authority without waiting for worker/RPC cleanup."""

        self._pending = None
        self._active_switch = None
        self.state = OwnerState.OWNER_UNCONFIRMED
        self._retire_local(reason)
        if self.epoch >= 0:
            self._set_b("takeover", UNVERIFIED, "locally_revoked")
            self._set_b("return", UNVERIFIED, "locally_revoked")

    def evidence(self, direction):
        if direction not in ("takeover", "return"):
            raise ContractError("Direction must be takeover or return")
        epoch = self.epoch
        reasons = tuple(
            value
            for key, value in self._reasons.items()
            if key[0] == direction and key[2] == epoch and value is not None
        )
        return OwnershipEvidence(
            direction,
            epoch,
            self._a.get((direction, epoch), NOT_RUN),
            self._b.get((direction, epoch), UNVERIFIED),
            reasons,
        )
