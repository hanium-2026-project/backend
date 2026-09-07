"""RemoteDirectSession — 실제 backend 계약 기준 REMOTE_DIRECT 비동기 handshake + callback fan-out.

실제 API 반영:
- SET_MODE 는 동기 "ACCEPTED" 반환이 아니라 비동기 reliable command:
      seq = server.send_set_mode(car_id:int, "REMOTE_DIRECT")   # seq(int) 반환
  ACCEPTED/거절은 이후 terminal STATUS/COMMAND_RESULT 로 도착한다.
  ★ ACCEPTED 관찰 경로는 오직 하나: server 에 추가한 on_command_result(car_id, seq, result, msg).
    (production_patch/backend.patch 로 comm/server.py 에 추가. on_status fallback 은 쓰지 않는다.)
  negative 는 기존 on_command_rejected(car_id, result, msg) 로도 계속 통지된다(보존).
- server.register_comm_callbacks() 는 존재하지 않는다. 속성 callback 을 **fan-out**으로 감싼다
  (기존 pipeline/orchestrator callback 을 덮어쓰지 않고 함께 호출).
- 실제 callback arity: on_comm_fail(car_id, info) / on_comm_recovered(car_id) /
  on_resync(car_id, hello) / on_command_rejected(car_id, result, msg) /
  on_command_result(car_id, seq, result, msg).
- AUTO_HOST 활성 시 server.direct_control_enabled = True 를 보장.
  ★ 다중 차량: 이 플래그는 server-global 이다. 한 차량 fault 로 끄지 않는다(다른 차량 stream 유지).
- car_id 는 int(1,2). wire "CAR_01" 은 서버 내부에서만.

handshake 흐름:
  READY → begin_handshake() = send_set_mode(seq 저장) → (ACCEPTED 대기) →
  on_command_result(seq==set_mode_seq, ACCEPTED) → accepted=True → arm_auto.
  ACCEPTED 전에는 arm 하지 않는다(=non-zero 금지). REJECTED/INVALID_STATE/timeout → FAULTED+zero.
  seq 가 다른 ACCEPTED 는 무시한다.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any, Callable, Optional

from host_control.host_controller import HostController


class ModeHandshakeError(RuntimeError):
    pass


class NegotiationState(Enum):
    WAIT_CONNECTION = "WAIT_CONNECTION"
    WAIT_HELLO = "WAIT_HELLO"
    WAIT_RESET_ACK = "WAIT_RESET_ACK"
    WAIT_SET_MODE_ACK = "WAIT_SET_MODE_ACK"
    READY_REMOTE_DIRECT = "READY_REMOTE_DIRECT"
    FAULT = "FAULT"


class _SessionChanged(RuntimeError):
    """Internal control flow: continue negotiation on the replacement session."""


def _chain(existing: Optional[Callable], new: Callable) -> Callable:
    """기존 callback 을 보존하며 new 를 추가로 호출하는 fan-out wrapper."""
    if existing is None:
        return new

    def _fanout(*args, **kwargs):
        existing(*args, **kwargs)   # 기존(SW pipeline/orchestrator) 먼저
        new(*args, **kwargs)        # host session 추가
    return _fanout


class RemoteDirectSession:
    """Single owner for RESET/SET_MODE negotiation for one physical car.

    Callers only request the desired REMOTE_DIRECT state through
    :meth:`ensure_remote_direct`.  The lock covers the complete reliable
    transaction, so duplicate pipeline/recovery callbacks wait for the same
    result instead of creating another command.
    """

    def __init__(self, host: HostController | None, server: Any, car_id: int) -> None:
        assert isinstance(car_id, int), "production car_id 는 int(1,2) 여야 함"
        self.host = host
        self._server = server
        self._car_id = car_id
        self._set_mode_seq: Optional[int] = None
        self._accepted = threading.Event()
        self._rejected_reason: Optional[str] = None
        self._attached = False
        self._hs_lock = threading.Lock()   # begin_handshake seq 대입과 콜백 매칭 race 방지
        self._session_identity: tuple[str, str] | None = None
        self._transport_epoch = 0
        self._transaction_key: tuple[str, str, int] | None = None
        self._negotiation_lock = threading.Lock()
        self._negotiation_state = NegotiationState.WAIT_CONNECTION
        self._negotiation_identity: tuple[str, str] | None = None
        self._negotiation_key: tuple[str, str, int] | None = None
        self._negotiation_generation = 0
        self._negotiation_error: BaseException | None = None

    @property
    def car_id(self) -> int:
        return self._car_id

    @property
    def accepted(self) -> bool:
        return self._accepted.is_set()

    @property
    def negotiation_state(self) -> NegotiationState:
        return self._negotiation_state

    @property
    def negotiation_generation(self) -> int:
        return self._negotiation_generation

    @property
    def negotiation_identity(self) -> tuple[str, str] | None:
        return self._negotiation_identity

    def _current_identity(self) -> tuple[str, str] | None:
        getter = getattr(self._server, "session_identity", None)
        if getter is not None:
            return getter(self._car_id)
        # Compatibility for the older integration contract double.  Production
        # VehicleServer always provides session_identity().
        cars = getattr(self._server, "_cars", None)
        state = cars.get(self._car_id) if isinstance(cars, dict) else None
        session_id = getattr(state, "session_id", None)
        return (str(session_id), "LEGACY_SPEC") if session_id else None

    # -------------------------------------------------- callback fan-out (수정 4)
    def attach(self) -> None:
        """server 속성 callback 에 host session 훅을 fan-out 으로 추가(덮어쓰지 않음)."""
        if self._attached:
            return
        srv = self._server
        srv.on_command_result = _chain(getattr(srv, "on_command_result", None),
                                       self._on_command_result)
        srv.on_command_rejected = _chain(getattr(srv, "on_command_rejected", None),
                                         self._on_command_rejected)
        srv.on_comm_fail = _chain(getattr(srv, "on_comm_fail", None), self._on_comm_fail)
        srv.on_comm_recovered = _chain(getattr(srv, "on_comm_recovered", None),
                                       self._on_comm_recovered)
        srv.on_resync = _chain(getattr(srv, "on_resync", None), self._on_resync)
        self._attached = True

    # -------------------------------------------------- SET_MODE (수정 3: async)
    def begin_handshake(self) -> int:
        """SET_MODE REMOTE_DIRECT 전송, seq 저장. ACCEPTED 는 콜백으로 나중에 도착.

        seq 대입과 콜백 매칭 사이의 race 를 막기 위해 lock 으로 감싼다(실제 mock 이 매우 빠르게
        ACCEPTED STATUS 를 보내면 대입 전에 콜백이 올 수 있음).
        """
        with self._hs_lock:
            identity = self._current_identity()
            if identity is None:
                raise ModeHandshakeError(
                    f"car {self._car_id}: no active session for SET_MODE")
            key = (identity[0], identity[1], self._transport_epoch)
            # A logical SET_MODE transaction belongs to a session generation.
            # Re-entrant desired-state requests must share it, regardless of
            # whether its ACK is pending or has just arrived.
            if key == self._transaction_key and self._set_mode_seq is not None:
                return self._set_mode_seq
            self._accepted.clear()
            self._rejected_reason = None
            self._set_mode_seq = None
            self._session_identity = identity
            self._transaction_key = key
            seq = self._server.send_set_mode(self._car_id, "REMOTE_DIRECT")  # → int
            self._set_mode_seq = int(seq)
            return self._set_mode_seq

    def ensure_remote_direct(self, *, wait_s: float = 2.0) -> None:
        """Idempotently converge the current physical session to REMOTE_DIRECT.

        RESET is emitted only for firmware states that accept it.  A session
        replacement invalidates the old transaction and starts one transaction
        for the new identity.  Calls arriving while SET_MODE is pending block
        on this owner and never submit a second reliable command.
        """
        deadline = time.monotonic() + max(0.0, float(wait_s))
        with self._negotiation_lock:
            while time.monotonic() < deadline:
                identity = self._current_identity()
                if identity is None:
                    self._negotiation_state = NegotiationState.WAIT_CONNECTION
                    time.sleep(0.01)
                    continue
                with self._hs_lock:
                    epoch = self._transport_epoch
                key = (identity[0], identity[1], epoch)
                if key != self._negotiation_key:
                    with self._hs_lock:
                        if key != self._transaction_key:
                            self._session_identity = identity
                            self._transaction_key = None
                            self._set_mode_seq = None
                            self._accepted.clear()
                            self._rejected_reason = None
                    self._negotiation_identity = identity
                    self._negotiation_key = key
                    self._negotiation_generation += 1
                    self._negotiation_error = None
                if (self._negotiation_state is NegotiationState.READY_REMOTE_DIRECT
                        and self._accepted.is_set()
                        and key == self._negotiation_key):
                    return
                if (self._negotiation_state is NegotiationState.FAULT
                        and self._negotiation_error is not None
                        and key == self._negotiation_key):
                    raise self._negotiation_error
                self._negotiation_state = NegotiationState.WAIT_HELLO
                try:
                    status = self._wait_current_status(identity, epoch, deadline)
                    state = str(status.get("state", ""))
                    if state in ("EMERGENCY_STOP", "ERROR"):
                        self._reset_exact(identity, epoch, deadline)
                        self._wait_ready(identity, epoch, deadline)
                    elif state == "COMM_TIMEOUT":
                        raise ModeHandshakeError(
                            f"car {self._car_id}: COMM_TIMEOUT requires reconnect")
                    self._set_mode_exact(identity, epoch, deadline)
                except _SessionChanged:
                    continue
                except BaseException as exc:
                    self._negotiation_state = NegotiationState.FAULT
                    self._negotiation_error = exc
                    raise
                self._negotiation_state = NegotiationState.READY_REMOTE_DIRECT
                self._negotiation_error = None
                return
        error = ModeHandshakeError(
            f"car {self._car_id}: REMOTE_DIRECT negotiation timeout")
        self._negotiation_state = NegotiationState.FAULT
        self._negotiation_error = error
        raise error

    def _wait_current_status(self, identity: tuple[str, str], epoch: int,
                             deadline: float) -> dict[str, Any]:
        while time.monotonic() < deadline:
            if (self._current_identity() != identity
                    or self._transport_epoch != epoch):
                raise _SessionChanged
            status = self._server.last_status(self._car_id)
            if status.get("state"):
                return status
            time.sleep(0.01)
        raise ModeHandshakeError(f"car {self._car_id}: STATUS unavailable")

    def _wait_ready(self, identity: tuple[str, str], epoch: int,
                    deadline: float) -> None:
        while time.monotonic() < deadline:
            status = self._wait_current_status(identity, epoch, deadline)
            if str(status.get("state", "")) == "READY":
                return
            time.sleep(0.01)
        raise ModeHandshakeError(f"car {self._car_id}: RESET ACK without READY")

    def _wait_exact_result(self, identity: tuple[str, str], epoch: int, seq: int,
                           deadline: float, command: str) -> str:
        remaining = max(0.0, deadline - time.monotonic())
        result = self._server.wait_reliable_result(
            self._car_id, identity[0], seq, remaining)
        if (self._current_identity() != identity
                or self._transport_epoch != epoch):
            raise _SessionChanged
        if result is None:
            raise ModeHandshakeError(
                f"car {self._car_id}: {command} terminal ACK timeout")
        return str(result)

    def _reset_exact(self, identity: tuple[str, str], epoch: int,
                     deadline: float) -> None:
        self._negotiation_state = NegotiationState.WAIT_RESET_ACK
        seq = self._server.send_reset(self._car_id)
        result = self._wait_exact_result(identity, epoch, seq, deadline, "RESET")
        if result != "ACCEPTED":
            raise ModeHandshakeError(
                f"car {self._car_id}: RESET rejected ({result})")

    def _set_mode_exact(self, identity: tuple[str, str], epoch: int,
                        deadline: float) -> None:
        self._negotiation_state = NegotiationState.WAIT_SET_MODE_ACK
        seq = self.begin_handshake()
        result = self._wait_exact_result(identity, epoch, seq, deadline, "SET_MODE")
        with self._hs_lock:
            if identity != self._session_identity:
                raise _SessionChanged
            if result == "ACCEPTED":
                self._accepted.set()
            else:
                self._rejected_reason = result
        if result != "ACCEPTED" or self._rejected_reason is not None:
            raise ModeHandshakeError(
                f"car {self._car_id}: REMOTE_DIRECT rejected "
                f"({self._rejected_reason or result})")

    def wait_accepted(self, timeout_s: float = 1.0) -> bool:
        """ACCEPTED 도착까지 대기(테스트/동기 실행용). 실서비스는 콜백 기반으로도 가능."""
        ok = self._accepted.wait(timeout_s)
        if not ok:
            if self.host is not None:
                self.host.fault("SET_MODE_TIMEOUT")   # timeout → FAULTED + zero
        return ok

    def arm_auto(self, *, wait_s: float = 1.0) -> None:
        """ACCEPTED 확인 후에만 AUTO_HOST 무장. 미확인이면 FAULTED (non-zero 금지)."""
        if not self._attached:
            self.attach()
        self.ensure_remote_direct(wait_s=wait_s)
        if self._rejected_reason is not None:
            if self.host is not None:
                self.host.fault(f"SET_MODE_{self._rejected_reason}")
            raise ModeHandshakeError(f"REMOTE_DIRECT 거절: {self._rejected_reason}")
        # ★ direct stream gate (수정 5)
        self._enable_direct_stream()
        if self.host is None:
            raise ModeHandshakeError("AUTO_HOST controller is unavailable")
        self.host.arm_auto()

    def _enable_direct_stream(self, *, release_control: bool = True) -> None:
        try:
            self._server.direct_control_enabled = True
            if release_control:
                release = getattr(self._server, "release_control", None)
                if release is not None:
                    release(self._car_id)
        except Exception:
            pass

    # -------------------------------------------------- 콜백 핸들러
    def _on_command_result(self, car_id: int, seq: int, result: str, status: dict) -> None:
        with self._hs_lock:
            if car_id != self._car_id or seq != self._set_mode_seq:
                return
            if (self._session_identity is not None
                    and status.get("session_id") not in
                        (None, self._session_identity[0])):
                return
            if result == "ACCEPTED":
                self._accepted.set()
            else:
                self._rejected_reason = result
                if self.host is not None:
                    self.host.fault(f"SET_MODE_{result}")
                self._accepted.set()  # 대기 해제(거절로)

    def _on_command_rejected(self, car_id: int, result: str, status: dict) -> None:
        """SET_MODE 거절만 처리한다.

        backend 결선 시 수정(2026-08-10): 원본은 이 차량의 **모든** 명령 거절에
        대해 host 를 FAULTED 로 만들었다. RESET 같은 다른 명령이 거절돼도
        주행 권한이 잠겨 복구가 불가능해진다. 우리 서버는 negative 결과에 대해
        on_command_result 와 on_command_rejected 를 **둘 다** 부르므로 이중
        fault 도 났다. seq 가 이번 SET_MODE 의 것일 때만 반응한다.
        """
        if car_id != self._car_id or self._set_mode_seq is None:
            return
        rejected = status.get("rejected_seq", status.get("ack_seq"))
        if rejected is not None and rejected != self._set_mode_seq:
            return
        self._rejected_reason = result
        if self.host is not None:
            self.host.fault(f"SET_MODE_{result}")
        self._accepted.set()

    def _on_comm_fail(self, car_id: int, _status: dict) -> None:
        if car_id != self._car_id:
            return
        if self.host is not None:
            self.host.fault("COMM_TIMEOUT")
        with self._hs_lock:
            self._transport_epoch += 1
            self._accepted.clear()
            self._transaction_key = None
            self._set_mode_seq = None
            self._rejected_reason = None
        # ★ 다중 차량 안전: server.direct_control_enabled 는 server-global 이므로 끄지 않는다.
        #   이 차량만 stop_control(car_id) 로 zero → 다른 AUTO_HOST 차량 stream 유지.
        stop = getattr(self._server, "stop_control", None)
        if stop is not None:
            stop(self._car_id)

    def _on_comm_recovered(self, car_id: int) -> None:
        # ★ 실제 backend: on_comm_recovered(car_id) — 인자 1개.
        # 복구돼도 자동 복귀 금지. FAULTED 유지.
        return None

    def _on_resync(self, car_id: int, _hello: dict) -> None:
        if car_id != self._car_id:
            return
        # 재접속 → 이전 host mission/control state 폐기, zero, mode 재협상 필요
        if self.host is not None:
            self.host.fault("RESYNC")
        stop = getattr(self._server, "stop_control", None)
        if stop is not None:
            stop(self._car_id)
        identity = self._current_identity()
        with self._hs_lock:
            if identity is not None and identity == self._session_identity:
                return
            self._session_identity = identity
            self._transaction_key = None
            self._accepted.clear()
            self._set_mode_seq = None
            self._rejected_reason = None

    # -------------------------------------------------- explicit re-arm
    def re_arm_auto(self, *, wait_s: float = 1.0) -> None:
        """stale/comm/resync fault 후 사용자 명시적 재출발. mode 재협상 후 복귀."""
        # FAULTED → clear 후 재 handshake
        if self.host is None:
            raise ModeHandshakeError("AUTO_HOST controller is unavailable")
        self.host.authority.clear_fault() if self.host.authority.is_faulted else None
        self.ensure_remote_direct(wait_s=wait_s)
        if self._rejected_reason is not None:
            self.host.fault(f"SET_MODE_{self._rejected_reason}")
            raise ModeHandshakeError(f"re-arm 거절: {self._rejected_reason}")
        self._enable_direct_stream()
        self.host.arm_auto()

    # 테스트 편의: 콜백이 없을 때 결과를 직접 주입
    def notify_command_result(self, seq: int, result: str) -> None:
        self._on_command_result(self._car_id, seq, result, {})
