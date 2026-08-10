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
from typing import Any, Callable, Optional

from host_control.host_controller import HostController


class ModeHandshakeError(RuntimeError):
    pass


def _chain(existing: Optional[Callable], new: Callable) -> Callable:
    """기존 callback 을 보존하며 new 를 추가로 호출하는 fan-out wrapper."""
    if existing is None:
        return new

    def _fanout(*args, **kwargs):
        existing(*args, **kwargs)   # 기존(SW pipeline/orchestrator) 먼저
        new(*args, **kwargs)        # host session 추가
    return _fanout


class RemoteDirectSession:
    def __init__(self, host: HostController, server: Any, car_id: int) -> None:
        assert isinstance(car_id, int), "production car_id 는 int(1,2) 여야 함"
        self.host = host
        self._server = server
        self._car_id = car_id
        self._set_mode_seq: Optional[int] = None
        self._accepted = threading.Event()
        self._rejected_reason: Optional[str] = None
        self._attached = False
        self._hs_lock = threading.Lock()   # begin_handshake seq 대입과 콜백 매칭 race 방지

    @property
    def car_id(self) -> int:
        return self._car_id

    @property
    def accepted(self) -> bool:
        return self._accepted.is_set()

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
            self._accepted.clear()
            self._rejected_reason = None
            self._set_mode_seq = None
            seq = self._server.send_set_mode(self._car_id, "REMOTE_DIRECT")  # → int
            self._set_mode_seq = int(seq)
            return self._set_mode_seq

    def wait_accepted(self, timeout_s: float = 1.0) -> bool:
        """ACCEPTED 도착까지 대기(테스트/동기 실행용). 실서비스는 콜백 기반으로도 가능."""
        ok = self._accepted.wait(timeout_s)
        if not ok:
            self.host.fault("SET_MODE_TIMEOUT")   # timeout → FAULTED + zero
        return ok

    def arm_auto(self, *, wait_s: float = 1.0) -> None:
        """ACCEPTED 확인 후에만 AUTO_HOST 무장. 미확인이면 FAULTED (non-zero 금지)."""
        if not self._attached:
            self.attach()
        self.begin_handshake()
        if not self.wait_accepted(wait_s):
            raise ModeHandshakeError("REMOTE_DIRECT ACCEPTED 미도착 → FAULTED")
        if self._rejected_reason is not None:
            self.host.fault(f"SET_MODE_{self._rejected_reason}")
            raise ModeHandshakeError(f"REMOTE_DIRECT 거절: {self._rejected_reason}")
        # ★ direct stream gate (수정 5)
        self._enable_direct_stream()
        self.host.arm_auto()

    def _enable_direct_stream(self) -> None:
        try:
            self._server.direct_control_enabled = True
        except Exception:
            pass

    # -------------------------------------------------- 콜백 핸들러
    def _on_command_result(self, car_id: int, seq: int, result: str, _status: dict) -> None:
        with self._hs_lock:
            if car_id != self._car_id or seq != self._set_mode_seq:
                return
            if result == "ACCEPTED":
                self._accepted.set()
            else:
                self._rejected_reason = result
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
        self.host.fault(f"SET_MODE_{result}")
        self._accepted.set()

    def _on_comm_fail(self, car_id: int, _status: dict) -> None:
        if car_id != self._car_id:
            return
        self.host.fault("COMM_TIMEOUT")
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
        self.host.fault("RESYNC")
        stop = getattr(self._server, "stop_control", None)
        if stop is not None:
            stop(self._car_id)
        self._accepted.clear()
        self._set_mode_seq = None

    # -------------------------------------------------- explicit re-arm
    def re_arm_auto(self, *, wait_s: float = 1.0) -> None:
        """stale/comm/resync fault 후 사용자 명시적 재출발. mode 재협상 후 복귀."""
        # FAULTED → clear 후 재 handshake
        self.host.authority.clear_fault() if self.host.authority.is_faulted else None
        self.begin_handshake()
        if not self.wait_accepted(wait_s):
            raise ModeHandshakeError("re-arm 중 ACCEPTED 미도착")
        if self._rejected_reason is not None:
            self.host.fault(f"SET_MODE_{self._rejected_reason}")
            raise ModeHandshakeError(f"re-arm 거절: {self._rejected_reason}")
        self._enable_direct_stream()
        self.host.arm_auto()

    # 테스트 편의: 콜백이 없을 때 결과를 직접 주입
    def notify_command_result(self, seq: int, result: str) -> None:
        self._on_command_result(self._car_id, seq, result, {})
