"""SpecVehicleServer — 실제 backend VehicleServer 계약을 정확히 재현한 test double.

⚠ 실제 comm/server.py 가 아니다. latest_backend_contract 에서 확인된 최신 API 를 재현:

- send_set_mode(car_id:int, mode:str) -> int   # reliable command seq(비동기). ACCEPTED 는 나중.
- push_control(car_id:int, throttle, steering) -> None   # ★ 반환 None. control_seq 서버 소유.
- stop_control(car_id:int) -> None
- 속성 callback: on_status/on_ready/on_resync/on_comm_fail/on_comm_recovered/
                 on_command_rejected/on_command_result   # register_comm_callbacks() 없음
- direct_control_enabled: bool   # _tick_loop 는 이 플래그 True 일 때만 latest_control 송신
- 내부 car_id 는 int(1,2). wire 경계에서만 "CAR_01".

async ACCEPTED: deliver_command_result(car_id, seq, result) 로 terminal 결과 주입 → 콜백 발화.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .mock_firmware_spec import MockFirmware

CONTROL_INTERVAL_S = 0.100
NEGATIVE_RESULTS = frozenset({"REJECTED", "INVALID_STATE", "HOLD", "STALE_ROUTE",
                              "TARGET_NOT_LOADED", "SEQ_CONFLICT", "STALE_SEQ"})


def wire_car_id(car_id: int) -> str:
    return f"CAR_{int(car_id):02d}"


@dataclass
class _CarState:
    session_id: str
    control_seq: int = 0
    reliable_seq: int = 0
    latest_throttle: float = 0.0
    latest_steering: float = 0.0
    has_control: bool = False


class SpecVehicleServer:
    def __init__(self, firmware: Optional[MockFirmware] = None,
                 known_car_ids: Optional[set] = None) -> None:
        self.firmware = firmware or MockFirmware()
        self.known_car_ids = known_car_ids or {1, 2}
        self._cars: Dict[int, _CarState] = {}
        self.on_status: Optional[Callable] = None
        self.on_ready: Optional[Callable] = None
        self.on_resync: Optional[Callable] = None
        self.on_comm_fail: Optional[Callable] = None
        self.on_comm_recovered: Optional[Callable] = None
        self.on_command_rejected: Optional[Callable] = None
        self.on_command_result: Optional[Callable] = None
        self.direct_control_enabled: bool = False
        self.mode_log: List[str] = []
        self._pending_result = None

    def register_car(self, car_id: int) -> None:
        if car_id not in self._cars:
            self._cars[car_id] = _CarState(session_id=f"S{car_id:04d}")

    def _session(self, car_id: int) -> _CarState:
        self.register_car(car_id)
        return self._cars[car_id]

    # reliable SET_MODE (비동기)
    def send_set_mode(self, car_id: int, mode: str) -> int:
        assert isinstance(car_id, int), "production car_id 는 int 여야 함"
        s = self._session(car_id)
        s.reliable_seq += 1
        self.mode_log.append(f"{car_id}:SET_MODE:{mode}:seq{s.reliable_seq}")
        self._pending_result = (car_id, s.reliable_seq, mode)
        return s.reliable_seq

    def deliver_command_result(self, car_id: int, seq: int, result: str) -> None:
        if result in NEGATIVE_RESULTS and self.on_command_rejected:
            self.on_command_rejected(car_id, result, {"seq": seq})
        if self.on_command_result:
            self.on_command_result(car_id, seq, result, {"seq": seq})

    def auto_accept_set_mode(self, car_id: int) -> str:
        cid, seq, mode = self._pending_result
        result = self.firmware.on_set_mode(mode)
        self.deliver_command_result(cid, seq, result)
        return result

    # 제어 API ★ 반환 None
    def push_control(self, car_id: int, throttle: float, steering: float) -> None:
        assert isinstance(car_id, int), "production car_id 는 int 여야 함"
        s = self._cars.get(car_id)
        if s is None:
            return
        s.control_seq += 1
        s.latest_throttle = float(throttle)
        s.latest_steering = float(steering)
        s.has_control = True

    def stop_control(self, car_id: int) -> None:
        self.push_control(car_id, 0.0, 0.0)

    def latest_control(self, car_id: int):
        s = self._cars[car_id]
        return (s.latest_throttle, s.latest_steering)

    def control_seq(self, car_id: int) -> int:
        return self._cars[car_id].control_seq

    # _tick_loop 재현: direct_control_enabled True 일 때만 firmware 로 송신
    def tick(self, car_id: int, now: float = 0.0) -> None:
        if not self.direct_control_enabled:
            return
        s = self._cars.get(car_id)
        if s is None or not s.has_control:
            return
        self.firmware.on_direct_control(s.control_seq, s.latest_throttle,
                                        s.latest_steering, now)

    # comm fault 트리거(테스트용)
    def trigger_comm_fail(self, car_id: int) -> None:
        if self.on_comm_fail:
            self.on_comm_fail(car_id, {"type": "COMM_TIMEOUT"})

    def trigger_comm_recovered(self, car_id: int) -> None:
        if self.on_comm_recovered:
            self.on_comm_recovered(car_id)   # ★ 실제 API: 인자 1개(car_id)

    def trigger_resync(self, car_id: int) -> None:
        if self.on_resync:
            self.on_resync(car_id, {"type": "HELLO"})
