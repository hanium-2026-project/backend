"""MockFirmware — 실제 ESP32 계약(FIRMWARE_CONTRACT_EXCERPTS)을 재현한 test double.

⚠ 이것은 실제 backend `comm/tests/mock_firmware.py` 가 아니다. 이 환경에는 실제 backend
full source 가 없어, reference 계약을 충실히 재현했다. 실제 backend 가 있으면
production integration test 는 실제 MockFirmware 를 자동으로 사용하도록 되어 있다
(integration/tests/test_production_integration.py 의 import 폴백 참고).

재현한 계약(actual firmware):
- SET_MODE       : READY 에서만. mode 전환이 direct state 를 무효화. ACCEPTED/REJECTED.
- DIRECT_CONTROL : REMOTE_DIRECT 에서만. control_seq 증가. throttle≠0→MOVING, 0(MOVING)→READY.
- DIRECT timeout : REMOTE_DIRECT+MOVING 에서 500ms 무갱신 → safeStop→WAITING.
- WAYPOINT       : WAYPOINT_AUTO 에서만(REMOTE_DIRECT 이면 거부·기록).
- GO             : ENABLE_WAYPOINT_AUTO_CONTROL=0 → HOLD.
- HEARTBEAT      : 1000ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class FwMode(str, Enum):
    SYNCING = "SYNCING"
    READY = "READY"
    REMOTE_DIRECT = "REMOTE_DIRECT"
    WAYPOINT_AUTO = "WAYPOINT_AUTO"


class FwState(str, Enum):
    READY = "READY"
    WAITING = "WAITING"
    MOVING = "MOVING"


DIRECT_CONTROL_TIMEOUT_S = 0.500
HEARTBEAT_TIMEOUT_S = 1.000


@dataclass
class MockFirmware:
    mode: FwMode = FwMode.READY
    state: FwState = FwState.READY
    last_control_seq: int = 0
    last_throttle: float = 0.0
    last_steering: float = 0.0
    last_direct_time: Optional[float] = None
    # EMERGENCY_STOP/ERROR/COMM_TIMEOUT 등 SET_MODE 를 막는 상태 에뮬레이션
    _blocked: bool = False
    # 관측/감사용
    received_types: List[str] = field(default_factory=list)
    direct_history: List[Tuple[int, float, float]] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)  # (type, reason)

    # ---------------------------------------------------------------- SET_MODE
    def on_set_mode(self, mode: str) -> str:
        """실제 mock 계약: state 가 EMERGENCY_STOP/ERROR/COMM_TIMEOUT/MOVING 이면 INVALID_STATE.
        그 외에는 mode 전환 후 ACCEPTED.
        """
        self.received_types.append("SET_MODE")
        if self.state is FwState.MOVING or self._blocked:
            self.rejected.append(("SET_MODE", "INVALID_STATE"))
            return "INVALID_STATE"
        try:
            self.mode = FwMode(mode)
        except ValueError:
            self.rejected.append(("SET_MODE", "UNKNOWN_MODE"))
            return "INVALID_STATE"
        # mode 전환은 direct state 무효화
        self.last_control_seq = 0
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.last_direct_time = None
        if self.mode is FwMode.REMOTE_DIRECT:
            self.state = FwState.READY
        return "ACCEPTED"

    # ---------------------------------------------------------- DIRECT_CONTROL
    def on_direct_control(self, control_seq: int, throttle: float, steering: float,
                          now: float) -> str:
        self.received_types.append("DIRECT_CONTROL")
        if self.mode is not FwMode.REMOTE_DIRECT:
            self.rejected.append(("DIRECT_CONTROL", "NOT_REMOTE_DIRECT"))
            return "IGNORED"
        if control_seq <= self.last_control_seq:
            self.rejected.append(("DIRECT_CONTROL", "SEQ_NOT_INCREASING"))
            return "IGNORED"
        self.last_control_seq = control_seq
        self.last_throttle = throttle
        self.last_steering = steering
        self.last_direct_time = now
        self.direct_history.append((control_seq, throttle, steering))
        if throttle != 0.0:
            self.state = FwState.MOVING
        elif self.state is FwState.MOVING:
            self.state = FwState.READY
        return "APPLIED"

    # ----------------------------------------------------------------- WAYPOINT
    def on_waypoint(self, **_kw) -> str:
        self.received_types.append("WAYPOINT")
        if self.mode is not FwMode.WAYPOINT_AUTO:
            self.rejected.append(("WAYPOINT", "NOT_WAYPOINT_AUTO"))
            return "INVALID_STATE"
        self.state = FwState.WAITING
        return "ACCEPTED"

    def on_go(self, **_kw) -> str:
        self.received_types.append("GO")
        # ENABLE_WAYPOINT_AUTO_CONTROL=0 → HOLD
        return "HOLD"

    # ------------------------------------------------------------------- timeout
    def tick_timeout(self, now: float) -> Optional[str]:
        """DIRECT_CONTROL timeout fail-safe. 최후 방어선(호스트가 zero 를 보내야 정상)."""
        if (self.mode is FwMode.REMOTE_DIRECT and self.state is FwState.MOVING
                and self.last_direct_time is not None
                and now - self.last_direct_time > DIRECT_CONTROL_TIMEOUT_S):
            self.state = FwState.WAITING
            self.last_throttle = 0.0
            self.last_steering = 0.0
            return "SAFE_STOP"
        return None

    # ------------------------------------------------------------------- helpers
    def count(self, msg_type: str) -> int:
        return self.received_types.count(msg_type)
