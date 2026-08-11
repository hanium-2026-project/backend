"""DIRECT_CONTROL transport — MANUAL/AUTO_HOST 가 공용으로 사용하는 저수준 송신 계층.

프롬프트 4장: 알고리즘/제어권은 분리하되 protocol/transport 는 공용화한다.

- control_seq 는 monotonic 증가(모든 DIRECT_CONTROL 에 대해, zero 포함).
- steering 은 이미 wire-ready(음수=LEFT) 라고 가정하고 그대로 실어보낸다.
- 실제 소켓/네트워크는 이 모듈이 다루지 않는다. 주입된 sink(callable) 로 payload 를 넘긴다.
  (backend 에서는 VehicleServer.push_control 을 sink 로 연결.)

타이밍 계약
----------
- host 송신 주기(send_period_s) 는 firmware DIRECT_CONTROL timeout(direct_timeout_s)보다
  충분히 빨라야 한다. 기본 100ms 송신 vs 500ms timeout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from controller.models import ControlCommand

# payload sink: dict 를 받아 실제로 전송하는 함수. 반환값 무시.
Sink = Callable[[Dict], None]


@dataclass(frozen=True)
class TransportTiming:
    """송신 주기와 firmware timeout 계약(초)."""

    send_period_s: float = 0.100      # host 송신 주기 (backend CONTROL_INTERVAL=100ms)
    direct_timeout_s: float = 0.500   # firmware DIRECT_CONTROL_TIMEOUT_MS=500ms
    heartbeat_timeout_s: float = 1.000  # firmware HEARTBEAT_TIMEOUT_MS=1000ms

    @property
    def is_safe(self) -> bool:
        """송신 주기가 timeout 보다 충분히 빠른가(여유 2x 이상)."""
        return self.send_period_s > 0.0 and self.send_period_s * 2.0 <= self.direct_timeout_s

    def margin_ratio(self) -> float:
        """direct_timeout / send_period. 클수록 안전."""
        if self.send_period_s <= 0:
            return float("inf")
        return self.direct_timeout_s / self.send_period_s


class DirectControlSender:
    """ControlCommand → DIRECT_CONTROL payload 전송(공용 transport)."""

    def __init__(
        self,
        sink: Optional[Sink] = None,
        timing: Optional[TransportTiming] = None,
    ) -> None:
        self._sink: Sink = sink or self._record
        self._timing = timing or TransportTiming()
        self._seq = 0
        self._sent: List[Dict] = []  # sink 미주입 시 테스트용 기록 버퍼

    @property
    def timing(self) -> TransportTiming:
        return self._timing

    @property
    def control_seq(self) -> int:
        return self._seq

    @property
    def sent(self) -> List[Dict]:
        """기본 sink 사용 시 전송된 payload 기록(테스트용)."""
        return self._sent

    def send_command(self, cmd: ControlCommand) -> Dict:
        """ControlCommand 를 DIRECT_CONTROL 로 전송하고 payload 를 반환."""
        return self._send(cmd.throttle, cmd.steering)

    def send_zero(self) -> Dict:
        """명시적 zero(0,0) DIRECT_CONTROL 전송. camera-loss/HOLD/FAULT 시 사용."""
        return self._send(0.0, 0.0)

    # --------------------------------------------------------------- 내부
    def _send(self, throttle: float, steering: float) -> Dict:
        self._seq += 1
        payload = {
            "type": "DIRECT_CONTROL",
            "control_seq": self._seq,
            "throttle": round(float(throttle), 4),
            "steering": round(float(steering), 4),  # wire-ready(음수=LEFT)
        }
        self._sink(payload)
        return payload

    def _record(self, payload: Dict) -> None:
        self._sent.append(payload)
