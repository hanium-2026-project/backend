"""제어기 입출력 데이터 모델.

단위 규약 (backend 내부 규약과 동일)
------------------------------------
- 위치: **mm**, 원점은 바닥판 좌하단, +X = 오른쪽, +Y = 위
- heading: **degree**, 0° = +X(오른쪽), 90° = +Y(위), 증가 방향 = CCW(반시계)
- 거리 오차 출력: cm (사람이 읽기 쉽게)
- timestamp: 초 단위 monotonic clock 값(외부에서 주입). Pose 신선도 판정에 사용.

이 모듈은 stdlib 만 사용한다 (backend/network 의존 없음).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ControlMode(str, Enum):
    """제어기 상태 라벨.

    - HOLD    : 안전 정지(무효/비신선 pose, drive 비활성 등). throttle=steering=0.
    - DRIVE   : 정상 주행 중(throttle > 0).
    - BRAKE   : 목표 접근 감속 중이지만 아직 도착 전(throttle 이 0 으로 떨어진 경우).
    - ALIGN   : 위치는 도착했으나 heading 정렬이 필요(전진 전용 controller 는 정지 후 보고).
    - ARRIVED : 위치(+필요 시 heading) 도착 완료. throttle=steering=0.
    """

    HOLD = "HOLD"
    DRIVE = "DRIVE"
    BRAKE = "BRAKE"
    ALIGN = "ALIGN"
    ARRIVED = "ARRIVED"


@dataclass(frozen=True)
class Pose:
    """카메라가 관측한 차량 pose (mm / degree)."""

    x_mm: float
    y_mm: float
    heading_deg: Optional[float]  # None 이면 heading 미관측 → 안전상 정지
    timestamp: float              # monotonic 초. 신선도 판정 기준.
    valid: bool = True            # 트래킹 실패/로스트 시 False → 안전상 정지

    @property
    def has_heading(self) -> bool:
        return self.heading_deg is not None


@dataclass(frozen=True)
class Waypoint:
    """목표 지점 (mm / degree).

    backend parking/waypoints.py 의 Waypoint 와 필드명을 최대한 맞춰
    adapter 매핑을 단순화한다. (단, 이 dataclass 는 backend 를 import 하지 않는다.)
    """

    x_mm: float
    y_mm: float
    target_heading_deg: Optional[float] = None
    speed_cm_s: float = 12.0
    position_tolerance_cm: float = 8.0
    heading_tolerance_deg: float = 30.0
    heading_required: bool = False
    is_final: bool = False
    # 아래는 라우팅/디버깅용 메타. 제어 계산에는 직접 쓰지 않음.
    route_id: Optional[int] = None
    waypoint_id: Optional[int] = None
    phase: Optional[str] = None


@dataclass(frozen=True)
class ControlCommand:
    """제어기 출력.

    steering / throttle 은 **wire-ready** 값이다. 즉 ESP32 DIRECT_CONTROL 에
    그대로 넣을 수 있다.

    steering  : [-1, 1], **ESP32 실제 wire 부호** — 음수 = LEFT, 0 = CENTER, 양수 = RIGHT.
    throttle  : [-1, 1], 양수 = 전진. (1차 controller 는 기본 전진 전용)
    """

    throttle: float
    steering: float
    mode: ControlMode
    arrived: bool
    distance_error_cm: float
    heading_error_deg: float
    target_bearing_deg: float
    reason: str = ""
    # 디버그: wire 변환 전 논리 steering(양수 = LEFT 요구). 테스트/로그용.
    logical_steering: float = field(default=0.0)

    @property
    def is_stopped(self) -> bool:
        return self.throttle == 0.0 and self.steering == 0.0
