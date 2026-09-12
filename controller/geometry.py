"""순수 기하 계산 유틸 (stdlib math 만 사용, 결정론적).

좌표 규약 (models.py 와 동일)
----------------------------
- +X = 오른쪽, +Y = 위 (mm)
- heading 0° = +X, 90° = +Y, 증가 = CCW
- bearing 도 동일 규약: atan2(dy, dx), 0° = +X, CCW 양수

heading_error 부호 의미
-----------------------
heading_error = wrap180(bearing - heading)
- > 0 : 목표가 현재 진행방향 기준 **CCW(왼쪽)** 에 있음  → LEFT 로 틀어야 함
- < 0 : 목표가 **CW(오른쪽)** 에 있음                    → RIGHT 로 틀어야 함
- = 0 : 정면
"""

from __future__ import annotations

import math


def wrap180(deg: float) -> float:
    """각도를 [-180, 180) 범위로 정규화한다.

    backend control/waypoint_controller.py 의 wrap180 과 동일한 공식/범위(±180 경계는 -180).
    """
    return (deg + 180.0) % 360.0 - 180.0


def wrap360(deg: float) -> float:
    """각도를 [0, 360) 범위로 정규화한다."""
    return deg % 360.0


def distance_mm(x0: float, y0: float, x1: float, y1: float) -> float:
    """두 점 사이 유클리드 거리 (mm)."""
    return math.hypot(x1 - x0, y1 - y0)


def bearing_deg(x_from: float, y_from: float, x_to: float, y_to: float) -> float:
    """from → to 방향의 bearing (degree, 0°=+X, CCW 양수).

    두 점이 동일하면 0.0 을 반환한다(atan2(0,0)=0).
    """
    return math.degrees(math.atan2(y_to - y_from, x_to - x_from))


def heading_error_deg(bearing: float, heading: float) -> float:
    """목표 bearing 과 현재 heading 의 차이를 (-180, 180] 로 반환.

    > 0 이면 목표가 왼쪽(CCW), < 0 이면 오른쪽(CW).
    """
    return wrap180(bearing - heading)


def clamp(value: float, lo: float, hi: float) -> float:
    """value 를 [lo, hi] 로 제한."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
