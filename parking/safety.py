"""충돌 감지 안전장치 (회의 8번 시퀀스의 트리거).

두 가지 위험을 감지한다:

1. 근접(PROXIMITY)      — 차간 거리가 안전거리 미만
2. 경로 충돌(PATH_CONFLICT) — 각 차량의 "현재위치→다음 waypoint" 선분이
                              간섭 반경 이내로 접근할 것으로 예측

감지 시 우선순위가 낮은 차량에 대해 WAIT 이벤트를 발생시킨다.
우선순위 규칙: 경로 진행률(현재 waypoint index / 전체)이 높은 차가 통과,
낮은 차가 대기. 동률이면 car_id가 작은 쪽이 통과 (결정적 규칙).

이 모듈은 판단만 한다. 실제 WAIT 전송·경로 재생성은 호출측
(VehicleLink.send_wait → route_id 증가 → build_waypoints → GO)이 수행한다.

좌표 단위: 내부 표준 mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .waypoints import Waypoint

# RC카 전장 약 300mm 기준 — 중심 간 거리가 이보다 가까우면 즉시 위험
SAFETY_DISTANCE_MM: float = 350.0
# 경로 선분 간 간섭 판정 반경 (차폭 + 여유)
PATH_CLEARANCE_MM: float = 250.0


@dataclass(frozen=True)
class VehiclePose:
    """안전 판단에 필요한 차량 스냅샷."""

    car_id: int
    position: tuple[float, float]          # mm
    next_waypoint: Waypoint | None         # 현재 향하는 waypoint (없으면 정지 상태)
    progress: float                        # 경로 진행률 0.0~1.0
    is_moving: bool = True


@dataclass(frozen=True)
class SafetyEvent:
    """충돌 위험 감지 결과 — stop_car_id 차량을 WAIT시켜야 한다."""

    stop_car_id: int
    keep_car_id: int
    reason: str                            # "PROXIMITY" | "PATH_CONFLICT"
    distance_mm: float


# ─── 기하 유틸 ───────────────────────────────────────────────────────────────

def _seg_seg_distance(
    p1: tuple[float, float], p2: tuple[float, float],
    q1: tuple[float, float], q2: tuple[float, float],
) -> float:
    """두 선분 사이 최소 거리 (2D)."""

    def dot(a, b): return a[0] * b[0] + a[1] * b[1]
    def sub(a, b): return (a[0] - b[0], a[1] - b[1])

    def point_seg(p, a, b) -> float:
        ab = sub(b, a)
        denom = dot(ab, ab)
        if denom == 0:
            return math.hypot(*sub(p, a))
        t = max(0.0, min(1.0, dot(sub(p, a), ab) / denom))
        proj = (a[0] + t * ab[0], a[1] + t * ab[1])
        return math.hypot(*sub(p, proj))

    def ccw(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    # 교차하면 거리 0
    d1, d2 = ccw(p1, p2, q1), ccw(p1, p2, q2)
    d3, d4 = ccw(q1, q2, p1), ccw(q1, q2, p2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return 0.0

    return min(
        point_seg(q1, p1, p2), point_seg(q2, p1, p2),
        point_seg(p1, q1, q2), point_seg(p2, q1, q2),
    )


def _pick_waiter(a: VehiclePose, b: VehiclePose) -> tuple[int, int]:
    """(stop_car_id, keep_car_id) — 진행률 낮은 차가 대기."""
    if a.progress == b.progress:
        return (max(a.car_id, b.car_id), min(a.car_id, b.car_id))
    return (a.car_id, b.car_id) if a.progress < b.progress else (b.car_id, a.car_id)


# ─── 감지기 ──────────────────────────────────────────────────────────────────

class CollisionMonitor:
    """프레임마다 check()를 호출해 충돌 위험 차량 쌍을 찾는다.

    이미 WAIT 중인 차량(is_moving=False)은 새로 멈출 대상에서 제외하되,
    근접 위험의 기준점으로는 계속 사용한다 (정지 차량 뒤에서 접근하는 차 감지).
    """

    def __init__(
        self,
        safety_distance_mm: float = SAFETY_DISTANCE_MM,
        path_clearance_mm: float = PATH_CLEARANCE_MM,
    ) -> None:
        self.safety_distance_mm = safety_distance_mm
        self.path_clearance_mm = path_clearance_mm

    def check(self, poses: list[VehiclePose]) -> list[SafetyEvent]:
        events: list[SafetyEvent] = []
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                ev = self._check_pair(poses[i], poses[j])
                if ev is not None:
                    events.append(ev)
        return events

    def _check_pair(self, a: VehiclePose, b: VehiclePose) -> SafetyEvent | None:
        # 1) 근접 — 둘 중 최소 한 대가 이동 중일 때만 의미
        gap = math.hypot(a.position[0] - b.position[0], a.position[1] - b.position[1])
        if gap < self.safety_distance_mm and (a.is_moving or b.is_moving):
            stop_id, keep_id = self._pick_moving_waiter(a, b)
            return SafetyEvent(stop_id, keep_id, "PROXIMITY", round(gap, 1))

        # 2) 경로 충돌 예측 — 양쪽 다 이동 중이고 다음 waypoint가 있을 때
        if a.is_moving and b.is_moving and a.next_waypoint and b.next_waypoint:
            seg_gap = _seg_seg_distance(
                a.position, (a.next_waypoint.x, a.next_waypoint.y),
                b.position, (b.next_waypoint.x, b.next_waypoint.y),
            )
            if seg_gap < self.path_clearance_mm:
                stop_id, keep_id = _pick_waiter(a, b)
                return SafetyEvent(stop_id, keep_id, "PATH_CONFLICT", round(seg_gap, 1))
        return None

    @staticmethod
    def _pick_moving_waiter(a: VehiclePose, b: VehiclePose) -> tuple[int, int]:
        """근접 시: 이미 정지한 차는 세울 수 없으므로 이동 중인 차를 세운다."""
        if not a.is_moving:
            return (b.car_id, a.car_id)
        if not b.is_moving:
            return (a.car_id, b.car_id)
        return _pick_waiter(a, b)
