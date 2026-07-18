"""궤적 기반 차량 heading 추정 (1차 구현).

회의 스펙 (SW팀 전달용 차량 인식 및 Waypoint 설계 변경사항):
- 각도 기준: 오른쪽 0°, 위쪽 90°, 왼쪽 180°, 아래쪽 270° (반시계, 맵 좌표계 +y 위)
- heading_source: TRAJECTORY(이동 중 궤적) / LAST_VALID(정지 시 마지막 유효값)
- 후진 시 궤적 heading은 차량 머리 방향과 180° 다를 수 있음 → 2차(FRONT_CUSHION)에서 해결

사용법::

    est = HeadingEstimator()
    h = est.update(track_id=1, position=(42.5, 76.0), timestamp=0.033)
    # h.heading_deg, h.source ("TRAJECTORY" | "LAST_VALID" | None)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


# 이동으로 인정하는 최소 변위 (같은 좌표 단위 — cm 권장). 탐지 지터(±1~2cm)보다 커야 함.
MIN_MOVE_DISTANCE: float = 3.0
# heading 계산에 사용할 궤적 창 크기 (프레임 수)
TRAJECTORY_WINDOW: int = 5


@dataclass(frozen=True)
class HeadingResult:
    """heading 추정 결과. heading_deg가 None이면 아직 유효한 추정 없음."""

    heading_deg: float | None
    source: str | None            # "TRAJECTORY" | "LAST_VALID" | None
    is_moving: bool


class HeadingEstimator:
    """track_id별 최근 궤적을 유지하며 heading을 추정한다.

    프레임마다 update()를 호출하면:
    - 창 안에서 MIN_MOVE_DISTANCE 이상 이동 → 변위 벡터로 heading 계산 (TRAJECTORY)
    - 정지 중 → 마지막 유효 heading 유지 (LAST_VALID)
    - 유효 이력이 아직 없으면 heading_deg=None
    """

    def __init__(
        self,
        window: int = TRAJECTORY_WINDOW,
        min_move: float = MIN_MOVE_DISTANCE,
    ) -> None:
        self.window = window
        self.min_move = min_move
        self._history: dict[int, deque[tuple[float, float]]] = {}
        self._last_valid: dict[int, float] = {}

    def update(self, track_id: int, position: tuple[float, float]) -> HeadingResult:
        """새 프레임의 위치를 반영하고 현재 heading 추정을 반환한다."""
        hist = self._history.setdefault(track_id, deque(maxlen=self.window))
        hist.append(position)

        if len(hist) >= 2:
            x0, y0 = hist[0]
            x1, y1 = hist[-1]
            dx, dy = x1 - x0, y1 - y0
            if math.hypot(dx, dy) >= self.min_move:
                # 맵 좌표계 +y=위 기준 반시계 각도 (오른쪽 0°, 위 90°)
                heading = math.degrees(math.atan2(dy, dx)) % 360.0
                self._last_valid[track_id] = heading
                return HeadingResult(heading, "TRAJECTORY", is_moving=True)

        if track_id in self._last_valid:
            return HeadingResult(self._last_valid[track_id], "LAST_VALID", is_moving=False)
        return HeadingResult(None, None, is_moving=False)

    def remove(self, track_id: int) -> None:
        """추적 종료된 차량의 이력 제거."""
        self._history.pop(track_id, None)
        self._last_valid.pop(track_id, None)
