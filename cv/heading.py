"""궤적 기반 차량 heading 추정 (1차 구현).

회의 스펙 (SW팀 전달용 차량 인식 및 Waypoint 설계 변경사항):
- 각도 기준: 오른쪽 0°, 위쪽 90°, 왼쪽 180°, 아래쪽 270° (반시계, 맵 좌표계 +y 위)
- heading_source 우선순위 (§6.6): FRONT_CUSHION > TRAJECTORY > LAST_VALID
- 궤적 방식은 정지 중에는 방향을 알 수 없고, 후진 시 머리 방향과 180° 반대다.
  전방 쿠션이 보이면 그 값을 우선 사용한다.

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
MAX_HEADING_JUMP_DEG: float = 45.0
FRONT_JUMP_CONFIRM_DEG: float = 15.0
FRONT_JUMP_CONFIRM_FRAMES: int = 2


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
        self._last_source: dict[int, str] = {}
        # A single bad cushion association must not poison a track forever.
        # Large FRONT_CUSHION corrections are accepted after two mutually
        # consistent current-frame observations.
        self._pending_front_jump: dict[int, tuple[float, int]] = {}

    @staticmethod
    def _delta(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _accept(self, track_id: int, heading: float, source: str,
                moving: bool) -> HeadingResult:
        previous = self._last_valid.get(track_id)
        previous_source = self._last_source.get(track_id)
        if (previous is not None
                and not (source == "FRONT_CUSHION"
                         and previous_source == "TRAJECTORY")
                and self._delta(heading, previous) > MAX_HEADING_JUMP_DEG):
            if source == "FRONT_CUSHION":
                pending, count = self._pending_front_jump.get(
                    track_id, (heading, 0))
                if self._delta(heading, pending) <= FRONT_JUMP_CONFIRM_DEG:
                    count += 1
                else:
                    pending, count = heading, 1
                if count < FRONT_JUMP_CONFIRM_FRAMES:
                    self._pending_front_jump[track_id] = (pending, count)
                    return HeadingResult(previous, "LAST_VALID", is_moving=moving)
                self._pending_front_jump.pop(track_id, None)
            else:
                self._pending_front_jump.pop(track_id, None)
                return HeadingResult(previous, "LAST_VALID", is_moving=moving)
        else:
            self._pending_front_jump.pop(track_id, None)
        self._last_valid[track_id] = heading
        self._last_source[track_id] = source
        return HeadingResult(heading, source, is_moving=moving)

    def update(self, track_id: int, position: tuple[float, float],
               front_point: tuple[float, float] | None = None) -> HeadingResult:
        """새 프레임의 위치를 반영하고 현재 heading 추정을 반환한다.

        Args:
            track_id: 추적 대상 식별자.
            position: 차량 중심 (맵 좌표).
            front_point: 전방 쿠션 중심 (맵 좌표). 주어지면 최우선으로 사용한다.
                         2클래스 모델이 없거나 쿠션이 가려지면 None 을 넘긴다.
        """
        hist = self._history.setdefault(track_id, deque(maxlen=self.window))
        hist.append(position)

        # 1순위: 전방 쿠션 — 정지 중에도, 후진 중에도 머리 방향을 그대로 준다
        if front_point is not None:
            dx = front_point[0] - position[0]
            dy = front_point[1] - position[1]
            if math.hypot(dx, dy) >= 1e-6:
                heading = math.degrees(math.atan2(dy, dx)) % 360.0
                moving = self._is_moving(hist)
                return self._accept(track_id, heading, "FRONT_CUSHION", moving)

        if len(hist) >= 2:
            x0, y0 = hist[0]
            x1, y1 = hist[-1]
            dx, dy = x1 - x0, y1 - y0
            if math.hypot(dx, dy) >= self.min_move:
                # 맵 좌표계 +y=위 기준 반시계 각도 (오른쪽 0°, 위 90°)
                heading = math.degrees(math.atan2(dy, dx)) % 360.0
                return self._accept(track_id, heading, "TRAJECTORY", True)

        if track_id in self._last_valid:
            return HeadingResult(self._last_valid[track_id], "LAST_VALID", is_moving=False)
        return HeadingResult(None, None, is_moving=False)

    def _is_moving(self, hist: deque[tuple[float, float]]) -> bool:
        """창 안에서 최소 이동 거리를 넘었는지 (쿠션 heading 사용 시 참고값)."""
        if len(hist) < 2:
            return False
        x0, y0 = hist[0]
        x1, y1 = hist[-1]
        return math.hypot(x1 - x0, y1 - y0) >= self.min_move

    def remove(self, track_id: int) -> None:
        """추적 종료된 차량의 이력 제거."""
        self._history.pop(track_id, None)
        self._last_valid.pop(track_id, None)
        self._last_source.pop(track_id, None)
        self._pending_front_jump.pop(track_id, None)
