"""전방 쿠션 ↔ 차량 매칭 (통합 문서 §6.7).

2클래스 모델(RC_CAR + FRONT_CUSHION)이 준비되면, 어느 쿠션이 어느 차량의
것인지 정해야 heading 을 계산할 수 있다. 다중 차량에서 잘못 묶이면 차가
반대 방향으로 정렬하므로, 확실한 후보만 채택하고 애매하면 포기한다
(포기해도 궤적 heading 으로 fallback 되므로 안전하다).

판단 기준(문서 순서대로):
  1. 쿠션 중심이 차량 bbox 내부 또는 인접 영역에 있는가
  2. 차량 중심과 쿠션 중심의 거리
  3. 이전 프레임의 tracking ID (같은 쌍이 유지되면 가산점)
  4. 이전 heading 과 후보 heading 의 차이
  5. 각 객체의 confidence
  6. 한 차량에 쿠션이 둘 이상 붙지 않도록 1:1 로 확정
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .vehicle_detector import LABEL_CAR, LABEL_CUSHION, Detection

# 차량 bbox 를 이 비율만큼 넓힌 영역까지 "인접"으로 본다
ADJACENT_MARGIN_RATIO = 0.6
# 차량 대각선 길이 대비 이 배수를 넘으면 다른 차량의 쿠션으로 본다
MAX_PAIR_DISTANCE_RATIO = 0.9
# 이전 heading 과 이 각도 이상 어긋나면 감점 (급회전은 프레임 간 크지 않다)
HEADING_JUMP_PENALTY_DEG = 90.0


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diagonal(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return math.hypot(x2 - x1, y2 - y1)


def _inside_or_adjacent(point: tuple[float, float],
                        bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    mx = (x2 - x1) * ADJACENT_MARGIN_RATIO
    my = (y2 - y1) * ADJACENT_MARGIN_RATIO
    return (x1 - mx) <= point[0] <= (x2 + mx) and (y1 - my) <= point[1] <= (y2 + my)


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class VehicleCushionPair:
    """확정된 차량-쿠션 쌍. 좌표는 모두 픽셀."""

    car: Detection
    cushion: Detection
    car_center_px: tuple[float, float]
    cushion_center_px: tuple[float, float]
    score: float

    @property
    def track_id(self) -> int | None:
        return self.car.track_id


def associate(
    detections: list[Detection],
    previous_heading: dict[int, float] | None = None,
    image_heading_of: "callable | None" = None,
) -> tuple[list[VehicleCushionPair], list[Detection]]:
    """탐지 목록을 (차량-쿠션 쌍, 쿠션 없는 차량) 으로 나눈다.

    Args:
        detections: 한 프레임의 전체 탐지 (차량·쿠션 혼재).
        previous_heading: track_id → 직전 heading(도). 연속성 가산에 쓴다.
        image_heading_of: (car_center, cushion_center) → heading(도) 변환 함수.
                          heading 은 맵 좌표계에서 계산해야 하므로(§6.5) 호출측이
                          homography 를 아는 함수를 넘긴다. None 이면 연속성
                          검사를 건너뛴다.

    Returns:
        (확정된 쌍 목록, 쿠션이 붙지 않은 차량 목록)
    """
    cars = [d for d in detections if d.label == LABEL_CAR]
    cushions = [d for d in detections if d.label == LABEL_CUSHION]
    if not cars:
        return [], []
    if not cushions:
        return [], cars

    prev = previous_heading or {}
    candidates: list[tuple[float, Detection, Detection]] = []
    for car in cars:
        car_c = _center(car.bbox)
        limit = _diagonal(car.bbox) * MAX_PAIR_DISTANCE_RATIO
        for cushion in cushions:
            cu_c = _center(cushion.bbox)
            if not _inside_or_adjacent(cu_c, car.bbox):
                continue                                   # 기준 1
            dist = math.hypot(cu_c[0] - car_c[0], cu_c[1] - car_c[1])
            if dist > limit:
                continue                                   # 기준 2
            score = 1.0 - (dist / limit)                   # 가까울수록 높게
            score += 0.3 * (car.confidence + cushion.confidence) / 2.0   # 기준 5
            if car.track_id is not None and car.track_id in prev and image_heading_of:
                cand = image_heading_of(car_c, cu_c)       # 기준 3·4
                if cand is not None:
                    jump = _angle_diff(cand, prev[car.track_id])
                    score += 0.4 * (1.0 - min(jump / HEADING_JUMP_PENALTY_DEG, 1.0))
            candidates.append((score, car, cushion))

    # 점수 높은 쌍부터 1:1 확정 (기준 6)
    candidates.sort(key=lambda c: c[0], reverse=True)
    used_cars: set[int] = set()
    used_cushions: set[int] = set()
    pairs: list[VehicleCushionPair] = []
    for score, car, cushion in candidates:
        if id(car) in used_cars or id(cushion) in used_cushions:
            continue
        used_cars.add(id(car))
        used_cushions.add(id(cushion))
        pairs.append(VehicleCushionPair(
            car=car, cushion=cushion,
            car_center_px=_center(car.bbox), cushion_center_px=_center(cushion.bbox),
            score=score,
        ))

    unpaired = [c for c in cars if id(c) not in used_cars]
    return pairs, unpaired
