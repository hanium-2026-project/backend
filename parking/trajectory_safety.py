"""Common pre-execution safety validation for every production trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .waypoints import (CAR_LENGTH_MM, CAR_WIDTH_MM, MIN_TURN_RADIUS_MM,
                        SlotSpec, _car_footprint, default_slot_specs)


@dataclass(frozen=True)
class TrajectorySafetyResult:
    safe: bool
    reason: str = ""
    sampled_poses: int = 0
    path_length_mm: float = 0.0
    min_clearance_mm: float = float("inf")


def _get(wp: Any, *names: str, default=None):
    for name in names:
        if hasattr(wp, name):
            value = getattr(wp, name)
            return value.value if hasattr(value, "value") else value
    return default


def _xy(wp: Any) -> tuple[float, float]:
    return float(_get(wp, "x_mm", "x")), float(_get(wp, "y_mm", "y"))


def _axes(poly: list[tuple[float, float]]):
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy)
        if norm:
            yield -dy / norm, dx / norm


def _overlap(a: list[tuple[float, float]],
             b: list[tuple[float, float]]) -> bool:
    for ax, ay in (*_axes(a), *_axes(b)):
        pa = [x * ax + y * ay for x, y in a]
        pb = [x * ax + y * ay for x, y in b]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return False
    return True


def _slot_poly(spec: SlotSpec) -> list[tuple[float, float]]:
    x0, x1 = spec.center_x - spec.width / 2, spec.center_x + spec.width / 2
    y0, y1 = spec.center_y - spec.length / 2, spec.center_y + spec.length / 2
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _vehicle_poly(x: float, y: float, heading: float, *,
                  margin_mm: float = 0.0) -> list[tuple[float, float]]:
    """Return an oriented vehicle polygon, optionally expanded for uncertainty."""
    if margin_mm <= 0.0:
        corners = _car_footprint(x, y, heading)
    else:
        r = math.radians(heading)
        c, s = math.cos(r), math.sin(r)
        half_l = CAR_LENGTH_MM / 2.0 + margin_mm
        half_w = CAR_WIDTH_MM / 2.0 + margin_mm
        corners = [
            (x + dl * half_l * c - dw * half_w * s,
             y + dl * half_l * s + dw * half_w * c)
            for dl, dw in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
    return [corners[i] for i in (0, 1, 3, 2)]


def _wrap_lerp(a: float, b: float, t: float) -> float:
    delta = (b - a + 180.0) % 360.0 - 180.0
    return (a + delta * t) % 360.0


def validate_trajectory(
    waypoints: Iterable[Any], *,
    start_pose: tuple[float, float, float],
    lot_size_mm: tuple[float, float] = (1200.0, 1200.0),
    target_slot: str | None = None,
    occupied_slots: Iterable[str] = (),
    obstacle_poses: Iterable[tuple[float, float, float]] = (),
    sample_step_mm: float = 25.0,
    min_turn_radius_mm: float = MIN_TURN_RADIUS_MM,
    max_segment_mm: float = 800.0,
    max_route_length_mm: float = 6000.0,
    initial_boundary_tolerance_mm: float = 20.0,
    obstacle_margin_mm: float = 0.0,
    stop_distance_mm: float = 0.0,
    require_reachable_first_wp: bool = False,
) -> TrajectorySafetyResult:
    """Reject unsafe geometry before it reaches a runner or mission.

    Waypoint planners already discretize arcs. This validator samples every chord
    at <=25 mm, applies the physical 250x150 mm footprint, and validates route
    structure, curvature, map/slot/obstacle collision, jumps and total length.

    ``stop_distance_mm`` (default 0.0 = off, so every existing caller is
    unchanged) additionally checks the **direction-change** waypoints against the
    distance the vehicle actually needs to stop. A waypoint where the motion
    direction flips is a point the car must physically halt at; its planned
    centre being inside the map says nothing about where the body ends up.

    실측 run_20260904_182908 route 13 (A2 후면주차 candidate):

        wp1 APPROACH (962.2,799.1)  footprint 여유  93.5mm
        wp2 ALIGN   (1050.0,842.8)  footprint 여유   4.2mm   <- 전진→후진 전환점
        wp8 FINAL    (650.0,150.0)  footprint 여유  25.0mm   (슬롯 안, 정상)

    validate_trajectory 는 이 경로를 safe=True 로 통과시켰다(min_clearance
    4.2mm, overflow 0). 실차는 wp2 로 |steering| 1.0 포화 + throttle 0.25 로
    접근했고, 예측 경계 감시가 overflow -29.0mm(맵 안)에서 정확히 zero 를
    걸었는데도 관성으로 135mm 더 밀려 **맵 밖 79.8mm** 까지 나갔다.

    전환점을 정지 여유만큼 진행 방향으로 밀어 본 자세로 판정하면 갈린다:

        022217 (known-good)  여유  50.2 -> +80mm 후  +10.2mm  통과
        183055 (PARKED)      여유  86.4 -> +80mm 후  +29.8mm  통과
        183503 (PARKED)      여유 288.6 -> +80mm 후 +248.6mm  통과
        182908 (맵 이탈)     여유   4.2 -> +80mm 후  -65.0mm  거부

    여유값은 새 상수가 아니다 — waypoint 자신의 position_tolerance_cm 에
    호출부가 넘긴 stop_distance_mm(ControllerConfig.stop_distance_cm)를 더한
    것으로, brake_radius_cm 와 같은 유도식이다. 마지막 waypoint(FINAL)에는
    적용하지 않는다: 거기서 멈추는 것이 목적이고 슬롯 뒤가 맵 경계인 것은
    후면주차의 설계 자체다. 런타임 경계 임계값은 하나도 바뀌지 않는다.
    """
    wps = list(waypoints)
    if not wps:
        return TrajectorySafetyResult(False, "EMPTY_ROUTE")
    if sample_step_mm <= 0.0:
        return TrajectorySafetyResult(False, "INVALID_SAMPLE_STEP")

    # ``require_reachable_first_wp`` (기본 off) 는 **차가 처음 향할 점**이
    # 전진으로 가까워질 수 있는 점인지 본다.
    #
    # 판정은 along-track 하나다: 첫 waypoint 가 차 뒤에 있으면(along < 0)
    # 전진은 거리를 **늘린다**. 이미 도착 반경 안이면 움직일 필요가 없으므로
    # 검사하지 않는다. 반경은 그 waypoint 자신의 position_tolerance_cm 다 —
    # 새 상수가 아니다.
    #
    # 실측 (staging 종료 직후 만들어진 인계 경로, 세 run 모두 waypoint 1개):
    #
    #     230944 (138.8,531.4,356.3deg) along +281.2  -> 통과. 실제로 완주했다
    #                                   (HANDOFF_CAPTURED, 57.6mm 초과 도착)
    #     231157 (247.5,639.6, 86.9deg) along  -29.9  -> 거부
    #     231338 (243.5,623.4, 88.8deg) along  -19.6  -> 거부
    #
    # 뒤 두 경우는 차가 통로축과 거의 수직(87~89deg)인데 목표가 오른쪽 뒤에
    # 있어, 적재 1.5초 만에 PATH_DEVIATION 으로 끝났다. plan_handoff 는 세 경우
    # 모두 feasible=True 를 줬다 — planner 와 controller 가 서로 다른 사실을
    # 보고 있었다.
    #
    # "waypoint 1개"를 금지하는 것이 아니다. 도달 가능한 1개짜리 경로
    # (230944, 그리고 통로에 정렬된 짧은 인계)는 그대로 통과한다.
    if require_reachable_first_wp and len(start_pose) >= 3:
        first = wps[0]
        if str(_get(first, "motion_direction",
                    default="FORWARD")).upper() == "FORWARD":
            fx, fy = _xy(first)
            dx, dy = fx - start_pose[0], fy - start_pose[1]
            angle = math.radians(start_pose[2])
            along = dx * math.cos(angle) + dy * math.sin(angle)
            arrival_mm = 10.0 * float(
                _get(first, "position_tolerance_cm", default=0.0) or 0.0)
            if along < 0.0 and math.hypot(dx, dy) > arrival_mm:
                return TrajectorySafetyResult(False, "FIRST_WP_BEHIND")

    route_ids = {_get(wp, "route_id") for wp in wps}
    if len(route_ids) != 1:
        return TrajectorySafetyResult(False, "MIXED_ROUTE_ID")
    wp_ids = [_get(wp, "waypoint_id") for wp in wps]
    if any(not isinstance(i, int) for i in wp_ids) or wp_ids != sorted(set(wp_ids)):
        return TrajectorySafetyResult(False, "INVALID_WAYPOINT_ORDER")

    reverse_phases = {"RECOVERY", "PARKING", "APPROACH", "ALIGN", "ENTRY", "FINAL"}
    for wp in wps:
        values = (*_xy(wp), float(_get(wp, "curvature", default=0.0) or 0.0))
        if not all(math.isfinite(v) for v in values):
            return TrajectorySafetyResult(False, "NONFINITE_WAYPOINT")
        direction = str(_get(wp, "motion_direction", default="FORWARD")).upper()
        phase = str(_get(wp, "phase", default="") or "").upper()
        if direction not in {"FORWARD", "REVERSE"}:
            return TrajectorySafetyResult(False, "INVALID_DIRECTION")
        if direction == "REVERSE" and phase not in reverse_phases:
            return TrajectorySafetyResult(False, f"REVERSE_PHASE_{phase}")
        curvature = abs(float(_get(wp, "curvature", default=0.0) or 0.0))
        if curvature > 1.0 / min_turn_radius_mm + 1e-9:
            return TrajectorySafetyResult(False, "CURVATURE_LIMIT")

    specs = default_slot_specs()
    # Empty painted bays are traversable map geometry; an occupied non-target bay
    # is a physical keepout even when its vehicle is temporarily not detected.
    blocked_slots = set(occupied_slots)
    blocked_slots.discard(target_slot)
    slot_polys = {sid: _slot_poly(specs[sid]) for sid in blocked_slots if sid in specs}
    obstacle_polys = [_vehicle_poly(*pose, margin_mm=obstacle_margin_mm)
                      for pose in obstacle_poses]
    start_fp = _vehicle_poly(*start_pose)
    initial_slot_overlap = {sid for sid, poly in slot_polys.items()
                            if _overlap(start_fp, poly)}
    initial_slot_distance = {
        sid: math.hypot(start_pose[0] - specs[sid].center_x,
                        start_pose[1] - specs[sid].center_y)
        for sid in initial_slot_overlap}

    width, height = lot_size_mm
    total = 0.0
    count = 0
    min_clearance = float("inf")
    previous = start_pose
    for wp in wps:
        tx, ty = _xy(wp)
        direction = str(_get(wp, "motion_direction", default="FORWARD")).upper()
        dx, dy = tx - previous[0], ty - previous[1]
        distance = math.hypot(dx, dy)
        if distance > max_segment_mm:
            return TrajectorySafetyResult(False, "SEGMENT_JUMP", count, total,
                                          min_clearance)
        total += distance
        if total > max_route_length_mm:
            return TrajectorySafetyResult(False, "ROUTE_TOO_LONG", count, total,
                                          min_clearance)
        tangent = previous[2] if distance < 1e-6 else math.degrees(math.atan2(dy, dx))
        if direction == "REVERSE":
            tangent = (tangent + 180.0) % 360.0
        target_heading = _get(wp, "target_heading_deg")
        end_heading = tangent if target_heading is None else float(target_heading)
        steps = max(1, math.ceil(distance / sample_step_mm))
        for index in range(steps + 1):
            t = index / steps
            x = previous[0] + dx * t
            y = previous[1] + dy * t
            heading = _wrap_lerp(previous[2], end_heading, t)
            fp = _vehicle_poly(x, y, heading)
            overflow = max(max(-px, px - width, -py, py - height)
                           for px, py in fp)
            clearance = min(min(px, width - px, py, height - py)
                            for px, py in fp)
            min_clearance = min(min_clearance, clearance)
            # 이미 맵 밖에 걸친 차는 **빠져나올 수 있어야 한다**. 허용치는
            # "요구 여유를 달성" 이 아니라 "지금보다 나빠지지 않기" 다.
            #
            # 예전에는 min() 으로 initial_boundary_tolerance_mm(20mm) 에
            # 가둬서, 차가 그보다 많이 나가 있으면 **자기 출발 자세조차**
            # MAP_FOOTPRINT 로 거절됐다. 그러면 어떤 탈출 경로도 실을 수 없다.
            # 실측 run_20260901_154551: 후면주차 FINAL 에서 56.8mm 밖으로
            # 나간 뒤, planner 는 여유를 전혀 악화시키지 않는 5-waypoint 탈출
            # 경로를 만들었는데 validator 가 세 번 모두 거절해
            # WAIT_RECOVERY_EXHAUSTED 로 굳었다 (RECOVERY_REJECTED x3).
            #
            # 규칙은 하나다: **출발보다 나빠지지 않기**.
            #   - 맵 안에서 출발 (initial_overflow=0) -> 허용치 0.
            #     경로는 맵을 한 톨도 벗어날 수 없다. 예전과 완전히 같다.
            #   - 맵 밖에서 출발 -> 그 지점까지만 허용. 더 깊어지면 거절.
            # plan_setup_recovery.blocked() 와 바로 아래 슬롯 keepout 이
            # 이미 쓰는 규칙과 같다. 어느 경우에도 맵이 넓어지지 않는다.
            initial_overflow = max(max(-px, px - width, -py, py - height)
                                   for px, py in start_fp)
            allowed = max(0.0, initial_overflow)
            if overflow > allowed + 1e-6:
                return TrajectorySafetyResult(False, "MAP_FOOTPRINT", count,
                                              total, min_clearance)
            for sid, poly in slot_polys.items():
                if not _overlap(fp, poly):
                    continue
                if sid in initial_slot_overlap:
                    now_d = math.hypot(x - specs[sid].center_x,
                                       y - specs[sid].center_y)
                    if now_d + 1e-6 >= initial_slot_distance[sid]:
                        continue
                return TrajectorySafetyResult(False, f"SLOT_FOOTPRINT_{sid}",
                                              count, total, min_clearance)
            if any(_overlap(fp, obstacle) for obstacle in obstacle_polys):
                return TrajectorySafetyResult(False, "OBSTACLE_FOOTPRINT", count,
                                              total, min_clearance)
            count += 1
        # 전/후진이 바뀌는 waypoint 는 차가 **실제로 멈춰야 하는** 지점이다.
        # 그 지점을 정지 여유만큼 진행 방향으로 민 자세도 맵 안이어야 한다.
        #
        # RECOVERY 는 제외한다. 이 검사의 근거는 **주행 속도로 접근하는 주차
        # 경로의 전환점**(182908 route 13 의 ALIGN: |steering| 1.0 포화 +
        # throttle 0.25)이다. RECOVERY/staging 기동은 5cm/s 로 도는 저속
        # 재배치이고, 입구 코너처럼 좁은 곳에서 도는 것이 그 목적이라
        # 같은 여유를 걸면 입구 staging 해가 전부 사라진다. 그쪽은 이미
        # min_clearance_mm(entry_staging_min_clearance_mm) 이라는 자기
        # hard constraint 를 갖고 있다. 증거가 있는 곳에만 건다.
        index_of = wps.index(wp)
        phase_here = str(_get(wp, "phase", default="") or "").upper()
        flips = (index_of + 1 < len(wps)
                 and str(_get(wps[index_of + 1], "motion_direction",
                              default="FORWARD")).upper() != direction)
        if stop_distance_mm > 0.0 and flips and phase_here != "RECOVERY":
            tolerance_mm = 10.0 * float(
                _get(wp, "position_tolerance_cm", default=0.0) or 0.0)
            allowance = tolerance_mm + stop_distance_mm
            sign = -1.0 if direction == "REVERSE" else 1.0
            rad = math.radians(end_heading)
            sx = tx + sign * allowance * math.cos(rad)
            sy = ty + sign * allowance * math.sin(rad)
            stop_fp = _vehicle_poly(sx, sy, end_heading)
            stop_overflow = max(max(-px, px - width, -py, py - height)
                                for px, py in stop_fp)
            initial_overflow = max(max(-px, px - width, -py, py - height)
                                   for px, py in start_fp)
            if stop_overflow > max(0.0, initial_overflow) + 1e-6:
                return TrajectorySafetyResult(False, "STOP_FOOTPRINT", count,
                                              total, min_clearance)
        previous = (tx, ty, end_heading)
    return TrajectorySafetyResult(True, sampled_poses=count,
                                  path_length_mm=total,
                                  min_clearance_mm=min_clearance)
