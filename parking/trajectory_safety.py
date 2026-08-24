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


def _vehicle_poly(x: float, y: float, heading: float) -> list[tuple[float, float]]:
    corners = _car_footprint(x, y, heading)
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
) -> TrajectorySafetyResult:
    """Reject unsafe geometry before it reaches a runner or mission.

    Waypoint planners already discretize arcs. This validator samples every chord
    at <=25 mm, applies the physical 250x150 mm footprint, and validates route
    structure, curvature, map/slot/obstacle collision, jumps and total length.
    """
    wps = list(waypoints)
    if not wps:
        return TrajectorySafetyResult(False, "EMPTY_ROUTE")
    if sample_step_mm <= 0.0:
        return TrajectorySafetyResult(False, "INVALID_SAMPLE_STEP")

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
    obstacle_polys = [_vehicle_poly(*pose) for pose in obstacle_poses]
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
            # A measured initial pose may be <=20 mm outside. Only an escape that
            # never exceeds that initial uncertainty is accepted.
            initial_overflow = max(max(-px, px - width, -py, py - height)
                                   for px, py in start_fp)
            allowed = min(initial_boundary_tolerance_mm,
                          max(0.0, initial_overflow))
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
        previous = (tx, ty, end_heading)
    return TrajectorySafetyResult(True, sampled_poses=count,
                                  path_length_mm=total,
                                  min_clearance_mm=min_clearance)
