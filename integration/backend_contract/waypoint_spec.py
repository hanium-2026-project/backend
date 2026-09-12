"""SpecWaypoint — 실제 parking.waypoints.Waypoint 계약 재현(x/y = mm).

⚠ 실제 backend `parking/waypoints.py` 가 아니다. reference(waypoint_schema_reference.py)에서
확인된 계약: .x/.y 는 mm, wire 변환(cm)은 to_wire() 에서만. 여기서는 host 제어에 필요한
필드만 재현한다. 실제 backend 가 있으면 실제 Waypoint 를 쓰고 waypoints_from_backend() 로 매핑.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SpecWaypoint:
    x: float               # mm
    y: float               # mm
    target_heading_deg: Optional[float] = None
    speed_cm_s: float = 12.0
    position_tolerance_cm: float = 8.0
    heading_tolerance_deg: float = 30.0
    heading_required: bool = False
    is_final: bool = False
    route_id: int = 1
    waypoint_id: int = 1
    phase: str = "CRUISE"
