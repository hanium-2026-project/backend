"""슬롯 정보 기반 phase별 waypoint 생성기 (회의 스펙 1차).

CRUISE → APPROACH → ALIGN → ENTRY → FINAL 순서의 waypoint 목록을
슬롯 중심·방향·크기 템플릿으로부터 생성한다 (슬롯별 하드코딩 금지).

좌표 단위: 현재 백엔드 내부 표준인 mm (rl.parking_env / parking.services와 동일).
TCP 송신층에서 cm 변환하여 내보낸다 (회의 좌표계 스펙은 cm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

from rl.parking_env import NODE_COORDINATES, SLOT_COORDINATES, SLOT_ROUTES

# ─── 스펙 데이터 구조 ─────────────────────────────────────────────────────────

PHASES = ("CRUISE", "APPROACH", "ALIGN", "ENTRY", "FINAL")

# phase별 기본 속도(cm/s → 내부 mm/s)와 허용오차
PHASE_DEFAULTS: dict[str, dict[str, float]] = {
    "CRUISE":   {"speed_cm_s": 12.0, "position_tolerance_cm": 8.0, "heading_tolerance_deg": 30.0},
    "APPROACH": {"speed_cm_s":  8.0, "position_tolerance_cm": 6.0, "heading_tolerance_deg": 20.0},
    "ALIGN":    {"speed_cm_s":  5.0, "position_tolerance_cm": 5.0, "heading_tolerance_deg": 12.0},
    "ENTRY":    {"speed_cm_s":  5.0, "position_tolerance_cm": 4.0, "heading_tolerance_deg": 12.0},
    "FINAL":    {"speed_cm_s":  4.0, "position_tolerance_cm": 5.0, "heading_tolerance_deg": 12.0},
}


@dataclass(frozen=True)
class SlotSpec:
    """주차 슬롯 정의 (회의 6번 스키마)."""

    slot_id: str
    center_x: float                # mm
    center_y: float                # mm
    target_heading_deg: float      # 주차 완료 시 차량 방향
    width: float = 200.0           # mm (슬롯 폭 — 바닥판 실측 200mm)
    length: float = 300.0          # mm (슬롯 깊이 — 바닥판 실측 300mm)
    entry_side: str = "BOTTOM"     # 진입 방향: BOTTOM(아래에서 위로) / TOP


@dataclass(frozen=True)
class Waypoint:
    """경로 생성기 출력 waypoint (회의 7번 스키마, 내부 mm)."""

    route_id: int
    waypoint_id: int
    phase: str
    x: float                        # mm
    y: float                        # mm
    target_heading_deg: float | None
    speed_cm_s: float
    position_tolerance_cm: float
    heading_tolerance_deg: float
    heading_required: bool
    is_final: bool

    def to_wire(self) -> dict[str, Any]:
        """TCP 전송용 dict — 좌표를 회의 스펙 단위(cm)로 변환."""
        d = asdict(self)
        d["x_cm"] = round(self.x / 10.0, 1)
        d["y_cm"] = round(self.y / 10.0, 1)
        del d["x"], d["y"]
        return d


# ─── 슬롯 템플릿 (rl 좌표 기반 자동 생성) ────────────────────────────────────

def default_slot_specs() -> dict[str, SlotSpec]:
    """rl.parking_env의 슬롯 좌표로부터 SlotSpec을 생성한다.

    A행(y=1050, 위쪽)은 아래(중앙차로)에서 진입 → entry_side=BOTTOM, 주차 방향 90°(위).
    B행(y=150, 아래쪽)은 위에서 진입 → entry_side=TOP, 주차 방향 270°(아래).
    """
    specs: dict[str, SlotSpec] = {}
    for name, (x, y) in SLOT_COORDINATES.items():
        if name.startswith("A"):
            specs[name] = SlotSpec(name, x, y, target_heading_deg=90.0, entry_side="BOTTOM")
        else:
            specs[name] = SlotSpec(name, x, y, target_heading_deg=270.0, entry_side="TOP")
    return specs


# ─── waypoint 생성 ───────────────────────────────────────────────────────────

def _make(route_id: int, wp_id: int, phase: str, x: float, y: float,
          heading: float | None, *, is_final: bool = False) -> Waypoint:
    p = PHASE_DEFAULTS[phase]
    return Waypoint(
        route_id=route_id, waypoint_id=wp_id, phase=phase,
        x=x, y=y, target_heading_deg=heading,
        speed_cm_s=p["speed_cm_s"],
        position_tolerance_cm=p["position_tolerance_cm"],
        heading_tolerance_deg=p["heading_tolerance_deg"],
        heading_required=heading is not None,
        is_final=is_final,
    )


def build_waypoints(
    slot: SlotSpec,
    route_id: int,
    route_nodes: list[str] | None = None,
) -> list[Waypoint]:
    """슬롯 스펙 → CRUISE~FINAL waypoint 목록 생성.

    Args:
        slot: 대상 슬롯 스펙.
        route_id: 이 경로의 식별자 (재생성 시 증가 — 회의 8번).
        route_nodes: 통로 주행 노드 목록 (기본: SLOT_ROUTES[slot_id]).
                     마지막 노드는 슬롯 앞(front)으로 간주한다.
    """
    nodes = route_nodes if route_nodes is not None else list(SLOT_ROUTES[slot.slot_id])
    if not nodes:
        raise ValueError("route_nodes must not be empty")

    # 진입 방향 단위 벡터 (BOTTOM: 아래→위 = +y, TOP: 위→아래 = -y)
    dir_y = 1.0 if slot.entry_side == "BOTTOM" else -1.0
    heading = slot.target_heading_deg

    # 슬롯 기하 기반 기준점 (x는 슬롯 중심에 정렬)
    entry_y = slot.center_y - dir_y * (slot.length / 2)          # 슬롯 입구선
    align_y = entry_y - dir_y * (slot.length * 0.5)              # 입구 앞 정렬 지점
    # 접근 지점 — 중앙차로(마지막 CRUISE)와 ALIGN 사이에 위치하도록 입구에서 0.9L 후방
    approach_y = entry_y - dir_y * (slot.length * 0.9)

    wps: list[Waypoint] = []
    wp_id = 0

    # CRUISE — 통로 노드들 (마지막 front 노드는 APPROACH 이후 단계가 대체)
    for node in nodes[:-1]:
        nx, ny = NODE_COORDINATES[node]
        wps.append(_make(route_id, wp_id, "CRUISE", nx, ny, None))
        wp_id += 1

    # APPROACH — 슬롯 x에 정렬된 접근점 (감속 시작)
    wps.append(_make(route_id, wp_id, "APPROACH", slot.center_x, approach_y, None)); wp_id += 1
    # ALIGN — 슬롯 방향으로 정렬 (target heading 필수)
    wps.append(_make(route_id, wp_id, "ALIGN", slot.center_x, align_y, heading)); wp_id += 1
    # ENTRY — 슬롯 입구선 통과 (heading 유지)
    wps.append(_make(route_id, wp_id, "ENTRY", slot.center_x, entry_y, heading)); wp_id += 1
    # FINAL — 슬롯 중심 도달 + 방향 일치 (둘 다 만족해야 완료)
    wps.append(_make(route_id, wp_id, "FINAL", slot.center_x, slot.center_y, heading, is_final=True)); wp_id += 1

    return wps


def is_waypoint_reached(
    waypoint: Waypoint,
    position: tuple[float, float],
    heading_deg: float | None,
) -> bool:
    """waypoint 도착 판정 (회의 7번 FINAL 조건 포함).

    position은 mm. heading_required=True인데 heading_deg가 None이면 미도착.
    """
    dist_mm = math.hypot(position[0] - waypoint.x, position[1] - waypoint.y)
    if dist_mm > waypoint.position_tolerance_cm * 10.0:
        return False
    if not waypoint.heading_required:
        return True
    if heading_deg is None:
        return False
    err = abs((heading_deg - (waypoint.target_heading_deg or 0.0) + 180.0) % 360.0 - 180.0)
    return err <= waypoint.heading_tolerance_deg
