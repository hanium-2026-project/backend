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

# 펌웨어(protocol.c::parse_phase)가 아는 phase. 이 밖의 이름은 wire 에서
# INVALID 로 거절된다. RECOVERY 는 host 내부(AUTO_HOST)에서만 쓰며 차량에
# 보내지 않는다 — 후진 복구는 DIRECT_CONTROL 로만 실행된다.
WIRE_PHASES = ("CRUISE", "APPROACH", "ALIGN", "ENTRY", "FINAL")
PHASES = WIRE_PHASES + ("RECOVERY",)

# ─── 차량 기하 ───────────────────────────────────────────────────────────────
# 2026-08-12 실측 (최대 조향 확인됨). 천장캠 영상 236프레임을 추적해
# 100° 호에 원을 맞췄다: R=610mm, 잔차 평균 8mm. v/ω 로도 635mm 로 일치.
#   출발 (128,154) 방향 96°  →  끝 (792,790) 방향 356°
# 2026-08-07 의 570mm 는 과소평가였다. 서보 운용각을 ±36→±40° 로 넓혀도
# 줄지 않았으므로 링키지 한계로 본다.
MIN_TURN_RADIUS_MM: float = 610.0
# 이전 측정값 (기록 보존용)
MEASURED_TURN_RADIUS_MM_20260807: float = 570.0
# 벽에서 확보할 최소 여유 = 차체 반폭 + 위치 오차.
# 실차 도착 정확도가 9~11cm 였으므로 100mm 미만은 벽 접촉을 각오해야 한다.
WALL_CLEARANCE_MM: float = 100.0
# 맵 크기 (rl.parking_env 와 동일)
LOT_SIZE_MM: float = 1200.0

# ─── 발렛 인계 모델 ──────────────────────────────────────────────────────────
# 슬롯 안으로 넣는 주차 기동은 **하드웨어팀 주차 공식**이 담당한다.
# backend 는 차를 "슬롯 앞에 통로 방향(가로)으로" 세워 인계하는 데까지만 책임진다.
# 그래서 경로에 슬롯 축으로 꺾어 들어가는 90° 선회가 없다 — 통로 직진뿐이다.
#
# 중앙 통로. A행 진입선(y=300)과 B행 진입선(y=900) 의 정중앙이라 양쪽 슬롯에서
# 같은 300mm 를 확보한다. 이 거리가 HW 주차 기동의 작업 공간이다.
AISLE_Y: float = 600.0
# 인계 지점을 슬롯 중심 x 에서 진행 방향으로 얼마나 지나칠지.
# 후진 주차 공식은 보통 슬롯을 조금 지나친 위치에서 시작한다.
# **HW 주차 공식 스펙을 받으면 이 값을 맞춰야 한다** (기본 0 = 슬롯 정면).
HANDOFF_OFFSET_MM: float = 0.0
# 감속·정밀 포착을 시작할 지점 (인계 지점 앞쪽 거리)
HANDOFF_LEAD_MM: float = 250.0
# 통로 위로 본다고 판정할 y 오차 (이 안이면 합류 구간도 필요 없다)
ON_AISLE_TOLERANCE_MM: float = 80.0
# 통로와 나란히 서 있다고 볼 heading 오차. 이 안이면 90° 진입 우회전이 아니라
# 완만한 S자 합류로 붙는다 — 몇 cm 벗어난 걸 90° 선회로 처리하면 반경이
# 모자라 경로가 통째로 거부된다.
ALONG_AISLE_HEADING_TOLERANCE_DEG: float = 50.0
# 통로 직선을 쪼개는 간격. 길게 두면 중간 진행 판정이 안 된다.
AISLE_SEGMENT_MM: float = 300.0
# 진입 우회전 원호를 몇 개 waypoint 로 쪼갤지. 5등분이면 현 오차 <8mm.
ENTRY_ARC_SEGMENTS: int = 5

# phase별 기본 속도(cm/s → 내부 mm/s)와 허용오차
PHASE_DEFAULTS: dict[str, dict[str, float]] = {
    "CRUISE":   {"speed_cm_s": 12.0, "position_tolerance_cm": 8.0, "heading_tolerance_deg": 30.0},
    "APPROACH": {"speed_cm_s":  8.0, "position_tolerance_cm": 6.0, "heading_tolerance_deg": 20.0},
    "ALIGN":    {"speed_cm_s":  5.0, "position_tolerance_cm": 5.0, "heading_tolerance_deg": 12.0},
    "ENTRY":    {"speed_cm_s":  5.0, "position_tolerance_cm": 4.0, "heading_tolerance_deg": 12.0},
    "FINAL":    {"speed_cm_s":  4.0, "position_tolerance_cm": 5.0, "heading_tolerance_deg": 12.0},
    # RECOVERY — 후진 복구. 실패 지점에서 물러나는 것뿐이라 정밀도를 요구하지 않는다.
    "RECOVERY": {"speed_cm_s":  5.0, "position_tolerance_cm": 8.0, "heading_tolerance_deg": 30.0},
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
    # APPROACH 1차 capture 반경(cm). None 이면 host 가
    # ControllerConfig.approach_capture_tolerance_cm(10cm) 로 폴백하고
    # position_tolerance_cm 이 2차 정밀 완료 반경이 된다.
    capture_tolerance_cm: float | None = None
    # "FORWARD" | "REVERSE". 후진은 복구 경로에서만 쓴다 — 제어기가
    # ControllerConfig.reverse_allowed_phases 로 한 번 더 막는다.
    motion_direction: str = "FORWARD"

    def to_wire(self) -> dict[str, Any]:
        """TCP 전송용 dict — 좌표는 cm 로 변환.

        펌웨어는 target_heading_deg 를 필수 double(0~359.999)로 읽으므로
        방향 무관 waypoint(heading_required=False)도 null 대신 0.0 을 보낸다.
        내부 표현은 None 을 유지해 "방향 무관"을 구분한다.
        """
        d = asdict(self)
        d["x_cm"] = round(self.x / 10.0, 1)
        d["y_cm"] = round(self.y / 10.0, 1)
        d["target_heading_deg"] = (
            round(self.target_heading_deg % 360.0, 3)
            if self.target_heading_deg is not None else 0.0
        )
        d.setdefault("arrival_mode", "STOP")
        del d["x"], d["y"]
        return d


# ─── 슬롯 템플릿 (rl 좌표 기반 자동 생성) ────────────────────────────────────

def default_slot_specs() -> dict[str, SlotSpec]:
    """rl.parking_env의 슬롯 좌표로부터 SlotSpec을 생성한다.

    A행(y=150, 아래쪽/입구방향)은 위(중앙차로)에서 진입 → entry_side=TOP, 주차 방향 270°(아래).
    B행(y=1050, 위쪽/출구방향)은 아래(중앙차로)에서 진입 → entry_side=BOTTOM, 주차 방향 90°(위).
    """
    specs: dict[str, SlotSpec] = {}
    for name, (x, y) in SLOT_COORDINATES.items():
        if name.startswith("A"):
            specs[name] = SlotSpec(name, x, y, target_heading_deg=270.0, entry_side="TOP")
        else:
            specs[name] = SlotSpec(name, x, y, target_heading_deg=90.0, entry_side="BOTTOM")
    return specs


# ─── waypoint 생성 ───────────────────────────────────────────────────────────

def _make(route_id: int, wp_id: int, phase: str, x: float, y: float,
          heading: float | None, *, is_final: bool = False,
          capture_tolerance_cm: float | None = None,
          motion_direction: str = "FORWARD") -> Waypoint:
    p = PHASE_DEFAULTS[phase]
    return Waypoint(
        route_id=route_id, waypoint_id=wp_id, phase=phase,
        x=x, y=y, target_heading_deg=heading,
        speed_cm_s=p["speed_cm_s"],
        position_tolerance_cm=p["position_tolerance_cm"],
        heading_tolerance_deg=p["heading_tolerance_deg"],
        heading_required=heading is not None,
        is_final=is_final,
        capture_tolerance_cm=capture_tolerance_cm,
        motion_direction=motion_direction,
    )


# ─── 인계 지점 계획 ──────────────────────────────────────────────────────────

def merge_run_mm(offset_mm: float, radius_mm: float = MIN_TURN_RADIUS_MM) -> float:
    """통로에서 offset 만큼 벗어난 차가 나란히 붙는 데 필요한 진행 거리.

    반경 R 원호 두 개(S자)로 횡방향 d 를 닫으면 진행 거리는
    ``2*sqrt(R*d - d^2/4)`` 다. 90° 선회와 달리 **횡방향으로 R 만큼 쓰지
    않으므로**, 통로에서 몇 cm 벗어난 정도는 이 방법으로 붙는다.
    """
    d = min(abs(offset_mm), 2.0 * radius_mm)
    return 2.0 * math.sqrt(max(0.0, radius_mm * d - d * d / 4.0))


@dataclass(frozen=True)
class EntryTurn:
    """출발점 → 통로 합류까지의 진입 우회전 90° 원호.

    차량은 출발점에서 **+y 방향**으로 서 있다가 우회전(시계방향)해서 통로에
    +x 방향으로 합류한다. 반경은 고를 수 있는 값이 아니라 **출발 위치에서
    결정된다** — 통로까지의 y 거리가 곧 반경이고, 같은 거리만큼 x 로 나아간다.

        R = aisle_y - start_y,   합류점 = (start_x + R, aisle_y)

    출발점 (150,100), 통로 600 이면 R=500, 합류점 (650,600) = A2 앞이다.
    """

    start: tuple[float, float]
    center: tuple[float, float]
    join_point: tuple[float, float]         # 통로 합류점
    radius_mm: float
    feasible: bool
    reason: str = ""


def plan_entry_turn(start: tuple[float, float], *, aisle_y: float = AISLE_Y,
                    min_radius_mm: float = MIN_TURN_RADIUS_MM) -> EntryTurn:
    """출발점에서 통로로 붙는 우회전 원호를 계산한다.

    반경이 최소 선회 반경보다 작으면 못 돈다 — 차를 통로에서 더 멀리(아래로)
    놓아야 한다. 반대로 너무 멀면 합류점이 오른쪽으로 밀려 왼쪽 슬롯을
    지나쳐 버린다.
    """
    r = aisle_y - start[1]
    join = (start[0] + r, aisle_y)
    center = (start[0] + r, start[1])       # 우회전 중심은 진행방향 오른쪽
    ok, reason = True, ""
    if r < min_radius_mm:
        ok = False
        reason = (f"통로까지 {r:.0f}mm 뿐이라 최소 선회 반경 "
                  f"{min_radius_mm:.0f}mm 로 못 돈다 (차를 더 아래에 놓을 것)")
    elif not _in_bounds(*join):
        ok = False
        reason = f"통로 합류점 {join[0]:.0f},{join[1]:.0f} 이 맵 밖"
    return EntryTurn(start=start, center=center, join_point=join,
                     radius_mm=r, feasible=ok, reason=reason)


def _entry_arc_points(turn: EntryTurn, segments: int = ENTRY_ARC_SEGMENTS
                      ) -> list[tuple[float, float]]:
    """진입 원호를 (x, y) 목록으로 쪼갠다. 시작점 제외, 합류점 포함."""
    cx, cy = turn.center
    a0 = math.atan2(turn.start[1] - cy, turn.start[0] - cx)   # = 180°
    # 우회전(시계방향) = 각도가 줄어든다. 180° → 90° 로 쓸어 합류점에 닿는다.
    sweep = -math.radians(90.0)
    out = []
    for i in range(1, segments + 1):
        a = a0 + sweep * (i / segments)
        out.append((cx + turn.radius_mm * math.cos(a),
                    cy + turn.radius_mm * math.sin(a)))
    return out


@dataclass(frozen=True)
class HandoffPlan:
    """슬롯 앞 인계 지점 한 개.

    차량은 중앙 통로를 `approach_heading_deg` 방향으로 달려와 `point` 에
    통로와 나란히(가로로) 선다. 여기서부터 슬롯 안으로 넣는 기동은
    하드웨어팀 주차 공식이 이어받는다.
    """

    slot_id: str
    point: tuple[float, float]              # 인계 지점 (mm)
    heading_deg: float                      # 인계 시 차량 방향 = 통로 방향
    lead_point: tuple[float, float]         # 감속·정밀 포착 시작 지점
    aisle_y: float
    approach_heading_deg: float             # 0 = +x, 180 = -x
    slot_entry_y: float                     # 슬롯 진입선 (HW 기동 시작선)
    gap_to_slot_mm: float                   # 인계 지점 ↔ 진입선 거리
    lead_in_turn_required: bool             # 90° 진입 우회전이 필요한가
    entry_turn: EntryTurn | None            # 진입 우회전 (통로와 수직으로 출발)
    merge_point: tuple[float, float] | None  # S자 합류점 (통로와 나란히 출발)
    feasible: bool
    reason: str = ""


def _in_bounds(x: float, y: float, clearance: float = WALL_CLEARANCE_MM) -> bool:
    return clearance <= x <= LOT_SIZE_MM - clearance and \
           clearance <= y <= LOT_SIZE_MM - clearance


def plan_handoff(slot: SlotSpec, *, from_pose: tuple[float, float] | None = None,
                 from_heading_deg: float | None = None,
                 aisle_y: float = AISLE_Y,
                 offset_mm: float = HANDOFF_OFFSET_MM,
                 lead_mm: float = HANDOFF_LEAD_MM,
                 min_radius_mm: float = MIN_TURN_RADIUS_MM) -> HandoffPlan:
    """슬롯 앞 인계 지점과, 거기까지 가는 진입 우회전을 계산한다.

    두 구간뿐이다::

        출발점 → (우회전 90° 원호) → 통로 합류 → (직진) → 슬롯 앞 인계

    슬롯 안으로 꺾어 들어가는 선회는 없다 — 그건 HW 주차 공식 담당이다.

    통로 밖(아래)에서 출발하면 진입 우회전 반경이 **출발 y 로 결정된다**.
    합류점이 목표 슬롯보다 오른쪽이면 그 슬롯은 지나쳐 버린 것이고, 최소
    선회 반경 때문에 전진으로는 되돌아올 수 없다 → feasible=False.
    """
    dir_y = 1.0 if slot.entry_side == "BOTTOM" else -1.0
    entry_y = slot.center_y - dir_y * (slot.length / 2)     # 슬롯 진입선

    dy = abs(from_pose[1] - aisle_y) if from_pose is not None else 0.0
    on_aisle = from_pose is not None and dy <= ON_AISLE_TOLERANCE_MM

    # 통로에서 벗어나 있을 때 붙는 방법이 두 가지다.
    #
    #   나란히 서 있다 → **S자 합류**. 횡방향으로 반경만큼 쓰지 않으므로
    #                    몇 cm 벗어난 건 이걸로 붙는다.
    #   수직으로 서 있다 → **90° 진입 우회전**. 횡·종 각각 반경만큼 필요하다.
    #
    # 방향을 모르면 합류로 본다. 90° 로 잘못 보면 반경 부족으로 경로가 통째로
    # 거부되지만, 합류로 보면 제어기가 완만히 붙는다 — 실측으로 잡힌 버그다
    # (통로에서 9.5cm 벗어났을 뿐인데 전 슬롯이 거부됐다).
    along = True
    if from_heading_deg is not None:
        e0 = abs((from_heading_deg + 180.0) % 360.0 - 180.0)
        e180 = abs((from_heading_deg - 180.0 + 180.0) % 360.0 - 180.0)
        along = min(e0, e180) <= ALONG_AISLE_HEADING_TOLERANCE_DEG

    # 나란히 달리는 중이면 **가던 방향 그대로** 합류한다. 반대로 틀어
    # 붙으려면 U턴이 필요한데 선회 지름이 맵 한 변과 맞먹어 불가능하다.
    travel = 1.0
    if from_heading_deg is not None:
        travel = -1.0 if abs((from_heading_deg - 180.0 + 180.0) % 360.0 - 180.0) \
            < abs((from_heading_deg + 180.0) % 360.0 - 180.0) else 1.0

    merge: tuple[float, float] | None = None
    turn: EntryTurn | None = None
    if from_pose is not None and not on_aisle:
        if along:
            merge = (from_pose[0] + travel * merge_run_mm(dy), aisle_y)
        else:
            turn = plan_entry_turn(from_pose, aisle_y=aisle_y,
                                   min_radius_mm=min_radius_mm)

    # 접근 방향.
    # 진입 우회전을 타고 들어오면 차는 통로에 **+x 를 보고** 나온다. 그
    # 상태에서 180° 접근은 U턴을 요구하는데 선회 지름이 맵 한 변과 맞먹어
    # 불가능하다. 그러니 우회전 진입이면 방향은 0° 로 고정이고, 슬롯이
    # 합류점보다 왼쪽인지 여부는 아래 feasible 검사가 잡는다.
    #
    # 이미 통로 위라면 차량 x 로 고른다. 이때 부동소수점 오차로 방향이
    # 뒤집히지 않게 여유를 둔다 — 합류점이 슬롯 중심보다 1nm 오른쪽이라고
    # 180° 로 접근하면 차가 되돌아가야 한다(실측으로 잡힌 버그).
    if turn is not None:
        approach = 0.0
    elif merge is not None:
        # 합류 중에는 방향을 바꿀 수 없다 — 가던 방향이 곧 접근 방향이다.
        approach = 0.0 if travel > 0 else 180.0
    elif from_heading_deg is not None:
        # 이미 통로 위라도 **가던 방향**이 접근 방향이다. 위치만 보고 고르면
        # 뒤쪽 슬롯에 180° 로 접근하라는 답이 나오는데, 그건 U턴이라 불가능하다.
        approach = 0.0 if travel > 0 else 180.0
    else:
        ref_x = from_pose[0] if from_pose is not None else 0.0
        approach = 0.0 if ref_x <= slot.center_x + ON_AISLE_TOLERANCE_MM else 180.0
    c = math.cos(math.radians(approach))

    point = (slot.center_x + offset_mm * c, aisle_y)
    # 감속 지점은 인계 앞 lead_mm 이되, 차가 이미 그보다 가까우면 그만큼만
    # 물러난다. 안 그러면 감속점이 차 뒤(또는 맵 밖)에 찍혀 경로가 거부된다.
    run = abs(point[0] - from_pose[0]) if from_pose is not None else lead_mm
    lead = (point[0] - min(lead_mm, run) * c, aisle_y)

    ok = _in_bounds(*point) and _in_bounds(*lead)
    reason = "" if ok else f"인계/감속 지점이 맵 밖 ({point[0]:.0f},{point[1]:.0f})"

    # 이미 통로 위인데 목표가 등 뒤면 못 간다 (U턴 불가). 합류/선회 경로는
    # 아래에서 따로 본다.
    if ok and on_aisle and from_heading_deg is not None:
        behind = (point[0] < from_pose[0] - ON_AISLE_TOLERANCE_MM if travel > 0
                  else point[0] > from_pose[0] + ON_AISLE_TOLERANCE_MM)
        if behind:
            ok = False
            reason = (f"인계 지점 x={point[0]:.0f} 이 차량 뒤쪽이다 "
                      f"(x={from_pose[0]:.0f}, {'+x' if travel > 0 else '-x'} 진행) "
                      f"— U턴 없이는 못 간다")

    if merge is not None and ok:
        # 합류를 마치면 이미 merge[0] 에 와 있다. 그보다 뒤(진행 반대쪽)에 있는
        # 슬롯은 지나쳐 버린 것이고 U턴 없이는 못 돌아온다.
        past = (merge[0] > point[0] + ON_AISLE_TOLERANCE_MM if travel > 0
                else merge[0] < point[0] - ON_AISLE_TOLERANCE_MM)
        if past:
            ok = False
            reason = (f"통로 합류를 마치는 x={merge[0]:.0f} 이 인계 지점 "
                      f"x={point[0]:.0f} 을 지나친다 (차를 통로에 더 붙이거나 "
                      f"뒤쪽에서 출발할 것)")

    if turn is not None and ok:
        if not turn.feasible:
            ok, reason = False, turn.reason
        elif turn.join_point[0] > point[0] + ON_AISLE_TOLERANCE_MM:
            # 우회전만으로는 합류점이 슬롯보다 오른쪽이다. 전진으로 못 돌아온다.
            ok = False
            reason = (f"진입 우회전 합류점 x={turn.join_point[0]:.0f} 이 "
                      f"인계 지점 x={point[0]:.0f} 보다 오른쪽 — 지나친다 "
                      f"(차를 더 왼쪽/아래에 놓거나 다른 슬롯을 쓸 것)")

    return HandoffPlan(
        slot_id=slot.slot_id, point=point, heading_deg=approach,
        lead_point=lead, aisle_y=aisle_y, approach_heading_deg=approach,
        slot_entry_y=entry_y, gap_to_slot_mm=abs(aisle_y - entry_y),
        lead_in_turn_required=turn is not None, entry_turn=turn,
        merge_point=merge, feasible=ok, reason=reason,
    )


def _dedupe(path: list[tuple[float, float]], eps: float = 1.0) -> None:
    """거의 같은 자리에 연속으로 찍힌 점을 지운다."""
    i = 1
    while i < len(path):
        if math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) < eps:
            path.pop(i - 1)                 # 뒤엣것(=인계 지점)을 남긴다
        else:
            i += 1


class InfeasibleRouteError(ValueError):
    """차량 선회 반경으로는 만들 수 없는 경로."""

    def __init__(self, slot_id: str, reason: str) -> None:
        super().__init__(f"slot {slot_id}: {reason}")
        self.slot_id = slot_id
        self.reason = reason


def build_waypoints(
    slot: SlotSpec,
    route_id: int,
    route_nodes: list[str] | None = None,
    *,
    from_pose: tuple[float, float] | None = None,
    from_heading_deg: float | None = None,
    aisle_y: float = AISLE_Y,
    offset_mm: float = HANDOFF_OFFSET_MM,
    min_radius_mm: float = MIN_TURN_RADIUS_MM,
    strict: bool = False,
) -> list[Waypoint]:
    """슬롯 스펙 → 통로 주행 waypoint 목록 (슬롯 앞 인계까지).

    경로 모양 — 전부 통로 위 직선이다::

        CRUISE   (통로 밖에서 출발하면) 통로 합류점
        CRUISE   통로 진입점
        APPROACH 인계 지점 25cm 앞      ← COARSE 10cm → FINE 6cm 정밀 포착
        FINAL    인계 지점              ← 통로 방향 heading 요구, 3회 관측 확인

    **슬롯 안으로 들어가는 waypoint 는 만들지 않는다.** ALIGN/ENTRY 로 슬롯
    축을 따라 꺾어 들어가려면 90° 선회가 필요한데, 통로에서 진입선까지가
    300mm 뿐이고 최소 선회 반경은 570mm 다. 그 기동은 하드웨어팀 주차
    공식이 인계 지점에서부터 담당한다.

    Args:
        slot: 대상 슬롯 스펙.
        route_id: 이 경로의 식별자 (재생성 시 증가).
        route_nodes: 출발 위치 힌트 (첫 노드). from_pose 가 우선한다.
        from_pose: 차량 현재 위치(mm). 접근 방향 선택에 쓴다.
        aisle_y: 통로 y. 기본은 중앙 통로.
        offset_mm: 인계 지점을 슬롯 정면에서 진행 방향으로 밀어내는 양.
                   HW 주차 공식이 요구하면 조정한다.
        strict: True 면 인계 지점이 맵 밖일 때 InfeasibleRouteError.

    Raises:
        InfeasibleRouteError: strict=True 이고 인계 지점을 만들 수 없을 때.
    """
    start = from_pose
    if start is None and route_nodes:
        start = NODE_COORDINATES.get(route_nodes[0])
    if start is None:
        start = NODE_COORDINATES[SLOT_ROUTES[slot.slot_id][0]]

    plan = plan_handoff(slot, from_pose=start, from_heading_deg=from_heading_deg,
                        aisle_y=aisle_y, offset_mm=offset_mm,
                        min_radius_mm=min_radius_mm)
    if strict and not plan.feasible:
        raise InfeasibleRouteError(slot.slot_id, plan.reason)

    wps: list[Waypoint] = []
    wp_id = 1          # 펌웨어가 waypoint_id >= 1 을 요구한다 (route_id 도 동일)

    def add(phase: str, x: float, y: float, hdg: float | None, **kw) -> None:
        nonlocal wp_id
        wps.append(_make(route_id, wp_id, phase, x, y, hdg, **kw))
        wp_id += 1

    # ─ 진입 우회전: 출발점 → 통로 합류 ─
    # 차량은 +y 로 서서 출발해 우회전으로 통로에 붙는다. 반경은 출발 y 가
    # 정한다 (R = 통로y − 출발y). 원호를 쪼개 깔아야 차가 실제로 원을 탄다.
    arc: list[tuple[float, float]] = []
    aisle_start = start
    if plan.entry_turn is not None:
        arc = _entry_arc_points(plan.entry_turn)
        aisle_start = plan.entry_turn.join_point
    elif plan.merge_point is not None:
        # S자 합류 — 통로 위 한 점만 찍어주면 제어기가 완만히 붙는다.
        arc = [plan.merge_point]
        aisle_start = plan.merge_point

    # ─ 감속 지점: 인계에서 통로를 따라 되짚는다 ─
    # 통로 구간보다 길게 되짚으면 원호를 잘라먹게 되므로 합류점까지만 물러난다.
    c = math.cos(math.radians(plan.approach_heading_deg))
    aisle_run = abs(plan.point[0] - aisle_start[0])
    lead = min(HANDOFF_LEAD_MM, aisle_run)
    tol_mm = PHASE_DEFAULTS["APPROACH"]["position_tolerance_cm"] * 10.0
    approach_pt = (plan.point[0] - lead * c, aisle_y) if lead >= tol_mm else None

    # ─ 통로 직진: 합류점 → 감속 지점을 일정 간격으로 쪼갠다 ─
    # 긴 직선을 waypoint 하나로 두면 "지금 어디쯤인가"를 알 수 없다.
    cruise_end = approach_pt if approach_pt is not None else plan.point
    run = cruise_end[0] - aisle_start[0]
    n = int(abs(run) // AISLE_SEGMENT_MM)
    aisle_pts = [(aisle_start[0] + run * (i / (n + 1)), aisle_y)
                 for i in range(1, n + 1)]

    path = arc + aisle_pts
    _dedupe(path)

    def _same(a, b, eps=1.0):
        return math.hypot(a[0] - b[0], a[1] - b[1]) < eps

    if approach_pt is not None:
        # 되짚은 지점이 합류점과 겹치면(A3 처럼 통로가 짧을 때) 새로 찍지 않는다
        # 합류점과 감속점이 겹치면 새로 찍지 않고 합류점을 감속점으로 쓴다
        if path and _same(path[-1], approach_pt, tol_mm):
            approach_idx = len(path) - 1
        else:
            path.append(approach_pt)
            approach_idx = len(path) - 1
    else:
        # A2 처럼 합류점이 곧 인계 지점이면 통로에 감속 구간이 없다.
        # 원호의 마지막 직전 점을 감속 지점으로 쓴다 (원 위에 있는 점이다).
        approach_idx = len(path) - 2 if len(path) >= 2 else None

    if path and _same(path[-1], plan.point):
        path[-1] = plan.point                       # 마지막 점을 인계로 승격
        if approach_idx == len(path) - 1:
            approach_idx = len(path) - 2
    else:
        path.append(plan.point)

    for i, (x, y) in enumerate(path):
        if i == len(path) - 1:
            add("FINAL", x, y, plan.heading_deg, is_final=True)
        elif i == approach_idx:
            add("APPROACH", x, y, None,
                capture_tolerance_cm=PHASE_DEFAULTS["APPROACH"]["position_tolerance_cm"] + 4.0)
        else:
            add("CRUISE", x, y, None)

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
