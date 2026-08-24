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
    # 이 waypoint 로 향하는 경로 구간의 곡률 (1/mm, + = 좌회전, 0 = 직선).
    #   dθ = curvature × ds  (ds = 차체 heading 방향 부호 있는 이동량)
    # 제어기가 steering feedforward 를 만드는 데 쓴다. 직선 구간은 0 이라
    # 기존 경로(발렛 인계)는 값이 그대로 0 이고 동작이 바뀌지 않는다.
    curvature: float = 0.0
    # 곡선 중간 waypoint의 endpoint tangent를 통과했을 때 허용할 원호
    # corridor(cm). 점 허용오차를 키우지 않고, 이미 지난 표본점 재획득만 막는다.
    # production wire에는 필요 없는 host-only metadata다.
    path_capture_tolerance_cm: float | None = None

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
        d.pop("path_capture_tolerance_cm", None)
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
          heading_required: bool | None = None,
          capture_tolerance_cm: float | None = None,
          motion_direction: str = "FORWARD",
          curvature: float = 0.0,
          path_capture_tolerance_cm: float | None = None,
          position_tolerance_cm: float | None = None,
          heading_tolerance_deg: float | None = None) -> Waypoint:
    p = PHASE_DEFAULTS[phase]
    return Waypoint(
        route_id=route_id, waypoint_id=wp_id, phase=phase,
        x=x, y=y, target_heading_deg=heading,
        speed_cm_s=p["speed_cm_s"],
        position_tolerance_cm=(p["position_tolerance_cm"]
                               if position_tolerance_cm is None
                               else position_tolerance_cm),
        heading_tolerance_deg=(p["heading_tolerance_deg"]
                               if heading_tolerance_deg is None
                               else heading_tolerance_deg),
        heading_required=(heading is not None
                          if heading_required is None else heading_required),
        is_final=is_final,
        capture_tolerance_cm=capture_tolerance_cm,
        motion_direction=motion_direction,
        curvature=curvature,
        path_capture_tolerance_cm=path_capture_tolerance_cm,
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


# ─── 후면주차 (REVERSE_START → ENTRY → FINAL) ────────────────────────────────
#
# 발렛 인계 모델(build_waypoints)과 달리 **슬롯 안까지 후진으로 넣는다.**
# ESP32 에 새 명령이나 FSM 을 만들지 않는다 — AUTO_HOST 를 유지하고
# waypoint 의 phase + motion_direction 으로만 표현한다.
#
#   CRUISE(통로 직진) → APPROACH(감속) → ALIGN(전진 setup 원호, 마지막이
#   REVERSE_START) → ENTRY(후진 원호) → FINAL(슬롯 중심, 후진)
#
# 후보는 접근 side(LEFT/RIGHT) × 후진 원호각 φ 의 조합이다. φ 를 고정하지
# 않는 이유: 90° 로 못박으면 REVERSE_START 가 슬롯 중심에서 (R,R) 떨어져
# 가운데 슬롯이 맵 밖으로 나간다. φ 를 가변으로 두면 y offset 이 줄어든다.

# 후진 원호를 몇 도 간격으로 샘플링할지. 시작/끝 두 점만 주면 host 가 원호를
# 모르고 직선으로 잘라 들어간다.
REAR_ARC_STEP_DEG: float = 12.0
# REVERSE_START 전용 허용오차. 후진 원호 전체가 이 자세를 기준으로 하므로
# 일반 waypoint 처럼 5cm/12° 를 주면 원호가 통째로 어긋난 채 시작된다.
#
# 값 선택 근거 (closed-loop 스윕): 너무 좁으면(2cm) 차가 도착 판정을 못 받고
# 그 점을 지나쳐 계속 달려 경로를 벗어난다. 너무 넓으면(>=8°) 어긋난 자세로
# 후진 원호에 들어가 되잡지 못한다. 4cm/5° 가 "못 맞추면 그 자리에 선다"는
# 안전한 실패로 떨어지는 구간이다.
REVERSE_START_TOLERANCE_CM: float = 4.0
REVERSE_START_HEADING_TOLERANCE_DEG: float = 5.0
# 현재 자세에서 곧바로 후진 원호에 올라탈 때 허용할 **반경 방향** 오차.
# 원호상 위치(φ)는 자유롭게 고르므로 여기서 보는 것은 "원에서 얼마나
# 벗어나 있나"뿐이다. 실차 도착 정확도(9~11cm)와 같은 눈금.
REAR_ENTRY_CAPTURE_MM: float = 100.0
# 그 지점 접선 대비 heading 오차 허용치. PD + 곡률 feedforward 가 흡수할
# 수 있는 범위여야 한다.
REAR_ENTRY_HEADING_TOLERANCE_DEG: float = 15.0
# ─ candidate planner (현재 자세 기반) ─
# 계획 반경 후보. 실측 최소 610mm 보다 커야 추종 여유가 생긴다 — 610 으로
# 계획하면 원호 내내 최대 조향이라 오차를 되잡을 여력이 0 이다.
# 2026-08-14 실차 표(PROVISIONAL) 기준으로 현실화:
#   |steering| 0.90 에서 실측 반경이 LEFT 780 / RIGHT 680mm 였다.
#   따라서 700mm 로 계획하면 원호 내내 사실상 최대 조향이라 보정 여력이 0 이다.
#   추종 여유를 두려면 |steering| 0.7 대(=800~970mm) 이상에서 계획해야 한다.
REAR_RADIUS_CANDIDATES: tuple[float, ...] = (800.0, 900.0, 1000.0, 1100.0)
# setup 시작점이 현재 진행선에서 벗어나도 되는 양 / 최소 직진 거리.
# 차는 옆으로 못 가므로, 이 안이어야 달리면서 흡수할 수 있다.
REAR_LATERAL_TOLERANCE_MM: float = 80.0
REAR_MIN_RUN_MM: float = 60.0
# 경로가 이보다 길면 슬롯 하나 대는 데 과하다고 본다.
REAR_MAX_PATH_MM: float = 2500.0
# 후진 원호로 인정할 φ 범위. 너무 작으면 슬롯 축과 거의 나란해 의미가 없고,
# 너무 크면 원호가 맵을 벗어난다.
REAR_ENTRY_MIN_PHI_DEG: float = 15.0
REAR_ENTRY_MAX_PHI_DEG: float = 85.0
# 시도할 후진 원호각 후보
REAR_PHI_CANDIDATES: tuple[float, ...] = (30.0, 45.0, 60.0, 70.0, 80.0)
# 차량 외형 (footprint 검사용). 04번 문서 기준 — 길이는 추정값이다.
CAR_LENGTH_MM: float = 250.0
CAR_WIDTH_MM: float = 150.0


def _car_footprint(x: float, y: float, heading_deg: float
                   ) -> list[tuple[float, float]]:
    """차량 4모서리 좌표. 점이 아니라 면으로 맵 경계를 봐야 한다."""
    c = math.cos(math.radians(heading_deg))
    s = math.sin(math.radians(heading_deg))
    return [(x + dl * CAR_LENGTH_MM * c - dw * CAR_WIDTH_MM * s,
             y + dl * CAR_LENGTH_MM * s + dw * CAR_WIDTH_MM * c)
            for dl, dw in ((.5, .5), (.5, -.5), (-.5, .5), (-.5, -.5))]


def _path_clearance(path: list[tuple[float, float, float]]
                    ) -> tuple[float, float]:
    """(맵 밖 최대 초과 mm, 맵 경계까지 최소 여유 mm)."""
    worst, clear = 0.0, float("inf")
    for x, y, h in path:
        for px, py in _car_footprint(x, y, h):
            for v in (px, py):
                worst = max(worst, -v, v - LOT_SIZE_MM)
                clear = min(clear, v, LOT_SIZE_MM - v)
    return max(worst, 0.0), clear


def _arc_pose(x: float, y: float, heading_deg: float, sweep_deg: float,
              turn: float, radius_mm: float) -> tuple[float, float, float]:
    """자세 (x,y,heading) 에서 반경 radius 원호를 sweep 만큼 **전진**한 자세.

    turn = +1 좌회전(heading 증가), -1 우회전. 닫힌형이라 누적 오차가 없다.
    """
    cdir = math.radians(heading_deg + 90.0 * turn)
    cx = x + radius_mm * math.cos(cdir)
    cy = y + radius_mm * math.sin(cdir)
    a0 = math.atan2(y - cy, x - cx)
    a = a0 + turn * math.radians(sweep_deg)
    return (cx + radius_mm * math.cos(a), cy + radius_mm * math.sin(a),
            (heading_deg + turn * sweep_deg) % 360.0)


@dataclass(frozen=True)
class RearParkingPlan:
    """후면주차 후보 하나. 좌표는 mm, heading 은 차량 **앞** 방향."""

    slot_id: str
    side: int                                  # +1 = 통로를 +x 로 달려 접근
    phi_deg: float                             # 후진 원호각
    psi_deg: float                             # 전진 setup 원호각
    aisle_heading_deg: float                   # 통로 주행 방향 (0 또는 180)
    setup_start: tuple[float, float, float]    # CRUISE 마지막 = setup 원호 시작
    reverse_start: tuple[float, float, float]  # ALIGN 마지막 = 후진 시작
    align_poses: list[tuple[float, float, float]]   # setup 원호 샘플 (전진)
    entry_poses: list[tuple[float, float, float]]   # 후진 원호 샘플 (후진)
    final_pose: tuple[float, float, float]     # 슬롯 중심 + rear heading
    turn_setup: float
    turn_reverse: float
    radius_mm: float
    overflow_mm: float                         # 맵 밖 초과량 (0 이어야 한다)
    clearance_mm: float                        # 맵 경계까지 최소 여유
    aisle_offset_mm: float                     # setup 시작점의 통로 이탈량
    feasible: bool
    reason: str = ""


def plan_rear_parking(slot: SlotSpec, side: int, phi_deg: float, *,
                      aisle_y: float = AISLE_Y,
                      min_radius_mm: float = MIN_TURN_RADIUS_MM,
                      step_deg: float = REAR_ARC_STEP_DEG) -> RearParkingPlan:
    """후보 하나를 만든다 (기하만 — 도달성 판단은 호출자가 한다).

    ① FINAL(슬롯 중심 + rear heading)에서 후진 원호를 φ 만큼 **역산**해
       REVERSE_START 를 구한다.
    ② REVERSE_START 에서 통로 방향까지 전진 setup 원호를 역산해 시작점을 구한다.
       setup 은 후진과 **반대 방향**으로 돌려야 한다 — 같은 방향이면 두 원호가
       겹쳐서 차가 갔던 선을 그대로 되짚는 퇴화 경로가 된다.
    """
    rear = (slot.target_heading_deg + 180.0) % 360.0
    aisle_h = 0.0 if side > 0 else 180.0

    # 후진 원호 회전부호. 통로에서 들어오는 쪽에 REVERSE_START 가 놓이도록 정한다.
    turn_rev = -1.0 if side > 0 else 1.0
    if slot.entry_side == "BOTTOM":
        turn_rev = -turn_rev

    rs = _arc_pose(slot.center_x, slot.center_y, rear, phi_deg, turn_rev,
                   min_radius_mm)

    # setup 원호: 통로 heading → REVERSE_START heading
    dh = (rs[2] - aisle_h + 180.0) % 360.0 - 180.0
    psi = abs(dh)
    turn_setup = 1.0 if dh > 0 else -1.0
    # REVERSE_START 에서 psi 만큼 되돌아가면 setup 시작 자세가 나온다.
    start = _arc_pose(rs[0], rs[1], rs[2], -psi, turn_setup, min_radius_mm)

    n_align = max(1, math.ceil(psi / step_deg)) if psi > 1e-6 else 0
    align = [_arc_pose(start[0], start[1], aisle_h, psi * i / n_align,
                       turn_setup, min_radius_mm)
             for i in range(1, n_align + 1)] if n_align else [rs]

    n_entry = max(1, round(phi_deg / step_deg))
    entry = [_arc_pose(slot.center_x, slot.center_y, rear,
                       phi_deg * (n_entry + 1 - i) / (n_entry + 1), turn_rev,
                       min_radius_mm)
             for i in range(1, n_entry + 1)]

    final_pose = (slot.center_x, slot.center_y, rear)
    overflow, clearance = _path_clearance(align + entry + [final_pose, start])
    aisle_offset = abs(start[1] - aisle_y)

    ok, reason = True, ""
    if overflow > 0.0:
        ok = False
        reason = f"경로가 맵 밖으로 {overflow:.0f}mm 나간다"
    elif aisle_offset > ON_AISLE_TOLERANCE_MM:
        # 통로에서 벗어난 자세로 setup 을 시작해야 하면, 거기까지 가는 경로가
        # 또 필요하다. 좌우 벽에 붙는 경우가 많아 활주로가 안 나온다.
        ok = False
        reason = (f"setup 시작점이 통로에서 {aisle_offset:.0f}mm 떨어져 있다 "
                  f"(허용 {ON_AISLE_TOLERANCE_MM:.0f}mm)")

    return RearParkingPlan(
        slot_id=slot.slot_id, side=side, phi_deg=phi_deg, psi_deg=psi,
        aisle_heading_deg=aisle_h, setup_start=start, reverse_start=rs,
        align_poses=align, entry_poses=entry, final_pose=final_pose,
        turn_setup=turn_setup, turn_reverse=turn_rev, radius_mm=min_radius_mm,
        overflow_mm=overflow, clearance_mm=clearance,
        aisle_offset_mm=aisle_offset, feasible=ok, reason=reason)


def choose_rear_parking_plan(slot: SlotSpec, *, aisle_y: float = AISLE_Y,
                             min_radius_mm: float = MIN_TURN_RADIUS_MM,
                             from_pose: tuple[float, float] | None = None,
                             step_deg: float = REAR_ARC_STEP_DEG,
                             ) -> tuple[RearParkingPlan | None, list[RearParkingPlan]]:
    """모든 (side, φ) 후보 중 가장 안전한 것을 고른다.

    ψ + φ 가 항상 90° 라 경로 길이와 방향전환 횟수가 전 후보 동일하다.
    그래서 **실차 강건성**으로 고른다:
      1순위 맵 경계 여유가 큰 것 (10mm 단위로 뭉쳐 비교)
      2순위 후진 원호가 짧은 것 (후진 중 heading 추정이 더 어렵다)
    from_pose 를 주면 차가 이미 있는 쪽에서 접근하는 후보를 먼저 본다.
    """
    cands = [plan_rear_parking(slot, side, phi, aisle_y=aisle_y,
                               min_radius_mm=min_radius_mm, step_deg=step_deg)
             for side in (1, -1) for phi in REAR_PHI_CANDIDATES]
    ok = [c for c in cands if c.feasible]
    if from_pose is not None:
        # 차 뒤쪽에서 시작하는 후보는 지나쳐 버린 것이라 전진으로 못 잡는다.
        ahead = [c for c in ok
                 if (c.setup_start[0] - from_pose[0]) * c.side > 0]
        if ahead:
            ok = ahead
    if not ok:
        return None, cands
    # 맵 여유는 FINAL 자세(슬롯 깊이 300 - 차 길이 250)가 지배해서 후보 간
    # 변별이 거의 없다. 그래서 실제로 갈리는 값 — setup 시작점이 통로에
    # 얼마나 가까운가 — 를 두 번째 기준으로 둔다. 차를 통로에 놓고 시작하므로
    # 이 값이 곧 초기 배치 오차 허용량이다.
    ok.sort(key=lambda c: (-round(c.clearance_mm / 10.0), c.aisle_offset_mm,
                           c.phi_deg))
    return ok[0], cands


# ─── 주차 setup recovery ─────────────────────────────────────────────────────
# 인계 자세에서 곧바로 후진 원호에 못 올라타는 경우가 있다. 인계는 통로와
# 나란한데(heading 0/180) 원호 진입점은 통로에서 30~60cm 옆에 있기 때문이고,
# 차는 옆으로 못 간다.
#
# 이때 옛 waypoint 를 고집하지 않는다 — **차체 각도를 만들어 주는 짧은 기동**
# 을 넣고 fresh pose 로 다시 계획한다. 후진하며 조향하면 heading 이 바뀌므로
# 한 segment 로 충분한 경우가 많다.
#
# bounded search: (반경 × 스윕각 × 회전방향) 조합만 본다. Hybrid A* 아님.
SETUP_RECOVERY_SWEEPS_DEG: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0,
                                                35.0, 40.0, 45.0)
SETUP_RECOVERY_STRAIGHTS_MM: tuple[float, ...] = (100.0, 200.0, 300.0)
SETUP_RECOVERY_MAX_MM: float = 700.0        # 기동 이동거리 상한
SETUP_RECOVERY_MAX_SEGMENTS: int = 3
SETUP_RECOVERY_BEAM: int = 256


@dataclass(frozen=True)
class SetupSegment:
    """setup recovery를 이루는 하나의 Ackermann motion primitive."""

    poses: list[tuple[float, float, float]]
    radius_mm: float
    sweep_deg: float
    turn: float
    reverse: bool
    length_mm: float


@dataclass(frozen=True)
class SetupRecovery:
    """주차 진입 자세를 만들기 위한 1~2 segment bounded 기동."""

    slot_id: str
    poses: list[tuple[float, float, float]]   # 샘플 (마지막이 목표 자세)
    end_pose: tuple[float, float, float]
    radius_mm: float
    sweep_deg: float
    turn: float
    reverse: bool
    length_mm: float
    clearance_mm: float
    feasible: bool
    reason: str = ""
    segments: tuple[SetupSegment, ...] = ()


def _slot_keepout(specs: dict[str, SlotSpec], exclude: str
                  ) -> list[tuple[float, float, float, float]]:
    """다른 슬롯의 사각 영역 (x0,y0,x1,y1). 기동이 남의 자리를 밟지 않게 한다."""
    out = []
    for sid, s in specs.items():
        if sid == exclude:
            continue
        out.append((s.center_x - s.width / 2, s.center_y - s.length / 2,
                    s.center_x + s.width / 2, s.center_y + s.length / 2))
    return out


def plan_setup_recovery(slot: SlotSpec, from_pose: tuple[float, float],
                        from_heading_deg: float, *,
                        radii_mm: tuple[float, ...] = REAR_RADIUS_CANDIDATES,
                        step_deg: float = REAR_ARC_STEP_DEG,
                        obstacle_poses: tuple[tuple[float, float, float], ...] = (),
                        ) -> SetupRecovery | None:
    """현재 자세에서 주차 계획이 가능해지는 **가장 짧은** 기동을 찾는다.

    각 후보 기동을 실제로 굴려 끝 자세를 구하고, 그 자세에서
    choose_rear_candidate 가 성립하는지로 판정한다 — 기동 자체가 목적이
    아니라 "계획 가능한 자세"를 만드는 것이 목적이다.
    """
    specs = default_slot_specs()
    keepout = _slot_keepout(specs, slot.slot_id)
    start_pose = (from_pose[0], from_pose[1], from_heading_deg)
    initial_keepout: dict[int, float] = {}
    for index, (x0, y0, x1, y1) in enumerate(keepout):
        if any(x0 <= px <= x1 and y0 <= py <= y1
               for px, py in _car_footprint(*start_pose)):
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            initial_keepout[index] = math.hypot(start_pose[0] - cx,
                                                start_pose[1] - cy)

    def overlaps_vehicle(x: float, y: float, h: float,
                         obstacle: tuple[float, float, float]) -> bool:
        """Return whether two oriented physical footprints overlap (SAT)."""
        ca = _car_footprint(x, y, h)
        cb = _car_footprint(*obstacle)
        a = [ca[i] for i in (0, 1, 3, 2)]
        b = [cb[i] for i in (0, 1, 3, 2)]
        axes = []
        for poly in (a, b):
            for i in (0, 1):
                dx = poly[i + 1][0] - poly[0][0]
                dy = poly[i + 1][1] - poly[0][1]
                norm = math.hypot(dx, dy)
                axes.append((-dy / norm, dx / norm))
        for ax, ay in axes:
            pa = [px * ax + py * ay for px, py in a]
            pb = [px * ax + py * ay for px, py in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
        return True

    def blocked(poses) -> tuple[float, str]:
        overflow, clearance = _path_clearance(poses)
        if overflow > 0.0:
            return clearance, f"맵 밖 {overflow:.0f}mm"
        for x, y, h in poses:
            if any(overlaps_vehicle(x, y, h, obstacle)
                   for obstacle in obstacle_poses):
                return clearance, "obstacle footprint"
            for px, py in _car_footprint(x, y, h):
                for index, (x0, y0, x1, y1) in enumerate(keepout):
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        # 실차가 이미 빈 인접 슬롯 경계에 걸친 상태라면 탈출은
                        # 허용하되, 해당 슬롯 중심 쪽으로 더 깊어지는 기동은 금지한다.
                        if index in initial_keepout:
                            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                            if math.hypot(x - cx, y - cy) + 1e-6 \
                                    >= initial_keepout[index]:
                                continue
                        return clearance, "다른 슬롯 침범"
        return clearance, ""

    def primitives(start: tuple[float, float, float]):
        for reverse in (True, False):      # 통로에서는 후진 후보를 먼저 본다
            sign = -1.0 if reverse else 1.0
            heading = math.radians(start[2])
            for distance in SETUP_RECOVERY_STRAIGHTS_MM:
                n = max(2, math.ceil(distance / 50.0))
                poses = [(start[0] + sign * distance * i / n * math.cos(heading),
                          start[1] + sign * distance * i / n * math.sin(heading),
                          start[2]) for i in range(n + 1)]
                yield SetupSegment(poses, float("inf"), 0.0, 0.0, reverse,
                                   distance)
            for radius in radii_mm:
                for turn in (1.0, -1.0):
                    for sweep in SETUP_RECOVERY_SWEEPS_DEG:
                        length = math.radians(sweep) * radius
                        if length > SETUP_RECOVERY_MAX_MM:
                            continue
                        signed = -sweep if reverse else sweep
                        n = max(2, math.ceil(sweep / step_deg))
                        poses = [_arc_pose(start[0], start[1], start[2],
                                           signed * i / n, turn, radius)
                                 for i in range(n + 1)]
                        yield SetupSegment(poses, radius, sweep, turn, reverse,
                                           length)

    best: SetupRecovery | None = None

    def consider(segments: tuple[SetupSegment, ...]) -> None:
        nonlocal best
        length = sum(s.length_mm for s in segments)
        if length > SETUP_RECOVERY_MAX_MM:
            return
        if best is not None and length >= best.length_mm:
            return
        poses = list(segments[0].poses)
        for seg in segments[1:]:
            poses.extend(seg.poses[1:])
        clearance, why = blocked(poses)
        if why:
            return
        end = poses[-1]
        cand, _ = choose_rear_candidate(slot, end[:2], end[2],
                                        radii_mm=radii_mm, step_deg=step_deg)
        if cand is None:
            return
        first = segments[0]
        best = SetupRecovery(
            slot_id=slot.slot_id, poses=poses, end_pose=end,
            radius_mm=first.radius_mm, sweep_deg=first.sweep_deg,
            turn=first.turn, reverse=first.reverse, length_mm=length,
            clearance_mm=clearance, feasible=True, segments=segments)

    # 1~3 segment bounded beam search. 각 terminal pose에서 기존 rear planner를
    # 그대로 호출해 성공하는 pose만 goal로 인정한다.
    start = start_pose
    frontier: list[tuple[SetupSegment, ...]] = [()]
    for _depth in range(SETUP_RECOVERY_MAX_SEGMENTS):
        next_by_cell: dict[tuple[int, int, int, bool, int],
                           tuple[float, tuple[SetupSegment, ...]]] = {}
        for prefix in frontier:
            terminal = start if not prefix else prefix[-1].poses[-1]
            used = sum(s.length_mm for s in prefix)
            for segment in primitives(terminal):
                if used + segment.length_mm > SETUP_RECOVERY_MAX_MM:
                    continue
                if prefix:
                    prev = prefix[-1]
                    if (segment.reverse == prev.reverse
                            and segment.turn == prev.turn
                            and segment.radius_mm == prev.radius_mm):
                        continue
                candidate = (*prefix, segment)
                consider(candidate)
                end = segment.poses[-1]
                clearance, why = blocked(segment.poses)
                if why:
                    continue
                key = (round(end[0] / 50.0), round(end[1] / 50.0),
                       round((end[2] % 360.0) / 10.0), segment.reverse,
                       int(math.copysign(1, segment.turn)) if segment.turn else 0)
                score = (math.hypot(end[0] - slot.center_x,
                                    end[1] - AISLE_Y)
                         + 0.15 * (used + segment.length_mm)
                         - 0.05 * clearance)
                old = next_by_cell.get(key)
                if old is None or score < old[0]:
                    next_by_cell[key] = (score, candidate)
        if best is not None:
            break
        frontier = [item[1] for item in sorted(next_by_cell.values(),
                                                key=lambda item: item[0])
                    [:SETUP_RECOVERY_BEAM]]
    return best


def build_setup_recovery_waypoints(slot: SlotSpec, route_id: int, *,
                                   from_pose: tuple[float, float],
                                   from_heading_deg: float,
                                   radii_mm: tuple[float, ...] = REAR_RADIUS_CANDIDATES,
                                   obstacle_poses: tuple[tuple[float, float, float], ...] = (),
                                   ) -> list[Waypoint]:
    """setup recovery 기동을 waypoint 로. 없으면 빈 목록."""
    rec = plan_setup_recovery(slot, from_pose, from_heading_deg,
                              radii_mm=radii_mm,
                              obstacle_poses=obstacle_poses)
    if rec is None:
        return []
    phase = "RECOVERY"
    wps: list[Waypoint] = []
    segments = rec.segments or (
        SetupSegment(rec.poses, rec.radius_mm, rec.sweep_deg, rec.turn,
                     rec.reverse, rec.length_mm),)
    wp_id = 1
    for seg_idx, seg in enumerate(segments):
        direction = "REVERSE" if seg.reverse else "FORWARD"
        k = 0.0 if math.isinf(seg.radius_mm) else seg.turn / seg.radius_mm
        for pose_idx, (x, y, h) in enumerate(seg.poses[1:], start=1):
            last = (seg_idx == len(segments) - 1
                    and pose_idx == len(seg.poses) - 1)
            wps.append(_make(route_id, wp_id, phase, x, y, h,
                             heading_required=last,
                             motion_direction=direction, curvature=k,
                             path_capture_tolerance_cm=(
                                 REAR_ENTRY_CAPTURE_MM / 10.0 if k else None)))
            wp_id += 1
    return wps


@dataclass(frozen=True)
class RearCandidate:
    """현재 자세에서 슬롯 FINAL 까지 가는 후보 하나.

    구조는 (직진) → (전진 setup 원호 ψ) → (후진 원호 φ) → FINAL 로 고정이다.
    자유 변수는 **반경 R, 후진 원호각 φ, 접근 side** 뿐인 bounded parametric
    search 다. 연속 최적화나 그래프 탐색을 하지 않는다.
    """

    slot_id: str
    radius_mm: float
    phi_deg: float
    psi_deg: float
    side: int
    setup_start: tuple[float, float, float]
    reverse_start: tuple[float, float, float]
    align_poses: list[tuple[float, float, float]]
    entry_poses: list[tuple[float, float, float]]
    final_pose: tuple[float, float, float]
    turn_setup: float
    turn_reverse: float
    run_mm: float                # 현재 자세에서 setup 시작까지 직진 거리
    lateral_mm: float            # 그 지점이 현재 heading 광선에서 벗어난 양
    clearance_mm: float          # 맵 경계까지 최소 여유 (swept footprint)
    maneuver_clearance_mm: float # ENTRY 전 APPROACH/ALIGN의 최소 경계 여유
    overflow_mm: float
    path_length_mm: float
    steering_demand: float       # 원호 유지에 필요한 조향 (1.0 = 최대)
    feasible: bool
    reason: str = ""


def plan_rear_candidate(slot: SlotSpec, from_pose: tuple[float, float],
                        from_heading_deg: float, *, side: int, phi_deg: float,
                        radius_mm: float,
                        step_deg: float = REAR_ARC_STEP_DEG) -> RearCandidate:
    """(R, φ, side) 조합 하나를 현재 자세 기준으로 평가한다.

    통로를 전제하지 않는다 — **차의 현재 heading 이 곧 진입 직선**이다.
    그래서 같은 슬롯이라도 Pose 가 달라지면 통과하는 조합이 달라진다.
    """
    rear = (slot.target_heading_deg + 180.0) % 360.0
    turn_rev = -1.0 if side > 0 else 1.0
    if slot.entry_side == "BOTTOM":
        turn_rev = -turn_rev

    def fail(reason: str, **kw) -> RearCandidate:
        base = dict(slot_id=slot.slot_id, radius_mm=radius_mm, phi_deg=phi_deg,
                    psi_deg=0.0, side=side,
                    setup_start=(0.0, 0.0, 0.0), reverse_start=(0.0, 0.0, 0.0),
                    align_poses=[], entry_poses=[],
                    final_pose=(slot.center_x, slot.center_y, rear),
                    turn_setup=0.0, turn_reverse=turn_rev, run_mm=0.0,
                    lateral_mm=float("inf"), clearance_mm=0.0,
                    maneuver_clearance_mm=0.0,
                    overflow_mm=float("inf"), path_length_mm=float("inf"),
                    steering_demand=1.0, feasible=False, reason=reason)
        base.update(kw)
        return RearCandidate(**base)

    if radius_mm < MIN_TURN_RADIUS_MM:
        return fail(f"계획 반경 {radius_mm:.0f}mm 가 최소 선회 반경 미만")

    rs = _arc_pose(slot.center_x, slot.center_y, rear, phi_deg, turn_rev,
                   radius_mm)

    # 전진 setup 원호: 현재 heading → REVERSE_START heading
    dh = (rs[2] - from_heading_deg + 180.0) % 360.0 - 180.0
    psi = abs(dh)
    turn_setup = 1.0 if dh > 0 else -1.0
    if psi < 1e-6:
        start = (rs[0], rs[1], from_heading_deg)
        align = [rs]
    else:
        # 두 원호가 같은 방향이면 왔던 길을 되짚는 퇴화 경로가 된다.
        if turn_setup * turn_rev > 0:
            return fail("setup 과 후진 원호가 같은 방향 (경로가 겹친다)")
        start = _arc_pose(rs[0], rs[1], rs[2], -psi, turn_setup, radius_mm)
        n_align = max(1, math.ceil(psi / step_deg))
        align = [_arc_pose(start[0], start[1], from_heading_deg,
                           psi * i / n_align, turn_setup, radius_mm)
                 for i in range(1, n_align + 1)]

    # setup 시작점이 현재 heading 광선 위에 (앞쪽으로) 있는가
    cx, cy = math.cos(math.radians(from_heading_deg)), \
        math.sin(math.radians(from_heading_deg))
    dx, dy = start[0] - from_pose[0], start[1] - from_pose[1]
    run = dx * cx + dy * cy
    lateral = abs(-dx * math.sin(math.radians(from_heading_deg))
                  + dy * math.cos(math.radians(from_heading_deg)))
    if run < REAR_MIN_RUN_MM:
        return fail(f"setup 시작점이 뒤에 있거나 너무 가깝다 (전진 {run:.0f}mm)",
                    run_mm=run, lateral_mm=lateral)
    if lateral > REAR_LATERAL_TOLERANCE_MM:
        return fail(f"setup 시작점이 진행선에서 {lateral:.0f}mm 벗어나 있다",
                    run_mm=run, lateral_mm=lateral)
    entry_heading_error = math.degrees(math.atan2(lateral, run))
    if entry_heading_error > REVERSE_START_HEADING_TOLERANCE_DEG:
        return fail(
            f"setup 원호 진입 heading 오차 {entry_heading_error:.1f}deg가 "
            f"허용 {REVERSE_START_HEADING_TOLERANCE_DEG:.0f}deg 초과",
            run_mm=run, lateral_mm=lateral)

    n_entry = max(1, round(phi_deg / step_deg))
    entry = [_arc_pose(slot.center_x, slot.center_y, rear,
                       phi_deg * (n_entry + 1 - i) / (n_entry + 1), turn_rev,
                       radius_mm)
             for i in range(1, n_entry + 1)]
    final_pose = (slot.center_x, slot.center_y, rear)

    # swept footprint — 직진 구간까지 포함해 촘촘히 훑는다
    swept: list[tuple[float, float, float]] = []
    n_run = max(2, int(run / 25.0))
    for i in range(n_run + 1):
        swept.append((from_pose[0] + cx * run * i / n_run,
                      from_pose[1] + cy * run * i / n_run, from_heading_deg))
    for sweep, turn, ox, oy, oh in ((psi, turn_setup, start[0], start[1],
                                     from_heading_deg),):
        m = max(2, int(math.radians(sweep) * radius_mm / 25.0))
        for i in range(m + 1):
            swept.append(_arc_pose(ox, oy, oh, sweep * i / m, turn, radius_mm))
    _, maneuver_clearance = _path_clearance(swept)
    m = max(2, int(math.radians(phi_deg) * radius_mm / 25.0))
    for i in range(m + 1):
        swept.append(_arc_pose(slot.center_x, slot.center_y, rear,
                               phi_deg * i / m, turn_rev, radius_mm))
    swept.append(final_pose)
    overflow, clearance = _path_clearance(swept)

    length = run + math.radians(psi + phi_deg) * radius_mm
    demand = MIN_TURN_RADIUS_MM / radius_mm

    ok, reason = True, ""
    if overflow > 0.0:
        ok, reason = False, f"경로가 맵 밖으로 {overflow:.0f}mm 나간다"
    elif length > REAR_MAX_PATH_MM:
        ok, reason = False, f"경로가 {length:.0f}mm 로 과도하다"

    return RearCandidate(
        slot_id=slot.slot_id, radius_mm=radius_mm, phi_deg=phi_deg,
        psi_deg=psi, side=side, setup_start=start, reverse_start=rs,
        align_poses=align, entry_poses=entry, final_pose=final_pose,
        turn_setup=turn_setup, turn_reverse=turn_rev, run_mm=run,
        lateral_mm=lateral, clearance_mm=clearance,
        maneuver_clearance_mm=maneuver_clearance, overflow_mm=overflow,
        path_length_mm=length, steering_demand=demand,
        feasible=ok, reason=reason)


def choose_rear_candidate(slot: SlotSpec, from_pose: tuple[float, float],
                          from_heading_deg: float, *,
                          radii_mm: tuple[float, ...] = REAR_RADIUS_CANDIDATES,
                          step_deg: float = REAR_ARC_STEP_DEG,
                          ) -> tuple[RearCandidate | None, list[RearCandidate]]:
    """현재 자세에서 가능한 후보를 모두 만들고 하나를 고른다.

    점수 순서 (앞이 우선):
      1. 맵 경계 여유가 큰 것 (20mm 단위로 뭉쳐 비교 — 잡음으로 순서가 바뀌지 않게)
      2. 최대 조향 의존이 적은 것 (반경이 큰 것) — 추종 여유가 곧 성공률이다
      3. 경로가 짧은 것
    """
    cands = [plan_rear_candidate(slot, from_pose, from_heading_deg, side=side,
                                 phi_deg=phi, radius_mm=r, step_deg=step_deg)
             for r in radii_mm
             for side in (1, -1)
             for phi in REAR_PHI_CANDIDATES]
    ok = [c for c in cands if c.feasible]
    if not ok:
        return None, cands
    # Full-route clearance is always dominated by the target-slot FINAL pose
    # (25 mm for the physical car), so it cannot distinguish candidates.  Rank
    # the actually tracked forward maneuver independently.
    ok.sort(key=lambda c: (-round(c.maneuver_clearance_mm / 20.0),
                           c.steering_demand, c.path_length_mm))
    return ok[0], cands


def build_rear_candidate_waypoints(slot: SlotSpec, route_id: int, *,
                                   from_pose: tuple[float, float],
                                   from_heading_deg: float,
                                   radii_mm: tuple[float, ...] = REAR_RADIUS_CANDIDATES,
                                   step_deg: float = REAR_ARC_STEP_DEG,
                                   strict: bool = True) -> list[Waypoint]:
    """현재 자세 → 슬롯 FINAL 까지의 waypoint (candidate planner)."""
    cand, all_c = choose_rear_candidate(slot, from_pose, from_heading_deg,
                                        radii_mm=radii_mm, step_deg=step_deg)
    if cand is None:
        best = min(all_c, key=lambda c: (c.overflow_mm, c.lateral_mm))
        if strict:
            raise InfeasibleRouteError(slot.slot_id, best.reason)
        return []

    wps: list[Waypoint] = []
    wp_id = 1

    def add(phase: str, x: float, y: float, hdg: float | None, **kw) -> None:
        nonlocal wp_id
        wps.append(_make(route_id, wp_id, phase, x, y, hdg, **kw))
        wp_id += 1

    sx, sy, sh = cand.setup_start
    # This point is the tangent entry to the planned setup arc.  Entering it
    # with endpoint/chord heading (as the two real runs did) starts the next arc
    # one whole sample ahead in heading and makes its feedforward fight the
    # endpoint-bearing correction.  A mismatch is a replan, not permission to
    # execute a different physical arc.
    add("APPROACH", sx, sy, sh, heading_required=True,
        heading_tolerance_deg=REVERSE_START_HEADING_TOLERANCE_DEG,
        capture_tolerance_cm=PHASE_DEFAULTS["APPROACH"]["position_tolerance_cm"] + 4.0)

    k_setup = cand.turn_setup / cand.radius_mm
    for i, (x, y, h) in enumerate(cand.align_poses):
        last = i == len(cand.align_poses) - 1
        add("ALIGN", x, y, h, heading_required=last, curvature=k_setup,
            path_capture_tolerance_cm=REAR_ENTRY_CAPTURE_MM / 10.0,
            position_tolerance_cm=(REVERSE_START_TOLERANCE_CM if last else None),
            heading_tolerance_deg=(REVERSE_START_HEADING_TOLERANCE_DEG
                                   if last else None))

    k_entry = cand.turn_reverse / cand.radius_mm
    for x, y, h in cand.entry_poses:
        add("ENTRY", x, y, h, heading_required=False,
            motion_direction="REVERSE", curvature=k_entry,
            path_capture_tolerance_cm=REAR_ENTRY_CAPTURE_MM / 10.0)

    fx, fy, fh = cand.final_pose
    add("FINAL", fx, fy, fh, is_final=True, motion_direction="REVERSE")
    return wps


@dataclass(frozen=True)
class RearEntryPlan:
    """현재 자세에서 곧바로 후진 원호에 올라타는 계획 (ALIGN 없음)."""

    slot_id: str
    phi_deg: float
    entry_poses: list[tuple[float, float, float]]
    final_pose: tuple[float, float, float]
    turn_reverse: float
    radius_mm: float
    offset_mm: float                 # 현재 위치 ↔ 그 φ 의 진입점 거리
    overflow_mm: float
    feasible: bool
    reason: str = ""


def plan_rear_entry_from_pose(slot: SlotSpec, pose_mm: tuple[float, float],
                              heading_deg: float, *,
                              min_radius_mm: float = MIN_TURN_RADIUS_MM,
                              step_deg: float = REAR_ARC_STEP_DEG,
                              max_offset_mm: float = REAR_ENTRY_CAPTURE_MM,
                              ) -> RearEntryPlan:
    """지금 자세에서 후진 원호에 바로 올라탈 수 있는지 본다.

    후진 원호는 **원**이고 φ 는 "그 원의 어디로 들어가느냐"일 뿐이다. 그래서
    계획해 둔 REVERSE_START 를 고집할 이유가 없다 — 차의 **현재 heading 이
    접선이 되는 지점**을 φ 로 역산하고, 위치가 그 지점에서 얼마나 떨어졌는지만
    본다. 가까우면 ALIGN 없이 ENTRY 부터 시작하는 새 경로를 만든다.

    이 방식이라 특정 heading 오차값(예: 8.8°)을 코드에 박지 않는다 —
    오차가 곧 φ 의 차이로 흡수된다.
    """
    rear = (slot.target_heading_deg + 180.0) % 360.0
    best: RearEntryPlan | None = None
    for side in (1, -1):
        turn_rev = -1.0 if side > 0 else 1.0
        if slot.entry_side == "BOTTOM":
            turn_rev = -turn_rev

        # 후진 원호의 중심. FINAL 에서 rear heading 기준 turn 쪽 90°, 거리 R.
        cdir = math.radians(rear + 90.0 * turn_rev)
        cx = slot.center_x + min_radius_mm * math.cos(cdir)
        cy = slot.center_y + min_radius_mm * math.sin(cdir)

        # 차를 그 원에 **투영**한다. φ 를 heading 에서 역산하면 반경 오차와
        # 원호상 위치가 뒤섞여 실제보다 훨씬 멀어 보인다.
        dx, dy = pose_mm[0] - cx, pose_mm[1] - cy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        radial_err = abs(d - min_radius_mm)
        a_car = math.degrees(math.atan2(dy, dx))
        a_final = math.degrees(math.atan2(slot.center_y - cy,
                                          slot.center_x - cx))
        phi = turn_rev * ((a_car - a_final + 180.0) % 360.0 - 180.0)
        # 그 지점에서의 접선 heading (후진 진행 기준)
        tangent = (a_car + 90.0 * turn_rev) % 360.0
        heading_err = abs((heading_deg - tangent + 180.0) % 360.0 - 180.0)

        if not (REAR_ENTRY_MIN_PHI_DEG <= phi <= REAR_ENTRY_MAX_PHI_DEG):
            continue
        rs = _arc_pose(slot.center_x, slot.center_y, rear, phi, turn_rev,
                       min_radius_mm)
        offset = radial_err

        n_entry = max(1, round(phi / step_deg))
        entry = [_arc_pose(slot.center_x, slot.center_y, rear,
                           phi * (n_entry + 1 - i) / (n_entry + 1), turn_rev,
                           min_radius_mm)
                 for i in range(1, n_entry + 1)]
        final_pose = (slot.center_x, slot.center_y, rear)
        overflow, _ = _path_clearance(entry + [final_pose, rs])

        ok, reason = True, ""
        if overflow > 0.0:
            ok, reason = False, f"후진 원호가 맵 밖으로 {overflow:.0f}mm 나간다"
        elif offset > max_offset_mm:
            ok, reason = False, (f"후진 원호에서 반경으로 {offset:.0f}mm 벗어나 있다 "
                                 f"(허용 {max_offset_mm:.0f}mm)")
        elif heading_err > REAR_ENTRY_HEADING_TOLERANCE_DEG:
            ok, reason = False, (f"접선 대비 heading 이 {heading_err:.0f}° 틀어져 있다 "
                                 f"(허용 {REAR_ENTRY_HEADING_TOLERANCE_DEG:.0f}°)")
        cand = RearEntryPlan(
            slot_id=slot.slot_id, phi_deg=phi, entry_poses=entry,
            final_pose=final_pose, turn_reverse=turn_rev,
            radius_mm=min_radius_mm, offset_mm=offset, overflow_mm=overflow,
            feasible=ok, reason=reason)
        if best is None or (cand.feasible, -cand.offset_mm) > (best.feasible,
                                                               -best.offset_mm):
            best = cand
    if best is None:
        return RearEntryPlan(slot.slot_id, 0.0, [], (slot.center_x, slot.center_y,
                                                     rear), 0.0, min_radius_mm,
                             float("inf"), 0.0, False,
                             f"heading {heading_deg:.0f}° 로는 후진 원호에 못 올라탄다")
    return best


def build_rear_entry_waypoints(slot: SlotSpec, route_id: int, *,
                               from_pose: tuple[float, float],
                               from_heading_deg: float,
                               min_radius_mm: float = MIN_TURN_RADIUS_MM,
                               step_deg: float = REAR_ARC_STEP_DEG,
                               strict: bool = True) -> list[Waypoint]:
    """현재 자세에서 ENTRY → FINAL 만으로 이루어진 경로를 만든다.

    ALIGN 이 없다 — 이미 후진 시작 자세에 있다고 보고 바로 원호를 탄다.
    REVERSE_START 에서 heading 이 어긋났을 때 "옛 waypoint 로 되돌아가는" 대신
    쓰는 경로다.
    """
    plan = plan_rear_entry_from_pose(slot, from_pose, from_heading_deg,
                                     min_radius_mm=min_radius_mm,
                                     step_deg=step_deg)
    if not plan.feasible:
        if strict:
            raise InfeasibleRouteError(slot.slot_id, plan.reason)
        return []

    wps: list[Waypoint] = []
    wp_id = 1
    k_entry = plan.turn_reverse / plan.radius_mm
    for x, y, h in plan.entry_poses:
        wps.append(_make(route_id, wp_id, "ENTRY", x, y, h,
                         heading_required=False,
                         motion_direction="REVERSE", curvature=k_entry,
                         path_capture_tolerance_cm=REAR_ENTRY_CAPTURE_MM / 10.0))
        wp_id += 1
    fx, fy, fh = plan.final_pose
    wps.append(_make(route_id, wp_id, "FINAL", fx, fy, fh, is_final=True,
                     motion_direction="REVERSE"))
    return wps


def build_rear_parking_waypoints(
    slot: SlotSpec,
    route_id: int,
    *,
    from_pose: tuple[float, float] | None = None,
    from_heading_deg: float | None = None,
    aisle_y: float = AISLE_Y,
    min_radius_mm: float = MIN_TURN_RADIUS_MM,
    step_deg: float = REAR_ARC_STEP_DEG,
    strict: bool = True,
) -> list[Waypoint]:
    """슬롯 → 후면주차 waypoint 목록 (슬롯 안까지 후진으로 넣는다).

    phase 는 기존 것만 쓴다 (`CRUISE/APPROACH/ALIGN/ENTRY/FINAL`).
    펌웨어 `parse_phase` 가 아는 이름이 이 다섯뿐이라 새 phase 를 만들면
    wire 에서 거절된다.

    heading 요구는 **REVERSE_START 와 FINAL 에만** 건다. 중간 원호점까지
    heading 을 요구하면, 도착 반경 안에서 각도가 안 맞을 때 제어기가
    HEADING_OUT_OF_TOLERANCE 로 정지해 원호 중간에 서 버린다.

    Raises:
        InfeasibleRouteError: strict 이고 실현 가능한 후보가 없을 때.
    """
    plan, cands = choose_rear_parking_plan(
        slot, aisle_y=aisle_y, min_radius_mm=min_radius_mm, from_pose=from_pose,
        step_deg=step_deg)
    if plan is None:
        best = min(cands, key=lambda c: (c.overflow_mm, c.aisle_offset_mm))
        if strict:
            raise InfeasibleRouteError(slot.slot_id, best.reason)
        return []

    wps: list[Waypoint] = []
    wp_id = 1

    def add(phase: str, x: float, y: float, hdg: float | None, **kw) -> None:
        nonlocal wp_id
        wps.append(_make(route_id, wp_id, phase, x, y, hdg, **kw))
        wp_id += 1

    sx, sy, setup_heading = plan.setup_start
    direction = 1.0 if plan.side > 0 else -1.0

    # 주행 차선은 통로 중심이 아니라 **setup 시작점의 y** 다. 이 값이 통로에서
    # 얼마나 벗어나는지는 plan_rear_parking 이 이미 걸러 놨다(ON_AISLE_TOLERANCE_MM).
    lane_y = sy

    # ─ CRUISE: 현재 위치에서 차선을 따라 setup 시작점까지 ─
    start_x = from_pose[0] if from_pose is not None else None
    if start_x is not None and (sx - start_x) * direction <= 0.0:
        # setup 시작점을 이미 지나쳤다. 전진으로는 되돌아올 수 없다.
        if strict:
            raise InfeasibleRouteError(
                slot.slot_id,
                f"setup 시작점 x={sx:.0f} 을 이미 지나쳤다 (현재 x={start_x:.0f})")
        start_x = None

    # APPROACH 는 **선회 시작점 자체**다. 인계 모델처럼 목표 앞에 따로 감속점을
    # 두지 않는다 — 여기서 감속이 끝나야 원호에 올라탈 수 있고, 활주로가
    # 300mm 남짓이라 감속점을 더 앞에 두면 출발과 겹쳐 단계가 건너뛰어진다.
    if start_x is not None:
        cruise_run = sx - start_x
        n = int(abs(cruise_run) // AISLE_SEGMENT_MM)
        for i in range(1, n + 1):
            # 차선 구간도 APPROACH 로 낸다. CRUISE 로 두면 그 구간만 일반 주행
            # 상한(+정지마찰 예외)이 걸려 선회 시작 직전에 throttle 이 0.70 까지
            # 튄다. 후면주차 경로는 전 구간이 정밀 주차 상한 아래에 있어야 한다.
            add("APPROACH", start_x + cruise_run * (i / (n + 1)), lane_y, None)

    add("APPROACH", sx, lane_y, setup_heading, heading_required=True,
        heading_tolerance_deg=REVERSE_START_HEADING_TOLERANCE_DEG,
        capture_tolerance_cm=PHASE_DEFAULTS["APPROACH"]["position_tolerance_cm"] + 4.0)

    # ─ ALIGN: 전진 setup 원호. 마지막이 REVERSE_START (heading 필수) ─
    # 곡률 부호는 원호를 만들 때 쓴 회전방향 그대로다 (+1 = 좌회전).
    k_setup = plan.turn_setup / plan.radius_mm
    for i, (x, y, h) in enumerate(plan.align_poses):
        last = i == len(plan.align_poses) - 1
        # REVERSE_START 는 뒤따르는 후진 원호 전체의 기준점이다. 여기서 남긴
        # 오차는 원호 내내 줄일 수 없다 — 계획 반경(R)과 물리 최소 반경의
        # 차이(약 13%)보다 큰 오차로 들어가면 되잡을 방법이 없기 때문이다.
        # 그래서 일반 ALIGN 보다 훨씬 좁게 잡고, 못 맞추면 여기서 선다.
        add("ALIGN", x, y, h, heading_required=last, curvature=k_setup,
            path_capture_tolerance_cm=REAR_ENTRY_CAPTURE_MM / 10.0,
            position_tolerance_cm=(REVERSE_START_TOLERANCE_CM if last else None),
            heading_tolerance_deg=(REVERSE_START_HEADING_TOLERANCE_DEG
                                   if last else None))

    # ─ ENTRY: 후진 원호 ─
    # 후진에서도 dθ = curvature × ds 정의를 그대로 쓴다. ds 가 음수라
    # 부호가 자동으로 맞으므로 전진과 같은 turn 부호를 그대로 싣는다.
    k_entry = plan.turn_reverse / plan.radius_mm
    for x, y, h in plan.entry_poses:
        add("ENTRY", x, y, h, heading_required=False,
            motion_direction="REVERSE", curvature=k_entry,
            path_capture_tolerance_cm=REAR_ENTRY_CAPTURE_MM / 10.0)

    # ─ FINAL: 슬롯 중심 + rear heading, 후진 ─
    # 곡률 0 — 원호가 끝나는 지점이라 feedforward 가 남으면 과회전한다.
    # 여기서는 위치/heading 되먹임만으로 자세를 맞춘다.
    fx, fy, fh = plan.final_pose
    add("FINAL", fx, fy, fh, is_final=True, motion_direction="REVERSE")
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
