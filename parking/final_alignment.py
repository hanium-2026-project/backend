"""Slot-local final pose evaluation, alignment, and straight rear entry.

후면주차 route 가 끝났다고 PARKED 가 아니다. waypoint 에 도착해도 차체가
주차선과 비스듬하면 그대로 확정하면 안 된다 (실측: ENTRY/FINAL 의 heading
허용오차 12° 는 아래 기하상 **횡방향 여유를 전부 소진**한다).

기하 (실측 바닥판):
    슬롯 200mm(폭) x 300mm(깊이),  차량 250mm(길이) x 150mm(폭)
    → 완벽히 정렬해도 좌우 여유는 각 25mm 뿐이다.

    heading 오차 θ 일 때 차체가 슬롯 폭 방향으로 차지하는 반폭:
        125·sin|θ| + 75·cos|θ|
      θ= 0°  → 75.0mm  (여유 25.0)
      θ= 5°  → 85.6mm  (여유 14.4)
      θ=10°  → 95.6mm  (여유  4.4)
      θ=12°  → 99.4mm  (여유  0.6)   ← 횡방향 오차가 0 일 때만 성립
      θ=13°  → 101.2mm (넘침)

따라서 "허용 오차" 를 임의로 정하지 않는다. 포함 여부는 **실제 footprint
다각형과 슬롯 다각형**으로 판정하고, 측정 불확실성만 기존 boundary 정책과
같은 방식으로 별도 band 로 둔다.

좌표계: 판정은 slot-local frame 에서 한다 — 슬롯 축 방향이 바뀌어도 같은
로직을 쓸 수 있다.
    depth   : 슬롯 안쪽으로 들어간 정도 (+ = 더 깊이)
    lateral : 슬롯 중심선에서의 좌우 오차
    heading : 후면주차 완료 자세(코가 통로를 향함) 대비 오차
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .waypoints import CAR_LENGTH_MM, CAR_WIDTH_MM, SlotSpec, _car_footprint

# 후면주차 완료 자세에서 차체 중심이 슬롯 중심에 오도록 한다.
# 차량 250mm / 슬롯 깊이 300mm 이므로 앞뒤 여유가 각 25mm 로 같아진다.
TARGET_DEPTH_MM: float = 0.0

# 판정 전용 측정 불확실성. 기존 boundary 정책의
# boundary_measurement_uncertainty_mm 과 같은 성격이며, planner 기하를
# 넓히는 값이 아니다.
POSE_UNCERTAINTY_MM: float = 10.0

# 이 값보다 heading 이 틀어져 있으면 직선 후진으로는 회복되지 않는다고 보고
# FINAL_ALIGNMENT 로 보낸다. 위 표에서 유도했다 — 횡오차를 15mm 쯤 허용하려면
# 125·sinθ + 75·cosθ <= 85 이어야 하고 그 해가 약 4.7° 다.
ALIGNED_HEADING_TOLERANCE_DEG: float = 4.5

# 최종 단계에서 다루는 횡오차 band.
#
# 제어기는 목표점을 추종하므로 횡오차는 줄어들지만, 그 대가로 **끝 자세의
# heading 이 틀어진다**. closed-loop 실측(staging 275mm 구간, 목표점을 슬롯
# 중심선에 둔 뒤):
#     시작 횡오차  →  끝 횡오차 / 끝 heading 오차
#        40mm      →   13.0mm /  10.9°   ← PARKED 4.5° 불합격
#        20mm      →    3.9mm /   4.1°   ← 합격
#        10mm      →    3.3mm /   2.7°   ← 합격
#         5mm      →    1.6mm /   1.4°
#
# 20mm 이 실측상 한계다. 정렬 목표와 직선후진 수용 한계를 **같은 값**으로
# 묶는다 — 정렬이 이 band 안으로 데려오고 직선후진이 그 band 를 흡수한다.
# 둘이 어긋나면 "정렬 → 아직 크다 → 다시 정렬" 로 진동한다.
FINAL_LATERAL_BAND_MM: float = 20.0
STRAIGHT_REVERSE_LATERAL_LIMIT_MM: float = FINAL_LATERAL_BAND_MM

# 제어기가 목표 앞에서 throttle 을 0 으로 내리는 거리
# (ControllerConfig.stop_distance_cm = 3.0cm). 직선 후진 목표점을 이만큼
# 더 안쪽으로 겨눠 실제 정지 위치가 목표 깊이에 오게 한다.
STOP_DISTANCE_COMPENSATION_MM: float = 30.0


def rear_parked_heading_deg(slot: SlotSpec) -> float:
    """후면주차 완료 시 차체 방향.

    SlotSpec.target_heading_deg 는 **진입 방향**이다. 후진으로 넣으면 차는
    코를 통로 쪽으로 두고 서므로 정확히 반대다 (실측 route: B1 FINAL 270°,
    SlotSpec.target_heading_deg 90°).
    """
    return (float(slot.target_heading_deg) + 180.0) % 360.0


def heading_error_deg(heading_deg: float, slot: SlotSpec) -> float:
    """후면주차 완료 자세 대비 부호 있는 heading 오차 (-180, 180]."""
    return (float(heading_deg) - rear_parked_heading_deg(slot) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class SlotLocalPose:
    """슬롯 축 기준으로 표현한 차량 자세."""

    depth_mm: float          # + = 슬롯 안쪽으로 더 깊이
    lateral_mm: float        # + = 완료 자세 기준 왼쪽
    heading_err_deg: float   # 완료 자세 대비 오차


def to_slot_local(slot: SlotSpec, x_mm: float, y_mm: float,
                  heading_deg: float) -> SlotLocalPose:
    """Map a map-frame pose into the slot's own frame."""
    parked = math.radians(rear_parked_heading_deg(slot))
    # 완료 자세에서 코가 향하는 방향 = 슬롯에서 나오는 방향. 깊이는 그 반대.
    out_x, out_y = math.cos(parked), math.sin(parked)
    dx = float(x_mm) - float(slot.center_x)
    dy = float(y_mm) - float(slot.center_y)
    depth = -(dx * out_x + dy * out_y)
    # 나오는 방향을 +x 로 봤을 때의 왼쪽 = (-out_y, out_x)
    lateral = dx * (-out_y) + dy * out_x
    return SlotLocalPose(depth_mm=depth, lateral_mm=lateral,
                         heading_err_deg=heading_error_deg(heading_deg, slot))


def slot_polygon(slot: SlotSpec) -> list[tuple[float, float]]:
    x0, x1 = slot.center_x - slot.width / 2.0, slot.center_x + slot.width / 2.0
    y0, y1 = slot.center_y - slot.length / 2.0, slot.center_y + slot.length / 2.0
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def footprint_overflow_mm(slot: SlotSpec, x_mm: float, y_mm: float,
                          heading_deg: float) -> tuple[float, float]:
    """Corner overflow out of the slot, split by slot axis (0 = inside).

    가로/세로를 나눠 돌려준다. 둘을 합치면 안 된다 — 덜 들어간 차는 슬롯
    **앞쪽**으로 넘치는데 그건 직선 후진이 고치는 것이고, 옆으로 넘치는 것은
    후진으로 절대 고쳐지지 않아 재정렬이 필요하다.

    슬롯 축이 맵 축과 나란하다는 사실을 쓰지 않고 slot-local frame 에서 재므로
    다른 방향의 슬롯에도 그대로 적용된다.
    """
    half_w = slot.width / 2.0
    half_l = slot.length / 2.0
    lateral = depth = 0.0
    for cx, cy in _car_footprint(float(x_mm), float(y_mm), float(heading_deg)):
        local = to_slot_local(slot, cx, cy, heading_deg)
        lateral = max(lateral, abs(local.lateral_mm) - half_w)
        depth = max(depth, abs(local.depth_mm) - half_l)
    return max(0.0, lateral), max(0.0, depth)


def in_final_region(slot: SlotSpec, x_mm: float, y_mm: float) -> bool:
    """Whether the vehicle is close enough that the *final* maneuver applies.

    슬롯 ID 나 waypoint 이름(``wp8/8``, ``phase == FINAL``)이 아니라 기하로
    판정한다. 경로가 어디까지 진행됐는지와 무관하게, 차가 물리적으로 슬롯
    입구권에 들어와 있으면 "지금 필요한 것은 재접근이 아니라 최종 정렬" 이다.

    범위는 슬롯 치수에서 유도한다 — 좌표 하드코딩 없음:
        depth   >= -slot.length/2                 차체 중심이 슬롯 사각형 안
        |lateral| <= slot.width/2                 차체 중심이 슬롯 폭 안

    depth 하한이 왜 슬롯 입구(중심 기준 -length/2)인가:

    예전에는 "입구 앞 한 대 길이" 까지(-(length/2 + CAR_LENGTH) = -400mm)
    허용했다. 그런데 B1 기준 depth -400mm 은 y=650 이고, 통로 주행 밴드
    (y 325~875) 한가운데다. 그래서 **슬롯 x 대역을 지나가는 통로 주행 차량이
    "최종 정렬 중" 으로 분류**됐다.

    실측 run_20260903_032539: 통로 (376,626) 에서 heading 오차 129도 인 차가
    이 판정을 통과해 FINAL_EVAL -> ALIGN -> NO_SAFE_FINAL_ALIGNMENT x3 ->
    FINAL_ALIGNMENT_EXHAUSTED 로 미션이 끝났다. 그 자세에 필요한 것은 최종
    정렬이 아니라 재접근이었다.

    기록된 실제 FINAL_POSE_EVAL depth 분포가 두 무리로 확실히 갈린다:
        정상(슬롯 안/입구): -11.0, -8.7, -1.2, +5.6, +55.5
        오분류(통로):       -247.6, -397.2, -397.2, -398.6  (heading 오차 60~130도)
    -150mm 는 그 사이를 통과하며 정상 쪽에 139mm 여유를 남긴다.

    횡방향은 슬롯 반폭으로 조인다. 실측 슬롯 간격이 225mm 이므로 이보다 넓게
    잡으면 **옆 슬롯에 있는 차를 이 슬롯의 최종 구간으로 오인**한다 (반폭
    100mm < 225/2). 옆 슬롯까지 밀린 차는 최종 정렬이 아니라 재접근 대상이다.

    **heading 을 쓰지 않는다.** depth/lateral 은 슬롯 축과 위치만으로 정해지므로
    heading 이 LAST_VALID 로 굳어 있어도 이 판정은 오염되지 않는다. 정렬을
    실제로 계획할 때 trusted heading 을 요구하는 것은 호출부의 몫이다.
    """
    local = to_slot_local(slot, x_mm, y_mm, rear_parked_heading_deg(slot))
    if local.depth_mm < -slot.length / 2.0:
        return False
    return abs(local.lateral_mm) <= slot.width / 2.0


@dataclass(frozen=True)
class FinalPoseVerdict:
    """What the vehicle should do next from its current final pose."""

    action: str              # PARKED_OK | STRAIGHT_REVERSE | ALIGN
    local: SlotLocalPose
    lateral_overflow_mm: float
    depth_overflow_mm: float
    reason: str = ""

    @property
    def parked(self) -> bool:
        return self.action == "PARKED_OK"


def evaluate_final_pose(
    slot: SlotSpec, x_mm: float, y_mm: float, heading_deg: float, *,
    heading_tolerance_deg: float = ALIGNED_HEADING_TOLERANCE_DEG,
    lateral_limit_mm: float = STRAIGHT_REVERSE_LATERAL_LIMIT_MM,
    uncertainty_mm: float = POSE_UNCERTAINTY_MM,
    target_depth_mm: float = TARGET_DEPTH_MM,
) -> FinalPoseVerdict:
    """Decide PARKED / straight reverse / realignment from one fresh pose.

    - heading 이 틀어져 있으면 깊이와 무관하게 ALIGN. 슬롯 깊숙이 들어간 채로
      크게 조향하면 차체가 라인을 넘거나 옆 슬롯을 침범한다.
    - 실제 footprint 가 측정 불확실성보다 더 슬롯을 넘으면 ALIGN.
    - heading 은 맞았는데 덜 들어갔으면, 20mm acquisition band 안에서만
      STRAIGHT_REVERSE. 이 band 는 correction 가능성 조건이지 PARKED veto 가
      아니다.
    - 깊이와 footprint 가 충분하면 center offset 자체로 다시 거절하지 않는다.
    """
    local = to_slot_local(slot, x_mm, y_mm, heading_deg)
    lateral_over, depth_over = footprint_overflow_mm(
        slot, x_mm, y_mm, heading_deg)

    def verdict(action: str, reason: str = "") -> FinalPoseVerdict:
        return FinalPoseVerdict(action, local, lateral_over, depth_over,
                                reason=reason)

    if abs(local.heading_err_deg) > heading_tolerance_deg:
        # 슬롯 깊숙이 들어간 채로 크게 조향하면 옆 슬롯을 침범한다.
        # 나왔다가 다시 곧게 들어가는 것이 유일하게 안전한 회복이다.
        return verdict("ALIGN", "HEADING_NOT_PARALLEL")
    if lateral_over > uncertainty_mm:
        # 실제 차체가 measurement band 보다 더 슬롯 옆선을 넘었다.
        return verdict("ALIGN", "LATERAL_OFFSET")
    # 깊이 여유는 **기하값**이다 — 차량 250mm 가 슬롯 깊이 300mm 안에 있으면
    # 앞뒤로 각 25mm 가 남는다. 자세 측정 불확실성(10mm)은 여기에 쓸 값이
    # 아니다. 실제 포함 여부는 아래 depth_over(정확한 footprint)가 판정한다.
    depth_margin = max(0.0, (slot.length - CAR_LENGTH_MM) / 2.0)
    if local.depth_mm < target_depth_mm - depth_margin:
        if abs(local.lateral_mm) > lateral_limit_mm:
            # 아직 덜 들어간 차량은 직선 후진을 해야 한다. 이때만 center
            # lateral band 가 correction-acquisition 조건으로 필요하다.
            return verdict("ALIGN", "LATERAL_OFFSET")
        return verdict("STRAIGHT_REVERSE", "NOT_FULLY_ENTERED")
    if depth_over > uncertainty_mm:
        # 너무 깊이 들어갔다 — 뒤쪽 라인을 넘었다. 나와서 다시 잡는다.
        return verdict("ALIGN", "TOO_DEEP")
    return verdict("PARKED_OK")


def alignment_staging_pose(slot: SlotSpec) -> tuple[float, float, float]:
    """Pose to line up in front of the slot before the final straight reverse.

    슬롯 중심선 위, 차체 뒷면이 슬롯 입구에 닿는 지점이다 — 여기서부터는
    조향 없이 곧게 후진하면 차가 슬롯 축과 나란히 들어간다.
    """
    parked = rear_parked_heading_deg(slot)
    rad = math.radians(parked)
    out_x, out_y = math.cos(rad), math.sin(rad)
    # 슬롯 입구(통로 쪽)까지 slot.length/2, 거기서 차체 반길이만큼 더 나온다.
    offset = slot.length / 2.0 + CAR_LENGTH_MM / 2.0
    return (slot.center_x + out_x * offset,
            slot.center_y + out_y * offset,
            parked)


def straight_reverse_distance_mm(slot: SlotSpec, x_mm: float, y_mm: float,
                                 *, target_depth_mm: float = TARGET_DEPTH_MM
                                 ) -> float:
    """How far to back straight in from the current pose (mm, >= 0)."""
    local = to_slot_local(slot, x_mm, y_mm, rear_parked_heading_deg(slot))
    return max(0.0, target_depth_mm - local.depth_mm)


# ─── 정렬 기동 ───────────────────────────────────────────────────────────────
# staging 자세에 얼마나 가까우면 "정렬됐다" 고 볼지.
#
# heading 은 PARKED 판정보다 **더 엄격**해야 한다. 같거나 느슨하면
# 정렬 → 재평가 → 다시 정렬 로 진동한다. 절반으로 잡는다.
ALIGN_GOAL_HEADING_DEG: float = ALIGNED_HEADING_TOLERANCE_DEG / 2.0
# FINAL_ALIGNMENT 종점은 일반 RECOVERY의 80mm 도착 반경을 쓸 수 없다.
# postcondition인 straight-reverse acquisition band(20mm)를 보장하도록:
#
#   |planned lateral| + terminal position tolerance <= 20mm
#
# 측정 불확실성 10mm를 terminal 위치 허용오차로 쓰고, 탐색 목표는 남은
# 10mm 안쪽으로 제한한다. 둘 다 기존 기하/측정 계약에서 유도된 값이다.
ALIGN_TERMINAL_POSITION_TOLERANCE_MM: float = POSE_UNCERTAINTY_MM
ALIGN_GOAL_LATERAL_MM: float = max(
    0.0, STRAIGHT_REVERSE_LATERAL_LIMIT_MM
    - ALIGN_TERMINAL_POSITION_TOLERANCE_MM)
# 계획 종점 자체가 ALIGN_GOAL_HEADING_DEG 이내이고, 실제 도착 오차까지
# 합쳐 PARKED heading 4.5deg 안에 남도록 한다.
ALIGN_TERMINAL_HEADING_TOLERANCE_DEG: float = (
    ALIGNED_HEADING_TOLERANCE_DEG - ALIGN_GOAL_HEADING_DEG)
# 정렬 종점이 슬롯 기준 어디까지 허용되는가 (staging depth 기준 상대값).
#
# 예전 값 (-200, +100) 은 staging(-275) 기준 절대 [-475, -175], B1 로는
# y [575, 875] 였다. 통로 중심이 y=600 이므로 **허용 영역의 먼 쪽 절반이
# 통로 안**이고, 먼 끝(575)은 통로 중심을 25mm 지난다. 국소 보정의 목적지로
# 통로를 허용한 것이다.
#
# 실측 그대로 나타났다:
#     run_20260903_193043  슬롯 안 depth +60.9 에서 시작 -> depth -420.0
#                          (y=630, 통로중심에서 30mm)
#     run_20260903_193518  depth +56.7 -> depth -479.4
#                          (y=570.6, 통로중심을 29mm 지남)
#     그 다음 정렬 시도는 아예 통로 안에서 3점 선회를 만든다
#     (193043 route5 종점 y=492.9, 193518 route8 종점 y=471.0).
# 다섯 run(191010/191220/193043/193223/193518) 모두 같은 패턴이다.
#
# 두 경계 모두 기하에서 유도한다 — 튜닝 상수가 아니다.
#   0     : staging 자세 그 자체. 차체 뒷면이 정확히 슬롯 입구에 있다.
#           이보다 더 나갈 기하적 이유가 없다 — 뒷면이 입구를 벗어난
#           순간부터는 heading 만 맞추고 곧게 후진하면 된다.
#   +125  : CAR_LENGTH_MM/2. 차체 **중심**이 슬롯 입구에 오는 깊이다.
#           staging(-275) + 125 = -150 = -slot.length/2 이므로 이 끝은
#           in_final_region 의 depth 하한과 정확히 같다. 즉 정렬 종점이
#           항상 최종구간 판정 안에 남아 lifecycle 이 스스로 닫힌다.
#
# 결과적으로 허용 영역은 "차가 슬롯 입구에 걸쳐 있는 구간" 이고,
# B1 기준 y [775, 900] — 통로(600)에 닿지 않는다.
ALIGN_GOAL_DEPTH_BAND_MM: tuple[float, float] = (0.0, CAR_LENGTH_MM / 2.0)


def alignment_goal_test(slot: SlotSpec):
    """Goal predicate for the setup search: lined up in front of the slot."""
    staging = alignment_staging_pose(slot)
    staging_depth = to_slot_local(slot, staging[0], staging[1],
                                  staging[2]).depth_mm
    low, high = ALIGN_GOAL_DEPTH_BAND_MM

    def reached(pose: tuple[float, float, float]) -> bool:
        local = to_slot_local(slot, pose[0], pose[1], pose[2])
        if abs(local.heading_err_deg) > ALIGN_GOAL_HEADING_DEG:
            return False
        if abs(local.lateral_mm) > ALIGN_GOAL_LATERAL_MM:
            return False
        return (staging_depth + low) <= local.depth_mm <= (staging_depth + high)

    return reached


def plan_final_alignment(slot: SlotSpec, from_pose: tuple[float, float],
                         from_heading_deg: float, *,
                         obstacle_poses: tuple[tuple[float, float, float], ...] = (),
                         ):
    """Bounded maneuver that ends lined up in front of the slot.

    기존 setup/recovery 탐색기를 그대로 쓰고 **도달 조건만** 바꾼다 (§19).
    맵/슬롯/장애물 footprint 검사, 최소 실행거리, segment 상한이 전부 동일하게
    적용된다. 새 optimizer 를 만들지 않는다.
    """
    from .waypoints import plan_setup_recovery       # 순환 import 방지
    return plan_setup_recovery(
        slot, from_pose, from_heading_deg,
        obstacle_poses=obstacle_poses,
        goal_test=alignment_goal_test(slot))


def build_final_alignment_waypoints(slot: SlotSpec, route_id: int, *,
                                    from_pose: tuple[float, float],
                                    from_heading_deg: float,
                                    obstacle_poses: tuple[tuple[float, float, float], ...] = (),
                                    ) -> list:
    """FINAL_ALIGNMENT maneuver as waypoints.  Empty list if none is safe.

    탐색/중간 waypoint 는 기존 setup recovery 를 그대로 재사용한다. 마지막
    waypoint 만 postcondition에 맞는 위치/heading 허용오차로 바꾼다. 따라서
    일반 RECOVERY의 80mm 완료 계약은 전혀 바뀌지 않는다.
    """
    from .waypoints import build_setup_recovery_waypoints
    wps = build_setup_recovery_waypoints(
        slot, route_id, from_pose=from_pose,
        from_heading_deg=from_heading_deg,
        obstacle_poses=obstacle_poses,
        goal_test=alignment_goal_test(slot))
    if not wps:
        return []
    wps[-1] = replace(
        wps[-1],
        position_tolerance_cm=(
            ALIGN_TERMINAL_POSITION_TOLERANCE_MM / 10.0),
        heading_tolerance_deg=ALIGN_TERMINAL_HEADING_TOLERANCE_DEG,
        heading_required=True)
    return wps


def build_final_straight_reverse_waypoints(slot: SlotSpec, route_id: int, *,
                                           from_pose: tuple[float, float],
                                           target_depth_mm: float = TARGET_DEPTH_MM,
                                           ) -> list:
    """Pure straight reverse down the slot centreline.

    curvature 0 은 **기준 경로가 직선**이라는 뜻이다 (feedforward 조향 0).
    제어기의 PD 되먹임은 그대로 살아 있어야 한다 — RECOVERY 만 조향을 고정하고
    (``ControllerConfig.reverse_straight_phases``) FINAL 은 고정하지 않는다.

    목표점은 **슬롯 중심선 위의 목표 깊이 지점**이다. 현재 위치를 뒤로 민
    지점을 목표로 삼으면 안 된다 — 제어기가 목표점을 향해 추종하므로, 목표가
    현재 횡오차를 그대로 물려받으면 곧게 뒤로만 가고 **횡오차가 영원히 남는다**
    (closed-loop 확인: lat +40mm 로 출발해 lat +40mm 로 도착).
    """
    from .waypoints import PHASE_DEFAULTS, Waypoint

    parked = rear_parked_heading_deg(slot)
    distance = straight_reverse_distance_mm(slot, from_pose[0], from_pose[1],
                                            target_depth_mm=target_depth_mm)
    if distance <= 0.0:
        return []
    rad = math.radians(parked)
    # 완료 자세의 코 방향이 슬롯 밖이므로, 안으로 들어가는 방향은 그 반대다.
    into_x, into_y = -math.cos(rad), -math.sin(rad)
    defaults = PHASE_DEFAULTS["FINAL"]
    # 제어기는 목표에서 stop_distance 만큼 **앞에서** throttle 을 0 으로 내린다
    # (PoseWaypointController._throttle: remaining = distance - stop_distance).
    # 목표점을 슬롯 중심에 그대로 두면 차는 늘 그만큼 덜 들어가 멈추고,
    # PARKED 깊이 판정을 영원히 통과하지 못한다 (closed-loop: 28.9mm 부족).
    # 그래서 그 거리만큼 더 안쪽을 겨눈다 — 허용오차를 넓히는 것이 아니라
    # 제어기의 알려진 정지 거리를 보상하는 것이다.
    # 겨눔점 자체가 **유효한 주차 자세**여야 한다. 슬롯보다 깊은 점을 겨누면
    # 그 점의 footprint 가 맵 밖으로 나가 trajectory validator 가 경로를 통째로
    # 거절한다 (실측: aim y=1080 → MAP_FOOTPRINT REJECTED). 그래서 보상량을
    # 슬롯 깊이 여유로 잘라낸다.
    depth_margin = max(0.0, (slot.length - CAR_LENGTH_MM) / 2.0)
    aim_mm = min(target_depth_mm + STOP_DISTANCE_COMPENSATION_MM, depth_margin)
    end_x = slot.center_x + into_x * aim_mm
    end_y = slot.center_y + into_y * aim_mm
    return [Waypoint(
        route_id=route_id, waypoint_id=1, phase="FINAL",
        x=end_x, y=end_y, target_heading_deg=parked,
        speed_cm_s=float(defaults["speed_cm_s"]),
        position_tolerance_cm=float(defaults["position_tolerance_cm"]),
        heading_tolerance_deg=ALIGNED_HEADING_TOLERANCE_DEG,
        heading_required=True, is_final=True,
        motion_direction="REVERSE", curvature=0.0)]
