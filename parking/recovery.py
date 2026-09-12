"""후진 복구 경로 생성기.

Ackermann 차량은 제자리 회전을 못 한다. 그래서 "위치는 닿았는데 방향이 안
맞는" 상황(REPLAN_REQUIRED)이나 APPROACH 목표를 지나쳐 버린 상황
(APPROACH_COARSE_MISSED / APPROACH_FINE_MISSED)에서는 앞으로 아무리 꺾어도
목표를 다시 잡을 수 없다. 유일한 해법은 **물러났다가 다시 들어가는 것**이다.

판정 기준 — 비홀로노믹 사각지대
================================
현재 자세에서 차량이 그릴 수 있는 최소 선회원은 좌·우 두 개다::

        C_L = pose + R·(진행방향 왼쪽 수직단위벡터)
        C_R = pose + R·(진행방향 오른쪽 수직단위벡터)

목표가 이 두 원 **안쪽**에 있으면 어떤 전진 경로로도 닿을 수 없다. 최대로
꺾어도 원 위를 돌 뿐 원 안으로는 들어가지 못하기 때문이다. 이게
`forward_unreachable()` 이고, 후진을 걸지 말지의 유일한 기준이다.

"지나쳤다" / "각도가 크다" 같은 휴리스틱과 달리 오판이 없다 — 도달 가능한데
후진하거나, 도달 불가능한데 계속 전진하는 경우가 생기지 않는다.

안전 게이트
-----------
복구 경로를 만들어도 실제로 후진이 나가려면 제어기 쪽 게이트를 통과해야 한다:

- ``ControllerConfig.allow_reverse`` 가 True
- waypoint phase 가 ``ControllerConfig.reverse_allowed_phases`` 에 포함
  (기본 RECOVERY/PARKING/ALIGN/ENTRY/FINAL + 우리가 추가한 APPROACH)
- ``HostWaypointMission.max_recovery_attempts`` (기본 3회) 미만

즉 CRUISE/TURN 중에는 이 모듈이 경로를 만들어도 제어기가 거부한다. 통로
주행 중 후진은 의도적으로 막혀 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from parking.waypoints import (
    MIN_TURN_RADIUS_MM,
    PHASE_DEFAULTS,
    Waypoint,
)

__all__ = [
    "RecoveryDecision",
    "forward_unreachable",
    "plan_reverse_recovery",
    "REVERSE_TRIGGER_REASONS",
]

# 후진 복구를 검토할 실패 사유. host_control 이 mission.request_replan(reason)
# 으로 올려주는 문자열과 같아야 한다.
#   - ALIGN_HEADING_*  : 위치는 닿았는데 heading 불일치 (제자리 회전 불가)
#   - APPROACH_*_MISSED: 목표를 지나쳐 pass-line 을 넘김
REVERSE_TRIGGER_REASONS: frozenset[str] = frozenset({
    "HEADING_OUT_OF_TOLERANCE",     # mission.notify_result: ControlMode.ALIGN
    "APPROACH_COARSE_MISSED",       # approach_guard: COARSE 포착 실패
    "APPROACH_FINE_MISSED",         # approach_guard: FINE 정밀 실패
    "PATH_DEVIATION",               # 파이프라인: 목표가 전진 사각지대에 들어감
})

# 최소 후진 거리. 실차에서 27cm 는 너무 짧아 다시 들어갈 활주로가 안 나왔다
# (2026-08-13). 카메라 위치 오차 9~11cm 도 감안해 50cm 로 잡는다.
MIN_BACKUP_MM: float = 500.0
# 최대 후진 거리. 통로를 벗어날 만큼 물러나면 안 된다.
MAX_BACKUP_MM: float = 700.0
# 후진 후 목표가 이만큼은 앞에 있어야 "다시 전진으로 잡을 수 있다"고 본다.
MIN_AHEAD_MM: float = 80.0
# 이 각도를 넘는 방향 오차만 후진 대상으로 본다.
# **후진 거리(MIN_BACKUP_MM)와 별개 개념이다.** 예전에는 "필요 거리가
# 최소 후진 거리보다 작으면 됐다"로 판단했는데, 최소 거리를 50cm 로 올리자
# 47도 오차까지 "괜찮다"가 되어 복구가 아예 안 걸렸다.
HEADING_TRIGGER_DEG: float = 5.0


@dataclass(frozen=True)
class RecoveryDecision:
    """후진 복구 판단 결과."""

    needed: bool
    reason: str
    backup_mm: float = 0.0
    # 후진 후 도달할 지점 (mm). needed=False 면 의미 없다.
    backup_point: tuple[float, float] = (0.0, 0.0)


def _turn_centers(x: float, y: float, heading_deg: float, radius_mm: float
                  ) -> tuple[tuple[float, float], tuple[float, float]]:
    """현재 자세의 좌/우 최소 선회원 중심."""
    h = math.radians(heading_deg)
    # 진행방향 왼쪽 = heading + 90°, 오른쪽 = heading - 90°
    left = (x + radius_mm * math.cos(h + math.pi / 2),
            y + radius_mm * math.sin(h + math.pi / 2))
    right = (x + radius_mm * math.cos(h - math.pi / 2),
             y + radius_mm * math.sin(h - math.pi / 2))
    return left, right


def forward_unreachable(pose_mm: tuple[float, float], heading_deg: float,
                        target_mm: tuple[float, float], *,
                        radius_mm: float = MIN_TURN_RADIUS_MM,
                        margin_mm: float = 0.0,
                        lot_mm: tuple[float, float] | None = None) -> bool:
    """전진만으로 목표에 닿을 수 없는가.

    두 가지를 본다.

    1. **사각지대** — 목표가 좌/우 최소 선회원 안에 있으면 아무리 꺾어도
       원 위를 돌 뿐 안으로 들어가지 못한다.
    2. **지나침** — 목표가 등 뒤에 있으면 한 바퀴 돌아 와야 하는데, 그 원의
       지름(2R = 114cm)이 맵 한 변(120cm)과 거의 같다. `lot_mm` 을 주면
       루프가 맵에 들어가는지까지 본다. 이게 없으면 "이론상 도달 가능"이라
       판정돼 지나친 차가 영영 후진하지 않는다.
    """
    left, right = _turn_centers(pose_mm[0], pose_mm[1], heading_deg, radius_mm)
    limit = radius_mm - margin_mm
    for cx, cy in (left, right):
        if math.hypot(target_mm[0] - cx, target_mm[1] - cy) < limit:
            return True

    if lot_mm is not None:
        h = math.radians(heading_deg)
        ahead = ((target_mm[0] - pose_mm[0]) * math.cos(h)
                 + (target_mm[1] - pose_mm[1]) * math.sin(h))
        if ahead < -abs(margin_mm):
            # 등 뒤다. 되돌아오려면 선회원을 한 바퀴 돌아야 하는데, 그 원이
            # **현재 위치 기준으로** 맵 안에 들어가야 한다. 지름만 비교하면
            # 안 된다 — 지름 114cm 는 맵 120cm 보다 작지만, 벽 가까이에서는
            # 원이 맵 밖으로 나간다.
            w, ht = lot_mm
            fits = any(cx - radius_mm >= 0.0 and cx + radius_mm <= w
                       and cy - radius_mm >= 0.0 and cy + radius_mm <= ht
                       for cx, cy in (left, right))
            if not fits:
                return True
    return False


def _backup_distance(pose_mm: tuple[float, float], heading_deg: float,
                     target_mm: tuple[float, float], radius_mm: float,
                     lot_mm: tuple[float, float] | None = None) -> float | None:
    """사각지대를 벗어날 때까지 필요한 최소 후진 거리.

    현재 heading 을 유지한 채 뒤로 물러나며 사각지대 판정을 다시 한다.
    직선 후진이라 해석해가 있긴 하지만, 25mm 씩 훑는 편이 경계 조건
    (두 원 중 하나만 걸린 경우 등)을 놓치지 않는다.
    """
    h = math.radians(heading_deg)
    cos_h, sin_h = math.cos(h), math.sin(h)
    step = 25.0
    d = MIN_BACKUP_MM
    while d <= MAX_BACKUP_MM:
        bx = pose_mm[0] - d * cos_h
        by = pose_mm[1] - d * sin_h
        ahead = (target_mm[0] - bx) * cos_h + (target_mm[1] - by) * sin_h
        # 두 조건을 모두 만족해야 다시 전진으로 잡을 수 있다.
        #
        #   1. 목표가 **앞쪽**에 와야 한다.
        #   2. 좌/우 선회원 **밖**이어야 한다.
        #
        # 여기에 반경 여유(margin)를 더하면 안 된다. 목표가 거의 일직선 앞에
        # 있으면 선회원 중심까지 거리가 sqrt(d^2+R^2) 라 R 을 조금만 넘는데,
        # 여유를 얹으면 아무리 물러나도 "사각지대"를 못 벗어난다고 나온다
        # (실측으로 잡힌 버그 — 후진 거리 계산이 늘 실패했다).
        if ahead >= MIN_AHEAD_MM and not forward_unreachable(
                (bx, by), heading_deg, target_mm, radius_mm=radius_mm):
            return d
        d += step
    return None


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def decide(pose_mm: tuple[float, float], heading_deg: float | None,
           target_mm: tuple[float, float], *, reason: str = "",
           target_heading_deg: float | None = None,
           radius_mm: float = MIN_TURN_RADIUS_MM,
           lot_mm: tuple[float, float] | None = None) -> RecoveryDecision:
    """후진이 필요한지, 필요하면 어디로 얼마나 물러나야 하는지 판단한다.

    실패 유형이 둘이고 물러나는 **방향이 서로 다르다**.

    1. 방향 불일치 (HEADING_OUT_OF_TOLERANCE)
       목표 위에 서 있는데 각도만 틀린 경우다. 거리가 0 이라 사각지대 판정에
       걸리지 않고, 현재 heading 으로 직선 후진해봐야 각도는 그대로다.
       **목표 heading 축을 따라** 뒤로 물러나야 다시 들어올 활주로가 생긴다.
       필요 거리는 조향으로 Δθ 를 만들 거리 d ≈ R·Δθ 다.

    2. 목표 지나침 (APPROACH_*_MISSED)
       목표가 선회원 안으로 들어가 버린 경우다. 현재 heading 방향으로
       사각지대를 벗어날 때까지 물러난다.
    """
    if heading_deg is None:
        # 방향을 모르면 선회원을 그릴 수 없다. 후진은 방향을 아는 상태에서만
        # 안전하다 — 모르면 하지 않는다.
        return RecoveryDecision(False, "NO_HEADING")

    # ─ 후진은 **현재 heading 축을 따라 곧게** 물러난다 (11자) ─
    # 물러날 거리는 두 요구의 큰 쪽이다.
    #   ① 방향을 Δθ 만큼 되돌릴 활주로   d ≈ R·Δθ
    #   ② 목표를 다시 전진 사각지대 밖으로 빼낼 거리
    need = 0.0
    why = reason or "DEAD_ZONE"
    heading_bad = False
    if target_heading_deg is not None:
        err = abs(_wrap180(heading_deg - target_heading_deg))
        if err > HEADING_TRIGGER_DEG:
            heading_bad = True
            need = math.radians(err) * radius_mm
            why = reason or "HEADING_MISMATCH"

    if forward_unreachable(pose_mm, heading_deg, target_mm,
                           radius_mm=radius_mm, lot_mm=lot_mm):
        clear = _backup_distance(pose_mm, heading_deg, target_mm, radius_mm, lot_mm)
        if clear is None:
            return RecoveryDecision(False, "BACKUP_INSUFFICIENT")
        need = max(need, clear)
    elif not heading_bad:
        # 전진으로 닿고 방향도 맞으면 후진할 이유가 없다.
        return RecoveryDecision(False, "FORWARD_REACHABLE")

    backup = min(MAX_BACKUP_MM, max(MIN_BACKUP_MM, need))
    h = math.radians(heading_deg)
    point = (pose_mm[0] - backup * math.cos(h),
             pose_mm[1] - backup * math.sin(h))
    return RecoveryDecision(True, why, backup, point)


def _target_xy(wp: Any) -> tuple[float, float]:
    """backend Waypoint(x/y) 와 controller Waypoint(x_mm/y_mm) 를 모두 받는다.

    미션이 들고 있는 건 adapter 가 변환한 controller 쪽 스키마라,
    실패한 target 을 그대로 넘기면 필드 이름이 다르다.
    """
    if hasattr(wp, "x_mm"):
        return float(wp.x_mm), float(wp.y_mm)
    return float(wp.x), float(wp.y)


def plan_reverse_recovery(pose_mm: tuple[float, float], heading_deg: float | None,
                          failed_target: Any, *, route_id: int,
                          reason: str = "",
                          radius_mm: float = MIN_TURN_RADIUS_MM,
                          bounds_mm: tuple[float, float] | None = None,
                          ) -> list[Waypoint] | None:
    """실패한 target 을 다시 잡기 위한 후진 waypoint 를 만든다.

    반환한 목록은 ``AutoHostRunner.load_recovery_waypoints()`` 에 그대로
    넘긴다. 미션이 복구 경로를 앞에 끼워 넣고, 다 마치면 실패했던
    ``failed_target`` 부터 원래 route 를 자동으로 이어간다. 따라서 여기서
    원래 경로를 복제할 필요가 없다 — **후진 한 점만** 만들면 된다.

    Returns:
        후진 waypoint 목록. 후진이 불필요하거나 불가능하면 None.
    """
    decision = decide(pose_mm, heading_deg, _target_xy(failed_target),
                      reason=reason, radius_mm=radius_mm, lot_mm=bounds_mm,
                      target_heading_deg=(failed_target.target_heading_deg
                                          if failed_target.heading_required else None))
    if not decision.needed:
        return None

    bx, by = decision.backup_point
    if bounds_mm is not None:
        w, h = bounds_mm
        # 후진해서 맵 밖으로 나가면 안 된다.
        if not (0.0 <= bx <= w and 0.0 <= by <= h):
            return None

    p = PHASE_DEFAULTS["RECOVERY"]
    return [Waypoint(
        route_id=route_id, waypoint_id=1, phase="RECOVERY",
        x=bx, y=by, target_heading_deg=None,
        speed_cm_s=p["speed_cm_s"],
        position_tolerance_cm=p["position_tolerance_cm"],
        heading_tolerance_deg=p["heading_tolerance_deg"],
        heading_required=False,
        is_final=False,               # 복구 waypoint 는 is_final 금지 (미션 계약)
        capture_tolerance_cm=None,
        motion_direction="REVERSE",
    )]
