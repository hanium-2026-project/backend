"""제어기 설정 값.

두 종류로 분리한다.

1. FirmwareConstants
   실제 ESP32 펌웨어(app_config.example.h, actuator.c)에서 **관측된 사실**.
   부호/PWM 임계값 등. 임의로 바꾸지 말 것.

2. ControllerConfig
   host-side 제어 gain/limit. **대부분 실측 전 provisional(잠정)** 값이다.
   실차 calibration 이후 갱신해야 한다. provisional 여부를 필드 주석에 명시.

모든 값은 magic number 를 코드에 박지 않기 위해 여기로 모은다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ─── steering → curvature (PROVISIONAL, 2026-08-14) ──────────────────────────
#
# ⚠ CALIBRATED 아님. **PROVISIONAL** 이다.
#
# 근거와 한계:
#   - 2026-08-14 실차 calib run 에서 얻은 대표 반경. 다만 조건당 유효 run 이
#     1~2개뿐이고(카메라 4.2FPS, run 당 유효점 4~18), 조건별 반복성 검증이
#     끝나지 않았다. exact fit 이 아니라 **초기 근사**로만 쓴다.
#   - LEFT/RIGHT 비대칭은 실차에서 반복 관측된 경향이라 그대로 반영한다
#     (RIGHT 가 같은 |steering| 에서 더 조인다).
#   - **후진은 데이터가 사실상 없다.** 아래 reverse 표는 전진 기하에서 온
#     initial guess 이고, 실차 E2E 로그로 국소 재보정할 대상이다.
#   - |steering| <= NEAR_STRAIGHT_STEERING 구간은 카메라 잡음 대비 곡률이
#     작아 반경을 못 잰다. 원점과 0.45 점을 잇는 직선으로만 취급한다
#     (near-straight / interpolation zone).
#
# 표는 (|steering|, 반경 mm). 곡률 = 1/반경.
STEERING_CURVATURE_TABLE_MM: dict[tuple[str, str], tuple[tuple[float, float], ...]] = {
    ("FORWARD", "LEFT"):  ((0.45, 1320.0), (0.70, 970.0), (0.90, 780.0)),
    ("FORWARD", "RIGHT"): ((0.45, 1330.0), (0.70, 800.0), (0.90, 680.0)),
    # reverse: 전진 기하 기반 initial guess (실측 아님)
    ("REVERSE", "LEFT"):  ((0.45, 1330.0), (0.70, 975.0), (0.90, 775.0)),
    ("REVERSE", "RIGHT"): ((0.45, 1330.0), (0.70, 825.0), (0.90, 715.0)),
}
# 이 이하는 곡률을 재지 못한 구간 — 원점과 첫 표점을 잇는 직선으로 본다.
NEAR_STRAIGHT_STEERING: float = 0.30


def _table_for(reverse: bool, left: bool) -> tuple[tuple[float, float], ...]:
    return STEERING_CURVATURE_TABLE_MM[
        ("REVERSE" if reverse else "FORWARD", "LEFT" if left else "RIGHT")]


def curvature_for_steering(logical_steering: float, *, reverse: bool) -> float:
    """논리 steering(양수=LEFT) → 경로 곡률(1/mm, 양수=좌회전).

    표 사이는 |steering| 선형보간, 표 밖은 양 끝 구간의 기울기로 외삽한다.
    """
    s = float(logical_steering)
    if s == 0.0:
        return 0.0
    a = abs(s)
    tbl = _table_for(reverse, s > 0.0)
    ks = [(x, 1.0 / r) for x, r in tbl]          # (|s|, 곡률)
    x0, k0 = ks[0]
    if a <= x0:
        # near-straight 포함: 원점 ~ 첫 표점 직선
        k = k0 * (a / x0)
    else:
        k = ks[-1][1]
        for (xa, ka), (xb, kb) in zip(ks, ks[1:]):
            if a <= xb:
                k = ka + (kb - ka) * (a - xa) / (xb - xa)
                break
        else:
            (xa, ka), (xb, kb) = ks[-2], ks[-1]
            k = kb + (kb - ka) * (a - xb) / (xb - xa)
    return math.copysign(k, s)


def steering_for_curvature(curvature: float, *, reverse: bool) -> float:
    """곡률 → 논리 steering. curvature_for_steering 의 역함수 (feedforward 용)."""
    k = float(curvature)
    if k == 0.0:
        return 0.0
    a = abs(k)
    tbl = _table_for(reverse, k > 0.0)
    ks = [(x, 1.0 / r) for x, r in tbl]
    x0, k0 = ks[0]
    if a <= k0:
        s = x0 * (a / k0)
    else:
        s = ks[-1][0]
        for (xa, ka), (xb, kb) in zip(ks, ks[1:]):
            if a <= kb:
                s = xa + (xb - xa) * (a - ka) / (kb - ka)
                break
        else:
            (xa, ka), (xb, kb) = ks[-2], ks[-1]
            s = xb + (xb - xa) * (a - kb) / (kb - ka)
    return math.copysign(min(s, 1.0), k)


@dataclass(frozen=True)
class FirmwareConstants:
    """실제 ESP32 펌웨어에서 관측된 사실값 (source: app_config.example.h, actuator.c).

    steering wire 부호 (actuator.c::steering_to_angle 에서 확정):
        -1.0 -> LEFT  strong (servo 46 deg)
        -0.5 -> LEFT  weak   (servo 66 deg)
         0.0 -> CENTER        (servo 86 deg)
        +0.5 -> RIGHT weak   (servo 106 deg)
        +1.0 -> RIGHT strong (servo 126 deg)
    → **wire steering: 음수 = LEFT, 양수 = RIGHT** (이것이 절대 기준)
    """

    # --- servo 각도 (degree). servo 명령각 != 실제 조향 바퀴각 (미보정) ---
    servo_center_deg: float = 86.0
    servo_left_strong_deg: float = 46.0
    servo_left_weak_deg: float = 66.0
    servo_right_weak_deg: float = 106.0
    servo_right_strong_deg: float = 126.0

    # --- 모터 PWM duty (latest real-car calibration) ---
    pwm_forward_min: int = 15
    pwm_forward_default: int = 23
    pwm_turn_min: int = 32
    pwm_turn_default: int = 40
    pwm_strong_turn_min: int = 38
    pwm_strong_turn_default: int = 50
    motor_pwm_max_duty: int = 255
    motor_deadband_throttle: float = 0.02

    # --- 타이밍(ms) ---
    direct_control_timeout_ms: int = 500     # REMOTE_DIRECT+MOVING 에서 이 시간 내 갱신 없으면 safeStop
    heartbeat_timeout_ms: int = 1000
    status_period_ms: int = 200

    # --- 프로토콜 ---
    server_port: int = 5000
    car_id: str = "CAR_01"
    protocol_version: int = 1


@dataclass(frozen=True)
class ControllerConfig:
    """host-side 제어 gain/limit.

    ⚠ PROVISIONAL 표시된 값은 실측 전 잠정값이다.
    실차 calibration(개발 순서 4단계) 전까지 물리적 정확도를 신뢰하지 말 것.
    """

    # === steering (heading 오차 → wire steering) ===============================
    # 논리 제어법: raw = steer_kp * err_rad + steer_kd * derr_rad_s
    steer_kp: float = 1.6                 # PROVISIONAL (backend 참고값과 동일)
    steer_kd: float = 0.25                # PROVISIONAL
    # 논리 steering 을 [-1,1] 로 정규화할 때 쓰는 최대 조향 각(라디안 기준 degree).
    # servo 각이 아니라 '이 각도 오차에서 wire steering 이 포화(±1)된다'는 정규화 상수.
    steer_normalize_deg: float = 30.0     # PROVISIONAL

    # wire 부호 변환 상수. **실제 ESP32 기준으로 반드시 -1.0.**
    #   논리 steering(양수 = LEFT 요구)  ── x wire_steering_sign ──▶  wire steering(음수 = LEFT)
    #   즉 wire = wire_steering_sign * logical, wire_steering_sign = -1.0.
    # 이 값은 provisional 이 아니라 firmware 로 고정된 값이다.
    wire_steering_sign: float = -1.0

    # wire steering 절댓값 상한 (1.0 = 제한 없음).
    # 펌웨어 throttle_to_duty 는 |steering|>0.5 구간에서 duty 를 강회전 기본값
    # (55)으로 끌어올리고 throttle 을 무시한다. 최대로 꺾으면 바퀴각은 18도
    # 더 먹지만 속도가 45% 빨라져, 저속 RC카에서는 미끄러지며 오히려 선회
    # 반경이 커진다. 0.5 로 묶으면 duty 가 중간 프로파일에 머물러 속도 제어가
    # 살아난다. 실차에서 어느 쪽이 반경이 작은지 재보고 정할 값이다.
    max_wire_steering: float = 1.0

    # === throttle (거리/heading 오차 → wire throttle) ==========================
    # ⚠ throttle ↔ 실제 속도(cm/s)는 아직 미보정. 아래는 잠정 정규화 스케줄.
    throttle_per_cm_s: float = 0.030      # PROVISIONAL (cm/s → normalized throttle, 미보정)
    min_move_throttle: float = 0.22       # PROVISIONAL. 주행 중 최소 throttle(정지마찰/deadband 극복용)
    max_throttle: float = 0.40            # PROVISIONAL. 1차 데모 보수적 상한
    turn_slowdown_deg: float = 60.0       # PROVISIONAL. 이 각도에서 회전 감속 최대
    turn_throttle_floor: float = 0.30     # PROVISIONAL. 회전 감속 하한 비율(0~1)
    slow_radius_cm: float = 25.0          # PROVISIONAL. 목표 이 거리 안에서 선형 감속 시작
    stop_distance_cm: float = 3.0         # PROVISIONAL. 정지 여유 거리
    allow_reverse: bool = False           # 일반 AUTO는 기본 전진 전용
    # 후진은 일반 CRUISE에 열지 않고 복구/주차 phase로 제한한다.
    # APPROACH 추가(2026-08-12): 원호 시작점을 지나치면(APPROACH_*_MISSED)
    # 전진으로는 되잡을 수 없어 후진 복구가 필요하다. TURN/CRUISE 는 계속 제외 —
    # 통로 주행 중 후진은 뒤를 못 보므로 허용하지 않는다.
    reverse_allowed_phases: tuple[str, ...] = (
        "RECOVERY", "PARKING", "APPROACH", "ALIGN", "ENTRY", "FINAL",
    )
    # 후진 중 앞바퀴를 중립(11자)으로 고정할지. **phase 별로 나뉜다.**
    #
    # 복구(RECOVERY)는 곧게 물러나야 한다. 후진 조향은 뒤를 못 보는 상태에서
    # 궤적을 휘게 만들고, 카메라 heading 이 궤적 기반이면 부호까지 뒤집혀
    # 상황을 악화시킨다. 복구는 "곧게 물러났다 다시 들어간다"로 충분하고,
    # 복구 경로(parking.recovery)도 현재 heading 축을 따라 만든다.
    #
    # 반면 후면주차 ENTRY/FINAL 은 **후진하면서 조향해야** 슬롯 축으로 원호를
    # 그리며 들어간다. 여기까지 11자로 묶으면 계획한 원호를 절대 못 탄다.
    # 이 구간은 목표 heading 이 waypoint 에 실려 있고 쿠션 기반 heading 이
    # 살아 있는 것을 전제로 하므로 궤적 부호 반전 위험이 없다.
    #
    # reverse_straight_steering 은 전체 on/off 마스터 스위치,
    # reverse_straight_phases 는 그중 실제로 고정할 phase 목록이다.
    reverse_straight_steering: bool = True
    reverse_straight_phases: tuple[str, ...] = ("RECOVERY",)

    # === 최대 조향 정지 마찰 =================================================
    # 바퀴를 끝까지 꺾으면 정지 마찰이 급증한다. 펌웨어 throttle_to_duty 는
    # |steering|>0.5 구간에서 duty = lerp(38, 50, throttle) 이라, max_throttle 을
    # 낮게 잡으면 duty 가 38~40 에 묶여 **차가 아예 못 움직인다**.
    #   실측 2026-08-12: max_throttle 0.20, |steering| 1.00 → duty 40,
    #                    12초간 pose 변화 0. 명령은 계속 나가는데 정지.
    # 조향이 strong_turn_steering 을 넘으면 throttle 하한을 올려 duty 를
    # 끌어올린다. 이 하한은 max_throttle 을 **의도적으로 넘을 수 있다** —
    # max_throttle 은 속도 상한인데, 최대 조향에서는 duty 를 올려도 차가
    # 기어가듯 움직이기 때문이다. None 이면 끈다.
    # 0.9 = **바퀴가 스토퍼에 닿기 직전**. 여기부터 정지 마찰이 급증한다.
    # 0.5 로 잡으면 heading 오차 9도만 넘어도 걸려 max_throttle 이 무의미해진다.
    strong_turn_steering: float = 0.9
    strong_turn_min_throttle: float | None = 0.7
    # 정밀 주차 구간에서는 이 하한이 **phase 상한을 넘지 못하게** 묶는다.
    #
    # 위 하한은 전진 최대조향 정지마찰(실측 2026-08-12: duty 40 에서 12초간
    # pose 변화 0)을 풀려고 넣은 값이고, 일반 주행에서는 max_throttle 을
    # 의도적으로 넘어야 의미가 있다. 그런데 주차 원호에서는 조향이 상시
    # 0.9 근처라 하한이 계속 발동해 throttle 이 0.10 → 0.70 으로 튀고,
    # 차가 원호 바깥으로 날아간다 (closed-loop 확인).
    #
    # 정밀 주차에서는 **속도 상한이 정지마찰보다 우선**이다. 그래도 안 움직이면
    # 상한 자체를 올린다 (--parking-throttle). 일반 주행 동작은 그대로 둔다.
    strong_turn_capped_phases: tuple[str, ...] = (
        "APPROACH", "ALIGN", "ENTRY", "FINAL", "PARKING",
    )

    # === 곡률 feedforward ====================================================
    # pose_controller 는 "다음 점을 향한 heading 오차 PD" 라서, 일정 곡률
    # 원호에서 필요한 조향을 미리 주지 못한다. 원호 시작에서 오차가 작아
    # 조향이 모자라고, 밀려난 뒤에야 포화된다 (closed-loop 확인).
    #
    # planner 가 waypoint.curvature 로 경로 기하를 넘겨주면 그만큼을 미리
    # 넣고, 기존 PD 는 **오차 보정**만 담당하게 한다. 특정 반경을 제어기에
    # 박지 않기 위한 구조다.
    #
    # 매핑: logical_steering = curvature × feedforward_full_lock_radius_mm
    #       (|logical|=1 = 최대 조향 = 최소 선회 반경)
    # ⚠ 실차 steering↔반경 calibration 전의 **1차 근사**다. 주행 로그로
    #    이 상수를 갱신하는 것이 다음 보정 항목이다.
    curvature_feedforward: bool = True
    feedforward_full_lock_radius_mm: float = 610.0
    # feedforward 를 쓸 phase. FINAL 은 제외한다 — 원호가 끝난 지점이라
    # feedforward 가 남아 있으면 과회전한다.
    curvature_feedforward_phases: tuple[str, ...] = (
        "ALIGN", "ENTRY", "PARKING", "RECOVERY",
    )
    # feedforward 가 걸린 구간에서 PD 가 낼 수 있는 **보정 폭**.
    #
    # feedforward 만으로 이미 최대 조향의 87%(R=700)를 쓰고 있어서, PD 를
    # 그대로 얹으면 오차 20° 만 나도 즉시 포화된다. 포화되면 계획 반경보다
    # 더 조여 돌아 오차가 커지고, 그 상태가 유지되며 나선으로 발산한다
    # (closed-loop 확인). 곡률 추종에서 PD 의 역할은 "경로로 되돌리는 보정"
    # 이지 "경로를 새로 정하는 것"이 아니므로 폭을 묶는다.
    curvature_feedback_limit: float = 0.25
    # 실차 표(PROVISIONAL)를 쓸지. False 면 기존 단일 반경 선형 모델.
    use_measured_curvature_table: bool = True

    # === 후진 heading 안전 ====================================================
    # 궤적 기반 heading(TRAJECTORY)은 후진하면 진행방향이 뒤집혀 180° 틀린다.
    # 그 값을 믿고 조향하면 정확히 반대로 꺾는다. 후진 정밀 구간에서는
    # LAST_VALID 역시 현재 프레임의 차체 방향이 아니라 예전 값이다.
    # 후진 원호에서 계속 쓰면 차체가 돌아가는 만큼 오차가 누적되므로
    # 물리적 전방 쿠션 heading이 돌아올 때까지 정지한다. 일시 손실은
    # HostController가 재획득하고, 지속 손실은 bounded replan으로 보낸다.
    unsafe_reverse_heading_sources: tuple[str, ...] = (
        "TRAJECTORY", "LAST_VALID",
    )
    heading_guard_reverse_phases: tuple[str, ...] = ("ENTRY", "FINAL", "PARKING")
    # A reverse phase may wait briefly for FRONT_CUSHION after ALIGN.  If fresh
    # unsafe headings keep arriving for longer, request parking replan instead
    # of holding an unbounded silent zero.
    reverse_heading_wait_timeout_s: float = 2.5
    # Reverse observation state machine.  Values are derived from the actual
    # 2026-08-21 runs: successful reverse windows had >= 30 mm net motion,
    # path/net linearity >= .94 and a maximum five-frame span of 1.39 s.
    # The production gate stays slightly inside those observations except for
    # the 1.5 s span ceiling.  Raw TRAJECTORY is never used as reverse body
    # heading; accepted reverse trajectories are direction-corrected by 180°.
    reverse_trajectory_window: int = 5
    reverse_trajectory_min_observations: int = 3
    reverse_trajectory_min_displacement_mm: float = 30.0
    reverse_trajectory_max_span_s: float = 1.5
    reverse_trajectory_min_linearity: float = 0.90
    reverse_trajectory_max_heading_delta_deg: float = 45.0
    # A reverse START without FRONT_CUSHION remains forbidden.  The sole
    # alternative is a same-route ALIGN terminal body-heading anchor proven by
    # recent forward motion.  Latest blocked runs delivered their first new
    # post-interlock frame 29.4 / 26.9 mm from that anchor; 35 mm includes only
    # the observed pose quantisation margin, not an open-ended dead reckoning.
    reverse_start_anchor_max_age_s: float = 0.75
    reverse_start_anchor_max_distance_mm: float = 35.0
    reverse_start_anchor_max_heading_delta_deg: float = 20.0
    # Once the first reverse command is issued from that anchor, keep the same
    # body heading only until reverse motion itself becomes measurable.  At the
    # minimum 610 mm turn radius, 30 mm can rotate the body by at most 2.8 deg.
    reverse_start_bootstrap_max_distance_mm: float = 30.0
    reverse_start_bootstrap_max_age_s: float = 1.5

    # === 자동주차 / recovery =================================================
    # APPROACH/ALIGN/ENTRY/FINAL은 stop_distance를 tolerance에 더하지 않고
    # waypoint 자체 허용오차로 도착을 판단한다. CRUISE는 기존 실차 baseline 유지.
    strict_arrival_phases: tuple[str, ...] = (
        "APPROACH", "ALIGN", "ENTRY", "FINAL", "PARKING",
    )
    precision_drive_phases: tuple[str, ...] = (
        "APPROACH", "ALIGN", "ENTRY", "FINAL", "PARKING", "RECOVERY",
    )

    # 동일 APPROACH target을 COARSE(1차 capture) -> FINE(정밀 완료)로 해석.
    approach_capture_tolerance_cm: float = 10.0
    approach_pass_margin_cm: float = 1.0

    # Rear ALIGN endpoint heading capture hysteresis.  The base waypoint
    # tolerance is deliberately unchanged.  This band is available only after
    # a previous, distinct camera observation already entered that base
    # tolerance while inside the planned endpoint corridor.  Actual runs
    # 155812/161336 crossed from 1.8/0.7 deg to 5.4/6.9 deg on the next frame;
    # successful captures were 1.4--4.7 deg.
    align_settled_hysteresis_deg: float = 2.0

    # FINAL은 서로 다른 fresh camera observation이 연속 N회 만족해야 DONE.
    final_confirm_observations: int = 3

    # 일반 CRUISE의 min_move_throttle=0.22는 유지하되, 정밀 주차에서는
    # 4~8cm/s waypoint 속도 차이가 0.22 하나로 뭉개지지 않게 낮은 floor 사용.
    parking_min_move_throttle: float | None = 0.08
    reverse_min_move_throttle: float | None = 0.10
    parking_max_throttle: float | None = 0.25
    reverse_max_throttle: float | None = 0.25

    # === 안전/신선도 =========================================================
    max_pose_age_s: float = 0.5           # 이 시간 초과한 pose 는 stale → 정지

    def brake_radius_cm(self, position_tolerance_cm: float) -> float:
        """기존 CRUISE 도착 반경(cm)."""
        return position_tolerance_cm + self.stop_distance_cm

    def arrival_radius_cm(self, position_tolerance_cm: float, phase: str | None) -> float:
        """phase별 실제 ARRIVED 판정 반경."""
        phase_key = (phase or "").upper()
        if phase_key in self.strict_arrival_phases:
            return float(position_tolerance_cm)
        return self.brake_radius_cm(position_tolerance_cm)

    def reverse_steering_locked(self, phase: str | None) -> bool:
        """이 phase 의 후진에서 앞바퀴를 11자로 고정하는가.

        RECOVERY 는 True(직선 후진 유지), ENTRY/FINAL 은 False(조향 허용).
        reverse_allowed_phases 와는 관심사가 다르다 — 그쪽은 "후진해도 되는가",
        이쪽은 "후진 중 조향해도 되는가"다. 두 게이트는 독립으로 걸린다.
        """
        if not self.reverse_straight_steering:
            return False
        return (phase or "").upper() in self.reverse_straight_phases

    def final_throttle_ceiling(self, phase: str | None, *,
                               reverse: bool) -> float | None:
        """stiction 하한까지 적용한 **뒤** 마지막으로 강제하는 상한.

        정지마찰 하한은 일반 주행에서 의도적으로 max_throttle 을 넘는다.
        그 예외가 정밀 주차와 후진까지 새어 들어가면 안 된다:

        - 후진은 어떤 phase 든 reverse_max_throttle 을 넘지 않는다.
          (뒤를 못 보는 상태에서 속도가 튀는 것이 가장 위험하다)
        - 정밀 주차 phase 는 그 phase 의 상한을 넘지 않는다.
        - 전진 CRUISE 는 그대로 둔다 — 실측된 정지마찰 대응이 여기 필요하다.

        None 이면 추가 상한 없음.
        """
        ceiling: float | None = None
        if reverse and self.reverse_max_throttle is not None:
            ceiling = max(0.0, float(self.reverse_max_throttle))
        if (phase or "").upper() in self.strong_turn_capped_phases:
            limit = self.throttle_limit(phase, reverse=reverse)
            ceiling = limit if ceiling is None else min(ceiling, limit)
        return ceiling

    def stiction_floor_for(self, phase: str | None, *, reverse: bool) -> float | None:
        """이 구간에서 적용할 최대조향 정지마찰 하한. None 이면 적용 안 함.

        일반 주행: 기존대로 max_throttle 을 넘어설 수 있다.
        정밀 주차: phase 상한을 넘지 못한다 (속도 상한이 우선).
        """
        floor = self.strong_turn_min_throttle
        if floor is None:
            return None
        if (phase or "").upper() in self.strong_turn_capped_phases:
            return min(float(floor), self.throttle_limit(phase, reverse=reverse))
        return float(floor)

    def feedforward_steering(self, phase: str | None, curvature: float,
                             *, reverse: bool = False) -> float:
        """경로 곡률 → 논리 steering feedforward (양수 = LEFT).

        curvature 는 planner 가 준 경로 기하다. 제어기는 반경을 모른 채
        이 값만 쓴다 — 특정 슬롯/반경이 제어기에 박히지 않는다.
        """
        if not self.curvature_feedforward or not curvature:
            return 0.0
        if (phase or "").upper() not in self.curvature_feedforward_phases:
            return 0.0
        if self.use_measured_curvature_table:
            # 실차 표(PROVISIONAL)의 역함수. 단일 반경 선형 가정보다
            # LEFT/RIGHT 비대칭과 비선형을 반영한다.
            return steering_for_curvature(curvature, reverse=reverse)
        return max(-1.0, min(1.0,
                             curvature * self.feedforward_full_lock_radius_mm))

    def reverse_heading_unsafe(self, phase: str | None, source: str | None,
                               *, reverse: bool) -> bool:
        """이 후진 구간에서 이 heading 출처를 믿으면 안 되는가."""
        if not reverse or source is None:
            return False
        if (phase or "").upper() not in self.heading_guard_reverse_phases:
            return False
        return str(source).upper() in self.unsafe_reverse_heading_sources

    def throttle_limit(self, phase: str | None, *, reverse: bool) -> float:
        """phase/direction별 normalized throttle 상한."""
        limit = float(self.max_throttle)
        phase_key = (phase or "").upper()
        if phase_key in self.precision_drive_phases and self.parking_max_throttle is not None:
            limit = min(limit, max(0.0, float(self.parking_max_throttle)))
        if reverse and self.reverse_max_throttle is not None:
            limit = min(limit, max(0.0, float(self.reverse_max_throttle)))
        return limit

    def min_move_throttle_for(self, phase: str | None, *, reverse: bool) -> float:
        """phase/direction별 stiction 극복용 throttle floor."""
        phase_key = (phase or "").upper()
        floor = float(self.min_move_throttle)
        if phase_key in self.precision_drive_phases and self.parking_min_move_throttle is not None:
            floor = max(0.0, float(self.parking_min_move_throttle))
        if reverse and self.reverse_min_move_throttle is not None:
            floor = max(0.0, float(self.reverse_min_move_throttle))
        return min(floor, self.throttle_limit(phase, reverse=reverse))


@dataclass(frozen=True)
class SimulationConfig:
    """수렴 시뮬레이션 전용 파라미터.

    ⚠ SIMULATION-ONLY / PROVISIONAL.
    실차 물리값이 아니라 controller 의 부호/수렴 거동을 확인하기 위한 값이다.
    실차 검증 결과로 절대 사용하지 말 것.
    """

    wheelbase_mm: float = 140.0                 # SIM-ONLY 잠정 축거
    max_wheel_angle_deg: float = 25.0           # SIM-ONLY 잠정 유효 조향 바퀴 최대각
    sim_speed_cm_s_at_full_throttle: float = 30.0  # SIM-ONLY throttle=1 일 때 가정 속도
    dt_s: float = 0.05                          # SIM 스텝(20Hz)
