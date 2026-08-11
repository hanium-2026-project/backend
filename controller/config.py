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

from dataclasses import dataclass


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
    pwm_forward_min: int = 12
    pwm_forward_default: int = 22
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
    reverse_allowed_phases: tuple[str, ...] = (
        "RECOVERY", "PARKING", "ALIGN", "ENTRY", "FINAL",
    )

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
