"""B안 주행 제어기 — 카메라 pose + 목표 waypoint → throttle/steering.

배경
----
ESP32 내부 waypoint 추종기는 미구현이고, `DIRECT_CONTROL` → 모터/서보 경로는
이미 실차 검증이 끝났다. 그래서 1차 데모는 노트북이 제어값을 계산해
스트림으로 내려주는 구조로 간다 (2026-08-07 하드웨어팀 합의).

역할 분담
    노트북 : pose ↔ waypoint 비교 → 거리·heading 오차 → throttle/steering
    ESP32  : 명령 검증, 모터·서보 출력, heartbeat/timeout, safe-stop, 상태 보고

이 모듈은 순수 계산만 한다 (소켓·스레드 없음). 송신은 `comm.server`,
목표 waypoint 선정은 `comm.orchestrator` 가 담당한다.

좌표·부호 규약
--------------
- 위치: 바닥판 좌하단 원점, +x 우, +y 상, 단위 mm (내부 표준)
- heading: 0~360°, 우측 0° / 위쪽 90° (반시계 양수)
- steering: -1.0~1.0. **양수 = 좌회전(heading 증가 방향)** 을 기본으로 두되,
  실차 서보 배선이 반대일 수 있어 `steering_sign` 으로 뒤집을 수 있다.
- throttle: -1.0~1.0. 현재 경로 설계는 전진만 쓴다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["VehicleLimits", "Pose", "ControlOutput", "WaypointController"]


def wrap180(deg: float) -> float:
    """각도 차이를 -180~180 으로 정규화."""
    return (deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class VehicleLimits:
    """실차 특성 파라미터.

    기본값은 소형 RC카 기준의 보수적 추정치다. 실차 튜닝 전에는
    `max_throttle` 을 낮게 유지하고, 하드웨어팀 실측치(최소 구동 PWM,
    정지 거리, 엔코더 환산)를 받으면 갱신한다.
    """

    # ── 조향 ──
    max_steer_deg: float = 30.0        # 서보 최대 조향각 (steering=±1.0 에 대응)
    steer_kp: float = 1.6              # heading 오차(rad) → 조향 명령
    steer_kd: float = 0.25             # 오차 변화율 감쇠 (진동 억제)
    steering_sign: float = 1.0         # 서보 방향이 반대면 -1.0

    # ── 구동 ──
    min_throttle: float = 0.22         # 이보다 작으면 차가 안 움직인다 (실측 필요)
    max_throttle: float = 0.45         # 1차 데모 안전 상한
    throttle_per_cm_s: float = 0.035   # 피드포워드: 목표 속도 → throttle
    speed_kp: float = 0.010            # 실측 속도 오차 보정 (약하게)
    max_speed_trim: float = 0.12       # 보정 항 상한 (카메라 속도는 노이즈가 크다)

    # ── 감속·정지 ──
    slow_radius_cm: float = 25.0       # 목표 이 거리 안에서 선형 감속
    stop_distance_cm: float = 3.0      # 제동 거리 — 이만큼 미리 throttle 을 끊는다
    turn_slowdown_deg: float = 60.0    # 이 이상 틀어져 있으면 최저속으로 회두

    # ── 안전 ──
    max_pose_age_s: float = 0.5        # 이보다 오래된 pose 로는 구동하지 않는다


@dataclass(frozen=True)
class Pose:
    x_mm: float
    y_mm: float
    heading_deg: float | None
    timestamp: float                   # time.monotonic() 기준
    valid: bool = True


@dataclass(frozen=True)
class ControlOutput:
    throttle: float
    steering: float
    mode: str                          # DRIVE | BRAKE | ALIGN | ARRIVED | HOLD
    distance_cm: float
    heading_error_deg: float
    reason: str = ""                   # 정지 사유 (mode 가 HOLD 일 때)

    @property
    def is_stopped(self) -> bool:
        return self.throttle == 0.0


class WaypointController:
    """차량 1대의 제어 상태 (조향 미분항·속도 추정)를 들고 있는다.

    프레임마다 `compute()` 를 부르면 되고, 전송 주기(10Hz)와 카메라 주기가
    달라도 무방하다 — 마지막 출력을 그대로 재전송하면 된다.
    """

    def __init__(self, limits: VehicleLimits | None = None) -> None:
        self.limits = limits or VehicleLimits()
        self._prev_err_rad: float | None = None
        self._prev_time: float | None = None
        self._prev_pose: Pose | None = None
        self._speed_cm_s: float = 0.0

    def reset(self) -> None:
        """경로 교체·정지 후 재출발 시 호출 — 미분항이 튀는 것을 막는다."""
        self._prev_err_rad = None
        self._prev_time = None
        self._prev_pose = None
        self._speed_cm_s = 0.0

    @property
    def measured_speed_cm_s(self) -> float:
        return self._speed_cm_s

    # ─── 메인 ────────────────────────────────────────────────────────────────

    def compute(self, pose: Pose, target: Any, *,
                allow_drive: bool = True,
                now: float | None = None) -> ControlOutput:
        """제어값 1스텝 계산.

        target 은 `parking.waypoints.Waypoint` (x/y 는 mm) 를 기대하지만,
        같은 속성을 가진 객체면 무엇이든 받는다.

        allow_drive=False 면 오차만 계산하고 출력은 0 으로 낸다
        (모터 OFF 검증, 미션 HELD, 충돌 회피 정지 등).

        now 는 pose 신선도 판정 기준 시각 (기본 time.monotonic()).
        """
        now = time.monotonic() if now is None else now
        dx = target.x - pose.x_mm
        dy = target.y - pose.y_mm
        distance_cm = math.hypot(dx, dy) / 10.0

        # ── 안전 게이트: 쓸 수 없는 pose 로는 절대 구동하지 않는다 ──
        if not pose.valid:
            return self._halt(distance_cm, 0.0, "POSE_INVALID")
        if pose.heading_deg is None:
            # heading 을 모르면 어느 쪽으로 틀어야 할지 알 수 없다.
            return self._halt(distance_cm, 0.0, "NO_HEADING")
        if now - pose.timestamp > self.limits.max_pose_age_s:
            # 카메라가 놓친 사이에 옛 위치로 조향하면 실제로는 엉뚱하게 간다.
            return self._halt(distance_cm, 0.0, "POSE_STALE")

        bearing_deg = math.degrees(math.atan2(dy, dx))
        err_deg = wrap180(bearing_deg - pose.heading_deg)
        err_rad = math.radians(err_deg)

        self._update_speed(pose)
        derr = self._derivative(err_rad, pose.timestamp)
        self._prev_pose = pose

        if not allow_drive:
            return self._halt(distance_cm, err_deg, "DRIVE_NOT_ALLOWED")

        # ── 도착 판정 (제동 거리만큼 미리 끊는다) ──
        # 최종 도착 확정은 오케스트레이터가 하고, 여기서는 구동만 멈춘다.
        brake_radius_cm = target.position_tolerance_cm + self.limits.stop_distance_cm
        if distance_cm <= brake_radius_cm:
            if self._needs_alignment(target, pose.heading_deg):
                head_err = wrap180((target.target_heading_deg or 0.0) - pose.heading_deg)
                # 제자리 회전이 불가능한 Ackermann 조향이라 여기서는 멈춘다.
                # 방향이 안 맞으면 상위가 재접근 경로를 다시 만든다.
                return ControlOutput(0.0, 0.0, "ALIGN", distance_cm, head_err,
                                     "HEADING_OUT_OF_TOLERANCE")
            return ControlOutput(0.0, 0.0, "ARRIVED", distance_cm, err_deg)

        steering = self._steering(err_rad, derr)
        throttle = self._throttle(target, distance_cm, err_deg)
        mode = "DRIVE" if throttle > 0.0 else "BRAKE"
        return ControlOutput(throttle, steering, mode, distance_cm, err_deg)

    # ─── 구성요소 ────────────────────────────────────────────────────────────

    def _steering(self, err_rad: float, derr: float) -> float:
        lim = self.limits
        raw = lim.steer_kp * err_rad + lim.steer_kd * derr
        # 조향각 한계로 정규화: 최대 조향각에 해당하는 오차에서 ±1.0 이 되게 한다.
        norm = raw / math.radians(lim.max_steer_deg)
        return round(_clamp(norm * lim.steering_sign, -1.0, 1.0), 4)

    def _throttle(self, target: Any, distance_cm: float, err_deg: float) -> float:
        lim = self.limits
        desired = float(target.speed_cm_s)

        # 크게 틀어져 있으면 속도를 낮춰 회두 반경을 줄인다.
        turn_scale = _clamp(
            1.0 - abs(err_deg) / max(lim.turn_slowdown_deg * 2.0, 1e-6), 0.3, 1.0)
        # 목표 근처에서는 선형 감속 — 제동 거리를 뺀 거리 기준.
        remaining = max(distance_cm - lim.stop_distance_cm, 0.0)
        approach_scale = _clamp(remaining / max(lim.slow_radius_cm, 1e-6), 0.0, 1.0)

        desired *= turn_scale * approach_scale
        if desired <= 0.0:
            return 0.0

        throttle = desired * lim.throttle_per_cm_s
        throttle += _clamp(lim.speed_kp * (desired - self._speed_cm_s),
                           -lim.max_speed_trim, lim.max_speed_trim)
        # 데드밴드 보상: 최소 구동값 미만은 소리만 나고 안 움직인다.
        throttle = max(throttle, lim.min_throttle)
        return round(_clamp(throttle, 0.0, lim.max_throttle), 4)

    def _needs_alignment(self, target: Any, heading_deg: float) -> bool:
        if not target.heading_required or target.target_heading_deg is None:
            return False
        err = abs(wrap180(target.target_heading_deg - heading_deg))
        return err > target.heading_tolerance_deg

    def _derivative(self, err_rad: float, now: float) -> float:
        prev_err, prev_t = self._prev_err_rad, self._prev_time
        self._prev_err_rad, self._prev_time = err_rad, now
        if prev_err is None or prev_t is None:
            return 0.0
        dt = now - prev_t
        if dt <= 1e-3:
            return 0.0
        return (err_rad - prev_err) / dt

    def _update_speed(self, pose: Pose) -> None:
        prev = self._prev_pose
        if prev is None:
            return
        dt = pose.timestamp - prev.timestamp
        if dt <= 1e-3:
            return
        step_cm = math.hypot(pose.x_mm - prev.x_mm, pose.y_mm - prev.y_mm) / 10.0
        raw = step_cm / dt
        # 카메라 좌표는 프레임마다 흔들린다 → 1차 저역통과로 다듬는다.
        self._speed_cm_s = 0.6 * self._speed_cm_s + 0.4 * raw

    def _halt(self, distance_cm: float, err_deg: float, reason: str) -> ControlOutput:
        self._prev_err_rad = None          # 정지 후 재출발 시 미분항 튐 방지
        self._prev_time = None
        return ControlOutput(0.0, 0.0, "HOLD", distance_cm, err_deg, reason)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
