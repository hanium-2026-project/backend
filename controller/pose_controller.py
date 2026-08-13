"""Pose + Waypoint → wire-ready ControlCommand 를 계산하는 코어 제어기.

설계 원칙
--------
- **순수/결정론적**: 동일 입력 + 동일 now → 동일 출력. 내부 상태는 PD 미분 항뿐.
- **stdlib only**: math, time, dataclasses 만 import. backend/network 의존 없음.
- **wire-ready 출력**: steering/throttle 을 ESP32 DIRECT_CONTROL 에 바로 넣을 수 있음.
- **안전 우선**: 무효/비신선 pose, heading 없음, drive 비활성 → 즉시 zero.

steering 부호 파이프라인 (실제 ESP32 기준)
-----------------------------------------
    heading_error > 0  (목표가 CCW/LEFT)
        → 논리 steering(양수 = LEFT 요구)
        → wire = wire_steering_sign(-1.0) * 논리
        → wire steering < 0  == 실제 ESP32 LEFT
"""

from __future__ import annotations

import math
import time
from typing import Optional

from . import geometry as geo
from .config import ControllerConfig
from .models import ControlCommand, ControlMode, MotionDirection, Pose, Waypoint


class PoseWaypointController:
    """단일 waypoint 안정 추종용 1차 제어기 (P/PD + 보수적 throttle 스케줄)."""

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()
        self._prev_err_rad: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._motion_direction: Optional[MotionDirection] = None

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        """mode 전환/재접속용 전체 제어상태 초기화."""
        self._reset_derivative()
        self._motion_direction = None

    def _reset_derivative(self) -> None:
        """PD 미분 이력만 초기화하고 구동 방향 latch는 보존한다."""
        self._prev_err_rad = None
        self._prev_time = None

    def compute(
        self,
        pose: Pose,
        waypoint: Waypoint,
        *,
        allow_drive: bool = True,
        now: Optional[float] = None,
    ) -> ControlCommand:
        """현재 pose 와 목표 waypoint 로부터 제어 명령을 계산한다.

        Parameters
        ----------
        pose : Pose
            카메라가 관측한 현재 차량 pose (mm/deg).
        waypoint : Waypoint
            목표 지점 (mm/deg).
        allow_drive : bool
            False 면 모든 계산을 하되 출력은 zero(HOLD). motor-OFF 검증에 사용.
        now : float | None
            현재 monotonic 시각(초). None 이면 time.monotonic(). 테스트 결정성을 위해 주입 권장.
        """
        cfg = self.config
        now = time.monotonic() if now is None else now

        dx = waypoint.x_mm - pose.x_mm
        dy = waypoint.y_mm - pose.y_mm
        distance_cm = math.hypot(dx, dy) / 10.0
        bearing = geo.bearing_deg(pose.x_mm, pose.y_mm, waypoint.x_mm, waypoint.y_mm)

        # ---- 안전 게이트 (무조건 zero) --------------------------------------
        if not pose.valid:
            return self._halt(distance_cm, 0.0, bearing, "POSE_INVALID")
        if (now - pose.timestamp) > cfg.max_pose_age_s:
            return self._halt(distance_cm, 0.0, bearing, "POSE_STALE")

        # ---- 도착 판정은 heading 없이도 가능하다 (backend 결선 시 수정) --------
        # 목표에 닿았는지는 위치만으로 정해진다. heading 게이트를 앞에 두면,
        # 이미 목표 위에 있는데 방향을 못 구한 차량이 영원히 다음 waypoint 로
        # 넘어가지 못한다 (정지 상태에서는 궤적으로 heading 을 못 구해 교착).
        # 방향이 필요한 waypoint(heading_required)일 때만 heading 을 요구한다.
        if not pose.has_heading:
            arrival_radius_nh = cfg.arrival_radius_cm(
                waypoint.position_tolerance_cm, waypoint.phase)
            if distance_cm <= arrival_radius_nh and not waypoint.heading_required:
                self.reset()
                return ControlCommand(
                    throttle=0.0, steering=0.0, mode=ControlMode.ARRIVED,
                    arrived=True, distance_error_cm=distance_cm,
                    heading_error_deg=0.0, target_bearing_deg=bearing,
                    reason="ARRIVED",
                )
            return self._halt(distance_cm, 0.0, bearing, "NO_HEADING")

        heading = pose.heading_deg  # not None (위에서 보장)
        direction = waypoint.motion_direction
        reverse = direction is MotionDirection.REVERSE

        # 후진은 명시적으로 허용된 주차/복구 phase에서만 실행한다.
        if reverse:
            if not cfg.allow_reverse:
                return self._halt(distance_cm, 0.0, bearing, "REVERSE_NOT_ALLOWED")
            phase = (waypoint.phase or "").upper()
            if phase not in cfg.reverse_allowed_phases:
                return self._halt(distance_cm, 0.0, bearing, "REVERSE_PHASE_NOT_ALLOWED")

        # 후진에서는 실제 이동방향이 body heading + 180°.
        motion_heading = geo.wrap180(heading + (180.0 if reverse else 0.0))
        err_deg = geo.heading_error_deg(bearing, motion_heading)
        err_rad = math.radians(err_deg)

        if not allow_drive:
            return self._halt(distance_cm, err_deg, bearing, "DRIVE_NOT_ALLOWED")

        # ---- 도착/정렬 판정 ------------------------------------------------
        arrival_radius = cfg.arrival_radius_cm(
            waypoint.position_tolerance_cm, waypoint.phase
        )
        if distance_cm <= arrival_radius:
            if self._needs_alignment(waypoint, heading):
                head_err = geo.wrap180((waypoint.target_heading_deg or 0.0) - heading)
                # 전진 전용 1차 controller 는 제자리 회전을 하지 않는다.
                # 위치는 도착했으니 정지하고 heading 오차만 보고 (정렬 기동은 향후 과제).
                return ControlCommand(
                    throttle=0.0,
                    steering=0.0,
                    mode=ControlMode.ALIGN,
                    arrived=False,
                    distance_error_cm=distance_cm,
                    heading_error_deg=head_err,
                    target_bearing_deg=bearing,
                    reason="HEADING_OUT_OF_TOLERANCE",
                )
            # 다음 waypoint가 FORWARD<->REVERSE로 바뀌는지 확인해야 하므로
            # 방향 latch는 보존하고 PD 이력만 지운다.
            self._reset_derivative()
            return ControlCommand(
                throttle=0.0,
                steering=0.0,
                mode=ControlMode.ARRIVED,
                arrived=True,
                distance_error_cm=distance_cm,
                heading_error_deg=err_deg,
                target_bearing_deg=bearing,
                reason="ARRIVED",
            )

        # 전/후진 방향이 주행 중 바뀌면 한 tick zero를 넣어 DIR 급전환을 막는다.
        if self._motion_direction is not None and direction is not self._motion_direction:
            self._motion_direction = direction
            self._prev_err_rad = None
            self._prev_time = None
            return ControlCommand(
                throttle=0.0, steering=0.0, mode=ControlMode.HOLD, arrived=False,
                distance_error_cm=distance_cm, heading_error_deg=err_deg,
                target_bearing_deg=bearing, reason="DIRECTION_CHANGE_STOP",
            )
        self._motion_direction = direction

        # PD 미분 항 갱신. 후진은 조향의 물리 효과가 반대라 제어 부호를 반전한다.
        derr = self._derivative(err_rad, now)
        steering_err_rad = -err_rad if reverse else err_rad
        steering_derr = -derr if reverse else derr

        # ---- 정상 주행: steering / throttle --------------------------------
        if reverse and getattr(cfg, "reverse_straight_steering", False):
            # 11자 후진 — 곧게 물러난다 (config 주석 참조)
            logical_steer, wire_steer = 0.0, 0.0
        else:
            logical_steer, wire_steer = self._steering(steering_err_rad, steering_derr)
        throttle_mag = self._throttle(
            waypoint, distance_cm, err_deg, reverse=reverse,
            wire_steering=wire_steer,
        )
        throttle = -throttle_mag if reverse else throttle_mag
        mode = ControlMode.DRIVE if abs(throttle) > 0.0 else ControlMode.BRAKE

        return ControlCommand(
            throttle=throttle,
            steering=wire_steer,
            mode=mode,
            arrived=False,
            distance_error_cm=distance_cm,
            heading_error_deg=err_deg,
            target_bearing_deg=bearing,
            reason="",
            logical_steering=logical_steer,
        )

    # ----------------------------------------------------------------- private
    def _steering(self, err_rad: float, derr_rad_s: float) -> tuple[float, float]:
        """(논리 steering, wire steering) 반환.

        논리 steering: 양수 = LEFT 요구 (heading_error > 0 과 같은 부호).
        wire steering : ESP32 실제 부호(음수 = LEFT). = wire_steering_sign * 논리.
        """
        cfg = self.config
        raw = cfg.steer_kp * err_rad + cfg.steer_kd * derr_rad_s
        norm = raw / math.radians(cfg.steer_normalize_deg)
        logical = geo.clamp(norm, -1.0, 1.0)
        wire = geo.clamp(cfg.wire_steering_sign * logical, -1.0, 1.0)
        # -0.0 정규화(로그 가독성)
        logical = round(logical, 4) or 0.0
        wire = round(wire, 4) or 0.0
        return logical, wire

    def _throttle(
        self,
        waypoint: Waypoint,
        distance_cm: float,
        err_deg: float,
        *,
        reverse: bool = False,
        wire_steering: float = 0.0,
    ) -> float:
        """보수적 정규화 throttle 스케줄.

        ⚠ throttle ↔ 실제 속도(cm/s)는 미보정. 아래는 잠정 스케줄이다.
        - 회전 감속: heading 오차가 클수록 느리게
        - 접근 감속: 목표에 가까울수록 느리게
        - 하한: 주행이 필요할 때는 min_move_throttle 로 stiction 극복
        - 상한: max_throttle
        """
        cfg = self.config

        # 회전 감속 (0..1). |err| 가 turn_slowdown_deg 이상이면 turn_throttle_floor 로.
        turn_scale = geo.clamp(
            1.0 - abs(err_deg) / max(cfg.turn_slowdown_deg, 1e-6),
            cfg.turn_throttle_floor,
            1.0,
        )
        # 접근 감속 (0..1).
        remaining = max(distance_cm - cfg.stop_distance_cm, 0.0)
        approach_scale = geo.clamp(remaining / max(cfg.slow_radius_cm, 1e-6), 0.0, 1.0)

        desired_cm_s = max(float(waypoint.speed_cm_s), 0.0) * turn_scale * approach_scale
        if desired_cm_s <= 0.0:
            return 0.0

        # phase별 저속 floor / 상한. 일반 CRUISE baseline은 유지하면서
        # 주차/Recovery에서는 waypoint speed가 실제 throttle 차이로 남게 한다.
        throttle = desired_cm_s * cfg.throttle_per_cm_s
        throttle = max(
            throttle,
            cfg.min_move_throttle_for(waypoint.phase, reverse=reverse),
        )
        throttle = geo.clamp(
            throttle,
            0.0,
            cfg.throttle_limit(waypoint.phase, reverse=reverse),
        )

        # 최대 조향 정지 마찰 극복 (config 주석 참조).
        # 상한 clamp **뒤에** 적용한다 — max_throttle 은 속도 상한이고, 최대
        # 조향에서는 duty 를 올려도 차가 기어가듯 움직이기 때문이다.
        floor = getattr(cfg, "strong_turn_min_throttle", None)
        if (floor is not None
                and abs(wire_steering) >= getattr(cfg, "strong_turn_steering", 0.5)):
            throttle = max(throttle, float(floor))

        return round(throttle, 4)

    def _needs_alignment(self, waypoint: Waypoint, heading_deg: float) -> bool:
        if not waypoint.heading_required or waypoint.target_heading_deg is None:
            return False
        head_err = geo.wrap180(waypoint.target_heading_deg - heading_deg)
        return abs(head_err) > waypoint.heading_tolerance_deg

    def _derivative(self, err_rad: float, now: float) -> float:
        prev_err, prev_t = self._prev_err_rad, self._prev_time
        self._prev_err_rad, self._prev_time = err_rad, now
        if prev_err is None or prev_t is None:
            return 0.0
        dt = now - prev_t
        if dt <= 1e-3:
            return 0.0
        return (err_rad - prev_err) / dt

    def _halt(
        self,
        distance_cm: float,
        err_deg: float,
        bearing: float,
        reason: str,
    ) -> ControlCommand:
        """안전 정지 명령(zero) 을 만들고 미분 상태를 초기화한다."""
        self.reset()
        return ControlCommand(
            throttle=0.0,
            steering=0.0,
            mode=ControlMode.HOLD,
            arrived=False,
            distance_error_cm=distance_cm,
            heading_error_deg=err_deg,
            target_bearing_deg=bearing,
            reason=reason,
        )
