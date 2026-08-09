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
from .models import ControlCommand, ControlMode, Pose, Waypoint


class PoseWaypointController:
    """단일 waypoint 안정 추종용 1차 제어기 (P/PD + 보수적 throttle 스케줄)."""

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()
        self._prev_err_rad: Optional[float] = None
        self._prev_time: Optional[float] = None

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        """PD 미분 상태 초기화. mode 전환/재접속 시 호출 권장."""
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
        # 목표에 닿았는지는 위치만으로 정해진다. heading 게이트를 먼저 두면,
        # 이미 목표 위에 있는데 방향을 아직 못 구한 차량이 영원히 다음 waypoint 로
        # 넘어가지 못한다(정지 상태에서는 궤적으로 heading 을 못 구하므로 교착).
        # 방향이 필요한 waypoint(heading_required)일 때만 heading 을 요구한다.
        brake_radius = cfg.brake_radius_cm(waypoint.position_tolerance_cm)
        if distance_cm <= brake_radius:
            if waypoint.heading_required and not pose.has_heading:
                return self._halt(distance_cm, 0.0, bearing, "NO_HEADING")
            if pose.has_heading and self._needs_alignment(waypoint, pose.heading_deg):
                head_err = geo.wrap180(
                    (waypoint.target_heading_deg or 0.0) - pose.heading_deg)
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
            self.reset()  # 도착: 미분 상태 정리
            return ControlCommand(
                throttle=0.0,
                steering=0.0,
                mode=ControlMode.ARRIVED,
                arrived=True,
                distance_error_cm=distance_cm,
                heading_error_deg=0.0 if not pose.has_heading else geo.heading_error_deg(
                    bearing, pose.heading_deg),
                target_bearing_deg=bearing,
                reason="ARRIVED",
            )

        # ---- 주행하려면 heading 이 있어야 한다 -------------------------------
        if not pose.has_heading:
            return self._halt(distance_cm, 0.0, bearing, "NO_HEADING")

        heading = pose.heading_deg  # not None (위에서 보장)
        err_deg = geo.heading_error_deg(bearing, heading)
        err_rad = math.radians(err_deg)

        # PD 미분 항 갱신(안전 게이트 통과 후에만). 미분은 이후 halt 시 reset.
        derr = self._derivative(err_rad, now)

        if not allow_drive:
            return self._halt(distance_cm, err_deg, bearing, "DRIVE_NOT_ALLOWED")

        # ---- 정상 주행: steering / throttle --------------------------------
        logical_steer, wire_steer = self._steering(err_rad, derr)
        throttle = self._throttle(waypoint, distance_cm, err_deg)
        mode = ControlMode.DRIVE if throttle > 0.0 else ControlMode.BRAKE

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

    def _throttle(self, waypoint: Waypoint, distance_cm: float, err_deg: float) -> float:
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

        # 잠정 물리 매핑 (미보정) + 상/하한.
        throttle = desired_cm_s * cfg.throttle_per_cm_s
        throttle = max(throttle, cfg.min_move_throttle)
        throttle = geo.clamp(throttle, 0.0, cfg.max_throttle)
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
