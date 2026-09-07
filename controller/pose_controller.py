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
        self._align_capture_key: tuple[object, ...] | None = None
        self._align_in_tolerance_timestamp: float | None = None
        # 마지막으로 **다른** 관측 (x, y, obs_time). 관측 사이 실제 속도를
        # 재는 데 쓴다. waypoint.speed_cm_s 는 요구값이지 실측이 아니다 —
        # 실차에서 FINAL 요구 40mm/s 에 실제 134mm/s 가 나왔다(3.3배).
        self._last_obs: tuple[float, float, float] | None = None
        self._observed_speed_mm_s: float = 0.0
        self._telemetry: dict[str, object] = {}

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        """mode 전환/재접속용 전체 제어상태 초기화."""
        self._reset_derivative()
        self._motion_direction = None
        self._align_capture_key = None
        self._align_in_tolerance_timestamp = None
        self._last_obs = None
        self._observed_speed_mm_s = 0.0

    def _reset_derivative(self) -> None:
        """PD 미분 이력만 초기화하고 구동 방향 latch는 보존한다."""
        self._prev_err_rad = None
        self._prev_time = None

    def _update_observed_speed(self, pose: Pose) -> None:
        """서로 다른 관측 두 개로 실제 속도를 잰다 (mm/s).

        같은 관측이 여러 tick 에 걸쳐 반복되므로 obs_time 이 바뀔 때만
        갱신한다. 갱신하지 않는 동안 마지막 추정치를 유지해야 stale 구간
        에서도 "이 속도로 가면 얼마나 가는가" 를 물어볼 수 있다.
        """
        last = self._last_obs
        if last is not None and pose.timestamp <= last[2]:
            return
        if last is not None:
            dt = pose.timestamp - last[2]
            if dt > 0.0:
                self._observed_speed_mm_s = math.hypot(
                    pose.x_mm - last[0], pose.y_mm - last[1]) / dt
        self._last_obs = (pose.x_mm, pose.y_mm, pose.timestamp)

    def _final_terminal_geometry(
        self, pose: Pose, waypoint: Waypoint
    ) -> tuple[float, float] | None:
        """(목표까지 진행축 거리, 목표 너머까지 포함한 물리 여유), mm.

        planner가 rear FINAL에 넣은 clearance가 있어야만 계약이 활성화된다.
        따라서 일반 GLOBAL/setup/recovery terminal의 기존 의미는 바뀌지 않는다.
        """
        clearance = waypoint.terminal_motion_clearance_mm
        if (
            clearance is None
            or waypoint.motion_direction is not MotionDirection.REVERSE
            or (waypoint.phase or "").upper() != "FINAL"
            or waypoint.target_heading_deg is None
        ):
            return None
        motion_deg = waypoint.target_heading_deg + 180.0
        ux = math.cos(math.radians(motion_deg))
        uy = math.sin(math.radians(motion_deg))
        remaining = ((waypoint.x_mm - pose.x_mm) * ux
                     + (waypoint.y_mm - pose.y_mm) * uy)
        return remaining, max(0.0, remaining + max(0.0, clearance))

    def _is_arrival_candidate(
        self, pose: Pose, waypoint: Waypoint, distance_cm: float
    ) -> bool:
        """현재 pose가 ARRIVED/FINAL confirmation으로 넘어갈 수 있는가.

        일반 waypoint는 기존 radial contract를 그대로 쓴다. rear FINAL은
        generic 50mm 반경 안이라도 진행축 overshoot가 terminal clearance의
        안전 부분을 넘으면 DONE으로 삼지 않는다. 이후 품질(heading/lateral)은
        기존 FINAL_POSE_EVAL이 담당한다.
        """
        arrival_cm = self.config.arrival_radius_cm(
            waypoint.position_tolerance_cm, waypoint.phase)
        if distance_cm > arrival_cm:
            return False
        geometry = self._final_terminal_geometry(pose, waypoint)
        if geometry is None:
            return True
        remaining_mm, _ = geometry
        safe_overrun_mm = max(
            0.0,
            float(waypoint.terminal_motion_clearance_mm or 0.0)
            - self.config.final_measurement_uncertainty_mm,
        )
        return remaining_mm >= -safe_overrun_mm

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
        self._telemetry = {
            "controller_compute_ts": now,
            "pose_timestamp": pose.timestamp,
            "pose_age_ms": max(0.0, now - pose.timestamp) * 1000.0,
            "pose_fresh": bool(pose.valid and (now - pose.timestamp) <= cfg.max_pose_age_s),
            "dx_to_target_mm": dx,
            "dy_to_target_mm": dy,
            "distance_to_target_mm": distance_cm * 10.0,
            "arrival_position_error_mm": distance_cm * 10.0,
            "desired_bearing_deg": bearing,
            "heading_required": waypoint.heading_required,
            "arrival_distance_tolerance_mm": (
                cfg.arrival_radius_cm(waypoint.position_tolerance_cm,
                                      waypoint.phase) * 10.0),
            "arrival_heading_tolerance_deg": waypoint.heading_tolerance_deg,
            "target_curvature": waypoint.curvature,
            "control_override_reason": "NONE",
        }
        telemetry = self._telemetry
        if waypoint.target_heading_deg is not None:
            axis = math.radians(waypoint.target_heading_deg)
            ux, uy = math.cos(axis), math.sin(axis)
            telemetry.update({
                "along_track_error_mm": dx * ux + dy * uy,
                "cross_track_error_mm": -dx * uy + dy * ux,
                "cross_track_definition": "SIGNED_TARGET_AXIS_LATERAL_ERROR",
            })

        # ---- 안전 게이트 (무조건 zero) --------------------------------------
        if not pose.valid:
            return self._halt(distance_cm, 0.0, bearing, "POSE_INVALID")
        if (now - pose.timestamp) > cfg.max_pose_age_s:
            return self._halt(distance_cm, 0.0, bearing, "POSE_STALE")

        # ---- blind travel 공간 계약 (정지해야 하는 waypoint 한정) -----------
        #
        # max_pose_age_s 는 **시간** 기준이라 거리를 모른다. FINAL 후진에서
        # 실측 속도는 134mm/s 였고 슬롯 깊이 여유는 25mm 뿐이다. 즉 카메라가
        # 정상(4.3fps, 233ms)이어도 프레임 하나 사이에 31mm 를 움직여 여유를
        # 이미 넘는다. 500ms 를 다 쓰면 67mm — 여유의 2.7배다.
        # 실측 run_20260901_154551: 목표까지 65mm 남은 상태에서 547ms 동안
        # 관측 없이 78.7mm 를 더 가 맵 밖 56.8mm 로 나갔고, BOUNDARY_HARD 가
        # 먼저 걸려 FINAL_POSE_EVAL 이 아예 실행되지 않았다.
        #
        # 계약: **관측 없이 목표를 지나칠 수 있으면 움직이지 않는다.**
        # 허용 blind 거리는 목표까지 남은 거리다 — 멀면 넉넉하고, 가까워질수록
        # 0 으로 수렴해 종점 근처에서는 신선한 관측을 강제한다.
        # 통과해야 하는 중간 waypoint 에는 걸지 않는다 (거기서 정지는 진행을
        # 끊는다). max_pose_age_s 를 낮추지 않으므로 다른 구간은 그대로다.
        self._update_observed_speed(pose)
        terminal_geometry = self._final_terminal_geometry(pose, waypoint)
        arrival_candidate = self._is_arrival_candidate(
            pose, waypoint, distance_cm)
        telemetry["arrival_position_ok"] = arrival_candidate
        telemetry["observed_speed_mm_s"] = self._observed_speed_mm_s
        if (
            waypoint.is_final
            and cfg.blind_travel_guard
            and terminal_geometry is not None
            and not arrival_candidate
        ):
            # arrival candidate이면 이미 정지/confirmation의 책임 영역이다.
            # 주행 중일 때만 blind + stopping + measurement budget을 terminal
            # 이후 slot/map clearance까지 포함한 available margin과 비교한다.
            _, available_mm = terminal_geometry
            speed_mm_s = max(
                self._observed_speed_mm_s,
                cfg.final_speed_estimate_floor_mm_s,
                max(0.0, waypoint.speed_cm_s) * 10.0,
            )
            pose_age_s = max(0.0, now - pose.timestamp)
            required_mm = (
                speed_mm_s * pose_age_s
                + speed_mm_s * cfg.final_stopping_time_s
                + cfg.final_measurement_uncertainty_mm
            )
            telemetry.update({
                "terminal_available_mm": available_mm,
                "terminal_required_mm": required_mm,
            })
            if required_mm >= available_mm:
                return self._halt(distance_cm, 0.0, bearing,
                                  "POSE_BLIND_TRAVEL",
                                  preserve_observed_speed=True)

        # ---- 도착 판정은 heading 없이도 가능하다 (backend 결선 시 수정) --------
        # 목표에 닿았는지는 위치만으로 정해진다. heading 게이트를 앞에 두면,
        # 이미 목표 위에 있는데 방향을 못 구한 차량이 영원히 다음 waypoint 로
        # 넘어가지 못한다 (정지 상태에서는 궤적으로 heading 을 못 구해 교착).
        # 방향이 필요한 waypoint(heading_required)일 때만 heading 을 요구한다.
        if not pose.has_heading:
            if arrival_candidate and not waypoint.heading_required:
                self.reset()
                return ControlCommand(
                    throttle=0.0, steering=0.0, mode=ControlMode.ARRIVED,
                    arrived=True, distance_error_cm=distance_cm,
                    heading_error_deg=0.0, target_bearing_deg=bearing,
                    reason="ARRIVED",
                    telemetry=dict(telemetry),
                )
            return self._halt(distance_cm, 0.0, bearing, "NO_HEADING")

        heading = pose.heading_deg  # not None (위에서 보장)
        direction = waypoint.motion_direction
        reverse = direction is MotionDirection.REVERSE
        telemetry["reverse"] = reverse
        telemetry["body_heading_deg"] = heading

        # 후진은 명시적으로 허용된 주차/복구 phase에서만 실행한다.
        if reverse:
            if not cfg.allow_reverse:
                return self._halt(distance_cm, 0.0, bearing, "REVERSE_NOT_ALLOWED")
            phase = (waypoint.phase or "").upper()
            if phase not in cfg.reverse_allowed_phases:
                return self._halt(distance_cm, 0.0, bearing, "REVERSE_PHASE_NOT_ALLOWED")
            # 궤적 기반 heading 은 후진에서 180° 뒤집힌다. 그 값으로 조향하면
            # 정확히 반대로 꺾으므로, 신뢰 가능한 heading 이 올 때까지 멈춘다.
            if cfg.reverse_heading_unsafe(waypoint.phase, pose.heading_source,
                                          reverse=True):
                return self._halt(distance_cm, 0.0, bearing,
                                  "REVERSE_HEADING_UNSAFE")

        # 후진에서는 실제 이동방향이 보통 body heading + 180°다.  다만 카메라
        # pose 기준점(차체 중심)은 선회 중 rear axle와 다른 접선을 가지며, 실차
        # 세 run에서 FRONT_CUSHION body heading과 중심점 이동 접선이 8--11°
        # 벌어졌다. 확인된 motion heading이 있으면 경로 guidance에만 사용한다.
        # 도착/최종 자세의 heading 판정은 아래에서 계속 body heading을 쓴다.
        motion_heading = geo.wrap180(
            pose.motion_heading_deg
            if reverse and pose.motion_heading_deg is not None
            else heading + (180.0 if reverse else 0.0)
        )
        # A curved segment must follow its planned circle, not continuously aim
        # at the segment endpoint.  Endpoint bearing is the chord direction; on
        # a finite arc it differs from the local tangent by half the remaining
        # sweep and can therefore cancel (or amplify) the curvature feedforward.
        guidance_heading = bearing
        if waypoint.curvature and waypoint.target_heading_deg is not None:
            guidance_heading = self._arc_guidance_heading(
                pose, waypoint, reverse=reverse, telemetry=telemetry)
        err_deg = geo.heading_error_deg(guidance_heading, motion_heading)
        err_rad = math.radians(err_deg)
        telemetry.update({
            "motion_heading_deg": motion_heading,
            "guidance_heading_deg": guidance_heading,
            "bearing_error_deg": geo.heading_error_deg(bearing, motion_heading),
            "steering_error_deg": err_deg,
            "target_heading_error_deg": (
                None if waypoint.target_heading_deg is None
                else geo.wrap180(waypoint.target_heading_deg - heading)),
            "arrival_heading_error_deg": (
                None if waypoint.target_heading_deg is None
                else geo.wrap180(waypoint.target_heading_deg - heading)),
        })

        if not allow_drive:
            return self._halt(distance_cm, err_deg, bearing, "DRIVE_NOT_ALLOWED")

        # ---- 도착/정렬 판정 ------------------------------------------------
        settled_alignment = self._settled_alignment_capture(
            pose, waypoint, distance_cm=distance_cm)
        telemetry["arrival_heading_ok"] = not self._needs_alignment(
            waypoint, heading)
        telemetry["settled_alignment_capture"] = settled_alignment
        continue_arc_alignment = False
        if arrival_candidate:
            needs_alignment = self._needs_alignment(waypoint, heading)
            if needs_alignment and not settled_alignment:
                # A curved primitive can enter its radial position tolerance
                # before reaching the endpoint tangent.  Stopping here made the
                # real route-6 ALIGN fail at (890,190,339.6deg), although only
                # 33 mm of safe on-circle travel remained and the stopped pose
                # reached (936,168,330.5deg).  The same ordering also let setup
                # RECOVERY primitives finish before performing their turn.
                # Continue only while the endpoint is still ahead *and* the
                # vehicle remains inside the already-declared arc corridor.
                if not self._arc_endpoint_approaching_in_corridor(
                        pose, waypoint, reverse=reverse):
                    head_err = geo.wrap180(
                        (waypoint.target_heading_deg or 0.0) - heading)
                    # 전진 전용 1차 controller 는 제자리 회전을 하지 않는다.
                    # 위치는 도착했으니 정지하고 heading 오차만 보고
                    # (정렬 기동은 상위 recovery coordinator 책임).
                    return ControlCommand(
                        throttle=0.0,
                        steering=0.0,
                        mode=ControlMode.ALIGN,
                        arrived=False,
                        distance_error_cm=distance_cm,
                        heading_error_deg=head_err,
                        target_bearing_deg=bearing,
                        reason="HEADING_OUT_OF_TOLERANCE",
                        telemetry=dict(telemetry),
                    )
                continue_arc_alignment = True
            else:
                # 다음 waypoint가 FORWARD<->REVERSE로 바뀌는지 확인해야 하므로
                # 방향 latch는 보존하고 PD 이력만 지운다.
                self._reset_derivative()
                return ControlCommand(
                    throttle=0.0,
                    steering=0.0,
                    mode=ControlMode.ARRIVED,
                    arrived=True,
                    distance_error_cm=distance_cm,
                    heading_error_deg=(
                        geo.wrap180((waypoint.target_heading_deg or 0.0) - heading)
                        if needs_alignment else err_deg),
                    target_bearing_deg=bearing,
                    reason=("ALIGN_SETTLED_CAPTURE" if needs_alignment
                            else "ARRIVED"),
                    telemetry=dict(telemetry),
                )

        # Curved non-terminal waypoints are samples of one continuous path.
        # If the vehicle crosses the endpoint tangent inside the planned-circle
        # corridor, chasing that sample backwards creates an orbit/replan loop.
        # This is deliberately separate from (and does not enlarge) the point
        # arrival radius; FINAL still requires the ordinary exact capture.
        if self._arc_endpoint_captured(pose, waypoint, reverse=reverse):
            needs_alignment = self._needs_alignment(waypoint, heading)
            if needs_alignment and not settled_alignment:
                head_err = geo.wrap180((waypoint.target_heading_deg or 0.0) - heading)
                return ControlCommand(
                    throttle=0.0, steering=0.0, mode=ControlMode.ALIGN,
                    arrived=False, distance_error_cm=distance_cm,
                    heading_error_deg=head_err, target_bearing_deg=bearing,
                    reason="HEADING_OUT_OF_TOLERANCE",
                    telemetry=dict(telemetry),
                )
            self._reset_derivative()
            return ControlCommand(
                throttle=0.0, steering=0.0, mode=ControlMode.ARRIVED,
                arrived=True, distance_error_cm=distance_cm,
                heading_error_deg=(
                    geo.wrap180((waypoint.target_heading_deg or 0.0) - heading)
                    if needs_alignment else err_deg),
                target_bearing_deg=bearing,
                reason=("ALIGN_SETTLED_CAPTURE" if needs_alignment
                        else "ARC_ENDPOINT_PASSED"),
                telemetry=dict(telemetry),
            )

        # Once an arc endpoint tangent has been crossed outside its corridor,
        # chasing the old sample cannot restore the intended continuous path.
        # Stop before loading the next phase; the parking coordinator will use
        # a fresh pose for bounded setup/replan.
        if self._arc_endpoint_missed(pose, waypoint, reverse=reverse):
            return self._halt(distance_cm, err_deg, bearing,
                              "ARC_CORRIDOR_MISSED")

        # 전/후진 방향이 주행 중 바뀌면 한 tick zero를 넣어 DIR 급전환을 막는다.
        if self._motion_direction is not None and direction is not self._motion_direction:
            self._motion_direction = direction
            self._prev_err_rad = None
            self._prev_time = None
            return ControlCommand(
                throttle=0.0, steering=0.0, mode=ControlMode.HOLD, arrived=False,
                distance_error_cm=distance_cm, heading_error_deg=err_deg,
                target_bearing_deg=bearing, reason="DIRECTION_CHANGE_STOP",
                telemetry=dict(telemetry),
            )
        self._motion_direction = direction

        # PD 미분 항 갱신. 후진은 조향의 물리 효과가 반대라 제어 부호를 반전한다.
        previous_err_rad = self._prev_err_rad
        previous_time = self._prev_time
        derr = self._derivative(err_rad, now)
        steering_err_rad = -err_rad if reverse else err_rad
        steering_derr = -derr if reverse else derr
        telemetry.update({
            "steering_error_previous_rad": previous_err_rad,
            "controller_dt_s": (
                None if previous_time is None else max(0.0, now - previous_time)),
            "steering_error_after_reverse_rad": steering_err_rad,
            "steering_derivative_after_reverse_rad_s": steering_derr,
        })

        # ---- 정상 주행: steering / throttle --------------------------------
        # FINAL 직선 후진이 이미 슬롯 축에 정렬돼 있으면 11자로 고정한다.
        # 정렬된 상태에서 끝점 bearing 을 향한 PD 는 목표에 가까워질수록
        # 조향을 키워 firmware turn-duty 를 밟고, 그 PWM 상승이 정지거리를
        # 늘려 슬롯 뒤(맵 경계)를 넘는다. 판정 임계값은 이 waypoint 의 FINAL
        # 도착 허용오차를 그대로 쓴다 (config 주석 참조).
        if (reverse and cfg.reverse_steering_locked(waypoint.phase)
                and not waypoint.curvature):
            # 11자 후진 — 곧게 물러난다 (RECOVERY 전용, config 주석 참조).
            # 곡률 0인 기존 recovery만 잠근다. setup recovery는 같은 phase를
            # 재사용하지만 명시적 곡률이 있어 계획한 원호를 타야 한다.
            logical_steer, wire_steer = 0.0, 0.0
            telemetry.update({
                "reverse_steering_lock_active": True,
                "steering_feedforward_term": 0.0,
                "steering_after_controller_clamp": 0.0,
                "steering_after_wire_sign": 0.0,
            })
        else:
            # 경로 곡률만큼 미리 넣고(feedforward), PD 는 오차 보정만 한다.
            # 원호에서 "오차가 생긴 뒤에야 꺾는" 지연이 사라진다.
            feed_fwd = cfg.feedforward_steering(waypoint.phase, waypoint.curvature,
                                               reverse=reverse)
            logical_steer, wire_steer = self._steering(
                steering_err_rad, steering_derr, feedforward=feed_fwd,
                use_available_headroom=(
                    reverse and pose.motion_heading_deg is not None),
                telemetry=telemetry,
            )
            telemetry["reverse_steering_lock_active"] = False
            # FINAL 직선 후진에서 차가 이미 슬롯 축에 정렬돼 있으면 조향을
            # **제한**한다(0 으로 잠그지 않는다). 정렬된 상태에서 끝점 bearing
            # 을 향한 PD 는 목표에 가까워질수록 조향을 키워(atan(횡오차/거리),
            # 거리→0) firmware turn-duty 를 밟는다. 실측 sim: 5mm 섭동에도
            # 종점 근처에서 |steer|→1.0 으로 포화했다. 그 PWM 상승이 개루프
            # 속도를 높여 정지거리를 늘리고 슬롯 뒤(맵 경계)를 넘는다
            # (실차 191010~224451 depth +51~88mm). cap 은 crab 보정에 필요한
            # 완만한 조향은 남기고 종점 포화만 깎는다 — 0 lock 은 8~11° crab
            # 을 미보정으로 남겨 작은 오차를 오히려 키운다(sim 확인).
            # cap 은 firmware forward↔turn duty 전이 중앙(|steer|=0.25,
            # duty≈pwm_forward 와 pwm_turn 의 중간)에서 유도한다 — 그 위로는
            # 조향이 duty 를 급히 turn 영역으로 밀어올린다.
            if (waypoint.is_final and not waypoint.curvature
                    and cfg.final_reverse_straight_when_aligned
                    and self._final_reverse_axis_aligned(pose, waypoint)):
                cap = cfg.final_reverse_aligned_steer_cap
                if abs(wire_steer) > cap:
                    wire_steer = math.copysign(cap, wire_steer)
                if abs(logical_steer) > cap:
                    logical_steer = math.copysign(cap, logical_steer)
                telemetry["final_phase_steering_cap_active"] = bool(
                    abs(float(telemetry.get(
                        "steering_after_wire_sign", wire_steer))) > cap)
            else:
                telemetry["final_phase_steering_cap_active"] = False
        telemetry.update({
            "steering_after_phase_cap": logical_steer,
            "steering_command_final": wire_steer,
            "steering_cap_active": bool(
                telemetry.get("steering_saturated", False)
                or telemetry.get("final_phase_steering_cap_active", False)),
        })
        throttle_mag = self._throttle(
            waypoint, distance_cm, err_deg, reverse=reverse,
            wire_steering=wire_steer,
            telemetry=telemetry,
        )
        if continue_arc_alignment and throttle_mag <= 0.0:
            # The ordinary distance schedule reaches zero inside stop_distance,
            # but this branch has deliberately not arrived: a small amount of
            # measured, on-circle motion is still required to reach the endpoint
            # tangent.  Reuse the phase's existing motion floor and ceiling.
            throttle_mag = min(
                cfg.min_move_throttle_for(waypoint.phase, reverse=reverse),
                cfg.throttle_limit(waypoint.phase, reverse=reverse),
            )
            ceiling = cfg.final_throttle_ceiling(
                waypoint.phase, reverse=reverse)
            if ceiling is not None:
                throttle_mag = min(throttle_mag, ceiling)
        throttle = -throttle_mag if reverse else throttle_mag
        mode = ControlMode.DRIVE if abs(throttle) > 0.0 else ControlMode.BRAKE
        telemetry.update({
            "throttle_command_final": throttle,
            "waypoint_arrived": False,
        })

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
            telemetry=dict(telemetry),
        )

    # ----------------------------------------------------------------- private
    def _final_reverse_axis_aligned(self, pose: Pose,
                                    waypoint: Waypoint) -> bool:
        """FINAL 직선 후진에서 차가 이미 슬롯 축에 정렬됐는가.

        정렬 = body heading 이 목표 heading 허용오차 안이고, 차체 중심이 슬롯
        축선에서 목표 위치 허용오차 안. 두 임계값 모두 waypoint 가 이미 들고
        있는 FINAL 도착 허용오차라 새 튜닝값이 아니다. FINAL 자세 품질 판정
        (heading/lateral/footprint)은 상위 FINAL_POSE_EVAL 이 그대로 하므로,
        여기서 조향을 잠근다고 잘못 정렬된 자세가 PARKED 로 인정되지 않는다.
        """
        if waypoint.target_heading_deg is None:
            return False
        head_err = abs(geo.wrap180(waypoint.target_heading_deg - pose.heading_deg))
        if head_err > waypoint.heading_tolerance_deg:
            return False
        axis_rad = math.radians(waypoint.target_heading_deg)
        ux, uy = math.cos(axis_rad), math.sin(axis_rad)
        dx = pose.x_mm - waypoint.x_mm
        dy = pose.y_mm - waypoint.y_mm
        cross_track_mm = abs(-dx * uy + dy * ux)
        return cross_track_mm <= waypoint.position_tolerance_cm * 10.0

    def _steering(self, err_rad: float, derr_rad_s: float,
                  *, feedforward: float = 0.0,
                  use_available_headroom: bool = False,
                  telemetry: dict[str, object] | None = None) -> tuple[float, float]:
        """(논리 steering, wire steering) 반환.

        논리 steering: 양수 = LEFT 요구 (heading_error > 0 과 같은 부호).
        wire steering : ESP32 실제 부호(음수 = LEFT). = wire_steering_sign * 논리.

        feedforward 는 경로 곡률에서 온 기본 조향이고, PD 항이 그 위에
        오차 보정을 얹는다. 직선 구간은 feedforward=0 이라 기존과 동일하다.
        """
        cfg = self.config
        proportional = cfg.steer_kp * err_rad
        derivative = cfg.steer_kd * derr_rad_s
        raw = proportional + derivative
        norm = raw / math.radians(cfg.steer_normalize_deg)
        norm_unclamped = norm
        feedback_limit = None
        if feedforward:
            # 곡률 추종 중에는 PD 를 보정 폭 안으로 묶는다 (config 주석 참조).
            lim = cfg.curvature_feedback_limit
            if use_available_headroom:
                # Confirmed motion-tangent feedback has the correct path-error
                # sign.  Let it use the actuator range that remains after
                # feedforward; the old fixed 0.25 cap left 7--16% unused in the
                # R=1000/1100 real routes while radial error grew to 75--80 mm.
                # Total steering is still hard-clamped below.
                lim = max(lim, 1.0 - abs(feedforward))
            norm = geo.clamp(norm, -lim, lim)
            feedback_limit = lim
        logical_unclamped = feedforward + norm
        logical = geo.clamp(logical_unclamped, -1.0, 1.0)
        wire = geo.clamp(cfg.wire_steering_sign * logical, -1.0, 1.0)
        if telemetry is not None:
            telemetry.update({
                "steering_error_rad": err_rad,
                "steering_error_derivative_rad_s": derr_rad_s,
                "steering_proportional_term": proportional,
                "steering_derivative_term": derivative,
                "steering_raw": raw,
                "steering_normalized_raw": norm_unclamped,
                "steering_feedforward_term": feedforward,
                "steering_feedback_limit": feedback_limit,
                "steering_feedback_after_limit": norm,
                "steering_logical_before_clamp": logical_unclamped,
                "steering_after_controller_clamp": logical,
                "steering_saturated": abs(logical_unclamped - logical) > 1e-12,
                "steering_after_wire_sign": wire,
            })
        # -0.0 정규화(로그 가독성)
        logical = round(logical, 4) or 0.0
        wire = round(wire, 4) or 0.0
        return logical, wire

    @staticmethod
    def _arc_guidance_heading(
        pose: Pose,
        waypoint: Waypoint,
        *,
        reverse: bool,
        telemetry: dict[str, object] | None = None,
    ) -> float:
        """Return the motion tangent of the waypoint's planned circle.

        ``waypoint.curvature`` follows the planner contract ``d(body_heading)
        = curvature * ds`` where reverse travel has negative ``ds``.  Guidance
        is expressed in the actual motion direction, so reverse motion uses the
        opposite signed curvature and the body endpoint heading + 180 degrees.

        The radial term is a geometry-scaled cross-track correction.  It is
        exactly zero on the planned circle and uses ``atan(error / radius)``;
        no new vehicle-specific gain or curvature calibration is introduced.
        """
        body_curvature = float(waypoint.curvature)
        motion_curvature = -body_curvature if reverse else body_curvature
        end_motion_heading = geo.wrap180(
            float(waypoint.target_heading_deg) + (180.0 if reverse else 0.0))
        heading_rad = math.radians(end_motion_heading)
        radius = 1.0 / abs(motion_curvature)

        # Signed-curvature circle centre: p + left_normal(heading) / k.
        center_x = waypoint.x_mm - math.sin(heading_rad) / motion_curvature
        center_y = waypoint.y_mm + math.cos(heading_rad) / motion_curvature
        dx = pose.x_mm - center_x
        dy = pose.y_mm - center_y
        radial_distance = math.hypot(dx, dy)
        radial_angle = math.degrees(math.atan2(dy, dx))
        tangent = radial_angle + (90.0 if motion_curvature > 0.0 else -90.0)

        radial_error = radial_distance - radius
        cross_track_correction = math.degrees(math.atan2(radial_error, radius))
        if motion_curvature < 0.0:
            cross_track_correction = -cross_track_correction
        guidance = geo.wrap180(tangent + cross_track_correction)
        if telemetry is not None:
            telemetry.update({
                "cross_track_error_mm": radial_error,
                "cross_track_definition": "SIGNED_ARC_RADIAL_ERROR",
                "arc_tangent_heading_deg": geo.wrap180(tangent),
                "arc_cross_track_correction_deg": cross_track_correction,
            })
        return guidance

    def _throttle(
        self,
        waypoint: Waypoint,
        distance_cm: float,
        err_deg: float,
        *,
        reverse: bool = False,
        wire_steering: float = 0.0,
        telemetry: dict[str, object] | None = None,
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
        if telemetry is not None:
            telemetry.update({
                "turn_scale": turn_scale,
                "approach_scale": approach_scale,
                "desired_speed_cm_s": desired_cm_s,
                "throttle_requested_raw": desired_cm_s * cfg.throttle_per_cm_s,
            })
        if desired_cm_s <= 0.0:
            if telemetry is not None:
                telemetry.update({
                    "throttle_after_stage_limit": 0.0,
                    "throttle_after_parking_limit": 0.0,
                    "throttle_after_safety_limit": 0.0,
                    "parking_throttle_cap_active": False,
                    "throttle_saturated": False,
                })
            return 0.0

        # phase별 저속 floor / 상한. 일반 CRUISE baseline은 유지하면서
        # 주차/Recovery에서는 waypoint speed가 실제 throttle 차이로 남게 한다.
        raw_throttle = desired_cm_s * cfg.throttle_per_cm_s
        throttle = raw_throttle
        throttle = max(
            throttle,
            cfg.min_move_throttle_for(waypoint.phase, reverse=reverse),
        )
        throttle = geo.clamp(
            throttle,
            0.0,
            cfg.throttle_limit(waypoint.phase, reverse=reverse),
        )
        stage_limited = throttle

        # 최대 조향 정지 마찰 극복 (config 주석 참조).
        # 상한 clamp **뒤에** 적용한다 — max_throttle 은 속도 상한이고, 최대
        # 조향에서는 duty 를 올려도 차가 기어가듯 움직이기 때문이다.
        # 단 후진 정밀 주차는 제외한다 — 거기서는 속도 상한이 우선이다.
        floor = cfg.stiction_floor_for(waypoint.phase, reverse=reverse)
        if floor is not None and abs(wire_steering) >= cfg.strong_turn_steering:
            throttle = max(throttle, floor)
        after_stiction = throttle

        # 마지막 안전 clamp — 어떤 조합(정지마찰/곡률/조향포화)이 와도
        # 후진과 정밀 주차 구간이 상한을 넘지 못하게 한다.
        ceiling = cfg.final_throttle_ceiling(waypoint.phase, reverse=reverse)
        if ceiling is not None:
            throttle = min(throttle, ceiling)

        if telemetry is not None:
            limit = cfg.throttle_limit(waypoint.phase, reverse=reverse)
            telemetry.update({
                "throttle_after_stage_limit": stage_limited,
                "throttle_after_parking_limit": stage_limited,
                "throttle_after_stiction_floor": after_stiction,
                "throttle_after_safety_limit": throttle,
                "current_phase_throttle_limit": limit,
                "parking_throttle_cap_active": stage_limited + 1e-12 < max(
                    raw_throttle,
                    cfg.min_move_throttle_for(waypoint.phase, reverse=reverse)),
                "throttle_saturated": throttle + 1e-12 < raw_throttle,
                "strong_turn_floor_active": after_stiction > stage_limited + 1e-12,
            })

        return round(throttle, 4)

    def _needs_alignment(self, waypoint: Waypoint, heading_deg: float) -> bool:
        if not waypoint.heading_required or waypoint.target_heading_deg is None:
            return False
        head_err = geo.wrap180(waypoint.target_heading_deg - heading_deg)
        return abs(head_err) > waypoint.heading_tolerance_deg

    def _settled_alignment_capture(
        self,
        pose: Pose,
        waypoint: Waypoint,
        *,
        distance_cm: float,
    ) -> bool:
        """Accept only a small subsequent ALIGN endpoint overshoot.

        This is state/geometry hysteresis, not a larger one-shot tolerance.
        Evidence is remembered only in the endpoint corridor and only for the
        same route waypoint.  Repeated controller ticks carrying the same
        camera timestamp do not count as a prior convergence.
        """
        phase = (waypoint.phase or "").upper()
        corridor_cm = waypoint.path_capture_tolerance_cm
        eligible = bool(
            phase == "ALIGN"
            and waypoint.heading_required
            and waypoint.target_heading_deg is not None
            and waypoint.curvature
            and not waypoint.is_final
            and corridor_cm is not None
            and corridor_cm > 0.0
        )
        key = (
            waypoint.route_id, waypoint.waypoint_id,
            waypoint.x_mm, waypoint.y_mm, waypoint.target_heading_deg,
        )
        if key != self._align_capture_key:
            self._align_capture_key = key
            self._align_in_tolerance_timestamp = None
        if not eligible:
            return False

        error = abs(geo.wrap180(
            float(waypoint.target_heading_deg) - float(pose.heading_deg)))
        previous_timestamp = self._align_in_tolerance_timestamp
        settled = bool(
            previous_timestamp is not None
            and pose.timestamp > previous_timestamp
            and error <= (waypoint.heading_tolerance_deg
                          + self.config.align_settled_hysteresis_deg)
        )
        if (distance_cm <= float(corridor_cm)
                and error <= waypoint.heading_tolerance_deg):
            self._align_in_tolerance_timestamp = pose.timestamp
        return settled

    @staticmethod
    def _arc_endpoint_approaching_in_corridor(
        pose: Pose,
        waypoint: Waypoint,
        *,
        reverse: bool,
    ) -> bool:
        """Whether a heading-required arc endpoint is still safely ahead."""
        tolerance_cm = waypoint.path_capture_tolerance_cm
        if (tolerance_cm is None or tolerance_cm <= 0.0 or waypoint.is_final
                or not waypoint.heading_required or not waypoint.curvature
                or waypoint.target_heading_deg is None):
            return False
        motion_curvature = (-waypoint.curvature if reverse
                            else waypoint.curvature)
        end_motion_heading = geo.wrap180(
            waypoint.target_heading_deg + (180.0 if reverse else 0.0))
        heading_rad = math.radians(end_motion_heading)
        progress_mm = (
            (pose.x_mm - waypoint.x_mm) * math.cos(heading_rad)
            + (pose.y_mm - waypoint.y_mm) * math.sin(heading_rad)
        )
        if progress_mm >= 0.0:
            return False
        center_x = waypoint.x_mm - math.sin(heading_rad) / motion_curvature
        center_y = waypoint.y_mm + math.cos(heading_rad) / motion_curvature
        radius_mm = 1.0 / abs(motion_curvature)
        radial_error_mm = abs(
            math.hypot(pose.x_mm - center_x, pose.y_mm - center_y) - radius_mm)
        return radial_error_mm <= tolerance_cm * 10.0

    @staticmethod
    def _arc_endpoint_captured(
        pose: Pose,
        waypoint: Waypoint,
        *,
        reverse: bool,
    ) -> bool:
        """Whether a non-final arc sample was safely crossed in its corridor."""
        tolerance_cm = waypoint.path_capture_tolerance_cm
        if (tolerance_cm is None or tolerance_cm <= 0.0 or waypoint.is_final
                or not waypoint.curvature
                or waypoint.target_heading_deg is None):
            return False

        motion_curvature = (-waypoint.curvature if reverse
                            else waypoint.curvature)
        end_motion_heading = geo.wrap180(
            waypoint.target_heading_deg + (180.0 if reverse else 0.0))
        heading_rad = math.radians(end_motion_heading)

        # Positive means the current position has crossed the endpoint tangent
        # in the segment's intended physical motion direction.
        progress_mm = (
            (pose.x_mm - waypoint.x_mm) * math.cos(heading_rad)
            + (pose.y_mm - waypoint.y_mm) * math.sin(heading_rad)
        )
        if progress_mm < 0.0:
            return False

        center_x = waypoint.x_mm - math.sin(heading_rad) / motion_curvature
        center_y = waypoint.y_mm + math.cos(heading_rad) / motion_curvature
        radius_mm = 1.0 / abs(motion_curvature)
        radial_error_mm = abs(
            math.hypot(pose.x_mm - center_x, pose.y_mm - center_y) - radius_mm)
        return radial_error_mm <= tolerance_cm * 10.0

    @staticmethod
    def _arc_endpoint_missed(
        pose: Pose,
        waypoint: Waypoint,
        *,
        reverse: bool,
    ) -> bool:
        """Whether the endpoint tangent was crossed outside its path corridor."""
        tolerance_cm = waypoint.path_capture_tolerance_cm
        if (tolerance_cm is None or tolerance_cm <= 0.0 or waypoint.is_final
                or not waypoint.curvature
                or waypoint.target_heading_deg is None):
            return False
        motion_curvature = (-waypoint.curvature if reverse
                            else waypoint.curvature)
        end_motion_heading = geo.wrap180(
            waypoint.target_heading_deg + (180.0 if reverse else 0.0))
        heading_rad = math.radians(end_motion_heading)
        progress_mm = (
            (pose.x_mm - waypoint.x_mm) * math.cos(heading_rad)
            + (pose.y_mm - waypoint.y_mm) * math.sin(heading_rad)
        )
        if progress_mm < 0.0:
            return False
        center_x = waypoint.x_mm - math.sin(heading_rad) / motion_curvature
        center_y = waypoint.y_mm + math.cos(heading_rad) / motion_curvature
        radius_mm = 1.0 / abs(motion_curvature)
        radial_error_mm = abs(
            math.hypot(pose.x_mm - center_x, pose.y_mm - center_y) - radius_mm)
        return radial_error_mm > tolerance_cm * 10.0

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
        *,
        preserve_observed_speed: bool = False,
    ) -> ControlCommand:
        """안전 정지 명령(zero) 을 만들고 미분 상태를 초기화한다."""
        last_obs = self._last_obs
        observed_speed = self._observed_speed_mm_s
        telemetry = dict(self._telemetry)
        telemetry.update({
            "control_override_reason": reason,
            "steering_command_final": 0.0,
            "throttle_command_final": 0.0,
            "waypoint_arrived": False,
        })
        self.reset()
        if preserve_observed_speed:
            # Spatial stop 뒤에도 차량은 관성으로 움직일 수 있다. 다음 fresh
            # observation에서 0으로 확인되기 전까지 직전 실측 속도를 버리면
            # 한 tick 재출발하는 stop/go race가 생긴다.
            self._last_obs = last_obs
            self._observed_speed_mm_s = observed_speed
        return ControlCommand(
            throttle=0.0,
            steering=0.0,
            mode=ControlMode.HOLD,
            arrived=False,
            distance_error_cm=distance_cm,
            heading_error_deg=err_deg,
            target_bearing_deg=bearing,
            reason=reason,
            telemetry=telemetry,
        )
