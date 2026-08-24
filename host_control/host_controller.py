"""HostController — authority + mission + producer + transport 를 묶는 최상위 tick 루프.

매 주기(기본 100ms) `tick()` 을 호출하면, 현재 authority/미션 상태에 따라
정확히 하나의 producer 출력을 골라 DIRECT_CONTROL 로 전송한다. zero 상태도 명시적으로 전송한다.

안전 설계 요약
--------------
- DISARMED/FAULTED → 항상 zero.
- MANUAL → 사람 입력만. auto 무시.
- AUTO_HOST → 자율만. manual 무시.
- camera stale/invalid(AUTO_HOST) → **FAULTED latch** → 자동 재출발 불가(explicit re-arm 필요).
- 실행 중인 rear reverse의 일시 pose/heading 손실은 즉시 zero 후
  bounded reacquisition을 거쳐 resume 또는 REPLAN_REQUIRED로 보낸다.
- 어떤 경우에도 sender.control_seq 는 monotonic.

이 모듈은 backend/network/camera/Django 를 import 하지 않는다.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Optional

from controller.config import ControllerConfig
from controller.models import (ControlCommand, ControlMode, MotionDirection,
                               Pose, Waypoint)

from .authority import Authority, ControlAuthority
from .approach_guard import ApproachProgressGuard
from .final_pose_guard import FinalPoseGuard
from .direct_control import DirectControlSender, TransportTiming
from .mission import HostWaypointMission, MissionStatus
from .producers import (
    AutoControlProducer,
    ManualControlProducer,
    ManualInput,
    _zero_command,
)
from .pose_source import CameraPoseSource

# camera 관련 stale/invalid 사유(→ AUTO_HOST 에서 latched fault 대상)
_STALE_REASONS = frozenset({"POSE_STALE", "POSE_INVALID", "NO_HEADING"})

# ── backend 결선 시 수정 (2026-08-11) ────────────────────────────────────────
# NO_HEADING 은 "아직 못 구한 상태"와 "구했다가 잃은 상태"가 섞여 있다. 우리
# heading 추정기는 전방 쿠션이 잡히거나 차가 움직여야 방향을 알 수 있어서,
# 무장 직후 몇 프레임은 항상 heading 이 없다. 여기서 latch 하면 출발도 못 해보고
# 죽는다. heading 을 한 번이라도 확보한 뒤 잃은 경우만 fault 로 본다.
_WARMUP_REASONS = frozenset({"NO_HEADING"})

# Rear reverse에서는 이 상태들이 모두 "현재 차체 자세를 신뢰할 수
# 없음"이다. 출력은 즉시 zero이지만, 일시적 camera gap까지 전역
# FAULTED로 잠그면 fresh frame이 돌아와도 recovery callback이 없는
# silent RUNNING+zero 교착이 생긴다.
_REVERSE_OBSERVATION_HOLD_REASONS = frozenset({
    "REVERSE_HEADING_UNSAFE", "POSE_STALE", "NO_POSE", "NO_HEADING",
})


class ReverseObservationState(str, Enum):
    """Semantic rear-reverse observation state (also exposed to recorder)."""

    IDLE = "IDLE"
    WAIT_PRIMARY = "REVERSE_WAIT_PRIMARY_HEADING"
    START_ANCHOR = "REVERSE_START_TRAJECTORY_ANCHOR"
    TRACK_PRIMARY = "REVERSE_TRACK_PRIMARY"
    TRACK_TRAJECTORY = "REVERSE_TRACK_TRAJECTORY_FALLBACK"
    OBSERVATION_LOST = "REVERSE_OBSERVATION_LOST"


@dataclass(frozen=True)
class TickResult:
    authority: Authority
    mission_status: MissionStatus
    command: ControlCommand
    payload: Dict


class HostController:
    def __init__(
        self,
        *,
        config: Optional[ControllerConfig] = None,
        mission: Optional[HostWaypointMission] = None,
        sender: Optional[DirectControlSender] = None,
        timing: Optional[TransportTiming] = None,
        fault_on_stale: bool = True,
    ) -> None:
        self.config = config or ControllerConfig()
        self.authority = ControlAuthority()
        self.pose_source = CameraPoseSource()
        self.mission = mission or HostWaypointMission()
        self.approach_guard = ApproachProgressGuard(self.config)
        self.final_pose_guard = FinalPoseGuard(self.config.final_confirm_observations)
        self.manual_producer = ManualControlProducer(self.config)
        self.auto_producer = AutoControlProducer(self.config)
        self.sender = sender or DirectControlSender(timing=timing)
        self.fault_on_stale = fault_on_stale
        self._had_heading = False        # heading 을 한 번이라도 확보했는가
        self._reverse_observation_wait_started: float | None = None
        window = max(3, int(self.config.reverse_trajectory_window))
        self._recent_observations: deque[Pose] = deque(maxlen=window)
        self._reverse_observations: deque[Pose] = deque(maxlen=window)
        self._reverse_observation_state = ReverseObservationState.IDLE
        self._reverse_motion_started = False
        self._reverse_start_anchor: Pose | None = None
        self._reverse_start_anchor_route_id: int | None = None
        self._reverse_bootstrap_origin: Pose | None = None
        self._reverse_bootstrap_started: float | None = None
        self._last_trusted_reverse_heading: float | None = None

    # ------------------------------------------------------------ 관측 입력
    def observe(self, pose: Pose) -> None:
        """새 카메라 관측 pose(관측 timestamp 포함) 기록."""
        self.pose_source.observe_pose(pose)

    # ------------------------------------------------------------ 메인 tick
    def tick(
        self,
        now: float,
        *,
        observation: Optional[Pose] = None,
        manual_input: Optional[ManualInput] = None,
    ) -> TickResult:
        if observation is not None:
            self.pose_source.observe_pose(observation)

        state = self.authority.state

        # --- 비주행 상태: 항상 zero -------------------------------------
        if state in (Authority.DISARMED, Authority.FAULTED):
            reason = self.authority.fault_reason or state.value
            cmd = _zero_command(reason)
            return self._finish(cmd, zero=True)

        # --- MANUAL: 사람 입력만 ----------------------------------------
        if state is Authority.MANUAL:
            cmd = self.manual_producer.compute(manual_input)
            return self._finish(cmd, zero=cmd.is_stopped)

        # --- AUTO_HOST: 자율만 ------------------------------------------
        target: Optional[Waypoint] = self.mission.current_target()
        # 추종할 target 없음(완료/재계획/미로드) → zero
        if target is None or not self.mission.is_active:
            cmd = _zero_command(f"MISSION_{self.mission.status.value}")
            return self._finish(cmd, zero=True)

        pose = self.pose_source.latest()
        self._record_distinct_observation(pose)
        if pose is not None and pose.has_heading:
            self._had_heading = True

        # APPROACH 2단계 gate. stale/invalid/no-heading은 기존 controller fail-safe가 우선.
        if (
            pose is not None
            and pose.valid
            and pose.has_heading
            and (now - pose.timestamp) <= self.config.max_pose_age_s
        ):
            progress = self.approach_guard.evaluate(
                pose, target, previous_target=self.mission.previous_target()
            )
            if progress.missed:
                reason = progress.event.value
                self.mission.request_replan(reason)
                self.auto_producer.reset()
                self.final_pose_guard.reset()
                cmd = _zero_command(reason)
                return self._finish(cmd, zero=True)

        control_pose = self._trusted_reverse_pose(pose, target, now=now)
        cmd = self.auto_producer.compute(
            control_pose, target, allow_drive=True, now=now
        )

        # Rear reverse observation contract:
        # unsafe/stale -> immediate zero -> short fresh-heading reacquisition
        # -> resume, or bounded timeout -> parking setup/replan.
        #
        # POSE_STALE must join the same hold once reverse is active.  Sending it
        # through the generic latched-fault branch leaves mission RUNNING and
        # prevents the status callback from scheduling recovery.
        reverse_observation_hold = bool(
            target.motion_direction is MotionDirection.REVERSE
            and (target.phase or "").upper()
                in self.config.heading_guard_reverse_phases
            and cmd.reason in _REVERSE_OBSERVATION_HOLD_REASONS
        )
        if reverse_observation_hold:
            if self._reverse_observation_wait_started is None:
                self._reverse_observation_wait_started = now
            elif (now - self._reverse_observation_wait_started
                  >= self.config.reverse_heading_wait_timeout_s):
                self.mission.request_replan("REVERSE_HEADING_TIMEOUT")
                self.auto_producer.reset()
                self._reverse_observation_wait_started = None
                cmd = _zero_command("REVERSE_HEADING_TIMEOUT")
                return self._finish(cmd, zero=True)
            return self._finish(cmd, zero=True)
        else:
            self._reverse_observation_wait_started = None

        # A direction flip must not reuse the observation that produced the stop.
        # Clear it so the first non-zero command in the new direction requires a
        # distinct camera frame, not merely the next scheduler tick.
        if cmd.reason == "DIRECTION_CHANGE_STOP":
            self.pose_source.clear()
            return self._finish(cmd, zero=True)

        # 아직 heading 을 한 번도 못 잡았다면 초기 구간이다 — 정지만 하고 기다린다.
        if cmd.reason in _WARMUP_REASONS and not self._had_heading:
            return self._finish(_zero_command("WARMUP_NO_HEADING"), zero=True)

        # camera stale/invalid → latched fault (자동 재출발 차단)
        if pose is not None and cmd.reason in _STALE_REASONS and self.fault_on_stale:
            self.authority.fault(cmd.reason)
            cmd = _zero_command(cmd.reason)
            return self._finish(cmd, zero=True)

        # FINAL은 한 번의 frame으로 DONE 처리하지 않는다. 첫 ARRIVED부터 motor는
        # zero이고 서로 다른 fresh camera observation이 연속 N회 만족해야 확정한다.
        final_progress = self.final_pose_guard.evaluate(pose, target, cmd)
        if (
            (target.phase or "").upper() == "FINAL"
            and cmd.arrived
            and not final_progress.confirmed
        ):
            cmd = ControlCommand(
                throttle=0.0,
                steering=0.0,
                mode=ControlMode.HOLD,
                arrived=False,
                distance_error_cm=cmd.distance_error_cm,
                heading_error_deg=cmd.heading_error_deg,
                target_bearing_deg=cmd.target_bearing_deg,
                reason=f"FINAL_CONFIRMING_{final_progress.count}_OF_{final_progress.required}",
                logical_steering=0.0,
            )
            return self._finish(cmd, zero=True)

        # A same-route ALIGN terminal may provide a bounded reverse START
        # anchor.  Capture it before notify_result advances to ENTRY.
        if (cmd.arrived
                and target.motion_direction is MotionDirection.FORWARD
                and (target.phase or "").upper() == "ALIGN"):
            self._capture_reverse_start_anchor(pose, target, now=now)

        if (cmd.throttle < 0.0
                and target.motion_direction is MotionDirection.REVERSE):
            if not self._reverse_motion_started:
                self._reverse_motion_started = True
                self._reverse_observations.clear()
                if pose is not None:
                    self._append_distinct(self._reverse_observations, pose)
                    self._reverse_bootstrap_origin = pose
                self._reverse_bootstrap_started = now

        # 미션 진행 반영(도착→다음 target, ALIGN→REPLAN, final→DONE)
        self.mission.notify_result(cmd)
        return self._finish(cmd, zero=cmd.is_stopped)

    # ------------------------------------------------------------ 편의 API
    def arm_auto(self) -> None:
        self.auto_producer.reset()
        self.approach_guard.reset()
        self.final_pose_guard.reset()
        self._had_heading = False
        self._reset_reverse_observation_contract(clear_history=True)
        self.authority.arm_auto()

    def arm_manual(self) -> None:
        self.authority.arm_manual()

    def disarm(self) -> None:
        self.authority.disarm()

    def stop(self) -> None:
        """비상 정지(latched fault)."""
        self.authority.stop()

    def fault(self, reason: str) -> None:
        self.authority.fault(reason)

    def re_arm_auto(self) -> None:
        self.auto_producer.reset()
        self.approach_guard.reset()
        self.final_pose_guard.reset()
        self._had_heading = False
        self._reset_reverse_observation_contract(clear_history=True)
        self.authority.re_arm_auto()

    def prepare_route_switch(self) -> Dict:
        """재계획/Recovery route 전환 직전 즉시 zero + fresh pose 강제.

        scheduler 를 멈추지는 않는다. 대신 마지막 카메라 pose 를 폐기하므로 새 route 가
        RUNNING 으로 바뀌어도 새 관측이 들어오기 전까지 tick() 은 NO_POSE zero를 보낸다.
        이 순서로 REPLAN_REQUIRED → Recovery 전환 중 예전 pose/제어값 재사용을 막는다.
        """
        payload = self.sender.send_zero()
        self.auto_producer.reset()
        self.approach_guard.reset()
        self.final_pose_guard.reset()
        self._had_heading = False
        self._reset_reverse_observation_contract(clear_history=True)
        self.pose_source.clear()
        return payload

    @property
    def reverse_observation_state(self) -> str:
        return self._reverse_observation_state.value

    @staticmethod
    def _append_distinct(samples: deque[Pose], pose: Pose) -> None:
        if samples and pose.timestamp <= samples[-1].timestamp:
            return
        samples.append(pose)

    def _record_distinct_observation(self, pose: Pose | None) -> None:
        if pose is None or not pose.valid:
            return
        before = self._recent_observations[-1].timestamp if self._recent_observations else None
        self._append_distinct(self._recent_observations, pose)
        if (self._reverse_motion_started
                and (before is None or pose.timestamp > before)):
            self._append_distinct(self._reverse_observations, pose)

    @staticmethod
    def _heading_delta(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _quality_trajectory_heading(
        self, samples: deque[Pose], *, reverse: bool,
    ) -> float | None:
        cfg = self.config
        if len(samples) < int(cfg.reverse_trajectory_min_observations):
            return None
        points = list(samples)[-max(3, int(cfg.reverse_trajectory_window)):]
        span = points[-1].timestamp - points[0].timestamp
        if span <= 0.0 or span > cfg.reverse_trajectory_max_span_s:
            return None
        dx = points[-1].x_mm - points[0].x_mm
        dy = points[-1].y_mm - points[0].y_mm
        net = math.hypot(dx, dy)
        if net < cfg.reverse_trajectory_min_displacement_mm:
            return None
        path = sum(
            math.hypot(b.x_mm - a.x_mm, b.y_mm - a.y_mm)
            for a, b in zip(points, points[1:])
        )
        if path <= 0.0 or net / path < cfg.reverse_trajectory_min_linearity:
            return None
        heading = math.degrees(math.atan2(dy, dx)) % 360.0
        return (heading + (180.0 if reverse else 0.0)) % 360.0

    def _capture_reverse_start_anchor(
        self, pose: Pose | None, target: Waypoint, *, now: float,
    ) -> None:
        if (pose is None or not pose.valid or not pose.has_heading
                or pose.heading_source != "TRAJECTORY"
                or now - pose.timestamp > self.config.max_pose_age_s):
            return
        heading = self._quality_trajectory_heading(
            self._recent_observations, reverse=False)
        if (heading is None
                or self._heading_delta(heading, float(pose.heading_deg))
                > self.config.reverse_start_anchor_max_heading_delta_deg):
            return
        self._reverse_start_anchor = replace(
            pose, heading_deg=heading,
            heading_source="REVERSE_START_TRAJECTORY_ANCHOR")
        self._reverse_start_anchor_route_id = target.route_id

    def _trusted_reverse_pose(
        self, pose: Pose | None, target: Waypoint, *, now: float,
    ) -> Pose | None:
        guarded = bool(
            target.motion_direction is MotionDirection.REVERSE
            and (target.phase or "").upper()
                in self.config.heading_guard_reverse_phases
        )
        if not guarded:
            self._reverse_observation_state = ReverseObservationState.IDLE
            self._reverse_motion_started = False
            self._reverse_observations.clear()
            self._last_trusted_reverse_heading = None
            return pose
        if (pose is None or not pose.valid
                or now - pose.timestamp > self.config.max_pose_age_s):
            self._reverse_observation_state = ReverseObservationState.OBSERVATION_LOST
            return pose

        source = str(pose.heading_source or "").upper()
        if source == "FRONT_CUSHION" and pose.has_heading:
            self._reverse_observation_state = ReverseObservationState.TRACK_PRIMARY
            self._last_trusted_reverse_heading = float(pose.heading_deg)
            return pose

        if self._reverse_motion_started:
            heading = self._quality_trajectory_heading(
                self._reverse_observations, reverse=True)
            if (heading is not None
                    and (self._last_trusted_reverse_heading is None
                         or self._heading_delta(
                             heading, self._last_trusted_reverse_heading)
                         <= self.config.reverse_trajectory_max_heading_delta_deg)):
                self._reverse_observation_state = ReverseObservationState.TRACK_TRAJECTORY
                self._last_trusted_reverse_heading = heading
                return replace(
                    pose, heading_deg=heading,
                    heading_source="REVERSE_TRAJECTORY")

            # The first command cannot instantly create a 30 mm reverse
            # trajectory.  Reuse only the validated ALIGN body heading, only
            # while fresh raw motion observations still say TRAJECTORY, and
            # only inside the bounded distance/time needed to form that track.
            origin = self._reverse_bootstrap_origin
            anchor = self._reverse_start_anchor
            if (origin is not None and anchor is not None
                    and source == "TRAJECTORY"
                    and self._reverse_bootstrap_started is not None
                    and now - self._reverse_bootstrap_started
                        <= self.config.reverse_start_bootstrap_max_age_s
                    and math.hypot(
                        pose.x_mm - origin.x_mm, pose.y_mm - origin.y_mm)
                        <= self.config.reverse_start_bootstrap_max_distance_mm):
                self._reverse_observation_state = ReverseObservationState.START_ANCHOR
                return replace(
                    pose, heading_deg=anchor.heading_deg,
                    heading_source="REVERSE_START_TRAJECTORY_ANCHOR")

        anchor = self._reverse_start_anchor
        same_route = bool(
            anchor is not None
            and self._reverse_start_anchor_route_id is not None
            and self._reverse_start_anchor_route_id == target.route_id)
        if (not self._reverse_motion_started and same_route
                and source == "TRAJECTORY"
                and now - anchor.timestamp <= self.config.reverse_start_anchor_max_age_s
                and math.hypot(
                    pose.x_mm - anchor.x_mm, pose.y_mm - anchor.y_mm)
                    <= self.config.reverse_start_anchor_max_distance_mm):
            self._reverse_observation_state = ReverseObservationState.START_ANCHOR
            self._last_trusted_reverse_heading = float(anchor.heading_deg)
            return replace(
                pose, heading_deg=anchor.heading_deg,
                heading_source="REVERSE_START_TRAJECTORY_ANCHOR")

        self._reverse_observation_state = (
            ReverseObservationState.OBSERVATION_LOST
            if self._reverse_motion_started else ReverseObservationState.WAIT_PRIMARY)
        return pose

    def _reset_reverse_observation_contract(self, *, clear_history: bool) -> None:
        self._reverse_observation_wait_started = None
        self._reverse_observation_state = ReverseObservationState.IDLE
        self._reverse_motion_started = False
        self._reverse_observations.clear()
        self._reverse_start_anchor = None
        self._reverse_start_anchor_route_id = None
        self._reverse_bootstrap_origin = None
        self._reverse_bootstrap_started = None
        self._last_trusted_reverse_heading = None
        if clear_history:
            self._recent_observations.clear()

    # ------------------------------------------------------------ 내부
    def _finish(self, cmd: ControlCommand, *, zero: bool) -> TickResult:
        payload = self.sender.send_zero() if zero else self.sender.send_command(cmd)
        return TickResult(
            authority=self.authority.state,
            mission_status=self.mission.status,
            command=cmd,
            payload=payload,
        )
