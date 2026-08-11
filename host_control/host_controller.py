"""HostController — authority + mission + producer + transport 를 묶는 최상위 tick 루프.

매 주기(기본 100ms) `tick()` 을 호출하면, 현재 authority/미션 상태에 따라
정확히 하나의 producer 출력을 골라 DIRECT_CONTROL 로 전송한다. zero 상태도 명시적으로 전송한다.

안전 설계 요약
--------------
- DISARMED/FAULTED → 항상 zero.
- MANUAL → 사람 입력만. auto 무시.
- AUTO_HOST → 자율만. manual 무시.
- camera stale/invalid(AUTO_HOST) → **FAULTED latch** → 자동 재출발 불가(explicit re-arm 필요).
- 어떤 경우에도 sender.control_seq 는 monotonic.

이 모듈은 backend/network/camera/Django 를 import 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from controller.config import ControllerConfig
from controller.models import ControlCommand, ControlMode, Pose, Waypoint

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

        cmd = self.auto_producer.compute(
            pose, target, allow_drive=True, now=now
        )

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

        # 미션 진행 반영(도착→다음 target, ALIGN→REPLAN, final→DONE)
        self.mission.notify_result(cmd)
        return self._finish(cmd, zero=cmd.is_stopped)

    # ------------------------------------------------------------ 편의 API
    def arm_auto(self) -> None:
        self.auto_producer.reset()
        self.approach_guard.reset()
        self.final_pose_guard.reset()
        self._had_heading = False
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
        self.pose_source.clear()
        return payload

    # ------------------------------------------------------------ 내부
    def _finish(self, cmd: ControlCommand, *, zero: bool) -> TickResult:
        payload = self.sender.send_zero() if zero else self.sender.send_command(cmd)
        return TickResult(
            authority=self.authority.state,
            mission_status=self.mission.status,
            command=cmd,
            payload=payload,
        )
