"""Host-owned waypoint mission — B안 전용.

프롬프트 3장의 요구를 구현한다:
- Route/Waypoint 목록과 현재 target 을 **host 내부에만** 유지한다.
- ESP32 로 WAYPOINT/GO wire command 를 **보내지 않는다** (이 모듈은 wire 를 전혀 다루지 않는다).
- 도착 판정은 controller 출력(camera pose 기반)의 arrived/mode 로 host 가 수행한다.
- 도착 → 다음 target 전환, final 도착 → DONE/PARKED.
- Ackermann 특성상 제자리 정렬 불가 → 위치 도달했으나 heading 불일치면 REPLAN_REQUIRED.
- REPLAN_REQUIRED 에서 recovery waypoint 를 삽입하고, 완료 후 실패했던 원래 target 으로 복귀한다.

기존 WAYPOINT_AUTO 용 MissionOrchestrator 계약과는 **완전히 분리**된다(향후 A안용으로 보존).
이 모듈은 backend/network 를 import 하지 않는다.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence

from controller.models import ControlCommand, ControlMode, Waypoint


class MissionStatus(str, Enum):
    EMPTY = "EMPTY"                 # waypoint 없음
    RUNNING = "RUNNING"            # 현재 target 추종 중
    REPLAN_REQUIRED = "REPLAN_REQUIRED"  # 현재 pose 로 목표 pose 달성 불가 → recovery 경로 필요
    RECOVERY_FAILED = "RECOVERY_FAILED"  # recovery 재시도 한도 초과 → 상위에서 중단/전역 재계획
    DONE = "DONE"                   # 마지막 waypoint 도착 완료(주차 후보)
    PARKED = "PARKED"               # DONE 을 주차 완료로 확정(상위가 확정)


class HostWaypointMission:
    """host 내부 waypoint 진행 관리기.

    일반 흐름(매 tick):
        target = mission.current_target()
        cmd = auto_producer.compute(pose, target, now=...)
        mission.notify_result(cmd)

    recovery 흐름:
        RUNNING
          → ALIGN/재접근 필요
          → REPLAN_REQUIRED (현재 target 이후 route 를 내부에 보존)
          → 상위가 recovery waypoint 생성
          → load_recovery(recovery_waypoints)
          → RUNNING (recovery → 실패했던 기존 target → 기존 남은 route)

    recovery 자체의 기하/경로 생성은 이 클래스가 하지 않는다. 이 클래스는
    "어떤 경로를 실행 중인지"와 "원래 route 로 언제 복귀하는지"만 관리한다.
    """

    def __init__(
        self,
        waypoints: Optional[Sequence[Waypoint]] = None,
        *,
        max_recovery_attempts: int = 3,
    ) -> None:
        if max_recovery_attempts < 1:
            raise ValueError("max_recovery_attempts must be >= 1")
        self._waypoints: List[Waypoint] = list(waypoints or [])
        self._index: int = 0
        self._status: MissionStatus = (
            MissionStatus.RUNNING if self._waypoints else MissionStatus.EMPTY
        )
        self._max_recovery_attempts = int(max_recovery_attempts)
        self._recovery_attempts = 0
        self._recovery_prefix_len = 0
        self._resume_waypoints: List[Waypoint] = []
        self._replan_reason: Optional[str] = None

    # --------------------------------------------------------------- 조회
    @property
    def status(self) -> MissionStatus:
        return self._status

    @property
    def index(self) -> int:
        return self._index

    @property
    def total(self) -> int:
        return len(self._waypoints)

    @property
    def is_active(self) -> bool:
        """아직 추종할 target 이 있는가."""
        return self._status is MissionStatus.RUNNING

    @property
    def is_recovering(self) -> bool:
        """현재 recovery prefix 를 실행 중인가."""
        return (
            self._status is MissionStatus.RUNNING
            and self._recovery_prefix_len > 0
            and self._index < self._recovery_prefix_len
        )

    @property
    def recovery_attempts(self) -> int:
        return self._recovery_attempts

    @property
    def max_recovery_attempts(self) -> int:
        return self._max_recovery_attempts

    @property
    def replan_reason(self) -> Optional[str]:
        return self._replan_reason

    @property
    def current_phase(self) -> Optional[str]:
        target = self.current_target()
        return None if target is None else target.phase

    @property
    def parking_active(self) -> bool:
        phase = (self.current_phase or "").upper()
        return phase in {"APPROACH", "ALIGN", "ENTRY", "FINAL", "PARKING"}

    def previous_target(self) -> Optional[Waypoint]:
        """현재 target으로 들어오는 segment의 직전 waypoint."""
        if self._status is not MissionStatus.RUNNING or self._index <= 0:
            return None
        return self._waypoints[self._index - 1]

    def current_target(self) -> Optional[Waypoint]:
        if self._status is not MissionStatus.RUNNING:
            return None
        if 0 <= self._index < len(self._waypoints):
            return self._waypoints[self._index]
        return None

    @property
    def current_is_terminal(self) -> bool:
        """Whether the active target is the route's actual completion boundary."""
        return (self._status is MissionStatus.RUNNING
                and 0 <= self._index < len(self._waypoints)
                and self._is_final_index())

    def _is_final_index(self) -> bool:
        wp = self._waypoints[self._index]
        # is_final 플래그가 명시됐으면 우선, 아니면 마지막 index 로 판정
        return bool(wp.is_final) or (self._index == len(self._waypoints) - 1)

    # --------------------------------------------------------------- 갱신
    def load(self, waypoints: Sequence[Waypoint]) -> None:
        """새 primary route 로 교체한다.

        슬롯 재배정/전역 재계획처럼 기존 mission 자체를 교체할 때 사용한다.
        recovery 카운터와 resume snapshot 도 새 mission 기준으로 초기화한다.
        """
        self._waypoints = list(waypoints)
        self._index = 0
        self._status = (
            MissionStatus.RUNNING if self._waypoints else MissionStatus.EMPTY
        )
        self._recovery_attempts = 0
        self._recovery_prefix_len = 0
        self._resume_waypoints = []
        self._replan_reason = None

    def load_recovery(self, recovery_waypoints: Sequence[Waypoint]) -> MissionStatus:
        """REPLAN_REQUIRED 상태에서 local recovery 경로를 앞에 삽입한다.

        recovery 경로를 완료하면, REPLAN_REQUIRED 를 발생시켰던 기존 target 부터
        자동으로 다시 추종한다. 따라서 상위 planner 는 "현재 pose → 재접근 pose"만
        생성하면 되고 기존 APPROACH/ENTRY/FINAL route 를 복제할 필요가 없다.

        안전 규칙:
        - REPLAN_REQUIRED 상태에서만 호출 가능.
        - 빈 recovery route 금지.
        - recovery waypoint 자체에 is_final=True 금지(원래 route 복귀 전에 DONE 방지).
        - max_recovery_attempts 초과 시 RECOVERY_FAILED 로 latch.
        """
        if self._status is not MissionStatus.REPLAN_REQUIRED:
            raise RuntimeError(
                f"recovery can only be loaded from REPLAN_REQUIRED, got {self._status.value}")

        recovery = list(recovery_waypoints)
        if not recovery:
            raise ValueError("recovery_waypoints must not be empty")
        if any(wp.is_final for wp in recovery):
            raise ValueError("recovery waypoint must not set is_final=True")
        if not self._resume_waypoints:
            raise RuntimeError("no saved route to resume after recovery")

        if self._recovery_attempts >= self._max_recovery_attempts:
            self._status = MissionStatus.RECOVERY_FAILED
            self._waypoints = []
            self._index = 0
            self._recovery_prefix_len = 0
            return self._status

        self._recovery_attempts += 1
        self._waypoints = recovery + list(self._resume_waypoints)
        self._recovery_prefix_len = len(recovery)
        self._index = 0
        self._status = MissionStatus.RUNNING
        self._replan_reason = None
        return self._status

    def request_replan(self, reason: str) -> MissionStatus:
        """현재 target을 보존하고 즉시 REPLAN_REQUIRED로 전환."""
        if self._status is not MissionStatus.RUNNING:
            return self._status
        was_recovering = self.is_recovering
        if not was_recovering:
            self._resume_waypoints = list(self._waypoints[self._index:])
        self._recovery_prefix_len = 0
        self._replan_reason = str(reason)
        self._status = MissionStatus.REPLAN_REQUIRED
        return self._status

    def notify_result(self, cmd: ControlCommand) -> MissionStatus:
        """controller 출력을 반영해 mission 상태를 갱신하고 새 상태를 반환.

        - ALIGN → REPLAN_REQUIRED. 현재 target 부터 남은 route 를 snapshot 으로 보존.
        - ARRIVED → 마지막이면 DONE, 아니면 다음 target.
        - recovery prefix 완료 → 자동으로 snapshot 의 기존 target 으로 복귀.
        - 그 외 → 변화 없음(RUNNING 유지).
        """
        if self._status is not MissionStatus.RUNNING:
            return self._status

        if cmd.mode is ControlMode.ALIGN:
            return self.request_replan("HEADING_OUT_OF_TOLERANCE")

        if cmd.arrived:  # ARRIVED
            if self._is_final_index():
                self._status = MissionStatus.DONE
                self._recovery_prefix_len = 0
                self._resume_waypoints = []
            else:
                self._index += 1
                if self._recovery_prefix_len and self._index >= self._recovery_prefix_len:
                    # recovery 구간 종료. 다음 target 부터는 보존해 둔 원래 route.
                    self._recovery_prefix_len = 0
            return self._status

        return self._status

    def confirm_parked(self) -> None:
        """상위 layer 가 DONE 을 주차 완료로 확정."""
        if self._status is MissionStatus.DONE:
            self._status = MissionStatus.PARKED

    def reset(self) -> None:
        self._index = 0
        self._recovery_attempts = 0
        self._recovery_prefix_len = 0
        self._resume_waypoints = []
        self._replan_reason = None
        self._status = (
            MissionStatus.RUNNING if self._waypoints else MissionStatus.EMPTY
        )
