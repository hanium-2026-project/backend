"""Host-owned waypoint mission — B안 전용.

프롬프트 3장의 요구를 구현한다:
- Route/Waypoint 목록과 현재 target 을 **host 내부에만** 유지한다.
- ESP32 로 WAYPOINT/GO wire command 를 **보내지 않는다** (이 모듈은 wire 를 전혀 다루지 않는다).
- 도착 판정은 controller 출력(camera pose 기반)의 arrived/mode 로 host 가 수행한다.
- 도착 → 다음 target 전환, final 도착 → DONE/PARKED.
- Ackermann 특성상 제자리 정렬 불가 → 위치 도달했으나 heading 불일치면 REPLAN_REQUIRED.

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
    REPLAN_REQUIRED = "REPLAN_REQUIRED"  # 위치 도달, heading 불일치 → 상위가 재접근 경로 필요
    DONE = "DONE"                 # 마지막 waypoint 도착 완료(주차 후보)
    PARKED = "PARKED"             # DONE 을 주차 완료로 확정(상위가 확정)


class HostWaypointMission:
    """host 내부 waypoint 진행 관리기.

    사용 흐름(매 tick):
        target = mission.current_target()
        cmd = auto_producer.compute(pose, target, now=...)
        mission.notify_result(cmd)   # arrived/align 에 따라 상태/index 갱신
    """

    def __init__(self, waypoints: Optional[Sequence[Waypoint]] = None) -> None:
        self._waypoints: List[Waypoint] = list(waypoints or [])
        self._index: int = 0
        self._status: MissionStatus = (
            MissionStatus.RUNNING if self._waypoints else MissionStatus.EMPTY
        )

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

    def current_target(self) -> Optional[Waypoint]:
        if self._status is not MissionStatus.RUNNING:
            return None
        if 0 <= self._index < len(self._waypoints):
            return self._waypoints[self._index]
        return None

    def _is_final_index(self) -> bool:
        wp = self._waypoints[self._index]
        # is_final 플래그가 명시됐으면 우선, 아니면 마지막 index 로 판정
        return bool(wp.is_final) or (self._index == len(self._waypoints) - 1)

    # --------------------------------------------------------------- 갱신
    def load(self, waypoints: Sequence[Waypoint]) -> None:
        """새 route 로드(재접근/재계획 시). index 0 부터 RUNNING."""
        self._waypoints = list(waypoints)
        self._index = 0
        self._status = (
            MissionStatus.RUNNING if self._waypoints else MissionStatus.EMPTY
        )

    def notify_result(self, cmd: ControlCommand) -> MissionStatus:
        """controller 출력을 반영해 mission 상태를 갱신하고 새 상태를 반환.

        - ALIGN(위치 도달, heading 불일치) → REPLAN_REQUIRED (전진 전용이라 제자리 정렬 안 함)
        - ARRIVED(위치[+heading] 도달) → 마지막이면 DONE, 아니면 다음 target
        - 그 외 → 변화 없음(RUNNING 유지)
        """
        if self._status is not MissionStatus.RUNNING:
            return self._status

        if cmd.mode is ControlMode.ALIGN:
            # 무리한 제자리 회전 금지. 상위 mission layer 가 재접근 경로를 만들도록 요청.
            self._status = MissionStatus.REPLAN_REQUIRED
            return self._status

        if cmd.arrived:  # ARRIVED
            if self._is_final_index():
                self._status = MissionStatus.DONE
            else:
                self._index += 1
            return self._status

        return self._status

    def confirm_parked(self) -> None:
        """상위 layer 가 DONE 을 주차 완료로 확정."""
        if self._status is MissionStatus.DONE:
            self._status = MissionStatus.PARKED

    def reset(self) -> None:
        self._index = 0
        self._status = (
            MissionStatus.RUNNING if self._waypoints else MissionStatus.EMPTY
        )
