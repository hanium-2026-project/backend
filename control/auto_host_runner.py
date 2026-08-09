"""AUTO_HOST 결선 — 하드웨어팀 host_autonomous_control 패키지를 backend 에 붙인다.

`host_autonomous_control_FINAL` (2026-08-08 수신) 의 `production_patch/auto_host_runner.py`
템플릿을 실제 파이프라인에 맞게 구현한 것이다. 원본과의 차이:

- 미션 상태 변화를 `on_status_change` 로 상위에 통지한다. 이게 없으면 FINAL 도착이
  파이프라인까지 올라오지 않아 **슬롯 점유·대시보드 갱신이 끊긴다**.
- `confirm_parked()` 를 러너가 직접 부르지 않는다. 정지 재확인(§11)은 카메라를 보는
  파이프라인의 몫이라, DONE 을 알리기만 하고 확정은 위에서 한다.
- 재계획(REPLAN_REQUIRED)도 같은 경로로 올려 기존 재계획 로직을 재사용한다.

AUTO_HOST 동안 ESP32 로 WAYPOINT/GO 를 보내지 않는다. 목표 waypoint 는 host 내부
(`HostWaypointMission`)에만 있고, ESP32 는 DIRECT_CONTROL 만 실행한다.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from host_control import HostController, HostWaypointMission
from host_control.mission import MissionStatus
from integration.backend_adapter import VehicleServerDirectSender, waypoints_from_backend
from integration.control_scheduler import ControlScheduler
from integration.remote_direct_session import ModeHandshakeError, RemoteDirectSession

log = logging.getLogger(__name__)

__all__ = ["AutoHostRunner", "ModeHandshakeError", "MissionStatus"]


class AutoHostRunner:
    """차량 1대의 AUTO_HOST 주행 (제어 소유권은 여기 하나뿐)."""

    def __init__(self, server: Any, car_id: int, backend_waypoints: Sequence[Any],
                 *, period_s: float = 0.100) -> None:
        if not isinstance(car_id, int):
            raise TypeError("car_id 는 int (wire 의 'CAR_01' 은 서버 내부 표현)")
        self.car_id = car_id
        self._server = server
        self.mission = HostWaypointMission(waypoints_from_backend(backend_waypoints))
        self.host = HostController(
            mission=self.mission,
            sender=VehicleServerDirectSender(server, car_id),
        )
        self.session = RemoteDirectSession(self.host, server, car_id)
        self.scheduler = ControlScheduler(self.host, period_s=period_s,
                                          on_tick=self._on_tick)
        self._last_status = self.mission.status
        # (car_id, 이전 상태, 새 상태) — 파이프라인이 슬롯 점유·재계획을 처리한다
        self.on_status_change: Callable[[int, MissionStatus, MissionStatus], None] | None = None
        self.session.attach()

    # ─── 라이프사이클 ────────────────────────────────────────────────────────

    def start(self, *, wait_s: float = 2.0) -> None:
        """SET_MODE REMOTE_DIRECT → ACCEPTED 확인 → arm → 100ms 제어 루프.

        ACCEPTED 전에는 arm 하지 않는다. 차량이 아직 REMOTE_DIRECT 가 아닌데
        제어값을 보내면 무시되거나 엉뚱한 모드에서 실행될 수 있다.
        """
        self.session.begin_handshake()
        if not self.session.wait_accepted(wait_s):
            raise ModeHandshakeError(
                f"car {self.car_id}: REMOTE_DIRECT ACCEPTED 미도착 → FAULTED")
        self.session._enable_direct_stream()
        self.host.arm_auto()
        self.scheduler.start()
        log.info("car %d: AUTO_HOST 시작 (waypoint %d개, %.0fms 주기)",
                 self.car_id, self.mission.total, self.scheduler.period_s * 1000)

    def stop(self, *, disable_global_direct: bool = False) -> None:
        self.host.stop()
        self.scheduler.stop()
        stop_control = getattr(self._server, "stop_control", None)
        if stop_control is not None:
            stop_control(self.car_id)
        if disable_global_direct:
            self._server.direct_control_enabled = False

    def re_arm(self, *, wait_s: float = 2.0) -> None:
        """FAULTED 이후 명시적 재출발. 자동 복귀는 하지 않는다."""
        self.session.re_arm_auto(wait_s=wait_s)

    # ─── 파이프라인 연동 ─────────────────────────────────────────────────────

    def on_camera_pose(self, x_mm: float, y_mm: float,
                       heading_deg: float | None, obs_time: float) -> None:
        """새 카메라 프레임에서만 호출. 제어 계산은 스케줄러가 한다.

        obs_time 은 **관측 시각**이어야 한다. tick 시각을 넣으면 카메라가 멈춰도
        pose 가 계속 신선해 보여서 stale 판정이 무력화된다.
        """
        self.host.pose_source.observe(x_mm, y_mm, heading_deg, obs_time)

    def load_route(self, backend_waypoints: Sequence[Any]) -> None:
        """재계획된 경로로 교체한다."""
        self.mission.load(waypoints_from_backend(backend_waypoints))
        self.host.auto_producer.reset()
        self._last_status = self.mission.status

    def confirm_parked(self) -> None:
        self.mission.confirm_parked()
        self._last_status = self.mission.status

    @property
    def current_target(self):
        return self.mission.current_target()

    @property
    def status(self) -> MissionStatus:
        return self.mission.status

    @property
    def is_faulted(self) -> bool:
        return self.host.authority.is_faulted

    # ─── 내부 ────────────────────────────────────────────────────────────────

    def _on_tick(self, result) -> None:
        """제어 루프가 매 tick 부른다. 상태가 바뀐 순간만 위로 올린다."""
        status = result.mission_status
        if status is self._last_status:
            return
        prev, self._last_status = self._last_status, status
        log.info("car %d: mission %s → %s (authority=%s, %s)",
                 self.car_id, prev.value, status.value,
                 result.authority.value, result.command.reason or result.command.mode.value)
        if self.on_status_change is not None:
            self.on_status_change(self.car_id, prev, status)
