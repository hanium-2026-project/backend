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

from controller.config import ControllerConfig
from host_control import HostController, HostWaypointMission
from host_control.mission import MissionStatus
from integration.backend_adapter import VehicleServerDirectSender, waypoints_from_backend
from integration.control_scheduler import ControlScheduler
from integration.remote_direct_session import (
    ModeHandshakeError,
    NegotiationState,
    RemoteDirectSession,
)

log = logging.getLogger(__name__)

__all__ = ["AutoHostRunner", "ModeHandshakeError", "MissionStatus",
           "NegotiationState"]


class AutoHostRunner:
    """차량 1대의 AUTO_HOST 주행 (제어 소유권은 여기 하나뿐)."""

    def __init__(self, server: Any, car_id: int, backend_waypoints: Sequence[Any],
                 *, period_s: float = 0.100,
                 config: ControllerConfig | None = None) -> None:
        if not isinstance(car_id, int):
            raise TypeError("car_id 는 int (wire 의 'CAR_01' 은 서버 내부 표현)")
        self.car_id = car_id
        self._server = server
        self.mission = HostWaypointMission(waypoints_from_backend(backend_waypoints))
        self.config = config or ControllerConfig()
        self.host = HostController(
            config=self.config,          # 안 넘기면 zip 기본값(max_throttle 0.40)이 쓰인다
            mission=self.mission,
            sender=VehicleServerDirectSender(server, car_id),
        )
        self.session = RemoteDirectSession(self.host, server, car_id)
        self.scheduler = ControlScheduler(self.host, period_s=period_s,
                                          on_tick=self._on_tick)
        self._last_status = self.mission.status
        self.last_tick_result = None
        # (car_id, 이전 상태, 새 상태) — 파이프라인이 슬롯 점유·재계획을 처리한다
        self.on_status_change: Callable[[int, MissionStatus, MissionStatus], None] | None = None
        self.session.attach()

    # ─── 라이프사이클 ────────────────────────────────────────────────────────

    def start(self, *, wait_s: float = 2.0) -> None:
        """(STATUS 대기 → 필요 시 RESET) → SET_MODE → ACCEPTED → arm → 제어 루프.

        ACCEPTED 전에는 arm 하지 않는다. 차량이 아직 REMOTE_DIRECT 가 아닌데
        제어값을 보내면 무시되거나 엉뚱한 모드에서 실행될 수 있다.

        SET_MODE 는 READY 에서만 수락되는데, 차량은 통신이 한 번만 끊겨도
        EMERGENCY_STOP 으로 간다. 그래서 거절되면 RESET 후 한 번 더 시도한다.
        """
        self.ensure_remote_direct(wait_s=wait_s)
        self._begin_loop()

    def arm_session(self, *, wait_s: float = 2.0,
                    release_control: bool = True) -> None:
        """REMOTE_DIRECT 협상만 하고 자동 주행은 시작하지 않는다.

        카메라 없이 수동(WASD)만 쓸 때 필요하다. 세션·모드는 열어두되
        auto 스케줄러는 띄우지 않아, 이어서 mux 가 MANUAL 권한을 가져간다.
        """
        self.ensure_remote_direct(wait_s=wait_s)
        self.session._enable_direct_stream(release_control=release_control)
        log.info("car %d: REMOTE_DIRECT 세션 확보 (자동 주행은 미시작)", self.car_id)

    def ensure_remote_direct(self, *, wait_s: float = 2.0) -> None:
        """Request REMOTE_DIRECT; the session layer owns all reliable commands."""
        self.session.ensure_remote_direct(wait_s=wait_s)
        # Negotiation readiness never releases the per-car zero latch.
        self.session._enable_direct_stream(release_control=False)

    # ─── 협상 세부 ───────────────────────────────────────────────────────────

    @property
    def negotiation_state(self) -> NegotiationState:
        return self.session.negotiation_state

    @property
    def negotiation_generation(self) -> int:
        return self.session.negotiation_generation

    @property
    def negotiation_identity(self) -> tuple[str, str] | None:
        return self.session.negotiation_identity

    def _begin_loop(self) -> None:
        self.session._enable_direct_stream()
        if self.host.authority.is_faulted:
            self.host.authority.clear_fault()
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
        """FAULTED 이후 명시적 재출발. 자동 복귀는 하지 않는다.

        stop() 은 제어 루프(ControlScheduler)까지 멈추는데 패키지의
        re_arm_auto() 는 권한만 되살린다. 스케줄러를 같이 켜지 않으면
        "무장됨" 이라고 나오면서 DIRECT_CONTROL 이 한 발도 안 나간다.
        """
        self.session.re_arm_auto(wait_s=wait_s)
        self.scheduler.start()          # 멈춰 있던 100ms 루프 재개
        log.info("car %d: AUTO_HOST 재무장 (제어 루프 재시작)", self.car_id)

    # ─── 파이프라인 연동 ─────────────────────────────────────────────────────

    def on_camera_pose(self, x_mm: float, y_mm: float,
                       heading_deg: float | None, obs_time: float,
                       heading_source: str | None = None) -> None:
        """새 카메라 프레임에서만 호출. 제어 계산은 스케줄러가 한다.

        obs_time 은 **관측 시각**이어야 한다. tick 시각을 넣으면 카메라가 멈춰도
        pose 가 계속 신선해 보여서 stale 판정이 무력화된다.
        """
        self.host.pose_source.observe(x_mm, y_mm, heading_deg, obs_time,
                                      heading_source=heading_source)

    def load_route(self, backend_waypoints: Sequence[Any]) -> None:
        """새 route 로 교체한다. 새 카메라 pose 가 올 때까지 zero 를 유지한다.

        HW 7fc17c6: route 를 갈아끼우는 순간 기존 pose·제어값을 재사용하면
        옛 관측으로 출발할 수 있다. prepare_route_switch() 가 즉시 zero 를
        내보내고 pose_source 를 비운다.
        """
        self.host.prepare_route_switch()
        self.mission.load(waypoints_from_backend(backend_waypoints))
        self._last_status = self.mission.status
        self.last_tick_result = None

    def prepare_route_switch(self) -> None:
        """Hold zero and invalidate the current pose before a staged replan."""
        self.host.prepare_route_switch()
        self.last_tick_result = None

    def load_recovery_waypoints(self, backend_waypoints: Sequence[Any]):
        """REPLAN_REQUIRED 에서 복구 경로를 끼워 넣는다 (HW 7fc17c6).

        복구 waypoint 를 마치면 실패했던 기존 target 과 남은 route 로 자동
        복귀한다. 복구 경로 생성 자체는 상위(파이프라인) 몫이다.
        """
        self.host.prepare_route_switch()
        status = self.mission.load_recovery(waypoints_from_backend(backend_waypoints))
        self._last_status = self.mission.status
        self.last_tick_result = None
        return status

    def confirm_parked(self) -> None:
        self.mission.confirm_parked()
        self._last_status = self.mission.status

    @property
    def current_target(self):
        return self.mission.current_target()

    @property
    def failed_target(self):
        """REPLAN_REQUIRED 를 일으킨 target.

        `current_target` 은 RUNNING 이 아니면 None 을 돌려주므로 재계획
        시점에는 쓸 수 없다. 실패한 target 은 미션이 복귀용 snapshot 의
        맨 앞에 보존해 두는데 공개 접근자가 없어 직접 읽는다.
        """
        resume = getattr(self.mission, "_resume_waypoints", None)
        return resume[0] if resume else None

    @property
    def replan_reason(self) -> str | None:
        return self.mission.replan_reason

    @property
    def current_phase(self):
        return self.mission.current_phase

    @property
    def current_is_terminal(self) -> bool:
        return self.mission.current_is_terminal

    @property
    def parking_active(self) -> bool:
        return self.mission.parking_active

    @property
    def approach_stage(self) -> str:
        return self.host.approach_guard.stage

    @property
    def status(self) -> MissionStatus:
        return self.mission.status

    @property
    def is_faulted(self) -> bool:
        return self.host.authority.is_faulted

    # ─── 내부 ────────────────────────────────────────────────────────────────

    def _on_tick(self, result) -> None:
        """제어 루프가 매 tick 부른다. 상태가 바뀐 순간만 위로 올린다."""
        self.last_tick_result = result
        status = result.mission_status
        if status is self._last_status:
            return
        prev, self._last_status = self._last_status, status
        log.info("car %d: mission %s → %s (authority=%s, %s)",
                 self.car_id, prev.value, status.value,
                 result.authority.value, result.command.reason or result.command.mode.value)
        if self.on_status_change is not None:
            self.on_status_change(self.car_id, prev, status)
