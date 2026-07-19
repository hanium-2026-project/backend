"""주행 오케스트레이터 — 노트북 도착 판정 기반 waypoint 순차 운용.

확정 설계:
- 도착 판정은 노트북이 카메라 pose로 수행 (ESP는 추종만)
- waypoint 전환 사이클: 도착 판정 → WAIT → WAYPOINT(target 교체) → GO
- FINAL 도착 시 PARKED 재검증 (§11) 후 슬롯 OCCUPIED 처리
- 충돌 위험(SafetyEvent) 시 WAIT → 현재 pose 기준 route_id 증가 재생성 (§32)

카메라 지연 보정: 판정 시 position_tolerance에 lead_cm(지연×속도)을 더해
차량이 목표를 지나치기 전에 전환을 시작한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from parking.waypoints import Waypoint, is_waypoint_reached

from .server import VehicleServer


class MissionState(Enum):
    IDLE = "IDLE"                  # 경로 없음
    DRIVING = "DRIVING"            # 현재 waypoint 추종 중
    SWITCHING = "SWITCHING"        # 도착 → WAIT 보냄, WAITING 확인 대기
    LOADING = "LOADING"            # 다음 WAYPOINT 전송, ack 대기
    RESUMING = "RESUMING"          # GO 전송, MOVING 확인 대기
    HELD = "HELD"                  # 안전 WAIT (충돌 회피) — 재생성/GO 대기
    PARKED_CHECK = "PARKED_CHECK"  # FINAL 도착, 카메라 재검증 중
    DONE = "DONE"                  # PARKED 확정


@dataclass
class Mission:
    car_id: int
    route_id: int
    waypoints: list[Waypoint]
    index: int = 0                 # 현재 목표 waypoint 인덱스
    state: MissionState = MissionState.IDLE
    slot_id: str | None = None

    @property
    def current(self) -> Waypoint | None:
        return self.waypoints[self.index] if self.index < len(self.waypoints) else None


class MissionOrchestrator:
    """차량별 Mission을 상태기계로 구동한다.

    호출 규약:
      - update_pose(car_id, pos_mm, heading_deg): CV 루프가 프레임마다 호출
      - on_vehicle_status(car_id, status): server.on_status에 연결
      - start_mission / hold / regenerate: 상위(할당·safety)에서 호출
    """

    def __init__(
        self,
        server: VehicleServer,
        camera_lead_cm: float = 2.0,
        on_parked: Callable[[int, str], None] | None = None,
    ) -> None:
        self.server = server
        self.camera_lead_cm = camera_lead_cm
        self.on_parked = on_parked
        self.missions: dict[int, Mission] = {}
        self._route_counter = 0

    # ─── 미션 시작 / 재생성 ──────────────────────────────────────────────────

    def next_route_id(self) -> int:
        self._route_counter += 1
        return self._route_counter

    def start_mission(self, car_id: int, waypoints: list[Waypoint],
                      slot_id: str | None = None) -> Mission:
        """새 경로 시작: 첫 WAYPOINT 전송 → (ack 후) GO."""
        if not waypoints:
            raise ValueError("waypoints empty")
        m = Mission(car_id, waypoints[0].route_id, list(waypoints), slot_id=slot_id)
        self.missions[car_id] = m
        self._load_current(m)
        return m

    def hold(self, car_id: int, reason: str = "COLLISION_RISK") -> None:
        """안전 정지 (SafetyEvent 처리). 재개는 resume() 또는 regenerate()."""
        m = self.missions.get(car_id)
        if m is None or m.state in (MissionState.DONE, MissionState.IDLE):
            return
        self.server.send_wait(car_id, reason)
        m.state = MissionState.HELD

    def resume(self, car_id: int) -> None:
        """HELD에서 동일 경로 재개 (위험 해소, 경로 유지 시)."""
        m = self.missions.get(car_id)
        if m is None or m.state is not MissionState.HELD or m.current is None:
            return
        self.server.send_go(car_id, m.route_id, m.current.waypoint_id)
        m.state = MissionState.RESUMING

    def regenerate(self, car_id: int, new_waypoints: list[Waypoint]) -> None:
        """HELD에서 새 경로로 교체 (§32): 새 route_id의 waypoint 목록 필요."""
        m = self.missions.get(car_id)
        if m is None:
            return
        m.route_id = new_waypoints[0].route_id
        m.waypoints = list(new_waypoints)
        m.index = 0
        self._load_current(m)

    # ─── CV 루프 연동 (도착 판정) ────────────────────────────────────────────

    def update_pose(self, car_id: int, position_mm: tuple[float, float],
                    heading_deg: float | None) -> None:
        m = self.missions.get(car_id)
        if m is None or m.current is None:
            return

        if m.state is MissionState.DRIVING and self._reached(m.current, position_mm, heading_deg):
            if m.current.is_final:
                # FINAL: 정지시키고 카메라 재검증 단계로
                self.server.send_wait(car_id, "FINAL_CHECK")
                m.state = MissionState.PARKED_CHECK
            else:
                self.server.send_wait(car_id, "WP_SWITCH")
                m.state = MissionState.SWITCHING

        elif m.state is MissionState.PARKED_CHECK:
            # 재검증: 선행 보정 없이 원 tolerance로, 정지 상태에서 재확인 (§11)
            if is_waypoint_reached(m.current, position_mm, heading_deg):
                m.state = MissionState.DONE
                if self.on_parked is not None and m.slot_id is not None:
                    self.on_parked(car_id, m.slot_id)
            # 미충족 시 상위에서 미세 조정 경로 재생성 판단 (DONE 안 됨)

    def _reached(self, wp: Waypoint, pos_mm: tuple[float, float],
                 heading: float | None) -> bool:
        """카메라 지연 선행 보정을 적용한 도착 판정."""
        lead_mm = self.camera_lead_cm * 10.0
        dist = math.hypot(pos_mm[0] - wp.x, pos_mm[1] - wp.y)
        if dist > wp.position_tolerance_cm * 10.0 + lead_mm:
            return False
        if not wp.heading_required:
            return True
        if heading is None:
            return False
        err = abs((heading - (wp.target_heading_deg or 0.0) + 180.0) % 360.0 - 180.0)
        return err <= wp.heading_tolerance_deg

    # ─── 차량 STATUS 연동 (상태 전이 구동) ───────────────────────────────────

    def on_vehicle_status(self, car_id: int, status: dict[str, Any]) -> None:
        m = self.missions.get(car_id)
        if m is None:
            return
        state = status.get("state")

        if m.state is MissionState.SWITCHING and state == "WAITING":
            m.index += 1                      # 다음 waypoint 로드
            if m.current is None:
                m.state = MissionState.DONE   # 방어적 처리 (FINAL은 보통 위에서 분기)
                return
            self._load_current(m)

        elif m.state is MissionState.LOADING and state == "WAITING" \
                and status.get("target_loaded") \
                and status.get("route_id") == m.route_id \
                and status.get("waypoint_id") == (m.current.waypoint_id if m.current else -1):
            self.server.send_go(car_id, m.route_id, m.current.waypoint_id)
            m.state = MissionState.RESUMING

        elif m.state is MissionState.RESUMING and state == "MOVING":
            m.state = MissionState.DRIVING

        elif m.state is MissionState.LOADING and state == "READY":
            # READY 상태에서 WAYPOINT를 받으면 문서상 즉시 MOVING (§18.4)
            pass

        elif m.state is MissionState.LOADING and state == "MOVING":
            m.state = MissionState.DRIVING    # READY→WAYPOINT→MOVING 경로 (§18.4)

    def _load_current(self, m: Mission) -> None:
        wp = m.current
        wire = wp.to_wire()
        self.server.send_waypoint(m.car_id, wire)
        m.state = MissionState.LOADING
