"""파이프라인 → 대시보드 WebSocket 브로드캐스트 어댑터.

파이프라인은 Django 프로세스 밖에서도 돌아야 하므로(카메라 루프 단독 실행),
Django 가 준비돼 있지 않으면 조용히 아무것도 하지 않는다. 대시보드가 없다고
차량 제어가 멈추면 안 된다.

- pose: 스트림. 초당 수 회 → 별도 타입으로 보내고 대시보드 재조회를 유발하지 않음
- event: 상태 변화 시점에만. 대시보드가 REST 를 다시 조회한다
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

_UNAVAILABLE_LOGGED = False


def _services():
    """Django 가 사용 가능할 때만 parking.services 를 로드한다."""
    global _UNAVAILABLE_LOGGED
    try:
        from django.apps import apps
        if not apps.ready:
            return None
        from parking import services
        return services
    except Exception as exc:                      # Django 미설정/미기동
        if not _UNAVAILABLE_LOGGED:
            log.info("dashboard broadcast disabled (%s)", exc)
            _UNAVAILABLE_LOGGED = True
        return None


class DashboardBridge:
    """차량 관측·이벤트를 대시보드로 중계한다 (실패해도 파이프라인에 영향 없음)."""

    def __init__(self, pose_interval_s: float = 0.2) -> None:
        self.pose_interval_s = pose_interval_s
        self._last_pose_at: dict[tuple[str, int | None], float] = {}

    def push_pose(self, car_id: int | None, position_mm: tuple[float, float],
                  status: str = "moving", heading_deg: float | None = None,
                  heading_source: str | None = None, parking_phase: str | None = None,
                  route_id: int | None = None, waypoint_id: int | None = None,
                  target_spot_id: int | None = None,
                  track_id: int | None = None) -> None:
        """실시간 위치를 보낸다. 화면 갱신에 필요한 정도로만 솎아낸다.

        car_id 가 없는(=아직 ESP32 와 바인딩되지 않은) 차량도 보낸다. 카메라가
        보고 있는데 화면에 없으면 관제 입장에서 인식 실패와 구분할 수 없다.
        """
        # 바인딩 전에는 track_id 가 유일한 키다. 두 키 공간이 섞이면 서로를
        # 덮어써서 한 대가 사라진다.
        key = ("car", car_id) if car_id is not None else ("track", track_id)
        now = time.monotonic()
        if now - self._last_pose_at.get(key, 0.0) < self.pose_interval_s:
            return
        services = _services()
        if services is None:
            return
        self._last_pose_at[key] = now
        from parking.protocol import VehicleTelemetryMessage
        services.broadcast_vehicle_pose(VehicleTelemetryMessage(
            car_id=car_id, license_plate="", pos=position_mm, status=status,
            target_spot_id=target_spot_id, heading_deg=heading_deg,
            heading_source=heading_source, parking_phase=parking_phase,
            route_id=route_id, waypoint_id=waypoint_id, track_id=track_id,
        ))

    def push_event(self, event: str, **payload) -> None:
        """상태 변화 알림 (슬롯 배정·주차 완료·충돌 정지 등)."""
        services = _services()
        if services is None:
            return
        services.broadcast_vehicle_event(event, payload)
