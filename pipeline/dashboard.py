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
        # car_id → 차량 번호. 화면이 "누구인지" 보여주려면 필요하다.
        # None 을 넣어 두면 "DB 에 없음"을 기억해 매 프레임 조회하지 않는다.
        self._plate_cache: dict[int, str | None] = {}

    def _plate_of(self, car_id: int | None) -> str:
        """car_id 에 해당하는 차량 번호를 찾는다 (없으면 빈 문자열).

        통신 계층의 car_id 는 "CAR_01" → 1 이고, DB 의 Vehicle.vehicle_id 와
        같은 번호를 쓰기로 한 약속이다. 조회 실패는 화면 라벨이 비는 것으로
        끝나야 하며 pose 전송을 막아서는 안 된다.
        """
        if car_id is None:
            return ""
        if car_id in self._plate_cache:
            return self._plate_cache[car_id] or ""
        plate = None
        try:
            from parking.models import Vehicle
            row = Vehicle.objects.filter(vehicle_id=car_id).only("license_plate").first()
            if row is not None:
                plate = row.license_plate
        except Exception as exc:
            log.debug("plate lookup failed (car=%s): %s", car_id, exc)
            return ""                      # 캐시하지 않는다 — 일시적 오류일 수 있다
        self._plate_cache[car_id] = plate
        return plate or ""

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
            car_id=car_id, license_plate=self._plate_of(car_id),
            pos=position_mm, status=status,
            target_spot_id=target_spot_id, heading_deg=heading_deg,
            heading_source=heading_source, parking_phase=parking_phase,
            route_id=route_id, waypoint_id=waypoint_id, track_id=track_id,
        ))

    def push_event(self, event: str, **payload) -> None:
        """상태 변화 알림 (슬롯 배정·주차 완료·충돌 정지 등)."""
        services = _services()
        if services is None:
            return
        # 슬롯 배정은 DB 에도 남긴다. 파이프라인은 지금까지 이벤트만 쏘고
        # 아무것도 기록하지 않아서, 화면은 차가 A2 로 가는데 출입차 기록은
        # 비어 있는 상태가 됐다. 배정을 곧 입차로 본다.
        if event == "slot_assigned":
            self._record_entry(services, payload)
        services.broadcast_vehicle_event(event, payload)

    def _record_entry(self, services, payload: dict) -> None:
        """slot_assigned → 입차 기록. 실패해도 파이프라인을 막지 않는다."""
        slot = payload.get("slot")
        plate = self._plate_of(payload.get("car_id"))
        if not slot or not plate:
            return                     # 번호를 모르면 남길 기록도 없다
        try:
            services.record_pipeline_entry(plate, str(slot))
        except Exception as exc:
            log.warning("입차 기록 실패 (car=%s slot=%s): %s",
                        payload.get("car_id"), slot, exc)
