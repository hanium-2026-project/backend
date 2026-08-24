"""WebSocket message schema helpers for live parking telemetry."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VehicleTelemetryMessage:
    """Browser/device message carrying a vehicle's position and route target.

    The JSON shape intentionally mirrors the requested MQTT/WebSocket sample so
    a future MQTT adapter can reuse the same DTO without changing UI contracts.
    """

    # None = 카메라는 보고 있지만 아직 차량(ESP32)과 바인딩되지 않은 상태.
    # 관제 화면에서 "안 보인다"와 "제어할 수 없다"는 다른 상황이므로 구분한다.
    car_id: int | None
    license_plate: str
    pos: tuple[float, float]              # (x, y) mm — ParkingSpot.coord_x/y 와 같은 좌표계
    status: str
    target_spot_id: int | None = None
    # CV 파이프라인이 채우는 실시간 관측/주행 정보 (없으면 None)
    heading_deg: float | None = None      # 0~360, 오른쪽 0° / 위 90°
    heading_source: str | None = None     # TRAJECTORY | LAST_VALID | FRONT_CUSHION
    parking_phase: str | None = None      # CRUISE | APPROACH | ALIGN | ENTRY | FINAL
    route_id: int | None = None
    waypoint_id: int | None = None
    # 추적 ID — car_id 가 없을 때 화면이 차량을 구분할 유일한 키다.
    track_id: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VehicleTelemetryMessage":
        """Validate a raw dict and convert it into a typed telemetry message."""
        pos = payload.get("pos", [0.0, 0.0])
        if not isinstance(pos, list | tuple) or len(pos) != 2:
            raise ValueError("pos must be a two-item coordinate list")
        return cls(
            car_id=(None if payload.get("car_id") is None else int(payload["car_id"])),
            # 카메라만으로는 번호판을 알 수 없으므로 파이프라인 발신 시 비어 있을 수 있다
            license_plate=str(payload.get("license_plate", "")),
            pos=(float(pos[0]), float(pos[1])),
            status=str(payload.get("status", "moving")),
            target_spot_id=payload.get("target_spot_id"),
            heading_deg=payload.get("heading_deg"),
            heading_source=payload.get("heading_source"),
            parking_phase=payload.get("parking_phase"),
            route_id=payload.get("route_id"),
            waypoint_id=payload.get("waypoint_id"),
            track_id=payload.get("track_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable payload for WebSocket broadcasting."""
        return {
            "car_id": self.car_id,
            "license_plate": self.license_plate,
            "pos": [self.pos[0], self.pos[1]],
            "status": self.status,
            "target_spot_id": self.target_spot_id,
            "heading_deg": self.heading_deg,
            "heading_source": self.heading_source,
            "parking_phase": self.parking_phase,
            "route_id": self.route_id,
            "waypoint_id": self.waypoint_id,
            "track_id": self.track_id,
        }
