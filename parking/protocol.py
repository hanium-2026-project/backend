"""WebSocket message schema helpers for live parking telemetry."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VehicleTelemetryMessage:
    """Browser/device message carrying a vehicle's position and route target.

    The JSON shape intentionally mirrors the requested MQTT/WebSocket sample so
    a future MQTT adapter can reuse the same DTO without changing UI contracts.
    """

    car_id: int
    license_plate: str
    pos: tuple[float, float]
    status: str
    target_spot_id: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VehicleTelemetryMessage":
        """Validate a raw dict and convert it into a typed telemetry message."""
        pos = payload.get("pos", [0.0, 0.0])
        if not isinstance(pos, list | tuple) or len(pos) != 2:
            raise ValueError("pos must be a two-item coordinate list")
        return cls(
            car_id=int(payload["car_id"]),
            license_plate=str(payload["license_plate"]),
            pos=(float(pos[0]), float(pos[1])),
            status=str(payload.get("status", "moving")),
            target_spot_id=payload.get("target_spot_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable payload for WebSocket broadcasting."""
        return {
            "car_id": self.car_id,
            "license_plate": self.license_plate,
            "pos": [self.pos[0], self.pos[1]],
            "status": self.status,
            "target_spot_id": self.target_spot_id,
        }
