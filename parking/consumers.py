"""Django Channels consumers for real-time dashboard updates."""

import json

from channels.generic.websocket import AsyncWebsocketConsumer

from .protocol import VehicleTelemetryMessage


class DashboardConsumer(AsyncWebsocketConsumer):
    """Broadcast parking telemetry and state changes to connected dashboards."""

    group_name = "parking_dashboard"

    async def connect(self) -> None:
        """Join the shared dashboard group and send a small connection event."""
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        await self.send_json({"type": "connected", "message": "dashboard websocket ready"})

    async def disconnect(self, close_code: int) -> None:
        """Leave the broadcast group when the browser disconnects."""
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data: str | None = None, bytes_data: bytes | None = None) -> None:
        """Validate incoming telemetry and rebroadcast it to the dashboard group."""
        if not text_data:
            return
        try:
            message = VehicleTelemetryMessage.from_dict(json.loads(text_data))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            await self.send_json({"type": "error", "message": str(exc)})
            return
        await self.channel_layer.group_send(
            self.group_name,
            {"type": "parking.telemetry", "payload": message.to_dict()},
        )

    async def parking_telemetry(self, event: dict) -> None:
        """Forward normalized vehicle telemetry to WebSocket clients."""
        await self.send_json({"type": "vehicle.telemetry", "payload": event["payload"]})

    async def parking_state(self, event: dict) -> None:
        """Forward REST-originated state changes to WebSocket clients."""
        await self.send_json({"type": "parking.state", "payload": event["payload"]})

    async def send_json(self, content: dict) -> None:
        """Serialize a Python dict to JSON for the WebSocket connection."""
        await self.send(text_data=json.dumps(content, ensure_ascii=False))
