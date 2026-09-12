"""integration — production 연동 계층 (backend/network import 없음, duck-typing).

- backend_adapter       : pose/waypoint 매핑 + VehicleServerDirectSender(push_control 위임)
- camera_adapter        : CameraObservationAdapter(관측 timestamp 보존)
- control_scheduler     : ControlScheduler(camera 독립 100ms control loop)
- remote_direct_session : RemoteDirectSession(SET_MODE handshake + comm fault 연동)
- backend_contract/     : 실제 backend 부재 시 사용하는 reference 계약 재현 test double
"""

from .backend_adapter import (
    pose_from_backend,
    waypoint_from_backend,
    waypoints_from_backend,
    VehicleServerDirectSender,
    BACKEND_REUSE_NOTE,
)
from .camera_adapter import CameraObservationAdapter
from .control_scheduler import ControlScheduler
from .remote_direct_session import RemoteDirectSession, ModeHandshakeError

__all__ = [
    "pose_from_backend",
    "waypoint_from_backend",
    "waypoints_from_backend",
    "VehicleServerDirectSender",
    "BACKEND_REUSE_NOTE",
    "CameraObservationAdapter",
    "ControlScheduler",
    "RemoteDirectSession",
    "ModeHandshakeError",
]
