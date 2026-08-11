"""backend_contract — 실제 backend 부재 시 사용하는 reference 계약 재현 test double.

실제 backend(comm.server.VehicleServer, comm.tests.mock_firmware.MockFirmware,
parking.waypoints.Waypoint)가 PYTHONPATH 에 있으면 real-wire E2E 테스트가 그쪽을 사용한다.
여기 spec 은 실제 최신 API 시그니처(send_set_mode->int, push_control->None, 속성 callback,
direct_control_enabled, int car_id)를 재현한다.
"""

from .mock_firmware_spec import MockFirmware, FwMode, FwState, DIRECT_CONTROL_TIMEOUT_S
from .vehicle_server_spec import SpecVehicleServer, CONTROL_INTERVAL_S, wire_car_id
from .waypoint_spec import SpecWaypoint

__all__ = [
    "MockFirmware", "FwMode", "FwState", "DIRECT_CONTROL_TIMEOUT_S",
    "SpecVehicleServer", "CONTROL_INTERVAL_S", "wire_car_id",
    "SpecWaypoint",
]
