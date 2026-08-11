"""backend 연동 adapter — backend 를 import 하지 않는다(duck-typing).

역할
----
- backend pose/waypoint 유사 객체 → core 모델 매핑
- HostController 의 DIRECT_CONTROL payload 를 backend VehicleServer.push_control 로 흘리는 sink 연결
- 호출 순서/모드 선택 예시 제공

주의: 이 adapter 는 예시/참고이며, 실제 연결 지점은 INTEGRATION_GUIDE.md 를 따른다.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from controller.models import MotionDirection, Pose, Waypoint


# --------------------------------------------------------------------- mapping
def pose_from_backend(obj: Any, *, obs_time: float) -> Pose:
    """backend/CV pose 유사 객체 → core Pose.

    ★ obs_time 은 **카메라 관측이 발생한 시각**이어야 한다(현재 tick 시각이 아님).
      이 값이 stale 판정의 기준이 되므로, 새 프레임이 없으면 갱신하지 말 것.
    """
    return Pose(
        x_mm=float(getattr(obj, "x_mm")),
        y_mm=float(getattr(obj, "y_mm")),
        heading_deg=_opt_float(getattr(obj, "heading_deg", None)),
        timestamp=float(obs_time),
        valid=bool(getattr(obj, "valid", True)),
    )


def waypoint_from_backend(obj: Any) -> Waypoint:
    """backend Waypoint(parking/waypoints.py, .x/.y mm) → core Waypoint(mm)."""
    x_mm = getattr(obj, "x_mm", None)
    if x_mm is None:
        x_mm = getattr(obj, "x")
    y_mm = getattr(obj, "y_mm", None)
    if y_mm is None:
        y_mm = getattr(obj, "y")
    raw_direction = getattr(obj, "motion_direction", MotionDirection.FORWARD)
    try:
        direction = (raw_direction if isinstance(raw_direction, MotionDirection)
                     else MotionDirection(str(raw_direction).upper()))
    except ValueError as exc:
        raise ValueError(f"invalid motion_direction: {raw_direction!r}") from exc
    return Waypoint(
        x_mm=float(x_mm),
        y_mm=float(y_mm),
        target_heading_deg=_opt_float(getattr(obj, "target_heading_deg", None)),
        speed_cm_s=float(getattr(obj, "speed_cm_s", 12.0)),
        position_tolerance_cm=float(getattr(obj, "position_tolerance_cm", 8.0)),
        capture_tolerance_cm=_opt_float(getattr(obj, "capture_tolerance_cm", None)),
        heading_tolerance_deg=float(getattr(obj, "heading_tolerance_deg", 30.0)),
        heading_required=bool(getattr(obj, "heading_required", False)),
        is_final=bool(getattr(obj, "is_final", False)),
        route_id=getattr(obj, "route_id", None),
        waypoint_id=getattr(obj, "waypoint_id", None),
        phase=getattr(obj, "phase", None),
        motion_direction=direction,
    )


def waypoints_from_backend(objs: List[Any]) -> List[Waypoint]:
    return [waypoint_from_backend(o) for o in objs]


# ------------------------------------------------------------------- transport
class VehicleServerDirectSender:
    """HostController.sender 인터페이스(send_command/send_zero)를 구현하되,
    실제 backend VehicleServer 에 위임한다.

    ★ 실제 API(최신 develop 계약):
      - server.push_control(car_id:int, throttle, steering) -> **None**
      - server.stop_control(car_id:int) -> None
      - control_seq / session_id / DIRECT_CONTROL wire 는 VehicleServer 소유.
    이 sender 는 자체 wire seq 를 관리하지 않는다(시뮬레이션용 DirectControlSender 와 구분).
    car_id 는 int(1,2). server 는 duck-typing(backend 미import).
    """

    def __init__(self, server: Any, car_id: int, *,
                 use_stop_control_for_zero: bool = False) -> None:
        assert isinstance(car_id, int), "production car_id 는 int(1,2) 여야 함"
        self._server = server
        self._car_id = car_id
        self._use_stop = use_stop_control_for_zero
        if not hasattr(server, "push_control"):
            raise AttributeError("server 에 push_control(car_id, throttle, steering) 이 없습니다.")

    @property
    def car_id(self) -> int:
        return self._car_id

    def send_command(self, cmd: Any) -> Dict:
        return self._push(cmd.throttle, cmd.steering)

    def send_zero(self) -> Dict:
        if self._use_stop and hasattr(self._server, "stop_control"):
            self._server.stop_control(self._car_id)   # 반환 None
            return self._payload(0.0, 0.0, via="stop_control")
        return self._push(0.0, 0.0)

    # ----------------------------------------------------------------- 내부
    def _push(self, throttle: float, steering: float) -> Dict:
        self._server.push_control(self._car_id, throttle, steering)  # ★ 반환 None
        return self._payload(throttle, steering)

    def _read_seq(self):
        # control_seq 는 서버 소유. 노출돼 있으면 관측용으로만 읽는다(제어에 사용 안 함).
        reader = getattr(self._server, "control_seq", None)
        if callable(reader):
            try:
                return reader(self._car_id)
            except (TypeError, KeyError):
                return None
        return None

    def _payload(self, throttle, steering, *, via: str = "push_control") -> Dict:
        return {
            "type": "DIRECT_CONTROL",
            "via": via,
            "car_id": self._car_id,          # int
            "throttle": round(float(throttle), 4),
            "steering": round(float(steering), 4),  # wire-ready(음수=LEFT)
            "control_seq": self._read_seq(),  # ← 서버 소유. 관측용(없으면 None).
        }


# NOTE: 기존 backend WaypointController 재사용 시 steering_sign=-1.0 필요(기본 +1.0 은 좌우 반전).
BACKEND_REUSE_NOTE = (
    "control.waypoint_controller.VehicleLimits(steering_sign=-1.0) 로 설정해야 "
    "실제 ESP32 wire 부호(음수=LEFT)와 일치."
)


def _opt_float(v: Any) -> Optional[float]:
    return None if v is None else float(v)
