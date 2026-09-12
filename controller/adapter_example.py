"""backend 연동 예시 (duck-typing). **backend 를 import 하지 않는다.**

이 파일은 나중에 SW backend 의 Pose/Waypoint 객체를 core 제어기에 넣고,
core 의 ControlCommand 를 ESP32 DIRECT_CONTROL payload 로 바꾸는 방법을 보여준다.
실제 backend 클래스를 import 하지 않고 getattr 로만 접근하므로,
core 의 독립성을 깨지 않는다.

주의
----
- core(pose_controller/geometry/models/config)만 실제 제어에 쓰고,
  이 adapter 는 "예시/참고"다. 실제 통합은 INTEGRATION_GUIDE.md 를 따른다.
- steering 은 이미 **wire-ready**(음수=LEFT)이므로 DIRECT_CONTROL 에 그대로 넣는다.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import ControllerConfig
from .models import ControlCommand, ControlMode, Pose, Waypoint
from .pose_controller import PoseWaypointController


# --------------------------------------------------------------------- mapping
def pose_from_backend(obj: Any, *, timestamp: float) -> Pose:
    """backend pose 유사 객체 → core Pose (duck-typing).

    기대 속성(있으면 사용, 없으면 합리적 기본):
        x_mm, y_mm, heading_deg, valid
    timestamp 는 신선도 판정을 위해 외부(호출자)에서 monotonic 으로 주입.
    """
    return Pose(
        x_mm=float(getattr(obj, "x_mm")),
        y_mm=float(getattr(obj, "y_mm")),
        heading_deg=_opt_float(getattr(obj, "heading_deg", None)),
        timestamp=float(timestamp),
        valid=bool(getattr(obj, "valid", True)),
    )


def waypoint_from_backend(obj: Any) -> Waypoint:
    """backend Waypoint(parking/waypoints.py) 유사 객체 → core Waypoint.

    backend Waypoint 는 위치를 mm 단위 .x / .y 로 들고 있다(코어도 mm 사용).
    """
    x_mm = getattr(obj, "x_mm", None)
    if x_mm is None:
        x_mm = getattr(obj, "x")  # backend Waypoint.x 는 mm
    y_mm = getattr(obj, "y_mm", None)
    if y_mm is None:
        y_mm = getattr(obj, "y")
    return Waypoint(
        x_mm=float(x_mm),
        y_mm=float(y_mm),
        target_heading_deg=_opt_float(getattr(obj, "target_heading_deg", None)),
        speed_cm_s=float(getattr(obj, "speed_cm_s", 12.0)),
        position_tolerance_cm=float(getattr(obj, "position_tolerance_cm", 8.0)),
        heading_tolerance_deg=float(getattr(obj, "heading_tolerance_deg", 30.0)),
        heading_required=bool(getattr(obj, "heading_required", False)),
        route_id=getattr(obj, "route_id", None),
        waypoint_id=getattr(obj, "waypoint_id", None),
        phase=getattr(obj, "phase", None),
    )


def command_to_direct_control(cmd: ControlCommand, control_seq: int) -> dict:
    """core ControlCommand → ESP32 DIRECT_CONTROL wire payload(dict).

    steering 은 이미 wire-ready(음수=LEFT). NDJSON 직렬화는 호출부에서.
    실제 필드명은 펌웨어 protocol.c 계약에 맞춰 최종 조정할 것(여기선 예시).
    """
    return {
        "type": "DIRECT_CONTROL",
        "control_seq": int(control_seq),
        "throttle": round(float(cmd.throttle), 4),
        "steering": round(float(cmd.steering), 4),  # wire-ready
    }


def zero_direct_control(control_seq: int) -> dict:
    """카메라 로스트/트랙 손실 시 반드시 보낼 zero 명령.

    ⚠ host 가 마지막 non-zero 명령을 계속 재송신하면 ESP32 500ms direct timeout 이
    발동하지 않을 수 있다. camera 실패 시 latest_control 을 이 zero 로 갱신해야 한다.
    """
    return {"type": "DIRECT_CONTROL", "control_seq": int(control_seq),
            "throttle": 0.0, "steering": 0.0}


# ------------------------------------------------------ 예시 통합 루프 (문서용)
class HostControlSession:
    """호스트 제어 루프 스켈레톤(예시).

    실제 네트워크/스레드/카메라는 이 클래스가 다루지 않는다.
    호출자가 100ms(10Hz)마다 step() 을 호출하고, 반환된 dict 를 그대로
    DIRECT_CONTROL 로 전송하면 된다.

    핵심 안전 규칙: camera frame 이 없으면 pose_or_none=None → **zero 를 전송**한다.
    """

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.controller = PoseWaypointController(config)
        self._seq = 0

    def step(
        self,
        pose_or_none: Optional[Pose],
        waypoint: Waypoint,
        *,
        allow_drive: bool,
        now: float,
    ) -> dict:
        """한 주기 실행. 항상 전송 가능한 DIRECT_CONTROL dict 를 반환한다."""
        self._seq += 1
        # camera 프레임 자체가 없으면(트랙 로스트/루프 정지) 무조건 zero.
        if pose_or_none is None:
            return zero_direct_control(self._seq)
        cmd = self.controller.compute(
            pose_or_none, waypoint, allow_drive=allow_drive, now=now
        )
        # arrived/HOLD/ALIGN 은 throttle=steering=0 이므로 그대로 전송해도 안전.
        return command_to_direct_control(cmd, self._seq)


def _opt_float(v: Any) -> Optional[float]:
    return None if v is None else float(v)


# NOTE: backend 의 기존 WaypointController 를 재사용하려면 steering_sign 을 -1.0 로
# 설정해야 실제 ESP32 wire 부호(음수=LEFT)와 맞는다. (기본 +1.0 은 좌우 반전됨.)
BACKEND_REUSE_NOTE = (
    "backend control.waypoint_controller.WaypointController 재사용 시 "
    "VehicleLimits(steering_sign=-1.0) 로 설정할 것 (기본 +1.0 은 실제 ESP32 에서 좌우 반전)."
)
