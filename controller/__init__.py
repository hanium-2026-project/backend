"""Host-side Pose -> Waypoint 주행 제어기 (standalone core).

이 패키지는 2026 한이음 RC카 자율주차 프로젝트의 B안 host-side 제어기 코어다.

목표
----
현재 차량 Pose(x, y, heading) + 목표 Waypoint 를 입력받아,
실제 ESP32 DIRECT_CONTROL 에 그대로 넣을 수 있는 **wire-ready**
throttle / steering 을 계산한다.

독립성
------
core 모듈(models, geometry, config, pose_controller)은
backend/comm/Django/Redis/YOLO/socket/thread 를 절대 import 하지 않는다.
backend 연동은 adapter_example.py 에서만 duck-typing 으로 처리한다.

Source of truth
---------------
실제 ESP32 펌웨어(actuator.c, app_config.example.h)의 부호/범위가 최우선이다.
- steering wire 부호: 음수 = LEFT, 0 = CENTER, 양수 = RIGHT
- throttle: steering 의존 PWM (아직 cm/s 미보정)
"""

from .models import Pose, Waypoint, ControlCommand, ControlMode
from .config import ControllerConfig, FirmwareConstants
from .pose_controller import PoseWaypointController

__all__ = [
    "Pose",
    "Waypoint",
    "ControlCommand",
    "ControlMode",
    "ControllerConfig",
    "FirmwareConstants",
    "PoseWaypointController",
]
