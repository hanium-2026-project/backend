"""Control producer 들 — 서로 다른 제어권(algorithm)이지만 동일한 ControlCommand 를 낸다.

- ManualControlProducer : 사람 입력(throttle, steering) → wire-ready ControlCommand
- AutoControlProducer   : Pose + Waypoint → PoseWaypointController → wire-ready ControlCommand

두 producer 모두 **wire-ready steering(음수=LEFT)** 을 출력해 downstream transport 를 공용화한다.
producer 는 backend/network 를 import 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from controller import geometry as geo
from controller.config import ControllerConfig
from controller.models import ControlCommand, ControlMode, Pose, Waypoint
from controller.pose_controller import PoseWaypointController


@dataclass(frozen=True)
class ManualInput:
    """사람 입력.

    throttle : [-1,1], 양수 = 전진 (allow_reverse=False 면 음수는 0 으로 clamp)
    steering : [-1,1], **논리 부호(+1 = LEFT)**. joystick/keyboard 직관에 맞춘 논리값.
               producer 가 config.wire_steering_sign 으로 wire(음수=LEFT)로 변환한다.
    """

    throttle: float
    steering: float


class ManualControlProducer:
    """사람 입력을 wire-ready ControlCommand 로 변환."""

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()

    def compute(self, manual: Optional[ManualInput]) -> ControlCommand:
        cfg = self.config
        if manual is None:
            return _zero_command("NO_MANUAL_INPUT")

        throttle = geo.clamp(float(manual.throttle), -1.0, 1.0)
        if not cfg.allow_reverse and throttle < 0.0:
            throttle = 0.0
        throttle = geo.clamp(throttle, 0.0 if not cfg.allow_reverse else -cfg.max_throttle,
                             cfg.max_throttle)

        logical = geo.clamp(float(manual.steering), -1.0, 1.0)  # +1 = LEFT(논리)
        wire = geo.clamp(cfg.wire_steering_sign * logical, -1.0, 1.0)  # 음수 = LEFT(wire)
        wire = round(wire, 4) or 0.0
        throttle = round(throttle, 4) or 0.0

        mode = ControlMode.DRIVE if throttle > 0.0 else ControlMode.HOLD
        return ControlCommand(
            throttle=throttle,
            steering=wire,
            mode=mode,
            arrived=False,
            distance_error_cm=0.0,
            heading_error_deg=0.0,
            target_bearing_deg=0.0,
            reason="MANUAL",
            logical_steering=round(logical, 4) or 0.0,
        )


class AutoControlProducer:
    """Pose + Waypoint → wire-ready ControlCommand (core controller 래핑)."""

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()
        self.controller = PoseWaypointController(self.config)

    def reset(self) -> None:
        self.controller.reset()

    def compute(
        self,
        pose: Optional[Pose],
        target: Optional[Waypoint],
        *,
        allow_drive: bool = True,
        now: float,
    ) -> ControlCommand:
        # target 이 없으면(미션 완료/미로드) 안전 정지.
        if target is None:
            return _zero_command("NO_TARGET")
        # pose 가 아직 없으면(관측 전) 안전 정지(WARMUP).
        if pose is None:
            return _zero_command("NO_POSE")
        return self.controller.compute(pose, target, allow_drive=allow_drive, now=now)


def _zero_command(reason: str) -> ControlCommand:
    return ControlCommand(
        throttle=0.0,
        steering=0.0,
        mode=ControlMode.HOLD,
        arrived=False,
        distance_error_cm=0.0,
        heading_error_deg=0.0,
        target_bearing_deg=0.0,
        reason=reason,
    )
