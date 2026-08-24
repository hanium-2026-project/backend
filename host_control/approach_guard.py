"""APPROACH COARSE->FINE capture / miss guard.

동일 APPROACH Pose를 두 단계로 해석한다.
- COARSE: capture 반경(기본 10cm) 안에 한 번 들어와야 정상 접근.
- FINE: 같은 목표를 position_tolerance_cm(초기 5cm)까지 정밀 추종.

목표 progress line을 FINE 완료 전에 지나치면 APPROACH_MISSED로 전환해
상위 HostController가 즉시 STOP + REPLAN_REQUIRED로 넘긴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from controller.config import ControllerConfig
from controller.models import Pose, Waypoint


class ApproachStage(str, Enum):
    INACTIVE = "INACTIVE"
    COARSE = "COARSE"
    FINE = "FINE"


class ApproachEvent(str, Enum):
    CAPTURED = "APPROACH_CAPTURED"
    COARSE_MISSED = "APPROACH_COARSE_MISSED"
    FINE_MISSED = "APPROACH_FINE_MISSED"


@dataclass(frozen=True)
class ApproachProgress:
    stage: ApproachStage
    event: Optional[ApproachEvent]
    distance_cm: float
    best_distance_cm: float
    along_track_cm: Optional[float]

    @property
    def missed(self) -> bool:
        return self.event in {ApproachEvent.COARSE_MISSED, ApproachEvent.FINE_MISSED}


class ApproachProgressGuard:
    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()
        self.reset()

    def reset(self) -> None:
        self._target_key = None
        self._stage = ApproachStage.INACTIVE
        self._anchor_mm: Optional[tuple[float, float]] = None
        self._best_distance_cm: Optional[float] = None

    @property
    def stage(self) -> ApproachStage:
        return self._stage

    @property
    def best_distance_cm(self) -> Optional[float]:
        return self._best_distance_cm

    def evaluate(self, pose: Pose, target: Waypoint, *, previous_target: Optional[Waypoint] = None) -> ApproachProgress:
        phase = (target.phase or "").upper()
        distance_cm = self._distance_cm(pose, target)
        if phase != "APPROACH":
            self.reset()
            return ApproachProgress(ApproachStage.INACTIVE, None, distance_cm, distance_cm, None)

        key = (target.route_id, target.waypoint_id, target.x_mm, target.y_mm, phase)
        if key != self._target_key:
            self._target_key = key
            self._stage = ApproachStage.COARSE
            self._best_distance_cm = None
            self._anchor_mm = (
                (previous_target.x_mm, previous_target.y_mm)
                if previous_target is not None
                else (pose.x_mm, pose.y_mm)
            )

        self._best_distance_cm = (
            distance_cm if self._best_distance_cm is None
            else min(self._best_distance_cm, distance_cm)
        )

        fine_cm = max(0.1, float(target.position_tolerance_cm))
        capture_cm = (
            float(target.capture_tolerance_cm)
            if target.capture_tolerance_cm is not None
            else float(self.config.approach_capture_tolerance_cm)
        )
        capture_cm = max(capture_cm, fine_cm)
        along_cm = self._along_track_cm(pose, target)

        # FINE 반경 안이면 controller ARRIVED 판정을 우선한다.
        if distance_cm <= fine_cm:
            self._stage = ApproachStage.FINE
            return self._result(None, distance_cm, along_cm)

        # 1차 capture 성공: waypoint advance 없이 같은 target을 FINE 추종.
        if self._stage is ApproachStage.COARSE and distance_cm <= capture_cm:
            self._stage = ApproachStage.FINE
            return self._result(ApproachEvent.CAPTURED, distance_cm, along_cm)

        if along_cm is not None and along_cm >= float(self.config.approach_pass_margin_cm):
            event = (
                ApproachEvent.COARSE_MISSED
                if self._stage is ApproachStage.COARSE
                else ApproachEvent.FINE_MISSED
            )
            return self._result(event, distance_cm, along_cm)

        return self._result(None, distance_cm, along_cm)

    def _result(self, event: Optional[ApproachEvent], distance_cm: float, along_cm: Optional[float]) -> ApproachProgress:
        best = distance_cm if self._best_distance_cm is None else self._best_distance_cm
        return ApproachProgress(self._stage, event, distance_cm, float(best), along_cm)

    def _along_track_cm(self, pose: Pose, target: Waypoint) -> Optional[float]:
        if self._anchor_mm is None:
            return None
        ax, ay = self._anchor_mm
        dx, dy = target.x_mm - ax, target.y_mm - ay
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        ux, uy = dx / norm, dy / norm
        return ((pose.x_mm - target.x_mm) * ux + (pose.y_mm - target.y_mm) * uy) / 10.0

    @staticmethod
    def _distance_cm(pose: Pose, target: Waypoint) -> float:
        return math.hypot(target.x_mm - pose.x_mm, target.y_mm - pose.y_mm) / 10.0
