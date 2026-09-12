"""FINAL waypoint 다중 fresh-camera 관측 확인 gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from controller.models import ControlCommand, Pose, Waypoint


@dataclass(frozen=True)
class FinalPoseProgress:
    count: int
    required: int
    confirmed: bool


class FinalPoseGuard:
    def __init__(self, required_observations: int = 3) -> None:
        self.required = max(1, int(required_observations))
        self.reset()

    def reset(self) -> None:
        self._target_key = None
        self._count = 0
        self._last_pose_timestamp: Optional[float] = None

    @property
    def count(self) -> int:
        return self._count

    def evaluate(self, pose: Optional[Pose], target: Waypoint, cmd: ControlCommand) -> FinalPoseProgress:
        if (target.phase or "").upper() != "FINAL":
            self.reset()
            return FinalPoseProgress(0, self.required, False)

        key = (target.route_id, target.waypoint_id, target.x_mm, target.y_mm, target.target_heading_deg)
        if key != self._target_key:
            self._target_key = key
            self._count = 0
            self._last_pose_timestamp = None

        if not cmd.arrived or pose is None or not pose.valid:
            self._count = 0
            self._last_pose_timestamp = None
            return FinalPoseProgress(0, self.required, False)

        if pose.timestamp != self._last_pose_timestamp:
            self._last_pose_timestamp = pose.timestamp
            self._count += 1

        return FinalPoseProgress(self._count, self.required, self._count >= self.required)
