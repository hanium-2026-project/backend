"""Low-cost planner/controller intent contract fuzz (deterministic)."""

from __future__ import annotations

import math
import random
import unittest

from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose, Waypoint
from controller.pose_controller import PoseWaypointController


class TestControlIntentProperties(unittest.TestCase):
    def test_400_representative_intents_never_invert(self) -> None:
        rng = random.Random(20260814)
        cfg = ControllerConfig(allow_reverse=True, steer_kd=0.0)

        for index in range(200):
            reverse = bool(index % 2)
            body_curvature = rng.choice((-1.0, 1.0)) / rng.choice(
                (800.0, 900.0, 1000.0, 1100.0))
            phase = rng.choice(("ENTRY", "PARKING") if reverse
                               else ("ALIGN", "RECOVERY"))
            direction = (MotionDirection.REVERSE if reverse
                         else MotionDirection.FORWARD)
            end_body_heading = rng.uniform(0.0, 360.0)
            end_motion_heading = end_body_heading + (180.0 if reverse else 0.0)
            motion_curvature = -body_curvature if reverse else body_curvature
            radius = 1.0 / abs(motion_curvature)
            end_rad = math.radians(end_motion_heading)
            center_x = -math.sin(end_rad) / motion_curvature
            center_y = math.cos(end_rad) / motion_curvature
            remaining = rng.uniform(5.0, 25.0)
            current_motion = end_motion_heading - math.copysign(
                remaining, motion_curvature)
            radial_heading = current_motion - math.copysign(
                90.0, motion_curvature)
            radial = radius + rng.uniform(-30.0, 30.0)
            x = center_x + radial * math.cos(math.radians(radial_heading))
            y = center_y + radial * math.sin(math.radians(radial_heading))
            body_heading = current_motion - (180.0 if reverse else 0.0)
            body_heading += rng.uniform(-5.0, 5.0)
            waypoint = Waypoint(
                0.0, 0.0, target_heading_deg=end_body_heading,
                speed_cm_s=5.0, position_tolerance_cm=1.0,
                heading_required=False, phase=phase,
                motion_direction=direction, curvature=body_curvature,
            )
            cmd = PoseWaypointController(cfg).compute(
                Pose(x, y, body_heading, timestamp=1.0), waypoint, now=1.0)
            self.assertEqual(math.copysign(1.0, cmd.logical_steering),
                             math.copysign(1.0, body_curvature))
            self.assertAlmostEqual(cmd.steering, -cmd.logical_steering, places=4)
            self.assertLess(cmd.throttle, 0.0) if reverse else \
                self.assertGreater(cmd.throttle, 0.0)

        for index in range(200):
            reverse = bool(index % 2)
            phase = rng.choice(("APPROACH", "ALIGN", "ENTRY", "FINAL"))
            direction = (MotionDirection.REVERSE if reverse
                         else MotionDirection.FORWARD)
            motion_heading = rng.uniform(0.0, 360.0)
            bearing = motion_heading + rng.uniform(-5.0, 5.0)
            distance = rng.uniform(150.0, 500.0)
            waypoint = Waypoint(
                distance * math.cos(math.radians(bearing)),
                distance * math.sin(math.radians(bearing)),
                speed_cm_s=5.0, position_tolerance_cm=1.0,
                heading_required=False, phase=phase,
                motion_direction=direction, curvature=0.0,
            )
            body_heading = motion_heading - (180.0 if reverse else 0.0)
            cmd = PoseWaypointController(cfg).compute(
                Pose(0.0, 0.0, body_heading, timestamp=1.0), waypoint, now=1.0)
            self.assertLess(abs(cmd.logical_steering), 0.5)
            self.assertAlmostEqual(cmd.steering, -cmd.logical_steering, places=4)
            self.assertLess(cmd.throttle, 0.0) if reverse else \
                self.assertGreater(cmd.throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
