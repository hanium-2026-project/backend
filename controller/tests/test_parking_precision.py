from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose, Waypoint
from controller.pose_controller import PoseWaypointController


def pose(x=0, y=0, h=0, t=100.0):
    return Pose(float(x), float(y), float(h), timestamp=t)


class TestParkingPrecision(unittest.TestCase):
    def test_cruise_keeps_legacy_arrival_radius(self):
        ctl = PoseWaypointController()
        wp = Waypoint(90.0, 0.0, phase="CRUISE", position_tolerance_cm=8.0)
        self.assertTrue(ctl.compute(pose(), wp, now=100.0).arrived)

    def test_final_uses_exact_position_tolerance(self):
        ctl = PoseWaypointController()
        wp = Waypoint(50.0, 0.0, phase="FINAL", position_tolerance_cm=4.0,
                      target_heading_deg=0.0, heading_required=True,
                      heading_tolerance_deg=12.0, is_final=True)
        out = ctl.compute(pose(), wp, now=100.0)
        self.assertFalse(out.arrived)
        self.assertGreater(out.throttle, 0.0)

    def test_forward_to_reverse_after_arrival_has_zero_interlock(self):
        ctl = PoseWaypointController(ControllerConfig(allow_reverse=True, steer_kd=0.0))
        fwd = Waypoint(1000.0, 0.0, phase="APPROACH", position_tolerance_cm=5.0)
        self.assertGreater(ctl.compute(pose(), fwd, now=100.0).throttle, 0.0)
        self.assertTrue(ctl.compute(pose(1000,0,0,100.1), fwd, now=100.1).arrived)
        rev = Waypoint(500.0, 0.0, phase="ENTRY", motion_direction=MotionDirection.REVERSE)
        stop = ctl.compute(pose(1000,0,0,100.2), rev, now=100.2)
        self.assertEqual(stop.throttle, 0.0)
        self.assertEqual(stop.reason, "DIRECTION_CHANGE_STOP")
        self.assertLess(ctl.compute(pose(1000,0,0,100.3), rev, now=100.3).throttle, 0.0)

    def test_precision_speed_not_forced_to_general_floor(self):
        cfg = ControllerConfig(allow_reverse=True, steer_kd=0.0,
                               min_move_throttle=0.22,
                               parking_min_move_throttle=0.08,
                               reverse_min_move_throttle=0.10,
                               parking_max_throttle=0.25,
                               reverse_max_throttle=0.25)
        ctl = PoseWaypointController(cfg)
        final = Waypoint(1000, 0, phase="FINAL", speed_cm_s=4.0, position_tolerance_cm=5.0)
        self.assertAlmostEqual(ctl.compute(pose(), final, now=100.0).throttle, 0.12, places=4)
        ctl.reset()
        cruise = Waypoint(1000, 0, phase="CRUISE", speed_cm_s=4.0, position_tolerance_cm=5.0)
        self.assertAlmostEqual(ctl.compute(pose(), cruise, now=100.1).throttle, 0.22, places=4)
        ctl.reset()
        rev = Waypoint(-1000,0,phase="RECOVERY",speed_cm_s=2.0,
                       motion_direction=MotionDirection.REVERSE)
        self.assertAlmostEqual(ctl.compute(pose(), rev, now=100.2).throttle, -0.10, places=4)


if __name__ == "__main__":
    unittest.main()
