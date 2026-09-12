from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import Pose, Waypoint
from host_control.approach_guard import ApproachEvent, ApproachProgressGuard, ApproachStage


def pose(x, y, t=100.0):
    return Pose(float(x), float(y), 0.0, timestamp=t)


def approach():
    return Waypoint(0.0, 0.0, phase="APPROACH", position_tolerance_cm=5.0,
                    capture_tolerance_cm=10.0, route_id=1, waypoint_id=2)


class TestApproachGuard(unittest.TestCase):
    def test_enter_10cm_switches_coarse_to_fine(self):
        g = ApproachProgressGuard(ControllerConfig())
        wp, prev = approach(), Waypoint(-300.0, 0.0, phase="CRUISE")
        self.assertIs(g.evaluate(pose(-200, 0), wp, previous_target=prev).stage, ApproachStage.COARSE)
        out = g.evaluate(pose(-90, 0), wp, previous_target=prev)
        self.assertIs(out.event, ApproachEvent.CAPTURED)
        self.assertIs(out.stage, ApproachStage.FINE)

    def test_coarse_miss_after_pass_line(self):
        g = ApproachProgressGuard(ControllerConfig(approach_pass_margin_cm=1.0))
        wp, prev = approach(), Waypoint(-300.0, 0.0, phase="CRUISE")
        g.evaluate(pose(-200, 0), wp, previous_target=prev)
        g.evaluate(pose(-135, 0), wp, previous_target=prev)
        out = g.evaluate(pose(20, 140), wp, previous_target=prev)
        self.assertIs(out.event, ApproachEvent.COARSE_MISSED)
        self.assertTrue(out.missed)

    def test_fine_miss_after_capture(self):
        g = ApproachProgressGuard(ControllerConfig(approach_pass_margin_cm=1.0))
        wp, prev = approach(), Waypoint(-300.0, 0.0, phase="CRUISE")
        g.evaluate(pose(-200, 0), wp, previous_target=prev)
        self.assertIs(g.evaluate(pose(-90, 0), wp, previous_target=prev).event, ApproachEvent.CAPTURED)
        self.assertIs(g.evaluate(pose(20, 60), wp, previous_target=prev).event, ApproachEvent.FINE_MISSED)

    def test_inside_fine_radius_never_reports_miss(self):
        g = ApproachProgressGuard(ControllerConfig(approach_pass_margin_cm=0.0))
        wp, prev = approach(), Waypoint(-300.0, 0.0, phase="CRUISE")
        g.evaluate(pose(-90, 0), wp, previous_target=prev)
        out = g.evaluate(pose(20, 30), wp, previous_target=prev)
        self.assertIsNone(out.event)
        self.assertIs(out.stage, ApproachStage.FINE)


if __name__ == "__main__":
    unittest.main()
