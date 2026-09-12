from __future__ import annotations

import unittest

from controller.models import ControlCommand, ControlMode, Pose, Waypoint
from host_control.final_pose_guard import FinalPoseGuard


def arrived_cmd():
    return ControlCommand(0.0, 0.0, ControlMode.ARRIVED, True, 2.0, 1.0, 0.0, "ARRIVED")

def driving_cmd():
    return ControlCommand(0.2, 0.0, ControlMode.DRIVE, False, 8.0, 1.0, 0.0, "")

def final_wp():
    return Waypoint(500.0, 500.0, phase="FINAL", route_id=1, waypoint_id=5, is_final=True)


class TestFinalPoseGuard(unittest.TestCase):
    def test_three_distinct_observations_required(self):
        g, wp = FinalPoseGuard(3), final_wp()
        self.assertEqual(g.evaluate(Pose(500,500,90,100.0), wp, arrived_cmd()).count, 1)
        self.assertEqual(g.evaluate(Pose(500,500,90,100.0), wp, arrived_cmd()).count, 1)
        self.assertFalse(g.evaluate(Pose(500,500,90,100.1), wp, arrived_cmd()).confirmed)
        self.assertTrue(g.evaluate(Pose(500,500,90,100.2), wp, arrived_cmd()).confirmed)

    def test_outside_pose_resets(self):
        g, wp = FinalPoseGuard(3), final_wp()
        g.evaluate(Pose(500,500,90,100.0), wp, arrived_cmd())
        g.evaluate(Pose(500,500,90,100.1), wp, arrived_cmd())
        out = g.evaluate(Pose(560,500,90,100.2), wp, driving_cmd())
        self.assertEqual(out.count, 0)
        self.assertFalse(out.confirmed)


if __name__ == "__main__":
    unittest.main()
