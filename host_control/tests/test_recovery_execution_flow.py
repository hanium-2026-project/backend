"""AUTO_HOST recovery 실행 계약: STOP → fresh pose → REVERSE → original target 복귀."""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose, Waypoint
from host_control import HostController, HostWaypointMission, MissionStatus


def pose(x, y, heading, t):
    return Pose(float(x), float(y), float(heading), timestamp=float(t))


class TestRecoveryExecutionFlow(unittest.TestCase):
    def test_reverse_recovery_then_original_approach(self) -> None:
        approach = Waypoint(
            30.0, 0.0,
            target_heading_deg=90.0,
            position_tolerance_cm=8.0,
            heading_tolerance_deg=12.0,
            heading_required=True,
            phase="APPROACH",
            is_final=True,
        )
        mission = HostWaypointMission([approach])
        hc = HostController(
            config=ControllerConfig(allow_reverse=True, steer_kd=0.0),
            mission=mission,
        )
        hc.arm_auto()

        r1 = hc.tick(100.0, observation=pose(0, 0, 0, 100.0))
        self.assertIs(r1.mission_status, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(r1.command.throttle, 0.0)

        hc.prepare_route_switch()
        mission.load_recovery([
            Waypoint(
                -500.0, 0.0,
                phase="RECOVERY",
                motion_direction=MotionDirection.REVERSE,
            )
        ])

        r2 = hc.tick(100.1)
        self.assertEqual(r2.command.throttle, 0.0)
        self.assertEqual(r2.command.reason, "NO_POSE")

        r3 = hc.tick(100.2, observation=pose(0, 0, 0, 100.2))
        self.assertLess(r3.command.throttle, 0.0)
        self.assertAlmostEqual(r3.command.steering, 0.0, places=6)
        self.assertTrue(mission.is_recovering)

        r4 = hc.tick(100.3, observation=pose(-500, 0, 0, 100.3))
        self.assertEqual(r4.command.throttle, 0.0)
        self.assertFalse(mission.is_recovering)
        self.assertIs(mission.current_target(), approach)

        # REVERSE recovery -> FORWARD approach 전환에도 zero interlock 필요.
        r5 = hc.tick(100.4, observation=pose(-500, 0, 0, 100.4))
        self.assertEqual(r5.command.throttle, 0.0)
        self.assertEqual(r5.command.reason, "DIRECTION_CHANGE_STOP")

        r6 = hc.tick(100.5, observation=pose(-500, 0, 0, 100.5))
        self.assertGreater(r6.command.throttle, 0.0)

        r7 = hc.tick(100.6, observation=pose(30, 0, 90, 100.6))
        self.assertEqual(r7.command.throttle, 0.0)
        self.assertIs(r7.mission_status, MissionStatus.DONE)


if __name__ == "__main__":
    unittest.main()
