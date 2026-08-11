"""AUTO_HOST automatic parking E2E state contract test."""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose, Waypoint
from host_control import HostController, HostWaypointMission, MissionStatus


def p(x, y, h, t):
    return Pose(float(x), float(y), float(h), timestamp=float(t))


def route():
    return [
        Waypoint(-300,0,phase="CRUISE",speed_cm_s=8,position_tolerance_cm=8,route_id=1,waypoint_id=1),
        Waypoint(0,0,phase="APPROACH",speed_cm_s=6,capture_tolerance_cm=10,
                 position_tolerance_cm=5,route_id=1,waypoint_id=2),
        Waypoint(100,0,phase="ALIGN",speed_cm_s=5,target_heading_deg=0,
                 heading_required=True,heading_tolerance_deg=12,position_tolerance_cm=5,
                 route_id=1,waypoint_id=3),
        Waypoint(0,0,phase="ENTRY",speed_cm_s=4,target_heading_deg=0,
                 heading_required=True,heading_tolerance_deg=12,position_tolerance_cm=4,
                 motion_direction=MotionDirection.REVERSE,route_id=1,waypoint_id=4),
        Waypoint(-200,0,phase="FINAL",speed_cm_s=4,target_heading_deg=0,
                 heading_required=True,heading_tolerance_deg=12,position_tolerance_cm=5,
                 motion_direction=MotionDirection.REVERSE,route_id=1,waypoint_id=5,is_final=True),
    ]


def make_host():
    mission = HostWaypointMission(route(), max_recovery_attempts=3)
    host = HostController(config=ControllerConfig(allow_reverse=True,steer_kd=0.0,
        final_confirm_observations=3, approach_capture_tolerance_cm=10.0,
        approach_pass_margin_cm=1.0), mission=mission)
    host.arm_auto()
    return host, mission


def finish_from_approach(tc, host, mission, t, *, expect_direction_interlock=False):
    tc.assertEqual(mission.current_phase, "APPROACH")
    r=host.tick(t,observation=p(-90,0,0,t))
    if expect_direction_interlock:
        tc.assertEqual(r.command.throttle, 0.0)
        tc.assertEqual(r.command.reason, "DIRECTION_CHANGE_STOP")
        t += .1
        r=host.tick(t,observation=p(-90,0,0,t))
    tc.assertGreater(r.command.throttle,0); tc.assertEqual(host.approach_guard.stage.value,"FINE")
    t+=.1; r=host.tick(t,observation=p(-40,0,0,t)); tc.assertEqual(r.command.throttle,0); tc.assertEqual(mission.current_phase,"ALIGN")
    t+=.1; tc.assertGreater(host.tick(t,observation=p(-40,0,0,t)).command.throttle,0)
    t+=.1; tc.assertEqual(host.tick(t,observation=p(100,0,0,t)).command.throttle,0); tc.assertEqual(mission.current_phase,"ENTRY")
    t+=.1; r=host.tick(t,observation=p(100,0,0,t)); tc.assertEqual(r.command.reason,"DIRECTION_CHANGE_STOP")
    t+=.1; tc.assertLess(host.tick(t,observation=p(100,0,0,t)).command.throttle,0)
    t+=.1; tc.assertEqual(host.tick(t,observation=p(0,0,0,t)).command.throttle,0); tc.assertEqual(mission.current_phase,"FINAL")
    t+=.1; tc.assertLess(host.tick(t,observation=p(0,0,0,t)).command.throttle,0)
    t+=.1; r=host.tick(t,observation=p(-200,0,0,t)); tc.assertEqual(r.command.reason,"FINAL_CONFIRMING_1_OF_3")
    same=host.tick(t+.03); tc.assertEqual(same.command.reason,"FINAL_CONFIRMING_1_OF_3")
    t+=.1; tc.assertEqual(host.tick(t,observation=p(-200,0,0,t)).command.reason,"FINAL_CONFIRMING_2_OF_3")
    t+=.1; r=host.tick(t,observation=p(-200,0,0,t)); tc.assertTrue(r.command.arrived); tc.assertIs(mission.status,MissionStatus.DONE)
    return t


class TestAutoParkingE2E(unittest.TestCase):
    def test_normal_cycle(self):
        host,mission=make_host()
        self.assertGreater(host.tick(100.0,observation=p(-600,0,0,100.0)).command.throttle,0)
        host.tick(100.1,observation=p(-300,0,0,100.1)); self.assertEqual(mission.current_phase,"APPROACH")
        finish_from_approach(self,host,mission,100.2)

    def test_approach_miss_reverse_recovery_then_success(self):
        host,mission=make_host()
        host.tick(200.0,observation=p(-600,0,0,200.0))
        host.tick(200.1,observation=p(-300,0,0,200.1)); self.assertEqual(mission.current_phase,"APPROACH")
        host.tick(200.2,observation=p(-150,0,0,200.2))
        miss=host.tick(200.3,observation=p(20,140,0,200.3))
        self.assertEqual(miss.command.reason,"APPROACH_COARSE_MISSED")
        self.assertIs(mission.status,MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(mission.replan_reason,"APPROACH_COARSE_MISSED")
        recovery=Waypoint(-200,0,phase="RECOVERY",speed_cm_s=4,position_tolerance_cm=5,
                          motion_direction=MotionDirection.REVERSE,route_id=2,waypoint_id=1)
        host.prepare_route_switch(); mission.load_recovery([recovery])
        no_pose=host.tick(200.4); self.assertEqual(no_pose.command.reason,"NO_POSE")
        self.assertLess(host.tick(200.5,observation=p(20,140,32.5,200.5)).command.throttle,0)
        arrived=host.tick(200.6,observation=p(-200,0,0,200.6)); self.assertEqual(arrived.command.throttle,0)
        self.assertFalse(mission.is_recovering); self.assertEqual(mission.current_phase,"APPROACH")
        finish_from_approach(self,host,mission,200.7,expect_direction_interlock=True)


if __name__ == "__main__":
    unittest.main()
