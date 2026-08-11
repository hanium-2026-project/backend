"""HostWaypointMission 단위 테스트 (wire 미개입, host-owned 진행)."""

from __future__ import annotations

import unittest

from controller.models import ControlCommand, ControlMode, Waypoint
from host_control.mission import HostWaypointMission, MissionStatus


def arrived_cmd():
    return ControlCommand(0.0, 0.0, ControlMode.ARRIVED, True, 2.0, 0.0, 0.0, "ARRIVED")


def align_cmd():
    return ControlCommand(0.0, 0.0, ControlMode.ALIGN, False, 2.0, 30.0, 0.0,
                          "HEADING_OUT_OF_TOLERANCE")


def driving_cmd():
    return ControlCommand(0.3, -0.2, ControlMode.DRIVE, False, 40.0, 5.0, 0.0)


class TestMission(unittest.TestCase):
    def test_empty_mission(self) -> None:
        m = HostWaypointMission([])
        self.assertIs(m.status, MissionStatus.EMPTY)
        self.assertIsNone(m.current_target())

    def test_single_waypoint_arrival_done(self) -> None:
        m = HostWaypointMission([Waypoint(500, 0, is_final=True)])
        self.assertIs(m.status, MissionStatus.RUNNING)
        m.notify_result(arrived_cmd())
        self.assertIs(m.status, MissionStatus.DONE)

    def test_multiple_waypoint_progression(self) -> None:
        m = HostWaypointMission([
            Waypoint(400, 50), Waypoint(800, 150), Waypoint(1100, 150, is_final=True),
        ])
        self.assertEqual(m.index, 0)
        m.notify_result(arrived_cmd())
        self.assertEqual(m.index, 1)
        self.assertIs(m.status, MissionStatus.RUNNING)
        m.notify_result(arrived_cmd())
        self.assertEqual(m.index, 2)
        m.notify_result(arrived_cmd())
        self.assertIs(m.status, MissionStatus.DONE)

    def test_driving_does_not_advance(self) -> None:
        m = HostWaypointMission([Waypoint(500, 0), Waypoint(900, 0, is_final=True)])
        m.notify_result(driving_cmd())
        self.assertEqual(m.index, 0)
        self.assertIs(m.status, MissionStatus.RUNNING)

    def test_align_requests_replan(self) -> None:
        m = HostWaypointMission([Waypoint(500, 0, heading_required=True,
                                          target_heading_deg=90.0, is_final=True)])
        m.notify_result(align_cmd())
        self.assertIs(m.status, MissionStatus.REPLAN_REQUIRED)
        # replan 요청 상태에서는 current_target None (상위가 load 로 재접근 경로 제공)
        self.assertIsNone(m.current_target())

    def test_last_index_without_is_final_still_done(self) -> None:
        m = HostWaypointMission([Waypoint(500, 0), Waypoint(900, 0)])  # is_final 미표기
        m.notify_result(arrived_cmd())  # index 1 (마지막)
        m.notify_result(arrived_cmd())
        self.assertIs(m.status, MissionStatus.DONE)

    def test_confirm_parked(self) -> None:
        m = HostWaypointMission([Waypoint(500, 0, is_final=True)])
        m.notify_result(arrived_cmd())
        m.confirm_parked()
        self.assertIs(m.status, MissionStatus.PARKED)

    def test_reload_resets(self) -> None:
        m = HostWaypointMission([Waypoint(500, 0, is_final=True)])
        m.notify_result(arrived_cmd())
        self.assertIs(m.status, MissionStatus.DONE)
        m.load([Waypoint(100, 100), Waypoint(200, 200, is_final=True)])
        self.assertIs(m.status, MissionStatus.RUNNING)
        self.assertEqual(m.index, 0)


if __name__ == "__main__":
    unittest.main()
