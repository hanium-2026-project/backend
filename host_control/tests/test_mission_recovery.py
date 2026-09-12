"""HostWaypointMission recovery 흐름 단위 테스트."""

from __future__ import annotations

import unittest

from controller.models import ControlCommand, ControlMode, Waypoint
from host_control.mission import HostWaypointMission, MissionStatus


def arrived_cmd():
    return ControlCommand(0.0, 0.0, ControlMode.ARRIVED, True, 2.0, 0.0, 0.0, "ARRIVED")


def align_cmd():
    return ControlCommand(0.0, 0.0, ControlMode.ALIGN, False, 2.0, 30.0, 0.0,
                          "HEADING_OUT_OF_TOLERANCE")


class TestMissionRecovery(unittest.TestCase):
    def _mission_at_replan(self, *, max_attempts=3):
        original = [
            Waypoint(100.0, 0.0, phase="APPROACH"),
            Waypoint(200.0, 0.0, phase="ENTRY"),
            Waypoint(300.0, 0.0, phase="FINAL", is_final=True),
        ]
        m = HostWaypointMission(original, max_recovery_attempts=max_attempts)
        m.notify_result(align_cmd())
        self.assertIs(m.status, MissionStatus.REPLAN_REQUIRED)
        return m, original

    def test_recovery_load_returns_to_running(self):
        m, _ = self._mission_at_replan()
        rec = [Waypoint(0.0, -100.0, phase="RECOVERY")]
        status = m.load_recovery(rec)
        self.assertIs(status, MissionStatus.RUNNING)
        self.assertTrue(m.is_recovering)
        self.assertEqual(m.recovery_attempts, 1)
        self.assertEqual(m.current_target().phase, "RECOVERY")
        self.assertEqual(m.current_phase, "RECOVERY")
        self.assertFalse(m.parking_active)

    def test_recovery_then_resumes_failed_original_target(self):
        m, original = self._mission_at_replan()
        rec = [
            Waypoint(0.0, -100.0, phase="RECOVERY"),
            Waypoint(50.0, -50.0, phase="RECOVERY"),
        ]
        m.load_recovery(rec)
        m.notify_result(arrived_cmd())
        self.assertTrue(m.is_recovering)
        m.notify_result(arrived_cmd())
        self.assertFalse(m.is_recovering)
        self.assertIs(m.current_target(), original[0])
        self.assertEqual(m.current_target().phase, "APPROACH")

    def test_original_route_continues_to_done_after_recovery(self):
        m, _ = self._mission_at_replan()
        m.load_recovery([Waypoint(0.0, -100.0, phase="RECOVERY")])
        for _ in range(4):
            m.notify_result(arrived_cmd())
        self.assertIs(m.status, MissionStatus.DONE)

    def test_recovery_only_allowed_after_replan(self):
        m = HostWaypointMission([Waypoint(100.0, 0.0, is_final=True)])
        with self.assertRaises(RuntimeError):
            m.load_recovery([Waypoint(0.0, -100.0, phase="RECOVERY")])

    def test_empty_recovery_rejected(self):
        m, _ = self._mission_at_replan()
        with self.assertRaises(ValueError):
            m.load_recovery([])

    def test_recovery_final_flag_rejected(self):
        m, _ = self._mission_at_replan()
        with self.assertRaises(ValueError):
            m.load_recovery([Waypoint(0.0, -100.0, phase="RECOVERY", is_final=True)])

    def test_recovery_attempt_limit_latches_failure(self):
        m, _ = self._mission_at_replan(max_attempts=1)
        m.load_recovery([Waypoint(0.0, -100.0, phase="RECOVERY")])
        m.notify_result(align_cmd())
        self.assertIs(m.status, MissionStatus.REPLAN_REQUIRED)
        status = m.load_recovery([Waypoint(-50.0, -100.0, phase="RECOVERY")])
        self.assertIs(status, MissionStatus.RECOVERY_FAILED)
        self.assertIsNone(m.current_target())
        self.assertFalse(m.is_active)

    def test_failed_recovery_keeps_original_resume_target(self):
        m, original = self._mission_at_replan(max_attempts=3)
        first_recovery = Waypoint(0.0, -100.0, phase="RECOVERY")
        m.load_recovery([first_recovery])

        m.notify_result(align_cmd())
        self.assertIs(m.status, MissionStatus.REPLAN_REQUIRED)
        second_recovery = Waypoint(-50.0, -100.0, phase="RECOVERY")
        m.load_recovery([second_recovery])
        m.notify_result(arrived_cmd())

        self.assertFalse(m.is_recovering)
        self.assertIs(m.current_target(), original[0])
        self.assertIsNot(m.current_target(), first_recovery)

    def test_primary_load_resets_recovery_counter(self):
        m, _ = self._mission_at_replan()
        m.load_recovery([Waypoint(0.0, -100.0, phase="RECOVERY")])
        self.assertEqual(m.recovery_attempts, 1)
        m.load([Waypoint(500.0, 500.0, is_final=True)])
        self.assertEqual(m.recovery_attempts, 0)
        self.assertFalse(m.is_recovering)
        self.assertIs(m.status, MissionStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
