from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools.run_recorder import _execution_gate_reason


class TestExecutionGateReason(unittest.TestCase):
    def session(self, **overrides):
        values = {"alive": True, "comm_failed": False, "control_held": False}
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_running_zero_behind_comm_latch_is_never_silent(self):
        reason = _execution_gate_reason(
            self.session(control_held=True), "ARMED", None, "RUNNING",
            0.0, 0.0, None)
        self.assertEqual(reason, "COMM_ZERO_LATCH")

    def test_controller_and_firmware_zero_are_distinguished(self):
        self.assertEqual(_execution_gate_reason(
            self.session(), "ARMED", None, "RUNNING", 0.0, 0.0,
            "DIRECTION_CHANGE_INTERLOCK"), "DIRECTION_CHANGE_INTERLOCK")
        self.assertEqual(_execution_gate_reason(
            self.session(), "ARMED", None, "RUNNING", 0.2, 0.0, None),
            "ESP_APPLIED_ZERO")

    def test_route_done_is_a_route_gate_not_parked(self):
        self.assertEqual(_execution_gate_reason(
            self.session(), "ARMED", None, "DONE", 0.0, 0.0, None),
            "MISSION_DONE")


if __name__ == "__main__":
    unittest.main()
