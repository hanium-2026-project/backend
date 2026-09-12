"""DirectControlSender / TransportTiming 테스트."""

from __future__ import annotations

import unittest

from controller.models import ControlCommand, ControlMode
from host_control.direct_control import DirectControlSender, TransportTiming


def cmd(throttle, steering):
    return ControlCommand(
        throttle=throttle, steering=steering, mode=ControlMode.DRIVE, arrived=False,
        distance_error_cm=0.0, heading_error_deg=0.0, target_bearing_deg=0.0,
    )


class TestSender(unittest.TestCase):
    def setUp(self) -> None:
        self.s = DirectControlSender()

    def test_type_is_direct_control(self) -> None:
        p = self.s.send_command(cmd(0.3, -0.5))
        self.assertEqual(p["type"], "DIRECT_CONTROL")

    def test_seq_monotonic(self) -> None:
        seqs = []
        for _ in range(5):
            seqs.append(self.s.send_command(cmd(0.2, 0.0))["control_seq"])
        self.assertEqual(seqs, [1, 2, 3, 4, 5])

    def test_zero_also_increments_seq_and_is_zero(self) -> None:
        self.s.send_command(cmd(0.3, 0.1))
        p = self.s.send_zero()
        self.assertEqual(p["control_seq"], 2)
        self.assertEqual(p["throttle"], 0.0)
        self.assertEqual(p["steering"], 0.0)

    def test_steering_passed_through_wire(self) -> None:
        p = self.s.send_command(cmd(0.3, -0.7))  # 이미 wire(음수=LEFT)
        self.assertEqual(p["steering"], -0.7)

    def test_custom_sink_receives_payloads(self) -> None:
        got = []
        s = DirectControlSender(sink=got.append)
        s.send_command(cmd(0.2, 0.0))
        s.send_zero()
        self.assertEqual(len(got), 2)
        self.assertTrue(all(g["type"] == "DIRECT_CONTROL" for g in got))


class TestTiming(unittest.TestCase):
    def test_default_contract_is_safe(self) -> None:
        # 100ms 송신 vs 500ms firmware timeout → 안전(여유 5x)
        t = TransportTiming()
        self.assertEqual(t.send_period_s, 0.100)
        self.assertEqual(t.direct_timeout_s, 0.500)
        self.assertTrue(t.is_safe)
        self.assertGreaterEqual(t.margin_ratio(), 2.0)

    def test_too_slow_send_is_unsafe(self) -> None:
        t = TransportTiming(send_period_s=0.4, direct_timeout_s=0.5)
        self.assertFalse(t.is_safe)  # 여유 2x 미만


if __name__ == "__main__":
    unittest.main()
