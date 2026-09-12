"""producer 테스트: manual/auto 모두 wire-ready(음수=LEFT) 출력."""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import ControlMode, Pose, Waypoint
from host_control.producers import (
    AutoControlProducer,
    ManualControlProducer,
    ManualInput,
)


class TestManualProducer(unittest.TestCase):
    def setUp(self) -> None:
        self.p = ManualControlProducer()

    def test_none_input_zero(self) -> None:
        cmd = self.p.compute(None)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.steering, 0.0)

    def test_logical_left_becomes_wire_negative(self) -> None:
        # 사람 입력 논리 +1 = LEFT → wire 음수(ESP32 LEFT)
        cmd = self.p.compute(ManualInput(throttle=0.3, steering=1.0))
        self.assertLess(cmd.steering, 0.0)
        self.assertGreater(cmd.logical_steering, 0.0)

    def test_logical_right_becomes_wire_positive(self) -> None:
        cmd = self.p.compute(ManualInput(throttle=0.3, steering=-1.0))
        self.assertGreater(cmd.steering, 0.0)

    def test_no_reverse_by_default(self) -> None:
        cmd = self.p.compute(ManualInput(throttle=-0.5, steering=0.0))
        self.assertGreaterEqual(cmd.throttle, 0.0)

    def test_throttle_clamped_to_max(self) -> None:
        cfg = ControllerConfig()
        cmd = self.p.compute(ManualInput(throttle=1.0, steering=0.0))
        self.assertLessEqual(cmd.throttle, cfg.max_throttle + 1e-9)


class TestAutoProducer(unittest.TestCase):
    def setUp(self) -> None:
        self.p = AutoControlProducer()

    def test_no_target_zero(self) -> None:
        cmd = self.p.compute(Pose(0, 0, 0.0, 1.0), None, now=1.0)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.reason, "NO_TARGET")

    def test_no_pose_zero(self) -> None:
        cmd = self.p.compute(None, Waypoint(500, 0), now=1.0)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.reason, "NO_POSE")

    def test_target_left_wire_negative(self) -> None:
        cmd = self.p.compute(Pose(0, 0, 0.0, 1.0), Waypoint(0, 1000), now=1.0)
        self.assertLess(cmd.steering, 0.0)  # 목표 +Y = LEFT

    def test_target_right_wire_positive(self) -> None:
        cmd = self.p.compute(Pose(0, 0, 0.0, 1.0), Waypoint(0, -1000), now=1.0)
        self.assertGreater(cmd.steering, 0.0)


if __name__ == "__main__":
    unittest.main()
