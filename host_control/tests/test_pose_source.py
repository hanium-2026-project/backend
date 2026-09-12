"""CameraPoseSource: 관측 timestamp 보존 테스트 (stale fail-safe 무효화 방지)."""

from __future__ import annotations

import unittest

from controller.models import Pose
from host_control.pose_source import CameraPoseSource


class TestCameraPoseSource(unittest.TestCase):
    def setUp(self) -> None:
        self.src = CameraPoseSource()

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(self.src.latest())
        self.assertFalse(self.src.had_observation)
        self.assertIsNone(self.src.age_s(now=5.0))

    def test_observe_sets_observation_timestamp(self) -> None:
        self.src.observe(100.0, 200.0, 0.0, obs_time=10.0)
        p = self.src.latest()
        assert p is not None
        self.assertEqual(p.timestamp, 10.0)
        self.assertTrue(self.src.had_observation)

    def test_timestamp_not_updated_between_observations(self) -> None:
        # 관측 시각 10.0. 이후 여러 tick(=latest 호출)에도 timestamp 는 그대로 10.0.
        self.src.observe(0.0, 0.0, 0.0, obs_time=10.0)
        for tick_now in (10.1, 10.2, 10.5, 10.9):
            p = self.src.latest()
            assert p is not None
            self.assertEqual(p.timestamp, 10.0,
                             "tick 시각으로 관측 timestamp 가 덮어써지면 안 됨")
        # 경과시간은 now 기준으로 커진다.
        self.assertAlmostEqual(self.src.age_s(now=10.9), 0.9)

    def test_new_observation_updates_timestamp(self) -> None:
        self.src.observe(0.0, 0.0, 0.0, obs_time=10.0)
        self.src.observe(5.0, 5.0, 0.0, obs_time=10.4)  # 새 프레임
        p = self.src.latest()
        assert p is not None
        self.assertEqual(p.timestamp, 10.4)

    def test_observe_pose_passthrough(self) -> None:
        self.src.observe_pose(Pose(1, 2, 3.0, timestamp=7.0))
        p = self.src.latest()
        assert p is not None
        self.assertEqual((p.x_mm, p.y_mm, p.heading_deg, p.timestamp), (1, 2, 3.0, 7.0))

    def test_clear(self) -> None:
        self.src.observe(0, 0, 0.0, obs_time=1.0)
        self.src.clear()
        self.assertIsNone(self.src.latest())
        self.assertFalse(self.src.had_observation)


if __name__ == "__main__":
    unittest.main()
