"""geometry.py 단위 테스트."""

from __future__ import annotations

import math
import unittest

from controller import geometry as geo


class TestWrap(unittest.TestCase):
    def test_wrap180_basic(self) -> None:
        self.assertAlmostEqual(geo.wrap180(0.0), 0.0)
        self.assertAlmostEqual(geo.wrap180(90.0), 90.0)
        self.assertAlmostEqual(geo.wrap180(180.0), -180.0)   # [-180,180) 경계는 -180
        self.assertAlmostEqual(geo.wrap180(-180.0), -180.0)  # backend 동일 규약
        self.assertAlmostEqual(geo.wrap180(270.0), -90.0)
        self.assertAlmostEqual(geo.wrap180(-270.0), 90.0)
        self.assertAlmostEqual(geo.wrap180(360.0), 0.0)
        self.assertAlmostEqual(geo.wrap180(450.0), 90.0)

    def test_wrap360(self) -> None:
        self.assertAlmostEqual(geo.wrap360(0.0), 0.0)
        self.assertAlmostEqual(geo.wrap360(360.0), 0.0)
        self.assertAlmostEqual(geo.wrap360(-90.0), 270.0)
        self.assertAlmostEqual(geo.wrap360(450.0), 90.0)


class TestDistance(unittest.TestCase):
    def test_distance(self) -> None:
        self.assertAlmostEqual(geo.distance_mm(0, 0, 3, 4), 5.0)
        self.assertAlmostEqual(geo.distance_mm(0, 0, 0, 0), 0.0)
        self.assertAlmostEqual(geo.distance_mm(1, 1, 1, 1), 0.0)


class TestBearing(unittest.TestCase):
    def test_bearing_cardinals(self) -> None:
        # 0° = +X (오른쪽)
        self.assertAlmostEqual(geo.bearing_deg(0, 0, 10, 0), 0.0)
        # 90° = +Y (위)
        self.assertAlmostEqual(geo.bearing_deg(0, 0, 0, 10), 90.0)
        # 180° = -X (왼쪽)
        self.assertAlmostEqual(abs(geo.bearing_deg(0, 0, -10, 0)), 180.0)
        # -90° = -Y (아래)
        self.assertAlmostEqual(geo.bearing_deg(0, 0, 0, -10), -90.0)

    def test_bearing_same_point(self) -> None:
        self.assertAlmostEqual(geo.bearing_deg(5, 5, 5, 5), 0.0)


class TestHeadingError(unittest.TestCase):
    def test_heading_error_sign(self) -> None:
        # heading 0°(오른쪽), 목표가 +Y(위) → bearing 90° → err +90 (왼쪽/CCW)
        self.assertAlmostEqual(geo.heading_error_deg(90.0, 0.0), 90.0)
        # 목표가 -Y(아래) → bearing -90° → err -90 (오른쪽/CW)
        self.assertAlmostEqual(geo.heading_error_deg(-90.0, 0.0), -90.0)
        # 정면
        self.assertAlmostEqual(geo.heading_error_deg(0.0, 0.0), 0.0)
        # wrap 경계: bearing 170, heading -170 → 340 → wrap → -20
        self.assertAlmostEqual(geo.heading_error_deg(170.0, -170.0), -20.0)


class TestClamp(unittest.TestCase):
    def test_clamp(self) -> None:
        self.assertEqual(geo.clamp(5, 0, 1), 1)
        self.assertEqual(geo.clamp(-5, 0, 1), 0)
        self.assertEqual(geo.clamp(0.5, 0, 1), 0.5)


if __name__ == "__main__":
    unittest.main()
