"""B안 주행 제어기 검증.

실차 없이 확인할 수 있는 것: 조향 부호, 감속 프로파일, 안전 게이트,
그리고 "목표를 향해 실제로 수렴하는가" (간단한 자전거 모델 시뮬레이션).

실행: python -m unittest control.tests.test_waypoint_controller -v
"""

from __future__ import annotations

import math
import unittest

from control import Pose, VehicleLimits, WaypointController, wrap180
from parking.waypoints import Waypoint


def wp(x: float, y: float, *, heading: float | None = None,
       tol_cm: float = 6.0, speed: float = 8.0,
       heading_required: bool = False) -> Waypoint:
    return Waypoint(
        route_id=1, waypoint_id=1, phase="CRUISE", x=x, y=y,
        target_heading_deg=heading, speed_cm_s=speed,
        position_tolerance_cm=tol_cm, heading_tolerance_deg=20.0,
        heading_required=heading_required, is_final=False,
    )


def pose(x: float, y: float, heading: float | None, t: float = 0.0,
         valid: bool = True) -> Pose:
    return Pose(x, y, heading, timestamp=t, valid=valid)


class TestSteeringSign(unittest.TestCase):
    """heading 은 반시계 양수(우 0°, 상 90°). 좌회전이 양수여야 한다."""

    def setUp(self) -> None:
        self.c = WaypointController()

    def test_target_to_the_left_steers_positive(self):
        # 우측(0°)을 보고 있는데 목표는 위쪽 → 좌회전
        out = self.c.compute(pose(0, 0, 0.0), wp(1000.0, 1000.0), now=0.0)
        self.assertGreater(out.steering, 0.0)
        self.assertAlmostEqual(out.heading_error_deg, 45.0, places=3)

    def test_target_to_the_right_steers_negative(self):
        out = self.c.compute(pose(0, 1000, 0.0), wp(1000.0, 0.0), now=0.0)
        self.assertLess(out.steering, 0.0)
        self.assertAlmostEqual(out.heading_error_deg, -45.0, places=3)

    def test_straight_ahead_no_steering(self):
        out = self.c.compute(pose(0, 0, 90.0), wp(0.0, 1000.0), now=0.0)
        self.assertAlmostEqual(out.steering, 0.0, places=6)

    def test_steering_sign_can_be_inverted(self):
        c = WaypointController(VehicleLimits(steering_sign=-1.0))
        out = c.compute(pose(0, 0, 0.0), wp(1000.0, 1000.0), now=0.0)
        self.assertLess(out.steering, 0.0)

    def test_steering_saturates_at_one(self):
        out = self.c.compute(pose(0, 0, 180.0), wp(1000.0, 100.0), now=0.0)
        self.assertLessEqual(abs(out.steering), 1.0)
        self.assertAlmostEqual(abs(out.steering), 1.0, places=6)


class TestThrottleProfile(unittest.TestCase):
    def setUp(self) -> None:
        self.c = WaypointController()

    def test_slows_down_near_target(self):
        far = self.c.compute(pose(0, 0, 90.0), wp(0.0, 3000.0), now=0.0).throttle
        self.c.reset()
        near = self.c.compute(pose(0, 0, 90.0), wp(0.0, 150.0), now=0.0).throttle
        self.assertGreater(far, near, "접근 감속이 없다")

    def test_never_below_min_throttle_while_driving(self):
        lim = self.c.limits
        out = self.c.compute(pose(0, 0, 90.0), wp(0.0, 1200.0), now=0.0)
        self.assertGreaterEqual(out.throttle, lim.min_throttle,
                                "데드밴드 미만이면 소리만 나고 안 움직인다")

    def test_respects_max_throttle(self):
        out = self.c.compute(pose(0, 0, 90.0), wp(0.0, 10000.0, speed=100.0),
                             now=0.0)
        self.assertLessEqual(out.throttle, self.c.limits.max_throttle)

    def test_slows_when_badly_misaligned(self):
        aligned = self.c.compute(pose(0, 0, 90.0), wp(0.0, 3000.0), now=0.0).throttle
        self.c.reset()
        skewed = self.c.compute(pose(0, 0, 0.0), wp(0.0, 3000.0), now=0.0).throttle
        self.assertLess(skewed, aligned, "크게 틀어졌는데 속도를 안 줄인다")

    def test_stops_inside_brake_radius(self):
        target = wp(0.0, 1000.0, tol_cm=6.0)
        # 제동 거리(3cm) + 허용오차(6cm) = 9cm 안쪽
        out = self.c.compute(pose(0.0, 920.0, 90.0), target, now=0.0)
        self.assertEqual(out.mode, "ARRIVED")
        self.assertEqual(out.throttle, 0.0)

    def test_brakes_earlier_with_larger_stop_distance(self):
        target = wp(0.0, 1000.0, tol_cm=6.0)
        far_braking = WaypointController(VehicleLimits(stop_distance_cm=15.0))
        out = far_braking.compute(pose(0.0, 800.0, 90.0), target, now=0.0)
        self.assertEqual(out.mode, "ARRIVED", "긴 제동거리를 반영하지 않는다")


class TestSafetyGates(unittest.TestCase):
    def setUp(self) -> None:
        self.c = WaypointController()
        self.target = wp(0.0, 2000.0)

    def test_no_heading_means_no_drive(self):
        out = self.c.compute(pose(0, 0, None), self.target, now=0.0)
        self.assertEqual((out.throttle, out.steering), (0.0, 0.0))
        self.assertEqual(out.reason, "NO_HEADING")

    def test_invalid_pose_means_no_drive(self):
        out = self.c.compute(pose(0, 0, 90.0, valid=False), self.target, now=0.0)
        self.assertEqual(out.reason, "POSE_INVALID")
        self.assertEqual(out.throttle, 0.0)

    def test_stale_pose_means_no_drive(self):
        out = self.c.compute(pose(0, 0, 90.0, t=0.0), self.target, now=5.0)
        self.assertEqual(out.reason, "POSE_STALE")
        self.assertEqual(out.throttle, 0.0)

    def test_allow_drive_false_still_reports_error(self):
        out = self.c.compute(pose(0, 0, 0.0), self.target,
                             allow_drive=False, now=0.0)
        self.assertEqual(out.throttle, 0.0)
        self.assertEqual(out.mode, "HOLD")
        self.assertAlmostEqual(out.heading_error_deg, 90.0, places=3)

    def test_alignment_required_but_off_stops_instead_of_creeping(self):
        target = wp(0.0, 1000.0, heading=90.0, heading_required=True)
        out = self.c.compute(pose(0.0, 960.0, 0.0), target, now=0.0)
        self.assertEqual(out.mode, "ALIGN")
        self.assertEqual(out.throttle, 0.0, "정렬 안 된 채 밀어 넣으면 안 된다")


class TestConvergence(unittest.TestCase):
    """자전거 모델로 실제 수렴을 확인한다.

    제어기가 부호만 맞고 발산하면 실차에서 그대로 사고가 된다. 완벽한
    모델은 아니지만 부호 오류·과도한 게인은 여기서 잡힌다.
    """

    WHEELBASE_MM = 165.0
    SPEED_PER_THROTTLE = 25.0        # throttle 1.0 → 25cm/s 라고 가정

    def simulate(self, start: tuple[float, float, float], target: Waypoint,
                 steps: int = 400, dt: float = 0.1,
                 limits: VehicleLimits | None = None):
        c = WaypointController(limits)
        x, y, heading = start
        t = 0.0
        for _ in range(steps):
            out = c.compute(pose(x, y, heading, t=t), target, now=t)
            if out.mode in ("ARRIVED", "ALIGN"):
                return (x, y, heading), out
            speed_mm_s = out.throttle * self.SPEED_PER_THROTTLE * 10.0
            steer_rad = math.radians(out.steering * c.limits.max_steer_deg)
            heading = (heading + math.degrees(
                speed_mm_s / self.WHEELBASE_MM * math.tan(steer_rad) * dt)) % 360.0
            x += speed_mm_s * math.cos(math.radians(heading)) * dt
            y += speed_mm_s * math.sin(math.radians(heading)) * dt
            t += dt
        return (x, y, heading), out

    def _assert_reaches(self, start, target, label):
        (x, y, _), out = self.simulate(start, target)
        dist_cm = math.hypot(target.x - x, target.y - y) / 10.0
        self.assertEqual(out.mode, "ARRIVED",
                         f"{label}: 미도달 (거리 {dist_cm:.1f}cm)")
        self.assertLessEqual(
            dist_cm, target.position_tolerance_cm + out.distance_cm + 1.0,
            f"{label}: 허용오차 밖 ({dist_cm:.1f}cm)")

    def test_straight_ahead(self):
        self._assert_reaches((150.0, 100.0, 90.0), wp(150.0, 900.0), "직진")

    def test_needs_left_turn(self):
        self._assert_reaches((150.0, 100.0, 90.0), wp(900.0, 700.0), "우선회")

    def test_needs_right_turn(self):
        self._assert_reaches((900.0, 100.0, 90.0), wp(150.0, 700.0), "좌선회")

    def test_target_behind(self):
        """뒤에 있는 목표도 선회해서 돌아간다 (후진은 쓰지 않는다)."""
        self._assert_reaches((600.0, 900.0, 90.0), wp(600.0, 200.0), "후방 목표")

    def test_does_not_oscillate_around_the_line(self):
        """직선 추종 중 조향이 좌우로 널뛰지 않아야 한다."""
        c = WaypointController()
        target = wp(600.0, 1100.0)
        x, y, heading, t = 600.0, 100.0, 90.0, 0.0
        signs = []
        for _ in range(120):
            out = c.compute(pose(x, y, heading, t=t), target, now=t)
            if out.mode == "ARRIVED":
                break
            signs.append(0 if abs(out.steering) < 1e-3 else
                         (1 if out.steering > 0 else -1))
            speed_mm_s = out.throttle * self.SPEED_PER_THROTTLE * 10.0
            steer_rad = math.radians(out.steering * c.limits.max_steer_deg)
            heading = (heading + math.degrees(
                speed_mm_s / self.WHEELBASE_MM * math.tan(steer_rad) * 0.1)) % 360.0
            x += speed_mm_s * math.cos(math.radians(heading)) * 0.1
            y += speed_mm_s * math.sin(math.radians(heading)) * 0.1
            t += 0.1
        flips = sum(1 for a, b in zip(signs, signs[1:])
                    if a != 0 and b != 0 and a != b)
        self.assertLessEqual(flips, 3, f"조향 진동 {flips}회")


class TestWrap180(unittest.TestCase):
    def test_wraps(self):
        self.assertAlmostEqual(wrap180(370.0), 10.0)
        self.assertAlmostEqual(wrap180(-190.0), 170.0)
        self.assertAlmostEqual(wrap180(180.0), -180.0)


if __name__ == "__main__":
    unittest.main()
