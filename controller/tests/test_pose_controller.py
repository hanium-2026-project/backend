"""pose_controller.py 단위 테스트: 부호 / 안전 / throttle / 도착."""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import ControlMode, MotionDirection, Pose, Waypoint
from controller.pose_controller import PoseWaypointController


def make_pose(x_mm=0.0, y_mm=0.0, heading_deg=0.0, t=100.0, valid=True) -> Pose:
    return Pose(x_mm=x_mm, y_mm=y_mm, heading_deg=heading_deg, timestamp=t, valid=valid)


def far_waypoint(x_mm, y_mm, **kw) -> Waypoint:
    kw.setdefault("speed_cm_s", 12.0)
    kw.setdefault("position_tolerance_cm", 8.0)
    return Waypoint(x_mm=x_mm, y_mm=y_mm, **kw)


class TestSteeringSign(unittest.TestCase):
    """실제 ESP32 wire 부호: 음수 = LEFT, 양수 = RIGHT."""

    def setUp(self) -> None:
        self.ctl = PoseWaypointController()

    def test_target_left_gives_negative_wire_steering(self) -> None:
        # heading 0°(오른쪽), 목표 +Y(위) → LEFT 필요 → wire steering < 0
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(0.0, 1000.0)  # 정북(+Y), 100cm
        cmd = self.ctl.compute(pose, wp, now=100.0)
        self.assertLess(cmd.steering, 0.0, "목표 LEFT 인데 wire steering 이 음수가 아님")
        self.assertGreater(cmd.heading_error_deg, 0.0)  # 논리상 왼쪽
        self.assertGreater(cmd.logical_steering, 0.0)    # 논리 steering 양수 = LEFT 요구

    def test_target_right_gives_positive_wire_steering(self) -> None:
        # heading 0°(오른쪽), 목표 -Y(아래) → RIGHT 필요 → wire steering > 0
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(0.0, -1000.0)
        cmd = self.ctl.compute(pose, wp, now=100.0)
        self.assertGreater(cmd.steering, 0.0, "목표 RIGHT 인데 wire steering 이 양수가 아님")
        self.assertLess(cmd.heading_error_deg, 0.0)

    def test_straight_gives_near_zero_steering(self) -> None:
        # 정면(오른쪽으로 100cm), heading 0° → steering ≈ 0
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(1000.0, 0.0)
        cmd = self.ctl.compute(pose, wp, now=100.0)
        self.assertAlmostEqual(cmd.steering, 0.0, places=6)

    def test_steering_clamped(self) -> None:
        # 목표가 거의 정반대(뒤) → 큰 오차 → wire steering 포화(±1) 이내
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(0.0, 30.0)  # 살짝 위, 큰 heading 오차
        cmd = self.ctl.compute(pose, wp, now=100.0)
        self.assertGreaterEqual(cmd.steering, -1.0)
        self.assertLessEqual(cmd.steering, 1.0)

    def test_wire_sign_is_opposite_of_logical(self) -> None:
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(0.0, 1000.0)
        cmd = self.ctl.compute(pose, wp, now=100.0)
        # wire = -1.0 * logical (부호 반대)
        self.assertAlmostEqual(cmd.steering, -cmd.logical_steering, places=6)


class TestSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.ctl = PoseWaypointController()
        self.wp = far_waypoint(1000.0, 0.0)

    def test_invalid_pose_zero(self) -> None:
        pose = make_pose(0, 0, heading_deg=0.0, valid=False)
        cmd = self.ctl.compute(pose, self.wp, now=100.0)
        self._assert_zero(cmd, "POSE_INVALID")

    def test_no_heading_zero(self) -> None:
        pose = Pose(x_mm=0, y_mm=0, heading_deg=None, timestamp=100.0)
        cmd = self.ctl.compute(pose, self.wp, now=100.0)
        self._assert_zero(cmd, "NO_HEADING")

    def test_stale_pose_zero(self) -> None:
        pose = make_pose(0, 0, heading_deg=0.0, t=100.0)
        # now 가 pose.timestamp 보다 0.6s 뒤 → max_pose_age_s(0.5) 초과
        cmd = self.ctl.compute(pose, self.wp, now=100.6)
        self._assert_zero(cmd, "POSE_STALE")

    def test_drive_disabled_zero(self) -> None:
        pose = make_pose(0, 0, heading_deg=0.0)
        cmd = self.ctl.compute(pose, self.wp, allow_drive=False, now=100.0)
        self._assert_zero(cmd, "DRIVE_NOT_ALLOWED")

    def _assert_zero(self, cmd, reason) -> None:
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.steering, 0.0)
        self.assertEqual(cmd.mode, ControlMode.HOLD)
        self.assertFalse(cmd.arrived)
        self.assertEqual(cmd.reason, reason)


class TestThrottle(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ControllerConfig()
        self.ctl = PoseWaypointController(self.cfg)

    def test_far_aligned_greater_than_near(self) -> None:
        pose = make_pose(0, 0, heading_deg=0.0)
        far = far_waypoint(1000.0, 0.0)   # 100cm 정면
        near = far_waypoint(200.0, 0.0)   # 20cm 정면 (slow_radius 25cm 안쪽)
        t_far = self.ctl.compute(pose, far, now=100.0).throttle
        self.ctl.reset()
        t_near = self.ctl.compute(pose, near, now=100.0).throttle
        self.assertGreater(t_far, t_near)

    def test_large_heading_error_slower(self) -> None:
        """회전 감속 — 단, 조향이 포화되기 전 구간에서다.

        최대 조향(|steering|>=0.9)에서는 정지 마찰 하한이 우선한다.
        거기서도 감속을 고집하면 duty 가 38~40 에 묶여 차가 아예 안 움직인다
        (실측 2026-08-12). test_full_lock_overrides_turn_slowdown 참조.
        """
        pose = make_pose(0, 0, heading_deg=0.0)
        aligned = far_waypoint(1000.0, 0.0)       # 정면
        skew = far_waypoint(1000.0, 250.0)        # 약 14° — 조향 0.75, 미포화
        t_aligned = self.ctl.compute(pose, aligned, now=100.0).throttle
        self.ctl.reset()
        t_skew = self.ctl.compute(pose, skew, now=100.0).throttle
        self.assertLess(abs(self.ctl.compute(pose, skew, now=100.0).steering), 0.9)
        self.assertGreater(t_aligned, t_skew)

    def test_full_lock_overrides_turn_slowdown(self) -> None:
        """바퀴를 끝까지 꺾으면 정지 마찰 하한이 회전 감속을 이긴다."""
        pose = make_pose(0, 0, heading_deg=0.0)
        full = far_waypoint(1000.0, 600.0)        # 약 31° — 조향 포화
        cmd = self.ctl.compute(pose, full, now=100.0)
        self.assertGreaterEqual(abs(cmd.steering), 0.9)
        self.assertGreaterEqual(cmd.throttle, 0.7)

    def test_stiction_floor_can_be_disabled(self) -> None:
        ctl = PoseWaypointController(ControllerConfig(strong_turn_min_throttle=None))
        cmd = ctl.compute(make_pose(0, 0, heading_deg=0.0),
                          far_waypoint(1000.0, 600.0), now=100.0)
        self.assertLess(cmd.throttle, 0.7)

    def test_arrival_zero(self) -> None:
        pose = make_pose(0, 0, heading_deg=0.0)
        # 목표까지 5cm(=50mm), tol 8cm → brake_radius 11cm 안 → 도착
        wp = far_waypoint(50.0, 0.0, position_tolerance_cm=8.0)
        cmd = self.ctl.compute(pose, wp, now=100.0)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertTrue(cmd.arrived)
        self.assertEqual(cmd.mode, ControlMode.ARRIVED)

    def test_max_throttle_clamp(self) -> None:
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(1000.0, 0.0, speed_cm_s=100.0)  # 과도한 속도 요구
        cmd = self.ctl.compute(pose, wp, now=100.0)
        self.assertLessEqual(cmd.throttle, self.cfg.max_throttle + 1e-9)

    def test_default_no_reverse(self) -> None:
        # 전진 전용: throttle 은 음수가 될 수 없다.
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(0.0, -1000.0)  # 뒤쪽
        cmd = self.ctl.compute(pose, wp, now=100.0)
        self.assertGreaterEqual(cmd.throttle, 0.0)


class TestAlignment(unittest.TestCase):
    def test_align_when_heading_required(self) -> None:
        ctl = PoseWaypointController()
        pose = make_pose(0, 0, heading_deg=0.0)  # 현재 0°
        # 위치 도착(3cm), heading_required, target 90° → 정렬 필요
        wp = far_waypoint(
            30.0, 0.0, position_tolerance_cm=8.0,
            target_heading_deg=90.0, heading_required=True, heading_tolerance_deg=12.0,
        )
        cmd = ctl.compute(pose, wp, now=100.0)
        self.assertEqual(cmd.mode, ControlMode.ALIGN)
        self.assertFalse(cmd.arrived)
        self.assertEqual(cmd.throttle, 0.0)

    def test_arrived_when_heading_ok(self) -> None:
        ctl = PoseWaypointController()
        pose = make_pose(0, 0, heading_deg=88.0)  # 목표 90°와 2° 차이
        wp = far_waypoint(
            30.0, 0.0, position_tolerance_cm=8.0,
            target_heading_deg=90.0, heading_required=True, heading_tolerance_deg=12.0,
        )
        cmd = ctl.compute(pose, wp, now=100.0)
        self.assertEqual(cmd.mode, ControlMode.ARRIVED)
        self.assertTrue(cmd.arrived)


class TestReverseControl(unittest.TestCase):
    def test_reverse_disabled_by_default(self) -> None:
        ctl = PoseWaypointController()
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(-1000.0, 0.0, phase="RECOVERY",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(pose, wp, now=100.0)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.reason, "REVERSE_NOT_ALLOWED")

    def test_reverse_only_in_allowed_phase(self) -> None:
        ctl = PoseWaypointController(ControllerConfig(allow_reverse=True))
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(-1000.0, 0.0, phase="CRUISE",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(pose, wp, now=100.0)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.reason, "REVERSE_PHASE_NOT_ALLOWED")

    def test_reverse_straight_outputs_negative_throttle(self) -> None:
        ctl = PoseWaypointController(ControllerConfig(allow_reverse=True))
        pose = make_pose(0, 0, heading_deg=0.0)
        wp = far_waypoint(-1000.0, 0.0, phase="RECOVERY",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(pose, wp, now=100.0)
        self.assertLess(cmd.throttle, 0.0)
        self.assertAlmostEqual(cmd.steering, 0.0, places=6)
        self.assertEqual(cmd.mode, ControlMode.DRIVE)

    def test_reverse_steering_sign_is_physically_reversed(self) -> None:
        # 부호 수학 자체를 보는 테스트다. 실차 기본값은 11자 후진
        # (reverse_straight_steering=True)이라 조향이 0 으로 고정되므로,
        # 여기서만 꺼서 계산식을 검증한다.
        ctl = PoseWaypointController(ControllerConfig(
            allow_reverse=True, steer_kd=0.0, reverse_straight_steering=False))
        pose = make_pose(0, 0, heading_deg=0.0)
        # 후진 기준 NW(135°)는 reverse 진행방향 180°의 오른쪽.
        # 후진 물리에서는 바퀴를 LEFT로 틀어야 rear trajectory가 그쪽으로 간다.
        wp = far_waypoint(-1000.0, 1000.0, phase="RECOVERY",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(pose, wp, now=100.0)
        self.assertLess(cmd.steering, 0.0)  # wire 음수 = LEFT
        self.assertGreater(cmd.logical_steering, 0.0)

    def test_reverse_keeps_wheels_straight_by_default(self) -> None:
        """실차 기본값은 11자 후진 — 뒤를 못 보는 상태에서 궤적을 휘게 하지 않는다."""
        ctl = PoseWaypointController(ControllerConfig(allow_reverse=True))
        wp = far_waypoint(-1000.0, 1000.0, phase="RECOVERY",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(make_pose(0, 0, heading_deg=0.0), wp, now=100.0)
        self.assertEqual(cmd.steering, 0.0)
        self.assertLess(cmd.throttle, 0.0)          # 후진은 하고 있다


if __name__ == "__main__":
    unittest.main()
