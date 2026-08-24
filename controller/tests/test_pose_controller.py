"""pose_controller.py 단위 테스트: 부호 / 안전 / throttle / 도착."""

from __future__ import annotations

import math
import unittest

from controller.config import ControllerConfig, curvature_for_steering
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


class TestArcGuidance(unittest.TestCase):
    """Curvature feedforward follows the planned tangent, not endpoint chord."""

    def setUp(self) -> None:
        self.cfg = ControllerConfig(allow_reverse=True, steer_kd=0.0)

    def test_forward_right_arc_on_circle_preserves_feedforward(self) -> None:
        radius = 800.0
        sweep = math.radians(12.0)
        wp = far_waypoint(
            radius * math.sin(sweep),
            -radius * (1.0 - math.cos(sweep)),
            phase="ALIGN", curvature=-1.0 / radius,
            target_heading_deg=348.0, heading_required=False,
        )
        cmd = PoseWaypointController(self.cfg).compute(
            make_pose(0.0, 0.0, 0.0), wp, now=100.0)
        ff = self.cfg.feedforward_steering("ALIGN", -1.0 / radius,
                                          reverse=False)
        self.assertAlmostEqual(cmd.heading_error_deg, 0.0, places=6)
        self.assertAlmostEqual(cmd.logical_steering, ff, places=4)

    def test_reverse_arc_on_circle_preserves_feedforward_and_sign(self) -> None:
        radius = 800.0
        sweep = math.radians(12.0)
        wp = far_waypoint(
            -radius * math.sin(sweep),
            radius * (1.0 - math.cos(sweep)),
            phase="ENTRY", curvature=1.0 / radius,
            target_heading_deg=348.0, heading_required=False,
            motion_direction=MotionDirection.REVERSE,
        )
        cmd = PoseWaypointController(self.cfg).compute(
            make_pose(0.0, 0.0, 0.0), wp, now=100.0)
        ff = self.cfg.feedforward_steering("ENTRY", 1.0 / radius,
                                          reverse=True)
        self.assertAlmostEqual(cmd.heading_error_deg, 0.0, places=6)
        self.assertAlmostEqual(cmd.logical_steering, ff, places=4)
        self.assertLess(cmd.throttle, 0.0)

    def test_174030_passed_reverse_arc_sample_advances_in_corridor(self) -> None:
        """Real route-4 wp4: the car missed the point sphere by 2.9 mm.

        It crossed the endpoint tangent only 46 mm off the planned 800 mm
        circle, so this intermediate sample must advance instead of being
        chased backwards.
        """
        wp = far_waypoint(
            612.0, 536.0, phase="ENTRY", target_heading_deg=310.0,
            heading_required=False, position_tolerance_cm=4.0,
            curvature=1.0 / 800.0, path_capture_tolerance_cm=10.0,
            motion_direction=MotionDirection.REVERSE,
        )
        ctl = PoseWaypointController(self.cfg)
        before = ctl.compute(make_pose(584.4, 502.9, 304.7), wp, now=100.0)
        crossed = ctl.compute(make_pose(560.7, 525.8, 301.5), wp, now=100.1)
        self.assertFalse(before.arrived)
        self.assertTrue(crossed.arrived)
        self.assertEqual(crossed.reason, "ARC_ENDPOINT_PASSED")
        self.assertEqual(crossed.throttle, 0.0)

    def test_passed_arc_sample_outside_corridor_is_not_captured(self) -> None:
        wp = far_waypoint(
            612.0, 536.0, phase="ENTRY", target_heading_deg=310.0,
            heading_required=False, position_tolerance_cm=4.0,
            curvature=1.0 / 800.0, path_capture_tolerance_cm=10.0,
            motion_direction=MotionDirection.REVERSE,
        )
        cmd = PoseWaypointController(self.cfg).compute(
            make_pose(560.7, 425.8, 301.5), wp, now=100.0)
        self.assertFalse(cmd.arrived)

    def test_final_never_uses_arc_pass_capture(self) -> None:
        wp = far_waypoint(
            612.0, 536.0, phase="FINAL", target_heading_deg=310.0,
            heading_required=False, is_final=True, position_tolerance_cm=4.0,
            curvature=1.0 / 800.0, path_capture_tolerance_cm=10.0,
            motion_direction=MotionDirection.REVERSE,
        )
        cmd = PoseWaypointController(self.cfg).compute(
            make_pose(560.7, 525.8, 301.5), wp, now=100.0)
        self.assertFalse(cmd.arrived)

    def test_144749_late_arc_pose_no_longer_aims_back_at_endpoint(self) -> None:
        wp = far_waypoint(825.0, 357.1797, phase="ALIGN",
                          curvature=-1.0 / 800.0,
                          target_heading_deg=330.0, heading_required=True)
        cmd = PoseWaypointController(self.cfg).compute(
            make_pose(988.9, 168.7, 313.3), wp, now=100.0)
        ff = self.cfg.feedforward_steering("ALIGN", wp.curvature,
                                          reverse=False)
        self.assertAlmostEqual(ff, -0.7, places=4)
        self.assertGreater(cmd.logical_steering, -0.7)
        self.assertLess(cmd.logical_steering, -0.55)

    def test_144902_late_arc_pose_reduces_endpoint_cancellation(self) -> None:
        wp = far_waypoint(747.1825, 272.1825, phase="ALIGN",
                          curvature=-1.0 / 1100.0,
                          target_heading_deg=315.0, heading_required=True,
                          position_tolerance_cm=4.0,
                          heading_tolerance_deg=5.0)
        cmd = PoseWaypointController(self.cfg).compute(
            make_pose(691.5, 229.9, 316.4), wp, now=100.0)
        ff = self.cfg.feedforward_steering("ALIGN", wp.curvature,
                                          reverse=False)
        old_final = ff + self.cfg.curvature_feedback_limit
        self.assertLess(abs(cmd.logical_steering - ff), abs(old_final - ff))
        self.assertLess(cmd.logical_steering, -0.35)

    def test_actual_route_arcs_hold_planned_radius_closed_loop(self) -> None:
        cases = (
            ((677.4419, 423.4859, 341.6),
             ((825.0, 357.1797, 330.0),), -1.0 / 800.0),
            ((393.6578, 509.2423, 337.3),
             ((581.9959, 407.9752, 326.15),
              (747.1825, 272.1825, 315.0)), -1.0 / 1100.0),
        )
        for start, targets, curvature in cases:
            with self.subTest(radius=round(1.0 / abs(curvature))):
                ctl = PoseWaypointController(self.cfg)
                x, y, heading = start
                start_heading = heading
                total = 0.0
                ticks = 0
                end_x, end_y, end_heading = targets[-1]
                end_rad = math.radians(end_heading)
                center_x = end_x - math.sin(end_rad) / curvature
                center_y = end_y + math.cos(end_rad) / curvature
                radius = 1.0 / abs(curvature)
                max_cross_track = 0.0
                for target_x, target_y, target_heading in targets:
                    arrived = False
                    for _ in range(200):
                        now = ticks * 0.1
                        ticks += 1
                        waypoint = far_waypoint(
                            target_x, target_y, phase="ALIGN",
                            curvature=curvature,
                            target_heading_deg=target_heading,
                            heading_required=False,
                            position_tolerance_cm=4.0,
                            speed_cm_s=5.0,
                        )
                        cmd = ctl.compute(
                            make_pose(x, y, heading, t=now), waypoint, now=now)
                        if cmd.arrived:
                            arrived = True
                            break
                        step_mm = 5.0
                        actual_k = curvature_for_steering(
                            cmd.logical_steering, reverse=False)
                        mid_heading = heading + math.degrees(actual_k * step_mm) / 2.0
                        x += step_mm * math.cos(math.radians(mid_heading))
                        y += step_mm * math.sin(math.radians(mid_heading))
                        heading = (heading + math.degrees(actual_k * step_mm)) % 360.0
                        total += step_mm
                        cross_track = abs(
                            math.hypot(x - center_x, y - center_y) - radius)
                        max_cross_track = max(max_cross_track, cross_track)
                    self.assertTrue(arrived)
                heading_change = abs(
                    (heading - start_heading + 180.0) % 360.0 - 180.0)
                effective_radius = total / math.radians(heading_change)
                self.assertLess(max_cross_track, 2.0)
                self.assertLess(abs(effective_radius - radius), radius * 0.02)


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
        """복구 후진은 11자 — 뒤를 못 보는 상태에서 궤적을 휘게 하지 않는다."""
        ctl = PoseWaypointController(ControllerConfig(allow_reverse=True))
        wp = far_waypoint(-1000.0, 1000.0, phase="RECOVERY",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(make_pose(0, 0, heading_deg=0.0), wp, now=100.0)
        self.assertEqual(cmd.steering, 0.0)
        self.assertLess(cmd.throttle, 0.0)          # 후진은 하고 있다


class TestReverseSteeringPhaseSplit(unittest.TestCase):
    """후진 조향 고정은 phase 별로 갈린다 (후면주차 ENTRY/FINAL 은 조향해야 한다).

    같은 기하(후진 목표가 진행방향 기준 한쪽으로 벌어진 자리)에 phase 만 바꿔
    넣고, RECOVERY 만 0 이 나오는지 본다.
    """

    CFG = ControllerConfig(allow_reverse=True, steer_kd=0.0)

    def _steer(self, phase: str) -> float:
        ctl = PoseWaypointController(self.CFG)
        wp = far_waypoint(-1000.0, 1000.0, phase=phase,
                          motion_direction=MotionDirection.REVERSE)
        return ctl.compute(make_pose(0, 0, heading_deg=0.0), wp, now=100.0).steering

    def test_recovery_reverse_is_locked_straight(self) -> None:
        self.assertEqual(self._steer("RECOVERY"), 0.0)

    def test_entry_reverse_allows_steering(self) -> None:
        steer = self._steer("ENTRY")
        self.assertNotEqual(steer, 0.0,
                            "ENTRY 후진이 11자로 묶이면 후면주차 원호를 못 탄다")

    def test_final_reverse_allows_steering(self) -> None:
        steer = self._steer("FINAL")
        self.assertNotEqual(steer, 0.0,
                            "FINAL 후진이 11자로 묶이면 슬롯 축 정렬을 못 한다")

    def test_entry_and_final_match_the_free_steering_math(self) -> None:
        """조향이 허용된 phase 는 잠금을 끈 것과 값이 같아야 한다 (경로만 다름)."""
        free = PoseWaypointController(ControllerConfig(
            allow_reverse=True, steer_kd=0.0, reverse_straight_steering=False))
        wp = far_waypoint(-1000.0, 1000.0, phase="ENTRY",
                          motion_direction=MotionDirection.REVERSE)
        expected = free.compute(make_pose(0, 0, heading_deg=0.0), wp, now=100.0).steering
        self.assertAlmostEqual(self._steer("ENTRY"), expected, places=9)
        self.assertAlmostEqual(self._steer("FINAL"), expected, places=9)

    def test_reverse_steering_keeps_physical_sign_in_entry(self) -> None:
        """ENTRY 후진에서도 후진 부호 반전이 유지된다 (11자 해제가 부호를 안 바꾼다)."""
        ctl = PoseWaypointController(self.CFG)
        # 후진 진행방향(180°) 기준 오른쪽인 NW 목표 → 바퀴는 LEFT(wire 음수)
        wp = far_waypoint(-1000.0, 1000.0, phase="ENTRY",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(make_pose(0, 0, heading_deg=0.0), wp, now=100.0)
        self.assertLess(cmd.steering, 0.0)              # wire 음수 = LEFT
        self.assertGreater(cmd.logical_steering, 0.0)
        self.assertLess(cmd.throttle, 0.0)              # 후진 유지

    def test_master_switch_off_frees_every_phase(self) -> None:
        ctl = PoseWaypointController(ControllerConfig(
            allow_reverse=True, steer_kd=0.0, reverse_straight_steering=False))
        wp = far_waypoint(-1000.0, 1000.0, phase="RECOVERY",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(make_pose(0, 0, heading_deg=0.0), wp, now=100.0)
        self.assertNotEqual(cmd.steering, 0.0)

    def test_forward_phases_are_untouched_by_the_gate(self) -> None:
        """전진에는 이 게이트가 걸리지 않는다 (RECOVERY 전진도 조향한다)."""
        ctl = PoseWaypointController(self.CFG)
        wp = far_waypoint(0.0, 1000.0, phase="RECOVERY")   # FORWARD 기본값
        self.assertNotEqual(ctl.compute(make_pose(0, 0, heading_deg=0.0),
                                        wp, now=100.0).steering, 0.0)


class TestReverseGatesAreIndependent(unittest.TestCase):
    """'후진해도 되는가'(reverse_allowed_phases)와 '후진 중 조향해도 되는가'
    (reverse_straight_phases)는 서로 다른 게이트다."""

    def test_default_config_separates_the_two_lists(self) -> None:
        cfg = ControllerConfig()
        # ENTRY/FINAL: 후진 허용 + 조향 허용
        for phase in ("ENTRY", "FINAL"):
            self.assertIn(phase, cfg.reverse_allowed_phases)
            self.assertFalse(cfg.reverse_steering_locked(phase))
        # RECOVERY: 후진 허용 + 조향 잠금
        self.assertIn("RECOVERY", cfg.reverse_allowed_phases)
        self.assertTrue(cfg.reverse_steering_locked("RECOVERY"))
        # CRUISE: 애초에 후진 금지
        self.assertNotIn("CRUISE", cfg.reverse_allowed_phases)

    def test_phase_name_is_case_insensitive(self) -> None:
        cfg = ControllerConfig()
        self.assertTrue(cfg.reverse_steering_locked("recovery"))
        self.assertFalse(cfg.reverse_steering_locked("entry"))

    def test_missing_phase_is_not_locked(self) -> None:
        """phase 가 없는 waypoint 를 11자로 묶지 않는다 (후면주차 기본이 조향)."""
        cfg = ControllerConfig()
        self.assertFalse(cfg.reverse_steering_locked(None))
        self.assertFalse(cfg.reverse_steering_locked(""))

    def test_reverse_phase_gate_still_blocks_cruise(self) -> None:
        """조향 분리가 후진 허용 게이트를 느슨하게 만들지 않았다."""
        ctl = PoseWaypointController(ControllerConfig(allow_reverse=True))
        wp = far_waypoint(-1000.0, 1000.0, phase="CRUISE",
                          motion_direction=MotionDirection.REVERSE)
        cmd = ctl.compute(make_pose(0, 0, heading_deg=0.0), wp, now=100.0)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.reason, "REVERSE_PHASE_NOT_ALLOWED")


if __name__ == "__main__":
    unittest.main()
