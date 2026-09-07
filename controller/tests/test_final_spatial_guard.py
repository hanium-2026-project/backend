"""Rear FINAL observation-gap spatial safety contract regressions."""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose, Waypoint
from controller.pose_controller import PoseWaypointController
from integration.backend_adapter import waypoint_from_backend
from parking.waypoints import (
    build_rear_candidate_waypoints,
    build_rear_entry_waypoints,
    build_rear_parking_waypoints,
    default_slot_specs,
)


def final_b1() -> Waypoint:
    return Waypoint(
        x_mm=425.0,
        y_mm=1050.0,
        target_heading_deg=270.0,
        speed_cm_s=4.0,
        position_tolerance_cm=5.0,
        heading_tolerance_deg=8.0,
        heading_required=True,
        is_final=True,
        phase="FINAL",
        motion_direction=MotionDirection.REVERSE,
        terminal_motion_clearance_mm=25.0,
    )


def pose(y: float, t: float, *, x: float = 425.0) -> Pose:
    return Pose(
        x_mm=x,
        y_mm=y,
        heading_deg=270.0,
        timestamp=t,
        heading_source="FRONT_CUSHION",
    )


class TestFinalSpatialGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ControllerConfig(allow_reverse=True, steer_kd=0.0)
        self.ctl = PoseWaypointController(self.cfg)
        self.wp = final_b1()

    def test_fresh_pose_with_sufficient_margin_keeps_tracking(self) -> None:
        cmd = self.ctl.compute(pose(900.0, 100.0), self.wp, now=100.0)
        self.assertLess(cmd.throttle, 0.0)
        self.assertNotEqual(cmd.reason, "POSE_BLIND_TRAVEL")

    def test_pose_age_is_allowed_while_spatial_budget_fits(self) -> None:
        cmd = self.ctl.compute(pose(900.0, 100.0), self.wp, now=100.2)
        self.assertLess(cmd.throttle, 0.0)

    def test_budget_near_terminal_returns_zero_before_stale_timeout(self) -> None:
        cmd = self.ctl.compute(pose(951.8, 100.0), self.wp, now=100.4)
        self.assertEqual(cmd.reason, "POSE_BLIND_TRAVEL")
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.steering, 0.0)

    def test_154551_measured_speed_stops_on_last_safe_fresh_pose(self) -> None:
        # Recorded observations immediately before the 547 ms gap.
        self.ctl.compute(pose(951.8, 100.000, x=462.6), self.wp,
                         now=100.000)
        cmd = self.ctl.compute(pose(996.9, 100.219, x=463.2), self.wp,
                               now=100.219)
        self.assertGreater(self.ctl._observed_speed_mm_s, 200.0)
        self.assertEqual(cmd.reason, "POSE_BLIND_TRAVEL")
        # Even the recorded 54.5 mm post-zero coast remains inside the map:
        # center y=1051.4, half vehicle length=125 -> edge y=1176.4 < 1200.
        projected_stop_y = 996.9 + 54.5
        self.assertLess(projected_stop_y + 125.0, 1200.0)

    def test_arrival_candidate_bypasses_guard_during_confirmation(self) -> None:
        self.ctl.compute(pose(950.0, 100.0), self.wp, now=100.0)
        cmd = self.ctl.compute(pose(1050.0, 100.1), self.wp, now=100.4)
        self.assertTrue(cmd.arrived)
        self.assertEqual(cmd.reason, "ARRIVED")
        self.assertEqual(cmd.throttle, 0.0)

    def test_unsafe_part_of_generic_50mm_tolerance_is_not_done(self) -> None:
        # Physical clearance is 25mm, and 10mm is reserved for uncertainty.
        # An 18mm overrun is radially "inside 50mm" but outside safe capture.
        cmd = self.ctl.compute(pose(1068.0, 100.0), self.wp, now=100.0)
        self.assertFalse(cmd.arrived)
        self.assertEqual(cmd.reason, "POSE_BLIND_TRAVEL")

    def test_safe_part_of_generic_tolerance_reaches_evaluator(self) -> None:
        cmd = self.ctl.compute(pose(1064.0, 100.0), self.wp, now=100.0)
        self.assertTrue(cmd.arrived)
        self.assertEqual(cmd.reason, "ARRIVED")

    def test_fresh_stationary_observation_does_not_latch_guard(self) -> None:
        self.ctl.compute(pose(900.0, 100.0), self.wp, now=100.0)
        cmd = self.ctl.compute(pose(900.0, 100.2), self.wp, now=100.2)
        self.assertLess(cmd.throttle, 0.0)
        self.assertNotEqual(cmd.reason, "POSE_BLIND_TRAVEL")

    def test_general_final_without_planner_clearance_keeps_old_contract(self) -> None:
        wp = Waypoint(
            x_mm=425.0, y_mm=1050.0, target_heading_deg=270.0,
            speed_cm_s=4.0, position_tolerance_cm=5.0,
            is_final=True, phase="FINAL",
            motion_direction=MotionDirection.REVERSE,
        )
        cmd = self.ctl.compute(pose(950.0, 100.0), wp, now=100.4)
        self.assertLess(cmd.throttle, 0.0)
        self.assertNotEqual(cmd.reason, "POSE_BLIND_TRAVEL")


class TestRearFinalClearanceMetadata(unittest.TestCase):
    def test_all_production_rear_builders_attach_25mm_geometry(self) -> None:
        b1 = default_slot_specs()["B1"]
        routes = (
            build_rear_parking_waypoints(
                b1, 1, from_pose=(150.0, 600.0)),
            build_rear_candidate_waypoints(
                b1, 2, from_pose=(240.0, 625.0),
                from_heading_deg=335.0, radii_mm=(1000.0,)),
            build_rear_entry_waypoints(
                b1, 3, from_pose=(684.9, 357.4),
                from_heading_deg=318.1, min_radius_mm=1000.0),
        )
        for route in routes:
            with self.subTest(route=route[0].route_id):
                final = route[-1]
                self.assertAlmostEqual(final.terminal_motion_clearance_mm, 25.0)
                core = waypoint_from_backend(final)
                self.assertAlmostEqual(core.terminal_motion_clearance_mm, 25.0)
                self.assertNotIn("terminal_motion_clearance_mm", final.to_wire())


if __name__ == "__main__":
    unittest.main()
