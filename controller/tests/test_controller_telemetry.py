from __future__ import annotations

import math
import unittest

from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose, Waypoint
from controller.pose_controller import PoseWaypointController


class TestControllerTelemetry(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ControllerConfig(
            allow_reverse=True,
            parking_max_throttle=0.25,
            reverse_max_throttle=0.25,
        )
        self.controller = PoseWaypointController(self.config)

    def test_drive_command_exposes_exact_pd_and_throttle_stages(self) -> None:
        pose = Pose(100.0, 100.0, 0.0, timestamp=10.0,
                    heading_source="FRONT_CUSHION")
        target = Waypoint(
            300.0, 200.0, target_heading_deg=20.0, speed_cm_s=8.0,
            phase="APPROACH", motion_direction=MotionDirection.FORWARD)

        command = self.controller.compute(pose, target, now=10.0)
        telemetry = command.telemetry

        self.assertAlmostEqual(telemetry["dx_to_target_mm"], 200.0)
        self.assertAlmostEqual(telemetry["dy_to_target_mm"], 100.0)
        self.assertAlmostEqual(
            telemetry["distance_to_target_mm"], math.hypot(200.0, 100.0))
        self.assertAlmostEqual(
            telemetry["steering_raw"],
            telemetry["steering_proportional_term"]
            + telemetry["steering_derivative_term"])
        self.assertEqual(
            telemetry["steering_command_final"], command.steering)
        self.assertEqual(
            telemetry["throttle_command_final"], command.throttle)
        self.assertIn("throttle_requested_raw", telemetry)
        self.assertIn("throttle_after_safety_limit", telemetry)
        self.assertIn("cross_track_error_mm", telemetry)

    def test_second_tick_records_previous_error_derivative_and_dt(self) -> None:
        target = Waypoint(300.0, 200.0, phase="APPROACH")
        first = self.controller.compute(
            Pose(100.0, 100.0, 0.0, timestamp=10.0), target, now=10.0)
        second = self.controller.compute(
            Pose(110.0, 100.0, 5.0, timestamp=10.1), target, now=10.1)

        self.assertAlmostEqual(
            second.telemetry["steering_error_previous_rad"],
            math.radians(first.heading_error_deg))
        self.assertAlmostEqual(second.telemetry["controller_dt_s"], 0.1)
        self.assertIn("steering_error_derivative_rad_s", second.telemetry)

    def test_stale_zero_carries_override_reason_without_changing_command(self) -> None:
        command = self.controller.compute(
            Pose(100.0, 100.0, 0.0, timestamp=1.0),
            Waypoint(300.0, 100.0), now=2.0)

        self.assertTrue(command.is_stopped)
        self.assertEqual(command.reason, "POSE_STALE")
        self.assertEqual(
            command.telemetry["control_override_reason"], "POSE_STALE")
        self.assertEqual(command.telemetry["steering_command_final"], 0.0)
        self.assertEqual(command.telemetry["throttle_command_final"], 0.0)

    def test_parking_phases_expose_the_same_tuning_contract(self) -> None:
        required = {
            "distance_to_target_mm", "bearing_error_deg",
            "cross_track_error_mm", "steering_proportional_term",
            "steering_derivative_term", "steering_raw",
            "steering_after_controller_clamp", "steering_after_phase_cap",
            "steering_after_wire_sign", "throttle_requested_raw",
            "throttle_after_stage_limit", "throttle_after_parking_limit",
            "throttle_after_safety_limit", "throttle_command_final",
            "arrival_position_ok", "arrival_heading_ok",
        }
        for phase in ("APPROACH", "ALIGN", "ENTRY", "FINAL"):
            with self.subTest(phase=phase):
                controller = PoseWaypointController(self.config)
                command = controller.compute(
                    Pose(100.0, 100.0, 0.0, timestamp=10.0,
                         heading_source="FRONT_CUSHION"),
                    Waypoint(
                        350.0, 200.0, target_heading_deg=10.0,
                        heading_required=True, phase=phase,
                        motion_direction=MotionDirection.FORWARD,
                    ),
                    now=10.0,
                )
                self.assertTrue(required.issubset(command.telemetry))


if __name__ == "__main__":
    unittest.main()
