"""Regressions from the 2026-08-27/28 nominal rear-parking runs."""

from __future__ import annotations

import math
import unittest

from controller.config import ControllerConfig, curvature_for_steering
from controller.models import MotionDirection, Pose, Waypoint
from host_control import HostController
from host_control.mission import HostWaypointMission, MissionStatus
from integration.backend_adapter import waypoint_from_backend
from parking.final_alignment import footprint_overflow_mm, to_slot_local
from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (build_rear_candidate_waypoints,
                               build_rear_entry_waypoints,
                               build_rear_parking_waypoints,
                               default_slot_specs,
                               plan_rear_entry_from_pose)


B1 = default_slot_specs()["B1"]


class TestRearPlannerTerminalGeometry(unittest.TestCase):
    def test_nominal_final_is_slot_center_and_footprint_safe(self) -> None:
        routes = (
            build_rear_parking_waypoints(
                B1, 1, from_pose=(150.0, 600.0)),
            build_rear_candidate_waypoints(
                B1, 2, from_pose=(240.0, 625.0),
                from_heading_deg=335.0, radii_mm=(1000.0,)),
        )
        for route in routes:
            with self.subTest(route=route[0].route_id):
                final = route[-1]
                local = to_slot_local(B1, final.x, final.y,
                                      final.target_heading_deg)
                self.assertAlmostEqual(local.depth_mm, 0.0)
                self.assertAlmostEqual(local.lateral_mm, 0.0)
                self.assertAlmostEqual(local.heading_err_deg, 0.0)
                self.assertEqual(
                    footprint_overflow_mm(
                        B1, final.x, final.y, final.target_heading_deg),
                    (0.0, 0.0),
                )

    def test_actual_start_route_is_safe_and_curvature_is_physical(self) -> None:
        route = build_rear_candidate_waypoints(
            B1, 3, from_pose=(240.0, 625.0),
            from_heading_deg=335.0, radii_mm=(1000.0,))
        result = validate_trajectory(
            route, start_pose=(240.0, 625.0, 335.0), target_slot="B1")
        self.assertTrue(result.safe, result.reason)
        self.assertTrue(all(
            abs(w.curvature) <= 1.0 / 610.0 + 1e-12 for w in route))

    def test_last_entry_cannot_use_general_100mm_sample_corridor(self) -> None:
        routes = [
            build_rear_parking_waypoints(
                B1, 1, from_pose=(150.0, 600.0)),
            build_rear_candidate_waypoints(
                B1, 2, from_pose=(240.0, 625.0),
                from_heading_deg=335.0, radii_mm=(1000.0,)),
            build_rear_entry_waypoints(
                B1, 3, from_pose=(684.9, 357.4),
                from_heading_deg=318.1, min_radius_mm=1000.0),
        ]
        for route in routes:
            entries = [w for w in route if w.phase == "ENTRY"]
            with self.subTest(route=route[0].route_id):
                self.assertGreater(len(entries), 0)
                self.assertEqual(
                    entries[-1].path_capture_tolerance_cm,
                    entries[-1].position_tolerance_cm,
                )


class TestActualRunEntryBoundary(unittest.TestCase):
    def test_203434_replan_metric_is_fresh_min_radius_geometry(self) -> None:
        """The logged 262/278 mm values are not stale route-6 state.

        They are reproducible from each fresh observation against the B1
        minimum-radius rear-entry circle.  The loaded R=1000 ALIGN route is a
        different geometry and must be allowed to finish before replanning.
        """
        failed = plan_rear_entry_from_pose(
            B1, (890.5, 190.1), 339.6)
        stopped = plan_rear_entry_from_pose(
            B1, (936.2, 167.8), 330.5)
        self.assertFalse(failed.feasible)
        self.assertFalse(stopped.feasible)
        self.assertAlmostEqual(failed.offset_mm, 261.96, delta=0.1)
        self.assertAlmostEqual(stopped.offset_mm, 277.72, delta=0.1)
        self.assertNotAlmostEqual(failed.offset_mm, stopped.offset_mm,
                                  delta=1.0)

    def test_203434_route6_reaches_reverse_entry_after_align_capture(self) -> None:
        """Replay the real failure/stop poses through the mission boundary."""
        route = build_rear_candidate_waypoints(
            B1, 6, from_pose=(173.0, 408.0),
            from_heading_deg=348.1, radii_mm=(1000.0,))
        host = HostController(
            mission=HostWaypointMission([
                waypoint_from_backend(route[2]),
                waypoint_from_backend(route[3]),
            ]),
            config=ControllerConfig(allow_reverse=True),
        )
        host.arm_auto()

        approaching = host.tick(
            100.0,
            observation=Pose(
                890.5, 190.1, 339.6, timestamp=100.0,
                heading_source="FRONT_CUSHION"),
        )
        self.assertIs(approaching.mission_status, MissionStatus.RUNNING)
        self.assertGreater(approaching.command.throttle, 0.0)
        self.assertNotEqual(approaching.command.reason,
                            "HEADING_OUT_OF_TOLERANCE")

        captured = host.tick(
            100.1,
            observation=Pose(
                936.2, 167.8, 330.5, timestamp=100.1,
                heading_source="FRONT_CUSHION"),
        )
        self.assertEqual(captured.command.reason, "ARRIVED")
        self.assertEqual(captured.command.throttle, 0.0)

        direction_stop = host.tick(
            100.2,
            observation=Pose(
                936.2, 167.8, 330.5, timestamp=100.2,
                heading_source="FRONT_CUSHION"),
        )
        self.assertEqual(direction_stop.command.reason,
                         "DIRECTION_CHANGE_STOP")
        self.assertEqual(direction_stop.command.throttle, 0.0)

        reverse_entry = host.tick(
            100.3,
            observation=Pose(
                936.2, 167.8, 330.5, timestamp=100.3,
                heading_source="FRONT_CUSHION"),
        )
        self.assertLess(reverse_entry.command.throttle, 0.0)

    def test_three_actual_runs_stop_large_residual_before_final(self) -> None:
        # Last ENTRY target and first FINAL pose from each recorded route.
        fixtures = (
            ("000315", 437.3116594, 893.5655350, 279.0, 0.001,
             362.6, 886.7, 273.5),
            ("234439", 438.5428253, 877.9220885, 279.0, 1.0 / 1100.0,
             362.6, 889.4, 272.3),
            ("234231", 437.2, 911.1, 280.0, 1.0 / 800.0,
             357.3, 901.8, 275.8),
        )
        for name, tx, ty, th, curvature, x, y, heading in fixtures:
            with self.subTest(run=name):
                target = Waypoint(
                    tx, ty, target_heading_deg=th,
                    speed_cm_s=5.0, position_tolerance_cm=4.0,
                    route_id=3, waypoint_id=7, phase="ENTRY",
                    motion_direction=MotionDirection.REVERSE,
                    curvature=curvature, path_capture_tolerance_cm=4.0,
                )
                host = HostController(
                    mission=HostWaypointMission([target]),
                    config=ControllerConfig(allow_reverse=True))
                host.arm_auto()
                result = host.tick(
                    100.0,
                    observation=Pose(
                        x, y, heading, timestamp=100.0,
                        heading_source="FRONT_CUSHION"),
                )
                self.assertEqual(result.command.reason,
                                 "ARC_CORRIDOR_MISSED")
                self.assertEqual(result.command.throttle, 0.0)
                self.assertEqual(result.command.steering, 0.0)
                self.assertIs(result.mission_status,
                              MissionStatus.REPLAN_REQUIRED)

    def test_primary_body_heading_and_motion_guidance_remain_distinct(self) -> None:
        # A curved target that remains ahead throughout this run_000315 slice.
        target = Waypoint(
            425.0, 1050.0, target_heading_deg=270.0,
            speed_cm_s=5.0, position_tolerance_cm=4.0,
            route_id=3, waypoint_id=99, phase="ENTRY",
            motion_direction=MotionDirection.REVERSE,
            curvature=0.001,
        )
        host = HostController(
            mission=HostWaypointMission([target]),
            config=ControllerConfig(allow_reverse=True),
        )
        host.arm_auto()
        samples = (
            (24.250, 567.2, 466.0, 297.0),
            (24.594, 542.0, 500.0, 295.4),
            (24.906, 520.0, 530.0, 294.2),
            (25.250, 500.9, 560.4, 292.9),
            (25.578, 476.0, 602.0, 290.8),
        )
        last = None
        for t, x, y, heading in samples:
            last = host.tick(
                t,
                observation=Pose(
                    x, y, heading, timestamp=t,
                    heading_source="FRONT_CUSHION"),
            )
        self.assertIsNotNone(last)
        self.assertEqual(
            host.reverse_observation_state,
            "REVERSE_TRACK_PRIMARY_MOTION_GUIDANCE",
        )
        # The primary body heading remains the sign/safety reference.
        self.assertAlmostEqual(host._last_trusted_reverse_heading, 290.8)
        self.assertLess(last.command.throttle, 0.0)


class TestRearMotionGuidanceClosedLoop(unittest.TestCase):
    """Deterministic pose-point model using the production curvature table."""

    @staticmethod
    def simulate(*, motion_offset_deg: float = 8.0,
                 guidance: bool = True,
                 final_reverse_cap: bool = False,
                 start_delta=(0.0, 0.0, 0.0)):
        # 이 점-모델 sim 은 motion_heading_deg 를 주지 않고(=crab 을 guidance 가
        # 모름) 정지거리/PWM 도 모델링하지 않는다. 그래서 종점 근처 point-bearing
        # 포화(→1.0)로만 crab 을 잡아 수렴한다. 실차는 그 포화가 PWM 을 밀어올려
        # 과주행(boundary)을 낸다. FINAL 조향 cap(Phase 1)은 그 포화를 깎으므로
        # 이 단일-shot 수렴 계약과는 맞지 않는다 — 기본은 legacy(cap off)로 두고,
        # cap 동작은 별도 테스트가 검증한다.
        route = build_rear_candidate_waypoints(
            B1, 10, from_pose=(240.0, 625.0),
            from_heading_deg=335.0, radii_mm=(1000.0,))
        targets = [waypoint_from_backend(w) for w in route
                   if w.phase in ("ENTRY", "FINAL")]
        reverse_start = [w for w in route if w.phase == "ALIGN"][-1]
        x = reverse_start.x + start_delta[0]
        y = reverse_start.y + start_delta[1]
        heading = reverse_start.target_heading_deg + start_delta[2]
        config = ControllerConfig(
            allow_reverse=True,
            reverse_trajectory_min_observations=(3 if guidance else 9999),
            final_reverse_straight_when_aligned=final_reverse_cap,
        )
        host = HostController(
            mission=HostWaypointMission(targets), config=config)
        host.arm_auto()

        dt = 0.1
        result = None
        for step in range(1000):
            now = 100.0 + step * dt
            result = host.tick(
                now,
                observation=Pose(
                    x, y, heading, timestamp=now,
                    heading_source="FRONT_CUSHION"),
            )
            if result.mission_status in {
                    MissionStatus.DONE, MissionStatus.REPLAN_REQUIRED,
                    MissionStatus.RECOVERY_FAILED}:
                break
            if result.command.throttle:
                travel_mm = abs(result.command.throttle) * 800.0 * dt
                motion = math.radians(
                    heading + 180.0 + motion_offset_deg)
                x += travel_mm * math.cos(motion)
                y += travel_mm * math.sin(motion)
                signed_body_distance = -travel_mm
                body_curvature = curvature_for_steering(
                    result.command.logical_steering, reverse=True)
                heading = (heading + math.degrees(
                    body_curvature * signed_body_distance)) % 360.0
        assert result is not None
        return result, to_slot_local(B1, x, y, heading), (x, y, heading)

    def test_real_motion_offset_lateral_error_converges(self) -> None:
        legacy, legacy_local, _ = self.simulate(guidance=False)
        guided, guided_local, pose = self.simulate(guidance=True)
        self.assertIs(legacy.mission_status, MissionStatus.REPLAN_REQUIRED)
        self.assertGreater(abs(legacy_local.lateral_mm), 60.0)
        self.assertIs(guided.mission_status, MissionStatus.DONE)
        self.assertLess(abs(guided_local.lateral_mm), 20.0)
        # FINAL uses strict 50 mm waypoint tolerance, not stop_distance padding.
        self.assertLessEqual(math.hypot(pose[0] - 425.0, pose[1] - 1050.0),
                             50.0 + 1e-6)

    def test_small_lateral_perturbations_converge(self) -> None:
        for delta in ((5.0, 0.0, 0.0), (-5.0, 0.0, 0.0),
                      (0.0, 5.0, 0.0), (0.0, -5.0, 0.0)):
            with self.subTest(delta=delta):
                result, local, _ = self.simulate(start_delta=delta)
                self.assertIs(result.mission_status, MissionStatus.DONE)
                self.assertLess(abs(local.lateral_mm), 30.0)

    def test_small_heading_perturbations_converge(self) -> None:
        for delta in (-2.0, 2.0):
            with self.subTest(delta=delta):
                result, local, _ = self.simulate(
                    start_delta=(0.0, 0.0, delta))
                self.assertIs(result.mission_status, MissionStatus.DONE)
                self.assertLess(abs(local.lateral_mm), 30.0)


if __name__ == "__main__":
    unittest.main()


class TestFinalReverseSteerCap(unittest.TestCase):
    """Phase 1: 정렬된 FINAL 직선 후진의 조향 포화를 깎는다.

    이 점-모델 sim 은 crab guidance(motion_heading)도 정지거리도 없어 종점
    포화(→1.0)로만 수렴한다. 실차에서는 그 포화가 PWM 을 밀어올려 슬롯 뒤(맵
    경계)를 넘는 과주행을 만든다. cap 은 그 포화를 깎는다. 그 대가로 이
    (불충실한) sim 에서는 한 번에 수렴하지 못하고 REPLAN 이 날 수 있는데,
    실 파이프라인에서는 REPLAN -> FINAL_POSE_EVAL -> (bounded) FINAL_ALIGNMENT
    로 이어지므로 회복 가능한 잔차다. 여기서 고정하는 것은 두 가지다:
    (1) cap 이 종점 조향 포화를 실제로 깎는다, (2) 잔차가 발산하지 않는다.
    """

    def _final_steer_and_lateral(self, cap_on: bool):
        route = build_rear_candidate_waypoints(
            B1, 10, from_pose=(240.0, 625.0),
            from_heading_deg=335.0, radii_mm=(1000.0,))
        targets = [waypoint_from_backend(w) for w in route
                   if w.phase in ("ENTRY", "FINAL")]
        reverse_start = [w for w in route if w.phase == "ALIGN"][-1]
        x, y = reverse_start.x + 5.0, reverse_start.y
        heading = reverse_start.target_heading_deg
        cfg = ControllerConfig(
            allow_reverse=True, reverse_trajectory_min_observations=3,
            final_reverse_straight_when_aligned=cap_on)
        host = HostController(mission=HostWaypointMission(targets), config=cfg)
        host.arm_auto()
        dt = 0.1
        peak_final_steer = 0.0
        for step in range(1000):
            now = 100.0 + step * dt
            r = host.tick(now, observation=Pose(
                x, y, heading, timestamp=now, heading_source="FRONT_CUSHION"))
            if r.command.throttle:
                # FINAL 구간(슬롯 입구 안쪽)에서의 조향만 본다.
                if to_slot_local(B1, x, y, heading).depth_mm > -150.0:
                    peak_final_steer = max(peak_final_steer,
                                           abs(r.command.steering))
                travel = abs(r.command.throttle) * 800.0 * dt
                motion = math.radians(heading + 180.0 + 8.0)
                x += travel * math.cos(motion)
                y += travel * math.sin(motion)
                bc = curvature_for_steering(r.command.logical_steering,
                                            reverse=True)
                heading = (heading + math.degrees(bc * (-travel))) % 360.0
            if r.mission_status in {MissionStatus.DONE,
                                    MissionStatus.REPLAN_REQUIRED,
                                    MissionStatus.RECOVERY_FAILED}:
                break
        return peak_final_steer, to_slot_local(B1, x, y, heading).lateral_mm

    def test_cap_cuts_the_terminal_steer_saturation(self) -> None:
        legacy_peak, _ = self._final_steer_and_lateral(cap_on=False)
        capped_peak, _ = self._final_steer_and_lateral(cap_on=True)
        self.assertGreater(legacy_peak, 0.9, "legacy 는 종점에서 포화한다")
        self.assertLessEqual(capped_peak,
                             ControllerConfig().final_reverse_aligned_steer_cap
                             + 1e-6)

    def test_capped_residual_does_not_diverge(self) -> None:
        """cap 잔차는 슬롯 기하가 감당하는 회복 가능 범위 안이다 (발산 아님)."""
        _, lateral = self._final_steer_and_lateral(cap_on=True)
        # 슬롯 반폭(100mm) 안 — FINAL_POSE_EVAL/FINAL_ALIGNMENT 가 회복한다.
        self.assertLess(abs(lateral), B1.width / 2.0)
