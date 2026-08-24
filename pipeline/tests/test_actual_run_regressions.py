"""Deterministic replay of the 2026-08-14 rear-parking E2E failures."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from control.auto_host_runner import MissionStatus
from controller.config import ControllerConfig
from controller.models import ControlMode, MotionDirection, Pose, Waypoint
from host_control import Authority, HostController
from host_control.mission import HostWaypointMission
from host_control.producers import AutoControlProducer
from integration.backend_adapter import waypoint_from_backend
from parking.waypoints import (build_rear_candidate_waypoints,
                               choose_rear_candidate, default_slot_specs,
                               plan_setup_recovery)
from parking.trajectory_safety import validate_trajectory
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView


class _Mission:
    is_recovering = False

    def __init__(self) -> None:
        self.requests: list[str] = []

    def request_replan(self, reason: str) -> None:
        self.requests.append(reason)


class _Runner:
    status = MissionStatus.RUNNING
    current_is_terminal = False

    def __init__(self, target) -> None:
        self.current_target = target
        self.mission = _Mission()


def _target(route_id: int, x: float, y: float):
    return SimpleNamespace(
        x_mm=x, y_mm=y, target_heading_deg=None,
        position_tolerance_cm=6.0, capture_tolerance_cm=10.0,
        heading_tolerance_deg=15.0, phase="APPROACH", is_final=False,
        route_id=route_id, waypoint_id=1,
        motion_direction=SimpleNamespace(value="FORWARD"),
    )


class TestActualRunRearAcquisition(unittest.TestCase):
    def setUp(self) -> None:
        p = ParkingPipeline.__new__(ParkingPipeline)
        p.config = PipelineConfig(parking_mode="rear", deviation_frames=3)
        p._parking_stage = {1: "PARKING"}
        p._deviation_streak = {}
        p._forward_closest = {}
        p._reverse_closest = {}
        p.dashboard = SimpleNamespace(push_event=lambda *a, **kw: None)
        self.p = p
        self.view = VehicleView(
            track_id=7, car_id=1, heading_source="FRONT_CUSHION")

    def replay(self, target, poses) -> _Runner:
        runner = _Runner(target)
        for i, (x, y, heading) in enumerate(poses, 1):
            self.view.position_mm = (x, y)
            self.view.heading_deg = heading
            self.view.last_obs_time = float(i)
            self.p._check_path_deviation(self.view, runner)
        return runner

    def test_run_142941_first_rear_route_gets_control_opportunity(self) -> None:
        runner = self.replay(
            _target(5, 330.4706, 533.4044),
            [(250.9, 628.8, 340.8), (208.0, 640.4, 340.8),
             (187.3, 644.8, 335.1)],
        )
        self.assertEqual(runner.mission.requests, [])

    def test_run_172010_post_comm_pose_has_safe_fresh_replan(self) -> None:
        """Never resume route 4; pose (336,549,343) can produce a new safe B1 route."""
        pose = (336.0, 549.0, 343.0)
        wps = build_rear_candidate_waypoints(
            default_slot_specs()["B1"], route_id=99,
            from_pose=pose[:2], from_heading_deg=pose[2], strict=True)
        result = validate_trajectory(
            wps, start_pose=pose, target_slot="B1",
            lot_size_mm=(1200.0, 1200.0))
        self.assertTrue(result.safe, result.reason)
        self.assertEqual(len(wps), 9)
        self.assertTrue(all(w.route_id == 99 for w in wps))

    def test_run_143048_replans_converge_without_thrashing(self) -> None:
        cases = [
            (_target(6, 344.7271, 528.3396),
             [(295.0, 605.9, 340.0), (263.0, 609.0, 339.1),
              (245.1, 616.2, 336.2)]),
            (_target(7, 414.5856, 500.2372),
             [(239.5, 617.6, 336.1), (240.8, 613.4, 337.2)]),
            (_target(8, 395.9768, 508.2696),
             [(242.2, 613.4, 337.2), (268.4, 600.5, 337.6)]),
            (_target(9, 392.2283, 509.8387),
             [(272.6, 599.0, 337.4), (304.3, 581.8, 336.2)]),
        ]
        for target, poses in cases:
            with self.subTest(route=target.route_id):
                self.p._forward_closest.clear()
                runner = self.replay(target, poses)
                self.assertEqual(runner.mission.requests, [])

        # Exercise the real controller as well as the reachability monitor.
        producer = AutoControlProducer(ControllerConfig(allow_reverse=True))
        target = Waypoint(
            414.5856, 500.2372, position_tolerance_cm=6.0,
            capture_tolerance_cm=10.0, route_id=7, waypoint_id=1,
            phase="APPROACH", motion_direction=MotionDirection.FORWARD)
        cmd = producer.compute(
            Pose(239.5, 614.8, 337.2, timestamp=10.0,
                 heading_source="FRONT_CUSHION"),
            target, now=10.0)
        self.assertGreater(cmd.throttle, 0.0)
        self.assertNotEqual(cmd.steering, 0.0)

    def test_route_load_resets_acquisition_and_deviation_state(self) -> None:
        self.p._deviation_streak[1] = 2
        self.p._forward_closest[1] = ((5, 1), 90.0)
        self.p._reverse_closest[1] = ((4, 2), 70.0)
        self.p._auto_host_route = {}
        self.p.on_route_load = None
        self.p._emit_route([_target(6, 344.7, 528.3)], car_id=1)
        self.assertNotIn(1, self.p._deviation_streak)
        self.assertNotIn(1, self.p._forward_closest)
        self.assertNotIn(1, self.p._reverse_closest)

    def test_run_173900_camera_gap_is_stale_before_map_exit(self) -> None:
        """Last real observation at 25.078 cannot drive until 27.812."""
        target = Waypoint(
            717.89, 342.89, target_heading_deg=315.0,
            position_tolerance_cm=4.0, heading_tolerance_deg=5.0,
            heading_required=True, route_id=5, waypoint_id=3, phase="ALIGN",
            curvature=-1.0 / 1000.0, path_capture_tolerance_cm=10.0,
        )
        cmd = AutoControlProducer(ControllerConfig(allow_reverse=True)).compute(
            Pose(614.8, 434.1, 323.3, timestamp=25.078,
                 heading_source="FRONT_CUSHION"),
            target, now=25.678)
        self.assertEqual(cmd.reason, "POSE_STALE")
        self.assertEqual(cmd.throttle, 0.0)

    def test_run_174030_reverse_arc_sample_does_not_replan_backwards(self) -> None:
        target = Waypoint(
            612.0, 536.0, target_heading_deg=310.0,
            position_tolerance_cm=4.0, heading_tolerance_deg=12.0,
            heading_required=False, route_id=4, waypoint_id=4, phase="ENTRY",
            motion_direction=MotionDirection.REVERSE, curvature=1.0 / 800.0,
            path_capture_tolerance_cm=10.0,
        )
        producer = AutoControlProducer(ControllerConfig(allow_reverse=True))
        before = producer.compute(
            Pose(584.4, 502.9, 304.7, timestamp=28.0,
                 heading_source="FRONT_CUSHION"), target, now=28.0)
        crossed = producer.compute(
            Pose(560.7, 525.8, 301.5, timestamp=28.2,
                 heading_source="FRONT_CUSHION"), target, now=28.2)
        self.assertFalse(before.arrived)
        self.assertTrue(crossed.arrived)
        self.assertEqual(crossed.reason, "ARC_ENDPOINT_PASSED")

    def test_run_174030_monitor_then_controller_advances_to_next_sample(self) -> None:
        wp4 = Waypoint(
            612.0, 536.0, target_heading_deg=310.0,
            position_tolerance_cm=4.0, heading_tolerance_deg=12.0,
            heading_required=False, route_id=4, waypoint_id=4, phase="ENTRY",
            motion_direction=MotionDirection.REVERSE, curvature=1.0 / 800.0,
            path_capture_tolerance_cm=10.0,
        )
        wp5 = Waypoint(
            540.0, 640.0, target_heading_deg=300.0,
            position_tolerance_cm=4.0, route_id=4, waypoint_id=5,
            phase="ENTRY", motion_direction=MotionDirection.REVERSE,
            curvature=1.0 / 800.0, path_capture_tolerance_cm=10.0,
        )
        mission = HostWaypointMission([wp4, wp5])
        runner = SimpleNamespace(
            current_target=wp4, current_is_terminal=False,
            status=MissionStatus.RUNNING, mission=mission)
        producer = AutoControlProducer(ControllerConfig(allow_reverse=True))

        # Production checks deviation on the camera callback before the 100 ms
        # controller tick.  The 9 mm distance increase is not a replan; the
        # same observation then captures the crossed arc sample.
        for t, pose in (
            (28.0, Pose(584.4, 502.9, 304.7, 28.0,
                        heading_source="FRONT_CUSHION")),
            (28.2, Pose(560.7, 525.8, 301.5, 28.2,
                        heading_source="FRONT_CUSHION")),
        ):
            self.view.position_mm = (pose.x_mm, pose.y_mm)
            self.view.heading_deg = pose.heading_deg
            self.view.last_obs_time = pose.timestamp
            runner.current_target = mission.current_target()
            self.p._check_path_deviation(self.view, runner)
            self.assertEqual(mission.status, MissionStatus.RUNNING)
            cmd = producer.compute(pose, mission.current_target(), now=t)
            mission.notify_result(cmd)

        self.assertEqual(mission.index, 1)
        self.assertIs(mission.current_target(), wp5)

    def test_153932_reverse_pause_waits_for_physical_heading_then_moves(self) -> None:
        producer = AutoControlProducer(ControllerConfig(allow_reverse=True))
        align = Waypoint(
            711.5, 276.2, target_heading_deg=319.7,
            position_tolerance_cm=4.0, phase="ALIGN",
            motion_direction=MotionDirection.FORWARD,
            curvature=-1.0 / 800.0,
        )
        entry = Waypoint(
            680.0, 360.0, target_heading_deg=310.0,
            position_tolerance_cm=4.0, phase="ENTRY",
            motion_direction=MotionDirection.REVERSE,
            curvature=1.0 / 800.0,
        )
        producer.compute(
            Pose(687.5, 292.4, 317.1, 23.125,
                 heading_source="TRAJECTORY"), align, now=23.125)
        interlock = producer.compute(
            Pose(711.5, 276.2, 319.7, 23.375,
                 heading_source="TRAJECTORY"), entry, now=23.484)
        unsafe = producer.compute(
            Pose(737.0, 247.0, 318.5, 23.625,
                 heading_source="TRAJECTORY"), entry, now=23.703)
        trusted = producer.compute(
            Pose(778.1, 164.9, 322.7, 24.609,
                 heading_source="FRONT_CUSHION"), entry, now=24.703)
        # In this real run the reverse-heading safety gate precedes the
        # direction latch because only TRAJECTORY heading was available.  It
        # still produces the required zero before any reverse command.
        self.assertEqual(interlock.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(interlock.throttle, 0.0)
        self.assertEqual(unsafe.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(unsafe.throttle, 0.0)
        self.assertLess(trusted.throttle, 0.0)

    def test_155812_wp3_uses_prior_settled_heading_at_endpoint(self) -> None:
        wp3 = Waypoint(
            717.8932188134522, 342.8932188134521,
            target_heading_deg=315.0, position_tolerance_cm=4.0,
            heading_tolerance_deg=5.0, heading_required=True,
            route_id=3, waypoint_id=3, phase="ALIGN",
            motion_direction=MotionDirection.FORWARD,
            curvature=-1.0 / 1000.0, path_capture_tolerance_cm=10.0,
        )
        producer = AutoControlProducer(ControllerConfig(allow_reverse=True))
        before = producer.compute(
            Pose(662.4, 396.5, 313.2, 1350.421,
                 heading_source="TRAJECTORY"), wp3, now=1350.421)
        capture = producer.compute(
            Pose(704.6, 350.0, 309.6, 1350.890,
                 heading_source="TRAJECTORY"), wp3, now=1350.890)
        self.assertFalse(before.arrived)
        self.assertTrue(capture.arrived)
        self.assertEqual(capture.mode, ControlMode.ARRIVED)
        self.assertEqual(capture.reason, "ALIGN_SETTLED_CAPTURE")
        self.assertAlmostEqual(capture.heading_error_deg, 5.4, places=1)
        self.assertEqual(capture.throttle, 0.0)

    def test_align_settled_capture_does_not_accept_one_shot_miss(self) -> None:
        wp3 = Waypoint(
            717.8932188134522, 342.8932188134521,
            target_heading_deg=315.0, position_tolerance_cm=4.0,
            heading_tolerance_deg=5.0, heading_required=True,
            route_id=3, waypoint_id=3, phase="ALIGN",
            motion_direction=MotionDirection.FORWARD,
            curvature=-1.0 / 1000.0, path_capture_tolerance_cm=10.0,
        )
        capture = AutoControlProducer(
            ControllerConfig(allow_reverse=True)).compute(
                Pose(704.6, 350.0, 309.6, 1350.890,
                     heading_source="TRAJECTORY"), wp3, now=1350.890)
        self.assertFalse(capture.arrived)
        self.assertEqual(capture.reason, "HEADING_OUT_OF_TOLERANCE")

    def test_161336_crossed_align_endpoint_uses_settled_observation(self) -> None:
        wp3 = Waypoint(
            1050.0, 842.820323027551, target_heading_deg=30.0,
            position_tolerance_cm=4.0, heading_tolerance_deg=5.0,
            heading_required=True, route_id=3, waypoint_id=3,
            phase="ALIGN", motion_direction=MotionDirection.FORWARD,
            curvature=1.0 / 800.0, path_capture_tolerance_cm=10.0,
        )
        producer = AutoControlProducer(ControllerConfig(allow_reverse=True))
        converged = producer.compute(
            Pose(1016.8, 867.9, 29.3, 2314.156,
                 heading_source="TRAJECTORY"), wp3, now=2314.156)
        crossed = producer.compute(
            Pose(1033.2, 884.4, 36.9, 2314.375,
                 heading_source="TRAJECTORY"), wp3, now=2314.375)
        self.assertFalse(converged.arrived)
        self.assertTrue(crossed.arrived)
        self.assertEqual(crossed.reason, "ALIGN_SETTLED_CAPTURE")

    def test_latest_map_exit_align_is_replanned_before_boundary(self) -> None:
        target = _target(3, 747.1825, 272.1825)
        target.phase = "ALIGN"
        runner = self.replay(
            target,
            [(691.5, 229.9, 316.4), (712.8, 204.9, 314.4),
             (726.9, 187.3, 311.0), (744.0, 168.1, 310.9)],
        )
        self.assertEqual(runner.mission.requests, ["PATH_DEVIATION"])

    def test_latest_map_exit_align_with_trajectory_heading_is_monitored(self) -> None:
        self.view.heading_source = "TRAJECTORY"
        target = _target(3, 747.1825, 272.1825)
        target.phase = "ALIGN"
        runner = self.replay(
            target,
            [(691.5, 229.9, 316.4), (712.8, 204.9, 314.4),
             (726.9, 187.3, 311.0), (744.0, 168.1, 310.9)],
        )
        self.assertEqual(runner.mission.requests, ["PATH_DEVIATION"])

    def test_real_pose_prefers_forward_maneuver_clearance(self) -> None:
        selected, candidates = choose_rear_candidate(
            default_slot_specs()["B1"], (310.6, 618.4), 337.3)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.radius_mm, 800.0)
        old = next(c for c in candidates
                   if c.radius_mm == 1100.0
                   and c.phi_deg == 45.0 and c.side == 1)
        self.assertFalse(old.feasible)
        self.assertIn("heading", old.reason)
        self.assertLessEqual(
            math.degrees(math.atan2(selected.lateral_mm, selected.run_mm)), 5.0)

    def test_real_run_requires_bounded_setup_before_arc(self) -> None:
        slot = default_slot_specs()["B1"]
        selected, _ = choose_rear_candidate(
            slot, (320.1, 604.3), 341.6)
        recovery = plan_setup_recovery(slot, (320.1, 604.3), 341.6)
        self.assertIsNone(selected)
        self.assertIsNotNone(recovery)

    def test_post_setup_arc_entry_heading_mismatch_stops_before_arc(self) -> None:
        slot = default_slot_specs()["B1"]
        recovery = plan_setup_recovery(slot, (320.1, 604.3), 341.6)
        self.assertIsNotNone(recovery)
        route = build_rear_candidate_waypoints(
            slot, 40, from_pose=recovery.end_pose[:2],
            from_heading_deg=recovery.end_pose[2])
        target = waypoint_from_backend(route[0])
        cmd = AutoControlProducer(ControllerConfig(allow_reverse=True)).compute(
            Pose(target.x_mm, target.y_mm,
                 target.target_heading_deg - 12.0, timestamp=10.0,
                 heading_source="FRONT_CUSHION"),
            target, now=10.0)
        self.assertEqual(cmd.mode, ControlMode.ALIGN)
        self.assertEqual(cmd.reason, "HEADING_OUT_OF_TOLERANCE")

    def test_matching_arc_entry_heading_can_complete_approach(self) -> None:
        slot = default_slot_specs()["B1"]
        recovery = plan_setup_recovery(slot, (320.1, 604.3), 341.6)
        self.assertIsNotNone(recovery)
        route = build_rear_candidate_waypoints(
            slot, 41, from_pose=recovery.end_pose[:2],
            from_heading_deg=recovery.end_pose[2])
        target = waypoint_from_backend(route[0])
        cmd = AutoControlProducer(ControllerConfig(allow_reverse=True)).compute(
            Pose(target.x_mm, target.y_mm, target.target_heading_deg,
                 timestamp=10.0, heading_source="FRONT_CUSHION"),
            target, now=10.0)
        self.assertTrue(cmd.arrived)


class TestActualRunReverseObservationContract(unittest.TestCase):
    """Replay the 16:37/16:39/16:40 rear ENTRY stop timestamps."""

    @staticmethod
    def _host(route_id: int, x: float, y: float, heading: float,
              curvature: float) -> HostController:
        mission = HostWaypointMission([
            Waypoint(
                x, y, target_heading_deg=heading,
                position_tolerance_cm=4.0,
                heading_tolerance_deg=12.0,
                route_id=route_id, waypoint_id=4,
                phase="ENTRY", motion_direction=MotionDirection.REVERSE,
                curvature=curvature, path_capture_tolerance_cm=10.0,
            )
        ])
        host = HostController(
            mission=mission,
            config=ControllerConfig(
                allow_reverse=True, reverse_heading_wait_timeout_s=2.5),
        )
        host.arm_auto()
        return host

    def test_163757_short_loss_zero_then_front_reacquisition_resumes(self) -> None:
        host = self._host(12, 860.0813, 796.5638, 54.0, -1.0 / 1100.0)

        moving = host.tick(
            46.578,
            observation=Pose(
                1019.5, 998.3, 45.4, timestamp=46.437,
                heading_source="FRONT_CUSHION"),
        )
        heading_lost = host.tick(
            46.687,
            observation=Pose(
                1023.6, 999.6, 45.4, timestamp=46.687,
                heading_source="LAST_VALID"),
        )
        camera_gap = host.tick(47.359)
        reacquired = host.tick(
            47.672,
            observation=Pose(
                964.8, 957.7, 43.9, timestamp=47.625,
                heading_source="FRONT_CUSHION"),
        )

        self.assertLess(moving.command.throttle, 0.0)
        self.assertEqual(heading_lost.command.reason,
                         "REVERSE_HEADING_UNSAFE")
        self.assertEqual(camera_gap.command.reason, "POSE_STALE")
        self.assertEqual(camera_gap.command.throttle, 0.0)
        self.assertIs(camera_gap.authority, Authority.AUTO_HOST)
        self.assertLess(reacquired.command.throttle, 0.0)
        self.assertIs(reacquired.mission_status, MissionStatus.RUNNING)

    def test_163920_persistent_last_valid_reaches_bounded_replan(self) -> None:
        host = self._host(4, 615.9830, 462.2147, 306.0, 1.0 / 1000.0)

        first = host.tick(
            21.297,
            observation=Pose(
                776.7, 237.9, 305.6, timestamp=21.234,
                heading_source="LAST_VALID"),
        )
        host.tick(
            22.844,
            observation=Pose(
                700.3, 318.3, 305.6, timestamp=22.734,
                heading_source="LAST_VALID"),
        )
        stale = host.tick(23.391)
        still_lost = host.tick(
            23.500,
            observation=Pose(
                611.8, 411.3, 305.6, timestamp=23.406,
                heading_source="LAST_VALID"),
        )
        timed_out = host.tick(
            23.797,
            observation=Pose(
                586.7, 444.4, 305.6, timestamp=23.625,
                heading_source="LAST_VALID"),
        )

        self.assertEqual(first.command.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(stale.command.reason, "POSE_STALE")
        self.assertEqual(still_lost.command.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(timed_out.command.reason, "REVERSE_HEADING_TIMEOUT")
        self.assertIs(timed_out.mission_status, MissionStatus.REPLAN_REQUIRED)
        self.assertIs(timed_out.authority, Authority.AUTO_HOST)

    def test_164011_persistent_last_valid_cannot_run_or_wait_forever(self) -> None:
        host = self._host(3, 635.0813, 403.4362, 306.0, 1.0 / 1100.0)

        first = host.tick(
            22.969,
            observation=Pose(
                763.9, 197.2, 301.3, timestamp=22.813,
                heading_source="LAST_VALID"),
        )
        timed_out = host.tick(
            25.500,
            observation=Pose(
                591.8, 369.8, 301.3, timestamp=25.406,
                heading_source="LAST_VALID"),
        )

        self.assertEqual(first.command.throttle, 0.0)
        self.assertEqual(first.command.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(timed_out.command.reason, "REVERSE_HEADING_TIMEOUT")
        self.assertIs(timed_out.mission_status, MissionStatus.REPLAN_REQUIRED)

    def test_161237_front_cushion_keeps_reverse_tracking(self) -> None:
        host = self._host(8, 650.0, 150.0, 90.0, 0.0)

        first = host.tick(
            30.750,
            observation=Pose(
                671.0, 313.0, 80.0, timestamp=30.703,
                heading_source="FRONT_CUSHION"),
        )
        second = host.tick(
            30.860,
            observation=Pose(
                666.0, 285.0, 82.0, timestamp=30.828,
                heading_source="FRONT_CUSHION"),
        )

        self.assertLess(first.command.throttle, 0.0)
        self.assertLess(second.command.throttle, 0.0)
        self.assertIs(second.mission_status, MissionStatus.RUNNING)

    @staticmethod
    def _align_to_reverse_host(
        *, route_id: int, align: Waypoint, entry: Waypoint,
    ) -> HostController:
        host = HostController(
            mission=HostWaypointMission([align, entry]),
            config=ControllerConfig(
                allow_reverse=True, reverse_heading_wait_timeout_s=2.5),
        )
        host.arm_auto()
        return host

    def test_165732_align_trajectory_anchor_starts_reverse_after_interlock(self) -> None:
        align = Waypoint(
            717.8932, 342.8932, target_heading_deg=315.0,
            speed_cm_s=5.0, position_tolerance_cm=4.0,
            heading_tolerance_deg=5.0, heading_required=True,
            route_id=3, waypoint_id=4, phase="ALIGN",
            motion_direction=MotionDirection.FORWARD,
            curvature=-0.001, path_capture_tolerance_cm=10.0,
        )
        entry = Waypoint(
            615.9830, 462.2147, target_heading_deg=306.0,
            speed_cm_s=5.0, position_tolerance_cm=4.0,
            route_id=3, waypoint_id=5, phase="ENTRY",
            motion_direction=MotionDirection.REVERSE,
            curvature=0.001, path_capture_tolerance_cm=10.0,
        )
        host = self._align_to_reverse_host(route_id=3, align=align, entry=entry)
        samples = (
            (19.02, 628.8, 426.9, 319.3),
            (19.34, 645.6, 412.4, 317.4),
            (19.56, 662.4, 390.7, 311.9),
            (20.00, 704.6, 347.2, 312.8),
        )
        last = None
        for t, x, y, h in samples:
            last = host.tick(
                t, observation=Pose(
                    x, y, h, timestamp=t, heading_source="TRAJECTORY"))
        self.assertIsNotNone(last)
        self.assertEqual(host.mission.current_target().waypoint_id, 5)

        interlock = host.tick(20.11)
        started = host.tick(
            20.22,
            observation=Pose(
                725.7, 326.8, 314.1, timestamp=20.22,
                heading_source="TRAJECTORY"),
        )
        self.assertEqual(interlock.command.reason, "DIRECTION_CHANGE_STOP")
        self.assertLess(started.command.throttle, 0.0)
        self.assertNotEqual(started.command.reason, "REVERSE_HEADING_UNSAFE")

    def test_165826_align_trajectory_anchor_starts_reverse_after_interlock(self) -> None:
        align = Waypoint(
            825.0, 357.1797, target_heading_deg=330.0,
            speed_cm_s=5.0, position_tolerance_cm=4.0,
            heading_tolerance_deg=5.0, heading_required=True,
            route_id=4, waypoint_id=3, phase="ALIGN",
            motion_direction=MotionDirection.FORWARD,
            curvature=-0.00125, path_capture_tolerance_cm=10.0,
        )
        entry = Waypoint(
            710.7699, 437.1644, target_heading_deg=320.0,
            speed_cm_s=5.0, position_tolerance_cm=4.0,
            route_id=4, waypoint_id=4, phase="ENTRY",
            motion_direction=MotionDirection.REVERSE,
            curvature=0.00125, path_capture_tolerance_cm=10.0,
        )
        host = self._align_to_reverse_host(route_id=4, align=align, entry=entry)
        samples = (
            (18.66, 693.5, 426.3, 336.9),
            (18.88, 716.0, 413.2, 334.5),
            (19.09, 732.9, 397.3, 331.5),
            (19.31, 756.8, 382.7, 326.8),
            (19.53, 772.4, 371.0, 325.0),
            (19.75, 799.2, 352.1, 323.7),
        )
        for t, x, y, h in samples:
            host.tick(
                t, observation=Pose(
                    x, y, h, timestamp=t, heading_source="TRAJECTORY"))
        self.assertEqual(host.mission.current_target().waypoint_id, 4)

        interlock = host.tick(19.86)
        started = host.tick(
            19.97,
            observation=Pose(
                821.8, 337.5, 326.1, timestamp=19.97,
                heading_source="TRAJECTORY"),
        )
        self.assertEqual(interlock.command.reason, "DIRECTION_CHANGE_STOP")
        self.assertLess(started.command.throttle, 0.0)

    def test_reverse_start_without_front_or_align_anchor_stays_zero(self) -> None:
        host = self._host(99, -500.0, 0.0, 0.0, 0.0)
        outputs = [
            host.tick(
                40.0 + i * 0.2,
                observation=Pose(
                    0.0, 0.0, 0.0, timestamp=40.0 + i * 0.2,
                    heading_source="TRAJECTORY"),
            )
            for i in range(3)
        ]
        self.assertTrue(all(r.command.throttle == 0.0 for r in outputs))
        self.assertTrue(all(
            r.command.reason == "REVERSE_HEADING_UNSAFE" for r in outputs))


if __name__ == "__main__":
    unittest.main()
