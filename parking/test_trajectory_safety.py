"""Pre-flight trajectory safety regression and property-style sweeps."""

from __future__ import annotations

import math
import random
import unittest
from types import SimpleNamespace

from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (Waypoint, build_rear_candidate_waypoints,
                               build_waypoints, default_slot_specs)
from pipeline import ParkingPipeline, PipelineConfig
from pipeline.runner import VehicleView


def wp(x: float, y: float, *, route: int = 1, index: int = 1,
       direction: str = "FORWARD", phase: str = "RECOVERY",
       curvature: float = 0.0) -> Waypoint:
    return Waypoint(route, index, phase, x, y, None, 5.0, 8.0, 30.0,
                    False, False, motion_direction=direction,
                    curvature=curvature)


class TestTrajectorySafetyRegression(unittest.TestCase):
    def test_real_unsafe_500mm_reverse_is_rejected_before_runtime(self) -> None:
        route = [wp(26.0, 659.0, direction="REVERSE")]
        result = validate_trajectory(
            route, start_pose=(521.0, 587.0, 352.0), target_slot="B1")
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "MAP_FOOTPRINT")

    def test_real_unsafe_reverse_never_reaches_auto_host_load_boundary(self) -> None:
        pipeline = ParkingPipeline(PipelineConfig(server_port=0))
        zeroed: list[int] = []
        pipeline.server = SimpleNamespace(
            stop_control=lambda car_id: zeroed.append(car_id))
        pipeline.dashboard = SimpleNamespace(push_event=lambda *a, **kw: None)
        view = VehicleView(track_id=7, car_id=1,
                           position_mm=(521.0, 587.0), heading_deg=352.0,
                           heading_source="FRONT_CUSHION")
        pipeline.views = {7: view}

        class FakeRunner:
            loaded = False

            def load_route(self, _route) -> None:
                self.loaded = True

        runner = FakeRunner()
        pipeline.auto_hosts = {1: runner}
        pipeline.hybrid_controls = {}
        events: list[tuple[str, dict]] = []
        pipeline.on_event_record = lambda name, **fields: events.append((name, fields))
        route = [wp(26.0, 659.0, route=99, direction="REVERSE")]

        loaded = pipeline._start_auto_host(1, "B1", route, view=view)

        self.assertFalse(loaded)
        self.assertFalse(runner.loaded)
        self.assertEqual(zeroed, [1])
        self.assertIn(
            ("ROUTE_REJECTED", {
                "car_id": 1, "route_id": 99, "slot": "B1",
                "reason": "MAP_FOOTPRINT",
            }), events)

    def test_ten_mm_initial_measurement_overflow_may_escape(self) -> None:
        route = [wp(200.0, 600.0)]
        result = validate_trajectory(
            route, start_pose=(115.0, 600.0, 0.0), target_slot="B1")
        self.assertTrue(result.safe)

    def test_curvature_below_physical_radius_is_rejected(self) -> None:
        result = validate_trajectory(
            [wp(200.0, 600.0, curvature=1.0 / 500.0)],
            start_pose=(150.0, 600.0, 0.0), target_slot="B1")
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "CURVATURE_LIMIT")

    def test_unreasonable_jump_is_rejected(self) -> None:
        result = validate_trajectory(
            [wp(1050.0, 600.0)], start_pose=(150.0, 600.0, 0.0))
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "SEGMENT_JUMP")

    def test_detected_obstacle_footprint_is_rejected(self) -> None:
        result = validate_trajectory(
            [wp(500.0, 600.0)], start_pose=(200.0, 600.0, 0.0),
            obstacle_poses=((400.0, 600.0, 0.0),))
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "OBSTACLE_FOOTPRINT")


class TestRepresentativeRearSweep(unittest.TestCase):
    def test_a1_a2_b1_b2_accepted_or_safe_failure(self) -> None:
        safe = planner_fail = 0
        for sid in ("A1", "A2", "B1", "B2"):
            spec = default_slot_specs()[sid]
            base = (spec.center_x - 105.0,
                    596.0 if sid.startswith("A") else 604.0,
                    17.0 if sid.startswith("A") else 343.0)
            for dx in (-50.0, -30.0, 0.0, 30.0, 50.0):
                for dy in (-50.0, -30.0, 0.0, 30.0, 50.0):
                    for dh in (-10.0, -5.0, 0.0, 5.0, 10.0):
                        pose = (base[0] + dx, base[1] + dy)
                        heading = base[2] + dh
                        try:
                            route = build_rear_candidate_waypoints(
                                spec, 7, from_pose=pose,
                                from_heading_deg=heading, strict=True)
                        except Exception:
                            planner_fail += 1       # explicit safe planner failure
                            continue
                        result = validate_trajectory(
                            route, start_pose=(*pose, heading), target_slot=sid)
                        self.assertTrue(
                            result.safe, (sid, pose, heading, result.reason))
                        safe += 1
        self.assertEqual(safe + planner_fail, 4 * 125)

    def test_global_sweep_is_safe_route_or_validator_rejection(self) -> None:
        safe = rejected = planner_fail = 0
        for sid in ("A1", "A2", "B1", "B2"):
            spec = default_slot_specs()[sid]
            for dx in (-50.0, -30.0, 0.0, 30.0, 50.0):
                for dy in (-50.0, -30.0, 0.0, 30.0, 50.0):
                    for dh in (-10.0, -5.0, 0.0, 5.0, 10.0):
                        pose = (150.0 + dx, 600.0 + dy)
                        heading = dh % 360.0
                        try:
                            route = build_waypoints(
                                spec, route_id=8, from_pose=pose,
                                from_heading_deg=heading, strict=True)
                        except Exception:
                            planner_fail += 1
                            continue
                        result = validate_trajectory(
                            route, start_pose=(*pose, heading), target_slot=sid)
                        if result.safe:
                            safe += 1
                        else:
                            rejected += 1           # production gate keeps zero control
        self.assertEqual(safe + rejected + planner_fail, 4 * 125)
        self.assertGreater(safe, 0)
        self.assertGreater(rejected + planner_fail, 0)


class TestAcceptedRouteProperty(unittest.TestCase):
    def test_random_routes_are_never_accepted_with_unsafe_sample(self) -> None:
        rng = random.Random(20260814)
        accepted = 0
        for route_id in range(500):
            sx, sy = rng.uniform(125, 1075), rng.uniform(125, 1075)
            heading = rng.uniform(0, 360)
            distance = rng.uniform(10, 900)
            bearing = math.radians(rng.uniform(0, 360))
            target = (sx + distance * math.cos(bearing),
                      sy + distance * math.sin(bearing))
            route = [wp(*target, route=route_id)]
            result = validate_trajectory(
                route, start_pose=(sx, sy, heading), target_slot="B1")
            if result.safe:
                accepted += 1
                self.assertGreater(result.sampled_poses, 0)
                self.assertGreaterEqual(result.min_clearance_mm, -20.0)
                self.assertLessEqual(result.path_length_mm, 6000.0)
        self.assertGreater(accepted, 20)


if __name__ == "__main__":
    unittest.main()
