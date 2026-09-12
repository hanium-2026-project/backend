"""Pre-flight trajectory safety regression and property-style sweeps."""

from __future__ import annotations

import math
import random
import unittest
from types import SimpleNamespace

from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (Waypoint, _car_footprint,
                               build_rear_candidate_waypoints,
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
        # 예전에는 여기서 "일부는 거절된다" 를 요구했다. 그 거절은 전부
        # **출발 자세가 이미 맵 밖** (이 sweep 은 x=100 에도 차를 놓는데
        # 250mm 차체가 5~36mm 삐져나온다) 인데 탈출 허용치가 20mm 로
        # 묶여 있어서 생긴 것이었다 — 즉 실제 위험을 잡은 것이 아니라
        # 자기 출발 자세를 거절한 것이다 (run_20260901_154551 에서 차가
        # 56.8mm 나간 뒤 어떤 복구 경로도 실을 수 없었던 것과 같은 원인).
        #
        # 지금 규칙은 "출발보다 나빠지지 않기" 이고, 이 sweep 의 500개
        # 경로 중 자기 출발 초과량을 넘는 것은 하나도 없다. 그래서 전부
        # 안전이 맞다. 게이트가 실제로 거절을 하는지는 아래에서 명시적으로
        # 확인한다.
        # 맵 안에서 출발해 맵 밖으로 걸어 나가는 경로는 반드시 거절된다.
        # 100mm 씩 이어 붙여 연속성을 유지하므로 SEGMENT_JUMP 가 아니라
        # 경계 판정으로 걸려야 한다.
        start = (150.0, 600.0, 0.0)
        self.assertLessEqual(
            max(max(-px, px - 1200.0, -py, py - 1200.0)
                for px, py in _car_footprint(*start)), 0.0,
            "이 검사는 맵 안에서 출발해야 의미가 있다")
        march = [Waypoint(route_id=99, waypoint_id=i, phase="CRUISE",
                          x=150.0 + 100.0 * i, y=600.0,
                          target_heading_deg=0.0, speed_cm_s=8.0,
                          position_tolerance_cm=6.0,
                          heading_tolerance_deg=12.0,
                          heading_required=False, is_final=(i == 11),
                          motion_direction="FORWARD")
                 for i in range(1, 12)]      # x 250 -> 1250 (맵 밖으로 나간다)
        result = validate_trajectory(march, start_pose=start, target_slot="B1")
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "MAP_FOOTPRINT")


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
