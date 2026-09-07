"""Deterministic Level-1/2 multi-car production contract regressions."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from parking.trajectory_safety import validate_trajectory
from parking.waypoints import build_waypoints, default_slot_specs
from pipeline.runner import ParkingPipeline, VehicleView
from rl.bridge import RealtimeAllocator
from rl.parking_env import SLOT_NAMES


def _wp(x: float, y: float, route_id: int = 1):
    return SimpleNamespace(
        route_id=route_id, waypoint_id=1, phase="CRUISE",
        x=x, y=y, target_heading_deg=0.0,
        motion_direction="FORWARD", curvature=0.0)


def _safety_pipeline() -> tuple[ParkingPipeline, list[int], list[tuple]]:
    p = ParkingPipeline.__new__(ParkingPipeline)
    p.config = SimpleNamespace(
        lot_width_mm=1200.0, lot_height_mm=1200.0,
        boundary_hard_margin_mm=20.0,
        boundary_measurement_uncertainty_mm=10.0,
        controller_config=SimpleNamespace(max_pose_age_s=0.5))
    p._plan_radius = 610.0
    p._parked_obstacles = {}
    p.views = {}
    p.allocator = SimpleNamespace(slot_statuses=[0.0] * len(SLOT_NAMES))
    zeroed: list[int] = []
    events: list[tuple] = []
    p.server = SimpleNamespace(stop_control=lambda car_id: zeroed.append(car_id))
    p.dashboard = SimpleNamespace(push_event=lambda *a, **k: None)
    p.on_event_record = lambda name, **fields: events.append((name, fields))
    return p, zeroed, events


class TestSlotReservation(unittest.TestCase):
    def test_two_allocations_cannot_receive_the_same_slot(self) -> None:
        allocator = RealtimeAllocator(model_path="unused.zip")
        allocator.update(101, (150.0, 600.0))
        allocator.update(202, (150.0, 600.0))

        def first_available(_obs, masks, **_kw):
            return next(i for i, allowed in enumerate(masks[:8]) if allowed)

        with patch("rl.bridge.select_action", side_effect=first_available):
            first = allocator.allocate(101)
            second = allocator.allocate(202)

        self.assertNotEqual(first, second)
        self.assertEqual(allocator.slot_statuses[SLOT_NAMES.index(first)], 1.0)
        self.assertEqual(allocator.slot_statuses[SLOT_NAMES.index(second)], 1.0)


class TestStaticVehicleObstacleContract(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline, self.zeroed, self.events = _safety_pipeline()
        self.active = VehicleView(
            track_id=22, car_id=2, position_mm=(150.0, 600.0),
            heading_deg=0.0, heading_source="FRONT_CUSHION",
            last_obs_time=10.0)
        self.pipeline.views = {22: self.active}

    def test_parked_verified_pose_persists_without_live_detection(self) -> None:
        self.pipeline._parked_obstacles[1] = (400.0, 600.0, 0.0)
        poses, uncertain = self.pipeline._planning_obstacle_snapshot(self.active)
        self.assertEqual(poses, ((400.0, 600.0, 0.0),))
        self.assertEqual(uncertain, ())
        self.assertFalse(self.pipeline._trajectory_safe(
            self.active, [_wp(650.0, 600.0)], slot_id=None))
        self.assertEqual(self.events[-1][1]["reason"], "OBSTACLE_FOOTPRINT")

    def test_waiting_vehicle_without_fresh_body_heading_blocks_new_route(self) -> None:
        waiting = VehicleView(
            track_id=11, car_id=1, position_mm=(400.0, 600.0),
            heading_deg=0.0, heading_source="LAST_VALID",
            last_obs_time=10.0)
        self.pipeline.views[11] = waiting
        self.assertFalse(self.pipeline._trajectory_safe(
            self.active, [_wp(650.0, 600.0)], slot_id=None))
        self.assertEqual(
            self.events[-1][1]["reason"], "OTHER_VEHICLE_POSE_UNCERTAIN")

    def test_planner_excludes_own_vehicle_by_car_id(self) -> None:
        self.pipeline._parked_obstacles[2] = (150.0, 600.0, 0.0)
        poses, uncertain = self.pipeline._planning_obstacle_snapshot(self.active)
        self.assertEqual(poses, ())
        self.assertEqual(uncertain, ())

        self.pipeline._parked_obstacles = {7: (400.0, 600.0, 0.0)}
        poses, _ = self.pipeline._planning_obstacle_snapshot(self.active)
        self.assertEqual(poses, ((400.0, 600.0, 0.0),))

    def test_obstacle_uncertainty_margin_is_geometry_not_slot_hardcode(self) -> None:
        route = [_wp(650.0, 600.0)]
        exact = validate_trajectory(
            route, start_pose=(150.0, 600.0, 0.0),
            obstacle_poses=((400.0, 760.0, 0.0),), obstacle_margin_mm=0.0)
        guarded = validate_trajectory(
            route, start_pose=(150.0, 600.0, 0.0),
            obstacle_poses=((400.0, 760.0, 0.0),), obstacle_margin_mm=10.0)
        self.assertTrue(exact.safe)
        self.assertFalse(guarded.safe)
        self.assertEqual(guarded.reason, "OBSTACLE_FOOTPRINT")

    def test_other_slot_routes_are_safe_for_multiple_parked_slot_geometries(self) -> None:
        specs = default_slot_specs()
        for parked_slot, parked_pose, target_slot in (
                ("B1", (425.0, 1050.0, 270.0), "A1"),
                ("A2", (650.0, 150.0, 90.0), "B1")):
            with self.subTest(parked=parked_slot, target=target_slot):
                route = build_waypoints(
                    specs[target_slot], route_id=7,
                    from_pose=(150.0, 600.0), from_heading_deg=0.0,
                    strict=True)
                result = validate_trajectory(
                    route, start_pose=(150.0, 600.0, 0.0),
                    target_slot=target_slot, occupied_slots=(parked_slot,),
                    obstacle_poses=(parked_pose,), obstacle_margin_mm=10.0)
                self.assertTrue(result.safe, result.reason)


class TestSequentialAutoHostPolicy(unittest.TestCase):
    def test_second_car_waits_zero_while_first_car_is_active(self) -> None:
        p = ParkingPipeline.__new__(ParkingPipeline)
        p.config = SimpleNamespace(
            control_mode="auto-host", manual_only=False,
            entry_nodes=("junction",))
        p._auto_host_slot = {1: "B1"}
        p._parking_stage = {1: "PARKING"}
        p._comm_lost = set()
        p._comm_recovery_context = {}
        p._allocation_state = {}
        zeroed: list[int] = []
        p.server = SimpleNamespace(stop_control=lambda car_id: zeroed.append(car_id))
        view = VehicleView(
            track_id=22, car_id=2, node="junction",
            position_mm=(150.0, 600.0), heading_deg=0.0,
            heading_source="FRONT_CUSHION", last_obs_time=10.0)

        p._ensure_mission(view, frame_index=1)

        self.assertEqual(p._allocation_state[2], "WAIT_OTHER_VEHICLE")
        self.assertEqual(zeroed, [2])
        self.assertEqual(p._parking_stage[1], "PARKING")


class TestMultiCarRecorder(unittest.TestCase):
    def test_one_camera_frame_records_every_vehicle(self) -> None:
        p = ParkingPipeline.__new__(ParkingPipeline)
        p._frame_seq = 0
        p._prev_frame_index = None
        p._dropped_frames = 0
        p._pose_observed_tracks = set()
        p._last_perception = {}
        p.last_pose_rec = None
        rows: list[dict] = []
        p.on_pose_record = lambda row: rows.append(dict(row))
        p.on_event_record = lambda *_a, **_k: None
        v1 = VehicleView(track_id=11, car_id=1, position_mm=(100.0, 200.0),
                         heading_deg=0.0, heading_source="FRONT_CUSHION",
                         last_obs_time=1.0)
        v2 = VehicleView(track_id=22, car_id=2, position_mm=(300.0, 400.0),
                         heading_deg=90.0, heading_source="FRONT_CUSHION",
                         last_obs_time=1.0)
        state = SimpleNamespace(frame_index=9, timestamp=1.0, fps=4.0)

        p._record_pose([v2, v1], state, time.monotonic())

        self.assertEqual([row["car_id"] for row in rows], [1, 2])
        self.assertEqual({row["frame_id"] for row in rows}, {1})


if __name__ == "__main__":
    unittest.main()
