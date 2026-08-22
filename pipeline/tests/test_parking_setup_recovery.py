"""Rear parking setup recovery의 production 단계 전이 회귀 테스트."""

from __future__ import annotations

import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

from control.auto_host_runner import MissionStatus
from parking.waypoints import (InfeasibleRouteError, choose_rear_candidate,
                               default_slot_specs, plan_setup_recovery)
from pipeline.runner import ParkingPipeline, VehicleView


class _Runner:
    def __init__(self) -> None:
        self.loaded = []
        self.stopped = False
        self.replan_reason = "HEADING_OUT_OF_TOLERANCE"

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        self.stopped = True


class _Dashboard:
    def __init__(self) -> None:
        self.events = []

    def push_event(self, name, **fields) -> None:
        self.events.append((name, fields))


class _Server:
    def __init__(self) -> None:
        self.zeroed = []

    def stop_control(self, car_id) -> None:
        self.zeroed.append(car_id)


class TestParkingSetupRecoveryTransition(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ParkingPipeline.__new__(ParkingPipeline)
        self.pipeline.config = SimpleNamespace(
            parking_mode="rear", max_parking_recovery_attempts=3,
            stationary_tolerance_mm=15.0, stationary_window=3)
        self.pipeline.auto_hosts = {1: _Runner()}
        self.pipeline._auto_host_slot = {1: "B1"}
        self.pipeline.track_of_car = {1: 7}
        self.pipeline.views = {
            7: VehicleView(track_id=7, car_id=1,
                           position_mm=(425.0, 600.0), heading_deg=0.0,
                           heading_source="FRONT_CUSHION")
        }
        for view in self.pipeline.views.values():
            view.recent.extend([view.position_mm] * 3)
        self.pipeline._parking_stage = {1: "DRIVING"}
        self.pipeline._parking_setup_wait = {}
        self.pipeline._parking_plan_wait = {}
        self.pipeline._parking_recovery_attempts = {}
        self.pipeline._initial_pose_samples = {}
        self.pipeline._heading_wait_state = {}
        self.pipeline.dashboard = _Dashboard()
        self.pipeline.server = _Server()
        self.pipeline.orchestrator = SimpleNamespace(next_route_id=lambda: 41)
        self.pipeline.on_route_load = None
        self.pipeline._auto_host_route = {}
        self.pipeline.on_event_record = None
        self.pipeline._trajectory_safe = lambda view, route, **kwargs: True
        self.pipeline._load_direct_rear_replan = lambda car_id, view: False

    def test_infeasible_rear_plan_loads_setup_instead_of_failing(self) -> None:
        setup = [SimpleNamespace(phase="RECOVERY")]
        self.pipeline._build_route = lambda *_: (_ for _ in ()).throw(
            InfeasibleRouteError("B1", "single arc infeasible"))
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=setup):
            self.assertTrue(self.pipeline._start_rear_parking_stage(1))
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1], setup)
        self.assertFalse(self.pipeline.auto_hosts[1].stopped)

    def test_setup_done_replans_from_new_camera_pose(self) -> None:
        fresh = (218.0, 627.0)
        self.pipeline._parking_stage[1] = "SETUP"
        self.pipeline.views[7].position_mm = fresh
        self.pipeline.views[7].heading_deg = 345.0
        seen = []
        parking = [SimpleNamespace(phase="ENTRY")]

        def build(_spec, view, _route_id):
            seen.append((view.position_mm, view.heading_deg))
            return parking

        self.pipeline._build_route = build
        self.pipeline._on_auto_host_status(1, MissionStatus.RUNNING,
                                           MissionStatus.DONE)
        self.assertEqual(seen, [])
        self.assertEqual(self.pipeline._parking_stage[1],
                         "PARKING_AFTER_SETUP_PENDING")
        self.pipeline.views[7].last_obs_time = 1.0
        self.pipeline._maybe_start_rear_after_stop(self.pipeline.views[7])
        self.assertEqual(seen, [(fresh, 345.0)])
        self.assertEqual(self.pipeline._parking_stage[1], "PARKING")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1], parking)

    def test_parking_replan_waits_for_fresh_pose_then_loads_setup(self) -> None:
        view = self.pipeline.views[7]
        view.last_obs_time = 10.0
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])

        setup = [SimpleNamespace(phase="RECOVERY")]
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=setup):
            self.pipeline._maybe_start_parking_setup(view)
            self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])
            view.last_obs_time = 10.1
            self.pipeline._maybe_start_parking_setup(view)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1], setup)

    def test_reverse_heading_timeout_enters_fresh_pose_setup_recovery(self) -> None:
        view = self.pipeline.views[7]
        view.last_obs_time = 30.0
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline.auto_hosts[1].replan_reason = "REVERSE_HEADING_TIMEOUT"

        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)

        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertEqual(self.pipeline._parking_setup_wait[1], 30.0)
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])

    def test_parking_recovery_attempts_exhaust_to_zero_wait(self) -> None:
        self.pipeline.config.max_parking_recovery_attempts = 1
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_RECOVERY_EXHAUSTED")
        self.assertEqual(self.pipeline.server.zeroed, [1])

    def test_no_safe_setup_stays_zero_and_never_uses_generic_replan(self) -> None:
        view = self.pipeline.views[7]
        view.last_obs_time = 20.0
        self.pipeline._parking_stage[1] = "SETUP_PENDING"
        self.pipeline._parking_setup_wait[1] = 19.0
        self.pipeline._load_parking_setup = lambda car_id, fresh: False
        self.pipeline._replan_auto_host = lambda car_id: self.fail(
            "parking setup failure escaped into generic/legacy replan")
        events = []
        self.pipeline.on_event_record = lambda name, **fields: events.append(
            (name, fields))

        self.pipeline._maybe_start_parking_setup(view)

        self.assertEqual(self.pipeline._parking_stage[1], "WAIT_SAFE_RECOVERY")
        self.assertEqual(self.pipeline.server.zeroed, [1])
        self.assertIn(("FAULT", {"car_id": 1,
                                  "reason": "NO_SAFE_PARKING_RECOVERY"}),
                      events)

    def test_155812_track_churn_rebinds_then_fresh_heading_loads_recovery(self) -> None:
        old = VehicleView(
            track_id=2, car_id=1, slot_id="B1",
            position_mm=(704.6, 350.0), heading_deg=309.6,
            heading_source="TRAJECTORY", last_seen_frame=55,
            last_obs_time=1350.890)
        new = VehicleView(
            track_id=7, position_mm=(746.9, 310.7), heading_deg=None,
            heading_source=None, last_seen_frame=70,
            last_obs_time=1354.578)
        new.recent.extend([new.position_mm] * 3)
        self.pipeline.views = {2: old, 7: new}
        self.pipeline.track_of_car = {1: 2}
        self.pipeline._parking_stage[1] = "SETUP_PENDING"
        self.pipeline._parking_setup_wait[1] = 1350.890
        self.pipeline.config.track_rebind_stale_frames = 8
        self.pipeline.config.track_rebind_max_distance_mm = 150.0
        self.pipeline._lock = threading.RLock()
        removed = []
        self.pipeline.allocator = SimpleNamespace(
            vehicles={
                2: SimpleNamespace(assigned_slot="B1", route=["old"]),
                7: SimpleNamespace(assigned_slot=None, route=[]),
            },
            remove_vehicle=lambda track_id: removed.append(track_id),
        )
        heading_removed = []
        self.pipeline.heading = SimpleNamespace(
            remove=lambda track_id: heading_removed.append(track_id))
        recorded = []
        self.pipeline.on_event_record = lambda name, **fields: recorded.append(
            (name, fields))

        self.pipeline._maybe_rebind_recovery_track(new, frame_index=70)
        self.assertEqual(self.pipeline.track_of_car[1], 7)
        self.assertEqual(new.car_id, 1)
        self.assertEqual(new.slot_id, "B1")
        self.assertIsNone(old.car_id)
        self.assertNotIn(2, self.pipeline.views)
        self.assertEqual(removed, [2])
        self.assertEqual(heading_removed, [2])
        self.assertEqual(recorded[0][0], "TRACK_REBOUND")

        # No heading is copied across the identity boundary, so it remains
        # safely stopped until a genuinely fresh physical heading arrives.
        self.pipeline._maybe_start_parking_setup(new)
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])
        new.heading_deg = 309.6
        new.heading_source = "FRONT_CUSHION"
        setup = [SimpleNamespace(phase="RECOVERY")]
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=setup):
            self.pipeline._maybe_start_parking_setup(new)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1], setup)

    def test_175349_comm_hold_allows_safe_track_rebind(self) -> None:
        old = VehicleView(
            track_id=2, car_id=1, slot_id="B1",
            position_mm=(779.6, 151.7), heading_deg=336.0,
            heading_source="LAST_VALID", last_seen_frame=55,
            last_obs_time=100.0)
        new = VehicleView(
            track_id=15, position_mm=(703.0, 272.0), heading_deg=None,
            heading_source=None, last_seen_frame=70, last_obs_time=104.0)
        self.pipeline.views = {2: old, 15: new}
        self.pipeline.track_of_car = {1: 2}
        self.pipeline._parking_stage[1] = "SETUP"
        self.pipeline._comm_lost = {1}
        self.pipeline._comm_recovery_context = {
            1: {"state": "WAIT_CONNECTION", "track_id": 2, "slot_id": "B1"}
        }
        self.pipeline.config.track_rebind_stale_frames = 8
        self.pipeline.config.track_rebind_max_distance_mm = 150.0
        self.pipeline._lock = threading.RLock()
        self.pipeline.allocator = SimpleNamespace(
            vehicles={
                2: SimpleNamespace(assigned_slot="B1", route=["old"]),
                15: SimpleNamespace(assigned_slot=None, route=[]),
            },
            remove_vehicle=lambda _track_id: None,
        )
        self.pipeline.heading = SimpleNamespace(remove=lambda _track_id: None)

        self.pipeline._maybe_rebind_recovery_track(new, frame_index=70)

        self.assertEqual(self.pipeline.track_of_car[1], 15)
        self.assertEqual(new.car_id, 1)
        self.assertIsNone(new.heading_deg, "stale heading crossed track identity")
        self.assertEqual(
            self.pipeline._comm_recovery_context[1]["track_id"], 15)

    def test_rebound_track_without_heading_times_out_to_explicit_zero_fault(self) -> None:
        view = self.pipeline.views[7]
        view.heading_deg = None
        view.heading_source = None
        view.last_obs_time = 10.0
        self.pipeline._parking_stage[1] = "SETUP_PENDING"
        self.pipeline._parking_setup_wait[1] = 9.0
        recorded = []
        self.pipeline.on_event_record = lambda name, **fields: recorded.append(
            (name, fields))

        self.pipeline._maybe_start_parking_setup(view)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertFalse(self.pipeline.auto_hosts[1].stopped)

        view.last_obs_time = 12.6  # >2.5 s, ten fresh frames at ~4 FPS
        self.pipeline._maybe_start_parking_setup(view)
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_FRESH_HEADING_FAULT")
        self.assertTrue(self.pipeline.auto_hosts[1].stopped)
        self.assertIn("WAIT_FOR_FRESH_HEADING_TIMEOUT",
                      [name for name, _ in recorded])
        self.assertIn(
            ("FAULT", {"car_id": 1, "reason": "FRESH_HEADING_TIMEOUT",
                       "boundary": "PARKING_RECOVERY_REPLAN"}),
            recorded)


class TestBidirectionalSetupSearch(unittest.TestCase):
    def test_real_failure_pose_finds_rear_feasible_terminal_pose(self) -> None:
        spec = default_slot_specs()["B1"]
        recovery = plan_setup_recovery(spec, (792.2, 339.2), 323.8)
        self.assertIsNotNone(recovery)
        self.assertLessEqual(len(recovery.segments), 3)
        candidate, _ = choose_rear_candidate(
            spec, recovery.end_pose[:2], recovery.end_pose[2])
        self.assertIsNotNone(candidate)
        self.assertTrue(all(segment.reverse in (True, False)
                            for segment in recovery.segments))

    def test_obstacle_footprint_blocks_setup_trajectory(self) -> None:
        spec = default_slot_specs()["B1"]
        recovery = plan_setup_recovery(
            spec, (792.2, 339.2), 323.8,
            obstacle_poses=((792.2, 339.2, 323.8),))
        self.assertIsNone(recovery)


if __name__ == "__main__":
    unittest.main()
