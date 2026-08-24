"""최초 pose 기록과 candidate-slot fallback의 production 회귀 테스트."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from cv.tracker import TrackState
from parking.waypoints import InfeasibleRouteError
from pipeline.runner import ParkingPipeline, VehicleView


class _Server:
    def __init__(self) -> None:
        self.zeroed = []

    def stop_control(self, car_id) -> None:
        self.zeroed.append(car_id)


class _Allocator:
    def __init__(self) -> None:
        self.slot_statuses = np.zeros(8, dtype=np.float32)
        self.reassigned = []
        self.allocate_calls = 0

    def update(self, _track_id, _position_mm) -> None:
        return None

    def allocate(self, _track_id) -> str:
        self.allocate_calls += 1
        return "B1"

    def reassign(self, track_id, slot) -> None:
        self.reassigned.append((track_id, slot))


class _Dashboard:
    def __init__(self) -> None:
        self.events = []

    def push_event(self, name, **fields) -> None:
        self.events.append((name, fields))


def bare_pipeline() -> ParkingPipeline:
    p = ParkingPipeline.__new__(ParkingPipeline)
    p.config = SimpleNamespace(initial_pose_observations=3,
                               initial_pose_stability_mm=30.0,
                               initial_heading_stability_deg=5.0)
    p.allocator = _Allocator()
    p.dashboard = _Dashboard()
    p.server = _Server()
    p._allocation_state = {}
    p._unreachable_slots = set()
    p._last_no_route_warn = 0.0
    p._initial_pose_samples = {}
    p._heading_wait_state = {}
    p._pose_observed_tracks = set()
    p.on_event_record = None
    return p


class TestCandidateFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.p = bare_pipeline()
        self.view = VehicleView(track_id=7, car_id=1,
                                position_mm=(150.0, 600.0), heading_deg=0.0)
        self.events = []
        self.p.on_event_record = lambda name, **kw: self.events.append((name, kw))
        self.p._reject_slot = lambda car, slot, reason: None
        self.p._warn_no_route = lambda view, reason: None
        self.p._trajectory_safe = lambda view, route, **kwargs: True

    def test_first_slot_infeasible_selects_second_feasible(self) -> None:
        def build(spec, _view, _route_id):
            if spec.slot_id == "A1":
                raise InfeasibleRouteError("A1", "too tight")
            return [spec.slot_id]

        self.p._build_route = build
        selected, route = self.p._feasible_route(self.view, "A1", 9)
        self.assertNotEqual(selected, "A1")
        self.assertEqual(route, [selected])
        self.assertIn((7, selected), self.p.allocator.reassigned)
        names = [name for name, _ in self.events]
        self.assertIn("SLOT_CANDIDATE", names)
        self.assertIn("SLOT_REJECTED", names)

    def test_all_slots_infeasible_stays_alive_and_zeroes_control(self) -> None:
        self.p._build_route = lambda spec, view, rid: (_ for _ in ()).throw(
            InfeasibleRouteError(spec.slot_id, "no route"))
        selected, route = self.p._feasible_route(self.view, "A1", 10)
        self.assertIsNone(selected)
        self.assertIsNone(route)
        self.assertEqual(self.p._allocation_state[1], "WAIT_NO_FEASIBLE_SLOT")
        self.assertEqual(self.p.server.zeroed, [1])
        self.assertIn("SLOT_WAIT", [name for name, _ in self.events])

    def test_infeasible_candidate_never_escapes_as_exception(self) -> None:
        self.p._build_route = lambda spec, view, rid: (_ for _ in ()).throw(
            InfeasibleRouteError(spec.slot_id, "blocked"))
        try:
            self.p._feasible_route(self.view, "A1", 11)
        except InfeasibleRouteError as exc:  # pragma: no cover
            self.fail(f"candidate exception escaped: {exc}")

    def test_reject_event_contains_reason(self) -> None:
        calls = 0

        def build(spec, _view, _route_id):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise InfeasibleRouteError(spec.slot_id, "radius 610")
            return [spec.slot_id]

        self.p._build_route = build
        self.p._feasible_route(self.view, "A1", 12)
        rejected = [fields for name, fields in self.events
                    if name == "SLOT_REJECTED"]
        self.assertEqual(rejected[0]["reason"], "radius 610")


class TestPoseBeforeMission(unittest.TestCase):
    def test_unbound_pose_is_recorded_before_route_exists(self) -> None:
        p = bare_pipeline()
        p._frame_seq = 0
        p._prev_frame_index = None
        p._dropped_frames = 0
        p.last_pose_rec = None
        rows = []
        events = []
        p.on_pose_record = rows.append
        p.on_event_record = lambda name, **kw: events.append((name, kw))
        view = VehicleView(track_id=7, car_id=None, position_mm=(151.0, 599.0),
                           heading_deg=1.0, last_obs_time=100.0)
        state = TrackState(frame_index=1, timestamp=100.0, detections=[],
                           fps=4.2, frame_size=(640, 480))
        p._record_pose([view], state, 100.0)
        self.assertEqual(rows[0]["x_mm"], 151.0)
        self.assertIsNone(rows[0]["car_id"])
        self.assertIn("POSE_OBSERVED", [name for name, _ in events])


class TestStableInitialPose(unittest.TestCase):
    def test_one_or_two_fresh_frames_are_not_ready(self) -> None:
        p = bare_pipeline()
        view = VehicleView(track_id=7, position_mm=(150.0, 600.0), heading_deg=0.0)
        for t in (1.0, 2.0):
            view.last_obs_time = t
            self.assertFalse(p._initial_pose_ready(view))

    def test_stable_third_fresh_pose_is_ready(self) -> None:
        p = bare_pipeline()
        view = VehicleView(track_id=7, position_mm=(150.0, 600.0), heading_deg=0.0)
        results = []
        for t, x in ((1.0, 150.0), (2.0, 152.0), (3.0, 151.0)):
            view.last_obs_time = t
            view.position_mm = (x, 600.0)
            results.append(p._initial_pose_ready(view))
        self.assertEqual(results, [False, False, True])

    def test_transient_movement_resets_the_streak(self) -> None:
        p = bare_pipeline()
        view = VehicleView(track_id=7, position_mm=(150.0, 600.0), heading_deg=0.0)
        for t in (1.0, 2.0):
            view.last_obs_time = t
            self.assertFalse(p._initial_pose_ready(view))
        view.last_obs_time = 3.0
        view.position_mm = (250.0, 600.0)
        self.assertFalse(p._initial_pose_ready(view))

    @staticmethod
    def _mission_pipeline() -> tuple[ParkingPipeline, VehicleView]:
        p = bare_pipeline()
        p.config.manual_only = False
        p.config.entry_nodes = ("junction",)
        p.config.alloc_retry_frames = 10
        p.config.control_mode = "auto-host"
        p._frame_seq = 20
        p._last_alloc_frame = {}
        p._manual_shell_starting = set()
        p._car_id = 1
        p.orchestrator = SimpleNamespace(missions={}, next_route_id=lambda: 22)
        p._feasible_route = lambda view, slot, route_id: (
            slot, [SimpleNamespace(node="entry")])
        p._start_auto_host = lambda car_id, slot, route, **kwargs: True
        p._emit_route = lambda route, **kwargs: None
        view = VehicleView(track_id=7, car_id=1, node="junction",
                           position_mm=(150.0, 600.0), heading_deg=0.0,
                           heading_source="FRONT_CUSHION")
        return p, view

    def test_transient_frames_do_not_create_mission(self) -> None:
        p, view = self._mission_pipeline()
        for timestamp in (1.0, 2.0):
            view.last_obs_time = timestamp
            p._ensure_mission(view, int(timestamp))
        self.assertEqual(p.allocator.allocate_calls, 0)
        self.assertEqual(p._allocation_state[1], "WAIT_STABLE_POSE")
        self.assertEqual(p.server.zeroed, [1, 1])

    def test_stable_pose_creates_mission_on_third_frame(self) -> None:
        p, view = self._mission_pipeline()
        for timestamp in (1.0, 2.0, 3.0):
            view.last_obs_time = timestamp
            p._ensure_mission(view, int(timestamp))
        self.assertEqual(p.allocator.allocate_calls, 1)
        self.assertEqual(view.slot_id, "B1")
        self.assertEqual(p._allocation_state[1], "ROUTE_LOADED")

    def test_last_valid_heading_never_starts_initial_mission(self) -> None:
        p, view = self._mission_pipeline()
        view.heading_deg = 206.9
        view.heading_source = "LAST_VALID"
        for timestamp in (1.0, 2.0, 3.0, 4.0):
            view.last_obs_time = timestamp
            p._ensure_mission(view, int(timestamp))
        self.assertEqual(p.allocator.allocate_calls, 0)
        self.assertEqual(p._allocation_state[1], "WAIT_FOR_FRESH_HEADING")


if __name__ == "__main__":
    unittest.main()
