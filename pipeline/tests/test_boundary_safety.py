"""Measurement-aware physical-footprint boundary safety tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from control.auto_host_runner import MissionStatus
from pipeline.runner import ParkingPipeline, VehicleView


class _Runner:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.status = MissionStatus.RUNNING
        self.requests = []
        self.prepared = 0
        self.parked_calls = 0
        self.mission = SimpleNamespace(
            request_replan=lambda reason: self.requests.append(reason))

    def stop(self) -> None:
        self.stop_calls += 1

    def prepare_route_switch(self) -> None:
        self.prepared += 1

    def confirm_parked(self) -> None:
        self.parked_calls += 1
        self.status = MissionStatus.PARKED


class _Dashboard:
    def __init__(self) -> None:
        self.events = []

    def push_event(self, name, **fields) -> None:
        self.events.append((name, fields))


class TestBoundaryHardMargin(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ParkingPipeline.__new__(ParkingPipeline)
        self.pipeline.config = SimpleNamespace(
            boundary_hard_margin_mm=20.0,
            lot_width_mm=1200.0,
            lot_height_mm=1200.0,
            heading_min_move_mm=30.0,
            boundary_prediction_horizon_s=0.5,
            boundary_measurement_uncertainty_mm=10.0,
            boundary_uncertain_confirm_frames=2,
            boundary_uncertain_increase_mm=1.0,
            boundary_terminal_confirm_frames=2,
            parking_mode="rear",
            stationary_tolerance_mm=5.0,
            stationary_window=3,
        )
        self.runner = _Runner()
        self.pipeline.auto_hosts = {1: self.runner}
        self.pipeline.dashboard = _Dashboard()
        self.pipeline._boundary_soft = set()
        self.pipeline._boundary_motion = {}
        self.pipeline._parking_stage = {}
        self.pipeline.server = SimpleNamespace(stop_control=lambda car_id: None)
        self.pipeline.on_event_record = None
        self.pipeline._boundary_uncertain = set()
        self.pipeline._boundary_hard = set()
        self.pipeline._boundary_terminal_streak = {}
        self.pipeline._boundary_uncertain_trend = {}
        self.pipeline._auto_host_slot = {1: "B1"}
        self.pipeline._on_parked = lambda car_id, slot_id: None

    def check_left_overflow(self, overflow_mm: float) -> None:
        # heading=0: physical footprint의 왼쪽 끝은 center_x - 125mm.
        view = VehicleView(track_id=7, car_id=1,
                           position_mm=(125.0 - overflow_mm, 600.0),
                           heading_deg=0.0)
        self.pipeline._check_boundary(view)

    def test_ten_mm_does_not_stop(self) -> None:
        self.check_left_overflow(10.0)
        self.assertEqual(self.runner.stop_calls, 0)
        self.assertEqual(self.pipeline.dashboard.events, [])

    def test_exactly_twenty_mm_does_not_stop(self) -> None:
        self.check_left_overflow(20.0)
        self.assertEqual(self.runner.stop_calls, 0)
        self.assertEqual(self.pipeline.dashboard.events, [])

    def test_exactly_thirty_mm_is_uncertain_not_hard(self) -> None:
        self.check_left_overflow(30.0)
        self.assertEqual(self.runner.stop_calls, 0)
        name, fields = self.pipeline.dashboard.events[0]
        self.assertEqual(name, "boundary_uncertain")
        self.assertEqual(fields["overflow_mm"], 30.0)

    def test_over_thirty_mm_running_always_hard_stops(self) -> None:
        for overflow in (30.1, 44.8, 51.6):
            with self.subTest(overflow=overflow):
                self.setUp()
                self.check_left_overflow(overflow)
                self.assertEqual(self.runner.stop_calls, 1)
                name, fields = self.pipeline.dashboard.events[0]
                self.assertEqual(name, "boundary_stop")
                self.assertEqual(fields["overflow_mm"], overflow)

    def test_single_22mm_running_frame_is_uncertain_only(self) -> None:
        view = VehicleView(track_id=7, car_id=1,
                           position_mm=(103.0, 600.0), heading_deg=0.0,
                           last_obs_time=10.0)
        self.pipeline._check_boundary(view)
        self.pipeline._check_boundary(view)  # same camera observation at 10 Hz
        self.assertEqual(self.runner.stop_calls, 0)
        self.assertEqual(self.runner.requests, [])
        self.assertEqual(
            [e[0] for e in self.pipeline.dashboard.events],
            ["boundary_uncertain"])

    def test_161237_22mm_after_predictive_replan_is_not_hard_fault(self) -> None:
        self.runner.status = MissionStatus.REPLAN_REQUIRED
        self.pipeline._check_boundary(VehicleView(
            track_id=7, car_id=1, position_mm=(102.8, 600.0),
            heading_deg=0.0, last_obs_time=32.140))
        self.assertEqual(self.runner.stop_calls, 0)
        self.assertNotIn("boundary_stop",
                         [e[0] for e in self.pipeline.dashboard.events])
        self.assertIn("boundary_uncertain",
                      [e[0] for e in self.pipeline.dashboard.events])

    def test_increasing_uncertain_band_zeroes_within_two_fresh_frames(self) -> None:
        for t, overflow in ((10.0, 22.0), (10.25, 24.0)):
            self.pipeline._check_boundary(VehicleView(
                track_id=7, car_id=1,
                position_mm=(125.0 - overflow, 600.0), heading_deg=0.0,
                last_obs_time=t))
        self.assertEqual(self.runner.stop_calls, 0)
        self.assertEqual(self.runner.requests, ["BOUNDARY_UNCERTAIN_TREND"])
        self.assertEqual(self.runner.prepared, 1)
        self.assertIn("boundary_uncertain_stop",
                      [e[0] for e in self.pipeline.dashboard.events])

    def test_154028_terminal_measurement_band_is_zero_and_not_spam(self) -> None:
        self.runner.status = MissionStatus.DONE
        self.pipeline._parking_stage[1] = "WAIT_SAFE_RECOVERY"
        # Real stable tail: rear-right footprint corner is ~23.7 mm left of x=0.
        view = VehicleView(track_id=2, car_id=1,
                           position_mm=(122.1, 650.9), heading_deg=328.7,
                           heading_source="FRONT_CUSHION")
        self.pipeline._check_boundary(view)
        self.pipeline._check_boundary(view)
        self.assertEqual(self.runner.stop_calls, 0)
        uncertain = [e for e in self.pipeline.dashboard.events
                     if e[0] == "boundary_uncertain"]
        self.assertEqual(len(uncertain), 1)

    def test_154028_single_30mm_terminal_spike_is_debounced(self) -> None:
        self.runner.status = MissionStatus.DONE
        self.pipeline._parking_stage[1] = "WAIT_SAFE_RECOVERY"
        spike = VehicleView(track_id=2, car_id=1,
                            position_mm=(115.0, 648.2), heading_deg=330.7,
                            heading_source="FRONT_CUSHION")
        settled = VehicleView(track_id=2, car_id=1,
                              position_mm=(116.4, 648.1), heading_deg=330.4,
                              heading_source="FRONT_CUSHION")
        self.pipeline._check_boundary(spike)    # ~30.7 mm, one observation
        self.pipeline._check_boundary(settled)  # back below 30 mm
        self.assertEqual(self.runner.stop_calls, 0)

    def test_terminal_true_excursion_requires_two_frames_then_stops_once(self) -> None:
        self.runner.status = MissionStatus.PARKED
        self.pipeline._parking_stage[1] = "PARKING"
        view = VehicleView(track_id=2, car_id=1,
                           position_mm=(85.0, 650.0), heading_deg=330.0,
                           heading_source="FRONT_CUSHION")
        self.pipeline._check_boundary(view)
        self.assertEqual(self.runner.stop_calls, 0)
        self.pipeline._check_boundary(view)
        self.pipeline._check_boundary(view)
        self.assertEqual(self.runner.stop_calls, 1)
        hard = [e for e in self.pipeline.dashboard.events
                if e[0] == "boundary_stop"]
        self.assertEqual(len(hard), 1)

    def test_setup_done_cannot_be_confirmed_as_parked(self) -> None:
        self.runner.status = MissionStatus.DONE
        self.pipeline._parking_stage[1] = "WAIT_SAFE_RECOVERY"
        view = VehicleView(track_id=2, car_id=1,
                           position_mm=(122.1, 650.9), heading_deg=328.7)
        view.recent.extend([(122.1, 650.9)] * 3)
        self.pipeline._check_auto_host_parked(view)
        self.assertEqual(self.runner.parked_calls, 0)

    def test_rear_final_done_can_still_be_confirmed_as_parked(self) -> None:
        self.runner.status = MissionStatus.DONE
        self.pipeline._parking_stage[1] = "PARKING"
        view = VehicleView(track_id=2, car_id=1,
                           position_mm=(425.0, 1050.0), heading_deg=270.0)
        view.recent.extend([(425.0, 1050.0)] * 3)
        self.pipeline._check_auto_host_parked(view)
        self.assertEqual(self.runner.parked_calls, 1)

    def test_actual_map_exit_runs_stop_predictively_before_hard_boundary(self) -> None:
        self.pipeline._parking_stage[1] = "PARKING"
        fixtures = (
            # run_20260814_144749: second pose still has 26.3 mm physical
            # clearance but its measured motion crosses HARD_BOUNDARY in 0.5 s.
            ((24.797, 742.7, 384.3, 329.2),
             (27.094, 988.9, 168.7, 313.3)),
            # run_20260814_144902: footprint is exactly at the map edge, still
            # before the first negative-clearance observation at t=23.437.
            ((22.437, 712.8, 204.9, 314.4),
             (22.875, 744.0, 168.1, 310.9),
             (23.094, 759.6, 144.5, 308.6)),
        )
        for observations in fixtures:
            with self.subTest(last=observations[-1]):
                self.pipeline._boundary_motion.clear()
                self.runner.requests.clear()
                self.runner.prepared = 0
                for t, x, y, h in observations:
                    self.pipeline._check_boundary(VehicleView(
                        track_id=7, car_id=1, position_mm=(x, y),
                        heading_deg=h, heading_source="FRONT_CUSHION",
                        last_obs_time=t))
                self.assertEqual(self.runner.requests,
                                 ["PREDICTED_BOUNDARY"])
                self.assertEqual(self.runner.prepared, 1)
                self.assertEqual(self.runner.stop_calls, 0)


if __name__ == "__main__":
    unittest.main()
