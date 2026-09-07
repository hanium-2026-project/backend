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
                           heading_deg=0.0,
                           heading_source="FRONT_CUSHION")
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
                           heading_source="FRONT_CUSHION",
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
            heading_deg=0.0, heading_source="FRONT_CUSHION",
            last_obs_time=32.140))
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
                heading_source="FRONT_CUSHION", last_obs_time=t))
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

    def test_rear_final_done_is_not_parked_by_the_legacy_path(self) -> None:
        """후면주차는 waypoint 도착만으로 PARKED 가 되지 않는다.

        최종 자세가 주차선과 나란한지는 정지 후 fresh pose 로 평가하고
        (FINAL_EVAL_PENDING → PARKED_VERIFY), 그 경로만 PARKED 를 확정한다.
        완벽한 자세여도 이 legacy 경로에서는 확정하지 않는다.
        """
        self.runner.status = MissionStatus.DONE
        self.pipeline._parking_stage[1] = "PARKING"
        view = VehicleView(track_id=2, car_id=1,
                           position_mm=(425.0, 1050.0), heading_deg=270.0)
        view.recent.extend([(425.0, 1050.0)] * 3)
        self.pipeline._check_auto_host_parked(view)
        self.assertEqual(self.runner.parked_calls, 0)

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


class TestBoundaryHeadingProvenance(unittest.TestCase):
    """LAST_VALID heading 으로 정확한 회전 footprint 를 만들지 않는다.

    x/y 는 매 프레임 신선하지만 heading 은 몇 초씩 얼어붙는다. 얼어붙은 값으로
    차체를 회전시키면 방향에 따라 과대·과소평가가 모두 생긴다
    (실측 223536: 실제 약 5mm 초과를 stale 303.7° 로 25.6mm 로 계산).
    """

    def setUp(self) -> None:
        base = TestBoundaryHardMargin()
        base.setUp()
        self.pipeline = base.pipeline
        self.runner = base.runner
        self.pipeline._heading_wait_state = {}
        self.pipeline._heading_wait_started = {}
        self.pipeline._heading_wait_faulted = set()
        self.pipeline._heading_fault_hold = set()
        self.pipeline._boundary_heading_hold = set()
        self.pipeline._initial_pose_samples = {}
        self.pipeline.config.critical_heading_wait_timeout_s = 2.5
        self.pipeline.server = SimpleNamespace(
            stop_control=lambda car_id: self.zeroed.append(car_id))
        self.zeroed = []
        self.events = []
        self.pipeline.on_event_record = (
            lambda n, **f: self.events.append((n, f)))

    def _view(self, x, y, h, source, t=1.0):
        return VehicleView(track_id=7, car_id=1, position_mm=(x, y),
                           heading_deg=h, heading_source=source,
                           last_obs_time=t)

    def _mode(self, x, y, h, source):
        return self.pipeline._boundary_overflow(
            self._view(x, y, h, source), hard_limit_mm=30.0)[0]

    def test_trusted_sources_use_the_exact_rotated_footprint(self) -> None:
        for source in ("FRONT_CUSHION", "TRAJECTORY"):
            with self.subTest(source):
                self.assertEqual(self._mode(600.0, 600.0, 45.0, source),
                                 "EXACT")

    def test_last_valid_and_none_never_use_exact_orientation(self) -> None:
        for source in ("LAST_VALID", None):
            with self.subTest(source):
                self.assertNotEqual(self._mode(600.0, 600.0, 45.0, source),
                                    "EXACT")

    def test_untrusted_but_safe_for_every_orientation_is_not_a_fault(self) -> None:
        """어떤 방향이어도 맵 안이면 불필요한 정지를 만들지 않는다."""
        self.assertEqual(self._mode(600.0, 600.0, 45.0, "LAST_VALID"),
                         "CONSERVATIVE")
        self.pipeline._check_boundary(
            self._view(600.0, 600.0, 45.0, "LAST_VALID"))
        self.assertEqual(self.runner.stop_calls, 0)

    def test_untrusted_and_unsafe_for_every_orientation_is_confirmed(self) -> None:
        """가장 유리한 방향으로 놓아도 넘으면 heading 없이도 확정 위반이다."""
        self.assertEqual(self._mode(-40.0, 600.0, 0.0, "LAST_VALID"),
                         "CONFIRMED")
        self.pipeline._check_boundary(self._view(-40.0, 600.0, 0.0, "LAST_VALID"))
        self.assertEqual(self.runner.stop_calls, 1)

    def test_untrusted_and_orientation_dependent_holds_instead_of_guessing(self) -> None:
        """방향에 따라 갈리면 추측하지 않고 정지 + fresh heading 대기."""
        self.assertEqual(self._mode(100.0, 600.0, 0.0, "LAST_VALID"),
                         "UNCERTAIN")
        self.pipeline._check_boundary(self._view(100.0, 600.0, 0.0, "LAST_VALID"))
        self.assertEqual(self.runner.stop_calls, 0)
        self.assertIn(1, self.zeroed)
        self.assertIn("BOUNDARY_HEADING_UNCERTAIN",
                      [n for n, _ in self.events])

    def test_uncertainty_hold_reuses_the_fresh_heading_contract(self) -> None:
        """새 대기 상태를 만들지 않는다 — 기존 liveness 복구가 그대로 붙는다."""
        self.pipeline._check_boundary(self._view(100.0, 600.0, 0.0, "LAST_VALID"))
        self.assertIn("WAIT_FOR_FRESH_HEADING", [n for n, _ in self.events])

    def test_uncertainty_event_is_not_spammed_every_frame(self) -> None:
        for i in range(4):
            self.pipeline._check_boundary(
                self._view(100.0, 600.0, 0.0, "LAST_VALID", t=1.0 + i * 0.25))
        held = [n for n, _ in self.events if n == "BOUNDARY_HEADING_UNCERTAIN"]
        self.assertEqual(len(held), 1)

    def test_fresh_trusted_heading_returns_to_exact_evaluation(self) -> None:
        self.pipeline._check_boundary(self._view(100.0, 600.0, 0.0, "LAST_VALID"))
        self.assertEqual(
            self._mode(100.0, 600.0, 0.0, "FRONT_CUSHION"), "EXACT")
        self.pipeline._check_boundary(
            self._view(100.0, 600.0, 0.0, "FRONT_CUSHION", t=2.0))
        # heading 이 돌아오면 재-hold 가 가능해야 한다 (dedupe 가 걸리지 않음).
        self.assertNotIn(1, self.pipeline._boundary_heading_hold)

    def test_nominal_parked_pose_is_reachable_under_the_new_contract(self) -> None:
        """보수적 판정이 정상 PARKED 자세를 막으면 안 된다."""
        from parking.waypoints import default_slot_specs
        b1 = default_slot_specs()["B1"]
        self.assertEqual(
            self._mode(b1.center_x, b1.center_y, 270.0, "LAST_VALID"),
            "CONSERVATIVE")
        self.pipeline._check_boundary(
            self._view(b1.center_x, b1.center_y, 270.0, "LAST_VALID"))
        self.assertEqual(self.runner.stop_calls, 0)


class TestPredictiveGuardCoversFinalStages(unittest.TestCase):
    def test_final_stages_are_under_predictive_protection(self) -> None:
        for stage in ("PARKING", "SETUP",
                      "FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE"):
            self.assertIn(stage, ParkingPipeline._PREDICTIVE_GUARD_STAGES)

    def test_waiting_stages_are_not_under_predictive_protection(self) -> None:
        for stage in ("FINAL_EVAL_PENDING", "PARKED_VERIFY", "PARKED"):
            self.assertNotIn(stage, ParkingPipeline._PREDICTIVE_GUARD_STAGES)


class TestPredictiveGuardHeadingProvenance(unittest.TestCase):
    """예측 정지도 신뢰 가능한 방향에서만 의미가 있다.

    run_20260827_234439: LAST_VALID heading 으로 predicted 31.0mm 를 만들어
    FINAL 에서 정지시켰지만, 방향 무관 최악값은 3.5mm 였다. heading 이 얼어붙으면
    heading_rate 가 구조적으로 0 이라 "회전하지 않는다"는 틀린 전제가 된다.
    """

    def setUp(self) -> None:
        base = TestBoundaryHardMargin()
        base.setUp()
        self.pipeline = base.pipeline
        self.runner = base.runner
        self.pipeline._parking_stage[1] = "PARKING"

    def _drive(self, source):
        self.pipeline._boundary_motion.clear()
        self.runner.requests.clear()
        for t, x in ((10.0, 200.0), (10.25, 160.0), (10.5, 120.0)):
            self.pipeline._check_boundary(VehicleView(
                track_id=7, car_id=1, position_mm=(x, 600.0),
                heading_deg=0.0, heading_source=source, last_obs_time=t))
        return list(self.runner.requests)

    def test_trusted_heading_still_predicts(self) -> None:
        self.assertEqual(self._drive("FRONT_CUSHION"), ["PREDICTED_BOUNDARY"])

    def test_stale_heading_never_drives_the_prediction(self) -> None:
        self.assertEqual(self._drive("LAST_VALID"), [])

    def test_stale_heading_is_still_evaluated_conservatively(self) -> None:
        """예측을 끄는 것이 검사를 끄는 것은 아니다."""
        mode, _ = self.pipeline._boundary_overflow(
            VehicleView(track_id=7, car_id=1, position_mm=(-40.0, 600.0),
                        heading_deg=0.0, heading_source="LAST_VALID"),
            hard_limit_mm=30.0)
        self.assertEqual(mode, "CONFIRMED")


class TestBoundaryGuardIsNeverSkipped(unittest.TestCase):
    """물리 경계 감시는 어떤 handler 의 조기 return 으로도 건너뛰면 안 된다.

    run_20260831_231000: ENTRY_STAGING_PENDING 이 1.64초 동안 조기 return 하는
    사이 차가 관성으로 맵 밖 38.6mm 까지 나갔고, BOUNDARY_HARD 는 그 상태가
    끝난 뒤에야 찍혔다.
    """

    def test_boundary_runs_before_every_early_return_handler(self) -> None:
        import inspect
        from pipeline.runner import ParkingPipeline
        src = inspect.getsource(ParkingPipeline._feed_auto_host)
        boundary = src.index("_check_boundary")
        for handler in ("_maybe_resume_comm_recovery",
                        "_maybe_resume_entry_staging"):
            with self.subTest(handler):
                self.assertLess(boundary, src.index(handler),
                                f"{handler} 가 _check_boundary 보다 먼저 return 한다")

    def test_boundary_is_checked_exactly_once_per_frame(self) -> None:
        import inspect
        from pipeline.runner import ParkingPipeline
        src = inspect.getsource(ParkingPipeline._feed_auto_host)
        self.assertEqual(src.count("self._check_boundary(view)"), 1)

    def test_entry_staging_drive_is_under_predictive_protection(self) -> None:
        from pipeline.runner import ParkingPipeline
        self.assertIn("ENTRY_STAGING",
                      ParkingPipeline._PREDICTIVE_GUARD_STAGES)

    def test_every_driving_stage_is_under_predictive_protection(self) -> None:
        """차를 실제로 모는 stage 가 guard set 밖에 있으면 안 된다."""
        from pipeline.runner import ParkingPipeline
        driving = {"PARKING", "SETUP", "ENTRY_STAGING",
                   "FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE"}
        self.assertTrue(driving <= ParkingPipeline._PREDICTIVE_GUARD_STAGES)
