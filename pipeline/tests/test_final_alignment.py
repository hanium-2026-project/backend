"""FINAL_ALIGNMENT / FINAL_STRAIGHT_REVERSE / PARKED 검증 회귀.

계약:
    후면주차 route 완료 → STOP → fresh Pose → 최종 자세 평가
        충분함   → PARKED_VERIFY → fresh 관측 N회 → PARKED
        비스듬함 → FINAL_ALIGNMENT → STOP → fresh Pose
                 → FINAL_STRAIGHT_REVERSE → STOP → fresh Pose → PARKED

기하(실측): 슬롯 200x300mm, 차량 250x150mm → 정렬해도 좌우 여유 25mm.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from control.auto_host_runner import MissionStatus
from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose
from host_control.producers import AutoControlProducer
from integration.backend_adapter import waypoint_from_backend
from parking.final_alignment import (ALIGNED_HEADING_TOLERANCE_DEG,
                                     ALIGN_GOAL_LATERAL_MM,
                                     ALIGN_TERMINAL_HEADING_TOLERANCE_DEG,
                                     ALIGN_TERMINAL_POSITION_TOLERANCE_MM,
                                     in_final_region,
                                     alignment_staging_pose,
                                     build_final_alignment_waypoints,
                                     build_final_straight_reverse_waypoints,
                                     evaluate_final_pose,
                                     footprint_overflow_mm,
                                     rear_parked_heading_deg, to_slot_local)
from parking.waypoints import (PHASE_DEFAULTS,
                               build_setup_recovery_waypoints,
                               default_slot_specs)
from parking.trajectory_safety import validate_trajectory
from pipeline.runner import ParkingPipeline, VehicleView

B1 = default_slot_specs()["B1"]
A1 = default_slot_specs()["A1"]


class TestSlotLocalFrame(unittest.TestCase):
    def test_rear_parked_heading_is_opposite_to_drive_in(self) -> None:
        # 실측 route: B1 FINAL waypoint 의 target_heading 은 270°.
        self.assertEqual(rear_parked_heading_deg(B1), 270.0)
        self.assertEqual(rear_parked_heading_deg(A1), 90.0)

    def test_slot_centre_is_the_local_origin(self) -> None:
        for spec in (B1, A1):
            local = to_slot_local(spec, spec.center_x, spec.center_y,
                                  rear_parked_heading_deg(spec))
            self.assertAlmostEqual(local.depth_mm, 0.0)
            self.assertAlmostEqual(local.lateral_mm, 0.0)
            self.assertAlmostEqual(local.heading_err_deg, 0.0)

    def test_depth_is_positive_deeper_into_the_slot(self) -> None:
        # B1 은 위쪽(y 증가)이 슬롯 안쪽이다.
        deeper = to_slot_local(B1, B1.center_x, B1.center_y + 50.0, 270.0)
        self.assertGreater(deeper.depth_mm, 0.0)
        # A1 은 아래쪽(y 감소)이 안쪽이다 — 같은 로직이 반대 방향에도 성립.
        deeper_a = to_slot_local(A1, A1.center_x, A1.center_y - 50.0, 90.0)
        self.assertGreater(deeper_a.depth_mm, 0.0)

    def test_overflow_splits_lateral_from_depth(self) -> None:
        # 덜 들어간 차는 슬롯 '앞쪽'으로 넘친다 — 옆으로 넘치는 게 아니다.
        lateral, depth = footprint_overflow_mm(B1, 425.0, 900.0, 270.0)
        self.assertEqual(lateral, 0.0)
        self.assertGreater(depth, 0.0)
        # 옆으로 밀린 차는 반대다.
        lateral, depth = footprint_overflow_mm(B1, 505.0, 1050.0, 270.0)
        self.assertGreater(lateral, 0.0)
        self.assertEqual(depth, 0.0)


class TestFinalPoseEvaluation(unittest.TestCase):
    def test_perfect_pose_is_parked(self) -> None:
        self.assertTrue(evaluate_final_pose(B1, 425.0, 1050.0, 270.0).parked)

    def test_skewed_pose_requires_alignment(self) -> None:
        v = evaluate_final_pose(B1, 425.0, 1050.0, 258.0)
        self.assertEqual(v.action, "ALIGN")
        self.assertEqual(v.reason, "HEADING_NOT_PARALLEL")

    def test_heading_tolerance_matches_slot_geometry(self) -> None:
        """125·sin θ + 75·cos θ <= 100 이 슬롯 폭 한계다."""
        r = math.radians(ALIGNED_HEADING_TOLERANCE_DEG)
        half_extent = 125.0 * math.sin(r) + 75.0 * math.cos(r)
        self.assertLess(half_extent, B1.width / 2.0)

    def test_shallow_pose_asks_for_straight_reverse_not_alignment(self) -> None:
        v = evaluate_final_pose(B1, 425.0, 950.0, 270.0)
        self.assertEqual(v.action, "STRAIGHT_REVERSE")
        self.assertEqual(v.reason, "NOT_FULLY_ENTERED")

    def test_lateral_offset_requires_alignment(self) -> None:
        # 직선 후진으로는 횡오차가 절대 줄지 않는다.
        v = evaluate_final_pose(B1, 495.0, 1050.0, 270.0)
        self.assertEqual(v.action, "ALIGN")
        self.assertEqual(v.reason, "LATERAL_OFFSET")

    def test_too_deep_requires_alignment(self) -> None:
        v = evaluate_final_pose(B1, 425.0, 1120.0, 270.0)
        self.assertEqual(v.action, "ALIGN")
        self.assertEqual(v.reason, "TOO_DEEP")

    def test_nose_in_is_never_parked(self) -> None:
        self.assertFalse(evaluate_final_pose(B1, 425.0, 1050.0, 90.0).parked)

    def test_works_for_the_opposite_slot_row(self) -> None:
        self.assertTrue(evaluate_final_pose(A1, 425.0, 150.0, 90.0).parked)
        self.assertEqual(
            evaluate_final_pose(A1, 425.0, 150.0, 78.0).action, "ALIGN")

    def test_171158_physical_containment_is_not_vetoed_by_center_band(self) -> None:
        """1.1mm overflow는 10mm uncertainty 안이며 heading/depth도 정상이다."""
        a2 = default_slot_specs()["A2"]
        verdict = evaluate_final_pose(a2, 675.7, 171.7, 89.8)
        self.assertAlmostEqual(verdict.local.depth_mm, -21.7, places=1)
        self.assertAlmostEqual(verdict.local.lateral_mm, -25.7, places=1)
        self.assertAlmostEqual(verdict.local.heading_err_deg, -0.2, places=1)
        self.assertAlmostEqual(verdict.lateral_overflow_mm, 1.1, delta=0.1)
        self.assertTrue(verdict.parked)

    def test_170954_real_slot_and_map_excursion_is_never_parked(self) -> None:
        """약 51mm 깊이 침범은 measurement uncertainty로 흡수하지 않는다."""
        verdict = evaluate_final_pose(B1, 451.3, 1125.7, 269.9)
        self.assertGreater(verdict.depth_overflow_mm, 50.0)
        self.assertFalse(verdict.parked)
        self.assertEqual(verdict.action, "ALIGN")

    def test_known_good_002703_remains_parked(self) -> None:
        a2 = default_slot_specs()["A2"]
        self.assertTrue(evaluate_final_pose(a2, 655.7, 158.7, 91.2).parked)

    def test_known_good_022217_remains_parked(self) -> None:
        a3 = default_slot_specs()["A3"]
        self.assertTrue(evaluate_final_pose(a3, 876.4, 161.0, 88.3).parked)

    def test_center_band_still_guards_shallow_straight_reverse(self) -> None:
        """PARKED veto만 제거했다. 덜 들어간 차의 correction gate는 유지한다."""
        verdict = evaluate_final_pose(B1, 451.0, 950.0, 270.0)
        self.assertEqual(verdict.action, "ALIGN")
        self.assertEqual(verdict.reason, "LATERAL_OFFSET")


class TestFinalRoutes(unittest.TestCase):
    def test_straight_reverse_has_no_steering(self) -> None:
        staging = alignment_staging_pose(B1)
        wps = build_final_straight_reverse_waypoints(
            B1, 9, from_pose=(staging[0], staging[1]))
        self.assertEqual(len(wps), 1)
        wp = wps[0]
        self.assertEqual(wp.curvature, 0.0)
        self.assertEqual(wp.motion_direction, "REVERSE")
        self.assertTrue(wp.is_final)
        self.assertAlmostEqual(wp.x, B1.center_x, places=6)
        # 목표점은 슬롯 중심선 위이되, 제어기 정지 거리만큼 더 안쪽을 겨눈다.
        local = to_slot_local(B1, wp.x, wp.y, wp.target_heading_deg)
        self.assertAlmostEqual(local.lateral_mm, 0.0, places=6)
        self.assertGreater(local.depth_mm, 0.0)
        self.assertEqual(wp.target_heading_deg, rear_parked_heading_deg(B1))

    def test_straight_reverse_is_empty_when_already_deep(self) -> None:
        self.assertEqual(
            build_final_straight_reverse_waypoints(
                B1, 9, from_pose=(B1.center_x, B1.center_y)), [])

    def test_alignment_route_ends_lined_up_in_front_of_the_slot(self) -> None:
        for label, pose, heading in (
                ("skew in slot", (425.0, 1020.0), 255.0),
                ("skew shallow", (430.0, 900.0), 245.0),
                ("lateral 80mm", (505.0, 1000.0), 270.0)):
            with self.subTest(label):
                wps = build_final_alignment_waypoints(
                    B1, 9, from_pose=pose, from_heading_deg=heading)
                self.assertTrue(wps, f"{label}: 정렬 기동을 못 만들었다")
                end = wps[-1]
                local = to_slot_local(B1, end.x, end.y,
                                      end.target_heading_deg)
                self.assertLessEqual(abs(local.heading_err_deg),
                                     ALIGNED_HEADING_TOLERANCE_DEG)
                self.assertLess(local.depth_mm, 0.0,
                                f"{label}: 슬롯 밖으로 나오지 않았다")

    def test_alignment_goal_is_stricter_than_parked_tolerance(self) -> None:
        """같거나 느슨하면 정렬 → 재평가 → 정렬 로 진동한다."""
        from parking.final_alignment import ALIGN_GOAL_HEADING_DEG
        self.assertLess(ALIGN_GOAL_HEADING_DEG,
                        ALIGNED_HEADING_TOLERANCE_DEG)

    def test_alignment_terminal_tolerance_guarantees_acquisition_band(self) -> None:
        """계획 종점 오차 + DONE 허용오차가 postcondition을 넘지 않는다."""
        wps = build_final_alignment_waypoints(
            B1, 10, from_pose=(505.0, 1000.0),
            from_heading_deg=270.0)
        self.assertTrue(wps)
        end = wps[-1]
        local = to_slot_local(B1, end.x, end.y, end.target_heading_deg)
        self.assertLessEqual(
            abs(local.lateral_mm) + end.position_tolerance_cm * 10.0,
            20.0 + 1e-6)
        self.assertLessEqual(
            abs(local.heading_err_deg) + end.heading_tolerance_deg,
            ALIGNED_HEADING_TOLERANCE_DEG + 1e-6)
        self.assertEqual(end.position_tolerance_cm,
                         ALIGN_TERMINAL_POSITION_TOLERANCE_MM / 10.0)
        self.assertEqual(end.heading_tolerance_deg,
                         ALIGN_TERMINAL_HEADING_TOLERANCE_DEG)

    def test_only_alignment_terminal_uses_precise_tolerance(self) -> None:
        wps = build_final_alignment_waypoints(
            B1, 11, from_pose=(505.0, 1000.0),
            from_heading_deg=270.0)
        self.assertTrue(wps)
        self.assertTrue(all(
            wp.position_tolerance_cm
            == PHASE_DEFAULTS["RECOVERY"]["position_tolerance_cm"]
            for wp in wps[:-1]))
        self.assertEqual(wps[-1].position_tolerance_cm, 1.0)

    def test_old_52mm_early_done_pose_no_longer_completes_terminal(self) -> None:
        """171158 route12는 과거 목표에서 52.8mm 떨어져도 80mm로 DONE이었다."""
        wps = build_final_alignment_waypoints(
            B1, 13, from_pose=(505.0, 1000.0),
            from_heading_deg=270.0)
        self.assertTrue(wps)
        target = waypoint_from_backend(wps[-1])
        result = AutoControlProducer(
            ControllerConfig(allow_reverse=True)).compute(
                Pose(target.x_mm - 52.8, target.y_mm,
                     target.target_heading_deg, timestamp=10.0,
                     heading_source="FRONT_CUSHION"),
                target, now=10.0)
        self.assertFalse(result.arrived)

    def test_general_recovery_keeps_80mm_tolerance(self) -> None:
        wps = build_setup_recovery_waypoints(
            B1, 12, from_pose=(505.0, 1000.0),
            from_heading_deg=270.0,
            goal_test=lambda pose: pose[1] < 900.0)
        self.assertTrue(wps)
        self.assertTrue(all(
            wp.position_tolerance_cm
            == PHASE_DEFAULTS["RECOVERY"]["position_tolerance_cm"]
            for wp in wps))

    def test_alignment_goal_margin_is_derived_from_terminal_tolerance(self) -> None:
        self.assertEqual(
            ALIGN_GOAL_LATERAL_MM + ALIGN_TERMINAL_POSITION_TOLERANCE_MM,
            20.0)

    def test_222253_exact_a2_alignment_route_is_safe(self) -> None:
        """실차 예방정지 pose 에서 평가기가 고른 correction 전체를 검사한다."""
        slot = default_slot_specs()["A2"]
        pose = (645.8, 155.9, 99.1)
        verdict = evaluate_final_pose(slot, *pose)
        self.assertEqual(verdict.action, "ALIGN")
        route = build_final_alignment_waypoints(
            slot, 2253, from_pose=pose[:2], from_heading_deg=pose[2])
        self.assertTrue(route)
        safety = validate_trajectory(
            route, start_pose=pose, target_slot="A2",
            lot_size_mm=(1200.0, 1200.0))
        self.assertTrue(safety.safe, safety.reason)


class _Runner:
    def __init__(self) -> None:
        self.loaded = []
        self.stopped = False
        self.parked_calls = 0
        self.replan_reason = ""
        self.host = SimpleNamespace(
            authority=SimpleNamespace(is_faulted=False, fault_reason=""),
            re_arm_auto=lambda: None)
        self.scheduler = SimpleNamespace(start=lambda: None, running=True)
        self.last_tick_result = SimpleNamespace(
            command=SimpleNamespace(throttle=0.0))

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def confirm_parked(self) -> None:
        self.parked_calls += 1

    def stop(self) -> None:
        self.stopped = True


class TestFinalAlignmentPipeline(unittest.TestCase):
    def setUp(self) -> None:
        p = ParkingPipeline.__new__(ParkingPipeline)
        self.pipeline = p
        p.config = SimpleNamespace(
            control_mode="auto-host",
            parking_mode="rear", max_parking_recovery_attempts=3,
            max_replan_attempts=3, initial_pose_stability_mm=30.0,
            max_final_alignment_attempts=3, parked_confirm_observations=3,
            stationary_tolerance_mm=15.0, stationary_window=3,
            critical_heading_wait_timeout_s=2.5,
            parking_stall_timeout_s=8.0,
            lot_width_mm=1200.0, lot_height_mm=1200.0,
            boundary_hard_margin_mm=20.0,
            controller_config=ControllerConfig())
        self.runner = _Runner()
        p.auto_hosts = {1: self.runner}
        p._auto_host_slot = {1: "B1"}
        p.track_of_car = {1: 7}
        p.views = {}
        p._parking_stage = {1: "FINAL_EVAL_PENDING"}
        p._parking_setup_wait = {}
        p._parking_plan_wait = {1: 0.0}
        p._parking_recovery_attempts = {}
        p._final_alignment_attempts = {}
        p._parked_confirmations = {}
        p._parked_last_obs = {}
        p._parked_obstacles = {}
        p._heading_wait_state = {}
        p._heading_wait_started = {}
        p._heading_wait_faulted = set()
        p._heading_fault_hold = set()
        p._stall_since = {}
        p._last_replan_signature = {}
        p._replan_attempts = {}
        p._auto_host_route = {}
        p._comm_lost = set()
        p._comm_recovery_context = {}
        p._comm_recovery_starting = set()
        p._mode_set = set()
        p._manual_shell_starting = set()
        p.controllers = {}
        p.last_control = {}
        p._last_control_mode = {}
        p._allocation_state = {}
        p.hybrid_controls = {}
        p._lock = __import__("threading").RLock()
        p._route_seq = 100
        def _next_route():
            p._route_seq += 1
            return p._route_seq
        p.orchestrator = SimpleNamespace(next_route_id=_next_route)
        p.dashboard = SimpleNamespace(push_event=lambda *a, **k: None)
        p.zeroed = []
        p.server = SimpleNamespace(
            stop_control=lambda c: p.zeroed.append(c),
            hold_control=lambda c: p.zeroed.append(c),
            session_identity=lambda _c: ("S-NEW", "B-NEW"))
        p.events = []
        p.on_event_record = lambda n, **f: p.events.append((n, f))
        p.on_route_load = None
        p._trajectory_safe = lambda view, route, **kw: True
        self.parked = []
        p._on_parked = lambda car, slot: self.parked.append((car, slot))

    def _view(self, x, y, h, t=1.0):
        v = VehicleView(track_id=7, car_id=1, position_mm=(x, y),
                        heading_deg=h, heading_source="FRONT_CUSHION",
                        last_obs_time=t)
        v.recent.extend([(x, y)] * 5)
        self.pipeline.views = {7: v}
        return v

    def _reasons(self, name):
        return [f for n, f in self.pipeline.events if n == name]

    def test_aligned_pose_enters_parked_verify(self) -> None:
        v = self._view(425.0, 1050.0, 270.0)
        self.assertTrue(self.pipeline._maybe_evaluate_final_pose(v))
        self.assertEqual(self.pipeline._parking_stage[1], "PARKED_VERIFY")
        self.assertEqual(self.runner.loaded, [])

    def test_skewed_pose_loads_an_alignment_route(self) -> None:
        v = self._view(425.0, 1020.0, 255.0)
        self.assertTrue(self.pipeline._maybe_evaluate_final_pose(v))
        self.assertEqual(self.pipeline._parking_stage[1], "FINAL_ALIGNMENT")
        self.assertTrue(self.runner.loaded)

    def test_shallow_pose_loads_a_straight_reverse(self) -> None:
        v = self._view(425.0, 950.0, 270.0)
        self.assertTrue(self.pipeline._maybe_evaluate_final_pose(v))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "FINAL_STRAIGHT_REVERSE")
        route = self.runner.loaded[-1]
        self.assertEqual(len(route), 1)
        self.assertEqual(route[0].curvature, 0.0)

    def test_parked_needs_repeated_fresh_observations(self) -> None:
        self.pipeline._parking_stage[1] = "PARKED_VERIFY"
        for i in range(1, 3):
            v = self._view(425.0, 1050.0, 270.0, t=float(i))
            self.pipeline._maybe_verify_parked(v)
            self.assertEqual(self.runner.parked_calls, 0)
            self.assertEqual(self.pipeline._parking_stage[1], "PARKED_VERIFY")
        v = self._view(425.0, 1050.0, 270.0, t=3.0)
        self.pipeline._maybe_verify_parked(v)
        self.assertEqual(self.runner.parked_calls, 1)
        self.assertEqual(self.pipeline._parking_stage[1], "PARKED")
        self.assertEqual(self.parked, [(1, "B1")])

    def test_same_frame_is_not_counted_twice(self) -> None:
        self.pipeline._parking_stage[1] = "PARKED_VERIFY"
        v = self._view(425.0, 1050.0, 270.0, t=1.0)
        for _ in range(5):
            self.pipeline._maybe_verify_parked(v)
        self.assertEqual(self.runner.parked_calls, 0)

    def test_verification_failure_returns_to_evaluation(self) -> None:
        self.pipeline._parking_stage[1] = "PARKED_VERIFY"
        self.pipeline._maybe_verify_parked(
            self._view(425.0, 1050.0, 270.0, t=1.0))
        # 다음 관측에서 차가 비스듬한 것으로 드러났다.
        self.pipeline._maybe_verify_parked(
            self._view(425.0, 1050.0, 255.0, t=2.0))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "FINAL_EVAL_PENDING")
        self.assertEqual(self.runner.parked_calls, 0)
        self.assertTrue(self._reasons("PARKED_VERIFY_FAILED"))

    def test_moving_vehicle_is_never_confirmed_parked(self) -> None:
        self.pipeline._parking_stage[1] = "PARKED_VERIFY"
        v = VehicleView(track_id=7, car_id=1, position_mm=(425.0, 1050.0),
                        heading_deg=270.0, heading_source="FRONT_CUSHION",
                        last_obs_time=1.0)
        v.recent.extend([(425.0, 1050.0), (450.0, 1050.0), (500.0, 1040.0)])
        self.pipeline.views = {7: v}
        self.pipeline._maybe_verify_parked(v)
        self.assertEqual(self.runner.parked_calls, 0)

    def test_untrusted_heading_is_never_confirmed_parked(self) -> None:
        self.pipeline._parking_stage[1] = "PARKED_VERIFY"
        v = self._view(425.0, 1050.0, 270.0)
        v.heading_source = "LAST_VALID"
        self.pipeline._maybe_verify_parked(v)
        self.assertEqual(self.runner.parked_calls, 0)

    def test_alignment_attempts_are_bounded_and_terminal_is_explicit(self) -> None:
        for i in range(self.pipeline.config.max_final_alignment_attempts):
            self.pipeline._parking_stage[1] = "FINAL_EVAL_PENDING"
            self.pipeline._parking_plan_wait[1] = 0.0
            self.pipeline._maybe_evaluate_final_pose(
                self._view(425.0, 1020.0, 255.0, t=float(i + 1)))
        self.pipeline._parking_stage[1] = "FINAL_EVAL_PENDING"
        self.pipeline._parking_plan_wait[1] = 0.0
        self.pipeline._maybe_evaluate_final_pose(
            self._view(425.0, 1020.0, 255.0, t=99.0))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_FINAL_ALIGNMENT_FAILED")
        self.assertTrue(self.runner.stopped)
        self.assertTrue([f for f in self._reasons("FAULT")
                         if f.get("reason") == "FINAL_ALIGNMENT_EXHAUSTED"])

    def test_unsafe_alignment_route_escalates_instead_of_stalling(self) -> None:
        self.pipeline._trajectory_safe = lambda view, route, **kw: False
        v = self._view(425.0, 1020.0, 255.0)
        self.assertTrue(self.pipeline._maybe_evaluate_final_pose(v))
        self.assertNotEqual(self.pipeline._parking_stage[1],
                            "FINAL_EVAL_PENDING")
        self.assertIn(self.pipeline._parking_stage[1],
                      {"SETUP_PENDING", "WAIT_SAFE_RECOVERY",
                       "WAIT_RECOVERY_EXHAUSTED"})

    def test_full_cycle_reaches_parked(self) -> None:
        """LIVENESS: 비스듬 → 정렬 → 직선후진 → PARKED 까지 bounded step."""
        stage_seq = []
        t = 1.0
        for _ in range(30):
            stage = self.pipeline._parking_stage.get(1)
            stage_seq.append(stage)
            if stage == "PARKED":
                break
            if stage in ("FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE"):
                # 기동이 끝나 계획 종점에 도착했다고 본다.
                end = self.runner.loaded[-1][-1]
                self.pipeline._parking_stage[1] = "FINAL_EVAL_PENDING"
                self.pipeline._parking_plan_wait[1] = t
                t += 1.0
                self._view(end.x, end.y, end.target_heading_deg, t=t)
                continue
            t += 1.0
            v = self.pipeline.views.get(7)
            pose = v.position_mm if v else (425.0, 1020.0)
            heading = v.heading_deg if v else 255.0
            view = self._view(pose[0], pose[1], heading, t=t)
            self.pipeline._maybe_evaluate_final_pose(view)
            self.pipeline._maybe_verify_parked(view)
        self.assertEqual(self.pipeline._parking_stage.get(1), "PARKED",
                         f"stages seen: {stage_seq}")
        self.assertEqual(self.runner.parked_calls, 1)
        self.assertEqual(stage_seq.count("FINAL_ALIGNMENT"), 1,
                         f"alignment repeated: {stage_seq}")

    def test_171158_pipeline_parks_without_alignment(self) -> None:
        """CASE A: 이미 물리적으로 충분한 original FINAL은 correction 0회."""
        self.pipeline._auto_host_slot[1] = "A2"
        view = self._view(675.7, 171.7, 89.8, t=1.0)
        self.assertTrue(self.pipeline._maybe_evaluate_final_pose(view))
        self.assertEqual(self.pipeline._parking_stage[1], "PARKED_VERIFY")
        self.assertEqual(self.runner.loaded, [])
        for t in (2.0, 3.0, 4.0):
            self.pipeline._maybe_verify_parked(
                self._view(675.7, 171.7, 89.8, t=t))
        self.assertEqual(self.pipeline._parking_stage[1], "PARKED")
        self.assertEqual(self.runner.parked_calls, 1)


class TestStraightReverseControlContract(unittest.TestCase):
    """curvature=0 FINAL 후진의 조향 계약.

    RECOVERY 는 phase 로 후진 조향을 고정한다(reverse_straight_phases).
    FINAL 은 phase 로는 고정하지 않지만(PD 유지), 차가 **이미 슬롯 축에 정렬**
    돼 있으면(FINAL 도착 허용오차 안) 11자로 고정한다
    (final_reverse_straight_when_aligned). 정렬된 상태에서 끝점 bearing 을
    쫓는 PD 는 목표에 가까워질수록 조향을 키워 PWM 을 밀어올리고 정지거리를
    늘려 슬롯 뒤(맵 경계)를 넘기 때문이다. 정렬 **밖**에서는 PD 되먹임이
    그대로 살아 있어 크게 틀어진 자세를 보정한다.
    """

    def _controller(self):
        from controller.pose_controller import PoseWaypointController
        cfg = ControllerConfig(allow_reverse=True)
        ctrl = PoseWaypointController(cfg)
        ctrl._motion_direction = MotionDirection.REVERSE
        return cfg, ctrl

    def _target(self):
        from controller.models import Waypoint as CWaypoint
        wp = build_final_straight_reverse_waypoints(
            B1, 9, from_pose=(425.0, 775.0))[0]
        return CWaypoint(
            wp.x, wp.y, target_heading_deg=wp.target_heading_deg,
            position_tolerance_cm=wp.position_tolerance_cm,
            heading_tolerance_deg=wp.heading_tolerance_deg,
            heading_required=True, is_final=True,
            motion_direction=MotionDirection.REVERSE, phase="FINAL",
            curvature=0.0, route_id=9, waypoint_id=1)

    def _steer(self, x, y, h):
        from controller.models import Pose as CPose
        cfg, ctrl = self._controller()
        cmd = ctrl.compute(CPose(x, y, h, timestamp=100.0,
                                 heading_source="FRONT_CUSHION"),
                           self._target(), allow_drive=True, now=100.0)
        return cmd

    def test_final_phase_does_not_lock_reverse_steering(self) -> None:
        cfg = ControllerConfig(allow_reverse=True)
        self.assertFalse(cfg.reverse_steering_locked("FINAL"))
        self.assertTrue(cfg.reverse_steering_locked("RECOVERY"))

    def test_zero_error_gives_centre_steering(self) -> None:
        self.assertAlmostEqual(self._steer(425.0, 775.0, 270.0).steering,
                               0.0, places=6)

    def test_large_lateral_error_still_produces_corrective_steering(self) -> None:
        """도착 허용오차 **밖**(60mm > postol 50mm)이면 PD 되먹임이 살아 있다."""
        right = self._steer(485.0, 775.0, 270.0).steering
        left = self._steer(365.0, 775.0, 270.0).steering
        self.assertNotAlmostEqual(right, 0.0, places=3)
        self.assertNotAlmostEqual(left, 0.0, places=3)
        self.assertLess(right * left, 0.0, "좌우 횡오차의 조향 부호가 같다")

    def test_within_tolerance_caps_the_steering(self) -> None:
        """정렬돼 있으면(도착 허용오차 안) 조향을 cap 한다 (Phase 1).

        FINAL_STRAIGHT_REVERSE 는 evaluate_final_pose 가 lateral<=20mm 일 때만
        선택하므로 실제 진입은 항상 정렬 범위 안이다. 끝점 bearing PD 로
        조향하면 목표에 가까워질수록 조향이 커져 PWM 이 튀고 과주행하므로,
        정렬 범위 안에서는 조향을 cap 한다(0 lock 이 아니라 완만한 crab 보정은
        남긴다). FINAL 자세 품질은 상위 FINAL_POSE_EVAL 이 계속 판정한다.
        """
        cap = ControllerConfig().final_reverse_aligned_steer_cap
        for x in (465.0, 385.0):   # 40mm(도착 허용오차 50mm 안)
            self.assertLessEqual(abs(self._steer(x, 775.0, 270.0).steering),
                                 cap + 1e-6)

    def test_the_lock_can_be_disabled(self) -> None:
        """진단 스위치를 끄면 예전 동작(정렬돼 있어도 조향)으로 돌아간다."""
        from controller.models import Pose as CPose
        from controller.pose_controller import PoseWaypointController
        cfg = ControllerConfig(allow_reverse=True,
                               final_reverse_straight_when_aligned=False)
        ctrl = PoseWaypointController(cfg)
        ctrl._motion_direction = MotionDirection.REVERSE
        cmd = ctrl.compute(CPose(465.0, 775.0, 270.0, timestamp=100.0,
                                 heading_source="FRONT_CUSHION"),
                           self._target(), allow_drive=True, now=100.0)
        self.assertNotAlmostEqual(cmd.steering, 0.0, places=3)

    def test_large_heading_error_produces_corrective_steering(self) -> None:
        """heading 오차가 도착 허용오차(4.5deg) 밖(8deg)이면 조향이 산다."""
        plus = self._steer(425.0, 775.0, 278.0).steering
        minus = self._steer(425.0, 775.0, 262.0).steering
        self.assertNotAlmostEqual(plus, 0.0, places=3)
        self.assertLess(plus * minus, 0.0, "좌우 heading 오차의 조향 부호가 같다")

    def test_reverse_throttle_is_negative(self) -> None:
        self.assertLess(self._steer(425.0, 775.0, 270.0).throttle, 0.0)

    def test_untrusted_heading_never_drives_the_straight_reverse(self) -> None:
        """LAST_VALID 로는 후진을 계속하지 않는다 (blind continue 금지)."""
        from controller.models import Pose as CPose
        cfg, ctrl = self._controller()
        cmd = ctrl.compute(CPose(425.0, 775.0, 270.0, timestamp=100.0,
                                 heading_source="LAST_VALID"),
                           self._target(), allow_drive=True, now=100.0)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.reason, "REVERSE_HEADING_UNSAFE")


class TestStraightReverseGeometry(unittest.TestCase):
    def test_endpoint_is_on_the_slot_centreline_not_the_current_pose(self) -> None:
        """현재 자세를 뒤로 민 점을 겨누면 횡오차가 영원히 남는다."""
        offset = build_final_straight_reverse_waypoints(
            B1, 9, from_pose=(465.0, 775.0))[0]
        centred = build_final_straight_reverse_waypoints(
            B1, 9, from_pose=(425.0, 775.0))[0]
        self.assertAlmostEqual(offset.x, centred.x, places=6)
        self.assertAlmostEqual(offset.y, centred.y, places=6)
        self.assertAlmostEqual(offset.x, B1.center_x, places=6)

    def test_endpoint_compensates_the_controller_stop_distance(self) -> None:
        from parking.final_alignment import STOP_DISTANCE_COMPENSATION_MM
        self.assertAlmostEqual(STOP_DISTANCE_COMPENSATION_MM,
                               ControllerConfig().stop_distance_cm * 10.0)
        wp = build_final_straight_reverse_waypoints(
            B1, 9, from_pose=(425.0, 775.0))[0]
        local = to_slot_local(B1, wp.x, wp.y, wp.target_heading_deg)
        # 보상하되 슬롯 깊이 여유를 넘지 않는다 — 넘으면 겨눔점 footprint 가
        # 맵 밖이라 trajectory validator 가 경로를 통째로 거절한다.
        from parking.waypoints import CAR_LENGTH_MM
        depth_margin = (B1.length - CAR_LENGTH_MM) / 2.0
        self.assertGreater(local.depth_mm, 0.0)
        self.assertLessEqual(local.depth_mm,
                             min(STOP_DISTANCE_COMPENSATION_MM, depth_margin))

    def test_arrival_radius_must_exceed_the_stop_distance(self) -> None:
        """도착 반경이 정지 거리보다 작으면 영원히 도착하지 못한다."""
        cfg = ControllerConfig()
        wp = build_final_straight_reverse_waypoints(
            B1, 9, from_pose=(425.0, 775.0))[0]
        radius = cfg.arrival_radius_cm(wp.position_tolerance_cm, "FINAL")
        self.assertGreaterEqual(radius, cfg.stop_distance_cm)

    def test_lateral_limit_keeps_terminal_heading_within_parked_tolerance(self) -> None:
        """횡오차를 직선후진으로 고치면 heading 이 틀어진다 — 한계를 조여둔다."""
        from parking.final_alignment import STRAIGHT_REVERSE_LATERAL_LIMIT_MM
        # 계획 목표 + terminal DONE 오차가 직선후진 acquisition band 안이어야
        # 정렬 → 재평가 → 정렬 진동이 없다.
        self.assertLessEqual(
            ALIGN_GOAL_LATERAL_MM + ALIGN_TERMINAL_POSITION_TOLERANCE_MM,
            STRAIGHT_REVERSE_LATERAL_LIMIT_MM)

    def test_depth_margin_is_geometric_not_measurement_noise(self) -> None:
        """깊이 여유는 (슬롯 깊이 - 차 길이)/2 이지 자세 잡음이 아니다."""
        from parking.waypoints import CAR_LENGTH_MM
        margin = (B1.length - CAR_LENGTH_MM) / 2.0
        just_inside = evaluate_final_pose(
            B1, B1.center_x, B1.center_y - (margin - 5.0), 270.0)
        self.assertTrue(just_inside.parked)
        self.assertEqual(just_inside.depth_overflow_mm, 0.0)
        too_short = evaluate_final_pose(
            B1, B1.center_x, B1.center_y - (margin + 25.0), 270.0)
        self.assertEqual(too_short.action, "STRAIGHT_REVERSE")


class TestFinalRegionClassification(unittest.TestCase):
    """FINAL_REGION 은 waypoint 이름이 아니라 slot 기하로 정한다."""

    def test_slot_interior_is_final_region(self) -> None:
        for x, y in ((425.0, 1050.0), (441.0, 1088.0), (430.0, 1077.0)):
            self.assertTrue(in_final_region(B1, x, y), f"({x},{y})")

    def test_aisle_poses_are_not_final_region(self) -> None:
        """통로에 있는 차는 최종 정렬 대상이 아니라 재접근 대상이다.

        예전 하한은 -(length/2 + CAR_LENGTH) = -400mm 였고, 그건 B1 기준
        y=650 — 통로 주행 밴드(325~875) 한가운데다. 그래서 슬롯 x 대역을
        지나가는 통로 주행 차량이 "최종 정렬 중" 으로 분류됐고, 실차에서
        두 번 미션을 죽였다:

          run_20260903_032539  (376,626) heading 오차 129도 -> depth -397.2
          run_20260828_003813  (371,802) heading 오차  60도 -> depth -247.6

        둘 다 FINAL_EVAL -> ALIGN -> NO_SAFE_FINAL_ALIGNMENT 반복으로 끝났다.
        그 자세들에 필요한 것은 정렬이 아니라 재접근이었다.

        (425,775) 는 정렬 staging 자세지만 역시 통로 안이다. 정렬 lifecycle
        자체는 FINAL_EVAL 경로가 몰기 때문에 이 술어에 의존하지 않는다 —
        여기 쓰이는 곳은 heading fault 복구 후 재개 지점 판단뿐이고, 통로에
        있는 차는 그때 재접근으로 보내는 것이 맞다.
        """
        for x, y in ((809.0, 355.0), (200.0, 600.0), (425.0, 600.0),
                     (425.0, 775.0), (376.2, 626.3), (370.8, 802.4)):
            self.assertFalse(in_final_region(B1, x, y), f"({x},{y})")

    def test_adjacent_slot_is_not_this_slots_final_region(self) -> None:
        """슬롯 간격 225mm — 옆 슬롯 차를 이 슬롯 최종구간으로 오인하면 안 된다."""
        self.assertFalse(in_final_region(B1, 650.0, 1050.0))
        self.assertTrue(in_final_region(default_slot_specs()["B2"],
                                        650.0, 1050.0))

    def test_classification_is_heading_independent(self) -> None:
        """heading 이 LAST_VALID 로 굳어도 구간 판정은 오염되지 않는다."""
        self.assertTrue(in_final_region(B1, 441.0, 1088.0))
        self.assertTrue(in_final_region(A1, 425.0, 150.0))

    def test_works_for_the_opposite_slot_row(self) -> None:
        self.assertTrue(in_final_region(A1, 425.0, 150.0))
        self.assertFalse(in_final_region(A1, 425.0, 900.0))


class TestFinalQualityReplanRouting(TestFinalAlignmentPipeline):
    """슬롯 최종 구간의 heading mismatch 는 재접근이 아니라 최종 정렬로 간다."""

    def _replan(self, reason, x, y, stage="PARKING"):
        self.pipeline._parking_stage[1] = stage
        self.runner.replan_reason = reason
        self.runner.failed_target = SimpleNamespace(
            x_mm=425.0, y_mm=1050.0, route_id=4, waypoint_id=8)
        v = self._view(x, y, 300.0, t=5.0)
        self.pipeline._is_global_handoff_terminal = lambda c, t: False
        self.pipeline._handoff_region_reached = lambda vw, t: False
        self.pipeline._begin_parking_handoff = lambda *a, **k: None
        self.pipeline._last_replan_signature = {}
        self.pipeline._replan_attempts = {}
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        return v

    def test_final_region_heading_mismatch_goes_to_final_eval(self) -> None:
        """224937 형: 슬롯 안에서 주차선과 어긋남 → FINAL_EVAL_PENDING."""
        self._replan("HEADING_OUT_OF_TOLERANCE", 441.0, 1088.0)
        self.assertEqual(self.pipeline._parking_stage[1], "FINAL_EVAL_PENDING")
        self.assertTrue([f for n, f in self.pipeline.events
                         if n == "FINAL_QUALITY_REPLAN"])

    def test_final_quality_replan_does_not_spend_recovery_budget(self) -> None:
        self._replan("HEADING_OUT_OF_TOLERANCE", 441.0, 1088.0)
        self.assertEqual(self.pipeline._parking_recovery_attempts.get(1, 0), 0)

    def test_heading_mismatch_outside_final_region_uses_rear_recovery(self) -> None:
        """접근 중 heading 오차는 그대로 기존 재접근 경로."""
        self._replan("HEADING_OUT_OF_TOLERANCE", 809.0, 355.0)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertEqual(self.pipeline._parking_recovery_attempts.get(1), 1)

    def test_predicted_boundary_in_final_region_stops_then_evaluates(self) -> None:
        """222253: 예방 정지는 route 재개 없이 fresh-pose 평가기로 간다."""
        self.pipeline._auto_host_slot[1] = "A2"
        self._replan("PREDICTED_BOUNDARY", 645.8, 155.9)
        self.assertEqual(self.pipeline._parking_stage[1],
                         "FINAL_EVAL_PENDING")
        self.assertEqual(self.pipeline._parking_recovery_attempts.get(1, 0), 0)
        self.assertTrue(self._reasons("FINAL_SAFETY_STOP_EVAL"))

    def test_hard_boundary_is_never_routed_to_final_alignment(self) -> None:
        """실제 boundary 침범은 기존 physical recovery/fault 계약을 유지한다."""
        self._replan("BOUNDARY_HARD", 441.0, 1088.0)
        self.assertNotEqual(self.pipeline._parking_stage[1],
                            "FINAL_EVAL_PENDING")

    def test_predicted_boundary_outside_final_region_uses_recovery(self) -> None:
        """222507: 맵 가장자리 pose 는 final evaluator 로 우회하지 않는다."""
        self._replan("PREDICTED_BOUNDARY", 1037.6, 976.3)
        self.assertNotEqual(self.pipeline._parking_stage[1],
                            "FINAL_EVAL_PENDING")

    def test_sensor_loss_replan_is_never_routed_to_final_alignment(self) -> None:
        for reason in ("REVERSE_HEADING_TIMEOUT", "POSE_STALE"):
            with self.subTest(reason):
                self.setUp()
                self._replan(reason, 441.0, 1088.0)
                self.assertNotEqual(self.pipeline._parking_stage[1],
                                    "FINAL_EVAL_PENDING")

    def test_final_region_but_untrusted_heading_plans_nothing(self) -> None:
        """FINAL region + LAST_VALID → 정렬 route 금지, fresh heading 대기."""
        self._replan("HEADING_OUT_OF_TOLERANCE", 441.0, 1088.0)
        v = self._view(441.0, 1088.0, 300.0, t=6.0)
        v.heading_source = "LAST_VALID"
        self.pipeline._maybe_evaluate_final_pose(v)
        self.assertEqual(self.runner.loaded, [])
        self.assertTrue([f for n, f in self.pipeline.events
                         if n == "WAIT_FOR_FRESH_HEADING"])

    def test_fresh_heading_then_evaluates_and_aligns(self) -> None:
        self._replan("HEADING_OUT_OF_TOLERANCE", 441.0, 1088.0)
        v = self._view(441.0, 1088.0, 300.0, t=7.0)
        self.pipeline._maybe_evaluate_final_pose(v)
        self.assertTrue([f for n, f in self.pipeline.events
                         if n == "FINAL_POSE_EVAL"])
        self.assertIn(self.pipeline._parking_stage[1],
                      {"FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE",
                       "SETUP_PENDING", "WAIT_SAFE_RECOVERY"})

    def test_222253_exact_pose_reaches_final_alignment(self) -> None:
        """실차 A2 예방정지 pose 는 early PARKED 가 아니라 ALIGN 대상이다."""
        self.pipeline._auto_host_slot[1] = "A2"
        self._replan("PREDICTED_BOUNDARY", 645.8, 155.9)
        view = self._view(645.8, 155.9, 99.1, t=6.0)
        self.pipeline._maybe_evaluate_final_pose(view)
        verdicts = self._reasons("FINAL_POSE_EVAL")
        self.assertTrue(verdicts)
        self.assertEqual(verdicts[-1]["action"], "ALIGN")
        self.assertEqual(self.pipeline._parking_stage[1], "FINAL_ALIGNMENT")
        self.assertTrue(self.runner.loaded)

    def test_222641_last_valid_cannot_evaluate_but_fresh_recovery_can(self) -> None:
        """B1 LAST_VALID 정지는 유지하고, primary heading 복귀 시 평가가 산다."""
        p = self.pipeline
        p._parking_stage[1] = "WAIT_FRESH_HEADING_FAULT"
        p._heading_fault_hold.add(1)
        stale = self._view(456.0, 1075.7, 261.1, t=5.0)
        stale.heading_source = "LAST_VALID"
        self.assertFalse(p._maybe_resume_heading_fault(stale))
        self.assertEqual(p._parking_stage[1], "WAIT_FRESH_HEADING_FAULT")
        self.assertEqual(self.runner.loaded, [])

        fresh = self._view(456.0, 1075.7, 261.1, t=6.0)
        self.assertTrue(p._maybe_resume_heading_fault(fresh))
        self.assertEqual(p._parking_stage[1], "FINAL_EVAL_PENDING")
        evaluated = self._view(456.0, 1075.7, 261.1, t=7.0)
        p._maybe_evaluate_final_pose(evaluated)
        self.assertTrue(self._reasons("FINAL_POSE_EVAL"))
        self.assertNotEqual(p._parking_stage[1], "PARKED")

    def test_still_moving_vehicle_is_not_evaluated_yet(self) -> None:
        """물리적 정지 확인 전에는 평가하지 않는다 (요청문 13절).

        예방 정지 시점의 pose 를 그대로 평가기에 넘기면, 관성으로 아직
        움직이는 차의 자세로 최종 품질을 판정하게 된다.
        """
        self.pipeline._auto_host_slot[1] = "A2"
        self._replan("PREDICTED_BOUNDARY", 645.8, 155.9)
        moving = self._view(645.8, 155.9, 99.1, t=6.0)
        moving.recent.clear()
        moving.recent.extend([(645.8 + 40.0 * i, 155.9) for i in range(5)])
        self.assertFalse(self.pipeline._maybe_evaluate_final_pose(moving))
        self.assertFalse(self._reasons("FINAL_POSE_EVAL"))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "FINAL_EVAL_PENDING")
        self.assertTrue([f for n, f in self.pipeline.events
                         if n == "WAIT_FOR_PHYSICAL_STOP"])

    def test_stale_observation_is_not_evaluated(self) -> None:
        """경계 시점보다 오래된 관측으로는 평가하지 않는다.

        _phase_boundary_stopped 는 경계 이후의 **새** 관측만 인정한다.
        """
        self.pipeline._auto_host_slot[1] = "A2"
        self._replan("PREDICTED_BOUNDARY", 645.8, 155.9)
        boundary_at = self.pipeline._parking_plan_wait[1]
        stale = self._view(645.8, 155.9, 99.1, t=boundary_at)
        self.assertFalse(self.pipeline._maybe_evaluate_final_pose(stale))
        self.assertFalse(self._reasons("FINAL_POSE_EVAL"))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "FINAL_EVAL_PENDING")


class TestParkedIsTerminal(TestFinalAlignmentPipeline):
    def test_stale_done_after_parked_never_restarts_parking(self) -> None:
        self.pipeline._parking_stage[1] = "PARKED"
        self._view(425.0, 1050.0, 270.0, t=9.0)
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.DONE)
        self.assertEqual(self.pipeline._parking_stage[1], "PARKED")
        self.assertEqual(self.runner.loaded, [])

    def test_exhausted_alignment_is_terminal_too(self) -> None:
        self.pipeline._parking_stage[1] = "WAIT_FINAL_ALIGNMENT_FAILED"
        self._view(425.0, 1050.0, 270.0, t=9.0)
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.DONE)
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_FINAL_ALIGNMENT_FAILED")

    def test_parked_is_not_a_rear_recoverable_stage(self) -> None:
        self.assertNotIn("PARKED", ParkingPipeline._REAR_RECOVERABLE_STAGES)
        self.assertIn("PARKED", ParkingPipeline._STALL_EXEMPT_STAGES)

    def test_comm_timeout_and_new_boot_cannot_resurrect_parked(self) -> None:
        p = self.pipeline
        p._parking_stage[1] = "PARKED"
        p._auto_host_route[1] = [SimpleNamespace(route_id=15)]
        parked_pose = (425.0, 1050.0, 270.0)
        p._parked_obstacles[1] = parked_pose

        p._on_comm_fail(1, {
            "type": "COMM_TIMEOUT", "session_id": "S-OLD",
            "boot_id": "B-OLD"})
        self.assertEqual(p._parking_stage[1], "PARKED")
        self.assertNotIn(1, p._comm_recovery_context)
        self.assertNotIn(1, p._auto_host_route)
        self.assertIn(1, p._comm_lost)
        self.assertEqual(self.runner.loaded, [])

        p._on_resync(1, {"boot_id": "B-NEW"})
        self.assertEqual(p._parking_stage[1], "PARKED")
        self.assertNotIn(1, p._comm_recovery_context)
        self.assertNotIn(1, p._comm_lost)
        self.assertEqual(p._allocation_state[1], "PARKED")
        self.assertEqual(p._parked_obstacles[1], parked_pose)
        self.assertEqual(self.runner.loaded, [])
        self.assertGreaterEqual(len(p.zeroed), 2)

    def test_parked_comm_fault_is_isolated_from_other_car(self) -> None:
        p = self.pipeline
        other = _Runner()
        p.auto_hosts[2] = other
        p._auto_host_slot[2] = "A1"
        p._parking_stage.update({1: "PARKED", 2: "PARKING"})
        peer_ctx = {"state": "WAIT_FRESH_POSE", "prior_stage": "PARKING"}
        p._comm_recovery_context[2] = dict(peer_ctx)

        p._on_comm_fail(1, {"type": "COMM_TIMEOUT"})
        p._on_resync(1, {"boot_id": "B-NEW"})

        self.assertEqual(p._parking_stage[2], "PARKING")
        self.assertEqual(p._comm_recovery_context[2], peer_ctx)
        self.assertFalse(other.stopped)

    def test_stale_parked_recovery_context_is_discarded_defensively(self) -> None:
        p = self.pipeline
        p._parking_stage[1] = "PARKED"
        p._comm_recovery_context[1] = {
            "prior_stage": "PARKED", "slot_id": "B1",
            "state": "WAIT_FRESH_POSE", "resume_after_obs_time": 0.0}
        v = self._view(425.0, 1050.0, 270.0, t=2.0)

        self.assertTrue(p._maybe_resume_comm_recovery(v))
        self.assertEqual(p._parking_stage[1], "PARKED")
        self.assertNotIn(1, p._comm_recovery_context)
        self.assertEqual(self.runner.loaded, [])




class TestSlotOutsideVehicleAlwaysRealigns(TestFinalAlignmentPipeline):
    """실차 ground truth: 차가 슬롯 밖에 남으면 절대 완료가 아니다.

    run_20260827_234439 최종 자세 (328,1087) — 슬롯 중심선에서 97mm 벗어나
    차체가 옆 라인을 77mm 넘었다. 예전 코드는 정지만 확인하고 PARKED 로
    확정했다. 이제는 반드시 재정렬로 이어져야 한다.
    """

    REAL = {"234439": (328.0, 1087.0, 272.3),
            "234231": (335.0, 1097.0, 254.3)}

    def test_real_final_poses_are_outside_the_slot(self) -> None:
        for run, (x, y, h) in self.REAL.items():
            with self.subTest(run):
                lateral, _ = footprint_overflow_mm(B1, x, y, h)
                self.assertGreater(lateral, 50.0,
                                   f"{run}: 슬롯 밖이어야 한다")

    def test_real_final_poses_are_never_parked(self) -> None:
        for run, (x, y, h) in self.REAL.items():
            with self.subTest(run):
                self.assertFalse(evaluate_final_pose(B1, x, y, h).parked)

    def test_real_final_poses_ask_for_alignment(self) -> None:
        for run, (x, y, h) in self.REAL.items():
            with self.subTest(run):
                self.assertEqual(evaluate_final_pose(B1, x, y, h).action,
                                 "ALIGN")

    def test_real_final_poses_are_inside_the_final_region(self) -> None:
        for run, (x, y, h) in self.REAL.items():
            with self.subTest(run):
                self.assertTrue(in_final_region(B1, x, y))

    def test_recovery_from_final_region_goes_to_final_eval_not_setup(self) -> None:
        """슬롯 안에 있는 차를 재접근(SETUP)으로 보내면 정렬이 영영 안 된다."""
        x, y, h = self.REAL["234439"]
        view = self._view(x, y, h, t=5.0)
        self.assertEqual(self.pipeline._post_recovery_stage(1, view),
                         "FINAL_EVAL_PENDING")

    def test_recovery_far_from_the_slot_still_uses_setup(self) -> None:
        view = self._view(809.0, 355.0, 330.0, t=5.0)
        self.assertEqual(self.pipeline._post_recovery_stage(1, view),
                         "SETUP_PENDING")

    def test_heading_fault_recovery_in_final_region_reaches_alignment(self) -> None:
        """LIVENESS: 슬롯 밖에서 멈춘 차가 heading 복귀 후 실제 정렬로 간다."""
        x, y, h = self.REAL["234439"]
        p = self.pipeline
        p._parking_stage[1] = "WAIT_FRESH_HEADING_FAULT"
        p._heading_fault_hold.add(1)
        p._heading_wait_started[1] = 0.0
        view = self._view(x, y, h, t=5.0)
        self.assertTrue(p._maybe_resume_heading_fault(view))
        self.assertEqual(p._parking_stage[1], "FINAL_EVAL_PENDING")

        view = self._view(x, y, h, t=6.0)
        p._maybe_evaluate_final_pose(view)
        self.assertTrue([f for n, f in p.events if n == "FINAL_POSE_EVAL"],
                        "최종 자세 평가가 아예 실행되지 않았다")
        # 97mm 횡오차를 슬롯 안에서 한 번에 정렬하는 기동은 현재 탐색 예산으로
        # 못 찾는다. 그때는 조용히 멈추는 것이 아니라 더 넓은 setup(빼내기)로
        # 넘어가야 한다 — 어느 쪽이든 **실제 교정 동작**으로 이어져야 한다.
        self.assertIn(p._parking_stage[1],
                      {"FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE",
                       "SETUP_PENDING"})
        self.assertNotEqual(p._parking_stage[1], "PARKED")

    def test_stuck_pose_can_at_least_escape_the_slot(self) -> None:
        """실차 자세에서 최소한 빼내는 기동은 계획 가능해야 한다.

        차체 뒤가 맵을 15mm 넘은 채 멈추면 예전에는 모든 후보가 거절돼
        NO_SAFE_SETUP_MANEUVER 로 아무 복구도 못 했다.
        """
        from parking.waypoints import plan_setup_recovery
        for run, (x, y, h) in self.REAL.items():
            with self.subTest(run):
                rec = plan_setup_recovery(B1, (x, y), h, goal_test=lambda p: True)
                self.assertIsNotNone(rec, f"{run}: 탈출 기동조차 계획 불가")

    def test_escape_never_goes_further_out_of_the_map(self) -> None:
        """탈출은 허용하되 더 나가는 것은 금지된다."""
        from parking.waypoints import _path_clearance, plan_setup_recovery
        x, y, h = self.REAL["234439"]
        initial, _ = _path_clearance([(x, y, h)])
        rec = plan_setup_recovery(B1, (x, y), h, goal_test=lambda p: True)
        self.assertIsNotNone(rec)
        overflow, _ = _path_clearance(rec.poses)
        self.assertLessEqual(overflow, initial + 1e-6)


if __name__ == "__main__":
    unittest.main()
