"""입구 staging 실행가능성 / 배경 계획 / telemetry 지속 (RUN 164904·165419·165710).

2026-09-04 baseline E2E 3회는 모두 같은 자리에서 끊겼다:

    입구 정상 pose -> B1 선택 -> 9-waypoint staging 적재
    -> 8~11초 뒤 ARC_CORRIDOR_MISSED / HEADING_OUT_OF_TOLERANCE
    -> 후보 재탐색이 vision 스레드를 3.6~17.5초 점유 (pose 로그 공백)
    -> 근거리 슬롯 전부 기하 reject -> A3 만 생존 -> 배회 -> FAULT

이 파일이 고정하는 계약은 네 가지다.

1. 기록기 telemetry 는 슬롯 배정 **이후에도** 계속 나온다.
   (lifecycle_snapshot 이 SlotSpec 에 없는 slot.x 를 읽어 3/3 run 에서
    control.jsonl 과 PHASE/MISSION/ESP_STATE 이벤트가 조용히 끊겼다.)

2. setup/staging 기동의 **구간 경계**는 도착 판정 기준으로 실제로 실행
   가능하다. 실측 staging 후진 구간은 끝점 변위 106.3mm 인데 호출부가 넘긴
   도착 반경은 110.0mm 였다 — 차가 앞 구간 끝에서 이미 다음 구간 끝을
   capture 범위 안에 두고 있었다.

3. 후보 탐색이 몇 초 걸려도 카메라/포즈 처리는 멈추지 않는다.

4. 지금 인계 가능한 자세라면, recovery 가 그 인계 가능성을 잃는 기동을
   싣지 않는다. (164904: (926.9,624.6,12.1°) -> 순수 후진 730mm ->
   (399.4,388.9,7.7°) 에서 A3 인계 불가 -> NO_SAFE_SETUP_MANEUVER)

새 임계값은 도입하지 않는다. 2번은 호출부가 이미 넘기던 min_executable_mm
(ControllerConfig.arrival_radius_cm 유래)을, 4번은 이미 쓰는 plan_handoff 를
그대로 쓴다.
"""

from __future__ import annotations

import math
import threading
import time
import unittest

from controller.config import ControllerConfig
from parking.waypoints import (PHASE_DEFAULTS,
                               REVERSE_START_HEADING_TOLERANCE_DEG,
                               build_setup_recovery_waypoints,
                               default_slot_specs, plan_handoff)
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView
from pipeline.tests.test_entry_staging_handoff import settle_staging_plan

# ── 실차 자세 ───────────────────────────────────────────────────────────────
POSE_ENTRANCE_164904 = (138.3, 166.9, 91.5)
POSE_ENTRANCE_165419 = (163.1, 189.4, 88.7)
POSE_ENTRANCE_165710 = (135.7, 151.1, 92.0)
# staging 재개 판정 시점 — 통로축에 거의 정렬돼 있는데 선호 슬롯을 지나쳤다
POSE_164904_RESUME = (581.6, 528.4, 359.5)
POSE_165710_RESUME = (515.4, 491.6, 3.0)
# A3 인계점의 좋은 자세와, setup recovery 가 실제로 만들어 낸 끝 자세
POSE_164904_AT_A3 = (926.9, 624.6, 12.1)
POSE_164904_AFTER_SETUP = (399.4, 388.9, 7.7)


def _pipeline() -> ParkingPipeline:
    return ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                          parking_mode="rear"))


def _view(pose, node="entrance") -> VehicleView:
    return VehicleView(track_id=2, car_id=1, node=node,
                       position_mm=(pose[0], pose[1]), heading_deg=pose[2],
                       heading_source="FRONT_CUSHION", last_obs_time=1.0)


class _StubRunner:
    def __init__(self) -> None:
        self.loaded: list[list] = []
        self.stopped = False

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        self.stopped = True


# ══ 1. TELEMETRY PERSISTS AFTER SLOT_SELECTED ═══════════════════════════════

class LifecycleSnapshotSurvivesSlotSelection(unittest.TestCase):
    """기록기 tick 이 슬롯 배정 이후에도 예외 없이 돌아야 한다."""

    def setUp(self) -> None:
        self.pipe = _pipeline()

    def test_snapshot_without_a_slot_is_null_but_valid(self) -> None:
        snap = self.pipe.lifecycle_snapshot(1)
        self.assertIsNone(snap["slot_id"])
        self.assertIsNone(snap["slot_center_x_mm"])
        self.assertIsNone(snap["slot_center_y_mm"])
        self.assertIsNone(snap["parked_heading_deg"])

    def test_snapshot_with_a_selected_slot_reports_the_spec_centre(self) -> None:
        """이 한 줄이 3/3 run 의 control.jsonl 을 죽였다 (slot.x 는 없다)."""
        for slot_id in default_slot_specs():
            with self.subTest(slot=slot_id):
                self.pipe._auto_host_slot[1] = slot_id
                snap = self.pipe.lifecycle_snapshot(1)
                spec = default_slot_specs()[slot_id]
                self.assertEqual(snap["slot_id"], slot_id)
                self.assertEqual(snap["slot_center_x_mm"], spec.center_x)
                self.assertEqual(snap["slot_center_y_mm"], spec.center_y)
                self.assertEqual(snap["parked_heading_deg"],
                                 spec.target_heading_deg)

    def test_the_spec_really_has_no_x_or_y_attribute(self) -> None:
        """회귀 방지: 누가 slot.x 로 되돌리면 다시 조용히 죽는다."""
        spec = default_slot_specs()["A3"]
        self.assertFalse(hasattr(spec, "x"))
        self.assertFalse(hasattr(spec, "y"))

    def test_repeated_snapshots_never_raise(self) -> None:
        """기록기 루프는 예외를 삼킨다 — 여기서 잡지 못하면 로그가 사라진다."""
        self.pipe._auto_host_slot[1] = "B1"
        for _ in range(50):
            self.pipe.lifecycle_snapshot(1)


# ══ 2·3. DEGENERATE SEGMENTS / DIRECTION CHANGES ════════════════════════════

class SetupSegmentsAreIndependentlyExecutable(unittest.TestCase):
    """구간 경계가 도착 반경 안에 겹치는 기동을 만들지 않는다."""

    def setUp(self) -> None:
        self.pipe = _pipeline()
        self.min_exec = self.pipe._setup_min_executable_mm()

    def test_the_arrival_radius_is_derived_not_invented(self) -> None:
        """새 상수가 아니라 기존 도착 판정에서 유도된 값이어야 한다."""
        cfg = ControllerConfig()
        tolerance = PHASE_DEFAULTS["RECOVERY"]["position_tolerance_cm"]
        self.assertAlmostEqual(
            self.min_exec, 10.0 * cfg.arrival_radius_cm(tolerance, "RECOVERY"))

    def _staging_route(self, pose):
        view = _view(pose)
        self.pipe.views = {2: view}
        self.pipe.allocator.update(2, view.position_mm)
        return self.pipe._build_entry_staging_route(1, view, "B1", 1)

    def _segment_ends(self, route, start):
        """구간(전/후진·곡률이 바뀌는 지점)의 끝점만 뽑는다."""
        ends = [start]
        for index, wp in enumerate(route):
            last = index == len(route) - 1
            changes = (not last
                       and (route[index + 1].motion_direction
                            != wp.motion_direction
                            or route[index + 1].curvature != wp.curvature))
            if changes or last:
                ends.append((wp.x, wp.y))
        return ends

    def test_every_staging_segment_boundary_clears_the_arrival_radius(self):
        for pose in (POSE_ENTRANCE_164904, POSE_ENTRANCE_165419,
                     POSE_ENTRANCE_165710):
            with self.subTest(pose=pose):
                route = self._staging_route(pose)
                if not route:
                    continue          # 해가 없으면 이 계약의 대상이 아니다
                ends = self._segment_ends(route, (pose[0], pose[1]))
                for a, b in zip(ends, ends[1:]):
                    self.assertGreater(
                        math.hypot(b[0] - a[0], b[1] - a[1]), self.min_exec,
                        f"구간 경계가 도착 반경({self.min_exec:.0f}mm) 안이다")

    def test_direction_changes_are_not_captured_back_to_back(self) -> None:
        """전/후진 전환 앞뒤 waypoint 가 한 자리에서 연속 capture 되지 않는다."""
        for pose in (POSE_ENTRANCE_164904, POSE_ENTRANCE_165419,
                     POSE_ENTRANCE_165710):
            with self.subTest(pose=pose):
                route = self._staging_route(pose)
                for index in range(len(route) - 1):
                    if (route[index + 1].motion_direction
                            == route[index].motion_direction):
                        continue
                    flip = route[index]
                    tail = [w for w in route[index + 1:]
                            if w.motion_direction != flip.motion_direction]
                    if not tail:
                        continue
                    end = tail[-1]
                    self.assertGreater(
                        math.hypot(end.x - flip.x, end.y - flip.y),
                        self.min_exec,
                        "방향 전환 구간 전체가 도착 반경 안에 들어간다")

    def test_the_recorded_106mm_reverse_leg_is_now_rejected(self) -> None:
        """실측 형상 그대로의 회귀 잠금.

        3개 baseline run 과 known-good run_20260903_022217 이 모두 실행한
        staging 기동의 후진 구간은 끝점 변위 106.3mm 였고, 그때 호출부가
        넘긴 도착 반경은 110.0mm 였다.
        """
        self.assertGreater(self.min_exec, 106.3,
                           "이 값이 106.3mm 아래로 내려가면 그 기동이 다시 통과한다")

    def test_a_planner_solution_still_exists_from_the_entrance(self) -> None:
        """실행가능성 제약이 입구에서 해를 전부 없애 버리면 안 된다."""
        solved = 0
        for pose in (POSE_ENTRANCE_164904, POSE_ENTRANCE_165419,
                     POSE_ENTRANCE_165710):
            if self._staging_route(pose):
                solved += 1
        self.assertGreater(solved, 0,
                           "세 입구 자세 어디에서도 staging 해가 없으면 과제약이다")


# ══ 4. STAGING GATE IS SEPARATE FROM REVERSE_START ══════════════════════════

class StagingHeadingGateIsNotReverseStart(unittest.TestCase):

    def test_generic_setup_recovery_keeps_the_reverse_start_default(self) -> None:
        """기본값은 그대로다 — 후진 원호 진입 계약을 바꾸지 않는다."""
        route = build_setup_recovery_waypoints(
            default_slot_specs()["B1"], route_id=1,
            from_pose=(600.0, 600.0), from_heading_deg=0.0)
        gates = {w.heading_tolerance_deg for w in route if w.heading_required}
        if gates:
            self.assertEqual(gates, {REVERSE_START_HEADING_TOLERANCE_DEG})

    def test_the_tolerance_is_a_parameter_not_a_hardcoded_reuse(self) -> None:
        route = build_setup_recovery_waypoints(
            default_slot_specs()["B1"], route_id=1,
            from_pose=(600.0, 600.0), from_heading_deg=0.0,
            segment_heading_tolerance_deg=15.0)
        gates = {w.heading_tolerance_deg for w in route if w.heading_required}
        if gates:
            self.assertEqual(gates, {15.0})

    def test_entry_staging_uses_its_own_existing_15deg_tolerance(self) -> None:
        pipe = _pipeline()
        expected = float(pipe.config.entry_staging_heading_tolerance_deg)
        self.assertNotEqual(expected, REVERSE_START_HEADING_TOLERANCE_DEG)
        view = _view(POSE_ENTRANCE_164904)
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        route = pipe._build_entry_staging_route(1, view, "B1", 1)
        if not route:
            self.skipTest("이 자세에서 staging 해가 없다")
        gates = {w.heading_tolerance_deg for w in route if w.heading_required}
        self.assertEqual(gates, {expected})


# ══ 4b. AISLE TRAVEL DIRECTION MUST NOT DEMAND A U-TURN ═════════════════════

class StagingHeadingTargetNeverRequiresAUTurn(unittest.TestCase):
    """선호 슬롯을 지나쳤다는 이유로 180° 를 요구하지 않는다."""

    def setUp(self) -> None:
        self.pipe = _pipeline()

    def _desired(self, pose, slot="B1"):
        view = _view(pose)
        self.pipe.views = {2: view}
        return self.pipe._entry_staging_heading(view, slot)

    def test_164904_and_165710_resume_poses_are_treated_as_aligned(self) -> None:
        """실측: 0.5°/3.0° 로 정렬돼 있는데 179.5°/177.0° 오차로 계산됐다."""
        for pose in (POSE_164904_RESUME, POSE_165710_RESUME):
            with self.subTest(pose=pose):
                view = _view(pose)
                self.pipe.views = {2: view}
                self.assertTrue(
                    self.pipe._entry_staging_heading_aligned(view, "B1"),
                    "통로축에 정렬된 차를 미정렬로 보면 staging 이 다시 돈다")

    def test_the_entrance_still_asks_for_plus_x(self) -> None:
        for pose in (POSE_ENTRANCE_164904, POSE_ENTRANCE_165419,
                     POSE_ENTRANCE_165710):
            with self.subTest(pose=pose):
                self.assertEqual(self._desired(pose), 0.0)

    def test_a_car_actually_travelling_minus_x_still_gets_180(self) -> None:
        self.assertEqual(self._desired((430.0, 600.0, 177.0)), 180.0)

    def test_it_agrees_with_the_handoff_planner(self) -> None:
        """staging 과 plan_handoff 가 같은 진행 방향 규약을 쓴다."""
        spec = default_slot_specs()["B1"]
        for pose in (POSE_164904_RESUME, POSE_165710_RESUME,
                     POSE_ENTRANCE_164904, (430.0, 600.0, 177.0)):
            with self.subTest(pose=pose):
                plan = plan_handoff(spec, from_pose=(pose[0], pose[1]),
                                    from_heading_deg=pose[2])
                self.assertEqual(self._desired(pose),
                                 plan.approach_heading_deg)


# ══ 5. RECOVERY MUST NOT DESTROY TARGET FEASIBILITY ═════════════════════════

class RecoveryPreservesTargetHandoff(unittest.TestCase):

    def setUp(self) -> None:
        self.pipe = _pipeline()

    class _WP:
        def __init__(self, x, y, h):
            self.x, self.y, self.target_heading_deg = x, y, h

    def test_the_recorded_164904_setup_is_rejected(self) -> None:
        """실측: A3 인계 가능한 자세에서 인계 불가능한 자세로 730mm 후진."""
        view = _view(POSE_164904_AT_A3, node="A3_front")
        before = plan_handoff(default_slot_specs()["A3"],
                              from_pose=view.position_mm,
                              from_heading_deg=view.heading_deg)
        self.assertTrue(before.feasible, "출발 자세는 인계 가능해야 전제가 성립한다")
        setup = [self._WP(*POSE_164904_AFTER_SETUP)]
        self.assertFalse(
            self.pipe._setup_keeps_target_feasible(view, "A3", setup))

    def test_a_maneuver_that_keeps_the_handoff_is_allowed(self) -> None:
        view = _view(POSE_164904_AT_A3, node="A3_front")
        setup = [self._WP(700.0, 600.0, 0.0)]
        self.assertTrue(
            self.pipe._setup_keeps_target_feasible(view, "A3", setup))

    def test_it_never_blocks_when_there_was_nothing_to_preserve(self) -> None:
        """지금도 인계 불가능한 자세라면 이 검사는 아무것도 막지 않는다."""
        view = _view(POSE_164904_AFTER_SETUP)
        before = plan_handoff(default_slot_specs()["A3"],
                              from_pose=view.position_mm,
                              from_heading_deg=view.heading_deg)
        self.assertFalse(before.feasible)
        setup = [self._WP(150.0, 150.0, 90.0)]
        self.assertTrue(
            self.pipe._setup_keeps_target_feasible(view, "A3", setup))

    def test_missing_pose_or_slot_is_not_treated_as_a_loss(self) -> None:
        view = _view(POSE_164904_AT_A3)
        self.assertTrue(self.pipe._setup_keeps_target_feasible(view, "A3", []))
        view.heading_deg = None
        self.assertTrue(self.pipe._setup_keeps_target_feasible(
            view, "A3", [self._WP(*POSE_164904_AFTER_SETUP)]))


# ══ 6·7·8. PLANNER OFF THE VISION THREAD ════════════════════════════════════

class StagingPlanningDoesNotBlockThePoseThread(unittest.TestCase):

    def setUp(self) -> None:
        self.pipe = _pipeline()
        self.runner = _StubRunner()
        self.pipe.auto_hosts = {1: self.runner}
        self.pipe.track_of_car = {1: 2}
        self.view = _view(POSE_ENTRANCE_164904)
        self.pipe.views = {2: self.view}
        self.pipe.allocator.update(2, self.view.position_mm)
        self.pipe.allocator.reassign(2, "B1")
        self.events: list[tuple] = []
        self.pipe.on_event_record = (
            lambda name, **fields: self.events.append((name, fields)))
        self.stops: list[int] = []
        self.pipe.server.stop_control = self.stops.append

    def _slow_planner(self, seconds: float, result):
        started = threading.Event()

        def build(car, view, slot, rid, goal_test=None):
            started.set()
            time.sleep(seconds)
            return result
        self.pipe._build_entry_staging_route = build
        return started

    def test_the_calling_thread_returns_immediately(self) -> None:
        started = self._slow_planner(0.6, [])
        t0 = time.monotonic()
        owned = self.pipe._start_entry_staging(1, self.view, "B1", 1)
        elapsed = time.monotonic() - t0
        self.assertTrue(owned, "계획 중에는 프레임을 점유한다")
        self.assertLess(elapsed, 0.3,
                        "후보 탐색이 호출 스레드를 막으면 pose 가 멈춘다")
        self.assertTrue(started.wait(2.0))
        settle_staging_plan(self.pipe, self.view)

    def test_frames_keep_being_processed_while_planning(self) -> None:
        self._slow_planner(0.6, [])
        self.pipe._start_entry_staging(1, self.view, "B1", 1)
        processed = 0
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            self.pipe._apply_entry_staging_plan(self.view)
            processed += 1
        self.assertGreater(processed, 10,
                           "계획 중에도 프레임 핸들러가 계속 돌아야 한다")
        settle_staging_plan(self.pipe, self.view)

    def test_the_vehicle_is_held_at_zero_while_planning(self) -> None:
        self._slow_planner(0.3, [])
        self.pipe._start_entry_staging(1, self.view, "B1", 1)
        self.assertIn(1, self.stops, "계획 시작 시 zero 를 걸어야 한다")
        self.assertEqual(self.runner.loaded, [],
                         "계획이 끝나기 전에 경로가 실리면 안 된다")
        settle_staging_plan(self.pipe, self.view)

    def test_only_one_search_runs_per_car(self) -> None:
        calls = []

        def build(car, view, slot, rid, goal_test=None):
            calls.append(slot)
            time.sleep(0.2)
            return []
        self.pipe._build_entry_staging_route = build
        self.pipe._entry_staging_candidate_slots = lambda preferred: ["B1"]
        for _ in range(5):
            self.pipe._start_entry_staging(1, self.view, "B1", 1)
        settle_staging_plan(self.pipe, self.view)
        self.assertEqual(len(calls), 1, "프레임마다 새 탐색을 띄우면 안 된다")

    def test_a_stale_plan_is_discarded_when_the_car_has_moved(self) -> None:
        sample = self.pipe._build_entry_staging_route(1, self.view, "B1", 1)
        if not sample:
            self.skipTest("이 자세에서 staging 해가 없다")
        self.pipe._entry_staging_signatures.clear()
        self.pipe._build_entry_staging_route = (
            lambda car, view, slot, rid, goal_test=None: sample)
        self.pipe._start_entry_staging(1, self.view, "B1", 1)
        while True:
            with self.pipe._lock:
                if 1 in self.pipe._entry_staging_plan:
                    break
            time.sleep(0.005)
        # 계획이 끝난 사이 차가 도착 반경 밖으로 움직였다.
        moved = self.pipe._setup_min_executable_mm() + 50.0
        self.view.position_mm = (POSE_ENTRANCE_164904[0] + moved,
                                 POSE_ENTRANCE_164904[1])
        self.assertFalse(self.pipe._apply_entry_staging_plan(self.view))
        self.assertEqual(self.runner.loaded, [],
                         "낡은 계획을 새 자세에 적용하면 안 된다")

    def test_a_stale_plan_is_discarded_when_the_stage_changed(self) -> None:
        sample = self.pipe._build_entry_staging_route(1, self.view, "B1", 1)
        if not sample:
            self.skipTest("이 자세에서 staging 해가 없다")
        self.pipe._entry_staging_signatures.clear()
        self.pipe._build_entry_staging_route = (
            lambda car, view, slot, rid, goal_test=None: sample)
        self.pipe._start_entry_staging(1, self.view, "B1", 1)
        while True:
            with self.pipe._lock:
                if 1 in self.pipe._entry_staging_plan:
                    break
            time.sleep(0.005)
        self.pipe._parking_stage[1] = "WAIT_COMM_RECOVERY_FAULT"
        self.assertFalse(self.pipe._apply_entry_staging_plan(self.view))
        self.assertEqual(self.runner.loaded, [])

    def test_a_fresh_plan_is_applied_on_a_later_frame(self) -> None:
        sample = self.pipe._build_entry_staging_route(1, self.view, "B1", 1)
        if not sample:
            self.skipTest("이 자세에서 staging 해가 없다")
        self.pipe._entry_staging_signatures.clear()
        self.pipe._build_entry_staging_route = (
            lambda car, view, slot, rid, goal_test=None: sample)
        self.pipe._start_entry_staging(1, self.view, "B1", 1)
        self.assertTrue(settle_staging_plan(self.pipe, self.view))
        self.assertEqual(len(self.runner.loaded), 1)
        self.assertEqual(self.pipe._parking_stage[1], "ENTRY_STAGING")
        names = [name for name, _ in self.events]
        self.assertIn("ENTRY_STAGING_LOADED", names)


# ══ 9. EXISTING SAFETY UNCHANGED ════════════════════════════════════════════

class SafetyContractsAreUntouched(unittest.TestCase):

    def test_pose_staleness_threshold_is_unchanged(self) -> None:
        self.assertEqual(ControllerConfig().max_pose_age_s, 0.5)

    def test_boundary_thresholds_are_unchanged(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.boundary_hard_margin_mm, 20.0)
        self.assertEqual(cfg.boundary_measurement_uncertainty_mm, 10.0)

    def test_staging_budget_is_unchanged(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.max_entry_staging_attempts, 3)
        self.assertEqual(cfg.entry_staging_heading_tolerance_deg, 15.0)

    def test_reverse_start_tolerance_itself_is_unchanged(self) -> None:
        self.assertEqual(REVERSE_START_HEADING_TOLERANCE_DEG, 5.0)

    def test_recovery_reverse_steering_lock_is_unchanged(self) -> None:
        cfg = ControllerConfig()
        self.assertTrue(cfg.reverse_steering_locked("RECOVERY"))
        self.assertFalse(cfg.reverse_steering_locked("FINAL"))


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()
