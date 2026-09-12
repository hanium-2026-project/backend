"""예약된 슬롯 계약과 그것을 깨던 네 지점 (run 205954 / 210209 / 210321).

allocator 가 B1 을 고른 순간 B1 은 soft preference 가 아니라 **예약된 미션
목표**다. 경로 실패는 "그 슬롯에 갈 자세를 다시 만드는" 문제이지 "다른 슬롯을
고르는" 문제가 아니다. 예산이 끝나면 SAFE STOP 하되 예약은 유지한다.

세 번째 실차 묶음이 이 계약을 네 곳에서 깼다.

A. ENTRY_STAGING 중 POSE_STALE latch (3/3)
       205954 t=17.6 / 210209 t=12.5 / 210321 t=30.4
   pose_age 가 594~599ms 로 max_pose_age_s 를 한 번 넘긴 뒤, 관측이 완전히
   정상으로 돌아왔는데도 210209 는 44초, 210321 은 48초 동안 zero 로 굳었다
   (그 사이 pose_age 는 300ms 를 한 번도 넘지 않았다). 205954 만 우연한
   COMM resync 로 빠져나왔다.

B. plan/route 의 중간 경유점이 횡방향으로 붕괴 (205954, 3회)
       route 2  자세 (173.7,526.6, 5도) -> wp1 (175,600)
       route 4  자세 (194.5,530.6, 6도) -> wp1 (195,600)
   along +7mm / cross +70mm / bearing 84도 — 최소 선회원 **안쪽**이라
   전진으로 곧장 못 간다. replan 이 같은 공식으로 같은 점을 다시 만들었다.

C. 초기 staging 이 슬롯을 순회 (210321)
       B1 -> A1 -> A2 -> A3 거절 -> A4 선택. B1 은 점유도 장애물도 아니었다.

D. PARKING 단계에 같은 슬롯 재배치가 없음 (205954 t=44.4)
       (437.7,646.4,26도) 에서 B1 인계는 가능한데 rear route 는 불가,
       유일한 setup 은 B1 을 잃어 가드가 차단 -> NO_SAFE_PARKING_RECOVERY.
       가드 판정은 옳았고, 없던 것은 "그럼 B1 로 갈 자세를 다시 만든다" 였다.

네 수정 모두 기존 lifecycle/예산/planner 를 재사용한다. 새 state·flag 없음.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from controller.config import ControllerConfig
from parking.waypoints import (MIN_TURN_RADIUS_MM, InfeasibleRouteError,
                               build_waypoints, default_slot_specs,
                               forward_reachable, plan_handoff)
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView
from pipeline.tests.test_entry_staging_handoff import settle_staging_plan
from rl.parking_env import SLOT_NAMES

OBS_PERIOD_S = 0.25

# ── 실차 자세 ───────────────────────────────────────────────────────────────
# 205954: ENTRY_STAGING 중 POSE_STALE 이 걸린 자세
POSE_205954_STALE = (180.8, 367.5, 69.4)
POSE_210209_STALE = (180.8, 367.5, 69.4)
POSE_210321_STALE = (276.6, 543.8, 58.7)
# 205954: 횡방향 붕괴 waypoint 를 받은 두 자세
POSE_205954_ROUTE2 = (173.7, 526.6, 5.0)
POSE_205954_ROUTE4 = (194.5, 530.6, 6.0)
# 205954: B1 인계는 가능한데 rear route 가 불가했던 최종 자세
POSE_205954_FINAL = (437.7, 646.4, 26.0)
# 210321: 입구 시작 자세 (여기서 B1 이 A4 로 바뀌었다)
POSE_210321_ENTRANCE = (134.3, 178.8, 95.2)


# ══ 공용 stub ═══════════════════════════════════════════════════════════════

class _Authority:
    def __init__(self) -> None:
        self.is_faulted = False
        self.fault_reason = ""

    def fault(self, reason: str = "STOP") -> None:
        self.is_faulted, self.fault_reason = True, reason


class _Host:
    def __init__(self) -> None:
        self.authority = _Authority()
        self.re_arms = 0

    def re_arm_auto(self) -> None:
        self.authority.is_faulted = False
        self.authority.fault_reason = ""
        self.re_arms += 1


class _Scheduler:
    def __init__(self) -> None:
        self.running = True

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class _Runner:
    def __init__(self) -> None:
        self.loaded: list[list] = []
        self.host = _Host()
        self.scheduler = _Scheduler()
        self.replan_reason = None
        self.stopped = False

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        self.stopped = True
        self.host.authority.fault("POSE_STALE")
        self.scheduler.stop()


class _Dashboard:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def push_event(self, name, **fields) -> None:
        self.events.append((name, fields))


class _Server:
    def __init__(self) -> None:
        self.stopped: list[int] = []

    def stop_control(self, car_id) -> None:
        self.stopped.append(car_id)

    def hold_control(self, car_id) -> None:
        self.stopped.append(car_id)


# ══ PHASE A — ENTRY_STAGING STALE LIVENESS ══════════════════════════════════

class EntryStagingSurvivesACameraGap(unittest.TestCase):
    """staging 중 관측 공백 1회가 미션을 영구 정지시키지 않는다."""

    def _pipeline(self, stage: str, pose, fault: str = "POSE_STALE"):
        p = ParkingPipeline.__new__(ParkingPipeline)
        p.config = SimpleNamespace(
            parking_mode="rear", max_parking_recovery_attempts=3,
            max_replan_attempts=3, max_entry_staging_attempts=3,
            initial_pose_stability_mm=30.0, stationary_tolerance_mm=15.0,
            stationary_window=3, critical_heading_wait_timeout_s=2.5,
            parking_stall_timeout_s=8.0,
            entry_staging_heading_tolerance_deg=15.0,
            entry_staging_alignment_max_mm=1100.0,
            entry_staging_min_clearance_mm=35.0,
            controller_config=ControllerConfig())
        runner = _Runner()
        p.auto_hosts = {1: runner}
        p._auto_host_slot = {1: "B1"}
        p.track_of_car = {1: 7}
        view = VehicleView(track_id=7, car_id=1, node="entrance",
                           position_mm=(pose[0], pose[1]), heading_deg=pose[2],
                           heading_source="FRONT_CUSHION")
        view.recent.extend([view.position_mm] * 3)
        view.last_obs_time = 12.47
        p.views = {7: view}
        p._parking_stage = {1: stage}
        runner.host.authority.fault(fault)
        runner.scheduler.stop()
        p._parking_setup_wait = {}
        p._parking_plan_wait = {}
        p._entry_staging_wait = {}
        p._entry_staging_attempts = {}
        p._parking_recovery_attempts = {}
        p._initial_pose_samples = {}
        p._heading_wait_state = {}
        p._heading_wait_started = {}
        p._heading_wait_faulted = set()
        p._heading_fault_hold = set()
        p.dashboard = _Dashboard()
        p.server = _Server()
        p.orchestrator = SimpleNamespace(next_route_id=lambda: 99)
        p.on_route_load = None
        p._auto_host_route = {}
        p.on_event_record = None
        p.events = []
        p._trajectory_safe = lambda view, route, **kw: True
        p._load_direct_rear_replan = lambda car_id, view: False
        p._build_route = lambda *_: (_ for _ in ()).throw(
            InfeasibleRouteError("B1", "no direct route mid-staging"))
        return p, view, runner

    def test_entry_staging_is_an_observation_resumable_stage(self) -> None:
        p, _v, _r = self._pipeline("ENTRY_STAGING", POSE_210209_STALE)
        self.assertTrue(p._stale_observation_latched(1))

    def test_205954_stale_replay(self) -> None:
        self._assert_resumes(POSE_205954_STALE)

    def test_210209_stale_replay(self) -> None:
        self._assert_resumes(POSE_210209_STALE)

    def test_210321_stale_replay(self) -> None:
        self._assert_resumes(POSE_210321_STALE)

    def _assert_resumes(self, pose) -> None:
        p, view, _runner = self._pipeline("ENTRY_STAGING", pose)
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        self.assertTrue(p._maybe_resume_heading_fault(view))
        # staging 중이던 차는 staging 경계로 되돌아간다 (기존 계약 지점).
        self.assertEqual(p._parking_stage.get(1), "ENTRY_STAGING_PENDING")
        self.assertEqual(p._entry_staging_wait.get(1), view.last_obs_time)
        self.assertEqual(p._parking_recovery_attempts.get(1, 0), 0,
                         "관측 지연으로 복구 예산을 태우면 안 된다")

    def test_zero_safety_still_fires_first(self) -> None:
        """POSE_STALE zero 자체는 약화하지 않는다."""
        p, _view, runner = self._pipeline("ENTRY_STAGING", POSE_210209_STALE)
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertFalse(runner.scheduler.running)
        self.assertEqual(runner.loaded, [])

    def test_no_motion_before_a_validated_fresh_pose_route(self) -> None:
        p, view, runner = self._pipeline("ENTRY_STAGING", POSE_210209_STALE)
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        p._maybe_resume_heading_fault(view)
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertEqual(runner.loaded, [])

    def test_the_resume_runs_exactly_once(self) -> None:
        p, view, _runner = self._pipeline("ENTRY_STAGING", POSE_210209_STALE)
        for _ in range(8):
            view.last_obs_time += OBS_PERIOD_S
            view.recent.append(view.position_mm)
            p._maybe_resume_heading_fault(view)
        recovered = [e for e in p.dashboard.events
                     if e[0] == "heading_recovered"]
        self.assertEqual(len(recovered), 1, f"{len(recovered)}회 재진입")

    def test_a_moving_car_still_waits_for_a_physical_stop(self) -> None:
        p, view, _runner = self._pipeline("ENTRY_STAGING", POSE_210209_STALE)
        view.recent.clear()
        view.recent.extend([(180.0, 360.0), (220.0, 420.0), (260.0, 480.0)])
        view.position_mm = (260.0, 480.0)
        view.last_obs_time += OBS_PERIOD_S
        self.assertFalse(p._maybe_resume_heading_fault(view))
        self.assertEqual(p._parking_stage.get(1), "ENTRY_STAGING")

    def test_an_untrusted_heading_does_not_resume(self) -> None:
        p, view, _runner = self._pipeline("ENTRY_STAGING", POSE_210209_STALE)
        view.heading_source = "LAST_VALID"
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        self.assertFalse(p._maybe_resume_heading_fault(view))
        self.assertEqual(p._parking_stage.get(1), "ENTRY_STAGING")

    def test_physical_and_comm_faults_are_never_resumed(self) -> None:
        for reason in ("BOUNDARY_HARD", "COMM_TIMEOUT", "UNSAFE_ROUTE",
                       "SOCKET_DISCONNECTED"):
            with self.subTest(reason=reason):
                p, view, runner = self._pipeline(
                    "ENTRY_STAGING", POSE_210209_STALE, fault=reason)
                self.assertFalse(p._stale_observation_latched(1))
                view.last_obs_time += OBS_PERIOD_S
                view.recent.append(view.position_mm)
                self.assertFalse(p._maybe_resume_heading_fault(view))
                self.assertEqual(p._parking_stage.get(1), "ENTRY_STAGING")
                self.assertEqual(runner.loaded, [])


# ══ PHASE B — KINEMATIC REACHABILITY OF GENERATED WAYPOINTS ═════════════════

class GeneratedWaypointsAreForwardReachable(unittest.TestCase):

    def test_the_predicate_is_the_minimum_turning_circle(self) -> None:
        """새 임계값이 아니라 MIN_TURN_RADIUS_MM 하나만 쓴다."""
        pose, heading = (0.0, 0.0), 0.0
        # 좌우 선회원 중심에서 반경 안쪽인 점은 전진 불가.
        self.assertFalse(forward_reachable(
            pose, heading, (0.0, MIN_TURN_RADIUS_MM * 0.5), MIN_TURN_RADIUS_MM))
        # 정면 먼 곳은 가능.
        self.assertTrue(forward_reachable(
            pose, heading, (2000.0, 0.0), MIN_TURN_RADIUS_MM))

    def test_the_recorded_205954_lead_points_were_unreachable(self) -> None:
        """이 수정의 출발점 — 생성 순간부터 전진 불가였다."""
        for pose, lead in ((POSE_205954_ROUTE2, (175.0, 600.0)),
                           (POSE_205954_ROUTE4, (195.0, 600.0))):
            with self.subTest(pose=pose):
                self.assertFalse(forward_reachable(
                    pose[:2], pose[2], lead, MIN_TURN_RADIUS_MM))

    def test_route2_and_route4_no_longer_contain_a_lateral_collapse(self):
        b1 = default_slot_specs()["B1"]
        for pose in (POSE_205954_ROUTE2, POSE_205954_ROUTE4):
            with self.subTest(pose=pose):
                wps = build_waypoints(b1, route_id=1, from_pose=pose[:2],
                                      from_heading_deg=pose[2],
                                      min_radius_mm=MIN_TURN_RADIUS_MM,
                                      strict=True)
                self.assertTrue(wps)
                for wp in wps[:-1]:
                    self.assertTrue(
                        forward_reachable(pose[:2], pose[2], (wp.x, wp.y),
                                          MIN_TURN_RADIUS_MM),
                        f"중간 경유점 ({wp.x:.0f},{wp.y:.0f}) 이 선회원 안이다")

    def test_the_handoff_point_itself_is_never_dropped(self) -> None:
        """'선회원 안' 은 곧장 못 간다는 뜻이지 도달 불가가 아니다."""
        b1 = default_slot_specs()["B1"]
        for pose in (POSE_205954_ROUTE2, POSE_205954_ROUTE4,
                     (344.2, 592.5, 18.0)):
            with self.subTest(pose=pose):
                wps = build_waypoints(b1, route_id=1, from_pose=pose[:2],
                                      from_heading_deg=pose[2],
                                      min_radius_mm=MIN_TURN_RADIUS_MM,
                                      strict=True)
                last = wps[-1]
                self.assertTrue(last.is_final)
                self.assertAlmostEqual(last.x, 425.0, delta=1.0)
                self.assertAlmostEqual(last.y, 600.0, delta=1.0)

    def test_a_replan_from_the_next_pose_is_also_clean(self) -> None:
        """같은 붕괴를 replan 이 다시 만들지 않는다."""
        b1 = default_slot_specs()["B1"]
        pose = POSE_205954_ROUTE2
        for _ in range(3):
            wps = build_waypoints(b1, route_id=1, from_pose=pose[:2],
                                  from_heading_deg=pose[2],
                                  min_radius_mm=MIN_TURN_RADIUS_MM, strict=True)
            for wp in wps[:-1]:
                self.assertTrue(forward_reachable(pose[:2], pose[2],
                                                  (wp.x, wp.y),
                                                  MIN_TURN_RADIUS_MM))
            # 차가 20mm 전진했다고 보고 다시 계획한다.
            a = math.radians(pose[2])
            pose = (pose[0] + 20.0 * math.cos(a),
                    pose[1] + 20.0 * math.sin(a), pose[2])

    def test_known_good_aisle_routes_are_unchanged(self) -> None:
        """통로에 정렬된 정상 자세에서는 아무것도 지워지지 않는다."""
        cases = [("A3", (478.0, 515.2), 13.3), ("A3", (600.0, 600.0), 0.0),
                 ("B3", (300.0, 600.0), 0.0), ("A2", (200.0, 605.0), 0.0)]
        for slot, pose, heading in cases:
            with self.subTest(slot=slot, pose=pose):
                wps = build_waypoints(default_slot_specs()[slot], route_id=1,
                                      from_pose=pose, from_heading_deg=heading,
                                      min_radius_mm=MIN_TURN_RADIUS_MM,
                                      strict=True)
                self.assertTrue(wps)
                for wp in wps:
                    self.assertTrue(forward_reachable(pose, heading,
                                                      (wp.x, wp.y),
                                                      MIN_TURN_RADIUS_MM))

    def test_no_heading_means_no_filtering(self) -> None:
        """heading 을 모르면 판단하지 않는다 (기존 동작 보존)."""
        wps = build_waypoints(default_slot_specs()["A3"], route_id=1,
                              from_pose=(300.0, 600.0), from_heading_deg=None,
                              min_radius_mm=MIN_TURN_RADIUS_MM)
        self.assertTrue(wps)


# ══ PHASE C — INITIAL STAGING KEEPS THE RESERVED SLOT ═══════════════════════

class InitialStagingNeverSwapsTheReservedSlot(unittest.TestCase):

    def _pipeline(self):
        p = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                           parking_mode="rear"))
        runner = _Runner()
        p.auto_hosts = {1: runner}
        p.track_of_car = {1: 2}
        view = VehicleView(track_id=2, car_id=1, node="entrance",
                           position_mm=POSE_210321_ENTRANCE[:2],
                           heading_deg=POSE_210321_ENTRANCE[2],
                           heading_source="FRONT_CUSHION", last_obs_time=1.0)
        p.views = {2: view}
        p.allocator.update(2, view.position_mm)
        p.allocator.reassign(2, "B1")
        p._auto_host_slot[1] = "B1"
        p.events = []
        p.on_event_record = lambda n, **f: p.events.append((n, f))
        return p, view, runner

    def test_210321_b1_to_a4_regression_is_blocked(self) -> None:
        p, view, runner = self._pipeline()
        p._build_entry_staging_route = (
            lambda car, v, slot, rid, goal_test=None: [])
        p._start_entry_staging(1, view, "B1", 1, initial=True)
        settle_staging_plan(p, view)
        tried = [f.get("slot") for n, f in p.events
                 if n == "ENTRY_STAGING_CANDIDATE"]
        self.assertEqual(set(tried), {"B1"}, f"다른 슬롯 시도: {tried}")
        self.assertNotIn("A4", [f.get("slot") for n, f in p.events
                                if n == "SLOT_SELECTED"])
        self.assertEqual(p._auto_host_slot[1], "B1")
        self.assertEqual(p.allocator.vehicles[2].assigned_slot, "B1")

    def test_exhausted_staging_is_a_safe_stop_with_the_slot_reserved(self):
        p, view, runner = self._pipeline()
        p._build_entry_staging_route = (
            lambda car, v, slot, rid, goal_test=None: [])
        p._start_entry_staging(1, view, "B1", 1, initial=True)
        settle_staging_plan(p, view)
        self.assertEqual(p._parking_stage[1], "WAIT_ENTRY_STAGING_FAILED")
        self.assertTrue(runner.stopped)
        self.assertGreaterEqual(
            p.allocator.slot_statuses[SLOT_NAMES.index("B1")], 0.5)
        self.assertEqual(p.allocator.vehicles[2].assigned_slot, "B1")

    def test_a_solvable_entrance_still_loads_the_reserved_slot(self) -> None:
        p, view, runner = self._pipeline()
        p._start_entry_staging(1, view, "B1", 1, initial=False)
        settle_staging_plan(p, view)
        if runner.loaded:
            self.assertEqual(p._auto_host_slot[1], "B1")
            self.assertEqual(p._parking_stage[1], "ENTRY_STAGING")

    def test_the_budget_is_the_existing_one(self) -> None:
        p, view, _runner = self._pipeline()
        p._entry_staging_attempts[1] = PipelineConfig().max_entry_staging_attempts
        p._start_entry_staging(1, view, "B1", 1)
        self.assertEqual(p._parking_stage.get(1), "WAIT_ENTRY_STAGING_FAILED")


# ══ PHASE D — PARKING-STAGE SAME-SLOT REPOSITION ════════════════════════════

class ParkingStageRepositionsInsteadOfGivingUp(unittest.TestCase):

    def _pipeline(self):
        p = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                           parking_mode="rear"))
        runner = _Runner()
        runner.host.authority.is_faulted = False
        p.auto_hosts = {1: runner}
        p.track_of_car = {1: 2}
        p._auto_host_slot[1] = "B1"
        view = VehicleView(track_id=2, car_id=1, node="B1_front",
                           position_mm=POSE_205954_FINAL[:2],
                           heading_deg=POSE_205954_FINAL[2],
                           heading_source="FRONT_CUSHION", last_obs_time=50.0)
        view.recent.extend([view.position_mm] * 3)
        p.views = {2: view}
        p.allocator.update(2, view.position_mm)
        p.allocator.reassign(2, "B1")
        p.events = []
        p.on_event_record = lambda n, **f: p.events.append((n, f))
        return p, view, runner

    def test_the_recorded_situation_is_reproduced(self) -> None:
        """B1 인계는 가능한데 rear route 는 불가한 자세."""
        from parking.waypoints import build_rear_candidate_waypoints
        b1 = default_slot_specs()["B1"]
        self.assertTrue(plan_handoff(
            b1, from_pose=POSE_205954_FINAL[:2],
            from_heading_deg=POSE_205954_FINAL[2]).feasible)
        with self.assertRaises(InfeasibleRouteError):
            build_rear_candidate_waypoints(
                b1, route_id=1, from_pose=POSE_205954_FINAL[:2],
                from_heading_deg=POSE_205954_FINAL[2], strict=True)

    def test_a_same_slot_reposition_is_loaded_instead_of_faulting(self) -> None:
        p, view, runner = self._pipeline()
        self.assertTrue(p._load_slot_reposition(1, view))
        self.assertEqual(p._auto_host_slot[1], "B1")
        self.assertEqual(p._parking_stage[1], "SETUP")
        self.assertTrue(runner.loaded)
        self.assertIn("SLOT_REPOSITION", [n for n, _ in p.events])

    def test_the_reposition_end_pose_restores_b1(self) -> None:
        p, view, runner = self._pipeline()
        self.assertTrue(p._load_slot_reposition(1, view))
        end = runner.loaded[-1][-1]
        self.assertTrue(plan_handoff(
            default_slot_specs()["B1"], from_pose=(end.x, end.y),
            from_heading_deg=end.target_heading_deg).feasible)

    def test_the_bad_setup_is_still_rejected_by_the_existing_guard(self) -> None:
        """가드는 KEEP — 이 수정은 가드를 우회하는 것이 아니다."""
        from parking.waypoints import build_setup_recovery_waypoints
        p, view, _runner = self._pipeline()
        bad = build_setup_recovery_waypoints(
            default_slot_specs()["B1"], route_id=1,
            from_pose=POSE_205954_FINAL[:2],
            from_heading_deg=POSE_205954_FINAL[2],
            min_executable_mm=p._setup_min_executable_mm())
        self.assertTrue(bad)
        self.assertFalse(p._setup_keeps_target_feasible(view, "B1", bad))

    def test_the_slot_is_never_changed_by_the_reposition(self) -> None:
        p, view, _runner = self._pipeline()
        before = list(p.allocator.slot_statuses)
        p._load_slot_reposition(1, view)
        self.assertEqual(before, list(p.allocator.slot_statuses))
        self.assertEqual(p.allocator.vehicles[2].assigned_slot, "B1")

    def test_it_is_bounded_by_the_existing_recovery_budget(self) -> None:
        p, view, _runner = self._pipeline()
        budget = PipelineConfig().max_parking_recovery_attempts
        p._parking_recovery_attempts[1] = budget
        self.assertFalse(p._load_slot_reposition(1, view))

    def test_no_solution_means_stop_with_the_slot_still_reserved(self) -> None:
        p, view, runner = self._pipeline()
        p._parking_reposition_goal = lambda slot: (lambda pose: False)
        self.assertFalse(p._load_slot_reposition(1, view))
        self.assertEqual(p._auto_host_slot[1], "B1")
        self.assertEqual(runner.loaded, [])
        self.assertGreaterEqual(
            p.allocator.slot_statuses[SLOT_NAMES.index("B1")], 0.5)


# ══ 이전 사이클 수정 회귀 ═══════════════════════════════════════════════════

class PreviousFixesStillHold(unittest.TestCase):

    def test_stop_footprint_guard_still_rejects_182908_route_13(self) -> None:
        from pipeline.tests.test_safety_liveness_commitment import (
            StopFootprintGuard, _route_from)
        from parking.trajectory_safety import validate_trajectory
        route = _route_from("run_20260904_182908", 13)
        if route is None:
            self.skipTest("route.json 없음")
        r = validate_trajectory(route, start_pose=(634.1, 663.6, 23.0),
                                target_slot="A2",
                                min_turn_radius_mm=MIN_TURN_RADIUS_MM,
                                stop_distance_mm=StopFootprintGuard.STOP_MM)
        self.assertFalse(r.safe)
        self.assertEqual(r.reason, "STOP_FOOTPRINT")

    def test_global_is_still_observation_resumable(self) -> None:
        self.assertIn("GLOBAL", ParkingPipeline._OBSERVATION_RESUMABLE_STAGES)

    def test_reverse_start_tolerance_is_unchanged(self) -> None:
        from parking.waypoints import REVERSE_START_HEADING_TOLERANCE_DEG
        self.assertEqual(REVERSE_START_HEADING_TOLERANCE_DEG, 5.0)

    def test_untouched_constants(self) -> None:
        from parking.waypoints import ON_AISLE_TOLERANCE_MM
        cfg, ctl = PipelineConfig(), ControllerConfig()
        self.assertEqual(ON_AISLE_TOLERANCE_MM, 80.0)
        self.assertEqual(MIN_TURN_RADIUS_MM, 610.0)
        self.assertEqual(cfg.boundary_hard_margin_mm, 20.0)
        self.assertEqual(cfg.max_entry_staging_attempts, 3)
        self.assertEqual(ctl.steer_kp, 1.6)
        self.assertEqual(ctl.max_pose_age_s, 0.5)
        self.assertEqual(ctl.stop_distance_cm, 3.0)


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()
