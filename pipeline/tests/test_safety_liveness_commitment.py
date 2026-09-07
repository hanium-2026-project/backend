"""세 개의 독립 계약: 경계 안전 / GLOBAL 생존성 / 슬롯 확약.

2026-09-04 수정 후 baseline 4회에서 서로 다른 세 문제가 드러났다.

A. run_20260904_182908 — 차체가 실제로 맵 밖 79.8mm 까지 나갔다 (SAFETY).
   A2 후면주차 candidate 의 ALIGN waypoint 가 (1050.0, 842.8, 30도) 였고,
   그 자세의 footprint 여유는 4.2mm 였다. 예측 경계 감시는 정상 동작했다 —
   overflow -29.0mm(맵 안)에서 zero 를 걸었다. 그런데 그 지점은 전진→후진
   전환점이라 차가 실제로 멈춰야 했고, |steering| 1.0 포화 + throttle 0.25
   상태의 관성이 135mm 를 더 밀어냈다. 즉 문제는 임계값이 아니라 **계획
   중심만 보고 정지한 차체를 보지 않은 것**이다.

B. run_20260904_183319 — 1회 530ms 관측 공백의 POSE_STALE latch 가 66초간
   풀리지 않았다 (LIVENESS). 그 66초 동안 pose_age 는 2.9~247ms 로 정상,
   comm_fault 0건, mission RUNNING 이었다. stage 가 GLOBAL 이었고 GLOBAL 이
   _OBSERVATION_RESUMABLE_STAGES 에 없었다.

C. 4/4 모두 preferred B1 을 즉시 포기하고 A2 로 갔다 (REPEATED). 그러나 그
   거부 자세 각각에서 기존 bounded planner 로 B1 인계 가능 자세에 닿는 해가
   존재한다. "이 자세에서 직접 경로 없음" 과 "그 슬롯이 도달 불가" 가 구분되지
   않았던 것이다.

세 수정 모두 기존 기하/상수만 재사용한다. 새 state 도 새 임계값도 없다.
"""

from __future__ import annotations

import json
import math
import os
import time
import unittest
from types import SimpleNamespace

from controller.config import ControllerConfig
from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (AISLE_Y, LOT_SIZE_MM, MIN_TURN_RADIUS_MM,
                               ON_AISLE_TOLERANCE_MM, InfeasibleRouteError,
                               _car_footprint, default_slot_specs,
                               plan_handoff)
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView
from pipeline.tests.test_entry_staging_handoff import settle_staging_plan

OBS_PERIOD_S = 0.25

# ── 실차 자세 ───────────────────────────────────────────────────────────────
# 수정 후 baseline 4회에서 B1/A1 이 거부된 바로 그 자세
REJECT_POSES = {
    "182908": (184.2, 497.7, 346.892),
    "183055": (150.3, 501.6, 355.121),
    "183319": (150.3, 487.2, 359.447),
    "183503": (142.6, 474.0, 2.067),
}
# 183319 가 POSE_STALE 로 굳은 GLOBAL 순항 자세
POSE_183319_GLOBAL = (456.3, 572.9, 12.3)


def _route_from(run: str, route_id: int | None = None):
    """기록된 route.json 을 그대로 waypoint 객체로 되살린다."""
    path = os.path.join("runs", run, "route.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if route_id is not None:
        rows = [r for r in rows if r["route_id"] == route_id]
    if not rows:
        return None
    rows.sort(key=lambda r: r["waypoint_id"])

    class _W:
        pass
    out = []
    for row in rows:
        w = _W()
        w.route_id = row["route_id"]
        w.waypoint_id = row["waypoint_id"]
        w.phase = row["phase"]
        w.x, w.y = row["x_mm"], row["y_mm"]
        w.target_heading_deg = row["target_heading_deg"]
        w.motion_direction = row["motion_direction"]
        w.curvature = row["curvature"]
        w.position_tolerance_cm = row["position_tolerance_cm"]
        out.append(w)
    return out


def _clearance(x, y, h):
    return min(min(px, LOT_SIZE_MM - px, py, LOT_SIZE_MM - py)
               for px, py in _car_footprint(x, y, h))


# ══ PHASE A — SAFETY ROUTE GUARD ════════════════════════════════════════════

class StopFootprintGuard(unittest.TestCase):
    """전/후진 전환점은 '정지한 차체' 로 판정한다."""

    STOP_MM = 10.0 * ControllerConfig().stop_distance_cm

    def test_the_allowance_is_derived_from_the_controller(self) -> None:
        """새 상수가 아니라 제어기가 아는 자기 정지거리다."""
        self.assertEqual(ControllerConfig().stop_distance_cm, 3.0)
        pipe = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                              parking_mode="rear"))
        self.assertAlmostEqual(pipe._route_stop_distance_mm(), 30.0)

    def test_the_recorded_182908_align_waypoint_is_4mm_from_the_wall(self):
        """이 수치가 이 수정의 출발점이다."""
        route = _route_from("run_20260904_182908", 13)
        if route is None:
            self.skipTest("run_20260904_182908/route.json 없음")
        align = [w for w in route if w.phase == "ALIGN"][0]
        self.assertAlmostEqual(
            _clearance(align.x, align.y, align.target_heading_deg), 4.2,
            delta=0.5)

    def test_182908_route_13_passes_without_the_guard(self) -> None:
        """가드 이전에는 통과했다 — 그래서 실차가 나갔다."""
        route = _route_from("run_20260904_182908", 13)
        if route is None:
            self.skipTest("route.json 없음")
        r = validate_trajectory(route, start_pose=(634.1, 663.6, 23.0),
                                target_slot="A2",
                                min_turn_radius_mm=MIN_TURN_RADIUS_MM)
        self.assertTrue(r.safe)
        self.assertLess(r.min_clearance_mm, 5.0)

    def test_182908_route_13_is_rejected_with_the_guard(self) -> None:
        route = _route_from("run_20260904_182908", 13)
        if route is None:
            self.skipTest("route.json 없음")
        r = validate_trajectory(route, start_pose=(634.1, 663.6, 23.0),
                                target_slot="A2",
                                min_turn_radius_mm=MIN_TURN_RADIUS_MM,
                                stop_distance_mm=self.STOP_MM)
        self.assertFalse(r.safe)
        self.assertEqual(r.reason, "STOP_FOOTPRINT")

    def test_known_good_rear_candidates_still_pass(self) -> None:
        """알려진 성공 경로는 하나도 막지 않는다."""
        cases = [("run_20260903_022217", 13, (478.0, 515.2, 13.3)),
                 ("run_20260904_183055", 4, (881.5, 620.6, 7.1)),
                 ("run_20260904_183503", 9, (601.0, 376.0, 53.0))]
        checked = 0
        for run, rid, start in cases:
            route = _route_from(run, rid)
            if route is None:
                continue
            with self.subTest(run=run):
                r = validate_trajectory(route, start_pose=start,
                                        target_slot="A2",
                                        min_turn_radius_mm=MIN_TURN_RADIUS_MM,
                                        stop_distance_mm=self.STOP_MM)
                self.assertTrue(r.safe, r.reason)
                checked += 1
        self.assertGreater(checked, 0, "검증할 기록 경로가 하나도 없다")

    def test_the_guard_is_off_by_default(self) -> None:
        """기존 호출부/테스트는 전혀 영향받지 않는다."""
        route = _route_from("run_20260904_182908", 13)
        if route is None:
            self.skipTest("route.json 없음")
        self.assertTrue(validate_trajectory(
            route, start_pose=(634.1, 663.6, 23.0), target_slot="A2",
            min_turn_radius_mm=MIN_TURN_RADIUS_MM).safe)

    def test_the_final_waypoint_is_never_penalised(self) -> None:
        """후면주차 FINAL 은 슬롯 뒤가 맵 경계인 것이 설계다."""
        route = _route_from("run_20260904_183055", 4)
        if route is None:
            self.skipTest("route.json 없음")
        final = route[-1]
        self.assertLess(_clearance(final.x, final.y,
                                   final.target_heading_deg), 40.0)
        self.assertTrue(validate_trajectory(
            route, start_pose=(881.5, 620.6, 7.1), target_slot="A2",
            min_turn_radius_mm=MIN_TURN_RADIUS_MM,
            stop_distance_mm=self.STOP_MM).safe)

    def test_runtime_boundary_thresholds_are_unchanged(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.boundary_hard_margin_mm, 20.0)
        self.assertEqual(cfg.boundary_measurement_uncertainty_mm, 10.0)


# ══ PHASE B — GLOBAL STALE LIVENESS ═════════════════════════════════════════

class _Authority:
    def __init__(self) -> None:
        self.is_faulted = False
        self.fault_reason = ""

    def fault(self, reason: str = "STOP") -> None:
        self.is_faulted, self.fault_reason = True, reason


class _Host:
    def __init__(self) -> None:
        self.authority = _Authority()

    def re_arm_auto(self) -> None:
        self.authority.is_faulted = False
        self.authority.fault_reason = ""


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

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
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


class GlobalCruiseSurvivesACameraGap(unittest.TestCase):
    """GLOBAL 순항도 관측-latch 재활성 대상이어야 한다 (183319)."""

    def _pipeline(self, stage: str, fault: str = "POSE_STALE"):
        p = ParkingPipeline.__new__(ParkingPipeline)
        p.config = SimpleNamespace(
            parking_mode="rear", max_parking_recovery_attempts=3,
            max_replan_attempts=3, initial_pose_stability_mm=30.0,
            stationary_tolerance_mm=15.0, stationary_window=3,
            critical_heading_wait_timeout_s=2.5, parking_stall_timeout_s=8.0,
            controller_config=ControllerConfig())
        runner = _Runner()
        p.auto_hosts = {1: runner}
        p._auto_host_slot = {1: "A2"}
        p.track_of_car = {1: 7}
        view = VehicleView(track_id=7, car_id=1,
                           position_mm=POSE_183319_GLOBAL[:2],
                           heading_deg=POSE_183319_GLOBAL[2],
                           heading_source="FRONT_CUSHION")
        view.recent.extend([view.position_mm] * 3)
        view.last_obs_time = 21.78
        p.views = {7: view}
        p._parking_stage = {1: stage}
        runner.host.authority.fault(fault)
        runner.scheduler.stop()
        p._parking_setup_wait = {}
        p._parking_plan_wait = {}
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
            InfeasibleRouteError("A2", "no direct rear from cruise"))
        return p, view, runner

    def test_global_is_an_observation_resumable_stage(self) -> None:
        p, _view, _runner = self._pipeline("GLOBAL")
        self.assertTrue(p._stale_observation_latched(1))

    def test_183319_replay_recovers_once_observations_return(self) -> None:
        """66초 정지의 회귀 잠금."""
        p, view, _runner = self._pipeline("GLOBAL")
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        self.assertTrue(p._maybe_resume_heading_fault(view))
        self.assertNotEqual(p._parking_stage.get(1), "GLOBAL")
        self.assertEqual(p._parking_recovery_attempts.get(1, 0), 0,
                         "관측 지연으로 복구 예산을 태우면 안 된다")

    def test_the_resume_runs_exactly_once(self) -> None:
        p, view, _runner = self._pipeline("GLOBAL")
        for _ in range(8):
            view.last_obs_time += OBS_PERIOD_S
            view.recent.append(view.position_mm)
            p._maybe_resume_heading_fault(view)
            p._maybe_start_parking_setup(view)
        recovered = [e for e in p.dashboard.events
                     if e[0] == "heading_recovered"]
        self.assertEqual(len(recovered), 1, f"{len(recovered)}회 재진입")

    def test_no_motion_before_a_validated_fresh_pose_route(self) -> None:
        p, view, runner = self._pipeline("GLOBAL")
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        p._maybe_resume_heading_fault(view)
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertFalse(runner.scheduler.running)
        self.assertEqual(runner.loaded, [])

    def test_a_moving_car_still_waits_for_a_physical_stop(self) -> None:
        p, view, _runner = self._pipeline("GLOBAL")
        view.recent.clear()
        view.recent.extend([(400.0, 500.0), (450.0, 540.0), (500.0, 580.0)])
        view.position_mm = (500.0, 580.0)
        view.last_obs_time += OBS_PERIOD_S
        self.assertFalse(p._maybe_resume_heading_fault(view))
        self.assertEqual(p._parking_stage.get(1), "GLOBAL")

    def test_an_untrusted_heading_does_not_resume(self) -> None:
        p, view, _runner = self._pipeline("GLOBAL")
        view.heading_source = "LAST_VALID"
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        self.assertFalse(p._maybe_resume_heading_fault(view))
        self.assertEqual(p._parking_stage.get(1), "GLOBAL")

    def test_a_physical_fault_during_global_is_never_resumed(self) -> None:
        """BOUNDARY_HARD / COMM 은 stage 와 무관하게 latch 유지."""
        for reason in ("BOUNDARY_HARD", "COMM_TIMEOUT", "UNSAFE_ROUTE"):
            with self.subTest(reason=reason):
                p, view, runner = self._pipeline("GLOBAL", fault=reason)
                self.assertFalse(p._stale_observation_latched(1))
                view.last_obs_time += OBS_PERIOD_S
                view.recent.append(view.position_mm)
                self.assertFalse(p._maybe_resume_heading_fault(view))
                self.assertEqual(p._parking_stage.get(1), "GLOBAL")
                self.assertEqual(runner.loaded, [])

    def test_terminal_and_confirmed_stages_are_still_excluded(self) -> None:
        # ENTRY_STAGING 은 이제 대상이다 (아래 별도 테스트). 나머지는 그대로.
        for stage in ("PARKED_VERIFY", "PARKED", "WAIT_SAFE_RECOVERY",
                      "WAIT_ENTRY_STAGING_FAILED", "ENTRY_STAGING_PENDING"):
            with self.subTest(stage=stage):
                p, _view, _runner = self._pipeline(stage)
                self.assertFalse(
                    p._stale_observation_latched(1),
                    "증거 없이 범위를 넓히지 않는다")

    def test_a_bounded_plan_exists_at_the_recorded_stale_pose(self) -> None:
        """되살려도 갈 곳이 있어야 의미가 있다 (실측 자세에서 확인)."""
        pipe = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                              parking_mode="rear"))
        from parking.waypoints import build_setup_recovery_waypoints
        wps = build_setup_recovery_waypoints(
            default_slot_specs()["A2"], route_id=1,
            from_pose=POSE_183319_GLOBAL[:2],
            from_heading_deg=POSE_183319_GLOBAL[2],
            min_executable_mm=pipe._setup_min_executable_mm())
        self.assertTrue(wps)


# ══ PHASE C — SLOT COMMITMENT ═══════════════════════════════════════════════

class PreferredSlotIsNotAbandonedWhileRepositionExists(unittest.TestCase):

    def _pipeline(self):
        p = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                           parking_mode="rear"))
        return p

    def _view(self, pose):
        return VehicleView(track_id=2, car_id=1, node="entrance",
                           position_mm=(pose[0], pose[1]), heading_deg=pose[2],
                           heading_source="FRONT_CUSHION", last_obs_time=1.0)

    # ── 사실 확인: 직접 경로는 없지만 슬롯은 도달 가능하다 ─────────────────

    def test_b1_direct_route_fails_at_all_four_reject_poses(self) -> None:
        p = self._pipeline()
        for run, pose in REJECT_POSES.items():
            with self.subTest(run=run):
                view = self._view(pose)
                p.views = {2: view}
                p.allocator.update(2, view.position_mm)
                self.assertTrue(p._preferred_slot_needs_reposition(view, "B1"))
                self.assertFalse(plan_handoff(
                    default_slot_specs()["B1"], from_pose=view.position_mm,
                    from_heading_deg=view.heading_deg).feasible)

    def test_a_bounded_reposition_exists_at_all_four_reject_poses(self) -> None:
        p = self._pipeline()
        for run, pose in REJECT_POSES.items():
            with self.subTest(run=run):
                view = self._view(pose)
                p.views = {2: view}
                p.allocator.update(2, view.position_mm)
                self.assertTrue(p._slot_reposition_available(view, "B1"))

    def test_the_parking_reposition_goal_is_plan_handoff_itself(self) -> None:
        """주차 단계 재배치는 새 기하를 만들지 않는다 (plan_handoff 그대로)."""
        p = self._pipeline()
        goal = p._parking_reposition_goal("B1")
        spec = default_slot_specs()["B1"]
        for pose in [(361.3, 520.5, 15.1), (150.3, 501.6, 355.1),
                     (600.0, 600.0, 0.0)]:
            with self.subTest(pose=pose):
                self.assertEqual(
                    goal(pose),
                    plan_handoff(spec, from_pose=(pose[0], pose[1]),
                                 from_heading_deg=pose[2]).feasible)

    def test_the_staging_reposition_goal_is_the_staging_contract(self) -> None:
        """입구 단계 재배치 목표 = staging 종료 계약 그 자체.

        run_20260905_024023 / _024123 실측: 이전 재배치 목표는 plan_handoff
        만 봤기 때문에 heading 73~80deg 인 자세를 '도달'로 인정했고, staging
        종료 게이트(밴드 AND 정렬)는 그 자세를 거부해서 재배치가 아무것도
        전진시키지 못했다. 두 술어는 같은 계약이어야 한다.
        """
        p = self._pipeline()
        view = self._view(REJECT_POSES["182908"])
        goal = p._entry_staging_goal(view, "B1")
        spec = default_slot_specs()["B1"]
        tol = float(p.config.entry_staging_heading_tolerance_deg)
        desired = p._entry_staging_heading(view, "B1")
        for pose in [(361.3, 520.5, 15.1), (150.3, 501.6, 355.1),
                     (600.0, 600.0, 0.0), (450.0, 600.0, 75.0),
                     (450.0, 600.0, 0.0), (450.0, 300.0, 0.0)]:
            with self.subTest(pose=pose):
                banded = abs(pose[1] - AISLE_Y) <= ON_AISLE_TOLERANCE_MM
                aligned = (desired is None
                           or p._heading_delta(pose[2], desired) <= tol)
                reachable = plan_handoff(
                    spec, from_pose=(pose[0], pose[1]),
                    from_heading_deg=pose[2]).feasible
                self.assertEqual(goal(pose), banded and aligned and reachable)

    def test_staging_reposition_goal_rejects_a_sideways_pose_in_band(self)            -> None:
        """밴드 안이어도 heading 70~80deg 는 목표가 아니다 (024023/024123)."""
        p = self._pipeline()
        view = self._view(REJECT_POSES["182908"])
        goal = p._entry_staging_goal(view, "B1")
        for heading in (70.0, 73.2, 75.0, 80.2):
            with self.subTest(heading=heading):
                self.assertFalse(goal((450.0, AISLE_Y, heading)))

    # ── lifecycle: 슬롯을 유지한 채 재배치한다 ─────────────────────────────

    def _staging_pipeline(self, pose, attempts=0):
        p = self._pipeline()
        runner = _Runner()
        p.auto_hosts = {1: runner}
        p.track_of_car = {1: 2}
        p._auto_host_slot[1] = "B1"
        p._parking_stage[1] = "ENTRY_STAGING_PENDING"
        p._entry_staging_attempts[1] = attempts
        view = self._view(pose)
        p.views = {2: view}
        p.allocator.update(2, view.position_mm)
        p.allocator.reassign(2, "B1")
        for _ in range(8):
            view.recent.append((pose[0], pose[1]))
        p.events = []
        p.on_event_record = lambda name, **f: p.events.append((name, f))
        return p, view, runner

    def _drive_boundary(self, p, view):
        for i in range(6):
            view.last_obs_time = 100.0 + (i + 1) * 0.25
            p._maybe_resume_entry_staging(view)
            settle_staging_plan(p, view)
            if p._parking_stage.get(1) != "ENTRY_STAGING_PENDING":
                break

    def test_replay_keeps_b1_and_emits_a_reposition(self) -> None:
        """4/4 실측 자세: 슬롯을 바꾸지 않고 재배치를 싣는다."""
        for run, pose in REJECT_POSES.items():
            with self.subTest(run=run):
                p, view, runner = self._staging_pipeline(pose)
                self._drive_boundary(p, view)
                names = [n for n, _ in p.events]
                self.assertIn("SLOT_REPOSITION", names)
                self.assertNotIn("SLOT_REJECTED", names)
                self.assertEqual(p._auto_host_slot[1], "B1")
                self.assertEqual(
                    p.allocator.vehicles[2].assigned_slot, "B1")
                self.assertTrue(runner.loaded, "재배치 경로가 실려야 한다")

    def test_the_reposition_end_pose_makes_b1_handoff_feasible(self) -> None:
        for run, pose in REJECT_POSES.items():
            with self.subTest(run=run):
                p, view, runner = self._staging_pipeline(pose)
                self._drive_boundary(p, view)
                if not runner.loaded:
                    self.fail("재배치 경로 없음")
                end = runner.loaded[-1][-1]
                self.assertTrue(plan_handoff(
                    default_slot_specs()["B1"],
                    from_pose=(end.x, end.y),
                    from_heading_deg=end.target_heading_deg).feasible)

    def test_slot_occupancy_bookkeeping_is_untouched(self) -> None:
        """재배치 중 예약이 풀려 다른 차에게 노출되면 안 된다."""
        from rl.parking_env import SLOT_NAMES
        for run, pose in REJECT_POSES.items():
            with self.subTest(run=run):
                p, view, _runner = self._staging_pipeline(pose)
                before = list(p.allocator.slot_statuses)
                self._drive_boundary(p, view)
                after = list(p.allocator.slot_statuses)
                self.assertEqual(before, after,
                                 "재배치는 점유/예약을 바꾸지 않는다")
                self.assertGreaterEqual(
                    after[SLOT_NAMES.index("B1")], 0.5,
                    "B1 예약이 유지되어야 다른 차가 못 가져간다")

    def test_reposition_is_bounded_by_the_existing_budget(self) -> None:
        """무한 반복 금지 — 기존 max_entry_staging_attempts 를 쓴다."""
        budget = PipelineConfig().max_entry_staging_attempts
        p, view, _runner = self._staging_pipeline(
            REJECT_POSES["183055"], attempts=budget)
        self._drive_boundary(p, view)
        names = [n for n, _ in p.events]
        self.assertNotIn("SLOT_REPOSITION", names,
                         "예산이 끝나면 재배치하지 않는다")

    def test_no_reposition_solution_stops_instead_of_switching_slots(self):
        """재배치 해가 없어도 다른 칸으로 넘어가지 않는다.

        예전 계약은 "그때는 종전대로 다음 슬롯"이었다. 실차
        run_20260904_214712 에서 그 경로가 그대로 재배정(B1->A3)이 됐다.
        이제는 예약 슬롯 하나만 평가하고, 안 되면 정지한다.
        """
        from rl.parking_env import SLOT_NAMES
        p, view, _runner = self._staging_pipeline(REJECT_POSES["183055"])
        p._slot_reposition_available = lambda v, s: False
        self._drive_boundary(p, view)
        names = [n for n, _ in p.events]
        self.assertNotIn("SLOT_REPOSITION", names)
        picked = {f.get("slot") for n, f in p.events
                  if n in ("SLOT_SELECTED", "ENTRY_STAGING_COMPLETE")}
        self.assertFalse(picked - {"B1"}, f"다른 칸이 선택됐다: {picked}")
        self.assertEqual(p._auto_host_slot[1], "B1")
        self.assertEqual(p.allocator.vehicles[2].assigned_slot, "B1")
        self.assertGreaterEqual(
            p.allocator.slot_statuses[SLOT_NAMES.index("B1")], 0.5)

    def test_a_directly_feasible_slot_never_repositions(self) -> None:
        """이미 갈 수 있으면 아무것도 하지 않는다 (022217 known-good 자세)."""
        p, view, _runner = self._staging_pipeline((478.0, 515.2, 13.3))
        p._auto_host_slot[1] = "A3"
        p.allocator.reassign(2, "A3")
        self.assertFalse(p._preferred_slot_needs_reposition(view, "A3"))


# ══ 회귀: 이미 확정된 계약들 ════════════════════════════════════════════════

class ExistingContractsUnchanged(unittest.TestCase):

    def test_controller_gains_and_limits_are_unchanged(self) -> None:
        cfg = ControllerConfig()
        self.assertEqual(cfg.steer_kp, 1.6)
        self.assertEqual(cfg.stop_distance_cm, 3.0)
        self.assertEqual(cfg.max_pose_age_s, 0.5)

    def test_staging_budget_and_tolerance_are_unchanged(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.max_entry_staging_attempts, 3)
        self.assertEqual(cfg.entry_staging_heading_tolerance_deg, 15.0)

    def test_aisle_band_is_unchanged(self) -> None:
        from parking.waypoints import ON_AISLE_TOLERANCE_MM
        self.assertEqual(ON_AISLE_TOLERANCE_MM, 80.0)

    def test_slot_geometry_is_unchanged(self) -> None:
        specs = default_slot_specs()
        self.assertEqual((specs["B1"].center_x, specs["B1"].center_y),
                         (425.0, 1050.0))
        self.assertEqual((specs["A2"].center_x, specs["A2"].center_y),
                         (650.0, 150.0))


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()
