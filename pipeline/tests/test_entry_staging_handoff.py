"""ENTRY_STAGING 종료 조건: staging 은 목적지가 아니라 basin 진입이다.

실차 run_20260901_123352 / run_20260901_123533 에서 차는 입구를 벗어나 통로
영역까지 잘 이동했는데도 주차로 이어지지 않고 ENTRY_STAGING_EXHAUSTED /
POSE_STALE 로 끝났다.

원인은 경로 생성이 아니라 **종료 판정**이었다. staging 종료 게이트가
기하 근사(_entry_staging_ready = 통로 밴드 |y-600|<=80 AND heading 정렬
<=15도)였는데, 두 run 모두 첫 HEADING_OUT_OF_TOLERANCE 시점에 이미 정상
parking flow 가 경로를 만들 수 있는 자세였다:

    123352  (362.6, 564.4,  49.6도)  ->  B1 정상 경로 존재
    123533  (365.3, 558.7,  45.6도)  ->  B1 정상 경로 존재

heading 이 45~50도라 근사는 "아직 아니다" 라고 했고, staging 을 두 번 더
강제한 결과 차가 오히려 통로 밖으로 나가(y=489, |y-600|=111mm) 정상 경로가
**사라진** 자세에서 끝났다. 즉 추가 staging 이 상황을 악화시켰다.

여기서 고정하는 계약: staging 종료는 production planner 가 판단한다.
좌표 임계값이 아니고, heading tolerance 를 넓히는 것도 아니다.
"""

from __future__ import annotations

import time
import unittest

from parking.waypoints import AISLE_Y, ON_AISLE_TOLERANCE_MM
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView


def settle_staging_plan(pipe, view, timeout_s: float = 5.0) -> bool:
    """Drive the two-phase staging flow to completion in a test.

    후보 탐색은 카메라 콜백을 막지 않도록 배경 스레드에서 돈다. 프로덕션에서는
    다음 프레임이 결과를 적재하는데(_on_view -> _apply_entry_staging_plan),
    테스트는 프레임 루프를 돌리지 않으므로 여기서 그 한 프레임을 대신 준다.
    프로덕션 코드에 동기 실행 스위치를 만들지 않기 위한 테스트 전용 도우미다.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with pipe._lock:
            busy = view.car_id in pipe._entry_staging_planning
            ready = view.car_id in pipe._entry_staging_plan
        if ready:
            return pipe._apply_entry_staging_plan(view)
        if not busy:
            return False
        time.sleep(0.005)
    raise AssertionError("배경 staging 계획이 끝나지 않았다")

# ── 실차 로그에서 그대로 가져온 자세 ────────────────────────────────────────
# 첫 HEADING_OUT_OF_TOLERANCE 재계획 시점 (staging 이 포기 대신 인계했어야 할 곳)
POSE_123352_REPLAN = (362.6, 564.4, 49.6)
POSE_123533_REPLAN = (365.3, 558.7, 45.6)
# staging 을 더 강제한 뒤 도달한 최종 자세 (정상 경로가 사라진 곳)
POSE_123352_FINAL = (526.8, 489.2, 11.1)
# 2026-09-02 entrance regressions: staging failed at these fresh observations,
# but the production GLOBAL planner can still hand them to the parking boundary.
POSE_203231_HANDOFF = (407.0, 638.7, 46.5)
POSE_203616_TERMINAL = (443.6, 660.9, 44.6)
# run_20260903_022217: 입구 시작에서 유일하게 PARKED_OK 까지 간 인계 자세
POSE_022217_HANDOFF = (478.0, 515.2, 13.3)
# 통로 위 중앙 baseline
POSE_CENTER = (600.0, 600.0, 0.0)


def _pipeline() -> ParkingPipeline:
    return ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                          parking_mode="rear"))


def _view(pose, node="entrance") -> VehicleView:
    return VehicleView(track_id=2, car_id=1, node=node,
                       position_mm=(pose[0], pose[1]), heading_deg=pose[2],
                       heading_source="FRONT_CUSHION", last_obs_time=1.0)


class HandoffFeasibilityProbe(unittest.TestCase):
    """_handoff_feasible 은 production planner 의 답을 그대로 돌려준다."""

    def setUp(self) -> None:
        self.pipe = _pipeline()

    def _probe(self, pose, slot="B1"):
        view = _view(pose)
        self.pipe.views = {2: view}
        self.pipe.allocator.update(2, view.position_mm)
        return self.pipe._handoff_feasible(view, slot)

    # ── A/B: 두 실차 자세는 이미 인계 가능했다 ─────────────────────────────

    def test_123352_replan_pose_is_already_handoff_feasible(self):
        self.assertIsNotNone(self._probe(POSE_123352_REPLAN))

    def test_123533_replan_pose_is_already_handoff_feasible(self):
        self.assertIsNotNone(self._probe(POSE_123533_REPLAN))

    def test_203231_terminal_pose_has_a_safe_global_handoff(self):
        """인계점이 아직 앞에 있는 자세는 종전대로 인계 가능하다."""
        self.assertIsNotNone(self._probe(POSE_203231_HANDOFF))

    def test_203616_handoff_point_is_behind_the_car(self):
        """전진으로 멀어지기만 하는 목표는 인계 가능이라고 하지 않는다.

        (443.6,660.9,44.6도) 에서 B1 인계점 (425,600) 은 차 진행축 기준
        **뒤로 56.0mm**, 거리 63.7mm 다 (도착 반경 50mm 밖). 전진하면 거리가
        늘어난다. 예전에는 planner 가 feasible 을 줘서 staging 을 끝냈고,
        그 뒤 controller 가 첫 tick 에 PATH_DEVIATION 을 냈다 — 실측
        run_20260904_231157/_231338 의 along -29.9 / -19.6mm 와 같은 형태다.

        이제는 인계하지 않고 예약 슬롯을 유지한 채 재배치/정지로 간다.
        """
        import math
        from parking.waypoints import (default_slot_specs, build_waypoints,
                                       MIN_TURN_RADIUS_MM)
        pose = POSE_203616_TERMINAL
        wps = build_waypoints(default_slot_specs()["B1"], route_id=1,
                              from_pose=pose[:2], from_heading_deg=pose[2],
                              min_radius_mm=MIN_TURN_RADIUS_MM, strict=True)
        first = wps[0]
        angle = math.radians(pose[2])
        along = ((first.x - pose[0]) * math.cos(angle)
                 + (first.y - pose[1]) * math.sin(angle))
        self.assertLess(along, 0.0, "이 자세의 전제는 목표가 뒤에 있다는 것")
        self.assertIsNone(self._probe(POSE_203616_TERMINAL))

    def test_those_poses_fail_the_old_geometric_gate(self):
        """근사와 planner 의 답이 실제로 엇갈렸다는 것이 이 수정의 근거다.

        이게 같아지면 이 사이클의 변경은 의미가 없어진 것이므로 알아야 한다.
        """
        for pose in (POSE_123352_REPLAN, POSE_123533_REPLAN):
            with self.subTest(pose=pose):
                view = _view(pose)
                self.pipe.views = {2: view}
                self.assertFalse(self.pipe._entry_staging_ready(view, "B1"))

    # ── C: 정말 불가능한 자세는 계속 staging ───────────────────────────────

    def test_pose_outside_the_basin_is_not_feasible(self):
        """staging 을 더 강제한 끝에 도달한 자세에는 정상 경로가 없다.

        통로에서 111mm 벗어나 있다. 여기서는 인계하면 안 되고 staging 이
        계속되어야 한다 — 게이트가 무조건 통과하는 것이 아님을 고정한다.
        """
        self.assertGreater(abs(POSE_123352_FINAL[1] - AISLE_Y),
                           ON_AISLE_TOLERANCE_MM)
        self.assertIsNone(self._probe(POSE_123352_FINAL))

    def test_probe_is_read_only(self):
        """실패해도 슬롯을 unreachable 로 굳히거나 상태를 바꾸지 않는다.

        _feasible_route 는 실패 시 _reject_slot / WAIT_NO_FEASIBLE_SLOT /
        stop_control 을 하므로 탐침으로 쓸 수 없다. 그래서 별도 함수다.
        """
        view = _view(POSE_123352_FINAL)
        self.pipe.views = {2: view}
        self.pipe.allocator.update(2, view.position_mm)
        before_state = dict(self.pipe._allocation_state)
        self.assertIsNone(self.pipe._handoff_feasible(view, "B1"))
        self.assertEqual(self.pipe._unreachable_slots.get(1, set()), set())
        self.assertEqual(self.pipe._allocation_state, before_state)

    def test_probe_requires_trusted_heading(self):
        """LAST_VALID heading 으로는 인계 판정을 하지 않는다."""
        view = _view(POSE_123352_REPLAN)
        view.heading_source = "LAST_VALID"
        self.pipe.views = {2: view}
        self.pipe.allocator.update(2, view.position_mm)
        self.assertIsNone(self.pipe._handoff_feasible(view, "B1"))

    def test_probe_agrees_with_the_production_gate(self):
        """탐침이 통과시킨 자세는 실제 commit 경로도 통과해야 한다.

        둘이 갈라지면 staging 을 끝내 놓고 경로를 못 만드는 상태
        (NO_SAFE_GLOBAL_AFTER_STAGING) 가 된다. 둘 다 이제 **예약 슬롯
        하나만** 본다 — 다른 칸이 된다는 답은 종료 게이트에서 쓸 수 없다
        (그 답이 곧 재배정이었다: run_20260904_214712).
        """
        for pose in (POSE_123352_REPLAN, POSE_123533_REPLAN):
            with self.subTest(pose=pose):
                pipe = _pipeline()
                view = _view(pose)
                pipe.views = {2: view}
                pipe.allocator.update(2, view.position_mm)
                probed = pipe._handoff_feasible(view, "B1")
                self.assertEqual(probed, "B1",
                                 "탐침은 예약 슬롯만 돌려줘야 한다")
                selected, wps = pipe._feasible_route(view, "B1", 99)
                self.assertEqual(selected, "B1")
                self.assertTrue(wps)

    def test_the_probe_never_answers_with_a_different_slot(self):
        """예약 슬롯이 안 되면 '다른 칸이 된다'가 아니라 None 이다.

        (600,600,0도) 에서 B1(x=425) 은 차량 뒤라 전진으로 못 간다. 예전에는
        탐침이 A3 를 돌려줘 staging 이 종료됐고, 곧바로 _feasible_route 가
        그 A3 로 재배정했다.
        """
        pipe = _pipeline()
        view = _view(POSE_CENTER)
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        self.assertIsNone(pipe._handoff_feasible(view, "B1"))
        selected, _wps = pipe._feasible_route(view, "B1", 99)
        self.assertIsNone(selected, "배정된 칸 외 다른 칸을 고르지 않는다")


class StagingTerminationGate(unittest.TestCase):
    """_maybe_resume_entry_staging 의 종료 분기."""

    def _staged_pipeline(self, pose):
        pipe = _pipeline()
        view = _view(pose)
        pipe.views = {2: view}
        pipe.track_of_car = {1: 2}
        pipe.allocator.update(2, view.position_mm)
        pipe._parking_stage[1] = "ENTRY_STAGING_PENDING"
        pipe._auto_host_slot[1] = "B1"
        return pipe, view

    def test_gate_accepts_a_planner_feasible_pose_that_the_proxy_rejects(self):
        """핵심 회귀: 근사는 거부하지만 planner 는 가능한 자세 → 인계."""
        pipe, view = self._staged_pipeline(POSE_123352_REPLAN)
        self.assertFalse(pipe._entry_staging_ready(view, "B1"))
        self.assertIsNotNone(pipe._handoff_feasible(view, "B1"))

    def test_gate_still_rejects_a_genuinely_infeasible_pose(self):
        pipe, view = self._staged_pipeline(POSE_123352_FINAL)
        self.assertFalse(pipe._entry_staging_ready(view, "B1"))
        self.assertIsNone(pipe._handoff_feasible(view, "B1"))

    def test_heading_tolerance_was_not_widened(self):
        """이 수정은 tolerance 를 넓혀서 통과시키는 것이 아니다."""
        self.assertEqual(
            PipelineConfig().entry_staging_heading_tolerance_deg, 15.0)


class CenterStartBaseline(unittest.TestCase):

    def test_center_start_never_enters_staging(self):
        """중앙 시작은 node 가 entrance 가 아니므로 staging 자체를 타지 않는다."""
        pipe = _pipeline()
        view = _view(POSE_CENTER, node="junction")
        pipe.views = {2: view}
        self.assertFalse(pipe._entry_staging_needed(view, "B1"))

    def test_center_start_keeps_its_assigned_slot(self):
        """중앙 시작에서도 배정된 칸은 바뀌지 않는다.

        (600,600,0도) 에서 B1(x=425)은 차량 뒤라 전진 경로가 없다. 예전에는
        여기서 도달 가능한 다른 칸으로 옮겼지만, allocate() 가 이미 B1 을
        예약으로 기록한 뒤이므로 그 대체는 배정을 뒤집는 것이다.
        """
        pipe = _pipeline()
        view = _view(POSE_CENTER, node="junction")
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        pipe.allocator.reassign(2, "B1")
        selected, _wps = pipe._feasible_route(view, "B1", 1)
        self.assertIsNone(selected)
        self.assertEqual(pipe.allocator.vehicles[2].assigned_slot, "B1")

    def test_a_reachable_assigned_slot_still_routes(self):
        """도달 가능하면 종전대로 그 칸으로 경로가 나온다."""
        pipe = _pipeline()
        view = _view(POSE_CENTER, node="junction")
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        pipe.allocator.reassign(2, "A3")
        selected, wps = pipe._feasible_route(view, "A3", 1)
        self.assertEqual(selected, "A3")
        self.assertTrue(wps)


class StagingRouteVisualization(unittest.TestCase):
    """요청문 12/17절: staging 중에도 overlay 에 경로가 보여야 한다."""

    def test_staging_route_reaches_the_overlay_source(self):
        """차는 움직이는데 화면이 wp0/0 이던 문제.

        _emit_route(recovery=True) 가 overlay 소스(_auto_host_route)를
        건너뛰어서, staging 실행 중 polyline 도 target marker 도 없었다.
        """
        pipe = _pipeline()
        view = _view((133.0, 209.0, 95.0))
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        route = pipe._build_entry_staging_route(1, view, "B1", 1)
        self.assertTrue(route, "staging route 자체가 없으면 이 검사는 무의미하다")

        pipe._emit_route(route, car_id=1, recovery=True, executing=True)
        shown = pipe._auto_host_route.get(1) or []
        self.assertEqual(len(shown), len(route))
        self.assertEqual([(w.x, w.y) for w in shown],
                         [(w.x, w.y) for w in route])

    def test_recorder_split_is_unchanged(self):
        """executing 은 화면용이고 recovery 는 기록기용이다 — 섞이지 않는다."""
        pipe = _pipeline()
        seen: list[tuple[int, bool]] = []
        pipe.on_route_load = lambda wps, recovery: seen.append((len(wps),
                                                                recovery))
        view = _view((133.0, 209.0, 95.0))
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        route = pipe._build_entry_staging_route(1, view, "B1", 1)
        pipe._emit_route(route, car_id=1, recovery=True, executing=True)
        self.assertEqual(seen, [(len(route), True)])

    def test_no_route_means_no_overlay_entry(self):
        """route 가 없으면 wp0/0 이 맞는 표시다."""
        pipe = _pipeline()
        self.assertEqual(pipe._auto_host_route.get(1, []), [])

    def test_default_emit_behaviour_is_unchanged(self):
        """executing 을 주지 않는 기존 호출부는 그대로다."""
        pipe = _pipeline()
        view = _view((133.0, 209.0, 95.0))
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        route = pipe._build_entry_staging_route(1, view, "B1", 1)
        pipe._emit_route(route, car_id=1, recovery=True)
        self.assertEqual(pipe._auto_host_route.get(1, []), [])
        pipe._emit_route(route, car_id=1)
        self.assertEqual(len(pipe._auto_host_route.get(1) or []), len(route))


class _StubRunner:
    def __init__(self) -> None:
        self.loaded: list[list] = []

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        pass


class StagingBoundaryEndToEnd(unittest.TestCase):
    """staging 경계 한 번을 실제로 돌려 상태 전이까지 확인한다."""

    def _run_boundary(self, pose, attempts: int = 0, slot: str = "B1"):
        pipe = _pipeline()
        runner = _StubRunner()
        pipe.auto_hosts = {1: runner}
        pipe.track_of_car = {1: 2}
        pipe._auto_host_slot[1] = slot
        pipe._parking_stage[1] = "ENTRY_STAGING_PENDING"
        pipe._entry_staging_attempts[1] = attempts
        view = _view(pose)
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        # 정지 확인용 관측 창 (_phase_boundary_stopped 의 전제)
        for _ in range(8):
            view.recent.append((pose[0], pose[1]))
        for i in range(6):
            view.last_obs_time = 100.0 + (i + 1) * 0.25
            pipe._maybe_resume_entry_staging(view)
            # 계획이 배경으로 나갔으면 그 결과를 적재하는 프레임을 한 번 준다.
            settle_staging_plan(pipe, view)
            if pipe._parking_stage.get(1) != "ENTRY_STAGING_PENDING":
                break
        return pipe, runner

    def test_misaligned_poses_keep_staging_instead_of_handing_off(self):
        """45도 부근에서 인계하지 않는다 — 그게 회귀의 출발점이었다.

        기록된 입구 인계 17건(ENTRY_STAGING_COMPLETE 이벤트)을 통로축
        heading 편차로 정리하면:

            편차 >= 39도 : 10건 -> PARKED 0건 (실패 8, FINAL_POSE_EVAL 2)
            편차 <= 23도 :  7건 -> PARKED 1건 (022217, 13.3도)

        편차 중앙값은 41.5도다. 40도대에서 주차에 성공한 적이 **한 번도
        없다**. 그런데
        "planner 가 경로를 만드는가" 는 heading 에 둔감해서(통로 (430,600)
        에서 0~50도 전부 동일하게 통과) 그 질문만으로 인계하면 40~47도에서
        끝난다. 그 자세를 받은 rear 계층은 setup 을 2wp 가 아니라 6wp 로
        풀어야 하고, 구간마다 추종 오차가 쌓여 replan churn 이 된다.
        """
        for pose in (POSE_123352_REPLAN, POSE_123533_REPLAN,
                     POSE_203231_HANDOFF, POSE_203616_TERMINAL):
            with self.subTest(pose=pose):
                pipe, _runner = self._run_boundary(pose)
                self.assertNotEqual(pipe._parking_stage.get(1), "GLOBAL")

    def test_aligned_pose_hands_off_immediately(self):
        """정렬되면 곧바로 인계한다 — 알려진 basin 이므로 더 돌 이유가 없다.

        위 misaligned 사례와 x/y 를 같게 두고 heading 만 바꾼다. B1 은
        x=425 이므로 (430,600) 에서 슬롯으로 가는 통로 진행 방향은 180도다.
        """
        aligned = (430.0, 600.0, 177.0)
        pipe, runner = self._run_boundary(aligned)
        self.assertTrue(pipe._entry_staging_ready(pipe.views[2], "B1"))
        self.assertEqual(pipe._parking_stage.get(1), "GLOBAL")
        self.assertEqual(pipe._allocation_state.get(1), "ROUTE_LOADED")
        self.assertEqual(len(runner.loaded), 1)

    def test_the_only_successful_entrance_handoff_is_preserved(self):
        """입구에서 유일하게 PARKED_OK 로 끝난 자세를 계속 인계해야 한다.

        run_20260903_022217: A3 인계 자세 (478.0, 515.2, 13.3도).
        heading 편차는 13.3도로 성공 basin 한가운데지만 통로 밴드로는
        |y-600|=84.8mm 로 4.8mm 벗어나 있다. 종료 조건에 밴드를 넣으면
        (= _entry_staging_ready 를 그대로 쓰면) 이 자세가 거부되고
        staging 이 더 돌아간다 — 유일한 성공을 깨는 수정이 된다.

        그래서 종료 게이트는 heading 만 보고, 위치/boundary 는
        _handoff_feasible 이 실제 경로를 만들어 검증한다.
        """
        pipe, runner = self._run_boundary(POSE_022217_HANDOFF, slot="A3")
        self.assertGreater(
            abs(POSE_022217_HANDOFF[1] - AISLE_Y), ON_AISLE_TOLERANCE_MM,
            "이 자세는 통로 밴드 밖이어야 회귀 테스트로 의미가 있다")
        self.assertEqual(pipe._parking_stage.get(1), "GLOBAL")
        self.assertEqual(len(runner.loaded), 1)

    def test_exhausted_budget_does_not_bypass_the_alignment_gate(self):
        """예산 소진은 재시도 상한이지 기하 조건의 면제가 아니다.

        예전 계약은 `feasible and (aligned or exhausted)` 였다. 재시도를 다
        썼다는 사실이 15도 정렬 게이트를 통과시키는 근거로 쓰였다.

        실측 run_20260904_231157 / _231338: SLOT_REPOSITION 이
        _start_entry_staging 을 통해 같은 예산을 쓰므로 3/3 이 금방 소진됐고,
        그 직후 exhausted=True 로 heading 86.9도 / 88.8도 — 통로축과 거의
        수직인 자세에서 인계했다. 그 자세의 인계 경로는 waypoint 1개였고 그
        하나가 차 뒤(along -29.9 / -19.6mm)라 1.5초 만에 PATH_DEVIATION 이
        났다. 이 파일이 인용하는 실측 통계와도 맞는다 — 통로축 편차 39도
        이상 인계 17건 중 PARKED 0건.

        예산이 끝나면 ENTRY_STAGING_EXHAUSTED 로 안전 정지하고 예약 슬롯은
        유지된다. 무한 staging 방지(123352)는 그 예산이 계속 담당한다.
        """
        pipe, runner = self._run_boundary(
            POSE_123352_REPLAN,
            attempts=PipelineConfig().max_entry_staging_attempts)
        self.assertNotEqual(pipe._parking_stage.get(1), "GLOBAL",
                            "미정렬 자세로 인계하면 안 된다")
        self.assertEqual(pipe._auto_host_slot.get(1), "B1",
                         "예약 슬롯은 유지된다")

    def test_an_aligned_pose_still_hands_off_within_budget(self):
        """정렬된 자세는 종전대로 인계한다 — 게이트를 좁힌 것이 아니다."""
        pipe, runner = self._run_boundary(POSE_022217_HANDOFF, slot="A3")
        self.assertEqual(pipe._parking_stage.get(1), "GLOBAL")
        self.assertEqual(len(runner.loaded), 1)

    def test_handoff_route_is_visible_on_the_overlay(self):
        pipe, runner = self._run_boundary(POSE_123352_REPLAN)
        self.assertEqual(len(pipe._auto_host_route.get(1) or []),
                         len(runner.loaded[0]))

    def test_infeasible_pose_keeps_staging_instead_of_handing_off(self):
        """게이트가 무조건 통과하는 것이 아님을 상태 전이로 확인한다."""
        pipe, _runner = self._run_boundary(POSE_123352_FINAL)
        self.assertEqual(pipe._parking_stage.get(1), "ENTRY_STAGING")
        self.assertNotEqual(pipe._allocation_state.get(1), "ROUTE_LOADED")

    def test_staging_continuation_is_also_visible_on_the_overlay(self):
        """staging 이 계속되는 경우에도 화면에 경로가 있어야 한다 (wp0/0 아님)."""
        pipe, _runner = self._run_boundary(POSE_123352_FINAL)
        self.assertTrue(pipe._auto_host_route.get(1))


if __name__ == "__main__":
    unittest.main()
