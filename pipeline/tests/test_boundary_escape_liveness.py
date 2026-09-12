"""맵 밖으로 나간 차의 탈출 계약 + 중앙 시작 baseline 회귀.

실차 run_20260901_154551 은 입구에서 출발해 처음으로 슬롯(B1)까지 후면주차를
끝냈다. 그런데 주차 정렬(FINAL_ALIGNMENT)이 한 번도 돌지 않았다.

이유는 정렬 기능이 아니라 그 앞 단계다:

  1. FINAL 후진 중 카메라 프레임이 547ms 끊겼다 (정상 간격 ~266ms). 그동안
     차는 throttle -0.25 로 계속 후진했고, 다음 프레임에서 이미 78.7mm 를
     더 간 뒤였다. 최종 정지 자세는 슬롯 중심보다 71mm 깊었다.
  2. 그 자세는 맵 밖 47mm (기록상 56.8mm) 였다 -> BOUNDARY_HARD.
     mission 이 DONE 이 아니라 fault 로 갈라져 최종 자세 평가 자체가
     실행되지 않았다.
  3. 그리고 **빠져나올 수 없었다**. validate_trajectory 의 탈출 허용치가
     min(initial_boundary_tolerance_mm, initial_overflow) 라서 20mm 로
     묶여 있었고, 차가 56.8mm 나가 있으니 **자기 출발 자세부터** 거절됐다.
     planner 는 여유를 전혀 악화시키지 않는 탈출 경로를 만들었는데
     validator 가 세 번 모두 RECOVERY_REJECTED(MAP_FOOTPRINT) 로 막아
     WAIT_RECOVERY_EXHAUSTED 로 굳었다.

여기서 고정하는 것은 3번이다. 허용치는 "요구 여유 달성"이 아니라
"지금보다 나빠지지 않기" 여야 한다 — plan_setup_recovery.blocked() 와
슬롯 keepout 이 이미 쓰는 규칙과 같다. 맵을 넓히는 것이 아니다.

같은 파일에서 중앙 시작 baseline 이 최근 ENTRY_STAGING 작업으로 깨지지
않았다는 것도 함께 고정한다.
"""

from __future__ import annotations

import math
import unittest

from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (CAR_LENGTH_MM, PHASE_DEFAULTS,
                               _path_clearance,
                               build_setup_recovery_waypoints,
                               default_slot_specs)
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView

LOT = 1200.0

# run_20260901_154551 이 BOUNDARY_HARD 로 멈춘 실제 자세 (슬롯 B1 안, 너무 깊음)
STUCK_154551 = (479.8, 1121.5, 279.1)
# 중앙 baseline
POSE_222708 = (314.0, 572.0, 340.0)      # known-good 중앙 시작
POSE_153620 = (205.0, 630.6, 3.2)        # 2026-09-01 유일한 실제 중앙 시작
POSE_CENTER = (600.0, 600.0, 0.0)


def _pipeline() -> ParkingPipeline:
    return ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                          parking_mode="rear"))


def _view(pose, node="junction") -> VehicleView:
    return VehicleView(track_id=2, car_id=1, node=node,
                       position_mm=(pose[0], pose[1]), heading_deg=pose[2],
                       heading_source="FRONT_CUSHION", last_obs_time=1.0)


def _escape_route(pose, slot_id="B1"):
    return build_setup_recovery_waypoints(
        default_slot_specs()[slot_id], route_id=1, from_pose=(pose[0], pose[1]),
        from_heading_deg=pose[2], min_executable_mm=110.0)


class OutsideMapEscape(unittest.TestCase):
    """맵 밖에 걸친 차는 반드시 빠져나올 수 있어야 한다."""

    def test_the_stuck_pose_really_is_outside_the_map(self):
        overflow, _ = _path_clearance([STUCK_154551])
        self.assertGreater(overflow, PipelineConfig().boundary_hard_margin_mm)

    def test_planner_still_produces_an_escape_route(self):
        wps = _escape_route(STUCK_154551)
        self.assertTrue(wps)

    def test_escape_route_never_makes_the_overflow_worse(self):
        pose = STUCK_154551
        wps = _escape_route(pose)
        start_overflow, _ = _path_clearance([pose])
        route_overflow, _ = _path_clearance(
            [pose] + [(w.x, w.y, w.target_heading_deg) for w in wps])
        self.assertLessEqual(route_overflow, start_overflow + 1e-6)

    def test_validator_accepts_that_escape(self):
        """이게 실차에서 세 번 거절됐던 지점이다."""
        wps = _escape_route(STUCK_154551)
        result = validate_trajectory(
            wps, start_pose=STUCK_154551, target_slot="B1",
            initial_boundary_tolerance_mm=(
                PipelineConfig().boundary_hard_margin_mm))
        self.assertTrue(result.safe, result.reason)

    def test_a_route_that_goes_deeper_is_still_rejected(self):
        """탈출만 허용한다 — 맵이 넓어지는 것이 아니다."""
        wps = _escape_route(STUCK_154551)
        deeper = [type(wps[0])(**{**wps[0].__dict__, "x": 479.8, "y": 1180.0})]
        result = validate_trajectory(
            deeper, start_pose=STUCK_154551, target_slot="B1",
            initial_boundary_tolerance_mm=(
                PipelineConfig().boundary_hard_margin_mm))
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "MAP_FOOTPRINT")

    def test_inside_map_tolerance_is_unchanged(self):
        """맵 안에서 출발하면 허용치는 예전 그대로 20mm 다."""
        inside = (600.0, 620.0, 10.0)
        self.assertEqual(_path_clearance([inside])[0], 0.0)
        # 20mm 를 넘게 나가는 경로는 여전히 거절되어야 한다.
        wps = _escape_route(inside)
        self.assertTrue(wps)
        bad = [type(wps[0])(**{**wps[0].__dict__, "x": 600.0, "y": 1160.0})]
        result = validate_trajectory(
            bad, start_pose=inside, target_slot="B1",
            initial_boundary_tolerance_mm=20.0)
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "MAP_FOOTPRINT")


class FinalArrivalGeometry(unittest.TestCase):
    """FINAL 도착 허용오차와 슬롯 기하의 관계 (154551 의 물리적 배경).

    현재 값들은 서로 모순된다. 고치지는 않되, 숫자가 바뀌면 드러나도록
    고정해 둔다 — 어느 쪽을 바꿀지는 별도 판단이 필요하다.
    """

    def test_slot_end_is_the_map_edge(self):
        """B행 슬롯의 안쪽 끝이 곧 맵 경계다. 뒤에 여유 공간이 없다."""
        b1 = default_slot_specs()["B1"]
        self.assertAlmostEqual(b1.center_y + b1.length / 2, LOT, places=6)

    def test_geometric_margin_is_25mm(self):
        b1 = default_slot_specs()["B1"]
        self.assertAlmostEqual((b1.length - CAR_LENGTH_MM) / 2, 25.0)

    def test_final_position_tolerance_exceeds_that_margin(self):
        """도착 허용오차(50mm) > 기하 여유(25mm).

        즉 허용오차 가장자리에서 '도착' 판정을 받은 차는 이미 맵 밖
        25mm 이고, 그것만으로 boundary_hard_margin(20mm)을 넘는다.
        154551 은 여기에 관측 공백까지 겹쳐 47mm 를 넘었다.
        """
        b1 = default_slot_specs()["B1"]
        tol_mm = PHASE_DEFAULTS["FINAL"]["position_tolerance_cm"] * 10.0
        margin = (b1.length - CAR_LENGTH_MM) / 2
        self.assertGreater(tol_mm, margin)
        # 허용오차 끝에서 도착했을 때의 실제 초과량
        rear_y = b1.center_y + tol_mm + CAR_LENGTH_MM / 2
        self.assertGreater(rear_y - LOT,
                           PipelineConfig().boundary_hard_margin_mm)


class FinalAlignmentReachability(unittest.TestCase):
    """154551 에서 정렬이 안 돈 이유는 정렬 로직이 아니다."""

    def test_the_final_pose_would_have_been_judged_ALIGN(self):
        """평가만 돌았다면 올바르게 '정렬 필요' 라고 했을 자세다.

        즉 오판(PARKED_OK)이 아니라 **평가 자체가 실행되지 않은 것**이다.
        """
        from parking import final_alignment as fa
        b1 = default_slot_specs()["B1"]
        verdict = fa.evaluate_final_pose(b1, *STUCK_154551)
        self.assertEqual(verdict.action, "ALIGN")
        self.assertEqual(verdict.reason, "HEADING_NOT_PARALLEL")

    def test_the_final_pose_is_not_contained_in_the_slot(self):
        """'슬롯에 도착했다' 와 '주차 완료' 는 다르다."""
        from parking import final_alignment as fa
        b1 = default_slot_specs()["B1"]
        lateral, depth = fa.footprint_overflow_mm(b1, *STUCK_154551)
        self.assertGreater(lateral, 0.0)
        self.assertGreater(depth, 0.0)

    def test_ordinary_mis_parks_are_alignable_within_the_default_budget(self):
        """정상 범위의 어긋남은 기본 경로 예산(700mm)으로 정렬된다.

        154551 자세가 실패하는 것은 예산 문제가 아니라 그 자세가 이미
        맵 밖 56.8mm 라는 설계 범위 밖이기 때문이다 — 예산을 늘리는 것이
        답이 아니라는 근거로 고정한다.
        """
        from parking import final_alignment as fa
        from parking.waypoints import build_setup_recovery_waypoints
        b1 = default_slot_specs()["B1"]
        cases = ((40.0, 20.0, 8.0), (-40.0, 20.0, -8.0), (10.0, 0.0, 12.0),
                 (-10.0, 0.0, -12.0), (20.0, 60.0, 6.0), (-20.0, -60.0, -6.0))
        for lateral, depth, herr in cases:
            with self.subTest(lateral=lateral, depth=depth, herr=herr):
                pose = (b1.center_x + lateral, b1.center_y + depth,
                        (270.0 + herr) % 360.0)
                wps = build_setup_recovery_waypoints(
                    b1, route_id=50, from_pose=(pose[0], pose[1]),
                    from_heading_deg=pose[2],
                    goal_test=fa.alignment_goal_test(b1))
                self.assertTrue(wps)


class CenterStartBaseline(unittest.TestCase):
    """최근 ENTRY_STAGING 작업이 중앙 시작 경로를 건드리지 않았다."""

    def _probe(self, pose):
        pipe = _pipeline()
        view = _view(pose)
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        return pipe, view

    def test_222708_known_good_center_still_routes(self):
        pipe, view = self._probe(POSE_222708)
        self.assertFalse(pipe._entry_staging_needed(view, "B1"))
        slot, wps = pipe._feasible_route(view, "B1", 1)
        self.assertIsNotNone(slot)
        self.assertTrue(wps)

    def test_153620_center_start_still_routes(self):
        pipe, view = self._probe(POSE_153620)
        self.assertFalse(pipe._entry_staging_needed(view, "B1"))
        slot, wps = pipe._feasible_route(view, "B1", 1)
        self.assertIsNotNone(slot)
        self.assertTrue(wps)

    def test_center_never_enters_entry_staging(self):
        for pose in (POSE_222708, POSE_153620, POSE_CENTER):
            with self.subTest(pose=pose):
                pipe, view = self._probe(pose)
                self.assertFalse(pipe._entry_staging_needed(view, "B1"))

    def test_center_path_never_calls_the_handoff_probe(self):
        """_handoff_feasible 은 ENTRY_STAGING_PENDING 에서만 호출된다.

        중앙 시작은 그 단계를 거치지 않으므로 비싼 탐침을 절대 타지 않는다.
        """
        pipe, view = self._probe(POSE_CENTER)
        calls = []
        original = pipe._handoff_feasible
        pipe._handoff_feasible = lambda *a, **k: (calls.append(a) or
                                                  original(*a, **k))
        self.assertFalse(pipe._maybe_resume_entry_staging(view))
        self.assertEqual(calls, [])


class SharedVerdictEquivalence(unittest.TestCase):
    """_trajectory_verdict 분리가 판정 의미를 바꾸지 않았다."""

    def _cases(self):
        specs = default_slot_specs()
        for pose in (POSE_CENTER, POSE_222708, POSE_153620, STUCK_154551):
            wps = build_setup_recovery_waypoints(
                specs["B1"], route_id=1, from_pose=(pose[0], pose[1]),
                from_heading_deg=pose[2], min_executable_mm=110.0)
            if wps:
                yield pose, wps

    def test_verdict_matches_the_old_inline_branching(self):
        """예전 인라인 구조를 그대로 재현해 결과를 대조한다."""
        pipe = _pipeline()
        for pose, wps in self._cases():
            view = _view(pose)
            pipe.views = {2: view}
            pipe.allocator.update(2, view.position_mm)

            result, reason = pipe._trajectory_verdict(view, wps, "B1")

            # --- 예전 코드의 분기 구조 ---
            if (view.heading_deg is None
                    or view.heading_source not in {"FRONT_CUSHION",
                                                   "TRAJECTORY"}):
                old_reason, old_result = "NO_FRESH_HEADING", None
            else:
                _obs, uncertain = pipe._planning_obstacle_snapshot(view)
                if uncertain:
                    old_reason, old_result = "OTHER_VEHICLE_POSE_UNCERTAIN", None
                else:
                    old_result = result       # 같은 validate_trajectory 호출
                    old_reason = result.reason
            with self.subTest(pose=pose):
                self.assertEqual(reason, old_reason)
                self.assertIs(result, old_result)

    def test_verdict_has_no_side_effects(self):
        """탐침 경로가 제어/상태를 건드리면 안 된다 (요청문 7절)."""
        pipe = _pipeline()
        pose, wps = next(iter(self._cases()))
        view = _view(pose)
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        before = (dict(pipe._allocation_state), dict(pipe._parking_stage),
                  dict(pipe._auto_host_slot),
                  {k: set(v) for k, v in pipe._unreachable_slots.items()},
                  dict(pipe._auto_host_route))
        pipe._trajectory_verdict(view, wps, "B1")
        after = (dict(pipe._allocation_state), dict(pipe._parking_stage),
                 dict(pipe._auto_host_slot),
                 {k: set(v) for k, v in pipe._unreachable_slots.items()},
                 dict(pipe._auto_host_route))
        self.assertEqual(before, after)

    def test_trajectory_safe_still_reports_failures(self):
        """분리 후에도 실패 시 이벤트/정지 처리는 남아 있어야 한다."""
        pipe = _pipeline()
        events: list[str] = []
        pipe.on_event_record = lambda name, **kw: events.append(name)
        view = _view(STUCK_154551)
        view.heading_source = "LAST_VALID"       # NO_FRESH_HEADING 경로
        pipe.views = {2: view}
        wps = _escape_route(STUCK_154551)
        self.assertFalse(pipe._trajectory_safe(view, wps, slot_id="B1"))
        self.assertIn("ROUTE_REJECTED", events)


class OverlayDoesNotTouchMissionState(unittest.TestCase):
    """요청문 18절: overlay 갱신은 control state 를 바꾸지 않는다."""

    def test_emit_route_executing_only_changes_the_overlay(self):
        pipe = _pipeline()
        pose = (133.0, 209.0, 95.0)
        view = _view(pose, node="entrance")
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        route = pipe._build_entry_staging_route(1, view, "B1", 1)
        self.assertTrue(route)
        before = (dict(pipe._allocation_state), dict(pipe._parking_stage),
                  dict(pipe._auto_host_slot))
        pipe._emit_route(route, car_id=1, recovery=True, executing=True)
        after = (dict(pipe._allocation_state), dict(pipe._parking_stage),
                 dict(pipe._auto_host_slot))
        self.assertEqual(before, after)
        self.assertTrue(pipe._auto_host_route.get(1))


if __name__ == "__main__":
    unittest.main()
