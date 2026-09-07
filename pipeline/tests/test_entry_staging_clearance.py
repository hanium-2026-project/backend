"""ENTRY_STAGING 경로의 최소 boundary 여유 계약.

실차 run_20260826_231000 에서 차가 맵 밖 36.9mm 까지 나갔다. 원인은 제어가
아니라 **계획**이었다: 입구 staging planner 가 여유를 전혀 보지 않고 최단
경로만 골라, 좌벽에서 9.8mm 떨어진 3-point turn 을 실행했다. 실측 추종오차가
약 159mm 였으니 그 경로는 애초에 실행 가능한 해가 아니었다.

여기서 고정하는 계약은 세 가지다.

1. staging 경로는 최소 여유를 **hard constraint** 로 지킨다. 단 그 요구치는
   출발 자세의 여유로 floor 된다 — 이미 벽에 붙어 있는 차가 자기 출발
   자세 때문에 못 빠져나오면 안 된다.
2. 통로 진입과 통로 정렬은 한 목표다. 진입만 먼저 풀면 통로 밴드 안에서
   대각선으로 서 있는 자세가 나오고, 그 2단계는 해가 없다.
3. 해가 없으면 조용히 넘어가지 않고 빈 경로(= 상위에서 bounded fault)를
   낸다. 여유를 깎아서 억지로 만들지 않는다.

좌표는 231000 의 실제 관측 자세다. 특정 run 을 통과시키는 예외문은 없고,
검사하는 것은 전부 기하 불변식이다.
"""

from __future__ import annotations

import math
import unittest

from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (AISLE_Y, ON_AISLE_TOLERANCE_MM,
                               STAGING_RADIUS_CANDIDATES,
                               REAR_RADIUS_CANDIDATES,
                               MIN_TURN_RADIUS_MM,
                               _path_clearance,
                               build_setup_recovery_waypoints,
                               default_slot_specs)
from pipeline.config import PipelineConfig

# run_20260826_231000 의 최초 확정 자세 (입구, 통로 직각)
START_231000 = (127.3, 166.6, 89.7)
# run_20260831_222643 의 입구 자세 (기존 entrance staging 회귀가 쓰는 자세)
START_222643 = (133.0, 209.0, 95.0)
# 중앙 baseline: 지금까지 성공하던 통로 위 시작
CENTER_START = (600.0, 600.0, 0.0)

HEADING_TOL = PipelineConfig().entry_staging_heading_tolerance_deg
MAX_LEN = PipelineConfig().entry_staging_alignment_max_mm
MIN_CLEAR = PipelineConfig().entry_staging_min_clearance_mm
MIN_EXEC = 110.0  # RECOVERY position_tolerance_cm(8) * 10 + camera lead


def _heading_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _stage(pose, slot_id="B1", clearance=MIN_CLEAR,
           radii=STAGING_RADIUS_CANDIDATES, obstacles=(), max_len=MAX_LEN,
           fallback=STAGING_RADIUS_CANDIDATES):
    """runner._build_entry_staging_route 와 동일한 계획 호출."""
    slot = default_slot_specs()[slot_id]
    desired = 0.0 if slot.center_x >= pose[0] else 180.0

    def goal(p):
        if abs(p[1] - AISLE_Y) > ON_AISLE_TOLERANCE_MM:
            return False
        return _heading_delta(p[2], desired) <= HEADING_TOL

    return build_setup_recovery_waypoints(
        slot, route_id=99, from_pose=(pose[0], pose[1]),
        from_heading_deg=pose[2], radii_mm=radii,
        obstacle_poses=obstacles, min_executable_mm=MIN_EXEC,
        goal_test=goal, max_segments=4, max_total_length_mm=max_len,
        min_clearance_mm=clearance, fallback_radii_mm=fallback)


def _clearance(pose, wps) -> float:
    poses = [pose] + [(w.x, w.y, w.target_heading_deg) for w in wps]
    return _path_clearance(poses)[1]


def _reversals(wps) -> int:
    return sum(1 for a, b in zip(wps, wps[1:])
               if a.motion_direction != b.motion_direction)


def _length(pose, wps) -> float:
    poses = [pose] + [(w.x, w.y, w.target_heading_deg) for w in wps]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(poses, poses[1:]))


class ClearanceContract(unittest.TestCase):
    """요구치 자체의 유도와 planner 계약."""

    def test_minimum_clearance_exceeds_runtime_hard_stop_band(self):
        """계획 여유는 runtime hard-stop band 보다 반드시 커야 한다.

        그렇지 않으면 '계획상 안전'한 경로가 측정 잡음만으로 hard stop 을
        밟는다. 이 관계가 깨지면 상수 하나만 고쳐도 정합성이 무너진다.
        """
        cfg = PipelineConfig()
        band = (cfg.boundary_hard_margin_mm
                + cfg.boundary_measurement_uncertainty_mm)
        self.assertGreater(cfg.entry_staging_min_clearance_mm, band)

    def test_measured_turn_radius_is_a_staging_candidate(self):
        """실측 최소 선회반경이 후보에 있어야 입구에서 해가 존재한다."""
        self.assertIn(MIN_TURN_RADIUS_MM, STAGING_RADIUS_CANDIDATES)
        self.assertLess(min(STAGING_RADIUS_CANDIDATES),
                        min(REAR_RADIUS_CANDIDATES))

    def test_wide_rear_radii_alone_cannot_stage_from_every_entrance_pose(self):
        """후면주차용 800~1100 만으로는 입구 자세에 따라 해가 없다.

        실측 자세 (133,209,95도) 에서 800~1100 조합은 4 구간을 줘도
        infeasible 이고, 610mm 를 넣으면 여유 43.1mm 로 풀린다.
        이게 STAGING_RADIUS_CANDIDATES 가 존재하는 이유다.

        fallback 을 꺼서 **1단계 탐색만** 본다. 프로덕션은 1단계가 비면
        610mm 를 넣어 한 번 더 보므로, fallback 을 켠 채로는 "넓은 반경만
        으로는 불가능하다" 를 관측할 수 없다.
        """
        self.assertEqual(
            _stage(START_222643, radii=REAR_RADIUS_CANDIDATES,
                   fallback=None), [])
        self.assertTrue(_stage(START_222643))

    def test_measured_radius_gives_a_shorter_route_with_fewer_reversals(self):
        """해가 둘 다 있는 자세에서도 610mm 쪽이 실행하기 쉽다.

        231000 자세: 800~1100 은 1046mm / 방향전환 3회, 610mm 포함은
        797mm / 2회. 여유는 41.2 vs 41.1mm 로 사실상 같다 — 즉 610mm 는
        여유를 사서 넣는 게 아니라 기동을 줄여서 넣는 것이다.
        """
        wide = _stage(START_231000, radii=REAR_RADIUS_CANDIDATES)
        tight = _stage(START_231000)
        self.assertTrue(wide)
        self.assertTrue(tight)
        self.assertLess(_length(START_231000, tight),
                        _length(START_231000, wide))
        self.assertLess(_reversals(tight), _reversals(wide))

    def test_shared_planner_default_is_unconstrained(self):
        """min_clearance_mm 기본값 0 = 기존 호출부 동작 불변 (opt-in)."""
        slot = default_slot_specs()["B1"]
        free = build_setup_recovery_waypoints(
            slot, route_id=1, from_pose=(CENTER_START[0], CENTER_START[1]),
            from_heading_deg=CENTER_START[2], min_executable_mm=MIN_EXEC)
        explicit = build_setup_recovery_waypoints(
            slot, route_id=1, from_pose=(CENTER_START[0], CENTER_START[1]),
            from_heading_deg=CENTER_START[2], min_executable_mm=MIN_EXEC,
            min_clearance_mm=0.0)
        self.assertEqual([(w.x, w.y) for w in free],
                         [(w.x, w.y) for w in explicit])


class Entrance231000(unittest.TestCase):
    """실제 실패 자세에서의 회귀."""

    def test_exact_start_produces_a_route(self):
        wps = _stage(START_231000)
        self.assertTrue(wps)

    def test_exact_start_route_clears_the_wall(self):
        """231000 이 실행한 경로의 여유는 9.8mm 였다. 이제 요구치 이상이다."""
        wps = _stage(START_231000)
        self.assertGreaterEqual(_clearance(START_231000, wps), MIN_CLEAR)

    def test_exact_start_route_is_trajectory_safe(self):
        wps = _stage(START_231000)
        result = validate_trajectory(wps, start_pose=START_231000,
                                     target_slot="B1")
        self.assertTrue(result.safe, result.reason)

    def test_exact_start_route_reaches_aisle_and_alignment(self):
        """정렬을 다음 route 로 미루지 않는다 — 한 경로가 둘 다 만족한다."""
        end = _stage(START_231000)[-1]
        self.assertLessEqual(abs(end.y - AISLE_Y), ON_AISLE_TOLERANCE_MM)
        self.assertLessEqual(_heading_delta(end.target_heading_deg, 0.0),
                             HEADING_TOL)

    def test_exact_start_route_stays_within_length_budget(self):
        wps = _stage(START_231000)
        self.assertLessEqual(_length(START_231000, wps), MAX_LEN)

    def test_old_two_stage_goal_reproduces_the_231000_failure(self):
        """대조군: 예전 설정은 실제 실패를 그대로 재현한다.

        1단계가 heading 을 요구하지 않으므로 (195.6, 542.5, 69.7도) 에서
        끝난다 — 실차 로그의 2단계 시작 자세 (214, 550, 71도) 와 pose 잡음
        범위에서 일치한다. 그 자세에서 2단계를 풀면 좌벽 9.8mm 짜리
        3-point turn 이 나온다. 이게 차를 맵 밖으로 보낸 경로다.

        통과 이유가 우연이 아니라는 증거이자, 회귀하면 즉시 잡히는 지점이다.
        """
        slot = default_slot_specs()["B1"]

        def stage1_goal(p):  # 예전 goal: 통로 밖이면 y 밴드만 본다
            return abs(p[1] - AISLE_Y) <= ON_AISLE_TOLERANCE_MM

        stage1 = build_setup_recovery_waypoints(
            slot, route_id=9, from_pose=START_231000[:2],
            from_heading_deg=START_231000[2],
            radii_mm=REAR_RADIUS_CANDIDATES, min_executable_mm=MIN_EXEC,
            goal_test=stage1_goal, max_segments=3, max_total_length_mm=700.0)
        self.assertTrue(stage1)
        handoff = (stage1[-1].x, stage1[-1].y, stage1[-1].target_heading_deg)
        # 통로에는 닿았지만 통로 방향과 크게 어긋난 자세를 넘긴다.
        self.assertGreater(_heading_delta(handoff[2], 0.0), 60.0)

        def stage2_goal(p):
            return (abs(p[1] - AISLE_Y) <= ON_AISLE_TOLERANCE_MM
                    and _heading_delta(p[2], 0.0) <= HEADING_TOL)

        stage2 = build_setup_recovery_waypoints(
            slot, route_id=9, from_pose=handoff[:2],
            from_heading_deg=handoff[2], radii_mm=REAR_RADIUS_CANDIDATES,
            min_executable_mm=MIN_EXEC, goal_test=stage2_goal,
            max_segments=3, max_total_length_mm=MAX_LEN)
        self.assertTrue(stage2)
        self.assertLess(_clearance(handoff, stage2), 15.0)

    def test_single_stage_goal_never_hands_off_a_misaligned_pose(self):
        """새 goal 은 애초에 그 2단계를 만들지 않는다."""
        end = _stage(START_231000)[-1]
        self.assertLessEqual(_heading_delta(end.target_heading_deg, 0.0),
                             HEADING_TOL)


class EntrancePerturbation(unittest.TestCase):
    """손으로 놓는 시작 자세의 오차에 대한 거동."""

    def _assert_safe_route(self, pose, slot_id="B1"):
        wps = _stage(pose, slot_id=slot_id)
        self.assertTrue(wps, f"{pose} {slot_id}: no route")
        start_clear = _path_clearance([pose])[1]
        # 계약: 요구치를 지키되, 출발 여유보다 더 요구하지는 않는다.
        self.assertGreaterEqual(_clearance(pose, wps) + 1e-6,
                                min(MIN_CLEAR, start_clear))
        self.assertTrue(validate_trajectory(
            wps, start_pose=pose, target_slot=slot_id).safe)
        return wps

    def test_x_plus_20mm(self):
        self._assert_safe_route((147.3, 166.6, 89.7))

    def test_y_plus_20mm(self):
        self._assert_safe_route((127.3, 186.6, 89.7))

    def test_y_minus_20mm_holds_its_start_clearance(self):
        """출발 여유가 21.2mm 뿐이어도 더 나빠지지 않는 탈출로가 있다."""
        pose = (127.3, 146.6, 89.7)
        wps = self._assert_safe_route(pose)
        self.assertGreaterEqual(_clearance(pose, wps) + 1e-6,
                                _path_clearance([pose])[1])

    def test_x_minus_20mm_refuses_rather_than_scraping_the_wall(self):
        """좌벽 쪽으로 20mm 밀린 시작 자세는 해가 없다 — 그게 정답이다.

        여기서 나오는 모든 경로는 출발(31.6mm)보다 벽에 더 붙는다. 여유를
        깎아 통과시키지 않고 빈 경로를 내서 상위 bounded fault 로 보낸다.
        """
        pose = (107.3, 166.6, 89.7)
        self.assertEqual(_stage(pose), [])
        # 제약을 풀면 해는 있지만 출발보다 나빠진다 — 거절 이유의 확인.
        loose = _stage(pose, clearance=0.0)
        self.assertTrue(loose)
        self.assertLess(_clearance(pose, loose), _path_clearance([pose])[1])

    def test_heading_minus_5deg(self):
        self._assert_safe_route((127.3, 166.6, 84.7))

    def test_heading_plus_5deg(self):
        self._assert_safe_route((127.3, 166.6, 94.7))

    def test_heading_perturbation_is_not_rejected_by_an_over_tight_bar(self):
        """±5도 자세의 달성 여유는 35mm 대다.

        요구치를 40mm 로 올리면 이 자세들이 불필요하게 거절된다 —
        35mm 선택의 근거를 코드로 고정한다.
        """
        for pose in ((127.3, 166.6, 84.7), (127.3, 166.6, 94.7)):
            with self.subTest(pose=pose):
                achieved = _clearance(pose, _stage(pose))
                self.assertGreaterEqual(achieved, MIN_CLEAR)
                self.assertLess(achieved, 40.0)


class OtherSlotsAndObstacles(unittest.TestCase):

    def test_every_slot_stages_from_the_entrance(self):
        """staging 은 통로 정렬까지다 — 슬롯이 달라도 해가 있어야 한다."""
        for slot_id in default_slot_specs():
            with self.subTest(slot=slot_id):
                wps = _stage(START_231000, slot_id=slot_id)
                self.assertTrue(wps)
                self.assertGreaterEqual(_clearance(START_231000, wps),
                                        MIN_CLEAR)

    def test_blocked_aisle_fails_safely_instead_of_planning_through(self):
        """통로 한가운데 주차 차량이 있으면 빈 경로 — 뚫고 가지 않는다."""
        self.assertEqual(
            _stage(START_231000, obstacles=((500.0, 600.0, 0.0),)), [])

    def test_length_budget_exhaustion_fails_safely(self):
        """예산이 모자라면 여유를 깎는 대신 계획을 포기한다."""
        self.assertEqual(_stage(START_231000, max_len=300.0), [])


class CenterStartBaseline(unittest.TestCase):
    """지금까지 성공하던 중앙 시작 경로가 영향을 받지 않아야 한다."""

    def test_center_start_is_already_staged(self):
        """통로 위 통로 방향이면 staging 자체가 필요 없다 (goal 즉시 만족)."""
        self.assertLessEqual(abs(CENTER_START[1] - AISLE_Y),
                             ON_AISLE_TOLERANCE_MM)
        self.assertLessEqual(_heading_delta(CENTER_START[2], 0.0), HEADING_TOL)

    def test_center_setup_recovery_route_is_not_changed(self):
        """중앙 복구 planner 는 이번 사이클로 바뀌지 않는다.

        여유 제약은 opt-in 이고 ENTRY_STAGING 만 그것을 켠다. 중앙 복구는
        기본값(0.0) 경로를 그대로 쓴다.
        """
        slot = default_slot_specs()["B1"]
        wps = build_setup_recovery_waypoints(
            slot, route_id=1, from_pose=(500.0, 700.0), from_heading_deg=30.0,
            min_executable_mm=MIN_EXEC)
        self.assertTrue(wps)
        self.assertTrue(validate_trajectory(
            wps, start_pose=(500.0, 700.0, 30.0), target_slot="B1").safe)

    def test_global_constraint_would_change_center_recovery(self):
        """전역 제약이 왜 위험한지 고정한다.

        (500,700,30도) 에서 기본 복구 경로의 최소 여유는 26.2mm 다. 여기에
        35mm 를 전역으로 걸면 planner 가 **다른 경로**를 고른다. 즉 공유
        planner 에 전역 제약을 넣는 것은 중앙 시작 회귀를 뜻한다 —
        이번 사이클이 opt-in 을 택한 이유이자, 별도로 판단해야 할 사안이다.
        """
        slot = default_slot_specs()["B1"]
        default = build_setup_recovery_waypoints(
            slot, route_id=1, from_pose=(500.0, 700.0), from_heading_deg=30.0,
            min_executable_mm=MIN_EXEC)
        constrained = build_setup_recovery_waypoints(
            slot, route_id=1, from_pose=(500.0, 700.0), from_heading_deg=30.0,
            min_executable_mm=MIN_EXEC, min_clearance_mm=MIN_CLEAR)
        self.assertNotEqual([(w.x, w.y) for w in default],
                            [(w.x, w.y) for w in constrained])


if __name__ == "__main__":
    unittest.main()
