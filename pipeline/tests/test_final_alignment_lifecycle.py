"""최종 보정은 끝까지 최종 보정이어야 한다 (run_20260903_191010 / _191220).

두 실차 run 모두 차가 슬롯에 들어갔다 나왔다를 반복하다 수동 중단됐다.

    191010  t=31.6  FINAL_POSE_EVAL ALIGN (depth 83.8 lat 35.9 hdg -0.2)
            t=31.7  FINAL_ALIGNMENT route 5 적재 (FORWARD x4, 종점 depth -211)
            t=34.9  HEADING_OUT_OF_TOLERANCE -> PARKING_SETUP_WAIT
                    (자세 (438.5,696.0,272.1), depth -354)
            t=36.8  generic setup recovery, stage=SETUP
            t=43.2  ROUTE_LOADED count=5  = 통로 재접근 + 전체 후면주차 재시작
            t=52.5  BOUNDARY_HARD 30.6 ...
    191220  t=30.2  FINAL_POSE_EVAL ALIGN (depth 51.3 lat 46.4 hdg 2.7)
            t=31.8  NO_SAFE_FINAL_ALIGNMENT -> REPEATED_IDENTICAL_REPLAN
            t=32.3~ setup recovery attempt 2,3,4,5,6,7 ...

구조적 원인은 서로 맞물리지 않는 두 기하 구간이다.

    in_final_region   depth 하한 = -slot.length/2                   = -150mm
    alignment staging depth      = -(length/2 + CAR_LENGTH/2)       = -275mm
    alignment_goal    허용 대역   = [staging-200, staging+100]       = [-475, -175]

    교집합 = 공집합.  가장 얕은 정렬 목표(-175)조차 in_final_region 밖이다.

즉 FINAL_ALIGNMENT 는 **설계상 자기가 최종구간 판정 밖으로 차를 몰고 나간다**.
그 상태에서 REPLAN 이 나면 _route_final_quality_replan 이 기하 게이트에서
False 를 돌려주고, 아래 generic 분기가 stage 를 SETUP_PENDING 으로 바꾼다.
거기서부터는 통로 재접근 + 전체 후면주차다. 그것이 "들어감→나옴" 루프다.

수정: 기하 게이트는 최종 lifecycle 에 **들어올 때(stage PARKING)만** 적용한다.
이미 보정 중인 차(FINAL_ALIGNMENT / FINAL_STRAIGHT_REVERSE)는 최종 평가기가
계속 맡는다. 반복은 max_final_alignment_attempts 가 이미 막는다.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import parking.final_alignment as fa
from parking.waypoints import AISLE_Y, CAR_LENGTH_MM, default_slot_specs
from pipeline.runner import ParkingPipeline, VehicleView

B1 = default_slot_specs()["B1"]

# ── 실차 계측값 ────────────────────────────────────────────────────────────
POSE_191010_REPLAN = (438.5, 696.0, 272.1)     # FINAL_ALIGNMENT 실행 중 replan
POSE_191010_ALIGN_END = (434.0, 838.7, 269.8)  # 그 route 의 계획 종점
POSE_032539_AISLE = (376.2, 626.3, 44.1)       # 통로를 지나가던 차 (오인 사례)


class TheAlignmentGoalRegionIsLocal(unittest.TestCase):
    """정렬 목표 영역이 슬롯 입구권 안에 있어야 lifecycle 이 스스로 닫힌다.

    예전 band (-200,+100) 은 staging(-275) 기준 절대 [-475,-175] 였고,
    in_final_region 의 하한 -slot.length/2 = -150 과 교집합이 **비어 있었다**.
    그래서 정렬이 끝나면 차가 최종구간 밖에 있었고, B1 기준 먼 끝(y=575)은
    통로 중심(600)을 지났다. 지금은 두 경계가 정확히 맞닿는다.
    """

    def test_the_goal_region_reaches_the_final_region_boundary(self) -> None:
        staging_depth = fa.to_slot_local(
            B1, *fa.alignment_staging_pose(B1)).depth_mm
        low, high = fa.ALIGN_GOAL_DEPTH_BAND_MM
        # staging = 차체 뒷면이 슬롯 입구에 있는 깊이
        self.assertAlmostEqual(staging_depth,
                               -(B1.length / 2.0 + CAR_LENGTH_MM / 2.0),
                               places=6)
        # 얕은 끝 = 차체 중심이 슬롯 입구 = in_final_region 하한과 동일
        self.assertAlmostEqual(staging_depth + high, -B1.length / 2.0,
                               places=6)
        # 깊은 끝은 staging 보다 더 나가지 않는다
        self.assertAlmostEqual(low, 0.0, places=6)

    def test_the_goal_region_never_reaches_the_aisle(self) -> None:
        """국소 보정의 목적지는 통로가 아니다 — 193043/193518 의 실패 지점."""
        staging_depth = fa.to_slot_local(
            B1, *fa.alignment_staging_pose(B1)).depth_mm
        low, _high = fa.ALIGN_GOAL_DEPTH_BAND_MM
        farthest_y = B1.center_y + staging_depth + low
        self.assertGreater(farthest_y, AISLE_Y + B1.length / 2.0)

    def test_the_planned_alignment_endpoint_is_outside_the_final_region(self):
        self.assertFalse(fa.in_final_region(B1, *POSE_191010_ALIGN_END[:2]))

    def test_the_recorded_replan_pose_is_outside_the_final_region(self) -> None:
        self.assertFalse(fa.in_final_region(B1, *POSE_191010_REPLAN[:2]))


class _Runner:
    def __init__(self, reason: str) -> None:
        self.replan_reason = reason


class _Dashboard:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def push_event(self, name, **fields) -> None:
        self.events.append((name, fields))


class FinalQualityReplanStaysLocalWhileCorrecting(unittest.TestCase):

    def _pipeline(self, stage: str, pose, reason="HEADING_OUT_OF_TOLERANCE"):
        p = ParkingPipeline.__new__(ParkingPipeline)
        p.config = SimpleNamespace(parking_mode="rear",
                                   max_final_alignment_attempts=3)
        # rear_parking_mode 는 config 에서 파생되는 읽기 전용 property 다.
        p.auto_hosts = {1: _Runner(reason)}
        p._auto_host_slot = {1: "B1"}
        p._parking_stage = {1: stage}
        p._parking_plan_wait = {}
        p.dashboard = _Dashboard()
        p.events = []
        p.on_event_record = None
        view = VehicleView(track_id=7, car_id=1, position_mm=pose[:2],
                           heading_deg=pose[2],
                           heading_source="FRONT_CUSHION")
        view.last_obs_time = 34.9
        return p, view

    # ── 핵심 회귀 ──────────────────────────────────────────────────────────

    def test_a_replan_while_aligning_stays_in_the_final_lifecycle(self) -> None:
        """191010 t=34.9 그대로. 여기서 generic setup 으로 떨어지면 루프다."""
        for stage in ("FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE"):
            with self.subTest(stage=stage):
                p, view = self._pipeline(stage, POSE_191010_REPLAN)
                self.assertTrue(p._route_final_quality_replan(1, view))
                self.assertEqual(p._parking_stage[1], "FINAL_EVAL_PENDING")

    def test_that_pose_then_asks_for_a_straight_reverse_not_a_reapproach(self):
        """local correction 의 다음 단계는 직선 후진이지 통로 재접근이 아니다."""
        verdict = fa.evaluate_final_pose(B1, *POSE_191010_REPLAN)
        self.assertEqual(verdict.action, "STRAIGHT_REVERSE")
        self.assertLessEqual(abs(verdict.local.lateral_mm),
                             fa.STRAIGHT_REVERSE_LATERAL_LIMIT_MM)

    def test_the_planned_alignment_endpoint_also_leads_to_straight_reverse(self):
        verdict = fa.evaluate_final_pose(B1, *POSE_191010_ALIGN_END)
        self.assertEqual(verdict.action, "STRAIGHT_REVERSE")

    # ── 넓히면 안 되는 쪽 ─────────────────────────────────────────────────

    def test_a_passing_car_still_needs_the_geometric_gate(self) -> None:
        """032539: 통로에서 heading 오차 129도인 차를 최종 정렬로 보내면 안 된다.

        그 오인은 stage PARKING 에서 일어났고, 진입 게이트는 그대로 남는다.
        """
        p, view = self._pipeline("PARKING", POSE_032539_AISLE)
        self.assertFalse(fa.in_final_region(B1, *POSE_032539_AISLE[:2]))
        self.assertFalse(p._route_final_quality_replan(1, view))
        self.assertEqual(p._parking_stage[1], "PARKING")

    def test_entry_from_parking_inside_the_slot_still_works(self) -> None:
        p, view = self._pipeline("PARKING", (451.3, 1125.7, 269.9))
        self.assertTrue(fa.in_final_region(B1, 451.3, 1125.7))
        self.assertTrue(p._route_final_quality_replan(1, view))
        self.assertEqual(p._parking_stage[1], "FINAL_EVAL_PENDING")

    def test_other_stages_are_untouched(self) -> None:
        """SETUP/SETUP_PENDING 등은 예전 그대로 이 경로를 타지 않는다."""
        for stage in ("SETUP", "SETUP_PENDING", "PARKING_HANDOFF_PENDING",
                      "ENTRY_STAGING"):
            with self.subTest(stage=stage):
                p, view = self._pipeline(stage, POSE_191010_REPLAN)
                self.assertFalse(p._route_final_quality_replan(1, view))

    def test_unrelated_replan_reasons_are_untouched(self) -> None:
        """boundary/센서 손실 사유는 여전히 최종 정렬로 우회되지 않는다."""
        for reason in ("BOUNDARY_UNCERTAIN_TREND", "REVERSE_HEADING_TIMEOUT",
                       "ARC_CORRIDOR_MISSED"):
            with self.subTest(reason=reason):
                p, view = self._pipeline("FINAL_ALIGNMENT",
                                         POSE_191010_REPLAN, reason=reason)
                self.assertFalse(p._route_final_quality_replan(1, view))

    def test_the_retry_budget_is_still_the_alignment_one(self) -> None:
        """이 경로는 recovery 예산을 쓰지 않고, 반복은 정렬 예산이 막는다."""
        p, view = self._pipeline("FINAL_ALIGNMENT", POSE_191010_REPLAN)
        p._route_final_quality_replan(1, view)
        self.assertFalse(hasattr(p, "_parking_recovery_attempts"))
        self.assertEqual(p.config.max_final_alignment_attempts, 3)


class KnownGoodFinalPosesAreUnaffected(unittest.TestCase):

    def test_002703_and_022217_still_park(self) -> None:
        specs = default_slot_specs()
        self.assertTrue(
            fa.evaluate_final_pose(specs["A2"], 655.7, 158.7, 91.2).parked)
        self.assertTrue(
            fa.evaluate_final_pose(specs["A3"], 876.4, 161.0, 88.3).parked)

    def test_171158_parks_without_any_alignment(self) -> None:
        self.assertTrue(
            fa.evaluate_final_pose(default_slot_specs()["A2"],
                                   675.7, 171.7, 89.8).parked)

    def test_170954_is_still_rejected(self) -> None:
        verdict = fa.evaluate_final_pose(B1, 451.3, 1125.7, 269.9)
        self.assertFalse(verdict.parked)
        self.assertGreater(verdict.depth_overflow_mm, 50.0)


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()


class AlignmentRoutesStayNearTheSlot(unittest.TestCase):
    """실차 5개 run 의 정렬 시작 자세에서 경로가 국소인지 검증한다.

    예전 band 로는 다음 종점이 나왔다 (depth / 통로중심까지):
        193043 -313 / +137    193518 -418 / +32
        191010 -211 / +239    191220 -365 / +85    193223 -210 / +240
    193518 의 -418 은 통로에서 32mm 떨어진 지점이고, 그 다음 시도는 통로
    안에서 3점 선회를 만들었다. 지금은 모든 종점이 슬롯 입구선 안쪽이어야
    한다.
    """

    # 각 run 의 첫 FINAL_ALIGNMENT 시작 자세 (control.jsonl 실측)
    START_POSES = {
        "193043": (451.1, 1110.9, 274.8),
        "191010": (460.9, 1133.8, 269.8),
        "191220": (471.4, 1101.3, 272.1),
        "193223": (451.3, 1135.0, 269.8),
    }
    SLOT_ENTRANCE_Y = B1.center_y - B1.length / 2.0      # 900

    def test_every_alignment_endpoint_straddles_the_slot_entrance(self) -> None:
        for run, pose in self.START_POSES.items():
            with self.subTest(run=run):
                wps = fa.build_final_alignment_waypoints(
                    B1, 1, from_pose=pose[:2], from_heading_deg=pose[2])
                self.assertTrue(wps, f"{run}: 정렬 경로가 없다")
                end = wps[-1]
                local = fa.to_slot_local(B1, end.x, end.y,
                                         end.target_heading_deg)
                # 종점은 "차가 슬롯 입구에 걸쳐 있는" 구간 안이다.
                self.assertGreaterEqual(local.depth_mm, -275.0 - 1e-6)
                self.assertLessEqual(local.depth_mm, -150.0 + 1e-6)

    def test_no_alignment_endpoint_reaches_the_aisle(self) -> None:
        """**종점**은 통로에 닿지 않는다.

        경계는 종점에 대한 것이다. 중간 waypoint 는 여전히 통로 쪽으로
        내려갈 수 있다 — 이번 수정은 목표 영역을 좁힌 것이지 경로 전체를
        가둔 것이 아니다 (191220 은 8 waypoint 중 최저 y=633 을 지난다).
        그건 별도 문제로 남겨둔다.
        """
        for run, pose in self.START_POSES.items():
            with self.subTest(run=run):
                wps = fa.build_final_alignment_waypoints(
                    B1, 1, from_pose=pose[:2], from_heading_deg=pose[2])
                self.assertTrue(wps)
                self.assertGreater(
                    wps[-1].y, AISLE_Y + B1.length / 2.0,
                    f"{run}: 정렬 종점이 통로 밴드까지 내려간다")

    def test_the_endpoint_is_ready_for_a_straight_reverse(self) -> None:
        """정렬 종점 = 직선 후진으로 넘어갈 수 있는 자세여야 한다."""
        for run, pose in self.START_POSES.items():
            with self.subTest(run=run):
                wps = fa.build_final_alignment_waypoints(
                    B1, 1, from_pose=pose[:2], from_heading_deg=pose[2])
                self.assertTrue(wps)
                end = wps[-1]
                verdict = fa.evaluate_final_pose(B1, end.x, end.y,
                                                 end.target_heading_deg)
                self.assertEqual(verdict.action, "STRAIGHT_REVERSE")

    def test_general_setup_recovery_still_aims_at_the_aisle(self) -> None:
        """일반 setup recovery 는 그대로다 \u2014 goal_test 를 주지 않으면 무변경."""
        from parking.waypoints import build_setup_recovery_waypoints
        wps = build_setup_recovery_waypoints(
            B1, 12, from_pose=(505.0, 1000.0), from_heading_deg=270.0)
        self.assertTrue(wps)
        self.assertLess(wps[-1].y, AISLE_Y + B1.length)
