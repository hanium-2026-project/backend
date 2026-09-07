"""staging 예산 소유권 + 재배치 목표 계약 — 실제 lifecycle 회귀.

run_20260905_024023 / _024123 (동일 소프트웨어, 하드웨어 우전진 수리 후):

    t= 7  ENTRY_STAGING_LOADED  attempt 1
    t= 9  FAULT POSE_STALE                 <- 카메라 프레임 공백
    t=12  HEADING_RECOVERED -> SLOT_REPOSITION  attempt 2
    t=15  FAULT POSE_STALE                 <- 카메라 프레임 공백
    t=17  HEADING_RECOVERED -> SLOT_REPOSITION  attempt 3
    t=21  ENTRY_STAGING_EXHAUSTED          <- 미션 종료 (21초)

known-good 022217 / 183055 / 183503 은 staging 적재 1회 / 재배치 0회였다.
즉 예산을 태운 것은 실패한 기동이 아니라 **관측 중단** 두 번이었다.

두 번째 사실: 그 재배치들이 만든 목표 자세는 heading 73.2도 / 80.2도였다.
재배치 목표는 plan_handoff 만 물었고, staging 종료 게이트는 밴드 AND 정렬을
물었다. 두 술어가 다르니 재배치가 성공해도 staging 은 끝나지 않는다 —
024123 에서는 heading 이 73.2 -> 80.2 도로 오히려 나빠졌다.

이 파일은 handler 단위가 아니라 `_feed_auto_host` 프레임 루프를 그대로
돌리는 lifecycle 하니스(test_entrance_lifecycle_integration._Lifecycle)를
재사용한다. TARGET_SLOT 은 B1 고정이 아니라 임의 슬롯으로 parameterize 한다.
"""

from __future__ import annotations

import math
import time
import unittest

from control.auto_host_runner import MissionStatus
from parking.waypoints import (AISLE_Y, ON_AISLE_TOLERANCE_MM,
                               default_slot_specs, plan_handoff)
from pipeline.config import PipelineConfig
from pipeline.tests.test_entrance_lifecycle_integration import (DRIVE_TRACK,
                                                                SETTLED,
                                                                _Lifecycle)

# 임의 TARGET_SLOT X — B1 은 예시일 뿐이다 (두 행 / 서로 다른 x).
SLOTS = ("B1", "A2", "B3", "A4")


def _staging_loads(pipe) -> list[dict]:
    return [f for n, f in pipe.events if n == "ENTRY_STAGING_LOADED"]


def _faults(pipe) -> list[str]:
    """`_entry_staging_fault` 는 FAULT 이벤트로 남는다."""
    return [f.get("reason") for n, f in pipe.events if n == "FAULT"]


# ══ TEST A. 카메라 중단 복구는 예산을 쓰지 않는다 ═══════════════════════════

class CameraInterruptionDoesNotConsumeTheBudget(_Lifecycle):

    def _stall_and_recover(self) -> None:
        """주행 -> POSE_STALE -> 정지 -> fresh pose -> 복구 -> staging 재개."""
        self.drive_then_stall()
        self.settle()

    def test_one_interruption_keeps_the_slot_and_the_budget(self) -> None:
        for slot in SLOTS:
            with self.subTest(slot=slot):
                self.build(slot)
                before = self.pipe._entry_staging_attempts[1]
                self._stall_and_recover()
                names = [n for n, _ in self.pipe.events]
                self.assertIn("HEADING_RECOVERED", names)
                self.assertEqual(self.pipe._auto_host_slot[1], slot,
                                 "TARGET_SLOT 은 X 그대로여야 한다")
                self.assertEqual(self.pipe._entry_staging_attempts[1], before,
                                 "관측 중단 복구가 예산을 태웠다")
                self.assertEqual(_faults(self.pipe), [])
                self.assertTrue(self.runner.loaded, "미션이 계속되지 않았다")

    def test_two_interruptions_do_not_add_up_to_exhaustion(self) -> None:
        """예산 소진이 카메라 중단 횟수에 비례하면 안 된다 (024023/024123)."""
        for slot in SLOTS:
            with self.subTest(slot=slot):
                self.build(slot)
                before = self.pipe._entry_staging_attempts[1]
                self._stall_and_recover()
                self._stall_and_recover()
                names = [n for n, _ in self.pipe.events]
                self.assertEqual(names.count("HEADING_RECOVERED"), 2,
                                 "두 번 다 복구돼야 한다")
                self.assertEqual(self.pipe._entry_staging_attempts[1], before,
                                 "중단 2회가 예산 2회를 태웠다")
                self.assertNotIn("ENTRY_STAGING_EXHAUSTED",
                                 _faults(self.pipe))
                self.assertEqual(self.pipe._auto_host_slot[1], slot)
                self.assertEqual(self.pipe.allocator.vehicles[2].assigned_slot,
                                 slot, "예약은 유지된다")

    def test_the_resumed_reload_is_labelled_in_the_log(self) -> None:
        """로그만 보고 '중단 복구 재적재'와 '새 시도'를 구분할 수 있어야 한다."""
        self.build("B1")
        self._stall_and_recover()
        loads = _staging_loads(self.pipe)
        self.assertTrue(loads, "재개 시 route 가 실리지 않았다")
        self.assertTrue(all(f.get("resumed") for f in loads), loads)
        budget = self.pipe._entry_staging_attempts[1]
        self.assertTrue(all(f.get("attempt") == budget for f in loads),
                        f"attempt 로그가 실제 예산과 다르다: {loads}")


# ══ TEST B. 진짜 실패한 기동은 여전히 예산을 쓴다 ═══════════════════════════

class GenuineStagingFailuresStillExhaustTheBudget(_Lifecycle):

    def arm(self, slot: str) -> None:
        self.build(slot)
        self.pipe._entry_staging_attempts[1] = 0
        self.runner.replan_reason = "PATH_DEVIATION"
        self.runner.prepare_route_switch = lambda: None
        self.runner.failed_target = None
        self.pose = SETTLED

    def settle_at(self, pose, ticks: int = 20) -> None:
        """`_Lifecycle.settle` 과 같되 **주어진** 자세로 정지한다."""
        for _ in range(ticks):
            self.tick(pose)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with self.pipe._lock:
                    busy = 1 in self.pipe._entry_staging_planning
                if not busy:
                    break
                time.sleep(0.005)

    def fail_one_maneuver(self) -> None:
        """production 경로: 러너가 REPLAN_REQUIRED 를 올린다 = 기동 실패.

        실패한 기동은 실제로 **조금 움직인 뒤** 이탈한다. 자세를 그대로 두면
        같은 route 가 다시 나와 기존 REPEATED_IDENTICAL_STAGING 계약에 걸려
        예산 이전 단계에서 멈춘다 — 그래서 직전 route 의 초반 waypoint 를
        이탈 지점으로 삼는다 (경로를 거의 못 따라간 실패).
        """
        self.tick(self.pose)
        self.pipe._on_auto_host_status(1, MissionStatus.RUNNING,
                                       MissionStatus.REPLAN_REQUIRED)
        self.settle_at(self.pose)
        if self.runner.loaded:
            wps = self.runner.loaded[-1]
            wp = wps[min(1, len(wps) - 1)]
            self.pose = (wp.x, wp.y, wp.target_heading_deg)

    def test_each_failed_maneuver_costs_exactly_one_attempt(self) -> None:
        for slot in SLOTS:
            with self.subTest(slot=slot):
                self.arm(slot)
                for expected in (1, 2, 3):
                    self.fail_one_maneuver()
                    self.assertEqual(self.pipe._entry_staging_attempts[1],
                                     expected,
                                     f"{expected}번째 실패가 예산에 반영 안 됨")
                    self.assertNotIn(1, self.pipe._heading_fault_hold,
                                     "관측 중단이 아닌데 hold 가 찍혔다")
                self.assertTrue(
                    all(f.get("resumed") is False
                        for f in _staging_loads(self.pipe)),
                    "진짜 실패가 '중단 복구' 로 분류됐다")

    def test_three_genuine_failures_reach_exhausted(self) -> None:
        """Phase 1 수정이 예산을 사실상 비활성화하지 않았다."""
        for slot in SLOTS:
            with self.subTest(slot=slot):
                self.arm(slot)
                for _ in range(4):
                    self.fail_one_maneuver()
                self.assertIn("ENTRY_STAGING_EXHAUSTED", _faults(self.pipe))
                self.assertEqual(
                    self.pipe._entry_staging_attempts[1],
                    int(PipelineConfig().max_entry_staging_attempts))
                self.assertEqual(self.pipe._auto_host_slot[1], slot,
                                 "소진돼도 예약 슬롯은 유지된다")
                self.assertEqual(self.pipe.allocator.vehicles[2].assigned_slot,
                                 slot)

    def test_the_budget_value_itself_is_untouched(self) -> None:
        self.assertEqual(PipelineConfig().max_entry_staging_attempts, 3)


# ══ TEST C/D. 재배치 목표 = staging 종료 계약 (임의 슬롯) ═══════════════════

class RepositionGoalMatchesTheStagingContract(_Lifecycle):

    def _goal(self, slot: str):
        self.build(slot)
        return self.pipe._entry_staging_goal(self.view, slot)

    def test_a_sideways_pose_inside_the_band_is_not_a_goal(self) -> None:
        """024023/024123 이 실제로 만든 목표 자세대 (heading 70~80도)."""
        for slot in SLOTS:
            goal = self._goal(slot)
            spec = default_slot_specs()[slot]
            for heading in (70.0, 73.2, 75.0, 80.2):
                with self.subTest(slot=slot, heading=heading):
                    pose = (spec.center_x - 200.0, AISLE_Y, heading)
                    self.assertLessEqual(abs(pose[1] - AISLE_Y),
                                         ON_AISLE_TOLERANCE_MM,
                                         "전제: 자세는 통로 밴드 안이다")
                    self.assertFalse(
                        goal(pose),
                        "밴드 안이라도 옆으로 선 자세는 목표가 아니다")

    def test_the_same_place_within_tolerance_can_be_a_goal(self) -> None:
        tol = float(PipelineConfig().entry_staging_heading_tolerance_deg)
        self.assertLessEqual(tol, 15.0, "heading tolerance 는 손대지 않는다")
        for slot in SLOTS:
            goal = self._goal(slot)
            spec = default_slot_specs()[slot]
            desired = self.pipe._entry_staging_heading(self.view, slot)
            self.assertIsNotNone(desired)
            for dx in (-300.0, -200.0, -100.0, 100.0, 200.0, 300.0):
                pose = (spec.center_x + dx, AISLE_Y, desired)
                if plan_handoff(spec, from_pose=(pose[0], pose[1]),
                                from_heading_deg=pose[2]).feasible:
                    with self.subTest(slot=slot, dx=dx):
                        self.assertTrue(goal(pose))
                    break
            else:
                self.fail(f"{slot}: 정렬된 통로 자세에서 인계 경로가 없다")

    def test_staging_and_reposition_ask_the_very_same_question(self) -> None:
        """두 경로가 같은 술어를 쓴다 — 재배치 성공 = staging 종료 가능."""
        for slot in SLOTS:
            with self.subTest(slot=slot):
                self.build(slot)
                goal = self.pipe._entry_staging_goal(self.view, slot)
                spec = default_slot_specs()[slot]
                tol = float(self.pipe.config
                            .entry_staging_heading_tolerance_deg)
                desired = self.pipe._entry_staging_heading(self.view, slot)
                for pose in [(spec.center_x - 250.0, AISLE_Y, desired),
                             (spec.center_x - 250.0, AISLE_Y, desired + 40.0),
                             (spec.center_x - 250.0, AISLE_Y - 300.0, desired),
                             (spec.center_x, AISLE_Y, desired)]:
                    banded = abs(pose[1] - AISLE_Y) <= ON_AISLE_TOLERANCE_MM
                    aligned = (self.pipe._heading_delta(pose[2], desired)
                               <= tol)
                    reachable = plan_handoff(
                        spec, from_pose=(pose[0], pose[1]),
                        from_heading_deg=pose[2]).feasible
                    self.assertEqual(goal(pose),
                                     banded and aligned and reachable,
                                     f"{slot} {pose}")

    def test_the_parking_stage_goal_stays_plan_handoff(self) -> None:
        """PHASE 3: 주차 단계 재배치 계약은 건드리지 않았다."""
        self.build("B1")
        for slot in SLOTS:
            goal = self.pipe._parking_reposition_goal(slot)
            spec = default_slot_specs()[slot]
            for pose in [(361.3, 520.5, 15.1), (600.0, 600.0, 0.0),
                         (spec.center_x - 250.0, AISLE_Y, 0.0)]:
                with self.subTest(slot=slot, pose=pose):
                    self.assertEqual(
                        goal(pose),
                        plan_handoff(spec, from_pose=(pose[0], pose[1]),
                                     from_heading_deg=pose[2]).feasible)


# ══ 전제 검증 ═══════════════════════════════════════════════════════════════

class HarnessPreconditions(_Lifecycle):

    def test_the_first_recovery_tick_is_still_moving(self) -> None:
        moved = [math.hypot(b[0] - a[0], b[1] - a[1])
                 for a, b in zip(DRIVE_TRACK[2:5], DRIVE_TRACK[3:5])]
        self.assertTrue(
            all(m > PipelineConfig().stationary_tolerance_mm for m in moved),
            f"첫 복구 tick 이 정지 상태면 이 파일은 아무것도 증명하지 않는다:"
            f" {moved}")

    def test_the_settled_pose_is_off_the_aisle(self) -> None:
        """staging 이 실제로 필요한 자세여야 이 회귀가 의미를 가진다."""
        self.assertGreater(abs(SETTLED[1] - AISLE_Y), ON_AISLE_TOLERANCE_MM)


if __name__ == "__main__":
    unittest.main()
