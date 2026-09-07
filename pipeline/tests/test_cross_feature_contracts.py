"""기능 경계에서 깨지던 세 계약 (run 230944 / 231157 / 231338).

최신 실차 3회는 개별 기능의 버그가 아니라 **기능 사이의 계약 불일치**로
실패했다. 세 가지 모두 한쪽은 정상 동작하고 다른 쪽이 그 사실을 모른다.

1. 새 route × 정체 감시
   run_20260904_231338
       t=34.644 RECOVERY_ROUTE_LOADED count=7 attempt=4
       t=34.865 PARKING_STALLED stalled_s=8.31     (0.16초 뒤)
       t=46.833 RECOVERY_ROUTE_LOADED count=6 attempt=5
       t=47.069 PARKING_STALLED stalled_s=8.84     (0.24초 뒤)
   `_stall_since` 가 route 적재에서 초기화되지 않아, route 가 생기기도 전의
   시각을 기준으로 8초 초과 판정이 나고 방금 실은 route 가 즉시 폐기됐다.

2. tracker identity × multi-vehicle safety
   run_20260904_230944, 실제 RC 카 1대
       ego(track 1)  t=65.7  (304.7, 508.3)  heading 371건
       track 13/19/26/27/30/33/36/40/42  (300.7~306.1, 506.9~512.3) heading 0건
   self 제외가 track_id 하나뿐이라 같은 차의 재획득 track 이 "다른 차량"이
   됐고, heading 이 없어 uncertain 으로 분류돼 route/recovery 가 전부 거부됐다
   (t=63.48 에 3건이 5ms 안에). 51초 정지.

3. staging 종료 × 다음 route 실행가능성
   run_20260904_231157 / _231338
   SLOT_REPOSITION 이 staging 예산을 함께 쓰므로 3/3 이 소진되고,
   `aligned or exhausted` 때문에 heading 86.9도 / 88.8도 에서 인계했다.
   그 자세의 인계 경로는 waypoint 1개였고 그 하나가 차 뒤였다
   (along -29.9 / -19.6mm). 1.5초 만에 PATH_DEVIATION.

세 수정 모두 기존 값·기존 함수만 쓴다. 새 state·flag·임계값 없음.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from controller.config import ControllerConfig
from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (MIN_TURN_RADIUS_MM, build_waypoints,
                               default_slot_specs)
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView
from rl.parking_env import SLOT_NAMES

# ── 실차 자세 ───────────────────────────────────────────────────────────────
POSE_230944_EGO = (304.7, 508.3, 339.3)          # bound ego, heading 있음
GHOST_230944 = (300.7, 512.3)                    # 같은 차의 재획득 track
POSE_231157_HANDOFF = (247.5, 639.6, 86.9)       # staging 이 인계한 수직 자세
POSE_231338_HANDOFF = (243.5, 623.4, 88.8)
POSE_230944_HANDOFF = (138.8, 531.4, 356.3)      # 1wp 지만 실제로 완주한 자세
POSE_022217_HANDOFF = (478.0, 515.2, 13.3)       # known-good


def _view(pose, track_id=2, car_id=1, node="entrance", heading=True):
    return VehicleView(
        track_id=track_id, car_id=car_id, node=node,
        position_mm=(pose[0], pose[1]),
        heading_deg=(pose[2] if heading and len(pose) > 2 else None),
        heading_source=("FRONT_CUSHION" if heading and len(pose) > 2 else None),
        last_obs_time=10.0)


# ══ TEST A — 새 route 는 정체 기준선을 물려받지 않는다 ══════════════════════

class RouteLoadResetsTheStallBaseline(unittest.TestCase):

    def _pipeline(self):
        p = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                           parking_mode="rear"))
        p.on_route_load = None
        return p

    def test_the_baseline_is_route_local_state(self) -> None:
        """route 적재는 다른 route-지역 캐시들과 같은 취급을 받아야 한다."""
        p = self._pipeline()
        p._stall_since[1] = 26.5
        p._deviation_streak[1] = 3
        p._emit_route([], car_id=1)
        self.assertNotIn(1, p._stall_since,
                         "새 route 가 이전 정체 시간을 물려받으면 안 된다")
        self.assertNotIn(1, p._deviation_streak)

    def test_231338_immediate_stall_regression(self) -> None:
        """실측 재현: 적재 0.2초 뒤에 8초 정체 판정이 나오면 안 된다."""
        p = self._pipeline()
        cfg = PipelineConfig()
        timeout = float(cfg.parking_stall_timeout_s)
        view = _view((243.5, 623.4, 88.8))
        view.recent.extend([view.position_mm] * 3)
        # 이전 route 에서 정체가 이미 timeout 을 넘게 쌓여 있었다.
        view.last_obs_time = 26.5
        p._stall_since[1] = view.last_obs_time
        view.last_obs_time = 26.5 + timeout + 0.5
        self.assertGreater(view.last_obs_time - p._stall_since[1], timeout,
                           "전제: 옛 기준선이면 이미 timeout 초과다")
        # 새 recovery route 적재
        p._emit_route([], car_id=1, recovery=True)
        # 0.2초 뒤
        view.last_obs_time += 0.2
        started = p._stall_since.get(1)
        self.assertIsNone(started, "적재 시점에 기준선이 지워져야 한다")

    def test_the_timeout_itself_is_unchanged(self) -> None:
        self.assertEqual(PipelineConfig().parking_stall_timeout_s, 8.0)


# ══ TEST B — 같은 차의 중복 track 은 다른 차량이 아니다 ═════════════════════

class EgoDuplicateTrackIsNotAnObstacle(unittest.TestCase):

    def _pipeline(self):
        return ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                              parking_mode="rear"))

    def _snapshot(self, others):
        p = self._pipeline()
        ego = _view(POSE_230944_EGO)
        views = {ego.track_id: ego}
        for v in others:
            views[v.track_id] = v
        p.views = views
        return p, ego, p._planning_obstacle_snapshot(ego)

    def test_230944_ghost_tracks_do_not_block_planning(self) -> None:
        """실측 재현: ego 와 ±5mm 인 heading 없는 track 9개."""
        ghosts = [_view((GHOST_230944[0], GHOST_230944[1]), track_id=tid,
                        car_id=None, heading=False)
                  for tid in (13, 19, 26, 27, 30, 33, 36, 40, 42)]
        for g in ghosts:
            g.last_obs_time = 10.0
        _p, _ego, (poses, uncertain) = self._snapshot(ghosts)
        self.assertEqual(poses, (), "자기 자신을 장애물로 넣으면 안 된다")
        self.assertEqual(uncertain, (),
                         "자기 자신 때문에 모든 경로가 거부되면 안 된다")

    def test_a_route_is_accepted_again(self) -> None:
        ghost = _view((GHOST_230944[0], GHOST_230944[1]), track_id=13,
                      car_id=None, heading=False)
        ghost.last_obs_time = 10.0
        p, ego, _snap = self._snapshot([ghost])
        wps = build_waypoints(default_slot_specs()["B1"], route_id=1,
                              from_pose=POSE_022217_HANDOFF[:2],
                              from_heading_deg=POSE_022217_HANDOFF[2],
                              min_radius_mm=MIN_TURN_RADIUS_MM)
        result, reason = p._trajectory_verdict(ego, wps, "B1")
        self.assertNotEqual(reason, "OTHER_VEHICLE_POSE_UNCERTAIN")

    def test_a_genuine_other_vehicle_is_still_an_obstacle(self) -> None:
        """실제 다른 차량은 종전대로 장애물이다 (안전 기능 유지)."""
        other = _view((700.0, 600.0, 0.0), track_id=9, car_id=2)
        other.last_obs_time = 10.0
        _p, _ego, (poses, uncertain) = self._snapshot([other])
        self.assertEqual(poses, ((700.0, 600.0, 0.0),))
        self.assertEqual(uncertain, ())

    def test_a_genuine_other_vehicle_without_heading_is_still_uncertain(self):
        """멀리 있는 heading 없는 track 은 종전대로 불확실 처리한다."""
        other = _view((700.0, 600.0), track_id=9, car_id=None, heading=False)
        other.last_obs_time = 10.0
        _p, _ego, (poses, uncertain) = self._snapshot([other])
        self.assertEqual(poses, ())
        self.assertEqual(len(uncertain), 1)

    def test_a_bound_second_car_is_never_dismissed_by_distance(self) -> None:
        """car_id 가 있는 track 은 가까워도 예외를 타지 않는다."""
        near = _view((POSE_230944_EGO[0] + 10.0, POSE_230944_EGO[1]),
                     track_id=9, car_id=2, heading=False)
        near.last_obs_time = 10.0
        _p, _ego, (poses, uncertain) = self._snapshot([near])
        self.assertEqual(poses, ())
        self.assertEqual(uncertain, (2,))

    def test_the_distance_is_an_existing_notion_not_a_new_constant(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.track_rebind_max_distance_mm, 150.0)


# ══ TEST C — 예산 소진이 정렬 게이트를 우회하지 않는다 ══════════════════════

class ExhaustedBudgetDoesNotBypassAlignment(unittest.TestCase):

    def _staged(self, pose, slot, attempts):
        p = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                           parking_mode="rear"))

        class _Runner:
            def __init__(self):
                self.loaded = []
                self.stopped = False

            def load_route(self, route):
                self.loaded.append(list(route))

            def stop(self):
                self.stopped = True
        runner = _Runner()
        p.auto_hosts = {1: runner}
        p.track_of_car = {1: 2}
        p._auto_host_slot[1] = slot
        p._parking_stage[1] = "ENTRY_STAGING_PENDING"
        p._entry_staging_attempts[1] = attempts
        view = _view(pose)
        p.views = {2: view}
        p.allocator.update(2, view.position_mm)
        p.allocator.reassign(2, slot)
        for _ in range(8):
            view.recent.append(view.position_mm)
        p.events = []
        p.on_event_record = lambda n, **f: p.events.append((n, f))
        for i in range(6):
            view.last_obs_time = 100.0 + (i + 1) * 0.25
            p._maybe_resume_entry_staging(view)
            if p._parking_stage.get(1) != "ENTRY_STAGING_PENDING":
                break
        return p, runner

    def test_a_perpendicular_pose_never_completes_even_when_exhausted(self):
        budget = PipelineConfig().max_entry_staging_attempts
        for pose in (POSE_231157_HANDOFF, POSE_231338_HANDOFF):
            for slot in ("B1", "A2", "B3"):
                with self.subTest(pose=pose, slot=slot):
                    p, _runner = self._staged(pose, slot, budget)
                    self.assertNotEqual(p._parking_stage.get(1), "GLOBAL",
                                        "수직 자세로 인계하면 안 된다")
                    names = [n for n, _ in p.events]
                    self.assertNotIn("ENTRY_STAGING_COMPLETE", names)

    def test_the_reserved_slot_survives_exhaustion(self) -> None:
        budget = PipelineConfig().max_entry_staging_attempts
        for slot in ("B1", "A2", "B3"):
            with self.subTest(slot=slot):
                p, _runner = self._staged(POSE_231157_HANDOFF, slot, budget)
                self.assertEqual(p._auto_host_slot[1], slot)
                self.assertEqual(p.allocator.vehicles[2].assigned_slot, slot)
                self.assertGreaterEqual(
                    p.allocator.slot_statuses[SLOT_NAMES.index(slot)], 0.5)
                picked = {f.get("slot") for n, f in p.events if f.get("slot")}
                self.assertFalse(picked - {slot}, f"다른 칸 등장: {picked}")

    def test_the_budget_value_is_unchanged(self) -> None:
        self.assertEqual(PipelineConfig().max_entry_staging_attempts, 3)
        self.assertEqual(PipelineConfig().entry_staging_heading_tolerance_deg,
                         15.0)


# ══ TEST D — 뒤에 있는 첫 waypoint 는 싣지 않는다 ═══════════════════════════

class FirstWaypointMustNotBeBehind(unittest.TestCase):

    STOP_MM = 10.0 * ControllerConfig().stop_distance_cm

    def _route(self, pose, slot):
        return build_waypoints(default_slot_specs()[slot], route_id=1,
                               from_pose=pose[:2], from_heading_deg=pose[2],
                               min_radius_mm=MIN_TURN_RADIUS_MM, strict=True)

    def _check(self, pose, slot):
        return validate_trajectory(
            self._route(pose, slot), start_pose=pose, target_slot=slot,
            min_turn_radius_mm=MIN_TURN_RADIUS_MM,
            stop_distance_mm=self.STOP_MM, require_reachable_first_wp=True)

    def _along(self, pose, slot):
        first = self._route(pose, slot)[0]
        angle = math.radians(pose[2])
        return ((first.x - pose[0]) * math.cos(angle)
                + (first.y - pose[1]) * math.sin(angle))

    def test_231157_and_231338_routes_are_rejected(self) -> None:
        for lab, pose in (("231157", POSE_231157_HANDOFF),
                          ("231338", POSE_231338_HANDOFF)):
            with self.subTest(run=lab):
                self.assertEqual(len(self._route(pose, "B1")), 1)
                self.assertLess(self._along(pose, "B1"), 0.0)
                r = self._check(pose, "B1")
                self.assertFalse(r.safe)
                self.assertEqual(r.reason, "FIRST_WP_BEHIND")

    def test_a_reachable_single_waypoint_route_is_allowed(self) -> None:
        """'waypoint 1개' 자체를 금지하지 않는다.

        run_20260904_230944 의 인계 경로도 1개였지만 along +281mm 로 앞에
        있었고, 실차에서 실제로 HANDOFF_CAPTURED 까지 갔다.
        """
        self.assertEqual(len(self._route(POSE_230944_HANDOFF, "B1")), 1)
        self.assertGreater(self._along(POSE_230944_HANDOFF, "B1"), 0.0)
        self.assertTrue(self._check(POSE_230944_HANDOFF, "B1").safe)

    def test_known_good_routes_still_pass(self) -> None:
        for pose, slot in ((POSE_022217_HANDOFF, "A3"),
                           ((600.0, 600.0, 0.0), "A3"),
                           ((430.0, 600.0, 177.0), "B1")):
            with self.subTest(pose=pose, slot=slot):
                self.assertTrue(self._check(pose, slot).safe)

    def test_a_target_inside_the_arrival_radius_is_not_rejected(self) -> None:
        """이미 도착 반경 안이면 움직일 필요가 없다 — 뒤여도 거부하지 않는다."""
        pose = (430.0, 600.0, 177.0)
        self.assertLess(
            math.hypot(425.0 - pose[0], 600.0 - pose[1]),
            10.0 * self._route(pose, "B1")[0].position_tolerance_cm)
        self.assertTrue(self._check(pose, "B1").safe)

    def test_the_gate_is_off_by_default(self) -> None:
        r = validate_trajectory(
            self._route(POSE_231157_HANDOFF, "B1"),
            start_pose=POSE_231157_HANDOFF, target_slot="B1",
            min_turn_radius_mm=MIN_TURN_RADIUS_MM)
        self.assertTrue(r.safe, "기존 호출부는 영향받지 않는다")

    def test_reverse_first_waypoints_are_not_gated(self) -> None:
        """후진 구간은 이 판정의 대상이 아니다."""
        class _W:
            route_id = 1
            waypoint_id = 1
            phase = "RECOVERY"
            x, y = 100.0, 600.0
            target_heading_deg = 0.0
            motion_direction = "REVERSE"
            curvature = 0.0
            position_tolerance_cm = 8.0
        r = validate_trajectory([_W()], start_pose=(400.0, 600.0, 0.0),
                                min_turn_radius_mm=MIN_TURN_RADIUS_MM,
                                require_reachable_first_wp=True)
        self.assertNotEqual(r.reason, "FIRST_WP_BEHIND")


# ══ 회귀: 손대지 않기로 한 값들 ═════════════════════════════════════════════

class UntouchedContracts(unittest.TestCase):

    def test_safety_thresholds_are_unchanged(self) -> None:
        cfg, ctl = PipelineConfig(), ControllerConfig()
        self.assertEqual(ctl.max_pose_age_s, 0.5)
        self.assertEqual(ctl.stop_distance_cm, 3.0)
        self.assertEqual(ctl.steer_kp, 1.6)
        self.assertEqual(cfg.stationary_tolerance_mm, 15.0)
        self.assertEqual(cfg.boundary_hard_margin_mm, 20.0)
        self.assertEqual(cfg.boundary_measurement_uncertainty_mm, 10.0)

    def test_geometry_is_unchanged(self) -> None:
        from parking.waypoints import ON_AISLE_TOLERANCE_MM
        self.assertEqual(ON_AISLE_TOLERANCE_MM, 80.0)
        self.assertEqual(MIN_TURN_RADIUS_MM, 610.0)

    def test_no_slot_literals_in_production_logic(self) -> None:
        import re
        for path in ("pipeline/runner.py", "parking/waypoints.py",
                     "parking/trajectory_safety.py"):
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    code = line.split("#", 1)[0]
                    if '"""' in code or "'''" in code:
                        continue
                    self.assertIsNone(
                        re.search(r'["\'][AB][1-4]["\']', code),
                        f"{path}:{number} 슬롯 리터럴: {code.strip()}")


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()
