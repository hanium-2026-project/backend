"""Entrance-to-aisle production staging regressions from run 20260831_222643."""

from __future__ import annotations

import unittest

from control.auto_host_runner import MissionStatus
from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (AISLE_Y, MIN_TURN_RADIUS_MM,
                               build_waypoints, default_slot_specs)
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView
from pipeline.tests.test_entry_staging_handoff import settle_staging_plan


class _Runner:
    def __init__(self) -> None:
        self.loaded = []
        self.stopped = False

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        self.stopped = True


class EntranceGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ParkingPipeline(PipelineConfig(
            control_mode="auto-host", parking_mode="rear"))

    @staticmethod
    def _view(pose=(133.0, 209.0, 95.0)) -> VehicleView:
        return VehicleView(
            track_id=7, car_id=1, node="entrance",
            position_mm=pose[:2], heading_deg=pose[2],
            heading_source="FRONT_CUSHION", last_obs_time=1.0)

    def test_actual_222643_pose_reaches_safe_global_basin(self) -> None:
        view = self._view()
        self.pipeline.views = {7: view}
        first = self.pipeline._build_entry_staging_route(1, view, "B1", 1)
        self.assertTrue(first)
        self.assertTrue(validate_trajectory(
            first, start_pose=(133.0, 209.0, 95.0), target_slot="B1",
            min_turn_radius_mm=MIN_TURN_RADIUS_MM,
            initial_boundary_tolerance_mm=20.0).safe)

        view.position_mm = (first[-1].x, first[-1].y)
        view.heading_deg = first[-1].target_heading_deg
        view.last_obs_time = 2.0
        second = self.pipeline._build_entry_staging_route(1, view, "B1", 2)
        self.assertTrue(second)
        view.position_mm = (second[-1].x, second[-1].y)
        view.heading_deg = second[-1].target_heading_deg
        self.assertTrue(self.pipeline._entry_staging_ready(view))

        global_route = build_waypoints(
            default_slot_specs()["B1"], route_id=3,
            from_pose=view.position_mm, from_heading_deg=view.heading_deg,
            min_radius_mm=MIN_TURN_RADIUS_MM, strict=True)
        self.assertTrue(validate_trajectory(
            global_route,
            start_pose=(*view.position_mm, view.heading_deg),
            target_slot="B1", min_turn_radius_mm=MIN_TURN_RADIUS_MM,
            initial_boundary_tolerance_mm=20.0).safe)

    def test_center_start_keeps_direct_global_route(self) -> None:
        view = self._view((170.0, 604.0, 0.0))
        view.node = "junction"
        self.assertFalse(self.pipeline._entry_staging_needed(view))
        route = build_waypoints(
            default_slot_specs()["B1"], route_id=1,
            from_pose=view.position_mm, from_heading_deg=view.heading_deg,
            min_radius_mm=MIN_TURN_RADIUS_MM, strict=True)
        self.assertTrue(route)

    def test_staging_is_slot_general_not_b1_coordinate_hardcoded(self) -> None:
        endpoints = []
        for route_id, slot_id in enumerate(("A1", "A2", "B1", "B3"), 1):
            self.pipeline._entry_staging_signatures.clear()
            view = self._view()
            self.pipeline.views = {7: view}
            route = self.pipeline._build_entry_staging_route(
                1, view, slot_id, route_id)
            self.assertTrue(route, slot_id)
            endpoints.append((round(route[-1].x, 1), round(route[-1].y, 1)))
        # The entrance goal is aisle geometry, not a per-slot fixed point.
        self.assertTrue(all(abs(y - AISLE_Y) <= 80.0 for _, y in endpoints))

    def test_repeated_identical_staging_is_rejected(self) -> None:
        view = self._view()
        self.pipeline.views = {7: view}
        self.assertTrue(self.pipeline._build_entry_staging_route(
            1, view, "B1", 1))
        self.assertEqual(self.pipeline._build_entry_staging_route(
            1, view, "B1", 2), [])

    def test_parked_vehicle_is_never_crossed(self) -> None:
        view = self._view()
        self.pipeline.views = {7: view}
        self.pipeline._parked_obstacles = {2: (160.0, 520.0, 75.0)}
        route = self.pipeline._build_entry_staging_route(1, view, "B1", 1)
        # A safe alternative is acceptable; a bounded zero failure is also
        # acceptable.  An unsafe executable route is not.
        if route:
            verdict = validate_trajectory(
                route, start_pose=(133.0, 209.0, 95.0), target_slot="B1",
                obstacle_poses=((160.0, 520.0, 75.0),),
                obstacle_margin_mm=10.0,
                min_turn_radius_mm=MIN_TURN_RADIUS_MM,
                initial_boundary_tolerance_mm=20.0)
            self.assertTrue(verdict.safe, verdict.reason)


class EntranceCoordinatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ParkingPipeline(PipelineConfig(
            control_mode="auto-host", parking_mode="rear"))
        self.view = EntranceGeometryTest._view()
        self.pipeline.views = {7: self.view}
        self.pipeline.track_of_car = {1: 7}
        self.pipeline.auto_hosts = {1: _Runner()}
        self.pipeline._auto_host_slot = {1: "B1"}
        self.pipeline.allocator.update(7, self.view.position_mm)
        self.pipeline.allocator.reassign(7, "B1")
        self.events = []
        self.pipeline.on_event_record = lambda name, **fields: self.events.append(
            (name, fields))

    def test_blocked_preferred_slot_is_never_swapped_for_another(self) -> None:
        """예약된 슬롯은 경로 실패로 바뀌지 않는다 (run_20260904_210321).

        예전 계약은 "첫 후보가 막히면 다음 슬롯을 쓴다" 였다. 실차에서 그
        순회가 곧 재배정이 됐다: 입구에서 B1/A1/A2/A3 이 차례로
        NO_SAFE_PRODUCTIVE_MANEUVER 로 거절되고 A4 가 선택돼(slot_id 가
        t=25.8~78.1 전 구간 A4) 미션 내내 B1 로 돌아오지 못했다. B1 은
        점유도 장애물도 도달 불가도 아니었다.

        새 계약: 슬롯 선택(WHERE)과 경로 생성(HOW)을 분리한다. staging 은
        예약된 슬롯 하나만 풀고, 못 풀면 SAFE STOP 하되 예약은 유지한다.
        """
        from rl.parking_env import SLOT_NAMES
        self.pipeline._entry_staging_signatures.clear()
        self.pipeline._build_entry_staging_route = (
            lambda car, view, slot, rid, goal_test=None: [])

        self.pipeline._start_entry_staging(1, self.view, "B1", 2)
        self.assertTrue(settle_staging_plan(self.pipeline, self.view))

        # 다른 슬롯으로 새지 않는다.
        self.assertEqual(self.pipeline._auto_host_slot[1], "B1")
        self.assertEqual(self.pipeline.allocator.vehicles[7].assigned_slot, "B1")
        tried = [f.get("slot") for name, f in self.events
                 if name == "ENTRY_STAGING_CANDIDATE"]
        self.assertEqual(set(tried), {"B1"}, f"다른 슬롯을 시도했다: {tried}")
        # 예약은 유지된 채 안전 정지한다.
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_ENTRY_STAGING_FAILED")
        self.assertTrue(self.pipeline.auto_hosts[1].stopped)
        self.assertGreaterEqual(
            self.pipeline.allocator.slot_statuses[SLOT_NAMES.index("B1")], 0.5,
            "B1 예약이 풀리면 다른 차가 가져간다")

    def test_all_candidates_infeasible_is_bounded_zero_fault(self) -> None:
        self.pipeline._entry_staging_candidate_slots = lambda preferred: ["B1", "B2"]
        self.pipeline._build_entry_staging_route = lambda *args: []
        self.pipeline._start_entry_staging(1, self.view, "B1", 2)
        self.assertTrue(settle_staging_plan(self.pipeline, self.view))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_ENTRY_STAGING_FAILED")
        self.assertTrue(self.pipeline.auto_hosts[1].stopped)
        self.assertIn("FAULT", [name for name, _ in self.events])

    def test_done_waits_for_distinct_fresh_stopped_pose(self) -> None:
        self.pipeline._parking_stage[1] = "ENTRY_STAGING"
        self.view.last_obs_time = 10.0
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.DONE)
        self.assertEqual(self.pipeline._parking_stage[1],
                         "ENTRY_STAGING_PENDING")
        self.assertEqual(self.pipeline._entry_staging_wait[1], 10.0)


if __name__ == "__main__":
    unittest.main()
