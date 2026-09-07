"""Rear parking setup recovery의 production 단계 전이 회귀 테스트."""

from __future__ import annotations

import math
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

from control.auto_host_runner import MissionStatus
from controller.config import ControllerConfig
from parking.waypoints import (PHASE_DEFAULTS, SETUP_MIN_EXECUTABLE_MM,
                               InfeasibleRouteError,
                               REVERSE_START_HEADING_TOLERANCE_DEG,
                               build_setup_recovery_waypoints,
                               choose_rear_candidate,
                               default_slot_specs, plan_setup_recovery)
from pipeline.runner import ParkingPipeline, VehicleView


class _Runner:
    def __init__(self) -> None:
        self.loaded = []
        self.stopped = False
        self.replan_reason = "HEADING_OUT_OF_TOLERANCE"

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        self.stopped = True


class _Dashboard:
    def __init__(self) -> None:
        self.events = []

    def push_event(self, name, **fields) -> None:
        self.events.append((name, fields))


class _Server:
    def __init__(self) -> None:
        self.zeroed = []

    def stop_control(self, car_id) -> None:
        self.zeroed.append(car_id)


class TestParkingSetupRecoveryTransition(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ParkingPipeline.__new__(ParkingPipeline)
        self.pipeline.config = SimpleNamespace(
            parking_mode="rear", max_parking_recovery_attempts=3,
            stationary_tolerance_mm=15.0, stationary_window=3,
            # 같은 슬롯 재배치(_load_slot_reposition)가 쓰는 기존 staging 값들.
            # 실제 PipelineConfig 기본값과 같다.
            entry_staging_alignment_max_mm=1100.0,
            entry_staging_min_clearance_mm=35.0,
            entry_staging_heading_tolerance_deg=15.0)
        self.pipeline.auto_hosts = {1: _Runner()}
        self.pipeline._auto_host_slot = {1: "B1"}
        self.pipeline.track_of_car = {1: 7}
        self.pipeline.views = {
            7: VehicleView(track_id=7, car_id=1,
                           position_mm=(425.0, 600.0), heading_deg=0.0,
                           heading_source="FRONT_CUSHION")
        }
        for view in self.pipeline.views.values():
            view.recent.extend([view.position_mm] * 3)
        self.pipeline._parking_stage = {1: "DRIVING"}
        self.pipeline._parking_setup_wait = {}
        self.pipeline._parking_plan_wait = {}
        self.pipeline._parking_recovery_attempts = {}
        self.pipeline._initial_pose_samples = {}
        self.pipeline._heading_wait_state = {}
        self.pipeline.dashboard = _Dashboard()
        self.pipeline.server = _Server()
        self.pipeline.orchestrator = SimpleNamespace(next_route_id=lambda: 41)
        self.pipeline.on_route_load = None
        self.pipeline._auto_host_route = {}
        self.pipeline.on_event_record = None
        self.pipeline._trajectory_safe = lambda view, route, **kwargs: True
        self.pipeline._load_direct_rear_replan = lambda car_id, view: False

    def test_infeasible_rear_plan_loads_setup_instead_of_failing(self) -> None:
        setup = [SimpleNamespace(phase="RECOVERY")]
        self.pipeline._build_route = lambda *_: (_ for _ in ()).throw(
            InfeasibleRouteError("B1", "single arc infeasible"))
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=setup):
            self.assertTrue(self.pipeline._start_rear_parking_stage(1))
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1], setup)
        self.assertFalse(self.pipeline.auto_hosts[1].stopped)

    def test_setup_done_replans_from_new_camera_pose(self) -> None:
        fresh = (218.0, 627.0)
        self.pipeline._parking_stage[1] = "SETUP"
        self.pipeline.views[7].position_mm = fresh
        self.pipeline.views[7].heading_deg = 345.0
        seen = []
        parking = [SimpleNamespace(phase="ENTRY")]

        def build(_spec, view, _route_id):
            seen.append((view.position_mm, view.heading_deg))
            return parking

        self.pipeline._build_route = build
        self.pipeline._on_auto_host_status(1, MissionStatus.RUNNING,
                                           MissionStatus.DONE)
        self.assertEqual(seen, [])
        self.assertEqual(self.pipeline._parking_stage[1],
                         "PARKING_AFTER_SETUP_PENDING")
        self.pipeline.views[7].last_obs_time = 1.0
        self.pipeline._maybe_start_rear_after_stop(self.pipeline.views[7])
        self.assertEqual(seen, [(fresh, 345.0)])
        self.assertEqual(self.pipeline._parking_stage[1], "PARKING")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1], parking)

    def test_parking_replan_waits_for_fresh_pose_then_loads_setup(self) -> None:
        view = self.pipeline.views[7]
        view.last_obs_time = 10.0
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])

        setup = [SimpleNamespace(phase="RECOVERY")]
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=setup):
            self.pipeline._maybe_start_parking_setup(view)
            self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])
            view.last_obs_time = 10.1
            self.pipeline._maybe_start_parking_setup(view)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1], setup)

    def test_reverse_heading_timeout_enters_fresh_pose_setup_recovery(self) -> None:
        view = self.pipeline.views[7]
        view.last_obs_time = 30.0
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline.auto_hosts[1].replan_reason = "REVERSE_HEADING_TIMEOUT"

        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)

        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertEqual(self.pipeline._parking_setup_wait[1], 30.0)
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])

    def test_parking_recovery_attempts_exhaust_to_zero_wait(self) -> None:
        self.pipeline.config.max_parking_recovery_attempts = 1
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_RECOVERY_EXHAUSTED")
        self.assertEqual(self.pipeline.server.zeroed, [1])

    def test_no_safe_setup_stays_zero_and_never_uses_generic_replan(self) -> None:
        view = self.pipeline.views[7]
        view.last_obs_time = 20.0
        self.pipeline._parking_stage[1] = "SETUP_PENDING"
        self.pipeline._parking_setup_wait[1] = 19.0
        self.pipeline._load_parking_setup = lambda car_id, fresh: False
        # 같은 슬롯 재배치도 해가 없을 때의 계약이다 — 있으면 그쪽이 먼저다.
        self.pipeline._load_slot_reposition = lambda car_id, fresh: False
        self.pipeline._replan_auto_host = lambda car_id: self.fail(
            "parking setup failure escaped into generic/legacy replan")
        events = []
        self.pipeline.on_event_record = lambda name, **fields: events.append(
            (name, fields))

        self.pipeline._maybe_start_parking_setup(view)

        self.assertEqual(self.pipeline._parking_stage[1], "WAIT_SAFE_RECOVERY")
        self.assertEqual(self.pipeline.server.zeroed, [1])
        self.assertIn(("FAULT", {"car_id": 1,
                                  "reason": "NO_SAFE_PARKING_RECOVERY"}),
                      events)

    def test_initial_handoff_no_solution_is_an_explicit_safe_fault(self) -> None:
        events = []
        self.pipeline.on_event_record = lambda name, **fields: events.append(
            (name, fields))
        self.pipeline._build_route = lambda *_: (_ for _ in ()).throw(
            InfeasibleRouteError("B1", "no direct rear"))
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=[]):
            self.assertFalse(self.pipeline._start_rear_parking_stage(1))
        self.assertEqual(self.pipeline._parking_stage[1], "WAIT_SAFE_RECOVERY")
        self.assertTrue(self.pipeline.auto_hosts[1].stopped)
        self.assertIn(("FAULT", {"car_id": 1,
                                  "reason": "NO_SAFE_PARKING_RECOVERY"}),
                      events)

    def test_155812_track_churn_rebinds_then_fresh_heading_loads_recovery(self) -> None:
        old = VehicleView(
            track_id=2, car_id=1, slot_id="B1",
            position_mm=(704.6, 350.0), heading_deg=309.6,
            heading_source="TRAJECTORY", last_seen_frame=55,
            last_obs_time=1350.890)
        new = VehicleView(
            track_id=7, position_mm=(746.9, 310.7), heading_deg=None,
            heading_source=None, last_seen_frame=70,
            last_obs_time=1354.578)
        new.recent.extend([new.position_mm] * 3)
        self.pipeline.views = {2: old, 7: new}
        self.pipeline.track_of_car = {1: 2}
        self.pipeline._parking_stage[1] = "SETUP_PENDING"
        self.pipeline._parking_setup_wait[1] = 1350.890
        self.pipeline.config.track_rebind_stale_frames = 8
        self.pipeline.config.track_rebind_max_distance_mm = 150.0
        self.pipeline._lock = threading.RLock()
        removed = []
        self.pipeline.allocator = SimpleNamespace(
            vehicles={
                2: SimpleNamespace(assigned_slot="B1", route=["old"]),
                7: SimpleNamespace(assigned_slot=None, route=[]),
            },
            remove_vehicle=lambda track_id: removed.append(track_id),
        )
        heading_removed = []
        self.pipeline.heading = SimpleNamespace(
            remove=lambda track_id: heading_removed.append(track_id))
        recorded = []
        self.pipeline.on_event_record = lambda name, **fields: recorded.append(
            (name, fields))

        self.pipeline._maybe_rebind_recovery_track(new, frame_index=70)
        self.assertEqual(self.pipeline.track_of_car[1], 7)
        self.assertEqual(new.car_id, 1)
        self.assertEqual(new.slot_id, "B1")
        self.assertIsNone(old.car_id)
        self.assertNotIn(2, self.pipeline.views)
        self.assertEqual(removed, [2])
        self.assertEqual(heading_removed, [2])
        self.assertEqual(recorded[0][0], "TRACK_REBOUND")

        # No heading is copied across the identity boundary, so it remains
        # safely stopped until a genuinely fresh physical heading arrives.
        self.pipeline._maybe_start_parking_setup(new)
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])
        new.heading_deg = 309.6
        new.heading_source = "FRONT_CUSHION"
        setup = [SimpleNamespace(phase="RECOVERY")]
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=setup):
            self.pipeline._maybe_start_parking_setup(new)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1], setup)

    def test_203616_entry_staging_pending_track_churn_rebinds(self) -> None:
        """A new track 5.6 mm away must not strand wp9/9 for 29 seconds."""
        old = VehicleView(
            track_id=1, car_id=1, slot_id="B1",
            position_mm=(443.6, 666.5), heading_deg=45.1,
            heading_source="FRONT_CUSHION", last_seen_frame=55,
            last_obs_time=22.562)
        new = VehicleView(
            track_id=12, position_mm=(443.6, 660.9), heading_deg=44.6,
            heading_source="FRONT_CUSHION", last_seen_frame=105,
            last_obs_time=33.890)
        self.pipeline.views = {1: old, 12: new}
        self.pipeline.track_of_car = {1: 1}
        self.pipeline._parking_stage[1] = "ENTRY_STAGING_PENDING"
        self.pipeline.config.track_rebind_stale_frames = 8
        self.pipeline.config.track_rebind_max_distance_mm = 150.0
        self.pipeline._lock = threading.RLock()
        self.pipeline.allocator = SimpleNamespace(
            vehicles={
                1: SimpleNamespace(assigned_slot="B1", route=["old"]),
                12: SimpleNamespace(assigned_slot=None, route=[]),
            },
            remove_vehicle=lambda _track_id: None,
        )
        self.pipeline.heading = SimpleNamespace(remove=lambda _track_id: None)

        self.pipeline._maybe_rebind_recovery_track(new, frame_index=105)

        self.assertEqual(self.pipeline.track_of_car[1], 12)
        self.assertEqual(new.car_id, 1)
        self.assertIsNone(old.car_id)

    def test_175349_comm_hold_allows_safe_track_rebind(self) -> None:
        old = VehicleView(
            track_id=2, car_id=1, slot_id="B1",
            position_mm=(779.6, 151.7), heading_deg=336.0,
            heading_source="LAST_VALID", last_seen_frame=55,
            last_obs_time=100.0)
        new = VehicleView(
            track_id=15, position_mm=(703.0, 272.0), heading_deg=None,
            heading_source=None, last_seen_frame=70, last_obs_time=104.0)
        self.pipeline.views = {2: old, 15: new}
        self.pipeline.track_of_car = {1: 2}
        self.pipeline._parking_stage[1] = "SETUP"
        self.pipeline._comm_lost = {1}
        self.pipeline._comm_recovery_context = {
            1: {"state": "WAIT_CONNECTION", "track_id": 2, "slot_id": "B1"}
        }
        self.pipeline.config.track_rebind_stale_frames = 8
        self.pipeline.config.track_rebind_max_distance_mm = 150.0
        self.pipeline._lock = threading.RLock()
        self.pipeline.allocator = SimpleNamespace(
            vehicles={
                2: SimpleNamespace(assigned_slot="B1", route=["old"]),
                15: SimpleNamespace(assigned_slot=None, route=[]),
            },
            remove_vehicle=lambda _track_id: None,
        )
        self.pipeline.heading = SimpleNamespace(remove=lambda _track_id: None)

        self.pipeline._maybe_rebind_recovery_track(new, frame_index=70)

        self.assertEqual(self.pipeline.track_of_car[1], 15)
        self.assertEqual(new.car_id, 1)
        self.assertIsNone(new.heading_deg, "stale heading crossed track identity")
        self.assertEqual(
            self.pipeline._comm_recovery_context[1]["track_id"], 15)

    def test_rebound_track_without_heading_times_out_to_explicit_zero_fault(self) -> None:
        view = self.pipeline.views[7]
        view.heading_deg = None
        view.heading_source = None
        view.last_obs_time = 10.0
        self.pipeline._parking_stage[1] = "SETUP_PENDING"
        self.pipeline._parking_setup_wait[1] = 9.0
        recorded = []
        self.pipeline.on_event_record = lambda name, **fields: recorded.append(
            (name, fields))

        self.pipeline._maybe_start_parking_setup(view)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertFalse(self.pipeline.auto_hosts[1].stopped)

        view.last_obs_time = 12.6  # >2.5 s, ten fresh frames at ~4 FPS
        self.pipeline._maybe_start_parking_setup(view)
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_FRESH_HEADING_FAULT")
        self.assertTrue(self.pipeline.auto_hosts[1].stopped)
        self.assertIn("WAIT_FOR_FRESH_HEADING_TIMEOUT",
                      [name for name, _ in recorded])
        self.assertIn(
            ("FAULT", {"car_id": 1, "reason": "FRESH_HEADING_TIMEOUT",
                       "boundary": "PARKING_RECOVERY_REPLAN"}),
            recorded)


class TestBidirectionalSetupSearch(unittest.TestCase):
    def test_real_failure_pose_finds_rear_feasible_terminal_pose(self) -> None:
        spec = default_slot_specs()["B1"]
        recovery = plan_setup_recovery(spec, (792.2, 339.2), 323.8)
        self.assertIsNotNone(recovery)
        self.assertLessEqual(len(recovery.segments), 3)
        candidate, _ = choose_rear_candidate(
            spec, recovery.end_pose[:2], recovery.end_pose[2])
        self.assertIsNotNone(candidate)
        self.assertTrue(all(segment.reverse in (True, False)
                            for segment in recovery.segments))

    def test_obstacle_footprint_blocks_setup_trajectory(self) -> None:
        spec = default_slot_specs()["B1"]
        recovery = plan_setup_recovery(
            spec, (792.2, 339.2), 323.8,
            obstacle_poses=((792.2, 339.2, 323.8),))
        self.assertIsNone(recovery)


def _wp(x, y, *, tolerance_cm=8.0, phase="RECOVERY"):
    return SimpleNamespace(x=x, y=y, phase=phase,
                           position_tolerance_cm=tolerance_cm,
                           motion_direction="REVERSE", route_id=41,
                           waypoint_id=1, curvature=0.0)


class TestDegenerateSetupRejection(unittest.TestCase):
    """run_20260824_192746 route 4~16: 서 있는 자리에서 즉시 DONE 되는 setup.

    실측: 차량 (292,571) 에서 wp1 (243,584) / wp2 (195,597) — 총 100mm 기동인데
    waypoint 허용오차가 8cm 라 두 점 모두 도착 반경 안에 있었다. 13개 route 가
    5.6초 동안 throttle 0 / encoder 0 으로 생성·완료를 반복했다.
    """

    def setUp(self) -> None:
        self.pipeline = ParkingPipeline.__new__(ParkingPipeline)
        self.pipeline.config = SimpleNamespace(
            parking_mode="rear", max_parking_recovery_attempts=3,
            stationary_tolerance_mm=15.0, stationary_window=3,
            controller_config=ControllerConfig())
        self.pipeline.auto_hosts = {1: _Runner()}
        self.pipeline._auto_host_slot = {1: "B1"}
        self.pipeline.track_of_car = {1: 7}
        self.pipeline.views = {
            7: VehicleView(track_id=7, car_id=1,
                           position_mm=(292.0, 571.0), heading_deg=344.6,
                           heading_source="FRONT_CUSHION")
        }
        for view in self.pipeline.views.values():
            view.recent.extend([view.position_mm] * 3)
        self.pipeline._parking_stage = {1: "PARKING"}
        self.pipeline._parking_setup_wait = {}
        self.pipeline._parking_plan_wait = {}
        self.pipeline._parking_recovery_attempts = {}
        self.pipeline._initial_pose_samples = {}
        self.pipeline._heading_wait_state = {}
        self.pipeline.dashboard = _Dashboard()
        self.pipeline.server = _Server()
        self.pipeline.orchestrator = SimpleNamespace(next_route_id=lambda: 41)
        self.pipeline.on_route_load = None
        self.pipeline._auto_host_route = {}
        self.pipeline.on_event_record = None
        self.pipeline._trajectory_safe = lambda view, route, **kwargs: True
        self.pipeline._load_direct_rear_replan = lambda car_id, view: False
        self.pipeline._build_route = lambda *_: (_ for _ in ()).throw(
            InfeasibleRouteError("B1", "single arc infeasible"))

    # 실측 route 4 기하 — 차량 (292,571) 로부터 50.7mm / 100.4mm
    DEGENERATE = [_wp(244.9, 584.6), _wp(196.9, 598.5)]
    # 같은 방향이지만 도착 반경 밖까지 실제로 물러나는 기동
    PRODUCTIVE = [_wp(150.0, 620.0), _wp(0.0, 660.0)]

    def test_degenerate_setup_is_detected(self) -> None:
        view = self.pipeline.views[7]
        self.assertTrue(
            self.pipeline._setup_is_degenerate(view, self.DEGENERATE))
        self.assertFalse(
            self.pipeline._setup_is_degenerate(view, self.PRODUCTIVE))

    def test_rear_stage_rejects_degenerate_setup_instead_of_looping(self) -> None:
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=self.DEGENERATE):
            self.assertFalse(self.pipeline._start_rear_parking_stage(1))
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])
        self.assertEqual(self.pipeline._parking_stage[1], "WAIT_SAFE_RECOVERY")
        self.assertTrue(self.pipeline.auto_hosts[1].stopped)

    def test_parking_setup_load_rejects_degenerate_setup(self) -> None:
        view = self.pipeline.views[7]
        view.last_obs_time = 10.0
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=self.DEGENERATE):
            self.assertFalse(self.pipeline._load_parking_setup(1, view))
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])

    def test_productive_setup_still_loads(self) -> None:
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=self.PRODUCTIVE):
            self.assertTrue(self.pipeline._start_rear_parking_stage(1))
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP")
        self.assertEqual(self.pipeline.auto_hosts[1].loaded[-1],
                         self.PRODUCTIVE)

    def test_productive_setups_do_not_consume_recovery_budget(self) -> None:
        """run_20260824_204027: 실제로 차를 옮긴 setup 3회가 예산을 먹으면 안 된다.

        실측 route2/3/4 는 각각 30.7 / 36.5 / 41.0mm 를 실제로 이동하고
        encoder 194 / 213 / 242 를 냈다. 그 결과 8-waypoint 주차 경로가
        생성됐는데, 그 직후 첫 정당한 REPLAN_REQUIRED 에서
        PARKING_RECOVERY_EXHAUSTED 로 죽었다.
        """
        runner = self.pipeline.auto_hosts[1]
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=self.PRODUCTIVE):
            for _ in range(5):
                self.assertTrue(self.pipeline._start_rear_parking_stage(1))
        self.assertEqual(len(runner.loaded), 5)
        self.assertFalse(runner.stopped)
        self.assertEqual(self.pipeline._parking_recovery_attempts.get(1, 0), 0)
        self.assertNotEqual(self.pipeline._parking_stage[1],
                            "WAIT_RECOVERY_EXHAUSTED")

    def test_first_genuine_replan_after_setups_still_has_full_budget(self) -> None:
        """setup 을 여러 번 한 뒤에도 첫 REPLAN_REQUIRED 는 attempt 1 이어야 한다."""
        view = self.pipeline.views[7]
        view.last_obs_time = 10.0
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=self.PRODUCTIVE):
            for _ in range(3):
                self.pipeline._start_rear_parking_stage(1)
        # 주차 경로 진행 중 첫 genuine recovery 요청
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(self.pipeline._parking_recovery_attempts.get(1), 1)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")

    def test_recovery_budget_still_exhausts_on_repeated_replan(self) -> None:
        """예산 자체(max 3)는 그대로 살아 있어야 한다."""
        self.pipeline._parking_stage[1] = "PARKING"
        for _ in range(self.pipeline.config.max_parking_recovery_attempts):
            self.pipeline._parking_stage[1] = "PARKING"
            self.pipeline._on_auto_host_status(
                1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_RECOVERY_EXHAUSTED")
        self.assertIn(1, self.pipeline.server.zeroed)


class TestSetupMinimumExecutableDistance(unittest.TestCase):
    """planner 가 도착 반경 안에서 끝나는 기동을 best 로 고르면 안 된다.

    run_20260824_192746 / run_20260824_204240 에서 확인된 결함: 최단 직선 후보
    100mm 가 50mm 간격으로 표본화돼 waypoint 가 50mm/100mm 에 놓이는데, RECOVERY
    도착 반경은 8cm 허용오차 + 3cm 정지여유 = 110mm 다.
    """

    RADIUS_MM = 10.0 * ControllerConfig().arrival_radius_cm(
        PHASE_DEFAULTS["RECOVERY"]["position_tolerance_cm"], "RECOVERY")

    # 실측 실패 자세들
    POSES = (
        ("204240", (296.3, 603.0), 343.1),
        ("192746", (292.0, 571.0), 344.6),
        ("204027-r2", (450.0, 600.0), 340.0),
    )

    def test_arrival_radius_assumption_holds(self) -> None:
        self.assertAlmostEqual(self.RADIUS_MM, 110.0, places=6)

    def test_planner_never_returns_maneuver_inside_arrival_radius(self) -> None:
        spec = default_slot_specs()["B1"]
        for label, pose, heading in self.POSES:
            with self.subTest(pose=label):
                wps = build_setup_recovery_waypoints(
                    spec, route_id=99, from_pose=pose,
                    from_heading_deg=heading,
                    min_executable_mm=self.RADIUS_MM)
                if not wps:
                    continue        # 명시적 infeasible 도 허용되는 결과다
                end = wps[-1]
                self.assertGreater(
                    math.hypot(end.x - pose[0], end.y - pose[1]),
                    self.RADIUS_MM,
                    f"{label}: 마지막 waypoint 가 도착 반경 안이다")

    def test_previously_degenerate_poses_now_yield_drivable_setup(self) -> None:
        """100mm 후보가 걸러진 뒤에도 실행 가능한 대안이 나와야 한다."""
        spec = default_slot_specs()["B1"]
        for label, pose, heading in self.POSES:
            with self.subTest(pose=label):
                wps = build_setup_recovery_waypoints(
                    spec, route_id=99, from_pose=pose,
                    from_heading_deg=heading,
                    min_executable_mm=self.RADIUS_MM)
                self.assertTrue(wps, f"{label}: 대안 기동이 없다")

    def test_planner_default_is_derived_not_hardcoded(self) -> None:
        self.assertAlmostEqual(
            SETUP_MIN_EXECUTABLE_MM,
            PHASE_DEFAULTS["RECOVERY"]["position_tolerance_cm"] * 10.0)

    def test_each_setup_primitive_requires_its_planned_terminal_heading(self) -> None:
        """Direction/curvature changes cannot be captured without executing."""
        spec = default_slot_specs()["B1"]
        start = (414.9, 606.3)
        heading = 46.9
        recovery = plan_setup_recovery(
            spec, start, heading, min_executable_mm=self.RADIUS_MM)
        self.assertIsNotNone(recovery)
        waypoints = build_setup_recovery_waypoints(
            spec, route_id=99, from_pose=start,
            from_heading_deg=heading, min_executable_mm=self.RADIUS_MM)
        cursor = 0
        for segment in recovery.segments:
            cursor += len(segment.poses) - 1
            terminal = waypoints[cursor - 1]
            self.assertTrue(terminal.heading_required)
            self.assertEqual(terminal.heading_tolerance_deg,
                             REVERSE_START_HEADING_TOLERANCE_DEG)


class TestRepositionRadiusFallback(unittest.TestCase):
    """차의 실측 선회 능력까지 내려가는 2단계 탐색 (2026-09-03 회귀).

    run_20260903_013050 / _013212 는 둘 다 NO_SAFE_SETUP_MANEUVER ->
    NO_SAFE_PARKING_RECOVERY 로 끝났다. 그런데 그 자세들은 물리적으로 불가능한
    것이 아니라, 재배치 탐색이 800~1100mm 만 보고 있어서 차가 실제로 낼 수 있는
    610mm(2026-08-12 실측 최소 선회반경)를 쓰지 않았기 때문이었다.

    1단계(넓은 원호)는 그대로 두고, 1단계가 **아무 해도 못 찾을 때만**
    610mm 를 포함해 한 번 더 본다. 안전 기준은 하나도 바뀌지 않는다.
    """

    # 실차 로그의 종료 자세 (WAIT_SAFE_RECOVERY 로 굳은 지점)
    POSE_013050 = (399.7, 613.5, 47.7)
    POSE_013212 = (886.8, 617.9, 9.8)
    # 재배치가 잘 되던 자세 (1단계에서 풀린다)
    POSE_222253 = (645.0, 300.0, 95.0)

    @staticmethod
    def _plan(pose, slot_id, **kw):
        return build_setup_recovery_waypoints(
            default_slot_specs()[slot_id], route_id=1,
            from_pose=(pose[0], pose[1]), from_heading_deg=pose[2],
            min_executable_mm=110.0, **kw)

    def test_013050_terminal_pose_now_has_a_maneuver(self):
        self.assertTrue(self._plan(self.POSE_013050, "B1"))

    def test_013212_terminal_pose_now_has_a_maneuver(self):
        self.assertTrue(self._plan(self.POSE_013212, "A3"))

    def test_those_poses_have_no_solution_in_tier_one(self):
        """이게 실차에서 mission 을 끝낸 지점이다 — 근거를 고정한다."""
        for pose, slot in ((self.POSE_013050, "B1"),
                           (self.POSE_013212, "A3")):
            with self.subTest(pose=pose):
                self.assertEqual(
                    self._plan(pose, slot, fallback_radii_mm=None), [])

    def test_tier_one_result_is_untouched_where_it_already_worked(self):
        """넓은 원호로 풀리던 자세는 fallback 이 있어도 같은 해를 낸다."""
        with_fb = self._plan(self.POSE_222253, "A2")
        tier1 = self._plan(self.POSE_222253, "A2", fallback_radii_mm=None)
        self.assertTrue(tier1)
        self.assertEqual([(w.x, w.y) for w in with_fb],
                         [(w.x, w.y) for w in tier1])

    def test_fallback_maneuvers_still_pass_trajectory_safety(self):
        """탐색만 넓혔지 안전 게이트는 그대로 통과해야 한다."""
        from parking.trajectory_safety import validate_trajectory
        for pose, slot in ((self.POSE_013050, "B1"),
                           (self.POSE_013212, "A3")):
            with self.subTest(pose=pose):
                wps = self._plan(pose, slot)
                self.assertTrue(wps)
                result = validate_trajectory(wps, start_pose=pose,
                                             target_slot=slot)
                self.assertTrue(result.safe, result.reason)

    def test_fallback_does_not_invent_a_solution_where_none_exists(self):
        """진짜 불가능한 자세는 여전히 빈 목록이다."""
        boxed = (60.0, 60.0, 45.0)
        self.assertEqual(
            self._plan(boxed, "B4", max_total_length_mm=200.0), [])


class TestGlobalRejectionIsNotPhysicalImpossibility(unittest.TestCase):
    """run_20260903_013212: GLOBAL 이 버린 슬롯을 재배치는 풀 수 있었다.

    (346.4, 514.9, 342.95deg) 에서 GLOBAL 은 A1/A2/B1/B2 를 "통로 합류를
    마치는 x=794 가 인계 지점을 지나친다" 로 전부 후보에서 제외했다. 그 규칙은
    **단일 통과(one-pass) 접근 가정**이지 물리적 불가능이 아니다 — 같은
    자세에서 setup/recovery 는 네 슬롯 모두에 대해 재배치 기동을 찾는다.

    지금은 동작을 바꾸지 않는다. 이 사실을 고정해 두어, 나중에 GLOBAL 후보
    선택을 재배치와 연결할지 판단할 근거로 삼는다.
    """

    POSE = (346.4, 514.9, 342.95)

    def test_global_rejects_those_slots(self):
        from parking.waypoints import build_waypoints, MIN_TURN_RADIUS_MM
        for slot_id in ("A1", "A2", "B1", "B2"):
            with self.subTest(slot=slot_id):
                with self.assertRaises(InfeasibleRouteError):
                    build_waypoints(
                        default_slot_specs()[slot_id], route_id=1,
                        from_pose=(self.POSE[0], self.POSE[1]),
                        from_heading_deg=self.POSE[2],
                        min_radius_mm=MIN_TURN_RADIUS_MM, strict=True)

    def test_reposition_can_still_serve_those_slots(self):
        for slot_id in ("A1", "A2", "B1", "B2"):
            with self.subTest(slot=slot_id):
                wps = build_setup_recovery_waypoints(
                    default_slot_specs()[slot_id], route_id=1,
                    from_pose=(self.POSE[0], self.POSE[1]),
                    from_heading_deg=self.POSE[2], min_executable_mm=110.0)
                self.assertTrue(wps, f"{slot_id}: 재배치 해가 없다")


if __name__ == "__main__":
    unittest.main()
