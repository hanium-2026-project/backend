"""Parking lifecycle liveness: 관측이 돌아오면 반드시 다시 진행해야 한다.

안전 정지와 소프트웨어 정지를 구분하는 회귀들이다. 안전 정지(boundary,
unsafe route, COMM)는 그대로 latch 되어야 하고, "관측을 기다리는" 상태는
관측이 돌아오면 bounded time 안에 motion 또는 새 route 로 진행해야 한다.

근거 run: run_20260827_212955 / run_20260827_213332 — WAIT_FRESH_HEADING_FAULT
진입 시 runner.stop() 이 authority 를 FAULTED 로 latch 했는데 파이프라인에
이를 푸는 경로가 없어, heading 이 돌아와도 영구 zero 로 남았다.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from control.auto_host_runner import MissionStatus
from controller.config import ControllerConfig
from parking.waypoints import InfeasibleRouteError
from pipeline.runner import ParkingPipeline, VehicleView


class _Authority:
    def __init__(self) -> None:
        self.is_faulted = False
        self.re_armed = 0
        self.fault_reason = ""

    def fault(self, reason: str = "STOP") -> None:
        self.is_faulted = True
        self.fault_reason = reason


class _Host:
    def __init__(self) -> None:
        self.authority = _Authority()

    def re_arm_auto(self) -> None:
        self.authority.is_faulted = False
        self.authority.fault_reason = ""
        self.authority.re_armed += 1


class _Scheduler:
    def __init__(self) -> None:
        self.running = True
        self.starts = 0

    def start(self) -> None:
        self.running = True
        self.starts += 1

    def stop(self) -> None:
        self.running = False


class _Runner:
    def __init__(self) -> None:
        self.loaded = []
        self.stopped = False
        self.replan_reason = "REVERSE_HEADING_TIMEOUT"
        self.host = _Host()
        self.scheduler = _Scheduler()

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        self.stopped = True
        self.host.authority.fault()
        self.scheduler.stop()


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


def _wp(x, y, *, tolerance_cm=8.0, phase="RECOVERY"):
    return SimpleNamespace(x=x, y=y, phase=phase,
                           position_tolerance_cm=tolerance_cm,
                           motion_direction="REVERSE", route_id=77,
                           waypoint_id=1, curvature=0.0)


SETUP = [_wp(150.0, 620.0), _wp(0.0, 660.0)]


class TestHeadingFaultLiveness(unittest.TestCase):
    def setUp(self) -> None:
        p = ParkingPipeline.__new__(ParkingPipeline)
        self.pipeline = p
        p.config = SimpleNamespace(
            parking_mode="rear", max_parking_recovery_attempts=3,
            max_replan_attempts=3, initial_pose_stability_mm=30.0,
            stationary_tolerance_mm=15.0, stationary_window=3,
            critical_heading_wait_timeout_s=2.5,
            parking_stall_timeout_s=8.0,
            controller_config=ControllerConfig())
        p.auto_hosts = {1: _Runner()}
        p._auto_host_slot = {1: "B1"}
        p.track_of_car = {1: 7}
        p.views = {7: VehicleView(track_id=7, car_id=1,
                                  position_mm=(292.0, 571.0), heading_deg=344.6,
                                  heading_source="FRONT_CUSHION")}
        for v in p.views.values():
            v.recent.extend([v.position_mm] * 3)
            v.last_obs_time = 10.0
        p._parking_stage = {1: "PARKING"}
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
        p.orchestrator = SimpleNamespace(next_route_id=lambda: 77)
        p.on_route_load = None
        p._auto_host_route = {}
        p.on_event_record = None
        p._trajectory_safe = lambda view, route, **kw: True
        p._load_direct_rear_replan = lambda car_id, view: False
        p._build_route = lambda *_: (_ for _ in ()).throw(
            InfeasibleRouteError("B1", "single arc infeasible"))

    def _starve_heading(self) -> None:
        """관측이 끊겨 heading 대기 timeout 이 나도록 한다."""
        view = self.pipeline.views[7]
        view.heading_source = "LAST_VALID"
        self.pipeline._require_critical_heading(view, "PARKING_RECOVERY_REPLAN")
        view.last_obs_time += 3.0
        self.pipeline._require_critical_heading(view, "PARKING_RECOVERY_REPLAN")

    def test_heading_timeout_latches_zero(self) -> None:
        self._starve_heading()
        runner = self.pipeline.auto_hosts[1]
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_FRESH_HEADING_FAULT")
        self.assertTrue(runner.stopped)
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertFalse(runner.scheduler.running)
        self.assertIn(1, self.pipeline._heading_fault_hold)

    def test_recovered_heading_leaves_the_wait_state(self) -> None:
        self._starve_heading()
        view = self.pipeline.views[7]
        view.heading_source = "FRONT_CUSHION"
        view.last_obs_time += 0.25
        self.assertTrue(self.pipeline._maybe_resume_heading_fault(view))
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")

    def test_recovery_does_not_move_before_a_validated_route(self) -> None:
        """재개 시점에는 아직 zero 여야 한다 — 옛 route 로 달리면 안 된다."""
        self._starve_heading()
        view = self.pipeline.views[7]
        view.heading_source = "FRONT_CUSHION"
        view.last_obs_time += 0.25
        self.pipeline._maybe_resume_heading_fault(view)
        runner = self.pipeline.auto_hosts[1]
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertFalse(runner.scheduler.running)
        self.assertEqual(runner.loaded, [])

    def test_validated_route_reactivates_and_progresses(self) -> None:
        """LIVENESS: 관측 복귀 + feasible route → 실제로 다시 진행한다."""
        self._starve_heading()
        view = self.pipeline.views[7]
        view.heading_source = "FRONT_CUSHION"
        view.last_obs_time += 0.25
        self.pipeline._maybe_resume_heading_fault(view)

        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            view.last_obs_time += 0.25
            self.pipeline._maybe_start_parking_setup(view)

        runner = self.pipeline.auto_hosts[1]
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP")
        self.assertEqual(runner.loaded[-1], SETUP)
        self.assertFalse(runner.host.authority.is_faulted)
        self.assertTrue(runner.scheduler.running)
        self.assertEqual(runner.host.authority.re_armed, 1)
        self.assertNotIn(1, self.pipeline._heading_fault_hold)

    def test_no_infinite_zero_loop_when_perception_returns(self) -> None:
        """bounded steps 안에 반드시 motion 또는 새 route 로 진행해야 한다."""
        self._starve_heading()
        view = self.pipeline.views[7]
        runner = self.pipeline.auto_hosts[1]
        progressed = False
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            for step in range(12):
                view.heading_source = "FRONT_CUSHION"
                view.last_obs_time += 0.25
                self.pipeline._maybe_resume_heading_fault(view)
                self.pipeline._maybe_start_parking_setup(view)
                if runner.loaded and runner.scheduler.running:
                    progressed = True
                    break
        self.assertTrue(progressed,
                        "perception 이 돌아왔는데도 12 step 동안 진행이 없다")

    def test_still_stale_heading_never_reactivates(self) -> None:
        self._starve_heading()
        view = self.pipeline.views[7]
        view.heading_source = "LAST_VALID"
        view.last_obs_time += 0.25
        self.assertFalse(self.pipeline._maybe_resume_heading_fault(view))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_FRESH_HEADING_FAULT")
        self.assertTrue(self.pipeline.auto_hosts[1].host.authority.is_faulted)

    def test_moving_vehicle_never_reactivates(self) -> None:
        """정지 확인 전에는 재개하지 않는다."""
        self._starve_heading()
        view = self.pipeline.views[7]
        view.heading_source = "FRONT_CUSHION"
        view.recent.extend([(292.0, 571.0), (400.0, 600.0), (500.0, 640.0)])
        view.last_obs_time += 0.25
        self.assertFalse(self.pipeline._maybe_resume_heading_fault(view))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_FRESH_HEADING_FAULT")

    def test_physical_safety_fault_is_never_reactivated(self) -> None:
        """boundary/unsafe 같은 진짜 fault 는 이 경로로 풀리지 않는다."""
        runner = self.pipeline.auto_hosts[1]
        self.pipeline._parking_stage[1] = "PARKING"
        runner.stop()                      # BOUNDARY_HARD 등으로 latch 되었다고 가정
        self.assertNotIn(1, self.pipeline._heading_fault_hold)
        self.pipeline._reactivate_after_heading_fault(1)
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertFalse(runner.scheduler.running)
        self.assertEqual(runner.host.authority.re_armed, 0)




class TestStaleObservationLiveness(unittest.TestCase):
    """카메라 gap 으로 POSE_STALE latch 된 뒤 관측이 돌아오는 경우.

    run_20260827_213110 (0.70s gap) / run_20260827_213150 (21.7s gap):
    max_pose_age_s 초과로 controller 가 FAULTED latch → 파이프라인에 복귀
    경로가 없어 run 이 끝날 때까지 zero.
    """

    def setUp(self) -> None:
        base = TestHeadingFaultLiveness()
        base.setUp()
        self.pipeline = base.pipeline

    def _stale_latch(self) -> None:
        runner = self.pipeline.auto_hosts[1]
        self.pipeline._parking_stage[1] = "PARKING"
        runner.stop()
        runner.host.authority._reason = "POSE_STALE"
        runner.host.authority.fault_reason = "POSE_STALE"

    def test_pose_stale_latch_is_recognised_as_observation_wait(self) -> None:
        self._stale_latch()
        self.assertTrue(self.pipeline._stale_observation_latched(1))

    def test_pose_stale_recovers_and_progresses(self) -> None:
        self._stale_latch()
        view = self.pipeline.views[7]
        view.heading_source = "FRONT_CUSHION"
        view.last_obs_time += 1.0
        self.assertTrue(self.pipeline._maybe_resume_heading_fault(view))
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            view.last_obs_time += 0.25
            self.pipeline._maybe_start_parking_setup(view)
        runner = self.pipeline.auto_hosts[1]
        self.assertEqual(runner.loaded[-1], SETUP)
        self.assertFalse(runner.host.authority.is_faulted)
        self.assertTrue(runner.scheduler.running)

    def test_non_observation_fault_is_not_recovered(self) -> None:
        runner = self.pipeline.auto_hosts[1]
        self.pipeline._parking_stage[1] = "PARKING"
        runner.stop()
        runner.host.authority.fault_reason = "BOUNDARY_HARD"
        self.assertFalse(self.pipeline._stale_observation_latched(1))
        view = self.pipeline.views[7]
        view.heading_source = "FRONT_CUSHION"
        self.assertFalse(self.pipeline._maybe_resume_heading_fault(view))
        self.assertTrue(runner.host.authority.is_faulted)

    def test_stale_latch_outside_parking_is_ignored(self) -> None:
        runner = self.pipeline.auto_hosts[1]
        self.pipeline._parking_stage[1] = "DRIVING"
        runner.stop()
        runner.host.authority.fault_reason = "POSE_STALE"
        self.assertFalse(self.pipeline._stale_observation_latched(1))


class TestRepeatedReplanLiveness(unittest.TestCase):
    """run_20260827_212848: perception 99% 정상인데 48초 무동작.

    handoff 직후(PARKING_HANDOFF_PENDING) REPLAN_REQUIRED 가 통로 인계용 legacy
    전역 재계획으로 빠졌고, 거기서 identical replan 이 감지되자 읽는 곳이 없는
    WAIT_REPEATED_REPLAN 으로 죽었다.
    """

    def setUp(self) -> None:
        base = TestHeadingFaultLiveness()
        base.setUp()
        self.pipeline = p = base.pipeline
        p._replan_attempts = {}
        p._last_replan_signature = {}
        p._stall_since = {}
        p._begin_parking_handoff = lambda *a, **k: None
        p._is_global_handoff_terminal = lambda car_id, target: False
        p._handoff_region_reached = lambda view, target: False
        runner = p.auto_hosts[1]
        runner.failed_target = SimpleNamespace(
            x_mm=430.0, y_mm=600.0, route_id=5, waypoint_id=1)
        runner.replan_reason = "HEADING_OUT_OF_TOLERANCE"
        runner.last_tick_result = SimpleNamespace(
            command=SimpleNamespace(throttle=0.0))

    def _signature(self):
        view = self.pipeline.views[7]
        return ("HEADING_OUT_OF_TOLERANCE", 430.0, 600.0,
                view.position_mm[0], view.position_mm[1])

    def test_handoff_pending_replan_uses_rear_coordinator(self) -> None:
        """212848 핵심: handoff 대기 중 replan 이 legacy 경로로 새면 안 된다."""
        self.pipeline._parking_stage[1] = "PARKING_HANDOFF_PENDING"
        self.pipeline._on_auto_host_status(
            1, MissionStatus.RUNNING, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertEqual(self.pipeline._parking_recovery_attempts.get(1), 1)

    def test_repeated_identical_replan_escalates_to_setup(self) -> None:
        self.pipeline._last_replan_signature[1] = self._signature()
        self.pipeline._parking_stage[1] = "WAIT_REPEATED_REPLAN"
        self.pipeline._replan_auto_host(1)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")
        self.assertNotIn(1, self.pipeline._last_replan_signature)

    def test_repeated_replan_is_never_a_silent_zero_state(self) -> None:
        """LIVENESS: perception 정상 + feasible setup → bounded step 내 진행."""
        view = self.pipeline.views[7]
        self.pipeline._last_replan_signature[1] = self._signature()
        self.pipeline._parking_stage[1] = "PARKING_HANDOFF_PENDING"
        runner = self.pipeline.auto_hosts[1]
        progressed = False
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            self.pipeline._replan_auto_host(1)
            for _ in range(10):
                view.last_obs_time += 0.25
                self.pipeline._maybe_start_parking_setup(view)
                if runner.loaded:
                    progressed = True
                    break
        self.assertTrue(progressed,
                        "identical replan 뒤 10 step 동안 아무 진행이 없다")

    def test_escalation_is_bounded_by_existing_budget(self) -> None:
        """max 3 유지 — 무한 escalation 금지."""
        view = self.pipeline.views[7]
        for _ in range(self.pipeline.config.max_parking_recovery_attempts):
            self.pipeline._parking_stage[1] = "PARKING"
            self.assertTrue(
                self.pipeline._escalate_repeated_replan(1, view, "X"))
        self.pipeline._parking_stage[1] = "PARKING"
        self.assertTrue(self.pipeline._escalate_repeated_replan(1, view, "X"))
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_RECOVERY_EXHAUSTED")
        self.assertIn(1, self.pipeline.server.zeroed)

    def test_non_rear_mode_ends_in_explicit_terminal_fault(self) -> None:
        """전역 모델에서는 escalation 대상이 아니므로 명시적 terminal."""
        self.pipeline.config.parking_mode = "handoff"
        self.pipeline._last_replan_signature[1] = self._signature()
        self.pipeline._replan_auto_host(1)
        runner = self.pipeline.auto_hosts[1]
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_REPEATED_REPLAN")
        self.assertTrue(runner.stopped)


class TestParkingProgressWatchdog(unittest.TestCase):
    """마지막 방어선: 아무도 몰지 않는 상태를 bounded time 안에 검출."""

    def setUp(self) -> None:
        base = TestRepeatedReplanLiveness()
        base.setUp()
        self.pipeline = base.pipeline
        self.pipeline.config.parking_stall_timeout_s = 8.0

    def _tick(self, view, seconds: float) -> None:
        view.last_obs_time += seconds
        self.pipeline._check_parking_progress(view)

    def test_silent_stall_is_detected_and_escalated(self) -> None:
        view = self.pipeline.views[7]
        self.pipeline._parking_stage[1] = "PARKING"
        self._tick(view, 0.25)
        self.assertEqual(self.pipeline._parking_stage[1], "PARKING")
        self._tick(view, 9.0)
        stalled = [n for n, _ in self.pipeline.dashboard.events
                   if n == "parking_stalled"]
        self.assertTrue(stalled)
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")

    def test_watchdog_never_commands_motion(self) -> None:
        view = self.pipeline.views[7]
        self.pipeline._parking_stage[1] = "PARKING"
        self._tick(view, 0.25)
        self._tick(view, 9.0)
        self.assertEqual(self.pipeline.auto_hosts[1].loaded, [])

    def test_moving_vehicle_is_not_a_stall(self) -> None:
        view = self.pipeline.views[7]
        self.pipeline._parking_stage[1] = "PARKING"
        self._tick(view, 0.25)
        view.recent.extend([(292.0, 571.0), (400.0, 600.0), (520.0, 650.0)])
        self._tick(view, 9.0)
        self.assertEqual(self.pipeline._parking_stage[1], "PARKING")

    def test_driving_vehicle_is_not_a_stall(self) -> None:
        view = self.pipeline.views[7]
        self.pipeline._parking_stage[1] = "PARKING"
        self.pipeline.auto_hosts[1].last_tick_result = SimpleNamespace(
            command=SimpleNamespace(throttle=-0.12))
        self._tick(view, 0.25)
        self._tick(view, 9.0)
        self.assertEqual(self.pipeline._parking_stage[1], "PARKING")

    def test_explicit_sensor_wait_is_not_a_stall(self) -> None:
        view = self.pipeline.views[7]
        for stage in ("SETUP_PENDING", "PARKING_HANDOFF_PENDING",
                      "WAIT_FRESH_HEADING_FAULT", "WAIT_SAFE_RECOVERY",
                      "WAIT_RECOVERY_EXHAUSTED"):
            with self.subTest(stage=stage):
                self.pipeline._parking_stage[1] = stage
                self.pipeline._stall_since.clear()
                self._tick(view, 0.25)
                self._tick(view, 9.0)
                self.assertEqual(self.pipeline._parking_stage[1], stage)

    def test_perception_loss_is_not_a_stall(self) -> None:
        view = self.pipeline.views[7]
        self.pipeline._parking_stage[1] = "PARKING"
        view.heading_source = "LAST_VALID"
        self._tick(view, 0.25)
        self._tick(view, 9.0)
        self.assertEqual(self.pipeline._parking_stage[1], "PARKING")




class TestNewWaitBoundariesEscalate(unittest.TestCase):
    """새로 도입한 heading 대기 경계도 bounded timeout 을 가져야 한다.

    등록되지 않은 경계는 timeout 이 없어 조용한 영구 zero 가 된다 — 지금까지
    두 번 겪은 dead state 와 정확히 같은 형태다.
    """

    def setUp(self) -> None:
        base = TestHeadingFaultLiveness()
        base.setUp()
        self.pipeline = base.pipeline

    def _starve(self, boundary):
        view = self.pipeline.views[7]
        view.heading_source = "LAST_VALID"
        self.pipeline._require_critical_heading(view, boundary)
        view.last_obs_time += 3.0
        self.pipeline._require_critical_heading(view, boundary)

    def test_every_new_wait_boundary_reaches_a_bounded_fault(self) -> None:
        for boundary in ("FINAL_POSE_EVAL", "BOUNDARY_HEADING_UNCERTAIN"):
            with self.subTest(boundary=boundary):
                base = TestHeadingFaultLiveness()
                base.setUp()
                self.pipeline = base.pipeline
                self._starve(boundary)
                self.assertEqual(self.pipeline._parking_stage[1],
                                 "WAIT_FRESH_HEADING_FAULT",
                                 f"{boundary}: timeout 이 없어 영구 대기한다")
                self.assertIn(1, self.pipeline._heading_fault_hold)

    def test_those_faults_recover_when_heading_returns(self) -> None:
        self._starve("BOUNDARY_HEADING_UNCERTAIN")
        view = self.pipeline.views[7]
        view.heading_source = "FRONT_CUSHION"
        view.last_obs_time += 0.25
        self.assertTrue(self.pipeline._maybe_resume_heading_fault(view))
        self.assertEqual(self.pipeline._parking_stage[1], "SETUP_PENDING")




class TestHeadingWaitTimerIsNotResetByCallers(unittest.TestCase):
    """run_20260827_234439: 두 호출부가 번갈아 물어 timeout 이 안 쌓였다.

    _maybe_start_parking_setup 은 PARKING_RECOVERY_REPLAN 으로,
    _check_boundary 는 BOUNDARY_HEADING_UNCERTAIN 으로 매 프레임 물었고,
    boundary 이름이 바뀔 때마다 대기 시계가 0 으로 돌아가 10.7초 동안
    WAIT_FRESH_HEADING_TIMEOUT 이 한 번도 발생하지 않았다.
    """

    def setUp(self) -> None:
        base = TestHeadingFaultLiveness()
        base.setUp()
        self.pipeline = base.pipeline

    def test_alternating_boundaries_still_time_out(self) -> None:
        view = self.pipeline.views[7]
        view.heading_source = "LAST_VALID"
        for i in range(14):
            view.last_obs_time += 0.25
            # 실차와 같이 한 프레임에 두 호출부가 서로 다른 이름으로 묻는다.
            self.pipeline._require_critical_heading(
                view, "PARKING_RECOVERY_REPLAN")
            self.pipeline._require_critical_heading(
                view, "BOUNDARY_HEADING_UNCERTAIN")
        self.assertEqual(self.pipeline._parking_stage[1],
                         "WAIT_FRESH_HEADING_FAULT",
                         "번갈아 묻는 동안 timeout 이 누적되지 않았다")
        self.assertIn(1, self.pipeline._heading_fault_hold)

    def test_timer_restarts_only_after_a_trusted_heading(self) -> None:
        view = self.pipeline.views[7]
        view.heading_source = "LAST_VALID"
        view.last_obs_time += 0.25
        self.pipeline._require_critical_heading(view, "PARKING_RECOVERY_REPLAN")
        first = self.pipeline._heading_wait_started[1]
        view.last_obs_time += 0.25
        self.pipeline._require_critical_heading(view, "BOUNDARY_HEADING_UNCERTAIN")
        self.assertEqual(self.pipeline._heading_wait_started[1], first)
        # trusted heading 이 돌아오면 대기 자체가 끝나고 시계도 사라진다.
        view.heading_source = "FRONT_CUSHION"
        view.last_obs_time += 0.25
        self.assertTrue(
            self.pipeline._require_critical_heading(view, "FINAL_POSE_EVAL"))
        self.assertNotIn(1, self.pipeline._heading_wait_started)


if __name__ == "__main__":
    unittest.main()
