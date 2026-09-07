"""POSE_STALE 복구는 **한 번만** 진행해야 한다 (run_20260903_154026).

실차 run_20260903_154026, route 3 / waypoint 2 (ALIGN, FORWARD):

    t=27.75~28.27  throttle 0.08 / PWM 25~26 / encoder_delta 66~95  정상 주행
    t=28.36        pose 가 (459.4,485.5) 에 고정, pose_age 125→625ms
    t=28.45        POSE_STALE → authority FAULTED → PWM 0, ESP READY
    t=28.77~       관측 복귀 (pose_age 78/16/46/31ms, 이후 중앙값 125ms)
    t=30.06~53.03  HEADING_RECOVERED 약 100회, stage SETUP_PENDING 232 tick,
                   allocation_state ROUTE_LOADED 불변, 새 route 0건,
                   motor_pwm_calc = {0}, 25초 뒤 수동 ABORT

관측은 300ms 만에 정상으로 돌아왔는데 25초 동안 다시 움직이지 못했다.
원인은 인지도 stiction 도 아니고 복구 계약의 **재진입**이다.

파이프라인은 관측마다 같은 view 로 두 함수를 연달아 부른다
(pipeline/runner.py 의 _on_view 순서):

    self._maybe_resume_heading_fault(view)   # _parking_setup_wait = last_obs_time
    self._maybe_start_parking_setup(view)    # last_obs_time <= wait_after → return

_stale_observation_latched 는 stage 가 SETUP_PENDING 이어도 참이고 authority
는 새 route 가 검증될 때까지 FAULTED 이므로, 첫 전이 뒤에도 관측마다 다시
들어와 방금 찍은 fresh-observation marker 를 현재 시각으로 다시 찍는다.
그래서 뒤따르는 setup 게이트가 그 경계를 **영원히** 통과하지 못한다.

이 파일은 실차와 같은 tick 순서로 재현한다. 기존 test_parking_liveness 가
잡지 못한 이유는 두 함수 사이에서 last_obs_time 을 수동으로 올려 부르기
때문이다 — 실차에는 그런 틈이 없다.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from controller.config import ControllerConfig
from parking.waypoints import InfeasibleRouteError
from pipeline.runner import ParkingPipeline, VehicleView


# ── 실차 계측값 ────────────────────────────────────────────────────────────
POSE_154026 = (459.4, 485.5, 328.7)     # POSE_STALE 직전 마지막 유효 관측
OBS_PERIOD_S = 0.125                    # 관측 간격 중앙값 (pose_age 중앙값과 동일)


class _Authority:
    def __init__(self) -> None:
        self.is_faulted = False
        self.fault_reason = ""
        self.re_armed = 0

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
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class _Runner:
    def __init__(self) -> None:
        self.loaded: list[list] = []
        self.replan_reason = "POSE_STALE"
        self.host = _Host()
        self.scheduler = _Scheduler()

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
        self.zeroed: list[int] = []

    def stop_control(self, car_id) -> None:
        self.zeroed.append(car_id)


def _wp(x, y, *, tolerance_cm=8.0, phase="RECOVERY"):
    return SimpleNamespace(x=x, y=y, phase=phase,
                           position_tolerance_cm=tolerance_cm,
                           motion_direction="REVERSE", route_id=88,
                           waypoint_id=1, curvature=0.0)


SETUP = [_wp(430.0, 520.0), _wp(360.0, 560.0)]


class PoseStaleRecoveryIsOneShot(unittest.TestCase):
    """POSE_STALE → STOP → fresh 관측 → 정지 확인 → bounded replan, 한 번만."""

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
        self.runner = _Runner()
        p.auto_hosts = {1: self.runner}
        p._auto_host_slot = {1: "B1"}
        p.track_of_car = {1: 7}
        view = VehicleView(track_id=7, car_id=1,
                           position_mm=POSE_154026[:2],
                           heading_deg=POSE_154026[2],
                           heading_source="FRONT_CUSHION")
        view.recent.extend([view.position_mm] * 3)
        view.last_obs_time = 28.45          # POSE_STALE 이 뜬 시각
        p.views = {7: view}
        self.view = view
        # 실차 그대로: 주차 진행 중(PARKING)에 controller 가 POSE_STALE 로 latch.
        p._parking_stage = {1: "PARKING"}
        self.runner.stop()
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
        p.orchestrator = SimpleNamespace(next_route_id=lambda: 88)
        p.on_route_load = None
        p._auto_host_route = {}
        p.on_event_record = None
        p.events: list[tuple] = []
        p._trajectory_safe = lambda view, route, **kw: True
        p._load_direct_rear_replan = lambda car_id, view: False
        p._build_route = lambda *_: (_ for _ in ()).throw(
            InfeasibleRouteError("B1", "single arc infeasible"))

    def _tick(self) -> None:
        """관측 하나 = 실차와 같은 호출 순서 (runner.py 의 두 줄)."""
        self.view.last_obs_time += OBS_PERIOD_S
        self.view.recent.append(self.view.position_mm)
        self.pipeline._maybe_resume_heading_fault(self.view)
        self.pipeline._maybe_start_parking_setup(self.view)

    # ── 전제 확인 ──────────────────────────────────────────────────────────

    def test_the_fault_really_is_an_observation_latch(self) -> None:
        """물리 안전 정지가 아니라 관측 지연 latch 여야 이 계약이 적용된다."""
        self.assertTrue(self.runner.host.authority.is_faulted)
        self.assertEqual(self.runner.host.authority.fault_reason, "POSE_STALE")
        self.assertTrue(self.pipeline._stale_observation_latched(1))

    # ── 핵심 회귀 ──────────────────────────────────────────────────────────

    def test_recovery_marker_is_not_restamped_every_observation(self) -> None:
        """복구 marker 가 관측마다 다시 찍히면 그 경계는 영원히 못 넘는다.

        이게 154026 의 25초 정지다. marker 는 첫 전이에서 한 번만 찍혀야
        하고, 그 뒤 관측은 marker 보다 뒤여야 한다.
        """
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            self._tick()
            marker = self.pipeline._parking_setup_wait.get(1)
            self.assertIsNotNone(marker, "첫 전이에서 marker 가 찍혀야 한다")
            self._tick()
            self.assertNotEqual(
                self.pipeline._parking_setup_wait.get(1), self.view.last_obs_time,
                "marker 가 현재 관측 시각으로 다시 찍혔다 — self-blocking loop")

    def test_setup_pending_is_left_within_a_few_observations(self) -> None:
        """LIVENESS: 관측이 정상인데 SETUP_PENDING 에 갇히면 안 된다.

        154026 은 여기서 232 tick(약 23초) 동안 나오지 못했다.
        """
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            for _ in range(8):
                self._tick()
                if self.pipeline._parking_stage.get(1) != "SETUP_PENDING":
                    break
        self.assertNotEqual(self.pipeline._parking_stage.get(1), "SETUP_PENDING")

    def test_a_new_route_is_loaded_and_authority_returns(self) -> None:
        """복구의 끝은 새 route 검증과 AUTO authority 복귀다."""
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            for _ in range(8):
                self._tick()
                if self.runner.loaded:
                    break
        self.assertEqual(self.runner.loaded[-1], SETUP)
        self.assertFalse(self.runner.host.authority.is_faulted)
        self.assertTrue(self.runner.scheduler.running)
        self.assertEqual(self.runner.host.authority.re_armed, 1)
        self.assertNotIn(1, self.pipeline._heading_fault_hold)

    def test_resume_contract_runs_once_not_once_per_observation(self) -> None:
        """HEADING_RECOVERED 가 관측마다 반복되면 계약이 재진입한 것이다.

        실차 로그에는 약 100회 찍혔다. 한 fault 당 한 번이어야 한다.
        """
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            for _ in range(8):
                self._tick()
        recovered = [e for e in self.pipeline.dashboard.events
                     if e[0] == "heading_recovered"]
        self.assertEqual(len(recovered), 1, f"{len(recovered)}회 재진입")

    # ── 안전 계약 (완화되지 않았는지) ─────────────────────────────────────

    def test_no_motion_before_a_validated_fresh_pose_route(self) -> None:
        """fresh pose 로 만든 route 가 검증되기 전에는 항상 zero 다.

        옛 route 재개도, stale pose 주행도 금지다.
        """
        self._tick()
        self.assertTrue(self.runner.host.authority.is_faulted)
        self.assertFalse(self.runner.scheduler.running)
        self.assertEqual(self.runner.loaded, [])

    def test_a_moving_car_still_waits_for_a_physical_stop(self) -> None:
        """정지 확인 단계는 유지된다 — 움직이는 중이면 route 를 싣지 않는다."""
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            self._tick()                      # 전이: PARKING → SETUP_PENDING
            for step in range(1, 5):          # 계속 움직이는 관측
                self.view.last_obs_time += OBS_PERIOD_S
                self.view.recent.append((POSE_154026[0] + 40.0 * step,
                                         POSE_154026[1]))
                self.view.position_mm = (POSE_154026[0] + 40.0 * step,
                                         POSE_154026[1])
                self.pipeline._maybe_resume_heading_fault(self.view)
                self.pipeline._maybe_start_parking_setup(self.view)
        self.assertEqual(self.runner.loaded, [])
        self.assertIn(1, self.pipeline.server.zeroed)

    def test_an_untrusted_heading_does_not_resume(self) -> None:
        """LAST_VALID heading 으로는 복구 계약을 시작하지 않는다."""
        self.view.heading_source = "LAST_VALID"
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            for _ in range(4):
                self._tick()
        self.assertEqual(self.pipeline._parking_stage.get(1), "PARKING")
        self.assertEqual(self.runner.loaded, [])

    def test_a_physical_safety_fault_is_never_resumed_here(self) -> None:
        """boundary/unsafe/COMM latch 는 이 경로로 풀리지 않는다."""
        self.runner.host.authority.fault("BOUNDARY_HARD")
        self.assertFalse(self.pipeline._stale_observation_latched(1))
        with patch("pipeline.runner.build_setup_recovery_waypoints",
                   return_value=SETUP):
            for _ in range(4):
                self._tick()
        self.assertEqual(self.pipeline._parking_stage.get(1), "PARKING")
        self.assertTrue(self.runner.host.authority.is_faulted)
        self.assertEqual(self.runner.loaded, [])


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()


class FinalAlignmentSurvivesACameraGap(unittest.TestCase):
    """최종 정렬 구간도 카메라 gap 에서 되살아나야 한다 (run_20260903_161534).

    161534 는 FINAL_ALIGNMENT 로 (482.1,1083.7,274.7) 에서 슬롯으로 되돌아오던
    중 POSE_STALE 을 두 번 맞았다. 그 stage 가 관측-latch 재활성 대상이 아니어서
    매번 8s PARKING_STALLED watchdog 까지 가야 풀렸고(t=49.86→60.72,
    t=64.03→75.59, 각 약 11초), 그때마다 recovery 예산을 하나씩 태워
    PARKING_RECOVERY_EXHAUSTED attempts=3 으로 끝났다. 그 사이 pose_age 중앙값은
    125ms 로 인지는 정상이었다.

    새 임계값이나 예산 확대가 아니라, 이미 있던 재활성 계약의 적용 범위를
    "주차가 명목상 진행 중인 stage" 로 맞추는 것이다.
    """

    # 161534 의 FINAL_ALIGNMENT 진행 중 자세 (슬롯 B1 안)
    POSE_161534_ALIGNING = (460.2, 977.8, 273.0)

    def _pipeline(self, stage: str):
        p = ParkingPipeline.__new__(ParkingPipeline)
        p.config = SimpleNamespace(
            parking_mode="rear", max_parking_recovery_attempts=3,
            max_replan_attempts=3, initial_pose_stability_mm=30.0,
            stationary_tolerance_mm=15.0, stationary_window=3,
            critical_heading_wait_timeout_s=2.5,
            parking_stall_timeout_s=8.0,
            controller_config=ControllerConfig())
        runner = _Runner()
        p.auto_hosts = {1: runner}
        p._auto_host_slot = {1: "B1"}
        p.track_of_car = {1: 7}
        view = VehicleView(track_id=7, car_id=1,
                           position_mm=self.POSE_161534_ALIGNING[:2],
                           heading_deg=self.POSE_161534_ALIGNING[2],
                           heading_source="FRONT_CUSHION")
        view.recent.extend([view.position_mm] * 3)
        view.last_obs_time = 49.86
        p.views = {7: view}
        p._parking_stage = {1: stage}
        runner.stop()                       # POSE_STALE latch
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
            InfeasibleRouteError("B1", "single arc infeasible"))
        return p, view, runner

    def test_final_alignment_is_an_observation_resumable_stage(self) -> None:
        """차를 실제로 모는 최종 정렬 구간이 재활성 대상이어야 한다."""
        for stage in ("FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE",
                      "FINAL_EVAL_PENDING"):
            with self.subTest(stage=stage):
                p, _view, _runner = self._pipeline(stage)
                self.assertTrue(p._stale_observation_latched(1))

    def test_a_camera_gap_in_final_alignment_recovers_without_the_watchdog(self):
        """관측이 돌아오면 8s watchdog 을 기다리지 않고 즉시 진행한다."""
        p, view, _runner = self._pipeline("FINAL_ALIGNMENT")
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        self.assertTrue(p._maybe_resume_heading_fault(view))
        self.assertNotEqual(p._parking_stage.get(1), "FINAL_ALIGNMENT")
        self.assertEqual(p._parking_recovery_attempts.get(1, 0), 0,
                         "복구 예산을 태우지 않아야 한다")

    def test_the_recovery_still_runs_only_once(self) -> None:
        """FINAL_* 로 넓혀도 self-blocking loop 가 되살아나면 안 된다."""
        p, view, _runner = self._pipeline("FINAL_ALIGNMENT")
        for _ in range(8):
            view.last_obs_time += OBS_PERIOD_S
            view.recent.append(view.position_mm)
            p._maybe_resume_heading_fault(view)
            p._maybe_start_parking_setup(view)
        recovered = [e for e in p.dashboard.events if e[0] == "heading_recovered"]
        self.assertEqual(len(recovered), 1, f"{len(recovered)}회 재진입")

    def test_no_motion_before_a_validated_route(self) -> None:
        p, view, runner = self._pipeline("FINAL_ALIGNMENT")
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        p._maybe_resume_heading_fault(view)
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertFalse(runner.scheduler.running)
        self.assertEqual(runner.loaded, [])

    # ── 넓히면 안 되는 stage ───────────────────────────────────────────────

    def test_confirmed_and_exhausted_stages_are_not_resumed(self) -> None:
        """확정된 주차와 예산이 끝난 terminal 은 되살리지 않는다.

        PARKED_VERIFY 는 정지한 채 관측을 세는 것이 정상 동작이고,
        WAIT_* 는 명시적으로 끝난 상태다.
        """
        for stage in ("PARKED_VERIFY", "PARKED", "WAIT_SAFE_RECOVERY",
                      "WAIT_RECOVERY_EXHAUSTED", "WAIT_ENTRY_STAGING_FAILED"):
            with self.subTest(stage=stage):
                p, _view, _runner = self._pipeline(stage)
                self.assertFalse(p._stale_observation_latched(1))

    def test_a_physical_fault_in_final_alignment_is_still_never_resumed(self):
        """BOUNDARY_HARD 같은 물리 fault 는 stage 와 무관하게 latch 유지."""
        p, view, runner = self._pipeline("FINAL_ALIGNMENT")
        runner.host.authority.fault("BOUNDARY_HARD")
        self.assertFalse(p._stale_observation_latched(1))
        view.last_obs_time += OBS_PERIOD_S
        view.recent.append(view.position_mm)
        self.assertFalse(p._maybe_resume_heading_fault(view))
        self.assertEqual(p._parking_stage.get(1), "FINAL_ALIGNMENT")
        self.assertEqual(runner.loaded, [])
