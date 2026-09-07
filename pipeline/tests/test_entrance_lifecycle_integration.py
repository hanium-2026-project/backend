"""프레임 루프를 실제로 돌리는 lifecycle 통합 테스트.

세 번 연속으로 "unit PASS / 실차 FAIL" 이 났다. 원인은 언제나 같았다:
테스트가 handler 를 **하나씩 직접** 부르면서 그 handler 의 전제조건을
미리 만족시켜 놓았기 때문이다. 실차에는 그런 틈이 없다.

가장 비싼 사례가 stale 복구다. 테스트는 view.recent 에 같은 좌표를 미리
채워 첫 tick 부터 정지 상태로 만들었다. 실차에서는 POSE_STALE 이 **주행
중에** 걸리고, zero 이후 차가 관성으로 미끄러지므로 첫 tick 은 반드시
움직이는 상태다:

    run_20260904_214600  프레임간 최대 38.6mm / 표류 54.6mm
    run_20260904_214835                 18.2mm /       50.1mm
    run_20260904_214712                 16.5mm /       24.7mm
                                        (stationary_tolerance_mm = 15)

그래서 hold 표시만 찍히고 계약은 완료되지 않은 채 43~74초 정지했다.

이 파일은 production 프레임 진입점 `_feed_auto_host` 를 그대로 여러 tick
돌린다. handler 순서, 조기 return, 마커 재각인까지 실제와 같은 경로를 탄다.
**첫 복구 tick 은 반드시 움직이는 상태여야 한다** — 미리 정지 이력을 넣어
통과시키지 않는다.

TARGET_SLOT 은 B1 고정이 아니라 여러 슬롯으로 parameterize 한다.
"""

from __future__ import annotations

import math
import time
import unittest
from types import SimpleNamespace

from controller.config import ControllerConfig
from parking.waypoints import InfeasibleRouteError, default_slot_specs
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView
from rl.parking_env import SLOT_NAMES

# 실차 214600 의 staging 기동 초반 궤적 (주행 -> POSE_STALE -> 관성 표류)
DRIVE_TRACK = [
    (146.1, 184.1, 95.2), (150.0, 185.4, 95.9), (156.3, 271.2, 83.3),
    (161.4, 309.5, 77.5), (162.6, 325.3, 75.5), (163.9, 329.2, 76.1),
    (169.1, 327.9, 77.6), (171.8, 327.9, 78.2), (171.8, 326.6, 79.0),
    (173.1, 325.3, 79.7), (174.4, 322.7, 80.0),
]
# 표류가 끝난 뒤의 정지 구간
SETTLED = (174.4, 322.7, 80.0)


class _Authority:
    def __init__(self) -> None:
        self.is_faulted = False
        self.fault_reason = ""

    def fault(self, reason: str = "STOP") -> None:
        self.is_faulted, self.fault_reason = True, reason


class _Host:
    def __init__(self) -> None:
        self.authority = _Authority()
        self.re_arms = 0

    def re_arm_auto(self) -> None:
        self.authority.is_faulted = False
        self.authority.fault_reason = ""
        self.re_arms += 1


class _Scheduler:
    def __init__(self) -> None:
        self.running = True

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class _Runner:
    """AutoHostRunner 대역. pose 를 받고 route 를 싣는 최소 계약만 가진다."""

    def __init__(self) -> None:
        self.loaded: list[list] = []
        self.host = _Host()
        self.scheduler = _Scheduler()
        self.poses: list[tuple] = []
        self.replan_reason = None
        self.stopped = False
        self.current_target = None
        self.status = None

    def on_camera_pose(self, x, y, heading, obs_time, source) -> None:
        self.poses.append((x, y, heading, obs_time, source))

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        self.stopped = True
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

    def hold_control(self, car_id) -> None:
        self.zeroed.append(car_id)


class _Lifecycle(unittest.TestCase):
    """`_feed_auto_host` 를 실제로 돌리는 공용 하니스."""

    OBS_DT = 0.25

    def build(self, target_slot: str, stage: str = "ENTRY_STAGING"):
        pipe = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                              parking_mode="rear"))
        runner = _Runner()
        pipe.auto_hosts = {1: runner}
        pipe.track_of_car = {1: 2}
        pipe.hybrid_controls = {}
        pipe.dashboard = _Dashboard()
        pipe.server = _Server()
        pipe.events = []
        pipe.on_event_record = lambda n, **f: pipe.events.append((n, f))
        view = VehicleView(track_id=2, car_id=1, node="entrance",
                           position_mm=DRIVE_TRACK[0][:2],
                           heading_deg=DRIVE_TRACK[0][2],
                           heading_source="FRONT_CUSHION", last_obs_time=0.0)
        pipe.views = {2: view}
        pipe.allocator.update(2, view.position_mm)
        pipe.allocator.reassign(2, target_slot)
        pipe._auto_host_slot[1] = target_slot
        pipe._parking_stage[1] = stage
        pipe._entry_staging_attempts[1] = 1
        self.pipe, self.view, self.runner = pipe, view, runner
        self.target = target_slot
        self.t = 0.0
        return pipe, view, runner

    def tick(self, pose) -> None:
        """카메라 관측 1건을 production 프레임 경로로 흘린다."""
        self.t += self.OBS_DT
        self.view.position_mm = (pose[0], pose[1])
        self.view.heading_deg = pose[2]
        self.view.heading_source = "FRONT_CUSHION"
        self.view.last_obs_time = self.t
        self.view.recent.append(self.view.position_mm)
        self.pipe._feed_auto_host(self.view)

    def drive_then_stall(self, fault: str = "POSE_STALE") -> None:
        """주행 -> POSE_STALE latch -> **아직 움직이는** 첫 복구 tick 들.

        DRIVE_TRACK[3:5] 는 프레임간 38.6mm / 15.8mm 로 둘 다
        stationary_tolerance_mm(15) 를 넘는다 — 실차 214600 의 표류 그대로다.
        여기서 복구가 완료되면 안 된다.
        """
        for pose in DRIVE_TRACK[:3]:
            self.tick(pose)
        # 제어기가 stale 로 zero 를 걸고 authority 가 잠긴다 (safety 그대로).
        if fault == "POSE_STALE":
            self.runner.stop()
        else:
            self.runner.host.authority.fault(fault)
            self.runner.scheduler.stop()
        for pose in DRIVE_TRACK[3:5]:
            self.tick(pose)

    def settle(self, ticks: int = 20) -> None:
        """표류가 멎은 뒤의 정지 관측들.

        배경 staging 계획이 프레임 사이에 끝나야 적재까지 이어지므로,
        계획 중이면 잠깐 양보하며 tick 을 이어간다 (프로덕션의 다음 프레임).
        """
        for _ in range(ticks):
            self.tick(SETTLED)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with self.pipe._lock:
                    busy = 1 in self.pipe._entry_staging_planning
                if not busy:
                    break
                time.sleep(0.005)

    def slot_index(self, slot: str) -> int:
        return SLOT_NAMES.index(slot)


# ══ 1. 주행 중 POSE_STALE -> 정지 후 복구 ═══════════════════════════════════

class MovingFirstTickStaleRecovery(_Lifecycle):

    def test_the_first_recovery_tick_really_is_moving(self) -> None:
        """전제 검증 — 이 조건이 깨지면 이 테스트는 의미가 없다."""
        moved = [math.hypot(b[0] - a[0], b[1] - a[1])
                 for a, b in zip(DRIVE_TRACK[2:5], DRIVE_TRACK[3:5])]
        tolerance = PipelineConfig().stationary_tolerance_mm
        self.assertTrue(all(m > tolerance for m in moved),
                        f"첫 복구 tick 들이 정지 상태다: {moved}")

    def test_recovery_survives_a_moving_first_tick(self) -> None:
        for slot in ("B1", "A2", "B3"):
            with self.subTest(slot=slot):
                self.build(slot)
                self.drive_then_stall()
                # 아직 움직이는 동안에는 되살아나지 않는다 (계약 유지).
                self.assertEqual(self.pipe._parking_stage[1], "ENTRY_STAGING")
                self.assertTrue(self.runner.host.authority.is_faulted)
                self.assertEqual(self.runner.loaded, [])
                # 표류가 멎으면 되살아난다. 완주하면 stage 는 다시
                # ENTRY_STAGING 이 되므로(새 route 적재) stage 가 아니라
                # **복구가 실제로 일어났는지**를 본다.
                self.settle()
                self.assertIn("HEADING_RECOVERED",
                              [n for n, _ in self.pipe.events],
                              "관측이 돌아왔는데 복구 계약이 돌지 않았다")
                self.assertFalse(self.runner.host.authority.is_faulted,
                                 "새 route 검증 후 authority 가 재무장돼야 한다")
                self.assertTrue(self.runner.scheduler.running)
                self.assertTrue(self.runner.loaded,
                                "복구는 새 route 적재로 끝난다")
                self.assertEqual(self.pipe._auto_host_slot[1], slot)

    def test_the_hold_is_not_stamped_before_the_preconditions(self) -> None:
        """이 회귀가 43~74초 정지의 직접 원인이었다.

        표시가 전제조건 통과 **전에** 찍히면, 함수 입구의 재진입 금지가
        그 차량을 자기 복구 계약에서 영구히 배제한다.
        """
        self.build("B1")
        self.drive_then_stall()
        self.assertNotIn(1, self.pipe._heading_fault_hold,
                         "전제조건 통과 전에 표시가 찍히면 재진입이 막힌다")
        # 표시가 없으므로 다음 tick 들에서 계속 재시도할 수 있다.
        self.settle()
        self.assertIn("HEADING_RECOVERED",
                      [n for n, _ in self.pipe.events])

    def test_the_contract_still_runs_exactly_once(self) -> None:
        self.build("B1")
        self.drive_then_stall()
        self.settle(ticks=12)
        recovered = [e for e in self.pipe.dashboard.events
                     if e[0] == "heading_recovered"]
        self.assertEqual(len(recovered), 1, f"{len(recovered)}회 재진입")

    def test_no_motion_while_still_drifting(self) -> None:
        self.build("B1")
        self.drive_then_stall()
        self.assertEqual(self.runner.loaded, [])
        self.assertFalse(self.runner.scheduler.running)

    def test_zero_safety_is_unchanged(self) -> None:
        cfg = ControllerConfig()
        self.assertEqual(cfg.max_pose_age_s, 0.5)
        self.assertEqual(PipelineConfig().stationary_tolerance_mm, 15.0)


class GlobalStaleUsesTheSameRootFix(_Lifecycle):

    def test_global_recovers_after_a_moving_first_tick(self) -> None:
        for slot in ("B1", "A3"):
            with self.subTest(slot=slot):
                self.build(slot, stage="GLOBAL")
                self.drive_then_stall()
                self.assertEqual(self.pipe._parking_stage[1], "GLOBAL",
                                 "움직이는 동안에는 되살아나지 않는다")
                self.settle()
                self.assertIn("HEADING_RECOVERED",
                              [n for n, _ in self.pipe.events])
                self.assertEqual(self.pipe._auto_host_slot[1], slot)


class PhysicalFaultsAreNeverResumed(_Lifecycle):

    def test_physical_and_comm_faults_stay_latched(self) -> None:
        for reason in ("BOUNDARY_HARD", "COMM_TIMEOUT",
                       "SOCKET_DISCONNECTED", "UNSAFE_ROUTE"):
            with self.subTest(reason=reason):
                self.build("B1")
                self.drive_then_stall(fault=reason)
                self.settle()
                self.assertEqual(self.pipe._parking_stage[1], "ENTRY_STAGING")
                self.assertTrue(self.runner.host.authority.is_faulted)
                self.assertEqual(self.runner.loaded, [])


# ══ 2. 예산 소진 = SAFE STOP + 예약 유지 ════════════════════════════════════

class ExhaustedStagingKeepsTheReservation(_Lifecycle):

    def _exhaust(self, slot: str):
        pipe, view, runner = self.build(slot, stage="ENTRY_STAGING_PENDING")
        pipe._entry_staging_attempts[1] = \
            PipelineConfig().max_entry_staging_attempts
        # 예약 슬롯으로는 지금 자세에서 경로가 안 나온다.
        pipe._build_route = lambda *_: (_ for _ in ()).throw(
            InfeasibleRouteError(slot, "no route from this pose"))
        pipe._build_entry_staging_route = (
            lambda car, v, s, rid, goal_test=None: [])
        before = list(pipe.allocator.slot_statuses)
        self.settle(ticks=10)
        return pipe, runner, before

    def test_exhaustion_never_selects_another_slot(self) -> None:
        for slot in ("B1", "A2", "B3"):
            with self.subTest(slot=slot):
                pipe, _runner, before = self._exhaust(slot)
                picked = {f.get("slot") for n, f in pipe.events
                          if n in ("SLOT_SELECTED", "ENTRY_STAGING_COMPLETE",
                                   "SLOT_CANDIDATE")}
                self.assertFalse(picked - {slot},
                                 f"예약 {slot} 외 슬롯이 등장했다: {picked}")
                self.assertNotIn("SLOT_REJECTED",
                                 [n for n, _ in pipe.events])

    def test_exhaustion_is_a_safe_stop_with_the_slot_reserved(self) -> None:
        for slot in ("B1", "A2", "B3"):
            with self.subTest(slot=slot):
                pipe, runner, before = self._exhaust(slot)
                self.assertEqual(pipe._auto_host_slot[1], slot)
                self.assertEqual(pipe.allocator.vehicles[2].assigned_slot, slot)
                self.assertEqual(before, list(pipe.allocator.slot_statuses),
                                 "예약/점유 장부가 바뀌면 안 된다")
                self.assertGreaterEqual(
                    pipe.allocator.slot_statuses[self.slot_index(slot)], 0.5)
                self.assertTrue(runner.stopped or pipe.server.zeroed)

    def test_no_allocator_reassign_on_route_failure(self) -> None:
        for slot in ("B1", "A2"):
            with self.subTest(slot=slot):
                pipe, _runner, _before = self.build(
                    slot, stage="ENTRY_STAGING_PENDING"), None, None
                pipe = self.pipe
                calls: list = []
                pipe.allocator.reassign = lambda *a, **k: calls.append(a)
                pipe._entry_staging_attempts[1] = \
                    PipelineConfig().max_entry_staging_attempts
                pipe._build_route = lambda *_: (_ for _ in ()).throw(
                    InfeasibleRouteError(slot, "no route"))
                pipe._build_entry_staging_route = (
                    lambda car, v, s, rid, goal_test=None: [])
                self.settle(ticks=10)
                self.assertEqual(calls, [],
                                 "주행 실패로 allocator.reassign 이 불려서는 안 된다")


# ══ 3. 임의 슬롯 일반성 ═════════════════════════════════════════════════════

class ContractIsSlotGeneric(_Lifecycle):

    def test_every_slot_behaves_the_same(self) -> None:
        for slot in sorted(default_slot_specs()):
            with self.subTest(slot=slot):
                self.build(slot)
                self.assertEqual(self.pipe._auto_host_slot[1], slot)
                self.drive_then_stall()
                self.settle()
                self.assertEqual(self.pipe._auto_host_slot[1], slot)
                self.assertEqual(
                    self.pipe.allocator.vehicles[2].assigned_slot, slot)

    def test_production_logic_has_no_slot_literals(self) -> None:
        """B1 오버피팅 방지 — 프로덕션 코드에 슬롯 리터럴이 없어야 한다."""
        import re
        for path in ("pipeline/runner.py", "parking/waypoints.py"):
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    code = line.split("#", 1)[0]
                    if '"""' in code or "'''" in code:
                        continue
                    hit = re.search(r'["\'][AB][1-4]["\']', code)
                    self.assertIsNone(
                        hit, f"{path}:{number} 에 슬롯 리터럴: {code.strip()}")


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()
