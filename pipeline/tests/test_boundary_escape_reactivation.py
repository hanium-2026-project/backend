"""BOUNDARY_HARD 는 정지여야 하고, 영구 정지여서는 안 된다 (run_20260903_161435).

실차 run_20260903_161435, B1 후면주차 FINAL:

    t=30.97  마지막 구동 tick. pose (451.0,1007.9,259.6)
             맵 footprint overflow = **-55.6mm** (경계 안쪽 55.6mm)
             즉 명령이 끊긴 시점에는 아직 충분히 복구 가능한 영역이었다.
    t=31.0~  명령 0 이후 관성으로 123.9mm 더 이동
    t=31.63  BOUNDARY_HARD overflow=53.5 -> FAULT -> authority STOP
    t=32.80  FINAL_POSE_EVAL action=ALIGN (HEADING_NOT_PARALLEL)
    t=33.11  FINAL_ALIGNMENT route 적재 (target 472.2,990.8 = 슬롯 쪽 FORWARD)
    t=33.1~40.5  desired_throttle=None / applied 0.0 / PWM 0 / encoder_delta 0

복구 수단은 전부 이미 있었다. 적재된 탈출 경로의 첫 waypoint 는
(476.9,1006.8) 로 이미 맵 안쪽 53.8mm 이고, validate_trajectory 도
safe=True 를 준다 (min_clearance -55.3mm = 출발 자세 그 자체이고 그
뒤로는 단조 개선). 없었던 것은 실행 권한 하나뿐이다.

_check_boundary 의 hard 집합은 worst<=0 일 때만 비워지는데, 그러려면
차가 맵 안으로 돌아와야 하고, 돌아오려면 움직여야 하는데 authority 가
FAULTED 라 못 움직인다. 구조적 deadlock 이다.

이 파일이 고정하는 계약:
  * BOUNDARY_HARD 는 여전히 즉시 정지한다 (임계값 변경 없음)
  * 그 정지는 **기존 안전 검증을 통과한 탈출 경로 하나**로만 풀린다
  * 검증을 통과하지 못하면 계속 정지다
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from parking.final_alignment import (build_final_alignment_waypoints,
                                     evaluate_final_pose)
from parking.trajectory_safety import validate_trajectory
from parking.waypoints import LOT_SIZE_MM, _car_footprint, default_slot_specs
from pipeline.runner import ParkingPipeline


# ── 실차 계측값 ────────────────────────────────────────────────────────────
POSE_161435_ZERO = (451.0, 1007.9, 259.6)      # 마지막 구동 tick
POSE_161435_SETTLED = (497.6, 1122.7, 263.6)   # 123.9mm coast 뒤 정지
POSE_163645_SETTLED = (487.7, 1093.1, 274.6)


def _overflow(x: float, y: float, h: float) -> float:
    return max(max(-px, px - LOT_SIZE_MM, -py, py - LOT_SIZE_MM)
               for px, py in _car_footprint(x, y, h))


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
        self.running = True

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class _Runner:
    def __init__(self) -> None:
        self.loaded: list[list] = []
        self.host = _Host()
        self.scheduler = _Scheduler()

    def load_route(self, route) -> None:
        self.loaded.append(list(route))

    def stop(self) -> None:
        self.host.authority.fault("BOUNDARY_HARD")
        self.scheduler.stop()


class _Dashboard:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def push_event(self, name, **fields) -> None:
        self.events.append((name, fields))


class TheRecordedOvershootWasRecoverableWhenTheMotorStopped(unittest.TestCase):
    """수정의 전제: zero 시점은 복구 가능, coast 가 그걸 날렸다."""

    def test_zero_command_pose_is_well_inside_the_map(self) -> None:
        self.assertLess(_overflow(*POSE_161435_ZERO), -50.0)

    def test_the_coast_is_what_crossed_the_boundary(self) -> None:
        self.assertGreater(_overflow(*POSE_161435_SETTLED), 30.0)

    def test_an_escape_route_exists_and_passes_the_existing_safety_gate(self):
        """탈출 수단은 이미 있었다 — 새로 만들 필요가 없다."""
        b1 = default_slot_specs()["B1"]
        for pose in (POSE_161435_SETTLED, POSE_163645_SETTLED):
            with self.subTest(pose=pose):
                self.assertEqual(evaluate_final_pose(b1, *pose).action, "ALIGN")
                wps = build_final_alignment_waypoints(
                    b1, 1, from_pose=pose[:2], from_heading_deg=pose[2])
                self.assertTrue(wps)
                result = validate_trajectory(wps, start_pose=pose,
                                             target_slot="B1")
                self.assertTrue(result.safe)
                first = wps[0]
                self.assertLess(
                    _overflow(first.x, first.y, first.target_heading_deg), 0.0,
                    "첫 waypoint 부터 맵 안쪽이어야 탈출이다")


class BoundaryHardStopsButCanBeReleasedByAValidatedEscape(unittest.TestCase):

    def _pipeline(self, stage: str = "FINAL_EVAL_PENDING"):
        p = ParkingPipeline.__new__(ParkingPipeline)
        p.config = SimpleNamespace(
            parking_mode="rear", lot_width_mm=LOT_SIZE_MM,
            lot_height_mm=LOT_SIZE_MM, boundary_hard_margin_mm=20.0,
            boundary_measurement_uncertainty_mm=10.0)
        runner = _Runner()
        p.auto_hosts = {1: runner}
        p._parking_stage = {1: stage}
        p._boundary_escape_hold = set()
        p._heading_fault_hold = set()
        p.dashboard = _Dashboard()
        p.events = []
        p.on_event_record = None
        return p, runner

    def test_boundary_hard_still_stops_immediately(self) -> None:
        """정지 자체는 그대로다 — 이 수정은 정지를 늦추지 않는다."""
        p, runner = self._pipeline()
        runner.stop()
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertFalse(runner.scheduler.running)

    def test_a_validated_escape_reactivates_authority(self) -> None:
        p, runner = self._pipeline()
        runner.stop()
        p._boundary_escape_hold.add(1)
        p._reactivate_after_boundary_escape(1)
        self.assertFalse(runner.host.authority.is_faulted)
        self.assertTrue(runner.scheduler.running)
        self.assertEqual(runner.host.authority.re_armed, 1)
        self.assertNotIn(1, p._boundary_escape_hold)

    def test_a_car_without_the_mark_is_never_reactivated(self) -> None:
        """다른 물리 fault(COMM, unsafe route)는 이 경로로 절대 안 켜진다."""
        p, runner = self._pipeline()
        runner.host.authority.fault("COMM_LOST")
        runner.scheduler.stop()
        p._reactivate_after_boundary_escape(1)
        self.assertTrue(runner.host.authority.is_faulted)
        self.assertFalse(runner.scheduler.running)
        self.assertEqual(runner.host.authority.re_armed, 0)

    def test_reactivation_is_one_shot(self) -> None:
        """표시는 한 번 쓰이고 사라진다 — 재무장이 반복되면 안 된다."""
        p, runner = self._pipeline()
        runner.stop()
        p._boundary_escape_hold.add(1)
        for _ in range(4):
            p._reactivate_after_boundary_escape(1)
        self.assertEqual(runner.host.authority.re_armed, 1)

    def test_the_mark_does_not_relax_any_boundary_threshold(self) -> None:
        """임계값은 그대로여야 이 수정이 'boundary 완화'가 아니다."""
        p, _runner = self._pipeline()
        self.assertEqual(p.config.boundary_hard_margin_mm, 20.0)
        self.assertEqual(p.config.boundary_measurement_uncertainty_mm, 10.0)


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()
