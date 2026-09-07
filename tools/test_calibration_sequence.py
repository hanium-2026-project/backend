"""calibration 시퀀스의 안전 경로를 실차 없이 전부 고정한다.

이 코드는 실제 actuator 를 움직인다. 그러므로 "실차 앞에서 처음 돌려보는
코드" 가 되면 안 된다. 경계·무동작·정지·통신·관측·중단 경로를 여기서 먼저
증명한다.

가장 중요한 불변식 하나: **어떤 경로로 끝나든 마지막 명령은 zero 다.**
"""

from __future__ import annotations

import math
import unittest

from tools.calibration_sequence import (CalibrationAborted, CalibrationLimits,
                                        CalibrationSequence, Primitive,
                                        TERM_BOUNDARY, TERM_COMM_FAIL,
                                        TERM_DISTANCE, TERM_DRIFT,
                                        TERM_DURATION, TERM_HEADING,
                                        TERM_NO_MOTION, TERM_POSE_STALE,
                                        build_plan)


class FakeVehicle:
    """명령을 받으면 단순 자전거 모델로 움직이는 가짜 차량.

    시간은 tick 단위로 흐른다 — 테스트가 실제로 기다리지 않는다.
    """

    def __init__(self, *, speed_per_throttle=800.0, radius_mm=800.0,
                 x=600.0, y=600.0, heading=0.0, deadband=0.09,
                 coast_mm=60.0, clearance=500.0):
        self.x, self.y, self.heading = x, y, heading
        self.speed_per_throttle = speed_per_throttle
        self.radius_mm = radius_mm
        self.deadband = deadband
        self.coast_mm = coast_mm
        self._clearance = clearance
        self.t = 0.0
        self.commands: list[tuple[float, float]] = []
        self.pose_available = True
        self.comm_up = True
        self._pending_coast = 0.0
        self._last_dir = 0.0

    # 콜백 표면
    def drive(self, throttle: float, steering: float) -> None:
        self.commands.append((round(throttle, 4), round(steering, 4)))
        self._apply(throttle, steering)

    def pose(self):
        if not self.pose_available:
            return None
        return (self.x, self.y, self.heading)

    def clearance(self):
        return self._clearance

    def comm_ok(self) -> bool:
        return self.comm_up

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds

    # 물리
    def _apply(self, throttle: float, steering: float) -> None:
        dt = 0.05
        if abs(throttle) <= self.deadband:
            # deadband: 구동 명령이 있어도 거의 안 움직인다.
            speed = 0.0
        else:
            speed = self.speed_per_throttle * throttle
        if speed == 0.0 and self._pending_coast > 0.0:
            step = min(self._pending_coast, 400.0 * dt)
            self._pending_coast -= step
            self._advance(step * self._last_dir, steering)
            return
        if speed != 0.0:
            self._last_dir = 1.0 if speed > 0 else -1.0
            self._pending_coast = self.coast_mm
            self._advance(speed * dt, steering)

    def _advance(self, distance: float, steering: float) -> None:
        if abs(steering) < 1e-6:
            rad = math.radians(self.heading)
            self.x += distance * math.cos(rad)
            self.y += distance * math.sin(rad)
            return
        radius = self.radius_mm / max(abs(steering), 1e-6)
        dtheta = distance / radius * (1.0 if steering > 0 else -1.0)
        self.heading = (self.heading + math.degrees(dtheta)) % 360.0
        rad = math.radians(self.heading)
        self.x += distance * math.cos(rad)
        self.y += distance * math.sin(rad)


def _seq(vehicle: FakeVehicle, **kw) -> CalibrationSequence:
    events: list[dict] = []
    seq = CalibrationSequence(
        drive=vehicle.drive, pose=vehicle.pose, clearance=vehicle.clearance,
        now=vehicle.now, sleep=vehicle.sleep, comm_ok=vehicle.comm_ok,
        emit=events.append, **kw)
    seq.events = events            # 테스트 편의
    return seq


STRAIGHT = Primitive(1, "SPEED", "STRAIGHT", "FORWARD", 0.15, 0.0)
ARC = Primitive(2, "STEERING", "ARC", "FORWARD", 0.0, 0.8)


class AlwaysEndsAtZero(unittest.TestCase):
    """§36-J: 어떤 종료 경로든 마지막 명령은 zero 다."""

    def _assert_last_zero(self, vehicle):
        self.assertEqual(vehicle.commands[-1], (0.0, 0.0))

    def test_normal_completion(self):
        v = FakeVehicle()
        _seq(v).run_primitive(STRAIGHT)
        self._assert_last_zero(v)

    def test_no_motion(self):
        v = FakeVehicle(deadband=0.5)          # 0.15 는 deadband 안
        _seq(v).run_primitive(STRAIGHT)
        self._assert_last_zero(v)

    def test_boundary(self):
        v = FakeVehicle(clearance=50.0)
        _seq(v).run_primitive(STRAIGHT)
        self._assert_last_zero(v)

    def test_pose_lost_midway(self):
        v = FakeVehicle()
        seq = _seq(v)
        original = v.pose
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            return original() if calls["n"] < 5 else None

        seq.pose = flaky
        seq.run_primitive(STRAIGHT)
        self._assert_last_zero(v)

    def test_comm_lost_midway(self):
        v = FakeVehicle()
        seq = _seq(v)
        ticks = {"n": 0}

        def comm():
            ticks["n"] += 1
            return ticks["n"] < 6

        seq.comm_ok = comm
        seq.run_primitive(STRAIGHT)
        self._assert_last_zero(v)


class Termination(unittest.TestCase):

    def test_straight_stops_on_target_distance(self):
        v = FakeVehicle(speed_per_throttle=1000.0)
        res = _seq(v).run_primitive(STRAIGHT)
        self.assertEqual(res.termination_reason, TERM_DISTANCE)
        self.assertGreaterEqual(res.displacement_mm, 150.0)
        self.assertTrue(res.motion_detected)

    def test_arc_stops_on_target_heading(self):
        v = FakeVehicle(speed_per_throttle=1000.0, radius_mm=600.0)
        res = _seq(v).run_primitive(ARC)
        self.assertEqual(res.termination_reason, TERM_HEADING)
        self.assertGreaterEqual(abs(res.heading_change_deg), 25.0)

    def test_slow_but_moving_stops_on_duration(self):
        """움직이긴 하는데 목표 거리에 못 미치면 시간으로 끝난다."""
        v = FakeVehicle(speed_per_throttle=200.0)      # 0.15 -> 30mm/s
        res = _seq(v).run_primitive(STRAIGHT)
        self.assertEqual(res.termination_reason, TERM_DURATION)
        self.assertTrue(res.motion_detected)

    def test_deadband_is_no_motion_not_duration(self):
        """§36-A: 0.10 이 안 움직이면 NO_MOTION 으로 분류된다."""
        v = FakeVehicle(deadband=0.12)
        prim = Primitive(1, "SPEED", "STRAIGHT", "FORWARD", 0.10, 0.0)
        res = _seq(v).run_primitive(prim)
        self.assertEqual(res.termination_reason, TERM_NO_MOTION)
        self.assertFalse(res.motion_detected)
        self.assertLess(res.displacement_mm, 15.0)

    def test_no_motion_ends_early_not_after_full_duration(self):
        """유예 시간만 쓰고 끝낸다 — 안 움직이는 차를 4초 밀지 않는다."""
        v = FakeVehicle(deadband=0.5)
        seq = _seq(v)
        res = seq.run_primitive(STRAIGHT)
        self.assertEqual(res.termination_reason, TERM_NO_MOTION)
        drive_s = (res.zero_t or 0.0) - (res.command_start_t or 0.0)
        self.assertLess(drive_s, seq.limits.straight_max_s)

    def test_boundary_precheck_sends_no_drive_command(self):
        """§36-D: 여유가 없으면 명령을 아예 보내지 않는다."""
        v = FakeVehicle(clearance=50.0)
        res = _seq(v).run_primitive(STRAIGHT)
        self.assertEqual(res.termination_reason, TERM_BOUNDARY)
        self.assertEqual({c[0] for c in v.commands}, {0.0})

    def test_pose_stale_precheck_drives_nothing(self):
        """§36-E: 구동 명령은 없고, 명시적 zero 만 남긴다."""
        v = FakeVehicle()
        v.pose_available = False
        res = _seq(v).run_primitive(STRAIGHT)
        self.assertEqual(res.termination_reason, TERM_POSE_STALE)
        self.assertEqual({c[0] for c in v.commands}, {0.0})
        self.assertEqual(v.commands[-1], (0.0, 0.0))

    def test_comm_fail_precheck_drives_nothing(self):
        """§36-F: 같은 계약."""
        v = FakeVehicle()
        v.comm_up = False
        res = _seq(v).run_primitive(STRAIGHT)
        self.assertEqual(res.termination_reason, TERM_COMM_FAIL)
        self.assertEqual({c[0] for c in v.commands}, {0.0})
        self.assertEqual(v.commands[-1], (0.0, 0.0))


class SafetyLimits(unittest.TestCase):

    def test_throttle_never_exceeds_the_cap(self):
        v = FakeVehicle()
        prim = Primitive(1, "SPEED", "STRAIGHT", "FORWARD", 0.90, 0.0)
        _seq(v).run_primitive(prim)
        self.assertLessEqual(max(abs(c[0]) for c in v.commands), 0.25)

    def test_reverse_is_negative_throttle(self):
        v = FakeVehicle()
        prim = Primitive(1, "SPEED", "STRAIGHT", "REVERSE", 0.15, 0.0)
        _seq(v).run_primitive(prim)
        self.assertTrue(any(c[0] < 0 for c in v.commands))

    def test_arc_uses_the_steering_throttle_not_the_primitive_throttle(self):
        v = FakeVehicle()
        seq = _seq(v, steering_throttle=0.20)
        seq.run_primitive(ARC)
        driving = {abs(c[0]) for c in v.commands if c[0] != 0.0}
        self.assertEqual(driving, {0.20})

    def test_user_abort_still_zeroes(self):
        """§36-G: Ctrl+C 여도 zero 를 보내고 나간다."""
        v = FakeVehicle()
        seq = _seq(v)
        n = {"i": 0, "raised": False}

        def boom(throttle, steering):
            n["i"] += 1
            if n["i"] > 4 and not n["raised"]:
                n["raised"] = True          # 실제 drive 는 계속 던지지 않는다
                raise KeyboardInterrupt
            v.drive(throttle, steering)

        seq.drive = boom
        with self.assertRaises(KeyboardInterrupt):
            seq.run_primitive(STRAIGHT)
        self.assertEqual(v.commands[-1], (0.0, 0.0))


class StoppingMeasurement(unittest.TestCase):

    def test_coasting_distance_is_measured(self):
        """§36-B: zero 이후 실제로 밀린 거리를 잰다."""
        v = FakeVehicle(speed_per_throttle=1000.0, coast_mm=60.0)
        res = _seq(v).run_primitive(STRAIGHT)
        self.assertGreater(res.coast_mm, 20.0)
        self.assertIsNotNone(res.zero_t)
        self.assertIsNotNone(res.physical_stop_t)
        self.assertGreaterEqual(res.physical_stop_t, res.zero_t)

    def test_a_car_that_never_moved_has_no_coast(self):
        v = FakeVehicle(deadband=0.5)
        res = _seq(v).run_primitive(STRAIGHT)
        self.assertLess(res.coast_mm, 15.0)


class PlanShape(unittest.TestCase):

    def test_speed_phase_precedes_steering_phase(self):
        plan = build_plan([0.10, 0.15], [0.4, -0.4])
        phases = [p.phase for p in plan]
        self.assertEqual(phases.index("STEERING"), phases.count("SPEED"))

    def test_every_condition_is_a_forward_reverse_pair(self):
        """§36-C: 되짚기 쌍 — 공간 절약 + 전후진 비대칭."""
        plan = build_plan([0.15], [0.8])
        for i in range(0, len(plan), 2):
            self.assertEqual(plan[i].direction, "FORWARD")
            self.assertEqual(plan[i + 1].direction, "REVERSE")
            self.assertEqual(plan[i].throttle, plan[i + 1].throttle)
            self.assertEqual(plan[i].steering, plan[i + 1].steering)

    def test_repeats_multiply_primitives(self):
        one = build_plan([0.15], [0.8], repeats=1)
        two = build_plan([0.15], [0.8], repeats=2)
        self.assertEqual(len(two), 2 * len(one))

    def test_primitive_ids_are_unique(self):
        plan = build_plan([0.10, 0.15, 0.25], [0.4, -0.4, 1.0])
        ids = [p.primitive_id for p in plan]
        self.assertEqual(len(ids), len(set(ids)))


class SequenceLevel(unittest.TestCase):

    def test_full_plan_emits_one_event_per_primitive(self):
        """§36-K: 각 primitive 가 metadata 로 분리되어 남는다."""
        v = FakeVehicle(speed_per_throttle=1000.0)
        seq = _seq(v)
        plan = build_plan([0.15], [0.8])
        results = seq.run(plan)
        self.assertEqual(len(results), len(plan))
        self.assertEqual(len(seq.events), len(plan))
        self.assertEqual([e["primitive_id"] for e in seq.events],
                         [p.primitive_id for p in plan])
        for e in seq.events:
            self.assertIn("termination_reason", e)
            self.assertIn("schema_version", e)

    def test_drift_guard_aborts_instead_of_continuing(self):
        """§36 누적 drift: 위험하게 원점 복귀를 시도하지 않는다.

        되짚기 쌍은 보통 출발점 근처로 돌아오므로, 실제 차처럼 전/후진이
        비대칭이어서 누적 이탈이 생기는 상황을 만든다.
        """
        v = FakeVehicle(speed_per_throttle=1000.0)
        original_apply = v._apply

        def asymmetric(throttle, steering):
            # 후진이 전진의 절반만 간다 -> 쌍마다 앞으로 누적 이탈한다
            original_apply(throttle * (0.5 if throttle < 0 else 1.0), steering)

        v._apply = asymmetric
        seq = _seq(v)
        seq.limits = CalibrationLimits(drift_limit_mm=5.0,
                                       drift_limit_deg=1.0)
        with self.assertRaises(CalibrationAborted) as ctx:
            seq.run(build_plan([0.15, 0.20], []))
        self.assertEqual(ctx.exception.reason, TERM_DRIFT)
        self.assertEqual(v.commands[-1], (0.0, 0.0))

    def test_comm_failure_stops_the_whole_session(self):
        """§36-F: 재접속 후 stale 상태로 이어서 하지 않는다."""
        v = FakeVehicle(speed_per_throttle=1000.0)
        seq = _seq(v)
        plan = build_plan([0.15, 0.20], [])
        n = {"i": 0}

        def comm():
            n["i"] += 1
            return n["i"] < 30

        seq.comm_ok = comm
        with self.assertRaises(CalibrationAborted) as ctx:
            seq.run(plan)
        self.assertEqual(ctx.exception.reason, TERM_COMM_FAIL)
        self.assertEqual(v.commands[-1], (0.0, 0.0))
        self.assertLess(len(seq.results), len(plan))

    def test_boundary_precheck_ends_the_session_not_each_primitive(self):
        """가장자리에 있으면 남은 primitive 를 하나씩 거절하지 않는다.

        차가 이미 벽에 붙어 있으면 이후 모든 primitive 가 BOUNDARY 로 거절돼
        세션이 통째로 낭비된다. 한 번만 명확히 알리고 끝낸다.
        """
        v = FakeVehicle(clearance=50.0)
        seq = _seq(v)
        plan = build_plan([0.15, 0.20, 0.25], [])
        with self.assertRaises(CalibrationAborted) as ctx:
            seq.run(plan)
        self.assertEqual(ctx.exception.reason, TERM_BOUNDARY)
        self.assertEqual(len(seq.results), 1)          # 첫 거절에서 멈춘다
        self.assertEqual(v.commands[-1], (0.0, 0.0))

    def test_boundary_hit_while_driving_does_not_end_the_session(self):
        """주행 중 경계는 그 primitive 만 끝낸다 — 데이터는 유효하다."""
        v = FakeVehicle(speed_per_throttle=1000.0)
        seq = _seq(v)
        state = {"n": 0}

        def shrinking():
            state["n"] += 1
            return 500.0 if state["n"] < 8 else 50.0

        seq.clearance = shrinking
        first = seq.run_primitive(Primitive(1, "SPEED", "STRAIGHT",
                                            "FORWARD", 0.15, 0.0))
        self.assertEqual(first.termination_reason, TERM_BOUNDARY)
        self.assertIsNotNone(first.command_start_t)     # 실제로 주행은 했다

    def test_session_refuses_to_start_without_pose(self):
        """§36-I: 세션/관측이 없으면 아무것도 구동하지 않는다."""
        v = FakeVehicle()
        v.pose_available = False
        seq = _seq(v)
        with self.assertRaises(CalibrationAborted):
            seq.run(build_plan([0.15], []))
        self.assertEqual({c[0] for c in v.commands} or {0.0}, {0.0})


if __name__ == "__main__":
    unittest.main()
