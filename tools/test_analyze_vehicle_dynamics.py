"""analyze_vehicle_dynamics 추정기 자체가 틀리지 않는다는 것을 고정한다.

이 도구는 실차 튜닝의 근거를 만든다. 도구가 조용히 틀리면 **실차가 아니라
분석이 만든 오차**를 차량 모델이라고 믿게 된다 — 실제로 2026-09-03 audit 에서
연속 pose 3점 원 맞춤으로 "계획 대비 오차 64%" 라는 잘못된 결론이 나왔다.

그래서 반경/속도/정지거리를 **알고 있는 합성 데이터**로 먼저 검증한다.
"""

from __future__ import annotations

import math
import os
import random
import unittest

from tools.analyze_vehicle_dynamics import (STOP_SETTLE_S, TRUSTED_HEADING,
                                            analyze, arc_samples, confidence,
                                            deadband_samples, speed_samples,
                                            stop_samples)


def _arc_rows(radius_mm: float, *, steering: float, direction: str = "FORWARD",
              turn_deg: float = 60.0, dt: float = 0.23, noise_mm: float = 0.0,
              seed: int = 7, start=(600.0, 600.0), start_heading: float = 0.0):
    """알려진 반경의 원호를 4fps 관측으로 흉내낸다."""
    rng = random.Random(seed)
    rows = []
    steps = max(2, int(abs(turn_deg) / 6.0))
    sign = 1.0 if turn_deg >= 0 else -1.0
    cx = start[0] - radius_mm * math.sin(math.radians(start_heading)) * sign
    cy = start[1] + radius_mm * math.cos(math.radians(start_heading)) * sign
    for i in range(steps + 1):
        theta = math.radians(turn_deg) * i / steps
        h = start_heading + math.degrees(theta)
        px = cx + radius_mm * math.sin(math.radians(h)) * sign
        py = cy - radius_mm * math.cos(math.radians(h)) * sign
        rows.append({
            "t_s": i * dt,
            "pose_x_mm": px + rng.uniform(-noise_mm, noise_mm),
            "pose_y_mm": py + rng.uniform(-noise_mm, noise_mm),
            "pose_heading_deg": h % 360.0,
            "pose_heading_source": "FRONT_CUSHION",
            "wire_steering": steering,
            "throttle_cmd": 0.2,
            "motion_direction": direction,
            "phase": "ENTRY",
            "curvature": 1.0 / radius_mm,
        })
    return rows


def _straight_rows(speed_mm_s: float, *, dt: float = 0.23, n: int = 8,
                   throttle: float = 0.2, direction: str = "FORWARD"):
    return [{
        "t_s": i * dt,
        "pose_x_mm": 300.0 + speed_mm_s * i * dt,
        "pose_y_mm": 600.0,
        "pose_heading_deg": 0.0,
        "pose_heading_source": "FRONT_CUSHION",
        "wire_steering": 0.0,
        "throttle_cmd": throttle,
        "motion_direction": direction,
        "phase": "CRUISE",
        "curvature": 0.0,
    } for i in range(n)]


class SyntheticArcRecovery(unittest.TestCase):
    """알려진 반경을 되찾아내는가."""

    def test_clean_arcs_are_recovered_within_5_percent(self):
        for radius in (610.0, 800.0, 1000.0, 1100.0):
            with self.subTest(radius=radius):
                got = arc_samples("run_x", _arc_rows(radius, steering=0.8))
                self.assertEqual(len(got), 1)
                self.assertAlmostEqual(got[0].radius_mm, radius,
                                       delta=radius * 0.05)

    def test_noisy_arcs_stay_within_15_percent(self):
        """4fps 카메라 잡음(±5mm)에서도 결론이 뒤집히지 않아야 한다."""
        for radius in (610.0, 1000.0):
            with self.subTest(radius=radius):
                got = arc_samples("run_x", _arc_rows(radius, steering=0.8,
                                                     noise_mm=5.0))
                self.assertEqual(len(got), 1)
                self.assertAlmostEqual(got[0].radius_mm, radius,
                                       delta=radius * 0.15)

    def test_left_and_right_are_not_confused(self):
        left = arc_samples("r", _arc_rows(800.0, steering=-0.8, turn_deg=-60.0))
        right = arc_samples("r", _arc_rows(800.0, steering=0.8, turn_deg=60.0))
        self.assertLess(left[0].heading_change_deg, 0.0)
        self.assertGreater(right[0].heading_change_deg, 0.0)
        self.assertLess(left[0].steering, 0.0)
        self.assertGreater(right[0].steering, 0.0)

    def test_reverse_direction_is_kept(self):
        got = arc_samples("r", _arc_rows(800.0, steering=0.8,
                                         direction="REVERSE"))
        self.assertEqual(got[0].direction, "REVERSE")

    def test_planned_radius_is_carried_through(self):
        got = arc_samples("r", _arc_rows(900.0, steering=0.7))
        self.assertAlmostEqual(got[0].planned_radius_mm, 900.0, places=6)


class SamplingRules(unittest.TestCase):
    """잘못된 표본을 걸러내는 규칙 — 여기가 무너지면 도구가 거짓말을 한다."""

    def test_changing_steering_is_split_not_merged(self):
        """조향이 바뀌면 한 원호로 묶지 않는다 (예전 64% artifact 의 원인)."""
        rows = _arc_rows(610.0, steering=1.0, turn_deg=40.0)
        tail = _arc_rows(1100.0, steering=0.3, turn_deg=40.0,
                         start=(rows[-1]["pose_x_mm"], rows[-1]["pose_y_mm"]),
                         start_heading=rows[-1]["pose_heading_deg"])
        for i, r in enumerate(tail):
            r["t_s"] = rows[-1]["t_s"] + (i + 1) * 0.23
        got = arc_samples("r", rows + tail)
        self.assertEqual(len(got), 2, "조향이 다른 두 구간이 하나로 합쳐졌다")
        radii = sorted(s.radius_mm for s in got)
        self.assertAlmostEqual(radii[0], 610.0, delta=90.0)
        self.assertAlmostEqual(radii[1], 1100.0, delta=160.0)

    def test_last_valid_heading_is_excluded(self):
        """LAST_VALID 는 이전 heading 복사본이라 반경이 무한대로 튄다."""
        rows = _arc_rows(800.0, steering=0.8)
        for r in rows:
            r["pose_heading_source"] = "LAST_VALID"
        self.assertEqual(arc_samples("r", rows), [])

    def test_trusted_sources_are_exactly_the_measured_ones(self):
        self.assertEqual(TRUSTED_HEADING, {"FRONT_CUSHION", "TRAJECTORY"})

    def test_too_little_turn_is_rejected(self):
        """거의 직진인 구간에서 반경을 추정하면 값이 발산한다."""
        self.assertEqual(arc_samples("r", _arc_rows(800.0, steering=0.05,
                                                    turn_deg=2.0)), [])

    def test_too_short_window_is_rejected(self):
        self.assertEqual(
            arc_samples("r", _arc_rows(800.0, steering=0.8, turn_deg=60.0)[:2]),
            [])

    def test_repeated_identical_pose_does_not_inflate_samples(self):
        """같은 관측이 여러 tick 반복돼도 표본 수가 부풀지 않는다."""
        rows = _arc_rows(800.0, steering=0.8)
        doubled = [dict(r) for r in rows for _ in (0, 1)]
        for i, r in enumerate(doubled):
            r["t_s"] = i * 0.11
        got = arc_samples("r", doubled)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].samples, len(rows))


class SpeedEstimation(unittest.TestCase):

    def test_known_straight_speed_is_recovered(self):
        got = speed_samples("r", _straight_rows(120.0))
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0].speed_mm_s, 120.0, delta=6.0)

    def test_turning_windows_are_not_used_for_speed(self):
        rows = _straight_rows(120.0)
        for r in rows:
            r["wire_steering"] = 0.9
        self.assertEqual(speed_samples("r", rows), [])

    def test_zero_throttle_is_not_a_speed_sample(self):
        rows = _straight_rows(120.0, throttle=0.0)
        self.assertEqual(speed_samples("r", rows), [])


class StoppingEstimation(unittest.TestCase):

    def test_coasting_distance_is_measured(self):
        rows = _straight_rows(120.0, n=4)
        x = rows[-1]["pose_x_mm"]
        t = rows[-1]["t_s"]
        # zero 명령 후 60mm 더 밀린 뒤 정지
        for i, dx in enumerate((0.0, 35.0, 55.0, 60.0, 60.0, 60.0, 60.0,
                                60.0, 60.0, 60.0)):
            rows.append({**rows[-1], "t_s": t + (i + 1) * 0.23,
                         "pose_x_mm": x + dx, "throttle_cmd": 0.0})
        got = stop_samples("r", rows)
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0].coast_mm, 60.0, delta=5.0)
        self.assertAlmostEqual(got[0].throttle_before, 0.2, places=6)

    def test_resuming_drive_is_not_a_stop_sample(self):
        rows = _straight_rows(120.0, n=4)
        t = rows[-1]["t_s"]
        rows.append({**rows[-1], "t_s": t + 0.23, "throttle_cmd": 0.0})
        rows.append({**rows[-1], "t_s": t + 0.46, "throttle_cmd": 0.2})
        self.assertEqual(stop_samples("r", rows), [])


class ConfidenceReporting(unittest.TestCase):
    """표본이 적으면 적다고 말해야 한다 — 이 도구는 통제 실험이 아니다."""

    def test_confidence_never_claims_high(self):
        self.assertEqual(confidence(1000), "MEDIUM")

    def test_small_samples_are_unusable(self):
        self.assertEqual(confidence(3), "UNUSABLE")
        self.assertEqual(confidence(10), "LOW")


class DeadbandDetection(unittest.TestCase):
    """§29: 안 움직인 조건이 조용히 사라지면 안 된다."""

    @staticmethod
    def _held(throttle, net_mm, *, n=12, dt=0.22, direction="FORWARD"):
        """throttle 을 유지했는데 net_mm 만 움직인 구간."""
        rows = []
        for i in range(n):
            frac = i / max(1, n - 1)
            rows.append({
                "t_s": i * dt,
                "pose_x_mm": 600.0 + net_mm * frac,
                "pose_y_mm": 600.0,
                "pose_heading_deg": 0.0,
                "pose_heading_source": "FRONT_CUSHION",
                "wire_steering": 0.0,
                "throttle_cmd": throttle if direction == "FORWARD" else -throttle,
                "motion_direction": None,          # 계측 run 에는 waypoint 가 없다
                "phase": None,
                "curvature": None,
            })
        return rows

    def test_no_motion_is_reported_not_dropped(self):
        got = deadband_samples("r", self._held(0.10, 10.0))
        self.assertEqual(len(got), 1)
        self.assertFalse(got[0].motion_detected)
        self.assertAlmostEqual(got[0].displacement_mm, 10.0, delta=1.0)

    def test_real_motion_is_marked_moving(self):
        got = deadband_samples("r", self._held(0.25, 200.0))
        self.assertTrue(got[0].motion_detected)

    def test_calibration_rows_without_motion_direction_still_parse(self):
        """계측 run 은 motion_direction=None 이다 — 버려지면 안 된다."""
        got = deadband_samples("r", self._held(0.10, 10.0))
        self.assertEqual(got[0].direction, "FORWARD")

    def test_reverse_direction_comes_from_throttle_sign(self):
        got = deadband_samples("r", self._held(0.10, 10.0, direction="REVERSE"))
        self.assertEqual(got[0].direction, "REVERSE")

    def test_pose_jitter_is_not_mistaken_for_motion(self):
        """제자리 떨림의 누적 경로길이로 판정하면 안 된다."""
        rows = self._held(0.10, 0.0)
        for i, r in enumerate(rows):
            r["pose_x_mm"] = 600.0 + (3.0 if i % 2 else -3.0)   # ±3mm 진동
        got = deadband_samples("r", rows)
        self.assertEqual(len(got), 1)
        self.assertFalse(got[0].motion_detected)

    def test_short_windows_are_not_judged(self):
        self.assertEqual(deadband_samples("r", self._held(0.10, 10.0, n=3)), [])


class Run125917Regression(unittest.TestCase):
    """실차 run_20260903_125917: 0.10 을 2.4초 유지, 순변위 약 10mm.

    speed 표본으로는 n=0 이라 사라졌지만, deadband 로는 남아야 한다.
    """

    RUN = os.path.join("runs", "run_20260903_125917")

    def setUp(self):
        if not os.path.isdir(self.RUN):
            self.skipTest("run 기록이 없는 환경")

    def test_the_run_yields_a_deadband_sample(self):
        result = analyze([self.RUN])
        bands = result["deadband"]
        self.assertTrue(bands, "0.10 구간이 deadband 로도 안 잡혔다")
        self.assertTrue(any(not b["motion_detected"] for b in bands))

    def test_it_still_yields_no_speed_sample(self):
        """움직이지 않았으므로 속도 표본은 없는 것이 맞다."""
        self.assertEqual(analyze([self.RUN])["speeds"], [])


if __name__ == "__main__":
    unittest.main()
