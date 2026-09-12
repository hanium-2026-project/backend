"""전방 쿠션 매칭·heading 검증 (2클래스 모델 도입 전 선행 구현분).

실제 가중치가 없어도 계약을 고정해두기 위해, 탐지 결과를 직접 만들어 검증한다.
가중치가 준비되면 이 테스트가 회귀 방지선이 된다.
"""

from __future__ import annotations

import math
import unittest

from cv.association import associate
from cv.heading import HeadingEstimator
from cv.vehicle_detector import LABEL_CAR, LABEL_CUSHION, Detection


def car(cx: float, cy: float, tid: int, size: float = 100.0, conf: float = 0.9) -> Detection:
    h = size / 2
    return Detection(LABEL_CAR, conf, (int(cx-h), int(cy-h), int(cx+h), int(cy+h)), tid)


def cushion(cx: float, cy: float, tid: int | None = None,
            size: float = 30.0, conf: float = 0.85) -> Detection:
    h = size / 2
    return Detection(LABEL_CUSHION, conf, (int(cx-h), int(cy-h), int(cx+h), int(cy+h)), tid)


class TestAssociation(unittest.TestCase):
    def test_single_pair(self):
        """차량 앞에 놓인 쿠션이 그 차량과 묶인다."""
        pairs, unpaired = associate([car(500, 500, 1), cushion(500, 460)])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(unpaired, [])
        self.assertEqual(pairs[0].track_id, 1)

    def test_no_cushion_falls_back(self):
        """1클래스 모델처럼 쿠션이 없으면 전부 unpaired 로 나온다."""
        pairs, unpaired = associate([car(500, 500, 1), car(800, 500, 2)])
        self.assertEqual(pairs, [])
        self.assertEqual(len(unpaired), 2)

    def test_far_cushion_not_paired(self):
        """멀리 떨어진 쿠션은 다른 차량 것으로 보고 묶지 않는다."""
        pairs, unpaired = associate([car(500, 500, 1), cushion(1100, 500)])
        self.assertEqual(pairs, [])
        self.assertEqual(len(unpaired), 1)

    def test_two_cars_each_get_own_cushion(self):
        """두 대가 가까이 있어도 각자의 쿠션에 1:1 로 붙는다."""
        dets = [car(400, 500, 1), cushion(400, 460),
                car(700, 500, 2), cushion(700, 460)]
        pairs, unpaired = associate(dets)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(unpaired, [])
        mapping = {p.track_id: p.cushion_center_px[0] for p in pairs}
        self.assertAlmostEqual(mapping[1], 400.0, delta=1.0)
        self.assertAlmostEqual(mapping[2], 700.0, delta=1.0)

    def test_one_cushion_never_shared(self):
        """쿠션 하나를 두 차량이 나눠 갖지 않는다 (더 가까운 쪽만)."""
        dets = [car(480, 500, 1), car(560, 500, 2), cushion(490, 500)]
        pairs, unpaired = associate(dets)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(unpaired), 1)
        self.assertEqual(pairs[0].track_id, 1, "더 가까운 차량에 붙어야 한다")

    def test_heading_continuity_breaks_tie(self):
        """거리가 비슷하면 직전 heading 과 이어지는 쪽을 고른다."""
        # 차량 좌우에 쿠션이 하나씩 — 거리만으로는 동점
        dets = [car(500, 500, 1), cushion(560, 500), cushion(440, 500)]

        def heading_of(car_px, cu_px):
            # 픽셀=맵 이라고 가정하되 y 축만 뒤집는다
            dx = cu_px[0] - car_px[0]
            dy = -(cu_px[1] - car_px[1])
            return math.degrees(math.atan2(dy, dx)) % 360.0

        pairs, _ = associate(dets, previous_heading={1: 0.0},   # 직전엔 오른쪽
                             image_heading_of=heading_of)
        self.assertEqual(len(pairs), 1)
        self.assertGreater(pairs[0].cushion_center_px[0], 500.0,
                           "직전 방향(오른쪽)과 이어지는 쿠션을 골라야 한다")


class TestHeadingWithCushion(unittest.TestCase):
    def test_stationary_vehicle_has_heading(self):
        """궤적 방식의 최대 약점 — 정지 중에도 방향을 알 수 있다."""
        est = HeadingEstimator(min_move=30.0)
        for _ in range(3):
            r = est.update(1, (500.0, 500.0), front_point=(500.0, 560.0))
        self.assertEqual(r.source, "FRONT_CUSHION")
        self.assertAlmostEqual(r.heading_deg, 90.0, delta=0.1)
        self.assertFalse(r.is_moving)

    def test_reverse_keeps_true_heading(self):
        """후진 시 궤적은 180° 반대지만 쿠션은 머리 방향을 유지한다."""
        est = HeadingEstimator(min_move=30.0)
        for y in (500.0, 440.0, 380.0, 320.0):            # 아래로 이동 = 후진
            r = est.update(1, (500.0, y), front_point=(500.0, y + 60.0))
        self.assertEqual(r.source, "FRONT_CUSHION")
        self.assertAlmostEqual(r.heading_deg, 90.0, delta=0.1)
        self.assertTrue(r.is_moving)

    def test_priority_over_trajectory(self):
        """쿠션이 보이면 궤적보다 우선한다 (§6.6)."""
        est = HeadingEstimator(min_move=30.0)
        for x in (300.0, 400.0, 500.0):                   # 오른쪽으로 이동 → 궤적 0°
            est.update(1, (x, 500.0))
        r = est.update(1, (600.0, 500.0), front_point=(600.0, 560.0))
        self.assertEqual(r.source, "FRONT_CUSHION")
        self.assertAlmostEqual(r.heading_deg, 90.0, delta=0.1)

    def test_falls_back_when_cushion_lost(self):
        """쿠션이 가려지면 궤적으로 내려가고, 그것도 없으면 마지막 값을 쓴다."""
        est = HeadingEstimator(min_move=30.0)
        est.update(1, (500.0, 500.0), front_point=(560.0, 500.0))
        for x in (500.0, 560.0, 620.0):                   # 쿠션 없이 이동
            r = est.update(1, (x, 500.0))
        self.assertEqual(r.source, "TRAJECTORY")
        r2 = est.update(1, (620.0, 500.0))                # 정지 → 마지막 유효값
        for _ in range(4):
            r2 = est.update(1, (620.0, 500.0))
        self.assertEqual(r2.source, "LAST_VALID")


class TestLabelNormalization(unittest.TestCase):
    def test_class_name_variants(self):
        """학습 시 클래스 표기가 달라도 같은 내부 라벨로 받는다."""
        from cv.vehicle_detector import YoloVehicleDetector
        det = YoloVehicleDetector.__new__(YoloVehicleDetector)
        det.custom_model = True
        for name in ("rc_car", "RC_CAR", "RC Car", "rc-car"):
            self.assertEqual(det._resolve_label(name), LABEL_CAR, name)
        for name in ("front_cushion", "FRONT_CUSHION", "Front Cushion", "cushion"):
            self.assertEqual(det._resolve_label(name), LABEL_CUSHION, name)

    def test_unknown_class_treated_as_car(self):
        """커스텀 모델의 모르는 클래스는 차량으로 본다 (1클래스 모델 호환)."""
        from cv.vehicle_detector import YoloVehicleDetector
        det = YoloVehicleDetector.__new__(YoloVehicleDetector)
        det.custom_model = True
        self.assertEqual(det._resolve_label("class_0"), LABEL_CAR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
