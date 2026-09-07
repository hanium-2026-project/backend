"""FINAL 직선 후진: 정렬돼 있으면 조향을 cap, 아니면 PD 유지 (Phase 1).

배경 (실차 감사): 입구 시작 rear 주차의 최초 FINAL 이 반복적으로 슬롯 뒤
(=맵 경계, 여유 25mm)를 넘어 BOUNDARY_HARD 를 냈다. 그 판별자는 FINAL
직선 후진에서의 조향 크기였다:

    성공 002703/022217   FINAL |steer| 0.15~0.21  -> PWM 20~22  -> 정지
    실패 191010~224451   FINAL |steer| 0.28~0.84  -> PWM 25~37  -> 과주행 +51~88mm

FINAL waypoint 는 슬롯 중심을 향한 곡률 0 직선 후진인데, 제어기는 끝점
bearing 을 향해 조향한다. 목표에 가까워질수록 그 bearing 오차가
atan(횡오차/남은거리) 로 커져 firmware turn-duty 를 밟고, 그 PWM 상승이
개루프 속도를 높여 정지거리를 늘린다.

Phase 1 수정: 차가 이미 슬롯 축에 정렬(FINAL 도착 허용오차 안)돼 있으면
FINAL 직선 후진의 wire steering 을 cap 한다(0 lock 이 아니다 — 8~11° crab
보정에 필요한 완만한 조향은 남긴다). 정렬 판정 임계값은 waypoint 가 이미 들고
있는 heading_tolerance_deg / position_tolerance_cm, cap 값은 firmware duty
전이 중앙에서 유도한다. 새 튜닝값이 아니다.

이건 steering-PWM-overshoot 가설을 실차로 검증하기 위한 진단 변경이다.
FINAL 자세 품질(heading/lateral/footprint)은 상위 FINAL_POSE_EVAL 이 그대로
판정하므로, 조향을 잠근다고 잘못 정렬된 자세가 PARKED 로 인정되지 않는다.
"""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose, Waypoint
from controller.pose_controller import PoseWaypointController
from parking.waypoints import default_slot_specs

SPECS = default_slot_specs()


def _rear_final_wp(slot: str) -> Waypoint:
    sp = SPECS[slot]
    parked = (sp.target_heading_deg + 180.0) % 360.0
    # 실제 rear FINAL waypoint 가 들고 있는 값 (PHASE_DEFAULTS["FINAL"]).
    return Waypoint(sp.center_x, sp.center_y, target_heading_deg=parked,
                    position_tolerance_cm=5.0, heading_tolerance_deg=12.0,
                    heading_required=True, is_final=True,
                    motion_direction=MotionDirection.REVERSE, phase="FINAL",
                    curvature=0.0, route_id=1, waypoint_id=1)


def _steer(slot, x, y, h, *, enabled=True):
    cfg = ControllerConfig(allow_reverse=True,
                           final_reverse_straight_when_aligned=enabled)
    ctrl = PoseWaypointController(cfg)
    ctrl._motion_direction = MotionDirection.REVERSE
    cmd = ctrl.compute(Pose(x, y, h, timestamp=100.0,
                            heading_source="FRONT_CUSHION"),
                       _rear_final_wp(slot), allow_drive=True, now=100.0)
    return cmd


class AlignedFinalReverseCapsSteering(unittest.TestCase):
    """정렬된 B1 FINAL 후진(실패 run 들의 실제 자세)은 조향이 cap 된다."""

    # 실차 FINAL 직선 후진 중간 자세 (control.jsonl). target B1 (425,1050,270).
    B1_ALIGNED = [
        ("191010", 428.0, 866.0, 267.0),
        ("191220", 445.0, 942.0, 262.0),
        ("193043", 437.0, 915.0, 266.0),
        ("193223", 442.0, 952.0, 260.0),
        ("193518", 457.0, 931.0, 268.0),
        ("224157", 439.0, 920.0, 262.0),
        ("224451", 439.0, 942.0, 258.0),
    ]

    def test_all_failed_b1_final_poses_are_capped(self) -> None:
        cap = ControllerConfig().final_reverse_aligned_steer_cap
        for run, x, y, h in self.B1_ALIGNED:
            with self.subTest(run=run):
                self.assertLessEqual(abs(_steer("B1", x, y, h).steering),
                                     cap + 1e-6)

    def test_the_big_spikes_are_actually_reduced(self) -> None:
        """cap 이 실제로 큰 조향을 깎는지 (아래에서 이미 컸던 자세만)."""
        cap = ControllerConfig().final_reverse_aligned_steer_cap
        reduced = 0
        for run, x, y, h in self.B1_ALIGNED:
            old = abs(_steer("B1", x, y, h, enabled=False).steering)
            new = abs(_steer("B1", x, y, h, enabled=True).steering)
            if old > cap + 1e-3:
                self.assertLessEqual(new, cap + 1e-6)
                self.assertLess(new, old)
                reduced += 1
        self.assertGreaterEqual(reduced, 4, "큰 조향 spike 가 실제로 깎여야 한다")

    def test_disabling_the_flag_restores_pd_steering(self) -> None:
        """진단 스위치를 끄면 조향이 되살아난다 (실차 A/B 검증용)."""
        nonzero = 0
        for run, x, y, h in self.B1_ALIGNED:
            if abs(_steer("B1", x, y, h, enabled=False).steering) > 1e-3:
                nonzero += 1
        self.assertGreaterEqual(nonzero, 5,
                                "flag off 면 대부분 PD 조향이 있어야 한다")

    def test_022217_known_good_is_below_the_cap_so_unchanged(self) -> None:
        """알려진 성공(정렬 양호)은 원래 조향이 cap 아래라 그대로다."""
        cap = ControllerConfig().final_reverse_aligned_steer_cap
        old = abs(_steer("A3", 883.0, 328.0, 90.0, enabled=False).steering)
        new = abs(_steer("A3", 883.0, 328.0, 90.0, enabled=True).steering)
        self.assertLessEqual(old, cap + 1e-6)
        self.assertAlmostEqual(old, new, places=6)


class MisalignedFinalReverseKeepsSteering(unittest.TestCase):
    """정렬 밖(heading 게이트 초과)이면 기존 PD 되먹임을 유지한다."""

    def test_002703_heading_artifact_is_not_locked(self) -> None:
        """002703 FINAL 후진은 body heading 이 진행방향과 28deg 벌어진 측정

        자세다(target 90, body 62). heading 게이트(12deg)를 통과하지 못하므로
        기존 동작을 그대로 유지한다 — 알려진 성공을 깨지 않는다.
        """
        cmd = _steer("A2", 644.0, 281.0, 62.0)
        self.assertNotAlmostEqual(cmd.steering, 0.0, places=3)
        # flag on/off 가 이 자세에서는 같아야 한다 (회귀 없음).
        self.assertAlmostEqual(
            _steer("A2", 644.0, 281.0, 62.0, enabled=True).steering,
            _steer("A2", 644.0, 281.0, 62.0, enabled=False).steering,
            places=6)

    def test_a_clearly_crooked_reverse_still_corrects(self) -> None:
        """heading 15deg 어긋난 자세는 게이트 밖이라 조향이 산다."""
        self.assertNotAlmostEqual(_steer("B1", 425.0, 950.0, 285.0).steering,
                                  0.0, places=3)


class OnlyFinalReverseIsAffected(unittest.TestCase):

    def test_forward_final_is_not_locked(self) -> None:
        """전진 FINAL 은 이 잠금 대상이 아니다 (후진 전용)."""
        cfg = ControllerConfig(allow_reverse=True)
        ctrl = PoseWaypointController(cfg)
        ctrl._motion_direction = MotionDirection.FORWARD
        wp = _rear_final_wp("B1")
        wp = Waypoint(wp.x_mm, wp.y_mm, target_heading_deg=wp.target_heading_deg,
                      position_tolerance_cm=wp.position_tolerance_cm,
                      heading_tolerance_deg=wp.heading_tolerance_deg,
                      heading_required=True, is_final=True,
                      motion_direction=MotionDirection.FORWARD, phase="FINAL",
                      curvature=0.0, route_id=1, waypoint_id=1)
        cmd = ctrl.compute(Pose(440.0, 1000.0, 90.0, timestamp=100.0,
                                heading_source="FRONT_CUSHION"),
                           wp, allow_drive=True, now=100.0)
        # 전진이므로 lock 분기를 타지 않는다 — 조향 계산이 그대로 수행된다.
        self.assertIsNotNone(cmd)

    def test_recovery_reverse_lock_is_unchanged(self) -> None:
        cfg = ControllerConfig(allow_reverse=True)
        self.assertTrue(cfg.reverse_steering_locked("RECOVERY"))
        self.assertFalse(cfg.reverse_steering_locked("FINAL"))


if __name__ == "__main__":                     # pragma: no cover
    unittest.main()
