"""vision 으로 확정된 정적 주차 차량을 planner 가 어떻게 보는가.

실측 배경 (run_20260906_190856 / _191008)
----------------------------------------
전원이 꺼진 CAR_02 를 B1 에 놓았더니:

    rc_car          매 프레임 2대 모두 검출
    front_cushion   대부분 1개(=CAR_01)만 검출
    -> CAR_02 heading_source = LAST_VALID  (190856 74%, 191008 95%)

그러면 _planning_obstacle_snapshot 의 trusted 검사에 걸려 uncertain 이 되고,
_trajectory_verdict 가 **모든** route/recovery 를 OTHER_VEHICLE_POSE_UNCERTAIN
으로 거절한다. 두 run 다 정확히 그렇게 끝났다:

    190856  t=18.358  RECOVERY_REJECTED  OTHER_VEHICLE_POSE_UNCERTAIN
    191008  t=29.788  RECOVERY_REJECTED  OTHER_VEHICLE_POSE_UNCERTAIN

이 파일이 지키는 계약
--------------------
확정된 칸의 차는 "자세를 모르는 차" 가 아니라 "그 칸에 세워둔 차" 다.
기존 _parked_obstacles 와 **같은 정적 의미**로 취급하되,

    - 장애물 목록에서 빼지 않는다 (물리 차체는 그대로 남는다)
    - 가짜 car_id 를 주지 않는다
    - base allocator 예약/PARKED 상태를 건드리지 않는다
    - 점유가 풀리면 정적 장애물도 같이 사라진다
"""

from __future__ import annotations

import math
import unittest
from collections import deque

from parking.final_alignment import rear_parked_heading_deg
from parking.waypoints import Waypoint, default_slot_specs
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView
from rl.parking_env import SLOT_NAMES

B1 = default_slot_specs()["B1"]
A2 = default_slot_specs()["A2"]

# run_20260906_190856 / _191008 의 실측 자세.
CAR02_190856 = (416.6, 1046.0)          # heading_source 74% LAST_VALID
CAR02_191008 = (417.4, 1054.6)          # heading_source 95% LAST_VALID
CAR01_190856_REPOSITION = (518.4, 577.8, 19.3)   # SLOT_REPOSITION 시점
CAR01_191008_REPOSITION = (497.7, 456.8, 359.3)
AISLE = (450.0, 600.0)


def _view(track_id, pos, *, car_id=None, heading=268.7,
          source="LAST_VALID", t=100.0, stationary=True) -> VehicleView:
    v = VehicleView(track_id=track_id, car_id=car_id, position_mm=pos,
                    heading_deg=heading, heading_source=source,
                    confidence=0.94, last_obs_time=t)
    v.recent = deque([pos] * 8, maxlen=8) if stationary else deque(
        [(pos[0] - 40.0 * i, pos[1]) for i in range(8, 0, -1)], maxlen=8)
    return v


def _wp(x, y, route_id=9):
    return Waypoint(route_id=route_id, waypoint_id=1, phase="RECOVERY",
                    x=x, y=y, target_heading_deg=0.0, speed_cm_s=5.0,
                    position_tolerance_cm=8.0, heading_tolerance_deg=12.0,
                    heading_required=False, is_final=True,
                    motion_direction="FORWARD")


class _Base(unittest.TestCase):

    def setUp(self) -> None:
        self.pipe = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                                   parking_mode="rear"))
        self.pipe.events = []
        self.pipe.on_event_record = lambda n, **f: self.pipe.events.append((n, f))
        self.t = 100.0

    def tick(self, views, dt=0.25):
        self.t += dt
        for v in views:
            v.last_obs_time = self.t
            self.pipe.views[v.track_id] = v
        self.pipe._update_vision_occupancy(list(views), self.t)

    def confirm(self, views, n=5):
        for _ in range(n):
            self.tick(views)

    def snapshot(self, ego):
        self.pipe.views[ego.track_id] = ego
        return self.pipe._planning_obstacle_snapshot(ego)

    def statics(self):
        return dict(self.pipe._vision_parked_obstacles)


# ══ A. 확정된 칸의 LAST_VALID 차량은 uncertain 이 아니다 ════════════════════

class ConfirmedSlotVehicleIsNotUncertain(_Base):

    def test_last_valid_heading_no_longer_blocks_every_route(self) -> None:
        car02 = _view(3, CAR02_190856)                     # LAST_VALID
        ego = _view(1, CAR01_190856_REPOSITION[:2], car_id=1,
                    heading=CAR01_190856_REPOSITION[2],
                    source="FRONT_CUSHION")
        self.confirm([car02, ego])
        poses, uncertain = self.snapshot(ego)
        self.assertEqual(uncertain, (), "확정된 칸의 차가 여전히 uncertain 이다")
        self.assertTrue(poses, "장애물이 통째로 사라졌다")

    def test_the_verdict_no_longer_returns_other_vehicle_pose_uncertain(self):
        car02 = _view(3, CAR02_190856)
        ego = _view(1, CAR01_190856_REPOSITION[:2], car_id=1,
                    heading=CAR01_190856_REPOSITION[2],
                    source="FRONT_CUSHION")
        self.confirm([car02, ego])
        _result, reason = self.pipe._trajectory_verdict(
            ego, [_wp(430.0, 600.0)], "A2")
        self.assertNotEqual(reason, "OTHER_VEHICLE_POSE_UNCERTAIN")

    def test_it_holds_for_both_recorded_runs(self) -> None:
        for label, c2, ego_pose in (("190856", CAR02_190856,
                                     CAR01_190856_REPOSITION),
                                    ("191008", CAR02_191008,
                                     CAR01_191008_REPOSITION)):
            with self.subTest(run=label):
                self.setUp()
                car02 = _view(3, c2)
                ego = _view(1, ego_pose[:2], car_id=1, heading=ego_pose[2],
                            source="FRONT_CUSHION")
                self.confirm([car02, ego])
                _poses, uncertain = self.snapshot(ego)
                self.assertEqual(uncertain, (), label)


# ══ B. 그래도 B1 은 물리적으로 막혀 있어야 한다 ═════════════════════════════

class ConfirmedSlotStaysAPhysicalObstacle(_Base):

    def test_the_static_pose_is_recorded_at_the_measured_position(self) -> None:
        car02 = _view(3, CAR02_190856)
        self.confirm([car02])
        st = self.statics()
        self.assertIn("B1", st)
        x, y, h = st["B1"]
        self.assertAlmostEqual(x, CAR02_190856[0], places=3)
        self.assertAlmostEqual(y, CAR02_190856[1], places=3)
        # 못 믿는 축(heading)만 슬롯 기하로 대체된다.
        self.assertAlmostEqual(h, rear_parked_heading_deg(B1), places=6)

    def test_the_obstacle_is_present_in_the_planning_snapshot(self) -> None:
        car02 = _view(3, CAR02_190856)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        self.confirm([car02, ego])
        poses, _ = self.snapshot(ego)
        self.assertTrue(any(math.hypot(p[0] - CAR02_190856[0],
                                       p[1] - CAR02_190856[1]) < 1.0
                            for p in poses),
                        f"CAR_02 가 장애물 목록에서 사라졌다: {poses}")

    def test_a_trajectory_through_b1_is_still_rejected(self) -> None:
        car02 = _view(3, CAR02_190856)
        ego = _view(1, (425.0, 700.0), car_id=1, heading=90.0,
                    source="FRONT_CUSHION")
        self.confirm([car02, ego])
        # B1 한가운데를 지나는 경로
        safe = self.pipe._trajectory_safe(
            ego, [_wp(B1.center_x, B1.center_y)], slot_id="A2")
        self.assertFalse(safe, "점유된 B1 을 통과하는 경로가 통과됐다")

    def test_the_vehicle_is_never_counted_twice(self) -> None:
        """heading 이 되살아나도 static + dynamic 중복 삽입 금지."""
        car02 = _view(3, CAR02_190856)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        self.confirm([car02, ego])
        car02.heading_source = "FRONT_CUSHION"      # 쿠션이 다시 잡혔다
        self.tick([car02, ego])
        poses, uncertain = self.snapshot(ego)
        near = [p for p in poses
                if math.hypot(p[0] - CAR02_190856[0],
                              p[1] - CAR02_190856[1]) < 60.0]
        self.assertEqual(len(near), 1, f"같은 차가 두 번 들어갔다: {poses}")
        self.assertEqual(uncertain, ())


# ══ C. 점유 해제 → 정적 장애물도 사라진다 ═══════════════════════════════════

class ClearingRemovesTheStaticObstacle(_Base):

    def test_release_grace_then_the_obstacle_disappears(self) -> None:
        car02 = _view(3, CAR02_190856)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        self.confirm([car02, ego])
        self.assertIn("B1", self.statics())
        release = float(self.pipe.config.vision_occupancy_release_s)
        self.tick([ego], dt=release * 0.5)
        self.assertIn("B1", self.statics(), "유예 안에서 사라졌다")
        self.tick([ego], dt=release)
        self.assertNotIn("B1", self.statics())
        self.assertIn("VISION_SLOT_CLEARED",
                      [n for n, _ in self.pipe.events])

    def test_a_short_dropout_keeps_the_obstacle(self) -> None:
        car02 = _view(3, CAR02_190856)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        self.confirm([car02, ego])
        self.tick([ego])
        self.tick([ego])
        self.assertIn("B1", self.statics())
        poses, _ = self.snapshot(ego)
        self.assertTrue(any(math.hypot(p[0] - CAR02_190856[0],
                                       p[1] - CAR02_190856[1]) < 1.0
                            for p in poses))


# ══ D~F. 기존 semantics 는 그대로 ═══════════════════════════════════════════

class ExistingSemanticsUnchanged(_Base):

    def test_an_unbound_vehicle_outside_any_slot_is_still_uncertain(self) -> None:
        stray = _view(7, (700.0, 600.0), source="LAST_VALID")   # 통로 한복판
        ego = _view(1, (200.0, 600.0), car_id=1, heading=0.0,
                    source="FRONT_CUSHION")
        self.confirm([stray, ego])
        _poses, uncertain = self.snapshot(ego)
        self.assertEqual(uncertain, (None,),
                         "슬롯 밖 unbound 차량의 기존 계약이 깨졌다")

    def test_before_confirmation_the_old_semantics_apply(self) -> None:
        car02 = _view(3, CAR02_190856)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        self.tick([car02, ego])                 # 1 프레임 — 아직 미확정
        self.assertEqual(self.statics(), {})
        _poses, uncertain = self.snapshot(ego)
        self.assertEqual(uncertain, (None,))

    def test_a_bound_car_in_a_slot_never_becomes_a_vision_static(self) -> None:
        ego = _view(1, CAR02_190856, car_id=1, heading=270.0,
                    source="FRONT_CUSHION")
        self.confirm([ego], n=12)
        self.assertEqual(self.statics(), {})
        self.assertEqual(
            [n for n, _ in self.pipe.events if n.startswith("VISION_SLOT")], [])

    def test_a_moving_vehicle_in_a_slot_is_not_promoted(self) -> None:
        passing = _view(7, CAR02_190856, stationary=False)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        self.confirm([passing, ego], n=10)
        self.assertEqual(self.statics(), {})

    def test_normal_parked_obstacles_are_untouched(self) -> None:
        self.pipe._parked_obstacles[2] = (650.0, 150.0, 90.0)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        poses, uncertain = self.snapshot(ego)
        self.assertIn((650.0, 150.0, 90.0), poses)
        self.assertEqual(uncertain, ())

    def test_ego_own_parked_pose_is_still_excluded(self) -> None:
        self.pipe._parked_obstacles[1] = (650.0, 150.0, 90.0)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        poses, _ = self.snapshot(ego)
        self.assertNotIn((650.0, 150.0, 90.0), poses)


# ══ H. base allocator 상태 불변 ═════════════════════════════════════════════

class BaseAllocatorStateUntouched(_Base):

    def test_vision_static_promotion_never_writes_base_status(self) -> None:
        idx = SLOT_NAMES.index("B1")
        car02 = _view(3, CAR02_190856)
        self.confirm([car02])
        self.assertEqual(float(self.pipe.allocator.slot_statuses[idx]), 0.0,
                         "base 예약 상태가 오염됐다")
        self.assertEqual(float(self.pipe.allocator.vision_occupied[idx]), 1.0)

    def test_clearing_does_not_free_a_real_reservation(self) -> None:
        idx = SLOT_NAMES.index("B1")
        self.pipe.allocator.set_slot_occupied("B1", True)      # 예약
        car02 = _view(3, CAR02_190856)
        ego = _view(1, AISLE, car_id=1, heading=0.0, source="FRONT_CUSHION")
        self.confirm([car02, ego])
        self.tick([ego], dt=float(self.pipe.config.vision_occupancy_release_s) * 2)
        self.assertNotIn("B1", self.statics())
        self.assertGreaterEqual(
            float(self.pipe.allocator.slot_statuses[idx]), 0.5)

    def test_no_fake_car_id_is_created(self) -> None:
        car02 = _view(3, CAR02_190856)
        self.confirm([car02])
        self.assertIsNone(car02.car_id)
        self.assertEqual(self.pipe.track_of_car, {})
        self.assertEqual(self.pipe.orchestrator.missions, {})
        self.assertNotIn(None, self.pipe._parked_obstacles)
        self.assertEqual(self.pipe._parked_obstacles, {})


# ══ 실측 replay — 두 run 의 거절 조건을 그대로 재현 ═════════════════════════

class RecordedRunReplay(_Base):

    def _replay(self, car02_pos, ego_pose, cushion_ratio):
        """LAST_VALID 가 대부분인 실측 heading 패턴을 그대로 흘린다."""
        car02 = _view(3, car02_pos)
        ego = _view(1, ego_pose[:2], car_id=1, heading=ego_pose[2],
                    source="FRONT_CUSHION")
        for i in range(12):
            car02.heading_source = ("FRONT_CUSHION"
                                    if (i % 10) < cushion_ratio
                                    else "LAST_VALID")
            self.tick([car02, ego])
        return ego

    def test_190856_reposition_is_no_longer_rejected(self) -> None:
        ego = self._replay(CAR02_190856, CAR01_190856_REPOSITION, 3)
        _poses, uncertain = self.snapshot(ego)
        self.assertEqual(uncertain, ())
        _r, reason = self.pipe._trajectory_verdict(
            ego, [_wp(430.0, 600.0)], "A2")
        self.assertNotEqual(reason, "OTHER_VEHICLE_POSE_UNCERTAIN")

    def test_191008_reposition_is_no_longer_rejected(self) -> None:
        ego = self._replay(CAR02_191008, CAR01_191008_REPOSITION, 0)
        _poses, uncertain = self.snapshot(ego)
        self.assertEqual(uncertain, ())
        _r, reason = self.pipe._trajectory_verdict(
            ego, [_wp(430.0, 600.0)], "A2")
        self.assertNotEqual(reason, "OTHER_VEHICLE_POSE_UNCERTAIN")

    def test_b1_remains_unavailable_to_the_allocator(self) -> None:
        self._replay(CAR02_190856, CAR01_190856_REPOSITION, 0)
        masks = self.pipe.allocator._build_masks()
        self.assertFalse(bool(masks[SLOT_NAMES.index("B1")]))
        for slot in ("A1", "A2", "A3", "A4", "B2", "B3", "B4"):
            self.assertTrue(bool(masks[SLOT_NAMES.index(slot)]), slot)


if __name__ == "__main__":
    unittest.main()
