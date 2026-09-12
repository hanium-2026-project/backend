"""카메라만으로 판정하는 정적 슬롯 점유 (촬영 시나리오 전용).

촬영 계약:
    CAR_02 는 **전원 OFF / ESP 미연결 / CAR_ID binding 없음** 상태로 슬롯에
    손으로 놓인다. 천장 카메라가 rc_car 로 검출하는 것이 전부다.
    시스템은 그 영상 정보만으로 "그 칸은 이미 찼다" 를 판정하고,
    CAR_01 의 allocator 가 그 칸을 후보에서 제외해야 한다.

여기서 검증하는 것은 production 함수 그 자체다:
    ParkingPipeline._update_vision_occupancy   (프레임 훅)
    RealtimeAllocator.set_vision_occupied      (overlay)
    RealtimeAllocator.effective_slot_statuses  (base OR vision)
    RealtimeAllocator._build_masks             (실제 배정 마스크)

절대 계약 두 가지:
    - vision clear 가 기존 예약/PARKED 를 **절대** 지우지 않는다.
    - CAR_01(ego) 이 자기 자신을 외부 주차 차량으로 오인하지 않는다.
"""

from __future__ import annotations

import unittest
from collections import deque

import numpy as np

from parking.final_alignment import in_final_region
from parking.waypoints import default_slot_specs
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView
from rl.bridge import RealtimeAllocator
from rl.parking_env import SLOT_NAMES

B1 = default_slot_specs()["B1"]
A2 = default_slot_specs()["A2"]

# CAR_02 를 B1 한가운데 세워 둔 자세 (실촬영 배치).
B1_CENTRE = (B1.center_x, B1.center_y)
A2_CENTRE = (A2.center_x, A2.center_y)
# 통로 한복판 — 어떤 슬롯에도 들어가면 안 된다.
AISLE = (450.0, 600.0)
# 입구 — CAR_01 이 출발하는 곳.
ENTRANCE = (140.0, 175.0)


def _view(track_id: int, pos, *, car_id=None, stationary=True,
          heading=90.0) -> VehicleView:
    """정지/이동 이력을 갖춘 관측 1건."""
    v = VehicleView(track_id=track_id, car_id=car_id, position_mm=pos,
                    heading_deg=heading, heading_source="FRONT_CUSHION",
                    confidence=0.94)
    if stationary:
        v.recent = deque([pos] * 8, maxlen=8)
    else:
        # 프레임마다 40mm 씩 움직인다 (stationary_tolerance_mm = 15).
        v.recent = deque([(pos[0] - 40.0 * i, pos[1]) for i in range(8, 0, -1)],
                         maxlen=8)
        v.position_mm = pos
    return v


class _Base(unittest.TestCase):

    def setUp(self) -> None:
        self.pipe = ParkingPipeline(PipelineConfig(control_mode="auto-host",
                                                   parking_mode="rear"))
        self.pipe.events = []
        self.pipe.on_event_record = lambda n, **f: self.pipe.events.append((n, f))
        self.t = 100.0

    def tick(self, views, dt: float = 0.25) -> None:
        """카메라 프레임 1장을 production 훅에 그대로 흘린다."""
        self.t += dt
        self.pipe._update_vision_occupancy(list(views), self.t)

    def occupied(self, slot: str) -> bool:
        idx = SLOT_NAMES.index(slot)
        return bool(self.pipe.allocator.vision_occupied[idx] >= 0.5)

    def selectable(self, slot: str) -> bool:
        """allocator 가 실제로 이 칸을 고를 수 있는가 (production 마스크)."""
        masks = self.pipe.allocator._build_masks()
        return bool(masks[SLOT_NAMES.index(slot)])

    def names(self) -> list[str]:
        return [n for n, _ in self.pipe.events]


# ══ TEST A. 빈 주차장 ═══════════════════════════════════════════════════════

class EmptyLot(_Base):

    def test_no_vehicle_means_every_slot_stays_selectable(self) -> None:
        for _ in range(10):
            self.tick([])
        for slot in SLOT_NAMES:
            with self.subTest(slot=slot):
                self.assertFalse(self.occupied(slot))
                self.assertTrue(self.selectable(slot))
        self.assertEqual(self.names(), [])

    def test_a_car_in_the_aisle_is_not_an_occupied_slot(self) -> None:
        """통로를 지나가는(또는 서 있는) 차를 슬롯 점유로 오인하면 안 된다."""
        for _ in range(10):
            self.tick([_view(3, AISLE)])
        for slot in SLOT_NAMES:
            self.assertFalse(self.occupied(slot), slot)
        # 전제 확인 — 통로 자세는 어떤 슬롯의 최종 구역에도 없다.
        for slot_id, spec in default_slot_specs().items():
            self.assertFalse(in_final_region(spec, *AISLE), slot_id)

    def test_the_entrance_is_not_inside_any_slot(self) -> None:
        for slot_id, spec in default_slot_specs().items():
            self.assertFalse(in_final_region(spec, *ENTRANCE), slot_id)


# ══ TEST B. 전원 꺼진 CAR_02 가 B1 에 서 있다 ═══════════════════════════════

class UnboundStaticVehicleOccupiesTheSlot(_Base):

    def test_b1_becomes_vision_occupied_without_any_car_id(self) -> None:
        car02 = _view(7, B1_CENTRE)          # car_id 없음 = ESP 미연결
        self.assertIsNone(car02.car_id)
        confirm = self.pipe.config.vision_occupancy_confirm_observations
        for _ in range(confirm):
            self.tick([car02])
        self.assertTrue(self.occupied("B1"))
        self.assertIn("VISION_SLOT_OCCUPIED", self.names())

    def test_the_allocator_can_no_longer_select_b1(self) -> None:
        car02 = _view(7, B1_CENTRE)
        for _ in range(5):
            self.tick([car02])
        self.assertFalse(self.selectable("B1"), "B1 이 아직 후보로 남아 있다")
        for slot in ("A1", "A2", "A3", "A4", "B2", "B3", "B4"):
            self.assertTrue(self.selectable(slot), slot)

    def test_it_works_for_an_arbitrary_slot_too(self) -> None:
        """B1 은 예시일 뿐이다."""
        for slot in ("A2", "B3", "A4"):
            with self.subTest(slot=slot):
                self.setUp()
                spec = default_slot_specs()[slot]
                for _ in range(5):
                    self.tick([_view(7, (spec.center_x, spec.center_y))])
                self.assertTrue(self.occupied(slot))
                self.assertFalse(self.selectable(slot))

    def test_a_moving_vehicle_inside_a_slot_is_not_occupancy(self) -> None:
        """지나가는 차와 세워둔 차를 가른다 (기존 is_stationary 재사용)."""
        for _ in range(10):
            self.tick([_view(7, B1_CENTRE, stationary=False)])
        self.assertFalse(self.occupied("B1"))


# ══ TEST C. 한 프레임짜리 오검출 ════════════════════════════════════════════

class SingleFrameDetectionDoesNotConfirm(_Base):

    def test_one_frame_is_not_enough(self) -> None:
        self.tick([_view(7, B1_CENTRE)])
        self.assertFalse(self.occupied("B1"))
        self.assertTrue(self.selectable("B1"))
        self.assertNotIn("VISION_SLOT_OCCUPIED", self.names())

    def test_confirm_needs_consecutive_observations(self) -> None:
        confirm = int(self.pipe.config.vision_occupancy_confirm_observations)
        self.assertGreaterEqual(confirm, 2, "1프레임 확정은 계약 위반")
        for i in range(1, confirm):
            self.tick([_view(7, B1_CENTRE)])
            self.assertFalse(self.occupied("B1"), f"{i}번째 관측에서 확정됨")
        self.tick([_view(7, B1_CENTRE)])
        self.assertTrue(self.occupied("B1"))

    def test_an_interrupted_streak_restarts_the_count(self) -> None:
        self.tick([_view(7, B1_CENTRE)])
        self.tick([])                        # 끊김
        self.tick([_view(7, B1_CENTRE)])
        self.assertFalse(self.occupied("B1"))


# ══ TEST D. 짧은 bbox 유실 ══════════════════════════════════════════════════

class ShortDropoutKeepsOccupancy(_Base):

    def _confirm(self) -> None:
        for _ in range(5):
            self.tick([_view(7, B1_CENTRE)])
        self.assertTrue(self.occupied("B1"))

    def test_one_or_two_missing_frames_do_not_free_the_slot(self) -> None:
        self._confirm()
        self.tick([])                        # 1프레임 유실
        self.assertTrue(self.occupied("B1"))
        self.tick([])                        # 2프레임 유실
        self.assertTrue(self.occupied("B1"))
        self.assertFalse(self.selectable("B1"))
        self.assertNotIn("VISION_SLOT_CLEARED", self.names())

    def test_it_recovers_when_the_bbox_comes_back(self) -> None:
        self._confirm()
        self.tick([])
        self.tick([_view(7, B1_CENTRE)])
        self.assertTrue(self.occupied("B1"))


# ══ TEST E. 차를 실제로 치웠다 ══════════════════════════════════════════════

class RemovingTheVehicleClearsOccupancy(_Base):

    def test_occupancy_clears_after_the_release_grace(self) -> None:
        for _ in range(5):
            self.tick([_view(7, B1_CENTRE)])
        self.assertTrue(self.occupied("B1"))
        release = float(self.pipe.config.vision_occupancy_release_s)
        # 유예 시간 직전까지는 유지된다.
        self.tick([], dt=release * 0.5)
        self.assertTrue(self.occupied("B1"))
        # 유예를 넘기면 풀린다 — 영구 점유로 굳으면 안 된다.
        self.tick([], dt=release)
        self.assertFalse(self.occupied("B1"))
        self.assertTrue(self.selectable("B1"))
        self.assertIn("VISION_SLOT_CLEARED", self.names())

    def test_set_and_clear_are_asymmetric(self) -> None:
        """SET 은 연속 관측, CLEAR 는 시간 유예 — 서로 다른 기준이어야 한다."""
        confirm = int(self.pipe.config.vision_occupancy_confirm_observations)
        release = float(self.pipe.config.vision_occupancy_release_s)
        self.assertGreater(release, confirm * 0.25,
                           "해제가 확정보다 민감하면 깜빡임이 생긴다")


# ══ TEST F. CAR_01 자기 자신 오인 방지 ══════════════════════════════════════

class EgoVehicleNeverCreatesVisionOccupancy(_Base):

    def test_a_bound_car_inside_its_own_slot_is_not_external(self) -> None:
        ego = _view(1, B1_CENTRE, car_id=1)   # CAR_01, ESP 연결됨
        for _ in range(12):
            self.tick([ego])
        self.assertFalse(self.occupied("B1"),
                         "CAR_01 이 자기 자신 때문에 칸을 잠갔다")
        self.assertNotIn("VISION_SLOT_OCCUPIED", self.names())

    def test_ego_passing_through_other_slots_creates_nothing(self) -> None:
        ego_id = 1
        for slot in ("A1", "A2", "B1", "B2"):
            spec = default_slot_specs()[slot]
            for _ in range(6):
                self.tick([_view(ego_id, (spec.center_x, spec.center_y),
                                 car_id=1)])
        self.assertEqual(
            [n for n in self.names() if n.startswith("VISION_SLOT")], [])

    def test_car02_and_car01_together(self) -> None:
        """CAR_02 는 B1 을 잠그고, CAR_01 은 아무것도 잠그지 않는다."""
        car02 = _view(7, B1_CENTRE)                  # unbound, 정지
        ego = _view(1, A2_CENTRE, car_id=1)          # bound
        for _ in range(6):
            self.tick([car02, ego])
        self.assertTrue(self.occupied("B1"))
        self.assertFalse(self.occupied("A2"))
        self.assertFalse(self.selectable("B1"))
        self.assertTrue(self.selectable("A2"))


# ══ TEST G. 기존 예약을 vision 이 풀면 안 된다 ══════════════════════════════

class ReservationIsNeverClearedByVision(_Base):

    def test_an_allocator_reservation_survives_an_empty_frame(self) -> None:
        self.pipe.allocator.set_slot_occupied("A2", True)     # allocate() 가 하는 일
        for _ in range(20):
            self.tick([])                                     # 아무 차도 안 보임
        self.assertGreaterEqual(
            self.pipe.allocator.slot_statuses[SLOT_NAMES.index("A2")], 0.5,
            "vision clear 가 예약을 지웠다")
        self.assertFalse(self.selectable("A2"))

    def test_vision_clear_only_touches_the_overlay(self) -> None:
        self.pipe.allocator.set_slot_occupied("B1", True)     # 예약
        for _ in range(5):
            self.tick([_view(7, B1_CENTRE)])                  # vision 도 점유
        self.assertTrue(self.occupied("B1"))
        release = float(self.pipe.config.vision_occupancy_release_s)
        self.tick([], dt=release * 2)                         # 차를 치움
        self.assertFalse(self.occupied("B1"))                 # overlay 는 풀림
        self.assertGreaterEqual(                              # base 는 그대로
            self.pipe.allocator.slot_statuses[SLOT_NAMES.index("B1")], 0.5)
        self.assertFalse(self.selectable("B1"))

    def test_base_and_overlay_are_separate_arrays(self) -> None:
        a = RealtimeAllocator.__new__(RealtimeAllocator)
        a.slot_statuses = np.zeros(len(SLOT_NAMES), dtype=np.float32)
        a.vision_occupied = np.zeros(len(SLOT_NAMES), dtype=np.float32)
        a.set_vision_occupied("B1", True)
        self.assertEqual(a.slot_statuses[SLOT_NAMES.index("B1")], 0.0)
        a.set_slot_occupied("B1", False)
        self.assertEqual(a.vision_occupied[SLOT_NAMES.index("B1")], 1.0)


# ══ TEST H. 정상 PARKED 는 vision dropout 과 무관하게 유지 ══════════════════

class ParkedStateSurvivesVisionDropout(_Base):

    def test_a_parked_slot_stays_unavailable_without_any_detection(self) -> None:
        self.pipe.allocator.set_slot_occupied("A2", True)     # _on_parked() 결과
        for _ in range(30):
            self.tick([])
        self.assertFalse(self.selectable("A2"))

    def test_vision_occupancy_on_a_parked_slot_is_idempotent(self) -> None:
        self.pipe.allocator.set_slot_occupied("B1", True)
        for _ in range(6):
            self.tick([_view(7, B1_CENTRE)])
        idx = SLOT_NAMES.index("B1")
        self.assertEqual(
            float(self.pipe.allocator.effective_slot_statuses[idx]), 1.0)
        self.assertFalse(self.selectable("B1"))


# ══ TEST I. track_id 가 바뀌어도 점유가 유지된다 ════════════════════════════

class OccupancyIsSlotCentricNotTrackCentric(_Base):

    def test_a_new_track_id_in_the_same_slot_keeps_it_occupied(self) -> None:
        for _ in range(5):
            self.tick([_view(4, B1_CENTRE)])
        self.assertTrue(self.occupied("B1"))
        # 추적기가 track 4 를 잃고 track 9 로 다시 잡았다.
        self.tick([_view(9, B1_CENTRE)])
        self.tick([_view(9, B1_CENTRE)])
        self.assertTrue(self.occupied("B1"))
        self.assertEqual(self.names().count("VISION_SLOT_CLEARED"), 0)

    def test_a_track_id_change_can_still_confirm_from_scratch(self) -> None:
        self.tick([_view(4, B1_CENTRE)])
        self.tick([_view(9, B1_CENTRE)])     # id 만 바뀌고 같은 자리
        self.tick([_view(9, B1_CENTRE)])
        self.assertTrue(self.occupied("B1"),
                        "슬롯 중심 상태라면 id 변경이 확정을 막지 않는다")


# ══ 안전 스위치 / 회귀 ══════════════════════════════════════════════════════

class SafetyAndRegression(_Base):

    def test_the_feature_can_be_turned_off_completely(self) -> None:
        self.pipe.config.vision_occupancy_enabled = False
        for _ in range(20):
            self.tick([_view(7, B1_CENTRE)])
        self.assertFalse(self.occupied("B1"))
        self.assertTrue(self.selectable("B1"))
        self.assertEqual(self.names(), [])

    def test_vision_occupancy_creates_no_mission_and_no_binding(self) -> None:
        """CAR_02 때문에 미션/세션/제어 대상이 생기면 안 된다 (§24)."""
        car02 = _view(7, B1_CENTRE)
        for _ in range(8):
            self.tick([car02])
        self.assertTrue(self.occupied("B1"))
        self.assertIsNone(car02.car_id)
        self.assertEqual(self.pipe.orchestrator.missions, {})
        self.assertEqual(self.pipe.auto_hosts, {})
        self.assertEqual(self.pipe.track_of_car, {})

    def test_effective_status_is_base_or_vision(self) -> None:
        a = self.pipe.allocator
        a.set_slot_occupied("A1", True)
        a.set_vision_occupied("B4", True)
        eff = a.effective_slot_statuses
        self.assertEqual(float(eff[SLOT_NAMES.index("A1")]), 1.0)
        self.assertEqual(float(eff[SLOT_NAMES.index("B4")]), 1.0)
        self.assertEqual(float(eff[SLOT_NAMES.index("A3")]), 0.0)

    def test_the_state_change_is_logged_once_not_every_frame(self) -> None:
        for _ in range(20):
            self.tick([_view(7, B1_CENTRE)])
        self.assertEqual(self.names().count("VISION_SLOT_OCCUPIED"), 1)


if __name__ == "__main__":
    unittest.main()
