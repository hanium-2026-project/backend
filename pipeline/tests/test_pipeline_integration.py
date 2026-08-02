"""CV→RL→통신 결선 통합 검증.

실카메라·실가중치 없이 파이프라인 배선을 검증한다. 탐지 결과를 직접
합성해 넣으므로 차량을 원하는 좌표로 정확히 움직일 수 있고, 실영상으로는
도달하지 못하는 구간(FINAL 진입, 2대 충돌)까지 결정론적으로 확인된다.

좌표 규약: 프레임 1200×1200px ↔ 맵 1200×1200mm (1px = 1mm),
이미지 y축은 아래로 증가하므로 맵 y = 1200 - 픽셀 y.

실행: python -m unittest pipeline.tests.test_pipeline_integration -v
"""

from __future__ import annotations

import logging
import time
import unittest

from comm import MissionState
from comm.tests.mock_firmware import MockFirmware
from cv.tracker import TrackState
from cv.vehicle_detector import Detection
from parking.waypoints import default_slot_specs
from pipeline import ParkingPipeline, PipelineConfig

FRAME = 1200
CAR_PX = 60          # 화면상 차량 크기 (px = mm)


def wait_until(cond, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def detection_at(map_xy: tuple[float, float], track_id: int,
                 confidence: float = 0.9) -> Detection:
    """맵 좌표(mm)에 차량이 있는 것처럼 보이는 bbox 를 만든다."""
    mx, my = map_xy
    px, py = mx, FRAME - my                      # 맵 → 픽셀 (y 반전)
    half = CAR_PX / 2
    return Detection(
        label="rc_car", confidence=confidence, track_id=track_id,
        bbox=(int(px - half), int(py - half), int(px + half), int(py + half)),
    )


class PipelineTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 충돌 경고 등 정상 동작 로그가 테스트 출력을 가리지 않게 한다
        logging.getLogger("pipeline.runner").setLevel(logging.CRITICAL)

    def setUp(self) -> None:
        self.config = PipelineConfig(
            server_port=0,
            lot_width_mm=FRAME, lot_height_mm=FRAME,
            policy_path="models/sb3_parking_policy.zip",
            stationary_window=3,
        )
        self.pipeline = ParkingPipeline(self.config)
        self.pipeline.start()
        self.frame_no = 0
        self.esps: list[MockFirmware] = []

    def tearDown(self) -> None:
        for e in self.esps:
            e.close()
        self.pipeline.stop()

    def connect(self, car_id: int) -> MockFirmware:
        esp = MockFirmware(self.pipeline.server.bound_port,
                           car_id=f"CAR_{car_id:02d}", boot_id=f"B{car_id:07d}")
        wait_until(lambda: esp.hello_result is not None)
        self.esps.append(esp)
        return esp

    def feed(self, *positions: tuple[int, tuple[float, float]], settle: float = 0.12):
        """(track_id, 맵좌표) 들을 한 프레임으로 밀어넣는다."""
        self.frame_no += 1
        dets = [detection_at(pos, tid) for tid, pos in positions]
        self.pipeline.on_frame(TrackState(
            frame_index=self.frame_no, timestamp=time.monotonic(),
            detections=dets, fps=30.0, frame_size=(FRAME, FRAME),
        ))
        time.sleep(settle)

    def ensure_two_missions(self, max_steps: int = 30) -> None:
        """두 차량 모두 슬롯을 배정받아 미션이 생길 때까지 진행시킨다.

        RL 은 선행 차량이 통로를 비울 때까지 후속 차량에 WAIT 을 주므로,
        car1 을 경로대로 움직이며 car2 의 배정을 기다린다.
        """
        orch = self.pipeline.orchestrator
        self.feed((7, (150.0, 100.0)))
        self.feed((8, (150.0, 100.0)))
        for _ in range(max_steps):
            if 1 in orch.missions and 2 in orch.missions:
                return
            m1 = orch.missions.get(1)
            wp = m1.current if m1 else None
            if wp is not None:
                self.feed((7, (wp.x, wp.y)), (8, (150.0, 100.0)))
            else:
                self.feed((8, (150.0, 100.0)))
        self.fail(f"두 차량 미션 생성 실패: {sorted(orch.missions)}")

    def drive_mission(self, car_id: int, track_id: int,
                      max_steps: int = 40) -> MissionState:
        """현재 목표 waypoint 좌표로 차량을 순차 이동시켜 미션을 완주한다."""
        orch = self.pipeline.orchestrator
        for _ in range(max_steps):
            m = orch.missions.get(car_id)
            if m is None or m.state is MissionState.DONE:
                break
            wp = m.current
            if wp is None:
                break
            self.feed((track_id, (wp.x, wp.y)))
        return orch.missions[car_id].state


class TestSingleVehicle(PipelineTestBase):
    def test_entry_triggers_allocation_and_mission(self):
        """진입 노드 등장 → car 매핑 → RL 슬롯 배정 → waypoint 전송."""
        esp = self.connect(1)
        self.assertEqual(self.pipeline._pending_cars, [1])

        self.feed((7, (150.0, 100.0)))            # entrance 위치

        view = self.pipeline.views[7]
        self.assertEqual(view.car_id, 1, "track↔car 매핑 실패")
        self.assertEqual(view.node, "entrance")
        self.assertIsNotNone(view.slot_id, "슬롯 미배정")
        self.assertIn(1, self.pipeline.orchestrator.missions)
        self.assertTrue(wait_until(lambda: esp.target is not None),
                        f"WAYPOINT 미도달: {esp.rejects}")
        self.assertEqual(esp.rejects, [], "계약 위반 발생")

    def test_full_drive_to_parked(self):
        """전 구간 주행 → FINAL → 정지 확인 → PARKED, 슬롯 점유."""
        esp = self.connect(1)
        self.feed((7, (150.0, 100.0)))
        slot_id = self.pipeline.views[7].slot_id

        state = self.drive_mission(car_id=1, track_id=7)
        self.assertIs(state, MissionState.DONE, f"미완주 (rejects={esp.rejects})")
        self.assertEqual(esp.rejects, [])

        idx = list(default_slot_specs()).index(slot_id)
        self.assertEqual(self.pipeline.allocator.slot_statuses[idx], 1.0,
                         "PARKED 후 슬롯이 점유로 갱신되지 않음")

    def test_parked_requires_stationary(self):
        """움직이는 중에는 FINAL 위치에 있어도 PARKED 로 확정하지 않는다 (§11)."""
        self.connect(1)
        self.feed((7, (150.0, 100.0)))
        orch = self.pipeline.orchestrator
        m = orch.missions[1]

        # FINAL 직전까지 주행
        for _ in range(40):
            wp = m.current
            if wp is None or wp.is_final:
                break
            self.feed((7, (wp.x, wp.y)))

        final = m.current
        self.assertTrue(final.is_final)
        # 슬롯 방향(heading)을 유지한 채 FINAL 로 접근 — 위치·방향은 맞지만
        # 계속 이동 중이므로 PARKED 로 확정되면 안 된다
        approach = -1 if final.target_heading_deg == 90.0 else 1
        for d in (120, 80, 40, 0):
            self.feed((7, (final.x, final.y + approach * d)))
        self.assertIsNot(m.state, MissionState.DONE, "움직이는데 PARKED 확정됨")

        for _ in range(4):                      # 같은 자리 유지 → 정지 성립
            self.feed((7, (final.x, final.y)))
        self.assertIs(m.state, MissionState.DONE, "정지 후에도 PARKED 미확정")

    def test_heading_flows_into_pose(self):
        """궤적 heading 이 계산되어 차량 관측에 반영된다."""
        self.connect(1)
        self.feed((7, (150.0, 100.0)))
        for y in (150.0, 250.0, 350.0, 450.0):   # 위로 이동 → 90°
            self.feed((7, (150.0, y)))
        view = self.pipeline.views[7]
        self.assertIsNotNone(view.heading_deg)
        self.assertEqual(view.heading_source, "TRAJECTORY")
        self.assertAlmostEqual(view.heading_deg, 90.0, delta=15.0)


class TestMultiVehicle(PipelineTestBase):
    def test_two_vehicles_get_different_slots(self):
        """두 대 모두 매핑되고, 서로 다른 슬롯을 받는다.

        RL 은 혼잡 시 WAIT(미배정)을 반환할 수 있으므로, 진입 노드에 머무는
        동안 재시도되어 결국 배정되는지까지 확인한다.
        """
        self.connect(1)
        self.connect(2)
        self.ensure_two_missions()

        v1, v2 = self.pipeline.views[7], self.pipeline.views[8]
        self.assertEqual({v1.car_id, v2.car_id}, {1, 2}, "car 매핑 중복/누락")
        self.assertIsNotNone(v1.slot_id)
        self.assertIsNotNone(v2.slot_id, "재시도했는데도 끝내 슬롯 미배정")
        self.assertNotEqual(v1.slot_id, v2.slot_id, "같은 슬롯에 두 대 배정")

    def test_collision_triggers_hold(self):
        """두 차량이 안전거리 안으로 접근하면 한 대가 WAIT 한다."""
        e1 = self.connect(1)
        e2 = self.connect(2)
        self.ensure_two_missions()
        orch = self.pipeline.orchestrator

        # 두 차량을 안전거리(350mm) 안으로 접근시킨다
        for _ in range(3):
            self.feed((7, (600.0, 600.0)), (8, (700.0, 600.0)))

        held = [cid for cid, m in orch.missions.items() if m.state is MissionState.HELD]
        self.assertTrue(held, "근접했는데 hold 되지 않음")
        self.assertEqual(len(held), 1, "두 대가 모두 정지 — 교착")
        stopped_esp = e1 if held[0] == 1 else e2
        self.assertTrue(wait_until(lambda: stopped_esp.state == "WAITING"),
                        "WAIT 명령이 차량에 반영되지 않음")
        self.assertEqual(stopped_esp.wait_reason, "COLLISION_RISK")

    def test_hold_then_resume_when_cleared(self):
        """위험이 해소되면 정지시킨 차량을 자동 재개한다."""
        self.connect(1)
        self.connect(2)
        self.ensure_two_missions()
        orch = self.pipeline.orchestrator

        for _ in range(3):                       # 근접 → 한 대 정지
            self.feed((7, (600.0, 600.0)), (8, (700.0, 600.0)))
        held = [c for c, m in orch.missions.items() if m.state is MissionState.HELD]
        self.assertEqual(len(held), 1, f"정지 대상이 1대가 아님: {held}")
        stopped = held[0]

        for _ in range(3):                       # 충분히 벌어짐 → 재개
            self.feed((7, (300.0, 300.0)), (8, (1000.0, 900.0)))
        self.assertIsNot(orch.missions[stopped].state, MissionState.HELD,
                         "위험이 해소됐는데 재개되지 않음")
        self.assertNotIn(stopped, self.pipeline._collision_held)


class TestRecovery(PipelineTestBase):
    def test_reconnect_discards_route(self):
        """재접속 시 기존 경로를 폐기하고 자동 재개하지 않는다 (§21)."""
        self.connect(1)
        self.feed((7, (150.0, 100.0)))
        self.assertIn(1, self.pipeline.orchestrator.missions)

        self.esps[0].close()
        esp2 = self.connect(1)                   # 같은 car_id 재접속
        self.assertTrue(wait_until(lambda: 1 not in self.pipeline.orchestrator.missions),
                        "재접속 후에도 기존 미션이 남음")
        self.assertIsNone(self.pipeline.views[7].car_id, "매핑이 해제되지 않음")
        self.assertEqual(esp2.hello_result, "READY_ALLOWED")

    def test_rejected_command_triggers_replan(self):
        """STALE_ROUTE 계열 거절 → 현재 pose 기준 새 route 로 재생성."""
        self.connect(1)
        self.feed((7, (150.0, 100.0)))
        orch = self.pipeline.orchestrator
        before = orch.missions[1].route_id

        orch.on_command_rejected(1, "STALE_ROUTE", {})
        # 거절 → HELD 후 파이프라인이 즉시 재계획하므로 새 route 로드 상태가 된다
        self.assertTrue(wait_until(lambda: orch.missions[1].route_id != before),
                        "route_id 가 증가하지 않음 (재생성 누락)")
        self.assertIn(orch.missions[1].state,
                      (MissionState.HELD, MissionState.LOADING, MissionState.DRIVING))
        self.assertGreater(orch.missions[1].route_id, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
