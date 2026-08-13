"""B안 제어 결선 검증 — 카메라 pose → 제어값 → DIRECT_CONTROL 송신.

여기서 확인하는 것은 "제어 알고리즘이 옳은가"(control/tests 담당)가 아니라
파이프라인 배선이다: 값이 실제로 계산되어 차량까지 도달하는가, 그리고
주행이 허용되지 않는 구간에서 0 이 나가는가.

실행: python -m unittest pipeline.tests.test_direct_control -v
"""

from __future__ import annotations

import time
import unittest

from comm import MissionState
from comm.tests.mock_firmware import MockFirmware
from control import VehicleLimits
from cv.tracker import TrackState
from pipeline import ParkingPipeline, PipelineConfig
from pipeline.tests.test_pipeline_integration import detection_at, wait_until

FRAME = 1200


class DirectControlTestBase(unittest.TestCase):
    direct_control = True

    def setUp(self) -> None:
        self.config = PipelineConfig(
            server_port=0,
            lot_width_mm=FRAME, lot_height_mm=FRAME,
            policy_path="models/sb3_parking_policy.zip",
            stationary_window=3,
            direct_control=self.direct_control,
            vehicle_limits=VehicleLimits(),
        )
        self.pipeline = ParkingPipeline(self.config)
        self.pipeline.start()
        self.frame_no = 0
        self.esps: list[MockFirmware] = []

    def tearDown(self) -> None:
        for e in self.esps:
            e.close()
        self.pipeline.stop()

    def connect(self, car_id: int = 1) -> MockFirmware:
        esp = MockFirmware(self.pipeline.server.bound_port,
                           car_id=f"CAR_{car_id:02d}", boot_id=f"B{car_id:07d}")
        wait_until(lambda: esp.hello_result is not None)
        self.esps.append(esp)
        return esp

    def feed(self, *positions, settle: float = 0.12) -> None:
        self.frame_no += 1
        dets = [detection_at(pos, tid) for tid, pos in positions]
        self.pipeline.on_frame(TrackState(
            frame_index=self.frame_no, timestamp=time.monotonic(),
            detections=dets, fps=30.0, frame_size=(FRAME, FRAME),
        ))
        time.sleep(settle)


class TestControlStream(DirectControlTestBase):
    def test_mode_set_to_remote_direct_on_ready(self):
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.mode == "REMOTE_DIRECT"),
                        f"SET_MODE 미도달 (mode={esp.mode}, rejects={esp.rejects})")

    def test_control_reaches_vehicle_while_driving(self):
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.state == "READY"))
        self.feed((7, (150.0, 600.0)))            # 진입 → 슬롯 배정 → 미션 시작

        orch = self.pipeline.orchestrator
        self.assertTrue(wait_until(
            lambda: orch.missions.get(1) is not None
            and orch.missions[1].state is MissionState.DRIVING),
            "미션이 DRIVING 에 도달하지 못함")

        # 목표에서 충분히 떨어진 위치를 몇 프레임 먹여 heading 을 잡는다
        for pos in [(150.0, 120.0), (150.0, 160.0), (150.0, 210.0)]:
            self.feed((7, pos))

        out = self.pipeline.last_control.get(1)
        self.assertIsNotNone(out, "제어값이 계산되지 않음")
        self.assertEqual(out.mode, "DRIVE", f"주행 중인데 {out.mode} ({out.reason})")
        self.assertGreater(out.throttle, 0.0)

        self.assertTrue(wait_until(lambda: esp.direct_controls > 0),
                        "DIRECT_CONTROL 이 차량에 도달하지 않음")
        last = esp.last_direct_control
        self.assertIsNotNone(last)
        self.assertGreater(last["throttle"], 0.0)
        self.assertEqual(esp.rejects, [], f"계약 위반: {esp.rejects}")

    def test_control_seq_increases(self):
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.state == "READY"))
        self.feed((7, (150.0, 600.0)))
        for pos in [(150.0, 140.0), (150.0, 190.0), (150.0, 240.0)]:
            self.feed((7, pos))
        self.assertTrue(wait_until(lambda: esp.direct_controls >= 2))
        self.assertGreaterEqual(len(esp.control_seqs), 2)
        self.assertEqual(esp.control_seqs, sorted(esp.control_seqs),
                         "control_seq 가 단조 증가하지 않는다")

    def test_no_heading_yields_zero_throttle(self):
        """heading 을 모르는 첫 프레임에서 차를 밀면 안 된다."""
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.state == "READY"))
        self.feed((7, (150.0, 600.0)))
        out = self.pipeline.last_control.get(1)
        if out is not None and self.pipeline.views[7].heading_deg is None:
            self.assertEqual(out.throttle, 0.0)
            self.assertEqual(out.reason, "NO_HEADING")

    def test_zero_control_when_mission_not_driving(self):
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.state == "READY"))
        self.feed((7, (150.0, 600.0)))
        orch = self.pipeline.orchestrator
        self.assertTrue(wait_until(lambda: 1 in orch.missions))

        for pos in [(150.0, 140.0), (150.0, 190.0)]:
            self.feed((7, pos))
        orch.missions[1].state = MissionState.HELD      # 충돌 회피 등으로 정지
        self.feed((7, (150.0, 240.0)))

        out = self.pipeline.last_control[1]
        self.assertEqual(out.throttle, 0.0, "정지 상태인데 구동값이 나간다")
        self.assertEqual(out.mode, "HOLD")

    def test_controller_reset_on_resync(self):
        """재접속하면 제어기 상태를 버린다 (옛 오차로 조향이 튀지 않게)."""
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.state == "READY"))
        self.feed((7, (150.0, 600.0)))
        self.feed((7, (150.0, 160.0)))
        self.assertIn(1, self.pipeline.controllers)

        self.pipeline._on_resync(1, {"boot_id": "B0000002"})
        self.assertNotIn(1, self.pipeline.controllers)
        self.assertNotIn(1, self.pipeline.last_control)


class TestControlDisabledByDefault(DirectControlTestBase):
    direct_control = False

    def test_no_control_stream_when_disabled(self):
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.state == "READY"))
        self.feed((7, (150.0, 600.0)))
        for pos in [(150.0, 140.0), (150.0, 190.0), (150.0, 240.0)]:
            self.feed((7, pos))
        time.sleep(0.3)
        self.assertEqual(esp.direct_controls, 0,
                         "direct_control 이 꺼져 있는데 제어값이 나갔다")
        self.assertEqual(self.pipeline.last_control, {})
        self.assertNotEqual(esp.mode, "REMOTE_DIRECT",
                            "꺼져 있으면 모드도 바꾸지 않는다")


if __name__ == "__main__":
    unittest.main()
