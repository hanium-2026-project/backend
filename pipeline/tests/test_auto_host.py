"""AUTO_HOST 경로 결선 검증 (하드웨어팀 host_autonomous_control 패키지).

제어 수학은 controller/tests 가, 권한·미션 상태기계는 host_control/tests 가 본다.
여기서 보는 건 backend 와 붙는 지점이다:

- SET_MODE REMOTE_DIRECT ACCEPTED 전에는 제어값이 나가지 않는다
- AUTO_HOST 동안 WAYPOINT/GO wire 전송이 0회다
- 카메라가 멈추면 스트림이 0 으로 떨어진다 (마지막 값이 남아 계속 달리면 안 된다)
- **최종 도착 → 정지 확인 → 슬롯 점유** 가 이어진다 (WAYPOINT_AUTO 의 PARKED_CHECK 대체)

실행: python -m unittest pipeline.tests.test_auto_host -v
"""

from __future__ import annotations

import time
import unittest

from comm.tests.mock_firmware import MockFirmware
from control.auto_host_runner import MissionStatus
from cv.tracker import TrackState
from pipeline import ParkingPipeline, PipelineConfig
from pipeline.tests.test_pipeline_integration import detection_at, wait_until
from rl.parking_env import SLOT_NAMES

FRAME = 1200


class AutoHostTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig(
            server_port=0,
            lot_width_mm=FRAME, lot_height_mm=FRAME,
            policy_path="models/sb3_parking_policy.zip",
            stationary_window=3,
            control_mode="auto-host",
            direct_control=True,
            auto_host_period_s=0.02,          # 테스트는 빠르게 돌린다
            auto_host_handshake_s=2.0,
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
        self.assertTrue(wait_until(lambda: esp.state == "READY"), "READY 미도달")
        self.esps.append(esp)
        return esp

    def feed(self, *positions, settle: float = 0.08) -> None:
        self.frame_no += 1
        dets = [detection_at(pos, tid) for tid, pos in positions]
        self.pipeline.on_frame(TrackState(
            frame_index=self.frame_no, timestamp=time.monotonic(),
            detections=dets, fps=30.0, frame_size=(FRAME, FRAME),
        ))
        time.sleep(settle)

    def runner(self, car_id: int = 1):
        return self.pipeline.auto_hosts.get(car_id)

    def wait_slot(self, car_id: int = 1, tries: int = 25) -> str | None:
        """슬롯 배정까지 프레임을 계속 먹인다.

        러너 존재만으로는 부족하다 — READY 직후 수동 셸이 먼저 러너를 만들고,
        슬롯 배정(=자동 주행 시작)은 그 뒤 프레임에서 일어난다.
        """
        for _ in range(tries):
            slot = self.pipeline._auto_host_slot.get(car_id)
            if slot:
                return slot
            self.feed((7, (150.0, 100.0)), settle=0.1)
        return None


class TestHandshake(AutoHostTestBase):
    def test_arms_only_after_set_mode_accepted(self):
        esp = self.connect()
        self.feed((7, (150.0, 100.0)))            # 진입 → 슬롯 배정 → AUTO_HOST 시작

        self.assertTrue(wait_until(lambda: self.runner() is not None),
                        "AUTO_HOST 러너가 뜨지 않음")
        self.assertEqual(esp.mode, "REMOTE_DIRECT", "SET_MODE 미적용")
        self.assertTrue(self.runner().session.accepted, "ACCEPTED 확인 없이 arm 됨")
        self.assertTrue(self.pipeline.server.direct_control_enabled)
        self.assertEqual(esp.rejects, [], f"계약 위반: {esp.rejects}")

    def test_no_waypoint_or_go_on_the_wire(self):
        """AUTO_HOST 는 host 내부 waypoint 를 쓴다 — 차량에 경로를 보내지 않는다."""
        esp = self.connect()
        self.feed((7, (150.0, 100.0)))
        self.assertTrue(wait_until(lambda: self.runner() is not None))
        for pos in [(150.0, 140.0), (150.0, 190.0), (150.0, 240.0)]:
            self.feed((7, pos))

        sent = [m["type"] for m in esp.received]
        self.assertNotIn("WAYPOINT", sent, "AUTO_HOST 인데 WAYPOINT 를 보냈다")
        self.assertNotIn("GO", sent, "AUTO_HOST 인데 GO 를 보냈다")
        self.assertIsNone(esp.target, "차량에 target 이 적재됐다")


class TestControlStream(AutoHostTestBase):
    def test_control_reaches_vehicle_with_firmware_sign(self):
        esp = self.connect()
        self.feed((7, (150.0, 100.0)))
        self.assertTrue(wait_until(lambda: self.runner() is not None))
        # 위쪽으로 이동시켜 heading 을 잡는다 (목표도 위쪽이므로 직진 상황)
        for pos in [(150.0, 150.0), (150.0, 210.0), (150.0, 270.0)]:
            self.feed((7, pos))

        self.assertTrue(wait_until(lambda: esp.direct_controls > 0),
                        "DIRECT_CONTROL 미도달")
        last = esp.last_direct_control
        self.assertIsNotNone(last)
        self.assertGreaterEqual(last["throttle"], 0.0)
        self.assertGreaterEqual(last["steering"], -1.0)
        self.assertLessEqual(last["steering"], 1.0)
        self.assertEqual(esp.rejects, [], f"계약 위반: {esp.rejects}")

    def test_stream_drops_to_zero_when_camera_stalls(self):
        """카메라가 멈췄는데 마지막 제어값이 계속 나가면 차가 그대로 달린다."""
        esp = self.connect()
        self.feed((7, (150.0, 100.0)))
        self.assertTrue(wait_until(lambda: self.runner() is not None))
        for pos in [(150.0, 150.0), (150.0, 210.0), (150.0, 270.0)]:
            self.feed((7, pos))

        # 프레임 공급을 끊는다 (카메라 가림 / 탐지 실패)
        self.assertTrue(
            wait_until(lambda: (esp.last_direct_control or {}).get("throttle") == 0.0,
                       timeout=3.0),
            f"stale 인데 제어값이 0 이 아니다: {esp.last_direct_control}")
        self.assertTrue(self.runner().is_faulted,
                        "stale 이면 FAULTED 로 latch 돼야 한다")

    def test_faulted_does_not_auto_restart(self):
        """복구돼도 명시적 re-arm 전에는 다시 출발하지 않는다."""
        self.connect()
        self.feed((7, (150.0, 100.0)))
        self.assertTrue(wait_until(lambda: self.runner() is not None))
        for pos in [(150.0, 150.0), (150.0, 210.0)]:
            self.feed((7, pos))
        self.assertTrue(wait_until(lambda: self.runner().is_faulted, timeout=3.0))

        for pos in [(150.0, 260.0), (150.0, 310.0)]:      # 카메라 복귀
            self.feed((7, pos))
        self.assertTrue(self.runner().is_faulted, "자동 재출발했다")


class TestParkedWiring(AutoHostTestBase):
    """구멍 2번: AUTO_HOST 최종 도착이 슬롯 점유까지 이어지는지."""

    def _drive(self, max_steps: int = 120) -> None:
        """목표 위치로 순간이동시켜 미션을 진행시킨다.

        HW 7fc17c6 부터 route 교체 시 pose_source 를 비우므로(옛 관측으로
        출발하지 않기 위해) 같은 목표에 프레임을 두 번 먹여야 제어기가
        새 pose 를 받고 도착을 판정한다.
        """
        r = self.runner()
        for _ in range(max_steps):
            if r.status in (MissionStatus.DONE, MissionStatus.PARKED,
                            MissionStatus.REPLAN_REQUIRED):
                return
            target = r.current_target
            if target is None:
                return
            self.feed((7, (target.x_mm, target.y_mm)), settle=0.05)
            self.feed((7, (target.x_mm, target.y_mm)), settle=0.05)

    def test_final_arrival_marks_slot_occupied(self):
        self.connect()
        slot_id = self.wait_slot()
        self.assertIsNotNone(slot_id, "슬롯이 배정되지 않음")

        self._drive()
        self.assertIn(self.runner().status,
                      (MissionStatus.DONE, MissionStatus.PARKED,
                       MissionStatus.REPLAN_REQUIRED),
                      f"미완주: {self.runner().status}")

        if self.runner().status is MissionStatus.DONE:
            # 같은 자리를 계속 먹여 정지 판정을 통과시킨다 (§11)
            final = self.pipeline.views[7].position_mm
            for _ in range(self.config.stationary_window + 2):
                self.feed((7, final), settle=0.05)
            self.assertIs(self.runner().status, MissionStatus.PARKED,
                          "정지했는데 PARKED 로 확정되지 않음")
            idx = SLOT_NAMES.index(slot_id)
            self.assertEqual(self.pipeline.allocator.slot_statuses[idx], 1.0,
                             "PARKED 인데 슬롯이 점유로 갱신되지 않음")

    def test_moving_car_is_not_confirmed_parked(self):
        """최종 위치에 있어도 움직이는 중이면 확정하지 않는다."""
        self.connect()
        self.assertIsNotNone(self.wait_slot(), "슬롯이 배정되지 않음")
        self._drive()
        if self.runner().status is not MissionStatus.DONE:
            self.skipTest("DONE 에 도달하지 못한 경로 — 이 검증 대상 아님")
        # 정지 판정 창(stationary_window)을 확실히 깨도록 매 프레임 크게 움직인다.
        # 도착 직전 순간이동으로 같은 좌표가 창에 쌓여 있어, 첫 프레임부터
        # 허용오차를 넘겨야 "움직이는 중"으로 인식된다.
        base = self.pipeline.views[7].position_mm
        for i in range(1, self.config.stationary_window + 4):
            self.feed((7, (base[0] + 60 * i, base[1])), settle=0.05)
        self.assertIsNot(self.runner().status, MissionStatus.PARKED,
                         "움직이는 중인데 PARKED 로 확정됐다")


class TestLegacyPathUntouched(unittest.TestCase):
    def test_waypoint_auto_is_the_default(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.control_mode, "waypoint-auto")
        self.assertFalse(cfg.direct_control)


if __name__ == "__main__":
    unittest.main()
