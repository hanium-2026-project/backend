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
from unittest.mock import patch

from comm.tests.mock_firmware import MockFirmware
from control.auto_host_runner import MissionStatus
from cv.tracker import TrackState
from pipeline import ParkingPipeline, PipelineConfig
from pipeline.tests.test_pipeline_integration import (
    detection_at, detections_with_heading, wait_until,
)
from rl.parking_env import SLOT_NAMES

FRAME = 1200


class AutoHostTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.policy_patch = patch(
            "rl.bridge.select_action",
            side_effect=lambda obs, masks, **kw: next(
                i for i, allowed in enumerate(masks[:8]) if allowed))
        self.policy_patch.start()
        self.config = PipelineConfig(
            server_port=0,
            lot_width_mm=FRAME, lot_height_mm=FRAME,
            policy_path="models/sb3_parking_policy.zip",
            stationary_window=3,
            initial_pose_observations=1,
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
        self.policy_patch.stop()

    def connect(self, car_id: int = 1) -> MockFirmware:
        esp = MockFirmware(self.pipeline.server.bound_port,
                           car_id=f"CAR_{car_id:02d}", boot_id=f"B{car_id:07d}")
        self.assertTrue(wait_until(lambda: esp.state == "READY"), "READY 미도달")
        self.esps.append(esp)
        return esp

    def feed(self, *positions, settle: float = 0.08) -> None:
        self.frame_no += 1
        dets = [det for tid, pos in positions
                for det in detections_with_heading(pos, tid)]
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
            self.feed((7, (150.0, 600.0)), settle=0.1)
        return None


class TestHandshake(AutoHostTestBase):
    def test_arms_only_after_set_mode_accepted(self):
        esp = self.connect()
        self.feed((7, (150.0, 600.0)))            # 진입 → 슬롯 배정 → AUTO_HOST 시작

        self.assertTrue(wait_until(lambda: self.runner() is not None),
                        "AUTO_HOST 러너가 뜨지 않음")
        self.assertEqual(esp.mode, "REMOTE_DIRECT", "SET_MODE 미적용")
        self.assertTrue(self.runner().session.accepted, "ACCEPTED 확인 없이 arm 됨")
        self.assertTrue(self.pipeline.server.direct_control_enabled)
        self.assertEqual(esp.rejects, [], f"계약 위반: {esp.rejects}")

    def test_no_waypoint_or_go_on_the_wire(self):
        """AUTO_HOST 는 host 내부 waypoint 를 쓴다 — 차량에 경로를 보내지 않는다."""
        esp = self.connect()
        self.feed((7, (150.0, 600.0)))
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
        self.feed((7, (150.0, 600.0)))
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
        # 출발 지점에 머무는 동안 슬롯을 받아야 한다. 진입 우회전 반경은
        # 출발 y 가 정하므로(R = 통로y − y), 차가 위로 올라간 뒤에는 어떤
        # 슬롯으로도 경로를 만들 수 없다.
        self.assertIsNotNone(self.wait_slot(), "슬롯이 배정되지 않음")
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
        self.assertIsNotNone(self.wait_slot(), "슬롯이 배정되지 않음")
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


class TestPathDeviation(AutoHostTestBase):
    """경로를 벗어나 목표를 전진으로 못 잡으면 후진 복구가 걸린다.

    기존 트리거(APPROACH 놓침 / ALIGN 방향 불일치)는 주차 단계에서만 돌아서,
    통로·진입 원호 구간에서 밀려나면 아무것도 걸리지 않고 계속 앞으로만 갔다.
    """

    def feed_pair(self, car, cushion, settle: float = 0.06) -> None:
        """차체 + 전방 쿠션 → heading_source=FRONT_CUSHION (후진 허용 조건)."""
        from cv.vehicle_detector import Detection
        self.frame_no += 1

        def det(pos, label, tid):
            px, py = pos[0], FRAME - pos[1]
            return Detection(label=label, confidence=0.9, track_id=tid,
                             bbox=(int(px - 30), int(py - 30),
                                   int(px + 30), int(py + 30)))
        self.pipeline.on_frame(TrackState(
            frame_index=self.frame_no, timestamp=time.monotonic(),
            detections=[det(car, "rc_car", 7), det(cushion, "front_cushion", 8)],
            fps=30.0, frame_size=(FRAME, FRAME)))
        time.sleep(settle)

    def arm(self):
        self.connect()
        for _ in range(25):
            if self.pipeline._auto_host_slot.get(1):
                return self.pipeline.auto_hosts[1]
            self.feed_pair((150.0, 600.0), (210.0, 600.0), settle=0.08)
        self.fail("슬롯 배정 실패")

    def test_overshoot_triggers_reverse_then_resumes(self):
        r = self.arm()
        target = r.current_target
        past = [target.x_mm + 250.0, 600.0]          # 목표를 한참 지나침

        for _ in range(6):
            self.feed_pair(tuple(past), (past[0] + 60, past[1]))
            cur = r.current_target
            if cur is not None and cur.motion_direction.value == "REVERSE":
                break
        else:
            self.fail("경로를 벗어났는데 후진 복구가 걸리지 않았다")

        self.assertEqual(r.mission.recovery_attempts, 1)
        self.assertEqual(r.current_target.phase, "RECOVERY")

        # 차가 실제로 물러나면 원래 경로로 복귀해야 한다
        for _ in range(6):
            self.feed_pair(tuple(past), (past[0] + 60, past[1]))
            past[0] -= 90.0
            cur = r.current_target
            if cur is not None and cur.motion_direction.value == "FORWARD":
                break
        else:
            self.fail("후진을 마쳤는데 원래 경로로 복귀하지 않았다")
        self.assertEqual(r.mission.recovery_attempts, 1, "복구가 중복으로 걸렸다")

    def test_reverse_waypoint_does_not_retrigger_itself(self):
        """복구 waypoint 는 정의상 등 뒤에 있다 — 그걸 이탈로 잡으면 안 된다."""
        r = self.arm()
        past = (r.current_target.x_mm + 250.0, 600.0)
        for _ in range(6):
            self.feed_pair(past, (past[0] + 60, past[1]))
            cur = r.current_target
            if cur is not None and cur.motion_direction.value == "REVERSE":
                break
        # 차를 그대로 둔 채 여러 프레임 — 재시도 횟수가 늘면 안 된다
        for _ in range(5):
            self.feed_pair(past, (past[0] + 60, past[1]))
        self.assertLessEqual(r.mission.recovery_attempts, 1)

    def test_on_path_never_triggers(self):
        """경로 위에 있으면 후진이 걸리지 않는다."""
        r = self.arm()
        for _ in range(6):
            t = r.current_target
            if t is None:
                break
            self.feed_pair((t.x_mm - 200.0, 600.0), (t.x_mm - 140.0, 600.0))
        self.assertEqual(r.mission.recovery_attempts, 0)


class TestRecoverPhases(AutoHostTestBase):
    """후진 복구를 거는 phase 는 config 로 정한다 (기본 APPROACH + FINAL)."""

    def test_default_covers_approach_and_final(self):
        from pipeline import PipelineConfig
        self.assertEqual(PipelineConfig().recover_phases, ("APPROACH", "FINAL"))

    def _wp(self, phase: str, is_final: bool = False):
        class W:
            pass
        w = W()
        w.phase, w.is_final = phase, is_final
        return w

    def test_cruise_is_not_recoverable(self):
        """통로 중간은 허용오차가 넓고 다음 점이 이어진다 — 후진하면 진행이 끊긴다."""
        self.assertFalse(self.pipeline._recoverable(self._wp("CRUISE")))

    def test_approach_and_final_are_recoverable(self):
        self.assertTrue(self.pipeline._recoverable(self._wp("APPROACH")))
        self.assertTrue(self.pipeline._recoverable(self._wp("FINAL", True)))

    def test_empty_tuple_disables_recovery(self):
        self.pipeline.config.recover_phases = ()
        self.assertFalse(self.pipeline._recoverable(self._wp("FINAL", True)))


class TestCommRecoveryContract(AutoHostTestBase):
    """COMM loss: zero -> session -> fresh pose -> replan, never stale resume."""

    def _arm_route(self):
        esp = self.connect()
        slot = self.wait_slot()
        self.assertIsNotNone(slot, "초기 AUTO_HOST route 미적재")
        return esp, slot

    def test_175307_comm_before_mission_allows_only_new_validated_activation(self):
        esp = self.connect()
        self.assertTrue(wait_until(lambda: self.runner() is not None),
                        "manual REMOTE_DIRECT shell not ready")
        sess = self.pipeline.server._session(1)

        self.pipeline.server._comm_fail(
            1, {"type": "COMM_TIMEOUT"}, expected_session=sess)
        self.assertTrue(sess.control_held)
        self.assertIsNone(self.pipeline._auto_host_slot.get(1))

        esp.send_periodic_status()
        self.assertTrue(wait_until(
            lambda: (self.pipeline._comm_recovery_context.get(1) or {}).get("state")
                    == "WAIT_FRESH_POSE", timeout=5.0))

        # Mission allocation is forbidden while recovery owns the lifecycle.
        for _ in range(self.config.stationary_window):
            self.feed((7, (150.0, 600.0)), settle=0.08)
            if 1 in self.pipeline._comm_recovery_context:
                self.assertIsNone(self.pipeline._auto_host_slot.get(1))

        self.assertNotIn(1, self.pipeline._comm_recovery_context)
        self.assertTrue(sess.control_held,
                        "session recovery alone released the transport latch")

        slot = self.wait_slot(tries=30)
        self.assertIsNotNone(slot, "fresh safe GLOBAL route was not activated")
        self.assertFalse(sess.control_held)
        self.assertIn(self.pipeline.hybrid_controls[1].mode,
                      {"AUTO_PENDING", "AUTO_HOST"})

        # AUTO_PENDING still needs a distinct camera observation.  Only the
        # new route may produce the first non-zero command; no old command ran.
        for _ in range(8):
            self.feed((7, (150.0, 600.0)), settle=0.06)
            if abs(float(sess.latest_control.get("throttle", 0.0))) > 0.0:
                break
        self.assertGreater(abs(float(sess.latest_control["throttle"])), 0.0)

    def test_route_done_is_not_parking_workflow_done(self):
        self.connect()
        slot = self.wait_slot()
        self.assertIsNotNone(slot)
        runner = self.pipeline.auto_hosts[1]
        runner.mission._status = MissionStatus.DONE
        self.pipeline._parking_stage[1] = "PARKING_AFTER_SETUP_PENDING"
        self.assertEqual(self.pipeline.workflow_status(1),
                         "SETUP_DONE_WAIT_STOP")
        self.assertNotEqual(self.pipeline.workflow_status(1), "PARKED")

    def test_175219_175349_new_route_resets_route_local_replan_state(self):
        self.connect()
        self.assertIsNotNone(self.wait_slot())
        runner = self.pipeline.auto_hosts[1]
        route = list(self.pipeline._auto_host_route[1])
        runner.mission.request_replan("PATH_DEVIATION")
        self.assertIs(runner.status, MissionStatus.REPLAN_REQUIRED)

        runner.load_route(route)

        self.assertIs(runner.status, MissionStatus.RUNNING)
        self.assertIsNone(runner.replan_reason)
        self.assertEqual(runner.mission.recovery_attempts, 0)
        self.assertIsNone(runner.last_tick_result)

    def test_same_session_recovery_keeps_slot_and_requires_fresh_pose(self):
        esp, slot = self._arm_route()
        sess = self.pipeline.server._session(1)
        old_route = self.pipeline.auto_hosts[1].current_target.route_id

        self.pipeline.server._comm_fail(
            1, {"type": "COMM_TIMEOUT"}, expected_session=sess)
        self.assertTrue(sess.control_held)
        self.assertEqual(sess.latest_control["throttle"], 0.0)
        self.assertEqual(self.pipeline._auto_host_slot[1], slot)
        self.assertEqual(self.pipeline.views[7].slot_id, slot)

        # Periodic STATUS recovers transport, but must not release motion.
        esp.send_periodic_status()
        self.assertTrue(wait_until(
            lambda: (self.pipeline._comm_recovery_context.get(1) or {}).get("state")
                    == "WAIT_FRESH_POSE", timeout=5.0))
        self.pipeline.server.push_control(1, 0.5, 0.5)
        self.assertEqual(sess.latest_control["throttle"], 0.0)
        self.assertTrue(sess.control_held)

        pos = self.pipeline.views[7].position_mm
        self.feed((7, pos), settle=0.12)
        self.assertNotIn(1, self.pipeline._comm_recovery_context)
        self.assertEqual(self.pipeline._auto_host_slot[1], slot)
        self.assertGreater(self.pipeline.auto_hosts[1].current_target.route_id,
                           old_route)
        self.assertEqual(self.pipeline.hybrid_controls[1].mode, "AUTO_PENDING")
        self.assertFalse(sess.control_held)
        self.assertEqual(sess.latest_control["throttle"], 0.0,
                         "planning frame에서 즉시 stale route가 움직였다")

    def test_replacement_session_same_boot_preserves_context_and_replans(self):
        esp, slot = self._arm_route()
        old_session_id = self.pipeline.server._session(1).session_id
        esp2 = MockFirmware(
            self.pipeline.server.bound_port, car_id="CAR_01",
            boot_id=esp.boot_id)
        self.esps.append(esp2)

        self.assertTrue(wait_until(
            lambda: self.pipeline.server._session(1).session_id != old_session_id,
            timeout=5.0))
        new_sess = self.pipeline.server._session(1)
        self.assertEqual(new_sess.boot_id, esp.boot_id)
        self.assertTrue(new_sess.control_held)
        self.assertEqual(self.pipeline._auto_host_slot[1], slot)
        self.assertEqual(self.pipeline.track_of_car[1], 7)
        self.assertTrue(wait_until(
            lambda: (self.pipeline._comm_recovery_context.get(1) or {}).get("state")
                    == "WAIT_FRESH_POSE", timeout=5.0))

        pos = self.pipeline.views[7].position_mm
        self.feed((7, pos), settle=0.12)
        self.assertNotIn(1, self.pipeline._comm_recovery_context)
        self.assertEqual(self.pipeline._auto_host_slot[1], slot)
        self.assertEqual(self.pipeline.track_of_car[1], 7)
        self.assertEqual(self.pipeline.hybrid_controls[1].mode, "AUTO_PENDING")
        self.assertFalse(new_sess.control_held)
        sent = [m["type"] for m in esp2.received]
        self.assertNotIn("WAYPOINT", sent)
        self.assertNotIn("GO", sent)
