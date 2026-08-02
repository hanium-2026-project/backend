"""펌웨어 계약 준수 검증.

우리 편의대로 만든 목이 아니라 protocol.c 의 필수 필드 검증을 이식한
MockFirmware 를 상대로, 실제 ESP32 와 붙였을 때 깨질 지점을 잡는다.

실행: python -m unittest comm.tests.test_firmware_contract -v
"""

from __future__ import annotations

import time
import unittest

from comm import MissionOrchestrator, MissionState, VehicleServer, protocol
from comm.tests.mock_firmware import MockFirmware
from parking.waypoints import build_waypoints, default_slot_specs


def wait_until(cond, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


class ContractTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = VehicleServer(port=0, known_car_ids={1, 2})
        self.server.start()
        self.esp: MockFirmware | None = None

    def tearDown(self) -> None:
        if self.esp is not None:
            self.esp.close()
        self.server.stop()

    def connect(self, **kwargs) -> MockFirmware:
        self.esp = MockFirmware(self.server.bound_port, **kwargs)
        wait_until(lambda: self.esp.hello_result is not None)
        return self.esp


class TestHandshake(ContractTestBase):
    def test_hello_ack_has_required_fields(self):
        """boot_id 에코 + command_seq_start 없으면 펌웨어가 파싱 실패한다."""
        esp = self.connect()
        self.assertEqual(esp.hello_result, "READY_ALLOWED")
        self.assertTrue(esp.session_id, "session_id 미발급")
        self.assertEqual(esp.command_seq_start, 1)
        self.assertEqual(esp.rejects, [], f"계약 위반: {esp.rejects}")

    def test_car_id_is_wire_string(self):
        """car_id 는 'CAR_01' 문자열이어야 하며 정수면 전 메시지가 거절된다."""
        esp = self.connect()
        hello_ack = next(m for m in esp.received if m["type"] == "HELLO_ACK")
        self.assertEqual(hello_ack["car_id"], "CAR_01")
        self.assertEqual(protocol.parse_car_id("CAR_01"), 1)
        self.assertEqual(protocol.wire_car_id(2), "CAR_02")

    def test_version_mismatch_rejected(self):
        """버전 불일치는 REJECTED 로 사유를 알려주되 session 은 발급하지 않는다."""
        esp = self.connect(version=99)
        self.assertEqual(esp.hello_result, "REJECTED")
        self.assertFalse(esp.session_id, "REJECTED 인데 session 발급됨")
        self.assertIsNone(esp.command_seq_start)
        reject = next(m for m in esp.received if m["type"] == "HELLO_ACK")
        self.assertEqual(reject.get("reject_reason"), "VERSION_MISMATCH")


class TestHeartbeat(ContractTestBase):
    def test_server_sends_heartbeat(self):
        """노트북이 HEARTBEAT 를 보내지 않으면 차량이 1초 뒤 COMM_TIMEOUT 된다."""
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.heartbeats >= 3, timeout=2.0),
                        f"HEARTBEAT 수신 {esp.heartbeats}회 — 주기 송신 누락")
        self.assertFalse(esp.comm_timeout_fired, "COMM_TIMEOUT 발생")

    def test_pose_update_not_sent_while_unsupported(self):
        """펌웨어 수신 enum 에 없는 POSE_UPDATE 를 보내면 거절 로그만 쌓인다."""
        esp = self.connect()
        self.server.push_pose(1, 42.5, 76.0, 91.2, "TRAJECTORY")
        time.sleep(0.4)
        self.assertEqual(esp.pose_updates, 0)
        self.assertFalse([r for r in esp.rejects if "POSE_UPDATE" in r[1]],
                         "POSE_UPDATE 가 전송됨 (펌웨어 미지원 구간)")


class TestWaypointContract(ContractTestBase):
    def test_waypoint_accepted_by_firmware_parser(self):
        """waypoint_id>=1, target_heading_deg 비-null, motion/arrival 필수."""
        esp = self.connect()
        wps = build_waypoints(default_slot_specs()["A4"], route_id=1)
        self.server.send_waypoint(1, wps[0].to_wire())     # CRUISE = heading 무관
        self.assertTrue(wait_until(lambda: esp.target is not None),
                        f"WAYPOINT 거절됨: {esp.rejects}")
        self.assertEqual(esp.rejects, [])
        self.assertEqual(esp.target["waypoint_id"], 1)

    def test_all_phases_pass_contract(self):
        esp = self.connect()
        for slot in ("A4", "B2"):
            for wp in build_waypoints(default_slot_specs()[slot], route_id=2):
                wire = wp.to_wire()
                self.assertIsNotNone(wire["target_heading_deg"])
                self.assertGreaterEqual(wire["waypoint_id"], 1)
                self.assertIn(wire["motion_direction"], ("FORWARD", "REVERSE"))
                self.assertIn(wire["arrival_mode"], ("STOP", "PASS"))
                self.assertLessEqual(len(protocol.encode(
                    protocol.make_waypoint(1, "S1", 1, wire))), 512)


class TestWaitContract(ContractTestBase):
    def test_wait_requires_route_and_reason_enum(self):
        """WAIT 이 거절되면 안전정지가 실패한다 — 가장 위험한 항목."""
        esp = self.connect()
        self.server.send_wait(1, "COLLISION_RISK", route_id=1, waypoint_id=2)
        self.assertTrue(wait_until(lambda: esp.state == "WAITING"),
                        f"WAIT 거절됨: {esp.rejects}")
        self.assertEqual(esp.wait_reason, "COLLISION_RISK")

    def test_invalid_reason_blocked_locally(self):
        """펌웨어 enum 에 없는 사유는 송신 전에 막는다."""
        self.connect()
        with self.assertRaises(ValueError):
            protocol.make_wait(1, "S1", 1, 1, 1, "WP_SWITCH")

    def test_orchestrator_uses_firmware_reasons(self):
        esp = self.connect()
        orch = MissionOrchestrator(self.server)
        self.server.on_status = orch.on_vehicle_status
        wps = build_waypoints(default_slot_specs()["A4"], route_id=1)
        m = orch.start_mission(1, wps, slot_id="A4")
        wait_until(lambda: m.state is MissionState.DRIVING)
        cur = m.current
        orch.update_pose(1, (cur.x, cur.y), 90.0)          # 도착 판정 → WAIT

        # 전환 사이클이 진행되면 wait_reason 은 GO 로 초기화되므로,
        # 순간값이 아니라 실제로 나간 WAIT 메시지의 reason 을 확인한다.
        def wait_sent() -> bool:
            return any(r["type"] == "WAIT" for r in esp.received)
        self.assertTrue(wait_until(wait_sent),
                        f"WAIT 미전송 (rejects={esp.rejects})")
        reasons = {r["reason"] for r in esp.received if r["type"] == "WAIT"}
        self.assertTrue(reasons <= protocol.WAIT_REASONS,
                        f"펌웨어 enum 밖의 사유: {reasons - protocol.WAIT_REASONS}")
        self.assertIn("WAYPOINT_REACHED", reasons)
        self.assertEqual(esp.rejects, [])


class TestAckSemantics(ContractTestBase):
    def test_periodic_status_does_not_ack(self):
        """command_result=NONE 인 주기 STATUS 를 승인으로 처리하면 안 된다."""
        esp = self.connect()
        wire = build_waypoints(default_slot_specs()["A4"], route_id=1)[0].to_wire()

        # 목의 응답을 끊고 outstanding 상태를 만든 뒤 주기 STATUS 만 흘린다
        real_handler = esp._on_waypoint
        esp._on_waypoint = lambda msg: None                # 응답하지 않음
        self.server.send_waypoint(1, wire)
        time.sleep(0.1)
        esp.last_processed_cmd_seq = 1                     # 주기 STATUS 가 seq 를 반복
        esp.send_periodic_status()
        time.sleep(0.2)

        session = self.server.sessions[1]
        self.assertIsNotNone(session.sender.outstanding,
                             "command_result=NONE 인 STATUS 가 ack 로 오인됨")
        esp._on_waypoint = real_handler

    def test_terminal_result_acks(self):
        esp = self.connect()
        wire = build_waypoints(default_slot_specs()["A4"], route_id=1)[0].to_wire()
        self.server.send_waypoint(1, wire)
        self.assertTrue(wait_until(
            lambda: self.server.sessions[1].sender.outstanding is None),
            "ACCEPTED 응답 후에도 outstanding 이 남음")

    def test_rejection_is_reported(self):
        """거절을 무시하면 오케스트레이터가 영구 대기에 빠진다."""
        esp = self.connect()
        rejected: list[tuple[int, str]] = []
        self.server.on_command_rejected = lambda cid, res, st: rejected.append((cid, res))
        # WAITING 이 아닌 상태에서 GO → INVALID_STATE
        self.server.send_go(1, 1, 1)
        self.assertTrue(wait_until(lambda: bool(rejected)),
                        f"거절 통지 없음 (esp.state={esp.state})")
        self.assertEqual(rejected[0][1], "INVALID_STATE")

    def test_idempotent_retransmit(self):
        """동일 seq 재전송은 재실행 없이 이전 결과만 돌려받는다."""
        esp = self.connect()
        wire = build_waypoints(default_slot_specs()["A4"], route_id=1)[0].to_wire()
        seq = self.server.send_waypoint(1, wire)
        wait_until(lambda: esp.last_processed_cmd_seq == seq)
        before = len([m for m in esp.received if m["type"] == "WAYPOINT"])
        self.server.sessions[1].sender._send_raw(
            protocol.make_waypoint(1, esp.session_id, seq, wire))
        time.sleep(0.2)
        after = len([m for m in esp.received if m["type"] == "WAYPOINT"])
        self.assertEqual(after, before + 1)
        self.assertEqual(esp.last_result, "ACCEPTED")
        self.assertEqual(esp.rejects, [])


class TestFullMission(ContractTestBase):
    def test_end_to_end_parking(self):
        """계약 위반 없이 A4 슬롯까지 전 waypoint 주행 후 PARKED."""
        esp = self.connect()
        parked: list[tuple[int, str]] = []
        orch = MissionOrchestrator(self.server, on_parked=lambda c, s: parked.append((c, s)))
        self.server.on_status = orch.on_vehicle_status
        self.server.on_command_rejected = orch.on_command_rejected

        wps = build_waypoints(default_slot_specs()["A4"], route_id=orch.next_route_id())
        m = orch.start_mission(1, wps, slot_id="A4")

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and m.state is not MissionState.DONE:
            cur = m.current
            if cur is not None and m.state in (MissionState.DRIVING, MissionState.PARKED_CHECK):
                orch.update_pose(1, (cur.x, cur.y),
                                 cur.target_heading_deg if cur.heading_required else 90.0)
            time.sleep(0.05)

        self.assertEqual(esp.rejects, [], f"계약 위반 발생: {esp.rejects}")
        self.assertIs(m.state, MissionState.DONE, f"미완주 (state={m.state})")
        self.assertEqual(parked, [(1, "A4")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
