"""하드웨어팀 실물 연동 피드백 4건에 대한 회귀 검증 (2026-08-07).

1. WAYPOINT 수신 시 READY → WAITING (실물 확인). 출발은 GO 로만.
2. HELLO_ACK 가 HOLD 여도 연결을 끊지 않고, 조건이 풀리면 재판정한다.
3. COMM_FAIL 은 같은 장애 동안 1회만 통지되고, 복구 시 1회 통지된다.
4. WAIT 이 응답 대기 중인 STOP 을 선점하지 못한다.

실행: python -m unittest comm.tests.test_hw_feedback_fixes -v
"""

from __future__ import annotations

import time
import unittest

from comm import VehicleServer, protocol
from comm.reliability import ReliableSender
from comm.tests.mock_firmware import MockFirmware


def wait_until(cond, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


class ServerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = VehicleServer(port=0, known_car_ids={1, 2})
        self.esp: MockFirmware | None = None

    def tearDown(self) -> None:
        if self.esp is not None:
            self.esp.close()
        self.server.stop()

    def connect(self, **kwargs) -> MockFirmware:
        self.esp = MockFirmware(self.server.bound_port, **kwargs)
        return self.esp


# ─── ① WAYPOINT → WAITING ────────────────────────────────────────────────────

class TestWaypointLoadsWithoutMoving(ServerTestBase):
    def test_waypoint_puts_car_in_waiting_not_moving(self):
        """실물 펌웨어는 WAYPOINT 로 target 만 적재하고 WAITING 에 머문다."""
        self.server.start()
        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.state == "READY"))

        wire = {
            "route_id": 1, "waypoint_id": 1, "phase": "CRUISE",
            "x_cm": 15.0, "y_cm": 10.0, "target_heading_deg": 90.0,
            "motion_direction": "FORWARD", "arrival_mode": "STOP",
            "speed_cm_s": 8.0, "position_tolerance_cm": 6.0,
            "heading_tolerance_deg": 20.0, "heading_required": False,
            "is_final": False,
        }
        self.server.send_waypoint(1, wire)
        self.assertTrue(wait_until(lambda: esp.target is not None),
                        "target 미적재")
        self.assertEqual(esp.state, "WAITING", "WAYPOINT 만으로 출발하면 안 된다")

        self.assertTrue(wait_until(lambda: self.server._session(1).sender.outstanding is None))
        self.server.send_go(1, 1, 1)
        self.assertTrue(wait_until(lambda: esp.state == "MOVING"), "GO 로 출발해야 한다")
        self.assertEqual(esp.rejects, [], f"계약 위반: {esp.rejects}")


# ─── ② HOLD 는 연결을 끊지 않는다 ────────────────────────────────────────────

class TestHoldKeepsConnection(ServerTestBase):
    def test_hold_does_not_close_and_rejudges(self):
        """HOLD 뒤에도 소켓이 살아 있고, 조건이 풀리면 다음 HELLO 에서 승인된다."""
        blocked = {"on": True}
        self.server.hold_check = (
            lambda car_id, hello: "CAR_NOT_DETECTED" if blocked["on"] else None)
        self.server.start()

        esp = self.connect()
        self.assertTrue(wait_until(lambda: esp.hello_result == "HOLD"), "HOLD 미판정")
        # 소켓이 닫히지 않아야 HELLO 재전송이 서버까지 닿는다
        self.assertTrue(wait_until(lambda: esp.hello_sent >= 2),
                        "연결이 끊겨 HELLO 재전송이 불가")
        self.assertTrue(esp.link_up, "HOLD 인데 연결이 끊겼다")

        blocked["on"] = False                     # 카메라가 차량을 잡았다
        self.assertTrue(wait_until(lambda: esp.hello_result == "READY_ALLOWED"),
                        "조건 해소 후에도 재판정되지 않음")
        self.assertTrue(wait_until(lambda: esp.state == "READY"))
        self.assertTrue(esp.session_id)
        self.assertEqual(esp.rejects, [], f"계약 위반: {esp.rejects}")

    def test_rejected_still_closes(self):
        """REJECTED 는 종전대로 연결을 닫는다."""
        self.server.known_car_ids = {2}
        self.server.start()
        esp = self.connect(car_id="CAR_01")
        self.assertTrue(wait_until(lambda: esp.hello_result == "REJECTED"))
        self.assertTrue(wait_until(lambda: not esp.link_up), "REJECTED 인데 연결 유지")


# ─── ③ COMM_FAIL debounce ────────────────────────────────────────────────────

class TestCommFailDebounce(ServerTestBase):
    def test_fires_once_per_outage_and_once_on_recovery(self):
        fails: list[dict] = []
        recovered: list[int] = []
        self.server.on_comm_fail = lambda cid, info: fails.append(info)
        self.server.on_comm_recovered = lambda cid: recovered.append(cid)
        self.server.start()

        # 주기 STATUS 를 끊어 두면 서버가 COMM_TIMEOUT 을 낸다
        esp = self.connect(status_interval=0)
        self.assertTrue(wait_until(lambda: esp.state == "READY"))

        self.assertTrue(wait_until(lambda: len(fails) >= 1, timeout=3.0),
                        "COMM_TIMEOUT 미통지")
        first = len(fails)
        time.sleep(protocol.TIMING["COMM_TIMEOUT"] / 1000.0 * 3)
        self.assertEqual(len(fails), first,
                         f"같은 장애로 반복 통지됨 ({len(fails)}회)")

        esp.send_periodic_status()                # 통신 복구
        self.assertTrue(wait_until(lambda: recovered == [1]),
                        f"복구 통지 이상: {recovered}")

    def test_pending_command_cleared_on_failure(self):
        """장애 중 pending 이 남으면 복구 후 새 명령이 전부 막힌다."""
        self.server.on_comm_fail = lambda cid, info: None
        self.server.start()
        esp = self.connect(status_interval=0)
        self.assertTrue(wait_until(lambda: esp.state == "READY"))

        sender = self.server._session(1).sender
        sender.send(protocol.make_stop(1, self.server._session(1).session_id, 0))
        self.assertTrue(wait_until(lambda: sender.outstanding is None, timeout=4.0),
                        "장애 통지 후에도 pending 이 남아 있다")

    def test_timeout_latches_zero_until_explicit_safe_release(self):
        """RX recovery alone must never revive the pre-timeout command."""
        self.server.direct_control_enabled = True
        self.server.start()
        esp = self.connect(status_interval=0)
        self.assertTrue(wait_until(lambda: esp.state == "READY"))

        self.server.push_control(1, 0.25, -0.3)
        self.assertTrue(wait_until(
            lambda: (esp.last_direct_control or {}).get("throttle") == 0.25))
        sess = self.server._session(1)
        self.server._comm_fail(1, {"type": "COMM_TIMEOUT"},
                               expected_session=sess)
        self.assertTrue(sess.control_held)
        self.assertEqual(sess.latest_control["throttle"], 0.0)

        # A producer racing with the fault cannot overwrite the zero latch.
        self.server.push_control(1, 0.4, 0.4)
        self.assertEqual(sess.latest_control["throttle"], 0.0)
        esp.send_periodic_status()
        self.assertTrue(wait_until(lambda: not sess.comm_failed))
        self.assertTrue(sess.control_held, "RX 재개만으로 latch가 풀렸다")
        self.server.push_control(1, 0.4, 0.4)
        self.assertEqual(sess.latest_control["throttle"], 0.0)

        self.assertTrue(self.server.release_control(1))
        self.server.push_control(1, 0.2, 0.1)
        self.assertEqual(sess.latest_control["throttle"], 0.2)

    def test_wrong_session_packet_does_not_refresh_or_recover(self):
        self.server.start()
        esp = self.connect(status_interval=0)
        self.assertTrue(wait_until(lambda: esp.state == "READY"))
        sess = self.server._session(1)
        self.server._comm_fail(1, {"type": "COMM_TIMEOUT"},
                               expected_session=sess)
        before = sess.last_rx_ms
        self.server._dispatch(sess, {
            "version": 1, "type": "STATUS", "car_id": 1,
            "session_id": "STALE_SESSION", "status_seq": 999,
        })
        self.assertEqual(sess.last_rx_ms, before)
        self.assertTrue(sess.comm_failed)


# ─── ④ STOP > WAIT 선점 우선순위 ─────────────────────────────────────────────

class TestPreemptionPriority(unittest.TestCase):
    def setUp(self) -> None:
        self.sent: list[dict] = []
        self.sender = ReliableSender(1, send_raw=self.sent.append,
                                     on_fail=lambda m: None)

    def _msg(self, mtype: str) -> dict:
        return {"version": 1, "type": mtype, "car_id": "CAR_01",
                "session_id": "S1", "seq": 0}

    def test_wait_cannot_preempt_pending_stop(self):
        self.sender.send(self._msg("STOP"))
        with self.assertRaises(RuntimeError):
            self.sender.send(self._msg("WAIT"))
        self.assertEqual(self.sender.outstanding["type"], "STOP",
                         "비상정지가 일시정지로 뒤집혔다")

    def test_stop_preempts_pending_wait(self):
        self.sender.send(self._msg("WAIT"))
        self.sender.send(self._msg("STOP"))
        self.assertEqual(self.sender.outstanding["type"], "STOP")

    def test_wait_preempts_general_command(self):
        self.sender.send(self._msg("WAYPOINT"))
        self.sender.send(self._msg("WAIT"))
        self.assertEqual(self.sender.outstanding["type"], "WAIT")

    def test_same_priority_does_not_preempt(self):
        """동일 우선순위 재송신은 막는다 — 재전송은 tick 이 담당한다."""
        self.sender.send(self._msg("STOP"))
        with self.assertRaises(RuntimeError):
            self.sender.send(self._msg("STOP"))


if __name__ == "__main__":
    unittest.main()
