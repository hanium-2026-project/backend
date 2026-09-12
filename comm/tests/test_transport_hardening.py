"""Transport/session fault-injection tests against the production server."""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest

from comm import protocol
from comm.reliability import ReliableSender
from comm.server import VehicleServer, VehicleSession
from comm.tests.mock_firmware import MockFirmware


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def status(seq: int, *, session: str = "S1", boot: str = "B1") -> dict:
    return {
        "version": 1,
        "type": "STATUS",
        "car_id": 1,
        "boot_id": boot,
        "session_id": session,
        "status_seq": seq,
        "last_processed_cmd_seq": 0,
        "command_result": "NONE",
        "state": "READY",
        "mode": "REMOTE_DIRECT",
    }


class TestNdjsonStream(unittest.TestCase):
    def setUp(self) -> None:
        self.server = VehicleServer()
        self.server._running = True
        self.client, server_sock = socket.socketpair()
        sender = ReliableSender(1, lambda message: None)
        self.session = VehicleSession(1, "S1", "B1", server_sock, sender)
        self.server.sessions[1] = self.session
        self.seen: list[int] = []
        self.server.on_status = lambda _cid, msg: self.seen.append(msg["status_seq"])
        self.thread = threading.Thread(
            target=self.server._rx_loop, args=(self.session,), daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server._running = False
        for sock in (self.client, self.session.conn):
            try:
                sock.close()
            except OSError:
                pass

    @staticmethod
    def _wire(message: dict) -> bytes:
        wire = dict(message)
        wire["car_id"] = protocol.wire_car_id(int(wire["car_id"]))
        return (json.dumps(wire, separators=(",", ":")) + "\n").encode()

    def test_fragmented_partial_and_coalesced_frames(self) -> None:
        first = self._wire(status(1))
        second = self._wire(status(2))
        third = self._wire(status(3))
        self.client.sendall(first[:7])
        time.sleep(0.02)
        self.client.sendall(first[7:] + second + third[:11])
        time.sleep(0.02)
        self.client.sendall(third[11:])
        self.assertTrue(wait_until(lambda: self.seen == [1, 2, 3]))

    def test_rx_loop_survives_idle_socket_timeout(self) -> None:
        self.session.conn.settimeout(0.02)
        time.sleep(0.08)
        self.assertTrue(self.session.alive)
        self.assertTrue(self.thread.is_alive())
        self.client.sendall(self._wire(status(1)))
        self.assertTrue(wait_until(lambda: self.seen == [1]))

    def test_queued_bytes_get_only_bounded_rx_drain_grace(self) -> None:
        client, server_sock = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server_sock.close)
        session = VehicleSession(
            1, "S2", "B1", server_sock, ReliableSender(1, lambda message: None))
        self.server.sessions[1] = session
        client.sendall(b"{")
        self.assertTrue(self.server._defer_for_queued_rx(session, 1000.0))
        self.assertTrue(self.server._defer_for_queued_rx(session, 1050.0))
        self.assertFalse(self.server._defer_for_queued_rx(session, 1110.0))

    def test_oversized_partial_frame_closes_session(self) -> None:
        self.client.sendall(b"x" * (protocol.MAX_MESSAGE_BYTES + 1))
        self.assertTrue(wait_until(lambda: not self.session.alive))


class TestFreshnessAndIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self.server = VehicleServer()
        left, self.peer = socket.socketpair()
        self.session = VehicleSession(
            1, "S1", "B1", left, ReliableSender(1, lambda message: None))
        self.server.sessions[1] = self.session

    def tearDown(self) -> None:
        self.session.conn.close()
        self.peer.close()

    def test_only_new_current_session_status_refreshes_liveness(self) -> None:
        self.server._dispatch(self.session, status(10))
        fresh = self.session.last_rx_ms
        cases = [
            status(10),
            status(9),
            status(11, session="OLD"),
            status(11, boot="OTHER_BOOT"),
            {**status(11), "car_id": 2},
            {**status(11), "version": 99},
        ]
        for message in cases:
            self.server._dispatch(self.session, message)
            self.assertEqual(self.session.last_rx_ms, fresh, message)
        self.server._dispatch(self.session, status(11))
        self.assertGreaterEqual(self.session.last_rx_ms, fresh)
        self.assertEqual(self.session.last_status["status_seq"], 11)

    def test_unrelated_delayed_command_result_does_not_refresh(self) -> None:
        before = self.session.last_rx_ms
        self.server._dispatch(self.session, {
            "version": 1,
            "type": "COMMAND_RESULT",
            "car_id": 1,
            "boot_id": "B1",
            "session_id": "S1",
            "last_processed_cmd_seq": 77,
            "command_result": "ACCEPTED",
        })
        self.assertEqual(self.session.last_rx_ms, before)

    def test_old_session_failure_cannot_poison_replacement(self) -> None:
        old = self.session
        new_left, new_peer = socket.socketpair()
        self.addCleanup(new_peer.close)
        replacement = VehicleSession(
            1, "S2", "B1", new_left, ReliableSender(1, lambda message: None))
        self.addCleanup(replacement.conn.close)
        self.server.sessions[1] = replacement
        self.server._comm_fail(
            1, {"type": "RETRANSMIT_EXHAUSTED"}, expected_session=old)
        self.assertFalse(replacement.comm_failed)
        self.assertFalse(replacement.control_held)

    def test_stale_old_session_command_result_cannot_ack_replacement(self) -> None:
        old = self.session
        old.sender.send(protocol.make_reset(1, "S1", 0))
        new_left, new_peer = socket.socketpair()
        self.addCleanup(new_peer.close)
        replacement = VehicleSession(
            1, "S2", "B1", new_left, ReliableSender(1, lambda message: None))
        self.addCleanup(replacement.conn.close)
        self.server.sessions[1] = replacement
        replacement.sender.send(protocol.make_set_mode(1, "S2", 0, "REMOTE_DIRECT"))
        callbacks: list[tuple[int, int, str]] = []
        self.server.on_command_result = (
            lambda cid, seq, result, _msg: callbacks.append((cid, seq, result)))

        self.server._dispatch(old, {
            "version": 1, "type": "COMMAND_RESULT", "car_id": 1,
            "boot_id": "B1", "session_id": "S1",
            "last_processed_cmd_seq": 1, "command_result": "ACCEPTED",
        })

        self.assertIsNotNone(replacement.sender.outstanding)
        self.assertEqual(callbacks, [])


class TestStatusLossBudget(unittest.TestCase):
    def setUp(self) -> None:
        self.server = VehicleServer(port=0, known_car_ids={1})
        self.failures: list[dict] = []
        self.recoveries: list[int] = []
        self.server.on_comm_fail = lambda _cid, info: self.failures.append(info)
        self.server.on_comm_recovered = self.recoveries.append
        self.server.start()
        self.esp = MockFirmware(self.server.bound_port, status_interval=0.2)
        self.assertTrue(wait_until(lambda: self.esp.state == "READY"))

    def tearDown(self) -> None:
        self.esp.close()
        self.server.stop()

    def test_four_missed_status_opportunities_do_not_trip_but_six_do(self) -> None:
        self.esp.status_paused = True
        time.sleep(0.75)  # fewer than the 1000 ms safety budget
        self.esp.status_paused = False
        self.esp.send_periodic_status()
        time.sleep(0.1)
        self.assertEqual(self.failures, [])

        self.esp.status_paused = True
        self.assertTrue(wait_until(lambda: bool(self.failures), timeout=1.5))
        self.assertEqual(self.failures[0]["type"], "COMM_TIMEOUT")
        self.assertTrue(self.server.control_is_held(1))
        self.esp.status_paused = False
        self.esp.send_periodic_status()
        self.assertTrue(wait_until(lambda: self.recoveries == [1]))
        self.assertTrue(self.server.control_is_held(1),
                        "RX recovery must not release zero automatically")


class TestDisconnectReconnectSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.server = VehicleServer(port=0, known_car_ids={1})
        self.server.direct_control_enabled = True
        self.failures: list[dict] = []
        self.server.on_comm_fail = lambda _cid, info: self.failures.append(info)
        self.server.start()
        self.first = MockFirmware(
            self.server.bound_port, boot_id="BOOT_A", status_interval=0.05)
        self.assertTrue(wait_until(lambda: self.first.state == "READY"))

    def tearDown(self) -> None:
        self.first.close()
        replacement = getattr(self, "replacement", None)
        if replacement is not None:
            replacement.close()
        self.server.stop()

    def test_nonzero_disconnect_same_boot_reconnect_stays_zero_held(self) -> None:
        old_session = self.server._session(1)
        self.server.push_control(1, 0.6, -0.2)
        self.assertTrue(wait_until(
            lambda: (self.first.last_direct_control or {}).get("throttle") == 0.6))
        self.first.close()
        self.assertTrue(wait_until(lambda: bool(self.failures)))
        self.assertTrue(old_session.control_held)
        self.assertEqual(old_session.latest_control["throttle"], 0.0)

        self.replacement = MockFirmware(
            self.server.bound_port, boot_id="BOOT_A", status_interval=0.05)
        self.assertTrue(wait_until(
            lambda: self.replacement.state == "READY" and
            self.server._session(1) is not old_session))
        new_session = self.server._session(1)
        self.assertTrue(new_session.control_held)
        self.server.push_control(1, 0.8, 0.4)
        self.assertEqual(new_session.latest_control["throttle"], 0.0)

        # Even a delayed packet handed directly to the old session cannot
        # mutate/recover the replacement.
        before = new_session.last_rx_ms
        self.server._dispatch(old_session, status(
            999, session=old_session.session_id, boot="BOOT_A"))
        self.assertEqual(new_session.last_rx_ms, before)

    def test_reboot_new_boot_gets_new_held_session(self) -> None:
        old_session = self.server._session(1)
        self.replacement = MockFirmware(
            self.server.bound_port, boot_id="BOOT_B", status_interval=0.05)
        self.assertTrue(wait_until(
            lambda: self.replacement.state == "READY" and
            self.server._session(1) is not old_session))
        new_session = self.server._session(1)
        self.assertEqual(new_session.boot_id, "BOOT_B")
        self.assertNotEqual(new_session.session_id, old_session.session_id)
        self.assertTrue(new_session.control_held)


class _SlowSocket:
    """A socket-like sink that would interleave without a shared send lock."""

    def __init__(self) -> None:
        self.data = bytearray()

    def sendall(self, payload: bytes) -> None:
        for byte in payload:
            self.data.append(byte)
            time.sleep(0.00005)


class TestConcurrentWriters(unittest.TestCase):
    def test_session_writers_preserve_ndjson_frames(self) -> None:
        server = VehicleServer()
        sock = _SlowSocket()
        session = VehicleSession(
            1, "S1", "B1", sock, ReliableSender(1, lambda message: None))  # type: ignore[arg-type]
        server.sessions[1] = session
        one = protocol.make_heartbeat(1, "S1", 1)
        two = protocol.make_direct_control(1, "S1", 2, 0.0, 0.0)
        threads = [
            threading.Thread(target=server._safe_send_session, args=(session, one)),
            threading.Thread(target=server._safe_send_session, args=(session, two)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        decoded = [json.loads(line) for line in bytes(sock.data).splitlines()]
        self.assertEqual({item["type"] for item in decoded},
                         {"HEARTBEAT", "DIRECT_CONTROL"})

    def test_partial_send_failure_retires_stream_instead_of_reusing_it(self) -> None:
        class PartialFailureSocket:
            def __init__(self) -> None:
                self.calls = 0
                self.closed = False

            def sendall(self, _payload: bytes) -> None:
                self.calls += 1
                raise OSError("partial write")

            def close(self) -> None:
                self.closed = True

        server = VehicleServer()
        sock = PartialFailureSocket()
        session = VehicleSession(
            1, "S1", "B1", sock, ReliableSender(1, lambda message: None))  # type: ignore[arg-type]
        session.latest_control = protocol.make_direct_control(1, "S1", 1, 0.0, 0.0)
        server.sessions[1] = session
        failures: list[dict] = []
        server.on_comm_fail = lambda _cid, info: failures.append(info)

        self.assertFalse(server._safe_send_session(
            session, protocol.make_heartbeat(1, "S1", 1)))
        self.assertFalse(session.alive)
        self.assertTrue(sock.closed)
        self.assertEqual(sock.calls, 1)
        self.assertEqual(failures[0]["type"], "SOCKET_SEND_FAILED")
        self.assertFalse(server._safe_send_session(
            session, protocol.make_heartbeat(1, "S1", 2)))
        self.assertEqual(sock.calls, 1, "retired NDJSON stream was reused")


if __name__ == "__main__":
    unittest.main()
