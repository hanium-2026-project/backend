"""Deterministic recovery-handshake races from run_20260825_005100."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any, Callable

from control.auto_host_runner import AutoHostRunner, NegotiationState
from pipeline.runner import ParkingPipeline


def wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class ScriptedServer:
    """One-outstanding reliable transport with controllable ACK/session edges."""

    def __init__(self, *, session_id: str = "S1", boot_id: str = "B1",
                 state: str = "READY") -> None:
        self.identity: tuple[str, str] | None = (session_id, boot_id)
        self.state = state
        self.mode = "WAYPOINT_AUTO"
        self.direct_control_enabled = False
        self.control_held = True
        self.controls: list[tuple[float, float]] = []
        self.sent: list[tuple[str, str, int]] = []
        self.pending: tuple[str, str, int] | None = None
        self.results: dict[tuple[str, int], str] = {}
        self._seq = 0
        self._condition = threading.Condition()

        self.on_command_result = None
        self.on_command_rejected = None
        self.on_comm_fail = None
        self.on_comm_recovered = None
        self.on_resync = None

    def session_identity(self, _car_id: int) -> tuple[str, str] | None:
        with self._condition:
            return self.identity

    def last_status(self, _car_id: int) -> dict[str, Any]:
        with self._condition:
            if self.identity is None:
                return {}
            return {"state": self.state, "mode": self.mode,
                    "session_id": self.identity[0], "boot_id": self.identity[1]}

    def _send(self, command: str) -> int:
        with self._condition:
            if self.identity is None:
                raise RuntimeError("no active session")
            if self.pending is not None:
                raise RuntimeError(
                    f"outstanding command exists (seq={self.pending[2]}, "
                    f"type={self.pending[1]}); wait for ack")
            self._seq += 1
            session_id = self.identity[0]
            self.pending = (session_id, command, self._seq)
            self.sent.append(self.pending)
            return self._seq

    def send_reset(self, _car_id: int) -> int:
        return self._send("RESET")

    def send_set_mode(self, _car_id: int, mode: str) -> int:
        assert mode == "REMOTE_DIRECT"
        return self._send("SET_MODE")

    def wait_reliable_result(self, _car_id: int, session_id: str, seq: int,
                             timeout_s: float) -> str | None:
        deadline = time.monotonic() + timeout_s
        key = (session_id, seq)
        with self._condition:
            while True:
                if key in self.results:
                    return self.results.pop(key)
                if self.identity is None or self.identity[0] != session_id:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(min(remaining, 0.02))

    def complete(self, command: str, result: str = "ACCEPTED", *,
                 session_id: str | None = None) -> bool:
        callback = None
        rejected = None
        status: dict[str, Any] = {}
        with self._condition:
            current = self.identity
            sid = session_id or (current[0] if current else "")
            if (self.pending is None or self.pending[0] != sid
                    or self.pending[1] != command):
                return False
            _, _, seq = self.pending
            self.pending = None
            if result == "ACCEPTED":
                if command == "RESET":
                    self.state = "READY"
                elif command == "SET_MODE":
                    self.mode = "REMOTE_DIRECT"
            self.results[(sid, seq)] = result
            status = {"session_id": sid, "boot_id": current[1] if current else "",
                      "last_processed_cmd_seq": seq,
                      "rejected_seq": 0 if result == "ACCEPTED" else seq}
            callback = self.on_command_result
            rejected = self.on_command_rejected if result != "ACCEPTED" else None
            self._condition.notify_all()
        if callback is not None:
            callback(1, seq, result, status)
        if rejected is not None:
            rejected(1, result, status)
        return True

    def retry_pending(self) -> None:
        with self._condition:
            if self.pending is not None:
                self.sent.append(self.pending)

    def replace(self, session_id: str, boot_id: str, *, state: str = "READY") -> None:
        callback = None
        with self._condition:
            self.identity = (session_id, boot_id)
            self.state = state
            self.mode = "WAYPOINT_AUTO"
            self.pending = None
            self._seq = 0
            self.control_held = True
            callback = self.on_resync
            self._condition.notify_all()
        if callback is not None:
            callback(1, {"boot_id": boot_id})

    def push_control(self, _car_id: int, throttle: float, steering: float) -> None:
        if self.control_held:
            throttle = steering = 0.0
        self.controls.append((throttle, steering))

    def stop_control(self, car_id: int) -> None:
        self.push_control(car_id, 0.0, 0.0)

    def release_control(self, _car_id: int) -> bool:
        self.control_held = False
        return True


class RecoveryNegotiationTest(unittest.TestCase):
    def make_runner(self, server: ScriptedServer) -> AutoHostRunner:
        return AutoHostRunner(server, 1, [])

    def start_arm(self, runner: AutoHostRunner, errors: list[BaseException],
                  *, release_control: bool = False) -> threading.Thread:
        def target() -> None:
            try:
                runner.arm_session(wait_s=1.5, release_control=release_control)
            except BaseException as exc:  # test thread must report failures
                errors.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    @staticmethod
    def count(server: ScriptedServer, command: str,
              session_id: str | None = None) -> int:
        return sum(1 for sid, kind, _seq in server.sent
                   if kind == command and (session_id is None or sid == session_id))

    def finish(self, thread: threading.Thread, errors: list[BaseException]) -> None:
        thread.join(2.0)
        self.assertFalse(thread.is_alive(), "negotiation worker did not finish")
        self.assertEqual(errors, [])

    def pipeline_harness(self, server: ScriptedServer,
                         runner: AutoHostRunner) -> tuple[ParkingPipeline, list]:
        pipeline = ParkingPipeline.__new__(ParkingPipeline)
        pipeline.server = server
        pipeline.config = SimpleNamespace(
            auto_host_handshake_s=1.5, control_mode="auto-host")
        pipeline.auto_hosts = {1: runner}
        pipeline._comm_recovery_context = {
            1: {"state": "WAIT_SESSION", "generation": 1,
                "negotiation_session_id": server.identity[0]}}
        pipeline._comm_recovery_starting = set()
        pipeline._comm_lost = {1}
        pipeline._lock = threading.Lock()
        pipeline.track_of_car = {}
        pipeline.views = {}
        events: list[tuple[str, dict[str, Any]]] = []
        faults: list[tuple[str, dict[str, Any]]] = []
        pipeline._emit_event = lambda name, **fields: events.append((name, fields))
        pipeline._comm_recovery_fault = (
            lambda _car_id, reason, **fields: faults.append((reason, fields)))
        pipeline.dashboard = SimpleNamespace(push_event=lambda *args, **kwargs: None)
        pipeline._test_faults = faults
        return pipeline, events

    def test_run_005100_delayed_reset_ack_serializes_duplicate_trigger(self) -> None:
        server = ScriptedServer(state="EMERGENCY_STOP")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        first = self.start_arm(runner, errors)
        second = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "RESET") == 1))
        self.assertEqual(self.count(server, "SET_MODE"), 0)
        self.assertIs(runner.negotiation_state, NegotiationState.WAIT_RESET_ACK)

        server.complete("RESET")
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE") == 1))
        server.complete("SET_MODE")
        self.finish(first, errors)
        self.finish(second, errors)
        self.assertIs(runner.negotiation_state,
                      NegotiationState.READY_REMOTE_DIRECT)
        self.assertEqual(self.count(server, "RESET"), 1)
        self.assertEqual(self.count(server, "SET_MODE"), 1)

    def test_lost_reset_ack_retries_same_seq_before_set_mode(self) -> None:
        server = ScriptedServer(state="EMERGENCY_STOP")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        thread = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "RESET") == 1))
        server.retry_pending()
        server.retry_pending()
        reset_seqs = {seq for _sid, kind, seq in server.sent if kind == "RESET"}
        self.assertEqual(reset_seqs, {1})
        self.assertEqual(self.count(server, "SET_MODE"), 0)
        server.complete("RESET")
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE") == 1))
        server.complete("SET_MODE")
        self.finish(thread, errors)

    def test_delayed_set_mode_ack_is_idempotent(self) -> None:
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        first = self.start_arm(runner, errors)
        second = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE") == 1))
        server.retry_pending()
        self.assertEqual({seq for _sid, kind, seq in server.sent if kind == "SET_MODE"},
                         {1})
        server.complete("SET_MODE")
        self.finish(first, errors)
        self.finish(second, errors)
        self.assertEqual(self.count(server, "SET_MODE"), 2)  # one wire retry, one command

    def test_duplicate_hello_resync_for_same_session_does_not_restart(self) -> None:
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        thread = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE") == 1))
        server.on_resync(1, {"boot_id": "B1"})
        server.on_resync(1, {"boot_id": "B1"})
        server.complete("SET_MODE")
        self.finish(thread, errors)
        self.assertTrue(runner.session.accepted)
        self.assertEqual(self.count(server, "SET_MODE"), 1)

    def test_socket_loss_during_reset_restarts_on_new_session(self) -> None:
        server = ScriptedServer(state="EMERGENCY_STOP")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        thread = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "RESET", "S1") == 1))
        server.replace("S2", "B1", state="READY")
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S2") == 1))
        self.assertFalse(server.complete("RESET", session_id="S1"),
                         "old-session RESET ACK was accepted")
        server.complete("SET_MODE")
        self.finish(thread, errors)

    def test_socket_loss_during_set_mode_restarts_exactly_once(self) -> None:
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        thread = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S1") == 1))
        server.replace("S2", "B1", state="READY")
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S2") == 1))
        server.complete("SET_MODE")
        self.finish(thread, errors)
        self.assertEqual(self.count(server, "SET_MODE", "S2"), 1)

    def test_two_rapid_reconnects_use_only_final_session_result(self) -> None:
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        thread = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S1") == 1))
        server.replace("S2", "B1", state="READY")
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S2") == 1))
        server.replace("S3", "B1", state="READY")
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S3") == 1))
        server.complete("SET_MODE")
        self.finish(thread, errors)
        self.assertEqual(runner.negotiation_generation, 3)

    def test_new_boot_runs_clean_reset_then_set_mode(self) -> None:
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        server.replace("S2", "B2", state="EMERGENCY_STOP")
        errors: list[BaseException] = []
        thread = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "RESET", "S2") == 1))
        server.complete("RESET")
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S2") == 1))
        server.complete("SET_MODE")
        self.finish(thread, errors)

    def test_remote_direct_negotiation_does_not_release_stale_motion(self) -> None:
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        thread = self.start_arm(runner, errors, release_control=False)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE") == 1))
        server.complete("SET_MODE")
        self.finish(thread, errors)
        server.push_control(1, 0.8, -0.4)
        self.assertTrue(server.control_held)
        self.assertEqual(server.controls[-1], (0.0, 0.0))

    def test_latest_natural_race_pipeline_restart_does_not_duplicate_set_mode(self) -> None:
        """COMM recovery -> socket replacement -> restart while new ACK pending."""
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        errors: list[BaseException] = []

        initial = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S1") == 1))
        server.complete("SET_MODE")
        self.finish(initial, errors)

        # Same-session recovery creates a new negotiation epoch.
        runner.session._on_comm_fail(1, {})
        pipeline, _events = self.pipeline_harness(server, runner)
        ParkingPipeline._start_comm_recovery_handshake(pipeline, 1)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S1") == 2))

        # The socket is replaced before that SET_MODE receives its ACK.
        server.replace("S2", "B1", state="READY")
        pipeline._comm_recovery_context[1]["generation"] = 2
        pipeline._comm_recovery_context[1]["negotiation_session_id"] = "S2"
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE", "S2") == 1))
        self.assertFalse(server.complete("SET_MODE", session_id="S1"),
                         "stale old-session SET_MODE ACK was accepted")

        # Pipeline restart + COMM_RECOVERED + terminal ACK race on one edge.
        gate = threading.Event()
        racers = [
            threading.Thread(
                target=lambda: (gate.wait(), ParkingPipeline._start_comm_recovery_handshake(
                    pipeline, 1)), daemon=True),
            threading.Thread(
                target=lambda: (gate.wait(), ParkingPipeline._on_comm_recovered(
                    pipeline, 1)), daemon=True),
            threading.Thread(
                target=lambda: (gate.wait(), server.complete("SET_MODE")),
                daemon=True),
        ]
        for racer in racers:
            racer.start()
        gate.set()
        for racer in racers:
            racer.join(1.0)
        time.sleep(0.03)
        self.assertEqual(self.count(server, "SET_MODE", "S2"), 1)
        self.assertEqual(pipeline._test_faults, [])

        self.assertTrue(wait_until(
            lambda: pipeline._comm_recovery_context[1]["state"]
            == "WAIT_FRESH_POSE"))
        self.assertEqual(self.count(server, "SET_MODE", "S2"), 1)
        self.assertEqual(pipeline._test_faults, [])
        self.assertTrue(server.control_held)
        self.assertTrue(all(throttle == 0.0 for throttle, _ in server.controls))

    def test_duplicate_comm_recovered_callbacks_share_pending_transaction(self) -> None:
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        initial = self.start_arm(runner, errors)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE") == 1))
        server.complete("SET_MODE")
        self.finish(initial, errors)

        runner.session._on_comm_fail(1, {})
        pipeline, _events = self.pipeline_harness(server, runner)
        pipeline._comm_recovery_context[1]["state"] = "WAIT_CONNECTION"
        ParkingPipeline._on_comm_recovered(pipeline, 1)
        ParkingPipeline._on_comm_recovered(pipeline, 1)
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE") == 2))
        time.sleep(0.03)
        self.assertEqual(self.count(server, "SET_MODE"), 2)
        server.complete("SET_MODE")
        self.assertTrue(wait_until(
            lambda: pipeline._comm_recovery_context[1]["state"]
            == "WAIT_FRESH_POSE"))
        self.assertEqual(pipeline._test_faults, [])

    def test_many_ensure_calls_during_pending_ack_emit_one_logical_command(self) -> None:
        server = ScriptedServer(state="READY")
        runner = self.make_runner(server)
        errors: list[BaseException] = []
        threads = [self.start_arm(runner, errors) for _ in range(8)]
        self.assertTrue(wait_until(lambda: self.count(server, "SET_MODE") == 1))
        time.sleep(0.03)
        self.assertEqual(self.count(server, "SET_MODE"), 1)
        server.complete("SET_MODE")
        for thread in threads:
            self.finish(thread, errors)
        logical = {(sid, kind, seq) for sid, kind, seq in server.sent}
        self.assertEqual(logical, {("S1", "SET_MODE", 1)})


if __name__ == "__main__":
    unittest.main()
