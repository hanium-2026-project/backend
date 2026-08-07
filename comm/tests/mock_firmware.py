"""실펌웨어 계약을 그대로 흉내내는 ESP32 목(mock).

integrated/esp32_main/main/protocol.c 의 필수 필드 검증과
remote-direct-bridge/tools/mock_esp32.py 의 상태머신을 이식했다.
우리 편의대로 느슨하게 받아주지 않는 것이 이 목의 존재 이유다:
필드가 하나라도 빠지거나 범위를 벗어나면 실제 펌웨어처럼 거절한다.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

CAR_ID = "CAR_01"
VERSION = 1
MAX_LINE = 512
HEARTBEAT_TIMEOUT_S = 1.0

WAIT_REASONS = {
    "REMOTE_WAIT", "COLLISION_RISK", "OBSTACLE", "REROUTING",
    "WAYPOINT_REACHED", "FINAL_WAYPOINT_REACHED", "OPERATOR_REQUEST",
}
PHASES = {"CRUISE", "APPROACH", "ALIGN", "ENTRY", "FINAL"}


class ContractError(Exception):
    """펌웨어가 파싱 단계에서 거절하는 상황."""


def _req(obj: dict[str, Any], key: str, kind: type) -> Any:
    if key not in obj or obj[key] is None:
        raise ContractError(f"missing required field: {key}")
    val = obj[key]
    if kind is float and isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if kind is int and isinstance(val, int) and not isinstance(val, bool):
        return val
    if kind is bool and isinstance(val, bool):
        return val
    if kind is str and isinstance(val, str):
        return val
    raise ContractError(f"field {key} has wrong type: {val!r}")


def _range(name: str, val: float, lo: float, hi: float) -> float:
    if not (lo <= val <= hi):
        raise ContractError(f"{name} out of range [{lo},{hi}]: {val}")
    return val


class MockFirmware:
    """단일 차량 ESP32 목. 별도 스레드에서 수신 루프를 돈다."""

    def __init__(self, port: int, host: str = "127.0.0.1", boot_id: str = "B0000001",
                 version: int = VERSION, car_id: str = CAR_ID,
                 status_interval: float = 0.2) -> None:
        self.boot_id, self.version, self.car_id = boot_id, version, car_id
        self.state = "SYNCING"
        self.mode = "WAYPOINT_AUTO"
        self.session_id = ""
        self.hello_result: str | None = None
        self.command_seq_start: int | None = None
        self.status_seq = 0
        self.last_processed_cmd_seq = 0
        self.last_fingerprint: str | None = None
        self.last_result = "NONE"
        self.target: dict[str, Any] | None = None
        self.wait_reason = "NONE"

        self.rejects: list[tuple[str, str]] = []      # (사유, 원본 라인) — 계약 위반 기록
        self.received: list[dict[str, Any]] = []
        self.heartbeats = 0
        self.pose_updates = 0
        self.last_heartbeat_at = time.monotonic()
        self.comm_timeout_fired = False
        # 실물 펌웨어는 약 200ms 주기로 STATUS 를 올린다. 이게 없으면 노트북이
        # 정상 상황에서도 COMM_TIMEOUT 으로 판정한다. 0 이면 주기 송신을 끈다.
        self.status_interval = status_interval
        # HELLO 재전송 주기 — 실물 펌웨어는 READY 가 되기 전까지 HELLO 를 반복한다
        self.hello_interval = 0.2
        self.hello_sent = 0
        self.link_up = True                    # 서버와의 TCP 링크 생존 여부

        self.sock = socket.create_connection((host, port))
        self._alive = True
        threading.Thread(target=self._rx_loop, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()
        if status_interval > 0:
            threading.Thread(target=self._status_loop, daemon=True).start()
        threading.Thread(target=self._hello_retry_loop, daemon=True).start()
        self._send(self._hello())

    # ─── 송신 ────────────────────────────────────────────────────────────────

    def _send(self, msg: dict[str, Any]) -> None:
        try:
            self.sock.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode())
        except OSError:
            self._alive = False

    def _hello(self) -> dict[str, Any]:
        self.hello_sent += 1
        return {
            "version": self.version, "type": "HELLO", "car_id": self.car_id,
            "boot_id": self.boot_id, "firmware_version": "mock-contract",
            "state": "SYNCING", "previous_state": "BOOT",
            "last_processed_cmd_seq": 0, "target_loaded": False,
            "motor_stopped": True, "error_code": "NONE",
        }

    def _status(self, result: str = "NONE", rejected_seq: int = 0) -> None:
        self.status_seq += 1
        msg = {
            "version": VERSION, "type": "STATUS", "car_id": self.car_id,
            "boot_id": self.boot_id, "session_id": self.session_id,
            "status_seq": self.status_seq,
            "last_processed_cmd_seq": self.last_processed_cmd_seq,
            "rejected_seq": rejected_seq, "command_result": result,
            "state": self.state, "mode": self.mode,
            "target_loaded": self.target is not None,
            "route_id": (self.target or {}).get("route_id", -1),
            "waypoint_id": (self.target or {}).get("waypoint_id", -1),
            "phase": (self.target or {}).get("phase", "NONE"),
            "wait_reason": self.wait_reason, "error_code": "NONE",
        }
        self._send(msg)

    def send_periodic_status(self) -> None:
        """command_result=NONE 인 주기 STATUS — 이것이 ack 로 오인되면 안 된다."""
        self._status()

    # ─── 수신 ────────────────────────────────────────────────────────────────

    def _rx_loop(self) -> None:
        buf = b""
        while self._alive:
            try:
                chunk = self.sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break                          # 서버가 연결을 닫았다
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                if len(line) + 1 > MAX_LINE:
                    self.rejects.append(("OVERSIZED", line.decode(errors="replace")))
                    continue
                self._handle(line)
        self.link_up = False                   # 수신 루프 종료 = 링크 단절

    def _handle(self, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            self.rejects.append(("BAD_JSON", line.decode(errors="replace")))
            return
        self.received.append(msg)
        try:
            if _req(msg, "version", int) != VERSION:
                raise ContractError("version mismatch")
            if _req(msg, "car_id", str) != self.car_id:
                raise ContractError(f"car_id mismatch: {msg.get('car_id')!r}")
            mtype = _req(msg, "type", str)
            handler = {
                "HELLO_ACK": self._on_hello_ack,
                "HEARTBEAT": self._on_heartbeat,
                "WAYPOINT": self._on_waypoint,
                "WAIT": self._on_wait,
                "GO": self._on_go,
                "STOP": self._on_stop,
                "RESET": self._on_reset,
                "SET_MODE": self._on_set_mode,
                "DIRECT_CONTROL": self._on_direct,
            }.get(mtype)
            if handler is None:
                # 펌웨어 수신 enum 에 없는 타입 (예: POSE_UPDATE, EVENT_ACK)
                raise ContractError(f"unsupported message type: {mtype}")
            handler(msg)
        except ContractError as exc:
            self.rejects.append((str(exc), line.decode(errors="replace")))

    # ─── 핸들러 (필수 필드 검증은 펌웨어 protocol.c 그대로) ──────────────────

    def _on_hello_ack(self, msg: dict[str, Any]) -> None:
        _req(msg, "boot_id", str)
        result = _req(msg, "result", str)
        if result not in ("READY_ALLOWED", "HOLD", "REJECTED"):
            raise ContractError("unsupported HELLO_ACK result")
        self.hello_result = result
        if result == "REJECTED":
            return
        self.session_id = _req(msg, "session_id", str)
        self.command_seq_start = _req(msg, "command_seq_start", int)
        if result == "READY_ALLOWED":
            self.state = "READY"
            self.last_heartbeat_at = time.monotonic()
            self._status()

    def _on_heartbeat(self, msg: dict[str, Any]) -> None:
        _req(msg, "session_id", str)
        _req(msg, "heartbeat_seq", int)
        self.heartbeats += 1
        self.last_heartbeat_at = time.monotonic()

    def _on_waypoint(self, msg: dict[str, Any]) -> None:
        _req(msg, "session_id", str)
        seq = _req(msg, "seq", int)
        route_id = _range("route_id", _req(msg, "route_id", int), 1, 9999)
        wp_id = _range("waypoint_id", _req(msg, "waypoint_id", int), 1, 9999)
        phase = _req(msg, "phase", str)
        if phase not in PHASES:
            raise ContractError(f"unsupported phase: {phase}")
        _range("x_cm", _req(msg, "x_cm", float), 0.0, 500.0)
        _range("y_cm", _req(msg, "y_cm", float), 0.0, 500.0)
        _range("target_heading_deg", _req(msg, "target_heading_deg", float), 0.0, 359.999)
        if _req(msg, "motion_direction", str) not in ("FORWARD", "REVERSE"):
            raise ContractError("unsupported motion_direction")
        if _req(msg, "arrival_mode", str) not in ("STOP", "PASS"):
            raise ContractError("unsupported arrival_mode")
        _range("speed_cm_s", _req(msg, "speed_cm_s", float), 0.01, 100.0)
        _range("position_tolerance_cm", _req(msg, "position_tolerance_cm", float), 0.01, 100.0)
        _range("heading_tolerance_deg", _req(msg, "heading_tolerance_deg", float), 0.0, 180.0)
        _req(msg, "heading_required", bool)
        _req(msg, "is_final", bool)

        if not self._reliable_pre(seq, f"WAYPOINT|{route_id}|{wp_id}"):
            return
        self.target = {"route_id": int(route_id), "waypoint_id": int(wp_id), "phase": phase}
        # 실물 펌웨어 확인(2026-08-07): WAYPOINT 는 target 을 적재만 하고
        # READY → WAITING 으로 간다. 실제 출발은 GO 를 받아야 한다.
        if self.state == "READY":
            self.state = "WAITING"
        self._reliable_done(seq, "ACCEPTED")

    def _on_wait(self, msg: dict[str, Any]) -> None:
        _req(msg, "session_id", str)
        seq = _req(msg, "seq", int)
        _range("route_id", _req(msg, "route_id", int), 0, 9999)
        _range("waypoint_id", _req(msg, "waypoint_id", int), 0, 9999)
        reason = _req(msg, "reason", str)
        if reason not in WAIT_REASONS:
            raise ContractError(f"unsupported WAIT reason: {reason}")
        if not self._reliable_pre(seq, f"WAIT|{reason}"):
            return
        self.state = "WAITING"
        self.wait_reason = reason
        self._reliable_done(seq, "ACCEPTED")

    def _on_go(self, msg: dict[str, Any]) -> None:
        _req(msg, "session_id", str)
        seq = _req(msg, "seq", int)
        route_id = _range("route_id", _req(msg, "route_id", int), 1, 9999)
        wp_id = _range("waypoint_id", _req(msg, "waypoint_id", int), 1, 9999)
        if not self._reliable_pre(seq, f"GO|{route_id}|{wp_id}"):
            return
        if self.state != "WAITING":
            self._reliable_done(seq, "INVALID_STATE")
            return
        if self.target is None:
            self._reliable_done(seq, "TARGET_NOT_LOADED")
            return
        if self.target["route_id"] != route_id:
            self._reliable_done(seq, "STALE_ROUTE")
            return
        self.state = "MOVING"
        self.wait_reason = "NONE"
        self._reliable_done(seq, "ACCEPTED")

    def _on_stop(self, msg: dict[str, Any]) -> None:
        _req(msg, "session_id", str)
        seq = _req(msg, "seq", int)
        if not self._reliable_pre(seq, "STOP"):
            return
        self.state = "EMERGENCY_STOP"
        self.target = None
        self._reliable_done(seq, "ACCEPTED")

    def _on_reset(self, msg: dict[str, Any]) -> None:
        _req(msg, "session_id", str)
        seq = _req(msg, "seq", int)
        if not self._reliable_pre(seq, "RESET"):
            return
        if self.state not in ("EMERGENCY_STOP", "ERROR"):
            self._reliable_done(seq, "INVALID_STATE")
            return
        self.state = "READY"
        self.wait_reason = "NONE"
        self._reliable_done(seq, "ACCEPTED")

    def _on_set_mode(self, msg: dict[str, Any]) -> None:
        _req(msg, "session_id", str)
        seq = _req(msg, "seq", int)
        mode = _req(msg, "mode", str)
        if not self._reliable_pre(seq, f"SET_MODE|{mode}"):
            return
        if self.state in ("EMERGENCY_STOP", "ERROR", "COMM_TIMEOUT", "MOVING"):
            self._reliable_done(seq, "INVALID_STATE")
            return
        self.mode = mode
        self._reliable_done(seq, "ACCEPTED")

    def _on_direct(self, msg: dict[str, Any]) -> None:
        _req(msg, "session_id", str)
        _req(msg, "control_seq", int)
        _range("throttle", _req(msg, "throttle", float), -1.0, 1.0)
        _range("steering", _req(msg, "steering", float), -1.0, 1.0)

    # ─── 멱등 처리 (§15) ─────────────────────────────────────────────────────

    def _reliable_pre(self, seq: int, fingerprint: str) -> bool:
        """이미 처리한 seq 면 재실행 없이 이전 결과만 재전송한다."""
        if seq == self.last_processed_cmd_seq:
            if fingerprint == self.last_fingerprint:
                self._status(result=self.last_result)
            else:
                self._status(result="SEQ_CONFLICT", rejected_seq=seq)
            return False
        if seq < self.last_processed_cmd_seq:
            self._status(result="STALE_SEQ", rejected_seq=seq)
            return False
        self.last_fingerprint = fingerprint
        return True

    def _reliable_done(self, seq: int, result: str) -> None:
        self.last_processed_cmd_seq = seq
        self.last_result = result
        self._status(result=result, rejected_seq=0 if result == "ACCEPTED" else seq)

    # ─── COMM 감시 ───────────────────────────────────────────────────────────

    def _hello_retry_loop(self) -> None:
        """READY 가 될 때까지 HELLO 를 재전송한다 (HOLD 면 재판정 기회를 준다)."""
        while self._alive:
            time.sleep(self.hello_interval)
            if not self._alive or self.hello_result == "REJECTED":
                return
            if self.hello_result != "READY_ALLOWED":
                try:
                    self._send(self._hello())
                except OSError:
                    return

    def _status_loop(self) -> None:
        """세션이 열린 뒤 주기 STATUS 송신 (command_result 는 NONE)."""
        while self._alive:
            time.sleep(self.status_interval)
            if self.hello_result == "READY_ALLOWED" and self._alive:
                try:
                    self._status()
                except OSError:
                    break

    def _watchdog(self) -> None:
        while self._alive:
            if self.hello_result == "READY_ALLOWED" and \
                    time.monotonic() - self.last_heartbeat_at > HEARTBEAT_TIMEOUT_S:
                if not self.comm_timeout_fired:
                    self.comm_timeout_fired = True
                    self.state = "COMM_TIMEOUT"
            time.sleep(0.05)

    def close(self) -> None:
        self._alive = False
        try:
            self.sock.close()
        except OSError:
            pass
