"""노트북측 TCP 서버 (§12.1, §26) — ESP32 클라이언트를 수용한다.

역할
----
- accept → HELLO 수신 → 판정(HELLO_ACK 초안 기준) → session_id 발급
- car_id별 활성 세션 관리 (중복 연결 시 기존 session 무효화, §26.5)
- NDJSON 수신 루프: STATUS(ack 겸용)/HELLO/ARRIVED(하위호환) 분류
- 차량별 ReliableSender 보유, POSE_UPDATE 스트림(최신값만) 송신 스레드
- 재접속 시 자동 재개 금지: 세션이 새로 열리면 활성 route를 무효화하고
  상위(on_resync 콜백)에 재계획을 요청한다 (§21·26.3·26.4)
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import protocol
from .protocol import TIMING, encode, parse_message
from .reliability import ReliableSender


@dataclass
class VehicleSession:
    """활성 차량 세션 1개 (car_id당 최대 1개)."""

    car_id: int
    session_id: str
    boot_id: str
    conn: socket.socket
    sender: ReliableSender
    last_status: dict[str, Any] = field(default_factory=dict)
    last_rx_ms: float = field(default_factory=lambda: time.monotonic() * 1000)
    pose_seq: int = 0
    heartbeat_seq: int = 0
    latest_pose: dict[str, Any] | None = None      # 최신값만 (§19)
    control_seq: int = 0
    latest_control: dict[str, Any] | None = None   # DIRECT_CONTROL 최신값만
    latest_control_at: float = 0.0                 # 갱신 시각 (ms) — 신선도 판정용
    alive: bool = True
    comm_failed: bool = False                      # 통신 장애 통지 상태 (엣지 판정용)


class VehicleServer:
    """차량 통신 서버.

    Usage::

        srv = VehicleServer(port=9000, known_car_ids={1, 2})
        srv.on_status = lambda car_id, st: ...
        srv.on_ready = lambda car_id: ...          # HELLO_ACK 승인 완료 시
        srv.on_resync = lambda car_id, hello: ...  # 재접속 → 재계획 필요 통지
        srv.start()
        srv.send_waypoint(1, wp.to_wire()); srv.send_go(1, route_id, wp_id)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000,
                 known_car_ids: set[int] | None = None) -> None:
        self.host, self.port = host, port
        self.known_car_ids = known_car_ids or {1, 2}
        self.sessions: dict[int, VehicleSession] = {}
        self._lock = threading.Lock()
        self._srv_sock: socket.socket | None = None
        self._running = False

        # 상위 콜백
        self.on_status: Callable[[int, dict[str, Any]], None] | None = None
        self.on_ready: Callable[[int], None] | None = None
        self.on_resync: Callable[[int, dict[str, Any]], None] | None = None
        # 통신 장애 통지 — 장애 "시작" 시 1회만 호출된다 (같은 장애 중 반복 없음)
        self.on_comm_fail: Callable[[int, dict[str, Any]], None] | None = None
        # 장애 이후 수신이 재개되면 1회 호출
        self.on_comm_recovered: Callable[[int], None] | None = None
        # B안 제어 스트림 송신 여부. 실차 안전을 위해 기본은 꺼둔다.
        self.direct_control_enabled = False
        # 제어값이 이 시간 넘게 갱신되지 않으면 0 을 대신 보낸다.
        # 갱신이 끊겼는데 마지막 값을 계속 재전송하면 차가 그대로 달린다 —
        # 게다가 스트림이 계속 도착하므로 펌웨어의 500ms DIRECT 타임아웃도 안 걸린다.
        self.control_stale_ms = 300.0
        # 신뢰성 명령이 거절된 경우 (car_id, result, status) — 상위에서 복구 판단
        self.on_command_rejected: Callable[[int, str, dict[str, Any]], None] | None = None
        # terminal 결과 전체 (car_id, seq, result, status). 거절뿐 아니라 ACCEPTED 도 통지한다 —
        # REMOTE_DIRECT 전환이 실제로 수락됐는지 확인해야 제어값을 내보낼 수 있다.
        self.on_command_result: Callable[[int, int, str, dict[str, Any]], None] | None = None
        # HOLD 판정 훅: 카메라 검출 여부 등 외부 조건 (car_id → 사유 문자열 | None)
        self.hold_check: Callable[[int, dict[str, Any]], str | None] | None = None
        # HOLD 로 판정될 때마다 호출 (car_id, 사유) — 상위에서 원인 해소를 유도
        self.on_hold: Callable[[int, str], None] | None = None
        # HOLD 가 이만큼 반복되면 연결을 정리한다 (무한 재시도 방지)
        self.max_hold_retries = 20
        # 재접속 HELLO 의 previous_state=EMERGENCY_STOP 을 HOLD 로 볼지.
        # 기본 False — 켜면 복구 불가 고리에 빠진다 (_judge_hello 주석 참고).
        self.hold_on_estop_history = False

    # ─── 라이프사이클 ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._srv_sock = socket.socket()
        self._srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv_sock.bind((self.host, self.port))
        self._srv_sock.listen(4)
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._tick_loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for s in self.sessions.values():
                s.alive = False
                try: s.conn.close()
                except OSError: pass
            self.sessions.clear()
        if self._srv_sock is not None:
            self._srv_sock.close()

    @property
    def bound_port(self) -> int:
        return self._srv_sock.getsockname()[1] if self._srv_sock else self.port

    # ─── 송신 API ────────────────────────────────────────────────────────────

    def send_waypoint(self, car_id: int, wire_wp: dict[str, Any]) -> int:
        s = self._session(car_id)
        return s.sender.send(protocol.make_waypoint(car_id, s.session_id, 0, wire_wp))

    def send_go(self, car_id: int, route_id: int, waypoint_id: int) -> int:
        s = self._session(car_id)
        return s.sender.send(protocol.make_go(car_id, s.session_id, 0, route_id, waypoint_id))

    def send_wait(self, car_id: int, reason: str = "REMOTE_WAIT",
                  route_id: int = 0, waypoint_id: int = 0) -> int:
        """WAIT 송신. 펌웨어가 route_id/waypoint_id/reason 을 모두 요구한다."""
        s = self._session(car_id)
        return s.sender.send(
            protocol.make_wait(car_id, s.session_id, 0, route_id, waypoint_id, reason))

    def send_stop(self, car_id: int) -> int:
        s = self._session(car_id)
        return s.sender.send(protocol.make_stop(car_id, s.session_id, 0))

    def send_reset(self, car_id: int) -> int:
        s = self._session(car_id)
        return s.sender.send(protocol.make_reset(car_id, s.session_id, 0))

    def send_set_mode(self, car_id: int, mode: str) -> int:
        s = self._session(car_id)
        return s.sender.send(protocol.make_set_mode(car_id, s.session_id, 0, mode))

    def push_control(self, car_id: int, throttle: float, steering: float) -> None:
        """DIRECT_CONTROL 스트림 갱신 (B안). 실제 송신은 tick 루프가 주기 수행.

        스트림이므로 ack·재전송이 없다. 최신값만 유지하며, 갱신이 끊겨도
        마지막 값을 계속 보낸다 — 차량은 HEARTBEAT 가 끊기면 자체 안전정지한다.
        """
        with self._lock:
            s = self.sessions.get(car_id)
            if s is None:
                return
            s.control_seq += 1
            s.latest_control = protocol.make_direct_control(
                car_id, s.session_id, s.control_seq, throttle, steering)
            s.latest_control_at = time.monotonic() * 1000

    def stop_control(self, car_id: int) -> None:
        """제어 스트림을 0 으로 고정한다 (미션 종료·정지)."""
        self.push_control(car_id, 0.0, 0.0)

    def clear_outstanding(self, car_id: int) -> None:
        """진행 중인 신뢰성 명령을 폐기한다 (재계획 등으로 무효가 됐을 때)."""
        with self._lock:
            sess = self.sessions.get(car_id)
        if sess is not None:
            sess.sender.clear_pending()

    def push_pose(self, car_id: int, x_cm: float, y_cm: float,
                  heading_deg: float | None, heading_source: str | None,
                  position_confidence: float = 1.0, heading_confidence: float = 0.0,
                  measurement_age_ms: int = 0, valid: bool = True) -> None:
        """POSE 스트림 갱신 — 최신값만 저장, 송신은 tick 루프가 주기 수행."""
        with self._lock:
            s = self.sessions.get(car_id)
            if s is None:
                return
            s.pose_seq += 1
            s.latest_pose = protocol.make_pose_update(
                car_id, s.session_id, s.pose_seq, x_cm, y_cm, heading_deg,
                position_confidence, heading_confidence, heading_source,
                measurement_age_ms, valid,
            )

    def last_status(self, car_id: int) -> dict[str, Any]:
        with self._lock:
            s = self.sessions.get(car_id)
            return dict(s.last_status) if s else {}

    def _session(self, car_id: int) -> VehicleSession:
        with self._lock:
            s = self.sessions.get(car_id)
        if s is None or not s.alive:
            raise RuntimeError(f"car {car_id}: no active session")
        return s

    # ─── accept / HELLO 판정 (§26) ───────────────────────────────────────────

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._srv_sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handshake, args=(conn,), daemon=True).start()

    def _handshake(self, conn: socket.socket) -> None:
        """HELLO 를 받아 판정하고 세션을 연다.

        HOLD 는 "조건이 해소되면 다시 판정" 이라는 뜻이므로 연결을 끊지 않는다
        (§26.2). 차량은 HELLO_ACK 를 못 받거나 HOLD 를 받으면 HELLO 를 재전송하므로,
        여기서 다음 HELLO 를 기다렸다가 재판정한다. REJECTED 만 연결을 닫는다.
        """
        conn.settimeout(TIMING["COMM_TIMEOUT"] / 1000.0 * 3)
        buf = b""
        rest = b""
        hold_count = 0

        while True:
            try:
                while b"\n" not in buf:
                    chunk = conn.recv(1024)
                    if not chunk:
                        conn.close(); return
                    buf += chunk
                line, rest = buf.split(b"\n", 1)
                buf = rest
                hello = parse_message(line)
            except (OSError, ValueError):
                conn.close(); return

            if hello.get("type") != "HELLO":
                continue                     # HELLO 가 올 때까지 다른 메시지는 흘려보낸다

            car_id = hello.get("car_id", -1)
            boot_id = str(hello.get("boot_id", ""))
            result, reason = self._judge_hello(hello)
            session_id = ("S" + secrets.token_hex(4).upper()
                          if result == "READY_ALLOWED" else "")
            # 새 세션의 명령 seq 시작값 — 이전 세션 seq 와 겹치지 않게 1부터 새로 발급
            command_seq_start = 1
            try:
                conn.sendall(encode(protocol.make_hello_ack(
                    car_id, session_id, result, reason,
                    boot_id=boot_id, command_seq_start=command_seq_start)))
            except OSError:
                conn.close(); return

            if result == "REJECTED":
                conn.close(); return
            if result == "HOLD":
                hold_count += 1
                # HOLD 중에도 링크는 살려둬야 한다. HEARTBEAT 가 끊기면 차량이
                # 1초 뒤 COMM_TIMEOUT 으로 떨어져 나가 조건을 해소할 기회조차 없다.
                threading.Thread(target=self._hold_heartbeat, args=(conn, car_id),
                                 daemon=True).start()
                if self.on_hold is not None:
                    self.on_hold(car_id, reason or "HOLD")
                if hold_count >= self.max_hold_retries:
                    log_reason = reason or "HOLD"
                    self._safe_close(conn)
                    if self.on_comm_fail is not None:
                        self.on_comm_fail(car_id, {"type": "HOLD_EXHAUSTED",
                                                   "reason": log_reason})
                    return
                continue                     # 다음 HELLO 를 기다려 재판정
            break                            # READY_ALLOWED

        # 기존 세션 무효화 (§26.5) 후 신규 등록
        with self._lock:
            old = self.sessions.get(car_id)
            if old is not None:
                old.alive = False
                try: old.conn.close()
                except OSError: pass
            sender = ReliableSender(
                car_id,
                send_raw=lambda m, c=conn: self._safe_send(c, m),
                on_fail=lambda m, cid=car_id: self._comm_fail(cid, m),
            )
            sender.set_seq_start(command_seq_start)
            sess = VehicleSession(car_id, session_id, boot_id, conn, sender)
            self.sessions[car_id] = sess

        conn.settimeout(None)
        # 재접속/재부팅 공통: 자동 재개 금지 → 상위에 재계획 통지 (§21·26.3·26.4)
        if self.on_resync is not None:
            self.on_resync(car_id, hello)
        if self.on_ready is not None:
            self.on_ready(car_id)
        self._rx_loop(sess, rest)

    def _hold_heartbeat(self, conn: socket.socket, car_id: int) -> None:
        """HOLD 대기 구간 전용 HEARTBEAT. 세션이 열리면 tick 루프가 이어받는다."""
        seq = 0
        deadline = time.monotonic() + TIMING["COMM_TIMEOUT"] / 1000.0 * 4
        while time.monotonic() < deadline:
            with self._lock:
                if car_id in self.sessions:
                    return                      # 세션 생김 → tick 루프가 담당
            seq += 1
            try:
                conn.sendall(encode(protocol.make_heartbeat(car_id, "", seq)))
            except OSError:
                return
            time.sleep(TIMING["HEARTBEAT_INTERVAL"] / 1000.0)

    def _judge_hello(self, hello: dict[str, Any]) -> tuple[str, str | None]:
        """HELLO_ACK 판정 초안 (REJECTED → HOLD → READY_ALLOWED)."""
        if hello.get("type") != "HELLO":
            return "REJECTED", "NOT_HELLO"
        if hello.get("version") != protocol.PROTOCOL_VERSION:
            return "REJECTED", "VERSION_MISMATCH"                       # R1
        if hello.get("car_id") not in self.known_car_ids:
            return "REJECTED", "UNKNOWN_CAR_ID"                         # R2
        if str(hello.get("error_code", "NONE")) not in ("NONE", ""):
            return "HOLD", "ERROR_NOT_CLEARED"                          # H1
        if self.hold_on_estop_history and \
                hello.get("previous_state") == "EMERGENCY_STOP":
            # 기본 비활성. 실물 확인(2026-08-10): 펌웨어는 통신이 한 번만 끊겨도
            # EMERGENCY_STOP 으로 가고, 재접속 HELLO 에 previous_state=EMERGENCY_STOP
            # 을 실어 보낸다. 이걸 HOLD 로 잡으면 세션이 안 열리고 → HEARTBEAT 를
            # 못 보내고 → 차량이 다시 COMM_TIMEOUT 으로 EMERGENCY_STOP 이 되어
            # 영원히 빠져나올 수 없다 (EMERGENCY_STOP 해제는 RESET 뿐인데 RESET 을
            # 보내려면 세션이 있어야 한다). 정지 상태 유지는 펌웨어가 이미 강제하므로
            # (RESET 전에는 GO/DIRECT_CONTROL 을 받지 않는다) 여기서 막을 필요가 없다.
            return "HOLD", "ESTOP_HISTORY"                              # H2
        if self.hold_check is not None:
            reason = self.hold_check(int(hello["car_id"]), hello)
            if reason:
                return "HOLD", reason                                   # H3/H4 (외부 훅)
        return "READY_ALLOWED", None

    # ─── 수신 루프 ───────────────────────────────────────────────────────────

    def _rx_loop(self, sess: VehicleSession, initial: bytes = b"") -> None:
        buf = initial
        while self._running and sess.alive:
            try:
                chunk = sess.conn.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = parse_message(line)
                except ValueError:
                    continue
                self._dispatch(sess, msg)
        sess.alive = False

    def _dispatch(self, sess: VehicleSession, msg: dict[str, Any]) -> None:
        sess.last_rx_ms = time.monotonic() * 1000
        self._comm_recovered(sess)
        # 이전 세션의 늦은 메시지 차단 (§13.3)
        if msg.get("session_id") and msg["session_id"] != sess.session_id:
            return
        # 소켓이 속한 차량과 다른 car_id 가 실려 오면 신뢰하지 않는다.
        # 정상 펌웨어는 자기 CAR_ID 만 보내므로, 불일치는 배선/설정 오류 신호다.
        if msg.get("car_id") != sess.car_id:
            return
        mtype = msg.get("type")
        if mtype in ("STATUS", "COMMAND_RESULT"):
            # 오래된 스냅샷은 상태를 되돌리지 않는다. 단, 명령 응답은 여전히 유효할
            # 수 있으므로 ack 판정은 스냅샷 갱신과 분리해서 먼저 수행한다 (§13.6)
            self._consume_command_result(sess, msg)
            if mtype == "COMMAND_RESULT":
                return
            if msg.get("status_seq", -1) <= sess.last_status.get("status_seq", -1):
                return
            sess.last_status = msg
            if self.on_status is not None:
                self.on_status(sess.car_id, msg)
        elif mtype == "ARRIVED":
            # 하위호환 (§22.4): 판정은 노트북이 하므로 ACK만 응답하고 무시
            if "event_id" in msg:
                self._safe_send(sess.conn, protocol.make_event_ack(
                    sess.car_id, sess.session_id, int(msg["event_id"])))

    def _consume_command_result(self, sess: VehicleSession, msg: dict[str, Any]) -> None:
        """신뢰성 명령의 최종 응답만 ack 로 인정한다.

        주기 STATUS 는 command_result=NONE 과 함께 last_processed_cmd_seq 를 계속
        반복하므로, 그것을 승인으로 처리하면 실행되지도 않은 명령이 완료 처리된다.
        따라서 terminal 결과값이면서 seq 가 일치할 때만 outstanding 을 해제한다.
        """
        pending = sess.sender.outstanding
        if pending is None:
            return
        result = str(msg.get("command_result", "NONE"))
        if result not in protocol.TERMINAL_RESULTS:
            return
        seq = pending.get("seq")
        acked = msg.get("last_processed_cmd_seq")
        rejected = msg.get("rejected_seq", msg.get("ack_seq"))
        if acked != seq and rejected != seq:
            return
        sess.sender.on_ack(int(seq))
        if self.on_command_result is not None:
            self.on_command_result(sess.car_id, int(seq), result, msg)
        if result in protocol.NEGATIVE_RESULTS and self.on_command_rejected is not None:
            self.on_command_rejected(sess.car_id, result, msg)

    # ─── 주기 루프: 재전송 + POSE 스트림 + COMM 감시 ─────────────────────────

    def _tick_loop(self) -> None:
        last_pose_ms = 0.0
        last_hb_ms = 0.0
        last_control_ms = 0.0
        while self._running:
            now_ms = time.monotonic() * 1000
            with self._lock:
                sessions = list(self.sessions.values())
            for s in sessions:
                if not s.alive:
                    continue
                s.sender.tick()
                if now_ms - s.last_rx_ms > TIMING["COMM_TIMEOUT"]:
                    # 장애가 지속되는 동안 매 주기 통지하지 않는다 — 상위 미션 로직이
                    # 같은 장애로 반복 트리거되면 복구 처리가 계속 리셋된다.
                    self._comm_fail(s.car_id, {"type": "COMM_TIMEOUT"})
            # HEARTBEAT — 노트북이 주기 송신해야 차량이 COMM_TIMEOUT 에 빠지지 않는다
            if now_ms - last_hb_ms >= TIMING["HEARTBEAT_INTERVAL"]:
                last_hb_ms = now_ms
                for s in sessions:
                    if not s.alive:
                        continue
                    s.heartbeat_seq += 1
                    self._safe_send(s.conn, protocol.make_heartbeat(
                        s.car_id, s.session_id, s.heartbeat_seq))
            # DIRECT_CONTROL — B안 제어값 스트림 (§18.2)
            if self.direct_control_enabled and \
                    now_ms - last_control_ms >= TIMING["CONTROL_INTERVAL"]:
                last_control_ms = now_ms
                for s in sessions:
                    if not s.alive or s.latest_control is None:
                        continue
                    if now_ms - s.latest_control_at > self.control_stale_ms:
                        s.control_seq += 1
                        s.latest_control = protocol.make_direct_control(
                            s.car_id, s.session_id, s.control_seq, 0.0, 0.0)
                        s.latest_control_at = now_ms
                    self._safe_send(s.conn, s.latest_control)
            # POSE_UPDATE — 펌웨어 수신 enum 에 추가되기 전까지는 비활성
            if protocol.POSE_UPDATE_ENABLED and now_ms - last_pose_ms >= TIMING["POSE_INTERVAL"]:
                last_pose_ms = now_ms
                for s in sessions:
                    if s.alive and s.latest_pose is not None:
                        self._safe_send(s.conn, s.latest_pose)
            time.sleep(0.02)

    # ─── 내부 유틸 ───────────────────────────────────────────────────────────

    @staticmethod
    def _safe_close(conn: socket.socket) -> None:
        try:
            conn.close()
        except OSError:
            pass

    def _safe_send(self, conn: socket.socket, msg: dict[str, Any]) -> None:
        try:
            conn.sendall(encode(msg))
        except (OSError, ValueError):
            pass

    def _comm_fail(self, car_id: int, msg: dict[str, Any]) -> None:
        """장애 시작 엣지에서만 상위에 통지한다."""
        with self._lock:
            sess = self.sessions.get(car_id)
            if sess is not None:
                if sess.comm_failed:
                    return                     # 이미 통지된 장애 — debounce
                sess.comm_failed = True
                # 끊긴 링크에 재전송해도 소용없다. pending 을 비워 복구 후
                # 새 명령이 막히지 않게 한다.
                sess.sender.clear_pending()
        if self.on_comm_fail is not None:
            self.on_comm_fail(car_id, msg)

    def _comm_recovered(self, sess: VehicleSession) -> None:
        """수신이 재개되면 장애 상태를 푼다 (엣지에서 1회 통지)."""
        if not sess.comm_failed:
            return
        sess.comm_failed = False
        if self.on_comm_recovered is not None:
            self.on_comm_recovered(sess.car_id)
