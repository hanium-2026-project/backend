"""신뢰성 명령 송신기 (§14~17).

- 차량당 일반 outstanding 명령 1개 (§16)
- 응답(STATUS의 ack_seq / last_processed_cmd_seq) 없으면 동일 seq·동일 payload 재전송
- WAIT/STOP은 대기 중 일반 명령을 선점 (§17), STOP > WAIT
- 같은 seq에 다른 payload 송신 시도는 프로그래밍 오류로 차단 (§15.2 예방)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .protocol import PREEMPTIVE_TYPES, TIMING


@dataclass
class _Pending:
    seq: int
    msg: dict[str, Any]
    sent_at: float
    attempts: int
    timeout_ms: int


class ReliableSender:
    """단일 차량용 신뢰성 명령 관리자.

    send_raw: 실제 소켓 송신 콜백 (server가 주입).
    on_fail: 최대 재전송 초과 시 호출 (COMM 이상 통지).
    tick()을 주기적으로 호출해 재전송 타이머를 구동한다 (스레드 안전).
    """

    def __init__(
        self,
        car_id: int,
        send_raw: Callable[[dict[str, Any]], None],
        on_fail: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.car_id = car_id
        self._send_raw = send_raw
        self._on_fail = on_fail
        self._seq = 0
        self._pending: _Pending | None = None
        self._lock = threading.Lock()

    # ─── 송신 ────────────────────────────────────────────────────────────────

    def set_seq_start(self, start: int) -> None:
        """HELLO_ACK 로 발급한 command_seq_start 에 맞춰 seq 카운터를 초기화한다."""
        with self._lock:
            self._seq = max(0, int(start) - 1)

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def send(self, msg: dict[str, Any]) -> int:
        """신뢰성 명령 송신. 반환값 = 부여된 seq.

        일반 명령: outstanding이 있으면 RuntimeError (§16 — 상위에서 순서 제어).
        WAIT/STOP: outstanding을 폐기하고 즉시 선점 송신 (§17).
        """
        msg_type = msg.get("type", "")
        with self._lock:
            if self._pending is not None:
                if msg_type not in PREEMPTIVE_TYPES:
                    raise RuntimeError(
                        f"outstanding command exists (seq={self._pending.seq}, "
                        f"type={self._pending.msg.get('type')}); wait for ack"
                    )
                # 선점: 기존 일반 명령 재전송 중단 (§17.1 '재전송 일시 중단/취소')
                self._pending = None

            self._seq += 1
            msg["seq"] = self._seq
            timeout = self._timeout_for(msg_type)
            self._pending = _Pending(self._seq, dict(msg), time.monotonic(), 1, timeout)
        self._send_raw(msg)
        return msg["seq"]

    # ─── 응답 처리 ───────────────────────────────────────────────────────────

    def on_ack(self, acked_seq: int) -> bool:
        """STATUS의 ack_seq / last_processed_cmd_seq 수신 처리."""
        with self._lock:
            if self._pending is not None and acked_seq >= self._pending.seq:
                self._pending = None
                return True
        return False

    @property
    def outstanding(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._pending.msg) if self._pending else None

    # ─── 재전송 타이머 ───────────────────────────────────────────────────────

    def tick(self) -> None:
        """주기 호출 — 타임아웃 시 동일 seq·동일 payload 재전송 (§15.1)."""
        resend: dict[str, Any] | None = None
        failed: dict[str, Any] | None = None
        with self._lock:
            p = self._pending
            if p is None:
                return
            elapsed_ms = (time.monotonic() - p.sent_at) * 1000.0
            if elapsed_ms < p.timeout_ms:
                return
            if p.attempts >= TIMING["MAX_RETRANSMIT"]:
                failed = dict(p.msg)
                self._pending = None
            else:
                p.attempts += 1
                p.sent_at = time.monotonic()
                resend = dict(p.msg)
        if resend is not None:
            self._send_raw(resend)
        if failed is not None and self._on_fail is not None:
            self._on_fail(failed)

    @staticmethod
    def _timeout_for(msg_type: str) -> int:
        return {
            "WAYPOINT": TIMING["RESP_TIMEOUT_WAYPOINT"],
            "WAIT": TIMING["RESP_TIMEOUT_WAIT"],
            "STOP": TIMING["RESP_TIMEOUT_STOP"],
        }.get(msg_type, TIMING["RESP_TIMEOUT_DEFAULT"])
