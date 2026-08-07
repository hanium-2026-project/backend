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

# 선점 우선순위 (§16): STOP > WAIT > 일반 신뢰성 명령.
# 숫자가 클수록 우선. 같거나 낮은 우선순위는 진행 중인 명령을 밀어내지 못한다.
_PRIORITY: dict[str, int] = {"STOP": 2, "WAIT": 1}


def _priority(msg_type: str) -> int:
    return _PRIORITY.get(msg_type, 0)


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
        WAIT/STOP: 더 낮은 우선순위의 명령만 선점한다 (§17).

        STOP 이 응답을 기다리는 중에 WAIT 이 이를 취소해서는 안 된다.
        비상정지가 일시정지로 뒤집히는 셈이라 안전상 허용할 수 없다.
        """
        msg_type = msg.get("type", "")
        with self._lock:
            pending = self._pending
            if pending is not None:
                pending_type = str(pending.msg.get("type", ""))
                if msg_type not in PREEMPTIVE_TYPES:
                    raise RuntimeError(
                        f"outstanding command exists (seq={pending.seq}, "
                        f"type={pending_type}); wait for ack"
                    )
                if _priority(msg_type) <= _priority(pending_type):
                    raise RuntimeError(
                        f"{msg_type} cannot preempt pending {pending_type} "
                        f"(seq={pending.seq}) — 우선순위가 같거나 낮다"
                    )
                # 선점: 더 높은 우선순위만 기존 명령의 재전송을 중단시킨다 (§17.1)
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

    def clear_pending(self) -> None:
        """진행 중인 명령을 폐기한다 (링크 단절 등 — 재전송해도 의미가 없을 때).

        링크가 끊긴 채 pending 이 남아 있으면 복구 후 새 명령이 전부
        "outstanding command exists" 로 막힌다.
        """
        with self._lock:
            self._pending = None

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
