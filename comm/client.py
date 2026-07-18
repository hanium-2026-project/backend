"""PC측 차량 통신 클라이언트 (라즈베리파이 relay 서버로 접속).

- 라즈베리파이가 TCP 서버(listen), PC가 클라이언트로 접속한다.
- 송신: WAYPOINTS / GO / WAIT (protocol.py 빌더 사용)
- 수신: STATE / ARRIVED / ACK — 콜백으로 전달하되, 현재 활성 route_id와
  다른 route의 주행 이벤트는 여기서 걸러 무시한다 (회의 8번).
"""

from __future__ import annotations

import socket
import threading
from typing import Any, Callable

from . import protocol
from .protocol import encode, parse_message


class VehicleLink:
    """단일 차량(라즈베리파이)과의 TCP 링크.

    Usage::

        link = VehicleLink("192.168.0.42", 9000, car_id=1)
        link.connect(on_message=handle)
        link.send_waypoints(route_id=1, waypoints=[wp.to_wire() for wp in wps])
        link.send_go(route_id=1)
        ...
        link.send_wait()          # 비상 정지
        link.close()
    """

    def __init__(self, host: str, port: int, car_id: int, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.car_id = car_id
        self.timeout = timeout
        self.active_route_id: int | None = None
        self._sock: socket.socket | None = None
        self._rx_thread: threading.Thread | None = None
        self._running = False

    # ─── 연결 관리 ───────────────────────────────────────────────────────────

    def connect(self, on_message: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(None)
        self._running = True
        self._rx_thread = threading.Thread(
            target=self._rx_loop, args=(on_message,), daemon=True
        )
        self._rx_thread.start()

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    # ─── 명령 송신 ───────────────────────────────────────────────────────────

    def send_waypoints(self, route_id: int, waypoints: list[dict[str, Any]]) -> None:
        """새 경로 등록. 기존 활성 route는 폐기된 것으로 간주한다."""
        self.active_route_id = route_id
        self._send(protocol.make_waypoints_msg(self.car_id, route_id, waypoints))

    def send_go(self, route_id: int | None = None) -> None:
        rid = route_id if route_id is not None else self.active_route_id
        if rid is None:
            raise RuntimeError("no active route: send_waypoints() first")
        self._send(protocol.make_go_msg(self.car_id, rid))

    def send_wait(self) -> None:
        """비상 정지 — 활성 route와 무관하게 즉시 송신."""
        self._send(protocol.make_wait_msg(self.car_id))

    def _send(self, msg: dict[str, Any]) -> None:
        if self._sock is None:
            raise RuntimeError("not connected")
        self._sock.sendall(encode(msg))

    # ─── 수신 루프 ───────────────────────────────────────────────────────────

    def _rx_loop(self, on_message: Callable[[dict[str, Any]], None] | None) -> None:
        buf = b""
        while self._running and self._sock is not None:
            try:
                chunk = self._sock.recv(4096)
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
                except (ValueError, UnicodeDecodeError):
                    continue                      # 손상 라인은 버림
                if self._is_stale(msg):
                    continue                      # 폐기된 route의 늦은 보고 무시
                if on_message is not None:
                    on_message(msg)

    def _is_stale(self, msg: dict[str, Any]) -> bool:
        """주행 이벤트(ARRIVED 등)가 폐기된 route에서 온 것인지 판정."""
        if "route_id" not in msg:
            return False                          # STATE/ACK 등 route 무관 메시지
        return self.active_route_id is not None and msg["route_id"] != self.active_route_id
