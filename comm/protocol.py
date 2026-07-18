"""차량 제어 메시지 스키마 (JSON Lines).

전송 방향
---------
PC → 라즈베리파이 (명령):
    WAYPOINTS  경로 등록 (route_id + waypoint 목록, cm 단위)
    GO         등록된 경로 주행 시작/재개
    WAIT       즉시 정지 (최우선 처리)

라즈베리파이 → PC (보고):
    STATE      vehicle_state 변경 보고 (READY/MOVING/WAITING/ERROR)
    ARRIVED    waypoint 도착 보고 (route_id, waypoint_id 포함)
    ACK        명령 수신 확인 (seq 에코)

공통 필드
---------
    seq        송신측 단조 증가 순번 — 수신측은 오래된 seq 무시
    car_id     차량 식별자
    type       메시지 종류

회의 8번 규칙: route_id가 현재 활성 route와 다른 ARRIVED/상태 보고는 무시한다.
WAITING 상태에서 WAYPOINTS를 새로 등록해도 GO 전까지 이동하지 않는다.
"""

from __future__ import annotations

import json
from itertools import count
from typing import Any

_seq_counter = count(1)


def _base(msg_type: str, car_id: int) -> dict[str, Any]:
    return {"type": msg_type, "seq": next(_seq_counter), "car_id": car_id}


# ─── PC → 차량 명령 ──────────────────────────────────────────────────────────

def make_waypoints_msg(car_id: int, route_id: int, waypoints: list[dict[str, Any]]) -> dict[str, Any]:
    """경로 등록 명령. waypoints는 Waypoint.to_wire() 결과(cm 단위) 목록."""
    msg = _base("WAYPOINTS", car_id)
    msg["route_id"] = route_id
    msg["waypoints"] = waypoints
    return msg


def make_go_msg(car_id: int, route_id: int) -> dict[str, Any]:
    """주행 시작/재개. route_id가 차량의 등록 경로와 일치해야 유효."""
    msg = _base("GO", car_id)
    msg["route_id"] = route_id
    return msg


def make_wait_msg(car_id: int) -> dict[str, Any]:
    """즉시 정지 — 차량은 다른 어떤 명령보다 우선 처리해야 한다."""
    return _base("WAIT", car_id)


# ─── 직렬화 ──────────────────────────────────────────────────────────────────

def encode(msg: dict[str, Any]) -> bytes:
    """JSON Lines 인코딩 (개행 종단)."""
    return (json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def parse_message(line: bytes | str) -> dict[str, Any]:
    """수신 라인 1건 파싱. 필수 필드(type, seq, car_id) 검증."""
    obj = json.loads(line)
    for key in ("type", "seq", "car_id"):
        if key not in obj:
            raise ValueError(f"missing required field: {key}")
    return obj
