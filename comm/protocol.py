"""차량 제어 프로토콜 v2 — 시스템 통합 문서 기준.

- 노트북 = TCP Server, ESP32 = Client
- NDJSON (JSON 한 줄 + '\\n'), 최대 512 byte
- car_id = 정수(1, 2), slot_id = "A1"~"B4"
- 좌표 wire 단위 = cm, heading = 0~360° (우 0°, 상 90°)

메시지 분류
-----------
신뢰성 명령 (seq 멱등·재전송): SET_MODE, WAYPOINT, WAIT, GO, STOP, RESET
스트림 (최신값만):            POSE_UPDATE, DIRECT_CONTROL, HEARTBEAT, STATUS
연결 협상:                    HELLO(차→PC), HELLO_ACK(PC→차)

도착 판정은 노트북이 수행하므로 ARRIVED/EVENT_ACK는 사용하지 않는다
(수신 시 무시하지 않고 EVENT_ACK만 응답하는 하위호환 처리는 server가 담당).
"""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 512

# 신뢰성 명령 타입 (seq 부여 + 응답 대기 + 재전송 대상)
RELIABLE_TYPES = frozenset({"SET_MODE", "WAYPOINT", "WAIT", "GO", "STOP", "RESET"})
# 선점 명령 (outstanding 일반 명령을 무시하고 즉시 송신 가능)
PREEMPTIVE_TYPES = frozenset({"WAIT", "STOP"})

# §35 타이밍 초기값 (ms) — 실측 후 조정
TIMING = {
    "HEARTBEAT_INTERVAL": 250,
    "COMM_TIMEOUT": 1000,
    "POSE_INTERVAL": 100,
    "POSE_TIMEOUT": 400,
    "STATUS_INTERVAL": 150,
    "RESP_TIMEOUT_WAYPOINT": 300,
    "RESP_TIMEOUT_WAIT": 100,
    "RESP_TIMEOUT_STOP": 100,
    "RESP_TIMEOUT_DEFAULT": 300,
    "MAX_RETRANSMIT": 5,
}

# 차량 상태 (§24)
VEHICLE_STATES = frozenset({
    "BOOT", "WIFI_CONNECTING", "SYNCING", "READY", "MOVING",
    "WAITING", "EMERGENCY_STOP", "COMM_TIMEOUT", "ERROR",
})

# 제어 모드 (§5)
CONTROL_MODES = frozenset({"MANUAL_SERIAL", "REMOTE_DIRECT", "WAYPOINT_AUTO"})


# ─── 명령 빌더 (PC → ESP32) ──────────────────────────────────────────────────

def _base(msg_type: str, car_id: int, session_id: str) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": msg_type,
        "car_id": int(car_id),
        "session_id": session_id,
    }


def make_waypoint(
    car_id: int, session_id: str, seq: int, wire_waypoint: dict[str, Any]
) -> dict[str, Any]:
    """단건 WAYPOINT (§21). wire_waypoint = Waypoint.to_wire() 결과."""
    msg = _base("WAYPOINT", car_id, session_id)
    msg["seq"] = seq
    msg.update(wire_waypoint)          # route_id/waypoint_id/phase/x_cm/... 포함
    msg.setdefault("motion_direction", "FORWARD")
    msg.setdefault("arrival_mode", "STOP")
    return msg


def make_go(car_id: int, session_id: str, seq: int, route_id: int, waypoint_id: int) -> dict[str, Any]:
    msg = _base("GO", car_id, session_id)
    msg.update(seq=seq, route_id=route_id, waypoint_id=waypoint_id)
    return msg


def make_wait(car_id: int, session_id: str, seq: int, reason: str = "REMOTE_WAIT") -> dict[str, Any]:
    msg = _base("WAIT", car_id, session_id)
    msg.update(seq=seq, reason=reason)
    return msg


def make_stop(car_id: int, session_id: str, seq: int) -> dict[str, Any]:
    msg = _base("STOP", car_id, session_id)
    msg["seq"] = seq
    return msg


def make_reset(car_id: int, session_id: str, seq: int) -> dict[str, Any]:
    msg = _base("RESET", car_id, session_id)
    msg["seq"] = seq
    return msg


def make_set_mode(car_id: int, session_id: str, seq: int, mode: str) -> dict[str, Any]:
    if mode not in CONTROL_MODES:
        raise ValueError(f"unknown mode: {mode}")
    msg = _base("SET_MODE", car_id, session_id)
    msg.update(seq=seq, mode=mode)
    return msg


def make_pose_update(
    car_id: int, session_id: str, pose_seq: int,
    x_cm: float, y_cm: float, heading_deg: float | None,
    position_confidence: float, heading_confidence: float,
    heading_source: str | None, measurement_age_ms: int, valid: bool = True,
) -> dict[str, Any]:
    """POSE_UPDATE (§19) — 스트림, seq 재전송 없음."""
    msg = _base("POSE_UPDATE", car_id, session_id)
    msg.update(
        pose_seq=pose_seq,
        x_cm=round(x_cm, 1), y_cm=round(y_cm, 1),
        heading_deg=round(heading_deg, 1) if heading_deg is not None else None,
        position_confidence=round(position_confidence, 2),
        heading_confidence=round(heading_confidence, 2),
        heading_source=heading_source,
        measurement_age_ms=int(measurement_age_ms),
        valid=bool(valid),
    )
    return msg


def make_direct_control(car_id: int, session_id: str, speed: float, steering: float) -> dict[str, Any]:
    """REMOTE_DIRECT 모드 개발용 (§18.2). 최신값만, deadman timeout은 ESP 처리."""
    msg = _base("DIRECT_CONTROL", car_id, session_id)
    msg.update(speed=round(speed, 1), steering=round(steering, 1))
    return msg


def make_hello_ack(
    car_id: int, session_id: str, result: str, reason: str | None = None
) -> dict[str, Any]:
    """HELLO_ACK (§26.2). result ∈ READY_ALLOWED | HOLD | REJECTED."""
    if result not in ("READY_ALLOWED", "HOLD", "REJECTED"):
        raise ValueError(f"invalid HELLO_ACK result: {result}")
    msg = _base("HELLO_ACK", car_id, session_id)
    msg.update(result=result, reason=reason)
    return msg


def make_event_ack(car_id: int, session_id: str, event_id: int) -> dict[str, Any]:
    """하위호환: ESP가 ARRIVED를 보내는 펌웨어일 경우에만 응답."""
    msg = _base("EVENT_ACK", car_id, session_id)
    msg["event_id"] = event_id
    return msg


# ─── 직렬화 (§12.3~12.5) ─────────────────────────────────────────────────────

def encode(msg: dict[str, Any]) -> bytes:
    """NDJSON 인코딩. 512 byte 초과 시 예외 (송신 전 검출)."""
    data = (json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message exceeds {MAX_MESSAGE_BYTES} bytes: {len(data)}")
    return data


def parse_message(line: bytes | str) -> dict[str, Any]:
    """수신 라인 파싱 + 필수 필드/버전 검증. 실패 시 ValueError."""
    if isinstance(line, bytes) and len(line) > MAX_MESSAGE_BYTES:
        raise ValueError("oversized message")
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("message must be a JSON object")
    for key in ("type", "car_id"):
        if key not in obj:
            raise ValueError(f"missing required field: {key}")
    obj["car_id"] = int(obj["car_id"])
    return obj
