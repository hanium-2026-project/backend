"""차량 제어 프로토콜 v2 — 시스템 통합 문서 기준.

- 노트북 = TCP Server, ESP32 = Client
- NDJSON (JSON 한 줄 + '\\n'), 최대 512 byte
- car_id: 내부 정수(1, 2) ↔ wire 문자열("CAR_01") — 펌웨어가 strcmp 비교하므로
  직렬화 경계에서만 변환한다 (wire_car_id / parse_car_id)
- 좌표 wire 단위 = cm, heading = 0~360° (우 0°, 상 90°)

펌웨어 계약(integrated/esp32_main/main/protocol.c) 준수 사항
------------------------------------------------------------
- HELLO_ACK: boot_id 에코 + command_seq_start 필수 (REJECTED 제외)
- HEARTBEAT: 노트북이 250ms 주기로 송신 (미송신 시 차량 COMM_TIMEOUT)
- WAYPOINT: route_id/waypoint_id ≥ 1, target_heading_deg 는 null 불가(0~359.999)
- WAIT: route_id/waypoint_id/reason 모두 필수, reason 은 WAIT_REASONS 중 하나
- ACK: STATUS 의 command_result 가 terminal 값이고 seq 가 일치할 때만 승인
  (주기 STATUS 의 command_result=NONE 은 승인이 아님)

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

# WAIT 사유 — 펌웨어 parse_wait_reason() 이 받는 값만 유효
WAIT_REASONS = frozenset({
    "REMOTE_WAIT", "COLLISION_RISK", "OBSTACLE", "REROUTING",
    "WAYPOINT_REACHED", "FINAL_WAYPOINT_REACHED", "OPERATOR_REQUEST",
})

# 신뢰성 명령의 최종 응답 값 — 이 값이 와야 outstanding 을 해제한다
TERMINAL_RESULTS = frozenset({
    "ACCEPTED", "ALREADY_STOPPED", "INVALID_STATE", "SESSION_MISMATCH",
    "SEQ_CONFLICT", "STALE_SEQ", "TARGET_MISMATCH", "TARGET_NOT_LOADED",
    "NEW_ROUTE_REQUIRED", "STALE_ROUTE", "POSE_REQUIRED", "LOCKED_STATE",
    "PROTOCOL_ERROR", "HOLD", "REJECTED",
})
# 명령이 실행되지 않은 결과 (상위에서 복구 처리 필요)
NEGATIVE_RESULTS = TERMINAL_RESULTS - {"ACCEPTED", "ALREADY_STOPPED"}

# POSE_UPDATE 는 현재 펌웨어 수신 enum 에 없다 (implementation_status.md).
# 하드웨어팀이 구현하면 True 로 바꾼다.
POSE_UPDATE_ENABLED = False


def wire_car_id(car_id: int) -> str:
    """내부 정수 car_id 를 펌웨어가 비교하는 문자열로 변환한다 (1 → "CAR_01")."""
    return f"CAR_{int(car_id):02d}"


def parse_car_id(raw: Any) -> int:
    """wire 의 "CAR_01" 또는 정수를 내부 정수 car_id 로 되돌린다."""
    if isinstance(raw, int):
        return raw
    text = str(raw)
    return int(text[4:]) if text.startswith("CAR_") else int(text)


# ─── 명령 빌더 (PC → ESP32) ──────────────────────────────────────────────────

def _base(msg_type: str, car_id: int, session_id: str) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": msg_type,
        "car_id": wire_car_id(car_id),
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


def make_wait(car_id: int, session_id: str, seq: int, route_id: int = 0,
              waypoint_id: int = 0, reason: str = "REMOTE_WAIT") -> dict[str, Any]:
    """WAIT (§17.1). 펌웨어는 route_id/waypoint_id/reason 을 모두 필수로 읽는다.

    주행 중이 아니면 route_id/waypoint_id = 0 을 허용한다 (펌웨어 하한이 0).
    """
    if reason not in WAIT_REASONS:
        raise ValueError(f"unsupported WAIT reason: {reason}")
    msg = _base("WAIT", car_id, session_id)
    msg.update(seq=seq, route_id=int(route_id), waypoint_id=int(waypoint_id), reason=reason)
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


def make_direct_control(car_id: int, session_id: str, control_seq: int,
                        throttle: float, steering: float) -> dict[str, Any]:
    """REMOTE_DIRECT 개발용 (§18.2).

    throttle/steering 은 -1.0~1.0 정규화 값이며 ESP32 가 PWM/서보로 매핑한다.
    스트림 메시지이므로 신뢰성 seq 가 아닌 control_seq 를 쓴다.
    """
    msg = _base("DIRECT_CONTROL", car_id, session_id)
    msg.update(
        control_seq=int(control_seq),
        throttle=round(max(-1.0, min(1.0, throttle)), 4),
        steering=round(max(-1.0, min(1.0, steering)), 4),
    )
    return msg


def make_hello_ack(
    car_id: int, session_id: str, result: str, reason: str | None = None,
    boot_id: str = "", command_seq_start: int = 1,
) -> dict[str, Any]:
    """HELLO_ACK (§26.2). result ∈ READY_ALLOWED | HOLD | REJECTED.

    펌웨어 파싱 요구사항:
      - boot_id 는 필수이며 HELLO 의 값을 그대로 에코한다
      - REJECTED 가 아니면 session_id 와 command_seq_start 가 필수
      - 사유 키는 결과별로 다르다 (hold_reason / reject_reason)
    """
    if result not in ("READY_ALLOWED", "HOLD", "REJECTED"):
        raise ValueError(f"invalid HELLO_ACK result: {result}")
    msg = _base("HELLO_ACK", car_id, session_id)
    msg["boot_id"] = boot_id
    msg["result"] = result
    if result != "REJECTED":
        msg["command_seq_start"] = int(command_seq_start)
    if reason:
        msg["hold_reason" if result == "HOLD" else "reject_reason"] = reason
    return msg


def make_heartbeat(car_id: int, session_id: str, heartbeat_seq: int) -> dict[str, Any]:
    """HEARTBEAT (§12) — 노트북이 250ms 주기로 송신. 끊기면 차량이 COMM_TIMEOUT."""
    msg = _base("HEARTBEAT", car_id, session_id)
    msg["heartbeat_seq"] = int(heartbeat_seq)
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
    obj["car_id"] = parse_car_id(obj["car_id"])
    return obj
