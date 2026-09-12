"""노트북 ↔ ESP32 차량 제어 통신 계층 (v2 — 시스템 통합 문서 기준).

노트북 = TCP Server, ESP32 = Client. NDJSON, 신뢰성 명령 멱등/재전송,
WAIT/STOP 선점, POSE 스트림 최신값, 노트북 도착 판정 오케스트레이션.
"""

from .protocol import PROTOCOL_VERSION, TIMING, encode, parse_message
from .reliability import ReliableSender
from .server import VehicleServer, VehicleSession
from .orchestrator import Mission, MissionOrchestrator, MissionState

__all__ = [
    "PROTOCOL_VERSION",
    "TIMING",
    "encode",
    "parse_message",
    "ReliableSender",
    "VehicleServer",
    "VehicleSession",
    "Mission",
    "MissionOrchestrator",
    "MissionState",
]
