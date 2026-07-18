"""CV → RL 실시간 연동 브리지.

카메라 파이프라인(cv.tracker)이 산출한 실좌표(mm)를 RL 관측 벡터로 변환하고,
학습된 정책(rl.inference.select_action)으로 슬롯 할당을 수행한다.

데이터 흐름::

    TrackState (bbox, track_id)
      └─ homography → (x, y) mm
           └─ position_to_node()      : 최근접 그래프 노드 스냅
                └─ RealtimeAllocator  : 21-dim obs + action mask 구성
                     └─ select_action : 슬롯 인덱스 (0-7) / WAIT(8) / -1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .inference import select_action
from .parking_env import (
    BOTTLENECK_NODES,
    MAX_ACTIVE_VEHICLES,
    NODE_COORDINATES,
    NODE_INDEX,
    NODE_TRAVEL_TIME,
    NUM_SLOTS,
    SLOT_NAMES,
    SLOT_ROUTES,
    STATE_DIM,
    WAIT_ACTION,
    _ALL_NODES,
    _MAX_ROUTE_LEN,
)

# 좌표가 어느 노드에도 충분히 가깝지 않으면 노드 스냅을 보류하는 임계값 (mm)
NODE_SNAP_RADIUS_MM: float = 200.0


def position_to_node(pos: tuple[float, float]) -> str | None:
    """실좌표(mm)를 최근접 그래프 노드 이름으로 스냅한다.

    NODE_SNAP_RADIUS_MM 밖이면 None (차로 사이 주행 중 등).
    """
    x, y = pos
    best_node, best_dist = None, float("inf")
    for name, (nx, ny) in NODE_COORDINATES.items():
        d = math.hypot(nx - x, ny - y)
        if d < best_dist:
            best_node, best_dist = name, d
    return best_node if best_dist <= NODE_SNAP_RADIUS_MM else None


@dataclass
class TrackedVehicle:
    """브리지가 유지하는 추적 차량의 최소 상태."""

    track_id: int
    position: tuple[float, float]          # mm
    current_node: str | None = None
    assigned_slot: str | None = None       # 슬롯 이름 (예: "A4")
    route: list[str] = field(default_factory=list)


class RealtimeAllocator:
    """실시간 슬롯 할당기.

    - slot_statuses: 0.0=빈 슬롯, 1.0=점유 (ParkingSpot DB와 동기화 예정)
    - update(): CV 위치 갱신
    - allocate(): 미할당 차량에 대해 RL 정책으로 슬롯 결정
    """

    def __init__(self, model_path: str = "models/sb3_parking_policy.zip") -> None:
        self.model_path = model_path
        self.slot_statuses = np.zeros(NUM_SLOTS, dtype=np.float32)
        self.vehicles: dict[int, TrackedVehicle] = {}

    # ─── 상태 갱신 ────────────────────────────────────────────────────────────

    def update(self, track_id: int, position_mm: tuple[float, float]) -> TrackedVehicle:
        """CV 프레임 1건 반영: 차량 위치·노드 갱신 (신규 track_id면 등록)."""
        v = self.vehicles.get(track_id)
        if v is None:
            v = TrackedVehicle(track_id=track_id, position=position_mm)
            self.vehicles[track_id] = v
        v.position = position_mm
        node = position_to_node(position_mm)
        if node is not None:
            v.current_node = node
        return v

    def set_slot_occupied(self, slot_name: str, occupied: bool = True) -> None:
        self.slot_statuses[SLOT_NAMES.index(slot_name)] = 1.0 if occupied else 0.0

    def remove_vehicle(self, track_id: int) -> None:
        self.vehicles.pop(track_id, None)

    # ─── RL 연동 ─────────────────────────────────────────────────────────────

    def allocate(self, track_id: int) -> str | None:
        """차량에 슬롯을 할당하고 경로를 부여한다.

        Returns 슬롯 이름 ("A1"~"B4") / None (WAIT 또는 할당 불가).
        """
        v = self.vehicles.get(track_id)
        if v is None:
            raise KeyError(f"unknown track_id: {track_id}")
        if v.assigned_slot is not None:
            return v.assigned_slot

        obs = self._build_obs()
        masks = self._build_masks()
        action = select_action(obs, masks, model_path=self.model_path)

        if action < 0 or action >= NUM_SLOTS:      # -1(불가) 또는 WAIT(8)
            return None

        slot = SLOT_NAMES[action]
        v.assigned_slot = slot
        v.route = list(SLOT_ROUTES[slot])
        self.set_slot_occupied(slot)               # 이중 할당 방지 선점
        return slot

    # ─── 관측/마스크 빌더 (parking_env._get_obs 스키마와 동일) ────────────────

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(STATE_DIM, dtype=np.float32)
        obs[:NUM_SLOTS] = self.slot_statuses

        num_nodes = max(len(_ALL_NODES), 1)
        max_eta = _MAX_ROUTE_LEN * NODE_TRAVEL_TIME
        moving = [v for v in self.vehicles.values() if v.assigned_slot and v.route]
        for i, v in enumerate(moving[:MAX_ACTIVE_VEHICLES]):
            base = NUM_SLOTS + i * 3
            cur = NODE_INDEX.get(v.current_node or "", 0)
            # 경로상 현재 노드의 다음 노드를 next로 사용 (없으면 현재 유지)
            nxt = cur
            if v.current_node in v.route:
                idx = v.route.index(v.current_node)
                if idx + 1 < len(v.route):
                    nxt = NODE_INDEX.get(v.route[idx + 1], cur)
            remaining = len(v.route) - (v.route.index(v.current_node) + 1) if v.current_node in v.route else len(v.route)
            obs[base] = cur / num_nodes
            obs[base + 1] = nxt / num_nodes
            obs[base + 2] = float(np.clip(remaining * NODE_TRAVEL_TIME / max_eta, 0.0, 1.0))

        base = NUM_SLOTS + MAX_ACTIVE_VEHICLES * 3
        for j, node in enumerate(BOTTLENECK_NODES):
            count = sum(1 for v in moving if v.current_node == node)
            obs[base + j] = min(count / max(MAX_ACTIVE_VEHICLES, 1), 1.0)

        # 실환경에서는 도착 간격 통계가 없으므로 중립값 0
        obs[STATE_DIM - 1] = 0.0
        return obs

    def _build_masks(self) -> np.ndarray:
        masks = np.zeros(NUM_SLOTS + 1, dtype=bool)
        masks[:NUM_SLOTS] = self.slot_statuses < 0.5   # 빈 슬롯만 선택 가능
        masks[WAIT_ACTION] = True
        return masks
