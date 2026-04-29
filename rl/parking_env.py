"""Gymnasium-style environment for parking assignment experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reward import assignment_reward


@dataclass(frozen=True)
class ParkingState:
    """Compact state used by the MVP environment and mock inference policy."""

    vehicle_type: int
    spot_statuses: np.ndarray
    spot_types: np.ndarray
    spot_coordinates: np.ndarray


class ParkingRoutingEnv:
    """Small Gymnasium-style environment without requiring Gymnasium at import.

    Action is the index of a target spot. The observation is a dict-like state
    that can later be wrapped with Gymnasium spaces for SB3/RLlib training.
    """

    metadata = {"render_modes": []}

    def __init__(self, spot_types: list[int] | None = None, coordinates: list[tuple[float, float]] | None = None) -> None:
        self.spot_types = np.asarray(spot_types or [0, 0, 1, 2], dtype=np.int64)
        self.coordinates = np.asarray(coordinates or [(1, 1), (2, 1), (3, 1), (4, 1)], dtype=np.float32)
        self.vehicle_type = 0
        self.spot_statuses = np.zeros(len(self.spot_types), dtype=np.int64)
        self._terminated = False

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset all spots to vacant and return the initial observation."""
        if seed is not None:
            np.random.seed(seed)
        self.vehicle_type = int((options or {}).get("vehicle_type", 0))
        self.spot_statuses = np.zeros(len(self.spot_types), dtype=np.int64)
        self._terminated = False
        return self._observation(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Occupy the selected spot and return reward plus termination flags."""
        if self._terminated:
            raise RuntimeError("Environment is terminated. Call reset before step.")
        if action < 0 or action >= len(self.spot_statuses):
            raise ValueError("Action must be a valid spot index.")
        if self.spot_statuses[action] == 1:
            reward = -5.0
            self._terminated = True
            return self._observation(), reward, True, False, {"reason": "occupied"}

        self.spot_statuses[action] = 1
        distance = float(np.linalg.norm(self.coordinates[action]))
        reward = assignment_reward(distance=distance, is_type_match=int(self.spot_types[action]) == self.vehicle_type)
        self._terminated = True
        return self._observation(), reward, True, False, {"assigned_index": action}

    def _observation(self) -> dict:
        """Return a serializable observation consumed by heuristic policies."""
        return {
            "vehicle_type": self.vehicle_type,
            "spot_statuses": self.spot_statuses.copy(),
            "spot_types": self.spot_types.copy(),
            "spot_coordinates": self.coordinates.copy(),
        }
