"""Inference entrypoints for heuristic and future trained parking policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def heuristic_policy(observation: dict[str, Any]) -> int:
    """Select the closest vacant spot with a vehicle/spot type preference."""
    statuses = np.asarray(observation["spot_statuses"])
    spot_types = np.asarray(observation["spot_types"])
    coordinates = np.asarray(observation["spot_coordinates"], dtype=float)
    vehicle_type = int(observation.get("vehicle_type", 0))
    vacant_indices = np.where(statuses == 0)[0]
    if len(vacant_indices) == 0:
        raise ValueError("No vacant spots are available for inference.")

    def score(index: int) -> tuple[int, float, int]:
        type_penalty = 0 if int(spot_types[index]) == vehicle_type else 1
        distance = float(np.linalg.norm(coordinates[index]))
        return (type_penalty, distance, int(index))

    return int(sorted(vacant_indices, key=score)[0])


def load_policy(model_path: str | Path | None = None):
    """Return a callable policy; fall back to heuristic when no model is present.

    TODO: Dispatch to Stable Baselines3 or RLlib policy loading when a trained
    model artifact is supplied. Keeping this callable contract makes the REST
    recommendation service easy to upgrade later.
    """
    if not model_path:
        return heuristic_policy
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy model not found: {path}")
    raise NotImplementedError("Trained policy loading is not implemented in the MVP.")
