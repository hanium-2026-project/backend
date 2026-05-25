"""Inference utilities for the parking slot allocation RL environment.

Public API
----------
    select_action(observation, action_masks, model_path) → int
        Unified entry point: uses a trained MaskablePPO model when available,
        falls back to the distance-based heuristic otherwise.

    load_policy(model_path) → model | None
        Load a MaskablePPO model from disk; return None if file is absent.

    heuristic_policy(action_masks) → int
        Greedy nearest-slot policy (no trained model required).
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

# Module-level cache — avoids reloading the model on every inference call
_loaded_model: Any | None = None
_model_path_cache: str = ""


# ─── Unified Entry Point ──────────────────────────────────────────────────────

def select_action(
    observation: np.ndarray,
    action_masks: np.ndarray,
    model_path: str = "models/sb3_parking_policy.zip",
) -> int:
    """Select a slot index using trained policy or heuristic fallback.

    Parameters
    ----------
    observation  : 20-dim float32 state vector from ParkingRoutingEnv._get_obs().
    action_masks : Boolean array (shape=(8,)) from ParkingRoutingEnv.action_masks().
    model_path   : Path to a saved MaskablePPO .zip file.

    Returns
    -------
    int — Slot index in [0, 7].  Returns -1 if no slot is available.
    """
    policy = load_policy(model_path)

    if policy is not None:
        # SB3 predict expects a batch dimension
        obs_batch = observation.reshape(1, -1)
        action, _ = policy.predict(
            obs_batch,
            action_masks=action_masks,
            deterministic=True,
        )
        return int(action)

    return heuristic_policy(action_masks)


# ─── Policy Loader ────────────────────────────────────────────────────────────

def load_policy(model_path: str = "models/sb3_parking_policy.zip") -> Any | None:
    """Load a trained MaskablePPO model from disk.

    Returns None (instead of raising) when the file does not exist so that
    callers can transparently fall back to the heuristic.

    The loaded model is cached in a module-level variable; subsequent calls
    with the same path skip the disk read.

    Raises
    ------
    ImportError if sb3-contrib is not installed and a model file is present.
    """
    global _loaded_model, _model_path_cache

    if not os.path.exists(model_path):
        return None

    # Return cached model if path has not changed
    if _loaded_model is not None and model_path == _model_path_cache:
        return _loaded_model

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise ImportError(
            "sb3-contrib is required to load a trained policy. "
            "Install with: pip install stable-baselines3 sb3-contrib"
        ) from exc

    _loaded_model = MaskablePPO.load(model_path)
    _model_path_cache = model_path
    return _loaded_model


# ─── Heuristic Fallback ───────────────────────────────────────────────────────

def heuristic_policy(action_masks: np.ndarray) -> int:
    """Distance-based greedy policy: select the nearest available slot.

    Uses Euclidean distance from ENTRY_POINT to each slot's mm coordinate.
    No trained model is required.

    Parameters
    ----------
    action_masks : Boolean array (shape=(8,)) from ParkingRoutingEnv.action_masks().

    Returns
    -------
    int — Index of the best available slot, or -1 if all slots are masked.
    """
    from .parking_env import ENTRY_POINT, SLOT_COORDINATES, SLOT_NAMES

    valid_indices = [i for i, ok in enumerate(action_masks) if ok]
    if not valid_indices:
        return -1  # No slot available

    ex, ey = ENTRY_POINT
    return min(
        valid_indices,
        key=lambda i: math.hypot(
            SLOT_COORDINATES[SLOT_NAMES[i]][0] - ex,
            SLOT_COORDINATES[SLOT_NAMES[i]][1] - ey,
        ),
    )
