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

def _valid_slots(action_masks: np.ndarray, n_slots: int) -> list[int]:
    """Indices of slot actions (0..n_slots-1) currently allowed by the mask."""
    return [i for i in range(n_slots) if i < len(action_masks) and action_masks[i]]


def _wait_or_dead(action_masks: np.ndarray, wait_action: int) -> int:
    """Return WAIT if it is still valid; otherwise -1 (full deadlock)."""
    if wait_action < len(action_masks) and action_masks[wait_action]:
        return wait_action
    return -1


# ─── V1: Nearest-Greedy Baseline ──────────────────────────────────────────────

def heuristic_policy(action_masks: np.ndarray) -> int:
    """V1 baseline: greedy nearest slot (Euclidean from ENTRY_POINT).

    Uses only `action_masks` (mask filters out conflicting/taken slots).
    Never voluntarily WAITs — only chooses WAIT when all slots are masked.
    Kept under the original name for backwards compatibility.
    """
    from .parking_env import ENTRY_POINT, SLOT_COORDINATES, SLOT_NAMES, WAIT_ACTION

    valid = _valid_slots(action_masks, len(SLOT_NAMES))
    if not valid:
        return _wait_or_dead(action_masks, WAIT_ACTION)

    ex, ey = ENTRY_POINT
    return min(valid, key=lambda i: math.hypot(
        SLOT_COORDINATES[SLOT_NAMES[i]][0] - ex,
        SLOT_COORDINATES[SLOT_NAMES[i]][1] - ey,
    ))


# ─── V2 / V3 / V4: Stronger Rule-Based Baselines ─────────────────────────────
#
# These take the live ParkingRoutingEnv so they can read the reservation table
# (something V1 cannot do).  They model what a well-designed dispatching rule
# would look like and form a fairer benchmark for PPO.

def _bottleneck_load(env: Any, route: list[str]) -> int:
    """Number of reservations sitting on bottleneck nodes along *route*."""
    from .parking_env import BOTTLENECK_NODES
    return sum(
        len(env._reservations.get(n, []))
        for n in route if n in BOTTLENECK_NODES
    )


def heuristic_policy_v2(action_masks: np.ndarray, env: Any) -> int:
    """V2 — Congestion-Aware Nearest.

    score(slot) = (dist / MAX_DISTANCE) + 0.5 · (bottleneck_load / 10)
    Picks the slot with the lowest score.  Same WAIT policy as V1.
    """
    from .parking_env import (
        ENTRY_POINT, MAX_DISTANCE, SLOT_COORDINATES, SLOT_NAMES,
        SLOT_ROUTES, WAIT_ACTION,
    )

    valid = _valid_slots(action_masks, len(SLOT_NAMES))
    if not valid:
        return _wait_or_dead(action_masks, WAIT_ACTION)

    ex, ey = ENTRY_POINT
    def score(i: int) -> float:
        name = SLOT_NAMES[i]
        sx, sy = SLOT_COORDINATES[name]
        dist_norm = math.hypot(sx - ex, sy - ey) / MAX_DISTANCE
        cong_norm = min(_bottleneck_load(env, SLOT_ROUTES[name]) / 10.0, 1.0)
        return dist_norm + 0.5 * cong_norm
    return min(valid, key=score)


def heuristic_policy_v3(action_masks: np.ndarray, env: Any) -> int:
    """V3 — Proactive WAIT.

    Same scoring as V2, but voluntarily WAITs when the central lane is
    critically loaded (junction + lane_pt_4 reservations ≥ 4) AND the agent
    has not just chained 3+ consecutive WAITs.
    """
    from .parking_env import WAIT_ACTION

    # Proactive WAIT trigger
    junction_load = len(env._reservations.get("junction",  []))
    lane4_load    = len(env._reservations.get("lane_pt_4", []))
    can_wait = (
        WAIT_ACTION < len(action_masks)
        and action_masks[WAIT_ACTION]
        and env._consecutive_waits < 3
    )
    if can_wait and (junction_load + lane4_load) >= 4:
        return WAIT_ACTION

    return heuristic_policy_v2(action_masks, env)


def heuristic_policy_v4(action_masks: np.ndarray, env: Any) -> int:
    """V4 — Route-Length-Aware.

    Picks slots whose route is short (fewer nodes ⇒ shorter lane occupancy)
    AND whose nodes are not heavily reserved already.

    score(slot) = (route_len / 7) + 0.3 · (total_route_load / 30)
    """
    from .parking_env import SLOT_NAMES, SLOT_ROUTES, WAIT_ACTION

    valid = _valid_slots(action_masks, len(SLOT_NAMES))
    if not valid:
        return _wait_or_dead(action_masks, WAIT_ACTION)

    def score(i: int) -> float:
        route = SLOT_ROUTES[SLOT_NAMES[i]]
        len_score  = len(route) / 7.0
        load_total = sum(len(env._reservations.get(n, [])) for n in route)
        cong_score = min(load_total / 30.0, 1.0)
        return len_score + 0.3 * cong_score
    return min(valid, key=score)
