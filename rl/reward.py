"""Reward functions for the parking slot allocation RL environment.

Composite reward formula
------------------------
    r = α · efficiency_reward
      + β · congestion_reward
      + γ · conflict_penalty

Default weights:  α = 1.0,  β = 0.5,  γ = 1.0

Design notes
------------
- conflict_penalty dominates all other terms (−10 vs max positive ≈ 1.5).
- efficiency_reward is positive and bounded [0, 1].
- congestion_reward is non-positive; it penalises routes that load bottlenecks.
- No imports from other rl modules — reward.py is a pure computation leaf.
"""

from __future__ import annotations

import math

# ─── Default Reward Weights ───────────────────────────────────────────────────
ALPHA: float = 1.0   # efficiency weight
BETA: float = 0.5    # congestion weight
GAMMA: float = 1.0   # conflict penalty weight

CONFLICT_PENALTY: float = -10.0


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_reward(
    *,
    slot_name: str,
    route: list[str],
    conflict: bool,
    slot_coordinates: dict[str, tuple[float, float]],
    entry_point: tuple[float, float],
    max_distance: float,
    bottleneck_nodes: list[str],
    reservations: dict[str, list[tuple[float, float]]],
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Compute the composite reward for one slot assignment decision.

    Parameters
    ----------
    slot_name        : Name of the chosen slot (e.g. "A1").
    route            : Ordered waypoint list for the chosen slot.
    conflict         : True if the Safety Shield detected a conflict.
    slot_coordinates : Dict mapping slot name → (x_mm, y_mm).
    entry_point      : (x_mm, y_mm) of the parking lot entrance.
    max_distance     : Lot diagonal in mm (used for normalisation).
    bottleneck_nodes : List of high-traffic waypoint names.
    reservations     : Current reservation table {node: [(t_start, t_end), ...]}.
    alpha / beta / gamma : Override default weights (useful for ablation).

    Returns
    -------
    float — scalar reward signal for this step.
    """
    if conflict:
        return gamma * CONFLICT_PENALTY

    eff = _efficiency_reward(slot_name, slot_coordinates, entry_point, max_distance)
    cong = _congestion_reward(route, bottleneck_nodes, reservations)

    return alpha * eff + beta * cong


# ─── Component Functions ──────────────────────────────────────────────────────

def _efficiency_reward(
    slot_name: str,
    slot_coordinates: dict[str, tuple[float, float]],
    entry_point: tuple[float, float],
    max_distance: float,
) -> float:
    """Normalised proximity reward.

    Returns 1.0 for the slot closest to the entrance, 0.0 for the farthest.
    """
    sx, sy = slot_coordinates[slot_name]
    ex, ey = entry_point
    dist = math.hypot(sx - ex, sy - ey)
    return 1.0 - min(dist / max_distance, 1.0)


def _congestion_reward(
    route: list[str],
    bottleneck_nodes: list[str],
    reservations: dict[str, list[tuple[float, float]]],
) -> float:
    """Non-positive penalty for loading shared bottleneck nodes.

    Each bottleneck node on the route contributes its current reservation count
    as a penalty.  The total is normalised by the worst-case load
    (all bottleneck nodes each carrying 5 reservations) and clamped to [−1, 0].
    """
    if not bottleneck_nodes:
        return 0.0

    total_load = sum(
        len(reservations.get(node, []))
        for node in route
        if node in bottleneck_nodes
    )

    worst_case = float(len(bottleneck_nodes) * 5)
    return -min(total_load / worst_case, 1.0)
