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

efficiency_reward (v2, this branch)
------------------------------------
The efficiency signal now combines two components:

  1. Distance score  (weight 0.6)
     1 - dist(slot, ENTRY_POINT) / MAX_DISTANCE
     A4 = B4 ≈ 0.69,  A1 = B1 ≈ 0.38

  2. Route-length score  (weight 0.4)
     1 - (route_len - MIN_ROUTE) / (MAX_ROUTE - MIN_ROUTE)
     Shorter routes occupy the shared central lane for less time,
     directly reducing conflict probability for subsequent vehicles.
     A4 / B4 (4 nodes) → 1.0,  A1 / B1 (7 nodes) → 0.0

     MIN_ROUTE = 4,  MAX_ROUTE = 7  (L-shaped single-lane layout)

Combined:  eff = 0.6 · dist_score + 0.4 · len_score
  A4 / B4: 0.6·0.69 + 0.4·1.0 = 0.81   ← highest
  A3 / B3: 0.6·0.60 + 0.4·0.67 = 0.63
  A2 / B2: 0.6·0.50 + 0.4·0.33 = 0.43
  A1 / B1: 0.6·0.38 + 0.4·0.0  = 0.23  ← lowest

Rationale: In the L-shaped lot, longer routes mean the vehicle occupies
junction / lane_pt_4 / lane_pt_3 longer, blocking entering and exiting
traffic.  The route-length component makes this cost explicit in the
reward signal so PPO can learn to prefer shorter routes not only for
distance efficiency but also for lane contention reduction.
"""

from __future__ import annotations

import math

# ─── Default Reward Weights ───────────────────────────────────────────────────
ALPHA: float = 1.0   # efficiency weight
BETA: float  = 0.5   # congestion weight
GAMMA: float = 1.0   # conflict penalty weight

CONFLICT_PENALTY: float = -10.0

# ─── Efficiency Sub-weights (route-length v2) ─────────────────────────────────
# The efficiency signal is a weighted sum of two normalised scores:
#   DIST_WEIGHT  : proximity from junction (shorter Euclidean distance → better)
#   LEN_WEIGHT   : inverse route length   (fewer nodes → less lane occupancy)
# Must sum to 1.0.
DIST_WEIGHT: float = 0.6
LEN_WEIGHT:  float = 0.4
MIN_ROUTE_LEN: int = 4   # A4 / B4  (closest slots in L-shaped layout)
MAX_ROUTE_LEN: int = 7   # A1 / B1  (farthest slots)


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
    beta: float  = BETA,
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

    eff  = _efficiency_reward(slot_name, slot_coordinates, entry_point,
                              max_distance, route=route)
    cong = _congestion_reward(route, bottleneck_nodes, reservations)

    return alpha * eff + beta * cong


# ─── Component Functions ──────────────────────────────────────────────────────

def _efficiency_reward(
    slot_name: str,
    slot_coordinates: dict[str, tuple[float, float]],
    entry_point: tuple[float, float],
    max_distance: float,
    route: list[str] | None = None,
) -> float:
    """Normalised efficiency reward combining distance and route length.

    Components
    ----------
    dist_score : 1 - dist(slot, entry_point) / MAX_DISTANCE
                 Rewards proximity; A4=B4 highest, A1=B1 lowest.

    len_score  : 1 - (len(route) - MIN_ROUTE_LEN) / (MAX_ROUTE_LEN - MIN_ROUTE_LEN)
                 Rewards shorter routes; A4=B4 → 1.0, A1=B1 → 0.0.
                 Shorter route = fewer nodes on shared central lane
                 = less blocking time for other vehicles.
                 Falls back to 0.0 (neutral) when route is not supplied.

    Combined   : DIST_WEIGHT * dist_score + LEN_WEIGHT * len_score
                 A4/B4 ≈ 0.81,  A3/B3 ≈ 0.63,  A2/B2 ≈ 0.43,  A1/B1 ≈ 0.23
    """
    # Distance component
    sx, sy = slot_coordinates[slot_name]
    ex, ey = entry_point
    dist       = math.hypot(sx - ex, sy - ey)
    dist_score = 1.0 - min(dist / max_distance, 1.0)

    # Route-length component
    if route is not None:
        route_len  = len(route)
        span       = MAX_ROUTE_LEN - MIN_ROUTE_LEN          # 3
        len_score  = 1.0 - min(
            (route_len - MIN_ROUTE_LEN) / span, 1.0
        )
    else:
        len_score = 0.0   # fallback: route not provided → neutral

    return DIST_WEIGHT * dist_score + LEN_WEIGHT * len_score


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
