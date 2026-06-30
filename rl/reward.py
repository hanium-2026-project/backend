"""Reward functions for the parking slot allocation RL environment.

Goal-aligned composite reward (v2)
------------------------------------
Core goal: maximise parking lot FLOW efficiency.
  1. Bottleneck complexity reduction — keep shared central lane clear
  2. Optimal slot assignment          — near slots first, spread load
  3. Efficient parking management     — maximise vehicles processed per episode

Formula
-------
    r = α · R_efficiency          slot proximity reward
      + β · R_congestion          bottleneck load penalty
      + δ · R_flow                throughput / flow contribution  ← NEW
      + γ · R_conflict            safety shield penalty

Changes from v1
---------------
v1 problem: CONFLICT_PENALTY = -10 dominated the signal, turning PPO into a
"conflict avoidance specialist" that over-WAITs and sacrifices throughput.
Throughput is the primary goal yet had zero reward signal.

v2 changes:
  1. CONFLICT_PENALTY: -10 → -5
     Conflict is a *means* (it hurts flow), not the end goal.
     Halving the penalty lets throughput signals compete, so PPO does
     not sacrifice 9 % throughput just to shave 5 % off conflict rate.

  2. R_flow added (δ = 0.8):
     Rewards two flow-positive events on a successful assignment:
       a. slot_reuse_bonus (+SLOT_REUSE_BONUS)  if this slot was already
          used once this episode  → departure happened → slot is cycling.
       b. free_slot_bonus  (+FREE_SLOT_BONUS)   proportional to how many
          slots are still free  → rewards spreading load early rather
          than packing into the same popular slot.
     Both are normalised to [0, 1] and weighted by δ = 0.8.

  3. R_efficiency unchanged (distance-based, α = 1.0).
  4. R_congestion unchanged (bottleneck load, β = 0.5).

Reward scale reference (no conflict, all clear)
------------------------------------------------
  Best case  (A4, empty lot, reuse):  1.0 + 0 + 0.8·1.0 = 1.8
  Worst case (A1, busy lot, no reuse): 0.38 + (-0.5) + 0 = -0.12
  Conflict:                           -5.0

The new floor is -5 (was -10), narrowing the range so positive flow
signals are no longer drowned by rare large conflict penalties.
"""

from __future__ import annotations

import math

# ─── Default Reward Weights ───────────────────────────────────────────────────
ALPHA: float = 1.0   # efficiency weight
BETA:  float = 0.5   # congestion weight
DELTA: float = 0.8   # flow reward weight  ← NEW
GAMMA: float = 1.0   # conflict penalty weight

# ─── Penalty / Bonus Constants ────────────────────────────────────────────────
# Conflict penalty halved (v1 = -10).  Conflict is a flow impediment, not the
# primary optimisation target; reducing its dominance lets throughput signals
# influence PPO's policy.
CONFLICT_PENALTY: float = -5.0

# Slot-reuse bonus: awarded when the chosen slot has been used at least once
# before in the current episode, proving a departure happened and the lot is
# cycling.  Value chosen so one reuse ≈ half the best-case efficiency reward.
SLOT_REUSE_BONUS: float = 0.5

# Free-slot bonus scale: awarded proportionally to the fraction of currently
# free slots (rewards spreading assignments rather than clustering on A4/B4).
# At 8 free slots → +0.3, at 4 free → +0.15, at 0 free → +0.0.
FREE_SLOT_BONUS_MAX: float = 0.3


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
    # flow context (new in v2)
    slot_use_count: int  = 0,   # how many times this slot was used this ep
    free_slot_count: int = 0,   # number of currently free slots
    num_slots: int       = 8,   # total slot count (for normalisation)
    alpha: float = ALPHA,
    beta:  float = BETA,
    delta: float = DELTA,
    gamma: float = GAMMA,
) -> float:
    """Compute the goal-aligned composite reward for one assignment decision.

    Parameters
    ----------
    slot_name        : Name of the chosen slot (e.g. "A4").
    route            : Ordered waypoint list for the slot.
    conflict         : True if the Safety Shield detected a conflict.
    slot_coordinates : Dict mapping slot name → (x_mm, y_mm).
    entry_point      : (x_mm, y_mm) of the parking lot entrance (junction).
    max_distance     : Lot diagonal in mm (for distance normalisation).
    bottleneck_nodes : High-traffic waypoint names.
    reservations     : Current reservation table {node: [...]}.
    slot_use_count   : Times this slot was assigned earlier this episode.
                       ≥ 1 → a departure happened → reuse bonus applies.
    free_slot_count  : Free slots at assignment time (for spread bonus).
    num_slots        : Total slots (default 8).
    alpha/beta/delta/gamma : Weight overrides.

    Returns
    -------
    float — scalar reward signal for this step.
    """
    if conflict:
        return gamma * CONFLICT_PENALTY

    eff  = _efficiency_reward(slot_name, slot_coordinates,
                              entry_point, max_distance)
    cong = _congestion_reward(route, bottleneck_nodes, reservations)
    flow = _flow_reward(slot_use_count, free_slot_count, num_slots)

    return alpha * eff + beta * cong + delta * flow


# ─── Component Functions ──────────────────────────────────────────────────────

def _efficiency_reward(
    slot_name: str,
    slot_coordinates: dict[str, tuple[float, float]],
    entry_point: tuple[float, float],
    max_distance: float,
) -> float:
    """Normalised proximity reward  [0, 1].

    Returns 1.0 for the slot closest to the junction, 0.0 for the farthest.
    A4 = B4 ≈ 0.69, A1 = B1 ≈ 0.38.
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
    """Non-positive bottleneck load penalty  [−1, 0].

    Each bottleneck node on the route contributes its reservation count.
    Normalised by worst-case (all bottlenecks × 5 reservations each).
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


def _flow_reward(
    slot_use_count: int,
    free_slot_count: int,
    num_slots: int,
) -> float:
    """Flow contribution reward  [0, ~1].

    Two sub-components:

    1. Slot-reuse bonus (SLOT_REUSE_BONUS = 0.5)
       Fires when slot_use_count >= 1, meaning this slot was previously
       assigned and then freed by a departure.  Rewards the agent for
       enabling departure cycles rather than just filling slots.

    2. Free-slot spread bonus (FREE_SLOT_BONUS_MAX = 0.3)
       Proportional to the fraction of free slots remaining.
       Encourages early assignment of less-popular slots so the lot
       does not get stuck with only near-entrance slots free.
       At max occupancy (0 free) → 0; at empty lot → 0.3.

    Combined maximum ≈ 0.8 (reuse + full spread bonus).
    """
    reuse_bonus = SLOT_REUSE_BONUS if slot_use_count >= 1 else 0.0
    free_frac   = free_slot_count / max(num_slots, 1)
    spread_bonus = FREE_SLOT_BONUS_MAX * free_frac
    return reuse_bonus + spread_bonus
