"""Reward functions for parking spot assignment and routing policies."""

from __future__ import annotations


def assignment_reward(distance: float, is_type_match: bool, congestion: float = 0.0) -> float:
    """Score an assignment by distance, spot suitability, and congestion.

    Higher reward is better. The coefficients are intentionally simple for MVP
    explainability and can be tuned from RLPolicyLog records later.
    """
    match_bonus = 2.0 if is_type_match else -1.0
    distance_penalty = min(distance, 50.0) * 0.05
    congestion_penalty = max(0.0, congestion) * 0.5
    return match_bonus - distance_penalty - congestion_penalty


def route_reward(steps: int, reached_target: bool, collision_risk: float = 0.0) -> float:
    """Reward shorter, successful routes while penalizing unsafe trajectories."""
    completion_bonus = 10.0 if reached_target else -2.0
    return completion_bonus - (steps * 0.1) - (collision_risk * 5.0)
