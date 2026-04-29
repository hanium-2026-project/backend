"""Stable Baselines3 training hook for the parking environment."""

from __future__ import annotations


def train(total_timesteps: int = 10_000, output_path: str = "models/sb3_parking_policy.zip") -> str:
    """Train an SB3 policy once optional ML dependencies are installed.

    TODO: Wrap ParkingRoutingEnv with Gymnasium spaces and call PPO/A2C here.
    The MVP exposes the function so CI and callers have a stable integration
    point without pulling heavyweight torch dependencies into the base install.
    """
    raise NotImplementedError(
        "Stable Baselines3 training is an optional extension. Install stable-baselines3 and implement this hook."
    )
