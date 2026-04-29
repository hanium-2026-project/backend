"""Ray RLlib training hook for future centralized training experiments."""

from __future__ import annotations


def train(config: dict | None = None, output_dir: str = "models/rllib") -> str:
    """Train an RLlib policy for parking assignment or routing.

    TODO: Connect RLlib multi-agent configs for CTDE/MARL experiments after the
    simulator and multi-camera state representation are finalized.
    """
    raise NotImplementedError("Ray RLlib training is an optional extension and is not required for the MVP.")
