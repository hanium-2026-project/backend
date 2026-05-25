"""MaskablePPO training script for the parking slot allocation environment.

Requirements
------------
    pip install stable-baselines3 sb3-contrib gymnasium

Usage (CLI)
-----------
    python -m rl.train_sb3

Usage (Python)
--------------
    from rl.train_sb3 import train
    saved_path = train(total_timesteps=100_000)
"""

from __future__ import annotations

import os


def train(
    total_timesteps: int = 100_000,
    output_path: str = "models/sb3_parking_policy.zip",
) -> str:
    """Train a MaskablePPO agent on ParkingRoutingEnv and save the model.

    Parameters
    ----------
    total_timesteps : Total environment steps for training.
    output_path     : Destination path for the saved .zip model.

    Returns
    -------
    str — Absolute path to the saved model file.

    Raises
    ------
    ImportError if sb3-contrib or stable-baselines3 are not installed.
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError as exc:
        raise ImportError(
            "sb3-contrib is required for training. "
            "Install with: pip install stable-baselines3 sb3-contrib"
        ) from exc

    from .parking_env import ParkingRoutingEnv

    def _mask_fn(env: ParkingRoutingEnv):
        """Callback used by ActionMasker to retrieve valid action mask."""
        return env.action_masks()

    # Wrap environment so MaskablePPO can query action masks each step
    env = ActionMasker(ParkingRoutingEnv(), _mask_fn)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=total_timesteps)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model.save(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"[train_sb3] Model saved → {abs_path}")
    return abs_path


if __name__ == "__main__":
    train()
