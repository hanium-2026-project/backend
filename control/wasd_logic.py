"""WASD key-state logic reused from the verified REMOTE_DIRECT bridge.

Sign convention:
- This module keeps the old bridge wire-style steering:
  A/LEFT = negative, D/RIGHT = positive.
- HybridControlMux converts it to HostController logical steering
  (+ = LEFT) before creating ManualInput.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet

STEERING_RAMP_STEP = 0.10
STEERING_RAMP_FAST_STEP = 0.20
STEERING_RAMP_INTERVAL_MS = 100


@dataclass(frozen=True)
class DriveIntent:
    throttle: float
    steering: float
    label: str


def _keys(pressed: AbstractSet[str]) -> set[str]:
    return {key.lower() for key in pressed}


def compute_throttle(pressed: AbstractSet[str]) -> float:
    keys = _keys(pressed)
    forward = "w" in keys
    reverse = "s" in keys
    if forward == reverse:
        return 0.0
    return 1.0 if forward else -1.0


def steering_direction(pressed: AbstractSet[str]) -> int:
    keys = _keys(pressed)
    left = "a" in keys
    right = "d" in keys
    if left == right:
        return 0
    return -1 if left else 1


def advance_steering(current: float, pressed: AbstractSet[str]) -> float:
    keys = _keys(pressed)
    direction = steering_direction(keys)
    if direction == 0:
        return 0.0

    strong = bool({"shift_l", "shift_r", "shift"} & keys)
    step = STEERING_RAMP_FAST_STEP if strong else STEERING_RAMP_STEP
    value = max(-1.0, min(1.0, float(current)))

    if value * direction < 0.0:
        value = 0.0

    value += direction * step
    return round(max(-1.0, min(1.0, value)), 3)


def compute_drive_intent(
    pressed: AbstractSet[str],
    steering: float | None = None,
) -> DriveIntent:
    keys = _keys(pressed)
    throttle = compute_throttle(keys)

    if steering is None:
        direction = steering_direction(keys)
        strong = bool({"shift_l", "shift_r", "shift"} & keys)
        steering = direction * (1.0 if strong else 0.5)

    steering = max(-1.0, min(1.0, float(steering)))
    direction_label = (
        "STOP" if throttle == 0.0 else ("FORWARD" if throttle > 0 else "REVERSE")
    )
    if abs(steering) < 1e-9:
        turn_label = "CENTER"
    else:
        side = "LEFT" if steering < 0.0 else "RIGHT"
        turn_label = f"{side} {round(abs(steering) * 100):d}%"

    return DriveIntent(
        throttle=throttle,
        steering=steering,
        label=f"{direction_label} / {turn_label}",
    )
