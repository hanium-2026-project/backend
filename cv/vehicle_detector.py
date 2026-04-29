"""Vehicle detector interfaces and mock/YOLO-compatible implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Normalized object detection result in image coordinates."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class VehicleDetector(Protocol):
    """Detector interface shared by mock and future YOLO26 adapters."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return vehicle detections for a BGR/RGB image array."""


class MockVehicleDetector:
    """Deterministic detector used for tests and demos without model weights."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect a synthetic vehicle near the center of any non-empty image."""
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return []
        x1 = max(0, width // 4)
        y1 = max(0, height // 4)
        x2 = min(width - 1, x1 + width // 3)
        y2 = min(height - 1, y1 + height // 3)
        return [Detection(label="vehicle", confidence=0.91, bbox=(x1, y1, x2, y2))]


class YoloVehicleDetector:
    """Adapter placeholder for YOLO26-compatible object detectors.

    TODO: Load actual YOLO26 weights once the model artifact and runtime package
    are selected. The public `detect` method should keep returning Detection so
    downstream code remains unchanged.
    """

    def __init__(self, weights_path: str | Path, confidence_threshold: float = 0.3) -> None:
        self.weights_path = Path(weights_path)
        self.confidence_threshold = confidence_threshold
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run YOLO inference when real weights/runtime are connected."""
        raise NotImplementedError("YOLO26 runtime integration is intentionally left as a replaceable adapter.")
