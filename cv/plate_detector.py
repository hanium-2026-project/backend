"""License plate detector interfaces and deterministic MVP implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PlateDetection:
    """Detected plate text and image-space bounding box."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


class PlateDetector(Protocol):
    """Protocol for OCR/plate detectors that can be swapped in later."""

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Return detected license plates."""


class MockPlateDetector:
    """Return a stable Korean plate sample for tests and UI simulations."""

    def __init__(self, plate_text: str = "12가3456") -> None:
        self.plate_text = plate_text

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Produce one deterministic plate box for any non-empty frame."""
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return []
        return [
            PlateDetection(
                text=self.plate_text,
                confidence=0.88,
                bbox=(width // 3, height // 3, min(width - 1, width // 3 + 80), min(height - 1, height // 3 + 24)),
            )
        ]


class OcrPlateDetector:
    """Future OCR-backed plate detector adapter.

    TODO: Connect a production OCR model or API after privacy, latency, and
    deployment constraints are finalized.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Run a real OCR detector once model integration is complete."""
        raise NotImplementedError("OCR plate detector is a replaceable production adapter.")
