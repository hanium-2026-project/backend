"""OpenCV capture utilities with test-friendly fallback helpers."""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised when OpenCV is unavailable.
    cv2 = None


@dataclass
class Frame:
    """A captured frame and lightweight metadata used by downstream CV steps."""

    image: np.ndarray
    source: str
    frame_index: int


class CameraCapture:
    """Read frames from a webcam index or a video file using OpenCV.

    The class keeps capture handling isolated so production camera adapters can
    add reconnect/backoff behavior without changing detector interfaces.
    """

    def __init__(self, source: int | str = 0) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed. Install opencv-python-headless to use CameraCapture.")
        self.source = source
        self._capture = cv2.VideoCapture(source)
        self._frame_index = 0
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera/video source: {source}")

    def read_frame(self) -> Frame | None:
        """Return the next frame, or None when the stream is exhausted."""
        ok, image = self._capture.read()
        if not ok:
            return None
        self._frame_index += 1
        return Frame(image=image, source=str(self.source), frame_index=self._frame_index)

    def release(self) -> None:
        """Release the OpenCV capture resource."""
        self._capture.release()


def create_synthetic_frame(width: int = 320, height: int = 180) -> Frame:
    """Create a deterministic image for detector and homography tests."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[40:120, 80:180] = np.array([40, 180, 90], dtype=np.uint8)
    image[55:75, 110:160] = np.array([230, 230, 230], dtype=np.uint8)
    return Frame(image=image, source="synthetic", frame_index=1)


def frame_to_jpeg_bytes(frame: Frame) -> bytes:
    """Encode a frame as JPEG for future streaming APIs."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required to encode frames.")
    ok, buffer = cv2.imencode(".jpg", frame.image)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return bytes(buffer)
