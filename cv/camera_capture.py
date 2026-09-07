"""OpenCV capture utilities with test-friendly fallback helpers."""

import logging
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised when OpenCV is unavailable.
    cv2 = None

log = logging.getLogger(__name__)


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
        # Windows 기본 백엔드(MSMF)는 이 장비에서 카메라 **인덱스**를 열면
        # isOpened() 가 True 인데 grabFrame 이 계속 실패한다
        # ("CvCapture_MSMF::grabFrame can't grab frame"). 같은 카메라가
        # DirectShow 로는 open/read 모두 정상인 것을 확인했다.
        # 그래서 Windows + 카메라 인덱스일 때만 DirectShow 를 명시한다.
        # 동영상 파일 경로(str)와 다른 OS 는 기존 경로 그대로다.
        backend = getattr(cv2, "CAP_DSHOW", None)
        self._backend = ("CAP_DSHOW"
                         if (sys.platform == "win32"
                             and isinstance(source, int)
                             and not isinstance(source, bool)
                             and backend is not None)
                         else None)
        if self._backend is not None:
            self._capture = cv2.VideoCapture(source, backend)
        else:
            self._capture = cv2.VideoCapture(source)
        self._frame_index = 0
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Unable to open camera/video source: {source}"
                f" (backend={self._backend or 'default'})")

    def read_frame(self) -> Frame | None:
        """Return the next frame, or None when the stream is exhausted."""
        ok, image = self._capture.read()
        if not ok:
            if self._frame_index == 0:
                # open 은 됐는데 첫 프레임을 못 읽는 상태. MSMF 증상과 같은
                # 모양이라 어느 백엔드로 열렸는지 남긴다.
                log.error("camera %r: opened (backend=%s) but the first frame "
                          "could not be read", self.source,
                          self._backend or "default")
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
