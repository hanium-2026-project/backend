"""RC카 실시간 추적 파이프라인.

CameraCapture → YoloVehicleDetector.detect_and_track → 콜백/WebSocket 전달
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .camera_capture import CameraCapture, Frame
from .vehicle_detector import Detection, YoloVehicleDetector


@dataclass
class TrackState:
    """한 프레임에서의 추적 결과 스냅샷."""

    frame_index: int
    timestamp: float
    detections: list[Detection]
    fps: float = 0.0


class RCCarTracker:
    """카메라 스트림에서 RC카를 실시간으로 추적하는 고수준 루프.

    Usage::

        detector = YoloVehicleDetector("yolo11n.pt")
        tracker = RCCarTracker(source=0, detector=detector)
        tracker.run(on_frame=lambda state: print(state))
    """

    def __init__(
        self,
        source: int | str = 0,
        detector: YoloVehicleDetector | None = None,
        weights_path: str = "yolo26n.pt",
        confidence_threshold: float = 0.4,
        max_fps: float = 30.0,
    ) -> None:
        self._source = source
        self._detector = detector or YoloVehicleDetector(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
        )
        self._frame_interval = 1.0 / max_fps
        self._running = False

    def run(
        self,
        on_frame: Callable[[TrackState], None] | None = None,
        max_frames: int | None = None,
    ) -> None:
        """카메라에서 프레임을 읽어 RC카를 추적하고 콜백을 호출합니다.

        Args:
            on_frame: 각 프레임 처리 후 호출되는 콜백 (None이면 stdout 출력).
            max_frames: 지정 시 해당 수만큼만 처리 후 종료.
        """
        camera = CameraCapture(self._source)
        self._running = True
        prev_time = time.perf_counter()

        try:
            while self._running:
                frame = camera.read_frame()
                if frame is None:
                    break

                detections = self._detector.detect_and_track(frame.image)
                now = time.perf_counter()
                elapsed = now - prev_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                prev_time = now

                state = TrackState(
                    frame_index=frame.frame_index,
                    timestamp=now,
                    detections=detections,
                    fps=fps,
                )

                if on_frame:
                    on_frame(state)
                else:
                    _default_log(state)

                if max_frames and frame.frame_index >= max_frames:
                    break

                # FPS 상한 적용
                sleep_time = self._frame_interval - (time.perf_counter() - now)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            camera.release()
            self._running = False

    def stop(self) -> None:
        """외부에서 루프를 종료합니다 (멀티스레드 환경용)."""
        self._running = False


def _default_log(state: TrackState) -> None:
    ids = [d.track_id for d in state.detections]
    print(
        f"[frame {state.frame_index:05d}] fps={state.fps:.1f}  "
        f"rc_cars={len(state.detections)}  track_ids={ids}"
    )
