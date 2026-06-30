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

        detector = YoloVehicleDetector("yolo26n.pt")
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
        show: bool = False,
    ) -> None:
        """카메라에서 프레임을 읽어 RC카를 추적하고 콜백을 호출합니다.

        Args:
            on_frame: 각 프레임 처리 후 호출되는 콜백 (None이면 stdout 출력).
            max_frames: 지정 시 해당 수만큼만 처리 후 종료.
            show: True이면 바운딩박스가 그려진 웹캠 화면을 띄웁니다 (q로 종료).
        """
        import cv2

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

                if show:
                    vis = _draw_detections(frame.image.copy(), state)
                    cv2.imshow("RC Car Tracker", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if max_frames and frame.frame_index >= max_frames:
                    break

                # FPS 상한 적용
                sleep_time = self._frame_interval - (time.perf_counter() - now)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            camera.release()
            self._running = False
            import cv2 as _cv2
            _cv2.destroyAllWindows()

    def stop(self) -> None:
        """외부에서 루프를 종료합니다 (멀티스레드 환경용)."""
        self._running = False


def _default_log(state: TrackState) -> None:
    ids = [d.track_id for d in state.detections]
    print(
        f"[frame {state.frame_index:05d}] fps={state.fps:.1f}  "
        f"rc_cars={len(state.detections)}  track_ids={ids}"
    )


def _draw_detections(image, state: TrackState):
    """바운딩박스, track_id, FPS를 이미지에 그립니다."""
    import cv2

    for det in state.detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.label} #{det.track_id} {det.confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(
        image, f"FPS: {state.fps:.1f}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
    )
    return image
