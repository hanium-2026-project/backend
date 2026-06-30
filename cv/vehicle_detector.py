"""Vehicle detector interfaces and YOLO-based implementations."""

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
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel)
    track_id: int | None = None


class VehicleDetector(Protocol):
    """Detector interface shared by mock and YOLO adapters."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return vehicle detections for a BGR image array."""


class MockVehicleDetector:
    """Deterministic detector used for tests and demos without model weights."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return []
        x1 = max(0, width // 4)
        y1 = max(0, height // 4)
        x2 = min(width - 1, x1 + width // 3)
        y2 = min(height - 1, y1 + height // 3)
        return [Detection(label="rc_car", confidence=0.91, bbox=(x1, y1, x2, y2), track_id=1)]


class YoloVehicleDetector:
    """RC카 탐지 및 추적기 (ultralytics YOLO26 + ByteTrack).

    RC카 전용 커스텀 가중치가 없을 경우 기본 COCO 가중치(yolo26n.pt)를 사용하며,
    COCO의 'car' 클래스(2번)를 RC카로 취급합니다.
    커스텀 가중치 파일을 지정하면 해당 모델의 전체 클래스를 RC카로 인식합니다.
    """

    # COCO 클래스 중 차량 관련 클래스 ID
    _COCO_VEHICLE_CLASSES: frozenset[int] = frozenset({2, 3, 5, 7})  # car, motorcycle, bus, truck

    def __init__(
        self,
        weights_path: str | Path = "yolo26n.pt",
        confidence_threshold: float = 0.4,
        tracker: str = "bytetrack.yaml",
        device: str = "",
        custom_model: bool = False,
    ) -> None:
        """
        Args:
            weights_path: YOLO 가중치 파일 경로. 기본값은 ultralytics 자동 다운로드.
            confidence_threshold: 탐지 최소 신뢰도.
            tracker: 추적 알고리즘 설정 파일 (bytetrack.yaml / botsort.yaml).
            device: 추론 디바이스 ('cpu', '0', '' → 자동).
            custom_model: True이면 모든 클래스를 RC카로 처리. False이면 COCO 차량 클래스 필터링.
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics 패키지가 필요합니다: pip install ultralytics") from e

        self.weights_path = Path(weights_path)
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker
        self.device = device
        self.custom_model = custom_model

        self._model = YOLO(str(self.weights_path))

    def detect(self, image: np.ndarray) -> list[Detection]:
        """단일 프레임에서 RC카를 탐지합니다 (추적 없이 순수 detect 모드)."""
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_and_track(self, image: np.ndarray) -> list[Detection]:
        """단일 프레임에서 RC카를 탐지하고 track_id를 부여합니다."""
        results = self._model.track(
            source=image,
            conf=self.confidence_threshold,
            tracker=self.tracker_config,
            device=self.device,
            persist=True,
            verbose=False,
        )
        return self._parse_results(results)

    def _parse_results(self, results) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if not self.custom_model and cls_id not in self._COCO_VEHICLE_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                label = result.names.get(cls_id, "rc_car") if not self.custom_model else "rc_car"
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                ))
        return detections
