"""Vehicle detector interfaces and YOLO-based implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


# 내부 표준 라벨 — 파이프라인 전역에서 이 값으로만 비교한다
LABEL_CAR = "rc_car"
LABEL_CUSHION = "front_cushion"


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

    가중치 경로와 custom_model 여부는 반드시 외부에서 명시적으로 전달합니다.
    기본 가중치(yolo26n.pt)는 RC카 전용 가중치가 없을 때 COCO 모델로 동작하는 fallback입니다.

    Args:
        weights_path: YOLO 가중치 파일 경로. 파인튜닝된 RC카 모델은 실행 시 인자로 전달.
        confidence_threshold: 탐지 최소 신뢰도.
        tracker: 추적 알고리즘 설정 파일 (bytetrack.yaml / botsort.yaml).
        device: 추론 디바이스 ('cpu', '0', '' → 자동).
        custom_model: 파인튜닝 가중치 사용 여부. True 이면 모델이 낸 클래스 이름을
                      내부 라벨(rc_car / front_cushion)로 정규화하고, 모르는 이름은
                      차량으로 본다. False 이면 COCO 차량 클래스만 필터링한다.
    """

    # COCO 클래스 중 차량 관련 클래스 ID
    _COCO_VEHICLE_CLASSES: frozenset[int] = frozenset({2, 3, 5, 7})  # car, motorcycle, bus, truck

    # 파인튜닝 모델의 클래스 이름 → 내부 라벨. 학습 시 표기(대소문자·구분자)가
    # 달라도 같은 의미로 받아들인다.
    _LABEL_ALIASES: dict[str, str] = {
        "rc_car": LABEL_CAR, "rccar": LABEL_CAR, "car": LABEL_CAR, "vehicle": LABEL_CAR,
        "front_cushion": LABEL_CUSHION, "frontcushion": LABEL_CUSHION,
        "cushion": LABEL_CUSHION, "front": LABEL_CUSHION,
    }

    def __init__(
        self,
        weights_path: str | Path = "yolo26n.pt",
        confidence_threshold: float = 0.4,
        tracker: str = "bytetrack.yaml",
        device: str = "",
        custom_model: bool = False,
        imgsz: int = 1280,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics 패키지가 필요합니다: pip install ultralytics") from e

        self.weights_path = Path(weights_path)
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker
        self.device = device
        self.custom_model = custom_model
        # 천장 카메라에서 RC카가 화면 대비 작게 보이므로 기본 추론 해상도를 높게 유지
        self.imgsz = imgsz

        self._model = YOLO(str(self.weights_path))

    def detect(self, image: np.ndarray) -> list[Detection]:
        """단일 프레임에서 RC카를 탐지합니다 (추적 없이 순수 detect 모드)."""
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_and_track(self, image: np.ndarray) -> list[Detection]:
        """단일 프레임에서 RC카를 탐지하고 track_id를 부여합니다."""
        results = self._model.track(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
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
                label = self._resolve_label(result.names.get(cls_id, ""))
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                ))
        return detections

    def _resolve_label(self, raw_name: str) -> str:
        """모델 클래스 이름을 내부 라벨로 정규화한다.

        1클래스 모델(rc_car 만 학습)은 모든 탐지가 차량이고, 2클래스 모델은
        전방 쿠션을 함께 낸다. 알 수 없는 이름은 커스텀 모델에서는 차량으로,
        COCO 모델에서는 원래 이름을 유지한다.
        """
        key = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
        alias = self._LABEL_ALIASES.get(key)
        if alias is not None:
            return alias
        return LABEL_CAR if self.custom_model else (raw_name or "vehicle")
