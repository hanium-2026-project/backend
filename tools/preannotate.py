"""기존 가중치로 차량 bbox 를 미리 라벨링한다 (라벨링 시간 단축용).

RC_CAR 는 이미 잘 잡는 모델(best05 등)이 있으므로 자동으로 뽑아두고, 사람은
검수와 FRONT_CUSHION 추가만 하면 된다. 결과는 YOLO 포맷(.txt)이라 Roboflow 에
이미지와 함께 올리면 박스가 미리 그려진 상태로 열린다.

주의: 자동 라벨은 완벽하지 않다. 반드시 전수 검수할 것 — 특히 놓친 차량,
헐거운 박스, 오탐. 검수를 건너뛰면 잘못된 라벨이 그대로 학습된다.

사용법::

    python tools/preannotate.py frames/ --weights best05.pt
    python tools/preannotate.py frames/ --weights best05.pt --conf 0.3 --review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 2클래스 데이터셋의 클래스 순서 — data.yaml 과 반드시 일치해야 한다
CLASS_NAMES = ["rc_car", "front_cushion"]
CLASS_CAR = 0

DATA_YAML = """\
# Roboflow 등에서 재생성해도 되지만, 클래스 순서는 아래와 같아야 한다.
path: {root}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: rc_car
  1: front_cushion
"""


def to_yolo_line(bbox, img_w: int, img_h: int, cls: int = CLASS_CAR) -> str:
    """픽셀 bbox → YOLO 정규화 포맷 (cls cx cy w h)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="기존 가중치로 차량 bbox 사전 라벨링")
    ap.add_argument("images", help="이미지 디렉토리")
    ap.add_argument("--weights", required=True, help="사전 라벨링에 쓸 가중치")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="탐지 임계값 (낮게 잡고 검수에서 지우는 편이 낫다)")
    ap.add_argument("--imgsz", type=int, default=1280, help="추론 해상도")
    ap.add_argument("--labels-dir", default=None,
                    help="라벨 저장 위치 (기본: 이미지와 같은 폴더)")
    ap.add_argument("--review", action="store_true",
                    help="결과를 눈으로 확인할 시각화 이미지도 저장")
    args = ap.parse_args()

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        print(f"디렉토리가 아닙니다: {img_dir}")
        return 1

    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        print(f"이미지가 없습니다: {img_dir}")
        return 1

    from cv.vehicle_detector import YoloVehicleDetector

    detector = YoloVehicleDetector(
        weights_path=args.weights, confidence_threshold=args.conf,
        imgsz=args.imgsz, custom_model=True,
    )

    label_dir = Path(args.labels_dir) if args.labels_dir else img_dir
    label_dir.mkdir(parents=True, exist_ok=True)
    review_dir = img_dir / "_review"
    if args.review:
        review_dir.mkdir(exist_ok=True)

    total_boxes = 0
    empty: list[str] = []
    multi: list[str] = []

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        dets = detector.detect(image)

        lines = [to_yolo_line(d.bbox, w, h) for d in dets]
        (label_dir / f"{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        total_boxes += len(lines)

        if not dets:
            empty.append(path.name)
        elif len(dets) > 1:
            multi.append(f"{path.name}({len(dets)})")

        if args.review:
            vis = image.copy()
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{d.confidence:.2f}", (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(str(review_dir / path.name), vis)

    yaml_path = img_dir.parent / "data.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(DATA_YAML.format(root=img_dir.parent.resolve()),
                             encoding="utf-8")

    print(f"\n이미지 {len(paths)}장 → 박스 {total_boxes}개 ({label_dir}/*.txt)")
    if empty:
        print(f"\n탐지 0건 {len(empty)}장 — 차량이 없거나 모델이 놓친 것입니다. 확인 필요:")
        print("  " + ", ".join(empty[:10]) + (" ..." if len(empty) > 10 else ""))
    if multi:
        print(f"\n2개 이상 탐지 {len(multi)}장 — 오탐일 수 있습니다:")
        print("  " + ", ".join(multi[:10]) + (" ..." if len(multi) > 10 else ""))
    if args.review:
        print(f"\n시각화: {review_dir}/ — 먼저 여기를 훑어보고 검수 범위를 잡으세요.")

    print("\n다음 단계")
    print("  1. 이미지와 .txt 를 함께 Roboflow 에 업로드 (박스가 미리 그려진 채로 열립니다)")
    print("  2. 차량 박스 검수 — 놓친 것 추가, 헐거운 것 조정, 오탐 삭제")
    print("  3. FRONT_CUSHION 을 수동으로 추가 (클래스 1)")
    print("  4. 70/20/10 으로 split 후 YOLO 포맷 export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
