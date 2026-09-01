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
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 2클래스 데이터셋의 클래스 순서 — data.yaml 과 반드시 일치해야 한다.
#
# 주의: 이 순서는 **가중치 내부 순서와 다를 수 있다.** 실제로 현재 best.pt 는
# {0: FRONT_CUSHION, 1: rc_car} 라 정확히 반대다. 그래서 box.cls 를 그대로
# 쓰면 안 되고, detector 가 이름으로 정규화한 label(LABEL_CAR/LABEL_CUSHION)
# 을 아래 표로 옮겨야 한다.
CLASS_NAMES = ["rc_car", "front_cushion"]
CLASS_CAR = 0
CLASS_CUSHION = 1

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
                    help="탐지 임계값 (낮게 잡고 검수에서 지우는 편이 낫다). "
                         "하드케이스 수집용이면 0.10~0.15 를 권장한다 — production "
                         "값(0.4)으로 돌리면 정작 우리가 원하는 실패 프레임에서만 "
                         "쿠션 박스가 안 생겨서 사람이 전부 새로 그려야 한다")
    ap.add_argument("--imgsz", type=int, default=1280, help="추론 해상도")
    ap.add_argument("--labels-dir", default=None,
                    help="라벨 저장 위치 (기본: 이미지와 같은 폴더)")
    ap.add_argument("--review", action="store_true",
                    help="결과를 눈으로 확인할 시각화 이미지도 저장")
    # 마커를 흰 폼 → 빨간 판으로 교체한 뒤로는, 흰 폼으로 학습된 기존 가중치가
    # 새 마커를 못 잡는다. 채도 높은 단색이라 색으로 찾는 편이 훨씬 정확하다.
    ap.add_argument("--marker-color", choices=["none", "red"], default="none",
                    help="front_cushion 을 YOLO 대신 색으로 찾는다 (빨간 마커용)")
    # 손으로 차를 옮기는 동안 몸이 카메라를 가린 프레임이 상당수 섞인다. 라벨이
    # 없는 채로 학습에 들어가면 "여기엔 차가 없다"를 가르치므로 걸러내야 한다.
    # (일부는 사람=차 오탐을 줄이는 배경 이미지로 유용하지만 1/3은 과하다.)
    ap.add_argument("--quarantine", default=None,
                    help="차량이 안 잡힌 프레임을 이 디렉토리로 옮긴다")
    ap.add_argument("--calibration", default=None,
                    help="바닥판 캘리브레이션. 주면 판 밖 검출을 버린다 — "
                         "사람 발/삼각대 오탐이 rc_car 라벨로 굳는 것을 막는다")
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

    from cv.vehicle_detector import (LABEL_CAR, LABEL_CUSHION,
                                     YoloVehicleDetector)
    from cv.association import _inside_or_adjacent
    from tools.check_coverage import find_markers

    # 판 밖 판정기. 캘리브레이션을 안 주면 필터를 걸지 않는다(기존 동작 유지).
    in_lot = None
    if args.calibration:
        import json
        import numpy as np
        calib = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        W = float(calib.get("lot_width_mm", 1200.0))
        Hm = float(calib.get("lot_height_mm", 1200.0))
        Hmat = cv2.getPerspectiveTransform(
            np.asarray(calib["homography_src"], np.float32),
            np.asarray([[0, Hm], [W, Hm], [W, 0], [0, 0]], np.float32))
        # 가장자리에 걸친 차를 잘라내지 않도록 여유를 준다.
        margin = 150.0

        def in_lot(bbox):
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            q = cv2.perspectiveTransform(
                np.asarray([[[cx, cy]]], np.float32), Hmat)[0][0]
            return (-margin <= q[0] <= W + margin
                    and -margin <= q[1] <= Hm + margin)

    class_of = {LABEL_CAR: CLASS_CAR, LABEL_CUSHION: CLASS_CUSHION}

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
    no_car: list[str] = []
    no_cushion: list[str] = []
    multi: list[str] = []

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        dets = detector.detect(image)

        # 판 밖 검출을 먼저 버린다. 이걸 안 하면 좌상단 사람 발/삼각대 오탐이
        # rc_car 정답 라벨로 굳어서, 재학습이 오탐을 오히려 학습해버린다.
        if in_lot is not None:
            dets = [d for d in dets if in_lot(d.bbox)]

        cars = [d for d in dets if d.label == LABEL_CAR]
        if args.marker_color == "red":
            # 색으로 찾은 마커는 차량에 인접한 것만 채택한다 (association 과 같은
            # 기준). 배선·바닥 반사 같은 붉은 잡티가 라벨로 들어가지 않게 한다.
            cushions = []
            if cars:
                car = max(cars, key=lambda d: d.confidence)
                ccx, ccy = ((car.bbox[0] + car.bbox[2]) / 2.0,
                            (car.bbox[1] + car.bbox[3]) / 2.0)
                # 장면에 붉은 물체가 여럿일 수 있다. 가장 큰 것이 아니라
                # 차량에 인접한 것 중 가장 가까운 것을 고른다.
                near = [(math.hypot(c[0][0] - ccx, c[0][1] - ccy), c)
                        for c in find_markers(image)
                        if _inside_or_adjacent(c[0], car.bbox)]
                if near:
                    cushions = [min(near, key=lambda t: t[0])[1][3]]
        else:
            cushions = [d.bbox for d in dets if d.label == LABEL_CUSHION]

        # 라벨별로 클래스 번호를 붙인다. 예전에는 인자 기본값(rc_car)이 그대로
        # 쓰여서 2클래스 가중치로 돌리면 쿠션까지 전부 rc_car 로 라벨링됐다.
        lines = ([to_yolo_line(d.bbox, w, h, CLASS_CAR) for d in cars]
                 + [to_yolo_line(b, w, h, CLASS_CUSHION) for b in cushions])
        # 끝 개행을 붙인다. 없으면 cat 으로 라벨을 모아 점검할 때 파일 경계에서
        # 두 줄이 한 줄로 붙어 클래스 집계가 틀린다.
        (label_dir / f"{path.stem}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8")
        total_boxes += len(lines)

        if not cars:
            no_car.append(path.name)
        if cars and not cushions:
            # 우리가 노리는 하드 포지티브다 — 쿠션은 실제로 있는데 모델이 놓쳤다.
            no_cushion.append(path.name)
        if len(cars) > 1 or len(cushions) > 1:
            multi.append(f"{path.name}(car {len(cars)}/cushion {len(cushions)})")

        if args.review:
            vis = image.copy()
            # 차량은 초록, 마커는 주황 — 클래스가 뒤바뀐 라벨을 눈으로 잡아낸다.
            for d in cars:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"rc_car {d.confidence:.2f}", (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for b in cushions:
                x1, y1, x2, y2 = b
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(vis, "front_cushion", (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imwrite(str(review_dir / path.name), vis)

    yaml_path = img_dir.parent / "data.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(DATA_YAML.format(root=img_dir.parent.resolve()),
                             encoding="utf-8")

    print(f"\n이미지 {len(paths)}장 → 박스 {total_boxes}개 ({label_dir}/*.txt)")
    if no_car:
        print(f"\n차량 미검출 {len(no_car)}장 ({100*len(no_car)/len(paths):.0f}%) "
              "— 대개 옮기는 동안 몸에 가린 프레임입니다.")
        print("  " + ", ".join(no_car[:10]) + (" ..." if len(no_car) > 10 else ""))
        if args.quarantine:
            qdir = Path(args.quarantine)
            qdir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for name in no_car:
                src = img_dir / name
                if src.is_file():
                    src.replace(qdir / name)
                    moved += 1
                # 함께 만든 빈 라벨도 데이터셋에 남기지 않는다
                lbl = label_dir / f"{Path(name).stem}.txt"
                if lbl.is_file():
                    lbl.unlink()
                # 검수 이미지도 같이 치운다 — 남겨두면 격리된 프레임을
                # 검수 대상으로 착각한다.
                rv = review_dir / name
                if rv.is_file():
                    rv.unlink()
            print(f"  → {moved}장을 {qdir}/ 로 옮기고 빈 라벨을 지웠습니다.")
    if no_cushion:
        print(f"\n쿠션 미검출 {len(no_cushion)}장 "
              f"({100 * len(no_cushion) / len(paths):.0f}%) — **이게 재학습의 핵심 데이터입니다.**")
        print("  쿠션은 실제로 프레임 안에 있으므로, 검수 때 반드시 손으로 박스를 그려주세요.")
        print("  " + ", ".join(no_cushion[:10]) + (" ..." if len(no_cushion) > 10 else ""))
    if multi:
        print(f"\n한 클래스가 2개 이상 {len(multi)}장 — 오탐일 수 있습니다:")
        print("  " + ", ".join(multi[:10]) + (" ..." if len(multi) > 10 else ""))
    if args.review:
        print(f"\n시각화: {review_dir}/ — 먼저 여기를 훑어보고 검수 범위를 잡으세요.")

    print("\n다음 단계")
    print("  1. 이미지와 .txt 를 함께 Roboflow 에 업로드 (박스가 미리 그려진 채로 열립니다)")
    print("  2. 차량·쿠션 박스 검수 — 헐거운 것 조정, 오탐 삭제")
    print("  3. 위 '쿠션 미검출' 목록은 손으로 쿠션 박스를 추가 (클래스 1)")
    print("  4. 70/20/10 으로 split 후 YOLO 포맷 export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
