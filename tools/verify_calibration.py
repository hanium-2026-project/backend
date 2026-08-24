#!/usr/bin/env python3
"""캘리브레이션을 눈으로 검증한다 — 맵 좌표계를 카메라 영상 위에 되그린다.

calibrate_camera.py 는 네 모서리를 찍으면 저장까지 끝나므로, 검증(아는 지점
클릭)을 건너뛰기 쉽다. 모서리를 잘못된 순서로 찍어도 homography 자체는
'정상적으로' 만들어지기 때문에 숫자만 봐서는 뒤집힘을 알아채기 어렵다.

이 도구는 100mm 격자와 슬롯 중심을 영상에 되그린다. 격자가 바닥판 테이프와
맞고 A1~A4 가 입구쪽(아래), B1~B4 가 출구쪽(위)에 찍히면 정상이다.

사용법::

    python tools/verify_calibration.py --camera 0
    python tools/verify_calibration.py --image snapshot.png --show
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.parking_env import SLOT_COORDINATES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="캘리브레이션 시각 검증")
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--camera", type=int, help="카메라 인덱스")
    src_group.add_argument("--image", help="정지 이미지 경로")
    ap.add_argument("--calibration", default="calibration_new.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--grid-mm", type=float, default=100.0)
    a = ap.parse_args()

    calib_path = Path(a.calibration)
    if not calib_path.is_file():
        raise SystemExit(f"[ERROR] 캘리브레이션 파일이 없습니다: {calib_path}")
    calib = json.loads(calib_path.read_text(encoding="utf-8"))

    W = float(calib.get("lot_width_mm", 1200.0))
    Hm = float(calib.get("lot_height_mm", 1200.0))
    src = np.asarray(calib["homography_src"], dtype=np.float32)
    dst = np.asarray([[0, Hm], [W, Hm], [W, 0], [0, 0]], dtype=np.float32)
    # 맵(mm) → 픽셀. calibrate_camera 가 만든 것의 역방향이다.
    mm_to_px = cv2.getPerspectiveTransform(dst, src)

    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            raise SystemExit(f"[ERROR] 이미지를 열 수 없습니다: {a.image}")
    else:
        cap = cv2.VideoCapture(a.camera)
        if not cap.isOpened():
            raise SystemExit(f"[ERROR] 카메라 {a.camera} 를 열 수 없습니다. "
                             "인덱스를 바꾸거나 카메라 권한을 확인하세요.")
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("[ERROR] 프레임을 읽지 못했습니다.")

    h, w = frame.shape[:2]
    print(f"[INFO] 프레임 {w}x{h}")
    expect = calib.get("frame_size")
    if expect and [w, h] != [int(expect[0]), int(expect[1])]:
        print(f"[WARN] 캘리브레이션 기준 해상도 {int(expect[0])}x{int(expect[1])} 와 "
              "다릅니다 — 좌표가 어긋납니다.")

    def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        q = cv2.perspectiveTransform(
            np.asarray([[[x_mm, y_mm]]], dtype=np.float32), mm_to_px)[0][0]
        return int(round(float(q[0]))), int(round(float(q[1])))

    vis = frame.copy()

    # 100mm 격자 — 바닥판 테이프와 맞는지 보는 것이 1차 검증이다.
    n_x = int(W // a.grid_mm)
    n_y = int(Hm // a.grid_mm)
    for i in range(n_x + 1):
        x = i * a.grid_mm
        cv2.line(vis, to_px(x, 0.0), to_px(x, Hm), (90, 90, 90), 1)
    for j in range(n_y + 1):
        y = j * a.grid_mm
        cv2.line(vis, to_px(0.0, y), to_px(W, y), (90, 90, 90), 1)

    # 바닥판 외곽 + 원점
    border = [to_px(0, 0), to_px(W, 0), to_px(W, Hm), to_px(0, Hm)]
    cv2.polylines(vis, [np.asarray(border, np.int32)], True, (0, 255, 0), 2)
    ox, oy = to_px(0, 0)
    cv2.circle(vis, (ox, oy), 8, (255, 0, 255), -1)
    cv2.putText(vis, "origin (0,0)", (ox + 12, oy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # 슬롯 중심 — A행이 아래(입구), B행이 위(출구)로 찍혀야 정상이다.
    print("[INFO] 슬롯 중심 픽셀 좌표:")
    for name, (sx, sy) in SLOT_COORDINATES.items():
        px, py = to_px(sx, sy)
        print(f"       {name} ({sx:.0f},{sy:.0f})mm → 픽셀 ({px},{py})")
        cv2.circle(vis, (px, py), 7, (0, 165, 255), -1)
        cv2.putText(vis, name, (px + 10, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    out = Path(a.out) if a.out else Path("runs") / (
        "calib_check_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"[DONE] 저장: {out}")
    print("       격자가 바닥판 테이프와 맞고, A행이 아래·B행이 위면 정상입니다.")

    if a.show:
        cv2.imshow("CALIBRATION CHECK (아무 키나 누르면 종료)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
