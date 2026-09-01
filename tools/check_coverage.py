#!/usr/bin/env python3
"""수집한 프레임의 위치·방향 커버리지를 표로 보여준다.

재학습 데이터의 품질은 장수가 아니라 커버리지가 결정한다. 8월 실주행 로그를
보면 차가 실제로 다닌 구간만 쿠션 검출이 잘 됐고, 안 가본 구간은 학습 데이터가
없어서 무너졌다. 그래서 촬영 중간중간 "어디가 아직 비었는지"를 봐야 한다.

마커는 채도 높은 빨강이라 YOLO 없이 색으로 찾는다 — 구 모델은 흰 폼으로
학습돼서 새 마커를 못 잡기 때문이다.

사용법::

    python tools/check_coverage.py data/red_marker_20260823
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

import cv2
import numpy as np


# 마커로 볼 색상 범위 (OpenCV H 는 0~179).
#
# 낮 자연광에서는 H≈177(빨강)이지만, 2026-08-31 저녁 실내조명에서 H≈145(자홍)
# 까지 밀렸다 — 조명 색온도가 바뀌면 같은 물체도 색상이 이동한다. 빨강만
# 보면 저녁 프레임에서 마커를 통째로 놓치므로 자홍 대역까지 함께 본다.
HUE_LOW_MAX = 12       # 0 쪽 (주황~빨강)
HUE_HIGH_MIN = 135     # 180 쪽 (자홍~빨강)


def _red_mask(hsv):
    return (((hsv[:, :, 0] < HUE_LOW_MAX) | (hsv[:, :, 0] > HUE_HIGH_MIN))
            & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)).astype(np.uint8)


def find_markers(img) -> list:
    """빨간 덩어리 후보 전체를 (중심, 긴변, 각도, bbox) 목록으로 돌려준다.

    장면에 마커 말고도 붉은 물체가 있을 수 있다 — 실제로 2026-08-31 저녁
    영상에서는 화면 우측의 다른 빨간 물체가 마커보다 커서, "가장 큰 것"만
    보면 매번 엉뚱한 것을 집었다. 호출측이 차량과의 인접성으로 고르게 한다.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = _red_mask(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        if not (800 < st[i, cv2.CC_STAT_AREA] < 12000):
            continue
        pts = cv2.findNonZero((lab == i).astype(np.uint8) * 255)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        if rw < rh:
            ang += 90
        x, y, bw, bh = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                        st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        out.append(((cx, cy), max(rw, rh), ang % 180.0,
                    (int(x), int(y), int(x + bw), int(y + bh))))
    return out


def find_marker(img):
    """가장 큰 빨간 덩어리 하나. 커버리지 집계용 (차량 정보가 없을 때).

    라벨링처럼 정확도가 중요한 곳에서는 find_markers() 로 후보를 모두 받아
    차량과의 인접성으로 고를 것.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = _red_mask(hsv)
    # OPEN 으로 잡티를, CLOSE 로 배선에 끊긴 마커를 메운다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    cand = [i for i in range(1, n) if 800 < st[i, cv2.CC_STAT_AREA] < 12000]
    if not cand:
        return None
    i = max(cand, key=lambda i: st[i, cv2.CC_STAT_AREA])
    pts = cv2.findNonZero((lab == i).astype(np.uint8) * 255)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:
        ang += 90
    x, y, bw, bh = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                    st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
    return ((cx, cy), max(rw, rh), ang % 180.0,
            (int(x), int(y), int(x + bw), int(y + bh)))


def main() -> int:
    ap = argparse.ArgumentParser(description="수집 프레임 커버리지 점검")
    ap.add_argument("frames", help="프레임 디렉토리")
    ap.add_argument("--calibration", default="calibration_new.json")
    ap.add_argument("--cell", type=float, default=300.0, help="격자 한 칸 mm")
    ap.add_argument("--target", type=int, default=40,
                    help="칸당 목표 장수 (이 미만이면 부족 표시)")
    a = ap.parse_args()

    calib = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    W = float(calib.get("lot_width_mm", 1200.0))
    Hm = float(calib.get("lot_height_mm", 1200.0))
    H = cv2.getPerspectiveTransform(
        np.asarray(calib["homography_src"], np.float32),
        np.asarray([[0, Hm], [W, Hm], [W, 0], [0, 0]], np.float32))

    files = sorted(glob.glob(str(Path(a.frames) / "*.jpg")))
    if not files:
        raise SystemExit(f"[ERROR] 프레임이 없습니다: {a.frames}")

    pos = collections.Counter()
    ang = collections.Counter()
    miss = outside = 0
    for f in files:
        img = cv2.imread(f)
        r = find_marker(img) if img is not None else None
        if r is None:
            miss += 1
            continue
        (cx, cy), _, deg, _bbox = r
        q = cv2.perspectiveTransform(np.asarray([[[cx, cy]]], np.float32), H)[0][0]
        if not (0 <= q[0] <= W and 0 <= q[1] <= Hm):
            outside += 1
            continue
        pos[(int(q[0] // a.cell), int(q[1] // a.cell))] += 1
        ang[int(deg // 30)] += 1

    nx, ny = int(W // a.cell), int(Hm // a.cell)
    print(f"\n프레임 {len(files)}장 — 마커 검출 {len(files)-miss}, "
          f"판 밖 {outside}, 미검출 {miss}")
    print(f"\n=== 위치 커버리지 ({a.cell:.0f}mm 격자, 칸당 목표 {a.target}장) ===")
    for y in range(ny - 1, -1, -1):
        cells = []
        for x in range(nx):
            n = pos.get((x, y), 0)
            mark = "·" if n == 0 else ("!" if n < a.target else " ")
            cells.append(f"{n:>6}{mark}")
        print(f"  y{y*a.cell:>5.0f} " + "".join(cells))
    print("         " + "".join(f"{x*a.cell:>7.0f}" for x in range(nx)))
    empty = [(x, y) for y in range(ny) for x in range(nx) if not pos.get((x, y))]
    thin = [(x, y) for y in range(ny) for x in range(nx)
            if 0 < pos.get((x, y), 0) < a.target]
    print(f"\n  빈 칸 {len(empty)}/{nx*ny}, 부족(!) {len(thin)}")
    if empty:
        print("  비어 있는 구역(mm): "
              + ", ".join(f"({x*a.cell:.0f}~{(x+1)*a.cell:.0f}, "
                          f"{y*a.cell:.0f}~{(y+1)*a.cell:.0f})" for x, y in empty[:8])
              + (" ..." if len(empty) > 8 else ""))

    print(f"\n=== 마커 방향 커버리지 (직사각형이라 0~180°) ===")
    top = max(ang.values()) if ang else 1
    for b in range(6):
        n = ang.get(b, 0)
        print(f"  {b*30:>3}~{b*30+30:<3}° {n:>6}  {'#' * int(40 * n / top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
