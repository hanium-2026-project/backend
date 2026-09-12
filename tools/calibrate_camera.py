"""천장 카메라 캘리브레이션 — 바닥판 네 모서리의 픽셀 좌표를 측정한다.

측정한 값을 `PipelineConfig.homography_src` 에 넣으면 픽셀 → 실좌표(mm) 변환이
실제 설치 상태에 맞춰진다.

주의: 이 작업 이후 카메라를 움직이면 값이 무효가 된다. 카메라를 완전히 고정한
뒤에 실행하고, 데이터셋 촬영도 같은 상태에서 진행할 것.

사용법::

    python tools/calibrate_camera.py --camera 0
    python tools/calibrate_camera.py --image snapshot.png --lot 1200 1200

조작:
    좌클릭   모서리 지정 (좌상 → 우상 → 우하 → 좌하 순서)
    u        마지막 점 취소
    r        전부 지우고 다시
    s        결과 저장 후 종료
    q        저장 없이 종료

네 점을 다 찍으면 검증 모드로 넘어간다. 바닥판 위 아는 지점(예: 슬롯 중심)을
클릭해 표시되는 mm 좌표가 실측과 맞는지 확인한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv.homography import compute_homography, warp_point  # noqa: E402

# 클릭 순서 — pipeline.config.homography_pairs 의 dst 순서와 일치해야 한다
CORNER_LABELS = [
    "TL 좌상 (맵 0, H)",
    "TR 우상 (맵 W, H)",
    "BR 우하 (맵 W, 0)",
    "BL 좌하 (맵 0, 0) = 원점",
]
WINDOW = "calibration"
FRAME_SIZE = (640, 480)


class Calibrator:
    def __init__(self, lot_w: float, lot_h: float) -> None:
        self.lot_w, self.lot_h = lot_w, lot_h
        self.corners: list[tuple[float, float]] = []
        self.probes: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.homography: np.ndarray | None = None

    # ─── 입력 ────────────────────────────────────────────────────────────────

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.corners) < 4:
            self.corners.append((float(x), float(y)))
            index = len(self.corners) - 1
            print(f"  {CORNER_LABELS[index]}: 픽셀({x}, {y})")
            if len(self.corners) == 4:
                self._build()
            else:
                print(f"  다음: {CORNER_LABELS[len(self.corners)]} 클릭")
        elif self.homography is not None:
            world = warp_point((float(x), float(y)), self.homography)
            self.probes.append(((float(x), float(y)), world))
            print(f"  검증점 픽셀({x}, {y}) → 맵({world[0]:.1f}, {world[1]:.1f}) mm")

    def undo(self) -> None:
        if self.probes:
            self.probes.pop()
        elif self.corners:
            self.corners.pop()
            self.homography = None
            print(f"  다시: {CORNER_LABELS[len(self.corners)]} 클릭")

    def reset(self) -> None:
        self.corners.clear()
        self.probes.clear()
        self.homography = None
        print(f"  초기화 완료: {CORNER_LABELS[0]} 클릭")

    def _build(self) -> None:
        dst = [(0.0, self.lot_h), (self.lot_w, self.lot_h), (self.lot_w, 0.0), (0.0, 0.0)]
        self.homography = compute_homography(self.corners, dst)
        print("\n네 모서리 지정 완료 — 검증 모드")
        print("  바닥판 위 아는 지점을 클릭하면 mm 좌표가 표시됩니다.")
        print("  (예: 슬롯 A1 중심을 클릭 → 425, 1050 근처가 나와야 정상)\n")

    # ─── 표시 ────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        for x, y in self.corners:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 255), -1, cv2.LINE_AA)
        return vis

    # ─── 저장 ────────────────────────────────────────────────────────────────

    def save(self, path: Path, frame_size: tuple[int, int]) -> None:
        data = {
            "frame_size": list(frame_size),
            "lot_width_mm": self.lot_w,
            "lot_height_mm": self.lot_h,
            "homography_src": [[round(x, 1), round(y, 1)] for x, y in self.corners],
            "probes": [{"pixel": list(p), "map_mm": [round(w[0], 1), round(w[1], 1)]}
                       for p, w in self.probes],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n저장: {path}")
        print("\nPipelineConfig 에 다음을 넣으세요:\n")
        pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in self.corners)
        print(f"    homography_src=[{pts}],")
        print(f"    lot_width_mm={self.lot_w}, lot_height_mm={self.lot_h},\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="천장 카메라 캘리브레이션")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="카메라 인덱스 (예: 0)")
    src.add_argument("--image", type=str, help="정지 이미지 경로")
    ap.add_argument("--lot", type=float, nargs=2, default=[1200.0, 1200.0],
                    metavar=("W", "H"), help="바닥판 실측 크기 mm (기본 1200 1200)")
    ap.add_argument("--out", type=str, default="calibration.json", help="저장 경로")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"이미지를 열 수 없습니다: {args.image}")
            return 1
        cap = None
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"카메라를 열 수 없습니다: {args.camera}")
            return 1
        ok, frame = cap.read()
        if not ok:
            print("첫 프레임을 읽지 못했습니다.")
            return 1

    h, w = frame.shape[:2]
    if (w, h) != FRAME_SIZE:
        if cap is not None:
            cap.release()
        print(f"입력은 640x480 원본이어야 합니다: 현재 {w}x{h}")
        return 1

    calib = Calibrator(args.lot[0], args.lot[1])
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, *FRAME_SIZE)
    cv2.moveWindow(WINDOW, 40, 40)
    cv2.setMouseCallback(WINDOW, calib.on_mouse)
    print(__doc__.split("사용법")[0])
    print(f"바닥판 크기: {args.lot[0]:.0f} x {args.lot[1]:.0f} mm")
    print(f"{CORNER_LABELS[0]} 부터 순서대로 클릭하세요.\n")

    while True:
        if cap is not None:
            ok, live = cap.read()
            if ok:
                frame = live
        cv2.imshow(WINDOW, calib.draw(frame))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u"):
            calib.undo()
        elif key == ord("r"):
            calib.reset()
        elif key == ord("s"):
            if len(calib.corners) < 4:
                print("네 모서리를 모두 지정해야 저장할 수 있습니다.")
                continue
            h, w = frame.shape[:2]
            calib.save(Path(args.out), (w, h))
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
