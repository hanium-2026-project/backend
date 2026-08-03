"""캘리브레이션·A/B 배치 검증용 — RL·통신 없이 좌표만 콘솔에 찍는다.

ESP32 연결이나 RL 정책과 무관하게, 탐지→homography→heading 만 돌려서
"차량을 여기 놓으면 이 좌표가 나와야 한다"를 즉시 대조할 수 있게 한다.

사용법::

    python tools/debug_pose.py --camera 0 --weights best05.pt \
        --calibration calibration.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv.heading import HeadingEstimator
from cv.homography import compute_homography, warp_point
from cv.tracker import RCCarTracker
from cv.vehicle_detector import YoloVehicleDetector
from rl.parking_env import NODE_COORDINATES, SLOT_COORDINATES
from rl.bridge import position_to_node


def nearest_slot(pos: tuple[float, float]) -> tuple[str, float]:
    import math
    best, best_d = None, float("inf")
    for name, c in SLOT_COORDINATES.items():
        d = math.hypot(c[0] - pos[0], c[1] - pos[1])
        if d < best_d:
            best, best_d = name, d
    return best, best_d


def main() -> int:
    ap = argparse.ArgumentParser(description="좌표 진단 (ESP32/RL 없이)")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calibration", required=True,
                    help="tools/calibrate_camera.py 로 저장한 JSON")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--every", type=int, default=5, help="N 프레임마다 출력")
    args = ap.parse_args()

    import json
    calib = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    src = [tuple(p) for p in calib["homography_src"]]
    lot_w, lot_h = calib["lot_width_mm"], calib["lot_height_mm"]
    dst = [(0.0, lot_h), (lot_w, lot_h), (lot_w, 0.0), (0.0, 0.0)]
    H = compute_homography(src, dst)
    print(f"캘리브레이션 로드: {args.calibration} ({lot_w:.0f}x{lot_h:.0f}mm)")
    print(f"참고 슬롯 좌표: A1={SLOT_COORDINATES['A1']}  A4={SLOT_COORDINATES['A4']}  "
          f"B1={SLOT_COORDINATES['B1']}  B4={SLOT_COORDINATES['B4']}")
    print(f"entrance={NODE_COORDINATES['entrance']}  junction={NODE_COORDINATES['junction']}\n")

    camera = int(args.camera) if args.camera.isdigit() else args.camera
    detector = YoloVehicleDetector(weights_path=args.weights, confidence_threshold=args.conf,
                                   imgsz=args.imgsz, custom_model=True)
    heading = HeadingEstimator(min_move=30.0)

    def on_frame(state):
        if state.frame_index % args.every != 0:
            return
        if not state.detections:
            print(f"f{state.frame_index:05d}  탐지 없음")
            return
        for d in state.detections:
            x1, y1, x2, y2 = d.bbox
            px, py = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            mx, my = warp_point((px, py), H)
            node = position_to_node((mx, my))
            slot, dist = nearest_slot((mx, my))
            hr = heading.update(d.track_id or 0, (mx, my))
            hd = f"{hr.heading_deg:.0f}°({hr.source})" if hr.heading_deg is not None else "-"
            print(f"f{state.frame_index:05d}  px=({px:.0f},{py:.0f})  "
                  f"mm=({mx:6.1f},{my:6.1f})  node={node or '-':10s} "
                  f"nearest_slot={slot}({dist:.0f}mm)  heading={hd}")

    tracker = RCCarTracker(source=camera, detector=detector, max_fps=15.0)
    print("Ctrl+C 로 종료\n")
    try:
        tracker.run(on_frame=on_frame)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
