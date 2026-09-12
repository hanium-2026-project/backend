"""캘리브레이션·A/B 배치·2클래스(쿠션) 검증용 — RL·통신 없이 좌표만 콘솔에 찍는다.

ESP32 연결이나 RL 정책과 무관하게, 탐지→쿠션 매칭→homography→heading 만 돌려서
"차량을 여기 놓으면 이 좌표가 나와야 한다"를 즉시 대조할 수 있게 한다.

2클래스 가중치(rc_car+front_cushion)를 넣으면 pipeline/runner.py 와 동일하게
cv.association 으로 쿠션을 차량과 짝지어 FRONT_CUSHION heading 을 계산한다.
1클래스 가중치를 넣으면 쿠션이 아예 안 나오므로 자동으로 TRAJECTORY 로만 동작한다.

사용법::

    python tools/debug_pose.py --camera 0 --weights best.pt \
        --calibration calibration.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv.association import associate
from cv.heading import HeadingEstimator
from cv.homography import compute_homography, warp_point
from cv.tracker import RCCarTracker
from cv.vehicle_detector import LABEL_CUSHION, YoloVehicleDetector
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
    prev_headings: dict[int, float] = {}

    def to_map(px_pt: tuple[float, float]) -> tuple[float, float]:
        return warp_point(px_pt, H)

    def heading_from_pixels(car_px, front_px):
        cx, cy = to_map(car_px)
        fx, fy = to_map(front_px)
        import math
        if math.hypot(fx - cx, fy - cy) < 1e-6:
            return None
        return math.degrees(math.atan2(fy - cy, fx - cx)) % 360.0

    cushion_seen_total = 0

    def on_frame(state):
        nonlocal cushion_seen_total
        cushions = [d for d in state.detections if d.label == LABEL_CUSHION]
        cushion_seen_total += len(cushions)

        # detect_and_track 은 뒤섞인 상태로 나오므로 파이프라인과 동일하게 매칭한다
        pairs, unpaired = associate(state.detections, prev_headings,
                                    image_heading_of=heading_from_pixels)

        if state.frame_index % args.every != 0:
            return
        if not state.detections:
            print(f"f{state.frame_index:05d}  탐지 없음")
            return

        rows = []
        for pair in pairs:
            rows.append((pair.car, pair.cushion_center_px))
        for det in unpaired:
            rows.append((det, None))

        for det, front_px in rows:
            x1, y1, x2, y2 = det.bbox
            px, py = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            mx, my = to_map((px, py))
            node = position_to_node((mx, my))
            slot, dist = nearest_slot((mx, my))
            front_mm = to_map(front_px) if front_px else None
            hr = heading.update(det.track_id or 0, (mx, my), front_point=front_mm)
            if det.track_id is not None and hr.heading_deg is not None:
                prev_headings[det.track_id] = hr.heading_deg
            hd = f"{hr.heading_deg:.0f}°({hr.source})" if hr.heading_deg is not None else "-"
            cushion_tag = " [+cushion]" if front_px else ""
            print(f"f{state.frame_index:05d}  px=({px:.0f},{py:.0f})  "
                  f"mm=({mx:6.1f},{my:6.1f})  node={node or '-':10s} "
                  f"nearest_slot={slot}({dist:.0f}mm)  heading={hd}{cushion_tag}")

    tracker = RCCarTracker(source=camera, detector=detector, max_fps=15.0)
    print("Ctrl+C 로 종료\n")
    try:
        tracker.run(on_frame=on_frame)
    except KeyboardInterrupt:
        pass
    finally:
        tag = "OK — 2클래스 모델이 쿠션을 잡고 있습니다" if cushion_seen_total > 0 \
              else "쿠션 탐지 0건 — 1클래스 가중치이거나 쿠션이 안 보이는 상태입니다"
        print(f"\n쿠션 탐지 누적: {cushion_seen_total}회 ({tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
