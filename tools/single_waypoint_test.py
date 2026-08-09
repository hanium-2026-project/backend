"""단일 waypoint 실차 추종 시험 — 카메라 ↔ 제어기 ↔ 실제 ESP32 폐루프.

전체 파이프라인(RL 슬롯 배정 → 7단계 waypoint → 주차 판정)을 빼고,
**목표점 하나**만 두고 차가 실제로 그쪽으로 가는지만 본다. 실차 첫 주행에서
변수를 줄이기 위한 도구다.

    카메라 → YOLO → homography(mm) → heading
        → AutoHostRunner(waypoint 1개) → DIRECT_CONTROL → ESP32

목표점은 **화면을 클릭**해서 정한다. 클릭 전에는 아무 명령도 나가지 않는다.

실행 예::

    python tools/single_waypoint_test.py --weights ~/Downloads/best.pt \\
        --camera 0 --lot 1200x1200 --max-throttle 0.25

키:
    마우스 좌클릭 : 목표점 지정 (다시 클릭하면 목표 변경)
    space        : 즉시 정지 + 목표 해제
    q            : 종료

안전:
    - 목표 미지정 / 차량 미탐지 / heading 없음 → throttle 0
    - --max-throttle 로 상한을 걸고 시작할 것 (기본 0.25)
    - ESP32 는 500ms 안에 제어값이 안 오면 스스로 safeStop 한다
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from comm import VehicleServer                                    # noqa: E402
from control import VehicleLimits                                 # noqa: E402
from control.auto_host_runner import AutoHostRunner, ModeHandshakeError  # noqa: E402
from cv.association import associate                              # noqa: E402
from cv.heading import HeadingEstimator                           # noqa: E402
from cv.homography import compute_homography, warp_point          # noqa: E402
from cv.tracker import RCCarTracker, TrackState                   # noqa: E402
from cv.vehicle_detector import YoloVehicleDetector               # noqa: E402
from parking.waypoints import Waypoint                            # noqa: E402

log = logging.getLogger("single_wp")


class SingleWaypointTest:
    def __init__(self, args) -> None:
        self.args = args
        self.lot_w, self.lot_h = args.lot
        self.homography = None
        self.inv = None
        self.target_mm: tuple[float, float] | None = None
        self.pending_target: tuple[float, float] | None = None
        self.heading = HeadingEstimator(min_move=args.heading_min_move)
        self.pose_mm: tuple[float, float] | None = None
        self.heading_deg: float | None = None
        self.heading_source: str | None = None
        self.runner: AutoHostRunner | None = None
        self.armed = False
        self._lock = threading.Lock()

        limits = VehicleLimits(max_throttle=args.max_throttle,
                               steering_sign=args.steering_sign)
        self.limits = limits

        self.server = VehicleServer(port=args.port, known_car_ids={args.car})
        self.server.on_ready = self._on_ready
        self.server.direct_control_enabled = True
        self.server.start()
        print(f"차량 서버 :{self.server.bound_port} — ESP32 접속 대기")

    # ─── 통신 ────────────────────────────────────────────────────────────────

    def _on_ready(self, car_id: int) -> None:
        print(f"\n*** car {car_id} 접속 — 목표점을 화면에서 클릭하세요 ***\n")

    def _arm(self, target_mm: tuple[float, float]) -> bool:
        """목표 waypoint 하나로 AutoHostRunner 를 띄운다."""
        wp = Waypoint(
            route_id=1, waypoint_id=1, phase="CRUISE",
            x=target_mm[0], y=target_mm[1],
            target_heading_deg=None,
            speed_cm_s=self.args.speed,
            position_tolerance_cm=self.args.tolerance,
            heading_tolerance_deg=180.0,
            heading_required=False,
            is_final=True,
        )
        if self.runner is not None:
            self.runner.load_route([wp])
            print(f"목표 변경 → ({target_mm[0]:.0f}, {target_mm[1]:.0f})mm")
            return True
        try:
            runner = AutoHostRunner(self.server, self.args.car, [wp], period_s=0.1)
            runner.start(wait_s=3.0)
        except (ModeHandshakeError, RuntimeError) as exc:
            print(f"✗ AUTO_HOST 시작 실패: {exc}")
            return False
        self.runner = runner
        self.armed = True
        print(f"✓ AUTO_HOST 무장 — 목표 ({target_mm[0]:.0f}, {target_mm[1]:.0f})mm, "
              f"throttle 상한 {self.limits.max_throttle}")
        return True

    def stop_drive(self) -> None:
        with self._lock:
            self.target_mm = None
        if self.runner is not None:
            self.runner.stop()
            self.runner = None
            self.armed = False
        self.server.stop_control(self.args.car)
        print("■ 정지 — 목표 해제")

    # ─── 프레임 ──────────────────────────────────────────────────────────────

    def on_frame(self, state: TrackState) -> None:
        if self.homography is None:
            w, h = state.frame_size
            src = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
            dst = [(0.0, self.lot_h), (self.lot_w, self.lot_h), (self.lot_w, 0.0), (0.0, 0.0)]
            if self.args.calibration:
                data = json.loads(Path(self.args.calibration).read_text())
                src = [tuple(p) for p in data["homography_src"]]
                self.lot_w = data["lot_width_mm"]
                self.lot_h = data["lot_height_mm"]
                dst = [(0.0, self.lot_h), (self.lot_w, self.lot_h),
                       (self.lot_w, 0.0), (0.0, 0.0)]
            self.homography = compute_homography(src, dst)
            import numpy as np
            self.inv = np.linalg.inv(np.asarray(self.homography, dtype=float))
            print(f"homography 준비 (프레임 {w}x{h} → {self.lot_w:.0f}x{self.lot_h:.0f}mm)")

        prev = {0: self.heading_deg} if self.heading_deg is not None else {}
        pairs, unpaired = associate(state.detections, prev,
                                    image_heading_of=self._heading_from_px)
        det = pairs[0].car if pairs else (unpaired[0] if unpaired else None)
        front_px = pairs[0].cushion_center_px if pairs else None
        if det is None:
            self.pose_mm = None
            return

        x1, y1, x2, y2 = det.bbox
        mx, my = warp_point(((x1 + x2) / 2.0, (y1 + y2) / 2.0), self.homography)
        self.pose_mm = (mx, my)
        front_mm = warp_point(front_px, self.homography) if front_px else None
        hr = self.heading.update(0, (mx, my), front_point=front_mm)
        self.heading_deg, self.heading_source = hr.heading_deg, hr.source

        # 클릭으로 들어온 목표를 프레임 스레드에서 반영 (스레드 안전)
        with self._lock:
            pending = self.pending_target
            self.pending_target = None
        if pending is not None:
            self.target_mm = pending
            self._arm(pending)

        if self.runner is not None:
            self.runner.on_camera_pose(mx, my, self.heading_deg, state.timestamp)

    def _heading_from_px(self, car_px, front_px):
        if self.homography is None:
            return None
        cx, cy = warp_point(car_px, self.homography)
        fx, fy = warp_point(front_px, self.homography)
        if math.hypot(fx - cx, fy - cy) < 1e-6:
            return None
        return math.degrees(math.atan2(fy - cy, fx - cx)) % 360.0

    # ─── 화면 ────────────────────────────────────────────────────────────────

    def to_px(self, mm):
        import numpy as np
        v = self.inv @ np.array([mm[0], mm[1], 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return int(v[0] / v[2]), int(v[1] / v[2])

    def on_mouse(self, event, x, y, flags, param):
        import cv2
        if event != cv2.EVENT_LBUTTONDOWN or self.homography is None:
            return
        mm = warp_point((float(x), float(y)), self.homography)
        with self._lock:
            self.pending_target = mm

    def overlay(self, image, state: TrackState):
        import cv2
        if self.homography is None:
            return image

        y = 30
        def line(text, color=(255, 255, 255)):
            nonlocal y
            cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 26

        if self.pose_mm:
            line(f"pose ({self.pose_mm[0]:.0f}, {self.pose_mm[1]:.0f})mm  "
                 f"heading {self.heading_deg:.0f}deg [{self.heading_source}]"
                 if self.heading_deg is not None else
                 f"pose ({self.pose_mm[0]:.0f}, {self.pose_mm[1]:.0f})mm  heading -")
        else:
            line("차량 미탐지", (0, 0, 255))

        if self.target_mm is None:
            line("목표 미지정 — 화면을 클릭하세요", (0, 200, 255))
        else:
            pt = self.to_px(self.target_mm)
            if pt:
                r = self.to_px((self.target_mm[0] + self.args.tolerance * 10.0,
                                self.target_mm[1]))
                radius = abs(r[0] - pt[0]) if r else 20
                cv2.circle(image, pt, max(radius, 10), (0, 200, 255), 2)
                cv2.drawMarker(image, pt, (0, 200, 255), cv2.MARKER_CROSS, 20, 2)
            if self.pose_mm:
                d = math.hypot(self.target_mm[0] - self.pose_mm[0],
                               self.target_mm[1] - self.pose_mm[1]) / 10.0
                line(f"목표까지 {d:.1f}cm")

        ctrl = self.server.sessions.get(self.args.car)
        if ctrl is not None and ctrl.latest_control:
            lc = ctrl.latest_control
            thr, st = lc.get("throttle", 0.0), lc.get("steering", 0.0)
            color = (0, 255, 120) if thr > 0 else (160, 160, 160)
            line(f"thr {thr:+.2f}  steering {st:+.2f} (wire)", color)
            if self.pose_mm and self.heading_deg is not None:
                origin = self.to_px(self.pose_mm)
                if origin:
                    logical = st * self.limits.steering_sign
                    ang = math.radians(self.heading_deg
                                       + logical * self.limits.max_steer_deg)
                    tip = (int(origin[0] + 80 * math.cos(ang)),
                           int(origin[1] - 80 * math.sin(ang)))
                    cv2.arrowedLine(image, origin, tip, color, 3, tipLength=0.3)
        if self.runner is not None and self.runner.is_faulted:
            line(f"FAULTED: {self.runner.host.authority.fault_reason} "
                 f"(space 로 해제 후 재클릭)", (0, 0, 255))
        return image

    # ─── 실행 ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        import cv2
        detector = YoloVehicleDetector(
            weights_path=self.args.weights,
            confidence_threshold=self.args.conf,
            imgsz=self.args.imgsz,
            custom_model=True,
        )
        tracker = RCCarTracker(source=self.args.camera, detector=detector,
                               max_fps=self.args.max_fps)
        tracker.overlay = self.overlay

        def frame_cb(state: TrackState):
            self.on_frame(state)
            # 창이 뜬 뒤 마우스 콜백을 건다 (한 번만)
            if not getattr(self, "_mouse_set", False):
                try:
                    cv2.setMouseCallback("RC Car Tracker", self.on_mouse)
                    self._mouse_set = True
                except cv2.error:
                    pass
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                self.stop_drive()

        try:
            tracker.run(on_frame=frame_cb, show=True)
        finally:
            self.stop_drive()
            self.server.stop()
            print("종료")


def parse_lot(s: str) -> tuple[float, float]:
    w, h = s.lower().split("x")
    return float(w), float(h)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--camera", default="0")
    p.add_argument("--calibration", default=None,
                   help="캘리브레이션 JSON. 없으면 프레임 전체를 --lot 크기로 간주")
    p.add_argument("--lot", type=parse_lot, default=(1200.0, 1200.0),
                   help="촬영 영역 실측 크기 WxH (mm). 예: 1500x1000")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--car", type=int, default=1)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--max-fps", type=float, default=30.0)
    p.add_argument("--speed", type=float, default=8.0, help="목표 속도 cm/s")
    p.add_argument("--tolerance", type=float, default=8.0, help="도착 허용오차 cm")
    p.add_argument("--max-throttle", type=float, default=0.25)
    p.add_argument("--steering-sign", type=float, default=-1.0, choices=[1.0, -1.0])
    p.add_argument("--heading-min-move", type=float, default=30.0)
    args = p.parse_args()
    if isinstance(args.camera, str) and args.camera.isdigit():
        args.camera = int(args.camera)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    SingleWaypointTest(args).run()


if __name__ == "__main__":
    main()
