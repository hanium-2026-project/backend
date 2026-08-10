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
from controller.config import ControllerConfig                    # noqa: E402
from control.auto_host_runner import AutoHostRunner, ModeHandshakeError  # noqa: E402
from cv.association import associate                              # noqa: E402
from cv.heading import HeadingEstimator                           # noqa: E402
from cv.homography import compute_homography, warp_point          # noqa: E402
from cv.tracker import RCCarTracker, TrackState                   # noqa: E402
from cv.vehicle_detector import YoloVehicleDetector               # noqa: E402
from parking.waypoints import Waypoint                            # noqa: E402
from tools.drive_logger import DriveLogger                        # noqa: E402

log = logging.getLogger("single_wp")

WINDOW = "RC Car Tracker"        # cv.tracker 가 쓰는 창 이름과 같아야 한다


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
        self._last_arm_try = 0.0
        self._last_ctrl_log = 0.0
        self._peak_steer = 0.0          # 이번 주행의 최대 |조향| — 포화 여부 판단용
        self._prev_obs = None           # (x, y, heading, t) — 속도·선회반경 추정용
        self._radius_samples: list[float] = []
        self._last_speed = 0.0
        self._lock = threading.Lock()

        # 실제 제어기(zip PoseWaypointController)가 쓰는 설정.
        # VehicleLimits 는 화면 화살표 표시용일 뿐이라 여기 값이 진짜다.
        self.ctrl_config = ControllerConfig(
            max_throttle=args.max_throttle,
            wire_steering_sign=args.steering_sign,
            steer_kp=args.steer_kp,
            steer_normalize_deg=args.steer_normalize,
            max_wire_steering=args.max_steering,
        )
        self.limits = VehicleLimits(max_throttle=args.max_throttle,
                                    steering_sign=args.steering_sign,
                                    max_steer_deg=args.steer_normalize)

        self.server = VehicleServer(port=args.port, known_car_ids={args.car})
        self.server.on_ready = self._on_ready
        self.server.direct_control_enabled = True
        self.server.start()
        print(f"차량 서버 :{self.server.bound_port} — ESP32 접속 대기")

        self.logger: DriveLogger | None = None
        if args.log:
            self.logger = DriveLogger(
                args.log, self.server, car_id=args.car,
                pose_provider=lambda: (
                    (self.pose_mm[0], self.pose_mm[1],
                     self.heading_deg, self.heading_source or "")
                    if self.pose_mm else None),
                target_provider=lambda: self.target_mm,
            )
            self.logger.start()
            print(f"주행 로그: {args.log}")

    # ─── 통신 ────────────────────────────────────────────────────────────────

    def _on_ready(self, car_id: int) -> None:
        print(f"\n*** car {car_id} 접속 — 목표점을 화면에서 클릭하세요 ***\n")

    def _arm(self, target_mm: tuple[float, float]) -> bool:
        """목표 waypoint 하나로 AUTO_HOST 를 무장한다.

        러너는 한 번만 만들고 재사용한다. 실패할 때마다 새로 만들면
        RemoteDirectSession 이 서버 콜백에 계속 덧붙어 누적된다.
        """
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
        if self.runner is None:
            self.runner = AutoHostRunner(self.server, self.args.car, [wp],
                                         period_s=0.1, config=self.ctrl_config)
            self.runner.on_status_change = self._on_mission_status
        else:
            self.runner.load_route([wp])

        if self.armed and not self.runner.is_faulted:
            print(f"목표 변경 → ({target_mm[0]:.0f}, {target_mm[1]:.0f})mm")
            return True

        try:
            if self.runner.is_faulted:
                self.runner.re_arm(wait_s=3.0)
            else:
                self.runner.start(wait_s=3.0)
        except (ModeHandshakeError, RuntimeError) as exc:
            print(f"✗ AUTO_HOST 무장 실패: {exc}")
            self.armed = False
            return False
        self._peak_steer = 0.0
        self.armed = True
        self.runner.scheduler.start()   # 이미 돌고 있으면 무시된다
        print(f"✓ AUTO_HOST 무장 — 목표 ({target_mm[0]:.0f}, {target_mm[1]:.0f})mm, "
              f"throttle 상한 {self.limits.max_throttle}")
        return True

    def _radius_summary(self) -> str:
        """포화 조향으로 돈 구간의 반경 중앙값 = 최소 선회 반경 추정."""
        rs = sorted(self._radius_samples)
        if len(rs) < 3:
            return ""
        return f"  최소선회반경 ~{rs[len(rs)//2]:.0f}cm ({len(rs)}표본)"

    def _on_mission_status(self, car_id, prev, status) -> None:
        if status.value == "DONE":
            err = ""
            if self.pose_mm and self.target_mm:
                err = " 최종오차 %.1fcm" % (math.hypot(
                    self.target_mm[0] - self.pose_mm[0],
                    self.target_mm[1] - self.pose_mm[1]) / 10.0)
            print(f"   ▶ 도착{err}  (최대 조향 {self._peak_steer:.2f}"
                  f"{' = 포화' if self._peak_steer >= 0.999 else ''}){self._radius_summary()}")

    def stop_drive(self) -> None:
        with self._lock:
            self.target_mm = None
        if self.runner is not None:
            self.runner.stop()          # FAULTED latch + zero. 객체는 재사용한다
            self.armed = False
        self.server.stop_control(self.args.car)
        print(f"■ 정지 — 목표 해제 (최대 조향 {self._peak_steer:.2f}"
              f"{' = 포화' if self._peak_steer >= 0.999 else ''}){self._radius_summary()}")
        self._radius_samples.clear()

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

        # 클릭 반영은 차량 탐지보다 먼저 한다 — 미탐지 상태에서 찍어둘 수 있어야 한다
        with self._lock:
            pending = self.pending_target
            self.pending_target = None
        if pending is not None:
            self.target_mm = pending
            print(f"목표 지정 ({pending[0]:.0f}, {pending[1]:.0f})mm")
            self._arm(pending)

        # 접속 전에 목표를 찍어둔 경우: 세션이 생기면 자동으로 무장한다.
        # (차량 접속보다 클릭이 빠른 경우가 흔하다)
        if (self.target_mm is not None and not self.armed
                and self.args.car in self.server.sessions
                and time.monotonic() - self._last_arm_try > 3.0):
            self._last_arm_try = time.monotonic()
            self._arm(self.target_mm)

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

        if self.runner is not None:
            self.runner.on_camera_pose(mx, my, self.heading_deg, state.timestamp)
            self._log_control()

    def _update_motion(self, x: float, y: float, t: float) -> tuple[float, float | None]:
        """실측 속도(cm/s)와 선회 반경(cm)을 pose 이력에서 추정한다.

        R = v / ω. 조향을 최대로 걸고 도는 중에 나오는 값이 곧 최소 선회 반경이다.
        이 값이 목표까지 거리보다 크면 차는 목표를 영원히 못 잡고 주위를 돈다.
        """
        prev, self._prev_obs = self._prev_obs, (x, y, self.heading_deg, t)
        if prev is None or self.heading_deg is None or prev[2] is None:
            return 0.0, None
        dt = t - prev[3]
        if dt < 0.40:
            # 창이 짧으면 heading 노이즈가 반경 추정을 크게 흔든다 (39~94cm 편차).
            # 0.4초 이상 모아서 계산한다.
            self._prev_obs = prev
            return self._last_speed, None
        speed = math.hypot(x - prev[0], y - prev[1]) / 10.0 / dt        # cm/s
        self._last_speed = speed
        dtheta = abs(math.radians((self.heading_deg - prev[2] + 180) % 360 - 180)) / dt
        if dtheta < math.radians(3.0) or speed < 1.0:
            return speed, None               # 거의 직진이면 반경 무의미
        return speed, speed / dtheta

    def _log_control(self) -> None:
        """주행 중 제어값을 터미널에도 주기적으로 남긴다 (화면만 보면 읽기 어렵다)."""
        sess = self.server.sessions.get(self.args.car)
        if sess is None or not sess.latest_control:
            return
        thr = float(sess.latest_control.get("throttle", 0.0))
        st = float(sess.latest_control.get("steering", 0.0))
        self._peak_steer = max(self._peak_steer, abs(st))
        now = time.monotonic()
        speed, radius = (0.0, None)
        if self.pose_mm:
            speed, radius = self._update_motion(self.pose_mm[0], self.pose_mm[1], now)
        if radius is not None and abs(st) >= 0.9 and thr > 0.0:
            self._radius_samples.append(radius)      # 포화 상태의 반경 = 최소 선회 반경
        if thr <= 0.0 or now - self._last_ctrl_log < 0.5:
            return
        self._last_ctrl_log = now
        motion = f"  {speed:4.1f}cm/s"
        if radius is not None:
            motion += f"  R={radius:5.1f}cm"
        dist = ""
        if self.pose_mm and self.target_mm:
            dist = " dist %5.1fcm" % (math.hypot(
                self.target_mm[0] - self.pose_mm[0],
                self.target_mm[1] - self.pose_mm[1]) / 10.0)
        sat = "  ★포화" if abs(st) >= 0.999 else ""
        head = f"  {self.heading_deg:.0f}deg" if self.heading_deg is not None else ""
        print(f"   thr {thr:+.2f}  str {st:+.3f}{dist}{head}{motion}{sat}")

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

        connected = self.args.car in self.server.sessions
        line(f"ESP32 {'접속됨' if connected else '미접속 — 대기 중'}",
             (0, 255, 120) if connected else (0, 0, 255))

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

        # 창을 먼저 만들어 두고 콜백을 건다. tracker 의 imshow 는 같은 이름의
        # 창을 재사용한다. 프레임 콜백 안에서 걸면 첫 프레임에는 창이 없어
        # 조용히 실패하고, 이후 다시 걸지 않아 클릭이 영영 안 먹는다.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        def frame_cb(state: TrackState):
            self.on_frame(state)
            if (cv2.waitKey(1) & 0xFF) == ord(" "):
                self.stop_drive()

        try:
            tracker.run(on_frame=frame_cb, show=True)
        finally:
            self.stop_drive()
            if self.logger is not None:
                self.logger.stop()
                print(f"로그 {self.logger.rows}행 기록: {self.args.log}")
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
    p.add_argument("--log", default=None,
                   help="주행 로그 CSV 경로 (예: logs/run1.csv)")
    p.add_argument("--steer-kp", type=float, default=1.6,
                   help="조향 비례 게인 (크면 급하게 꺾는다)")
    p.add_argument("--steer-normalize", type=float, default=30.0,
                   help="이 각도 오차에서 조향이 최대로 포화한다")
    p.add_argument("--max-steering", type=float, default=1.0,
                   help="조향 절댓값 상한. 0.5 로 두면 펌웨어 강회전 PWM(55) "
                        "구간을 피해 속도 제어가 살아난다")
    args = p.parse_args()
    if isinstance(args.camera, str) and args.camera.isdigit():
        args.camera = int(args.camera)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    SingleWaypointTest(args).run()


if __name__ == "__main__":
    main()
