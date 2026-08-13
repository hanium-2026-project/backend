"""CV → RL → 통신 통합 파이프라인.

지금까지 개별 검증한 모듈을 하나의 상시 루프로 조립한다.

    카메라 프레임
      → YOLO 탐지·추적 (track_id)
      → homography (픽셀 → mm) + bbox offset 보정
      → heading 추정 (궤적 기반)
      → track_id ↔ car_id 매핑 (순차 진입)
      → 신규 차량이면 RL 슬롯 배정 → waypoint 생성 → 미션 시작
      → 도착 판정 (노트북 담당) → WAIT → 다음 WAYPOINT → GO
      → 충돌 감지 → hold / 경로 재생성
      → POSE_UPDATE 스트림 (펌웨어 지원 시)

실행::

    from pipeline import ParkingPipeline, PipelineConfig
    p = ParkingPipeline(PipelineConfig(camera_source=0,
                                       weights_path="best05.pt"))
    p.start()          # TCP 서버 기동 (ESP32 접속 대기)
    p.run_camera()     # 카메라 루프 (블로킹)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from comm import MissionOrchestrator, MissionState, VehicleServer
from control import ControlOutput, Pose, WaypointController
from control.auto_host_runner import AutoHostRunner, MissionStatus, ModeHandshakeError
from control.hybrid_control import HybridControlMux
from cv.association import associate
from cv.heading import HeadingEstimator
from cv.homography import compute_homography, warp_point
from cv.tracker import RCCarTracker, TrackState
from cv.vehicle_detector import YoloVehicleDetector
from parking.recovery import (REVERSE_TRIGGER_REASONS, forward_unreachable,
                              plan_reverse_recovery)
from parking.safety import CollisionMonitor, VehiclePose
from parking.waypoints import (AISLE_Y, MIN_TURN_RADIUS_MM, InfeasibleRouteError,
                               build_waypoints, default_slot_specs, plan_handoff)
from rl.bridge import RealtimeAllocator, position_to_node
from rl.parking_env import SLOT_NAMES

from .config import PipelineConfig
from .dashboard import DashboardBridge

log = logging.getLogger(__name__)


@dataclass
class VehicleView:
    """카메라가 보는 차량 1대의 최신 관측값."""

    track_id: int
    car_id: int | None = None
    position_mm: tuple[float, float] = (0.0, 0.0)
    heading_deg: float | None = None
    heading_source: str | None = None
    confidence: float = 0.0
    node: str | None = None
    slot_id: str | None = None
    last_seen_frame: int = 0
    # 관측이 발생한 monotonic 시각. tick 시각을 쓰면 카메라가 멈춰도 pose 가
    # 신선해 보여서 stale 판정이 무력화된다 — 반드시 프레임 시각을 넣는다.
    last_obs_time: float = 0.0
    last_alloc_frame: int = -10_000        # 슬롯 배정 재시도 조절용
    recent: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))
    # 로그용 원본 픽셀값 (요청문 6절). homography 검증에 필요해 변환 전 값을 남긴다.
    last_pixel: tuple[float, float] | None = None
    last_bbox: tuple[int, int, int, int] | None = None

    def is_stationary(self, tolerance_mm: float, window: int) -> bool:
        """최근 창 안에서 거의 움직이지 않았는지 (PARKED 재검증용, §11)."""
        pts = list(self.recent)[-window:]
        if len(pts) < window:
            return False
        x0, y0 = pts[0]
        return all(math.hypot(x - x0, y - y0) <= tolerance_mm for x, y in pts[1:])


class ParkingPipeline:
    """카메라 한 대 + 차량 N대를 묶는 최상위 실행기."""

    def __init__(self, config: PipelineConfig | None = None,
                 detector: YoloVehicleDetector | None = None) -> None:
        self.config = config or PipelineConfig()
        self._detector = detector
        self._homography = None

        self.server = VehicleServer(
            host=self.config.server_host,
            port=self.config.server_port,
            known_car_ids=set(self.config.known_car_ids),
        )
        self.orchestrator = MissionOrchestrator(
            self.server,
            camera_lead_cm=self.config.camera_lead_cm,
            on_parked=self._on_parked,
        )
        self.allocator = RealtimeAllocator(model_path=self.config.policy_path)
        self.heading = HeadingEstimator(min_move=self.config.heading_min_move_mm)
        self.collision = CollisionMonitor()

        self.dashboard = DashboardBridge(pose_interval_s=self.config.dashboard_pose_interval_s)
        self._collision_held: set[int] = set()           # 충돌로 정지시킨 차량
        self._comm_lost: set[int] = set()                # 통신 장애로 정지시킨 차량
        # B안: 차량별 주행 제어기 (throttle/steering 계산)
        self.controllers: dict[int, WaypointController] = {}
        self.last_control: dict[int, ControlOutput] = {}
        self._mode_set: set[int] = set()                 # SET_MODE 완료 차량
        self._last_control_mode: dict[int, str] = {}     # 로그 중복 억제
        # AUTO_HOST 모드: 차량별 제어 소유자 (waypoint-auto 에서는 비어 있다)
        self.auto_hosts: dict[int, AutoHostRunner] = {}
        self._auto_host_slot: dict[int, str] = {}        # car_id → 배정 슬롯
        self._replan_attempts: dict[int, int] = {}       # car_id → 재계획 횟수
        self._deviation_streak: dict[int, int] = {}      # car_id → 연속 이탈 프레임
        self._last_no_route_warn = 0.0                   # 경로 불가 경고 간격 제한
        self._unreachable_slots: set[str] = set()        # 현재 위치에서 못 가는 슬롯
        self._auto_host_route: dict[int, list[Any]] = {}  # car_id → 화면 표시용 경로
        # 경로 계획 반경. 실측값(610mm)이 기본이고, 진입 우회전 실험 때만 낮춘다.
        self._plan_radius = (self.config.plan_turn_radius_mm
                             if self.config.plan_turn_radius_mm is not None
                             else MIN_TURN_RADIUS_MM)
        # 수동 WASD ↔ 자동 전환 mux (하드웨어팀 통합본). auto-host 모드에서만 쓴다.
        self.hybrid_controls: dict[int, HybridControlMux] = {}
        self._manual_shell_starting: set[int] = set()
        self.views: dict[int, VehicleView] = {}          # track_id → 관측
        self.track_of_car: dict[int, int] = {}           # car_id → track_id
        self._pending_cars: list[int] = []               # HELLO 순서 대기열
        self._lock = threading.Lock()
        self._tracker: RCCarTracker | None = None

        # ─ Run 기록기 연결점 (tools/run_recorder.py) ─
        # 파이프라인은 기록기를 알지 못한다. 프레임마다 pose 레코드를 만들어
        # last_pose_rec 에 두고, 콜백이 붙어 있으면 밀어준다. 기본은 꺼짐.
        self.on_pose_record: Callable[[dict], None] | None = None
        # (waypoints, is_recovery) — route.json / recovery_route.json 용
        self.on_route_load: Callable[[list, bool], None] | None = None
        self.last_pose_rec: dict | None = None
        self._frame_seq = 0
        self._prev_frame_index: int | None = None
        self._dropped_frames = 0

        self.server.on_status = self.orchestrator.on_vehicle_status
        self.server.on_command_rejected = self.orchestrator.on_command_rejected
        self.server.on_ready = self._on_vehicle_ready
        self.server.on_resync = self._on_resync
        self.server.hold_check = self._hold_check
        self.server.on_comm_fail = self._on_comm_fail
        self.server.on_comm_recovered = self._on_comm_recovered
        self.orchestrator.on_replan_required = self._on_replan_required
        self.server.direct_control_enabled = self.config.direct_control

    # ─── 라이프사이클 ────────────────────────────────────────────────────────

    def start(self) -> None:
        """TCP 서버를 띄우고 ESP32 접속을 받는다 (논블로킹)."""
        self.server.start()
        log.info("vehicle server listening on %s:%d",
                 self.config.server_host, self.server.bound_port)

    def stop(self) -> None:
        # 제어 루프를 먼저 세운다. 서버보다 나중에 멈추면 그 사이에 마지막
        # 제어값이 한 번 더 나갈 수 있다.
        for mux in list(self.hybrid_controls.values()):
            try:
                mux.stop()
            except Exception:                      # noqa: BLE001
                pass
        self.hybrid_controls.clear()
        for runner in list(self.auto_hosts.values()):
            runner.stop()
        self.auto_hosts.clear()
        if self._tracker is not None:
            self._tracker.stop()
        self.server.stop()

    def run_camera(self, max_frames: int | None = None, show: bool = False) -> None:
        """카메라 루프를 돈다 (블로킹). 프레임마다 on_frame 처리."""
        if self._detector is None:
            self._detector = YoloVehicleDetector(
                weights_path=self.config.weights_path,
                confidence_threshold=self.config.confidence_threshold,
                imgsz=self.config.imgsz,
                custom_model=self.config.custom_model,
            )
        self._tracker = RCCarTracker(
            source=self.config.camera_source,
            detector=self._detector,
            max_fps=self.config.max_fps,
        )
        if show:
            self._tracker.overlay = self._draw_targets
        self._tracker.run(on_frame=self.on_frame, max_frames=max_frames, show=show)

    def _draw_targets(self, image, state: TrackState):
        """현재 목표 waypoint 를 화면에 표시한다 (show=True 일 때만).

        차를 어디로 옮겨야 하는지 눈으로 보이지 않으면 실차 검증이 어렵다.
        맵 좌표를 역투영해 목표점·허용 반경·남은 거리를 그린다.
        """
        import cv2
        import numpy as np

        if self._homography is None:
            return image
        inv = np.linalg.inv(np.asarray(self._homography, dtype=float))

        def to_px(mm: tuple[float, float]) -> tuple[int, int] | None:
            v = inv @ np.array([mm[0], mm[1], 1.0])
            if abs(v[2]) < 1e-9:
                return None
            return int(v[0] / v[2]), int(v[1] / v[2])

        y = 60
        if self.auto_host_mode:
            return self._draw_auto_host(image, to_px, y)

        for car_id, m in self.orchestrator.missions.items():
            wp = m.current
            if wp is None:
                continue
            pt = to_px((wp.x, wp.y))
            if pt is not None:
                # 허용 반경도 픽셀로 환산 (x 방향 기준 근사)
                edge = to_px((wp.x + wp.position_tolerance_cm * 10.0, wp.y))
                radius = abs(edge[0] - pt[0]) if edge else 20
                cv2.circle(image, pt, max(radius, 8), (0, 200, 255), 2)
                cv2.drawMarker(image, pt, (0, 200, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(image, f"car{car_id} wp{wp.waypoint_id} {wp.phase}",
                            (pt[0] + 12, pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 200, 255), 2)

            track_id = self.track_of_car.get(car_id)
            view = self.views.get(track_id) if track_id is not None else None
            dist = ""
            if view is not None:
                dist = f"  남은거리 {math.hypot(wp.x - view.position_mm[0], wp.y - view.position_mm[1]):.0f}mm"
            cv2.putText(image, f"car{car_id} {m.state.name} {m.slot_id or '-'} "
                               f"wp{wp.waypoint_id}/{len(m.waypoints)}{dist}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            y += 26

            out = self.last_control.get(car_id)
            if out is not None:
                # 실차 튜닝 중에는 이 줄만 보면 된다: 어디로 얼마나 틀고 미는지.
                color = (0, 255, 120) if out.throttle > 0 else (120, 120, 255)
                cv2.putText(
                    image,
                    f"   {out.mode} thr {out.throttle:+.2f} str {out.steering:+.2f} "
                    f"err {out.heading_error_deg:+.0f}deg"
                    + (f" [{out.reason}]" if out.reason else ""),
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y += 24
                if view is not None and view.heading_deg is not None:
                    # 조향 방향을 화살표로 — 부호 규약(좌=+)을 눈으로 검증한다
                    origin = to_px(view.position_mm)
                    if origin is not None:
                        # wire 부호(음수=좌)를 화면 방향(반시계 양수)으로 되돌린다
                        logical = out.steering * self.config.vehicle_limits.steering_sign
                        ang = math.radians(view.heading_deg
                                           + logical * self.config.vehicle_limits.max_steer_deg)
                        tip = (int(origin[0] + 70 * math.cos(ang)),
                               int(origin[1] - 70 * math.sin(ang)))
                        cv2.arrowedLine(image, origin, tip, color, 2, tipLength=0.3)
        return image

    def _draw_auto_host(self, image, to_px, y: int):
        """AUTO_HOST 경로를 화면에 그린다.

        기존 오버레이는 orchestrator.missions 를 봤는데 auto-host 에서는 그게
        항상 비어 있어 아무것도 안 보였다. 실차에서 "지금 어디로 가라고 하는
        중인지"가 안 보이면 검증이 불가능하다.
        """
        import cv2

        CYAN, GREEN, GREY, RED = (255, 200, 0), (80, 255, 120), (170, 170, 170), (80, 80, 255)

        # 통로선 — 인계 지점이 놓이는 기준선
        a, b = to_px((0.0, AISLE_Y)), to_px((self.config.lot_width_mm, AISLE_Y))
        if a and b:
            cv2.line(image, a, b, GREY, 1, cv2.LINE_AA)

        for car_id, runner in self.auto_hosts.items():
            route = self._auto_host_route.get(car_id) or []
            target = runner.current_target
            tx = (target.x_mm, target.y_mm) if target is not None else None

            pts = [to_px((w.x, w.y)) for w in route]
            for p, q in zip(pts, pts[1:]):
                if p and q:
                    cv2.line(image, p, q, CYAN, 1, cv2.LINE_AA)

            for w, pt in zip(route, pts):
                if pt is None:
                    continue
                cur = tx is not None and abs(w.x - tx[0]) < 1 and abs(w.y - tx[1]) < 1
                col = GREEN if cur else CYAN
                edge = to_px((w.x + w.position_tolerance_cm * 10.0, w.y))
                r = abs(edge[0] - pt[0]) if edge else 14
                cv2.circle(image, pt, max(r, 8), col, 2 if cur else 1)
                cv2.drawMarker(image, pt, col, cv2.MARKER_CROSS, 14, 2 if cur else 1)
                cv2.putText(image, f"{w.waypoint_id} {w.phase}", (pt[0] + 10, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                if w.target_heading_deg is not None:      # 인계 자세(가로) 화살표
                    ang = math.radians(w.target_heading_deg)
                    cv2.arrowedLine(image, pt,
                                    (int(pt[0] + 55 * math.cos(ang)),
                                     int(pt[1] - 55 * math.sin(ang))),
                                    col, 2, tipLength=0.3)

            slot = self._auto_host_slot.get(car_id, "-")
            idx = runner.mission.index + 1 if route else 0
            head = (f"car{car_id} {runner.status.value} slot={slot} "
                    f"wp{idx}/{len(route)}")
            track_id = self.track_of_car.get(car_id)
            view = self.views.get(track_id) if track_id is not None else None
            if view is not None and tx is not None:
                head += (f"  남은거리 "
                         f"{math.hypot(tx[0] - view.position_mm[0], tx[1] - view.position_mm[1]):.0f}mm")
            cv2.putText(image, head, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        RED if runner.is_faulted else GREEN, 2, cv2.LINE_AA)
            y += 26

            if view is not None:
                cv2.putText(image,
                            f"   pose ({view.position_mm[0]:.0f},{view.position_mm[1]:.0f})mm "
                            f"hdg {view.heading_deg if view.heading_deg is None else round(view.heading_deg)}"
                            f" [{view.heading_source}]",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
                y += 22

        for slot in sorted(self._unreachable_slots):
            cv2.putText(image, f"   슬롯 {slot} 도달불가", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
            y += 20
        return image

    # ─── 프레임 처리 (핵심) ──────────────────────────────────────────────────

    def on_frame(self, state: TrackState) -> None:
        """탐지 결과 1프레임을 좌표·판정·명령까지 흘린다."""
        t_recv = time.monotonic()
        if self._homography is None:
            self._init_homography(state)

        # 2클래스 모델이면 전방 쿠션을 차량과 짝지어 heading 정확도를 높인다.
        # 1클래스 모델에서는 쿠션이 없으므로 pairs 가 비고 기존 동작과 같아진다.
        prev_headings = {v.track_id: v.heading_deg for v in self.views.values()
                         if v.heading_deg is not None}
        pairs, unpaired = associate(state.detections, prev_headings,
                                    image_heading_of=self._heading_from_pixels)

        seen: list[VehicleView] = []
        for pair in pairs:
            if pair.car.track_id is None:
                continue
            view = self._update_view(pair.car, state.frame_index,
                                     front_px=pair.cushion_center_px,
                                     obs_time=state.timestamp)
            seen.append(view)
        for det in unpaired:
            if det.track_id is None:
                continue
            seen.append(self._update_view(det, state.frame_index,
                                          obs_time=state.timestamp))

        self._record_pose(seen, state, t_recv)

        for view in seen:
            self._ensure_mission(view, state.frame_index)
            self._push_to_vehicle(view)
            if self.auto_host_mode:
                self._check_auto_host_parked(view)

        self._check_collisions(seen)
        self._forget_stale(state.frame_index)

    def _record_pose(self, seen: list[VehicleView], state: TrackState,
                     t_recv: float) -> None:
        """이번 프레임의 pose 원본을 기록기에 넘긴다 (요청문 6절).

        car_id 가 붙은 차량 하나만 남긴다 — 기록기가 차량 1대 기준이라
        여러 track 을 섞으면 시계열이 뒤엉킨다.
        """
        self._frame_seq += 1
        if self._prev_frame_index is not None:
            gap = state.frame_index - self._prev_frame_index - 1
            if gap > 0:
                self._dropped_frames += gap
        self._prev_frame_index = state.frame_index

        view = next((v for v in seen if v.car_id is not None), None)
        if view is None:
            return
        px = view.last_pixel or (None, None)
        self.last_pose_rec = {
            "frame_id": self._frame_seq,
            "tracker_frame_index": state.frame_index,
            "capture_ts": state.timestamp,
            "pose_ts": time.monotonic(),
            "obs_time": view.last_obs_time,
            "car_id": view.car_id,
            "track_id": view.track_id,
            "pixel_x": None if px[0] is None else round(px[0], 1),
            "pixel_y": None if px[1] is None else round(px[1], 1),
            "bbox": list(view.last_bbox) if view.last_bbox else None,
            "x_mm": round(view.position_mm[0], 1),
            "y_mm": round(view.position_mm[1], 1),
            "heading_deg": (None if view.heading_deg is None
                            else round(view.heading_deg, 1)),
            "heading_source": view.heading_source,
            "node": view.node,
            "slot_id": view.slot_id,
            "valid": True,
            "confidence": view.confidence,
            "latency_ms": round((time.monotonic() - t_recv) * 1000, 2),
            "fps": round(state.fps, 1),
            "dropped_total": self._dropped_frames,
        }
        if self.on_pose_record is not None:
            self.on_pose_record(self.last_pose_rec)

    def _init_homography(self, state: TrackState) -> None:
        """첫 프레임 크기로 캘리브레이션 행렬을 만든다."""
        w, h = state.frame_size
        if not w or not h:
            raise RuntimeError("frame_size 가 비어 있어 homography 를 만들 수 없다")
        src, dst = self.config.homography_pairs(w, h)
        self._homography = compute_homography(src, dst)
        log.info("homography ready (frame %dx%d)", w, h)

    def _heading_from_pixels(self, car_px: tuple[float, float],
                             front_px: tuple[float, float]) -> float | None:
        """픽셀 두 점을 맵 좌표로 옮겨 heading 을 구한다 (§6.5).

        heading 은 반드시 실좌표에서 계산해야 한다. 픽셀에서 각도를 재면
        카메라 투영 왜곡이 그대로 각도 오차가 된다.
        """
        if self._homography is None:
            return None
        cx, cy = warp_point(car_px, self._homography)
        fx, fy = warp_point(front_px, self._homography)
        if math.hypot(fx - cx, fy - cy) < 1e-6:
            return None
        return math.degrees(math.atan2(fy - cy, fx - cx)) % 360.0

    def _update_view(self, det, frame_index: int,
                     front_px: tuple[float, float] | None = None,
                     obs_time: float | None = None) -> VehicleView:
        x1, y1, x2, y2 = det.bbox
        center_px = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        mx, my = warp_point(center_px, self._homography)
        mx += self.config.bbox_offset_mm[0]
        my += self.config.bbox_offset_mm[1]

        with self._lock:
            view = self.views.get(det.track_id)
            if view is None:
                view = VehicleView(track_id=det.track_id)
                self.views[det.track_id] = view
            view.position_mm = (mx, my)
            view.confidence = det.confidence
            view.last_seen_frame = frame_index
            view.last_obs_time = time.monotonic() if obs_time is None else obs_time
            view.last_pixel = center_px
            view.last_bbox = (int(x1), int(y1), int(x2), int(y2))
            view.recent.append((mx, my))

        front_mm = warp_point(front_px, self._homography) if front_px else None
        hr = self.heading.update(det.track_id, (mx, my), front_point=front_mm)
        view.heading_deg, view.heading_source = hr.heading_deg, hr.source
        view.node = position_to_node((mx, my))
        # RL 관측이 실제 차량 진행을 반영해야 한다. 이 갱신이 없으면 정책은
        # 모든 차량이 입구에 멈춰 있다고 보고 후속 차량에 영원히 WAIT 을 준다.
        self.allocator.update(det.track_id, (mx, my))
        return view

    # ─── track_id ↔ car_id 매핑 (순차 진입) ──────────────────────────────────

    def _on_vehicle_ready(self, car_id: int) -> None:
        """HELLO 승인 순서를 대기열에 넣는다 — 카메라 진입 순서와 대조."""
        with self._lock:
            if car_id not in self._pending_cars and car_id not in self.track_of_car:
                self._pending_cars.append(car_id)
        log.info("car %d ready (awaiting camera match)", car_id)
        if self.auto_host_mode:
            # 카메라·슬롯배정을 기다리지 않고 먼저 수동 조작을 열어둔다.
            # 이게 없으면 카메라를 안 쓰는 실행(GUI)에서 mux 가 영영 안 생겨
            # hybrid_mode() 가 UNAVAILABLE 에 머문다.
            with self._lock:
                start = (car_id not in self.auto_hosts
                         and car_id not in self._manual_shell_starting)
                if start:
                    self._manual_shell_starting.add(car_id)
            if start:
                threading.Thread(target=self._start_manual_shell, args=(car_id,),
                                 name=f"manual-shell-{car_id}", daemon=True).start()
        # AUTO_HOST 에서는 RemoteDirectSession 이 ACCEPTED 를 확인하며 협상한다.
        # 여기서 또 보내면 seq 두 개가 뜨고 ACCEPTED 매칭이 어긋난다.
        if self.config.direct_control and self.config.direct_control_set_mode \
                and not self.auto_host_mode and car_id not in self._mode_set:
            # B안에서는 ESP32 가 waypoint 를 추종하지 않고 제어값만 실행한다.
            try:
                self.server.send_set_mode(car_id, "REMOTE_DIRECT")
                self._mode_set.add(car_id)
                log.info("car %d: SET_MODE REMOTE_DIRECT (B안 제어)", car_id)
            except RuntimeError as exc:
                log.warning("car %d: SET_MODE 실패 (%s)", car_id, exc)

    def _bind_car(self, view: VehicleView) -> int | None:
        """진입 노드에 나타난 미매핑 track 을 대기열 앞쪽 car_id 와 연결한다."""
        with self._lock:
            if view.car_id is not None:
                return view.car_id
            if not self._pending_cars:
                return None
            car_id = self._pending_cars.pop(0)
            view.car_id = car_id
            self.track_of_car[car_id] = view.track_id
        log.info("bound track %d ↔ car %d", view.track_id, car_id)
        return car_id

    # ─── 미션 시작 / 진행 ────────────────────────────────────────────────────

    def _ensure_mission(self, view: VehicleView, frame_index: int) -> None:
        """진입 노드의 신규 차량에 슬롯을 배정하고 주행을 시작한다.

        RL 정책은 혼잡할 때 의도적으로 WAIT(슬롯 미배정)을 반환한다. 이 경우
        차량을 방치하면 안 되므로, 진입 노드에 머무는 동안 주기적으로 재시도한다.
        """
        if view.slot_id is not None:
            return                       # 이미 배정 완료
        if view.node not in self.config.entry_nodes:
            return
        car_id = view.car_id if view.car_id is not None else self._bind_car(view)
        if car_id is None:
            return                       # 아직 접속한 차량이 없음
        if self.config.manual_only:
            return                       # 수동 계측 — 매핑만 하고 자동 주행 안 함
        if car_id in self.orchestrator.missions:
            return                       # 미션 진행 중
        if car_id in self._manual_shell_starting:
            # 수동 셸이 백그라운드로 REMOTE_DIRECT 협상 중이다. 지금 미션을
            # 시작하면 러너가 두 개 생겨 같은 제어 스트림을 다투게 된다.
            return                       # 다음 프레임에 재시도
        if frame_index - view.last_alloc_frame < self.config.alloc_retry_frames:
            return                       # 재시도 주기 대기
        view.last_alloc_frame = frame_index

        self.allocator.update(view.track_id, view.position_mm)
        slot_id = self.allocator.allocate(view.track_id)
        if slot_id is None:
            log.info("car %d: no slot available (RL WAIT) — will retry", car_id)
            return
        route_id = self.orchestrator.next_route_id()
        # 접근 방향(좌/우)과 진입 선회 반경은 차량 현재 위치가 정한다. 슬롯에
        # 따라 "지나쳐 버려서 전진으로 못 돌아오는" 경우가 있으므로 반드시
        # strict 로 만들고, 불가능하면 무장하지 않는다 — 예전에는 그대로 실어
        # 보내서 차가 뒤로 가야 하는 경로를 받았다.
        slot_id, wps = self._feasible_route(view, slot_id, route_id)
        if slot_id is None:
            return                       # 갈 수 있는 칸이 없다 — 다음 프레임 재시도

        if self.auto_host_mode:
            if not self._start_auto_host(car_id, slot_id, wps):
                return
            view.slot_id = slot_id
            self._emit_route(wps, car_id=car_id)
            log.info("car %d → slot %s (route %d, %d waypoints)",
                     car_id, slot_id, route_id, len(wps))
            return
        try:
            self.orchestrator.start_mission(car_id, wps, slot_id=slot_id)
        except RuntimeError as exc:
            # 직전 명령(SET_MODE 등)의 ack 를 아직 못 받았다. 다음 재시도 주기에
            # 다시 붙는다 — 슬롯은 아직 확정하지 않는다.
            log.info("car %d: mission start deferred (%s)", car_id, exc)
            return
        view.slot_id = slot_id
        log.info("car %d → slot %s (route %d, %d waypoints)",
                 car_id, slot_id, route_id, len(wps))
        self.dashboard.push_event("slot_assigned", car_id=car_id, slot=slot_id,
                                  route_id=route_id)

    def _feasible_route(self, view: VehicleView, slot_id: str, route_id: int):
        """RL 이 고른 칸으로 경로를 만들되, 물리적으로 불가하면 대체한다.

        접근 방향과 진입 선회 반경은 차량 현재 자세가 정한다. 합류점보다
        왼쪽에 있는 칸은 지나쳐 버려 전진으로 되돌아올 수 없으므로, 그런
        경로를 그대로 실으면 차가 뒤로 가야 하는 목표를 받는다(실차 확인).

        RL 은 혼잡도만 보고 고르지 이 기하를 모른다. 그래서 여기서 한 번
        걸러 **도달 가능한 가장 가까운 빈 칸**으로 바꾼다. 슬롯 점유 상태를
        직접 조작해 RL 을 우회하면 정책이 WAIT 으로 굳으므로 하지 않는다.

        Returns:
            (확정 슬롯, waypoint 목록). 갈 수 있는 칸이 없으면 (None, None).
        """
        specs = default_slot_specs()
        try:
            return slot_id, build_waypoints(
                specs[slot_id], route_id=route_id, from_pose=view.position_mm,
                from_heading_deg=view.heading_deg,
                min_radius_mm=self._plan_radius, strict=True)
        except InfeasibleRouteError as exc:
            reason = exc.reason

        alt = self._nearest_feasible_slot(view, exclude=slot_id)
        if alt is None:
            # 어느 칸도 안 되면 슬롯 문제가 아니라 **차량 자세 문제**다
            # (통로 합류 자체가 불가능하면 8칸이 같은 이유로 실패한다).
            # RL 배정은 캐시되므로 다음 프레임에도 같은 칸이 나온다 — 조용히
            # 반복하지 말고 무엇을 해야 하는지 주기적으로 알린다.
            self._reject_slot(view.car_id, slot_id, reason)
            self._warn_no_route(view, reason)
            return None, None
        log.warning("car %s: RL 배정 %s 는 현재 자세에서 불가 (%s) — %s 로 대체",
                    view.car_id, slot_id, reason, alt)
        self.allocator.reassign(view.track_id, alt)
        self.dashboard.push_event("slot_reassigned", car_id=view.car_id,
                                  slot=alt, replaced=slot_id, reason=reason)
        return alt, build_waypoints(
            specs[alt], route_id=route_id, from_pose=view.position_mm,
            from_heading_deg=view.heading_deg,
            min_radius_mm=self._plan_radius, strict=True)

    def _warn_no_route(self, view: VehicleView, reason: str) -> None:
        """갈 수 있는 칸이 하나도 없을 때 주기적으로 사유를 알린다."""
        now = time.monotonic()
        if now - self._last_no_route_warn < 3.0:
            return
        self._last_no_route_warn = now
        x, y = view.position_mm
        need = (AISLE_Y - y) / 10.0
        log.warning(
            "car %s: 현재 자세(%.0f,%.0f)mm hdg %s 에서 **갈 수 있는 슬롯이 없다** — %s",
            view.car_id, x, y,
            "None" if view.heading_deg is None else f"{view.heading_deg:.0f}deg",
            reason)
        if 0.0 < need:
            log.warning("   → 차를 통로(y=%.0fcm)에 더 붙이거나, "
                        "--turn-radius %.0f 이하로 주고 다시 실행",
                        AISLE_Y / 10.0, max(need - 1.0, 1.0))

    def _nearest_feasible_slot(self, view: VehicleView, *,
                               exclude: str | None = None) -> str | None:
        """현재 자세에서 갈 수 있는 빈 칸 중 인계 지점이 가장 가까운 것."""
        best, best_d = None, float("inf")
        for slot_id, spec in default_slot_specs().items():
            if slot_id == exclude:
                continue
            idx = SLOT_NAMES.index(slot_id)
            if self.allocator.slot_statuses[idx] >= 1.0:
                continue                                  # 이미 점유/선점됨
            plan = plan_handoff(spec, from_pose=view.position_mm,
                                from_heading_deg=view.heading_deg,
                                min_radius_mm=self._plan_radius)
            if not plan.feasible:
                continue
            d = math.hypot(plan.point[0] - view.position_mm[0],
                           plan.point[1] - view.position_mm[1])
            if d < best_d:
                best, best_d = slot_id, d
        return best

    def _reject_slot(self, car_id: int, slot_id: str, reason: str) -> None:
        """차량 현재 위치에서 갈 수 없는 슬롯을 배정 후보에서 뺀다.

        점유 상태를 조작하지는 않는다 — 정책이 그걸 혼잡으로 읽어 WAIT 으로
        굳어버린다(실측). 기록·보고용으로만 남긴다.
        """
        if slot_id in self._unreachable_slots:
            return
        self._unreachable_slots.add(slot_id)
        log.warning("car %d: 슬롯 %s 경로 불가 — %s. 배정 후보에서 제외하고 재배정",
                    car_id, slot_id, reason)
        self.dashboard.push_event("slot_unreachable", car_id=car_id, slot=slot_id,
                                  reason=reason)

    def _push_to_vehicle(self, view: VehicleView) -> None:
        """도착 판정 + POSE 스트림 갱신."""
        if view.car_id is None:
            return
        mission = self.orchestrator.missions.get(view.car_id)
        # PARKED 재검증은 실제 정지를 함께 확인한다 (§11)
        if mission is not None and mission.state is MissionState.PARKED_CHECK:
            if not view.is_stationary(self.config.stationary_tolerance_mm,
                                      self.config.stationary_window):
                return
        mission = self.orchestrator.missions.get(view.car_id)
        self.dashboard.push_pose(
            car_id=view.car_id, position_mm=view.position_mm,
            status="parked" if mission and mission.state is MissionState.DONE else "moving",
            heading_deg=view.heading_deg, heading_source=view.heading_source,
            parking_phase=mission.current.phase if mission and mission.current else None,
            route_id=mission.route_id if mission else None,
            waypoint_id=mission.current.waypoint_id if mission and mission.current else None,
        )
        if self.auto_host_mode:
            self._feed_auto_host(view)
        else:
            self.orchestrator.update_pose(view.car_id, view.position_mm, view.heading_deg)
        self.server.push_pose(
            view.car_id,
            x_cm=view.position_mm[0] / 10.0,
            y_cm=view.position_mm[1] / 10.0,
            heading_deg=view.heading_deg,
            heading_source=view.heading_source,
            position_confidence=view.confidence,
            heading_confidence=0.9 if view.heading_source == "TRAJECTORY" else 0.5,
            valid=view.heading_deg is not None,
        )
        # AUTO_HOST 에서는 제어 소유자가 AutoHostRunner 하나뿐이다.
        # 여기서 push_control 을 또 부르면 두 곳이 같은 스트림을 다투게 된다.
        if self.config.direct_control and not self.auto_host_mode:
            self._update_control(view)

    # ─── AUTO_HOST (하드웨어팀 패키지 경로) ──────────────────────────────────

    @property
    def auto_host_mode(self) -> bool:
        return self.config.control_mode == "auto-host"

    def _start_manual_shell(self, car_id: int) -> None:
        """READY 직후 세션만 열고 수동(WASD) 조작을 가능하게 한다."""
        runner = AutoHostRunner(self.server, car_id, [],
                                period_s=self.config.auto_host_period_s,
                                config=self.config.controller_config)
        runner.on_status_change = self._on_auto_host_status
        mux = None
        try:
            runner.arm_session(wait_s=self.config.auto_host_handshake_s)
            mux = HybridControlMux(runner)
            mux.switch_to_manual()
            with self._lock:
                self.auto_hosts[car_id] = runner
                self.hybrid_controls[car_id] = mux
            log.info("car %d: 수동 셸 준비됨 (MANUAL_WASD)", car_id)
        except Exception as exc:                    # noqa: BLE001
            if mux is not None:
                mux.stop()
            runner.stop()
            log.warning("car %d: 수동 셸 준비 실패 (%s)", car_id, exc)
        finally:
            with self._lock:
                self._manual_shell_starting.discard(car_id)

    def _start_auto_host(self, car_id: int, slot_id: str,
                         waypoints: list[Any]) -> bool:
        """AUTO_HOST 주행을 건다. 수동 셸이 이미 있으면 경로만 갈아끼운다."""
        runner = self.auto_hosts.get(car_id)
        mux = self.hybrid_controls.get(car_id)
        if runner is not None:
            runner.load_route(waypoints)
            if mux is not None:
                # AUTO_PENDING → 새 카메라 pose 를 받은 뒤에 주행이 재개된다
                mux.switch_to_auto()
            else:
                self.hybrid_controls[car_id] = HybridControlMux(runner)
            self._auto_host_slot[car_id] = slot_id
            self.dashboard.push_event("auto_host_armed", car_id=car_id, slot=slot_id)
            return True

        runner = AutoHostRunner(self.server, car_id, waypoints,
                                period_s=self.config.auto_host_period_s,
                                config=self.config.controller_config)
        runner.on_status_change = self._on_auto_host_status
        try:
            runner.start(wait_s=self.config.auto_host_handshake_s)
        except (ModeHandshakeError, RuntimeError) as exc:
            log.warning("car %d: AUTO_HOST 시작 실패 (%s) — 재시도", car_id, exc)
            runner.stop()
            return False
        self.auto_hosts[car_id] = runner
        self.hybrid_controls[car_id] = HybridControlMux(runner)
        self._auto_host_slot[car_id] = slot_id
        self._replan_attempts.pop(car_id, None)
        self.dashboard.push_event("auto_host_armed", car_id=car_id, slot=slot_id)
        return True

    # ─── 수동/자동 전환 API (hybrid_gui.py 가 호출) ──────────────────────────

    def hybrid_available(self, car_id: int) -> bool:
        return car_id in self.hybrid_controls

    def hybrid_mode(self, car_id: int) -> str:
        mux = self.hybrid_controls.get(car_id)
        return mux.mode if mux is not None else "UNAVAILABLE"

    def switch_to_manual(self, car_id: int) -> None:
        mux = self._require_mux(car_id)
        mux.switch_to_manual()

    def switch_to_auto(self, car_id: int) -> None:
        """AUTO_PENDING 으로 두고, 새 카메라 pose 가 오면 그때 주행을 재개한다."""
        mux = self._require_mux(car_id)
        mux.switch_to_auto()

    def set_manual_drive(self, car_id: int, throttle: float, steering: float) -> None:
        mux = self.hybrid_controls.get(car_id)
        if mux is not None:
            mux.set_manual_wire(throttle, steering)

    def manual_stop(self, car_id: int) -> None:
        """즉시 정지. AUTO 가 100ms 뒤에 덮어쓰지 않도록 MANUAL 로 내린다."""
        mux = self._require_mux(car_id)
        if mux.mode != "MANUAL_WASD":
            mux.switch_to_manual()
        mux.set_manual_wire(0.0, 0.0)
        self.server.stop_control(car_id)

    def _require_mux(self, car_id: int) -> HybridControlMux:
        mux = self.hybrid_controls.get(car_id)
        if mux is None:
            raise RuntimeError(
                f"car {car_id}: 아직 AUTO_HOST 세션이 없습니다 "
                "(차량 접속·슬롯 배정 후에 사용 가능)")
        return mux

    def _feed_auto_host(self, view: VehicleView) -> None:
        """카메라 관측만 넘긴다. 제어 계산·송신은 러너의 100ms 루프가 한다."""
        runner = self.auto_hosts.get(view.car_id)
        if runner is None:
            return
        # mux 가 있으면 그쪽으로 넣는다 — MANUAL 중에도 pose 는 최신으로 유지하되
        # 구동은 하지 않고, AUTO_PENDING 이면 새 pose 를 받은 뒤에만 자동 재개한다.
        mux = self.hybrid_controls.get(view.car_id)
        target = mux if mux is not None else runner
        target.on_camera_pose(view.position_mm[0], view.position_mm[1],
                              view.heading_deg, view.last_obs_time)
        self._check_path_deviation(view, runner)

    def _recoverable(self, target: Any) -> bool:
        """이 waypoint 에서 후진 복구를 걸어도 되는가 (phase 기준)."""
        phase = str(getattr(target, "phase", "") or "").upper()
        return phase in self.config.recover_phases

    def _check_path_deviation(self, view: VehicleView, runner: AutoHostRunner) -> None:
        """매 프레임 "지금 목표를 전진으로 잡을 수 있나"를 본다.

        기존 트리거(APPROACH 놓침 / ALIGN 방향 불일치)는 주차 단계에서만
        돈다. 진입 원호는 전부 CRUISE 라, 차가 원호 바깥으로 밀려 목표를
        지나쳐도 아무것도 걸리지 않고 계속 앞으로만 갔다.

        판정은 후진 계획기와 같은 기준이다 — 목표가 좌/우 최소 선회원 안에
        들어갔거나, 등 뒤인데 되돌아올 원이 맵에 안 들어가면 이탈이다.
        반경은 **실측값**을 쓴다. 계획 반경을 낮춰 잡았더라도 차가 실제로
        돌 수 있는 크기가 도달 가능성을 정한다.

        연속 관측을 요구해 pose 잡음에 걸리지 않게 한다.
        """
        car_id = view.car_id
        if car_id is None or runner.status is not MissionStatus.RUNNING:
            return
        target = runner.current_target
        if target is None or view.heading_deg is None:
            return
        # 궤적 heading 이면 후진 시 180° 뒤집히므로 애초에 판정하지 않는다.
        if view.heading_source == "TRAJECTORY":
            return
        # 후진 목표에 전진 도달성을 따지면 안 된다. 복구 waypoint 는 정의상
        # 등 뒤에 있어서 매번 "이탈"로 잡히고, 복구가 복구를 부르며 몇 프레임
        # 만에 재시도 횟수를 태워버린다.
        if (runner.mission.is_recovering
                or getattr(getattr(target, "motion_direction", None), "value", "")
                == "REVERSE"):
            self._deviation_streak.pop(car_id, None)
            return
        # 통로 중간 점은 허용오차가 넓고 다음 점이 이어진다 — 조금 밀려도 계속
        # 가면 된다. 여기서 후진을 걸면 진행이 끊긴다.
        if not self._recoverable(target):
            self._deviation_streak.pop(car_id, None)
            return

        off = forward_unreachable(
            view.position_mm, view.heading_deg, (target.x_mm, target.y_mm),
            radius_mm=MIN_TURN_RADIUS_MM,
            lot_mm=(self.config.lot_width_mm, self.config.lot_height_mm))
        if not off:
            self._deviation_streak.pop(car_id, None)
            return

        n = self._deviation_streak.get(car_id, 0) + 1
        self._deviation_streak[car_id] = n
        if n < self.config.deviation_frames:
            return
        self._deviation_streak.pop(car_id, None)

        # 미션을 REPLAN_REQUIRED 로 올리면 기존 배선(_on_auto_host_status)이
        # 후진 복구를 만들어 끼운다. 여기서 직접 만들지 않는다.
        log.warning("car %d: 경로 이탈 — 목표 wp%s(%.0f,%.0f) 를 전진으로 못 잡는다 "
                    "(pose %.0f,%.0f hdg %.0f°)", car_id,
                    getattr(target, "waypoint_id", "?"), target.x_mm, target.y_mm,
                    view.position_mm[0], view.position_mm[1], view.heading_deg)
        runner.mission.request_replan("PATH_DEVIATION")
        self.dashboard.push_event("path_deviation", car_id=car_id,
                                  waypoint_id=getattr(target, "waypoint_id", None))

    def _on_auto_host_status(self, car_id: int, prev: MissionStatus,
                             status: MissionStatus) -> None:
        """AUTO_HOST 미션 상태 변화 → 기존 슬롯·대시보드 로직에 연결.

        이 경로가 없으면 FINAL 도착이 파이프라인까지 올라오지 않아 슬롯이
        영원히 비어 있는 것으로 남는다 (WAYPOINT_AUTO 의 PARKED_CHECK 에 해당).
        """
        if status is MissionStatus.DONE:
            # 정지 재확인은 카메라를 보는 이쪽 몫이다 (§11). 프레임 루프에서 판정한다.
            log.info("car %d: AUTO_HOST 최종 waypoint 도착 — 정지 확인 중", car_id)
        elif status is MissionStatus.REPLAN_REQUIRED:
            # 먼저 후진 복구를 시도한다. 전진으로 못 잡는 자세라 재계획을 해봐야
            # 같은 기하가 다시 나오기 때문이다 (같은 슬롯 = 같은 원호).
            if self._recover_auto_host(car_id):
                return
            log.info("car %d: 후진 복구 불가 — 전체 재계획", car_id)
            self._replan_auto_host(car_id)

    def _check_auto_host_parked(self, view: VehicleView) -> None:
        """DONE 인 차량이 실제로 멈췄는지 확인하고 PARKED 를 확정한다."""
        runner = self.auto_hosts.get(view.car_id)
        if runner is None or runner.status is not MissionStatus.DONE:
            return
        if not view.is_stationary(self.config.stationary_tolerance_mm,
                                  self.config.stationary_window):
            return
        runner.confirm_parked()
        runner.stop()                      # 제어 스트림 0 으로 고정
        slot_id = self._auto_host_slot.get(view.car_id)
        if slot_id is not None:
            self._on_parked(view.car_id, slot_id)

    def _recover_auto_host(self, car_id: int) -> bool:
        """전진으로 못 잡는 자세면 후진 복구 경로를 끼워 넣는다.

        미션이 복구 구간을 마치면 실패했던 target 부터 원래 route 로 자동
        복귀하므로, 여기서는 "얼마나 물러날지" 한 점만 만들면 된다.

        Returns:
            복구 경로를 실제로 적재했으면 True.
        """
        runner = self.auto_hosts.get(car_id)
        if runner is None:
            return False
        # REPLAN_REQUIRED 에서는 current_target 이 None 이다 — 실패한 target 을 쓴다.
        target = runner.failed_target
        if target is None:
            return False

        if not self._recoverable(target):
            log.info("car %d: 후진 대상 phase 가 아니라 보류 (%s, 허용 %s)",
                     car_id, getattr(target, "phase", "?"),
                     "/".join(self.config.recover_phases) or "없음")
            return False

        reason = runner.replan_reason or ""
        if reason not in REVERSE_TRIGGER_REASONS:
            log.info("car %d: 후진 대상 아닌 사유 (%s)", car_id, reason)
            return False

        track_id = self.track_of_car.get(car_id)
        view = self.views.get(track_id) if track_id is not None else None
        if view is None:
            return False

        # heading 을 이동 궤적에서 추정하는 동안에는 후진하면 안 된다.
        # 후진하면 진행 방향이 뒤집혀 추정 heading 이 180° 틀어지고, 제어기가
        # 그 값을 믿고 반대로 조향해 상황을 더 나쁘게 만든다. 전방 쿠션을
        # 잡아 방향을 직접 재는 동안(heading_source != TRAJECTORY)만 허용한다.
        if view.heading_source == "TRAJECTORY":
            log.info("car %d: heading 이 궤적 추정이라 후진 보류 "
                     "(전방 쿠션 미탐지)", car_id)
            return False

        wps = plan_reverse_recovery(
            view.position_mm, view.heading_deg, target,
            route_id=self.orchestrator.next_route_id(), reason=reason,
            bounds_mm=(self.config.lot_width_mm, self.config.lot_height_mm),
        )
        if not wps:
            return False

        try:
            status = runner.load_recovery_waypoints(wps)
        except (RuntimeError, ValueError) as exc:
            log.warning("car %d: 후진 복구 적재 실패 (%s)", car_id, exc)
            return False
        if status is MissionStatus.RECOVERY_FAILED:
            log.warning("car %d: 후진 복구 횟수 초과 — 포기", car_id)
            return False

        self._emit_route(wps, recovery=True)
        log.info("car %d: 후진 복구 (%s) → (%.0f,%.0f) 로 %.0fmm 후진",
                 car_id, reason, wps[0].x, wps[0].y,
                 math.hypot(wps[0].x - view.position_mm[0],
                            wps[0].y - view.position_mm[1]))
        self.dashboard.push_event("reverse_recovery", car_id=car_id, reason=reason)
        return True

    def _replan_auto_host(self, car_id: int) -> None:
        """현재 pose 에서 슬롯까지 새 경로를 만들어 러너에 갈아 끼운다.

        같은 슬롯이면 기하가 거의 같아 같은 지점에서 다시 실패하기 쉽다.
        횟수를 제한하지 않으면 REPLAN_REQUIRED ↔ RUNNING 을 무한히 오간다.
        """
        runner = self.auto_hosts.get(car_id)
        slot_id = self._auto_host_slot.get(car_id)
        track_id = self.track_of_car.get(car_id)
        view = self.views.get(track_id) if track_id is not None else None
        if runner is None or slot_id is None or view is None:
            log.warning("car %d: AUTO_HOST 재계획 불가 (pose/슬롯 없음)", car_id)
            return

        tries = self._replan_attempts.get(car_id, 0) + 1
        self._replan_attempts[car_id] = tries
        if tries > self.config.max_replan_attempts:
            log.warning("car %d: 재계획 %d회 초과 — 정지하고 REPLAN_REQUIRED 유지 "
                        "(경로가 차량 선회 반경으로 불가능할 수 있음)", car_id, tries - 1)
            runner.stop()
            self.dashboard.push_event("replan_exhausted", car_id=car_id,
                                      slot=slot_id, attempts=tries - 1)
            return
        route_id = self.orchestrator.next_route_id()
        # 현재 pose 를 기준으로 다시 만든다. route_nodes=[node] 만 넘기던 옛
        # 방식은 노드가 CRUISE 목록에서 잘려나가 결국 원래와 같은 경로가
        # 나왔고, 같은 지점에서 다시 실패해 무한 재계획이 됐다.
        try:
            wps = build_waypoints(default_slot_specs()[slot_id], route_id=route_id,
                                  from_pose=view.position_mm,
                                  from_heading_deg=view.heading_deg, strict=True)
        except InfeasibleRouteError as exc:
            log.warning("car %d: 현재 위치에서 슬롯 %s 재계획 불가 (%s) — 정지",
                        car_id, slot_id, exc.reason)
            runner.stop()
            return
        runner.load_route(wps)
        self._emit_route(wps, car_id=car_id)
        log.info("car %d: AUTO_HOST 재계획 → route %d (%d개)",
                 car_id, route_id, len(wps))

    def _emit_route(self, waypoints: list[Any], *, recovery: bool = False,
                    car_id: int | None = None) -> None:
        """적재한 route 를 기록기·화면에 넘긴다 (route.json / overlay)."""
        if car_id is not None and not recovery:
            self._auto_host_route[car_id] = list(waypoints)
        if self.on_route_load is not None:
            self.on_route_load(list(waypoints), recovery)

    # ─── B안 주행 제어 ───────────────────────────────────────────────────────

    def _update_control(self, view: VehicleView) -> None:
        """현재 pose 와 목표 waypoint 로 throttle/steering 을 만들어 스트림에 싣는다.

        구동을 허용하는 건 미션이 DRIVING 인 동안뿐이다. 전환(SWITCHING/
        LOADING/RESUMING)·정지(HELD)·주차 확인(PARKED_CHECK) 구간에서는
        0 을 계속 내보낸다 — 마지막 값이 유지되는 스트림이라 명시적으로
        0 을 실어야 차가 타력 주행하지 않는다.
        """
        car_id = view.car_id
        if car_id is None:
            return
        mission = self.orchestrator.missions.get(car_id)
        target = mission.current if mission is not None else None
        if target is None:
            self.server.stop_control(car_id)
            self.last_control.pop(car_id, None)
            return

        ctrl = self.controllers.get(car_id)
        if ctrl is None:
            ctrl = self.controllers[car_id] = WaypointController(self.config.vehicle_limits)

        allow = (mission.state is MissionState.DRIVING
                 and car_id not in self._comm_lost
                 and car_id not in self._collision_held)
        out = ctrl.compute(
            # 이 프레임에 실제로 탐지된 차량이므로 위치는 유효하다.
            # heading 유무는 제어기가 따로 판정한다 (NO_HEADING).
            Pose(view.position_mm[0], view.position_mm[1], view.heading_deg,
                 timestamp=time.monotonic(), valid=True),
            target, allow_drive=allow)
        self.last_control[car_id] = out
        self.server.push_control(car_id, out.throttle, out.steering)

        prev = self._last_control_mode.get(car_id)
        if prev != out.mode:
            self._last_control_mode[car_id] = out.mode
            log.info("car %d control %s → %s (dist %.1fcm, err %.0f°, "
                     "thr %.2f, str %.2f)%s",
                     car_id, prev or "-", out.mode, out.distance_cm,
                     out.heading_error_deg, out.throttle, out.steering,
                     f" [{out.reason}]" if out.reason else "")

    # ─── 안전 ────────────────────────────────────────────────────────────────

    def _check_collisions(self, seen: list[VehicleView]) -> None:
        poses = []
        for v in seen:
            if v.car_id is None:
                continue
            m = self.orchestrator.missions.get(v.car_id)
            if m is None or m.current is None:
                continue
            progress = m.index / max(len(m.waypoints), 1)
            # HELD/DONE 처럼 정지가 확정된 상태가 아니면 언제든 움직일 수 있다고 본다.
            # (LOADING·RESUMING 은 곧 출발하는 상태이므로 정지로 취급하면 위험)
            moving = m.state not in (MissionState.HELD, MissionState.DONE)
            poses.append(VehiclePose(
                car_id=v.car_id, position=v.position_mm, next_waypoint=m.current,
                progress=progress, is_moving=moving,
            ))
        if len(poses) < 2:
            self._resume_cleared(set())
            return

        at_risk: set[int] = set()
        for event in self.collision.check(poses):
            at_risk.add(event.stop_car_id)
            stop_m = self.orchestrator.missions.get(event.stop_car_id)
            keep_m = self.orchestrator.missions.get(event.keep_car_id)
            if stop_m is None or stop_m.state is MissionState.HELD:
                continue                      # 이미 조치된 차량
            if event.stop_car_id in self._comm_lost:
                continue                      # 링크 단절 — 명령이 닿지 않는다
            if keep_m is not None and keep_m.state is MissionState.HELD:
                # 상대가 이미 멈춰 있다. 여기서 이쪽까지 세우면 두 대 모두
                # 정지해 교착이 된다 — 통과 우선순위를 받은 차를 보낸다.
                continue
            log.warning("collision risk: car %d holds (%s, %.0fmm)",
                        event.stop_car_id, event.reason, event.distance_mm)
            self.orchestrator.hold(event.stop_car_id, "COLLISION_RISK")
            self._collision_held.add(event.stop_car_id)
            self.dashboard.push_event("vehicle_hold", car_id=event.stop_car_id,
                                      reason=event.reason,
                                      distance_mm=event.distance_mm)

        self._resume_cleared(at_risk)

    def _resume_cleared(self, at_risk: set[int]) -> None:
        """위험이 해소된 차량을 재개한다 (충돌로 세운 차량만 대상)."""
        for car_id in list(self._collision_held):
            if car_id in at_risk:
                continue
            if car_id in self._comm_lost:
                continue      # 링크가 죽어 있다. 복구 시 재계획으로 풀린다
            self._collision_held.discard(car_id)
            m = self.orchestrator.missions.get(car_id)
            if m is not None and m.state is MissionState.HELD:
                log.info("collision cleared: car %d resumes", car_id)
                self.orchestrator.resume(car_id)
                self.dashboard.push_event("vehicle_resume", car_id=car_id)

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _on_parked(self, car_id: int, slot_id: str) -> None:
        log.info("car %d PARKED at %s", car_id, slot_id)
        self.allocator.set_slot_occupied(slot_id, True)
        # 주차를 마친 차량은 더 이상 통로를 점유하지 않는다 → 혼잡 계산에서 제외
        track_id = self.track_of_car.get(car_id)
        tracked = self.allocator.vehicles.get(track_id) if track_id is not None else None
        if tracked is not None:
            tracked.route = []
        self.dashboard.push_event("parked", car_id=car_id, slot=slot_id)

    def _on_replan_required(self, car_id: int, reason: str) -> None:
        """현재 pose 기준으로 새 route_id 경로를 만들어 재시작한다 (§32)."""
        track_id = self.track_of_car.get(car_id)
        view = self.views.get(track_id) if track_id is not None else None
        mission = self.orchestrator.missions.get(car_id)
        if view is None or mission is None or mission.slot_id is None:
            log.warning("car %d replan requested (%s) but no pose/slot", car_id, reason)
            return
        node = view.node or position_to_node(view.position_mm)
        route_id = self.orchestrator.next_route_id()
        spec = default_slot_specs()[mission.slot_id]
        nodes = [node] if node else None
        wps = build_waypoints(spec, route_id=route_id, route_nodes=nodes)
        log.info("car %d replanning from %s (%s) → route %d", car_id, node, reason, route_id)
        ctrl = self.controllers.get(car_id)
        if ctrl is not None:
            ctrl.reset()          # 경로가 바뀌면 이전 오차 미분항은 무의미하다
        self.orchestrator.regenerate(car_id, wps)

    def _on_comm_fail(self, car_id: int, info: dict[str, Any]) -> None:
        """통신 장애 — 미션을 정지 상태로 두고 복구를 기다린다.

        서버가 장애 시작 엣지에서만 부르므로 여기서 다시 debounce 하지 않는다.
        차량 자체는 펌웨어 COMM_TIMEOUT 으로 안전정지하므로 명령을 보내지 않는다.
        """
        kind = info.get("type", "COMM_FAIL")
        with self._lock:
            self._comm_lost.add(car_id)
        held = self.orchestrator.mark_link_lost(car_id)
        log.warning("car %d comm fail (%s) — mission %s",
                    car_id, kind, "held" if held else "none active")
        self.dashboard.push_event("comm_fail", car_id=car_id, reason=kind)

    def _on_comm_recovered(self, car_id: int) -> None:
        """통신 복구 — 단절 중 위치를 신뢰할 수 없으므로 현재 pose 로 재계획한다."""
        with self._lock:
            if car_id not in self._comm_lost:
                return
            self._comm_lost.discard(car_id)
        log.info("car %d comm recovered — replanning from current pose", car_id)
        self.dashboard.push_event("comm_recovered", car_id=car_id)
        self._on_replan_required(car_id, "COMM_RECOVERED")

    def _on_resync(self, car_id: int, hello: dict[str, Any]) -> None:
        """재접속·재부팅 후에는 기존 경로를 폐기하고 재계획한다 (§21·26)."""
        log.info("car %d resync (boot_id=%s) — discarding route",
                 car_id, hello.get("boot_id"))
        self.orchestrator.missions.pop(car_id, None)
        with self._lock:
            self._comm_lost.discard(car_id)      # 새 세션이므로 복구 재계획은 불필요
            self._mode_set.discard(car_id)       # 새 세션에서 모드를 다시 잡는다
            self._manual_shell_starting.discard(car_id)
            runner = self.auto_hosts.pop(car_id, None)
            mux = self.hybrid_controls.pop(car_id, None)
            self._auto_host_slot.pop(car_id, None)
            self.controllers.pop(car_id, None)
            self.last_control.pop(car_id, None)
            self._last_control_mode.pop(car_id, None)
            track_id = self.track_of_car.pop(car_id, None)
            if track_id is not None and track_id in self.views:
                self.views[track_id].car_id = None
                self.views[track_id].slot_id = None
        if mux is not None:
            try:
                mux.stop()
            except Exception:                      # noqa: BLE001
                pass
        if runner is not None:
            # 세션이 바뀌었으므로 REMOTE_DIRECT 협상부터 다시 해야 한다.
            runner.stop()

    def _hold_check(self, car_id: int, hello: dict[str, Any]) -> str | None:
        """HELLO 판정 훅 — 카메라가 차량을 못 보면 매핑 불가이므로 HOLD (H3)."""
        if not str(hello.get("motor_stopped", "true")).lower() in ("true", "1"):
            return "MOTOR_NOT_CONFIRMED_STOPPED"
        return None

    def _forget_stale(self, frame_index: int, max_age: int = 60) -> None:
        """오래 안 보이는 track 정리 (탐지 유실 대비 여유를 둔다)."""
        with self._lock:
            stale = [t for t, v in self.views.items()
                     if frame_index - v.last_seen_frame > max_age and v.car_id is None]
            for t in stale:
                self.views.pop(t, None)
                self.heading.remove(t)
                self.allocator.remove_vehicle(t)
