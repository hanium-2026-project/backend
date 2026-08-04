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
from typing import Any

from comm import MissionOrchestrator, MissionState, VehicleServer
from cv.association import associate
from cv.heading import HeadingEstimator
from cv.homography import compute_homography, warp_point
from cv.tracker import RCCarTracker, TrackState
from cv.vehicle_detector import YoloVehicleDetector
from parking.safety import CollisionMonitor, VehiclePose
from parking.waypoints import build_waypoints, default_slot_specs
from rl.bridge import RealtimeAllocator, position_to_node

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
    last_alloc_frame: int = -10_000        # 슬롯 배정 재시도 조절용
    recent: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))

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
        self.views: dict[int, VehicleView] = {}          # track_id → 관측
        self.track_of_car: dict[int, int] = {}           # car_id → track_id
        self._pending_cars: list[int] = []               # HELLO 순서 대기열
        self._lock = threading.Lock()
        self._tracker: RCCarTracker | None = None

        self.server.on_status = self.orchestrator.on_vehicle_status
        self.server.on_command_rejected = self.orchestrator.on_command_rejected
        self.server.on_ready = self._on_vehicle_ready
        self.server.on_resync = self._on_resync
        self.server.hold_check = self._hold_check
        self.orchestrator.on_replan_required = self._on_replan_required

    # ─── 라이프사이클 ────────────────────────────────────────────────────────

    def start(self) -> None:
        """TCP 서버를 띄우고 ESP32 접속을 받는다 (논블로킹)."""
        self.server.start()
        log.info("vehicle server listening on %s:%d",
                 self.config.server_host, self.server.bound_port)

    def stop(self) -> None:
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
        return image

    # ─── 프레임 처리 (핵심) ──────────────────────────────────────────────────

    def on_frame(self, state: TrackState) -> None:
        """탐지 결과 1프레임을 좌표·판정·명령까지 흘린다."""
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
                                     front_px=pair.cushion_center_px)
            seen.append(view)
        for det in unpaired:
            if det.track_id is None:
                continue
            seen.append(self._update_view(det, state.frame_index))

        for view in seen:
            self._ensure_mission(view, state.frame_index)
            self._push_to_vehicle(view)

        self._check_collisions(seen)
        self._forget_stale(state.frame_index)

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
                     front_px: tuple[float, float] | None = None) -> VehicleView:
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
        if car_id in self.orchestrator.missions:
            return                       # 미션 진행 중
        if frame_index - view.last_alloc_frame < self.config.alloc_retry_frames:
            return                       # 재시도 주기 대기
        view.last_alloc_frame = frame_index

        self.allocator.update(view.track_id, view.position_mm)
        slot_id = self.allocator.allocate(view.track_id)
        if slot_id is None:
            log.info("car %d: no slot available (RL WAIT) — will retry", car_id)
            return
        view.slot_id = slot_id
        route_id = self.orchestrator.next_route_id()
        wps = build_waypoints(default_slot_specs()[slot_id], route_id=route_id)
        self.orchestrator.start_mission(car_id, wps, slot_id=slot_id)
        log.info("car %d → slot %s (route %d, %d waypoints)",
                 car_id, slot_id, route_id, len(wps))
        self.dashboard.push_event("slot_assigned", car_id=car_id, slot=slot_id,
                                  route_id=route_id)

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
        self.orchestrator.regenerate(car_id, wps)

    def _on_resync(self, car_id: int, hello: dict[str, Any]) -> None:
        """재접속·재부팅 후에는 기존 경로를 폐기하고 재계획한다 (§21·26)."""
        log.info("car %d resync (boot_id=%s) — discarding route",
                 car_id, hello.get("boot_id"))
        self.orchestrator.missions.pop(car_id, None)
        with self._lock:
            track_id = self.track_of_car.pop(car_id, None)
            if track_id is not None and track_id in self.views:
                self.views[track_id].car_id = None
                self.views[track_id].slot_id = None

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
