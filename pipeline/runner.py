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
from cv.vehicle_detector import LABEL_CAR, LABEL_CUSHION
from cv.heading import HeadingEstimator
from cv.homography import compute_homography, warp_point
from cv.tracker import RCCarTracker, TrackState
from cv.vehicle_detector import YoloVehicleDetector
from parking.recovery import (REVERSE_TRIGGER_REASONS, forward_unreachable,
                              plan_reverse_recovery)
from parking.final_alignment import (build_final_alignment_waypoints,
                                     build_final_straight_reverse_waypoints,
                                     evaluate_final_pose, in_final_region,
                                     rear_parked_heading_deg)
from parking.safety import CollisionMonitor, VehiclePose
from parking.trajectory_safety import validate_trajectory
from parking.waypoints import (AISLE_Y, ALONG_AISLE_HEADING_TOLERANCE_DEG,
                               HANDOFF_LEAD_MM, MIN_TURN_RADIUS_MM,
                               CAR_LENGTH_MM, CAR_WIDTH_MM,
                               ON_AISLE_TOLERANCE_MM, PHASE_DEFAULTS,
                               InfeasibleRouteError,
                               _car_footprint,
                               build_rear_candidate_waypoints,
                               build_rear_entry_waypoints,
                               build_rear_parking_waypoints,
                               build_setup_recovery_waypoints,
                               choose_rear_parking_plan,
                               build_waypoints, default_slot_specs, plan_handoff,
                               STAGING_RADIUS_CANDIDATES)
from rl.bridge import RealtimeAllocator, position_to_node
from rl.parking_env import SLOT_NAMES
from integration.remote_direct_session import RemoteDirectSession

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
        # COMM loss 뒤에도 목적(slot/parking stage)은 보존하되 executable route는
        # 폐기한다. 새 세션 + fresh pose/heading + trajectory 검증 뒤에만 해제한다.
        self._comm_recovery_context: dict[int, dict[str, Any]] = {}
        self._comm_recovery_starting: set[int] = set()
        # B안: 차량별 주행 제어기 (throttle/steering 계산)
        self.controllers: dict[int, WaypointController] = {}
        self.last_control: dict[int, ControlOutput] = {}
        self._mode_set: set[int] = set()                 # SET_MODE 완료 차량
        # Legacy DIRECT_CONTROL also requests mode through the same session
        # negotiation owner; pipeline never emits SET_MODE itself.
        self._direct_sessions: dict[int, RemoteDirectSession] = {}
        self._direct_mode_starting: set[int] = set()
        self._last_control_mode: dict[int, str] = {}     # 로그 중복 억제
        # AUTO_HOST 모드: 차량별 제어 소유자 (waypoint-auto 에서는 비어 있다)
        self.auto_hosts: dict[int, AutoHostRunner] = {}
        self._auto_host_slot: dict[int, str] = {}        # car_id → 배정 슬롯
        self._replan_attempts: dict[int, int] = {}       # car_id → 재계획 횟수
        self._deviation_streak: dict[int, int] = {}      # car_id → 연속 이탈 프레임
        self._last_no_route_warn = 0.0                   # 경로 불가 경고 간격 제한
        self._unreachable_slots: dict[int, set[str]] = {}  # car_id → pose 기준 경로불가
        self._boundary_soft: set[int] = set()
        self._boundary_uncertain: set[int] = set()
        self._boundary_hard: set[int] = set()
        self._boundary_terminal_streak: dict[int, int] = {}
        self._boundary_uncertain_trend: dict[int, tuple[float, float, int]] = {}
        self._boundary_motion: dict[
            int, tuple[float, float, float, float]] = {}
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
        # 후진 구간에서 '지금까지 목표에 가장 가까웠던 거리' (car_id -> (wp키, mm))
        self._reverse_closest: dict[int, tuple[Any, float]] = {}
        # Forward acquisition progress, keyed by route/waypoint.  A freshly
        # loaded APPROACH must not be rejected by a static turn-circle test
        # before it has had a real chance to converge.
        self._forward_closest: dict[int, tuple[Any, float]] = {}
        # 후면주차 단계: GLOBAL → (필요 시 SETUP) → LOCAL PARKING.
        # 각 DONE 경계의 fresh camera pose 로 다음 단계를 다시 계획한다.
        self._parking_stage: dict[int, str] = {}
        self._parking_setup_wait: dict[int, float] = {}
        self._parking_plan_wait: dict[int, float] = {}
        # Entrance → aisle staging is a bounded, geometry-driven pre-GLOBAL
        # phase.  Signatures prevent route-id-only retries of the same move.
        self._entry_staging_wait: dict[int, float] = {}
        self._entry_staging_attempts: dict[int, int] = {}
        self._entry_staging_signatures: dict[int, set[tuple[int, ...]]] = {}
        # 배경 staging 계획: 진행중 표시와 완료 결과함. comm 재협상 워커와 같은
        # 구조로 self._lock 아래에서만 만진다.
        self._entry_staging_planning: set[int] = set()
        self._entry_staging_plan: dict[int, dict[str, Any]] = {}
        self._physical_stop_wait_announced: set[tuple[int, str]] = set()
        self._parking_recovery_attempts: dict[int, int] = {}
        self._last_replan_signature: dict[
            int, tuple[str, float, float, float, float]] = {}
        self._initial_pose_samples: dict[
            int, deque[tuple[float, float, float, float | None]]] = {}
        self._allocation_state: dict[int, str] = {}
        self._heading_wait_state: dict[int, str] = {}
        self._heading_wait_started: dict[int, float] = {}
        self._heading_wait_faulted: set[int] = set()
        # heading 대기 timeout 이 authority 를 latch 한 차량. 이건 물리 안전
        # fault 가 아니라 "관측을 기다리다 멈춘" 상태이므로, trusted heading 이
        # 돌아오고 **새 route 가 검증되면** 다시 켤 수 있어야 한다.
        # BOUNDARY_HARD 같은 진짜 terminal fault 와 구분하기 위해 따로 표시한다.
        self._heading_fault_hold: set[int] = set()
        # 슬롯별 vision 정적 점유 상태 (track_id 비의존).
        self._vision_occupancy: dict[str, dict[str, Any]] = {}
        # 확정된 칸의 **정적 장애물 자세**. _parked_obstacles 와 같은 의미이고
        # 키만 car_id 대신 slot_id 다 (전원 꺼진 차에 가짜 car_id 를 주지 않는다).
        self._vision_parked_obstacles: dict[str, tuple[float, float, float]] = {}
        # BOUNDARY_HARD 로 멈춘 차량. 여기 있다고 경계 판정이 완화되는 것은
        # 전혀 없다 — hard/soft/uncertain 임계값도, _check_boundary 도 그대로다.
        # 이 표시가 뜻하는 것은 하나뿐이다: "이 차는 **탈출 경로가 기존
        # 안전 검증을 통과했을 때만** 다시 켤 수 있다".
        self._boundary_escape_hold: set[int] = set()
        # progress watchdog: 아무 진행도 없이 서 있기 시작한 관측 시각.
        self._stall_since: dict[int, float] = {}
        # 최종 정렬/직선후진 시도 횟수(bounded)와 PARKED 연속 확인 카운터.
        self._final_alignment_attempts: dict[int, int] = {}
        self._parked_confirmations: dict[int, int] = {}
        self._parked_last_obs: dict[int, float] = {}
        # PARKED 판정에 실제로 사용된 fresh body pose. 이후 detection/heading
        # dropout이나 COMM resync가 와도 다른 차량의 static obstacle이 사라지면
        # 안 된다. live LAST_VALID로 덮지 않고 다음 명시적 mission까지 보존한다.
        self._parked_obstacles: dict[int, tuple[float, float, float]] = {}
        self._pose_observed_tracks: set[int] = set()
        self._lock = threading.Lock()
        self._tracker: RCCarTracker | None = None

        # ─ Run 기록기 연결점 (tools/run_recorder.py) ─
        # 파이프라인은 기록기를 알지 못한다. 프레임마다 pose 레코드를 만들어
        # last_pose_rec 에 두고, 콜백이 붙어 있으면 밀어준다. 기본은 꺼짐.
        self.on_pose_record: Callable[[dict], None] | None = None
        self.on_event_record: Callable[..., None] | None = None
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

    def run_camera(self, max_frames: int | None = None, show: bool = False,
                   frame_sink=None) -> None:
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
        if show or frame_sink is not None:
            self._tracker.overlay = self._draw_targets
        self._tracker.run(on_frame=self.on_frame, max_frames=max_frames, show=show,
                          frame_sink=frame_sink)

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
            head = (f"car{car_id} {self.workflow_status(car_id)} slot={slot} "
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

            tick = getattr(runner, "last_tick_result", None)
            command = getattr(tick, "command", None)
            if command is not None:
                reason = getattr(command, "reason", "")
                cv2.putText(
                    image,
                    f"   {getattr(command.mode, 'value', command.mode)} "
                    f"thr {command.throttle:+.2f} str {command.steering:+.2f} "
                    f"err {command.heading_error_deg:+.1f}deg"
                    + (f" [{reason}]" if reason else ""),
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    RED if runner.is_faulted else GREEN, 1, cv2.LINE_AA)
                y += 21

        for car_id, slots in sorted(self._unreachable_slots.items()):
            for slot in sorted(slots):
                cv2.putText(image, f"   car{car_id} 슬롯 {slot} 도달불가", (10, y),
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

        # Perception provenance for post-run diagnosis.
        #
        # 2026-08-22 runs died at planning boundaries with heading_source
        # LAST_VALID for 20+ s while the car sat still. The logs could not tell
        # whether the cushion was never *detected* or was detected and dropped
        # by association / the heading jump gate. Those need different fixes, so
        # record the raw counts here rather than guessing again.
        self._last_perception = {
            "det_total": len(state.detections),
            "det_rc_car": sum(1 for d in state.detections
                              if d.label == LABEL_CAR),
            "det_front_cushion": sum(1 for d in state.detections
                                     if d.label == LABEL_CUSHION),
            "assoc_pairs": len(pairs),
            "assoc_unpaired_cars": len(unpaired),
        }

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
        # 배정(_ensure_mission → allocator.allocate)보다 **먼저** 돌아야
        # 이번 프레임의 점유가 이번 프레임의 배정에 반영된다.
        self._update_vision_occupancy(seen, state.timestamp)

        for view in seen:
            self._ensure_mission(view, state.frame_index)
            self._push_to_vehicle(view)
            if self.auto_host_mode:
                self._check_auto_host_parked(view)

        self._check_collisions(seen)
        self._forget_stale(state.frame_index)

    def _update_vision_occupancy(self, seen: list[VehicleView],
                                 obs_time: float) -> None:
        """카메라만으로 "그 칸에 이미 차가 서 있다" 를 판정한다.

        촬영 시나리오: CAR_02 를 전원 OFF 로 슬롯에 손으로 놓는다. ESP 연결도
        CAR_ID binding 도 없다. 따라서 판정은 **pose + 슬롯 기하** 만 쓴다.

        설계 요점
          - slot-centric: 상태를 슬롯별로 들고 있으므로 track_id 가 바뀌어도
            같은 칸에 정지한 차가 계속 보이면 점유가 유지된다.
          - 기존 기하 재사용: in_final_region(=차체 중심이 슬롯 사각형 안)은
            이미 "통로를 지나가는 차를 슬롯 안으로 오인" 하는 문제를 잡으려고
            depth 하한을 -slot.length/2 로 조인 함수다(실측 032539). 새 기하를
            만들지 않는다.
          - 정지 조건: 기존 is_stationary(stationary_tolerance_mm,
            stationary_window)를 그대로 쓴다. 지나가는 차와 세워둔 차를 가른다.
          - ego 제외: car_id 가 붙은 차(=CAR_01, 그리고 정상 미션으로 PARKED 된
            차)는 후보에서 뺀다. 그 칸들은 예약/PARKED 로 base 상태가 이미
            담당하므로 중복 판정할 필요가 없고, CAR_01 이 자기 칸에 들어가면서
            자신을 "외부 주차 차량" 으로 오인하는 것도 여기서 막힌다(§14).
          - base 를 건드리지 않는다: allocator.set_vision_occupied 는 별도
            overlay 만 쓴다. 예약/PARKED 는 vision clear 로 절대 지워지지 않는다.

        운용 순서 (실측으로 확인된 조건)
        --------------------------------
        **정적 차량의 점유가 확정된 뒤에 자율주행 차량을 활성화한다.**

        확정에는 정지 이력(stationary_window) + 연속 관측
        (vision_occupancy_confirm_observations)이 필요해 시간이 걸린다. 실측
        run_20260906_190856 / _191008 에서는 4.5초였다 — 정지한 차인데도 pose
        지터가 14~17mm 로 stationary_tolerance_mm(15) 문턱에 걸쳐 확정 카운터가
        반복해서 되감겼기 때문이다.

        그 사이에 자율주행 차량의 배정(_ensure_mission → allocator.allocate)이
        먼저 일어나면 아직 확정되지 않은 칸이 배정될 수 있다. 위 두 run 에서
        실제로 그렇게 됐다 (배정이 확정보다 0.445s / 0.778s 빨랐고, 결과적으로
        정적 차량이 서 있는 칸이 배정됐다).

        이 경합을 코드로 막는 게이트는 **이 변경 범위에 없다**. 현재 검증된
        조건은 운용 순서뿐이다:

            정적 차량 배치 → backend 시작 → VISION_SLOT_OCCUPIED 확정 로그 확인
            → 그 다음에 자율주행 차량 전원 ON / bind → 배정

        확정 전에 배정이 일어나면 그 칸은 제외되지 않는다.
        """
        if not bool(getattr(self.config, "vision_occupancy_enabled", False)):
            return
        setter = getattr(self.allocator, "set_vision_occupied", None)
        if setter is None:
            return                       # overlay 없는 allocator = 기능 비활성
        specs = default_slot_specs()
        state = self._vision_occupancy
        # 이번 프레임에 "슬롯 안에 정지한 외부 차량" 이 보인 칸.
        hits: dict[str, int] = {}
        seen_pose: dict[str, tuple[float, float, float]] = {}
        for view in seen:
            if view.car_id is not None:
                continue                     # ego / bound 차량은 base 담당
            if not view.is_stationary(self.config.stationary_tolerance_mm,
                                      self.config.stationary_window):
                continue                     # 지나가는 차는 점유가 아니다
            for slot_id, spec in specs.items():
                if in_final_region(spec, view.position_mm[0],
                                   view.position_mm[1]):
                    hits[slot_id] = view.track_id
                    # 정적 장애물 자세: **측정된 위치 + 슬롯의 주차 방향**.
                    #
                    # 위치는 rc_car 검출이 매 프레임 주므로 믿을 수 있다.
                    # 못 믿는 것은 heading 하나뿐이다 — 전원이 꺼진 차는
                    # front_cushion 이 잘 안 잡히고(실측 190856 25.9% /
                    # 191008 4.7%), 정지 상태라 TRAJECTORY 도 못 만든다.
                    # 그 한 축만 슬롯 기하로 바꾼다. 차가 그 칸 안에 있다는
                    # 것은 in_final_region 이 이미 보장했으므로, 슬롯의
                    # 후면주차 완료 방향이 그 차의 방향이다 (실측 CAR_02
                    # heading 268.7 vs B1 주차방향 270.0 — 1.3도 차이).
                    seen_pose[slot_id] = (
                        float(view.position_mm[0]), float(view.position_mm[1]),
                        rear_parked_heading_deg(spec))
                    break

        confirm = int(getattr(
            self.config, "vision_occupancy_confirm_observations", 3))
        release = float(getattr(
            self.config, "vision_occupancy_release_s", 3.0))
        for slot_id in specs:
            st = state.setdefault(
                slot_id, {"hits": 0, "last_seen": None, "occupied": False})
            track_id = hits.get(slot_id)
            if track_id is not None:
                st["hits"] += 1
                st["last_seen"] = obs_time
                # 마지막으로 검증된 자세를 계속 갱신한다. _parked_obstacles 와
                # 같은 이유로, 검출이 잠깐 끊겨도 장애물이 사라지면 안 된다.
                if slot_id in seen_pose:
                    st["pose"] = seen_pose[slot_id]
                    if st["occupied"]:
                        self._vision_parked_obstacles[slot_id] = st["pose"]
                if not st["occupied"] and st["hits"] >= confirm:
                    st["occupied"] = True
                    setter(slot_id, True)
                    pose = st.get("pose")
                    if pose is not None:
                        self._vision_parked_obstacles[slot_id] = pose
                    self._emit_event("VISION_SLOT_OCCUPIED", slot=slot_id,
                                     track_id=track_id, observations=st["hits"],
                                     x_mm=None if pose is None else round(pose[0], 1),
                                     y_mm=None if pose is None else round(pose[1], 1),
                                     heading_deg=None if pose is None else round(pose[2], 1))
                    log.info("[VISION_OCCUPANCY] confirmed %s track=%s",
                             slot_id, track_id)
                continue
            # 이번 프레임에 안 보였다 — 확정 카운터만 되감고, 이미 확정된
            # 점유는 유예 시간이 지나야 푼다(bbox 한두 프레임 유실 대비, §13).
            st["hits"] = 0
            if (st["occupied"] and st["last_seen"] is not None
                    and obs_time - st["last_seen"] > release):
                st["occupied"] = False
                setter(slot_id, False)
                # 점유가 풀리면 정적 장애물도 같은 순간에 사라진다.
                self._vision_parked_obstacles.pop(slot_id, None)
                self._emit_event("VISION_SLOT_CLEARED", slot=slot_id,
                                 absent_s=round(obs_time - st["last_seen"], 2))
                log.info("[VISION_OCCUPANCY] cleared %s (%.1fs 미검출)",
                         slot_id, obs_time - st["last_seen"])

    def _record_pose(self, seen: list[VehicleView], state: TrackState,
                     t_recv: float) -> None:
        """이번 프레임의 pose 원본을 기록기에 넘긴다 (요청문 6절).

        같은 frame_id로 차량별 row를 각각 남긴다. car_id/track_id로 후처리해
        독립 시계열을 복원할 수 있으므로 두 번째 차량을 버리지 않는다.
        """
        self._frame_seq += 1
        if self._prev_frame_index is not None:
            gap = state.frame_index - self._prev_frame_index - 1
            if gap > 0:
                self._dropped_frames += gap
        self._prev_frame_index = state.frame_index

        if not seen:
            return
        for view in sorted(seen, key=lambda v: (
                v.car_id is None,
                v.car_id if v.car_id is not None else v.track_id)):
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
                **(getattr(self, "_last_perception", None) or {}),
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
            if view.track_id not in self._pose_observed_tracks:
                self._pose_observed_tracks.add(view.track_id)
                self._emit_event(
                    "POSE_OBSERVED", track_id=view.track_id,
                    car_id=view.car_id, x_mm=round(view.position_mm[0], 1),
                    y_mm=round(view.position_mm[1], 1),
                    heading_deg=(None if view.heading_deg is None
                                 else round(view.heading_deg, 1)))

    def _slot_occupancy(self):
        """배정/안전 판단이 보는 실효 점유 = base OR vision overlay.

        overlay 가 없는 allocator(테스트 스텁 등)에서는 base 를 그대로 쓴다 —
        기능이 없을 때의 동작은 production 과 완전히 같다.
        """
        return getattr(self.allocator, "effective_slot_statuses",
                       self.allocator.slot_statuses)

    def _emit_event(self, name: str, **fields: Any) -> None:
        if self.on_event_record is not None:
            self.on_event_record(name, **fields)

    def _trajectory_verdict(self, view: VehicleView, waypoints: list[Any],
                            slot_id: str | None):
        """Production safety verdict, **without** side effects.

        _trajectory_safe 가 이걸 쓰고 그 위에 실패 처리(zero/이벤트/로그)를
        얹는다. 분리한 이유는 "지금 자세에서 정상 경로가 가능한가" 를 부작용
        없이 물어봐야 하는 곳이 생겼기 때문이다 (ENTRY_STAGING 종료 판정).
        두 경로가 같은 판정을 쓰도록 강제하는 것이 목적이다 — 판정을 복제하면
        언젠가 갈라진다.
        """
        if (view.heading_deg is None
                or view.heading_source not in {"FRONT_CUSHION", "TRAJECTORY"}):
            return None, "NO_FRESH_HEADING"
        obstacles, uncertain = self._planning_obstacle_snapshot(view)
        if uncertain:
            return None, "OTHER_VEHICLE_POSE_UNCERTAIN"
        result = validate_trajectory(
            waypoints,
            start_pose=(view.position_mm[0], view.position_mm[1],
                        view.heading_deg),
            lot_size_mm=(self.config.lot_width_mm,
                         self.config.lot_height_mm),
            target_slot=slot_id, occupied_slots=tuple(
                sid for sid in SLOT_NAMES
                if self._slot_occupancy()[SLOT_NAMES.index(sid)] >= 0.5
                and sid != slot_id),
            obstacle_poses=obstacles,
            obstacle_margin_mm=float(getattr(
                self.config, "boundary_measurement_uncertainty_mm", 10.0)),
            min_turn_radius_mm=self._plan_radius,
            initial_boundary_tolerance_mm=self.config.boundary_hard_margin_mm,
            # 전/후진 전환점은 차가 실제로 멈추는 곳이다. 제어기가 아는 자기
            # 정지거리를 그대로 넘겨 "계획 중심이 맵 안" 이 아니라 "정지한
            # 차체가 맵 안" 으로 판정하게 한다 (실측 182908 route 13).
            stop_distance_mm=self._route_stop_distance_mm(),
            # planner 와 controller 가 같은 도달가능성을 쓰게 한다. 차가 처음
            # 향할 점이 최소 선회원 안이면 그 경로는 실행할 수 없다
            # (실측 230944/231157/231338 의 1-waypoint 인계 경로).
            require_reachable_first_wp=True)
        return result, result.reason

    def _trajectory_safe(self, view: VehicleView, waypoints: list[Any], *,
                         slot_id: str | None, recovery: bool = False) -> bool:
        """Single production gate used before every executable trajectory load."""
        result, reason = self._trajectory_verdict(view, waypoints, slot_id)
        if result is not None and result.safe:
            return True
        car_id = view.car_id
        if car_id is not None:
            self.server.stop_control(car_id)
        event = "RECOVERY_REJECTED" if recovery else "ROUTE_REJECTED"
        route_id = getattr(waypoints[0], "route_id", None) if waypoints else None
        self._emit_event(event, car_id=car_id, route_id=route_id,
                         slot=slot_id, reason=reason)
        self.dashboard.push_event(event.lower(), car_id=car_id,
                                  route_id=route_id, slot=slot_id,
                                  reason=reason)
        log.warning("car %s: %s before load (%s)", car_id, event, reason)
        return False

    def _planning_obstacle_snapshot(
            self, view: VehicleView,
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[int | None, ...]]:
        """Return static obstacle poses for one planning transaction.

        PARKED vehicles use their last verified pose and therefore survive
        perception/COMM dropout. Other detected vehicles must have a recent,
        body-trusted heading; an identified but uncertain vehicle makes a new
        route fail safe instead of silently disappearing from geometry.
        """
        poses: list[tuple[float, float, float]] = []
        uncertain: list[int | None] = []
        parked = getattr(self, "_parked_obstacles", {})
        included: set[int] = set()
        for car_id, pose in parked.items():
            if car_id == view.car_id:
                continue
            poses.append(pose)
            included.add(car_id)

        # ── vision 으로 확정된 정적 주차 차량 ─────────────────────────────
        #
        # 전원이 꺼진 채 슬롯에 세워둔 차는 rc_car 로는 매 프레임 잡히지만
        # front_cushion 이 거의 안 잡혀 heading_source 가 LAST_VALID 로 굳는다
        # (실측 run_20260906_190856 25.9% / _191008 4.7%). 그러면 아래의
        # trusted 검사에 걸려 uncertain 이 되고, _trajectory_verdict 가 **모든**
        # route/recovery 를 OTHER_VEHICLE_POSE_UNCERTAIN 으로 거절한다.
        # 두 run 다 정확히 그렇게 끝났다.
        #
        # 그런데 그 차는 "자세를 모르는 움직이는 차" 가 아니라 "그 칸에 세워둔
        # 차" 다. 그건 이 파이프라인이 이미 PARKED 차량에 쓰는 의미이고, 바로
        # 위 _parked_obstacles 가 그 표현이다 — 마지막으로 검증된 정적 자세를
        # 쓰고 heading 신선도를 요구하지 않는다. 같은 의미를 슬롯 단위로 쓴다.
        #
        # 장애물에서 빼는 것이 **아니다**. 실제 차체는 그대로 남고, 못 믿는
        # heading 한 축만 슬롯의 주차 방향으로 대체된다.
        vision_parked = dict(getattr(self, "_vision_parked_obstacles", {}))
        for pose in vision_parked.values():
            poses.append(pose)
        specs = default_slot_specs() if vision_parked else {}

        max_age = float(getattr(
            getattr(self.config, "controller_config", None),
            "max_pose_age_s", 0.5))
        now = view.last_obs_time
        # 같은 물리 차량의 중복/재획득 track 은 장애물이 아니다.
        #
        # self 제외가 track_id 하나뿐이라, tracker 가 같은 차를 다시 잡아 새 id
        # 를 주면 그 관측이 **자기 자신**인데도 "다른 차량"이 됐다. heading 이
        # 아직 없으므로 trusted 도 아니어서 uncertain 으로 분류되고, 그러면
        # _trajectory_verdict 가 모든 route/recovery 를 거절한다.
        #
        # 실측 run_20260904_230944: 실제 RC 카는 1대인데
        #     ego(track 1)  t=65.7  (304.7, 508.3)  heading 371건 전부 있음
        #     track 13/19/26/27/30/33/36/40/42  (300.7~306.1, 506.9~512.3)
        #                                       heading 0건, 각 1~2 관측
        # ego 와 ±5mm 안의 같은 자리다. t=63.48 에 route 1건 + recovery 2건이
        # 5ms 안에 전부 OTHER_VEHICLE_POSE_UNCERTAIN 으로 거절됐고 51초 정지.
        #
        # 판정 거리는 새 상수가 아니라 이 파이프라인이 이미 "같은 차로 본다" 는
        # 뜻으로 쓰는 track_rebind_max_distance_mm 다. 물리적으로도 안전하다 —
        # 250x150mm 차 두 대는 어떤 방향이어도 중심이 CAR_WIDTH_MM(150mm)
        # 안으로 들어올 수 없다(겹친다). 즉 이 거리 안의 관측은 다른 차량일 수
        # 없다.
        #
        # **다른 car_id 로 바인딩된 track 은 이 예외를 타지 않는다** — 실제 다른
        # 차량은 종전대로 장애물/불확실 판정 대상이다.
        same_vehicle_mm = float(getattr(
            self.config, "track_rebind_max_distance_mm", 150.0))
        for other in self.views.values():
            if other.track_id == view.track_id:
                continue
            if other.car_id is not None and other.car_id in included:
                continue
            if (other.car_id is None
                    and math.hypot(
                        other.position_mm[0] - view.position_mm[0],
                        other.position_mm[1] - view.position_mm[1])
                    <= same_vehicle_mm):
                continue
            # 이미 정적 장애물로 들어간 그 차다 (bound 차량은 대상이 아니다).
            # heading 이 FRONT_CUSHION 으로 되살아나도 같은 차를 static +
            # dynamic 으로 두 번 넣지 않는다.
            if other.car_id is None and any(
                    in_final_region(specs[sid], other.position_mm[0],
                                    other.position_mm[1])
                    for sid in vision_parked if sid in specs):
                continue
            age = now - other.last_obs_time
            fresh = (other.last_obs_time > 0.0 and age >= -0.05
                     and age <= max_age)
            trusted = (other.heading_deg is not None
                       and other.heading_source in {"FRONT_CUSHION", "TRAJECTORY"})
            if fresh and trusted:
                poses.append((other.position_mm[0], other.position_mm[1],
                              other.heading_deg))
            elif fresh or other.car_id is not None:
                uncertain.append(other.car_id)
        return tuple(poses), tuple(uncertain)

    def _planner_obstacle_poses(self, view: VehicleView
                                ) -> tuple[tuple[float, float, float], ...]:
        """Planner hint; the common load gate remains the final authority."""
        poses, _ = self._planning_obstacle_snapshot(view)
        return poses

    @staticmethod
    def _heading_delta(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _initial_pose_ready(self, view: VehicleView) -> bool:
        """서로 다른 fresh observation N개가 한 자세에 안정됐을 때만 True."""
        if view.last_obs_time <= 0.0:
            return False
        samples = self._initial_pose_samples.setdefault(
            view.track_id, deque(maxlen=self.config.initial_pose_observations))
        sample = (view.last_obs_time, view.position_mm[0], view.position_mm[1],
                  view.heading_deg)
        if samples and sample[0] <= samples[-1][0]:
            return False
        if samples:
            _, px, py, ph = samples[-1]
            if (math.hypot(sample[1] - px, sample[2] - py)
                    > self.config.initial_pose_stability_mm
                    or (sample[3] is not None and ph is not None
                        and self._heading_delta(sample[3], ph)
                        > self.config.initial_heading_stability_deg)):
                samples.clear()
        samples.append(sample)
        return len(samples) >= self.config.initial_pose_observations

    @staticmethod
    def _critical_heading_ready(view: VehicleView) -> bool:
        """A route-planning boundary needs a heading measured this frame."""
        return (view.heading_deg is not None
                and view.heading_source in {"FRONT_CUSHION", "TRAJECTORY"})

    def _require_critical_heading(self, view: VehicleView,
                                  boundary: str) -> bool:
        if not hasattr(self, "_heading_wait_state"):
            self._heading_wait_state = {}
        if not hasattr(self, "_heading_wait_started"):
            self._heading_wait_started = {}
        if not hasattr(self, "_heading_wait_faulted"):
            self._heading_wait_faulted = set()
        if self._critical_heading_ready(view):
            if view.car_id is not None:
                self._heading_wait_state.pop(view.car_id, None)
                self._heading_wait_started.pop(view.car_id, None)
                self._heading_wait_faulted.discard(view.car_id)
            return True
        initial_samples = getattr(self, "_initial_pose_samples", None)
        if initial_samples is not None:
            initial_samples.pop(view.track_id, None)
        car_id = view.car_id
        if car_id is not None:
            self.server.stop_control(car_id)
            # 대기 시계는 **"신뢰 가능한 heading 을 기다린다"** 는 사실에만
            # 달려 있다. 어느 호출부가 물었는지(boundary 이름)로 재시작하면
            # 안 된다 — 한 프레임에 두 호출부가 서로 다른 이름으로 물으면
            # 매번 시계가 0 으로 돌아가 timeout 이 영원히 오지 않는다.
            # 실측 run_20260827_234439: PARKING_RECOVERY_REPLAN 과
            # BOUNDARY_HEADING_UNCERTAIN 이 매 프레임 번갈아 물어 10.7초 동안
            # timeout 이 누적되지 않았고, 그래서 liveness 복구가 실행되지 않았다.
            already_waiting = car_id in self._heading_wait_started
            if self._heading_wait_state.get(car_id) != boundary:
                self._heading_wait_state[car_id] = boundary
                self._emit_event("WAIT_FOR_FRESH_HEADING", car_id=car_id,
                                 boundary=boundary,
                                 heading_source=view.heading_source)
            if not already_waiting:
                self._heading_wait_started[car_id] = view.last_obs_time
                self._heading_wait_faulted.discard(car_id)
            # 이 경계에서 heading 을 못 얻으면 bounded timeout 뒤 명시적 fault 로
            # 보내고, 거기서 기존 liveness 복구(HEADING_RECOVERED →
            # PARKING_REACTIVATED)가 다시 살려낸다. 여기 없는 경계는 timeout 이
            # 없어 **조용한 영구 zero** 가 된다 — 새 경계를 추가할 때 반드시
            # 같이 등록해야 한다.
            parking_boundaries = {
                "PARKING_PHASE_BOUNDARY", "PARKING_RECOVERY_REPLAN",
                "DIRECT_REAR_REPLAN", "PARKING_SETUP",
                "COMM_RECOVERY_REPLAN",
                "FINAL_POSE_EVAL", "BOUNDARY_HEADING_UNCERTAIN",
            }
            started = self._heading_wait_started.get(
                car_id, view.last_obs_time)
            timeout_s = float(getattr(
                self.config, "critical_heading_wait_timeout_s", 2.5))
            if (boundary in parking_boundaries
                    and view.last_obs_time - started >= timeout_s
                    and car_id not in self._heading_wait_faulted):
                self._heading_wait_faulted.add(car_id)
                self._parking_stage[car_id] = "WAIT_FRESH_HEADING_FAULT"
                # 물리 안전 fault 가 아니라 관측 대기다. 재활성 대상으로 표시해
                # heading 이 돌아오면 _maybe_resume_heading_fault 가 살려낸다.
                if not hasattr(self, "_heading_fault_hold"):
                    self._heading_fault_hold = set()
                self._heading_fault_hold.add(car_id)
                runner = self.auto_hosts.get(car_id)
                if runner is not None:
                    runner.stop()
                self._emit_event(
                    "WAIT_FOR_FRESH_HEADING_TIMEOUT", car_id=car_id,
                    boundary=boundary, timeout_s=timeout_s)
                self._emit_event(
                    "FAULT", car_id=car_id,
                    reason="FRESH_HEADING_TIMEOUT", boundary=boundary)
                self.dashboard.push_event(
                    "fresh_heading_timeout", car_id=car_id,
                    boundary=boundary, timeout_s=timeout_s)
        return False

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
        self._maybe_rebind_recovery_track(view, frame_index)
        return view

    def _maybe_rebind_recovery_track(self, view: VehicleView,
                                     frame_index: int) -> None:
        """Recover a detector track-id change at a stopped recovery boundary.

        This is intentionally narrower than general multi-car association:
        only an explicitly stopped planning boundary, one stale bound car, and
        a spatially nearby unbound track are eligible.  Heading is never copied;
        the existing critical-heading gate still requires a fresh source on
        the new track before any route can load.
        """
        def eligible(car_id: int) -> bool:
            if self._parking_stage.get(car_id) in {
                    "ENTRY_STAGING_PENDING", "SETUP_PENDING",
                    "PARKING_AFTER_SETUP_PENDING", "PARKING_HANDOFF_PENDING",
                    "WAIT_SAFE_RECOVERY"}:
                return True
            ctx = getattr(self, "_comm_recovery_context", {}).get(car_id)
            return bool(car_id in getattr(self, "_comm_lost", set()) and ctx is not None
                        and ctx.get("state") in {
                            "WAIT_CONNECTION", "WAIT_SESSION", "WAIT_FRESH_POSE",
                        })

        if (view.car_id is not None
                or not any(eligible(car_id) for car_id in self.track_of_car)):
            return
        stale_frames = int(getattr(
            self.config, "track_rebind_stale_frames", 8))
        max_distance = float(getattr(
            self.config, "track_rebind_max_distance_mm", 150.0))
        candidates: list[tuple[int, int, VehicleView]] = []
        for car_id, old_track_id in list(self.track_of_car.items()):
            if not eligible(car_id):
                continue
            old = self.views.get(old_track_id)
            if old is None or old.track_id == view.track_id:
                continue
            if frame_index - old.last_seen_frame < stale_frames:
                continue
            distance = math.hypot(
                view.position_mm[0] - old.position_mm[0],
                view.position_mm[1] - old.position_mm[1])
            if distance <= max_distance:
                candidates.append((car_id, old_track_id, old))
        if len(candidates) != 1:
            return

        car_id, old_track_id, old = candidates[0]
        with self._lock:
            if self.track_of_car.get(car_id) != old_track_id:
                return
            old.car_id = None
            view.car_id = car_id
            view.slot_id = old.slot_id or self._auto_host_slot.get(car_id)
            self.track_of_car[car_id] = view.track_id
            self._initial_pose_samples.pop(old_track_id, None)
            self.views.pop(old_track_id, None)

        vehicles = getattr(self.allocator, "vehicles", {})
        old_vehicle = vehicles.get(old_track_id)
        new_vehicle = vehicles.get(view.track_id)
        if old_vehicle is not None and new_vehicle is not None:
            new_vehicle.assigned_slot = old_vehicle.assigned_slot
            new_vehicle.route = list(old_vehicle.route)
        self.allocator.remove_vehicle(old_track_id)
        self.heading.remove(old_track_id)
        ctx = getattr(self, "_comm_recovery_context", {}).get(car_id)
        if ctx is not None:
            ctx["track_id"] = view.track_id
        self._emit_event(
            "TRACK_REBOUND", car_id=car_id, old_track_id=old_track_id,
            new_track_id=view.track_id,
            distance_mm=round(math.hypot(
                view.position_mm[0] - old.position_mm[0],
                view.position_mm[1] - old.position_mm[1]), 1),
            state=(ctx or {}).get("state", self._parking_stage.get(car_id)))

    # ─── track_id ↔ car_id 매핑 (순차 진입) ──────────────────────────────────

    def _on_vehicle_ready(self, car_id: int) -> None:
        """HELLO 승인 순서를 대기열에 넣는다 — 카메라 진입 순서와 대조."""
        if car_id in self._comm_recovery_context:
            self._start_comm_recovery_handshake(car_id)
            return
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
        if (self.config.direct_control and self.config.direct_control_set_mode
                and not self.auto_host_mode):
            self._request_remote_direct(car_id)

    def _request_remote_direct(self, car_id: int) -> None:
        """Set desired mode for legacy direct control without owning commands."""
        with self._lock:
            if car_id in self._mode_set or car_id in self._direct_mode_starting:
                return
            session = self._direct_sessions.get(car_id)
            if session is None:
                session = RemoteDirectSession(None, self.server, car_id)
                session.attach()
                self._direct_sessions[car_id] = session
            self._direct_mode_starting.add(car_id)

        def worker() -> None:
            try:
                session.ensure_remote_direct(
                    wait_s=self.config.auto_host_handshake_s)
                session._enable_direct_stream(release_control=True)
                with self._lock:
                    self._mode_set.add(car_id)
                log.info("car %d: REMOTE_DIRECT ready (legacy direct control)",
                         car_id)
            except Exception as exc:                    # noqa: BLE001
                log.warning("car %d: REMOTE_DIRECT request failed (%s)",
                            car_id, exc)
            finally:
                with self._lock:
                    self._direct_mode_starting.discard(car_id)

        threading.Thread(target=worker, name=f"remote-direct-{car_id}",
                         daemon=True).start()

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
            # A track may have accumulated an unverified heading before it was
            # bound to the physical car.  Binding starts a new planning epoch.
            self.heading.remove(view.track_id)
            self._initial_pose_samples.pop(view.track_id, None)
            view.heading_deg = None
            view.heading_source = None
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
        # A COMM hold owns execution until session renegotiation and a distinct
        # fresh observation complete.  In particular, do not allocate a brand
        # new mission behind a transport latch (175307: RUNNING + silent zero).
        if (car_id in getattr(self, "_comm_lost", set())
                or car_id in getattr(self, "_comm_recovery_context", {})):
            self._allocation_state[car_id] = "WAIT_COMM_RECOVERY"
            self.server.stop_control(car_id)
            return
        if self.auto_host_mode:
            active_peer = next((
                peer for peer in getattr(self, "_auto_host_slot", {})
                if peer != car_id
                and self._parking_stage.get(peer)
                    not in self._PARKING_TERMINAL_STAGES
            ), None)
            if active_peer is not None:
                # Level 1/2 production policy: at most one AUTO_HOST vehicle
                # moves. The waiting car remains visible/bound but receives no
                # slot or route until the active peer reaches a terminal state.
                self._allocation_state[car_id] = "WAIT_OTHER_VEHICLE"
                self.server.stop_control(car_id)
                return
        if not self._require_critical_heading(view, "INITIAL_ALLOCATION"):
            self._allocation_state[car_id] = "WAIT_FOR_FRESH_HEADING"
            return
        if not self._initial_pose_ready(view):
            self._allocation_state[car_id] = "WAIT_STABLE_POSE"
            self.server.stop_control(car_id)
            return
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
        if (self.auto_host_mode and self.rear_parking_mode
                and self._entry_staging_needed(view, slot_id)):
            if self._start_entry_staging(
                    car_id, view, slot_id, route_id, initial=True):
                return
            return
        # 접근 방향(좌/우)과 진입 선회 반경은 차량 현재 위치가 정한다. 슬롯에
        # 따라 "지나쳐 버려서 전진으로 못 돌아오는" 경우가 있으므로 반드시
        # strict 로 만들고, 불가능하면 무장하지 않는다 — 예전에는 그대로 실어
        # 보내서 차가 뒤로 가야 하는 경로를 받았다.
        slot_id, wps = self._feasible_route(view, slot_id, route_id)
        if slot_id is None:
            return                       # 갈 수 있는 칸이 없다 — 다음 프레임 재시도

        if self.auto_host_mode:
            if not self._start_auto_host(car_id, slot_id, wps, view=view):
                return
            view.slot_id = slot_id
            self._allocation_state[car_id] = "ROUTE_LOADED"
            self._emit_event("SLOT_SELECTED", car_id=car_id, slot=slot_id,
                             route_id=route_id)
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
        self._allocation_state[car_id] = "ROUTE_LOADED"
        self._emit_event("SLOT_SELECTED", car_id=car_id, slot=slot_id,
                         route_id=route_id)
        log.info("car %d → slot %s (route %d, %d waypoints)",
                 car_id, slot_id, route_id, len(wps))
        self.dashboard.push_event("slot_assigned", car_id=car_id, slot=slot_id,
                                  route_id=route_id)

    def _feasible_route(self, view: VehicleView, slot_id: str, route_id: int):
        """Build the route for the **assigned** slot. Never picks another one.

        allocator 가 고른 칸은 soft preference 가 아니라 예약된 미션 목표다.
        경로가 안 나오는 것은 "그 칸에 갈 자세를 다시 만드는" 문제이지 "다른
        칸을 고르는" 문제가 아니다 — 자세는 언제든 바꿀 수 있지만, 슬롯을
        바꾸면 그 미션은 원래 목표로 돌아오지 못한다.

        예전에는 이 함수가 두 책임을 겸했다: 경로 생성과 슬롯 재선택. 그래서
        모든 주행 실패가 재배정으로 번역됐다. 실측 run_20260904_214712:

            t=24.53  staging 예산 3/3 소진 -> 완료 분기 -> 여기 호출
            t=24.54  B1 거절 -> A1 거절 -> A2 거절 -> B2 거절 -> A3 채택
            t=24.55  slot_id 가 B1 -> A3 로 바뀜 (control.jsonl)

        allocator.allocate() 는 반환하기 전에 이미 외부 상태를 쓴다:
        `v.assigned_slot = slot`, `v.route = ...`, `set_slot_occupied(slot)`
        (이중 할당 방지 선점). 즉 **그 칸을 돌려준 순간부터 그것은 예약된 미션
        목표**다. 그 뒤에 주행 경로가 안 나온다는 이유로 다른 칸을 고르면,
        allocator 가 기록한 배정을 주행 계층이 뒤집는 것이다.

        "지금 자세에서 직접 경로가 없다" 와 "그 칸이 실제로 도달 불가하다" 는
        다르다. 앞의 것은 자세를 다시 만들어 풀 문제이고, 자세는 언제든 바꿀 수
        있다. 칸을 바꾸면 그 미션은 원래 목표로 돌아오지 못한다.

        실패는 (None, None) 이고 호출부는 그것을 정지 + 예약 유지로 처리한다.
        점유/차단으로 칸을 바꾸는 것은 allocator.allocate 의 책임이지 주행
        실패의 결과가 아니다.

        Returns:
            (예약 슬롯, waypoint 목록). 지금 자세에서 못 만들면 (None, None).
        """
        self._emit_event("SLOT_CANDIDATE", car_id=view.car_id,
                         slot=slot_id, route_id=route_id,
                         x_mm=round(view.position_mm[0], 1),
                         y_mm=round(view.position_mm[1], 1),
                         heading_deg=view.heading_deg)
        try:
            wps = self._build_route(default_slot_specs()[slot_id], view,
                                    route_id)
        except InfeasibleRouteError as exc:
            reason = exc.reason
        else:
            if self._trajectory_safe(view, wps, slot_id=slot_id):
                return slot_id, wps
            reason = "TRAJECTORY_SAFETY_REJECTED"
        # 칸을 버리는 것이 아니라 **지금 자세로는 못 간다**는 기록이다.
        self._reject_slot(view.car_id, slot_id, reason)
        self._emit_event("SLOT_ROUTE_UNAVAILABLE", car_id=view.car_id,
                         slot=slot_id, reason=reason, route_id=route_id)
        self._allocation_state[view.car_id] = "WAIT_NO_FEASIBLE_SLOT"
        self.server.stop_control(view.car_id)
        self._emit_event("SLOT_WAIT", car_id=view.car_id,
                         state="WAIT_NO_FEASIBLE_SLOT",
                         reason=f"{slot_id}: {reason}")
        self._warn_no_route(view, f"{slot_id}: {reason}")
        return None, None

    @property
    def rear_parking_mode(self) -> bool:
        return getattr(self.config, "parking_mode", "handoff") == "rear"

    def _entry_staging_heading(self, view: VehicleView,
                               slot_id: str | None) -> float | None:
        """Aisle travel direction, taken from the handoff planner's own rule.

        예전에는 `0.0 if spec.center_x >= view.x else 180.0` — 즉 **슬롯이
        차보다 뒤에 있으면 180° 로 달리라**고 요구했다. 그런데 이 프로젝트는
        U턴을 못 한다(선회 지름이 맵 한 변과 맞먹는다). plan_handoff 가 바로
        그 이유로 "가던 방향이 곧 접근 방향"을 이미 구현하고 있는데, staging
        만 위치 부호로 방향을 정해 두 규약이 어긋나 있었다.

        실측 결과 (3/3 재현):
            run_20260904_164904 t=35.4  (581.6,528.4,359.5°)  선호 슬롯 B1
            run_20260904_165710 t=23.5  (515.4,491.6,  3.0°)  선호 슬롯 B1
        둘 다 통로축에 0.5°/3.0° 로 거의 완벽히 정렬돼 있었지만, B1(x=425)을
        이미 지나쳤다는 이유만으로 desired=180° 가 되어 heading 오차가
        179.5°/177.0° 로 계산됐다. aligned=False → staging 완료 대신 **새
        staging 시도**가 걸렸고, 164904 는 그 시점에 A2/B2/A3/B3 네 슬롯이
        이미 routable 이었는데도 다시 기동을 돌았다.

        plan_handoff 의 approach_heading_deg 를 그대로 쓴다 — 새 임계값도,
        새 기하도 없다. 입구(heading 90°)에서는 종전과 같이 0.0 이 나온다.
        """
        if slot_id is None:
            return None
        spec = default_slot_specs().get(slot_id)
        if spec is None:
            return None
        plan = plan_handoff(spec, from_pose=view.position_mm,
                            from_heading_deg=view.heading_deg)
        return float(plan.approach_heading_deg)

    def _entry_staging_ready(self, view: VehicleView,
                             slot_id: str | None = None) -> bool:
        """Whether GLOBAL can start from a measured aisle-aligned pose."""
        if view.heading_deg is None:
            return False
        desired = self._entry_staging_heading(view, slot_id)
        heading_error = (min(self._heading_delta(view.heading_deg, 0.0),
                             self._heading_delta(view.heading_deg, 180.0))
                         if desired is None else
                         self._heading_delta(view.heading_deg, desired))
        return (abs(view.position_mm[1] - AISLE_Y)
                <= ON_AISLE_TOLERANCE_MM
                and heading_error <= float(
                    self.config.entry_staging_heading_tolerance_deg))

    def _entry_staging_heading_aligned(self, view: VehicleView,
                                      slot_id: str | None) -> bool:
        """통로 진행축 기준 heading 만 본다 (위치는 planner 가 본다).

        _entry_staging_ready 는 통로 밴드(|y-600|<=80)와 heading 을 함께
        묶는다. staging 을 **시작할지** 정하는 데는 그게 맞지만, staging 을
        **끝낼지** 정하는 데 그대로 쓰면 위치 조건이 두 번 걸린다 —
        _handoff_feasible 이 이미 실제 경로를 만들어 boundary/충돌까지
        검증하기 때문이다. 그리고 그 중복은 무해하지 않다: 입구에서
        유일하게 PARKED_OK 로 끝난 run_20260903_022217 의 인계 자세
        (478.0, 515.2, 13.3도) 는 heading 이 13.3도로 좋은 basin 안인데
        |y-600|=84.8mm 로 밴드에서 4.8mm 벗어나 있다. 밴드를 종료 조건에
        넣으면 유일한 성공 사례를 거부한다.
        """
        if view.heading_deg is None:
            return False
        desired = self._entry_staging_heading(view, slot_id)
        if desired is None:
            error = min(self._heading_delta(view.heading_deg, 0.0),
                        self._heading_delta(view.heading_deg, 180.0))
        else:
            error = self._heading_delta(view.heading_deg, desired)
        return error <= float(
            self.config.entry_staging_heading_tolerance_deg)

    def _handoff_feasible(self, view: VehicleView,
                          slot_id: str | None) -> str | None:
        """정상 parking flow 가 **지금 이 자세에서** 인수할 수 있는가.

        기하 근사가 아니라 production planner 에게 직접 묻는다. 반환값은
        그 planner 가 실제로 쓸 슬롯이고, 없으면 None 이다.

        읽기 전용이다 — 슬롯을 unreachable 로 표시하지 않고, 이벤트도 내지
        않고, 제어에 손대지 않는다. _feasible_route 는 실패 시 그 셋을 전부
        하므로 "될까?" 를 묻는 용도로 쓸 수 없다. 후보 순서와 판정 자체는
        _feasible_route 와 동일하게 유지한다.
        """
        # 예약 슬롯 하나만 묻는다. _feasible_route 와 같은 판정을 유지한다는
        # 이 함수의 목적은 그대로이고, 그쪽이 더 이상 후보를 순회하지 않으므로
        # 여기서도 순회하지 않는다. 다른 칸이 된다는 답은 이 게이트에서 쓸
        # 수 없다 — staging 종료 판정이 곧 재배정이 되어 버린다(214712).
        if slot_id is None or view.heading_deg is None:
            return None
        try:
            wps = self._build_route(default_slot_specs()[slot_id], view, 0)
        except InfeasibleRouteError:
            return None
        if not wps:
            return None
        result, _reason = self._trajectory_verdict(view, wps, slot_id)
        return slot_id if (result is not None and result.safe) else None

    def _entry_staging_needed(self, view: VehicleView,
                              slot_id: str | None = None) -> bool:
        """Entrance starts outside the direct handoff planner's valid basin."""
        return (view.node == "entrance"
                and not self._entry_staging_ready(view, slot_id))

    def _entry_staging_candidate_slots(self, preferred: str) -> list[str]:
        specs = default_slot_specs()
        others = [sid for sid in specs if sid != preferred
                  and self._slot_occupancy()[SLOT_NAMES.index(sid)] < 0.5]
        return [preferred, *others]

    def _preferred_slot_needs_reposition(self, view: VehicleView,
                                         slot_id: str) -> bool:
        """Whether the preferred slot is *directly* infeasible right now.

        _feasible_route 와 같은 판정을 쓴다 (같은 _build_route + 같은
        _trajectory_verdict). 부작용은 없다 — 슬롯을 거부하지도, 이벤트를
        내지도 않는다.
        """
        if slot_id is None or slot_id not in default_slot_specs():
            return False
        try:
            wps = self._build_route(default_slot_specs()[slot_id], view, 0)
        except InfeasibleRouteError:
            return True
        if not wps:
            return True
        result, _reason = self._trajectory_verdict(view, wps, slot_id)
        return not (result is not None and result.safe)

    def _entry_staging_goal(self, view: VehicleView, slot_id: str | None):
        """The one staging completion contract: aisle band **and** alignment.

        staging 이 만들려는 자세와 staging 이 끝났다고 인정하는 자세는 같은
        사실이어야 한다. 예전에는 둘이 달랐다 — 경로 생성은 이 술어를 썼고,
        SLOT_REPOSITION 은 `plan_handoff(...).feasible` 을 썼다. 그런데
        plan_handoff 의 on-aisle 분기는 |y-600|<=80 과 "인계점이 뒤인가" 만
        보고 **heading 을 보지 않는다**. 그래서 재배치가 통로 밴드만 겨우
        넘는 61~122mm 짜리 최소 기동을 만들고 끝냈다.

        실측 run_20260905_024023 / _024123:
            생성된 재배치 목표 heading 59~88도, 실제 도달 자세 73~80도
            (통로축 목표 0도, 허용 15도)
            024123 은 재배치 뒤 heading 이 73.2 -> 80.2 로 오히려 나빠졌다
        그 자세는 곧바로 aligned 게이트에서 거부되어 다시 재배치를 부르고,
        그때마다 staging 예산만 소모됐다.

        여기서는 새 술어를 만들지 않는다 — _build_entry_staging_route 가 이미
        쓰던 그 goal 을 꺼내서 재배치와 공유할 뿐이다. 임계값도 그대로
        (ON_AISLE_TOLERANCE_MM, entry_staging_heading_tolerance_deg).
        """
        heading_tol = float(self.config.entry_staging_heading_tolerance_deg)
        desired_heading = self._entry_staging_heading(view, slot_id)
        spec = None if slot_id is None else default_slot_specs().get(slot_id)

        def goal(pose: tuple[float, float, float]) -> bool:
            if abs(pose[1] - AISLE_Y) > ON_AISLE_TOLERANCE_MM:
                return False
            if desired_heading is not None and (
                    self._heading_delta(pose[2], desired_heading)
                    > heading_tol):
                return False
            # staging 종료 게이트는 aligned 뿐 아니라 "그 칸으로 갈 수 있는가"
            # 도 함께 본다. 목표에도 같은 조건을 넣어야 만들어 낸 자세가 곧바로
            # 종료 판정을 통과한다 — 예전 재배치 목표가 쓰던 그 조건이다.
            return spec is None or plan_handoff(
                spec, from_pose=(pose[0], pose[1]),
                from_heading_deg=pose[2]).feasible
        return goal

    def _parking_reposition_goal(self, slot_id: str):
        """Parking-stage goal: 'a bounded move from which THIS slot hands off'.

        주차 단계(_load_slot_reposition)는 차가 이미 인계점 근처에 있고, 목적이
        "그 칸으로 다시 갈 수 있는 자세"다. 입구 staging 의 통로 밴드 조건과는
        다른 사실이므로 술어를 따로 둔다 — 이름으로 소유권을 분명히 한다.
        """
        spec = default_slot_specs()[slot_id]

        def goal(pose: tuple[float, float, float]) -> bool:
            return plan_handoff(spec, from_pose=(pose[0], pose[1]),
                                from_heading_deg=pose[2]).feasible
        return goal

    def _slot_reposition_available(self, view: VehicleView,
                                   slot_id: str) -> bool:
        """Whether a bounded maneuver can restore this slot's handoff.

        "지금 이 자세에서 직접 경로가 없다" 와 "그 슬롯이 실제로 도달
        불가하다" 를 가르는 질문이다. 판정자는 이미 쓰고 있는 두 가지뿐이다:
        기존 bounded planner(build_setup_recovery_waypoints)와 기존
        plan_handoff. 새 planner 도, 새 상수도 만들지 않는다.

        실측 (수정 후 baseline 4/4, B1 이 거부된 바로 그 자세):
            182908 (184.2,497.7,346.9°) -> 5wp -> B1 GLOBAL route 2wp
            183055 (150.3,501.6,355.1°) -> 2wp -> B1 GLOBAL route 2wp
            183319 (150.3,487.2,359.4°) -> 2wp -> B1 GLOBAL route 2wp
            183503 (142.6,474.0,  2.1°) -> 2wp -> B1 GLOBAL route 1wp
        3/4 는 좌선회 20~21도, 호 213~224mm 하나면 됐다. 그런데도 네 run 모두
        B1/A1 을 버리고 A2 로 갔다.
        """
        if view.heading_deg is None or slot_id not in default_slot_specs():
            return False
        return bool(build_setup_recovery_waypoints(
            default_slot_specs()[slot_id], route_id=0,
            from_pose=view.position_mm, from_heading_deg=view.heading_deg,
            radii_mm=STAGING_RADIUS_CANDIDATES,
            obstacle_poses=self._planner_obstacle_poses(view),
            min_executable_mm=self._setup_min_executable_mm(),
            goal_test=self._entry_staging_goal(view, slot_id),
            max_total_length_mm=float(
                self.config.entry_staging_alignment_max_mm),
            min_clearance_mm=float(
                self.config.entry_staging_min_clearance_mm),
            max_segments=4,
            segment_heading_tolerance_deg=float(
                self.config.entry_staging_heading_tolerance_deg)))

    def _build_entry_staging_route(self, car_id: int, view: VehicleView,
                                   slot_id: str, route_id: int,
                                   goal_test: Any = None) -> list[Any]:
        """Build one productive entrance maneuver, never a fixed map waypoint."""
        heading_tol = float(self.config.entry_staging_heading_tolerance_deg)

        # 통로 진입과 통로 정렬을 **한 목표로** 푼다.
        #
        # 예전에는 통로 밖이면 y 밴드만 요구하고(`if not on_aisle: return True`)
        # heading 은 다음 route 로 미뤘다. 그러면 1단계가 통로에 도달하되
        # heading 이 70도 틀어진 자세를 고르고, 2단계는 폭 80mm 통로 밴드
        # 안에서 좌벽에 붙은 채 70도를 돌려야 한다 — offline 확인 결과 그
        # 2단계는 어떤 최소 여유에서도 해가 없고, 여유 없는 3-point turn 만
        # 남아서 231000 에서 차가 맵 밖으로 나갔다. heading 을 처음부터
        # 목표에 넣으면 planner 가 통로 도달 자세 자체를 정렬 가능한 쪽으로
        # 고른다.
        #
        # 그 술어는 이제 _entry_staging_goal 하나에만 있다 — SLOT_REPOSITION 과
        # 같은 사실을 판정하기 위해서다.
        goal = goal_test or self._entry_staging_goal(view, slot_id)
        wps = build_setup_recovery_waypoints(
            default_slot_specs()[slot_id], route_id=route_id,
            from_pose=view.position_mm, from_heading_deg=view.heading_deg,
            radii_mm=STAGING_RADIUS_CANDIDATES,
            obstacle_poses=self._planner_obstacle_poses(view),
            min_executable_mm=self._setup_min_executable_mm(),
            goal_test=goal,
            max_total_length_mm=float(
                self.config.entry_staging_alignment_max_mm),
            min_clearance_mm=float(
                self.config.entry_staging_min_clearance_mm),
            # 입구는 3 구간으로는 부족하다. 실측 자세 (133,209,95도) 에서
            # 3 구간은 B1 하나만 풀리고 나머지 7 슬롯이 전부 infeasible 인데,
            # 4 구간이면 8 슬롯 모두 풀리고 여유도 35.4 -> 43.1mm 로 늘어난다.
            # 5 구간은 아무것도 더 얻지 못한다.
            #
            # 대가는 계획 시간이다 (실측 3.4~4.2s, 3 구간은 1.7~2.2s). 차가
            # 정지해 있을 때 한 번 도는 계획이지만, 이 시간 동안 host 가
            # DIRECT_CONTROL 을 못 보내면 펌웨어 500ms 타임아웃이 먼저 걸린다.
            # 실차 전에 벤치에서 확인할 것.
            max_segments=4,
            # 구간 경계 heading 게이트. 기본값 5° 는 후진 원호 진입점
            # (REVERSE_START) 용으로 교정된 값이라 입구 staging 구간 경계에
            # 그대로 쓸 근거가 없다. staging 계열이 이미 갖고 있는
            # entry_staging_heading_tolerance_deg 를 그대로 쓴다 — 새 숫자가
            # 아니고, staging 완료 판정(_entry_staging_heading_aligned)과도
            # 같은 눈금이 된다.
            segment_heading_tolerance_deg=heading_tol)
        if not wps or self._setup_is_degenerate(view, wps):
            return []
        end = wps[-1]
        signature = (
            round(view.position_mm[0] / 25.0),
            round(view.position_mm[1] / 25.0),
            round(float(view.heading_deg) / 5.0),
            round(float(end.x) / 25.0),
            round(float(end.y) / 25.0),
            round(float(end.target_heading_deg or 0.0) / 5.0),
        )
        seen = self._entry_staging_signatures.setdefault(car_id, set())
        if signature in seen:
            self._emit_event("ENTRY_STAGING_REJECTED", car_id=car_id,
                             slot=slot_id, route_id=route_id,
                             reason="REPEATED_IDENTICAL_STAGING")
            return []
        if not self._trajectory_safe(view, wps, slot_id=slot_id,
                                     recovery=True):
            return []
        seen.add(signature)
        return wps

    def _entry_staging_fault(self, car_id: int, reason: str) -> None:
        self._parking_stage[car_id] = "WAIT_ENTRY_STAGING_FAILED"
        self._allocation_state[car_id] = "WAIT_ENTRY_STAGING_FAILED"
        runner = self.auto_hosts.get(car_id)
        if runner is not None:
            runner.stop()
        self.server.stop_control(car_id)
        self._emit_event("FAULT", car_id=car_id, reason=reason,
                         attempts=self._entry_staging_attempts.get(car_id, 0))
        self.dashboard.push_event("entry_staging_failed", car_id=car_id,
                                  reason=reason)

    def _start_entry_staging(self, car_id: int, view: VehicleView,
                             preferred_slot: str, route_id: int, *,
                             initial: bool = False,
                             commit_slot: bool = False) -> bool:
        """Ask for one staging maneuver; the search itself runs off-thread.

        후보 탐색(`_build_entry_staging_route` → bounded beam search)은 실측
        후보 1개당 7.4~9.1초가 걸린다. 그 호출이 카메라 콜백 스레드에서 그대로
        돌면 프레임 소비 자체가 멈춘다 — run_20260904_164904/_165419/_165710 의
        pose 공백(3.6~17.5초, 합계 66.2/24.3/37.5초)이 전부 이 구간이었고,
        공백 시작 시각은 ENTRY_STAGING_CANDIDATE 시각과 일치한다.

        그래서 **계획만** 배경 스레드로 옮긴다. 적재/상태 전이는 종전대로
        프레임 스레드(`_apply_entry_staging_plan`)에서 일어난다. 구조는 이
        파일에 이미 있는 comm 재협상 워커와 같다: `self._lock` 아래 진행중
        표시 → daemon 스레드 → 완료 후 신원 재확인 → finally 에서 표시 해제.
        POSE_STALE zero 안전장치는 그대로다 — 계획 동안 차는 서 있는다.
        """
        attempts = self._entry_staging_attempts.get(car_id, 0)
        if attempts >= int(self.config.max_entry_staging_attempts):
            self._entry_staging_fault(car_id, "ENTRY_STAGING_EXHAUSTED")
            return False
        return self._request_entry_staging_plan(
            car_id, view, preferred_slot, route_id, initial=initial,
            commit_slot=commit_slot)

    def _request_entry_staging_plan(self, car_id: int, view: VehicleView,
                                    preferred_slot: str, route_id: int, *,
                                    initial: bool,
                                    commit_slot: bool = False) -> bool:
        """Start (or keep) one background staging search for this car."""
        with self._lock:
            if car_id in self._entry_staging_planning:
                return True              # 이미 계획 중 — 프레임을 계속 점유한다
            if car_id in self._entry_staging_plan:
                return True              # 결과 대기 — 다음 프레임이 적재한다
            self._entry_staging_planning.add(car_id)
        # 계획이 끝날 때까지 차는 서 있어야 한다. 이 시점의 미션은 이미
        # REPLAN_REQUIRED/EMPTY 이지만, 명시적으로 zero 를 건다.
        self.server.stop_control(car_id)
        pose = (float(view.position_mm[0]), float(view.position_mm[1]))
        heading = view.heading_deg
        attempts = self._entry_staging_attempts.get(car_id, 0)
        stage = self._parking_stage.get(car_id)
        # staging 은 **선호 슬롯 하나만** 본다.
        #
        # allocator 가 고른 슬롯은 soft preference 가 아니라 예약된 미션
        # 목표다. 경로 실패는 "그 슬롯에 갈 자세를 다시 만드는" 문제이지
        # "다른 슬롯을 고르는" 문제가 아니다. 후보를 순회하면 그 순회 자체가
        # 재배정이 된다 — 실측 run_20260904_210321: 입구에서 B1/A1/A2/A3 이
        # 차례로 NO_SAFE_PRODUCTIVE_MANEUVER 로 거절되고 A4 가 선택돼
        # (slot_id 가 t=25.8~78.1 전 구간 A4) 미션 내내 B1 로 돌아오지
        # 못했다. B1 은 점유도, 장애물도, 도달 불가도 아니었다.
        #
        # 해가 없으면 _entry_staging_fault 가 SAFE STOP 을 건다. 그 경로는
        # allocator 를 건드리지 않으므로 슬롯 예약은 그대로 유지된다.
        candidates = [preferred_slot]
        # commit_slot 재배치도 staging 과 **같은 종료 계약**을 목표로 한다.
        # _build_entry_staging_route 의 기본 goal 이 곧 그 계약이므로 별도
        # 오버라이드가 필요 없다 (예전에는 여기서 plan_handoff 기반 goal 로
        # 갈아끼워 두 경로가 서로 다른 사실을 판정했다).
        goal_test = None

        def worker() -> None:
            result: dict[str, Any] = {
                "route_id": route_id, "initial": initial,
                "preferred_slot": preferred_slot, "pose": pose,
                "heading_deg": heading, "attempts": attempts, "stage": stage,
                "slot": None, "waypoints": None, "rejected": [],
            }
            try:
                for candidate in candidates:
                    wps = self._build_entry_staging_route(
                        car_id, view, candidate, route_id,
                        goal_test=goal_test)
                    if not wps:
                        result["rejected"].append(candidate)
                        continue
                    result["slot"] = candidate
                    result["waypoints"] = wps
                    break
            except Exception as exc:                 # noqa: BLE001
                result["error"] = str(exc)
            finally:
                with self._lock:
                    self._entry_staging_plan[car_id] = result
                    self._entry_staging_planning.discard(car_id)

        threading.Thread(target=worker, name=f"entry-staging-{car_id}",
                         daemon=True).start()
        return True

    def _apply_entry_staging_plan(self, view: VehicleView) -> bool:
        """Load a finished background staging plan; True while owning the frame."""
        car_id = view.car_id
        if car_id is None:
            return False
        with self._lock:
            if car_id in self._entry_staging_planning:
                return True              # 계획 중 — 차는 정지 상태로 대기
            result = self._entry_staging_plan.pop(car_id, None)
        if result is None:
            return False
        # 계획이 도는 동안 상태가 바뀌었으면 그 결과는 버린다. 낡은 계획을
        # 새 pose 에 적용하지 않는다 (comm 재협상 워커의 신원 재확인과 같다).
        if (self._parking_stage.get(car_id) != result["stage"]
                or self._entry_staging_attempts.get(car_id, 0)
                != result["attempts"]):
            return False
        # 계획 입력 자세에서 차가 도착 반경 이상 움직였으면 역시 버린다.
        moved = math.hypot(view.position_mm[0] - result["pose"][0],
                           view.position_mm[1] - result["pose"][1])
        if moved > self._setup_min_executable_mm():
            return False
        for rejected in result["rejected"]:
            self._emit_event("ENTRY_STAGING_CANDIDATE", car_id=car_id,
                             slot=rejected, route_id=result["route_id"])
            self._emit_event("ENTRY_STAGING_REJECTED", car_id=car_id,
                             slot=rejected, route_id=result["route_id"],
                             reason="NO_SAFE_PRODUCTIVE_MANEUVER")
        wps = result.get("waypoints")
        if not wps:
            self._entry_staging_fault(car_id, "NO_SAFE_ENTRY_STAGING")
            return True
        self._emit_event("ENTRY_STAGING_CANDIDATE", car_id=car_id,
                         slot=result["slot"], route_id=result["route_id"])
        return self._apply_entry_staging_route(
            car_id, view, result["slot"], result["preferred_slot"],
            result["route_id"], wps, initial=result["initial"])

    def _apply_entry_staging_route(self, car_id: int, view: VehicleView,
                                   candidate: str, preferred_slot: str,
                                   route_id: int, wps: list[Any], *,
                                   initial: bool) -> bool:
        """Load one safe, distinct entrance staging maneuver."""
        attempts = self._entry_staging_attempts.get(car_id, 0)
        # staging 후보는 예약 슬롯 하나뿐이므로(_request_entry_staging_plan)
        # 여기서 슬롯이 바뀔 일이 없다. 남아 있던 재배정 분기는 계약상
        # 도달 불가이자 위반 경로라 제거한다 — 예약은 allocator 가 소유한다.
        assert candidate == preferred_slot, "staging 은 예약 슬롯만 싣는다"
        if initial:
            if not self._start_auto_host(car_id, candidate, wps, view=view):
                return False
        else:
            runner = self.auto_hosts.get(car_id)
            if runner is None:
                self._entry_staging_fault(car_id,
                                          "ENTRY_STAGING_RUNNER_MISSING")
                return False
            runner.load_route(wps)
        # ── 예산은 "실패한 staging 기동" 을 센다 ────────────────────────────
        #
        # 관측이 끊겨 멈췄다가 같은 staging 목표로 돌아오는 것은 새로운 실패가
        # 아니다. 그런데 그 재진입도 여기까지 와서 예산을 하나 태우고 있었다.
        #
        # 실측 run_20260905_024023 / _024123 (둘 다 동일):
        #     t= 7    staging 적재 (attempt 1)
        #     t= 9    FAULT POSE_STALE          <- 카메라 공백
        #     t=12    HEADING_RECOVERED -> SLOT_REPOSITION (attempt 2)
        #     t=15    FAULT POSE_STALE          <- 카메라 공백
        #     t=17    HEADING_RECOVERED -> SLOT_REPOSITION (attempt 3)
        #     t=21    ENTRY_STAGING_EXHAUSTED   <- 21초 만에 미션 종료
        # 즉 프레임 두 번 놓친 것이 예산 전체를 소모했다. known-good
        # 022217/183055/183503 은 staging 적재 1회 / 재배치 0회였다.
        #
        # 구분 근거는 이미 있는 관측-latch 계약이다: 관측 지연으로 멈춘 차는
        # _heading_fault_hold 에 들어 있고, 검증된 새 route 가 실릴 때
        # _reactivate_after_heading_fault 가 그 표시를 지운다(바로 아래).
        # 그러므로 이 시점의 표시 유무가 곧 "이 적재가 중단 복구인가, 새
        # 기동 시도인가" 다. 새 state 도 새 임계값도 없다.
        #
        # 예산 기능 자체는 그대로다 — 실제로 기동이 끝났는데 목표를 못 만든
        # 경우(hold 없음)는 종전대로 하나씩 소모하고, 3회면
        # ENTRY_STAGING_EXHAUSTED 로 안전 정지한다.
        resumed = car_id in getattr(self, "_heading_fault_hold", ())
        if not resumed:
            self._entry_staging_attempts[car_id] = attempts + 1
        self._auto_host_slot[car_id] = candidate
        view.slot_id = candidate
        self._parking_stage[car_id] = "ENTRY_STAGING"
        self._allocation_state[car_id] = "ENTRY_STAGING_LOADED"
        self._emit_event("SLOT_SELECTED", car_id=car_id, slot=candidate,
                         route_id=route_id)
        # 로그의 attempt 는 실제 예산 값이어야 한다 — 중단 복구 재적재는
        # 예산을 쓰지 않으므로 값이 그대로다. resumed 표시도 함께 남겨
        # 로그만 보고도 두 경우를 구분할 수 있게 한다.
        self._emit_event("ENTRY_STAGING_LOADED", car_id=car_id,
                         slot=candidate, route_id=route_id,
                         attempt=self._entry_staging_attempts.get(car_id, 0),
                         resumed=resumed)
        # 관측 latch 로 authority 가 잠긴 채 여기까지 왔다면, 검증된 새 경로가
        # 실린 지금이 그 계약의 재무장 지점이다 (_load_parking_setup 과 동일).
        # hold 에 없는 차에게는 아무 일도 하지 않는다.
        self._reactivate_after_heading_fault(car_id)
        # recovery=True 는 기록기용(recovery_route.json)이고, 이 경로는
        # 러너가 지금 실행하는 경로이기도 하다. executing 을 주지 않으면
        # overlay 소스(_auto_host_route)가 비어 있어 차는 움직이는데
        # 화면에는 waypoint 도 polyline 도 없고 wp0/0 으로 보인다.
        self._emit_route(wps, car_id=car_id, recovery=True, executing=True)
        return True

    def _maybe_resume_entry_staging(self, view: VehicleView) -> bool:
        """At each staging boundary: STOP → fresh pose → next safe plan."""
        car_id = view.car_id
        if (car_id is None
                or self._parking_stage.get(car_id) != "ENTRY_STAGING_PENDING"):
            return False
        wait_after = self._entry_staging_wait.get(car_id, 0.0)
        if not self._phase_boundary_stopped(
                view, wait_after, "ENTRY_STAGING_BOUNDARY"):
            return True
        if not self._require_critical_heading(view, "ENTRY_STAGING_BOUNDARY"):
            return True
        slot_id = self._auto_host_slot.get(car_id)
        runner = self.auto_hosts.get(car_id)
        if slot_id is None or runner is None:
            self._entry_staging_fault(car_id, "ENTRY_STAGING_CONTEXT_MISSING")
            return True
        self._entry_staging_wait.pop(car_id, None)
        # ENTRY_STAGING 은 목적지가 아니다. 목적은 입구의 어려운 자세를
        # **기존 중앙-start parking flow 가 처리할 수 있는 자세**까지 옮기는
        # 것이다. 그러므로 종료 판정은 "staging waypoint 를 완벽히 끝냈는가"
        # 가 아니라 "production planner 가 지금 자세에서 경로를 만드는가" 다.
        #
        # 기하 근사(_entry_staging_ready = 통로 밴드 + heading 정렬)만 쓰면
        # 이미 인계 가능한 자세에서도 staging 을 계속 강제한다. 실측
        # run_20260901_123352 / _123533: 첫 HEADING_OUT_OF_TOLERANCE 시점의
        # 자세 (362.6,564.4,49.6도) / (365.3,558.7,45.6도) 는 둘 다 이미 정상
        # 경로가 나오는 자세였다(B1 2wp / B1 1wp). 그런데도 staging 을 두 번
        # 더 강제한 결과 차가 오히려 통로 밖(y=489, |y-600|=111mm)으로 나가
        # basin 을 벗어났고 ENTRY_STAGING_EXHAUSTED 로 끝났다.
        #
        # 좌표 임계값이 아니라 planner 답이므로 특정 입구/슬롯 hardcode 가
        # 아니다. heading tolerance 도 건드리지 않는다 — 근사를 넓히는 대신
        # 근사를 쓰지 않는 경로를 하나 더 두는 것이다.
        #
        # 다만 "planner 가 경로를 만든다" 를 **단독** 조건으로 쓰면 안 된다.
        # 그 질문은 heading 에 둔감하다: 통로 (430,600) 에서 heading 0~50도
        # 전 구간에 대해 _handoff_feasible 이 똑같이 B1 을 돌려준다. 그래서
        # staging 이 40~47도에서 끝나고, 그 자세를 받은 rear 계층은 setup 을
        # 2 waypoint 가 아니라 6 waypoint 로 풀어야 한다 — 구간마다 추종
        # 오차가 쌓여 replan churn 이 된다.
        #
        # 기록된 입구 인계 17건(runs/*/events.log 의 ENTRY_STAGING_COMPLETE
        # 이벤트가 x/y/heading 을 그대로 남긴다)을 통로축 heading 편차로
        # 정리하면 경계가 뚜렷하다:
        #     편차 >= 39도 : 10건 -> PARKED 0건 (실패 8, FINAL_POSE_EVAL 2)
        #     편차 <= 23도 :  7건 -> PARKED 1건
        #                     (입구 시작 유일의 성공 022217, 편차 13.3도)
        # 17건의 편차 중앙값은 41.5도다. 즉 현재 게이트는 아직 한 번도
        # 성공한 적 없는 자세대에서 staging 을 끝내고 있다.
        #
        # 그래서 두 질문을 나눠서 둘 다 묻는다.
        #   feasible : planner 가 지금 자세에서 안전한 경로를 만드는가
        #              (위치/boundary/충돌 — 통로에서 111mm 벗어난
        #               (526.8,489.2) 는 여기서 걸러진다)
        #   aligned  : 통로축 heading 이 tolerance 안인가
        #              (planner 가 답하지 못하는, 성공을 가르는 축)
        # feasible 은 항상 필요하고, aligned 가 아니면 staging 예산이
        # 끝날 때까지 더 시도한다. 이 escape 가 123352 식 무한 staging 을
        # 막고, 정렬 우선이 40도 인계를 막는다. 새 임계값은 없다 —
        # entry_staging_heading_tolerance_deg 와 max_entry_staging_attempts
        # 둘 다 이미 있던 값이다.
        #
        # 종료 조건에 통로 밴드(|y-600|<=80)를 넣지 않는 이유는
        # _entry_staging_heading_aligned docstring 참조 — 입구에서 유일하게
        # PARKED_OK 로 끝난 022217 의 인계 자세가 밴드 밖(84.8mm)이었다.
        aligned = self._entry_staging_heading_aligned(view, slot_id)
        exhausted = (self._entry_staging_attempts.get(car_id, 0)
                     >= int(self.config.max_entry_staging_attempts))
        feasible = self._handoff_feasible(view, slot_id) is not None
        # ── 예산 소진은 정렬 조건을 면제하지 않는다 ────────────────────────
        #
        # 예전에는 `feasible and (aligned or exhausted)` 였다. 재시도 횟수를
        # 다 썼다는 사실이 **기하 조건을 통과시키는 근거**로 쓰인 것이다.
        # 예산은 재시도 상한이지 feasibility override 가 아니다.
        #
        # 실측 run_20260904_231157 / _231338: SLOT_REPOSITION 이
        # _start_entry_staging 을 통해 staging 예산을 함께 쓰기 때문에 3/3 이
        # 금방 소진됐고, 그 다음 프레임에 exhausted=True 로 15° 게이트를
        # 건너뛰어 heading 86.9° / 88.8° — 통로축과 거의 수직인 자세에서
        # ENTRY_STAGING_COMPLETE 했다. 그 자세에서 만들어진 인계 경로는
        # waypoint 가 1개뿐이었고 그 하나마저 최소 선회원 안이라(along -30 /
        # -20mm) 1.5초 만에 PATH_DEVIATION 으로 끝났다.
        #
        # 이 파일의 다른 주석이 기록한 실측 통계와도 일치한다: 통로축 편차
        # >= 39° 인 인계 17건 중 PARKED 는 0건이다.
        #
        # 예산이 끝나면 _start_entry_staging 이 ENTRY_STAGING_EXHAUSTED 로
        # 안전 정지하고 예약 슬롯은 그대로 유지된다 — 이미 있는 계약이다.
        # 무한 staging 방지(123352)는 그 예산이 계속 담당한다.
        # ── SLOT COMMITMENT ────────────────────────────────────────────────
        # _handoff_feasible 은 선호 슬롯이 안 되면 **다른 슬롯**을 돌려준다.
        # 그래서 여기서 곧바로 완료하면 _feasible_route 가 선호 슬롯을
        # SLOT_REJECTED 하고 재배정한다. 수정 후 baseline 4/4 가 그렇게 B1 을
        # 잃었다 (merge_x 630~683 > 인계점 425).
        #
        # 그런데 그 거부는 "이 자세에서 직접 경로가 없다" 이지 "그 슬롯이
        # 도달 불가하다" 가 아니다. 예산이 남아 있고 bounded reposition 이
        # 존재하면 슬롯을 바꾸지 않고 한 번 더 옮긴다 — 새 state 도, 새
        # 예산도 없다. staging lifecycle 과 max_entry_staging_attempts 를
        # 그대로 쓴다. 해가 없으면 종전대로 완료(→ 재배정)로 흘러간다.
        if (not exhausted
                and self._preferred_slot_needs_reposition(view, slot_id)
                and self._slot_reposition_available(view, slot_id)):
            route_id = self.orchestrator.next_route_id()
            # 예산을 실제로 태울지는 _apply_entry_staging_route 가 정한다
            # (관측 중단 복구 재적재는 태우지 않는다). 로그의 attempt 도 같은
            # 술어를 써야 실제 값과 어긋나지 않는다.
            costs = 0 if car_id in getattr(self, "_heading_fault_hold", ()) \
                else 1
            self._emit_event("SLOT_REPOSITION", car_id=car_id, slot=slot_id,
                             route_id=route_id,
                             attempt=(self._entry_staging_attempts.get(car_id, 0)
                                      + costs),
                             x_mm=round(view.position_mm[0], 1),
                             y_mm=round(view.position_mm[1], 1),
                             heading_deg=round(float(view.heading_deg), 1))
            self._start_entry_staging(car_id, view, slot_id, route_id,
                                      commit_slot=True)
            return True
        if feasible and aligned:
            route_id = self.orchestrator.next_route_id()
            selected, wps = self._feasible_route(view, slot_id, route_id)
            if selected is None or not wps:
                self._entry_staging_fault(car_id,
                                          "NO_SAFE_GLOBAL_AFTER_STAGING")
                return True
            runner.load_route(wps)
            self._auto_host_slot[car_id] = selected
            view.slot_id = selected
            self._parking_stage[car_id] = "GLOBAL"
            self._allocation_state[car_id] = "ROUTE_LOADED"
            self._emit_event("ENTRY_STAGING_COMPLETE", car_id=car_id,
                             slot=selected, route_id=route_id,
                             x_mm=round(view.position_mm[0], 1),
                             y_mm=round(view.position_mm[1], 1),
                             heading_deg=round(float(view.heading_deg), 1))
            self._emit_route(wps, car_id=car_id)
            return True
        route_id = self.orchestrator.next_route_id()
        self._start_entry_staging(car_id, view, slot_id, route_id)
        return True

    def _start_rear_parking_stage(self, car_id: int) -> bool:
        """인계 도착 → 정지 → **fresh pose** → LOCAL 주차 경로로 전환.

        FinalPoseGuard 가 서로 다른 fresh 관측 3회를 세고 나서야 DONE 을
        내므로, 이 시점의 view pose 는 이미 "정지 + 신선" 이 보장된 값이다.
        그 pose 를 그대로 주차 planner 입력으로 쓴다 — 배정 시점 pose 가 아니다.
        """
        runner = self.auto_hosts.get(car_id)
        slot_id = self._auto_host_slot.get(car_id)
        track_id = self.track_of_car.get(car_id)
        view = self.views.get(track_id) if track_id is not None else None
        if runner is None or slot_id is None or view is None:
            return False
        if not self._require_critical_heading(view, "PARKING_STAGE"):
            log.warning("car %d: 인계 지점 heading 미확보 — 주차 전환 보류", car_id)
            return False

        route_id = self.orchestrator.next_route_id()
        self._parking_stage[car_id] = "PARKING"
        try:
            wps = self._build_route(default_slot_specs()[slot_id], view, route_id)
        except InfeasibleRouteError as exc:
            # 직접 후진 경로가 안 나오면 즉시 실패시키지 않는다. 현재 fresh pose
            # 에서 짧은 setup 기동을 실행하고, 완료 뒤 실제 카메라 pose 로 이
            # 메서드를 다시 호출한다. 계산된 terminal pose 를 다음 계획 입력으로
            # 재사용하지 않는 것이 핵심이다.
            obstacles = self._planner_obstacle_poses(view)
            setup = build_setup_recovery_waypoints(
                default_slot_specs()[slot_id], route_id=route_id,
                from_pose=view.position_mm, from_heading_deg=view.heading_deg,
                obstacle_poses=obstacles,
                min_executable_mm=self._setup_min_executable_mm())
            degenerate = False
            loses_target = False
            if setup and self._setup_is_degenerate(view, setup):
                # 서 있는 자리에서 곧바로 DONE 되는 기동이다. 실으면 같은
                # pose 로 되돌아와 같은 경로를 무한히 다시 만든다.
                setup = []
                degenerate = True
            elif setup and not self._setup_keeps_target_feasible(
                    view, slot_id, setup):
                # 지금 인계 가능한 자세를 잃는 기동이다 (_setup_keeps_target_
                # feasible 주석의 164904 사례). 실으면 다음 recovery 가
                # NO_SAFE_SETUP_MANEUVER 로 끝난다.
                setup = []
                loses_target = True
            if not setup:
                self._parking_stage[car_id] = "WAIT_SAFE_RECOVERY"
                self._emit_event("RECOVERY_REJECTED", car_id=car_id,
                                 route_id=route_id, slot=slot_id,
                                 reason=("INFEASIBLE_DEGENERATE_SETUP"
                                         if degenerate
                                         else "SETUP_LOSES_TARGET_HANDOFF"
                                         if loses_target
                                         else "NO_SAFE_SETUP_MANEUVER"))
                log.warning("car %d: 인계 자세(%.0f,%.0f hdg %.0f°)에서 슬롯 %s 주차 "
                            "및 setup recovery 불가 (%s) — 정지", car_id,
                            view.position_mm[0], view.position_mm[1],
                            view.heading_deg, slot_id, exc.reason)
                runner.stop()
                self._emit_event("FAULT", car_id=car_id,
                                 reason="NO_SAFE_PARKING_RECOVERY")
                self.dashboard.push_event("parking_plan_failed", car_id=car_id,
                                          slot=slot_id, reason=exc.reason)
                return False
            if not self._trajectory_safe(view, setup, slot_id=slot_id,
                                         recovery=True):
                self._parking_stage[car_id] = "WAIT_SAFE_RECOVERY"
                return False
            # setup → 새 pose → 여기로 재진입은 이 경로의 **정상 순환**이다.
            # 차를 실제로 옮겨 후진 가능한 자세를 만드는 과정이지 실패가 아니므로
            # recovery 예산을 쓰지 않는다. 예산은 REPLAN_REQUIRED 에서만 센다.
            # (실측 run_20260824_204027: 30.7/36.5/41.0mm 를 실제로 이동한 setup
            #  3회가 예산 3/3 을 소진해, 정작 주차 경로의 첫 정당한 recovery 에서
            #  PARKING_RECOVERY_EXHAUSTED 로 죽었다.)
            # 이동 없는 setup 의 무한 재생성은 _setup_is_degenerate 와 planner 의
            # 최소 실행가능 거리 조건이 막는다 — 카운터로 막지 않는다.
            self._parking_stage[car_id] = "SETUP"
            runner.load_route(setup)
            self._reactivate_after_heading_fault(car_id)
            self._emit_route(setup, car_id=car_id, recovery=True)
            log.info("car %d: 직접 주차 경로 불가 (%s) — fresh pose 기준 setup "
                     "recovery 시작 (route %d, %d개)", car_id, exc.reason,
                     route_id, len(setup))
            self.dashboard.push_event("parking_setup_recovery", car_id=car_id,
                                      slot=slot_id, reason=exc.reason)
            return True
        if not self._trajectory_safe(view, wps, slot_id=slot_id):
            self._parking_stage[car_id] = "WAIT_SAFE_ROUTE"
            return False
        runner.load_route(wps)
        self._reactivate_after_heading_fault(car_id)
        self._emit_route(wps, car_id=car_id)
        log.info("car %d: 인계 도착 → fresh pose (%.0f,%.0f hdg %.0f°) 로 주차 "
                 "경로 생성 (route %d, %d개)", car_id, view.position_mm[0],
                 view.position_mm[1], view.heading_deg, route_id, len(wps))
        self.dashboard.push_event("parking_stage", car_id=car_id, slot=slot_id)
        return True

    def _replan_rear_entry(self, spec, view: VehicleView,
                           route_id: int) -> list[Any] | None:
        """현재 자세에서 ENTRY→FINAL 만으로 슬롯에 닿는 경로. 안 되면 None.

        후진 원호는 원이고 φ 는 "그 원의 어디로 들어가느냐"일 뿐이라,
        REVERSE_START 에서 heading 이 어긋나도 지금 있는 자리에서 원호에
        올라탈 수 있는 경우가 많다. 옛 waypoint 로 되돌아갈 이유가 없다.
        """
        if not self.rear_parking_mode or view.heading_deg is None:
            return None
        # ① 현재 자세에서 후보 planner 로 새 주차 경로를 만든다 (반경도 다시 고른다).
        try:
            return build_rear_candidate_waypoints(
                spec, route_id=route_id, from_pose=view.position_mm,
                from_heading_deg=view.heading_deg, strict=True)
        except InfeasibleRouteError:
            pass
        # ② 이미 후진 원호에 올라타 있으면 ENTRY 부터 이어간다.
        try:
            return build_rear_entry_waypoints(
                spec, route_id=route_id, from_pose=view.position_mm,
                from_heading_deg=view.heading_deg,
                min_radius_mm=self._plan_radius, strict=True)
        except InfeasibleRouteError as exc:
            log.info("car %s: 현재 자세에서 후진 진입 불가 (%s) — 전체 재계획으로",
                     view.car_id, exc.reason)
            return None

    def _build_route(self, spec, view: VehicleView, route_id: int) -> list[Any]:
        """parking_mode 에 따라 인계 경로 / 후면주차 경로를 만든다.

        두 생성기 모두 실현 불가하면 InfeasibleRouteError 를 던지므로,
        호출자의 대체 슬롯 로직은 그대로 쓸 수 있다.
        """
        if self.rear_parking_mode:
            # ── 2단계 경계 ────────────────────────────────────────────────
            # GLOBAL 구간에서는 **슬롯 앞 인계 지점까지만** 만든다. 주차 경로를
            # 배정 시점 pose 로 미리 확정하면, 인계에 도착했을 때 실제 자세와
            # 어긋난 계획을 그대로 실행하게 된다. 인계 도착 → 정지 →
            # fresh pose → LOCAL planner 순서로 간다.
            if self._parking_stage.get(view.car_id) != "PARKING":
                return build_waypoints(
                    spec, route_id=route_id, from_pose=view.position_mm,
                    from_heading_deg=view.heading_deg,
                    min_radius_mm=self._plan_radius, strict=True)
            # 현재 자세에서 (반경 × 진입각 × side) 후보를 만들어 고른다.
            # heading 을 아직 모르면 고정 반경 생성기로 폴백한다.
            if view.heading_deg is not None:
                return build_rear_candidate_waypoints(
                    spec, route_id=route_id, from_pose=view.position_mm,
                    from_heading_deg=view.heading_deg, strict=True)
            return build_rear_parking_waypoints(
                spec, route_id=route_id, from_pose=view.position_mm,
                from_heading_deg=view.heading_deg,
                min_radius_mm=self._plan_radius, strict=True)
        return build_waypoints(
            spec, route_id=route_id, from_pose=view.position_mm,
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
            if self._slot_occupancy()[idx] >= 1.0:
                continue                          # 점유/선점/카메라 점유
            if self.rear_parking_mode:
                rear, _ = choose_rear_parking_plan(
                    spec, min_radius_mm=self._plan_radius,
                    from_pose=view.position_mm)
                if rear is None:
                    continue
                target = rear.setup_start[:2]
            else:
                plan = plan_handoff(spec, from_pose=view.position_mm,
                                    from_heading_deg=view.heading_deg,
                                    min_radius_mm=self._plan_radius)
                if not plan.feasible:
                    continue
                target = plan.point
            d = math.hypot(target[0] - view.position_mm[0],
                           target[1] - view.position_mm[1])
            if d < best_d:
                best, best_d = slot_id, d
        return best

    def _reject_slot(self, car_id: int, slot_id: str, reason: str) -> None:
        """차량 현재 위치에서 갈 수 없는 슬롯을 배정 후보에서 뺀다.

        점유 상태를 조작하지는 않는다 — 정책이 그걸 혼잡으로 읽어 WAIT 으로
        굳어버린다(실측). 기록·보고용으로만 남긴다.
        """
        slots = self._unreachable_slots.setdefault(car_id, set())
        if slot_id in slots:
            return
        slots.add(slot_id)
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
        runner = self.auto_hosts.get(view.car_id) if self.auto_host_mode else None
        target = runner.current_target if runner is not None else None
        workflow = self.workflow_status(view.car_id) if self.auto_host_mode else None
        if workflow == "PARKED":
            status = "parked"
        elif (workflow and (workflow.startswith("WAIT")
                            or workflow.endswith("FAULT"))):
            status = "waiting"
        else:
            status = ("parked" if mission and mission.state is MissionState.DONE
                      else "moving")
        self.dashboard.push_pose(
            car_id=view.car_id, position_mm=view.position_mm,
            status=status,
            heading_deg=view.heading_deg, heading_source=view.heading_source,
            parking_phase=(getattr(target, "phase", None) if target is not None
                           else mission.current.phase if mission and mission.current else None),
            route_id=(getattr(target, "route_id", None) if target is not None
                      else mission.route_id if mission else None),
            waypoint_id=(getattr(target, "waypoint_id", None) if target is not None
                         else mission.current.waypoint_id
                         if mission and mission.current else None),
            track_id=view.track_id,
            assigned_slot=self._auto_host_slot.get(view.car_id, view.slot_id),
            parking_stage=workflow,
            connection_state=("COMM_LOST" if view.car_id in self._comm_lost
                              else "CONNECTED"),
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
            self._parking_stage.pop(car_id, None)

    def _start_auto_host(self, car_id: int, slot_id: str,
                         waypoints: list[Any], *, view: VehicleView) -> bool:
        """AUTO_HOST 주행을 건다. 수동 셸이 이미 있으면 경로만 갈아끼운다."""
        # This is the final production load boundary.  Keep the check here even
        # though allocation already validates candidates: future callers must
        # not be able to construct or replace an executable mission unchecked.
        if not self._trajectory_safe(view, waypoints, slot_id=slot_id):
            return False
        self._replan_attempts.pop(car_id, None)
        self._last_replan_signature.pop(car_id, None)
        runner = self.auto_hosts.get(car_id)
        mux = self.hybrid_controls.get(car_id)
        if runner is not None:
            held = bool(getattr(self.server, "control_is_held", lambda _c: False)(
                car_id))
            if held and not bool(getattr(
                    self.server, "control_ready", lambda _c: False)(car_id)):
                self._allocation_state[car_id] = "WAIT_COMM_SESSION"
                self.server.stop_control(car_id)
                return False
            runner.load_route(waypoints)
            if mux is not None:
                # AUTO_PENDING → 새 카메라 pose 를 받은 뒤에 주행이 재개된다
                mux.switch_to_auto()
            else:
                mux = HybridControlMux(runner)
                self.hybrid_controls[car_id] = mux
                mux.switch_to_auto()
            if held:
                release = getattr(self.server, "release_control", None)
                if release is None or not release(car_id):
                    runner.prepare_route_switch()
                    hold = getattr(self.server, "hold_control", None)
                    hold(car_id) if hold is not None else self.server.stop_control(car_id)
                    self._allocation_state[car_id] = "WAIT_COMM_SESSION"
                    self._emit_event(
                        "FAULT", car_id=car_id,
                        reason="NEW_ROUTE_CONTROL_RELEASE_REJECTED")
                    return False
                self._emit_event(
                    "NEW_ROUTE_ACTIVATED", car_id=car_id, slot=slot_id,
                    state="AUTO_PENDING", comm_latch_cleared=True)
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
        self.dashboard.push_event("auto_host_armed", car_id=car_id, slot=slot_id)
        return True

    # ─── 수동/자동 전환 API (hybrid_gui.py 가 호출) ──────────────────────────

    def hybrid_available(self, car_id: int) -> bool:
        return car_id in self.hybrid_controls

    def hybrid_mode(self, car_id: int) -> str:
        mux = self.hybrid_controls.get(car_id)
        return mux.mode if mux is not None else "UNAVAILABLE"

    def workflow_status(self, car_id: int) -> str:
        """Parking workflow state, distinct from the current route mission."""
        runner = self.auto_hosts.get(car_id)
        mission = getattr(runner, "status", None)
        mission = getattr(mission, "value", mission)
        if mission == "PARKED":
            return "PARKED"
        stage = self._parking_stage.get(car_id)
        if stage == "PARKING_AFTER_SETUP_PENDING":
            return "SETUP_DONE_WAIT_STOP"
        if stage == "PARKING_HANDOFF_PENDING":
            return "GLOBAL_DONE_WAIT_STOP"
        if stage:
            return stage
        return str(mission or self._allocation_state.get(car_id, "IDLE"))

    def lifecycle_snapshot(self, car_id: int) -> dict[str, Any]:
        """Recorder-facing ownership snapshot for post-run zero diagnosis."""
        route = self._auto_host_route.get(car_id) or []
        route_id = getattr(route[-1], "route_id", None) if route else None
        ctx = self._comm_recovery_context.get(car_id)
        slot_id = self._auto_host_slot.get(car_id)
        slot = default_slot_specs().get(slot_id) if slot_id else None
        return {
            "workflow_status": self.workflow_status(car_id),
            "parking_stage": self._parking_stage.get(car_id),
            "allocation_state": self._allocation_state.get(car_id),
            "owned_route_id": route_id,
            "comm_recovery_state": None if ctx is None else ctx.get("state"),
            "slot_id": slot_id,
            # SlotSpec 의 좌표 필드는 center_x / center_y 다. slot.x / slot.y 는
            # 존재하지 않아 슬롯이 배정된 **직후부터** 매 tick AttributeError 가
            # 났고, 기록기 루프가 그 예외를 삼키는 바람에 control.jsonl 과
            # PHASE/MISSION/ESP_STATE 파생 이벤트가 조용히 끊겼다
            # (run_20260904_164904/_165419/_165710 3/3, SLOT_SELECTED 시각과
            # 마지막 control 행 시각이 0.1초 이내로 일치).
            "slot_center_x_mm": None if slot is None else slot.center_x,
            "slot_center_y_mm": None if slot is None else slot.center_y,
            "parked_heading_deg": (
                None if slot is None else slot.target_heading_deg),
        }

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
                              view.heading_deg, view.last_obs_time,
                              view.heading_source)
        # 물리 경계 감시는 **어떤 handler 보다 먼저, 무조건** 돈다.
        # 아래 handler 들은 조기 return 하므로 그 뒤에 두면 건너뛴다 —
        # 실측 run_20260831_231000: ENTRY_STAGING_PENDING 이 1.64초 동안
        # 조기 return 하는 사이 차가 관성으로 맵 밖 38.6mm 까지 나갔는데
        # BOUNDARY_HARD 는 그 상태가 끝난 뒤에야 찍혔다.
        self._check_boundary(view)
        # COMM recovery owns this frame until it has replaced the stale route.
        # Even the successful planning frame remains zero; AUTO_PENDING needs
        # one more distinct camera observation before the scheduler restarts.
        if self._maybe_resume_comm_recovery(view):
            return
        # 배경 staging 계획이 돌고 있거나 방금 끝났으면 이 프레임은 그쪽이
        # 소유한다. 계획 중에는 차가 정지 상태로 기다리고, 끝났으면 여기서
        # (프레임 스레드에서) 적재한다.
        if self._apply_entry_staging_plan(view):
            return
        if self._maybe_resume_entry_staging(view):
            return
        # heading 이 돌아왔으면 관측 대기 fault 에서 먼저 빠져나온다. 실제 주행
        # 재개는 아래 setup coordinator 가 새 route 를 검증한 뒤에 일어난다.
        self._maybe_resume_heading_fault(view)
        self._maybe_start_parking_setup(view)
        self._maybe_evaluate_final_pose(view)
        self._maybe_verify_parked(view)
        # 위 handler 들이 아무것도 하지 않은 채 시간이 흐르면 마지막 그물이 잡는다.
        self._check_parking_progress(view)
        self._maybe_start_rear_after_stop(view)
        self._check_path_deviation(view, runner)

    def _recoverable(self, target: Any) -> bool:
        """이 waypoint 에서 후진 복구를 걸어도 되는가 (phase 기준)."""
        phase = str(getattr(target, "phase", "") or "").upper()
        return phase in self.config.recover_phases

    def _is_global_handoff_terminal(self, car_id: int, target: Any) -> bool:
        """True only for the GLOBAL route's terminal handoff waypoint."""
        return bool(
            self.rear_parking_mode
            and getattr(target, "is_final", False)
            and str(getattr(target, "phase", "") or "").upper() == "FINAL"
            and self._parking_stage.get(car_id) not in {
                "PARKING", "SETUP", "SETUP_PENDING",
                "PARKING_AFTER_SETUP_PENDING", "PARKING_HANDOFF_PENDING",
            }
        )

    def _handoff_region_reached(self, view: VehicleView, target: Any) -> bool:
        """Use existing handoff geometry, not an enlarged waypoint tolerance."""
        distance = math.hypot(target.x_mm - view.position_mm[0],
                              target.y_mm - view.position_mm[1])
        if distance > HANDOFF_LEAD_MM:
            return False
        if abs(view.position_mm[1] - AISLE_Y) > ON_AISLE_TOLERANCE_MM:
            return False
        desired = getattr(target, "target_heading_deg", None)
        return (desired is None or view.heading_deg is None
                or self._heading_delta(view.heading_deg, desired)
                <= ALONG_AISLE_HEADING_TOLERANCE_DEG)

    def _begin_parking_handoff(self, view: VehicleView, runner: AutoHostRunner,
                               target: Any, *, reason: str) -> None:
        """Stop GLOBAL pursuit and wait for a distinct pose before parking."""
        car_id = view.car_id
        if car_id is None:
            return
        runner.prepare_route_switch()
        self.server.stop_control(car_id)
        self._parking_stage[car_id] = "PARKING_HANDOFF_PENDING"
        self._parking_plan_wait[car_id] = view.last_obs_time
        self._deviation_streak.pop(car_id, None)
        self._last_replan_signature.pop(car_id, None)
        self._emit_event("HANDOFF_CAPTURED", car_id=car_id,
                         route_id=getattr(target, "route_id", None),
                         waypoint_id=getattr(target, "waypoint_id", None),
                         reason=reason,
                         distance_mm=round(math.hypot(
                             target.x_mm - view.position_mm[0],
                             target.y_mm - view.position_mm[1]), 1))
        self.dashboard.push_event("handoff_captured", car_id=car_id,
                                  reason=reason)

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
        # 후진 목표에 전진 도달성을 따지면 안 된다. 복구 waypoint 는 정의상
        # 등 뒤에 있어서 매번 "이탈"로 잡히고, 복구가 복구를 부르며 몇 프레임
        # 만에 재시도 횟수를 태워버린다.
        reverse = (getattr(getattr(target, "motion_direction", None), "value", "")
                   == "REVERSE")
        phase = str(getattr(target, "phase", "") or "").upper()
        # TRAJECTORY is a valid forward motion heading.  Only reverse
        # reachability is ambiguous because body heading is reconstructed 180
        # degrees away from that motion vector.
        if view.heading_source == "TRAJECTORY" and reverse:
            return

        # Every route boundary uses its declared completion radius before any
        # behind-target/deviation logic.  This includes GLOBAL FINAL, setup's
        # last RECOVERY waypoint, and rear FINAL.
        distance = math.hypot(target.x_mm - view.position_mm[0],
                              target.y_mm - view.position_mm[1])
        if distance <= float(target.position_tolerance_cm) * 10.0:
            self._deviation_streak.pop(car_id, None)
            self._reverse_closest.pop(car_id, None)
            return

        # GLOBAL handoff is a phase boundary, not an intermediate waypoint.
        # The existing FINAL tolerance (50 mm) must get the first chance to
        # complete in HostController.  Otherwise the asynchronous pipeline
        # monitor can turn a valid 20 mm capture into PATH_DEVIATION first.
        if self._is_global_handoff_terminal(car_id, target):
            tolerance = float(target.position_tolerance_cm) * 10.0
            if distance <= tolerance:
                self._deviation_streak.pop(car_id, None)
                return
            if (forward_unreachable(
                    view.position_mm, view.heading_deg,
                    (target.x_mm, target.y_mm), radius_mm=MIN_TURN_RADIUS_MM,
                    lot_mm=(self.config.lot_width_mm,
                            self.config.lot_height_mm))
                    and self._handoff_region_reached(view, target)):
                self._begin_parking_handoff(
                    view, runner, target, reason="HANDOFF_REGION_OVERSHOOT")
                return
        if runner.mission.is_recovering or (reverse and phase == "RECOVERY"):
            # 복구 waypoint 는 정의상 등 뒤에 있어 매번 "이탈"로 잡힌다.
            # RECOVERY 는 기존 특별 취급을 그대로 유지한다.
            self._deviation_streak.pop(car_id, None)
            self._reverse_closest.pop(car_id, None)
            return
        if reverse:
            # 후진 주차(ENTRY/FINAL): 전진 도달성 대신 **수렴 여부**를 본다.
            self._check_reverse_deviation(view, runner, target)
            return
        self._reverse_closest.pop(car_id, None)
        # 통로 중간 점은 허용오차가 넓고 다음 점이 이어진다 — 조금 밀려도 계속
        # 가면 된다. 여기서 후진을 걸면 진행이 끊긴다.
        setup_terminal = (
            self._parking_stage.get(car_id) == "SETUP"
            and getattr(runner, "current_is_terminal", False)
            and phase == "RECOVERY")
        # Rear ALIGN was previously invisible to deviation monitoring.  Both
        # map-exit runs passed a missed ALIGN target and kept driving until the
        # physical boundary stop.  In PARKING, ALIGN failure is handled by the
        # existing bidirectional setup coordinator, never legacy reverse.
        monitored_parking_align = (
            self._parking_stage.get(car_id) == "PARKING"
            and phase == "ALIGN")
        if (not self._recoverable(target)
                and not setup_terminal and not monitored_parking_align):
            self._deviation_streak.pop(car_id, None)
            return

        key = (getattr(target, "route_id", None),
               getattr(target, "waypoint_id", None))
        forward_closest = getattr(self, "_forward_closest", None)
        if forward_closest is None:
            forward_closest = self._forward_closest = {}
        best_key, best = forward_closest.get(
            car_id, (None, float("inf")))
        if key != best_key:
            best = distance
            forward_closest[car_id] = (key, distance)
            self._deviation_streak.pop(car_id, None)
        elif distance < best:
            forward_closest[car_id] = (key, distance)
            self._deviation_streak.pop(car_id, None)
            return

        h = math.radians(view.heading_deg)
        ahead = ((target.x_mm - view.position_mm[0]) * math.cos(h)
                 + (target.y_mm - view.position_mm[1]) * math.sin(h))
        diverged = (distance - best
                    >= self.config.initial_pose_stability_mm)

        off = forward_unreachable(
            view.position_mm, view.heading_deg, (target.x_mm, target.y_mm),
            radius_mm=MIN_TURN_RADIUS_MM,
            lot_mm=(self.config.lot_width_mm, self.config.lot_height_mm))
        # A fresh APPROACH point may be inside the point-turning circle while
        # still ahead of the car and converging toward its capture radius.  Do
        # not let this asynchronous monitor pre-empt that acquisition.  Behind
        # targets and trajectories that materially diverge remain candidates.
        if not off or (ahead >= 0.0 and not diverged):
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

    def _check_reverse_deviation(self, view: VehicleView, runner: AutoHostRunner,
                                 target: Any) -> None:
        """후진 주차 구간에서 목표로 수렴하지 못하면 멈춘다.

        후진 목표에 ``forward_unreachable`` 을 쓰면 안 된다 — 후진 목표는
        정의상 등 뒤라 항상 참이 된다. 대신 "지금까지 가장 가까웠던 거리보다
        다시 멀어졌는가"를 본다. 방향과 무관하게 성립하는 판정이다.
        """
        car_id = view.car_id
        key = (getattr(target, "route_id", None),
               getattr(target, "waypoint_id", None))
        best_key, best = self._reverse_closest.get(car_id, (None, float("inf")))
        dist = math.hypot(target.x_mm - view.position_mm[0],
                          target.y_mm - view.position_mm[1])
        if key != best_key:
            self._reverse_closest[car_id] = (key, dist)
            self._deviation_streak.pop(car_id, None)
            return
        if dist < best:
            self._reverse_closest[car_id] = (key, dist)
            self._deviation_streak.pop(car_id, None)
            return

        if dist - best < self.config.reverse_divergence_mm:
            return
        n = self._deviation_streak.get(car_id, 0) + 1
        self._deviation_streak[car_id] = n
        if n < self.config.deviation_frames:
            return
        self._deviation_streak.pop(car_id, None)
        self._reverse_closest.pop(car_id, None)
        log.warning("car %s: 후진 이탈 — 목표 wp%s 까지 %.0fmm 로 다시 멀어짐 "
                    "(최근접 %.0fmm, pose %.0f,%.0f)", car_id,
                    getattr(target, "waypoint_id", "?"), dist, best,
                    view.position_mm[0], view.position_mm[1])
        runner.mission.request_replan("REVERSE_PATH_DEVIATION")
        self.dashboard.push_event("reverse_deviation", car_id=car_id,
                                  waypoint_id=getattr(target, "waypoint_id", None))

    def _boundary_uncertain_held(self, car_id: int) -> bool:
        """Whether the heading-uncertainty hold event was already emitted."""
        held = getattr(self, "_boundary_heading_hold", None)
        if held is None:
            held = self._boundary_heading_hold = set()
        if car_id in held:
            return True
        held.add(car_id)
        return False

    def _boundary_overflow(self, view: VehicleView, *, hard_limit_mm: float):
        """Map overflow for this pose, honouring heading provenance.

        heading 은 **위치와 신뢰도가 다르다**. rc_car bbox 에서 오는 x/y 는
        매 프레임 신선하지만 heading 은 LAST_VALID 로 몇 초씩 얼어붙는다.
        얼어붙은 값으로 차체를 회전시켜 "정확한 footprint" 라고 부르면
        방향에 따라 과대평가도 과소평가도 된다 — 안전 게이트로서 unsound 다
        (실측 223536: 실제 약 5mm 초과를 stale 303.7° 로 25.6mm 로 계산).

        그래서 orientation 을 믿을 수 있을 때만 정확 계산을 쓰고, 못 믿을 때는
        **방향과 무관한 상·하한**으로 세 갈래로 나눈다:

            내접(반폭 75mm)   초과  → 어떤 방향이어도 넘는다 → CONFIRMED
            외접(145.8mm)     안전  → 어떤 방향이어도 안 넘는다 → 안전
            그 사이                 → 방향에 따라 갈린다 → UNCERTAIN

        Returns (mode, overflow_mm): mode 는 EXACT / CONSERVATIVE / UNCERTAIN.
        UNCERTAIN 이면 overflow 는 참고값(외접 기준)이고 판정에 쓰지 않는다.
        """
        x, y = view.position_mm
        w, h = self.config.lot_width_mm, self.config.lot_height_mm
        trusted = view.heading_source in {"FRONT_CUSHION", "TRAJECTORY"}
        if trusted and view.heading_deg is not None:
            worst = 0.0
            for px, py in _car_footprint(x, y, view.heading_deg):
                worst = max(worst, -px, px - w, -py, py - h)
            return "EXACT", max(0.0, worst)

        def disc_overflow(radius: float) -> float:
            return max(0.0, -(x - radius), (x + radius) - w,
                       -(y - radius), (y + radius) - h)

        inscribed = CAR_WIDTH_MM / 2.0                       # 방향 무관 최소 반폭
        circumscribed = math.hypot(CAR_LENGTH_MM / 2.0, CAR_WIDTH_MM / 2.0)
        best = disc_overflow(inscribed)
        worst = disc_overflow(circumscribed)
        if best > hard_limit_mm:
            # 가장 유리한 방향으로 놓아도 한계를 넘는다 → 확정 위반.
            return "CONFIRMED", best
        if worst <= 0.0:
            # 가장 불리한 방향으로 놓아도 맵 안이다 → 확정 안전.
            return "CONSERVATIVE", 0.0
        return "UNCERTAIN", worst

    def _check_boundary(self, view: VehicleView) -> None:
        """차체가 맵을 벗어나면 즉시 세운다 — 마지막 물리적 방어선.

        이탈 감시가 늦거나 못 잡는 경우에도 몇 백 mm 를 더 나가지 않게 한다.
        재계획을 걸지 않고 바로 정지시킨다: 이미 맵 밖이면 계획을 다시 세울
        근거 자체가 없다.
        """
        car_id = view.car_id
        runner = self.auto_hosts.get(car_id)
        if runner is None or view.heading_deg is None:
            return
        # 이 margin은 실시간 hard-stop 판정에만 적용한다. planner의 map/swept
        # footprint constraint는 그대로 두어 실제 경로가 맵 밖으로 계획되지는 않는다.
        margin = self.config.boundary_hard_margin_mm
        uncertainty = float(getattr(
            self.config, "boundary_measurement_uncertainty_mm", 10.0))
        uncertain_limit = margin + uncertainty
        mode, worst = self._boundary_overflow(
            view, hard_limit_mm=uncertain_limit)
        if mode != "UNCERTAIN" and car_id is not None:
            getattr(self, "_boundary_heading_hold", set()).discard(car_id)
        if mode == "UNCERTAIN":
            # 방향을 모르는 채로 non-zero 로 계속 가지 않는다. 정지시키고
            # 기존 fresh-heading 계약에 태운다 — 새 대기 상태를 만들지 않으므로
            # 기존 liveness 복구(WAIT_FRESH_HEADING → HEADING_RECOVERED)와
            # watchdog 예외가 그대로 적용된다.
            if car_id is not None and not self._boundary_uncertain_held(car_id):
                self._emit_event(
                    "BOUNDARY_HEADING_UNCERTAIN", car_id=car_id,
                    heading_source=view.heading_source,
                    worst_case_overflow_mm=round(worst, 1),
                    hard_above_mm=round(uncertain_limit, 1))
                self.dashboard.push_event(
                    "boundary_heading_uncertain", car_id=car_id,
                    heading_source=view.heading_source)
            self._require_critical_heading(view, "BOUNDARY_HEADING_UNCERTAIN")
            return
        if mode == "CONFIRMED":
            # 어떤 방향이어도 넘는다 — heading 없이도 확정할 수 있는 위반이다.
            worst = max(worst, uncertain_limit + 1.0)
        if worst <= uncertain_limit and self._predictive_boundary_stop(
                view, runner, current_overflow_mm=worst):
            return
        uncertain = getattr(self, "_boundary_uncertain", None)
        if uncertain is None:
            uncertain = self._boundary_uncertain = set()
        hard = getattr(self, "_boundary_hard", None)
        if hard is None:
            hard = self._boundary_hard = set()
        terminal_streak = getattr(self, "_boundary_terminal_streak", None)
        if terminal_streak is None:
            terminal_streak = self._boundary_terminal_streak = {}
        uncertain_trend = getattr(self, "_boundary_uncertain_trend", None)
        if uncertain_trend is None:
            uncertain_trend = self._boundary_uncertain_trend = {}

        # 20--30 mm is measurement uncertainty for both moving and stopped
        # states.  One 4-FPS observation never becomes a hard fault.  While
        # RUNNING, two distinct and increasing observations request a safe
        # zero/replan; predictive motion normally wins before that fallback.
        terminal_zero = runner.status in {MissionStatus.DONE, MissionStatus.PARKED}
        if margin < worst <= uncertain_limit:
            terminal_streak.pop(car_id, None)
            if car_id not in uncertain:
                uncertain.add(car_id)
                self._emit_event(
                    "BOUNDARY_UNCERTAIN", car_id=car_id,
                    overflow_mm=round(worst, 1),
                    running=runner.status is MissionStatus.RUNNING,
                    hard_above_mm=round(uncertain_limit, 1))
                self.dashboard.push_event(
                    "boundary_uncertain", car_id=car_id,
                    overflow_mm=round(worst, 1))
            if runner.status is MissionStatus.RUNNING:
                previous = uncertain_trend.get(car_id)
                now = float(view.last_obs_time)
                if previous is not None and previous[2] < 0:
                    return
                increase = float(getattr(
                    self.config, "boundary_uncertain_increase_mm", 1.0))
                if previous is None:
                    count = 1
                elif now <= previous[0]:
                    return
                elif worst >= previous[1] + increase:
                    count = previous[2] + 1
                else:
                    count = 1
                uncertain_trend[car_id] = (now, worst, count)
                required = max(2, int(getattr(
                    self.config, "boundary_uncertain_confirm_frames", 2)))
                if count >= required:
                    runner.mission.request_replan("BOUNDARY_UNCERTAIN_TREND")
                    runner.prepare_route_switch()
                    self.server.stop_control(car_id)
                    uncertain_trend[car_id] = (now, worst, -1)
                    self._emit_event(
                        "BOUNDARY_UNCERTAIN_STOP", car_id=car_id,
                        overflow_mm=round(worst, 1), observations=count)
                    self.dashboard.push_event(
                        "boundary_uncertain_stop", car_id=car_id,
                        overflow_mm=round(worst, 1))
            return

        # A completed host mission has no target and therefore emits zero.
        # Preserve the existing two-frame confirmation for a single >30 mm
        # stationary spike; an active mission is hard-stopped immediately.
        if terminal_zero and worst > uncertain_limit:
            terminal_streak[car_id] = terminal_streak.get(car_id, 0) + 1
            required = max(1, int(getattr(
                self.config, "boundary_terminal_confirm_frames", 2)))
            if terminal_streak.get(car_id, 0) < required:
                if car_id not in uncertain:
                    uncertain.add(car_id)
                    self._emit_event(
                        "BOUNDARY_UNCERTAIN", car_id=car_id,
                        overflow_mm=round(worst, 1), terminal_zero=True,
                        confirm_above_mm=round(uncertain_limit, 1))
                    self.dashboard.push_event(
                        "boundary_uncertain", car_id=car_id,
                        overflow_mm=round(worst, 1))
                return
        if 0.0 < worst <= margin:
            uncertain.discard(car_id)
            uncertain_trend.pop(car_id, None)
            terminal_streak.pop(car_id, None)
            if car_id not in self._boundary_soft:
                self._boundary_soft.add(car_id)
                self._emit_event("BOUNDARY_SOFT", car_id=car_id,
                                 overflow_mm=round(worst, 1))
            return
        if worst <= 0.0:
            self._boundary_soft.discard(car_id)
            uncertain.discard(car_id)
            hard.discard(car_id)
            self._escape_hold().discard(car_id)
            terminal_streak.pop(car_id, None)
            uncertain_trend.pop(car_id, None)
            return
        if car_id in hard:
            return
        hard.add(car_id)
        log.error("car %s: 차체가 맵을 %.0fmm 벗어남 (pose %.0f,%.0f hdg %.0f°) "
                  "— 즉시 정지", car_id, worst, view.position_mm[0],
                  view.position_mm[1], view.heading_deg)
        runner.stop()
        # 정지는 그대로 유지한다. 다만 "다시 켤 수 있는 유일한 조건" 을
        # 표시해 둔다 — 기존 최종자세 평가가 만든 **탈출 경로가
        # _trajectory_safe 를 통과했을 때만**.
        #
        # 표시가 없으면 이 상태에서 나올 길이 아예 없다. hard 집합은
        # worst<=0 일 때만 비워지는데, 그러려면 차가 맵 안으로 돌아와야 하고,
        # 돌아오려면 움직여야 하는데 authority 가 FAULTED 라 못 움직인다.
        # 실측 run_20260903_161435: 맵을 55.3mm 벗어난 채 멈춘 뒤
        # FINAL_POSE_EVAL(ALIGN) 도 나왔고 FINAL_ALIGNMENT route 도
        # 실제로 적재됐는데, 7초 내내 PWM 0 / encoder_delta 0 이었다.
        # 그 경로의 첫 waypoint (476.9,1006.8) 는 이미 맵 안쪽 53.8mm 이고
        # validate_trajectory 도 safe=True 를 준다(min_clearance -55.3mm =
        # 출발 자세 그 자체, 이후 단조 개선). 즉 복구 수단은 이미 다 있었고
        # 실행 권한만 없었다.
        if self._parking_stage.get(car_id) in self._REAR_RECOVERABLE_STAGES:
            self._escape_hold().add(car_id)
        self._emit_event("BOUNDARY_HARD", car_id=car_id,
                         overflow_mm=round(worst, 1))
        self._emit_event("FAULT", car_id=car_id, reason="BOUNDARY_HARD")
        self.dashboard.push_event("boundary_stop", car_id=car_id,
                                  overflow_mm=round(worst, 1))

    def _predictive_boundary_stop(self, view: VehicleView,
                                  runner: AutoHostRunner, *,
                                  current_overflow_mm: float) -> bool:
        """Stop parking if measured motion crosses HARD_BOUNDARY within 0.5 s."""
        car_id = view.car_id
        parking_stage = getattr(self, "_parking_stage", {}).get(car_id)
        if (car_id is None
                or parking_stage not in self._PREDICTIVE_GUARD_STAGES
                or runner.status is not MissionStatus.RUNNING):
            return False
        # 예측도 **정확한 방향을 안다고 가정할 때만** 의미가 있다. heading 이
        # LAST_VALID 로 얼어붙으면 heading_rate 가 구조적으로 0 이 되어 "차가
        # 회전하지 않는다"는 틀린 전제로 회전 footprint 를 외삽한다.
        # 실측 run_20260827_234439: stale heading 으로 predicted 31.0mm 를
        # 만들어 FINAL 에서 정지시켰는데, 방향 무관 최악값은 3.5mm 였다.
        # 방향을 못 믿을 때의 판정은 _check_boundary 의 보수 계약이 담당한다.
        if view.heading_source not in {"FRONT_CUSHION", "TRAJECTORY"}:
            return False
        motion = getattr(self, "_boundary_motion", None)
        if motion is None:
            motion = self._boundary_motion = {}
        previous = motion.get(car_id)
        now = view.last_obs_time
        if previous is None or now <= previous[0]:
            if now > 0.0:
                motion[car_id] = (now, view.position_mm[0],
                                  view.position_mm[1], view.heading_deg)
            return False

        dt = now - previous[0]
        dx = view.position_mm[0] - previous[1]
        dy = view.position_mm[1] - previous[2]
        displacement = math.hypot(dx, dy)
        # Accumulate across sparse/noisy frames until the same displacement
        # already required by trajectory heading estimation is observable.
        if displacement < self.config.heading_min_move_mm:
            return False

        horizon = self.config.boundary_prediction_horizon_s
        heading_rate = ((view.heading_deg - previous[3] + 180.0)
                        % 360.0 - 180.0) / dt
        predicted = (
            view.position_mm[0] + dx / dt * horizon,
            view.position_mm[1] + dy / dt * horizon,
            (view.heading_deg + heading_rate * horizon) % 360.0,
        )
        w, h = self.config.lot_width_mm, self.config.lot_height_mm
        predicted_overflow = max(
            max(-px, px - w, -py, py - h)
            for px, py in _car_footprint(*predicted))
        margin = self.config.boundary_hard_margin_mm
        uncertainty = float(getattr(
            self.config, "boundary_measurement_uncertainty_mm", 10.0))
        uncertain_limit = margin + uncertainty
        # From inside the normal band, preserve the existing prediction to the
        # 20 mm boundary.  Once a measurement is already in the uncertainty
        # band, require a genuine outward prediction (not merely predicted >20
        # because current is 22) or a projected crossing above 30 mm.
        increase = float(getattr(
            self.config, "boundary_uncertain_increase_mm", 1.0))
        prediction_threshold = margin
        if current_overflow_mm > margin:
            prediction_threshold = min(
                uncertain_limit, current_overflow_mm + increase)
        if predicted_overflow <= prediction_threshold:
            # Keep accumulating from the earlier observation while the
            # projected footprint is moving outward.  Sparse 4 FPS frames may
            # each move slightly less than heading_min_move_mm; resetting here
            # would hide a persistent approach to the wall.
            if predicted_overflow <= current_overflow_mm:
                motion[car_id] = (now, view.position_mm[0],
                                  view.position_mm[1], view.heading_deg)
            return False

        runner.mission.request_replan("PREDICTED_BOUNDARY")
        runner.prepare_route_switch()
        self.server.stop_control(car_id)
        motion.pop(car_id, None)
        self._emit_event(
            "PREDICTIVE_BOUNDARY_STOP", car_id=car_id,
            current_overflow_mm=round(current_overflow_mm, 1),
            predicted_overflow_mm=round(predicted_overflow, 1),
            horizon_s=horizon)
        self.dashboard.push_event(
            "predictive_boundary_stop", car_id=car_id,
            predicted_overflow_mm=round(predicted_overflow, 1))
        return True

    def _on_auto_host_status(self, car_id: int, prev: MissionStatus,
                             status: MissionStatus) -> None:
        """AUTO_HOST 미션 상태 변화 → 기존 슬롯·대시보드 로직에 연결.

        이 경로가 없으면 FINAL 도착이 파이프라인까지 올라오지 않아 슬롯이
        영원히 비어 있는 것으로 남는다 (WAYPOINT_AUTO 의 PARKED_CHECK 에 해당).
        """
        if status is MissionStatus.DONE:
            # PARKED 는 terminal zero 다. stale DONE 콜백이 여기 도착해도
            # 주차 lifecycle 을 다시 시작시키면 안 된다.
            if self._parking_stage.get(car_id) in self._PARKING_TERMINAL_STAGES:
                return
            if self._parking_stage.get(car_id) == "ENTRY_STAGING":
                track_id = self.track_of_car.get(car_id)
                view = self.views.get(track_id) if track_id is not None else None
                if view is not None:
                    self._parking_stage[car_id] = "ENTRY_STAGING_PENDING"
                    self._entry_staging_wait[car_id] = view.last_obs_time
                return
            # setup 기동의 계산상 종점이 아니라 DONE 뒤의 fresh camera pose 로
            # LOCAL rear planner 를 다시 호출한다.
            if (self.rear_parking_mode
                    and self._parking_stage.get(car_id) == "SETUP"):
                track_id = self.track_of_car.get(car_id)
                view = self.views.get(track_id) if track_id is not None else None
                if view is not None:
                    self._parking_stage[car_id] = "PARKING_AFTER_SETUP_PENDING"
                    self._parking_plan_wait[car_id] = view.last_obs_time
                    return
            # 주차 기동이 끝났다 = waypoint 에 도착했다는 뜻일 뿐이다.
            # 주차선과 나란한지는 정지 후 fresh pose 로 따로 평가한다.
            if (self.rear_parking_mode
                    and self._parking_stage.get(car_id)
                        in self._FINAL_EVAL_SOURCE_STAGES):
                track_id = self.track_of_car.get(car_id)
                view = self.views.get(track_id) if track_id is not None else None
                if view is not None:
                    self._parking_stage[car_id] = "FINAL_EVAL_PENDING"
                    self._parking_plan_wait[car_id] = view.last_obs_time
                    return
            # 후면주차 GLOBAL 구간이 끝났으면 여기가 **인계 경계**다.
            # PARKED 로 확정하지 않고, 정지 상태에서 새로 관측한 pose 로
            # LOCAL 주차 경로를 만들어 갈아 끼운다.
            if (self.rear_parking_mode
                    and self._parking_stage.get(car_id) != "PARKING"):
                track_id = self.track_of_car.get(car_id)
                view = self.views.get(track_id) if track_id is not None else None
                if view is not None:
                    self._parking_stage[car_id] = "PARKING_HANDOFF_PENDING"
                    self._parking_plan_wait[car_id] = view.last_obs_time
                    return
            # 정지 재확인은 카메라를 보는 이쪽 몫이다 (§11). 프레임 루프에서 판정한다.
            log.info("car %d: AUTO_HOST 최종 waypoint 도착 — 정지 확인 중", car_id)
        elif status is MissionStatus.REPLAN_REQUIRED:
            track_id = self.track_of_car.get(car_id)
            view = self.views.get(track_id) if track_id is not None else None
            failed = getattr(self.auto_hosts[car_id], "failed_target", None)
            if self._parking_stage.get(car_id) in {
                    "ENTRY_STAGING", "ENTRY_STAGING_PENDING"}:
                if view is None:
                    self._entry_staging_fault(
                        car_id, "ENTRY_STAGING_POSE_UNAVAILABLE")
                    return
                self.auto_hosts[car_id].prepare_route_switch()
                self.server.stop_control(car_id)
                self._parking_stage[car_id] = "ENTRY_STAGING_PENDING"
                self._entry_staging_wait[car_id] = view.last_obs_time
                self._emit_event(
                    "ENTRY_STAGING_WAIT", car_id=car_id,
                    reason=self.auto_hosts[car_id].replan_reason)
                return
            if (view is not None and failed is not None
                    and self._is_global_handoff_terminal(car_id, failed)
                    and self._handoff_region_reached(view, failed)):
                self._begin_parking_handoff(
                    view, self.auto_hosts[car_id], failed,
                    reason=self.auto_hosts[car_id].replan_reason
                           or "HANDOFF_TERMINAL_REPLAN")
                return
            if self._route_final_quality_replan(car_id, view):
                return
            if (self.rear_parking_mode
                    and self._parking_stage.get(car_id)
                        in self._REAR_RECOVERABLE_STAGES):
                attempts = self._parking_recovery_attempts.get(car_id, 0) + 1
                self._parking_recovery_attempts[car_id] = attempts
                if attempts > self.config.max_parking_recovery_attempts:
                    self._parking_stage[car_id] = "WAIT_RECOVERY_EXHAUSTED"
                    self.server.stop_control(car_id)
                    self._emit_event("FAULT", car_id=car_id,
                                     reason="PARKING_RECOVERY_EXHAUSTED",
                                     attempts=attempts - 1)
                    self.dashboard.push_event(
                        "parking_recovery_exhausted", car_id=car_id,
                        attempts=attempts - 1)
                    return
                track_id = self.track_of_car.get(car_id)
                view = self.views.get(track_id) if track_id is not None else None
                if view is not None:
                    # REPLAN_REQUIRED is already zero-control. Wait for the next
                    # distinct camera observation before planning a setup maneuver.
                    # 여기는 reason 을 이미 _route_final_quality_replan 이
                    # 걸러낸 뒤다. 남은 사유(boundary/센서 손실 등)는 최종
                    # 정렬로 우회시키면 안 되므로 항상 setup 으로 보낸다.
                    self._parking_stage[car_id] = "SETUP_PENDING"
                    self._parking_setup_wait[car_id] = view.last_obs_time
                    self._emit_event("PARKING_SETUP_WAIT", car_id=car_id,
                                     reason=self.auto_hosts[car_id].replan_reason)
                    return
            # 먼저 후진 복구를 시도한다. 전진으로 못 잡는 자세라 재계획을 해봐야
            # 같은 기하가 다시 나오기 때문이다 (같은 슬롯 = 같은 원호).
            if self._recover_auto_host(car_id):
                return
            log.info("car %d: 후진 복구 불가 — 전체 재계획", car_id)
            self._replan_auto_host(car_id)

    # 후면주차 진행 중인 stage. 여기서 REPLAN_REQUIRED 가 나면 **항상** rear
    # coordinator(STOP → fresh pose → 현재 자세 기준 setup)로 보낸다.
    #
    # 예전에는 {"PARKING","SETUP"} 만 봤다. 그래서 handoff 직후
    # (PARKING_HANDOFF_PENDING) 나 setup 대기 중에 REPLAN 이 나면 통로 인계용
    # legacy 전역 재계획으로 빠졌고, 거기서 identical replan 이 감지되면
    # 읽는 곳이 없는 WAIT_REPEATED_REPLAN 으로 죽었다
    # (run_20260827_212848: perception 99% 정상인데 48초 무동작).
    #
    # 예산이 소진된 terminal stage 는 넣지 않는다 — 그건 다시 시도할 상태가
    # 아니라 명시적으로 끝난 상태다.
    _REAR_RECOVERABLE_STAGES = frozenset({
        "PARKING", "SETUP", "SETUP_PENDING",
        "PARKING_AFTER_SETUP_PENDING", "PARKING_HANDOFF_PENDING",
        "WAIT_SAFE_RECOVERY", "WAIT_SAFE_ROUTE", "WAIT_REPEATED_REPLAN",
        "FINAL_EVAL_PENDING", "FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE",
        "PARKED_VERIFY",
    })

    # 카메라가 늦게 준 것이지 물리적으로 위험한 것이 아닌 controller latch 사유.
    # HostController._STALE_REASONS 와 같은 집합이다 (그쪽은 import 하지 않는다 —
    # 파이프라인은 controller 내부 상수에 의존하지 않는다).
    _OBSERVATION_FAULT_REASONS = frozenset({
        "POSE_STALE", "POSE_INVALID", "NO_HEADING",
    })

    # 관측 지연 latch 에서 되살릴 stage. 조건은 "지금 주차가 명목상 진행
    # 중인가" 다 — 예산이 끝난 WAIT_* terminal 과, 정지한 채 관측을 세는 것이
    # 정상인 PARKED_VERIFY 는 넣지 않는다 (거기서 되살리면 확정된 주차를
    # 되돌린다).
    #
    # 최종 정렬 구간(FINAL_*)이 빠져 있었다. 그 구간도 실제로 차를 모는
    # 구간이라 카메라 gap 을 똑같이 맞는데, 되살릴 경로가 없어 8s
    # PARKING_STALLED watchdog 까지 가고 그때마다 recovery 예산을 하나씩
    # 태웠다. 실측 run_20260903_161534: FINAL_ALIGNMENT 로 (482.1,1083.7)
    # 에서 슬롯으로 되돌아오던 중 POSE_STALE 두 번 — t=49.86 은 t=60.72 에,
    # t=64.03 은 t=75.59 에야 풀려(각 약 11초) 예산 3개를 모두 소진하고
    # PARKING_RECOVERY_EXHAUSTED 로 끝났다. 그 사이 pose_age 는 중앙값
    # 125ms 로 정상이었다.
    #
    # GLOBAL 도 빠져 있었다. 인계 지점까지 통로를 순항하는 그 구간도 실제로
    # 차를 모는 구간인데, 되살릴 경로가 없어 단 한 번의 관측 공백이 미션을
    # 영구 정지시켰다. 실측 run_20260904_183319: t=21.78 에 pose_age 가
    # 533ms 로 max_pose_age_s(0.5s)를 한 번 넘겨 POSE_STALE latch. 그 뒤
    # **66초 동안** pose_age 는 2.9~247ms 로 완전히 정상이었고 COMM 도
    # 무결(comm_fault 0건)이었는데, stage 가 GLOBAL 이라 복구 계약을 타지
    # 못하고 throttle/steering 0 인 채로 끝났다.
    #
    # 수정 전 baseline 이 GLOBAL 순항까지 안정적으로 못 갔기 때문에 드러나지
    # 않았던 구멍이다. 계약은 그대로다 — 정지 유지, 옛 route 재개 금지,
    # 현재 fresh pose 에서 새 경로를 검증한 뒤에만 재무장.
    #
    # ENTRY_STAGING 도 같은 이유로 넣는다 — 이제 증거가 3/3 이다.
    # run_20260904_205954 t=17.6 / _210209 t=12.5 / _210321 t=30.4 에서
    # pose_age 가 각각 594~599ms 로 max_pose_age_s(0.5s)를 **한 번** 넘겨
    # POSE_STALE latch 가 걸렸고, 그 뒤 210209 는 44초, 210321 은 48초 동안
    # pose_age 가 300ms 를 한 번도 넘지 않았는데도 zero 로 굳었다. 205954 만
    # 우연히 COMM_TIMEOUT resync 가 새 route 를 실어 14초 만에 빠져나왔다.
    #
    # staging 이 자기 boundary 생존 경로를 갖고 있다는 이유로 지난 사이클에
    # 제외했는데, 그 경로는 미션이 DONE/REPLAN 으로 **끝났을 때** 도는 것이라
    # authority 가 FAULTED 로 잠긴 상태에는 도달하지 못한다.
    #
    # ENTRY_STAGING_PENDING 은 넣지 않는다: 그 stage 는 이미 정지한 채
    # 관측을 세는 것이 정상 동작이고(_maybe_resume_entry_staging 의
    # _phase_boundary_stopped), 3 run 중 그 stage 에서 latch 된 증거가 없다.
    _OBSERVATION_RESUMABLE_STAGES = frozenset({
        "GLOBAL", "ENTRY_STAGING",
        "PARKING", "SETUP", "SETUP_PENDING",
        "PARKING_AFTER_SETUP_PENDING", "PARKING_HANDOFF_PENDING",
        "FINAL_EVAL_PENDING", "FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE",
    })

    # 주차 기동이 끝난 뒤 최종 자세 평가로 보내야 하는 stage.
    # runtime predictive boundary 감시 대상. 최종 정렬/직선후진은 슬롯 뒤쪽
    # 경계에 가장 가까이 가는 기동이라 반드시 포함해야 한다. 예전에는
    # {PARKING, SETUP} 뿐이라 새 FINAL_* stage 가 보호 밖이었다.
    _PREDICTIVE_GUARD_STAGES = frozenset({
        "PARKING", "SETUP", "FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE",
        # 입구 staging 도 실제로 차를 모는 구간이다. 빠져 있으면 예측 보호가
        # 전혀 걸리지 않는다 (실측 231000: 12초 주행 내내 미적용).
        "ENTRY_STAGING",
    })

    _FINAL_EVAL_SOURCE_STAGES = frozenset({
        "PARKING", "FINAL_ALIGNMENT", "FINAL_STRAIGHT_REVERSE",
    })

    # 여기 들어오면 주차 lifecycle 은 끝났다. 어떤 stale 콜백도 되살리지 않는다.
    _PARKING_TERMINAL_STAGES = frozenset({
        "PARKED", "WAIT_FINAL_ALIGNMENT_FAILED", "WAIT_ENTRY_STAGING_FAILED",
    })

    # 슬롯 최종 구간에서 이 사유의 REPLAN 은 "경로 실패" 가 아니라 "최종 자세가
    # 아직 주차선과 안 맞는다" 는 뜻이다. 재접근이 아니라 최종 정렬로 보낸다.
    #
    # PREDICTED_BOUNDARY 는 guard 가 이미 zero + REPLAN 으로 멈춘 뒤의
    # 예방 정지다. 슬롯 final region 안이라면 그 route 를 재개하지 않고,
    # fresh trusted pose 를 요구하는 최종 평가기로 넘긴다. 이것은 boundary
    # 우회가 아니라 STOP -> fresh Pose -> FINAL_POSE_EVAL 전이이다.
    #
    # 실제 침범인 BOUNDARY_HARD 와 센서 손실(REVERSE_HEADING_TIMEOUT,
    # POSE_STALE, COMM_*)은 **절대 넣지 않는다** — 그것들은 final evaluator
    # handoff 가 아니라 기존 물리/관측 fault recovery 를 유지해야 한다.
    _FINAL_QUALITY_REPLAN_REASONS = frozenset({
        "HEADING_OUT_OF_TOLERANCE",
        "PREDICTED_BOUNDARY",
    })

    def _route_final_quality_replan(self, car_id: int,
                                    view: VehicleView | None) -> bool:
        """Send a final-region quality REPLAN to the final evaluator.

        예전에는 슬롯 안에 들어간 차가 주차선과 어긋나 HEADING_OUT_OF_TOLERANCE
        를 내면 generic rear recovery(재접근)로 갔다. 그래서 **FINAL_ALIGNMENT
        가 가장 필요한 상태가 오히려 FINAL_ALIGNMENT 진입조건(mission DONE)을
        만족하지 못하는** 구조였다.

        여기서는 stage/waypoint 이름이 아니라 slot-local 기하로 최종 구간을
        판정한다. 그 판정은 heading 을 쓰지 않으므로 heading 이 LAST_VALID 로
        굳어 있어도 오염되지 않는다. 실제 정렬 계획은
        ``_maybe_evaluate_final_pose`` 가 trusted heading 을 얻은 뒤에만 한다.

        recovery 예산은 쓰지 않는다 — 이건 실패한 재계획이 아니라 정상적인
        최종 단계 전이이고, 반복은 max_final_alignment_attempts 가 따로 막는다.
        """
        if not self.rear_parking_mode or view is None:
            return False
        stage = self._parking_stage.get(car_id)
        if stage not in self._FINAL_EVAL_SOURCE_STAGES:
            return False
        runner = self.auto_hosts.get(car_id)
        reason = str(getattr(runner, "replan_reason", "") or "").upper()
        if reason not in self._FINAL_QUALITY_REPLAN_REASONS:
            return False
        slot_id = self._auto_host_slot.get(car_id)
        if slot_id is None:
            return False
        # 기하 게이트는 최종 lifecycle 에 **들어올 때만** 적용한다.
        #
        # in_final_region 의 depth 하한은 -slot.length/2 (B1 기준 -150mm) 인데,
        # FINAL_ALIGNMENT 가 겨냥하는 staging 자세는 -(length/2 + CAR_LENGTH/2)
        # = -275mm 이고 alignment_goal_test 가 받아주는 대역은
        # [staging-200, staging+100] = [-475, -175] 이다. 두 구간의 교집합은
        # **비어 있다** — 가장 얕은 목표 -175 조차 -150 보다 25mm 바깥이다.
        #
        # 즉 FINAL_ALIGNMENT 는 설계상 자기가 이 판정 밖으로 차를 몰고 나간다.
        # 그 상태에서 REPLAN 이 나면 여기서 걸러져 아래 generic 분기로 떨어지고,
        # stage 가 SETUP_PENDING 으로 바뀌면서 통로 재접근 + 전체 후면주차가
        # 다시 시작된다. 실측 run_20260903_191010 t=34.9 (438.5,696.0,272.1,
        # depth -354) / run_20260903_191220 t=55.6 — 둘 다 "들어감→나옴" 루프의
        # 시작점이 정확히 여기다. 171158 도 같은 패턴이었다.
        #
        # 통로를 지나가던 차를 최종 정렬로 오인하지 않으려고 이 게이트를 넣었고
        # (실측 032539: heading 오차 129도), 그 오인은 stage PARKING /
        # SETUP_PENDING 에서 일어났다. FINAL_ALIGNMENT / FINAL_STRAIGHT_REVERSE
        # 는 지나가는 차가 아니라 **시스템이 방금 계획한 보정을 실행 중인 차**다.
        # 그래서 진입(PARKING)에만 기하 게이트를 걸고, 이미 보정 중인 차는
        # 최종 평가기가 계속 맡는다. 반복은 max_final_alignment_attempts(3) 가
        # 이미 막고 있고 이 경로는 recovery 예산을 쓰지 않는다.
        if (stage == "PARKING"
                and not in_final_region(default_slot_specs()[slot_id],
                                        view.position_mm[0],
                                        view.position_mm[1])):
            return False
        self._parking_stage[car_id] = "FINAL_EVAL_PENDING"
        self._parking_plan_wait[car_id] = view.last_obs_time
        event = ("FINAL_SAFETY_STOP_EVAL"
                 if reason == "PREDICTED_BOUNDARY"
                 else "FINAL_QUALITY_REPLAN")
        self._emit_event(event, car_id=car_id, slot=slot_id, reason=reason)
        self.dashboard.push_event(event.lower(), car_id=car_id,
                                  slot=slot_id, reason=reason)
        return True

    def _maybe_evaluate_final_pose(self, view: VehicleView) -> bool:
        """STOP → fresh Pose → 최종 자세 평가 → 정렬 / 직선후진 / PARKED 검증.

        waypoint 도착만으로 PARKED 로 확정하지 않는다. 슬롯 200x300mm 에 차량
        250x150mm 라 완벽히 정렬해도 좌우 여유가 25mm 뿐이고, 비스듬하면
        차체가 주차선을 넘는다.
        """
        car_id = view.car_id
        if (car_id is None
                or self._parking_stage.get(car_id) != "FINAL_EVAL_PENDING"):
            return False
        if not self._phase_boundary_stopped(
                view, self._parking_plan_wait.get(car_id, 0.0),
                "FINAL_POSE_EVAL"):
            return False
        if not self._require_critical_heading(view, "FINAL_POSE_EVAL"):
            return False
        slot_id = self._auto_host_slot.get(car_id)
        runner = self.auto_hosts.get(car_id)
        if slot_id is None or runner is None:
            return False
        self._parking_plan_wait.pop(car_id, None)

        spec = default_slot_specs()[slot_id]
        verdict = evaluate_final_pose(
            spec, view.position_mm[0], view.position_mm[1], view.heading_deg)
        self._emit_event(
            "FINAL_POSE_EVAL", car_id=car_id, slot=slot_id,
            action=verdict.action, reason=verdict.reason,
            depth_mm=round(verdict.local.depth_mm, 1),
            lateral_mm=round(verdict.local.lateral_mm, 1),
            heading_err_deg=round(verdict.local.heading_err_deg, 1))

        if verdict.parked:
            self._parking_stage[car_id] = "PARKED_VERIFY"
            self._parked_confirmations[car_id] = 0
            self._parked_last_obs.pop(car_id, None)
            return True

        attempts = self._final_alignment_attempts.get(car_id, 0) + 1
        self._final_alignment_attempts[car_id] = attempts
        limit = int(getattr(self.config, "max_final_alignment_attempts", 3))
        if attempts > limit:
            # 조용히 멈추지 않는다 — 명시적 terminal 이다.
            self._parking_stage[car_id] = "WAIT_FINAL_ALIGNMENT_FAILED"
            self.server.stop_control(car_id)
            runner.stop()
            self._emit_event("FAULT", car_id=car_id, slot=slot_id,
                             reason="FINAL_ALIGNMENT_EXHAUSTED",
                             attempts=attempts - 1)
            self.dashboard.push_event("final_alignment_exhausted",
                                      car_id=car_id, attempts=attempts - 1)
            return True

        obstacles = self._planner_obstacle_poses(view)
        route_id = self.orchestrator.next_route_id()
        if verdict.action == "STRAIGHT_REVERSE":
            wps = build_final_straight_reverse_waypoints(
                spec, route_id, from_pose=view.position_mm)
            next_stage = "FINAL_STRAIGHT_REVERSE"
        else:
            wps = build_final_alignment_waypoints(
                spec, route_id, from_pose=view.position_mm,
                from_heading_deg=view.heading_deg,
                obstacle_poses=obstacles)
            next_stage = "FINAL_ALIGNMENT"

        if not wps or not self._trajectory_safe(view, wps, slot_id=slot_id,
                                                recovery=True):
            # 최종 정렬 기동을 못 만들면 넓은 setup 탐색으로 넘긴다 (bounded).
            self._emit_event("RECOVERY_REJECTED", car_id=car_id, slot=slot_id,
                             route_id=route_id,
                             reason=f"NO_SAFE_{next_stage}")
            if not self._escalate_repeated_replan(car_id, view, next_stage):
                self._parking_stage[car_id] = "WAIT_SAFE_RECOVERY"
                self.server.stop_control(car_id)
                self._emit_event("FAULT", car_id=car_id, slot=slot_id,
                                 reason="NO_SAFE_FINAL_ALIGNMENT")
            return True

        self._parking_stage[car_id] = next_stage
        runner.load_route(wps)
        self._reactivate_after_heading_fault(car_id)
        self._reactivate_after_boundary_escape(car_id)
        self._emit_route(wps, car_id=car_id, recovery=True)
        self.dashboard.push_event("final_alignment", car_id=car_id,
                                  slot=slot_id, action=verdict.action,
                                  attempt=attempts)
        return True

    def _maybe_verify_parked(self, view: VehicleView) -> bool:
        """서로 다른 fresh 관측 N회가 모두 조건을 만족해야 PARKED 로 확정한다."""
        car_id = view.car_id
        if (car_id is None
                or self._parking_stage.get(car_id) != "PARKED_VERIFY"):
            return False
        slot_id = self._auto_host_slot.get(car_id)
        runner = self.auto_hosts.get(car_id)
        if slot_id is None or runner is None:
            return False
        if not view.is_stationary(self.config.stationary_tolerance_mm,
                                  self.config.stationary_window):
            return False
        if not self._critical_heading_ready(view):
            return False
        last = self._parked_last_obs.get(car_id)
        if last is not None and view.last_obs_time <= last:
            return False               # 같은 프레임을 두 번 세지 않는다
        self._parked_last_obs[car_id] = view.last_obs_time

        spec = default_slot_specs()[slot_id]
        verdict = evaluate_final_pose(
            spec, view.position_mm[0], view.position_mm[1], view.heading_deg)
        if not verdict.parked:
            # 확인 중에 조건이 깨졌다 — 다시 평가부터.
            self._parked_confirmations.pop(car_id, None)
            self._parking_stage[car_id] = "FINAL_EVAL_PENDING"
            self._parking_plan_wait[car_id] = view.last_obs_time
            self._emit_event("PARKED_VERIFY_FAILED", car_id=car_id,
                             slot=slot_id, reason=verdict.reason)
            return True

        count = self._parked_confirmations.get(car_id, 0) + 1
        self._parked_confirmations[car_id] = count
        required = int(getattr(self.config, "parked_confirm_observations", 3))
        self._emit_event("PARKED_CONFIRMING", car_id=car_id, slot=slot_id,
                         count=count, required=required)
        if count < required:
            return True

        runner.confirm_parked()
        runner.stop()
        self._parked_confirmations.pop(car_id, None)
        self._parked_last_obs.pop(car_id, None)
        if not hasattr(self, "_parked_obstacles"):
            self._parked_obstacles = {}
        self._parked_obstacles[car_id] = (
            view.position_mm[0], view.position_mm[1], float(view.heading_deg))
        self._parking_stage[car_id] = "PARKED"
        self._emit_event(
            "PARKED", car_id=car_id, slot=slot_id,
            depth_mm=round(verdict.local.depth_mm, 1),
            lateral_mm=round(verdict.local.lateral_mm, 1),
            heading_err_deg=round(verdict.local.heading_err_deg, 1))
        self._on_parked(car_id, slot_id)
        return True

    # watchdog 이 "진행 중"으로 보지 않는 stage. 명시적 sensor 대기이거나
    # 이미 명시적으로 끝난 상태라 stall 이 아니다.
    _STALL_EXEMPT_STAGES = frozenset({
        "SETUP_PENDING", "PARKING_AFTER_SETUP_PENDING",
        "PARKING_HANDOFF_PENDING", "ENTRY_STAGING_PENDING",
        "WAIT_FRESH_HEADING_FAULT",
        # A deterministic direct+setup planner rejection is an explicit safe
        # terminal wait.  Re-running it from the same stationary pose produced
        # the recorded 278/275/275 mm sequence without any new information.
        "WAIT_SAFE_RECOVERY",
        "WAIT_RECOVERY_EXHAUSTED",
        # 아래 둘은 정지한 채 fresh 관측을 세는 것이 정상 동작이다.
        "FINAL_EVAL_PENDING", "PARKED_VERIFY",
        "PARKED", "WAIT_FINAL_ALIGNMENT_FAILED", "WAIT_ENTRY_STAGING_FAILED",
    })

    def _check_parking_progress(self, view: VehicleView) -> None:
        """Last-resort silent-deadlock detector.  Never moves the vehicle.

        Fires only when parking is nominally active but nothing at all is
        happening: car stationary, control zero, perception valid, no explicit
        sensor wait, no route/recovery transaction.  That combination is not a
        safe stop — it is a state nobody is driving.
        """
        car_id = view.car_id
        if car_id is None or not self.rear_parking_mode:
            return
        if not hasattr(self, "_stall_since"):
            self._stall_since = {}
        stage = self._parking_stage.get(car_id)
        runner = self.auto_hosts.get(car_id)
        active = (stage in self._REAR_RECOVERABLE_STAGES
                  and stage not in self._STALL_EXEMPT_STAGES
                  and runner is not None
                  and self._critical_heading_ready(view)
                  and view.is_stationary(self.config.stationary_tolerance_mm,
                                         self.config.stationary_window)
                  and self._control_is_zero(car_id))
        if not active:
            self._stall_since.pop(car_id, None)
            return
        started = self._stall_since.setdefault(car_id, view.last_obs_time)
        timeout_s = float(getattr(self.config, "parking_stall_timeout_s", 8.0))
        if view.last_obs_time - started < timeout_s:
            return
        self._stall_since.pop(car_id, None)
        self._emit_event("PARKING_STALLED", car_id=car_id, stage=stage,
                         stalled_s=round(view.last_obs_time - started, 2))
        self.dashboard.push_event("parking_stalled", car_id=car_id, stage=stage)
        # 여기서 차를 움직이지 않는다. 현재 pose 로 recovery coordinator 만 부른다.
        self._escalate_repeated_replan(car_id, view, "PARKING_STALLED")

    def _control_is_zero(self, car_id: int) -> bool:
        """Whether the last computed control output was zero (no motion asked)."""
        runner = self.auto_hosts.get(car_id)
        result = getattr(runner, "last_tick_result", None)
        command = getattr(result, "command", None)
        throttle = getattr(command, "throttle", None)
        if throttle is None:
            return True                 # 계산 결과가 없으면 구동 중이 아니다
        return abs(float(throttle)) < 1e-9

    def _escalate_repeated_replan(self, car_id: int, view: VehicleView,
                                  reason: str) -> bool:
        """Send a stuck direct replan to the rear setup coordinator.

        An identical replan means *this planner* has no further move from this
        pose — not that the vehicle should stop forever.  The bounded setup
        planner searches a different maneuver family, so escalate to it through
        the normal STOP → fresh pose → validate contract.

        Bounded by the existing ``max_parking_recovery_attempts`` budget; when
        that is spent the caller ends in an explicit terminal fault instead.
        """
        if not self.rear_parking_mode:
            return False
        if self._parking_stage.get(car_id) not in self._REAR_RECOVERABLE_STAGES:
            return False
        attempts = self._parking_recovery_attempts.get(car_id, 0) + 1
        self._parking_recovery_attempts[car_id] = attempts
        if attempts > self.config.max_parking_recovery_attempts:
            self._parking_stage[car_id] = "WAIT_RECOVERY_EXHAUSTED"
            self.server.stop_control(car_id)
            self._emit_event("FAULT", car_id=car_id,
                             reason="PARKING_RECOVERY_EXHAUSTED",
                             attempts=attempts - 1)
            self.dashboard.push_event("parking_recovery_exhausted",
                                      car_id=car_id, attempts=attempts - 1)
            return True
        # 같은 signature 로 다시 걸리지 않게 비운다 — 다음 판단은 setup 이
        # 만든 새 자세에서 이루어진다.
        self._last_replan_signature.pop(car_id, None)
        # escalation 은 "이 planner 로는 수가 없다" 는 탈출구다. 방금 실패한
        # 평가로 되돌리면 같은 실패를 반복하므로, 항상 더 넓은 setup 탐색으로
        # 보낸다 (final region 이어도 마찬가지다).
        self._parking_stage[car_id] = "SETUP_PENDING"
        self._parking_setup_wait[car_id] = view.last_obs_time
        self._emit_event("PARKING_SETUP_WAIT", car_id=car_id,
                         reason=f"REPEATED_IDENTICAL_REPLAN:{reason}")
        return True

    def _stale_observation_latched(self, car_id: int) -> bool:
        """Whether this car is latched only because observations went missing.

        run_20260827_213110 / _213150: a 0.7 s and a 21.7 s camera gap exceeded
        ``max_pose_age_s``; the controller latched FAULTED and the pipeline had
        no path back, so the run sat at zero until it was aborted.
        """
        if (self._parking_stage.get(car_id)
                not in self._OBSERVATION_RESUMABLE_STAGES):
            return False
        runner = self.auto_hosts.get(car_id)
        authority = getattr(getattr(runner, "host", None), "authority", None)
        if authority is None or not getattr(authority, "is_faulted", False):
            return False
        reason = str(getattr(authority, "fault_reason", "") or "").upper()
        return reason in self._OBSERVATION_FAULT_REASONS

    def _post_recovery_stage(self, car_id: int, view: VehicleView) -> str:
        """Where a car should resume after a wait/recovery, by geometry.

        슬롯 최종 구간 안에 있는 차를 generic setup(재접근)으로 보내면
        FINAL_ALIGNMENT 가 영원히 실행되지 않는다. 위치만으로 판정하므로
        heading 이 아직 LAST_VALID 여도 안전하다 — 실제 정렬 계획은
        _maybe_evaluate_final_pose 가 trusted heading 을 얻은 뒤에만 한다.
        """
        if not self.rear_parking_mode:
            return "SETUP_PENDING"
        # staging 중이던 차는 staging 경계로 되돌린다. 거기가 이미 "정지 →
        # fresh pose → 다시 계획" 을 하는 지점이라 새 경로가 필요 없다.
        # 입구 자세를 generic 주차 setup 으로 보내면 슬롯 앞이 아닌 곳에서
        # rear 계획을 시도하게 된다.
        if self._parking_stage.get(car_id) == "ENTRY_STAGING":
            return "ENTRY_STAGING_PENDING"
        slot_id = self._auto_host_slot.get(car_id)
        if slot_id is None:
            return "SETUP_PENDING"
        if in_final_region(default_slot_specs()[slot_id],
                           view.position_mm[0], view.position_mm[1]):
            return "FINAL_EVAL_PENDING"
        return "SETUP_PENDING"

    def _maybe_resume_heading_fault(self, view: VehicleView) -> bool:
        """Leave an observation-wait latch once a trusted heading is back.

        That state is a *waiting* state, not a physical safety stop: it is
        entered because no measured heading arrived within the timeout.  But
        entering it calls ``runner.stop()``, which latches authority FAULTED,
        and nothing in the pipeline ever re-armed it — so even after perception
        recovered the car stayed at zero forever (liveness violation seen in
        run_20260827_212955 / _213332).

        Resuming is not a stale resume: authority stays faulted here, the old
        route is never continued, and the car only moves again after the normal
        setup coordinator validates a **new** route from the current fresh pose.
        """
        car_id = view.car_id
        if car_id is None:
            return False
        if self._parking_stage.get(car_id) != "WAIT_FRESH_HEADING_FAULT":
            if car_id in getattr(self, "_heading_fault_hold", ()):
                # 이 fault 에 대한 재활성 계약은 **이미 시작됐다**. 여기서
                # 다시 들어오면 안 된다.
                #
                # _stale_observation_latched 는 전이 뒤에도 참으로 남는다:
                # SETUP_PENDING 은 그 stage 집합에 있고, authority 는 새 route
                # 가 검증될 때까지(=_reactivate_after_heading_fault) FAULTED 다.
                # 그래서 관측마다 재진입해 아래에서 찍는 fresh-observation
                # marker 를 매번 현재 시각으로 다시 찍었다. 파이프라인은 같은
                # view 로 곧바로 _maybe_start_parking_setup 을 부르는데, 그
                # 게이트는 last_obs_time > marker 를 요구하므로 방금 같은 값이
                # 찍힌 경계를 **영원히** 통과하지 못한다.
                #
                # 실측 run_20260903_154026 (route 3 / wp 2 ALIGN): 625ms 관측
                # 공백 한 번으로 POSE_STALE → zero. 관측은 300ms 만에 복귀해
                # 이후 pose_age 중앙값 125ms 였는데도 HEADING_RECOVERED 가 약
                # 100회 반복되며 SETUP_PENDING 에 232 tick 갇혔고, 새 route 0건
                # / PWM 0 으로 25초 뒤 수동 중단됐다.
                #
                # 계약은 그대로다 — 정지 유지, 옛 route 재개 금지, stale pose
                # 주행 금지. 계약을 **한 번만** 태울 뿐이다. hold 는
                # _reactivate_after_heading_fault 가 새 route 검증 후 푼다.
                return False
            # 카메라 gap 으로 controller 가 POSE_STALE 로 latch 한 경우도 같은
            # 종류다 — 물리 위험이 아니라 관측이 늦은 것이다. 관측이 돌아왔고
            # 주차 진행 중이었다면 같은 재활성 계약을 태운다.
            if not self._stale_observation_latched(car_id):
                return False
            latched_here = True
        else:
            latched_here = False
        # ── 표시는 **전제조건을 통과한 뒤에** 찍는다 ──────────────────────
        #
        # 예전에는 latch 를 확인한 그 자리에서 곧바로 hold 를 찍었다. 그러면
        # 아래 두 게이트 중 하나라도 실패한 첫 프레임에서 return 하는데,
        # 다음 프레임부터는 함수 입구의 `car_id in _heading_fault_hold` 가
        # 곧바로 return 시켜 **영원히 다시 못 들어온다**. hold 는
        # _reactivate_after_heading_fault 만 지우고, 거기까지 가려면 route 가
        # 실려야 하는데 그 route 는 여기를 통과해야만 만들어진다.
        #
        # 그리고 첫 프레임은 구조적으로 반드시 실패한다: POSE_STALE 은 주행
        # 중에 걸리고, zero 이후 차는 관성으로 미끄러진다. 실측 3/3 —
        #   run_20260904_214600  stale 직후 프레임간 최대 38.6mm / 표류 54.6mm
        #   run_20260904_214835                     18.2mm /       50.1mm
        #   run_20260904_214712                     16.5mm /       24.7mm
        # 셋 다 stationary_tolerance_mm(15) 를 넘는다. 그래서 stage 가
        # ENTRY_STAGING/GLOBAL 에 43~74초 고정된 채 끝났다.
        #
        # 재진입 금지 계약(run_20260903_154026 의 marker 재각인 방지)은 그대로
        # 유지된다 — 계약이 **완료된 뒤**에만 표시가 남기 때문이다. 새 state 도
        # 새 임계값도 없고, POSE_STALE/stationary 기준은 손대지 않는다.
        if not self._critical_heading_ready(view):
            return False
        if not view.is_stationary(self.config.stationary_tolerance_mm,
                                  self.config.stationary_window):
            return False
        if latched_here:
            self._heading_fault_hold.add(car_id)
        self._heading_wait_faulted.discard(car_id)
        self._heading_wait_state.pop(car_id, None)
        self._heading_wait_started.pop(car_id, None)
        # 현재 pose 기준으로 다시 계획하게 한다. 옛 route 는 재개하지 않는다.
        # 이미 슬롯 최종 구간이면 재접근이 아니라 최종 평가로 간다.
        stage = self._post_recovery_stage(car_id, view)
        self._parking_stage[car_id] = stage
        if stage == "FINAL_EVAL_PENDING":
            self._parking_plan_wait[car_id] = view.last_obs_time
        elif stage == "ENTRY_STAGING_PENDING":
            self._entry_staging_wait[car_id] = view.last_obs_time
        else:
            self._parking_setup_wait[car_id] = view.last_obs_time
        self._emit_event("HEADING_RECOVERED", car_id=car_id,
                         heading_source=view.heading_source)
        self.dashboard.push_event("heading_recovered", car_id=car_id,
                                  heading_source=view.heading_source)
        return True

    def _reactivate_after_heading_fault(self, car_id: int) -> None:
        """A validated fresh-pose route is the explicit activation boundary.

        Only reached for cars whose authority was latched by the heading wait.
        A physical fault (boundary, unsafe trajectory, COMM) is never cleared
        here — those cars are not in ``_heading_fault_hold``.
        """
        if car_id not in getattr(self, "_heading_fault_hold", ()):
            return
        runner = self.auto_hosts.get(car_id)
        if runner is None:
            return
        host = getattr(runner, "host", None)
        authority = getattr(host, "authority", None)
        if authority is not None and getattr(authority, "is_faulted", False):
            # producer/guard/reverse observation contract 까지 초기화된다.
            host.re_arm_auto()
        scheduler = getattr(runner, "scheduler", None)
        if scheduler is not None:
            scheduler.start()
        self._heading_fault_hold.discard(car_id)
        self._emit_event("PARKING_REACTIVATED", car_id=car_id,
                         reason="FRESH_HEADING_ROUTE_VALIDATED")

    def _escape_hold(self) -> set[int]:
        """BOUNDARY_HARD 표시 집합 (다른 boundary 상태 집합들과 같은 지연 생성)."""
        held = getattr(self, "_boundary_escape_hold", None)
        if held is None:
            held = self._boundary_escape_hold = set()
        return held

    def _reactivate_after_boundary_escape(self, car_id: int) -> None:
        """검증된 탈출 경로 하나가 BOUNDARY_HARD 정지의 해제 경계다.

        경계 판정을 완화하지 않는다. 임계값(hard 20mm / uncertainty 10mm)도,
        _check_boundary 도, validate_trajectory 의 "시작보다 나빠지지 않는다"
        규칙도 그대로다. 여기까지 왔다는 것은 호출부가 이미
        _trajectory_safe 를 통과한 경로를 적재했다는 뜻이고, 그 경로는
        정의상 현재 overflow 를 넘지 않는다.

        이 표시가 없는 차량(다른 물리 fault, COMM)은 여기서 절대 켜지지
        않는다 — _heading_fault_hold 의 계약과 같은 방식이다.
        """
        if car_id not in self._escape_hold():
            return
        runner = self.auto_hosts.get(car_id)
        if runner is None:
            return
        host = getattr(runner, "host", None)
        authority = getattr(host, "authority", None)
        if authority is not None and getattr(authority, "is_faulted", False):
            host.re_arm_auto()
        scheduler = getattr(runner, "scheduler", None)
        if scheduler is not None:
            scheduler.start()
        self._escape_hold().discard(car_id)
        self._emit_event("BOUNDARY_ESCAPE_REACTIVATED", car_id=car_id,
                         reason="VALIDATED_ESCAPE_ROUTE")
        self.dashboard.push_event("boundary_escape_reactivated", car_id=car_id)

    def _load_slot_reposition(self, car_id: int, view: VehicleView) -> bool:
        """Move back to a pose from which **the same slot** is reachable.

        슬롯을 바꾸지 않는 마지막 복구 수단이다. 목표는 이미 쓰고 있는
        plan_handoff(_parking_reposition_goal), 탐색은 이미 쓰고 있는
        build_setup_recovery_waypoints, 예산은 이미 있는
        max_parking_recovery_attempts. 새 planner/state/threshold 없음.
        """
        runner = self.auto_hosts.get(car_id)
        slot_id = self._auto_host_slot.get(car_id)
        if runner is None or slot_id is None or view.heading_deg is None:
            return False
        attempts = self._parking_recovery_attempts.get(car_id, 0) + 1
        if attempts > int(self.config.max_parking_recovery_attempts):
            return False
        route_id = self.orchestrator.next_route_id()
        wps = build_setup_recovery_waypoints(
            default_slot_specs()[slot_id], route_id=route_id,
            from_pose=view.position_mm, from_heading_deg=view.heading_deg,
            radii_mm=STAGING_RADIUS_CANDIDATES,
            obstacle_poses=self._planner_obstacle_poses(view),
            min_executable_mm=self._setup_min_executable_mm(),
            goal_test=self._parking_reposition_goal(slot_id),
            max_total_length_mm=float(
                self.config.entry_staging_alignment_max_mm),
            min_clearance_mm=float(
                self.config.entry_staging_min_clearance_mm),
            max_segments=4,
            segment_heading_tolerance_deg=float(
                self.config.entry_staging_heading_tolerance_deg))
        if not wps or self._setup_is_degenerate(view, wps):
            return False
        if not self._trajectory_safe(view, wps, slot_id=slot_id,
                                     recovery=True):
            return False
        self._parking_recovery_attempts[car_id] = attempts
        self._parking_stage[car_id] = "SETUP"
        runner.load_route(wps)
        self._reactivate_after_heading_fault(car_id)
        self._emit_event("SLOT_REPOSITION", car_id=car_id, slot=slot_id,
                         route_id=route_id, attempt=attempts,
                         x_mm=round(view.position_mm[0], 1),
                         y_mm=round(view.position_mm[1], 1),
                         heading_deg=round(float(view.heading_deg), 1))
        self._emit_route(wps, car_id=car_id, recovery=True)
        return True

    def _maybe_start_parking_setup(self, view: VehicleView) -> None:
        """Load bounded setup on the first fresh pose after a parking replan."""
        car_id = view.car_id
        if (car_id is None
                or self._parking_stage.get(car_id) != "SETUP_PENDING"):
            return
        wait_after = self._parking_setup_wait.get(car_id, 0.0)
        if not self._phase_boundary_stopped(
                view, wait_after, "PARKING_RECOVERY_REPLAN"):
            return
        if not self._require_critical_heading(
                view, "PARKING_RECOVERY_REPLAN"):
            return
        self._parking_setup_wait.pop(car_id, None)
        if self._load_direct_rear_replan(car_id, view):
            return
        if not self._load_parking_setup(car_id, view):
            # 마지막 수단: **같은 슬롯**을 다시 갈 수 있는 자세로 되돌리는
            # 기동. 슬롯을 바꾸지 않는다.
            #
            # 실측 run_20260904_205954 t=44.4, 자세 (437.7,646.4,26도), B1:
            #   plan_handoff(B1)                 -> feasible=True
            #   build_rear_candidate_waypoints   -> INFEASIBLE (전진 -442mm)
            #   build_setup_recovery_waypoints   -> 5wp, 끝 자세에서 B1 불가
            #   _setup_keeps_target_feasible     -> 정상 차단
            # 가드 판정은 옳았지만 "그럼 B1 로 갈 자세를 다시 만든다" 단계가
            # 없어 그대로 NO_SAFE_PARKING_RECOVERY 로 끝났다. 이미 있는
            # _slot_reposition_* 와 기존 recovery 예산을 그대로 쓴다.
            if self._load_slot_reposition(car_id, view):
                return
            # Direct rear and bounded bidirectional setup were both evaluated
            # from this fresh observation.  Do not fall through to the generic
            # (legacy) recovery/replan path or reuse the same pose to move.
            self._parking_stage[car_id] = "WAIT_SAFE_RECOVERY"
            self.server.stop_control(car_id)
            self._emit_event("FAULT", car_id=car_id,
                             reason="NO_SAFE_PARKING_RECOVERY")
            self.dashboard.push_event(
                "parking_recovery_unavailable", car_id=car_id,
                reason="NO_SAFE_PARKING_RECOVERY")

    def _maybe_start_rear_after_stop(self, view: VehicleView) -> None:
        """Require a distinct, physically stopped observation at phase boundary."""
        car_id = view.car_id
        if car_id is None or self._parking_stage.get(car_id) not in {
                "PARKING_AFTER_SETUP_PENDING", "PARKING_HANDOFF_PENDING"}:
            return
        wait_after = self._parking_plan_wait.get(car_id, 0.0)
        if not self._phase_boundary_stopped(
                view, wait_after, "PARKING_PHASE_BOUNDARY"):
            return
        if not self._require_critical_heading(
                view, "PARKING_PHASE_BOUNDARY"):
            return
        self._parking_plan_wait.pop(car_id, None)
        self._start_rear_parking_stage(car_id)

    def _phase_boundary_stopped(self, view: VehicleView, after_obs_time: float,
                                boundary: str) -> bool:
        """Zero until a new observation window proves physical stationarity."""
        car_id = view.car_id
        if car_id is None or view.last_obs_time <= after_obs_time:
            return False
        key = (car_id, boundary)
        announced = getattr(self, "_physical_stop_wait_announced", None)
        if announced is None:
            announced = set()
            self._physical_stop_wait_announced = announced
        tolerance = float(getattr(self.config, "stationary_tolerance_mm", 15.0))
        window = int(getattr(self.config, "stationary_window", 5))
        if view.is_stationary(tolerance, window):
            announced.discard(key)
            return True
        self.server.stop_control(car_id)
        if key not in announced:
            announced.add(key)
            self._emit_event(
                "WAIT_FOR_PHYSICAL_STOP", car_id=car_id, boundary=boundary,
                observations=len(view.recent))
        return False

    def _load_direct_rear_replan(self, car_id: int, view: VehicleView) -> bool:
        """Prefer the shortest direct rear replan before any setup maneuver."""
        runner = self.auto_hosts.get(car_id)
        slot_id = self._auto_host_slot.get(car_id)
        if (runner is None or slot_id is None
                or not self._require_critical_heading(
                    view, "DIRECT_REAR_REPLAN")):
            return False
        spec = default_slot_specs()[slot_id]
        route_id = self.orchestrator.next_route_id()
        wps = self._replan_rear_entry(spec, view, route_id)
        if wps is None:
            try:
                wps = build_rear_candidate_waypoints(
                    spec, route_id=route_id, from_pose=view.position_mm,
                    from_heading_deg=view.heading_deg, strict=True)
            except InfeasibleRouteError:
                return False
        if not self._trajectory_safe(view, wps, slot_id=slot_id):
            return False
        self._parking_stage[car_id] = "PARKING"
        runner.load_route(wps)
        self._reactivate_after_heading_fault(car_id)
        self._emit_route(wps, car_id=car_id)
        self.dashboard.push_event("parking_direct_replan", car_id=car_id,
                                  slot=slot_id, route_id=route_id)
        return True

    def _route_stop_distance_mm(self) -> float:
        """The controller's own stop distance, in mm, for route validation.

        새 상수가 아니다 — ControllerConfig.stop_distance_cm 는 brake_radius_cm
        가 이미 쓰는 값이고, 여기서는 그것을 그대로 mm 로 넘긴다. 제어기가
        없으면 0.0 (= 검사 비활성) 이라 기존 동작과 같다.
        """
        cfg = getattr(self.config, "controller_config", None)
        if cfg is None:
            return 0.0
        return 10.0 * float(getattr(cfg, "stop_distance_cm", 0.0) or 0.0)

    def _setup_min_executable_mm(self) -> float:
        """Smallest end-point offset a setup maneuver must have to be drivable.

        Derived from the controller's own arrival semantics for the phase the
        setup planner emits (RECOVERY), so the planner and `_setup_is_degenerate`
        agree by construction instead of by a shared magic number.
        """
        cfg = getattr(self.config, "controller_config", None)
        tolerance_cm = PHASE_DEFAULTS["RECOVERY"]["position_tolerance_cm"]
        if cfg is None:
            return 10.0 * float(tolerance_cm)
        return 10.0 * float(cfg.arrival_radius_cm(tolerance_cm, "RECOVERY"))

    def _setup_is_degenerate(self, view: VehicleView,
                             waypoints: list[Any]) -> bool:
        """Whether a setup maneuver can complete without the vehicle moving.

        Arrival is per-waypoint, so a maneuver whose every waypoint already sits
        inside its own arrival radius is captured point by point from a standing
        start: the mission reports DONE, the pipeline re-plans the same maneuver
        from the same pose, and nothing advances.

        Measured in run_20260824_192746 (routes 4-16): a ~100 mm two-leg reverse
        against an 8 cm waypoint tolerance produced 13 routes in 5.6 s with zero
        commanded throttle and zero encoder counts.

        The radius is derived from each waypoint's own tolerance and the existing
        arrival semantics — no separate distance constant is introduced here.
        """
        if not waypoints:
            return False
        targets = []
        for wp in waypoints:
            wx = getattr(wp, "x_mm", None)
            wy = getattr(wp, "y_mm", None)
            if wx is None or wy is None:
                wx, wy = getattr(wp, "x", None), getattr(wp, "y", None)
            if wx is None or wy is None:
                return False        # 좌표를 못 읽으면 판단하지 않는다
            phase = getattr(wp, "phase", None)
            targets.append((
                float(wx), float(wy),
                float(getattr(wp, "position_tolerance_cm", 0.0) or 0.0),
                getattr(phase, "value", phase),
            ))
        cfg = getattr(self.config, "controller_config", None)
        if cfg is None:
            return False
        x, y = view.position_mm
        for wx, wy, tolerance_cm, phase in targets:
            radius_mm = 10.0 * cfg.arrival_radius_cm(tolerance_cm, phase)
            if math.hypot(wx - x, wy - y) > radius_mm:
                return False
        return True

    def _setup_keeps_target_feasible(self, view: VehicleView, slot_id: str,
                                     waypoints: list[Any]) -> bool:
        """Whether a setup maneuver still leaves the target slot handoff-able.

        RUN1/RUN2 에서 반복된 실패 형태다: 차가 이미 통로 위 인계점에 좋은
        자세로 서 있는데(실측 (926.9,624.6,12.1°) / (881.5,620.6,7.1°)),
        setup recovery 가 순수 후진 6-waypoint 로 730mm / 644mm 를 되돌려
        놓았다. 164904 는 그 결과 자세 (399.4,388.9,7.7°) 에서 A3 인계가
        불가능해졌고(합류 완료 x=1085 가 인계점 875 를 지나침), 다음
        recovery 는 NO_SAFE_SETUP_MANEUVER 로 끝났다.

        판정은 새 목적함수가 아니라 **이미 쓰고 있는 plan_handoff** 다. 규칙도
        이 모듈의 기존 관용구와 같다 — "요구치 달성"이 아니라 "지금보다
        나빠지지 않기": 지금 인계가 불가능한 자세라면 이 검사는 아무것도 막지
        않고, 가능한 자세일 때만 그것을 잃는 기동을 거부한다.
        """
        if not waypoints or view.heading_deg is None:
            return True
        spec = default_slot_specs().get(slot_id)
        if spec is None:
            return True
        now = plan_handoff(spec, from_pose=view.position_mm,
                           from_heading_deg=view.heading_deg)
        if not now.feasible:
            return True                  # 잃을 것이 없다
        end = waypoints[-1]
        ex = getattr(end, "x", None)
        ey = getattr(end, "y", None)
        eh = getattr(end, "target_heading_deg", None)
        if ex is None or ey is None or eh is None:
            return True                  # 끝 자세를 못 읽으면 판단하지 않는다
        after = plan_handoff(spec, from_pose=(float(ex), float(ey)),
                             from_heading_deg=float(eh))
        return bool(after.feasible)

    def _load_parking_setup(self, car_id: int, view: VehicleView) -> bool:
        """Generate and load bidirectional setup from one fresh physical pose."""
        runner = self.auto_hosts.get(car_id)
        slot_id = self._auto_host_slot.get(car_id)
        if (runner is None or slot_id is None
                or not self._require_critical_heading(
                    view, "PARKING_SETUP")):
            return False
        obstacles = self._planner_obstacle_poses(view)
        route_id = self.orchestrator.next_route_id()
        setup = build_setup_recovery_waypoints(
            default_slot_specs()[slot_id], route_id=route_id,
            from_pose=view.position_mm, from_heading_deg=view.heading_deg,
            obstacle_poses=obstacles,
            min_executable_mm=self._setup_min_executable_mm())
        if not setup:
            self._emit_event("RECOVERY_REJECTED", car_id=car_id,
                             route_id=route_id, slot=slot_id,
                             reason="NO_SAFE_SETUP_MANEUVER")
            return False
        if self._setup_is_degenerate(view, setup):
            self._emit_event("RECOVERY_REJECTED", car_id=car_id,
                             route_id=route_id, slot=slot_id,
                             reason="INFEASIBLE_DEGENERATE_SETUP")
            return False
        if not self._setup_keeps_target_feasible(view, slot_id, setup):
            self._emit_event("RECOVERY_REJECTED", car_id=car_id,
                             route_id=route_id, slot=slot_id,
                             reason="SETUP_LOSES_TARGET_HANDOFF")
            return False
        if not self._trajectory_safe(view, setup, slot_id=slot_id,
                                     recovery=True):
            return False
        self._parking_stage[car_id] = "SETUP"
        runner.load_route(setup)
        self._reactivate_after_heading_fault(car_id)
        self._emit_route(setup, car_id=car_id, recovery=True)
        self.dashboard.push_event("parking_setup_recovery", car_id=car_id,
                                  slot=slot_id, route_id=route_id,
                                  reason=runner.replan_reason)
        return True

    def _check_auto_host_parked(self, view: VehicleView) -> None:
        """DONE 인 차량이 실제로 멈췄는지 확인하고 PARKED 를 확정한다."""
        runner = self.auto_hosts.get(view.car_id)
        if runner is None or runner.status is not MissionStatus.DONE:
            return
        # rear mode의 SETUP도 독립 mission이라 DONE이 된다. 그 상태를 슬롯
        # FINAL 도착으로 오인하면 WAIT_SAFE_RECOVERY에서도 PARKED가 표시된다.
        # 후면주차는 최종 자세 평가 → PARKED_VERIFY 경로가 확정한다.
        # 여기서 먼저 PARKED 를 찍으면 비스듬해도 통과한다.
        if self.rear_parking_mode:
            return
        if not view.is_stationary(self.config.stationary_tolerance_mm,
                                  self.config.stationary_window):
            return
        if not self._critical_heading_ready(view):
            return
        runner.confirm_parked()
        runner.stop()                      # 제어 스트림 0 으로 고정
        slot_id = self._auto_host_slot.get(view.car_id)
        if slot_id is not None:
            self._parked_obstacles[view.car_id] = (
                view.position_mm[0], view.position_mm[1],
                float(view.heading_deg))
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
        if not self._require_critical_heading(view, "LEGACY_RECOVERY"):
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
            self._emit_event("RECOVERY_REJECTED", car_id=car_id,
                             slot=self._auto_host_slot.get(car_id),
                             reason="NO_SAFE_LEGACY_RECOVERY")
            return False

        slot_id = self._auto_host_slot.get(car_id)
        if not self._trajectory_safe(view, wps, slot_id=slot_id,
                                     recovery=True):
            return False

        try:
            status = runner.load_recovery_waypoints(wps)
        except (RuntimeError, ValueError) as exc:
            log.warning("car %d: 후진 복구 적재 실패 (%s)", car_id, exc)
            return False
        if status is MissionStatus.RECOVERY_FAILED:
            log.warning("car %d: 후진 복구 횟수 초과 — 포기", car_id)
            self.server.stop_control(car_id)
            self._emit_event("FAULT", car_id=car_id,
                             reason="LEGACY_RECOVERY_EXHAUSTED")
            return False

        self._emit_route(wps, recovery=True, car_id=car_id)
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
        if not self._require_critical_heading(view, "GLOBAL_REPLAN"):
            return

        failed = runner.failed_target
        if failed is not None:
            reason = runner.replan_reason or ""
            current = (reason, float(failed.x_mm), float(failed.y_mm),
                       float(view.position_mm[0]), float(view.position_mm[1]))
            previous = self._last_replan_signature.get(car_id)
            self._last_replan_signature[car_id] = current
            if previous is not None:
                same_reason = previous[0] == current[0]
                same_target = math.hypot(previous[1] - current[1],
                                         previous[2] - current[2]) < 10.0
                same_pose = math.hypot(previous[3] - current[3],
                                       previous[4] - current[4]) < max(
                                           10.0,
                                           self.config.initial_pose_stability_mm)
                if same_reason and same_target and same_pose:
                    # 같은 자세에서 같은 경로가 또 나왔다 = 이 planner 로는
                    # 진전이 없다. 하지만 그게 "가만히 있어라" 는 뜻은 아니다.
                    self.server.stop_control(car_id)
                    self._emit_event(
                        "FAULT", car_id=car_id,
                        route_id=getattr(failed, "route_id", None),
                        waypoint_id=getattr(failed, "waypoint_id", None),
                        reason="REPEATED_IDENTICAL_REPLAN")
                    self.dashboard.push_event(
                        "repeated_replan_blocked", car_id=car_id,
                        reason=reason)
                    if self._escalate_repeated_replan(car_id, view, reason):
                        return
                    # 후면주차가 아니거나 escalation 대상이 아니면 여기서
                    # 끝난다. 조용한 zero 가 아니라 **명시적 terminal** 이다.
                    self._parking_stage[car_id] = "WAIT_REPEATED_REPLAN"
                    runner.stop()
                    self._emit_event(
                        "FAULT", car_id=car_id,
                        reason="PLANNER_INFEASIBLE_REPEATED_REPLAN")
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
        # 재계획도 현재 주차 모델을 따라야 한다. 여기서 인계 경로를 박아 두면
        # 후면주차 도중 재계획이 조용히 통로 인계 경로로 바뀐다.
        spec = default_slot_specs()[slot_id]
        # 후면주차 중이면 **현재 자세에서 곧바로 후진 원호에 올라타는** 경로를
        # 먼저 본다. 계획해 둔 REVERSE_START 로 되돌아가는 것이 목적이 아니라
        # 슬롯 FINAL 에 닿는 것이 목적이기 때문이다.
        wps = self._replan_rear_entry(spec, view, route_id)
        if wps is not None:
            if not self._trajectory_safe(view, wps, slot_id=slot_id):
                runner.stop()
                return
            runner.load_route(wps)
            self._emit_route(wps, car_id=car_id)
            log.info("car %d: 현재 자세에서 후진 진입 재계획 → route %d (%d개)",
                     car_id, route_id, len(wps))
            return
        try:
            wps = self._build_route(spec, view, route_id)
        except InfeasibleRouteError as exc:
            log.warning("car %d: 현재 위치에서 슬롯 %s 재계획 불가 (%s) — 정지",
                        car_id, slot_id, exc.reason)
            runner.stop()
            self._emit_event("FAULT", car_id=car_id,
                             reason="REPLAN_EXHAUSTED", attempts=tries - 1)
            return

        if not self._trajectory_safe(view, wps, slot_id=slot_id):
            runner.stop()
            return
        runner.load_route(wps)
        self._emit_route(wps, car_id=car_id)
        log.info("car %d: AUTO_HOST 재계획 → route %d (%d개)",
                 car_id, route_id, len(wps))

    def _emit_route(self, waypoints: list[Any], *, recovery: bool = False,
                     car_id: int | None = None,
                     executing: bool | None = None) -> None:
        """적재한 route 를 기록기·화면에 넘긴다 (route.json / overlay).

        recovery 는 **기록기** 구분이다 (route.json vs recovery_route.json).
        executing 은 **화면** 구분이다 — 지금 러너가 실제로 실행 중인 경로인가.
        기본값은 기존 동작(not recovery)이라 지정하지 않은 호출부는 그대로다.
        """
        if car_id is not None:
            getattr(self, "_deviation_streak", {}).pop(car_id, None)
            getattr(self, "_forward_closest", {}).pop(car_id, None)
            getattr(self, "_reverse_closest", {}).pop(car_id, None)
            getattr(self, "_boundary_motion", {}).pop(car_id, None)
            getattr(self, "_boundary_uncertain_trend", {}).pop(car_id, None)
            # 정체 감시 기준선도 route 지역 상태다. 새 route 는 새 mission 이므로
            # 이전 route 에서 쌓인 정체 시간을 물려받으면 안 된다.
            #
            # 실측 run_20260904_231338 — 이 한 줄이 없어서 모든 recovery 가
            # 실려 있는 채로 즉사했다:
            #     t=34.644 RECOVERY_ROUTE_LOADED count=7 attempt=4
            #     t=34.865 PARKING_STALLED stalled_s=8.31   (0.16초 뒤)
            #     t=46.833 RECOVERY_ROUTE_LOADED count=6 attempt=5
            #     t=47.069 PARKING_STALLED stalled_s=8.84   (0.24초 뒤)
            # 적재 직후 차는 아직 정지·control 0 이라 _check_parking_progress 의
            # active 조건이 그대로 참이고, _stall_since 에는 route 가 생기기도
            # 전의 시각이 남아 있었다.
            #
            # timeout 값(parking_stall_timeout_s)도 판정식도 바꾸지 않는다 —
            # 기준선만 새 route 시점으로 옮긴다. 이 블록의 다른 route-지역
            # 캐시들과 같은 취급이다.
            getattr(self, "_stall_since", {}).pop(car_id, None)
            if executing if executing is not None else not recovery:
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
        if not self._trajectory_safe(view, wps, slot_id=mission.slot_id):
            self._allocation_state[car_id] = "WAIT_SAFE_ROUTE"
            return
        log.info("car %d replanning from %s (%s) → route %d", car_id, node, reason, route_id)
        ctrl = self.controllers.get(car_id)
        if ctrl is not None:
            ctrl.reset()          # 경로가 바뀌면 이전 오차 미분항은 무의미하다
        self.orchestrator.regenerate(car_id, wps)

    def _snapshot_comm_recovery(self, car_id: int, reason: str) -> dict[str, Any]:
        """Preserve intent, never the executable route, across a link outage."""
        existing = self._comm_recovery_context.get(car_id)
        runner = self.auto_hosts.get(car_id)
        track_id = self.track_of_car.get(car_id)
        view = self.views.get(track_id) if track_id is not None else None
        target = None
        if runner is not None:
            target = runner.current_target or runner.failed_target
        if existing is None:
            phase = getattr(runner, "current_phase", None) if runner else None
            phase = getattr(phase, "value", phase)
            existing = {
                "slot_id": self._auto_host_slot.get(car_id),
                "track_id": track_id,
                "prior_stage": self._parking_stage.get(car_id),
                "prior_phase": phase,
                "route_id": getattr(target, "route_id", None),
                "waypoint_id": getattr(target, "waypoint_id", None),
                "generation": 0,
            }
            self._comm_recovery_context[car_id] = existing
        existing["reason"] = reason
        existing["resume_after_obs_time"] = (
            view.last_obs_time if view is not None else 0.0)
        return existing

    def _enter_comm_hold(self, car_id: int, reason: str) -> dict[str, Any]:
        ctx = self._snapshot_comm_recovery(car_id, reason)
        hold = getattr(self.server, "hold_control", None)
        if hold is not None:
            hold(car_id)
        else:
            self.server.stop_control(car_id)
        runner = self.auto_hosts.get(car_id)
        if runner is not None:
            try:
                runner.prepare_route_switch()
            except Exception:                       # noqa: BLE001
                pass
        mux = self.hybrid_controls.get(car_id)
        if mux is not None:
            try:
                mux.hold_for_comm_recovery(reason)
            except Exception:                       # noqa: BLE001
                runner.stop() if runner is not None else None
        elif runner is not None:
            runner.stop()
        self._allocation_state[car_id] = "WAIT_COMM_RECOVERY"
        return ctx

    def _start_comm_recovery_handshake(self, car_id: int) -> None:
        """Re-negotiate REMOTE_DIRECT in background while zero remains latched."""
        ctx = self._comm_recovery_context.get(car_id)
        runner = self.auto_hosts.get(car_id)
        if ctx is None or runner is None or ctx.get("state") != "WAIT_SESSION":
            return
        with self._lock:
            if car_id in self._comm_recovery_starting:
                return
            self._comm_recovery_starting.add(car_id)
        def worker() -> None:
            try:
                # The session manager follows replacement sessions internally.
                # Pipeline owns only the desired state and must not discard a
                # successful new-session negotiation then launch another one.
                runner.ensure_remote_direct(
                    wait_s=self.config.auto_host_handshake_s)
                current = self._comm_recovery_context.get(car_id)
                identity = getattr(
                    self.server, "session_identity", lambda _c: None)(car_id)
                if current is None or current.get("state") != "WAIT_SESSION":
                    return
                if (identity is None
                        or runner.negotiation_identity != identity
                        or current.get("negotiation_session_id") != identity[0]):
                    # A replacement raced the completion edge.  The same owner
                    # converges once more; begin_handshake remains idempotent.
                    runner.ensure_remote_direct(
                        wait_s=self.config.auto_host_handshake_s)
                    identity = getattr(
                        self.server, "session_identity", lambda _c: None)(car_id)
                    current = self._comm_recovery_context.get(car_id)
                if (current is None or current.get("state") != "WAIT_SESSION"
                        or identity is None
                        or runner.negotiation_identity != identity
                        or current.get("negotiation_session_id") != identity[0]):
                    return
                track_id = self.track_of_car.get(car_id)
                view = self.views.get(track_id) if track_id is not None else None
                current["resume_after_obs_time"] = (
                    view.last_obs_time if view is not None else 0.0)
                current["state"] = "WAIT_FRESH_POSE"
                self._emit_event(
                    "COMM_SESSION_READY", car_id=car_id,
                    slot=current.get("slot_id"),
                    generation=int(current.get("generation", 0)),
                    state="WAIT_FRESH_POSE")
            except Exception as exc:                 # noqa: BLE001
                current = self._comm_recovery_context.get(car_id)
                if current is not None and current.get("state") == "WAIT_SESSION":
                    self._comm_recovery_fault(
                        car_id, "REMOTE_DIRECT_RENEGOTIATION_FAILED",
                        detail=str(exc))
            finally:
                with self._lock:
                    self._comm_recovery_starting.discard(car_id)

        threading.Thread(target=worker, name=f"comm-recovery-{car_id}",
                         daemon=True).start()

    def _comm_recovery_fault(self, car_id: int, reason: str, **fields: Any) -> None:
        ctx = self._comm_recovery_context.get(car_id)
        if ctx is not None:
            ctx["state"] = "FAULT"
        self._parking_stage[car_id] = "WAIT_COMM_RECOVERY_FAULT"
        self._allocation_state[car_id] = "WAIT_COMM_RECOVERY_FAULT"
        hold = getattr(self.server, "hold_control", None)
        hold(car_id) if hold is not None else self.server.stop_control(car_id)
        self._emit_event("FAULT", car_id=car_id, reason=reason, **fields)
        self.dashboard.push_event("comm_recovery_fault", car_id=car_id,
                                  reason=reason, **fields)

    def _maybe_resume_comm_recovery(self, view: VehicleView) -> bool:
        """Plan from a post-recovery observation; return True while owning frame."""
        car_id = view.car_id
        if car_id is None:
            return False
        ctx = self._comm_recovery_context.get(car_id)
        if ctx is None:
            return False
        prior_stage = ctx.get("prior_stage")
        if prior_stage in self._PARKING_TERMINAL_STAGES:
            # Defence in depth for a context created by an older callback or a
            # race at the terminal edge. A terminal mission never replans.
            self._parking_stage[car_id] = prior_stage
            self._hold_terminal_comm_state(car_id, connected=True)
            self._emit_event(
                "COMM_RECOVERY_SKIPPED_TERMINAL", car_id=car_id,
                slot=ctx.get("slot_id"), state=prior_stage)
            return True
        self.server.stop_control(car_id)
        if ctx.get("state") != "WAIT_FRESH_POSE":
            return True
        if view.last_obs_time <= float(ctx.get("resume_after_obs_time", 0.0)):
            return True
        if not self._phase_boundary_stopped(
                view, float(ctx.get("resume_after_obs_time", 0.0)),
                "COMM_RECOVERY_REPLAN"):
            return True
        if not self._require_critical_heading(view, "COMM_RECOVERY_REPLAN"):
            return True

        slot_id = ctx.get("slot_id")
        if slot_id is None:
            # There is no stale mission to resume.  Session and physical pose
            # are now valid, but keep the transport latch until the next frame
            # builds and explicitly activates a brand-new validated route.
            self._comm_lost.discard(car_id)
            self._allocation_state[car_id] = "WAIT_NEW_MISSION"
            self._emit_event(
                "COMM_RECOVERY_READY_FOR_NEW_MISSION", car_id=car_id,
                state="CONTROL_HELD")
            self._comm_recovery_context.pop(car_id, None)
            return True
        specs = default_slot_specs()
        if slot_id not in specs:
            self._comm_recovery_fault(car_id, "COMM_CONTEXT_INVALID_SLOT",
                                      slot=slot_id)
            return True
        if (view.slot_id is not None and view.slot_id != slot_id):
            self._comm_recovery_fault(
                car_id, "COMM_CONTEXT_SLOT_MISMATCH", slot=slot_id,
                observed_slot=view.slot_id)
            return True
        slot_index = SLOT_NAMES.index(slot_id)
        tracked = getattr(self.allocator, "vehicles", {}).get(view.track_id)
        owns_reservation = (tracked is not None
                            and tracked.assigned_slot == slot_id)
        if (self._slot_occupancy()[slot_index] >= 0.5
                and not owns_reservation):
            self._comm_recovery_fault(car_id, "COMM_CONTEXT_SLOT_OCCUPIED",
                                      slot=slot_id)
            return True
        runner = self.auto_hosts.get(car_id)
        mux = self.hybrid_controls.get(car_id)
        if runner is None or mux is None:
            self._comm_recovery_fault(car_id, "COMM_CONTEXT_RUNNER_MISSING",
                                      slot=slot_id)
            return True

        ctx["state"] = "PLANNING"
        view.slot_id = slot_id
        prior_stage = ctx.get("prior_stage")
        parking_stages = {
            "PARKING", "SETUP", "SETUP_PENDING",
            "PARKING_AFTER_SETUP_PENDING", "PARKING_HANDOFF_PENDING",
            "WAIT_SAFE_RECOVERY",
        }
        loaded = False
        route_id = None
        if self.rear_parking_mode and prior_stage in parking_stages:
            self._parking_stage[car_id] = "PARKING"
            loaded = self._load_direct_rear_replan(car_id, view)
            if not loaded:
                loaded = self._load_parking_setup(car_id, view)
        else:
            if prior_stage is None:
                self._parking_stage.pop(car_id, None)
            else:
                self._parking_stage[car_id] = prior_stage
            route_id = self.orchestrator.next_route_id()
            try:
                wps = self._build_route(specs[slot_id], view, route_id)
            except InfeasibleRouteError as exc:
                self._emit_event(
                    "ROUTE_REJECTED", car_id=car_id, slot=slot_id,
                    route_id=route_id, reason=exc.reason)
            else:
                if self._trajectory_safe(view, wps, slot_id=slot_id):
                    runner.load_route(wps)
                    self._emit_route(wps, car_id=car_id)
                    loaded = True

        if not loaded:
            self._comm_recovery_fault(
                car_id, "NO_SAFE_ROUTE_AFTER_COMM_RECOVERY", slot=slot_id,
                prior_stage=prior_stage)
            return True

        target = runner.current_target
        route_id = getattr(target, "route_id", route_id)
        try:
            mux.switch_to_auto()                 # AUTO_PENDING, still zero
        except Exception as exc:                 # noqa: BLE001
            self._comm_recovery_fault(
                car_id, "COMM_RECOVERY_REARM_FAILED", detail=str(exc))
            return True
        release = getattr(self.server, "release_control", None)
        if release is not None and not release(car_id):
            self._comm_recovery_fault(car_id, "COMM_CONTROL_RELEASE_REJECTED")
            return True
        self._comm_lost.discard(car_id)
        self._allocation_state[car_id] = "ROUTE_LOADED"
        self._emit_event(
            "COMM_RECOVERY_REPLANNED", car_id=car_id, slot=slot_id,
            route_id=route_id, prior_stage=prior_stage,
            state="AUTO_PENDING")
        self.dashboard.push_event(
            "comm_recovery_replanned", car_id=car_id, slot=slot_id,
            route_id=route_id)
        self._comm_recovery_context.pop(car_id, None)
        return True

    def _hold_terminal_comm_state(self, car_id: int, *, connected: bool) -> bool:
        """Keep a completed parking lifecycle terminal across link churn.

        Transport/session state may recover, but no route, recovery context or
        controller state is allowed to resurrect a PARKED/explicit terminal
        vehicle. Returns False when this is not a parking terminal.
        """
        stage = getattr(self, "_parking_stage", {}).get(car_id)
        if stage not in self._PARKING_TERMINAL_STAGES:
            return False
        hold = getattr(self.server, "hold_control", None)
        hold(car_id) if hold is not None else self.server.stop_control(car_id)
        runner = self.auto_hosts.get(car_id)
        if runner is not None:
            runner.stop()
        with self._lock:
            self._comm_recovery_context.pop(car_id, None)
            if connected:
                self._comm_lost.discard(car_id)
            else:
                self._comm_lost.add(car_id)
            self._mode_set.discard(car_id)
            self._manual_shell_starting.discard(car_id)
            self.controllers.pop(car_id, None)
            self.last_control.pop(car_id, None)
            self._last_control_mode.pop(car_id, None)
            self._auto_host_route.pop(car_id, None)  # executable route is stale
            self._allocation_state[car_id] = stage
        return True

    def _on_comm_fail(self, car_id: int, info: dict[str, Any]) -> None:
        """통신 장애 — 미션을 정지 상태로 두고 복구를 기다린다.

        서버가 장애 시작 엣지에서만 부르므로 여기서 다시 debounce 하지 않는다.
        펌웨어 watchdog에만 기대지 않고 backend stream도 즉시 zero latch한다.
        """
        kind = info.get("type", "COMM_FAIL")
        with self._lock:
            self._comm_lost.add(car_id)
        if not self.auto_host_mode:
            # WAYPOINT_AUTO keeps its existing orchestrator recovery contract;
            # VehicleServer still supplies the new transport-level zero latch.
            held = self.orchestrator.mark_link_lost(car_id)
            self._emit_event("COMM_FAIL", car_id=car_id, reason=kind,
                             session_id=info.get("session_id"),
                             boot_id=info.get("boot_id"),
                             firmware_version=info.get("firmware_version"),
                             rx_gap_ms=info.get("rx_gap_ms"),
                             last_status_seq=info.get("last_status_seq"),
                             last_rx_gap_ms=info.get("last_rx_gap_ms"),
                             max_rx_gap_ms=info.get("max_rx_gap_ms"))
            log.warning("car %d comm fail (%s) — mission %s",
                        car_id, kind, "held" if held else "none active")
            self.dashboard.push_event("comm_fail", car_id=car_id, reason=kind)
            return
        if self._hold_terminal_comm_state(car_id, connected=False):
            self._emit_event(
                "COMM_FAIL", car_id=car_id, reason=kind,
                slot=self._auto_host_slot.get(car_id),
                session_id=info.get("session_id"), boot_id=info.get("boot_id"),
                firmware_version=info.get("firmware_version"),
                rx_gap_ms=info.get("rx_gap_ms"),
                state=f"{self._parking_stage[car_id]}_TERMINAL")
            self.dashboard.push_event(
                "comm_fail", car_id=car_id, reason=kind,
                state=f"{self._parking_stage[car_id]}_TERMINAL")
            return
        ctx = self._enter_comm_hold(car_id, kind)
        ctx["state"] = "WAIT_CONNECTION"
        held = self.orchestrator.mark_link_lost(car_id)
        log.warning("car %d comm fail (%s) — mission %s",
                    car_id, kind, "held" if held else "none active")
        self._emit_event(
            "COMM_FAIL", car_id=car_id, reason=kind,
            slot=ctx.get("slot_id"), route_id=ctx.get("route_id"),
            waypoint_id=ctx.get("waypoint_id"),
            session_id=info.get("session_id"), boot_id=info.get("boot_id"),
            firmware_version=info.get("firmware_version"),
            rx_gap_ms=info.get("rx_gap_ms"),
            last_rx_type=info.get("last_rx_type"),
            last_tx_type=info.get("last_tx_type"),
            last_status_seq=info.get("last_status_seq"),
            last_rx_gap_ms=info.get("last_rx_gap_ms"),
            max_rx_gap_ms=info.get("max_rx_gap_ms"),
            last_rx_ms=info.get("last_rx_ms"),
            last_tx_ms=info.get("last_tx_ms"))
        self.dashboard.push_event("comm_fail", car_id=car_id, reason=kind)

    def _on_comm_recovered(self, car_id: int) -> None:
        """Same-session RX recovery; mode handshake and fresh pose still required."""
        with self._lock:
            if car_id not in self._comm_lost:
                return
        if not self.auto_host_mode:
            with self._lock:
                self._comm_lost.discard(car_id)
            self.dashboard.push_event("comm_recovered", car_id=car_id)
            self._on_replan_required(car_id, "COMM_RECOVERED")
            return
        if self._hold_terminal_comm_state(car_id, connected=True):
            self._emit_event(
                "COMM_RECOVERED", car_id=car_id,
                slot=self._auto_host_slot.get(car_id),
                state=f"{self._parking_stage[car_id]}_TERMINAL")
            self.dashboard.push_event(
                "comm_recovered", car_id=car_id,
                state=f"{self._parking_stage[car_id]}_TERMINAL")
            return
        ctx = self._comm_recovery_context.get(car_id)
        if ctx is None:
            ctx = self._enter_comm_hold(car_id, "COMM_RECOVERED")
        elif ctx.get("state") != "WAIT_CONNECTION":
            # STATUS and COMMAND_RESULT may arrive back-to-back on recovery.
            # Duplicate callbacks must converge on the in-flight negotiation,
            # not advance its generation or launch RESET/SET_MODE again.
            if ctx.get("state") == "WAIT_SESSION":
                self._start_comm_recovery_handshake(car_id)
            return
        ctx["state"] = "WAIT_SESSION"
        ctx["generation"] = int(ctx.get("generation", 0)) + 1
        identity = getattr(self.server, "session_identity", lambda _c: None)(car_id)
        ctx["negotiation_session_id"] = identity[0] if identity else None
        track_id = self.track_of_car.get(car_id)
        view = self.views.get(track_id) if track_id is not None else None
        ctx["resume_after_obs_time"] = view.last_obs_time if view else 0.0
        log.info("car %d comm recovered — zero 유지, 세션 재협상", car_id)
        self._emit_event("COMM_RECOVERED", car_id=car_id,
                         slot=ctx.get("slot_id"), state="WAIT_SESSION")
        self.dashboard.push_event("comm_recovered", car_id=car_id)
        self._start_comm_recovery_handshake(car_id)

    def _on_resync(self, car_id: int, hello: dict[str, Any]) -> None:
        """Reconnect keeps intent but invalidates route/control/pose (§21·26)."""
        log.info("car %d resync (boot_id=%s) — discarding route",
                 car_id, hello.get("boot_id"))
        if (self.auto_host_mode
                and self._hold_terminal_comm_state(car_id, connected=True)):
            self._emit_event(
                "COMM_RESYNC", car_id=car_id,
                slot=self._auto_host_slot.get(car_id),
                boot_id=hello.get("boot_id"),
                state=f"{self._parking_stage[car_id]}_TERMINAL")
            return
        active_auto = (self.auto_host_mode
                       and (car_id in self._comm_recovery_context
                            or self._auto_host_slot.get(car_id) is not None))
        if active_auto:
            identity = getattr(self.server, "session_identity", lambda _c: None)(car_id)
            session_id = identity[0] if identity else None
            existing = self._comm_recovery_context.get(car_id)
            if (existing is not None
                    and existing.get("negotiation_session_id") == session_id
                    and existing.get("state") in {
                        "WAIT_SESSION", "WAIT_FRESH_POSE", "PLANNING",
                    }):
                return
            ctx = self._enter_comm_hold(car_id, "RESYNC")
            ctx["state"] = "WAIT_SESSION"
            ctx["generation"] = int(ctx.get("generation", 0)) + 1
            ctx["negotiation_session_id"] = session_id
            with self._lock:
                self._comm_lost.add(car_id)
                self._mode_set.discard(car_id)
                self._manual_shell_starting.discard(car_id)
                self.controllers.pop(car_id, None)
                self.last_control.pop(car_id, None)
                self._last_control_mode.pop(car_id, None)
                self._auto_host_route.pop(car_id, None)
                track_id = self.track_of_car.get(car_id)
                if track_id is not None and track_id in self.views:
                    self.heading.remove(track_id)
                    self._initial_pose_samples.pop(track_id, None)
                    self.views[track_id].heading_deg = None
                    self.views[track_id].heading_source = None
                    ctx["resume_after_obs_time"] = self.views[track_id].last_obs_time
                self._deviation_streak.pop(car_id, None)
                self._forward_closest.pop(car_id, None)
                self._reverse_closest.pop(car_id, None)
                self._boundary_motion.pop(car_id, None)
                self._boundary_uncertain_trend.pop(car_id, None)
            self._emit_event(
                "COMM_RESYNC", car_id=car_id, slot=ctx.get("slot_id"),
                route_id=ctx.get("route_id"), boot_id=hello.get("boot_id"),
                state="WAIT_SESSION")
            return

        # No active AUTO_HOST intent (initial connection/manual-only): retain
        # the legacy cleanup path.
        self.orchestrator.missions.pop(car_id, None)
        with self._lock:
            self._comm_lost.discard(car_id)      # 새 세션이므로 복구 재계획은 불필요
            self._mode_set.discard(car_id)       # 새 세션에서 모드를 다시 잡는다
            self._manual_shell_starting.discard(car_id)
            self._parking_stage.pop(car_id, None)
            runner = self.auto_hosts.pop(car_id, None)
            mux = self.hybrid_controls.pop(car_id, None)
            self._auto_host_slot.pop(car_id, None)
            self.controllers.pop(car_id, None)
            self.last_control.pop(car_id, None)
            self._last_control_mode.pop(car_id, None)
            track_id = self.track_of_car.pop(car_id, None)
            if track_id is not None and track_id in self.views:
                self.heading.remove(track_id)
                self._initial_pose_samples.pop(track_id, None)
                self.views[track_id].car_id = None
                self.views[track_id].slot_id = None
                self.views[track_id].heading_deg = None
                self.views[track_id].heading_source = None
            self._deviation_streak.pop(car_id, None)
            self._forward_closest.pop(car_id, None)
            self._reverse_closest.pop(car_id, None)
            self._heading_wait_state.pop(car_id, None)
            self._heading_wait_started.pop(car_id, None)
            self._heading_wait_faulted.discard(car_id)
            self._boundary_motion.pop(car_id, None)
            self._boundary_uncertain_trend.pop(car_id, None)
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
