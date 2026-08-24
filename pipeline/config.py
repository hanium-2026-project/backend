"""파이프라인 실행 설정.

실환경(천장 실카메라)과 검증환경(시뮬 영상)의 차이는 여기서만 흡수한다.
캘리브레이션이 끝나면 homography_src 만 실측값으로 교체하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from control import VehicleLimits
from controller.config import ControllerConfig

# 바닥판 실측 (mm) — rl.parking_env / parking.services 와 동일
LOT_WIDTH_MM = 1200.0
LOT_HEIGHT_MM = 1200.0


@dataclass
class PipelineConfig:
    """CV → RL → 통신 파이프라인 구동 파라미터."""

    # ─── 카메라 / 탐지 ───────────────────────────────────────────────────────
    camera_source: int | str = 0
    weights_path: str = "yolo26n.pt"
    confidence_threshold: float = 0.4
    imgsz: int = 1280                     # 천장캠 소형 객체 대응
    custom_model: bool = True
    max_fps: float = 30.0

    # ─── 캘리브레이션 ────────────────────────────────────────────────────────
    # 화면상 바닥판 네 모서리 픽셀 좌표 (좌상, 우상, 우하, 좌하).
    # None 이면 프레임 전체를 바닥판으로 간주한다 (시뮬 영상용).
    homography_src: list[tuple[float, float]] | None = None
    lot_width_mm: float = LOT_WIDTH_MM
    lot_height_mm: float = LOT_HEIGHT_MM
    # bbox 중심과 차량 기하 중심의 고정 오차 보정 (§6.2) — 실측 후 채운다
    bbox_offset_mm: tuple[float, float] = (0.0, 0.0)

    # ─── 통신 ────────────────────────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = 5000
    known_car_ids: frozenset[int] = frozenset({1, 2})

    # ─── 판정 ────────────────────────────────────────────────────────────────
    # 카메라 지연(프레임+추론) × 속도만큼 도착을 앞당겨 판정한다
    camera_lead_cm: float = 2.0
    # 이동으로 인정하는 최소 변위 (heading 추정용, mm)
    heading_min_move_mm: float = 30.0
    # PARKED 재검증의 정지 판정: 최근 창에서 이 값 미만으로 움직이면 정지 (mm)
    stationary_tolerance_mm: float = 15.0
    stationary_window: int = 5

    # ─── RL ──────────────────────────────────────────────────────────────────
    policy_path: str = "models/sb3_parking_policy.zip"
    # 이 노드에 스냅되면 신규 차량으로 보고 슬롯을 배정한다.
    #
    # junction(150,600) = 통로 왼쪽 끝. entrance(150,100) 이 아닌 이유:
    # 최소 선회 반경이 610mm(2026-08-12 실측)인데 바닥 아래쪽에서 통로(y=600)
    # 까지는 600mm 뿐이라, 코너에서 출발하면 90° 우회전을 통로 안에서 끝낼 수
    # 없다(정렬 완료 지점이 y=764mm 로 B행 코앞이다). 그래서 지금은 차를
    # **통로 위에 통로 방향으로 놓고** 시작한다.
    # 진입 우회전을 지원하게 되면 entrance 로 되돌린다 (plan_entry_turn 은
    # 그때를 위해 남겨뒀다).
    entry_nodes: tuple[str, ...] = ("junction", "entrance")
    # RL 이 WAIT 을 반환했을 때 슬롯 배정을 다시 시도하는 간격 (프레임)
    alloc_retry_frames: int = 10
    # 최초 allocation 전에 손으로 차량을 놓는 transient pose를 거른다.
    initial_pose_observations: int = 3
    initial_pose_stability_mm: float = 30.0
    initial_heading_stability_deg: float = 5.0
    # Parking recovery가 fresh pose를 기다리는 동안 detector track_id가
    # 바뀌었을 때만 쓰는 보수적 재결합 조건. 현재 production은 1대 전용이다.
    track_rebind_stale_frames: int = 8
    track_rebind_max_distance_mm: float = 150.0
    # Rebind itself takes about 8 frames (~2 s at the measured 4 FPS).  Allow
    # another ten fresh frames for a physical/trajectory heading, then enter an
    # explicit zero-control fault instead of an unbounded silent wait.
    critical_heading_wait_timeout_s: float = 2.5

    # ─── B안 주행 제어 (DIRECT_CONTROL) ──────────────────────────────────────
    # 노트북이 throttle/steering 을 계산해 내려준다. 실차 안전을 위해 기본은
    # 끔 — 켜도 ESP32 의 ENABLE_ACTUATOR_OUTPUT 이 0 이면 모터는 돌지 않는다.
    # 제어 방식:
    #   "waypoint-auto" — 기존 경로. WAYPOINT/GO 로 ESP32 상태기계를 몰고,
    #                     direct_control 을 켜면 그 위에 제어값을 얹는다.
    #   "auto-host"     — 하드웨어팀 AUTO_HOST 패키지. WAYPOINT/GO 를 보내지 않고
    #                     host 내부 waypoint + DIRECT_CONTROL 만 쓴다 (현재 1대 전용).
    control_mode: str = "waypoint-auto"
    direct_control: bool = False
    # 켜기 전에 SET_MODE REMOTE_DIRECT 를 보낼지 (펌웨어가 모드를 요구할 때)
    direct_control_set_mode: bool = True
    vehicle_limits: VehicleLimits = field(default_factory=VehicleLimits)
    # AUTO_HOST 제어기 설정. vehicle_limits 는 waypoint-auto 전용이라
    # auto-host 에서는 아무 효과가 없다 — 실차 튜닝값은 반드시 이쪽에 넣는다.
    #
    # allow_reverse 를 켜둔다. parking.recovery 가 만드는 후진 복구 waypoint 는
    # 이 값이 False 면 제어기가 REVERSE_NOT_ALLOWED 로 정지시켜, 복구가 조용히
    # 무력화된다. 통로 주행 중 후진은 phase 게이트
    # (ControllerConfig.reverse_allowed_phases)가 계속 막는다.
    controller_config: ControllerConfig = field(
        default_factory=lambda: ControllerConfig(allow_reverse=True))
    # AUTO_HOST 제어 루프 주기 (초). 펌웨어 DIRECT 타임아웃 500ms 대비 5배 여유.
    auto_host_period_s: float = 0.100
    # SET_MODE REMOTE_DIRECT 의 ACCEPTED 를 기다리는 시간 (초)
    auto_host_handshake_s: float = 2.0
    # 같은 슬롯으로 경로를 다시 만드는 최대 횟수. 초과하면 정지한다 —
    # 기하가 같으면 같은 지점에서 계속 실패해 무한 루프가 된다.
    max_replan_attempts: int = 3
    max_parking_recovery_attempts: int = 3
    # 후진 복구를 걸 waypoint phase.
    #
    # 통로 중간(CRUISE)은 허용오차가 넓고 다음 점이 이어지므로, 조금 밀려도
    # 계속 가면 된다 — 거기서 후진을 걸면 진행이 끊긴다(실차 확인).
    # APPROACH 는 인계 25cm 앞 감속점이라 사실상 인계와 한 몸이고, 여기서
    # 지나치면(APPROACH_*_MISSED) 전진으로는 되잡을 수 없다. FINAL 은
    # 정확한 자세가 HW 주차 공식의 전제라 반드시 되잡아야 한다.
    # 빈 튜플이면 후진 복구를 끈다.
    recover_phases: tuple[str, ...] = ("APPROACH", "FINAL")
    # 목표가 전진 사각지대에 들어간 상태로 이만큼 연속 관측되면 경로 이탈로
    # 보고 후진 복구를 건다. 1프레임으로 판단하면 pose 잡음에 걸린다.
    deviation_frames: int = 3
    # 후진 주차(ENTRY/FINAL) 전용 이탈 감시.
    #
    # 기존 이탈 감시는 "목표를 전진으로 잡을 수 있나"라서 후진 목표에는 의미가
    # 없다. 그래서 후진은 통째로 건너뛰고 있었는데, 후면주차 ENTRY 가 발산하면
    # 아무도 멈추지 않아 차가 맵 밖 1m 까지 나간다(closed-loop 확인).
    # 대신 **목표까지 거리가 다시 멀어지는지**를 본다 — 후진이든 전진이든
    # 수렴하지 않으면 경로가 틀린 것이다.
    reverse_divergence_mm: float = 150.0
    # 차체 4모서리가 맵 경계를 이만큼 넘어서면 즉시 정지시킨다.
    # 후진 발산은 몇 초 만에 수백 mm 를 벌리므로 마지막 물리적 방어선이 필요하다.
    # Safety 판정 전용 측정 불확실성 허용치. 맵/planner geometry는 확장하지 않는다.
    # footprint 초과가 이 값 이하이면 pose/calibration/heading 오차로 취급한다.
    boundary_hard_margin_mm: float = 20.0
    # RUNNING에서도 20--30mm는 카메라/기준점 불확실성 band로 분리한다.
    # 단일 fresh frame은 hard fault가 아니며, 연속 증가 또는 predictive
    # guard가 진행을 인정하면 즉시 zero/replan한다. Planner geometry는
    # 이 값으로 확장하지 않는다.
    boundary_measurement_uncertainty_mm: float = 10.0
    boundary_uncertain_confirm_frames: int = 2
    boundary_uncertain_increase_mm: float = 1.0
    # DONE/PARKED는 controller가 이미 zero를 고정한다. 이 terminal-zero
    # 상태의 30mm 초과 단발 spike만 연속 관측로 확정한다.
    boundary_terminal_confirm_frames: int = 2
    # Five 100 ms control ticks, equal to the controller's 0.5 s pose-freshness
    # deadline.  Prediction stops parking before the unchanged hard boundary is
    # crossed; it does not enlarge map geometry or tolerance.
    boundary_prediction_horizon_s: float = 0.5
    # 주차 모델.
    #   "handoff" — 슬롯 앞 통로에 가로로 세우고 끝낸다. 08-12 실차 성공 경로.
    #   "rear"    — 후진으로 슬롯 안까지 넣는다 (build_rear_parking_waypoints).
    # 후면주차가 실차에서 안 되면 돌아갈 길이 필요하므로 기존 모델을 지우지
    # 않고 스위치로 둔다. **rear 는 현재 B1 만 검증됐다.**
    parking_mode: str = "handoff"
    # 경로 계획에 쓸 최소 선회 반경(mm). None 이면 실측값(610mm).
    # 진입 우회전 실험용으로 낮춰 잡을 수 있다 — 차가 실제로 그 반경을 못 돌면
    # 원호 바깥으로 밀리므로, 제어값을 맞춘 뒤에만 의미가 있다.
    plan_turn_radius_mm: float | None = None
    # 수동 계측 모드: 슬롯 배정·자동 주행을 하지 않는다. 차량은 READY 직후
    # 잡히는 MANUAL_WASD 셸에 머물고, 카메라는 계속 pose 를 기록한다.
    # 선회 반경·속도 실측처럼 "사람이 몰고 로그로 재는" 용도.
    manual_only: bool = False

    # ─── 대시보드 ────────────────────────────────────────────────────────────
    # 차량 위치를 대시보드로 보내는 최소 간격 (초). 탐지 주기보다 성기게 둔다.
    dashboard_pose_interval_s: float = 0.2

    def homography_pairs(self, frame_width: int, frame_height: int):
        """(src, dst) 대응점 4쌍을 만든다.

        이미지 y축은 아래로 증가하고 맵 y축은 위로 증가하므로 상하를 뒤집는다.
        """
        if self.homography_src is not None:
            src = list(self.homography_src)
        else:
            src = [(0.0, 0.0), (float(frame_width), 0.0),
                   (float(frame_width), float(frame_height)), (0.0, float(frame_height))]
        dst = [
            (0.0, self.lot_height_mm),                      # 좌상 → 맵 좌상
            (self.lot_width_mm, self.lot_height_mm),        # 우상
            (self.lot_width_mm, 0.0),                       # 우하
            (0.0, 0.0),                                     # 좌하 = 원점
        ]
        return src, dst
