"""파이프라인 실행 설정.

실환경(천장 실카메라)과 검증환경(시뮬 영상)의 차이는 여기서만 흡수한다.
캘리브레이션이 끝나면 homography_src 만 실측값으로 교체하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from control import VehicleLimits

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
    # 이 노드에 스냅되면 신규 차량으로 보고 슬롯을 배정한다
    entry_nodes: tuple[str, ...] = ("entrance",)
    # RL 이 WAIT 을 반환했을 때 슬롯 배정을 다시 시도하는 간격 (프레임)
    alloc_retry_frames: int = 10

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
    # AUTO_HOST 제어 루프 주기 (초). 펌웨어 DIRECT 타임아웃 500ms 대비 5배 여유.
    auto_host_period_s: float = 0.100
    # SET_MODE REMOTE_DIRECT 의 ACCEPTED 를 기다리는 시간 (초)
    auto_host_handshake_s: float = 2.0

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
