"""camera 관측 adapter — 관측 timestamp 보존을 강제한다.

프롬프트 6장의 핵심: **새 관측이 있을 때만** pose 를 갱신하고, tick 시각으로 timestamp 를
덮어쓰지 않는다. 이 adapter 는 그 규칙을 캡슐화한다.

전형적 사용
----------
    cam = CameraObservationAdapter(host.pose_source)
    # CV 파이프라인에서 새 프레임이 나올 때만:
    cam.on_new_observation(x_mm, y_mm, heading_deg, obs_time)
    # 매 tick:
    host.tick(now)   # 새 관측이 없으면 pose_source 는 이전 관측 timestamp 유지 → 결국 stale

이 모듈은 backend/network/cv2 를 import 하지 않는다.
"""

from __future__ import annotations

from typing import Optional

from controller.models import Pose
from host_control.pose_source import CameraPoseSource


class CameraObservationAdapter:
    """CV 파이프라인 → CameraPoseSource 연결. 관측 시각 보존."""

    def __init__(self, pose_source: CameraPoseSource) -> None:
        self._source = pose_source
        self._last_obs_time: Optional[float] = None

    def on_new_observation(
        self,
        x_mm: float,
        y_mm: float,
        heading_deg: Optional[float],
        obs_time: float,
        *,
        valid: bool = True,
    ) -> None:
        """새 카메라 프레임에서 나온 관측만 여기로 전달한다.

        obs_time 은 프레임 캡처(또는 CV 완료) 시각. 동일 프레임을 중복 전달하지 말 것.
        """
        self._source.observe(x_mm, y_mm, heading_deg, obs_time, valid=valid)
        self._last_obs_time = obs_time

    def on_track_lost(self) -> None:
        """트랙 로스트: 관측을 갱신하지 않는다.

        일부러 아무 것도 하지 않는 것이 핵심이다. 이전 관측 timestamp 가 유지되어
        controller 가 곧 stale 로 판정하고 정지/FAULT 한다. (명령 자체는 host tick 이
        zero DIRECT_CONTROL 로 계속 전송한다.)
        """
        return None

    @property
    def last_obs_time(self) -> Optional[float]:
        return self._last_obs_time
