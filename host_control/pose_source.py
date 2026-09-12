"""카메라 관측 pose 소스.

이 모듈은 프롬프트 6장의 **stale fail-safe 무효화 버그**를 구조적으로 막는다.

핵심 규칙
--------
- Pose.timestamp = **실제 카메라 관측이 발생한 시각** (tick 시각이 아니다).
- controller tick 은 별도 `now` 로 실행되며, 새 관측이 없으면 **timestamp 를 갱신하지 않는다.**
- 따라서 관측이 끊기면 now - timestamp 가 커지고 max_pose_age 초과 → stale.

이 모듈은 backend/camera/network 를 import 하지 않는다(stdlib only).
"""

from __future__ import annotations

from typing import Optional

from controller.models import Pose


class CameraPoseSource:
    """최신 카메라 관측 pose 를 관측 시각과 함께 보관한다.

    observe() 가 호출될 때만 timestamp 가 갱신된다. latest() 를 여러 번 호출해도
    (= controller tick 을 여러 번 돌려도) timestamp 는 그대로다.
    """

    def __init__(self) -> None:
        self._pose: Optional[Pose] = None
        self._had_observation = False

    def observe(
        self,
        x_mm: float,
        y_mm: float,
        heading_deg: Optional[float],
        obs_time: float,
        *,
        valid: bool = True,
        heading_source: Optional[str] = None,
    ) -> None:
        """새 카메라 관측을 기록한다.

        obs_time 은 **관측이 발생한 monotonic 시각**이어야 한다(현재 tick 시각이 아님).
        heading_source 는 heading 을 무엇으로 쟀는지다 — 후진 안전 판정에 쓴다.
        """
        self._pose = Pose(
            heading_source=heading_source,
            x_mm=float(x_mm),
            y_mm=float(y_mm),
            heading_deg=None if heading_deg is None else float(heading_deg),
            timestamp=float(obs_time),
            valid=bool(valid),
        )
        self._had_observation = True

    def observe_pose(self, pose: Pose) -> None:
        """이미 만들어진 Pose(관측 timestamp 포함)를 그대로 기록한다."""
        self._pose = pose
        self._had_observation = True

    def latest(self) -> Optional[Pose]:
        """마지막 관측 pose(관측 timestamp 유지). 관측이 없었다면 None."""
        return self._pose

    @property
    def had_observation(self) -> bool:
        return self._had_observation

    def age_s(self, now: float) -> Optional[float]:
        """마지막 관측으로부터 경과 시간(초). 관측이 없으면 None."""
        if self._pose is None:
            return None
        return now - self._pose.timestamp

    def clear(self) -> None:
        """관측 상태 초기화(재시작/재접속 시)."""
        self._pose = None
        self._had_observation = False
