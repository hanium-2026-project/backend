"""카메라 프레임을 Redis 로 흘려 웹 대시보드가 MJPEG 로 받아가게 한다.

왜 Redis 를 거치나 — 카메라는 한 프로세스만 열 수 있다. run_pipeline 이
장치를 점유하고 있으므로 Django 웹서버가 같은 카메라를 다시 열 수 없다.
이미 채널레이어용으로 Redis 를 쓰고 있으니 그걸 그대로 재사용한다.

설계 원칙 — 스트리밍은 부가 기능이고 제어 루프를 절대 방해하면 안 된다:
  - Redis 가 없거나 죽어도 예외를 올리지 않는다 (한 번만 경고하고 조용히 끈다)
  - 인코딩 비용을 제한한다 (프레임 솎기 + 다운스케일)
  - TTL 을 짧게 준다. 파이프라인이 죽으면 키가 사라져 웹에서 옛 화면이
    계속 살아있는 것처럼 보이지 않는다
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# 이 시간 안에 갱신이 없으면 키가 만료된다 = 파이프라인이 멈춘 것으로 본다.
FRAME_TTL_S = 3


def frame_key(camera_id: int) -> str:
    return f"camera:{camera_id}:frame"


class FramePublisher:
    """주석이 그려진 프레임을 JPEG 로 인코딩해 Redis 에 올린다."""

    def __init__(self, redis_url: str, camera_id: int = 1,
                 fps: float = 8.0, max_width: int = 960,
                 quality: int = 70) -> None:
        self.camera_id = camera_id
        self.min_interval = 1.0 / fps if fps > 0 else 0.0
        self.max_width = max_width
        self.quality = quality
        self._last_at = 0.0
        self._warned = False
        self._client = None
        try:
            import redis
            self._client = redis.Redis.from_url(redis_url, socket_timeout=0.5)
            self._client.ping()
        except Exception as exc:
            log.info("camera stream disabled (%s)", exc)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def publish(self, image, _state=None) -> None:
        """tracker.frame_sink 로 연결된다. 어떤 예외도 밖으로 내보내지 않는다."""
        if self._client is None:
            return
        now = time.monotonic()
        if now - self._last_at < self.min_interval:
            return                                   # 프레임 솎기
        self._last_at = now
        try:
            import cv2

            h, w = image.shape[:2]
            if w > self.max_width:                   # 대역폭·인코딩 비용 절감
                scale = self.max_width / float(w)
                image = cv2.resize(image, (self.max_width, int(h * scale)))
            ok, buf = cv2.imencode(".jpg", image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                return
            self._client.setex(frame_key(self.camera_id), FRAME_TTL_S, buf.tobytes())
        except Exception as exc:
            if not self._warned:                     # 매 프레임 로그를 쏟지 않는다
                log.warning("camera stream publish failed: %s", exc)
                self._warned = True
