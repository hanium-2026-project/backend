"""ControlScheduler — camera callback 과 독립된 control output loop.

프롬프트 4장/15-D 의 핵심 문제를 해결한다:
  카메라 콜백에서만 controller 를 돌리면, 관측이 끊겼을 때 새 계산이 없어 VehicleServer 가
  마지막 non-zero 값을 계속 재전송한다 → ESP32 500ms timeout 도 안 걸린다.

해결: control loop 를 camera 와 분리해 고정 주기(100ms)로 host.tick() 을 돌린다.
  - 새 관측이 없어도 loop 는 계속 → host 가 stale 판정 → FAULTED + zero 를
    VehicleServer.push_control 로 갱신(latest_control 이 zero 가 됨).

두 가지 구동 방식:
  - step(now): 테스트/수동 구동 1 tick.
  - run(): 실차/CLI 용 백그라운드 스레드 루프(threading, stdlib only).

camera 쪽은 CameraObservationAdapter.on_new_observation() 으로 pose_source 만 갱신한다
(관측 timestamp 보존). 이 스케줄러는 camera 를 호출하지 않는다.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from host_control.host_controller import HostController, TickResult
from host_control.producers import ManualInput

_log = logging.getLogger(__name__)


class ControlScheduler:
    def __init__(
        self,
        host: HostController,
        *,
        period_s: float = 0.100,      # 10Hz. firmware DIRECT timeout 500ms 대비 5x 여유
        clock: Callable[[], float] = time.monotonic,
        on_tick: Optional[Callable[[TickResult], None]] = None,
    ) -> None:
        self.host = host
        self.period_s = period_s
        self._clock = clock
        self._on_tick = on_tick
        self._manual_input: Optional[ManualInput] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- manual 입력
    def set_manual_input(self, manual: Optional[ManualInput]) -> None:
        with self._lock:
            self._manual_input = manual

    # ------------------------------------------------------------- 1 tick
    def step(self, now: Optional[float] = None) -> TickResult:
        """control loop 1회 실행. 새 관측 유무와 무관하게 항상 push 한다."""
        t = self._clock() if now is None else now
        with self._lock:
            manual = self._manual_input
        result = self.host.tick(t, manual_input=manual)
        if self._on_tick is not None:
            # 관측자 예외가 제어 루프를 죽이면 차가 마지막 값으로 계속 달린다.
            try:
                self._on_tick(result)
            except Exception:                       # noqa: BLE001
                _log.exception("on_tick 콜백 실패 (제어 루프는 계속한다)")
        return result

    # ------------------------------------------------------------- 백그라운드 루프
    def run_forever(self) -> None:
        """블로킹 루프. 별도 스레드에서 start()/stop() 로 제어하는 것을 권장."""
        self._stop.clear()
        next_t = self._clock()
        while not self._stop.is_set():
            self.step()
            next_t += self.period_s
            sleep = next_t - self._clock()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                next_t = self._clock()  # 밀렸으면 리셋

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
