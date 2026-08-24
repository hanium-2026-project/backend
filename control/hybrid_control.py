"""수동 WASD ↔ AUTO_HOST 권한 mux (하드웨어팀 2026-08-10 통합본에서 반입).

우리 AutoHostRunner 에 맞춘 수정 2건:
  1) on_camera_pose 에서 스케줄러를 새로 만들지 않고 기존 것을 재시작한다.
     새로 만들면 러너가 걸어둔 on_tick 콜백이 사라져 미션 상태 변화(도착·재계획)가
     파이프라인까지 안 올라온다.
  2) host.disarm() 은 FAULTED 에서 예외를 던지므로 먼저 fault 를 푼다.

Manual WASD <-> AUTO_HOST authority mux.

Both modes keep the ESP32 in REMOTE_DIRECT.

Safety rule for MANUAL -> AUTO:
- Never restart AUTO using an old camera observation.
- switch_to_auto() enters AUTO_PENDING and keeps output at zero.
- The *next fresh camera pose* is recorded first.
- Only then is the AUTO scheduler started.

Camera poses may continue to be observed while MANUAL owns authority; they never
drive the vehicle because the AUTO scheduler is stopped and HostController is
armed MANUAL.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from host_control.producers import ManualInput
from integration.control_scheduler import ControlScheduler


class HybridControlMux:
    PERIOD_S = 0.10

    def __init__(self, auto_runner) -> None:
        self.runner = auto_runner
        self.host = auto_runner.host
        self.server = auto_runner._server
        self.car_id = auto_runner.car_id

        self._lock = threading.Lock()
        self._manual = ManualInput(0.0, 0.0)
        self._manual_stop = threading.Event()
        self._manual_thread: Optional[threading.Thread] = None
        self.mode = "AUTO_HOST"

    def _safe_disarm(self) -> None:
        """FAULTED 에서는 disarm() 이 AuthorityError 를 던진다. 먼저 풀어준다."""
        if self.host.authority.is_faulted:
            self.host.authority.clear_fault()
        else:
            self.host.disarm()

    def _send_zero_now(self) -> None:
        try:
            self.server.stop_control(self.car_id)
        except Exception:
            pass

    def _stop_manual_loop(self) -> None:
        self._manual_stop.set()
        t = self._manual_thread
        if t is not None and t.is_alive():
            t.join(timeout=0.5)
        self._manual_thread = None

    def _manual_loop(self) -> None:
        while not self._manual_stop.is_set():
            with self._lock:
                manual = self._manual
            self.host.tick(time.monotonic(), manual_input=manual)
            self._manual_stop.wait(self.PERIOD_S)

    def on_camera_pose(
        self,
        x_mm: float,
        y_mm: float,
        heading_deg: float,
        obs_time: float,
        heading_source: str | None = None,
    ) -> None:
        # Always keep the pose source current, including while in MANUAL.
        self.runner.on_camera_pose(x_mm, y_mm, heading_deg, obs_time,
                                   heading_source)

        # AUTO resume is deliberately gated by a genuinely new CV observation.
        if self.mode == "AUTO_PENDING":
            # "새 관측"은 도착 자체가 아니라 **아직 신선한** 관측이어야 한다.
            #
            # obs_time 은 프레임을 찍은 시각인데, 그 프레임을 처리하는 동안
            # 느린 작업(RL 정책 첫 추론 등)이 끼면 여기 도달했을 땐 이미
            # max_pose_age_s 를 넘겨 있다. 그대로 스케줄러를 켜면 첫 tick 이
            # POSE_STALE 을 보고 FAULTED 로 latch 되고, 자동 복구 경로가 없어
            # 차가 슬롯을 배정받자마자 영구 정지한다 (실측: allocate 1.2s).
            #
            # 만료된 관측이면 켜지 않고 AUTO_PENDING 을 유지한다 — 다음
            # 프레임이 오면 그때 출발한다. stale 판정 자체는 그대로 살린다.
            age_s = time.monotonic() - obs_time
            if age_s > self.host.config.max_pose_age_s:
                return
            # 기존 스케줄러를 재시작한다 (새로 만들면 on_tick 콜백이 날아간다)
            self.runner.scheduler.start()
            self.mode = "AUTO_HOST"

    def switch_to_manual(self) -> None:
        if self.mode == "MANUAL_WASD":
            return

        self._send_zero_now()
        self.runner.scheduler.stop()

        self._safe_disarm()

        # MANUAL_WASD gets its own limits; AUTO config remains untouched.
        from dataclasses import replace
        manual_cfg = replace(
            self.host.config,
            max_throttle=1.0,
            allow_reverse=True,
        )
        self.host.manual_producer = type(self.host.manual_producer)(manual_cfg)
        self.host.arm_manual()

        with self._lock:
            self._manual = ManualInput(0.0, 0.0)

        self._manual_stop.clear()
        self._manual_thread = threading.Thread(
            target=self._manual_loop,
            name=f"manual-wasd-{self.car_id}",
            daemon=True,
        )
        self._manual_thread.start()
        self.mode = "MANUAL_WASD"

    def set_manual_wire(self, throttle: float, wire_steering: float) -> None:
        if self.mode != "MANUAL_WASD":
            return

        throttle = max(-1.0, min(1.0, float(throttle)))
        wire_steering = max(-1.0, min(1.0, float(wire_steering)))

        # Old WASD: wire - = LEFT.
        # Current ManualInput: logical + = LEFT.
        logical_steering = -wire_steering

        with self._lock:
            self._manual = ManualInput(throttle, logical_steering)

    def switch_to_auto(self) -> None:
        if self.mode in ("AUTO_HOST", "AUTO_PENDING"):
            return

        if self.mode == "COMM_RECOVERY_HOLD":
            self._send_zero_now()
            self._safe_disarm()
            self.host.arm_auto()
            self.mode = "AUTO_PENDING"
            return

        neutral = ManualInput(0.0, 0.0)
        with self._lock:
            self._manual = neutral

        # Zero while manual still owns authority.
        self.host.tick(time.monotonic(), manual_input=neutral)
        self._stop_manual_loop()
        self._send_zero_now()

        self._safe_disarm()
        self.host.arm_auto()

        # IMPORTANT: do not start scheduler here.
        # Starting before a new camera frame can latch STALE_POSE -> FAULTED.
        self.mode = "AUTO_PENDING"

    def hold_for_comm_recovery(self, reason: str = "COMM_TIMEOUT") -> None:
        """Stop both owners and latch authority until a fresh route is loaded."""
        self._stop_manual_loop()
        self.runner.scheduler.stop()
        self._send_zero_now()
        self.host.fault(reason)
        self.mode = "COMM_RECOVERY_HOLD"

    def stop(self) -> None:
        if self.mode == "MANUAL_WASD":
            neutral = ManualInput(0.0, 0.0)
            with self._lock:
                self._manual = neutral
            try:
                self.host.tick(time.monotonic(), manual_input=neutral)
            except Exception:
                pass

        self._stop_manual_loop()
        try:
            self.runner.scheduler.stop()
        except Exception:
            pass
        self._send_zero_now()
