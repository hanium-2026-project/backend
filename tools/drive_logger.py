"""실차 주행 로그를 CSV 로 남긴다 (튜닝 근거 확보용).

10Hz 로 아래를 한 줄씩 적는다. 카메라를 안 쓰는 실행(수동 GUI 등)에서는
pose/target 칸이 비고 나머지는 그대로 채워진다.

    시각 / 목표 waypoint / 카메라 pose / 거리·방향 오차
    / 우리가 보낸 throttle·steering / 차량이 적용했다고 보고한 값
    / 서보각·PWM (계산값) / 엔코더 카운트 / 상태

무선으로 오지 않는 값
---------------------
`motor_pwm` 과 `servo_angle_deg` 는 STATUS 에 실리지 않는다. 펌웨어가
512바이트 상한 때문에 뺐고, 주석에 "hardware-mapping details remain in the
ESP32 serial log" 라고 명시돼 있다. 그래서 이 두 값은 **펌웨어의 매핑 함수를
그대로 재현해 계산한 값**이며 컬럼 이름에 `_calc` 를 붙였다. 실측이 필요하면
하드웨어팀에 STATUS 확장을 요청해야 한다.

사용::

    logger = DriveLogger("run.csv", server, car_id=1,
                         pose_provider=lambda: (x_mm, y_mm, heading, source),
                         target_provider=lambda: (tx_mm, ty_mm))
    logger.start()
    ...
    logger.stop()
"""

from __future__ import annotations

import csv
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from controller.config import FirmwareConstants

__all__ = ["DriveLogger", "servo_angle_for", "motor_duty_for"]

_FW = FirmwareConstants()

COLUMNS = [
    "wall_time", "t_s",
    "target_x_mm", "target_y_mm",
    "cam_x_mm", "cam_y_mm", "heading_deg", "heading_source",
    "dist_err_cm", "heading_err_deg",
    "cmd_throttle", "cmd_steering",
    "applied_throttle", "applied_steering",
    "servo_deg_calc", "motor_pwm_calc",
    "encoder_count", "state", "mode", "control_seq",
]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def servo_angle_for(steering: float) -> float:
    """펌웨어 actuator.c::steering_to_angle 재현 (구간 선형)."""
    s = max(-1.0, min(1.0, steering))
    if s <= -0.5:
        return _lerp(_FW.servo_left_strong_deg, _FW.servo_left_weak_deg, (s + 1.0) / 0.5)
    if s < 0.0:
        return _lerp(_FW.servo_left_weak_deg, _FW.servo_center_deg, (s + 0.5) / 0.5)
    if s <= 0.5:
        return _lerp(_FW.servo_center_deg, _FW.servo_right_weak_deg, s / 0.5)
    return _lerp(_FW.servo_right_weak_deg, _FW.servo_right_strong_deg, (s - 0.5) / 0.5)


def motor_duty_for(throttle: float, steering: float) -> int:
    """펌웨어 actuator.c::throttle_to_duty 재현.

    조향이 클수록 duty 하한·기본값이 올라간다. |steering|=1.0 에서는
    min=default=55 라 throttle 이 무시된다.
    """
    mag = abs(max(-1.0, min(1.0, throttle)))
    if mag <= _FW.motor_deadband_throttle:
        return 0
    a = abs(max(-1.0, min(1.0, steering)))
    if a <= 0.5:
        t = a / 0.5
        lo = _lerp(_FW.pwm_forward_min, _FW.pwm_turn_min, t)
        hi = _lerp(_FW.pwm_forward_default, _FW.pwm_turn_default, t)
    else:
        t = (a - 0.5) / 0.5
        lo = _lerp(_FW.pwm_turn_min, _FW.pwm_strong_turn_default, t)
        hi = _lerp(_FW.pwm_turn_default, _FW.pwm_strong_turn_default, t)
    return max(0, min(_FW.motor_pwm_max_duty, round(_lerp(lo, hi, mag))))


class DriveLogger:
    """주행 로그 CSV 기록기 (백그라운드 스레드, 기본 10Hz)."""

    def __init__(self, path: str | Path, server: Any, car_id: int = 1, *,
                 period_s: float = 0.1,
                 pose_provider: Callable[[], tuple | None] | None = None,
                 target_provider: Callable[[], tuple | None] | None = None,
                 only_while_driving: bool = False) -> None:
        self.path = Path(path)
        self.server = server
        self.car_id = car_id
        self.period_s = period_s
        self.pose_provider = pose_provider
        self.target_provider = target_provider
        # True 면 throttle 0 인 구간은 안 적는다 (파일이 작아진다)
        self.only_while_driving = only_while_driving
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self.rows = 0

    # ─── 라이프사이클 ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._fh)
        self._csv.writerow(COLUMNS)
        self._t0 = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="drive-logger",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        fh = getattr(self, "_fh", None)
        if fh is not None and not fh.closed:
            fh.close()

    # ─── 내부 ────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._write_row()
            except Exception:                       # noqa: BLE001
                pass                                # 로깅이 주행을 막으면 안 된다
            self._stop.wait(self.period_s)

    def _write_row(self) -> None:
        status = self.server.last_status(self.car_id) or {}
        sess = self.server.sessions.get(self.car_id)
        ctrl = (sess.latest_control if sess is not None else None) or {}

        cmd_thr = ctrl.get("throttle")
        cmd_str = ctrl.get("steering")
        if self.only_while_driving and not cmd_thr:
            return

        tx = ty = cx = cy = hd = None
        src = ""
        if self.target_provider is not None:
            tgt = self.target_provider()
            if tgt:
                tx, ty = tgt[0], tgt[1]
        if self.pose_provider is not None:
            pose = self.pose_provider()
            if pose:
                cx, cy = pose[0], pose[1]
                hd = pose[2] if len(pose) > 2 else None
                src = pose[3] if len(pose) > 3 else ""

        dist_cm = heading_err = None
        if None not in (tx, ty, cx, cy):
            dist_cm = round(math.hypot(tx - cx, ty - cy) / 10.0, 2)
            if hd is not None:
                bearing = math.degrees(math.atan2(ty - cy, tx - cx))
                heading_err = round((bearing - hd + 180.0) % 360.0 - 180.0, 1)

        # 차량이 보고한 적용값이 있으면 그걸로, 없으면 우리가 보낸 값으로 환산
        a_thr = status.get("applied_throttle")
        a_str = status.get("applied_steering")
        base_thr = a_thr if a_thr is not None else cmd_thr
        base_str = a_str if a_str is not None else cmd_str
        servo = pwm = None
        if base_str is not None:
            servo = round(servo_angle_for(float(base_str)), 1)
        if base_thr is not None and base_str is not None:
            pwm = motor_duty_for(float(base_thr), float(base_str))

        self._csv.writerow([
            datetime.now().strftime("%H:%M:%S.%f")[:-3],
            round(time.monotonic() - self._t0, 3),
            _r(tx), _r(ty), _r(cx), _r(cy), _r(hd, 1), src,
            dist_cm, heading_err,
            cmd_thr, cmd_str,
            a_thr, a_str,
            servo, pwm,
            status.get("encoder_count"),
            status.get("state", ""), status.get("mode", ""),
            status.get("latest_control_seq"),
        ])
        self._fh.flush()
        self.rows += 1


def _r(v, nd: int = 0):
    return None if v is None else (round(v, nd) if nd else round(v))
