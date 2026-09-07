"""실차 Run 기록기 — 요청문(2026-08-11) 6~11절 구조.

한 Run 을 디렉터리 하나로 묶는다. 요약만 남기지 않고 원본을 보존한다::

    runs/run_YYYYMMDD_HHMMSS/
        metadata.json        코드 기준점·캘리브레이션·제어 파라미터·시각
        route.json           이 Run 의 waypoint 원문
        recovery_route.json  복구 경로 (발생 시, 배열로 누적)
        pose.jsonl           카메라 프레임마다 1행 (원시 픽셀 → mm 변환 포함)
        control.jsonl        제어 tick 마다 1행 (10Hz)
        events.log           상태 전환만 따로
        summary.json         Run 종료 시 자동 집계

설계 원칙
--------
- **계산값은 `_calc` 로 표시한다.** `motor_pwm` / `servo_angle_deg` 는 STATUS 에
  실려 오지 않아(512바이트 상한) 펌웨어 매핑을 재현한 값이다. 실측처럼 쓰면 안 된다.
- **없는 값은 null 로 남긴다.** 배터리 전압처럼 수집 경로가 없는 항목은 비운다.
- 기록이 주행을 막으면 안 되므로 모든 쓰기는 예외를 삼킨다.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from controller.config import FirmwareConstants

__all__ = ["RunRecorder", "servo_angle_for", "motor_duty_for"]

_FW = FirmwareConstants()
log = logging.getLogger(__name__)


class _RunVideoRecorder:
    """Encode annotated camera frames off the perception/control threads."""

    _SENTINEL = object()

    def __init__(self, run_dir: Path, t0: float, *, fps: float = 4.0,
                 queue_size: int = 8) -> None:
        self.path = run_dir / "e2e.mp4"
        self.frames_path = run_dir / "video_frames.jsonl"
        self.t0 = t0
        self.fps = fps
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._accepting = False
        self.frames_written = 0
        self.frames_dropped = 0
        self.error: str | None = None

    def start(self) -> None:
        self._accepting = True
        self._thread = threading.Thread(
            target=self._run, name="run-video-recorder", daemon=True)
        self._thread.start()

    def submit(self, image, state) -> None:
        if not self._accepting or self.error is not None:
            return
        try:
            import cv2

            elapsed = max(0.0, float(state.timestamp) - self.t0)
            wall = datetime.now().isoformat(timespec="milliseconds")
            # ``image`` is already the tracker's private annotated copy.  Add
            # sync text in place so recording does not allocate another frame.
            cv2.putText(
                image,
                f"t={elapsed:8.3f}s  frame={state.frame_index}  {wall[11:]}",
                (10, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 2, cv2.LINE_AA,
            )
            item = (image, {
                "tracker_frame_index": int(state.frame_index),
                "monotonic_timestamp": float(state.timestamp),
                "elapsed_s": round(elapsed, 6),
                "wall": wall,
            })
            self._queue.put_nowait(item)
        except queue.Full:
            # Observability may drop a video frame; it must never delay control.
            self.frames_dropped += 1
        except Exception as exc:  # noqa: BLE001
            self.error = f"video frame enqueue failed: {exc}"
            log.error(self.error)

    def _run(self) -> None:
        writer = None
        frames_file = None
        try:
            import cv2

            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                image, timing = item
                if writer is None:
                    height, width = image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(self.path), fourcc, self.fps, (width, height))
                    if not writer.isOpened():
                        self.error = "OpenCV mp4v VideoWriter could not be opened"
                        log.error("Video recording disabled: %s", self.error)
                        writer.release()
                        writer = None
                        break
                    frames_file = self.frames_path.open(
                        "w", buffering=8192, encoding="utf-8")
                writer.write(image)
                timing["video_frame_index"] = self.frames_written
                frames_file.write(json.dumps(timing, ensure_ascii=False) + "\n")
                self.frames_written += 1
        except Exception as exc:  # noqa: BLE001
            self.error = f"video writer failed: {exc}"
            log.exception("Video recording failed")
        finally:
            if writer is not None:
                writer.release()
            if frames_file is not None:
                frames_file.close()

    def stop(self) -> dict[str, Any]:
        self._accepting = False
        if self._thread is not None:
            try:
                self._queue.put(self._SENTINEL, timeout=2.0)
            except queue.Full:
                self.error = self.error or "video writer queue did not drain"
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                self.error = self.error or "video writer did not stop cleanly"
            self._thread = None
        playable = bool(
            self.error is None and self.frames_written > 0
            and self.path.exists() and self.path.stat().st_size > 0)
        return {
            "path": str(self.path) if playable else None,
            "frames_path": str(self.frames_path) if playable else None,
            "fps": self.fps,
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
            "error": self.error,
        }


def _execution_gate_reason(sess: Any, authority: str | None,
                           fault_reason: str | None,
                           mission_status: str | None,
                           throttle_cmd: float | None,
                           applied_throttle: float | None,
                           controller_reason: str | None) -> str:
    """Return the first owner that explains why execution is or is not moving."""
    if sess is None:
        return "NO_SESSION"
    if not getattr(sess, "alive", False):
        return "SESSION_CLOSED"
    if getattr(sess, "comm_failed", False):
        return "COMM_FAILED"
    if getattr(sess, "control_held", False):
        return "COMM_ZERO_LATCH"
    if authority == "FAULTED":
        return f"HOST_FAULT:{fault_reason}"
    if mission_status != "RUNNING":
        return f"MISSION_{mission_status or 'NONE'}"
    if throttle_cmd is not None and abs(float(throttle_cmd)) < 1e-9:
        return controller_reason or "CONTROLLER_ZERO"
    if (applied_throttle is not None
            and abs(float(applied_throttle)) < 1e-9
            and throttle_cmd is not None
            and abs(float(throttle_cmd)) >= 1e-9):
        return "ESP_APPLIED_ZERO"
    return "EXECUTING"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def servo_angle_for(steering: float) -> float:
    """펌웨어 actuator.c::steering_to_angle 재현."""
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

    HW 7fc17c6 부터 강회전 최소 duty(PWM_STRONG_TURN_MIN=38)가 기본값(50)과
    분리돼, 최대 조향에서도 throttle 이 duty 를 움직인다.
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
        strong_min = getattr(_FW, "pwm_strong_turn_min", _FW.pwm_turn_min)
        lo = _lerp(_FW.pwm_turn_min, strong_min, t)
        hi = _lerp(_FW.pwm_turn_default, _FW.pwm_strong_turn_default, t)
    return max(0, min(_FW.motor_pwm_max_duty, round(_lerp(lo, hi, mag))))


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent.parent,
                              timeout=5).stdout.strip() or None
    except Exception:                                    # noqa: BLE001
        return None


def _wp_dict(wp: Any) -> dict[str, Any]:
    """Waypoint(백엔드/코어 어느 쪽이든)를 JSON 으로."""
    def g(*names, default=None):
        for n in names:
            if hasattr(wp, n):
                v = getattr(wp, n)
                return v.value if hasattr(v, "value") else v
        return default
    return {
        "route_id": g("route_id"), "waypoint_id": g("waypoint_id"),
        "phase": g("phase"),
        "x_mm": g("x_mm", "x"), "y_mm": g("y_mm", "y"),
        "target_heading_deg": g("target_heading_deg"),
        "speed_cm_s": g("speed_cm_s"),
        "position_tolerance_cm": g("position_tolerance_cm"),
        "capture_tolerance_cm": g("capture_tolerance_cm"),
        "heading_tolerance_deg": g("heading_tolerance_deg"),
        "heading_required": g("heading_required"),
        "motion_direction": g("motion_direction", default="FORWARD"),
        "curvature": g("curvature", default=0.0),
        "path_capture_tolerance_cm": g("path_capture_tolerance_cm"),
        "is_final": g("is_final"),
    }


class RunRecorder:
    """Run 하나를 디렉터리로 기록한다.

    Parameters
    ----------
    server : VehicleServer
    car_id : int
    pose_provider : () -> dict | None
        프레임 정보. `frame_id/capture_ts/pose_ts/obs_time/track_id/pixel_x/
        pixel_y/x_mm/y_mm/heading_deg/heading_source/valid/confidence/
        latency_ms/fps/dropped` 중 있는 것만 담아 반환.
    runner_provider : () -> AutoHostRunner | None
        미션·approach·recovery 상태를 읽기 위해 필요. 없으면 해당 칸이 빈다.
    """

    def __init__(self, base_dir: str | Path, server: Any, car_id: int = 1, *,
                 pose_provider: Callable[[], dict | None] | None = None,
                 runner_provider: Callable[[], Any] | None = None,
                 lifecycle_provider: Callable[[], dict | None] | None = None,
                 control_period_s: float = 0.1,
                 params: dict[str, Any] | None = None,
                 calibration: dict[str, Any] | None = None) -> None:
        self.server = server
        self.car_id = car_id
        self.pose_provider = pose_provider
        self.runner_provider = runner_provider
        self.lifecycle_provider = lifecycle_provider
        self.control_period_s = control_period_s

        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir = Path(base_dir) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._video: _RunVideoRecorder | None = None

        self._pose_f = (self.dir / "pose.jsonl").open("w", buffering=1, encoding="utf-8")
        self._ctrl_f = (self.dir / "control.jsonl").open("w", buffering=1, encoding="utf-8")
        self._evt_f = (self.dir / "events.log").open("w", buffering=1, encoding="utf-8")

        # 집계
        self.pose_rows = self.ctrl_rows = 0
        self._prev_enc: int | None = None
        self._prev_status: str | None = None
        self._prev_phase: str | None = None
        self._prev_stage: str | None = None
        self._prev_state: str | None = None
        self._prev_authority: str | None = None
        self._prev_confirm = 0
        self._control_sample_index = 0
        self._prev_control_sample_at: float | None = None
        self._sat_ticks = 0
        self._thr: list[float] = []
        self._pose_age: list[float] = []
        self._enc_first: int | None = None
        self._enc_last: int | None = None
        self._recoveries = 0
        self._comm_events = 0
        self._min_approach: float | None = None
        self._max_after_min: float | None = None
        self._phase_time: dict[str, float] = {}
        self._recovery_routes: list[list[dict]] = []

        self._write_metadata(params or {}, calibration)
        self.event("RUN_START", note=stamp)

    # ─── 메타 / 경로 ─────────────────────────────────────────────────────────

    def _write_metadata(self, params: dict, calibration: dict | None) -> None:
        meta = {
            "run_dir": self.dir.name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "backend": {
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain")),
            },
            # 하드웨어팀 기준 커밋. 자동으로 알 수 없어 상수로 박는다.
            "hw_autohost_commit": "7fc17c6",
            "hw_autohost_branch": "feat/auto-parking-recovery",
            "firmware_constants": {
                "servo_deg": [_FW.servo_left_strong_deg, _FW.servo_left_weak_deg,
                              _FW.servo_center_deg, _FW.servo_right_weak_deg,
                              _FW.servo_right_strong_deg],
                "pwm_forward": [_FW.pwm_forward_min, _FW.pwm_forward_default],
                "pwm_turn": [_FW.pwm_turn_min, _FW.pwm_turn_default],
                "pwm_strong_turn": [getattr(_FW, "pwm_strong_turn_min", None),
                                    _FW.pwm_strong_turn_default],
                "direct_control_timeout_ms": _FW.direct_control_timeout_ms,
            },
            "controller_params": params,
            "calibration": calibration,
            "coordinate_convention": {
                "origin": "바닥판 좌하단",
                "x": "우측 +", "y": "위쪽 +", "unit": "mm",
                "heading_zero": "우측 0도", "heading_increase": "반시계(위쪽 90도)",
                "wire_steering": "음수 = 좌회전 (실물 확인)",
            },
            "notes": [
                "servo_angle_deg_calc / motor_pwm_calc 는 STATUS 에 없어 펌웨어 매핑을 재현한 계산값이다.",
                "battery_voltage 는 수집 경로가 없어 기록하지 않는다 (STATUS 확장 필요).",
            ],
        }
        (self.dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_route(self, waypoints, *, recovery: bool = False) -> None:
        rows = [_wp_dict(w) for w in waypoints]
        if recovery:
            self._recoveries += 1
            self._recovery_routes.append(rows)
            (self.dir / "recovery_route.json").write_text(
                json.dumps(self._recovery_routes, ensure_ascii=False, indent=2),
                encoding="utf-8")
            self.event("RECOVERY_ROUTE_LOADED", count=len(rows),
                       attempt=self._recoveries)
        else:
            (self.dir / "route.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            self.event("ROUTE_LOADED", count=len(rows))

    # ─── 이벤트 ──────────────────────────────────────────────────────────────

    def event(self, name: str, **fields: Any) -> None:
        try:
            if name == "COMM_FAIL":
                with self._lock:
                    self._comm_events += 1
            t = time.monotonic() - self.t0
            extra = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
            self._evt_f.write(
                f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
                f"t={t:8.3f} {name}{(' ' + extra) if extra else ''}\n")
        except Exception:                                # noqa: BLE001
            pass

    # ─── 프레임 (카메라) ─────────────────────────────────────────────────────

    def start_video(self, *, fps: float = 4.0) -> Path:
        """Enable asynchronous annotated-frame recording in this run dir."""
        if self._video is None:
            self._video = _RunVideoRecorder(self.dir, self.t0, fps=fps)
            self._video.start()
        return self._video.path

    def log_video_frame(self, image, state) -> None:
        """Non-blocking frame observer used by ``RCCarTracker``."""
        if self._video is not None:
            self._video.submit(image, state)

    def log_pose(self, rec: dict) -> None:
        """카메라 프레임 1건. 호출자가 아는 필드만 채워 넘긴다."""
        try:
            row = {"t_s": round(time.monotonic() - self.t0, 4),
                   "wall": datetime.now().isoformat(timespec="milliseconds")}
            row.update(rec)
            self._pose_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.pose_rows += 1
        except Exception:                                # noqa: BLE001
            pass

    # ─── 제어 tick ───────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="run-recorder",
                                            daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:                            # noqa: BLE001
                pass
            self._stop.wait(self.control_period_s)

    def _tick(self) -> None:
        now = time.monotonic()
        sample_dt_s = (None if self._prev_control_sample_at is None
                       else now - self._prev_control_sample_at)
        self._prev_control_sample_at = now
        self._control_sample_index += 1
        status = self.server.last_status(self.car_id) or {}
        sess = self.server.sessions.get(self.car_id)
        ctrl = (sess.latest_control if sess is not None else None) or {}
        runner = self.runner_provider() if self.runner_provider else None
        pose = self.pose_provider() if self.pose_provider else None
        lifecycle = self.lifecycle_provider() if self.lifecycle_provider else {}
        lifecycle = lifecycle or {}

        target = getattr(runner, "current_target", None) if runner else None
        mission = getattr(runner, "mission", None) if runner else None
        guard = getattr(getattr(runner, "host", None), "approach_guard", None)
        fguard = getattr(getattr(runner, "host", None), "final_pose_guard", None)
        authority_obj = getattr(getattr(runner, "host", None), "authority", None)
        reverse_observation_state = getattr(
            getattr(runner, "host", None), "reverse_observation_state", None)
        authority = getattr(authority_obj, "state", None)
        authority = authority.value if hasattr(authority, "value") else authority

        px = py = ph = None
        pose_age_ms = None
        if pose:
            px, py, ph = pose.get("x_mm"), pose.get("y_mm"), pose.get("heading_deg")
            if pose.get("obs_time") is not None:
                pose_age_ms = round((now - pose["obs_time"]) * 1000, 1)

        tx = ty = th = None
        dist_cm = endpoint_head_err = None
        if target is not None:
            tx, ty = getattr(target, "x_mm", None), getattr(target, "y_mm", None)
            th = getattr(target, "target_heading_deg", None)
            if None not in (px, py, tx, ty):
                dist_cm = round(math.hypot(tx - px, ty - py) / 10.0, 2)
                if ph is not None:
                    bearing = math.degrees(math.atan2(ty - py, tx - px))
                    endpoint_head_err = round(
                        (bearing - ph + 180.0) % 360.0 - 180.0, 1)

        tick = getattr(runner, "last_tick_result", None) if runner else None
        command = getattr(tick, "command", None)
        controller_telemetry = dict(
            getattr(command, "telemetry", None) or {})
        head_err = (getattr(command, "heading_error_deg", None)
                    if command is not None else endpoint_head_err)
        logical = (getattr(command, "logical_steering", None)
                   if command is not None else None)
        curvature = float(getattr(target, "curvature", 0.0) or 0.0)
        raw_direction = getattr(target, "motion_direction", "")
        direction = getattr(raw_direction, "value", raw_direction)
        reverse = str(direction).upper() == "REVERSE"
        feedforward = None
        if runner is not None and target is not None:
            feedforward = runner.config.feedforward_steering(
                getattr(target, "phase", None), curvature, reverse=reverse)
        feedback = (None if logical is None or feedforward is None
                    else round(float(logical) - float(feedforward), 4))

        thr = ctrl.get("throttle")
        wire_str = ctrl.get("steering")
        desired_thr = getattr(command, "throttle", None)
        desired_str = getattr(command, "steering", None)
        a_thr, a_str = status.get("applied_throttle"), status.get("applied_steering")
        base_t = a_thr if a_thr is not None else thr
        base_s = a_str if a_str is not None else wire_str

        enc = status.get("encoder_count")
        enc_delta = None
        if enc is not None:
            enc = int(enc)
            if self._prev_enc is not None:
                enc_delta = enc - self._prev_enc
            self._prev_enc = enc
            self._enc_first = enc if self._enc_first is None else self._enc_first
            self._enc_last = enc

        stage = getattr(guard, "stage", None)
        stage = stage.value if hasattr(stage, "value") else stage
        mstatus = getattr(mission, "status", None)
        mstatus = mstatus.value if hasattr(mstatus, "value") else mstatus
        phase = getattr(mission, "current_phase", None)
        confirm = getattr(fguard, "count", None)
        controller_reason = getattr(command, "reason", None)
        execution_gate = _execution_gate_reason(
            sess, authority, getattr(authority_obj, "fault_reason", None),
            mstatus, thr, a_thr, controller_reason)
        override_reason = controller_reason or execution_gate

        row = {
            "t_s": round(now - self.t0, 3),
            "wall": datetime.now().isoformat(timespec="milliseconds"),
            "car_id": self.car_id,
            "control_sample_index": self._control_sample_index,
            "control_sample_dt_s": sample_dt_s,
            "pose_x_mm": px, "pose_y_mm": py, "pose_heading_deg": ph,
            "pose_heading_source": (pose.get("heading_source") if pose else None),
            "pose_timestamp": (pose.get("obs_time") if pose else None),
            "pose_age_ms": pose_age_ms,
            "pose_fresh": controller_telemetry.get("pose_fresh"),
            "pose_stale": (None if controller_telemetry.get("pose_fresh") is None
                           else not controller_telemetry["pose_fresh"]),
            "route_id": (getattr(target, "route_id", None)
                         or lifecycle.get("owned_route_id")),
            "waypoint_id": getattr(target, "waypoint_id", None),
            "waypoint_index": getattr(mission, "index", None),
            "waypoint_total_count": getattr(mission, "total", None),
            "phase": phase,
            "motion_direction": (lambda v: v.value if hasattr(v, "value") else v)(
                getattr(target, "motion_direction", None)),
            "target_x_mm": tx, "target_y_mm": ty, "target_heading_deg": th,
            "heading_required": getattr(target, "heading_required", None),
            "target_curvature": getattr(target, "curvature", None),
            # planner 가 **요구한** 속도. 이게 없으면 "요구 40mm/s 에 실제
            # 얼마가 나왔는가" 를 로그만으로 복원할 수 없다 — throttle 명령은
            # 있지만 그건 controller 출력이지 planner 의도가 아니다.
            # 단위는 mm/s 로 통일한다 (waypoint 는 cm/s 로 들고 있다).
            "target_speed_mm_s": (
                None if getattr(target, "speed_cm_s", None) is None
                else 10.0 * float(target.speed_cm_s)),
            "curvature": getattr(target, "curvature", None),
            "path_capture_tolerance_cm": getattr(
                target, "path_capture_tolerance_cm", None),
            "distance_error_cm": dist_cm, "heading_error_deg": head_err,
            "distance_to_target_mm": (
                None if dist_cm is None else dist_cm * 10.0),
            "dx_to_target_mm": (None if None in (px, tx) else tx - px),
            "dy_to_target_mm": (None if None in (py, ty) else ty - py),
            "desired_bearing_deg": getattr(command, "target_bearing_deg", None),
            "endpoint_heading_error_deg": endpoint_head_err,
            "curvature_feedforward": feedforward,
            "steering_feedback": feedback,
            "approach_stage": stage,
            "approach_best_distance_cm": getattr(guard, "best_distance_cm", None),
            "capture_tolerance_cm": getattr(target, "capture_tolerance_cm", None),
            "position_tolerance_cm": getattr(target, "position_tolerance_cm", None),
            "heading_tolerance_deg": getattr(target, "heading_tolerance_deg", None),
            "mission_status": mstatus,
            "route_mission_status": mstatus,
            "workflow_status": lifecycle.get("workflow_status"),
            "parking_stage": lifecycle.get("parking_stage"),
            "allocation_state": lifecycle.get("allocation_state"),
            "slot_id": lifecycle.get("slot_id"),
            "slot_center_x_mm": lifecycle.get("slot_center_x_mm"),
            "slot_center_y_mm": lifecycle.get("slot_center_y_mm"),
            "parked_heading_deg": lifecycle.get("parked_heading_deg"),
            "comm_recovery_state": lifecycle.get("comm_recovery_state"),
            "authority": authority,
            "fault_reason": getattr(authority_obj, "fault_reason", None),
            "controller_reason": controller_reason,
            "command_reason": override_reason,
            "waypoint_arrived": getattr(command, "arrived", None),
            "execution_gate": execution_gate,
            "control_override_reason": override_reason,
            "direction_change_interlock_active": (
                controller_reason == "DIRECTION_CHANGE_STOP"),
            "stale_zero_override": controller_reason in {
                "POSE_STALE", "POSE_INVALID", "NO_POSE", "NO_HEADING",
                "POSE_BLIND_TRAVEL"},
            "comm_zero_override": execution_gate in {
                "COMM_FAILED", "COMM_ZERO_LATCH", "SESSION_CLOSED",
                "NO_SESSION"},
            "boundary_zero_override": bool(
                override_reason and "BOUNDARY" in override_reason),
            "safety_zero_override": bool(
                override_reason and override_reason != "EXECUTING"),
            "direct_zero_latch": (None if sess is None else bool(
                getattr(sess, "control_held", False))),
            "comm_failed": (None if sess is None else bool(
                getattr(sess, "comm_failed", False))),
            "firmware_version": (None if sess is None else getattr(
                sess, "firmware_version", None)),
            "transport_last_rx_gap_ms": (None if sess is None else round(
                float(getattr(sess, "last_rx_gap_ms", 0.0)), 1)),
            "transport_max_rx_gap_ms": (None if sess is None else round(
                float(getattr(sess, "max_rx_gap_ms", 0.0)), 1)),
            "reverse_observation_state": reverse_observation_state,
            "replan_reason": getattr(mission, "replan_reason", None),
            "recovery_attempt": getattr(mission, "recovery_attempts", None),
            "final_confirm_count": confirm,
            "desired_throttle": desired_thr,
            "desired_steering": desired_str,
            "throttle_command_final": thr,
            "steering_command_final": wire_str,
            "throttle_cmd": thr, "steering_cmd": wire_str,
            "wire_steering": wire_str,
            "logical_steering": logical if logical is not None else (None if wire_str is None else
                                 round(-float(wire_str), 4)),   # wire 음수=좌 → 논리 양수=좌
            "applied_throttle": a_thr, "applied_steering": a_str,
            "servo_deg_calc": (None if base_s is None
                               else round(servo_angle_for(float(base_s)), 1)),
            "motor_pwm_calc": (None if None in (base_t, base_s)
                               else motor_duty_for(float(base_t), float(base_s))),
            "control_seq": status.get("latest_control_seq"),
            "encoder_count": enc, "encoder_delta": enc_delta,
            "esp_state": status.get("state"), "esp_mode": status.get("mode"),
            "wait_reason": status.get("wait_reason"),
            "boot_id": status.get("boot_id"), "session_id": status.get("session_id"),
            "status_seq": status.get("status_seq"),
        }
        # Diagnostics are attached to the exact immutable ControlCommand that
        # produced this sample.  Legacy recorder fields above remain the source
        # of truth when a name overlaps.
        for key, value in controller_telemetry.items():
            row.setdefault(key, value)
        self._ctrl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.ctrl_rows += 1
        self._collect(row, thr, wire_str, dist_cm, pose_age_ms, phase,
                      mstatus, stage, confirm, status)

    # ─── 집계 + 상태 전환 이벤트 ─────────────────────────────────────────────

    def _collect(self, row, thr, wire_str, dist_cm, pose_age_ms, phase,
                 mstatus, stage, confirm, status) -> None:
        if thr is not None:
            self._thr.append(float(thr))
        if wire_str is not None and abs(float(wire_str)) >= 0.999:
            self._sat_ticks += 1
        if pose_age_ms is not None:
            self._pose_age.append(pose_age_ms)
        if dist_cm is not None:
            if self._min_approach is None or dist_cm < self._min_approach:
                self._min_approach = dist_cm
                self._max_after_min = dist_cm
            elif self._max_after_min is not None and dist_cm > self._max_after_min:
                self._max_after_min = dist_cm
        if phase:
            self._phase_time[phase] = self._phase_time.get(phase, 0.0) + self.control_period_s

        if phase != self._prev_phase:
            self.event("PHASE", frm=self._prev_phase, to=phase)
            self._prev_phase = phase
        if stage != self._prev_stage:
            self.event("APPROACH_STAGE", frm=self._prev_stage, to=stage)
            self._prev_stage = stage
        if mstatus != self._prev_status:
            self.event("MISSION", frm=self._prev_status, to=mstatus,
                       reason=row.get("replan_reason"))
            if mstatus == "REPLAN_REQUIRED":
                self.event("REPLAN_REQUIRED", reason=row.get("replan_reason"))
            self._prev_status = mstatus
        authority = row.get("authority")
        if authority != self._prev_authority:
            if authority == "FAULTED":
                self.event("FAULT", reason=row.get("fault_reason"))
            self._prev_authority = authority
        st = status.get("state")
        if st != self._prev_state:
            self.event("ESP_STATE", frm=self._prev_state, to=st,
                       wait_reason=status.get("wait_reason"))
            self._prev_state = st
        if confirm is not None and confirm != self._prev_confirm:
            if confirm:
                self.event("FINAL_CONFIRMING", count=confirm)
            self._prev_confirm = confirm

    # ─── 종료 ────────────────────────────────────────────────────────────────

    def stop(self, *, outcome: str = "UNKNOWN", note: str = "") -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        video = self._video.stop() if self._video is not None else None
        self.event("RUN_END", outcome=outcome, note=note or None)

        dur = time.monotonic() - self.t0
        overshoot = None
        if self._min_approach is not None and self._max_after_min is not None:
            overshoot = round(self._max_after_min - self._min_approach, 2)
        summary = {
            "outcome": outcome,
            "note": note,
            "duration_s": round(dur, 2),
            "phase_time_s": {k: round(v, 2) for k, v in self._phase_time.items()},
            "approach_min_distance_cm": self._min_approach,
            "approach_overshoot_cm": overshoot,
            "recovery_count": self._recoveries,
            "pose_rows": self.pose_rows,
            "control_rows": self.ctrl_rows,
            "valid_pose_fps": (round(self.pose_rows / dur, 2) if dur > 0 else None),
            "pose_age_ms": {
                "max": (max(self._pose_age) if self._pose_age else None),
                "mean": (round(sum(self._pose_age) / len(self._pose_age), 1)
                         if self._pose_age else None),
            },
            "throttle": {
                "max": (max(self._thr) if self._thr else None),
                "mean": (round(sum(self._thr) / len(self._thr), 3) if self._thr else None),
            },
            "steering_saturation": {
                "ticks": self._sat_ticks,
                "seconds": round(self._sat_ticks * self.control_period_s, 2),
            },
            "encoder_total_delta": (None if None in (self._enc_first, self._enc_last)
                                    else self._enc_last - self._enc_first),
            "comm_fault_events": self._comm_events,
            "battery_voltage": None,       # STATUS 에 없음 — 수집 불가
        }
        # Keep the historical summary schema byte-for-byte compatible when
        # video recording was not requested.
        if video is not None:
            summary["video"] = video
        (self.dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        for f in (self._pose_f, self._ctrl_f, self._evt_f):
            try:
                f.close()
            except Exception:                            # noqa: BLE001
                pass
        return summary
