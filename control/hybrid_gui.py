"""Developer GUI for switching MANUAL_WASD <-> AUTO_HOST.

The GUI shares the same ParkingPipeline/VehicleServer session.
It never opens automatically on import.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from control.wasd_logic import (
    STEERING_RAMP_INTERVAL_MS,
    advance_steering,
    compute_drive_intent,
    steering_direction,
)


class HybridControlWindow:
    RELEASE_DEBOUNCE_MS = 60      # 자동반복 릴리스 무시 창
    def __init__(self, root: tk.Tk, pipeline, car_id: int = 1) -> None:
        self.root = root
        self.pipeline = pipeline
        self.car_id = int(car_id)

        self.pressed: set[str] = set()
        self.current_steering = 0.0
        # macOS Tk 는 키를 누르고 있으면 KeyPress/KeyRelease 를 반복 발생시킨다.
        # 릴리스를 그대로 믿으면 조향 누적이 매번 0 으로 리셋돼 10% 에서 멈춘다.
        # 릴리스를 잠깐 미뤄두고, 그 사이 같은 키의 프레스가 오면 취소한다.
        self._release_jobs: dict[str, str] = {}
        self.last_sent = (None, None)

        root.title("Hanium RC Car - MANUAL WASD / AUTO HOST")
        root.geometry("650x430")
        root.minsize(590, 390)
        root.protocol("WM_DELETE_WINDOW", self._close)

        self.mode_var = tk.StringVar(value="mode: UNAVAILABLE")
        self.drive_var = tk.StringVar(value="STOP / CENTER")
        self.value_var = tk.StringVar(value="throttle=0.0  steering=0.0")
        self.notice_var = tk.StringVar(value="AUTO_HOST mission/session 생성 대기 중")

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="RC카 개발용 제어",
            font=("Malgun Gothic", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(outer, text=f"CAR ID: {self.car_id}").pack(anchor="w", pady=(4, 0))
        ttk.Label(
            outer,
            textvariable=self.mode_var,
            font=("Consolas", 13, "bold"),
        ).pack(anchor="w", pady=(10, 0))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=12)
        ttk.Button(buttons, text="MANUAL WASD", command=self._manual_mode).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(buttons, text="AUTO HOST", command=self._auto_mode).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(buttons, text="STOP / HOLD", command=self._stop).pack(side="left")

        box = ttk.LabelFrame(outer, text="현재 WASD 명령", padding=12)
        box.pack(fill="x", pady=8)
        ttk.Label(
            box,
            textvariable=self.drive_var,
            font=("Consolas", 17, "bold"),
        ).pack()
        ttk.Label(
            box,
            textvariable=self.value_var,
            font=("Consolas", 12),
        ).pack(pady=(6, 0))

        ttk.Label(
            outer,
            text=(
                "W 전진 / S 후진(HostController 설정에서 허용된 경우)\n"
                "A/D 조향, 오래 누르면 10%씩 증가 / Shift+A,D는 20%씩 증가\n"
                "Space = STOP/HOLD / F1 = MANUAL / F2 = AUTO\n"
                "AUTO 전환 후 새 Camera Pose가 올 때까지 AUTO_PENDING 정지 유지"
            ),
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(
            outer,
            textvariable=self.notice_var,
            wraplength=610,
        ).pack(anchor="w", pady=(12, 0))

        root.bind_all("<KeyPress>", self._key_press)
        root.bind_all("<KeyRelease>", self._key_release)
        root.bind("<FocusOut>", self._focus_out)

        root.after(STEERING_RAMP_INTERVAL_MS, self._steering_tick)
        root.after(100, self._poll)

    @staticmethod
    def _norm(keysym: str) -> str:
        return keysym.lower()

    def _manual_mode(self) -> None:
        self._clear_keys(send_zero=True)
        try:
            self.pipeline.switch_to_manual(self.car_id)
            self.notice_var.set("MANUAL_WASD: WASD 입력 가능")
        except RuntimeError as exc:
            self.notice_var.set(str(exc))

    def _auto_mode(self) -> None:
        self._clear_keys(send_zero=True)
        try:
            self.pipeline.switch_to_auto(self.car_id)
            self.notice_var.set("AUTO_PENDING: 새 Camera Pose 수신 후 AUTO_HOST 재개")
        except RuntimeError as exc:
            self.notice_var.set(str(exc))

    def _stop(self) -> None:
        self._clear_keys(send_zero=True)
        try:
            self.pipeline.manual_stop(self.car_id)
            self.notice_var.set("STOP/HOLD: MANUAL 중립 유지. AUTO 재개는 F2")
        except RuntimeError as exc:
            self.notice_var.set(str(exc))

    def _key_press(self, event: tk.Event) -> None:
        key = self._norm(str(event.keysym))

        if key == "f1":
            self._manual_mode()
            return
        if key == "f2":
            self._auto_mode()
            return
        if key == "space":
            self._stop()
            return

        if key == "c":                       # 조향 중앙 복귀
            self.current_steering = 0.0
            self._apply_keys()
            return
        if key not in {"w", "a", "s", "d", "shift_l", "shift_r"}:
            return
        if key in self.pressed:
            return

        job = self._release_jobs.pop(key, None)
        if job is not None:
            self.root.after_cancel(job)     # 자동반복이었다 → 릴리스 취소

        self.pressed.add(key)
        if key in {"a", "d"}:
            self.current_steering = advance_steering(
                self.current_steering, self.pressed
            )
        self._apply_keys()

    def _key_release(self, event: tk.Event) -> None:
        key = self._norm(str(event.keysym))
        if key not in self.pressed:
            return
        # 진짜 뗀 것인지 자동반복인지 구분되지 않으므로 짧게 유예한다.
        job = self._release_jobs.pop(key, None)
        if job is not None:
            self.root.after_cancel(job)
        self._release_jobs[key] = self.root.after(
            self.RELEASE_DEBOUNCE_MS, lambda k=key: self._commit_release(k))

    def _commit_release(self, key: str) -> None:
        """키를 뗀 것으로 확정. 조향은 **초기화하지 않는다**(래치).

        macOS 는 자동반복을 마지막에 누른 키에만 적용한다. D 를 누른 채 W 를
        누르면 D 의 반복이 끊기고 릴리스가 확정돼 조향이 0 으로 돌아갔다.
        릴리스로 중앙 복귀시키는 방식은 이 환경에서 신뢰할 수 없어, 조향은
        A/D 로 올린 값을 유지하고 C(또는 정지)로만 중앙으로 되돌린다.
        """
        self._release_jobs.pop(key, None)
        self.pressed.discard(key)
        self._apply_keys()

    def _steering_tick(self) -> None:
        # A/D 를 누르고 있는 동안만 값을 올리고, 떼면 **그 값을 유지**한다(래치).
        # 0 으로 되돌리면 안 된다 — macOS 는 W 를 누르는 순간 D 의 자동반복을
        # 끊어 pressed 에서 빠지므로, 여기서 리셋하면 D 100% 가 풀린다.
        direction = steering_direction(self.pressed)
        next_value = (
            advance_steering(self.current_steering, self.pressed)
            if direction != 0
            else self.current_steering
        )
        if next_value != self.current_steering:
            self.current_steering = next_value
            self._apply_keys()

        self.root.after(STEERING_RAMP_INTERVAL_MS, self._steering_tick)

    def _apply_keys(self) -> None:
        intent = compute_drive_intent(self.pressed, self.current_steering)
        current = (intent.throttle, intent.steering)

        if current != self.last_sent:
            self.pipeline.set_manual_drive(
                self.car_id,
                intent.throttle,
                intent.steering,
            )
            self.last_sent = current

        self.drive_var.set(intent.label)
        self.value_var.set(
            f"throttle={intent.throttle:+.1f}  steering={intent.steering:+.1f}"
        )

    def _clear_keys(self, *, send_zero: bool) -> None:
        self.pressed.clear()
        self.current_steering = 0.0
        self.last_sent = (0.0, 0.0)

        if send_zero:
            self.pipeline.set_manual_drive(self.car_id, 0.0, 0.0)

        self.drive_var.set("STOP / CENTER")
        self.value_var.set("throttle=+0.0  steering=+0.0")

    def _focus_out(self, _event: tk.Event) -> None:
        self._clear_keys(send_zero=True)

    def _poll(self) -> None:
        available = self.pipeline.hybrid_available(self.car_id)
        mode = self.pipeline.hybrid_mode(self.car_id)
        self.mode_var.set(f"mode: {mode}")

        if not available:
            self.notice_var.set(
                "AUTO_HOST mission/session 생성 대기 중 - 차량 배정 후 MANUAL/AUTO 전환 가능"
            )

        self.root.after(100, self._poll)

    def _close(self) -> None:
        self._clear_keys(send_zero=True)
        try:
            self.pipeline.manual_stop(self.car_id)
        except RuntimeError:
            pass
        self.root.destroy()


def run_gui(pipeline, car_id: int = 1) -> None:
    root = tk.Tk()
    HybridControlWindow(root, pipeline, car_id)
    root.mainloop()
