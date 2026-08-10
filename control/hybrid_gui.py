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
    def __init__(self, root: tk.Tk, pipeline, car_id: int = 1) -> None:
        self.root = root
        self.pipeline = pipeline
        self.car_id = int(car_id)

        self.pressed: set[str] = set()
        self.current_steering = 0.0
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

        if key not in {"w", "a", "s", "d", "shift_l", "shift_r"}:
            return
        if key in self.pressed:
            return

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

        self.pressed.discard(key)
        if key in {"a", "d"}:
            if steering_direction(self.pressed) == 0:
                self.current_steering = 0.0
            else:
                self.current_steering = advance_steering(0.0, self.pressed)
        self._apply_keys()

    def _steering_tick(self) -> None:
        direction = steering_direction(self.pressed)
        next_value = (
            advance_steering(self.current_steering, self.pressed)
            if direction != 0
            else 0.0
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
