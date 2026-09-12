from __future__ import annotations

import unittest

from control.hybrid_gui import HybridControlWindow


class _Var:
    def __init__(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value


class _Pipeline:
    def __init__(self) -> None:
        self.commands = []
        self.stop_calls = []

    def set_manual_drive(self, car_id, throttle, steering) -> None:
        self.commands.append((car_id, throttle, steering))

    def manual_stop(self, car_id) -> None:
        self.stop_calls.append(car_id)


def _window(*, pressed, steering) -> HybridControlWindow:
    window = HybridControlWindow.__new__(HybridControlWindow)
    window.pipeline = _Pipeline()
    window.car_id = 1
    window.pressed = set(pressed)
    window.current_steering = steering
    window._release_jobs = {}
    window.last_sent = (None, None)
    window.drive_var = _Var()
    window.value_var = _Var()
    window.notice_var = _Var()
    return window


class TestHybridGuiManualSteering(unittest.TestCase):
    def test_reverse_alone_clears_previous_steering(self) -> None:
        window = _window(pressed={"s"}, steering=0.8)

        window._apply_keys()

        self.assertEqual(window.current_steering, 0.0)
        self.assertEqual(window.pipeline.commands[-1], (1, -1.0, 0.0))

    def test_last_steering_key_release_returns_to_center(self) -> None:
        window = _window(pressed={"s", "a"}, steering=-0.7)

        window._commit_release("a")

        self.assertEqual(window.pressed, {"s"})
        self.assertEqual(window.current_steering, 0.0)
        self.assertEqual(window.pipeline.commands[-1], (1, -1.0, 0.0))

    def test_reverse_with_held_steering_still_turns(self) -> None:
        window = _window(pressed={"s", "d"}, steering=0.6)

        window._apply_keys()

        self.assertEqual(window.current_steering, 0.6)
        self.assertEqual(window.pipeline.commands[-1], (1, -1.0, 0.6))

    def test_space_stop_clears_and_centers(self) -> None:
        window = _window(pressed={"s", "a"}, steering=-0.7)

        window._stop()

        self.assertEqual(window.pressed, set())
        self.assertEqual(window.current_steering, 0.0)
        self.assertEqual(window.pipeline.commands[-1], (1, 0.0, 0.0))
        self.assertEqual(window.pipeline.stop_calls, [1])


if __name__ == "__main__":
    unittest.main()
