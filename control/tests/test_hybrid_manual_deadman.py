"""Manual GUI input is a renewable lease, never an infinite motion latch."""

from __future__ import annotations

import threading
import unittest

from control.hybrid_control import HybridControlMux
from host_control.producers import ManualInput


class TestManualInputLease(unittest.TestCase):
    def setUp(self) -> None:
        self.mux = HybridControlMux.__new__(HybridControlMux)
        self.mux._lock = threading.Lock()
        self.mux._manual = ManualInput(1.0, -0.5)
        self.mux._manual_updated_at = 10.0

    def test_fresh_input_is_preserved(self) -> None:
        current = self.mux._manual_for_tick(10.2)
        self.assertEqual((current.throttle, current.steering), (1.0, -0.5))

    def test_lost_release_expires_to_zero(self) -> None:
        expired = self.mux._manual_for_tick(
            10.0 + self.mux.MANUAL_INPUT_LEASE_S + 0.01)
        self.assertEqual((expired.throttle, expired.steering), (0.0, 0.0))
        self.assertEqual((self.mux._manual.throttle, self.mux._manual.steering),
                         (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
