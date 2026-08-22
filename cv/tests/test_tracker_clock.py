"""Tracker observation timestamps share the controller monotonic clock."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cv.tracker import RCCarTracker


class _Camera:
    def __init__(self, _source) -> None:
        self.released = False

    def read_frame(self):
        return SimpleNamespace(frame_index=1, image=np.zeros((2, 3, 3)))

    def release(self) -> None:
        self.released = True


class _Detector:
    def detect_and_track(self, _image):
        return []


class TestTrackerClock(unittest.TestCase):
    def test_track_state_uses_monotonic_not_perf_counter(self) -> None:
        seen = []
        fake_cv2 = SimpleNamespace(destroyAllWindows=lambda: None)
        with (patch("cv.tracker.CameraCapture", _Camera),
              patch("cv.tracker.time.monotonic", side_effect=[10.0, 10.1, 10.11]),
              patch("cv.tracker.time.perf_counter",
                    side_effect=AssertionError("wrong clock domain")),
              patch("cv.tracker.time.sleep"),
              patch.dict(sys.modules, {"cv2": fake_cv2})):
            RCCarTracker(detector=_Detector()).run(
                on_frame=seen.append, max_frames=1, show=False)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].timestamp, 10.1)


if __name__ == "__main__":
    unittest.main()
