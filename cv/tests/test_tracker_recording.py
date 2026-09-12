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
        return SimpleNamespace(
            frame_index=1, image=np.zeros((480, 640, 3), dtype=np.uint8))

    def release(self) -> None:
        self.released = True


class _Detector:
    def detect_and_track(self, _image):
        return []


class TestTrackerRecording(unittest.TestCase):
    def _run(self, *, show: bool):
        frames = []
        displayed = []
        fake_cv2 = SimpleNamespace(
            imshow=lambda _name, image: displayed.append(image.copy()),
            waitKey=lambda _delay: -1,
            destroyAllWindows=lambda: None,
        )
        tracker = RCCarTracker(detector=_Detector())
        tracker.overlay = lambda image, _state: image + 2
        with (patch("cv.tracker.CameraCapture", _Camera),
              patch("cv.tracker._draw_detections",
                    side_effect=lambda image, _state: image + 1),
              patch.dict(sys.modules, {"cv2": fake_cv2})):
            tracker.run(max_frames=1, show=show,
                        frame_sink=lambda image, state: frames.append(
                            (image.copy(), state.frame_index)))
        return frames, displayed

    def test_frame_sink_receives_annotated_frame_without_show(self) -> None:
        frames, displayed = self._run(show=False)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][1], 1)
        self.assertTrue(np.all(frames[0][0] == 3))
        self.assertEqual(displayed, [])

    def test_show_and_frame_sink_share_same_annotated_frame(self) -> None:
        frames, displayed = self._run(show=True)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(displayed), 1)
        self.assertTrue(np.array_equal(frames[0][0], displayed[0]))


if __name__ == "__main__":
    unittest.main()
