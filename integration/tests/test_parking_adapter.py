from types import SimpleNamespace
import unittest
from integration.backend_adapter import waypoint_from_backend


class TestParkingAdapter(unittest.TestCase):
    def test_capture_tolerance_mapping(self):
        obj = SimpleNamespace(x=100.0, y=200.0, phase="APPROACH",
                              position_tolerance_cm=5.0, capture_tolerance_cm=10.0)
        wp = waypoint_from_backend(obj)
        self.assertEqual(wp.position_tolerance_cm, 5.0)
        self.assertEqual(wp.capture_tolerance_cm, 10.0)


if __name__ == "__main__":
    unittest.main()
