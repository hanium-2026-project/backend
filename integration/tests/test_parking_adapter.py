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

    def test_arc_tangent_is_preserved_without_forcing_arrival_heading(self):
        obj = SimpleNamespace(
            x=100.0, y=200.0, phase="ALIGN",
            target_heading_deg=330.0, heading_required=False,
            curvature=-1.0 / 800.0,
            path_capture_tolerance_cm=10.0,
        )
        wp = waypoint_from_backend(obj)
        self.assertEqual(wp.target_heading_deg, 330.0)
        self.assertFalse(wp.heading_required)
        self.assertAlmostEqual(wp.curvature, -1.0 / 800.0)
        self.assertEqual(wp.path_capture_tolerance_cm, 10.0)


if __name__ == "__main__":
    unittest.main()
