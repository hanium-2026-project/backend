from types import SimpleNamespace
import unittest
from controller.models import MotionDirection
from integration.backend_adapter import waypoint_from_backend

class TestBackendDirectionAdapter(unittest.TestCase):
    def _obj(self, **kw):
        base=dict(x=100.0,y=200.0,phase="RECOVERY")
        base.update(kw)
        return SimpleNamespace(**base)

    def test_default_forward(self):
        wp=waypoint_from_backend(self._obj())
        self.assertIs(wp.motion_direction, MotionDirection.FORWARD)

    def test_reverse_string(self):
        wp=waypoint_from_backend(self._obj(motion_direction="REVERSE"))
        self.assertIs(wp.motion_direction, MotionDirection.REVERSE)

    def test_invalid_direction_rejected(self):
        with self.assertRaises(ValueError):
            waypoint_from_backend(self._obj(motion_direction="SIDEWAYS"))

if __name__ == "__main__": unittest.main()
