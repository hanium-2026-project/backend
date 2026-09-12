import unittest

from cv.heading import HeadingEstimator


class TestHeadingJumpSafety(unittest.TestCase):
    def test_implausible_front_cushion_flip_holds_last_valid(self) -> None:
        est = HeadingEstimator()
        first = est.update(1, (100.0, 100.0), front_point=(160.0, 100.0))
        jumped = est.update(1, (100.0, 100.0), front_point=(40.0, 100.0))
        confirmed = est.update(1, (100.0, 100.0), front_point=(40.0, 100.0))
        self.assertEqual(first.heading_deg, 0.0)
        self.assertEqual(jumped.heading_deg, 0.0)
        self.assertEqual(jumped.source, "LAST_VALID")
        self.assertEqual(confirmed.heading_deg, 180.0)
        self.assertEqual(confirmed.source, "FRONT_CUSHION")

    def test_bad_first_cushion_does_not_latch_forever(self) -> None:
        est = HeadingEstimator()
        bad = est.update(1, (165.0, 656.0), front_point=(115.0, 630.5))
        held = est.update(1, (165.0, 656.0), front_point=(225.0, 656.0))
        corrected = est.update(1, (165.0, 656.0), front_point=(225.0, 656.0))
        self.assertAlmostEqual(bad.heading_deg, 207.0, delta=1.0)
        self.assertEqual(held.source, "LAST_VALID")
        self.assertAlmostEqual(corrected.heading_deg, 0.0, delta=0.1)
        self.assertEqual(corrected.source, "FRONT_CUSHION")

    def test_small_wrapped_change_is_accepted(self) -> None:
        est = HeadingEstimator()
        est.update(1, (100.0, 100.0), front_point=(160.0, 95.0))
        changed = est.update(1, (100.0, 100.0), front_point=(160.0, 105.0))
        self.assertEqual(changed.source, "FRONT_CUSHION")


if __name__ == "__main__":
    unittest.main()
