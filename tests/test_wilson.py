import unittest

from mlcache.calibration import wilson_upper_bound


class WilsonUpperBoundTests(unittest.TestCase):
    def test_wilson_upper_bound_values(self) -> None:
        self.assertAlmostEqual(wilson_upper_bound(successes=25, n=500), 0.07277, places=5)
        self.assertAlmostEqual(wilson_upper_bound(successes=0, n=30), 0.1135, places=4)

    def test_wilson_upper_bound_empty_sample_returns_none(self) -> None:
        self.assertIsNone(wilson_upper_bound(successes=0, n=0))

    def test_wilson_upper_bound_invalid_inputs(self) -> None:
        cases = [
            (-1, 10, 1.96, "successes must be non-negative"),
            (11, 10, 1.96, "successes must be less than or equal to n"),
            (0, -1, 1.96, "n must be non-negative"),
            (0, 10, 0.0, "z must be positive"),
        ]

        for successes, n, z, message in cases:
            with self.subTest(successes=successes, n=n, z=z):
                with self.assertRaisesRegex(ValueError, message):
                    wilson_upper_bound(successes=successes, n=n, z=z)


if __name__ == "__main__":
    unittest.main()
