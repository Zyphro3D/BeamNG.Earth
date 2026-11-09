"""The slivers of ground between two carriageways (surface.narrow_gaps)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import surface  # noqa: E402
import vegetation  # noqa: E402


class NarrowGapsTest(unittest.TestCase):
    def two(self, gap):
        m = np.zeros((60, 60), bool)
        m[:, 10:20] = True
        m[:, 20 + gap:30 + gap] = True
        return m

    def test_a_narrow_sliver_is_filled(self):
        self.assertTrue(surface.narrow_gaps(self.two(4), 3)[30, 22])

    def test_a_wide_strip_is_left(self):
        self.assertFalse(surface.narrow_gaps(self.two(12), 3).any())

    def test_the_outer_edges_stay(self):
        got = surface.narrow_gaps(self.two(4), 3)
        self.assertFalse(got[:, :10].any() or got[:, 34:].any())

    def test_nothing_on_an_empty_mask(self):
        self.assertFalse(surface.narrow_gaps(np.zeros((8, 8), bool), 2).any())


class ConiferTest(unittest.TestCase):
    def test_lowland_and_mountain(self):
        lat, lon = np.array([43.7, 42.7]), np.array([1.4, 1.0])
        self.assertEqual(list(vegetation._french_lowland(lat, lon)), [True, False])


if __name__ == "__main__":
    unittest.main()
