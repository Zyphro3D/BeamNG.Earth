"""IGN orthophoto colours for the ground (ignlook.py)."""
import os
import sys
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import franceroads as fr  # noqa: E402
import ignlook  # noqa: E402


def one(r, g, b):
    return ignlook.adjust(np.array([[[r, g, b]]], np.uint8))[0, 0].astype(int)


class AdjustTest(unittest.TestCase):
    def test_a_blue_shadow_loses_its_cast(self):
        r, g, b = one(47, 64, 67)
        self.assertLess(b, g)
        self.assertLess(b, 56)

    def test_a_dark_green_tree_keeps_its_colour(self):
        self.assertEqual(list(one(30, 60, 30)), [30, 60, 30])

    def test_a_mid_tone_is_left_alone(self):
        self.assertEqual(list(one(120, 125, 110)), [120, 125, 110])

    def test_a_near_white_yard_is_no_longer_snow(self):
        r, g, b = one(230, 235, 232)
        self.assertLess((r + g + b) / 3, 200)
        self.assertLess(b, r)

    def test_the_image_keeps_its_shape_and_type(self):
        got = ignlook.adjust(np.zeros((4, 3, 3), np.uint8))
        self.assertEqual(got.shape, (4, 3, 3))
        self.assertEqual(got.dtype, np.uint8)


class FranceBoxTest(unittest.TestCase):
    def test_inside_and_outside(self):
        with unittest.mock.patch.object(fr, "ON", True):
            self.assertTrue(fr.in_france(43.7, 1.4))
            self.assertFalse(fr.in_france(40.0, -3.7))

    def test_off_without_the_french_roads(self):
        with unittest.mock.patch.object(fr, "ON", False):
            self.assertFalse(fr.in_france(43.7, 1.4))


if __name__ == "__main__":
    unittest.main()
