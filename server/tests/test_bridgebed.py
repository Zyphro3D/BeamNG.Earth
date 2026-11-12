"""The ground under road bridges out of the elevation model (bridgebed.lower, BedDem)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bridgebed as bb  # noqa: E402

LAT0, LON0 = 45.0, 5.0
KX = bb.M_PER_DEG * math.cos(math.radians(LAT0))


def ll(x, y):
    """(lat, lon) of points x m east, y m north of (LAT0, LON0)."""
    return LAT0 + np.asarray(y, np.float64) / bb.M_PER_DEG, LON0 + np.asarray(x, np.float64) / KX


def xy(la, lo):
    return (np.asarray(lo) - LON0) * KX, (np.asarray(la) - LAT0) * bb.M_PER_DEG


# a 100 m deck along x, its road 8 m wide; the DEM: ground at 10 m, a mound to 18 m under the deck (|y| <= 5)
la_d, lo_d = ll([0.0, 100.0], [0.0, 0.0])
PRINTS = [(np.c_[lo_d, la_d], 8.0)]


def dem(la, lo):
    x, y = xy(la, lo)
    return np.where((np.abs(y) <= 5.0) & (x > -5) & (x < 105), 18.0, 10.0)


def lowered(x, y, prints=PRINTS):
    la, lo = ll([x], [y])
    return float(bb.lower(la, lo, dem(la, lo), prints, dem)[0])


class LowerTest(unittest.TestCase):
    def test_the_mound_under_the_middle_goes(self):
        self.assertAlmostEqual(lowered(50.0, 0.0), 10.0, places=6)
        self.assertAlmostEqual(lowered(50.0, 4.5), 10.0, places=6)    # (under the slab's overhang)

    def test_the_abutments_stay(self):
        self.assertEqual(lowered(5.0, 0.0), 18.0)
        self.assertEqual(lowered(95.0, 0.0), 18.0)

    def test_the_change_comes_in_over_the_ramp(self):
        got = lowered(bb.END_KEEP_M + bb.RAMP_M / 2, 0.0)
        self.assertAlmostEqual(got, 14.0, places=6)

    def test_beside_the_deck_left_alone(self):
        self.assertEqual(lowered(50.0, 7.0), 10.0)
        la, lo = ll([50.0], [7.0])
        z = np.array([25.0])
        self.assertIs(bb.lower(la, lo, z, PRINTS, dem), z)

    def test_never_raised(self):
        la, lo = ll([50.0], [0.0])
        self.assertEqual(float(bb.lower(la, lo, np.array([4.0]), PRINTS, dem)[0]), 4.0)

    def test_a_slope_across_followed(self):
        def tilted(la, lo):
            x, y = xy(la, lo)
            return 10.0 + 0.5 * y + np.where(np.abs(y) <= 5.0, 8.0, 0.0)
        la, lo = ll([50.0, 50.0], [0.0, 3.0])
        got = bb.lower(la, lo, tilted(la, lo), PRINTS, tilted)
        np.testing.assert_allclose(got, [10.0, 11.5], atol=1e-6)

    def test_a_cells_bridges_as_footprints(self):
        from test_bridge_scenarios import scenario
        f = scenario([[(0, -300), (0, 300)]], ["primary"])     # (a 60 m deck over a street: BRIDGE)
        got = bb.footprints(f)
        self.assertEqual(len(got), 1)
        lonlat, width = got[0]
        la0 = float(lonlat[0, 1])
        k = bb.M_PER_DEG * math.cos(math.radians(la0))
        length = float(np.hypot(np.diff(lonlat[:, 0]) * k, np.diff(lonlat[:, 1]) * bb.M_PER_DEG).sum())
        self.assertAlmostEqual(length, 60.0, delta=8.0)
        self.assertGreater(width, 5.0)

    def test_shape_and_dtype_kept(self):
        la, lo = ll(np.array([[50.0, 50.0], [5.0, 60.0]]), np.zeros((2, 2)))
        z = dem(la, lo).astype(np.float32)
        got = bb.lower(la, lo, z, PRINTS, dem)
        self.assertEqual(got.shape, (2, 2))
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_allclose(got, [[10.0, 10.0], [18.0, 10.0]])


class FakeDem:
    def __init__(self):
        self.calls = 0

    def sample(self, la, lo, zoom, progress=None):
        self.calls += 1
        return dem(la, lo)

    coarse_at = "passed through"


class BedDemTest(unittest.TestCase):
    def test_reads_the_cells_footprints_once_and_passes_the_rest(self):
        asked = []
        d = bb.BedDem(FakeDem(), lambda key: asked.append(str(key)) or "feats")
        orig = bb.footprints
        bb.footprints = lambda feats: PRINTS if feats == "feats" else []
        try:
            la, lo = ll([50.0], [0.0])
            self.assertAlmostEqual(float(d.sample(la, lo, 16)[0]), 10.0, places=6)
            n = len(asked)
            d.sample(la, lo, 16)
            self.assertEqual(len(asked), n, "a cell's footprints are kept")
            self.assertEqual(d.coarse_at, "passed through")
        finally:
            bb.footprints = orig

    def test_off(self):
        d = bb.BedDem(FakeDem(), lambda key: "feats")
        orig, on = bb.footprints, bb.ON
        bb.footprints, bb.ON = (lambda feats: PRINTS), False
        try:
            la, lo = ll([50.0], [0.0])
            self.assertEqual(float(d.sample(la, lo, 16)[0]), 18.0)
        finally:
            bb.footprints, bb.ON = orig, on


if __name__ == "__main__":
    unittest.main()
