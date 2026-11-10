"""French toll plazas (frtolls.py): islands between the toll lanes, a raised barrier per lane, the bundled data."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import buildings                  # noqa: E402
import frtolls as ft              # noqa: E402
import geodesy                    # noqa: E402


def lane(fid, x, width=3.5):
    """A northbound toll lane's way at x (anchor frame)."""
    line = np.c_[np.array([[x, -150.0], [x, 150.0]]), np.zeros(2)]
    return (fid, SimpleNamespace(width=width, oneway=True, dir=1, kind="paved", cls="motorway", link=False), line, None)


def node(frame, x, y=0.0):
    lat, lon, _ = frame.enu_to_geodetic(x, y, 0.0)
    return [round(float(lon) * 1e7), round(float(lat) * 1e7)]


class Tolls(unittest.TestCase):
    def setUp(self):
        self.frame = geodesy.EnuFrame(45.0, 2.0)
        self.box = (-200.0, -200.0, 200.0, 200.0)

    def test_data_loads(self):
        rows = [r for cell in ft._catalogue().values() for r in cell]
        self.assertGreater(len(rows), 1000)

    def test_islands_between_and_beside_the_lanes(self):
        xs = (0.0, 5.5, 11.0)
        roads = [lane(i, x) for i, x in enumerate(xs)]
        got = ft.rows(self.frame, roads, self.box, data=[node(self.frame, x) for x in xs])
        booths = sorted(r[1] for r in got if r[0] == buildings.G_TOLLBOOTH)
        np.testing.assert_allclose(booths, [-2.75, 2.75, 8.25, 13.75], atol=0.05)
        for r in got:
            if r[0] == buildings.G_TOLLBOOTH:
                self.assertAlmostEqual(r[4] % 180.0, 0.0, delta=0.5, msg="the island's length along the lanes")
        gates = sorted(r[1] for r in got if r[0] == ft.G_GATE)
        self.assertEqual(len(gates), 3, "one barrier per lane")
        for x, g in zip(xs, gates):
            self.assertGreater(g, x + 1.5, "on the lane's right island, clear of the lane")
            self.assertLess(g, x + 5.5)
        for r in got:
            if r[0] == ft.G_GATE:
                self.assertAlmostEqual(r[2], ft.GATE_ALONG_M, delta=0.05, msg="just past the booth")

    def test_lane_never_covered(self):
        xs = (0.0, 4.6, 9.2, 13.8)
        roads = [lane(i, x) for i, x in enumerate(xs)]
        got = ft.rows(self.frame, roads, self.box, data=[node(self.frame, x) for x in xs])
        for r in got:
            half = ft.ISLAND_HALF_M if r[0] == buildings.G_TOLLBOOTH else 0.3
            for x in xs:
                self.assertGreater(abs(r[1] - x), 1.5 + half - 1e-6, f"{r[0]} on the lane at {x}")

    def test_lone_node_on_a_wide_carriageway(self):
        roads = [lane(0, 0.0, width=11.0)]
        got = ft.rows(self.frame, roads, self.box, data=[node(self.frame, 0.0)])
        booths = sorted(r[1] for r in got if r[0] == buildings.G_TOLLBOOTH)
        self.assertEqual(len(booths), 2)
        self.assertLess(booths[0], -5.5 - ft.ISLAND_HALF_M + 1e-6)
        self.assertGreater(booths[1], 5.5 + ft.ISLAND_HALF_M - 1e-6)

    def test_node_off_the_roads_is_ignored(self):
        got = ft.rows(self.frame, [lane(0, 0.0)], self.box, data=[node(self.frame, 40.0)])
        self.assertEqual(got, [])


if __name__ == "__main__":
    unittest.main()
