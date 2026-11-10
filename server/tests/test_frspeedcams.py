"""French fixed speed cameras (frspeedcams.py): the bundled data, which way a camera looks, where its post stands."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import frspeedcams as fc          # noqa: E402
import geodesy                    # noqa: E402
import signmesh                   # noqa: E402
import signplacement              # noqa: E402

LAT, LON = 45.0, 2.0


def road(fid, pts, width=7.0, oneway=False, direction=0, kind="paved", cls="primary"):
    line = np.c_[np.asarray(pts, float), np.zeros(len(pts))]
    return (fid, SimpleNamespace(width=width, oneway=oneway, dir=direction, kind=kind, cls=cls, link=False), line, None)


def row(frame, x, y, heading=-1, tower=0):
    lat, lon, _ = frame.enu_to_geodetic(x, y, 0.0)
    return [round(float(lon) * 1e7), round(float(lat) * 1e7), heading, tower]


def place(roads, rows, frame):
    mesh = signmesh.Mesh((0, 0, 0))
    pavement = signplacement.pavement([(sp.width / 2, line) for _, sp, line, _ in roads])
    got = fc.add(frame, roads, (-500.0, -500.0, 500.0, 500.0), lambda x, y, z: True, mesh, pavement,
                 signplacement.Context(ground=lambda x, y: np.zeros(len(x))), signplacement.Context(), data=rows)
    return got, mesh


class SpeedCameras(unittest.TestCase):
    def setUp(self):
        self.frame = geodesy.EnuFrame(LAT, LON)

    def test_data_loads(self):
        rows = [r for cell in fc._catalogue().values() for r in cell]
        self.assertGreater(len(rows), 3000)
        mainland = [r for r in rows if -55_000_000 <= r[0] <= 100_000_000 and 410_000_000 <= r[1] <= 515_000_000]
        self.assertGreater(len(mainland), 0.9 * len(rows), "mostly mainland France (and its overseas regions)")
        for lon, lat, heading, tower in rows:
            self.assertTrue(heading == -1 or 0 <= heading < 360)

    def test_nearby_finds_a_known_camera(self):
        frame = geodesy.EnuFrame(48.8874023, 2.2517028)
        self.assertEqual(len(fc.nearby(frame, (-50.0, -50.0, 50.0, 50.0))), 1)

    def test_post_beside_the_road_on_the_right_of_the_enforced_traffic(self):
        roads = [road(1, [(-200, 0), (200, 0)])]                 # an east-west road, 7 m wide
        got, mesh = place(roads, [row(self.frame, 0, 0, heading=90)], self.frame)     # eastbound traffic
        self.assertEqual(len(got), 1)
        x, y, _ = got[0]["pos"]
        self.assertLess(y, -3.5, "on the south side, the right of eastbound traffic, off the pavement")
        self.assertGreater(y, -3.5 - 4.0)
        self.assertTrue(mesh.t, "drawn")

    def test_heading_picks_the_side(self):
        roads = [road(1, [(-200, 0), (200, 0)])]
        got, _ = place(roads, [row(self.frame, 0, 0, heading=270)], self.frame)      # westbound
        self.assertGreater(got[0]["pos"][1], 3.5)

    def test_untagged_node_keeps_its_mapped_side(self):
        roads = [road(1, [(-200, 0), (200, 0)])]
        got, _ = place(roads, [row(self.frame, 0, 6)], self.frame)       # mapped north of a west-to-east line
        self.assertGreater(got[0]["pos"][1], 3.5, "north side: it watches the westbound traffic")

    def test_one_way_carriageway_against_the_heading_is_skipped(self):
        roads = [road(1, [(-200, 0), (200, 0)], oneway=True, direction=1),
                 road(2, [(-200, 20), (200, 20)], oneway=True, direction=-1)]
        got, _ = place(roads, [row(self.frame, 0, 2, heading=270)], self.frame)
        self.assertGreater(got[0]["pos"][1], 20 + 3.5, "the westbound carriageway's right, the north one")

    def test_in_the_median_between_two_carriageways(self):
        roads = [road(1, [(-200, 0), (200, 0)], width=8.0, oneway=True, direction=1, cls="motorway"),
                 road(2, [(-200, 14), (200, 14)], width=8.0, oneway=True, direction=-1, cls="motorway")]
        got, _ = place(roads, [row(self.frame, 0, 7, heading=90)], self.frame)
        self.assertEqual(got[0]["support"], "median")
        self.assertAlmostEqual(got[0]["pos"][1], 7.0, delta=0.5)

    def test_far_from_any_road_is_not_placed(self):
        roads = [road(1, [(-200, 0), (200, 0)])]
        got, _ = place(roads, [row(self.frame, 0, 60)], self.frame)
        self.assertEqual(got, [])

    def test_tower_is_drawn(self):
        roads = [road(1, [(-200, 0), (200, 0)])]
        _, cabin = place(roads, [row(self.frame, 0, -6)], self.frame)
        _, tower = place(roads, [row(self.frame, 0, -6, tower=1)], self.frame)
        self.assertTrue(cabin.t and tower.t)
        self.assertGreater(np.asarray(tower.p)[:, 2].max(), 3.9)
        self.assertLess(np.asarray(cabin.p)[:, 2].max(), 3.0)


if __name__ == "__main__":
    unittest.main()
