"""racekerbs.py: a race track's bends get an inside kerb round the apex and an outside one at the exit; straights,
gentle curves and other roads none."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import racekerbs as rk  # noqa: E402


def bend(radius=60.0, turn=np.pi / 2):
    t = np.linspace(0, turn, 40)
    arc = np.stack([radius * np.sin(t), radius - radius * np.cos(t)], 1)
    end = arc[-1] + np.outer(np.linspace(0, 100, 30), [np.cos(turn), np.sin(turn)])
    line = np.vstack([np.stack([np.linspace(-100, 0, 30), np.zeros(30)], 1)[:-1], arc, end[1:]])
    return np.c_[line, np.zeros(len(line))]


class Spec:
    def __init__(self, cls="raceway", width=12.0, kind="paved"):
        self.cls, self.width, self.kind = cls, width, kind


class KerbTest(unittest.TestCase):
    def test_a_left_bend_inside_left_exit_right(self):
        ks = rk.kerb_lines(bend(), 12.0)
        self.assertEqual(len(ks), 2)
        centre = np.array([0.0, 60.0])
        inner, outer = (np.hypot(*(k[:, :2] - centre).T).mean() for k in ks)
        self.assertAlmostEqual(inner, 60.0 - (6.0 + rk.KERB_W / 2 - rk.KERB_INSET_M), delta=1.0)
        self.assertGreater(outer, 60.0)
        self.assertTrue(all(abs(float(k[0, 2]) - rk.KERB_LIFT_M) < 1e-9 for k in ks))

    def test_straights_and_gentle_curves_none(self):
        straight = np.c_[np.linspace(0, 500, 50), np.zeros(50), np.zeros(50)]
        self.assertEqual(rk.kerb_lines(straight, 12.0), [])
        self.assertEqual(rk.kerb_lines(bend(radius=600.0, turn=np.radians(10)), 12.0), [])

    def test_chunks_only_for_race_tracks(self):
        line = bend()
        got = rk.chunks([("osm-raceway-1", Spec(), line, 0.0)], (-500, -500, 500, 500))
        self.assertTrue(got and all(c["m"] == rk.MATERIAL and c["fid"] == "osm-raceway-1" for c in got))
        self.assertEqual(rk.chunks([("r", Spec("primary"), line, 0.0)], (-500, -500, 500, 500)), [])
        self.assertEqual(rk.chunks([("osm-raceway-1", Spec(), line, 0.0)], (1000, 1000, 1200, 1200)), [])


class LayoutsTest(unittest.TestCase):
    def test_a_crossing_track_cuts_the_kerb_a_shared_one_does_not_double_it(self):
        line = bend()
        box = (-500, -500, 500, 500)
        alone = rk.chunks([("osm-raceway-1", Spec(), line, 0.0)], box)
        n_alone = sum(len(c["n"]) for c in alone)
        same = rk.chunks([("osm-raceway-1", Spec(), line, 0.0), ("osm-raceway-2", Spec(), line.copy(), 0.0)], box)
        self.assertEqual(sum(len(c["n"]) for c in same), n_alone, "a layout on the same asphalt: no second kerb")
        across = np.c_[np.full(50, 30.0), np.linspace(-100, 200, 50), np.zeros(50)]     # (a track across the bend)
        cut = rk.chunks([("osm-raceway-1", Spec(), line, 0.0), ("osm-raceway-3", Spec(), across, 0.0)], box)
        pts = np.array([q for c in cut if c["fid"] == "osm-raceway-1" for q in c["n"]])
        self.assertTrue(len(pts) and (np.abs(pts[:, 0] - 30.0) > 6.0 - rk.KERB_INSET_M).all(), "none on its asphalt")


if __name__ == "__main__":
    unittest.main()
