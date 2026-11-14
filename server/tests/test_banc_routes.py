"""tools/banc_routes.py's pure detectors (the measuring needs the server and the map data: not here)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import banc_routes as br  # noqa: E402


class RunsTest(unittest.TestCase):
    def test_short_runs_are_noise(self):
        arc = np.arange(10.0)
        m = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1], bool)
        self.assertEqual(br.runs(m, arc), [(3, 5)])
        self.assertEqual(br.runs(m, arc, 1.0), [(3, 5), (8, 9)])


class StationsTest(unittest.TestCase):
    def test_every_metre_and_the_end(self):
        a, p, t = br.stations(np.array([[0.0, 0.0], [10.5, 0.0]]))
        self.assertEqual(len(a), 12)
        self.assertAlmostEqual(a[-1], 10.5)
        np.testing.assert_allclose(t, [[1.0, 0.0]] * 12, atol=1e-12)


class AcrossTest(unittest.TestCase):
    def test_a_bank_is_no_bump(self):
        offs = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        z = 10.0 + 0.05 * offs[None]
        slope, res = br.across_fit(offs, z)
        self.assertAlmostEqual(slope[0], 0.05)
        self.assertLess(res[0], 1e-9)

    def test_a_trough_in_the_lanes(self):
        offs = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        _s, res = br.across_fit(offs, np.array([[10.0, 10.0, 9.7, 10.0, 10.0]]))
        self.assertGreater(res[0], br.ACROSS_M)


class StepsTest(unittest.TestCase):
    def test_a_step_the_plan_does_not_take(self):
        arc = np.arange(6.0)
        plan = 0.05 * arc                                  # a 5 % grade
        ground = plan.copy()
        ground[3:] += 0.4
        self.assertEqual(list(np.flatnonzero(br.steps(arc, ground, plan))), [3])
        self.assertFalse(br.steps(arc, plan + 0.02, plan).any())


class NodesTest(unittest.TestCase):
    def test_three_roads_make_a_junction(self):
        ends = np.array([[0.0, 0.0], [50.0, 0.0], [0.5, 0.2], [0.0, 40.0], [0.3, -0.4], [9.0, 9.0]])
        rids = ["a", "a", "b", "b", "c", "c"]
        got = br.nodes(ends, rids)
        self.assertEqual(len(got), 1)
        self.assertEqual(sorted(got[0][1]), [0, 2, 4])
        self.assertEqual(br.nodes(ends[:4], rids[:4]), [])           # (two roads: a bend, no junction)


class PlaneTest(unittest.TestCase):
    def test_a_tilted_plane_fits(self):
        xy = np.array([[x, y] for x in range(-5, 6) for y in range(-5, 6)], float)
        slope, res = br.plane_residual(xy, 3.0 + 0.03 * xy[:, 0] + 0.04 * xy[:, 1])
        self.assertAlmostEqual(slope, 0.05)
        self.assertLess(res, 1e-9)


class CauseTest(unittest.TestCase):
    def layers(self, after):        # (6 layers, 1 station, 5 offsets): flat road at 10 m, `after` (layer: z) changed
        L = np.full((6, 1, 5), 10.0)
        for j, z in after.items():
            L[j:, 0, :] = z
        return L

    def test_the_step_that_moved_the_ground(self):
        offs = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        zp = np.array([10.0])
        self.assertEqual(br.cause("bas", self.layers({5: 8.0}), 0, zp, offs)[0], "murs")
        self.assertEqual(br.cause("haut", self.layers({3: 11.0}), 0, zp, offs)[0], "arches")
        self.assertEqual(br.cause("haut", self.layers({2: 11.0}), 0, zp, offs)[0], "nivellement")
        L = self.layers({})
        L[4:, 0, 2] = 9.0                                   # (a trough under a deck, mid-lane)
        self.assertEqual(br.cause("travers", L, 0, zp, offs)[0], "tabliers")

    def test_the_water_carve_is_noted(self):
        L = np.full((6, 1, 5), 10.0)
        L[1:, 0, 0] = 9.0
        self.assertTrue(br.cause("bas", L, 0, np.array([10.0]), np.array([-3.0, -1.5, 0.0, 1.5, 3.0]))[1])


class FarTest(unittest.TestCase):
    def test_points_off_every_line(self):
        lines = [np.array([[0.0, 0.0], [100.0, 0.0]])]
        pts = np.array([[50.0, 1.0], [50.0, 5.0], [150.0, 0.0]])
        self.assertEqual(list(br.far_from(pts, lines, 2.0)), [False, True, True])
        self.assertTrue(br.far_from(pts, [], 2.0).all())


if __name__ == "__main__":
    unittest.main()
