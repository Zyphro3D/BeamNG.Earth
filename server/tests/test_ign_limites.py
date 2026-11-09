"""Walls, fences and hedges on the property lines (ign_limites.py), the canopy from IGN's lidar (ign_lidar.py)."""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import geodesy  # noqa: E402
import ign_lidar  # noqa: E402
import ign_limites as L  # noqa: E402
import mapcells  # noqa: E402
import revisit  # noqa: E402

N = 100


def scene(height=None):
    h = np.zeros((N, N), np.float32)
    if height is not None:
        h[50, 20:60] = height
    return h, np.zeros((N, N), bool), np.full((N, N, 3), 120.0)


class DetectTest(unittest.TestCase):
    EDGE = [(10.5, 50.5, 80.5, 50.5)]

    def test_a_wall_on_the_line_is_found_with_its_height(self):
        h, mask, photo = scene(1.8)
        got = L.detect(self.EDGE, h, mask, photo, .5)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][4], 0)
        self.assertAlmostEqual(got[0][5], 1.8, places=3)
        self.assertLess(abs((got[0][2] - got[0][0]) * .5 - 20), 2.5)

    def test_green_along_the_line_is_a_hedge(self):
        h, mask, photo = scene(1.5)
        photo[..., 1] = 160
        self.assertEqual(L.detect(self.EDGE, h, mask, photo, .5)[0][4], 2)

    def test_bare_ground_and_tall_things_are_nothing(self):
        for height in (None, .3, 8.0):
            h, mask, photo = scene(height)
            self.assertEqual(L.detect(self.EDGE, h, mask, photo, .5), [])

    def test_a_short_run_is_nothing(self):
        h, mask, photo = scene()
        h[50, 20:24] = 1.5
        self.assertEqual(L.detect(self.EDGE, h, mask, photo, .5), [])

    def test_nothing_along_a_building(self):
        h, mask, photo = scene(1.8)
        mask[45:55, :] = True
        self.assertEqual(L.detect(self.EDGE, h, mask, photo, .5), [])

    def test_a_missing_height_is_no_barrier(self):
        h, mask, photo = scene()
        h[:] = np.nan
        self.assertEqual(L.detect(self.EDGE, h, mask, photo, .5), [])


class PanelsTest(unittest.TestCase):
    def test_a_low_barrier_is_a_fence_and_a_high_one_a_wall(self):
        self.assertEqual({p[0] for p in L._panels(0, 0, 30, 0, 0, 1.2)}, {L.FENCE})
        self.assertEqual({p[0] for p in L._panels(0, 0, 30, 0, 0, 2.0)}, {L.WALL})
        self.assertEqual({p[0] for p in L._panels(0, 0, 30, 0, 2, 2.0)}, {L.HEDGE})

    def test_panels_follow_the_line(self):
        panels = list(L._panels(0, 0, 0, 33, 0, 2.0))
        self.assertEqual(len(panels), 10)
        self.assertAlmostEqual(panels[0][3], 90.0)
        self.assertTrue(all(abs(p[1]) < 1e-9 for p in panels))
        self.assertLess(panels[-1][2], 33)


class CanopyTest(unittest.TestCase):
    CELL = mapcells.CellKey(14, 8254, 5977)

    def test_heights_become_the_canopy_grid(self):
        got = ign_lidar.to_canopy(np.array([[0.2, 7.6], [-99999.0, 300.0]], np.float32))
        self.assertEqual(got.tolist(), [[0, 8], [ign_lidar.NODATA, ign_lidar.NODATA]])

    def base(self):
        return SimpleNamespace(data=np.full((1024, 1024), 3, np.uint8))

    def test_a_pixel_without_lidar_keeps_the_world_model(self):
        values = np.full((512, 512), -99999.0, np.float32)
        values[0, 0] = 12.0

        class F:
            def get(self, url, kind=None, lane=None):
                return SimpleNamespace(body=values.tobytes())
        base, names = ign_lidar.canopy(F(), self.CELL, self.base(), None)
        self.assertEqual(base.data[0, 0], 12)
        self.assertEqual(base.data[5, 5], 3)
        self.assertEqual(names, ["ign-lidar-hd-mnh"])

    def test_a_failure_keeps_the_world_model(self):
        class Down:
            def get(self, *a, **k):
                raise OSError("no network")
        base, names = ign_lidar.canopy(Down(), self.CELL, self.base(), None)
        self.assertTrue((base.data == 3).all())
        self.assertEqual(names, [])

    def test_outside_france_nothing_is_asked(self):
        class Never:
            def get(self, *a, **k):
                raise AssertionError("asked")
        far = mapcells.CellKey(14, 2000, 6000)
        self.assertEqual(ign_lidar.canopy(Never(), far, self.base(), None)[1], [])


class BackgroundTest(unittest.TestCase):
    cell = mapcells.CellKey(14, 8254, 5976)
    frame = geodesy.EnuFrame(43.7, 1.37, 0.0)
    box = (-50.0, -50.0, 50.0, 50.0)

    def provider(self, tmp, seg=None):
        p = L.Provider(tmp, None)
        p.built = []

        def build(cell):
            p.built.append(cell)
            return np.zeros((0, 6)) if seg is None else seg
        p._build = build
        return p

    def drain(self, p):
        if p._later is not None:
            p._later.shutdown(wait=True)
            p._later = None

    def test_a_tile_does_not_wait_and_is_not_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.provider(tmp)
            before = revisit.faults()
            self.assertEqual(L.rows(self.frame, p, self.box, lambda la, lo: np.zeros(np.size(la))), [])
            self.assertGreater(revisit.faults(), before)
            self.drain(p)
            self.assertTrue(p.built)

    def test_once_worked_out_a_cell_is_read_without_the_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.provider(tmp)
            p.fetch_later(self.cell)
            p.fetch_later(self.cell)
            self.drain(p)
            self.assertEqual(len(p.built), 1)
            self.assertTrue(p.ready(self.cell))
            self.assertFalse(os.path.exists(p._file(self.cell) + ".claim"))
            self.assertTrue(L.Provider(tmp, None).ready(self.cell))

    def test_another_process_at_it_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.provider(tmp)
            os.makedirs(os.path.dirname(p._file(self.cell)))
            open(p._file(self.cell) + ".claim", "w").close()
            p.fetch_later(self.cell)
            self.drain(p)
            self.assertEqual(p.built, [])

    def test_a_dead_process_claim_runs_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.provider(tmp)
            os.makedirs(os.path.dirname(p._file(self.cell)))
            claim = p._file(self.cell) + ".claim"
            open(claim, "w").close()
            old = 1000.0
            os.utime(claim, (old, old))
            p.fetch_later(self.cell)
            self.drain(p)
            self.assertEqual(len(p.built), 1)


if __name__ == "__main__":
    unittest.main()
