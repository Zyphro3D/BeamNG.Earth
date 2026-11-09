"""A sidewalk slab is cut where it lies on a carriageway (roads.walks_off_roads)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import roads  # noqa: E402


def run(y0=6.0, x0=-50.0, x1=50.0, **kw):
    return {"pts": np.array([(x0, y0, 0.0), (x1, y0, 0.0)]), "side": 1, "w": 2.5, "fid": "a", "j": (True, True),
            "open": (True, True), **kw}


def crossing(x=0.0, half=3.0):
    return (np.array([(x, -40.0), (x, 40.0)]), half)


class WalksOffRoadsTest(unittest.TestCase):
    def test_a_run_clear_of_every_road_is_the_same_object(self):
        r = run()
        self.assertIs(roads.walks_off_roads([r], [(np.array([(-50.0, 0.0), (50.0, 0.0)]), 3.0)])[0], r)

    def test_a_road_crossing_between_two_points_splits_the_run(self):
        got = roads.walks_off_roads([run()], [crossing()])
        self.assertEqual(len(got), 2)
        self.assertLess(got[0]["pts"][:, 0].max(), -2.5)
        self.assertGreater(got[1]["pts"][:, 0].min(), 2.5)

    def test_only_the_outer_ends_stay_joined(self):
        a, b = roads.walks_off_roads([run()], [crossing()])
        self.assertEqual(a["j"], (True, False))
        self.assertEqual(b["j"], (False, True))

    def test_a_slab_whose_centre_is_off_the_road_is_still_cut(self):
        beside = (np.array([(-50.0, 9.5), (50.0, 9.5)]), 3.0)      # the slab's far edge on it, its centre line not
        self.assertEqual(roads.walks_off_roads([run()], [beside]), [])

    def test_a_median_band_is_left_alone(self):
        r = run(median=True)
        self.assertIs(roads.walks_off_roads([r], [crossing()])[0], r)

    def test_a_scrap_between_two_roads_is_dropped(self):
        got = roads.walks_off_roads([run(x0=-5.0, x1=5.0)], [crossing(-5.0, 4.0), crossing(5.0, 4.0)])
        self.assertEqual(got, [])


if __name__ == "__main__":
    unittest.main()
