"""IGN BD TOPO road widths joined to the roads layer (ign_largeur.py)."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ign_largeur as IL  # noqa: E402
import mapcells  # noqa: E402
import mapgeom  # noqa: E402


def road_feats(lines, osm_width=None):
    class F:
        ids = np.array([f"r{i}" for i in range(len(lines))])
        kind = np.full(len(lines), mapgeom.LINE)
        attrs = {} if osm_width is None else {"osm_width": np.array(osm_width)}
        rules = {}

        def __len__(self):
            return len(lines)

        def parts(self, i, xy=None):
            return [[np.array(lines[i], float)]]
    return F()


EAST = [[1.40, 43.60], [1.41, 43.60]]
NORTH = [[1.405, 43.595], [1.405, 43.605]]


class JoinTest(unittest.TestCase):
    def test_a_road_takes_the_width_of_the_segment_along_it(self):
        f = road_feats([EAST])
        self.assertEqual(IL.join(f, [[[[1.40, 43.60002], [1.41, 43.60002]], 7.5]]), 1)
        self.assertEqual(float(f.attrs["osm_width"][0]), 7.5)

    def test_a_crossing_segment_gives_nothing(self):
        f = road_feats([EAST])
        self.assertEqual(IL.join(f, [[NORTH, 7.0]]), 0)
        self.assertEqual(str(f.attrs["osm_width"][0]), "")

    def test_a_segment_too_far_gives_nothing(self):
        f = road_feats([EAST])
        self.assertEqual(IL.join(f, [[[[1.40, 43.6005], [1.41, 43.6005]], 5.5]]), 0)

    def test_a_mapped_width_wins(self):
        f = road_feats([EAST], osm_width=["9"])
        self.assertEqual(IL.join(f, [[EAST, 5.5]]), 0)
        self.assertEqual(str(f.attrs["osm_width"][0]), "9")

    def test_the_median_of_the_pieces(self):
        f = road_feats([EAST])
        rows = [[[[1.40, 43.60], [1.408, 43.60]], 6.0], [[[1.408, 43.60], [1.41, 43.60]], 3.0]]
        IL.join(f, rows)
        self.assertEqual(float(f.attrs["osm_width"][0]), 6.0)

    def test_a_point_road_is_left_alone(self):
        f = road_feats([EAST])
        f.kind = np.full(1, mapgeom.POINT)
        self.assertEqual(IL.join(f, [[EAST, 5.0]]), 0)


class TrustTest(unittest.TestCase):
    def with_class(self, cls):
        f = road_feats([EAST])
        f.attrs = {"class": np.array([cls])}
        return f

    def test_a_small_street_at_the_default_keeps_its_class_width(self):
        f = self.with_class("residential")
        self.assertEqual(IL.join(f, [[EAST, 4.0]]), 0)

    def test_a_small_street_above_the_default_takes_it(self):
        f = self.with_class("residential")
        self.assertEqual(IL.join(f, [[EAST, 6.0]]), 1)

    def test_a_motorway_keeps_its_own(self):
        f = self.with_class("motorway")
        self.assertEqual(IL.join(f, [[EAST, 8.0]]), 0)

    def test_a_two_way_through_road_is_never_narrower_than_two_lanes(self):
        f = self.with_class("secondary")
        self.assertEqual(IL.join(f, [[EAST, 5.0]]), 1)
        self.assertEqual(float(f.attrs["osm_width"][0]), IL.THROUGH_TWO_WAY_MIN_M)

    def test_a_one_way_through_road_keeps_a_lane_and_room(self):
        f = self.with_class("secondary")
        f.rules = {"oneway": SimpleNamespace(feature=np.array([0]), major=lambda i: 0)}
        IL.join(f, [[EAST, 3.0]])
        self.assertEqual(float(f.attrs["osm_width"][0]), IL.THROUGH_ONE_WAY_MIN_M)

    def test_a_wide_through_road_keeps_its_width(self):
        f = self.with_class("primary")
        IL.join(f, [[EAST, 10.5]])
        self.assertEqual(float(f.attrs["osm_width"][0]), 10.5)


class ParseTest(unittest.TestCase):
    def feature(self, **p):
        return {"geometry": {"type": "LineString", "coordinates": [[1.0, 43.0, 100.0], [1.1, 43.0, 100.0]]}, "properties": p}

    def test_no_width_or_fictive_or_absurd_is_dropped(self):
        got = IL.parse([self.feature(largeur_de_chaussee=None), self.feature(largeur_de_chaussee=6.0, fictif=True),
                        self.feature(largeur_de_chaussee=0.0), self.feature(largeur_de_chaussee=99.0)])
        self.assertEqual(got, [])

    def test_a_width_is_kept_in_two_dimensions(self):
        self.assertEqual(IL.parse([self.feature(largeur_de_chaussee=7, fictif=False)]), [[[[1.0, 43.0], [1.1, 43.0]], 7.0]])


class Fetcher:
    def __init__(self, features):
        self.features, self.calls = features, 0

    def get(self, url, kind=None, lane=None):
        self.calls += 1
        return SimpleNamespace(body=json.dumps({"features": self.features}).encode())


class ProviderTest(unittest.TestCase):
    key = mapcells.CellKey(14, 8255, 5975)

    def test_once_from_the_network_then_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            net = Fetcher([])
            self.assertEqual(IL.Provider(tmp, net).rows(self.key), [])
            self.assertEqual(IL.Provider(tmp, net).rows(self.key), [])
            self.assertEqual(net.calls, 1)

    def test_a_failed_fetch_leaves_the_roads_alone(self):
        class Down:
            def get(self, *a, **k):
                raise OSError("no network")
        f = road_feats([EAST])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(IL.Provider(tmp, Down()).enrich(self.key, f), 0)
        self.assertNotIn("osm_width", f.attrs)


if __name__ == "__main__":
    unittest.main()
