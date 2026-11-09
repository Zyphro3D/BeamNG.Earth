"""France: the width and sidewalks of an unmapped street in town (franceroads.adjust_specs)."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import franceroads as fr  # noqa: E402
import roads  # noqa: E402


def adjust(spec, attrs=None):
    feats = SimpleNamespace(rules={}, attrs=attrs or {})
    with mock.patch.object(fr, "ON", True), mock.patch.object(fr, "french_feature", lambda f, i: True), \
            mock.patch.object(roads, "is_ring", lambda f, i: False):
        return fr.adjust_specs(feats, [spec])[0]


def street(cls="residential", width=10.6, urban=True, oneway=False, **kw):
    return roads.Spec(width, "paved", urban, cls, oneway, parking=True, walk=2.5, **kw)


class TownStreetTest(unittest.TestCase):
    def test_an_unmapped_street_loses_its_invented_parking_lanes(self):
        got = adjust(street())
        self.assertEqual(got.width, fr.URBAN_WIDTH["residential"])
        self.assertFalse(got.parking)

    def test_a_one_way_street_is_one_lane_wide(self):
        self.assertEqual(adjust(street(oneway=True)).width, fr.URBAN_ONE_WAY_M)

    def test_a_mapped_width_stands_and_still_gets_a_narrower_sidewalk(self):
        got = adjust(street(width=8.0), {"osm_width": ["8"]})
        self.assertEqual(got.width, 8.0)
        self.assertEqual(got.walk, fr.URBAN_WALK_M)

    def test_a_mapped_lane_count_keeps_its_width(self):
        self.assertEqual(adjust(street(lanes=4)).width, 10.6)

    def test_a_country_road_is_not_a_town_street(self):
        self.assertEqual(adjust(street(urban=False, width=5.5, cls="unclassified")).walk, 2.5)


class KerbTest(unittest.TestCase):
    def test_no_gutter_along_a_french_kerb(self):
        sp = roads.Spec(6.0, "paved", True, "residential", False, walk=1.6, france=True)
        pts = np.array([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)])
        with mock.patch.object(fr, "ON", True):
            got = roads.extras(sp, pts, 1.4, country="FR")
        self.assertTrue(any(look["m"] == roads.SIDEWALK["m"] for look, _, _ in got))
        self.assertFalse(any(look["m"] == roads.GUTTER["m"] for look, _, _ in got))


class ThroughRoadTest(unittest.TestCase):
    def test_two_lanes_one_way_get_six_metres(self):
        got = adjust(roads.Spec(4.0, "paved", True, "primary", True, lanes=2, walk=1.6), {"osm_width": ["4"]})
        self.assertEqual(got.width, 6.0)

    def test_a_divided_road_keeps_clear_of_its_other_side(self):
        got = adjust(roads.Spec(5.0, "paved", True, "primary", True, lanes=2, walk=1.6, median=6.0), {"osm_width": ["5"]})
        self.assertAlmostEqual(got.width, 5.4)

    def test_a_ramp_is_left_alone(self):
        self.assertEqual(adjust(roads.Spec(4.0, "paved", False, "primary", True, lanes=2, link=True)).width, 4.0)


class OneStreetTest(unittest.TestCase):
    def run_it(self, widths, lengths, names=None, cls="tertiary"):
        specs = [roads.Spec(w, "paved", True, cls, True, france=True, walk=1.6) for w in widths]
        feats = SimpleNamespace(rules={}, attrs={"name": names or ["Rue A"] * len(widths)})
        with mock.patch.object(roads, "is_ring", lambda f, i: False), \
                mock.patch.object(roads, "_lengths", lambda f: np.array(lengths, float)):
            return [s.width for s in fr._one_width_per_street(feats, specs)]

    def test_the_pieces_take_the_longest_ones_width(self):
        self.assertEqual(self.run_it([6.0, 5.0, 6.0], [200.0, 30.0, 150.0]), [6.0, 6.0, 6.0])

    def test_really_different_roads_are_left_alone(self):
        self.assertEqual(self.run_it([5.0, 10.5], [100.0, 100.0]), [5.0, 10.5])

    def test_other_names_are_other_streets(self):
        self.assertEqual(self.run_it([5.0, 6.0], [100.0, 300.0], ["Rue A", "Rue B"]), [5.0, 6.0])

    def test_no_name_is_no_street(self):
        self.assertEqual(self.run_it([5.0, 6.0], [100.0, 300.0], ["", ""]), [5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
