"""France: the width and sidewalks of an unmapped street in town (franceroads.adjust_specs)."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
