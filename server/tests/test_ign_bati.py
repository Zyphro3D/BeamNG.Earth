"""IGN BD TOPO building attributes joined to the buildings layer (ign_bati.py)."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ign_bati  # noqa: E402
import mapcells  # noqa: E402


def feature(x, y, **props):
    d = .0005
    return {"geometry": {"type": "Polygon", "coordinates": [[[x, y], [x + d, y], [x + d, y + d], [x, y + d], [x, y]]]},
            "properties": props}


def buildings(points):
    n = len(points)
    cx = np.array([p[0] for p in points], float)
    cy = np.array([p[1] for p in points], float)
    attrs = {"cx": cx, "cy": cy, "height": np.full(n, np.nan, np.float32), "num_floors": np.full(n, np.nan, np.float32),
             "roof_height": np.full(n, np.nan, np.float32)}
    for k in ign_bati.TEXT_COLUMNS:
        attrs[k] = np.array([""] * n, dtype="<U1")
    return SimpleNamespace(attrs=attrs, ids=np.array([f"b{i}" for i in range(n)]), __len__=lambda: n)


class Feats:
    def __init__(self, points):
        b = buildings(points)
        self.attrs, self.ids, self._n = b.attrs, b.ids, len(points)

    def __len__(self):
        return self._n


class Fetcher:
    def __init__(self, features):
        self.features, self.calls = features, 0

    def get(self, url, kind=None, lane=None):
        self.calls += 1
        return SimpleNamespace(body=json.dumps({"features": self.features}).encode())


class AttributesTest(unittest.TestCase):
    def test_a_pitched_roof_adds_its_rise_to_the_eaves(self):
        got = ign_bati.attributes({"hauteur": 6.0, "altitude_minimale_toit": 150.0, "altitude_maximale_toit": 153.0,
                                   "nombre_d_etages": 2, "materiaux_de_la_toiture": "10"}, "x")
        self.assertEqual(got["height"], 9.0)
        self.assertEqual(got["roof_height"], 3.0)
        self.assertEqual(got["num_floors"], 2.0)
        self.assertIn(got["roof_shape"], ("hipped", "gabled"))
        self.assertEqual(got["roof_material"], "tile")

    def test_a_small_rise_is_a_flat_roof(self):
        got = ign_bati.attributes({"hauteur": 12.0, "altitude_minimale_toit": 150.0, "altitude_maximale_toit": 150.2}, "x")
        self.assertEqual(got["roof_shape"], "flat")
        self.assertEqual(got["height"], 12.0)
        self.assertNotIn("roof_height", got)

    def test_brick_is_rendered_on_a_low_house_and_shown_on_a_block(self):
        house = ign_bati.attributes({"hauteur": 5.0, "nombre_d_etages": 1, "materiaux_des_murs": "40"}, "x")
        block = ign_bati.attributes({"hauteur": 15.0, "nombre_d_etages": 5, "materiaux_des_murs": "40"}, "x")
        self.assertEqual(house["facade_material"], "plaster")
        self.assertEqual(block["facade_material"], "brick")

    def test_nothing_known_gives_nothing(self):
        self.assertEqual(ign_bati.attributes({"materiaux_des_murs": "00", "materiaux_de_la_toiture": None}, "x"), {})

    def test_the_roof_shape_does_not_depend_on_the_run(self):
        props = {"altitude_minimale_toit": 1.0, "altitude_maximale_toit": 4.0}
        self.assertEqual(ign_bati.attributes(props, "same")["roof_shape"], ign_bati.attributes(props, "same")["roof_shape"])


class JoinTest(unittest.TestCase):
    def test_only_a_building_inside_takes_the_attributes(self):
        rows = ign_bati.parse([feature(1.4, 43.6, hauteur=7.0, nombre_d_etages=2)])
        f = Feats([(1.40025, 43.60025), (1.41, 43.61)])
        self.assertEqual(ign_bati.join(f, rows), 1)
        self.assertEqual(f.attrs["height"][0], 7.0)
        self.assertTrue(np.isnan(f.attrs["height"][1]))
        self.assertEqual(f.attrs["height_src"][0], "IGN BD TOPO")
        self.assertEqual(f.attrs["height_src"][1], "")

    def test_text_columns_grow_for_the_new_values(self):
        rows = ign_bati.parse([feature(1.4, 43.6, altitude_minimale_toit=1.0, altitude_maximale_toit=1.1,
                                       materiaux_de_la_toiture="20")])
        f = Feats([(1.40025, 43.60025)])
        ign_bati.join(f, rows)
        self.assertEqual(f.attrs["roof_shape"][0], "flat")
        self.assertEqual(f.attrs["roof_material"][0], "slate")


class ProviderTest(unittest.TestCase):
    def key(self):
        return mapcells.CellKey(14, 8255, 5975)

    def test_a_cell_is_fetched_once_then_read_from_disk(self):
        lon0, lat0, _, _ = self.key().bounds()
        fetcher = Fetcher([feature(lon0 + .001, lat0 + .001, hauteur=4.0)])
        with tempfile.TemporaryDirectory() as tmp:
            first = ign_bati.Provider(tmp, fetcher).rows(self.key())
            again = ign_bati.Provider(tmp, fetcher).rows(self.key())
        self.assertEqual(fetcher.calls, 1)
        self.assertEqual(first, again)

    def test_a_failed_fetch_leaves_the_buildings_alone(self):
        class Down:
            def get(self, *a, **k):
                raise OSError("no network")
        lon0, lat0, _, _ = self.key().bounds()
        f = Feats([(lon0 + .001, lat0 + .001)])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ign_bati.Provider(tmp, Down()).enrich(self.key(), f), 0)
        self.assertTrue(np.isnan(f.attrs["height"][0]))

    def test_outside_france_nothing_is_asked(self):
        fetcher = Fetcher([])
        far = mapcells.CellKey(14, 2000, 6000)
        f = Feats([(0.0, 0.0)])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ign_bati.Provider(tmp, fetcher).enrich(far, f), 0)
        self.assertEqual(fetcher.calls, 0)


if __name__ == "__main__":
    unittest.main()
