"""France's asphalt look (roadbase.fr_asphalt_look, tools/build_fr_asphalt.py's materials) and its square texture."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import roads as roadbase  # noqa: E402  (roads.py: look_of, _chunk, _deck_look, _corner_look)


def spec(cls="residential", kind="paved", surf=None, france=True, width=6.0):
    return roadbase.Spec(width, kind, True, cls, False, surf=surf, france=france)


class FrAsphaltTest(unittest.TestCase):
    def test_off_by_default_the_kinds_look(self):
        with mock.patch.object(roadbase, "FR_ASPHALT_ON", False):
            self.assertEqual(roadbase.look_of(spec())["m"], roadbase.LOOK["paved"]["m"])

    def test_wear_by_class(self):
        with mock.patch.object(roadbase, "FR_ASPHALT_ON", True):
            self.assertEqual(roadbase.look_of(spec("motorway"))["m"], "bne_fr_asphalt_fresh")
            self.assertEqual(roadbase.look_of(spec("service"))["m"], "bne_fr_asphalt_worn")
            self.assertEqual(roadbase.look_of(spec("residential"))["m"], "bne_fr_asphalt_city")
            self.assertEqual(roadbase.look_of(spec(france=False))["m"], roadbase.LOOK["paved"]["m"])
            self.assertEqual(roadbase.look_of(spec(surf="concrete")), roadbase.CONCRETE)
            self.assertNotIn("bne_fr", roadbase.look_of(spec(kind="cobble"))["m"])

    def test_square_texels_across(self):
        with mock.patch.object(roadbase, "FR_ASPHALT_ON", True):
            look = roadbase.look_of(spec(width=7.5))
        c = roadbase._chunk(look, 7.5, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)])
        self.assertAlmostEqual(c["tu"], 7.5 / roadbase.FR_ASPHALT_M)
        self.assertNotIn("tu", roadbase._chunk(roadbase.LOOK["paved"], 7.5, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]))

    def test_bridges_and_corners_take_it(self):
        with mock.patch.object(roadbase, "FR_ASPHALT_ON", True):
            self.assertEqual(roadbase._deck_look(spec("motorway"))["m"], "bne_fr_asphalt_fresh")
            import numpy as np
            roads = [("a", spec(), np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), None)]
            self.assertEqual(roadbase._corner_look((5.0, 1.0), roads)["m"], "bne_fr_asphalt_city")
        with mock.patch.object(roadbase, "FR_ASPHALT_ON", False):
            self.assertNotIn("bne_fr", roadbase._deck_look(spec("motorway"))["m"])

    def test_french_bridges_marked_only_when_on(self):
        import franceroads
        from types import SimpleNamespace
        feats = SimpleNamespace(_bnge_french=None)
        specs = [spec(france=False), spec(kind="gravel", france=False), None]
        with mock.patch.object(franceroads, "ON", True), \
                mock.patch.object(franceroads, "french_feature", lambda f, i: True):
            with mock.patch.object(roadbase, "FR_ASPHALT_ON", False):
                self.assertIs(roadbase._fr_asphalt_decks(feats, specs), specs)
            with mock.patch.object(roadbase, "FR_ASPHALT_ON", True):
                got = roadbase._fr_asphalt_decks(feats, specs)
        self.assertTrue(got[0].france)
        self.assertFalse(got[1].france)             # (not paved)
        self.assertIsNone(got[2])

    def test_materials_shipped(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mod", "beamng_earth", "levels",
                            "beamng_earth", "art", "shapes", "beamngEarth", "fr_asphalt")
        import json
        mats = json.load(open(os.path.join(root, "fr_asphalt.materials.json"), encoding="utf-8"))
        for wear in ("fresh", "worn", "city"):
            stage = mats[f"bne_fr_asphalt_{wear}"]["Stages"][0]
            for key in ("baseColorMap", "normalMap", "detailMap", "detailNormalMap"):
                self.assertTrue(os.path.exists(os.path.join(root, os.path.basename(stage[key]))), stage[key])


if __name__ == "__main__":
    unittest.main()
