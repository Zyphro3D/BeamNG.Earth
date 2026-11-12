"""The terrain held under bridge decks (roadbridges.ground_under_decks, blockworker.under_decks)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import blockworker  # noqa: E402
import geodesy  # noqa: E402
import roadbridges as rb  # noqa: E402

# a 100 m deck along x at 10 m, its road 8 m wide
DECK = [(np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]), np.array([10.0, 10.0, 10.0]), 8.0)]
UNDER = 10.0 - rb.DECK_GROUND_UNDER_M


def capped(x, y, z):
    return rb.ground_under_decks(np.array([x], float), np.array([y], float), np.array([z], float), DECK)[0]


class GroundUnderDecksTest(unittest.TestCase):
    def test_ground_over_the_road_is_held_under_it(self):
        self.assertAlmostEqual(capped(30.0, 0.0, 12.5), UNDER)

    def test_at_a_side_and_under_the_slab_overhang(self):
        self.assertAlmostEqual(capped(30.0, 4.0, 11.0), UNDER)
        self.assertAlmostEqual(capped(30.0, -4.0 - rb.DECK_GROUND_SIDE_M + 0.05, 11.0), UNDER)

    def test_beside_the_deck_left_alone(self):
        self.assertEqual(capped(30.0, 6.0, 11.0), 11.0)

    def test_past_an_end_left_alone(self):
        self.assertEqual(capped(-1.0, 0.0, 11.0), 11.0)
        self.assertEqual(capped(101.0, 0.0, 11.0), 11.0)

    def test_the_deck_s_slope_followed(self):
        slope = [(np.array([[0.0, 0.0], [100.0, 0.0]]), np.array([10.0, 20.0]), 8.0)]
        got = rb.ground_under_decks(np.array([25.0]), np.array([0.0]), np.array([30.0]), slope)
        self.assertAlmostEqual(got[0], 12.5 - rb.DECK_GROUND_UNDER_M)

    def test_lower_ground_unchanged_and_the_same_array(self):
        z = np.array([5.0, 9.0])
        self.assertIs(rb.ground_under_decks(np.array([30.0, 60.0]), np.array([0.0, 1.0]), z, DECK), z)

    def test_the_lower_of_two_decks_over_a_point(self):
        two = DECK + [(np.array([[30.0, -20.0], [30.0, 20.0]]), np.array([16.0, 16.0]), 8.0)]
        got = rb.ground_under_decks(np.array([30.0]), np.array([0.0]), np.array([20.0]), two)
        self.assertAlmostEqual(got[0], UNDER)


def _embankment_over_a_floor(floor=lambda x: 0.0):
    """A deck along x at 10 m from x = 0 to 100; the ground: an embankment at 10 m up to x = 30, sloping down to the
    floor by x = 48 (the elevation's smoothed abutment), the floor (floor(x)) on to x = 100."""
    def ground(xy):
        x = np.asarray(xy, float)[:, 0]
        f = np.array([floor(v) for v in x])
        return np.where(x <= 30.0, 10.0, np.where(x >= 48.0, f, 10.0 + (x - 30.0) / 18.0 * (f - 10.0)))
    deck = [(np.array([[0.0, 0.0], [100.0, 0.0]]), np.array([10.0, 10.0]), 8.0)]
    return deck, ground


def _faced(walls, ground, x, y=0.0):
    z = ground(np.array([[x, y]]))
    return float(rb.ground_under_decks(np.array([x]), np.array([y]), z, walls, 0.0, rb.ABUT_SIDE_M, rb.ABUT_FALL_M)[0])


class AbutmentWallTest(unittest.TestCase):
    def test_the_slope_under_the_deck_cut_to_a_face(self):
        deck, ground = _embankment_over_a_floor()
        walls = rb.abutment_walls(deck, ground)
        self.assertEqual(len(walls), 1)
        self.assertAlmostEqual(_faced(walls, ground, 29.0), 10.0)           # (the embankment stays)
        self.assertLess(_faced(walls, ground, 33.5), 10.0 - 3.5 * 10 / 18 - 2.0)  # (steeper than the slope)
        for x in (35.0, 40.0, 45.0):                                        # (the slope gone: the floor)
            self.assertAlmostEqual(_faced(walls, ground, x), rb.ABUT_FOOT_M, places=6)
        self.assertAlmostEqual(_faced(walls, ground, 70.0), 0.0)            # (the floor itself unchanged)

    def test_across_the_deck_and_a_bit_past_its_side(self):
        deck, ground = _embankment_over_a_floor()
        walls = rb.abutment_walls(deck, ground)
        self.assertAlmostEqual(_faced(walls, ground, 40.0, 4.0 + rb.ABUT_SIDE_M - 0.1), rb.ABUT_FOOT_M, places=6)
        here = float(ground(np.array([[40.0, 0.0]]))[0])
        self.assertAlmostEqual(_faced(walls, ground, 40.0, 4.0 + rb.ABUT_SIDE_M + 1.0), rb.ABUT_FOOT_M + 1.0)
        self.assertAlmostEqual(_faced(walls, ground, 40.0, 4.0 + rb.ABUT_SIDE_M + rb.ABUT_FALL_M + 0.5), here)

    def test_a_road_on_the_ground_beside_the_deck_is_not_dug(self):
        # (a link 6 m beside the deck, on the embankment: inside the face's cut and its 45 degree side)
        deck, ground = _embankment_over_a_floor()
        walls = rb.abutment_walls(deck, ground)
        xs = np.linspace(20.0, 60.0, 81)
        for y, on_road in ((6.0, True), (8.9, True), (11.0, False)):
            x, yy = xs, np.full(len(xs), y)
            z = np.full(len(xs), 10.0)                              # (the link's graded ground: level)
            cut = rb.ground_under_decks(x, yy, z, walls, 0.0, rb.ABUT_SIDE_M, rb.ABUT_FALL_M)
            self.assertLess(float(cut.min()), 9.0)                  # (the face reaches it)
            link = [(np.array([[0.0, 6.0], [100.0, 6.0]]), 2.5)]
            got = rb.keep_roads(x, yy, z, cut, link)
            if on_road:
                np.testing.assert_array_equal(got, z)
            else:
                self.assertIs(got, cut)

    def test_the_face_stands_past_the_end_of_the_road_onto_the_deck(self):
        deck, ground = _embankment_over_a_floor()
        walls = rb.abutment_walls(deck, ground)
        x, y = np.linspace(20.0, 60.0, 81), np.zeros(81)
        z = ground(np.stack([x, y], 1))
        cut = rb.ground_under_decks(x, y, z, walls, 0.0, rb.ABUT_SIDE_M, rb.ABUT_FALL_M)
        own = [(np.array([[-50.0, 0.0], [30.0, 0.0]]), 4.0)]        # (its own road: ends where the deck starts)
        got = rb.keep_roads(x, y, z, cut, own)
        np.testing.assert_array_equal(got[x > 30.0], cut[x > 30.0])
        self.assertIs(rb.keep_roads(x, y, z, cut, []), cut)

    def test_not_over_a_valley(self):
        # (a viaduct's valley: no flat floor under the span, its side stays)
        deck, ground = _embankment_over_a_floor(lambda x: 0.0 if x < 48.0 else -0.3 * (x - 48.0))
        self.assertEqual(rb.abutment_walls(deck, ground), [])

    def test_not_where_the_deck_never_stands_on_its_ground(self):
        deck, ground = _embankment_over_a_floor()
        high = [(deck[0][0], np.array([20.0, 20.0]), 8.0)]
        self.assertEqual(rb.abutment_walls(high, ground), [])

    def test_off_in_the_block(self):
        orig = rb.ABUT_WALL_ON
        rb.ABUT_WALL_ON = False
        try:
            deck, ground = _embankment_over_a_floor()
            self.assertEqual(blockworker._abutment_walls(geodesy.EnuFrame(45.0, 5.0), lambda la, lo: la * 0, deck), [])
        finally:
            rb.ABUT_WALL_ON = orig


class UnderDecksGeoTest(unittest.TestCase):
    def test_heights_at_lat_lon_held_under_a_deck_in_the_frame(self):
        frame = geodesy.EnuFrame(45.0, 5.0)
        la, lo, _ = frame.enu_to_geodetic(np.array([30.0, 30.0]), np.array([0.0, 30.0]), 0.0)
        la, lo = np.asarray(la), np.asarray(lo)
        u0 = np.asarray(frame.geodetic_to_enu(la, lo, 0.0)[2])
        z = np.array([12.5, 12.5]) - u0                     # (heights whose frame up is 12.5 m)
        got = blockworker.under_decks(frame, DECK, la, lo, z)
        up = got + u0
        self.assertAlmostEqual(up[0], UNDER, places=3)      # under the deck
        self.assertAlmostEqual(up[1], 12.5, places=6)       # 30 m beside it

    def test_no_decks_returns_z_itself(self):
        z = np.array([1.0])
        self.assertIs(blockworker.under_decks(geodesy.EnuFrame(45.0, 5.0), [], np.array([45.0]), np.array([5.0]), z), z)



class LightChainTest(unittest.TestCase):
    """A light store (a terrain block's) gives the decks the full one gives, without piers or trusses."""

    def parts(self, light):
        from test_approaches import flat_elev
        from test_bridge_scenarios import scenario
        from test_roads import FRAME
        import roads
        f = scenario([[(0, -300), (0, 300)]], ["primary"])
        bp = roads.BridgeProfiles(FRAME, lambda key: f, flat_elev)
        bp.light = light
        return {fid: (np.asarray(xy)[:, :2], np.asarray(z)) for fid, _sp, xy, z, _l in
                bp.parts([f], (-400, -400, 400, 400))}, bp

    def test_same_decks_no_structure(self):
        full, fbp = self.parts(False)
        light, lbp = self.parts(True)
        self.assertTrue(full)
        self.assertEqual(sorted(full), sorted(light))
        for fid in full:
            np.testing.assert_array_equal(full[fid][0], light[fid][0])
            np.testing.assert_array_equal(full[fid][1], light[fid][1])
        self.assertFalse(lbp.trusses)
        self.assertFalse([k for k in lbp.piers if lbp.piers[k]])


if __name__ == "__main__":
    unittest.main()
