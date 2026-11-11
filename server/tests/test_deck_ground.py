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


if __name__ == "__main__":
    unittest.main()
