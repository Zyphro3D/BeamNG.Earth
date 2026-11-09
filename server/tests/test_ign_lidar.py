"""IGN LiDAR HD canopy heights (ign_lidar.py)."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ign_lidar  # noqa: E402
import mapcells  # noqa: E402


class Flaky:
    """Answers BLOCK blocks of `value` metres; the listed call numbers (from 0) raise."""

    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), 0

    def get(self, url, kind=None, lane=None):
        n, self.calls = self.calls, self.calls + 1
        if n in self.fail:
            raise OSError("HTTP 400")
        return SimpleNamespace(body=np.full((ign_lidar.BLOCK, ign_lidar.BLOCK), 7.0, "<f4").tobytes())


class MeasuredTest(unittest.TestCase):
    cell = mapcells.CellKey(14, 8254, 5978)

    def run_it(self, fetcher):
        with mock.patch.object(ign_lidar.time, "sleep"):
            return ign_lidar.measured(fetcher, self.cell, ign_lidar.BLOCK * 2)

    def test_a_refused_block_is_asked_again(self):
        out = self.run_it(Flaky(fail=[1]))
        self.assertTrue((out == 7).all())

    def test_a_block_still_refused_does_not_lose_the_others(self):
        out = self.run_it(Flaky(fail=range(ign_lidar.BLOCK_TRIES)))
        self.assertTrue((out[:ign_lidar.BLOCK, :ign_lidar.BLOCK] == ign_lidar.NODATA).all())
        self.assertEqual(int((out == 7).sum()), 3 * ign_lidar.BLOCK ** 2)


if __name__ == "__main__":
    unittest.main()
