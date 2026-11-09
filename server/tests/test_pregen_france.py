"""Generating a region ahead of play (tools/pregen_france.py)."""
import os
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mapcells  # noqa: E402
import pregen_france as P  # noqa: E402

KEY = mapcells.CellKey(14, 8254, 5976)
OTHER = mapcells.CellKey(14, 8255, 5976)


def touch(root, *parts):
    path = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")
    return path


class RunTest(unittest.TestCase):
    def test_a_cell_done_is_not_done_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            touch(tmp, "ign_limites", "v1", "14", "8254", "5976.npz")
            self.assertEqual(P.pending([KEY, OTHER], tmp), [OTHER])

    def test_a_failure_is_tried_once_more(self):
        calls = []

        def build(k):
            calls.append(k)
            if calls.count(k) == 1:
                raise OSError("x")
        self.assertEqual(P.run([KEY], build, 1, sink=lambda s: None), [])
        self.assertEqual(len(calls), 2)

    def test_what_keeps_failing_is_reported(self):
        def build(k):
            raise OSError("x")
        self.assertEqual(P.run([KEY], build, 1, sink=lambda s: None), [KEY])


class BoxTest(unittest.TestCase):
    def test_a_centre_gives_a_box_round_it(self):
        s, w, n, e = P.box_of(NS(bbox=None, centre=[43.7, 1.38], km=11.1))
        self.assertAlmostEqual(n - s, 0.2, places=3)
        self.assertGreater(e - w, n - s)

    def test_a_box_the_wrong_way_round_is_refused(self):
        with self.assertRaises(SystemExit):
            P.box_of(NS(bbox="44,1,43,2", centre=None, km=1))


class PackTest(unittest.TestCase):
    def test_a_pack_goes_back_in_the_cache(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            touch(a, "ign_limites", "v1", "14", "8254", "5976.npz")
            touch(a, "ign_limites", "v1", "14", "8254", "5976.npz.claim")
            touch(a, "ign_limites", "v1", "14", "8255", "5976.npz")
            touch(a, "dem", "other.bin")
            zip_path = os.path.join(b, "p.zip")
            self.assertEqual(P.pack(a, zip_path, [KEY]), 1)
            target = os.path.join(b, "cache")
            self.assertEqual(P.unpack(zip_path, target), 1)
            self.assertTrue(os.path.exists(os.path.join(target, "ign_limites", "v1", "14", "8254", "5976.npz")))

    def test_a_pack_cannot_write_outside_its_folders(self):
        with tempfile.TemporaryDirectory() as b:
            zip_path = os.path.join(b, "evil.zip")
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("ign_limites/../../evil.txt", "x")
                z.writestr("dem/evil.bin", "x")
                z.writestr("ign_bati/ok.json.gz", "x")
            target = os.path.join(b, "cache")
            self.assertEqual(P.unpack(zip_path, target), 1)
            self.assertFalse(os.path.exists(os.path.join(b, "evil.txt")))


if __name__ == "__main__":
    unittest.main()
