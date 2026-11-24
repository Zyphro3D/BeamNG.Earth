"""The packaged app's --share-helper mode (launcher.py): server/sharehelper.py in the app's own Python, so a BeamMP
server's machine runs the helper of the players' release without installing Python packages or Rust."""
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launcher


class ShareHelperModeTest(unittest.TestCase):
    def test_share_helper_runs_sharehelper_with_its_options(self):
        seen = {}
        fake = types.ModuleType("sharehelper")
        fake.main = lambda: seen.setdefault("argv", list(sys.argv))
        argv = ["BeamNG.Earth", "--share-helper", "--share-dir", "/srv/BeamNGEarth", "--cache", "/var/cache/h",
                "--procs", "1"]
        with mock.patch.dict(sys.modules, {"sharehelper": fake}), mock.patch.object(sys, "argv", argv), \
                mock.patch.object(launcher, "ensure_ca_certs") as certs, \
                mock.patch.object(launcher, "tee_log") as tee, mock.patch.object(launcher, "run_server") as server:
            launcher.main()
        self.assertEqual(seen["argv"], ["sharehelper", "--share-dir", "/srv/BeamNGEarth", "--cache", "/var/cache/h",
                                        "--procs", "1"])
        certs.assert_called_once()
        tee.assert_not_called()         # (its log is the service's journal)
        server.assert_not_called()

    def test_the_real_module_has_the_entry_point(self):
        import sharehelper
        self.assertTrue(callable(sharehelper.main))


if __name__ == "__main__":
    unittest.main()
