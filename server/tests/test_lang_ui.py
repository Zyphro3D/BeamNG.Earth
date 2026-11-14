"""The mod's screens in French (mod/tests/test_i18n.mjs, when node is installed) and the dictionary against the texts
the screens have (tools/lang_extract.py): every settings name and option label has an entry."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


class LangUiTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node")
    def test_i18n_js(self):
        r = subprocess.run(["node", str(ROOT / "mod" / "tests" / "test_i18n.mjs")], capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_settings_are_all_in_french(self):
        import re
        fr = json.loads((ROOT / "mod" / "beamng_earth" / "lua" / "ge" / "extensions" / "beamngEarth" /
                         "lang_fr.json").read_text(encoding="utf-8"))["exact"]
        store = (ROOT / "mod" / "beamng_earth" / "ui" / "ui-vue" / "mods" / "beamngEarth" /
                 "earthSettingsStore.js").read_text(encoding="utf-8")
        names = set(re.findall(r'\bname:\s*"([^"]+)"', store))
        same = {"Add-ons", "BeamNG.Earth", "Interface", "Ultra"}    # (the same word in French)
        self.assertEqual(sorted(n for n in names if n not in fr and n not in same), [])


if __name__ == "__main__":
    unittest.main()
