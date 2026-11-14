"""prelaunch.py: the window before the game, against a fake world server (the real framing, no game, no network)."""
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prelaunch

HEADER = struct.Struct("<I")


class FakeServer:
    """A world server's socket: answers like earthd's connmap (geocode, area_plan, area_add, area_cmd, map_tiles,
    prelaunch_status) and keeps what it was sent."""

    def __init__(self, tile_root):
        self.tile_root = tile_root
        self.got = []
        self.areas = []
        self.lsock = socket.socket()
        self.lsock.bind(("127.0.0.1", 0))
        self.lsock.listen(1)
        self.port = self.lsock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _send(self, msg):
        data = json.dumps(msg).encode()
        self.conn.sendall(HEADER.pack(len(data)) + data)

    def _areas(self, result=None):
        m = {"type": "areas", "areas": list(self.areas), "fast": "", "pinnedBytes": 5}
        if result is not None:
            m["result"] = result
        return m

    def _serve(self):
        self.conn, _ = self.lsock.accept()
        f = self.conn.makefile("rb")
        while True:
            head = f.read(4)
            if len(head) < 4:
                return
            msg = json.loads(f.read(HEADER.unpack(head)[0]))
            self.got.append(msg)
            kind = msg["type"]
            if kind == "geocode":
                self._send({"type": "geocode_result", "id": msg["id"],
                            "results": [{"name": "Castres", "detail": "Tarn", "lat": 43.6, "lon": 2.24}]})
            elif kind == "route":           # (connmap.route logs int(id): the ids must be numbers)
                self._send({"type": "route_result", "id": int(msg["id"]), "points": [[43.6, 1.44], [42.5, 1.73]],
                            "dist": 150000.0, "dur": 7200.0})
            elif kind == "road_spot":
                self._send({"type": "road_spot_result", "id": msg["id"], "lat": msg["lat"] + 0.001, "lon": msg["lon"]})
            elif kind == "area_plan":
                self._send({"type": "area_plan_result", "id": msg["id"], "ok": True, "km2": 16.0,
                            "downloadBytes": 1e8, "diskBytes": 2e8, "seconds": 300})
            elif kind == "area_add":
                self.areas.append({"id": "a1", "name": msg["name"], "s": msg["s"], "w": msg["w"], "n": msg["n"],
                                   "e": msg["e"], "state": "queued", "done": 0, "total": 10, "msg": "",
                                   "mode": msg["mode"]})
                self._send(self._areas({"id": msg["id"], "ok": True, "area": "a1"}))
            elif kind == "area_cmd":
                self._send(self._areas({"id": msg["id"], "ok": msg["area"] == "a1"}))
            elif kind == "prelaunch_status":
                self._send({"type": "cache", "usedBytes": 10, "capBytes": 20, "pinnedBytes": 5})
                self._send(self._areas())
            elif kind == "map_tiles":
                for z, x, y, layer in msg["tiles"]:
                    path = os.path.join(self.tile_root, str(z), str(x), f"{y}.jpg")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as fh:
                        fh.write(b"JPEG")
                self._send({"type": "map_tiles_ready", "tiles": [[z, x, y, True, layer] for z, x, y, layer in msg["tiles"]]})


class PrelaunchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tiles = os.path.join(self.tmp.name, "temp", "beamngEarth", "maptiles")
        self.fake = FakeServer(tiles)
        self.url, self.launch, self.httpd, self.bridge = prelaunch.start(self.tmp.name, self.fake.port, page_port=0)
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, obj, ctype="application/json"):
        req = urllib.request.Request(self.url + path.lstrip("/"), data=json.dumps(obj).encode(),
                                     headers={"Content-Type": ctype}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def get(self, path):
        with urllib.request.urlopen(self.url + path.lstrip("/"), timeout=10) as r:
            return r.read()

    def test_page_and_status(self):
        self.assertIn(b"Lancer le jeu", self.get("/"))
        st = json.loads(self.get("/api/status"))
        self.assertEqual(st["cache"]["capBytes"], 20)
        self.assertEqual(st["areas"]["areas"], [])
        self.assertFalse(st["launched"])

    def test_search_estimate_add_delete(self):
        r = self.post("/api/ask", {"type": "geocode", "q": "Castres"})
        self.assertEqual(r["results"][0]["name"], "Castres")
        r = self.post("/api/ask", {"type": "route", "lat0": 43.6, "lon0": 1.44, "lat1": 42.5, "lon1": 1.73})
        self.assertEqual(r["dist"], 150000.0)
        self.assertEqual([m for m in self.fake.got if m["type"] == "route"][0]["download"], "off")
        box = {"s": 43.58, "w": 2.21, "n": 43.62, "e": 2.27}
        r = self.post("/api/ask", dict(box, type="area_plan", mode="world"))
        self.assertTrue(r["ok"])
        self.assertEqual(r["km2"], 16.0)
        places = os.path.join(self.tmp.name, "settings", "beamngEarth_places.json")
        os.makedirs(os.path.dirname(places))
        with open(places, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "nextId": 3, "places": [{"id": 2, "name": "Stelvio", "lat": 46.5, "lon": 10.4}],
                       "recent": [{"lat": 1.0, "lon": 2.0, "kind": "start"}]}, fh)
        r = self.post("/api/ask", dict(box, type="area_add", mode="world", name="Castres", fast=True,
                                       start=[43.6, 2.24]))
        self.assertTrue(r["ok"])
        with open(places, encoding="utf-8") as fh:
            got = json.load(fh)
        self.assertEqual(got["places"][0]["name"], "Stelvio")
        self.assertEqual(got["places"][1], {"id": 3, "name": "Castres (préparé)", "lat": 43.601, "lon": 2.24})
        self.assertEqual(got["nextId"], 4)
        self.assertEqual(got["recent"], [{"lat": 1.0, "lon": 2.0, "kind": "start"}])
        self.assertNotIn("start", [m for m in self.fake.got if m["type"] == "area_add"][0])
        self.assertEqual(r["areas"]["areas"][0]["name"], "Castres")
        sent = [m for m in self.fake.got if m["type"] == "area_add"][0]
        self.assertIs(sent["fast"], True)
        self.assertEqual(sent["mode"], "world")
        r = self.post("/api/ask", {"type": "area_cmd", "area": "a1", "op": "delete", "name": "Castres"})
        self.assertTrue(r["ok"])
        with open(places, encoding="utf-8") as fh:
            self.assertEqual([p["name"] for p in json.load(fh)["places"]], ["Stelvio"])

    def test_requests_are_checked(self):
        for bad in ({"type": "hello"}, {"type": "road_spot", "lat": 1, "lon": 2}, {"type": "prepare_anchor"}, {"type": "area_cmd", "area": "a1", "op": "rm"},
                    {"type": "area_plan", "s": 99, "w": 0, "n": 1, "e": 1}):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.post("/api/ask", bad)
            self.assertEqual(cm.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as cm:      # (a form post from another site)
            self.post("/api/launch", {}, ctype="text/plain")
        self.assertEqual(cm.exception.code, 415)
        self.assertFalse(self.launch.is_set())
        self.assertFalse(any(m["type"] in ("hello", "prepare_anchor") for m in self.fake.got))

    def test_tiles_come_from_the_server(self):
        self.assertEqual(self.get("/tiles/sat/12/2070/1490"), b"JPEG")
        asked = [m for m in self.fake.got if m["type"] == "map_tiles"]
        self.assertEqual(asked[0]["tiles"], [[12, 2070, 1490, "sat"]])
        self.assertEqual(self.get("/tiles/sat/12/2070/1490"), b"JPEG")       # (then from the file)
        self.assertEqual(len([m for m in self.fake.got if m["type"] == "map_tiles"]), 1)
        with self.assertRaises(urllib.error.HTTPError):
            self.get("/tiles/sat/40/0/0")

    def test_launch(self):
        self.assertTrue(self.post("/api/launch", {})["ok"])
        self.assertTrue(self.launch.is_set())


if __name__ == "__main__":
    unittest.main()


try:
    import tkinter
    tkinter.Tk().destroy()
    HAVE_TK = True
except Exception:       # (no tkinter, or no display)
    HAVE_TK = False


@unittest.skipUnless(HAVE_TK, "tkinter with a display")
class WindowTest(unittest.TestCase):
    """prelaunch_window.Window against the fake server: search, estimate, prepare, list, delete, start."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeServer(os.path.join(self.tmp.name, "maptiles"))
        self.url, self.launch, self.httpd, self.bridge = prelaunch.start(self.tmp.name, self.fake.port, page_port=0)
        self.addCleanup(self.httpd.shutdown)
        import prelaunch_window
        self.w = prelaunch_window.Window(self.bridge, self.launch, self.tmp.name, self.url)
        self.w.root.after(prelaunch_window.POLL_MS, self.w._poll)

    def pump(self, until, s=10.0):
        import time
        end = time.monotonic() + s
        while time.monotonic() < end:
            self.w.root.update()
            if until():
                return True
            time.sleep(0.02)
        return False

    def test_town_to_game(self):
        w = self.w
        w.q_town.insert(0, "Castres")
        w.search()
        self.assertTrue(self.pump(lambda: w.pick is not None and w.pick.get("mode")))      # (one result: chosen)
        self.assertIn("16 km²", w.est.cget("text"))
        self.assertEqual(str(w.b_now.cget("state")), "normal")
        w.prepare(True)
        self.assertTrue(self.pump(lambda: "ajouté" in w.est.cget("text")))
        w.refresh()
        self.assertTrue(self.pump(lambda: any("Castres (préparé)" == str(c.cget("text"))
                                              for c in w.list.winfo_children() if c.winfo_class() == "TLabel")))
        with open(os.path.join(self.tmp.name, "settings", "beamngEarth_places.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["places"][0]["name"], "Castres (préparé)")
        add = [m for m in self.fake.got if m["type"] == "area_add"][0]
        self.assertEqual((add["mode"], add["fast"], add["name"]), ("world", True, "Castres"))
        w.start_game()
        self.assertTrue(self.launch.is_set())


class PortTest(unittest.TestCase):
    """launcher.port_free / spare_port: the world server's port when Windows keeps 47613 (WSL mirrored networking)."""

    def test_taken_port_is_not_free(self):
        import launcher
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            self.assertFalse(launcher.port_free(s.getsockname()[1]))
        p = launcher.spare_port()
        self.assertTrue(launcher.port_free(p))
