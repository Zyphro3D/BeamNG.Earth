"""The window before the game (the user, 2026-11-13: "a window before the game starts, to choose where to go, see what
is loaded and delete it from there too"): a page in the browser, served by the launcher once the world server runs,
that prepares places ahead (the server's saved areas, areas.py: a box, a town or an A -> B route, its map data
downloaded and its world built) and lists them with the cache's use, then starts the game.

Nothing new is built here: the page speaks to the world server over its own connection (protocol: a 4-byte
little-endian length + JSON), as the in-game map does: map_tiles, geocode, route, area_plan, area_add, area_cmd and
prelaunch_status. That connection never says hello, so it takes no stream lease and the game's own connection is left
alone. The map's tiles are the server's (maptiles.py, written under <user>/temp/beamngEarth/maptiles): no tile host of
its own.

    run(user_dir, port)  serves http://127.0.0.1:PAGE_PORT/, opens it, and returns when "Lancer le jeu" is pressed
                         (or Enter in the console); the page and the bridge keep running for the game's session.
"""
import json
import os
import socket
import struct
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE_PORT = 47616                # (47614: the companion relay, 47615: its feed, 47620-47622: helpers)
HEADER = struct.Struct("<I")
MAX_FRAME = 16 * 1024 * 1024
ASK_S = 60.0                    # an answer from the server waited for at most (a route or an estimate: seconds)
TILE_S = 20.0                   # a map tile waited for at most
TILE_BATCH_S = 0.05             # the tiles the page asks for within this long go to the server as one request
LAYERS = {"sat": "jpg", "map": "jpg", "labels": "png"}      # (maptiles.LAYERS: the extensions)
MAX_Z = 18                      # (maptiles.MAX_Z)
ASKS = {"geocode": "geocode_result", "route": "route_result", "area_plan": "area_plan_result",
        "area_add": "areas", "area_cmd": "areas", "road_spot": "road_spot_result"}           # what the page may ask, and the answer's type


def encode(msg):
    payload = json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return HEADER.pack(len(payload)) + payload


class Bridge:
    """One connection to the world server: requests by id, the last 'areas' and 'cache' messages, map tiles."""

    def __init__(self, port, tile_root, connect=socket.create_connection):
        self.sock = connect(("127.0.0.1", port), timeout=10)
        self.sock.settimeout(None)
        self.tile_root = tile_root
        self.wlock = threading.Lock()
        self.cond = threading.Condition()
        self.answers = {}           # id -> message
        self.areas = None           # the last 'areas' message
        self.cache = None           # the last 'cache' message
        self.tiles = {}             # (layer, z, x, y) -> True / False (ready, failed) or None (asked)
        self.seq = 0
        self.closed = False
        self._batch = None
        threading.Thread(target=self._read, name="prelaunch-read", daemon=True).start()

    def send(self, msg):
        with self.wlock:
            self.sock.sendall(encode(msg))

    def _read(self):
        f = self.sock.makefile("rb")
        try:
            while True:
                head = f.read(HEADER.size)
                if len(head) < HEADER.size:
                    break
                (n,) = HEADER.unpack(head)
                if n > MAX_FRAME:
                    break
                body = f.read(n)
                if len(body) < n:
                    break
                try:
                    msg = json.loads(body.decode("utf-8"))
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    self._on(msg)
        except OSError:
            pass
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def _on(self, msg):
        kind = msg.get("type")
        with self.cond:
            if kind == "areas":
                self.areas = {k: v for k, v in msg.items() if k not in ("type", "result")}
            elif kind == "cache":
                self.cache = {k: v for k, v in msg.items() if k != "type"}
            elif kind == "map_tiles_ready":
                for t in msg.get("tiles") or []:
                    if isinstance(t, list) and len(t) >= 5:
                        self.tiles[(t[4], int(t[0]), int(t[1]), int(t[2]))] = bool(t[3])
            ident = msg.get("id")
            if ident is None and isinstance(msg.get("result"), dict):
                ident = msg["result"].get("id")
            if isinstance(ident, int) and not isinstance(ident, bool):     # (numbers: connmap.route logs int(id))
                self.answers[ident] = msg
            self.cond.notify_all()

    def ask(self, msg, timeout=ASK_S):
        """msg (one of ASKS) sent with an id of ours; its answer, or {"error"} when none comes."""
        kind = msg.get("type")
        if kind not in ASKS:
            return {"error": f"unknown request {kind!r}"}
        with self.cond:
            self.seq += 1
            ident = self.seq
        self.send(dict(msg, id=ident))
        end = time.monotonic() + timeout
        with self.cond:
            while ident not in self.answers and not self.closed:
                left = end - time.monotonic()
                if left <= 0:
                    return {"error": "le serveur du monde ne répond pas"}
                self.cond.wait(left)
            if ident not in self.answers:
                return {"error": "le serveur du monde s'est arrêté"}
            out = self.answers.pop(ident)
        if out.get("type") == "areas":       # (area_add / area_cmd: the result, and the list as it is now)
            res = out.get("result") or {}
            return {"ok": bool(res.get("ok")), "error": res.get("error"), "areas": self.areas}
        return out

    def status(self, timeout=5.0):
        """The saved areas and the cache's use (prelaunch_status: earthd sends both)."""
        with self.cond:
            areas, cache = self.areas, self.cache
        self.send({"type": "prelaunch_status"})
        end = time.monotonic() + timeout
        with self.cond:
            while (self.areas is areas or self.cache is cache) and not self.closed:
                left = end - time.monotonic()
                if left <= 0:
                    break
                self.cond.wait(left)
            return {"areas": self.areas, "cache": self.cache}

    def tile(self, layer, z, x, y, timeout=TILE_S):
        """A map tile's bytes from the server's tile files, or None."""
        key = (layer, z, x, y)
        path = os.path.join(self.tile_root, *([] if layer == "sat" else [layer]), str(z), str(x),
                            f"{y}.{LAYERS[layer]}")
        if os.path.isfile(path):
            return _read_file(path)
        with self.cond:
            if key not in self.tiles or self.tiles[key] is False:
                self.tiles[key] = None
                if self._batch is None:     # (map_tiles replaces the server's queue: every tile still wanted at once)
                    self._batch = threading.Timer(TILE_BATCH_S, self._ask_tiles)
                    self._batch.daemon = True
                    self._batch.start()
            end = time.monotonic() + timeout
            while self.tiles.get(key) is None and not self.closed:
                left = end - time.monotonic()
                if left <= 0:
                    break
                self.cond.wait(left)
            ok = self.tiles.get(key)
            if ok is None:
                self.tiles.pop(key, None)
        return _read_file(path) if ok else None

    def _ask_tiles(self):
        with self.cond:
            self._batch = None
            want = [[z, x, y, layer] for (layer, z, x, y), ok in self.tiles.items() if ok is None]
        if want:
            self.send({"type": "map_tiles", "tiles": want[:256]})


def _read_file(path):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _num(v, lo, hi):
    v = float(v)
    if not (lo <= v <= hi):
        raise ValueError(v)
    return v


def clean_request(msg):
    """The page's request as sent on to the server: only the fields each kind takes, numbers checked."""
    if not isinstance(msg, dict):
        raise ValueError("not an object")
    kind = msg.get("type")
    out = {"type": kind}
    if kind == "geocode":
        out["q"], out["full"] = str(msg.get("q") or "")[:200], True
    elif kind == "route":
        for k, lim in (("lat0", 85), ("lon0", 180), ("lat1", 85), ("lon1", 180)):
            out[k] = _num(msg.get(k), -lim, lim)
        out["download"] = "off"         # (the prefetch along the route is the game's; the page saves an area)
    elif kind in ("area_plan", "area_add"):
        if msg.get("route") == "current":
            out["route"], out["half"] = "current", _num(msg.get("half", 300), 50, 2000)
        else:
            for k, lim in (("s", 85), ("n", 85), ("w", 180), ("e", 180)):
                out[k] = _num(msg.get(k), -lim, lim)
        out["mode"] = "world" if msg.get("mode") == "world" else "data"
        if kind == "area_add":
            out["name"] = str(msg.get("name") or "")[:80]
            out["fast"] = msg.get("fast") is True
    elif kind == "area_cmd":
        if msg.get("op") not in ("pause", "resume", "delete", "fast", "slow"):
            raise ValueError("op")
        out["area"], out["op"] = str(msg.get("area") or "")[:64], msg["op"]
    else:
        raise ValueError(f"unknown request {kind!r}")
    return out


PLACES_FILE = os.path.join("settings", "beamngEarth_places.json")    # (the mod's places.lua M.FILE, in the user folder)
PLACE_TAG = " (préparé)"
NAME_MAX = 60                   # (places.lua M.NAME_MAX)


def place_name(name):
    return (name or "Zone")[:NAME_MAX - len(PLACE_TAG)] + PLACE_TAG


def _write_places(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_places(path):
    """The mod's places file as a dict, a new one when there is none, or None when it cannot be read (left alone)."""
    if not os.path.exists(path):
        return {"version": 1, "nextId": 1, "places": [], "recent": []}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("places"), list):
        return None
    return data


def add_place(user_dir, name, lat, lon):
    """A prepared place in the mod's saved places (places.lua): a start point in the level select and in the game's
    Places list; the mod reads the file again when it changes. One per name (a second preparation moves it)."""
    path = os.path.join(user_dir, PLACES_FILE)
    data = _read_places(path)
    if data is None:
        return False
    name = place_name(name)
    lat, lon = round(float(lat), 6), round(float(lon), 6)
    for p in data["places"]:
        if isinstance(p, dict) and p.get("name") == name:
            p["lat"], p["lon"] = lat, lon
            break
    else:
        ids = [p.get("id") for p in data["places"] if isinstance(p, dict) and isinstance(p.get("id"), int)]
        nid = max([int(data.get("nextId") or 1)] + [i + 1 for i in ids])
        data["places"].append({"id": nid, "name": name, "lat": lat, "lon": lon})
        data["nextId"] = nid + 1
    _write_places(path, data)
    return True


def remove_place(user_dir, name):
    """The prepared place named for an area taken out of the mod's saved places (the player's own ones stay)."""
    path = os.path.join(user_dir, PLACES_FILE)
    data = _read_places(path)
    if data is None or not os.path.exists(path):
        return False
    name = place_name(name)
    keep = [p for p in data["places"] if not (isinstance(p, dict) and p.get("name") == name)]
    if len(keep) == len(data["places"]):
        return False
    data["places"] = keep
    _write_places(path, data)
    return True


def place_start(bridge, user_dir, name, start):
    """A prepared area's start (a town's point, a box's middle, a route's start: [lat, lon]), on the nearest road when
    the server finds one, as a saved place."""
    try:
        lat, lon = _num(start[0], -85, 85), _num(start[1], -180, 180)
    except (TypeError, ValueError, IndexError):
        return False
    spot = bridge.ask({"type": "road_spot", "lat": lat, "lon": lon}, timeout=30)
    if isinstance(spot.get("lat"), (int, float)) and isinstance(spot.get("lon"), (int, float)):
        lat, lon = spot["lat"], spot["lon"]
    try:
        return add_place(user_dir, name, lat, lon)
    except OSError:
        return False


def prepare(bridge, user_dir, msg):
    """area_add (the page's or the window's request) and, when it is saved, its start as a saved place."""
    req = clean_request(dict(msg, type="area_add"))
    out = bridge.ask(req)
    if out.get("ok"):
        place_start(bridge, user_dir, req.get("name"), msg.get("start"))
    return out


def delete(bridge, user_dir, area_id, name):
    """area_cmd delete and the area's saved place taken out."""
    out = bridge.ask(clean_request({"type": "area_cmd", "area": area_id, "op": "delete"}))
    if out.get("ok"):
        remove_place(user_dir, str(name or "")[:80])
    return out


def launch_game(bridge, launch):
    """The game may start: an area running fast goes on in the background (the mod pauses the game meanwhile)."""
    fast = (bridge.areas or {}).get("fast")
    if fast:
        bridge.ask({"type": "area_cmd", "area": fast, "op": "slow"}, timeout=10)
    launch.set()


def handler_for(bridge, launch, user_dir):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):      # (quiet: the launcher's console is for the launcher)
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(dict(bridge.status(), launched=launch.is_set()))
            elif path.startswith("/tiles/"):
                parts = path.split("/")[2:]
                try:
                    layer, z, x, y = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                    if layer not in LAYERS or not (0 <= z <= MAX_Z and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
                        raise ValueError
                except (ValueError, IndexError):
                    self._send(404, b"", "text/plain")
                    return
                data = bridge.tile(layer, z, x, y)
                if data is None:
                    self._send(404, b"", "text/plain")
                else:
                    self._send(200, data, "image/jpeg" if LAYERS[layer] == "jpg" else "image/png")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            # (a page of another site cannot post here: no CORS headers, and a JSON body is required)
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self._send(415, b"", "text/plain")
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > 65536:
                self._send(413, b"", "text/plain")
                return
            try:
                msg = json.loads(self.rfile.read(n).decode("utf-8"))
            except ValueError:
                self._send(400, b"", "text/plain")
                return
            if self.path == "/api/launch":
                launch_game(bridge, launch)
                self._json({"ok": True})
                return
            if self.path != "/api/ask":
                self._send(404, b"", "text/plain")
                return
            try:
                req = clean_request(msg)
            except (ValueError, TypeError) as e:
                self._json({"error": f"requête refusée ({e})"}, 400)
                return
            if req["type"] == "area_add":
                self._json(prepare(bridge, user_dir, msg))
            elif req["type"] == "area_cmd" and req["op"] == "delete":
                self._json(delete(bridge, user_dir, req["area"], msg.get("name")))
            else:
                self._json(bridge.ask(req))
    return Handler


def start(user_dir, port, page_port=PAGE_PORT):
    """The bridge and the page's server (threads); (url, launch event, http server, bridge)."""
    bridge = Bridge(port, os.path.join(user_dir, "temp", "beamngEarth", "maptiles"))
    launch = threading.Event()
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", page_port), handler_for(bridge, launch, user_dir))
    except OSError:     # (taken, or in a range Windows keeps for WSL's mirrored network: any free port)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(bridge, launch, user_dir))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="prelaunch-http", daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/", launch, httpd, bridge


def run(user_dir, port, say=print):
    """The window before the game (prelaunch_window.py, tkinter), else the page in the browser; return when the game
    should start (the window's or the page's button, or Enter here)."""
    try:
        url, launch, _httpd, bridge = start(user_dir, port)
    except OSError as e:
        say(f"  Fenêtre de préparation indisponible ({e}) : le jeu démarre directement.")
        return
    try:
        import prelaunch_window
    except ImportError:         # (a Python without tkinter: the page)
        prelaunch_window = None
    if prelaunch_window is not None:
        say("  Fenêtre « Avant de jouer » ouverte : préparer des lieux, puis « Lancer le jeu ».")
        try:
            prelaunch_window.Window(bridge, launch, user_dir, url).run()
            return
        except Exception as e:  # (no display, a Tk error: the page instead)
            say(f"  Fenêtre indisponible ({type(e).__name__}: {e}) : la page s'ouvre dans le navigateur.")
    say(f"  Préparer des lieux avant de jouer : {url}")
    say("  (bouton « Lancer le jeu » dans la page, ou Entrée ici)")
    webbrowser.open(url)

    def enter():
        try:
            input()
        except (EOFError, OSError):
            return
        launch.set()
    threading.Thread(target=enter, name="prelaunch-enter", daemon=True).start()
    launch.wait()


PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Préparer le monde</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--bg:#16181c;--panel:#1f2228;--line:#30343c;--text:#e8e9ec;--muted:#9aa0aa;--accent:#e0393e;--ok:#3fb27f;--warn:#e0a63a}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif;display:grid;grid-template-columns:1fr 380px;grid-template-rows:100%}
#map{height:100%;background:#0d0f12}
aside{background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
section{padding:14px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0 0 4px}h2{font-size:13px;margin:0 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.muted{color:var(--muted);font-size:12.5px}
input{width:100%;background:#15171b;border:1px solid var(--line);color:var(--text);border-radius:6px;padding:7px 9px;font:inherit}
button{background:#2b2f37;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px 11px;font:inherit;cursor:pointer}
button:hover{border-color:#4a505c}button:disabled{opacity:.45;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button:focus-visible,input:focus-visible{outline:2px solid #7aa7ff;outline-offset:1px}
.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}.row>button{flex:none}
.tabs{display:flex;gap:6px;margin-bottom:10px}.tabs button[aria-pressed=true]{border-color:var(--accent);color:#fff}
.results button{display:block;width:100%;text-align:left;margin-top:4px;background:transparent}
#estimate{margin-top:10px}
.areas{overflow:auto;flex:1;min-height:0;padding:0 16px}
.area{padding:10px 0;border-bottom:1px solid var(--line)}
.area .top{display:flex;justify-content:space-between;gap:8px}.area b{font-weight:600}
.bar{height:5px;background:#2b2f37;border-radius:3px;margin:6px 0;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
.bar i.run{background:var(--warn)}
.area .acts{display:flex;gap:6px}.area .acts button{padding:3px 8px;font-size:12.5px}
footer{padding:12px 16px;border-top:1px solid var(--line);display:flex;gap:8px}footer button{flex:1;padding:10px}
#cache{font-size:13px}
@media (max-width:820px){body{grid-template-columns:1fr;grid-template-rows:45vh auto}aside{border-left:0}}
</style></head><body>
<div id="map" role="application" aria-label="Carte : Maj + glisser pour encadrer une zone"></div>
<aside>
  <section><h1>Préparer le monde</h1>
    <div class="muted">Un lieu préparé se charge vite au téléport, pour toi et pour tes amis sur le même serveur.</div></section>
  <section>
    <div class="tabs" role="group" aria-label="Façon de choisir">
      <button id="t-town" aria-pressed="true">Une ville</button><button id="t-box" aria-pressed="false">Une zone</button><button id="t-route" aria-pressed="false">Un trajet</button></div>
    <div id="p-town"><label class="muted" for="q">Ville ou adresse</label>
      <div class="row"><input id="q" placeholder="Castres, Pas de la Case…" autocomplete="off"><button id="go-q">Chercher</button></div>
      <div class="results" id="res-q"></div></div>
    <div id="p-box" hidden><div class="muted">Sur la carte : <b>Maj + glisser</b> pour encadrer la zone.</div></div>
    <div id="p-route" hidden>
      <label class="muted" for="qa">Départ</label><input id="qa" placeholder="Toulouse">
      <label class="muted" for="qb">Arrivée</label><input id="qb" placeholder="Pas de la Case">
      <div class="row" style="margin-top:8px"><button id="go-r">Tracer le trajet</button></div>
      <div class="muted" id="r-info"></div></div>
    <div id="estimate" class="muted" aria-live="polite"></div>
    <div class="row" style="margin-top:8px">
      <button id="add-fast" class="primary" disabled>Préparer à fond</button>
      <button id="add-slow" disabled>En arrière-plan</button></div>
    <div class="muted" style="margin-top:6px">« À fond » prend tout le processeur avant de jouer ; « en arrière-plan » continue doucement pendant le jeu.</div>
  </section>
  <section><h2>Déjà préparé</h2><div id="cache" class="muted">…</div></section>
  <div class="areas" id="areas" aria-live="polite"></div>
  <footer><button id="launch" class="primary">Lancer le jeu</button></footer>
</aside>
<script>
"use strict";
const $ = id => document.getElementById(id);
const map = L.map("map", {zoomControl: true, boxZoom: false}).setView([46.6, 2.4], 6);
L.tileLayer("/tiles/sat/{z}/{x}/{y}", {maxZoom: 18, attribution: "Images : serveur BeamNG.Earth"}).addTo(map);
L.tileLayer("/tiles/labels/{z}/{x}/{y}", {maxZoom: 18}).addTo(map);
const areaLayer = L.layerGroup().addTo(map);
let pick = null, pickShape = null, launched = false;

async function ask(msg) {
  const r = await fetch("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(msg)});
  return r.json();
}
const mb = b => b >= 1e9 ? (b / 1e9).toFixed(1).replace(".", ",") + " Go" : Math.max(1, Math.round(b / 1e6)) + " Mo";
const dur = s => s < 90 ? Math.round(s) + " s" : s < 5400 ? Math.round(s / 60) + " min" : (s / 3600).toFixed(1).replace(".", ",") + " h";
function esc(s) { const d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }

function setTab(t) {
  for (const k of ["town", "box", "route"]) { $("t-" + k).setAttribute("aria-pressed", k === t); $("p-" + k).hidden = k !== t; }
}
for (const k of ["town", "box", "route"]) $("t-" + k).onclick = () => setTab(k);

function setPick(p, shape) {
  pick = p;
  if (pickShape) map.removeLayer(pickShape);
  pickShape = shape ? shape.addTo(map) : null;
  $("add-fast").disabled = $("add-slow").disabled = true;
  $("estimate").textContent = p ? "Estimation…" : "";
  if (p) estimate();
}
function boxAround(lat, lon, km) {
  const dl = km / 2 / 111.32, dn = dl / Math.cos(lat * Math.PI / 180);
  return {s: lat - dl, n: lat + dl, w: lon - dn, e: lon + dn};
}
function showBox(b, name, start) {
  setPick(Object.assign({name, start: start || [(b.s + b.n) / 2, (b.w + b.e) / 2]}, b), L.rectangle([[b.s, b.w], [b.n, b.e]], {color: "#e0393e", weight: 2, fillOpacity: .08}));
  map.fitBounds([[b.s, b.w], [b.n, b.e]], {maxZoom: 14});
}
async function estimate() {
  const p = pick, base = p.route ? {route: "current", half: p.half} : {s: p.s, w: p.w, n: p.n, e: p.e};
  let r = await ask(Object.assign({type: "area_plan", mode: "world"}, base)), mode = "world";
  if (!r.ok) { const d = await ask(Object.assign({type: "area_plan", mode: "data"}, base)); if (d.ok) { r = d; mode = "data"; } }
  if (p !== pick) return;
  if (!r.ok) { $("estimate").textContent = "Pas d'estimation : " + (r.error || "erreur"); return; }
  p.mode = mode;
  $("estimate").innerHTML = `<b>${esc(p.name || "Zone")}</b> : ${Math.round(r.km2)} km²${r.km ? ", " + Math.round(r.km) + " km de route" : ""}, `
    + `${mb(r.downloadBytes || 0)} à télécharger, ${mb(r.diskBytes || 0)} sur le disque, environ ${dur(r.seconds || 0)}`
    + (mode === "data" ? "<br>Trop grande pour tout construire d'avance : seulement les téléchargements." : "");
  $("add-fast").disabled = $("add-slow").disabled = false;
}
async function add(fast) {
  const p = pick; if (!p) return;
  $("add-fast").disabled = $("add-slow").disabled = true;
  const base = p.route ? {route: "current", half: p.half} : {s: p.s, w: p.w, n: p.n, e: p.e};
  const r = await ask(Object.assign({type: "area_add", mode: p.mode || "world", name: p.name || "", fast, start: p.start}, base));
  if (!r.ok) { $("estimate").textContent = "Refusé : " + (r.error || "erreur"); $("add-fast").disabled = $("add-slow").disabled = false; return; }
  setPick(null); refresh();
  $("estimate").innerHTML = `<b>${esc(p.name || "Zone")} (préparé)</b> est ajouté aux lieux de départ : menu de la session libre et liste « Places » du jeu.`;
}
$("add-fast").onclick = () => add(true);
$("add-slow").onclick = () => add(false);

async function search() {
  const q = $("q").value.trim(); if (!q) return;
  $("res-q").textContent = "Recherche…";
  const r = await ask({type: "geocode", q});
  const list = r.results || [];
  $("res-q").innerHTML = list.length ? "" : esc(r.text || r.error || "Rien trouvé");
  for (const it of list.slice(0, 6)) {
    const b = document.createElement("button");
    b.textContent = it.detail ? `${it.name}, ${it.detail}` : it.name;
    b.onclick = () => { $("res-q").innerHTML = ""; showBox(boxAround(+it.lat, +it.lon, 4), it.name, [+it.lat, +it.lon]); };
    $("res-q").appendChild(b);
  }
}
$("go-q").onclick = search;
$("q").onkeydown = e => { if (e.key === "Enter") search(); };

async function place(q) { const r = await ask({type: "geocode", q}); return (r.results || [])[0]; }
$("go-r").onclick = async () => {
  $("r-info").textContent = "Recherche des lieux…";
  const a = await place($("qa").value.trim()), b = await place($("qb").value.trim());
  if (!a || !b) { $("r-info").textContent = "Départ ou arrivée introuvable."; return; }
  $("r-info").textContent = "Calcul du trajet…";
  const r = await ask({type: "route", lat0: +a.lat, lon0: +a.lon, lat1: +b.lat, lon1: +b.lon});
  if (r.error || !(r.points || []).length) { $("r-info").textContent = "Trajet impossible : " + (r.error || "aucun"); return; }
  $("r-info").textContent = `${Math.round(r.dist / 1000)} km, environ ${dur(r.dur)} de route`;
  const line = L.polyline(r.points, {color: "#e0393e", weight: 4});
  setPick({route: true, half: 300, name: `${$("qa").value.trim()} → ${$("qb").value.trim()}`, start: r.points[0]}, line);
  map.fitBounds(line.getBounds());
};

// Maj + glisser : encadrer une zone
let drag = null;
map.on("mousedown", e => {
  if (!e.originalEvent.shiftKey) return;
  map.dragging.disable(); setTab("box");
  drag = {a: e.latlng, r: L.rectangle([e.latlng, e.latlng], {color: "#e0393e", weight: 2, fillOpacity: .08}).addTo(map)};
});
map.on("mousemove", e => { if (drag) drag.r.setBounds([drag.a, e.latlng]); });
map.on("mouseup", e => {
  if (!drag) return;
  map.dragging.enable();
  const b = L.latLngBounds(drag.a, e.latlng); map.removeLayer(drag.r); drag = null;
  if (b.getNorth() - b.getSouth() < 0.002) return;
  showBox({s: b.getSouth(), n: b.getNorth(), w: b.getWest(), e: b.getEast()}, "Zone");
});

const STATE = {queued: "en attente", running: "en cours", done: "prête", paused: "en pause", waiting: "en attente (quota du jour)", failed: "en échec"};
async function cmd(id, op, name) { await ask({type: "area_cmd", area: id, op, name}); refresh(); }
async function refresh() {
  let st; try { st = await (await fetch("/api/status")).json(); } catch (e) { return; }
  const c = st.cache || {}, a = st.areas || {areas: []};
  $("cache").innerHTML = c.capBytes != null
    ? `Cache : <b>${mb(c.usedBytes || 0)}</b> sur ${mb(c.capBytes)}, dont ${mb(c.pinnedBytes || a.pinnedBytes || 0)} de lieux préparés (jamais effacés).<br>Le reste se nettoie seul quand il manque de place.`
    : "Cache : en cours de lecture…";
  areaLayer.clearLayers();
  const box = $("areas"); box.innerHTML = "";
  if (!a.areas.length) box.innerHTML = '<p class="muted">Aucun lieu préparé pour l\'instant.</p>';
  for (const it of a.areas) {
    const pct = it.total ? Math.min(100, Math.round(100 * it.done / it.total)) : (it.state === "done" ? 100 : 0);
    const fast = a.fast === it.id, d = document.createElement("div");
    d.className = "area";
    d.innerHTML = `<div class="top"><b>${esc(it.name || "Zone")}</b><span class="muted">${esc(STATE[it.state] || it.state)}${fast ? ", à fond" : ""} · ${pct} %</span></div>`
      + `<div class="bar"><i class="${it.state === "done" ? "" : "run"}" style="width:${pct}%"></i></div>`
      + (it.msg ? `<div class="muted">${esc(it.msg)}</div>` : "")
      + `<div class="acts"></div>`;
    const acts = d.querySelector(".acts");
    const btn = (label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn; acts.appendChild(b); };
    btn("Voir", () => map.fitBounds([[it.s, it.w], [it.n, it.e]], {maxZoom: 14}));
    if (it.state !== "done") {
      if (it.state === "paused") btn("Reprendre", () => cmd(it.id, "resume")); else btn("Pause", () => cmd(it.id, "pause"));
      if (!fast && !launched) btn("À fond", () => cmd(it.id, "fast"));
    }
    btn("Supprimer", () => { if (confirm(`Supprimer « ${it.name || "Zone"} » ? Ses données ne sont plus protégées et le cache les efface quand il a besoin de place ; son lieu de départ est retiré.`)) cmd(it.id, "delete", it.name || "Zone"); });
    box.appendChild(d);
    const shape = it.route ? L.polyline(it.route, {color: "#3fb27f", weight: 3}) : L.rectangle([[it.s, it.w], [it.n, it.e]], {color: "#3fb27f", weight: 1, fillOpacity: .06});
    areaLayer.addLayer(shape.bindTooltip(it.name || "Zone"));
  }
  launched = !!st.launched;
  $("launch").textContent = launched ? "Jeu lancé : la préparation continue" : "Lancer le jeu";
  $("launch").disabled = launched;
}
$("launch").onclick = async () => {
  await fetch("/api/launch", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  refresh();
};
refresh(); setInterval(refresh, 3000);
</script></body></html>
"""
