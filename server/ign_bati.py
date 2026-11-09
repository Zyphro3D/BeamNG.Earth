"""France: real building attributes from IGN BD TOPO (Etalab Licence Ouverte 2.0), joined to the buildings layer.

Overture knows the footprints of French buildings but few heights (1-9 % around Toulouse) and hardly any roof or wall.
BD TOPO `batiment` has them, per z14 map-data cell, kept on disk under <cache>/ign_bati/v<VERSION>/<z>/<x>/<y>.json.gz:

- height: `hauteur` (to the eaves) plus the roof's rise (altitude_maximale_toit - altitude_minimale_toit);
- roof_shape: flat under ROOF_FLAT_M of rise, else pitched (hipped or gabled by a hash of the building's id);
- num_floors: `nombre_d_etages`;
- roof_material and facade_material: the first digit of the land registry's codes (materiaux_de_la_toiture, _des_murs).
  Brick there is what the wall is built of, and a low house is usually rendered over it, so low buildings get plaster.

A building takes the BD TOPO building its footprint's centre lies in. A map value that BD TOPO lacks is left alone.
Off: BNE_IGN_BATI=off. Pure given the cached data; a fetch that fails leaves the cell as Overture has it.
"""
import gzip
import json
import logging
import os
import threading
import zlib

import numpy as np

log = logging.getLogger("earthd.ign_bati")

VERSION = 1
WFS_URL = "https://data.geopf.fr/wfs/ows"
WFS_LANE = "ign_wfs"
WFS_PAGE = 5000
WFS_PAGES = 20
ON = os.environ.get("BNE_IGN_BATI", "on").lower() != "off"
FRANCE = (-5.5, 41.0, 9.8, 51.3)        # metropolitan France, west south east north
MARGIN = 0.002                          # degrees round the cell: a building centred here may be listed by a neighbour
GRID = 500                              # candidate grid, cells per degree
ROOF_FLAT_M = 0.6
ROOF_MAX_M = 8.0
WALLS = {"1": "stone", "2": "stone", "3": "concrete", "5": "concrete", "6": "wood"}
ROOFS = {"1": "tile", "2": "slate", "3": "zinc", "4": "concrete"}
BRICK_MIN_FLOORS, BRICK_MIN_M = 3, 9.0  # below this a brick building is taken as rendered
PROPS = ("geometrie,hauteur,nombre_d_etages,altitude_maximale_toit,altitude_minimale_toit,"
         "materiaux_des_murs,materiaux_de_la_toiture")
ROOF_SHAPES = ("flat", "gabled", "hipped")
TEXT_COLUMNS = ("roof_shape", "roof_material", "facade_material", "height_src")


def in_france(lon0, lat0, lon1, lat1):
    return lon1 >= FRANCE[0] and lon0 <= FRANCE[2] and lat1 >= FRANCE[1] and lat0 <= FRANCE[3]


def parse(features):
    """Compact rows [ring, props] of a GeoJSON page: each outer ring is its own row."""
    rows = []
    for ft in features:
        geom = ft.get("geometry") or {}
        polys = [geom.get("coordinates")] if geom.get("type") == "Polygon" else geom.get("coordinates") or []
        p = ft.get("properties") or {}
        keep = {k: p.get(k) for k in ("hauteur", "nombre_d_etages", "altitude_maximale_toit", "altitude_minimale_toit",
                                      "materiaux_des_murs", "materiaux_de_la_toiture")}
        for poly in polys:
            if poly and len(poly[0]) >= 4:
                rows.append([[[round(c[0], 7), round(c[1], 7)] for c in poly[0]], keep])
    return rows


def attributes(props, seed):
    """BeamNG.Earth's building attributes from one BD TOPO building (only those it knows)."""
    out = {}
    h, top, low = props.get("hauteur"), props.get("altitude_maximale_toit"), props.get("altitude_minimale_toit")
    rise = None
    if top is not None and low is not None and top > -100 and low > -100:
        rise = max(0.0, min(ROOF_MAX_M, float(top) - float(low)))
    pitched = rise is not None and rise >= ROOF_FLAT_M
    if h and h > 0:
        out["height"] = float(h) + (rise if pitched else 0.0)
        if pitched:
            out["roof_height"] = rise
    floors = props.get("nombre_d_etages")
    if floors and floors > 0:
        out["num_floors"] = float(floors)
    if rise is not None:
        out["roof_shape"] = "flat" if not pitched else ("hipped" if zlib.crc32(seed.encode()) % 100 < 60 else "gabled")
    code = str(props.get("materiaux_de_la_toiture") or "0")[0]
    if code in ROOFS:
        out["roof_material"] = ROOFS[code]
    code = str(props.get("materiaux_des_murs") or "0")[0]
    if code == "4":
        out["facade_material"] = "brick" if (floors or 0) >= BRICK_MIN_FLOORS or (h or 0) >= BRICK_MIN_M else "plaster"
    elif code in WALLS:
        out["facade_material"] = WALLS[code]
    return out


def _inside(ring, x, y):
    hit, n = False, len(ring)
    for i in range(n):
        ax, ay = ring[i]
        bx, by = ring[i - 1]
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            hit = not hit
    return hit


def ring_key(lon, lat):
    return f"{lon:.6f},{lat:.6f}"


def join(feats, rows, roofs=None):
    """Writes the attributes of `rows` over feats (a buildings Features). Returns how many buildings took one. roofs:
    {ring_key of a BD TOPO ring's mean: shape}, the shape measured by the LiDAR (ign_limites.roof_shape)."""
    items, grid = [], {}
    for ring, props in rows:
        xs, ys = [c[0] for c in ring], [c[1] for c in ring]
        box = (min(xs), min(ys), max(xs), max(ys))
        k = len(items)
        items.append((ring, box, props))
        for gx in range(int(box[0] * GRID), int(box[2] * GRID) + 1):
            for gy in range(int(box[1] * GRID), int(box[3] * GRID) + 1):
                grid.setdefault((gx, gy), []).append(k)
    a = feats.attrs
    cx, cy = a["cx"], a["cy"]
    numbers = {k: np.array(a[k], copy=True) for k in ("height", "num_floors", "roof_height")}
    texts = {k: np.asarray(a[k]).astype(object) for k in TEXT_COLUMNS}
    joined = 0
    for i in range(len(feats)):
        x, y = float(cx[i]), float(cy[i])
        found, found_ring = None, None
        for k in grid.get((int(x * GRID), int(y * GRID)), ()):
            ring, b, props = items[k]
            if b[0] <= x <= b[2] and b[1] <= y <= b[3] and _inside(ring, x, y):
                found, found_ring = props, ring
                break
        if found is None:
            continue
        got = attributes(found, str(feats.ids[i]))
        if roofs:
            measured = roofs.get(ring_key(*np.asarray(found_ring, np.float64).mean(axis=0)))
            if measured is not None:
                got["roof_shape"] = ROOF_SHAPES[int(measured)]
                if not measured:
                    got.pop("roof_height", None)
        if got:
            joined += 1
        for k in numbers:
            if k in got:
                numbers[k][i] = got[k]
        for k in ("roof_shape", "roof_material", "facade_material"):
            if k in got:
                texts[k][i] = got[k]
        if "height" in got:
            texts["height_src"][i] = "IGN BD TOPO"
    a.update(numbers)
    for k, v in texts.items():
        a[k] = v.astype(str)
    return joined


class Provider:
    """The BD TOPO buildings per map cell, read through the server's fetcher and kept on disk."""

    def __init__(self, cache_dir, fetcher):
        self.dir = os.path.join(cache_dir, "ign_bati", f"v{VERSION}") if cache_dir else None
        self.cache_dir = cache_dir
        self.fetcher = fetcher
        self._locks = {}
        self._lock = threading.Lock()

    def _path(self, key):
        return os.path.join(self.dir, str(key.z), str(key.x), f"{key.y}.json.gz")

    def _fetch(self, key):
        lon0, lat0, lon1, lat1 = key.bounds()
        box = f"{lon0 - MARGIN:.6f},{lat0 - MARGIN:.6f},{lon1 + MARGIN:.6f},{lat1 + MARGIN:.6f}"
        rows = []
        for page in range(WFS_PAGES):
            url = (f"{WFS_URL}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES=BDTOPO_V3:batiment"
                   f"&PROPERTYNAME={PROPS}&BBOX={box},CRS:84&SRSNAME=CRS:84&OUTPUTFORMAT=application/json"
                   f"&COUNT={WFS_PAGE}&STARTINDEX={page * WFS_PAGE}&SORTBY=cleabs")
            res = self.fetcher.get(url, kind="demand", lane=WFS_LANE)
            features = json.loads(res.body.decode("utf-8")).get("features") or []
            rows += parse(features)
            if len(features) < WFS_PAGE:
                break
        return rows

    def rows(self, key):
        path = self._path(key) if self.dir else None
        with self._lock:
            lock = self._locks.setdefault(str(key), threading.Lock())
        with lock:
            if path and os.path.exists(path):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            rows = self._fetch(key)
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
                with gzip.open(tmp, "wt", encoding="utf-8") as f:
                    json.dump(rows, f, separators=(",", ":"))
                os.replace(tmp, path)
            return rows

    def roofs(self, key):
        """{ring_key: shape} of the cell's roofs measured by the LiDAR (ign_limites' file), {} until it is worked out."""
        if not self.cache_dir:
            return {}
        path = os.path.join(self.cache_dir, "ign_limites", "v1", str(key.z), str(key.x), f"{key.y}.npz")
        try:
            with np.load(path, allow_pickle=False) as z:
                if "roofs" not in z.files:
                    return {}
                return {ring_key(lon, lat): shape for lon, lat, shape in z["roofs"].tolist()}
        except (OSError, ValueError):
            return {}

    def enrich(self, key, feats):
        """Joins BD TOPO to a buildings Features in France. A failure leaves it as it was."""
        if not ON or feats is None or not len(feats) or feats.__dict__.get("_ign_bati"):
            return 0
        if not in_france(*key.bounds()):
            return 0
        try:
            joined = join(feats, self.rows(key), self.roofs(key))
        except Exception as e:      # (no network, a changed answer: Overture's buildings stand)
            log.warning("ign_bati %s: %s: %s", key, type(e).__name__, e)
            return 0
        feats.__dict__["_ign_bati"] = True
        log.debug("ign_bati %s: %d of %d buildings", key, joined, len(feats))
        return joined
