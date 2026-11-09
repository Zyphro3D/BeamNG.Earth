"""France: carriageway widths from IGN BD TOPO (troncon_de_route, Etalab Licence Ouverte 2.0).

Overture gives few French roads a width, so most take their class's default. BD TOPO has `largeur_de_chaussee` on most
of them. A road the map gives no width takes the median over BD TOPO segments running along it: written to the same
column as OpenStreetMap's width, so a mapped width wins and everything downstream (road_specs, grading, the meshes, the
masks) reads one value. Per z14 cell, kept under <cache>/ign_largeur/v<VERSION>/<z>/<x>/<y>.json.gz.
Off: BNE_IGN_LARGEUR=off. A fetch that fails leaves the cell as it was.
"""
import gzip
import json
import logging
import math
import os
import threading

import numpy as np

import ign_bati
import mapgeom

log = logging.getLogger("earthd.ign_largeur")

VERSION = 1
WFS_URL = "https://data.geopf.fr/wfs/ows"
WFS_LANE = "ign_wfs"
WFS_PAGE = 5000
WFS_PAGES = 20
ON = os.environ.get("BNE_IGN_LARGEUR", "on").lower() != "off"
MARGIN = 0.002              # degrees round the cell
MATCH_M = 8.0               # m from a road's point to the BD TOPO segment it takes
SAME_WAY_COS = 0.85         # |cos| of the angle between the road and that segment
GRID = 2000                 # candidate grid, cells per degree
MIN_M, MAX_M = 2.0, 30.0
SAMPLES = 5
MIN_VOTES = 2               # sampled points that must find a segment
# BD TOPO against OpenStreetMap's mapped widths (323 roads round Toulouse): no bias, a typical 1 m off. But its small
# streets are nearly all 3 or 4 m, a default rather than a measure (real ones are drawn 5.5 m or more), and on a motorway
# it is one carriageway, not the road. So: the through roads' values, and a small street's only when above the default.
THROUGH = {"primary", "secondary", "tertiary"}
SMALL = {"residential", "unclassified", "living_street"}
SMALL_MIN_M = 5.0
# Its 5 m values are often one carriageway of a split section read onto the two-way road beside it, and 3-4 m on a
# through road is a default: a road with two directions drawn that narrow leaves no room to pass.
THROUGH_TWO_WAY_MIN_M = 6.0
THROUGH_ONE_WAY_MIN_M = 4.0


def parse(features):
    """Rows [line, width] of a GeoJSON page: only real segments with a width."""
    rows = []
    for ft in features:
        p = ft.get("properties") or {}
        w = p.get("largeur_de_chaussee")
        if p.get("fictif") in (True, "true", "Vrai") or not isinstance(w, (int, float)) or not MIN_M <= w <= MAX_M:
            continue
        g = ft.get("geometry") or {}
        lines = [g.get("coordinates")] if g.get("type") == "LineString" else g.get("coordinates") or []
        for line in lines:
            if line and len(line) >= 2:
                rows.append([[[round(c[0], 7), round(c[1], 7)] for c in line], float(w)])
    return rows


def _index(rows):
    segs, grid = [], {}
    for line, w in rows:
        for a, b in zip(line[:-1], line[1:]):
            k = len(segs)
            segs.append((a[0], a[1], b[0], b[1], w))
            pad = MATCH_M / 111000.0
            for gx in range(int(math.floor((min(a[0], b[0]) - pad) * GRID)), int(math.floor((max(a[0], b[0]) + pad) * GRID)) + 1):
                for gy in range(int(math.floor((min(a[1], b[1]) - pad) * GRID)), int(math.floor((max(a[1], b[1]) + pad) * GRID)) + 1):
                    grid.setdefault((gx, gy), []).append(k)
    return segs, grid


def _width_at(px, py, ux, uy, segs, grid, kx):
    """The width of the nearest segment running along (ux, uy) at (px, py) in degrees, or None."""
    best, width = MATCH_M, None
    for k in grid.get((int(math.floor(px * GRID)), int(math.floor(py * GRID))), ()):
        ax, ay, bx, by, w = segs[k]
        dx, dy = (bx - ax) * kx, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-9 or abs(dx * ux + dy * uy) / length < SAME_WAY_COS:
            continue
        t = max(0.0, min(1.0, (((px - ax) * kx) * dx + (py - ay) * dy) / (length * length)))
        d = math.hypot((px - ax) * kx - t * dx, (py - ay) - t * dy) * 111000.0
        if d <= best:
            best, width = d, w
    return width


def _longest(feats, i):
    out = None
    for part in feats.parts(i):
        for ring in part:
            if len(ring) >= 2 and (out is None or len(ring) > len(out)):
                out = np.asarray(ring, np.float64)
    return out


def _one_way(feats, i):
    rule = feats.rules.get("oneway")
    return rule is not None and len(rule.feature) > 0 and rule.major(i) >= 0


def join(feats, rows):
    """Writes BD TOPO's widths to feats' osm_width where the map has none. Returns how many roads took one."""
    n = len(feats)
    if not n or not rows:
        return 0
    segs, grid = _index(rows)
    old = feats.attrs.get("osm_width")
    col = np.full(n, "", dtype=object) if old is None else np.asarray(old).astype(object)
    classes = feats.attrs.get("class")
    joined = 0
    for i in range(n):
        if feats.kind[i] != mapgeom.LINE or str(col[i]).strip() not in ("", "nan", "None"):
            continue
        cls = str(classes[i]) if classes is not None else "primary"
        if cls not in THROUGH and cls not in SMALL:
            continue
        ring = _longest(feats, i)
        if ring is None:
            continue
        kx = math.cos(math.radians(float(ring[:, 1].mean())))
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(ring[:, 0]) * kx, np.diff(ring[:, 1])))]
        if s[-1] <= 0:
            continue
        got = []
        for f in (np.arange(SAMPLES) + 0.5) / SAMPLES * s[-1]:
            j = min(int(np.searchsorted(s, f, side="right")) - 1, len(ring) - 2)
            ds = max(s[j + 1] - s[j], 1e-12)
            t = (f - s[j]) / ds
            ux, uy = (ring[j + 1, 0] - ring[j, 0]) * kx / ds, (ring[j + 1, 1] - ring[j, 1]) / ds
            w = _width_at(ring[j, 0] + t * (ring[j + 1, 0] - ring[j, 0]), ring[j, 1] + t * (ring[j + 1, 1] - ring[j, 1]),
                          ux, uy, segs, grid, kx)
            if w is not None:
                got.append(w)
        if len(got) < MIN_VOTES:
            continue
        width = float(np.median(got))
        if cls in SMALL and width < SMALL_MIN_M:
            continue
        if cls in THROUGH:
            width = max(width, THROUGH_ONE_WAY_MIN_M if _one_way(feats, i) else THROUGH_TWO_WAY_MIN_M)
        col[i] = f"{width:g}"
        joined += 1
    feats.attrs["osm_width"] = col.astype(str)
    return joined


class Provider:
    """BD TOPO's road widths per map cell, read through the server's fetcher and kept on disk."""

    def __init__(self, cache_dir, fetcher):
        self.dir = os.path.join(cache_dir, "ign_largeur", f"v{VERSION}") if cache_dir else None
        self.fetcher = fetcher
        self._locks, self._lock = {}, threading.Lock()

    def _path(self, key):
        return os.path.join(self.dir, str(key.z), str(key.x), f"{key.y}.json.gz")

    def _fetch(self, key):
        lon0, lat0, lon1, lat1 = key.bounds()
        box = f"{lon0 - MARGIN:.6f},{lat0 - MARGIN:.6f},{lon1 + MARGIN:.6f},{lat1 + MARGIN:.6f}"
        rows = []
        for page in range(WFS_PAGES):
            url = (f"{WFS_URL}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES=BDTOPO_V3:troncon_de_route"
                   f"&PROPERTYNAME=geometrie,largeur_de_chaussee,fictif&BBOX={box},CRS:84&SRSNAME=CRS:84"
                   f"&OUTPUTFORMAT=application/json&COUNT={WFS_PAGE}&STARTINDEX={page * WFS_PAGE}&SORTBY=cleabs")
            features = json.loads(self.fetcher.get(url, kind="demand", lane=WFS_LANE).body.decode("utf-8")).get("features") or []
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

    def enrich(self, key, feats):
        """Joins BD TOPO to a roads Features in France. A failure leaves it as it was."""
        if not ON or feats is None or not len(feats) or feats.__dict__.get("_ign_largeur"):
            return 0
        if not ign_bati.in_france(*key.bounds()):
            return 0
        try:
            joined = join(feats, self.rows(key))
        except Exception as e:      # (no network, a changed answer: the class defaults stand)
            log.warning("ign_largeur %s: %s: %s", key, type(e).__name__, e)
            return 0
        feats.__dict__["_ign_largeur"] = True
        log.debug("ign_largeur %s: %d of %d roads", key, joined, len(feats))
        return joined
