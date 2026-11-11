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
# BD TOPO against OpenStreetMap's mapped widths (323 roads round a French city): no bias, a typical 1 m off. But its small
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


def _segments(rows):
    """The rows' segments (m, 5): ax, ay, bx, by (degrees), width, in the rows' order (one array for the whole cell: one
    small one a line was ~1.1 s of a city's first block, 2026-11-11)."""
    lines = [(line, w) for line, w in rows if len(line) >= 2]
    if not lines:
        return np.zeros((0, 5))
    n = np.array([len(line) for line, _ in lines])
    pts = np.array([pt for line, _ in lines for pt in line], np.float64).reshape(-1, 2)
    keep = np.ones(len(pts) - 1, bool)
    keep[np.cumsum(n)[:-1] - 1] = False         # (no segment from one line's last point to the next line's first)
    return np.c_[pts[:-1][keep], pts[1:][keep], np.repeat(np.array([float(w) for _, w in lines]), n - 1)]


def _one_way(feats, i):
    rule = feats.rules.get("oneway")
    return rule is not None and len(rule.feature) > 0 and rule.major(i) >= 0


def join(feats, rows):
    """Writes BD TOPO's widths to feats' osm_width where the map has none. Returns how many roads took one.
    Each candidate road's SAMPLES points (along its longest ring) look up the nearest segment running along it, in one
    native call for the whole cell (earthfast.ign_join); a road takes the median of what MIN_VOTES or more found."""
    n = len(feats)
    if not n or not rows:
        return 0
    import fast
    fast.require(172, "ign_join")
    old = feats.attrs.get("osm_width")
    col = np.full(n, "", dtype=object) if old is None else np.asarray(old).astype(object)
    classes = feats.attrs.get("class")
    cls = np.asarray(classes).astype(str) if classes is not None else np.full(n, "primary")
    empty = np.array([str(v).strip() in ("", "nan", "None") for v in col], bool)
    cand = np.flatnonzero((np.asarray(feats.kind) == mapgeom.LINE) & empty
                          & np.isin(cls, sorted(THROUGH | SMALL)))
    if len(cand):
        votes, medians = fast._ef.ign_join(
            np.ascontiguousarray(_segments(rows)), np.ascontiguousarray(feats.part_off, np.int64),
            np.ascontiguousarray(feats.ring_off, np.int64), np.ascontiguousarray(feats.coord_off, np.int64),
            np.ascontiguousarray(feats.lonlat, np.float64), np.ascontiguousarray(cand, np.int64), SAMPLES,
            float(GRID), MATCH_M / 111000.0, MATCH_M, SAME_WAY_COS)
    else:
        votes, medians = np.zeros(0, np.int32), np.zeros(0)
    joined = 0
    for i, v, m in zip(cand.tolist(), np.asarray(votes).tolist(), np.asarray(medians).tolist()):
        if v < MIN_VOTES:
            continue
        width = float(m)
        if cls[i] in SMALL and width < SMALL_MIN_M:
            continue
        if cls[i] in THROUGH:
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
