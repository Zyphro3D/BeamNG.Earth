"""France: the real walls, fences and hedges of a street, found along the property lines.

Neither Overture nor OpenStreetMap maps garden walls and hedges in most of France. The cadastre (PCI, Parcellaire
Express) has every property line, and IGN's LiDAR HD (MNH, height above ground) shows what stands on it. Per z14 cell:

- along each parcel edge, the MNH is read every half metre; a run of 3.5 m or more between MIN_M and MAX_M, away from
  the buildings, is a barrier;
- the colour of the aerial photo (BD ORTHO) along it tells a hedge (green) from a wall or fence.

The runs are kept on disk (<cache>/ign_limites/v<VERSION>/<z>/<x>/<y>.npz) and rows() makes them Forest rows: stock
walls (a mesh fence on a low wall under FENCE_MAX_M, a plain wall above it) and shrubs. All open data under Etalab
Licence Ouverte 2.0. Off: BNE_IGN_LIMITES=off. A failure leaves the cell without barriers, as it was.
"""
import io
import json
import logging
import math
import os
import threading
import time

import numpy as np

import ign_bati
import mapcells

log = logging.getLogger("earthd.ign_limites")

VERSION = 1
ON = os.environ.get("BNE_IGN_LIMITES", "on").lower() != "off"
WFS_URL, WFS_LANE = ign_bati.WFS_URL, ign_bati.WFS_LANE
WMS_URL = "https://data.geopf.fr/wms-r/wms"
MNH_LAYER = "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
ORTHO_LAYER = "HR.ORTHOIMAGERY.ORTHOPHOTOS"
PARCELS = "CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle"
MERC_HALF = 20037508.342789244
MNH_PX, ORTHO_PX, BLOCK = 4096, 2048, 1024
GET_TRIES = 3
PAGE, PAGES = 5000, 40
MIN_M, MAX_M = 0.6, 3.5
RUN_M, STEP_M = 3.5, 0.5
GREEN = 4.0                 # G - R of the photo above which a run is a hedge
WALL, FENCE, HEDGE = "frwall", "frfence", "shrub"
FENCE_MAX_M = 1.7
WALL_LEN, WALL_TOP = 3.3, 1.0   # italy_wall_concrete_plain: 3.36 m long, its top 1.0 m over its origin, 1.53 under it
FENCE_LEN, FENCE_TOP = 6.25, 1.59   # italy_fence_mesh_6m_conc_base
HEDGE_STEP = 1.5
WALL_H_MAX = 2.4
PAD_M = 8.0


def merc(lon, lat):
    return (np.radians(lon) * 6378137.0, np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * 6378137.0)


def cell_box(cell):
    size = 2 * MERC_HALF / (1 << cell.z)
    return cell.x * size - MERC_HALF, MERC_HALF - cell.y * size, size


def dilate(a):
    """3 x 3 maximum of a 2D array (bool or float), the edges left as they are."""
    out = a.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                np.maximum(out[1:-1, 1:-1], a[1 + dy:a.shape[0] - 1 + dy, 1 + dx:a.shape[1] - 1 + dx], out=out[1:-1, 1:-1])
    return out


def detect(edges, height, mask, photo, metres_per_px):
    """Barriers along edges ((N, 4) x0 y0 x1 y1 in raster pixels). height: metres (NaN none); mask: True where no
    barrier can be (buildings); photo: RGB of the same area, any size. Returns rows [x0, y0, x1, y1, kind, h]."""
    h = np.nan_to_num(height, nan=0.0)
    ny, nx = h.shape
    wide, blocked = dilate(h), dilate(mask)
    sy, sx = photo.shape[0] / ny, photo.shape[1] / nx
    out = []
    for x0, y0, x1, y1 in edges:
        length = math.hypot(x1 - x0, y1 - y0) * metres_per_px
        if length < RUN_M:
            continue
        n = int(length / STEP_M) + 1
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        ix, iy = xs.astype(np.int64), ys.astype(np.int64)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ok = np.zeros(n, bool)
        v = np.zeros(n)
        v[inside] = wide[iy[inside], ix[inside]]
        ok[inside] = ~blocked[iy[inside], ix[inside]]
        ok &= (v >= MIN_M) & (v <= MAX_M)
        step = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(step == 1), np.flatnonzero(step == -1)):
            if (b - a) * length / n < RUN_M:
                continue
            px = np.clip((xs[a:b] * sx).astype(np.int64), 0, photo.shape[1] - 1)
            py = np.clip((ys[a:b] * sy).astype(np.int64), 0, photo.shape[0] - 1)
            col = photo[py, px].mean(axis=0)
            kind = 2 if col[1] > col[0] + GREEN else 0
            out.append([float(xs[a]), float(ys[a]), float(xs[b - 1]), float(ys[b - 1]), kind, float(np.median(v[a:b]))])
    return out


class Provider:
    def __init__(self, cache_dir, fetcher, bati=None):
        self.dir = os.path.join(cache_dir, "ign_limites", f"v{VERSION}") if cache_dir else None
        self.fetcher, self.bati = fetcher, bati
        self._locks, self._lock, self._mem = {}, threading.Lock(), {}

    def _get(self, url):
        for attempt in range(GET_TRIES):
            try:
                return self.fetcher.get(url, kind="demand", lane=WFS_LANE).body
            except Exception:       # (a 400 that the same request answers a moment later: a whole cell's barriers lost)
                if attempt == GET_TRIES - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))

    def _parcels(self, cell):
        lon0, lat0, lon1, lat1 = cell.bounds()
        rings = []
        for page in range(PAGES):
            url = (f"{WFS_URL}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES={PARCELS}&PROPERTYNAME=geom"
                   f"&BBOX={lon0:.6f},{lat0:.6f},{lon1:.6f},{lat1:.6f},CRS:84&SRSNAME=CRS:84"
                   f"&OUTPUTFORMAT=application/json&COUNT={PAGE}&STARTINDEX={page * PAGE}&SORTBY=gid")
            features = json.loads(self._get(url).decode("utf-8")).get("features") or []
            for ft in features:
                g = ft.get("geometry") or {}
                for poly in ([g.get("coordinates")] if g.get("type") == "Polygon" else g.get("coordinates") or []):
                    rings.extend(poly)
            if len(features) < PAGE:
                break
        return rings

    def _wms(self, cell, layer, fmt, npx):
        mx0, my0, size = cell_box(cell)
        px = size / npx
        out = {}
        for r0 in range(0, npx, BLOCK):
            for c0 in range(0, npx, BLOCK):
                r1, c1 = min(npx, r0 + BLOCK), min(npx, c0 + BLOCK)
                url = (f"{WMS_URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}&STYLES=&CRS=EPSG:3857"
                       f"&BBOX={mx0 + c0 * px:.3f},{my0 - r1 * px:.3f},{mx0 + c1 * px:.3f},{my0 - r0 * px:.3f}"
                       f"&WIDTH={c1 - c0}&HEIGHT={r1 - r0}&FORMAT={fmt}")
                out[(r0, c0, r1, c1)] = self._get(url)
        return out

    def _build(self, cell):
        from PIL import Image, ImageDraw
        rings = self._parcels(cell)
        if not rings:
            return np.zeros((0, 6))
        h = np.full((MNH_PX, MNH_PX), np.nan, np.float32)
        for (r0, c0, r1, c1), body in self._wms(cell, MNH_LAYER, "image/x-bil;bits=32", MNH_PX).items():
            if len(body) == (r1 - r0) * (c1 - c0) * 4:
                b = np.frombuffer(body, "<f4").reshape(r1 - r0, c1 - c0)
                h[r0:r1, c0:c1] = np.where((b > -100) & (b < 200), b, np.nan)
        if not np.isfinite(h).any():
            return np.zeros((0, 6))         # (no lidar here yet)
        photo = np.zeros((ORTHO_PX, ORTHO_PX, 3), np.float32)
        for (r0, c0, r1, c1), body in self._wms(cell, ORTHO_LAYER, "image/jpeg", ORTHO_PX).items():
            try:
                photo[r0:r1, c0:c1] = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"), np.float32)
            except Exception:           # (an XML error instead of a picture: no colour, so not a hedge)
                pass
        mx0, my0, size = cell_box(cell)
        scale = MNH_PX / size

        def pixels(ring):
            a = np.asarray(ring, np.float64)[:, :2]
            x, y = merc(a[:, 0], a[:, 1])
            return np.column_stack(((x - mx0) * scale, (my0 - y) * scale))
        im = Image.new("1", (MNH_PX, MNH_PX), 0)
        draw = ImageDraw.Draw(im)
        for ring, _ in (self.bati.rows(cell) if self.bati is not None else ()):
            draw.polygon([tuple(p) for p in pixels(ring)], fill=1)
        mask = np.asarray(im, bool)
        seen, edges = set(), []
        for ring in rings:
            p = pixels(ring)
            for a, b in zip(p[:-1], p[1:]):
                key = tuple(sorted(((round(a[0], 1), round(a[1], 1)), (round(b[0], 1), round(b[1], 1)))))
                if key not in seen:
                    seen.add(key)
                    edges.append((a[0], a[1], b[0], b[1]))
        lat = cell.center()[0]
        found = detect(edges, h, mask, photo, size / MNH_PX * math.cos(math.radians(lat)))
        out = np.zeros((len(found), 6))
        for i, (x0, y0, x1, y1, kind, height) in enumerate(found):
            out[i] = (x0 / scale + mx0, my0 - y0 / scale, x1 / scale + mx0, my0 - y1 / scale, kind, height)
        return out

    def segments(self, cell):
        """(N, 6): x0 y0 x1 y1 (EPSG:3857 metres), kind (0 wall or fence, 2 hedge), height."""
        k = str(cell)
        with self._lock:
            if k in self._mem:
                return self._mem[k]
            lock = self._locks.setdefault(k, threading.Lock())
        with lock:
            path = os.path.join(self.dir, str(cell.z), str(cell.x), f"{cell.y}.npz") if self.dir else None
            if path and os.path.exists(path):
                with np.load(path, allow_pickle=False) as z:
                    seg = z["seg"]
            else:
                seg = self._build(cell)
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
                    np.savez_compressed(tmp, seg=seg)
                    os.replace(tmp, path)
            with self._lock:
                if len(self._mem) > 32:
                    self._mem.clear()
                self._mem[k] = seg
            return seg


def _panels(ex, ey, fx, fy, kind, h):
    """(group, x, y, yaw, top above ground, height) of the panels of one run in ENU metres."""
    length = math.hypot(fx - ex, fy - ey)
    if length < 1e-6:
        return
    tx, ty = (fx - ex) / length, (fy - ey) / length
    yaw = math.degrees(math.atan2(ty, tx)) % 360.0
    if kind == 2:
        group, step, top, natural = HEDGE, HEDGE_STEP, min(3.5, max(1.0, h)), min(3.5, max(1.0, h))
    elif h < FENCE_MAX_M:
        group, step, top, natural = FENCE, FENCE_LEN, FENCE_TOP, FENCE_TOP
    else:
        group, step, top, natural = WALL, WALL_LEN, min(WALL_H_MAX, h), WALL_TOP
    n = max(1, int(round(length / step)))
    for i in range(n):
        s = (i + .5) * length / n
        yield group, ex + tx * s, ey + ty * s, yaw, top, natural


def rows(frame, provider, box, elev):
    """Forest rows [group, x, y, z, yaw, scale, height, rank] of the barriers whose panel centre is inside box."""
    if provider is None or not ON:
        return []
    x0, y0, x1, y1 = map(float, box)
    la, lo, _ = frame.enu_to_geodetic(np.array([x0, x0, x1, x1]), np.array([y0, y1, y0, y1]), np.zeros(4))
    if not ign_bati.in_france(lo.min(), la.min(), lo.max(), la.max()):
        return []
    try:
        keys = mapcells.cells_for_points(la, lo, pad_lat=0.0001, pad_lon=0.0001, max_cells=16)
    except (ValueError, OverflowError):
        return []
    pending = []
    for key in keys:
        try:
            seg = provider.segments(key)
        except Exception as e:          # (no network, a changed service: no barriers here)
            log.warning("ign_limites %s: %s: %s", key, type(e).__name__, e)
            continue
        if not len(seg):
            continue
        lon = np.degrees(seg[:, [0, 2]] / 6378137.0)
        lat = np.degrees(2 * np.arctan(np.exp(seg[:, [1, 3]] / 6378137.0)) - np.pi / 2)
        e, n, _ = frame.geodetic_to_enu(lat.ravel(), lon.ravel(), np.zeros(lat.size))
        e, n = e.reshape(-1, 2), n.reshape(-1, 2)
        near = ((e.max(1) >= x0 - PAD_M) & (e.min(1) <= x1 + PAD_M) & (n.max(1) >= y0 - PAD_M) & (n.min(1) <= y1 + PAD_M))
        for i in np.flatnonzero(near):
            for row in _panels(e[i, 0], n[i, 0], e[i, 1], n[i, 1], int(seg[i, 4]), float(seg[i, 5])):
                if x0 <= row[1] < x1 and y0 <= row[2] < y1:
                    pending.append(row)
    if not pending:
        return []
    xy = np.array([[p[1], p[2]] for p in pending], np.float64)
    lat, lon, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], np.zeros(len(xy)))
    _, _, z = frame.geodetic_to_enu(lat, lon, np.asarray(elev(lat, lon), np.float64))
    out = []
    for (group, x, y, yaw, top, natural), zz in zip(pending, np.asarray(z, np.float64)):
        lift = 0.0 if group == HEDGE else top - natural
        out.append([group, round(float(x), 2), round(float(y), 2), round(float(zz) + lift, 2), round(float(yaw), 1), 1.0,
                    float(natural), 0.0])
    return out
