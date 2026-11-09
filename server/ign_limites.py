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
import concurrent.futures
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
import revisit

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
MNH_BLOCK = 512             # pixels (220 m) a LiDAR request: the unit read only round the buildings
WMS_THREADS = 4
GET_TRIES = 3
EMPTY_SEG, EMPTY_ROOFS = np.zeros((0, 6)), np.zeros((0, 3))
ROOF_MIN_PX = 40            # pixels of height (0.43 m): under this a house is too small to tell
ROOF_ERODE_PX = 3           # pixels (1.3 m) in from the footprint: only the roof itself is read
ROOF_FLAT_M = 0.7           # m of relief (10th to 90th percentile) under which a roof is flat
ROOF_STRIP = 0.25           # share of the half width round the ridge
ROOF_MID, ROOF_END = 0.3, 0.75   # share of the half length: the ridge's middle, and its ends
ROOF_HIP_DROP = 0.4         # share of the relief the ridge falls by at its ends for a hipped roof
CLAIM_MAX_S = 600.0
PAGE, PAGES = 5000, 40
MIN_M, MAX_M = 0.6, 3.5
RUN_M, STEP_M = 3.5, 0.5
GREEN = 4.0                 # G - R of the photo above which a run is a hedge
WALL, FENCE, HEDGE = "frwall", "frfence", "shrub"
FENCE_MAX_M = 1.7
WALL_LEN, WALL_TOP = 3.3, 1.0   # italy_wall_concrete_plain: 3.36 m long, its top 1.0 m over its origin, 1.53 under it
FENCE_LEN, FENCE_TOP = 6.25, 1.59   # italy_fence_mesh_6m_conc_base
HEDGE_STEP = 1.9
HEDGE_MAX_M = 2.4           # m: a garden hedge; a taller reading is the tree or the wall behind it
WALL_H_MAX = 2.4
WALL_MIN_RUN_M = 12.0       # m: a shorter "wall" on its own is a parked van or a skip seen by the LiDAR
PAD_M = 8.0


def blocks_near(points, npx, block=None):
    """The (row, column) blocks of an npx grid that hold a point (x, y in pixels), and the blocks round them."""
    block = block or BLOCK
    n = int(math.ceil(npx / block))
    out = set()
    for x, y in points:
        r, c = int(y // block), int(x // block)
        for a in range(max(0, r - 1), min(n, r + 2)):
            for b in range(max(0, c - 1), min(n, c + 2)):
                out.add((a, b))
    return out


def roof_shape(h, mask):
    """0 flat, 1 gabled, 2 hipped from the measured heights of one roof (h, mask: arrays of a building's pixels), or None
    with too little to tell. A pitched roof's ridge runs along its long axis: a gable keeps the ridge's height to the end
    of the house, a hipped roof falls to the eaves there."""
    inner = np.array(mask, bool)
    for _ in range(ROOF_ERODE_PX):      # (at its edge the footprint's pixels fall to the ground: every flat roof looked pitched)
        inner[[0, -1], :] = False       # (first: np.roll wraps round, and a footprint can touch its crop)
        inner[:, [0, -1]] = False
        inner = inner & np.roll(inner, 1, 0) & np.roll(inner, -1, 0) & np.roll(inner, 1, 1) & np.roll(inner, -1, 1)
    ys, xs = np.nonzero(inner & np.isfinite(h))
    if len(xs) < ROOF_MIN_PX:
        return None
    z = h[ys, xs]
    low, high = np.percentile(z, 10), np.percentile(z, 90)
    if high - low < ROOF_FLAT_M:
        return 0
    x, y = xs - xs.mean(), ys - ys.mean()
    w, v = np.linalg.eigh(np.cov(np.vstack((x, y))))
    long_axis = v[:, int(np.argmax(w))]
    s = x * long_axis[0] + y * long_axis[1]
    t = -x * long_axis[1] + y * long_axis[0]
    half_s, half_t = max(np.abs(s).max(), 1e-6), max(np.abs(t).max(), 1e-6)
    strip = np.abs(t) < ROOF_STRIP * half_t
    u = s / half_s
    mid, ends = strip & (np.abs(u) < ROOF_MID), strip & (np.abs(u) > ROOF_END)
    if mid.sum() < 3 or ends.sum() < 3:
        return None
    return 2 if (np.median(z[mid]) - np.median(z[ends])) / (high - low) > ROOF_HIP_DROP else 1


def roof_of_ring(h, pixels):
    """roof_shape of a footprint given in pixels of the height grid."""
    from PIL import Image, ImageDraw
    x0, y0 = np.floor(pixels.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(pixels.max(axis=0)).astype(int) + 2
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), h.shape[1]), min(int(y1), h.shape[0])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    im = Image.new("1", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(im).polygon([(float(a) - x0, float(b) - y0) for a, b in pixels], fill=1)
    return roof_shape(h[y0:y1, x0:x1], np.asarray(im, bool))


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
        self._later, self._asked = None, set()      # (fetch_later: one background download at a time, per process)

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

    def _wms(self, cell, layer, fmt, npx, only=None, block=None):
        """{(r0, c0, r1, c1): body} of the grid's blocks (only: those (row, column) of them), a few at a time."""
        block = block or BLOCK
        mx0, my0, size = cell_box(cell)
        px = size / npx
        boxes = [(r0, c0, min(npx, r0 + block), min(npx, c0 + block)) for r0 in range(0, npx, block)
                 for c0 in range(0, npx, block) if only is None or (r0 // block, c0 // block) in only]

        def one(box):
            r0, c0, r1, c1 = box
            return box, self._get(f"{WMS_URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}&STYLES=&CRS=EPSG:3857"
                                  f"&BBOX={mx0 + c0 * px:.3f},{my0 - r1 * px:.3f},{mx0 + c1 * px:.3f},{my0 - r0 * px:.3f}"
                                  f"&WIDTH={c1 - c0}&HEIGHT={r1 - r0}&FORMAT={fmt}")
        with concurrent.futures.ThreadPoolExecutor(WMS_THREADS) as pool:
            return dict(pool.map(one, boxes))

    def _build(self, cell):
        """(segments, roofs) of the cell. The LiDAR is read only round the buildings: walls and hedges are wanted beside
        them, and it is nearly all of what a cell downloads (64 MB)."""
        from PIL import Image, ImageDraw
        rings = self._parcels(cell)
        if not rings:
            return EMPTY_SEG, EMPTY_ROOFS
        mx0, my0, size = cell_box(cell)
        scale = MNH_PX / size

        def pixels(ring):
            a = np.asarray(ring, np.float64)[:, :2]
            x, y = merc(a[:, 0], a[:, 1])
            return np.column_stack(((x - mx0) * scale, (my0 - y) * scale))
        houses = [ring for ring, _ in self.bati.rows(cell)] if self.bati is not None else None
        if houses is not None and not houses:
            return EMPTY_SEG, EMPTY_ROOFS
        only = None if houses is None else blocks_near([pixels(r).mean(axis=0) for r in houses], MNH_PX, MNH_BLOCK)
        h = np.full((MNH_PX, MNH_PX), np.nan, np.float32)
        for (r0, c0, r1, c1), body in self._wms(cell, MNH_LAYER, "image/x-bil;bits=32", MNH_PX, only, MNH_BLOCK).items():
            if len(body) == (r1 - r0) * (c1 - c0) * 4:
                b = np.frombuffer(body, "<f4").reshape(r1 - r0, c1 - c0)
                h[r0:r1, c0:c1] = np.where((b > -100) & (b < 200), b, np.nan)
        if not np.isfinite(h).any():
            return EMPTY_SEG, EMPTY_ROOFS   # (no lidar here yet)
        photo = np.zeros((ORTHO_PX, ORTHO_PX, 3), np.float32)
        for (r0, c0, r1, c1), body in self._wms(cell, ORTHO_LAYER, "image/jpeg", ORTHO_PX).items():
            try:
                photo[r0:r1, c0:c1] = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"), np.float32)
            except Exception:           # (an XML error instead of a picture: no colour, so not a hedge)
                pass
        im = Image.new("1", (MNH_PX, MNH_PX), 0)
        draw = ImageDraw.Draw(im)
        roofs = []
        for ring in houses or ():
            p = pixels(ring)
            draw.polygon([tuple(q) for q in p], fill=1)
            shape = roof_of_ring(h, p)
            if shape is not None:
                lon, lat = np.asarray(ring, np.float64)[:, :2].mean(axis=0)
                roofs.append((lon, lat, shape))
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
        return out, np.array(roofs, np.float64).reshape(-1, 3)

    def _file(self, cell):
        return os.path.join(self.dir, str(cell.z), str(cell.x), f"{cell.y}.npz") if self.dir else None

    def ready(self, cell):
        """Are the cell's barriers at hand (a tile can read them without the network)?"""
        with self._lock:
            if str(cell) in self._mem:
                return True
        path = self._file(cell)
        return bool(path) and os.path.exists(path)

    def _claim(self, cell):
        """This process builds the cell (the road processes each have a provider), unless another one is at it. A claim
        left by a process that died stops counting after CLAIM_MAX_S."""
        path = self._file(cell)
        if not path:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        claim = path + ".claim"
        try:
            if time.time() - os.path.getmtime(claim) > CLAIM_MAX_S:
                os.remove(claim)
        except OSError:
            pass
        try:
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except OSError:
            return False
        return claim

    def fetch_later(self, cell):
        """Downloads and works out the cell in the background, once: a road tile never waits for it (its first visit took
        over 75 s a tile, the road missing all that while)."""
        k = str(cell)
        with self._lock:
            if k in self._asked:
                return
            self._asked.add(k)
            if self._later is None:
                self._later = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="ign_limites")
            pool = self._later

        def job():
            claim = self._claim(cell)
            if claim is False:
                return
            try:
                self.segments(cell)
            except Exception as e:      # (no network, a changed service: asked again by the next tile that wants it)
                log.warning("ign_limites %s: %s: %s", cell, type(e).__name__, e)
                with self._lock:
                    self._asked.discard(k)
            finally:
                if claim:
                    try:
                        os.remove(claim)
                    except OSError:
                        pass
        pool.submit(job)

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
                seg, roofs = self._build(cell)
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
                    np.savez_compressed(tmp, seg=seg, roofs=roofs)
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
        group, step, top, natural = HEDGE, HEDGE_STEP, min(HEDGE_MAX_M, max(1.0, h)), min(HEDGE_MAX_M, max(1.0, h))
    elif h < FENCE_MAX_M:
        group, step, top, natural = FENCE, FENCE_LEN, FENCE_TOP, FENCE_TOP
    else:
        if length < WALL_MIN_RUN_M:
            return
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
        if not provider.ready(key):     # (this tile goes without, and is not kept: its next build has them)
            provider.fetch_later(key)
            revisit.fault()
            continue
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
