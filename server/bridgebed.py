"""The ground under road bridges, out of the elevation model (the user's capture 2026-11-11: a mound of earth under a
deck over a railway; the DEM's surface there is partly the bridge itself).

Under a deck's middle the DEM's height is taken down to the ground either side of the bridge: at each point under the
deck, the line between the DEM's heights LATERAL_M beyond the deck's two edges (across it), never higher than the DEM
was. Within END_KEEP_M of a deck's ends (its abutments, the embankment the road climbs) the DEM stays as it is, and the
change comes in over RAMP_M after that. Every user of the elevation model reads it the same way (BedDem wraps
dem.Dem.sample): the road plans, the decks' solve and their piers, the terrain blocks.

Off: BNE_BRIDGE_BED=off. Pure given the map's cells and the DEM."""
import math
import os
import threading
from collections import OrderedDict

import numpy as np

ON = os.environ.get("BNE_BRIDGE_BED", "on") != "off"
LATERAL_M = 6.0         # m beyond a deck's edge where the ground either side is read
SIDE_M = 1.5            # m beyond a deck's edge still under it (its slab and parapet overhang)
END_KEEP_M = 12.0       # m from a deck's end where the DEM stays (the abutment and its embankment)
RAMP_M = 10.0           # m over which the change comes in past END_KEEP_M
MIN_LEN_M = 2 * END_KEEP_M + 4.0     # a shorter deck keeps the DEM (all abutment)
MIN_ZOOM = 14           # coarser reads (the far landscape) keep the DEM
MAX_SPAN_M = 4000.0     # a read wider than this (a far tile, the horizon) keeps the DEM: a terrain block is 1 km
M_PER_DEG = 111_320.0


def _local(lat0, lon0):
    """(to_xy, to_ll) of a flat frame at (lat0, lon0): metres east / north, good over a deck's length."""
    kx = M_PER_DEG * math.cos(math.radians(lat0))

    def to_xy(la, lo):
        return (np.asarray(lo, np.float64) - lon0) * kx, (np.asarray(la, np.float64) - lat0) * M_PER_DEG

    def to_ll(x, y):
        return lat0 + np.asarray(y, np.float64) / M_PER_DEG, lon0 + np.asarray(x, np.float64) / kx
    return to_xy, to_ll


def footprints(feats):
    """[(lonlat (k, 2), width m)] of the road bridge parts of a cell's features (roadspecs: bridge_specs /
    bridge_parts), each MIN_LEN_M long or more."""
    import roadspecs
    out = []
    if feats is None or not len(feats):
        return out
    specs = roadspecs.bridge_specs(feats)
    for i, sp in enumerate(specs):
        if sp is None:
            continue
        for ring in roadspecs.bridge_parts(feats, i):
            ll = np.asarray(ring, np.float64)[:, :2]
            if len(ll) < 2:
                continue
            to_xy, _ = _local(float(ll[0, 1]), float(ll[0, 0]))
            x, y = to_xy(ll[:, 1], ll[:, 0])
            if float(np.hypot(np.diff(x), np.diff(y)).sum()) >= MIN_LEN_M:
                out.append((ll, float(sp.width)))
    return out


def lower(la, lo, z, prints, sample):
    """z (the DEM's heights at (la, lo), any shape) taken down under the bridges prints [(lonlat, width)];
    sample(la, lo) -> the DEM's heights (unchanged) where the ground either side is read. Returns z itself when no
    point is under a deck's middle."""
    if not prints:
        return z
    la_ = np.asarray(la, np.float64).ravel()
    lo_ = np.asarray(lo, np.float64).ravel()
    zf = np.asarray(z, np.float64).ravel()
    out = None
    for ll, width in prints:
        half = width / 2 + SIDE_M
        reach = half + 1e-3
        pad_la = reach / M_PER_DEG
        pad_lo = reach / (M_PER_DEG * max(0.05, math.cos(math.radians(float(ll[0, 1])))))
        near = ((la_ >= ll[:, 1].min() - pad_la) & (la_ <= ll[:, 1].max() + pad_la) &
                (lo_ >= ll[:, 0].min() - pad_lo) & (lo_ <= ll[:, 0].max() + pad_lo))
        if not near.any():
            continue
        idx = np.nonzero(near)[0]
        to_xy, to_ll = _local(float(ll[0, 1]), float(ll[0, 0]))
        lx, ly = to_xy(ll[:, 1], ll[:, 0])
        px, py = to_xy(la_[idx], lo_[idx])
        seg = np.hypot(np.diff(lx), np.diff(ly))
        cum = np.r_[0.0, np.cumsum(seg)]
        total = float(cum[-1])
        best_d = np.full(len(idx), np.inf)
        best_s = np.zeros(len(idx))
        best_t = np.zeros(len(idx))           # signed offset across (left +)
        best_u = np.zeros((len(idx), 2))      # the segment's direction at the nearest point
        for j in range(len(seg)):
            if seg[j] <= 1e-9:
                continue
            ux, uy = (lx[j + 1] - lx[j]) / seg[j], (ly[j + 1] - ly[j]) / seg[j]
            dx, dy = px - lx[j], py - ly[j]
            a = np.clip(dx * ux + dy * uy, 0.0, seg[j])
            ex, ey = dx - a * ux, dy - a * uy
            d = np.hypot(ex, ey)
            better = d < best_d
            best_d = np.where(better, d, best_d)
            best_s = np.where(better, cum[j] + a, best_s)
            best_t = np.where(better, -ex * uy + ey * ux, best_t)
            best_u[better] = (ux, uy)
        inside = (best_d <= half) & (best_s > END_KEEP_M) & (best_s < total - END_KEEP_M)
        if not inside.any():
            continue
        k = idx[inside]
        s, t, u = best_s[inside], best_t[inside], best_u[inside]
        # the deck's centre line at s, and the ground LATERAL_M beyond each edge, across it
        cx, cy = px[inside] - t * -u[:, 1], py[inside] - t * u[:, 0]
        off = width / 2 + LATERAL_M
        nx, ny = -u[:, 1], u[:, 0]
        lla, llo = to_ll(cx + nx * off, cy + ny * off)
        rla, rlo = to_ll(cx - nx * off, cy - ny * off)
        zz = np.asarray(sample(np.r_[lla, rla], np.r_[llo, rlo]), np.float64)
        zl, zr = zz[:len(k)], zz[len(k):]
        f = np.clip((t + off) / (2 * off), 0.0, 1.0)
        bed = zr + (zl - zr) * f
        w = np.clip((np.minimum(s, total - s) - END_KEEP_M) / RAMP_M, 0.0, 1.0)
        cur = (out if out is not None else zf)[k]
        new = cur - w * np.maximum(cur - bed, 0.0)
        new = np.where(np.isfinite(new), new, cur)
        if np.any(new < cur):
            if out is None:
                out = zf.copy()
            out[k] = new
    if out is None:
        return z
    return out.reshape(np.shape(z)).astype(np.asarray(z).dtype, copy=False)


class BedDem:
    """dem.Dem whose sample() gives the ground under road bridges (lower); everything else passes through. cell_fn(key)
    -> a cell's road features (or None: not there yet: the DEM as it is there)."""

    KEPT = 256          # cells' footprints kept

    def __init__(self, dem, cell_fn):
        self._dem, self._cell_fn = dem, cell_fn
        self._prints = OrderedDict()
        self._lock = threading.Lock()

    def _cell_prints(self, key):
        k = str(key)
        with self._lock:
            got = self._prints.get(k)
            if got is not None:
                self._prints.move_to_end(k)
                return got
        try:
            feats = self._cell_fn(key)
        except Exception:
            return []           # (not cached: asked again)
        if feats is None:
            return []
        got = footprints(feats)
        with self._lock:
            self._prints[k] = got
            while len(self._prints) > self.KEPT:
                self._prints.popitem(last=False)
        return got

    def sample(self, lat, lon, zoom, progress=None):
        base = self._dem.sample(lat, lon, zoom, progress) if progress is not None else self._dem.sample(lat, lon, zoom)
        if not ON:
            return base
        la, lo = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        if la.size == 0 or int(zoom) < MIN_ZOOM:
            return base
        span_la = (float(la.max()) - float(la.min())) * M_PER_DEG
        span_lo = (float(lo.max()) - float(lo.min())) * M_PER_DEG * math.cos(math.radians(float(la.min())))
        if max(span_la, span_lo) > MAX_SPAN_M:
            return base         # (the far landscape's wide reads: no deck to see under there, nor its cells to load)
        import mapcells
        pad = (END_KEEP_M + 50.0) / M_PER_DEG
        keys = mapcells.cells_for_bbox(float(la.min()) - pad, float(lo.min()) - pad,
                                       float(la.max()) + pad, float(lo.max()) + pad)
        prints, seen = [], set()
        for key in keys:
            for p in self._cell_prints(key):
                pid = (tuple(np.round(p[0][0], 7)), tuple(np.round(p[0][-1], 7)), len(p[0]), p[1])
                if pid not in seen:         # (a bridge in several cells: once, lowered once)
                    seen.add(pid)
                    prints.append(p)
        if not prints:
            return base
        return lower(la, lo, base, prints, lambda a, b: self._dem.sample(a, b, zoom))

    def __getattr__(self, name):
        return getattr(self._dem, name)
