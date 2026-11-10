"""Procedural buildings: their surroundings (split out of buildings.py): shared walls, building parts, the roads,
bridges, decks, tunnels and water near a footprint.
"""
import math
import numpy as np
import fast
import mapgeom
import ships
from bldbase import (
    ACROSS_W,
    DECK_INSIDE_M,
    DECK_REACH_M,
    DECK_SEA_M,
    DECK_STAND_M,
    FLOOR_M,
    GARAGE_TAGS,
    ML_BRIDGE_M,
    ML_CLIFF_DEG,
    ML_FAR_FROM_WAY_M,
    ML_MEDIAN_M,
    ML_ROAD_EDGE_M,
    ML_ROAD_M,
    ML_STEEP_DEG,
    PARTS_COVER_MIN_H,
    PARTS_COVER_SAMPLES,
    PARTS_GROUND_M,
    PASSAGE_BOTTOM_M,
    PASSAGE_CLEAR,
    PASSAGE_H,
    PASSAGE_W,
    PORTAL_SKIP_CLASSES,
    THROUGH_CLASSES,
    _area2,
    clean_ring,
    oriented_box,
)
from bldmesh import (
    _inside,
)


def _shared_walls(rings, tol=0.6):
    """Per ring, the number of other rings with a vertex within tol of one of its edges (row houses)."""
    n = len(rings)
    cnt = np.zeros(n, np.int64)
    if n < 2:
        return cnt
    cents = np.array([r.mean(0) for r in rings])
    rads = np.array([np.hypot(*(r - r.mean(0)).T).max() for r in rings])
    order = np.argsort(cents[:, 0])
    fast.require(1, "shared_walls")
    offs = np.zeros(n + 1, np.int64)
    offs[1:] = np.cumsum([len(r) for r in rings])
    coords = np.ascontiguousarray(np.concatenate(rings)[:, :2], np.float64)
    return np.asarray(fast._ef.shared_walls(coords, offs, np.ascontiguousarray(cents, np.float64),
                                            np.ascontiguousarray(rads, np.float64), order.astype(np.int64),
                                            float(tol)), np.int64)

def _part_heights(a, i):
    """A building part's (own height, its bottom) from its attributes, or None without a height."""
    top = a["height"][i] if "height" in a else np.nan
    mh = a["min_height"][i] if "min_height" in a else np.nan
    mh = float(mh) if np.isfinite(mh) and mh > 0 else 0.0
    if not np.isfinite(top) or top <= mh + 1.0:
        nf, f0 = (a[k][i] if k in a else np.nan for k in ("num_floors", "min_floor"))
        if np.isfinite(nf) and nf > 0:
            f0 = float(f0) if np.isfinite(f0) and f0 > 0 else 0.0
            mh = mh or f0 * FLOOR_M
            return max(FLOOR_M, (float(nf) - f0) * FLOOR_M), mh
        return None
    return float(top) - mh, mh


def tunnel_lines(frame, road_layers, x0, y0, margin=50.0, size=256.0):
    """Polylines (k, 2) relative to (x0, y0) of the road tunnels near the tile (roads.tunnel_parts; for
    road_through: a building a tunnel runs into is its portal building, its floor over the tube), but those of
    PORTAL_SKIP_CLASSES."""
    import roads as roadsmod
    out = []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        ts = roadsmod.tunnel_specs(f)
        for i in range(len(f.kind)):
            if ts[i] is None or ts[i].cls in PORTAL_SKIP_CLASSES:
                continue
            for ring in roadsmod.tunnel_parts(f, i):
                e, n = mapgeom.enu_xy(f, frame, ring)
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                out.append(xy)
    return out


def garage_rings(frame, infra_layers, x0, y0, margin=50.0, size=256.0):
    """Outer rings (k, 2) relative to (x0, y0) of the mapped multi-storey car parks near the tile (Overture
    infrastructure, class parking, OSM parking=multi-storey). OSM often keeps the car park's tags on its own area
    and the building beside it as building=yes or =garage (a shed), so the class alone missed 25 of downtown
    Baltimore's 40 garages (2026-10-17, the user's 'parking garages are solid boxes')."""
    import json
    out, seen = [], set()
    for f in infra_layers:
        if f is None or len(f.kind) == 0:
            continue
        a = f.attrs
        cls, tags = a.get("class"), a.get("tags")
        if cls is None or tags is None:
            continue
        for i in np.flatnonzero((cls.astype(str) == "parking") & (f.kind == 3)):
            t = str(tags[i])
            if '"parking"' not in t:
                continue
            try:
                if str(json.loads(t).get("parking", "")).lower() not in GARAGE_TAGS:
                    continue
            except (ValueError, AttributeError):
                continue
            fid = f.ids[i]
            if fid in seen:
                continue
            seen.add(fid)
            for part in f.parts(i):
                if not part or len(part[0]) < 3:
                    continue
                e, n = mapgeom.enu_xy(f, frame, part[0])
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                out.append(xy)
    return out


def support_rings(frame, infra_layers, x0, y0, margin=50.0, size=256.0):
    """[(ring (k, 2) relative to (x0, y0), paint)] of the mapped bridge supports near the tile (Overture
    infrastructure, class bridge_support: OSM bridge:support towers and piers, their building:colour as a
    bridgetypes paint, grey without one). The Golden Gate Bridge's towers are mapped as building parts over them and
    were drawn as a 225 m glass tower (2026-10-18)."""
    import json
    import bridgetypes
    out, seen = [], set()
    for f in infra_layers:
        if f is None or len(f.kind) == 0:
            continue
        a = f.attrs
        cls, tags = a.get("class"), a.get("tags")
        if cls is None:
            continue
        cs = cls.astype(str)
        # a bridge's supports, and its own outlines mapped as buildings (Tower Bridge's high walkways: building:colour
        # lightblue, 45 m); not a bare bridge outline (Ponte Vecchio's shops stand on one)
        three_d = f.__dict__.get("_three_d")         # (once per cell's features: every tile of it asked)
        if three_d is None:
            three_d = f._three_d = (np.array(['"building' in str(t) for t in tags], bool) if tags is not None
                                    else np.zeros(len(cs), bool))
        for i in np.flatnonzero(((cs == "bridge_support") | ((cs == "bridge") & three_d)) & (f.kind == 3)):
            fid = f.ids[i]
            if fid in seen:
                continue
            seen.add(fid)
            pt = None
            if tags is not None and "colour" in str(tags[i]):
                try:
                    td = json.loads(str(tags[i]))
                    pt = bridgetypes.paint(td.get("building:colour") or td.get("colour"))
                except ValueError:
                    pt = None
            for part in f.parts(i):
                if not part or len(part[0]) < 3:
                    continue
                e, n = mapgeom.enu_xy(f, frame, part[0])
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                out.append((xy, pt or "grey"))
    return out


def _on_support(ring, supports):
    """The paint of the bridge support (support_rings) holding the footprint's centre, else None."""
    if not supports:
        return None
    from polygon import Path
    c = ring.mean(0)
    for r, pt in supports:
        if r[:, 0].min() <= c[0] <= r[:, 0].max() and r[:, 1].min() <= c[1] <= r[:, 1].max() \
                and Path(r).contains_points(c[None])[0]:
            return pt
    return None


def _covers_garage(ring, garages):
    """A footprint inside a mapped car park's area (its centre in it) or holding the car park's centre."""
    c = ring.mean(0)
    for g in garages:
        if _inside(g, c) or _inside(ring, g.mean(0)):
            return True
    return False


class RoadLines(list):
    """road_lines' polylines, a list as before, with .half: each line's drawn half-width (m; roads.road_specs' rural
    width / 2, None where it has no spec): _ml_false's on-the-lanes test."""
    def __init__(self, *a):
        super().__init__(*a)
        self.half = []


def road_lines(frame, road_layers, x0, y0, margin=50.0, size=256.0):
    """Polylines (k, 2) relative to (x0, y0) of the public roads on the ground near the tile (for road_through), a
    RoadLines (their drawn half-widths in .half)."""
    import roads as roadsmod
    out = RoadLines()
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        cls = f.attrs.get("class")
        if cls is None:
            continue
        bb = f.boxes(frame)             # (only the roads near the tile: the test below, all at once)
        near = np.flatnonzero((bb[:, 2] - x0 >= -margin) & (bb[:, 0] - x0 <= size + margin)
                              & (bb[:, 3] - y0 >= -margin) & (bb[:, 1] - y0 <= size + margin))
        specs = None
        for i in near.tolist():
            if str(cls[i]) not in THROUGH_CLASSES:
                continue
            if specs is None:
                specs = roadsmod.road_specs(f)      # (remembered on the layer: the road tiles ask for them anyway)
            sp = specs[i] if i < len(specs) else None
            half = float(sp.width) / 2 if sp is not None and np.isfinite(sp.width) and sp.width > 0 else None
            if roadsmod.level_ranges(f, i):
                # its pieces on the ground (a motorway segment with a short overpass in it was left out whole, and the
                # lorries machine-learned as buildings on its lanes stayed: the A7, GitHub #239; PR #71's finding)
                lls = [np.asarray(p, np.float64) for p in roadsmod.ground_parts(f, i)]
            else:
                a = f.coord_off[f.ring_off[f.part_off[i]]]
                b = f.coord_off[f.ring_off[f.part_off[i + 1]]]
                lls = [f.lonlat[a:b]]
            for ll in lls:
                if len(ll) < 2:
                    continue
                e, n = mapgeom.enu_xy(f, frame, ll)
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                out.append(xy)
                out.half.append(half)
    return out


def _clip_half(ring, p0, nrm, keep_ge):
    """The part of the ring (k, 2) on one side of the line through p0 with normal nrm (Sutherland-Hodgman)."""
    out = []
    pts = list(ring[:-1]) if np.allclose(ring[0], ring[-1]) else list(ring)
    side = [float((q - p0) @ nrm) for q in pts]
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        sa, sb = side[k], side[(k + 1) % len(pts)]
        ina = (sa >= 0) == keep_ge or sa == 0
        if ina:
            out.append(a)
        if (sa > 0) != (sb > 0) and sa != 0 and sb != 0:
            t = sa / (sa - sb)
            out.append(a + t * (b - a))
    if len(out) < 3:
        return None
    r = np.array(out + [out[0]], np.float64)
    return r if _area2(r) / 2 >= 2.0 else None


def on_deck(ring, decks, ground):
    """(over the lanes, deck height) when footprint ring (k, 2) has its centre on a deck (decks: [(xy (k, 2), z (k,),
    width m)], the same frame) standing DECK_STAND_M or more over ground (m, its highest ground, DECK_SEA_M at most:
    over water), else None. A building
    on the old Sunshine Skyway span kept as a fishing pier (its bait shop) was raised from the seabed: its top 0.5 m
    stood over the deck (2026-10-27). Over the lanes: a corner more than DECK_INSIDE_M inside the road's edge. Pure."""
    r = np.asarray(ring, np.float64)[:, :2]
    c = r.mean(0)
    best = None
    for d in decks:
        if len(d) < 3 or len(d[0]) < 2:
            continue
        xy, z, w = np.asarray(d[0], np.float64), np.asarray(d[1], np.float64), float(d[2])
        a, b = xy[:-1], xy[1:]
        ab = b - a
        t = np.clip(((c - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        dist = np.hypot(*(a + t[:, None] * ab - c).T)
        k = int(np.argmin(dist))
        half = w / 2 + DECK_REACH_M         # (a pier's map width is its lane: its deck reaches past it)
        if dist[k] > half:
            continue
        zd = float(z[k] + t[k] * (z[k + 1] - z[k]))
        if zd - ground < DECK_STAND_M:
            continue
        if best is None or dist[k] < best[0]:
            best = (dist[k], zd, a[k], ab[k] / max(float(np.hypot(*ab[k])), 1e-9), w)
    if best is None or ground > DECK_SEA_M:
        return None
    _, zd, p, u, w = best
    off = (r - p) @ np.array([-u[1], u[0]])            # (signed: a footprint across the lanes has corners both sides)
    inner = w / 2 - DECK_INSIDE_M
    return bool(off.min() < inner and off.max() > -inner), zd


def _passages(rings, attrs, looks, mins, bridges_xy, decks=None, ground_at=None):
    """Buildings a road bridge runs through split round a PASSAGE_W wide way (Tower Bridge's road ran into its towers'
    stone and its Southern Gate, the user 2026-10-18: "the buildings are still in the middle of the bridge", "this
    building is still blocking the bridge"). With the decks' heights (decks: [(xy (k, 2), z (k,))] relative to the
    tile, ground_at(c (2,)) -> the ground under a footprint's centre): only where the deck passes through the
    building's height (a building under a viaduct stays whole): solid under the deck, the way PASSAGE_CLEAR over it,
    the whole footprint above. Without them, a bridge's own structure (bridge_part / bridge_masonry) starting within
    PASSAGE_BOTTOM_M gets the way PASSAGE_H over its bottom (its deck: the part starts at the road)."""
    R, A, L, M = [], [], [], []
    lines = [d[0] for d in decks] if decks else bridges_xy
    for ring, at, look, mh in zip(rings, attrs, looks, mins):
        bridge_own = bool(at.get("bridge_part") or at.get("bridge_masonry"))
        if not lines or road_through(ring, lines) < 3.0 or (not decks and (not bridge_own or mh > PASSAGE_BOTTOM_M)):
            R.append(ring), A.append(at), L.append(look), M.append(mh)
            continue
        c = ring.mean(0)
        best = None
        for k, ln in enumerate(lines):
            for q in range(len(ln) - 1):
                a, b = ln[q], ln[q + 1]
                ab = b - a
                t = float(np.clip((c - a) @ ab / max(ab @ ab, 1e-9), 0, 1))
                d = float(np.hypot(*(a + t * ab - c)))
                if best is None or d < best[0]:
                    zd = None
                    if decks:
                        z = decks[k][1]
                        zd = float(z[q] + t * (z[q + 1] - z[q]))
                    best = (d, a + t * ab, ab / max(float(np.hypot(*ab)), 1e-9), zd)
        _, p0, u, zd = best
        nrm = np.array([-u[1], u[0]])
        h = at.get("height")
        own = look is not None          # (a building's part carries its own height over its bottom, a building its top)
        top = (mh + float(h) if own else float(h)) if h is not None and np.isfinite(h) else mh + PASSAGE_H + 3.0
        if zd is not None and ground_at is not None and np.isfinite(zd):
            dh = zd - float(ground_at(c))           # the deck's height over the building's ground
            if top <= dh + 1.0 or mh >= dh + PASSAGE_CLEAR:
                R.append(ring), A.append(at), L.append(look), M.append(mh)
                continue                            # under the deck, or over its way
            low, cut = max(mh, dh - 1.5), dh + PASSAGE_CLEAR
        else:
            low, cut = mh, mh + PASSAGE_H
        cut = min(cut, top)
        if low > mh + 0.5:                          # solid under the deck
            R.append(ring), A.append(dict(at, height=(low - mh) if own else low)), L.append(look), M.append(mh)
        for sgn in (1, -1):
            piece = _clip_half(ring, p0 + nrm * sgn * PASSAGE_W / 2, nrm * sgn, True)
            if piece is not None and cut > low + 0.5:
                R.append(piece), A.append(dict(at, height=(cut - low) if own else cut)), L.append(look), M.append(low)
        if top > cut + 1.0:
            R.append(ring), A.append(dict(at, height=(top - cut) if own else top)), L.append(look), M.append(cut)
    return R, A, L, M


def _line_dist(p, lines):
    """Distance from point p (2,) to the nearest of the polylines lines (k, 2), or inf."""
    best = np.inf
    for ln in lines:
        a, b = ln[:-1], ln[1:]
        d = b - a
        t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
        q = a + t[:, None] * d
        best = min(best, float(np.hypot(*(q - p).T).min()))
    return best


def _on_lanes(c, roads_xy):
    """Does point c (2,) lie on a public road's drawn surface: within its half-width + ML_ROAD_EDGE_M of its centre
    line (roads_xy.half, road_lines), and within ML_ROAD_M of any (a plain list without widths: that alone)?"""
    halves = getattr(roads_xy, "half", None)
    for k, ln in enumerate(roads_xy):
        h = halves[k] if halves is not None and k < len(halves) else None
        reach = max(ML_ROAD_M, h + ML_ROAD_EDGE_M) if h is not None else ML_ROAD_M
        if len(ln) < 2 or ln[:, 0].min() > c[0] + reach or ln[:, 0].max() < c[0] - reach or \
                ln[:, 1].min() > c[1] + reach or ln[:, 1].max() < c[1] - reach:
            continue
        if _line_dist(c, [ln]) <= reach:
            return True
    return False


def _nearest_on(p, ln):
    """(distance, unit direction of the segment) of polyline ln (k, 2) nearest point p (2,)."""
    a, b = ln[:-1], ln[1:]
    d = b - a
    t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
    q = a + t[:, None] * d
    dist = np.hypot(*(q - p).T)
    k = int(np.argmin(dist))
    u = d[k] / max(float(np.hypot(*d[k])), 1e-9)
    return float(dist[k]), q[k], u


def _in_median(c, roads_xy):
    """Does point c (2,) lie between two parallel roads each within ML_MEDIAN_M, on opposite sides of it?"""
    near = []
    for ln in roads_xy:
        if len(ln) < 2 or ln[:, 0].min() > c[0] + ML_MEDIAN_M or ln[:, 0].max() < c[0] - ML_MEDIAN_M or \
                ln[:, 1].min() > c[1] + ML_MEDIAN_M or ln[:, 1].max() < c[1] - ML_MEDIAN_M:
            continue
        dist, q, u = _nearest_on(c, ln)
        if dist <= ML_MEDIAN_M:
            near.append((q - c, u))
    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            (vi, ui), (vj, uj) = near[i], near[j]
            if abs(float(ui @ uj)) > 0.9 and float(vi @ vj) < 0:        # parallel, one on each side
                return True
    return False


def _ml_false(ring, bridges_xy, water_xy, roads_xy=None):
    """A machine-learned footprint (Microsoft's, Google's) that is a bridge deck, a boat or a vehicle: a road bridge
    runs through it, its centre lies on mapped water (Tower Bridge's river: barges as brick buildings, 2026-10-18), on
    a public road's lanes (_on_lanes: a lorry on a Norwegian trunk road stood as a white building over the lanes,
    GitHub #2; one in I-280's fast lane, ML_ROAD_EDGE_M) or in a dual carriageway's median (_in_median, GitHub #90)."""
    if bridges_xy and road_through(ring, bridges_xy) >= ML_BRIDGE_M:
        return True
    if roads_xy and (_on_lanes(ring.mean(0), roads_xy) or _in_median(ring.mean(0), roads_xy)):
        return True
    if water_xy:
        from polygon import Path
        c = ring.mean(0)
        for w, holes in water_xy:
            if w[:, 0].min() <= c[0] <= w[:, 0].max() and w[:, 1].min() <= c[1] <= w[:, 1].max() \
                    and Path(w).contains_points(c[None])[0] \
                    and not any(Path(h).contains_points(c[None])[0] for h in holes):    # (an island: land)
                return True
    return False


def _ml_steep(ring, g, ways_xy):
    """Is a machine-learned footprint (ring, relative to the tile's corner; g its ground at each corner) on ground too
    steep for it (ML_*)? The slope of the plane through its corners; the distance from its centre to the nearest
    mapped way (ways_xy: way_lines, the same corner). Pure."""
    r = np.asarray(ring, np.float64)[:, :2]
    g = np.asarray(g, np.float64)
    if len(r) < 3 or len(g) != len(r):
        return False
    A = np.c_[r - r.mean(0), np.ones(len(r))]
    try:
        (a, b, _), *_ = np.linalg.lstsq(A, g, rcond=None)
    except np.linalg.LinAlgError:
        return False
    slope = math.degrees(math.atan(math.hypot(a, b)))
    if slope > ML_CLIFF_DEG:
        return True
    if slope <= ML_STEEP_DEG:
        return False
    return not ways_xy or _line_dist(r.mean(0), ways_xy) > ML_FAR_FROM_WAY_M


def way_lines(frame, road_layers, x0, y0, margin=ML_FAR_FROM_WAY_M + 20.0, size=256.0):
    """Polylines (k, 2) relative to (x0, y0) of every mapped way near the tile (roads of any class, tracks, paths,
    steps: _ml_steep's access test)."""
    out = []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        bb = f.boxes(frame)
        near = np.flatnonzero((bb[:, 2] - x0 >= -margin) & (bb[:, 0] - x0 <= size + margin)
                              & (bb[:, 3] - y0 >= -margin) & (bb[:, 1] - y0 <= size + margin))
        for i in near.tolist():
            a = f.coord_off[f.ring_off[f.part_off[i]]]
            b = f.coord_off[f.ring_off[f.part_off[i + 1]]]
            ll = f.lonlat[a:b]
            if len(ll) < 2:
                continue
            e, n = mapgeom.enu_xy(f, frame, ll)
            out.append(np.stack([e - x0, n - y0], 1))
    return out


def bridge_lines(frame, road_layers, x0, y0, margin=50.0, size=256.0):
    """Polylines (k, 2) relative to (x0, y0) of the road bridges near the tile (their bridge ranges)."""
    import roads as roadsmod
    out = []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        bs = roadsmod.bridge_specs(f)
        for i in range(len(f.kind)):
            if bs[i] is None:
                continue
            for ring in roadsmod.bridge_parts(f, i):
                e, n = mapgeom.enu_xy(f, frame, ring)
                xy = np.stack([e - x0, n - y0], 1)
                if len(xy) < 2 or xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or \
                        xy[:, 1].max() < -margin or xy[:, 1].min() > size + margin:
                    continue
                out.append(xy)
    return out


def water_rings(frame, water_layers, x0, y0, margin=50.0, size=256.0):
    """[(outer ring (k, 2), [hole rings])] relative to (x0, y0) of the mapped water areas (sea, rivers, lakes, docks)
    near the tile (their holes: islands)."""
    out = []
    for f in water_layers:
        if f is None or len(f.kind) == 0:
            continue
        for i in np.flatnonzero(f.kind == 3):
            for part in f.parts(i):
                if not part or len(part[0]) < 4:
                    continue
                e, n = mapgeom.enu_xy(f, frame, part[0])
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                holes = []
                for h in part[1:]:
                    if len(h) >= 4:
                        he, hn = mapgeom.enu_xy(f, frame, h)
                        holes.append(np.stack([he - x0, hn - y0], 1))
                out.append((xy, holes))
    return out


_BOXES = {}                     # id(list of polylines) -> (that list, its boxes (n, 4)): the last few lists asked


def line_boxes(lines):
    """(n, 4) x0 y0 x1 y1 of each polyline (k, 2+) of the list, made once per list (a tile's buildings test the same
    lists of roads one footprint after another: ~1.2M small numpy min / max calls a Baltimore tile set)."""
    got = _BOXES.get(id(lines))
    if got is not None and got[0] is lines:
        return got[1]
    b = np.array([(ln[:, 0].min(), ln[:, 1].min(), ln[:, 0].max(), ln[:, 1].max()) for ln in lines],
                 np.float64).reshape(-1, 4)
    if len(_BOXES) >= 16:
        _BOXES.clear()
    _BOXES[id(lines)] = (lines, b)      # (the list kept: its id cannot be another list's meanwhile)
    return b


def road_through(ring, lines, step=2.0):
    """Metres of the polylines lines (k, 2) inside ring (sampled every step m)."""
    lo, hi = ring.min(0), ring.max(0)
    b = line_boxes(lines)
    near = np.flatnonzero((b[:, 2] >= lo[0]) & (b[:, 0] <= hi[0]) & (b[:, 3] >= lo[1]) & (b[:, 1] <= hi[1]))
    if not len(near):
        return 0.0
    fast.require(74, "road_through")         # (the Rust loop: every sample's even-odd test there)
    sel = [np.asarray(lines[li], np.float64)[:, :2] for li in near]
    starts = np.r_[0, np.cumsum([len(ln) for ln in sel])].astype(np.int64)
    return float(fast._ef.road_through(np.ascontiguousarray(np.asarray(ring, np.float64)[:, :2]),
                                       np.ascontiguousarray(np.concatenate(sel)), starts, float(step)))


def road_across(ring, lines, max_w=ACROSS_W):
    """Does one of the polylines lines (k, 2) cross the narrow footprint ring (at most max_w wide) along its length,
    over its long middle line?"""
    c, u, hl, hw = oriented_box(ring)
    if 2 * hw > max_w or hl <= hw:
        return False
    a, b = c - u * hl, c + u * hl
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    bx = line_boxes(lines)
    for li in np.flatnonzero((bx[:, 2] >= lo[0]) & (bx[:, 0] <= hi[0]) & (bx[:, 3] >= lo[1]) & (bx[:, 1] <= hi[1])):
        ln = lines[li]
        p, q = ln[:-1], ln[1:]
        d = q - p
        e = b - a
        den = d[:, 0] * e[1] - d[:, 1] * e[0]
        ok = np.abs(den) > 1e-9
        den = np.where(ok, den, 1.0)
        w = a - p
        t = (w[:, 0] * e[1] - w[:, 1] * e[0]) / den         # along the road segment
        s_ = (w[:, 0] * d[:, 1] - w[:, 1] * d[:, 0]) / den  # along the middle line
        if np.any(ok & (t >= 0) & (t <= 1) & (s_ >= 0) & (s_ <= 1)):
            return True
    return False


def _parts_index(f):
    """{building id: [rows]} of a building parts layer, kept on it (the cells stay decoded in memory: every tile of
    them looks parts up again)."""
    memo = f.__dict__.get("_bnge_parts_by_bid")
    if memo is None:
        memo = {}
        col = f.attrs.get("building_id")
        if col is not None:
            for i in range(len(f.kind)):
                bid = str(col[i])
                if bid:
                    memo.setdefault(bid, []).append(i)
        f.__dict__["_bnge_parts_by_bid"] = memo
    return memo


def _parts_tops(f):
    """Each part's own height + bottom (_part_heights' sum; nan without a height) for the whole layer, kept on it."""
    tops = f.__dict__.get("_bnge_parts_tops")
    if tops is None:
        a, n = f.attrs, len(f.kind)

        def col(k):
            return np.asarray(a[k], np.float64) if k in a else np.full(n, np.nan)
        with np.errstate(invalid="ignore"):
            top, mh, nf, f0 = col("height"), col("min_height"), col("num_floors"), col("min_floor")
            mh = np.where(np.isfinite(mh) & (mh > 0), mh, 0.0)
            f0 = np.where(np.isfinite(f0) & (f0 > 0), f0, 0.0)
            floors = np.maximum(FLOOR_M, (nf - f0) * FLOOR_M) + np.where(mh > 0, mh, f0 * FLOOR_M)
            tops = np.where(np.isfinite(top) & (top > mh + 1.0), top,
                            np.where(np.isfinite(nf) & (nf > 0), floors, np.nan))
        f.__dict__["_bnge_parts_tops"] = tops
    return tops


class _PartsOf:
    """The building parts of a tile's cells by building id, each [(ring in the anchor frame, _part_heights)], made
    when asked for: a tile needs those of its own buildings, and the toll booths (booths), not all the cells' parts
    (converting every part for every tile was half of tile_buildings' time in Manhattan, 2026-10-24). A part in two
    cells' layers is taken once (the first)."""

    def __init__(self, frame, part_layers):
        self.frame = frame
        self.layers = [f for f in part_layers if f is not None and len(f.kind)]
        self.memo = {}

    def _part(self, f, i):
        pp = f.parts(i)
        if not pp or not pp[0]:
            return None
        outer = np.asarray(pp[0][0], np.float64)
        if len(outer) < 4:
            return None
        e, n = mapgeom.enu_xy(f, self.frame, outer)
        a = f.attrs
        # (its own roof shape and mapped colours: hudsontadams' PR #167)
        own = {k: (a[k][i] if k in a else None) for k in ("roof_shape", "facade_color", "roof_color")}
        return np.stack([e, n], 1), _part_heights(a, i), own

    def get(self, bid, default=None):
        if bid not in self.memo:
            out, seen = [], set()
            for f in self.layers:
                for i in _parts_index(f).get(bid, ()):
                    pid = f.ids[i]
                    if pid in seen:
                        continue
                    seen.add(pid)
                    p = self._part(f, i)
                    if p is not None:
                        out.append(p)
            self.memo[bid] = out or None
        got = self.memo[bid]
        return default if got is None else got

    def booths(self, lo, hi):
        """The parts (ring, _part_heights) whose top (own height + bottom) is within [lo, hi], of any building."""
        out, seen = [], set()
        for f in self.layers:
            index = _parts_index(f)
            if not index:
                continue
            col = f.attrs["building_id"]
            tops = _parts_tops(f)
            with np.errstate(invalid="ignore"):
                rows = np.flatnonzero((tops >= lo - 1e-6) & (tops <= hi + 1e-6))
            for i in rows.tolist():
                pid = f.ids[i]
                if pid in seen or not str(col[i]):
                    continue
                seen.add(pid)
                p = self._part(f, i)
                if p is not None:
                    out.append(p[:2])
        return out


def _hulls_nearby(frame, layers, x0, y0, x1, y1, water_xy, reach=ships.PIER_REACH_M):
    """Local rings of the moored ships (ships.is_ship) whose outline centres lie outside the tile but within reach of
    it (a quay in this tile may lie alongside a ship the next tile draws)."""
    out = []
    for f in layers:
        if f is None or len(f.kind) == 0:
            continue
        a = f.attrs
        rows = range(len(f.kind))
        if isinstance(f, mapgeom.Features):
            cx_, cy_ = f.outer_centres(frame)
            bx = f.boxes(frame)
            big = np.maximum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]) >= ships.MIN_L / math.sqrt(2)
            rows = np.flatnonzero(big & (cx_ >= x0 - reach) & (cx_ < x1 + reach) & (cy_ >= y0 - reach)
                                  & (cy_ < y1 + reach) & ~((cx_ >= x0) & (cx_ < x1) & (cy_ >= y0) & (cy_ < y1))
                                  & (bx[:, 0] <= x1 + ships.PIER_GAP_M) & (bx[:, 2] >= x0 - ships.PIER_GAP_M)
                                  & (bx[:, 1] <= y1 + ships.PIER_GAP_M) & (bx[:, 3] >= y0 - ships.PIER_GAP_M)).tolist()
        for i in rows:
            if "src_dataset" in a and str(a["src_dataset"][i]) != "OpenStreetMap":
                continue
            if not water_xy and not ships.has_evidence({k: (a[k][i] if k in a else None)
                                                        for k in ("class", "subtype", "name")}):
                continue                        # (no water near and no ship's tags: cheaper than the outline)
            parts = f.parts(i)
            if not parts or not parts[0] or len(parts[0][0]) < 4:
                continue
            e, n = mapgeom.enu_xy(f, frame, np.asarray(parts[0][0], np.float64))
            cx, cy = float(e[:-1].mean()), float(n[:-1].mean())
            if (x0 <= cx < x1 and y0 <= cy < y1) or max(np.ptp(e), np.ptp(n)) < ships.MIN_L / math.sqrt(2):
                continue                        # (the tile's own: tested already)
            r = clean_ring(np.stack([e - x0, n - y0], 1))
            at = {k: (a[k][i] if k in a else None) for k in ("class", "subtype", "name")}
            if r is not None and ships.is_ship(r, at, lambda: ships.on_water(r.mean(0), water_xy)):
                out.append(r)
    return out


def parts_cover(outline, part_rings):
    """The share of outline (k, 2) that any of part_rings [(m, 2)] covers (same frame), from about
    PARTS_COVER_SAMPLES grid points inside it; 1.0 for an outline too small to sample. Pure."""
    from polygon import Path
    o = np.asarray(outline, np.float64)[:, :2]
    area = abs(_area2(o)) / 2.0
    lo, hi = o.min(0), o.max(0)
    step = max(math.sqrt(area / PARTS_COVER_SAMPLES), 0.25)
    gx, gy = np.meshgrid(np.arange(lo[0] + step / 2, hi[0], step), np.arange(lo[1] + step / 2, hi[1], step))
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    g = g[Path(o).contains_points(g)] if len(g) else g
    if not len(g):
        return 1.0
    hit = np.zeros(len(g), bool)
    for r in part_rings:
        hit |= np.asarray(Path(np.asarray(r, np.float64)[:, :2]).contains_points(g), bool)
    return float(hit.mean())


def _outline_under_parts(at, bparts):
    """May a building drawn as its parts (bparts: [(ring, (own height, bottom) or None, ...)]) have its outline drawn
    under them where they cover too little of it: its own height (or levels) mapped, PARTS_COVER_MIN_H or more, and
    no part off the ground."""
    h, fl = at.get("height"), at.get("num_floors")
    own = (h is not None and np.isfinite(h) and h >= PARTS_COVER_MIN_H) or \
        (h is None or not np.isfinite(h)) and fl is not None and np.isfinite(fl) and fl >= 1
    return bool(own) and all(p[1] is None or p[1][1] <= PARTS_GROUND_M for p in bparts)
