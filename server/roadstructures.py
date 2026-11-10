"""Road builder: what is built round the roads (split out of roads.py): tunnel holes and arches, retaining walls and
parapets, tunnel and bridge chunks, deck pads and sidewalks on decks.
"""
import json
import math
import hashlib
import numpy as np
import junctions
import mapcells
import mapgeom
import roadmesh
from roadbase import (
    ABUT_GROUND_M,
    ARCH_SIDE_M,
    ARCH_SOIL_MIN,
    BACK_TOP_OVER_M,
    BUILT,
    CLIFF_EDGE_M,
    CLIFF_MIN_CUT_M,
    CLIFF_OVER_M,
    CLIFF_REACH_M,
    CLIFF_ROAD_GAP_M,
    CLIFF_SHAPES,
    CLIFF_SINK_M,
    CLIFF_SPACING,
    CLIFF_TALL_M,
    CLIFF_WALLS_ON,
    CONCRETE,
    CONCRETE_WALL_CLASSES,
    CORNER_CONCRETE_M,
    DECK,
    DECK_CLEAR,
    DECK_CONCRETE,
    DECK_CONCRETE_CLASSES,
    DECK_CONCRETE_ON,
    DECK_CONCRETE_TILE_M,
    DECK_CONTEXT_M,
    DECK_FULL_DEPTH,
    DECK_JOIN_M,
    DECK_STEP,
    DECK_WALK,
    DECK_WALK_CHAIN,
    DECK_WALK_LIFT_M,
    DRAIN,
    DRAIN_EVERY,
    DRAIN_IN,
    EDGE_DECAL,
    GOING_ON_DEG,
    GUTTER,
    HOLE_JOIN_M,
    HOLE_ROAD_MARGIN,
    LINE_RANK,
    LOOK,
    LOW_HOLE_M,
    LOW_HOLE_STEP,
    NEAR_M,
    NO_PARAPET_CLASSES,
    PADS_ON,
    PAD_FORK_COS,
    PAD_FORK_MARGIN_M,
    PAD_FORK_OPEN_M,
    PAD_FORK_STRAIGHT_COS,
    PAD_GROUND_M,
    PAD_JOIN_M,
    PAD_JOIN_Z_M,
    PAD_MARGIN_M,
    PAD_OPEN_M,
    PAD_REACH_M,
    PAD_RIM_STEP_M,
    PAD_STEP_M,
    PAD_STRAIGHT_COS,
    PAD_WALL_DROP_M,
    PARAPET_DROP_M,
    PARAPET_DROP_MOUNTAIN_M,
    PARAPET_H,
    PARAPET_OFF_M,
    PARAPET_PROBE_M,
    PARAPET_PROBE_MOUNTAIN_M,
    PARAPET_ROAD_GAP_M,
    PARAPET_W,
    SIDEWALK,
    SIMPLIFY_Z_M,
    TUNNEL,
    TUNNEL_HOLE_MARGIN,
    TWIN_DZ_M,
    TWIN_SHARE,
    TWIN_STEP_M,
    WALL_COPING_MIN_M,
    WALL_DECK_MARGIN_M,
    WALL_DECK_OVER_M,
    WALL_DIAG_M,
    WALL_EVEN_N,
    WALL_FOOT_M,
    WALL_FULL,
    WALL_GAP_N,
    WALL_KEEP_M,
    WALL_MIN_N,
    WALL_OTHER_GAP_M,
    WALL_PROBE_M,
    WALL_SLAB_DZ,
    WALL_STEEP,
    WALL_STEEP_SPAN,
    WALL_STEP,
    WALL_STEP_UP_M,
    WALL_TAPER_LOW_M,
    WALL_TAPER_M,
    WALL_THICK_MAX_M,
    WALL_THICK_MIN_M,
    WALL_TOP_EDGE_M,
    WALL_TOP_M,
    densify,
    log,
    look_of,
    split,
    terrain_surface,
)
from roadspecs import (
    bridge_parts,
    bridge_specs,
    carriageway,
    copies_of,
    ground_parts,
    hairpin_tapers,
    joint_desc,
    lane_shifts,
    road_specs,
    shift_at,
    taper_widths,
)
from roaddecks import (
    _EndJoin,
    _smooth_pinned,
    deck_heights,
)
from roadgeom import (
    cut_outside,
    offset_polyline,
)
from roadwalks import (
    clip_polyline,
    extras,
    has_sidewalks,
    simplify,
    walk_sides,
)
from roadtiles import (
    box_owns,
)
from roadbridges import (
    _chunk,
)


def tunnel_hole_mask(parts, u, cx, cy, square, heights=None, mouths=None):
    """Where the terrain block (heights u [iy, ix]: sample (ix, iy) at (cx - half + ix * square, cy - half +
    iy * square), half = n * square / 2) must be a hole for the tunnels `parts` ([(fid, spec, xy, z, lon)]: a
    TunnelProfiles' parts): the samples over a tube's footprint whose ground lies between its floor and its
    vault (+- TUNNEL_HOLE_MARGIN): the portals' openings into the hillside (the ground over the vault stays, as
    does ground under a tube crossing a hollow). Pure."""
    import fast
    import roadmesh
    n = u.shape[0]
    half = n * square / 2
    x0, y0 = cx - half, cy - half
    mask = np.zeros(u.shape, bool)
    # twin tunnels (twin_tunnels): their one tube round both reaches over the strip between them too, and its mouths are
    # cut whatever the height (the ground left between the carriageways sloped into the shared opening, its portal face
    # covering what is above)
    parts = list(parts)
    n_own = len(parts)
    parts += [(parts[i][0], parts[i][1]._replace(width=w), mid[:, :2], mid[:, 2], parts[i][4])
              for i, j, mid, w in twin_tunnels([p for p in parts if np.isfinite(np.asarray(p[3])).all()]) if len(mid) >= 2]
    fast.require(88, "tunnel_holes")
    u = np.ascontiguousarray(u, np.float64)
    for k_, (_fid, spec, xy, z, _lon) in enumerate(parts):
        xy = np.asarray(xy, np.float64)
        z = np.asarray(z, np.float64)
        if len(xy) < 2 or not np.isfinite(z).all():
            continue
        hw = spec.width / 2 + roadmesh.TUNNEL_SHOULDER + TUNNEL_HOLE_MARGIN
        th = (heights or {}).get(_fid, roadmesh.TUNNEL_H)       # heights: a low arch's tube (TunnelProfiles.heights)
        low = th < roadmesh.TUNNEL_H - 0.05
        if k_ >= n_own and not low:     # a twin tube's mouths, whatever the height (the low arches' cut, no side reach)
            m0, m1 = (mouths or {}).get(_fid, (True, True))
            mouth_mask = np.zeros(u.shape, bool)
            fast._ef.tunnel_holes(mouth_mask, u, float(x0), float(y0), float(square), np.ascontiguousarray(xy),
                                  np.ascontiguousarray(z), float(hw), float(th), True, bool(m0), bool(m1),
                                  float(TUNNEL_HOLE_MARGIN), float(LOW_HOLE_M + LOW_HOLE_STEP), 0.0, True)
            mask |= mouth_mask
        # the Rust kernel: the samples over the part between its floor and vault (a low arch: its mouths only)
        m0, m1 = (mouths or {}).get(_fid, (True, True))
        # (a low arch: the squares from its flat mouth band, LOW_HOLE_M, up to the soil over its vault)
        fast._ef.tunnel_holes(mask, u, float(x0), float(y0), float(square), np.ascontiguousarray(xy),
                              np.ascontiguousarray(z), float(hw), float(th), bool(low), bool(m0), bool(m1),
                              float(TUNNEL_HOLE_MARGIN), float(LOW_HOLE_M if low else LOW_HOLE_M + LOW_HOLE_STEP),
                              float(ARCH_SIDE_M + 1.0), not low)
    return mask


def arch_raise(parts, heights, px, py, z, mouths=None):
    """z (ground heights at the anchor-frame points px, py) raised over the low arches among parts (TunnelProfiles
    parts; heights: their tubes, TunnelProfiles.heights) to ARCH_SOIL_MIN over the vault, beside it too, but for
    their mouths (LOW_HOLE_M, cut there by tunnel_hole_mask): the ground dipping into a low tube near its mouths
    made one-sided terrain triangles from the road's level up to the soil, seen as dark wedges in the opening from
    outside (Central Park, 2026-10-17). Pure."""
    import fast
    import roadmesh
    z = np.array(z, np.float64, copy=True)
    fast.require(0, "arch_raise_part")
    px = np.ascontiguousarray(px, np.float64)
    py = np.ascontiguousarray(py, np.float64)
    for fid, spec, xy, zf, _lon in parts:
        th = (heights or {}).get(fid)
        if th is None or th >= roadmesh.TUNNEL_H - 0.05:
            continue
        xy = np.asarray(xy, np.float64)
        zf = np.asarray(zf, np.float64)
        if len(xy) < 2 or not np.isfinite(zf).all():
            continue
        hw = spec.width / 2 + roadmesh.TUNNEL_SHOULDER + ARCH_SIDE_M
        # the Rust kernel (per part) on the points in its box
        m0, m1 = (mouths or {}).get(fid, (True, True))
        lo, hi = xy.min(0) - hw, xy.max(0) + hw
        k = np.flatnonzero((px >= lo[0]) & (px <= hi[0]) & (py >= lo[1]) & (py <= hi[1]))
        if len(k):
            zk = np.ascontiguousarray(z[k])
            fast._ef.arch_raise_part(zk, np.ascontiguousarray(px[k]), np.ascontiguousarray(py[k]),
                                     np.ascontiguousarray(xy), np.ascontiguousarray(zf), float(hw), float(th),
                                     float(ARCH_SOIL_MIN), float(LOW_HOLE_M), bool(m0), bool(m1))
            z[k] = zk
    return z


def tunnel_floor_at(parts, x, y):
    """The floor height of the tunnel tubes among parts ([(fid, spec, xy, z, lon)]) over the point (x, y), the lowest
    one, or None outside them (within the road's half width + roadmesh.TUNNEL_SHOULDER). Pure."""
    import roadmesh
    best = None
    p = np.array([x, y], np.float64)
    for _fid, spec, xy, z, _lon in parts:
        xy = np.asarray(xy, np.float64)
        z = np.asarray(z, np.float64)
        if len(xy) < 2 or not np.isfinite(z).all():
            continue
        hw = spec.width / 2 + roadmesh.TUNNEL_SHOULDER
        a, b = xy[:-1], xy[1:]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
        d = np.hypot(*(a + t[:, None] * ab - p).T)
        j = int(np.argmin(d))
        if d[j] <= hw:
            f = float(z[j] + t[j] * (z[j + 1] - z[j]))
            best = f if best is None else min(best, f)
    return best


def block_arch_raise(tunnels, cell_fn, frame, la, lo, z, box):
    """arch_raise for ground heights at the points (la, lo) of a block in the anchor-frame box (x0, y0, x1, y1),
    the tunnels from the map cells under it. z as it was when the cells are not at hand."""
    try:
        x0, y0, x1, y1 = box
        cla, clo, _ = frame.enu_to_geodetic(np.array([x0, x1]), np.array([y0, y1]), np.zeros(2))
        keys = mapcells.cells_for_bbox(float(cla.min()), float(clo.min()), float(cla.max()), float(clo.max()))
        cells = []
        for key in keys:
            try:
                cells.append(cell_fn(key))
            except Exception:
                cells.append(None)
        parts = tunnels.parts(cells, box)
        if not parts or not tunnels.heights:
            return z
        e, n, _ = frame.geodetic_to_enu(np.asarray(la, np.float64).ravel(), np.asarray(lo, np.float64).ravel(), 0.0)
        return arch_raise(parts, tunnels.heights, np.asarray(e), np.asarray(n),
                          np.asarray(z, np.float64).ravel(), getattr(tunnels, "mouths", None)).reshape(np.shape(z))
    except Exception as ex:         # no raise rather than no block (a network error quietly; a bug where it shows)
        (log.debug if isinstance(ex, OSError) else log.warning)("arch raise: %s: %s", type(ex).__name__, ex)
        return z


def block_tunnel_holes(tunnels, cell_fn, frame, keep, cx, cy, square):
    """tunnel_hole_mask for a terrain block being built (heightmap.generate's keep: lat, lon, u), its tunnels from
    the map cells under it (cell_fn(key) -> roads Features) through the anchor's TunnelProfiles."""
    lat, lon, u = keep["lat"], keep["lon"], keep["u"]
    keys = mapcells.cells_for_points(lat, lon)
    cells = []
    for k in keys:
        try:
            cells.append(cell_fn(k))
        except Exception as e:      # a cell not at hand (offline, a network error): no tunnels from it this time
            log.debug("tunnel holes: cell %s: %s: %s", k, type(e).__name__, e)
            cells.append(None)
    half = u.shape[0] * square / 2
    try:
        parts = tunnels.parts(cells, (cx - half, cy - half, cx + half, cy + half))
    except Exception as e:          # a chain's other cells not at hand (offline): no holes this time
        (log.debug if isinstance(e, OSError) else log.warning)("tunnel holes: %s: %s", type(e).__name__, e)
        return None
    if not parts:
        return None
    mask = tunnel_hole_mask(parts, u, cx, cy, square, getattr(tunnels, "heights", None),
                            getattr(tunnels, "mouths", None))
    if mask is not None and mask.any():
        mask = spare_carriageways(mask, parts, cells, frame, cx, cy, square)
    return mask


def spare_carriageways(mask, parts, cells, frame, cx, cy, square):
    """mask with the hole samples inside the carriageway of a ground road that is not the tunnel's (another feature, not
    an approach joined at a mouth) filled again: a portal's opening reaching into a road beside or over it left a hole in
    its lanes (GitHub #450, the A-7 near Málaga, a 2 m drop at the edge of a carriageway beside an underpass's mouth)."""
    import fast
    own = {str(p[0]).split("#")[0] for p in parts}
    ends = np.array([np.asarray(p[2], np.float64)[k, :2] for p in parts for k in (0, -1) if len(p[2])], np.float64)
    others = []
    for feats in cells:
        if feats is None or len(feats) == 0:
            continue
        rs = road_specs(feats)
        for i, sp in enumerate(rs):
            fid = str(feats.ids[i])
            if sp is None or sp.kind not in BUILT or fid in own:
                continue
            for ring in ground_parts(feats, i):
                e, n = mapgeom.enu_xy(feats, frame, ring)
                xy = np.c_[e, n]
                if len(xy) < 2:
                    continue
                if len(ends) and min(float(np.min(np.hypot(*(ends - xy[0]).T))),
                                     float(np.min(np.hypot(*(ends - xy[-1]).T)))) < HOLE_JOIN_M:
                    continue            # (its approach: the mouth is cut into it)
                others.append((fid, sp.width / 2, xy))
    if not others:
        return mask
    oth = fast.Others(others)
    n = mask.shape[0]
    x0, y0 = cx - n * square / 2, cy - n * square / 2
    out = mask.copy()
    for iy, ix in zip(*np.nonzero(mask)):
        if not oth.clear((x0 + ix * square, y0 + iy * square), None, HOLE_ROAD_MARGIN):
            out[iy, ix] = False
    return out


def _wall_runs(cut):
    """Which samples along one side of a road get a wall, from the cut's depth there (m). Pure."""
    import grading
    cut = np.asarray(cut, np.float64)
    keep = cut > WALL_KEEP_M
    out = np.zeros(len(cut), bool)
    i = 0
    while i < len(cut):                     # runs over WALL_KEEP_M that reach WALL_CUT_M somewhere
        if not keep[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(cut) and keep[j + 1]:
            j += 1
        if (cut[i:j + 1] > grading.WALL_CUT_M).any():
            out[i:j + 1] = True
        i = j + 1
    i = 0
    while i < len(out):                     # runs shorter than WALL_MIN_N dropped ...
        if not out[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(out) and out[j + 1]:
            j += 1
        if j - i + 1 < WALL_MIN_N:
            out[i:j + 1] = False
        i = j + 1
    idx = np.flatnonzero(out)               # ... then gaps of up to WALL_GAP_N filled
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= WALL_GAP_N + 1:
            out[a:b] = True
    return out


def _under_decks(xy, z, decks):
    """Which points xy (k, 2) at road heights z lie under a bridge deck (deck chunks: "n" points [x, y, z], "w" m):
    within its half width + WALL_DECK_MARGIN_M of its nearest segment and the deck WALL_DECK_OVER_M or more over them
    (earthfast.under_decks). Pure."""
    import fast
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    pack = _deck_pack(decks)
    if pack is None or not len(xy):
        return np.zeros(len(xy), bool)
    fast.require(74, "under_decks")
    return np.asarray(fast._ef.under_decks(np.ascontiguousarray(xy), np.ascontiguousarray(np.asarray(z, np.float64)),
                                           *pack, float(WALL_DECK_OVER_M)), bool)


_DECK_PACK = [(None, None)]     # [(the decks list, its packed arrays)]: one tile's walls ask with the same list


def _deck_pack(decks):
    """(points (n, 3), starts (C + 1,), reach (C,)) of the deck chunks with 2 points or more, or None."""
    if not decks:
        return None
    last = _DECK_PACK[0]
    if last[0] is decks:
        return last[1]
    ns = [np.asarray(ch["n"], np.float64)[:, :3] for ch in decks]
    keep = [k for k, n in enumerate(ns) if len(n) >= 2]
    pack = None
    if keep:
        pts = np.ascontiguousarray(np.concatenate([ns[k] for k in keep]))
        starts = np.r_[0, np.cumsum([len(ns[k]) for k in keep])].astype(np.int64)
        reach = np.array([float(decks[k].get("w") or 0) / 2 + WALL_DECK_MARGIN_M for k in keep], np.float64)
        pack = (pts, starts, reach)
    _DECK_PACK[0] = (decks, pack)
    return pack


def _off_wall_tops(chunks, walls):
    """chunks with the ground roads' surfaces and lines cut away over the retaining walls' tops (a path on the bank
    above a cutting lay on the lattice's low strip under the top, in patches through it, Munich's McGraw-Graben).
    Decks, tunnels and filled polygons are left as they are."""
    if not walls:
        return chunks
    blockers = []
    for w in walls:
        th = np.asarray(w["thick"], np.float64)
        mid = np.asarray(w["base"], np.float64) - np.asarray(w["toward"], np.float64) * (th[:, None] / 2)
        blockers.append((mid, max(float(th.max()) / 2 - WALL_TOP_EDGE_M, 0.1)))
    lo = np.min([b[0].min(0) - b[1] for b in blockers], axis=0)
    hi = np.max([b[0].max(0) + b[1] for b in blockers], axis=0)
    out = []
    for c in chunks:
        n = c.get("n")
        if c.get("deck") or c.get("tunnel") or c.get("poly") is not None or not n or len(n) < 2:
            out.append(c)
            continue
        pts = np.asarray(n, np.float64)
        if (pts[:, 0].max() < lo[0] or pts[:, 0].min() > hi[0] or pts[:, 1].max() < lo[1]
                or pts[:, 1].min() > hi[1]):
            out.append(c)
            continue
        pieces = cut_outside(pts, blockers, 1.0)
        if len(pieces) == 1 and len(pieces[0]) >= len(pts) and np.allclose(pieces[0][[0, -1]], pts[[0, -1]]):
            out.append(c)
            continue
        for piece in pieces:
            out.append(dict(c, n=[[round(float(v), 3) for v in q] for q in simplify(piece)]))
    return out


def _taper(xy, run):
    """Per point of a road's wall sample line xy (k, 2): its wall's height factor, 0 at the ends of each run of run (k,)
    rising (smoothstep) to 1 over WALL_TAPER_M (no full-height wall ending in a cliff, the user 2026-10-21). Pure."""
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.asarray(xy, np.float64)[:, :2], axis=0).T))]
    f = np.zeros(len(xy))
    i = 0
    while i < len(run):
        if not run[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(run) and run[j + 1]:
            j += 1
        d = np.minimum(s[i:j + 1] - s[i], s[j] - s[i:j + 1]) / WALL_TAPER_M
        d = np.clip(d, 0.0, 1.0)
        f[i:j + 1] = d * d * (3 - 2 * d)
        i = j + 1
    return f


class _Segs:
    """others {fid: (centreline (m, 2), half width)} as one table of segments (a, b, half width, its road's fid), with
    each segment's box: the carriageway tests below in one Rust call each (one per road cost retaining_walls ~2 s of
    a Canton tile set, 2026-10-25)."""

    def __init__(self, others):
        rows = [(f, np.asarray(ln, np.float64), h) for f, (ln, h) in others.items() if len(ln) >= 2]
        f64 = np.float64
        self.a = np.concatenate([ln[:-1, :2] for _f, ln, _h in rows]) if rows else np.zeros((0, 2), f64)
        self.b = np.concatenate([ln[1:, :2] for _f, ln, _h in rows]) if rows else np.zeros((0, 2), f64)
        self.h = np.concatenate([np.full(len(ln) - 1, h, f64) for _f, ln, h in rows]) if rows else np.zeros(0, f64)
        self.road = np.concatenate([np.full(len(ln) - 1, n) for n, (_f, ln, _h) in enumerate(rows)]) if rows else \
            np.zeros(0, np.int64)
        self.fids = [f for f, _ln, _h in rows]
        self.lo = np.minimum(self.a, self.b)
        self.hi = np.maximum(self.a, self.b)

    def near(self, pts, gap, skip=None):
        """Which points (k, 2) lie within a segment's half width + gap (its road's fid not skip). Pure."""
        import fast
        pts = np.ascontiguousarray(np.asarray(pts, np.float64))
        if not len(pts) or not len(self.h):
            return np.zeros(len(pts), bool)
        lo, hi = pts.min(0), pts.max(0)
        r = self.h + gap
        keep = (self.hi[:, 0] + r >= lo[0]) & (self.hi[:, 1] + r >= lo[1]) & (self.lo[:, 0] - r <= hi[0]) & \
            (self.lo[:, 1] - r <= hi[1])
        if skip is not None and skip in self.fids:
            keep &= self.road != self.fids.index(skip)
        if not keep.any():
            return np.zeros(len(pts), bool)
        fast.require(0, "points_near_segments")
        return np.asarray(fast._ef.points_near_segments(pts, np.ascontiguousarray(self.a[keep]),
                                                        np.ascontiguousarray(self.b[keep]),
                                                        np.ascontiguousarray(r[keep])), bool)


def _on_carriageways(pts, others, gap):
    """Which points (k, 2) lie on a carriageway of others {fid: (centreline (m, 2), half width)} or a _Segs of them
    (+ gap). Pure."""
    segs = others if isinstance(others, _Segs) else _Segs(others)
    return segs.near(pts, gap)


def _clear_depth(base, out, depth, others, own):
    """Per point of a wall's face line base (k, 2): how far (up to depth (k,)) out from it along out (k, 2) stays off
    every other road's carriageway in others {fid: (centreline (m, 2), half width)} or a _Segs of them (less
    WALL_OTHER_GAP_M). Pure."""
    segs = others if isinstance(others, _Segs) else _Segs(others)
    steps = np.arange(0.5, float(np.max(depth)) + 0.51, 0.5)
    q = (base[:, None, :] + out[:, None, :] * steps[None, :, None]).reshape(-1, 2)
    # (its own road too: a hairpin's other leg; the face stands past its own carriageway, so the stretch beside it
    # never hits)
    hit = segs.near(q, WALL_OTHER_GAP_M).reshape(len(base), len(steps))
    first = np.where(hit.any(axis=1), steps[np.argmax(hit, axis=1)] - 0.5, np.inf)
    return np.minimum(np.asarray(depth, np.float64), np.maximum(first, 0.0))


def _parapets(xy, z, out_n, rh, inside, steps, gs, others, own, decks, drop_m=None, probe_m=None):
    """The stone parapets along one side of a road (see PARAPET_*): wall dicts as retaining_walls' (roadmesh draws them
    with a back face). gs: the graded ground at steps (m off the centre line) per point. Pure but for decks."""
    near = np.flatnonzero((steps >= rh + 0.5) & (steps <= rh + (PARAPET_PROBE_M if probe_m is None else probe_m)))
    if not len(near):
        return []
    drop = (z[:, None] - gs[:, near]).max(axis=1) > (PARAPET_DROP_M if drop_m is None else drop_m)
    base = xy + out_n * (rh + PARAPET_OFF_M)
    ok = drop & inside & ~_under_decks(base, z, decks) & ~_on_carriageways(base, others, 0.0)
    # not across a road joining it (its junctions), nor where another road runs beside
    if ok.any():
        segs = others if isinstance(others, _Segs) else _Segs(others)
        ok &= ~segs.near(base, PARAPET_ROAD_GAP_M, skip=own)
    out, i = [], 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        if j - i + 1 >= WALL_MIN_N:
            sl = slice(i, j + 1)
            # (its foot down to the ground behind it, which falls away: it stood in the air seen from below)
            back = np.flatnonzero((steps >= rh) & (steps <= rh + 2.5))
            g_back = gs[sl][:, back].min(axis=1) if len(back) else z[sl] - 1.0
            out.append({"base": base[sl], "top": z[sl] + PARAPET_H, "foot": np.minimum(z[sl] - 1.0, g_back - 0.3),
                        "toward": -out_n[sl],
                        "thick": np.full(j - i + 1, PARAPET_W), "concrete": False, "rock": False, "parapet": True,
                        "at": (i, j + 1),
                        "road_z": z[sl]})
        i = j + 1
    return out


def cliff_rows(walls, roads=()):
    """trees.lua rows [group, x, y, z, yaw, scale, height, rank] of rock-face pieces along the rural walls' faces
    (retaining_walls' "rock" walls). roads: build_tile's roads: no piece's footprint on a carriageway (shrunk until
    clear, else left out). Deterministic (hashed by position)."""
    import fast
    lines = [(np.asarray(pts, np.float64)[:, :2], spec.width / 2 + CLIFF_ROAD_GAP_M) for _, spec, pts, _ in roads
             if len(pts) >= 2]
    A = np.concatenate([ln[:-1] for ln, _ in lines]) if lines else np.zeros((0, 2))
    B = np.concatenate([ln[1:] for ln, _ in lines]) if lines else np.zeros((0, 2))
    H = np.concatenate([np.full(len(ln) - 1, h) for ln, h in lines]) if lines else np.zeros(0)

    def on_road(px, py, yaw, sc, width, front, depth):
        """Does the piece's footprint (its corners and edge midpoints) touch a carriageway?"""
        if not len(A):
            return False
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ux, uy = np.array([c, s_]), np.array([-s_, c])          # its local x (along) and y (back from the road)
        pts = [np.array([px, py]) + ux * a * width * sc / 2 + uy * b * sc
               for a in (-1.0, -0.5, 0.0, 0.5, 1.0) for b in (-front * 0.6, depth - front)]
        fast.require(0, "points_near_segments")
        return bool(np.asarray(fast._ef.points_near_segments(np.ascontiguousarray(np.array(pts)), np.ascontiguousarray(A),
                                                             np.ascontiguousarray(B), np.ascontiguousarray(H))).any())
    rows = []
    for w in walls:
        # (their fronts just off the carriageway, not at the wall's face: the ground climbing in front of the face
        # showed as a mound under the rocks)
        b = np.asarray(w.get("edge", w["base"]), np.float64)
        if len(b) < 2:
            continue
        tw = np.asarray(w["toward"], np.float64)
        road_z, top = np.asarray(w["road_z"], np.float64), np.asarray(w["top"], np.float64)
        clear = np.asarray(w["clear"], np.float64)
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(b, axis=0).T))]
        at = 0.0
        while at <= s[-1]:
            i = min(int(np.searchsorted(s, at)), len(b) - 1)
            hh = float(np.interp(at, s, top) - np.interp(at, s, road_z)) + CLIFF_SINK_M + CLIFF_OVER_M
            jit = _hash01(round(float(b[i, 0]), 1), round(float(b[i, 1]), 1))
            hh *= 0.9 + 0.2 * jit
            sh, frac, front, width, depth = CLIFF_SHAPES[1 if hh >= CLIFF_TALL_M else 0]
            sc = hh / sh
            sc = min(sc, float(clear[i]) / depth)          # (not over another road)
            if sc < 0.2:
                at += 3.0
                continue
            t = tw[i]
            x, y = float(np.interp(at, s, b[:, 0])), float(np.interp(at, s, b[:, 1]))
            jit2 = _hash01(round(float(b[i, 0]), 1), round(float(b[i, 1]), 1), "z")
            yaw = math.degrees(math.atan2(t[0], -t[1])) + (jit - 0.5) * 30.0
            while sc >= 0.2:            # (shrunk until its footprint is off every carriageway, this road's included)
                px, py = x - t[0] * front * sc, y - t[1] * front * sc   # its front (-y, toward the road) on the face
                if not on_road(px, py, yaw, sc, width, front, depth):
                    break
                sc *= 0.8
            if sc < 0.2:
                at += 3.0
                continue
            hh = sc * sh
            zb = float(np.interp(at, s, road_z)) - CLIFF_SINK_M - jit2 * 0.8 * sc
            rows.append(["cliff", round(px, 2), round(py, 2), round(zb + frac * hh, 2), round(yaw % 360.0, 1), 1.0,
                         round(hh, 2), round(jit, 3)])
            at += max(CLIFF_SPACING * width * sc, 1.5)
    return rows


def _hash01(*key):
    import hashlib
    h = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


def _cut_behind(base, out_n, z, ground):
    """Whether the graded ground behind a wall's face line base (k, 2) (out_n: unit vectors away from the road; z: the
    road's heights) climbs as steeply as a cutting does (WALL_STEEP over 1 m, its median along the wall)."""
    offs = np.arange(WALL_STEEP_SPAN[0], WALL_STEEP_SPAN[1] + 0.01, 0.5)
    P = (base[:, None, :] + out_n[:, None, :] * offs[None, :, None]).reshape(-1, 2)
    g = np.asarray(ground(P), np.float64).reshape(len(base), len(offs))
    rise = (g[:, 2:] - g[:, :-2]).max(axis=1)           # (over 1 m: two 0.5 m steps)
    return bool(np.median(rise) >= WALL_STEEP)


def hairpin_extra(frame, spec, xy):
    """Per point of a road's line xy (k, 2) in the frame: how much wider than its width / 2 a pass road is there round
    its widened hairpins (hairpin_tapers on the line itself), or None (not a pass road, no hairpin)."""
    if not getattr(spec, "mountain", False) or len(xy) < 3:
        return None
    la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], np.zeros(len(xy)))
    lo, la = np.asarray(lo, np.float64), np.asarray(la, np.float64)
    ws = taper_widths(spec.width, hairpin_tapers(np.c_[lo, la], spec.width), lo, la)
    return None if ws is None else (ws - spec.width) / 2


def retaining_walls(frame, roads, elev, exact, decks=()):
    """Stone retaining walls along the ground roads cut deeper than grading.WALL_CUT_M into the ground (the grading
    leaves the ground at its natural height right behind them): [dict(base (k, 2) the wall's face line, top (k,),
    foot (k,), toward (k, 2) unit vectors to the road)] of the walls in the tile box `exact`. The edge of the road's
    flat core (as grading plans it), from the road level up to the ground behind; none where another road comes in
    (its ground is graded to the road), nor under a bridge deck (decks: the deck chunks near the tile; the ground beside
    a motorway under an overpass is its abutment's fill, and the wall rose through the deck, the user's report on the
    Bruce Highway, 2026-10-21). Central Park's 86th Street Transverse between its stone walls (the user's photos,
    2026-10-17). Pure given elev."""
    import grading
    import roadmesh
    x0, y0, x1, y1 = exact
    pts_all, info = [], []
    centres = []                # (the road's own ground along it, read with the probes below)
    for fid, spec, pts, _ in roads:
        if spec.kind not in BUILT or len(pts) < 2:
            continue
        xy, s = grading.resample(np.asarray(pts[:, :2], np.float64), WALL_STEP)
        if len(xy) < 2:
            continue
        seg = np.hypot(*np.diff(pts[:, :2], axis=0).T)
        z = np.interp(s, np.r_[0.0, np.cumsum(seg)], pts[:, 2])
        inside = (xy[:, 0] >= x0 - 2) & (xy[:, 0] < x1 + 2) & (xy[:, 1] >= y0 - 2) & (xy[:, 1] < y1 + 2)
        if not inside.any():
            continue
        centres.append(xy)
        d = np.gradient(xy, axis=0)
        d /= np.maximum(np.hypot(*d.T), 1e-9)[:, None]
        nl = np.stack([-d[:, 1], d[:, 0]], axis=1)
        width = spec.width if spec.mountain else max(spec.width, carriageway(spec.cls, spec.kind, True, None, spec.oneway))
        core = width / 2 + grading.shoulder(spec)
        # (the grading's sharp cuttings end the core sooner; its flat strip behind a wall reaches past the core)
        steps = np.arange(width / 2 + 0.5, core + grading.WALL_FLAT_M + 1.51, 1.0)
        for side in (1.0, -1.0):
            probe = xy + nl * side * (core + grading.WALL_FLAT_M + grading.WALL_T_M + WALL_PROBE_M)
            pts_all.append(probe)
            info.append((xy, z, nl * side, core, inside, width / 2, steps, spec.cls in CONCRETE_WALL_CLASSES,
                         fid, not spec.urban,
                         # (parapets: (the drop that gets one, within m of the edge), None none)
                         ((PARAPET_DROP_MOUNTAIN_M, PARAPET_PROBE_MOUNTAIN_M) if spec.mountain
                          else (PARAPET_DROP_M, PARAPET_PROBE_M))
                         if not spec.urban and spec.cls not in NO_PARAPET_CLASSES and not spec.median else None,
                         grading.wall_max_h(spec), spec.mountain, hairpin_extra(frame, spec, xy)))
    if not pts_all:
        return []

    def ground(P):
        la, lo, _ = frame.enu_to_geodetic(P[:, 0], P[:, 1], np.zeros(len(P)))
        return np.asarray(frame.geodetic_to_enu(la, lo, np.asarray(elev(la, lo), np.float64))[2], np.float64)
    nprobe = sum(len(P) for P in pts_all)
    g_all = ground(np.concatenate(pts_all + centres))
    g = g_all[:nprobe]
    # the road's height along it from the ground it is painted into, not a straight line between its map points: those
    # can be 100 m apart, and across a creek whose bridge has no deck the line ran 3 m over the drawn road, which read as
    # a raised road and got parapets down to the ground (GitHub #782, Western Australia: 3.5 m stone walls both sides of
    # Port Gregory Road at its creek crossings)
    k = nprobe
    for n, item in enumerate(info):
        if n % 2 == 0:                  # (two sides a road: one centre line)
            zc = g_all[k:k + len(item[0])]
            k += len(item[0])
        info[n] = (item[0], zc) + tuple(item[2:])
    # the graded ground at the steps out from the road only where a wall or a parapet can stand: a side whose probe
    # finds no cutting (_wall_runs) and that gets no parapets has no use for them (most of a town's streets: 8 of 9 of
    # the ground lookups, 2026-10-25)
    probes, k = [], 0
    for xy, z, out_n, core, inside, rh, steps, concrete, fid, rural, parapets, max_h, _mtn, _hx in info:
        gg = g[k:k + len(xy)]
        k += len(xy)
        need = len(steps) and (parapets is not None or _wall_runs(gg - z).any())
        probes.append((gg, need))
    sp = []
    for (xy, z, out_n, core, inside, rh, steps, *_r), (gg, need) in zip(info, probes):
        if need:
            sp.extend(xy + out_n * o for o in steps)
    gsteps = ground(np.concatenate(sp)) if sp else np.zeros(0)
    out, k = [], 0
    others = {fid: (np.asarray(pts, np.float64)[:, :2], spec.width / 2) for fid, spec, pts, _ in roads if len(pts) >= 2}
    others = _Segs(others)          # (one table for all the carriageway tests below)
    bend_of = {}
    for (xy, z, out_n, core, inside, rh, steps, concrete, fid, rural, parapets, max_h, mountain, hx), (gg, need) in \
            zip(info, probes):
        key = (id(xy), len(xy))
        if key not in bend_of:          # (a tight bend: no wall into it, grading.bend_zone)
            bend_of[key] = grading.bend_zone(xy)
        bends = bend_of[key]
        face = np.full(len(xy), core + 0.1)
        thick = np.full(len(xy), roadmesh.WALL_THICK_M)
        ground_at = None
        top_g = gg
        if need:                    # the wall where the graded ground starts to climb (a sharp cutting: before the core's edge)
            gs = gsteps[k:k + len(xy) * len(steps)].reshape(len(steps), len(xy)).T
            k += len(xy) * len(steps)
            up = gs - z[:, None] > WALL_STEP_UP_M
            first = np.argmax(up, axis=1)
            # (the grading's flat strip in front of the climb: the terrain's 2 m lattice draws a step at an angle to it
            # as teeth reaching up to a grid diagonal out, under the wall's top; its face in front of them)
            face = np.where(up.any(axis=1), np.maximum(steps[first] - 1.0 - grading.WALL_FLAT_M, rh + 0.1), face)
            # its top back to where the graded ground is up again (no trough behind it: a street above drove into one)
            full = gs >= (z + WALL_FULL * (gg - z))[:, None]
            back = np.argmax(full, axis=1)
            # (and a grid diagonal more: the terrain's lattice reaches the top that much later between its points)
            thick = np.where(full.any(axis=1), np.clip(steps[back] - face + WALL_DIAG_M, WALL_THICK_MIN_M,
                                                       WALL_THICK_MAX_M), thick)
            # (its top over the highest graded ground under it, not the one probe's: the bank poked through it in
            # patches where that probe sat in a dip)
            under = steps[None, :] >= face[:, None]
            top_g = np.maximum(gg, np.where(under, gs, -np.inf).max(axis=1))
            # (and never past where the ground behind falls away again: between a hairpin's legs the bank tops out
            # and drops to the other leg, and the top hung out over that slope, its underside seen from the road
            # above, the Stelvio Pass 2026-10-21)
            # (only past the grading's own flat strip and its climb behind the face, grading.WALL_FLAT_M / WALL_T_M:
            # that strip at the road's level read as the ground falling away and cut the top to its coping, a trench
            # between the wall and the slope behind it, "they float in front of the terrain", the user 2026-10-31)
            low = (gs < (top_g - WALL_SLAB_DZ)[:, None]) & (steps[None, :] > np.maximum(
                face + 1.0, core + grading.WALL_FLAT_M + grading.WALL_T_M)[:, None])
            reach = np.where(low.any(axis=1), steps[np.argmax(low, axis=1)] - face, np.inf)
            thick = np.minimum(thick, np.maximum(reach - 0.5, WALL_COPING_MIN_M))
            # the graded ground at the top's back edge (the steps', then the probe's: back_top)
            pd = core + grading.WALL_FLAT_M + grading.WALL_T_M + WALL_PROBE_M
            ground_at = lambda d, idx: np.array([np.interp(d[n], np.r_[steps, pd], np.r_[gs[m], gg[m]])  # noqa: E731
                                                 for n, m in enumerate(idx)])
        if parapets is not None and need:
            par = _parapets(xy, z, out_n, rh, inside, steps, gs, others, fid, decks, *parapets)
            if hx is not None:          # (round a widened hairpin: at its edge there, hairpin_extra)
                for w_ in par:
                    a_, b_ = w_["at"]
                    w_["base"] = w_["base"] + out_n[a_:b_] * hx[a_:b_, None]
            out.extend(par)
        full = _wall_runs(gg - z) & ~_under_decks(xy + out_n * core, z, decks) & ~bends
        # (no face standing on a carriageway: a hairpin's other leg, the same road, the Stelvio Pass 2026-10-21)
        full &= ~_on_carriageways(xy + out_n * face[:, None], others, 0.0)
        # (its ends lowered over WALL_TAPER_M toward the road, the whole wall's ends, not a tile edge's)
        taper = _taper(xy, full)
        cut = full & inside
        i = 0
        while i < len(cut):
            if not cut[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(cut) and cut[j + 1]:
                j += 1
            if j > i:
                sl = slice(i, j + 1)
                # (its top's depth evened out along it: the 1 m probe steps drew a jagged back edge)
                th = thick[sl]
                th = np.maximum.reduce([np.r_[th[:1].repeat(m), th[:len(th) - m]] for m in range(WALL_EVEN_N + 1)]
                                       + [np.r_[th[m:], th[-1:].repeat(m)] for m in range(1, WALL_EVEN_N + 1)])
                base = xy[sl] + out_n[sl] * face[sl][:, None]
                if not _cut_behind(base, out_n[sl], z[sl], ground):        # (a natural slope: no wall, WALL_STEEP)
                    i = j + 1
                    continue
                # (never over another road: a hairpin's wall top covered the road above, the Stelvio Pass, the user's
                # report 2026-10-21)
                th = np.minimum(th, _clear_depth(base, out_n[sl], th, others, fid))
                tp = np.minimum(top_g[sl], z[sl] + max_h) + WALL_TOP_M      # (no taller than that)
                tp = z[sl] + WALL_TAPER_LOW_M + (tp - z[sl] - WALL_TAPER_LOW_M).clip(0) * taper[sl]
                # (rock faces only as an experiment, BNE_CLIFF_WALLS=on, for tall rural cuts: the Stelvio's hairpins are
                # walled in dry stone, tall cuts too, Mapillary 2026-10-21; the user chose stone walls for passes)
                tall = CLIFF_WALLS_ON and float(np.max(top_g[sl] - z[sl])) > CLIFF_MIN_CUT_M
                top = np.maximum(tp, z[sl] + WALL_TAPER_LOW_M)
                # (its top behind the coping rises to the graded ground at its back edge, never under the coping: the
                # slope above goes on from the wall's top, held up by it; flat at the top's height, a band of grass
                # standing out in front of the slope read as a thick freestanding wall, the user 2026-10-31)
                back_top = top if ground_at is None else np.maximum(top, np.minimum(
                    ground_at(face[sl] + th, range(i, j + 1)), z[sl] + max_h + BACK_TOP_OVER_M))
                out.append({"base": base, "top": top, "back_top": back_top, "foot": z[sl] - WALL_FOOT_M,
                            "mountain": bool(mountain),
                            "toward": -out_n[sl], "thick": th, "concrete": concrete, "rock": rural and tall,
                            "road_z": z[sl], "clear": _clear_depth(base, out_n[sl], np.full(len(base), CLIFF_REACH_M),
                                                                   others, fid),
                            "edge": xy[sl] + out_n[sl] * (rh + CLIFF_EDGE_M)})
            i = j + 1
    return out


def _line_near(xy, other):
    """Per point of xy (k, 2): (distance to the polyline other (m, 2), the nearest point on it, its index t along it
    as a float segment index). Pure."""
    a, b = other[:-1], other[1:]
    d = b - a
    L2 = np.maximum((d * d).sum(1), 1e-12)
    out_d, out_q = np.empty(len(xy)), np.empty((len(xy), 2))
    for i, p in enumerate(xy):
        t = np.clip(((p - a) * d).sum(1) / L2, 0.0, 1.0)
        q = a + t[:, None] * d
        dd = np.hypot(*(q - p).T)
        j = int(np.argmin(dd))
        out_d[i], out_q[i] = dd[j], q[j]
    return out_d, out_q


def twin_tunnels(parts):
    """Pairs of tunnel parts ([(fid, spec, xy, z, lon)]) side by side: [(i, j, midline (k, 3), width)] with i < j their
    indices, the midline along part i (each point halfway to j's line, the floors' mean) and the width of a road
    surface covering both carriageways (the tube round both: roadmesh.tunnel). Each part in one pair at most. Pure."""
    import roadmesh
    taken, out = set(), []
    for i, (fa, sa, xa, za, _) in enumerate(parts):
        if i in taken or len(xa) < 2:
            continue
        xa = np.asarray(xa, np.float64)
        sj = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xa, axis=0).T))]
        if sj[-1] < 1.0:
            continue
        ss = np.linspace(0.0, sj[-1], max(2, int(sj[-1] / TWIN_STEP_M) + 1))
        P = np.c_[np.interp(ss, sj, xa[:, 0]), np.interp(ss, sj, xa[:, 1])]
        Z = np.interp(ss, sj, np.asarray(za, np.float64))
        for j in range(i + 1, len(parts)):
            fb, sb, xb, zb, _ = parts[j]
            if j in taken or len(xb) < 2:
                continue
            xb = np.asarray(xb, np.float64)
            reach = (sa.width + sb.width) / 2 + 2 * roadmesh.TUNNEL_SHOULDER + 1.0      # their tubes would overlap
            d, q = _line_near(P, xb)
            near = d < reach
            if near.mean() < TWIN_SHARE:
                continue
            tb = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xb, axis=0).T))]
            _, qb = _line_near(q, xb)           # (q on xb: its floor by the arc to it)
            sb_at = np.array([tb[k] + np.hypot(*(qb[n] - xb[k])) for n, k in enumerate(
                [int(np.argmin(np.hypot(*(xb[:-1] - p).T) + np.hypot(*(xb[1:] - p).T))) for p in qb])])
            zq = np.interp(sb_at, tb, np.asarray(zb, np.float64))
            if np.abs(Z[near] - zq[near]).max() > TWIN_DZ_M:
                continue
            mid = np.c_[(P + q) / 2, (Z + zq) / 2]
            k0, k1 = int(np.flatnonzero(near)[0]), int(np.flatnonzero(near)[-1])     # (where they run side by side)
            width = float(np.median(d[near])) + (sa.width + sb.width) / 2
            out.append((i, j, mid[k0:k1 + 1], width))
            taken.update((i, j))
            break
    return out


def _tunnel_chunks(tunnels, cells, exact, terrain_paint=True):
    """The chunks of the tunnels crossing the tile box `exact`: the road surface ("deck", "tunnel": the tube is
    built around it, roadmesh.tunnel) and its lines, every one "fixed" at the floor's heights; a surface chunk
    ending at its part's end carries "portal": [start is one, end is one]. Twin tunnels (twin_tunnels): their surfaces
    are floors without a tube ("no_tube"), and one "tube_only" chunk along their midline (not drawn as a road) carries
    the tube round both, with the first's portals."""
    import rampmerges, roadmarkings
    out = []
    if getattr(tunnels, "open_parts", None) is not None:
        for fid, spec, xy, z, lon in tunnels.open_parts(cells, exact):
            if not np.isfinite(z).all():
                continue
            pts = np.c_[xy, z]
            for look, w, line in [(look_of(spec), spec.width, pts)] + [
                    (lk, w2, ln) for lk, w2, ln in extras(spec, pts, lon, defer_dashes=True)
                    if lk["m"] not in (SIDEWALK["m"], GUTTER["m"], EDGE_DECAL["m"])]:
                for piece in clip_polyline(line, *exact):
                    for c in split(densify(simplify(piece))):
                        if len(c) >= 2:
                            # on the ground as any road (the grading lays the ground at its floor, Grader._open_tunnels:
                            # at the floor's own heights the ground graded from the grader's profile covered it
                            # outside 97th Street's arch, 2026-10-17)
                            out.append(_chunk(look, w, c))
    parts = [p for p in tunnels.parts(cells, exact) if np.isfinite(p[3]).all()]
    layouts = rampmerges.plans([(fid, sp, np.c_[xy, z], lon) for fid, sp, xy, z, lon in parts])
    twins = twin_tunnels(parts)
    twinned = {k for i, j, _, _ in twins for k in (i, j)}
    jobs = []                                   # (fid, its ends, [(look, width, line)])
    for i, j, mid, width in twins:              # the tube round both, with the first's portals
        if len(mid) >= 2:
            jobs.append((parts[i][0], (mid[0, :2], mid[-1, :2]), [({**TUNNEL, "tunnel": True, "tube_only": True}, width, mid)],
                         parts[i][1]))
    for n, (fid, spec, xy, z, lon) in enumerate(parts):
        pts = np.c_[xy, z]
        # A covered road needs its own colliding floor even when its mapped surface normally belongs to the
        # terrain paint (rock/grass). That paint lies on the hill/seabed above the bore, never inside it.
        look = _deck_look(spec)
        lines = [] if look is None else [({**look, "deck": True, "tunnel": True, "no_tube": n in twinned}, spec.width, pts)]
        for look, w, line in rampmerges.trim(fid, extras(spec, pts, lon, defer_dashes=True), layouts):
            if look["m"] not in (SIDEWALK["m"], GUTTER["m"], EDGE_DECAL["m"]):     # (no ground beside a deck)
                lines.append((look, w, line))
        jobs.append((fid, (pts[0, :2], pts[-1, :2]), lines, spec))
    for fid, ends, lines, spec in jobs:
        length = max((float(np.hypot(*np.diff(np.asarray(ln)[:, :2], axis=0).T).sum()) for _, _, ln in lines
                      if len(ln) >= 2), default=0.0)
        for look, w, line in lines:
            guard = None
            if look.get("tunnel") and getattr(tunnels, "frame", None) is not None:
                # Whole sections, in a global datum: neighbouring tiles deduplicate them; anchor rebases and
                # geotile pivots cannot put the water exclusion at a different height from its mesh.
                q = simplify(np.asarray(line, np.float64), ztol=.1)
                import geotiles
                la, lo, h = geotiles.world_to_geodetic(tunnels.frame, q)
                nodes = np.c_[np.round(la, 8), np.round(lo, 8), np.round(h, 3)].tolist()
                height = tunnels.height(fid) + .2     # enclose both polylines' simplification/rounding and end overlap
                identity = hashlib.sha1(json.dumps([nodes, w, height], separators=(",", ":")).encode()).hexdigest()[:16]
                guard = {"id": f"{fid}:{identity}", "r": w / 2 + roadmesh.TUNNEL_SHOULDER + 1.,
                         "h": height, "nodes": nodes}
            for piece in clip_polyline(line, *exact):
                for c in split(densify(simplify(piece, ztol=SIMPLIFY_Z_M))):
                    if len(c) >= 2:
                        ch = _chunk(look, w, c)
                        ch["fixed"] = True
                        if look.get("tunnel"):
                            if guard is not None:
                                ch["waterguard"] = guard
                            ch["tunnel"] = True
                            ch["deck"] = not look.get("tube_only")      # (a twin's tube: no floor, no road surface)
                            ch["tunnel_len"] = round(length, 1)             # (the whole sub-part's: sound_boxes)
                            if spec.cls in CONCRETE_WALL_CLASSES:
                                ch["concrete"] = True   # (a major road's low arch: concrete, as its cutting's walls)
                            for flag in ("no_tube", "tube_only"):
                                if look.get(flag):
                                    ch[flag] = True
                            if getattr(tunnels, "height", None) is not None:
                                ch["tunnel_h"] = tunnels.height(fid)
                            if getattr(tunnels, "portal_ups", None) is not None:
                                u0, u1 = tunnels.portal_ups(fid)
                                at = lambda p: u0 if np.hypot(*(p - ends[0])) < 0.5 else u1    # noqa: E731
                                ch["portal_up"] = [at(c[0, :2]), at(c[-1, :2])]
                            if getattr(tunnels, "cap_m", None) is not None:
                                c0, c1 = tunnels.cap_m(fid)
                                if c0 is not None or c1 is not None:
                                    atc = lambda p: c0 if np.hypot(*(p - ends[0])) < 0.5 else c1    # noqa: E731
                                    ch["cap_m"] = [atc(c[0, :2]), atc(c[-1, :2])]
                            m0, m1 = tunnels.mouth(fid) if getattr(tunnels, "mouth", None) is not None else (True, True)
                            isp = lambda p: bool((np.hypot(*(p - ends[0])) < 0.5 and m0) or      # noqa: E731
                                                 (np.hypot(*(p - ends[1])) < 0.5 and m1))
                            ch["portal"] = [isp(c[0, :2]), isp(c[-1, :2])]
                        out.append(ch)
    out.extend(rampmerges.chunks(layouts, exact, fixed=True))
    return roadmarkings.paint(out)


def _bridge_chunks(frame, bseen, elev, exact, bridges=None, cells=None, walks=None, shift_of=None, width_of=None,
                   context_of=None, context_m=None):
    """The chunks of the bridges crossing the tile box `exact` (deck surfaces flagged "deck", their
    centre / edge lines; every bridge chunk "fixed": its z is the deck's, not the ground's). Deck
    heights: from `bridges` (BridgeProfiles: whole chains, shared by every tile), else from the
    chains found among the given cells (deck_heights; tests). context_m: surrounding decks to retain (default
    DECK_CONTEXT_M); furniture needs enough context to decide the next support of an owned wire span."""
    if bridges is not None:
        margin = DECK_CONTEXT_M if context_m is None else float(context_m)
        near = (exact[0] - margin, exact[1] - margin, exact[2] + margin, exact[3] + margin)
        got = [(fid, spec, np.c_[xy, z], lon) for fid, spec, xy, z, lon in bridges.parts(cells, near)
               if np.isfinite(z).all()]
        if any(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() <= roadmesh.JUNCTION_MAX_PART_M
               for _fid, _sp, p, _lon in got):
            # A short branch can meet its fork outside the usual parapet context. Both tiles need all incident
            # complete profiles, even the tile holding only that branch's far abutment. parts() adds NEAR_M.
            reach = 3 * roadmesh.JUNCTION_MAX_PART_M + DECK_CONTEXT_M
            margin = max(0.0, reach - NEAR_M)
            x0, y0, x1, y1 = exact
            cell_fn = getattr(bridges, 'cell_fn', None)
            if cell_fn is not None:
                # Enlarging only the geometric filter is insufficient: parts() enumerates the supplied cells,
                # so a fork just across a source-cell boundary would still be missing from one tile's surface.
                la, lo, _ = frame.enu_to_geodetic(np.array([x0-reach, x1+reach, x0-reach, x1+reach]),
                                                  np.array([y0-reach, y0-reach, y1+reach, y1+reach]), 0.0)
                keys = mapcells.cells_for_bbox(float(la.min()), float(lo.min()), float(la.max()), float(lo.max()))
                cells = list(cells or ())
                known = {str(meta[0]) for f in cells if f is not None
                         for meta in [getattr(f, '_bnge_cell', None)] if meta}
                for key in keys:
                    if str(key) not in known:
                        features = cell_fn(key)
                        if features is None:
                            # (a neighbour cell not at hand, a slow first visit or offline: the tile is built without
                            # its fork and not cached as complete, built again next visit; raising failed the whole
                            # road tile, PR #720's first version)
                            import revisit
                            revisit.fault()
                            continue
                        cells.append(features)
            if context_of is not None:
                walks, shift_of, width_of = context_of(cells)
            got = [(fid, spec, np.c_[xy, z], lon) for fid, spec, xy, z, lon in
                   bridges.parts(cells, (x0 - margin, y0 - margin, x1 + margin, y1 + margin))
                   if np.isfinite(z).all()]
        dd = getattr(bridges, "depth_of", None)
        return _deck_chunks(got, exact, bridges._ground, {d[0]: dd(d[0]) for d in got} if dd else None, walks,
                            shift_of, width_of, context_m)
    if not bseen:
        return []
    parts = []                                  # (fid, spec, xy (k, 2), lon)
    for fid in sorted(bseen):
        feats, i, spec = bseen[fid]
        for ring in bridge_parts(feats, i):
            if len(ring) >= 2:
                e, n = mapgeom.enu_xy(feats, frame, ring)
                parts.append((fid, spec, densify(np.stack([e, n], axis=1), DECK_STEP),
                              float(np.mean(ring[:, 0]))))
    if not parts:
        return []

    def ground(xy):
        la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        h = np.asarray(elev(la, lo), np.float64)
        return frame.geodetic_to_enu(la, lo, h)[2]
    zs = deck_heights([p[2] for p in parts], ground)
    allxy = np.concatenate([p[2] for p in parts])
    gz = ground(allxy)                          # one call: the ground under every deck point
    decks, k = [], 0
    for (fid, spec, xy, lon), z in zip(parts, zs):
        g = gz[k:k + len(xy)]
        k += len(xy)
        if not np.isfinite(z).all():
            continue
        # never below the ground under it (2026-10-11: abutments on low spots buried decks), then
        # smoothed along it like a road profile (the ends keep their heights: the abutments)
        g = g + DECK_CLEAR
        z = np.maximum(z, g)
        if len(z) > 3:          # (the ends' trend kept: smoothed then pinned, the deck bent up beside an abutment)
            import grading
            z = np.maximum(_smooth_pinned(z, DECK_STEP, grading.SIGMA_M), g)
        decks.append((fid, spec, np.c_[xy, z], lon))
    return _deck_chunks(decks, exact, ground, walks=walks, shift_of=shift_of, width_of=width_of, context_m=context_m)


def _deck_look(spec):
    """A bridge deck's look: the concrete of a `surface=concrete` street (it stays concrete over the bridge) or of a
    motorway's or trunk road's (DECK_CONCRETE_CLASSES), else asphalt — and asphalt for rock/grass surfaces, whose ground mask cannot be painted on a deck at all."""
    if not terrain_surface(spec) and spec.kind == "paved" and spec.surf == "concrete":
        return CONCRETE
    if DECK_CONCRETE_ON and spec.kind == "paved" and spec.cls in DECK_CONCRETE_CLASSES:
        return DECK_CONCRETE
    return DECK


def _corner_look(point, roads):
    """A junction's corner-fill polygon look: the asphalt of the paved streets meeting there, or the concrete of
    them when a concrete street is within CORNER_CONCRETE_M of the corner (its wedges would otherwise be asphalt
    wedges in a concrete junction)."""
    px, py = float(point[0]), float(point[1])
    for _fid, spec, pts, _lon in roads:
        if spec.kind == "paved" and spec.surf == "concrete":
            if np.hypot(pts[:, 0] - px, pts[:, 1] - py).min() <= CORNER_CONCRETE_M:
                return CONCRETE
    return LOOK["paved"]


def _abutments(pts, ground_fn):
    """[(x, y)] of the ends of a deck part (pts (k, 3)) that land on the ground (within ABUT_GROUND_M); [] without
    ground_fn. Never raises."""
    if ground_fn is None or len(pts) < 2:
        return []
    try:
        ends = np.asarray(pts, np.float64)[[0, -1]]
        g = np.asarray(ground_fn(ends[:, :2]), np.float64)
        return [(float(e[0]), float(e[1])) for e, gz in zip(ends, g) if e[2] - gz <= ABUT_GROUND_M]
    except Exception:
        return []


def _fork(parts):
    """Whether the parts at a joint ([(line from the joint (k, 2), z, half width)]) make a fork or merge: every pair
    goes straight on from each other or runs on side by side (PAD_FORK_*), none at an angle. A ramp parting from its
    motorway on a viaduct had a junction pad: a box wider than both decks, its railings standing out past their
    parapets (the user, 2026-10-31, I-95 by the Middle Branch: "right at the start of a split the shoulder lane and
    barrier can bug out"): its pad reaches only the decks' own edges (PAD_FORK_MARGIN_M). Pure."""
    d = []
    for line, _z, _h in parts:
        ln = np.asarray(line, np.float64)[:, :2]
        s_ = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ln, axis=0).T))]
        v = ln[min(int(np.searchsorted(s_, 10.0)), len(ln) - 1)] - ln[0]
        n = float(np.hypot(*v))
        if n < 1e-6:
            return False
        d.append(v / n)
    for a in range(len(d)):
        for b in range(a + 1, len(d)):
            c = float(np.dot(d[a], d[b]))
            if PAD_FORK_STRAIGHT_COS < c < PAD_FORK_COS:
                return False
    return True


def deck_shift_fn(frame, cells, urban_fn=None, widths=None, left_hand=False):
    """shift_of(fid, pts (k, 2+)) -> per point the lateral shift (m, + left) of deck part fid at a merge / split
    (lane_shifts of its feature in cells, the tile's road specs), or None: one rule for the deck chunks (build_tile)
    and the junction pads round a tile (deck_pads' callers), so a fork's pad meets its decks."""
    where = copies_of(cells)

    def shift_of(fid, pts):
        got = where.get(str(fid))
        if not got:
            return None
        ls = joint_desc(got, lambda f: lane_shifts(f, urban_fn, widths, left_hand))
        if not ls:
            return None
        pts = np.asarray(pts, np.float64)
        la, lo, _ = frame.enu_to_geodetic(pts[:, 0], pts[:, 1], 0.0)
        return shift_at(ls, lo, la)
    return shift_of


def shifted_parts(parts, shift_of):
    """bridges.parts' (fid, spec, xy, z, lon) with each xy shifted by shift_of (deck_shift_fn)."""
    out = []
    for fid, spec, xy, z, lon in parts:
        sh = shift_of(fid, xy) if shift_of else None
        if sh is not None:
            xy = offset_polyline(np.c_[np.asarray(xy, np.float64)[:, :2], np.zeros(len(xy))], sh)[:, :2]
        out.append((fid, spec, xy, z, lon))
    return out


def _pad_height_groups(parts):
    """Split XY-coincident incident roads by endpoint height, with no transitive joins between levels.

    Height ordering makes membership independent of deck input order. Keep the original order within each
    group so an ordinary connected junction retains the native hull's existing input and tie behavior.
    """
    groups = []
    for i in sorted(range(len(parts)), key=lambda k: float(parts[k][1][0])):
        z = float(parts[i][1][0])
        if not groups or z - float(parts[groups[-1][0]][1][0]) > PAD_JOIN_Z_M:
            groups.append([])
        groups[-1].append(i)
    return [[parts[i] for i in sorted(group)] for group in groups]


def deck_pads(decks, ground_roads, box, ground_fn, own=True):
    """[(ring (k, 3), centre (3,), walls [(pts (m, 3), out (m, 2))])] of the junction pads of the joints of decks
    [(fid, spec, xy (k, 2), z (k,))] (the bridge parts round tile box, as bridges.parts gives them) whose joint the box
    owns (own False: every joint, the parapets' test round the tile); ground_roads [(fid, spec, pts (k, 2+), _)] the
    ground carriageways near it (build_tile roads_out), the ground's heights ground_fn(xy) (the decks' abutments').
    Only endpoints at matching elevations share a pad; XY-coincident stacked roads stay separate.
    walls: the rim's railings (PAD_*). earthfast.deck_pad, deck_pad_rim. Pure given ground_fn."""
    if not PADS_ON or not decks:
        return []
    import fast
    fast.require(113, "deck_pad_rim")
    join = _EndJoin(PAD_JOIN_M)
    nodes, inc = [], {}
    for fid, spec, xy, z in decks:
        xy = np.asarray(xy, np.float64)[:, :2]
        z = np.asarray(z, np.float64)
        if len(xy) < 2 or not np.isfinite(z).all():
            continue
        for at in (0, -1):
            nd = join(xy[at])
            if nd == len(nodes):
                nodes.append(xy[at].copy())
            line, zz = (xy, z) if at == 0 else (xy[::-1], z[::-1])
            inc.setdefault(nd, []).append((line, zz, spec.width / 2))
    out = []
    gl = [(np.asarray(pts, np.float64)[:, :2], spec.width / 2) for _fid, spec, pts, _ in ground_roads
          if spec is not None and len(pts) >= 2]
    joints = []
    for nd, incident in sorted(inc.items()):
        q = nodes[nd]
        if own and not box_owns(box, float(q[0]), float(q[1])):
            continue
        groups = _pad_height_groups(incident)
        for line, half in gl:
            for at in (0, -1):
                if np.hypot(*(line[at] - q)) < PAD_GROUND_M:
                    ln = line if at == 0 else line[::-1]
                    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ln, axis=0).T))]
                    keep = s <= PAD_REACH_M + 2 * PAD_STEP_M
                    ln = ln[keep] if keep.sum() >= 2 else ln[:2]
                    zz = np.asarray(ground_fn(ln), np.float64)
                    if not np.isfinite(zz).all():
                        continue
                    # A street underneath an elevated endpoint is not its approach. If a street could match
                    # two nearby groups, attach it only to the nearest one; it must not join them transitively.
                    matches = []
                    for k, group in enumerate(groups):
                        heights = [float(p[1][0]) for p in group]
                        if max(max(heights), zz[0]) - min(min(heights), zz[0]) <= PAD_JOIN_Z_M:
                            matches.append((abs(float(zz[0]) - float(np.mean(heights))), k))
                    if matches:
                        groups[min(matches)[1]].append((ln, zz, half))
        joints.extend((q, parts) for parts in groups)
    for q, parts in joints:
        if len(parts) < 2:
            continue
        if len(parts) == 2:
            d = []
            for line, _z, _h in parts:
                v = line[min(len(line) - 1, 1)] - line[0]
                d.append(v / max(np.hypot(*v), 1e-9))
            if float(np.dot(d[0], d[1])) < PAD_STRAIGHT_COS:
                continue                        # (a deck going on straight: its own end, an abutment, no corner)
        # a ramp parting from or joining its road (no corner to cut): its joint paved just to the decks' edges (their
        # slabs' PARAPET_W), the rim's railing the outer one where the parts' own stop, overlapping each other there
        margin = PAD_FORK_MARGIN_M if _fork(parts) else PAD_MARGIN_M
        pts = np.ascontiguousarray(np.concatenate([p[0] for p in parts]), np.float64)
        zz = np.ascontiguousarray(np.concatenate([p[1] for p in parts]), np.float64)
        off = np.r_[0, np.cumsum([len(p[0]) for p in parts])].astype(np.int64)
        halves = np.array([p[2] for p in parts], np.float64)
        hull = np.asarray(fast._ef.deck_pad(pts, zz, off, halves + margin, PAD_REACH_M, PAD_STEP_M), np.float64)
        if len(hull) < 3:
            continue
        zc = float(np.mean([p[1][0] for p in parts]))
        rim = np.asarray(fast._ef.deck_pad_rim(np.ascontiguousarray(hull), pts, off, halves, PAD_RIM_STEP_M,
                                               PAD_FORK_OPEN_M if margin == PAD_FORK_MARGIN_M else PAD_OPEN_M),
                         np.float64)
        walls = []
        if len(rim):
            drop = rim[:, 2] - np.asarray(ground_fn(rim[:, :2]), np.float64)
            wall = (rim[:, 5] < 0.5) & (drop >= PAD_WALL_DROP_M)
            # runs round the ring (one closing over its start joined)
            k = len(rim)
            if wall.all():
                walls.append((np.r_[rim[:, :3], rim[:1, :3]], np.r_[rim[:, 3:5], rim[:1, 3:5]]))
            elif wall.any():
                start = int(np.flatnonzero(~wall)[0])
                order = [(start + i) % k for i in range(1, k + 1)]
                run = []
                for i in order:
                    if wall[i]:
                        run.append(i)
                    elif run:
                        if len(run) >= 2:
                            walls.append((rim[run, :3], rim[run, 3:5]))
                        run = []
                if len(run) >= 2:
                    walls.append((rim[run, :3], rim[run, 3:5]))
        out.append((hull, np.array([q[0], q[1], zc]), walls))
    return out


def pad_inside(pads, xy, shrink=0.3, z=None):
    """Inside a junction pad by shrink m; with z, only at its local surface height.

    The optional height check uses the same centre fan as the pad mesh. A pad on a lower street must not
    remove an overpass's parapets, even if the pad's height range reaches the overpass elsewhere. Pure.
    """
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    heights = None if z is None else np.broadcast_to(np.asarray(z, np.float64), (len(xy),))
    out = np.zeros(len(xy), bool)
    for pad in pads:
        r = np.asarray(pad[0], np.float64)[:, :2]
        a, b = r, np.roll(r, -1, axis=0)
        e = b - a
        ln = np.maximum(np.hypot(*e.T), 1e-9)
        # signed distance inside every edge (counter-clockwise: left of each edge is inside)
        d = ((e[None, :, 0] * (xy[:, None, 1] - a[None, :, 1]) - e[None, :, 1] * (xy[:, None, 0] - a[None, :, 0]))
             / ln[None, :])
        inside = (d >= shrink).all(1)
        if heights is not None and inside.any():
            ring, centre = np.asarray(pad[0], np.float64), np.asarray(pad[1], np.float64)
            aa, bb = ring[:, :2] - centre[:2], np.roll(ring[:, :2], -1, axis=0) - centre[:2]
            det = aa[:, 0] * bb[:, 1] - aa[:, 1] * bb[:, 0]
            valid = np.abs(det) > 1e-10
            aa, bb, det = aa[valid], bb[valid], det[valid]
            delta = xy[:, None] - centre[:2]
            wa = (delta[:, :, 0] * bb[:, 1] - delta[:, :, 1] * bb[:, 0]) / det
            wb = (aa[:, 0] * delta[:, :, 1] - aa[:, 1] * delta[:, :, 0]) / det
            wc = 1.0 - wa - wb
            triangle = (wa >= -1e-8) & (wb >= -1e-8) & (wc >= -1e-8)
            local_z = wa * ring[valid, 2] + wb * np.roll(ring[:, 2], -1)[valid] + wc * centre[2]
            inside &= (triangle & (np.abs(local_z - heights[:, None]) <= PAD_JOIN_Z_M)).any(1)
        out |= inside
    return out


def _travel(spec):
    """+1 / -1: a one-way deck's traffic along / against its line, 0 two-way (roadmesh._median_gaps)."""
    if not spec.oneway:
        return 0
    return -1 if getattr(spec, "dir", 0) and spec.dir < 0 else 1


def _deck_frame(spec, walk):
    """(deck width m, its centreline's offset m to the left) of a deck with walk ((m, sides) or None): the carriageway
    and the sidewalks either side (one side: the deck off-centre that way)."""
    if not walk:
        return float(spec.width), 0.0
    wm, sides = walk
    return float(spec.width) + wm * len(sides), wm / 2 * float(sum(sides))


def _end_nodes(feats):
    """(rounded end point -> [(feature, 0 | 1)], (feature, 0 | 1) -> its end point) of a layer's features' first-to-last
    coordinates, made once (kept on it): deck_walks made them for the whole layer at each deck, ~0.5 s of a Manhattan
    tile set (2026-11-02). Pure given feats."""
    got = feats.__dict__.get("_end_nodes")
    if got is not None:
        return got
    nodes = {}                                  # rounded end point -> [(feature, 0 | 1)]
    for j in range(len(feats)):
        a = feats.coord_off[feats.ring_off[feats.part_off[j]]]
        b = feats.coord_off[feats.ring_off[feats.part_off[j + 1]]]
        if b - a < 2:
            continue
        for e, k in ((0, a), (1, b - 1)):
            nodes.setdefault((round(float(feats.lonlat[k, 0]), 6), round(float(feats.lonlat[k, 1]), 6)), []).append((j, e))
    ends = {}
    for key, lst in nodes.items():
        for j, e in lst:
            ends[(j, e)] = key
    got = feats.__dict__["_end_nodes"] = (nodes, ends)
    return got


def deck_walks(feats, i, ground_specs, left_hand=False):
    """(sidewalk width m, sides) a deck of feature i carries, or None: its own ground part's (ground_specs[i]: the
    tile's road_specs, land cover and all) where that has sidewalks, else those of the nearest road of its class with
    a ground part going on from it, along a chain of roads all bridge (DECK_WALK_CHAIN at most), on the side of the
    deck they are on: a deck between two others had none, its neighbours theirs, a 2.5 m step in the bridge's width
    (Baltimore by the Jones Falls Expressway, deckedges.py 2026-10-31). Pure."""
    gs = ground_specs[i]
    if gs is not None:
        return (float(gs.walk), tuple(walk_sides(gs, left_hand))) if has_sidewalks(gs) else None
    bridge = bridge_specs(feats)
    if bridge[i] is None:
        return None
    cls = bridge[i].cls
    nodes, ends = _end_nodes(feats)
    seen, frontier = {i}, [(i, False)]          # (feature, whether its direction runs against i's)
    for _ in range(DECK_WALK_CHAIN):
        nxt, best = [], None
        for k, flip in frontier:
            for e in (0, 1):
                key = ends.get((k, e))
                for j, ej in nodes.get(key, ()) if key is not None else ():
                    if j in seen:
                        continue
                    sp = ground_specs[j]
                    same_class = (sp or bridge[j]) is not None and (sp or bridge[j]).cls == cls
                    if not same_class:
                        continue
                    jflip = flip ^ (e == ej)    # (end to end or start to start: they run opposite ways)
                    if sp is not None:
                        if has_sidewalks(sp):
                            sides = tuple(walk_sides(sp, left_hand))
                            if jflip:
                                sides = tuple(-x for x in sides)
                            if best is None or sp.walk < best[0]:
                                best = (float(sp.walk), sides)
                    elif bridge[j] is not None:
                        seen.add(j)
                        nxt.append((j, jflip))
        if best is not None:
            return best
        frontier = nxt
        if not frontier:
            break
    return None


def _deck_chunks(decks, exact, ground_fn=None, depths=None, walks=None, shift_of=None, width_of=None, context_m=None):
    """Deck + line chunks of bridge parts [(fid, spec, pts (k, 3), lon)] clipped to the box `exact`, and their decks
    within context_m (default DECK_CONTEXT_M) as "ctx" chunks (not drawn: build_tile's caller hands them to roadmesh.deck).
    ground_fn(xy) -> heights: a part's ends on the ground are its abutments (each deck chunk's "abut" points: the
    parapets ramp down there). walks: {fid: (sidewalk m, sides)} of the decks that carry sidewalks (deck_walks).
    shift_of(fid, pts): its lane lines' shift at each point (lane_shifts) or None. width_of(fid, pts): its carriageway's
    width at each point where it tapers into another deck (deck_width_fn) or None."""
    import rampmerges, roadmarkings
    def geometry(fid, spec, pts):
        sh = shift_of(fid, pts) if shift_of else None
        pts = offset_polyline(pts, sh) if sh is not None else pts
        ws = width_of(fid, pts) if width_of else None
        return pts, spec.width if ws is None else ws
    layouts = rampmerges.plans(decks, geometry=geometry, pavement=geometry)
    walks = walks or {}
    out = []
    abut = {}
    ends = [(i, e, np.asarray(pts, np.float64)[e, :2]) for i, (_, _, pts, _) in enumerate(decks) if len(pts) >= 2
            for e in (0, -1)]
    for i, (fid, spec, pts, lon) in enumerate(decks):
        abut.setdefault(fid, []).extend(_abutments(pts, ground_fn))
        # an end no other deck part goes on from is an end of its parapets too (it stood there full height, square, on
        # the slab's edge: a short channel bridge's ends on the Sunshine Skyway causeway, 2026-10-27)
        for i2, e, q in ends:
            if i2 != i:
                continue
            if not any(j != i and np.hypot(*(q - q2)) < DECK_JOIN_M for j, _, q2 in ends):
                abut[fid].append((float(q[0]), float(q[1])))
    x0, y0, x1, y1 = exact
    margin = DECK_CONTEXT_M if context_m is None else float(context_m)
    wide = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    shifted = {}                                # (the deck itself shifted at a merge / split: lane_shifts)
    for i, (fid, spec, pts, lon) in enumerate(decks):
        sh = shift_of(fid, pts) if shift_of else None
        shifted[i] = offset_polyline(pts, sh) if sh is not None else pts
    # Complete profiles survive tile clipping. The bounded common field currently covers constant-width,
    # unshifted carriageways; keep existing sidewalk/taper/lane-shift geometry on its current surface.
    counts = {}
    for fid, _sp, _pts, _lon in decks:
        counts[fid] = counts.get(fid, 0) + 1
    joint_profiles, joint_excluded = {}, set()
    for i, (fid, spec, pts, _lon) in enumerate(decks):
        pts = np.asarray(pts, np.float64)
        if (counts[fid] != 1 or
                np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum() > roadmesh.JUNCTION_MAX_PART_M):
            continue
        joint_profiles[fid] = np.round(pts[:, :3], 3).tolist()
        ws = width_of(fid, pts) if width_of else None
        if (walks.get(fid) or not np.allclose(shifted[i], pts, rtol=0, atol=1e-6) or
                (ws is not None and not np.allclose(ws, spec.width, rtol=0, atol=1e-6))):
            # Retain the original endpoints for exclusion too: one unsupported arm must exclude the whole fork,
            # not leave three eligible arms sharing a surface beside the fourth arm's independent surface.
            joint_excluded.add(fid)
    # Keep only unsupported endpoints, not entire long profiles. A shared field must not
    # omit an incident long or ambiguous multipart arm, even when three short arms qualify.
    blocked_ends = []
    for fid, _spec, pts, _lon in decks:
        if fid not in joint_profiles:
            pts = np.asarray(pts, np.float64)
            if len(pts) >= 2:
                blocked_ends.extend(np.round(pts[[0, -1], :3], 3))
    for fid, profile in joint_profiles.items():
        ends = np.asarray(profile, np.float64)[[0, -1]]
        if any(np.linalg.norm(p[:2] - q[:2]) <= roadmesh.JUNCTION_JOIN_M and
               abs(p[2] - q[2]) <= roadmesh.JUNCTION_JOIN_Z_M for p in ends for q in blocked_ends):
            joint_excluded.add(fid)
    junction_margin = 3 * roadmesh.JUNCTION_MAX_PART_M + DECK_CONTEXT_M
    junction_box = (x0 - junction_margin, y0 - junction_margin, x1 + junction_margin, y1 + junction_margin)
    for i, (fid, spec, pts, lon) in enumerate(decks):
        pts = shifted[i]
        dw, dc = _deck_frame(spec, walks.get(fid))
        for piece in clip_polyline(offset_polyline(pts, dc) if dc else pts,
                                   *(junction_box if fid in joint_profiles else wide)):
            for c in split(densify(simplify(piece, ztol=SIMPLIFY_Z_M))):
                if len(c) >= 2:
                    ch = _chunk({**_deck_look(spec), "deck": True}, dw, c)
                    ch.update(fixed=True, deck=True, fid=fid, ctx=True, tdir=_travel(spec))
                    if fid in joint_profiles:
                        ch['jn'] = joint_profiles[fid]
                        ch['jskip'] = fid in joint_excluded
                    ws = width_of(fid, c) if width_of else None
                    if ws is not None:          # (tapering into another deck: roadmesh.ribbons' "ws")
                        ch["ws"] = [round(float(x + dw - spec.width), 3) for x in ws]
                    if ch["m"] == DECK_CONCRETE["m"]:
                        ch["tu"] = round(max(1.0, dw / DECK_CONCRETE_TILE_M), 3)
                    if depths and fid in depths:
                        ch["dd"] = round(float(depths[fid]), 3)
                    out.append(ch)
    for i, (fid, spec, pts, lon) in enumerate(decks):
        # at a merge / split the whole deck shifted toward the ramp (its lines with it): the far edge goes straight on
        # from the narrower part's and the added lane widens it on the ramp's side alone; centred on its line it
        # stepped out both sides (the I-95 viaduct by the Middle Branch against the aerial, the user 2026-10-31)
        pts = shifted[i]
        dw, dc = _deck_frame(spec, walks.get(fid))
        lines = [({**_deck_look(spec), "deck": True}, dw, offset_polyline(pts, dc) if dc else pts)]
        for look, w, line in rampmerges.trim(fid, extras(spec, pts, lon, ws=width_of(fid, pts) if width_of else None, defer_dashes=True), layouts):
            if look["m"] not in (SIDEWALK["m"], GUTTER["m"], EDGE_DECAL["m"]):     # (no ground beside a deck)
                lines.append((look, w, line))
        if walks.get(fid):                      # its sidewalks, flush on the deck beside the carriageway
            wm, sides = walks[fid]
            for side in sides:
                line = offset_polyline(pts, side * (spec.width / 2 + wm / 2))
                line[:, 2] += DECK_WALK_LIFT_M
                lines.append((DECK_WALK, wm, line))
        for look, w, line in lines:
            for piece in clip_polyline(line, *exact):
                for c in split(densify(simplify(piece, ztol=SIMPLIFY_Z_M))):
                    if len(c) >= 2:
                        ch = _chunk(look, w, c)
                        ch["fixed"] = True
                        ch['fid'] = fid
                        if ch["m"] == DECK_CONCRETE["m"]:    # (its tile two lanes wide: DECK_CONCRETE_TILE_M)
                            ch["tu"] = round(max(1.0, w / DECK_CONCRETE_TILE_M), 3)
                        if look.get("deck"):
                            ch["deck"] = True
                            if fid in joint_profiles:
                                ch['jn'] = joint_profiles[fid]
                                ch['jskip'] = fid in joint_excluded
                            ch["fid"] = fid             # (roadmesh.deck: no parapet between parts side by side)
                            ws = width_of(fid, c) if width_of else None
                            if ws is not None:          # (tapering into another deck: roadmesh.ribbons' "ws")
                                ch["ws"] = [round(float(x + w - spec.width), 3) for x in ws]
                            ch["tdir"] = _travel(spec)  # (roadmesh: opposite traffic beside it, a median)
                            if depths and fid in depths:
                                ch["dd"] = round(float(depths[fid]), 3)     # (roadmesh.deck: its slab and girders)
                            if abut.get(fid):
                                ch["abut"] = abut[fid]
                        out.append(ch)
    out.extend(rampmerges.chunks(layouts, exact, fixed=True))
    return roadmarkings.paint(out)


def _deck_clearance(ch):
    """A generated deck chunk's (points, half-width per segment, slab-plus-girder depth) for RoadClearance."""
    half = np.asarray(ch.get("ws", ch["w"]), np.float64) / 2
    if half.ndim:
        half = np.maximum(half[:-1], half[1:])
    return np.asarray(ch["n"], np.float64), half, float(ch.get("dd") or DECK_FULL_DEPTH)


def _on_box(p, box, eps=1e-6):
    x0, y0, x1, y1 = box
    return abs(p[0] - x0) < eps or abs(p[0] - x1) < eps or abs(p[1] - y0) < eps or abs(p[1] - y1) < eps


def drains(run, side, caps, walk_w=None):
    """The storm drain chunks along a sidewalk run (k, 3) on `side` of its road: DRAIN_IN m before its
    end if that is a real end (caps: (start, end) real) and every DRAIN_EVERY m along it. Pure."""
    import sidewalks
    curb, out = sidewalks.run_geometry(run, side, walk_w or SIDEWALK["w"])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(curb[:, :2], axis=0).T))]
    L = s[-1]
    if L < 2 * DRAIN_IN:
        return []
    at = []
    if caps[1]:
        at.append(L - DRAIN_IN)
    if L > DRAIN_EVERY * 1.5:
        at += list(np.arange(DRAIN_EVERY, L - DRAIN_EVERY / 2, DRAIN_EVERY))
    res = []
    for a in sorted(at):
        pts = []
        for d in (a - DRAIN["len"] / 2, a + DRAIN["len"] / 2):
            x = np.interp(d, s, curb[:, 0])
            y = np.interp(d, s, curb[:, 1])
            z = np.interp(d, s, curb[:, 2])
            k = min(int(np.searchsorted(s, d)), len(s) - 1)
            o = out[k] / max(np.hypot(*out[k]), 1e-9)
            pts.append([x - o[0] * (GUTTER["w"] / 2), y - o[1] * (GUTTER["w"] / 2), z])
        res.append(_chunk(DRAIN, DRAIN["w"], np.array(pts)))
    return res


def _going_on(roads):
    """{(fid, fid2)} (both ways) of the roads [(fid, spec, pts (k, 3), lon)] whose gravel edge decals need not keep
    EDGE_DECAL_GAP_M off each other's carriageway (meant for a ramp's gore, it left a ~10 m gap on the far side of every
    T junction, the user 2026-10-28, Carroll County MD): roads at one junction node, not both one-way (a merge's or a
    diverge's gore keeps it) and the second without sidewalks (its slab's end), and a road going on in another at a
    shared end (the map splits a road at every junction:
    the straightest of each other's there). Pure."""
    at_node = {}                                # node key -> {fid: one-way}
    for fid, sp, pts, _ in roads:
        if len(pts) < 2 or sp.kind not in BUILT:
            continue
        keys = np.round(np.asarray(pts[:, :2], np.float64) / junctions.NODE_TOL).astype(np.int64)
        for key in {(int(a), int(b)) for a, b in keys}:
            at_node.setdefault(key, {})[fid] = (bool(sp.oneway), has_sidewalks(sp))
    out = set()
    for key in sorted(at_node):
        fids = at_node[key]
        if len(fids) >= 2:
            for f1 in fids:
                for f2 in fids:
                    if f1 != f2 and not (fids[f1][0] and fids[f2][0]) and not fids[f2][1]:
                        out.add((f1, f2))       # (f1's decal by f2: up to f2's curb when it has one)
    return out | _straight_on(_road_ends(roads))


def _road_ends(roads):
    """{node key: [(fid, outward unit direction)]} of the road ends [(fid, spec, pts (k, 3), lon)]. Pure."""
    ends = {}
    for fid, sp, pts, _ in roads:
        if len(pts) < 2 or sp.kind not in BUILT:
            continue
        for a, b in ((0, 1), (-1, -2)):
            d = pts[a, :2] - pts[b, :2]
            n = math.hypot(d[0], d[1])
            if n > 1e-6:
                key = np.round(np.asarray(pts[a, :2], np.float64) / junctions.NODE_TOL).astype(np.int64)
                ends.setdefault((int(key[0]), int(key[1])), []).append((fid, d / n))
    return ends


def _straight_on(ends):
    """{(fid, fid2)} (both ways) of the roads going on in each other at a node of ends (_road_ends): each the straightest
    of the other's there (a ramp leaving a motorway is not its going on). Pure."""
    out = set()
    lim = -math.cos(math.radians(GOING_ON_DEG))
    for key in sorted(ends):
        at = ends[key]
        best = {}
        for fid, u in at:
            # the straightest other road ending here: going on is turning least (outward directions opposite)
            got = min(((float(u @ v), f2) for f2, v in at if f2 != fid), default=None)
            if got is not None and got[0] <= lim:
                best[fid] = got[1]
        for fid, f2 in best.items():
            if best.get(f2) == fid:
                out.add((fid, f2))
                out.add((f2, fid))
    return out


def _off_line(q, pts):
    """m from point q (2,) to the polyline pts (k, 2+). Pure."""
    a, d = pts[:-1, :2], np.diff(pts[:, :2], axis=0)
    t = np.clip(((q - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-9), 0.0, 1.0)
    return float(np.hypot(*(a + d * t[:, None] - q).T).min())


def line_through(roads):
    """Which roads a road's lines run past (the user's report 2026-11-02: short stretches with no markings between
    marked ones; every road cut every other's lines, 5.4% of the centre lines around Boras in bites of 5-10 m at every
    side road, drive and way split): (straight, stems, whole). straight: {(fid, f2)} of a road going on in f2 at a node
    (no line stops there, edge lines included); stems: {(fid, f2)} of f2 joining fid's road from the side, a lesser road
    or the stem of a T (its centre and lane lines run across the mouth, its edge lines stop); whole: the fids going on
    at both ends (a piece of a through road between two junctions, not a road inside one). Pure."""
    ends = _road_ends(roads)
    spec = {fid: sp for fid, sp, _, _ in roads}
    pairs = {p for p in _straight_on(ends) if spec[p[0]].oneway == spec[p[1]].oneway}
    stems = set()
    count = {}
    goes_on = set()                             # (fid, node key) of a road going on past that end
    for key, at in ends.items():
        here = {f for f, _ in at}
        through = {f for f in here for g in here if (f, g) in pairs}
        for f in through:
            count[f] = count.get(f, 0) + 1
            goes_on.add((f, key))
            stems.update((f, g) for g in here - through)
    # a lesser road ending on it (a drive, a side road the map does not split it at); one going on across it still cuts
    tol = junctions.NODE_TOL
    for fid, sp, pts, _ in roads:
        rank = LINE_RANK.get(sp.cls)
        if rank is None or len(pts) < 2:
            continue
        lo, hi = pts[:, :2].min(0) - sp.width, pts[:, :2].max(0) + sp.width
        for f2, s2, p2, _ in roads:
            if f2 == fid or LINE_RANK.get(s2.cls, 1) >= rank or len(p2) < 2:
                continue
            on = [p2[e, :2] for e in (0, -1) if (lo <= p2[e, :2]).all() and (p2[e, :2] <= hi).all()
                  and _off_line(p2[e, :2], pts) <= sp.width / 2 + tol]
            if on and not any((f2, tuple(int(v) for v in np.round(q / tol))) in goes_on for q in on):
                stems.add((fid, f2))
    return pairs, stems, {f for f, n in count.items() if n >= 2}
