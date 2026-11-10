"""Road builder: bridge and tunnel structures (split out of roads.py): surface pieces, deck depth and clearance, pier
sites, BridgeProfiles / TunnelProfiles and the chain store.
"""
import math
import threading
from collections import OrderedDict
import numpy as np
import mapcells
import mapgeom
import roadmesh
import roadbase
from roadbase import (
    ARCH_SIZE_SKIP_M,
    AT_GRADE_REACH_M,
    BRIDGE_HEADROOM_MARGIN_M,
    CHAIN_CELLS,
    CONCRETE_WALL_CLASSES,
    DECK_CLEAR_REACH_M,
    DECK_DESIGN_GRADE,
    DECK_FULL_DEPTH,
    DECK_MIN_DEPTH,
    DECK_SPAN_RATIO,
    DECK_STEP,
    DIRT_LOOK,
    DIRT_SPLIT,
    END_ON_STEP_M,
    FULL_DEEP_M,
    FULL_MIN_M,
    GAP_DIP_M,
    GAP_LOW_M,
    GAP_SEA_M,
    MAJOR_LOW_H,
    MIN_BRIDGE_HEADROOM_M,
    NAV_RUN_M,
    NEAR_M,
    OBS_BLEND_DG,
    OBS_BLEND_MIN_M,
    OBS_END_MAX_M,
    OBS_JOINT_GRADE,
    OBS_JOINT_SHORT_M,
    OBS_MAX_GRADE,
    PIER_CHANNEL_CLEAR,
    PIER_IN_WATER_M,
    PIER_MIN_CLEAR,
    PIER_ROAD_M,
    PIER_SPAN_M,
    PIER_SPAN_MAX,
    PIER_SPAN_PER_M,
    PIN_M,
    PORTAL_GROUND_M,
    PORTAL_LOOK_M,
    PORTAL_UP_MIN,
    ROAD_CAP_GAP_M,
    ROAD_CAP_JOIN_M,
    ROAD_CAP_OVER_M,
    SECTION_MIN_M,
    STRIP_JOIN_M,
    TUNNEL_COVER,
    TUNNEL_LOW_H,
    TUNNEL_OPEN_COVER,
    TUNNEL_SOIL_M,
    TUNNEL_WET_COVER_M,
    TUNNEL_WET_RUN_M,
    TUNNEL_WET_SHARE,
    WATER_STEP_M,
    densify,
    log,
    split,
)
from roadspecs import (
    bridge_levels,
    bridge_parts,
    bridge_specs,
    ground_parts,
    road_specs,
    tunnel_parts,
    tunnel_specs,
)
from roaddecks import (
    _EndJoin,
    _deck_beside,
    _deck_crossings,
    _dedupe,
    _end_joins,
    _end_ons,
    _ground_pins,
    _joined_lines,
    _tie_beside,
    bridge_profile_mode,
    deck_profile,
    water_clear,
)
from roadgeom import (
    cover_roads,
    portal_cover,
)
from roadwalks import (
    clip_polyline,
    simplify,
)
from roadtiles import (
    box_owns,
)


def _chunk(look, w, c):
    out = {"m": look["m"], "w": round(float(w), 2), "tl": look["tl"], "p": look["p"],
           "n": [[round(float(v), 3) for v in p] for p in c]}
    if look.get("layer"):
        out["layer"] = look["layer"]
    if isinstance(look.get("dash"), tuple):
        import roadmarkings
        out.update(dash=look["dash"], phase=roadmarkings.phase(look.get("_source", c), c[0]))
    if "ramp_merge" in look:
        out["ramp_merge"] = look["ramp_merge"]
    return out


def _surface_pieces(look, width, line, box, ends=(True, True)):
    """The carriageway chunks of `line` (k, 3) cut to box. A dirt or gravel strip (look "aspect", DIRT_LOOK) is rounded
    at its sharp bends first, the whole line (roadmesh.round_dirt_centerline, before the 12 m densify; every tile rounds
    the same arcs), and each chunk carries s0 / s1 (m along the line to its ends: the ruts run on across a tile's edge),
    slen (the line's length) and fends (ends: which of the line's ends are open, its strip fading out there:
    _strip_open_ends). From PR #455; the distance rides through the cut as a fourth column. Pure."""
    line = np.asarray(line, np.float64)
    if not look.get("aspect") or len(line) < 2:
        return [_chunk(look, width, c) for piece in clip_polyline(line, *box)
                for c in split(densify(simplify(piece))) if len(c) >= 2]
    rl = densify(roadmesh.round_dirt_centerline(simplify(line[:, :3]), float(width) / 2.0))
    d = np.diff(rl[:, :2], axis=0)
    s = np.r_[0.0, np.cumsum(np.hypot(d[:, 0], d[:, 1]))]
    slen = round(float(s[-1]), 2)
    fends = [int(bool(ends[0])), int(bool(ends[1]))]
    out = []
    for piece in clip_polyline(np.c_[rl, s], *box):
        for c in split(piece, DIRT_SPLIT):
            if len(c) >= 2:
                ch = _chunk(look, width, c[:, :3])
                ch.update(s0=round(float(c[0, 3]), 3), s1=round(float(c[-1, 3]), 3), slen=slen, fends=fends)
                out.append(ch)
    return out


def _strip_open_ends(roads, rings, seen, near, skip=lambda spec: False):
    """{index in roads: (start open, end open)} of the dirt and gravel strips among roads ((fid, spec, centreline, lon),
    build_tile's, rings its lon/lat parts in the same order, seen fid -> (feats, i, spec)). An end is open (its strip
    fades out, roadmesh END_FADE_M) unless another strip road's line passes within STRIP_JOIN_M of it (a junction, the
    next segment of the road: a fade there left a gap in the road), the road comes back to it (a loop), or it is not an
    end of its feature's own line (the road goes on over a bridge or through a tunnel). Only the ends in the box near
    (x0, y0, x1, y1: NEAR_M round the tile) are tested, against the lines there: every road at such an end passes
    through it, so the tiles agree. skip(spec): roads without a strip (rough ground). Pure."""
    idx = [k for k, r in enumerate(roads) if r[1].kind in DIRT_LOOK and not skip(r[1])]
    if not idx:
        return {}
    cell = STRIP_JOIN_M
    nx0, ny0, nx1, ny1 = near
    V = np.concatenate([roads[k][2][:, :2] for k in idx])
    R = np.repeat(np.arange(len(idx)), [len(roads[k][2]) for k in idx])
    same = R[:-1] == R[1:]
    A, B, SR = V[:-1][same], V[1:][same], R[:-1][same]
    lo, hi = np.minimum(A, B), np.maximum(A, B)
    keep = (hi[:, 0] >= nx0 - cell) & (lo[:, 0] <= nx1 + cell) & (hi[:, 1] >= ny0 - cell) & (lo[:, 1] <= ny1 + cell)
    A, B, SR = A[keep], B[keep], SR[keep]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-12)
    tips = np.array([[roads[k][2][0, :2], roads[k][2][-1, :2]] for k in idx])      # (n, 2 ends, 2)
    loop = np.hypot(*(tips[:, 0] - tips[:, 1]).T) < cell
    joined = np.zeros((len(idx), 2), bool)
    for e in (0, 1):
        E = tips[:, e]
        inn = np.nonzero((E[:, 0] >= nx0) & (E[:, 0] <= nx1) & (E[:, 1] >= ny0) & (E[:, 1] <= ny1))[0]
        for blk in range(0, len(inn), 256):
            q = inn[blk:blk + 256]
            D = E[q][:, None, :] - A[None]                               # (ends, segments, 2)
            t = np.clip((D * AB[None]).sum(2) / L2[None], 0.0, 1.0)
            dist = np.hypot(*(D - t[..., None] * AB[None]).transpose(2, 0, 1))
            other = SR[None, :] != q[:, None]
            joined[q, e] = ((dist <= cell) & other).any(1)
    out = {}
    for n, k in enumerate(idx):
        own = (True, True)
        got = seen.get(roads[k][0])
        if k < len(rings) and got is not None:          # (a ground part's end inside its feature's line: it goes on)
            ring = rings[k][2]
            feats, i, _ = got
            ft = [r_[t] for part in feats.parts(i) for r_ in part if len(r_) >= 2 for t in (0, -1)]
            own = tuple(any(abs(float(q[0]) - float(p[0])) < 1e-9 and abs(float(q[1]) - float(p[1])) < 1e-9
                            for q in ft) for p in (ring[0], ring[-1]))
        out[k] = (own[0] and not joined[n, 0] and not loop[n], own[1] and not joined[n, 1] and not loop[n])
    return out


def deck_depth(span):
    """m of structure (slab + girders) under a deck's road for a main span of span m. A short overpass or channel
    span drew 2.2 m of slab and girders whatever its span, and had to climb that much more to clear what is under it
    (a crown on 24 m overpasses and 2.7 m humps on the Sunshine Skyway causeway's channel bridges, 2026-10-27). Pure."""
    return float(np.clip(float(span) / DECK_SPAN_RATIO, DECK_MIN_DEPTH, DECK_FULL_DEPTH))


def deck_split(depth):
    """(slab m, girders m) of a deck depth m: the full deck's 1.2 : 1.0 kept. Pure."""
    slab = roadmesh.DECK_DEPTH * float(depth) / DECK_FULL_DEPTH
    return slab, float(depth) - slab


def deck_road_clear(depth=None):
    """m a deck's road stands over a road under it: the headroom, its margin and the deck's depth. Pure."""
    return MIN_BRIDGE_HEADROOM_M + BRIDGE_HEADROOM_MARGIN_M + (DECK_FULL_DEPTH if depth is None else float(depth))


def geom_span(lines):
    """m: a bridge chain's main span from its geometry alone: its longest part, or (one longer than 1.5 PIER_SPAN_M)
    that part's pier spacing (pier_sites' PIER_SPAN_M grid). Pure."""
    L = max((float(np.hypot(*np.diff(np.asarray(ln, np.float64)[:, :2], axis=0).T).sum())
             for ln in lines if len(ln) >= 2), default=0.0)
    if L <= 1.5 * PIER_SPAN_M:
        return L
    return L / max(1, int(round(L / PIER_SPAN_M)))


def _rail_drawn(feats, i):
    """Is feature i a railway line drawn on the map (railways.drawn: worn ones too)?"""
    import railways
    return railways.drawn(feats, i) is not None


def _clearance_deficits(lines, zs, crossings, depth=None):
    """(part, usable m) below the road target in a modelled crossing corridor.

    Road heights are the graded lower road's height at the crossing. Keep true
    abutment junctions at grade; data/grade constraints cannot promise global clearance.
    """
    out = []
    for pi, sc, zc in crossings:
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(lines[pi][:, :2], axis=0).T))]
        if min(sc, s[-1] - sc) <= AT_GRADE_REACH_M:
            continue
        lo, hi = max(0.0, sc - DECK_CLEAR_REACH_M), min(s[-1], sc + DECK_CLEAR_REACH_M)
        at = np.r_[lo, s[(s > lo) & (s < hi)], hi]
        headroom = float(np.min(np.interp(at, s, zs[pi]))) - zc - (DECK_FULL_DEPTH if depth is None else depth)
        if not np.isfinite(headroom) or headroom < MIN_BRIDGE_HEADROOM_M - 1e-6:
            out.append((pi, headroom))
    return out


def _crossings(lines, ground_lines):
    """(part index, arc m, point xy) where a ground road (polylines in metres) crosses under a deck
    part (earthfast.segment_crossings). Pure."""
    import fast
    lp = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    gp = [np.asarray(gl, np.float64)[:, :2] for gl in ground_lines]
    if not lp or not gp:
        return []
    fast.require(57, "segment_crossings")
    starts = lambda ps: np.r_[0, np.cumsum([len(p) for p in ps])].astype(np.int64)
    got = fast._ef.segment_crossings(np.ascontiguousarray(np.vstack(lp)), starts(lp),
                                     np.ascontiguousarray(np.vstack(gp)), starts(gp))
    return [(pi, arc, np.array([x, y])) for pi, arc, x, y in got]


def _crossings_by(lines, ground_lines):
    """_crossings for each of ground_lines apart, in one kernel call: [[(part index, arc m, point xy)] per ground line],
    each list as _crossings(lines, [that line]) gives it. Pure."""
    import fast
    lp = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    gp = [np.asarray(gl, np.float64)[:, :2] for gl in ground_lines]
    out = [[] for _ in gp]
    if not lp or not gp:
        return out
    fast.require(76, "segment_crossings_g")
    starts = lambda ps: np.r_[0, np.cumsum([len(p) for p in ps])].astype(np.int64)
    for pi, g, arc, x, y in fast._ef.segment_crossings_g(np.ascontiguousarray(np.vstack(lp)), starts(lp),
                                                          np.ascontiguousarray(np.vstack(gp)), starts(gp)):
        out[g].append((pi, arc, np.array([x, y])))
    return out


def pier_sites(xy, z, width, ground_fn, road_lines=(), wet_fn=None, depth=None):
    """The piers under a bridge deck (xy (k, 2), deck top z (k,), width m): [(x, y, ux, uy, z_top, z_bottom, width)],
    (ux, uy) along the deck, z_top under its girders (roadmesh.DECK_DEPTH + GIRDER_D), z_bottom in the ground (or
    PIER_IN_WATER_M under water); evenly spaced inside the part (its ends are abutments or joints) at PIER_SPAN_M,
    longer under a higher deck; none where the deck is within PIER_MIN_CLEAR of the ground, moved along up to 40 % of
    a span off a road under it (PIER_ROAD_M) or left out; none in a ship channel. The official maps' bridges stand
    on piers (Italy's tapered walls, East Coast's) where ours floated (the user, 2026-10-17: "Bridges need a lot of
    work in general"). Pure given ground_fn(xy) -> heights, wet_fn(xy) -> bool."""
    import fast
    import roadmesh
    xy = np.asarray(xy, np.float64)
    if len(xy) < 2 or not np.isfinite(z).all():
        return []
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    L = float(s[-1])
    g_all = np.asarray(ground_fn(xy), np.float64)
    h = float(np.median(z - g_all))
    span = float(np.clip(PIER_SPAN_M + PIER_SPAN_PER_M * max(h, 0.0), PIER_SPAN_M, PIER_SPAN_MAX))
    n = int(round(L / span))
    if n < 2:
        return []
    under = DECK_FULL_DEPTH if depth is None else float(depth)
    segs = [np.asarray(ln, np.float64) for ln in road_lines if len(ln) >= 2]
    A = np.concatenate([ln[:-1] for ln in segs]) if segs else np.zeros((0, 2))
    B = np.concatenate([ln[1:] for ln in segs]) if segs else np.zeros((0, 2))
    wet_runs = None
    if wet_fn is not None:
        w = np.asarray(wet_fn(xy), bool)
        wet_runs = (w, s)
    out = []
    pw = float(np.clip(0.7 * width, 2.5, 14.0))
    for i in range(1, n):
        placed = None
        for off in (0.0, 0.25, -0.25, 0.4, -0.4):
            sc = L * i / n + off * span
            if sc < 0.5 * span or sc > L - 0.5 * span:
                continue
            p = np.array([np.interp(sc, s, xy[:, 0]), np.interp(sc, s, xy[:, 1])])
            j = min(int(np.searchsorted(s, sc)), len(s) - 1)
            t = xy[j] - xy[max(j - 1, 0)]
            t = t / max(float(np.hypot(*t)), 1e-9)
            nrm = np.array([-t[1], t[0]])
            foot = np.array([p + nrm * pw / 2, p - nrm * pw / 2, p])
            if len(A):
                fast.require(0, "points_near_segments")
                hit = fast._ef.points_near_segments(foot, A, B, np.full(len(A), PIER_ROAD_M))
                if np.asarray(hit).any():
                    continue
            zt = float(np.interp(sc, s, z)) - under
            g = float(ground_fn(p[None])[0])
            if zt - g < PIER_MIN_CLEAR:
                continue
            zb = g - roadmesh.FOOTING_M
            if wet_runs is not None:
                w, ss = wet_runs
                k = int(np.argmin(np.abs(ss - sc)))
                if w[k]:
                    a = k
                    while a > 0 and w[a - 1]:
                        a -= 1
                    b = k
                    while b + 1 < len(w) and w[b + 1]:
                        b += 1
                    if water_clear(ss[b] - ss[a]) >= PIER_CHANNEL_CLEAR and a < k < b:
                        continue
                    zb = g - PIER_IN_WATER_M
            placed = (float(p[0]), float(p[1]), float(t[0]), float(t[1]), zt, zb, float(width))
            break
        if placed is not None:
            out.append(placed)
    return out


class BridgeProfiles:
    """Deck heights per bridge part for one anchor frame, each chain resolved once from all of it, so
    every tile gets the same deck (2026-10-11: computed per tile from the tile's cells, a chain's
    two halves disagreed at a tile seam). cell_fn(key) -> the roads Features of a map cell; elev:
    the physics elevation. Worker threads."""

    wet = None      # wet(lat, lon) -> water depth (m) or None: decks clear the water (water_clear)
    infra_fn = None     # infra_fn(cell key) -> the infrastructure Features (bridge:structure tags: bridgetypes)

    def __init__(self, frame, cell_fn, elev):
        self.frame, self.cell_fn, self.elev = frame, cell_fn, elev
        self.done = {}                          # (fid, j) -> (spec, xy (k, 2), z (k,), lon)
        self.piers = {}                         # (fid, j) -> pier_sites of its deck
        self.trusses = {}                       # a chain's first part key -> its truss beams (bridgetypes)
        self.depth = {}                         # (fid, j) -> m of slab and girders under its road (deck_depth)
        self.lock = threading.Lock()

    def _water(self, lines, nav=0.0, depth=None, channel=0.0):
        """Clearance requirements over water (deck_profile_once crossings with their own clearance): every
        WATER_STEP_M along each wet run of each line, water_clear of the run's length over the ground there; over a
        run of NAV_RUN_M or more at least nav (m: the bridge's navigation clearance, bridgetypes.nav_clearance) plus
        the slab and girders under the road."""
        out = []
        if self.wet is None:
            return out
        under = DECK_FULL_DEPTH if depth is None else float(depth)
        for k, ln in enumerate(lines):
            xy = np.asarray(ln, np.float64)[:, :2]
            if len(xy) < 2:
                continue
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
            w = np.asarray(self.wet(la, lo), np.float64) > 0.5
            if not w.any():
                continue
            g = self._ground(xy)
            sea = self.frame.geodetic_to_enu(la, lo, np.zeros(len(la)))[2]
            i = 0
            while i < len(w):
                if not w[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(w) and w[j + 1]:
                    j += 1
                c = water_clear(s[j] - s[i] + (s[1] - s[0]))
                base = g
                if nav > 0:
                    # the inventory's clearance decides over the share-of-span rule, from the water's surface (sea
                    # level) not the carved bed: a causeway's 100 m channel bridge with 3.8 m of clearance was asked
                    # for 15 m over a 3 m deep bed and humped to its grade cap (the Sunshine Skyway causeway, 2026-10-27)
                    c = nav + under if s[j] - s[i] >= NAV_RUN_M else min(c, nav + under)
                    base = np.maximum(g, sea)
                run = s[j] - s[i]
                # the nav clearance holds over the channel only: the main span (channel m: NBI item 48) over the
                # run's deepest water, else the middle half of the run; outside it the clearance asked eases off at the
                # design grade to the run's own (water_clear: a causeway's low). Held flat across the whole run the
                # Sunshine Skyway stood 58 m up for 3.5 km of its causeway, and where the grade cap only just reached
                # the run's start the solver humped the Vincent Thomas Bridge 5 m over its span (2026-10-28)
                c_low = min(c, water_clear(run + (s[1] - s[0])))
                if nav > 0 and channel > 0 and 3 * channel < run:      # (a river crossing: its whole run)
                    seg = np.arange(i, j + 1)
                    mid = float(s[seg[int(np.argmin(g[seg]))]])
                    mid = float(np.clip(mid, s[i] + channel / 2, s[j] - channel / 2))
                    c0, c1 = mid - channel / 2, mid + channel / 2
                else:
                    c0, c1 = s[i] + run / 4, s[j] - run / 4
                for sc in np.arange(s[i], s[j] + 1e-6, WATER_STEP_M):
                    out_by = max(0.0, c0 - sc, sc - c1)
                    cc = max(c_low, c - DECK_DESIGN_GRADE * out_by) if nav > 0 else c
                    out.append((k, float(sc), float(np.interp(sc, s, base)), cc))
                i = j + 1
        return out

    def _ground(self, xy):
        la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        return self.frame.geodetic_to_enu(la, lo, np.asarray(self.elev(la, lo), np.float64))[2]

    specs_of = staticmethod(lambda feats: bridge_specs(feats))
    parts_of = staticmethod(lambda feats, i: bridge_parts(feats, i))
    levels_of = staticmethod(lambda feats, i: bridge_levels(feats, i))

    def _load(self, key, pool, ground):
        """Cell key's bridge parts into pool and its ground roads into ground (as the chain solves gather them)."""
        parts, roads_ = self._cell_entries(key)
        for k, v in parts:
            # a part in several cells can come clipped differently: the longest (the same part whichever
            # cell a process loads first: the terrain and the road tiles took different ones, 2026-10-17)
            old = pool.get(k)
            if old is None or v[4] > old[4]:
                pool[k] = v
        for k, lines in roads_:
            if k not in ground:
                ground[k] = lines

    def _cell_entries(self, key):
        """([((fid, j), (spec, deck line, lon, level, ring points))], [((fid, "g"), [ground lines])]) of cell key, made
        once (every chain solved gathered its cells again: each a loop over all their roads, 2026-10-25). Not to be
        changed by the caller."""
        cache = self.__dict__.setdefault("cell_entries", {})
        with self.lock:
            got = cache.get(str(key))
        if got is not None:
            return got
        parts, roads_ = [], []
        feats = self.cell_fn(key)
        if feats is not None and len(feats):
            bs = self.specs_of(feats)
            rs = road_specs(feats)
            names = feats.attrs.get("name")
            for i in range(len(feats)):
                fid = str(feats.ids[i])
                if bs[i] is not None:
                    lvls = self.levels_of(feats, i)
                    best = {}
                    for j, ring in enumerate(self.parts_of(feats, i)):
                        if (fid, j) in best and len(ring) <= best[(fid, j)][4]:
                            continue
                        e, n = mapgeom.enu_xy(feats, self.frame, ring)
                        best[(fid, j)] = (bs[i], densify(_dedupe(np.stack([e, n], axis=1)), DECK_STEP),
                                          float(np.mean(ring[:, 0])), lvls[j] if j < len(lvls) else 1.0, len(ring),
                                          str(names[i]) if names is not None else "")     # (roadz: name matching)
                    parts.extend(best.items())
                if rs[i] is not None:
                    lines = []
                    for ring in ground_parts(feats, i):
                        e, n = mapgeom.enu_xy(feats, self.frame, ring)
                        lines.append(np.stack([e, n], axis=1))
                    roads_.append(((fid, "g"), lines))
                elif _rail_drawn(feats, i):
                    # a railway on the ground: the decks over it keep their headroom too (GitHub #324: a road bridge
                    # lay on the tracks; only roads under a deck were counted)
                    import railways
                    lines = []
                    for kind, rings in railways._pieces(feats, i):
                        if kind == "ground":
                            for ring in rings:
                                e, n = mapgeom.enu_xy(feats, self.frame, ring)
                                lines.append(np.stack([e, n], axis=1))
                    if lines:
                        roads_.append(((fid, "r"), lines))
        got = (parts, roads_)
        with self.lock:
            if len(cache) > 64:
                cache.clear()
            cache.setdefault(str(key), got)
            return cache[str(key)]

    END_JOINS = True        # (TunnelProfiles: none)

    def _end_joins_of(self, lines, halves, ground):
        """_end_joins of deck lines with the ground roads of ground ({(fid, "g"): lines}); [] for tunnels."""
        if not self.END_JOINS:
            return []
        return _end_joins(lines, halves, [g for k, v in ground.items() if k[1] == "g" for g in v])

    def _end_ons_of(self, lines, halves, levels, ground):
        """_end_ons of deck lines with the ground roads of ground; [] for tunnels."""
        if not self.END_JOINS:
            return []
        return _end_ons(lines, halves, levels, [g for k, v in ground.items() if k[1] == "g" for g in v])

    def _end_join_pairs(self, lines, halves, ground, levels=None):
        """(i, j): line i's free end joins line j's end (_end_joins) or stands on line j (_end_ons)."""
        joins = self._end_joins_of(lines, halves, ground)
        out = set()
        if levels is not None:
            for k, _at, j, _arc in self._end_ons_of(lines, halves, levels, ground):
                out.add((min(k, j), max(k, j)))
        if not joins:
            return sorted(out)
        moved = {(k, at): p for k, at, p in joins}
        for k, at, p in joins:
            for j, ln in enumerate(lines):
                if j != k and any(np.hypot(*(np.asarray(moved.get((j, a2), np.asarray(ln)[a2, :2]), np.float64) -
                                             np.asarray(p))) < 1e-6 for a2 in (0, -1)):
                    out.add((min(k, j), max(k, j)))
        return sorted(out)

    def _resolve(self, start, pool, ground, loaded):
        """The chain of part `start` (keys), loading the cell at each loose end until closed."""
        while True:
            nk = _EndJoin()
            by_node = {}
            for k, (_, xy, *_) in pool.items():
                by_node.setdefault(nk(xy[0]), []).append(k)
                by_node.setdefault(nk(xy[-1]), []).append(k)
            keys = sorted(pool)
            over = _deck_crossings([pool[k][1] for k in keys])
            crossing = {}
            for a, b, _, _ in over:
                crossing.setdefault(keys[a], set()).add(keys[b])
                crossing.setdefault(keys[b], set()).add(keys[a])
            # and side by side / one over the other (solved apart, the Queensboro Bridge's two upper roadways met at
            # a 0.5 m step and its tiles' decks at different heights stood as blocks in the lanes, 2026-10-17)
            for a, b in _deck_beside([pool[k][1] for k in keys], [pool[k][0].width for k in keys]):
                crossing.setdefault(keys[a], set()).add(keys[b])
                crossing.setdefault(keys[b], set()).add(keys[a])
            # and a free end joining another part's end (_end_joins): solved as one
            for a, b in self._end_join_pairs([pool[k][1] for k in keys], [pool[k][0].width / 2 for k in keys], ground,
                                             [pool[k][3] for k in keys]):
                crossing.setdefault(keys[a], set()).add(keys[b])
                crossing.setdefault(keys[b], set()).add(keys[a])
            comp, stack = {start}, [start]
            while stack:                        # joined at nodes, or crossing (solved together)
                k = stack.pop()
                xy = pool[k][1]
                nxt = [k2 for p in (xy[0], xy[-1]) for k2 in by_node.get(nk(p), ())]
                for k2 in nxt + sorted(crossing.get(k, ())):
                    if k2 not in comp:
                        comp.add(k2)
                        stack.append(k2)
            deg = {}
            for k in comp:
                xy = pool[k][1]
                for p in (xy[0], xy[-1]):
                    deg[nk(p)] = deg.get(nk(p), 0) + 1
            # the loose ends first, in key order: a cell loaded can put a longer copy of a part in the pool (_load), whose
            # ends deg never saw (KeyError: a Chicago road tile on a Windows test PC, 2026-10-24), and the set's order
            # changes with each process's string hashing (which cells CHAIN_CELLS let in)
            ends = [p for k in sorted(comp) for p in (pool[k][1][0], pool[k][1][-1]) if deg[nk(p)] == 1]
            grew = False
            for p in ends:
                la, lo, _ = self.frame.enu_to_geodetic(np.array([p[0]]), np.array([p[1]]), 0.0)
                key = mapcells.cell_of(float(la[0]), float(lo[0]))
                if str(key) not in loaded and len(loaded) < CHAIN_CELLS:
                    loaded.add(str(key))
                    self._load(key, pool, ground)
                    grew = True
            if not grew:
                return sorted(comp)

    def _solve(self, comp, pool, ground):
        """Deck heights of the chain comp (keys of pool)."""
        lines = [pool[k][1] for k in comp]
        # (solved with its free ends that join another part's moved onto those joints: _end_joins; the heights are
        # per point, so the parts keep their own lines)
        lines = _joined_lines(lines, self._end_joins_of(lines, [pool[k][0].width / 2 for k in comp], ground))
        gl = [g for k, v in ground.items() if k[1] == "g" for g in v]
        rl = [g for k, v in ground.items() if k[1] == "r" for g in v]
        cr = _crossings(lines, gl)
        road_z = self._ground(np.array([c[2] for c in cr])) if cr else []
        # (the railways under the chain: the same headroom, given as their own clearance, so never a junction at
        # grade by an abutment as a road can be)
        rcr = _crossings(lines, rl) if rl else []
        rail_z = self._ground(np.array([c[2] for c in rcr])) if rcr else []
        import bridgetypes
        try:
            nav = bridgetypes.nav_clearance(self.frame, lines)
        except Exception as ex:             # the share-of-span rule alone
            log.debug("nav clearance: %s: %s", type(ex).__name__, ex)
            nav = 0.0
        required = [(c[0], c[1], z) for c, z in zip(cr, road_z)]
        depth = self._depth(lines) if bridge_profile_mode() == "curve" else None
        dd = self.__dict__.setdefault("depth", {})
        for k in comp:
            dd[k] = DECK_FULL_DEPTH if depth is None else depth
        try:
            channel = bridgetypes.nav_span(self.frame, lines) if nav > 0 else 0.0
        except Exception:
            channel = 0.0
        water = self._water(lines, nav, depth, channel)
        if not water and nav > 0 and bridge_profile_mode() == "curve":
            water = self._gap(lines, nav, depth)
        rails = [(c[0], c[1], z, deck_road_clear(depth)) for c, z in zip(rcr, rail_z)]
        levels = [pool[k][3] for k in comp]
        pins = _ground_pins(lines, gl)          # (once: pure in lines and gl, which stay)

        def solve(ground_fn):
            return deck_profile(lines, ground_fn, required + rails + water, clear=deck_road_clear(depth),
                                levels=levels, widths=[pool[k][0].width for k in comp], pins=pins)
        zs = solve(self._ground)
        # a free end standing on another deck: solved again with it at that deck's height as drawn (measured too:
        # _observed) (_end_ons)
        over = {}
        ons = self._end_ons_of(lines, [pool[c][0].width / 2 for c in comp], levels, ground)
        final = self._observed(comp, pool, lines, zs, pins=pins) if ons else zs
        for k, at, j, arc in ons:
            lj = np.asarray(lines[j], np.float64)[:, :2]
            sj = np.r_[0.0, np.cumsum(np.hypot(*np.diff(lj, axis=0).T))]
            zt = float(np.interp(arc, sj, np.asarray(final[j], np.float64)))
            if np.isfinite(zt) and abs(zt - float(final[k][at])) > END_ON_STEP_M:
                q = np.asarray(lines[k], np.float64)[at, :2]
                over[(float(q[0]), float(q[1]))] = zt
        if over:
            keys_ = np.array(list(over), np.float64)
            vals = np.array(list(over.values()), np.float64)

            def ground_on(xy):
                z = np.asarray(self._ground(xy), np.float64)
                xy = np.asarray(xy, np.float64).reshape(-1, 2)
                for (px, py), zt in zip(keys_, vals):
                    z = np.where((np.abs(xy[:, 0] - px) < 1e-6) & (np.abs(xy[:, 1] - py) < 1e-6), zt, z)
                return z
            zs = solve(ground_on)
        deficits = _clearance_deficits(lines, zs, required + [r[:3] for r in rails], depth)
        if deficits:
            # (debug: a city has hundreds of such bridges, each solved again in every process and tile; as a warning it
            # filled players' consoles, GitHub #391, TGForever Discord 2026-10-28)
            log.debug("bridge clearance below %.1f m target: %s; abutment/grade or map constraints prevent full headroom",
                      MIN_BRIDGE_HEADROOM_M, [(comp[i], round(h, 2)) for i, h in deficits[:3]])
        out = final if (ons and not over) else self._observed(comp, pool, lines, zs, pins=pins)
        hook = roadbase.DECK_Z_HOOK         # (set by grading at import: read here, not copied)
        if hook is not None:
            # (a carriageway's deck: the height its roads and decks were solved for together, grading's
            # carriageways.py, so the road meets it at its ends whatever the ground there)
            try:
                got = hook(self.frame, lines, [k[0] for k in comp])
            except Exception as ex:
                log.debug("deck heights from the carriageways: %s: %s", type(ex).__name__, ex)
                got = None
            if got is not None:
                out = [g if g is not None else z for g, z in zip(got, out)]
        return out

    def _observed(self, comp, pool, lines, zs, level=1, pins=None):
        """The chain's deck heights where a source measured them (roadz.py: IGN BD TOPO, NVDB, US lidar), the solved
        ones (the inference) elsewhere. Each joint ends up at one height: an abutment (a joint no other part of the
        chain shares) at the solved one (the graded road it meets), a joint between parts at the mean of their measured
        ends; each part's profile is eased onto its joints' heights (earthfast.deck_blend: the grade changes by
        OBS_BLEND_DG at most). A part whose measured profile still steps or climbs past OBS_MAX_GRADE keeps the
        solve. Pure given the provider's cached data."""
        import roadz
        prov = roadz.PROVIDER
        if prov is None or not prov.enabled or not lines:
            return zs
        obs = []
        for k, ln in zip(comp, lines):
            xy = np.asarray(ln, np.float64)[:, :2]
            got = None
            try:
                spec = pool[k][0]
                nm = pool[k][5] if len(pool[k]) > 5 else ""
                z = prov.observed(self.frame, xy, spec.width / 2, level, "" if nm in ("None", "nan") else nm)
                if z is not None:
                    la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
                    ok = np.isfinite(z)
                    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
                    zf = np.interp(s, s[ok], np.asarray(z)[ok])         # (the ends past the last measurement: flat)
                    got = np.asarray(self.frame.geodetic_to_enu(la, lo, zf)[2], np.float64)
            except Exception as ex:             # the inference rather than no bridge
                log.debug("observed deck %s: %s: %s", k, type(ex).__name__, ex)
            obs.append(got)
        if all(o is None for o in obs):
            return zs
        nk = _EndJoin()
        ends = [(nk(np.asarray(ln)[0]), nk(np.asarray(ln)[-1])) for ln in lines]
        deg = {}
        for a, b in ends:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        # a joint a ground road goes on from (pins: the solve's abutment there) keeps the solve's height, the graded
        # road's, as an abutment does: eased to the decks' measured mean, the street went on 2.5 m over them (Charles
        # Street over I-83 in Baltimore, 2026-10-29)
        if pins is not None and len(pins):
            P = np.asarray(pins, np.float64).reshape(-1, 2)
            for i_, (a, b) in enumerate(ends):
                for nd, at in ((a, 0), (b, -1)):
                    q = np.asarray(lines[i_], np.float64)[at, :2]
                    if deg[nd] > 1 and (np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) < PIN_M).any():
                        deg[nd] = 1
        # a measured deck whose end is far off the graded road it meets there is a map gap (a flyover's untagged
        # stretch drawn on the ground: Springfield's 15 m): the inference keeps that part rather than a long ramp
        for i, (o, z, (a, b)) in enumerate(zip(obs, zs, ends)):
            z = np.asarray(z, np.float64)
            if o is not None and len(z) and OBS_END_MAX_M is not None and (
                    (deg[a] == 1 and abs(o[0] - z[0]) > OBS_END_MAX_M)
                    or (deg[b] == 1 and abs(o[-1] - z[-1]) > OBS_END_MAX_M)):
                log.debug("observed deck %s: its end %.1f m off the road it meets, the solve kept", comp[i],
                          max(abs(o[0] - z[0]), abs(o[-1] - z[-1])))
                obs[i] = None
        if all(o is None for o in obs):
            return zs
        measured = self.__dict__.setdefault("_obs_tl", threading.local()).__dict__.setdefault("measured", set())
        target = {}
        for node in deg:
            if deg[node] == 1:
                continue
            got = [(o[0] if a == node else o[-1]) for o, (a, b) in zip(obs, ends) if o is not None and node in (a, b)]
            if got:
                target[node] = float(np.mean(got))
        # a part takes the whole step between a joint and its far end: a short one between a measured deck and its road
        # climbed 36 % (15 m of a French river bridge, 2.3 m): the joint comes down to what that part can
        # climb, and the measured deck eases to it
        for z, ln, (a, b) in zip(zs, lines, ends):
            if not len(z):
                continue
            pts = np.asarray(ln, np.float64)[:, :2]
            reach = OBS_JOINT_GRADE * float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
            for node, far, zfar in ((a, b, float(z[-1])), (b, a, float(z[0]))):
                if node in target:
                    h = zfar if deg[far] == 1 else target.get(far, zfar)
                    target[node] = min(max(target[node], h - reach), h + reach)
        import fast
        out = []
        for k, (ln, z, o, (a, b)) in enumerate(zip(lines, zs, obs, ends)):
            z = np.asarray(z, np.float64)
            if not np.isfinite(z).all():
                out.append(z)
                continue
            base = o if o is not None else z
            t0 = z[0] if deg[a] == 1 else target.get(a, z[0])
            t1 = z[-1] if deg[b] == 1 else target.get(b, z[-1])
            if o is None and t0 == z[0] and t1 == z[-1]:
                out.append(z)
                continue
            xy = np.asarray(ln, np.float64)[:, :2]
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            zz = np.asarray(fast._ef.deck_blend(s, np.ascontiguousarray(base), float(t0), float(t1), OBS_BLEND_DG,
                                                OBS_BLEND_MIN_M, 0.5), np.float64)
            if o is not None and len(s) > 3:
                step = np.interp(np.arange(0.0, s[-1], 2.0), s, zz)
                g = np.abs(np.diff(step)) / 2.0 if len(step) > 1 else np.zeros(1)
                if len(g) and float(g.max()) > OBS_MAX_GRADE:
                    log.debug("observed deck %s: %.0f %% after the blend, the solve kept", comp[k], 100 * g.max())
                    zz = np.asarray(fast._ef.deck_blend(s, np.ascontiguousarray(z), float(t0), float(t1), OBS_BLEND_DG,
                                                        OBS_BLEND_MIN_M, 0.5), np.float64)
                else:
                    # never under both the solve (its clearances) and the measurement: an easing toward a joint
                    # pulled a deck through the road under it (-1.8 m over I-95 at Springfield)
                    # (the ends as eased: the parts at a joint keep meeting at one height)
                    zz = np.r_[zz[0], np.maximum(zz, np.minimum(z, o))[1:-1], zz[-1]]
                    measured.add(comp[k])
            if 0 < s[-1] < OBS_JOINT_SHORT_M and float(np.max(np.abs(np.diff(zz)) / np.maximum(np.diff(s), 0.5))) > OBS_JOINT_GRADE * 1.25:
                zz = float(t0) + (float(t1) - float(t0)) * s / s[-1]    # (the easing's minimum length bunched a short part's climb)
            out.append(zz)
        return out

    def _depth(self, lines):
        """m of slab and girders of a chain: deck_depth of the inventory's main span where a record matches, else of
        geom_span. Never raises."""
        span = 0.0
        try:
            import bridgetypes
            got = bridgetypes.matches(self.frame, lines)
            if got:
                span = bridgetypes.record(got[0][1])["span"]
        except Exception as ex:
            log.debug("deck span: %s: %s", type(ex).__name__, ex)
        return deck_depth(span if span > 0 else geom_span(lines))

    def depth_of(self, fid):
        """m of slab and girders under part fid's road (DECK_FULL_DEPTH when not solved here)."""
        ds = [d for k, d in list(self.__dict__.get("depth", {}).items()) if self._part_id(k) == fid]
        return max(ds) if ds else DECK_FULL_DEPTH

    def _gap(self, lines, nav, depth=None):
        """Clearance requirements where the sea's depth knows no water but the inventory gives the bridge a
        navigation clearance (a river: the Calcasieu, the Mississippi at New Orleans, 2026-10-19 stayed 10-14 m up
        under a 41-46 m clearance): every WATER_STEP_M along the chain's low stretch (ground within GAP_LOW_M of its
        lowest point, which lies at sea level, GAP_SEA_M at most, and GAP_DIP_M or more under the chain's highest
        end), nav plus the slab and girders over it. Only at sea level: up a river the inventory's clearances are
        often the span raised or measured from another datum (Portland's lift and bascule bridges came out 7-29 m
        high with it). Pure given the ground."""
        under = DECK_FULL_DEPTH if depth is None else float(depth)
        gs = [np.asarray(self._ground(np.asarray(ln, np.float64)[:, :2]), np.float64) for ln in lines]
        if not gs or not all(len(g) for g in gs):
            return []
        lo = float(min(g.min() for g in gs))
        top = float(max(max(g[0], g[-1]) for g in gs))
        if lo > GAP_SEA_M or lo > top - GAP_DIP_M:
            return []
        out = []
        for k, (ln, g) in enumerate(zip(lines, gs)):
            xy = np.asarray(ln, np.float64)[:, :2]
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            low = g <= lo + GAP_LOW_M
            for at in np.arange(0.0, s[-1] + 1e-9, WATER_STEP_M):
                if low[min(int(np.searchsorted(s, at)), len(s) - 1)]:
                    out.append((k, float(at), lo, nav + under))
        return out

    def parts(self, cells_feats, box):
        """[(fid, spec, xy, z, lon)] of the bridge parts in these cells that come near box."""
        x0, y0, x1, y1 = box
        todo, queued = [], set()
        for feats in cells_feats:
            if feats is None or len(feats) == 0:
                continue
            bs = self.specs_of(feats)
            for i in range(len(feats)):
                if bs[i] is None:
                    continue
                fid = str(feats.ids[i])
                for j, ring in enumerate(self.parts_of(feats, i)):
                    if (fid, j) in queued:          # a feature in several of the cells: once (2026-10-17: every
                        continue                    # part came once per cell it is in, decks made 2-3 times over)
                    e, n = mapgeom.enu_xy(feats, self.frame, ring)
                    if e.max() < x0 - NEAR_M or e.min() > x1 + NEAR_M or n.max() < y0 - NEAR_M or n.min() > y1 + NEAR_M:
                        continue
                    queued.add((fid, j))
                    todo.append(((fid, j), feats, i))
        return BridgeProfiles.parts_todo(self, todo)      # (the whole parts: TunnelProfiles splits them)

    owner_of = None     # owner_of(self, comp, pool) -> the profiles that solve chain comp (ChainStore: the one of its
                        # first part's region), or None: this one

    def parts_todo(self, todo, solved_only=False, part_ids=False):
        """[(fid, spec, xy, z, lon)] of the parts todo [((fid, j), feats, i)], each solved with its chain (once);
        solved_only: only those solved already; part_ids: return (fid, j) rather than fid."""
        out = []
        for key, feats, i in todo:
            with self.lock:
                got = self.done.get(key)
            if got is None and not solved_only:
                ring = self.parts_of(feats, i)[key[1]]
                got = self._chain_of(key, [(float(p[1]), float(p[0])) for p in (ring[0], ring[-1])])
            if got is not None:
                spec, xy, z, lon = got
                out.append((key if part_ids else self._part_id(key), spec, xy, z, lon))
        return out

    def _chain_of(self, key, ends, adopt=True):
        """Solve the chain of part key (its ring's ends (lat, lon) say where to start loading): its parts into done.
        Returns done[key] or None. A chain another store owns (owner_of) is solved there and its result taken in, its
        geometry mapped into this frame (one deck profile per chain, whichever store a tile asks)."""
        with self.lock:
            got = self.done.get(key)
        if got is not None:
            return got
        pool, ground, loaded = {}, {}, set()
        for la_, lo_ in ends:
            ck = mapcells.cell_of(la_, lo_)
            if str(ck) not in loaded:
                loaded.add(str(ck))
                self._load(ck, pool, ground)
        if key not in pool:
            return None
        comp = self._resolve(key, pool, ground, loaded)
        owner = self.owner_of(self, comp, pool) if (adopt and self.owner_of is not None) else None
        if owner is not None and owner is not self:
            try:
                if self._adopt(owner, comp, pool):
                    with self.lock:
                        got = self.done.get(key)
                    if got is not None:
                        return got
            except Exception as ex:         # (solved here rather than not at all)
                log.debug("chain %s from its owner: %s: %s", key, type(ex).__name__, ex)
        tl = self.__dict__.setdefault("_obs_tl", threading.local())
        tl.measured = set()
        zs = self._solve(comp, pool, ground)
        zs = _tie_beside([pool[k][1] for k in comp], zs, [pool[k][0] for k in comp],     # (ramps beside their road)
                         [k in tl.measured for k in comp])
        with self.lock:
            for k, z in zip(comp, zs):
                spec, xy, lon = pool[k][:3]
                old = self.done.get(k)
                # the first solve of a part is kept, and what goes with it, unless this one saw more of it
                if old is None or len(xy) > len(old[1]):
                    self.done[k] = (spec, xy, z, lon)
                    self._commit(k, ground)
            chain = [(self.done[k][1], self.done[k][2], self.done[k][0].width) for k in comp
                     if k in self.done]
            ck = min(comp) if comp else None
            fresh = ck is not None and ck not in self.trusses
            if fresh:
                self.trusses[ck] = []
            got = self.done.get(key)
        if fresh:
            beams, tpiers = self._structure(chain)
            with self.lock:
                self.trusses[ck] = beams
                self.piers[("truss", ck)] = tpiers
        return got

    def _ends_ll(self, xy):
        """(lat, lon) of a deck line's two ends (this frame's world)."""
        la, lo, _ = self.frame.enu_to_geodetic(np.array([xy[0][0], xy[-1][0]]), np.array([xy[0][1], xy[-1][1]]), 0.0)
        return [(float(la[0]), float(lo[0])), (float(la[1]), float(lo[1]))]

    def _adopt(self, owner, comp, pool):
        """Chain comp solved by owner (from its first part), its parts, piers, trusses and depths taken into this
        frame. False when owner's solve did not cover comp's parts."""
        first = min(comp)
        if owner._chain_of(first, self._ends_ll(pool[first][1]), adopt=False) is None:
            return False
        with owner.lock:
            mine = [k for k in comp if k in owner.done]
            if len(mine) != len(comp):
                return False
            done = {k: owner.done[k] for k in mine}
            piers = {k: owner.piers.get(k, []) for k in mine}
            ck = min(comp)
            beams = owner.trusses.get(ck)
            tpiers = owner.piers.get(("truss", ck))
            depth = {k: owner.__dict__.get("depth", {}).get(k) for k in mine}
            extra = owner._adopt_extra(comp)
        mp = ChainView.mapper(owner.frame, self.frame)
        with self.lock:
            for k, (spec, xy, z, lon) in done.items():
                if k in self.done:
                    continue
                xyz = mp.pts(np.c_[np.asarray(xy, np.float64)[:, :2], np.asarray(z, np.float64)])
                self.done[k] = (spec, xyz[:, :2].copy(), xyz[:, 2].copy(), lon)
                self.piers[k] = [mp.pier(q) for q in piers[k]]
                if depth.get(k) is not None:
                    self.__dict__.setdefault("depth", {})[k] = depth[k]
            if beams is not None and ck not in self.trusses:
                self.trusses[ck] = [mp.beam(b) for b in beams]
                self.piers[("truss", ck)] = [mp.pier(q) for q in (tpiers or [])]
            self._adopt_take(extra)
        return True

    def _adopt_extra(self, comp):
        """Under self.lock: what else goes with comp's solve (TunnelProfiles: its sections)."""
        return None

    def _adopt_take(self, extra):
        """Under self.lock: _adopt_extra's from another store."""

    def _commit(self, k, ground=None):
        """Under self.lock: part k's solve was kept: its piers (pier_sites; TunnelProfiles keeps its sections)."""
        spec, xy, z, _lon = self.done[k]
        lines = [ln for v in (ground or {}).values() for ln in v]
        try:
            self.piers[k] = pier_sites(np.asarray(xy, np.float64), np.asarray(z, np.float64), spec.width,
                                       self._ground, lines, self._wet_at, self.__dict__.get("depth", {}).get(k))
        except Exception as ex:             # no piers rather than no bridge
            log.debug("piers of %s: %s: %s", k, type(ex).__name__, ex)
            self.piers[k] = []

    def _wet_at(self, xy):
        if self.wet is None:
            return np.zeros(len(xy), bool)
        la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        return np.asarray(self.wet(la, lo), np.float64) > 0.5

    def _roads_near(self, xy):
        """The ground road lines (k, 2) of the map cells around the points xy (n, 2) (an anchor pier's site)."""
        la, lo, _ = self.frame.enu_to_geodetic(np.asarray(xy)[:, 0], np.asarray(xy)[:, 1], 0.0)
        out = []
        for key in {mapcells.cell_of(float(a), float(b)) for a, b in zip(la, lo)}:
            feats = self.cell_fn(key)
            if feats is None or len(feats) == 0:
                continue
            rs = road_specs(feats)
            for i in range(len(feats)):
                if rs[i] is not None:
                    for ring in ground_parts(feats, i):
                        e, n = mapgeom.enu_xy(feats, self.frame, ring)
                        out.append(np.stack([e, n], axis=1))
        return out

    def _structure(self, chain):
        """The truss beams of a solved chain [(xy, z, width)] and the piers under its ends (bridgetypes); ([], []) for a
        plain deck."""
        import bridgetypes
        try:
            infra = []
            if self.infra_fn is not None:
                pts = np.concatenate([np.asarray(xy, np.float64) for xy, _, _ in chain])
                la, lo, _ = self.frame.enu_to_geodetic(pts[:, 0], pts[:, 1], 0.0)
                infra = [self.infra_fn(k) for k in mapcells.cells_for_points(la, lo)]
            wet = self._wet_at if self.wet is not None else None
            beams, piers = [], []
            for info in bridgetypes.chain_structures(self.frame, chain, wet, infra):
                b, p = bridgetypes.structure_members(chain, info, self._ground, wet, road_lines=self._roads_near)
                beams += b
                piers += p
            return beams, piers
        except Exception as ex:             # a plain deck rather than no bridge
            log.debug("bridge structure: %s: %s", type(ex).__name__, ex)
            return [], []

    def beams_in(self, box):
        """The truss beams (p, q, w, h) of the chains solved so far whose middle lies in box (each in one tile)."""
        x0, y0, x1, y1 = box
        with self.lock:
            allb = [b for v in self.trusses.values() for b in v]
        return [b for b in allb if box_owns(box, (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2)]

    def piers_in(self, box):
        """The piers (pier_sites) of the parts solved so far standing in box (x0, y0, x1, y1): x0 <= x < x1, y0 <= y
        < y1 (each in one tile)."""
        x0, y0, x1, y1 = box
        with self.lock:
            allp = [p for v in self.piers.values() for p in v]
        return [p for p in allp if box_owns(box, p[0], p[1])]

    @staticmethod
    def _part_id(key):
        return key[0]


def tunnel_sections(s, g, open_cover=None, full_cover=None, min_m=SECTION_MIN_M):
    """A tunnel's road level and its sections from the ground over it (g at arc lengths s): (floor, kinds). floor: the
    lower convex hull of (s, g), the road's level under the covered parts and the ground where it lies open; kinds per
    sample: 0 open road, 1 a low arch, 2 a full tunnel (a run with TUNNEL_OPEN_COVER over the floor, full where it
    reaches TUNNEL_COVER; runs and gaps shorter than min_m merged away). One long 'tunnel' in the map can be a sunken
    road with arches and a rock tunnel along it (Central Park's 79th Street Transverse, 683 m: one tube, 2026-10-17).
    The Rust kernel (fast). Pure."""
    s = np.ascontiguousarray(s, np.float64)
    g = np.ascontiguousarray(g, np.float64)
    oc = TUNNEL_OPEN_COVER if open_cover is None else open_cover
    fc = TUNNEL_COVER if full_cover is None else full_cover
    step = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    min_n = max(1, int(math.ceil(min_m / max(step, 1e-6))))
    import fast
    fast.require(28, "tunnel_sections")
    f, k = fast._ef.tunnel_sections(s, g, float(oc), float(fc), int(min_n))
    return _shallow_full_low(s, g, np.asarray(f), np.asarray(k))


def _shallow_full_low(s, g, floor, kinds):
    """kinds with each full-tunnel run (2) shorter than FULL_MIN_M and nowhere FULL_DEEP_M under the ground a low arch
    (1). Pure."""
    kinds = np.array(kinds, np.int64)
    i, n = 0, len(kinds)
    while i < n:
        if kinds[i] != 2:
            i += 1
            continue
        j = i
        while j + 1 < n and kinds[j + 1] == 2:
            j += 1
        if s[j] - s[i] < FULL_MIN_M and float(np.max(g[i:j + 1] - floor[i:j + 1])) < FULL_DEEP_M:
            kinds[i:j + 1] = 1
        i = j + 1
    return floor, kinds


class TunnelProfiles(BridgeProfiles):
    """Floor heights per tunnel part for one anchor frame, each chain (joined parts: a tunnel split over several
    segments) resolved once, as BridgeProfiles; each part cut into its sections (tunnel_sections): open roads
    (open_parts), low arches (heights) and full tunnels, each a part of its own (fid#k) with its own portals. Worker
    threads."""
    specs_of = staticmethod(lambda feats: tunnel_specs(feats))
    parts_of = staticmethod(lambda feats, i: tunnel_parts(feats, i))
    levels_of = staticmethod(lambda feats, i: [-1.0] * len(tunnel_parts(feats, i)))
    END_JOINS = False

    def _structure(self, chain):                # (no truss over a tunnel)
        return [], []

    wet = None      # wet(lat, lon) -> water depth (m): identify chains needing a constrained wet profile and finite
                    # client water columns. Unsupported wet chains stay excluded; earthd/blockworker set the sources.
    open = None
    water = None    # water(lat, lon) -> (depth, surface level): required for supported underwater profiles
    raw = None      # raw(lat, lon) -> the ground without the roads' grading (the cover over a tunnel is read on it)
    ends = None     # ends(lat, lon) -> the graded ground (its portals' floor), else the profiles' elev. Both at one DEM
                    # zoom (grading.PROFILE_ZOOM) in every process: the terrain blocks and the road tiles are built in
                    # different processes at different zooms, and cut 97th Street's arch 15 m apart: its west mouth
                    # filled with ground behind the portal face (2026-10-17)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.open = set()           # sub-part fids that are open roads
        self.heights = {}           # sub-part fid -> a low arch's tube height
        self.ups = {}               # sub-part fid -> its portal faces' reach over the vault at its start and end
        self.sections = {}          # part fid -> [(i0, i1, kind)] (inclusive sample ranges, sharing their ends)
        self.caps = {}              # sub-part fid -> a low arch's stone top's depth at its start and end (None: the
                                    # full roadmesh.LOW_CAP_M), stopping short of a road over its mouth (_road_caps)
        self._hw = {}               # (fid, "g") -> half the carriageway of a ground road in `ground` (_load)
        self.mouths = {}            # sub-part fid -> (its start is a portal, its end is one): not where a low arch
                                    # meets a full tunnel inside the hill (no hole, no portal face there)
        self._tls = threading.local()   # a thread's solved sections, kept with its parts (_commit) or dropped
        self._drowned = {}          # part id (fid.j) -> its chain has a part under water (drowned)
        self._subsea = set()        # part ids whose underwater profile was solved successfully

    def _raw_ground(self, xy):
        la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        return self.frame.geodetic_to_enu(la, lo, np.asarray(self.raw(la, lo), np.float64))[2]

    @staticmethod
    def _part_id(key):
        # one feature can come as several parts (clipped differently in the map cells around: 97th Street's arch as
        # 45 m and 30 m pieces, the shorter one's sections cut the longer one's terrain 15 m short, 2026-10-17)
        return f"{key[0]}.{key[1]}"

    def _commit(self, k, ground=None):
        staged = getattr(self._tls, "staged", {}).pop(k, None)
        if staged is None:
            return
        pid = self._part_id(k)
        secs, subs, underwater = staged
        if underwater:
            self._subsea.add(pid)
        self.sections[pid] = secs
        for n, (kind, th, ups, mouths, caps) in subs.items():
            sub = f"{pid}#{n}"
            if kind == 0:
                self.open.add(sub)
                continue
            if kind == 1:
                self.heights[sub] = th
            self.ups[sub] = ups
            self.mouths[sub] = mouths
            if caps is not None:
                self.caps[sub] = caps

    def _mark_drowned(self, comp, pool):
        """Mark every part of the chain comp drowned when one of its parts is under water for more than
        TUNNEL_WET_SHARE of its points: unsupported underwater tunnels must be left out whole. Overture splits one into pieces (the
        Burnley Tunnel, Melbourne: the piece under the Yarra was left out, the dry piece from Batman Avenue to the
        bank was built, an open trench across the avenue with its mouth at the river; GitHub #128)."""
        if self.wet is None:
            return
        wet = False
        gfn = self._raw_ground if self.raw is not None else self._ground
        low = None                      # (the lowest ground at the parts' ends: the lower portal's)
        for k in comp:
            xy = np.asarray(pool[k][1], np.float64)
            if len(xy) >= 2:
                ge = np.asarray(gfn(xy[[0, -1]]), np.float64)
                if np.isfinite(ge).any():
                    low = float(np.nanmin(ge)) if low is None else min(low, float(np.nanmin(ge)))
        for k in comp:
            xy = np.asarray(pool[k][1], np.float64)
            if len(xy) < 2:
                continue
            la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
            w = np.asarray(self.wet(la, lo), np.float64) > 0.5
            if w.any() and low is not None:
                # (water far above the portals is a mountain lake over a deep tunnel, not a river over a shallow one:
                # one pond 900 m over the Gotthard road tunnel left all 17 km of it out, 2026-10-28)
                g = np.asarray(gfn(xy), np.float64)
                w &= ~(g - low > TUNNEL_WET_COVER_M)
            if w.mean() > TUNNEL_WET_SHARE or wet_run_inside(xy, w) >= TUNNEL_WET_RUN_M:
                wet = True
                break
        with self.lock:
            for k in comp:
                self._drowned[self._part_id(k)] = wet

    def drowned(self, key, ring):
        """Is tunnel part key (fid, j) (ring: its lon/lat points) in a chain with a part under water
        (_mark_drowned)? The grader asks the same (grading.Grader._tunnel_sections), so the ground under a left-out
        tunnel is not cut for it either."""
        pid = self._part_id(key)
        with self.lock:
            got = self._drowned.get(pid)
        if got is not None or self.wet is None:
            return bool(got)
        pool, ground, loaded = {}, {}, set()
        for p in (ring[0], ring[-1]):
            ck = mapcells.cell_of(float(p[1]), float(p[0]))
            if str(ck) not in loaded:
                loaded.add(str(ck))
                self._load(ck, pool, ground)
        if key not in pool:
            return False
        self._mark_drowned(self._resolve(key, pool, ground, loaded), pool)
        with self.lock:
            return bool(self._drowned.get(pid))

    def _ends_ground(self, xy):
        if self.ends is None:
            return self._ground(xy)
        la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        return self.frame.geodetic_to_enu(la, lo, np.asarray(self.ends(la, lo), np.float64))[2]

    def _solve(self, comp, pool, ground):
        """Per part: the floor (the hull of the ungraded ground, its ends at the roads' graded height at its portals,
        where the approaches arrive) and its sections; a low arch's tube under the thin part of the cover over its
        middle (a 20th percentile of the middle 60 %, less TUNNEL_SOIL_M; sized from the most, the ground came down
        into it and was cut away, a bare stone vault), a full tunnel's roadmesh.TUNNEL_H."""
        import roadmesh
        self._mark_drowned(comp, pool)
        gfn = self._raw_ground if self.raw is not None else self._ground
        out = []
        self._tls.staged = {}           # (committed by parts() right after, the rest dropped with the next solve)
        underwater_floors = {}
        if any(self._drowned.get(self._part_id(k)) for k in comp):
            import subsea
            answer = subsea.profiles(self, comp, pool)
            if answer is not None:
                underwater_floors, membership = answer
                with self.lock:
                    for k, wet in membership.items():
                        self._drowned[self._part_id(k)] = wet
            elif self.water is not None:
                import revisit
                revisit.fault()     # incomplete/unsolved wet geometry must not enter persistent road/plan caches
        nk = _EndJoin()                 # (the parts' ends another part of the chain goes on from: no portal there)
        join = lambda k, p: subsea.node(p) if k in underwater_floors else nk(p)
        jends = {k: (join(k, np.asarray(pool[k][1])[0]), join(k, np.asarray(pool[k][1])[-1])) for k in comp
                 if len(pool[k][1]) >= 2}
        jdeg = {}
        for a, b in jends.values():
            jdeg[a] = jdeg.get(a, 0) + 1
            jdeg[b] = jdeg.get(b, 0) + 1
        for k in comp:
            xy = np.asarray(pool[k][1], np.float64)
            if len(xy) < 2:
                out.append(np.full(len(xy), np.nan))
                continue
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            g = np.asarray(gfn(xy), np.float64)
            underwater = k in underwater_floors
            ge = g[[0, -1]] if underwater else np.asarray(self._ends_ground(xy[[0, -1]]), np.float64)
            g = g.copy()
            g[0], g[-1] = ge[0], ge[-1]
            if underwater:
                floor = underwater_floors[k]
                la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.)
                d, level = self.water(la, lo)[:2]
                level = np.asarray(level, np.float64)
                lz = self.frame.geodetic_to_enu(la, lo, np.where(np.isfinite(level), level, 0.))[2]
                cover_ground = np.where(np.asarray(d) > .5, np.maximum(g, lz), g)
                kinds = np.where(cover_ground - floor >= TUNNEL_COVER - .02, 2, 0)
            else:
                floor, kinds = tunnel_sections(s, g)
                kinds = portal_cover(kinds, g - floor, pool[k][0])
                kinds = self._cover_roads(xy, s, kinds, ground)
            secs, i = [], 0
            while i < len(kinds):
                j = i
                while j + 1 < len(kinds) and kinds[j + 1] == kinds[i]:
                    j += 1
                secs.append((i, j, int(kinds[i])))
                i = j + 1
            # neighbouring sections share their end sample (no gap between a portal and the open road)
            secs = [(a if n == 0 else a - 1, b, kd) for n, (a, b, kd) in enumerate(secs)]
            if not underwater:
                floor = self._observed_floor(k, pool, xy, s, floor, secs,
                                             tuple(jdeg.get(e, 0) > 1 for e in jends.get(k, (None, None))))
            cover = g - floor
            log.debug("tunnel %s: sections %s over %.0f m", k, secs, s[-1])
            subs = {}
            for n, (a, b, kd) in enumerate(secs):
                if kd == 0:
                    subs[n] = (0, None, None, None, None)
                    continue
                th = roadmesh.TUNNEL_H
                if kd == 1:
                    c_ = cover[a:b + 1]
                    k0, k1 = int(len(c_) * 0.2), max(int(len(c_) * 0.8), int(len(c_) * 0.2) + 1)
                    base = float(np.percentile(c_[k0:k1], 20))
                    # nor over the thinnest cover along it (ARCH_SIZE_SKIP_M at its mouths aside, where the ground
                    # rises over it): the vault follows a sloping floor, and sized from the middle it rose out of the
                    # ground near a mouth, its stone top 2 m over the E6 beside it (an underpass in Norway, GitHub #2)
                    s_ = s[a:b + 1]
                    inner = (s_ >= s_[0] + ARCH_SIZE_SKIP_M) & (s_ <= s_[-1] - ARCH_SIZE_SKIP_M)
                    if inner.any():
                        base = min(base, float(c_[inner].min()))
                    low_h = MAJOR_LOW_H if pool[k][0] is not None and pool[k][0].cls in CONCRETE_WALL_CLASSES \
                        else TUNNEL_LOW_H         # (a motorway's box clears a truck, its top in a mound if need be)
                    th = float(np.clip(base - TUNNEL_SOIL_M, low_h, roadmesh.TUNNEL_H))
                ups = []
                for near in (s[a:b + 1] <= s[a] + PORTAL_LOOK_M, s[a:b + 1] >= s[b] - PORTAL_LOOK_M):
                    cv = float(np.max(cover[a:b + 1][near])) if near.any() else th
                    ups.append(float(np.clip(cv + PORTAL_GROUND_M - th, PORTAL_UP_MIN, roadmesh.PORTAL_FRAME)))
                mouths = (n == 0 or secs[n - 1][2] == 0, n == len(secs) - 1 or secs[n + 1][2] == 0)
                if underwater:
                    joined = tuple(jdeg.get(e, 0) > 1 for e in jends.get(k, (None, None)))
                    mouths = (mouths[0] and not (a == 0 and joined[0]),
                              mouths[1] and not (b == len(xy) - 1 and joined[1]))
                caps = None
                if kd == 1:
                    caps, ups = self._road_caps(xy[a:b + 1], floor[a:b + 1], th, ups, mouths, pool[k][0].width, ground)
                subs[n] = (kd, th, ups, mouths, caps)
            self._tls.staged[k] = (secs, subs, underwater)
            out.append(floor)
        return out

    def _observed_floor(self, k, pool, xy, s, floor, secs, joints=(False, False)):
        """floor with each covered section (kind 1 / 2) at the measured height where a source has one (roadz.py,
        level < 0: BD TOPO, NVDB) and it is deeper: never over the ground's lower hull (the cover stays); at a portal
        (next to an open section, or the part's end where no other part of the chain goes on: joints (start, end))
        at the hull's (the portals, the open sections and the grader's mouth plans keep theirs), eased in as the decks
        are (earthfast.deck_blend); at a joint inside the hill as measured (the next part reads the same source).
        Pure given the provider's cached data."""
        import roadz
        prov = roadz.PROVIDER
        if prov is None or not prov.enabled or not any(kd for _a, _b, kd in secs):
            return floor
        try:
            spec = pool[k][0]
            nm = pool[k][5] if len(pool[k]) > 5 else ""
            z = prov.observed(self.frame, xy, spec.width / 2, -1, "" if nm in ("None", "nan") else nm)
            if z is None:
                return floor
            import fast
            la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
            z = np.asarray(z, np.float64)
            zf = np.where(np.isfinite(z), z, 0.0)
            zg = np.where(np.isfinite(z), np.asarray(self.frame.geodetic_to_enu(la, lo, zf)[2], np.float64), np.nan)
            out = np.array(floor, np.float64)
            last = len(s) - 1
            for n, (a, b, kd) in enumerate(secs):
                seg = zg[a:b + 1]
                if not kd or b - a < 2 or np.isfinite(seg).mean() < roadz.MIN_COVER:
                    continue
                ok = np.isfinite(seg)
                ss = s[a:b + 1]
                filled = np.interp(ss, ss[ok], seg[ok])
                inside0 = a == 0 and joints[0]
                inside1 = b == last and joints[1]
                if n > 0 and secs[n - 1][2]:            # (another covered section before it: no portal between)
                    inside0 = True
                if n + 1 < len(secs) and secs[n + 1][2]:
                    inside1 = True
                eased = np.asarray(fast._ef.deck_blend(ss - ss[0], np.ascontiguousarray(filled),
                                                       float("nan") if inside0 else float(out[a]),
                                                       float("nan") if inside1 else float(out[b]),
                                                       OBS_BLEND_DG, OBS_BLEND_MIN_M, 0.5), np.float64)
                out[a:b + 1] = np.minimum(eased, out[a:b + 1])
            return out
        except Exception as ex:             # the hull rather than no tunnel
            log.debug("observed tunnel floor %s: %s: %s", k, type(ex).__name__, ex)
            return floor

    def _load(self, key, pool, ground):
        super()._load(key, pool, ground)
        done = self.__dict__.setdefault("hw_done", set())
        if str(key) in done:            # (the same half widths each time: once per cell)
            return
        done.add(str(key))
        feats = self.cell_fn(key)
        if feats is None or len(feats) == 0:
            return
        rs = road_specs(feats)
        for i in range(len(feats)):
            if rs[i] is not None:
                self._hw[(str(feats.ids[i]), "g")] = rs[i].width / 2

    def _cover_roads(self, xy, s, kinds, ground):
        """kinds with each covered section carried on until the ground roads crossing over it are covered (their
        carriageway and TUNNEL_ROAD_CLEAR_M past it): earthfast.tunnel_cover_roads (GitHub #450)."""
        lines, halves = [], []
        for gk, lns in (ground or {}).items():
            hw = self._hw.get(gk)
            if hw is None:
                continue
            for ln in lns:
                ln = np.asarray(ln, np.float64).reshape(-1, 2)
                if len(ln) >= 2:
                    lines.append(ln)
                    halves.append(hw)
        return cover_roads(xy, s, kinds, lines, halves)

    def _road_caps(self, xy, floor, th, ups, mouths, width, ground):
        """A low arch's stone tops (roadmesh.LOW_CAP_M back from each portal face over its mouth) and face reaches where
        a road on the ground runs over a mouth (not one joined there: ends of it within ROAD_CAP_JOIN_M): the top stops
        ROAD_CAP_GAP_M short of the carriageway, the face ROAD_CAP_OVER_M over the road at most (an underpass under
        the E6 in Norway: its stone top 1.7 m over the lanes, 4 m into them, GitHub #2). ((start, end) depths or None
        where nothing changes, ups)."""
        import fast
        import roadmesh
        fast.require(45, "first_probe_near_lines")
        side = width / 2 + roadmesh.TUNNEL_SHOULDER + roadmesh.LOW_PORTAL_SIDE
        caps, ups, any_ = [None, None], list(ups), False
        lines, radius = [], []
        for gk, lns in ground.items():
            hw = self._hw.get(gk)
            if hw is None:
                continue
            for ln in lns:
                lines.append(np.asarray(ln, np.float64).reshape(-1, 2))
                radius.append(hw)
        if not lines or len(xy) < 2:
            return None, ups
        pts = np.concatenate(lines)
        starts = np.concatenate([[0], np.cumsum([len(ln) for ln in lines])]).astype(np.int64)
        radius = np.asarray(radius, np.float64)
        ds = np.arange(0.0, roadmesh.LOW_CAP_M + 0.25, 0.5)
        for e, (at, nb) in enumerate(((0, 1), (-1, -2))):
            if not mouths[e]:
                continue
            p = xy[at]
            tin = xy[nb] - p
            tin = tin / max(float(np.hypot(*tin)), 1e-9)
            nrm = np.array([-tin[1], tin[0]])
            mid = p + tin * ds[:, None]
            probes = np.stack([mid, mid + nrm * side, mid - nrm * side], 1).reshape(-1, 2)
            got = fast._ef.first_probe_near_lines(pts, starts, radius, ROAD_CAP_GAP_M, (float(p[0]), float(p[1])),
                                                  ROAD_CAP_JOIN_M, probes)
            depth = None
            if got is not None:
                i, hx, hy = got
                depth = max(0.0, float(ds[i // 3]) - ROAD_CAP_GAP_M)
                road_z = float(self._ground(np.array([[hx, hy]]))[0])
                top = road_z + ROAD_CAP_OVER_M - float(floor[at]) - th
                ups[e] = float(max(roadmesh.LOW_PORTAL_UP, min(ups[e], top)))
            if depth is not None:
                caps[e], any_ = depth, True
        return (tuple(caps) if any_ else None), ups

    def cap_m(self, fid):
        """(start, end): a low arch's stone top depth over each mouth, None for roadmesh.LOW_CAP_M."""
        with self.lock:
            return self.caps.get(fid, (None, None))

    def portal_ups(self, fid):
        """(at the part's start, at its end): the portal faces' reach over the vault, PORTAL_GROUND_M over the ground
        just inside (within PORTAL_LOOK_M), PORTAL_UP_MIN..roadmesh.PORTAL_FRAME (a 6 m frame stood ~10 m over the
        flat park at the 79th Street Transverse, 2026-10-17); the full frame when not known."""
        import roadmesh
        with self.lock:
            return self.ups.get(fid, [roadmesh.PORTAL_FRAME, roadmesh.PORTAL_FRAME])

    def height(self, fid):
        """The tube's height (floor to vault) of a sub-part: a low arch's, else roadmesh.TUNNEL_H."""
        import roadmesh
        with self.lock:
            return self.heights.get(fid, roadmesh.TUNNEL_H)

    def mouth(self, fid):
        """(start, end) of a sub-part: True where it is a portal."""
        with self.lock:
            return self.mouths.get(fid, (True, True))

    def _split(self, parts):
        """parts (whole map parts) as their sections' sub-parts (fid#n, spec, xy, z, lon)."""
        out = []
        for fid, spec, xy, z, lon in parts:
            with self.lock:
                secs = self.sections.get(fid)
            if not secs:
                out.append((f"{fid}#0", spec, xy, z, lon))
                continue
            for n, (a, b, kd) in enumerate(secs):
                if b > a:
                    out.append((f"{fid}#{n}", spec, np.asarray(xy)[a:b + 1], np.asarray(z)[a:b + 1], lon))
        return out

    def open_parts(self, cells_feats, box):
        """[(fid, spec, xy, z, lon)] of the open sections near box: roads at the floor (the ground graded to it,
        grading.Grader._open_tunnels; _tunnel_chunks)."""
        return self._open_of(super().parts(cells_feats, box))

    def _open_of(self, whole):
        with self.lock:
            opened = set(self.open)
        # (an open section under water is an underwater tunnel with no ground over it: the Sydney Harbour Tunnel
        # was drawn as a road across the harbour, 2026-10-18)
        return self._dry([p for p in self._split(whole) if p[0] in opened])

    def parts(self, cells_feats, box):
        return self._closed_of(super().parts(cells_feats, box))

    def _closed_of(self, whole):
        out = self._split(whole)
        with self.lock:
            opened = set(self.open)
        return self._dry([p for p in out if p[0] not in opened])

    def parts_todo(self, todo, solved_only=False, open_=False):
        """BridgeProfiles.parts_todo split into sections: the closed ones (parts), or the open ones (open_)."""
        whole = super().parts_todo(todo, solved_only)
        return self._open_of(whole) if open_ else self._closed_of(whole)

    def _adopt_extra(self, comp):
        pids = {self._part_id(k) for k in comp}

        def mine(d):
            return {k: v for k, v in d.items() if str(k).split("#")[0] in pids}
        return {"sections": mine(self.sections), "heights": mine(self.heights), "ups": mine(self.ups),
                "caps": mine(self.caps), "mouths": mine(self.mouths), "open": {o for o in self.open
                                                                               if o.split("#")[0] in pids},
                "drowned": {k: v for k, v in self._drowned.items() if k in pids},
                "subsea": self._subsea & pids}

    def _adopt_take(self, extra):
        if not extra:
            return
        for name in ("sections", "heights", "ups", "caps", "mouths"):
            for k, v in extra[name].items():
                getattr(self, name).setdefault(k, v)
        self.open.update(extra["open"])
        self._subsea.update(extra.get("subsea", ()))
        for k, v in extra["drowned"].items():
            self._drowned.setdefault(k, v)

    def _dry(self, out):
        """Dry parts and successfully solved wet profiles; incomplete/unsupported wet chains remain excluded."""
        if self.wet is None or not out:
            return out
        keep = []
        for p in out:
            with self.lock:
                if str(p[0]).split("#")[0] in self._subsea:
                    keep.append(p)
                    continue
                if self._drowned.get(str(p[0]).split("#")[0]):      # (its chain is under water: _mark_drowned)
                    continue
            xy = np.asarray(p[2], np.float64)
            la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
            # under water for a good part of it (a portal on a river bank, Fort Pitt's east end 0.6 m "in" the
            # Monongahela, is not: that dropped one of its two tubes, 2026-10-17)
            if (np.asarray(self.wet(la, lo)) > 0.5).mean() <= TUNNEL_WET_SHARE:
                keep.append(p)
        return keep


class ChainStore:
    """Bridge (or tunnel) chain profiles shared by every anchor of a process (geotiles, 2026-10-29): one
    BridgeProfiles / TunnelProfiles per geotiles.CHAIN_Z region in that region's frame (a WorldFrame: nearly level
    over a chain), a part asked for in the store of its first point's region, a chain solved in the store of its first
    part's (owner_of): one deck profile per chain, whatever the anchor or the store a tile asks. cell_fn, elev (the
    physics ground: anchor-free, as grading's heights are) as BridgeProfiles'; setup(profiles) sets wet, infra_fn,
    raw, ends on each store made. Anchors see it through a ChainView."""

    KEPT = 12           # stores kept (the farthest regions behind go first)

    def __init__(self, cls, cell_fn, elev, setup=None):
        self.cls, self.cell_fn, self.elev, self.setup = cls, cell_fn, elev, setup
        self.stores = OrderedDict()     # region -> profiles
        self.lock = threading.Lock()

    def store(self, region):
        region = (int(region[0]), int(region[1]))
        with self.lock:
            got = self.stores.get(region)
            if got is not None:
                self.stores.move_to_end(region)
                return got
            import geotiles
            got = self.cls(geotiles.frame(region[0], region[1], geotiles.CHAIN_Z), self.cell_fn, self.elev)
            got.owner_of = self._owner
            got.region = region
            if self.setup is not None:
                self.setup(got)
            self.stores[region] = got
            while len(self.stores) > self.KEPT:
                self.stores.popitem(last=False)
            return got

    def store_at(self, lat, lon):
        import geotiles
        return self.store(geotiles.tile_of(float(lat), float(lon), geotiles.CHAIN_Z))

    def _owner(self, prof, comp, pool):
        """The store whose region holds chain comp's first part's start."""
        first = min(comp)
        (la, lo), _ = prof._ends_ll(pool[first][1])
        return self.store_at(la, lo)


class _Mapper:
    """Points of one frame's world in another's (geotiles.world_map), rigid parts by the rotation between them."""

    def __init__(self, src, dst):
        import geotiles
        self.src, self.dst, self.same = src, dst, src.lat0 == dst.lat0 and src.lon0 == dst.lon0
        self._gt = geotiles
        if not self.same:
            r, _t, _c = geotiles.placement(src, dst, (0.0, 0.0, 0.0))
            self.r = r

    def pts(self, p):
        p = np.asarray(p, np.float64)
        if self.same or not len(p):
            return p.copy()
        return self._gt.world_map(self.src, self.dst, p)

    def dir2(self, ux, uy):
        if self.same:
            return ux, uy
        v = self.r[:2, :2] @ np.array([ux, uy])
        n = max(float(np.hypot(*v)), 1e-12)
        return float(v[0] / n), float(v[1] / n)

    def pier(self, q):
        if self.same:
            return q
        x, y, ux, uy, zt, zb = q[:6]
        a = self.pts(np.array([[x, y, zt], [x, y, zb]], np.float64))
        ux, uy = self.dir2(ux, uy)
        return (float(a[0, 0]), float(a[0, 1]), ux, uy, float(a[0, 2]), float(a[1, 2])) + tuple(q[6:])

    def beam(self, b):
        if self.same:
            return b
        a = self.pts(np.array([np.asarray(b[0], np.float64)[:3], np.asarray(b[1], np.float64)[:3]]))
        return (a[0].copy(), a[1].copy()) + tuple(b[2:])


class ChainView:
    """An anchor's view of a ChainStore, with BridgeProfiles' / TunnelProfiles' read side (parts, open_parts,
    piers_in, beams_in, depth_of, _ground, heights, mouths, height, portal_ups, cap_m, mouth): the parts asked for in
    their regions' stores, their geometry mapped exactly into this frame's world (kept per part). prefer: another
    ChainStore whose chains, where solved already, are taken first (the AI graph and the far bridges take the road
    tiles' decks: one profile per chain)."""

    def __init__(self, cs, frame, prefer=None):
        self.cs, self.frame, self.prefer = cs, frame, prefer
        self.cell_fn = cs.cell_fn
        self.elev = cs.elev
        self.lock = threading.Lock()
        self._mapped = {}           # (store region, pid, id(xy)) -> (xy ref, mapped part)
        self._piers = {}            # store region -> (version, mapped piers, mapped beams)
        self.specs_of, self.parts_of = cs.cls.specs_of, cs.cls.parts_of

    mapper = staticmethod(lambda src, dst: _Mapper(src, dst))

    def __getattr__(self, name):        # wet, infra_fn, raw, ends: the stores' settings (setup)
        if name in ("wet", "infra_fn", "raw", "ends"):
            st = list(self.cs.stores.values())
            return getattr(st[0], name, None) if st else None
        raise AttributeError(name)

    def _todo(self, cells_feats, box):
        """{store: [((fid, j), feats, i)]} of the parts in these cells that come near box (this frame)."""
        x0, y0, x1, y1 = box
        by, queued = {}, set()
        for feats in cells_feats:
            if feats is None or len(feats) == 0:
                continue
            bs = self.specs_of(feats)
            for i in range(len(feats)):
                if bs[i] is None:
                    continue
                fid = str(feats.ids[i])
                for j, ring in enumerate(self.parts_of(feats, i)):
                    if (fid, j) in queued:
                        continue
                    e, n = mapgeom.enu_xy(feats, self.frame, ring)
                    if e.max() < x0 - NEAR_M or e.min() > x1 + NEAR_M or n.max() < y0 - NEAR_M or n.min() > y1 + NEAR_M:
                        continue
                    queued.add((fid, j))
                    st = self.cs.store_at(float(ring[0][1]), float(ring[0][0]))
                    by.setdefault(st.region, (st, []))[1].append(((fid, j), feats, i))
        return by

    def _map(self, st, got):
        out = []
        mp = None
        for pid, spec, xy, z, lon in got:
            k = (st.region, pid, id(xy))
            with self.lock:
                hit = self._mapped.get(k)
            if hit is None or hit[0] is not xy:
                mp = mp or _Mapper(st.frame, self.frame)
                xyz = mp.pts(np.c_[np.asarray(xy, np.float64)[:, :2], np.asarray(z, np.float64)])
                hit = (xy, (pid, spec, xyz[:, :2].copy(), xyz[:, 2].copy(), lon))
                with self.lock:
                    if len(self._mapped) > 20000:
                        self._mapped.clear()
                    self._mapped[k] = hit
            out.append(hit[1])
        return out

    def _parts(self, cells_feats, box, **kw):
        out = []
        for _r, (st, todo) in sorted(self._todo(cells_feats, box).items()):
            if self.prefer is not None:
                pst = self.prefer.store(st.region)
                with pst.lock:
                    solved = [t for t in todo if t[0] in pst.done]
                out += self._map(pst, pst.parts_todo(solved, solved_only=True, **kw))
                todo = [t for t in todo if t not in solved]
            out += self._map(st, st.parts_todo(todo, **kw))
        return out

    def parts(self, cells_feats, box):
        return self._parts(cells_feats, box)

    def parts_with_ids(self, cells_feats, box):
        """Bridge parts with their stable (fid, part index), including when tiles ask for different spans."""
        return self._parts(cells_feats, box, part_ids=True)

    def open_parts(self, cells_feats, box):
        return self._parts(cells_feats, box, open_=True)

    def _stores(self):
        out = list(self.cs.stores.values())
        if self.prefer is not None:
            out += list(self.prefer.stores.values())
        return out

    def _mapped_piers(self, st):
        with st.lock:
            ver = (len(st.piers), sum(len(v) for v in st.piers.values()), len(st.trusses),
                   sum(len(v) for v in st.trusses.values()))
            piers = [q for v in st.piers.values() for q in v]
            beams = [b for v in st.trusses.values() for b in v]
        key = (id(st), st.region)
        with self.lock:
            hit = self._piers.get(key)
        if hit is None or hit[0] != ver:
            mp = _Mapper(st.frame, self.frame)
            hit = (ver, [mp.pier(q) for q in piers], [mp.beam(b) for b in beams])
            with self.lock:
                self._piers[key] = hit
        return hit[1], hit[2]

    def piers_in(self, box):
        x0, y0, x1, y1 = box
        out, seen = [], set()
        for st in self._stores():
            for q in self._mapped_piers(st)[0]:
                k = (round(q[0], 3), round(q[1], 3))
                if box_owns(box, q[0], q[1]) and k not in seen:
                    seen.add(k)
                    out.append(q)
        return out

    def beams_in(self, box):
        x0, y0, x1, y1 = box
        out, seen = [], set()
        for st in self._stores():
            for b in self._mapped_piers(st)[1]:
                mx, my = (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2
                k = (round(float(mx), 3), round(float(my), 3), round(float(b[0][2]), 3))
                if box_owns(box, mx, my) and k not in seen:
                    seen.add(k)
                    out.append(b)
        return out

    def depth_of(self, fid):
        ds = []
        for st in self._stores():
            with st.lock:
                ds += [d for k, d in st.__dict__.get("depth", {}).items() if st._part_id(k) == fid and d is not None]
        return max(ds) if ds else DECK_FULL_DEPTH

    def _ground(self, xy):
        xy = np.asarray(xy, np.float64)
        la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        return self.frame.geodetic_to_enu(la, lo, np.asarray(self.elev(la, lo), np.float64))[2]

    def _merged(self, name):
        out = {}
        for st in self._stores():
            with st.lock:
                for k, v in getattr(st, name, {}).items():
                    out.setdefault(k, v)
        return out

    heights = property(lambda self: self._merged("heights"))
    mouths = property(lambda self: self._merged("mouths"))

    def _sub(self, method, fid, default):
        for st in self._stores():
            with st.lock:
                if any(fid in getattr(st, d, {}) for d in ("heights", "ups", "caps", "mouths")):
                    break
        else:
            st = None
        if st is None:
            return default
        return getattr(st, method)(fid)

    def height(self, fid):
        return self._sub("height", fid, roadmesh.TUNNEL_H)

    def portal_ups(self, fid):
        return self._sub("portal_ups", fid, [roadmesh.PORTAL_FRAME, roadmesh.PORTAL_FRAME])

    def cap_m(self, fid):
        return self._sub("cap_m", fid, (None, None))

    def mouth(self, fid):
        return self._sub("mouth", fid, (True, True))


def wet_run_inside(xy, wet):
    """The longest run (m) of wet points along the line xy that touches neither end: a tunnel under a river (the
    Burnley Tunnel's 70 m under the Yarra, 2 % of its 3.2 km, GitHub #128), not one whose portal is on a bank (Fort
    Pitt's east end 0.6 m "in" the Monongahela). earthfast.wet_run_inside (Rust)."""
    import fast
    fast.require(48, "wet_run_inside")
    return float(fast._ef.wet_run_inside(np.ascontiguousarray(xy, np.float64), np.ascontiguousarray(wet, bool)))
