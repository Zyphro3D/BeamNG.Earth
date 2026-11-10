"""Road builder: polyline geometry (split out of roads.py): offsets, cuts outside blockers, portal covers.
"""
import numpy as np
from roadbase import (
    CONCRETE_WALL_CLASSES,
    MIN_PIECE_M,
    PORTAL_SHALLOW_COVER_M,
    SECTION_MIN_M,
    TUNNEL_COVER,
    TUNNEL_GRADE,
    TUNNEL_ROAD_CLEAR_M,
    TUNNEL_ROAD_MIN_SIN,
    TUNNEL_ROAD_REACH_M,
    TUNNEL_STEP,
    WALK_STEP_M,
    densify,
)
from roaddecks import (
    _end_dists,
    _smooth_pinned,
    deck_heights,
)


def portal_cover(kinds, cover, spec):
    """kinds with a major road's (CONCRETE_WALL_CLASSES) open run at either end of its tunnel covered (a low arch, in
    concrete: roadmesh) where the median ground over its floor is PORTAL_SHALLOW_COVER_M or more: the map's portal is a
    cut-and-cover box under shallow ground there (the Gotthard road tunnel's north portal under the meadow at
    Goeschenen stood 90 m further in, an open cut before it, 2026-10-28). Park roads keep their open cuts (Central
    Park's transverses). The meshes (TunnelProfiles) and the grading both."""
    kinds = np.array(kinds, np.int64)
    if spec is None or spec.cls not in CONCRETE_WALL_CLASSES or not (kinds > 0).any():
        return kinds
    cover = np.asarray(cover, np.float64)
    n = len(kinds)
    for rng in (range(n), range(n - 1, -1, -1)):
        run = []
        for q in rng:
            if kinds[q] != 0:
                break
            run.append(q)
        if run and len(run) < n and np.isfinite(cover[run]).any() and \
                float(np.nanmedian(cover[run])) >= PORTAL_SHALLOW_COVER_M:
            kinds[run] = 1
    return kinds


def cover_roads(xy, s, kinds, lines, halves):
    """kinds (tunnel_sections') with each covered section carried on until the ground roads crossing over it (lines
    (k, 2) in xy's frame, carriageway half widths) are covered, TUNNEL_ROAD_*: earthfast.tunnel_cover_roads. The
    road meshes (TunnelProfiles) and the grading (grading._tunnel_sections) both: an open section the grading did
    not know of cut its trench through the A-7's lanes near Malaga (GitHub #450)."""
    kinds = np.asarray(kinds, np.int64)
    if not lines or not (kinds > 0).any() or len(xy) < 2:
        return kinds
    import fast
    fast.require(95, "tunnel_cover_roads")
    pts = np.ascontiguousarray(np.concatenate(lines))
    starts = np.concatenate([[0], np.cumsum([len(ln) for ln in lines])]).astype(np.int64)
    return np.asarray(fast._ef.tunnel_cover_roads(
        np.ascontiguousarray(np.asarray(xy, np.float64)[:, :2]), np.ascontiguousarray(s, np.float64),
        np.ascontiguousarray(kinds), pts, starts, np.asarray(halves, np.float64), TUNNEL_ROAD_CLEAR_M,
        TUNNEL_ROAD_MIN_SIN, TUNNEL_ROAD_REACH_M, SECTION_MIN_M), np.int64)


def tunnel_profile(lines, ground_fn):
    """Floor heights of chained tunnel parts (as deck_heights: polylines (k, 2), joined at their ends; the chain's
    ends are the portals, at the ground's height there). Straight between the portals, but at least TUNNEL_COVER
    under the ground over it (an underwater tunnel dips under the bed: Fort McHenry's portals are on land), falling
    no steeper than TUNNEL_GRADE from a portal; smoothed. One array per line (NaN for a closed loop). Pure given
    ground_fn(xy (m, 2)) -> heights."""
    if not lines:
        return []
    straight = deck_heights(lines, ground_fn)
    edges, ez, dist = _end_dists(lines, ground_fn)
    allxy = np.concatenate([np.asarray(ln[:, :2], np.float64) for ln in lines])
    gz = np.asarray(ground_fn(allxy), np.float64)
    out, k = [], 0
    import grading
    for (a, b, s), z0 in zip(edges, straight):
        g = gz[k:k + len(s)]
        k += len(s)
        if not dist or not np.isfinite(z0).all():
            out.append(np.full(len(s), np.nan))
            continue
        # under TUNNEL_COVER of ground where the straight line has less (a hollow, a low hill), reached at no more than
        # TUNNEL_GRADE from the portals (a mountain's steep portal face gives the cover within metres)
        z = np.minimum(z0, g - TUNNEL_COVER)
        low = np.full(len(s), -np.inf)
        for e, d in enumerate(dist):
            dk = np.minimum(d[a] + s, d[b] + (s[-1] - s))
            low = np.maximum(low, np.where(np.isfinite(dk), ez[e] - TUNNEL_GRADE * dk, -np.inf))
        z = np.maximum(z, low)
        if len(z) > 3:          # (the ends' trend kept: smoothed then pinned, a floor bent up beside a portal)
            z = _smooth_pinned(z, TUNNEL_STEP, grading.SIGMA_M)
        out.append(z)
    return out


def centre_colour(lon):
    """Centre line colour: yellow in the Americas, white elsewhere (the country rule comes later)."""
    return "line_yellow" if -170.0 <= float(lon) <= -30.0 else "line_white"


def offset_polyline(pts, d):
    """The polyline moved d m (or d[k] m at each point) to its left (xy; z kept), with mitred joints (miter at most
    2x). Pure."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts.copy()
    t = np.diff(pts[:, :2], axis=0)
    L = np.hypot(t[:, 0], t[:, 1])
    L[L < 1e-9] = 1e-9
    t /= L[:, None]
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)          # left normals per segment
    vn = np.empty((len(pts), 2))
    vn[0], vn[-1] = nrm[0], nrm[-1]
    if len(pts) > 2:
        m = nrm[:-1] + nrm[1:]
        ml = np.hypot(m[:, 0], m[:, 1])
        ml[ml < 1e-9] = 1e-9
        m /= ml[:, None]
        cosh = np.clip(np.sum(m * nrm[1:], axis=1), 0.5, 1.0)  # miter scale 1/cos(half angle), at most 2
        vn[1:-1] = m / cosh[:, None]
    out = pts.copy()
    d = np.asarray(d, np.float64)
    out[:, :2] += vn * (d[:, None] if d.ndim else d)
    return out


def _seg_dist(p, a, b):
    """Distance from points p (N, 2) to segments a-b (M, 2): (N, M)."""
    ab = b - a
    L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    ap = p[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(ap * ab[None], axis=2) / L2[None], 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.hypot(p[:, None, 0] - q[..., 0], p[:, None, 1] - q[..., 1])


def _length(pts):
    """m along a polyline (k, >= 2). Pure."""
    return float(np.sum(np.hypot(*np.diff(np.asarray(pts)[:, :2], axis=0).T)))


def _within_any(c, cuts, boxes):
    """Whether point c (2,) lies within half of any (centreline (m, 2), half) of cuts: their boxes first (boxes: a
    dict kept per cut list), the exact distance only for the few boxes holding c (PR #505's paint polygons asked
    every road of the tile: ~90k distances per 16 tiles)."""
    bb = boxes.get(id(cuts))
    if bb is None:
        if not cuts:
            return False
        lo = np.array([np.asarray(p)[:, :2].min(0) - h for p, h in cuts])
        hi = np.array([np.asarray(p)[:, :2].max(0) + h for p, h in cuts])
        bb = boxes[id(cuts)] = (lo, hi, cuts)
    lo, hi, _ = bb
    import furniture
    for i in np.flatnonzero((lo[:, 0] <= c[0]) & (hi[:, 0] >= c[0]) & (lo[:, 1] <= c[1]) & (hi[:, 1] >= c[1])):
        if furniture._seg_dist_min(c, cuts[i][0]) < cuts[i][1]:
            return True
    return False


def cut_outside(pts, blockers, spacing=2.0, min_piece=None):
    """The pieces of a polyline (k, 3) outside every blocker (centreline xy (m, 2), half width): densified
    to `spacing`, points inside a blocker removed, runs of at least min_piece (default MIN_PIECE_M) kept. Pure."""
    if not blockers:
        return [pts]
    d = densify(np.asarray(pts, np.float64), spacing)
    import fast
    fast.require(0, "inside_blockers")      # the Rust test (712 calls a tile set, ~1 s of numpy)
    segs, starts, hws = _blocker_segments(blockers)
    inside = np.asarray(fast._ef.inside_blockers(np.ascontiguousarray(d[:, :2]), segs, starts, hws))
    return _runs_outside(d, inside, MIN_PIECE_M if min_piece is None else min_piece)


_BLOCKERS = ()                  # ((blockers list, its segment arrays), ...): the last few lists (a tile alternates
                                # between its sidewalk and road blockers over many calls)


def _blocker_segments(blockers):
    """(segs (m, 4), starts (B + 1,), half widths (B,)) of [(centreline (k, 2), half width)] (lines of < 2
    points left out)."""
    global _BLOCKERS
    for last, arr in _BLOCKERS:         # one read of the tuple: another thread may replace it meanwhile
        if last is blockers and arr[3] == len(blockers):
            return arr[:3]
    # every line's segments in one go (an hstack per line: a road's blockers are all the other roads near the tile,
    # ~30k hstacks a 9-tile set, 2026-10-25)
    lines = [np.asarray(line, np.float64) for line, _ in blockers]
    keep = [k for k, line in enumerate(lines) if len(line) >= 2]
    if keep:
        n = np.array([len(lines[k]) for k in keep], np.int64)
        pts = np.concatenate([lines[k][:, :2] for k in keep])
        first = np.ones(len(pts), bool)
        first[np.cumsum(n) - 1] = False         # a line's last point starts no segment
        a = np.flatnonzero(first)
        segs = np.empty((len(a), 4))
        segs[:, :2] = pts[a]
        segs[:, 2:] = pts[a + 1]
        starts = np.concatenate((np.zeros(1, np.int64), np.cumsum(n - 1)))
        hws = np.array([float(blockers[k][1]) for k in keep], np.float64)
    else:
        segs, starts, hws = np.zeros((0, 4)), np.zeros(1, np.int64), np.zeros(0, np.float64)
    arr = (segs, starts, hws, len(blockers))
    _BLOCKERS = ((blockers, arr),) + _BLOCKERS[:3]
    return arr[:3]


def _on_level(frame, elev, run):
    """The pieces of a sidewalk run (k, 3) whose ground (elev(lat, lon), the graded ground) lies within WALK_STEP_M of
    the run's own level (MIN_PIECE_M long at least)."""
    if len(run) < 2:
        return [run]
    la, lo, _ = frame.enu_to_geodetic(run[:, 0], run[:, 1], np.zeros(len(run)))
    g = np.asarray(frame.geodetic_to_enu(la, lo, np.asarray(elev(la, lo), np.float64))[2], np.float64)
    off = ~(np.abs(g - run[:, 2]) <= WALK_STEP_M)
    return [run] if not off.any() else _runs_outside(run, off)


def _runs_outside(d, inside, min_piece=MIN_PIECE_M):
    out, start = [], None
    for k in range(len(d) + 1):
        if k < len(d) and not inside[k]:
            if start is None:
                start = k
        elif start is not None:
            run = d[start:k]
            if len(run) >= 2 and float(np.sum(np.hypot(*np.diff(run[:, :2], axis=0).T))) >= min_piece:
                out.append(run)
            start = None
    return out
