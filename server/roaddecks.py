"""Road builder: bridge deck heights (split out of roads.py): chaining a bridge's ways, deck profiles and curves, and
tunnel profiles.
"""
import math
import os
import numpy as np
from roadbase import (
    APPROACH_GRADE,
    AT_GRADE_M,
    AT_GRADE_REACH_M,
    CHORD_KINK,
    CHORD_SHELF_M,
    CHORD_TOL,
    DECK_CLEAR,
    DECK_CLEAR_REACH_M,
    DECK_CREST_R,
    DECK_CURVE_K,
    DECK_CURVE_KMIN,
    DECK_CURVE_MIN,
    DECK_CURVE_STEP,
    DECK_DESIGN_GRADE,
    DECK_EASE_M,
    DECK_GRADE,
    DECK_MAX_GRADE,
    DECK_ROAD_CLEAR,
    DECK_ROAD_HALF,
    DECK_SIGMA,
    DECK_SIGMA_MAX,
    DECK_SIGMA_SHARE,
    DECK_STEEP_GRADE,
    DUAL_DECK_M,
    END_JOIN_COS,
    END_JOIN_GROUND_M,
    END_JOIN_M,
    END_JOIN_MIN_M,
    END_ON_MARGIN_M,
    LIFT_EASE_M,
    LIFT_END_GRADE,
    LIFT_GRADE,
    MERGE_TOUCH_M,
    PIN_M,
    PIN_ROAD_M,
    STACK_STEP_M,
    TIE_EASE_GRADE,
    TIE_EASE_MIN_M,
    TIE_JOIN_M,
    TIE_MEASURED_M,
    WATER_CAUSEWAY_CLEAR,
    WATER_CAUSEWAY_M,
    WATER_CLEAR_MAX,
    WATER_CLEAR_MIN,
    WATER_CLEAR_SHARE,
)


class _EndJoin:
    """Line end points joined into nodes when within tol m of a point already in one (the first point seen names the
    node; a later one joins the node of its nearest point within tol, and is one of its points after). Rounding to a
    0.5 m grid split ends a few cm apart either side of a grid line and joined ends 0.68 m apart on a diagonal (from
    cody196's PR #368). Against the node's first point alone, four ramps' ends within 0.5 m of their neighbours at a
    Springfield junction made two nodes: one ramp's end an abutment at the ground, 9 m under the others (2026-10-31,
    rampcheck.py). A grid of tol cells, the 3 x 3 around a point looked up. Deterministic for a given order; nodes
    numbered in the order they are made."""

    def __init__(self, tol=0.5):
        self.tol, self.grid, self.pts, self.node, self.n = tol, {}, [], [], 0

    def __call__(self, p):
        x, y = float(p[0]), float(p[1])
        gx, gy = math.floor(x / self.tol), math.floor(y / self.tol)
        best, bd = None, self.tol * self.tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in self.grid.get((gx + dx, gy + dy), ()):
                    q = self.pts[i]
                    d = (q[0] - x) ** 2 + (q[1] - y) ** 2
                    if d < bd or (d == bd and best is not None and i < best):
                        best, bd = i, d
        if best is not None:
            nd = self.node[best]
            if bd > 1e-12:                  # (a point of its node now: the next end within tol of it joins too)
                self.pts.append((x, y))
                self.node.append(nd)
                self.grid.setdefault((gx, gy), []).append(len(self.pts) - 1)
            return nd
        self.pts.append((x, y))
        self.node.append(self.n)
        self.grid.setdefault((gx, gy), []).append(len(self.pts) - 1)
        self.n += 1
        return self.n - 1


def _chain_graph(lines):
    """Nodes (end points within 0.5 m joined) and edges (start node, end node, arc lengths) of parts."""
    join, nodes = _EndJoin(), []

    def node(p):
        k = join(p)
        if k == len(nodes):
            nodes.append(np.asarray(p[:2], np.float64))
        return k
    edges = []
    for ln in lines:
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ln[:, :2], axis=0).T))]
        edges.append((node(ln[0]), node(ln[-1]), s))
    adj = [[] for _ in nodes]
    for a, b, s in edges:
        adj[a].append((b, s[-1]))
        adj[b].append((a, s[-1]))
    return nodes, edges, adj


def _dijkstra(adj, seeds):
    """Shortest distances along the chain from seeds {node: start distance}."""
    import heapq
    d = np.full(len(adj), np.inf)
    q = []
    for n, d0 in seeds.items():
        if d0 < d[n]:
            d[n] = d0
            q.append((d0, n))
    heapq.heapify(q)
    while q:
        dd, u = heapq.heappop(q)
        if dd > d[u]:
            continue
        for v, w in adj[u]:
            if dd + w < d[v]:
                d[v] = dd + w
                heapq.heappush(q, (dd + w, v))
    return d


# A deck part's end that no other part's end meets (within _EndJoin's 0.5 m) and no ground road goes on from (within
# END_JOIN_GROUND_M) is no abutment when it lies on another deck part's end, along it: a carriageway's split or merge
# drawn a few metres off the joint. Held at the ground as an abutment, the branch's deck rose from the street under the
# viaduct beside the main deck, a cliff of 7-14 m across a motorway's lanes the AI drove off (I-83 in Baltimore, 3.3 m
# off the joint, the Springfield Interchange; 2026-10-29). Such an end is solved as joined there (earthfast.
# deck_end_joins): within min(END_JOIN_M, the wider part's half width), directions within acos(END_JOIN_COS).
END_JOIN_ON = os.environ.get("BNE_DECK_END_JOINS", "on") != "off"


def _end_joins(lines, halves, ground_lines):
    """[(part, 0 | -1, (x, y))]: the free ends of deck lines (k, 2+) that join another line's end there (END_JOIN_*;
    earthfast.deck_end_joins), halves their half widths, ground_lines the ground roads near them. Pure."""
    if not END_JOIN_ON or len(lines) < 2:
        return []
    import fast
    fast.require(105, "deck_end_joins")
    pts = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    off = np.r_[0, np.cumsum([len(p) for p in pts])].astype(np.int64)
    gl = [np.asarray(g, np.float64)[:, :2] for g in ground_lines if len(g) >= 2]
    gends = np.array([p for g in gl for p in (g[0], g[-1])], np.float64).reshape(-1, 2)
    got = np.asarray(fast._ef.deck_end_joins(
        np.ascontiguousarray(np.concatenate(pts)), off, np.asarray(halves, np.float64), np.ascontiguousarray(gends),
        np.array([0.5, END_JOIN_M, END_JOIN_COS, END_JOIN_GROUND_M, END_JOIN_MIN_M], np.float64)), np.float64)
    return [(e // 2, 0 if e % 2 == 0 else -1, (float(got[e, 0]), float(got[e, 1])))
            for e in range(len(got)) if np.isfinite(got[e, 0])]


def _end_ons(lines, halves, levels, ground_lines):
    """[(part, 0 | -1, the part it stands on, arc m along that one)]: the free ends of deck lines standing on another
    deck of their level (END_ON_*; earthfast.deck_end_on). Pure."""
    if not END_JOIN_ON or len(lines) < 2:
        return []
    import fast
    fast.require(107, "deck_end_on")
    pts = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    off = np.r_[0, np.cumsum([len(q) for q in pts])].astype(np.int64)
    gl = [np.asarray(g, np.float64)[:, :2] for g in ground_lines if len(g) >= 2]
    gends = np.array([q for g in gl for q in (g[0], g[-1])], np.float64).reshape(-1, 2)
    got = np.asarray(fast._ef.deck_end_on(
        np.ascontiguousarray(np.concatenate(pts)), off, np.asarray(halves, np.float64),
        np.asarray([float(v) for v in levels], np.float64), np.ascontiguousarray(gends),
        np.array([0.5, END_JOIN_GROUND_M, END_ON_MARGIN_M], np.float64)), np.float64)
    return [(e // 2, 0 if e % 2 == 0 else -1, int(got[e, 0]), float(got[e, 1]))
            for e in range(len(got)) if got[e, 0] >= 0]


def _tie_beside(lines, zs, specs, measured=None):
    """zs with each branch (a ramp, else the narrower part) that leaves or joins another part (its end on that part's
    deck) held at
    that part's height while their slabs are joined (TIE_*), eased into its own profile after. lines: (k, >= 2) per
    part, zs: heights per point, specs: their Specs. Pure."""
    zs = [np.asarray(z, np.float64).copy() for z in zs]
    pts = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    n = len(pts)
    arcs = [np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))] if len(p) >= 2 else np.zeros(len(p)) for p in pts]

    def project(j, q):
        """(distance, height) of q on part j's line."""
        P, ab = pts[j][:-1], np.diff(pts[j], axis=0)
        t = np.clip(((q - P) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        near = P + t[:, None] * ab
        d = np.hypot(*(q - near).T)
        k = int(np.argmin(d))
        return float(d[k]), float(zs[j][k] + t[k] * (zs[j][k + 1] - zs[j][k]))

    own = [z.copy() for z in zs]

    def travel(i, e):
        """The unit direction traffic takes on part i at its end e (its geometry's, reversed for dir -1)."""
        p = pts[i]
        v = (p[1] - p[0]) if e == 0 else (p[-1] - p[-2])
        v = v * (getattr(specs[i], "dir", 0) or 1)
        return v / max(float(np.hypot(*v)), 1e-12)

    def opposite(i, e, j):
        """Whether parts i and j are both one way with their traffic opposite where i's end e meets j: the other
        carriageway of a divided road, never the road a branch leaves (a branch runs with its road's traffic). The
        narrower carriageway's deck end, 5.3 m from the wider one's centre line, was held at that one's unraised height:
        a 1.2 m step onto its own raised approach (#2373, 2026-11-10)."""
        if not (getattr(specs[i], "oneway", False) and getattr(specs[j], "oneway", False)):
            return False
        P, ab = pts[j][:-1], np.diff(pts[j], axis=0)
        q = pts[i][e]
        t = np.clip(((q - P) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        k = int(np.argmin(np.hypot(*(q - (P + t[:, None] * ab)).T)))
        vj = ab[k] * (getattr(specs[j], "dir", 0) or 1)
        vj = vj / max(float(np.hypot(*vj)), 1e-12)
        return float(travel(i, e) @ vj) < -0.5

    def fixed(i, e, j):
        """Whether end e of part i keeps its own height when i is held at part j's: an abutment (the road going on
        on the ground from it) or a joint with another part (that part goes on at i's height there); only an end on
        j's own deck and on nothing else follows j."""
        p = pts[i][e]
        for k in range(n):
            if k in (i, j) or len(pts[k]) < 2:
                continue
            if min(np.hypot(*(pts[k][0] - p)), np.hypot(*(pts[k][-1] - p))) <= 1.0:
                return True
        return project(j, p[None])[0] > float(specs[j].width) / 2 + 1.0

    for i in range(n):
        if len(pts[i]) < 2:
            continue
        wi = float(specs[i].width)
        for e in (0, -1):
            p = pts[i][e]
            best = None
            for j in range(n):
                if j == i or len(pts[j]) < 2:
                    continue
                wj = float(specs[j].width)
                # (its end on the other's deck: at its end, or a few metres off it, where the map starts a ramp beside
                # its road rather than on its joint: I-95 northbound by the Middle Branch, 2.2 m off, their slabs
                # overlapping 0.2 m apart in height)
                if project(j, p[None])[0] > wj / 2 + 1.0:
                    continue
                branch = (getattr(specs[i], "link", False) and not getattr(specs[j], "link", False)) or \
                    (getattr(specs[i], "link", False) == getattr(specs[j], "link", False) and wi < wj - 0.5)
                if not branch or opposite(i, e, j):
                    continue
                # the part it runs beside from the fork (of the parts at it: the nearest ~15 m along)
                a = 15.0 if e == 0 else arcs[i][-1] - 15.0
                q = np.array([np.interp(a, arcs[i], pts[i][:, 0]), np.interp(a, arcs[i], pts[i][:, 1])])
                d, _ = project(j, q[None])
                if best is None or d < best[0]:
                    best = (d, j, wj)
            if best is None:
                continue
            _, j, wj = best
            # (its other end an abutment or a joint: held no nearer it than its ease back to its own profile takes,
            # its own height there kept. A ramp beside its road the whole way from the fork to its abutment had been
            # held at the road's height to its very start, 4.9 m over the ground road going on from it (MLK
            # Boulevard by I-395, Baltimore); another to its joint with the next part, a 6.9 m step there)
            far = fixed(i, -1 - e, j)
            order = range(len(pts[i])) if e == 0 else range(len(pts[i]) - 1, -1, -1)
            s0 = arcs[i][0 if e == 0 else -1]
            sep, delta = None, 0.0
            s_far = arcs[i][-1 if e == 0 else 0]
            for k in order:
                d, zj = project(j, pts[i][k][None])
                if d - (wi + wj) / 2 >= TIE_JOIN_M:
                    sep = k
                    break
                if abs(zj - zs[i][k]) > (TIE_MEASURED_M if measured is not None and measured[i]
                                         else 2.0 + abs(arcs[i][k] - s0) * 0.1):
                    sep = k                             # (another level: stacked, not side by side)
                    break
                if far and abs(s_far - arcs[i][k]) < max(TIE_EASE_MIN_M, abs(zj - own[i][k]) / TIE_EASE_GRADE):
                    sep = k
                    break
                delta = zj - zs[i][k]
                zs[i][k] = zj
            if sep is None or abs(delta) < 1e-3:
                continue
            ss = arcs[i][sep]
            span = max(TIE_EASE_MIN_M, abs(delta) / TIE_EASE_GRADE)
            if far:
                span = min(span, max(abs(s_far - ss), 1e-3))
            for k in order:
                along = abs(arcs[i][k] - ss)
                if (e == 0 and k < sep) or (e == -1 and k > sep):
                    continue
                if along >= span:
                    break
                zs[i][k] += delta * (1.0 - along / span)
    return zs


def _joined_lines(lines, joins):
    """lines with each end of joins moved onto its joint (copies; the others as they are)."""
    out = list(lines)
    for k, at, p in joins:
        ln = np.array(out[k], np.float64, copy=True)
        ln[at, 0], ln[at, 1] = p
        out[k] = ln
    return out


def _ground_pins(lines, ground_lines):
    """The joints of decks (lines' ends shared by two or more) a ground road of PIN_ROAD_M or more goes on from (an
    end of it within PIN_M): _pin's points (earthfast.deck_ground_pins: every end against every end, then each road's
    two ends against the joints, was ~30 ms a Manhattan chain solve in numpy). Pure."""
    ends = np.array([p[:2] for ln in lines for p in (ln[0], ln[-1])], np.float64).reshape(-1, 2)
    if not len(ends) or not ground_lines:
        return []
    import fast
    return list(fast.deck_ground_pins(ends, ground_lines, PIN_ROAD_M, PIN_M))


def _merge_overlaps(lines, zs, widths, levels, rounds=3, pins=None):
    """Heights of decks lying over each other (a ramp leaving the mainline, parallel carriageways of
    one level) pulled to their mean where their surfaces overlap, so a split is one surface until the
    decks part (2026-10-11 the user: exits and splits not clean). Parts are related when they share an
    end node or a level and run within ~30 degrees of each other. Abutment ends stay (pins: deck_profile's). Pure."""
    nodes, edges, adj = _chain_graph(lines)
    deg = np.zeros(len(nodes), int)
    for a, b, _ in edges:
        deg[a] += 1
        deg[b] += 1
    deg = _pin(nodes, deg, pins)
    n = len(lines)
    pairs = []
    # the related pairs (i, j), i then j ascending, as a double loop testing each would find them, one line against
    # all others at a time (that loop was ~4 s of a Fells Point teleport, on the event loop)
    if n:
        ea = np.array([e[0] for e in edges[:n]])
        eb = np.array([e[1] for e in edges[:n]])
        lo = np.array([np.asarray(ln)[:, :2].min(0) for ln in lines])
        hi = np.array([np.asarray(ln)[:, :2].max(0) for ln in lines])
        lv = None if levels is None else np.empty(n, object)
        if lv is not None:
            lv[:] = list(levels)[:n]
        idx = np.arange(n)
        for i in range(n):
            ok = (ea[i] == ea) | (ea[i] == eb) | (eb[i] == ea) | (eb[i] == eb)
            if lv is not None:
                ok |= ~(lv != lv[i]).astype(bool)          # elementwise, the same != as levels[i] != levels[j]
            ok &= ~(hi[i] < lo - 30).any(1) & ~(hi < lo[i] - 30).any(1)
            ok &= idx != i
            pairs.extend((i, int(j)) for j in np.flatnonzero(ok))
    if not pairs:
        return zs
    floors = [np.array(z, np.float64) for z in zs]
    zs = [z.copy() for z in floors]
    # never raised faster than DECK_MAX_GRADE from any abutment of the chain, along it (a lift at a joint reached the
    # parts beyond it: Spaghetti Junction's ramps, lifted to a higher deck of their level beside them, stood 21 m up at
    # the joint of a 48 m part ending on the ground, 40 %)
    reach = [np.full(len(z), np.inf) for z in zs]
    for k0, (a0, b0, _s) in enumerate(edges[:n]):
        for node, zend in ((a0, floors[k0][0]), (b0, floors[k0][-1])):
            if deg[node] != 1 or not np.isfinite(zend):
                continue
            d = _dijkstra(adj, {node: 0.0})
            for k, (ea, eb, es) in enumerate(edges[:n]):
                reach[k] = np.minimum(reach[k], zend + DECK_MAX_GRADE * np.minimum(d[ea] + es, d[eb] + (es[-1] - es)))
    import fast
    merge = fast.DeckMerge(lines, widths, pairs, [levels is not None and levels[i] == levels[j] for i, j in pairs],
                           MERGE_TOUCH_M, DUAL_DECK_M)
    for _ in range(rounds):
        new = [z.copy() for z in zs]
        # (every pair's points projected onto the other's line, weighted by overlap and parallelism, in Rust:
        # earthfast.deck_merge_round, 0.75 s of a cold Manhattan road tile in numpy, 2026-11-08)
        acc, cnt, up = merge.round(zs)
        for i in range(n):
            m = cnt[i] > 0
            new[i][m] = (zs[i][m] + acc[i][m]) / (1 + cnt[i][m])
            new[i] = np.maximum(new[i], zs[i] + up[i])
            a, b, es = edges[i]
            # a lift fades along the deck no faster than DECK_MAX_GRADE: it ended where the parts drew apart, over a
            # few metres of their spacing (a High Five ramp lifted 11 m to its twin fell back over ~60 m, 35 %)
            lifted = new[i] > zs[i] + 1e-6
            if lifted.any():
                new[i] = np.maximum(new[i], _ground_envelope(es, np.where(lifted, new[i], -np.inf), DECK_MAX_GRADE))
            # never raised faster than DECK_MAX_GRADE from an abutment (a road's bridge forking off uphill near one,
            # taken for a carriageway beside it, pulled US-24's deck 1.1 m up in the first 3 m off its abutment near
            # Buena Vista CO, 2026-10-27)
            # (fading out toward a joint with other parts, where the parts must meet: capped right up to it, an 8 m
            # ramp part of a Calgary diverging diamond stood 0.7 m under the joint it leaves, #369)
            rise = np.full(len(es), np.inf)
            to_joint = np.full(len(es), np.inf)
            for end, d, at in ((a, es, 0), (b, es[-1] - es, -1)):
                if deg[end] == 1:
                    rise = np.minimum(rise, floors[i][at] + DECK_MAX_GRADE * d)
                else:
                    to_joint = np.minimum(to_joint, d)
            capped = np.minimum(new[i], np.maximum(floors[i], rise))     # (from the unmerged heights: no creep per round)
            if np.isfinite(to_joint).any() and np.isfinite(rise).any():
                w = np.clip(to_joint / max(float(es[-1]), 1e-6), 0.0, 1.0)
                capped = w * capped + (1.0 - w) * new[i]
            new[i] = np.minimum(capped, np.maximum(reach[i], floors[i]))
            if deg[a] == 1:
                new[i][0] = zs[i][0]
            if deg[b] == 1:
                new[i][-1] = zs[i][-1]
        zs = new
    out = []
    for i, z in enumerate(zs):                  # re-smoothed, ends kept
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(lines[i][:, :2], axis=0).T))]
        sm = _smooth_pinned(z, s[-1] / max(len(s) - 1, 1), DECK_SIGMA) if len(z) > 3 else z
        out.append(np.maximum(sm, floors[i]))
    return out


def deck_profile(lines, ground_fn, crossings=(), clear=None, grade=None, lift=None, levels=None, widths=None, pins=None):
    """deck_profile_once, then repeated with the deck-over-deck requirements: where two parts of
    different levels cross, the higher one clears the lower one's deck by `clear` (2026-10-11: crossing
    decks clipped into each other). pins: points where a ground road goes on from a joint of decks (abutments too: _pin).
    Pure given ground_fn."""
    zs = deck_profile_once(lines, ground_fn, crossings, clear, grade, lift, pins=pins)
    n_road = len(crossings)
    if levels is None or len(lines) < 2:
        return zs
    over = _deck_crossings(lines)
    stacks = _deck_stacks(lines, levels, widths) if widths is not None else []
    if not over and not stacks:
        zs = _merge_overlaps(lines, zs, widths, levels, pins=pins) if widths is not None else zs
        return _ease(lines, zs, ground_fn, lift) if bridge_profile_mode() == "curve" else zs
    arcs = [np.r_[0.0, np.cumsum(np.hypot(*np.diff(ln[:, :2], axis=0).T))] for ln in lines]
    # curve mode: one round. Its grades carry a requirement along the line to the parts joined to it whatever their
    # level, so a ramp under the upper deck rose with it and lifted it again each round (the Morrison Bridge 21 ->
    # 41 m over the Willamette, 2026-10-19)
    for _ in range(1 if bridge_profile_mode() == "curve" else 4):
        extra = []
        for i, j, si, sj in over:
            if levels[i] == levels[j]:
                continue
            hi, lo, sh, sl = (i, j, si, sj) if levels[i] > levels[j] else (j, i, sj, si)
            extra.append((hi, sh, float(np.interp(sl, arcs[lo], zs[lo])), None, levels[hi]))
        for hi, sh, lo, sl in stacks:               # a deck over another along it (an upper roadway)
            extra.append((hi, sh, float(np.interp(sl, arcs[lo], zs[lo])), None, levels[hi]))
        if not extra:
            break
        new = deck_profile_once(lines, ground_fn, list(crossings) + extra, clear, grade, lift, n_road=n_road,
                                levels=levels, pins=pins)
        if all(np.allclose(a, b, atol=0.02, equal_nan=True) for a, b in zip(new, zs)):
            zs = new
            break
        zs = new
    zs = _merge_overlaps(lines, zs, widths, levels, pins=pins) if widths is not None else zs
    return _ease(lines, zs, ground_fn, lift) if bridge_profile_mode() == "curve" else zs


def _deck_stacks(lines, levels, widths, step=STACK_STEP_M):
    """(upper part, arc on it, lower part, arc on that) every `step` m where a part of a higher level runs over one
    of a lower level, their surfaces overlapping in plan: the Queensboro Bridge's upper and lower roadways lay at
    one height, one road through the other (the user's report, 2026-10-17); _deck_crossings only finds crossings.
    Pure."""
    out = []
    n = len(lines)
    if levels is None or n < 2:
        return out
    pts = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    arcs = [np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))] for p in pts]
    lo_b = [p.min(0) for p in pts]
    hi_b = [p.max(0) for p in pts]
    for i in range(n):
        for j in range(n):
            if levels[i] <= levels[j] or len(pts[i]) < 2 or len(pts[j]) < 2:
                continue
            reach = (float(widths[i]) + float(widths[j])) / 2
            if (hi_b[i] + reach < lo_b[j]).any() or (hi_b[j] + reach < lo_b[i]).any():
                continue
            # over it means over at least half the narrower deck: two carriageways side by side that the map gives
            # different layers touch edge to edge (the Vincent Thomas Bridge's, 8.5 m apart, 11.8 m wide: the upper
            # was lifted 10 m over its twin, 2026-10-27)
            reach -= 0.5 * min(float(widths[i]), float(widths[j]))
            s = np.arange(0.0, arcs[i][-1], step)
            q = np.c_[np.interp(s, arcs[i], pts[i][:, 0]), np.interp(s, arcs[i], pts[i][:, 1])]
            P, ab = pts[j][:-1], np.diff(pts[j], axis=0)
            L2 = np.maximum((ab * ab).sum(1), 1e-12)
            t = np.clip(((q[:, None, :] - P[None]) * ab[None]).sum(2) / L2[None], 0.0, 1.0)
            near = P[None] + t[..., None] * ab[None]
            d = np.hypot(*(q[:, None, :] - near).transpose(2, 0, 1))
            m = np.argmin(d, axis=1)
            dd = d[np.arange(len(q)), m]
            # (two parts joined at an end: a ramp leaving the mainline overlaps it from the fork on. A requirement there
            # falls back to the shared end at DECK_GRADE and lifted the upper part's end alone: a ramp of a higher
            # level started 2.2 m over the fork, a ledge across the lanes, I-83 in Baltimore 2026-10-21. None within
            # the distance it takes to fall away.)
            shared = [ei for ei in (0, -1) for ej in (0, -1) if np.hypot(*(pts[i][ei] - pts[j][ej])) < 0.5]
            for a in np.flatnonzero(dd < reach):
                k = m[a]
                sl = arcs[j][k] + t[a, k] * (arcs[j][k + 1] - arcs[j][k])
                if any((s[a] if ei == 0 else arcs[i][-1] - s[a]) * DECK_GRADE < DECK_ROAD_CLEAR for ei in shared):
                    continue
                out.append((i, float(s[a]), j, float(sl)))
    return out


def _deck_beside(lines, widths, step=STACK_STEP_M):
    """(i, j), i < j: deck parts whose surfaces come within MERGE_TOUCH_M of each other in plan (side by side or one
    over the other: solved together): 2 or more of part i's points every `step` m along it within half their widths
    + MERGE_TOUCH_M of part j's line. earthfast.deck_beside (Rust) on the samples. Pure."""
    import fast
    fast.require(89, "deck_beside")
    pts = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    qs = []
    for p in pts:
        if len(p) < 2:
            qs.append(np.zeros((0, 2)))
            continue
        si = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        t = np.arange(0.0, si[-1] + 1e-9, step)
        qs.append(np.c_[np.interp(t, si, p[:, 0]), np.interp(t, si, p[:, 1])])
    off = lambda arrs: np.r_[0, np.cumsum([len(a) for a in arrs])].astype(np.int64)      # noqa: E731
    pairs = fast._ef.deck_beside(np.ascontiguousarray(np.concatenate(qs) if qs else np.zeros((0, 2))), off(qs),
                                 np.ascontiguousarray(np.concatenate(pts) if pts else np.zeros((0, 2))), off(pts),
                                 np.asarray(widths, np.float64), float(MERGE_TOUCH_M))
    return [(int(a), int(b)) for a, b in pairs]


def water_clear(w):
    """m a deck clears the water under it for a wet run w m long: a share of the span (a river's channel wants a
    ship's room: the Queensboro Bridge's ~40 m over 250 m channels), at least WATER_CLEAR_MIN; a run longer than
    WATER_CAUSEWAY_M is a causeway, low over the water. Pure."""
    if w > WATER_CAUSEWAY_M:
        return WATER_CAUSEWAY_CLEAR
    return float(np.clip(WATER_CLEAR_SHARE * w, WATER_CLEAR_MIN, WATER_CLEAR_MAX))


def _curve_grade(lines, zs):
    """The steepest grade (rise / run) between consecutive points of any part of a deck profile. Pure."""
    g = 0.0
    for ln, z in zip(lines, zs):
        if len(ln) < 2:
            continue
        run = np.maximum(np.hypot(*np.diff(np.asarray(ln, np.float64)[:, :2], axis=0).T), 0.5)
        g = max(g, float(np.max(np.abs(np.diff(np.asarray(z, np.float64))) / run)))
    return g


def _end_step(lines, zs):
    """The steepest grade of a deck's first or last segment at a free end (an abutment: no other part ends within
    0.75 m), where a curve that cannot meet its end leaves a step. Pure."""
    ends = [np.asarray(ln, np.float64)[k, :2] for ln in lines for k in (0, -1)]
    g = 0.0
    for ln, z in zip(lines, zs):
        ln = np.asarray(ln, np.float64)
        if len(ln) < 2:
            continue
        for a, b in ((0, 1), (-1, -2)):
            if sum(np.hypot(*(e - ln[a, :2])) < 0.75 for e in ends) > 1:
                continue
            run = max(float(np.hypot(*(ln[b, :2] - ln[a, :2]))), 0.5)
            g = max(g, abs(float(z[b]) - float(z[a])) / run)
    return g


def _curve_crossings(crossings, n_road=None, levels=None):
    """deck_profile_once's crossings for deck_curve: (part, arc, height under it, its clearance or None). A
    deck-over-deck requirement holds on the upper part itself (deck_curve applies a requirement where it is, not
    falling along the chain, so it never reaches the lower deck). Pure."""
    return [(cr[0], cr[1], cr[2], cr[3] if len(cr) > 3 else None) for cr in crossings]


def deck_profile_once(lines, ground_fn, crossings=(), clear=None, grade=None, lift=None, n_road=None, levels=None,
                      pins=None, curve=True):
    """Deck heights (one array per part) of a bridge chain: deck_heights between the abutments, at
    least `lift` over the ground under it, and at least `clear` over each crossing road below
    ((part index, arc m, road height)), the requirement falling at `grade` along the chain; then
    smoothed per part with its end heights kept. A road crossing (the first n_road crossings, default all, without
    their own clearance) by an abutment the deck cannot get AT_GRADE_M over is a junction at grade: the deck
    meets the road's height there. Pure given ground_fn(xy (m, 2)) -> heights."""
    clear = DECK_ROAD_CLEAR if clear is None else clear
    grade = DECK_GRADE if grade is None else grade
    lift = DECK_CLEAR if lift is None else lift
    # (curve mode for a deck that clears something; one over nothing but the ground keeps the ground envelope's rounded
    # crest: Lombard Street, Baltimore's deck over its embankment, test_no_step_onto_a_steep_abutment)
    if curve and lines and bridge_profile_mode() == "curve" and len(crossings):
        got = deck_curve(lines, ground_fn, _curve_crossings(crossings, n_road, levels), (), clear, lift, pins=pins)
        if got is not None:
            g = _curve_grade(lines, got)
            if g <= DECK_STEEP_GRADE or _end_step(lines, got) <= DECK_STEEP_GRADE:
                return got
            # (a curve steeper than that may not meet its ends: a "bridge" mapped along a hillside stepped 40 m at
            # its abutment, GitHub #382 near Tata. The ground envelope's profile when it is the gentler one)
            old = deck_profile_once(lines, ground_fn, crossings, clear, grade, lift, n_road, levels, pins, False)
            return got if not old or _end_step(lines, got) <= _end_step(lines, old) + 0.05 else old
    zs = deck_heights(lines, ground_fn, pins)
    if not zs:
        return zs
    nodes, edges, adj = _chain_graph(lines)
    allxy = np.concatenate([ln[:, :2] for ln in lines])
    gz = np.asarray(ground_fn(allxy), np.float64)
    deg = np.zeros(len(nodes), int)
    for a, b, _ in edges:
        deg[a] += 1
        deg[b] += 1
    deg = _pin(nodes, deg, pins)
    # no steeper than DECK_MAX_GRADE from an abutment: a short bridge over a road cannot reach full
    # headroom (2026-10-11: such decks became steep wedges, "walls" in the user's screenshot)
    cap = [np.full(len(ln), np.inf) for ln in lines]
    dab = [np.full(len(ln), np.inf) for ln in lines]      # m to the nearest abutment
    steep = [np.full(len(ln), np.inf) for ln in lines]    # DECK_STEEP_GRADE off each abutment (below)
    for k, (a, b, s) in enumerate(edges):
        for node, zend in ((a, zs[k][0]), (b, zs[k][-1])):
            if deg[node] != 1 or not np.isfinite(zend):
                continue
            d = _dijkstra(adj, {node: 0.0})
            for k2, (ea, eb, es) in enumerate(edges):
                dk = np.minimum(d[ea] + es, d[eb] + (es[-1] - es))
                cap[k2] = np.minimum(cap[k2], zend + DECK_MAX_GRADE * dk)
                dab[k2] = np.minimum(dab[k2], dk)
                steep[k2] = np.minimum(steep[k2], zend + DECK_STEEP_GRADE * dk)
    n_road = len(crossings) if n_road is None else n_road
    req = [np.full(len(ln), -np.inf) for ln in lines]
    for ci, cr in enumerate(crossings):     # (part, arc, height under it[, its own clearance: over water])
        pi, sc, zc = cr[:3]
        cl = cr[3] if len(cr) > 3 and cr[3] is not None else clear
        # a deck-over-deck requirement (its 5th element: the upper part's level) lifts the parts of that level and
        # above only: falling along the chain it reached the lower deck it was measured on, lifting it, and so the
        # upper again each round (the Tsing Ma Bridge's upper deck 33 m over its lower one, 7 m in life, 2026-10-18)
        min_lv = cr[4] if len(cr) > 4 and levels is not None else None
        a, b, s = edges[pi]
        d = _dijkstra(adj, {a: sc, b: s[-1] - sc})
        # a junction at grade (Queens Plaza, the user's report #55, 2026-10-17: a ramp's map line starts in the
        # intersection it climbs from; its deck rose 0.8 m over the cross street there, a hump across it)
        flat = (ci < n_road and len(cr) == 3 and float(np.interp(sc, s, dab[pi])) <= AT_GRADE_REACH_M
                and float(np.interp(sc, s, cap[pi])) < zc + AT_GRADE_M)
        for k, (ea, eb, es) in enumerate(edges):
            dk = np.minimum(d[ea] + es, d[eb] + (es[-1] - es))
            if k == pi:
                dk = np.minimum(dk, np.abs(es - sc))
            if flat:
                cap[k] = np.minimum(cap[k], zc + DECK_MAX_GRADE * dk)
            elif min_lv is None or levels[k] >= min_lv:
                # A lower road has width: keep a level clearance corridor around its
                # centreline instead of reaching the target at just one point.
                reach = DECK_CLEAR_REACH_M if len(cr) == 3 or cr[3] is None else 0.0
                req[k] = np.maximum(req[k], zc + cl - grade * np.maximum(dk - reach, 0.0))
                if k == pi:
                    # A crossing can lie between mesh vertices. Both bracketing vertices
                    # need full height, otherwise linear interpolation loses headroom.
                    for point in (sc - reach, sc, sc + reach):
                        if not 0 <= point <= es[-1]:
                            continue
                        j = int(np.searchsorted(es, point, side="left"))
                        near = [j] if j < len(es) and abs(es[j] - point) < 1e-8 else [j - 1, j]
                        for at in near:
                            if 0 <= at < len(es):
                                req[k][at] = max(req[k][at], zc + cl)
    # the caps (from an abutment, from a junction at grade) never under the straight line between the abutments: two
    # abutments further apart in height than DECK_MAX_GRADE allows had the higher one pulled under its own road, a
    # step onto the deck (0.9 m up onto the I-20 bridge near Temple, GA, once its approaches were raised, 2026-10-20)
    cap = [np.where(np.isfinite(z), np.maximum(c, z), c) for c, z in zip(cap, zs)]
    out, k0 = [], 0
    for k, (z, (a, b, s)) in enumerate(zip(zs, edges)):
        # Meet the graded approach exactly. A constant skin allowance right up
        # to a pinned abutment otherwise restores a small lip after smoothing.
        blend = np.ones(len(z))
        for distance in ([s] if deg[a] == 1 else []) + ([s[-1] - s] if deg[b] == 1 else []):
            u = np.clip(distance / DECK_SIGMA, 0.0, 1.0)
            blend = np.minimum(blend, u * u * (3.0 - 2.0 * u))
        g = gz[k0:k0 + len(z)] + lift * blend
        k0 += len(z)
        if not np.isfinite(z).all():
            out.append(z)
            continue
        # ground standing over a cap raises it, falling at DECK_MAX_GRADE either way: capped from its abutment, a deck
        # met a steep embankment's ground in a step where the ground took over (0.86 m in 4 m onto Lombard Street's
        # east abutment in Baltimore, the bridge's map line ending down the embankment; cars stopped dead, 2026-10-20)
        # the floor: the ground rounded over its peaks (a vertical curve of DECK_CREST_R), so a deck smoothed under a
        # sharp peak and floored there has no spike
        g = np.maximum(g, _ground_envelope(s, g, 0.0, 1.0 / (2.0 * DECK_CREST_R)))
        cap[k] = np.maximum(cap[k], _ground_envelope(s, g, DECK_MAX_GRADE))
        # ... but never steeper than DECK_STEEP_GRADE off an abutment where the ground under it is lower (nor under
        # the straight line between the abutments): raised by the ground 90 m off at the far, 10 m higher abutment, a
        # water clearance stood the New York Thruway's deck over the Vloman Kill 5 m up in its first 3.6 m, a wall an
        # AI car hit at 65 mph (2026-10-27)
        cap[k] = np.minimum(cap[k], np.maximum(np.maximum(steep[k], g), np.where(np.isfinite(z), z, -np.inf)))
        zz = np.maximum(np.minimum(np.maximum(z, req[k]), cap[k]), g)
        if deg[a] == 1:                         # an abutment keeps the ground's height
            zz[0] = z[0]
        if deg[b] == 1:
            zz[-1] = z[-1]
        if len(zz) > 3:
            # no sag between two higher points (the Queensboro Bridge dipped 7 m over Roosevelt Island between its
            # channel spans: one arch in life, 2026-10-18), within the caps
            zz = np.maximum(np.minimum(_no_sag(s, zz), np.maximum(cap[k], zz)), zz)
            step = s[-1] / (len(s) - 1)
            sigma = float(np.clip(s[-1] / DECK_SIGMA_SHARE, DECK_SIGMA, DECK_SIGMA_MAX))   # a long deck: a long curve
            sm = np.minimum(_smooth_pinned(zz, step, sigma), cap[k])   # no hump over an at-grade junction
            # Smoothing is aesthetic: it must not remove feasible structural headroom.
            # Keep the abutment/grade cap, rather than turning a short bridge into a wall.
            floor = np.maximum(np.minimum(req[k], cap[k]), g)
            zz = np.maximum(sm, floor)
            zz[0], zz[-1] = sm[0], sm[-1]
            zz = _curve_deck(s, zz, floor, cap[k])
        out.append(zz)
    return out


def bridge_profile_mode():
    """'curve' (default) or 'legacy', from BNE_BRIDGE_PROFILE (set by earthd --bridge-profile)."""
    return "legacy" if os.environ.get("BNE_BRIDGE_PROFILE", "curve").lower() == "legacy" else "curve"


def _upper_hull(s, z):
    """Indices of the least concave majorant's vertices of points (s ascending, z). Pure."""
    h = []
    for i in range(len(s)):
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (s[b] - s[a]) * (z[i] - z[a]) - (z[b] - z[a]) * (s[i] - s[a]) >= 0:
                h.pop()
            else:
                break
        h.append(i)
    return h


def _simple_path(lines, graph=None):
    """[(edge index, reversed)] in order along the chain if the parts form one open path, else None."""
    nodes, edges, adj = _chain_graph(lines) if graph is None else graph
    deg = np.zeros(len(nodes), int)
    for a, b, _ in edges:
        deg[a] += 1
        deg[b] += 1
    ends = np.flatnonzero(deg == 1)
    if len(ends) != 2 or deg.max() > 2 or len(edges) != len(nodes) - 1:
        return None
    at, used, order = int(ends[0]), set(), []
    while len(order) < len(edges):
        for k, (a, b, _) in enumerate(edges):
            if k in used or at not in (a, b):
                continue
            used.add(k)
            order.append((k, at == b and a != b))
            at = a if at == b else b
            break
        else:
            return None
    return order


def _chord(s, g, need, z0, zL, g0, gL, max_grade, cap=None):
    """curve_profile's straight line or single parabola between the abutments (see there), or None. Pure."""
    L = float(s[-1])
    if L <= 0:
        return None
    t = s / L
    line = z0 + (zL - z0) * t
    gl = (zL - z0) / L
    if abs(gl) > max_grade:
        return None
    p = 4.0 * t * (1.0 - t)
    # (where the ground stands within CHORD_SHELF_M of it, an abutment's shelf, nothing under it is to be cleared: the
    # crossing's +-DECK_CLEAR_REACH_M corridor reached over the abutments of a 24 m overpass)
    on = g >= line - CHORD_SHELF_M
    shelf = (np.cumsum(~on) == 0) | (np.cumsum((~on)[::-1])[::-1] == 0)     # (from each end, while on it)
    if shelf.all():
        shelf[:] = False
    short = np.maximum(np.where(shelf, -np.inf, need - CHORD_TOL - line), g - line)
    h = 0.0
    inner = p > 1e-6
    if (short[~inner] > 1e-6).any():
        return None
    if (short[inner] > 0).any():
        h = float(np.max(short[inner] / p[inner]))
    # its grade out of each abutment, against the approach's there
    if abs(gl + 4.0 * h / L - g0) > CHORD_KINK or abs(gl - 4.0 * h / L - gL) > CHORD_KINK             or 4.0 * h / L > CHORD_KINK or abs(gl) + 4.0 * h / L > max_grade:
        return None
    z = line + h * p
    if cap is not None and not bool(np.all(z <= np.asarray(cap) + 1e-6)):
        return None
    return z


def curve_profile(s, g, req, z0, zL, g0=0.0, gL=0.0, max_grade=None, lift=None, iters=20, cap=None):
    """One deck line's heights on a uniform grid s (0..L): from z0 to zL (the abutments), meeting the
    approach grades g0 (dz/ds just before s=0) and gL (just after L); at least req (the clearances) where
    the grade cap lets it, reached and left at DECK_DESIGN_GRADE (no steeper than max_grade from either end), joined by parabolic
    vertical curves; never below g + lift nor the straight line between the abutments; under cap (m, per point: a
    junction at grade's) where given. Pure."""
    max_grade = DECK_MAX_GRADE if max_grade is None else max_grade
    lift = DECK_CLEAR if lift is None else lift
    s = np.asarray(s, np.float64)
    L = s[-1]
    floor = np.asarray(g, np.float64) + lift
    req = np.maximum(np.asarray(req, np.float64), floor)
    cap = np.minimum(np.minimum(z0 + max_grade * s, zL + max_grade * (L - s)),
                     np.inf if cap is None else cap)        # (and under a junction at grade's own cap)
    line = z0 + (zL - z0) * s / max(L, 1e-9)    # never under the straight line between the abutments
    # the straight line between the abutments, or one parabola over it rising at most CHORD_KINK more steeply out of
    # each abutment, when that clears everything (the requirements within CHORD_TOL: they carry 0.21 m of margin;
    # the ground itself, which it may rest on) and meets both approaches within CHORD_KINK. The curve's floor stood
    # DECK_CLEAR over the approach ground beside each abutment and its smoothing crowned a 24 m overpass by 0.2-0.3 m
    # (I-275 by the Sunshine Skyway, 2026-10-27: "it just has a HUMP in the middle that's not necessary")
    chord = _chord(s, np.asarray(g, np.float64), np.asarray(req, np.float64) - lift, z0, zL, g0, gL, max_grade, cap)
    if chord is not None:
        return chord
    want = np.minimum(np.maximum(req, line), cap)
    want[0], want[-1] = z0, zL
    step = L / max(len(s) - 1, 1)
    # every curve length from long to the shortest allowed: of those holding the clearance (within 5 cm)
    # the one that rises least above what is needed (a long curve over a short clearance stretch had to
    # be lifted 14 m over the Fremont Bridge's 58 m), else the one falling least short
    best = None
    for kf in (1.0, 0.5, 0.25, 0.12, 0.067):
        z = _curve_try(s, want, cap, z0, zL, g0, gL, step, L, DECK_CURVE_K * kf, iters)
        short = float(np.max(want - z))
        over = float(np.max(z) - np.max(want))
        key = (0, over) if short <= 0.05 else (1, short)
        if best is None or key < best[0]:
            best = (key, z)
    z = best[1].copy()
    return _lift_smooth(s, z, floor)                # the abutments stay on the approach road


def _tents(s, z, grade):
    """max over points i of z_i - grade * |s - s_i|: every requirement reached at `grade` and left at it
    (the deck rides low where nothing is under it, then climbs to what it must clear; earthfast.deck_tents). Pure."""
    import fast
    fast.require(141, "deck_tents")
    return fast._ef.deck_tents(np.ascontiguousarray(s, np.float64), np.ascontiguousarray(z, np.float64), float(grade))


def _curve_try(s, want, cap, z0, zL, g0, gL, step, L, K, iters):
    raise_ = np.zeros(len(s))
    z = want
    for _ in range(iters):
        pts = want + raise_
        pts[0], pts[-1] = z0, zL
        hull = np.minimum(_tents(s, pts, DECK_DESIGN_GRADE), cap)
        gr = np.diff(hull) / np.maximum(np.diff(s), 1e-9)
        # the whole crest's change (close hull corners act as one curve), or an end's
        A = max(float(np.max(gr) - np.min(gr)), abs(gr[0] - g0), abs(gr[-1] - gL))
        W = float(np.clip(K * A, max(DECK_CURVE_MIN, DECK_CURVE_KMIN * A), max(DECK_CURVE_MIN, 0.8 * L)))
        n = max(int(round(W / step)) | 1, 1)
        r = n // 2
        ext = np.r_[z0 - g0 * step * np.arange(r, 0, -1), hull, zL + gL * step * np.arange(1, r + 1)]
        sm = np.convolve(ext, np.ones(n) / n, mode="valid") if r else hull.copy()
        sm = sm + (z0 - sm[0]) + ((zL - sm[-1]) - (z0 - sm[0])) * (s / max(L, 1e-9))
        z = sm
        short = want - z
        short[0] = short[-1] = 0.0
        if short.max() <= 0.02:
            break
        # (not where the grade cap holds want down: no raise reaches it there, and raising it again and again
        # tented a 5 m hump past the cap onto the Vincent Thomas Bridge's span, 2026-10-28)
        raise_ += np.maximum(short, 0.0)
    return z



def _path_curve(lines, order, ground_fn, crossings, spans, clear, lift, max_grade, ends=(None, None)):
    """curve_profile along one path of parts (order: [(part, reversed)]). ends: per end None (an abutment:
    the ground's height, the approach grade from the ground beyond it) or (z, grade dz/ds arriving) (a
    junction with a part already solved). Returns {part: heights}, the path's polyline (m, 2), its grid s,
    heights z. None if degenerate."""
    pts, off, info = [], 0.0, {}
    for k, rev in order:
        ln = np.asarray(lines[k], np.float64)[:, :2]
        a = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ln, axis=0).T))]
        info[k] = (off, rev, a)
        seq = ln[::-1] if rev else ln
        pts.append(seq if not pts else seq[1:])
        off += a[-1]
    P = np.concatenate(pts)
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    L = S[-1]
    if L < 1.0 or len(P) < 2:
        return None
    # A rotation can put an integral length a few ulps above its metre station.
    n = max(int(math.ceil(L / DECK_CURVE_STEP - 1e-9)), 2)
    s = np.linspace(0.0, L, n + 1)
    xy = np.c_[np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1])]

    def to_s(k, arc):
        o, rev, a = info[k]
        return o + (a[-1] - arc if rev else arc)
    t0 = (P[0] - P[1]) / max(np.hypot(*(P[0] - P[1])), 1e-9)
    t1 = (P[-1] - P[-2]) / max(np.hypot(*(P[-1] - P[-2])), 1e-9)
    out_pts = np.array([P[0] + t0 * 12.0, P[-1] + t1 * 12.0])
    gz = np.asarray(ground_fn(np.r_[xy, out_pts]), np.float64)
    g, go = gz[:len(s)], gz[len(s):]
    if ends[0] is None:
        z0, g0 = float(g[0]), float(np.clip((g[0] - go[0]) / 12.0, -APPROACH_GRADE, APPROACH_GRADE))
    else:
        z0, g0 = ends[0]
    if ends[1] is None:
        zL, gL = float(g[-1]), float(np.clip((go[1] - g[-1]) / 12.0, -APPROACH_GRADE, APPROACH_GRADE))
    else:
        zL, gL = ends[1]
    req = g + lift
    junction = np.full(len(s), np.inf)
    for cr in crossings:
        k, arc, zc = cr[:3]
        own = len(cr) > 3 and cr[3] is not None
        cl = cr[3] if own else clear
        if k in info:
            sc = to_s(k, arc)
            # a road by an abutment the deck cannot get AT_GRADE_M over is a junction at grade (the intersection a
            # ramp climbs from: deck_profile_once's rule): no requirement, the deck leaves the road's height
            d_end = min(sc if ends[0] is None else np.inf, L - sc if ends[1] is None else np.inf)
            if not own and d_end <= AT_GRADE_REACH_M and                     min(z0, zL) + DECK_MAX_GRADE * d_end < zc + AT_GRADE_M:
                junction = np.minimum(junction, zc + DECK_MAX_GRADE * np.abs(s - sc))
                continue
            # (upstream's full-headroom corridor, DECK_CLEAR_REACH_M either side, and the curve's 5 cm acceptance
            # taken up front: the corridor came out 4 cm short, test_bridge_clearance)
            m = np.abs(s - sc) <= max(DECK_ROAD_HALF, DECK_CLEAR_REACH_M if not own else 0.0) + 1e-8
            req[m] = np.maximum(req[m], zc + cl + (0.06 if not own else 0.0))
    for k, a0, a1, zmin in spans:
        if k in info:
            x0, x1 = sorted((to_s(k, a0), to_s(k, a1)))
            m = (s >= x0 - 1e-8) & (s <= x1 + 1e-8)
            req[m] = np.maximum(req[m], zmin)
    z = curve_profile(s, g, req, z0, zL, g0, gL, max_grade, lift, cap=junction)
    out = {}
    for k, rev in order:
        o, r, a = info[k]
        out[k] = np.interp(o + (a[-1] - a if r else a), s, z)
    return out, xy, s, z


def deck_curve(lines, ground_fn, crossings=(), spans=(), clear=None, lift=None, max_grade=None, pins=None):
    """Deck heights (one array per part) of a bridge chain by curve_profile. A chain of parts in one line is
    solved as one profile; a branched chain (ramps leaving a viaduct) as its longest abutment-to-abutment
    line first, then each remaining branch from the solved deck it leaves (its height and grade there) to
    its own abutment or the deck it rejoins. None when the chain has no abutment (a closed loop). crossings:
    (part, arc m, height under it[, its own clearance]) roads and water under it (held `clear`, or their own
    clearance, over +-DECK_ROAD_HALF); spans: (part, arc from, arc to, min deck height). pins: joints at ground
    roads, split into independent abutments in the graph without moving the mapped geometry. Pure given ground_fn."""
    clear = DECK_ROAD_CLEAR if clear is None else clear
    lift = DECK_CLEAR if lift is None else lift
    if not lines:
        return None
    args = (ground_fn, crossings, spans, clear, lift, max_grade)
    graph = _curve_graph(lines, pins)
    order = _simple_path(lines, graph)
    if order is not None:
        got = _path_curve(lines, order, *args)
        return None if got is None else [got[0][k] for k in range(len(lines))]
    nodes, edges, adj = graph
    # parts not joined at all (a chain also holds the decks crossing it, solved together for the deck-over-deck
    # rule): each joined group on its own (2026-10-19: one crossing deck sent a whole interchange to the legacy
    # profile)
    comp = list(range(len(nodes)))

    def find(a):
        while comp[a] != a:
            comp[a] = comp[comp[a]]
            a = comp[a]
        return a
    for a, b, _ in edges:
        comp[find(a)] = find(b)
    groups = {}
    for k, (a, _, _) in enumerate(edges):
        groups.setdefault(find(a), []).append(k)
    if len(groups) > 1:
        out = [None] * len(lines)
        for ks in groups.values():
            idx = {k: i for i, k in enumerate(ks)}
            sub = deck_curve([lines[k] for k in ks], ground_fn,
                             [(idx[cr[0]],) + tuple(cr[1:]) for cr in crossings if cr[0] in idx],
                             [(idx[k], a0, a1, z) for k, a0, a1, z in spans if k in idx], clear, lift, max_grade, pins)
            if sub is None:
                sub = deck_profile_once([lines[k] for k in ks], ground_fn,
                                        [(idx[cr[0]],) + tuple(cr[1:]) for cr in crossings if cr[0] in idx], clear, None,
                                        lift, pins=pins)
                if bridge_profile_mode() == "curve" and sub is not None and any(
                        s is None for s in sub):
                    return None
            for k, z in zip(ks, sub):
                out[k] = z
        return out
    deg = np.zeros(len(nodes), int)
    inc = [[] for _ in nodes]
    for k, (a, b, _) in enumerate(edges):
        deg[a] += 1
        deg[b] += 1
        inc[a].append(k)
        inc[b].append(k)
    abut = [i for i in range(len(nodes)) if deg[i] == 1]
    if not abut:
        return None
    out = [None] * len(lines)
    solved = {}                                 # node -> (z, [(unit tangent arriving, grade along it)])

    def walk(start, done_edges, stop):
        """Longest path from start through edges not in done_edges, ending at an abutment or a solved node
        (stop): [(edge, reversed)] (Dijkstra on lengths, the farthest such end)."""
        import heapq
        dist, prev = {start: 0.0}, {}
        q = [(0.0, start)]
        while q:
            d, u = heapq.heappop(q)
            if d > dist.get(u, np.inf):
                continue
            if u != start and stop(u):
                continue
            for k in inc[u]:
                if k in done_edges:
                    continue
                a, b, s = edges[k]
                v = b if u == a else a
                if d + s[-1] < dist.get(v, np.inf):
                    dist[v] = d + s[-1]
                    prev[v] = (u, k)
                    heapq.heappush(q, (d + s[-1], v))
        ends = [v for v in dist if v != start and stop(v)]
        if not ends:
            return None
        v = max(ends, key=lambda x: dist[x])
        path = []
        while v != start:
            u, k = prev[v]
            path.append((k, edges[k][0] != u))
            v = u
        return path[::-1]

    def record(order, s, z, xy):
        """The solved path's nodes: their heights and the tangents / grades arriving there both ways."""
        node_s = 0.0
        seq = []
        for k, rev in order:
            a, b, es = edges[k]
            seq.append((a if not rev else b, node_s))
            node_s += es[-1]
        last_k, last_rev = order[-1]
        seq.append((edges[last_k][1] if not last_rev else edges[last_k][0], node_s))
        gr = np.gradient(z, s)
        tg = np.gradient(xy, s, axis=0)
        for nd, sn in seq:
            j = int(np.clip(np.searchsorted(s, sn), 0, len(s) - 1))
            t = tg[j] / max(np.hypot(*tg[j]), 1e-9)
            zz, lst = solved.get(nd, (float(np.interp(sn, s, z)), []))
            lst += [(t, float(gr[j])), (-t, -float(gr[j]))]
            solved[nd] = (zz, lst)

    def end_of(node, direction):
        """(z, grade arriving) for a path leaving / reaching a solved node in this unit direction."""
        z, lst = solved[node]
        t, gr = max(lst, key=lambda tg: float(np.dot(tg[0], direction)))
        return (z, gr)

    # the trunk: of the abutment-to-abutment paths, the one carrying the highest requirement (the span over the
    # river: solved first, so a branch never pins it low), then the longest. (2026-10-19: the Crescent City
    # Connection's river decks were branches of a longer, low line through the approaches and stayed 17 m up)
    need = {}
    for cr in crossings:
        k, zc = cr[0], cr[2]
        cl = cr[3] if len(cr) > 3 and cr[3] is not None else clear
        need[k] = max(need.get(k, -np.inf), zc + cl)
    for k, _, _, zmin in spans:
        need[k] = max(need.get(k, -np.inf), zmin)
    best = None
    for a0 in abut:
        for a1 in abut:
            if a1 <= a0:
                continue
            p = walk(a0, set(), lambda v, a1=a1: v == a1)
            if p is None:
                continue
            L = sum(edges[k][2][-1] for k, _ in p)
            top = max((need.get(k, -np.inf) for k, _ in p), default=-np.inf)
            key = (top if np.isfinite(top) else -1e9, L)
            if best is None or key > best[0]:
                best = (key, p)
    if best is None:
        return None
    done = set()
    todo = [best[1]]
    first = True
    while todo:
        order = todo.pop()
        k0, r0 = order[0]
        start = edges[k0][1] if r0 else edges[k0][0]
        kl, rl = order[-1]
        end = edges[kl][0] if rl else edges[kl][1]
        e0 = e1 = None
        if not first:
            ln0 = np.asarray(lines[k0], np.float64)[:, :2]
            ln0 = ln0[::-1] if r0 else ln0
            d0 = (ln0[1] - ln0[0]) / max(np.hypot(*(ln0[1] - ln0[0])), 1e-9)
            e0 = end_of(start, d0)
            if end in solved:
                lnl = np.asarray(lines[kl], np.float64)[:, :2]
                lnl = lnl[::-1] if rl else lnl
                d1 = (lnl[-1] - lnl[-2]) / max(np.hypot(*(lnl[-1] - lnl[-2])), 1e-9)
                z1, g1 = end_of(end, d1)
                e1 = (z1, g1)
        first = False
        got = _path_curve(lines, order, *args, ends=(e0, e1))
        for k, _ in order:
            done.add(k)
        if got is None:
            continue
        res, xy, s, z = got
        for k, v in res.items():
            out[k] = v
        record(order, s, z, xy)
        if not todo:                            # the next branch: from any solved node with an open edge
            for nd in list(solved):
                if any(k not in done for k in inc[nd]):
                    p = walk(nd, done, lambda v: deg[v] == 1 or v in solved)
                    if p:
                        todo.append(p)
                        break
    if any(o is None for o in out):             # parts not reached (should not happen): the legacy answer
        return None
    return out


def _ease(lines, zs, ground_fn, lift=None):
    """Each part's heights after _merge_overlaps / the deck-over-deck rounds, run through a DECK_EASE_M moving
    average on a 1 m grid with its end heights kept (a linear correction) and never under the ground + lift:
    blending a part with its neighbour where their surfaces overlap left 2.5-4.3 % grade changes between deck
    points (Steel, Broadway, 2026-10-19). Pure given ground_fn."""
    lift = DECK_CLEAR if lift is None else lift
    out = []
    for ln, z in zip(lines, zs):
        z = np.asarray(z, np.float64)
        if len(z) < 3 or not np.isfinite(z).all():
            out.append(z)
            continue
        xy = np.asarray(ln, np.float64)[:, :2]
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        L = s[-1]
        n = int(round(DECK_EASE_M))
        if L < 2 * n:
            out.append(z)
            continue
        g = np.arange(0.0, L + 1e-9, 1.0)
        zg = np.interp(g, s, z)
        r = n // 2
        ext = np.r_[2 * zg[0] - zg[r:0:-1], zg, 2 * zg[-1] - zg[-2:-r - 2:-1]]
        sm = np.convolve(ext, np.ones(2 * r + 1) / (2 * r + 1), mode="valid")
        sm = sm + (zg[0] - sm[0]) + ((zg[-1] - sm[-1]) - (zg[0] - sm[0])) * g / L
        zz = np.interp(s, g, sm)
        gz = np.asarray(ground_fn(xy), np.float64) + lift
        zz[0], zz[-1] = z[0], z[-1]
        out.append(_lift_smooth(s, zz, gz))
    return out


def _lift_smooth(s, z, floor, iters=8):
    """z (heights at arcs s) raised to at least floor between its ends. The required *addition* to z runs out at
    LIFT_GRADE and is rounded over LIFT_EASE_M. Spreading absolute floor heights instead turned a legitimate 5 %
    uphill deck into a 3 % envelope held metres above its low abutment, then reset that end in one step.
    The ends keep their heights; whatever the rounding leaves short is clamped last. Pure."""
    s = np.asarray(s, np.float64)
    z = np.array(z, np.float64)
    floor = np.asarray(floor, np.float64)
    if len(z) < 3 or s[-1] <= 0:
        z[1:-1] = np.maximum(z[1:-1], floor[1:-1])
        return z
    # Uniform and end-inclusive: tiny rotation/length roundoff must not add an almost-zero final interval.
    g = np.linspace(0.0, s[-1], max(1, int(math.ceil(s[-1] - 1e-9))) + 1)
    zg = np.interp(g, s, z)
    # Within LIFT_EASE_M / 2 of an abutment, limit the lift to the landing grade,
    # but never put the deck under the ground itself.
    def bounded_floor(at, levels):
        d0, d1 = at, s[-1] - at
        cap = np.minimum(np.where(d0 < LIFT_EASE_M / 2 - 1e-8, z[0] + LIFT_END_GRADE * d0, np.inf),
                         np.where(d1 < LIFT_EASE_M / 2 - 1e-8, z[-1] + LIFT_END_GRADE * d1, np.inf))
        return np.maximum(np.minimum(levels, cap), levels - DECK_CLEAR)
    fg = bounded_floor(g, np.interp(g, s, floor))
    fs = bounded_floor(s, floor)
    r = int(LIFT_EASE_M // 2)
    inner = np.ones(len(g), bool)
    inner[0] = inner[-1] = False
    if not (fg[inner] > zg[inner] + 0.01).any() and not (fs[1:-1] > z[1:-1] + 0.01).any():
        out = np.array(z)
        out[1:-1] = np.maximum(out[1:-1], np.maximum(fs, np.interp(s, g, fg))[1:-1])
        return out
    # The metre grid can straddle a terrain peak sampled at a mapped vertex.
    # Feed each original requirement into both bracketing grid points before
    # smoothing, rather than losing it in the interpolation to and from g.
    source = s[1:-1]
    hi = np.minimum(np.searchsorted(g, source), len(g) - 1)
    left = np.maximum(hi - 1, 0)
    at_left = np.abs(g[left] - source) < 1e-9
    at_right = np.abs(g[hi] - source) < 1e-9
    lo = np.where(at_right, hi, left)
    hi = np.where(at_left, left, hi)
    want = np.maximum(fg - zg, 0.0)
    source_want = np.maximum(fs[1:-1] - z[1:-1], 0.0)
    np.maximum.at(want, lo, source_want)
    np.maximum.at(want, hi, source_want)
    addition = np.zeros_like(zg)
    for _ in range(iters):
        cone = _tents(g, np.where(inner, want, -np.inf), LIFT_GRADE)
        tgt = np.maximum(cone, 0.0)
        ext = np.r_[np.full(r, tgt[0]), tgt, np.full(r, tgt[-1])]
        addition = np.convolve(ext, np.ones(2 * r + 1) / (2 * r + 1), mode="valid")
        addition[0], addition[-1] = 0.0, 0.0
        zz = zg + addition
        short = np.where(inner, fg - zz, 0.0)
        source_short = np.maximum(fs[1:-1] - z[1:-1] - np.interp(source, g, addition), 0.0)
        if max(short.max(), source_short.max()) <= 0.01:
            break
        increase = np.maximum(short, 0.0)
        np.maximum.at(increase, lo, source_short)
        np.maximum.at(increase, hi, source_short)
        want += increase
    # Interpolate only the lift: resampling the solved profile itself could
    # lower an existing peak and lose its clearance at a mapped station.
    out = z + np.interp(s, g, addition)
    out[0], out[-1] = z[0], z[-1]
    ff = np.maximum(fs, np.interp(s, g, fg))
    out[1:-1] = np.maximum(out[1:-1], ff[1:-1])
    return out


def _curve_deck(s, z, floor, cap):
    """The deck profile z (at arcs s) projected onto its bounds [min(floor, cap), cap] and a change of grade of at most
    1/300 per metre (a 300 m vertical radius), its ends pinned; z itself when the map's constraints cannot all hold.
    Only bridge decks use this (wyo-backroads, PR #374: I-80 over Otto Road's 6 % crests). Pure."""
    import fast
    fast.require(83, "curve_deck")      # (Dykstra's projections, up to 2000 rounds: ~170 ms a deck in Python)
    f64 = np.float64
    return fast._ef.curve_deck(np.ascontiguousarray(s, f64), np.ascontiguousarray(z, f64),
                               np.ascontiguousarray(floor, f64), np.ascontiguousarray(cap, f64))


def _no_sag(s, z):
    """The upper concave hull of the profile (s, z) at s: every dip between two higher points filled with the
    straight grade between them. Pure."""
    hull = []
    for p in zip(s, z):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:      # hull[-1] under the chord: dropped
                hull.pop()
            else:
                break
        hull.append(p)
    hx, hy = zip(*hull)
    return np.interp(s, hx, hy)


def _dedupe(xy, tol=0.5):
    """A polyline without points closer than tol m to the previous one (ends kept). Pure."""
    import fast
    fast.require(66, "dedupe_polyline")     # (the Rust loop: 14-21 CPU-s a Deck teleport in each road process)
    xy = np.asarray(xy, np.float64)
    return xy[fast._ef.dedupe_polyline(xy, float(tol))]


def _ground_envelope(s, g, grade, curve=0.0):
    """The least profile over the ground heights g (at arcs s) that falls from each no faster than grade * d + curve *
    d^2 at d m from it: max_j g_j - grade |s - s_j| - curve (s - s_j)^2. Pure."""
    s, g = np.asarray(s, np.float64), np.asarray(g, np.float64)
    out = np.full(len(s), -np.inf)
    for i0 in range(0, len(s), 512):        # (in blocks: a long chain's (n, n) would be large)
        d = np.abs(s[None, :] - s[i0:i0 + 512, None])
        out = np.maximum(out, (g[i0:i0 + 512, None] - grade * d - curve * d * d).max(axis=0))
    return out


def _smooth_pinned(z, step, sigma):
    """Gaussian smoothing that keeps both end values and the trend at the ends (the profile is
    point-reflected about each end: plain mirroring pulled a hump's height into its pinned ends and
    made 15 % steps there). Pure."""
    z = np.asarray(z, np.float64)
    r = int(math.ceil(3 * sigma / step))
    r = min(r, len(z) - 1)
    if r < 1:
        return z.copy()
    k = np.exp(-0.5 * (np.arange(-r, r + 1) * step / sigma) ** 2)
    k /= k.sum()
    pad = np.r_[2 * z[0] - z[r:0:-1], z, 2 * z[-1] - z[-2:-r - 2:-1]]
    out = np.convolve(pad, k, mode="valid")
    out[0], out[-1] = z[0], z[-1]
    return out


def _curve_graph(lines, pins):
    """The deck graph with each incident end at a ground pin its own abutment.

    Splitting by displacing endpoints east could join a different nearby deck instead, leaving the real
    junction elevated. It also changed arc lengths and approach tangents according to part order and bearing.
    Only connectivity changes here; clearance arcs and physical/rendered polylines keep their mapped positions.
    """
    nodes, edges, adj = _chain_graph(lines)
    if pins is None or not len(pins):
        return nodes, edges, adj
    deg = np.array([len(neighbours) for neighbours in adj])
    pinned = _pin(nodes, deg, pins)
    split = (deg > 1) & (pinned == 1)
    if not split.any():
        return nodes, edges, adj
    seen = set()
    out = []
    for a, b, s in edges:
        ends = []
        for nd in (a, b):
            if split[nd]:
                if nd in seen:
                    nodes.append(nodes[nd])
                    ends.append(len(nodes) - 1)
                    continue
                seen.add(nd)
            ends.append(nd)
        out.append((ends[0], ends[1], s))
    adj = [[] for _ in nodes]
    for a, b, s in out:
        adj[a].append((b, s[-1]))
        adj[b].append((a, s[-1]))
    return nodes, out, adj


def _pin(nodes, deg, pins):
    """deg with the joints at pins made ends (1): abutments, held at the ground. Two motorway decks met at a node the
    motorway went on from on the ground; taken for a joint of decks it stood 3.9 m over the road it ran on to, a wall
    (the Four Level Interchange, Los Angeles, 2026-10-27). Pure."""
    if pins is None or not len(pins):
        return deg
    P = np.asarray(pins, np.float64).reshape(-1, 2)
    deg = deg.copy()
    for i, nd in enumerate(nodes):
        if deg[i] > 1 and (np.hypot(P[:, 0] - nd[0], P[:, 1] - nd[1]) < PIN_M).any():
            deg[i] = 1
    return deg


def deck_heights(lines, end_z, pins=None):
    """Deck heights of chained bridge parts. lines: polylines (k, 2) in metres (one per bridge part);
    parts sharing an end point (within 0.5 m) are joined. end_z(xy (m, 2)) -> heights of the ground at
    the abutments (chain ends: points used by one part only). A joint's height is the abutments' heights weighted by
    1 / their distance along the chain, and each part runs straight between its two ends (between two abutments: a
    straight line). Weighting every point, a part from an abutment to a joint of three parts rose ~16 % off its
    abutment and flattened toward the joint, 38 % once smoothed: US-24's deck near Buena Vista CO, 1.1 m up in its
    first 3 m, 2026-10-27. Returns one height array per line. Pure given end_z."""
    if not lines:
        return []
    edges, ez, dist = _end_dists(lines, end_z, pins)
    out = []

    def weighted(dk):
        num, den = np.zeros(len(dk[0])), np.zeros(len(dk[0]))
        for k, d in enumerate(dk):
            ok = np.isfinite(d)
            w = np.where(ok, 1.0 / np.maximum(d, 1e-6), 0.0)
            num += w * np.where(ok, ez[k], 0.0)
            den += w
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    for a, b, s in edges:
        if not dist:                            # a closed loop: no abutment known
            out.append(np.full(len(s), np.nan))
            continue
        za, zb = weighted([np.array([d[a], d[b]], np.float64) for d in dist])
        if np.isfinite(za) and np.isfinite(zb) and s[-1] > 0:
            out.append(za + (zb - za) * (s / s[-1]))
        else:
            out.append(weighted([np.minimum(d[a] + s, d[b] + (s[-1] - s)) for d in dist]))
    return out


def _end_dists(lines, end_z, pins=None):
    """The chain of lines (joined at end points within 0.5 m): its edges (start node, end node, arc (k,)),
    the heights end_z gives at its ends (points of one line only) and per end the distance along the chain
    to every node. Pure given end_z."""
    join, nodes = _EndJoin(), []

    def node(p):
        k = join(p)                                                 # 0.5 m
        if k == len(nodes):
            nodes.append(np.asarray(p[:2], np.float64))
        return k
    edges = []
    for ln in lines:
        seg = np.hypot(*np.diff(ln[:, :2], axis=0).T)
        s = np.r_[0.0, np.cumsum(seg)]
        edges.append((node(ln[0]), node(ln[-1]), s))
    deg = np.zeros(len(nodes), int)
    adj = [[] for _ in nodes]
    for a, b, s in edges:
        deg[a] += 1
        deg[b] += 1
        adj[a].append((b, s[-1]))
        adj[b].append((a, s[-1]))
    deg = _pin(nodes, deg, pins)
    ends = [i for i in range(len(nodes)) if deg[i] == 1]
    ez = np.asarray(end_z(np.array([nodes[i] for i in ends]).reshape(-1, 2)), np.float64) if ends else np.zeros(0)
    import heapq
    dist = []                                   # per end: distance along the chain to every node
    for e in ends:
        d = np.full(len(nodes), np.inf)
        d[e] = 0.0
        q = [(0.0, e)]
        while q:
            dd, u = heapq.heappop(q)
            if dd > d[u]:
                continue
            for v, w in adj[u]:
                if dd + w < d[v]:
                    d[v] = dd + w
                    heapq.heappush(q, (dd + w, v))
        dist.append(d)
    return edges, ez, dist


def _deck_crossings(lines):
    """(i, j, arc on i, arc on j) where deck parts i < j cross (not at their end points). Pure."""
    import fast
    if not lines:
        return []
    fast.require(0, "deck_crossings")
    pts = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    starts = np.r_[0, np.cumsum([len(p) for p in pts])].astype(np.int64)
    return fast._ef.deck_crossings(np.ascontiguousarray(np.vstack(pts)), starts)
