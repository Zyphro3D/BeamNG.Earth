"""Road grading: a motorway's carriageways and decks solved as one profile (split out of grading.py's _Planner;
carriageways.py does the solving).
"""
import math
import os
import time
import numpy as np
import mapcells
import roads as roadsmod
import seatiles
import crossings as _xs
import carriageways as _cw
from gradbase import (
    APPROACH_MIN_CROSS_M,
    CW_HEADROOM_M,
    CW_MARGIN_M,
    CW_MAX_DIG_M,
    CW_MIN_SIN,
    CW_NONFAM_HI,
    CW_OVERLAP_MAX_M,
    CW_OVERLAP_STEP_M,
    CW_STACK_SIN,
    CW_UNDER_M,
    MEDIAN_M,
    Plan,
    ROAD_LIFT,
    SIGMA_MAJOR_M,
    UNGRADED,
    _cw_half,
    _cw_level,
    _is_link,
    _seg_dist,
    log,
    median_profile,
    smooth_profile,
)


class _CarriagewaysMixin:
    """_Planner's carriageway solving (split out of grading.py; _Planner inherits it)."""

    def _cw_cell_parts(self, k):
        """Cell k's parts of the motorway and trunk families: [(key, line (n, 2), is_bridge, cls, feature id, level,
        half width)], each once (level: a bridge's OSM layer, 1 without one; 0 on the ground)."""
        memo = self.__dict__.setdefault("_cw_cells", {})
        sk = str(k)
        with self.lock:
            got = memo.get(sk)
        if got is not None:
            return got
        c = self._cell_enu(k)
        out, seen = [], set()
        nb = len(c["bridges"])
        items = [(ln, False, cls, gid[0], 0.0, float(h)) for (ln, h, cls), gid in zip(c["ground"], c["ground_ids"])]
        items += [(ln, True, cls, fid, float(lv), float(h))
                  for ln, cls, fid, lv, h in zip(c["bridges"], c.get("bridge_cls", [None] * nb),
                                                 c.get("bridge_ids", [None] * nb), c.get("bridge_levels", [1.0] * nb),
                                                 c.get("bridge_half", [4.0] * nb))]
        for ln, br, cls, fid, lv, half in items:
            if _cw.family(cls) is None or len(ln) < 2:
                continue
            ln = np.asarray(ln, np.float64)[:, :2]
            key = ("b" if br else "g", round(float(ln[0, 0]), 1), round(float(ln[0, 1]), 1),
                   round(float(ln[-1, 0]), 1), round(float(ln[-1, 1]), 1))
            if key in seen:
                continue
            seen.add(key)
            out.append((key, ln, br, str(cls), fid, lv, half))
        with self.lock:
            if len(memo) > 512:
                memo.clear()
            memo[sk] = out
        return out

    def _z_raw(self, p):
        """The raw elevation at points p (n, 2) of this frame."""
        p = np.asarray(p, np.float64).reshape(-1, 2)
        a, b, _ = self.frame.enu_to_geodetic(p[:, 0], p[:, 1], 0.0)
        return np.asarray(self.elev(np.asarray(a, np.float64), np.asarray(b, np.float64)), np.float64).ravel()

    def _cw_region(self, home):
        """Cell home's carriageways: the family parts within CW_MARGIN_M of it (from every cell there, each once), their
        chains (carriageways.chains) and those solved together (_cw_solve): {"parts": [...], "chains": [...],
        "prof": {part key: (chain samples xy, z)}}, cached. Every road and deck is answered by the region of the cell
        holding its middle, so every cell and process asking gets the same heights."""
        memo = self.__dict__.setdefault("_cw_regions", {})
        hk = str(home)
        with self.lock:
            got = memo.get(hk)
        if got is not None:
            return got
        # (the run's other processes: the first to need a region solves it, the rest read it; each road-tile process
        # solving the same regions for itself, a worker's tiles waited past earthd's 300 s, 2026-11-02)
        sh, mine = self.shared, False
        hkey = (int(home.z), int(home.x), int(home.y)) if hasattr(home, "z") else None
        tag = f"{float(self.frame.lat0):.6f}_{float(self.frame.lon0):.6f}"
        if sh is not None and hkey is not None and hasattr(sh, "region_claim"):
            try:
                got, mine = sh.region_claim(tag, hkey)
            except Exception as e:      # (no shared copy: solved here)
                log.debug("grading: carriageways of %s: shared: %s: %s", hk, type(e).__name__, e)
                got, mine = None, False
            if got is not None:
                with self.lock:
                    if len(memo) > 64:
                        memo.clear()
                    memo[hk] = got
                return got
        try:
            got = self._cw_region_calc(home, hk)
            if mine and not got.get("failed"):
                sh.region_store(tag, hkey, got)
        finally:
            if mine:
                sh.region_release(tag, hkey)
        with self.lock:
            if len(memo) > 64:
                memo.clear()
            memo[hk] = got
        return got

    def _cw_region_calc(self, home, hk):
        lon0, lat0, lon1, lat1 = (float(v) for v in home.bounds())
        ce, cn, _ = self.frame.geodetic_to_enu(np.array([lat0, lat0, lat1, lat1]), np.array([lon0, lon1, lon0, lon1]),
                                               0.0)
        box_lo = np.array([np.min(ce), np.min(cn)]) - CW_MARGIN_M
        box_hi = np.array([np.max(ce), np.max(cn)]) + CW_MARGIN_M
        pad = CW_MARGIN_M / 111320.0
        mid_la = (lat0 + lat1) / 2
        keys = sorted(set(mapcells.cells_for_bbox(lat0 - pad, lon0 - pad / max(0.05, math.cos(math.radians(mid_la))),
                                                  lat1 + pad, lon1 + pad / max(0.05, math.cos(math.radians(mid_la))))),
                      key=str)
        parts, seen = [], set()
        try:
            cell_parts = [self._cw_cell_parts(k) for k in keys]
        except Exception as e:      # (a cell that cannot be read: no carriageways here, the rules as before)
            log.debug("grading: carriageways of %s: %s: %s", hk, type(e).__name__, e)
            return {"parts": [], "prof": {}, "failed": True}
        for cp in cell_parts:
            for q in cp:
                if q[0] in seen:
                    continue
                if ((q[1] >= box_lo) & (q[1] <= box_hi)).all(1).any():
                    seen.add(q[0])
                    parts.append(q)
        parts.sort(key=lambda q: str(q[0]))
        chains = []
        for seq in _cw.chains(parts):
            chains.append(([parts[i] for i, _r in seq], [(n, r) for n, (_i, r) in enumerate(seq)]))
        chains = [ch for ch in chains if any(p_[2] for p_ in ch[0]) or len(ch[0]) > 0]
        prof = {}
        try:
            if chains:
                prof = self._cw_cluster_solve(chains)
        except Exception as e:      # the rules as before rather than no grading
            import revisit
            revisit.fault()         # incomplete profiles must not become a persistent plan/region
            log.warning("grading: carriageways of %s: %s: %s", hk, type(e).__name__, e)
            return {"parts": parts, "prof": {}, "failed": True}
        return {"parts": parts, "prof": prof}

    def _cw_cluster_solve(self, chains):
        f64 = np.float64
        data, samples = [], []
        for ch in chains:
            parts, order = ch
            xy, s, pi, br, fe = _cw.sample(parts, order)
            la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
            raw = np.asarray(self.elev(np.asarray(la, f64), np.asarray(lo, f64)), f64).ravel()
            ok = np.isfinite(raw)
            if not ok.any():
                raw = np.zeros(len(raw))
                ok[:] = True
            elif not ok.all():                          # (no elevation there: the nearest that is)
                raw = np.interp(s, s[ok], raw[ok])
            g = ~br
            if g.sum() >= 2:
                t = np.interp(s, s[g], raw[g])          # (the ground across the bridges, for the filters)
                t = smooth_profile(median_profile(t, _cw.STEP_M, MEDIAN_M), _cw.STEP_M, SIGMA_MAJOR_M)
            else:
                t = raw.copy()
            w = _cw.weights(br, fe) * _cw.GROUND_W
            # (its ends, where it meets the rest of the network, on their ground: unless on a bridge, a joint of decks)
            for e in (0, len(w) - 1):
                if not br[e]:
                    w[e] = _cw.ANCHOR_W
            if g.sum() < 2:
                w = np.where(np.arange(len(w)) < 0, 0.0, 1e-3)      # (all bridge: held lightly to the ground)
            # the steepest of its parts' classes' limits (a chain is of one family; a link's limits are looser)
            gr = max(_cw.limits(p_[3])[0] for p_ in parts)
            K = min(_cw.limits(p_[3])[1] for p_ in parts)
            data.append(dict(t=t, w=w, grade=gr, K=K, lo=np.full(len(s), -np.inf), hi=np.full(len(s), np.inf)))
            samples.append((xy, s, pi, br, raw))
        # bounds: tracks under its decks; other decks (not of the family) over its roads
        allpts = np.concatenate([x[0] for x in samples])
        la, lo, _ = self.frame.enu_to_geodetic(allpts[:, 0], allpts[:, 1], 0.0)
        pad = 40.0 / 111320.0
        keys = sorted(set(mapcells.cells_for_points(np.asarray(la, f64), np.asarray(lo, f64), pad,
                                                    pad / max(0.05, math.cos(math.radians(float(np.mean(la))))))),
                      key=str)
        cs = [self._cell_enu(k) for k in keys]
        rails = [np.asarray(r, f64)[:, :2] for c in cs for r in c.get("rail_ground", ()) if len(r) >= 2]
        nonfam = []
        for c in cs:
            for line, bc in zip(c["bridges"], c.get("bridge_cls", [None] * len(c["bridges"]))):
                if bc is not None and _cw.family(bc) is None and len(line) >= 2:
                    nonfam.append(np.asarray(line, f64)[:, :2])

        def zraw(q):
            a, b_, _ = self.frame.enu_to_geodetic(np.array([q[0]]), np.array([q[1]]), 0.0)
            return float(np.asarray(self.elev(np.asarray(a, f64), np.asarray(b_, f64)), f64).ravel()[0])

        def zat(p):
            p = np.asarray(p, f64).reshape(-1, 2)
            a, b_, _ = self.frame.enu_to_geodetic(p[:, 0], p[:, 1], 0.0)
            return np.asarray(self.elev(np.asarray(a, f64), np.asarray(b_, f64)), f64).ravel()
        # the other roads on the ground (under the decks: they dip to clear them, except on low ground, where the
        # deck rises instead: a dug road there is under the water table, Wezep's street under the A28)
        streets = [(np.asarray(line, f64)[:, :2], float(half)) for c in cs for line, half, gc in c["ground"]
                   if _cw.family(gc) is None and gc not in _xs.PATHS and gc not in UNGRADED and len(line) >= 2]

        def nearest(n, part_i, q):
            xy, s, pi, br, raw = samples[n]
            cand = np.flatnonzero(pi == part_i)
            return int(cand[np.argmin(np.hypot(*(xy[cand] - q).T))])
        # (bounding boxes: a deck meets only the streets and tracks near it, a scan of them all at each crossing was
        # most of a region's time: 43 s of a Perth cell's 110, 2026-11-02)
        def boxes_of(ls):
            if not ls:
                return np.zeros((0, 2)), np.zeros((0, 2))
            return np.array([l_.min(0) for l_ in ls]), np.array([l_.max(0) for l_ in ls])
        st_lo, st_hi = boxes_of([st[0] for st in streets])
        rl_lo, rl_hi = boxes_of(rails)

        def near_of(lo_, hi_, line, pad=2.0):
            a, b_ = line.min(0) - pad, line.max(0) + pad
            return np.flatnonzero(((lo_ <= b_) & (hi_ >= a)).all(1))
        for n, ch in enumerate(chains):
            parts = ch[0]
            for i, p_ in enumerate(parts):
                clear = CW_HEADROOM_M + roadsmod.deck_depth(roadsmod.geom_span([p_[1]])) + ROAD_LIFT
                rails_n = [rails[m] for m in near_of(rl_lo, rl_hi, p_[1])] if p_[2] and rails else []
                streets_n = [streets[m] for m in near_of(st_lo, st_hi, p_[1])] if p_[2] and streets else []
                if p_[2] and rails_n:
                    for c in roadsmod._crossings([p_[1]], rails_n):
                        j = nearest(n, i, np.asarray(c[2], f64))
                        data[n]["lo"][j] = max(data[n]["lo"][j], zraw(c[2]) + clear)
                if p_[2] and streets_n:
                    for c in roadsmod._crossings([p_[1]], [st[0] for st in streets_n]):
                        L = float(np.hypot(*np.diff(p_[1], axis=0).T).sum())
                        if c[1] < APPROACH_MIN_CROSS_M or c[1] > L - APPROACH_MIN_CROSS_M:
                            continue
                        q = np.asarray(c[2], f64)
                        sl = min(streets_n, key=lambda st: _seg_dist(st[0], q))[0]
                        tl = _xs.trusted_under(q, _xs.dir_at(sl, q), 8.0, zat)
                        if tl is None or tl >= seatiles.SEA_E + _xs.LOW_M:
                            continue
                        j = nearest(n, i, q)
                        data[n]["lo"][j] = max(data[n]["lo"][j], tl + clear)
                if not p_[2] and nonfam and CW_NONFAM_HI:
                    for c in roadsmod._crossings(nonfam, [p_[1]]):
                        b = nonfam[c[0]]
                        L = float(np.hypot(*np.diff(b, axis=0).T).sum())
                        if c[1] < APPROACH_MIN_CROSS_M or c[1] > L - APPROACH_MIN_CROSS_M:
                            continue
                        ze = np.array([zraw(b[0]) + self._end_raised(b[0]), zraw(b[-1]) + self._end_raised(b[-1])])
                        deck = float(ze[0] + (ze[1] - ze[0]) * c[1] / max(L, 1e-9))
                        bc = roadsmod.deck_road_clear(roadsmod.deck_depth(roadsmod.geom_span([b]))) + ROAD_LIFT
                        j = nearest(n, i, np.asarray(c[2], f64))
                        # (lowered under it CW_MAX_DIG_M at most: past that the deck is too low, as before)
                        data[n]["hi"][j] = min(data[n]["hi"][j], max(deck - bc, data[n]["t"][j] - CW_MAX_DIG_M))
        # the ground under any deck (and beside it, its abutments' fill) is not the road's: the elevation fills the gap
        # under a bridge at about the deck's height (a freeway read 2 m up under each street bridge over it)
        decks_all = [np.asarray(line, f64)[:, :2] for c in cs for line in c["bridges"] if len(line) >= 2]
        if decks_all:
            A = np.concatenate([d[:-1] for d in decks_all])
            B = np.concatenate([d[1:] for d in decks_all])
            AB = B - A
            L2 = np.maximum((AB * AB).sum(1), 1e-12)
            for n, (xy, s_, pi, br, raw) in enumerate(samples):
                g_ = np.flatnonzero(~br)
                if not len(g_):
                    continue
                lo_q, hi_q = xy[g_].min(0) - 30.0, xy[g_].max(0) + 30.0
                near = np.flatnonzero(~((np.maximum(A, B) < lo_q).any(1) | (np.minimum(A, B) > hi_q).any(1)))
                if not len(near):
                    continue
                q = xy[g_]
                t_ = np.clip(((q[:, None, :] - A[near][None]) * AB[near][None]).sum(2) / L2[near][None], 0.0, 1.0)
                d = np.hypot(*(q[:, None, :] - (A[near][None] + t_[..., None] * AB[near][None])).transpose(2, 0, 1)).min(1)
                u = np.clip((d - CW_UNDER_M) / CW_UNDER_M, 0.0, 1.0)
                data[n]["w"][g_] *= u * u * (3.0 - 2.0 * u)
        # couplings: ends at a branch point equal; one carriageway over another
        couplings = []
        ends = []
        for n, (xy, s, pi, br, raw) in enumerate(samples):
            ends += [(n, 0, xy[0]), (n, len(xy) - 1, xy[-1])]
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                na, ia, pa = ends[a]
                nb, ib, pb = ends[b]
                if na != nb and np.hypot(*(pa - pb)) <= _cw.COUPLE_M:
                    couplings.append((na, ia, nb, ib, "eq", 0.0))
        # (an end meeting the inside of another chain: a slip road at its overpass road's node, solved with it)
        joined = _cw.branch_ties([x[0] for x in samples], [x[3] for x in samples])
        couplings += [(na, ia, nb, jb, "eq", 0.0) for na, ia, nb, jb in joined]
        cross = []
        boxes = [(x[0].min(0), x[0].max(0)) for x in samples]
        for na in range(len(chains)):
            for nb in range(na + 1, len(chains)):
                if (boxes[na][1] < boxes[nb][0]).any() or (boxes[nb][1] < boxes[na][0]).any():
                    continue
                hits = roadsmod._crossings_by([p_[1] for p_ in chains[na][0]], [p_[1] for p_ in chains[nb][0]])
                for ib, lst in enumerate(hits):
                    pb = chains[nb][0][ib]
                    for c in lst:
                        ia = int(c[0])
                        pa = chains[na][0][ia]
                        if pa[2] == pb[2] and not pa[2]:
                            continue            # (two roads on the ground: a junction, not a separation)
                        q = np.asarray(c[2], f64)
                        ua, ub = _xs.dir_at(pa[1], q), _xs.dir_at(pb[1], q)
                        sin = abs(ua[0] * ub[1] - ua[1] * ub[0])
                        # (two decks mapped at different levels: a separation however shallow the angle, the ramps of a
                        # stack cross at a few degrees; else lines lying along each other are a duplicate or a merge)
                        stacked = pa[2] and pb[2] and _cw_level(pa) != _cw_level(pb)
                        if sin < CW_MIN_SIN and not (stacked and sin >= CW_STACK_SIN):
                            continue
                        if min(np.hypot(*(pa[1][0] - q)), np.hypot(*(pa[1][-1] - q)), np.hypot(*(pb[1][0] - q)),
                               np.hypot(*(pb[1][-1] - q))) < 2.0:
                            continue            # (joined there)
                        cross.append((na, ia, nb, ib, q, sin, ua))
        # the interchange built from the ground up (OSM's layer the order, 2026-11-02): each chain's level is its
        # highest part's (a bridge's layer, 1 without one; 0 for a chain all on the ground). The lowest level's chains
        # are solved first, on the elevation, and kept; then each level up, over and joined to what is below and not
        # moving it: a crossing of a kept chain is a bound (its height plus the clearance, wherever the two overlap),
        # a branch point on one is a pin. Only two parts of one level crossing within a level are ordered by a
        # solve of that level without them.
        stage = [max(_cw_level(p_) for p_ in ch[0]) for ch in chains]
        # (one road through a branch point: the chains either side continuing the same way (its parts there of one
        # feature) are built at one level, together: Graham Farmer's ground beyond its deck, solved first and kept,
        # made the deck dive to it)
        par = list(range(len(chains)))

        def root(n):
            while par[n] != n:
                par[n] = par[par[n]]
                n = par[n]
            return n
        for a_, i_, b_, j_, kind, _v in couplings:
            if kind != "eq":
                continue
            pa_ = chains[a_][0][int(samples[a_][2][i_])]
            pb_ = chains[b_][0][int(samples[b_][2][j_])]
            if pa_[4] is not None and pa_[4] == pb_[4]:
                par[root(a_)] = root(b_)
        for na, _ia, nb, _jb in joined:
            par[root(na)] = root(nb)
        # (the roads leaving a deck's end: its approaches rise with it. Solved first, on the ground, they pinned the deck to
        # it, a 95 m span that must clear a road 7 m under it left 2 m over its own approaches)
        for a_, i_, b_, j_, kind, _v in couplings:
            if kind == "eq" and bool(samples[a_][3][i_]) != bool(samples[b_][3][j_]):
                par[root(a_)] = root(b_)
        top = {}
        for n in range(len(chains)):
            top[root(n)] = max(top.get(root(n), 0.0), stage[n])
        stage = [top[root(n)] for n in range(len(chains))]
        seps = []           # (upper chain, its sample, lower chain, its sample, clearance) per overlap sample; or a tie
        ties = []           # (na, ia, nb, ib, q, sin, ua): one level's two parts, the order its solve's
        for na, ia, nb, ib, q, sin, ua in cross:
            pa, pb = chains[na][0][ia], chains[nb][0][ib]
            if pa[2] and not pb[2]:
                a_up = True
            elif pb[2] and not pa[2]:
                a_up = False
            elif _cw_level(pa) != _cw_level(pb):
                a_up = _cw_level(pa) > _cw_level(pb)
            elif stage[na] != stage[nb]:
                a_up = stage[na] > stage[nb]        # (one level's decks: the one of the higher-built chain over)
            else:
                ties.append((na, ia, nb, ib, q, sin, ua))
                continue
            seps += self._cw_overlap(chains, samples, nearest, (na, ia, nb, ib, q, sin, ua), a_up)
        zs = [None] * len(chains)
        # (one budget for the whole region, every level's solves: per solve, the High Five's six levels' took minutes
        # and every process needing the region waited on it, 2026-11-02)
        deadline = time.monotonic() + _cw.BUDGET_S
        for L in sorted(set(stage)):
            idx = [n for n in range(len(chains)) if stage[n] == L]
            loc = {n: m for m, n in enumerate(idx)}
            d_ = [dict(data[n], t=data[n]["t"].copy(), w=data[n]["w"].copy(), lo=data[n]["lo"].copy(),
                       hi=data[n]["hi"].copy()) for n in idx]
            cp = []
            for a_, i_, b_, j_, kind, v in couplings:          # (the branch points: "eq")
                if a_ in loc and b_ in loc:
                    cp.append((loc[a_], i_, loc[b_], j_, kind, v))
                elif a_ in loc and zs[b_] is not None:
                    d = d_[loc[a_]]
                    d["lo"][i_] = max(d["lo"][i_], zs[b_][j_])
                    d["hi"][i_] = min(d["hi"][i_], zs[b_][j_])
                elif b_ in loc and zs[a_] is not None:
                    d = d_[loc[b_]]
                    d["lo"][j_] = max(d["lo"][j_], zs[a_][i_])
                    d["hi"][j_] = min(d["hi"][j_], zs[a_][i_])

            def bound(lst, cp_):
                for nu, ju, nl, jl, clear in lst:
                    if nu in loc and nl in loc:
                        cp_.append((loc[nu], ju, loc[nl], jl, "ge", clear))
                    elif nu in loc and zs[nl] is not None:      # (over a kept chain)
                        d_[loc[nu]]["lo"][ju] = max(d_[loc[nu]]["lo"][ju], zs[nl][jl] + clear)
                    elif nl in loc and zs[nu] is not None:      # (under a kept chain's deck)
                        d_[loc[nl]]["hi"][jl] = min(d_[loc[nl]]["hi"][jl], zs[nu][ju] - clear)
            bound(seps, cp)
            mine = [t_ for t_ in ties if t_[0] in loc]
            def solve_stage():
                info = {}
                profiles = _cw.solve(d_, cp, deadline=deadline, diagnostics=info)
                if not info["converged"]:
                    raise RuntimeError(f"carriageway level {L}: {info['status']} (feasible={info['feasible']})")
                return profiles

            if mine:
                pre = solve_stage()
                tie_seps = []
                for na, ia, nb, ib, q, sin, ua in mine:
                    a_up = pre[loc[na]][nearest(na, ia, q)] >= pre[loc[nb]][nearest(nb, ib, q)]
                    tie_seps += self._cw_overlap(chains, samples, nearest, (na, ia, nb, ib, q, sin, ua), a_up)
                bound(tie_seps, cp)
            got = solve_stage()
            for n in idx:
                zs[n] = got[loc[n]]
        couplings = couplings + [(nu, ju, nl, jl, "ge", c_) for nu, ju, nl, jl, c_ in seps]
        if os.environ.get("BNE_CW_DUMP"):
            import pickle
            with open(os.environ["BNE_CW_DUMP"] + f".{len(chains)}.{len(couplings)}.pkl", "wb") as fh:
                pickle.dump(dict(data=data, couplings=couplings, zs=zs, samples=samples,
                                 parts=[[(p_[0], p_[2], p_[3]) for p_ in ch[0]] for ch in chains]), fh)
        out = {}
        for n, ch in enumerate(chains):
            for p_ in ch[0]:
                out[p_[0]] = (samples[n][0], zs[n])
        moved = [float(np.max(np.abs((zs[n] - d["t"])[~samples[n][3]]))) if (~samples[n][3]).any() else 0.0
                 for n, d in enumerate(data)]
        log.debug("grading: carriageway cluster of %d chains, %d couplings: moved at most %.1f m from the ground",
                  len(chains), len(couplings), max(moved) if moved else 0.0)
        if os.environ.get("BNE_CW_DEBUG") and moved:
            n = int(np.argmax(moved))
            xy, s_, pi, br, raw = samples[n]
            d = np.where(~br, np.abs(zs[n] - data[n]["t"]), 0.0)
            j = int(np.argmax(d))
            log.debug("grading: CWDBG chain %d (%s) at %s: t %.1f z %.1f raw %.1f w %.2f; lo %s hi %s; parts %s",
                      n, chains[n][0][0][3], np.round(xy[j]), data[n]["t"][j], zs[n][j], raw[j], data[n]["w"][j],
                      np.round(data[n]["lo"][np.isfinite(data[n]["lo"])], 1)[:5],
                      np.round(data[n]["hi"][np.isfinite(data[n]["hi"])], 1)[:5], [(p_[2], len(p_[1])) for p_ in chains[n][0]])
        return out

    def _cw_overlap(self, chains, samples, nearest, x, a_up):
        """The clearance couplings of crossing x = (na, ia, nb, ib, q, sin, ua), a over b when a_up: (upper chain, its
        sample, lower chain, its sample, clearance), at the crossing and along the length the two decks overlap (at a
        shallow angle tens of metres, a stack's ramps)."""
        f64 = np.float64
        na, ia, nb, ib, q, sin, ua = x
        pa, pb = chains[na][0][ia], chains[nb][0][ib]
        (nu, iu, pu), (nl, il, pl_) = ((na, ia, pa), (nb, ib, pb)) if a_up else ((nb, ib, pb), (na, ia, pa))
        clear = CW_HEADROOM_M + roadsmod.deck_depth(roadsmod.geom_span([pu[1]])) + ROAD_LIFT
        wide = _cw_half(pa) + _cw_half(pb)
        run = min(CW_OVERLAP_MAX_M, wide / max(sin, 1e-3))
        st = CW_OVERLAP_STEP_M      # (between them the curve limits keep the deck smooth: fewer couplings)
        offs = [0.0] + [d for k in range(1, 64) for d in (k * st, -k * st) if k * st <= run]
        out, seen = [], set()
        for d in offs:
            qd = q + d * np.asarray(ua, f64)
            ju, jl = nearest(nu, iu, qd), nearest(nl, il, qd)
            xu = samples[nu][0][ju]
            if d and (np.hypot(*(xu - qd)) > 3.0 or _seg_dist(pl_[1], xu) > wide):
                continue        # (past the overlap, or off these parts)
            if (ju, jl) not in seen:
                seen.add((ju, jl))
                out.append((nu, ju, nl, jl, clear))
        return out

    def _cw_z_at(self, line, pts, fid=None):
        """Heights of points pts (n, 2) on the family part line (of feature fid) from its carriageway's profile, or
        None."""
        line = np.asarray(line, np.float64)[:, :2]
        mid = line[len(line) // 2]
        la, lo, _ = self.frame.enu_to_geodetic(np.array([mid[0]]), np.array([mid[1]]), 0.0)
        try:
            reg = self._cw_region(mapcells.cell_of(float(la[0]), float(lo[0])))
        except Exception as e:      # (the rules as before)
            log.debug("grading: carriageway at %s: %s: %s", np.round(mid, 1), type(e).__name__, e)
            return None
        if reg.get("failed"):
            import revisit
            revisit.fault()         # also on later reads of the session's memoized failed region
            return None
        best = None
        for q in reg["parts"]:
            if fid is not None and q[4] != fid:
                continue
            if (q[1].max(0) < mid - 2.0).any() or (q[1].min(0) > mid + 2.0).any():
                continue
            d = _seg_dist(q[1], mid)
            if d <= 1.0 and (best is None or d < best[0]):
                best = (d, q)
        if best is None:
            return None
        prof = reg["prof"].get(best[1][0])
        if prof is None:
            return None
        xy, z = prof
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        a, b = xy[:-1], xy[1:]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(1), 1e-12)
        out = np.empty(len(pts))
        for n, q in enumerate(pts):
            t = np.clip(((q - a) * ab).sum(1) / L2, 0.0, 1.0)
            d = np.hypot(*(a + t[:, None] * ab - q).T)
            j = int(np.argmin(d))
            if d[j] > 3.0:
                return None
            out[n] = z[j] + (z[j + 1] - z[j]) * t[j]
        return out

    def _apply_carriageways(self, plans, specs):
        """CW_ON: the plans of motorway- and trunk-family roads set to their carriageways' profiles (in place)."""
        for n, ((fid, p), sp) in enumerate(zip(plans, specs)):
            if sp is None or _cw.family(getattr(sp, "cls", None) + ("_link" if _is_link(sp) else "")) is None:
                continue
            xy = np.asarray(p.xy, np.float64)[:, :2]
            if len(xy) < 2:
                continue
            z = self._cw_z_at(xy, xy, str(fid).split(":")[0])
            if z is None:
                continue
            np_ = Plan(p.xy, z, p.half, p.abut, p.cut, p.road_half)
            np_.rid, np_.max_wall, np_.bank = p.rid, p.max_wall, getattr(p, "bank", None)
            np_.steep = getattr(p, "steep", False)
            plans[n] = (fid, np_)
