"""Road grading: bridge approaches and the roads under bridges (split out of grading.py's _Planner): approach
lifts, underpass dips and the ground under decks.
"""
import math
import os
import numpy as np
import mapcells
import railways as railwaysmod
import roads as roadsmod
import seatiles
import crossings as _xs
import carriageways as _cw
from gradbase import (
    APPROACH_MAX_LIFT,
    APPROACH_MIN_CROSS_M,
    APPROACH_MIN_LIFT,
    APPROACH_SKIP,
    LINK_BESIDE_ON,
    CW_HEADROOM_M,
    CW_ON,
    MEASURED_SIDE_RUN,
    PROFILE_STEP,
    Plan,
    ROAD_LIFT,
    SLOPE_RUN,
    UNDER_ANY_ON,
    UNDER_CHAIN_MAX_M,
    UNDER_CHAIN_ON,
    UNDER_DECK_ON,
    UNDER_GRADE,
    UNDER_LIDAR_DECK_M,
    UNDER_LIDAR_ON,
    UNDER_MAX_BRIDGE_M,
    UNDER_MAX_DROP_M,
    UNDER_NEAR_M,
    UNDER_POLDER_M,
    UNDER_REACH_M,
    UNDER_RUN_ON,
    UNDER_SEA_M,
    UNDER_SEA_ON,
    UNDER_STUB_M,
    UNGRADED,
    XS_FIXED_GRADE,
    XS_FLOOR_M,
    XS_ON,
    _Decks,
    _merge_span_chains,
    _link_of_fixed,
    _never_raised,
    beside_fixed_span,
    _road_fids,
    approach_add,
    deck_capped,
    log,
    median_profile,
    set_widen,
    smooth_profile,
    street_runs,
    underpass_offsets,
)


class _UnderpassMixin:
    """_Planner's approaches and underpasses (split out of grading.py; _Planner inherits it)."""

    def _lift_approaches(self, key, plans, specs):
        """Raise, in place, the plans of the roads leading onto overpasses (APPROACH_ON). A road is raised where one of
        its ends is a bridge's abutment that must go up for the deck to clear the road under it (_end_lift); the raise
        falls off along the road (approach_add) and never comes nearer another road than its fill slope allows. Only
        what the plan's own line and the map around its ends say counts, the same in every cell that plans it (a long
        road is planned in each cell it passes; two cells raising the same road differently drew the road under a
        raised ground, 2026-10-20 I-20 in Greene County, AL). Major roads (APPROACH_SKIP) are never raised."""
        # every plan end at once: near the cell, and on a bridge end of its own cell (most touch none)
        pe = np.array([[p.xy[0][:2], p.xy[-1][:2]] for _f, p in plans], np.float64).reshape(-1, 2)
        ok = np.repeat([s is not None for s in specs], 2)       # (a motorway's or trunk's: over water only, below)
        touch = self._abut_ends(key, pe, ok).reshape(-1, 2)
        for i, (fid, p) in enumerate(plans):
            if not touch[i].any():
                continue
            add = None
            for at_end in (False, True):
                e = p.xy[-1] if at_end else p.xy[0]
                if not touch[i, int(at_end)]:
                    continue
                got = self._end_lift(e)
                if got is None:
                    continue
                if _never_raised(specs[i]) and not got[3] and not (len(got) > 5 and got[5]):
                    continue
                lift, others = got[0], got[1]
                zp = float(np.asarray(p.z, np.float64)[-1 if at_end else 0])
                if len(got) > 6 and got[5]:
                    # (crossings.ON: up to the solved end's height from the plan's own there)
                    lift = min(float(got[6]) - zp, APPROACH_MAX_LIFT)
                    if lift < APPROACH_MIN_LIFT:
                        continue
                elif len(got) > 4 and got[4] is not None:
                    # (a measured deck end: up to its height from the plan's own there, not from the ground under the
                    # point, which may lie down the cut the deck spans: Charles Street's approach over I-83 rose 2.2 m
                    # over its plan to a deck end measured at the street's height, 2026-10-29)
                    lift = min(float(got[4]) - zp, APPROACH_MAX_LIFT)
                    if lift < APPROACH_MIN_LIFT:
                        continue
                else:
                    # (the lift is over the ground at the bridge's end point (bridge_lifts): where the plan is higher
                    # there, filled up the embankment its map line ends down (fill_abutment), only what it still lacks.
                    # The fill and the whole lift on top took Kurna - Tuhala's deck end 6 m over the embankment it
                    # stands on, over Tallinna ringtee, 59.3364 N 24.8239 E, GitHub #684)
                    ela, elo, _ = self.frame.enu_to_geodetic(np.array([e[0]]), np.array([e[1]]), 0.0)
                    lift -= max(0.0, zp - float(np.asarray(self.elev(ela, elo), np.float64).reshape(-1)[0]))
                    if lift < APPROACH_MIN_LIFT:
                        continue
                measured = len(got) > 4 and got[4] is not None
                # (a solved crossing's approach walled beside the road at grade, as a measured one: the street splitting
                # into its two carriageways past Wezep's A28 deck, 19 m off its abutment, held the ramp's 1:2 fill to
                # 3.3 m of 4.7, its deck end 1 m under the ramp, a dip where the deck met it, the user 2026-11-01)
                solved = len(got) > 6 and bool(got[5])
                # (a motorway's or trunk's ramp at XS_FIXED_GRADE: a hump at 5 % on a freeway, Perth's Mitchell
                # Freeway rising 3-4 m to each of its street bridges, the user 2026-11-01)
                a = approach_add(p.xy, at_end, lift, p.half, others,
                                 MEASURED_SIDE_RUN if measured or solved else SLOPE_RUN,
                                 XS_FIXED_GRADE if solved and _never_raised(specs[i]) else None)
                if a is not None:
                    add = a if add is None else np.maximum(add, a)
            if add is not None:
                np_ = Plan(p.xy, p.z + add, p.half, p.abut, p.cut, p.road_half)
                np_.rid, np_.max_wall, np_.bank = p.rid, p.max_wall, getattr(p, "bank", None)
                np_.steep = getattr(p, "steep", False)
                set_widen(np_, getattr(p, "widen", None))
                plans[i] = (fid, np_)

    def _lower_underpasses(self, key, plans):
        """Lower, in place, the plans that cross under a short never-raised overpass (UNDER_*), and the roads joined
        to them (underpass_offsets), only within the cell's box (each plan is trimmed to it after). Solved on every
        ground road and span within UNDER_REACH_M of the box, from whichever cells they lie in, each road whole, its
        depth from its ground (_under_ground) rather than its plan: every cell gets the same dips. (On its own cell's roads alone, a road ending at a junction whose other roads
        lay only in the next cell was pinned there by one cell and lowered 3.5 m by the other; their copies met as a
        step on US-101 by Los Angeles' Union Station, 2026-10-27.)"""
        f64 = np.float64
        box = self._clip_box(key, UNDER_NEAR_M)
        if box is None:
            return
        xys = [np.asarray(p.xy, f64)[:, :2] for _fid, p in plans]
        lo = np.array(box[:2]) - UNDER_REACH_M
        hi = np.array(box[2:]) + UNDER_REACH_M
        # (the spans' cells a span's length further: one joined to another past the box is no single span)
        pad = UNDER_MAX_BRIDGE_M + 10.0
        cla, clo, _ = self.frame.enu_to_geodetic(np.array([lo[0] - pad, hi[0] + pad] * 2),
                                                 np.array([lo[1] - pad] * 2 + [hi[1] + pad] * 2), 0.0)
        cla, clo = np.asarray(cla, f64), np.asarray(clo, f64)
        keys = sorted(set(mapcells.cells_for_bbox(float(cla.min()), float(clo.min()), float(cla.max()),
                                                  float(clo.max()))) | {key}, key=str)
        bridges, ground, over, links = [], {}, [], []
        for k in keys:
            c = self._cell_enu(k)
            if UNDER_DECK_ON:               # (every deck, for the ground under it: _under_ground)
                over += [(line, half) for line, half in zip(c["bridges"], c.get("bridge_half", ())) if half > 0]
                over += [(line, railwaysmod.DECK_HALF_M) for line in c.get("rail_bridges", ())]
            for line, cls in zip(c["bridges"], c.get("bridge_cls", [None] * len(c["bridges"]))):
                if cls is None:
                    continue
                if not (UNDER_ANY_ON or cls in APPROACH_SKIP or cls.endswith("_link")):
                    continue
                if LINK_BESIDE_ON and _link_of_fixed(cls):
                    links.append(len(bridges))
                bridges.append((line, cls in APPROACH_SKIP, False))     # (a link's "<cls>_link": raised)
            # (a line's deck: never raised; straight between its ends whatever its length or what it joins, as
            # railways.deck_heights lays it: no single-span test)
            bridges += [(line, True, True) for line in c.get("rail_bridges", ())]
            for (line, _h, cls), (fid, j, sg, med) in zip(c["ground"], c["ground_ids"]):
                if cls in UNGRADED or len(line) < 2:
                    continue
                if (line.max(0) < lo).any() or (line.min(0) > hi).any():
                    continue
                got = ground.get((fid, j))      # (the longest copy of a part, as each cell has it)
                if got is None or len(line) > len(got[0]):
                    ground[(fid, j)] = (line, sg, med, cls)
        if not bridges or not ground:
            return
        if links:
            # (a link's deck beside its mainline's over the same road: never raised either, the road dips under both:
            # gradbase.LINK_BESIDE_*)
            fixed_lines = [b for b, fixed, rail in bridges if fixed and not rail]
            ground_lines = [g[0] for g in ground.values()]
            for n in links:
                if beside_fixed_span(bridges[n][0], fixed_lines, ground_lines):
                    bridges[n] = (bridges[n][0], True, False)
        # (single spans: neither end continues into another bridge part; deduplicated across cells)
        uniq, seen = [], set()
        for b, fixed, rail in bridges:
            k2 = (round(float(b[0, 0]), 1), round(float(b[0, 1]), 1), round(float(b[-1, 0]), 1), len(b))
            if k2 not in seen:
                seen.add(k2)
                uniq.append((b, fixed, rail))
        chained = set()
        if UNDER_CHAIN_ON:
            n0 = [len(b) for b, _f, _r in uniq]
            uniq = _merge_span_chains(uniq)
            chained = {i for i, (b, _f, _r) in enumerate(uniq) if len(b) not in n0}
        spans, rail_span = [], []
        ue = np.array([[b[0, :2], b[-1, :2]] for b, _f, _r in uniq], f64)        # (n, 2 ends, 2)
        stub = np.array([float(np.hypot(*np.diff(b[:, :2], axis=0).T).sum()) < UNDER_STUB_M for b, _f, _r in uniq])
        coarse = self._coarse(key)      # (there the railway rises over the road: railways.coarse_underpasses)
        for n, (b, fixed, rail) in enumerate(uniq):
            if rail and coarse:
                continue
            L = float(np.hypot(*np.diff(b, axis=0).T).sum())
            # (a carriageway's deck, any length, joined or not: its height is its profile's, the roads under it dip
            # to clear it)
            cw_deck = CW_ON and not rail and self._cw_z_at(b, b[:1]) is not None
            max_l = UNDER_CHAIN_MAX_M if n in chained else UNDER_MAX_BRIDGE_M
            if L < 2 * APPROACH_MIN_CROSS_M or (L > max_l and not rail and not cw_deck):
                continue
            d = np.hypot(ue[:, :, None, 0] - ue[n, None, None, :, 0], ue[:, :, None, 1] - ue[n, None, None, :, 1])
            d[n] = np.inf                       # (another part's end within 0.75 m of either of its ends: joined)
            if UNDER_CHAIN_ON and L >= UNDER_STUB_M:
                d[stub] = np.inf                # (a stub at its end, too short to span a road: no inner end)
            if rail or cw_deck or not (d < 0.75).any():
                spans.append((b, L, fixed, rail))
                rail_span.append(rail)
        if not spans:
            return
        gids = sorted(ground)
        lines = [ground[g][0] for g in gids]
        crs = roadsmod._crossings_by([b for b, _L, _f, _r in spans], lines)
        ez = {}

        def deck_ends(j):           # (an overpass's abutments' ground, raised when it is: as high as its ramps could
            got = ez.get(j)         # take them; asked only for one a road crosses under: every span's took 2 s a
            if got is None:         # Canton cell, 2026-10-25)
                b, _L, fixed, rail = spans[j]
                ends = np.array([b[0], b[-1]], f64)
                ela, elo, _ = self.frame.enu_to_geodetic(ends[:, 0], ends[:, 1], 0.0)
                got = np.asarray(self.elev(np.asarray(ela), np.asarray(elo)), f64).reshape(2).copy()
                if rail:            # (the track's level beyond its ends, as railways.profile lays the deck)
                    def ground_xy(q):
                        qla, qlo, _ = self.frame.enu_to_geodetic(q[:, 0], q[:, 1], 0.0)
                        return np.asarray(self.elev(np.asarray(qla), np.asarray(qlo)), f64)
                    for e, lv in enumerate(railwaysmod.approach_levels(b[:, :2], ground_xy)):
                        if lv is not None:
                            got[e] = max(got[e], lv[0])
                if not fixed or (XS_ON and not rail and self._xsolve(b) is not None):
                    got[0] += self._end_raised(b[0])
                    got[1] += self._end_raised(b[-1])
                if not rail:        # (or on its embankment's top, where its line ends down the slope: #684's fill)
                    got = np.maximum(got, [self._end_filled(b, False), self._end_filled(b, True)])
                ez[j] = got
            return got
        cands, decks, cand_xs, cand_cw = [], [], [], []
        for i, cr in enumerate(crs):
            if not cr:
                continue
            xy = lines[i]
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            ab = np.diff(xy, axis=0)
            for c in cr:
                bi, arc_b, pt = c[0], c[1], np.asarray(c[2], f64)
                b, L, fixed, rail = spans[bi]
                if arc_b < APPROACH_MIN_CROSS_M or arc_b > L - APPROACH_MIN_CROSS_M:
                    continue                    # a junction at the bridge's end
                corridor = np.clip([arc_b - roadsmod.DECK_CLEAR_REACH_M, arc_b + roadsmod.DECK_CLEAR_REACH_M], 0.0, L)
                cz = None
                if CW_ON and not rail_span[bi]:
                    sb = np.r_[0.0, np.cumsum(np.hypot(*np.diff(b, axis=0).T))]
                    cpts = np.stack([np.interp(corridor, sb, b[:, 0]), np.interp(corridor, sb, b[:, 1])], axis=1)
                    cz = self._cw_z_at(b, np.r_[cpts, pt[None]])
                if cz is not None:
                    deck = float(np.min(cz))
                else:
                    ze = deck_ends(bi)
                    deck = float(np.min(ze[0] + (ze[1] - ze[0]) * corridor / L))
                # where along this road (its nearest vertex segment)
                t = np.clip(((pt - xy[:-1]) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
                j = int(np.argmin(np.linalg.norm(xy[:-1] + t[:, None] * ab - pt, axis=1)))
                # (the deck's own clearance: its depth scales with its span, roads.deck_depth; the full-depth
                # UNDER_CLEAR_M dug the road out under a short overpass that sits lower)
                # (a rail deck's height is its bed: its own depth under the ballast's top, railways.DECK_D)
                depth = railwaysmod.DECK_D - railwaysmod.BALLAST_H if rail else \
                    roadsmod.deck_depth(roadsmod.geom_span([b]))
                clear = roadsmod.deck_road_clear(depth) + ROAD_LIFT
                cands.append((i, float(s[j] + t[j] * (s[j + 1] - s[j])), deck - clear))
                decks.append((deck, fixed))
                cand_xs.append((bi, pt))
                cand_cw.append(None if cz is None else deck - (CW_HEADROOM_M + roadsmod.deck_depth(
                    roadsmod.geom_span([b])) + ROAD_LIFT))
        if not cands:
            return
        # (over as _Decks: its boxes and segments made once for every road's test, not per deck per road)
        zc = self._under_ground(lines, [ground[g][1:3] for g in gids], cands, None if coarse else _Decks(over))
        dem = getattr(self, "dem", None) or getattr(self.md, "dem", None)
        lidar = UNDER_LIDAR_ON and not coarse and dem is not None and hasattr(dem, "coarse_at")
        requirements, xs_floor = [], []
        floor = seatiles.SEA_E + UNDER_SEA_M
        for (i, sc, top), (deck, fixed), z, (bi, pt), top_cw in zip(cands, decks, zc, cand_xs, cand_cw):
            if top_cw is not None and _cw.family(ground[gids[i]][3]) is not None:
                continue                        # (a carriageway under a carriageway's deck: solved together)
            if top_cw is not None:
                # (a carriageway's deck, its height solved: the road under at its trusted height (beside the deck's
                # embankment, not lidar's fill under it), lower only where the deck needs it)
                u = _xs.dir_at(lines[i], pt)
                tl = _xs.trusted_under(pt, u, 8.0, lambda p_: self._z_raw(p_))
                tl = z if tl is None else min(tl, z)
                target = max(min(tl, top_cw), z - UNDER_MAX_DROP_M)
                if UNDER_SEA_ON and z >= floor:
                    target = max(target, floor)
                if os.environ.get("BNE_CW_DEBUG"):
                    log.debug("grading: CWDIP at %s: road %.1f trusted %.1f deck %.1f -> %.1f", np.round(pt), z, tl,
                              deck, target)
                if target < z - 0.1:
                    requirements.append((i, sc, z - target))
                    xs_floor.append((np.asarray(pt, f64), target, _road_fids(i, lines, [ground[g][3] for g in gids],
                                                                            gids), 8.0 + _xs.UNDER_GAP_M))
                continue
            sol = self._xsolve(spans[bi][0]) if XS_ON and not rail_span[bi] else None
            if sol is not None and sol["cross"]:
                # (crossings.ON: the road's target from the solve: its trusted height, less its share of the
                # separation; lidar's fill under the deck taken off whatever the deck does)
                k = int(np.argmin([np.hypot(*(q - pt)) for q, _t in sol["cross"]]))
                if np.hypot(*(sol["cross"][k][0] - pt)) <= 5.0:
                    # (off lidar, no higher than what the deck as its ramps raised it clears: one held down by the
                    # roads beside its ramps leaves the rest to the road under, as before. On lidar, as #504: the deck
                    # rises over the road instead, whose ground the lidar has)
                    target = sol["cross"][k][1] if lidar else min(sol["cross"][k][1], top)
                    target = max(target, z - UNDER_MAX_DROP_M)
                    if UNDER_SEA_ON and z >= floor:
                        target = max(target, floor)
                    if target < z - 0.1:
                        requirements.append((i, sc, z - target))
                        xs_floor.append((np.asarray(pt, f64), target, _road_fids(i, lines, [ground[g][3] for g in gids],
                                                                                gids), sol["reach"]))
                    continue
            if lidar and not fixed and deck < z + UNDER_LIDAR_DECK_M:
                continue                        # (the ends' ground is not the deck: the road keeps its own, the deck
                                                # rises; a never-raised deck, a railway's or a motorway's, cannot)
            target = max(top, z - UNDER_MAX_DROP_M)
            if UNDER_SEA_ON and z >= floor:
                target = max(target, floor)     # (not under the sea's surface: its water tiles would flood it)
            if target < z - 0.1:
                requirements.append((i, sc, z - target))
        if not requirements:
            return
        # each plan on its road's line (of its feature's parts, the nearest)
        by_fid = {}
        for n, (fid, _j) in enumerate(gids):
            by_fid.setdefault(fid, []).append(n)
        queries, which = [], []
        inside = []
        for n, (fid, p) in enumerate(plans):
            cand = by_fid.get(fid)
            xy = xys[n]
            near = (xy[:, 0] >= box[0]) & (xy[:, 1] >= box[1]) & (xy[:, 0] <= box[2]) & (xy[:, 1] <= box[3])
            if not cand or not near.any():
                continue
            mid = xy[near][int(near.sum()) // 2]
            best = min(cand, key=lambda m: float(np.min(np.hypot(*(lines[m] - mid).T))))
            queries.append((best, xy[near]))
            which.append(n)
            inside.append(near)
        if not queries:
            return
        # on coarse elevation a street's dip stays on its run (UNDER_RUN_*: street_runs), each run alone. (Every cell
        # sees the same runs: a dip reaches at most UNDER_FLAT_M + its longest ramp (~221 m) along its street, so the
        # street's nodes it can depend on lie within UNDER_REACH_M of every cell's box the dip reaches, as do all the
        # lines meeting there)
        net, runs = requirements, []
        if coarse and UNDER_RUN_ON:
            net, runs = street_runs(lines, [ground[g][3] for g in gids], requirements)
        offsets = underpass_offsets(lines, net, queries)
        for rxy, members, reqs in runs:
            on = [k for k, (ln, _q) in enumerate(queries) if ln in members]
            if on:
                for k, got in zip(on, underpass_offsets([rxy], reqs, [(0, queries[k][1]) for k in on])):
                    offsets[k] = np.minimum(offsets[k], got)
        for n, near, got in zip(which, inside, offsets):
            if not np.any(got < -1e-8):
                continue
            fid, p = plans[n]
            offset = np.zeros(len(near))        # (past the box: the plan's piece there is another cell's)
            offset[near] = got
            z0 = np.asarray(p.z, f64)
            z1 = z0 + offset
            if xs_floor:
                # (crossings.ON: a solved crossing's target is a height, not a depth: the dip, made as a depth from the
                # road's ground there, went on 1.5 m under it where the plan beside lidar's fill was already lower,
                # Wezep's street 10 m from the A28. Nothing near one goes under it, nor over its own height)
                # The road itself is held at its target across the fill (the dip, a depth from the ground at the
                # crossing, left the fill's far shoulders standing), rising away at UNDER_GRADE past it.
                xy = xys[n]
                # (the floor: the lowest target near, not each in turn: a street under a motorway and its slip road
                # side by side was held at the slip road's target, 1.3 m over the motorway's, and the motorway's deck
                # humped over it, Perth's Mitchell Freeway at Roe Street, the user 2026-11-01)
                low_t = np.full(len(z1), np.inf)
                for q, t, qfids, reach in xs_floor:
                    dq = np.hypot(xy[:, 0] - q[0], xy[:, 1] - q[1])
                    if dq.min() > XS_FLOOR_M:
                        continue
                    if fid in qfids and not (coarse and UNDER_RUN_ON):
                        # (and the parts continuing it, of its class: a dual carriageway's halves past the deck. Not on
                        # coarse elevation: there the dip stays as deep as fits on its street's run, #655's rather a
                        # lower bridge than a sunken neighbourhood; held at the solve's target it went on 8 m deep)
                        cap = t + np.maximum(dq - reach, 0.0) * UNDER_GRADE
                        z1 = np.where(dq <= XS_FLOOR_M, np.minimum(z1, np.maximum(cap, np.minimum(z0, t))), z1)
                    low_t = np.where(dq <= XS_FLOOR_M, np.minimum(low_t, t), low_t)
                z1 = np.where(np.isfinite(low_t), np.maximum(z1, np.minimum(z0, low_t)), z1)
            if UNDER_SEA_ON:                    # (every point: a dip's flat and ramps, a joined road's share)
                z1 = np.where(z0 >= UNDER_POLDER_M, np.maximum(z1, np.minimum(z0, floor)), z1)
            np_ = Plan(p.xy, z1, p.half, p.abut, p.cut, p.road_half)
            np_.rid, np_.max_wall, np_.bank = p.rid, p.max_wall, getattr(p, "bank", None)
            np_.steep = getattr(p, "steep", False)
            set_widen(np_, getattr(p, "widen", None))
            plans[n] = (fid, np_)

    def _under_ground(self, lines, smooth, cands, decks=None):
        """The heights of ground roads where they cross under a bridge, cands [(line, arc m, _)]: their ground along
        them, read beside the decks over them (decks: deck_capped's, None: as it is), median-filtered and smoothed as
        their plans are (_plan: smooth (sigma, median width) per line), at the crossing. A pure function of the road
        and the ground (its plan's own height is only in the cells planning it)."""
        f64 = np.float64
        pts, spans = [], []
        for i, sc, _d in cands:
            sg, med = smooth[i]
            xy = lines[i]
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            k = int(math.ceil(3.0 * sg / PROFILE_STEP))
            raw = sc + PROFILE_STEP * np.arange(-k, k + 1)
            a = np.clip(raw, 0.0, s[-1])
            pts.append(np.stack([np.interp(a, s, xy[:, 0]), np.interp(a, s, xy[:, 1])], axis=1))
            ext = None
            if decks:
                # (and on straight past the part's ends, for deck_capped's ground beside a deck near one: Järve tee's
                # part ends 12 m past Tartu mnt's deck, still on the lidar's hump)
                d0 = (xy[0] - xy[1]) / max(float(np.hypot(*(xy[0] - xy[1]))), 1e-9)
                d1 = (xy[-1] - xy[-2]) / max(float(np.hypot(*(xy[-1] - xy[-2]))), 1e-9)
                ext = pts[-1] + np.where(raw < 0.0, -raw, 0.0)[:, None] * d0 \
                    + np.where(raw > s[-1], raw - s[-1], 0.0)[:, None] * d1
            spans.append((k, sg, med, raw, a, ext))
        allp = np.concatenate(pts)
        la, lo, _ = self.frame.enu_to_geodetic(allp[:, 0], allp[:, 1], 0.0)
        z = np.asarray(self.elev(np.asarray(la, f64), np.asarray(lo, f64)), f64).ravel()
        zp = None                   # (the ground past the parts' ends, at once; capped on the parts alone without it)
        past = [raw != a for _k, _sg, _med, raw, a, _e in spans]
        if decks and any(m.any() for m in past):
            try:
                ep = np.concatenate([sp[5][m] for sp, m in zip(spans, past)])
                ela, elo, _ = self.frame.enu_to_geodetic(ep[:, 0], ep[:, 1], 0.0)
                zp = np.asarray(self.elev(np.asarray(ela, f64), np.asarray(elo, f64)), f64).ravel()
            except Exception as e:
                log.debug("grading: ground past the ends under decks: %s: %s", type(e).__name__, e)
        out, k0, j0 = [], 0, 0
        for (k, sg, med, raw, a, ext), p, m in zip(spans, pts, past):
            zz = z[k0:k0 + 2 * k + 1]
            k0 += 2 * k + 1
            if decks:
                zx, q = zz, p
                if zp is not None and m.any():
                    zx, q = zz.copy(), ext
                    zx[m] = zp[j0:j0 + int(m.sum())]
                j0 += int(m.sum())
                capped = deck_capped(q, zx, decks)
                if capped is not zx:            # (the clipped samples at most the capped ground there)
                    zz = np.minimum(zz, np.interp(a, raw, capped))
            out.append(float(smooth_profile(median_profile(zz, PROFILE_STEP, med), PROFILE_STEP, sg)[k]))
        return out
