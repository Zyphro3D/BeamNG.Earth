"""Road grading: bridge ends and abutments (split out of grading.py's _Planner): fills at the ends, end lifts, the
decks near a cell, the cross-section solve and water lifts.
"""
import math
import threading
from collections import OrderedDict
import numpy as np
import mapcells
import railways as railwaysmod
import roads as roadsmod
import seatiles
import carriageways as _cw
from gradbase import (
    ABUT_FAR_M,
    ABUT_FILL_ON,
    APPROACH_CELL_M,
    APPROACH_MAX_LIFT,
    APPROACH_MAX_RUN,
    APPROACH_MIN_CROSS_M,
    APPROACH_MIN_LIFT,
    APPROACH_SKIP,
    APPROACH_TOUCH_M,
    CW_ON,
    DIRT_MEDIAN_M,
    ENU_CACHE_MAX,
    MEASURED_SIDE_RUN,
    MEDIAN_M,
    PROFILE_STEP,
    ROAD_LIFT,
    SLOPE_RUN,
    LINK_BESIDE_ON,
    TWIN_M,
    TWIN_ON,
    TWIN_SHARE,
    UNDER_DECK_ON,
    UNDER_DECK_PAD_M,
    UNDER_MAX_BRIDGE_M,
    UNDER_MAX_DROP_M,
    UNDER_SEA_M,
    UNDER_SEA_ON,
    UNGRADED,
    WALK_SPAN_CLASSES,
    WALK_SPAN_HALF_M,
    WATER_CURVE_M,
    WATER_LIFT_ON,
    WATER_MAX_LIFT,
    XS_ON,
    _Decks,
    _is_link,
    _link_of_fixed,
    beside_fixed_span,
    _seg_dist,
    approach_add,
    bridge_chain,
    bridge_chain_ends_free,
    bridge_lifts,
    fill_abutment,
    log,
    profile_sigma,
    shoulder,
)


class _EndsMixin:
    """_Planner's bridge ends and abutments (split out of grading.py; _Planner inherits it)."""

    def _abut_ends(self, key, pe, ok=None):
        """Which of the points pe (n, 2) (road ends; ok: those to look at) near cell key are within APPROACH_TOUCH_M
        of a bridge's end in their own cell."""
        la, lo, _ = self.frame.enu_to_geodetic(pe[:, 0], pe[:, 1], 0.0)
        la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
        lon0, lat0, lon1, lat1 = (float(v) for v in key.bounds())
        dla = np.maximum(np.maximum(lat0 - la, la - lat1), 0.0) * 111320.0
        dlo = np.maximum(np.maximum(lon0 - lo, lo - lon1), 0.0) * 111320.0 * np.cos(np.radians(la))
        touch = np.hypot(dla, dlo) <= APPROACH_CELL_M
        if ok is not None:
            touch &= ok
        homes = {}
        tj = np.flatnonzero(touch)
        for j, home in zip(tj.tolist(), mapcells.cells_of(la[tj], lo[tj])):
            homes.setdefault(home, []).append(j)
        for home, js in homes.items():
            ends = self._cell_enu(home, ground=False)["ends"]
            js = np.asarray(js)
            if not len(ends):
                touch[js] = False
                continue
            d = np.hypot(pe[js, None, 0] - ends[None, :, 0], pe[js, None, 1] - ends[None, :, 1])
            touch[js] = (d <= APPROACH_TOUCH_M).any(axis=1)
        return touch

    def _end_raised(self, xy):
        """m the abutment at bridge end xy is raised by its approach ramps (_lift_approaches: the most any road leading
        onto it rises there, its fill slope kept clear of the roads beside), 0 when it is not raised."""
        got = self._end_lift(xy)
        if got is None:
            return 0.0
        lift, others, approaches = got[:3]
        best = 0.0
        for line, half in approaches:
            at_end = bool(np.hypot(*(line[-1] - xy)) <= np.hypot(*(line[0] - xy)))
            a = approach_add(line, at_end, lift, half, others,
                             MEASURED_SIDE_RUN if (len(got) > 4 and got[4] is not None) or (len(got) > 6 and got[5])
                             else SLOPE_RUN)
            if a is not None:
                best = max(best, float(a[-1] if at_end else a[0]))
        return best

    def _end_filled(self, line, at_end):
        """m: the ground at a bridge's first (at_end False) or last end as fill_abutment lays its approach there, read
        straight on past the end (the approach roads' own plans are not made yet): its embankment's top where the
        line ends down the slope, else the ground at the end. -inf when ABUT_FILL_ON is off."""
        if not ABUT_FILL_ON:
            return -np.inf
        b = np.asarray(line, np.float64)[:, :2]
        p, q = (b[-1], b[-2]) if at_end else (b[0], b[1])
        u = p - q
        n = float(np.hypot(*u))
        if n < 1e-9:
            return -np.inf
        d = np.arange(0.0, ABUT_FAR_M + 1e-6, PROFILE_STEP)
        pts = p[None] + d[:, None] * (u / n)[None]
        la, lo, _ = self.frame.enu_to_geodetic(pts[:, 0], pts[:, 1], 0.0)
        try:
            z = np.asarray(self.elev(np.asarray(la), np.asarray(lo)), np.float64).reshape(-1)
        except Exception as e:      # (ground past the end unread: the end's own estimate, as before)
            log.debug("grading: fill past bridge end %s: %s: %s", np.round(p), type(e).__name__, e)
            return -np.inf
        return float(fill_abutment(z, False)[0])

    def _end_lift(self, xy):
        """(lift m, other road lines [(xy (k, 2), half width)], the roads leading onto it [(xy, half width)]) when the
        road end xy is a bridge abutment to raise, else None. A pure function of the map around the bridge (its cells,
        fetched as needed), cached per point."""
        k = (round(float(xy[0]) * 2), round(float(xy[1]) * 2))
        with self.lock:
            if not hasattr(self, "end_cache"):
                self.end_cache = {}
            if k in self.end_cache:
                return self.end_cache[k]
        res = None
        try:
            res = self._end_lift_calc(np.asarray(xy, np.float64))
        except Exception as e:      # no ramp rather than no grading
            log.debug("grading: approach at %s: %s: %s", k, type(e).__name__, e)
        with self.lock:
            return self.end_cache.setdefault(k, res)

    def _cell_enu(self, k, ground=True):
        """Cell k's road parts in the anchor frame, made once: {"bridges": [line (k, 2)], "ends": (n, 2) bridge end
        points, "end_of": [bridge index per end], "ground": [(line, half width, cls)], "ground_ids":
        [(feature id, part index, profile sigma, median width)] per ground part} (the longest copy of each part).
        With ground False, only the bridges (most plan ends touch none: their cells' ground roads are not needed).
        Kept for the ENU_CACHE_MAX cells used last; a cell another thread is making is waited for (cleared whole at
        128 cells, while the planning threads made the same cells at once, a 36-tile downtown Los Angeles set made
        318 cells for 242, ~125 ms each, 2026-10-27)."""
        key = (str(k), ground)
        with self.lock:
            if not hasattr(self, "enu_cache"):
                self.enu_cache, self.enu_making = OrderedDict(), {}
            while True:
                got = self.enu_cache.get(key) or (self.enu_cache.get((str(k), True)) if not ground else None)
                if got is not None:
                    self.enu_cache.move_to_end(key if key in self.enu_cache else (str(k), True))
                    return got
                ev = self.enu_making.get(key)
                if ev is None:
                    ev = self.enu_making[key] = threading.Event()
                    break
                self.lock.release()
                try:
                    ev.wait(60.0)
                finally:
                    self.lock.acquire()
                if key not in self.enu_cache and self.enu_making.get(key) is ev:    # (its maker gave up: make it here)
                    break
        try:
            out = self._cell_enu_calc(k, ground)
            with self.lock:
                self.enu_cache[key] = out
                self.enu_cache.move_to_end(key)
                while len(self.enu_cache) > ENU_CACHE_MAX:
                    self.enu_cache.popitem(last=False)
            return out
        finally:
            with self.lock:
                if self.enu_making.get(key) is ev:
                    del self.enu_making[key]
            ev.set()

    def _cell_enu_calc(self, k, ground):
        parts = {}
        f = self.md.cell(k, layers=("roads",), kind="demand").layers.get("roads")
        span_ll = []
        if f is not None and len(f):
            bs = roadsmod.bridge_specs(f)
            rs = roadsmod.road_specs(f, None, self.widths) if ground else [None] * len(f)
            classes = f.attrs.get("class")
            subtypes = f.attrs.get("subtype")
            for i in range(len(f)):
                if bs[i] is None and rs[i] is None:
                    cls = str(classes[i]) if classes is not None else ""
                    sub = str(subtypes[i]) if subtypes is not None else "road"
                    if sub == "road" and cls in WALK_SPAN_CLASSES:
                        for ring in roadsmod.bridge_parts(f, i):
                            ring = np.asarray(ring, np.float64)
                            if len(ring) >= 2:
                                span_ll.append(ring)
                    continue
                fid = str(f.ids[i])
                # (a bridge part's level: OSM's layer as Overture's level rules, 1 without one; stacked decks' order)
                blv = roadsmod.bridge_levels(f, i) if bs[i] is not None else []
                for kind, sp, ps in (("bridge", bs[i], roadsmod.bridge_parts(f, i) if bs[i] is not None else ()),
                                     ("ground", rs[i], roadsmod.ground_parts(f, i) if rs[i] is not None else ())):
                    for j, ring in enumerate(ps):
                        ring = np.asarray(ring, np.float64)
                        kk = (fid, kind, j)
                        lv = float(blv[j]) if kind == "bridge" and j < len(blv) else 0.0
                        if len(ring) >= 2 and (kk not in parts or len(ring) > len(parts[kk][0])):
                            parts[kk] = (ring, sp, lv)
        # the rail bridges (railways.DRAWN lines' bridge pieces): the roads under them dip as under a road overpass
        # (_lower_underpasses), nothing else (no ramps, no abutments: a line's deck is never raised for a road).
        # (GitHub #311 / #345 / #364: the track lay across the road under a rail bridge)
        rail, rail_g = [], []
        if f is not None and len(f) and f.attrs.get("subtype") is not None:
            import railways
            sub = f.attrs["subtype"]
            for i in range(len(f)):
                got = railways.drawn(f, i, bridges=True) if str(sub[i]) == "rail" else None
                if got is None:
                    continue
                for kind, rings in railways._pieces(f, i):
                    if kind == "bridge":
                        rail += [np.asarray(r, np.float64) for r in rings if len(r) >= 2]
                    elif kind == "ground" and ground and got[0] != "tram":
                        # (and the tracks on the ground: a road bridge's approaches rise over them as over a road,
                        # GitHub #324)
                        rail_g += [np.asarray(r, np.float64) for r in rings if len(r) >= 2]
        bridges, ground_l, ends, end_of, bcls, ground_ids, bids, blevels, bhalf = [], [], [], [], [], [], [], [], []
        items = sorted(parts.items())
        if items:               # (the frame's conversion once for every part: it is per call)
            allr = np.concatenate([ring for _k, (ring, _sp, _lv) in items])
            ae, an, _ = self.frame.geodetic_to_enu(allr[:, 1], allr[:, 0], 0.0)
            axy = np.stack([np.asarray(ae, np.float64), np.asarray(an, np.float64)], axis=1)
        k0 = 0
        for (fid, kind, j), (ring, sp, lv) in items:
            line = axy[k0:k0 + len(ring)]
            k0 += len(ring)
            if kind == "bridge":
                end_of += [len(bridges), len(bridges)]
                ends += [line[0], line[-1]]
                bridges.append(line)
                bids.append(fid)
                blevels.append(lv)
                # (one half width a bridge: two merged changes each appended it, and bridge i got bridge i // 2's)
                bhalf.append(float(getattr(sp, "width", None) or 8.0) / 2)
                cls = getattr(sp, "cls", None)
                bcls.append(cls + "_link" if cls is not None and _is_link(sp) else cls)
            else:
                ground_l.append((line, sp.width / 2 + shoulder(sp), sp.cls))
                ground_ids.append((fid, j, profile_sigma(sp), MEDIAN_M if sp.kind in roadsmod.BUILT else DIRT_MEDIAN_M))
        rail_xy, rail_gxy = [], []
        if rail or rail_g:
            allr = np.concatenate(rail + rail_g)
            re_, rn, _ = self.frame.geodetic_to_enu(allr[:, 1], allr[:, 0], 0.0)
            rxy = np.stack([np.asarray(re_, np.float64), np.asarray(rn, np.float64)], axis=1)
            k0 = 0
            for ring, out_ in [(r, rail_xy) for r in rail] + [(r, rail_gxy) for r in rail_g]:
                out_.append(rxy[k0:k0 + len(ring)])
                k0 += len(ring)
        span_xy = []
        if span_ll:
            alls = np.concatenate(span_ll)
            se, sn, _ = self.frame.geodetic_to_enu(alls[:, 1], alls[:, 0], 0.0)
            sxy = np.stack([np.asarray(se, np.float64), np.asarray(sn, np.float64)], axis=1)
            k0 = 0
            for ring in span_ll:
                span_xy.append(sxy[k0:k0 + len(ring)])
                k0 += len(ring)
        return {"bridges": bridges, "bridge_cls": bcls, "bridge_ids": bids, "bridge_levels": blevels,
                "bridge_half": bhalf, "ground": ground_l, "ground_ids": ground_ids,
                "end_of": end_of, "span_bridges": span_xy,
                "ends": np.asarray(ends, np.float64).reshape(-1, 2), "rail_bridges": rail_xy, "rail_ground": rail_gxy}

    def _decks_ll(self, lonlat):
        """_decks_near of the cell holding the middle of a lon/lat part (a pin height's plan: the cell's own plans
        read the same decks)."""
        if not UNDER_DECK_ON or lonlat is None or not len(lonlat):
            return []
        mid = np.asarray(lonlat, np.float64)[len(lonlat) // 2]
        return self._decks_near(mapcells.cell_of(float(mid[1]), float(mid[0])))

    def _decks_near(self, key):
        """[(line (k, 2), half width)] of the road and rail decks within UNDER_DECK_PAD_M of cell key (deck_capped's),
        each once, made once per cell; [] with UNDER_DECK_ON off."""
        if not UNDER_DECK_ON or key is None:
            return []
        memo = self.__dict__.setdefault("_decks_memo", {})
        with self.lock:
            got = memo.get(str(key))
        if got is not None:
            return got
        lon0, lat0, lon1, lat1 = (float(v) for v in key.bounds())
        pla = UNDER_DECK_PAD_M / 111320.0
        plo = pla / max(0.05, math.cos(math.radians((lat0 + lat1) / 2)))
        keys = sorted(set(mapcells.cells_for_bbox(lat0 - pla, lon0 - plo, lat1 + pla, lon1 + plo)) | {key}, key=str)
        out, seen, short = [], set(), False
        for k in keys:
            try:
                # (its bridges only: the decks and their widths are in both tables; the ground roads of every cell
                # within UNDER_DECK_PAD_M were ~0.35 s of each Manhattan road tile, 2026-11-03)
                c = self._cell_enu(k, ground=False)
            except Exception as e:      # (that cell's decks left out rather than no plans; not kept: asked again)
                log.debug("grading: decks of %s: %s: %s", k, type(e).__name__, e)
                short = True
                continue
            halves = c.get("bridge_half") or [0.0] * len(c["bridges"])
            for line, half in (list(zip(c["bridges"], halves))
                               + [(r, railwaysmod.DECK_HALF_M) for r in c.get("rail_bridges", ())]
                               + [(r, WALK_SPAN_HALF_M) for r in c.get("span_bridges", ())]):
                if len(line) < 2 or half <= 0:
                    continue
                k2 = (round(float(line[0, 0]), 1), round(float(line[0, 1]), 1), round(float(line[-1, 0]), 1), len(line))
                if k2 not in seen:
                    seen.add(k2)
                    out.append((line, half))
        out = _Decks(out)
        if not short:
            with self.lock:
                memo[str(key)] = out
        return out

    def _ground_table(self, keys, cs):
        """The ground road parts of the cells keys (cs: their _cell_enu), each once (the first copy, by its first point
        and length), in their order: (lines, half widths, boxes' lo (n, 2), hi (n, 2), first points, last points).
        Kept per cell set (every bridge end near them asks: a loop over each part per end was most of
        _lift_approaches, 2026-10-25)."""
        mk = tuple(str(k) for k in keys)
        memo = self.__dict__.setdefault("_ground_tables", {})
        with self.lock:
            got = memo.get(mk)
        if got is not None and all(a is b for a, b in zip(got[0], cs)):
            return got[1]
        lines, halves, seen = [], [], set()
        for c in cs:
            for line, half, _cls in c["ground"]:
                key = (round(float(line[0, 0]), 1), round(float(line[0, 1]), 1), len(line))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(line)
                halves.append(half)
        f64 = np.float64
        if lines:
            tab = (lines, halves, np.array([ln.min(0) for ln in lines], f64), np.array([ln.max(0) for ln in lines], f64),
                   np.array([ln[0] for ln in lines], f64), np.array([ln[-1] for ln in lines], f64))
        else:
            z = np.zeros((0, 2), f64)
            tab = (lines, halves, z, z, z, z)
        with self.lock:
            if len(memo) > 64:
                memo.clear()
            memo[mk] = (list(cs), tab)
        return tab

    def _end_lift_calc(self, xy):
        """_end_lift_raw with the roads beside the ramp that lead onto another abutment near it, itself raised at least
        TWIN_SHARE as much, left out of its caps: the two rise together (a divided road's two carriageways onto their
        decks, side by side 10 m apart: each capped the other's ramp to nothing, both abutments 3.3 m under the decks'
        measured ends, Springfield's Old Keene Mill Road, 2026-10-31, approachlidar.py)."""
        got = self._end_lift_raw(xy)
        if got is None or not TWIN_ON or not got[1]:
            return got
        la, lo, _ = self.frame.enu_to_geodetic(np.array([xy[0]]), np.array([xy[1]]), 0.0)
        pad = (TWIN_M + APPROACH_TOUCH_M) / 111320.0
        keys = sorted(set(mapcells.cells_for_points(la, lo, pad, pad / max(0.05, math.cos(math.radians(float(la[0])))))),
                      key=str)
        ends = [np.asarray(self._cell_enu(k, ground=False)["ends"], np.float64).reshape(-1, 2) for k in keys]
        ends = np.unique(np.round(np.concatenate(ends), 3), axis=0) if ends else np.zeros((0, 2))
        if not len(ends):
            return got
        d = np.hypot(ends[:, 0] - xy[0], ends[:, 1] - xy[1])
        cand = ends[(d > APPROACH_TOUCH_M) & (d <= TWIN_M)]
        if not len(cand):
            return got
        keep, lifts = [], {}
        for line, half in got[1]:
            line_ = np.asarray(line, np.float64)
            twin = False
            for p in (line_[0], line_[-1]):
                dd = np.hypot(cand[:, 0] - p[0], cand[:, 1] - p[1])
                k = int(np.argmin(dd))
                if dd[k] <= APPROACH_TOUCH_M:
                    kk = (round(float(cand[k, 0]), 2), round(float(cand[k, 1]), 2))
                    if kk not in lifts:
                        other = self._end_lift_raw(cand[k])
                        lifts[kk] = 0.0 if other is None else float(other[0])
                    twin = twin or lifts[kk] >= TWIN_SHARE * float(got[0])
            if not twin:
                keep.append((line, half))
        return (got[0], keep) + tuple(got[2:])

    def _xsolve(self, bridge):
        """crossings.ON: the bridge (line (k, 2), a single span no other bridge part joins) and the roads under it
        solved together (crossings.solve), cached per bridge: {"ends": (2, 2) its end points, "raise": (2,) m each end
        is raised over its own ground (its trusted height, filled up to its road's past it, + the lift), "lift": m,
        "cross": [(point (2,), the road's target height there)]}. None where it does not apply (a chain, a long or a
        rail bridge, no road under it, no trusted ground), and the rules as before decide."""
        bridge = np.asarray(bridge, np.float64)[:, :2]
        k = (round(float(bridge[0, 0]), 1), round(float(bridge[0, 1]), 1), round(float(bridge[-1, 0]), 1),
             round(float(bridge[-1, 1]), 1), len(bridge))
        with self.lock:
            memo = self.__dict__.setdefault("_xsolve_cache", {})
            if k in memo:
                return memo[k]
        res = None
        try:
            res = self._xsolve_calc(bridge)
        except Exception as e:      # the rules as before rather than no grading
            log.debug("grading: crossing solve at %s: %s: %s", k, type(e).__name__, e)
        with self.lock:
            if len(memo) > 4096:
                memo.clear()
            return memo.setdefault(k, res)

    def _xsolve_calc(self, bridge):
        import crossings as xs
        f64 = np.float64
        L = float(np.hypot(*np.diff(bridge, axis=0).T).sum())
        if L < 2 * APPROACH_MIN_CROSS_M or L > UNDER_MAX_BRIDGE_M:
            return None
        la, lo, _ = self.frame.enu_to_geodetic(bridge[:, 0], bridge[:, 1], 0.0)
        la, lo = np.asarray(la, f64), np.asarray(lo, f64)
        pad = (xs.TIE_M + 20.0) / 111320.0
        keys = sorted(set(mapcells.cells_for_points(la, lo, pad, pad / max(0.05, math.cos(math.radians(float(la[0]))))))
                      | {mapcells.cell_of(float(la[0]), float(lo[0]))}, key=str)
        cs = [self._cell_enu(kk) for kk in keys]
        bridges, cls = [], None
        for c in cs:
            for line, bc in zip(c["bridges"], c.get("bridge_cls", [None] * len(c["bridges"]))):
                bridges.append(line)
                if cls is None and len(line) == len(bridge) and np.allclose(line[:, :2], bridge, atol=0.05):
                    cls = bc
        if cls is None or bridge_chain(bridge, bridges):
            return None                 # (not a road bridge we know, or a chain: stage 1 is single spans)
        if CW_ON and _cw.family(cls) is not None and self._cw_z_at(bridge, bridge[:1]) is not None:
            return None                 # (a carriageway's: its profile decides, carriageways.py)
        # the ground roads near it, each once, with their class (paths apart: they neither tie nor weigh much)
        lo_b, hi_b = bridge.min(0) - xs.TIE_M - 20.0, bridge.max(0) + xs.TIE_M + 20.0
        lines, halves, classes, seen = [], [], [], set()
        for c in cs:
            for line, half, gc in c["ground"]:
                if gc in UNGRADED or len(line) < 2:
                    continue
                if (line.max(0) < lo_b).any() or (line.min(0) > hi_b).any():
                    continue
                kk = (round(float(line[0, 0]), 1), round(float(line[0, 1]), 1), len(line))
                if kk in seen:
                    continue
                seen.add(kk)
                lines.append(np.asarray(line, f64)[:, :2])
                halves.append(float(half))
                classes.append(gc)
        if not lines:
            return None
        crs = roadsmod._crossings([bridge], lines)
        crs = [c for c in crs if APPROACH_MIN_CROSS_M <= c[1] <= L - APPROACH_MIN_CROSS_M]
        if not crs:
            return None
        # which line each crossing is on (roads._crossings reports the point; the nearest line holds it)
        cr_line = []
        for c in crs:
            pt = np.asarray(c[2], f64)
            cr_line.append(int(np.argmin([_seg_dist(ln, pt) for ln in lines])))

        def z_at(p):
            p = np.asarray(p, f64).reshape(-1, 2)
            la_, lo_, _ = self.frame.enu_to_geodetic(p[:, 0], p[:, 1], 0.0)
            return np.asarray(self.elev(np.asarray(la_, f64), np.asarray(lo_, f64)), f64).ravel()
        # the bridge: its half width (its approaches' carriageway, as wide as the widest road meeting its ends)
        ends = np.array([bridge[0], bridge[-1]], f64)
        half_over = 0.0
        for ln, h in zip(lines, halves):
            if min(np.hypot(*(ln[0] - ends).T).min(), np.hypot(*(ln[-1] - ends).T).min()) <= APPROACH_TOUCH_M:
                half_over = max(half_over, h)
        half_over = half_over or 6.0
        g_end = z_at(ends)
        u0, u1 = xs.end_dirs(bridge)
        t_end = [xs.trusted_over(ends[0], u0, z_at), xs.trusted_over(ends[1], u1, z_at)]
        base = np.array([g if t is None else max(g, min(t, g + xs.FILL_MAX_M)) for g, t in zip(g_end, t_end)], f64)
        clear = roadsmod.deck_road_clear(roadsmod.deck_depth(roadsmod.geom_span([bridge]))) + ROAD_LIFT
        low_ground = bool(np.min(base) < seatiles.SEA_E + xs.LOW_M)
        fixed = str(cls) in APPROACH_SKIP
        # (a motorway or trunk off low ground is not lifted: the roads under it give way, as #126. Lifted, Perth's
        # Mitchell Freeway rose 1-4 m to each of its street bridges, humps along it where a street dipping 1-2 m under
        # it was all it took. Its ends are still brought up to its road's trend past them: left on their embankments'
        # slopes, the freeway dipped 1-2 m into each bridge and humped over the street, the user 2026-11-01)
        roads_ = [ln for ln, gc in zip(lines, classes) if gc not in xs.PATHS]
        deck, low, w_down, drop_max, pts = [], [], [], [], []
        floor = seatiles.SEA_E + UNDER_SEA_M
        for c, li in zip(crs, cr_line):
            pt = np.asarray(c[2], f64)
            arc = float(c[1])
            ln = lines[li]
            u = xs.dir_at(ln, pt)
            t_low = xs.trusted_under(pt, u, half_over, z_at)
            here = float(z_at(pt)[0])
            # (a real cutting the DEM has keeps its depth: the trusted height is never over the road's own there)
            tl = here if t_low is None else min(t_low, here)
            deck.append(base[0] + (base[1] - base[0]) * arc / max(L, 1e-9))
            low.append(tl)
            nt = 0 if classes[li] in xs.PATHS else xs.ties(pt, roads_, ln)
            w_down.append(xs.down_cost(classes[li], nt, low_ground))
            dm = UNDER_MAX_DROP_M
            if UNDER_SEA_ON and tl >= floor:
                dm = min(dm, max(0.0, tl - floor))
            drop_max.append(dm)
            pts.append(pt)
        cap = xs.MAX_LIFT_FIXED_M if str(cls) in APPROACH_SKIP else min(xs.MAX_LIFT_M, APPROACH_MAX_LIFT)
        if fixed and not low_ground:
            cap = 0.0
        lift, drops = xs.solve(deck, low, clear, xs.up_cost(cls), np.array(w_down), cap, np.array(drop_max))
        out = {"ends": ends, "raise": base - g_end + lift, "lift": lift, "top": base + lift,
               "reach": half_over + xs.UNDER_GAP_M,
               "cross": [(p, float(t - d)) for p, t, d in zip(pts, low, drops)]}
        log.debug("grading: crossing solve %s (%.0f m, ends %s, trusted %s, clear %.1f, cost %.1f) over %s: lift %.1f, "
                  "roads to %s", cls, L, np.round(g_end, 1), np.round(base, 1), clear, xs.up_cost(cls),
                  [(classes[li], round(t, 1), round(w, 1)) for li, t, w in zip(cr_line, low, w_down)], lift,
                  [round(t, 1) for _p, t in out["cross"]])
        return out

    def _end_lift_raw(self, xy):
        la, lo, _ = self.frame.enu_to_geodetic(np.array([xy[0]]), np.array([xy[1]]), 0.0)
        home = mapcells.cell_of(float(la[0]), float(lo[0]))
        hc = self._cell_enu(home, ground=False)
        if not len(hc["ends"]):
            return None
        d = np.hypot(hc["ends"][:, 0] - xy[0], hc["ends"][:, 1] - xy[1])
        hit = np.flatnonzero(d <= APPROACH_TOUCH_M)
        if not len(hit):
            return None
        bridge = hc["bridges"][hc["end_of"][int(hit[0])]]
        bridge_cls = hc.get("bridge_cls", [None] * len(hc["bridges"]))[hc["end_of"][int(hit[0])]]
        sol = self._xsolve(bridge) if XS_ON else None
        # the bridge's own cells (a few, it is short): the roads crossing under it and the parts it joins
        lam, lom, _ = self.frame.enu_to_geodetic(bridge[:, 0], bridge[:, 1], 0.0)
        pad = 50.0 / 111320.0
        padlon = pad / max(0.05, math.cos(math.radians(float(np.mean(lam)))))
        keys = sorted(set(mapcells.cells_for_points(lam, lom, pad, padlon)) | {home}, key=str)
        cs = [self._cell_enu(k) for k in keys]
        bridges = [b for c in cs for b in c["bridges"]]
        lo_b, hi_b = bridge.min(0) - APPROACH_MAX_RUN - 50.0, bridge.max(0) + APPROACH_MAX_RUN + 50.0
        ground, others, approaches = [], [], []
        chain = bridge_chain(bridge, bridges)
        spans = [bridge] + chain        # (a road under any part of the chain is spanned, not beside the ramp: #138)
        if sol is not None:
            # (and under the decks beside it: a dual carriageway's other deck over the same road. The street passing
            # under Wezep's other A28 deck, 10 m off, capped this one's ramp at its abutment: the deck's end 1 m under
            # its ramp, a dip where it met the ground, the user 2026-11-01)
            spans = spans + [b for b in bridges if b is not bridge and len(b) >= 2
                             and float(np.min(np.hypot(*(b[:, :2] - xy).T))) <= TWIN_M]
        blo = np.min([b.min(0) for b in spans], axis=0)
        bhi = np.max([b.max(0) for b in spans], axis=0)
        lines, halves, lo_l, hi_l, first, last = self._ground_table(keys, cs)
        near = np.flatnonzero(~((hi_l < lo_b).any(1) | (lo_l > hi_b).any(1)))   # (the rest nowhere near the bridge)
        tch = np.minimum(np.hypot(first[near, 0] - xy[0], first[near, 1] - xy[1]),
                         np.hypot(last[near, 0] - xy[0], last[near, 1] - xy[1])) <= APPROACH_TOUCH_M
        box = ~((hi_l[near] < blo).any(1) | (lo_l[near] > bhi).any(1))
        inbox = np.flatnonzero(box)     # (which of them cross under the chain: one kernel call for all)
        crossing = np.zeros(len(near), bool)
        for n, cr in zip(inbox, roadsmod._crossings_by(spans, [lines[near[n]] for n in inbox])):
            crossing[n] = bool(cr)
        for n, i in enumerate(near):
            line, half = lines[i], halves[i]
            ground.append(line)
            touches = bool(tch[n])
            under = bool(crossing[n])
            if touches:
                approaches.append((line, half))
            # the roads beside the ramp, which its fill slope must not reach (not the ones the bridge spans: its
            # abutment stands between)
            if not touches and not under:
                others.append((line, half))

        # (the tracks near it: crossed under like a road, never an approach or a road beside the ramp)
        rails = [line for c in cs for line in c.get("rail_ground", ())
                 if not ((line.max(0) < lo_b).any() or (line.min(0) > hi_b).any())]

        def z_at(p):
            la_, lo_, _ = self.frame.enu_to_geodetic(p[:, 0], p[:, 1], 0.0)
            return np.asarray(self.elev(la_, lo_), np.float64)
        best = None
        for end, lift in bridge_lifts(bridge, bridges, ground, z_at, chain, rails):
            if np.hypot(*(end - xy)) <= APPROACH_TOUCH_M:
                best = lift
                break
        if best is not None and LINK_BESIDE_ON and _link_of_fixed(bridge_cls) and beside_fixed_span(
                bridge, [b for c in cs for b, bc in zip(c["bridges"], c.get("bridge_cls", ())) if bc in APPROACH_SKIP],
                lines):
            best = None                 # (a link's deck beside its mainline's over the same road: the road dips)
        seen = self._observed_lift(bridge, xy, z_at)
        if seen is not None:            # (a measured deck: its own end, not what the clearance rules would need)
            seen, target = seen
            return None if seen < APPROACH_MIN_LIFT else (min(seen, APPROACH_MAX_LIFT), others, approaches, False,
                                                          target)
        if sol is not None:             # (crossings.ON: the bridge and the roads under it solved together)
            i = int(np.argmin(np.hypot(*(sol["ends"] - xy).T)))
            lift = float(sol["raise"][i])
            # (lift: over the abutment's own ground; the last: the height its roads reach there, the trusted end + the
            # lift, which _lift_approaches raises a plan to from its own height, as for a measured deck)
            return None if lift < APPROACH_MIN_LIFT else (min(lift, APPROACH_MAX_LIFT), others, approaches, False,
                                                          None, True, float(sol["top"][i]))
        wl = self._water_lift(bridge, chain, xy, z_at) if bridge_chain_ends_free(bridge, bridges, xy) else 0.0
        if wl >= APPROACH_MIN_LIFT and (best is None or wl > best):
            return min(wl, WATER_MAX_LIFT), others, approaches, True
        return None if best is None else (best, others, approaches, False)

    def _water_lift(self, bridge, chain, xy, z_at):
        """m the abutment at xy of a bridge over water must be raised for its deck to reach the water's clearance
        (roads.BridgeProfiles._water, the deck's own depth) at DECK_DESIGN_GRADE after WATER_CURVE_M of vertical curve,
        0 when it need not. A causeway's 100 m channel bridge over 3.8 m of navigation clearance, its ends 0.7 m short
        of it 12 m out, humped 1.8 m over a curve squeezed under its grade cap (the Sunshine Skyway, 2026-10-27)."""
        if self.wet is None or not WATER_LIFT_ON:
            return 0.0
        try:
            import bridgetypes
            ch = self._chains_for()
            lines = [roadsmod.densify(np.asarray(c, np.float64)[:, :2], roadsmod.DECK_STEP)       # (as the deck's)
                     for c in [bridge] + list(chain or ())]
            try:
                nav = bridgetypes.nav_clearance(self.frame, lines)
            except Exception:
                nav = 0.0
            req = ch._water(lines, nav, ch._depth(lines))
            if not req:
                return 0.0
            pts, need = [], []
            for k, arc, base, c in req:
                ln = lines[k]
                s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ln, axis=0).T))]
                pts.append([np.interp(arc, s, ln[:, 0]), np.interp(arc, s, ln[:, 1])])
                need.append(base + c)
            d = np.hypot(*(np.asarray(pts) - np.asarray(xy)[:2]).T)
            reach = np.asarray(need) - roadsmod.DECK_DESIGN_GRADE * np.maximum(d - roadsmod.DECK_ROAD_HALF - WATER_CURVE_M, 0.0)
            return max(0.0, float(reach.max() - z_at(np.asarray(xy, np.float64)[None, :2])[0]))
        except Exception as e:      # no fill rather than no grading
            log.debug("grading: water lift: %s: %s", type(e).__name__, e)
            return 0.0

    def _longest(self, fid, j, ring, key):
        """Part j of tunnel fid as the cells at its ends have it, the longest (a part comes clipped differently in
        the cells around: graded as the 30 m piece, 97th Street's 45 m arch got an open road graded in it; the
        same rule as roads.BridgeProfiles._load, 2026-10-17)."""
        best = ring
        for p in (ring[0], ring[-1]):
            ck = mapcells.cell_of(float(p[1]), float(p[0]))
            if key is not None and str(ck) == str(key):
                continue
            try:
                f = self.md.cell(ck, layers=("roads",), kind="demand").layers.get("roads")
            except Exception:
                continue
            if f is None or not len(f):
                continue
            ids = [str(x) for x in f.ids]
            if fid not in ids:
                continue
            i = ids.index(fid)
            parts = roadsmod.tunnel_parts(f, i)
            if j < len(parts) and len(parts[j]) > len(best):
                best = parts[j]
        return best
