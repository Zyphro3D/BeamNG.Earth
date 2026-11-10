"""Road builder: what each map road is (split out of roads.py): its level ranges (ground, bridge, tunnel parts), surface,
width, lanes and class (road_specs and their tapers, shifts and lane flow).
"""
import math
import os
import threading
import time
import numpy as np
import mapcells
import roadmesh
import roadbase
from roadbase import (
    BUILT,
    COBBLE,
    COBBLE_ALIASES,
    DUAL_ANGLE_DEG,
    DUAL_CLASSES,
    DUAL_GAP_M,
    DUAL_MIN_MEDIAN_M,
    DUAL_MIN_W,
    DUAL_SHARE,
    EDGE_STRIP_CLASS_M,
    EDGE_STRIP_M,
    FACTS_LOG,
    FACTS_SHORT_S,
    FACTS_WAIT_S,
    GAP_DECK_M,
    GAP_JOIN_M,
    GAP_JUNCTION_M,
    GOOD_SMOOTHNESS,
    GRAVEL,
    HAIRPIN_SPAN_M,
    HAIRPIN_WIDEN_M,
    HOME_RANGES_ON,
    LANE_FLOW_CLASSES,
    LANE_FLOW_MAX,
    LANE_M,
    LANE_W,
    LANE_W_DEFAULT,
    LIDAR_GAP_REACH_M,
    LIDAR_GAP_TOUCH_M,
    LINK_LANE_M,
    LINK_M,
    LINK_SHOULDER_M,
    MEASURE_MAX,
    MEASURE_SKIP,
    MIN_PART,
    MOTORWAY_RURAL_M,
    MOUNTAIN_CLASSES,
    MOUNTAIN_MAX_LANES,
    MOUNTAIN_ON,
    MOUNTAIN_ROAD_M,
    MOUNTAIN_SPAN_M,
    MOUNTAIN_SWITCHBACKS,
    MOUNTAIN_TURN_DEG,
    ONEWAY_STYLE,
    OSM_LANES,
    OSM_WIDTH_M,
    PARKING_CLASSES,
    PARKING_M,
    PASSAGE_M,
    PAVED,
    PORTAL_CLASSES,
    PORTAL_CLEAR_M,
    PORTAL_EXTEND_MAX_M,
    PORTAL_JOIN_M,
    RAMP_JOIN_DEG,
    REGIONAL_CLASSES,
    REGIONAL_MIN_M,
    REGIONAL_POOL_MIN_M,
    REGIONAL_SHARE,
    RING_EXTRA_LANE_M,
    RING_JUNCTIONS,
    RING_LANE_M,
    SE_GRAVEL_CLASSES,
    SE_NO_STRIPS,
    SE_TOWN_MAX_KMH,
    SHIFT_M,
    SHIFT_MIN_M,
    SHIFT_M_PER_M,
    SHIFT_SHARE,
    SHOULDER_M,
    SHOULDER_WIDE_LANES,
    SHOULDER_WIDE_M,
    SIDEWALK_CLASSES,
    SIDEWALK_TAG,
    SKEW_MARGIN_M,
    SKEW_MAX_M,
    SKEW_MIN_SIN,
    SKEW_ON,
    SKEW_SLOPE_M,
    STYLE,
    Spec,
    TAPER_M,
    TAPER_MAX_TURN,
    TAPER_MIN_DW,
    TAPER_M_PER_M,
    TAPER_SHARE,
    TUNNEL_GAP_M,
    TUNNEL_PORTAL_CLEAR_M,
    TUNNEL_PORTAL_END_M,
    TUNNEL_PORTAL_GROW_M,
    TUNNEL_PORTAL_MIN_M,
    TUNNEL_PORTAL_REACH_M,
    TUNNEL_PORTAL_SHRINK_M,
    TUNNEL_ROAD_MIN_SIN,
    UNBUILT_SHARE,
    UNDERPASS_MAX_M,
    UNPAVED,
    URBAN_ARTERIAL,
    URBAN_BESIDE_M,
    URBAN_BOTH_SHARE,
    URBAN_DUAL_MAX_W,
    URBAN_MEDIAN_GAP_M,
    URBAN_MEDIAN_M,
    URBAN_SAMPLE_M,
    URBAN_SHARE,
    _FACTS_LOCK,
    _FACTS_MEM,
    _beside,
    _facts_load,
    _facts_store,
    _home_cells,
    _osm,
    _ranges_mem_key,
    _rule_value,
    _run_facts_load,
    _run_facts_path,
    _run_facts_store,
    home_copy,
    log,
)


URBAN_BESIDE_SHARE = float(os.environ.get("BNE_URBAN_BESIDE_SHARE", "0.35"))
                                # sustained development beside at least 35% of length samples, rather than a few
                                # school/pavement endpoints; the land cover counts wide pavement as built-up, and the Eyre Highway
                                # through Eucla (centre 94 %, beside it 3 %) got a town's footpaths, bins and hydrants
                                # across the Nullarbor (GitHub #267). Baltimore and Chicago keep every urban road.


def level_ranges(feats, i, _home=True):
    """The off-ground ranges of feature i as sorted (start, end, "bridge" | "tunnel") fractions of its
    length, from its flags (is_bridge / is_tunnel) and level (> 0 bridge, < 0 tunnel) rules. Overture
    puts most bridges on a range of a longer segment (2026-10-11: a street with a few metres of bridge
    was taken whole as a bridge and dropped). Pure. A cell's are made all at once and shared (cell_facts).
    Its home cell's copy's (home_copy): lengthened over the roads crossing under it and joined over the lidar's
    elevated stretches, the copy in each cell saw other roads (a ramp by I-395 in Baltimore: one deck from 9.5 % in
    one cell, two with a ground gap at 14-16 % in the next, both drawn, one 8 m under the other)."""
    if _home and HOME_RANGES_ON:
        h = home_copy(feats, i)
        if h is not None:
            return level_ranges(h[0], h[1], _home=False)
    memo = feats.__dict__.setdefault("_level_ranges", {})
    got = memo.get(i)
    if got is None:
        _cell_ranges(feats)
        got = memo.get(i)
        if got is None:
            # (the lidar's elevated stretches gap-filled again: one ending 12 m short of the map's deck left a ground
            # piece between them mid-ramp, by I-395 in Baltimore)
            got = memo[i] = _deck_gaps(feats, i, _lidar_gaps(feats, i, rule_level_ranges(feats, i)))
    if feats.__dict__.get("_lidar_rule"):
        import revisit                      # (the rule for want of the lidar: nothing built on it is kept)
        revisit.fault(lidar=True)
    return list(got)


def rule_level_ranges(feats, i):
    """level_ranges before the lidar's elevated stretches (_lidar_gaps): the map's ranges lengthened (_skew_decks) and
    their short ground gaps between decks filled (_deck_gaps). roadz.py measures the lidar along these (its answer is
    what level_ranges adds). Remembered on feats."""
    memo = feats.__dict__.setdefault("_rule_level_ranges", {})
    got = memo.get(i)
    if got is None:
        got = memo[i] = _deck_gaps(feats, i, _skew_decks(feats, i, raw_level_ranges(feats, i)))
    return list(got)


def raw_level_ranges(feats, i):
    """level_ranges as the map has them, before _skew_decks (road_specs reads these: _skew_decks needs the specs)."""
    memo = feats.__dict__.setdefault("_raw_level_ranges", {})
    got = memo.get(i)
    if got is None:
        _cell_ranges(feats, load_only=True)
        got = memo.get(i)
        if got is None:
            got = memo[i] = _level_ranges(feats, i)
    return list(got)




def _cell_ranges(feats, load_only=False):
    """Every feature's raw and lengthened level ranges into feats' memos: from the shared facts, or (not load_only)
    made here and kept there."""
    st = feats.__dict__
    with _FACTS_LOCK:
        if st.get("_ranges_state") is not None:
            if load_only or st["_ranges_state"] != "loadable":
                return
        tried = st.get("_ranges_state")
        st["_ranges_state"] = "busy"
    if tried is None:
        mk = _ranges_mem_key(st)
        with _FACTS_LOCK:
            mem = _FACTS_MEM.get(mk) if mk is not None else None
            if mem is not None:
                _FACTS_MEM.move_to_end(mk)
        if mem is not None and mem.get("until") is not None and time.monotonic() > mem["until"]:
            mem = None                  # (made short of something: made again once FACTS_SHORT_S is past)
        if mem is not None and len(mem["raw"]) == len(feats):
            if mem.get("until") is not None:
                import revisit
                revisit.fault()         # (made short of something: what is built on it now is not kept either)
            raw, lvl = st.setdefault("_raw_level_ranges", {}), st.setdefault("_level_ranges", {})
            mraw, mlvl = mem["raw"], mem["lvl"]     # (shared: level_ranges and the rest hand out copies)
            for i in range(len(feats)):
                raw.setdefault(i, mraw[i])
                lvl.setdefault(i, mlvl[i])
            if mem["rule"]:
                st["_lidar_rule"] = True    # (built on the rule: level_ranges says so, nothing on it kept)
            st["_ranges_state"] = "done"
            return
        got = _facts_load(feats, "ranges", None)
        if got is not None and len(got.get("raw", ())) == len(feats) and len(got.get("lvl", ())) == len(feats):
            raw, lvl = st.setdefault("_raw_level_ranges", {}), st.setdefault("_level_ranges", {})
            for i in range(len(feats)):
                raw.setdefault(i, [tuple(r) for r in got["raw"][i]])
                lvl.setdefault(i, [tuple(r) for r in got["lvl"][i]])
            st["_ranges_state"] = "done"
            return
    if load_only:
        st["_ranges_state"] = "loadable"        # (the eager build waits for level_ranges: road_specs reads the raw)
        return
    import revisit
    run_path = _run_facts_path(st)
    mine = False
    if run_path is not None:            # (this run's: another process's, or theirs being made, waited for briefly)
        import xproc
        got, mine = xproc.once(run_path + ".lock", lambda: _run_facts_load(run_path, len(feats)), FACTS_WAIT_S)
        if got is not None:
            raw_m, lvl_m = st.setdefault("_raw_level_ranges", {}), st.setdefault("_level_ranges", {})
            for i in range(len(feats)):
                raw_m.setdefault(i, [tuple(r) for r in got["raw"][i]])
                lvl_m.setdefault(i, [tuple(r) for r in got["lvl"][i]])
            if got.get("rule"):
                st["_lidar_rule"] = True
            st["_ranges_state"] = "done"
            mk = _ranges_mem_key(st)
            if mk is not None:
                with _FACTS_LOCK:
                    _FACTS_MEM[mk] = {"raw": [raw_m[i] for i in range(len(feats))],
                                      "lvl": [lvl_m[i] for i in range(len(feats))], "rule": bool(got.get("rule")),
                                      "until": None}
            return
    f0 = revisit.thread_faults()
    t0 = time.perf_counter()
    try:
        raw = [raw_level_ranges(feats, i) for i in range(len(feats))]
        # (most roads have no range, no end near a bridge's and no lidar answer: their level ranges are [] whatever the
        # chain below does, and its per-feature calls were ~1 s of CPU a cell on first visits, Fells Point 2026-11-06)
        near = _near_bridge_ends(feats)
        answers = _lidar_answers(feats) if roadbase.LIDAR_GAPS_ON else {}
        lvl_memo = st.setdefault("_level_ranges", {})
        lvl = []
        for i in range(len(feats)):
            if (near is not None and not raw[i] and not near[i] and not answers.get(str(feats.ids[i]))
                    and i not in lvl_memo):
                lvl_memo[i] = []
                lvl.append([])
            else:
                lvl.append(level_ranges(feats, i, _home=False))
    except BaseException:
        if mine:
            import xproc
            xproc.release(run_path + ".lock")
        raise
    st["_ranges_state"] = "done"
    if FACTS_LOG:                       # (BNGE_FACTS_LOG: which process made which cell's facts, how long)
        log.info("cell facts %s made in %.2f s (pid %d)%s", st.get("_bnge_cell"), time.perf_counter() - t0, os.getpid(),
                 "" if revisit.thread_faults() == f0 else " short")
    clean = revisit.thread_faults() == f0
    mk = _ranges_mem_key(st)
    if mk is not None:                  # (the lidar rule's too: a cell's answer stays the same for the session)
        with _FACTS_LOCK:
            # (made short of something (a neighbour not downloaded yet: a first visit): kept FACTS_SHORT_S, not made
            # again for every reload meanwhile, one cell 14 times in a process down Manhattan, 2026-11-02)
            _FACTS_MEM[mk] = {"raw": raw, "lvl": lvl, "rule": bool(st.get("_lidar_rule")),
                              "until": None if clean else time.monotonic() + FACTS_SHORT_S}
            while len(_FACTS_MEM) > roadbase.FACTS_MEM_KEPT:
                _FACTS_MEM.popitem(last=False)
    data = {"raw": [[[float(a), float(b), k] for a, b, k in r] for r in raw],
            "lvl": [[[float(a), float(b), k] for a, b, k in r] for r in lvl]}
    if clean and run_path is not None:
        _run_facts_store(run_path, dict(data, rule=bool(st.get("_lidar_rule"))))
    if mine:
        import xproc
        xproc.release(run_path + ".lock")
    if clean and not st.get("_lidar_rule"):
        _facts_store(feats, "ranges", None, data)


def _feature_boxes(feats):
    """(n, 4) lon/lat boxes (lon0, lat0, lon1, lat1) of the features, remembered on feats."""
    got = feats.__dict__.get("_lonlat_boxes")
    if got is None:
        n = len(feats)
        got = np.full((n, 4), np.nan)
        po = np.asarray(feats.part_off, np.int64)
        co, ro = np.asarray(feats.coord_off, np.int64), np.asarray(feats.ring_off, np.int64)
        a, b = co[ro[po[:-1]]], co[ro[po[1:]]]
        ok = b > a
        ll = np.asarray(feats.lonlat, np.float64)
        if ok.any() and np.all(a[ok][1:] == b[ok][:-1]) and b[ok][-1] == len(ll):   # (contiguous: one reduce each)
            st = a[ok]
            got[ok, 0], got[ok, 1] = np.minimum.reduceat(ll[:, 0], st), np.minimum.reduceat(ll[:, 1], st)
            got[ok, 2], got[ok, 3] = np.maximum.reduceat(ll[:, 0], st), np.maximum.reduceat(ll[:, 1], st)
        else:
            for i in np.flatnonzero(ok):
                seg = ll[a[i]:b[i]]
                got[i] = (seg[:, 0].min(), seg[:, 1].min(), seg[:, 0].max(), seg[:, 1].max())
        feats.__dict__["_lonlat_boxes"] = got
    return got


def _raw_ground_parts(feats, j):
    """Feature j's on-ground pieces from its raw_level_ranges (_skew_decks: level_ranges of the others would recurse)."""
    pieces, at = [], 0.0
    for a, b, _ in raw_level_ranges(feats, j):
        if a - at > MIN_PART:
            pieces.append((at, a))
        at = max(at, b)
    if 1.0 - at > MIN_PART:
        pieces.append((at, 1.0))
    out = []
    for part in feats.parts(j):
        for ring in part:
            if len(ring) >= 2:
                for a, b in pieces:
                    out.append(ring if (a, b) == (0.0, 1.0) else sub_polyline(ring, a, b))
    return out


def _seg_hits(m, lines):
    """(line index, arc m along polyline m, sin of the angle) where each of the polylines lines (all (k, 2) m) crosses
    m (earthfast.seg_hits). Pure."""
    import fast
    fast.require(73, "seg_hits")
    starts = np.r_[0, np.cumsum([len(g) for g in lines])].astype(np.int64)
    return fast._ef.seg_hits(np.ascontiguousarray(m, np.float64),
                             np.ascontiguousarray(np.vstack(lines), np.float64), starts)


def _structure_width(feats, i):
    """m: the carriageway of road feature i as _bridge_specs makes it, without its level ranges (a road bridged along
    its whole length has no road_specs entry: its ranges were never lengthened over the roads under it; from cody196's
    PR #368), or None (not a drawn road)."""
    cls = str(feats.attrs["class"][i]) if "class" in feats.attrs else "unknown"
    if str(feats.attrs.get("subtype", ["road"] * len(feats))[i]) != "road" or cls not in STYLE or unbuilt(feats, i):
        return None
    rules = feats.rules
    width = _rule_value(rules.get("width"), i)
    rule_w = float(width) if width is not None and 1.0 <= float(width) <= 40.0 else None
    if rule_w is None:
        rule_w = _osm(feats, "width", i, *OSM_WIDTH_M)
    lanes = _lane_count(feats, i, cls)
    kind = surface_kind(cls, _surface_value(feats, i))
    ow = _rule_value(rules.get("oneway"), i) is not None
    import roadlanes
    if rule_w is None and roadlanes.is_link(feats, i):
        rule_w = _link_width(lanes or (1 if ow else 2))
    return carriageway(cls, kind, False, rule_w, ow, lanes, link=roadlanes.is_link(feats, i))


def _bridge_reach(feats, i):
    """Bridge feature i's ranges lengthened over the ground roads crossing under them (SKEW_*), and how far past each
    end of the feature they would reach: {"ranges": [...], "over": [m past its start, m past its end]} (_skew_decks
    puts that on the road joined there). Remembered on feats. Pure given feats."""
    memo = feats.__dict__.setdefault("_bridge_reach", {})
    got = memo.get(i)
    if got is not None:
        return got
    rngs = raw_level_ranges(feats, i)
    got = {"ranges": rngs, "over": [0.0, 0.0]}
    memo[i] = got
    specs = road_specs(feats)
    up_w = specs[i].width if specs[i] is not None else _structure_width(feats, i)
    rings = [np.asarray(r, np.float64) for part in feats.parts(i) for r in part if len(r) >= 2]
    if up_w is None or not rings or not any(k == "bridge" for _, _, k in rngs):
        return got
    ring = rings[0]
    kx, ky = 111320.0 * math.cos(math.radians(float(ring[0, 1]))), 111320.0
    m = (ring - ring[0]) * [kx, ky]
    sm = np.r_[0.0, np.cumsum(np.hypot(*np.diff(m, axis=0).T))]
    L = float(sm[-1])
    if L <= 0:
        return got
    boxes = _feature_boxes(feats)
    out, over = [], [0.0, 0.0]
    for n, (a, b, k) in enumerate(rngs):
        if k == "bridge":
            a0, b0 = a, b
            lo_m, hi_m = a0 * L, b0 * L
            # (only the bridge's own stretch: a crossing counts under it; a 3 km segment's whole line and every road
            # near it cost a rail-yard tile seconds, 2026-10-25)
            inner = (sm > lo_m) & (sm < hi_m)      # (metres: sub_polyline scales lon/lat)
            sub = np.vstack([[np.interp(lo_m, sm, m[:, 0]), np.interp(lo_m, sm, m[:, 1])], m[inner],
                             [np.interp(hi_m, sm, m[:, 0]), np.interp(hi_m, sm, m[:, 1])]])
            pad = 1.0
            blo = (sub.min(0) - pad) / [kx, ky] + ring[0]
            bhi = (sub.max(0) + pad) / [kx, ky] + ring[0]
            near = np.flatnonzero((boxes[:, 0] <= bhi[0]) & (boxes[:, 2] >= blo[0]) & (boxes[:, 1] <= bhi[1]) &
                                  (boxes[:, 3] >= blo[1]))
            gl, gj = [], []
            for j in near:
                if j == i or specs[j] is None:
                    continue
                for g in _raw_ground_parts(feats, j):
                    gl.append((np.asarray(g, np.float64) - ring[0]) * [kx, ky])
                    gj.append(j)
            lo0, hi0 = lo_m, hi_m           # (arcs along sub: from lo0; a crossing at either end of it does not count)
            for g, sc, sin in (_seg_hits(sub, gl) if gl else ()):
                sc += lo0
                if not lo0 < sc < hi0:
                    continue
                j = gj[g]
                sin = max(sin, SKEW_MIN_SIN)
                cos = math.sqrt(max(0.0, 1.0 - sin * sin))
                h = (specs[j].width / 2 + SKEW_MARGIN_M + SKEW_SLOPE_M) / sin + (up_w / 2) * cos / sin
                lo_m = min(lo_m, max(a0 * L - SKEW_MAX_M, sc - h))
                hi_m = max(hi_m, min(b0 * L + SKEW_MAX_M, sc + h))
            prev = out[-1][1] if out else 0.0   # (never into a neighbouring range; past the feature's ends: "over")
            nxt = rngs[n + 1][0] if n + 1 < len(rngs) else 1.0
            if lo_m < 0 and a0 <= 1e-9:
                over[0] = max(over[0], -lo_m)
            if hi_m > L and b0 >= 1.0 - 1e-9:
                over[1] = max(over[1], hi_m - L)
            a, b = max(lo_m / L, prev, 0.0), min(hi_m / L, nxt, 1.0)
        out.append((a, b, k))
    got = memo[i] = {"ranges": out, "over": over}
    return got


def _tunnel_reach(feats, i, rngs):
    """rngs with feature i's tunnel ranges grown over the ground roads crossing it at their portals
    (TUNNEL_PORTAL_*), never into a neighbouring range. Pure given feats."""
    specs = road_specs(feats)
    rings = [np.asarray(r, np.float64) for part in feats.parts(i) for r in part if len(r) >= 2]
    if specs[i] is None or not rings:
        return rngs
    ring = rings[0]
    kx, ky = 111320.0 * math.cos(math.radians(float(ring[0, 1]))), 111320.0
    m = (ring - ring[0]) * [kx, ky]
    sm = np.r_[0.0, np.cumsum(np.hypot(*np.diff(m, axis=0).T))]
    L = float(sm[-1])
    if L <= 0:
        return rngs
    tun = [n for n, (_, _, k) in enumerate(rngs) if k == "tunnel"]
    pad = (TUNNEL_PORTAL_REACH_M + TUNNEL_PORTAL_GROW_M + 1.0) / np.array([kx, ky])
    boxes = _feature_boxes(feats)
    near = np.zeros(len(boxes), bool)
    for n in tun:                   # (the roads near the portals only)
        for f in rngs[n][:2]:
            p = (np.array([np.interp(f * L, sm, m[:, 0]), np.interp(f * L, sm, m[:, 1])]) / [kx, ky]) + ring[0]
            near |= ((boxes[:, 0] <= p[0] + pad[0]) & (boxes[:, 2] >= p[0] - pad[0]) & (boxes[:, 1] <= p[1] + pad[1]) &
                     (boxes[:, 3] >= p[1] - pad[1]))
    gl, hw = [], []
    for j in np.flatnonzero(near):
        if j == i or specs[j] is None:
            continue
        for g in _raw_ground_parts(feats, j):
            gl.append((np.asarray(g, np.float64)[:, :2] - ring[0]) * [kx, ky])
            hw.append(specs[j].width / 2)
    if not gl:
        return rngs
    import fast
    fast.require(97, "tunnel_reach")
    got = fast._ef.tunnel_reach(
        np.ascontiguousarray(m), np.ascontiguousarray(sm),
        np.ascontiguousarray([[rngs[n][0] * L, rngs[n][1] * L] for n in tun], np.float64),
        np.ascontiguousarray(np.concatenate(gl)), np.r_[0, np.cumsum([len(g) for g in gl])].astype(np.int64),
        np.asarray(hw, np.float64), np.array([TUNNEL_PORTAL_CLEAR_M, TUNNEL_ROAD_MIN_SIN, TUNNEL_PORTAL_REACH_M,
                                              TUNNEL_PORTAL_GROW_M, TUNNEL_PORTAL_END_M, TUNNEL_PORTAL_SHRINK_M,
                                              TUNNEL_PORTAL_MIN_M], np.float64))
    out = list(rngs)
    for q, n in enumerate(tun):
        prev = out[n - 1][1] if n > 0 else 0.0
        nxt = out[n + 1][0] if n + 1 < len(out) else 1.0
        out[n] = (max(float(got[q, 0]) / L, prev), min(float(got[q, 1]) / L, nxt), "tunnel")
    return [r for r in out if r[1] > r[0]]      # (a tunnel left shorter than nothing: ground)


def _bridge_ends(feats):
    """(indices (k,), end points lon/lat (k, 2, 2)) of the features with a bridge range, once per cell."""
    got = feats.__dict__.get("_bridge_ends")
    if got is None:
        idx, ends = [], []
        for b in range(len(feats)):
            if not any(k == "bridge" for _, _, k in raw_level_ranges(feats, b)):
                continue
            rings = [r for part in feats.parts(b) for r in part if len(r) >= 2]
            if rings:
                idx.append(b)
                ends.append((rings[0][0], rings[0][-1]))
        got = (np.asarray(idx, np.int64), np.asarray(ends, np.float64).reshape(-1, 2, 2))
        feats.__dict__["_bridge_ends"] = got
    return got


def _near_bridge_ends(feats):
    """Per feature: may an end of its first ring lie within _skew_decks' 0.75 m of a bridge feature's end? A box test
    in lon / lat, wider than that everywhere (earthfast.ends_near; most features are nowhere near a bridge, and their
    exact test was most of _skew_decks' time, 2026-10-25). None: no pre-check. Kept on feats."""
    got = feats.__dict__.get("_near_bridge_ends")
    if got is not None:
        return got
    import fast
    bidx, bends = _bridge_ends(feats)
    n = len(feats)
    ends = np.full((n, 2, 2), np.nan)
    exact = np.zeros(n, bool)       # (no first ring of 2 points or more: _skew_decks' own test, which finds its ring)
    for i in range(n):
        if feats.part_off[i + 1] <= feats.part_off[i]:
            continue                # (no parts: nothing to lengthen)
        r = feats.ring_off[feats.part_off[i]]
        a, b = feats.coord_off[r], feats.coord_off[r + 1]
        if b - a >= 2:
            ends[i, 0], ends[i, 1] = feats.lonlat[a], feats.lonlat[b - 1]
        else:
            exact[i] = True
    lat = float(np.nanmax(np.abs(ends[:, :, 1]))) if np.isfinite(ends[:, :, 1]).any() else 0.0
    if not np.isfinite(lat) or lat > 89.0:
        feats.__dict__["_near_bridge_ends"] = got = None
        return got
    ty = 2.0 / 111320.0                                         # (2 m: 0.75 m and room)
    tx = ty / max(math.cos(math.radians(min(lat + 0.1, 89.0))), 1e-3)
    fast.require(75, "ends_near")
    m = np.asarray(fast._ef.ends_near(np.ascontiguousarray(ends.reshape(-1, 2)),
                                     np.ascontiguousarray(bends.reshape(-1, 2)), float(tx), float(ty)), bool)
    got = m.reshape(n, 2).any(axis=1) | exact
    feats.__dict__["_near_bridge_ends"] = got
    return got


def _skew_decks(feats, i, rngs):
    """rngs (feature i's raw_level_ranges) with its bridge ranges lengthened over the roads crossing under them
    (_bridge_reach), and a bridge range at an end of i where a bridge feature joined there reaches past its own end
    onto i (Overture makes most short bridges a segment of their own). Pure given feats."""
    if any(k == "tunnel" for _, _, k in rngs):
        rngs = _tunnel_reach(feats, i, rngs)
    if not SKEW_ON:
        return rngs
    if any(k == "bridge" for _, _, k in rngs):
        rngs = _bridge_reach(feats, i)["ranges"]
    bidx, bends = _bridge_ends(feats)
    if not len(bidx):
        return rngs
    near = _near_bridge_ends(feats)
    if near is not None and not near[i]:
        return rngs                             # (no end of it near a bridge's end: nothing joins it)
    rings = [np.asarray(r, np.float64) for part in feats.parts(i) for r in part if len(r) >= 2]
    if not rings:
        return rngs
    ring = rings[0]
    kx, ky = 111320.0 * math.cos(math.radians(float(ring[0, 1]))), 111320.0
    tol = 0.75
    add = []
    L = None
    for end, p in ((0, ring[0]), (1, ring[-1])):
        if any((a <= 1e-9 if end == 0 else b >= 1.0 - 1e-9) for a, b, _ in rngs):
            continue                            # off the ground at that end already
        d = np.hypot((bends[:, :, 0] - p[0]) * kx, (bends[:, :, 1] - p[1]) * ky)     # (k, 2) to each bridge's ends
        best = 0.0
        for r, e2 in zip(*np.nonzero(d < tol)):
            b = int(bidx[r])
            if b != i:
                best = max(best, _bridge_reach(feats, b)["over"][int(e2)])
        if best > 0:
            if L is None:
                L = float(np.hypot(*(np.diff(ring, axis=0) * [kx, ky]).T).sum())
            if L <= 0:
                break
            f = min(best, SKEW_MAX_M) / L
            add.append((0.0, min(f, 1.0), "bridge") if end == 0 else (max(1.0 - f, 0.0), 1.0, "bridge"))
    if not add:
        return rngs
    out = []
    for a, b, k in sorted(list(rngs) + add):
        if out and a <= out[-1][1] + 1e-9:     # (overlapping a range already there: joined, as level_ranges does)
            pa, pb, pk = out[-1]
            out[-1] = (pa, max(pb, b), "tunnel" if "tunnel" in (pk, k) else "bridge")
        else:
            out.append((a, b, k))
    return out


# A ground piece of a road shorter than GAP_DECK_M with a deck on both sides (a bridge range of its own, or another
# road's bridge going on from its end with nothing else meeting there) and no road joining it is deck too: a viaduct's
# stretch the map leaves untagged between two bridges was drawn on the ground, each deck eased down to it and up again
# (with the measured decks, 10-15 m: the Springfield Interchange, 2026-10-28). A junction on it keeps it on the ground.
GAP_DECK_ON = os.environ.get("BNE_DECK_GAPS", "on") != "off"


def _first_ring_ends(feats):
    """(n, 4) lon / lat of each feature's first ring's two ends (lon0, lat0, lon1, lat1; NaN: none), kept on feats
    (earthfast.first_ring_ends)."""
    got = feats.__dict__.get("_first_ring_ends")
    if got is None:
        import fast
        fast.require(169, "first_ring_ends")
        ll = np.asarray(feats.lonlat, np.float64)
        got = feats.__dict__["_first_ring_ends"] = fast._ef.first_ring_ends(
            ll if ll.ndim == 2 else ll.reshape(-1, 2), np.ascontiguousarray(feats.coord_off, np.int64),
            np.ascontiguousarray(feats.ring_off, np.int64), np.ascontiguousarray(feats.part_off, np.int64))
    return got


def _spec_mask(feats, specs):
    """(n,) uint8: 1 where specs (_road_specs_kept(feats)) has a spec, kept on feats for that list."""
    got = feats.__dict__.get("_deck_gap_specs")
    if got is None or got[0] is not specs:
        got = feats.__dict__["_deck_gap_specs"] = (specs, np.fromiter((s is not None for s in specs), np.uint8,
                                                                       len(specs)))
    return got[1]


def _deck_gaps(feats, i, rngs):
    """rngs (feature i's level ranges) with its short ground gaps between decks made bridge (GAP_DECK_*;
    earthfast.deck_gaps). A road with a tunnel range keeps its own. Pure given feats."""
    if not GAP_DECK_ON:
        return rngs
    kinds = [r[2] for r in rngs] if rngs else ()   # (most roads have no range: 127k calls in two cells' planning)
    if "tunnel" in kinds:
        return rngs
    if "bridge" not in kinds:
        near = _near_bridge_ends(feats)
        if near is not None and not near[i]:
            return rngs                         # (no bridge of its own, none ending at it)
    if _structure_width(feats, i) is None:
        return rngs                             # (railways: their own pieces)
    rings = [np.asarray(r, np.float64) for part in feats.parts(i) for r in part if len(r) >= 2]
    if not rings:
        return rngs
    ring = rings[0][:, :2]
    kx, ky = 111320.0 * math.cos(math.radians(float(ring[0, 1]))), 111320.0
    m = (ring - ring[0]) * [kx, ky]
    L = float(np.hypot(*np.diff(m, axis=0).T).sum())
    if L <= 0:
        return rngs
    specs = _road_specs_kept(feats)
    import fast
    fast.require(169, "deck_gaps_round")
    # (the other roads' ends at its two ends and near it, earthfast.deck_gaps_round: numpy over the whole layer for
    # each road near a bridge was ~3/4 of this, 2026-11-08)
    hits, others = fast._ef.deck_gaps_round(ring, _first_ring_ends(feats), _feature_boxes(feats),
                                            _spec_mask(feats, specs), i, kx, ky, GAP_JOIN_M, GAP_JUNCTION_M + 1.0)
    end_deck = []
    for e in (0, 1):
        j, k = hits[2 * e], hits[2 * e + 1]
        ok = j >= 0
        if ok:
            rj = raw_level_ranges(feats, j)
            ok = any(kk == "bridge" and (a <= 1e-6 if k == 0 else b >= 1.0 - 1e-6) for a, b, kk in rj)
        end_deck.append(ok)
    bridges = [(a * L, b * L) for a, b, k in rngs if k == "bridge"]
    got = fast._ef.deck_gaps(np.ascontiguousarray(m), np.asarray(bridges, np.float64).reshape(-1, 2),
                             np.asarray(end_deck, np.int64), others, GAP_DECK_M, GAP_JUNCTION_M)
    # (np.allclose's test, rtol 1e-5 / atol 1e-8, on these few numbers)
    if len(got) == len(bridges) and all(x == y or (math.isfinite(y) and abs(x - y) <= 1e-8 + 1e-5 * abs(y))
                                        for g, r in zip(got.tolist(), bridges) for x, y in zip(g, r)):
        return rngs
    out = [(float(min(max(a / L, 0.0), 1.0)), float(min(max(b / L, 0.0), 1.0)), "bridge") for a, b in got]
    out[0] = (0.0, out[0][1], "bridge") if got[0, 0] <= 1e-6 else out[0]
    out[-1] = (out[-1][0], 1.0, "bridge") if got[-1, 1] >= L - 1e-6 else out[-1]
    return out


def _ll_metres(lat):
    return 111320.0 * math.cos(math.radians(lat)), 111320.0


def rule_level_ranges_near(feats):
    """{i: rule_level_ranges} of the features that can have a bridge range by the rule: one of their own (raw) or a
    bridge ending near them (_skew_decks / _deck_gaps add none elsewhere). Kept on feats."""
    got = feats.__dict__.get("_rule_near")
    if got is not None:
        return got
    near = _near_bridge_ends(feats)
    memo = feats.__dict__.get("_raw_level_ranges", {})
    got = {}
    for i in range(len(feats)):
        raw = memo.get(i)
        raw = raw if raw is not None else raw_level_ranges(feats, i)
        if any(k == "bridge" for _, _, k in raw) or near is None or near[i]:
            got[i] = rule_level_ranges(feats, i)
    feats.__dict__["_rule_near"] = got
    return got


def lidar_gap_candidates(feats):
    """[(fid, line lon/lat (k, 2) from the deck's end outward, LIDAR_GAP_REACH_M at most, half width m)] of the drawn
    roads' ground pieces (rule_level_ranges) with an end within LIDAR_GAP_TOUCH_M of a bridge piece's end (any road's).
    Pure given feats; kept on it."""
    got = feats.__dict__.get("_lidar_gap_candidates")
    if got is not None:
        return got
    n = len(feats)
    rule = [None] * n
    ends = []
    for i, rr in rule_level_ranges_near(feats).items():
        rule[i] = rr                            # (the others: no bridge of their own, none ending at them)
        for r in _parts_by_level_new(feats, i, "bridge", rule[i]):
            r = np.asarray(r, np.float64)
            ends += [r[0, :2], r[-1, :2]]
    out = []
    if ends:
        E = np.asarray(ends, np.float64)
        for i in range(n):
            if rule[i] is None or any(k == "tunnel" for _, _, k in rule[i]):
                continue
            w = _structure_width(feats, i)
            if w is None:
                continue
            fid = str(feats.ids[i])
            for r in _parts_by_level_new(feats, i, "ground", rule[i]):
                r = np.asarray(r, np.float64)[:, :2]
                if len(r) < 2:
                    continue
                kx, ky = _ll_metres(float(r[0, 1]))
                for at in (0, -1):
                    d = np.hypot((E[:, 0] - r[at, 0]) * kx, (E[:, 1] - r[at, 1]) * ky)
                    if not (d < LIDAR_GAP_TOUCH_M).any():
                        continue
                    line = r if at == 0 else r[::-1]
                    m = (line - line[0]) * [kx, ky]
                    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(m, axis=0).T))]
                    if s[-1] > LIDAR_GAP_REACH_M:
                        keep = s < LIDAR_GAP_REACH_M
                        cut = [np.interp(LIDAR_GAP_REACH_M, s, line[:, k]) for k in (0, 1)]
                        line = np.vstack([line[keep], cut])
                    out.append((fid, np.ascontiguousarray(line), max(2.5, w / 2)))
    feats.__dict__["_lidar_gap_candidates"] = out
    return out


def _lidar_cell(cid):
    """May the lidar know cell cid ("z/x/y", or a test's name: yes)? (No candidates looked for elsewhere.)"""
    try:
        z, x, y = (int(v) for v in str(cid).split("/"))
    except ValueError:
        return True
    import roadz
    la, lo = mapcells.CellKey(z, x, y).center()
    return "lidar" in roadz.sources_at(la, lo, True)


def _lidar_answers(feats):
    """{fid: [(lon/lat from, lon/lat to)]} of the elevated stretches roadz found on feats' candidates (lidar_gap_
    candidates), from the answers of the cells holding their deck ends; {} without lidar. Kept on feats."""
    st = feats.__dict__
    got = st.get("_lidar_answers")
    if got is not None:
        return got
    lock = st.setdefault("_lidar_answers_lock", threading.Lock())
    with lock:
        got = st.get("_lidar_answers")
        if got is not None:
            return got
        got = {}
        try:
            import roadz
            prov = roadz.PROVIDER
            cid = st.get("_bnge_cell")
            if (roadbase.LIDAR_GAPS_ON and prov is not None and prov.enabled and prov.lidar and cid is not None
                    and hasattr(prov, "gap_answers") and _lidar_cell(cid[0])):
                cands = lidar_gap_candidates(feats)
                if cands:
                    import mapcells
                    owners = sorted({mapcells.cell_of(float(c[1][0, 1]), float(c[1][0, 0])) for c in cands}, key=str)
                    fids = {c[0] for c in cands}
                    for _key, gaps in prov.gap_answers(owners).items():
                        if gaps is None:
                            st["_lidar_rule"] = True
                        for fid, a, b in gaps or ():
                            if fid in fids:
                                got.setdefault(fid, []).append((tuple(a), tuple(b)))
        except Exception as ex:             # (the rule rather than no ranges)
            log.debug("lidar gaps: %s: %s", type(ex).__name__, ex)
            got = {}
        st["_lidar_answers"] = got
    return got


def _ring_fraction(ring, p):
    """The fraction of lon/lat polyline ring's length (sub_polyline's metric) at its point nearest p (lon, lat)."""
    ring = np.asarray(ring, np.float64)[:, :2]
    c = math.cos(math.radians(float(np.mean(ring[:, 1]))))
    q = ring * [c, 1.0]
    pp = np.asarray(p, np.float64)[:2] * [c, 1.0]
    a, b = q[:-1], q[1:]
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(1), 1e-30)
    t = np.clip(((pp - a) * ab).sum(1) / l2, 0.0, 1.0)
    d = np.hypot(*(a + t[:, None] * ab - pp).T)
    k = int(np.argmin(d))
    seg = np.sqrt(l2)
    s = np.r_[0.0, np.cumsum(seg)]
    return float((s[k] + t[k] * seg[k]) / max(s[-1], 1e-30))


def _lidar_gaps(feats, i, rngs):
    """rngs (rule_level_ranges) with the stretches the lidar found elevated (_lidar_answers) made bridge, never over a
    tunnel's range. Pure given feats and roadz's answer."""
    if not roadbase.LIDAR_GAPS_ON or any(k == "tunnel" for _, _, k in rngs):
        return rngs
    got = _lidar_answers(feats).get(str(feats.ids[i]))
    if not got:
        return rngs
    rings = [np.asarray(r, np.float64) for part in feats.parts(i) for r in part if len(r) >= 2]
    if not rings:
        return rngs
    add = []
    for a, b in got:
        fa, fb = sorted((_ring_fraction(rings[0], a), _ring_fraction(rings[0], b)))
        if fb - fa > MIN_PART:
            add.append((0.0 if fa < 1e-4 else fa, 1.0 if fb > 1.0 - 1e-4 else fb, "bridge"))
    out = []
    for a, b, k in sorted(list(rngs) + add):
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b), "bridge")
        else:
            out.append((a, b, k))
    return out


def _level_ranges(feats, i):
    """tag_level_ranges with the grade separations the map has no bridge for (gradesep.py: the cell's own, from its
    roads and its neighbours'), where no range of the map's is, and its underpasses' tunnel ranges made ground."""
    out = tag_level_ranges(feats, i)
    hook = feats.__dict__.get("_bnge_gradesep")
    extra = hook(feats).get(i) if hook is not None else None
    if not extra:
        return out
    for a, b, _k in (r for r in extra if r[2] == "ground"):       # (an underpass: its tunnel range is ground)
        cut = []
        for fa, fb, k in out:
            if k != "tunnel" or fb <= a or fa >= b:
                cut.append((fa, fb, k))
                continue
            if fa < a - 1e-5:
                cut.append((fa, a, k))
            if fb > b + 1e-5:
                cut.append((b, fb, k))
        out = cut
    add = []
    for a, b, _k in (r for r in extra if r[2] == "bridge"):
        for fa, fb, _ in out:
            if fb <= a or fa >= b:
                continue
            if fa > a:
                add.append((a, fa, "bridge"))
            a = max(a, fb)
            if a >= b:
                break
        if b - a > 1e-5:
            add.append((a, b, "bridge"))
    merged = []                                 # (touching bridge ranges joined: the map's fractions are float32s)
    for a, b, k in sorted(out + add):
        if merged and k == merged[-1][2] and a <= merged[-1][1] + 1e-5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), k)
        else:
            merged.append((a, b, k))
    return merged


def tag_level_ranges(feats, i):
    """Feature i's off-ground ranges as the map's flags and level rules have them (level_ranges, before gradesep):
    the flags first, a level range only where no flag says otherwise (Tower Bridge: level 1 over 0.03-0.88 of its
    segment, flags bridge / tunnel (through a tower) / bridge / tunnel / bridge: merged, tunnel winning, the whole
    bridge was a tunnel, under water for most of it, and left out, 2026-10-18); overlapping ranges merged (a tunnel
    wins over a bridge), touching ones of the same kind joined; a tunnel shorter than PASSAGE_M between two bridge
    ranges is a passage through a bridge's tower (the bridge goes on); a segment's tunnel ranges closer than
    TUNNEL_GAP_M are one tunnel (Overture splits a tunnel's range: 0.08-0.09, 0.10-0.13, 0.14-0.69 of the Fort McHenry
    Tunnel's segment). Every feature of the layer at once (earthfast.tag_level_ranges), kept on feats."""
    got = feats.__dict__.get("_tag_ranges")
    if got is None:
        got = feats.__dict__["_tag_ranges"] = _tag_ranges_all(feats)
    return list(got[i])


def _tag_rule_rows(t, name):
    """(feature, start, end, kind) arrays of a flags / level rule table (kind 1 bridge, 2 tunnel, 0 neither)."""
    if t is None or len(t.feature) == 0:
        return (np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, np.int8))
    kind = np.zeros(len(t.feature), np.int8)
    for j, v in enumerate(t.value.tolist() if hasattr(t.value, "tolist") else t.value):
        if name == "flags":
            v = str(v)
            kind[j] = 2 if "is_tunnel" in v else 1 if "is_bridge" in v else 0
        else:
            try:
                lv = float(v)
            except (TypeError, ValueError):
                lv = 0.0
            kind[j] = 1 if lv > 0 else 2 if lv < 0 else 0
    return (np.ascontiguousarray(t.feature, np.int64), np.ascontiguousarray(t.start, np.float64),
            np.ascontiguousarray(t.end, np.float64), kind)


_KIND_NAMES = {1: "bridge", 2: "tunnel"}


def _tag_ranges_list(feats):
    """Every feature's tag_level_ranges (the lists themselves: callers must not change them), kept on feats."""
    got = feats.__dict__.get("_tag_ranges")
    if got is None:
        got = feats.__dict__["_tag_ranges"] = _tag_ranges_all(feats)
    return got


def _tag_ranges_all(feats):
    import fast
    fast.require(142, "tag_level_ranges")
    ll = np.asarray(feats.lonlat, np.float64)
    off, rng, kinds = fast._ef.tag_level_ranges(
        len(feats), *_tag_rule_rows(feats.rules.get("flags"), "flags"),
        *_tag_rule_rows(feats.rules.get("level"), "level"), np.ascontiguousarray(ll),
        np.ascontiguousarray(feats.coord_off, np.int64), np.ascontiguousarray(feats.ring_off, np.int64),
        np.ascontiguousarray(feats.part_off, np.int64), PASSAGE_M, TUNNEL_GAP_M)
    rl, kl, ol = rng.tolist(), kinds.tolist(), off.tolist()
    return [[(rl[r][0], rl[r][1], _KIND_NAMES[kl[r]]) for r in range(ol[i], ol[i + 1])] for i in range(len(feats))]


def _feature_length_m(feats, i):
    """Length of feature i in metres (its rings, lon scaled by cos lat). Pure."""
    total = 0.0
    for part in feats.parts(i):
        for ring in part:
            r = np.asarray(ring, np.float64)
            if len(r) >= 2:
                c = math.cos(math.radians(float(np.mean(r[:, 1]))))
                total += float(np.hypot(np.diff(r[:, 0]) * c, np.diff(r[:, 1])).sum()) * 111320.0
    return total


def sub_polyline(ring, a, b):
    """The part of a lon/lat polyline (k, >=2) between fractions a < b of its length (metres along it,
    lon scaled by cos lat). Pure."""
    ring = np.asarray(ring, np.float64)
    c = math.cos(math.radians(float(np.mean(ring[:, 1]))))
    seg = np.hypot(np.diff(ring[:, 0]) * c, np.diff(ring[:, 1]))
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return ring[:1]
    sa, sb = a * s[-1], b * s[-1]
    inner = ring[(s > sa) & (s < sb)]
    pa = np.array([np.interp(sa, s, ring[:, d]) for d in range(ring.shape[1])])
    pb = np.array([np.interp(sb, s, ring[:, d]) for d in range(ring.shape[1])])
    return np.vstack([pa, inner, pb])


def _parts_by_level(feats, i, want):
    memo = feats.__dict__.setdefault("_parts_by_level", {})      # remembered: cells stay decoded (mapdata)
    got = memo.get((i, want))
    if got is None:
        got = memo[(i, want)] = _parts_by_level_new(feats, i, want)
    return list(got)


def ground_ranges(rngs):
    """The on-ground (start, end) fractions of a feature with level_ranges rngs."""
    pieces, at = [], 0.0
    for a, b, _ in rngs:
        if a - at > MIN_PART:
            pieces.append((at, a))
        at = max(at, b)
    if 1.0 - at > MIN_PART:
        pieces.append((at, 1.0))
    return pieces


def _parts_by_level_new(feats, i, want, rngs=None):
    rngs = level_ranges(feats, i) if rngs is None else rngs
    if want == "ground":
        pieces = ground_ranges(rngs)
    else:
        pieces = [(a, b) for a, b, k in rngs if k == want and b - a > MIN_PART]
    out = []
    for part in feats.parts(i):
        for ring in part:
            if len(ring) < 2:
                continue
            for a, b in pieces:
                r = ring if (a, b) == (0.0, 1.0) else sub_polyline(ring, a, b)
                if len(r) >= 2:
                    out.append(r)
    return out


def bridge_levels(feats, i):
    """The level of each of bridge_parts(feats, i) (same order): the highest level rule overlapping its
    range, 1 for a bridge flagged without one. Overture's levels order stacked roads. Pure."""
    t = feats.rules.get("level")
    lv = []
    for a, b, k in level_ranges(feats, i):
        if k != "bridge" or b - a <= MIN_PART:
            continue
        best = 1.0
        if t is not None and len(t.feature):
            for j in t.rows(i):
                if t.end[j] > a and t.start[j] < b:
                    try:
                        best = max(best, float(t.value[j]))
                    except (TypeError, ValueError):
                        pass
        lv.append(best)
    rings = sum(1 for part in feats.parts(i) for ring in part if len(ring) >= 2)
    return lv * rings


def ground_parts(feats, i):
    """The on-ground pieces of feature i (lon/lat polylines): its rings minus the bridge / tunnel ranges."""
    return _parts_by_level(feats, i, "ground")


def bridge_parts(feats, i):
    """The bridge pieces of feature i (lon/lat polylines)."""
    return _parts_by_level(feats, i, "bridge")


def tunnel_parts(feats, i):
    """The tunnel pieces of feature i (lon/lat polylines), each end made longer where its portal would stand on a road
    it passes under (_clear_portals)."""
    memo = feats.__dict__.setdefault("_parts_by_level", {})
    got = memo.get((i, "tunnel+"))
    if got is None:
        got = memo[(i, "tunnel+")] = _clear_portals(feats, i, _parts_by_level(feats, i, "tunnel"))
    return list(got)


def _clear_portals(feats, i, parts):
    """parts (lon/lat polylines of feature i's tunnel ranges) with each end near a road the tube passes under (a road
    on the ground the part crosses, not joined at that end) carried on straight until the portal face (roadmesh: the
    tube's half width and its frame either side) is PORTAL_CLEAR_M clear of its carriageway and the mouth past the
    ground graded with it (grading.T_MAX beyond its edge: the floor at a mouth is that ground's height, and a mouth on
    the road's level put the tube's end over the road); an end that does not clear within PORTAL_EXTEND_MAX_M stays.
    Pure."""
    if not parts:
        return parts
    ts = tunnel_specs(feats)
    if ts[i] is None:
        return parts
    half = ts[i].width / 2 + roadmesh.TUNNEL_SHOULDER + roadmesh.PORTAL_FRAME
    out = []
    for ring in parts:
        ring = np.asarray(ring, np.float64)
        kx = 111320.0 * math.cos(math.radians(float(ring[0, 1])))
        if float(np.hypot(*(np.diff(ring, axis=0) * [kx, 111320.0]).T).sum()) <= UNDERPASS_MAX_M:
            for at_start in (True, False):
                ring = _clear_end(feats, i, ring, at_start, half)
        out.append(ring)
    return out


def _crosses(a, b):
    """Do the polylines a and b (k, 2) cross? (Every segment pair at once.)"""
    if len(a) < 2 or len(b) < 2:
        return False
    p, r = a[:-1, None, :], (a[1:] - a[:-1])[:, None, :]
    c, s_ = b[None, :-1, :], (b[1:] - b[:-1])[None, :, :]
    den = r[..., 0] * s_[..., 1] - r[..., 1] * s_[..., 0]
    ok = np.abs(den) > 1e-12
    w = c - p
    dn = np.where(ok, den, 1.0)
    t = np.where(ok, (w[..., 0] * s_[..., 1] - w[..., 1] * s_[..., 0]) / dn, -1.0)
    u = np.where(ok, (w[..., 0] * r[..., 1] - w[..., 1] * r[..., 0]) / dn, -1.0)
    return bool(np.any(ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)))


def _portal_roads(feats):
    """The ground parts of feats' PORTAL_CLASSES roads (road_specs(feats)) as float arrays with their bounds, in feature
    order, and their half widths: made once per layer for _clear_end (each tunnel end looked over every feature: 0.85 s
    of 8 Midtown tiles)."""
    got = feats.__dict__.get("_portal_roads")
    if got is not None:
        return got
    rs = road_specs(feats)
    js, parts = [], []
    for j in range(len(feats)):
        if rs[j] is None or rs[j].cls not in PORTAL_CLASSES:
            continue
        for g in ground_parts(feats, j):
            js.append(j)
            parts.append(np.asarray(g, np.float64))
    bb = np.array([(g[:, 0].min(), g[:, 0].max(), g[:, 1].min(), g[:, 1].max()) for g in parts]).reshape(-1, 4)
    idx = {"j": np.asarray(js, np.int64), "parts": parts, "minx": bb[:, 0], "maxx": bb[:, 1], "miny": bb[:, 2],
           "maxy": bb[:, 3], "hw": [rs[j].width / 2 for j in js]}
    feats.__dict__["_portal_roads"] = idx
    return idx


def _clear_end(feats, i, ring, at_start, half):
    import grading
    p0 = ring[0] if at_start else ring[-1]
    kx, ky = 111320.0 * math.cos(math.radians(p0[1])), 111320.0      # m per degree there
    m = (ring - p0) * [kx, ky]
    tail = m if at_start else m[::-1]                                  # from the end inwards
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(tail, axis=0).T))]
    if s[-1] < 1.0:
        return ring
    q = tail[min(np.searchsorted(s, 3.0), len(tail) - 1)]
    d = -q / max(float(np.hypot(*q)), 1e-9)                            # out of the tunnel
    nrm = np.array([-d[1], d[0]])
    reach = (half + PORTAL_EXTEND_MAX_M + 40.0) / ky                   # roads that could matter, in degrees
    roads_m = []
    idx = _portal_roads(feats)
    near = np.flatnonzero((idx["miny"] <= p0[1] + reach) & (idx["maxy"] >= p0[1] - reach) &
                          ((idx["minx"] - p0[0]) * kx / ky <= reach) & ((p0[0] - idx["maxx"]) * kx / ky <= reach) &
                          (idx["j"] != i)) if len(idx["j"]) else ()
    for n_ in near:
        g = idx["parts"][n_]
        gm = (g - p0) * [kx, ky]
        a_, d_ = gm[:-1], np.diff(gm, axis=0)
        t_ = np.clip(-(a_ * d_).sum(1) / np.maximum((d_ * d_).sum(1), 1e-9), 0.0, 1.0)
        if float(np.hypot(*(a_ + t_[:, None] * d_).T).min()) < PORTAL_JOIN_M or not _crosses(m, gm):
            continue                        # joined here (an end at it or a T junction) / not over the tunnel
        hw = idx["hw"][n_]
        roads_m.append((gm, ((0, hw + grading.T_MAX), (1, hw + PORTAL_CLEAR_M), (2, hw + PORTAL_CLEAR_M))))

    def clear(c):
        pts = np.array([c, c + nrm * half, c - nrm * half])
        for gm, needs in roads_m:
            a, b = gm[:-1], gm[1:]
            dd = b - a
            L2 = np.maximum((dd * dd).sum(1), 1e-9)
            for k, need in needs:
                t = np.clip(((pts[k] - a) * dd).sum(1) / L2, 0.0, 1.0)
                if np.hypot(*(a + t[:, None] * dd - pts[k]).T).min() < need:
                    return False
        return True
    if not roads_m or clear(np.zeros(2)):
        return ring
    for ext in np.arange(1.0, PORTAL_EXTEND_MAX_M + 1e-6, 1.0):
        if clear(d * ext):
            k = max(1, int(math.ceil(ext / 4.0)))
            add = np.array([d * ext * (t / k) for t in range(1, k + 1)]) / [kx, ky] + p0
            return np.vstack([add[::-1], ring]) if at_start else np.vstack([ring, add])
    return ring


def surface_token(surface):
    """A surface rule value's base token: the detail after a colon and a second value after a semicolon dropped
    (`sett:pattern` -> `sett`, `sett;asphalt` -> `sett`); None for no value. surface_kind and look_of read it."""
    if surface is None:
        return None
    return str(surface).lower().split(";")[0].split(":")[0].strip()


def surface_kind(cls, surface):
    """'paved', 'cobble', 'gravel' or 'dirt' from the class and the surface rule value (None if unknown).

    The value's base token decides: OSM writes the detail after a colon (`cobblestone:flattened`,
    `paving_stones:30`, `sett:pattern`) and more than one value after a semicolon, and matched whole they all fell
    through to "dirt", which is drawn with gravel's own material (LOOK) and painted DIRT: İzmir's sett streets read
    as gravel tracks (2026-10-22). Interlocking pavers have their own names in the map. A value not understood at
    all is the class's own surface, not dirt: an unknown tag must not turn a residential street into a track."""
    if surface is not None:
        s = surface_token(surface)
        if s in PAVED:
            return "paved"
        if s in COBBLE or s in COBBLE_ALIASES:
            return "cobble"
        if s in GRAVEL:
            return "gravel"
        if s == "unpaved" and cls != "track":
            # (generic "unpaved" on a road cars use is a gravel road far more often than a dirt track: the
            # mapped unpaved roads of Ontario, Finland and Estonia, 2026-11-03; a track stays dirt)
            return "gravel"
        if s in UNPAVED:
            return "dirt"
    return "paved" if STYLE.get(cls, STYLE["unknown"])[1] else "dirt"


def _surface_value(feats, i):
    """Feature i's surface value: OpenStreetMap's own (osm_surface, joined by way id: osmtags.py), else Overture's
    rule. Overture keeps a short list and folds OSM's compacted / fine_gravel into "unpaved" (drawn as dirt): Finland's
    compacted-gravel forest roads came out as dirt tracks (the user 2026-11-03: players asking for gravel roads)."""
    osm = feats.attrs.get("osm_surface")
    if osm is not None and i < len(osm):
        v = str(osm[i]).strip()
        if v and v.lower() not in ("nan", "none"):
            return v
    return _rule_value(feats.rules.get("surface"), i)


REGIONAL_ON = os.environ.get("BNE_REGIONAL_SURFACE", "on") != "off"


def _regional_rows(feats):
    """(ids, classes, lengths m, unpaved) of feats' REGIONAL_CLASSES roads that have a surface value. Kept on feats."""
    got = feats.__dict__.get("_regional_rows")
    if got is not None:
        return got
    cls_a = feats.attrs.get("class")
    ids, cl, un, idx = [], [], [], []
    if cls_a is not None:
        for i in range(len(feats)):
            c = str(cls_a[i])
            if c not in REGIONAL_CLASSES:
                continue
            v = _surface_value(feats, i)
            if v is None:
                continue
            ids.append(str(feats.ids[i]))
            cl.append(c)
            un.append(surface_kind(c, v) in ("gravel", "dirt"))
            idx.append(i)
    lens = _lengths(feats)[np.asarray(idx, np.int64)] if idx else np.zeros(0)
    got = feats.__dict__["_regional_rows"] = (ids, cl, np.nan_to_num(np.asarray(lens, np.float64)), un)
    return got


def regional_unpaved(feats):
    """The REGIONAL_CLASSES whose surfaced roads in feats' cell and the eight around it (mapdata's _bnge_neighbours)
    are mostly unpaved (REGIONAL_*): a class by its own roads when it has REGIONAL_MIN_M of them, else by the minor
    classes' together. Each road once. Kept on feats."""
    got = feats.__dict__.get("_regional_unpaved")
    if got is not None:
        return got
    layers = [feats]
    hook = feats.__dict__.get("_bnge_neighbours")
    if hook is not None:
        try:
            layers += [f for f in hook() if f is not None and len(f)]
        except Exception as e:          # (this cell alone)
            log.debug("regional surface: neighbours: %s: %s", type(e).__name__, e)
    seen, tot, unp = set(), {}, {}
    for f in layers:
        ids, cl, lens, un = _regional_rows(f)
        for fid, c, ln, u in zip(ids, cl, lens.tolist(), un):
            if fid in seen:
                continue
            seen.add(fid)
            tot[c] = tot.get(c, 0.0) + ln
            if u:
                unp[c] = unp.get(c, 0.0) + ln
    pool_t, pool_u = sum(tot.values()), sum(unp.values())
    pooled = pool_t >= REGIONAL_POOL_MIN_M and pool_u >= REGIONAL_SHARE * pool_t
    got = frozenset(c for c in REGIONAL_CLASSES
                    if (unp.get(c, 0.0) >= REGIONAL_SHARE * tot[c] if tot.get(c, 0.0) >= REGIONAL_MIN_M else pooled))
    feats.__dict__["_regional_unpaved"] = got
    return got


def _regional_gravel(feats, i, cls, surf):
    """Is untagged road i of class cls a REGIONAL_* gravel candidate (before the town test)?"""
    if not REGIONAL_ON or surf is not None or cls not in REGIONAL_CLASSES:
        return False
    if _osm(feats, "lanes", i, *OSM_LANES) is not None:
        return False
    sm = feats.attrs.get("osm_smoothness")
    if sm is not None and str(sm[i]).strip().lower() in GOOD_SMOOTHNESS:
        return False
    return cls in regional_unpaved(feats)


def carriageway(cls, kind, urban, rule_width=None, oneway=False, lanes=None, link=False):
    """Paved / dirt width (m) of a road: the width rule if any, else the driving width (OpenStreetMap's lanes x the
    class's LANE_W when known, two lanes for untagged highways, else the class's) plus parking lanes (urban paved
    streets; one side on a one-way street)
    or edge strips (rural paved roads). One-way: ONEWAY_STYLE's driving width (motorways and trunks are per carriageway
    already). Pure."""
    # Preserve the customized travel lanes and shoulders in towns and on bridges too.
    # A narrow mapped width must not consume the shoulders or a mapped travel lane.
    if kind in BUILT and link and cls in ("motorway", "trunk"):
        return max(rule_width or 0., (lanes or (1 if oneway else 2)) * LINK_LANE_M + 2 * LINK_SHOULDER_M)
    if kind in BUILT and cls == "motorway" and oneway:
        return max(rule_width or 0., (lanes or 2) * LANE_W[cls] + 2 * SHOULDER_M)
    if rule_width is not None:
        return rule_width
    if lanes is None and cls in ("motorway", "trunk"):
        lanes = 2
    w = STYLE.get(cls, STYLE["unknown"])[0]
    if oneway and cls in ONEWAY_STYLE:
        w = ONEWAY_STYLE[cls]
    if lanes:
        w = lanes * LANE_W.get(cls, LANE_W_DEFAULT)
    if kind not in BUILT:
        return w
    if lanes and not urban and oneway and cls == "motorway":
        return w + 2 * (SHOULDER_WIDE_M if lanes >= SHOULDER_WIDE_LANES else SHOULDER_M)
    if urban and not oneway and cls in URBAN_ARTERIAL and not lanes:
        return URBAN_ARTERIAL[cls]          # four lanes, no parking (a walled street is measured instead)
    if urban and cls in PARKING_CLASSES:
        return w + (1 if oneway else 2) * PARKING_M
    if not urban and cls != "service":
        if oneway and cls == "motorway":
            return MOTORWAY_RURAL_M + 2 * SHOULDER_M
        return w + 2 * EDGE_STRIP_CLASS_M.get(cls, EDGE_STRIP_M)
    return w


def _link_width(lanes):
    """Existing ramp pavement widths, extended by a travel lane for larger counts."""
    return LINK_M.get(lanes, LINK_M[2] + (lanes - 2) * LINK_LANE_M)


def _lane_count(feats, i, cls, flow=True):
    """A road's lane count: a correction (rampoverrides), OpenStreetMap's own count, PR #471's validated ones (a
    geometry match, a gap between two agreeing ends: roadlanes.generated_counts), then (flow) the one-way roads joined
    to it (lane_flow), else two lanes per motorway / trunk way, one on a one-way ramp (PR #471), None otherwise."""
    import roadlanes
    over = _lane_overrides(feats).get(i)
    if over is not None:
        return over
    lanes = _osm(feats, "lanes", i, *OSM_LANES)
    if lanes is not None:
        return int(lanes)
    lanes = roadlanes.generated_counts(feats)[i]
    # what flows in or out (lane_flow): a motorway's / trunk's only where a ramp merges or splits at its end beside a
    # through road (I-95 by the Middle Branch: the deck after a ramp's merge at the 2-lane default, a 4.3 m step), not
    # from a ramp alone, the other direction, a crossing road or a fork (PR #471: two agreeing ends, roadlanes)
    if lanes is None and flow and (cls not in roadlanes.HIGHWAYS or _ramp_junction(feats)[i]):
        lanes = lane_flow(feats)[i]
    if lanes is None and not is_ring(feats, i):
        if roadlanes.is_link(feats, i):
            return 1 if _rule_value(feats.rules.get("oneway"), i) is not None else 2
        if cls in roadlanes.HIGHWAYS:
            return 2
    return lanes


def _ramp_junction(feats):
    """Per feature: a motorway / trunk (not a ramp) one of whose ends meets both a ramp's end and another through
    motorway / trunk's (a merge or a split), in this cell. Pure; once per Features."""
    got = feats.__dict__.get("_ramp_junction")
    if got is not None and len(got) == len(feats):
        return got
    import roadlanes
    n = len(feats)
    cls = feats.attrs.get("class", [""] * n)
    ends = {}                   # rounded end point -> [(feature, is a ramp)]
    mine = [[] for _ in range(n)]
    for i in range(n):
        if cls[i] not in roadlanes.HIGHWAYS:
            continue
        parts = feats.parts(i)
        if not parts or not parts[0] or len(parts[0][0]) < 2:
            continue
        ll = np.asarray(parts[0][0], np.float64)
        link = roadlanes.is_link(feats, i)
        for p in (ll[0], ll[-1]):
            k = (int(round(p[0] / RAMP_JOIN_DEG)), int(round(p[1] / RAMP_JOIN_DEG)))
            ends.setdefault(k, []).append((i, link))
            mine[i].append(k)
    out = np.zeros(n, bool)
    for i in range(n):
        if cls[i] not in roadlanes.HIGHWAYS or roadlanes.is_link(feats, i):
            continue
        for k in mine[i]:
            near = [e for dk in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
                    for e in ends.get((k[0] + dk[0], k[1] + dk[1]), ()) if e[0] != i]
            if any(lk for _j, lk in near) and any(not lk for _j, lk in near):
                out[i] = True
                break
    feats.__dict__["_ramp_junction"] = out
    return out


def _lane_overrides(feats):
    """{feature: lanes} of rampoverrides (PR #471), once per Features."""
    d = feats.__dict__.get("_lane_overrides")
    if d is None:
        import rampoverrides
        d = feats.__dict__["_lane_overrides"] = rampoverrides.lane_counts(feats)
    return d


def rural_motorway(spec):
    """One carriageway of a rural motorway: two lanes between shoulders (carriageway)."""
    return spec.cls == "motorway" and spec.oneway and not spec.urban and spec.kind in BUILT


def highway_shoulders(spec):
    return not spec.mountain and spec.kind in BUILT and (spec.cls == "motorway" and spec.oneway
                                  or spec.link and spec.cls in ("motorway", "trunk"))


def edge_margin(spec, side=None):
    """Shoulder/edge-strip metres. side is +1 left or -1 right of the geometry.

    One-way highways retain the earlier 1.5 m left / 4.5 m right shoulders;
    ramps use .875 / 2.625 m. No side requests their mean for lane_layout.
    """
    if highway_shoulders(spec):
        base = motorway_shoulder(spec)
        factor = 1. if side is None or not spec.oneway else .5 if side == (spec.dir or 1) else 1.5
        return base * factor
    if spec.urban or spec.kind not in BUILT or spec.cls == "service" or spec.mountain:
        return 0.0
    return EDGE_STRIP_CLASS_M.get(spec.cls, EDGE_STRIP_M)


def driving_offset(spec):
    """Signed offset of the travel lanes from the pavement's geometric centre."""
    return (edge_margin(spec, -1) - edge_margin(spec, 1)) / 2


def motorway_shoulder(spec):
    """Mean shoulder outside the mapped travel lanes, bounded by the configured width."""
    lane = LINK_LANE_M if spec.link else LANE_W.get(spec.cls, LANE_W_DEFAULT)
    count = spec.lanes or (1 if spec.link and spec.oneway else 2)
    cap = LINK_SHOULDER_M if spec.link else SHOULDER_WIDE_M if count >= SHOULDER_WIDE_LANES else SHOULDER_M
    return float(np.clip((spec.width - count * lane) / 2, 0., cap))


def _dir_kmh(feats, i, ncon, lengths=None):
    """(one-way direction along the geometry +1 / -1 or 0, speed limit km/h or None, m per junction or None)
    of feature i; ncon: connectors per feature (None: unknown); lengths: _lengths(feats) (None: measured
    here)."""
    rules = feats.rules
    v = _rule_value(rules.get("oneway"), i)
    d = 0 if v is None else (-1 if float(v) < 0 else 1)
    k = _rule_value(rules.get("max_speed_kmh"), i)
    gap = None
    if ncon is not None and ncon[i] >= 2:
        L = (lengths if lengths is not None else _lengths(feats, i))[i if lengths is not None else 0]
        if L is not None and not np.isnan(L):
            gap = float(L) / (ncon[i] - 1)
    return d, (float(k) if k is not None and float(k) > 0 else None), gap


def _lengths(feats, only=None):
    """Length (m, lon scaled by the cos of each feature's first latitude) of every feature's geometry, NaN for
    fewer than two points; only: just that feature (an array of one). All features at once: one numpy pass
    instead of ~3000 small ones a cell."""
    po, ro, co = feats.part_off, feats.ring_off, feats.coord_off
    idx = np.arange(len(feats)) if only is None else np.array([only])
    a = np.asarray(co[ro[po[idx]]], np.int64)
    b = np.asarray(co[ro[po[idx + 1]]], np.int64)
    out = np.full(len(idx), np.nan)
    ll = np.asarray(feats.lonlat, np.float64)
    cnt = b - a
    if not len(idx) or cnt.sum() < 2:
        return out
    m = 111320.0
    fid = np.repeat(np.arange(len(idx)), cnt)
    pos = np.concatenate([np.arange(x, y) for x, y in zip(a, b)]) if only is not None else np.arange(a[0], b[-1])
    pl = ll[pos]
    same = fid[1:] == fid[:-1]
    c = np.cos(np.radians(ll[np.minimum(a, len(ll) - 1), 1]))
    seg = np.hypot(np.diff(pl[:, 0]) * m * c[fid[:-1]], np.diff(pl[:, 1]) * m)
    L = np.bincount(fid[:-1][same], weights=seg[same], minlength=len(idx))
    ok = cnt >= 2
    out[ok] = L[ok]
    return out


def _dual_side(a, b):
    """For polylines a, b (k, 2) in metres: the share of a's points (every ~10 m) with b beside it (DUAL_GAP_M,
    parallel within DUAL_ANGLE_DEG) and the mean side b is on (+1 left / -1 right of a's direction)."""
    import fast
    fast.require(0, "dual_side")       # the Rust kernel (~25k calls a Las Vegas road tile set)
    return fast._ef.dual_side(np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64),
                              DUAL_GAP_M[0], DUAL_GAP_M[1], math.cos(math.radians(DUAL_ANGLE_DEG)))


def dual_carriageways(feats, specs, left_hand=False):
    """specs with the two carriageways of a divided road made one-way when the map says neither is (Boston St,
    Baltimore: two "primary" roads either side of a wide median, each with its own double yellow centre line, the
    user's report 2026-10-14). Two roads of one name and class running side by side DUAL_GAP_M apart over most of
    the shorter one; each runs so that the other is on its left (driving on the right) or right. Pure."""
    groups = {}
    names = feats.attrs.get("name")
    if names is None:
        return specs
    for i, sp in enumerate(specs):
        if sp is None or sp.cls not in DUAL_CLASSES:     # (one-way already: paired for its median only)
            continue
        nm = str(names[i] or "").strip()
        named = bool(nm) and nm != "None"
        if named:
            groups.setdefault((nm, sp.cls), []).append(i)
        if sp.oneway and not sp.link and (not named or sp.cls in ("motorway", "trunk")):
            # (a numbered road has no name: Sweden's 2+1 roads, below; and most motorways: a French city's ring road, two
            # carriageways 18 and 14 m wide 10.4 m apart, overlapped 5.6 m, no room for its median barrier, #2374); a
            # named motorway's / trunk's too, kept from overlapping a one-way carriageway of another name (a French ring road's: "Périphérique
            # Intérieur" and "Extérieur", never paired as one road)
            groups.setdefault(("", sp.cls), []).append(i)
    out = list(specs)
    for (group_name, _), idx in groups.items():
        if len(idx) < 2:
            continue
        lines = {}
        for i in idx:
            a = feats.coord_off[feats.ring_off[feats.part_off[i]]]
            b = feats.coord_off[feats.ring_off[feats.part_off[i + 1]]]
            ll = feats.lonlat[a:b]
            lines[i] = ll
        lat0 = float(np.mean(np.concatenate(list(lines.values()))[:, 1]))
        m = np.array([111320.0 * math.cos(math.radians(lat0)), 110540.0])
        ref = lines[idx[0]][0]
        xy = {i: (ll - ref) * m for i, ll in lines.items()}
        for i in idx:
            best = None
            for j in idx:
                if i == j:
                    continue
                share, side = _dual_side(xy[i], xy[j])
                if share < DUAL_SHARE and abs(side) > 0.5:
                    # (the shorter one's share: a long carriageway beside a shorter piece of the other was never paired
                    # with it, only that piece with the long one)
                    share = max(share, _dual_side(xy[j], xy[i])[0])
                if share >= DUAL_SHARE and abs(side) > 0.5:
                    # the nearest of the carriageways beside it (by share, cells holding different neighbours
                    # paired one piece of Boston Street 20 m and 38 m away: a sidewalk on its median in one tile)
                    # (from either side's points: a long carriageway drawn with few points has none beside a
                    # shorter piece of the other, its own read 149 m for 11.4)
                    gap = min(_centre_gap(xy[i], xy[j]), _centre_gap(xy[j], xy[i]))
                    if best is None or gap < best[3]:
                        best = (share, side, j, gap)
            if best is None:
                continue
            sp = out[i]
            # driving on the right the other carriageway is on the left
            gap = best[3]
            if not group_name:          # nameless: only so they don't overlap. Sweden's 2+1 carriageways are 6-8 m
                other = specs[best[2]]  # apart, and 11 m wide where no lane count came through: no room for a barrier.
                if (sp.width + other.width) / 2 + DUAL_MIN_MEDIAN_M > gap:  # (no median: a town's raised strip would
                    out[i] = sp._replace(width=dual_cap(gap, sp.width, sp.lanes, other.lanes))      # stand there in
                continue                # place of the barrier, the user's report 2026-11-02)
            if sp.oneway:               # the map's own one-way carriageway: its direction stands
                out[i] = sp._replace(median=gap, width=_dual_width(sp, sp.width, gap, feats, i))
                continue
            import roadlanes
            # A mapped two-way count is evidence against dividing this road (Portland's Rosa Parks Way,
            # 1+1 lanes and a turn lane). The highway fallback supplies no such evidence.
            if roadlanes.generated_counts(feats)[i]:
                continue
            d = (1 if best[1] > 0 else -1) * (-1 if left_hand else 1)
            w = sp.width if (not sp.parking or _rule_value(feats.rules.get("width"), i) is not None
                             or _osm(feats, "width", i, *OSM_WIDTH_M) is not None) else \
                carriageway(sp.cls, sp.kind, sp.urban, None, True, sp.lanes, link=sp.link)
            out[i] = sp._replace(oneway=True, dir=d, width=_dual_width(sp, w, gap, feats, i), median=gap)
    return out


def dual_cap(gap, w, lanes=None, other=None):
    """w capped so carriageways gap m apart (centre to centre) do not overlap (DUAL_MIN_MEDIAN_M between them), the room
    shared by their lanes where both counts are known: a 2+1 road's two-lane side next to its one-lane side, not two of
    one width (the user's report 2026-11-02: two lanes' markings on a road far too narrow for them)."""
    share = lanes / (lanes + other) if lanes and other else 0.5
    return float(min(w, max(2 * (gap - DUAL_MIN_MEDIAN_M) * share, DUAL_MIN_W)))


def _dual_width(sp, w, gap, feats, i):
    """A divided road carriageway's width: a mapped width stands, but never over the other carriageway (a whole road's
    width read onto each of its carriageways, IGN BD TOPO's in France: two 9 m carriageways 5.9 m apart, 2026-11-10);
    in a city with its centre lines within URBAN_MEDIAN_GAP_M, out to an URBAN_MEDIAN_M median (at most
    URBAN_DUAL_MAX_W); never overlapping the other (dual_cap)."""
    if (_rule_value(feats.rules.get("width"), i) is not None
            or _osm(feats, "width", i, *OSM_WIDTH_M) is not None):
        return float(min(w, max(gap, DUAL_MIN_W)))
    if sp.urban and sp.parking and gap <= URBAN_MEDIAN_GAP_M:
        w = float(np.clip(gap - URBAN_MEDIAN_M, w, URBAN_DUAL_MAX_W))
    return dual_cap(gap, w)


def _centre_gap(a, b):
    """The median distance (m) from polyline a's points to polyline b (both (k, 2) in metres) where b is beside it
    (within DUAL_GAP_M): a divided road's centre lines apart."""
    import fast
    fast.require(0, "polyline_dist")
    d = np.asarray(fast._ef.polyline_dist(np.ascontiguousarray(b, np.float64), np.ascontiguousarray(a, np.float64)))
    beside = d[d <= DUAL_GAP_M[1]]       # where it runs beside the other (a longer piece's far end inflated it 2x)
    return float(np.median(beside)) if len(beside) else float(np.median(d))


def road_specs(feats, urban_fn=None, widths=None, left_hand=False):
    """Per feature: None (left out) or (width m, surface kind). urban_fn(lat, lon) -> bool array (built-up
    land cover), None = rural everywhere; called once for every candidate's points (its cost is per
    call: 2026-10-11, one call per road made 4.4 of a physics block's 6 s). Pure given urban_fn.
    Remembered on feats (mapdata keeps decoded cells in memory: every tile of a cell asked again)."""
    return list(_road_specs_kept(feats, urban_fn, widths, left_hand))


def _road_specs_kept(feats, urban_fn=None, widths=None, left_hand=False):
    """road_specs' list itself, the one remembered on feats when it was (callers must not change it)."""
    memo = feats.__dict__.setdefault("_road_specs", {})
    import franceroads
    k = (urban_fn, widths, bool(left_hand))
    if franceroads.ON:
        k += (franceroads.VERSION,)             # isolate an in-process comparison; off keeps the existing key
    got = memo.get(k)
    if got is None:             # (the measured widths on the specs without them: those are asked for too, _bridge_reach)
        keep = True             # (specs made while a download failed are not remembered: asked again, PR #634)
        extra = (widths is not None, bool(left_hand)) + ((franceroads.VERSION,) if franceroads.ON else ())
        cid = feats.__dict__.get("_bnge_cell")
        # (the land cover's too, by this process's urban_fn: its own Surface, the same all along; a city layer's
        # specs with its parking widths were ~0.37 s of each of its road tiles after every reload, 2026-11-02)
        mk = None if cid is None else (cid, "specs", extra) if urban_fn is None else \
            (cid, "specs_urban", extra, id(getattr(urban_fn, "__self__", urban_fn)), getattr(urban_fn, "__name__", ""))
        if mk is not None:      # (_FACTS_MEM: made in this process before, for another copy of the layer)
            with _FACTS_LOCK:
                mem = _FACTS_MEM.get(mk)
            if mem is not None and len(mem) == len(feats):
                got = mem
        if got is None and urban_fn is None:    # (the land cover's are the server's own: not shared, cell_facts)
            got = _specs_from(_facts_load(feats, "specs", extra), len(feats))
        if got is None:
            import revisit
            f0 = revisit.thread_faults()
            got = (_road_specs(feats, urban_fn, left_hand) if widths is None else
                   _measured_widths(feats, road_specs(feats, urban_fn, None, left_hand), widths))
            got = _marking_specs(feats, got)
            clean = revisit.thread_faults() == f0
            if urban_fn is None and clean:
                _facts_store(feats, "specs", extra, [None if sp is None else _spec_json(sp) for sp in got])
            keep = clean
            if clean:
                if mk is not None:
                    with _FACTS_LOCK:
                        _FACTS_MEM[mk] = list(got)
                        while len(_FACTS_MEM) > roadbase.FACTS_MEM_KEPT:
                            _FACTS_MEM.popitem(last=False)
        if keep:
            got = memo.setdefault(k, got)
    return got


def road_tapers(feats, urban_fn=None, widths=None, left_hand=False):
    """Per feature: its tapers ((lon, lat, width at the joint m, length m), ...) (TAPER_*), with the widths of
    road_specs(feats, urban_fn, widths, left_hand); () none. Pure; remembered on feats."""
    memo = feats.__dict__.setdefault("_road_tapers", {})
    k = (urban_fn, widths, bool(left_hand))
    got = memo.get(k)
    if got is None:
        specs = road_specs(feats, urban_fn, widths, left_hand)
        got = _tapers(feats, specs)
        for i, sp in enumerate(specs):          # (a pass road's hairpins: hairpin_tapers)
            if sp is not None and sp.mountain:
                a = feats.coord_off[feats.ring_off[feats.part_off[i]]]
                b = feats.coord_off[feats.ring_off[feats.part_off[i] + 1]]
                hp = hairpin_tapers(feats.lonlat[a:b], sp.width)
                if hp:
                    got[i] = tuple(got[i] or ()) + hp
        memo[k] = got
    return got


def _tapers(feats, specs, decks=False):
    """road_tapers' kernel (earthfast.road_tapers): the per-road rows marshalled, the joints found in Rust."""
    import fast
    fast.require(120 if decks else 117, "road_tapers")
    n = len(feats)
    out = [()] * n
    bridge = bridge_specs(feats)
    if not n:
        return out
    starts = feats.coord_off[feats.ring_off[feats.part_off[:-1]]]
    stops = feats.coord_off[feats.ring_off[feats.part_off[1:]]]
    off = np.zeros(n + 1, np.int64)
    off[1:] = np.cumsum(stops - starts)
    lonlat = np.concatenate([feats.lonlat[a:b] for a, b in zip(starts, stops)]) if off[-1] else np.zeros((0, 2))
    attrs = np.full((n, 5), np.nan)
    codes, own = {}, []
    for i, sp in enumerate(specs):
        bs = bridge[i]
        if sp is None and bs is None:
            attrs[i, 1] = -1.0
            continue
        cls = (sp or bs).cls
        attrs[i, 1] = codes.setdefault(cls, len(codes))
        attrs[i, 2:4] = 0.0
        if sp is not None:
            attrs[i, 0] = sp.width
        if bs is not None:
            attrs[i, 4] = bs.width
            for a0, b0, kind in level_ranges(feats, i):
                if kind != "bridge":
                    continue
                if a0 <= 1e-3:
                    attrs[i, 2] = 1.0
                if b0 >= 1 - 1e-3:
                    attrs[i, 3] = 1.0
                own.extend((i, f) for f in (a0, b0) if 1e-3 < f < 1 - 1e-3)
    rows = fast._ef.road_tapers(np.ascontiguousarray(lonlat, np.float64), off, attrs,
                                np.asarray(own, np.float64).reshape(-1, 2),
                                np.array([TAPER_M_PER_M, TAPER_M[0], TAPER_M[1], TAPER_SHARE, TAPER_MIN_DW,
                                          TAPER_MAX_TURN] + ([1.0] if decks else []), np.float64))
    got = {}
    for r in np.asarray(rows):
        got.setdefault(int(r[0]), []).append((float(r[1]), float(r[2]), float(r[3]), float(r[4])))
    for i, t in got.items():
        out[i] = tuple(t)
    return out


def joint_desc(copies, fn):
    """The tapers / shifts (fn(feats)[i]: ((lon, lat, ...), ...)) of a road from its copies [(feats, i)] in the cells at
    hand, each joint's from the copy in the cell the joint lies in: that cell holds every road meeting there, a copy
    elsewhere may not and found another (a Springfield Interchange ramp's deck tapered in one cell and not in the
    next, a 0.9 m step; deckedges.py 2026-10-31). A joint whose cell is not at hand: any copy's. Pure."""
    out, other = {}, {}
    for f, i in copies:
        cid = f.__dict__.get("_bnge_cell")
        cid = cid[0] if cid else None
        for d in fn(f)[i] or ():
            key = (round(d[0], 6), round(d[1], 6))
            if cid is None or str(mapcells.cell_of(d[1], d[0])) == cid:
                out[key] = d
            else:
                other.setdefault(key, d)
    for key, d in other.items():
        out.setdefault(key, d)
    return tuple(out[k] for k in sorted(out))


def _id_rows(f):
    """{road id: [rows]} of a layer, made once (kept on it)."""
    got = f.__dict__.get("_id_rows")
    if got is None:
        got = {}
        for i, fid in enumerate(f.ids.tolist()):
            got.setdefault(str(fid), []).append(i)
        f.__dict__["_id_rows"] = got
    return got


class _Copies:
    """copies_of's answer: get(road id, default) -> [(feats, i)] of every copy, in the cells' order, from each layer's
    own index (_id_rows: a dict of every road of the cells made at each call was ~0.4 s of a Manhattan tile set)."""

    def __init__(self, cells):
        self.cells = [f for f in (cells or ()) if f is not None]

    def get(self, fid, default=None):
        fid = str(fid)
        out = [(f, i) for f in self.cells for i in _id_rows(f).get(fid, ())]
        return out if out else default


def copies_of(cells):
    """{road id: [(feats, i)]} (a _Copies, read with .get): every copy of each road in cells (joint_desc)."""
    return _Copies(cells)


def deck_tapers(feats):
    """Per feature: the tapers of its deck where it meets another deck end to end at another width (both run to half
    way: _tapers' deck mode), as road_tapers. A deck kept its width at such a joint: the I-95 viaduct by the Middle
    Branch stepped 1.8 m where a 24.0 m deck went on as a 20.4 m one (deckedges.py, the user 2026-10-31). Pure;
    remembered on feats."""
    got = feats.__dict__.get("_deck_tapers")
    if got is None:
        got = feats.__dict__["_deck_tapers"] = _tapers(feats, road_specs(feats), decks=True)
    return got


def deck_width_fn(frame, cells):
    """ws_of(fid, pts (k, 2+)) -> per point the width of deck part fid (its bridge spec's, tapered where it meets
    another deck: deck_tapers of its feature in cells), or None where no taper reaches it."""
    where = copies_of(cells)

    def ws_of(fid, pts):
        got = where.get(str(fid))
        if not got:
            return None
        tp = joint_desc(got, deck_tapers)
        bs = bridge_specs(got[0][0])[got[0][1]]
        if not tp or bs is None:
            return None
        pts = np.asarray(pts, np.float64)
        la, lo, _ = frame.enu_to_geodetic(pts[:, 0], pts[:, 1], 0.0)
        return taper_widths(bs.width, tp, lo, la)
    return ws_of


def taper_widths(width, tapers, lon, lat):
    """Per point (lon, lat arrays) the drawn width of a road width m wide with these tapers (road_tapers), smoothstepped
    from the width at each joint to its own over the taper's length; None where no taper reaches any point (width
    all along). Deterministic from absolute positions: every tile draws a road alike."""
    if not tapers:
        return None
    lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
    out, hit = np.full(lon.shape, float(width)), False
    for tlon, tlat, target, span in tapers:
        d = np.hypot((lon - tlon) * 111320.0 * math.cos(math.radians(tlat)), (lat - tlat) * 110540.0)
        near = d < span
        if span > 0 and near.any():
            hit = True
            t = np.clip(d / span, 0.0, 1.0)
            out += np.where(near, (target - width) * (1.0 - t * t * (3.0 - 2.0 * t)), 0.0)
    return out if hit else None


def lane_flow(feats):
    """Per feature: its lane count found from the one-way roads joined to it (LANE_FLOW_*), None where none was (or the
    map gives one): its home cell's (the cell of its first point, among this cell and the eight around it), solved
    over that cell and the eight around it (_lane_flow_local). A road crossing a cell's edge had another count in each
    cell (I-95 by the Middle Branch: 4 lanes, 20.4 m, in one and the default 11.8 m in the other, a 4.3 m step where
    their tiles met; then 24.0 m against 20.4 m where the 3 x 3 round each saw another stretch of the chain). Pure
    given the cells; remembered on feats."""
    return _cell_memo(feats, "_lane_flow_home", lambda: _lane_flow_home(feats))


def _lane_flow_home(feats):
    out = list(_lane_flow_local(feats))
    me = feats.__dict__.get("_bnge_cell")
    hook = feats.__dict__.get("_bnge_neighbours")
    if me is not None and hook is not None:
        try:
            near = {f.__dict__["_bnge_cell"][0]: f for f in hook() if f is not None and f.__dict__.get("_bnge_cell")}
        except Exception as e:          # (this cell's own)
            log.debug("lane flow: neighbours: %s: %s", type(e).__name__, e)
            near = {}
        rows = {}
        for i, home in enumerate(_home_cells(feats)):
            if home != me[0] and home in near:
                rows.setdefault(home, []).append(i)
        for home, idx in rows.items():
            other = near[home]
            res = _lane_flow_local(other)
            pos = {fid: j for j, fid in enumerate(other.ids.tolist())}
            for i in idx:
                j = pos.get(feats.ids[i])
                if j is not None:
                    out[i] = res[j]
    return out


_LANE_FLOW_DFLT = {}            # road class -> its one-way carriageway's lanes when the map gives none (_lane_flow_rows)


def _lane_flow_ids(f):
    """Layer f's road ids as UCS-4 rows (roadlanes.ucs4: lane_flow's dedup), kept on it."""
    got = f.__dict__.get("_lane_flow_ids")
    if got is None:
        import roadlanes
        got = f.__dict__["_lane_flow_ids"] = roadlanes.ucs4(f.ids)
    return got


def _lane_flow_rows(f):
    """Layer f's rows for lane_flow, made once per layer (kept on it): (first and last coordinate (n, 4: lon, lat, lon,
    lat; 0 without any), coordinate count, attrs (n, 4): one-way sign or 0, the map's lanes or nan, link, the class's
    default lanes or nan). Each cell's solve covers the 3 x 3 around it, so every layer's rows were made again for each
    of its neighbours' solves: 6 of lane_flow's 7 CPU-s in Manhattan's first visit, the solve itself 0.3 (2026-11-02);
    the rows of all features at once (2026-11-08)."""
    got = f.__dict__.get("_lane_flow_rows")
    if got is not None:
        return got
    import roadlanes
    n = len(f)
    po = np.asarray(f.part_off, np.int64)
    starts = np.asarray(f.coord_off, np.int64)[np.asarray(f.ring_off, np.int64)[po[:n]]]
    ends = np.asarray(f.coord_off, np.int64)[np.asarray(f.ring_off, np.int64)[po[1:n + 1]]]
    npts = np.ascontiguousarray(ends - starts, np.int64)
    pts = np.zeros((n, 4))
    some = npts > 0
    if some.any():
        ll = np.asarray(f.lonlat, np.float64)
        pts[some, :2] = ll[starts[some]]
        pts[some, 2:] = ll[ends[some] - 1]
    attrs = np.zeros((n, 4))
    attrs[:, 1] = np.nan
    attrs[:, 3] = np.nan
    cls_a, sub_a = f.attrs.get("class"), f.attrs.get("subtype")
    counts = roadlanes.generated_counts(f)
    table = f.rules.get("oneway")
    if cls_a is not None and n and table is not None and len(table.feature):
        cls = np.asarray(cls_a).astype(str)
        sel = np.isin(cls, sorted(LANE_FLOW_CLASSES))
        if sub_a is not None:
            sel &= np.asarray(sub_a).astype(str) == "road"
        maj = table.majors(n)
        sel &= maj >= 0
        idx = np.flatnonzero(sel)
        if len(idx):
            v = np.asarray(table.value)[maj[idx]].astype(np.float64)
            attrs[idx, 0] = np.where(v < 0, -1.0, 1.0)
            attrs[idx, 1] = [np.nan if counts[i] is None else float(int(counts[i])) for i in idx.tolist()]
            attrs[idx, 2] = np.where(roadlanes.links(f)[idx], 1.0, 0.0)
            for c in np.unique(cls[idx]).tolist():
                if c not in _LANE_FLOW_DFLT:
                    _LANE_FLOW_DFLT[c] = lane_layout(Spec(carriageway(c, "paved", False, None, True, None), "paved",
                                                          False, c, True))[1]
                attrs[idx[cls[idx] == c], 3] = float(_LANE_FLOW_DFLT[c])
    got = f.__dict__["_lane_flow_rows"] = (pts, npts, attrs)
    return got


def _lane_flow_local(feats):
    """lane_flow solved over this cell and the eight around it (mapdata's _bnge_neighbours), per feature of this
    cell: earthfast.lane_flow_cells (each road id once, this cell's first). Remembered on feats."""
    got = feats.__dict__.get("_lane_flow")
    if got is not None:
        return got
    import fast
    fast.require(169, "lane_flow_cells")
    layers = [feats]
    hook = feats.__dict__.get("_bnge_neighbours")
    if hook is not None:
        try:
            layers += [f for f in hook() if f is not None and len(f)]
        except Exception as e:          # (this cell alone)
            log.debug("lane flow: neighbours: %s: %s", type(e).__name__, e)
    rows = [_lane_flow_rows(f) for f in layers]
    ids = [_lane_flow_ids(f) for f in layers]
    got = np.asarray(fast._ef.lane_flow_cells(ids, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                                              float(LANE_FLOW_MAX)))
    out = [None] * len(feats)
    for i in np.flatnonzero(np.isfinite(got) & ~(rows[0][2][:, 2] > 0.5)).tolist():
        out[i] = int(got[i])
    feats.__dict__["_lane_flow"] = out
    return out


def lane_shifts(feats, urban_fn=None, widths=None, left_hand=False):
    """Per feature: its lane-line shifts ((lon, lat, shift m + left of its geometry, length m), ...) at merges and
    splits (SHIFT_*), with the widths of road_specs(feats, urban_fn, widths, left_hand) (its deck's where it has no
    ground spec); () none. Pure; remembered on feats."""
    memo = feats.__dict__.setdefault("_lane_shifts", {})
    k = (urban_fn, widths, bool(left_hand))
    got = memo.get(k)
    if got is not None:
        return got
    import fast
    fast.require(119, "lane_shifts")
    specs = road_specs(feats, urban_fn, widths, left_hand)
    bridge = bridge_specs(feats)
    n = len(feats)
    out = [()] * n
    if n:
        starts = feats.coord_off[feats.ring_off[feats.part_off[:-1]]]
        stops = feats.coord_off[feats.ring_off[feats.part_off[1:]]]
        off = np.zeros(n + 1, np.int64)
        off[1:] = np.cumsum(stops - starts)
        if off[-1]:
            lonlat = np.concatenate([feats.lonlat[a:b] for a, b in zip(starts, stops)])
            attrs = np.zeros((n, 4))
            codes = {}
            rules = feats.rules
            for i in range(n):
                sp = specs[i] or bridge[i]
                if sp is None or not sp.oneway:
                    continue
                v = _rule_value(rules.get("oneway"), i)
                attrs[i, 0] = -1.0 if v is not None and float(v) < 0 else 1.0
                attrs[i, 1] = sp.width
                attrs[i, 2] = 1.0 if _rule_value(rules.get("subclass"), i) == "link" else 0.0
                attrs[i, 3] = codes.setdefault(sp.cls, len(codes))
            rows = np.asarray(fast._ef.lane_shifts(
                np.ascontiguousarray(lonlat, np.float64), off, attrs,
                np.array([SHIFT_M_PER_M, SHIFT_M[0], SHIFT_M[1], SHIFT_SHARE, SHIFT_MIN_M], np.float64)))
            got = {}
            for r in rows:
                got.setdefault(int(r[0]), []).append((float(r[1]), float(r[2]), float(r[3]), float(r[4])))
            for i, t in got.items():
                out[i] = tuple(t)
    memo[k] = out
    return out


def shift_at(shifts, lon, lat):
    """Per point (lon, lat arrays) the lane-line shift (m, + left) of a road with these shifts (lane_shifts), eased
    from each joint's over its length; None where none reaches any point."""
    if not shifts:
        return None
    lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
    out, hit = np.zeros(lon.shape), False
    for slon, slat, shift, span in shifts:
        d = np.hypot((lon - slon) * 111320.0 * math.cos(math.radians(slat)), (lat - slat) * 110540.0)
        near = d < span
        if span > 0 and near.any():
            hit = True
            t = np.clip(d / span, 0.0, 1.0)
            out += np.where(near, shift * (1.0 - t * t * (3.0 - 2.0 * t)), 0.0)
    return out if hit else None


def _spec_json(sp):
    return {f: (v.item() if hasattr(v, "item") else list(v) if isinstance(v, tuple) else v)
            for f, v in sp._asdict().items()}


def _specs_from(data, n):
    """road_specs' list from the shared facts (_spec_json), or None."""
    if not isinstance(data, list) or len(data) != n:
        return None
    try:
        return [None if d is None else Spec(**{**d, "sides": tuple(d["sides"]) if d.get("sides") is not None else None})
                for d in data]
    except (TypeError, KeyError):
        return None


def _osm_sidewalks(feats, specs):
    """specs with sides set where OpenStreetMap's sidewalk tag says which (SIDEWALK_TAG). Pure."""
    tags = feats.attrs.get("osm_sidewalk")
    if tags is None:
        return specs
    out = list(specs)
    for i, sp in enumerate(out):
        if sp is None or sp.kind not in BUILT:
            continue
        tag = str(tags[i]).strip().lower()
        if tag in SIDEWALK_TAG:
            out[i] = sp._replace(sides=SIDEWALK_TAG[tag])
    return out


def unbuilt(feats, i):
    """Is feature i a road not built yet (OpenStreetMap's highway=construction / proposed: osmtags osm_highway, or
    Overture's is_under_construction flag over most of it)? Not drawn, as on the stock map views (GitHub #184). (The
    Gotthard road tunnel's second bore, highway=construction until ~2029, was built beside the real one, its portal in
    the A2's lanes: Layercake leaves construction ways out, and the join took the open tunnel's tags, 2026-10-28.)"""
    return bool(unbuilt_mask(feats)[i])


def unbuilt_mask(feats):
    """unbuilt of every feature of a layer (n,) bool, kept on it."""
    got = feats.__dict__.get("_unbuilt")
    if got is None:
        got = feats.__dict__["_unbuilt"] = _unbuilt_all(feats)
    return got


def _unbuilt_all(feats):
    """unbuilt for every feature of a layer at once (105k calls in two cells' planning, 2026-11-02): each feature's
    construction share summed over its flag rows in their order, as one at a time did."""
    n = len(feats)
    out = np.zeros(n, bool)
    col = feats.attrs.get("osm_highway")
    if col is not None:
        out |= np.isin(np.asarray([str(v) for v in col.tolist()] if hasattr(col, "tolist") else
                                  [str(v) for v in col], dtype=object), ["construction", "proposed"])
    t = feats.rules.get("flags")
    if t is not None and len(t.feature):
        vals = t.value.tolist() if hasattr(t.value, "tolist") else list(t.value)
        uc = np.fromiter(("is_under_construction" in str(v) for v in vals), bool, len(vals))
        if uc.any():
            share = np.zeros(n)
            st = np.asarray(t.start, np.float64)
            en = np.asarray(t.end, np.float64)
            for j in np.flatnonzero(uc).tolist():       # (rows in order: the same sums as before)
                i = int(t.feature[j])
                if 0 <= i < n:
                    a, b = float(st[j]), float(en[j])
                    share[i] += (b - a) if b > a else 1.0      # (no range: the whole feature)
            out |= share >= UNBUILT_SHARE
    return out


def _urban_samples(lonlat):
    """Sample the same road uniformly regardless of source-vertex clustering.

    The endpoint-inclusive spacing is at most URBAN_SAMPLE_M; original vertices
    are interpolated by cumulative distance, never counted as urban-context votes.
    Geometry itself remains untouched for widths, banking, bridges and rendering.
    """
    ll = np.asarray(lonlat, np.float64)
    if len(ll) < 2:
        return ll
    latitude = .5 * (float(ll[:, 1].min()) + float(ll[:, 1].max()))
    scale = np.array([111320. * math.cos(math.radians(latitude)), 110540.])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(ll, axis=0) * scale, axis=1))]
    if distance[-1] < 1e-8:
        return ll[:1]
    count = max(2, int(math.ceil(distance[-1] / URBAN_SAMPLE_M)) + 1)
    stations = np.linspace(0., distance[-1], count)
    return np.column_stack([np.interp(stations, distance, ll[:, axis]) for axis in (0, 1)])


def _road_specs(feats, urban_fn, left_hand):
    out = []
    rules = feats.rules
    ask = []                                    # (index in out, cls, kind, samples, oneway, direction, lanes, surface, link, mapped width)
    developed_both = {}
    regional = set()                            # out indices: untagged roads gravel when rural (_regional_gravel)
    cr = rules.get("connectors")
    ncon = np.bincount(np.asarray(cr.feature), minlength=len(feats)) if cr is not None and len(cr.feature) else None
    lens = _lengths(feats) if ncon is not None else None
    for i in range(len(feats)):
        cls = str(feats.attrs["class"][i]) if "class" in feats.attrs else "unknown"
        if str(feats.attrs.get("subtype", ["road"] * len(feats))[i]) != "road" or cls not in STYLE or unbuilt(feats, i):
            out.append(None)
            continue
        rngs = raw_level_ranges(feats, i)        # (raw: _skew_decks reads these specs)
        if rngs and sum(b - a for a, b, _ in rngs) > 1.0 - MIN_PART:     # entirely off the ground
            out.append(None)
            continue
        width = _rule_value(rules.get("width"), i)
        rule_w = float(width) if width is not None and 1.0 <= float(width) <= 40.0 else None
        if rule_w is None:
            rule_w = _osm(feats, "width", i, *OSM_WIDTH_M)      # OpenStreetMap's width (osmtags.py)
        lanes = _lane_count(feats, i, cls)     # (through merges / splits: lane_flow)
        surf = _surface_value(feats, i)
        kind = surface_kind(cls, surf)
        if _regional_gravel(feats, i, cls, surf):
            regional.add(len(out))          # (gravel unless the town test finds it urban, below)
        ow = _rule_value(rules.get("oneway"), i) is not None
        import roadlanes
        link = roadlanes.is_link(feats, i)
        if rule_w is None and link:
            rule_w = _link_width(lanes or (1 if ow else 2))
        dk = _dir_kmh(feats, i, ncon, lens)
        if dk[1] is None:                       # OpenStreetMap's speed limit where Overture has none
            ms = _osm(feats, "maxspeed", i, 5.0, 200.0)
            if ms is not None:
                dk = (dk[0], ms, dk[2])
        if urban_fn is not None and kind in BUILT:
            a = feats.coord_off[feats.ring_off[feats.part_off[i]]]
            b = feats.coord_off[feats.ring_off[feats.part_off[i + 1]]]
            if b > a:
                ask.append((len(out), cls, kind, _urban_samples(feats.lonlat[a:b]), ow, dk, lanes,
                            surface_token(surf), link, rule_w))
                out.append(None)
                continue
        if len(out) in regional:        # (no land cover to tell town from country: rural)
            kind = "gravel"
        out.append(Spec(carriageway(cls, kind, False, rule_w, ow, lanes, link=link), kind, False, cls, ow, 2.5, *dk,
                        lanes=lanes, surf=surface_token(surf), link=link))
    if ask:
        ll = np.concatenate([samples for _, _, _, samples, *_ in ask])
        left, right = _beside(ll, [len(samples) for _, _, _, samples, *_ in ask], URBAN_BESIDE_M)
        n = len(ll)
        pts = np.concatenate([ll, left, right])
        u_all = np.asarray(urban_fn(pts[:, 1], pts[:, 0]), bool)      # (one call: its cost is per call)
        u, by, both = u_all[:n], u_all[n:2 * n] | u_all[2 * n:], u_all[n:2 * n] & u_all[2 * n:]
        k = 0
        for j, cls, kind, samples, ow, dk, lanes, token, link, rule_w in ask:
            count = len(samples)
            developed_both[j] = float(np.mean(both[k:k + count])) >= URBAN_BOTH_SHARE
            urban = (float(np.mean(u[k:k + count])) >= URBAN_SHARE
                     and float(np.mean(by[k:k + count])) >= URBAN_BESIDE_SHARE
                     and (cls not in ("motorway", "trunk")
                          or float(np.mean(both[k:k + count])) >= URBAN_BOTH_SHARE))
            k += count
            if j in regional and not urban:
                kind = "gravel"
            out[j] = Spec(carriageway(cls, kind, urban, rule_w, ow, lanes, link=link), kind, urban, cls, ow, 2.5, *dk, lanes=lanes,
                          surf=token, link=link)
    out = _swedish_specs(feats, out)
    out = _rings(feats, out)
    out = dual_carriageways(feats, out, left_hand)
    # A primary's opposite carriageway can supply all of its apparent development
    # on one side. Apply the major-highway rule after pairing, without changing
    # ordinary town main streets or explicitly mapped sidewalk/lighting tags.
    for j, sp in enumerate(out):
        if (sp is not None and sp.cls == "primary" and sp.median and sp.urban
                and not developed_both.get(j, True)):
            width = _rule_value(rules.get("width"), j)
            if width is None:
                width = _osm(feats, "width", j, *OSM_WIDTH_M)
            w = carriageway(sp.cls, sp.kind, False, width, sp.oneway, sp.lanes, link=sp.link)
            out[j] = sp._replace(urban=False, width=_dual_width(sp._replace(urban=False), w, sp.median, feats, j))
    import franceroads
    return franceroads.adjust_specs(feats, _mountain_roads(feats, _osm_sidewalks(feats, _rural_connectors(feats, out))))


def switchbacks(lonlat):
    """How many switchbacks a lon/lat polyline (k, 2) has (MOUNTAIN_*)."""
    import fast
    fast.require(42, "tight_bends")
    ll = np.asarray(lonlat, np.float64)
    if len(ll) < 3:
        return 0
    kx = 111320.0 * math.cos(math.radians(float(ll[:, 1].mean())))
    xy = np.ascontiguousarray(np.c_[(ll[:, 0] - ll[0, 0]) * kx, (ll[:, 1] - ll[0, 1]) * 111320.0])
    m = np.asarray(fast._ef.tight_bends(xy, MOUNTAIN_SPAN_M, MOUNTAIN_TURN_DEG), bool)
    return int(np.count_nonzero(m[1:] & ~m[:-1]) + (1 if m[0] else 0))


def hairpin_apexes(lonlat):
    """The (lon, lat) of each hairpin's apex on a lon/lat polyline (k, 2): the middle point of each run of points in a
    MOUNTAIN_TURN_DEG turn within 2 x MOUNTAIN_SPAN_M (earthfast.tight_bends)."""
    ll = np.asarray(lonlat, np.float64)
    if len(ll) < 3:
        return []
    kx = 111320.0 * math.cos(math.radians(float(ll[:, 1].mean())))
    xy = np.c_[(ll[:, 0] - ll[0, 0]) * kx, (ll[:, 1] - ll[0, 1]) * 111320.0]
    return [(float(ll[k, 0]), float(ll[k, 1])) for k in hairpin_apex_rows(xy)]


def hairpin_apex_rows(xy):
    """hairpin_apexes on a polyline xy (k, 2) in metres: the index of each apex's point."""
    import fast
    fast.require(42, "tight_bends")
    xy = np.ascontiguousarray(np.asarray(xy, np.float64)[:, :2])
    if len(xy) < 3:
        return []
    m = np.asarray(fast._ef.tight_bends(xy, MOUNTAIN_SPAN_M, MOUNTAIN_TURN_DEG), bool)
    out, i = [], 0
    while i < len(m):
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(m) and m[j + 1]:
            j += 1
        out.append((i + j) // 2)
        i = j + 1
    return out


def hairpin_tapers(lonlat, width):
    """The tapers (as road_tapers': (lon, lat, width there, length)) of a pass road's hairpins (HAIRPIN_*)."""
    return tuple((lo, la, float(width) + HAIRPIN_WIDEN_M, HAIRPIN_SPAN_M) for lo, la in hairpin_apexes(lonlat))


def _mountain_roads(feats, specs):
    """specs with the mountain pass roads narrowed and marked (MOUNTAIN_*). Pure."""
    if not MOUNTAIN_ON:
        return specs
    rules = feats.rules
    out = list(specs)
    for i, sp in enumerate(specs):
        if sp is None or sp.urban or sp.cls not in MOUNTAIN_CLASSES or sp.kind != "paved" or sp.oneway \
                or sp.median or (sp.lanes or 0) > MOUNTAIN_MAX_LANES:
            continue
        if _rule_value(rules.get("width"), i) is not None or _osm(feats, "width", i, *OSM_WIDTH_M) is not None:
            continue
        a = feats.coord_off[feats.ring_off[feats.part_off[i]]]
        b = feats.coord_off[feats.ring_off[feats.part_off[i] + 1]]
        if b - a < 3 or switchbacks(feats.lonlat[a:b]) < MOUNTAIN_SWITCHBACKS:
            continue
        out[i] = sp._replace(width=min(sp.width, MOUNTAIN_ROAD_M), mountain=True)
    return out


def _swedish_specs(feats, specs, widths_only=False):
    """specs with Sweden's own reading of a road, by its first point: a town street there has no parking lanes in its
    width, and one width in town and out of it: which of the two a road is comes from the land cover beside each map
    way, and flipped the width at 32% of the joins along roads round Boras, up to 4.6 m at once. Mapped widths stand.
    Pure."""
    import sweden
    if not len(feats):
        return specs
    first = feats.lonlat[feats.coord_off[feats.ring_off[feats.part_off[:-1]]]]
    inside = sweden.contains(first[:, 0], first[:, 1])
    if not inside.any():
        return specs
    rules, out = feats.rules, list(specs)
    tags = feats.attrs.get("osm_surface")
    subclass = rules.get("subclass")
    for j in np.flatnonzero(inside):
        sp = out[j]
        if sp is None:
            continue
        mapped = sp.link or (_rule_value(rules.get("width"), j) is not None
                             or _osm(feats, "width", j, *OSM_WIDTH_M) is not None)
        urban, kind, surf = sp.urban and not (sp.kmh or 0.0) >= SE_TOWN_MAX_KMH, sp.kind, sp.surf
        if not widths_only:
            tag = None if tags is None or tags[j] is None else str(tags[j])
            if surf is None and tag not in (None, "", "None", "nan"):       # (Overture's rule has it only for some)
                kind, surf = surface_kind(sp.cls, tag), surface_token(tag)
            elif (surf is None and not urban and sp.cls in SE_GRAVEL_CLASSES and not sp.lanes and not mapped
                  and _rule_value(subclass, j) not in ("parking_aisle",)):
                kind = "gravel"
        w = sp.width
        if not mapped and sp.cls != "motorway" and sp.cls != "track":
            w = carriageway(sp.cls, "gravel" if sp.cls in SE_NO_STRIPS else "paved", False, None, sp.oneway, sp.lanes)
        out[j] = sp._replace(width=w, urban=urban, kind=kind, surf=surf, parking=False)
    return out


def _rural_connectors(feats, specs):
    """Short links across a rural divided highway inherit its roadside context.

    The two carriageways themselves can make a median crossing look built-up.
    Require both ends to meet distinct rural divided roads; an urban connection
    or a road merely running nearby is not evidence. OSM overrides follow this.
    """
    candidates = [i for i, sp in enumerate(specs) if sp is not None and sp.link and sp.urban
                  and sp.cls in ("trunk", "primary", "secondary", "tertiary")]
    rural = [i for i, sp in enumerate(specs) if sp is not None and not sp.urban and sp.oneway
             and sp.median and sp.cls in ("motorway", "trunk", "primary")]
    if not candidates or len(rural) < 2:
        return specs
    ends = {}
    for i in candidates + rural:
        a = feats.coord_off[feats.ring_off[feats.part_off[i]]]
        b = feats.coord_off[feats.ring_off[feats.part_off[i + 1]]]
        if b > a:
            ends[i] = feats.lonlat[[a, b - 1]]
    points = np.concatenate([ends[i] for i in rural if i in ends])
    owners = np.repeat([i for i in rural if i in ends], 2)
    scale = np.array([111320.0 * math.cos(math.radians(float(points[:, 1].mean()))), 110540.0])
    out = list(specs)
    lengths = _lengths(feats)
    for i in candidates:
        if i not in ends or lengths[i] > 2 * URBAN_BESIDE_M:
            continue
        distance = np.linalg.norm((ends[i][:, None] - points[None]) * scale, axis=2)
        attached = [set(owners[distance[j] < 2.5]) for j in (0, 1)]
        if not attached[0] or not attached[1] or len(attached[0] | attached[1]) < 2:
            continue
        sp = out[i]
        width = _rule_value(feats.rules.get("width"), i)
        if width is None:
            width = _osm(feats, "width", i, *OSM_WIDTH_M)
        out[i] = sp._replace(urban=False, width=carriageway(sp.cls, sp.kind, False, width, sp.oneway, sp.lanes, link=sp.link))
    return out


def ring_width(lanes):
    """The circulatory roadway's width for its lane count (None: one)."""
    return RING_LANE_M + RING_EXTRA_LANE_M * (max(1, int(lanes or 1)) - 1)


def is_ring(feats, i):
    """Is feature i a mapped roundabout ring (osmtags osm_junction)?"""
    col = feats.attrs.get("osm_junction")
    return col is not None and str(col[i]) in RING_JUNCTIONS


def _rings(feats, specs):
    """specs with the mapped roundabout rings widened to ring_width (or their mapped width, if wider). Pure."""
    if feats.attrs.get("osm_junction") is None:
        return specs
    out = list(specs)
    rules = feats.rules
    for i, sp in enumerate(out):
        if sp is None or sp.kind not in BUILT or not is_ring(feats, i):
            continue
        width = _rule_value(rules.get("width"), i)
        mapped = float(width) if width is not None and 1.0 <= float(width) <= 40.0 else None
        if mapped is None:
            mapped = _osm(feats, "width", i, *OSM_WIDTH_M)
        out[i] = sp._replace(width=max(mapped or 0.0, ring_width(sp.lanes)), urban=False, median=None)
    return out


def _measured_widths(feats, specs, widths):
    """specs with the widths measured between the buildings (roadwidth.py: widths(id, lonlat) -> (carriageway m,
    sidewalk m) or None) where the map gives none. Pure given widths."""
    rules = feats.rules
    out = list(specs)
    for j, sp in enumerate(out):
        if sp is None or sp.kind not in BUILT or sp.cls not in SIDEWALK_CLASSES or sp.cls in MEASURE_SKIP:
            continue
        if _rule_value(rules.get("width"), j) is not None or not sp.parking:
            continue
        if sp.median:       # one carriageway of a divided road: the width between the buildings is both of them
            continue        # and the median (each took all of it, one wide road over Boston Street's median)
        if is_ring(feats, j):
            continue        # (a roundabout: _rings' circulatory width; the buildings stand round the whole circle)
        a = feats.coord_off[feats.ring_off[feats.part_off[j]]]
        b = feats.coord_off[feats.ring_off[feats.part_off[j + 1]]]
        if _osm(feats, "width", j, *OSM_WIDTH_M) is not None or sp.lanes:
            continue        # (OpenStreetMap's own width, or its lane count's)
        m = widths(str(feats.ids[j]), feats.lonlat[a:b]) if b > a else None
        dflt = carriageway(sp.cls, sp.kind, True, None, sp.oneway, sp.lanes, link=sp.link)
        # Historic streets can be one shared lane between close building fronts. Rejecting these measurements
        # kept a default carriageway wider than the street itself.
        minimum = 3.0 if sp.cls in {"residential", "living_street", "unclassified", "tertiary"} else 0.6 * dflt
        if m is not None and minimum <= m[0] <= MEASURE_MAX * dflt:
            out[j] = sp._replace(width=m[0], walk=m[1])
    return out


def _cell_memo(feats, name, make):
    """make() of a layer, remembered on it and in _FACTS_MEM by its cell (another copy of the layer, after the map
    data's memory cache reloaded it, takes it: 2026-11-02) when made with no fault of this thread."""
    st = feats.__dict__
    got = st.get(name)
    if got is not None:
        return got
    mk = _ranges_mem_key(st)
    mk = None if mk is None else mk[:1] + (name,) + mk[2:]
    if mk is not None:
        with _FACTS_LOCK:
            got = _FACTS_MEM.get(mk)
        if got is not None and len(got) == len(feats):
            st[name] = got
            return got
    import revisit
    f0 = revisit.thread_faults()
    got = st[name] = make()
    if mk is not None and revisit.thread_faults() == f0:
        with _FACTS_LOCK:
            _FACTS_MEM[mk] = got
            while len(_FACTS_MEM) > roadbase.FACTS_MEM_KEPT:
                _FACTS_MEM.popitem(last=False)
    return got


def bridge_specs(feats):
    """Per feature: the Spec of a road bridge (flag is_bridge, or level above ground; not a tunnel), else
    None. No parking widths on bridges (urban False). Pure."""
    return list(_cell_memo(feats, "_bridge_specs", lambda: _marking_specs(feats, _bridge_specs(feats))))  # (cells stay decoded: mapdata)


def tunnel_specs(feats):
    """Per feature: the Spec of a road with a tunnel range, else None (as bridge_specs). Pure."""
    return list(_cell_memo(feats, "_tunnel_specs", lambda: _marking_specs(feats, _bridge_specs(feats, "tunnel"))))


def _bridge_specs(feats, want="bridge"):
    out, free = [], {}              # free: i -> its width its own (one-way or mapped), not a two-way class width
    rules = feats.rules
    ground = road_specs(feats)      # (rural: no parking on a deck; its divided carriageways, dual_carriageways)
    for i in range(len(feats)):
        cls = str(feats.attrs["class"][i]) if "class" in feats.attrs else "unknown"
        if str(feats.attrs.get("subtype", ["road"] * len(feats))[i]) != "road" or cls not in STYLE or unbuilt(feats, i):
            out.append(None)
            continue
        # (the lengthened ranges: a bridge's range carried onto the road joined at its end (_skew_decks) is a deck
        # too; with the raw ranges that stretch was neither deck nor ground, the deck's end lost its approach ramp
        # and sat ~4.5 m low over BR-343 at Teresina, GitHub #306)
        if not any(k == want for _, _, k in (level_ranges(feats, i) if want == "bridge" else raw_level_ranges(feats, i))):
            out.append(None)
            continue
        width = _rule_value(rules.get("width"), i)
        rule_w = float(width) if width is not None and 1.0 <= float(width) <= 40.0 else None
        # (OpenStreetMap's width and lane count, as _road_specs: a 4-lane bridge was drawn at its class's 8 m, one lane
        # each way, Kommunalny Bridge in Perm, GitHub #378 by RiDexSS; tunnels too)
        if rule_w is None:
            rule_w = _osm(feats, "width", i, *OSM_WIDTH_M)
        lanes = _lane_count(feats, i, cls)     # (through merges / splits: lane_flow)
        surf = _surface_value(feats, i)
        kind = surface_kind(cls, surf)
        ow = _rule_value(rules.get("oneway"), i) is not None
        import roadlanes
        link = roadlanes.is_link(feats, i)
        if rule_w is None and link:
            rule_w = _link_width(lanes or (1 if ow else 2))
        ov = _rule_value(rules.get("oneway"), i)
        out.append(Spec(carriageway(cls, kind, False, rule_w, ow, lanes, link=link), kind, False, cls, ow, lanes=lanes,
                        surf=surface_token(surf), link=link, dir=(-1 if float(ov) < 0 else 1) if ov is not None else 0))
        free[i] = ow or rule_w is not None
    out = _swedish_specs(feats, out, widths_only=True)      # (first: a divided road's deck is capped from its width)
    for i, sp in enumerate(out):
        g = ground[i] if ground is not None and sp is not None else None
        if g is not None and g.median:
            # one carriageway of a divided road (dual_carriageways on its ground specs): its deck one carriageway wide,
            # as its approaches; it was the class's two-way width (Baltimore's motorways: 18.8 m decks on 12 m roads,
            # the user 2026-10-31: "bridge widths dont always match the road widths")
            w = sp.width if free[i] else carriageway(sp.cls, sp.kind, False, None, True, sp.lanes, link=sp.link)
            out[i] = sp._replace(oneway=True, dir=g.dir, median=g.median, width=dual_cap(g.median, w))
        elif (g is not None and sp is not None and g.oneway and sp.oneway and not sp.link
              and sp.cls in ("motorway", "trunk") and g.width < sp.width):
            # a one-way motorway carriageway with no median read (a nameless one, capped against the other only where
            # they overlap, or narrowed by the country's rules: franceroads): its deck as wide as its road
            out[i] = sp._replace(width=g.width)
    return out


def _marking_specs(feats, specs):
    """Attach paint geography without changing lane, shoulder or structural facts."""
    states = feats.__dict__.get('_marking_states')
    if states is None:
        import signstates
        starts = feats.coord_off[feats.ring_off[feats.part_off[:-1]]]
        ends = feats.coord_off[feats.ring_off[feats.part_off[1:]]]
        states = np.full(len(specs), '', dtype='U2')
        valid = ends > starts
        mids = ((starts[valid] + ends[valid] - 1) // 2).astype(int)
        ll = feats.lonlat[mids] if len(mids) else np.empty((0, 2))
        states[valid] = signstates.states(ll[:, 1], ll[:, 0])
        feats._marking_states = states
    return [sp._replace(state=str(states[i])) if sp else None for i, sp in enumerate(specs)]


def lane_layout(spec):
    """(driving width m, lanes) of a road as its lane lines are painted (extras) and the AI drives it (navgraph): the
    carriageway less a town street's parking lanes or a rural road's edge strips (counted as lanes, every rural
    Interstate had three, GitHub #43), its lanes OpenStreetMap's count or as many LANE_M as fit (at least 1). Pure."""
    w, cls = spec.width, spec.cls
    import franceroads
    if franceroads.enabled(spec) and not spec.urban:
        drive = max(2.5, w - 2 * min(.15, edge_margin(spec)))
        lanes = int(spec.lanes) if spec.lanes else max(1 if spec.oneway else 2, int(drive // LANE_M))
        return drive, lanes
    parked = spec.urban and spec.parking and cls in PARKING_CLASSES and not (not spec.oneway and cls in URBAN_ARTERIAL)
    drive = w - ((PARKING_M if spec.oneway else 2 * PARKING_M) if parked else 2 * edge_margin(spec))
    lanes = int(spec.lanes) if spec.lanes else int(drive // LANE_M)
    import markings
    if lanes > 1 and w / lanes < markings.MIN_LANE_M:   # (a count the road has no room for: a tag, or a way of a stretch)
        lanes = int(w // markings.MIN_LANE_M)
    return drive, max(1, lanes)


def lane_tags(feats, i):
    """OpenStreetMap's lane tags of feature i ({key: value}, railtags.LANE_COLUMNS joined by MapData), or None."""
    import railtags
    out = {}
    for k in railtags.LANE_COLUMNS:
        if k in ("junction", "lanes"):
            continue                            # (read by roads._rings / roadlanes.generated_counts)
        col = feats.attrs.get(railtags.lane_attr(k))
        if col is not None and col[i] not in (None, ""):
            out[k] = col[i]
    import franceroads
    return franceroads.observed_lane_tags(feats, i, out)


def lane_tag_parts(feats, i):
    """Feature i's stretches of OpenStreetMap's lane tags: [(start, end, way id or None, tags or None)] fractions along
    it, one stretch whole unless it was made from several ways (Overture merges the ways between two junctions; each has
    its own lane tags, a 2+1 road's alternate along one segment). Tags hold the way's own "lanes" where the way has lane
    tags; lane_tags(feats, i) is the way covering most of the feature."""
    import railtags
    parts = feats.attrs.get(railtags.LANE_PARTS)
    if parts is not None and parts[i]:
        return [(a, b, w, t or None) for a, b, w, t in parts[i]]
    rec = feats.attrs.get("src_record")
    return [(0.0, 1.0, railtags.way_id(str(rec[i])) if rec is not None else None, lane_tags(feats, i))]
