"""Road builder: sidewalks and road markings (split out of roads.py): which roads get sidewalks and on which side, the
marking lines (extras), sidewalk runs and their draping.
"""
import math
import numpy as np
from roadbase import (
    ARTERIAL_GAP_M,
    ARTERIAL_KMH,
    BUILT,
    CENTRE,
    CENTRE_CLASSES,
    EDGE_DECAL,
    EDGE_DECAL_AT,
    EDGE_LINE,
    EDGE_LINE_CLASSES,
    GUTTER,
    JOIN_GAP_M,
    JOIN_TURN_DEG,
    LANE_CLASSES,
    LANE_LINE,
    LANE_M,
    MEDIAN_CURB_W,
    MEDIAN_MAX_M,
    MEDIAN_MIN_M,
    MEDIAN_SLAB_M,
    NO_WALK_KMH,
    PARKING_CLASSES,
    PARKING_M,
    SIDEWALK,
    SIDEWALK_CLASSES,
    SIMPLIFY_M,
    URBAN_ARTERIAL,
    WALK_DRAPE_M,
    WALK_DRAPE_TOL_M,
    WALK_GROUND_M,
    WALK_ON_ROAD_M,
    WALK_TRIM_KEEP_M,
    WALK_TRIM_STEP_M,
    densify,
)
from roadspecs import (
    driving_offset,
    edge_margin,
    highway_shoulders,
    lane_layout,
    lane_tag_parts,
    level_ranges,
)
from roadgeom import (
    _length,
    centre_colour,
    offset_polyline,
)


def has_sidewalks(spec):
    # a two-way urban trunk is a town's main street on a state highway (Bishop's US-395, Moab's US-191);
    # divided trunks are one-way per carriageway: expressways get none, slow ones (Ocean City's Coastal
    # Highway, 35 mph, 2026-10-14 street photos) one on their outer side (walk_sides)
    if getattr(spec, "sides", None) is not None:      # the map says (_osm_sidewalks)
        return spec.kind in BUILT and bool(spec.sides)
    if spec.oneway and spec.kmh is not None and spec.kmh >= NO_WALK_KMH:
        return False
    return spec.kind in BUILT and spec.urban and not limited_access(spec) and (
        spec.cls in SIDEWALK_CLASSES or (spec.cls == "trunk" and (not spec.oneway or _arterial(spec))))


def limited_access(spec):
    """A road no pedestrian walks beside whatever the land cover says: a motorway's or trunk's ramp (a ramp has few
    junctions, so _arterial's spacing rule read it as a town street: a cloverleaf's ramps had a sidewalk each), or a
    one-way carriageway of a primary / trunk with a mapped limit over ARTERIAL_KMH. An unmapped limit leaves the
    class rule. Pure."""
    if spec.link and spec.cls in ("motorway", "trunk"):
        return True
    return bool(spec.oneway and spec.cls in ("primary", "trunk") and spec.kmh is not None and spec.kmh > ARTERIAL_KMH)


def _arterial(spec):
    if not spec.oneway or limited_access(spec):
        return False
    if spec.kmh is not None:
        return spec.kmh <= ARTERIAL_KMH
    return spec.gap is not None and spec.gap <= ARTERIAL_GAP_M


def walk_sides(spec, left_hand=False):
    """The sides (+1 left of the geometry, -1 right) of a road with sidewalks that get one: both, or a
    divided trunk carriageway's outer side only (the curb on the driver's right, left where they drive on
    the left; the other side is the median). Where OpenStreetMap's sidewalk tag says which: those (_osm_sidewalks)."""
    if getattr(spec, "sides", None):
        return tuple(spec.sides)
    # (a divided road's median: its curbed band; carriageways closer than MEDIAN_MIN_M of ground: the inner one would
    # lie on the other carriageway, two pale strips down the middle of a French divided bridge, 2026-11-10)
    narrow = bool(spec.median) and median_width(spec) < MEDIAN_MIN_M
    if (spec.cls == "trunk" or has_median(spec) or narrow) and spec.oneway:
        right = -1 if spec.dir >= 0 else 1
        return (-right,) if left_hand else (right,)
    return (1, -1)


def has_median(spec):
    """A divided road's carriageway with a median of MEDIAN_MIN_M..MEDIAN_MAX_M of ground beside it."""
    return bool(spec.median) and spec.oneway and MEDIAN_MIN_M <= median_width(spec) <= MEDIAN_MAX_M


def median_side(spec, left_hand=False):
    """The side (+1 left / -1 right of the geometry) of a divided road's carriageway that faces its median, or None."""
    if not has_median(spec):
        return None
    return -walk_sides(spec, left_hand)[0]


def median_width(spec):
    """m of ground between a divided road's two carriageways (its centre lines' distance less both halves; the
    other's width taken as this one's), or 0."""
    return max(0.0, float(spec.median or 0.0) - spec.width) if spec.median else 0.0


def sidewalk_stop(spec, fid, other, ofid):
    """How far from the centreline of a crossing road (other) the sidewalks of road spec stop. The corner
    square is one sidewalk's: when both roads have sidewalks the wider road's (then the smaller id) runs on
    to the other's curb and the other's stops against its outer edge; with none on the other road, they
    run on to its curb. No gap and no overlap (two slabs at one height would z-fight). Pure."""
    if has_sidewalks(other) and (-other.width, ofid) < (-spec.width, fid):
        return other.width / 2 + other.walk
    return other.width / 2


def _motesfri_section(spec, osm):
    import motesfri
    return motesfri.is_section(spec, osm)


def way_lines(ex, feats, i, spec, pts, lon, left_hand, country, ws=None, shift=None):
    """extras' lines ex of feature i, its centre and lane lines drawn way by way: a 2+1 road's lanes alternate along one
    segment (lane_tag_parts). pts is the whole feature on the ground, so not where a bridge cuts it (they stay as ex);
    ws, shift: extras', per point of pts (each way's piece takes its own stretch of them)."""
    parts = lane_tag_parts(feats, i)
    if len(parts) < 2 or level_ranges(feats, i):
        return ex
    import markings
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts[:, :2], axis=0).T))]
    t = s / max(s[-1], 1e-9)
    at = lambda f: np.array([[np.interp(f, t, pts[:, c]) for c in range(3)]])       # noqa: E731
    out = [e for e in ex if e[0].get("kind") not in ("centre", "lane")]
    cut = lambda v, a, b: None if v is None else np.r_[np.interp(a, t, v), np.asarray(v, np.float64)[(t > a) & (t < b)],
                                                       np.interp(b, t, v)]                               # noqa: E731
    for a, b, _, tags in parts:
        piece = np.vstack([at(a), pts[(t > a) & (t < b)], at(b)])
        way_spec = spec._replace(lanes=markings._count((tags or {}).get("lanes")) or spec.lanes)
        out += [e for e in extras(way_spec, piece, lon, left_hand, country, tags, cut(ws, a, b),
                                 cut(shift, a, b))
                if e[0].get("kind") in ("centre", "lane")]
    return out


def edge_inset(spec, side=None):
    """Distance to paint from the asphalt edge, shared by ordinary and merge lines."""
    return max(EDGE_LINE['inset'], edge_margin(spec, side)) if highway_shoulders(spec) else EDGE_LINE['inset']


def edge_colour(spec, side):
    return 'line_yellow' if spec.oneway and side == (spec.dir or 1) else 'line_white'


def extras(spec, pts, lon, left_hand=False, country=None, osm=None, ws=None, shift=None, defer_dashes=False):
    """The sidewalk / gutter / marking polylines of one road: [(look dict with m / tl / p, width, pts)].
    pts: its centreline (k, 3) in the anchor frame (a whole ground part: its broken lines' dashes start their cycle at
    its start, so they meet across tiles); lon: its longitude; country: ISO code (markings.style: the country's line
    colours, widths and dash patterns; by longitude when unknown); osm: OpenStreetMap's lane tags of the road
    (railtags.LANE_COLUMNS: the lines lane by lane, markings.lane_plan); ws: its width at each point of pts where
    it tapers (taper_widths: its edges, curbs and lane lines follow); shift: its lines' shift (m, + left) at
    each point at a merge or split (shift_at: its markings only). Pure."""
    import markings
    out = []
    st = markings.style(country, lon)
    import roadmarkings
    profile = roadmarkings.for_road(spec, country, lon)
    w, kind, urban, cls = spec[:4]
    wv = w if ws is None else np.asarray(ws, np.float64)     # (offsets: tapered; decisions: its width)
    if kind not in BUILT:
        return out
    if has_sidewalks(spec):
        for side in walk_sides(spec, left_hand):
            out.append(({**SIDEWALK, "side": side}, spec.walk,
                        offset_polyline(pts, side * (wv / 2 + spec.walk / 2))))
            out.append(({**GUTTER, "side": side}, GUTTER["w"], offset_polyline(pts, side * (wv / 2 - GUTTER["w"] / 2))))
        ms = median_side(spec, left_hand)
        if ms is not None:
            # the median: a curbed band, lawn beyond it (two sidewalks back to back made a concrete strip, the user's
            # report #48 on Boston Street, 2026-10-17: "Improve how medians are made")
            mw = median_width(spec)
            # (each carriageway's half, a hair short of the middle: touching the other's carriageway where the gap
            # narrows, the band was cut away)
            band = max(0.3, mw / 2 - 0.15) if mw <= MEDIAN_SLAB_M else MEDIAN_CURB_W
            out.append(({**SIDEWALK, "side": ms, "median": True}, band, offset_polyline(pts, ms * (wv / 2 + band / 2))))
            if kind == "paved":     # a double yellow line along it (Boston Street, Mapillary 2026-10-17) ...
                look = {"m": st.centre, "tl": CENTRE["tl"], "p": CENTRE["p"]}
                # ... where centre lines are white one solid white edge line along the median's curb (PR #146)
                for k in ((0,) if st.centre == "line_white" else (0, 1)):
                    out.append((look, st.w, offset_polyline(
                        pts, ms * (wv / 2 - EDGE_LINE["inset"] - k * (st.w + st.double_gap)))))
            out.append(({**GUTTER, "side": ms}, GUTTER["w"], offset_polyline(pts, ms * (wv / 2 - GUTTER["w"] / 2))))
    base = pts                              # (the asphalt's own line: its edge decal)
    if shift is not None:                   # (the markings from here on: shifted at a merge / split)
        pts = offset_polyline(pts, shift)
    motorway = cls == "motorway"
    lw_ = st.w_motorway if motorway else (st.w_urban or st.w) if urban else st.w           # a line's width
    sweden = st is markings.STYLES["SE"]
    barrier = sweden and kind == "paved" and _motesfri_section(spec, osm)   # (a 2+1 road's: motesfri.py)

    def mark(kind_, mat, width, along, pattern=None, side=None):
        """A solid line, or a broken one's dashes (pattern: dash, gap); kind_: "centre", "lane" or "edge"; side: an edge
        line's (+1 left / -1 right of the geometry: build_tile cuts it at a rounded corner, roadcorners.tile)."""
        if profile is not None:
            if pattern is None:
                width = profile.solid
            else:
                dimensions = profile.ramp if spec.link else profile.travel
                if dimensions is None:
                    return
                pattern, width = dimensions[:2], dimensions[2]
        if pattern is None:
            look = {"m": mat, "tl": EDGE_LINE["tl"], "p": CENTRE["p"], "kind": kind_}
            if side is not None:
                look["side"] = side
            out.append((look, width, along))
            return
        if defer_dashes and profile is not None:
            out.append((dict(m=mat, tl=max(1., pattern[0]), p=CENTRE["p"], kind=kind_,
                             dash=pattern, _source=along), width, along))
            return
        for piece in markings.dashes(along, pattern):
            look = {"m": mat, "tl": max(1.0, pattern[0]), "p": CENTRE["p"], "kind": kind_, "dash": True}
            if side is not None:
                look["side"] = side
            out.append((look, width, piece))
    import franceroads
    if franceroads.enabled(spec):
        out = [({**look, "m": franceroads.PAINT_MATERIAL}, width, line)
               if look["m"] == "line_white" else (look, width, line) for look, width, line in out
               if look["m"] != GUTTER["m"]]         # (no rust-stained American gutter along French kerbs)
        out.extend(franceroads.paint(spec, pts, osm))
        if kind == "paved" and not has_sidewalks(spec) and spec.cls != "service":     # (a yard's lane has no gravel edge)
            dw = EDGE_DECAL["w"]
            for side in (1, -1):
                line = offset_polyline(pts, side * (w / 2 + (EDGE_DECAL_AT - .5) * dw))
                out.append(({**EDGE_DECAL, "side": side}, dw, line if side > 0 else line[::-1].copy()))
        return out
    # the driving width (a town street's parking lanes and a rural road's edge strips left out: three lanes on every
    # rural Interstate, GitHub #43) and its lanes (OpenStreetMap's count where known)
    parked = urban and spec.parking and cls in PARKING_CLASSES and not (not spec.oneway and cls in URBAN_ARTERIAL)
    drive, lanes = lane_layout(spec)
    # (tapered: the lanes' lines spread with the driving width)
    scale = 1.0 if ws is None else np.maximum(drive + (wv - w), 0.5) / max(drive, 0.5)
    if not spec.oneway and spec.lanes and lanes >= 3 and lanes % 2:
        osm = {**(osm or {}), "lanes": lanes}    # keep odd mapped totals in the paint as well as the AI
    plan = markings.lane_plan(osm, not spec.oneway, drive, lanes, left_hand) \
        if kind == "paved" and (cls in CENTRE_CLASSES or cls in LANE_CLASSES) else None
    if plan is not None and not spec.oneway and w - (2 * PARKING_M if parked else 0.0) < st.min_centre_m:
        plan = None                     # (the map's lanes do not widen it: as the count's, too narrow a road has none)
    if plan is not None:
        # the map's lanes: each line where its lanes meet (OpenStreetMap lanes:forward / backward / both_ways,
        # overtaking, change:lanes; research §5)
        pat = st.lane_motorway if motorway else (st.lane_urban if urban else st.lane_rural)
        cpat = st.centre_urban if urban else st.centre_rural
        d = (st.double_gap + lw_) / 2

        def at(off):
            return offset_polyline(pts, driving_offset(spec) + off * scale)
        for off, role, how in plan:
            if role == "centre" and barrier:
                for sd in (1, -1):
                    mark("centre", st.centre, markings.BARRIER_LINE_W, at(off + sd * markings.BARRIER_LINE_M))
            elif role == "lane":
                mark("lane", "line_white", lw_, at(off), None if how else pat)
            elif role == "twltl":           # solid on the through side, broken on the turn lane's (MUTCD 3B.05)
                mark("centre", st.centre, lw_, at(off + how * d))
                mark("centre", st.centre, lw_, at(off - how * d), cpat)
            elif how == "double" or (sweden and lanes >= 4 and how is None):     # (TSFS 2010:171 2 kap. 17 §)
                if st.warning and w < markings.WARNING_BELOW_M:
                    mark("centre", st.centre, lw_, at(off), st.warning)
                elif st.centre == "line_yellow" or sweden or st is markings.STYLES.get("GB"):
                    for sd in (1, -1):
                        mark("centre", st.centre, lw_, at(off + sd * d))
                else:
                    mark("centre", st.centre, lw_, at(off))
            elif how in ("solid_right", "solid_left"):
                sd = -1 if how == "solid_right" else 1
                mark("centre", st.centre, lw_, at(off + sd * d))
                mark("centre", st.centre, lw_, at(off - sd * d), cpat)
            else:
                mark("centre", st.centre, lw_, at(off), cpat)
    if plan is None and kind == "paved" and cls in CENTRE_CLASSES \
            and w - (2 * PARKING_M if parked else 0.0) >= st.min_centre_m \
            and not spec.oneway and not getattr(spec, "mountain", False):     # (a pass road: none, _mountain_roads)   # no paint on cobbles (the width between the parking lanes, edge strips included)
        # the centre line (MUTCD 3B.01-02, TSRGD / RMS / IISR centre lines): too narrow a road has none; four lanes or
        # more get a solid line (the Americas: double solid yellow), and in the US a town's street a double solid one;
        # otherwise one broken line in the country's colour and town / country pattern
        solid = lanes >= 4 or (st.double_centre_urban and urban)
        if solid and (st.centre == "line_yellow" or sweden):
            for side in (1, -1):
                mark("centre", st.centre, lw_, offset_polyline(pts, side * (st.double_gap + lw_) / 2))
        elif solid:
            mark("centre", st.centre, lw_, pts.copy())
        else:
            mark("centre", st.centre, lw_, pts.copy(), st.centre_urban if urban else st.centre_rural)
    if plan is None and kind == "paved" and cls in LANE_CLASSES:
        # lane lines between the lanes of one direction, broken in the country's pattern
        pat = st.lane_motorway if motorway else (st.lane_urban if urban else st.lane_rural)
        if spec.oneway and lanes >= 2:
            lw = drive / lanes
            for k in range(1, lanes):
                mark("lane", "line_white", lw_, offset_polyline(pts, driving_offset(spec) + (-drive / 2 + k * lw) * scale), pat)
        elif not spec.oneway and lanes >= 4:
            n = lanes // 2
            lw = drive / 2 / n
            for side in (1, -1):
                for k in range(1, n):
                    mark("lane", "line_white", lw_, offset_polyline(pts, side * k * lw * scale), pat)
    if kind == "paved" and not has_sidewalks(spec):          # no curb: the asphalt's edge
        dw = EDGE_DECAL["w"]
        for side in (1, -1):
            line = offset_polyline(base, side * (wv / 2 + (EDGE_DECAL_AT - 0.5) * dw))
            # (u runs from the ribbon's left to its right: the road is on the right of the left side's, so the right
            # side's runs backwards)
            out.append(({**EDGE_DECAL, "side": side}, dw, line if side > 0 else line[::-1].copy()))
    if kind == "paved" and (not urban or not has_sidewalks(spec)) and cls in (st.edge_classes or EDGE_LINE_CLASSES):
        # no curb: edge lines
        # the right edge white; a one-way carriageway's left edge yellow where the country says so (MUTCD 3B.09:
        # divided roads, ramps); wider on motorways (TSRGD 1012, RMS: 0.20-0.30 m)
        # Sweden: broken except on motorways, a dual carriageway and a 2+1 road
        wide = motorway or (st.edge_pattern is not None and spec.oneway and not urban)
        ew = markings.MOTESFRI_EDGE_W if barrier else st.edge_motorway if wide else (st.w_urban or st.edge) if urban \
            else st.edge
        epat = None if wide or barrier else st.edge_pattern
        travel_left = 1 if getattr(spec, "dir", 0) >= 0 else -1      # (dir < 0: one way against the line)
        # (VGU 2024 table 12.5, mötesfri väg: left 0.30 m, right 0.20 m: a 2+1 road's carriageway, by its one or two lanes)
        pair = sweden and spec.oneway and not motorway and not urban and (spec.lanes or 3) <= 2
        for side in (1, -1):
            inset = max(EDGE_LINE["inset"], edge_margin(spec, side)) if highway_shoulders(spec) else EDGE_LINE["inset"]
            mat = st.left_edge if (st.left_edge and spec.oneway and side == travel_left) else "line_white"
            mark("edge", mat, markings.MOTESFRI_EDGE_W if pair and side != travel_left else ew,
                 offset_polyline(pts, side * (wv / 2 - inset)), epat, side=side)
    return out
    if has_sidewalks(spec):
        for side in walk_sides(spec, left_hand):
            out.append(({**SIDEWALK, "side": side}, spec.walk,
                        offset_polyline(pts, side * (w / 2 + spec.walk / 2))))
            out.append(({**GUTTER, "side": side}, GUTTER["w"], offset_polyline(pts, side * (w / 2 - GUTTER["w"] / 2))))
        ms = median_side(spec, left_hand)
        if ms is not None:
            # the median: a curbed band, lawn beyond it (two sidewalks back to back made a concrete strip, the user's
            # report #48 on Boston Street, 2026-10-17: "Improve how medians are made")
            mw = median_width(spec)
            # (each carriageway's half, a hair short of the middle: touching the other's carriageway where the gap
            # narrows, the band was cut away)
            band = max(0.3, mw / 2 - 0.15) if mw <= MEDIAN_SLAB_M else MEDIAN_CURB_W
            out.append(({**SIDEWALK, "side": ms, "median": True}, band, offset_polyline(pts, ms * (w / 2 + band / 2))))
            if kind == "paved":     # a double yellow line along it (Boston Street, Mapillary 2026-10-17) ...
                look = {"m": centre_colour(lon), "tl": CENTRE["tl"], "p": CENTRE["p"]}
                # ... outside the Americas one solid white edge line along the median's curb (PR #146)
                for k in ((0,) if centre_colour(lon) == "line_white" else (0, 1)):
                    out.append((look, CENTRE["w"], offset_polyline(
                        pts, ms * (w / 2 - EDGE_LINE["inset"] - 2 * k * CENTRE["gap"]))))
            out.append(({**GUTTER, "side": ms}, GUTTER["w"], offset_polyline(pts, ms * (w / 2 - GUTTER["w"] / 2))))
    if kind == "paved" and cls in CENTRE_CLASSES and w >= 6.0 and not spec.oneway:   # no paint on cobbles
        if centre_colour(lon) == "line_white":
            # outside the Americas one broken white line (the Gotthard road on Mapillary, 2026-10-18: ours had the
            # Americas' double line in white)
            out.append((LANE_LINE, LANE_LINE["w"], pts.copy()))
        else:
            look = {"m": centre_colour(lon), "tl": CENTRE["tl"], "p": CENTRE["p"]}
            for side in (1, -1):
                out.append((look, CENTRE["w"], offset_polyline(pts, side * CENTRE["gap"])))
    if kind == "paved" and cls in LANE_CLASSES:
        # lane lines (white dashes) where the driving width holds two or more lanes a direction
        parked = urban and cls in PARKING_CLASSES and not (not spec.oneway and cls in URBAN_ARTERIAL)
        # (a rural road's edge strips counted as lanes: three on every rural Interstate, GitHub #43)
        drive = w - ((PARKING_M if spec.oneway else 2 * PARKING_M) if parked else 2 * edge_margin(spec))
        lanes = spec.lanes if spec.lanes else int(drive // LANE_M)      # (OpenStreetMap's count where known)
        if spec.oneway and lanes >= 2:
            lw = drive / lanes
            for k in range(1, lanes):
                out.append((LANE_LINE, LANE_LINE["w"], offset_polyline(pts, driving_offset(spec) - drive / 2 + k * lw)))
        elif not spec.oneway and lanes >= 4:
            n = lanes // 2
            lw = drive / 2 / n
            for side in (1, -1):
                for k in range(1, n):
                    out.append((LANE_LINE, LANE_LINE["w"], offset_polyline(pts, side * k * lw)))
    if kind == "paved" and not has_sidewalks(spec):          # no curb: the asphalt's edge
        dw = EDGE_DECAL["w"]
        for side in (1, -1):
            line = offset_polyline(pts, side * (w / 2 + (EDGE_DECAL_AT - 0.5) * dw))
            # (u runs from the ribbon's left to its right: the road is on the right of the left side's, so the right
            # side's runs backwards)
            out.append(({**EDGE_DECAL, "side": side}, dw, line if side > 0 else line[::-1].copy()))
    if kind == "paved" and (not urban or not has_sidewalks(spec)) and cls in EDGE_LINE_CLASSES:   # no curb: edge lines
        for side in (1, -1):
            inset = max(EDGE_LINE["inset"], edge_margin(spec, side)) if highway_shoulders(spec) else EDGE_LINE["inset"]
            out.append((EDGE_LINE, EDGE_LINE["w"], offset_polyline(pts, side * (w / 2 - inset))))
    return out


def _end_dir(pts, at_end):
    """Unit direction a run leaves by at its start (at_end False: pointing out of the run backwards) or end."""
    if at_end:
        d = pts[-1, :2] - pts[-2, :2]
    else:
        d = pts[0, :2] - pts[1, :2]
    n = math.hypot(*d)
    return d / n if n > 1e-9 else d


def join_walk_ends(runs, carriageways):
    """Sidewalk runs whose open ends face each other across a short gap merged into one run (the user,
    2026-10-14: sidewalks at sharp corners do not join; junctions.py makes no curb return under 35 degrees or with a
    long tangent, so each street's sidewalk stopped at the other's carriageway with bare ground between). runs: the
    runs near the tile before clipping ({pts, side, w, fid, j, open, ramp_fids}); two are merged when the straight
    gap between their ends stays off every carriageway and they are on the same side (one continuous polyline: the
    mesh mitres it, no seam). Pairs found greedily, nearest first. Returns the new list of runs."""
    ends = []                                   # (run index, at_end, point, leaving direction)
    for k, rn in enumerate(runs):
        p = rn["pts"]
        if len(p) < 2:
            continue
        for at_end in (False, True):
            if not rn["open"][1 if at_end else 0] or rn["j"][1 if at_end else 0]:
                continue                        # cut by the near box, or at a curb return: not a loose end
            ends.append((k, at_end, (p[-1] if at_end else p[0]), _end_dir(p, at_end)))
    pairs = []
    cos_t = math.cos(math.radians(JOIN_TURN_DEG))
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            ki, ei, qi, di = ends[a]
            kj, ej, qj, dj = ends[b]
            if ki == kj:
                continue
            gap = qj[:2] - qi[:2]
            L = math.hypot(*gap)
            if L < 0.3 or L > JOIN_GAP_M:
                continue
            u = gap / L
            if float(u @ di) < cos_t or float(-u @ dj) < cos_t:
                continue                        # each end must leave toward the other
            pairs.append((L, a, b))
    runs = [dict(r) for r in runs]
    alias = list(range(len(runs)))              # a merged run's index (runs consumed by merges point to it)
    used = set()
    for L, a, b in sorted(pairs):
        if a in used or b in used:
            continue
        ki, ei, qi, di = ends[a]
        kj, ej, qj, dj = ends[b]
        while alias[ki] != ki:
            ki = alias[ki]
        while alias[kj] != kj:
            kj = alias[kj]
        if ki == kj:
            continue
        ri, rj = runs[ki], runs[kj]
        # the pieces in travel order: ri toward its joined end, then rj away from its joined end
        pi_ = ri["pts"] if ei else ri["pts"][::-1]
        pj_ = rj["pts"] if not ej else rj["pts"][::-1]
        si = ri["side"] if ei else -ri["side"]
        sj = rj["side"] if not ej else -rj["side"]
        if si != sj:
            continue                            # opposite sides of the streets: not one sidewalk
        gap_pts = np.linspace(pi_[-1], pj_[0], max(2, int(L / 2.0) + 2))[1:-1]
        hw = 0.25 * (ri["w"] + rj["w"])
        if any(_seg_dist_min(cl, q[:2]) < h + hw * 0.5 for cl, h in carriageways for q in gap_pts):
            continue                            # the gap would cross a road
        used.update((a, b))
        pts = np.vstack([pi_, gap_pts, pj_]) if len(gap_pts) else np.vstack([pi_, pj_])
        j0 = ri["j"][0] if ei else ri["j"][1]
        j1 = rj["j"][1] if not ej else rj["j"][0]
        o0 = ri["open"][0] if ei else ri["open"][1]
        o1 = rj["open"][1] if not ej else rj["open"][0]
        runs[ki] = {"pts": densify(simplify(pts)), "side": si, "w": ri["w"], "fid": ri["fid"], "j": (j0, j1),
                    "open": (o0, o1), "ramp_keys": sorted(set(ri.get("ramp_keys", [(ri["fid"], ri["side"])])
                                                               + rj.get("ramp_keys", [(rj["fid"], rj["side"])])))}
        runs[kj] = None
        alias[kj] = ki
    return [r for r in runs if r is not None]


def _left_normal(two):
    d = two[1] - two[0]
    n = math.hypot(*d)
    return np.array([-d[1], d[0]]) / n if n > 1e-9 else np.array([0.0, 0.0])


def _seg_dist_min(line, p):
    import fast
    fast.require(0, "seg_dist_min")
    return fast._ef.seg_dist_min(np.asarray(line, np.float64), float(p[0]), float(p[1]))


def clip_polyline(pts, x0, y0, x1, y1):
    """The pieces of a polyline (k, D >= 2, xy first) inside the box, as a list of arrays (Liang-Barsky
    per segment, consecutive pieces joined; earthfast.clip_polyline). Pure."""
    import fast
    fast.require(74, "clip_polyline")
    P = np.ascontiguousarray(np.asarray(pts, np.float64))
    if P.ndim != 2 or len(P) < 2:
        return []
    out, starts = fast._ef.clip_polyline(P, float(x0), float(y0), float(x1), float(y1))
    out, starts = np.asarray(out), np.asarray(starts)
    return [out[starts[i]:starts[i + 1]].copy() for i in range(len(starts) - 1)]


def simplify(pts, tol=SIMPLIFY_M, ztol=None):
    """Ramer-Douglas-Peucker on the xy of a polyline (k, D). ztol (D >= 3): a point is also kept where its height is
    more than this off the line between the kept ones (a deck drawn at its own heights: on xy alone a straight bridge
    kept its two ends in a tile and lost its rise between, 2.5 m below the abutment it met; a car drove into the
    ground, Baltimore's Lombard Street 2026-10-20). Pure."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        a, b = pts[i, :2], pts[j, :2]
        ab = b - a
        L = math.hypot(ab[0], ab[1])
        seg = pts[i + 1:j, :2]
        if len(seg) == 0:
            continue
        if L < 1e-9:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        if ztol is not None and pts.shape[1] >= 3:
            t = (np.clip(((seg[:, 0] - a[0]) * ab[0] + (seg[:, 1] - a[1]) * ab[1]) / (L * L), 0.0, 1.0)
                 if L >= 1e-9 else np.zeros(len(seg)))
            dz = np.abs(pts[i + 1:j, 2] - (pts[i, 2] + t * (pts[j, 2] - pts[i, 2])))
            d = np.maximum(d, dz * (tol / ztol))
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep]


def walks_off_roads(runs, carriageways):
    """runs with every stretch whose slab lies on a carriageway ([(line (k, 2), half width)]) removed. Their centre lines
    were cut at the roads, but a slab is 2.5 m wide and its points ~9 m apart: at a split section's short one-way pieces
    slabs stood across the road. A median's band lies between carriageways: left alone. Runs off every road are returned
    unchanged. Pure."""
    import sidewalks
    a, b, h = [], [], []
    for line, half in carriageways:
        p = np.asarray(line, np.float64)[:, :2]
        if len(p) >= 2 and np.ndim(half) == 0:
            a.append(p[:-1])
            b.append(p[1:])
            h.append(np.full(len(p) - 1, float(half)))
    if not a:
        return runs
    a, b, h = np.concatenate(a), np.concatenate(b), np.concatenate(h)
    out = []
    for rn in runs:
        if rn.get("median") or len(rn["pts"]) < 2:
            out.append(rn)
            continue
        w = rn["w"]
        pts = densify(np.asarray(rn["pts"], np.float64), WALK_TRIM_STEP_M)
        low, high = pts[:, :2].min(0) - w - h.max(), pts[:, :2].max(0) + w + h.max()
        near = (np.minimum(a, b) <= high).all(1) & (np.maximum(a, b) >= low).all(1)
        if not near.any():
            out.append(rn)
            continue
        na, d = a[near], b[near] - a[near]
        curb, outv = sidewalks.run_geometry(pts, rn["side"], w)
        on = np.zeros(len(pts), bool)
        for t in (0.3, w / 2, w - 0.3):
            v = (curb[:, :2] + outv * t)[:, None] - na
            u = np.clip((v * d).sum(2) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
            on |= (np.hypot(*np.moveaxis(v - u[..., None] * d, 2, 0)) < h[near] - WALK_ON_ROAD_M).any(1)
        if not on.any():
            out.append(rn)
            continue
        out.extend(walk_pieces(rn, pts, on))
    return out


def walk_pieces(rn, pts, off):
    """The parts of sidewalk run rn (densified to pts) where bool off is False, each a run like it (the short ones
    dropped): only the run's own ends keep their junctions and open ends."""
    out = []
    edges = np.flatnonzero(np.diff(np.r_[0, (~off).astype(np.int8), 0]))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        piece = pts[i0:i1]
        if len(piece) < 2 or _length(piece) < WALK_TRIM_KEEP_M:
            continue
        out.append({**rn, "pts": densify(simplify(piece)),
                    "j": (rn["j"][0] and i0 == 0, rn["j"][1] and i1 == len(pts)),
                    "open": (rn["open"][0] and i0 == 0, rn["open"][1] and i1 == len(pts))})
    return out


def drape_walks(frame, elev, walks, ground=None):
    """The sidewalk runs and corner pieces of build_tile's walks put on the ground, in place: z = elev at the curb line,
    as the road mesh's draped edges there. The runs carried the road's centreline z, sampled at its vertices only and
    interpolated along straight stretches (simplify keeps the xy bends), so a sidewalk crossing a dip ran as a chord
    up to the dip's depth in the air beside the draped road. Each run is sampled every WALK_DRAPE_M m, thinned to the
    points the ground's shape needs (WALK_DRAPE_TOL_M) and densified again: a flat street keeps exactly its points.
    One elev call (its cost is per call). ground(x (n,), y (n,)) -> z (n,): the terrain block's drawn ground in the
    frame (roadmesh.drawn_ground), which the road surfaces lie on (roadmesh.build follow) and elev is off by its
    height quantum and more between its samples; the curbs take it instead, sampled every WALK_GROUND_M m (the user,
    2026-10-21: sidewalks on the drawn ground with upstream's road surfaces). One ground call."""
    import sidewalks
    items, xy = [], []
    for wk in walks:
        if "corner" in wk:
            arc = np.asarray(wk["corner"]["arc"], np.float64)
            items.append((wk, arc))
            xy.append(arc[:, :2])
            continue
        pts = densify(np.asarray(wk["pts"], np.float64), WALK_DRAPE_M if ground is None else WALK_GROUND_M)
        curb, _ = sidewalks.run_geometry(pts, wk["side"], wk.get("w", SIDEWALK["w"]))
        items.append((wk, pts))
        xy.append(curb[:, :2])
    if not items:
        return
    xy = np.concatenate(xy)
    if ground is not None:
        u = np.asarray(ground(xy[:, 0], xy[:, 1]), np.float64)
    else:
        la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        _, _, u = frame.geodetic_to_enu(la, lo, np.asarray(elev(la, lo), np.float64))
    k = 0
    for wk, pts in items:
        z = u[k:k + len(pts)]
        k += len(pts)
        if "corner" in wk:
            c = wk["corner"]
            arc = pts.copy()
            arc[:, 2] = z
            # the outer corners: the ends at the ends of the arc (where the runs' slabs end), the rest between
            zo = 0.5 * (z[0] + z[-1])
            outer = [tuple(q) for q in c["outer"]]
            outer = [(*outer[0][:2], z[-1])] + [(*q[:2], zo) for q in outer[1:-1]] + [(*outer[-1][:2], z[0])]
            wk["corner"] = {**c, "arc": arc, "outer": outer}
            continue
        pts = pts.copy()
        pts[:, 2] = z
        wk["pts"] = densify(pts[sidewalks.simplify(pts, WALK_DRAPE_TOL_M)])
