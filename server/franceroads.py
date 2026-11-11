"""Opt-in French road scenery (BNE_FR_ROADS=on).

Mapped widths/lane rules win. Unmapped rural widths, edge paint and bend
restrictions are scenery estimates, not surveyed road markings or traffic law.
The flag is read at process start and included by revisit.code_key in caches.
"""
import math
import os

import numpy as np

ON = os.environ.get("BNE_FR_ROADS", "off").lower() in {"on", "1", "true"}
VERSION = "fr-roads-signs-v8"
PAINT_MATERIAL = "line_bne_fr_white"
RURAL_WIDTH = {"primary": 7.0, "secondary": 6.0, "tertiary": 5.0,
               "unclassified": 4.5, "residential": 5.0, "service": 3.5}
# Unmapped streets in town: the base class width plus two parking lanes drew 10 m residential streets; 2.5 m sidewalks
# on both sides took the room left between the houses.
URBAN_WIDTH = {"secondary": 7.0, "tertiary": 6.5, "unclassified": 6.0, "residential": 5.5, "living_street": 4.5}
URBAN_ONE_WAY_M = 4.0
URBAN_WALK_M = 1.6
# A through road's lanes need their room whatever width was read for it (two lanes one way drawn 4 or 5 m wide), and the
# pieces of one street take one width: BD TOPO's differ piece by piece, and the kerbs pinched in at every change.
THROUGH = {"primary", "secondary", "tertiary"}
LANE_MIN_M = 3.0
LANES_MAX_M = 12.0
# A motorway carriageway: 3.5 m lanes, a 3 m hard shoulder on the right (bande d'arrêt d'urgence) and a 1 m strip on the
# left (bande dérasée de gauche): ICTAAL / ICTAVRU. The class's default was the US interstate's (4 m lanes, 3 m both
# sides: 18 m for three lanes, 14.5 here)
MOTORWAY_LANE_M = 3.5
MOTORWAY_SHOULDERS_M = 3.0 + 1.0
SPLIT_PIECE_M = 150.0           # m: an unnamed one-way piece this short is part of a junction: no sidewalks on it
WALK_STEP_M = 5.0               # m between the points a sidewalk's surroundings are read at (walks_in_town)
WALK_TOWN_REACH_M = 20.0        # m each side of it the land cover is read
WALK_TOWN_WINDOW_M = 70.0       # m along it a built-up reading counts (shorter chopped a street between two houses)
WALK_TOWN_KEEP_M = 40.0         # m: a shorter stretch left after cutting is dropped
WALK_ISOLATED_M = 12.0          # m: a shorter run joined to nothing is a scrap in a junction
SAME_STREET_SPREAD_M = 3.5      # m: wider apart than this, the pieces really are different roads
T1, TP1, T3, T2, T4 = (3., 10.), (1.5, 5.), (3., 1.33), (3., 3.5), (39., 13.)

# Appearance reference supplied by Alexandre for the forest road north of
# Haguenau (old D27, now D927). Scope is one identified OSM way, not all roads.
# The photos establish a mixed line without verges, not every change of passing
# permission along its length. This trial heading is therefore representative;
# explicit OSM restrictions always take precedence.
OBSERVED_MIXED = {15954261: 330.}


def observed_lane_tags(feats, i, tags):
    import osmtags
    tags = dict(tags or {})
    if not ON or any(k.startswith("overtaking") or k == "divider" for k in tags):
        return tags or None
    col = feats.attrs.get("src_record")
    way = osmtags.way_id(str(col[i])) if col is not None else None
    if way not in OBSERVED_MIXED:
        return tags or None
    a = feats.coord_off[feats.ring_off[feats.part_off[i]]]
    b = feats.coord_off[feats.ring_off[feats.part_off[i + 1]]]
    ll = np.asarray(feats.lonlat[a:b])
    if len(ll) < 2:
        return tags or None
    delta = ll[-1] - ll[0]
    delta[0] *= math.cos(math.radians(float(ll[:, 1].mean())))
    if np.linalg.norm(delta) < 1e-10:
        return tags or None
    bearing = math.radians(OBSERVED_MIXED[way])
    forward = float(delta @ np.array([math.sin(bearing), math.cos(bearing)])) > 0
    tags["overtaking:forward"] = "no" if forward else "yes"
    tags["overtaking:backward"] = "yes" if forward else "no"
    tags["_fr_edge"] = "no"
    return tags


def in_france(lat, lon):
    """Is the point in metropolitan France's box, with the French scenery on?"""
    return ON and 41.3 < lat < 51.2 and -5.3 < lon < 9.8


def french_feature(feats, i):
    """Conservative whole-feature gate, using its geography, never an airport.

    Border-crossing/unknown features keep upstream geometry and paint. Testing
    vertices and segment midpoints also catches a road crossing a narrow enclave.
    """
    return bool(french_features(feats)[i])


def french_features(feats):
    """french_feature of every feature of feats, made once (one border lookup for all their points) and remembered on
    feats: a cell's roads were each looked up again by every spec of it (ground, bridges ...), ~3300 calls for ~1000
    roads, the most of a first visit's block planning in France (2026-11-11)."""
    got = feats.__dict__.get("_bnge_french")
    if got is not None and len(got) == len(feats):
        return got
    import countrybounds
    n = len(feats)
    out = np.zeros(n, bool)
    if n:
        # every feature's coordinates at once (a road's run of them: coord_off from its first part's first ring to the
        # next road's), its points and the middles of its segments; one road at a time was ~3 s of a city's first
        # block (2026-11-11)
        a = np.asarray(feats.coord_off)[np.asarray(feats.ring_off)[np.asarray(feats.part_off)[:-1]]].astype(np.int64)
        b = np.asarray(feats.coord_off)[np.asarray(feats.ring_off)[np.asarray(feats.part_off)[1:]]].astype(np.int64)
        ll = np.asarray(feats.lonlat, np.float64)
        cnt = b - a
        own = np.repeat(np.arange(n), cnt)
        idx = np.concatenate([np.arange(x, y) for x, y in zip(a.tolist(), b.tolist())]) if cnt.sum() else \
            np.zeros(0, np.int64)
        pv = ll[idx] if len(idx) else np.zeros((0, 2))
        # Metropolitan France only; overseas territories keep their own codes.
        good = np.isfinite(pv).all(axis=1) & (pv[:, 0] > -5.3) & (pv[:, 0] < 9.8) & (pv[:, 1] > 41.3) & (pv[:, 1] < 51.2)
        bad = np.zeros(n, bool)
        bad[own[~good]] = True
        ok = (cnt >= 2) & ~bad
        keep = ok[own]
        if keep.any():
            pk, ok_own = pv[keep], own[keep]
            same = ok_own[1:] == ok_own[:-1]            # (consecutive points of one road: a segment)
            mid = ((pk[:-1] + pk[1:]) / 2)[same]
            p = np.concatenate([pk, mid])
            po = np.concatenate([ok_own, ok_own[:-1][same]])
            fr = np.array([code == "FR" for code in countrybounds.bundled().codes_at(np.c_[p[:, 1], p[:, 0]])], bool)
            # a feature is French when all its points are
            out[np.unique(po)] = True
            out[np.unique(po[~fr])] = False
    feats.__dict__["_bnge_french"] = out
    return out


def adjust_specs(feats, specs):
    import roads
    if not ON:
        return specs
    out = list(specs)
    names, lengths = feats.attrs.get("name"), None
    for i, sp in enumerate(out):
        if sp is None or sp.kind != "paved" or not french_feature(feats, i):
            continue
        sp = sp._replace(france=True)
        mapped = roads._rule_value(feats.rules.get("width"), i)
        if mapped is None:
            mapped = roads._osm(feats, "width", i, *roads.OSM_WIDTH_M)
        if (not sp.urban and not sp.link and not sp.median and not getattr(sp, "mountain", False)
                and not roads.is_ring(feats, i)
                and sp.cls in RURAL_WIDTH and mapped is None):
            if sp.lanes:
                # Two mapped lanes describe traffic directions, not a measured
                # six-metre width: narrow French roads can have two small lanes.
                width = RURAL_WIDTH[sp.cls] if int(sp.lanes) == 2 and not sp.oneway else \
                    max(3.0, int(sp.lanes) * 3.0)
            else:
                width = 3.5 if sp.oneway else RURAL_WIDTH[sp.cls]
            sp = sp._replace(width=width)
        if sp.cls == "motorway" and sp.oneway and not sp.link and mapped is None:
            sp = sp._replace(width=min(sp.width, int(sp.lanes or 2) * MOTORWAY_LANE_M + MOTORWAY_SHOULDERS_M))
        if sp.urban:
            if (mapped is None and not sp.link and not sp.median and not roads.is_ring(feats, i)
                    and sp.cls in URBAN_WIDTH and not sp.lanes):
                sp = sp._replace(width=URBAN_ONE_WAY_M if sp.oneway else URBAN_WIDTH[sp.cls], parking=False)
            sp = sp._replace(walk=min(sp.walk, URBAN_WALK_M))
            if names is not None and sp.oneway and sp.cls != "service" and sp.sides is None \
                    and str(names[i]).strip() in ("", "nan", "None"):
                lengths = roads._lengths(feats) if lengths is None else lengths
                if lengths[i] < SPLIT_PIECE_M:
                    sp = sp._replace(sides=())
        if sp.cls in THROUGH and not sp.link and sp.lanes and int(sp.lanes) >= 2:
            sp = sp._replace(width=_clear_of_median(sp, max(sp.width, min(int(sp.lanes) * LANE_MIN_M, LANES_MAX_M))))
        out[i] = sp
    return _one_width_per_street(feats, out)


def walks_in_town(runs, frame, urban_fn, french):
    """runs without the stretches of a French road's sidewalk (fids in french) that have nothing built within
    WALK_TOWN_REACH_M: the road was urban as a whole, its sidewalks ran on between the fields; and without the scraps
    left in junctions (WALK_ISOLATED_M, joined to nothing). One land-cover call."""
    import roads
    import sidewalks
    todo, pts_of, xy = [], [], []
    for k, rn in enumerate(runs):
        if rn.get("median") or rn.get("fid") not in french or len(rn["pts"]) < 2:
            continue
        pts = roads.densify(np.asarray(rn["pts"], np.float64), WALK_STEP_M)
        _, way = sidewalks.run_geometry(pts, rn["side"], rn["w"])
        todo.append(k)
        pts_of.append(pts)
        xy.append(np.concatenate([pts[:, :2], pts[:, :2] + way * WALK_TOWN_REACH_M, pts[:, :2] - way * WALK_TOWN_REACH_M]))
    if not todo:
        return runs
    all_xy = np.concatenate(xy)
    lat, lon, _ = frame.enu_to_geodetic(all_xy[:, 0], all_xy[:, 1], 0.0)
    urban = np.asarray(urban_fn(np.asarray(lat), np.asarray(lon)), bool)
    done, at = {}, 0
    reach = int(round(WALK_TOWN_WINDOW_M / WALK_STEP_M))
    for k, pts in zip(todo, pts_of):
        n = len(pts)
        near = urban[at:at + n] | urban[at + n:at + 2 * n] | urban[at + 2 * n:at + 3 * n]
        at += 3 * n
        rn = runs[k]
        if roads._length(pts) < WALK_ISOLATED_M and not any(rn["j"]):
            done[k] = []
        elif not near.all():
            built = np.convolve(near.astype(np.int32), np.ones(2 * reach + 1, np.int32), "same") > 0
            if not built.all():
                done[k] = [p for p in roads.walk_pieces(rn, pts, ~built) if roads._length(p["pts"]) >= WALK_TOWN_KEEP_M]
    return [p for k, rn in enumerate(runs) for p in done.get(k, [rn])]


def _clear_of_median(sp, width):
    """width, but no wider than a divided road's carriageways leave room for (never below what it had)."""
    return max(sp.width, min(width, sp.median - .6)) if sp.median else width


def _one_width_per_street(feats, specs):
    """specs with the French pieces of one named street (same class and direction rule) at their length-weighted median
    width, where their widths are close enough to be one road (SAME_STREET_SPREAD_M). Ramps and roundabouts keep theirs."""
    import roads
    names = feats.attrs.get("name")
    if names is None:
        return specs
    groups = {}
    for i, sp in enumerate(specs):
        name = str(names[i]).strip()
        if sp is None or not sp.france or sp.link or name in ("", "nan", "None") or roads.is_ring(feats, i):
            continue
        groups.setdefault((name, sp.cls, bool(sp.oneway)), []).append(i)
    if not any(len(g) > 1 for g in groups.values()):
        return specs
    lengths = roads._lengths(feats)
    out = list(specs)
    for group in groups.values():
        widths = np.array([specs[i].width for i in group], np.float64)
        if len(group) < 2 or widths.max() - widths.min() > SAME_STREET_SPREAD_M:
            continue
        order = np.argsort(widths)
        weight = np.nan_to_num(lengths[np.array(group)][order], nan=0.0)
        if weight.sum() <= 0:
            continue
        width = float(widths[order][np.searchsorted(np.cumsum(weight), weight.sum() / 2.0)])
        for i in group:
            out[i] = specs[i]._replace(width=_clear_of_median(specs[i], width) if specs[i].median and width > specs[i].width
                                       else width)
    return out


def enabled(spec):
    return ON and bool(getattr(spec, "france", False)) and not getattr(spec, "mountain", False)


def narrow(spec):
    return (not spec.urban and not spec.oneway and not spec.link and not spec.median
            and spec.kind == "paved" and 4.0 <= spec.width < 5.2
            and spec.cls in {"primary", "secondary", "tertiary", "unclassified"})


def edge_pattern(spec):
    """None = no inferred edge paint, False = solid, tuple = dash/gap metres."""
    if spec.kind != "paved" or spec.urban or narrow(spec):
        return None
    if spec.cls == "motorway":
        return T4
    if spec.cls == "trunk" and spec.oneway:
        return T4
    if spec.cls in {"primary", "secondary", "trunk"} and spec.width >= 6.5:
        return T2
    return None


def unit(spec):
    return .075 if spec.cls == "motorway" or spec.median or (spec.lanes or 0) >= 4 else \
        .06 if spec.cls in {"trunk", "primary"} else .05


def continuations(rows):
    """Mutual straightest road ends; keep paint across an ordinary map split.

    Unlike the gravel-edge rule, crossing/T arms are not all continuations.
    Other junction arms still clip the markings normally.
    """
    import furniture
    result = set()
    for arms in furniture._arms(rows).values():
        best = {}
        for a in arms:
            options = [(float(a[2] @ b[2]), b[0]) for b in arms if a[0] != b[0]
                       and a[1].oneway == b[1].oneway
                       and (not a[1].oneway or furniture._outbound(a) != furniture._outbound(b))]
            if options:
                dot, other = min(options)
                if dot <= -math.cos(math.radians(20)):
                    best[a[0]] = other
        for a, b in best.items():
            if best.get(b) == a:
                result.add((a, b))
    return result


def stations(pts):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.asarray(pts)[:, :2], axis=0), axis=1))]


def at(pts, s):
    """Position and horizontal forward tangent, with duplicate nodes removed."""
    p = np.asarray(pts, dtype=float)
    p = p[np.r_[True, np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) > 1e-6]]
    if len(p) < 2:
        return p[0].copy(), np.array([0., 1.])
    ss = stations(p)
    s = float(np.clip(s, 0., ss[-1]))
    i = min(int(np.searchsorted(ss, s, side="right")) - 1, len(p) - 2)
    t = (s - ss[i]) / (ss[i + 1] - ss[i])
    d = p[i + 1, :2] - p[i, :2]
    return p[i] * (1 - t) + p[i + 1] * t, d / np.linalg.norm(d)


def sample(pts, values):
    """Batch version of at, so long features cost O(vertices + samples)."""
    p = np.asarray(pts, float)
    p = p[np.r_[True, np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) > 1e-6]]
    values = np.asarray(values, float)
    if len(p) < 2:
        return np.tile(p[0], (len(values), 1)), np.tile([0., 1.], (len(values), 1))
    ss = stations(p)
    values = np.clip(values, 0., ss[-1])
    ii = np.clip(np.searchsorted(ss, values, side="right") - 1, 0, len(p) - 2)
    t = ((values - ss[ii]) / (ss[ii + 1] - ss[ii]))[:, None]
    q = p[ii] * (1 - t) + p[ii + 1] * t
    d = p[ii + 1, :2] - p[ii, :2]
    d /= np.linalg.norm(d, axis=1)[:, None]
    return q, d


def segment(pts, a, b):
    ss = stations(pts)
    return np.vstack([at(pts, a)[0], np.asarray(pts)[(ss > a) & (ss < b)], at(pts, b)[0]])


def bends(pts):
    """Pronounced bends, sampled on the whole road before any tile clipping.

    These are candidates, not a map of official dangerous-bend sign locations.
    Return (station m, signed turn radians); positive is left along the geometry.
    """
    length = stations(pts)[-1]
    ss = np.arange(30., length - 30., 10.)
    a, _ = sample(pts, ss - 30.)
    p, _ = sample(pts, ss)
    b, _ = sample(pts, ss + 30.)
    u, v = p[:, :2] - a[:, :2], b[:, :2] - p[:, :2]
    turns = np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], np.sum(u * v, axis=1))
    keep = (np.minimum(np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)) >= 10) & (np.abs(turns) >= math.radians(25))
    candidates = list(zip(ss[keep].tolist(), turns[keep].tolist()))
    # Keep the strongest sample of each bend; no row of signs on one curve.
    groups = []
    for event in candidates:
        if groups and event[0] - groups[-1][-1][0] <= 40 and event[1] * groups[-1][-1][1] > 0:
            groups[-1].append(event)
        else:
            groups.append([event])
    return [max(g, key=lambda e: abs(e[1])) for g in groups]


def mre_polygons(pts):
    """IISR D9 MRE: 10 cm half-width, 25 cm sides, 8 cm step, 120° angle.

    Two parallelograms per mark, 15 m apart. Their true 20 cm width is preserved
    through clipping and mesh export; these are not a stretched dash texture.
    """
    length = stations(pts)[-1]
    shear = .1 / math.sqrt(3)
    positions, directions = sample(pts, np.arange(7.5, length - .5, 15.))
    for p, d in zip(positions, directions):
        left = np.array([-d[1], d[0]])
        for x0, x1, step in ((-.1, 0., 0.), (0., .1, .08 + shear)):
            xy = [(x0, step), (x1, step + shear), (x1, step + shear + .25), (x0, step + .25)]
            q = np.tile(p, (4, 1))
            for i, (x, y) in enumerate(xy):
                q[i, :2] += left * x + d * (y - (.25 + .08 + 2 * shear) / 2)
            yield q


def recall_polygons(pts, s, direction):
    """Approximate six-metre curved return arrow, bending to the driver's right.

    Convex pieces avoid concave-polygon triangulation ambiguities. Use only for
    an inferred end of an overtaking section, never as generic lane-turn arrows.
    """
    p, d = at(pts, s)
    d *= direction
    left = np.array([-d[1], d[0]])
    t = np.linspace(0., 1., 9)
    xy = np.c_[.55 - .85 * t * t, -3. + 5. * t]

    def world(local):
        q = np.tile(p, (len(local), 1))
        local = np.asarray(local)
        q[:, :2] += local[:, 0, None] * left + local[:, 1, None] * d
        # The arrow follows the road's grade rather than floating horizontally.
        for k, y in enumerate(local[:, 1]):
            q[k, 2] = at(pts, s + direction * y)[0][2]
        return q

    for a, b in zip(xy[:-1], xy[1:]):
        n = np.array([b[1] - a[1], a[0] - b[0]])
        n = n / np.linalg.norm(n) * .075
        yield world([a - n, b - n, b + n, a + n])
    yield world([(-.76, 1.45), (-.5, 3.), (.12, 1.85)])


def paint(spec, pts, osm=None):
    """roads.extras tuples. Whole-road phases; geometry-inferred fallback is explicit.

    Where OSM describes overtaking, it wins for the entire mapped feature. Without
    that evidence, only pronounced bends receive estimated no-passing approaches.
    Observed local references are supplied by observed_lane_tags; their limits
    are documented separately from mapped passing restrictions.
    """
    import markings
    import roads
    out = []
    osm = {k: v for k, v in (osm or {}).items() if v is not None}
    if osm.get("overtaking") in {"forward", "backward"}:
        allowed = osm.pop("overtaking")
        osm.setdefault("overtaking:forward", "yes" if allowed == "forward" else "no")
        osm.setdefault("overtaking:backward", "yes" if allowed == "backward" else "no")
    if osm.get("divider") == "solid_line" and not any(k.startswith("overtaking") for k in osm):
        osm["overtaking"] = "no"
    elif osm.get("divider") == "dashed_line" and not any(k.startswith("overtaking") for k in osm):
        osm["overtaking"] = "yes"
    if osm.get("overtaking") == "both":
        osm["overtaking"] = "yes"
    if narrow(spec):
        # MRE guides an undivided road: OSM lane_markings=no is compatible.
        return [({"m": PAINT_MATERIAL, "tl": 1., "p": 5, "kind": "mre", "polygon": True}, 0., q)
                for q in mre_polygons(pts)]
    no_lanes = str(osm.get("lane_markings", "")).lower() == "no"
    width = 2 * unit(spec)
    length = stations(pts)[-1]
    if length < .01:
        return out

    def line(kind, offset, pattern=None, a=0., b=None, side=None, arrows=None):
        run = segment(pts, a, length if b is None else b)
        if abs(offset) > 1e-8:
            run = roads.offset_polyline(run, offset)
        look = {"m": PAINT_MATERIAL, "tl": 1., "p": 5, "kind": kind}
        if side is not None:
            look["side"] = side
        for p in [run] if pattern is None else markings.dashes(run, pattern, phase=a):
            if arrows is not None and any(
                not arrows.visible(start, end, width / 2 + .025)
                for start, end in zip(p[:-1], p[1:])
            ):
                continue              # a recall arrow takes the place of the overlapping dash
            out.append((look.copy(), 3 * unit(spec) if kind == "edge" else width, p))

    drive, lanes = roads.lane_layout(spec)
    plan = markings.lane_plan(osm, not spec.oneway, drive, lanes, False)
    if plan is None:
        plan = markings.lane_plan({"lanes:forward": str(lanes if spec.oneway else max(1, lanes // 2))},
                                  not spec.oneway, drive, lanes, False)
    if no_lanes:
        plan = []                         # edge guidance is independent of lanes
    explicit = any(str(osm.get(k, "")).lower() in {"yes", "no", "both", "forward", "backward"}
                   for k in ("overtaking", "overtaking:forward", "overtaking:backward", "divider"))
    events = bends(pts) if not explicit and not spec.urban and not spec.oneway and spec.width >= 5.2 else []
    for off, role, how in plan or []:
        if role == "lane":
            line("lane", off, None if how else (TP1 if spec.urban else T1))
        elif role == "centre" and spec.width >= 5.2:
            gap = (width + .1) / 2
            pattern = TP1 if spec.urban else T1
            if how == "double":
                line("centre", off)
            elif how in {"solid_right", "solid_left"}:
                sign = -1 if how == "solid_right" else 1
                line("centre", off + sign * gap)
                line("centre", off - sign * gap, pattern)
            elif events:
                import signplacement

                recall = [
                    q
                    for s, _ in events
                    for direction in (1, -1)
                    for back in (210., 185., 160.)
                    if 4. <= s - direction * back <= length - 4.
                    for q in recall_polygons(pts, s - direction * back, direction)
                ]
                arrows = signplacement.Context(recall) if recall else None
                cuts = sorted({0., length} | {float(np.clip(s + d, 0, length))
                                             for s, _ in events for d in (-225., -150., -35., 35., 150., 225.)})
                for a, b in zip(cuts[:-1], cuts[1:]):
                    if b - a < .1:
                        continue
                    mid = (a + b) / 2
                    fwd = any(s - 150 <= mid <= s + 35 for s, _ in events)
                    bwd = any(s - 35 <= mid <= s + 150 for s, _ in events)
                    warning = any(s - 225 <= mid <= s - 150 or s + 150 <= mid <= s + 225 for s, _ in events)
                    if fwd and bwd:
                        line("centre", off, a=a, b=b)
                    elif fwd or bwd:
                        sign = -1 if fwd else 1
                        line("centre", off + sign * gap, a=a, b=b)
                        line("centre", off - sign * gap, T3, a, b, arrows=arrows)
                    else:
                        line("centre", off, T3 if warning else pattern, a, b, arrows=arrows)
                out.extend(({"m": PAINT_MATERIAL, "tl": 1., "p": 5, "kind": "recall", "polygon": True}, 0., q)
                           for q in recall)
            else:
                line("centre", off, pattern)
        elif role == "twltl":
            # Preserve upstream's mapped shared turn lane, in French white.
            gap = (width + .1) / 2
            line("centre", off + how * gap)
            line("centre", off - how * gap, TP1 if spec.urban else T1)
    ep = edge_pattern(spec)
    if ep is not None and not roads.has_sidewalks(spec) and osm.get("_fr_edge") != "no":
        inset = roads.motorway_shoulder(spec) if roads.rural_motorway(spec) else .25
        for side in (1, -1):
            # The central reservation's edge is continuous, the hard shoulder T4.
            pat = None if spec.median and side == (1 if spec.dir >= 0 else -1) else ep
            line("edge", side * (spec.width / 2 - inset), pat, side=side)
    return out
