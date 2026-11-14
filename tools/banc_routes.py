#!/usr/bin/env python3
"""banc_routes.py: the roads on the ground of a zone measured against the terrain the game gets (after EarthNG's
road_audit.py, Alezx311, MIT: the road's line walked against the ground, defects counted per km).

    cd server && python ../tools/banc_routes.py LAT0 LON0 LAT1 LON1 --out DIR [--hours 2]

Run from server/ with the app's switches (launcher.FORK_DEFAULTS, as banc_qualite.py). For each PLACE_M square of the
zone (one anchor each), the final ground of a terrain block (blockworker.build's: graded roads, arches over tunnels,
held under the decks, the abutments' faces) is read along every grading plan of a road on the ground (its core: the
carriageway the car drives on) every STEP_M, and these are counted:

  haut       the ground over the road's planned height by more than AXIS_M on its line (something laid over the road:
             a fill, an embankment, a face's rise; at grade with ROADS_IN_GROUND the ground IS the road surface)
  bas        the ground under it by more than AXIS_M (a cut, an abutment's face dug into it, a dip)
  travers    the ground across the carriageway off a straight line (its bank) by more than ACROSS_M: a bump or a
             trough in the lanes
  devers     the carriageway's cross slope over BANK_MAX (a lane tilted off the road's own banking)
  marche     the ground along the line stepping by STEP_JUMP_M more than the plan does within 1 m (a seam, a wall)
  carrefour  a junction (3+ roads' plan ends within NODE_M) whose ground over the disc of its widest road is off its
             best plane by more than NODE_RES_M, or whose roads' planned heights at the node spread over NODE_SPREAD_M
  sans_plan  a stretch of a drawn road (road_specs) with no grading plan within NO_PLAN_M of its line

Stations within JUNCTION_SKIP_M of a junction node are left to the junction check (several plans blend there), and
within PORTAL_SKIP_M of a tunnel's portal skipped. Railways' bed plans and junction corner patches are not roads.
Writes DIR/routes.jsonl (one line per defect, a run of stations merged), DIR/zones.jsonl (per place: km measured by
class, counts) and DIR/rapport.md (the worst first, rates per km). Nothing else is written outside DIR (the caches the
server writes anyway: see beamng_earth_bancs, BNE_REVISIT=off).
"""
import argparse
import json
import logging
import math
import multiprocessing
import os
import sys
import time

import numpy as np

PLACE_M = 400.0         # m: a place's square (one anchor, one block's ground)
STEP_M = 1.0            # m between stations along a plan
AXIS_M = 0.10           # m: the ground off the road's planned height on its line
ACROSS_M = 0.08         # m: the ground across the lanes off its straight line
BANK_MAX = 0.10         # cross slope (m/m) of a carriageway past this: tilted
STEP_JUMP_M = 0.25      # m: a step within 1 m beyond what the plan itself does there
NODE_M = 1.5            # m: plan ends this close are one node
JUNCTION_SKIP_M = 4.0   # m past a junction's widest half its stations are not walked
PORTAL_SKIP_M = 10.0    # m round a tunnel's portal its road's stations are not walked (the hill over the mouth)
NODE_RES_M = 0.15       # m: a junction's ground off its best plane
NODE_SPREAD_M = 0.30    # m: the spread of its roads' planned heights at the node
NO_PLAN_M = 2.0         # m: a drawn road's point this far from every plan has none
NO_PLAN_STEP_M = 5.0    # m between a drawn road's sampled points
MIN_RUN_M = 2.0         # m: a shorter run of off stations is not a defect (one station's noise)
EDGE_IN_M = 0.5         # m inside the carriageway's edges the across samples stand (at the edge a road beside it on a
                        # bank or in a cutting legitimately starts its slope)


def args_parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("lat0", "lon0", "lat1", "lon1"):
        ap.add_argument(k, type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=2.0, help="stop starting new places after this long")
    ap.add_argument("--cache", default=os.path.expanduser(os.path.join("~", ".cache", "beamng-earth")))
    ap.add_argument("--places", type=int, default=0, help="at most this many places (0: all)")
    return ap.parse_args(argv)


def setup_env():
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.getcwd()))
    try:
        import launcher
        launcher.apply_fork_defaults()
    except Exception as ex:             # (the server's own defaults then)
        print("launcher defaults:", type(ex).__name__, ex, flush=True)
    os.environ.setdefault("BNE_REVISIT", "off")


# ---- pure parts (tests/test_banc_routes.py) ----

def runs(mask, arc, min_len=MIN_RUN_M):
    """[(i0, i1)] index ranges (i1 inclusive) of the True runs of mask whose arc length is min_len or more."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if arc[j] - arc[i] >= min_len - 1e-9:
            out.append((i, j))
        i = j + 1
    return out


def stations(xy, step=STEP_M):
    """(arc (k,), points (k, 2), unit tangents (k, 2)) every step m along the polyline xy (k >= 2), its last point too."""
    xy = np.asarray(xy, np.float64)[:, :2]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    a = np.arange(0.0, s[-1], step)
    a = np.r_[a, s[-1]] if len(a) == 0 or s[-1] - a[-1] > 1e-6 else a
    p = np.stack([np.interp(a, s, xy[:, 0]), np.interp(a, s, xy[:, 1])], 1)
    tx, ty = np.gradient(p[:, 0]), np.gradient(p[:, 1])
    nn = np.hypot(tx, ty) + 1e-12
    return a, p, np.stack([tx / nn, ty / nn], 1)


def across_fit(offs, z):
    """(slope (k,), worst residual (k,)) of the straight line through each row of z (k, m) over the offsets offs (m,)."""
    offs = np.asarray(offs, np.float64)
    z = np.asarray(z, np.float64)
    o = offs - offs.mean()
    slope = (z - z.mean(1, keepdims=True)) @ o / float((o * o).sum())
    fit = z.mean(1, keepdims=True) + slope[:, None] * o[None]
    return slope, np.abs(z - fit).max(1)


def steps(arc, ground, plan, jump=STEP_JUMP_M):
    """Stations where the ground steps by jump more than the plan does from the station 1 m before."""
    arc = np.asarray(arc, np.float64)
    if len(arc) < 2:
        return np.zeros(len(arc), bool)
    dg, dp = np.diff(np.asarray(ground, np.float64)), np.diff(np.asarray(plan, np.float64))
    hit = (np.abs(dg - dp) > jump) & (np.diff(arc) <= 1.5 * STEP_M)
    return np.r_[False, hit]


def nodes(ends, rids, tol=NODE_M):
    """[(xy (2,), [plan indices])] of the plan ends ends (n, 2) grouped within tol, each with 3+ distinct rids:
    the junctions (rids: the road id of each end's plan)."""
    ends = np.asarray(ends, np.float64).reshape(-1, 2)
    left = list(range(len(ends)))
    out = []
    while left:
        i = left.pop(0)
        grp = [i] + [j for j in left if np.hypot(*(ends[j] - ends[i])) <= tol]
        left = [j for j in left if j not in grp]
        if len({rids[k] for k in grp}) >= 3:
            out.append((ends[grp].mean(0), grp))
    return out


def plane_residual(xy, z):
    """(slope m/m, worst |residual|) of the best plane through z (n,) at xy (n, 2)."""
    A = np.c_[np.asarray(xy, np.float64), np.ones(len(z))]
    c, *_ = np.linalg.lstsq(A, np.asarray(z, np.float64), rcond=None)
    return float(np.hypot(c[0], c[1])), float(np.abs(A @ c - z).max())


def far_from(points, lines, reach):
    """(n,) bool: the points (n, 2) farther than reach from every polyline of lines (a grid of reach-sized cells)."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if not len(points):
        return np.zeros(0, bool)
    if not lines:
        return np.ones(len(points), bool)
    segs = np.concatenate([np.stack([ln[:-1, :2], ln[1:, :2]], 1) for ln in lines if len(ln) >= 2])
    cell = {}
    lo = np.minimum(segs[:, 0], segs[:, 1]) - reach
    hi = np.maximum(segs[:, 0], segs[:, 1]) + reach
    for k, (a, b) in enumerate(zip(np.floor(lo / reach).astype(int), np.floor(hi / reach).astype(int))):
        for gx in range(a[0], b[0] + 1):
            for gy in range(a[1], b[1] + 1):
                cell.setdefault((gx, gy), []).append(k)
    out = np.ones(len(points), bool)
    for i, p in enumerate(points):
        ks = cell.get((int(math.floor(p[0] / reach)), int(math.floor(p[1] / reach))))
        if not ks:
            continue
        a, b = segs[ks, 0], segs[ks, 1]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        if (np.hypot(*(a + t[:, None] * ab - p).T) <= reach).any():
            out[i] = False
    return out


LAYER_NAMES = ("relief", "sol", "nivellement", "arches", "tabliers", "murs")    # (the last two: our fork's only)


def _bridge_lines(roads, cell_fn, frame, reach):
    """The bridges' lines (ENU) within reach of the place's middle, from the map cells (roads.bridge_parts): where a
    server holds no deck list of its own (upstream's blockworker)."""
    import mapcells
    m = (reach + 30.0) / 111320.0
    la0, lo0, _ = frame.enu_to_geodetic(np.array([0.0]), np.array([0.0]), 0.0)
    la0, lo0 = float(la0[0]), float(lo0[0])
    k = max(0.05, math.cos(math.radians(la0)))
    out = []
    for key in mapcells.cells_for_bbox(la0 - m, lo0 - m / k, la0 + m, lo0 + m / k):
        f = cell_fn(key)
        if f is None or not len(f):
            continue
        for i in range(len(f)):
            for ring in roads.bridge_parts(f, i):
                ring = np.asarray(ring, np.float64)
                if len(ring) < 2:
                    continue
                e, n, _ = frame.geodetic_to_enu(ring[:, 1], ring[:, 0], 0.0)
                out.append(np.stack([np.asarray(e), np.asarray(n)], 1))
    return out


def cause(kind, L, k, zp, offs):
    """(the step of the ground's making that makes a defect at station k, whether the water's or the airports' carve
    (sol under relief) is there): L (6 layers, stations, offsets) as layers() gives it, zp the planned heights, offs
    the offsets across. The grading's own when its layer already shows most of it, else the later step that moved
    the ground most toward it. Pure."""
    wet = bool(np.min(L[1, k] - L[0, k]) < -0.3)
    if kind in ("haut", "bas"):
        dev = L[2:, k, len(offs) // 2] - zp[k]            # (nivellement .. murs, on the line)
    elif kind == "marche":
        if k == 0:
            return "nivellement", wet
        dev = (L[2:, k, len(offs) // 2] - L[2:, k - 1, len(offs) // 2]) - (zp[k] - zp[k - 1])
    elif kind == "travers":
        dev = np.array([across_fit(offs, L[j, k][None])[1][0] for j in range(2, len(L))])
    else:                                                   # (devers)
        dev = np.array([across_fit(offs, L[j, k][None])[0][0] for j in range(2, len(L))])
    final = dev[-1]
    if abs(final) < 1e-9 or abs(dev[0]) >= 0.7 * abs(final):
        return "nivellement", wet
    step = np.diff(dev) * np.sign(final)
    return LAYER_NAMES[3 + int(np.argmax(step))], wet


# ---- the place ----

def measure_place(bw, lat, lon, out, stats, aid=1):
    import geodesy
    import grading
    import mapcells
    import roads
    try:
        import roadstructures   # (our fork's split of roads.py)
    except ImportError:
        roadstructures = roads
    s = bw._s
    half = PLACE_M / 2
    g, _ = s.anchor(aid, lat, lon)     # (an anchor of its own: a grader's plans are mapped into its anchor's frame)
    frame = g.frame if g is not None else geodesy.EnuFrame(lat, lon)
    t = bw.plan_ahead(g, frame, 0.0, 0.0, half + 200.0)
    if t is not None:
        t.join()
    import dem as demmod
    zoom = demmod.zoom_for(lat, 1.25, demmod.FINE_ZOOM)     # (the 1 m terrain block's: earthd's hm_zoom)

    def elev(la, lo):
        z = bw.ground(frame, la, lo, s.dem.sample(la, lo, zoom))
        return g.apply(la, lo, z, grading.ROAD_LIFT)
    cell_fn = lambda key: s.md.cell(key, layers=("roads",), kind="demand").layers.get("roads")   # noqa: E731
    box = (-half - 60.0, -half - 60.0, half + 60.0, half + 60.0)
    tun = bw.tunnel_profiles(aid, frame, elev, g)
    held = hasattr(bw, "_block_decks")      # (the ground held under decks and the abutments' faces: our fork's steps)
    if held:
        decks = bw._block_decks(aid, frame, elev, cell_fn, box)
        walls = bw._abutment_walls(frame, elev, decks)
        cores = bw._wall_cores(g, frame, walls)
        deck_lines = [xy for xy, _z, _w in decks]
    else:
        deck_lines = _bridge_lines(roads, cell_fn, frame, half + 60.0)

    def final(xy):              # (the terrain block's ground: blockworker.build's gen)
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
        z = np.asarray(elev(la, lo), np.float64)
        z = roadstructures.block_arch_raise(tun, cell_fn, frame, la, lo, z, box)
        if not held:
            return np.asarray(z, np.float64)
        z = bw.under_decks(frame, decks, la, lo, z)
        return np.asarray(bw.faced(frame, walls, cores, la, lo, z), np.float64)

    LAYERS = LAYER_NAMES if held else LAYER_NAMES[:4]

    def layers(xy):             # (final's steps one by one: (6, n), its last row final's)
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
        z0 = np.asarray(s.dem.sample(la, lo, zoom), np.float64)
        z1 = np.asarray(bw.ground(frame, la, lo, z0.copy()), np.float64)
        z2 = np.asarray(g.apply(la, lo, z1.copy(), grading.ROAD_LIFT), np.float64)
        z3 = np.asarray(roadstructures.block_arch_raise(tun, cell_fn, frame, la, lo, z2.copy(), box), np.float64)
        if not held:
            return np.stack([z0, z1, z2, z3])
        z4 = np.asarray(bw.under_decks(frame, decks, la, lo, z3.copy()), np.float64)
        z5 = np.asarray(bw.faced(frame, walls, cores, la, lo, z4.copy()), np.float64)
        return np.stack([z0, z1, z2, z3, z4, z5])


    def ll(p):
        la, lo, _ = frame.enu_to_geodetic(np.array([p[0]]), np.array([p[1]]), 0.0)
        return [round(float(la[0]), 6), round(float(lo[0]), 6)]

    gx = np.linspace(-half, half, 9)
    GX, GY = np.meshgrid(gx, gx)
    la, lo, _ = frame.enu_to_geodetic(GX.ravel(), GY.ravel(), 0.0)
    plans, road, portals, rails, seen = [], [], [], [], set()
    for key in g._keys(np.asarray(la), np.asarray(lo)):        # (Grader.plans, with each plan's feature id)
        for fid, p in g._cell_plans(key):
            xy = np.asarray(p.xy, np.float64)[:, :2]
            k = (str(fid), round(float(xy[0, 0]), 1), round(float(xy[0, 1]), 1))
            if len(xy) < 2 or k in seen or not ((np.abs(xy) <= half + 60.0).all(1)).any():
                continue
            seen.add(k)
            plans.append(p)
            if grading.TUNNEL_TAG in str(fid):          # (a tunnel's own floor: its ends are the portals)
                portals += [xy[0], xy[-1]]
            elif str(fid).startswith(grading.RAIL_TAG):
                rails.append(xy)
            elif str(getattr(p, "rid", -1)) != "-1":
                road.append(p)          # (not a railway's bed, nor a junction corner's patch)
    portals = np.array(portals, np.float64).reshape(-1, 2)
    ends = np.array([q for p in road for q in (p.xy[0][:2], p.xy[-1][:2])], np.float64).reshape(-1, 2)
    rids = [str(p.rid) for p in road for _ in (0, 1)]
    junctions = nodes(ends, rids)
    jxy = np.array([c for c, _g in junctions], np.float64).reshape(-1, 2)
    jr = np.array([max(float(road[k // 2].road_half) for k in grp) for _c, grp in junctions], np.float64)

    def near_m(q, lines):
        d = [float(np.min(np.hypot(*(np.asarray(ln)[:, :2] - q).T))) for ln in lines if len(ln)]
        return round(min(d), 1) if d else None

    def emit(kind, plan, arc, pts, vals, i0, i1, extra=None, L=None, zp=None, offs=None):
        k = i0 + int(np.argmax(np.abs(vals[i0:i1 + 1])))
        row = {"kind": kind, "at": ll(pts[k]), "value": round(float(vals[k]), 3),
               "len": round(float(arc[i1] - arc[i0]) + STEP_M, 1), "rid": str(plan.rid),
               "half": round(float(plan.road_half), 2), "place": [round(lat, 5), round(lon, 5)],
               "pont_m": near_m(pts[k], deck_lines), "rail_m": near_m(pts[k], rails),
               "carrefour_m": round(float(np.min(np.hypot(*(jxy - pts[k]).T))), 1) if len(jxy) else None}
        if L is not None:
            row["cause"], row["eau"] = cause(kind, L, k, zp, offs)
        if extra:
            row.update(extra)
        out.write(json.dumps(row) + "\n")
        stats[kind] = stats.get(kind, 0) + 1

    for p in road:
        xy = np.asarray(p.xy, np.float64)[:, :2]
        arc, pts, tan = stations(xy)
        inside = (np.abs(pts) <= half).all(1)
        if len(jxy):
            d = np.hypot(pts[:, None, 0] - jxy[None, :, 0], pts[:, None, 1] - jxy[None, :, 1])
            inside &= ~(d <= jr[None] + JUNCTION_SKIP_M).any(1)
        if len(portals):            # (a portal: the ground over the vault, the block's hole cut there)
            d = np.hypot(pts[:, None, 0] - portals[None, :, 0], pts[:, None, 1] - portals[None, :, 1])
            inside &= ~(d <= PORTAL_SKIP_M).any(1)
        if inside.sum() < 2:
            continue
        sp = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        zp = np.interp(arc, sp, np.asarray(p.z, np.float64)) + grading.ROAD_LIFT
        rh = max(float(p.road_half) - EDGE_IN_M, 1.0)     # (inside the lanes: at the edge a neighbour's slope starts)
        offs = np.array([-rh, -rh / 2, 0.0, rh / 2, rh])
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        P = (pts[:, None, :] + nrm[:, None, :] * offs[None, :, None]).reshape(-1, 2)
        L = layers(P).reshape(len(LAYERS), len(pts), len(offs))
        Z = L[-1]
        zc = Z[:, 2]
        km = float(inside.sum()) * STEP_M / 1000.0
        stats["km"] = stats.get("km", 0.0) + km
        dz = zc - zp
        for kind, m in (("haut", (dz > AXIS_M) & inside), ("bas", (dz < -AXIS_M) & inside)):
            for i0, i1 in runs(m, arc):
                emit(kind, p, arc, pts, dz, i0, i1, L=L, zp=zp, offs=offs)
        slope, res = across_fit(offs, Z)
        for i0, i1 in runs((res > ACROSS_M) & inside, arc):
            emit("travers", p, arc, pts, res, i0, i1, L=L, zp=zp, offs=offs)
        for i0, i1 in runs((np.abs(slope) > BANK_MAX) & inside, arc):
            emit("devers", p, arc, pts, slope, i0, i1, L=L, zp=zp, offs=offs)
        st = steps(arc, zc, zp) & inside
        for i in np.flatnonzero(st):
            emit("marche", p, arc, pts, np.r_[0.0, np.diff(zc) - np.diff(zp)], int(i), int(i), L=L, zp=zp,
                 offs=offs)
        stats.setdefault("travers_p95", []).append(float(np.percentile(res[inside], 95)))

    for (c, grp), r in zip(junctions, jr):
        if not (np.abs(c) <= half).all():
            continue
        rr = np.arange(-r, r + 1e-6, 1.0)
        DX, DY = np.meshgrid(rr, rr)
        keep = np.hypot(DX, DY) <= r
        disc = np.stack([c[0] + DX[keep], c[1] + DY[keep]], 1)
        zg = final(disc)
        slope, res = plane_residual(disc, zg)
        zn = []
        for k in grp:
            pl = road[k // 2]
            zn.append(float(np.asarray(pl.z, np.float64)[0 if k % 2 == 0 else -1]))
        spread = max(zn) - min(zn)
        stats["noeuds"] = stats.get("noeuds", 0) + 1
        if res > NODE_RES_M or spread > NODE_SPREAD_M:
            out.write(json.dumps({"kind": "carrefour", "at": ll(c), "value": round(res, 3), "spread": round(spread, 3),
                                  "slope": round(slope, 3), "roads": len({rids[k] for k in grp}), "radius": round(r, 1),
                                  "place": [round(lat, 5), round(lon, 5)]}) + "\n")
            stats["carrefour"] = stats.get("carrefour", 0) + 1

    # drawn roads with no plan
    m = 30.0 / 111320.0
    cla, clo, _ = frame.enu_to_geodetic(np.array([-half, half]), np.array([-half, half]), 0.0)
    lines = [np.asarray(p.xy, np.float64)[:, :2] for p in plans]
    for key in mapcells.cells_for_bbox(float(min(cla)) - m, float(min(clo)) - m, float(max(cla)) + m, float(max(clo)) + m):
        f = cell_fn(key)
        if f is None or not len(f):
            continue
        specs = roads.road_specs(f)
        for i in range(len(f)):
            sp = specs[i]
            if sp is None or sp.cls in grading.UNGRADED:
                continue
            for ring in roads.ground_parts(f, i):
                ring = np.asarray(ring, np.float64)
                e, n, _ = frame.geodetic_to_enu(ring[:, 1], ring[:, 0], 0.0)
                rxy = np.stack([np.asarray(e), np.asarray(n)], 1)
                if len(rxy) < 2:
                    continue
                arc, pts, _t = stations(rxy, NO_PLAN_STEP_M)
                ins = (np.abs(pts) <= half).all(1)
                if not ins.any():
                    continue
                far = far_from(pts, lines, NO_PLAN_M) & ins
                for i0, i1 in runs(far, arc, 2 * NO_PLAN_STEP_M):
                    out.write(json.dumps({"kind": "sans_plan", "at": ll(pts[i0]), "value": 0.0,
                                          "len": round(float(arc[i1] - arc[i0]) + NO_PLAN_STEP_M, 1), "cls": sp.cls,
                                          "fid": str(f.ids[i])[:16], "place": [round(lat, 5), round(lon, 5)]}) + "\n")
                    stats["sans_plan"] = stats.get("sans_plan", 0) + 1
    out.flush()


def report(out_dir):
    def rows(name):
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    rs, zs = rows("routes.jsonl"), rows("zones.jsonl")
    km = sum(z.get("km", 0.0) for z in zs)
    p95 = [v for z in zs for v in z.get("travers_p95", [])]
    kinds = ("haut", "bas", "travers", "devers", "marche", "carrefour", "sans_plan")
    lines = [f"# Banc routes au sol ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"{len(zs)} carrés de {PLACE_M:.0f} m, {km:.1f} km de chaussée mesurés, "
             f"{sum(z.get('noeuds', 0) for z in zs)} carrefours.", "",
             "| défaut | nombre | par km | longueur cumulée (m) |", "|---|---|---|---|"]
    for k in kinds:
        sel = [r for r in rs if r["kind"] == k]
        lines.append(f"| {k} | {len(sel)} | {len(sel) / max(km, 1e-9):.2f} | {sum(r.get('len', 0.0) for r in sel):.0f} |")
    if p95:
        lines += ["", f"Écart en travers (p95 par route) : médiane {np.median(p95):.3f} m, p90 {np.percentile(p95, 90):.3f} m."]
    caused = [r for r in rs if r.get("cause")]
    if caused:
        def ctx(r):
            if r.get("eau"):
                return "eau"
            if r.get("pont_m") is not None and r["pont_m"] <= 30:
                return "pont"
            if r.get("rail_m") is not None and r["rail_m"] <= 15:
                return "rail"
            if r.get("carrefour_m") is not None and r["carrefour_m"] <= 15:
                return "carrefour"
            return "ailleurs"
        causes = sorted({r["cause"] for r in caused})
        ctxs = ("eau", "pont", "rail", "carrefour", "ailleurs")
        for title, sel in (("tous", caused), ("> 0,5 m", [r for r in caused if abs(r["value"]) > 0.5])):
            lines += ["", f"### Cause × lieu ({title}, {len(sel)} défauts)", "",
                      "| cause | " + " | ".join(ctxs) + " | total |", "|---" * (len(ctxs) + 2) + "|"]
            for c in causes:
                n = [sum(1 for r in sel if r["cause"] == c and ctx(r) == x) for x in ctxs]
                lines.append(f"| {c} | " + " | ".join(map(str, n)) + f" | {sum(n)} |")
    titles = {"haut": "Sol au-dessus de la route (m)", "bas": "Sol sous la route (m)",
              "travers": "Bosse ou creux en travers (m)", "devers": "Dévers excessif (m/m)",
              "marche": "Marche le long de la route (m)", "carrefour": "Carrefours (écart au plan, m)",
              "sans_plan": "Routes sans nivellement"}
    for k in kinds:
        sel = sorted((r for r in rs if r["kind"] == k), key=lambda r: -abs(r["value"]) if k != "sans_plan"
                     else -r.get("len", 0.0))[:60]
        if not sel:
            continue
        lines += ["", f"## {titles[k]} ({sum(1 for r in rs if r['kind'] == k)})", ""]
        if k == "carrefour":
            lines += ["| écart | étalement des branches | pente | branches | où (lat, lon) |", "|---|---|---|---|---|"]
            lines += [f"| {r['value']:.2f} | {r['spread']:.2f} | {r['slope']:.1%} | {r['roads']} | {r['at'][0]}, {r['at'][1]} |"
                      for r in sel]
        elif k == "sans_plan":
            lines += ["| longueur (m) | classe | où (lat, lon) |", "|---|---|---|"]
            lines += [f"| {r['len']:.0f} | {r['cls']} | {r['at'][0]}, {r['at'][1]} |" for r in sel]
        else:
            lines += ["| valeur | longueur (m) | demi-chaussée (m) | cause | pont / rail (m) | où (lat, lon) |",
                      "|---|---|---|---|---|---|"]
            lines += [f"| {r['value']:+.2f} | {r['len']:.0f} | {r['half']:.1f} | {r.get('cause', '')}"
                      f"{' (eau)' if r.get('eau') else ''} | {r.get('pont_m')} / {r.get('rail_m')} | "
                      f"{r['at'][0]}, {r['at'][1]} |" for r in sel]
    with open(os.path.join(out_dir, "rapport.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    a = args_parse(argv)
    setup_env()
    import blockworker as bw
    os.makedirs(a.out, exist_ok=True)
    t0 = time.monotonic()
    bw.init(a.cache, False, True, True, True, True, multiprocessing.Queue(1), logging.ERROR, True, True)
    lat0, lat1 = sorted((a.lat0, a.lat1))
    lon0, lon1 = sorted((a.lon0, a.lon1))
    dla = PLACE_M / 111320.0
    dlo = dla / math.cos(math.radians((lat0 + lat1) / 2))
    places = [(lat0 + (i + 0.5) * dla, lon0 + (j + 0.5) * dlo)
              for i in range(max(1, int((lat1 - lat0) / dla))) for j in range(max(1, int((lon1 - lon0) / dlo)))]
    if a.places:
        places = places[:a.places]
    print(f"{len(places)} carrés de {PLACE_M:.0f} m", flush=True)
    with open(os.path.join(a.out, "routes.jsonl"), "a", encoding="utf-8") as out, \
            open(os.path.join(a.out, "zones.jsonl"), "a", encoding="utf-8") as zout:
        for k, (la, lo) in enumerate(places):
            if time.monotonic() - t0 > a.hours * 3600:
                print("temps écoulé", flush=True)
                break
            stats, t1 = {}, time.monotonic()
            try:
                measure_place(bw, la, lo, out, stats, aid=k + 1)
            except Exception as ex:
                stats["erreur"] = f"{type(ex).__name__}: {ex}"
            stats.update(place=[round(la, 5), round(lo, 5)], s=round(time.monotonic() - t1, 1))
            zout.write(json.dumps(stats) + "\n")
            zout.flush()
            print(f"{k + 1}/{len(places)} {la:.5f},{lo:.5f} {stats.get('km', 0):.1f} km {stats['s']} s"
                  + (f" ERREUR {stats['erreur']}" if "erreur" in stats else ""), flush=True)
            if k % 5 == 4:
                report(a.out)
    report(a.out)


if __name__ == "__main__":
    main()
