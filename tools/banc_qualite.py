#!/usr/bin/env python3
"""banc_qualite.py: a night's survey of a zone's bridges and motorways, the defects ranked in a report.

    cd server && python ../tools/banc_qualite.py LAT0 LON0 LAT1 LON1 --out DIR [--hours 8] [--country FR]

Run from server/ (it imports the server's modules), with the switches the app runs with (the fork's launcher defaults:
launcher.FORK_DEFAULTS, applied here too). Reads the download cache (--cache, default ~/.cache/beamng-earth) and fetches
what is missing, as the server does. Nothing is written outside --out.

Per bridge deck (every deck of the map data in the box, grouped into places ~PLACE_M across, one anchor each):
  over      the graded ground over the deck's surface: its worst (m) and how many samples stand over it by more than
            OVER_M, over the deck's whole length (every STEP_M) and width (every 0.5 m), as the game draws them
  step      at each free end (an abutment: the ground within END_GROUND_M of the deck there), the ground's height at the
            end minus the deck's: the step a car meets between the road and the bridge
Per motorway stretch (every BOX_M box a motorway crosses): the median barrier modules (medians.modules, the country's
style) against the length of its divided carriageways (two one-way carriageways of opposite travel within
medians.MAX_GAP_M), a box with such a pair and no module listed.

Writes DIR/bridges.jsonl and DIR/motorways.jsonl a line at a time (a stopped run keeps what it measured), then
DIR/rapport.md: the worst first. One process, one place at a time, at the priority it was started with.
"""
import argparse
import json
import logging
import math
import multiprocessing
import os
import sys
import time

PLACE_M = 400.0         # m: bridges within this of a place's first are measured from its anchor
STEP_M = 2.0            # m along a deck between samples
OVER_M = 0.05           # m: ground over a deck by more than this is a defect sample
END_GROUND_M = 3.0      # m: an end whose ground is within this of the deck is an abutment (else a joint, a high end)
STEP_REPORT_M = 0.15    # m: a step at an abutment past this is listed
BOX_M = 256.0           # m: a motorway stretch's box (a road tile)


def args_parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lat0", type=float)
    ap.add_argument("lon0", type=float)
    ap.add_argument("lat1", type=float)
    ap.add_argument("lon1", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=8.0, help="stop starting new places after this long")
    ap.add_argument("--country", default="FR", help="the median style's country (ISO alpha-2)")
    ap.add_argument("--cache", default=os.path.expanduser(os.path.join("~", ".cache", "beamng-earth")))
    ap.add_argument("--no-bridges", action="store_true")
    ap.add_argument("--no-motorways", action="store_true")
    return ap.parse_args(argv)


def setup_env():
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.getcwd()))
    try:
        import launcher
        launcher.apply_fork_defaults()
    except Exception as ex:             # (the server's own defaults then)
        print("launcher defaults:", type(ex).__name__, ex, flush=True)
    os.environ.setdefault("BNE_REVISIT", "off")     # (measure the current code, not a cached tile)


def discover(md, box):
    """([(lat, lon, fid)] bridge parts' middles, [(lat, lon)] motorway points every ~BOX_M) in box (lat0, lon0, lat1,
    lon1), from the map data's cells."""
    import numpy as np
    import mapcells
    import roads
    lat0, lon0, lat1, lon1 = box
    bridges, motorways, seen = [], [], set()
    for key in mapcells.cells_for_bbox(lat0, lon0, lat1, lon1):
        try:
            f = md.cell(key, layers=("roads",), kind="demand").layers.get("roads")
        except Exception as ex:
            print("cell", key, type(ex).__name__, ex, flush=True)
            continue
        if f is None or not len(f):
            continue
        bs = roads.bridge_specs(f)
        for i in range(len(f)):
            fid = str(f.ids[i])
            if bs[i] is not None:
                for j, ring in enumerate(roads.bridge_parts(f, i)):
                    ring = np.asarray(ring, np.float64)
                    m = ring[len(ring) // 2]
                    k = (fid, j)
                    if k not in seen and lat0 <= m[1] <= lat1 and lon0 <= m[0] <= lon1:
                        seen.add(k)
                        bridges.append((float(m[1]), float(m[0]), fid))
            if str(f.attrs["class"][i]) == "motorway":
                a = f.coord_off[f.ring_off[f.part_off[i]]]
                b = f.coord_off[f.ring_off[f.part_off[i + 1]]]
                ll = np.asarray(f.lonlat[a:b], np.float64)
                for p in ll[:: max(1, len(ll) // 4)]:
                    if lat0 <= p[1] <= lat1 and lon0 <= p[0] <= lon1:
                        motorways.append((float(p[1]), float(p[0])))
    return bridges, motorways


def cluster(points, reach_m):
    """Greedy groups of (lat, lon, ...) within reach_m of each group's first: [(lat, lon, [items])]."""
    out = []
    for p in sorted(points):
        for g in out:
            dy = (p[0] - g[0]) * 111320.0
            dx = (p[1] - g[1]) * 111320.0 * math.cos(math.radians(g[0]))
            if dx * dx + dy * dy <= reach_m * reach_m:
                g[2].append(p)
                break
        else:
            out.append((p[0], p[1], [p]))
    return out


def measure_place(bw, lat, lon, fids, out):
    """The decks of fids near (lat, lon): one JSON line each into out."""
    import numpy as np
    import geodesy
    import grading
    import roads
    s = bw._s
    g, _ = s.anchor(1, lat, lon)
    frame = geodesy.EnuFrame(lat, lon)
    t = bw.plan_ahead(g, frame, 0.0, 0.0, PLACE_M + 200.0)
    if t is not None:
        t.join()

    def final(xy):
        la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        z0 = np.asarray(bw.ground(frame, la, lo, np.asarray(s.dem.sample(la, lo, 15), float)), float)
        z = np.asarray(g.apply(la, lo, z0.copy(), grading.ROAD_LIFT), float)
        return np.asarray(frame.geodetic_to_enu(la, lo, z)[2], float)

    def elev(la, lo):
        z = bw.ground(frame, la, lo, s.dem.sample(la, lo, 15))
        return g.apply(la, lo, z, grading.ROAD_LIFT)

    bridges = bw._bridges(1, frame, elev)
    decks = {}
    for tx, ty in roads.tiles_near(0.0, 0.0, PLACE_M):
        feats = [s.md.cell(k, layers=("roads",), kind="demand").layers.get("roads")
                 for k in roads.cells_of_tile(frame, tx, ty)]
        for fid, spec, xy, z, _ in bridges.parts(feats, roads.tile_bbox(tx, ty)):
            if str(fid) in fids:
                xy = np.asarray(xy, float)[:, :2]
                decks.setdefault((str(fid), round(float(xy[0, 0]), 1), round(float(xy[0, 1]), 1)),
                                 (xy, np.asarray(z, float), spec))
    for (fid, *_), (xy, z, spec) in sorted(decks.items()):
        arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        L = float(arc[-1])
        if L < 2.0 or not np.isfinite(z).all():
            continue
        a = np.arange(0.0, L + 1e-6, STEP_M)
        px, py, pz = np.interp(a, arc, xy[:, 0]), np.interp(a, arc, xy[:, 1]), np.interp(a, arc, z)
        dx, dy = np.gradient(px), np.gradient(py)
        n = np.hypot(dx, dy) + 1e-9
        nx, ny = -dy / n, dx / n
        offs = np.arange(-spec.width / 2, spec.width / 2 + 1e-6, 0.5)
        P = np.stack([(px[:, None] + nx[:, None] * offs[None]).ravel(),
                      (py[:, None] + ny[:, None] * offs[None]).ravel()], 1)
        over = final(P).reshape(len(a), len(offs)) - pz[:, None]
        i, j = np.unravel_index(int(np.argmax(over)), over.shape)
        wla, wlo, _ = frame.enu_to_geodetic(np.array([px[i]]), np.array([py[i]]), 0.0)
        steps = []
        for k in (0, -1):
            zg = float(final(xy[[k]])[0])
            d = zg - float(z[k])
            if abs(d) <= END_GROUND_M:
                ela, elo, _ = frame.enu_to_geodetic(np.array([xy[k, 0]]), np.array([xy[k, 1]]), 0.0)
                steps.append({"end": "start" if k == 0 else "end", "step": round(d, 2),
                              "at": [round(float(ela[0]), 6), round(float(elo[0]), 6)]})
        row = {"fid": fid, "cls": spec.cls, "len": round(L, 1), "width": round(float(spec.width), 1),
               "over_max": round(float(over.max()), 2), "over_n": int((over > OVER_M).sum()),
               "over_share": round(float((over > OVER_M).mean()), 3),
               "over_at": [round(float(wla[0]), 6), round(float(wlo[0]), 6)], "over_along": round(float(a[i]), 1),
               "over_across": round(float(offs[j]), 1), "steps": steps}
        out.write(json.dumps(row) + "\n")
        out.flush()


def measure_motorway(bw, lat, lon, country, out, done):
    """The median modules of the BOX_M box at (lat, lon) (once per box: done)."""
    import numpy as np
    import geodesy
    import medians
    import roads
    s = bw._s
    frame = geodesy.EnuFrame(lat, lon)
    box = (-BOX_M / 2, -BOX_M / 2, BOX_M / 2, BOX_M / 2)
    key = (round(lat * 400), round(lon * 300))          # (~250 m cells: a box measured once)
    if key in done:
        return
    done.add(key)
    near, seen = [], set()
    for tx, ty in roads.tiles_near(0.0, 0.0, BOX_M):
        for k in roads.cells_of_tile(frame, tx, ty):
            f = s.md.cell(k, layers=("roads",), kind="demand").layers.get("roads")
            if f is None:
                continue
            specs = roads.road_specs(f)
            for i in range(len(f)):
                fid = str(f.ids[i])
                if fid in seen or specs[i] is None:
                    continue
                seen.add(fid)
                for ring in roads.ground_parts(f, i):
                    e, nn, _ = frame.geodetic_to_enu(ring[:, 1], ring[:, 0], 0.0)
                    xy = np.stack([np.asarray(e), np.asarray(nn), np.zeros(len(ring))], 1)
                    if len(xy) >= 2:
                        near.append((fid, specs[i], xy, None))
    mods = medians.modules(near, box, False, country, (), ())
    # the divided carriageways' length in the box, and whether both travel directions are there
    length, dirs = 0.0, []
    for fid, spec, xy, _ in near:
        if not (medians.flags_of(spec, False, country) & 1):
            continue
        inside = (xy[:, 0] >= box[0]) & (xy[:, 0] <= box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] <= box[3])
        if inside.sum() < 2:
            continue
        seg = np.hypot(*np.diff(xy[inside][:, :2], axis=0).T)
        length += float(seg.sum())
        v = (xy[inside][-1, :2] - xy[inside][0, :2]) * (spec.dir or 1)
        dirs.append(v / max(float(np.hypot(*v)), 1e-9))
    pair = any(float(a @ b) < -0.5 for a in dirs for b in dirs)
    row = {"at": [round(lat, 6), round(lon, 6)], "divided_m": round(length, 1), "pair": pair,
           "modules": int(len(mods)), "cover": round(len(mods) * medians.MODULE_M / max(length / 2, 1.0), 2)}
    out.write(json.dumps(row) + "\n")
    out.flush()


def report(out_dir):
    """rapport.md from the JSON lines: the worst first."""
    def rows(name):
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    br, mw = rows("bridges.jsonl"), rows("motorways.jsonl")
    lines = [f"# Banc qualité ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"{len(br)} tabliers mesurés, {len(mw)} tronçons d'autoroute.", ""]
    over = sorted((r for r in br if r["over_max"] > OVER_M), key=lambda r: -r["over_max"])
    lines += [f"## Sol au-dessus du tablier ({len(over)} tabliers)", "",
              "| sol - tablier (m) | points | part | classe | longueur | où (lat, lon) | le long / en travers (m) |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['over_max']:+.2f} | {r['over_n']} | {r['over_share']:.0%} | {r['cls']} | {r['len']:.0f} | "
              f"{r['over_at'][0]}, {r['over_at'][1]} | {r['over_along']} / {r['over_across']} |" for r in over[:150]]
    steps = sorted(((s_["step"], s_, r) for r in br for s_ in r["steps"] if abs(s_["step"]) > STEP_REPORT_M),
                   key=lambda t: -abs(t[0]))
    lines += ["", f"## Marches au bout d'un pont ({len(steps)})", "",
              "| marche (m) | bout | classe | où (lat, lon) |", "|---|---|---|---|"]
    lines += [f"| {d:+.2f} | {s_['end']} | {r['cls']} | {s_['at'][0]}, {s_['at'][1]} |" for d, s_, r in steps[:150]]
    bare = sorted((r for r in mw if r["pair"] and r["cover"] < 0.5), key=lambda r: (r["cover"], -r["divided_m"]))
    lines += ["", f"## Autoroutes à chaussées séparées sans terre-plein ({len(bare)} tronçons sur "
              f"{sum(1 for r in mw if r['pair'])})", "",
              "| couverture | chaussées (m) | éléments | où (lat, lon) |", "|---|---|---|---|"]
    lines += [f"| {r['cover']:.0%} | {r['divided_m']:.0f} | {r['modules']} | {r['at'][0]}, {r['at'][1]} |"
              for r in bare[:150]]
    with open(os.path.join(out_dir, "rapport.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    a = args_parse(argv)
    setup_env()
    import blockworker as bw
    os.makedirs(a.out, exist_ok=True)
    t0 = time.monotonic()
    bw.init(a.cache, False, True, True, True, True, multiprocessing.Queue(1), logging.ERROR, True, True)
    box = (min(a.lat0, a.lat1), min(a.lon0, a.lon1), max(a.lat0, a.lat1), max(a.lon0, a.lon1))
    bridges, motorways = discover(bw._s.md, box)
    places = cluster(bridges, PLACE_M)
    print(f"{len(bridges)} parties de pont en {len(places)} lieux, {len(motorways)} points d'autoroute", flush=True)
    progress = os.path.join(a.out, "avancement.txt")

    def note(msg):
        with open(progress, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    if not a.no_motorways:
        done = set()
        with open(os.path.join(a.out, "motorways.jsonl"), "a", encoding="utf-8") as out:
            for k, (la, lo) in enumerate(motorways):
                if time.monotonic() - t0 > a.hours * 3600:
                    note("temps écoulé: autoroutes arrêtées")
                    break
                try:
                    measure_motorway(bw, la, lo, a.country, out, done)
                except Exception as ex:
                    note(f"autoroute {la:.5f},{lo:.5f}: {type(ex).__name__}: {ex}")
                if k % 50 == 0:
                    note(f"autoroutes {k}/{len(motorways)}")
        report(a.out)
    if not a.no_bridges:
        with open(os.path.join(a.out, "bridges.jsonl"), "a", encoding="utf-8") as out:
            for k, (la, lo, items) in enumerate(places):
                if time.monotonic() - t0 > a.hours * 3600:
                    note("temps écoulé: ponts arrêtés")
                    break
                try:
                    measure_place(bw, la, lo, {it[2] for it in items}, out)
                except Exception as ex:
                    note(f"lieu {la:.5f},{lo:.5f}: {type(ex).__name__}: {ex}")
                note(f"ponts: lieu {k + 1}/{len(places)}")
                if k % 10 == 9:
                    report(a.out)
    report(a.out)
    note("fini")


if __name__ == "__main__":
    main()
