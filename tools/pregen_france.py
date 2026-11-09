#!/usr/bin/env python3
"""Generates a French region's IGN data ahead of play, so a first visit has its walls, hedges, widths and roofs.

The game builds these on demand in the background; a tile made before they are ready goes without. This fills the
cache beforehand: a cell done is never done again, so it can be stopped and started again at will. What it keeps is
small (the LiDAR is read, worked out and thrown away), so a region can be shared as a zip: --pack writes one, --unpack
puts one into the cache.

    python tools/pregen_france.py --centre 43.70 1.38 --km 10 --estimate
    python tools/pregen_france.py --bbox 42.69,0.44,43.92,2.05 --pack haute-garonne.zip
    python tools/pregen_france.py --unpack haute-garonne.zip
"""
import argparse
import concurrent.futures as cf
import math
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))

MB_PER_CELL = 70.0          # at most, per cell (the LiDAR round its buildings and the photo)
MAX_CELLS = 200000
RETRY_PAUSE_S = 90.0
FOLDERS = ("ign_bati", "ign_largeur", "ign_limites")


def default_cache():
    return os.path.join(os.path.expanduser("~"), ".cache", "beamng-earth")


def box_of(a):
    if a.bbox:
        s, w, n, e = (float(v) for v in a.bbox.split(","))
    else:
        lat, lon = a.centre
        dlat, dlon = a.km / 111.0, a.km / (111.0 * math.cos(math.radians(lat)))
        s, w, n, e = lat - dlat, lon - dlon, lat + dlat, lon + dlon
    if not (s < n and w < e):
        raise SystemExit("the box must be south,west,north,east")
    return s, w, n, e


def cell_files(cache, key):
    return (os.path.join(cache, "ign_limites", "v1", str(key.z), str(key.x), f"{key.y}.npz"),)


def pending(cells, cache):
    return [k for k in cells if not all(os.path.exists(p) for p in cell_files(cache, k))]


def build_one(md, key):
    """The cell's data, each from its own provider: on disk afterwards."""
    if md.ign_bati is not None:
        md.ign_bati.rows(key)
    if getattr(md, "ign_largeur", None) is not None:
        md.ign_largeur.rows(key)
    md.ign_limites.segments(key)


def run(cells, build, workers, sink=print, pause=RETRY_PAUSE_S):
    """Calls build(key) for each cell, workers at a time; the cells that fail are tried once more at the end, one at a
    time after a pause (a service that refuses is what makes them fail). Returns the cells that still failed."""
    def pass_over(todo, label):
        failed, t0, done = [], time.time(), 0
        with cf.ThreadPoolExecutor(workers) as pool:
            jobs = {pool.submit(build, k): k for k in todo}
            for job in cf.as_completed(jobs):
                done += 1
                err = job.exception()
                if err is not None:
                    failed.append(jobs[job])
                    sink(f"  {jobs[job]}: {type(err).__name__}: {err}")
                if done == len(todo) or done % max(1, workers) == 0:
                    rate = done / max(time.time() - t0, 1e-6)
                    sink(f"{label} {done}/{len(todo)}, {rate * 60:.1f} cells a minute, "
                         f"{(len(todo) - done) / max(rate, 1e-6) / 3600:.1f} h left")
        return failed
    failed = pass_over(cells, "cells") if cells else []
    if failed:
        sink(f"{len(failed)} failed: again in {pause:.0f} s, one at a time")
        time.sleep(pause)
        workers = 1
        failed = pass_over(failed, "again")
    return failed


def pack(cache, out, cells=None):
    """A zip of the IGN folders, paths relative to the cache. cells: only these. Returns the files written."""
    wanted = None if cells is None else {(str(k.x), str(k.y)) for k in cells}
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for folder in FOLDERS:
            root = os.path.join(cache, folder)
            for here, _dirs, files in os.walk(root):
                for name in files:
                    if name.endswith((".claim", ".tmp", ".tmp.npz")):
                        continue
                    y = name.split(".")[0]
                    if wanted is not None and (os.path.basename(here), y) not in wanted:
                        continue
                    path = os.path.join(here, name)
                    z.write(path, os.path.relpath(path, cache).replace(os.sep, "/"))
                    n += 1
    return n


def unpack(archive, cache):
    """Puts a pack into the cache: only its known folders, and nothing outside them. Returns the files written."""
    root = os.path.realpath(cache)
    n = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            target = os.path.realpath(os.path.join(root, *name.split("/")))
            if info.is_dir() or name.split("/")[0] not in FOLDERS or not target.startswith(root + os.sep):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target + ".part", "wb") as dst:
                dst.write(src.read())
            os.replace(target + ".part", target)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", help="south,west,north,east in degrees")
    ap.add_argument("--centre", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--km", type=float, default=10.0, help="half side round --centre")
    ap.add_argument("--workers", type=int, default=1, help="cells at a time (each already reads 4 blocks at once)")
    ap.add_argument("--cache", default=default_cache())
    ap.add_argument("--estimate", action="store_true", help="say what it would do, nothing else")
    ap.add_argument("--limit", type=int, help="at most this many cells (a first trial)")
    ap.add_argument("--pack", help="then write this zip of the region")
    ap.add_argument("--unpack", help="put this zip into the cache and stop")
    a = ap.parse_args()
    if a.unpack:
        print(f"{unpack(a.unpack, a.cache)} files into {a.cache}")
        return
    if not a.bbox and not a.centre:
        ap.error("--bbox or --centre")
    import mapcells
    s, w, n, e = box_of(a)
    cells = list(mapcells.cells_for_bbox(s, w, n, e, max_cells=MAX_CELLS))
    todo = pending(cells, a.cache)
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(cells)} cells, {len(todo)} to do, at most {len(todo) * MB_PER_CELL / 1024:.1f} GB to download")
    if a.estimate:
        return
    import mapdata
    md = mapdata.MapData(a.cache)
    try:
        failed = run(todo, lambda k: build_one(md, k), a.workers)
    finally:
        md.close()
    print(f"done, {len(failed)} failed" + (": run it again for them" if failed else ""))
    if a.pack:
        print(f"{pack(a.cache, a.pack, cells)} files in {a.pack}")


if __name__ == "__main__":
    main()
