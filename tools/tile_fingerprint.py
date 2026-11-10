#!/usr/bin/env python3
"""Fingerprints of what the world server builds at reference places, to prove a refactor changes nothing.

For each place: its road tiles (with buildings, sidewalks, street furniture), its far building tiles and a terrain
block, built offline from the download cache into an empty folder, then the SHA-256 of every file written. Two runs
of the same code must give the same fingerprints (check that once with --twice); a refactor must too.

    python3 tools/tile_fingerprint.py --places 43.6491,1.4252 --out before.json
    python3 tools/tile_fingerprint.py --places 43.6491,1.4252 --out after.json --compare before.json

The places' map data must already be in the cache (the build is offline: a missing cell fails that job, which is
reported and fingerprinted as the error's type, so a refactor that fails differently still shows).
"""
import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

# file names written with the time or the process in them, or whole files of timings: left out of the fingerprint
VOLATILE_KEYS = ("started", "finished", "seconds", "stages", "elapsed", "time", "pid")


def _digest(path):
    """SHA-256 of a file; a JSON file's volatile keys (VOLATILE_KEYS, at any depth) dropped first."""
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".json"):
        try:
            doc = json.loads(data)
        except ValueError:
            return hashlib.sha256(data).hexdigest()

        def clean(v):
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items() if k not in VOLATILE_KEYS}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v
        data = json.dumps(clean(doc), sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def build_place(lat, lon, tiles, blocks, cache_dir, online=False):
    """{relative file path: sha256} of everything built at (lat, lon), and {job: error type} of the jobs that failed."""
    import blockworker
    import dem as demmod
    import earthd
    import geodesy
    out_dir = tempfile.mkdtemp(prefix="bnge_fp_")
    errors = {}

    def job(name, fn, *args):
        try:
            fn(*args)
        except Exception as e:      # noqa: BLE001 (a failure is part of the fingerprint)
            import traceback
            errors[name] = type(e).__name__
            print(f"{name}: {type(e).__name__}: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    try:
        blockworker.init(cache_dir, not online, True, True, True, True, queue.Queue(), 30, True, True)
        zoom = demmod.zoom_for(lat, 2.5)
        for tx in range(-tiles, tiles + 1):
            for ty in range(-tiles, tiles + 1):
                job(f"roads({tx},{ty})", blockworker.roads_tile, 1, lat, lon, zoom, tx, ty, out_dir, "/temp/x",
                    True, 0.0, True, False, True)
        for tx in range(-tiles, tiles + 1):
            job(f"far_buildings({tx},0)", blockworker.far_buildings_tile, 1, lat, lon, zoom, tx, 0, out_dir, "/temp/x")
        frame = geodesy.EnuFrame(lat, lon)
        q = earthd.survey_q(frame, blockworker._s.dem, 1024)
        for k in range(blocks):
            job(f"block({k})", blockworker.build, 1, lat, lon, zoom, q, 128.0 * k, 0.0, 512, out_dir, f"hm_{k}.png",
                f"hm_{k}.ter")
        files = {}
        for base, _, names in os.walk(out_dir):
            for n in names:
                p = os.path.join(base, n)
                files[os.path.relpath(p, out_dir).replace(os.sep, "/")] = _digest(p)
        return files, errors
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def run(places, tiles, blocks, cache_dir, online=False):
    """Each place built in a process of its own: the server's in-memory caches (far tiles, plans) would otherwise
    skip a second build's files; and with the revisit cache off (BNE_REVISIT, the plans' too), so that everything is
    built from the map data again rather than read back from an earlier run."""
    out = {}
    for lat, lon in places:
        t0 = time.perf_counter()
        got = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", f"{lat},{lon}", "--tiles", str(tiles),
                              "--blocks", str(blocks), "--cache-dir", cache_dir] + (["--online"] if online else []),
                             stdout=subprocess.PIPE, check=True, env=dict(os.environ, BNE_REVISIT="off")).stdout
        files, errors = json.loads(got.decode().strip().splitlines()[-1])
        total = hashlib.sha256(json.dumps([files, errors], sort_keys=True).encode()).hexdigest()
        out[f"{lat},{lon}"] = {"total": total, "files": files, "errors": errors}
        print(f"{lat},{lon}: {len(files)} files, {len(errors)} failed jobs, {total[:16]} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return out


def compare(a, b):
    """Lines describing every difference between two runs' fingerprints (none: identical)."""
    lines = []
    for place in sorted(set(a) | set(b)):
        pa, pb = a.get(place), b.get(place)
        if pa is None or pb is None:
            lines.append(f"{place}: only in {'the new run' if pa is None else 'the reference'}")
            continue
        if pa["total"] == pb["total"]:
            continue
        for f in sorted(set(pa["files"]) | set(pb["files"])):
            if pa["files"].get(f) != pb["files"].get(f):
                lines.append(f"{place}: {f} differs")
        for j in sorted(set(pa["errors"]) | set(pb["errors"])):
            if pa["errors"].get(j) != pb["errors"].get(j):
                lines.append(f"{place}: job {j}: {pa['errors'].get(j)} -> {pb['errors'].get(j)}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--places", default="", help="lat,lon[;lat,lon...] (areas whose map data is cached)")
    ap.add_argument("--tiles", type=int, default=1, help="road tiles from -n to n on each axis (256 m each)")
    ap.add_argument("--blocks", type=int, default=1, help="terrain blocks per place")
    ap.add_argument("--cache-dir", default=os.path.expanduser(os.path.join("~", ".cache", "beamng-earth")))
    ap.add_argument("--out", help="write the fingerprints here (JSON)")
    ap.add_argument("--compare", help="a reference run's JSON: list the differences, exit 1 if any")
    ap.add_argument("--twice", action="store_true", help="build twice and compare (the build must be deterministic)")
    ap.add_argument("--online", action="store_true", help="fetch what the cache lacks (a first run at new places)")
    ap.add_argument("--one", help=argparse.SUPPRESS)       # (internal: one place, its JSON on stdout)
    a = ap.parse_args()
    if a.one:
        lat, lon = (float(v) for v in a.one.split(","))
        print(json.dumps(build_place(lat, lon, a.tiles, a.blocks, a.cache_dir, a.online)))
        return
    places = [tuple(float(v) for v in p.split(",")) for p in a.places.split(";") if p.strip()]
    got = run(places, a.tiles, a.blocks, a.cache_dir, a.online)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(got, f, indent=1, sort_keys=True)
    ref = None
    if a.twice:
        ref = run(places, a.tiles, a.blocks, a.cache_dir)
    elif a.compare:
        with open(a.compare, encoding="utf-8") as f:
            ref = json.load(f)
    if ref is not None:
        diff = compare(ref, got)
        print("\n".join(diff) if diff else "identical")
        sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
