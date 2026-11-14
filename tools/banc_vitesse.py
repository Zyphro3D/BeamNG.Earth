#!/usr/bin/env python3
"""banc_vitesse.py: does the world keep up with a fast car? A straight drive at a set speed, the real server against
the mod's own terrain rules, without the game.

    cd server && python ../tools/banc_vitesse.py LAT LON --speed 69 --heading 330 --km 10 --out DIR [--cache DIR]

The world server (earthd.py) runs as the app's launcher starts it (launcher.RECOMMENDED, the fork's defaults:
launcher.FORK_DEFAULTS) on the cache given (--cache: a new empty folder for a first visit, the player's for a known
place). A client replays the mod frame by frame in real time with tests/modsim.py (follow.lua's terrain-follow, rebase.lua,
edgeguard.lua: every threshold read from the Lua files) and sends the 2 Hz poses the server prefetches by. When the edge
guard would pause the game (the car within PAUSE_LEAD_S of its block's stop line), the car stands still until the next
block is committed, as in the game.

Writes DIR/vitesse.json: the pauses (count, total s, the longest; pause_list: [s into the drive, m driven, s] each),
the blocks asked and their latency (median, max), the
blocks that came late, the closest the car came to an edge, and the server's log (DIR/server.log). Nothing else is
written outside DIR and the cache.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SERVER = Path.cwd()
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tests"))
sys.path.insert(0, str(SERVER.parent))


def args_parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lat", type=float)
    ap.add_argument("lon", type=float)
    ap.add_argument("--speed", type=float, default=69.4, help="m/s (69.4: 250 km/h)")
    ap.add_argument("--heading", type=float, default=0.0, help="degrees from east, counterclockwise (90: north)")
    ap.add_argument("--km", type=float, default=10.0)
    ap.add_argument("--wait", type=float, default=0.0,
                    help="s standing after the first block before driving off (a longer loading screen)")
    ap.add_argument("--accel", type=float, default=0.0,
                    help="m/s/s from standing to --speed (0: at --speed from the start; ~3: a quick car's full throttle)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="the server's cache folder (default: the player's)")
    ap.add_argument("--rebase-m", type=float, default=None, help="the rebase distance (default: the mod's)")
    return ap.parse_args(argv)


def main(argv=None):
    a = args_parse(argv)
    import geodesy
    import launcher
    import modsim
    from modsim import CORE, REBASE, cheb
    from netclient import Client, stop_server
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for k, v in launcher.FORK_DEFAULTS.items():
        env.setdefault(k, v)
    user = tempfile.mkdtemp(prefix="banc_vitesse_")
    log = open(out_dir / "server.log", "w", encoding="utf-8")
    cmd = [sys.executable, str(SERVER / "earthd.py"), "--port", "0", "--user-dir", user,
           # (a cache given: never trimmed, it may link to the player's downloads)
           *([] if a.cache is None else ["--cache-dir", a.cache, "--cache-cap", "0"]), *launcher.RECOMMENDED]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=str(SERVER))
    result = {"lat": a.lat, "lon": a.lon, "speed": a.speed, "heading": a.heading, "km": a.km,
              "cache": a.cache or "player's"}
    try:
        line = proc.stdout.readline()
        if "listening on 127.0.0.1:" not in line:
            raise RuntimeError(f"the server did not start: {line!r}")
        client = Client(int(line.split("127.0.0.1:")[1].split()[0]))
        mod_version = __import__("re").search(r"MOD_VERSION = '([^']+)'",
                                              (modsim.MOD_DIR / "core.lua").read_text()).group(1)
        client.send({"type": "hello", "protocol": int(CORE["PROTOCOL"]), "format": int(CORE["FORMAT"]),
                     "modVersion": mod_version})
        h, _ = client.until(lambda m: m["type"] == "hello", 120)
        size = int(h.get("blockSize") or 1024)
        t0 = time.monotonic()
        client.send({"type": "prepare_anchor", "requestId": 1, "lat": a.lat, "lon": a.lon, "cx": 0, "cy": 0})
        ready, _ = client.until(lambda m: m["type"] in ("anchor_ready", "anchor_error"), 600)
        if ready["type"] != "anchor_ready":
            raise RuntimeError(f"no anchor: {ready}")
        result["first_anchor_s"] = round(time.monotonic() - t0, 1)
        aid, frame = ready["anchorId"], geodesy.EnuFrame(a.lat, a.lon)
        client.send({"type": "commit_anchor", "anchorId": aid})
        client.send({"type": "commit_terrain", "anchorId": aid, "cx": 0, "cy": 0})
        result.update(drive(client, frame, aid, ready, size, a, modsim, REBASE, cheb))
    finally:
        try:
            stop_server(proc)
        finally:
            log.close()
            shutil.rmtree(user, ignore_errors=True)
    (out_dir / "vitesse.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


def drive(client, frame, aid, ready, size, a, modsim, REBASE, cheb):
    import geodesy
    DT = 1 / 60
    T = modsim.follow_table(size)
    rebase_m = a.rebase_m or float(REBASE["DEFAULT_DISTANCE"])
    st = {"block": (ready["terrain"]["cx"], ready["terrain"]["cy"]), "tp": None, "pending": None, "next": 2,
          "t_lat": None, "a_lat": None}
    out = {"pause_list": [], "pauses": 0, "paused_s": 0.0, "longest_pause_s": 0.0, "late": 0, "commits": 0,
           "rebases": 0,
           "prepares": 0, "latency": [], "edge_min": math.inf, "edge_at": None, "errors": []}
    hd = math.radians(a.heading)
    v = a.speed if a.accel <= 0 else 0.5         # m/s now (accelerating at --accel to --speed)
    vx, vy = v * math.cos(hd), v * math.sin(hd)
    x = y = 0.0
    driven = clock = 0.0
    paused_since = None
    guard = modsim.EdgeGuard()
    wall0 = time.monotonic()
    frames = 0

    def to_frame(src, dst, px, py):
        la, lo, h = src.enu_to_geodetic(px, py, 0.0)
        e, n, _ = dst.geodetic_to_enu(la, lo, h)
        return float(e), float(n)

    def handle(m):
        t = m["type"]
        if t in ("anchor_error", "terrain_error"):
            out["errors"].append(t)
        elif t == "terrain_ready":
            p = st["tp"]
            if p and m["requestId"] == p["requestId"] and m["anchorId"] == aid:
                p["ready"], p["msg"] = True, m
                st["t_lat"] = clock - p["sent"]
                out["latency"].append(round(st["t_lat"], 2))
                if p["on_time"] and cheb(x, y, *st["block"]) > T["COMMIT_DIST"]:
                    out["late"] += 1
        elif t == "anchor_ready":
            p = st["pending"]
            if p and m["requestId"] == p["requestId"]:
                p["ready"], p["msg"] = True, m
                st["a_lat"] = clock - p["sent"]

    while driven < a.km * 1000.0:
        frames += 1
        clock += DT
        while True:                             # real time: the server's latency is the real one
            try:
                handle(client.recv(max(0.0005, wall0 + clock - time.monotonic())))
            except TimeoutError:
                break
            if time.monotonic() >= wall0 + clock:
                break
        box = modsim.ground_box(*st["block"], size)
        mode, _v, _put, pause = guard.step(clock, x, y, 0.0, vx, vy, box, 0.0)
        if pause or (paused_since is not None and mode == "hold"):
            if paused_since is None:
                paused_since = clock
                out["pauses"] += 1
        elif paused_since is not None:
            dur = clock - paused_since
            out["pause_list"].append([round(paused_since, 1), round(driven), round(dur, 1)])
            out["paused_s"] += dur
            out["longest_pause_s"] = max(out["longest_pause_s"], dur)
            paused_since = None
        if clock < a.wait:                      # (still standing: a longer loading screen)
            pass
        elif paused_since is None:
            if v < a.speed:
                k = min(a.speed, v + a.accel * DT) / v
                v, vx, vy = v * k, vx * k, vy * k
            x, y = x + vx * DT, y + vy * DT
            driven += v * DT
        e = min(x - box[0], box[2] - x, y - box[1], box[3] - y)
        if e < out["edge_min"]:
            out["edge_min"], out["edge_at"] = e, (round(clock, 2), round(driven))
        speed_now = 0.0 if paused_since is not None or clock < a.wait else v
        # origin rebase first (core.lua updateWorld)
        p = st["pending"]
        stale = edge = None
        if p and p.get("ready"):
            m = p["msg"]
            nf = geodesy.EnuFrame(m["lat"], m["lon"])
            nx, ny = to_frame(frame, nf, x, y)
            stale = cheb(nx, ny, m["terrain"]["cx"], m["terrain"]["cy"])
            gx0, gy0, gx1, gy1 = modsim.ground_box(m["terrain"]["cx"], m["terrain"]["cy"], size)
            edge = min(nx - gx0, gx1 - nx, ny - gy0, gy1 - ny)
        act = modsim.rebase_decide(math.hypot(x, y), p, stale, rebase_m, edge=edge, size=size)
        if act == "prefetch":
            lat, lon, _ = frame.enu_to_geodetic(x, y, 0.0)
            cx, cy = modsim.rebase_hint(vx, vy, math.hypot(x, y), rebase_m, size, st["a_lat"])
            st["pending"] = {"requestId": st["next"], "purpose": "rebase", "sent": clock}
            st["next"] += 1
            client.send({"type": "prepare_anchor", "requestId": st["pending"]["requestId"], "lat": float(lat),
                         "lon": float(lon), "cx": cx, "cy": cy})
        elif act == "commit":
            m = st["pending"]["msg"]
            nf = geodesy.EnuFrame(m["lat"], m["lon"])
            ex, ey = to_frame(frame, nf, x + vx, y + vy)
            x, y = to_frame(frame, nf, x, y)
            vx, vy = ex - x, ey - y
            frame, aid = nf, m["anchorId"]
            st["block"], st["tp"], st["pending"] = (m["terrain"]["cx"], m["terrain"]["cy"]), None, None
            client.send({"type": "commit_anchor", "anchorId": aid})
            client.send({"type": "commit_terrain", "anchorId": aid, "cx": st["block"][0], "cy": st["block"][1]})
            guard.new_frame()
            out["rebases"] += 1
        elif act == "discard":
            st["pending"] = None
        if act != "commit":
            b, tp = st["block"], st["tp"]
            fa = modsim.follow_decide(x, y, vx, vy, b[0], b[1], tp, clock, 0.0, None, st["t_lat"], size)
            if fa and fa[0] == "prepare":
                st["tp"] = {"requestId": st["next"], "cx": fa[1], "cy": fa[2], "sent": clock,
                            "on_time": cheb(x, y, *b) <= T["COMMIT_DIST"]}
                st["next"] += 1
                out["prepares"] += 1
                client.send({"type": "prepare_terrain", "requestId": st["tp"]["requestId"], "anchorId": aid,
                             "cx": fa[1], "cy": fa[2]})
            elif fa and fa[0] == "commit":
                st["block"], st["tp"] = (tp["cx"], tp["cy"]), None
                out["commits"] += 1
                client.send({"type": "commit_terrain", "anchorId": aid, "cx": tp["cx"], "cy": tp["cy"]})
            elif fa and fa[0] == "discard":
                st["tp"] = None
        if frames % 30 == 0:                    # poses at 2 Hz: the server's prefetch
            lat, lon, _ = frame.enu_to_geodetic(x, y, 0.0)
            client.send({"type": "pose", "anchorId": aid, "x": x, "y": y, "z": 0.0,
                         "vx": vx if speed_now else 0.0, "vy": vy if speed_now else 0.0,
                         "lat": float(lat), "lon": float(lon), "speed": speed_now})
        if clock > 3 * a.km * 1000.0 / a.speed + 600 + a.wait:      # (stuck: a wall-time bound)
            out["errors"].append("timeout")
            break
    if paused_since is not None:
        out["paused_s"] += clock - paused_since
        out["longest_pause_s"] = max(out["longest_pause_s"], clock - paused_since)
    lat = sorted(out.pop("latency"))
    out["blocks"] = len(lat)
    out["latency_median_s"] = lat[len(lat) // 2] if lat else None
    out["latency_max_s"] = lat[-1] if lat else None
    out["drive_s"] = round(clock, 1)
    out["paused_s"] = round(out["paused_s"], 1)
    out["longest_pause_s"] = round(out["longest_pause_s"], 1)
    out["edge_min"] = round(out["edge_min"], 1)
    return out


if __name__ == "__main__":
    main()
