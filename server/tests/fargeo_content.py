"""fargeo_content.py OUT LAT LON BLAT BLON ANCHOR_LAT ANCHOR_LON: a helper process of test_fargeo.py. Builds the anchor
(ANCHOR_LAT, ANCHOR_LON)'s own far buildings / far bridge tiles under (LAT, LON) and (BLAT, BLON) first (its frames' chain
views and caches warm, as a session would have them), then the far geotiles: the far buildings tile and the skyline
square under (LAT, LON), the far bridge square under (BLAT, BLON), in their own frames; prints the files' sha1s (one
"name hash" line each). Map data from ~/.cache/beamng-earth, offline; the revisit cache off."""
import hashlib
import logging
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
logging.basicConfig(level=logging.WARNING)
import blockworker as bw  # noqa: E402
import geodesy  # noqa: E402
import geotiles  # noqa: E402
import revisit  # noqa: E402
import roads  # noqa: E402

out, lat, lon, blat, blon, alat, alon = sys.argv[1], *map(float, sys.argv[2:8])
revisit.ON = False
# (online: the place is fetched where this machine has not been; the tests that run this are opt-in, realdata.py)
bw._s = bw._State(os.path.expanduser("~/.cache/beamng-earth"), False, True, True, True, True, True, True)
os.makedirs(os.path.join(out, "anchor"), exist_ok=True)
f = geodesy.EnuFrame(alat, alon)
for la, lo in ((lat, lon), (blat, blon)):
    x, y, _ = f.geodetic_to_enu(la, lo, 0.0)
    tx, ty = int(math.floor(float(x) / roads.TILE)), int(math.floor(float(y) / roads.TILE))
    bw.far_buildings_tile(1, alat, alon, 15, tx, ty, os.path.join(out, "anchor"), "/x")
    bw.far_bridge_tile(1, alat, alon, 15, tx, ty, os.path.join(out, "anchor"), "/x")
for kind, la, lo in (("bldfar", lat, lon), ("bldsky", lat, lon), ("brfar", blat, blon)):
    gx, gy = geotiles.tile_of(la, lo, bw.far_geo_zoom(kind))
    # (far_geo answers None while its cells download and is asked again later, as the server's tracker does: since
    # 2026-11-07 it no longer blocks on a first visit's downloads)
    import time
    deadline = time.monotonic() + bw.FAR_CELLS_WAIT_S + 60.0
    while bw.far_geo(kind, gx, gy, 15, out, "/x") is None:
        assert time.monotonic() < deadline, kind
        time.sleep(1.0)
for n in sorted(os.listdir(out)):
    if n[:3] in ("bf_", "bs_", "bb_"):
        print(n, hashlib.sha1(open(os.path.join(out, n), "rb").read()).hexdigest())
