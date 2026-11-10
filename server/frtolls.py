"""French toll plazas: an island with its booth between the toll lanes, and a lifted barrier on each lane.

The user's report (2026-11-09): a French toll was only its canopy, a block driven under. OpenStreetMap maps each toll
lane of a plaza with a barrier=toll_booth node on that lane's way (an export bundled once: tools/build_fr_tolls.py);
the canopy is a building=roof, already drawn. Here the lanes of one plaza are put side by side across the traffic, an
island (West Coast USA's toll booth island, buildings.G_TOLLBOOTH, 1.56 m wide and 22 m long) stands midway between
two neighbouring lanes and one beyond each outer lane, and each lane has the Italian map's boom barrier on its right
island just past the booth, its arm raised so the lane stays open to drive through.
A plaza mapped with one node on a wide carriageway gets islands only beside the carriageway (its lanes are not known).
"""

from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import buildings
import furniture
import geotiles

DATA_PATH = Path(__file__).with_name("data") / "fr_toll_lanes.json.gz"
DATA_SHA256 = "93ec8695e0be0116a20b6a90dfacdc1e656f8fab7ee5ffd191eb09639f52925f"
SIGNATURE = "fr-tolls-1:" + DATA_SHA256
GRID = 1_000_000
G_GATE = "tollgate"         # trees.lua: the Italian map's boom barrier, arm raised (italy_bld_boomgate_4.5m)
GATE_H = 5.89               # m: that shape's height (its raised arm)
SNAP_M = 6.0                # m: a toll node is on its lane's way
PLAZA_M = 60.0              # m: lanes of one plaza are this near each other ...
PLAZA_COS = 0.9             # ... and run the same way
LANE_GAP_M = (4.6, 12.0)    # m between two neighbouring lanes' centres for an island between them (3 m lane + island)
DEFAULT_GAP_M = 5.5         # m: a lone lane's distance to its islands' centres (3.5 m lane, island and kerbs)
ISLAND_HALF_M = 0.78        # m: half the island's width
GATE_ALONG_M = 4.0          # m past the booth (its centre) along the traffic
MARGIN_M = 80.0             # m: the plaza's lanes looked up this far beyond the tile


@lru_cache(maxsize=1)
def _catalogue():
    raw = DATA_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("French toll lane checksum mismatch")
    doc = json.loads(gzip.decompress(raw))
    if doc["format"] != 1:
        raise ValueError("Unsupported French toll lane format")
    grid = defaultdict(list)
    for row in doc["rows"]:
        grid[row[0] // GRID, row[1] // GRID].append(row)
    return grid


def nearby(frame, box, margin=MARGIN_M):
    """The catalogue's rows [lon 1e7, lat 1e7] within the box grown by margin (anchor frame)."""
    x0, y0, x1, y1 = box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin
    lat, lon, _ = frame.enu_to_geodetic(np.array([x0, x0, x1, x1]), np.array([y0, y1, y0, y1]), np.zeros(4))
    a, b = math.floor(min(lon) * 1e7), math.floor(min(lat) * 1e7)
    c, d = math.ceil(max(lon) * 1e7), math.ceil(max(lat) * 1e7)
    grid = _catalogue()
    return [row for x in range(a // GRID, c // GRID + 1) for y in range(b // GRID, d // GRID + 1)
            for row in grid.get((x, y), ()) if a <= row[0] <= c and b <= row[1] <= d]


def _lanes(frame, roads, rows):
    """[(point (3,) on the lane's centre line, unit travel (2,), road width)] of the toll nodes on a paved road."""
    out = []
    for lon, lat in rows:
        at = np.asarray(frame.geodetic_to_enu(lat * 1e-7, lon * 1e-7, 0.), float).ravel()
        best = None
        for _, spec, line, _ in roads:
            if spec.kind != "paved" or len(line) < 2:
                continue
            s = furniture._project(line, at[None, :2])
            points, tangents = furniture._at(line, s)
            dist = float(np.hypot(*(at[:2] - points[0][:2])))
            if dist <= SNAP_M and (best is None or dist < best[0]):
                t = tangents[0] * (spec.dir if spec.oneway and spec.dir else 1)
                best = (dist, points[0], t, float(spec.width))
        if best is not None:
            out.append(best[1:])
    return out


def _plazas(lanes):
    """The lanes grouped by plaza (near each other, the same way)."""
    groups = []
    for lane in lanes:
        for g in groups:
            q, t = g[0][0], g[0][1]
            if float(np.hypot(*(lane[0][:2] - q[:2]))) <= PLAZA_M and float(lane[1] @ t) >= PLAZA_COS:
                g.append(lane)
                break
        else:
            groups.append([lane])
    return groups


def plaza_rows(lanes):
    """Prop rows [group, x, y, z, yaw, scale, height, rank] (anchor frame) of one plaza's islands and gates."""
    travel = np.mean([t for _, t, _ in lanes], axis=0)
    travel /= max(float(np.hypot(*travel)), 1e-9)
    right = np.array([travel[1], -travel[0]])
    ref = lanes[0][0][:2]
    lanes = sorted(lanes, key=lambda lane: float((lane[0][:2] - ref) @ right))
    across = [float((q[:2] - ref) @ right) for q, _, _ in lanes]
    gaps = [b - a for a, b in zip(across, across[1:])]
    usable = [g for g in gaps if LANE_GAP_M[0] <= g <= LANE_GAP_M[1]]
    if len(lanes) == 1:
        step = max(DEFAULT_GAP_M, lanes[0][2] / 2 + 1.0 + ISLAND_HALF_M) * 2
    else:
        step = float(np.median(usable)) if usable else 2 * DEFAULT_GAP_M
    yaw_booth = (math.degrees(math.atan2(travel[1], travel[0])) - 90.0) % 360.0      # the island's length: local y
    islands = []        # (across, along point (3,))
    for k, (q, _, _) in enumerate(lanes):
        if k == 0:
            islands.append((across[k] - step / 2, q))
        if k + 1 < len(lanes):
            g = gaps[k]
            if LANE_GAP_M[0] <= g <= LANE_GAP_M[1]:
                islands.append((across[k] + g / 2, (q + lanes[k + 1][0]) / 2))
        if k + 1 == len(lanes) or not LANE_GAP_M[0] <= gaps[k] <= LANE_GAP_M[1]:
            islands.append((across[k] + step / 2, q))
    rows = []
    for c, q in islands:
        along = float((q[:2] - ref) @ travel)
        p = ref + right * c + travel * along
        rows.append([buildings.G_TOLLBOOTH, round(float(p[0]), 2), round(float(p[1]), 2), round(float(q[2]), 2),
                     round(yaw_booth, 1), 1.0, buildings.TOLLBOOTH_H, 0.5])
    for (q, _, _), c in zip(lanes, across):
        right_island = min((ic for ic, _ in islands if ic > c), default=None)
        if right_island is None:
            continue
        p = ref + right * (right_island - ISLAND_HALF_M + 0.3) + travel * (float((q[:2] - ref) @ travel) + GATE_ALONG_M)
        rows.append([G_GATE, round(float(p[0]), 2), round(float(p[1]), 2), round(float(q[2]), 2),
                     round(math.degrees(math.atan2(travel[1], travel[0])) % 360.0, 1), 1.0, GATE_H, 0.5])
    return rows


def rows(frame, roads, box, data=None):
    """The prop rows of the toll plazas whose rows fall in box (each plaza laid out whole, its rows kept by owner)."""
    found = nearby(frame, box) if data is None else data
    if not found:
        return []
    out = []
    for plaza in _plazas(_lanes(frame, roads, found)):
        out.extend(r for r in plaza_rows(plaza) if geotiles.box_owns(box, r[1], r[2]))
    return out
