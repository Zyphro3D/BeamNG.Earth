"""French fixed speed cameras: a grey cabin on its post beside the road, or in the median between two carriageways.

The positions are OpenStreetMap's (ODbL, an export bundled once: tools/build_fr_speed_cameras.py). A camera looks
at the traffic coming towards it, so the cabin's dark windows face against the enforced direction. That direction is
the tag's heading when it has one (about 40 %); otherwise the side of the road the node was mapped on, and on a
two-way road mapped on its centre line, the right of the way's drawn direction (France drives on the right).
The mount type (post, wall, gantry) is almost never tagged: a post, or a column where tagged "tourelle".
"""

from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import frmotorwaymarkers
import frroadmesh
import furniture
import geotiles
import signplacement

DATA_PATH = Path(__file__).with_name("data") / "fr_speed_cameras.json.gz"
DATA_SHA256 = "e435eb8af8ebe3dccfb456f261e15caf5052feb3f0bb0aff77d7d55b5a8c4e13"
SIGNATURE = "fr-speed-cameras-1:" + DATA_SHA256
GRID = 1_000_000
SNAP_M = 30.0               # m: a camera this far from any paved road is not placed
HEADING_COS = 0.7           # the tagged heading within ~45 degrees of the road's direction
OFF_CENTRE_M = 1.0          # m: a node this far off the centre line was mapped on that side
CLEARANCE_M = (1.0, 1.5, 2.0, 2.7, 3.5)     # m beyond the pavement's edge, the first that is free
CABIN_W = 0.7               # m: the cabin's width across the road (the span kept free)
MAX_DROP_M = 2.5            # m: the ground at the post this far from the road's height: no post (an embankment)


@lru_cache(maxsize=1)
def _catalogue():
    raw = DATA_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("French speed camera checksum mismatch")
    doc = json.loads(gzip.decompress(raw))
    if doc["format"] != 1:
        raise ValueError("Unsupported French speed camera format")
    grid = defaultdict(list)
    for row in doc["rows"]:
        grid[row[0] // GRID, row[1] // GRID].append(row)
    return grid


def nearby(frame, box):
    """The catalogue's rows [lon 1e7, lat 1e7, heading, tower] within the box (anchor frame)."""
    x0, y0, x1, y1 = box
    lat, lon, _ = frame.enu_to_geodetic(np.array([x0, x0, x1, x1]), np.array([y0, y1, y0, y1]), np.zeros(4))
    a, b = math.floor(min(lon) * 1e7), math.floor(min(lat) * 1e7)
    c, d = math.ceil(max(lon) * 1e7), math.ceil(max(lat) * 1e7)
    grid = _catalogue()
    return [row for x in range(a // GRID, c // GRID + 1) for y in range(b // GRID, d // GRID + 1)
            for row in grid.get((x, y), ()) if a <= row[0] <= c and b <= row[1] <= d]


def _nearest(roads, at):
    """[(distance, road, point on its centre line (3,), unit tangent (2,))] of the paved roads within SNAP_M."""
    out = []
    for road in roads:
        _, spec, line, _ = road
        if spec.kind != "paved" or len(line) < 2:
            continue
        s = furniture._project(line, at[None, :2])
        points, tangents = furniture._at(line, s)
        q, t = points[0], tangents[0]
        dist = float(np.hypot(*(at[:2] - q[:2])))
        if dist <= SNAP_M:
            out.append((dist, road, q, t))
    out.sort(key=lambda r: r[0])
    return out


def travel_of(near, at, heading):
    """(road, point, unit direction of the enforced traffic) or None. near: _nearest's list."""
    if heading is not None and heading >= 0:
        h = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
        for _, road, q, t in near:
            spec = road[1]
            c = float(t @ h)
            if abs(c) < HEADING_COS:
                continue
            if spec.oneway and spec.dir and c * spec.dir < 0:
                continue            # this carriageway runs the other way: the camera watches the next one
            return road, q, t if c > 0 else -t
        return None
    if not near:
        return None
    _, road, q, t = near[0]
    spec = road[1]
    if spec.oneway and spec.dir:
        return road, q, t * spec.dir
    side = float(signplacement.cross(t, at[:2] - q[:2]))
    if abs(side) > OFF_CENTRE_M and side > 0:
        return road, q, -t          # mapped left of the drawn direction: it watches the other way
    return road, q, t


def _median(roads, at, travel):
    """The median's centre (3,) when the node lies between two opposite carriageways, else None."""
    ones = [r for r in roads if r[1].oneway and r[1].dir and r[1].kind == "paved" and len(r[2]) > 1]
    if len(ones) < 2:
        return None
    pair = frmotorwaymarkers._median(ones, at)
    if pair is None:
        return None
    centre, _ta, _normal, gap, _ = pair
    if float(np.hypot(*(at[:2] - centre[:2]))) > max(gap / 2, 1.0):
        return None                 # mapped on a carriageway or beside it, not in the median
    return centre, gap


def add(frame, roads, box, french, mesh, pavement, obstacles, rails, data=None):
    """Draw the cameras owned by box into mesh; their records."""
    records = []
    rows = nearby(frame, box) if data is None else data
    for lon, lat, heading, tower in rows:
        at = np.asarray(frame.geodetic_to_enu(lat * 1e-7, lon * 1e-7, 0.), float).ravel()
        if not geotiles.box_owns(box, *at[:2]) or not french(*at):
            continue
        got = travel_of(_nearest(roads, at), at, heading)
        if got is None:
            continue
        road, q, travel = got
        base, support = None, "roadside"
        median = _median(roads, at, travel)
        if median is not None:
            centre, gap = median
            if gap >= CABIN_W + 0.5:
                a, _, b = signplacement.span(centre, travel, CABIN_W)
                if pavement.visible(a, b, .15) and obstacles.visible(a, b, .14) and rails.clear(centre[:2], .2):
                    base, support = np.r_[centre[:2], obstacles.z(centre, centre[2])], "median"
        if base is None:
            right = np.array([travel[1], -travel[0]])
            half = road[1].width / 2
            for d in CLEARANCE_M:
                p = q[:2] + right * (half + d)
                a, _, b = signplacement.span(p, travel, CABIN_W)
                if not (pavement.visible(a, b, .15) and obstacles.visible(a, b, .14) and rails.clear(p, .3)):
                    continue
                z = obstacles.z(p, q[2])
                if abs(z - q[2]) <= MAX_DROP_M:
                    base = np.r_[p, z]
                    break
        if base is None:
            continue
        frroadmesh.speed_camera(mesh, base, travel, tower=bool(tower))
        records.append({"id": f"radar/{lon},{lat}", "kind": "speed-camera", "pos": [float(v) for v in base],
                        "source": "OpenStreetMap", "support": support})
    return records
