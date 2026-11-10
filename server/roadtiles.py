"""Road builder: the road tiles' grid (split out of roads.py): tile boxes, which tiles a player wants, the cells a tile
needs.
"""
import math
import types
import numpy as np
import mapcells
from roadbase import (
    CENTRE,
    CORRIDOR_M,
    EDGE_LINE,
    FAST_MS,
    LEAD_MAX_M,
    LEAD_S,
    OVERLAP,
    SEND_M,
    SIDE_PENALTY,
    TILE,
    TILE_BUILDING_LAYERS,
)


def tile_bbox(tx, ty):
    """A tile's box (x0, y0, x1, y1) in its frame: an anchor tile's square; a geotile's (tx a geotiles.GeoBox, ty
    None) its quad's bounding box in its own frame, an OwnedBox (box_owns: its own points exactly)."""
    b = getattr(tx, "box", None)
    if b is not None:
        return b
    return tx * TILE, ty * TILE, (tx + 1) * TILE, (ty + 1) * TILE


def box_owns(box, x, y):
    """geotiles.box_owns: the tile's own points (a geotile's exactly, an anchor tile's half-open box)."""
    import geotiles
    return geotiles.box_owns(box, x, y)


def tile_side(tx, ty):
    """m: the side of a square of the tile's area (TILE for an anchor tile)."""
    side = getattr(tx, "side", None)
    return side() if side is not None else TILE


def box_gap(x0, y0, x1, y1, x, y):
    """m from (x, y) to the nearest point of the box [x0, x1] x [y0, y1] (Euclidean; 0 inside)."""
    return math.hypot(max(x0 - x, x - x1, 0.0), max(y0 - y, y - y1, 0.0))


def tiles_near(x, y, radius):
    """Tile keys whose square comes within `radius` of (x, y): the Euclidean distance to its nearest point, a disc (the
    user, 2026-10-28: "change the quality/view distance radius from a square to a circle"). Before it was the square
    (Chebyshev): its corners reached 41 % farther than its sides, ~27 % more tiles than the disc for no reach ahead
    (the anchor frame's axes are not the way the car drives). Order: tx, then ty (as before)."""
    t0x, t1x = int(math.floor((x - radius) / TILE)), int(math.floor((x + radius) / TILE))
    t0y, t1y = int(math.floor((y - radius) / TILE)), int(math.floor((y + radius) / TILE))
    r2 = radius * radius
    out = []
    for tx in range(t0x, t1x + 1):
        dx = max(tx * TILE - x, x - (tx + 1) * TILE, 0.0)
        for ty in range(t0y, t1y + 1):
            dy = max(ty * TILE - y, y - (ty + 1) * TILE, 0.0)
            if dx * dx + dy * dy <= r2:
                out.append((tx, ty))
    return out


def tile_distance(tx, ty, x, y):
    """m from (x, y) to tile (tx, ty)'s square (Euclidean, as tiles_near; 0 inside)."""
    return box_gap(*tile_bbox(tx, ty), x, y)


def wanted_tiles(x, y, vx=0.0, vy=0.0):
    """The tile keys the mod should have around (x, y): those within SEND_M, and moving at FAST_MS or more also those
    within CORRIDOR_M of the path ahead, up to LEAD_S of driving (LEAD_MAX_M at most)."""
    want = set(tiles_near(x, y, SEND_M))
    s = math.hypot(vx, vy)
    lead = min(LEAD_MAX_M, s * LEAD_S)
    if s < FAST_MS or lead <= SEND_M:
        return want
    ux, uy = vx / s, vy / s
    d = 0.0
    while d <= lead:
        want.update(tiles_near(x + ux * d, y + uy * d, CORRIDOR_M))
        d += TILE / 2
    return want


def path_key(x, y, vx, vy):
    """The build order at speed: by how far along the path ahead the tile is, its sideways distance from the path
    counting SIDE_PENALTY times, a tile behind the car by its distance times SIDE_PENALTY (after those ahead)."""
    s = math.hypot(vx, vy)
    ux, uy = vx / s, vy / s

    def key(k):
        d = tile_distance(k[0], k[1], x, y)
        if d <= 0:
            return 0.0
        x0, y0, x1, y1 = tile_bbox(k[0], k[1])
        dx, dy = (x0 + x1) / 2 - x, (y0 + y1) / 2 - y
        along, side = dx * ux + dy * uy, abs(dy * ux - dx * uy)
        if along < -TILE / 2:
            return d * SIDE_PENALTY + LEAD_MAX_M
        return max(0.0, along) + SIDE_PENALTY * max(0.0, side - TILE / 2)
    return key


def view_key(conn, aid, x, y):
    """The build order of a tracker's tiles: nearest first, a tile outside the camera's view cone counting
    farfield.VIEW_PENALTY times as far (the far field's own rule, farfield.view_distance): what the player sees
    comes first, the tiles behind the car later. Plain distance without the pose's camera."""
    import farfield
    cam = getattr(conn, "cam", None)
    cone = farfield.view_cone(cam) if isinstance(cam, dict) and getattr(conn, "cam_aid", None) == aid else None

    def key(k):
        d = tile_distance(k[0], k[1], x, y)
        if cone is None or d <= 0:
            return d
        # (the far field's own rule, in Rust: the flat cone, or the 3D map's 3D one)
        return farfield.view_distances([types.SimpleNamespace(bbox=tile_bbox(k[0], k[1]))], cone, x, y)[0]
    return key




def cells_of_tile(frame, tx, ty, margin=OVERLAP):
    x0, y0, x1, y1 = tile_bbox(tx, ty)
    lat, lon, _ = frame.enu_to_geodetic(np.array([x0 - margin, x1 + margin, x0 - margin, x1 + margin]),
                                        np.array([y0 - margin, y0 - margin, y1 + margin, y1 + margin]), 0.0)
    return mapcells.cells_for_bbox(float(np.min(lat)), float(np.min(lon)), float(np.max(lat)), float(np.max(lon)))


def tile_needs(frame, tx, ty, buildings=True, widths=True, water_ring=0):
    """The (cell, layer)s a road tile asks the map data for, nearest first (MapData.warm, before building it): the
    roads, infrastructure (bridges) and land use of the cells under it and their buildings (buildings on), the roads of the ring
    around (grading's neighbours), the buildings of two rings around (widths on: roadwidth's walls) and the water of
    water_ring cells around (seacarve's rasters and river lines), and the roads of the second ring (gradesep's
    neighbours of the first)."""
    keys = cells_of_tile(frame, tx, ty)
    if not keys:
        return []
    z = keys[0].z
    n = 1 << z
    x0, x1 = min(k.x for k in keys), max(k.x for k in keys)
    y0, y1 = min(k.y for k in keys), max(k.y for k in keys)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    def ring(r):
        out = [mapcells.CellKey(z, x % n, y) for x in range(x0 - r, x1 + r + 1) for y in range(y0 - r, y1 + r + 1)
               if 0 <= y < n]
        return sorted(out, key=lambda k: (abs((k.x - cx + n / 2) % n - n / 2) + abs(k.y - cy), k))
    own = set(keys)
    # land use with every tile: parks, yards and billboards (worldprops), and industry._named_plants. Asked for
    # alone after the rest it was ~5 s of a first visit's Manhattan road tile, 2026-11-02.
    tile = ("roads", "infrastructure", "land_use") + (TILE_BUILDING_LAYERS if buildings else ())
    needs = [(k, ly) for k in sorted(keys) for ly in tile]
    needs += [(k, "roads") for k in ring(1) if k not in own]
    # (and the ring past that: gradesep reads the neighbours of every cell whose level ranges are asked, the ring's too)
    import gradesep
    if gradesep.ON:
        r1 = set(ring(1))
        needs += [(k, "roads") for k in ring(2) if k not in r1]
    if widths:      # the walls of the cells the ring's roads cross, each with its neighbours' buildings
        needs += [(k, "buildings") for k in ring(2) if k not in own]
    if water_ring:
        done = {k for k, ly in needs if ly == "water"}
        needs += [(k, "water") for k in ring(water_ring) if k not in done]
    return needs


def mark_poly(pts):
    """A white painted shape's road chunk: pts (k, 2 or 3), a convex polygon (draped by roadmesh: its z doesn't count)."""
    return {"m": "line_white", "w": 0.0, "tl": EDGE_LINE["tl"], "p": CENTRE["p"], "layer": "mark",
            "poly": [[round(float(v), 3) for v in q] for q in pts], "uvq": [[0.5, 0.5]] * len(pts), "n": []}
