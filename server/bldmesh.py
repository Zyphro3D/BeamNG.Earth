"""Procedural buildings: their meshes (split out of buildings.py): walls, cornices, roofs and domes, porches and stoops,
steeples, parking decks and garages, steel frames, collision.
"""
from runtimeprofiler import step as profile_step
import math
import numpy as np
import fast
import landmarks
import ships
from bldbase import (
    BANDS,
    BAND_SHOP,
    BRIDGE_CLASSES,
    CANOPY_POSTS_MAX_M,
    CANOPY_POST_HALF_M,
    CANOPY_POST_INSET_M,
    CANOPY_POST_STEP_M,
    CHURCH_CLASSES,
    CORNICE_DOWN_M,
    CORNICE_OUT_M,
    CORNICE_UP_M,
    DECK_COLUMN_M,
    DECK_EDGE_M,
    DECK_HEADROOM_M,
    DECK_LANDING_M,
    DECK_LANE_M,
    DECK_LEVEL_M,
    DECK_PARAPET_M,
    DECK_RAMP_EDGES,
    DECK_SLAB_M,
    DECK_SLOPE,
    DECK_STOREY_M,
    DOME_PROFILE,
    DOME_ROOFS,
    DOOR_GAP_M,
    DRIVEWAY,
    DRIVEWAY_W,
    EAVE_M,
    FOUNDATION_M,
    GAMBREL_BREAK,
    GAMBREL_PITCHES,
    GARAGE_DRIVE_MAX_M,
    GARAGE_DRIVE_W,
    GARAGE_SKINS,
    GROUND_FLOOR_SHOP_M,
    HALF_HIP,
    HOUSE_BAY_M,
    LOOKS,
    MANSARD_INSET_M,
    MANSARD_PITCHES,
    MARBLE_LOOK,
    MATERIALS,
    MAX_MESH_VERTS,
    OPEN_CANOPY_CLASSES,
    OVER_TUNNEL_CLEAR_M,
    PANEL,
    PIER_EVERY_M,
    PIER_W,
    PITCH,
    PLAIN_STRIP,
    PORCH_DECK_M,
    PORCH_DEPTH_M,
    PORCH_H_M,
    PORTAL_NEAR_M,
    ROOF_CLUTTER_M2,
    ROOF_SLATE,
    ROOF_UV_M,
    SHOP,
    SIDING_UV_M,
    SIGN_H,
    SIGN_OUT,
    SIGN_SHOP_M,
    SIGN_TINTS,
    SKILLION_PITCH,
    SKIRT_LITE_M,
    SKIRT_M,
    SPANDREL_H,
    SPANDREL_T,
    STAIR_TOWER_M,
    STAIR_TOWER_UP,
    STEEL,
    STEEL_BAY_M,
    STEEL_LEG_MAX,
    STEEL_TOP_SHARE,
    STOOP_H,
    STOOP_LANDING,
    STOOP_MAX_STEPS,
    STOOP_RISE,
    STOOP_RISE_MAX,
    STOOP_STEPS,
    STOOP_TREAD,
    STOOP_W,
    TILE_PX_M,
    WALL_TINTS,
    WIN_H,
    WIN_OUT_M,
    WIN_SILL,
    WIN_UV,
    WIN_W,
    _Looks,
    _area2,
    _hash01,
    oriented_box,
    triangulate,
)
from bldplan import (
    _landmark_token,
    _landmark_tokens,
    roof_pitch,
)


class Mesh:
    """Positions, uvs, colours, triangles and their material index, appended to. A material is a token
    (index, RGBA) as _Looks gives. Kept as flat lists of numbers (x y z x y z ..., one material per triangle):
    a road tile adds ~25k quads, and lists of tuples flattened at the end took ~2.5 s of a Steam Deck's 87 s."""

    def __init__(self):
        self.P, self.UV, self.C, self.T, self.M = [], [], [], [], []
        self.n = 0

    def quad(self, a, b, c, d, uva, uvb, uvc, uvd, mat):
        """Corners counter-clockwise seen from outside."""
        i = self.n
        self.P += (*a, *b, *c, *d)
        self.UV += (*uva, *uvb, *uvc, *uvd)
        self.C += mat[1] * 4
        self.T += (i, i + 1, i + 2, i, i + 2, i + 3)
        self.M += (mat[0], mat[0])
        self.n += 4

    def quads(self, P, UV, mid, mats):
        """Quads from a kernel (earthfast.building_walls): P 12 numbers per quad, UV 8, mid the index into mats (material
        tokens) per quad; the same lists quad() would make."""
        q = len(mid)
        if not q:
            return
        i = self.n
        self.P += P
        self.UV += UV
        cols = [tuple(mt[1]) * 4 if mt is not None else None for mt in mats]
        for k in mid:
            self.C += cols[k]
        self.T += (np.arange(q, dtype=np.int64)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3], np.int64) + i).ravel().tolist()
        idx = [mt[0] if mt is not None else None for mt in mats]
        self.M += [idx[k] for k in mid for _ in (0, 1)]
        self.n += 4 * q

    def tri(self, a, b, c, uva, uvb, uvc, mat):
        i = self.n
        self.P += (*a, *b, *c)
        self.UV += (*uva, *uvb, *uvc)
        self.C += mat[1] * 3
        self.T += (i, i + 1, i + 2)
        self.M.append(mat[0])
        self.n += 3

    def poly(self, pts3, uv, mat):
        """A flat polygon facing up when counter-clockwise seen from above, down when clockwise (its underside:
        the ear clipping wants counter-clockwise rings and gave a clockwise one no triangles at all, so the
        parking decks' and floating parts' undersides were missing, the user's report 2026-10-14)."""
        xy = np.asarray(pts3)[:, :2]
        if _area2(xy) < 0:
            k = len(xy)
            tris = np.array([(k - 1 - c, k - 1 - b, k - 1 - a) for a, b, c in triangulate(xy[::-1])],
                            np.int64).reshape(-1, 3)
        else:
            tris = triangulate(xy)
        i = self.n
        self.P += np.asarray(pts3, np.float64)[:, :3].ravel().tolist()
        self.UV += np.asarray(uv, np.float64)[:, :2].ravel().tolist()
        self.C += tuple(mat[1]) * len(pts3)
        self.T += (np.asarray(tris, np.int64).reshape(-1, 3) + i).ravel().tolist()
        self.M += [mat[0]] * len(tris)
        self.n += len(pts3)

    def arrays(self):
        return (np.array(self.P, np.float64).reshape(-1, 3), np.array(self.UV, np.float64).reshape(-1, 2),
                np.array(self.T, np.int64).reshape(-1, 3), np.array(self.M, np.int64),
                np.array(self.C, np.uint8).reshape(-1, 4))


_WALL_CONSTS = None


def _walls(m, spec, ring, z0, z_eave, mat_idx, lite=False):
    """Each wall of the ring, floor by floor: one band of the window sheet, a whole number of bays; a plain wall with
    sash windows laid over it for a plain tileable material (spec.wall; a big box's storefront band); a plain plinth
    under the floors on a slope; a shopfront ground floor; the good bays only where a band has some; the front door's
    bay as the band's plain strip, at the sheet's scale. The Rust loop (earthfast.building_walls: the quads the Python
    made, in its order and arithmetic)."""
    global _WALL_CONSTS
    if _WALL_CONSTS is None:
        _WALL_CONSTS = [float(v) for v in (SIDING_UV_M, TILE_PX_M, FOUNDATION_M, BANDS, BAND_SHOP[0], BAND_SHOP[1],
                                           DOOR_GAP_M, PLAIN_STRIP, GROUND_FLOOR_SHOP_M, WIN_OUT_M, HOUSE_BAY_M,
                                           *WIN_UV, WIN_W, WIN_H, WIN_SILL)]
    fast.require(76, "building_walls")
    band = spec.band or (0, 1)
    good = list(band[2]) if len(band) > 2 else []
    door = spec.door
    P, UV, mid = fast._ef.building_walls(
        np.ascontiguousarray(np.asarray(ring, np.float64)[:, :2]), float(z0), float(z_eave), bool(spec.wall),
        bool(spec.plain), int(band[0]), int(band[1]), [int(g) for g in good], int(spec.floors), float(spec.floor_h),
        bool(spec.shop), int(door[0]) if door else -1, float(door[1]) if door else 0.0, spec.roof == "flat",
        float(spec.seed) if isinstance(spec.seed, float) else 0.0, spec.windows is False,
        int(spec.front) if spec.front is not None else -1, bool(lite), _WALL_CONSTS)
    if mid:
        m.quads(P, UV, mid, [mat_idx[spec.facade], mat_idx[SHOP] if (spec.shop or spec.wall) else None,
                             mat_idx["bne_bld_fac_white"] if spec.wall else None])


def _cornice(m, spec, ring, z_eave, mat_idx):
    """A box moulding round a flat roof's edge (row houses' cornices, parapets), the plain white plaster."""
    mat = mat_idx["bne_bld_fac_white"]
    o, z0, z1 = CORNICE_OUT_M, z_eave - CORNICE_DOWN_M, z_eave + CORNICE_UP_M
    uv = ((0.002, 0.995), (0.012, 0.995), (0.012, 0.999), (0.002, 0.999))
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        e = b - a
        L = float(math.hypot(*e))
        if L < 0.3:
            continue
        d = e / L
        nn = np.array([d[1], -d[0]])                  # outward (counter-clockwise ring)
        a2, b2 = a + nn * o - d * o, b + nn * o + d * o
        m.quad((a2[0], a2[1], z0), (b2[0], b2[1], z0), (b2[0], b2[1], z1), (a2[0], a2[1], z1), *uv, mat)
        m.quad((a[0], a[1], z1), (a2[0], a2[1], z1), (b2[0], b2[1], z1), (b[0], b[1], z1), *uv, mat)      # top
        m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b2[0], b2[1], z0), (a2[0], a2[1], z0), *uv, mat)      # under
        # the lip's inner face, above the roof (seen from on the roof it was missing: the user's report, 2026-10-14)
        m.quad((b[0], b[1], z_eave), (a[0], a[1], z_eave), (a[0], a[1], z1), (b[0], b[1], z1), *uv, mat)


def _inside(ring, p):
    """Point p inside the ring (even-odd)."""
    x, y = p
    return fast._ef.point_inside(np.asarray(ring, np.float64), float(x), float(y))     # (Rust; even-odd)


def _box(m, c, u, hx, hy, z0, z1, mat):
    """An axis-u box (half sizes hx along u, hy across) from z0 to z1: four sides and a top."""
    v = np.array([-u[1], u[0]])
    k = [c + u * sx * hx + v * sy * hy for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    uv = ((0.002, 0.995), (0.012, 0.995), (0.012, 0.999), (0.002, 0.999))
    for i in range(4):
        a, b = k[i], k[(i + 1) % 4]
        m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1), *uv, mat)
    m.quad(*[(p[0], p[1], z1) for p in k], *uv, mat)


def _signs(m, spec, ring, z_top, mat_idx):
    """The sign band over a shop's street wall (spec.sign: its edge), shop by shop, a few without one."""
    edge = spec.sign
    a, b = ring[edge], ring[(edge + 1) % len(ring)]
    L = float(math.hypot(*(b - a)))
    if L < 3.0:
        return
    d = (b - a) / L
    out = np.array([d[1], -d[0]])
    t, k = 0.3, 0
    while t < L - 1.5:
        w = SIGN_SHOP_M[0] + (SIGN_SHOP_M[1] - SIGN_SHOP_M[0]) * _hash01(spec.seed, "signw", k)
        t1 = min(L - 0.3, t + w)
        if L - 0.3 - t1 < 2.5:                         # no sliver of a shop at the end
            t1 = L - 0.3
        if _hash01(spec.seed, "signon", k) < 0.85:
            c = a + d * ((t + t1) / 2) + out * (SIGN_OUT / 2)
            look = f"sign|{int(_hash01(spec.seed, 'signc', k) * len(SIGN_TINTS))}"
            _box(m, c, d, (t1 - t) / 2 - 0.15, SIGN_OUT / 2, z_top - 0.15 - SIGN_H, z_top - 0.15, mat_idx[look])
        t, k = t1, k + 1


def _portal_lift(frame, ring, x0, y0, tunnels_xy, elev):
    """How far (m) over its lowest ground a building over a tunnel stands: to OVER_TUNNEL_CLEAR_M over the lowest ground
    along the tunnel lines within PORTAL_NEAR_M of it (the road at the mouth), 0 when its ground is that high."""
    lo = ring.min(0) - PORTAL_NEAR_M
    hi = ring.max(0) + PORTAL_NEAR_M
    pts = []
    for line in tunnels_xy:
        ln = np.asarray(line, np.float64)
        if len(ln) < 2:
            continue
        seg = np.hypot(*np.diff(ln, axis=0).T)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        t = np.arange(0.0, s[-1] + 1e-9, 2.0)
        p = np.stack([np.interp(t, s, ln[:, 0]), np.interp(t, s, ln[:, 1])], axis=1)
        p = p[(p[:, 0] >= lo[0]) & (p[:, 0] <= hi[0]) & (p[:, 1] >= lo[1]) & (p[:, 1] <= hi[1])]
        if len(p):
            pts.append(p)
    if not pts:
        return 0.0
    q = np.concatenate(pts + [ring])
    la, lo_, _ = frame.enu_to_geodetic(q[:, 0] + x0, q[:, 1] + y0, np.zeros(len(q)))
    z = np.asarray(elev(la, lo_), np.float64)
    nr = len(ring)
    road, ground = z[:-nr], z[-nr:]
    if not len(road) or not np.isfinite(road).any() or not np.isfinite(ground).any():
        return 0.0
    return max(0.0, float(np.nanmin(road)) + OVER_TUNNEL_CLEAR_M - float(np.nanmin(ground)))


def base_z(spec, gmin, gmax):
    """The ground a building's floors stand on: the footprint's highest, but the lowest for one over a tunnel's portal
    (the road at its mouth: from the hill over it, Wabash's portal house floated ~10 m over the portal and
    Eisenhower's portal building over its frames, 2026-10-17)."""
    return gmin if spec.portal else gmax


def drawn_ring(spec):
    """The footprint its walls are drawn on: a pitched roof's house is its oriented box (build); its door, steps and
    porch go on these walls (on the map's outline they stood off a box wall at an angle, or inside it, only the
    door's frame showing: Stelvio pass, the user's reports 2026-10-17)."""
    ring = spec.ring
    if spec.roof != "flat" and spec.roof not in DOME_ROOFS:     # (a dome stands on its own round drum)
        if spec.fr_house:
            import francehouse_mesh
            c, u, hl, hw = francehouse_mesh.model_box(ring)
        else:
            c, u, hl, hw = oriented_box(ring)
        v = np.array([-u[1], u[0]])
        ring = np.array([c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw, c - u * hl + v * hw])
    return ring


def _porch(m, spec, base, mat_idx):
    """A front porch on the front door's wall (spec.door: (edge, distance along it) of spec.ring): a deck, three
    posts and a shallow roof, the house's trim white and its roof's colour."""
    edge, sd = spec.door
    ring = drawn_ring(spec)
    a, b = ring[edge], ring[(edge + 1) % len(ring)]
    L = float(math.hypot(*(b - a)))
    if L < 4.0:
        return
    d = (b - a) / L
    out = np.array([d[1], -d[0]])
    w = min(L * 0.75, 7.5)
    c0 = float(np.clip(sd, w / 2, L - w / 2))
    c = a + d * c0
    white = mat_idx["bne_bld_fac_white"]
    D = PORCH_DEPTH_M
    _box(m, c + out * (D / 2), d, w / 2, D / 2, base - 0.3, base + PORCH_DECK_M, white)              # deck
    top = base + PORCH_DECK_M + PORCH_H_M
    for t in (-w / 2 + 0.15, 0.0, w / 2 - 0.15):                                                   # posts
        _box(m, c + d * t + out * (D - 0.15), d, 0.08, 0.08, base + PORCH_DECK_M, top, white)
    roof = mat_idx[spec.roof_mat] if spec.roof_mat else white
    _box(m, c + out * (D / 2 + 0.1), d, w / 2 + 0.15, D / 2 + 0.1, top, top + 0.18, roof)          # roof


def door_ground(spec, edge, along, gmax):
    """The ground at a point `along` m on edge `edge` of the footprint (its corners' ground, spec.gring), or gmax
    when not known: a row house's door and steps (from the footprint's highest ground, its back yard's on a slope,
    the steps on Bradford Street stood 1.5 m over the sidewalk, five or six of them; Mapillary: one or two)."""
    g = getattr(spec, "gring", None)
    ring = spec.ring
    if g is None or len(g) != len(ring):
        return gmax
    a, b = ring[edge], ring[(edge + 1) % len(ring)]
    L = float(math.hypot(*(b - a)))
    t = 0.0 if L < 1e-6 else min(max(along / L, 0.0), 1.0)
    return float(g[edge] + t * (g[(edge + 1) % len(ring)] - g[edge]))


def _stoop(m, spec, ring, gmin, base, mat_idx):
    """Marble steps up to the front door (spec.stoop: (edge, distance along it); base: the ground there, the door
    STOOP_H over it): STOOP_STEPS treads to a landing, each block down under the footprint's lowest ground (they floated where the street falls away, Fells Point, 2026-10-17)."""
    edge, sd = spec.stoop
    a, b = ring[edge], ring[(edge + 1) % len(ring)]
    L = float(math.hypot(*(b - a)))
    if L < 1e-6:
        return
    d = (b - a) / L
    out = np.array([d[1], -d[0]])
    c = a + d * sd
    mat = mat_idx[MARBLE_LOOK]
    base = door_ground(spec, edge, sd, base)     # (the ground at the door)
    top = base + STOOP_H
    z0 = min(gmin, base) - 0.3
    # as many steps as it takes down to the street (the footprint's lowest ground): from the highest one only the
    # bottom step stood a metre over the sidewalk on a white block (Bradford Street, Mapillary 2026-10-17)
    n = int(np.clip(round((top - min(gmin, base)) / STOOP_RISE), STOOP_STEPS - 1, STOOP_MAX_STEPS))
    rise = min((top - min(gmin, base)) / n, STOOP_RISE_MAX)    # (n risers: the landing's and n - 1 steps' fronts;
                                                                # higher, the bottom step on a plinth)
    for k in range(n):
        depth = STOOP_LANDING + k * STOOP_TREAD
        zt = top - k * rise
        _box_tiled(m, c + out * depth / 2, d, STOOP_W / 2, depth / 2, z0, zt, mat)


def _canopy_posts(m, spec, gmin, z_under, mat):
    """Steel posts under a carport's roof (OPEN_CANOPY_CLASSES) along its length, from its lowest ground to its
    underside: one row down its middle, two along the edges of one over 12 m wide."""
    c, u, hl, hw = oriented_box(spec.ring)
    v = np.array([-u[1], u[0]])
    n = max(2, int(round(2 * hl / CANOPY_POST_STEP_M)) + 1)
    along = np.linspace(-hl + CANOPY_POST_INSET_M, hl - CANOPY_POST_INSET_M, n)
    # a solar car park's canopy over one or two rows of bays stands on one row of posts down its middle
    rows = (0.0,) if hw <= 6.0 else (-(hw - CANOPY_POST_INSET_M), hw - CANOPY_POST_INSET_M)
    for a in along:
        for b in rows:
            _box(m, c + u * a + v * b, u, CANOPY_POST_HALF_M, CANOPY_POST_HALF_M, gmin - 0.3, z_under, mat)


def _box_tiled(m, c, u, hx, hy, z0, z1, mat, tile=1.0):
    """_box with the material tiled at `tile` m per repeat (a stone texture, not the facade sheets' colour patches)."""
    v = np.array([-u[1], u[0]])
    k = [c + u * sx * hx + v * sy * hy for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for i in range(4):
        p, q = k[i], k[(i + 1) % 4]
        w = float(math.hypot(*(q - p))) / tile
        hz0, hz1 = z0 / tile, z1 / tile
        m.quad((p[0], p[1], z0), (q[0], q[1], z0), (q[0], q[1], z1), (p[0], p[1], z1),
               (0.0, hz0), (w, hz0), (w, hz1), (0.0, hz1), mat)
    m.quad(*[(p[0], p[1], z1) for p in k], *[(p[0] / tile, p[1] / tile) for p in k], mat)


def _steeple(m, spec, z_eave, mat_idx):
    """A church's square tower with a pyramid spire, at one end of its nave (hashed)."""
    c, u, hl, hw = oriented_box(spec.ring)
    if _hash01(spec.seed, "steeple") < 0.5:
        u = -u
    side = float(np.clip(1.1 * hw, 3.0, 6.0)) / 2
    p = c + u * (hl - side)
    for k in range(1, 12):              # on the footprint (an irregular one's box end can be outside it)
        if _inside(spec.ring, p):
            break
        p = c + u * (hl - side) * (1 - k / 12)
    rise = (min(hw, 12.0) + EAVE_M) * math.tan(math.radians(spec.pitch or PITCH["gabled"]))
    top = z_eave + rise + 6.0
    _box(m, p, u, side, side, z_eave - 1.0, top, mat_idx[spec.facade])
    v = np.array([-u[1], u[0]])
    k = [p + u * sx * side + v * sy * side for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    apex = (float(p[0]), float(p[1]), top + 2.6 * side * 2)
    rm = mat_idx[f"{ROOF_SLATE}|3"]
    for i in range(4):
        a, b = k[i], k[(i + 1) % 4]
        m.tri((a[0], a[1], top), (b[0], b[1], top), apex, (0, 0), (1, 0), (0.5, 2.0), rm)


def _clutter(m, spec, ring, z_eave, mat_idx):
    area = _area2(ring) / 2
    if spec.roof != "flat" or spec.plain or area < ROOF_CLUTTER_M2 or spec.cls in CHURCH_CLASSES:
        return
    c, u, hl, hw = oriented_box(ring)
    v = np.array([-u[1], u[0]])
    n = min(6, 1 + int(area / 600.0))
    mat = mat_idx["bne_bld_fac_grey"]
    for k in range(n):
        s = (_hash01(spec.seed, "cx", k) * 2 - 1) * (hl - 3.0)
        t = (_hash01(spec.seed, "cy", k) * 2 - 1) * (hw - 3.0)
        p = c + u * s + v * t
        if not _inside(ring, p):
            continue
        big = _hash01(spec.seed, "cb", k) < 0.25
        hx, hy, h = (2.0, 1.6, 2.6) if big else (1.1, 0.7, 1.1)
        _box(m, p, u, hx, hy, z_eave, z_eave + h, mat)


def _dome(m, ring, z_eave, rise, mat, shape="dome", lite=False):
    """A dome (or onion dome) of `rise` on the footprint ring at z_eave: the ring scaled toward its centre up
    DOME_PROFILE (every other ring when lite). mat: a material index (or the skyline's (index, colour))."""
    ring = np.asarray(ring, np.float64)[:, :2]
    if len(ring) < 3 or rise <= 0:
        return
    if _area2(ring) < 0:
        ring = ring[::-1]
    c = ring.mean(0)
    prof = DOME_PROFILE.get(shape, DOME_PROFILE["dome"])
    if lite:
        prof = prof[::2] if len(prof) % 2 else prof[::2] + prof[-1:]
    n = len(ring)
    levels = [[(float(c[0] + (p[0] - c[0]) * s), float(c[1] + (p[1] - c[1]) * s), float(z_eave + rise * t))
               for p in ring] for s, t in prof]

    def uv(q):
        return (q[0] / ROOF_UV_M, q[1] / ROOF_UV_M)
    for lo, hi in zip(levels[:-1], levels[1:]):
        for i in range(n):
            a, b, bu, au = lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]
            if au == bu:                    # (the apex)
                m.tri(a, b, au, uv(a), uv(b), uv(au), mat)
            else:
                m.quad(a, b, bu, au, uv(a), uv(b), uv(bu), uv(au), mat)


def _roof(m, spec, ring, z_eave, mat_idx, lite=False):
    rm = mat_idx[spec.roof_mat]
    if spec.roof in DOME_ROOFS:
        _c, _u, _hl, hw = oriented_box(ring)
        _dome(m, ring, z_eave, hw * math.tan(math.radians(roof_pitch(spec))), rm, spec.roof, lite)
        return
    if spec.roof == "flat":
        pts = [(p[0], p[1], z_eave) for p in ring]
        uv = [(p[0] / ROOF_UV_M, p[1] / ROOF_UV_M) for p in ring]
        m.poly(pts, uv, rm)
        return
    c, u, hl, hw = oriented_box(ring)
    v = np.array([-u[1], u[0]])
    hl2, hw2 = hl + EAVE_M, hw + EAVE_M
    rise = hw2 * math.tan(math.radians(roof_pitch(spec)))
    ze = z_eave - EAVE_M * math.tan(math.radians(roof_pitch(spec)))
    zr = ze + rise

    def at(s, t, z):
        p = c + u * s + v * t
        return (float(p[0]), float(p[1]), float(z))
    slope = math.hypot(hw2, rise)

    def uv(s, t):            # along the ridge, up the slope (m / ROOF_UV_M)
        return (s / ROOF_UV_M, (hw2 - abs(t)) / hw2 * slope / ROOF_UV_M)

    def end_wall(sgn, outline):
        """The wall up to the roof at an end of the house: outline (t, z) left to right seen from outside, fanned; the
        facade's plain colour, as a gable's."""
        s = sgn * hl
        pts = [at(s, sgn * t, z) for t, z in outline]
        for b, c in zip(pts[1:], pts[2:]):
            m.tri(pts[0], b, c, (0, 0.995), (0.02, 0.995), (0.01, 0.999), mat_idx[spec.facade])

    if spec.roof == "skillion":
        tan = math.tan(math.radians(SKILLION_PITCH))
        hi = 1.0 if spec.seed < 0.5 else -1.0           # (no direction in the data: which side is high, per building)

        def z(t):
            return z_eave + (hi * t + hw) * tan
        m.quad(at(-hl2, -hw2, z(-hw2)), at(hl2, -hw2, z(-hw2)), at(hl2, hw2, z(hw2)), at(-hl2, hw2, z(hw2)),
               uv(-hl2, -hw2), uv(hl2, -hw2), (hl2 / ROOF_UV_M, 2 * hw2 / ROOF_UV_M), (-hl2 / ROOF_UV_M, 2 * hw2 / ROOF_UV_M),
               rm)
        top = z(hi * hw)
        fac = mat_idx[spec.facade]
        a, b = (hl, -hl) if hi > 0 else (-hl, hl)       # the high wall above the eave line
        m.quad(at(a, hi * hw, z_eave), at(b, hi * hw, z_eave), at(b, hi * hw, top), at(a, hi * hw, top),
               (0, 0.995), (0.02, 0.995), (0.02, 0.999), (0, 0.999), fac)
        for sgn in (-1, 1):
            end_wall(sgn, [(-hw, z_eave), (hw, z_eave), (sgn * hi * hw, top)])
        return
    if spec.roof == "gambrel":
        lo, up = (math.tan(math.radians(p)) for p in GAMBREL_PITCHES)
        tb = hw * GAMBREL_BREAK                          # the break line, out from the ridge
        ze = z_eave - EAVE_M * lo
        zb = z_eave + (hw - tb) * lo
        zr = zb + tb * up
        for sgn in (-1, 1):                              # (the slopes of one side, counter-clockwise from above)
            for t0, z0, t1, z1 in ((hw2, ze, tb, zb), (tb, zb, 0.0, zr)):
                a, b = (-hl2, hl2) if sgn < 0 else (hl2, -hl2)
                m.quad(at(a, sgn * t0, z0), at(b, sgn * t0, z0), at(b, sgn * t1, z1), at(a, sgn * t1, z1),
                       uv(a, t0), uv(b, t0), uv(b, t1), uv(a, t1), rm)
        for sgn in (-1, 1):
            end_wall(sgn, [(-hw, z_eave), (hw, z_eave), (tb, zb), (0.0, zr - EAVE_M * 0.2), (-tb, zb)])
        return
    if spec.roof == "mansard":
        lo, up = (math.tan(math.radians(p)) for p in MANSARD_PITCHES)
        inset = min(0.3 * hw, MANSARD_INSET_M)           # (from the walls: the overhang carries the slope on down)
        ze = z_eave - EAVE_M * lo
        zb = z_eave + inset * lo
        il, iw = hl - inset, hw - inset
        zr = zb + iw * up
        r = max(il - iw, 0.0)
        for (a, ta), (b, tb) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            m.quad(at(a * hl2, ta * hw2, ze), at(b * hl2, tb * hw2, ze), at(b * il, tb * iw, zb), at(a * il, ta * iw, zb),
                   (0, 0), (2 * hl2 / ROOF_UV_M, 0), (2 * hl2 / ROOF_UV_M, (inset + EAVE_M) / ROOF_UV_M), (0, (inset + EAVE_M) / ROOF_UV_M), rm)
        m.quad(at(-il, -iw, zb), at(il, -iw, zb), at(r, 0, zr), at(-r, 0, zr), uv(-il, -iw), uv(il, -iw), uv(r, 0), uv(-r, 0), rm)
        m.quad(at(il, iw, zb), at(-il, iw, zb), at(-r, 0, zr), at(r, 0, zr), uv(il, iw), uv(-il, iw), uv(-r, 0), uv(r, 0), rm)
        m.tri(at(il, -iw, zb), at(il, iw, zb), at(r, 0, zr), (-iw / ROOF_UV_M, 0), (iw / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)
        m.tri(at(-il, iw, zb), at(-il, -iw, zb), at(-r, 0, zr), (-iw / ROOF_UV_M, 0), (iw / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)
        return
    if spec.roof == "pyramidal":
        apex = at(0, 0, zr)
        for (a, ta), (b, tb) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            m.tri(at(a * hl2, ta * hw2, ze), at(b * hl2, tb * hw2, ze), apex,
                  (-hl2 / ROOF_UV_M, 0), (hl2 / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)
        return
    if spec.roof == "half-hipped":
        tw = hw2 * HALF_HIP                              # where the small hip's foot meets the slopes
        zc = zr - rise * HALF_HIP
        r = max(hl2 - tw, 0.0)
        m.quad(at(-hl2, -hw2, ze), at(hl2, -hw2, ze), at(hl2, -tw, zc), at(-hl2, -tw, zc),
               uv(-hl2, -hw2), uv(hl2, -hw2), uv(hl2, -tw), uv(-hl2, -tw), rm)
        m.quad(at(-hl2, -tw, zc), at(hl2, -tw, zc), at(r, 0, zr), at(-r, 0, zr),
               uv(-hl2, -tw), uv(hl2, -tw), uv(r, 0), uv(-r, 0), rm)
        m.quad(at(hl2, hw2, ze), at(-hl2, hw2, ze), at(-hl2, tw, zc), at(hl2, tw, zc),
               uv(hl2, hw2), uv(-hl2, hw2), uv(-hl2, tw), uv(hl2, tw), rm)
        m.quad(at(hl2, tw, zc), at(-hl2, tw, zc), at(-r, 0, zr), at(r, 0, zr),
               uv(hl2, tw), uv(-hl2, tw), uv(-r, 0), uv(r, 0), rm)
        m.tri(at(hl2, -tw, zc), at(hl2, tw, zc), at(r, 0, zr), (-tw / ROOF_UV_M, 0), (tw / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)
        m.tri(at(-hl2, tw, zc), at(-hl2, -tw, zc), at(-r, 0, zr), (-tw / ROOF_UV_M, 0), (tw / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)
        top = zc + EAVE_M * rise / hw2                   # (the wall reaches the hip above it, under the overhang)
        for sgn in (-1, 1):
            end_wall(sgn, [(-hw, z_eave), (hw, z_eave), (tw, top), (-tw, top)])
        return
    if spec.roof == "gabled":
        r = hl2
        m.quad(at(-hl2, -hw2, ze), at(hl2, -hw2, ze), at(r, 0, zr), at(-r, 0, zr),
               uv(-hl2, -hw2), uv(hl2, -hw2), uv(r, 0), uv(-r, 0), rm)
        m.quad(at(hl2, hw2, ze), at(-hl2, hw2, ze), at(-r, 0, zr), at(r, 0, zr),
               uv(hl2, hw2), uv(-hl2, hw2), uv(-r, 0), uv(r, 0), rm)
        # the gable ends: wall triangles from the eave line up to the ridge, the facade's plain colour
        fac = mat_idx[spec.facade]
        for sgn in (-1, 1):
            s = sgn * hl
            a, b, top = at(s, -sgn * hw, z_eave), at(s, sgn * hw, z_eave), at(s, 0, zr - EAVE_M * 0.2)
            if sgn > 0:
                m.tri(a, b, top, (0, 0.995), (0.02, 0.995), (0.01, 0.999), fac)
            else:
                m.tri(a, b, top, (0, 0.995), (0.02, 0.995), (0.01, 0.999), fac)
    else:                    # hipped
        r = max(hl2 - hw2, 0.0)
        m.quad(at(-hl2, -hw2, ze), at(hl2, -hw2, ze), at(r, 0, zr), at(-r, 0, zr),
               uv(-hl2, -hw2), uv(hl2, -hw2), uv(r, 0), uv(-r, 0), rm)
        m.quad(at(hl2, hw2, ze), at(-hl2, hw2, ze), at(-r, 0, zr), at(r, 0, zr),
               uv(hl2, hw2), uv(-hl2, hw2), uv(-r, 0), uv(r, 0), rm)
        m.tri(at(hl2, -hw2, ze), at(hl2, hw2, ze), at(r, 0, zr),
              (-hw2 / ROOF_UV_M, 0), (hw2 / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)
        m.tri(at(-hl2, hw2, ze), at(-hl2, -hw2, ze), at(-r, 0, zr),
              (-hw2 / ROOF_UV_M, 0), (hw2 / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)


def _concat(meshes):
    """build() meshes (the same MATERIALS) as one."""
    if len(meshes) == 1:
        return meshes[0]
    off = np.cumsum([0] + [len(m[0]) for m in meshes[:-1]])
    return (np.concatenate([m[0] for m in meshes]), np.concatenate([m[1] for m in meshes]),
            np.concatenate([m[2] + o for m, o in zip(meshes, off)]), np.concatenate([m[3] for m in meshes]),
            meshes[0][4], np.concatenate([m[5] for m in meshes]))


@profile_step('Building mesh assembly')
def build_parts(specs_and_ground, lite=False, max_verts=MAX_MESH_VERTS, with_lod=False):
    """build() split into meshes of at most max_verts vertices each (whole buildings per mesh): a list, [] if empty.
    Each building is built once (build draws them one by one) and the meshes joined. with_lod: each mesh carries the
    lite mesh of its buildings as a 7th element (write: the shape's coarse detail)."""
    parts, cur, cur_lite, n = [], [], [], 0
    if not isinstance(specs_and_ground, (list, tuple)):
        specs_and_ground = list(specs_and_ground)
    prepared = None
    if any(item[0].fr_house for item in specs_and_ground) and max_verts == MAX_MESH_VERTS:
        import francehouse_mesh
        prepared = francehouse_mesh.prepared_parts(specs_and_ground, lite, with_lod, max_verts)

    def flush():
        if cur:
            mesh = _concat(cur)
            parts.append(mesh + (_concat(cur_lite) if cur_lite else None,) if with_lod and not lite else mesh)

    for i, item in enumerate(specs_and_ground):
        one = prepared[i][0] if prepared is not None else build([item], lite)
        if one is None:
            continue
        if cur and n + len(one[0]) > max_verts:
            flush()
            cur, cur_lite, n = [], [], 0
        cur.append(one)
        n += len(one[0])
        if with_lod and not lite:
            o = prepared[i][1] if prepared is not None else build([item], lite=True)
            if o is not None:
                cur_lite.append(o)
    flush()
    return parts


def _deck_layout(ring, storey):
    """The ramp's frame on the longest edge of the counter-clockwise ring it fits along: (edge index i, A, along
    unit u, inward unit n, s0, s1), the ramps between arc lengths s0 and s1; None when it fits along none of the
    DECK_RAMP_EDGES longest (too short for DECK_SLOPE, or the well outside the ring: the longest edge of an
    L-shaped garage can be the short leg's, 3 of downtown Baltimore's 15 garages were boxes, 2026-10-17)."""
    lens = [float(np.hypot(*(ring[(k + 1) % len(ring)] - ring[k]))) for k in range(len(ring))]
    depth = DECK_EDGE_M + 2 * DECK_LANE_M
    for i in sorted(range(len(ring)), key=lambda k: -lens[k])[:DECK_RAMP_EDGES]:
        a, b = ring[i], ring[(i + 1) % len(ring)]
        L = lens[i]
        s0, s1 = 0.5 + DECK_LANDING_M, L - 0.5 - DECK_LANDING_M
        if s1 - s0 < storey / DECK_SLOPE:
            return None                         # the rest are shorter
        u = (b - a) / L
        n = np.array([-u[1], u[0]])             # the inside is on the left
        if all(_inside(ring, a + u * s + n * d) for s in np.linspace(0.5, L - 0.5, 9)
               for d in (0.3, depth / 2, depth + 0.5, depth + DECK_LANDING_M / 2)):
            return i, a, u, n, s0, s1
    return None


def _deck_wall(m, p, q, z0, z1, mat, t=0.2):
    """A thin wall (both faces and a top) from p to q, z0 to z1."""
    d = q - p
    L = float(np.hypot(*d))
    if L > 0.05:
        _box(m, (p + q) / 2, d / L, L / 2, t / 2, z0, z1, mat)


def deck_plan(spec, gmin, gmax):
    """(counter-clockwise ring, floors, storey height, _deck_layout) of a parking structure drawn as open decks, or
    None (not a garage, ground too uneven, storeys too low, the ramp does not fit: a closed box)."""
    if not spec.deck or gmax - gmin > DECK_LEVEL_M or not spec.height or (spec.min_h or 0.0) > 0.5:
        return None
    floors = max(1, int(spec.floors or 1))
    storey = spec.height / floors
    if storey < DECK_HEADROOM_M:                # split levels counted as floors (10 in 20.8 m): whole decks
        floors = max(1, int(spec.height // DECK_STOREY_M))
        storey = spec.height / floors
    if storey < DECK_HEADROOM_M:
        return None
    ring = np.asarray(spec.ring, np.float64)
    if _area2(ring) < 0:
        ring = ring[::-1]
    lay = _deck_layout(ring, storey)
    return None if lay is None else (ring, floors, storey, lay)


def _parking_deck(m, spec, gmin, gmax, mat_idx):
    """The open parking structure (see DECK_*); False when it does not apply (the caller draws the box)."""
    plan_ = deck_plan(spec, gmin, gmax)
    if plan_ is None:
        return False
    ring, floors, storey, (i, A, u, n, s0, s1) = plan_
    # weathered concrete (Baltimore's street photos: the white decks read as a skeleton; city garages are grey)
    mat = mat_idx[f"{PANEL}|{len(WALL_TINTS[PANEL]) - 1 - int(_hash01(spec.seed, 'deck') * 2)}"]
    W, e = DECK_LANE_M, DECK_EDGE_M
    depth = e + 2 * W
    zs = [gmax + 0.03 + k * storey for k in range(floors + 1)]

    def at(s, d):
        return A + u * s + n * d

    def uvp(p):
        return (float(p[0]) / 4.0, float(p[1]) / 4.0)

    k_ring = len(ring)
    m.poly([(p[0], p[1], zs[0]) for p in ring], [uvp(p) for p in ring], mat)          # the ground deck, whole
    if gmax - gmin > 0.05:                      # its plinth down to the lowest ground (a sloping street)
        for j in range(k_ring):
            p, q = ring[j], ring[(j + 1) % k_ring]
            m.quad((p[0], p[1], gmin - FOUNDATION_M), (q[0], q[1], gmin - FOUNDATION_M), (q[0], q[1], zs[0]),
                   (p[0], p[1], zs[0]), (0, 0), (1, 0), (1, 0.1), (0, 0.1), mat)
    # the upper decks: the ring with the ramp well (both lanes, s0..s1) cut in from the longest edge
    well = [at(s0, 0.0), at(s0, depth), at(s1, depth), at(s1, 0.0)]
    pts = [A] + well + [ring[(i + 1 + j) % k_ring] for j in range(k_ring - 1)]
    for k in range(1, floors + 1):
        z = zs[k]
        m.poly([(p[0], p[1], z) for p in pts], [uvp(p) for p in pts], mat)
        m.poly([(p[0], p[1], z - DECK_SLAB_M) for p in pts[::-1]], [uvp(p) for p in pts[::-1]], mat)
        for j in range(len(pts)):
            p, q = pts[j], pts[(j + 1) % len(pts)]
            m.quad((p[0], p[1], z - DECK_SLAB_M), (q[0], q[1], z - DECK_SLAB_M), (q[0], q[1], z), (p[0], p[1], z),
                   (0, 0), (1, 0), (1, 0.1), (0, 0.1), mat)
            if not (1 <= j <= 3):               # the parapet, not round the well (below)
                _deck_wall(m, p, q, z, z + DECK_PARAPET_M, mat)
        # round the well: its inner side and the end away from the ramps' landing (ramp k - 1 comes up at s1 when
        # k - 1 is even, and ramp k leaves from that same end); the top deck has no ramp leaving: its lane closed
        land = s1 if (k - 1) % 2 == 0 else s0
        far = s0 if land == s1 else s1
        _deck_wall(m, at(s0, depth), at(s1, depth), z, z + DECK_PARAPET_M, mat)
        _deck_wall(m, at(far, 0.0), at(far, depth), z, z + DECK_PARAPET_M, mat)
        if k == floors:
            d_free = e + (k % 2) * W
            _deck_wall(m, at(land, d_free), at(land, d_free + W), z, z + DECK_PARAPET_M, mat)
    # the ramps: ramp k from deck k to deck k + 1 on lane k % 2, even ones climbing from s0 to s1, walled both sides
    for k in range(floors):
        d0 = e + (k % 2) * W
        sa, sb = (s0, s1) if k % 2 == 0 else (s1, s0)
        za, zb = zs[k], zs[k + 1]
        top = [(at(sa, d0), za), (at(sb, d0), zb), (at(sb, d0 + W), zb), (at(sa, d0 + W), za)]
        if sb < sa:                             # counter-clockwise seen from above
            top = top[::-1]
        quad = [(p[0], p[1], z) for p, z in top]
        m.quad(*quad, (0, 0), (1, 0), (1, 1), (0, 1), mat)
        m.quad(*[(x, y, z - DECK_SLAB_M) for x, y, z in quad[::-1]], (0, 0), (1, 0), (1, 1), (0, 1), mat)
        for d in (d0, d0 + W):
            steps = 6
            for j in range(steps):
                f0, f1 = j / steps, (j + 1) / steps
                ca, cb = at(sa + (sb - sa) * f0, d), at(sa + (sb - sa) * f1, d)
                _deck_wall(m, ca, cb, za + (zb - za) * f0, za + (zb - za) * f1 + DECK_PARAPET_M, mat, 0.15)
    _garage_skin(m, spec, ring, zs, i, mat_idx)
    # the columns: round the outline every DECK_COLUMN_M, from the foundation to the top deck
    for j in range(k_ring):
        p, q = ring[j], ring[(j + 1) % k_ring]
        L = float(np.hypot(*(q - p)))
        if L < 0.5:
            continue
        d = (q - p) / L
        inward = np.array([-d[1], d[0]])
        for t in np.arange(0.0, L, DECK_COLUMN_M):
            _box(m, p + d * t + inward * 0.35, d, 0.25, 0.25, zs[0] - FOUNDATION_M, zs[-1], mat)
    return True


def _garage_skin(m, spec, ring, zs, ramp_edge, mat_idx):
    """The facade of an open parking structure: spandrels round its outline at every deck above the ground,
    piers at its corners and along it, a stair tower; brick or precast concrete by its seed (GARAGE_SKINS)."""
    mat = mat_idx[GARAGE_SKINS[int(_hash01(spec.seed, "skin") * len(GARAGE_SKINS)) % len(GARAGE_SKINS)]]
    k_ring = len(ring)
    for j in range(k_ring):
        p, q = ring[j], ring[(j + 1) % k_ring]
        L = float(np.hypot(*(q - p)))
        if L < 0.5:
            continue
        d = (q - p) / L
        out = np.array([d[1], -d[0]])           # (counter-clockwise ring: the outside on the right)
        c = (p + q) / 2 + out * (SPANDREL_T / 2 - 0.05)
        for z in zs[1:]:
            _box_tiled(m, c, d, L / 2, SPANDREL_T / 2, z - DECK_SLAB_M - 0.1, z + SPANDREL_H, mat, 2.0)
        for t in [0.0] + list(np.arange(PIER_EVERY_M, L - PIER_EVERY_M / 2, PIER_EVERY_M)):
            _box_tiled(m, p + d * t + out * 0.05, d, PIER_W / 2, PIER_W / 2, zs[0] - FOUNDATION_M,
                       zs[-1] + SPANDREL_H + 0.2, mat, 2.0)
    # the stair tower: at the ring's vertex farthest from the ramps' edge
    a, b = ring[ramp_edge], ring[(ramp_edge + 1) % k_ring]
    ab = b - a
    L2 = max(float(ab @ ab), 1e-9)

    def dist(v):
        t = float(np.clip(((v - a) @ ab) / L2, 0.0, 1.0))
        return float(np.hypot(*(v - (a + t * ab))))
    jv = max(range(k_ring), key=lambda j: dist(ring[j]))
    v = ring[jv]
    inward = ring.mean(0) - v
    inward = inward / max(float(np.hypot(*inward)), 1e-9)
    c = v + inward * (STAIR_TOWER_M * 0.75)
    if _inside(ring, c):
        e = ring[(jv + 1) % k_ring] - v
        e = e / max(float(np.hypot(*e)), 1e-9)
        _box_tiled(m, c, e, STAIR_TOWER_M / 2, STAIR_TOWER_M / 2, zs[0] - FOUNDATION_M, zs[-1] + STAIR_TOWER_UP,
                   mat, 2.0)


def garage_drives(items, lines, x0, y0):
    """Driveway chunks (road tile chunk dicts, draped by roadmesh) into each open parking structure: from the
    middle of its outline's edge nearest a road out to that road's edge (report #51, 2026-10-17: "No ramp leading
    into the garage")."""
    out = []
    if not lines:
        return out
    segs = []
    for ln in lines:
        pts, half = np.asarray(ln[0], np.float64)[:, :2], ln[1]
        for k in range(len(pts) - 1):
            segs.append((pts[k], pts[k + 1], half))
    if not segs:
        return out
    A = np.array([s[0] for s in segs])
    B = np.array([s[1] for s in segs])
    H = np.array([s[2] for s in segs])
    D = B - A
    L2 = np.maximum((D * D).sum(1), 1e-12)
    for spec, gmin, gmax in items:
        plan_ = deck_plan(spec, gmin, gmax)
        if plan_ is None:
            continue
        ring = plan_[0] + np.array([x0, y0])
        best = None
        for j in range(len(ring)):
            p, q = ring[j], ring[(j + 1) % len(ring)]
            if np.hypot(*(q - p)) < 2 * DRIVEWAY_W:
                continue
            mid = (p + q) / 2
            t = np.clip(((mid - A) * D).sum(1) / L2, 0.0, 1.0)
            near = A + t[:, None] * D
            dd = np.hypot(*(mid - near).T) - H
            k = int(np.argmin(dd))
            if best is None or dd[k] < best[0]:
                best = (float(dd[k]), mid, near[k], H[k], p, q)
        if best is None or not (0.5 < best[0] < GARAGE_DRIVE_MAX_M):
            continue
        _, mid, qn, half, p, q = best
        d = (q - p) / max(float(np.hypot(*(q - p))), 1e-9)
        inside = np.array([-d[1], d[0]])
        v = mid - qn
        dl = float(np.hypot(*v))
        p0 = mid + inside * 3.0             # a few metres in, onto the ground deck
        p1 = qn + v / max(dl, 1e-9) * (half + 0.1)
        n = max(2, int(np.hypot(*(p1 - p0)) // 4.0) + 1)
        pts = [p0 + (p1 - p0) * (j / (n - 1)) for j in range(n)]
        out.append({"m": DRIVEWAY["m"], "w": GARAGE_DRIVE_W, "tl": DRIVEWAY["tl"], "p": DRIVEWAY["p"],
                    "n": [[round(float(pp[0]), 3), round(float(pp[1]), 3), 0.0] for pp in pts]})
    return out


def build(specs_and_ground, lite=False):
    """The tile's mesh from [(Spec, ground z min, ground z max)] with rings in the tile's local frame:
    (positions, uvs, tris, tri material indices, MATERIALS) or None if empty. lite (far tiles): no
    cornices, whole walls instead of one quad per bay."""
    m = Mesh()
    for spec, gmin, gmax in specs_and_ground:
        if spec.cls == "toll_booth":        # a stock prop (toll_booth_rows)
            continue
        if spec.cls == landmarks.LANDMARK:
            landmarks.build(m, spec, gmin, _landmark_token(spec), lite, _landmark_tokens())
            continue
        mat_idx = _Looks(spec.seed)
        if spec.fr_house:
            import francehouse_mesh
            francehouse_mesh.draw(m, spec, gmin, gmax, mat_idx, lite)
            continue
        if spec.cls in BRIDGE_CLASSES:
            _steel_frame(m, spec, gmin, mat_idx[spec.facade if spec.facade in LOOKS else STEEL], lite)
            continue
        if spec.cls == "ship":
            ships.draw(m, spec, gmin, gmax, mat_idx, lite)
            continue
        ring = drawn_ring(spec)
        mh = spec.min_h or 0.0
        if spec.deck and not lite and _parking_deck(m, spec, gmin, gmax, mat_idx):
            continue
        z0 = gmin - (SKIRT_LITE_M if lite else SKIRT_M) if mh <= 0.5 else base_z(spec, gmin, gmax) + mh
        z_eave = base_z(spec, gmin, gmax) + mh + spec.height
        _walls(m, spec, ring, z0, z_eave, mat_idx, lite)
        if mh > 0.5:                                  # a part standing on another / over a street: its underside
            m.poly([(p[0], p[1], z0) for p in ring[::-1]], [(0.002, 0.995)] * len(ring), mat_idx[spec.facade])
        if spec.cls in OPEN_CANOPY_CLASSES and 0.5 < mh <= CANOPY_POSTS_MAX_M:
            _canopy_posts(m, spec, gmin, z0, mat_idx[spec.facade])
        _roof(m, spec, ring, z_eave, mat_idx, lite)
        if spec.roof == "flat" and not spec.plain and spec.floors <= 6 and not lite:
            _cornice(m, spec, ring, z_eave, mat_idx)
        if not lite:
            _clutter(m, spec, ring, z_eave, mat_idx)
        if spec.steeple:
            _steeple(m, spec, z_eave, mat_idx)
        if spec.stoop and not lite:
            _stoop(m, spec, ring, gmin, gmax, mat_idx)
        if spec.porch and spec.door and not lite:
            _porch(m, spec, gmax, mat_idx)
        if spec.sign is not None and spec.roof == "flat" and not lite:
            if spec.windows is False:           # a big box: over its storefront band (_plain_wall_with_windows)
                z_top = min(gmax + GROUND_FLOOR_SHOP_M + 1.15, z_eave)
            else:
                z_top = z_eave - (spec.floors - 1) * spec.floor_h if spec.floors > 1 else z_eave
            _signs(m, spec, ring, z_top, mat_idx)
    if m.n == 0:
        return None
    P, UV, T, M, C = m.arrays()
    return P, UV, T, M, MATERIALS, C


def _steel_frame(m, spec, gmin, mat, lite=False, legs_only=False):
    """A bridge structure (BRIDGE_CLASSES) as an open steel frame on its oriented box: a leg at each corner, bracing
    along the short sides every STEEL_BAY_M (beside the decks running along them), cross beams over the long sides
    only near the top (STEEL_TOP_SHARE, over any deck); a narrow footprint is one pier. lite: legs and top only."""
    c, u, hl, hw = oriented_box(np.asarray(spec.ring, np.float64))
    v = np.array([-u[1], u[0]])
    z0, z1 = gmin - FOUNDATION_M, gmin + spec.height
    if hl < 6.0:                                            # a pier
        _box(m, c, u, hl, hw, z0, z1, mat)
        return
    if hw < 3.0:    # a thin wall across the decks (the Queensboro Bridge's towers, 33 x 4.6 m): a leg at each end
        t = float(min(hw, STEEL_LEG_MAX))
        for sx in (-1, 1):
            _box(m, c + u * sx * (hl - t), u, t, hw, z0, z1, mat)
        if not legs_only:
            b = t * 0.6
            top = gmin + STEEL_TOP_SHARE * spec.height
            for z in [z1 - b] + ([] if lite else list(np.arange(z1 - b - STEEL_BAY_M / 2, top, -STEEL_BAY_M / 2))):
                _box(m, c, u, hl - 2 * t, b, z - b, z + b, mat)
        return
    t = float(np.clip(hw * 0.5, 0.5, STEEL_LEG_MAX))        # leg half size
    corners = [c + u * sx * (hl - t) + v * sy * (hw - t) for sx in (-1, 1) for sy in (-1, 1)]
    for k in corners:
        _box(m, k, u, t, t, z0, z1, mat)
    if legs_only:
        return
    b = t * 0.6                                             # beam half size
    levels = [z1 - b] if lite else list(np.arange(gmin + STEEL_BAY_M, z1 - b, STEEL_BAY_M)) + [z1 - b]
    for z in levels:
        for sx in (-1, 1):          # along the short sides (along v): beside a deck running through the frame
            _box(m, c + u * sx * (hl - t), v, hw - 2 * t, b, z - b, z + b, mat)
        if z >= gmin + STEEL_TOP_SHARE * spec.height:       # across the top only
            for sy in (-1, 1):
                _box(m, c + v * sy * (hw - t), u, hl - 2 * t, b, z - b, z + b, mat)


def collision_mesh(specs_and_ground):
    """The tile's buildings for physics only: one quad per wall and a flat lid at the eaves (pitched roofs on their
    oriented box), parking decks as drawn; (positions, uvs, tris, tri materials, MATERIALS, colours) or None.
    The drawn mesh (windows a quad per bay and floor, cornices, signs) cost the game's collision rebuild ~30 ms of
    a 40 ms rebuild in Fells Point (2026-10-15), every few seconds while driving; this one is ~1/20 of its triangles."""
    m = Mesh()
    tok = (MATERIALS.index(PANEL), (128, 128, 128, 255))
    zero = (0.0, 0.0)
    for spec, gmin, gmax in specs_and_ground:
        if spec.fr_house:
            import francehouse_mesh
            francehouse_mesh.draw(m, spec, gmin, gmax, _Looks(spec.seed), collision=True)
            continue
        if spec.cls in BRIDGE_CLASSES:          # its legs (the decks run between them)
            _steel_frame(m, spec, gmin, tok, True, legs_only=True)
            continue
        if spec.cls == landmarks.LANDMARK:      # (its legs and decks: walked and driven under)
            landmarks.collision(m, spec, gmin, tok)
            continue
        if deck_plan(spec, gmin, gmax) is not None:
            _parking_deck(m, spec, gmin, gmax, _Looks(spec.seed))
            continue
        ring = np.asarray(spec.ring, np.float64)
        if spec.roof != "flat" and spec.roof not in DOME_ROOFS:   # (a dome's drum: its own round outline, as drawn)
            c, u, hl, hw = oriented_box(ring)
            v = np.array([-u[1], u[0]])
            ring = np.array([c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw, c - u * hl + v * hw])
        if _area2(ring) < 0:
            ring = ring[::-1]
        mh = spec.min_h or 0.0
        z0 = gmin - FOUNDATION_M if mh <= 0.5 else base_z(spec, gmin, gmax) + mh
        z1 = base_z(spec, gmin, gmax) + mh + (spec.floor_h if spec.cls == "ship" else spec.height)  # (a ship: its deck)
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1), zero, zero, zero, zero, tok)
        m.poly([(p[0], p[1], z1) for p in ring], [zero] * len(ring), tok)
        if mh > 0.5:                # over a street: its underside
            m.poly([(p[0], p[1], z0) for p in ring[::-1]], [zero] * len(ring), tok)
    if m.n == 0:
        return None
    P, UV, T, M, C = m.arrays()
    return P, UV, T, M, MATERIALS, C
