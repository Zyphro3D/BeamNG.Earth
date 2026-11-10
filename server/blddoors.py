"""Procedural buildings: front doors and what goes with them (split out of buildings.py): company wall signs and front
fences.
"""
import math
import numpy as np
import fast
from bldbase import (
    BACK_ROAD_CLASSES,
    BAND_BRICK,
    BAND_OFFICE,
    BAND_OFFICE2,
    CHURCH_CLASSES,
    DOOR_OUT_M,
    DOOR_ROAD_M,
    DRIVEWAY,
    DRIVEWAY_W,
    FENCE_BACK_M,
    FENCE_CLEAR_M,
    FENCE_MIN_YARD_M,
    FENCE_SEG_M,
    FENCE_SHARE,
    G_DOOR,
    G_FENCE,
    MAIN_STREET_CLASSES,
    MAIN_STREET_SETBACK_M,
    MAIN_STREET_SHOPS,
    METALWALL,
    PORCH_DECK_M,
    RESTAURANT_CLASSES,
    STOOP_H,
    STOOP_SHARE,
    WALL_SIGNS,
    WALL_SIGN_SHARE,
    _hash01,
)
from bldmesh import (
    _inside,
    door_ground,
    drawn_ring,
)


def doors(items, lines, x0, y0, drives=None):
    """Prop rows (trees.lua rows, group "door") of a front door per house: on the wall whose middle is
    nearest a road and faces it, a row house's to one side, others' in the middle. lines: [(centreline
    (k, 2) in the anchor frame, half width[, class, urban])]. A building close to an urban main street gets
    a shopfront ground floor instead (spec.shop, most of them). drives (a list) gets a driveway chunk (road
    tile chunk dicts, draped by roadmesh) from each detached house's front to the road's edge."""
    if not lines:
        return []
    segs, smain, shalf, sminor = [], [], [], []
    for ln in lines:
        pts, half = ln[0], ln[1]
        main = len(ln) > 3 and ln[3] and ln[2] in MAIN_STREET_CLASSES     # an urban main street
        p = np.asarray(pts, np.float64)
        if len(p) >= 2:
            segs.append(np.concatenate([p[:-1], p[1:]], 1))
            smain.append(np.full(len(p) - 1, bool(main)))
            shalf.append(np.full(len(p) - 1, float(half)))
            sminor.append(np.full(len(p) - 1, len(ln) > 2 and ln[2] in BACK_ROAD_CLASSES))
    if not segs:
        return []
    S = np.concatenate(segs)                             # (n, 4) x0 y0 x1 y1
    SMAIN, SHALF, SMINOR = np.concatenate(smain), np.concatenate(shalf), np.concatenate(sminor)
    A, B = S[:, :2], S[:, 2:]
    D = B - A
    L2 = np.maximum((D * D).sum(1), 1e-12)
    rows = []
    others = [(spec, spec.ring + np.array([x0, y0])) for spec, _, _ in items if (spec.min_h or 0.0) <= 0.5]
    for spec, gmin, gmax in items:
        no_door = False
        if spec.plain or spec.floors >= 7 or spec.band in (BAND_OFFICE, BAND_OFFICE2) or spec.wall:
            if not spec.wall or spec.plain or spec.floors >= 7:
                no_door = True
        # shops and big boxes: their sign faces the street, not the alley or the parking aisle behind
        storefront = (spec.shop or spec.windows is False) and not spec.plain and spec.roof == "flat"
        if no_door and not storefront:
            continue
        ring = drawn_ring(spec) + np.array([x0, y0])
        best = None
        # every wall's middle against every road segment at once (a wall at a time: ~6 numpy calls per wall), the same
        # numbers: each wall's nearest segment k, the nearest point on it Q and its distance
        nxt = np.concatenate((ring[1:], ring[:1]))
        E = nxt - ring
        MID = (ring + nxt) / 2
        T = np.clip(((MID[:, None, :] - A[None]) * D[None]).sum(2) / L2[None], 0, 1)
        Q = A[None] + T[..., None] * D[None]
        DIST = np.hypot(*(Q - MID[:, None, :]).transpose(2, 0, 1))
        K = DIST.argmin(1)
        for i in range(len(ring)):
            a = ring[i]
            e = E[i]
            Le = float(math.hypot(*e))
            if Le < 2.5:
                continue
            mid = MID[i]
            dist = DIST[i]
            k = int(K[i])
            if dist[k] > DOOR_ROAD_M:
                continue
            out = np.array([e[1], -e[0]]) / Le
            if float(np.dot(Q[i, k] - mid, out)) <= 0:
                continue                                 # the wall faces away from that road
            rank = (bool(SMINOR[k]) if storefront else False, float(dist[k]))
            if best is None or rank < best[0]:
                best = (rank, a, e / Le, Le, out, i, bool(SMAIN[k]) and dist[k] < SHALF[k] + MAIN_STREET_SETBACK_M,
                        k)
        if best is None:
            continue
        _, a, d, Le, out, edge, on_main, kseg = best
        if spec.windows is False:
            spec.front = edge                   # a big box: its storefront faces the road
            if storefront:
                spec.sign = edge
            rows.extend(_sign_in_open(_wall_sign(spec, a, d, Le, out, gmax), spec, d, out, others))
            continue
        if spec.cls in RESTAURANT_CLASSES and spec.floors == 1:
            rows.extend(_sign_in_open(_wall_sign(spec, a, d, Le, out, gmax), spec, d, out, others))
        if spec.shop:
            if storefront and not spec.wall:
                spec.sign = edge                # the sign band over the street wall's shopfront
            continue
        if no_door:
            continue
        if on_main and not spec.wall and spec.roof == "flat" and spec.cls not in CHURCH_CLASSES \
                and _hash01(spec.seed, "shop") < MAIN_STREET_SHOPS:
            spec.shop = True                    # a main street's ground floor: a shopfront, no house door
            if not spec.plain:
                spec.sign = edge
            continue
        narrow = Le < 9.0
        f = (0.22 if _hash01(spec.seed, "door") < 0.5 else 0.78) if narrow else 0.5
        spec.door = (edge, Le * f)          # _walls leaves the ground floor's band plain there
        # brick row houses (an eastern city's): white marble steps up to the door
        if narrow and spec.band == BAND_BRICK and spec.roof == "flat" and _hash01(spec.seed, "stoop") < STOOP_SHARE:
            spec.stoop = (edge, Le * f)
        if drives is not None and spec.wall and Le >= 7.0:
            # the driveway beside the door, straight out to the road's edge
            g = 0.82 if f <= 0.5 else 0.18
            p0 = a + d * (Le * g) + out * 0.2
            t = float(np.clip(((p0 - A[kseg]) * D[kseg]).sum() / L2[kseg], 0, 1))
            q = A[kseg] + t * D[kseg]
            v = p0 - q
            dl = float(np.hypot(*v))
            if SHALF[kseg] + 1.0 < dl < DOOR_ROAD_M:
                p1 = q + v / dl * (SHALF[kseg] + 0.1)
                n = max(2, int(dl // 4.0) + 1)
                pts = [p0 + (p1 - p0) * (j / (n - 1)) for j in range(n)]
                drives.append({"m": DRIVEWAY["m"], "w": DRIVEWAY_W, "tl": DRIVEWAY["tl"], "p": DRIVEWAY["p"],
                               "n": [[round(float(pp[0]), 3), round(float(pp[1]), 3), 0.0] for pp in pts]})
        if spec.roof != "flat" and not narrow and _hash01(spec.seed, "fence") < FENCE_SHARE:
            rows.extend(_front_fence(spec, a, d, Le, f, A[kseg], D[kseg], L2[kseg], SHALF[kseg], gmin,
                                     drives is not None and spec.wall and Le >= 7.0, (A, B, SHALF)))
        p = a + d * (Le * f) + out * DOOR_OUT_M
        gd = door_ground(spec, edge, Le * f, gmax) if spec.stoop else gmax     # a stoop's door: over its own ground
        base = gd + (STOOP_H if spec.stoop else 0.0) + (PORCH_DECK_M if spec.porch else 0.0)   # (the deck / stoop)
        if spec.fr_house:
            continue                    # the regional model contains its own fitted front door
        rows.append([G_DOOR, round(float(p[0]), 2), round(float(p[1]), 2), round(base, 2),
                     round(math.degrees(math.atan2(d[1], d[0])) % 360.0, 1), 1.0, 2.2,
                     round(_hash01(spec.seed, "dv"), 3)])
    return rows


def _sign_in_open(rows, spec, d, out, others):
    """rows (a _wall_sign) kept only when the panel and the air in front of it are clear of the other buildings:
    a sign on a wall hidden behind a neighbour stood out through the neighbour at the wall's angle (Stelvio
    pass, a big box's sign through the shop in front of it, the user's report 2026-10-17)."""
    if not rows:
        return rows
    _, x, y, _, _, _, h, _ = rows[0]
    w = WALL_SIGNS[next(k for k, v in WALL_SIGNS.items() if v[0] == rows[0][0])][1] * h / \
        WALL_SIGNS[next(k for k, v in WALL_SIGNS.items() if v[0] == rows[0][0])][2]
    c = np.array([x, y])
    probes = [c + d * (w / 2 * f) + out * o for f in (-1.0, 0.0, 1.0) for o in (0.0, 1.5)]
    for other, ring in others:
        if other is spec:
            continue
        if any(_inside(ring, p) for p in probes):
            return []
    return rows


def _wall_sign(spec, a, d, Le, out, gmax):
    """A company sign centred high on a big box's or restaurant's street wall (WALL_SIGNS), if it fits. Only in
    North America: the firms are West Coast USA's (a big box in France carried an English slogan on its front)."""
    if not spec.us_signs or _hash01(spec.seed, "wallsign") >= WALL_SIGN_SHARE:
        return []
    if spec.cls in RESTAURANT_CLASSES:
        key = "burger"
    elif spec.wall and str(spec.wall).startswith(METALWALL):
        key = "logistics" if _hash01(spec.seed, "ws") < 0.5 else "rentabox"
    else:
        key = "lemon"
    group, w, h0, thin_x = WALL_SIGNS[key]
    h = min(h0, spec.height * 0.45)                     # scaled down to the wall (trees.lua scales to the height)
    w *= h / h0
    if h < 1.8 or Le < w + 2.0:
        return []
    zc = gmax + spec.height - 0.6 - h / 2               # its top 0.6 m under the parapet
    p = a + d * (Le / 2) + out * 0.35
    # the logo faces the model's local -x (Lemon, Logistics, Rentabox: seen in game 2026-10-14) or its -y
    yaw = math.degrees(math.atan2(out[1], out[0])) + (180.0 if thin_x else 90.0)
    return [[group, round(float(p[0]), 2), round(float(p[1]), 2), round(float(zc), 2), round(yaw % 360.0, 1), 1.0,
             round(h, 2), 0.0]]


def _front_fence(spec, a, d, Le, f, A, D, L2, half, z, driveway, roads=None):
    """Prop rows of a fence along the front of a house's yard, parallel to the road at the back of the sidewalk,
    a little wider than the house, with gaps for the path to the door and the driveway. roads: (starts (n, 2), ends
    (n, 2), half widths (n,)) of the roads near: no piece within FENCE_CLEAR_M of their edges."""
    mid = a + d * (Le / 2)
    t = float(np.clip(((mid - A) * D).sum() / L2, 0, 1))
    q = A + t * D
    v = mid - q
    dist = float(np.hypot(*v))
    if dist - (half + FENCE_BACK_M) < FENCE_MIN_YARD_M:
        return []
    nv = v / dist
    dr = D / math.sqrt(L2)
    centre = q + nv * (half + FENCE_BACK_M)
    yaw = round(math.degrees(math.atan2(dr[1], dr[0])) % 360.0, 1)
    gaps = [float((a + d * (Le * f) - centre) @ dr)]                          # the path to the door
    if driveway:
        gaps.append(float((a + d * (Le * (0.82 if f <= 0.5 else 0.18)) - centre) @ dr))
    n = max(1, int(math.ceil((Le + 3.0) / FENCE_SEG_M)))
    rows = []
    for k in range(n):
        sk = (k - (n - 1) / 2) * FENCE_SEG_M
        if any(abs(sk - g) < FENCE_SEG_M / 2 + 0.6 for g in gaps):
            continue
        p = centre + dr * sk
        if roads is not None and len(roads[0]):
            ends = np.array([p - dr * (FENCE_SEG_M / 2), p, p + dr * (FENCE_SEG_M / 2)])
            fast.require(0, "points_near_segments")
            if np.asarray(fast._ef.points_near_segments(ends, np.ascontiguousarray(roads[0], np.float64),
                                                        np.ascontiguousarray(roads[1], np.float64),
                                                        np.asarray(roads[2], np.float64) + FENCE_CLEAR_M)).any():
                continue
        rows.append([G_FENCE, round(float(p[0]), 2), round(float(p[1]), 2), round(float(z), 2), yaw, 1.0, 1.3,
                     round(_hash01(spec.seed, "fr", k), 3)])
    return rows


def _suppress_under_models(items, radii, x0, y0):
    """items (tile_buildings', rings relative to (x0, y0)) with every footprint whose centre falls inside one
    of radii's circles dropped: a baked 3dmr.eu model (_custom_tile_entries' radii, [(id, cx, cy, radius)]
    in the anchor frame) stands there instead, and a procedural stub under/through it would poke
    out of it (a user's report at Baghdad's Al-Shaheed Monument: two OSM "sector" footprints, the domes'
    lower tiers, stayed and intersected model 24's real mesh once it loaded); every model gets its own circle
    over any procedural buildings standing where it is placed. Toll booths (spec.cls "toll_booth": the stock
    prop a toll island is drawn as, toll_booth_rows) are never suppressed - a monument standing near a toll
    plaza should not eat its booth. Applied one model at a time (not a single logged count) so each model's
    own effect can be told apart in the log."""
    import logging

    log = logging.getLogger("earthd.buildings")
    for cid, cx, cy, r in radii:
        kept = []
        n = 0
        for it in items:
            spec = it[0]
            c = spec.ring.mean(0) + (x0, y0)
            if (
                spec.cls != "toll_booth"
                and (c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= r * r
            ):
                n += 1
                continue
            kept.append(it)
        items = kept
        if n:
            log.info(
                "3dmr model %s: suppressed %d procedural building(s) under it", cid, n
            )
    return items
