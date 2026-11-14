"""Race track kerbs (the France fork; the user, 2026-11-14, at the Circuit de Barcelona-Catalunya: "the kerbs do not
exist"): red and white kerbs along a raceway's bends, as the road tiles' mesh chunks (roads._chunk's format).

OpenStreetMap seldom maps a circuit's kerbs, so they are placed by the track's own shape: a bend (the heading turning
faster than one over BEND_R_M for at least BEND_TURN_DEG in all) gets an inside kerb round its apex and an outside
kerb from its last part out past its exit (EXIT_M), as a circuit's are laid. The kerb lies flat on the track's edge
(KERB_W wide, KERB_INSET_M of it over the asphalt, KERB_LIFT_M over it), its texture red and white every STRIPE_M
(tools/build_race_kerbs.py: bne_race_kerb). Pure (no I/O): chunks(roads, box) from build_tile's roads.
"""
import numpy as np

import dnp

KERB_ON = True
MATERIAL = "bne_race_kerb"
KERB_W = 1.3            # m across
KERB_INSET_M = 0.25     # m of it over the track's own asphalt (no gap at the edge)
KERB_LIFT_M = 0.03      # m over the track
STRIPE_M = 1.0          # m of red, then of white: the texture's length (tl) is two of them
PRIORITY = 4            # (over the asphalt, 20, and its edge lines' 2 under: roads.EDGE_LINE)
STEP_M = 2.0            # m between the samples of a track's line
SMOOTH_M = 12.0         # m the heading's rate is averaged over
BEND_R_M = 250.0        # a bend: the line turning faster than a circle of this radius ...
BEND_TURN_DEG = 15.0    # ... and this much in all
APEX = (0.25, 0.85)     # the inside kerb: this share of the bend's length to this
EXIT_FROM = 0.6         # the outside kerb: from this share of the bend ...
EXIT_M = 25.0           # ... to this far past its end
MIN_KERB_M = 8.0        # m: a shorter kerb is left out
LAID_M = 1.0            # m: a kerb this close to one laid already is the same one (layouts sharing their asphalt)


def _resample(line, step):
    p = np.asarray(line, np.float64)
    seg = dnp.hypot(*np.diff(p[:, :2], axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < step * 3:
        return None, None
    t = np.arange(0.0, s[-1] + 1e-9, step)
    out = np.stack([np.interp(t, s, p[:, k]) for k in range(p.shape[1])], 1)
    return out, t


def bends(xy, step=STEP_M):
    """[(i0, i1, side)] of the samples xy (n, 2) every step m: each bend's first and last sample and the side of its
    inside (+1 left, -1 right). Pure."""
    if len(xy) < 5:
        return []
    d = np.diff(xy, axis=0)
    head = np.unwrap(dnp.arctan2(d[:, 1], d[:, 0]))
    rate = np.gradient(head) / step                      # rad per m
    k = max(1, int(round(SMOOTH_M / step)))
    rate = dnp.convolve(rate, np.ones(k) / k, mode="same")
    on = np.abs(rate) > 1.0 / BEND_R_M
    out = []
    i = 0
    n = len(rate)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1] and np.sign(rate[j + 1]) == np.sign(rate[i]):
            j += 1
        turn = abs(float(head[j] - head[i])) if j > i else 0.0
        if np.degrees(turn) >= BEND_TURN_DEG:
            out.append((i, j + 1, 1 if rate[(i + j) // 2] > 0 else -1))
        i = j + 1
    return out


def _offset(p, side, off):
    """p (k, 3) moved off m to its side (+1 left of its direction)."""
    d = np.gradient(p[:, :2], axis=0)
    norm = dnp.hypot(d[:, 0], d[:, 1])
    norm[norm == 0] = 1.0
    left = np.stack([-d[:, 1] / norm, d[:, 0] / norm], 1)
    q = p.copy()
    q[:, :2] = p[:, :2] + left * side * off
    q[:, 2] = p[:, 2] + KERB_LIFT_M
    return q


def kerb_lines(line, width):
    """The kerbs' centre lines [(k, 3)] of one track centre line (k, 3) of width m. Pure."""
    p, _t = _resample(line, STEP_M)
    if p is None:
        return []
    off = width / 2.0 + KERB_W / 2.0 - KERB_INSET_M
    n = len(p)
    out = []
    for i0, i1, side in bends(p[:, :2]):
        L = i1 - i0
        a, b = i0 + int(APEX[0] * L), i0 + int(np.ceil(APEX[1] * L))
        e0, e1 = i0 + int(EXIT_FROM * L), min(n - 1, i1 + int(EXIT_M / STEP_M))
        for lo, hi, s in ((a, b, side), (e0, e1, -side)):
            if (hi - lo) * STEP_M < MIN_KERB_M:
                continue
            out.append(_offset(p[lo:hi + 1], s, off))
    return out


def _dist_to_line(p, xy):
    """Each point's distance (m) to the polyline xy (k, 2). Pure."""
    a, b = xy[:-1], xy[1:]
    d = b - a
    ll = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip(((p[:, None, :] - a[None]) * d[None]).sum(2) / ll[None], 0.0, 1.0)
    q = a[None] + t[..., None] * d[None]
    return dnp.hypot(p[:, None, 0] - q[..., 0], p[:, None, 1] - q[..., 1]).min(1)


def _off_tracks(kerb, others, laid=()):
    """kerb (k, 3) cut where it lies on another track's asphalt (others: [(centre line (m, 2), half width)]: a circuit
    mapped in several layouts has its tracks crossing: the kerbs ran across them, Barcelona 2026-11-14; a layout sharing
    the asphalt runs beside it, its kerb out there too) and where a kerb laid already runs (laid: [(k, 2)], the shared
    stretches' second copy): its pieces of MIN_KERB_M or more. Pure."""
    if (not others and not laid) or len(kerb) < 2:
        return [kerb]
    on = np.zeros(len(kerb), bool)
    for xy, hw in others:
        if len(xy) < 2:
            continue
        on |= _dist_to_line(kerb[:, :2], xy) < hw - KERB_INSET_M
    for xy in laid:
        if len(xy) >= 2:
            on |= _dist_to_line(kerb[:, :2], xy) < LAID_M
    out, start = [], None
    for i, o in enumerate(list(on) + [True]):
        if not o and start is None:
            start = i
        elif o and start is not None:
            piece = kerb[start:i]
            if len(piece) >= 2 and (i - start - 1) * STEP_M >= MIN_KERB_M:
                out.append(piece)
            start = None
    return out


def chunks(roads, box):
    """The kerbs of the race tracks among roads ([(fid, spec, centre line (k, 3), lon)], build_tile's) as mesh chunks
    cut to box (x0, y0, x1, y1). Pure."""
    if not KERB_ON:
        return []
    import roads as roadsmod
    tracks = [(fid, np.asarray(line, np.float64)[:, :2], float(spec.width) / 2.0) for fid, spec, line, _lon in roads
              if getattr(spec, "cls", None) == "raceway" and getattr(spec, "kind", None) == "paved" and len(line) >= 2]
    out, laid = [], []
    for fid, spec, line, _lon in roads:
        if getattr(spec, "cls", None) != "raceway" or getattr(spec, "kind", None) != "paved":
            continue
        others = [(xy, hw) for f2, xy, hw in tracks if f2 != fid]
        for k0 in kerb_lines(np.asarray(line, np.float64), float(spec.width)):
            for k in _off_tracks(k0, others, laid):
                laid.append(k[:, :2])
                for piece in roadsmod.clip_polyline(k, *box):
                    for c in roadsmod.split(roadsmod.densify(piece)):
                        if len(c) >= 2:
                            # (fid: the track's own, so the mesh follows its ground as the track does: roadmesh's RW)
                            out.append({"m": MATERIAL, "w": KERB_W, "tl": 2.0 * STRIPE_M, "p": PRIORITY,
                                        "fid": str(fid), "n": [[round(float(v), 3) for v in q[:3]] for q in c]})
    return out
