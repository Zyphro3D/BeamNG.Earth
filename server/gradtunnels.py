"""Road grading: tunnels (split out of grading.py's _Planner): subsea parts, tunnel sections and mouths, the roads
over tunnels and raceway ground.
"""
import math
from collections import OrderedDict
import numpy as np
import roads as roadsmod
from gradbase import (
    MOUTH_PLAN_M,
    Plan,
    PlanPack,
    RACEWAY_SMOOTH_ON,
    ROAD_LIFT,
    TUNNEL_KEEP_IN_M,
    TUNNEL_KEEP_M,
    TUNNEL_TAG,
    VERTEX_STEP,
    _twin_trimmed,
    grade_points,
    log,
    raceway_ground,
    raceway_surface,
    resample,
    shoulder,
)


class _TunnelsMixin:
    """_Planner's tunnels (split out of grading.py; _Planner inherits it)."""

    def _chains_for(self):
        """A roads.TunnelProfiles of this frame for its chain questions (drowned): the same chains, from the same map
        cells, as the road tiles' and terrain blocks' own."""
        if getattr(self, "_chains", None) is None:
            md = self.md
            t = roadsmod.TunnelProfiles(self.frame, lambda key: md.cell(key, layers=("roads",), kind="demand")
                                        .layers.get("roads"), self.elev)
            t.wet = self.wet
            self._chains = t
        return self._chains

    def _drowned(self, key, ring):
        """roads.TunnelProfiles.drowned for tunnel part key (ring: its lon / lat points): the process's tunnel chain
        store's (the road tiles' and blocks', blockworker: store.tunnel_store), whose chains are resolved once for all
        of them, when it is there; else this frame's own (_chains_for)."""
        get = getattr(self.store, "tunnel_store", None) if getattr(self, "store", None) is not None else None
        cs = get() if callable(get) else None
        if cs is not None:
            try:
                st = cs.store_at(float(ring[0][1]), float(ring[0][0]))
                if st.wet is not None:
                    return st.drowned(key, ring)
            except Exception as e:      # (this frame's own then)
                log.debug("grading: the chain store's drowned: %s: %s", type(e).__name__, e)
        return self._chains_for().drowned(key, ring)

    def _subsea_part(self, feats, i, j, ring, portals):
        """The canonical region's solved wet profile, including its dry approaches. None: ordinary tunnel rules.

        The grading reader uses the same raw source, region frame and native solve as the mesh reader. It never
        calls a grader from a profile endpoint: planning the tunnel must not recursively plan itself.
        """
        import geotiles
        holder = self.store if self.store is not None else self
        with holder.lock:
            cs = getattr(holder, "_subsea_chains", None)
            if cs is None:
                raw = self.tunnel_raw or self.elev
                md, wet, water = self.md, self.wet, self.water
                def setup(t):
                    t.wet, t.water, t.raw = wet, water, raw
                cs = holder._subsea_chains = roadsmod.ChainStore(roadsmod.TunnelProfiles,
                    lambda k: md.cell(k, layers=("roads",), kind="demand").layers.get("roads"), raw, setup)
        st = cs.store_at(float(ring[0][1]), float(ring[0][0]))
        k = (str(feats.ids[i]), j)
        if not st.drowned(k, ring):
            return None
        whole = roadsmod.BridgeProfiles.parts_todo(st, [(k, feats, i)])
        pid = st._part_id(k)
        with st.lock:
            supported = pid in st._subsea
            secs = list(st.sections.get(pid, ()))
        if not supported:
            return []
        out = []
        for _pid, spec, xy, z, _lon in whole:
            la, lo, h = geotiles.world_to_geodetic(st.frame, np.c_[xy, z])
            e, n, _ = self.frame.geodetic_to_enu(la, lo, 0.)
            xy = np.c_[e, n]
            for index, (a, b, kind) in enumerate(secs):
                if b <= a: continue
                fid = f"{pid}#{index}"
                # Plans are the terrain datum: apply(..., ROAD_LIFT) supplies the physical road surface.
                # The native tunnel profile already IS that surface, so do not add the lift a second time.
                out.append((fid, spec, xy[a:b + 1], np.asarray(h)[a:b + 1] - ROAD_LIFT, kind))
                if kind and portals is not None:
                    for at, is_mouth in zip((a, b), st.mouth(fid)):
                        portals[(fid, tuple(np.round(xy[at], 3)))] = is_mouth
        return out

    def _tunnel_sections(self, feats, key=None, road_plans=(), portals=None):
        """[(fid, spec, xy, z, kind)] of the tunnels' sections (roads.tunnel_sections on the ungraded ground, its
        ends graded by road_plans: the cell's road Plans, as roads.TunnelProfiles.ends grades them with the roads'
        plans alone): kind 0 an open road at z, 1 a low arch, 2 a full tunnel."""
        out = []
        ts = roadsmod.tunnel_specs(feats)
        pack = None                     # (the road plans packed once: per tunnel part, ~6 ms each, 0.9 s of 8 tiles)
        over = None                     # (the ground roads near the tunnels, as roads.TunnelProfiles covers them)
        for i in range(len(feats)):
            if ts[i] is None:
                continue
            for j, ring in enumerate(roadsmod.tunnel_parts(feats, i)):
                ring = self._longest(str(feats.ids[i]), j, ring, key)
                ll = np.asarray(ring, np.float64)
                if len(ll) < 2:
                    continue
                if self.water is not None:
                    solved = self._subsea_part(feats, i, j, ring, portals)
                    if solved is not None:
                        out.extend(solved)
                        continue
                E, N, _ = self.frame.geodetic_to_enu(ll[:, 1], ll[:, 0], 0.0)
                xy, s = resample(np.stack([np.asarray(E), np.asarray(N)], axis=1), VERTEX_STEP)
                if len(xy) < 2 or s[-1] <= 0:
                    continue
                la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
                if self.wet is not None and \
                        (np.asarray(self.wet(la, lo)) > 0.5).mean() > roadsmod.TUNNEL_WET_SHARE:
                    continue
                if self.wet is not None and self._drowned((str(feats.ids[i]), j), ring):
                    continue            # (a dry piece of an underwater tunnel: left out with it, GitHub #128)
                g = np.asarray(self.elev(la, lo), np.float64)
                if road_plans:
                    g = g.copy()
                    pack = PlanPack(list(road_plans)) if pack is None else pack
                    g[[0, -1]] = grade_points(xy[[0, -1], 0], xy[[0, -1], 1], g[[0, -1]], pack)
                floor, kinds = roadsmod.tunnel_sections(s, g)
                kinds = roadsmod.portal_cover(kinds, g - floor, ts[i])
                if over is None:
                    over = self._roads_over_tunnels(feats, ts)
                kinds = roadsmod.cover_roads(xy, s, kinds, *over)
                i0 = 0
                while i0 < len(kinds):
                    i1 = i0
                    while i1 + 1 < len(kinds) and kinds[i1 + 1] == kinds[i0]:
                        i1 += 1
                    a = max(0, i0 - 1)          # sharing the end sample with the section before
                    if i1 > a:
                        out.append((str(feats.ids[i]), ts[i], xy[a:i1 + 1], floor[a:i1 + 1], int(kinds[i0])))
                    i0 = i1 + 1
        return out

    def _roads_over_tunnels(self, feats, ts):
        """([ground lines (k, 2) in the frame], [carriageway half widths]) of the roads near feats' tunnels (ts:
        tunnel_specs), as roads.TunnelProfiles._cover_roads has them."""
        import mapgeom
        tun = [i for i in range(len(feats)) if ts[i] is not None]
        boxes = roadsmod._feature_boxes(feats)
        if not tun:
            return [], []
        pad = (roadsmod.TUNNEL_ROAD_REACH_M + 40.0) / 111320.0      # (lon/lat degrees, generous at high latitudes)
        tb = boxes[tun]
        rs = roadsmod.road_specs(feats)
        lines, halves = [], []
        for i in range(len(feats)):
            if rs[i] is None:
                continue
            b = boxes[i]
            k = 1.0 / max(math.cos(math.radians(0.5 * (b[1] + b[3]))), 0.05)
            if not ((tb[:, 0] - pad * k <= b[2]) & (tb[:, 2] + pad * k >= b[0]) & (tb[:, 1] - pad <= b[3]) &
                    (tb[:, 3] + pad >= b[1])).any():
                continue
            for ring in roadsmod.ground_parts(feats, i):
                e, n = mapgeom.enu_xy(feats, self.frame, ring)
                lines.append(np.stack([e, n], axis=1))
                halves.append(rs[i].width / 2)
        return lines, halves

    @staticmethod
    def _mouth_plans(secs, portals=None):
        """(fid, Plan) of the covered sections' mouths: the floor MOUTH_PLAN_M into the tube from a portal (where an
        open section or nothing comes before / after it): the ground at the portal graded to the road on both sides
        of it (the grading faded from the road's plan into the ungraded hill there, and the road tile and the
        terrain block, sampled at different zooms, came out a few cm apart: the block poked through the road
        outside 97th Street's arch, 2026-10-17)."""
        out = []
        for n, (fid, spec, xy, z, kind) in enumerate(secs):
            if not kind or len(xy) < 2:
                continue
            same = lambda m: 0 <= m < len(secs) and secs[m][0] == fid     # noqa: E731
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            hw = spec.width / 2 + shoulder(spec)
            for at_start in (True, False):
                endpoint = xy[0 if at_start else -1]
                if portals is not None and portals.get((fid, tuple(np.round(endpoint, 3)))) is False:
                    continue
                nb = n - 1 if at_start else n + 1
                if same(nb) and secs[nb][4] != 0:
                    continue                    # inside the hill: a full tunnel goes on
                if at_start:
                    m = s <= MOUTH_PLAN_M
                    t = np.interp(MOUTH_PLAN_M, s, np.arange(len(s)))
                else:
                    m = s >= s[-1] - MOUTH_PLAN_M
                    t = np.interp(s[-1] - MOUTH_PLAN_M, s, np.arange(len(s)))
                i0 = int(np.floor(t))
                p = xy[i0] + (t - i0) * (xy[min(i0 + 1, len(xy) - 1)] - xy[i0])
                pz = z[i0] + (t - i0) * (z[min(i0 + 1, len(z) - 1)] - z[i0])
                if at_start:
                    pxy, pzz = np.r_[xy[m], p[None]], np.r_[z[m], pz]
                else:
                    pxy, pzz = np.r_[p[None], xy[m]], np.r_[pz, z[m]]
                if len(pxy) >= 2:
                    out.append((fid + TUNNEL_TAG + ("m0" if at_start else "m1"), Plan(pxy, pzz, hw)))
        return out

    def _covered_tunnels(self, feats):
        """[(xy, half width)] of the tunnels' covered sections (not open roads): the grading leaves the ground over
        them as it is (apply): the approaches' grading reached up to T_MAX past a portal and cut a park bridge's soil
        down under its arch's vault, the terrain then cut away over it (2026-10-17)."""
        return [(xy, spec.width / 2 + TUNNEL_KEEP_M) for _f, spec, xy, _z, kind in _twin_trimmed(self._tunnel_sections(feats))
                if kind]

    def _open_tunnels(self, feats):
        """(fid, Plan) of the tunnels' open sections (roads.TUNNEL_OPEN_COVER): the ground graded to the road's floor
        (the lower hull of the ground), as the road is drawn (roads._tunnel_chunks): over a footbridge's hump in the
        DEM the road climbed it, the ground over the road (Central Park's transverses, 2026-10-17)."""
        return [(fid, Plan(xy, z, spec.width / 2 + shoulder(spec)))
                for fid, spec, xy, z, kind in self._tunnel_sections(feats) if kind == 0]

    def _raceway_ground(self, e, n, base):
        """base at the points on a raceway (_on_kept) smoothed along it (raceway_ground; RACEWAY_SMOOTH_ON)."""
        if not RACEWAY_SMOOTH_ON or self.elev is None:
            return base
        with self.lock:
            lines = sorted({(round(float(xy[0, 0]), 2), round(float(xy[0, 1]), 2), round(float(xy[-1, 0]), 2),
                             round(float(xy[-1, 1]), 2), len(xy), round(float(hw), 3)): (xy, hw)
                            for v in self.keeps.values() for xy, hw in v}.items())
        memo = self.__dict__.setdefault("_raceway_memo", OrderedDict())
        zones = []
        for key, (xy, hw) in lines:
            with self.lock:
                got = memo.get(key)
            if got is None:
                got = raceway_surface(self.frame, self.elev, xy, hw) or ()
                with self.lock:
                    memo[key] = got
                    while len(memo) > 256:
                        memo.popitem(last=False)
            if got:
                zones.append(got)
        return raceway_ground(e, n, base, zones)

    def _on_kept(self, e, n):
        """Which points lie on an UNGRADED road's carriageway (+ UNGRADED_MARGIN_M), or None."""
        with self.lock:
            zones = [z for v in self.keeps.values() for z in v]
        if not zones or not len(e):
            return None
        import fast
        fast.require(0, "over_tunnel")
        mask = np.zeros(len(e), bool)
        ee, nn = np.ascontiguousarray(e, np.float64), np.ascontiguousarray(n, np.float64)
        for xy, hw in zones:
            fast._ef.over_tunnel(mask, ee, nn, np.ascontiguousarray(xy[:, :2], np.float64), float(hw), 0.0)
        return mask if mask.any() else None

    def _over_tunnels(self, e, n):
        """Which points lie over a covered tunnel (its half width + TUNNEL_KEEP_M), or None."""
        with self.lock:
            zones = [z for v in self.covers.values() for z in v]
        if not zones or not len(e):
            return None
        import fast
        fast.require(0, "over_tunnel")      # (Rust: per segment near the points, without the GIL)
        mask = np.zeros(len(e), bool)
        ee, nn = np.ascontiguousarray(e, np.float64), np.ascontiguousarray(n, np.float64)
        for xy, hw in zones:
            fast._ef.over_tunnel(mask, ee, nn, np.ascontiguousarray(xy[:, :2], np.float64), float(hw),
                                 float(TUNNEL_KEEP_IN_M))
        return mask if mask.any() else None
