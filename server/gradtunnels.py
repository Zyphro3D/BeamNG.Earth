"""Road grading: tunnels (split out of grading.py's _Planner): subsea parts, tunnel sections and mouths, the roads
over tunnels and raceway ground.
"""
import math
from collections import OrderedDict
import numpy as np
import roads as roadsmod
from gradbase import (  # (re-exported: callers use grading.<name>)
    ABUT_FAR_M,  # noqa: F401
    ABUT_FILL_ON,  # noqa: F401
    ABUT_MAX_M,  # noqa: F401
    ABUT_MIN_M,  # noqa: F401
    ABUT_NEAR_M,  # noqa: F401
    ABUT_STEEP,  # noqa: F401
    ABUT_UNDER_GRADE,  # noqa: F401
    APPROACH_CAP_GRADE,  # noqa: F401
    APPROACH_CELL_M,  # noqa: F401
    APPROACH_FLAT_M,  # noqa: F401
    APPROACH_GAP_M,  # noqa: F401
    APPROACH_GRADE,  # noqa: F401
    APPROACH_LIDAR_ON,  # noqa: F401
    APPROACH_MARGIN_M,  # noqa: F401
    APPROACH_MAX_LIFT,  # noqa: F401
    APPROACH_MAX_RUN,  # noqa: F401
    APPROACH_MIN_CROSS_M,  # noqa: F401
    APPROACH_MIN_LIFT,  # noqa: F401
    APPROACH_MIN_RUN,  # noqa: F401
    APPROACH_ON,  # noqa: F401
    APPROACH_SKIP,  # noqa: F401
    APPROACH_TOUCH_M,  # noqa: F401
    BANK_DESIGN,  # noqa: F401
    BANK_E_MAX,  # noqa: F401
    BANK_FADE_M,  # noqa: F401
    BANK_LINK,  # noqa: F401
    BANK_ON,  # noqa: F401
    BANK_RUNOFF_M,  # noqa: F401
    BANK_SIGMA_M,  # noqa: F401
    BANK_STENCIL_M,  # noqa: F401
    BLEND_R,  # noqa: F401
    BUCKET,  # noqa: F401
    CHAIN_JOIN_M,  # noqa: F401
    CHAIN_LOOK_M,  # noqa: F401
    CHAIN_ON,  # noqa: F401
    CHAIN_TURN_DEG,  # noqa: F401
    CLASS_SIGMA_M,  # noqa: F401
    CLIP_MARGIN_M,  # noqa: F401
    COARSE_EASE_ON,  # noqa: F401
    COARSE_MAX_SLOPE,  # noqa: F401
    COARSE_MAX_STEP_M,  # noqa: F401
    COARSE_REACH_M,  # noqa: F401
    COARSE_SIDE_M,  # noqa: F401
    COARSE_SIGMA_M,  # noqa: F401
    CORE_FADE_M,  # noqa: F401
    CORNERS_ON,  # noqa: F401
    CW_CLUSTER_M,  # noqa: F401
    CW_HEADROOM_M,  # noqa: F401
    CW_MARGIN_M,  # noqa: F401
    CW_MAX_DIG_M,  # noqa: F401
    CW_MAX_M,  # noqa: F401
    CW_MIN_SIN,  # noqa: F401
    CW_NONFAM_HI,  # noqa: F401
    CW_ON,  # noqa: F401
    CW_OVERLAP_MAX_M,  # noqa: F401
    CW_OVERLAP_STEP_M,  # noqa: F401
    CW_STACK_SIN,  # noqa: F401
    CW_UNDER_M,  # noqa: F401
    DIRT_BANK_EDGE_M,  # noqa: F401
    DIRT_BANK_M,  # noqa: F401
    DIRT_BANK_WAVES,  # noqa: F401
    DIRT_BASE_M,  # noqa: F401
    DIRT_DEPTH_MAX_M,  # noqa: F401
    DIRT_MAJOR_SHARE,  # noqa: F401
    DIRT_MEDIAN_M,  # noqa: F401
    DIRT_PATCH_D_M,  # noqa: F401
    DIRT_PATCH_EVERY_M,  # noqa: F401
    DIRT_PATCH_W_M,  # noqa: F401
    DIRT_PIT_D_M,  # noqa: F401
    DIRT_PIT_EVERY_M,  # noqa: F401
    DIRT_PIT_W_M,  # noqa: F401
    DIRT_POTHOLE_D_M,  # noqa: F401
    DIRT_POTHOLE_EVERY_M,  # noqa: F401
    DIRT_POTHOLE_W_M,  # noqa: F401
    DIRT_RISE_MAX_M,  # noqa: F401
    DIRT_ROUGH,  # noqa: F401
    DIRT_SIGMA_COARSE_M,  # noqa: F401
    DIRT_SIGMA_M,  # noqa: F401
    DIRT_SIGMA_MINOR_M,  # noqa: F401
    DIRT_SOFT_CLASSES,  # noqa: F401
    DIRT_STEP_MAX_M,  # noqa: F401
    DIRT_TAPER_M,  # noqa: F401
    DIRT_TRACK_SHARE,  # noqa: F401
    DIRT_TRACK_WAVE_SHARE,  # noqa: F401
    DIRT_UNDULATION_M,  # noqa: F401
    DIRT_UNDULATION_WAVES,  # noqa: F401
    DIRT_VERTEX_STEP,  # noqa: F401
    DUPE_XY_M,  # noqa: F401
    ENU_CACHE_MAX,  # noqa: F401
    EXTRAP_M,  # noqa: F401
    HAIRPIN_FOLD_RATIO,  # noqa: F401
    HAIRPIN_PAD_M,  # noqa: F401
    HAIRPIN_SPAN_M,  # noqa: F401
    HAIRPIN_TURN_DEG,  # noqa: F401
    JUNCTION_EASE_GRADE,  # noqa: F401
    JUNCTION_EASE_MINOR_GRADE,  # noqa: F401
    JUNCTION_EASE_MINOR_MIN_M,  # noqa: F401
    JUNCTION_EASE_MIN_M,  # noqa: F401
    JUNCTION_END_M,  # noqa: F401
    JUNCTION_HOLD_GRADE,  # noqa: F401
    JUNCTION_HOLD_PAD_M,  # noqa: F401
    JUNCTION_MAX_M,  # noqa: F401
    JUNCTION_RANK,  # noqa: F401
    JUNCTION_REACH_M,  # noqa: F401
    JUNCTION_SNAP_ON,  # noqa: F401
    JUNCTION_TAPER_M,  # noqa: F401
    LEG_RID_SHIFT,  # noqa: F401
    LINK_TURN_DEG,  # noqa: F401
    LL_GRID_M,  # noqa: F401
    LONG_PART_M,  # noqa: F401
    MAJOR_CLASSES,  # noqa: F401
    MAPPED_MAX,  # noqa: F401
    MEASURED_SIDE_RUN,  # noqa: F401
    MEDIAN_M,  # noqa: F401
    MOUNTAIN_SHOULDER_M,  # noqa: F401
    MOUTH_PLAN_M,  # noqa: F401
    NEIGHBOUR_FULL_M,  # noqa: F401
    NEIGHBOUR_GAP_M,  # noqa: F401
    NEIGHBOUR_MAX_M,  # noqa: F401
    NEIGHBOUR_ON,  # noqa: F401
    NEIGHBOUR_PARALLEL_DEG,  # noqa: F401
    NEIGHBOUR_SMOOTH_M,  # noqa: F401
    NEIGHBOUR_TREND_SIGMA_M,  # noqa: F401
    OBS_END_M,  # noqa: F401
    OBS_HALF_M,  # noqa: F401
    OVER_TUNNEL_SMALL_M,  # noqa: F401
    PAD_BANK_TAPER_M,  # noqa: F401
    PAD_CLASS_W,  # noqa: F401
    PAD_CROSS_MAX_M,  # noqa: F401
    PAD_FACE_DEG,  # noqa: F401
    PAD_FIT_POW,  # noqa: F401
    PAD_FLAT_W,  # noqa: F401
    PAD_GAMMA,  # noqa: F401
    PAD_GRADE_M,  # noqa: F401
    PAD_GRID_M,  # noqa: F401
    PAD_LANDING_M,  # noqa: F401
    PAD_LANDING_MINOR_M,  # noqa: F401
    PAD_LANDING_MIN_W,  # noqa: F401
    PAD_LINK_F,  # noqa: F401
    PAD_MAX_TILT,  # noqa: F401
    PAD_MIN,  # noqa: F401
    PAD_MIN_DELTA_M,  # noqa: F401
    PAD_MIN_PART_M,  # noqa: F401
    PAD_MIN_SIN,  # noqa: F401
    PAD_ON,  # noqa: F401
    PAD_OTHER_W,  # noqa: F401
    PAD_REACH_F,  # noqa: F401
    PAD_RIDGE,  # noqa: F401
    PAD_SINK_M,  # noqa: F401
    PAD_STEM_F,  # noqa: F401
    PAD_TAPERS_M,  # noqa: F401
    PAD_UNBUILT_F,  # noqa: F401
    PLANNERS_KEPT,  # noqa: F401
    PLAN_SETS_MAX,  # noqa: F401
    PLAN_SLOW_S,  # noqa: F401
    PLAN_THREADS,  # noqa: F401
    PROFILE_STEP,  # noqa: F401
    PROFILE_ZOOM,  # noqa: F401
    Plan,  # noqa: F401
    PlanPack,  # noqa: F401
    RACEWAY_ACROSS_M,  # noqa: F401
    RACEWAY_EXTEND,  # noqa: F401
    RACEWAY_SIGMA_M,  # noqa: F401
    RACEWAY_SMOOTH_ON,  # noqa: F401
    RACEWAY_STEP_M,  # noqa: F401
    RAIL_GRADE_ON,  # noqa: F401
    RAIL_TAG,  # noqa: F401
    RETRY_S,  # noqa: F401
    RING_LEVEL_ON,  # noqa: F401
    ROAD_LIFT,  # noqa: F401
    SEP_FROM_M,  # noqa: F401
    SEP_FULL_M,  # noqa: F401
    SEP_ON,  # noqa: F401
    SHARP_GRADE,  # noqa: F401
    SHARP_LONE_MIN_M,  # noqa: F401
    SHARP_MIN_M,  # noqa: F401
    SHARP_ON,  # noqa: F401
    SHARP_REACH_M,  # noqa: F401
    SHARP_RUN_N,  # noqa: F401
    SHARP_STEP_M,  # noqa: F401
    SHOULDER_M,  # noqa: F401
    SIDEWALK_MARGIN_M,  # noqa: F401
    SIDE_FADE_M,  # noqa: F401
    SIGMA_M,  # noqa: F401
    SIGMA_MAJOR_M,  # noqa: F401
    SLOPE_RUN,  # noqa: F401
    SPREAD_CELLS,  # noqa: F401
    STEEP_RUN,  # noqa: F401
    STEEP_T_MIN,  # noqa: F401
    TERRAIN_BUMPS_ON,  # noqa: F401
    TERRAIN_ROUGH_M,  # noqa: F401
    TERRAIN_ROUGH_WAVES,  # noqa: F401
    TUNNEL_KEEP_IN_M,  # noqa: F401
    TUNNEL_KEEP_M,  # noqa: F401
    TUNNEL_TAG,  # noqa: F401
    TWIN_M,  # noqa: F401
    TWIN_ON,  # noqa: F401
    TWIN_SHARE,  # noqa: F401
    T_MAX,  # noqa: F401
    T_MIN,  # noqa: F401
    UNDER_ANY_ON,  # noqa: F401
    UNDER_CHAIN_JOIN_M,  # noqa: F401
    UNDER_CHAIN_MAX_M,  # noqa: F401
    UNDER_CHAIN_ON,  # noqa: F401
    UNDER_CLEAR_M,  # noqa: F401
    UNDER_DECK_MAX_M,  # noqa: F401
    UNDER_DECK_ON,  # noqa: F401
    UNDER_DECK_PAD_M,  # noqa: F401
    UNDER_DECK_SIDE_M,  # noqa: F401
    UNDER_FLAT_M,  # noqa: F401
    UNDER_GRADE,  # noqa: F401
    UNDER_JOINT_FADE_M,  # noqa: F401
    UNDER_JOINT_FLAT_M,  # noqa: F401
    UNDER_LIDAR_DECK_M,  # noqa: F401
    UNDER_LIDAR_ON,  # noqa: F401
    UNDER_MAX_BRIDGE_M,  # noqa: F401
    UNDER_MAX_DROP_M,  # noqa: F401
    UNDER_MAX_GRADE,  # noqa: F401
    UNDER_NEAR_M,  # noqa: F401
    UNDER_NODE_M,  # noqa: F401
    UNDER_ON,  # noqa: F401
    UNDER_POLDER_M,  # noqa: F401
    UNDER_REACH_M,  # noqa: F401
    UNDER_RUN_MIN_M,  # noqa: F401
    UNDER_RUN_ON,  # noqa: F401
    UNDER_SEA_M,  # noqa: F401
    UNDER_SEA_ON,  # noqa: F401
    UNDER_STUB_M,  # noqa: F401
    UNGRADED,  # noqa: F401
    UNGRADED_MARGIN_M,  # noqa: F401
    VERSION,  # noqa: F401
    VERTEX_STEP,  # noqa: F401
    WALK_SPAN_CLASSES,  # noqa: F401
    WALK_SPAN_HALF_M,  # noqa: F401
    WALL_AHEAD_M,  # noqa: F401
    WALL_CUT_M,  # noqa: F401
    WALL_FLAT_M,  # noqa: F401
    WALL_MAX_H_CITY_M,  # noqa: F401
    WALL_MAX_H_M,  # noqa: F401
    WALL_T_M,  # noqa: F401
    WATER_CURVE_M,  # noqa: F401
    WATER_LIFT_ON,  # noqa: F401
    WATER_MAX_LIFT,  # noqa: F401
    XS_FIXED_GRADE,  # noqa: F401
    XS_FLOOR_M,  # noqa: F401
    XS_ON,  # noqa: F401
    YIELD_RAMP_M,  # noqa: F401
    _Decks,  # noqa: F401
    _Junctions,  # noqa: F401
    _STENCILS,  # noqa: F401
    _UNSET,  # noqa: F401
    _add_cuts,  # noqa: F401
    _arcs,  # noqa: F401
    _cw_half,  # noqa: F401
    _cw_level,  # noqa: F401
    _dedupe_plan,  # noqa: F401
    _hash01,  # noqa: F401
    _hold_junction_corridors,  # noqa: F401
    _is_link,  # noqa: F401
    _merge_span_chains,  # noqa: F401
    _nearest_on,  # noqa: F401
    _never_raised,  # noqa: F401
    _normals,  # noqa: F401
    _pad_nodes,  # noqa: F401
    _plan,  # noqa: F401
    _point_along,  # noqa: F401
    _road_fids,  # noqa: F401
    _seg_dist,  # noqa: F401
    _stencil,  # noqa: F401
    _twin_trimmed,  # noqa: F401
    approach_add,  # noqa: F401
    bank,  # noqa: F401
    bend_zone,  # noqa: F401
    bridge_chain,  # noqa: F401
    bridge_chain_ends_free,  # noqa: F401
    bridge_lifts,  # noqa: F401
    clip_polyline,  # noqa: F401
    couple_neighbours,  # noqa: F401
    deck_capped,  # noqa: F401
    dirt_bank,  # noqa: F401
    dirt_bumps,  # noqa: F401
    dirt_taper,  # noqa: F401
    ease_length,  # noqa: F401
    enu_to_ll_dense,  # noqa: F401
    field_heights,  # noqa: F401
    fill_abutment,  # noqa: F401
    grade_points,  # noqa: F401
    hairpin_cores,  # noqa: F401
    junction_pads,  # noqa: F401
    log,  # noqa: F401
    median_profile,  # noqa: F401
    pad_authority,  # noqa: F401
    pad_class_w,  # noqa: F401
    pad_plane,  # noqa: F401
    plan_road,  # noqa: F401
    plan_roads,  # noqa: F401
    profile_sigma,  # noqa: F401
    raceway_ground,  # noqa: F401
    raceway_surface,  # noqa: F401
    resample,  # noqa: F401
    road_id,  # noqa: F401
    road_rank,  # noqa: F401
    set_widen,  # noqa: F401
    shoulder,  # noqa: F401
    smooth_profile,  # noqa: F401
    snap_junctions,  # noqa: F401
    street_runs,  # noqa: F401
    terrain_bumps,  # noqa: F401
    under_dip,  # noqa: F401
    underpass_offsets,  # noqa: F401
    wall_max_h,  # noqa: F401
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
