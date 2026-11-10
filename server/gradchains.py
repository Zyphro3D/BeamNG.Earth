"""Road grading: a bridge chain's candidate heights (split out of grading.py's _Planner): chain candidates, pins,
the stronger of two answers, observed lifts and the coarse elevation's easing.
"""
import numpy as np
import mapcells
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


class _ChainsMixin:
    """_Planner's bridge chains' candidate heights: their ends, pins, the stronger of two answers and the observed lifts (split out of grading.py; _Planner inherits it)."""

    def _chain_cands(self, key, feats=None, specs=None):
        """The graded ground parts of cell key's roads, as the continuations at a node in it are looked for (every cell
        that plans a part looks in its nodes' own cells, so its plan is the same in each): [(xy (k, 2) in the frame,
        smoothing sigma, road id, lon/lat ring, spec, feature id)], in the map's feature order. Kept per cell. feats, specs: the cell's roads and their
        road_specs when the caller has them."""
        memo = self.__dict__.setdefault("_chain_memo", {})
        with self.lock:
            got = memo.get(key)
        if got is not None:
            return got
        out = []
        if feats is None:
            feats = self.md.cell(key, layers=("roads",), kind="demand").layers.get("roads")
        if feats is not None and len(feats):
            if specs is None:
                specs = roadsmod.road_specs(feats, None, self.widths)
            rows = []
            for i in range(len(feats)):
                if specs[i] is None or specs[i].cls in UNGRADED:
                    continue
                for ring in roadsmod.ground_parts(feats, i):
                    rows.append((i, np.asarray(ring, np.float64)))
            if rows:            # (the frame's conversion once for all: it is per call)
                allr = np.concatenate([r for _, r in rows])
                e, n, _ = self.frame.geodetic_to_enu(allr[:, 1], allr[:, 0], 0.0)
                en = np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64)], axis=1)
                k0 = 0
                for i, ring in rows:
                    out.append((en[k0:k0 + len(ring)], profile_sigma(specs[i]), road_id(str(feats.ids[i])), ring,
                                specs[i], str(feats.ids[i])))
                    k0 += len(ring)
        with self.lock:
            if len(memo) > 64:
                memo.clear()
            memo[key] = out
        return out

    def _extensions(self, parts, ids):
        """plan_roads' ext of a cell's parts [(lonlat, spec)] (ids: their features'): per part (before, after), the line
        from each end outward along the part the road goes on in there (earthfast.chain_extensions), 3 sigma of its
        smoothing long, or None."""
        import fast
        fast.require(69, "road chains")
        f64 = np.float64
        q_node, q_dir, q_class, q_road, q_reach, q_cell = [], [], [], [], [], []
        if not parts:
            return []
        # (the frame's conversion once for every part: it is per call)
        alll = np.concatenate([np.asarray(ll, f64) for (ll, _sp) in parts])
        ae, an, _ = self.frame.geodetic_to_enu(alll[:, 1], alll[:, 0], 0.0)
        axy = np.stack([np.asarray(ae, f64), np.asarray(an, f64)], axis=1)
        k0 = 0
        for (lonlat, spec), fid in zip(parts, ids):
            xy = axy[k0:k0 + len(lonlat)]
            k0 += len(lonlat)
            sg = profile_sigma(spec)
            for end in (0, 1):
                line = xy if end == 0 else xy[::-1]          # from this end inward
                p = _point_along(line, CHAIN_LOOK_M)
                q_node.append(line[0])
                q_dir.append(line[0] - p)                    # the way the road goes on out of this end
                q_class.append(int(round(sg * 1000)))
                q_road.append(road_id(fid))
                reach = 3.0 * sg + PROFILE_STEP
                if BANK_ON and spec.kind in roadsmod.BUILT and spec.cls in BANK_DESIGN:
                    # Runoff averages curvature beyond the join, and that
                    # curvature needs its own stencil on either side too.
                    reach = max(reach, BANK_STENCIL_M + BANK_RUNOFF_M / 2 + 3 * BANK_SIGMA_M + PROFILE_STEP)
                q_reach.append(reach)
                q_cell.append(lonlat[0] if end == 0 else lonlat[-1])
        if q_cell:                  # (the ends' cells at once)
            qll = np.asarray(q_cell, np.float64).reshape(-1, 2)
            q_cell = mapcells.cells_of(qll[:, 1], qll[:, 0])
        out = [[None, None] for _ in parts]
        for key in sorted(set(q_cell), key=str):
            cands = self._chain_cands(key)
            qs = [i for i, c in enumerate(q_cell) if c == key]
            if not cands or not qs:
                continue
            cxy, coff, ccls, croad = self._cand_arrays(key, cands)
            pts, off = fast._ef.chain_extensions(
                cxy, coff, ccls, croad,
                np.ascontiguousarray([q_node[i] for i in qs], f64), np.ascontiguousarray([q_dir[i] for i in qs], f64),
                np.array([q_class[i] for i in qs], np.int64), np.array([q_road[i] for i in qs], np.int64),
                np.array([q_reach[i] for i in qs], f64), float(CHAIN_JOIN_M), float(CHAIN_TURN_DEG), float(CHAIN_LOOK_M))
            for j, i in enumerate(qs):
                if off[j + 1] - off[j] >= 2:
                    out[i // 2][i % 2] = np.asarray(pts[off[j]:off[j + 1]])
        return [tuple(o) for o in out]

    def _cand_arrays(self, key, cands):
        """_chain_cands(key) as the Rust kernels take them (points, offsets, sigma class, road id), made once per cell."""
        memo = self.__dict__.setdefault("_cand_arrays_memo", {})
        got = memo.get(str(key))
        if got is not None and got[0] is cands:
            return got[1]
        arr = (np.ascontiguousarray(np.concatenate([c[0] for c in cands]), np.float64),
               np.r_[0, np.cumsum([len(c[0]) for c in cands])].astype(np.int64),
               np.array([int(round(c[1] * 1000)) for c in cands], np.int64),
               np.array([c[2] for c in cands], np.int64))
        with self.lock:
            if len(memo) > 64:
                memo.clear()
            memo[str(key)] = (cands, arr)
        return arr

    def _pins(self, parts, ids, ext):
        """plan_roads' pins of a cell's parts: an end that no part of its own smoothing goes on from (ext None there) and
        that ends on another road's part smoothed as much or more (a slip road on a motorway, a side road on a trunk or
        on a street of its own class) is held at that road's planned height there (_stronger_z), so the two meet at one
        height (a same-class side road ended at the raw ground, the street through the junction at its smoothed height:
        0.6 m apart on a hill). The most smoothed such road, the first of equals. None when no end is held."""
        import fast
        fast.require(70, "road pins")
        f64 = np.float64
        q_node, q_cell, q_who = [], [], []
        for j, ((lonlat, spec), fid) in enumerate(zip(parts, ids)):
            for end in (0, 1):
                if ext is not None and ext[j][end] is not None:
                    continue
                ll = lonlat[-1] if end else lonlat[0]
                q_node.append((float(ll[1]), float(ll[0])))         # (lat, lon: converted below, all at once)
                q_who.append((j, end, profile_sigma(spec), road_id(fid)))
        if q_node:
            qa = np.asarray(q_node, f64)
            q_cell = mapcells.cells_of(qa[:, 0], qa[:, 1])
            qe, qn, _ = self.frame.geodetic_to_enu(qa[:, 0], qa[:, 1], 0.0)
            q_node = list(zip(np.asarray(qe, f64).tolist(), np.asarray(qn, f64).tolist()))
        out = [[None, None] for _ in parts]
        held = False
        todo = []                           # (part, end, candidate, at its last point): heights asked all at once
        for key in sorted(set(q_cell), key=str):
            cands = self._chain_cands(key)
            qs = [i for i, c in enumerate(q_cell) if c == key]
            weakest = min(q_who[i][2] for i in qs)
            strong = [c for c in cands if c[1] >= weakest]
            if not strong:
                continue
            cxy = np.concatenate([c[0] for c in strong])
            coff = np.r_[0, np.cumsum([len(c[0]) for c in strong])].astype(np.int64)
            hits, off = fast._ef.node_parts(np.ascontiguousarray(cxy, f64), coff,
                                            np.ascontiguousarray([q_node[i] for i in qs], f64), float(CHAIN_JOIN_M))
            for k, i in enumerate(qs):
                j, end, sg, rid = q_who[i]
                best = None
                for h in hits[off[k]:off[k + 1]]:
                    c = strong[int(h) // 2]
                    if c[1] >= sg and c[2] != rid and (best is None or c[1] > best[0][1]):
                        best = (c, bool(int(h) % 2))
                if best is not None:
                    todo.append((j, end, best[0], best[1]))
        for (j, end, _c, _l), z in zip(todo, self._stronger_zs([(c, last) for _j, _e, c, last in todo])):
            if z is not None:
                out[j][end] = z
                held = True
        return [tuple(o) for o in out] if held else None

    def _stronger_zs(self, asks):
        """_stronger_z of each (candidate, at_last) of asks, the ones not kept yet planned in one plan_roads call (the
        same plans: each part is planned on its own; its cost is per call, ~1 ms, 935 calls in 4 Penn Station cells,
        2026-10-25)."""
        memo = self.__dict__.setdefault("_strong_memo", {})
        keys = [self._strong_key(c, last) for c, last in asks]
        with self.lock:
            miss = [n for n, k in enumerate(keys) if k not in memo]
        jobs, seen = [], set()
        for n in miss:
            if keys[n] not in seen:
                seen.add(keys[n])
                jobs.append(n)
        if len(jobs) > 1:
            try:
                preps = [self._strong_part(*asks[n]) for n in jobs]
                parts = [(part, asks[n][0][4]) for n, (part, _last, _whole) in zip(jobs, preps)]
                ext = self._extensions(parts, [asks[n][0][5] for n in jobs])
                plans = plan_roads(self.frame, parts, self.elev, None, ext, cuts=False,
                                   coarse=COARSE_EASE_ON and self._coarse_ll(parts[0][0]),
                                   obs=self._obs(None, parts[0][0]), decks=self._decks_ll(parts[0][0]))
                for n, (part, last, whole), p in zip(jobs, preps, plans):
                    self._strong_keep(asks[n][0], keys[n], p, last, whole)
            except Exception as e:      # one at a time, each its own failure (_stronger_z)
                log.debug("grading: pin heights at once: %s: %s", type(e).__name__, e)
        return [self._stronger_z(c, last) for c, last in asks]

    @staticmethod
    def _strong_key(cand, at_last):
        xy, rid = cand[0], cand[2]
        e = xy[-1 if at_last else 0]
        return (rid, round(float(e[0]), 1), round(float(e[1]), 1))

    def _strong_part(self, cand, at_last):
        """(the lon/lat part planned for _stronger_z, whose end is asked for: its last point?, the whole part?)"""
        xy, _sg, rid, ring, spec, fid = cand
        part = ring
        if float(np.hypot(*np.diff(xy, axis=0).T).sum()) > LONG_PART_M:
            end = xy[-1] if at_last else xy[0]
            m = CLIP_MARGIN_M
            runs = clip_polyline(xy, (end[0] - m, end[1] - m, end[0] + m, end[1] + m))
            run = min(runs, key=lambda r: min(np.hypot(*(r[0] - end)), np.hypot(*(r[-1] - end)))) if runs else None
            if run is not None:
                la, lo, _ = self.frame.enu_to_geodetic(run[:, 0], run[:, 1], 0.0)
                part = np.stack([np.asarray(lo), np.asarray(la)], axis=1)
                at_last = bool(np.hypot(*(run[-1] - end)) <= np.hypot(*(run[0] - end)))
        return part, at_last, part is ring

    def _strong_keep(self, cand, k, p, at_last, whole):
        """Keep a _stronger_z height (and the other end's, for a part planned whole)."""
        z, other = None, None
        if p is not None:
            z = float(p.z[-1] if at_last else p.z[0])
            if whole:
                xy, rid = cand[0], cand[2]
                e2 = xy[0 if at_last else -1]
                other = ((rid, round(float(e2[0]), 1), round(float(e2[1]), 1)), float(p.z[0] if at_last else p.z[-1]))
        memo = self.__dict__.setdefault("_strong_memo", {})
        with self.lock:
            if len(memo) > 4096:
                memo.clear()
            memo.setdefault(k, z)
            if other is not None:
                memo.setdefault(*other)

    def _stronger_z(self, cand, at_last):
        """The planned height of a candidate part (_chain_cands') at its first or last point: its own plan (its chains,
        no pins), within CLIP_MARGIN_M of that point when it is long. Kept per part end (a short part's plan gives both
        ends' heights: the other kept too)."""
        k = self._strong_key(cand, at_last)
        memo = self.__dict__.setdefault("_strong_memo", {})
        with self.lock:
            if k in memo:
                return memo[k]
        try:
            part, last, whole = self._strong_part(cand, at_last)
            # (the profile alone: its cutting's walls cost most of a plan and no height, 2026-10-25)
            p = plan_roads(self.frame, [(part, cand[4])], self.elev, None,
                           self._extensions([(part, cand[4])], [cand[5]]), cuts=False,
                           coarse=COARSE_EASE_ON and self._coarse_ll(part), obs=self._obs(None, part),
                           decks=self._decks_ll(part))[0]
            self._strong_keep(cand, k, p, last, whole)
        except Exception as e:      # not held rather than no grading
            log.debug("grading: pin height of %s: %s: %s", cand[5], type(e).__name__, e)
            with self.lock:
                memo[k] = None
        with self.lock:
            return memo.get(k)

    def _clip_quad(self, key):
        """The cell's quad (4, 2: its corners SW, SE, NE, NW at h = 0) in the frame, or None for a key without bounds:
        the cells' own pieces of the plans (trim_plan), each place in one cell's whatever the frame."""
        if not hasattr(key, "bounds"):
            return None
        lon0, lat0, lon1, lat1 = key.bounds()
        e, n, _ = self.frame.geodetic_to_enu(np.array([lat0, lat0, lat1, lat1]), np.array([lon0, lon1, lon1, lon0]),
                                             0.0)
        return np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64)], axis=1)

    def _clip_box(self, key, margin=CLIP_MARGIN_M):
        """The cell's box in the frame (east / north), margin wider each side; None for a key without bounds."""
        if not hasattr(key, "bounds"):
            return None
        lon0, lat0, lon1, lat1 = key.bounds()
        e, n, _ = self.frame.geodetic_to_enu(np.array([lat0, lat0, lat1, lat1]), np.array([lon0, lon1, lon0, lon1]),
                                             0.0)
        e, n = np.asarray(e), np.asarray(n)
        m = margin
        return float(e.min()) - m, float(n.min()) - m, float(e.max()) + m, float(n.max()) + m

    def _near_cell(self, ring, box):
        """A road part (lon/lat ring) as planned for a cell: as it is, or, longer than LONG_PART_M, the pieces of it
        within the cell's box (each its own plan). A cell's plans sampled the ground along every road through it
        to its ends: on the Bonneville salt flats, tracks one segment 27 km long, ~2,500 DEM tiles for one road
        tile's four cells (2026-10-20). The profile's smoothing (SIGMA_M) reaches a few tens of metres: the plan
        within the cell is the same."""
        if box is None or len(ring) < 2:
            return [ring]
        e, n, _ = self.frame.geodetic_to_enu(ring[:, 1], ring[:, 0], 0.0)
        xy = np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64)], axis=1)
        if float(np.hypot(*np.diff(xy, axis=0).T).sum()) <= LONG_PART_M:
            return [ring]
        out = []
        for run in clip_polyline(xy, box):
            la, lo, _ = self.frame.enu_to_geodetic(run[:, 0], run[:, 1], 0.0)
            out.append(np.stack([np.asarray(lo, np.float64), np.asarray(la, np.float64)], axis=1))
        return out

    def _obs(self, key, lonlat=None):
        """plan_roads' obs: the measured heights of ground roads (roadz.PROVIDER, level 0) where cell key's (or the
        part's) elevation is only the coarse global model (dem.coarse_at): on lidar terrain the DTM is the better
        height (decimetres; BD TOPO's Z is 1.5-2.5 m), so None there, as when roadz is off."""
        import roadz
        prov = roadz.PROVIDER
        if prov is None or not prov.enabled:
            return None
        if not (self._coarse(key) if key is not None else self._coarse_ll(lonlat)):
            if not APPROACH_LIDAR_ON or not getattr(prov, "lidar", False):
                return None

            def approach_obs(xy, spec):
                # (on lidar terrain: the lidar's measured approaches alone, the parts from a bridge's end: roadz
                # APPROACH_*; their embankments our DTM smooths away)
                if not self._touches_bridge_end(xy):
                    return None
                try:
                    return prov.observed(self.frame, xy, spec.width / 2, 0, lidar_ground=True)
                except Exception as e:  # the DEM's rather than no grading
                    log.debug("grading: measured approach: %s: %s", type(e).__name__, e)
                    return None
            return approach_obs

        def obs(xy, spec):
            try:
                return prov.observed(self.frame, xy, spec.width / 2, 0)
            except Exception as e:      # the DEM's rather than no grading
                log.debug("grading: measured road heights: %s: %s", type(e).__name__, e)
                return None
        return obs

    def _touches_bridge_end(self, xy):
        """Does a road part (xy (k, 2) in this frame) start or end at a bridge's end (APPROACH_TOUCH_M)?"""
        xy = np.asarray(xy, np.float64)
        if len(xy) < 2:
            return False
        tips = xy[[0, -1]]
        las, los, _ = self.frame.enu_to_geodetic(tips[:, 0], tips[:, 1], 0.0)     # (both ends in one call: per point)
        for p, la, lo in zip(tips, las.tolist(), los.tolist()):
            ends = np.asarray(self._cell_enu(mapcells.cell_of(la, lo), ground=False)["ends"], np.float64).reshape(-1, 2)
            if len(ends) and (np.hypot(ends[:, 0] - p[0], ends[:, 1] - p[1]) <= APPROACH_TOUCH_M).any():
                return True
        return False

    def _observed_lift(self, bridge, xy, z_at):
        """(m the abutment at bridge end xy is raised, the measured deck end's height) when a source measured the deck
        (roadz.PROVIDER, level 1): the measured deck end over the ground there, 0 or more; None when none did (the
        inference: bridge_lifts)."""
        import roadz
        prov = roadz.PROVIDER
        if prov is None or not prov.enabled:
            return None
        try:
            z = prov.observed(self.frame, bridge, OBS_HALF_M, 1)
        except Exception as e:
            log.debug("grading: measured deck at %s: %s: %s", xy, type(e).__name__, e)
            return None
        if z is None:
            return None
        z = np.asarray(z, np.float64)
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(bridge, axis=0).T))]
        at_end = bool(np.hypot(*(bridge[-1] - xy)) < np.hypot(*(bridge[0] - xy)))
        d = (s[-1] - s) if at_end else s
        near = np.flatnonzero(np.isfinite(z) & (d <= OBS_END_M))
        if not len(near):
            return None
        k = int(near[np.argmin(d[near])])
        return max(0.0, float(z[k]) - float(z_at(np.asarray([xy], np.float64))[0])), float(z[k])

    def _coarse(self, key):
        """Is cell key's elevation only the coarse global model (dem.coarse_at its centre)? False without a DEM."""
        dem = getattr(self, "dem", None) or getattr(self.md, "dem", None)
        if dem is None or not hasattr(dem, "coarse_at"):
            return False
        lon0, lat0, lon1, lat1 = key.bounds()
        try:
            return bool(dem.coarse_at((lat0 + lat1) / 2, (lon0 + lon1) / 2))
        except Exception:
            return False

    def _coarse_ll(self, lonlat):
        """dem.coarse_at the middle of a part's (lon, lat) points (the pins' and chains' own plans sample as the cell's)."""
        dem = getattr(self, "dem", None) or getattr(self.md, "dem", None)
        if dem is None or not hasattr(dem, "coarse_at") or not len(lonlat):
            return False
        lon, lat = np.asarray(lonlat, np.float64)[len(lonlat) // 2][:2]
        try:
            return bool(dem.coarse_at(float(lat), float(lon)))
        except Exception:
            return False

    def _ease_coarse(self, key, plans, pspecs):
        """COARSE_* above: the cell's plans eased together where its elevation is coarse (in place)."""
        if not self._coarse(key):
            return
        ps = [p for _f, p in plans]
        if len(ps) < 2:
            return
        import fast
        fast.require(94, "ease_coarse")
        off = np.r_[0, np.cumsum([len(p.xy) for p in ps])].astype(np.int64)
        xy = np.ascontiguousarray(np.concatenate([np.asarray(p.xy, np.float64)[:, :2] for p in ps]))
        z = np.ascontiguousarray(np.concatenate([np.asarray(p.z, np.float64) for p in ps]))
        rank = np.array([road_rank(sp) for sp in pspecs], np.int64)
        half = np.array([p.road_half for p in ps], np.float64)
        take = np.array([0 if (p.abut[0] or p.abut[1]) else 1 for p in ps], np.uint8)
        out = fast._ef.ease_coarse(xy, z, off, rank, half, take, COARSE_REACH_M, COARSE_MAX_STEP_M, COARSE_MAX_SLOPE,
                                   COARSE_SIGMA_M)
        for j, p in enumerate(ps):
            p.z = np.asarray(out[off[j]:off[j + 1]], np.float64)
