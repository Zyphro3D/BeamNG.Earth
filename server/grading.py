"""Ground graded under roads (docs/ROADMAP.md v1.6; docs/research/2026-10-11-official-road-study.md §3, §5.3):
the physics block and the near mesh tiles sample one graded elevation, so tyres, DecalRoads and the
visible ground agree.

Per road (roads.road_specs; bridges, tunnels and off-ground segments are left out by it):
- along: the centreline height is the DEM sampled every PROFILE_STEP m and smoothed (Gaussian,
  SIGMA_M m; dirt roads DIRT_SIGMA_M), so grades stay real and small bumps go (the official maps:
  |2nd difference| per 10 m 0.02-0.1 m);
- across: flat over the core (the carriageway as a city street would have it, parking included, plus
  SHOULDER_M, or the sidewalk plus SIDEWALK_MARGIN_M on the classes that get sidewalks in towns, so a
  raised sidewalk slab sits on flat ground), then back to the natural ground over a transition of 2x the height difference (a
  1:2 cut or fill slope), clamped to [T_MIN, T_MAX] m;
- several roads at one point (junctions, divided carriageways) blend their targets by inverse distance
  from each centreline (BLEND_R), so they meet in a gradual slope.
The core width never depends on the land cover (it is taken as urban for every built road), and
each map cell's plan is made once and cached, so the block and the tiles built later always agree.

Plans do not depend on the anchor (geotiles, 2026-10-29): a cell is planned in the frame of its planning region
(geotiles.PLAN_Z: the ENU plane at the region's centre; heights geodetic, as ever) by that region's _Planner, one per
region per process (PlanStore, shared by every anchor's Grader). A Grader is an anchor's view of them: apply() grades
points in their region's frame (the same heights whatever the anchor: the anchor never enters the sums), plans() and
_cell_plans() give the plans' vertices mapped exactly into the anchor's world (geotiles.world_xy_map). A region's
planner reads a neighbouring region's cells mapped into its own frame (_Planner._import).

GradedDem(dem, grader) is a drop-in for dem.Dem where only sample() is graded; use it for the block
and the levels <= 0 tiles only (coarser tiles sink under the block and cover far more ground).
"""

from runtimeprofiler import step as profile_step
import math
import threading
from collections import OrderedDict
import time
from runtimeprofiler import ThreadPoolExecutor
import numpy as np
import mapcells
import roads as roadsmod
# A single-span bridge and the roads under it solved as one height problem (crossings.py; BNE_INTERCHANGE=1, off by
# default): its ends raised, or the roads under it lowered, whichever costs less, from the ground they can be trusted
# for (not lidar's fill under the deck, not an abutment on its slope). Replaces the dip, its lidar exception and the
# ramps for those bridges (a motorway's too); the rules above decide for every other.
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
    LINK_BESIDE_M,  # noqa: F401
    LINK_BESIDE_ON,  # noqa: F401
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
    beside_fixed_span,  # noqa: F401
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
from gradchains import _ChainsMixin
from gradunder import _UnderpassMixin
from gradcw import _CarriagewaysMixin
from gradends import _EndsMixin
from gradtunnels import _TunnelsMixin


class _Planner(_ChainsMixin, _UnderpassMixin, _CarriagewaysMixin, _EndsMixin, _TunnelsMixin):
    """The road plans of one planning region (geotiles.PLAN_Z), in its frame, per map cell, made once (worker threads;
    it fetches the road cells it lacks). store: the PlanStore it belongs to (its other regions, its settings)."""

    def __init__(self, frame, mapdata, elev, widths=None, store=None, region=None):
        self.frame, self.md, self.elev, self.widths = frame, mapdata, elev, widths
        self.store, self.region = store, region
        self.cells = {}
        self.failed = {}        # cell key -> monotonic time of the last failed fetch
        self.lock = threading.Lock()
        self.key_locks = {}     # cell key -> Lock: one thread plans a cell, the others wait for it
        self.covers = {}        # cell key -> [(xy (k, 2), half width)]: the tunnels whose ground over them stays
        self.keeps = {}         # cell key -> [(xy (k, 2), half width)]: the UNGRADED roads' carriageways
        self.opens = {}         # cell key -> [(xy (k, 2), Spec)]: the tunnels' open sections (roads in the open)

    # the store's settings (set on the anchors' Graders after they are made): shared: plancache.PlanCache (the cells'
    # plans shared with the server's other processes); wet(lat, lon) -> water depth (m): tunnels under water are left
    # out, as roads.TunnelProfiles leaves them (their "open" sections graded a dam across the river: the Maastunnel's
    # bed raised from -6 m to -0.4 m, storm-ace's GitHub #85); dem: its urgent lane (_plan_missing)
    def _store_attr(name):
        def get(self):
            own = self.__dict__.get("_own_" + name, _UNSET)
            if own is not _UNSET:
                return own
            return getattr(getattr(self, "store", None), name, None)

        def put(self, v):           # (a planner of its own: tests)
            self.__dict__["_own_" + name] = v
        return property(get, put)
    shared = _store_attr("shared")
    wet = _store_attr("wet")
    water = _store_attr("water")
    tunnel_raw = _store_attr("tunnel_raw")
    dem = _store_attr("dem")
    del _store_attr

    def _foreign(self, key):
        """The planner of cell key's region when it is not this one's (None: this one plans it)."""
        if self.store is None:
            return None
        import geotiles
        r = geotiles.region_of_cell(key)
        return None if r == self.region else self.store.planner(r)

    def _import(self, src, key):
        """Cell key planned by src (another region's planner), its vertices mapped exactly into this frame."""
        plans = src._cell_plans(key)
        with src.lock:
            if key not in src.cells:            # (its fetch failed: asked again later)
                return []
            keeps, covers, opens = src.keeps.get(key, []), src.covers.get(key, []), src.opens.get(key, [])
        import geotiles

        def m(xy):
            return geotiles.world_xy_map(src.frame, self.frame, xy)
        got = ([(fid, mapped_plan(p, m)) for fid, p in plans], [(m(xy), hw) for xy, hw in keeps],
               [(m(xy), w) for xy, w in covers], [(m(xy), sp) for xy, sp in opens])
        with self.lock:
            self.cells.setdefault(key, got[0])
            self.keeps.setdefault(key, got[1])
            self.covers.setdefault(key, got[2])
            self.opens.setdefault(key, got[3])
            return self.cells[key]

    def _cell_plans(self, key):
        with self.lock:
            got = self.cells.get(key)
            kl = self.key_locks.setdefault(key, threading.Lock())
        if got is not None:
            return got
        src = self._foreign(key)
        if src is not None:
            with kl:
                return self._import(src, key)
        with kl:
            return self._plan_cell(key)

    def _plan_cell(self, key):
        with self.lock:
            got = self.cells.get(key)
            failed = self.failed.get(key)
        if got is not None:
            return got
        if failed is not None and time.monotonic() - failed < RETRY_S:
            return []               # offline: no block waits on the same failing fetch again
        shared, mine = (self.shared.claim(key) if self.shared is not None else (None, False))
        try:
            import revisit
            # (this thread's faults: the planning threads of a process plan cells side by side, and another's fault
            # kept this one's complete plans from later visits and baked packs: 2026-11-02)
            f0, l0 = revisit.thread_faults(), revisit.thread_lidar_faults()
            if shared is not None:
                got = shared
            else:
                import stackdump    # (a slow plan holds every process waiting for it: say where it waits)
                watch = stackdump.watch(PLAN_SLOW_S, f"cell {key}'s plans still being made after {PLAN_SLOW_S:.0f} s")
                try:
                    got = self._make_cell(key)
                finally:
                    watch.cancel()
            if got is None:
                return []
            if mine:        # (kept for later visits too when complete: nothing it read failed or was missing)
                self.shared.store(key, *got, persist=revisit.thread_faults() == f0
                                  and revisit.thread_lidar_faults() == l0)
        finally:
            if mine:
                self.shared.release(key)
        plans, keeps, covers, opens = got
        with self.lock:
            self.cells.setdefault(key, plans)
            self.keeps.setdefault(key, keeps)
            self.covers.setdefault(key, covers)
            self.opens.setdefault(key, opens)
            return self.cells[key]

    def _make_cell(self, key):
        """(plans, keeps, covers, opens) of cell key, or None when its roads could not be fetched."""
        widths = getattr(self, "widths", None)
        if widths is not None and hasattr(widths, "prefetch"):
            # the cell's buildings (its roads' widths) fetched while its roads are: one after the other before
            def walls():
                try:
                    widths.prefetch([key])
                except Exception as e:
                    log.debug("grading: walls of %s: %s: %s", key, type(e).__name__, e)
            threading.Thread(target=walls, daemon=True, name="walls-ahead").start()
        try:
            feats = self.md.cell(key, layers=("roads",), kind="demand").layers.get("roads")
        except Exception as e:      # no road data: ungraded (asked again after RETRY_S)
            log.debug("grading: roads of %s: %s: %s", key, type(e).__name__, e)
            with self.lock:
                self.failed[key] = time.monotonic()
            return None
        plans = []
        if feats is not None and len(feats):
            specs = roadsmod.road_specs(feats, None, self.widths)
            parts, ids, clipped, rings = [], [], [], []
            box = self._clip_box(key)
            keeps = []
            for i in range(len(feats)):
                if specs[i] is not None and specs[i].cls in UNGRADED:
                    for ring in roadsmod.ground_parts(feats, i):
                        e, n, _ = self.frame.geodetic_to_enu(ring[:, 1], ring[:, 0], 0.0)
                        keeps.append((np.stack([np.asarray(e), np.asarray(n)], axis=1),
                                      specs[i].width / 2 + UNGRADED_MARGIN_M))
                    continue
                if specs[i] is None:
                    continue
                for ring in roadsmod.ground_parts(feats, i):     # not bridges / tunnels
                    pieces = self._near_cell(np.asarray(ring, np.float64), box)
                    for r in pieces:
                        parts.append((r, specs[i]))
                        ids.append(str(feats.ids[i]))
                        clipped.append(len(pieces) > 1 or r is not ring)
                        rings.append(roadsmod.is_ring(feats, i))
            pspecs = []
            abut = None
            if (ABUT_FILL_ON or APPROACH_ON) and parts:
                try:
                    pe = np.array([[r[0], r[-1]] for r, _sp in parts], np.float64).reshape(-1, 2)
                    e, n, _ = self.frame.geodetic_to_enu(pe[:, 1], pe[:, 0], 0.0)
                    abut = self._abut_ends(key, np.stack([np.asarray(e), np.asarray(n)], axis=1)).reshape(-1, 2)
                except Exception as e:      # no fills rather than no grading
                    log.debug("grading: abutments of %s: %s: %s", key, type(e).__name__, e)
            ext = pins = None
            if CHAIN_ON and parts:
                try:
                    self._chain_cands(key, feats, specs)        # (this cell's own: its specs are at hand)
                    ext = self._extensions(parts, ids)
                    pins = self._pins(parts, ids, ext)
                except Exception as e:      # each part smoothed alone rather than no grading
                    log.debug("grading: road chains of %s: %s: %s", key, type(e).__name__, e)
            pclip = []
            pparts = []                         # (fid, spec, lon/lat ring) of each plan (roadcorners.grade_rows)
            coarse = COARSE_EASE_ON and self._coarse(key)
            for fid, part, cl, p in zip(ids, parts, clipped, plan_roads(self.frame, parts, self.elev, abut, ext, pins,
                                                                         coarse=coarse, obs=self._obs(key),
                                                                         decks=self._decks_near(key), rings=rings)):
                if p is not None:
                    p.rid = road_id(fid)
                    plans.append((fid, p))
                    pspecs.append(part[1])
                    pclip.append(cl)
                    pparts.append((fid, part[1], part[0]))
            if NEIGHBOUR_ON and plans and self._coarse(key):
                try:
                    r = 2.0 * NEIGHBOUR_TREND_SIGMA_M
                    couple_neighbours([p for _f, p in plans], pspecs,
                                      lambda xy: field_heights(self.frame, self.elev, xy, r, NEIGHBOUR_TREND_SIGMA_M))
                except Exception as e:      # each road on its own rather than no grading
                    log.debug("grading: neighbours of %s: %s: %s", key, type(e).__name__, e)
            pad_nodes = None
            if JUNCTION_SNAP_ON and plans:
                try:
                    snap_junctions([p for _f, p in plans], pspecs)
                    pad_nodes = junction_pads([p for _f, p in plans], pspecs)
                except Exception as e:      # blended as before rather than no grading
                    log.debug("grading: junctions of %s: %s: %s", key, type(e).__name__, e)
            if COARSE_EASE_ON and plans:
                try:
                    self._ease_coarse(key, plans, pspecs)
                except Exception as e:      # as they were rather than no grading
                    log.debug("grading: coarse easing of %s: %s: %s", key, type(e).__name__, e)
            if APPROACH_ON and plans:
                try:
                    self._lift_approaches(key, plans, pspecs)
                except Exception as e:      # ungraded approaches rather than no grading
                    log.debug("grading: approaches of %s: %s: %s", key, type(e).__name__, e)
            # (the carriageways' profiles before the dips: a motorway under another road's low deck still dips)
            if CW_ON and plans:
                try:
                    self._apply_carriageways(plans, pspecs)
                except Exception as e:      # as they were rather than no grading
                    log.debug("grading: carriageways of %s: %s: %s", key, type(e).__name__, e)
            if UNDER_ON and plans:
                try:
                    self._lower_underpasses(key, plans)
                except Exception as e:      # as the data has it rather than no grading
                    log.debug("grading: underpasses of %s: %s: %s", key, type(e).__name__, e)
            if JUNCTION_SNAP_ON and plans and (APPROACH_ON or UNDER_ON):
                # the side roads again, at their roads' heights after the approaches' lifts and the underpasses' dips
                # (snapped before them, every alley along Baltimore's Haven Street stood up to 3.5 m over the street
                # the dip had lowered: a hump at each, the user 2026-10-28)
                try:
                    snap_junctions([p for _f, p in plans], pspecs)
                    junction_pads([p for _f, p in plans], pspecs, pad_nodes)
                except Exception as e:
                    log.debug("grading: junctions after the dips of %s: %s: %s", key, type(e).__name__, e)
            if plans:
                try:            # a pass road's widened hairpins (roads.HAIRPIN_*): its core wider round each
                    hairpin_cores(plans, pspecs, self.frame)
                except Exception as e:      # the road's own core rather than none
                    log.debug("grading: hairpins of %s: %s: %s", key, type(e).__name__, e)
            if CORNERS_ON and plans:
                # the rounded corners of curbless junctions (roadcorners.py): the ground under each graded to its roads'
                # solved heights (trimmed to the cell's box with the roads' plans below)
                try:
                    import roadcorners
                    rings = roadcorners.ring_ids(feats)
                    crows, heights = roadcorners.grade_rows(pparts, [p for _f, p in plans], self.frame, rings)
                    corner_plans = roadcorners.grade(crows, rings, self._clip_box(key, 0.0), heights)
                    plans.extend(corner_plans)
                    pclip.extend([True] * len(corner_plans))
                except Exception as e:      # the roads' own grading rather than none
                    log.debug("grading: road corners of %s: %s: %s", key, type(e).__name__, e)
            # a long road's piece (planned with CLIP_MARGIN_M of the road either side for its profile) kept only within
            # the cell's own box: every stretch of road in one cell's plans, once (overlapping pieces from neighbouring
            # cells counted a road up to 9 times, and every batch of points graded differently: patches of ground
            # through the Stelvio's asphalt, GitHub #108's 2.5 m step at a bridge end in Meppen, 2026-10-21)
            own = self._clip_box(key, 0.0)
            sub = feats.attrs.get("subtype")
            if RAIL_GRADE_ON and own is not None and sub is not None and (np.asarray(sub).astype(str) == "rail").any():
                try:            # the railways' bed lines (railways.grade_plans): cuttings and embankments for the track
                    import railways     # (the cells around only where this one has a line: their fetches cost)
                    rcells = [self.md.cell(k, layers=("roads",), kind="demand").layers.get("roads")
                              for k in railways.cells_near(self.frame, own)]
                    for fid, xy, z in railways.grade_plans(self.frame, rcells, own, self.elev):
                        p = Plan(xy, z, railways.GRADE_HALF_M)
                        p.rid = road_id(RAIL_TAG + fid)
                        plans.append((RAIL_TAG + fid, p))
                        pclip.append(True)
                except Exception as e:      # no rail grading rather than no grading
                    log.debug("grading: railways of %s: %s: %s", key, type(e).__name__, e)
            # every plan only within the cell's own box, so each stretch of road is planned by one cell (its dips
            # made with every road around: _lower_underpasses); the tunnels' ends graded as before, a part shorter
            # than LONG_PART_M by all of it (TunnelProfiles.ends grades them by the roads' plans alone)
            tplans = plans
            if own is not None:
                # (by the cell's quad in this frame, not its box: a cell at a planning region's edge stands turned
                # ~0.16 deg in it, and boxes left slivers planned twice or not at all between neighbouring cells)
                quad = self._clip_quad(key)
                kept, tplans = [], []
                for (fid, p), cl in zip(plans, pclip):
                    pieces = trim_plan(p, own if quad is None else quad)
                    kept.extend((fid, q) for q in pieces)
                    tplans.extend((fid, q) for q in (pieces if cl else [p]))
                plans = kept
            portals = {}
            secs = self._tunnel_sections(feats, key, [p for _f, p in tplans], portals)
            plans.extend((fid + TUNNEL_TAG, Plan(xy, z, spec.width / 2 + shoulder(spec)))
                         for fid, spec, xy, z, k in secs if k == 0)
            plans.extend(self._mouth_plans(secs, portals))
            covers = [(xy, spec.width / 2 + TUNNEL_KEEP_M) for _f, spec, xy, _z, k in _twin_trimmed(secs) if k]
            opens = [(xy, spec) for _f, spec, xy, _z, k in secs if k == 0]
        else:
            covers, opens, keeps = [], [], []
        return plans, keeps, covers, opens

    # ---- carriageways (carriageways.py): a motorway's roads and decks as one profile ----

    def planned(self, key):
        """True once cell key's plans exist (made, or failed recently)."""
        src = self._foreign(key)
        if src is not None:
            return src.planned(key)
        with self.lock:
            f = self.failed.get(key)
            return key in self.cells or (f is not None and time.monotonic() - f < RETRY_S)

    def warm(self, key):
        """Make cell key's plans now (worker threads: the lookahead ahead of the player)."""
        self._cell_plans(key)

    def _plan_missing(self, keys):
        """The cells of keys not planned yet, planned at once on up to PLAN_THREADS threads (their map data and
        elevation downloads overlap: a fresh place's first block planned its 3-4 cells one after another, 27-35 s of
        its wait, 2026-10-22); the caller's urgent elevation lane (a teleport's first block) carried over."""
        with self.lock:
            missing = [k for k in keys if k not in self.cells]
        if len(missing) < 2:
            return
        dem = getattr(self, "dem", None)
        urgent = dem is not None and getattr(dem, "is_urgent", lambda: False)()

        def one(k):
            if urgent:
                with dem.urgent():
                    return self._cell_plans(k)
            return self._cell_plans(k)
        with ThreadPoolExecutor(min(len(missing), PLAN_THREADS), thread_name_prefix="plans") as ex:
            list(ex.map(one, missing))

    def plans(self, lat, lon, roads_only=False):
        """The Plans touching the points' area (each road once); roads_only: not the tunnels' (TUNNEL_TAG)."""
        return self._plan_set(self._keys(lat, lon), roads_only)[0]

    @staticmethod
    def _keys(lat, lon):
        """The map cells whose plans can reach the points."""
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        pad = (T_MAX + 40.0) / 111320.0
        padlon = pad / max(0.05, math.cos(math.radians(float(np.mean(lat)))))
        keys = mapcells.cells_for_points(lat, lon, pad, padlon)
        if len(keys) > SPREAD_CELLS:        # points far apart: only the cells around them
            keys = mapcells.cells_near_points(lat, lon, pad, padlon)
        return keys

    def _plan_set(self, keys, roads_only):
        """(plans, PlanPack of them) of the cells keys: each road once, in the order of their keys. Remembered once all
        the cells are planned (a road tile asks for the same few cells' plans dozens of times: merging them cost a
        Canton tile set ~1.6 s, 2026-10-25)."""
        self._plan_missing(keys)
        mk = (tuple(sorted(str(k) for k in keys)), bool(roads_only))
        memo = self.__dict__.setdefault("_plan_sets", {})
        with self.lock:
            hit = memo.get(mk)
        if hit is not None:
            return hit
        got = {}
        for key in keys:
            for fid_part, p in self._cell_plans(key):
                if roads_only and TUNNEL_TAG in fid_part:
                    continue
                got.setdefault((fid_part, float(p.xy[0, 0]), float(p.xy[0, 1])), p)
        # in the order of their keys: grade_points' sums run in plan order, so every caller (the block, a road
        # tile, a far tile) gets the same heights at a point whatever other cells its points reached
        plans = [got[k] for k in sorted(got)]
        out = (plans, PlanPack(plans))
        with self.lock:
            if all(k in self.cells for k in keys):      # (not while a cell's fetch failed: it is asked again)
                if len(memo) >= PLAN_SETS_MAX:
                    memo.pop(next(iter(memo)))
                memo[mk] = out
        return out

    @profile_step('Road grading')
    def apply(self, lat, lon, base, lift=0.0, roads_only=False, planned_only=False):
        """base (the ungraded heights at the points) graded under the roads. Same shape as base.
        lift: m added under the roads (tapering with the grading's weight): the physics block and the
        road mesh use ROAD_LIFT, the visible ground tiles 0, so the road mesh lies exactly on the
        ground the tyres touch while the visible ground stays a little below it (no z-fighting;
        2026-10-11 the user saw the tyres sink into a road mesh lifted above the physics ground).
        planned_only: graded only under the cells planned already (none planned here: the AI graph's wide window, whose
        planning held a Windows PC's road processes for minutes, 2026-10-25)."""
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        base = np.asarray(base, np.float64)
        shape = base.shape
        if base.size == 0:
            return base
        keys = self._keys(lat, lon)
        if planned_only:
            with self.lock:
                have = {k for k in keys if k in self.cells}
            for k in keys:              # (a neighbouring region's, planned there: read in)
                src = None if k in have else self._foreign(k)
                if src is not None:
                    with src.lock:
                        if k in src.cells:
                            have.add(k)
            keys = [k for k in keys if k in have]
            if not keys:
                return base
        plans, pack = self._plan_set(keys, roads_only)
        if not plans and not (RACEWAY_SMOOTH_ON and self._has_keeps()):
            return base
        e, n, _ = self.frame.geodetic_to_enu(lat.ravel(), lon.ravel(), 0.0)
        e, n = np.asarray(e), np.asarray(n)
        out = grade_points(e, n, base.ravel(), pack, lift) if plans else np.array(base.ravel(), np.float64, copy=True)
        race = self._on_kept(e, n)
        if race is not None:                # (the terrain is a racetrack's surface: GitHub #143, smoothed: #833)
            out[race] = self._raceway_ground(e[race], n[race], base.ravel()[race])
        keep = self._over_tunnels(e, n)
        if keep is not None:
            # (a small change stays: a road over the tunnel lies on its own graded profile, the ground over the tunnel
            # left ungraded under it stood a few cm over the E6's lanes as patches of bare ground, GitHub #2)
            keep &= np.abs(out - base.ravel()) >= OVER_TUNNEL_SMALL_M
            out[keep] = base.ravel()[keep]
        return out.reshape(shape)

    def open_tunnels(self, lat, lon):
        """[(lat (k,), lon (k,), Spec)] of the tunnels' open sections planned near the points: roads on the ground
        that the map calls tunnels (surface.paint_roads paints them: grass grew over the open road of Central Park's
        79th Street Transverse, 2026-10-17)."""
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        if lat.size == 0:
            return []
        for key in mapcells.cells_for_points(lat, lon):
            self._cell_plans(key)
        with self.lock:
            secs = [o for v in self.opens.values() for o in v]
        out, seen = [], set()
        for xy, spec in secs:
            k = (round(float(xy[0, 0]), 2), round(float(xy[0, 1]), 2), len(xy))
            if k in seen:
                continue
            seen.add(k)
            la, lo, _ = self.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
            la, lo = np.asarray(la), np.asarray(lo)
            if la.max() < lat.min() - 0.001 or la.min() > lat.max() + 0.001 or \
                    lo.max() < lon.min() - 0.0015 or lo.min() > lon.max() + 0.0015:
                continue
            out.append((la, lo, spec))
        return out

    def _has_keeps(self):
        with self.lock:
            return any(self.keeps.values())



def mapped_plan(p, fn):
    """Plan p with its vertices through fn (xy (n, 2) -> (n, 2): geotiles.world_xy_map); heights, widths, cuttings and
    banking as they are (geodetic, or along / across the road)."""
    q = Plan(fn(p.xy), p.z, p.half, p.abut, p.cut, p.road_half)
    q.rid, q.max_wall, q.bank = p.rid, p.max_wall, getattr(p, "bank", None)
    q.steep = getattr(p, "steep", False)
    set_widen(q, getattr(p, "widen", None))
    return q


class PlanStore:
    """Every road plan of a process, whatever the anchor: one _Planner per planning region (geotiles.PLAN_Z), made on
    first use. mapdata, elev (lat, lon -> m at PROFILE_ZOOM, raw), widths as the Grader's; wet, dem, shared (the
    planners' settings) set after it is made, as on a Grader (which forwards them here)."""

    def __init__(self, mapdata, elev, widths=None):
        self.md, self.elev, self.widths = mapdata, elev, widths
        global _ACTIVE_STORE
        _ACTIVE_STORE = self
        self.wet = None
        self.water = None
        self.tunnel_raw = None
        self.dem = None
        self.shared = None
        self.lock = threading.Lock()
        self.planners = OrderedDict()   # (rx, ry) -> _Planner, the least recently used first

    @staticmethod
    def frame_of(region):
        """The planning frame of a region (geotiles.region_frame; a test replaying samples captured in one frame
        sets its own)."""
        import geotiles
        return geotiles.region_frame(*region)

    def planner(self, region):
        region = (int(region[0]), int(region[1]))
        with self.lock:
            got = self.planners.get(region)
            if got is None:
                got = self.planners[region] = _Planner(self.frame_of(region), self.md, self.elev, self.widths,
                                                       store=self, region=region)
                # (the regions far behind go: their plans are on disk, plancache; a drive across a continent kept
                # every cell it passed)
                while len(self.planners) > PLANNERS_KEPT:
                    self.planners.popitem(last=False)
            else:
                self.planners.move_to_end(region)
            return got

    def planner_of_cell(self, key):
        import geotiles
        return self.planner(geotiles.region_of_cell(key))

    def planners_of(self, lat, lon):
        """[(planner, index array)] of the points' regions (most calls: one)."""
        import geotiles
        lat = np.asarray(lat, np.float64).ravel()
        lon = np.asarray(lon, np.float64).ravel()
        rx, ry = geotiles.region_of(lat, lon)
        rx, ry = np.atleast_1d(rx), np.atleast_1d(ry)
        if len(rx) and (rx == rx[0]).all() and (ry == ry[0]).all():
            return [(self.planner((int(rx[0]), int(ry[0]))), None)]
        out = []
        for a, b in sorted({(int(a), int(b)) for a, b in zip(rx, ry)}):
            out.append((self.planner((a, b)), np.flatnonzero((rx == a) & (ry == b))))
        return out


_ACTIVE_STORE = None    # the process's PlanStore (the last made): the carriageways' deck heights (_deck_z_hook)


def _deck_z_hook(frame, lines, fids=None):
    """roadbase.DECK_Z_HOOK: per deck line (frame ENU (k, 2+)), its heights (frame up m) from its carriageway's profile, or
    None where it is not one's."""
    store = _ACTIVE_STORE
    if store is None:
        return None
    out = []
    for n, ln in enumerate(lines):
        fid = None if fids is None else str(fids[n])
        ln = np.asarray(ln, np.float64)[:, :2]
        la, lo, _ = frame.enu_to_geodetic(ln[:, 0], ln[:, 1], 0.0)
        la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
        pl, _idx = store.planners_of(la[len(la) // 2:len(la) // 2 + 1], lo[len(lo) // 2:len(lo) // 2 + 1])[0]
        e, n, _ = pl.frame.geodetic_to_enu(la, lo, 0.0)
        xy = np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64)], axis=1)
        z = pl._cw_z_at(xy, xy, fid)
        if z is None:
            out.append(None)
            continue
        out.append(np.asarray(frame.geodetic_to_enu(la, lo, np.asarray(z, np.float64))[2], np.float64))
    return out if any(o is not None for o in out) else None


if CW_ON:
    import roadbase                     # (where roads' bridge code reads the hook: roadbridges.py)
    roadbase.DECK_Z_HOOK = _deck_z_hook


class Grader:
    """The road plans as one anchor frame sees them (the PlanStore's, store: shared by the process's anchors; its own
    without one). Heights (apply) are the planners', in their regions' frames: the same at a place whatever the
    anchor. Plans and outlines handed out (plans, _cell_plans) are in this frame's world."""

    def __init__(self, frame, mapdata, elev, widths=None, store=None):
        self.frame = frame
        self.store = store if store is not None else PlanStore(mapdata, elev, widths)
        self.md, self.elev, self.widths = self.store.md, self.store.elev, self.store.widths
        self.lock = threading.Lock()
        self._mapped = OrderedDict()        # cell key -> its plans in this frame

    def _set(name):
        return property(lambda self: getattr(self.store, name), lambda self, v: setattr(self.store, name, v))
    wet = _set("wet")
    water = _set("water")
    tunnel_raw = _set("tunnel_raw")
    dem = _set("dem")
    shared = _set("shared")
    del _set

    def planned(self, key):
        """True once cell key's plans exist (made, or failed recently), by any anchor of the store."""
        return self.store.planner_of_cell(key).planned(key)

    def warm(self, key):
        """Make cell key's plans now (worker threads: the lookahead ahead of the player)."""
        self.store.planner_of_cell(key)._cell_plans(key)

    def _cell_plans(self, key):
        """[(fid, Plan)] of cell key, in this frame's world (mapped from its region's frame; kept)."""
        with self.lock:
            got = self._mapped.get(key)
            if got is not None:
                self._mapped.move_to_end(key)
                return got
        pl = self.store.planner_of_cell(key)
        plans = pl._cell_plans(key)
        with pl.lock:
            done = key in pl.cells
        import geotiles

        def m(xy):
            return geotiles.world_xy_map(pl.frame, self.frame, xy)
        got = [(fid, mapped_plan(p, m)) for fid, p in plans]
        if done:
            with self.lock:
                got = self._mapped.setdefault(key, got)
                while len(self._mapped) > MAPPED_MAX:
                    self._mapped.popitem(last=False)
        return got

    _clip_box = _Planner._clip_box          # (in this frame: self.frame only)
    _keys = staticmethod(_Planner._keys)
    _over_tunnels = _Planner._over_tunnels  # (a namespace with covers and lock: tests)
    _on_kept = _Planner._on_kept
    _raceway_ground = _Planner._raceway_ground

    def plans(self, lat, lon, roads_only=False):
        """The Plans touching the points' area (each road once), in this frame's world; roads_only: not the tunnels'
        (TUNNEL_TAG). In the order of the plans' own region (as _Planner._plan_set)."""
        keys = self._keys(lat, lon)
        got = {}
        for key in keys:
            for fid_part, p in self._cell_plans(key):
                if roads_only and TUNNEL_TAG in fid_part:
                    continue
                got.setdefault((fid_part, float(p.xy[0, 0]), float(p.xy[0, 1])), p)
        return [got[k] for k in sorted(got)]

    def prepare(self, lat, lon):
        """Plan the cells the points' area needs now (several at once, _Planner._plan_missing), returning when they are:
        a block's builder starts it beside its own work (its sea floor waited on the water cells, then its grading on
        the road cells: 4 + 9 s of a fresh place's first block, one after the other, Lincoln 2026-10-30)."""
        lat = np.asarray(lat, np.float64).ravel()
        lon = np.asarray(lon, np.float64).ravel()
        for pl, idx in self.store.planners_of(lat, lon):
            keys = pl._keys(lat[idx], lon[idx])
            pl._plan_missing(keys)      # (two or more at once; one is planned here)
            for k in keys:
                pl._cell_plans(k)

    @profile_step('Road grading')
    def apply(self, lat, lon, base, lift=0.0, roads_only=False, planned_only=False):
        """_Planner.apply by the points' regions (the anchor plays no part). Same shape as base."""
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        base = np.asarray(base, np.float64)
        if base.size == 0:
            return base
        parts = self.store.planners_of(lat, lon)
        if len(parts) == 1:
            return parts[0][0].apply(lat, lon, base, lift, roads_only, planned_only)
        la, lo, b = lat.ravel(), lon.ravel(), base.ravel()
        out = np.array(b, np.float64, copy=True)
        for pl, idx in parts:
            out[idx] = pl.apply(la[idx], lo[idx], b[idx], lift, roads_only, planned_only)
        return out.reshape(base.shape)

    def open_tunnels(self, lat, lon):
        """_Planner.open_tunnels by the points' regions: [(lat (k,), lon (k,), Spec)], each section once."""
        lat = np.asarray(lat, np.float64).ravel()
        lon = np.asarray(lon, np.float64).ravel()
        if lat.size == 0:
            return []
        out, seen = [], set()
        for pl, idx in self.store.planners_of(lat, lon):
            for la, lo, spec in pl.open_tunnels(lat if idx is None else lat[idx], lon if idx is None else lon[idx]):
                k = (round(float(la[0]), 7), round(float(lo[0]), 7), len(la))
                if k not in seen:
                    seen.add(k)
                    out.append((la, lo, spec))
        return out


def trim_plan(p, box):
    """The pieces of Plan p within box (x0, y0, x1, y1), or a convex polygon (k, 2): cut at its edges (earthfast.clip_segments), heights and
    cuttings with them; an end not at p's own end leads onto no bridge. [Plan]."""
    import fast
    xy = np.ascontiguousarray(np.asarray(p.xy, np.float64)[:, :2])
    if len(xy) < 2:
        return [p]
    if np.ndim(box) == 2:               # a convex polygon (a cell's quad: _Planner._clip_quad)
        fast.require(104, "clip_segments_convex")
        t = np.asarray(fast._ef.clip_segments_convex(xy, np.ascontiguousarray(box, np.float64)))
    else:
        fast.require(44, "clip_segments")
        t = np.asarray(fast._ef.clip_segments(xy, tuple(float(v) for v in box)))
    ok = np.isfinite(t[:, 0])
    if ok.all() and (t[:, 0] == 0.0).all() and (t[:, 1] == 1.0).all():
        return [p]
    out, i, n = [], 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and t[j, 1] == 1.0 and t[j + 1, 0] == 0.0:
            j += 1
        t0, t1 = t[i, 0], t[j, 1]
        a = xy[i] + (xy[i + 1] - xy[i]) * t0
        b = xy[j] + (xy[j + 1] - xy[j]) * t1
        za = p.z[i] + (p.z[i + 1] - p.z[i]) * t0
        zb = p.z[j] + (p.z[j + 1] - p.z[j]) * t1
        pxy = np.vstack([a, xy[i + 1:j + 1], b])
        pz = np.r_[za, p.z[i + 1:j + 1], zb]
        pb = getattr(p, "bank", None)
        if pb is not None:
            pb = np.r_[pb[i] + (pb[i + 1] - pb[i]) * t0, pb[i + 1:j + 1], pb[j] + (pb[j + 1] - pb[j]) * t1]
        pw = getattr(p, "widen", None)
        if pw is not None:
            pw = np.r_[pw[i] + (pw[i + 1] - pw[i]) * t0, pw[i + 1:j + 1], pw[j] + (pw[j + 1] - pw[j]) * t1]
        if np.hypot(*(pxy[-1] - pxy[0])) > 1e-6 or len(pxy) > 2:
            q = Plan(pxy, pz, p.half, (p.abut[0] and i == 0 and t0 == 0.0, p.abut[1] and j == n - 1 and t1 == 1.0),
                     None if p.cut is None else np.asarray(p.cut)[i:j + 1].copy(), p.road_half)
            q.rid = getattr(p, "rid", -1)
            q.max_wall = getattr(p, "max_wall", WALL_MAX_H_M)
            q.bank = pb
            q.steep = getattr(p, "steep", False)
            set_widen(q, pw)
            out.append(q)
        i = j + 1
    return out


class GradedDem:
    """dem.Dem with sample() graded under the roads; everything else passes through."""

    def __init__(self, dem, grader):
        self._dem, self._grader = dem, grader

    def sample(self, lat, lon, zoom, progress=None):
        base = self._dem.sample(lat, lon, zoom, progress) if progress is not None else self._dem.sample(lat, lon, zoom)
        # the land off the roads is rougher than the roads themselves (the grading flattens the road's own strip)
        base = np.asarray(base, np.float64) + terrain_bumps(np.asarray(lat, np.float64), np.asarray(lon, np.float64))
        return self._grader.apply(lat, lon, base)

    def __getattr__(self, name):
        return getattr(self._dem, name)
