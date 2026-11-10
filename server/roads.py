"""Roads near the player (docs/ROADMAP.md v1.6 "Roads"; first cut 2026-10-11): Overture road segments
from the map-data cache (mapdata.py "roads"), drawn by the mod as DecalRoads on the physics ground, and
painted into the surface masks (surface.py) so the tyres feel asphalt or dirt.

Tiles: TILE x TILE m squares of the anchor's ENU frame, keyed (anchor, tx, ty). The Tracker of a
connection keeps the tiles within SEND_M of the player sent (built in a worker, written to
`roads_<a>_<tx>_<ty>.json` in the session folder, announced with a `roads_tile` message) and drops
those beyond DROP_M (`roads_drop`). An anchor change re-sends everything in the new frame; the mod
drops the old anchor's roads then.

A tile's file is a list of chunks, each one DecalRoad:
    {"m": material, "w": width m, "tl": texture length m, "p": render priority, "n": [[x, y, z], ...],
     "h": curb height m (sidewalks only: the mod raises a slab this far above the road near the player)}
with x, y, z in the anchor frame (z on the DEM, the DecalRoad snaps its nodes to the ground anyway)
and at most MAX_NODES nodes. Roads are clipped to the tile (OVERLAP m past its edge), simplified
(RDP, SIMPLIFY_M) and split. Deterministic: a pure function of the cells' road data, the frame and
the DEM.

Not yet: grading the ground under roads, markings, junction shapes, bridges and tunnels (segments
flagged as bridge / tunnel or off ground level are left out), the navgraph.
"""

import json
import math
import os
import time
import numpy as np
import junctions
import mapgeom
import roadmesh
import staging
import roadbase
from roadbase import (
    BLD_PARTS_MAX,
    BUILD_PER_POSE,
    BUILT,
    CENTRE,
    COLLIDER_MARGIN_M,
    COLLIDER_PIECE_M,
    CORRIDOR_M,
    COVER_MARGIN_M,
    CUT_MARGIN,
    DASH_MIN_M,
    DIRT_LOOK,
    DROP_M,
    EDGE_DECAL,
    EDGE_DECAL_GAP_M,
    EDGE_LINE,
    EXACT_TOL_COL_M,
    EXACT_TOL_M,
    FAST_MS,
    GUTTER,
    LADDER_LINE_W,
    LEAD_MAX_M,
    LEAD_S,
    MIN_PIECE_M,
    NEAR_M,
    OVERLAP,
    SEND_M,
    SIDEWALK,
    SIDE_PENALTY,
    SOUND_BOX_M,
    SOUND_BRIDGE,
    SOUND_DZ_M,
    SOUND_EXTEND_M,
    SOUND_PORTAL,
    SOUND_PORTAL_IN_M,
    SOUND_PORTAL_UP,
    SOUND_SHORT_TUNNEL_M,
    SOUND_TUNNEL,
    SOUND_TUNNEL_OPEN,
    SOUND_TURN,
    SOUND_UNDER_M,
    TILE,
    TILE_FAILS_WARN,
    TILE_RETRY_S,
    TILE_STUCK_S,
    VERSION,
    _rule_value,
    densify,
    log,
    look_of,
    split,
    terrain_surface,
)
from roadspecs import (
    bridge_specs,
    copies_of,
    deck_width_fn,
    driving_offset,
    ground_parts,
    highway_shoulders,
    is_ring,
    joint_desc,
    lane_layout,
    lane_shifts,
    lane_tags,
    road_specs,
    road_tapers,
    shift_at,
    taper_widths,
)
from roadgeom import (
    _length,
    _on_level,
    _within_any,
    cut_outside,
    offset_polyline,
)
from roadwalks import (
    clip_polyline,
    drape_walks,
    extras,
    has_sidewalks,
    join_walk_ends,
    sidewalk_stop,
    simplify,
    walks_off_roads,
    way_lines,
)
from roadtiles import (
    box_owns,
    mark_poly,
    path_key,
    tile_bbox,
    tile_distance,
    tiles_near,
    view_key,
    wanted_tiles,
)
from roadbridges import (
    _chunk,
    _strip_open_ends,
    _surface_pieces,
)
from roadstructures import (
    _bridge_chunks,
    _corner_look,
    _deck_clearance,
    _going_on,
    _off_wall_tops,
    _on_box,
    _tunnel_chunks,
    cliff_rows,
    deck_shift_fn,
    deck_walks,
    drains,
    line_through,
    retaining_walls,
)
from roadbase import (  # (re-exported: callers use roads.<name>)
    ABUT_GROUND_M,  # noqa: F401
    APPROACH_GRADE,  # noqa: F401
    ARCH_SIDE_M,  # noqa: F401
    ARCH_SIZE_SKIP_M,  # noqa: F401
    ARCH_SOIL_MIN,  # noqa: F401
    ARTERIAL_GAP_M,  # noqa: F401
    ARTERIAL_KMH,  # noqa: F401
    AT_GRADE_M,  # noqa: F401
    AT_GRADE_REACH_M,  # noqa: F401
    BACK_TOP_OVER_M,  # noqa: F401
    BRIDGE_HEADROOM_MARGIN_M,  # noqa: F401
    CENTRE_CLASSES,  # noqa: F401
    CHAIN_CELLS,  # noqa: F401
    CHORD_KINK,  # noqa: F401
    CHORD_SHELF_M,  # noqa: F401
    CHORD_TOL,  # noqa: F401
    CLIFF_EDGE_M,  # noqa: F401
    CLIFF_MIN_CUT_M,  # noqa: F401
    CLIFF_OVER_M,  # noqa: F401
    CLIFF_REACH_M,  # noqa: F401
    CLIFF_ROAD_GAP_M,  # noqa: F401
    CLIFF_SHAPES,  # noqa: F401
    CLIFF_SINK_M,  # noqa: F401
    CLIFF_SPACING,  # noqa: F401
    CLIFF_TALL_M,  # noqa: F401
    CLIFF_WALLS_ON,  # noqa: F401
    COBBLE,  # noqa: F401
    COBBLE_ALIASES,  # noqa: F401
    CONCRETE,  # noqa: F401
    CONCRETE_WALL_CLASSES,  # noqa: F401
    CORNER_CONCRETE_M,  # noqa: F401
    DECK,  # noqa: F401
    DECK_CLEAR,  # noqa: F401
    DECK_CLEAR_REACH_M,  # noqa: F401
    DECK_CONCRETE,  # noqa: F401
    DECK_CONCRETE_CLASSES,  # noqa: F401
    DECK_CONCRETE_ON,  # noqa: F401
    DECK_CONCRETE_TILE_M,  # noqa: F401
    DECK_CONTEXT_M,  # noqa: F401
    DECK_CREST_R,  # noqa: F401
    DECK_CURVE_K,  # noqa: F401
    DECK_CURVE_KMIN,  # noqa: F401
    DECK_CURVE_MIN,  # noqa: F401
    DECK_CURVE_STEP,  # noqa: F401
    DECK_DESIGN_GRADE,  # noqa: F401
    DECK_EASE_M,  # noqa: F401
    DECK_FULL_DEPTH,  # noqa: F401
    DECK_GRADE,  # noqa: F401
    DECK_JOIN_M,  # noqa: F401
    DECK_MAX_GRADE,  # noqa: F401
    DECK_MIN_DEPTH,  # noqa: F401
    DECK_ROAD_CLEAR,  # noqa: F401
    DECK_ROAD_HALF,  # noqa: F401
    DECK_SIGMA,  # noqa: F401
    DECK_SIGMA_MAX,  # noqa: F401
    DECK_SIGMA_SHARE,  # noqa: F401
    DECK_SPAN_RATIO,  # noqa: F401
    DECK_STEEP_GRADE,  # noqa: F401
    DECK_STEP,  # noqa: F401
    DECK_WALK,  # noqa: F401
    DECK_WALK_CHAIN,  # noqa: F401
    DECK_WALK_LIFT_M,  # noqa: F401
    DIRT_MATERIALS,  # noqa: F401
    DIRT_SPLIT,  # noqa: F401
    DIRT_VARIATIONS,  # noqa: F401
    DRAIN,  # noqa: F401
    DRAIN_EVERY,  # noqa: F401
    DRAIN_IN,  # noqa: F401
    DUAL_ANGLE_DEG,  # noqa: F401
    DUAL_CLASSES,  # noqa: F401
    DUAL_DECK_M,  # noqa: F401
    DUAL_GAP_M,  # noqa: F401
    DUAL_MIN_MEDIAN_M,  # noqa: F401
    DUAL_MIN_W,  # noqa: F401
    DUAL_SHARE,  # noqa: F401
    EDGE_DECAL_AT,  # noqa: F401
    EDGE_LINE_CLASSES,  # noqa: F401
    EDGE_STRIP_CLASS_M,  # noqa: F401
    EDGE_STRIP_M,  # noqa: F401
    END_JOIN_COS,  # noqa: F401
    END_JOIN_GROUND_M,  # noqa: F401
    END_JOIN_M,  # noqa: F401
    END_JOIN_MIN_M,  # noqa: F401
    END_ON_MARGIN_M,  # noqa: F401
    END_ON_STEP_M,  # noqa: F401
    FACTS_FORMAT,  # noqa: F401
    FACTS_LOG,  # noqa: F401
    FACTS_SHORT_S,  # noqa: F401
    FACTS_WAIT_S,  # noqa: F401
    FULL_DEEP_M,  # noqa: F401
    FULL_MIN_M,  # noqa: F401
    GAP_DECK_M,  # noqa: F401
    GAP_DIP_M,  # noqa: F401
    GAP_JOIN_M,  # noqa: F401
    GAP_JUNCTION_M,  # noqa: F401
    GAP_LOW_M,  # noqa: F401
    GAP_SEA_M,  # noqa: F401
    GOING_ON_DEG,  # noqa: F401
    GOOD_SMOOTHNESS,  # noqa: F401
    GRASSY,  # noqa: F401
    GRAVEL,  # noqa: F401
    HAIRPIN_SPAN_M,  # noqa: F401
    HAIRPIN_WIDEN_M,  # noqa: F401
    HOLE_JOIN_M,  # noqa: F401
    HOLE_ROAD_MARGIN,  # noqa: F401
    HOME_RANGES_ON,  # noqa: F401
    JOIN_GAP_M,  # noqa: F401
    JOIN_TURN_DEG,  # noqa: F401
    LANE_CLASSES,  # noqa: F401
    LANE_FLOW_CLASSES,  # noqa: F401
    LANE_FLOW_MAX,  # noqa: F401
    LANE_LINE,  # noqa: F401
    LANE_M,  # noqa: F401
    LANE_W,  # noqa: F401
    LANE_W_DEFAULT,  # noqa: F401
    LIDAR_GAP_REACH_M,  # noqa: F401
    LIDAR_GAP_TOUCH_M,  # noqa: F401
    LIFT_EASE_M,  # noqa: F401
    LIFT_END_GRADE,  # noqa: F401
    LIFT_GRADE,  # noqa: F401
    LINE_RANK,  # noqa: F401
    LINK_LANE_M,  # noqa: F401
    LINK_M,  # noqa: F401
    LINK_SHOULDER_M,  # noqa: F401
    LOOK,  # noqa: F401
    LOW_HOLE_M,  # noqa: F401
    LOW_HOLE_STEP,  # noqa: F401
    MAJOR_LOW_H,  # noqa: F401
    MAX_NODES,  # noqa: F401
    MAX_SPACING,  # noqa: F401
    MEASURE_MAX,  # noqa: F401
    MEASURE_SKIP,  # noqa: F401
    MEDIAN_CURB_W,  # noqa: F401
    MEDIAN_MAX_M,  # noqa: F401
    MEDIAN_MIN_M,  # noqa: F401
    MEDIAN_SLAB_M,  # noqa: F401
    MERGE_TOUCH_M,  # noqa: F401
    MIN_BRIDGE_HEADROOM_M,  # noqa: F401
    MIN_PART,  # noqa: F401
    MOTORWAY_RURAL_M,  # noqa: F401
    MOUNTAIN_CLASSES,  # noqa: F401
    MOUNTAIN_MAX_LANES,  # noqa: F401
    MOUNTAIN_ON,  # noqa: F401
    MOUNTAIN_ROAD_M,  # noqa: F401
    MOUNTAIN_SPAN_M,  # noqa: F401
    MOUNTAIN_SWITCHBACKS,  # noqa: F401
    MOUNTAIN_TURN_DEG,  # noqa: F401
    NAV_RUN_M,  # noqa: F401
    NO_PARAPET_CLASSES,  # noqa: F401
    NO_WALK_KMH,  # noqa: F401
    OBS_BLEND_DG,  # noqa: F401
    OBS_BLEND_MIN_M,  # noqa: F401
    OBS_END_MAX_M,  # noqa: F401
    OBS_JOINT_GRADE,  # noqa: F401
    OBS_JOINT_SHORT_M,  # noqa: F401
    OBS_MAX_GRADE,  # noqa: F401
    ONEWAY_STYLE,  # noqa: F401
    OSM_LANES,  # noqa: F401
    OSM_WIDTH_M,  # noqa: F401
    PADS_ON,  # noqa: F401
    PAD_FORK_COS,  # noqa: F401
    PAD_FORK_MARGIN_M,  # noqa: F401
    PAD_FORK_OPEN_M,  # noqa: F401
    PAD_FORK_STRAIGHT_COS,  # noqa: F401
    PAD_GROUND_M,  # noqa: F401
    PAD_JOIN_M,  # noqa: F401
    PAD_JOIN_Z_M,  # noqa: F401
    PAD_MARGIN_M,  # noqa: F401
    PAD_OPEN_M,  # noqa: F401
    PAD_REACH_M,  # noqa: F401
    PAD_RIM_STEP_M,  # noqa: F401
    PAD_STEP_M,  # noqa: F401
    PAD_STRAIGHT_COS,  # noqa: F401
    PAD_WALL_DROP_M,  # noqa: F401
    PARAPET_DROP_M,  # noqa: F401
    PARAPET_DROP_MOUNTAIN_M,  # noqa: F401
    PARAPET_H,  # noqa: F401
    PARAPET_OFF_M,  # noqa: F401
    PARAPET_PROBE_M,  # noqa: F401
    PARAPET_PROBE_MOUNTAIN_M,  # noqa: F401
    PARAPET_ROAD_GAP_M,  # noqa: F401
    PARAPET_W,  # noqa: F401
    PARKING_CLASSES,  # noqa: F401
    PARKING_M,  # noqa: F401
    PASSAGE_M,  # noqa: F401
    PAVED,  # noqa: F401
    PIER_CHANNEL_CLEAR,  # noqa: F401
    PIER_IN_WATER_M,  # noqa: F401
    PIER_MIN_CLEAR,  # noqa: F401
    PIER_ROAD_M,  # noqa: F401
    PIER_SPAN_M,  # noqa: F401
    PIER_SPAN_MAX,  # noqa: F401
    PIER_SPAN_PER_M,  # noqa: F401
    PIN_M,  # noqa: F401
    PIN_ROAD_M,  # noqa: F401
    PORTAL_CLASSES,  # noqa: F401
    PORTAL_CLEAR_M,  # noqa: F401
    PORTAL_EXTEND_MAX_M,  # noqa: F401
    PORTAL_GROUND_M,  # noqa: F401
    PORTAL_JOIN_M,  # noqa: F401
    PORTAL_LOOK_M,  # noqa: F401
    PORTAL_SHALLOW_COVER_M,  # noqa: F401
    PORTAL_UP_MIN,  # noqa: F401
    RAMP_JOIN_DEG,  # noqa: F401
    REGIONAL_CLASSES,  # noqa: F401
    REGIONAL_MIN_M,  # noqa: F401
    REGIONAL_POOL_MIN_M,  # noqa: F401
    REGIONAL_SHARE,  # noqa: F401
    RING_EXTRA_LANE_M,  # noqa: F401
    RING_JUNCTIONS,  # noqa: F401
    RING_LANE_M,  # noqa: F401
    ROAD_CAP_GAP_M,  # noqa: F401
    ROAD_CAP_JOIN_M,  # noqa: F401
    ROAD_CAP_OVER_M,  # noqa: F401
    ROAD_GEO,  # noqa: F401
    ROCKISH,  # noqa: F401
    SECTION_MIN_M,  # noqa: F401
    SE_GRAVEL_CLASSES,  # noqa: F401
    SE_NO_STRIPS,  # noqa: F401
    SE_TOWN_MAX_KMH,  # noqa: F401
    SHIFT_M,  # noqa: F401
    SHIFT_MIN_M,  # noqa: F401
    SHIFT_M_PER_M,  # noqa: F401
    SHIFT_SHARE,  # noqa: F401
    SHOULDER_M,  # noqa: F401
    SHOULDER_WIDE_LANES,  # noqa: F401
    SHOULDER_WIDE_M,  # noqa: F401
    SIDEWALK_CLASSES,  # noqa: F401
    SIDEWALK_TAG,  # noqa: F401
    SIMPLIFY_M,  # noqa: F401
    SIMPLIFY_Z_M,  # noqa: F401
    SKEW_MARGIN_M,  # noqa: F401
    SKEW_MAX_M,  # noqa: F401
    SKEW_MIN_SIN,  # noqa: F401
    SKEW_ON,  # noqa: F401
    SKEW_SLOPE_M,  # noqa: F401
    SOUND_GARAGE,  # noqa: F401
    SOUND_POWER_PLANT,  # noqa: F401
    SOUND_REFINERY,  # noqa: F401
    STACK_STEP_M,  # noqa: F401
    STRIP_JOIN_M,  # noqa: F401
    STYLE,  # noqa: F401
    Spec,  # noqa: F401
    TAPER_M,  # noqa: F401
    TAPER_MAX_TURN,  # noqa: F401
    TAPER_MIN_DW,  # noqa: F401
    TAPER_M_PER_M,  # noqa: F401
    TAPER_SHARE,  # noqa: F401
    TIE_EASE_GRADE,  # noqa: F401
    TIE_EASE_MIN_M,  # noqa: F401
    TIE_JOIN_M,  # noqa: F401
    TIE_MEASURED_M,  # noqa: F401
    TILE_BUILDING_LAYERS,  # noqa: F401
    TUNNEL,  # noqa: F401
    TUNNEL_COVER,  # noqa: F401
    TUNNEL_GAP_M,  # noqa: F401
    TUNNEL_GRADE,  # noqa: F401
    TUNNEL_HOLE_MARGIN,  # noqa: F401
    TUNNEL_LOW_H,  # noqa: F401
    TUNNEL_OPEN_COVER,  # noqa: F401
    TUNNEL_PORTAL_CLEAR_M,  # noqa: F401
    TUNNEL_PORTAL_END_M,  # noqa: F401
    TUNNEL_PORTAL_GROW_M,  # noqa: F401
    TUNNEL_PORTAL_MIN_M,  # noqa: F401
    TUNNEL_PORTAL_REACH_M,  # noqa: F401
    TUNNEL_PORTAL_SHRINK_M,  # noqa: F401
    TUNNEL_ROAD_CLEAR_M,  # noqa: F401
    TUNNEL_ROAD_MIN_SIN,  # noqa: F401
    TUNNEL_ROAD_REACH_M,  # noqa: F401
    TUNNEL_SOIL_M,  # noqa: F401
    TUNNEL_STEP,  # noqa: F401
    TUNNEL_WET_COVER_M,  # noqa: F401
    TUNNEL_WET_RUN_M,  # noqa: F401
    TUNNEL_WET_SHARE,  # noqa: F401
    TWIN_DZ_M,  # noqa: F401
    TWIN_SHARE,  # noqa: F401
    TWIN_STEP_M,  # noqa: F401
    UNBUILT_SHARE,  # noqa: F401
    UNDERPASS_MAX_M,  # noqa: F401
    UNPAVED,  # noqa: F401
    URBAN_ARTERIAL,  # noqa: F401
    URBAN_BESIDE_M,  # noqa: F401
    URBAN_BOTH_SHARE,  # noqa: F401
    URBAN_CLASS,  # noqa: F401
    URBAN_DUAL_MAX_W,  # noqa: F401
    URBAN_MEDIAN_GAP_M,  # noqa: F401
    URBAN_MEDIAN_M,  # noqa: F401
    URBAN_SAMPLE_M,  # noqa: F401
    URBAN_SHARE,  # noqa: F401
    WALK_DRAPE_M,  # noqa: F401
    WALK_DRAPE_TOL_M,  # noqa: F401
    WALK_GROUND_M,  # noqa: F401
    WALK_ON_ROAD_M,  # noqa: F401
    WALK_STEP_M,  # noqa: F401
    WALK_TRIM_KEEP_M,  # noqa: F401
    WALK_TRIM_STEP_M,  # noqa: F401
    WALL_COPING_MIN_M,  # noqa: F401
    WALL_DECK_MARGIN_M,  # noqa: F401
    WALL_DECK_OVER_M,  # noqa: F401
    WALL_DIAG_M,  # noqa: F401
    WALL_EVEN_N,  # noqa: F401
    WALL_FOOT_M,  # noqa: F401
    WALL_FULL,  # noqa: F401
    WALL_GAP_N,  # noqa: F401
    WALL_KEEP_M,  # noqa: F401
    WALL_MIN_N,  # noqa: F401
    WALL_OTHER_GAP_M,  # noqa: F401
    WALL_PROBE_M,  # noqa: F401
    WALL_SLAB_DZ,  # noqa: F401
    WALL_STEEP,  # noqa: F401
    WALL_STEEP_SPAN,  # noqa: F401
    WALL_STEP,  # noqa: F401
    WALL_STEP_UP_M,  # noqa: F401
    WALL_TAPER_LOW_M,  # noqa: F401
    WALL_TAPER_M,  # noqa: F401
    WALL_THICK_MAX_M,  # noqa: F401
    WALL_THICK_MIN_M,  # noqa: F401
    WALL_TOP_EDGE_M,  # noqa: F401
    WALL_TOP_M,  # noqa: F401
    WATER_CAUSEWAY_CLEAR,  # noqa: F401
    WATER_CAUSEWAY_M,  # noqa: F401
    WATER_CLEAR_MAX,  # noqa: F401
    WATER_CLEAR_MIN,  # noqa: F401
    WATER_CLEAR_SHARE,  # noqa: F401
    WATER_STEP_M,  # noqa: F401
    _FACTS_LOCK,  # noqa: F401
    _FACTS_MEM,  # noqa: F401
    _beside,  # noqa: F401
    _facts_baked,  # noqa: F401
    _facts_key,  # noqa: F401
    _facts_load,  # noqa: F401
    _facts_store,  # noqa: F401
    _home_cell,  # noqa: F401
    _home_cells,  # noqa: F401
    _osm,  # noqa: F401
    _ranges_mem_key,  # noqa: F401
    _run_facts_load,  # noqa: F401
    _run_facts_path,  # noqa: F401
    _run_facts_store,  # noqa: F401
    clear_facts_memory,  # noqa: F401
    home_copy,  # noqa: F401
    surface_material_kind,  # noqa: F401
)
from roadspecs import (  # (re-exported: callers use roads.<name>)
    GAP_DECK_ON,  # noqa: F401
    REGIONAL_ON,  # noqa: F401
    URBAN_BESIDE_SHARE,  # noqa: F401
    _Copies,  # noqa: F401
    _KIND_NAMES,  # noqa: F401
    _LANE_FLOW_DFLT,  # noqa: F401
    _bridge_ends,  # noqa: F401
    _bridge_reach,  # noqa: F401
    _bridge_specs,  # noqa: F401
    _cell_memo,  # noqa: F401
    _cell_ranges,  # noqa: F401
    _centre_gap,  # noqa: F401
    _clear_end,  # noqa: F401
    _clear_portals,  # noqa: F401
    _crosses,  # noqa: F401
    _deck_gaps,  # noqa: F401
    _dir_kmh,  # noqa: F401
    _dual_side,  # noqa: F401
    _dual_width,  # noqa: F401
    _feature_boxes,  # noqa: F401
    _feature_length_m,  # noqa: F401
    _first_ring_ends,  # noqa: F401
    _id_rows,  # noqa: F401
    _lane_count,  # noqa: F401
    _lane_flow_home,  # noqa: F401
    _lane_flow_ids,  # noqa: F401
    _lane_flow_local,  # noqa: F401
    _lane_flow_rows,  # noqa: F401
    _lane_overrides,  # noqa: F401
    _lengths,  # noqa: F401
    _level_ranges,  # noqa: F401
    _lidar_answers,  # noqa: F401
    _lidar_cell,  # noqa: F401
    _lidar_gaps,  # noqa: F401
    _link_width,  # noqa: F401
    _ll_metres,  # noqa: F401
    _marking_specs,  # noqa: F401
    _measured_widths,  # noqa: F401
    _mountain_roads,  # noqa: F401
    _near_bridge_ends,  # noqa: F401
    _osm_sidewalks,  # noqa: F401
    _parts_by_level,  # noqa: F401
    _parts_by_level_new,  # noqa: F401
    _portal_roads,  # noqa: F401
    _ramp_junction,  # noqa: F401
    _raw_ground_parts,  # noqa: F401
    _regional_gravel,  # noqa: F401
    _regional_rows,  # noqa: F401
    _ring_fraction,  # noqa: F401
    _rings,  # noqa: F401
    _road_specs,  # noqa: F401
    _road_specs_kept,  # noqa: F401
    _rural_connectors,  # noqa: F401
    _seg_hits,  # noqa: F401
    _skew_decks,  # noqa: F401
    _spec_json,  # noqa: F401
    _spec_mask,  # noqa: F401
    _specs_from,  # noqa: F401
    _structure_width,  # noqa: F401
    _surface_value,  # noqa: F401
    _swedish_specs,  # noqa: F401
    _tag_ranges_all,  # noqa: F401
    _tag_ranges_list,  # noqa: F401
    _tag_rule_rows,  # noqa: F401
    _tapers,  # noqa: F401
    _tunnel_reach,  # noqa: F401
    _unbuilt_all,  # noqa: F401
    _urban_samples,  # noqa: F401
    bridge_levels,  # noqa: F401
    bridge_parts,  # noqa: F401
    carriageway,  # noqa: F401
    deck_tapers,  # noqa: F401
    dual_cap,  # noqa: F401
    dual_carriageways,  # noqa: F401
    edge_margin,  # noqa: F401
    ground_ranges,  # noqa: F401
    hairpin_apex_rows,  # noqa: F401
    hairpin_apexes,  # noqa: F401
    hairpin_tapers,  # noqa: F401
    lane_flow,  # noqa: F401
    lane_tag_parts,  # noqa: F401
    level_ranges,  # noqa: F401
    lidar_gap_candidates,  # noqa: F401
    motorway_shoulder,  # noqa: F401
    raw_level_ranges,  # noqa: F401
    regional_unpaved,  # noqa: F401
    ring_width,  # noqa: F401
    rule_level_ranges,  # noqa: F401
    rule_level_ranges_near,  # noqa: F401
    rural_motorway,  # noqa: F401
    sub_polyline,  # noqa: F401
    surface_kind,  # noqa: F401
    surface_token,  # noqa: F401
    switchbacks,  # noqa: F401
    tag_level_ranges,  # noqa: F401
    tunnel_parts,  # noqa: F401
    tunnel_specs,  # noqa: F401
    unbuilt,  # noqa: F401
    unbuilt_mask,  # noqa: F401
)
from roaddecks import (  # (re-exported: callers use roads.<name>)
    END_JOIN_ON,  # noqa: F401
    _EndJoin,  # noqa: F401
    _chain_graph,  # noqa: F401
    _chord,  # noqa: F401
    _curve_crossings,  # noqa: F401
    _curve_deck,  # noqa: F401
    _curve_grade,  # noqa: F401
    _curve_graph,  # noqa: F401
    _curve_try,  # noqa: F401
    _deck_beside,  # noqa: F401
    _deck_crossings,  # noqa: F401
    _deck_stacks,  # noqa: F401
    _dedupe,  # noqa: F401
    _dijkstra,  # noqa: F401
    _ease,  # noqa: F401
    _end_dists,  # noqa: F401
    _end_joins,  # noqa: F401
    _end_ons,  # noqa: F401
    _end_step,  # noqa: F401
    _ground_envelope,  # noqa: F401
    _ground_pins,  # noqa: F401
    _joined_lines,  # noqa: F401
    _lift_smooth,  # noqa: F401
    _merge_overlaps,  # noqa: F401
    _no_sag,  # noqa: F401
    _path_curve,  # noqa: F401
    _pin,  # noqa: F401
    _simple_path,  # noqa: F401
    _smooth_pinned,  # noqa: F401
    _tents,  # noqa: F401
    _tie_beside,  # noqa: F401
    _upper_hull,  # noqa: F401
    bridge_profile_mode,  # noqa: F401
    curve_profile,  # noqa: F401
    deck_curve,  # noqa: F401
    deck_heights,  # noqa: F401
    deck_profile,  # noqa: F401
    deck_profile_once,  # noqa: F401
    water_clear,  # noqa: F401
)
from roadgeom import (  # (re-exported: callers use roads.<name>)
    _BLOCKERS,  # noqa: F401
    _blocker_segments,  # noqa: F401
    _runs_outside,  # noqa: F401
    _seg_dist,  # noqa: F401
    centre_colour,  # noqa: F401
    cover_roads,  # noqa: F401
    portal_cover,  # noqa: F401
    tunnel_profile,  # noqa: F401
)
from roadwalks import (  # (re-exported: callers use roads.<name>)
    _arterial,  # noqa: F401
    _end_dir,  # noqa: F401
    _left_normal,  # noqa: F401
    _motesfri_section,  # noqa: F401
    _seg_dist_min,  # noqa: F401
    edge_colour,  # noqa: F401
    edge_inset,  # noqa: F401
    has_median,  # noqa: F401
    limited_access,  # noqa: F401
    median_side,  # noqa: F401
    median_width,  # noqa: F401
    walk_pieces,  # noqa: F401
    walk_sides,  # noqa: F401
)
from roadtiles import (  # (re-exported: callers use roads.<name>)
    box_gap,  # noqa: F401
    cells_of_tile,  # noqa: F401
    tile_needs,  # noqa: F401
    tile_side,  # noqa: F401
)
from roadbridges import (  # (re-exported: callers use roads.<name>)
    BridgeProfiles,  # noqa: F401
    ChainStore,  # noqa: F401
    ChainView,  # noqa: F401
    TunnelProfiles,  # noqa: F401
    _Mapper,  # noqa: F401
    _clearance_deficits,  # noqa: F401
    _crossings,  # noqa: F401
    _crossings_by,  # noqa: F401
    _rail_drawn,  # noqa: F401
    _shallow_full_low,  # noqa: F401
    deck_depth,  # noqa: F401
    deck_road_clear,  # noqa: F401
    deck_split,  # noqa: F401
    geom_span,  # noqa: F401
    pier_sites,  # noqa: F401
    tunnel_sections,  # noqa: F401
    wet_run_inside,  # noqa: F401
)
from roadstructures import (  # (re-exported: callers use roads.<name>)
    _DECK_PACK,  # noqa: F401
    _Segs,  # noqa: F401
    _abutments,  # noqa: F401
    _clear_depth,  # noqa: F401
    _cut_behind,  # noqa: F401
    _deck_chunks,  # noqa: F401
    _deck_frame,  # noqa: F401
    _deck_look,  # noqa: F401
    _deck_pack,  # noqa: F401
    _end_nodes,  # noqa: F401
    _fork,  # noqa: F401
    _hash01,  # noqa: F401
    _line_near,  # noqa: F401
    _off_line,  # noqa: F401
    _on_carriageways,  # noqa: F401
    _pad_height_groups,  # noqa: F401
    _parapets,  # noqa: F401
    _road_ends,  # noqa: F401
    _straight_on,  # noqa: F401
    _taper,  # noqa: F401
    _travel,  # noqa: F401
    _under_decks,  # noqa: F401
    _wall_runs,  # noqa: F401
    arch_raise,  # noqa: F401
    block_arch_raise,  # noqa: F401
    block_tunnel_holes,  # noqa: F401
    deck_pads,  # noqa: F401
    hairpin_extra,  # noqa: F401
    pad_inside,  # noqa: F401
    shifted_parts,  # noqa: F401
    spare_carriageways,  # noqa: F401
    tunnel_floor_at,  # noqa: F401
    tunnel_hole_mask,  # noqa: F401
    twin_tunnels,  # noqa: F401
)


# carriageways painted into the drawn ground instead of meshes (surface.photo_roads; the user, 2026-10-22: "I dont love
# that we are applying a floating road texture over the normal terrain"; it also ended the white diamonds on the
# Stelvio's hairpins, GitHub #135). BNGE_ROADS_IN_GROUND=0: the carriageway meshes as before
ROADS_IN_GROUND = os.environ.get("BNGE_ROADS_IN_GROUND", "1") != "0"


def build_tile(frame, tx, ty, cells, elev, urban_fn=None, walks=None, bridges=None, widths=None, props=None,
               lines=None, wires=None, town=None, park=None, left_hand=False, structures=(), mph=True, tunnels=None,
               buried=0.0, lamp_region=None, country=None,
               walls=None, deck_context=None, ground=None, terrain_paint=True, cover_out=None, walk_cuts=(),
               signals=None, marks=None, roads_out=None, skin_margin=0.0, walks_raw=None, nodes=None,
               clearance_out=None, development=None, pole_profile=None, signs=None,
               sign_obstacles=None, roadside_jobs=None, towns=None):
    """The chunks of one tile. cells: the mapgeom.Features of the "roads" layer of every cell touching
    the tile; elev(lat, lon) -> metres (the DEM); urban_fn: as road_specs. Features in several cells are
    taken once (by id). The carriageways, then each road's sidewalks / gutters / markings (extras), cut
    where another road's carriageway crosses them. walks: a list that gets the sidewalk runs (dicts
    pts / side / caps, clipped exactly to the tile, caps: whether each end is a real end) for the
    sidewalk mesh (sidewalks.py) and storm drains among the chunks; None: sidewalks as decals.
    bridges: a BridgeProfiles (deck heights of whole chains, shared by every tile); None: from the
    chains among these cells. ground(x, y) -> z: the terrain block's drawn ground in the frame, the sidewalks' heights
    (drape_walks); None: elev's. walk_cuts: [(centreline (k, 2), half width)] where no sidewalk runs (a gas station's
    made-up driveway: gasstations.tile_sites cuts). roads_out: a list that gets the ground carriageways near the tile
    ((fid, spec, centreline (k, 3), lon): the median barriers', medians.tile_rows). skin_margin (m, a geotile's content:
    blockworker.roads_geo): the draped surfaces' pieces (sidewalk runs and corners, markings, crosswalks, corner fills)
    taken this far past the tile's box, its own, the skin clipping them exactly at the tile's quad in the anchor's
    world (a seam-straddling crosswalk drawn by both tiles, each its half); walks_raw: a list that gets the sidewalk
    pieces before they are put on the ground (the skin drapes them on its anchor's). clearance_out: a list that gets
    the shared furniture.RoadClearance for this tile's tapered ground roads and generated bridge decks (not tunnel
    vaults), also used by powerlines.tile_lines. development: a local context (development.py) sampled only by
    furniture.place for the street trees, hydrants, benches and bins on these sidewalks. towns(roads, box,
    left_hand, rings) -> prop rows: the town entry and exit signs among props (townentry.Towns.maker; BNE_TOWNSIGNS;
    rings: the roundabout rings' ids, which have none). Pure given its inputs (sorted by feature id)."""
    own_box = tile_bbox(tx, ty)                 # (a geotile's knows its own points: box_owns)
    x0, y0, x1, y1 = own_box
    import furniture
    traffic_context = clearance_out is not None or (props is not None and walks is not None)
    context_m = max(NEAR_M, furniture.POLE_SPACING + 32.0) if traffic_context else NEAR_M
    box = (x0 - OVERLAP, y0 - OVERLAP, x1 + OVERLAP, y1 + OVERLAP)
    near = (x0 - context_m, y0 - context_m, x1 + context_m, y1 + context_m)
    seen, bseen = {}, {}
    for feats in cells:
        if feats is None or len(feats) == 0:
            continue
        specs = road_specs(feats, urban_fn, widths, left_hand)
        bspecs = bridge_specs(feats)
        for i in range(len(feats)):
            if bspecs[i] is not None and str(feats.ids[i]) not in bseen:
                bseen[str(feats.ids[i])] = (feats, i, bspecs[i])
            if specs[i] is None or str(feats.ids[i]) in seen:
                continue
            seen[str(feats.ids[i])] = (feats, i, specs[i])
    rings = []                                  # (fid, spec, lon/lat ring) near the tile (xy at h = 0)
    for fid in sorted(seen):
        feats, i, spec = seen[fid]
        fb = feats.boxes(frame)[i]          # (the whole feature far from the tile: none of its parts is near)
        if fb[2] < near[0] or fb[0] > near[2] or fb[3] < near[1] or fb[1] > near[3]:
            continue
        for ring in ground_parts(feats, i):
            e, n = mapgeom.enu_xy(feats, frame, ring)
            if (e.max() < near[0] or e.min() > near[2] or n.max() < near[1] or n.min() > near[3]):
                continue
            rings.append((fid, spec, ring))
    roads = []                                  # (fid, spec, centreline (k, 3), lon) near the tile
    if rings:                                   # one elevation call for every ring (the grader's cost
        lon = np.concatenate([r[:, 0] for _, _, r in rings])     # is per call: 2026-10-11 profile,
        lat = np.concatenate([r[:, 1] for _, _, r in rings])     # 1,240 calls took 7 of a tile's 8 s)
        h = np.asarray(elev(lat, lon), np.float64)
        # east / north of the point at h = 0, as the ground's (heightmap.surface: its samples' lat / lon at plane
        # z 0) and every outline's (mapgeom.enu_xy); up of the point at its height. At its height the point lands
        # h x distance / earth radius farther out: US-163 in Utah at 1650 m, 13 km out, the lines lay 2.8 m east and
        # 2.0 m north of the asphalt and grip painted into the ground (the user's screenshot, 2026-10-29)
        e, n, _ = frame.geodetic_to_enu(lat, lon, 0.0)
        u = frame.geodetic_to_enu(lat, lon, h)[2]
        k = 0
        for fid, spec, ring in rings:
            m = len(ring)
            roads.append((fid, spec, np.stack([e[k:k + m], n[k:k + m], u[k:k + m]], axis=1),
                          float(np.mean(ring[:, 0]))))
            k += m
    # Geometric roundabout rings (islands.ring_directions): Overture often leaves junction=roundabout and oneway
    # off the ring ways (Chavagnes-en-Paillers 2026-10-28). No parking / inferred sidewalks on the ring, one-way so
    # centre lines are not painted through the circle. Short approaches that meet the ring lose sidewalks too:
    # their stubs against the circle were rectangular concrete blocks.
    ring_dir = {}
    if roads:
        import islands
        ring_dir = islands.ring_directions(roads, left_hand)
        if ring_dir:
            apps = islands.ring_approaches(roads, ring_dir)
            short_apps = {fid for fid, sp, pts, _ in roads if fid in apps
                          and float(np.sum(np.hypot(*np.diff(pts[:, :2], axis=0).T))) < islands.APPROACH_NO_WALK_M}
            roads = [(fid,
                      (sp._replace(urban=False, median=None, oneway=True, dir=ring_dir[fid]) if fid in ring_dir
                       else sp._replace(urban=False, median=None) if fid in short_apps
                       else sp),
                      pts, lon) for fid, sp, pts, lon in roads]
    chunks = []
    if roads_out is not None:
        roads_out.extend(roads)
    if cover_out is not None:
        # the ground carriageways for the decks' parapet test (roadmesh.deck ground=): drawn or painted into the ground,
        # a deck at grade across one has no parapet over its lanes (GitHub #96). With the roads in the ground there
        # were none, and a small bridge's wall stood across the road at its end (GitHub #258)
        for fid, spec, pts, _ in roads:
            for piece in clip_polyline(pts, x0 - COVER_MARGIN_M, y0 - COVER_MARGIN_M, x1 + COVER_MARGIN_M,
                                       y1 + COVER_MARGIN_M):
                for c in split(densify(simplify(piece))):
                    if len(c) >= 2:
                        cover_out.append(_chunk(look_of(spec), spec.width, c))
    # no smooth ribbon over rough ground — but only where the ground layer will paint it (--surface on); with the
    # surface pass off there is no ROCK/GRASS mask, and the road would otherwise vanish entirely
    rough = (lambda sp: terrain_paint and terrain_surface(sp) is not None)     # noqa: E731
    strip_ends = _strip_open_ends(roads, rings, seen, near, rough) if roadbase.DIRT_STRIPS else {}
    for k_road, (fid, spec, pts, _) in enumerate(roads):
        # the carriageway painted into the ground (surface.photo_roads); a dirt or gravel road's strip over it
        # (DIRT_LOOK)
        if ROADS_IN_GROUND and not (roadbase.DIRT_STRIPS and spec.kind in DIRT_LOOK):
            continue
        if rough(spec):
            continue
        look = look_of(spec, fid)
        if look.get("aspect"):                  # (a strip: whole, its distance along the road, its open ends)
            chunks.extend(_surface_pieces(look, spec.width, pts, box, strip_ends.get(k_road, (True, True))))
            continue
        lanes = [(0.0, spec.width)]
        if look.get("strip"):                   # parallel strips across the width (see LOOK)
            n = max(1, int(round(spec.width / look["strip"])))
            sw = spec.width / n
            lanes = [((k + 0.5) * sw - spec.width / 2, sw) for k in range(n)]
        for off, lw in lanes:
            line = pts if off == 0.0 else offset_polyline(pts, off)
            for piece in clip_polyline(line, *box):
                for c in split(densify(simplify(piece))):
                    if len(c) >= 2:
                        chunks.append(_chunk(look, lw, c))
    decks_near = []                             # (retaining_walls: none under a bridge)
    traffic_decks = []                          # genuine bridge geometry only, never the tunnel vaults below
    deck_walk = {}                              # the bridges that carry sidewalks (deck_walks)
    for fid in bseen:
        bf, bi, _ = bseen[fid]
        dwk = deck_walks(bf, bi, road_specs(bf, urban_fn, widths, left_hand), left_hand)
        if dwk:
            deck_walk[fid] = dwk
    deck_shift = deck_shift_fn(frame, cells, urban_fn, widths, left_hand)   # (a deck at a merge / split)
    road_copies = copies_of(cells)              # (a joint's taper / shift from its own cell: joint_desc)
    deck_width = deck_width_fn(frame, cells)    # (a deck tapering into another deck)
    def bridge_context(source_cells):
        # Eligibility must use the same complete source cells as the incident profiles.
        expanded_walk = {}
        for bf in source_cells:
            if bf is None or len(bf) == 0:
                continue
            specs = road_specs(bf, urban_fn, widths, left_hand)
            for bi, bs in enumerate(bridge_specs(bf)):
                fid = str(bf.ids[bi])
                if bs is not None and fid not in expanded_walk:
                    dwk = deck_walks(bf, bi, specs, left_hand)
                    if dwk:
                        expanded_walk[fid] = dwk
        return (expanded_walk, deck_shift_fn(frame, source_cells, urban_fn, widths, left_hand),
                deck_width_fn(frame, source_cells))

    for ch in _bridge_chunks(frame, bseen, elev, (x0, y0, x1, y1), bridges, cells, deck_walk, deck_shift, deck_width,
                             bridge_context, context_m=context_m if traffic_context else None):
        if ch.get("deck"):
            decks_near.append(ch)
            traffic_decks.append(_deck_clearance(ch))
        if not ch.get("ctx"):
            chunks.append(ch)
        elif deck_context is not None:          # the decks around the tile (roadmesh.deck's parapet test)
            deck_context.append(ch)
    if tunnels is not None:                     # TunnelProfiles: road surfaces and lines through tunnels
        tch = _tunnel_chunks(tunnels, cells, (x0, y0, x1, y1), terrain_paint)
        chunks.extend(tch)
        import roadmesh
        # (retaining walls: none in a tube either, its vault over them; between twin tunnels' carriageways one stood in
        # their shared mouth, GitHub #84)
        for c in tch:
            if c.get("tunnel") and not c.get("no_tube"):
                n = np.array(c["n"], np.float64)
                n[:, 2] += float(c.get("tunnel_h") or roadmesh.TUNNEL_H)
                decks_near.append({"n": n, "w": float(c["w"]) + 2 * roadmesh.TUNNEL_SHOULDER})
    exact = (x0, y0, x1, y1)
    sm = float(skin_margin or 0.0)
    wexact = (x0 - sm, y0 - sm, x1 + sm, y1 + sm) if sm else exact    # (the skin's pieces: skin_margin)

    def skin_own(px, py):                       # a draped piece's owner test (skin_margin: the margin box)
        if sm:
            return wexact[0] <= px < wexact[2] and wexact[1] <= py < wexact[3]
        return box_owns(own_box, px, py)
    if walls is not None:                       # retaining walls where a road is cut into the ground (roadmesh.tunnel)
        tile_walls = retaining_walls(frame, roads, elev, exact, decks=decks_near)
        if props is None:                       # (no props here: stone)
            for w in tile_walls:
                w["rock"] = False
        walls.extend(w for w in tile_walls if not w["rock"])
        if props is not None:                   # a rural cut: rock-face pieces instead (cliff_rows)
            props.extend(cliff_rows([w for w in tile_walls if w["rock"]], roads))
        chunks = _off_wall_tops(chunks, [w for w in tile_walls if not w["rock"]])
    corners, jcuts, crosswalks, jramps, jlines, jinternal = ({}, {}, [], {}, {}, set()) if walks is None else \
        junctions.find(roads, has_sidewalks, None)
    teeth = []                                  # Sweden's give-way lines (markings.give_way_lines), M16's squares
    zebras = []                                 # ... and its crossings, which a road's lines stop at (secrossings.place)
    crosswalk_look = junctions.CROSSWALK
    priority = None if nodes is None else nodes["priority"]
    trams = []                                  # ... and what no raised sidewalk or median runs over (rails.tram_lines)
    rail_cuts = []                              # ... nor over a railway crossing the road at grade (GitHub #579)
    if walks is not None:
        import railways
        rail_cuts = railways.walk_cut_lines(frame, cells, own_box)
    if str(country or "").upper() == "SE":
        import furniture
        import markings as _gw
        import rails
        trams = rails.tram_lines(frame, tx, ty, cells)
        stops = []
        if nodes is not None and walks is not None:
            import secrossings as _cx
            crosswalk_look = _cx.M15_LOOK
            crosswalks, teeth, jramps = _cx.place(roads, nodes)
            zebras = [(np.array([cw["a"], cw["b"]])[:, :2], cw["w"] / 2) for cw in crosswalks]
            stops, jlines = _gw.stop_lines(furniture.give_way_arms(roads, signals, country, priority, stops=True),
                                           crosswalks, nodes["stop_lines"], left_hand)
        m14, gone = _gw.give_way_lines(furniture.give_way_arms(roads, signals, country, priority), crosswalks, left_hand)
        teeth = teeth + m14
        crosswalks = [cw for j, cw in enumerate(crosswalks) if j not in gone] + stops
    # (no curb extensions: they were on every town street with parking, after Fells Point's, and most streets have none:
    # the corners stuck out into the road in Tustin's suburbs, GitHub #208)
    # small road loops (roundabouts, turnaround loops): no sidewalks facing their inside, a curbed island instead
    import islands
    isl = islands.find(roads, structures=structures) if walks is not None else []
    inner = islands.inner_sides(isl)
    if walks is not None:
        for f in isl:
            if f["ring"] is None:
                continue
            # (from its south-west-most point: each tile finds the loop from its own roads, starting anywhere)
            ring = np.roll(f["ring"], -int(np.lexsort((f["ring"][:, 1], f["ring"][:, 0]))[0]), axis=0)
            # (closed: offset with a vertex either side of its start, so its ends meet mitred, not 5 cm apart)
            band = offset_polyline(np.vstack([ring[-1:], ring, ring[:2]]), islands.ISLAND_BAND_M / 2)[1:-1]
            # tight simplify + denser nodes: the default 0.25 m / 12 m spacing faceted the curb into a hexagon (PR
            # #468). The whole ring simplified before it is clipped, so neighbouring tiles' pieces (each to its box and
            # the skin margin, the skin cutting them exactly at the tile's quad) share its vertices and meet
            band = simplify(band, tol=0.08)
            for piece in clip_polyline(band, *wexact):
                run = densify(piece, spacing=3.0)
                if len(run) >= 2:
                    walks.append({"pts": run, "side": 1, "caps": (False, False), "ramps": [],
                                  "w": islands.ISLAND_BAND_M, "fid": "island"})
    cornered = set()                            # road pairs whose sidewalks meet at a curb return
    if walks is not None:
        for (fid, side), cl in jcuts.items():
            for X, u, t in cl:
                for f2, s2, p2, _ in roads:
                    if f2 != fid and np.hypot(*(p2[:, :2] - X).T).min() < 1.0:
                        cornered.add((fid, f2))
        for c in corners:                       # each corner / crosswalk made by the tile it is in
            p = c["arc"][len(c["arc"]) // 2]
            if isl and islands.in_island(isl, p[:2]):
                continue                        # a curb return inside a loop: its island has its own curb
            # a curb return on the circulatory roadway or at a ring entry (approach sidewalks meeting there): the
            # ring has no sidewalk, so the corner was a rectangular block on the asphalt (PR #468)
            if any(float(np.hypot(*(p[:2] - f["poly"].mean(0))))
                   < float(np.max(np.hypot(*(f["poly"] - f["poly"].mean(0)).T))) + islands.ENTRY_CLEAR_M
                   for f in isl if f.get("round")):
                continue
            if skin_own(p[0], p[1]):
                walks.append({"corner": c})
                fill_look = _corner_look(p, roads)
                for fill in ([] if ROADS_IN_GROUND else ([c["fill"]] if c["fill"] is not None else []) + c.get("fills", [])):
                    chunks.append({"m": fill_look["m"], "w": 0.0, "tl": fill_look["tl"], "p": fill_look["p"],
                                   "poly": [[round(float(v), 3) for v in q] for q in fill], "n": []})
        import markings as _mk
        ladder = _mk.style(country, float(np.mean([r[3] for r in roads])) if roads else None).ladder
        for cw in crosswalks:                   # crosswalks and stop lines
            mid = 0.5 * (cw["a"] + cw["b"])
            if skin_own(mid[0], mid[1]) and cw.get("paint", True):
                look = junctions.STOP_LINE if cw.get("stop") else crosswalk_look
                ab = np.array([cw["a"], cw["b"]])
                chunks.append(_chunk(look, cw["w"], ab))
                if ladder and not cw.get("stop"):
                    # a ladder crosswalk (MUTCD 3C.07, Baltimore's on the MD iMAP 3-inch aerial): a solid line along
                    # each side of the bars
                    for sd in (1, -1):
                        chunks.append(_chunk(EDGE_LINE, LADDER_LINE_W, offset_polyline(ab, sd * cw["w"] / 2)))
    import markings
    for fid, spec, pts, lon in roads:               # turn arrows (OpenStreetMap turn:lanes; MUTCD 3B.20)
        tags = lane_tags(*seen[fid][:2]) if fid in seen else None
        if not tags or spec.kind != "paved" or fid in jinternal:
            continue
        drv, nl = lane_layout(spec)
        lane_pts = offset_polyline(pts, driving_offset(spec)) if highway_shoulders(spec) else pts
        for centre, t, kind_ in markings.arrows(tags, lane_pts, not spec.oneway, drv, nl, left_hand):
            corners, uv = markings.arrow_quad(centre, t, kind_)
            if (corners[:, 0].max() >= wexact[0] and corners[:, 0].min() < wexact[2]
                    and corners[:, 1].max() >= wexact[1] and corners[:, 1].min() < wexact[3]):
                chunks.append({"m": markings.ARROW_MATERIAL, "w": 0.0, "tl": 4.0, "p": CENTRE["p"], "layer": "mark",
                               "poly": [[round(float(v), 3) for v in q] for q in corners],
                               "uvq": uv.tolist(), "n": []})
    for tri in teeth:
        if skin_own(*tri[:2].mean(0)[:2]):
            chunks.append(mark_poly(tri))
    gore_look = {"m": "line_white", "tl": EDGE_LINE["tl"], "p": CENTRE["p"]}
    for kind_, gw, gl in markings.gores(roads):     # ramp gores: channelizing lines and stripes (MUTCD 3B.08 / 3B.25)
        if kind_ == "fill":
            continue                                # (painted into the ground: surface.photo_roads)
        if kind_ == "stripe":
            mid = 0.5 * (gl[0] + gl[-1])
            if skin_own(mid[0], mid[1]):
                chunks.append(_chunk(gore_look, gw, gl))
            continue
        for piece in clip_polyline(gl, x0, y0, x1, y1):
            for c in split(densify(simplify(piece))):
                if len(c) >= 2:
                    chunks.append(_chunk(gore_look, gw, c))
    runs_near = []                              # sidewalk runs near the tile, before clipping (walks: below)
    # a parking aisle lies in a paved lot (painted asphalt: surface.photo_lots): no gravel edge along it (a dirt strip
    # stood across the asphalt between the stall rows, Canton, 2026-10-28)
    lot_aisles = {fid for fid, (f, i, sp) in seen.items()
                  if sp.kind == "paved" and _rule_value(f.rules.get("subclass"), i) == "parking_aisle"}
    # rounded corners where curbless paved roads meet (roadcorners.py, from PR #363): their fills when the roads are
    # meshes (else painted into the ground: surface.py), the gravel edge decal along each arc and a curved edge line
    # where both roads have them; corner_cuts end those road sides' straight edge lines and decals at the tangents
    import roadcorners
    corner_chunks, corner_cuts = roadcorners.tile(
        roads, box, not ROADS_IN_GROUND, country, left_hand,
        rings={fid for fid, (f, i, sp) in seen.items() if is_ring(f, i)}, no_decal=lot_aisles, exact=wexact)
    chunks.extend(corner_chunks)
    import rampmerges, roadmarkings
    merge_layouts = rampmerges.ground(frame, cells, urban_fn, widths, left_hand)
    ramp_connections = rampmerges.connections(roads)
    going_on = _going_on(roads)
    straight, stems, whole = line_through(roads)
    traffic_lines = []
    import franceroads
    paint_continuations = franceroads.continuations(roads) if franceroads.ON else set()
    paint_joins = rampmerges.through_joins(roads, going_on)
    for fid, spec, pts, lon in roads:
        ws = None                               # (tapered where it meets a road of another width: road_tapers)
        if fid in seen:
            tp = joint_desc(road_copies.get(fid, [seen[fid][:2]]), lambda f: road_tapers(f, urban_fn, widths, left_hand))
            if tp:
                tla, tlo, _ = frame.enu_to_geodetic(pts[:, 0], pts[:, 1], 0.0)
                ws = taper_widths(spec.width, tp, tlo, tla)
        half = spec.width / 2 if ws is None else (ws[:-1] + ws[1:]) / 4  # surface paints each segment at its mean
        traffic_lines.append((half, pts))
        sh = None                               # (its lines shifted at a merge / split: lane_shifts)
        if fid in seen and fid not in jinternal:
            ls = joint_desc(road_copies.get(fid, [seen[fid][:2]]), lambda f: lane_shifts(f, urban_fn, widths, left_hand))
            if ls:
                sla, slo, _ = frame.enu_to_geodetic(pts[:, 0], pts[:, 1], 0.0)
                sh = shift_at(ls, slo, sla)
        ex = extras(spec, pts, lon, left_hand, country, lane_tags(*seen[fid][:2]) if fid in seen else None, ws, sh, defer_dashes=True) \
            if fid not in jinternal or fid in whole else []    # inside a junction: none
        if fid in jinternal:                    # (a through road's piece between two close junctions: its lines only)
            ex = [e for e in ex if e[0].get("kind")]
        if not ex:
            continue
        if fid in seen and str(country or "").upper() == "SE":
            ex = way_lines(ex, *seen[fid][:2], spec, pts, lon, left_hand, country, ws, sh)
        ex = rampmerges.trim(fid, ex, merge_layouts)
        blockers = [(p2[:, :2], s2.width / 2 + CUT_MARGIN) for f2, s2, p2, _ in roads if f2 != fid]
        edge_blockers = [(p2[:, :2], s2.width / 2 + CUT_MARGIN) for f2, s2, p2, _ in roads
                         if f2 != fid and (fid, f2) not in straight] + zebras
        centre_blockers = [(p2[:, :2], s2.width / 2 + CUT_MARGIN) for f2, s2, p2, _ in roads
                           if f2 != fid and (fid, f2) not in straight and (fid, f2) not in stems] + zebras
        paint_blockers = [(p2[:, :2], s2.width / 2 + CUT_MARGIN) for f2, s2, p2, _ in roads
                          if f2 != fid and (fid, f2) not in paint_continuations]
        decal_blockers = [(p2[:, :2], s2.width / 2 + (CUT_MARGIN if (fid, f2) in going_on else
                                                      EDGE_DECAL["w"] + EDGE_DECAL_GAP_M))
                          for f2, s2, p2, _ in roads if f2 != fid and s2.kind in BUILT]
        # ring roads: sidewalks stop ENTRY_CLEAR_M outside the carriageway (no stub into the circle)
        walk_blockers = [(p2[:, :2],
                          (s2.width / 2 + (islands.ENTRY_CLEAR_M if f2 in ring_dir else 0.0)
                           if (fid, f2) in cornered or f2 in ring_dir
                           else sidewalk_stop(spec, fid, s2, f2)))
                         for f2, s2, p2, _ in roads if f2 != fid] + list(walk_cuts) + trams
        crossing_rails = [c for c in rail_cuts if railways.crosses(pts[:, :2], c[0])]
        walk_blockers += crossing_rails
        # a median's band stops at the other carriageways' curbs only (their sidewalk is on their far side: it cut
        # one carriageway's half of Boston Street's median away)
        curb_blockers = [(p2[:, :2], s2.width / 2) for f2, s2, p2, _ in roads if f2 != fid] + trams + crossing_rails
        cut_boxes = {}                          # (_within_any's boxes of this road's cut lists)
        lb_memo = {}                            # (line_blockers by look kind and side, below)
        line_base = [(f2, s2, p2, l2) for f2, s2, p2, l2 in roads
                     if f2 != fid and (fid, f2) not in straight and not rampmerges.related(fid, f2, merge_layouts)
                     and not rampmerges.continuation(fid, spec, f2, s2, paint_joins)]
        for look, w, line in ex:
            french_paint = franceroads.enabled(spec) and look["m"] == franceroads.PAINT_MATERIAL
            paint_cuts = paint_blockers if french_paint else blockers
            if look.get("polygon"):
                # MRE and recall arrows are actual paint polygons. Clip all three
                # coordinates together, before the retained skin is generated.
                centre = np.asarray(line).mean(axis=0)
                if _within_any(centre[:2], paint_cuts, cut_boxes):
                    continue
                poly = mapgeom.clip_ring(np.asarray(line), box)
                if poly is not None:
                    chunks.append({"m": look["m"], "w": 0., "tl": look["tl"], "p": look["p"],
                                   "poly": np.round(poly[:-1], 4).tolist(), "n": []})
                continue
            walk = look["m"] == SIDEWALK["m"]
            side = look.get("side")
            # If an unusually short ramp cannot fit a full taper, its opening
            # still cannot remove the highway's through lines or median edge.
            # (one list per look kind and side of this road: built per line, it was roads^2 x lines in a city tile)
            lb_key = (look.get("kind"), side)
            line_blockers = lb_memo.get(lb_key)
            if line_blockers is None:
                protected = {other for (main, other), ramp_side in ramp_connections.items()
                             if main == fid and (look.get('kind') == 'lane'
                                                 or (look.get('kind') == 'edge' and side != ramp_side))}
                centre_kind = look.get("kind") in ("centre", "lane")
                line_blockers = lb_memo[lb_key] = [
                    (p2[:, :2], s2.width / 2 + CUT_MARGIN) for f2, s2, p2, _ in line_base
                    if f2 not in protected and not (centre_kind and (fid, f2) in stems)] + zebras
            wb = curb_blockers if look.get("median") else walk_blockers
            if (walk or look["m"] == GUTTER["m"]) and (fid, side) in inner:
                continue                        # facing a traffic island: no sidewalk there
            if walk and walks is not None:
                # whole runs near the tile first (joined across tile borders below), clipped to the tile after
                for piece in clip_polyline(line, *near):
                    for cpiece, (j0, j1) in junctions.apply_cuts(piece, jcuts.get((fid, side), [])):
                        for run in [r for r0 in cut_outside(cpiece, wb, 0.5)
                                    for r in _on_level(frame, elev, densify(simplify(r0)))]:
                            if len(run) < 2:
                                continue
                            at_j0 = j0 and np.hypot(*(run[0, :2] - cpiece[0, :2])) < 0.6
                            at_j1 = j1 and np.hypot(*(run[-1, :2] - cpiece[-1, :2])) < 0.6
                            runs_near.append({"pts": run, "side": side, "w": w, "fid": fid, "j": (at_j0, at_j1),
                                              "median": bool(look.get("median")),
                                              "open": (not _on_box(run[0], near), not _on_box(run[-1], near))})
                continue
            if walks is not None and look["m"] == GUTTER["m"]:     # gutters end where their curb turns
                gcuts = jcuts.get((fid, side), [])
                pieces = [cp for p in clip_polyline(line, *box) for cp, _ in junctions.apply_cuts(p, gcuts)]
            elif walks is not None and jlines.get(fid):             # markings stop at the crosswalks
                pieces = [cp for p in clip_polyline(line, *box) for cp, _ in junctions.apply_cuts(p, jlines[fid])]
            else:
                pieces = clip_polyline(line, *box)
            if (look.get("kind") == "edge" or look["m"] == EDGE_DECAL["m"]) and (fid, side) in corner_cuts:
                pieces = [cp for p in pieces for cp, _ in junctions.apply_cuts(p, corner_cuts[(fid, side)])]
            if look["m"] == EDGE_DECAL["m"]:
                if fid in lot_aisles:
                    continue
                # no gravel edge where another carriageway is near: a ramp's gore and a narrow median are paved
                pieces = [r for pc in pieces for r in cut_outside(pc, decal_blockers, 0.5)]
            lines_by = centre_blockers if look.get("kind") in ("centre", "lane") else \
                edge_blockers if look.get("kind") == "edge" else blockers
            if look.get("kind") in ("centre", "lane", "edge"):
                lines_by = line_blockers
            for piece in pieces:
                # (a dash, or a line's end in this tile, is short by nature: dropped as a stub, the 1 m dashes of a
                # Swedish edge line went from every tile with a second road, and a dash from every tile border)
                keep_m = None if walk or not look.get("kind") else \
                    DASH_MIN_M if look.get("dash") else min(MIN_PIECE_M, _length(piece) * 0.9)
                if french_paint:              # (PR #505: French paint keeps its short blocks, cut at its own continuations)
                    keep_m, lines_by = .02, paint_blockers
                for run in cut_outside(piece, wb if walk else lines_by, 0.5 if walk or look.get("dash") else 2.0, keep_m):
                    for c in split(densify(simplify(run))):
                        if len(c) >= 2:
                            chunks.append(_chunk(look, w, c))
    if walks is not None:
        runs_near = join_walk_ends(runs_near, [(p2[:, :2], s2.width / 2) for _, s2, p2, _ in roads]
                                   + list(walk_cuts) + rail_cuts)   # (a cut's gap stays open: a driveway, a railway)
        runs_near = walks_off_roads(runs_near, [(p2[:, :2], s2.width / 2) for _, s2, p2, _ in roads])
        if franceroads.ON and urban_fn is not None:
            runs_near = franceroads.walks_in_town(runs_near, frame, urban_fn,
                                                  {fid for fid, sp, _, _ in roads if getattr(sp, "france", False)})
        drain_runs = []                         # (walks entry, side, real ends, width): drains once on the ground
        for rn in runs_near:
            fid, side, w, run0 = rn["fid"], rn["side"], rn["w"], rn["pts"]
            for run in clip_polyline(run0, *wexact):
                if len(run) < 2:
                    continue
                first = np.hypot(*(run[0, :2] - run0[0, :2])) < 1e-6
                last = np.hypot(*(run[-1, :2] - run0[-1, :2])) < 1e-6
                at_j0 = first and rn["j"][0]
                at_j1 = last and rn["j"][1]
                caps = (first and not at_j0 and not _on_box(run[0], wexact),
                        last and not at_j1 and not _on_box(run[-1], wexact))
                rl = [r for key in rn.get("ramp_keys", [(fid, side)]) for r in jramps.get(key, [])]
                ramps = junctions.ramp_ranges(run, rl)
                walks.append({"pts": run, "side": side, "caps": caps, "ramps": ramps, "w": w, "fid": fid,
                              "median": rn.get("median", False)})
                if not rn.get("median"):                # (a median: a bare curb, no storm drains)
                    drain_runs.append((walks[-1], side, (caps[0] or at_j0, caps[1] or at_j1), w))
        if walks is not None and pole_profile is not None:
            import poleprofiles
            european = poleprofiles.europe_selector(pole_profile, frame)
            for wk in walks:
                points = np.asarray(wk['corner']['arc'] if 'corner' in wk else wk['pts'])
                if len(points):
                    wk['europe'] = bool(european(*points[len(points)//2]))
        if walks_raw is not None:               # (the skin's: put on its anchor's ground there)
            import copy
            walks_raw.extend(copy.deepcopy(walks))
        drape_walks(frame, elev, walks, ground)
        for wk, side, ends, w in drain_runs:
            chunks.extend(drains(wk["pts"], side, ends, w))
    if lines is not None:                           # the centrelines near the tile (buildings' front doors)
        lines.extend((pts[:, :2], spec.width / 2, spec.cls, spec.urban) for _, spec, pts, _ in roads)
    clearance = None
    if clearance_out is not None or (props is not None and walks is not None):
        import furniture
        clearance = furniture.RoadClearance(traffic_lines, traffic_decks, bounds=own_box)
        if clearance_out is not None:
            clearance_out.append(clearance)
    sign_decks = [(np.asarray(ch["n"], np.float64), ch["w"] / 2) for ch in chunks if ch.get("deck")] if props is not None else ()
    detailed_signs = False
    if props is not None and (walks is not None or signs is not None):
        import euguardrails
        import poleprofiles
        detailed_signs = euguardrails.enabled() and poleprofiles.kit_enabled(pole_profile)
    if props is not None and walks is not None:     # street lights and utility poles (furniture.py)
        # (the speed-limit signs are the sign tiles' since 2026-10-28: speedsigns.py)
        if sign_obstacles is not None:
            sign_obstacles.ground = ground
        props.extend(furniture.place(roads, walks, own_box, wires=wires, town=bool(town), left_hand=left_hand,
                                     decks=[(p, h) for p, h, _depth in traffic_decks], clearance=clearance,
                                     signals=signals, buried=buried, region=lamp_region, country=country,
                                     priority=priority, ground=ground, development=development,
                                     pole_profile=pole_profile, pole_frame=frame, sign_obstacles=sign_obstacles,
                                     defer_gore=detailed_signs,
                                     lighting={fid: str(f.attrs["osm_lit"][i]).strip().lower()
                                               for fid, (f, i, _sp) in seen.items() if "osm_lit" in f.attrs}))
    if signs is not None and props is not None:
        import townsigns
        import osmtags
        way_ids = {fid: osmtags.way_id(str(f.attrs.get("src_record", [""] * len(f))[i]))
                   for fid, (f, i, sp) in seen.items()}
        local_plans = []
        advance_context = []
        direction_margin = 0.0
        if detailed_signs and roads:
            import frlocaldirections
            reach = frlocaldirections.ADVANCE_REACH_M
            local_plans = frlocaldirections.nearby(frame, own_box, reach)
            if local_plans:
                ring_data = ring_dir or any("exit" in p for plan in local_plans for p in plan["panels"])
                direction_margin = 200.0 if ring_data else 48.0
        # Short features need their neighbours to test a sustained town approach.
        # Context only: these lines never add carriageways, poles or junctions.
        sign_context = list(roads)
        own_fids = {r[0] for r in roads}
        for fid, (f, i, sp) in sorted(seen.items()):
            town_approach = sp.cls in townsigns.CLASSES and not sp.link
            advance_road = direction_margin > 0 and sp.cls in frlocaldirections.LOCAL and sp.kind == 'paved'
            if fid in own_fids or not (town_approach or advance_road):
                continue
            fb = f.boxes(frame)[i]
            if fb[2] < x0-200 or fb[0] > x1+200 or fb[3] < y0-200 or fb[1] > y1+200:
                continue
            if advance_road:
                m = direction_margin
                advance_road = fb[2] >= x0-m and fb[0] <= x1+m and fb[3] >= y0-m and fb[1] <= y1+m
            if not (town_approach or advance_road):
                continue
            for ring in ground_parts(f, i):
                ex, ny = mapgeom.enu_xy(f, frame, ring)
                if len(ex) <= 1:
                    continue
                lon = float(np.mean(ring[:, 0]))
                if town_approach:
                    sign_context.append((fid, sp, np.c_[ex, ny, np.zeros(len(ex))], lon))
                if advance_road:
                    advance_context.append((fid, sp, np.c_[ex, ny], ring, lon))
        if advance_context:
            lon, lat = np.concatenate([row[3] for row in advance_context]).T
            heights = frame.geodetic_to_enu(lat, lon, np.asarray(elev(lat, lon), np.float64))[2]
            start = 0
            extra = []
            for fid, sp, xy, _, lon in advance_context:
                end = start + len(xy)
                extra.append((fid, sp, np.c_[xy, heights[start:end]], lon))
                start = end
            roundabouts = {**ring_dir, **islands.ring_directions([*roads, *extra], left_hand)}
            advance_context = [(fid, sp._replace(oneway=True, dir=roundabouts[fid]) if fid in roundabouts else sp,
                                line, lon) for fid, sp, line, lon in extra]
        else:
            roundabouts = ring_dir
        # (PR #597's town signs on, BNE_TOWNSIGNS: theirs, everywhere, instead of these boards)
        sign_mesh, sign_posts, sign_records = (None, [], []) if towns is not None else townsigns.place(frame, roads, own_box, pole_profile,
                                                           ground=ground, urban=urban_fn, way_ids=way_ids,
                                                           context_roads=sign_context,
                                                           sign_obstacles=sign_obstacles, decks=sign_decks,
                                                           local_directions=detailed_signs,
                                                           street_names={fid:str(f.attrs.get('name',['']*len(f))[i]or'')
                                                                         for fid,(f,i,sp)in seen.items()})
        if sign_mesh is not None:
            signs.append((sign_mesh, sign_records))
        props.extend(sign_posts)
    if detailed_signs:
        def finish_roadside(guardrail_rows, extra_pavement=()):
            import frgore
            import frlocaldirections
            import signmesh

            country_at = poleprofiles.country_selector(pole_profile, frame)
            if walks is not None:
                props.extend(frgore.place(frame, roads, own_box, country_at, ground, sign_decks,
                                         guardrails=guardrail_rows, obstacles=sign_obstacles))
            if signs is not None:
                mesh = sign_mesh if sign_mesh is not None else signmesh.Mesh((x0, y0, 0.))
                added = frlocaldirections.add(
                    frame, roads, own_box, poleprofiles.kit_selector(pole_profile, frame), mesh,
                    way_ids=way_ids, ground=ground, obstacles=sign_obstacles, decks=[*sign_decks, *extra_pavement],
                    guardrails=guardrail_rows, roundabouts=roundabouts, props=props, records=sign_records,
                    urban_fn=urban_fn, data=local_plans, context_roads=advance_context,
                    pavement_rings=[np.asarray(ch['poly'])[:, :2] for ch in chunks if len(ch.get('poly', ())) >= 3])
                if sum(sp.cls == "motorway" and sp.oneway and not sp.link for _, sp, _, _ in roads) >= 2:
                    import frmotorwaymarkers
                    import signplacement

                    markers = frmotorwaymarkers.nearby(frame, own_box)
                    if markers:
                        pavement = signplacement.pavement([(sp.width / 2, line) for _, sp, line, _ in roads]
                                                         + [(half, line) for line, half in [*sign_decks, *extra_pavement]])
                        obstacles = sign_obstacles if sign_obstacles is not None else signplacement.Context(ground=ground)
                        added.extend(frmotorwaymarkers.add(
                            frame, roads, own_box, poleprofiles.kit_selector(pole_profile, frame), mesh,
                            pavement, obstacles, frlocaldirections.barrier_context(guardrail_rows),
                            guardrails_rows=guardrail_rows, decks=[*sign_decks, *extra_pavement], data=markers))
                import frspeedcams
                cameras = frspeedcams.nearby(frame, own_box)
                if cameras:                     # fixed speed cameras (OSM), beside the road or in its median
                    import signplacement
                    pavement = signplacement.pavement([(sp.width / 2, line) for _, sp, line, _ in roads]
                                                     + [(half, line) for line, half in [*sign_decks, *extra_pavement]])
                    obstacles = sign_obstacles if sign_obstacles is not None else signplacement.Context(ground=ground)
                    added.extend(frspeedcams.add(
                        frame, roads, own_box, poleprofiles.kit_selector(pole_profile, frame), mesh, pavement,
                        obstacles, frlocaldirections.barrier_context(guardrail_rows), data=cameras))
                sign_records.extend(added)
                if sign_mesh is None and added:
                    signs.append((mesh, sign_records))

        if roadside_jobs is None:
            finish_roadside(())
        else:
            roadside_jobs.append(finish_roadside)
    if towns is not None and props is not None and walks is not None:     # town entry and exit signs (townentry.py)
        import townentry
        c = townentry.CLEAR_M                   # (the signs of the next tiles clear this one's edge too)
        town_rows = towns(roads, (x0 - c, y0 - c, x1 + c, y1 + c), left_hand,
                          {fid for fid, (f, i, sp) in seen.items() if is_ring(f, i)})
        props[:] = townentry.clear_around(props, town_rows)
        props.extend(s for s in town_rows if x0 <= s[1] < x1 and y0 <= s[2] < y1)
    if props is not None and walks is not None:     # French toll plazas: islands and barriers (frtolls.py)
        import franceroads
        if franceroads.in_france(frame.lat0, frame.lon0):
            import frtolls
            props.extend(frtolls.rows(frame, roads, own_box))
    aisles = {fid for fid, (f, i, sp) in seen.items()
              if sp.kind == "paved" and _rule_value(f.rules.get("subclass"), i) == "parking_aisle"}
    if (park is not None and walks is not None) or (marks is not None and aisles):
        import parking
        if park is not None and walks is not None:      # curbside parking spots (parking.py)
            dirs = {}
            for fid, (f, i, sp) in seen.items():
                if sp.oneway:
                    v = _rule_value(f.rules.get("oneway"), i)
                    dirs[fid] = (-1 if float(v) < 0 else 1) if v is not None else (sp.dir or 1)
            park.extend(parking.curb_spots(roads, walks, own_box, left_hand, dirs))
            if aisles:
                park.extend(parking.aisle_spots(roads, aisles, own_box))
        if marks is not None and aisles:            # stall lines along the aisles (parking.aisle_markings)
            marks.extend(parking.aisle_markings(roads, aisles, wexact))
    chunks.extend(rampmerges.chunks(merge_layouts, box))
    return roadmarkings.paint(chunks)


class Tracker:
    """Per connection: which road tiles the mod has, building and announcing the ones it needs."""

    def __init__(self):
        self.aid = None
        self.sent = set()       # (tx, ty) announced for self.aid
        self.building = set()
        self.pool = None        # its own BUILD_PER_POSE threads (vegetation.Tracker)
        self.futs = {}
        # (from KyzoFox's 1.5.4, PR #64) a tile that came back with no roads was built again at every pose: kept here
        # while it is within reach; a failed one is asked again after TILE_RETRY_S, not at the next pose
        self.empty = set()
        self.retry = {}         # tile -> monotonic time it may be built again
        self.fails = {}         # tile -> builds failed in a row (TILE_FAILS_WARN)
        self.started = {}       # tile -> monotonic time its build started (TILE_STUCK_S)
        self.stuck_said = set()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def again(self, keys):
        """The sent tiles among keys built again at the next pose and sent again (no roads_drop: the mod replaces a
        tile it has; the OSM test: a cell's OpenStreetMap fuel stations came after them, earthd Server.osm_fuel_want)."""
        self.sent -= set(keys)

    def progress(self, near_m=None):
        """(done, wanted) of the road tiles round the car at the last pose: done = sent to the mod, empty, or failed
        and waiting for its retry (nothing more comes for it now). (0, 0) before a pose. (near_m: as GeoTracker's;
        these tiles carry no distances, all count.)"""
        car = getattr(self, "car", ())
        done = sum(1 for k in car if k in self.sent or k in self.empty or k in self.retry)
        return done, len(car)

    def on_pose(self, conn, aid, x, y, loop, build, vel=(0.0, 0.0), ground_busy=False, view=None, foci=()):
        """Called with the player's position (and velocity, m/s) in the active anchor's frame. build(aid, tx, ty) runs
        in a worker and returns the tile's (path, n) or None; its result is announced with conn.send. ground_busy: a
        terrain block the mod asked for is being built: no tile beyond SEND_M starts meanwhile (the ground first; at
        180 mph the block ahead waited 7.5 s among them, 2026-10-19). view: the camera's (x, y) when it is away from
        the car (earthd camera_away): the tiles round it are wanted too, after the car's (which carry what it drives
        on and hits). foci: other vehicles' collision foci (x, y, r, trees) (the mod's collisionfoci.lua): the tiles
        within r + TILE / 2 of each are wanted too, after the car's and the camera's, and kept within r + TILE."""
        if aid != self.aid:
            for f in self.futs.values():
                f.cancel()
            self.aid, self.sent, self.building, self.futs = aid, set(), set(), {}
            self.empty, self.retry, self.fails = set(), {}, {}
            self.started, self.stuck_said = {}, set()
        if self.pool is None:
            from runtimeprofiler import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(BUILD_PER_POSE, thread_name_prefix="roads-tile")
        vx, vy = (float(vel[0]), float(vel[1])) if vel else (0.0, 0.0)
        fast = math.hypot(vx, vy) >= FAST_MS
        car = wanted_tiles(x, y, vx, vy)
        self.car = car                      # (progress: the loading line's road tiles)
        seen = set(tiles_near(*view, SEND_M)) - car if view else set()
        others = set()
        for f in foci:
            others.update(tiles_near(f[0], f[1], min(SEND_M, f[2] + TILE / 2)))
        others -= car | seen
        wanted = car | seen | others
        start = set(tiles_near(x, y, SEND_M)) if ground_busy else wanted
        now = time.monotonic()
        want = [k for k in start if k not in self.sent and k not in self.building and k not in self.empty
                and self.retry.get(k, 0.0) <= now]
        car_key = path_key(x, y, vx, vy) if fast else view_key(conn, aid, x, y)
        cam_key = view_key(conn, aid, *view) if seen else None
        def focus_d(k):
            return min(tile_distance(k[0], k[1], f[0], f[1]) for f in foci)
        want.sort(key=lambda k: (2, focus_d(k)) if k in others else (1, cam_key(k)) if k in seen else (0, car_key(k)))
        for tx, ty in want[:max(0, BUILD_PER_POSE - len(self.building))]:   # in flight at most (vegetation.Tracker)
            self.building.add((tx, ty))
            self.started[(tx, ty)] = now
            fut = self.futs[(tx, ty)] = loop.run_in_executor(self.pool, build, aid, tx, ty)
            fut.add_done_callback(lambda f, k=(tx, ty), a=aid: self._built(conn, a, k, f))
        # a build that never ends leaves the loading line at "roads & buildings 0/36" with nothing in the log (GitHub
        # #241, a Discord report): said once per tile, with how many are in flight
        for k in self.building:
            t0 = self.started.get(k)
            if t0 is not None and now - t0 > TILE_STUCK_S and k not in self.stuck_said:
                self.stuck_said.add(k)
                # (not "may be stuck": players read it as broken and cleared their caches, which only made the next
                # try slower; the time counts the wait for a free road process too)
                log.warning("roads tile %s of anchor %d: not ready after %.0f s (%d tiles in flight, %d sent): "
                            "a first visit on a slow or busy machine can take this long; a road process that hangs is restarted by itself after a few minutes (clearing the cache does not help)", k, aid, now - t0, len(self.building), len(self.sent))
        def far(k):
            return tile_distance(k[0], k[1], x, y) > DROP_M and (not view or tile_distance(k[0], k[1], *view) > DROP_M) \
                and all(tile_distance(k[0], k[1], f[0], f[1]) > f[2] + TILE for f in foci)
        drop = [k for k in self.sent if k not in wanted and far(k)]
        for k in drop:
            self.sent.discard(k)
            conn.send({"type": "roads_drop", "anchorId": aid, "key": f"{k[0]}_{k[1]}"})
        if self.empty:
            self.empty = {k for k in self.empty if k in wanted or not far(k)}
        if self.retry:
            self.retry = {k: t for k, t in self.retry.items() if t > now}

    def _built(self, conn, aid, key, fut):
        if aid != self.aid:                 # (an old anchor's: the same key may be in flight for the new one)
            return
        self.building.discard(key)
        self.futs.pop(key, None)
        self.started.pop(key, None)
        if fut.cancelled():
            return
        try:
            res = fut.result()
        except Exception as e:      # network trouble: the next pose asks again
            # (a bug failed every tile on a Steam Deck with nothing in the log, 2026-10-20: each kind once, in full)
            kind = f"{type(e).__name__}: {e}"[:200]
            fails = self.fails[key] = self.fails.get(key, 0) + 1
            if kind not in _FAILS_SEEN and not isinstance(e, (OSError, TimeoutError)):
                _FAILS_SEEN.add(kind)
                log.warning("roads tile %s failed: %s", key, kind, exc_info=e)
            elif fails == TILE_FAILS_WARN:      # network trouble that does not pass (GitHub #91: no roads, no log)
                log.warning("roads tile %s of anchor %d: %d builds failed in a row, the last: %s", key, aid, fails, kind)
            else:
                log.debug("roads tile %s: %s: %s", key, type(e).__name__, e)
            self.retry[key] = time.monotonic() + TILE_RETRY_S
            return
        self.fails.pop(key, None)
        if res is None:                     # no roads there: not built again while it stays within reach
            self.empty.add(key)
            return
        path, n = res
        self.sent.add(key)
        m = {"type": "roads_tile", "anchorId": aid, "key": f"{key[0]}_{key[1]}", "path": path, "n": n}
        out_dir = getattr(getattr(conn, "s", None), "out_dir", None)
        data = staging.inline_json(out_dir, path) if out_dir else None
        if data is not None:                # (the game need not open the file: staging.inline_json)
            m["data"] = data
        pool = getattr(getattr(conn, "s", None), "pool_shapes", None)
        if pool is not None:                # its meshes at pooled shape paths (matslots.py)
            slots = pool(conn, shape_files(aid, *key))
            if slots:
                m["slots"] = slots
        conn.send(m)


_FAILS_SEEN = set()             # the kinds of road tile failures logged in full (Tracker._built)


def geo_wanted(frame, x, y, vx=0.0, vy=0.0):
    """wanted_tiles for geotiles (their quads in frame's world): those within SEND_M, and at FAST_MS or more those
    within CORRIDOR_M of the path ahead (LEAD_S of driving, LEAD_MAX_M at most). {(gx, gy): m from (x, y)}."""
    import geotiles
    want = {(a, b): d for a, b, d in geotiles.tiles_near(frame, x, y, SEND_M)}
    s = math.hypot(vx, vy)
    lead = min(LEAD_MAX_M, s * LEAD_S)
    if s < FAST_MS or lead <= SEND_M:
        return want
    ux, uy = vx / s, vy / s
    d = 0.0
    while d <= lead:
        for a, b, _dd in geotiles.tiles_near(frame, x + ux * d, y + uy * d, CORRIDOR_M):
            want.setdefault((a, b), d)
        d += TILE / 2
    return want


class GeoTracker:
    """Per connection: the road geotiles the mod has. Content is sent once for the connection (kept by the mod across
    anchors, sent: its keys), skins per anchor (skins), exact variants per anchor (shown: the variant's anchor per
    part the mod shows; none: its base shape). build(aid, gx, gy, content, skin, exact) runs in a worker and returns
    blockworker.roads_geo's result."""

    geo = True

    def __init__(self):
        self.aid = None
        self.frame = None
        self.frames = {}        # anchor id -> its frame (the exact variants' error: _shown_err)
        self.sent = set()       # (gx, gy) whose content the mod has
        self.info = {}          # (gx, gy) -> its content's {"h", "parts": {what: {pivot, relief, col}}}
        self.skins = set()      # (gx, gy) with self.aid's skin sent
        self.shown = {}         # (gx, gy) -> {what: anchor id of the exact variant the mod shows}
        self.need = {}          # ((gx, gy), aid) -> {what: pivot} its exact variants due (exact_need, cached)
        self.building = set()
        self.pool = None
        self.futs = {}
        self.retry = {}
        self.fails = {}
        self.started = {}
        self.stuck_said = set()
        self.car = set()
        self.empty = set()      # (anchor tiles' interface: progress)
        self.stats = {"content": 0, "skins": 0, "exact": 0, "exact_base": 0, "anchors": 0,
                      "content_cpu": 0.0, "skin_cpu": 0.0}

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def progress(self, near_m=None):
        """(done, wanted) of the tiles round the car: done = content and this anchor's skin sent, or waiting for a
        retry. near_m: only those within it of the car (the catch-up boost: the ground under the car, not SEND_M's
        far edge, where a tile is always on its way while driving)."""
        car = self.car
        if near_m is not None:
            dist = getattr(self, "car_m", {}) or {}
            car = [k for k in car if dist.get(k, 0.0) <= near_m]
        done = sum(1 for k in car if (k in self.sent and k in self.skins) or k in self.retry)
        return done, len(car)

    def ready_tiles(self, frame, x, y, r):
        """The anchor road tiles (tx, ty) within r of (x, y) (tiles_near) whose road geotiles are all with the mod
        (content and this anchor's skin, or waiting for a retry): their highway and speed signs may come (signstream).
        Before (2026-10-29) the signs waited until every road geotile round the car was in, which while driving it
        seldom was: signs came only once the car slowed down or stopped, one tile at a time, right next to it."""
        import geotiles
        reach = TILE / 2 * math.sqrt(2.0)
        out = set()
        for tx, ty in tiles_near(x, y, r):
            ks = geotiles.tiles_near(frame, (tx + 0.5) * TILE, (ty + 0.5) * TILE, reach, exact=True)
            if all(((a, b) in self.sent and (a, b) in self.skins) or (a, b) in self.retry for a, b, _ in ks):
                out.add((tx, ty))
        return out

    def again(self, keys):
        """The sent geotiles among keys built again (content and this anchor's skin) at the next pose and sent again
        (no roads_drop: the mod replaces a geotile whose content hash changed, roads.lua onGeoTile; earthd
        Server.osm_fuel_want: a cell's OpenStreetMap fuel stations came after them)."""
        for k in set(keys) & self.sent:
            self.sent.discard(k)
            self.skins.discard(k)
            self.need = {nk: v for nk, v in self.need.items() if nk[0] != k}

    def _shown_err(self, key, what, part, frame):
        """m the part (pivot, relief) of tile key is off in frame's world as the mod shows it (its base shape, or the
        exact variant of the anchor in shown) under rigid placement."""
        import geotiles
        tf = geotiles.frame(*key)
        piv = float(part.get("pivot", 0.0))
        rel = float(part.get("relief", 0.0))
        r_b, _t, c_b = geotiles.placement(tf, frame, (0.0, 0.0, piv))
        src = self.shown.get(key, {}).get(what)
        if src is None or src not in self.frames:
            off = c_b
        else:
            r_a, _ta, c_a = geotiles.placement(tf, self.frames[src], (0.0, 0.0, piv))
            off = r_b @ r_a.T @ c_a - c_b
        return rel * float(np.hypot(off[0], off[1]))

    def exact_need(self, key, aid=None, frame=None):
        """({what: {pivot}} of tile key's parts whose exact variant for the anchor must be made, [what] of those
        whose base shape is exact enough again while the mod shows a variant)."""
        aid = self.aid if aid is None else aid
        frame = self.frame if frame is None else frame
        info = self.info.get(key)
        if info is None or frame is None:
            return {}, []
        make, base = {}, []
        for what, part in (info.get("parts") or {}).items():
            tol = EXACT_TOL_COL_M if part.get("col") else EXACT_TOL_M
            src = self.shown.get(key, {}).get(what)
            if src == aid:
                continue
            if self._shown_err(key, what, part, frame) <= tol:
                continue
            saved = self.shown.get(key, {}).pop(what, None)         # (the base shape's error, for the choice)
            base_err = self._shown_err(key, what, part, frame)
            if saved is not None:
                self.shown[key][what] = saved
            if base_err <= tol:
                base.append(what)
            else:
                make[what] = {"pivot": part.get("pivot", 0.0)}
        return make, base

    def on_pose(self, conn, aid, x, y, loop, build, vel=(0.0, 0.0), ground_busy=False, view=None, foci=(),
                frame=None):
        import geotiles
        if frame is None:
            return
        if aid != self.aid:
            for f in self.futs.values():
                f.cancel()
            self.aid, self.frame, self.building, self.futs = aid, frame, set(), {}
            self.frames[aid] = frame
            while len(self.frames) > 4:
                old = min(a for a in self.frames if a != aid)
                del self.frames[old]
            self.skins, self.retry, self.fails, self.started, self.stuck_said = set(), {}, {}, {}, set()
            self.need = {}
            self.stats["anchors"] += 1
        if self.pool is None:
            from runtimeprofiler import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(BUILD_PER_POSE, thread_name_prefix="roads-geo")
        vx, vy = (float(vel[0]), float(vel[1])) if vel else (0.0, 0.0)
        s = math.hypot(vx, vy)
        car = geo_wanted(frame, x, y, vx, vy)
        self.car = set(car)
        self.car_m = car                    # (progress(near_m): each wanted tile's m from the car)
        seen = {(a, b): d for a, b, d in geotiles.tiles_near(frame, *view, SEND_M)} if view else {}
        others = {}
        for f in foci:
            for a, b, d in geotiles.tiles_near(frame, f[0], f[1], min(SEND_M, f[2] + TILE / 2)):
                others.setdefault((a, b), d)
        wanted = set(car) | set(seen) | set(others)
        start = {(a, b) for a, b, _ in geotiles.tiles_near(frame, x, y, SEND_M)} if ground_busy else wanted
        now = time.monotonic()
        jobs = []
        for k in start:
            if k in self.building or self.retry.get(k, 0.0) > now:
                continue
            if k not in self.sent:
                jobs.append((k, True, True, "auto"))
                continue
            make = None
            if k not in self.skins or (k, self.aid) not in self.need:
                mk, base = self.exact_need(k)
                self.need[(k, self.aid)] = mk
                if base:                    # (the base shape is exact enough again: no build)
                    for w in base:
                        self.shown.get(k, {}).pop(w, None)
                    self.stats["exact_base"] += len(base)
                    conn.send({"type": "roads_exact", "anchorId": aid, "key": geotiles.key(*k),
                               "parts": {w: False for w in base}})
            make = self.need.get((k, self.aid)) or None
            if k not in self.skins or make:
                jobs.append((k, False, k not in self.skins, make))

        def order(job):
            k = job[0]
            if k in car:
                d = car[k]
                if s >= FAST_MS:        # (ahead first: the corridor's distance along the path)
                    q = geotiles.quad(frame, *k)
                    cx, cy = float(np.mean(q[:, 0])), float(np.mean(q[:, 1]))
                    along = ((cx - x) * vx + (cy - y) * vy) / s
                    if along < -TILE / 2:
                        d = d * SIDE_PENALTY + LEAD_MAX_M
                    else:
                        # (by the distance along the path, a tile off the corridor later by how far off it is: by
                        # their plain distance the sides within SEND_M all came before the corridor beyond it, and
                        # at 500 mph the car outran its road tiles, 0 m of them ahead, Indianapolis 2026-10-30)
                        side = abs(((cx - x) * vy - (cy - y) * vx) / s)
                        d = max(0.0, along) + max(0.0, side - CORRIDOR_M) * SIDE_PENALTY
                return (0, d)
            if k in seen:
                return (1, seen[k])
            return (2, others.get(k, 1e9))
        jobs.sort(key=order)
        for k, content, skin, make in jobs[:max(0, BUILD_PER_POSE - len(self.building))]:
            self.building.add(k)
            self.started[k] = now
            fut = self.futs[k] = loop.run_in_executor(self.pool, build, aid, k[0], k[1], content, skin, make)
            fut.add_done_callback(lambda f, kk=k, a=aid, c=content, sk=skin, m=make:
                                  self._built(conn, a, kk, f, c, sk, m))
        for k in self.building:
            t0 = self.started.get(k)
            if t0 is not None and now - t0 > TILE_STUCK_S and k not in self.stuck_said:
                self.stuck_said.add(k)
                log.warning("road geotile %s of anchor %d: not ready after %.0f s (%d in flight, %d sent): "
                            "a first visit on a slow or busy machine can take this long; a road process that hangs is restarted by itself after a few minutes (clearing the cache does not help)", k, aid, now - t0, len(self.building), len(self.sent))

        def far(k):
            q = geotiles.quad(frame, *k)
            if geotiles.quad_dist(q, x, y) <= DROP_M:
                return False
            if view and geotiles.quad_dist(q, *view) <= DROP_M:
                return False
            return all(geotiles.quad_dist(q, f[0], f[1]) > f[2] + TILE for f in foci)
        for k in [k for k in self.sent if k not in wanted and k not in self.building and far(k)]:
            self.sent.discard(k)
            self.skins.discard(k)
            self.info.pop(k, None)
            self.shown.pop(k, None)
            self.need = {nk: v for nk, v in self.need.items() if nk[0] != k}
            conn.send({"type": "roads_drop", "anchorId": aid, "key": geotiles.key(*k)})
            gone = getattr(getattr(conn, "s", None), "geo_dropped", None)
            if callable(gone):              # (its session files: deleted once no game has it, GitHub #599)
                gone("roads", k, geotiles.key(*k))
        if self.retry:
            self.retry = {k: t for k, t in self.retry.items() if t > now}

    def _built(self, conn, aid, key, fut, content, skin, make):
        import geotiles
        if self.futs.get(key) is fut:
            self.building.discard(key)
            self.futs.pop(key, None)
            self.started.pop(key, None)
        if fut.cancelled():
            return
        try:
            res = fut.result()
        except Exception as e:
            kind = f"{type(e).__name__}: {e}"[:200]
            fails = self.fails[key] = self.fails.get(key, 0) + 1
            if kind not in _FAILS_SEEN and not isinstance(e, (OSError, TimeoutError)):
                _FAILS_SEEN.add(kind)
                log.warning("road geotile %s failed: %s", key, kind, exc_info=e)
            elif fails == TILE_FAILS_WARN:
                log.warning("road geotile %s: %d builds failed in a row, the last: %s", key, fails, kind)
            else:
                log.debug("road geotile %s: %s: %s", key, type(e).__name__, e)
            if aid == self.aid:
                self.retry[key] = time.monotonic() + TILE_RETRY_S
            return
        if res is None:
            self.retry[key] = time.monotonic() + TILE_RETRY_S
            return
        self.fails.pop(key, None)
        gkey = geotiles.key(*key)
        s = getattr(conn, "s", None)
        out_dir = getattr(s, "out_dir", None)
        pool = getattr(s, "pool_shapes", None)
        tm = res.get("time") or {}
        self.stats["content_cpu"] += float(tm.get("content_cpu", 0.0))
        self.stats["skin_cpu"] += float(tm.get("skin_cpu", 0.0))
        c = res.get("content")
        ex = (res.get("exact") or {}) if aid == self.aid else {}
        if c is not None:
            path, n, info = c[0], c[1], c[2]
            self.sent.add(key)
            self.info[key] = info
            self.shown.pop(key, None)
            self.stats["content"] += 1
            m = {"type": "roads_tile", "anchorId": aid, "key": gkey, "path": path, "n": n,
                 "frame": list(geotiles.centre(*key)), "h": info.get("h")}
            data = staging.inline_json(out_dir, path) if out_dir else None
            if data is not None:
                m["data"] = data
            if ex:                      # (the parts this anchor needs exact: the mod loads those, not the base)
                m["exact"] = dict(ex)
            if pool is not None:
                import blockworker
                names = blockworker.geo_names(gkey)
                files = {w: n_ for w, n_ in names.items() if w not in ("json", "inputs") and w not in ex}
                files.update({f"x:{w}": os.path.basename(p) for w, p in ex.items()})
                slots = pool(conn, files)
                if slots:
                    m["slots"] = slots
            conn.send(m)
        if aid != self.aid:             # (an old anchor's skin and variants: the new anchor's come on their own)
            return
        sk = res.get("skin")
        if skin and sk is not None:
            self.skins.add(key)
            self.stats["skins"] += 1
            m = {"type": "roads_skin", "anchorId": aid, "key": gkey, "path": sk[0]}
            data = staging.inline_json(out_dir, sk[0]) if out_dir else None
            if data is not None:
                m["data"] = data
            if pool is not None:
                import blockworker
                names = blockworker.geo_names(gkey, aid)
                slots = pool(conn, {w: n_ for w, n_ in names.items() if w != "json"})
                if slots:
                    m["slots"] = slots
            conn.send(m)
        for w in ex:
            self.shown.setdefault(key, {})[w] = aid
        self.stats["exact"] += len(ex)
        if c is not None:               # (what is left due: none, with "auto")
            mk, _base = self.exact_need(key)
            self.need[(key, aid)] = mk
        elif ex:
            m = {"type": "roads_exact", "anchorId": aid, "key": gkey, "parts": dict(ex)}
            if pool is not None:
                slots = pool(conn, {w: os.path.basename(p) for w, p in ex.items()})
                if slots:
                    m["slots"] = slots
            conn.send(m)
        if make and c is None:
            self.need[(key, aid)] = {w: v for w, v in (self.need.get((key, aid)) or {}).items() if w not in ex}


def geo_exact_auto(tile_frame, frame, parts):
    """{what: {pivot}} of a fresh content's parts (parts: {what: {pivot, relief, col}}) whose base shape placed rigidly
    in frame's world would be off by more than their tolerance (EXACT_TOL_COL_M colliding, EXACT_TOL_M else)."""
    import geotiles
    out = {}
    for what, part in (parts or {}).items():
        piv = float(part.get("pivot", 0.0))
        _r, _t, c = geotiles.placement(tile_frame, frame, (0.0, 0.0, piv))
        tol = EXACT_TOL_COL_M if part.get("col") else EXACT_TOL_M
        if float(part.get("relief", 0.0)) * float(np.hypot(c[0], c[1])) > tol:
            out[what] = {"pivot": piv}
    return out


def shape_files(aid, tx, ty):
    """The shape files write_tile may write for a tile, by the field naming them in the tile ({what: file name})."""
    out = {"road": f"road_{aid}_{tx}_{ty}.cdae", "deck": f"deck_{aid}_{tx}_{ty}.cdae",
           "walk": f"walks_{aid}_{tx}_{ty}.cdae", "walkcol": f"walkcol_{aid}_{tx}_{ty}.cdae",
           "bldcol": f"bldcol_{aid}_{tx}_{ty}.cdae", "wire": f"wire_{aid}_{tx}_{ty}.cdae",
           "bld": f"bld_{aid}_{tx}_{ty}.cdae", "tunnel": f"tunnel_{aid}_{tx}_{ty}.cdae",
           "rail": f"rail_{aid}_{tx}_{ty}.cdae", "signs": f"signs_{aid}_{tx}_{ty}.cdae", "railcol": f"railcol_{aid}_{tx}_{ty}.cdae",
           "barrier": f"barrier_{aid}_{tx}_{ty}.cdae"}
    for k in range(2, BLD_PARTS_MAX + 1):
        out[f"bld{k}"] = f"bld_{aid}_{tx}_{ty}_{k}.cdae"
    return out


def _along(line, dist):
    """The point dist along polyline line (k, 3) from its start (its end if shorter)."""
    seg = np.hypot(*np.diff(line[:, :2], axis=0).T)
    for i, L in enumerate(seg):
        if dist <= L and L > 0:
            return line[i] + (line[i + 1] - line[i]) * (dist / L)
        dist -= L
    return line[-1]


def sound_boxes(chunks):
    """The sound spaces of a tile's chunks: along every tunnel tube (its floor to its vault) and under every bridge
    deck (from its slab down SOUND_UNDER_M), as earthfast.sound_boxes rows [kind, x, y, z, length, width, height, yaw]
    rounded, one flat list; first a closed tunnel's portal points (SOUND_PORTAL) at its mouths."""
    import fast
    lines, out = [], []
    for c in chunks:
        if c.get("tunnel") and not c.get("no_tube"):
            h = float(c.get("tunnel_h") or roadmesh.TUNNEL_H)
            short = h < roadmesh.TUNNEL_H - 0.01 or float(c.get("tunnel_len") or 1e9) < SOUND_SHORT_TUNNEL_M
            lines.append((SOUND_TUNNEL_OPEN if short else SOUND_TUNNEL, c["n"],
                          float(c["w"]) + 2 * roadmesh.TUNNEL_SHOULDER, 0.5, h + 0.5))
            n = np.asarray(c["n"], np.float64)[:, :3]
            for end, line in ((0, n), (1, n[::-1])):
                if not short and (c.get("portal") or (False, False))[end] and len(line) >= 2:
                    p = _along(line, SOUND_PORTAL_IN_M)
                    d = line[1, :2] - line[0, :2]
                    out += [float(SOUND_PORTAL), round(float(p[0]), 2), round(float(p[1]), 2),
                            round(float(p[2]) + SOUND_PORTAL_UP, 2), 0.0, 0.0, 0.0,
                            round(math.atan2(float(d[1]), float(d[0])), 4)]
        elif c.get("deck") and not c.get("tunnel"):
            lines.append((SOUND_BRIDGE, c["n"], float(c["w"]), SOUND_UNDER_M, -roadmesh.DECK_DEPTH))
    if not lines:
        return out
    fast.require(78, "sound_boxes")
    pts = np.ascontiguousarray(np.concatenate([np.asarray(n, np.float64)[:, :3] for _, n, _, _, _ in lines]))
    off = np.cumsum([0] + [len(n) for _, n, _, _, _ in lines]).astype(np.int64)
    col = lambda k: np.ascontiguousarray([ln[k] for ln in lines], np.float64)     # noqa: E731
    rows = fast._ef.sound_boxes(pts, off, col(0), col(2), col(3), col(4), SOUND_BOX_M, SOUND_TURN, SOUND_DZ_M,
                                SOUND_EXTEND_M)
    rows[:, 1:7] = np.round(rows[:, 1:7], 2)
    rows[:, 7] = np.round(rows[:, 7], 4)
    return out + rows.ravel().tolist()


def write_tile(out_dir, vfs_dir, aid, tx, ty, chunks, walks=None, road_mesh=None, deck_mesh=None, bld=None,
               props=None, wires=None, park=None, tunnel_mesh=None, fuel=None,
               rail_mesh=None, garage_sound=(), wells=(), guardrail_rows=None, custom_entries=None, decals=True,
               bld_none=False, barrier_mesh=None, barrier_chunks=(), turbines=(), signs=None, towns=None):
    """Write a tile's chunks (atomically), its sidewalk mesh (walks: build_tile's runs, if any:
    `walks_<a>_<tx>_<ty>.cdae`, announced in the tile as "walk": {shape, pos}) and its road mesh
    (road_mesh: roadmesh.build's result: `road_<a>_<tx>_<ty>.cdae`, "road": {shape, pos}; the mod then
    shows it instead of a DecalRoad per chunk) and its bridge decks (deck_mesh: roadmesh.deck's result:
    `deck_<a>_<tx>_<ty>.cdae`, "deck": {shape, pos}, colliding); returns (vfs path, number of chunks). bld_none: the
    buildings are on and the tile has none ("bldNone": the mod hides its far building tile, which has every outline;
    a gas station's canopy outlines are left out of the tile's buildings only, the user, 2026-10-29: "the third
    canopy is a building"; an anchor tile's: a geotile's far tile hides on its parts, roads.lua nearShowsBuildings).
    towns: the town signs' faces its props use ({group: {shape, texture, material}}: townsigns.write_face's, "townsigns"; the mod makes their
    materials before the props)."""
    import sidewalks
    name = f"roads_{aid}_{tx}_{ty}.json"
    path = os.path.join(out_dir, name)
    # (from PR #41, Madderose) with a road mesh the mod makes no DecalRoads: their chunks were ~36 % of the file it
    # parses on its main thread
    data = {"v": VERSION, "chunks": [] if road_mesh is not None or not decals else
            [{k: v for k, v in c.items() if k != "waterguard"} for c in chunks if not c.get("tube_only")]}
    guards = {c["waterguard"]["id"]: c["waterguard"] for c in chunks if c.get("waterguard")}
    if guards:
        data["waterguard"] = [guards[k] for k in sorted(guards)]
    sound = sound_boxes(chunks) + list(garage_sound)      # (garage_sound: buildings.garage_sound_boxes)
    if sound:
        data["sound"] = sound
    if wells:                                               # oil wells' pump jacks (oilwells.tile_wells)
        data["wells"] = list(wells)
    if turbines:                                            # mapped OSM wind turbines (near collision owners)
        data["turbines"] = list(turbines)
    if road_mesh is not None:
        x0, y0, _, _ = tile_bbox(tx, ty)
        shape = f"road_{aid}_{tx}_{ty}.cdae"
        roadmesh.write(os.path.join(out_dir, shape), road_mesh)
        data["road"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
    if deck_mesh is not None:
        x0, y0, _, _ = tile_bbox(tx, ty)
        shape = f"deck_{aid}_{tx}_{ty}.cdae"
        roadmesh.write(os.path.join(out_dir, shape), deck_mesh)
        data["deck"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
    if tunnel_mesh is not None:     # roadmesh.tunnel: the tubes and portals (colliding)
        x0, y0, _, _ = tile_bbox(tx, ty)
        shape = f"tunnel_{aid}_{tx}_{ty}.cdae"
        roadmesh.write(os.path.join(out_dir, shape), tunnel_mesh)
        data["tunnel"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
    if guardrail_rows:              # mapped guard rails and the median barriers (guardrails / medians): Forest rows
        import guardrails
        data["guardrails"] = {"items": guardrail_rows, "groups": list(guardrails.groups_for(guardrail_rows))}
    if rail_mesh is not None:       # railways.tile_mesh: ballast, sleepers and rails
        import cdae
        import railways
        x0, y0, _, _ = tile_bbox(tx, ty)
        shape = f"rail_{aid}_{tx}_{ty}.cdae"
        roadmesh.write(os.path.join(out_dir, shape), rail_mesh)
        data["rail"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
        col = railways.collision_mesh(rail_mesh)
        if col is not None:         # the lean physics-only mesh (the mod then only draws "rail"): its bed and decks
            shape = f"railcol_{aid}_{tx}_{ty}.cdae"
            cdae.write(os.path.join(out_dir, shape), col[0], col[1], uvs=col[2], material=col[3],
                       tri_materials=col[4], collision_only=True)
            data["railcol"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
    if signs:
        x0, y0, _, _ = tile_bbox(tx, ty)
        shape = f"signs_{aid}_{tx}_{ty}.cdae"
        signs[0][0].write(os.path.join(out_dir, shape))
        data["signs"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
        data["regionalSigns"] = signs[0][1]
    if walks:
        x0, y0, _, _ = tile_bbox(tx, ty)
        b = sidewalks.Builder((x0, y0, 0.0), collision=True)
        for wk in walks:
            if "corner" in wk:
                c = wk["corner"]
                b.add_corner(c["arc"], c["inward"], c["outer"], ramps=c.get("ramps", ()), europe=wk.get('europe', False))
                continue
            ww = wk.get("w", SIDEWALK["w"])
            curb, out = sidewalks.run_geometry(wk["pts"], wk["side"], ww)
            b.add_run(curb, out, ww, caps=wk["caps"], ramps=wk.get("ramps", ()),
                      slab=sidewalks.BRICK if wk.get("brick") else 1, europe=wk.get('europe', False))
        if not b.empty():
            shape = f"walks_{aid}_{tx}_{ty}.cdae"
            b.write(os.path.join(out_dir, shape))
            data["walk"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
            shape = f"walkcol_{aid}_{tx}_{ty}.cdae"     # the lean physics-only mesh (the drawn one then only renders)
            b.write_collision(os.path.join(out_dir, shape))
            data["walkcol"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
    boxes = collider_boxes(walks if data.get("walk") else None, chunks if data.get("deck") else None)
    if barrier_mesh is not None:
        import medianbarriers
        x0, y0, _, _ = tile_bbox(tx, ty)
        shape = shape_files(aid, tx, ty)["barrier"]
        roadmesh.write(os.path.join(out_dir, shape), barrier_mesh)
        data["barrier"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
        boxes["barrier"] = medianbarriers.collider_boxes(barrier_chunks)
    if "walk" in boxes and data.get("walkcol"):
        boxes["walkcol"] = boxes["walk"]
    if bld is None and bld_none:
        data["bldNone"] = True
    if bld is not None:             # buildings.road_tile: the tile's buildings, colliding ("bld")
        import buildings
        x0, y0, _, _ = tile_bbox(tx, ty)
        meshes = bld[0] if isinstance(bld[0], list) else [bld[0]]
        for k, mesh in enumerate(meshes[:BLD_PARTS_MAX]):   # under the engine's 16-bit index limit each
            what = "bld" if k == 0 else f"bld{k + 1}"
            shape = f"bld_{aid}_{tx}_{ty}.cdae" if k == 0 else f"bld_{aid}_{tx}_{ty}_{k + 1}.cdae"
            buildings.write(os.path.join(out_dir, shape), mesh)
            data[what] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
            boxes[what] = bld[1]
        if len(bld) > 2 and bld[2] is not None:     # the lean physics-only mesh (the drawn ones then only render)
            shape = f"bldcol_{aid}_{tx}_{ty}.cdae"
            buildings.write(os.path.join(out_dir, shape), bld[2], collision_only=True)
            data["bldcol"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
            boxes["bldcol"] = bld[1]
    if boxes:
        data["colliders"] = boxes
    if wires:                       # utility pole wires (furniture.wire_mesh), render-only
        import cdae
        import furniture
        x0, y0, _, _ = tile_bbox(tx, ty)
        wm = furniture.wire_mesh(wires, (x0, y0, 0.0))
        if wm is not None:
            shape = f"wire_{aid}_{tx}_{ty}.cdae"
            cdae.write(os.path.join(out_dir, shape), wm[0], wm[1], uvs=wm[2], material="bne_wire", object_name="wires")
            data["wire"] = {"shape": f"{vfs_dir}/{shape}", "pos": [x0, y0, 0.0]}
    if props:                       # furniture.place's rows: Forest items (trees.lua), groups by index
        groups = sorted({r[0] for r in props})
        data["props"] = {"groups": groups, "items": [[groups.index(r[0])] + r[1:] for r in props]}
        if towns:
            data["townsigns"] = {g: towns[g] for g in groups if g in towns}
    if park:                        # parking.py's spots [x, y, z, heading] for the stock parked cars (parking.lua)
        data["park"] = park
    if fuel:                        # gasstations.tile_sites' stations: pumps and refuel areas (gasstations.lua)
        data["fuel"] = fuel
    if custom_entries:              # buildings.road_tile's 3dmr.eu models (ours), [{shape, pos}]
        data["b3dmr"] = custom_entries

    tmp = staging.tmp_for(path)
    with open(tmp, "w") as f:
        f.write(json.dumps(data, separators=(",", ":")))    # json.dump to a file streams through the pure-Python encoder
    staging.replace(tmp, path)
    return f"{vfs_dir}/{name}", len(chunks)


def collider_boxes(walks, chunks):
    """{"walk": [...], "deck": [...]}: boxes [x0, y0, z0, x1, y1, z1] (anchor frame, m) around the
    sidewalk runs / corners and the bridge decks of a tile, in pieces of COLLIDER_PIECE_M. The mod
    does not make a colliding mesh solid while the player's vehicle is inside one of its boxes (a late
    curb or deck turning solid around a car; the intermittent NaN drives, 2026-10-13). Pure."""
    import sidewalks
    out = {}

    def add(what, pts, half_w, z_lo, z_hi):
        pts = np.asarray(pts, np.float64)
        seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts[:, :2], axis=0).T))] if len(pts) > 1 else np.zeros(1)
        cut = np.floor(seg / COLLIDER_PIECE_M).astype(int)
        for k in np.unique(cut):
            idx = np.flatnonzero(cut == k)
            sel = pts[max(idx[0] - 1, 0):idx[-1] + 1]     # the piece and the point before it
            m = half_w + COLLIDER_MARGIN_M
            out.setdefault(what, []).append([round(float(sel[:, 0].min() - m), 1), round(float(sel[:, 1].min() - m), 1),
                                             round(float(sel[:, 2].min() + z_lo), 1), round(float(sel[:, 0].max() + m), 1),
                                             round(float(sel[:, 1].max() + m), 1), round(float(sel[:, 2].max() + z_hi), 1)])
    for wk in walks or ():
        if "corner" in wk:
            c = wk["corner"]             # the curb arc and the outer corner points: one box over both
            pts = np.concatenate([np.asarray(c["arc"], np.float64), np.asarray(c["outer"], np.float64).reshape(-1, 3)])
            add("walk", pts[np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean()))],
                0.5, -1.0, 1.0)
            continue
        ww = wk.get("w", SIDEWALK["w"])
        curb, outv = sidewalks.run_geometry(wk["pts"], wk["side"], ww)
        mid = curb.copy()
        mid[:, :2] = curb[:, :2] + outv * (ww / 2)
        add("walk", mid, ww / 2 + 0.5, -1.0, 1.0)
    for c in chunks or ():
        if c.get("deck") and len(c["n"]) > 1:
            add("deck", c["n"], float(c["w"]) / 2 + 1.0, -roadmesh.DECK_DEPTH - 1.0, 2.0)
    return out
