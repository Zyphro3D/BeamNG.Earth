"""Buildings (v1.9; the user, 2026-10-13: "make the world look alive"): Overture footprints extruded into
walls with a window band per floor and a flat, gabled or hipped roof, one mesh per road tile (roads.TILE),
written next to the tile's road mesh and shown as a colliding TSStatic by the mod (roads.lua, "bld").

Overture rarely knows a building's height (Fells Point, 2026-10-13: 2 % of 12458; towns 50-80 %) and
almost never its roof shape or facade, so most of it is predicted from the footprint (area, narrowness,
class), deterministically per building (ROADMAP "deterministic generation": a hash of the footprint's
position, so it is the same building whichever way the player came).

Facades use the stock window sheets (asset survey 2026-10-13): m_highrise_plaster's four 256 px bands are
four floor styles (tall windows on plaster, square windows on panels, sash windows on plaster, sash windows
on red brick), m_highrise_metal's are office curtain wall and a shopfront; each floor of a wall is one band,
its windows a whole number of bays along every wall (no window cut at a corner). Pure functions of the
footprints and a ground-height function; no game state.
"""

import os
import math
import numpy as np
import fast
import landmarks
import mapgeom
import staging
import francehouses
from bldbase import (
    BRICK_WALK_COVERAGE,  # noqa: F401
    BRICK_WALK_SHARE,  # noqa: F401
    BRIDGE_SIDE_RGBA,  # noqa: F401
    BRIDGE_TOP_RGBA,  # noqa: F401
    DOME_ROOFS,  # noqa: F401
    FAR_LAMP_MAX,  # noqa: F401
    FAR_LAMP_MIN_BLD,  # noqa: F401
    FAR_LAMP_Z,  # noqa: F401
    FAR_LAYERS,  # noqa: F401
    FAR_NEED,  # noqa: F401
    FLOOR_M,  # noqa: F401
    JITTER,  # noqa: F401
    LOOKS,  # noqa: F401
    MARBLE,  # noqa: F401
    MATERIALS,  # noqa: F401
    MAT_MEAN,  # noqa: F401
    MAX_MESH_VERTS,  # noqa: F401
    PHOTO_FILL_M,  # noqa: F401
    PHOTO_GROW_M,  # noqa: F401
    PHOTO_LEAN,  # noqa: F401
    PHOTO_LEAN_MAX_M,  # noqa: F401
    ROOF_FLAT,  # noqa: F401
    SKIRT_LITE_M,  # noqa: F401
    SKY_BAY_M,  # noqa: F401
    SKY_BOX_AREA_M2,  # noqa: F401
    SKY_MAX_CORNERS,  # noqa: F401
    SKY_MIN_AREA_M2,  # noqa: F401
    SKY_UNIT,  # noqa: F401
    SKY_UV,  # noqa: F401
    SKY_WALL,  # noqa: F401
    SKY_WIN_BAYS,  # noqa: F401
    SKY_WIN_FLOORS,  # noqa: F401
    _area2,  # noqa: F401
    _hash01,  # noqa: F401
    oriented_box,  # noqa: F401
)
from bldplan import (
    _landmark_token,  # noqa: F401
    _landmark_tokens,  # noqa: F401
    brick_share,  # noqa: F401
    roof_pitch,  # noqa: F401
)
from bldmesh import (
    Mesh,  # noqa: F401
    _dome,  # noqa: F401
    base_z,  # noqa: F401
    build,  # noqa: F401
    build_parts,  # noqa: F401
    collision_mesh,  # noqa: F401
)
from bldsite import (
    bridge_lines,  # noqa: F401
    garage_rings,  # noqa: F401
    road_lines,  # noqa: F401
    support_rings,  # noqa: F401
    water_rings,  # noqa: F401
    way_lines,  # noqa: F401
)
from bldtile import (
    collider_boxes,  # noqa: F401
    tile_buildings,  # noqa: F401
    write,  # noqa: F401
)
from blddoors import (
    _suppress_under_models,  # noqa: F401
    doors,  # noqa: F401
)
from bldcustom import (
    coverage,  # noqa: F401
    custom_models,  # noqa: F401
    district_town,  # noqa: F401
    road_tile_items,  # noqa: F401
)
from bldbase import (  # (re-exported: callers use buildings.<name>)
    ACROSS_W,  # noqa: F401
    ALPINE_COVER,  # noqa: F401
    ALPINE_HUT_M2,  # noqa: F401
    BACK_ROAD_CLASSES,  # noqa: F401
    BANDS,  # noqa: F401
    BAND_BRICK,  # noqa: F401
    BAND_OFFICE,  # noqa: F401
    BAND_OFFICE2,  # noqa: F401
    BAND_PANEL,  # noqa: F401
    BAND_SASH,  # noqa: F401
    BAND_SHOP,  # noqa: F401
    BAND_TALL,  # noqa: F401
    BARN_CLASSES,  # noqa: F401
    BARN_REDS,  # noqa: F401
    BIG_BOX_M2,  # noqa: F401
    BLD_LOD_PX,  # noqa: F401
    BRICK_TINTS,  # noqa: F401
    BRICK_WALL,  # noqa: F401
    BRIDGE_ASPECT,  # noqa: F401
    BRIDGE_CLASSES,  # noqa: F401
    BRIDGE_FAR_DROP_M,  # noqa: F401
    BRIDGE_FAR_PER_POSE,  # noqa: F401
    BRIDGE_FAR_SEND_M,  # noqa: F401
    BRIDGE_MIN_SPAN_M,  # noqa: F401
    BRIDGE_PIER_M,  # noqa: F401
    CANOPY_CLEAR_M,  # noqa: F401
    CANOPY_POSTS_MAX_M,  # noqa: F401
    CANOPY_POST_HALF_M,  # noqa: F401
    CANOPY_POST_INSET_M,  # noqa: F401
    CANOPY_POST_STEP_M,  # noqa: F401
    CANOPY_SLAB_M,  # noqa: F401
    CENTRE_PREFILTER_M,  # noqa: F401
    CHURCH_CLASSES,  # noqa: F401
    COLOR_NAMES,  # noqa: F401
    CONCRETE,  # noqa: F401
    CORNICE_DOWN_M,  # noqa: F401
    CORNICE_OUT_M,  # noqa: F401
    CORNICE_UP_M,  # noqa: F401
    DECK_COLUMN_M,  # noqa: F401
    DECK_EDGE_M,  # noqa: F401
    DECK_HEADROOM_M,  # noqa: F401
    DECK_INSIDE_M,  # noqa: F401
    DECK_LANDING_M,  # noqa: F401
    DECK_LANE_M,  # noqa: F401
    DECK_LEVEL_M,  # noqa: F401
    DECK_PARAPET_M,  # noqa: F401
    DECK_RAMP_EDGES,  # noqa: F401
    DECK_REACH_M,  # noqa: F401
    DECK_SEA_M,  # noqa: F401
    DECK_SLAB_M,  # noqa: F401
    DECK_SLOPE,  # noqa: F401
    DECK_STAND_M,  # noqa: F401
    DECK_STOREY_M,  # noqa: F401
    DISTRICT_M,  # noqa: F401
    DOME_MAX_HW,  # noqa: F401
    DOME_PROFILE,  # noqa: F401
    DOME_RISE,  # noqa: F401
    DOOR_CLASSES_SKIP,  # noqa: F401
    DOOR_GAP_M,  # noqa: F401
    DOOR_OUT_M,  # noqa: F401
    DOOR_ROAD_M,  # noqa: F401
    DRIVEWAY,  # noqa: F401
    DRIVEWAY_W,  # noqa: F401
    EAVE_M,  # noqa: F401
    ELEVATOR_CLASSES,  # noqa: F401
    ELEVATOR_MAX_M2,  # noqa: F401
    ELEVATOR_MIN_H,  # noqa: F401
    ELEVATOR_SLENDER,  # noqa: F401
    FACADE,  # noqa: F401
    FACADE_TINTS,  # noqa: F401
    FALSE_FRONT_M,  # noqa: F401
    FAR_DROP_M,  # noqa: F401
    FAR_GEO,  # noqa: F401
    FAR_PER_POSE,  # noqa: F401
    FAR_SEND_M,  # noqa: F401
    FENCE_BACK_M,  # noqa: F401
    FENCE_CLEAR_M,  # noqa: F401
    FENCE_MIN_YARD_M,  # noqa: F401
    FENCE_SEG_M,  # noqa: F401
    FENCE_SHARE,  # noqa: F401
    FLAT_FOOTPRINT_M,  # noqa: F401
    FOUNDATION_M,  # noqa: F401
    GAMBREL_BREAK,  # noqa: F401
    GAMBREL_PITCHES,  # noqa: F401
    GARAGE_DRIVE_MAX_M,  # noqa: F401
    GARAGE_DRIVE_W,  # noqa: F401
    GARAGE_MIN_M2,  # noqa: F401
    GARAGE_SKINS,  # noqa: F401
    GARAGE_TAGS,  # noqa: F401
    GARAGE_TAG_CLASSES,  # noqa: F401
    GLASS_FLOORS,  # noqa: F401
    GROUND_FLOOR_SHOP_M,  # noqa: F401
    G_DOOR,  # noqa: F401
    G_FENCE,  # noqa: F401
    G_TOLLBOOTH,  # noqa: F401
    HALF_HIP,  # noqa: F401
    HOUSE_BAY_M,  # noqa: F401
    HOUSE_CLASSES,  # noqa: F401
    INDUSTRIAL_CLASSES,  # noqa: F401
    LANDMARK_DIAL,  # noqa: F401
    LANDMARK_FLOODLIT,  # noqa: F401
    LANDMARK_IRON,  # noqa: F401
    LANDMARK_LAMP,  # noqa: F401
    LANDMARK_MATERIALS,  # noqa: F401
    LANDMARK_STONE,  # noqa: F401
    LEVEL_THROUGH_M,  # noqa: F401
    MAIN_STREET_CLASSES,  # noqa: F401
    MAIN_STREET_SETBACK_M,  # noqa: F401
    MAIN_STREET_SHOPS,  # noqa: F401
    MANSARD_INSET_M,  # noqa: F401
    MANSARD_PITCHES,  # noqa: F401
    MARBLE_LOOK,  # noqa: F401
    MASONRY_MATERIALS,  # noqa: F401
    MAX_VERTS,  # noqa: F401
    METALWALL,  # noqa: F401
    MIDRISE_GLASS_OFFICE,  # noqa: F401
    MIN_AREA_M2,  # noqa: F401
    ML_BRIDGE_M,  # noqa: F401
    ML_CLIFF_DEG,  # noqa: F401
    ML_FAR_FROM_WAY_M,  # noqa: F401
    ML_MEDIAN_M,  # noqa: F401
    ML_ROAD_EDGE_M,  # noqa: F401
    ML_ROAD_M,  # noqa: F401
    ML_STEEP_DEG,  # noqa: F401
    MODEL_MAX_M,  # noqa: F401
    OFFICE,  # noqa: F401
    OLD_TOWER_GLASS,  # noqa: F401
    OPEN_CANOPY_CLASSES,  # noqa: F401
    OVER_ROAD_CLEAR_M,  # noqa: F401
    OVER_TUNNEL_CLEAR_M,  # noqa: F401
    PAINT_TINTS,  # noqa: F401
    PANEL,  # noqa: F401
    PARTS_COVER_MIN,  # noqa: F401
    PARTS_COVER_MIN_H,  # noqa: F401
    PARTS_COVER_ON,  # noqa: F401
    PARTS_COVER_SAMPLES,  # noqa: F401
    PARTS_GROUND_M,  # noqa: F401
    PASSAGE_BOTTOM_M,  # noqa: F401
    PASSAGE_CLEAR,  # noqa: F401
    PASSAGE_H,  # noqa: F401
    PASSAGE_W,  # noqa: F401
    PIER_EVERY_M,  # noqa: F401
    PIER_W,  # noqa: F401
    PITCH,  # noqa: F401
    PLAIN_STRIP,  # noqa: F401
    POLES_COVERAGE,  # noqa: F401
    PORCH_DECK_M,  # noqa: F401
    PORCH_DEPTH_M,  # noqa: F401
    PORCH_H_M,  # noqa: F401
    PORTAL_NEAR_M,  # noqa: F401
    PORTAL_SKIP_CLASSES,  # noqa: F401
    PYRAMID_MIN_M2,  # noqa: F401
    PYRAMID_PLINTH_M,  # noqa: F401
    PYRAMID_ROOF_SHARE,  # noqa: F401
    RESTAURANT_CLASSES,  # noqa: F401
    ROAD_THROUGH_M,  # noqa: F401
    ROOF_CLUTTER_M2,  # noqa: F401
    ROOF_METAL,  # noqa: F401
    ROOF_SHINGLE,  # noqa: F401
    ROOF_SLATE,  # noqa: F401
    ROOF_TINTS,  # noqa: F401
    ROOF_UV_M,  # noqa: F401
    ROW_CLASSES,  # noqa: F401
    SHED_CLASSES,  # noqa: F401
    SHOP,  # noqa: F401
    SHOP_CLASSES,  # noqa: F401
    SIDING,  # noqa: F401
    SIDING_UV_M,  # noqa: F401
    SIGN_H,  # noqa: F401
    SIGN_OUT,  # noqa: F401
    SIGN_SHOP_M,  # noqa: F401
    SIGN_TINTS,  # noqa: F401
    SIMPLIFY_M,  # noqa: F401
    SITE_M2,  # noqa: F401
    SKILLION_PITCH,  # noqa: F401
    SKIRT_M,  # noqa: F401
    SKY_DROP_M,  # noqa: F401
    SKY_PER_POSE,  # noqa: F401
    SKY_SEND_M,  # noqa: F401
    SPANDREL_H,  # noqa: F401
    SPANDREL_T,  # noqa: F401
    STAIR_TOWER_M,  # noqa: F401
    STAIR_TOWER_UP,  # noqa: F401
    STEEL,  # noqa: F401
    STEEL_BAY_M,  # noqa: F401
    STEEL_LEG_MAX,  # noqa: F401
    STEEL_MATERIALS,  # noqa: F401
    STEEL_TOP_SHARE,  # noqa: F401
    STONE,  # noqa: F401
    STONE_FACADES,  # noqa: F401
    STONE_FLOORS,  # noqa: F401
    STOOP_H,  # noqa: F401
    STOOP_LANDING,  # noqa: F401
    STOOP_MAX_STEPS,  # noqa: F401
    STOOP_RISE,  # noqa: F401
    STOOP_RISE_MAX,  # noqa: F401
    STOOP_SHARE,  # noqa: F401
    STOOP_STEPS,  # noqa: F401
    STOOP_TREAD,  # noqa: F401
    STOOP_W,  # noqa: F401
    STUCCO,  # noqa: F401
    SUPPORT_DZ_WARN_M,  # noqa: F401
    SUPPRESS_MARGIN_M,  # noqa: F401
    SUPPRESS_MAX_M,  # noqa: F401
    SUPPRESS_MIN_M,  # noqa: F401
    THROUGH_CLASSES,  # noqa: F401
    TILE_PX_M,  # noqa: F401
    TOLLBOOTH_H,  # noqa: F401
    TOLL_BOOTH_H,  # noqa: F401
    TOLL_BOOTH_MAX_AREA,  # noqa: F401
    TOLL_ISLAND_MAX_H,  # noqa: F401
    TOLL_ISLAND_MAX_W,  # noqa: F401
    TOLL_ISLAND_MIN_L,  # noqa: F401
    TOLL_ROAD_DEG,  # noqa: F401
    TOLL_ROAD_M,  # noqa: F401
    TOWER_SHOP_FLOORS,  # noqa: F401
    TOWN_COVERAGE,  # noqa: F401
    TUNNEL_THROUGH_M,  # noqa: F401
    US_WALL_TINTS,  # noqa: F401
    WALL_SIGNS,  # noqa: F401
    WALL_SIGN_SHARE,  # noqa: F401
    WALL_TINTS,  # noqa: F401
    WEST_TOWN_COVERAGE,  # noqa: F401
    WEST_TOWN_ONE_STOREY,  # noqa: F401
    WIN_H,  # noqa: F401
    WIN_OUT_M,  # noqa: F401
    WIN_SILL,  # noqa: F401
    WIN_UV,  # noqa: F401
    WIN_W,  # noqa: F401
    WOOD_TINTS,  # noqa: F401
    _Looks,  # noqa: F401
    _c,  # noqa: F401
    _cs,  # noqa: F401
    _k,  # noqa: F401
    _m,  # noqa: F401
    clean_ring,  # noqa: F401
    custom_look,  # noqa: F401
    parse_color,  # noqa: F401
    triangulate,  # noqa: F401
)
from bldplan import (  # (re-exported: callers use buildings.<name>)
    STYLES,  # noqa: F401
    Spec,  # noqa: F401
    Style,  # noqa: F401
    _CLAY,  # noqa: F401
    _DARK,  # noqa: F401
    _US_ROOFS,  # noqa: F401
    dome_rise,  # noqa: F401
    landmark_spec,  # noqa: F401
    plan,  # noqa: F401
    region_style,  # noqa: F401
    roof_rise,  # noqa: F401
    tree_line_m,  # noqa: F401
)
from bldmesh import (  # (re-exported: callers use buildings.<name>)
    _WALL_CONSTS,  # noqa: F401
    _box,  # noqa: F401
    _box_tiled,  # noqa: F401
    _canopy_posts,  # noqa: F401
    _clutter,  # noqa: F401
    _concat,  # noqa: F401
    _cornice,  # noqa: F401
    _deck_layout,  # noqa: F401
    _deck_wall,  # noqa: F401
    _garage_skin,  # noqa: F401
    _inside,  # noqa: F401
    _parking_deck,  # noqa: F401
    _porch,  # noqa: F401
    _portal_lift,  # noqa: F401
    _roof,  # noqa: F401
    _signs,  # noqa: F401
    _steel_frame,  # noqa: F401
    _steeple,  # noqa: F401
    _stoop,  # noqa: F401
    _walls,  # noqa: F401
    deck_plan,  # noqa: F401
    door_ground,  # noqa: F401
    drawn_ring,  # noqa: F401
    garage_drives,  # noqa: F401
)
from bldsite import (  # (re-exported: callers use buildings.<name>)
    RoadLines,  # noqa: F401
    _BOXES,  # noqa: F401
    _PartsOf,  # noqa: F401
    _clip_half,  # noqa: F401
    _covers_garage,  # noqa: F401
    _hulls_nearby,  # noqa: F401
    _in_median,  # noqa: F401
    _line_dist,  # noqa: F401
    _ml_false,  # noqa: F401
    _ml_steep,  # noqa: F401
    _nearest_on,  # noqa: F401
    _on_lanes,  # noqa: F401
    _on_support,  # noqa: F401
    _outline_under_parts,  # noqa: F401
    _part_heights,  # noqa: F401
    _parts_index,  # noqa: F401
    _parts_tops,  # noqa: F401
    _passages,  # noqa: F401
    _shared_walls,  # noqa: F401
    line_boxes,  # noqa: F401
    on_deck,  # noqa: F401
    parts_cover,  # noqa: F401
    road_across,  # noqa: F401
    road_through,  # noqa: F401
    tunnel_lines,  # noqa: F401
)
from bldtile import (  # (re-exported: callers use buildings.<name>)
    _on_box,  # noqa: F401
    _toll_islands,  # noqa: F401
    _toll_mark,  # noqa: F401
    garage_sound_boxes,  # noqa: F401
    toll_booth_rows,  # noqa: F401
)
from blddoors import (  # (re-exported: callers use buildings.<name>)
    _front_fence,  # noqa: F401
    _sign_in_open,  # noqa: F401
    _wall_sign,  # noqa: F401
)
from bldcustom import (  # (re-exported: callers use buildings.<name>)
    _cell_lock,  # noqa: F401
    _cell_lock_get,  # noqa: F401
    _cell_stats,  # noqa: F401
    _centred,  # noqa: F401
    _custom_tile_entries,  # noqa: F401
    _is_bridge,  # noqa: F401
    _model_mesh,  # noqa: F401
    _mostly_under,  # noqa: F401
    _section_mat_tex,  # noqa: F401
    _tex_to_png,  # noqa: F401
    _under_bridges,  # noqa: F401
    bridge_footprints,  # noqa: F401
    bridge_road_ends,  # noqa: F401
    bridge_stub_chunks,  # noqa: F401
    bridge_stub_parts,  # noqa: F401
    bridge_stub_piers,  # noqa: F401
    bridge_stub_props,  # noqa: F401
    cell_footprints,  # noqa: F401
    centres,  # noqa: F401
    district_coverage,  # noqa: F401
    near_rings,  # noqa: F401
)


def road_tile(
    frame,
    tx,
    ty,
    md,
    ground,
    lines=None,
    props=None,
    items=None,
    custom=None,
    out_dir=None,
    vfs_dir=None,
    entries=None,
    radii=None,
):
    """The buildings of road tile (tx, ty) (roads.TILE squares of the anchor frame): ((mesh, collider boxes,
    collision mesh), [b3dmr entries]). The first element is None without any procedural buildings. md:
    mapdata.MapData; ground(lat, lon): the physics ground without the road lift. lines (the road centrelines,
    roads.build_tile) and props (its prop rows): the houses' front doors are added to props, after custom's
    models (if any) have suppressed the footprints under them (_suppress_under_models) - a door on a stub
    that no longer builds would be a door on nothing. items, custom: road_tile_items' result (items, custom)
    when the caller has it (else fetched here). out_dir, vfs_dir: the session folder and its game path, to
    bake custom's 3dmr.eu models into (_custom_tile_entries); without them (or without custom footprints) the
    second element is [] and nothing is suppressed. entries, radii: custom_models()' result when the caller
    already baked the models (before roads.build_tile) - baked here instead when neither is given."""
    import os
    import roads

    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    if items is None:
        cache_dir = os.path.dirname(out_dir) if out_dir else None
        items, custom = road_tile_items(
            frame, tx, ty, md, ground, cache_dir=cache_dir
        )
    if entries is None and radii is None:
        entries, radii, _boxes = custom_models(
            frame, custom, ground, out_dir, vfs_dir
        )
    if radii:
        items = _suppress_under_models(items, radii, x0, y0)
    if props is not None and lines:
        props.extend(doors(items, lines, x0, y0))
    meshes = build_parts(items, with_lod=True)
    bld = (
        (meshes, collider_boxes(items, x0, y0), collision_mesh(items))
        if meshes
        else None
    )
    return bld, entries


def far_needs(frame, tile, ty):
    """The map cells and layers far_items reads for road tile (tile, ty) (blockworker: fetched together first)."""
    import roads
    box = roads.tile_bbox(tile, ty)
    house_layers = ("land_use",) if francehouses.needs_residential_land(frame, box) else ()
    return list(roads.cells_of_tile(frame, tile, ty, margin=0.0)), FAR_LAYERS + house_layers


def far_items(frame, tile, ty, md, ground, lamp_region=None):
    """The far (lite) buildings of road tile (tile, ty) (an anchor tile, or a geotile: tile a geotiles.GeoBox, ty None:
    its own buildings, those the road geotile of the same key holds) in frame: (items, lamps [[x, y, z], ...] of its
    streets in frame, the tile's box)."""
    import roads
    box = roads.tile_bbox(tile, ty)
    x0, y0, x1, y1 = box
    own = box if hasattr(box, "owns") else None
    keys, layers = far_needs(frame, tile, ty)
    cds = [md.cell(k, layers=layers, kind="demand") for k in keys]
    items = tile_buildings(frame, x0, y0, x1, y1, [c.layers.get("buildings") for c in cds], ground, own=own,
                           part_layers=[c.layers.get("building_parts") for c in cds],
                           house_land_layers=[c.layers.get("land_use") for c in cds],
                           west_town=district_town(frame, tile, ty, md),
                           garages_xy=garage_rings(frame, [c.layers.get("infrastructure") for c in cds], x0, y0),
                           supports=support_rings(frame, [c.layers.get("infrastructure") for c in cds], x0, y0),
                           bridges_xy=bridge_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
                           water_xy=water_rings(frame, [c.layers.get("water") for c in cds], x0, y0),
                           ml_roads_xy=road_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
                           ways_xy=way_lines(frame, [c.layers.get("roads") for c in cds], x0, y0, size=x1 - x0))
    size = max(x1 - x0, y1 - y0)
    lamps = far_lamps(frame, [c.layers.get("roads") for c in cds], x0, y0, size,
                      built_squares(items, 0.0, 0.0, size), ground, lamp_region, own=own)
    return items, lamps, box


def far_tile(frame, tx, ty, md, ground, out_dir, vfs_dir, aid, lamp_region=None):
    """A far (render-only) building tile written to the session folder: (vfs path of its JSON, building
    count) or None. The lite mesh of the same buildings as road_tile's."""
    import json
    import os
    items, lamps, (x0, y0, _x1, _y1) = far_items(frame, tx, ty, md, ground, lamp_region)
    mesh = build(items, lite=True)
    data = {"v": 1}
    if lamps:
        data["lamps"] = lamps
    if mesh is not None:
        shape = f"bldfar_{aid}_{tx}_{ty}.cdae"
        write(os.path.join(out_dir, shape), mesh)
        data["shape"] = f"{vfs_dir}/{shape}"
        data["pos"] = [x0, y0, 0.0]
    name = f"bldfar_{aid}_{tx}_{ty}.json"
    path = os.path.join(out_dir, name)
    tmp = staging.tmp_for(path)
    with open(tmp, "w") as f:
        f.write(json.dumps(data))       # (json.dump to a file: the pure-Python encoder)
    staging.replace(tmp, path)
    return f"{vfs_dir}/{name}", len(items)


def built_squares(items, x0, y0, size, sq=256.0):
    """The set of (i, j) 256 m squares of a tile (x0, y0 its corner; items' rings tile-relative) holding at least
    FAR_LAMP_MIN_BLD buildings."""
    import collections
    n = collections.Counter()
    for spec, _gmin, _gmax in items:
        c = np.asarray(spec.ring, np.float64)[:, :2].mean(0)
        n[(int(c[0] // sq), int(c[1] // sq))] += 1
    return {k for k, v in n.items() if v >= FAR_LAMP_MIN_BLD}


def far_lamps(frame, road_layers, x0, y0, size, built, ground, region=None, sq=256.0, own=None):
    """The lamps of a far tile's streets: [x, y, z] in frame (rounded; z the frame's up at FAR_LAMP_Z over the ground),
    at most FAR_LAMP_MAX. built: the tile's built-up 256 m squares (built_squares); ground(lat, lon) the ground
    (arrays); region: furniture.lamp_region's (the spacing); own: a geotile's box (geotiles.OwnedBox): only its own
    lamps (each in one tile)."""
    import furniture
    import roads as roadsmod
    if not built:
        return []
    table = furniture.lamp_spacing(region)
    hw = "na" if furniture._na(region) else None
    pts, off, spacing, phase, side, lit_any = [], [0], [], [], [], []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        specs = roadsmod.road_specs(f)
        bb = f.boxes(frame)
        near = np.flatnonzero((bb[:, 2] - x0 >= -20) & (bb[:, 0] - x0 <= size + 20)
                              & (bb[:, 3] - y0 >= -20) & (bb[:, 1] - y0 <= size + 20))
        for i in near.tolist():
            sp = specs[i]
            if sp is None:
                continue
            highway = sp.cls in furniture.HIGHWAY_CLASSES
            if highway and not furniture.highway_lit(region, sp.link, True):
                continue
            if highway:
                every = (furniture.LINK_LAMP_SPACING if sp.link else furniture.HIGHWAY_LAMP_SPACING)[hw]
                rules = [(every, -float(sp.dir) if sp.dir else (1.0 if furniture._h01(str(f.ids[i]), "pside") < 0.5
                                                               else -1.0), 0.0)]
            elif sp.cls in table:
                every, both = table[sp.cls]
                sd = 1.0 if furniture._h01(str(f.ids[i]), "side") < 0.5 else -1.0
                rules = [(every, sd, 0.0)] + ([(every, -sd, every / 2)] if both else [])
            else:
                continue
            ph0 = furniture._h01(str(f.ids[i]), "pphase")
            reach = max(1.0, sp.width / 2 - 1.0)
            parts = roadsmod.ground_parts(f, i) if roadsmod.level_ranges(f, i) else [None]
            for part in parts:
                if part is None:
                    a = f.coord_off[f.ring_off[f.part_off[i]]]
                    b = f.coord_off[f.ring_off[f.part_off[i + 1]]]
                    part = f.lonlat[a:b]
                if len(part) < 2:
                    continue
                e, n = mapgeom.enu_xy(f, frame, np.asarray(part, np.float64))
                xy = np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64)], 1)
                for every, sd, shift in rules:
                    pts.append(xy)
                    off.append(off[-1] + len(xy))
                    spacing.append(every)
                    phase.append(ph0 * every + shift)
                    side.append(sd * reach)
                    lit_any.append(bool(highway and sp.link))
    if not pts:
        return []
    fast.require(87, "lamp_stations")
    st = fast._ef.lamp_stations(np.ascontiguousarray(np.concatenate(pts)), np.asarray(off, np.int64),
                                np.asarray(spacing, np.float64), np.asarray(phase, np.float64),
                                np.asarray(side, np.float64), [x0, y0, x0 + size, y0 + size])
    if not len(st):
        return []
    keep = [k for k in range(len(st)) if lit_any[int(st[k, 2])]
            or (int((st[k, 0] - x0) // sq), int((st[k, 1] - y0) // sq)) in built]
    st = st[keep]
    if own is not None and len(st):
        st = st[np.asarray(own.owns(st[:, 0], st[:, 1]), bool)]
    st = st[:FAR_LAMP_MAX]
    if not len(st):
        return []
    lat, lon, _ = frame.enu_to_geodetic(st[:, 0], st[:, 1], 0.0)
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    h = np.asarray(ground(lat, lon), np.float64) + FAR_LAMP_Z
    z = np.asarray(frame.geodetic_to_enu(lat, lon, h)[2], np.float64)     # (the frame's up: the Earth's curvature)
    return np.round(np.c_[st[:, 0], st[:, 1], z], 1).tolist()


def _sky_colour(look, f):
    """RGBA bytes of a LOOKS name drawn plain: its material's mean colour x its tint x the jitter f."""
    if look not in LOOKS:
        return (150, 150, 150, 255)
    mat, tint = LOOKS[look]
    mean = MAT_MEAN.get(mat, (0.7, 0.7, 0.7))
    return tuple(int(max(0, min(255, round(255 * m * t * f)))) for m, t in zip(mean, tint)) + (255,)


def _sky_mesh(items):
    """The skyline mesh of [(Spec, gmin, gmax)]: per building a quad per wall and its flat roof, in MARBLE at one
    texel, coloured per vertex (~4 per corner + the roof's corners: a row house is 20 vertices, the lite mesh ~72).
    (positions, uvs, tris, tri material indices, MATERIALS, colours) as build, or None."""
    m = Mesh()
    mi = MATERIALS.index(MARBLE)
    ms = MATERIALS.index(SKY_WALL)
    uv = SKY_UV
    for spec, gmin, gmax in items:
        if spec.cls == landmarks.LANDMARK:      # (its far-away model, not a box of its footprint's size)
            landmarks.build(m, spec, gmin, _landmark_token(spec), lite=True, tokens=_landmark_tokens())
            continue
        ring = np.asarray(spec.ring, np.float64)[:, :2]
        if len(ring) < 3:
            continue
        if _area2(ring) < 0:
            ring = ring[::-1]
        f = 1.0 + JITTER * (2.0 * _hash01(spec.seed, "jit") - 1.0)
        windows = spec.windows is not False
        wall = (ms if windows else mi, _sky_colour(spec.facade, f))
        roof = (mi, _sky_colour(spec.roof_mat or ROOF_FLAT, f))
        z0 = gmin - SKIRT_LITE_M
        z1 = base_z(spec, gmin, gmax) + float(spec.min_h or 0.0) + float(spec.height or 0.0)
        if z1 <= z0 + 0.5:
            continue
        n = len(ring)
        # (windows: u along the walls in bays, v up in storeys from the ground, both over the map's count)
        zb = gmin
        fh = float(spec.floor_h or FLOOR_M) * SKY_WIN_FLOORS
        bw = SKY_BAY_M * SKY_WIN_BAYS
        v0, v1 = (z0 - zb) / fh, (z1 - zb) / fh
        u = 0.0
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            if windows:
                du = float(math.hypot(*(b - a))) / bw
                m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1),
                       (u, v0), (u + du, v0), (u + du, v1), (u, v1), wall)
                u += du
            else:
                m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1), uv, uv, uv, uv, wall)
        m.poly([(p[0], p[1], z1) for p in ring], [uv] * n, roof)
        if spec.roof in DOME_ROOFS:     # (a mapped dome: far away too, on its drum)
            _dome(m, ring, z1, oriented_box(ring)[3] * math.tan(math.radians(roof_pitch(spec))), roof, spec.roof,
                  lite=True)
    if not m.n:
        return None
    P, UV, T, M, C = m.arrays()
    return P, UV, T, M, MATERIALS, C


def sky_items(frame, box, keys, md, ground, lamp_region=None, own=None):
    """The skyline's buildings of box (x0, y0, x1, y1) in frame over map cells keys: (picked items, lamps), or None when
    the cells are not at hand (fetched in the background: asked again later). own: a geotile's box (its own buildings
    and lamps, each in one tile)."""
    x0, y0, x1, y1 = box
    # (and the roads: their street lamps, far_lamps)
    if not all(md.has(k, "buildings") and md.has(k, "roads") for k in keys):
        # (the background lane: never before what the player is near; in the map data's own threads, all at once: one
        # after another in the one far process they held the far buildings back, 2026-11-07)
        md.warm([(k, layer) for k in keys for layer in ("buildings", "roads")], kind="prefetch")
        return None
    cds = [md.cell(k, layers=("buildings", "roads"), kind="background") for k in keys]
    layers = [c.layers.get("buildings") for c in cds]
    items = tile_buildings(frame, x0, y0, x1, y1, layers, ground, min_size=(1e9, SKY_MIN_AREA_M2), own=own)
    size = max(x1 - x0, y1 - y0)
    lamps = far_lamps(frame, [c.layers.get("roads") for c in cds], x0, y0, size,
                      built_squares(items, 0.0, 0.0, size), ground, lamp_region, own=own)
    for spec, _gmin, _gmax in items:        # (the simplest shape: a box or a few corners, a flat roof, nothing on it)
        ring = np.asarray(spec.ring, np.float64)
        if len(ring) > 4 and abs(_area2(ring)) / 2 < SKY_BOX_AREA_M2:
            c, u, hl, hw = oriented_box(ring)
            v = np.array([-u[1], u[0]])
            ring = np.array([c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw, c - u * hl + v * hw])
        elif len(ring) > SKY_MAX_CORNERS:     # (rings are open: the last corner is not the first again)
            ring = ring[::int(np.ceil(len(ring) / SKY_MAX_CORNERS))]
        spec.ring = ring
        spec.roof, spec.steeple, spec.stoop, spec.porch, spec.sign, spec.door = "flat", None, None, None, None, None
    items.sort(key=lambda it: (-(float(it[0].height or 0.0) + float(it[0].min_h or 0.0)), str(it[0].seed)))
    picked, n = [], 0
    for it in items:                        # (the tallest first, up to the 16-bit vertex budget: 5 per corner)
        k = 5 * len(it[0].ring)
        if n + k > MAX_MESH_VERTS:
            break
        picked.append(it)
        n += k
    return picked, lamps


def skyline_tile(frame, ux, uy, md, ground, out_dir, vfs_dir, aid, lamp_region=None):
    """A skyline tile (render-only) written to the session folder: (vfs path of its JSON, building count) or None
    when its cells are not at hand (fetched in the background: asked again later). Key (ux, uy): SKY_UNIT x SKY_UNIT
    road tiles from road tile (ux * SKY_UNIT, uy * SKY_UNIT)."""
    import json
    import roads
    x0, y0, _, _ = roads.tile_bbox(ux * SKY_UNIT, uy * SKY_UNIT)
    x1, y1 = x0 + SKY_UNIT * roads.TILE, y0 + SKY_UNIT * roads.TILE
    keys = sorted({k for i in range(SKY_UNIT) for j in range(SKY_UNIT)
                   for k in roads.cells_of_tile(frame, ux * SKY_UNIT + i, uy * SKY_UNIT + j, margin=0.0)}, key=str)
    got = sky_items(frame, (x0, y0, x1, y1), keys, md, ground, lamp_region)
    if got is None:
        return None
    picked, lamps = got
    mesh = _sky_mesh(picked) if picked else None
    data = {"v": 1}
    if mesh is not None:
        shape = f"bldsky_{aid}_{ux}_{uy}.cdae"
        if lamps:
            data["lamps"] = lamps
        write(os.path.join(out_dir, shape), mesh)
        data["shape"] = f"{vfs_dir}/{shape}"
        data["pos"] = [x0, y0, 0.0]
    name = f"bldsky_{aid}_{ux}_{uy}.json"
    path = os.path.join(out_dir, name)
    tmp = staging.tmp_for(path)
    with open(tmp, "w") as f:
        f.write(json.dumps(data))
    staging.replace(tmp, path)
    return f"{vfs_dir}/{name}", len(picked) if mesh is not None else 0


def bridge_slabs(parts, origin, box=None, owners=None):
    """The slab meshes of the bridge parts [(xy (k, 2), z (k,), width m)] (in one frame), relative to origin (x, y),
    by owner (earthfast.bridge_slabs): box (x0, y0, x1, y1): the segments whose midpoint is in [x0, x1) x [y0, y1), one
    mesh under the key None; owners (frame, cz, pz, px, py): those whose midpoint's place lies in a zoom-cz geotile
    under the zoom-pz one (px, py), a mesh per geotile {(gx, gy): mesh}. Each (positions, uvs, tris, tri materials,
    MATERIALS, colours) as build: the deck's top, its sides and its underside down to its girders, plain grey."""
    import roadmesh
    parts = [(np.asarray(xy, np.float64)[:, :2], np.asarray(z, np.float64), float(w)) for xy, z, w in parts]
    parts = [p for p in parts if len(p[0]) >= 2]
    if not parts:
        return {}
    fast.require(FAR_NEED, "bridge_slabs")
    xy = np.ascontiguousarray(np.concatenate([p[0] for p in parts]))
    z = np.ascontiguousarray(np.concatenate([p[1] for p in parts]))
    off = np.r_[0, np.cumsum([len(p[0]) for p in parts])].astype(np.int64)
    w = np.array([p[2] for p in parts], np.float64)
    if owners is None:
        lat0 = lon0 = 0.0
        cz = pz = px = py = 0
        bb = tuple(float(v) for v in box)
    else:
        frame, cz, pz, px, py = owners
        lat0, lon0, bb = float(frame.lat0), float(frame.lon0), (0.0, 0.0, 0.0, 0.0)
    got = fast._ef.bridge_slabs(xy, z, off, w, (float(origin[0]), float(origin[1])),
                                float(roadmesh.DECK_DEPTH + roadmesh.GIRDER_D), BRIDGE_TOP_RGBA, BRIDGE_SIDE_RGBA, bb,
                                lat0, lon0, int(cz), int(pz), int(px), int(py))
    mi = MATERIALS.index(MARBLE)
    out = {}
    for gx, gy, P, T, C in got:
        if not len(T):
            continue
        UV = np.tile(np.asarray(SKY_UV, np.float64), (len(P), 1))
        out[None if owners is None else (int(gx), int(gy))] = (P, UV, T, np.full(len(T), mi, np.int64), MATERIALS, C)
    return out


def far_bridge_tile(frame, tx, ty, md, bridges, out_dir, vfs_dir, aid):
    """A far bridge tile (render-only) written to the session folder: (vfs path of its JSON, part count) or None when
    its road cells are not at hand (fetched in the background: asked again later). bridges: the anchor's
    roads.BridgeProfiles."""
    import json
    import roads
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    keys = roads.cells_of_tile(frame, tx, ty, margin=0.0)
    if not all(md.has(k, "roads") for k in keys):
        for k in keys:
            md.prefetch(k, layers=("roads",))
        return None
    cells = [md.cell(k, layers=("roads",), kind="background").layers.get("roads") for k in keys]
    parts = [(xy, z, float(spec.width)) for _fid, spec, xy, z, _lon in bridges.parts(cells, (x0, y0, x1, y1))]
    mesh = bridge_slabs(parts, (x0, y0), box=(x0, y0, x1, y1)).get(None) if parts else None
    data = {"v": 1}
    if mesh is not None:
        shape = f"brfar_{aid}_{tx}_{ty}.cdae"
        write(os.path.join(out_dir, shape), mesh)
        data["shape"] = f"{vfs_dir}/{shape}"
        data["pos"] = [x0, y0, 0.0]
    name = f"brfar_{aid}_{tx}_{ty}.json"
    path = os.path.join(out_dir, name)
    tmp = staging.tmp_for(path)
    with open(tmp, "w") as f:
        f.write(json.dumps(data))
    staging.replace(tmp, path)
    return f"{vfs_dir}/{name}", len(parts) if mesh is not None else 0


def photo_mask(frame, x0, y0, size, px, md):
    """A (px, px) bool mask (row 0 = north) of the ground photo's square [x0, x0 + size) x [y0, y0 + size)
    (anchor frame) where buildings stand, grown by their edges and lean: there the photo shows roofs and
    walls that the 3D buildings now stand on or next to. None without buildings."""
    from PIL import Image, ImageDraw
    import mapcells
    la, lo, _ = frame.enu_to_geodetic(np.array([x0, x0 + size, x0, x0 + size]),
                                      np.array([y0, y0, y0 + size, y0 + size]), 0.0)
    keys = mapcells.cells_for_bbox(float(la.min()), float(lo.min()), float(la.max()), float(lo.max()))
    img = Image.new("L", (px, px), 0)
    draw = ImageDraw.Draw(img)
    mpp = size / px
    seen = set()
    n = 0
    for k in keys:
        f = md.cell(k, layers=("buildings",), kind="demand").layers.get("buildings")
        if f is None:
            continue
        hs = f.attrs.get("height")
        ids = f.ids.tolist()
        rows, a, b = _photo_candidates(f, frame, x0, y0, size)
        keep = [j for j, i in enumerate(rows) if ids[i] not in seen]  # (a building in two cells: the first piece)
        if not keep:
            seen.update(ids)
            continue
        rows, a, b = rows[keep], a[keep], b[keep]
        # every kept outer ring grown out from its middle, all at once
        h = np.full(len(rows), 8.0) if hs is None else np.asarray(hs, np.float64)[rows]
        h = np.where(np.isfinite(h), h, 8.0)
        grow = (PHOTO_GROW_M + np.minimum(PHOTO_LEAN * h, PHOTO_LEAN_MAX_M)) / mpp
        ln = b - a
        idx = np.repeat(a - np.r_[0, np.cumsum(ln)[:-1]], ln) + np.arange(int(ln.sum()))
        xy = f.xy_in(frame)[idx]
        cx, cy = (xy[:, 0] - x0) / mpp, (y0 + size - xy[:, 1]) / mpp
        starts = np.r_[0, np.cumsum(ln)[:-1]]
        mx = np.repeat(np.add.reduceat(cx, starts) / ln, ln)
        my = np.repeat(np.add.reduceat(cy, starts) / ln, ln)
        dx, dy = cx - mx, cy - my
        g = np.repeat(grow, ln) / np.maximum(np.hypot(dx, dy), 1e-6)
        pts = np.stack([cx + dx * g, cy + dy * g], 1).tolist()
        for s0, m in zip(starts.tolist(), ln.tolist()):
            draw.polygon([tuple(q) for q in pts[s0:s0 + m]], fill=255)
        n += len(rows)
        seen.update(ids)
    if n == 0:
        return None
    return np.asarray(img) > 0


def _photo_candidates(f, frame, x0, y0, size):
    """Rows of f (a cell's buildings) whose outer ring's box meets the square [x0, x0 + size]² and that come first
    among the rows of their id, ascending, with their outer rings' coordinate ranges [a, b): photo_mask's buildings, its test made for all rows at once (one ring at a
    time it was ~19k Python steps a block, most of a block photo's roof mask; 2026-10-23)."""
    m = len(f.kind)
    none = (np.zeros(0, np.int64),) * 3
    if m == 0:
        return none
    p = f.part_off[:-1]
    has = (f.part_off[1:] > p)
    r = f.ring_off[np.minimum(p, len(f.ring_off) - 2)]
    has &= f.ring_off[np.minimum(p, len(f.ring_off) - 2) + 1] > r
    a = f.coord_off[np.minimum(r, len(f.coord_off) - 2)]
    b = f.coord_off[np.minimum(r, len(f.coord_off) - 2) + 1]
    has &= b > a
    _, first = np.unique(f.ids, return_index=True)
    head = np.zeros(m, bool)
    head[first] = True
    rows = np.flatnonzero(has & head)
    if len(rows) == 0:
        return none
    xy = f.xy_in(frame)
    a, b = a[rows], b[rows]
    order = np.argsort(a, kind="stable")       # (reduceat: ascending starts)
    e_lo = np.empty(len(rows)); e_hi = np.empty(len(rows)); n_lo = np.empty(len(rows)); n_hi = np.empty(len(rows))
    e_lo[order] = mapgeom._reduce(np.minimum, xy[:, 0], a[order], b[order])
    e_hi[order] = mapgeom._reduce(np.maximum, xy[:, 0], a[order], b[order])
    n_lo[order] = mapgeom._reduce(np.minimum, xy[:, 1], a[order], b[order])
    n_hi[order] = mapgeom._reduce(np.maximum, xy[:, 1], a[order], b[order])
    inside = ~((e_hi < x0) | (e_lo > x0 + size) | (n_hi < y0) | (n_lo > y0 + size))
    return rows[inside], a[inside], b[inside]


def fill_photo(rgb, mask, size, px):
    """rgb (px, px, 3) uint8 with the masked pixels replaced by a smooth fill from the unmasked ground
    around them: a push-pull pyramid, so a footprint of any size fills (one PHOTO_FILL_M blur left black
    holes in big blocks), softened by that blur (earthfast.fill_photo: the numpy / Pillow version's numbers,
    a third of its time; 2026-10-23)."""
    fast.require(68, "the ground photo's roof fill")
    r = max(2.0, PHOTO_FILL_M / (size / px))
    return fast._ef.fill_photo(np.ascontiguousarray(rgb, np.uint8), np.ascontiguousarray(mask, bool), float(r))


def brick_walks(walks, items, lat, lon, tile=256.0):
    """Marks the sidewalk runs (roads.build_tile's walks) of a dense brick-region tile that are brick pavers
    (Fells Point's historic streets): every run of a street the same."""
    if not walks or brick_share(lat, lon) < 0.7 or coverage(items, tile) < BRICK_WALK_COVERAGE:
        return
    for wk in walks:
        fid = wk.get("fid")
        if fid is not None and _hash01(str(fid), "brickwalk") < BRICK_WALK_SHARE:
            wk["brick"] = True
