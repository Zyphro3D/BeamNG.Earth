"""Procedural buildings: custom 3D models in place of footprints (split out of buildings.py), the road tile's items,
bridge stubs and the district density.
"""
import os
import math
import numpy as np
import mapgeom
import francehouses
from bldbase import (
    BRIDGE_ASPECT,
    BRIDGE_MIN_SPAN_M,
    DISTRICT_M,
    MODEL_MAX_M,
    SUPPORT_DZ_WARN_M,
    SUPPRESS_MARGIN_M,
    SUPPRESS_MAX_M,
    SUPPRESS_MIN_M,
    WEST_TOWN_COVERAGE,
    _area2,
)
from bldsite import (
    bridge_lines,
    garage_rings,
    road_lines,
    support_rings,
    tunnel_lines,
    water_rings,
    way_lines,
)
from bldtile import (
    tile_buildings,
)


def _tex_to_png(tex, ext):
    """A glTF texture's raw bytes re-encoded to PNG (the engine's TSStatic materials need one), or None for
    an untextured material/slot (tex is None)."""
    if tex is None:
        return None
    if ext != "jpeg":
        return tex
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(tex)).save(buf, "PNG")
    return buf.getvalue()


def _section_mat_tex(sec_res, factors, has_alphas, alpha_cutoffs, double_sideds=None):
    """[(material, texture file name or None, factor, has_alpha, alpha_cutoff, double_sided), ...] of one write_model
    section result, single-material ("material"/"texture") or several ("materials"/"textures") - told apart
    by which key is actually there, since a split model's own later sections (threedmr._write_section's
    force_multi) always come back plural even when the model as a whole only has one material. double_sideds:
    per slot (threedmr.gltf_mesh's textures' 6th field), None all doubleSided."""
    if double_sideds is None:
        double_sideds = [True] * len(factors)
    if "materials" in sec_res:
        return list(
            zip(
                sec_res["materials"],
                sec_res["textures"],
                factors,
                sec_res["has_alpha"],
                sec_res["alpha_cutoff"],
                double_sideds,
            )
        )
    return [
        (
            sec_res["material"],
            sec_res.get("texture"),
            factors[0],
            sec_res["has_alpha"],
            sec_res["alpha_cutoff"],
            double_sideds[0],
        )
    ]


def _model_mesh(c, cache_dir):
    """road_tile_items' custom entry c's mesh as it is baked: (pos, uvs, tris, nrm, textures, tri_materials), a local
    model off disk (threedmr.load_local_model), a 3dmr.eu one fetched and decoded (threedmr.gltf_mesh), turned by
    its forced (threedmr.position_override) or OSM (threedmr.osm_yaw) yaw when c has a "latlon"."""
    import logging
    import threedmr

    log = logging.getLogger("earthd.buildings")
    cid = c["id"]
    if "local" in c:
        # the player's own mesh, already placed in threedmr_cache by hand - no fetch, no OSM-footprint
        # yaw (it is not a 3dmr.eu upload to straighten; the player orients their own STL themselves)
        return threedmr.load_local_model(cache_dir, c["local"])
    glb = threedmr.fetch_model(cid, cache_dir)
    pos, uvs, tris, nrm, textures, tri_materials = threedmr.gltf_mesh(glb, model_id=cid)
    latlon = c.get("latlon")
    if latlon is not None and len(pos):
        zmin, zmax = float(pos[:, 2].min()), float(pos[:, 2].max())
        lower = pos[pos[:, 2] <= zmin + 0.1 * (zmax - zmin)][:, :2]
        override = threedmr.position_override(cid)
        if override is not None and override[2] is not None:
            yaw = override[2]
            log.info("3dmr model %s: yaw %.1f° (forced by override)", cid, yaw)
        else:
            yaw = threedmr.osm_yaw(latlon[0], latlon[1], cid, lower, cache_dir)
            if yaw:
                log.info("3dmr model %s: yaw %.1f° (auto from OSM)", cid, yaw)
        if yaw:
            pos = threedmr.rotate_xz(pos, yaw)
    return pos, uvs, tris, nrm, textures, tri_materials


def _is_bridge(pos):
    """A model whose horizontal bbox is a bridge deck's shape (BRIDGE_ASPECT, BRIDGE_MIN_SPAN_M)."""
    dx = float(pos[:, 0].max() - pos[:, 0].min()) if len(pos) else 0.0
    dy = float(pos[:, 1].max() - pos[:, 1].min()) if len(pos) else 0.0
    return (dx > BRIDGE_ASPECT * dy or dy > BRIDGE_ASPECT * dx) and max(dx, dy) > BRIDGE_MIN_SPAN_M


def _centred(pos):
    """pos centred on its own horizontal bbox, as threedmr.write_model does before anything else."""
    P = np.array(pos, np.float64).reshape(-1, 3)
    P[:, 0] -= (P[:, 0].min() + P[:, 0].max()) / 2
    P[:, 1] -= (P[:, 1].min() + P[:, 1].max()) / 2
    return P


def _custom_tile_entries(frame, custom, ground, out_dir, vfs_dir, road_ends=None):
    """custom's (tile_buildings') 3dmr.eu models fetched, plus any local (no-3dmr-base, a "local" key -
    road_tile_items') models loaded straight off disk (threedmr.load_local_model, no fetch_model/osm_yaw),
    baked into out_dir and registered in the mod's materials (one per distinct source material a model uses
    - threedmr.gltf_mesh's/load_local_model's textures/tri_materials):
    ([{"shape", "pos"}], [(id, cx, cy, radius)]). The first (anchor-frame
    xy, z 0: write_model beds the mesh's own z to the ground, as the tile meshes' pos already does -
    roads.write_tile's "bld" etc) holds one entry per model, except a model too big for Torque's 16-bit IBO
    (threedmr.SPLIT_VERTEX_TRIGGER, threedmr.write_model's "sections"), which lands one entry per section,
    all at the same pos. The second: a suppression circle per baked model (_suppress_under_models), radius
    SUPPRESS_MARGIN_M past half the model's own horizontal bbox, clamped to [SUPPRESS_MIN_M, SUPPRESS_MAX_M]
    - the real mesh's own footprint is not known until it is downloaded and decoded, so the radius can only
    be had here (a user's report at Baghdad's Al-Shaheed Monument, model 24: two procedural "sector"
    stubs of its domes' lower tiers stayed and poked through the real mesh once it loaded).

    A model whose horizontal bbox is a bridge deck's shape (BRIDGE_ASPECT, BRIDGE_MIN_SPAN_M - told apart
    from a monument's or a building's) carries the road itself (FV=24, a user's report: no stub under it, driven on, no
    friction): its deck's top is traced (threedmr.bridge_deck_surface), the model leaned so that deck meets the
    road network where the mapped bridge leaves it (road_ends: blockworker's bridge_road_ends; else the ground at
    the deck's own ends - threedmr.bridge_tilt), drawn without collision ("col": False - its two-sided lattice
    caught the car) beside an invisible closed collision slab on that deck (threedmr.write_bridge_deck, a
    "_deck" entry at the same pos); the procedural bridge under it is left out of every tile it reaches
    (bridge_footprints, bridge_stub_*). A slab that fails to write leaves the model its own collision (and the
    tile incomplete: revisit.fault). Before that: if custom's entry carries
    a "latlon" (road_tile_items' - the model's own mapped position), the mesh is rotated about its vertical
    axis to match its OSM footprint's edges (threedmr.osm_yaw/rotate_xz) - not every 3dmr.eu upload follows
    the orientation guide (a user's report: the Eiffel Tower, id 4, stood ~45 degrees off though its
    position was right), so this runs before write_model's own centering (rotating a mesh that is not yet
    centered rotates it about its own bbox centre, not an arbitrary point) and before the radius/bridge-deck
    check below (its horizontal bbox can shift a little with the rotation). A model with a
    threedmr.position_override whose own yaw_deg is not None (id 22) uses that forced yaw instead - osm_yaw's
    OSM lookup is skipped entirely, not just its cache: the override is the already-settled answer.

    A model's download or decode failing drops that one entry (logged, no suppression circle for it) rather
    than the rest of the tile's; unsupported geometry (threedmr.UnsupportedModelError - Draco) is a permanent
    skip, anything else marks the build incomplete (revisit.fault()) so the tile is rebuilt and the model
    retried on the next visit - and, meanwhile, this visit's procedural stand-ins stay (no radius for a model
    that is not actually standing there yet).

    SUPPORT_DZ_WARN_M: write_model already beds every model on its own detected support plane
    (threedmr._support_z), so this is a log line only, not a filter - a model whose support plane sits this
    far from its bare bbox floor just gets flagged (could be legitimate: a bridge deck over a riverbed, an
    elevated plaza), not dropped - rejecting on a threshold here would risk eating an author's own real,
    intentionally elevated geometry."""
    import logging
    import threedmr

    log = logging.getLogger("earthd.buildings")
    # out_dir = <user_dir>/temp/beamngEarth/<sid> (earthd.Server.__init__): the mod's own folder, three levels up
    user_dir = os.path.dirname(os.path.dirname(os.path.dirname(out_dir)))
    # fetch_model appends "threedmr_cache" itself: pass the sessions' parent (temp/beamngEarth), so
    # the cache lives beside the session dirs and survives them (a session dir is wiped per anchor)
    cache_dir = os.path.dirname(out_dir)
    # the vfs twin of cache_dir/threedmr_cache (threedmr._persistent_png_dir) - where a model's own texture(s)
    # are baked too (write_model's tex_cache_dir below), so a materials.json entry still points at a live file
    # after a revisit cache hit restores the tile's .cdae alone (revisit.shapes_in never sees a colorMap)
    tex_vfs_dir = vfs_dir.rsplit("/", 1)[0] + "/threedmr_cache"
    log.info("3dmr: baking %d custom model(s) in tile", len(custom))
    entries, radii = [], []
    for c in custom:
        cid = c["id"]
        try:
            pos, uvs, tris, nrm, textures, tri_materials = _model_mesh(c, cache_dir)
            dx = float(pos[:, 0].max() - pos[:, 0].min()) if len(pos) else 0.0
            dy = float(pos[:, 1].max() - pos[:, 1].min()) if len(pos) else 0.0
            if max(dx, dy) > MODEL_MAX_M:
                log.warning("3dmr model %s: %.0f x %.0f m, wider than %.0f m: not placed", cid, dx, dy, MODEL_MAX_M)
                continue
            is_bridge = _is_bridge(pos) and "local" not in c
            la, lo, _ = frame.enu_to_geodetic(
                np.array([c["xy"][0]]), np.array([c["xy"][1]]), np.zeros(1)
            )
            gz = float(np.asarray(ground(la, lo), np.float64)[0])
            tilt = surf = None
            if is_bridge:
                # the road over it is the model's: its deck traced (threedmr.bridge_deck_surface, in the frame
                # write_model bakes it in: centred, bedded), both leaned so the deck's ends meet the ground roads
                Pc = _centred(pos)
                Pc[:, 2] += (gz - 0.2) - threedmr._support_z(Pc[:, 2])
                surf = threedmr.bridge_deck_surface(Pc, tris)
                if surf is not None:

                    def _ground_at(x, y, _c=c):
                        gla, glo, _ = frame.enu_to_geodetic(
                            np.array([_c["xy"][0] + x]), np.array([_c["xy"][1] + y]), np.zeros(1)
                        )
                        return float(np.asarray(ground(gla, glo), np.float64)[0])

                    ends = [(x - c["xy"][0], y - c["xy"][1], z) for x, y, z in (road_ends or {}).get(cid, ())]
                    tilt = threedmr.bridge_tilt(surf, _ground_at, cid, road_ends=ends)
                else:
                    log.warning("3dmr model %s: bridge without a deck to drive on", cid)
            pngs = [_tex_to_png(t[0], t[1]) for t in textures]
            factors = [t[2] for t in textures]
            has_alphas = [t[3] for t in textures]
            alpha_cutoffs = [t[4] for t in textures]
            double_sideds = [bool(t[5]) if len(t) > 5 else True for t in textures]   # (one-sided: modelled back faces)
            if len(pngs) == 1:
                res = threedmr.write_model(
                    out_dir,
                    cid,
                    pos,
                    tris,
                    uvs,
                    nrm,
                    pngs[0],
                    gz,
                    has_alpha=has_alphas[0],
                    alpha_cutoff=alpha_cutoffs[0],
                    tex_cache_dir=cache_dir,
                    tilt=tilt,
                )
            else:
                res = threedmr.write_model(
                    out_dir,
                    cid,
                    pos,
                    tris,
                    uvs,
                    nrm,
                    pngs,
                    gz,
                    tri_materials=tri_materials,
                    has_alpha=has_alphas,
                    alpha_cutoff=alpha_cutoffs,
                    tex_cache_dir=cache_dir,
                    tilt=tilt,
                )
            # a mesh over Torque's 16-bit IBO limit (threedmr.SPLIT_VERTEX_TRIGGER) comes back as several
            # sections (threedmr.write_model's "sections"), each its own TSStatic entry at the same pos -
            # a model with none (the common case) is just [res] itself
            sections = res.get("sections", [res])
            pos_xy = [round(float(c["xy"][0]), 2), round(float(c["xy"][1]), 2), 0.0]
            plate = None
            if surf is not None:
                try:
                    plate = threedmr.write_bridge_deck(out_dir, cid, surf, tilt)
                except Exception:
                    import revisit

                    revisit.fault()     # (built again next visit; meanwhile the model keeps its own collision)
                    log.warning("3dmr model %s: deck slab failed", cid, exc_info=True)
            for sec in sections:
                for (
                    material,
                    tex_name,
                    factor,
                    has_alpha,
                    alpha_cutoff,
                    double_sided,
                ) in _section_mat_tex(sec, factors, has_alphas, alpha_cutoffs, double_sideds):
                    texture_vfs = f"{tex_vfs_dir}/{tex_name}" if tex_name else None
                    threedmr.update_materials(
                        user_dir, material, texture_vfs, factor, has_alpha, alpha_cutoff,
                        double_sided=double_sided,
                    )
                entry = {"shape": f"{vfs_dir}/{sec['shape']}", "pos": pos_xy}
                if plate is not None:
                    entry["col"] = False    # (drawn only: the car is carried by its deck slab)
                entries.append(entry)
            if plate is not None:
                entries.append({"shape": f"{vfs_dir}/{plate['shape']}", "pos": pos_xy})
                log.info(
                    "3dmr model %s: deck slab %s (%d tris), ends at %s",
                    cid,
                    plate["shape"],
                    len(plate["tris"]),
                    [tuple(round(v, 2) for v in e) for e in plate["ends"]],
                )
            log.info(
                "3dmr model %s: baked as %s (%d section(s))",
                cid,
                res["shape"],
                len(sections),
            )
            if abs(res.get("support_dz", 0.0)) > SUPPORT_DZ_WARN_M:
                log.warning(
                    "3dmr model %s: support plane %.0f m off the ground (author terrain?)",
                    cid,
                    res["support_dz"],
                )
            radius = min(
                max(max(dx, dy) / 2.0 + SUPPRESS_MARGIN_M, SUPPRESS_MIN_M),
                SUPPRESS_MAX_M,
            )
            radii.append((cid, float(c["xy"][0]), float(c["xy"][1]), radius))
        except threedmr.UnsupportedModelError:
            pass  # already logged (threedmr.gltf_mesh): not a transient failure, no revisit.fault()
        except Exception:
            import revisit

            revisit.fault()
            log.warning("3dmr model %s failed", cid, exc_info=True)
    return entries, radii, []


def custom_models(frame, custom, ground, out_dir, vfs_dir, road_ends=None):
    """(entries, radii) of custom's 3dmr.eu / local models, baked into out_dir (_custom_tile_entries);
    ([], []) without custom footprints or out_dir/vfs_dir. Exposed on its own so a caller that needs
    entries/radii before the road mesh is built can bake the models once here and hand this same
    (entries, radii) to road_tile() below, rather than road_tile() baking them a second time."""
    if not (custom and out_dir and vfs_dir):
        return [], [], []
    return _custom_tile_entries(frame, custom, ground, out_dir, vfs_dir, road_ends=road_ends)


def road_tile_items(
    frame, tx, ty, md, ground, decks=None, skip=frozenset(), cache_dir=None
):
    """tile_buildings of road tile (tx, ty) (items, custom), from the map cells under it, plus any
    3dmr.eu models found near it by geo-search (threedmr.region_ids/model_position), appended to custom as
    {"id", "xy" (anchor frame), "latlon" (its mapped lat/lon, for threedmr.osm_yaw's OSM lookup in
    _custom_tile_entries)}, and any local (no-3dmr-base) models whose own lat/lon (threedmr.local_models) falls
    in the tile, appended the same way plus a "local" key (the catalog entry itself - _custom_tile_entries'
    signal to skip fetch_model/gltf_mesh for it and use threedmr.load_local_model instead). decks: the
    bridge decks near it [(xy (k, 2), z (k,))] in the anchor frame (roads.BridgeProfiles.parts):
    buildings they run through get a way. skip: building ids left out (gas station canopies). cache_dir: threedmr's
    cache (None skips the geo-search and the local catalog: lite/far tiles and callers that do not bake custom
    models).

    The geo-search is the tile's region's (threedmr.region_ids: its REGION_DEG square out to SEARCH_CORRECT_R_M past
    its corners, one cached search): a model counts where model_position puts it, its desc-corrected position when
    its api point is stale and far off (id 22, the Dragon Bridge: api point 26 km off in the mountains)."""
    import roads
    import threedmr

    box = roads.tile_bbox(tx, ty)
    x0, y0, x1, y1 = box
    house_layers = ("land_use",) if francehouses.needs_residential_land(frame, box) else ()
    cds = [
        md.cell(
            k,
            layers=("buildings", "building_parts", "roads", "infrastructure", "water") + house_layers,
            kind="demand",
        )
        for k in roads.cells_of_tile(frame, tx, ty, margin=0.0)
    ]
    items = tile_buildings(
        frame,
        x0,
        y0,
        x1,
        y1,
        [c.layers.get("buildings") for c in cds],
        ground,
        own=box if hasattr(box, "owns") else None,
        part_layers=[c.layers.get("building_parts") for c in cds],
        west_town=district_town(frame, tx, ty, md),
        roads_xy=road_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
        ways_xy=way_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
        tunnels_xy=tunnel_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
        garages_xy=garage_rings(
            frame, [c.layers.get("infrastructure") for c in cds], x0, y0
        ),
        supports=support_rings(
            frame, [c.layers.get("infrastructure") for c in cds], x0, y0
        ),
        bridges_xy=bridge_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
        water_xy=water_rings(frame, [c.layers.get("water") for c in cds], x0, y0),
        decks=[
            (
                np.asarray(d[0], np.float64)[:, :2] - [x0, y0],
                np.asarray(d[1], np.float64),
            )
            + tuple(d[2:3])
            for d in decks
        ]
        if decks
        else None,
        skip=skip,
        house_land_layers=[c.layers.get("land_use") for c in cds],
    )
    custom = []
    if cache_dir is not None:
        import logging

        log = logging.getLogger("earthd.buildings")
        clat, clon, _ = frame.enu_to_geodetic((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
        # (the region's models, one cached search per threedmr.REGION_DEG square: every model whose position -
        # model_position's, corrected where its api point is stale - stands in the tile; the PR's two per-tile
        # searches, tile radius and SEARCH_CORRECT_R_M, found the same models with a request per tile)
        for mid in threedmr.region_ids(float(clat), float(clon), cache_dir):
            mid = int(mid)
            try:
                mlat, mlon, _corrected = threedmr.model_position(mid, cache_dir)
                mx, my, _ = frame.geodetic_to_enu(mlat, mlon, 0.0)
            except Exception:
                log.warning("3dmr model %s: info failed", mid, exc_info=True)
                continue
            if x0 <= mx < x1 and y0 <= my < y1:
                custom.append(
                    {"id": mid, "xy": (float(mx), float(my)), "latlon": (mlat, mlon)}
                )
        for entry in threedmr.local_models(cache_dir):
            mlat, mlon = float(entry["lat"]), float(entry["lon"])
            mx, my, _ = frame.geodetic_to_enu(mlat, mlon, 0.0)
            if x0 <= mx < x1 and y0 <= my < y1:
                custom.append(
                    {
                        "id": entry["id"],
                        "xy": (float(mx), float(my)),
                        "latlon": (mlat, mlon),
                        "local": entry,
                    }
                )
    return items, custom


def near_rings(frame, tx, ty, md, margin=2.0, skip=frozenset()):
    """The outer rings (k, 2) in the anchor frame of every building of the map within margin m of road tile (tx, ty):
    its own and its neighbours' (road_tile_items has only the ones whose centre is in it). skip: ids left out."""
    import roads
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    out = []
    seen = set()
    for k in roads.cells_of_tile(frame, tx, ty, margin=margin):
        f = md.cell(k, layers=("buildings",), kind="demand").layers.get("buildings")
        if f is None or not len(f):
            continue
        bx = f.boxes(frame)
        for i in range(len(f)):
            fid = str(f.ids[i])
            if fid in seen or fid in skip:
                continue
            b = bx[i]
            if b[2] < x0 - margin or b[0] > x1 + margin or b[3] < y0 - margin or b[1] > y1 + margin:
                continue
            seen.add(fid)
            for part in f.parts(i):
                if len(part) and len(part[0]) >= 3:
                    e, n = mapgeom.enu_xy(f, frame, part[0])
                    out.append(np.c_[e, n])
    return out


def bridge_footprints(frame, tx, ty, cache_dir):
    """[(footprint, (x, y) of its model in frame, model id)] of the 3dmr model bridges whose deck reaches road tile
    (tx, ty) (threedmr.bridge_footprint: every tile reads the same per-model answer, the one the model is baked in
    or not), found as road_tile_items finds models (threedmr.region_ids, model_position)
    within threedmr.BRIDGE_REACH_M of the tile. [] without cache_dir. A model it could not read marks the build
    incomplete (revisit.fault: tried again next visit)."""
    import logging
    import revisit
    import roads
    import threedmr

    if cache_dir is None:
        return []
    log = logging.getLogger("earthd.buildings")
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    clat, clon, _ = frame.enu_to_geodetic(cx, cy, 0.0)
    reach = math.hypot((x1 - x0) / 2, (y1 - y0) / 2) + threedmr.BRIDGE_REACH_M
    out = []
    for mid in threedmr.region_ids(float(clat), float(clon), cache_dir):
        mid = int(mid)
        try:
            mlat, mlon, _corrected = threedmr.model_position(mid, cache_dir)
            mx, my, _ = frame.geodetic_to_enu(mlat, mlon, 0.0)
            mx, my = float(mx), float(my)
            if math.hypot(mx - cx, my - cy) > reach:
                continue
            c = {"id": mid, "xy": (mx, my), "latlon": (mlat, mlon)}

            def mesh(_c=c):
                pos, _uvs, tris, _nrm, _tex, _tm = _model_mesh(_c, cache_dir)
                return _centred(pos), tris, _is_bridge(pos)

            fp = threedmr.bridge_footprint(mid, cache_dir, mesh)
        except threedmr.UnsupportedModelError:
            continue
        except Exception:
            log.warning("3dmr model %s: bridge footprint failed", mid, exc_info=True)
            revisit.fault()
            continue
        if fp.get("bridge"):
            out.append((fp, (mx, my), mid))
    return out


def _under_bridges(fps, xy):
    """Mask (k,) of points xy (k, 2) under any of bridge_footprints' fps."""
    import threedmr

    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    m = np.zeros(len(xy), bool)
    for fp, centre, _mid in fps:
        m |= threedmr.under_footprint(fp, centre, xy)
    return m


def _mostly_under(fps, xy):
    xy = np.asarray(xy, np.float64)
    return len(xy) > 0 and _under_bridges(fps, xy[:, :2]).mean() >= 0.5


def bridge_stub_chunks(fps, chunks):
    """chunks without the procedural bridge standing where a model bridge (bridge_footprints' fps) does: a bridge
    chunk (roads._bridge_chunks: a "deck" surface, or a "fixed" line on it - its lane markings at the stub's
    heights) half or more of whose nodes are under one (the stub's slab, abutments, parapets, skin and markings all
    come from these; the ground roads it joins stay; tunnels too)."""
    if not fps:
        return chunks
    return [c for c in chunks
            if not ((c.get("deck") or c.get("fixed")) and not c.get("tunnel") and c.get("n") is not None
                    and len(c["n"]) and _mostly_under(fps, c["n"]))]


def bridge_stub_parts(fps, parts):
    """roads.BridgeProfiles.parts' (fid, spec, xy, z, lon) without those under a model bridge (their guard rails,
    medians and junction pads stood at the stub's heights)."""
    if not fps:
        return parts
    return [p for p in parts if not _mostly_under(fps, p[2])]


def bridge_road_ends(fps, bridges, cells):
    """{model id: [(x, y, z)]}: the ends of the mapped bridges (roads.BridgeProfiles.parts: whole, with their deck
    heights - the ground road's where they leave the bridge) under each model bridge of fps, in the tile's frame:
    where the model's deck is to meet the road network (threedmr.bridge_tilt road_ends)."""
    import threedmr

    out = {}
    for fp, centre, mid in fps:
        cols = [(i, c) for i, c in enumerate(fp["cols"]) if c]
        if not cols:
            continue
        u0, u1 = fp["u0"] + fp["du"] * cols[0][0], fp["u0"] + fp["du"] * cols[-1][0]
        v0, v1 = min(c[0] for _i, c in cols), max(c[1] for _i, c in cols)
        lo = (u0, v0) if fp["ax"] == 0 else (v0, u0)
        hi = (u1, v1) if fp["ax"] == 0 else (v1, u1)
        box = (centre[0] + lo[0], centre[1] + lo[1], centre[0] + hi[0], centre[1] + hi[1])
        ends = []
        for _fid, _spec, xy, z, _lon in bridges.parts(cells, box):
            xy = np.asarray(xy, np.float64)[:, :2]
            if len(xy) and threedmr.under_footprint(fp, centre, xy).mean() >= 0.5:
                ends += [(float(xy[0, 0]), float(xy[0, 1]), float(z[0])),
                         (float(xy[-1, 0]), float(xy[-1, 1]), float(z[-1]))]
        if ends:
            out[mid] = ends
    return out


def bridge_stub_piers(fps, piers, beams):
    """(piers, beams) of the procedural bridges (piers_in / beams_in) without those under a model bridge."""
    if not fps:
        return piers, beams
    piers = [p for p in piers if not _under_bridges(fps, [[p[0], p[1]]])[0]]
    beams = [b for b in beams
             if not _under_bridges(fps, [[(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]])[0]]
    return piers, beams


def bridge_stub_props(fps, props):
    """props rows ([group, x, y, z, ...]) without those standing on / under a model bridge's deck."""
    if not fps or not props:
        return props
    return [r for r in props
            if not (isinstance(r, (list, tuple)) and len(r) > 3 and _under_bridges(fps, [[r[1], r[2]]])[0])]


def coverage(items, tile=256.0):
    """The share of the tile (a square of side tile m: roads.tile_side) covered by building footprints."""
    return sum(_area2(s.ring) / 2 for s, _, _ in items if (s.min_h or 0) <= 0.5) / (tile * tile)


def centres(items, tx, ty):
    """The anchor-frame (x, y) centres of a road tile's buildings (road_tile_items' rings are tile-local)."""
    import roads
    x0, y0, _, _ = roads.tile_bbox(tx, ty)
    return [(float(s.ring[:, 0].mean() + x0), float(s.ring[:, 1].mean() + y0)) for s, _, _ in items]


_cell_stats = {}                # map cell key -> (lon, lat, area m2) per footprint (district_coverage)
_cell_lock = None


def _cell_lock_get():
    global _cell_lock
    if _cell_lock is None:
        import threading
        _cell_lock = threading.Lock()
    return _cell_lock


def cell_footprints(key, layer):
    """(lon, lat, area m2) arrays of a cell's building footprints (outer rings), cached per cell (layer may
    be None once cached). Pure."""
    _cell_lock_get()
    with _cell_lock:
        got = _cell_stats.get(key)
    if got is not None:
        return got
    if layer is None or len(layer.ids) == 0:
        got = (np.zeros(0), np.zeros(0), np.zeros(0))
    else:
        f = layer
        first = f.ring_off[f.part_off[:-1]]
        a = f.coord_off[first]
        b = f.coord_off[first + 1]
        ll = f.lonlat
        lat0 = float(np.mean(ll[:, 1]))
        x = ll[:, 0] * 111320.0 * math.cos(math.radians(lat0))
        y = ll[:, 1] * 111320.0
        cross = np.r_[x[:-1] * y[1:] - x[1:] * y[:-1], 0.0]
        C = np.concatenate(([0.0], np.cumsum(cross)))
        CX, CY = np.concatenate(([0.0], np.cumsum(ll[:, 0]))), np.concatenate(([0.0], np.cumsum(ll[:, 1])))
        n = np.maximum(b - a, 1)
        last = np.maximum(b - 1, a)
        area = np.abs(C[last] - C[a] + x[last] * y[a] - x[a] * y[last]) / 2
        got = ((CX[b] - CX[a]) / n, (CY[b] - CY[a]) / n, area)
    with _cell_lock:
        if len(_cell_stats) > 64:
            _cell_stats.clear()
        _cell_stats[key] = got
    return got


def district_coverage(lat, lon, cells):
    """The share of the DISTRICT_M disc around (lat, lon) covered by building footprints; cells: [(key,
    buildings layer)] covering the disc."""
    seen = 0.0
    mx = 111320.0 * math.cos(math.radians(lat))
    for key, layer in cells:
        lo, la, ar = cell_footprints(key, layer)
        if len(ar):
            d2 = ((lo - lon) * mx) ** 2 + ((la - lat) * 111320.0) ** 2
            seen += float(ar[d2 <= DISTRICT_M * DISTRICT_M].sum())
    return seen / (math.pi * DISTRICT_M * DISTRICT_M)


def district_town(frame, tx, ty, md):
    """Is road tile (tx, ty) in a small town of the arid West (WEST_TOWN_COVERAGE)? The cells of the district
    are loaded like the tile's own (the far building tiles need them anyway); an unreadable one counts as
    empty."""
    import mapcells
    import roads
    import surface
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    lat, lon, _ = frame.enu_to_geodetic((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
    lat, lon = float(lat), float(lon)
    if not bool(surface.dry_west(lat, lon)):
        return False
    dlat = DISTRICT_M / 111320.0
    dlon = DISTRICT_M / (111320.0 * math.cos(math.radians(lat)))
    cells = []
    for key in mapcells.cells_for_bbox(lat - dlat, lon - dlon, lat + dlat, lon + dlon):
        with _cell_lock_get():
            known = key in _cell_stats
        if known:
            cells.append((key, None))
            continue
        try:
            cells.append((key, md.cell(key, layers=("buildings",), kind="demand").layers.get("buildings")))
        except Exception:           # not cached / unreadable: counts as empty
            continue
    return district_coverage(lat, lon, cells) < WEST_TOWN_COVERAGE
