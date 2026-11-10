"""World server: the near tiles' builders (split out of earthd.py's Server): roads, signs, vegetation, turbines,
navigation graph, far buildings and bridges, sky, terrain patches, the geotiles and the gas station pads.
"""
import os
import asyncio  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import concurrent.futures  # noqa: E402
import numpy as np  # noqa: E402
import blockworker  # noqa: E402
import secrossings  # noqa: E402
import navgraph as navgraphmod  # noqa: E402
import distanttraffic          # noqa: E402
import parking as parkingmod  # noqa: E402
import gasstations as gasstationsmod  # noqa: E402
import roads as roadsmod  # noqa: E402
import rails as railsmod  # noqa: E402
import urgency as urgencymod  # noqa: E402
import placenames as placenamesmod  # noqa: E402
import pools as poolsmod  # noqa: E402
import grading as gradingmod  # noqa: E402
import roadmesh as roadmeshmod  # noqa: E402
import revisit as revisitmod  # noqa: E402
import signstream  # noqa: E402
import dirsigns as dirsignsmod  # noqa: E402
import dem as demmod  # noqa: E402
import mapcells  # noqa: E402
import terrainpatch as terrainpatchmod  # noqa: E402
import seacarve as seacarvemod  # noqa: E402
import vegetation as vegetationmod
import buildings as buildingsmod  # noqa: E402
import furniture as furnituremod  # noqa: E402
import development as developmentmod  # noqa: E402
import townentry as townentrymod  # noqa: E402     (PR #597's town entry and exit signs)
import heightmap  # noqa: E402
import staging  # noqa: E402
import cpupin  # noqa: E402
from earthbase import (
    DETAIL_ROAD_LIFT,
    OSM_FUEL_WARM_TRIES,
    OSM_FUEL_WARM_WAIT_S,
    ROAD_GEO_SLOW_S,
    ROAD_TILE_TIMEOUT_S,
    TURBINE_Z,
    VEG_GEO_BATCH,
    fast_pool_size,
    log,
    osm_fuel_cells,
    osm_fuel_tiles,
)


class _BuildMixin:
    """Server's near tile builders (split out of earthd.py; Server inherits it)."""

    def build_veg_tile(self, aid, tx, ty, flat=False):
        """vegetation.py: one vegetation tile of anchor aid written to the session folder: (vfs path, n) or
        None; flat: as one flat array (a mod that reads it, hello.vegFlat). Worker threads (builds the map cells it
        lacks, ~3.5 s each the first time)."""
        anchor = self.anchors.get(aid)
        if anchor is None or self.veg is None:
            return None
        zoom = anchor.zooms[0]
        g = self.grader_for(anchor)

        def raw(la, lo):
            return self.ground(anchor.frame, la, lo, self.dem.sample(la, lo, zoom))

        def elev(la, lo):
            z = raw(la, lo)
            # graded under the roads as the block is: a rock beside a track cut into a slope floated 1.9 m over the cut
            # (Stelvio pass, the user's report 2026-10-17)
            return g.apply(la, lo, z, gradingmod.ROAD_LIFT) if g is not None else z
        rows = vegetationmod.tile_items(anchor.frame, tx, ty, self.veg, lambda k: True, elev, self.sea, as_json=True,
                                        raw=raw if g is not None else None)
        if rows is None:
            return None
        return vegetationmod.write_tile(self.out_dir, self.vfs_dir, aid, tx, ty, rows, flat=flat)

    def build_veg_geo(self, gx, gy, flat=False, aid=None):
        """vegetation.geo_tile_items: geotile (gx, gy)'s trees, bushes and rocks in its own frame, written to the
        session folder: (vfs path, n, frame (lat, lon), pivot) or None. No anchor in it: the ground is the graded one
        (the plans are the planning regions', whatever the anchor) at the active anchor's DEM zoom."""
        if self.veg is None:
            return None
        urgencymod.wait_turn()          # (a block the game waits for first: urgency.py)
        import geotiles
        anchor = self.anchors.get(aid) or (self.anchors[max(self.anchors)] if self.anchors else None)
        if anchor is None:
            return None
        zoom = anchor.zooms[0]
        if self.road_pool is not None and self.mapdata is not None:
            # (in a road tile process: here they took the server's one interpreter from the far textures and plans)
            t0, waited = time.perf_counter(), []
            fut, i = self._veg_geo_job(anchor, zoom, gx, gy, flat)
            got = poolsmod.result_running(fut, None, waited)
            n = len(got)
            got = got[i]
            if time.perf_counter() - t0 > ROAD_GEO_SLOW_S:
                log.info("trees geotile %d_%d in %.1f s (queue %.1f, %d in its job)", gx, gy,
                         time.perf_counter() - t0, waited[0] if waited else 0.0, n)
            if isinstance(got, Exception):
                raise got
            return got
        g = self.grader_for(anchor)

        def raw(la, lo):
            return self.ground(None, la, lo, self.dem.sample(la, lo, zoom))

        def elev(la, lo):
            z = raw(la, lo)
            return g.apply(la, lo, z, gradingmod.ROAD_LIFT) if g is not None else z
        got = vegetationmod.geo_tile_items(gx, gy, self.veg, lambda k: True, elev, self.sea, as_json=True,
                                           raw=raw if g is not None else None)
        if got is None:
            return None
        rows, pivot = got
        frame = geotiles.centre(gx, gy)
        path, n = vegetationmod.write_tile(self.out_dir, self.vfs_dir, None, gx, gy, rows, flat=flat,
                                           geo={"key": geotiles.key(gx, gy), "frame": frame, "pivot": pivot})
        return path, n, frame, pivot

    def _veg_geo_job(self, anchor, zoom, gx, gy, flat):
        """(future, index): the road processes' job (blockworker.veg_geo_many) that builds geotile (gx, gy), its result
        the index'th of the job's list. The tiles asked for while a job waits for its turn join it (VEG_GEO_BATCH at
        most): one at a time, after the road tiles (RespawningPool.submit_low), each waited out the whole queue of
        road tiles for its ~10 ms of work, and the trees fell behind the car (533 tree tiles pending, 22 s each,
        2026-11-08)."""
        submit = getattr(self.road_pool, "submit_low_lazy", None)
        if submit is None:                  # (a plain executor: one tile a job)
            return self.road_pool.submit(blockworker.veg_geo_many, anchor.id, anchor.lat, anchor.lon, zoom,
                                         [(gx, gy)], self.out_dir, self.vfs_dir, flat), 0
        key = (anchor.id, zoom, bool(flat))
        with self._veg_job_lock:
            job = self._veg_jobs.get(key)
            if job is None or len(job[1]) >= VEG_GEO_BATCH or job[0].done():
                tiles = []

                def make_args(key=key, tiles=tiles, a=anchor):
                    with self._veg_job_lock:            # (no more joins: the job is on its way)
                        if self._veg_jobs.get(key, (None, None))[1] is tiles:
                            del self._veg_jobs[key]
                        return (a.id, a.lat, a.lon, zoom, list(tiles), self.out_dir, self.vfs_dir, flat), {}
                job = self._veg_jobs[key] = (submit(blockworker.veg_geo_many, make_args), tiles)
            job[1].append((gx, gy))
            return job[0], len(job[1]) - 1

    def place_names(self):
        """The status bar's town / state / country lookup (placenames.PlaceNames), made at its first use: it shares the
        map data's Overture reader (footers and row groups read once) when there is one."""
        with self._place_names_lock:
            pn = getattr(self, "_place_names", None)
            if pn is None:
                md = getattr(self, "mapdata", None)
                pn = self._place_names = placenamesmod.PlaceNames(
                    md.cache_dir if md is not None else self.cache_dir, self.net,
                    reader=md.overture if md is not None else None)
            return pn

    def build_turbines_geo(self, gx, gy):
        """Mapped wind turbines through the existing geotile tracker, without building distant roads.

        Empty cells do no DEM work. Populated cells sample their exact mapped points in one batch, with the
        existing airport/sea ground rules; nearby road-owned records provide the graded installation height.
        The tracker limits simultaneous batches, and this session memo keeps delivered heights stable.
        """
        import geotiles
        import windturbines
        memo = self.__dict__.setdefault("_turbines_geo_done", {})
        key = geotiles.key(gx, gy, TURBINE_Z)
        filename = "turbines_" + key + ".json"
        path = os.path.join(self.out_dir, filename)
        cached = memo.get(key)
        if cached is not None and os.path.exists(path):
            return cached
        clat, clon, south, west, north, east = geotiles.info(gx, gy, TURBINE_Z)
        frame = geotiles.frame(gx, gy, TURBINE_Z)
        features = tuple(f for f in windturbines.near(south, west, north, east)
                         if geotiles.tile_of(f.lat, f.lon, TURBINE_Z) == (gx, gy))
        rows = []
        if features:
            zoom = demmod.zoom_for(clat, 1.25, demmod.FINE_ZOOM)

            def elev(la, lo):
                return self.ground(frame, la, lo, self.dem.sample(la, lo, zoom))
            rows = windturbines.rows_for_features(frame, features, elev, collide=False)
            if len(rows) != len(features):     # unavailable heights retry; never memoize a partial farm
                return None
        pivot = float(np.mean([r["z"] for r in rows])) if rows else 0.0
        data = {"key": key, "sid": self.sid, "frame": [clat, clon], "pivot": pivot, "turbines": rows}
        tmp = staging.tmp_for(path)
        with open(tmp, "w", encoding="utf-8") as target:
            json.dump(data, target, separators=(",", ":"))
        staging.replace(tmp, path)
        result = (f"{self.vfs_dir}/{filename}", len(rows), (clat, clon), pivot)
        if len(memo) >= 4096:
            memo.clear()
        memo[key] = result
        return result

    def build_navgraph(self, aid, x, y, serial):
        """navgraph.py: the AI road graph around (x, y) of anchor aid, written to the session folder (the
        anchor's earlier graphs deleted): (vfs path, nodes, edges) or None. Worker threads."""
        anchor = self.anchors.get(aid)
        if anchor is None or self.mapdata is None:
            return None
        if self.road_pool is not None:      # in a road process, with its bridges (blockworker.navgraph)
            t0 = time.perf_counter()
            cla, clo, _ = anchor.frame.enu_to_geodetic(x, y, 0.0)
            rules = navgraphmod.road_rules(navgraphmod.country_near(self.airports_db, float(cla), float(clo)))
            path, n_nodes, n_edges = self.road_pool.submit(
                blockworker.navgraph, aid, anchor.lat, anchor.lon, anchor.zooms[0], x, y, self.out_dir,
                self.vfs_dir, serial, self.left_hand(anchor), self.mph(anchor), rules).result()
            log.debug("navgraph %d of anchor %d: %d nodes in %.2f s (queue + build)", serial, aid, n_nodes,
                     time.perf_counter() - t0)
            keep = os.path.join(self.out_dir, os.path.basename(path))
            for old in glob.glob(os.path.join(self.out_dir, f"navgraph_{aid}_*.json")):    # one file at a time
                if old != keep:
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            return path, n_nodes, n_edges
        t0 = time.perf_counter()
        w = navgraphmod.WINDOW_M + 200.0
        la, lo, _ = anchor.frame.enu_to_geodetic(np.array([x - w, x + w]), np.array([y - w, y + w]), 0.0)
        keys = mapcells.cells_for_bbox(float(la.min()), float(lo.min()), float(la.max()), float(lo.max()))
        feats = [self.mapdata.cell(k, layers=("roads",), kind="demand").layers.get("roads") for k in keys]
        zoom = anchor.zooms[0]
        urban = self.surface.urban if self.surface is not None else None
        g = self.grader_for(anchor)

        def elev(la_, lo_):         # the physics ground, lifted under roads (as the road tiles')
            z = self.ground(anchor.frame, la_, lo_, self.dem.sample(la_, lo_, zoom))
            return g.apply(la_, lo_, z, gradingmod.ROAD_LIFT) if g is not None else z
        decks = {}
        try:            # the bridges' deck heights (the road tiles' BridgeProfiles, one per anchor)
            with self.grader_lock:
                if anchor.bridges is None:
                    anchor.bridges = self.chain_view(anchor, "bridges", elev)
            for fid, _spec, xy, z, _lon in anchor.bridges.parts(feats, (x - w, y - w, x + w, y + w)):
                decks.setdefault(fid, []).append((np.asarray(xy, np.float64)[:, :2], np.asarray(z, np.float64),
                                                  anchor.bridges.depth_of(fid)))
        except Exception as e:          # no decks: the graph follows the ground under bridges
            log.debug("navgraph decks: %s: %s", type(e).__name__, e)
        nodes = secrossings.read(self.mapdata, anchor.frame, keys) if str(self.country(anchor) or "").upper() == "SE" else None
        tun = {}
        try:            # (the tunnels the road tiles build: the AI drives through them; off: navgraph.AI_TUNNELS)
            if not navgraphmod.AI_TUNNELS:
                raise LookupError("AI tunnels off")
            with self.grader_lock:
                if anchor.tunnels is None:
                    anchor.tunnels = self.chain_view(anchor, "tunnels", elev, g)
            for fid, _spec, xy, z, _lon in anchor.tunnels.parts(feats, (x - w, y - w, x + w, y + w)):
                tun.setdefault(navgraphmod.part_fid(fid), []).append((np.asarray(xy, np.float64)[:, :2],
                                                                   np.asarray(z, np.float64)))
        except Exception as e:
            log.debug("navgraph tunnels: %s: %s", type(e).__name__, e)
        graph = navgraphmod.build(anchor.frame, feats, elev, (x, y), urban, decks=decks, tunnels=tun,
                                  widths=self.road_widths,
                                  left_hand=self.left_hand(anchor), mph=self.mph(anchor),
                                  signals=furnituremod.signals_for(self.mapdata, anchor.frame, keys),
                                  country=self.country(anchor), priority=None if nodes is None else nodes["priority"])
        cla, clo, _ = anchor.frame.enu_to_geodetic(x, y, 0.0)
        graph["rules"] = navgraphmod.road_rules(navgraphmod.country_near(self.airports_db, float(cla), float(clo)))
        for old in glob.glob(os.path.join(self.out_dir, f"navgraph_{aid}_*.json")):    # one file at a time
            try:
                os.remove(old)
            except OSError:
                pass
        path = navgraphmod.write(self.out_dir, self.vfs_dir, aid, serial, graph)
        log.debug("navgraph %d of anchor %d: %d nodes in %.2f s", serial, aid, len(graph["nodes"]),
                 time.perf_counter() - t0)
        return path, len(graph["nodes"]), len(graph["edges"])

    def load_boost(self, on, why="", full_only=False, half_only=False):
        """The loading boost (cpupin.boosted): the server's processes also on half of the game's cores while the
        player waits after a teleport, back on their own when it moves or the far field is done. With the settings
        page's teleportPriority (cpupin.full_split): the server on every core and the game on two, until the far field
        and the roads round the car are done (moving does not end it); full_only: that alone (joining the world)."""
        if on and not self.boost_on:
            full = cpupin.full_split(self.server_cpus, getattr(self, "game_cpus", None)) \
                if not half_only and cpupin.player_settings(self.user_dir).get("teleport_full") else None
            if full:
                try:
                    n = cpupin.pin_tree(full[0])
                    g = cpupin.pin_game(full[1])
                except OSError as e:
                    log.warning("full priority loading: %s", e)
                else:
                    self.boost_on = self.boost_full = True
                    log.info("full priority loading on (%s): %d server threads on CPUs %s, the game's %d on %s", why, n,
                             ",".join(map(str, sorted(full[0]))), g, ",".join(map(str, sorted(full[1]))))
                    return
            if full_only:
                return                  # (the half boost only after a teleport)
        if not on and self.boost_full:
            try:
                cpupin.pin_tree(self.server_cpus)
                cpupin.pin_game(getattr(self, "game_cpus", None))
            except OSError as e:
                log.warning("full priority loading: %s", e)
            self.boost_on = self.boost_full = False
            log.info("full priority loading off (%s)", why)
            return
        if on and not self.boost_cpus:
            log.debug("load boost (%s): no cores to lend (--pin / --load-boost off, or too few cores)", why)
        on = bool(on) and bool(self.boost_cpus)
        if on == self.boost_on:
            return
        if getattr(self, "split_on", False):
            self.boost_on = on          # (an edge hold's split in force: its end pins by this, split_boost)
            log.info("load boost %s (%s) after the edge hold", "on" if on else "off", why)
            return
        cpus = ((getattr(self, "catchup_cpus", None) or self.boost_cpus) if half_only else self.boost_cpus) if on \
            else self.server_cpus
        if not cpus:
            return
        try:
            n = cpupin.pin_tree(cpus)
        except OSError as e:
            log.warning("load boost: %s", e)
            return
        self.boost_on = on
        log.info("load boost %s (%s): %d threads on CPUs %s", "on" if on else "off", why, n,
                 ",".join(map(str, sorted(cpus))))

    def split_boost(self, on, why):
        """The full priority split (cpupin.full_split: the server on every CPU, the game on its first cores) on, or off
        back to the boost in force before (Server.load_boost's state is left as it was). True when on. Not over full
        priority loading (boost_full: that is the same split already)."""
        if on:
            if self.boost_full or getattr(self, "split_on", False):
                return False
            full = cpupin.full_split(self.server_cpus, getattr(self, "game_cpus", None))
            if not full:
                return False
            try:
                n = cpupin.pin_tree(full[0])
                g = cpupin.pin_game(full[1])
            except OSError as e:
                log.warning("%s boost: %s", why, e)
                return False
            self.split_on = True
            log.info("%s: every core (%d server threads on CPUs %s, the game's %d on %s)", why, n,
                     ",".join(map(str, sorted(full[0]))), g, ",".join(map(str, sorted(full[1]))))
            return True
        if not getattr(self, "split_on", False):
            return False
        self.split_on = False
        if self.boost_full:
            return False                # (full priority loading came meanwhile: its split stays)
        try:
            cpupin.pin_tree(self.boost_cpus if self.boost_on and self.boost_cpus else self.server_cpus)
            cpupin.pin_game(getattr(self, "game_cpus", None))
        except OSError as e:
            log.warning("%s: %s", why, e)
        log.info("%s: the cores back as before", why)
        return False

    def build_far_bld_tile(self, aid, tx, ty):
        """buildings.far_tile: a far (render-only) building tile of anchor aid, or None. Worker threads."""
        anchor = self.anchors.get(aid)
        if anchor is None or self.mapdata is None or not self.buildings:
            return None
        zoom = anchor.zooms[0]
        if self.far_bld_pool is not None:
            t0 = time.perf_counter()
            fut = self.far_bld_pool.submit(blockworker.far_buildings_tile, aid, anchor.lat, anchor.lon, zoom, tx, ty,
                                           self.out_dir, self.vfs_dir, self.lamp_region(anchor))
            res = fut.result()
            log.debug("far buildings tile %d_%d: %.2f s (queue + build)", tx, ty, time.perf_counter() - t0)
            return res
        return buildingsmod.far_tile(anchor.frame, tx, ty, self.mapdata,
                                     lambda la, lo: self.ground(anchor.frame, la, lo, self.dem.sample(la, lo, zoom)),
                                     self.out_dir, self.vfs_dir, aid, self.lamp_region(anchor))

    def build_signs_tile(self, aid, tx, ty):
        """signstream.build_tile: the highway signs of road tile (tx, ty) of anchor aid (its roads are with the mod
        already), or None. Worker threads (signstream.Tracker)."""
        anchor = self.anchors.get(aid)
        source = getattr(self.mapdata, "highway_signs", None)
        if anchor is None or source is None:
            return None
        source.version()            # (the sign data's build, learnt here off the event loop: Connection.on_pose)
        if self.sign_pool is not None:
            return poolsmod.result_running(self.sign_pool.submit(
                blockworker.signs_tile, aid, anchor.lat, anchor.lon, anchor.zooms[0], tx, ty, self.out_dir,
                self.vfs_dir, self.left_hand(anchor), self.mph(anchor), self.country(anchor), dirsignsmod.enabled(self)),
                ROAD_TILE_TIMEOUT_S)
        g = self.grader_for(anchor)
        zoom = anchor.zooms[0]

        def elev(la, lo):
            z = self.ground(anchor.frame, la, lo, self.dem.sample(la, lo, zoom))
            return g.apply(la, lo, z, gradingmod.ROAD_LIFT) if g is not None else z
        urban = self.surface.urban if self.surface is not None and hasattr(self.surface, "urban") else None
        return signstream.build_tile(self.mapdata, anchor.frame, aid, tx, ty, elev, self.out_dir, self.vfs_dir,
                                     urban, self.road_widths, self.left_hand(anchor), self.mph(anchor),
                                     self.country(anchor), dirsignsmod.enabled(self))

    def build_sky_tile(self, aid, ux, uy):
        """buildings.skyline_tile: a skyline (render-only) tile of anchor aid, or None (its cells not at hand: asked
        again later). Worker threads."""
        anchor = self.anchors.get(aid)
        if anchor is None or self.mapdata is None or not self.buildings:
            return None
        zoom = anchor.zooms[0]
        if self.far_bld_pool is not None:
            return self.far_bld_pool.submit(blockworker.skyline_tile, aid, anchor.lat, anchor.lon, zoom, ux, uy,
                                            self.out_dir, self.vfs_dir, self.lamp_region(anchor)).result()
        return buildingsmod.skyline_tile(anchor.frame, ux, uy, self.mapdata,
                                         lambda la, lo: self.ground(anchor.frame, la, lo, self.dem.sample(la, lo, zoom)),
                                         self.out_dir, self.vfs_dir, aid, self.lamp_region(anchor))

    def build_far_bridge_tile(self, aid, tx, ty):
        """buildings.far_bridge_tile: a far bridge (render-only) tile of anchor aid, or None (its road cells not at
        hand, or no worker processes: none then). Worker threads."""
        anchor = self.anchors.get(aid)
        if anchor is None or self.mapdata is None or self.far_bld_pool is None:
            return None
        return self.far_bld_pool.submit(blockworker.far_bridge_tile, aid, anchor.lat, anchor.lon, anchor.zooms[0],
                                        tx, ty, self.out_dir, self.vfs_dir).result()

    def build_terrain_patch(self, aid, key, box):
        """terrainpatch.py: the collision patch of the ground under a collision island off the terrain block (the
        mod's terrainpatch.lua), written to the session folder in pieces. ([(vfs path, (cx, cy))], triangles) or None
        if the anchor is gone. Worker threads."""
        anchor = self.anchors.get(aid)
        if anchor is None:
            return None
        box = terrainpatchmod.snap_box(*box)
        path = os.path.join(self.out_dir, f"patch_{aid}")      # (+ _<cell>.cdae a piece: shared by patches)
        if self.road_pool is not None:
            parts, n = self.road_pool.submit(blockworker.terrain_patch, aid, anchor.lat, anchor.lon,
                                             anchor.zooms[0], box, path).result()
        else:
            zoom, g = anchor.zooms[0], self.grader_for(anchor)

            def elev(la, lo):
                z = self.ground(anchor.frame, la, lo, self.dem.sample(la, lo, zoom))
                return g.apply(la, lo, z, gradingmod.ROAD_LIFT) if g is not None else z
            parts, n = terrainpatchmod.build(anchor.frame, elev, box, path)
        return [(f"{self.vfs_dir}/{os.path.basename(p)}", c) for p, c in parts], n

    def warm_place(self, anchor, x, y):
        """The roads and bridges (WARM_PLACE_LAYERS) of the road tiles the mod will want around (x, y) (roads.SEND_M),
        fetched at once, nearest first: the first block's grading and the road tiles wait for them, and on a fresh place
        they came one after another as each part of the work asked (GitHub #241). Returns how many were queued."""
        if self.mapdata is None:
            return 0
        water = 1 + seacarvemod.LINE_CELLS if self.sea is not None else 0
        needs = set()
        for tx, ty in roadsmod.tiles_near(x, y, roadsmod.SEND_M):
            needs.update(roadsmod.tile_needs(anchor.frame, tx, ty, self.buildings, self.road_widths is not None, water))

        def where(k):
            la, lo = k.center()
            e, n, _ = anchor.frame.geodetic_to_enu(la, lo, 0.0)
            return math.hypot(float(e) - x, float(n) - y)
        # (only what the first block's grading and the bridges wait for: each road tile asks for its own buildings and
        # water when it starts; all of it at once kept a Steam Deck's 4 cores from the road tiles themselves)
        order = sorted((kl for kl in needs if kl[1] in self.WARM_PLACE_LAYERS),
                       key=lambda kl: (self.WARM_ORDER.get(kl[1], 9), where(kl[0]), str(kl[0]), kl[1]))
        n = self.mapdata.warm(order)
        if getattr(self.sea, "prefetch_full", None) is not None:   # (the coast's sea floors: NOAA's slow requests
                                                                   # from now, not when a deck asks)
            own = {(k.x, k.y): where(k) for tx, ty in roadsmod.tiles_near(x, y, roadsmod.SEND_M)
                   for k in roadsmod.cells_of_tile(anchor.frame, tx, ty)}
            self.sea.prefetch_full(sorted(own, key=lambda k: (own[k], k)))
        import roadz        # (the measured road heights of the same cells, kept on disk for the road processes)
        if roadz.PROVIDER is not None:
            roadz.PROVIDER.warm(sorted({kl[0] for kl in order if kl[1] == "roads"}, key=lambda k: (where(k), str(k))))
        return n

    def roads_geo_ok(self):
        """Road tiles as geotiles (--geo-tiles on, road tiles in worker processes): content kept across rebases."""
        return bool(self.geo_tiles) and self.road_pool is not None and self.mapdata is not None

    def geo_rules(self, gx, gy, z=None):
        """(left_hand, mph, buried share, lamp region, country, DEM zoom) at geotile (gx, gy)'s centre (zoom z, the
        road geotiles' by default): its content's own place's rules (an anchor's were the anchor's), kept per tile."""
        import geotiles
        z = geotiles.Z if z is None else int(z)
        cache = self.__dict__.setdefault("_geo_rules", {})
        got = cache.get((gx, gy, z))
        if got is None:
            la, lo = geotiles.centre(gx, gy, z)
            c = navgraphmod.country_near(self.airports_db, la, lo)
            got = (bool(navgraphmod.road_rules(c)["rightHandDrive"]), bool(navgraphmod.road_rules(c)["mph"]),
                   furnituremod.buried_share(c),
                   furnituremod.lamp_region(c, navgraphmod.region_near(self.airports_db, la, lo)),
                   c if self.airports_db else None, demmod.zoom_for(la, 1.25, demmod.FINE_ZOOM))
            if len(cache) > 20000:
                cache.clear()
            cache[(gx, gy, z)] = got
        return got

    def far_geo_ok(self):
        """Far buildings, the skyline and far bridges as geotiles (--geo-tiles on, their worker process): built once
        per tile, kept by the mod across rebases."""
        return bool(self.geo_tiles) and self.far_bld_pool is not None and self.mapdata is not None and self.buildings

    def build_far_geo(self, kind, gx, gy):
        """blockworker.far_geo in the far buildings' process: far geotile (gx, gy) of kind ("bldfar", "bldsky",
        "brfar"), anchor-free (its own place's rules and DEM zoom); one built before in this session comes from its
        files. (vfs path, n, frame, pivot, shapes) or None (its cells not at hand yet). Worker threads
        (vegetation.Tracker geo)."""
        if self.far_bld_pool is None or self.mapdata is None:
            return None
        import geotiles
        z = blockworker.far_geo_zoom(kind)
        memo = self.__dict__.setdefault("_far_geo_done", {})
        got = memo.get((kind, gx, gy))
        if got is not None and all(os.path.exists(os.path.join(self.out_dir, n))
                                   for n in [os.path.basename(got[0])] + list(got[4].values())):
            return got
        _lh, _mph, _buried, lamps, _country, gzoom = self.geo_rules(gx, gy, z)
        t0 = time.perf_counter()
        got = self.far_bld_pool.submit(blockworker.far_geo, kind, gx, gy, gzoom, self.out_dir, self.vfs_dir,
                                       lamps).result()
        log.debug("far geotile %s %s: %.2f s (queue + build)", kind, geotiles.key(gx, gy, z), time.perf_counter() - t0)
        if got is not None:
            if len(memo) > self.FAR_GEO_KEPT:
                memo.clear()
            memo[(kind, gx, gy)] = got
        return got

    def build_traffic_geo(self, gx, gy, far=False):
        """One visual-road geotile, on the existing worker pool; session files survive anchor changes."""
        if self.far_bld_pool is None or self.mapdata is None:
            return None
        memo = self.__dict__.setdefault("_traffic_geo_done", {})
        got = memo.get((gx, gy, far))
        if got is not None and os.path.exists(os.path.join(self.out_dir, os.path.basename(got[0]))):
            return got
        lh, mph, _buried, _lamps, _country, gzoom = self.geo_rules(gx, gy, distanttraffic.Z)
        got = self.far_bld_pool.submit(blockworker.traffic_geo, gx, gy, gzoom, self.out_dir, self.vfs_dir, lh, mph, far).result()
        if got is not None:
            if len(memo) >= 512:
                memo.clear()
            memo[(gx, gy, far)] = got
        return got

    def build_roads_geo(self, aid, gx, gy, content=True, skin=True, exact=None):
        """blockworker.roads_geo in a road process: geotile (gx, gy)'s content (once: kept by the mod across
        rebases), anchor aid's skin and exact variants. Its gas station pads kept for the ground's painters. None
        when the anchor is gone. Worker threads (roads.GeoTracker)."""
        anchor = self.anchors.get(aid)
        if anchor is None or self.mapdata is None or self.road_pool is None:
            return None
        lh, mph, buried, lamps, country, gzoom = self.geo_rules(gx, gy)
        fut = self.road_pool.submit(
            blockworker.roads_geo, aid, anchor.lat, anchor.lon, anchor.zooms[0], gx, gy, gzoom, self.out_dir,
            self.vfs_dir, self.sidewalks, DETAIL_ROAD_LIFT if self.near_visual == "detail" else 0.0,
            self.buildings, lh, mph, self.anchor_q(anchor) if skin else None, self.gas_stations, buried, lamps,
            country, content, skin, exact, _BuildMixin.pole_profile(self))
        t0, waited = time.perf_counter(), []
        try:
            res = poolsmod.result_running(fut, ROAD_TILE_TIMEOUT_S, waited)
            took = time.perf_counter() - t0
            if took > ROAD_GEO_SLOW_S:      # (nyc_speed.py's road lead: queue against build)
                st = {}
                for part in ((res or {}).get("stages") or {}).values():
                    for k, v in (part or {}).items():
                        st[k] = st.get(k, 0.0) + v
                top = " ".join(f"{k} {v:.1f}" for k, v in sorted(st.items(), key=lambda kv: -kv[1])[:5])
                tm = (res or {}).get("time") or {}
                log.info("road geotile %d_%d in %.1f s (queue %.1f; content %.1f s %.1f cpu, skin %.1f s %.1f cpu, "
                         "exact %.1f s): %s",
                         gx, gy, took, waited[0] if waited else 0.0, tm.get("content_s", 0.0),
                         tm.get("content_cpu", 0.0), tm.get("skin_s", 0.0), tm.get("skin_cpu", 0.0),
                         tm.get("exact_s", 0.0), top)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            log.warning("road geotile %d_%d: no answer from its worker in %.0f s: built again", gx, gy,
                        ROAD_TILE_TIMEOUT_S)
            if hasattr(self.road_pool, "recycle"):
                self.road_pool.recycle(f"road geotile {gx}_{gy} hung for {ROAD_TILE_TIMEOUT_S:.0f} s")
            raise
        if res is not None and res.get("content") is not None:
            self.keep_fuel_pads(res["content"][3])
        return res

    def geo_content_key(self, gx, gy):
        """The revisit key geotile (gx, gy)'s road content has with this server's settings (blockworker._geo_content_key,
        its own place's rules), or None (no map data)."""
        if self.mapdata is None:
            return None
        import geotiles
        lh, mph, buried, lamps, country, gzoom = self.geo_rules(gx, gy)
        return blockworker._geo_content_key(geotiles.geobox(gx, gy).key, gzoom, self.buildings, lh, mph,
                                            self.gas_stations, buried, lamps, country, md=self.mapdata.versions(),
                                            models_dir=self.root, pole_profile=_BuildMixin.pole_profile(self))

    def geo_prebuilt(self, gx, gy):
        """Is geotile (gx, gy)'s road content in the revisit cache (no worker, no network)? -> its key or None."""
        k = self.geo_content_key(gx, gy)
        return k if k is not None and os.path.exists(revisitmod._path(self.cache_dir, "roadsg", k)) else None

    def fast_precache(self, on):
        """The saved areas' fast mode (areas.py; the mod pauses the game meanwhile): road processes of its own for the
        world phase (FAST_POOL_*) and the full priority split (every core for the server, the game on its first
        ones), or both back. Any thread."""
        with self._fast_lock:
            if on:
                if self.fast_pool is None and self.make_fast_pool is not None:
                    n = fast_pool_size()
                    try:
                        self.fast_pool = self.make_fast_pool(n)
                        log.info("fast pre-cache: %d road processes of its own", n)
                    except Exception as e:      # (no processes: the road pool builds the world phase)
                        log.warning("fast pre-cache: no road processes of its own: %s: %s", type(e).__name__, e)
                self.split_boost(True, "fast pre-cache")
                return
            pool, self.fast_pool = self.fast_pool, None
            if pool is not None:
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                log.info("fast pre-cache: its road processes closed")
            self.split_boost(False, "fast pre-cache")

    def prebuild_geo(self, gx, gy):
        """Saved areas' world phase: geotile (gx, gy)'s road content built into the revisit cache in a road process
        (blockworker.prebuild_geo), the game sent nothing. -> its revisit key, or None (not complete: tried later)."""
        if self.mapdata is None or self.road_pool is None or not revisitmod.ON:
            return None
        import roadz
        lh, mph, buried, lamps, country, gzoom = self.geo_rules(gx, gy)
        # (beside the session folders, as a session's: a 3dmr.eu model baked there finds the models' cache, its textures'
        # game path and the mod's materials file where the game's own tiles do: buildings._custom_tile_entries, PR #626)
        scratch = os.path.join(self.root, "areas_scratch")
        fast = self.fast_pool           # (fast mode: its own processes, the game paused: no need to stand back)
        submit = fast.submit if fast is not None else getattr(self.road_pool, "submit_low", self.road_pool.submit)
        # (the fast mode's own processes may wait for a cell's lidar; the live pool keeps its short wait)
        timeout = (max(ROAD_TILE_TIMEOUT_S, roadz.PRECACHE_LIDAR_WAIT_S + 120.0) if fast is not None
                   else ROAD_TILE_TIMEOUT_S)
        return poolsmod.result_running(submit(
            blockworker.prebuild_geo, gx, gy, gzoom, scratch, self.buildings, lh, mph, self.gas_stations, buried,
            lamps, country, "/temp/beamngEarth/areas_scratch", pole_profile=_BuildMixin.pole_profile(self),
            wait_gaps=fast is not None), timeout)

    def build_roads_tile(self, aid, tx, ty):
        """roads.py: one road tile of anchor aid, written to the session folder. Returns (vfs path,
        chunks) or None if the anchor is gone. Worker threads (fetches the map cells it lacks)."""
        import poleprofiles
        anchor = self.anchors.get(aid)
        if anchor is None or self.mapdata is None:
            return None
        if self.road_pool is not None:
            t0 = time.perf_counter()
            fut = self.road_pool.submit(
                blockworker.roads_tile, aid, anchor.lat, anchor.lon, anchor.zooms[0], tx, ty, self.out_dir,
                self.vfs_dir, self.sidewalks, DETAIL_ROAD_LIFT if self.near_visual == "detail" else 0.0,
                self.buildings, self.left_hand(anchor), self.mph(anchor), self.anchor_q(anchor),
                self.gas_stations, self.buried(anchor), self.lamp_region(anchor), self.country(anchor), _BuildMixin.pole_profile(self, anchor))
            try:
                # bounded: a worker that never answered held this thread and its roads.Tracker slot for good (4
                # slots: four hung tiles and no road tile ever again, the loading line at "roads & buildings 0/26")
                res = poolsmod.result_running(fut, ROAD_TILE_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                fut.cancel()
                log.warning("road tile %d_%d of anchor %d: no answer from its worker in %.0f s: built again",
                            tx, ty, aid, ROAD_TILE_TIMEOUT_S)
                if hasattr(self.road_pool, "recycle"):
                    self.road_pool.recycle(f"road tile {tx}_{ty} hung for {ROAD_TILE_TIMEOUT_S:.0f} s")
                raise
            log.debug("road tile %d_%d: %.2f s (queue + build)", tx, ty, time.perf_counter() - t0)
            self.keep_fuel_pads(res[2])
            return res[:2]
        self.mapdata.warm(roadsmod.tile_needs(anchor.frame, tx, ty, self.buildings, self.road_widths is not None,
                                              1 + seacarvemod.LINE_CELLS if self.sea is not None else 0))
        cells = [self.mapdata.cell(k, layers=("roads",), kind="demand").layers.get("roads")
                 for k in roadsmod.cells_of_tile(anchor.frame, tx, ty, 200 if poleprofiles.kit_enabled(_BuildMixin.pole_profile(self, anchor)) else roadsmod.OVERLAP)]
        zoom = anchor.zooms[0]
        urban = self.surface.urban if self.surface is not None and hasattr(self.surface, "urban") else None
        g = self.grader_for(anchor)

        def elev(la, lo):           # the physics ground (lifted under roads: grading.ROAD_LIFT)
            z = self.ground(anchor.frame, la, lo, self.dem.sample(la, lo, zoom))
            return g.apply(la, lo, z, gradingmod.ROAD_LIFT) if g is not None else z
        walks = []
        with self.grader_lock:
            if anchor.bridges is None:
                anchor.bridges = self.chain_view(anchor, "bridges", elev)
            if anchor.tunnels is None:
                anchor.tunnels = self.chain_view(anchor, "tunnels", elev, g)
        fuel, skip, fuel_cuts, fuel_pads = [], frozenset(), [], []
        if self.gas_stations:       # pumps and refuel areas; their canopy outlines left out of the buildings
            fuel, skip = gasstationsmod.tile_sites(anchor.frame, tx, ty, self.mapdata, elev, cuts=fuel_cuts,
                                                   pads=fuel_pads)
            self.keep_fuel_pads(fuel_pads)
        props = [] if self.buildings else None
        lines = [] if self.buildings else None
        wires = [] if self.buildings else None
        park = [] if self.buildings else None
        signs = [] if self.buildings and poleprofiles.kit_enabled(_BuildMixin.pole_profile(self, anchor)) else None
        items = town = None
        wrows = []
        _bridge_fps, _bridge_ends = [], {}      # 3dmr model bridges over the tile (as blockworker._tile_parts)
        if self.buildings:
            # stock towers, containers, billboards and barns; a replaced barn's id is left out of the buildings
            try:
                import worldprops
                wrows, barn_ids = worldprops.tile_props(anchor.frame, tx, ty, self.mapdata, elev)
                skip = frozenset(skip) | frozenset(barn_ids)
            except Exception as ex:
                log.warning("world props: tile %s_%s: %s: %s", tx, ty, type(ex).__name__, ex)
                wrows = []
            # (on the graded ground, as blockworker.roads_tile)
            items, _custom = buildingsmod.road_tile_items(anchor.frame, tx, ty, self.mapdata, elev,
                                                 decks=blockworker._tile_decks(anchor.bridges, cells, tx, ty), skip=skip,
                                                 cache_dir=os.path.dirname(self.out_dir))
            _bridge_fps = buildingsmod.bridge_footprints(anchor.frame, tx, ty, os.path.dirname(self.out_dir))
            if anchor.bridges is not None and any(m in {c["id"] for c in _custom} for _fp, _c, m in _bridge_fps):
                try:
                    _bridge_ends = buildingsmod.bridge_road_ends(_bridge_fps, anchor.bridges, cells)
                except Exception as ex:
                    log.warning("3dmr: mapped bridge ends: %s: %s", type(ex).__name__, ex)
            if wrows:
                props.extend(wrows)
            town = buildingsmod.coverage(items) < buildingsmod.POLES_COVERAGE
        rwalls = []
        ctx = []
        box = roadsmod.tile_bbox(tx, ty)
        if props is not None:                       # (the props' side sources, read while the tile is built)
            self.mapdata.side_prefetch(roadsmod.cells_of_tile(anchor.frame, tx, ty))
        cell_fn = lambda key: self.mapdata.cell(key, layers=("roads",), kind="demand").layers.get("roads")  # noqa: E731

        def drawn(la, lo):      # the block's ground: raised over the low arches as the terrain is (_make_heightmap_here)
            return roadsmod.block_arch_raise(anchor.tunnels, cell_fn, anchor.frame, la, lo, elev(la, lo),
                                             (box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8))
        cover = []                      # the ground carriageways for the decks' parapets (roadmesh.deck ground=)
        marks = []                  # stall lines along parking aisles (parking.finish_markings adds the lot polygons')
        near_roads = []             # the ground carriageways near the tile (the median barriers': medians.tile_rows)
        nkeys = roadsmod.cells_of_tile(anchor.frame, tx, ty, blockworker.SIGNALS_MARGIN_M)
        sign_nodes = secrossings.read(self.mapdata, anchor.frame, nkeys) if str(self.country(anchor) or "").upper() == "SE" \
            else None
        clearance = [] if self.buildings else None     # (build_tile's RoadClearance, PR #629: the powerlines' too)
        roadside_jobs = []
        drives = []
        import euguardrails
        sign_obstacles=None
        if self.buildings and euguardrails.enabled():
            import signplacement
            sign_obstacles=signplacement.for_tile(anchor.frame,tx,ty,self.mapdata,items,skip)
        town_faces = {}                 # the town signs' faces the tile's props use (BNE_TOWNSIGNS)
        towns = None
        if props is not None and townentrymod.ON and self.mapdata is not None and self.surface is not None:
            if getattr(self, "towns", None) is None:
                self.towns = townentrymod.Towns(self.mapdata, self.surface)
            towns = self.towns.maker(anchor.frame, roadsmod.cells_of_tile(anchor.frame, tx, ty,
                                                                          blockworker.TOWNS_MARGIN_M),
                                     self.out_dir, self.vfs_dir, town_faces)
        chunks = roadsmod.build_tile(anchor.frame, tx, ty, cells, elev, urban, walks=walks, deck_context=ctx, bridges=anchor.bridges,
                                     sign_obstacles=sign_obstacles,
                                     walk_cuts=fuel_cuts,    # (the stations' made-up driveways: no sidewalk across them)
                                     widths=self.road_widths, props=props, lines=lines, wires=wires, town=town, park=park,
                                     left_hand=self.left_hand(anchor), mph=self.mph(anchor), buried=self.buried(anchor),
                                     lamp_region=self.lamp_region(anchor), country=self.country(anchor), pole_profile=_BuildMixin.pole_profile(self, anchor), signs=signs,
                                     structures=buildingsmod.centres(items, tx, ty) if items else (),
                                     tunnels=anchor.tunnels, walls=rwalls,
                                     ground=lambda x, y: roadmeshmod.drawn_ground(anchor.frame, drawn, np.c_[x, y],
                                                                                  self.anchor_q(anchor)),
                                     terrain_paint=self.surface is not None, cover_out=cover, marks=marks,
                                     roads_out=near_roads, roadside_jobs=roadside_jobs, signals=furnituremod.signals_for(self.mapdata, anchor.frame, nkeys),
                                     nodes=sign_nodes, clearance_out=clearance,
                                     development=developmentmod.for_tile(self.mapdata, anchor.frame, tx, ty), towns=towns)
        # (railway and tram tracks on the ground)
        chunks += railsmod.tile_chunks(anchor.frame, tx, ty, cells, self.country(anchor), urban)
        if self.airports is not None:   # runway markings, drawn by the road mesh like the lane lines
            chunks += self.airports.tile_chunks(anchor.frame, tx, ty, elev, roadsmod.tile_bbox)
        x0, y0, _, _ = roadsmod.tile_bbox(tx, ty)
        if items is not None:           # front doors and shopfronts, driveways into the road mesh
            # (no street tree in a house: its own or the next tile's)
            rings = buildingsmod.near_rings(anchor.frame, tx, ty, self.mapdata, skip=skip)
            props[:] = furnituremod.off_buildings(props, rings)
            props.extend(buildingsmod.doors(items, lines, x0, y0, drives))
            props.extend(buildingsmod.toll_booth_rows(items, x0, y0))
            drives.extend(buildingsmod.garage_drives(items, lines, x0, y0))
            chunks += drives
            park = parkingmod.clear_of(park, props, drives)
            park = parkingmod.tile_lot_spots(anchor.frame, tx, ty, self.mapdata, elev, items, park)
            park = gasstationsmod.clear_park(park, fuel)
            buildingsmod.brick_walks(walks, items, anchor.lat, anchor.lon)
        chunks += parkingmod.finish_markings(anchor.frame, tx, ty, self.mapdata, elev, items, marks)
        if _bridge_fps:                 # (the model bridge is the road there: its deck slab)
            chunks = buildingsmod.bridge_stub_chunks(_bridge_fps, chunks)
            ctx[:] = buildingsmod.bridge_stub_chunks(_bridge_fps, ctx)
            if props is not None:
                props[:] = buildingsmod.bridge_stub_props(_bridge_fps, props)

        # Share the bridge mesh's exact parapet geometry with the tile builder.
        # Mapped rails on a deck are replaced by guardrails.bridge_rows(); the parapet runs also feed the fallback matching.
        pads_near = []
        if anchor.bridges is not None:  # (the junction pads round the tile: roads.deck_pads)
            try:
                pad_parts = list(anchor.bridges.parts(cells, (box[0] - 40, box[1] - 40, box[2] + 40, box[3] + 40)))
                pad_parts = roadsmod.shifted_parts(pad_parts, roadsmod.deck_shift_fn(
                    anchor.frame, cells, urban, self.road_widths, self.left_hand(anchor)))
                pad_parts = buildingsmod.bridge_stub_parts(_bridge_fps, pad_parts)
                pads_near = roadsmod.deck_pads([(f, sp, xy, z) for f, sp, xy, z, _lon in pad_parts], near_roads, box,
                                               anchor.bridges._ground, own=False)
            except Exception as ex:
                log.warning("deck pads: tile %s_%s: %s: %s", tx, ty, type(ex).__name__, ex)
        pads = [p for p in pads_near if roadsmod.box_owns(box, float(p[1][0]), float(p[1][1]))]
        parapets = roadmeshmod.parapet_runs(chunks, ctx, ground=cover, pads=pads_near)
        import guardrails as guardrailsmod
        infra_fn = lambda key: self.mapdata.cell(key, layers=("infrastructure",), kind="demand").layers.get("infrastructure")
        low_parapet_decks = set()
        deck_parts = []                 # the bridges' parts near the tile (the median barriers' on joined decks too)
        guardrail_decks = []
        rail_decks = []                 # railways.tile_mesh fills these in below (fences.py stands walls on them)
        try:
            deck_parts = list(anchor.bridges.parts(cells, (box[0]-10, box[1]-10, box[2]+10, box[3]+10)))
            deck_parts = buildingsmod.bridge_stub_parts(_bridge_fps, deck_parts)
            guardrail_decks = [{"n": np.column_stack((xy, z)), "w": spec.width}
                               for _fid, spec, xy, z, _lon in deck_parts]
            guardrail_rows = guardrailsmod.forest_rows(anchor.frame, infra_fn, box, elev,
                                                       decks=guardrail_decks, parapets=parapets,
                                                       bridge_rails=roadmeshmod.BRIDGE_RAILS,
                                                       supported_decks_out=low_parapet_decks, roads=near_roads)
        except Exception as ex:
            log.warning("mapped guardrails: tile %s_%s: %s: %s", tx, ty, type(ex).__name__, ex)
            # A failed mapped-data pass cannot establish that a bridge needs rails.
            low_parapet_decks.clear()
            guardrail_rows = []
        import medianbarriers
        barrier_mesh, barrier_chunks = medianbarriers.tile(
            self.mapdata, anchor.frame, tx, ty, elev, urban, self.road_widths, self.left_hand(anchor))
        guardrail_rows = medianbarriers.mapped_rows(guardrail_rows, barrier_chunks)
        guardrail_rows = guardrail_rows + blockworker._median_rows(
            anchor.frame, near_roads, box, elev, self.left_hand(anchor), self.country(anchor), guardrail_rows, tx, ty,
            deck_parts, barrier_chunks)
        wide = {}                       # (the wider cells' ways, both of these want the same)
        guardrail_rows = guardrail_rows + blockworker._motesfri_rows(
            anchor.frame, cell_fn, near_roads, box, elev, urban, self.road_widths, self.left_hand(anchor),
            self.country(anchor), guardrail_rows, tx, ty, chunks, wide)
        guardrail_rows = guardrail_rows + blockworker._kantstolpe_rows(
            anchor.frame, near_roads, box, elev, self.country(anchor), guardrail_rows, tx, ty)
        guardrail_rows = guardrail_rows + blockworker._sign_rows(
            anchor.frame, cell_fn, near_roads, sign_nodes, box, elev, urban, self.road_widths, self.left_hand(anchor),
            self.country(anchor), tx, ty, wide)
        for finish in roadside_jobs:
            finish(guardrail_rows, [(np.asarray(ch['n']), ch['w'] / 2) for ch in drives if 'n' in ch and 'w' in ch])
        mesh = roadmeshmod.build(anchor.frame, chunks, elev, (x0, y0, 0.0),
                                 DETAIL_ROAD_LIFT if self.near_visual == "detail" else 0.0,
                                 clip=box, follow=True, q=self.anchor_q(anchor), drawn=drawn)
        mesh = blockworker._with_plazas(mesh, anchor.frame, roadsmod.tile_bbox(tx, ty), elev, (x0, y0, 0.0),
                                        self.surface)
        _piers, _beams = buildingsmod.bridge_stub_piers(
            _bridge_fps,
            anchor.bridges.piers_in(roadsmod.tile_bbox(tx, ty)) if anchor.bridges is not None else (),
            anchor.bridges.beams_in(roadsmod.tile_bbox(tx, ty)) if anchor.bridges is not None else ())
        dmesh = roadmeshmod.deck(chunks, (x0, y0, 0.0), _piers, ctx, _beams,
                                 ground=cover, parapets=parapets,
                                 low_parapet_decks=low_parapet_decks, pads=pads)
        tmesh = roadmeshmod.tunnel(chunks, (x0, y0, 0.0), rwalls)
        import railways
        rmesh, rail_decks = railways.tile_mesh(anchor.frame, tx, ty,
                                   [cell_fn(k) for k in roadsmod.cells_of_tile(anchor.frame, tx, ty, railways.REACH_M)],
                                   railways.point_ground(anchor.frame, elev, (aid, anchor.lat, anchor.lon)),
                                   far_ground=railways.point_ground(
                                       anchor.frame,
                                       lambda la, lo: self.ground(anchor.frame, la, lo, self.dem.sample(la, lo, zoom)),
                                       ("raw", aid, anchor.lat, anchor.lon)),
                                   return_decks=True)
        bld = None
        garage_sound = ()
        oil_wells = []
        custom_entries = []             # the tile's baked 3dmr.eu / local models (roads.write_tile's "b3dmr")
        if self.buildings:
            import oilwells
            _baked_entries, _baked_radii = [], []
            if _custom and self.out_dir and self.vfs_dir:
                _baked_entries, _baked_radii, _boxes = buildingsmod.custom_models(
                    anchor.frame, _custom, elev, self.out_dir, self.vfs_dir, road_ends=_bridge_ends)
                custom_entries = _baked_entries
            bld, _ = buildingsmod.road_tile(anchor.frame, tx, ty, self.mapdata, elev, None, None,
                                            items, entries=_baked_entries, radii=_baked_radii)
            garage_sound = buildingsmod.garage_sound_boxes(items, *roadsmod.tile_bbox(tx, ty)[:2],
                                                           roadsmod.SOUND_GARAGE)
            oil_wells = oilwells.tile_wells(anchor.frame, tx, ty, elev)     # (the mod's pump jacks)
            import powerlines
            powerlines.tile_lines(anchor.frame, tx, ty, self.mapdata, elev, props, wires,
                                  clearance=clearance[0] if clearance else None)    # high-voltage lines
            import fences
            try:
                props.extend(fences.rows(anchor.frame, infra_fn, box, elev, decks=[
                    {"n": np.column_stack((xy, z)), "w": spec.width} for _f, spec, xy, z, _l in deck_parts] + rail_decks))
            except Exception as ex:
                log.warning("fences: tile %s_%s: %s: %s", tx, ty, type(ex).__name__, ex)
            import ign_limites
            try:
                props.extend(ign_limites.rows(anchor.frame, getattr(self.mapdata, "ign_limites", None), box, elev))
            except Exception as ex:
                log.warning("ign_limites: tile %s_%s: %s: %s", tx, ty, type(ex).__name__, ex)
            import picnic
            land_fn = lambda key: self.mapdata.cell(key, layers=("land_use",), kind="demand").layers.get("land_use")
            bld_fn = lambda key: self.mapdata.cell(key, layers=("buildings",), kind="demand").layers.get("buildings")
            osm = getattr(self.mapdata, "picnic_osm", None)
            try:
                props.extend(picnic.rows(anchor.frame, infra_fn, land_fn, box, elev,
                                         (lambda keys: osm.cells(keys, wait=False)) if osm is not None else None,
                                         bld_fn))
            except Exception as ex:
                log.warning("picnic: tile %s_%s: %s: %s", tx, ty, type(ex).__name__, ex)
            import shelters
            try:
                props.extend(shelters.rows(anchor.frame, infra_fn, box, elev, near_roads))
            except Exception as ex:
                log.warning("shelters: tile %s_%s: %s: %s", tx, ty, type(ex).__name__, ex)
            import industry
            garage_sound = (garage_sound + industry.refinery_sound_rows(anchor.frame, tx, ty, self.mapdata, elev)
                            + industry.power_plant_sound_rows(anchor.frame, tx, ty, self.mapdata, elev))
        import windturbines
        turbine_rows = windturbines.tile_turbines(anchor.frame, tx, ty, elev)
        if self.mapdata is not None:
            import golf
            try:
                land_fn = lambda key: self.mapdata.cell(key, layers=("land_use",), kind="demand").layers.get("land_use")
                flag_rows = golf.rows(anchor.frame, land_fn, box, elev)
                if flag_rows:
                    props = (props or []) + flag_rows
            except Exception as ex:
                log.warning("golf: tile %s_%s: %s: %s", tx, ty, type(ex).__name__, ex)
        return roadsmod.write_tile(self.out_dir, self.vfs_dir, aid, tx, ty, chunks, walks if self.sidewalks else None,
                                   mesh, dmesh, bld, props, wires, park, tunnel_mesh=tmesh, fuel=fuel,
                                   rail_mesh=rmesh, guardrail_rows=guardrail_rows, garage_sound=garage_sound,
                                   custom_entries=custom_entries, signs=signs, wells=oil_wells, bld_none=bool(self.buildings),
                                   barrier_mesh=barrier_mesh, barrier_chunks=barrier_chunks,
                                   turbines=turbine_rows, towns=town_faces)

    def _osm_fuel_warm(self, sleep=time.sleep):
        """Worker thread: the OSM fuel file's index read at the start, tried again every OSM_FUEL_WARM_WAIT_S when the
        host turns it away (in game, 2026-10-29: "busy, prefetch dropped", "bucket low" while the start's road tags
        took both slots), up to OSM_FUEL_WARM_TRIES times (then the first cell reads it)."""
        for i in range(OSM_FUEL_WARM_TRIES):
            try:
                t0 = time.perf_counter()
                self.mapdata.osm_fuel_warm()
                log.info("osm fuel: the file's index read in %.1f s", time.perf_counter() - t0)
                return
            except Exception as e:
                log.info("osm fuel: the file's index (try %d): %s: %s", i + 1, type(e).__name__, e)
                if i + 1 < OSM_FUEL_WARM_TRIES:
                    sleep(OSM_FUEL_WARM_WAIT_S)

    def osm_fuel_want(self, conn, aid, tiles):
        """The OpenStreetMap fuel cells of road tiles of anchor aid (tx, ty), or of road geotiles (gx, gy) when conn's
        tracker is roads.GeoTracker (their stations' reach: osm_fuel_cells), fetched in the background, each once; one
        with stations that comes after conn sent a tile it reaches has that tile built again (osm_fuel_tiles,
        roads.Tracker.again / GeoTracker.again)."""
        anchor = self.anchors.get(aid)
        if self.osm_fuel is None or anchor is None or not self.gas_stations:
            return
        geo = getattr(getattr(conn, "roads", None), "geo", False)
        loop = asyncio.get_running_loop()
        for t in tiles:
            for key in osm_fuel_cells(anchor.frame, t, geo):
                k = str(key)
                if k in self.osm_fuel_done:
                    continue
                waiting = self.osm_fuel_wait.get(k)
                if waiting is not None:
                    waiting.add((conn, aid))
                    continue
                self.osm_fuel_wait[k] = {(conn, aid)}
                fut = self.osm_fuel.submit(self._osm_fuel_cell, key)
                fut.add_done_callback(lambda f, k=k, key=key: loop.call_soon_threadsafe(self._osm_fuel_got, k, key, f))

    def _osm_fuel_cell(self, key):
        """Worker thread: the cell's OSM fuel stations, from the disk when fetched before."""
        rows = self.mapdata.osm_fuel_cached(key)
        if rows is not None:
            return rows, False
        t0 = time.perf_counter()
        rows = self.mapdata.osm_fuel_fetch(key)
        log.debug("osm fuel: cell %s, %d stations in %.2f s", key, len(rows), time.perf_counter() - t0)
        return rows, True

    def _osm_fuel_got(self, k, key, fut):
        waiting = self.osm_fuel_wait.pop(k, set())
        try:
            rows, fetched = fut.result()
        except Exception as e:          # (asked again with the next pose that wants it)
            log.debug("osm fuel: cell %s: %s: %s", k, type(e).__name__, e)
            return
        self.osm_fuel_done.add(k)
        if not rows or not fetched:     # (a cell from the disk was there when its tiles were built)
            return
        for conn, aid in waiting:
            anchor = self.anchors.get(aid)
            roads = getattr(conn, "roads", None)
            geo = getattr(roads, "geo", False)
            # (a geotile's content is the connection's, whatever the anchor: dev's geotiles, kept by the mod across
            # anchors; an anchor's road tiles only while it is the tracker's)
            if anchor is not None and roads is not None and (geo or roads.aid == aid):
                again = osm_fuel_tiles(anchor.frame, key, roads.sent, geo)
                if again:
                    log.info("osm fuel: cell %s brought %d stations: road tiles %s again", k, len(rows), sorted(again))
                    roads.again(again)

    def keep_fuel_pads(self, pads):
        """A road tile's gas station pads kept for the ground's painters (surface.Surface.add_fuel_pads); the blocks
        a new station's pad lies on get their photo rendered again (pads_stale)."""
        if pads and self.surface is not None and hasattr(self.surface, "add_fuel_pads"):
            new = self.surface.add_fuel_pads(pads)
            if new:
                self.pads_stale(new)

    def pads_stale(self, items=None):
        """The built blocks whose photo was painted without these pads [(id, rings)] (every built block: None)
        marked stale, so base_refresh renders it again and sends it (block_base): the ground under the car is built
        before its road tiles bring their stations' pads (a teleport to a station showed its forecourt bare: the
        user, 2026-10-24, "No asphalt at the station"); in detail mode (the block photo) only. The same blocks are
        marked for building again (pads_rebuild: their surface layers, the asphalt and no grass, come with the block;
        Connection.pads_refresh). A block still being built (hm_stage) is marked too: its build took the pads of its
        start (after a teleport, a station's pad came 0.1 s before the block under the car was done and its lot stayed
        dirt: the user, 2026-10-27, "the gasstation lot never gets painted")."""
        self.pads_changed_at = time.monotonic()
        h = self.layout.block * heightmap.SQUARE / 2
        for key in list(self.hm) + list(self.hm_stage):
            anchor = self.anchors.get(key[0])
            if anchor is None:
                continue
            _, cx, cy = key
            if items is None or any(self._ring_meets(anchor, ring, cx - h, cy - h, cx + h, cy + h)
                                    for _, rings in items for ring in rings):
                self.pads_rebuild.add(key)
                if self.near_visual == "detail":
                    self.base_stale.add(key)

    @staticmethod
    def _ring_meets(anchor, ring, x0, y0, x1, y1):
        la, lo = ring
        e, n, _ = anchor.frame.geodetic_to_enu(np.asarray(la), np.asarray(lo), np.zeros(len(la)))
        e, n = np.asarray(e), np.asarray(n)
        return bool(e.max() >= x0 and e.min() <= x1 and n.max() >= y0 and n.min() <= y1)

    def block_fuel_pads(self, anchor, cx, cy, size):
        """The kept gas station pads around a block (its square and a margin) for its process's job: [(id, rings)]."""
        if self.surface is None or not getattr(self.surface, "gas_pads", False):
            return []
        h = size / 2 + 64.0
        la, lo, _ = anchor.frame.enu_to_geodetic(np.array([cx - h, cx + h]), np.array([cy - h, cy + h]), np.zeros(2))
        return self.surface.fuel_pad_items(np.asarray(la), np.asarray(lo))

    def pole_profile(self, anchor=None):
        """Mode/data token only; roads resolve country at each pole, not at the anchor."""
        import poleprofiles
        return poleprofiles.geographic_profile(getattr(self, "regional_poles", "off"))
