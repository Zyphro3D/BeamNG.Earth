"""World server: far tiles, their textures and the mod's slots (split out of earthd.py's Server): tile names and
levels, horizon grids, sharing, making and retexturing tiles, road masks, world warm-up, slot pools.
"""
import os
import asyncio  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import replace  # noqa: E402
import numpy as np  # noqa: E402
import blockworker  # noqa: E402
import cdae  # noqa: E402
import credits as creditsmod  # noqa: E402
import surface as surfacemod  # noqa: E402
import roads as roadsmod  # noqa: E402
import urgency as urgencymod  # noqa: E402
import streamleases
import plancache as plancachemod  # noqa: E402
import grading as gradingmod  # noqa: E402
import revisit as revisitmod  # noqa: E402
import farfield  # noqa: E402
import seacarve as seacarvemod  # noqa: E402
import rockface as rockfacemod  # noqa: E402
import geotiles as geotilesmod  # noqa: E402
import staging  # noqa: E402
import imagery as imagerymod  # noqa: E402
import matslots as matslotsmod  # noqa: E402
import updates as updatesmod  # noqa: E402
import netfetch  # noqa: E402
import weather as weathermod  # noqa: E402
from earthbase import (
    ANCHOR_CENTRE_MAX,
    BLOCK_DRAWN,
    ELEVATION_CREDIT,
    FAR_TRACE,
    FIN_WORKER,
    FIN_WORKER_MIN_TRIES,
    FIN_WORKER_PROBE,
    FIN_WORKER_SHARE,
    FLY_TEX_EYE_M,
    GLOBAL_AID,
    HORIZON_BLOCK_CACHE,
    HORIZON_RETRY,
    MAP_RETEX_BOOST,
    MAP_SCREEN_H,
    MAP_TEX_MAX_LEVEL,
    MAP_TEX_MIN_PX,
    MAP_TEX_OFF_PX,
    MAP_TEX_OVERSAMPLE,
    MAP_TEX_PX,
    PIECE_PARTS,
    RETEX_REPEAT,
    TileInfo,
    WORLD_WARM_CONC,
    WORLD_WARM_Z,
    _BoundedLru,
    _app_modules,
    log,
)


class _FarMixin:
    """Server's far tiles, textures and slots (split out of earthd.py; Server inherits it)."""

    # --- far tiles -------------------------------------------------------
    def tile_name(self, aid, key):
        return f"far_{aid}_{key}.{self.far_format}"

    def fin_worker_worth(self):
        """With the imagery analyses on: try this far tile's texture in its far-tile process (from the finished tiles
        on disk)? While most recent tries found them (a place seen before) yes; on a first visit (none finished yet:
        the try fails and the texture is rendered here after the geometry, not beside it) only every
        FIN_WORKER_PROBE-th tile, to notice when they are there."""
        if not FIN_WORKER:
            return False
        h = self.fin_worker_hits
        self.fin_worker_tries += 1
        if len(h) < FIN_WORKER_MIN_TRIES:
            return True
        return sum(h) >= FIN_WORKER_SHARE * len(h) or self.fin_worker_tries % FIN_WORKER_PROBE == 0

    def attribution_rest(self):
        """The credit line's sources other than the imagery (weather, elevation)."""
        parts = [weathermod.CREDIT] if self.weather is not None else []
        return "; ".join(parts + [ELEVATION_CREDIT])

    def attribution(self):
        """The credit line the mod shows: imagery (if on) and elevation sources."""
        parts = [self.imagery.attribution()] if self.imagery is not None else []
        if self.weather is not None:
            parts.append(weathermod.CREDIT)
        return "; ".join(parts + [ELEVATION_CREDIT])

    def net_note(self):
        """The network state for the status text: "" when all is well, "Offline - using cached data"
        (--offline, or every imagery / DEM host unreachable), else the unreachable hosts and the
        tiles drawn without imagery. Hosts the world builds without (gravity, weather, search, routing, the update
        check) are not named: a player in a country that blocks SourceForge saw "Degraded" for good (GitHub #23)."""
        if self.net.offline:
            return netfetch.OFFLINE_TEXT
        parts = []
        down = self.net.unavailable_hosts(optional=False)
        if down:            # each with what it affects (netfetch.ROLES: the user, 2026-10-27)
            parts.append("; ".join(f"{h} unreachable ({netfetch.role_of(h)})" if netfetch.role_of(h)
                                   else f"{h} unreachable" for h in down))
        if self.degraded:
            parts.append(f"{len(self.degraded)} tiles without imagery")
        if self.net.slow() is not None:     # (map data at a crawl: a new place's roads take minutes: GitHub #241;
            # no number in it: the note is re-sent whenever its text changes)
            parts.append("slow internet connection: new places take a few minutes to load")
        disk = getattr(self, "disk_cache", None)
        if disk is not None and disk.managed and disk.disk_short() > 0:
            # (a full drive fails far tiles, blocks and caches with "No space left on device": the first week's most
            # frequent errors; the cache is cut back to its reserve, but other files may fill the drive)
            parts.append("your drive is nearly full: free some space (the download cache is cut back)")
        return "Degraded: " + "; ".join(parts) if parts else ""

    def level_summary(self):
        """'L<level> <geometry ms>/<texture ms> <shape KB>/<texture KB>' medians, for the log."""
        out = []
        for lvl, st in sorted(self.level_stats.items()):
            # list() copies the deque in one step: tile workers append to it meanwhile, and a
            # Python-level iteration over it could raise "deque mutated during iteration" here
            # (in the connection's streamer, which would stop streaming).
            st = list(st)
            med = [sorted(v[k] for v in st)[len(st) // 2] for k in range(4)]
            out.append(f"L{lvl} {1000 * med[0]:.1f}/{1000 * med[1]:.1f} {med[2] / 1024:.0f}/{med[3] / 1024:.0f}")
        return ", ".join(out)

    def texture_name(self, aid, key, rev=0):
        """A tile's texture file; one rendered again (--colormatch, rev > 0) gets a new name: the engine
        caches textures by path, and the mod points the tile's material at the new file."""
        return f"far_{aid}_{key}.dds" if rev == 0 else f"far_{aid}_{key}_r{rev}.dds"

    def tile_files(self, name, texture=None, rev=0):
        """A far tile's files: the shape, its texture (`texture`: a re-rendered one's name), every
        earlier revision's texture up to `rev` (a tile retextured live left them behind) and the
        engine's .cdae cache of it (DAE tiles)."""
        stem = os.path.splitext(name)[0]
        out = [os.path.join(self.out_dir, name), os.path.join(self.out_dir, stem + ".dds"),
               os.path.join(self.out_dir, stem + "_road.dds")]
        out += [os.path.join(self.out_dir, f"{stem}_r{r}.dds") for r in range(1, rev + 1)]
        if texture is not None and os.path.join(self.out_dir, texture) not in out:
            out.append(os.path.join(self.out_dir, texture))
        if name.endswith(".dae"):
            out.append(os.path.join(self.cdae_dir, os.path.splitext(name)[0].lower() + ".cdae"))
        return out

    def tex_size(self, level):
        """Texture edge of a level (the layout's per-level table: with 32 cells the two finest
        levels of the mode get tex_near, the others tex_far; horizon mode: farfield.HORIZON_TEX from L8). With the 3D
        map open, at least MAP_TEX_PX (the user, 2026-11-07: "i want the ground to look high res at every zoom level":
        256 px tiles were 4-8 m texels under ~2.4 m screen pixels from 3 mi up)."""
        if self.horizon and level in farfield.HORIZON_TEX:
            px = farfield.HORIZON_TEX[level]
        else:
            px = self.layout.tex_size(level)
        if level <= MAP_TEX_MAX_LEVEL and any(getattr(c, "map_view", False) or getattr(c, "map_open", False)
                                              for c in self.connections):
            px = max(px, MAP_TEX_PX)
        return px

    def tile_tex_px(self, anchor, tile):
        """A far tile's texture size: tex_size(level), but with the 3D map open as fine as the tile is on screen (the
        user, 2026-11-07, "photo detail matched to the screen": the photo zoom goes by it, each halving 4x fewer photo
        tiles): one off the screen (behind the camera, outside its view cone) MAP_TEX_OFF_PX, else at most
        MAP_TEX_OVERSAMPLE x its edge in screen pixels (at MAP_SCREEN_H px for the field of view), never under
        MAP_TEX_MIN_PX. Retextured as the camera turns (a texture's size may change, its UVs go by its own)."""
        if isinstance(tile, farfield.Piece):
            tile = tile.base                # (a piece takes its node's texture: the same size)
        px = self.tex_size(tile.level)
        conn = None
        for c in self.connections:
            if c.active != anchor.id or not isinstance(getattr(c, "cam", None), dict):
                continue
            if (getattr(c, "map_view", False) or getattr(c, "map_open", False)) and c.cam.get("tx") is not None:
                conn = c
                break
            # (flying high too, a plane or a free camera: the same rule; driving keeps every texture its level's, the
            # camera turns round there in a moment)
            if c.cam.get("fz") is not None and c.eye_clearance() > FLY_TEX_EYE_M:
                conn = c
                break
        if conn is None:
            return px
        cam = conn.cam
        try:
            x, y, z = float(cam["x"]), float(cam["y"]), float(cam["z"])
            fx, fy, fz = float(cam["fx"]), float(cam["fy"]), float(cam["fz"])
            gz = float(cam["tz"]) if cam.get("tz") is not None else z - conn.eye_clearance()
            fov = float(cam.get("fovDeg") or 60.0)
        except (KeyError, TypeError, ValueError):
            return px
        x0, y0, x1, y1 = tile.bbox
        s = max(x1 - x0, y1 - y0)
        dx, dy, dz = (x0 + x1) / 2 - x, (y0 + y1) / 2 - y, gz - z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < s:
            return px                       # (under the camera)
        n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        tv = math.tan(math.radians(fov) / 2)
        half = math.atan(math.hypot(tv, tv * farfield.ASPECT))
        slack = math.asin(min(1.0, s * 0.7071 / d))
        ang = math.acos(max(-1.0, min(1.0, (dx * fx + dy * fy + dz * fz) / (d * n))))
        if ang - slack > half:
            return min(px, MAP_TEX_OFF_PX)
        on_screen = s * (MAP_SCREEN_H / (2 * tv)) / d
        want = 2 ** math.ceil(math.log2(max(1.0, on_screen * MAP_TEX_OVERSAMPLE)))
        return int(max(MAP_TEX_MIN_PX, min(px, want)))

    @property
    def max_level(self):
        """The quadtree's root level: L9 (v1.3.1), L10 in horizon mode."""
        return farfield.HORIZON_MAX_LEVEL if self.horizon else farfield.MAX_LEVEL

    # --- horizon (M1h) -------------------------------------------------------
    def make_horizon_grid(self, anchor, half):
        """Worker thread: the anchor's zmax grid over [-half, half]² (a coarse DEM pass)."""
        t0 = time.perf_counter()
        cache = _BoundedLru(self.horizon_blocks, HORIZON_BLOCK_CACHE)
        grid = farfield.horizon_grid(anchor.frame, self.dem, half, cache=cache)
        log.info("anchor %d horizon grid: %.0f km half-size, DEM z%d, %d cells, highest %.0f m, in %.2f s",
                 anchor.id, half / 1000, grid.zoom, grid.base.size, float(grid.base.max()), time.perf_counter() - t0)
        return grid

    def horizon_grid(self, anchor, h_eye):
        """The anchor's zmax grid, or None before its first one is built. Starts a build (on the event
        loop's executor) when there is none, or when the one it has is smaller than an eye at h_eye
        needs (an aircraft climbing); a finished build re-selects every connection on the anchor."""
        half = farfield.horizon_extent(h_eye, ANCHOR_CENTRE_MAX)
        have = self.horizon_grids.get(anchor.id)
        busy = self.horizon_inflight.get(anchor.id)
        failed = self.horizon_failed.get(anchor.id)
        if ((have is None or have.half < half) and (busy is None or busy[1] < half)
                and (failed is None or time.monotonic() - failed > HORIZON_RETRY)):
            if busy is not None:
                busy[0].cancel()            # a smaller build that has not started yet
            fut = asyncio.get_running_loop().run_in_executor(self.horizon_pool, self.make_horizon_grid, anchor, half)
            self.horizon_inflight[anchor.id] = (fut, half)
            fut.add_done_callback(lambda f, aid=anchor.id: self._horizon_done(aid, f))
        return have

    def _horizon_done(self, aid, fut):
        cur = self.horizon_inflight.get(aid)
        if cur is not None and cur[0] is fut:
            del self.horizon_inflight[aid]
        if fut.cancelled():
            return
        if fut.exception() is not None:
            self.horizon_failed[aid] = time.monotonic()
            log.warning("anchor %d: horizon grid failed (%r), retrying in %.0f s", aid, fut.exception(), HORIZON_RETRY)
            return
        grid = fut.result()
        if aid not in self.anchors:
            return                          # pruned meanwhile
        have = self.horizon_grids.get(aid)
        if have is None or have.half < grid.half:
            self.horizon_grids[aid] = grid
            self.horizon_failed.pop(aid, None)
            for c in list(self.connections):
                if c.active == aid:
                    c.update_far()

    def prefetch_imagery(self, anchor, tile):
        """Start downloading a queued tile's imagery (fire and forget), so tile workers rarely
        wait on the network. Only the first source of the chain; errors surface at build time."""
        if self.imagery is not None:
            frame, x0, y0, size = self.tile_where(anchor, tile)
            px = self.tile_tex_px(anchor, tile)
            self.imagery.prefetch(frame, *imagerymod.gutter_square(x0, y0, size, px), px, soon=True)

    # --- BeamMP shared world generation v2 (mpshare.py, .claude/work/mp_share_v2.md) --------------------------------
    def share_parts(self):
        """What the shared items' keys are made of (plans: their format, the signature's widths / sea; cell facts: theirs;
        the build switches; the map data's versions): equal for two apps, their items have the same keys."""
        md = self.mapdata.versions() if self.mapdata is not None and hasattr(self.mapdata, "versions") else None
        return {"plans": plancachemod.plans_format(), "facts": roadsmod.FACTS_FORMAT,
                "sw": revisitmod.switches_key(), "md": md, "widths": self.road_widths is not None,
                "sea": self.sea is not None}

    def share_store(self):
        """A PlanCache of this server's plans signature and map data (the hub's has / _kept / _revisit_key / expect: the
        road processes' own, plancache.for_grader)."""
        sig = f"w{int(self.road_widths is not None)}s{int(self.sea is not None)}"
        md = self.mapdata.versions() if self.mapdata is not None and hasattr(self.mapdata, "versions") else None
        pc = plancachemod.PlanCache(os.path.join(self.cache_dir, "plans",
                                                 os.environ.get(plancachemod.RUN_ENV) or "share"), sig)
        pc.persist = (self.cache_dir, (plancachemod.FRAMES, sig, md))
        return pc

    def addons(self):
        """The Add-ons page's service (addons.py: the registry, GitHub's pages for it, the installed add-ons), made at
        its first use."""
        if getattr(self, "_addons", None) is None:
            import addons as addonsmod
            feed = getattr(self, "feed", None)
            try:
                data_dir = _app_modules()[0].data_dir()
            except Exception:           # (no apppaths: beside the cache)
                data_dir = None
            self._addons = addonsmod.Addons(self.user_dir, self.cache_dir, self.net,
                                            version=updatesmod.local_version() or "", data_dir=data_dir,
                                            feed_url=f"http://127.0.0.1:{feed.port}" if feed is not None and
                                            getattr(feed, "port", None) else "",
                                            game_dir=os.environ.get("BNGEARTH_GAME_DIR") or None)
        return self._addons

    def share_hub(self, send):
        """The BeamMP sharing hub (one per server: the session's mod connection sends through it; a new connection takes
        it over). -> (hub, created now)."""
        import mpshare
        hub = getattr(self, "_share_hub", None)
        if hub is None:
            hub = self._share_hub = mpshare.Hub(send, self.cache_dir, self.share_store)
            log.info("share: hub up (BeamMP session: plans shared through the server)")
            return hub, True
        hub.send_mod = send
        return hub, False

    def remote_slots(self):
        """Far tiles compute sharing's helpers build at once (0 without helpers)."""
        f = getattr(self.far_pool, "remote_slots", None)
        return f() if f is not None else 0

    def far_ck(self, aid, key):
        """The server's key of far tile `key` shown under anchor aid: (GLOBAL_AID, key) for a global tile (the same
        for every anchor), else (aid, key)."""
        return (GLOBAL_AID, key) if farfield.is_global_key(key) else (aid, key)

    def tile_where(self, anchor, tile):
        """(frame, x0, y0, size) of the square a tile's texture is rendered over in that frame: the node's for a piece;
        a global tile's own frame and its (Mercator) square, about [-s/2, s/2]^2 there."""
        if isinstance(tile, farfield.GTile):
            return geotilesmod.frame(tile.x, tile.y, tile.z), -tile.size / 2, -tile.size / 2, tile.size
        node = tile.base if isinstance(tile, farfield.Piece) else tile
        x0, y0, _, _ = node.bbox
        return anchor.frame, x0, y0, node.size

    def render_tile(self, anchor, tile, px, pending):
        """(rgb, used): a far tile's photo with its gutter (a global tile's: its Mercator square, render_merc)."""
        if isinstance(tile, farfield.GTile):
            return self.imagery.render_merc(tile.z, tile.x, tile.y, px, pending=pending)
        frame, x0, y0, size = self.tile_where(anchor, tile)
        return self.imagery.render(frame, *imagerymod.gutter_square(x0, y0, size, px), px, pending=pending)

    def make_tile(self, anchor, tile, prev=None):
        urgencymod.wait_turn()          # (a block the game waits for first: urgency.py)
        with self.far_gate:
            return self._make_tile(anchor, tile, prev)

    def map_on(self):
        """Some connection has the 3D map open (its view frame or near the car)."""
        return any(getattr(c, "map_view", False) or getattr(c, "map_open", False) for c in self.connections)

    def worker_retex(self):
        """Re-render textures in the far-tile processes (raw photos): the 3D map open, or a camera flying high (the
        textures there upgraded as the view turns: tile_tex_px)."""
        return self.map_on() or any(c.eye_clearance() > FLY_TEX_EYE_M for c in self.connections
                                    if hasattr(c, "eye_clearance"))

    def far_limit(self):
        """Far tiles built at once (far_gate): far_drive while driving (the game needs its cores), every worker with
        the 3D map open (the game paused) or while the loading boost is on (the player waits)."""
        if any(getattr(c, "map_view", False) or getattr(c, "map_open", False) for c in self.connections) \
                or getattr(self, "boost_on", False):
            return self.far_max
        return self.far_drive

    def _make_tile(self, anchor, tile, prev=None):
        """Worker thread: build and write one far tile (and its texture). Returns its TileInfo.
        prev: the tile's stale TileInfo (--colormatch): only its texture is rendered again.
        A global tile (farfield.GTile) is built in its own frame, the same for every anchor (ck (GLOBAL_AID, key));
        a piece (farfield.Piece) is its node's tile cut to its global tile's quad."""
        # (the 3D map open: quick textures, what is on disk at once and the slow hosts' photos later: Imagery.quick)
        quick = self.imagery is not None and any(getattr(c, "map_view", False) or getattr(c, "map_open", False)
                                                 for c in self.connections)
        later = []
        if prev is not None:
            return self.retexture(anchor, tile, prev)
        t0 = time.perf_counter()
        lvl = tile.level
        px_t = self.tile_tex_px(anchor, tile)       # (the 3D map: as fine as it is on screen)
        is_global = isinstance(tile, farfield.GTile)
        node = tile.base if isinstance(tile, farfield.Piece) else tile
        ck = self.far_ck(anchor.id, tile.key)
        path = os.path.join(self.out_dir, self.tile_name(*ck))
        done = None             # (texture file, bytes, sources used) when the worker rendered it (blockworker.far_tile)
        geo = None              # the geometry's future while this thread renders the texture
        if is_global:
            zoom = farfield.global_dem_zoom(tile.z, self.layout.cells)
            gref = farfield.global_dem_zoom(tile.z + 1, self.layout.cells)
            if self.far_pool is not None and self.imagery is not None:
                # the 3D map: the quick texture in the far-tile process too (raw, what is on disk; the rest downloads
                # meanwhile and re-renders it): the global tiles are most of a map view (2026-11-07, Spokane: 26 s each
                # here, waiting on their photos one interpreter at a time)
                im = self.imagery
                px = px_t
                texture = self.texture_name(*ck, self.tex_rev.get(ck, 0))
                frame_q, x0q, y0q, size_q = self.tile_where(anchor, tile)
                later = im.fetch_later(frame_q, *imagerymod.gutter_square(x0q, y0q, size_q, px), px) if quick else []
                if not later and (self.deshadow is not None or self.colormatch is not None):
                    later = [im.done_token()]
                tex = {"path": os.path.join(self.out_dir, texture), "px": px, "mask": None,
                       "chain": [s_.name for s_ in im.chain], "tone": self.tone, "leaf_on": im.leaf_on,
                       "season_mode": im.season_mode, "quick": quick and bool(later)}
                args = (blockworker.far_global_tile, tile.z, tile.x, tile.y, zoom, gref, self.layout, tex)
                built, got = self.far_pool.submit(*args).result()
                if got is None and im.fetch_needed(frame_q, *imagerymod.gutter_square(x0q, y0q, size_q, px), px,
                                                   last_only=bool(tex["quick"])):
                    built, got = self.far_pool.submit(*args).result()
                if got is not None:
                    done = (texture, got[0], got[1])
                else:
                    later = []
            elif self.far_pool is not None:
                geo = self.far_pool.submit(blockworker.far_global_geometry, tile.z, tile.x, tile.y, zoom, gref,
                                           self.layout)
                built = None
            else:
                base = self.airports.wrap(self.dem) if self.airports is not None else self.dem
                if self.sea is not None:
                    base = seacarvemod.SeaDem(base, None)
                built = farfield.build_global(base, tile.z, tile.x, tile.y, zoom, gref, self.layout)
        else:
            q = self.anchor_q(anchor) if lvl <= 0 else None
            ref_zoom = anchor.zooms[lvl - 1] if lvl > 0 else None
            zoom = anchor.zooms[max(lvl, 0)]
            part = self.piece_parts.get((anchor.id, node.key)) if node is not tile else None
            g = self.grader_for(anchor) if lvl <= 0 and self.far_pool is None else None   # graded like the block
            base = self.airports.wrap(self.dem) if self.airports is not None else self.dem
            if self.sea is not None:
                base = seacarvemod.SeaDem(base, self.sea if lvl <= seacarvemod.CARVE_MAX_LEVEL else None)
            src = gradingmod.GradedDem(base, g) if g is not None else base
            analyses = self.deshadow is not None or self.colormatch is not None
            if quick and analyses and self.far_pool is not None and self.imagery is not None and node is tile:
                # the 3D map at a place not seen yet (photos still to come): the quick texture from the raw photos in
                # the far-tile process (all cores), not the finished pyramid (de-shadow, colour match) in this one
                # interpreter, which held a fresh city's first tiles 12 s (2026-11-07); the background download's
                # token re-renders it finished here once the photos are in (far_retex)
                x0q, y0q, _, _ = tile.bbox
                later = self.imagery.fetch_later(anchor.frame, *imagerymod.gutter_square(
                    x0q, y0q, tile.size, px_t), px_t)
            use_fin = analyses and self.far_pool is not None and node is tile and self.fin_worker_worth()
            if analyses and not later and not use_fin and self.far_pool is not None and node is tile:
                # (no finished tiles here yet: a first visit; raw now in the worker, finished re-rendered right after.
                # Driving too, not only the 3D map (the user, 2026-11-07: "apply what we learned and optimized for the 3d
                # map into driving and flying"): this interpreter rendered them one at a time beside the event loop)
                later = [self.imagery.done_token()]
            raw_quick = bool(later) and analyses
            if FAR_TRACE:
                log.info("fartrace path L%d %s quick %s analyses %s later %d use_fin %s raw %s pool %s", lvl, tile.key,
                         quick, analyses, len(later), use_fin, raw_quick, self.far_pool is not None)
            if self.far_pool is not None and self.imagery is not None and node is tile and \
                    (not analyses or raw_quick or use_fin):
                # the texture in the far-tile process too, from the imagery on disk; with the analyses on (de-shadow,
                # colour match) from this process's finished tiles on disk alone (tex["fin"]: their directories), a
                # tile not finished yet (or with a correction to come) rendered here. They were all rendered here:
                # 81 s of one interpreter's time in a 10 min first visit beside the event loop (2026-11-04)
                texture = self.texture_name(anchor.id, tile.key, self.tex_rev.get(ck, 0))
                im = self.imagery
                tex = {"path": os.path.join(self.out_dir, texture), "px": px_t,
                       "mask": (os.path.join(self.out_dir, surfacemod.road_mask_name(texture)),
                                self.road_mask_px(lvl)) if self.road_masks else None,
                       "chain": [s.name for s in im.chain], "tone": self.tone, "leaf_on": im.leaf_on,
                       "season_mode": im.season_mode}
                if analyses and not raw_quick:
                    tex["fin"] = {s.name: im._fin_sigs.setdefault(s.name, im._fin_sig(s)) for s in im.chain}
                if raw_quick:
                    tex["quick"] = quick        # (driving: every photo, fetched first if missing; the map: what is there)
                elif quick:     # (the 3D map: what is on disk now, the slow hosts' photos downloaded meanwhile)
                    x0q, y0q, _, _ = tile.bbox
                    later = im.fetch_later(anchor.frame, *imagerymod.gutter_square(x0q, y0q, tile.size, tex["px"]),
                                           tex["px"])
                    if later:
                        tex["quick"] = True
                args = (blockworker.far_tile, anchor.id, anchor.lat, anchor.lon, tile, zoom, q,
                        ref_zoom, self.near_visual in BLOCK_DRAWN, self.layout, tex)
                built, got = self.far_pool.submit(*args).result()
                if analyses and not raw_quick:
                    self.fin_worker_hits.append(got is not None)
                if got is None and analyses and not raw_quick:
                    # the 3D map, the finished tiles not on disk (a first visit whose photos the prefetch brought
                    # already): raw in the worker too, finished right after it is shown, not in this interpreter
                    # (Tulsa 2026-11-07: ten first tiles 7 s each here, all at once)
                    later = later or [im.done_token()]
                    raw_quick = True
                    tex.pop("fin", None)
                    tex["quick"] = quick
                    args = args[:-1] + (tex,)
                    built, got = self.far_pool.submit(*args).result()
                if got is None and (not analyses or raw_quick):
                    # a photo tile was not on disk (its prefetch dropped: a host at its daily budget, GitHub #116, or a
                    # fast drive): fetched here, the render goes back to the workers (rendered in this process, one
                    # interpreter for every far thread, the far field waited on it); here only if that fails too
                    x0, y0, _, _ = tile.bbox
                    if im.fetch_needed(anchor.frame, *imagerymod.gutter_square(x0, y0, tile.size, tex["px"]),
                                       tex["px"], last_only=bool(tex.get("quick"))):
                        built, got = self.far_pool.submit(*args).result()
                        self.refetched += 1
                if got is not None:
                    done = (texture, got[0], got[1])
            elif part is not None:          # (another piece of this node built it)
                built = part["built"]
            elif self.far_pool is not None:
                # the geometry in a far-tile process while this thread warps the texture from the finished imagery (no
                # analysis waits there any more: imagery.Imagery.finished); the steps that need both after it
                geo = self.far_pool.submit(blockworker.far_geometry, anchor.id, anchor.lat, anchor.lon, node,
                                           zoom, q, ref_zoom, self.near_visual in BLOCK_DRAWN, self.layout)
                built = None
            else:
                built = farfield.build_tile(anchor.frame, src, node, zoom, q, ref_zoom,
                                            with_sink=self.near_visual in BLOCK_DRAWN, layout=self.layout)
        t1 = time.perf_counter()
        geo_wait = 0.0
        material = texture = used = None
        tex_bytes = 0
        pending = set() if self.colormatch is not None or self.deshadow is not None else None
        part = self.piece_parts.get((anchor.id, node.key)) if node is not tile else None
        if done is not None:
            texture, tex_bytes, used = done
            if later:
                pending = set(pending or ()) | set(later)
            material = self.matslots.take(ck)
            self.degraded.discard(ck)
            self.imagery_worker += 1
        elif part is not None and part.get("tex") is not None:      # the node's texture, another piece's
            texture = self.texture_name(*ck, self.tex_rev.get(ck, 0))
            data, tex_bytes, used, was = part["tex"]
            self.write_file(os.path.join(self.out_dir, texture), data)
            if pending is not None:
                pending |= set(was)
            material = self.matslots.take(ck)
            self.degraded.discard(ck)
        elif self.imagery is not None:
            try:
                px = px_t
                texture = self.texture_name(*ck, self.tex_rev.get(ck, 0))
                tk = self.tex_key(anchor, tile, px)
                hit = revisitmod.load(self.cache_dir, "fartex", tk)
                if hit is not None:         # (rendered in this place before: the revisit cache)
                    blobs, extra = hit
                    if "mask.dds" in blobs:         # (kept files: hard-linked, nothing written)
                        revisitmod.place(blobs["mask.dds"],
                                         os.path.join(self.out_dir, surfacemod.road_mask_name(texture)))
                    revisitmod.place(blobs["tex.dds"], os.path.join(self.out_dir, texture))
                    tex_bytes, used = int(extra["tex_bytes"]), extra["used"]
                else:
                    f0 = revisitmod.faults()
                    if quick and not is_global and not later:
                        frame_q, x0q, y0q, size_q = self.tile_where(anchor, tile)
                        later = self.imagery.fetch_later(frame_q, *imagerymod.gutter_square(x0q, y0q, size_q, px), px)
                    if later:
                        pending = set(pending or ()) | set(later)
                        with self.imagery.quick():
                            rgb, used = self.render_tile(anchor, tile, px, pending)
                    else:
                        rgb, used = self.render_tile(anchor, tile, px, pending)
                    if geo is not None:
                        w0 = time.perf_counter()
                        built, geo = geo.result(), None
                        geo_wait = time.perf_counter() - w0
                    rgb = self.rock_faces(anchor, tile, px, rgb, built.grid)
                    if not is_global:
                        rgb = self.road_faces(anchor, node, px, rgb)
                        self.write_road_mask(anchor, node, px, texture)
                    tex_bytes = imagerymod.write_dds(os.path.join(self.out_dir, texture), rgb)
                    self.keep_texture(tk, texture, tex_bytes, used, pending, f0)
                material = self.matslots.take(ck)
                self.degraded.discard(ck)
            except imagerymod.ImageryError as e:
                # Network trouble: this tile keeps the palette look (the mod's static material).
                self.degraded.add(ck)
                self.imagery_failures += 1
                if self.imagery_failures % 100 == 1:
                    log.warning("tile %s of anchor %d: no imagery (%s), palette material (%d tiles so far)",
                                tile.key, anchor.id, e, self.imagery_failures)
        if geo is not None:             # (a revisit hit, or no imagery)
            w0 = time.perf_counter()
            built = geo.result()
            geo_wait += time.perf_counter() - w0
        if node is not tile:            # a piece: its node cut to its global tile
            if part is None:
                tex = None
                if material is not None:
                    with open(os.path.join(self.out_dir, texture), "rb") as fh:
                        tex = (fh.read(), tex_bytes, used, frozenset(pending or ()))
                self.piece_parts[(anchor.id, node.key)] = {"built": built, "tex": tex}
                while len(self.piece_parts) > PIECE_PARTS:
                    self.piece_parts.popitem(last=False)
            built = farfield.clip_built(built, tile.quad)
        t2 = time.perf_counter()
        size = farfield.write_built(path, built, self.far_format, material or farfield.MATERIAL,
                                    textured=material is not None, tex_px=px_t)
        dt = time.perf_counter() - t0
        self.build_stats.append(dt)
        tex_s = t2 - t1 - geo_wait
        self.level_stats.setdefault(lvl, deque(maxlen=200)).append((dt - tex_s, tex_s, size, tex_bytes))
        sources = frozenset()
        if self.credits is not None:        # the tile's imagery and DEM sources (data credits)
            frame, x0, y0, s = self.tile_where(anchor, tile)
            sources = creditsmod.tile_sources(frame, (x0, y0, x0 + s, y0 + s), self.dem, zoom,
                                              used if material is not None else None)
        info = TileInfo(built.pos, built.err, built.sink, built.skirt, built.grid, size, dt,
                        material, texture, tex_bytes, tex_s, credits=sources)
        info.tile = tile
        info.tex_px = px_t
        if is_global:
            info.frame = farfield.global_frame_ll(tile.z, tile.x, tile.y)
        if material is not None and self.far_format == "cdae":    # the pooled shape path (matslots.py)
            info.shape = matslotsmod.MatSlots.file(material)
            matslotsmod.place(path, os.path.join(self.out_dir, info.shape))
        if material is not None:
            info.rev = self.tex_rev.get(ck, 0)
            info.cm_pending = frozenset(pending or ())
        return info

    def tex_key(self, anchor, tile, px):
        """The revisit cache's key of a far tile's texture rendered here (revisit.py): a global tile's names no anchor
        (any anchor's, any session's); a piece's is its node's (the same texture)."""
        im = self.imagery
        if isinstance(tile, farfield.GTile):
            return revisitmod.key("fartex", "m", farfield.GLOBAL_GEO_VERSION, tile.key, px, [s.name for s in im.chain],
                                  self.tone, im.leaf_on, im.season_mode, self.tex_modes, self.layout.cells,
                                  farfield.global_dem_zoom(tile.z, self.layout.cells))
        if isinstance(tile, farfield.Piece):
            tile = tile.base
        return revisitmod.key("fartex", round(float(anchor.lat), 7), round(float(anchor.lon), 7), tile.key, px,
                              self.road_mask_px(tile.level) if self.road_masks else 0, [s.name for s in im.chain],
                              self.tone, im.leaf_on, im.season_mode, self.tex_modes, roadsmod.ROADS_IN_GROUND,
                              self.mapdata.versions() if self.mapdata is not None else None,
                              self.road_widths is not None, self.surface is not None, self.layout.cells,
                              anchor.zooms[max(tile.level, 0)])

    def keep_texture(self, tk, texture, tex_bytes, used, pending, f0):
        """A far texture rendered here kept in the revisit cache when it is final: no colour-match or de-shadow region
        still to come (pending), nothing it read failed meanwhile (f0: revisit.faults() before; None: not checked, a
        render again for a region that came)."""
        if not revisitmod.ON or pending or (f0 is not None and revisitmod.faults() != f0):
            return
        try:
            blobs = {"tex.dds": revisitmod.Side(os.path.join(self.out_dir, texture))}     # (linked in, not copied)
            mp = os.path.join(self.out_dir, surfacemod.road_mask_name(texture))
            if os.path.exists(mp):
                blobs["mask.dds"] = revisitmod.Side(mp)
            revisitmod.store(self.cache_dir, "fartex", tk, blobs,
                             {"tex_bytes": int(tex_bytes), "used": {str(n): int(c) for n, c in (used or {}).items()}})
        except Exception as e:          # (not kept rather than no tile)
            log.debug("revisit: far texture %s not kept: %s: %s", texture, type(e).__name__, e)

    @staticmethod
    def write_file(path, data):
        """path written atomically (staging: the game sees only finished files)."""
        tmp = staging.tmp_for(path)
        with open(tmp, "wb") as f:
            f.write(data)
        staging.replace(tmp, path)

    def rock_faces(self, anchor, tile, px, rgb, grid):
        """The far tile's photo with its steep ground drawn as rock (rockface.py: cliffs were the photo's top
        stretched down the wall); as it was if that fails."""
        if grid is None:
            return rgb
        try:
            frame, x0, y0, size = self.tile_where(anchor, tile)
            g = np.asarray(grid)
            return rockfacemod.rock(rgb, frame, *imagerymod.gutter_square(x0, y0, size, px), g,
                                    x0, y0, size / (g.shape[1] - 1))
        except Exception as e:          # the photo as it is
            log.debug("rock faces of %s: %s", tile.key, e)
            return rgb

    def road_faces(self, anchor, tile, px, rgb):
        """The far tile's photo with the carriageways painted in (roads.ROADS_IN_GROUND, surface.paint_square: the
        near ring the drawn block does not cover); as it was otherwise or if that fails."""
        if not roadsmod.ROADS_IN_GROUND:
            return rgb
        try:
            x0, y0, _, _ = tile.bbox
            return surfacemod.paint_square(rgb, anchor.frame, *imagerymod.gutter_square(x0, y0, tile.size, px),
                                           self.surface, self.road_widths)
        except Exception as e:          # the photo as it is
            log.debug("road faces of %s: %s", tile.key, e)
            return rgb

    def road_mask_px(self, level):
        """The road mask's edge for a near level (farfield.NEAR_MASK_PX), or 0 (none)."""
        return farfield.NEAR_MASK_PX.get(level - self.layout.exact_level, 0)

    def write_road_mask(self, anchor, tile, px, texture):
        """The near tile's road mask beside its texture (self.road_masks: mesh mode; surface.road_mask, over the
        photo's square with its gutter, at road_mask_px), written before the texture (the mod looks for it when it
        sets the texture)."""
        mpx = self.road_mask_px(tile.level)
        if not self.road_masks or not mpx:
            return
        try:
            x0, y0, _, _ = tile.bbox
            mask = surfacemod.road_mask(anchor.frame, *imagerymod.gutter_square(x0, y0, tile.size, px), mpx,
                                        self.surface, self.road_widths)
            if mask is not None:
                imagerymod.write_dds(os.path.join(self.out_dir, surfacemod.road_mask_name(texture)),
                                     np.repeat(mask[..., None], 3, axis=2))
        except Exception as e:          # the tile without its asphalt layer
            log.debug("road mask of %s: %s", tile.key, e)

    def retexture(self, anchor, tile, prev):
        """Worker thread: render a stale tile's texture again (its colour-match regions are ready now)
        under the next revision's file name; the shape and its material name stay. Network trouble:
        the previous TileInfo, still stale."""
        t0 = time.perf_counter()
        px = self.tile_tex_px(anchor, tile)
        pending = set()
        if self.worker_retex() and self.far_pool is not None and self.imagery is not None and \
                isinstance(tile, farfield.GTile):
            ck = self.far_ck(anchor.id, tile.key)
            texture = self.texture_name(*ck, prev.rev + 1)
            im = self.imagery
            tex = {"path": os.path.join(self.out_dir, texture), "px": px, "mask": None,
                   "chain": [s_.name for s_ in im.chain], "tone": self.tone, "leaf_on": im.leaf_on,
                   "season_mode": im.season_mode}
            got = self.far_pool.submit(blockworker.far_global_tile, tile.z, tile.x, tile.y,
                                       farfield.global_dem_zoom(tile.z, self.layout.cells),
                                       farfield.global_dem_zoom(tile.z + 1, self.layout.cells), self.layout,
                                       tex).result()[1]
            if got is not None:
                return replace(prev, texture=texture, tex_bytes=got[0], tex_seconds=time.perf_counter() - t0,
                               cm_pending=frozenset(), stale=False, rev=prev.rev + 1, tex_px=px)
        if self.worker_retex() and self.far_pool is not None and self.imagery is not None and \
                type(tile) is farfield.Tile and tile.level > 0:
            # the 3D map: the sharper texture (its photos in now) from the raw photos in a far-tile process, many at
            # once, not the finished pyramid in this one interpreter (2026-11-07: a landed city's ground stayed blurry
            # for half a minute, two re-renders at a time here; its corrections changed nothing there anyway)
            ck = self.far_ck(anchor.id, tile.key)
            texture = self.texture_name(*ck, prev.rev + 1)
            im = self.imagery
            tex = {"path": os.path.join(self.out_dir, texture), "px": px,
                   "mask": (os.path.join(self.out_dir, surfacemod.road_mask_name(texture)),
                            self.road_mask_px(tile.level)) if self.road_masks else None,
                   "chain": [s_.name for s_ in im.chain], "tone": self.tone, "leaf_on": im.leaf_on,
                   "season_mode": im.season_mode}
            lvl = tile.level
            got = self.far_pool.submit(blockworker.far_tile, anchor.id, anchor.lat, anchor.lon, tile,
                                       anchor.zooms[lvl], None, anchor.zooms[lvl - 1], self.near_visual in BLOCK_DRAWN,
                                       self.layout, tex).result()[1]
            if got is not None:
                return replace(prev, texture=texture, tex_bytes=got[0], tex_seconds=time.perf_counter() - t0,
                               cm_pending=frozenset(), stale=False, rev=prev.rev + 1, tex_px=px)
        try:
            rgb, used = self.render_tile(anchor, tile, px, pending)
        except imagerymod.ImageryError as e:
            log.debug("tile %s of anchor %d: texture not rendered again (%s)", tile.key, anchor.id, e)
            return prev
        rgb = self.rock_faces(anchor, tile, px, rgb, prev.grid)
        node = tile.base if isinstance(tile, farfield.Piece) else tile
        ck = self.far_ck(anchor.id, tile.key)
        if not isinstance(tile, farfield.GTile):
            rgb = self.road_faces(anchor, node, px, rgb)
        texture = self.texture_name(*ck, prev.rev + 1)
        if not isinstance(tile, farfield.GTile):
            self.write_road_mask(anchor, node, px, texture)
        tex_bytes = imagerymod.write_dds(os.path.join(self.out_dir, texture), rgb)
        self.keep_texture(self.tex_key(anchor, tile, px), texture, tex_bytes, used, pending, None)
        sources = prev.credits
        if self.credits is not None:        # the new texture's sources (the corrected one names S2 too)
            frame, x0, y0, s = self.tile_where(anchor, tile)
            zoom = (farfield.global_dem_zoom(tile.z, self.layout.cells) if isinstance(tile, farfield.GTile)
                    else anchor.zooms[max(tile.level, 0)])
            sources = creditsmod.tile_sources(frame, (x0, y0, x0 + s, y0 + s), self.dem, zoom, used)
        return replace(prev, texture=texture, tex_bytes=tex_bytes, tex_seconds=time.perf_counter() - t0,
                       cm_pending=frozenset(pending), stale=False, rev=prev.rev + 1, credits=sources, tex_px=px)

    def start_world_warm(self, anchor):
        """The rough earth (the user, 2026-11-07: "keep a very rough version of earth that we can easaly load at all times
        ready in storage"): once per server, the first time the 3D map's camera is high (Conn.update_far), every
        WORLD_WARM_Z global far tile (L10 for the usual anchor zoom: ~260 km, a 128 px photo) built into the revisit
        cache, nearest the view first, WORLD_WARM_CONC at a time and only while no far tile of a view waits (a tile
        already in the cache is skipped: from then on a far-out view loads from disk)."""
        if getattr(self, "_world_warm", None) is not None or not revisitmod.ON:
            return
        self._world_warm = asyncio.get_running_loop().create_task(self._warm_world(anchor))

    async def _warm_world(self, anchor):
        z = WORLD_WARM_Z
        n = 1 << z
        lat0, lon0 = float(anchor.lat), float(anchor.lon)
        cands = []
        for y in range(n):
            la, _ = farfield.global_frame_ll(z, 0, y)
            if abs(la) > farfield.GLOBAL_MAX_LAT:
                continue
            for x in range(n):
                la, lo = farfield.global_frame_ll(z, x, y)
                d = math.hypot((la - lat0), (((lo - lon0 + 180.0) % 360.0) - 180.0) * math.cos(math.radians(lat0)))
                cands.append((d, x, y))
        cands.sort()
        level = farfield.GLOBAL_MIN_LEVEL + int(anchor.zf) - z
        px = self.tex_size(level)
        done = skipped = 0
        t0 = time.monotonic()
        sem = asyncio.Semaphore(WORLD_WARM_CONC)

        async def one(x, y):
            nonlocal done, skipped
            async with sem:
                while len(self.tile_inflight) > WORLD_WARM_CONC:      # (a view's tiles first)
                    await asyncio.sleep(0.5)
                q = tuple(float(v) for v in geotilesmod.quad(anchor.frame, x, y, z).ravel())
                edge = sum(math.hypot(q[(2 * k + 2) % 8] - q[2 * k], q[(2 * k + 3) % 8] - q[2 * k + 1])
                           for k in range(4)) / 4
                t = farfield.GTile(z, x, y, level, q, float(edge))
                if os.path.exists(revisitmod._path(self.cache_dir, "fartex", self.tex_key(anchor, t, px))):
                    skipped += 1
                    return
                try:
                    await self.tile(anchor, t)
                    done += 1
                except Exception as e:          # (offline, a source down: the next start tries again)
                    log.debug("world warm %s/%s/%s: %s", z, x, y, e)
                if (done + skipped) % 500 == 0:
                    log.info("world warm: %d built, %d cached, of %d (%.0f s)", done, skipped, len(cands),
                             time.monotonic() - t0)

        await asyncio.gather(*(one(x, y) for _, x, y in cands))
        log.info("world warm: done, %d built, %d already cached, in %.0f s", done, skipped, time.monotonic() - t0)

    def tile(self, anchor, tile):
        """asyncio future for the tile's TileInfo; generated at most once per (anchorId, key) (a global tile's: once
        for every anchor, far_ck)."""
        ck = self.far_ck(anchor.id, tile.key)
        loop = self.loop = asyncio.get_running_loop()
        prev = None
        if ck in self.tiles:
            self.tiles.move_to_end(ck)
            if not self.tiles[ck].stale:
                f = loop.create_future()
                f.set_result(self.tiles[ck])
                return f
            prev = self.tiles[ck]           # --colormatch: its texture is rendered again
        fut = self.tile_inflight.get(ck)
        if fut is None:
            fut = self.tile_inflight[ck] = loop.run_in_executor(self.tile_pool, self.make_tile, anchor, tile, prev)
            fut.add_done_callback(lambda f: self._tile_done(ck, f))
        # A connection may stop waiting at a rebase or disconnect. Cancelling the
        # executor Future does not stop a running thread: it can still write its
        # pooled shape. Keep that producer tracked until its writes have finished,
        # so reclaim_names cannot give its slot to another tile in the meantime.
        return asyncio.shield(fut)

    def in_use(self):
        """(anchorId, key) of every tile a connection shows or holds, or let go of less than
        prune_delay ago: after a far_swap / far_remove the mod still shows the tile until the
        swap has run (its adds load within the governor budget, then a 0.5 s crossover) and
        deletes it time-sliced, and a shape file must not vanish under a live TSStatic."""
        out = self.matslots.held_owners()
        now = time.monotonic()
        for c in self.connections:
            out |= set(c.shown)
            out |= {self.far_ck(c.active, k) for k in c.sent}
            # built for the active anchor but not sent yet: evicting these deleted the files of
            # tiles about to be sent (in-game 2026-10-10, after a 200 mph run had left thousands
            # of held previous-anchor tiles in the cache: 150 tiles failed with "No Mesh")
            out |= {self.far_ck(c.active, k) for k in c.desired}
            out |= set(c.inflight)
            out |= {ck for ck, t in c.retired.items() if now - t < self.retired_keep}
        held = self.held_anchors()
        if held:
            out |= {ck for ck in self.tiles if ck[0] in held}
        return out

    def names_in_use(self):
        """(anchorId, key) of the tiles a connection is about to send or build (desired, sent, in flight): they
        keep their names. Whether a game still draws a name is slot_held()."""
        out = set()
        for c in self.connections:
            out |= {self.far_ck(c.active, k) for k in c.sent}
            out |= {self.far_ck(c.active, k) for k in c.desired}
            out |= set(c.inflight)
        return out

    def slot_held(self, n):
        """Does some game still hold pooled slot n (a far_tile with it not reported gone)?"""
        # Allocation can query this from a tile worker while the event loop
        # connects/disconnects clients; do not iterate the mutable set directly.
        return self.stream_leases.held(n) or any(n in c.slot_holds for c in tuple(self.connections)
                                                 if getattr(c, "lease", None) is None)

    def attach_stream(self, conn, msg):
        """Reconcile a game's complete resource snapshot before sending it new pooled files."""
        resumable = msg.get("streamLease") == 1
        token = msg.get("leaseId") if msg.get("leaseSession") == self.sid else None
        if not isinstance(token, str):
            token = None
        same_session = msg.get("slotSession") == self.sid
        try:
            if resumable and token in self.stream_leases.entries and not same_session:
                raise ValueError("resuming a stream requires its complete slot snapshot")
            slots = streamleases.snapshot(msg.get("slotsHeld"), self.matslots.made) if same_session else None
            held = msg.get("held", []) if same_session else []
            if held == {}:
                held = []
            if not isinstance(held, list) or any(not streamleases.integer(n) for n in held):
                raise ValueError("held anchors must be an array of integers")
            anchors = {int(n) for n in held if n in self.anchors}
        except ValueError as e:
            # (lenient: a bad snapshot starts a fresh lease rather than failing the hello; the old lease, if any,
            # stays detached and expires: expire_leases)
            log.warning("client %s: stream snapshot not resumed (%s): a new lease", getattr(conn, "peer", None), e)
            token, slots, anchors = None, None, set()
        lease, released, previous = self.stream_leases.attach(conn, token, slots, anchors, resumable)
        if previous is not None:
            previous.abort_stream("replaced by this game's reconnect")
        self.slots_let_go(released)
        return lease.token

    def slots_let_go(self, slots):
        """Event loop: these slots were reported gone or excluded by a complete reconnect snapshot."""
        for n in slots:
            if not self.slot_held(n):
                self.matslots.unheld(n)

    def held_anchors(self):
        """Previous anchors whose tile objects some mod still has (pose.held): their files stay."""
        out = self.stream_leases.anchors()
        for c in self.connections:
            lease = getattr(c, "lease", None)
            if lease is None or lease.connection is c:
                out |= c.held
        out.discard(GLOBAL_AID)
        return out

    def _tile_done(self, ck, fut):
        self.tile_inflight.pop(ck, None)
        if fut.cancelled() or fut.exception() is not None:
            return
        before = self.tiles.get(ck)
        info = self.tiles[ck] = fut.result()
        if before is not None and info.texture != before.texture:
            self._send_retex(ck, info)      # a re-rendered texture: the mods showing the tile switch to it
        for rk in sorted(info.cm_pending, key=str):     # regions it was rendered without
            st = self._region_status(rk)
            if st is None:
                self.cm_waiting.setdefault(rk, set()).add(ck)
            else:                           # ready meanwhile (while the texture was being rendered)
                self._cm_settle(ck, rk, st, pump=False)
        if before is not None and info is not before:     # a re-render: stale at once again, or not
            info.retex_repeat = before.retex_repeat + 1 if info.stale else 0
        if info.stale:
            self._retex_pump()
        if self.matslots.short():
            self.reclaim_names()
        if len(self.tiles) > self.tile_cache:
            used = self.in_use()
            for old in [k for k in self.tiles if k not in used][:len(self.tiles) - self.tile_cache]:
                self._evict_tile(old)

    def reclaim_names(self):
        """Event loop: the pooled names (and shape files) of tiles about to be neither sent nor built back to the
        pool (matslots; one a game still holds waits for its report): names left by builds that never finished
        first, then idle cached tiles, least recently used first."""
        used = self.names_in_use()
        owners = self.matslots.holders()
        lost = [ck for ck in owners if ck not in self.tiles and ck not in self.tile_inflight and ck not in used]
        idle = (ck for ck in self.tiles if ck not in used and ck not in self.tile_inflight)
        n = self.matslots.reclaim(itertools.chain(lost, idle))
        log.debug("material names: %d reclaimed, %s", n, self.matslots.stats())

    def pool_shapes(self, conn, files):
        """Event loop: the road / building tile's shape files ({what: file name in the session folder}) as pooled ones
        (matslots.py; the far tiles' slots): each a hard link at a slot path, held by conn until the mod reports it
        gone (pose.slotsGone). Returns {what: [vfs path, serial]} for the message's `slots` (files that do not exist
        are left out; a slot that cannot be placed too: the mod then loads the tile's own file)."""
        out = {}
        lease = getattr(conn, "lease", None)
        if getattr(conn, "closing", False) or (lease is not None and lease.connection is not conn):
            return out
        if self.far_format != "cdae":
            return out
        for what, fname in files.items():
            src = os.path.join(self.out_dir, fname)
            if not os.path.exists(src):
                continue
            owner = ("shape", next(self.slot_serial))
            name = self.matslots.take(owner)
            slot = matslotsmod.MatSlots.file(name)
            try:
                matslotsmod.place(src, os.path.join(self.out_dir, slot))
            except OSError as e:
                log.debug("pooled shape %s: %s", fname, e)
                self.matslots.release(owner)
                continue
            serial = next(self.slot_serial)
            conn.slot_holds[matslotsmod.MatSlots.number(name)] = serial
            self.matslots.release(owner)        # no tile owns it: free again once the mod lets go
            out[what] = [f"{self.vfs_dir}/{slot}", serial]
        return out

    def reslot(self, ck, info):
        """Event loop: a cached tile whose material name went to another tile meanwhile (reclaim_names) is
        sent again: a copy of its shape with the new name in the new name's pooled file (slot_<n>; DAE tiles: a file
        _s<serial>, the old copy deleted), even when it got its own name back (its pooled file may hold another
        tile by now).
        On failure the tile keeps its name: the caller builds it again rather than send it with another tile's name."""
        new = self.matslots.take(ck)
        src = os.path.join(self.out_dir, self.tile_name(*ck))
        pooled = self.far_format == "cdae"
        name = matslotsmod.MatSlots.file(new) if pooled else f"{os.path.splitext(self.tile_name(*ck))[0]}_s{next(self.reslot_serial)}.{self.far_format}"
        try:
            with open(src, "rb") as fh:
                data = fh.read()
            packed = pooled and cdae.is_packed(data)
            if packed:                  # (a compressed shape: the names are inside its body; written back so)
                data = cdae.plain(data)
            # the name in the tile's own file: the one it was built with, not info.material once it was renamed before
            # (a global far tile kept across rebases is renamed again and again: restamping info.material failed from
            # its second rename on, and the tile was sent with a name another tile had, drawing that tile's texture:
            # Utah's red far tiles on the Pikes Peak ground, 2026-10-29)
            have = info.material if info.material.encode() in data else self.matslots.name_in(data)
            if have is None:
                raise ValueError(f"no pooled name in {os.path.basename(src)}")
            if new != have:
                data = matslotsmod.restamp(data, have, new)
            if packed:
                data = cdae.pack(data)
            dst = os.path.join(self.out_dir, name)
            tmp = staging.tmp_for(dst)
            with open(tmp, "wb") as fh:
                fh.write(data)
            staging.replace(tmp, dst)
        except (OSError, ValueError) as e:
            log.warning("tile %s of anchor %d: material name not renewed (%s)", ck[1], ck[0], e)
            return info
        if not pooled and info.shape is not None:
            self.queue_delete([os.path.join(self.out_dir, info.shape)])
        info = self.tiles[ck] = replace(info, material=new, shape=name)
        return info

    def _evict_tile(self, ck):
        info = self.tiles.pop(ck)
        self.degraded.discard(ck)
        self.matslots.release(ck)
        # --colormatch: a later build must not reuse its texture's name (re-rendered, or rendered raw
        # for a region that may finish before the tile is built again: the listener misses it then)
        if info.stale or info.rev or info.cm_pending:
            self.tex_rev[ck] = info.rev + 1
        # through the rate-limited queue: the mod loaded most of these files once (P9)
        files = self.tile_files(self.tile_name(*ck), info.texture if info.rev else None, info.rev)
        if info.shape is not None and not info.shape.startswith("slot_"):   # pooled files stay (re-used)
            files.append(os.path.join(self.out_dir, info.shape))
        self.queue_delete((p for p in files if os.path.exists(p)), ck)

    def _cm_listener(self, rk, changed):
        """ColorMatch worker thread: a region's correction is ready."""
        loop = self.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._cm_ready, rk, changed)

    def _ds_listener(self, rk, changed):
        """Deshadow worker thread: a region's relight values are ready (changed: it relights)."""
        self._cm_listener((imagerymod.DS_PENDING,) + tuple(rk), changed)

    def _region_status(self, rk):
        """None (not ready), False (ready, changes nothing) or True of a pending region: a colour-match
        (rx, ry) or a de-shadow (DS_PENDING, rx, ry) one (event loop: memory and stat calls only)."""
        if len(rk) == 3 and rk[0] == imagerymod.DS_PENDING:
            return None if self.deshadow is None else self.deshadow.status(tuple(rk[1:]))
        if len(rk) == 2 and rk[0] == imagerymod.IMG_PENDING:      # (a quick texture's photos: Imagery.fetch_later)
            return None if self.imagery is None else self.imagery.img_status(rk)
        return None if self.colormatch is None else self.colormatch.status(rk)

    def _cm_ready(self, rk, changed):
        """The far tiles whose textures were rendered without the region's correction (colour match or
        de-shadow): if it changes them they are stale. tile() renders the texture again (only the
        texture, under the next revision's name) whenever the tile is sent again, and a tile some mod
        shows (or holds, sent) is re-rendered live and swapped with far_retex (_retex_pump)."""
        for ck in sorted(self.cm_waiting.pop(rk, ())):
            self._cm_settle(ck, rk, changed, pump=False)
        self._retex_pump()                  # once all are queued: the nearest goes first

    def _cm_settle(self, ck, rk, changed, pump=True):
        info = self.tiles.get(ck)
        if info is None or rk not in info.cm_pending:
            return
        info.cm_pending = info.cm_pending - {rk}
        if changed:
            info.stale = True
            self.retex_request(ck, pump)

    # --- live retexture (far_retex) ---------------------------------------------------------
    def _retex_viewers(self, ck):
        """The connections that show the tile, or hold it (sent, not swapped in yet), under their
        active anchor (a previous anchor's tiles are on their way out)."""
        aid, key = ck
        return [c for c in self.connections if c.active is not None and self.far_ck(c.active, key) == ck
                and (ck in c.shown or key in c.sent)]

    def retex_request(self, ck, pump=True):
        """A stale tile to re-render live if some mod shows it (event loop)."""
        if self.retex_rate <= 0 or self.imagery is None:
            return
        self.retex_todo.add(ck)
        if pump:
            self._retex_pump()

    def _retex_pick(self):
        """The stale tile to re-render next: shown (or held) by a connection, nearest its player
        first. Drops the ones nobody shows (re-rendered when sent again), already re-rendering, or
        past RETEX_REPEAT re-renders in a row that came back stale at once."""
        best, bd = None, math.inf
        for ck in list(self.retex_todo):
            info = self.tiles.get(ck)
            viewers = self._retex_viewers(ck) if info is not None and info.stale else []
            if not viewers or ck in self.tile_inflight or (ck[0] not in self.anchors and ck[0] != GLOBAL_AID) \
                    or info.retex_repeat >= RETEX_REPEAT:
                self.retex_todo.discard(ck)
                if info is not None and info.stale and ck not in self.tile_inflight:
                    self.retex_stats["dropped"] += 1
                continue
            t = info.tile if info.tile is not None else farfield.tile_of(ck[1])
            d = min(t.distance(*c.view_focus) for c in viewers)
            if best is None or (d, ck) < (bd, best):
                best, bd = ck, d
        return best

    def _retex_pump(self):
        """Start live re-renders within the rate and in-flight bounds (event loop; never waits)."""
        if self._retex_timer is not None or self.loop is None or self.loop.is_closed():
            return
        if any(getattr(c, "lod_ramp", 1.0) < 1.0 and (c.map_view or c.map_open) for c in self.connections):
            return          # (the 3D map's coarse steps: their tiles go at the next step; sharpened once at full detail)
        boost = MAP_RETEX_BOOST if self.map_on() else 1      # (the 3D map: re-renders in the far-tile processes)
        while self.retex_todo and self.retex_busy < self.retex_inflight * boost:
            # (the 3D map: only into far-tile slots no build of a tile not shown yet needs; Tucson 2026-11-07: eight
            # re-renders held the gate and the far field built 110 tiles a minute; Conn._stream pumps again)
            if boost > 1 and self.retex_busy >= self.retex_inflight and \
                    sum(1 for c in self.connections for f, _ in c.inflight.values() if not f.done()) + \
                    self.retex_busy >= self.far_limit():
                return
            wait = self._retex_next - time.monotonic()
            if wait > 0:
                self._retex_timer = self.loop.call_later(wait, self._retex_wake)
                return
            ck = self._retex_pick()
            if ck is None:
                return
            self.retex_todo.discard(ck)
            self._retex_next = time.monotonic() + 1.0 / (self.retex_rate * boost)
            prev = self.tiles[ck]
            tile = prev.tile if prev.tile is not None else farfield.tile_of(ck[1])
            viewers = self._retex_viewers(ck)
            anchor = self.anchors.get(ck[0]) or self.anchors.get(viewers[0].active if viewers else None)
            if anchor is None:
                continue
            fut = self.tile_inflight[ck] = self.loop.run_in_executor(self.tile_pool, self.make_tile, anchor, tile, prev)
            fut.add_done_callback(lambda f, ck=ck: self._tile_done(ck, f))
            fut.add_done_callback(self._retex_done)
            self.retex_busy += 1
            self.retex_stats["started"] += 1

    def _retex_wake(self):
        self._retex_timer = None
        self._retex_pump()

    def _retex_done(self, fut):
        self.retex_busy -= 1
        if not fut.cancelled() and fut.exception() is not None:    # not the network (still stale): a bug, the disk
            self.retex_stats["failed"] = self.retex_stats.get("failed", 0) + 1
            if self.retex_stats["failed"] <= 3:
                log.warning("live retexture failed: %r", fut.exception())
        self._retex_pump()

    def _send_retex(self, ck, info):
        """far_retex to every connection showing (or holding) the tile: its new texture."""
        for c in self._retex_viewers(ck):
            c.send({"type": "far_retex", "anchorId": c.active if ck[0] == GLOBAL_AID else ck[0], "key": ck[1],
                    "texture": f"{self.vfs_dir}/{info.texture}"})
            self.retex_stats["sent"] += 1
            c.credits.poke(c)               # the new texture's sources (the corrected one names S2 too)
