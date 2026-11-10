"""World server connection: anchors and terrain (split out of earthd.py's Connection): preparing and committing an
anchor, provisional and refined terrain, boosts and catching up.
"""
import os
import asyncio  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import surface as surfacemod  # noqa: E402
import gasstations as gasstationsmod  # noqa: E402
import streamleases
import grading as gradingmod  # noqa: E402
import pacer  # noqa: E402
import farfield  # noqa: E402
import geodesy  # noqa: E402
import heightmap  # noqa: E402
import cpupin  # noqa: E402
from earthbase import (
    ANCHOR_CENTRE_MAX,
    ANCHOR_PROGRESS_S,
    CATCHUP_AFTER_S,
    CATCHUP_CALM_S,
    CATCHUP_LEND_SHARE,
    CATCHUP_MIN_MS,
    CATCHUP_MORE_S,
    CATCHUP_ROADS_M,
    FOCI_RANGE_M,
    HOLD_BOOST_CALM_S,
    MAP_RAMP_START,
    PATCH_KEY,
    PROVISIONAL_FIRST,
    PROVISIONAL_FOLLOW_S,
    PROVISIONAL_KEEP_S,
    PROVISIONAL_RUNNING_S,
    TELEPORT_PAUSE_M,
    TERRAIN_STUCK_S,
    TUNNEL_FLOOR_UNDER_M,
    VIEW_ANCHOR_Q,
    _num,
    _perf,
    app_paths_message,
    log,
    pick_app_path,
    set_app_path,
)


class _AnchorMixin:
    """Connection's anchors and terrain (split out of earthd.py; Connection inherits it)."""

    async def terrain_patch(self, msg):
        """The mod's terrainpatch.lua: the ground's collision patch over box (x0, y0, x1, y1, the active anchor's
        frame) under a collision island off the terrain block (collisionfoci.lua); answered with its path and centre
        (or an error)."""
        aid, key = msg.get("anchorId"), msg.get("key")
        try:
            box = tuple(float(msg[k]) for k in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            return
        if (aid not in self.s.anchors or not isinstance(key, str) or not PATCH_KEY.fullmatch(key)
                or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]
                or max(abs(v) for v in box) > FOCI_RANGE_M):
            return
        try:
            res = await asyncio.get_running_loop().run_in_executor(None, self.s.build_terrain_patch, aid, key, box)
        except Exception as e:      # the ground not at hand (offline): the mod asks again later
            log.warning("terrain patch %s: %s: %s", key, type(e).__name__, e)
            self.send({"type": "terrain_patch", "anchorId": aid, "key": key, "error": f"{type(e).__name__}: {e}"[:200]})
            return
        if res is not None:
            parts, n = res
            self.send({"type": "terrain_patch", "anchorId": aid, "key": key, "tris": n,
                       "parts": [{"path": p, "x": c[0], "y": c[1]} for p, c in parts]})

    async def app_path(self, msg):
        """The settings page's "Cache folder" and "BeamMP launcher" (apppaths.py, the user 2026-10-25: "Open the
        explorer for that operating system but have a text input for the path as fallback"): what ('cache' or
        'beammp') chosen in the system's window (pick: true) or typed (path), kept for the next start of the app;
        always answered with app_paths (the paths now, and how the choice went). Without what: the paths only."""
        what = msg.get("what")
        pick, path = msg.get("pick") is True, msg.get("path")
        if what not in ("cache", "beammp") or not (pick or isinstance(path, str)):
            self.send(await self.loop.run_in_executor(None, app_paths_message, self.s.cache_dir, None))
            return
        if pick:
            if self.s.picking:
                self.send(await self.loop.run_in_executor(None, app_paths_message, self.s.cache_dir,
                                                          {"what": what, "ok": False, "text": "a window is already open"}))
                return
            self.s.picking = what
            try:
                self.send(await self.loop.run_in_executor(None, app_paths_message, self.s.cache_dir, None))
                state, path = await self.loop.run_in_executor(None, pick_app_path, what)
            finally:
                self.s.picking = None
            if state != "ok":
                result = {"what": what, "ok": False, "window": state != "none",
                          "text": "nothing chosen" if state == "cancel" else "no file window here: type the path"}
                self.send(await self.loop.run_in_executor(None, app_paths_message, self.s.cache_dir, result))
                return
        result = await self.loop.run_in_executor(None, set_app_path, what, path[:1000])
        log.info("app path %s: %s", what, result.get("text"))
        self.send(await self.loop.run_in_executor(None, app_paths_message, self.s.cache_dir, result))

    def on_cache(self, msg):
        """The settings page: capGb sets the cap (0 = unlimited), clear: true empties the cache (but this
        session's files and the gravity model) on an executor; every 'cache' gets the usage back (after a
        clear, with clearedBytes). Without a Cache (Server()'s default): ignored."""
        cache = self.s.cache
        if cache is None:
            return
        cap = msg.get("capGb")
        if cap is not None:
            if isinstance(cap, bool) or not isinstance(cap, (int, float)):
                raise ValueError(f"capGb {cap!r}: a number")
            if cache.set_cap(cap):
                log.info("cache: cap %s (from the mod)", f"{cap:g} GB" if cap else "unlimited")
        if msg.get("clear") is True:
            async def run():
                _, freed = await self.loop.run_in_executor(None, cache.clear)
                self.send(cache.message(cleared=freed))
            self.spawn(run())
        else:
            self.send(cache.message())

    # --- anchors and terrain ------------------------------------------------
    def _progress(self):
        last = [0.0]

        def progress(done, total):      # called from DEM worker threads
            now = time.monotonic()
            if done == total or now - last[0] > 0.25:
                last[0] = now
                self.loop.call_soon_threadsafe(self.status, f"Elevation tiles {done}/{total}")
        return progress

    def terrain_msg(self, key, info):
        out = {"cx": key[1], "cy": key[2], "heightmap": self.s.hm_vfs(key), "size": self.s.layout.block,
               "squareSize": heightmap.SQUARE, "zBase": info["zBase"], "maxHeight": info["maxHeight"],
               "q": info["q"]}
        if info.get("layers"):      # --surface on: one mask per material, MATERIALS order
            # only the .ter's own materials, in its order (surface.write_ter; all of them for an older block)
            use = info.get("mats") or range(len(surfacemod.MATERIALS))
            out["layers"] = [f"{self.s.vfs_dir}/{info['layers'][i]}" for i in use]
            out["materials"] = [list(surfacemod.MATERIALS[i]) for i in use]
        if info.get("base"):        # detail mode: the base photo over the block's world size
            out["base"] = f"{self.s.vfs_dir}/{info['base']}"
            out["baseSize"] = self.s.layout.block * heightmap.SQUARE
        if info.get("ter"):         # the same block as the engine's terrain file (swapped in, no import)
            out["ter"] = f"{self.s.vfs_dir}/{info['ter']}"
        return out

    async def prepare_anchor(self, msg):
        """Every first-ground preparation has progress and a terminal reply, including setup/finalization errors."""
        req = msg.get("requestId")
        progress = {"stage": "preparing", "done": False}
        started = time.monotonic()

        def report():
            self.send({"type": "anchor_progress", "requestId": req, "stage": progress["stage"],
                       "elapsed": round(time.monotonic() - started, 1)})

        async def heartbeat():
            while True:
                await asyncio.sleep(ANCHOR_PROGRESS_S)
                report()

        report()
        tick = asyncio.create_task(heartbeat())
        try:
            await self._prepare_anchor(msg, progress)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("anchor request %s failed during %s", req, progress["stage"])
            if not progress["done"]:
                self.teleport_pending = False
                self.wake.set()
                self.send({"type": "anchor_error", "requestId": req, "stage": progress["stage"],
                           "error": f"{type(e).__name__}: {e}"})
        finally:
            tick.cancel()
            try:
                await tick
            except asyncio.CancelledError:
                pass

    async def _prepare_anchor(self, msg, progress):
        req = msg.get("requestId")
        try:
            lat, lon = float(msg["lat"]), float(msg["lon"])
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90):
                raise ValueError
            lim = ANCHOR_CENTRE_MAX
            # the player's spot in the new frame when it is not the origin (BeamMP: a teleport onto another player's
            # anchor, mpsession.adoptAnchor): its ground height is sent instead of the origin's
            place = None
            if msg.get("px") is not None:
                place = (min(max(float(msg["px"]), -lim), lim), min(max(float(msg["py"]), -lim), lim))
            cx = min(max(self.s.snap(_num(msg, "cx", 0.0)), -lim), lim)
            cy = min(max(self.s.snap(_num(msg, "cy", 0.0)), -lim), lim)
        except (KeyError, TypeError, ValueError):
            self.send({"type": "anchor_error", "requestId": req, "error": "invalid lat/lon/cx/cy"})
            return
        lon = (lon + 180.0) % 360.0 - 180.0
        old = self.s.anchors.get(self.active)
        if msg.get("purpose") == "view" and old is not None:
            # the 3D map's origin at the view (mod map3d.lua; the user, 2026-11-07: zooming into a spot far from the
            # car "the screen stays black"): the frame only. The car and its block stay in their own frame, hidden,
            # until the map closes; a first block here was built at the clamped car spot, neither the car's nor the
            # view's ground, and took 13 s of a fresh place (Milan) before anything could be drawn
            t0 = time.perf_counter()
            anchor = self.s.new_anchor(lat, lon)
            # (its height quantum without the survey: no block is ever driven on in this frame (the car goes back to
            # its own, 'carback'; Drive here is a teleport, a new anchor), and every far tile waited on the survey's
            # DEM, queued behind the last view's road and tree work: Turin's far field started 35 s late, 2026-11-07)
            anchor.q = VIEW_ANCHOR_Q
            self.s.anchors[anchor.id] = anchor
            if self.s.horizon:
                self.s.horizon_grid(anchor, self.eye if self.eye is not None else 0.0)
            log.info("anchor %d (%.5f, %.5f) for the map's view in %.2f s (no block)", anchor.id, lat, lon,
                     time.perf_counter() - t0)
            progress["done"] = True
            self.send({"type": "anchor_ready", "requestId": req, "anchorId": anchor.id, "lat": lat, "lon": lon,
                       "groundZ": None, "floorZ": None, "terrain": None, "provisional": False, "view": True})
            return
        # (a session's first place, or a jump: the provisional first block; the mod says so since 1.6.3, a teleport
        # under TELEPORT_PAUSE_M away waited 26 s for a graded first block in Manhattan, 2026-10-25)
        declared = msg.get("purpose") == "teleport"
        teleport = old is None or declared
        if old is None and not self.s.boost_on:
            # (joining the world: full priority loading as after a teleport, when the player chose it; the user,
            # 2026-10-31: "setting should trigger on joining the world too")
            self.boost_t0 = time.monotonic()
            self.s.load_boost(True, "joining", full_only=True)
        if old is not None:
            la, lo, _ = old.frame.enu_to_geodetic(*self.focus, 0.0)
            if declared or geodesy.haversine_m(float(la), float(lo), lat, lon) > TELEPORT_PAUSE_M:
                teleport = True
                self.teleport_pending, self.teleport_t0 = True, time.monotonic()
                log.debug("teleport to %.5f, %.5f: the old place's work paused", lat, lon)
                # the cores lent from the request on, not from the first block's arrival: that block is most of
                # the wait (16.9 s in Baltimore on a Steam Deck, the server on its own 4 CPUs, 2026-10-20)
                self.boost_t0 = time.monotonic()
                self.s.load_boost(True, "teleport")
        anchor = self.s.new_anchor(lat, lon)
        # (the map data the first block's grading and the road tiles will ask for, fetched meanwhile)
        asyncio.get_running_loop().run_in_executor(self.s.ui_pool, self.s.warm_place, anchor, cx, cy)
        self.status(f"Building terrain at {lat:.5f}, {lon:.5f}...")
        t0 = time.perf_counter()
        # a teleport's first block without its road plans, the graded one sent when built (make_provisional)
        provisional = teleport and PROVISIONAL_FIRST and self.s.grading and not os.environ.get("BNGE_WORLD_BUNDLE")
        if self.s.surface is not None and not provisional:
            # the anchor's first block is built once and kept while the player stands still: give the
            # land cover around it a moment to arrive (blocks built while driving never wait)
            # (on the map's threads: the default pool is full of the old place's road / vegetation builds after a
            # teleport, and this 2 s wait queued behind them for ~25 s, 2026-10-14)
            tw = time.perf_counter()
            progress["stage"] = "land_cover"
            await asyncio.get_running_loop().run_in_executor(self.s.ui_pool, self.s.surface.ready, lat, lon)
            log.info("anchor %s: land cover wait %.2f s", anchor.id, time.perf_counter() - tw)
        progress["stage"] = "terrain"
        pacer.foreground_begin()        # the player waits for this block: the background loops hold (pacer.hold)
        try:
            if provisional:
                info = await asyncio.get_running_loop().run_in_executor(
                    self.s.hm_pool, self.s.make_provisional, anchor, cx, cy, self._progress(), place or (0.0, 0.0))
            else:
                info = await self.s.terrain(anchor, cx, cy, self._progress())
        finally:
            pacer.foreground_end()
        progress["stage"] = "finalizing"
        key = (anchor.id, cx, cy)
        anchor.first_terrain = (cx, cy)
        self.s.anchors[anchor.id] = anchor
        if self.s.horizon:
            # its zmax grid now, before the commit (a far field selected without it is the HORIZON_MIN disc, uncut)
            self.s.horizon_grid(anchor, self.eye if self.eye is not None else 0.0)
        self.s.hm_sent.append(key)
        # groundZ is None when (0, 0) lies outside the block (a resync around a distant player)
        gz = info["groundZ"]
        if place is not None and (not provisional or gz is None):
            # (a provisional block's groundZ is at the place already: ground_z_at carves the sea floor, which waits on
            # the water cells: 30 s of a fresh place's teleport, 2026-10-25)
            gz = await asyncio.get_running_loop().run_in_executor(self.s.ui_pool, self.s.ground_z_at, anchor, *place)
        # in a tunnel: its floor under the ground, sent after (tunnel_floor_late: the mod moves the car down onto it once
        # the tube's collision is there). It was asked only at the origin of a full block, and since snapped anchors and
        # provisional first blocks (2026-10-25) a teleport into a tunnel left the car on the hill over it.
        floor_z = None
        log.info("anchor %d (%.5f, %.5f) ready in %.2f s, terrain at (%.0f, %.0f), groundZ %s; dem %s",
                 anchor.id, lat, lon, time.perf_counter() - t0, cx, cy,
                 "n/a" if gz is None else f"{gz:.1f}",
                 self.s.dem.stats_snapshot())
        ready = {"type": "anchor_ready", "requestId": req, "anchorId": anchor.id, "lat": lat, "lon": lon,
                 "groundZ": gz, "floorZ": floor_z, "terrain": self.terrain_msg(key, info), "provisional": provisional}
        if place is not None:           # (the player's spot off the origin, echoed: core places the car there)
            ready["px"], ready["py"] = place
        if self.s.gas_stations:         # the shapes of the region's gas stations, preloaded (only that region's)
            ready["fuelShapes"] = gasstationsmod.style_shapes(gasstationsmod.style_of(lat, lon))
        progress["done"] = True
        self.send(ready)
        self.status(f"Terrain ready ({time.perf_counter() - t0:.1f} s)")
        if gz is not None and declared:     # (a player's teleport: not a rebase while driving, nor the level start)
            self.spawn(self.tunnel_floor_late(anchor, place or (0.0, 0.0), gz))
        if provisional:
            self.spawn(self.refine_first(anchor, cx, cy, info.get("ter"), t0))

    async def tunnel_floor_late(self, anchor, place, gz):
        """A teleport's spot in a tunnel: its floor (more than TUNNEL_FLOOR_UNDER_M under the ground gz there) sent as
        'tunnel_floor' (core.floorCheck sets the car down on it once the tube collides); nothing when none."""
        e, n = float(place[0]), float(place[1])
        fl = await asyncio.get_running_loop().run_in_executor(self.s.ui_pool, self.s.tunnel_floor, anchor, e, n)
        if fl is None or fl >= gz - TUNNEL_FLOOR_UNDER_M:
            return
        z = round(fl + gradingmod.ROAD_LIFT, 3)
        log.info("anchor %d: in a tunnel at (%.0f, %.0f), its floor %.1f under the ground %.1f", anchor.id, e, n, z, gz)
        self.send({"type": "tunnel_floor", "anchorId": anchor.id, "floorZ": z, "x": round(e, 3), "y": round(n, 3)})

    async def refine_first(self, anchor, cx, cy, provisional_ter, t0):
        """The graded first block (roads levelled, tunnels, the sea floor) in place of the provisional one: built as
        any block, sent as terrain_refresh (the mod swaps it in while that block is still its own). The provisional
        .ter is deleted a while later (the game may still be reading it)."""
        key = (anchor.id, cx, cy)
        self.refining.add(key)              # (the loading line's ground: send_progress)
        try:
            info = await self.s.terrain(anchor, cx, cy)
        except Exception:
            log.exception("graded first block %s failed (the provisional one stays)", key)
            return
        finally:
            self.refining.discard(key)
        self.s.hm_sent.append(key)
        log.info("anchor %d: graded first block in %.2f s after the request", anchor.id, time.perf_counter() - t0)
        self.send({"type": "terrain_refresh", "anchorId": anchor.id, **self.terrain_msg(key, info)})
        if provisional_ter and provisional_ter != info.get("ter"):
            await asyncio.sleep(PROVISIONAL_KEEP_S)
            self.s._remove(os.path.join(self.s.out_dir, provisional_ter))

    async def prepare_terrain(self, msg):
        req = msg.get("requestId")
        try:
            aid = msg["anchorId"]
            anchor = self.s.anchors[aid]
            cx, cy = self.s.snap(_num(msg, "cx")), self.s.snap(_num(msg, "cy"))
        except (KeyError, TypeError, ValueError) as e:
            self.send({"type": "terrain_error", "requestId": req, "anchorId": msg.get("anchorId"),
                       "error": f"invalid request: {type(e).__name__} {e}"})
            return
        if aid == self.active and self.s.layout.block_at(cx, cy) != self.block:
            # The pending block's L0 tiles stream first, so the commit never waits (§2.3).
            self.pending_block = self.s.layout.block_at(cx, cy)
            self.pending_at = time.monotonic()
            self.pending_answered = False
            t1 = time.perf_counter()
            self.update_far()
            dt = time.perf_counter() - t1
            if dt > 0.1:
                (log.info if dt > 0.5 else log.debug)("terrain request %s: update_far %.2f s", (aid, cx, cy), dt)
        t0 = time.perf_counter()
        self.terrain_wait += 1
        try:
            job = asyncio.ensure_future(self.s.terrain(anchor, cx, cy))
            prov = await self.provisional_follow(anchor, cx, cy, job)
            if prov is not None:
                self.terrain_wait -= 1
                key = (aid, cx, cy)
                log.info("terrain_ready %s after %.0f ms (provisional: the graded block follows)", key,
                         (time.perf_counter() - t0) * 1000)
                self._answered(aid, cx, cy)
                self.send({"type": "terrain_ready", "requestId": req, "anchorId": aid, **self.terrain_msg(key, prov)})
                return
            try:
                info = await asyncio.wait_for(asyncio.shield(job), TERRAIN_STUCK_S)
            except asyncio.TimeoutError:
                # (a game waited 5 min for one block with nothing in the log, GitHub #263)
                log.warning("terrain request %s still building after %.0f s (waiting on %s): it may be stuck",
                            (aid, cx, cy), time.perf_counter() - t0,
                            self.s.hm_stage.get((aid, cx, cy), "a heightmap worker"))
                info = await job
        except Exception as e:
            self.terrain_wait -= 1
            log.exception("terrain request %s failed", req)
            self.send({"type": "terrain_error", "requestId": req, "anchorId": aid,
                       "error": f"{type(e).__name__}: {e}"})
            if aid == self.active and self.pending_block == self.s.layout.block_at(cx, cy):
                self.pending_block = None   # the mod will not commit it: stop forcing L0 there
                self.update_far()
            return
        self.terrain_wait -= 1
        key = (aid, cx, cy)
        self.s.hm_sent.append(key)
        log.info("terrain_ready %s after %.0f ms", key, (time.perf_counter() - t0) * 1000)
        self._answered(aid, cx, cy)
        self.send({"type": "terrain_ready", "requestId": req, "anchorId": aid, **self.terrain_msg(key, info)})

    def _answered(self, aid, cx, cy):
        """The pending block's terrain_ready went: from now on its commit is the mod's to make (the player reaching
        its commit distance, its far tiles going live), which more server cores do not hasten (behind)."""
        if aid == self.active and self.pending_block == self.s.layout.block_at(cx, cy):
            self.pending_answered = True

    async def provisional_follow(self, anchor, cx, cy, job):
        """A follow block's provisional stand-in (PROVISIONAL_FOLLOW_S): None when the graded block came in time (or
        provisional blocks are off), else the provisional block's info, the graded one sent as terrain_refresh when
        built (refine_follow). One provisional build per block (Server.prov_info)."""
        s = self.s
        if PROVISIONAL_FOLLOW_S <= 0 or not PROVISIONAL_FIRST or not s.grading or anchor.id != self.active:
            return None
        key = (anchor.id, cx, cy)
        info = s.prov_info.get(key)     # (asked again: its provisional block at once)
        wait, planned, t_check = 0.0, False, time.perf_counter()
        if info is None:
            # at once where the block's road plans are not made yet (a first visit's cells: 15-120 s of planning);
            # planned, after PROVISIONAL_FOLLOW_S: the graded block is usually quick then, but a first visit's map data
            # (its roads round the block, read on demand) still held one 84 s (nyc_speed.py --cold, 2026-11-02).
            # Raced with the block itself: one built already (the prefetch) goes at once. The check runs on
            # prov_pool and may wait for a planner's lock while a cell is planned: blocks built 15-34 s before their
            # request were answered 2.6-8.8 s after it, each an edge hold at 68 mph (GitHub #725; 11 and 65 s here
            # under load, 2026-11-04). The check gets PROVISIONAL_FOLLOW_S at most: past that it counts as planned,
            # the wait spent.
            pf = asyncio.get_running_loop().run_in_executor(s.prov_pool, s.block_planned, anchor, cx, cy)
            done, _ = await asyncio.wait({job, pf}, timeout=PROVISIONAL_FOLLOW_S, return_when=asyncio.FIRST_COMPLETED)
            if job in done:
                return None
            planned = True
            if pf in done:
                try:
                    planned = pf.result()
                except Exception:
                    pass
            else:
                def slow_check(f, key=key, t=t_check):
                    log.info("block %s: its plan check took %.1f s (a planner busy)", key, time.perf_counter() - t)
                pf.add_done_callback(slow_check)
            if planned:
                wait = max(0.0, PROVISIONAL_FOLLOW_S - (time.perf_counter() - t_check))
        done, _ = await asyncio.wait({job}, timeout=wait)
        if done:
            return None
        pf = s.hm_proc.get(key) if info is None and planned else None
        left = PROVISIONAL_RUNNING_S - (time.perf_counter() - t_check)
        if pf is not None and pf.running() and left > 0:
            # (planned and building already: a provisional block beside it took the CPU it needed, finished after it,
            # 150 mph down Midtown on a warm cache 2026-11-02; only if it runs long, a first visit's map data)
            done, _ = await asyncio.wait({job}, timeout=left)
            if done:
                return None
        if info is None:
            prov = asyncio.get_running_loop().run_in_executor(s.prov_pool, s.make_provisional_follow, anchor, cx, cy)
            done, _ = await asyncio.wait({job, prov}, return_when=asyncio.FIRST_COMPLETED)
            if job in done:
                return None
            try:
                info = prov.result()
            except Exception:
                log.exception("provisional block %s failed (the graded one is waited for)", key)
                return None
            s.prov_info[key] = info
            self.spawn(self.refine_follow(anchor, cx, cy, job, info.get("ter")))
        return info

    async def refine_follow(self, anchor, cx, cy, job, provisional_ter):
        """The graded follow block in place of its provisional one (as refine_first): sent as terrain_refresh, which the
        mod takes for its committed block or the one it is about to commit."""
        key = (anchor.id, cx, cy)
        try:
            info = await job
        except Exception:
            log.exception("graded block %s failed (the provisional one stays)", key)
            return
        finally:
            self.s.prov_info.pop(key, None)
        self.s.hm_sent.append(key)
        log.info("graded block %s sent in place of its provisional one", key)
        self.send({"type": "terrain_refresh", "anchorId": anchor.id, **self.terrain_msg(key, info)})
        if provisional_ter and provisional_ter != info.get("ter"):
            await asyncio.sleep(PROVISIONAL_KEEP_S)
            self.s._remove(os.path.join(self.s.out_dir, provisional_ter))

    def commit_anchor(self, aid):
        if aid == self.active:
            return
        if self.teleport_pending:
            self.teleport_pending = False
            self.wake.set()
        anchor = self.s.anchors.get(aid)
        if anchor is None:
            log.warning("commit of unknown anchor %r ignored", aid)
            return
        old = self.s.anchors.get(self.active)
        if old is not None:
            # Tiles of the old anchor stay visible (the mod moves them); track their
            # footprints in the new frame so they are removed once covered.
            self.shown = {k: farfield.transform_bbox(old.frame, anchor.frame, bb) for k, bb in self.shown.items()}
            fx, fy, _ = anchor.frame.ecef_to_enu(*old.frame.enu_to_ecef(*self.focus, 0.0))
            self.focus = (float(fx), float(fy))
            if self.view_at is not None:
                # the camera away from the car too (the far field's centre until the next pose: left in the old
                # frame, the far field was selected that far off at the commit and global tiles went and came back,
                # Pikes Peak 2026-10-29)
                vx, vy, _ = anchor.frame.ecef_to_enu(*old.frame.enu_to_ecef(*self.view_at, 0.0))
                self.view_at = (float(vx), float(vy))
            log.debug("commit anchor %d: player at (%.0f, %.0f) in its frame, first terrain %s", aid, fx, fy,
                      anchor.first_terrain)
            if anchor.first_terrain is not None:
                bx0, by0, bx1, by1 = self.s.layout.block_at(*anchor.first_terrain).bbox
                if not (bx0 <= fx <= bx1 and by0 <= fy <= by1):
                    # Not a rebase but a teleport: the mod places the player on the new
                    # anchor's first terrain, and its next pose may come only after that.
                    self.focus, self.vel = (float(anchor.first_terrain[0]), float(anchor.first_terrain[1])), (0.0, 0.0)
                    self.view_at = None
                    self.boost_t0 = time.monotonic()
                    self.s.load_boost(True, "teleport")
        else:
            self.shown = {k: bb for k, bb in self.shown.items() if k[0] in self.s.anchors}
            # No previous frame to map the player from (first anchor, or a resync after a server
            # restart, whose terrain is around the player): start at the anchor's first terrain.
            self.focus = tuple(float(v) for v in anchor.first_terrain) if anchor.first_terrain else (0.0, 0.0)
            self.vel = (0.0, 0.0)
            self.view_at = None
        for ck, (fut, _) in self.inflight.items():
            if ck[0] != aid:
                fut.cancel()            # this connection's waiter; the producer keeps its slot until it finishes
        # Tiles sent but not swapped in are dropped by the mod at the anchor switch, but for the global ones (any
        # anchor's: the mod keeps them loading, placed into the new anchor; not after a resync, old is None)
        keep = {k: t for k, t in self.sent.items() if farfield.is_global_key(k)} if old is not None else {}
        self._retire(self.ck(k) for k in self.sent if k not in keep)
        self.active = aid
        self.ppolys = {}
        self.desired, self.sent, self.groups, self.queue, self.qi = {}, keep, [], [], 0
        old_grid = self.s.horizon_grids.get(old.id) if old is not None else None
        self.prev_horizon = (old.frame, old_grid) if old_grid is not None and old_grid.merc is not None else None
        # (the global far tiles' split state is any anchor's: kept, so their hysteresis holds across the rebase)
        self.refined = {n for n in self.refined if len(n) == 4}
        self.node_changed = {n: t for n, t in self.node_changed.items() if len(n) == 4}
        self.failed.clear()
        self.block = self.pending_block = self.predicted_block = None
        if anchor.first_terrain is not None:
            self._set_block(aid, *anchor.first_terrain)
        log.info("commit anchor %d", aid)
        if self.map_view or self.map_open:
            # the 3D map at a new place: coarse first, then finer in steps as each is done (Conn._stream), as map apps
            # load; the finest first left the screen empty for seconds round the first tiles (2026-11-07)
            self.lod_ramp = MAP_RAMP_START
            self.zoom_ref = self.eye        # (the zoom rule's reference: this anchor's own ramp, not a second one)
        self.update_far()

    def _set_block(self, aid, cx, cy):
        self.block = self.s.layout.block_at(cx, cy)
        if self.pending_block == self.block:
            self.pending_block = None
        key = (aid, cx, cy)
        if key in self.s.hm_committed:
            self.s.hm_committed.remove(key)
        self.s.hm_committed.append(key)
        self.s.evict_heightmaps()

    def behind(self):
        """Is the world behind the driving player? The next block asked CATCHUP_AFTER_S ago and not answered yet (still
        building here), or the road tiles round the car not all built. A block answered but not committed is not:
        the mod commits it when the player reaches its commit distance (and its far tiles are live), which more cores
        here do not hasten. Counting it lent half the game's cores to the server again and again in town: blocks
        answered 0-4 s after their request, the boost on 13-22 s after it, at 20-60 mph (Fells Point, 2026-11-07)."""
        if self.pending_block is not None and self.pending_block != self.block and self.pending_at is not None \
                and not getattr(self, "pending_answered", False) \
                and time.monotonic() - self.pending_at > CATCHUP_AFTER_S:
            return True
        if self.s.mapdata is not None:
            try:
                done, want = self.roads.progress(near_m=CATCHUP_ROADS_M)
            except TypeError:           # (a tracker without the radius)
                done, want = self.roads.progress()
            return done < want
        return False

    def hold_boost(self, holding):
        """While the edge guard holds the car for its ground (pose.guardHold), the server on every core and the game on
        its few (Server.split_boost, the full priority split): the game is paused then, and the block it waits for
        came 36 s after a first visit's start at 100 mph with the server's own cores, ~8 s with every core
        (nyc_speed.py --cold 2026-11-02). Back as before HOLD_BOOST_CALM_S after the hold ends."""
        now = time.monotonic()
        if holding:
            self.hold_seen = now
            if not getattr(self, "hold_split", False):
                self.hold_split = self.s.split_boost(True, "edge hold")
        elif getattr(self, "hold_split", False) and now - getattr(self, "hold_seen", 0.0) > HOLD_BOOST_CALM_S:
            self.s.split_boost(False, "hold over")
            self.hold_split = False

    def catch_up(self, speed):
        """The loading boost (Server.load_boost: half of the game's cores lent) while the world is behind a driving
        player, off CATCHUP_CALM_S after it caught up. Not over a teleport's own boost."""
        if CATCHUP_LEND_SHARE <= 0:         # (BNGE_CATCHUP_LEND=0: none)
            return
        if self.catchup and not self.s.boost_on:
            self.catchup = False            # (ended elsewhere: free to start again)
            _perf(self.s).catchup(False)
        if self.s.boost_on and not self.catchup:
            return
        behind = speed > CATCHUP_MIN_MS and self.behind()
        now = time.monotonic()
        if behind:
            self.caught_up_at = None
            if not self.catchup:
                self.s.load_boost(True, "catching up", half_only=True)
                self.catchup = self.s.boost_on
                self.catchup_since, self.catchup_more = now, False
                if self.catchup:
                    _perf(self.s).catchup(True)
                    log.info("catch-up boost on (%.0f mph; next block asked %.1f s ago)", speed * 2.237,
                             now - self.pending_at if self.pending_at else -1)
            elif not getattr(self, "catchup_more", False) and now - getattr(self, "catchup_since", now) > CATCHUP_MORE_S \
                    and self.s.boost_cpus and not getattr(self.s, "split_on", False):
                # (behind all along: the larger lend, cpupin.boosted's half. A first visit at 100 mph on one core more:
                # the road tiles were never ahead of the car, 216 of 232 polls, nyc_speed.py 2026-11-02)
                try:
                    cpupin.pin_tree(self.s.boost_cpus)
                    self.catchup_more = True
                    log.info("catch-up boost: more cores (behind for %.0f s): CPUs %s", now - self.catchup_since,
                             ",".join(map(str, sorted(self.s.boost_cpus))))
                except OSError as e:
                    log.warning("catch-up boost: %s", e)
                    self.catchup_more = True
        elif self.catchup:
            if self.caught_up_at is None:
                self.caught_up_at = now
            elif now - self.caught_up_at > CATCHUP_CALM_S:
                self.s.load_boost(False, "caught up")
                self.catchup, self.catchup_more = False, False
                _perf(self.s).catchup(False)
                log.info("catch-up boost off (caught up)")

    def commit_terrain(self, msg):
        aid = msg.get("anchorId")
        cx, cy = self.s.snap(_num(msg, "cx")), self.s.snap(_num(msg, "cy"))
        if aid != self.active:
            if aid in self.s.anchors and (self.active is None or aid > self.active):
                self.commit_anchor(aid)
            else:
                log.warning("commit_terrain for anchor %r (active %r) ignored", aid, self.active)
                return
        if self.block == self.s.layout.block_at(cx, cy):
            return
        # Far-tile keys do not depend on the block; only the forced L0 area moves.
        self._set_block(aid, cx, cy)
        self.update_far()

    def slots_gone(self, gone):
        """pose.slotsGone [[slot, serial], ...]: the mod's objects (or never-spawned instances) of those far_tile
        messages are gone; a slot whose latest send is among them is no longer held by this game."""
        lease = getattr(self, "lease", None)
        if lease is not None and lease.connection is not self:
            return
        if isinstance(gone, dict):      # an empty Lua table may encode as {}
            gone = list(gone.values())
        if not isinstance(gone, list):
            return
        let_go = []
        for g in gone:
            if isinstance(g, dict):
                g = [g.get("1"), g.get("2")]
            if not isinstance(g, (list, tuple)) or len(g) != 2:
                continue
            n, serial = g
            if streamleases.integer(n) and streamleases.integer(serial) and self.slot_holds.get(int(n)) == serial:
                del self.slot_holds[int(n)]
                let_go.append(int(n))
        self.s.slots_let_go(let_go)
