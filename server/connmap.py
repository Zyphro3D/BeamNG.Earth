"""World server connection: the map, GPS routes, places and labels, areas, server list, updates and add-ons
(split out of earthd.py's Connection).
"""
import os
import asyncio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402
import areas as areasmod  # noqa: E402
import randomplace as randomplacemod  # noqa: E402
import placepreview as placepreviewmod  # noqa: E402
import placenames as placenamesmod  # noqa: E402
import maplabels as maplabelsmod  # noqa: E402
import mapdata as mapdatamod  # noqa: E402
import dem as demmod  # noqa: E402
import routeahead as routeaheadmod  # noqa: E402
import heightmap  # noqa: E402
import imagery as imagerymod  # noqa: E402
import maptiles as maptilesmod  # noqa: E402
import geocode as geocodemod  # noqa: E402
import routing as routingmod  # noqa: E402
import updates as updatesmod  # noqa: E402
import netfetch  # noqa: E402
import earthbase
from earthbase import (
    AHEAD_LAYERS,
    AHEAD_MARGIN_M,
    BASE_RENDER_PX,
    MAP_FOCUS_MAX_M,
    PLACE_PREVIEWS_MAX,
    ROUTE_DOWNLOAD_MODES,
    ROUTE_IMG_TRIES,
    ROUTE_RETRY_S,
    ROUTE_WHOLE_MAX_M,
    SERVER_LIST_URL,
    WHERE_PLACE_RETRY_S,
    WHERE_PLACE_TRIES,
    log,
    route_profile_smooth,
)


class _MapMixin:
    """Connection's map, routes, places, updates and add-ons (split out of earthd.py; Connection inherits it)."""

    async def map_tiles(self, msg):
        """The world map's tiles {"tiles": [[z, x, y, layer?], ...]} (layer "sat" by default, "map", "labels"; the
        most wanted first): each written once under <user>/temp/beamngEarth/maptiles/ (maptiles.tile_path; the
        game's UI loads it from there: its browser cancels every http request, even to 127.0.0.1) and said as soon
        as it is there, map_tiles_ready {"tiles": [[z, x, y, ok, layer]]}. Built maptiles.FILE_WORKERS at a time
        (one by one, a 48-tile view took ~10 s and was said only at the end, 2026-10-16); each request replaces
        the last one's queue (maptiles.TileFiles; an empty list cancels)."""
        mt = getattr(self.s, "map_tiles", None)
        if mt is None:
            return
        want = []
        tiles = msg.get("tiles")
        if not isinstance(tiles, list):         # (an empty Lua table may arrive as {})
            tiles = []
        for t in tiles[:maptilesmod.MAX_FILES_REQUEST]:
            try:
                z, x, y = maptilesmod.check_tile(*t[:3])
                layer = t[3] if len(t) > 3 else "sat"
            except (TypeError, ValueError, IndexError):
                continue
            if isinstance(layer, str) and layer in maptilesmod.LAYERS:
                want.append((layer, z, x, y))
        files = self.s.map_files
        if files is None:
            files = self.s.map_files = maptilesmod.TileFiles(mt, os.path.join(self.s.root, "maptiles"))
        loop, send = self.loop, self.send

        def ready(key, ok):                     # (a worker thread) the connection that asked last hears of it
            layer, z, x, y = key
            loop.call_soon_threadsafe(send, {"type": "map_tiles_ready", "tiles": [[z, x, y, ok, layer]]})
        files.on_ready = ready
        files.request(want)

    async def route(self, msg):
        """{"id", "lat0", "lon0", "lat1", "lon1", "download"?} -> route_result {"id", "points": [[lat, lon]...], "dist",
        "dur", "elev": [[dist_m, elev_m], ...]?, "error"?}. Also (re)starts this route's corridor prefetch
        (route_download, routeahead.py; download: "off", "ahead" (ROUTE_AHEAD_MAX_M, the default) or "route" (the
        whole route, up to ROUTE_WHOLE_MAX_M))."""
        self.route_dl_version += 1
        version = self.route_dl_version
        out = {"type": "route_result", "id": msg.get("id"), "points": [], "dist": 0.0, "dur": 0.0}
        try:
            r = await asyncio.get_running_loop().run_in_executor(
                self.s.ui_pool, routingmod.plan, float(msg["lat0"]), float(msg["lon0"]), float(msg["lat1"]),
                float(msg["lon1"]), self.s.net, msg.get("bearing"))
            out.update(r)
        except Exception as e:          # offline, no route, rate limited: the mod falls back to steering
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        if "error" not in out and self.s.dem is not None and out["points"]:
            try:                        # the map's elevation profile: a nice-to-have, the route works without it
                out["elev"] = await asyncio.get_running_loop().run_in_executor(
                    self.s.ui_pool, self._route_elevation, out["points"])
            except Exception as e:
                log.debug("route elevation: %s: %s", type(e).__name__, e)
        log.info("route %d (%s, %s -> %s, %s): %d points, %.1f km%s", int(msg.get("id") or 0), msg.get("lat0"),
                 msg.get("lon0"), msg.get("lat1"), msg.get("lon1"), len(out["points"]), out["dist"] / 1000,
                 (" (" + out["error"] + ")") if "error" in out else "")
        self.send(out)
        self.route_points = out["points"] if "error" not in out else []
        mode = msg.get("download")
        self.route_download(version, out["points"], mode if mode in ROUTE_DOWNLOAD_MODES else "ahead")

    def _route_elevation(self, points):
        """The route's elevation profile for the map (routing.resample onto even steps, dem.sample): [[dist_m,
        elev_m], ...]. Blocking (run_in_executor): called from route() only."""
        aid = self.active
        zoom = self.s.anchors[aid].hm_zoom if aid in self.s.anchors else demmod.FINE_ZOOM
        pts = routingmod.resample([(p[0], p[1]) for p in points])
        lats = np.array([p[0] for p in pts])
        lons = np.array([p[1] for p in pts])
        # (tiles about 4 steps wide, no finer than the terrain's: at the terrain's zoom a long route's ~120 points
        # each fetched their own tile before the route reached the map; a 220 px chart needs no more)
        step = pts[-1][2] / max(len(pts) - 1, 1) if len(pts) > 1 else 0.0
        if step > 0:
            tile_m = 40075016.7 * max(math.cos(math.radians(float(np.mean(lats)))), 0.05)
            zoom = int(min(zoom, max(8, math.floor(math.log2(tile_m / (4 * step))))))
        elev = route_profile_smooth(self.s.dem.sample(lats, lons, zoom))
        return [[round(p[2], 1), round(float(e), 1)] for p, e in zip(pts, elev)]

    def route_download(self, version, points, mode="ahead"):
        """Prefetch the route's corridor (elevation tiles and road/building map cells) up to
        ROUTE_AHEAD_MAX_M ahead (mode "ahead"; "route": ROUTE_WHOLE_MAX_M, "off": none), at "prefetch" priority,
        pushing route_download {total, done, active} (throttled like Server._progress) until it finishes or a newer
        route() supersedes it (checked by version, since self.route_pool's single worker already serializes a newer
        job behind this one)."""
        empty = {"type": "route_download", "total": 0, "done": 0, "active": False}
        if not earthbase.ROUTE_AHEAD_ON or mode == "off" or self.s.dem is None or len(points) < 2:
            self.send(empty)
            return
        aid = self.active
        zoom = self.s.anchors[aid].hm_zoom if aid in self.s.anchors else demmod.FINE_ZOOM
        reach = self.s.layout.block * heightmap.SQUARE / 2 + AHEAD_MARGIN_M
        layers = (AHEAD_LAYERS if self.s.buildings else ("roads",)) if self.s.mapdata is not None else ()
        pts = routeaheadmod.arc_length_points(points, ROUTE_WHOLE_MAX_M if mode == "route" else earthbase.ROUTE_AHEAD_MAX_M)
        dem_keys, cell_layer_pairs = routeaheadmod.need(self.s.dem, self.s.mapdata, pts, reach, zoom, layers)
        imagery = getattr(self.s, "imagery", None)
        if len(dem_keys) + len(cell_layer_pairs) == 0 and imagery is None:
            self.send(empty)
            return
        dem, mapdata = self.s.dem, self.s.mapdata
        texel = self.s.layout.block * heightmap.SQUARE / BASE_RENDER_PX

        def run():
            # the ground photo's high-resolution tiles along the corridor too (Imagery.route_keys, the blocks' base
            # photo zoom): the pre-cache fetched only heights and map cells, and a pre-cached long drive still
            # stopped to download every few km at 40-55 mph (GitHub #755); listed here, off the event loop
            img_keys = []
            if imagery is not None and pts:
                try:
                    img_keys = imagery.route_keys(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), reach,
                                                  texel)
                except Exception as e:
                    log.debug("route download: imagery keys: %s: %s", type(e).__name__, e)
            total = len(dem_keys) + len(cell_layer_pairs) + len(img_keys)
            if total == 0:
                self.loop.call_soon_threadsafe(self.send, empty)
                return
            # counted once, up front: an already-cached key is skipped below (has() is re-checked only to
            # decide what to fetch), never re-counted, or a route with nothing left to do would inflate past
            # total and spam a push every iteration (push's own throttle only skips while done < total)
            done = routeaheadmod.done_count(dem, mapdata, dem_keys, cell_layer_pairs)
            done += sum(1 for k in img_keys if imagery.key_done(k))
            last_push = [0.0]

            def push(active):
                now = time.monotonic()
                if active and done < total and now - last_push[0] < 0.25:
                    return
                last_push[0] = now
                self.loop.call_soon_threadsafe(
                    self.send, {"type": "route_download", "total": total, "done": done, "active": active})
            if done >= total:           # the whole corridor is already cached: nothing to show a bar for
                push(False)
                return
            push(True)
            for z, x, y in dem_keys:
                if self.route_dl_version != version:
                    return
                if dem.has(z, x, y):
                    continue
                try:
                    got, deferred = dem.prefetch([(z, x, y)])
                except Exception as e:
                    log.debug("route download: %s: %s", type(e).__name__, e)
                    continue
                if deferred:             # no spare tokens now: wait rather than give up on the corridor
                    time.sleep(ROUTE_RETRY_S)
                    continue
                done += got
                push(True)
            for key, layer in cell_layer_pairs:
                if self.route_dl_version != version:
                    return
                if mapdata.has(key, layer):
                    continue
                try:
                    mapdata.cell(key, layers=(layer,), kind="prefetch")
                    done += 1
                    push(True)
                except mapdatamod.DEFERRED:
                    time.sleep(ROUTE_RETRY_S)
                except Exception as e:
                    log.debug("route download: %s: %s", type(e).__name__, e)
            for key in img_keys:
                for _try in range(ROUTE_IMG_TRIES):     # (a deferred one asked again: an overnight pre-cache left no holes)
                    if self.route_dl_version != version:
                        return
                    try:
                        if imagery.fetch_key(key):
                            done += 1
                            push(True)
                        break
                    except imagerymod.ImageryError as e:
                        if not e.deferred:
                            break
                        time.sleep(ROUTE_RETRY_S)
                    except Exception as e:
                        log.debug("route download: imagery: %s: %s", type(e).__name__, e)
                        break
            push(False)
        self.s.route_pool.submit(run)

    async def server_list(self):
        """The BeamNG.Earth server list (native/serverlist, SERVER_LIST_URL) for the mod's tab in BeamMP's server browser
        (serverlist.lua): fetched here, as the game's Lua may only open connections to local addresses ("connect
        restricted"). A 'server_list' message with the servers, or with an error."""
        def get():
            return json.loads(self.s.net.get(SERVER_LIST_URL, kind="demand").body.decode("utf-8", "replace"))
        try:
            servers = await asyncio.get_running_loop().run_in_executor(self.s.ui_pool, get)
        except Exception as e:          # offline, the list down: said in the tab
            log.info("server list: %s: %s", type(e).__name__, e)
            self.send({"type": "server_list", "error": "the server list did not answer"})
            return
        self.send({"type": "server_list", "servers": servers if isinstance(servers, list) else []})

    async def check_update(self):
        """The update check after a hello: an 'update' message when GitHub has a newer version (updates.py); offline or
        up to date: nothing."""
        try:
            r = await asyncio.get_running_loop().run_in_executor(
                self.s.ui_pool, updatesmod.check, updatesmod.local_version(), self.s.net)
        except Exception as e:          # never a reason to drop the connection
            log.debug("update check: %s: %s", type(e).__name__, e)
            return
        if r:
            import selfupdate
            b = selfupdate.blocked()
            if b and not updatesmod.newer(r["latest"], b["version"]):
                # (Windows would not start that version: offered again, it would be downloaded and blocked again)
                log.info("update %s not offered: Windows blocked it (%s)", r["latest"], b["by"] or "?")
                return
            log.info("update available: %s -> %s (%s)", r["current"], r["latest"], r["url"])
            self.send({"type": "update", **r})

    async def update_now(self):
        """The update notice's "Update now": the latest release's file for this packaged app downloaded beside it and
        checked (selfupdate.download), with update_status {state: progress | ready | error, text, done, total}."""
        import selfupdate
        loop = asyncio.get_running_loop()
        last = [0.0]

        def progress(done, total):
            now = time.monotonic()
            if now - last[0] >= 0.5:
                last[0] = now
                loop.call_soon_threadsafe(self.send, {"type": "update_status", "state": "progress", "done": done,
                                                      "total": total, "text": "Downloading the update..."})
        self.send({"type": "update_status", "state": "progress", "done": 0, "total": 0, "text": "Asking GitHub..."})
        try:
            version, path = await loop.run_in_executor(self.s.ui_pool, selfupdate.download, self.s.net,
                                                       updatesmod.local_version(), None, progress)
        except selfupdate.UpdateError as e:
            log.info("update now: %s", e)
            self.send({"type": "update_status", "state": "error", "text": str(e)})
            return
        except Exception as e:          # never a reason to drop the connection
            log.warning("update now: %s: %s", type(e).__name__, e)
            self.send({"type": "update_status", "state": "error", "text": f"{type(e).__name__}: {e}"[:200]})
            return
        log.info("update now: %s downloaded to %s", version, path)
        self.send({"type": "update_status", "state": "ready", "version": version,
                   "text": f"BeamNG.Earth {version} is downloaded. Close the game: the next time you start "
                           f"BeamNG.Earth it starts {os.path.basename(path)}."})

    async def addons_list(self, refresh=False):
        """The Add-ons page's list: 'addons' {addons: [card...], error?} (addons.Addons.cards)."""
        try:
            # (the loop's own pool, not ui_pool: GitHub can take seconds, the map's tiles must not wait behind it)
            cards = await asyncio.get_running_loop().run_in_executor(None, self.s.addons().cards, refresh)
            self.send({"type": "addons", "addons": cards})
        except Exception as e:          # never a reason to drop the connection
            log.warning("add-ons: list: %s: %s", type(e).__name__, e)
            self.send({"type": "addons", "addons": [], "error": f"{type(e).__name__}: {e}"[:200]})

    async def addon_detail(self, aid, refresh=False):
        """One add-on's page: 'addon_detail' {id, card, readme, readmeLinks, releases, error} (addons.Addons.detail)."""
        if not isinstance(aid, str):
            return
        try:
            d = await asyncio.get_running_loop().run_in_executor(None, self.s.addons().detail, aid, refresh)
        except Exception as e:
            log.info("add-ons: detail %r: %s: %s", aid, type(e).__name__, e)
            d = {"id": aid, "readme": "", "readmeLinks": [], "releases": [], "error": str(e)[:200]}
        self.send({"type": "addon_detail", **d})

    async def addon_install(self, aid, consent, mp):
        """Install / Update: the release downloaded into the mods folder and checked (addons.Addons.stage), with
        addon_status {id, state: progress | ready | error, text, done, total}; then addon_apply for the mod, which hands
        it to the game's mod manager."""
        if not isinstance(aid, str):
            return
        loop = asyncio.get_running_loop()
        last = [0.0]

        def progress(done, total):
            now = time.monotonic()
            if now - last[0] >= 0.5:
                last[0] = now
                loop.call_soon_threadsafe(self.send, {"type": "addon_status", "id": aid, "state": "progress",
                                                      "done": done, "total": total, "text": "Downloading..."})
        self.send({"type": "addon_status", "id": aid, "state": "progress", "done": 0, "total": 0,
                   "text": "Asking GitHub..."})
        import addons as addonsmod
        try:
            apply = await loop.run_in_executor(None, lambda: self.s.addons().stage(aid, consent, mp, progress))
        except addonsmod.AddonError as e:
            log.info("add-ons: install %s: %s", aid, e)
            self.send({"type": "addon_status", "id": aid, "state": "error", "text": str(e)})
            return
        except Exception as e:
            log.warning("add-ons: install %s: %s: %s", aid, type(e).__name__, e)
            self.send({"type": "addon_status", "id": aid, "state": "error", "text": f"{type(e).__name__}: {e}"[:200]})
            return
        if apply is None:               # (a program beside the game: installed by the server itself)
            self.send({"type": "addon_status", "id": aid, "state": "done", "text": "Installed. Press Start to run it."})
            await self.addons_list()
            return
        self.send({"type": "addon_status", "id": aid, "state": "ready", "text": "Downloaded and checked: installing..."})
        self.send(apply)

    async def addon_add(self, text):
        """Add from GitHub: the pasted repository put on the local list (addons.Addons.add_repo), with
        addon_added {ok, id | error}, then the list again."""
        import addons as addonsmod
        try:
            aid = await asyncio.get_running_loop().run_in_executor(None, self.s.addons().add_repo, text)
            self.send({"type": "addon_added", "ok": True, "id": aid})
        except addonsmod.AddonError as e:
            self.send({"type": "addon_added", "ok": False, "error": str(e)})
        except Exception as e:
            log.warning("add-ons: add %r: %s: %s", text, type(e).__name__, e)
            self.send({"type": "addon_added", "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
        await self.addons_list()

    async def addon_options(self, aid, values):
        """An add-on's Options tab saved (addons.Addons.set_options: checked, kept; a running program restarted with
        them), with addon_status, then the list again."""
        import addons as addonsmod
        if not isinstance(aid, str):
            return
        try:
            again = await asyncio.get_running_loop().run_in_executor(None, self.s.addons().set_options, aid, values)
            self.send({"type": "addon_status", "id": aid, "state": "done",
                       "text": "Saved; restarted with them." if again else "Saved."})
        except addonsmod.AddonError as e:
            self.send({"type": "addon_status", "id": aid, "state": "error", "text": str(e)})
        except Exception as e:
            log.warning("add-ons: options %s: %s: %s", aid, type(e).__name__, e)
            self.send({"type": "addon_status", "id": aid, "state": "error", "text": f"{type(e).__name__}: {e}"[:200]})
        await self.addons_list()

    async def addon_forget(self, aid):
        """An add-on the player added taken off the list (addons.Addons.forget)."""
        import addons as addonsmod
        if not isinstance(aid, str):
            return
        try:
            await asyncio.get_running_loop().run_in_executor(None, self.s.addons().forget, aid)
        except addonsmod.AddonError as e:
            self.send({"type": "addon_status", "id": aid, "state": "error", "text": str(e)})
        await self.addons_list()

    async def addon_run(self, aid, op):
        """A program beside the game (kind companion): Start / Stop / Remove / autostart_on / autostart_off
        (addons.Addons.run_op), with addon_status, then the list again."""
        if not isinstance(aid, str) or op not in ("start", "stop", "uninstall", "autostart_on", "autostart_off"):
            return
        import addons as addonsmod
        try:
            await asyncio.get_running_loop().run_in_executor(None, self.s.addons().run_op, aid, op)
            text = {"start": "Running.", "stop": "Stopped.", "uninstall": "Removed.", "autostart_on":
                    "Starts with BeamNG.Earth.", "autostart_off": "Starts only when you press Start."}[op]
            self.send({"type": "addon_status", "id": aid, "state": "done", "text": text})
        except addonsmod.AddonError as e:
            self.send({"type": "addon_status", "id": aid, "state": "error", "text": str(e)})
        except Exception as e:
            log.warning("add-ons: %s %s: %s: %s", op, aid, type(e).__name__, e)
            self.send({"type": "addon_status", "id": aid, "state": "error", "text": f"{type(e).__name__}: {e}"[:200]})
        await self.addons_list()

    async def addon_applied(self, aid, op, ok, error):
        """The mod's answer after the game's mod manager installed / updated / enabled / disabled / removed the add-on:
        logged (addons.Addons.applied), then the list again."""
        if not isinstance(aid, str) or op not in ("install", "update", "uninstall", "enable", "disable"):
            return
        try:
            self.s.addons().applied(aid, op, ok, error if isinstance(error, str) else "")
        except Exception as e:
            log.info("add-ons: applied %r: %s", aid, e)
        await self.addons_list()

    async def place_previews(self, msg):
        """{"places": [{"key", "lat", "lon"}]} -> place_previews_ready {"keys"}: each place's picture for the level
        select, written into the game's user folder (placepreview.write) when the server has imagery."""
        items = [(p.get("key"), p.get("lat"), p.get("lon")) for p in (msg.get("places") or []) if isinstance(p, dict)]
        if not items or self.s.imagery is None:
            return
        try:
            keys = await asyncio.get_running_loop().run_in_executor(
                self.s.ui_pool, placepreviewmod.write, self.s.imagery, self.s.user_dir, items[:PLACE_PREVIEWS_MAX])
        except Exception as e:
            log.warning("place previews: %s: %s", type(e).__name__, e)
            return
        self.send({"type": "place_previews_ready", "keys": keys})

    async def random_place(self, msg):
        """{"id"} -> random_place_result {"id", "lat", "lon", "name"} on a road near a random town (randomplace.pick),
        or without lat / lon when none was found (the mod then starts where it would have)."""
        out = {"type": "random_place_result", "id": msg.get("id")}
        md = getattr(self.s, "mapdata", None)
        if md is not None:
            try:
                got = await asyncio.get_running_loop().run_in_executor(self.s.ui_pool, randomplacemod.pick, md)
            except Exception as e:
                log.warning("random place: %s: %s", type(e).__name__, e)
                got = None
            if got is not None:
                out["lat"], out["lon"], out["name"] = got
                log.info("random place: %s (%.5f, %.5f)", got[2], got[0], got[1])
        self.send(out)

    async def where(self, msg):
        """The status bar's location {"id", "lat", "lon", "online"} -> where_result {"id", "street", "place", "region",
        "country"}: the street from the cached map cells (never waits for a download), the town, state and country from
        our own copy of Overture's divisions (placenames.py; GitHub #602: no outside geocoder is asked), only when
        "online" (the mod's setting, and the place due: moved ~1.5 km or 2 min). A region not on disk yet is fetched at
        low priority after the answer: its place parts come in a second where_result once they are there."""
        try:
            lat, lon = float(msg["lat"]), float(msg["lon"])
        except (KeyError, TypeError, ValueError):
            return
        out = {"type": "where_result", "id": msg.get("id"), "street": ""}
        loop = asyncio.get_running_loop()
        if self.s.surface is not None:
            def street():
                layers, _ = self.s.surface.roads(lat, lon)
                return geocodemod.nearest_street(layers, lat, lon, routes=self.s.surface.road_routes(lat, lon))
            try:
                out["street"] = await loop.run_in_executor(self.s.ui_pool, street)
            except Exception as e:      # a cell being written, an odd feature: no street this time
                log.debug("where: street: %s: %s", type(e).__name__, e)
        if msg.get("online"):
            self._where_seq = seq = getattr(self, "_where_seq", 0) + 1
            pn = self.s.place_names()
            try:
                if await loop.run_in_executor(self.s.place_pool, pn.ready, lat, lon):
                    out.update(await loop.run_in_executor(self.s.place_pool, pn.lookup, lat, lon))
                else:
                    self.spawn(self._where_place_later(dict(out), lat, lon, seq))
            except Exception as e:      # a damaged cell file, odd data: the bar shows the street alone
                log.debug("where: place: %s: %s", type(e).__name__, e)
        self.send(out)

    def on_map_focus(self, msg):
        """map_focus {anchorId, x, y, eye, seconds}: the 3D map's flight ends at (x, y) with the eye that high; until
        then the far field is built there (on_pose) and at once now."""
        try:
            aid, x, y = int(msg["anchorId"]), float(msg["x"]), float(msg["y"])
            eye, secs = float(msg.get("eye") or 2000.0), min(10.0, float(msg.get("seconds") or 5.0))
        except (KeyError, TypeError, ValueError):
            return
        if not all(math.isfinite(v) for v in (x, y, eye, secs)) or math.hypot(x, y) > MAP_FOCUS_MAX_M:
            return          # (far from the origin: a rebase there is on its way, its own anchor's hint follows)
        self.focus_hint = (aid, x, y, max(0.0, eye), time.monotonic() + max(0.5, secs))
        if aid == self.active:
            self.view_at, self.eye = (x, y), max(0.0, eye)
            self.update_far()

    def _map_label_rows(self, lat, lon, zoom, radius, fetch):
        """maplabels' rows for the view (worker thread): (rows, complete). Without fetch only the cells on disk;
        complete is False when some were missing (fetch=True then asks for them as background jobs)."""
        rows = maplabelsmod.world_labels(lat, lon, zoom, radius)
        complete = True
        if zoom >= 7.5:
            towns = self.s.place_names().labels(lat, lon, min(radius, placenamesmod.LABELS_RADIUS_MAX_M),
                                                fetch=fetch)
            rows += maplabelsmod.town_labels(towns, rows)
        md = self.s.mapdata
        for layer, z0, r in (("hoods", maplabelsmod.HOOD_MIN_Z, maplabelsmod.HOOD_RADIUS_M),
                             ("roads", maplabelsmod.STREET_MIN_Z, maplabelsmod.STREET_RADIUS_M),
                             ("places", maplabelsmod.POI_MIN_Z, maplabelsmod.POI_RADIUS_M)):
            if md is None or zoom < z0:
                continue
            cells = []
            for k in maplabelsmod.near_cells(lat, lon, min(r, radius)):
                if md.has(k, layer) or fetch:
                    try:
                        cells.append(md.cell(k, layers=(layer,), kind="background").layers.get(layer))
                    except mapdatamod.DEFERRED:
                        complete = False
                else:
                    complete = False
            r = min(r, radius)
            rows += (maplabelsmod.street_labels(cells, lat, lon, zoom, r) if layer == "roads"
                     else maplabelsmod.hood_labels(cells, lat, lon, r) if layer == "hoods"
                     else maplabelsmod.poi_labels(cells, lat, lon, r))
        return rows, complete

    async def map_labels(self, msg):
        """The 3D map's labels {"id", "lat", "lon", "zoom", "radius"} -> map_labels_result {"id", "rows": [[kind, text,
        lat, lon, min zoom, max zoom, rank, extra, bearing], ...]} (maplabels.py). The cells on disk answer at once;
        when some are missing they are fetched as background jobs and a second answer follows, unless a newer request
        came meanwhile."""
        try:
            lat, lon = float(msg["lat"]), float(msg["lon"])
            zoom, radius = float(msg.get("zoom") or 10.0), float(msg.get("radius") or 20000.0)
        except (KeyError, TypeError, ValueError):
            return
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(zoom) and math.isfinite(radius)):
            return
        self._map_labels_seq = seq = getattr(self, "_map_labels_seq", 0) + 1
        loop = asyncio.get_running_loop()
        try:
            rows, complete = await loop.run_in_executor(self.s.place_pool, self._map_label_rows, lat, lon, zoom,
                                                        radius, False)
        except Exception as e:
            log.info("map labels: %s: %s", type(e).__name__, e)
            return
        self.send({"type": "map_labels_result", "id": msg.get("id"), "rows": rows})
        for _ in range(WHERE_PLACE_TRIES):
            if complete or getattr(self, "_map_labels_seq", 0) != seq:
                return
            try:
                rows, complete = await loop.run_in_executor(self.s.place_pool, self._map_label_rows, lat, lon, zoom,
                                                            radius, True)
            except mapdatamod.DEFERRED:
                await asyncio.sleep(WHERE_PLACE_RETRY_S)
                continue
            except Exception as e:
                (log.debug if isinstance(e, netfetch.FetchError) else log.info)("map labels: %s: %s",
                                                                                  type(e).__name__, e)
                return
            if getattr(self, "_map_labels_seq", 0) == seq:
                self.send({"type": "map_labels_result", "id": msg.get("id"), "rows": rows})
            if not complete:
                await asyncio.sleep(WHERE_PLACE_RETRY_S)

    async def _where_place_later(self, out, lat, lon, seq):
        """where()'s place parts for a region not on disk yet: fetched as a background job (netfetch "background":
        never ahead of the world's own downloads), asked again every WHERE_PLACE_RETRY_S while Overture's hosts are
        busy, sent in a where_result of their own unless the player asked for another place meanwhile."""
        loop = asyncio.get_running_loop()
        pn = self.s.place_names()
        for _ in range(WHERE_PLACE_TRIES):
            if getattr(self, "_where_seq", 0) != seq:
                return
            try:
                place = await loop.run_in_executor(self.s.place_pool, pn.lookup, lat, lon, "background")
            except mapdatamod.DEFERRED as e:     # busy (the world's downloads first), rate limited: later
                log.debug("where: place names deferred: %s: %s", type(e).__name__, e)
                await asyncio.sleep(WHERE_PLACE_RETRY_S)
                continue
            except Exception as e:      # offline, the data unreadable: the bar shows the street alone
                (log.debug if isinstance(e, netfetch.FetchError) else log.info)(
                    "where: place names: %s: %s", type(e).__name__, e)
                return
            if getattr(self, "_where_seq", 0) == seq:
                out.update(place)
                self.send(out)
            return

    async def geocode(self, msg):
        """The map's search {"id", "q", "lat"?, "lon"?, "full"?} -> geocode_result {"id", "results": [...], "error"?}."""
        q = str(msg.get("q") or "")[:200]
        lat, lon = msg.get("lat"), msg.get("lon")
        out = {"type": "geocode_result", "id": msg.get("id"), "results": []}
        try:
            out["results"] = await asyncio.get_running_loop().run_in_executor(
                self.s.search_pool, geocodemod.search, q, lat, lon, self.s.net,
                self.s.country_at(lat, lon) if lat is not None and lon is not None else None, bool(msg.get("full")))
        except Exception as e:          # offline, rate limited, refused, a bad answer: the map says which (GitHub #602)
            out["error"] = f"{type(e).__name__}: {e}"[:200]
            out["text"] = geocodemod.error_text(e)
            if out["text"] != getattr(self.s, "_geocode_said", None):     # once per change, not per keystroke
                self.s._geocode_said = out["text"]
                log.warning("address search: %s (%s)", out["text"], out["error"])
        else:
            self.s._geocode_said = None
        log.debug("geocode %r: %d results%s", q, len(out["results"]), (" (" + out["error"] + ")") if "error" in out else "")
        self.send(out)

    async def cache_map(self, msg):
        """The map's cache view {"id", "z", "x0", "y0", "x1", "y1", "mode"?} -> cache_map_result {"id", "z", "x0",
        "y0", "w", "h", "cells" (base64, a level per cell), "gen", "ready", "error"?} (covermap.CoverMap.query)."""
        out = {"type": "cache_map_result", "id": msg.get("id"), "z": 0, "x0": 0, "y0": 0, "w": 0, "h": 0, "cells": "",
               "gen": 0}
        cm = self.s.covermap
        if cm is None:
            out["error"] = "no cache view (no download cache)"
        else:
            try:
                out.update(await asyncio.get_running_loop().run_in_executor(
                    None, cm.query, msg.get("z"), msg.get("x0"), msg.get("y0"), msg.get("x1"),
                    msg.get("y1"), msg.get("mode") or "summary"))
            except ValueError as e:
                out["error"] = str(e)[:200]
        self.send(out)

    def _area_route(self, msg):
        """A saved route's points (area_plan / area_add with "route": "current": the last planned GPS route) as
        (lats, lons), or None (a box)."""
        if msg.get("route") != "current":
            return None
        pts = self.route_points or []
        return [p[0] for p in pts], [p[1] for p in pts]

    async def area_plan(self, msg):
        """{"id", "s", "w", "n", "e"} or {"id", "route": "current", "half"} -> area_plan_result {"id", "ok", km2, items,
        missing, downloadBytes, diskBytes, seconds, days, hosts, excluded, km? (a route's length), "error"?}
        (areas.Areas.plan / plan_route: no network)."""
        out = {"type": "area_plan_result", "id": msg.get("id"), "ok": False}
        if self.s.areas is None:
            out["error"] = "saved areas need the download cache"
        else:
            try:
                route = self._area_route(msg)
                mode = msg.get("mode") if isinstance(msg.get("mode"), str) else areasmod.DEFAULT_MODE
                if route is not None:
                    out.update(await asyncio.get_running_loop().run_in_executor(
                        None, self.s.areas.plan_route, route[0], route[1], msg.get("half"), mode))
                else:
                    out.update(await asyncio.get_running_loop().run_in_executor(
                        None, self.s.areas.plan, msg.get("s"), msg.get("w"), msg.get("n"), msg.get("e"), mode))
            except Exception as e:      # (the box stays without an estimate)
                out.update(ok=False, error=f"{type(e).__name__}: {e}"[:200])
        self.send(out)

    async def area_add(self, msg):
        """{"id", "s", "w", "n", "e", "name"?} or {"id", "route": "current", "half", "name"?} -> areas {..., "result":
        {"id", "ok", "area" or "error"}}."""
        if self.s.areas is None:
            self.send({"type": "areas", "areas": [], "result": {"id": msg.get("id"), "ok": False,
                                                                "error": "saved areas need the download cache"}})
            return
        name = msg.get("name")
        name = name if isinstance(name, str) else ""
        mode = msg.get("mode") if isinstance(msg.get("mode"), str) else areasmod.DEFAULT_MODE
        fast = msg.get("fast") is True      # (fast mode: areas.py; the mod pauses the game while it runs)
        route = self._area_route(msg)
        if route is not None:
            ok, info = await asyncio.get_running_loop().run_in_executor(
                None, self.s.areas.add_route, route[0], route[1], msg.get("half"), name, mode, fast)
        else:
            ok, info = await asyncio.get_running_loop().run_in_executor(
                None, self.s.areas.add, msg.get("s"), msg.get("w"), msg.get("n"), msg.get("e"), name, mode, fast)
        res = {"id": msg.get("id"), "ok": ok, ("area" if ok else "error"): info}
        self.send(self.s.areas_message(res))

    async def area_cmd(self, msg):
        """{"id", "area", "op": pause | resume | delete | fast | slow} -> areas {..., "result": {"id", "ok"}}."""
        if self.s.areas is None:
            return
        area, op = msg.get("area"), msg.get("op")
        ok = isinstance(area, str) and isinstance(op, str) and await asyncio.get_running_loop().run_in_executor(
            None, self.s.areas.command, area, op)
        self.send(self.s.areas_message({"id": msg.get("id"), "ok": bool(ok)}))
