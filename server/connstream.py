"""World server connection: the player's pose and the tile stream (split out of earthd.py's Connection): refreshes,
the sea and far tiles, the eye, prefetching, views and tile groups.
"""
import os
import asyncio  # noqa: E402
import functools  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import concurrent.futures  # noqa: E402
import numpy as np  # noqa: E402
import roads as roadsmod  # noqa: E402
import mapdata as mapdatamod  # noqa: E402
import dem as demmod  # noqa: E402
import mapcells  # noqa: E402
import pacer  # noqa: E402
import farfield  # noqa: E402
import seacarve as seacarvemod  # noqa: E402
import seatiles as seatilesmod  # noqa: E402
import buildings as buildingsmod  # noqa: E402
import geotiles as geotilesmod  # noqa: E402
import riverlines as riverlinesmod  # noqa: E402
import groundahead as groundaheadmod  # noqa: E402
import heightmap  # noqa: E402
import imagery as imagerymod  # noqa: E402
import matslots as matslotsmod  # noqa: E402
import earthbase
from earthbase import (
    BLOCK_DRAWN,
    BOOST_FULL_MAX_S,
    BOOST_GRACE_S,
    BOOST_MAX_S,
    BOOST_MOVE_MS,
    DELTA_MARGIN,
    EYE_ABOVE_PLAYER,
    EYE_LOD_MIN_M,
    EYE_STEP,
    EYE_STEP_REL,
    FAR_TRACE,
    FAST_KEEP_S,
    FLY_TEX_EYE_M,
    GLOBAL_AID,
    GROUND_RESEND_M,
    GROUND_RESEND_S,
    HM_BUSY_TILE_WORKERS,
    IMAGERY_PREFETCH,
    LANE_PREFETCH,
    MAP_BLD_FAR_EYE,
    MAP_BLD_FAR_MAX_M,
    MAP_BLD_SKY_EYE,
    MAP_BLD_SKY_MAX_M,
    MAP_BUILDINGS_EYE_M,
    MAP_DETAIL_EYE_M,
    MAP_HIGH_EYE_M,
    MAP_OUT_OF_VIEW_K,
    MAP_RAMP_START,
    MAP_RAMP_STEP,
    MAP_TEX_UPGRADE_S,
    MAP_ZOOM_RAMP_K,
    NAV_LEAD_MAX_M,
    NAV_LEAD_S,
    NAV_WAIT_S,
    PADS_REFRESH_MS,
    PADS_SETTLE_S,
    PREFETCHED_MAX,
    PREFETCH_AGAIN_S,
    PREFETCH_CHAIN,
    PREFETCH_PARALLEL,
    PROVISIONAL_KEEP_S,
    RADIUS_REPORT_REL,
    READY_SET_M,
    READY_STEPS,
    SEA_CHECK_S,
    SLOW_SELECT_S,
    TELEPORT_PAUSE_MAX_S,
    VIEW_AWAY_M,
    _num,
    log,
    parse_foci,
    predict_chain,
    predict_request,
    predict_route_chain,
)


class _StreamMixin:
    """Connection's pose and streaming (split out of earthd.py; Connection inherits it)."""

    def far_loading(self):
        """Is the far field round the active anchor still being built or sent (far_done not owed yet)?"""
        return bool(self.dirty or self.inflight)

    def on_pose(self, msg):
        self.slots_gone(msg.get("slotsGone"))
        held = msg.get("held")
        if isinstance(held, dict):      # an empty Lua table may encode as {}
            held = list(held.values())
        if isinstance(held, list):
            self.held = {int(a) for a in held if isinstance(a, (int, float)) and not isinstance(a, bool)}
        self.s.claim_sessions(self, msg.get("sessions"))   # earlier runs' folders still in use (or released)
        self.weather.on_pose(self, msg.get("lat"), msg.get("lon"))
        feed = getattr(self.s, "feed", None)
        if feed is not None:            # the companion feed (companionfeed.py)
            feed.publish(msg.get("lat"), msg.get("lon"), msg.get("speed"), msg.get("vx"), msg.get("vy"),
                         msg.get("anchorId"))
        if isinstance(msg.get("lat"), (int, float)) and isinstance(msg.get("lon"), (int, float)):
            self.pose_ll = (float(msg["lat"]), float(msg["lon"]))     # (the BeamMP share hub's prefetch)
        if self.s.surface is not None and isinstance(msg.get("lat"), (int, float)) and isinstance(msg.get("lon"), (int, float)):
            self.s.surface.prefetch(float(msg["lat"]), float(msg["lon"]))
        aid = msg.get("anchorId")
        # The latest committed anchor is the one the mod reports in its pose
        # (ids only grow, so never switch back to an older one).
        if aid in self.s.anchors and (self.active is None or aid > self.active):
            self.commit_anchor(aid)
        x, y = _num(msg, "x", 0.0), _num(msg, "y", 0.0)
        vx, vy = _num(msg, "vx", 0.0), _num(msg, "vy", 0.0)
        if aid == self.active:
            self.focus = (x, y)
            self.pz = _num(msg, "z", 0.0)
        elif aid in self.s.anchors and self.active in self.s.anchors:
            # A late pose of the previous anchor: move it into the active frame.
            src, dst = self.s.anchors[aid].frame, self.s.anchors[self.active].frame
            fx, fy, _ = dst.ecef_to_enu(*src.enu_to_ecef(x, y, 0.0))
            self.focus = (float(fx), float(fy))
        else:
            return
        self.vel = (vx, vy)
        now = time.monotonic()
        self.s.speed = math.hypot(vx, vy)
        if math.hypot(vx, vy) > earthbase.FAST_BASE_V:
            self.s.fast_at = now
        self.s.fast = now - getattr(self.s, "fast_at", -1e9) < FAST_KEEP_S
        self.hold_boost(msg.get("guardHold") is True)
        self.catch_up(math.hypot(vx, vy))
        if self.s.boost_on and not self.catchup:
            since = time.monotonic() - getattr(self, "boost_t0", 0.0)
            moving = math.hypot(vx, vy) > BOOST_MOVE_MS and since > BOOST_GRACE_S     # (the car settling: not moving)
            if moving and self.s.mapdata is not None:
                # (not while the roads round the car are still being built: a 2-core server's boost ended as the
                # player drove off on bare ground, its road tiles minutes late, GitHub #412)
                done, want = self.roads.progress()
                moving = done >= want
            if self.s.boost_full:
                # (full priority: until the far field and the roads round the car are done, moving or not)
                done, want = self.roads.progress() if self.s.mapdata is not None else (1, 1)
                # (its own far field done since the teleport: right after it nothing was loading yet, ended in 0.7 s)
                loaded = getattr(self, "far_done_at", 0.0) > getattr(self, "boost_t0", 0.0) and not self.far_loading()
                if (loaded and done >= want) or since > BOOST_FULL_MAX_S:
                    self.s.load_boost(False, "loaded" if since <= BOOST_FULL_MAX_S else "timeout")
            elif moving or since > BOOST_MAX_S:
                self.s.load_boost(False, "moving" if moving else "timeout")
        if isinstance(msg.get("cam"), dict):
            self.cam, self.cam_aid = msg["cam"], aid
        self.view_at = self.camera_away(msg.get("cam")) if aid == self.active else None
        self.foci = parse_foci(msg.get("foci")) if aid == self.active else ()
        if self.s.horizon:
            self.update_eye(self.s.anchors[aid].frame, *(self.view_at or (x, y)), msg)
        hint = getattr(self, "focus_hint", None)
        if hint is not None:
            if hint[0] == aid == self.active and time.monotonic() < hint[4]:
                # (the 3D map flying there: the far field built round where the view will end, at its detail)
                self.view_at, self.eye = (hint[1], hint[2]), hint[3]
            elif time.monotonic() >= hint[4]:
                self.focus_hint = None
        self.gravity.on_pose(self, self.s.anchors[aid].frame, x, y, _num(msg, "z", 0.0))
        self.ground_ahead(aid, x, y, vx, vy)
        if self.s.sea is not None:
            self.sea_tiles(aid, x, y)
        # the mod's altitude rule: high above the ground it does not move its block, so no block is
        # built ahead (prefetch, road plans) until it comes down
        self.following = msg.get("follow") is not False
        self.map_view = msg.get("view") is True     # (the 3D map's view frame: mod core st.viewFrame)
        self.map_open = msg.get("map") is True      # (the 3D map open, near the car or not: map3d.isActive)
        # the 3D map open: every core for the server (cpupin.full_split, the game on its first cores: it draws the map
        # and runs no physics there), so the ten far-tile processes are not squeezed onto the server's four CPUs
        # (2026-11-07: a fresh city's first far tile came 19 s after the flight began)
        # (asked again on every pose while it is not ours: full priority loading or an edge hold has the split, and
        # theirs ends while the map is still open)
        want = self.map_view or self.map_open
        if want and not getattr(self, "map_split", False) and time.monotonic() >= getattr(self, "map_split_try", 0.0):
            self.map_split_try = time.monotonic() + 2.0
            self.map_split = self.s.split_boost(True, "3D map")
        elif not want and getattr(self, "map_split", False):
            self.map_split = False
            self.s.split_boost(False, "3D map closed")
        self.predicted_block = None
        # The mod asks for its next terrain relative to the block it will have committed by
        # then: the pending one if it asked already (it commits that one first), else the
        # committed one. Predicting from the committed block while a request is pending built
        # terrains nobody asks for whenever the heading changed.
        base = self.pending_block or self.block
        if base is not None and self.following:
            px, py = self.focus
            rule = self.s.follow
            ahead = rule.mode(*self.vel)[2]         # M4 fast mode: the mod tests a point ahead
            qx, qy = px + self.vel[0] * ahead, py + self.vel[1] * ahead
            at = self.predicted_at = {}     # (centre -> the player where the mod asks for it: maybe_prefetch)
            if max(abs(qx - base.cx), abs(qy - base.cy)) >= rule.prefetch_start:
                where = []
                c = predict_request(px, py, *self.vel, base.cx, base.cy, rule, where)
                if c != (base.cx, base.cy) and self.s.layout.block_at(*c) != self.block:
                    self.predicted_block = self.s.layout.block_at(*c)
                    at[c] = where[0]
            self.predicted_chain = []
            if math.hypot(*self.vel) > earthbase.FAST_BASE_V:
                route_at = {}
                cs = predict_route_chain(self.route_xy(aid), px, py, *self.vel, base.cx, base.cy, rule,
                                         PREFETCH_CHAIN + 1, route_at)
                if cs:              # (on the GPS route: along it; its first is the next block, as predicted_block)
                    at.update(route_at)
                    if self.predicted_block is not None:
                        self.predicted_block = self.s.layout.block_at(*cs[0])
                    cs = cs[1:]
                else:
                    cs = predict_chain(px, py, *self.vel, base.cx, base.cy, rule, PREFETCH_CHAIN, at)
                self.predicted_chain = [self.s.layout.block_at(*c) for c in cs]
        if (self.map_view or self.map_open) and self.eye:
            # the 3D map zoomed far in (the user, 2026-11-07: "no matter where i am it should only take a couple seconds
            # to see detailed terrain"): coarse first there too, as at a new anchor (commit_anchor), not the finest of
            # ~360 tiles first and the screen filling in 15 s (Minneapolis, no origin move)
            ref = getattr(self, "zoom_ref", None)
            if ref is None or self.eye > ref:
                self.zoom_ref = self.eye
            elif self.eye < ref / MAP_ZOOM_RAMP_K and getattr(self, "lod_ramp", 1.0) >= 1.0:
                self.lod_ramp, self.zoom_ref = MAP_RAMP_START, self.eye
                log.info("3D map: zoomed in x%.0f, coarse first", ref / self.eye)
        self.update_far()
        if self.teleport_pending and time.monotonic() - self.teleport_t0 > TELEPORT_PAUSE_MAX_S:
            self.teleport_pending = False   # the mod never committed it (it stayed): the old place's work goes on
            self.wake.set()
        if self.teleport_pending:           # the old place's work waits: the new place has the CPU and the network
            self.refresh_status()
            return
        self.maybe_prefetch()
        self.send_ready_set()
        if aid == self.active and self.following:
            self.s.warm_roads(self.s.anchors[aid], *self.focus, *self.vel)
            self.s.warm_imagery(self.s.anchors[aid], *self.focus, *self.vel)
            self.s.download_ahead(self.s.anchors[aid], *self.focus, *self.vel)
            self.s.bake_sync(self.s.anchors[aid], *self.focus)
            self.base_refresh(aid)
            self.pads_refresh(aid)
            self.roads_refresh(aid)
        # (the 3D map's view frame high up, pose.view: no roads, trees, buildings, turbines or traffic round its target,
        # built as round a car: none of them shows from there, and their builds held the far tiles back and stalled the
        # event loop 2.1 s over Turin, 2026-11-07; nearer the ground they come, as what the view shows)
        detail = not (self.map_view and self.eye_clearance() > MAP_DETAIL_EYE_M)
        if self.s.mapdata is not None and aid == self.active and detail:
            if getattr(self.roads, "geo", False):
                anchor = self.s.anchors.get(aid)
                self.roads.on_pose(self, aid, *self.focus, asyncio.get_running_loop(), self.s.build_roads_geo,
                                   vel=self.vel, ground_busy=self.terrain_wait > 0, view=self.view_at,
                                   foci=self.foci, frame=anchor.frame if anchor is not None else None)
                # (the sign tiles stay the anchor's: each once the road geotiles under it are in)
                ready = self.roads.ready_tiles(anchor.frame, *self.focus, roadsmod.SEND_M) if anchor is not None \
                    else set()
            else:
                self.roads.on_pose(self, aid, *self.focus, asyncio.get_running_loop(), self.s.build_roads_tile,
                                   vel=self.vel, ground_busy=self.terrain_wait > 0, view=self.view_at,
                                   foci=self.foci)
                ready = self.roads.sent
            self.s.osm_fuel_want(self, aid, getattr(self.roads, "car", ()))   # (OpenStreetMap's fuel stations)
            source = getattr(self.s.mapdata, "highway_signs", None)
            if source is not None and self.signs_on:      # (after the roads: signs only for road tiles sent)
                self.signs.on_pose(self, aid, *self.focus, asyncio.get_running_loop(), self.s.build_signs_tile,
                                   ready, version=source.tags.known())
        if self.turbine_geo and aid == self.active and detail:
            anchor = self.s.anchors.get(aid)
            if anchor is not None:
                self.turbines.on_pose(self, aid, *self.focus, asyncio.get_running_loop(), self.s.build_turbines_geo,
                                      view=self.view_at, frame=anchor.frame)
        if self.s.veg is not None and aid == self.active and detail:
            anchor = self.s.anchors.get(aid)
            if anchor is not None:                            # a teleport's cells built at once, ahead
                la, lo, _ = anchor.frame.enu_to_geodetic(*self.focus, 0.0)
                self.s.veg.warm_around(float(la), float(lo))
            self.veg.on_pose(self, aid, *self.focus, asyncio.get_running_loop(),
                             self.build_veg_geo if self.veg.geo else self.build_veg_tile,
                             view=self.view_at, foci=[f for f in self.foci if f[3]],
                             frame=anchor.frame if anchor is not None else None)
        bld_on = not (self.map_view and self.eye_clearance() > MAP_BUILDINGS_EYE_M)
        if self.s.buildings and aid == self.active and bld_on:
            self.building_reach()
            geo = self.far_geo
            anchor = self.s.anchors.get(aid)
            frame = anchor.frame if anchor is not None else None
            self.bldfar.on_pose(self, aid, *self.focus, asyncio.get_running_loop(),
                                self.build_far_geo("bldfar") if geo else self.s.build_far_bld_tile,
                                view=self.view_at, frame=frame)
            if self.bldsky is not None and self.bldfar.progress()[0] >= self.bldfar.progress()[1]:
                # (only once the far buildings round the car are all there: the skyline is the least urgent)
                self.bldsky.on_pose(self, aid, *self.focus, asyncio.get_running_loop(),
                                    self.build_far_geo("bldsky") if geo else self.s.build_sky_tile,
                                    view=self.view_at, frame=frame)
            if self.brfar is not None and self.bldfar.progress()[0] >= self.bldfar.progress()[1]:
                self.brfar.on_pose(self, aid, *self.focus, asyncio.get_running_loop(),
                                   self.build_far_geo("brfar") if geo else self.s.build_far_bridge_tile,
                                   view=self.view_at, frame=frame)
        if self.s.traffic and aid == self.active and not self.map_view and self.nav_may_start(aid):
            # (centred ahead of a moving car: the minimap's roads, the AI's and the GPS's graph, ran out in front at
            # speed, a graph built round where the car was when it started; the user, 2026-11-04)
            vx, vy = getattr(self, "vel", (0.0, 0.0))
            lead = min(NAV_LEAD_S * math.hypot(vx, vy), NAV_LEAD_MAX_M)
            sp = max(math.hypot(vx, vy), 1e-6)
            cx, cy = self.focus[0] + vx / sp * lead, self.focus[1] + vy / sp * lead
            self.nav.on_pose(self, aid, cx, cy, asyncio.get_running_loop(), self.s.build_navgraph)
        if self.traffic_on and aid == self.active and not self.map_view and self.nav_may_start(aid):
            self.distant_traffic.on_pose(self, aid, *self.focus, asyncio.get_running_loop(), self.s.build_traffic_geo,
                                         view=self.view_at, frame=self.s.anchors[aid].frame)
            done, want = self.distant_traffic.progress()
            if done >= want / 2:
                self.distant_traffic_far.on_pose(self, aid, *self.focus, asyncio.get_running_loop(),
                                                functools.partial(self.s.build_traffic_geo, far=True),
                                                view=self.view_at, frame=self.s.anchors[aid].frame)
        if aid == self.active:
            self.send_progress(aid)
        self.refresh_status()
        self.credits.poke(self)             # the player moved: the near / far credits may change

    def nav_may_start(self, aid):
        """May the AI graph of anchor aid be built now? Its first one waits until half the road tiles round the car are
        made (or NAV_WAIT_S): it takes a road process for its whole build, and on a teleport the road tiles under the
        car came after it (a Windows PC's two road processes, 2026-10-25)."""
        if self.nav.aid == aid:
            return True                     # (a rebuild as the car moves on: the tiles round it are there)
        t0 = self.nav_wait.get(aid)
        if t0 is None:
            t0 = self.nav_wait[aid] = time.monotonic()
            for k in [k for k in self.nav_wait if k != aid]:
                del self.nav_wait[k]
        done, wanted = self.roads.progress() if self.roads is not None else (0, 0)
        return (wanted > 0 and done * 2 >= wanted) or time.monotonic() - t0 >= NAV_WAIT_S

    def route_xy(self, aid):
        """The last planned GPS route's line in anchor aid's frame ((k, 2), kept per route and anchor), or None."""
        pts = self.route_points
        anchor = self.s.anchors.get(aid)
        if not pts or anchor is None or len(pts) < 2:
            return None
        got = getattr(self, "_route_xy", None)
        if got is not None and got[0] == aid and got[1] is pts:
            return got[2]
        ll = np.asarray(pts, np.float64)
        e, n_, _ = anchor.frame.geodetic_to_enu(ll[:, 0], ll[:, 1], 0.0)
        xy = np.stack([np.asarray(e, np.float64), np.asarray(n_, np.float64)], 1)
        self._route_xy = (aid, pts, xy)
        return xy

    def camera_away(self, cam):
        """The camera's (x, y) from a pose's cam when it is more than VIEW_AWAY_M from the car (a free camera, a far
        chase camera), else None. What is drawn follows it: the far field, and the road, tree and far building tiles
        round it as well as round the car; what collides stays round the car (the terrain block, the road and
        building colliders in the tiles round the car, the nav graph). The user, 2026-10-20: "All collision ...
        should be following the car while visual stuff should follow the camera"."""
        if not isinstance(cam, dict):
            return None
        cx, cy = cam.get("x"), cam.get("y")
        if cam.get("tx") is not None and cam.get("ty") is not None:
            cx, cy = cam.get("tx"), cam.get("ty")       # (the 3D map's target on the ground: mod map3d.lua)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (cx, cy)):
            return None
        if math.hypot(cx - self.focus[0], cy - self.focus[1]) <= VIEW_AWAY_M:
            return None
        return float(cx), float(cy)

    @property
    def view_focus(self):
        """Where the far field is centred: the camera when it is away from the car (camera_away), else the car."""
        return self.view_at or self.focus

    def base_refresh(self, aid):
        """Detail mode: the block in use and the one handed out last, if their base photo was rendered before
        all its NAIP tiles were fetched (Server.base_stale) and they are all here now: rendered again (a new
        file name) and sent as block_base. One at a time."""
        s = self.s
        if self.base_busy or not s.base_stale or s.imagery is None or s.near_visual != "detail":
            return
        anchor = s.anchors.get(aid)
        cands = [k for k in list(s.hm_committed)[-1:] + list(s.hm_sent)[-1:] if k in s.base_stale and k[0] == aid]
        size = s.layout.block * heightmap.SQUARE
        # (a skipped photo (a fast player's prefetched block): at once, its imagery is fetched as it renders)
        key = next((k for k in cands if k in s.hm and (k in s.base_skipped or s.base_ready(anchor, k[1], k[2]))), None)
        if key is None:
            return
        self.base_busy = True

        async def run():
            try:
                # its BC7 encode: in the photo process when there is one (in the prefetch lane's, a block ahead waited
                # behind it and came late, 2026-11-05)
                pool = getattr(s, "photo_pool", None) or (s.block_pools[LANE_PREFETCH][0] if s.block_pools else None)
                rev = next(s.base_rev)
                name = None
                if pool is not None:
                    # (in the block process: from the finished tiles with the analyses on; they were all rendered
                    # here, 121 of a warm 8 min drive at 150 mph, beside the event loop, 2026-11-04)
                    name = await self.loop.run_in_executor(s.hm_pool, s._base_elsewhere, anchor, key[1], key[2],
                                                           pool, [], False, rev)
                if name is None:
                    name = await self.loop.run_in_executor(s.hm_pool, s.write_base, anchor, key[1], key[2], pool,
                                                           rev)
                info = s.hm.get(key)
                if name is None or info is None:
                    return
                old, info["base"] = info.get("base"), name
                if old and old != name and old != s.base_name(key):
                    # (kept until the block goes, prune_hm, with its pack: the mod may still show it a while, its
                    # refresh deferred at speed (core BASE_LATER_MS), and a material rebuild then needs the pack;
                    # deleted here, the game packed the photo again on its main thread, ~0.75 s frames, 2026-11-03)
                    info.setdefault("old_bases", []).append(old)
                log.info("base photo of %s rendered again with its NAIP (%s)", key, name)
                self.send({"type": "block_base", "anchorId": key[0], "cx": key[1], "cy": key[2],
                           "base": f"{s.vfs_dir}/{name}", "baseSize": size})
                s.base_skipped.discard(key)
            except Exception:
                log.exception("base photo refresh %s", key)
                s.base_stale.discard(key)
                s.base_skipped.discard(key)
            finally:
                self.base_busy = False
        self.spawn(run())

    def roads_refresh(self, aid):
        """The block in use built again (terrain_refresh) when it was built before every road cell under it was on disk
        (Server.roads_rebuild: surface.Surface.roads leaves a missing cell's roads out and fetches it; the block kept
        grass and gravel on those roads, and their grip, until it was built again: a fast first visit's mountain roads,
        Discord 2026-10-26), once those cells are here. One at a time, at any speed (the car is on it)."""
        s = self.s
        if self.roads_busy or not s.roads_rebuild or s.mapdata is None:
            return
        key = next((k for k in list(s.hm_committed)[-1:] if k in s.roads_rebuild and k[0] == aid), None)
        anchor = s.anchors.get(aid)
        if key is None or anchor is None:
            return
        half = s.layout.block * heightmap.SQUARE / 2
        la, lo, _ = anchor.frame.enu_to_geodetic(np.array([key[1] - half, key[1] + half]),
                                                 np.array([key[2] - half, key[2] + half]), 0.0)
        cells = mapcells.cells_for_bbox(float(np.min(la)), float(np.min(lo)), float(np.max(la)), float(np.max(lo)))
        if not all(s.mapdata.has(k, "roads") and (s.surface is None or s.mapdata.has(k, "landcover")) for k in cells):
            return                          # (not here yet: asked again at the next pose)
        self.roads_busy = True
        s.roads_rebuild.discard(key)

        async def run():
            old = s.hm.pop(key, None)
            try:
                # (the prefetch lane's process: a correction of the block in use must not hold up the next one the
                # mod asks for, a first visit's rebuilds queued the follow requests, 2026-11-01)
                info = await s.terrain(anchor, key[1], key[2], lane=LANE_PREFETCH)
            except Exception:
                log.exception("block %s again with its roads (the old one stays)", key)
                if old is not None:
                    s.hm[key] = old
                return
            finally:
                self.roads_busy = False
            s.hm_sent.append(key)
            log.info("block %s built again with its roads (their map cells came after it was built)", key)
            self.send({"type": "terrain_refresh", "anchorId": key[0], **self.terrain_msg(key, info)})
            old_ter = (old or {}).get("ter")
            if old_ter and old_ter != info.get("ter"):
                await asyncio.sleep(PROVISIONAL_KEEP_S)
                s._remove(os.path.join(s.out_dir, old_ter))
        self.spawn(run())

    def pads_refresh(self, aid, now=None):
        """The block in use built again (terrain_refresh, as refine_first) when gas station pads came in after it was
        built (Server.pads_rebuild: its surface layers then have the forecourt's asphalt, no grass on it; the user,
        2026-10-24: "looks like we painted it but it is still dirt"). Only with the car slower than PADS_REFRESH_MS
        (driving on, the next blocks are built with them) and PADS_SETTLE_S after the last new pad (a tile burst
        brings several); one at a time."""
        s = self.s
        now = time.monotonic() if now is None else now
        if (self.pads_busy or not s.pads_rebuild or math.hypot(*self.vel) > PADS_REFRESH_MS
                or now - s.pads_changed_at < PADS_SETTLE_S):
            return
        key = next((k for k in list(s.hm_committed)[-1:] if k in s.pads_rebuild and k[0] == aid), None)
        anchor = s.anchors.get(aid)
        if key is None or anchor is None:
            return
        self.pads_busy = True
        s.pads_rebuild.discard(key)

        async def run():
            old = s.hm.pop(key, None)
            try:
                # (the prefetch lane's process: a correction of the block in use must not hold up the next one the
                # mod asks for, a first visit's rebuilds queued the follow requests, 2026-11-01)
                info = await s.terrain(anchor, key[1], key[2], lane=LANE_PREFETCH)
            except Exception:
                log.exception("block %s again with its gas station pads (the old one stays)", key)
                if old is not None:
                    s.hm[key] = old
                return
            finally:
                self.pads_busy = False
            s.base_stale.discard(key)           # (its photo built with them too)
            s.hm_sent.append(key)
            log.info("block %s built again with its gas station pads", key)
            self.send({"type": "terrain_refresh", "anchorId": key[0], **self.terrain_msg(key, info)})
            old_ter = (old or {}).get("ter")
            if old_ter and old_ter != info.get("ter"):
                await asyncio.sleep(PROVISIONAL_KEEP_S)
                s._remove(os.path.join(s.out_dir, old_ter))
        self.spawn(run())

    def ground_ahead(self, aid, x, y, vx, vy):
        """Send the mod the highest ground near its straight path (groundahead.py; its altitude rule), one
        job at a time; a pose that arrives meanwhile is skipped (the next one comes in 0.5 s)."""
        if self.ground_busy:
            return
        self.ground_busy = True
        frame = self.s.anchors[aid].frame

        async def run():
            try:
                z = await self.loop.run_in_executor(self.s.ground_pool, groundaheadmod.highest, self.s.dem, frame,
                                                    x, y, vx, vy)
                now, last = time.monotonic(), self.ground_sent
                if math.isfinite(z) and (last is None or last[0] != aid or abs(z - last[1]) > GROUND_RESEND_M
                                         or now - last[2] >= GROUND_RESEND_S):
                    self.ground_sent = (aid, z, now)
                    self.send({"type": "ground_ahead", "anchorId": aid, "z": round(z, 1)})
            except Exception as e:      # offline, a DEM failure: the mod keeps following
                log.debug("ground ahead: %s: %s", type(e).__name__, e)
            finally:
                self.ground_busy = False
        self.spawn(run())

    def sea_tiles(self, aid, x, y):
        """Send the mod the water tiles around the player (seatiles.plan and plan_lakes; its sea.lua),
        planned again in another 512 m square or after SEA_CHECK_S, sent when the set changed."""
        last, now = self.sea_checked, time.monotonic()
        sq = seatilesmod.square_of(x, y)        # the plan is the same anywhere in this square
        # again after SEA_CHECK_S only when the last plan lacked map data (built meanwhile): a complete one stays
        # (re-planned every 15 s, the rivers' probes were a Steam Deck server's busiest thread, 64 s of a drive)
        if self.sea_busy or (last is not None and last[0] == aid and last[1] == sq
                             and (last[3] or now - last[2] < SEA_CHECK_S)):
            return
        self.sea_busy = True
        self.sea_checked = (aid, sq, now, False)
        missing = [False]               # a cell's map data not at hand for this plan
        frame = self.s.anchors[aid].frame
        sea, dem = self.s.sea, self.s.dem

        async def run():
            try:
                def cell_water(key):    # a cell's water lines if its map data is at hand (else built meanwhile)
                    md = sea.md
                    if not os.path.exists(md.path(mapdatamod._key(key), "water")):
                        missing[0] = True
                        sea.warm([(key.x, key.y)])
                        return None
                    return md.cell(key, layers=("water",)).layers.get("water")

                def tiled(la, lo):      # already water as tiles (the sea, lakes; not rivers)
                    d, lv, _ = sea.water(la, lo, wait=False)
                    return np.where(np.isfinite(lv) & ~sea.outline_at(la, lo), d, 0.0)

                def road_clear():   # the roads of the cells at hand around: the rivers keep off them
                    md = sea.md
                    r = riverlinesmod.RADIUS_M + riverlinesmod.PIECE_M
                    la0, lo0, _ = frame.enu_to_geodetic(x - r, y - r, 0.0)
                    la1, lo1, _ = frame.enu_to_geodetic(x + r, y + r, 0.0)
                    try:
                        every = mapcells.cells_for_bbox(float(la0), float(lo0), float(la1), float(lo1), max_cells=256)
                        keys = tuple(key for key in every if os.path.exists(md.path(mapdatamod._key(key), "roads")))
                        if len(keys) < len(every):
                            missing[0] = True
                        memo = getattr(self, "_river_clear", None)   # (anchor, cells) -> clearance: built once
                        if memo is not None and memo[0] == (aid, keys):
                            return memo[1]
                        feats = [md.cell(key, layers=("roads",)).layers.get("roads") for key in keys]
                        rc = riverlinesmod.RoadClearance.from_features(frame, feats)
                        self._river_clear = ((aid, keys), rc)
                        return rc
                    except Exception as e:      # no clearance: the widths as measured
                        log.debug("river road clearance: %s", e)
                        return None

                grader = self.s.grader_for(self.s.anchors[aid])

                def lake_ground(la, lo):
                    # the ground as graded under the roads (the terrain block's): a reservoir's tile kept over the
                    # raw DEM flooded Central Park's sunken transverse, graded 1.6 m under its water (2026-10-17)
                    z = dem.sample(la, lo, seatilesmod.LAKE_DEM_ZOOM)
                    return grader.apply(la, lo, z) if grader is not None else z

                def sea_ground(la, lo, size):
                    # the ground under a sea tile at about its samples' spacing (seatiles.plan: land under the sea,
                    # the polders, splits it)
                    n = min(max(size / 32.0, seatilesmod.SEA_SPLIT_SAMPLES[0]), seatilesmod.SEA_SPLIT_SAMPLES[1])
                    z = demmod.zoom_for(float(np.mean(la)), size / n)
                    z = min(max(z, getattr(demmod, "MIN_ZOOM", 0)), demmod.MAX_ZOOM)
                    return dem.sample(la, lo, z)

                def plan():
                    pacer.hold()        # (a teleport's first block first)
                    asked = sea.warm_asked
                    # with the cells at hand (the others are built in the background and used by the
                    # next plan, SEA_CHECK_S later): water appears at once at a new place
                    tiles = (seatilesmod.plan(frame, x, y, lambda la, lo, size: sea.sea_at(dem, la, lo, size, False),
                                              ground_at=sea_ground)
                             + seatilesmod.plan_lakes(frame, x, y, lambda la, lo: sea.lake_at(la, lo, False),
                                                      lake_ground))
                    def carved(la, lo):     # the river beds' level (seacarve), NaN elsewhere
                        lv = sea.water(la, lo, wait=False)[1]
                        return np.where(sea.outline_at(la, lo), lv, np.nan)
                    rivers = riverlinesmod.plan(frame, x, y, cell_water, tiled,
                                                lambda la, lo: dem.sample(la, lo, riverlinesmod.DEM_ZOOM),
                                                sea.outline_at, level_at=carved, road_clear=road_clear())
                    shores = seacarvemod.shore_plan(frame, x, y, sea)     # (the mod's shore sound)
                    return tiles, rivers, shores, not missing[0] and sea.warm_asked == asked
                tiles, rivers, shores, complete = await self.loop.run_in_executor(self.s.sea_pool, plan)
                if complete and self.sea_checked is not None and self.sea_checked[:2] == (aid, sq):
                    self.sea_checked = (aid, sq, self.sea_checked[2], True)
                msg = ([t.to_msg() for t in tiles], [r.to_msg() for r in rivers], shores)
                if self.sea_sent is None or self.sea_sent[0] != aid or self.sea_sent[1] != msg:
                    self.sea_sent = (aid, msg)
                    self.send({"type": "sea", "anchorId": aid, "tiles": msg[0], "rivers": msg[1], "shores": msg[2]})
            except Exception as e:      # offline, no map data: no water tiles
                log.debug("sea tiles: %s: %s", type(e).__name__, e)
            finally:
                self.sea_busy = False
        self.spawn(run())

    def update_eye(self, frame, x, y, msg):
        """M1h: the eye's height above sea level from a pose in `frame` (the pose's anchor): the camera's
        z, else EYE_ABOVE_PLAYER over the player's. The frame's z is the height above the anchor's
        tangent plane, whose origin is at sea level, so the exact geodetic height adds the d²/2R the
        plane lies above the ellipsoid at the player (17.7 m at 15 km). Quantised (EYE_STEP)."""
        cam = msg.get("cam")
        z = cam.get("z") if isinstance(cam, dict) else None
        if isinstance(z, (int, float)) and not isinstance(z, bool) and math.isfinite(z):
            z = float(z)
        else:
            z = _num(msg, "z", 0.0) + EYE_ABOVE_PLAYER
        h = float(frame.enu_to_geodetic(x, y, z)[2])
        if self.eye is None or abs(h - self.eye) > max(EYE_STEP, EYE_STEP_REL * abs(self.eye)):
            self.eye = h

    def horizon_select(self):
        """M1h: (radius, extra select() arguments) for the active anchor: the disc out to the
        line-of-sight horizon with the per-tile cull, L10 roots; until the anchor's zmax grid is
        built, the HORIZON_MIN disc without a cull. Reports a changed radius in a status."""
        anchor = self.s.anchors[self.active]
        eye = self.eye if self.eye is not None else 0.0
        grid = self.s.horizon_grid(anchor, eye)
        cull = None
        focus = self.view_focus
        if grid is None and self.global_far and self.prev_horizon is not None:
            # (a rebase whose grid is not built yet: the previous anchor's, the focus in its plane. Its Mercator
            # table culls the global tiles the same for any anchor; without it they went and came back once the new
            # grid was built. The anchor's own tiles lie within HORIZON_MIN: never culled)
            pframe, grid = self.prev_horizon
            fx, fy, _ = pframe.ecef_to_enu(*self.s.anchors[self.active].frame.enu_to_ecef(*focus, 0.0))
            focus = (float(fx), float(fy))
        if grid is None:
            radius = farfield.HORIZON_MIN
        else:
            radius, zr = grid.radius(*focus, eye)
            cull = farfield.Horizon(eye, grid)
            self._lod_grid = grid
            if zr != self.zmax_region or abs(radius - self.radius) > RADIUS_REPORT_REL * self.radius:
                log.info("horizon: eye %.0f m, highest visible terrain %.0f m, radius %.0f km",
                         eye, zr, radius / 1000)
            self.zmax_region = zr
        self.radius = radius
        if self.radius_sent is None or abs(radius - self.radius_sent) > RADIUS_REPORT_REL * self.radius_sent:
            self.status(self.last_status or f"Horizon {radius / 1000:.0f} km")
        return radius, {"max_level": self.s.max_level, "disc": True, "horizon": cull}

    def send_ready_set(self):
        """'terrain_ready_set' {anchorId, centres}: the blocks of the active anchor built already, within READY_SET_M of
        the player (the mod asks for one of them instead of its own target when it is close: follow.preferReady), when
        that set changed."""
        aid = self.active
        if aid is None or not self.s.fast or not getattr(self, "ready_set", False):   # (at speed: there the guesses
            return                                                                  # miss; a mod that takes it)
        px, py = self.focus
        cs = sorted((cx, cy) for (a, cx, cy) in list(self.s.hm) if a == aid
                    and max(abs(cx - px), abs(cy - py)) <= READY_SET_M)
        if cs != getattr(self, "ready_sent", None):
            self.ready_sent = cs
            self.send({"type": "terrain_ready_set", "anchorId": aid, "centres": [[float(cx), float(cy)] for cx, cy in cs]})

    def maybe_prefetch(self):
        """Speculatively build the terrain the mod is about to ask for: the predicted block, and a fast player's chain
        after it (PREFETCH_CHAIN), in order, PREFETCH_PARALLEL at once."""
        anchor = self.s.anchors.get(self.active)
        if anchor is None:
            return
        want = [b for b in [self.predicted_block] + list(getattr(self, "predicted_chain", ())) if b is not None]
        busy = getattr(self, "prefetch_busy", None)
        if busy is None:
            busy = self.prefetch_busy = set()
        at = getattr(self, "predicted_at", {})
        for b in want:
            if len(busy) >= PREFETCH_PARALLEL:
                return
            key = (anchor.id, b.cx, b.cy)
            if key in busy or self.s.terrain_cached(key):
                continue
            if self.ready_will_do(anchor, b, at.get((b.cx, b.cy)), busy):
                continue
            busy.add(key)
            self.prefetching = True

            async def run(cx=b.cx, cy=b.cy, key=key):
                try:
                    await self.s.terrain(anchor, cx, cy, lane=LANE_PREFETCH)
                    log.debug("prefetched terrain (%d, %.0f, %.0f)", anchor.id, cx, cy)
                except Exception:
                    log.exception("terrain prefetch failed")
                finally:
                    busy.discard(key)
                    self.prefetching = bool(busy)
            self.spawn(run())

    def ready_will_do(self, anchor, b, where, busy):
        """Would the mod take a block built (or being built) already instead of predicted block b (follow.lua
        preferReady: within READY_STEPS grid steps of its target, the player within reach of it where it asks)? Then b
        is not built: each pose's prediction a grid step off the last built another block, 55 built for 24 asked for
        at 150 mph down Midtown (nyc_speed.py 2026-11-02)."""
        if where is None:
            return False
        rule = self.s.follow
        _lead, reach, _ahead = rule.mode(*self.vel)
        lim = READY_STEPS * rule.grid
        cur = (self.block.cx, self.block.cy) if self.block is not None else None
        keys = [k for k in list(self.s.hm) + list(self.s.hm_inflight) + list(busy) if k[0] == anchor.id]
        for _a, kx, ky in keys:
            if (kx, ky) == (b.cx, b.cy) or (kx, ky) == cur:
                continue
            if max(abs(kx - b.cx), abs(ky - b.cy)) <= lim and max(abs(where[0] - kx), abs(where[1] - ky)) <= reach:
                return True
        return False

    # --- far field ----------------------------------------------------------
    def forced(self):
        """Boxes refined to L0 (terrain mode: the mod covers L0 under the block; detail mode: to L-1).
        None in mesh mode: the block is invisible, so tiles never depend on it (V1_3_SPEC §B).
        Detail mode: each box grown by DETAIL_RING_M, so the block's neighbours are L-1 tiles too (2 m
        cells, the block's own lattice: their edge meets the block's; L1 neighbours, 8 m cells, left a
        step and their dark skirt along the block's edge, in-game 2026-10-13)."""
        if self.s.near_visual not in BLOCK_DRAWN:
            return []
        g = earthbase.DETAIL_RING_M if self.s.near_visual == "detail" else 0.0
        return [(x0 - g, y0 - g, x1 + g, y1 + g) for x0, y0, x1, y1 in
                (b.bbox for b in (self.block, self.pending_block, self.predicted_block) if b is not None)]

    def _retire(self, cks):
        now = time.monotonic()
        for ck in cks:
            self.retired[ck] = now

    def update_far(self):
        """Recompute the desired set, the swap groups and the build queue; wake the streamer."""
        if self.active is None:
            return
        now = time.monotonic()
        self.retired = {ck: t for ck, t in self.retired.items() if now - t < self.s.retired_keep}
        self.node_changed = {n: t for n, t in self.node_changed.items() if now - t < farfield.HOLD_S}
        radius, extra = (self.horizon_select() if self.s.horizon else (self.s.radius, {}))
        if self.view_cap is not None:       # (a fixed view distance: nothing past it is drawn; a disc, as the horizon)
            radius, extra = min(radius, self.view_cap), dict(extra, disc=True)
        anchor = self.s.anchors[self.active]
        t_sel = time.perf_counter()
        if self.global_on(anchor, radius):
            # far levels from 3 up as global tiles (geotiles phase 3b): the same for every anchor, kept at a rebase
            high = self.eye is not None and self.eye > MAP_HIGH_EYE_M
            eye_h = self.eye_clearance()
            if high:
                self.s.start_world_warm(anchor)
            new, refined = farfield.select_global(anchor.frame, *self.view_focus, anchor.zf, self.forced(),
                                                  self.refined, self.kfun(), self.s.hysteresis, radius,
                                                  hold=set(self.node_changed), min_level=self.s.min_level,
                                                  radii=self.s.near_radii, layout=self.s.layout,
                                                  forced_level=self.s.forced_level, view=self.view(),
                                                  anchor_nodes=not high, eye_h=eye_h, cone=self.map_cone(aid=anchor.id),
                                                  **extra)
        else:
            new, refined = farfield.select(*self.view_focus, self.forced(), self.refined, self.kfun(),
                                           self.s.hysteresis, radius, hold=set(self.node_changed),
                                           min_level=self.s.min_level, radii=self.s.near_radii, layout=self.s.layout,
                                           forced_level=self.s.forced_level, view=self.view(), **extra)
        t_sel = time.perf_counter() - t_sel
        if t_sel > SLOW_SELECT_S:
            log.warning("far selection took %.2f s (radius %.0f km, eye %.0f m, %d tiles, %d refined nodes; global "
                        "part, anchor part s, regions, global tiles: %s)", t_sel, radius / 1000, self.eye or 0.0,
                        len(new), len(refined), farfield.LAST_GLOBAL_SELECT)
        if self.desired:                # the first selection of an anchor is not a change
            for n in refined ^ self.refined:
                self.node_changed[n] = now
        if FAR_TRACE and time.monotonic() - getattr(self, "_sel_logged", 0.0) > 0.5:
            self._sel_logged = time.monotonic()
            lv = {}
            for t in new.values():
                lv[t.level] = lv.get(t.level, 0) + 1
            log.info("fartrace select eye %.0f radius %.0f km focus %.0f,%.0f held %d levels %s", self.eye or -1,
                     radius / 1000, *self.view_focus, len(self.node_changed), dict(sorted(lv.items())))
        self.refined = refined
        changed = new.keys() != self.desired.keys()
        self.desired = new
        self._regroup()
        if changed and not self.dirty:
            self.dirty = True
            self.batch = [time.perf_counter(), 0, 0]
        self.wake.set()

    def building_reach(self):
        """The far buildings' and the skyline's send distances: the graphics setting's, farther with the 3D map open as
        the eye rises (the user, 2026-11-07: "make buildings start rendering further out"), at most MAP_BLD_*_MAX_M."""
        bld = getattr(self, "building_scale", None) or 1.0
        far = buildingsmod.FAR_SEND_M * bld
        sky = buildingsmod.SKY_SEND_M * bld
        if self.map_view or self.map_open:
            e = self.eye_clearance()
            far = max(far, min(e * MAP_BLD_FAR_EYE, MAP_BLD_FAR_MAX_M))
            sky = max(sky, min(e * MAP_BLD_SKY_EYE, MAP_BLD_SKY_MAX_M))
        if abs(self.bldfar.send_m - far) > 50.0:
            self.bldfar.send_m = far
            self.bldfar.drop_m = far + (buildingsmod.FAR_DROP_M - buildingsmod.FAR_SEND_M)
        if self.bldsky is not None and abs(self.bldsky.send_m - sky) > 50.0:
            self.bldsky.send_m = sky
            self.bldsky.drop_m = sky + (buildingsmod.SKY_DROP_M - buildingsmod.SKY_SEND_M)
        if self.bldsky is not None:
            self.bldsky.inner = (self.bldfar.send_m, buildingsmod.SKY_UNIT)

    def map_cone(self, aid):
        """The 3D map's view cone for the selection (farfield.out_of_view: off-screen nodes MAP_OUT_OF_VIEW_K times as
        far), or None (driving: every direction counts; no 3D camera yet)."""
        if not (self.map_view or self.map_open) or not isinstance(self.cam, dict) or \
                getattr(self, "cam_aid", None) != aid:
            return None
        c = farfield.view_cone(self.cam)
        return list(c) + [MAP_OUT_OF_VIEW_K] if c is not None and len(c) == 8 else None

    def ck(self, key):
        """The server's key of far tile `key` under the active anchor (Server.far_ck: a global tile's names none)."""
        return (GLOBAL_AID, key) if farfield.is_global_key(key) else (self.active, key)

    def eye_clearance(self):
        """The camera's height over the ground under the view (the far tiles' detail goes by the 3D distance:
        farfield.select_global eye_h): 0 below EYE_LOD_MIN_M (driving, the car's camera: as before), else the eye over
        the highest terrain of the horizon grid's cell there (an under-estimate on a mountain: finer, never coarser)."""
        eye = self.eye
        if eye is None or eye < EYE_LOD_MIN_M:
            return 0.0
        zg = 0.0
        grid = getattr(self, "_lod_grid", None)
        if grid is not None:
            s = farfield.level_size(farfield.HORIZON_GRID_LEVEL)
            fx, fy = self.view_focus
            z = grid.zmax(farfield.HORIZON_GRID_LEVEL, int(math.floor(fx / s)), int(math.floor(fy / s)))
            if math.isfinite(z):
                zg = max(0.0, z)
        return max(0.0, float(eye) - zg)

    def global_on(self, anchor, radius):
        """Global far tiles for this selection: the mod places them (hello.globalFar) and the radius stays within Web
        Mercator (farfield.GLOBAL_MAX_LAT; beyond it the anchor's quadtree as before)."""
        return bool(self.global_far) and abs(anchor.lat) + (radius + 20000.0) / 110574.0 < farfield.GLOBAL_MAX_LAT

    def poly_of(self, key):
        """A special tile's footprint polygon (flat) in the active anchor's world (swap groups): a desired tile's own,
        a shown global tile's quad, a piece's polygon as sent."""
        t = self.desired.get(key)
        if t is not None:
            return t.poly
        if farfield.is_global_key(key):
            ck = (self.active, key)
            q = self.gpolys.get(ck)
            if q is None:
                anchor = self.s.anchors.get(self.active)
                g = farfield.parse_global(key)
                if anchor is None or g is None:
                    return None
                q = self.gpolys[ck] = tuple(float(v) for v in
                                            geotilesmod.quad(anchor.frame, g[1], g[2], g[0]).ravel())
            return q
        return self.ppolys.get(key)

    def view(self):
        """The camera for screen-space LOD (farfield.View) from pose.cam (x, y, z, fovDeg, h: the screen's height px),
        or None: no camera yet, an older mod without h, a camera of another anchor's frame, or --lod-px 0. The
        tile errors are the built tiles' (a node never built splits by the distance rule only)."""
        c = self.cam
        if not isinstance(c, dict) or self.s.lod_px <= 0 or getattr(self, "cam_aid", None) != self.active:
            return None
        try:
            cx, cy, cz = float(c["x"]), float(c["y"]), float(c["z"])
            h, fov = float(c.get("h") or 0), float(c.get("fovDeg") or 60)
        except (KeyError, TypeError, ValueError):
            return None
        if not (h > 0 and 1 < fov < 170 and math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cz)):
            return None
        ppr = h / (2 * math.tan(math.radians(fov) / 2))     # BeamNG's fov: vertical
        aid, tiles = self.active, self.s.tiles

        def err(node):
            if len(node) != 3:
                return None
            info = tiles.get((aid, farfield.Tile(*node).key))
            return None if info is None else info.err
        reach, cap = farfield.SSE_REACH, farfield.SSE_SPACING_CAP
        tune = os.environ.get("BNGE_LOD")      # experiments: "reach,cap"
        if tune:
            reach, cap = (float(v) for v in tune.split(","))
        return farfield.View(cx, cy, cz - getattr(self, "pz", cz), ppr, err, tau=self.s.lod_px, reach=reach,
                             cells=self.s.layout.cells, cap=cap)

    def _regroup(self):
        aid = self.active
        cur = {k for a, k in self.shown if a == aid or a == GLOBAL_AID}
        if len(self.gpolys) > 4 * len(self.desired) + 1000:
            self.gpolys = {ck: q for ck, q in self.gpolys.items() if ck[0] == aid}
        self.groups = farfield.swap_groups(cur, self.desired, self.poly_of)
        # Sent tiles no group needs any more: the mod drops them. (A global one kept from a previous anchor takes
        # the tile of this selection: its quad in this anchor's world.)
        needed = {k for adds, _ in self.groups for k in adds}
        for k in self.sent:
            t = self.desired.get(k)
            if t is not None and self.sent[k] is not t:
                self.sent[k] = t
        for k in [k for k in self.sent if k not in needed]:
            del self.sent[k]
            self._retire([self.ck(k)])
            self.send({"type": "far_remove", "anchorId": aid, "key": k})
        # Build order: L0 under the committed / pending block (the mod's cover and commit
        # gating need them), then groups that are partly sent, then the rest; nearest first.
        gate = [b.bbox for b in (self.block, self.pending_block) if b is not None and self.s.near_visual in BLOCK_DRAWN]
        started = {k for adds, _ in self.groups if any(a in self.sent for a in adds) for k in adds}
        fx, fy = self.view_focus
        todo = [self.desired[k] for k in needed if k not in self.sent]

        cone = (farfield.view_cone(self.cam)
                if isinstance(self.cam, dict) and getattr(self, "cam_aid", None) == aid else None)
        dist = farfield.view_distances(todo, cone, fx, fy)

        def prio(k):
            t = todo[k]
            cls = 2
            if t.level in (0, self.s.forced_level) and any(farfield.overlaps(t.bbox, b) for b in gate):
                cls = 0
            elif t.key in started:
                cls = 1
            return (cls, dist[k], t.level, t.key)
        self.queue = [todo[k] for k in sorted(range(len(todo)), key=prio)]
        self.qi = 0
        self.failed.clear()

    def _next_tile(self):
        while self.qi < len(self.queue):
            t = self.queue[self.qi]
            self.qi += 1
            ck = self.ck(t.key)
            if t.key not in self.sent and ck not in self.inflight and t.key in self.desired:
                return t
        return None

    def emit_groups(self):
        """far_swap every group whose adds are all sent."""
        aid = self.active
        rest = []
        info = self.s.tiles
        live = set()
        for adds, removes in self.groups:
            if not all(k in self.sent for k in adds):
                rest.append((adds, removes))
                continue
            gk = (aid, tuple(adds), tuple(removes))
            live.add(gk)
            fut = self.deltas.get(gk)
            if fut is None:
                special = any(farfield.is_special_key(k) for k in adds + removes)
                try:
                    if special:
                        # a global tile's grid is on its Mercator vertices, a piece's its node's: the mod's bound,
                        # 1.2 (err_new + err_old) + 0.2 (crossover.delta) on the group's largest errors
                        en = max([info[self.ck(k)].err for k in adds if self.ck(k) in info] or [0.0])
                        eo = max([info[self.ck(k)].err for k in removes if self.ck(k) in info] or [0.0])
                        raise LookupError(1.2 * (en + eo) + 0.2)
                    args = ([(farfield.tile_of(k), info[(aid, k)].grid) for k in adds],
                            [(farfield.tile_of(k), info[(aid, k)].grid) for k in removes])
                except LookupError as bound:    # a tile record was evicted: fall back to the err bound
                    e = [info[self.ck(k)].err for k in adds + removes if self.ck(k) in info]
                    d = 1.2 * sum(e) + 0.2 if e else 0.0
                    if special:
                        d = float(bound.args[0])
                    fut = concurrent.futures.Future()
                    fut.set_result((d - DELTA_MARGIN, d - DELTA_MARGIN))
                else:
                    fut = self.s.delta_pool.submit(farfield.group_deltas, args)
                    loop = asyncio.get_running_loop()
                    fut.add_done_callback(lambda _f: loop.is_closed() or loop.call_soon_threadsafe(self.wake.set))
                self.deltas[gk] = fut
            if not fut.done():
                rest.append((adds, removes))
                continue
            del self.deltas[gk]
            if fut.cancelled() or fut.exception() is not None:
                log.warning("swap deltas of %s: %r", gk[1], None if fut.cancelled() else fut.exception())
                e = [info[self.ck(k)].err for k in adds + removes if self.ck(k) in info]
                d_in = d_out = (1.2 * sum(e) + 0.2 if e else 0.0) - DELTA_MARGIN
            else:
                d_in, d_out = fut.result()
            d_in, d_out = d_in + DELTA_MARGIN, d_out + DELTA_MARGIN
            gid = self.next_group
            self.next_group += 1
            self.send({"type": "far_swap", "anchorId": aid, "id": gid, "add": adds, "remove": removes,
                       "dIn": round(d_in, 2), "dOut": round(d_out, 2)})
            if log.isEnabledFor(logging.DEBUG) and adds and removes and \
                    not any(farfield.is_special_key(k) for k in adds + removes):    # the pop's size: amplitude / distance
                dist = min(farfield.tile_of(k).distance(*self.focus) for k in adds + removes)
                log.debug("far_swap %d L%d->L%d dIn %.1f dOut %.1f at %.0f m (%.2f mrad)", gid,
                          farfield.tile_of(removes[0]).level, farfield.tile_of(adds[0]).level, d_in, d_out, dist,
                          1000 * max(d_in, d_out) / max(dist, 1.0))
            for k in removes:
                self.shown.pop(self.ck(k), None)
            self._retire(self.ck(k) for k in removes)
            for k in adds:
                self.shown[self.ck(k)] = self.sent.pop(k).bbox
            self.batch[2] += 1
            self.credits.poke(self)         # other tiles on screen
        self.groups = rest
        for gk in [gk for gk in self.deltas if gk not in live]:     # groups re-planned meanwhile
            self.deltas.pop(gk).cancel()

    def flush_old(self):
        """far_remove every shown tile of a previous anchor once no desired tile that is not
        swapped in yet overlaps it (so nothing ever leaves a hole)."""
        aid = self.active
        cands = [ck for ck in self.shown if ck[0] != aid and ck[0] != GLOBAL_AID]
        if not cands:
            return
        pending = [t.bbox for k, t in self.desired.items() if self.ck(k) not in self.shown]
        if pending:
            c = np.array([self.shown[ck] for ck in cands])
            u = np.array(pending)
            blocked = ((c[:, None, 0] < u[None, :, 2]) & (u[None, :, 0] < c[:, None, 2]) &
                       (c[:, None, 1] < u[None, :, 3]) & (u[None, :, 1] < c[:, None, 3])).any(1)
        else:
            blocked = [False] * len(cands)
        for ck, b in zip(cands, blocked):
            if not b:
                del self.shown[ck]
                self._retire([ck])
                self.send({"type": "far_remove", "anchorId": ck[0], "key": ck[1]})
                self.credits.poke(self)

    def _blocked(self):
        """Groups that cannot complete now (an add failed to build)."""
        return [g for g in self.groups if any(k in self.failed for k in g[0])]

    async def stream(self):
        self._stream_waker = None
        try:
            await self._stream()
        except ConnectionError:
            pass
        except asyncio.CancelledError:
            for fut, _ in self.inflight.values():
                fut.cancel()
            raise
        finally:
            if self._stream_waker is not None:
                self._stream_waker.cancel()
                await asyncio.gather(self._stream_waker, return_exceptions=True)
                self._stream_waker = None

    async def _stream(self):
        """Build the needed tiles by priority, send them, emit swap groups as they complete;
        one task per connection."""
        while True:
            # Clear before reading the state, so an update from here on wakes us again.
            self.wake.clear()
            anchor = self.s.anchors.get(self.active)
            # While a physics block builds in this process, one far tile at a time: they share the interpreter
            # (GIL), and three far workers made a block build 3-10x slower (landcover 1.4 s vs 0.1 s), too slow
            # to follow a car at 58 m/s (2026-10-12). The far tile finishing next refills the workers.
            workers = HM_BUSY_TILE_WORKERS if self.s.hm_here else self.s.tile_workers
            # plus the helpers' slots (compute sharing): their tiles cost this process only the files (26 ms a
            # tile); kept to the local workers, a Steam Deck sent 2 at a time to 12 idle helper workers (2026-10-22)
            workers += self.s.remote_slots()
            while anchor is not None and not self.teleport_pending and len(self.inflight) < workers:
                t = self._next_tile()
                if t is None:
                    break
                self.inflight[self.ck(t.key)] = (self.s.tile(anchor, t), t)
            if anchor is not None and self.s.imagery is not None and not self.teleport_pending:
                # a tile at most every PREFETCH_AGAIN_S (a prefetch skips sources short of request tokens): the
                # loop wakes on every tile done and every message from the game, and asking again for the same 40
                # tiles each time kept the event loop busy (~22 s of a teleport on a Steam Deck, 2026-10-20)
                now = time.monotonic()
                if len(self.prefetched) > PREFETCHED_MAX:
                    self.prefetched.clear()
                for t in self.queue[self.qi:self.qi + IMAGERY_PREFETCH]:
                    ck = self.ck(t.key)
                    if ck not in self.s.tiles and now - self.prefetched.get(ck, -1e9) >= PREFETCH_AGAIN_S:
                        self.prefetched[ck] = now
                        self.s.prefetch_imagery(anchor, t)
            if anchor is not None:
                self.emit_groups()
                self.flush_old()
            if (anchor is not None and getattr(self, "lod_ramp", 1.0) < 1.0 and (self.map_view or self.map_open)
                    and not self.inflight and self.dirty and len(self.groups) == len(self._blocked())):
                # (only once a step is whole: stepping on a timer instead left the coarse tiles not built yet out of
                # the next selection, holes in the view for 15 s, Bozeman 2026-11-07)
                self.lod_ramp = min(1.0, self.lod_ramp * MAP_RAMP_STEP)     # (the next, finer step: commit_anchor)
                self.zoom_ref = self.eye    # (each step: what is shown is this eye's; a zoom from here starts over)
                log.info("anchor %d: 3D map detail step x%.2f after %.2f s", anchor.id, self.lod_ramp,
                         time.perf_counter() - self.batch[0])
                self.update_far()
                continue
            if (not self.inflight and self.dirty and anchor is not None
                    and len(self.groups) == len(self._blocked())):
                self.dirty = False
                n = sum(1 for k in self.desired if self.ck(k) in self.shown)
                self.send({"type": "far_done", "anchorId": anchor.id, "count": n})
                self.far_done_at = time.monotonic()
                if self.map_view or self.map_open:
                    self.zoom_ref = self.eye    # (all shown at this eye's detail: the zoom rule's reference)
                # (full priority: the roads too, on_pose; a catch-up boost is ended by catch_up only: ended here, it
                # stayed off for the rest of a 150 mph drive, its flag still on, 2026-11-02)
                if not self.s.boost_full and not self.catchup:
                    self.s.load_boost(False, "far field done")
                self.status(f"Far field ready ({n} tiles)")
                bs = sorted(self.s.build_stats)
                log.info("anchor %d far field done: %d tiles (%d sent, %d swaps in %.2f s; build median %.1f ms)",
                         anchor.id, n, self.batch[1], self.batch[2], time.perf_counter() - self.batch[0],
                         1000 * bs[len(bs) // 2] if bs else 0.0)
                log.info("tile build ms (geometry+write / texture) and KB (shape / texture) by level: %s",
                         self.s.level_summary())
                if getattr(self.s.far_pool, "stats", None) is not None and self.batch[1] >= 50:   # (a full one)
                    g = getattr(self.s.far_pool, "group", None)
                    log.info("compute sharing: %s; group %s", self.s.far_pool.stats,
                             g["ids"] if g else "none")
                if self.s.imagery is not None:
                    log.info("imagery: %s", self.s.imagery.summary())
                    if self.s.imagery_worker or self.s.base_worker:
                        log.info("imagery in the worker processes: %d far textures (%d after a fetch here), "
                                 "%d base photos", self.s.imagery_worker, self.s.refetched, self.s.base_worker)
                if self.s.retex_stats["started"]:
                    log.info("live retexture: %s, %d waiting", self.s.retex_stats, len(self.s.retex_todo))
                if not any(ck[0] != anchor.id and ck[0] != GLOBAL_AID for ck in self.shown):
                    self.s.schedule_prune(anchor.id)
            await self.writer.drain()
            if self._stream_waker is None:
                self._stream_waker = asyncio.ensure_future(self.wake.wait())
            futs = [f for f, _ in self.inflight.values()]
            done, _ = await asyncio.wait(futs + [self._stream_waker], return_when=asyncio.FIRST_COMPLETED)
            if self._stream_waker in done:
                self._stream_waker = None
            if (self.map_view or self.map_open or self.eye_clearance() > FLY_TEX_EYE_M) and anchor is not None and \
                    time.monotonic() - getattr(self, "_texup_at", 0.0) > MAP_TEX_UPGRADE_S:
                # (the camera turned: tiles shown with a smaller texture than they now need on screen, sharper)
                self._texup_at = time.monotonic()
                for ck in [ck for ck in self.shown if ck[0] == anchor.id or ck[0] == GLOBAL_AID]:
                    info = self.s.tiles.get(ck)
                    if info is None or info.material is None or not info.tex_px or info.tile is None:
                        continue
                    if self.s.tile_tex_px(anchor, info.tile) > info.tex_px and not info.stale:
                        info.stale = True
                        self.s.retex_request(ck, pump=False)
            if self.s.retex_todo and (self.map_view or self.map_open):
                self.s._retex_pump()        # (re-renders into the far slots builds left free)
            for ck in [ck for ck, (f, _) in self.inflight.items() if f.done()]:
                fut, t = self.inflight.pop(ck)
                if fut.cancelled():
                    continue
                if fut.exception() is not None:
                    log.error("far tile %s of anchor %d failed: %r", t.key, ck[0], fut.exception(),
                              exc_info=fut.exception() if t.key not in self.failed else None)   # (its first)
                    self.failed.add(t.key)
                    continue
                if ck != self.ck(t.key) or t.key not in self.desired or ck in self.shown or t.key in self.sent:
                    continue            # superseded meanwhile; the file stays cached
                # the server's latest record: a live retexture may have finished since this build
                info = self.s.tiles.get(ck, fut.result())
                if info.material is not None and not self.s.matslots.owns(ck, info.material):
                    info = self.s.reslot(ck, info)      # its material name went to another tile meanwhile
                    if not self.s.matslots.owns(ck, info.material):
                        # not renamed: sent with that name it would draw the other tile's texture; built again
                        # (its files are written anew; the cached entry and its claim on the name go)
                        self.s.tiles.pop(ck, None)
                        self.s.matslots.release(ck)
                        self.queue.append(t)
                        continue
                self.sent[t.key] = t
                if info.stale:          # a region turned ready since: re-rendered live, then far_retex
                    self.s.retex_request(ck)
                self.batch[1] += 1
                m = {"type": "far_tile", "anchorId": self.active, "key": t.key, "level": t.level,
                     "shape": f"{self.s.vfs_dir}/{info.shape or self.s.tile_name(*ck)}",
                     "pos": [round(v, 3) for v in info.pos], "err": round(info.err, 3),
                     "skirt": round(info.skirt, 2), "state": "group"}
                if info.sink is not None:
                    m["sink"] = round(info.sink, 2)
                if isinstance(t, farfield.GTile):   # geotiles phase 3b: placed by the mod from its own frame
                    m["frame"] = [info.frame[0], info.frame[1]]
                    m["pos"] = [0, 0, round(info.pos[2], 3)]
                    m["poly"] = [round(v, 2) for v in t.quad]
                elif isinstance(t, farfield.Piece):
                    m["poly"] = [round(v, 2) for v in t.poly]
                    self.ppolys[t.key] = t.poly
                if info.shape is not None and info.shape.startswith("slot_"):  # pooled (matslots.py)
                    m["slotSerial"] = serial = next(self.s.slot_serial)
                    self.slot_holds[matslotsmod.MatSlots.number(info.material)] = serial
                if info.material is not None:   # protocol 4: the tile's own imagery material
                    m["material"] = info.material
                    m["texture"] = f"{self.s.vfs_dir}/{info.texture}"
                    m["detail"] = True          # the shape has UV1 (world-locked detail map)
                    if self.s.layout.format >= 4:   # M1: texture px, the mod's governor unit kind
                        m["tex"] = info.tex_px or self.s.tex_size(t.level)
                self.send(m)
                if FAR_TRACE:
                    log.info("fartrace %.2f s L%d %s build %.0f ms tex %.0f ms quick %s", time.perf_counter() -
                             self.batch[0], t.level, t.key, 1000 * info.seconds, 1000 * info.tex_seconds,
                             any(isinstance(p, tuple) and p[:1] == (imagerymod.IMG_PENDING,) for p in info.cm_pending))
                if self.batch[1] % 50 == 0:
                    left = sum(1 for g in self.groups for k in g[0] if k not in self.sent)
                    # this change's tiles (sent / sent + still to send), not the whole far field
                    self.status(f"Far field loading: {self.batch[1]}/{left + self.batch[1]} tiles")
