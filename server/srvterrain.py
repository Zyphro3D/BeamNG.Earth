"""World server: terrain blocks and base layers (split out of earthd.py's Server): the ground and its grader,
heightmaps (provisional, follow, here), tunnels, base and quick base photos.
"""
import os
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import concurrent.futures  # noqa: E402
from runtimeprofiler import ThreadPoolExecutor  # noqa: E402
import numpy as np  # noqa: E402
import blockworker  # noqa: E402
import surface as surfacemod  # noqa: E402
import roads as roadsmod  # noqa: E402
import urgency as urgencymod  # noqa: E402
import plancache as plancachemod  # noqa: E402
import grading as gradingmod  # noqa: E402
import mapcells  # noqa: E402
import pacer  # noqa: E402
import bc7  # noqa: E402
import terraincache  # noqa: E402
import buildings as buildingsmod  # noqa: E402
import rockface as rockfacemod  # noqa: E402
import heightmap  # noqa: E402
import staging  # noqa: E402
import earthbase
from earthbase import (
    AHEAD_CELLS,
    AHEAD_LAYERS,
    AHEAD_MARGIN_M,
    AHEAD_MAX_M,
    AHEAD_MIN_V,
    AHEAD_MOVE_M,
    AHEAD_S,
    BAKE_FETCH,
    BASE_AO,
    BASE_FAST_WAIT_S,
    BASE_NO_STANDIN_MAX_S,
    BASE_RENDER_PX,
    BASE_SHIFT_M,
    BASE_WAIT_S,
    HM_CACHE,
    IMG_WARM_MARGIN,
    LANE_PREFETCH,
    LANE_REQUEST,
    PHOTO_SKIP_V,
    PROVISIONAL_ELSEWHERE,
    QUICK_BASE_PX,
    QUICK_MADE_KEPT,
    QUICK_MIN_COVER,
    QUICK_SHARE_WAIT_S,
    QUICK_SYNC_MAX_V,
    SLOW_BASE_S,
    STANDIN_WAIT_S,
    TUNNEL_NEAR_M,
    URGENT_ELSEWHERE,
    WARM_AROUND,
    WARM_STEPS_S,
    base_ao_vfs,
    log,
)


class _TerrainMixin:
    """Server's terrain blocks and base layers (split out of earthd.py; Server inherits it)."""

    def ground_z_at(self, anchor, e, n):
        """The ground's height (the frame's up) at (e, n) of anchor's frame, from the elevation data (not graded: the
        mod sets the car down on the collision there)."""
        la, lo, _ = anchor.frame.enu_to_geodetic(e, n, 0.0)
        la, lo = float(la), float(lo)
        z = self.ground(anchor.frame, la, lo, float(self.dem.sample(la, lo, anchor.hm_zoom)))
        return round(float(anchor.frame.geodetic_to_enu(la, lo, z)[2]), 3)

    def ground(self, frame, la, lo, z):
        """blockworker.ground here: runways graded in, the sea floor carved, the land off the roads roughened."""
        z = np.asarray(z, np.float64) + gradingmod.terrain_bumps(np.asarray(la, np.float64), np.asarray(lo, np.float64))
        if self.airports is not None:
            z = self.airports.apply(la, lo, z)
        if self.sea is not None:
            z = self.sea.carve(la, lo, z)
        return z

    def grader_for(self, anchor):
        """The anchor's grading.Grader (--grading on), else None."""
        if not self.grading:
            if os.environ.get("BNGE_WORLD_BUNDLE"):
                from worldbundle import BundleError
                raise BundleError("Pinned road plans require roads and grading enabled")
            return None
        with self.grader_lock:
            if anchor.grader is None:
                # (every anchor's view of one store: the plans are the planning regions', whatever the anchor)
                if getattr(self, "plan_store", None) is None:
                    self.plan_store = gradingmod.PlanStore(
                        self.mapdata, lambda la, lo: self.dem.sample(la, lo, gradingmod.PROFILE_ZOOM), self.road_widths)
                grader = gradingmod.Grader(anchor.frame, self.mapdata, self.plan_store.elev, self.road_widths,
                                          store=self.plan_store)
                grader.wet = self.sea.depth if self.sea is not None else None    # (no dams: underwater tunnels)
                grader.water = self.sea.water if self.sea is not None else None
                grader.tunnel_raw = lambda la, lo: self.ground(None, la, lo,
                    self.dem.sample(la, lo, gradingmod.PROFILE_ZOOM))
                grader.dem = self.dem                # (its urgent lane: _plan_missing)
                grader.shared = plancachemod.for_grader(self.cache_dir, grader, self.road_widths, grader.wet)
                anchor.grader = grader       # publish only after a pinned bundle passed compatibility checks
            return anchor.grader

    def warm_roads(self, anchor, x, y, vx, vy):
        """Queue the road plans of the cells the physics block will cover along the player's straight path
        (WARM_STEPS_S; the block's half width + the grading pad), then those within WARM_AROUND."""
        g = self.grader_for(anchor)
        if g is None:
            return
        reach = self.layout.block * heightmap.SQUARE / 2 + gradingmod.T_MAX + 40.0
        keys = []
        for t, r in [(t, reach) for t in WARM_STEPS_S] + [(0.0, WARM_AROUND)]:
            la, lo, _ = anchor.frame.enu_to_geodetic(x + vx * t, y + vy * t, 0.0)
            la, lo = float(la), float(lo)
            dlat = r / 111320.0
            dlon = r / (111320.0 * max(0.05, math.cos(math.radians(la))))
            for k in mapcells.cells_for_bbox(la - dlat, lo - dlon, la + dlat, lo + dlon):
                if k not in keys:
                    keys.append(k)
        for _, q in self.block_pools:
            try:
                q.put_nowait((anchor.id, anchor.lat, anchor.lon, keys))
            except Exception as e:      # a full / closed queue only costs the lookahead
                log.debug("road plans ahead: %s", e)
        if self.road_pool is not None:      # road tiles plan in their processes: this one needs none ahead
            return
        with self.warm_cv:
            self.warm_want = [(g, k) for k in keys]
            if self.warm_thread is None:
                self.warm_thread = threading.Thread(target=self._warm_loop, name="roads-ahead", daemon=True)
                self.warm_thread.start()
            self.warm_cv.notify()

    def bake_sync(self, anchor, x, y):
        """The data repo's baked packs near the player (bakepack.sync: road plans and cell facts made offline) fetched
        in a thread of their own, as the player moves (at most every bakepack.CHECK_S)."""
        if not BAKE_FETCH or self.mapdata is None or not self.grading or self.net is None:
            return
        import bakepack
        now = time.monotonic()
        if bakepack._sync["busy"] or now - bakepack._sync["at"] < bakepack.CHECK_S:
            return
        la, lo, _ = anchor.frame.enu_to_geodetic(x, y, 0.0)
        sig = f"w{int(self.road_widths is not None)}s{int(self.sea is not None)}"
        pool = self.__dict__.setdefault("bake_pool", ThreadPoolExecutor(1, thread_name_prefix="bake-packs"))
        pool.submit(bakepack.sync, self.cache_dir, self.net, self.mapdata, float(la), float(lo), sig)

    def download_ahead(self, anchor, x, y, vx, vy):
        """Fetch the elevation tiles (the blocks' zoom) and the road tiles' map cells along the player's straight
        path AHEAD_S s ahead into the disk caches (one round at a time, again every AHEAD_MOVE_M)."""
        v = math.hypot(vx, vy)
        if not earthbase.AHEAD_ON or v < AHEAD_MIN_V or self.ahead_busy or self.dem is None:
            return
        last = self.ahead_last
        if last is not None and last[0] == anchor.id and math.hypot(x - last[1], y - last[2]) < AHEAD_MOVE_M:
            return
        self.ahead_last = (anchor.id, x, y)
        k = min(1.0, AHEAD_MAX_M / (v * AHEAD_S[-1]))
        pts = []
        for t in AHEAD_S:
            la, lo, _ = anchor.frame.enu_to_geodetic(x + vx * t * k, y + vy * t * k, 0.0)
            pts.append((float(la), float(lo)))
        reach = self.layout.block * heightmap.SQUARE / 2 + AHEAD_MARGIN_M
        zoom = anchor.hm_zoom
        layers = (AHEAD_LAYERS if self.buildings else ("roads",)) if self.mapdata is not None else ()
        self.ahead_busy = True

        def run():
            try:
                dem_n = cells_n = 0
                for la, lo in pts:
                    got, deferred = self.dem.prefetch(self.dem.keys_around(la, lo, reach, zoom))
                    dem_n += got
                    if deferred:
                        self.ahead_stats["deferred"] += 1
                        break
                jobs = []
                for la, lo in pts if layers else ():
                    dlat = reach / 111320.0
                    dlon = reach / (111320.0 * max(0.05, math.cos(math.radians(la))))
                    for key in mapcells.cells_for_bbox(la - dlat, lo - dlon, la + dlat, lo + dlon):
                        todo = tuple(ly for ly in layers if not self.mapdata.has(key, ly))
                        if todo and all(key != j[0] for j in jobs):
                            jobs.append((key, todo))
                # (AHEAD_CELLS cells at once, nearest first: one after another, a slow connection's round fell behind
                # the car, 48 cells in a 20 km first visit while its road tiles waited for theirs, 2026-11-04)
                if jobs:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(AHEAD_CELLS, thread_name_prefix="download-ahead-cell") as pool:
                        for _ in pool.map(lambda j: self.mapdata.cell(j[0], layers=j[1], kind="prefetch"), jobs):
                            cells_n += 1
                self.ahead_stats["dem"] += dem_n
                self.ahead_stats["cells"] += cells_n
                if dem_n or cells_n:
                    log.info("download ahead: %d elevation tiles, %d map cells (%.0f m/s; session: %s)", dem_n,
                             cells_n, v, self.ahead_stats)
            except Exception as e:      # a refused / failed fetch only costs the lookahead
                log.debug("download ahead: %s: %s", type(e).__name__, e)
            finally:
                self.ahead_busy = False
        self.ahead_pool.submit(run)

    def warm_imagery(self, anchor, x, y, vx, vy):
        """The lazy imagery sources (NAIP) along the player's straight path (WARM_STEPS_S), over the block's
        footprint and a margin, at the base photo's texel size: fetched before the blocks there are built."""
        if self.imagery is None:
            return
        reach = self.layout.block * heightmap.SQUARE / 2 + IMG_WARM_MARGIN
        pts = [(x + vx * t, y + vy * t) for t in WARM_STEPS_S]
        self.imagery.warm(anchor.frame, pts, reach, self.layout.block * heightmap.SQUARE / BASE_RENDER_PX)

    def _warm_loop(self):
        while True:
            pacer.hold()                # (a teleport's first block first)
            with self.warm_cv:
                job = None
                while job is None:
                    job = next(((g, k) for g, k in self.warm_want if not g.planned(k)), None)
                    if job is None:
                        self.warm_cv.wait()
            try:
                t0 = time.perf_counter()
                job[0].warm(job[1])
                log.debug("road plans ahead: %s in %.2f s (%d wanted)", job[1], time.perf_counter() - t0,
                          len(self.warm_want))
            except Exception:
                log.exception("road plans ahead for %s", job[1])
                with self.warm_cv:      # not again for this pose list
                    self.warm_want = [w for w in self.warm_want if w[1] != job[1]]

    def make_heightmap(self, anchor, cx, cy, progress, lane=LANE_REQUEST, queued_at=None):
        """Worker thread: build and write one 1024² terrain. The follow / prefetch blocks (no progress
        callback) build in the block process when there is one; the anchor's first block here, so its
        DEM download reports progress."""
        t0 = time.perf_counter()
        if queued_at is not None and t0 - queued_at > 0.5:
            log.info("terrain %s waited %.2f s for a heightmap worker", (anchor.id, cx, cy), t0 - queued_at)
        q = self.anchor_q(anchor)
        fb, base = anchor.first_base, None
        if fb is not None and fb[0] == (cx, cy):        # the provisional first block's photo (make_provisional)
            anchor.first_base, base = None, fb[1]
        elif self.near_visual == "detail" and self.imagery is not None and self.fast \
                and (lane == LANE_PREFETCH or (progress is None and getattr(self, "speed", 0.0) >= PHOTO_SKIP_V)) \
                and (anchor.id, cx, cy) not in self.hm_committed:     # (a rebuild of the block in use keeps its photo)
            # (and faster than PHOTO_SKIP_V a requested one too: in use 2-3 s at 150 mph, its 3-7 s photo came after
            # the car had left it, 36 photos for 28 blocks of a 3 mile drive, two of the server's four CPUs;
            # base_refresh draws the block in use's, one at a time, and the stand-ins after it take its overlap)
            # a fast player's prefetched block: its photo only once the mod has it (base_refresh: a stale base); a
            # 3-8 s photo for every guessed block, half of them never asked for, took the CPU the road tiles needed
            # (500 mph, 2026-10-30)
            self.base_stale.add((anchor.id, cx, cy))
            self.base_skipped.add((anchor.id, cx, cy))
            if getattr(self, "speed", 0.0) < QUICK_SYNC_MAX_V:
                # (its stand-in beside the heights too: made after them it was 2-8 s more of every prefetched block,
                # which a request for it then waited out; nyc_speed.py 2026-11-02)
                self.quick_futs[(anchor.id, cx, cy)] = self.quick_early.submit(self._stand_in, anchor, cx, cy)
        elif self.near_visual == "detail" and self.imagery is not None:
            base = (self.urgent_base_pool if lane == LANE_REQUEST else self.base_pool).submit(
                self._base_photo, anchor, cx, cy, self._base_process(lane), lane == LANE_REQUEST)
            if lane == LANE_REQUEST and self.fast and getattr(self, "speed", 0.0) < QUICK_SYNC_MAX_V:
                # (its stand-in photo now, beside the heights, in the other lane's process: made after them it took
                # 1.3-5 s more of every block at 100-200 mph down Manhattan, nyc_speed.py 2026-11-02)
                self.quick_futs[(anchor.id, cx, cy)] = self.quick_early.submit(self._stand_in, anchor, cx, cy)
        pool = self.block_pools[lane][0] if self.block_pools and progress is None else None
        key = (anchor.id, cx, cy)
        try:
            if pool is not None:
                self.hm_stage[key] = "the block process"
                submitted, submitted_pc = time.time(), time.perf_counter()
                pfut = self.hm_proc[key] = pool.submit(
                    blockworker.build, anchor.id, anchor.lat, anchor.lon, anchor.hm_zoom, q, cx, cy,
                    self.layout.block, self.out_dir, self.hm_name(key), self.ter_name(key),
                    fuel_pads=self.block_fuel_pads(anchor, cx, cy, self.layout.block))
                try:
                    info = pfut.result()
                finally:
                    self.hm_proc.pop(key, None)
                if "started" in info:   # (waiting for the lane's process: a block before it, its road plans ahead)
                    started, finished = info.pop("started"), info.pop("finished", None)
                    info["stages"] = {"pre": submitted_pc - t0, "queue": max(0.0, started - submitted),
                                      **({"work": finished - started, "back": max(0.0, time.time() - finished)}
                                         if finished else {}),
                                      **info.get("stages", {})}
                self.hm_stage[key] = "its base photo"
                return self._finish_heightmap(anchor, cx, cy, info, t0, base, urgent=lane == LANE_REQUEST, lane=lane)
            self.hm_stage[key] = "the build in the server's process"
            urgent = getattr(self.dem, "urgent", None)     # (test DEMs have no lane)
            with urgent() if urgent is not None else contextlib.nullcontext():     # the player waits for it (teleport,
                # resync): ahead of every other download
                return self._make_heightmap_here(anchor, cx, cy, progress, t0, q, base)
        finally:
            self.hm_stage.pop(key, None)

    def block_planned(self, anchor, cx, cy):
        """Are the road plans of every map cell under block (cx, cy) made (here, or on disk: this run's plan cache, the
        revisit cache)? Then its graded build is quick and a provisional block only doubles the work."""
        g = self.grader_for(anchor)
        if g is None or not hasattr(g, "store"):
            return True
        half = self.layout.block * heightmap.SQUARE / 2
        X, Y = np.meshgrid(cx + np.linspace(-half, half, 5), cy + np.linspace(-half, half, 5))
        la, lo, _ = anchor.frame.enu_to_geodetic(X.ravel(), Y.ravel(), 0.0)
        la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
        for pl, idx in g.store.planners_of(la, lo):
            for k in pl._keys(la[idx], lo[idx]):
                sh = getattr(pl, "shared", None)
                if not (pl.planned(k) or (sh is not None and sh.has(k))):
                    return False
        return True

    def make_provisional_follow(self, anchor, cx, cy):
        """A follow block's provisional stand-in (Connection.provisional_follow) in a worker process (blockworker.build
        provisional: the DEM's ground, no road plans, sea floor or tunnels, the cached roads painted), its stand-in
        photo started beside it (at most STANDIN_WAIT_S; else none, the photo later). Built here (make_provisional) it
        took 11-33 s of a first visit beside the rest of this process's work (nyc_speed.py --cold, 2026-11-02)."""
        pool = self.photo_pool or (self.block_pools[LANE_PREFETCH][0] if self.block_pools else None)
        if pool is None:
            return self.make_provisional(anchor, cx, cy, None)
        t0 = time.perf_counter()
        key = (anchor.id, cx, cy)
        early = (self.quick_early.submit(self._stand_in, anchor, cx, cy)
                 if self.near_visual == "detail" and self.imagery is not None else None)
        info = pool.submit(blockworker.build, anchor.id, anchor.lat, anchor.lon, anchor.hm_zoom, self.anchor_q(anchor),
                           cx, cy, self.layout.block, self.out_dir, self.hm_name(key), self.ter_name(key),
                           provisional=True).result()
        info.pop("started", None)
        info.pop("finished", None)
        if early is not None:
            try:
                info["base"] = early.result(timeout=STANDIN_WAIT_S)
            except Exception:
                info["base"] = None
            if not info.get("base"):
                self.quick_pool.submit(self._quick_later, anchor, cx, cy)
        info["seconds"] = time.perf_counter() - t0
        info["zoom"] = anchor.hm_zoom
        log.info("provisional block %s in %.2f s in a worker (%s)", key, info["seconds"],
                 " ".join(f"{k} {v:.2f}" for k, v in info.get("stages", {}).items()))
        return info

    def make_provisional(self, anchor, cx, cy, progress, ground_at=(0.0, 0.0)):
        """Worker thread: a teleport's first block before its road plans (the user, 2026-10-22: "place the car as
        soon as possible"): the DEM's ground with the runways (no road grading, tunnels, arches or sea floor, which
        all wait on the map data of the cells around: 15-60 s of a fresh place's first block, ~5 s without), the
        cached land cover and the photo. The graded block replaces it when it is built (Connection.refine_first:
        terrain_refresh). Not kept in hm (the graded one is the block of its key), nor in anchor.chunks. ground_at: the
        point its groundZ is for (the car's spot)."""
        if os.environ.get("BNGE_WORLD_BUNDLE"):
            from worldbundle import BundleError
            raise BundleError("Ungraded provisional terrain is disabled for pinned road plans")
        t0 = time.perf_counter()
        q = self.anchor_q(anchor)
        base = None
        if self.near_visual == "detail" and self.imagery is not None:
            # (here, not in a block process: the other lane's process can still be busy with the old place's block,
            # and the photo waited 16 s behind it on a Deck, 2026-10-23)
            base = self.urgent_base_pool.submit(self._base_photo, anchor, cx, cy, None, True, True)
            anchor.first_base = ((cx, cy), base)
        pool = self.photo_pool or (self.block_pools[LANE_PREFETCH][0] if self.block_pools else None)
        if pool is not None and PROVISIONAL_ELSEWHERE:
            # in a block process: here it took 13-21 s at fresh dense places beside this interpreter's other work (its
            # roadPaint above all), 3.9 s for the same block in a fresh process (.claude/work/rust_port_plan.md 1a,
            # 2026-11-07); the follow blocks' stand-ins went there before (make_provisional_follow)
            key = (anchor.id, cx, cy)
            info = pool.submit(blockworker.build, anchor.id, anchor.lat, anchor.lon, anchor.hm_zoom, q, cx, cy,
                               self.layout.block, self.out_dir, self.hm_name(key), self.ter_name(key),
                               provisional=True, ground_at=tuple(ground_at)).result()
            info.pop("started", None)
            info.pop("finished", None)
            info = self._finish_heightmap(anchor, cx, cy, info, t0, base, urgent=True)
            log.info("provisional block %s in %.2f s in a worker (%s)", key, time.perf_counter() - t0,
                     " ".join(f"{k} {v:.2f}" for k, v in info.get("stages", {}).items()))
            return info
        urgent = getattr(self.dem, "urgent", None)     # (test DEMs have no lane)
        with urgent() if urgent is not None else contextlib.nullcontext():
            info = self._make_heightmap_here(anchor, cx, cy, progress, t0, q, base, provisional=True,
                                             ground_at=ground_at)
        log.info("provisional block %s in %.2f s (%s)", (anchor.id, cx, cy), time.perf_counter() - t0,
                 " ".join(f"{k} {v:.2f}" for k, v in info.get("stages", {}).items()))
        return info

    def _make_heightmap_here(self, anchor, cx, cy, progress, t0, q, base=None, provisional=False, ground_at=(0.0, 0.0)):
        with self.hm_here_lock:
            self.hm_here += 1
        try:
            return self._make_heightmap_here_(anchor, cx, cy, progress, t0, q, base, provisional, ground_at)
        finally:
            with self.hm_here_lock:
                self.hm_here -= 1

    def _make_heightmap_here_(self, anchor, cx, cy, progress, t0, q, base=None, provisional=False, ground_at=(0.0, 0.0)):
        if provisional and os.environ.get("BNGE_WORLD_BUNDLE"):
            from worldbundle import BundleError
            raise BundleError("Ungraded provisional terrain is disabled for pinned road plans")
        keep = {}
        g = None if provisional else self.grader_for(anchor)
        c0 = time.thread_time()

        grade_s = [0.0, 0.0]            # wall, this thread's CPU (the rest: waiting for map data, the interpreter)
        part_s = {"dem": 0.0, "sea": 0.0, "tunnels": 0.0}  # (the rest of "heights": the stages' log, 2026-10-30)

        def elev(la, lo):
            t1 = time.perf_counter()
            z = self.dem.sample(la, lo, anchor.hm_zoom, progress)
            part_s["dem"] += time.perf_counter() - t1
            if provisional:             # the runways only (the sea floor waits on the water cells)
                return self.airports.apply(la, lo, z) if self.airports is not None else z
            t1 = time.perf_counter()
            z = self.ground(anchor.frame, la, lo, z)
            part_s["sea"] += time.perf_counter() - t1
            if g is None:
                return z
            t1, c1 = time.perf_counter(), time.thread_time()
            z = g.apply(la, lo, z, gradingmod.ROAD_LIFT)
            grade_s[0] += time.perf_counter() - t1
            grade_s[1] += time.thread_time() - c1
            return z
        gen = elev
        if not provisional:
            blockworker.plan_ahead(g, anchor.frame, cx, cy, self.layout.block * heightmap.SQUARE / 2)
        if self.mapdata is not None and not provisional:
            # (made here when not yet: an anchor's first block was built before any road tile made them, without
            # its arches' raise and flat mouths, 2026-10-17)
            t1 = time.perf_counter()
            tun = self.tunnels_of(anchor, elev, g)
            part_s["tunnels"] += time.perf_counter() - t1
            half = self.layout.block * heightmap.SQUARE / 2
            cell_fn = lambda key: self.mapdata.cell(key, layers=("roads",), kind="demand").layers.get("roads")  # noqa: E731

            def gen(la, lo):        # the ground over the low arches raised over their vaults (roads.arch_raise)
                return roadsmod.block_arch_raise(tun, cell_fn, anchor.frame, la, lo, elev(la, lo),
                                                 (cx - half, cy - half, cx + half, cy + half))
        info = heightmap.generate(anchor.frame, gen,
                                  os.path.join(self.out_dir, self.hm_name((anchor.id, cx, cy))), cx, cy, q=q,
                                  size=self.layout.block, keep=keep, chunks=None if provisional else anchor.chunks,
                                  ground_at=ground_at)
        stages = {"heights": time.perf_counter() - t0, "grading": grade_s[0], "gradingCpu": grade_s[1],
                  "heightsCpu": time.thread_time() - c0, **part_s}
        idx = None
        if self.surface is not None:
            # ground models per 2 m sample from the cached land cover and the slope (surface.py);
            # rows flipped as the heightmap's (row 0 = north)
            t1 = time.perf_counter()
            classes, complete = self.surface.classes(keep["lat"], keep["lon"])
            idx = surfacemod.classify(classes, keep["u"], heightmap.SQUARE, keep["lat"], keep["lon"])
            if self.sea is not None and not provisional:    # under water: the bed (no grass through the surface)
                idx[self.sea.depth(keep["lat"], keep["lon"]) > blockworker.WATER_BED_M] = surfacemod.MUD
            road_feats, roads_complete = self.surface.roads(keep["lat"], keep["lon"])
            stages["landcover"] = time.perf_counter() - t1
            t1 = time.perf_counter()
            lots = (self.surface.lots(keep["lat"], keep["lon"]) if hasattr(self.surface, "lots") else None)
            if lots:                    # farmland under the carriageways (a field does not cover the asphalt)
                idx = surfacemod.paint_fields(idx, keep["lat"], keep["lon"], lots)
                idx = surfacemod.paint_golf(idx, keep["lat"], keep["lon"], lots)   # (golf over it: courses win)
            if road_feats:              # (provisional: the widths' guesses, not measured between the buildings)
                idx = surfacemod.paint_roads(idx, keep["lat"], keep["lon"], road_feats, heightmap.SQUARE,
                                             self.surface.urban, None if provisional else self.road_widths,
                                             blockworker._open_tunnels(g, keep))
            idx = surfacemod.paint_lots(idx, keep["lat"], keep["lon"], lots) if lots else idx
            if hasattr(self.surface, "fuel_pads"):      # gas station forecourts (--gas-stations on)
                idx = surfacemod.paint_pads(idx, keep["lat"], keep["lon"],
                                            self.surface.fuel_pads(keep["lat"], keep["lon"]))
            stages["roadPaint"] = time.perf_counter() - t1
            complete = complete and roads_complete
            info["layers"] = surfacemod.write_masks(idx[::-1], self.out_dir, self.hm_name((anchor.id, cx, cy))[:-4])
            info["surfaceComplete"] = complete
            info["roadsComplete"] = roads_complete
        if not provisional:
            idx = self._tunnel_holes(anchor, elev, keep, cx, cy, idx, info)
        info["ter"], mats = surfacemod.write_ter(self.out_dir, self.ter_name((anchor.id, cx, cy)), keep["qgrid"], idx)
        if mats is not None:          # the .ter's own material list (MATERIALS indices)
            info["mats"] = mats
        info["stages"] = stages
        return self._finish_heightmap(anchor, cx, cy, info, t0, base, urgent=True)       # (a teleport's, a first block: waited for)

    def tunnels_of(self, anchor, elev, g):
        """anchor.tunnels (roads.TunnelProfiles), made now if not yet (elev: the caller's ground; g: the anchor's
        Grader or None, taken before: grader_for takes grader_lock)."""
        with self.grader_lock:
            if anchor.tunnels is None:
                anchor.tunnels = self.chain_view(anchor, "tunnels", elev, g)
            return anchor.tunnels

    def tunnel_floor(self, anchor, e=0.0, n=0.0):
        """The floor of a tunnel tube at (e, n) of the anchor's frame (roads.tunnel_floor_at), or None: a teleport into a
        park arch set the car down on the soil over it, where it hung on the arch's top (2026-10-17). No tunnel in the
        map data within TUNNEL_NEAR_M: None at once (no tunnel profiles made)."""
        if self.mapdata is None:
            return None
        cell_fn = lambda key: self.mapdata.cell(key, layers=("roads",), kind="demand").layers.get("roads")   # noqa: E731
        try:
            r = TUNNEL_NEAR_M
            la, lo, _ = anchor.frame.enu_to_geodetic(np.array([e - r, e + r]), np.array([n - r, n + r]), np.zeros(2))
            cells = [cell_fn(k) for k in mapcells.cells_for_bbox(float(la.min()), float(lo.min()),
                                                                float(la.max()), float(lo.max()))]
            if not any(c is not None and any(s is not None for s in roadsmod.tunnel_specs(c)) for c in cells):
                return None
            g = self.grader_for(anchor)

            def elev(la, lo):
                z = self.ground(anchor.frame, la, lo, self.dem.sample(la, lo, gradingmod.PROFILE_ZOOM))
                return g.apply(la, lo, z, gradingmod.ROAD_LIFT) if g is not None else z
            self.tunnels_of(anchor, elev, g)
            box = (e - r, n - r, e + r, n + r)
            return roadsmod.tunnel_floor_at(anchor.tunnels.parts(cells, box), e, n)
        except Exception:
            log.exception("tunnel floor at anchor %s", anchor.id)
            return None

    def chain_view(self, anchor, kind, elev, g=None):
        """anchor's roads.ChainView of the server's bridge ("bridges") or tunnel ("tunnels") chains: one
        roads.ChainStore each for every anchor (its ground elev's values are the same whatever the anchor; the tunnels'
        raw / ends at grading.PROFILE_ZOOM, as in the block workers; g: a Grader or None). Under grader_lock."""
        stores = self.__dict__.setdefault("chain_stores", {})
        cs = stores.get(kind)
        if cs is None:
            cell_fn = lambda key: self.mapdata.cell(key, layers=("roads",), kind="demand").layers.get("roads")  # noqa: E731
            if kind == "bridges":
                def setup(p):
                    p.wet = self.sea.depth if self.sea is not None else None    # (decks clear the water)
                    p.infra_fn = lambda key: self.mapdata.cell(
                        key, layers=("infrastructure",), kind="demand").layers.get("infrastructure")
                cls = roadsmod.BridgeProfiles
            else:
                def setup(t):
                    t.wet = self.sea.depth if self.sea is not None else None
                    t.water = self.sea.water if self.sea is not None else None
                    t.raw = lambda la, lo: self.ground(None, la, lo, self.dem.sample(la, lo, gradingmod.PROFILE_ZOOM))
                    if g is not None:
                        t.ends = lambda la, lo: g.apply(la, lo, t.raw(la, lo), gradingmod.ROAD_LIFT, roads_only=True)
                cls = roadsmod.TunnelProfiles
            cs = stores[kind] = roadsmod.ChainStore(cls, cell_fn, elev, setup)
        return roadsmod.ChainView(cs, anchor.frame)

    def _tunnel_holes(self, anchor, elev, keep, cx, cy, idx, info):
        """idx with the block's tunnel portals as holes (roads.block_tunnel_holes; blockworker._tunnel_holes)."""
        g = self.grader_for(anchor)       # (before grader_lock: it takes it)
        if self.mapdata is None:
            return idx
        cell_fn = lambda key: self.mapdata.cell(key, layers=("roads",), kind="demand").layers.get("roads")   # noqa: E731
        try:
            self.tunnels_of(anchor, elev, g)
            holes = roadsmod.block_tunnel_holes(anchor.tunnels, cell_fn, anchor.frame, keep, cx, cy, heightmap.SQUARE)
        except Exception:
            log.exception("tunnel holes of block (%s, %s)", cx, cy)
            return idx
        if holes is None or not holes.any():
            return idx
        idx = np.zeros(holes.shape, np.uint8) if idx is None else idx
        idx[holes] = heightmap.TER_HOLE
        info["tunnelHoles"] = int(holes.sum())
        return idx

    def _base_process(self, lane):
        """The block process that renders the base photo of a block built in lane (blockworker.base_photo): the
        other lane's, usually idle while this one builds the heights; None (rendered here) without block processes.
        With the imagery analyses on (they live in this process) it renders from this process's finished tiles on
        disk (_base_elsewhere), else it is rendered here."""
        if not self.block_pools or self.imagery is None:
            return None
        photo = getattr(self, "photo_pool", None)
        if photo is not None and (lane != LANE_REQUEST or not getattr(self, "fast", False)):
            # (fast: the photo process makes a requested block's stand-in first, _stand_in.) A prefetched block's photo
            # there too: in the request lane's process it held the next block the game asked for, 0.2-1.8 s each
            # (a cold Chicago drive at 100 mph: requested blocks queued 2-4 s, the edge warning on, 2026-11-05)
            return photo
        return self.block_pools[LANE_PREFETCH if lane == LANE_REQUEST else LANE_REQUEST][0]

    def _base_photo(self, anchor, cx, cy, pool=None, urgent=False, quick=False):
        """base_pool: the block's base photo (write_base; in the block process pool when given and its imagery
        tiles are on disk) while its heights are built: (file name or None, [rgb] or [], seconds). urgent: the
        player waits on this block (a teleport's first, or one the mod asked for): its missing photo tiles are
        fetched ahead of the far field's (Imagery.urgent, netfetch "urgent"). quick: write_base's."""
        t1 = time.perf_counter()
        photo = []
        plain = URGENT_ELSEWHERE and urgent       # (analyses on: its photo without them at once, then refreshed)
        name = self._base_elsewhere(anchor, cx, cy, pool, photo, plain) if pool is not None else None
        if name is None:
            with self.imagery.urgent() if urgent and hasattr(self.imagery, "urgent") else contextlib.nullcontext():
                name = self.write_base(anchor, cx, cy, photo=photo, quick=quick)
        return name, photo, time.perf_counter() - t1

    def _base_elsewhere(self, anchor, cx, cy, pool, photo, plain=False, rev=0):
        """write_base in a block process (blockworker.base_photo): the file name, or None when it lacked an imagery
        tile on disk or failed (then rendered here). With the imagery analyses on, from this process's finished tiles
        on disk (tex["fin"], Imagery.finished_only: as write_base here would render it); one not finished yet fails
        it, and then plain (a block the player waits for) renders it there without them, refreshed later
        (base_stale). Every block's photo was rendered twice before (there without them, here with them: 2 x 12 MB
        of PNG and 2 x 5.6 MB of the game's packed copy a block, 2026-11-04)."""
        im = self.imagery
        if self.deshadow is not None or self.colormatch is not None:
            name = self._base_in(anchor, cx, cy, pool, photo,
                                 {s.name: im._fin_sigs.setdefault(s.name, im._fin_sig(s)) for s in im.chain}, rev)
            if name is not None or not plain:
                return name
        return self._base_in(anchor, cx, cy, pool, photo, None, rev)

    def _base_in(self, anchor, cx, cy, pool, photo, fin, rev=0):
        size = self.layout.block * heightmap.SQUARE
        key = (anchor.id, cx, cy)
        name = self.base_name(key, rev)
        im = self.imagery
        tex = {"chain": [s.name for s in im.chain], "tone": self.tone, "leaf_on": im.leaf_on,
               "season_mode": im.season_mode}
        if fin:
            tex["fin"] = fin
        try:
            got = pool.submit(blockworker.base_photo, anchor.lat, anchor.lon, cx - size / 2 - BASE_SHIFT_M,
                              cy - size / 2 - BASE_SHIFT_M, size,
                              BASE_RENDER_PX, tex, bool(self.buildings and self.mapdata is not None), anchor.zooms[0],
                              os.path.join(self.out_dir, name),
                              (self.user_dir, f"{self.vfs_dir}/{name}", base_ao_vfs(self.base_px), BASE_AO),
                              fuel_pads=self.block_fuel_pads(anchor, cx, cy, size), out_px=self.base_px).result()
        except Exception as e:
            log.info("base photo %s in the block process: %s", key, e)
            return None
        if got is None:
            return None
        rgb, _, pending = got
        self._base_kept(anchor, cx, cy, pending)
        if not fin and (self.deshadow is not None or self.colormatch is not None):
            # (rendered without the imagery analyses, which live in this process: rendered again here with them,
            # base_refresh, as a photo rendered before they were done)
            self.base_stale.add(key)
        photo.append(rgb)
        self.base_worker += 1
        return name

    def _base_kept(self, anchor, cx, cy, pending):
        """A base photo was rendered with these (imagery.render pending) still to come: base_refresh renders it
        again once they are here."""
        key = (anchor.id, cx, cy)
        self.base_pending[key] = frozenset(pending)
        if self.base_ready(anchor, cx, cy):
            self.base_stale.discard(key)
        else:
            self.base_stale.add(key)

    def _finish_heightmap(self, anchor, cx, cy, info, t0, base=None, urgent=False, lane=None):
        """_finish_heightmap_now; a block the game asked for (urgent) waiting for its photo: urgent work (urgency.py),
        the background jobs in this process wait to start meanwhile (its stand-in photo, 0.2 s of work, took 5 s and
        more beside the far tiles' and the vegetation's, the edge guard holding the car at 130-175 mph, 2026-10-31)."""
        if urgent and base is not None and not base.done():
            with urgencymod.urgent():
                return self._finish_heightmap_now(anchor, cx, cy, info, t0, base, lane)
        return self._finish_heightmap_now(anchor, cx, cy, info, t0, base, lane)

    def _finish_heightmap_now(self, anchor, cx, cy, info, t0, base=None, lane=None):
        """The base photo (detail mode; rendered here: the imagery corrections live in this process; base: its
        future from _base_photo, started with the heights), the timings."""
        if base is not None:
            t1 = time.perf_counter()
            if not base.done():
                # below FAST_BASE_V it waits for its own photo (BASE_WAIT_S at most): sent before it, each block's
                # ground swapped twice (stand-in, then the photo) and flashed at each (the user at 57-124 mph, 2026-10-30)
                concurrent.futures.wait([base], timeout=BASE_FAST_WAIT_S if self.fast else BASE_WAIT_S)
            quick = None
            early = self.quick_futs.pop((anchor.id, cx, cy), None)
            if not base.done() and early is not None:
                try:
                    quick = early.result(timeout=STANDIN_WAIT_S)
                except concurrent.futures.TimeoutError:
                    quick = None        # (the block goes without it: _quick_later / the photo when done)
                except Exception as e:
                    log.info("stand-in photo %s: %s", (anchor.id, cx, cy), e)
                info["stages"]["baseQuick"] = time.perf_counter() - t1
            if not base.done() and not quick and early is None and getattr(self, "speed", 0.0) < QUICK_SYNC_MAX_V \
                    and self.imagery is not None:     # (an early one tried: not again here, _quick_later)
                quick = self._quick_once(anchor, cx, cy, lambda: (
                    self._quick_elsewhere(anchor, cx, cy, lane) if lane == LANE_REQUEST else None)
                    or self.write_quick_base(anchor, cx, cy))
                info["stages"]["baseQuick"] = time.perf_counter() - t1
                if not quick and not self.fast:
                    # (no stand-in to show: slower, the photo is waited for, BASE_NO_STANDIN_MAX_S at most: without a
                    # limit an imagery host that kept failing held the block and the car at the edge for good,
                    # "paused until the server answers terrain requests", a player in Nova Scotia, 2026-11-03)
                    concurrent.futures.wait([base], timeout=BASE_NO_STANDIN_MAX_S)
            if not base.done():
                # the block goes now with a stand-in photo (below QUICK_SYNC_MAX_V made first: sent without one, the
                # ground the last blocks had already shown went to the flat base until it came, "it flashes already
                # painted ground", the user at 230 mph, 2026-10-30; faster, right after it) and its own when done
                info["base"] = quick
                if not quick:
                    self.quick_pool.submit(self._quick_later, anchor, cx, cy)
                info["stages"]["baseWait"] = time.perf_counter() - t1
                info["stages"]["baseLater"] = 1.0
                base.add_done_callback(lambda f, key=(anchor.id, cx, cy): self._base_later(key, f))
                info["seconds"] = time.perf_counter() - t0
                info["zoom"] = anchor.hm_zoom
                return info
            info["base"], photo, info["stages"]["baseRender"] = base.result()
            info["stages"]["baseWait"] = time.perf_counter() - t1
            if photo and info.get("layers") and info.get("ter"):
                try:                # lawns only where the photo is green (before the game reads the block)
                    surfacemod.photo_lawns(self.out_dir, info["ter"], info["layers"], photo[0])
                except Exception as e:
                    log.info("photo lawns %s: %s", (anchor.id, cx, cy), e)
        elif (anchor.id, cx, cy) in self.base_skipped and self.near_visual == "detail" and self.imagery is not None:
            # a block built ahead (a fast player's prefetch, not waited on yet) gets its stand-in before it is ready:
            # made after it, the mod could take the block first and show the level's green base until it came (a
            # player at 170 mph: road, brown, green, road again; 2026-10-30). Its own photo: base_refresh. Faster than
            # QUICK_SYNC_MAX_V: after it (on its own thread): made before it, the prefetch fell behind at 500 mph (the car
            # held 40 s of 90 against none); at 170 mph no hold, roads 1.2 km ahead.
            early = self.quick_futs.pop((anchor.id, cx, cy), None)
            if early is not None:
                t1 = time.perf_counter()
                try:
                    info["base"] = early.result(timeout=STANDIN_WAIT_S)
                except concurrent.futures.TimeoutError:
                    self.quick_pool.submit(self._quick_later, anchor, cx, cy)
                except Exception as e:
                    log.info("stand-in photo %s: %s", (anchor.id, cx, cy), e)
                info.setdefault("stages", {})["standIn"] = time.perf_counter() - t1
            elif getattr(self, "speed", 0.0) < QUICK_SYNC_MAX_V:
                # (in the block's own process when it can: just done with its heights; here it took 2-6 s, 2026-10-31)
                info["base"] = self._quick_once(anchor, cx, cy, lambda: (
                    self._quick_elsewhere(anchor, cx, cy, lane) if lane is not None else None)
                    or self.write_quick_base(anchor, cx, cy))
            else:
                self.quick_pool.submit(self._quick_later, anchor, cx, cy)
        info["seconds"] = time.perf_counter() - t0
        info["zoom"] = anchor.hm_zoom
        return info

    def _photo_lawns(self, key, ter, layers, rgb):
        try:
            surfacemod.photo_lawns(self.out_dir, ter, layers, rgb)
        except Exception as e:
            log.info("photo lawns %s: %s", key, e)

    def _quick_later(self, anchor, cx, cy):
        """quick_pool: a block sent without its photo gets its stand-in (write_quick_base) onto its entry and to the mod
        (block_base) unless its own photo came first. Off the block's thread: made there it held each block ~1.5 s,
        and at 500 mph the car waited 12.5 s a run for ground instead of none (2026-10-30)."""
        key = (anchor.id, cx, cy)
        info = self.hm.get(key)
        if info is not None and info.get("base"):
            return                              # (its own photo came while this waited its turn)
        name = self._quick_once(anchor, cx, cy, lambda: self._quick_elsewhere(anchor, cx, cy, LANE_PREFETCH)
                                or self.write_quick_base(anchor, cx, cy))
        conns = [c for c in list(self.connections) if getattr(c, "loop", None) is not None]
        if not name or not conns:
            return
        size = self.layout.block * heightmap.SQUARE

        def apply(left=8):
            info = self.hm.get(key)
            if info is None and left > 0:       # (the block's entry is stored just after it is built)
                conns[0].loop.call_later(0.25, apply, left - 1)
                return
            if info is None or info.get("base"):
                return                          # (gone, or its own photo is here already)
            info["base"] = name
            for c in list(self.connections):
                c.send({"type": "block_base", "anchorId": key[0], "cx": key[1], "cy": key[2],
                        "base": f"{self.vfs_dir}/{name}", "baseSize": size})
        try:
            conns[0].loop.call_soon_threadsafe(apply)
        except RuntimeError:
            pass                                # (the loop has closed)

    def _base_later(self, key, fut, tries=4):
        """A block sent without its photo (not done BASE_WAIT_S after its heights): the photo, once rendered, onto the block's entry (later
        terrain_ready answers) and to the mod (block_base: the block in use takes it now, a ready one at its commit)."""
        try:
            name, photo, _ = fut.result()
        except Exception:
            return
        conns = [c for c in list(self.connections) if getattr(c, "loop", None) is not None]
        if not name or not conns:
            return
        size = self.layout.block * heightmap.SQUARE
        loop = conns[0].loop

        def apply(left=tries):
            info = self.hm.get(key)
            if info is None and left > 0:       # (the block's entry is stored just after it is built)
                loop.call_later(0.25, apply, left - 1)
                return
            if info is not None and (not info.get("base") or info["base"] == self.quick_base_name(key)):
                info["base"] = name
                if photo and info.get("layers") and info.get("ter"):
                    # lawns only where the photo is green (_finish_heightmap), in time for a block not committed
                    # yet (the mod reads its .ter at the commit; one in use keeps its lawns until it is read again);
                    # off the event loop (7.6 s of a Steam Deck's loop in a drive)
                    self.quick_pool.submit(self._photo_lawns, key, info["ter"], info["layers"], photo[0])
            for c in list(self.connections):    # (the mod drops another anchor's)
                c.send({"type": "block_base", "anchorId": key[0], "cx": key[1], "cy": key[2],
                        "base": f"{self.vfs_dir}/{name}", "baseSize": size})
        try:
            loop.call_soon_threadsafe(apply)
        except RuntimeError:
            pass                                # (the loop has closed)

    def base_name(self, key, rev=0):
        ext = terraincache.PHOTO_EXT
        return self.hm_name(key)[:-4] + (f"_base{ext}" if not rev else f"_base_r{rev}{ext}")

    def quick_base_name(self, key):
        return self.hm_name(key)[:-4] + "_base_q" + terraincache.PHOTO_EXT

    def _stand_in(self, anchor, cx, cy):
        """A requested block's stand-in photo made beside its heights (fast): in the prefetch lane's process, else
        here; its file name or None."""
        return self._quick_once(anchor, cx, cy, lambda: self._quick_elsewhere(anchor, cx, cy, LANE_PREFETCH,
                                                                              self.photo_pool)
                                or self.write_quick_base(anchor, cx, cy))

    def _quick_once(self, anchor, cx, cy, make):
        """make() (a block's stand-in photo: its file name or None) once per block: a caller while it is being made
        waits for it (QUICK_SHARE_WAIT_S at most), one after takes the file made. The early stand-in, the provisional
        block's, the finished block's and _quick_later each made their own: up to 5 of one block, two at once, ~100
        stand-ins at 2-9 s for 26 blocks of a 6 min drive (nyc_speed.py --cold, 2026-11-02)."""
        key = (anchor.id, cx, cy)
        with self.quick_lock:
            name = self.quick_made.get(key)
            if name and os.path.exists(os.path.join(self.out_dir, name)):
                return name
            fut = self.quick_running.get(key)
            mine = fut is None
            if mine:
                fut = self.quick_running[key] = concurrent.futures.Future()
        if not mine:
            try:
                return fut.result(timeout=QUICK_SHARE_WAIT_S)
            except Exception:
                return None
        try:
            name = make()
        except BaseException as e:
            fut.set_exception(e)
            raise
        else:
            fut.set_result(name)
            return name
        finally:
            with self.quick_lock:
                self.quick_running.pop(key, None)
                if fut.done() and not fut.exception() and fut.result():
                    if len(self.quick_made) >= QUICK_MADE_KEPT:
                        self.quick_made.clear()
                    self.quick_made[key] = fut.result()

    def _quick_elsewhere(self, anchor, cx, cy, lane, pool=None):
        """write_quick_base in the block's own lane process (idle once its heights are built: blockworker.base_photo
        quick), at FAST_BASE_V and over or when no recent full photo overlaps it (those pasted in need this process;
        at speed every block overlaps the last and the stand-in here took 5-7 s): the file name, or None
        (then here). Its road paint, 0.1 s alone, took 3.7 s here at 185 mph beside the rest (2026-10-31)."""
        if not URGENT_ELSEWHERE or not self.block_pools or self.imagery is None:
            return None
        size = self.layout.block * heightmap.SQUARE
        with self.base_lock:
            recent = list(self.recent_full)
        if getattr(self, "speed", 0.0) < earthbase.FAST_BASE_V and any(
                aid == anchor.id and abs(ocx - cx) < size and abs(ocy - cy) < size for aid, ocx, ocy, _ in recent):
            return None                 # (slower: the overlap pasted in here; faster, every block overlaps the last)
        key = (anchor.id, cx, cy)
        name = self.quick_base_name(key)
        im = self.imagery
        tex = {"chain": [s_.name for s_ in im.chain], "tone": self.tone, "leaf_on": im.leaf_on,
               "season_mode": im.season_mode}
        t0 = time.perf_counter()
        try:
            got = (pool or self.block_pools[lane][0]).submit(
                blockworker.base_photo, anchor.lat, anchor.lon, cx - size / 2 - BASE_SHIFT_M,
                cy - size / 2 - BASE_SHIFT_M, size, QUICK_BASE_PX, tex, False, anchor.zooms[0],
                os.path.join(self.out_dir, name), (self.user_dir, f"{self.vfs_dir}/{name}", base_ao_vfs(self.base_px),
                                                   BASE_AO), out_px=self.base_px, quick=True).result()
        except Exception as e:
            log.info("quick base photo %s in the block process: %s", key, e)
            return None
        if got is None:
            log.info("quick base photo %s: not in the block process (an imagery tile not on disk)", key)
            return None
        log.info("quick base photo %s in the block process in %.2f s", key, time.perf_counter() - t0)
        return name

    def write_quick_base(self, anchor, cx, cy):
        """A block's stand-in photo while its own is drawn: the aerial imagery alone at QUICK_BASE_PX (no roofs, painted
        roads or rock), packed for the game like the photo. A block sent without its photo showed the level's flat
        green base until the photo came (GitHub #548: at ~155 mph the outback flashed between red desert and green).
        The file name, or None."""
        size = self.layout.block * heightmap.SQUARE
        x0, y0 = cx - size / 2 - BASE_SHIFT_M, cy - size / 2 - BASE_SHIFT_M
        key = (anchor.id, cx, cy)
        try:
            # the last full photos' overlap first: the block moves on every 128 m, and ground already shown with its
            # photo turned to the stand-in (or the flat base) with each new block (the user, 2026-10-30: "it flashes
            # already painted/loaded terrain instead of new terrain")
            n = self.base_px
            sc = n / size
            tq = [("start", time.perf_counter())]
            with self.base_lock:
                recent = list(self.recent_full)
            parts = []                  # (the oldest first: the newer is pasted over it)
            have = np.zeros((n, n), bool)
            for aid, ocx, ocy, orgb in recent:
                if aid != anchor.id or orgb.shape[:2] != (n, n):
                    continue
                dx, dy = int(round((cx - ocx) * sc)), int(round((ocy - cy) * sc))
                i0, i1, j0, j1 = max(0, -dy), min(n, n - dy), max(0, -dx), min(n, n - dx)
                if i0 < i1 and j0 < j1:
                    parts.append((i0, i1, j0, j1, dy, dx, orgb))
                    have[i0:i1, j0:j1] = True
            if have.all():
                rgb = np.empty((n, n, 3), np.uint8)
            else:                       # the rest (the new ground at the block's edge): the imagery alone
                tq.append(("overlap", time.perf_counter()))
                with (self.deshadow.no_wait() if self.deshadow is not None else contextlib.nullcontext()):
                    small, used = self.imagery.render(anchor.frame, x0, y0, size, QUICK_BASE_PX, set())
                tq.append(("imagery", time.perf_counter()))
                covered = sum(v for k, v in (used or {}).items() if k != "colormatch")
                if covered < QUICK_MIN_COVER * QUICK_BASE_PX * QUICK_BASE_PX and have.mean() < 0.5:
                    # its imagery not here yet and little of it shown before: the renderer's dirt-brown fill over the
                    # block ("the entire ground now turns brown"); none then, as before
                    return None
                small = np.asarray(small, np.uint8)
                if roadsmod.ROADS_IN_GROUND:    # the carriageways painted in at its own size (cached roads only)
                    try:
                        small = surfacemod.paint_square(small, anchor.frame, x0, y0, size, self.surface,
                                                        self.road_widths, simple=True)
                    except Exception as e:
                        log.debug("quick base photo %s: roads: %s", key, e)
                tq.append(("roads", time.perf_counter()))
                # (at the photo's own size: the engine refuses another, "dont have required size of 4096-4096")
                rgb = np.array(surfacemod.enlarge(small, n), np.uint8)
                tq.append(("enlarge", time.perf_counter()))
            # plain slices, not masks: masked copies took 0.9 s of a 4096² stand-in's 1.4 s, and at 230 mph (a block
            # every 1.2 s) the stand-ins fell behind and the flat base showed between them (the user, 2026-10-30)
            for i0, i1, j0, j1, dy, dx, orgb in parts:
                rgb[i0:i1, j0:j1] = orgb[i0 + dy:i1 + dy, j0 + dx:j1 + dx]
            tq.append(("paste", time.perf_counter()))
            name = self.quick_base_name(key)
            path = os.path.join(self.out_dir, name)
            tmp = staging.tmp_for(path)
            with open(tmp, "wb") as f:
                f.write(terraincache.stored_photo(rgb))
            staging.replace(tmp, path)
            tq.append(("png", time.perf_counter()))
            args = (self.user_dir, f"{self.vfs_dir}/{name}", base_ao_vfs(self.base_px), rgb, BASE_AO)
            pool = self.block_pools[LANE_PREFETCH][0] if (URGENT_ELSEWHERE and self.block_pools and
                                                          not bc7.gpu_ok()) else None
            if pool is not None:        # (the CPU's BC7 encode on a block process's core: ~1 s here beside the rest)
                pool.submit(blockworker.base_entry, *args).result()
            else:
                terraincache.base_entry(*args)
            tq.append(("cache", time.perf_counter()))
            if tq[-1][1] - tq[0][1] > 1.0:
                log.info("quick base photo %s in %.2f s: %s (%d px)", key, tq[-1][1] - tq[0][1],
                         " ".join(f"{k} {b - a:.2f}" for (_, a), (k, b) in zip(tq, tq[1:])), n)
            return name
        except Exception as e:      # the flat base then, as before
            log.info("quick base photo %s: %s", key, e)
            return None

    def write_base(self, anchor, cx, cy, pool=None, rev=0, photo=None, quick=False):
        """Detail mode: the block footprint's aerial photo (base_px² PNG, row 0 = north, the same imagery
        chain and corrections as the far tiles), the drawn block's base colour. The game would pack it into
        BC7 on its main thread (1.5 s for a new 1024² photo); terraincache writes that packed texture into
        the game's cache first (~0.4 s here), so the game takes it as it is. Returns the file name, or None
        when no imagery could be rendered (the block keeps the level's neutral base).
        quick (a teleport's provisional first block): without the roofs' mask and the painted roads when the place's
        buildings and roads are not on disk yet (their downloads, and the road widths measured from them, were 20 s
        of a fresh place's 22 s photo on a Deck, before the car could be placed; 2026-10-23); rendered again in full
        once the graded block is in (base_refresh), as for a lazy imagery source."""
        size = self.layout.block * heightmap.SQUARE
        x0, y0 = cx - size / 2 - BASE_SHIFT_M, cy - size / 2 - BASE_SHIFT_M
        quick = quick and not self._photo_data_cached(anchor, x0, y0, size)
        t0 = time.perf_counter()
        try:
            pending = set()
            with (self.deshadow.no_wait() if self.deshadow is not None else contextlib.nullcontext()):
                rgb, used = self.imagery.render(anchor.frame, x0, y0, size, BASE_RENDER_PX, pending)
            log.debug("base photo %s r%d: %s", (anchor.id, cx, cy), rev,
                      {k: round(v / BASE_RENDER_PX ** 2, 3) for k, v in used.items()})
        except Exception as e:      # an imagery failure must not cost the terrain
            log.info("base photo %s: %s", (anchor.id, cx, cy), e)
            return None
        name = self.base_name((anchor.id, cx, cy), rev)
        path = os.path.join(self.out_dir, name)
        rgb = np.asarray(rgb, np.uint8)
        stamps = [("imagery", time.perf_counter())]
        if self.buildings and self.mapdata is not None and not quick:   # no roofs beside the 3D buildings
            try:
                mask = buildingsmod.photo_mask(anchor.frame, x0, y0, size, BASE_RENDER_PX, self.mapdata)
                if mask is not None:
                    rgb = buildingsmod.fill_photo(rgb, mask, size, BASE_RENDER_PX)
            except Exception as e:      # the photo stays as it is
                log.info("base photo %s: building mask: %s", (anchor.id, cx, cy), e)
        if roadsmod.ROADS_IN_GROUND and not quick:      # the photo's own pavement beside the drawn roads
            try:
                rgb = surfacemod.unghost_square(rgb, anchor.frame, x0, y0, size, self.surface, self.road_widths)
            except Exception as e:      # the photo stays as it is
                log.info("base photo %s: road ghosts: %s", (anchor.id, cx, cy), e)
        rgb = surfacemod.enlarge(rgb, self.base_px)
        stamps.append(("buildings", time.perf_counter()))
        if roadsmod.ROADS_IN_GROUND and not quick:      # the carriageways in the ground (surface.photo_roads)
            try:
                rgb = surfacemod.paint_square(rgb, anchor.frame, x0, y0, size, self.surface, self.road_widths)
            except Exception as e:      # the photo stays as it is
                log.info("base photo %s: roads: %s", (anchor.id, cx, cy), e)
        stamps.append(("roads", time.perf_counter()))
        try:            # its cliffs as rock, like the far tiles' (rockface.py), from a 16 m grid of the ground
            n = 65
            xs = x0 + np.arange(n) * (size / (n - 1))
            ys = y0 + np.arange(n) * (size / (n - 1))
            zoom = anchor.zooms[0]
            _, U = heightmap.surface(anchor.frame, lambda la, lo: self.dem.sample(la, lo, zoom), xs, ys)
            rgb = rockfacemod.rock(rgb, anchor.frame, x0, y0, size, U, x0, y0, size / (n - 1))
        except Exception as e:      # the photo stays as it is
            log.debug("base photo %s: rock faces: %s", (anchor.id, cx, cy), e)
        # a lazy source (NAIP) not fetched yet for all of it: rendered again once it is (base_refresh)
        # ... and the analyses (de-shadow, colour match) it was rendered without (never waited for)
        key = (anchor.id, cx, cy)
        self._base_kept(anchor, cx, cy, pending)
        if quick:
            self.base_stale.add(key)
        if photo is not None:
            photo.append(rgb)
        with self.base_lock:    # (the stand-ins of the blocks after it take its overlap: write_quick_base)
            self.recent_full.append((anchor.id, cx, cy, rgb))
        t1 = time.perf_counter()
        tmp = staging.tmp_for(path)
        # (terraincache.stored_photo: the game reads the packed cache entry, this file only if that is stale)
        with open(tmp, "wb") as f:
            f.write(terraincache.stored_photo(rgb))
        staging.replace(tmp, path)
        t2 = time.perf_counter()
        try:            # after the photo: the cache entry must be the newer file
            args = (self.user_dir, f"{self.vfs_dir}/{name}", base_ao_vfs(self.base_px), rgb, BASE_AO)
            if pool is not None and not bc7.gpu_ok():   # the CPU's BC7 encode on the block process's core
                pool.submit(blockworker.base_entry, *args).result()
            else:
                terraincache.base_entry(*args)
        except OSError as e:
            log.info("terrain cache for %s: %s", name, e)
        stamps.append(("rock", t1))
        parts = " ".join(f"{k} {b - a:.2f}" for (_, a), (k, b) in zip([("", t0)] + stamps, stamps))
        log.log(logging.INFO if time.perf_counter() - t0 > SLOW_BASE_S else logging.DEBUG,
                "base photo %s r%d in %.2f s: %s, png %.2f, cache %.2f", key, rev, time.perf_counter() - t0, parts,
                t2 - t1, time.perf_counter() - t2)
        return name

    def _photo_data_cached(self, anchor, x0, y0, size):
        """Whether the buildings and roads of every map cell under the square are on disk (never fetches)."""
        if self.mapdata is None:
            return True
        la, lo, _ = anchor.frame.enu_to_geodetic(np.array([x0, x0 + size]), np.array([y0, y0 + size]), 0.0)
        keys = mapcells.cells_for_bbox(float(np.min(la)), float(np.min(lo)), float(np.max(la)), float(np.max(lo)))
        return all(self.mapdata.has(k, "buildings") and self.mapdata.has(k, "roads") for k in keys)

    def base_ready(self, anchor, cx, cy):
        """Rendering the block's base photo again would not change it: every lazy imagery tile is here and
        the regions it was rendered without (base_pending) are done."""
        size = self.layout.block * heightmap.SQUARE
        if not self.imagery.lazy_ready(anchor.frame, cx - size / 2 - BASE_SHIFT_M, cy - size / 2 - BASE_SHIFT_M, size,
                                       BASE_RENDER_PX):
            return False
        return all(self._region_status(rk) is not None for rk in self.base_pending.get((anchor.id, cx, cy), ()))

    def terrain_cached(self, key):
        return key in self.hm or key in self.hm_inflight

    async def terrain(self, anchor, cx, cy, progress=None, lane=LANE_REQUEST):
        """Info of the (anchorId, cx, cy) terrain, generating it once (deduplicated). lane: the block
        process that builds it (LANE_PREFETCH for the prefetch)."""
        key = (anchor.id, cx, cy)
        if key in self.hm:
            self.hm.move_to_end(key)
            return self.hm[key]
        fut = self.hm_inflight.get(key)
        if fut is None:
            loop = asyncio.get_running_loop()
            pool = self.hm_req_pool if lane == LANE_REQUEST and progress is None and self.block_pools else self.hm_pool
            fut = self.hm_inflight[key] = loop.run_in_executor(
                pool, self.make_heightmap, anchor, cx, cy, progress, lane, time.perf_counter())
            fut.add_done_callback(lambda f: self._hm_done(key, f))
        return await asyncio.shield(fut)

    def _hm_done(self, key, fut):
        self.hm_inflight.pop(key, None)
        if fut.cancelled() or fut.exception() is not None:
            return
        info = fut.result()
        log.info("terrain %s z%d in %.2f s (%s): zBase %.0f maxHeight %.0f (Q 1/%d)", key, info["zoom"],
                 info["seconds"], " ".join(f"{k} {v:.2f}" for k, v in info.get("stages", {}).items()),
                 info["zBase"], info["maxHeight"], round(1 / info["q"]))
        self.hm[key] = info
        # (its roads and land cover painted once their cells are here: Connection.roads_refresh. A first visit's blocks
        # came before the land cover and kept bare dirt for good, no grass, Harz 2026-10-27, GitHub #316)
        if info.get("roadsComplete") is False or info.get("surfaceComplete") is False:
            if key not in self.roads_rebuild:
                log.info("block %s built before its map cells were here (roads %s, surface %s): painted again later",
                         key, info.get("roadsComplete"), info.get("surfaceComplete"))
            self.roads_rebuild.add(key)
        else:
            self.roads_rebuild.discard(key)
        self.evict_heightmaps(fresh=key)

    def evict_heightmaps(self, fresh=None):
        """Keep at most HM_CACHE entries, never deleting the current terrain, the
        last two committed ones, the last two handed to the mod, or `fresh` (the terrain
        just built: a request about to be answered, or the prefetch the mod asks for next)."""
        keep = set(self.hm_committed) | set(self.hm_sent)
        if fresh is not None:
            keep.add(fresh)
        # The first terrain of an anchor no connection has committed yet: anchor_ready
        # handed it out and the rebase may commit it only after more terrain-follow requests
        # in the old frame. (Once committed it is in hm_committed while it matters; keeping
        # every anchor's first terrain made the protected set larger than HM_CACHE after a
        # rebase, so each new prefetch was deleted as soon as it was built, and rebuilt on
        # every pose.)
        newest = max((c.active for c in self.connections if c.active is not None), default=None)
        keep |= {(aid, *a.first_terrain) for aid, a in sorted(self.anchors.items())[-2:]
                 if a.first_terrain is not None and (newest is None or aid > newest)}
        for key in [k for k in self.hm if k not in keep][:max(0, len(self.hm) - HM_CACHE)]:
            info = self.hm.pop(key)
            self._remove(os.path.join(self.out_dir, self.hm_name(key)))
            self.base_stale.discard(key)
            self.base_pending.pop(key, None)
            # its photos (the base, a refreshed one, the stand-in) with their packed copies in the game's terrain
            # texture cache
            for name in {self.base_name(key), info.get("base") or self.base_name(key), self.quick_base_name(key),
                         *info.get("old_bases", ())}:
                self._remove_base(name)
            if info.get("ter"):
                self._remove(os.path.join(self.out_dir, info["ter"]))
            if self.surface is not None:        # its surface masks (only read at import)
                for name in surfacemod.mask_names(self.hm_name(key)[:-4]):
                    self._remove(os.path.join(self.out_dir, name))
