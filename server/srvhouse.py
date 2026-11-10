"""World server: sessions, cache files and pruning (split out of earthd.py's Server).
"""
import os
import asyncio  # noqa: E402
import glob  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from runtimeprofiler import ThreadPoolExecutor  # noqa: E402
import blockworker  # noqa: E402
import terraincache  # noqa: E402
from earthbase import (
    ANCHOR_TILE_FILES,
    BASE_SIZES,
    DELETE_DRAIN_S,
    GEO_FILES_GRACE_S,
    GEO_LIST_BATCH_S,
    GEO_TRACKERS,
    LEASE_GRACE_S,
    SESSION_HEARTBEAT_S,
    SESSION_MARKER,
    SESSION_STALE_S,
    TRASH_DIR,
    _app_modules,
    base_ao_vfs,
    game_running,
    log,
)


class _HousekeepingMixin:
    """Server's sessions, cache files and pruning (split out of earthd.py; Server inherits it)."""

    def _empty_trash(self):
        """What earlier runs left in the cache's trash (_rmtree_outside): a server ended with the game while it was
        deleting a session moved there (the game closing ends the server's processes) left the rest of it for good, GBs
        of tiles that no later run looked at (GitHub #599: 9 GB of old sessions). Deleted in a background thread (a
        whole session is thousands of files; the start does not wait). An entry whose server still runs (its pid in its
        name, xproc.alive) is that server's own delete under way and is left to it (a rename keeps the folder's old
        time, so its age says nothing). Returns the thread, or None (nothing there)."""
        cache = getattr(self, "cache_dir", None)
        trash = os.path.join(cache, TRASH_DIR) if cache else None
        try:
            names = os.listdir(trash) if trash else []
        except OSError:
            return None
        import xproc
        old = []
        for n in names:
            parts = n.rsplit("_", 2)
            pid = int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None
            if pid is not None and pid != os.getpid() and xproc.alive(pid):
                continue
            old.append(os.path.join(trash, n))
        if not old:
            return None
        log.info("trash: deleting %d folders earlier runs left in %s", len(old), trash)

        def run():
            for p in old:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    self._remove(p)
        t = threading.Thread(target=run, name="trash", daemon=True)
        t.start()
        return t

    def _scan_sessions(self):
        """Session folders left by earlier runs (only ones we marked): one whose marker is untouched
        for SESSION_STALE_S goes now, with the engine's .cdae caches of its shapes; a younger one is
        kept as an orphan until a game claims it or sweep_sessions deletes it. With no game running (the usual start:
        the launcher starts the server first) every earlier session goes now: no game can claim it, and no file
        watcher sees it go, where the sweep's later move of four sessions out of the user folder froze the game for
        2 x 8.6 s ("FileWatchers", 2026-11-02). Then the trash earlier runs left and these moves (_empty_trash)."""
        wall, now = time.time(), time.monotonic()
        no_game = not game_running()
        for d in glob.glob(os.path.join(self.root, "*", SESSION_MARKER)):
            sid = os.path.basename(os.path.dirname(d))
            if sid == self.sid:     # a restart in the same second with the same suffix: ours now, never released
                continue
            try:
                age = wall - os.path.getmtime(d)
            except OSError:
                continue
            if age > SESSION_STALE_S or no_game:
                self._rmtree_session(sid, later=True)
            else:
                self.orphans[sid] = now
                log.info("session %s of an earlier run (%.0f min old) kept until the game says whether it still "
                         "shows its tiles", sid, age / 60)
        if no_game:
            self._wipe_leaked_packs_once()
        self.trash_thread = self._empty_trash()

    def _base_cache_files(self, sid):
        """Detail mode's packed copies of session sid's base photos in the game's terrain texture cache (terraincache.py):
        they go with it (blocks evicted during a run took theirs, prune; whatever a run left behind stayed for good,
        ~1.4 MB a block, 3.9 GB of them by 2026-10-18)."""
        out = []
        for path in glob.glob(os.path.join(self.root, sid, "*_base*.png")) + \
                glob.glob(os.path.join(self.root, sid, "*_base*.jpg")):       # (.png: before 2026-11-04)
            for px in BASE_SIZES:           # (whichever size it was made at)
                key = terraincache.base_key(f"/temp/beamngEarth/{sid}/{os.path.basename(path)}", base_ao_vfs(px))
                p = os.path.join(terraincache.cache_dir(self.user_dir), key + ".dds")
                if os.path.exists(p):
                    out.append(p)
        return out

    def _remove_base(self, name):
        """A base photo name in out_dir gone with its packed copies in the game's terrain texture cache (whichever size it
        was made at). A replaced stand-in or earlier refresh lost only its photo, its 5.6-22 MB pack stayed for good:
        one a block at speed, 165 GB in the cache of a test PC by 2026-11-03."""
        self._remove(os.path.join(self.out_dir, name))
        for px in BASE_SIZES:
            self._remove(os.path.join(terraincache.cache_dir(self.user_dir),
                                      terraincache.base_key(f"{self.vfs_dir}/{name}", base_ao_vfs(px)) + ".dds"))

    def _wipe_leaked_packs_once(self):
        """Once (no game running): the game's terrain texture cache moved to the trash when it holds over
        PACK_WIPE_MIN_BYTES, the base photo packs earlier builds leaked (_remove_base). It is a cache: the game packs
        what it needs again (a few seconds at the next level load, stock maps' materials at their next use)."""
        mark = os.path.join(self.cache_dir, self.PACK_WIPE_MARK)
        if os.path.exists(mark):
            return
        d = terraincache.cache_dir(self.user_dir)
        total = 0
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        total += e.stat().st_size
                    except OSError:
                        pass
                    if total > self.PACK_WIPE_MIN_BYTES:
                        break
        except OSError:
            pass
        if total > self.PACK_WIPE_MIN_BYTES:
            log.info("terrain texture cache over %.0f GB (leaked base photo packs): moved out, deleted in the background",
                     self.PACK_WIPE_MIN_BYTES / 1024 ** 3)
            self._rmtree_outside(d, later=True)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            open(mark, "w").close()
        except OSError:
            pass

    def _rmtree_session(self, sid, later=False):
        """Session sid's folders (and its base photos' terrain cache copies) removed at once: moved out of the user
        folder and deleted (later: only moved, the trash thread deletes them). For when no game watches the folder
        (the start); a running game's sessions go file by file (_session_paths, queue_delete)."""
        for path in self._base_cache_files(sid):
            self._remove(path)
        for d in (os.path.join(self.root, sid), os.path.join(self.cdae_root, sid.lower())):
            self._rmtree_outside(d, later=later)

    def _session_paths(self, sid):
        """Session sid's files for the rate-limited delete queue, in order: its base photos' terrain cache copies, the
        session's and its .cdae cache's files, the marker, the folders. (Blocking: a whole session is ~11k files.)"""
        d = os.path.join(self.root, sid)
        cd = os.path.join(self.cdae_root, sid.lower())
        marker = os.path.join(d, SESSION_MARKER)
        files = sorted(p for p in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(cd, "*")) if p != marker)
        return self._base_cache_files(sid) + files + [marker, d, cd]

    def _rmtree_outside(self, d, later=False):
        """Remove folder d of the game's user folder: moved out of it first (into the cache's trash, one rename) and
        deleted there (later: left there for _empty_trash). Deleted in place, a whole session's thousands of files were
        as many notifications for the game's file watcher, on its main thread: a 4 s frame at Utrecht 2026-10-30
        ("FileWatchers 1501.7"); the move alone of four sessions froze it too (2 x 8.6 s, 2026-11-02), so a running
        game's sessions go file by file instead (_session_paths). In place when the move cannot be made (the cache on
        another drive)."""
        if not os.path.isdir(d):
            return
        cache = getattr(self, "cache_dir", None)
        if cache:
            trash = os.path.join(cache, TRASH_DIR, f"{os.path.basename(d)}_{os.getpid()}_{time.monotonic_ns()}")
            try:
                os.makedirs(os.path.dirname(trash), exist_ok=True)
                os.replace(d, trash)
                if later:
                    return
                d = trash
            except OSError:
                pass
        shutil.rmtree(d, ignore_errors=True)

    def _touch(self, sid):
        try:
            os.utime(os.path.join(self.root, sid, SESSION_MARKER))
        except OSError:
            pass

    def claimed_sessions(self):
        """Earlier runs' sessions still owned by connected or reconnecting games."""
        out = self.stream_leases.sessions()
        for c in self.connections:
            if getattr(c, "lease", None) is None:
                out |= c.sessions
        return out

    def check_user_dir(self, conn, game_path):
        """The game's own user folder (hello.userPath) against the one this server writes into: a game whose user
        folder is elsewhere (moved in BeamNG's launcher, its own -userpath, a folder name the launcher misread) never
        finds our tiles: the car on a flat plane, "loading terrain" for good (GitHub #770). Told to the player, and
        saved for the launcher's next start (apppaths.set_user_dir)."""
        if not isinstance(game_path, str) or not game_path.strip():
            return
        game = os.path.normpath(game_path.strip())
        if not os.path.isdir(game):     # (a path we cannot see: Proton's C:/users/steamuser/..., a Flatpak sandbox's
            return                      # home: not ours to judge)
        try:
            apppaths, _ = _app_modules()
            same = apppaths.same_dir(game, self.user_dir)
        except Exception:           # (no apppaths: compared as text)
            apppaths, same = None, os.path.normcase(game) == os.path.normcase(os.path.normpath(self.user_dir))
        if same:
            return
        log.error("the game's user folder is %s, but this server writes into %s: the game cannot find the world's "
                  "files (GitHub #770); the next start uses the game's folder", game, self.user_dir)
        if apppaths is not None:
            try:
                apppaths.set_user_dir(game)
            except OSError as e:
                log.warning("could not save the game's user folder: %s", e)
        conn.send({"type": "user_dir_mismatch", "game": game, "server": self.user_dir})

    def claim_sessions(self, conn, sessions):
        """hello / pose `sessions` of a connection: the session folders whose files the game's objects
        still use. Earlier runs' folders among them are adopted (kept, their markers touched); one this
        game named before and names no longer is released (release_session) unless another game
        claims it. Our own session and unknown names are ignored."""
        if isinstance(sessions, dict):      # an empty Lua table may encode as {}
            sessions = list(sessions.values())
        if not isinstance(sessions, list):
            return
        mine = {sid for sid in sessions if isinstance(sid, str) and sid in self.orphans}
        for sid in sorted(mine - conn.sessions):
            log.info("adopting session %s: the game still shows its tiles, its files stay until it lets go", sid)
            self._touch(sid)
        released = conn.sessions - mine
        conn.sessions = mine
        others = self.claimed_sessions()
        for sid in sorted(released - others):
            self.release_session(sid)

    def drop_claims(self, conn):
        """A connection closed: the sessions it claimed are orphans again (the game may come back,
        perhaps to yet another server), their grace starting now. Called once the connection left
        self.connections."""
        if getattr(conn, "lease", None) is not None:
            return                      # TCP EOF does not mean that the game's objects disappeared
        others = self.claimed_sessions()
        now = time.monotonic()
        for sid in conn.sessions - others:
            if sid in self.orphans:
                self.orphans[sid] = now
        conn.sessions = set()

    def release_session(self, sid):
        """An adopted session the game no longer uses: its files go through the rate-limited delete
        queue (the game loaded them, P9), the marker and the folders last."""
        if self.orphans.pop(sid, None) is None:
            return
        d = os.path.join(self.root, sid)
        cd = os.path.join(self.cdae_root, sid.lower())
        files = sorted(glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(cd, "*")))
        log.info("session %s released by the game: deleting its %d files", sid, len(files))
        self.queue_delete(files + [os.path.join(d, SESSION_MARKER), d, cd])

    def sweep_sessions(self, remove=True):
        """Every SESSION_SWEEP_S (main_async, via sweep_sessions_async): touch our own marker and the
        claimed sessions' every SESSION_HEARTBEAT_S (a later server sees them in use), and delete the
        orphans nobody claimed for orphan_s whose markers are as old (one a live server touches is in
        use). Returns the deleted session ids; remove=False leaves their folders to the caller."""
        now, wall = time.monotonic(), time.time()
        claimed = self.claimed_sessions()
        if now - self.heartbeat_at >= SESSION_HEARTBEAT_S:
            self.heartbeat_at = now
            for sid in [self.sid, *sorted(claimed)]:
                self._touch(sid)
        gone = []
        for sid, since in sorted(self.orphans.items()):
            if sid in claimed or now - since < self.orphan_s:
                continue
            try:
                idle = wall - os.path.getmtime(os.path.join(self.root, sid, SESSION_MARKER))
            except OSError:
                idle = math.inf
            if idle < self.orphan_s:
                continue
            del self.orphans[sid]
            log.info("session %s of an earlier run: no game claimed it, deleting it file by file", sid)
            if remove:
                self.queue_delete(self._session_paths(sid))
            gone.append(sid)
        return gone

    async def sweep_sessions_async(self):
        """sweep_sessions on the event loop (its state is the loop's): each session's files listed in a worker thread
        (a whole session is ~11k files, the streaming must not wait) and deleted through the rate-limited queue, as a
        released session's are: a game is running, and its file watcher took a whole session moved out of the user
        folder at once as an 8.6 s frame (2026-11-02)."""
        self.expire_leases()
        gone = self.sweep_sessions(remove=False)
        for sid in gone:
            self.queue_delete(await asyncio.to_thread(self._session_paths, sid))
        return gone

    def expire_leases(self, now=None):
        """Release the stream leases of games gone longer than LEASE_GRACE_S (event loop, sweep_sessions_async):
        their pooled slots go back to the pool, their anchors' files and their sessions' folders follow the usual
        rules (the sessions' grace starts now, as drop_claims's)."""
        now = time.monotonic() if now is None else now
        gone = self.stream_leases.expire(now, LEASE_GRACE_S)
        if not gone:
            return 0
        others = self.claimed_sessions()
        for lease in gone:
            log.info("stream lease %s: its game gone %.0f s: released (%d slots, %d anchors, %d sessions)",
                     lease.token[:8], now - lease.detached_at, len(lease.slots), len(lease.anchors), len(lease.sessions))
            self.slots_let_go(set(lease.slots))
            for sid in lease.sessions - others:
                if sid in self.orphans:
                    self.orphans[sid] = now
        return len(gone)

    @staticmethod
    def _remove(path):
        for p in (path, path + ".tmp"):
            try:
                os.remove(p)
            except OSError:
                pass

    def delete_gap(self):
        """Seconds until the next queued delete: delete_interval, or shorter for a long queue
        (up to delete_max_rate files/s, see DELETE_DRAIN_S)."""
        rate = min(self.delete_max_rate, len(self.delete_queue) / DELETE_DRAIN_S)
        return min(self.delete_interval, 1.0 / rate) if rate > 0 else self.delete_interval

    def _delete_one(self):
        path, ck = self.delete_queue.popleft()
        # an evicted tile built again since (same file name) keeps its new file; ck callable: kept while it says so
        if callable(ck):
            if ck():
                return
            self._remove(path)
        elif ck is None or (ck not in self.tiles and ck not in self.tile_inflight):
            if os.path.isdir(path):     # a released session's folder, queued after its files
                shutil.rmtree(path, ignore_errors=True)
            else:
                self._remove(path)

    def geo_held(self, kind, k):
        """Does some connection's tracker of kind (GEO_TRACKERS) have geotile k (gx, gy) sent or building?"""
        attr = GEO_TRACKERS.get(kind)
        for c in list(self.connections):
            t = getattr(c, attr, None) if attr else None
            if t is not None and getattr(t, "geo", False) and (k in t.sent or k in t.building):
                return True
        return False

    def geo_files_many(self, wanted):
        """geo_files of several (kind, key) at once: the session folder listed once. [files] in their order."""
        try:
            names = os.listdir(self.out_dir)
        except OSError:
            return [[] for _ in wanted]
        return [self.geo_files(kind, key, names) for kind, key in wanted]

    def geo_files(self, kind, key, names=None):
        """The session files of geotile key (geotiles.key) of kind: a road geotile's content (rg_), every anchor's skin
        (rs_) and exact variants (rx_); a far geotile's json, shape and pieces (blockworker.FAR_GEO prefixes)."""
        if kind == "roads":
            pat = re.compile(rf"(rg_{re.escape(key)}([._].*)?|r[sx]_\d+_{re.escape(key)}([._].*)?)")
            stems = ("rg_", "rs_", "rx_")
        else:
            pre = blockworker.FAR_GEO.get(kind)
            if pre is None:
                return []
            pat = re.compile(rf"{pre}_{re.escape(key)}([._].*)?")
            stems = (f"{pre}_",)
        # (one listing, filtered by the key first: glob listed the session folder once per stem in Python, ~52 ms a
        # call over 16k files, 89 s of the event loop in a 10 min first visit, 2026-11-04; this ~10 ms, off the loop)
        if names is None:
            try:
                names = os.listdir(self.out_dir)
            except OSError:
                return []
        return sorted(os.path.join(self.out_dir, n) for n in names
                      if key in n and n.startswith(stems) and pat.fullmatch(n))

    def geo_dropped(self, kind, k, key, grace=None):
        """A connection dropped geotile k (gx, gy; key its geotiles.key) of kind: its session files are deleted after
        GEO_FILES_GRACE_S unless some connection has it (sent or building) by then or when each file's turn comes
        (geo_held: one built again writes the same names). Event loop (the trackers' on_pose)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def run():
            if self.geo_held(kind, k):
                return
            # (the tiles whose grace ran out together share one listing of the session folder, on a thread of its
            # own: 32 road tiles expiring at once listed ~27k entries 32 times; GitHub #758 measured the old
            # per-tile globs on the event loop at 8.5 s for 32 tiles)
            batch = self.__dict__.setdefault("geo_expired", [])
            batch.append((kind, k, key))
            if len(batch) > 1:
                return
            loop.call_later(GEO_LIST_BATCH_S, flush)

        def flush():
            items, self.geo_expired = self.geo_expired, []
            items = [it for it in items if not self.geo_held(it[0], it[1])]
            if not items:
                return
            if self.geo_list_pool is None:
                self.geo_list_pool = ThreadPoolExecutor(1, thread_name_prefix="geo-files")

            def listed(fut):
                try:
                    found = fut.result()
                except Exception as e:
                    log.debug("geotile files: %s", e)
                    return
                for (kd, kk, _key), files in zip(items, found):
                    if files and not self.geo_held(kd, kk):
                        self.queue_delete(files, (lambda kd=kd, kk=kk: self.geo_held(kd, kk)))
            loop.run_in_executor(self.geo_list_pool, self.geo_files_many, [(kd, ky) for kd, _kk, ky in items]
                                 ).add_done_callback(listed)
        loop.call_later(GEO_FILES_GRACE_S if grace is None else grace, run)

    def queue_delete(self, paths, ck=None):
        """Delete files gradually (one at a time, delete_gap() apart) so the game's file watcher
        never sees a burst. ck: the far tile (anchorId, key) the files belong to, if it may be
        built again meanwhile (LRU eviction): then they are left alone. Needs a running event
        loop; falls back to deleting at once."""
        self.delete_queue.extend((p, ck) for p in paths)
        if self._deleting:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            while self.delete_queue:
                self._delete_one()
            return
        self._deleting = True
        due = [loop.time() + self.delete_gap()]

        def step():
            if self.delete_queue:
                self._delete_one()
            if self.delete_queue:
                # a gap after the last one was due, not after it ran: Windows' loop wakes up to ~16 ms late at every
                # timer (a 25 ms gap took 33, 40 files/s became 30); behind by more than a gap (a stall): afresh,
                # no burst to catch up
                now = loop.time()
                due[0] += self.delete_gap()
                if due[0] < now:
                    due[0] = now + self.delete_gap()
                loop.call_later(due[0] - now, step)
            else:
                self._deleting = False
        loop.call_later(due[0] - loop.time(), step)

    def schedule_prune(self, keep_from):
        """prune(keep_from) after prune_delay seconds (once per keep_from)."""
        if keep_from in self._prune_scheduled or not any(a < keep_from for a in self.anchors):
            return
        self._prune_scheduled.add(keep_from)

        loop = asyncio.get_running_loop()

        def run():
            # the mod may still show a previous anchor's tiles long after far_done (it applies
            # far_removes only once every tile received before them is shown; a rebase reload
            # can take minutes at speed): a file deleted under a live TSStatic gets reloaded
            if any(a < keep_from for a in self.held_anchors()):
                loop.call_later(self.prune_recheck, run)
                return
            self._prune_scheduled.discard(keep_from)
            self.prune(keep_from)
        loop.call_later(self.prune_delay, run)

    def prune(self, keep_from):
        """Forget anchors older than keep_from and delete their files."""
        busy = {k[0] for k in self.tile_inflight}
        retry = False
        # (the session folders listed once: a glob per anchor and file kind, ~10 each, scanned folders of thousands of
        # files on the event loop, 2 s stalls when the 3D map's view moved the origin often, 2026-11-07)
        # (and indexed once by "<kind>_<anchor id>_": a scan of every name for each anchor and kind was ~10 x anchors x
        # thousands of names, still 2-3 s on the loop after many origin moves, 2026-11-07; every kind is one word)
        listing = {}

        def names(d):
            if d not in listing:
                try:
                    found = os.listdir(d)
                except OSError:
                    found = []
                idx = {}
                for n in found:
                    parts = n.split("_", 2)
                    if len(parts) == 3:
                        idx.setdefault((parts[0], parts[1]), []).append(n)
                listing[d] = (set(found), idx)
            return listing[d]

        def starting(d, kind, aid):
            return [os.path.join(d, n) for n in names(d)[1].get((kind, str(aid)), ())]

        def listed(p):
            return os.path.basename(p) in names(os.path.dirname(p))[0]
        for aid in [a for a in self.anchors if a < keep_from]:
            if aid in busy:
                # A running worker still owns its files and pooled shape slot.
                # Its asyncio Future must not be cancelled to hurry pruning.
                retry = True
                continue
            del self.anchors[aid]
            self.horizon_grids.pop(aid, None)
            self.horizon_failed.pop(aid, None)
            for ck in [k for k in self.tiles if k[0] == aid]:
                del self.tiles[ck]
            for ck in [k for k in self.matslots.holders() if k[0] == aid]:
                self.matslots.release(ck)
            for ck in [ck for ck in list(self.degraded) if ck[0] == aid]:     # list(): workers add meanwhile
                self.degraded.discard(ck)
            for ck in [k for k in self.tex_rev if k[0] == aid]:
                del self.tex_rev[ck]
            self.retex_todo = {ck for ck in self.retex_todo if ck[0] != aid}
            for rk in list(self.cm_waiting):    # (a teleport: regions far from the new anchor never finish)
                self.cm_waiting[rk] = {ck for ck in self.cm_waiting[rk] if ck[0] != aid}
                if not self.cm_waiting[rk]:
                    del self.cm_waiting[rk]
            for key in [k for k in self.hm if k[0] == aid]:
                if key not in self.hm_committed and key not in self.hm_sent:
                    del self.hm[key]
            keep = {self.hm_name(k) for k in self.hm} | {i["ter"] for i in self.hm.values() if i.get("ter")}
            doomed = []
            for p in starting(self.out_dir, "far", aid):
                name = os.path.basename(p)
                doomed.append(p)
                if name.endswith(".dae"):
                    doomed.append(os.path.join(self.cdae_dir, os.path.splitext(name)[0].lower() + ".cdae"))
            doomed += starting(self.cdae_dir, "far", aid)  # caches whose source went earlier
            doomed += [p for p in starting(self.out_dir, "hm", aid) if os.path.basename(p) not in keep]
            # the road / building / tree tiles too (2026-10-14: a night of teleports left 13k files in the
            # session folder, and the game's file watchers cost 4.5 ms a frame)
            for stem in ANCHOR_TILE_FILES:
                doomed += starting(self.out_dir, stem, aid)
            self.queue_delete(p for p in dict.fromkeys(doomed) if listed(p))     # (there when listed: no stat each)
        if retry:
            self.schedule_prune(keep_from)
