"""World server, base layer (split out of earthd.py): its settings, the gates, follow rule, anchor and tile records,
and the free functions the server's classes share. DETAIL_RING_M, AHEAD_ON, ROUTE_AHEAD_ON, ROUTE_AHEAD_MAX_M and
FAST_BASE_V are set by main() from the command line: read them as earthbase.<name>, never as an imported copy.
"""
import os
# Heavy numpy work must not compete with the game's physics threads (§4.5): one BLAS/OpenMP
# thread per worker. Set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
import numpy as np  # noqa: E402
import cache as cachemod  # noqa: E402
import areas as areasmod  # noqa: E402
import gasstations as gasstationsmod  # noqa: E402
import roads as roadsmod  # noqa: E402
import perfstats  # noqa: E402
import roadwidth as roadwidthmod  # noqa: E402
import revisit as revisitmod  # noqa: E402
import mapdata as mapdatamod  # noqa: E402
import dem as demmod  # noqa: E402
import peerwork as peerworkmod  # noqa: E402
import farfield  # noqa: E402
import geodesy  # noqa: E402
import seacarve as seacarvemod  # noqa: E402
import vegetation as vegetationmod
import heightmap  # noqa: E402
import netfetch  # noqa: E402


PROTOCOL = 4
SEND_BUFFER_MAX = 32 * 1024 * 1024  # pending JSON control bytes per transport; reconcile on reconnect (PR #817 had 8 MiB:
                                    # more room for a teleport's burst before a reconnect)
WATERGUARD_VERSION = 1
TELEMETRY_CONSENT = 2           # telemetry.CONSENT_VERSION (the hello tells the welcome page; kept equal by a test)
_NO_PERF = perfstats.Collector()    # (a server object without one: tests' stand-ins)


def _perf(server):
    return getattr(server, "perf", None) or _NO_PERF


# The BeamNG.Earth server list (native/serverlist behind Caddy on the project's mini PC): server_list()
SERVER_LIST_URL = "https://earth.tgforever.net/servers"
# Tile/heightmap content format (hello.format) is the layout's: 3 = v1.3.1 (2 m lattice, fixed Q,
# 32-cell tiles), 4 = 64-cell tiles down to L-1 (farfield.CONTENT_FORMAT). A mod's hello.format is
# the newest format it reads (it reads every older one from 3: core.lua FORMATS), so a mod older
# than the server's format gets the mismatch status.
# Server(cells=...) default: the v1.3.1 layout (library and tests; test_v131_default pins it).
# The command line defaults to 64 cells in mesh mode since the in-game gate P8 passed
# (2026-10-10: 64-cell textured imports 4.0 ms at 256 px, 4.2 ms at 512 px; at 90 m/s the exact
# level was under the car 100 % of frames, queue max 35, vs 26 % / 2442 with 32 cells;
# docs/research/m1_d*); terrain mode stays frozen at 32.
CELLS_DEFAULT = 32
CLI_CELLS_MESH = 64


def resolve_block(block, near_visual):
    """The --block value to use: the given one, else CLI_BLOCK_MESH (1 km) in mesh / detail mode and TERRAIN_BLOCK
    (its frozen 2 km block) in terrain mode."""
    if block is not None:
        return block
    return CLI_BLOCK_MESH if near_visual in MESH_LAYOUT else TERRAIN_BLOCK


def resolve_colormatch(mode, chain):
    """--colormatch, else s2 when the imagery chain has USGS (the only source it corrects) and off
    otherwise. Server() keeps "off" (the golden message streams)."""
    if mode is not None:
        return mode
    names = [getattr(s, "name", s) for s in (chain or ())] if chain not in (None, "off") else []
    return "s2" if "usgs" in names else "off"


def resolve_horizon(horizon, near_visual):
    """--horizon / --no-horizon, else on in mesh mode (the owner's choice, 2026-10-11) and off in
    terrain mode. Server() keeps horizon=False (test_v131_default / test_cli_default)."""
    if horizon is not None:
        return horizon
    return near_visual in MESH_LAYOUT


def resolve_cells(cells, near_visual):
    """The --cells value to use: the given one, else 64 in mesh mode and 32 in terrain mode."""
    if cells is not None:
        return cells
    return CLI_CELLS_MESH if near_visual in MESH_LAYOUT else 32
SERVER_VERSION = netfetch.SERVER_VERSION      # one place: the User-Agent carries it too
NEAR_VISUALS = ("mesh", "terrain", "detail")
# detail (2026-10-12, docs/research/2026-10-12-engine-limits-and-worldgen.md §6-7): the mesh layout
# (64 cells, the 512² block, the horizon) with the physics block drawn as the near ground: aerial photo
# as its base colour, the surface materials' own detail textures; far tiles under it are hidden or sunk
# as in terrain mode. MESH_LAYOUT: the modes on the mesh-mode tile layout; BLOCK_DRAWN: the block is seen.
MESH_LAYOUT = ("mesh", "detail")
BLOCK_DRAWN = ("terrain", "detail")
TUNNEL_NEAR_M = 40.0          # m round a teleport's spot searched for a tunnel's tube (tunnel_floor)
TUNNEL_FLOOR_UNDER_M = 2.0     # a tunnel floor this far under the ground at a teleport's origin: the car goes on the floor
SEA_CHECK_S = 15.0          # sea (--sea on): the water tiles planned again when the player enters another
                            # 512 m square, or after this long (cells built meanwhile; seatiles.py)
VIEW_AWAY_M = 50.0              # m: a camera this far from the car centres what is drawn (Conn.camera_away)
PATCH_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")   # a terrain patch key (terrainpatch.lua)
FOCI_MAX = 16                   # other vehicles' collision foci taken from a pose (the mod's collisionfoci.lua)
FOCI_RANGE_M = 50000.0          # ... within this of the anchor's origin
FOCI_R_MAX_M = 640.0            # ... and radii at most this


def parse_foci(raw):
    """The pose's "foci" ([{x, y, r, trees}], the mod's collisionfoci.lua: other players' cars, traffic) as a tuple of
    (x, y, r, trees), FOCI_MAX at most, bad entries left out."""
    out = []
    if not isinstance(raw, list):
        return ()
    for f in raw[:FOCI_MAX]:
        try:
            x, y, r = float(f["x"]), float(f["y"]), float(f["r"])
        except (TypeError, KeyError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, r)) or max(abs(x), abs(y)) > FOCI_RANGE_M or r <= 0:
            continue
        out.append((x, y, min(r, FOCI_R_MAX_M), bool(f.get("trees"))))
    return tuple(out)
GROUND_RESEND_M, GROUND_RESEND_S = 1.0, 1.5   # ground_ahead: sent on a change, else this often (the mod's
                                               # follow.GROUND_STALE_S is 3 s)
DETAIL_RING_M = 128.0           # detail mode: L-1 tiles this far beyond the block (forced())
IMG_WARM_MARGIN = 200.0         # m beyond the block's half width that warm_imagery fetches ahead
# download_ahead: the raw data of the terrain blocks and road tiles ahead of a driving player, fetched into the
# disk caches before anything there is built (a block in a new area waited ~0.5-2 s on its elevation tiles,
# 2026-10-15). Points AHEAD_S s ahead on the straight path (at most AHEAD_MAX_M), at netfetch "prefetch"
# priority: only a host's spare tokens, so the builds' own downloads never wait for it
AHEAD_ON = True                 # --download-ahead
AHEAD_S = (10.0, 20.0, 30.0, 45.0, 60.0)
AHEAD_MAX_M = 4000.0
AHEAD_MIN_V = 8.0               # m/s: slower, the blocks around (and surface.prefetch's 3 km) cover it
AHEAD_MOVE_M = 150.0            # m the player moves before the next round
AHEAD_MARGIN_M = 250.0          # m around the block's half width at each point
# the road tiles' cells (landcover: surface.prefetch). With roads (surface.prefetch has them within 3 km, with the land
# cover on) and water, which a road tile otherwise fetched itself when it needed it (from KyzoFox's 1.5.4, PR #64)
AHEAD_CELLS = 4                 # map cells download_ahead fetches at once
AHEAD_LAYERS = ("roads", "buildings", "building_parts", "infrastructure", "water", "land_use")
# route_download: like download_ahead, but along the planned GPS route's own geometry (routeahead.py)
# instead of the player's current heading, since a road route can wander off that heading entirely.
ROUTE_AHEAD_ON = True            # --route-prefetch
ROUTE_AHEAD_MAX_M = 20000.0      # --route-prefetch-km * 1000 (15-25 km is sane; the user, 2026-10-22)
# the GPS's "Download along route" setting (the user, 2026-10-29: "options on pre downloading/generating along a gps
# route"): the route message's "download": off, ahead (ROUTE_AHEAD_MAX_M) or the whole route (at most this far)
ROUTE_DOWNLOAD_MODES = ("off", "ahead", "route")
ROUTE_WHOLE_MAX_M = 500000.0
ROUTE_IMG_TRIES = 5               # a route download's ground-photo tile the limiter deferred: asked this often
ROUTE_RETRY_S = 2.0               # a deferred tile/cell backs off this long before the next one in the corridor
# (70 m/s until 2026-10-31: at 130-157 mph each block waited 2.5 s for its photo, then its stand-in, the edge guard
# holding the car; the user: "lower it or do whatever you need to do to get high speed driving viable". The stand-in
# has the roads painted in and the last photos' overlap now, so the swap to the photo shows less)
PHOTO_SKIP_V = 55.0             # m/s (123 mph): faster, a requested block goes with its stand-in photo too (make_heightmap)
FAST_BASE_V = 40.0              # m/s (89 mph): faster, a block waits at most BASE_FAST_WAIT_S for its photo after its
SMALL_FAST_BASE_V = 35.0        # m/s (78 mph): FAST_BASE_V on a small machine (budget.small: a Steam Deck)
BASE_FAST_WAIT_S = 0.15         # heights, then goes without it (block_base later): blocks took 0.6-3 s at 600 mph on a
                                # Steam Deck, mostly the photo (heights 0.1-0.4 s), and the game paused 11 times a minute
# A block the game asked for has its photo rendered in the other lane's block process, its own core (_base_process),
# with the imagery analyses on too (de-shadow, colour match: they live in this process; the photo is rendered again
# with them, base_refresh): here, beside the far tiles' and the vegetation's work under this interpreter's lock, its
# road paint (0.3 s alone) took 6-10 s and the blocks were ready 6-12 s after they were asked for, the edge guard
# holding the car at 105-189 mph (2026-10-31, tools/perfmon.py). BNE_URGENT_ELSEWHERE=off: here, as before.
URGENT_ELSEWHERE = os.environ.get("BNE_URGENT_ELSEWHERE", "on") != "off"
# A process of its own for the requested blocks' photos and stand-ins (detail mode) on a machine with this many CPUs:
# in the other lane's block process they queued behind its prefetch heights and photos, a stand-in made beside a
# block 1.6-4 s late at 100-200 mph down Manhattan (nyc_speed.py, 2026-11-02); not on a small machine (memory)
PHOTO_PROCESS_MIN_CPUS = 12
# A block waits at most this long for its stand-in photo started beside it (cold: the photo process queued behind
# imagery downloads and a block built in 1.4 s waited 99.5 s for it, nyc_speed.py --cold 2026-11-02)
STANDIN_WAIT_S = 1.0
QUICK_SYNC_MAX_V = 150.0        # m/s (335 mph): slower, a block built ahead gets its stand-in photo before it is ready (224 mph left a player at 230 mph with brown flashes)
RECENT_FULL = 2                 # full ground photos kept for the stand-ins' overlap (write_quick_base)
QUICK_MIN_COVER = 0.8          # share of a stand-in's texels real imagery must cover (else no stand-in)
QUICK_SHARE_WAIT_S = 30.0       # Server._quick_once: a caller waits this long for the stand-in being made
QUICK_MADE_KEPT = 256           # blocks whose stand-in file names _quick_once remembers
QUICK_BASE_PX = 512             # a block's stand-in photo (write_quick_base): the imagery alone, 2 m texels
BASE_NO_STANDIN_MAX_S = 8.0     # a block with no stand-in possible waits this long at most for its photo
BASE_WAIT_S = 2.5               # ... and slower, this long: a slow photo held the heightmap thread and the blocks behind
                                # it (a city's first visit at 127 mph: 4.75 s for one photo, the next block 7.8 s and the
                                # car at the edge of its ground; a 4-core PC's photos took ~26 s, GitHub #496); it then
                                # goes with its stand-in, the photo after it (block_base)
FAST_KEEP_S = 20.0              # ... and this long after it last was: a car the game holds for its ground (the drop
                                # hold, the edge guard) reports no speed, and its next block waited for its photo
                                # again (5-10 s at 500 mph on a first visit, Indianapolis 2026-10-30: held half the time)
TRASH_DIR = "trash"         # in the cache: folders of the game's user folder moved out to be deleted (_rmtree_outside)
PROGRESS_S = 0.5            # load_progress at most this often (Connection.send_progress)
ANCHOR_PROGRESS_S = 2.0     # first ground: phase heartbeat, independent of the worker's optional tile counts
GROUND_WAIT_S = 90.0        # a block's photo still to be drawn again counts as loading this long at most
PERF_LOG_LINES, PERF_LOG_CHARS = 40, 2000   # a perf_log message's lines and characters a line, at most
SLOW_BASE_S = 3.0               # a base photo slower than this is logged with its stages
BASE_RENDER_PX = 1024           # detail mode: the block's base photo as the imagery renders it (the 1 km block: 1 m a
                                # texel; USGS is ~1.9 m, NAIP 0.6 m)
BASE_PX = 2048                  # ... and as the game gets it: enlarged (a plain resize), then the roads, pads and rock
                                # painted in at 0.5 m a texel (at 1 m the asphalt's edges blurred into the ground and
                                # the sand and the trees' green into the road: the user, 2026-10-28)
# The terrain shader samples a block's base colour half a sample on (block-local sample coordinates + 0.5): a photo
# over the block's own square showed every road painted into it a metre south-west of its lines and edge decals
# (painted dots against objects at the same points, in game 2026-10-28; the user: "the decals are still offset").
# The photo is rendered over the square moved back by that much.
BASE_SHIFT_M = heightmap.SQUARE / 2
BASE_AO_VFS = "/levels/beamng_earth/art/terrains/bngEarth_base_ao.png"   # every surface layer's aoBaseTex
# The ground photo sizes the mod may choose (its graphics setting groundPhoto, the 'graphics' message's photoPx; the
# user, 2026-10-28: "much more control in the graphics settings"): the level ships the flat base maps of each
# (tools/make_level_assets.py BASE_SIZES: BASE_PX's under the old names, the others with a _<px> suffix), and every
# layer's aoBaseTex is the one of the size in use, so the packed cache entry's key (terraincache.base_key) follows it.
# Measured 2026-10-28 (quality-presets notes): 4096 costs ~1-3 s a block on a PC, 6-12 s on a Steam Deck.
BASE_SIZES = (1024, 2048, 4096)


def base_ao_vfs(px):
    """The level's flat AO map of ground photo size px (every surface layer's aoBaseTex at that size)."""
    return BASE_AO_VFS if px == BASE_PX else f"{BASE_AO_VFS[:-4]}_{px}.png"
BASE_AO = 251                   # its flat value (tools/make_level_assets.py)
GRAPHICS_SCALE = (0.5, 3.0)     # the 'graphics' message's scales are clamped to this (Connection.set_graphics)
VIEW_CAP_MIN_M = 10000.0        # ... and its fixed view distance (viewM, the far field's radius cap) to at least this
DETAIL_ROAD_LIFT = 0.025        # detail mode: road surfaces over the drawn block, within roadmesh.FOLLOW_TOL of it (0.05
                                # before, over a coarser surface it still cut through on hills: the tyres sank into it,
                                # the user 2026-10-20)
TEX_NEAR, TEX_FAR = 256, 256    # 32 cells: texture edge (px) of the two finest levels / of the others (512
                                 # near was 15x finer than USGS: no visible gain for 2x build time and VRAM);
                                 # --tex gives one per level (64 cells: farfield.TEX_DEFAULT)
SESSION_MARKER = ".earthd-session"
# The engine caches every loaded .dae as a .cdae under <user-dir>/temp/<VFS path, lower-cased>
# (seen in-game: temp/beamngEarth/<sid>/far_*.dae -> temp/temp/beamngearth/<sid>/far_*.cdae,
# 1.8 GB / 24k files after one v1 session). Those caches are deleted along with their sources.
# Server-written .cdae tiles (the default) have no such duplicate.
CDAE_ROOT = ("temp", "temp", "beamngearth")
PLACE_PREVIEWS_MAX = 100     # place pictures made per request (a long list: the rest next time)
# the status bar's place names of a region not on disk yet (where(), placenames.py): fetched as a background job,
# asked again this often while Overture's hosts are busy with the world's downloads, this many times at most
WHERE_PLACE_RETRY_S, WHERE_PLACE_TRIES = 5.0, 60

HM_CACHE = 16               # heightmaps kept beyond the protected ones (6 before 2026-10-30: a fast player's prefetched
                            # chain (PREFETCH_CHAIN) was evicted before the mod asked for it and built again)
TERRAIN_STUCK_S = 90.0      # s: a terrain request still building this long is logged once (prepare_terrain)
HM_KEEP_COMMITTED = 3       # the current terrain and the last two
HM_KEEP_SENT = 2            # terrains handed to the mod but maybe not committed yet
# Generated far-tile files kept per server, by layout: a 64-cell tile is ~3.7x the bytes (348 KB)
# and a pose needs ~300 tiles instead of ~811 (docs/V1_4_SPEC.md §3.1).
TILE_CACHE = {32: 6000, 64: 2500}
HM_THREADS = 4
HM_REQUEST_THREADS = 2          # Server.hm_req_pool: the request lane's (they wait on its block process)
LAG_TICK_S, LAG_REPORT_S = 0.1, 0.25     # the event-loop stall report (-v)
FIN_WORKER_WINDOW, FIN_WORKER_MIN_TRIES, FIN_WORKER_SHARE, FIN_WORKER_PROBE = 40, 10, 0.3, 5   # (fin_worker_worth)
FIN_WORKER = os.environ.get("BNGE_FIN_WORKER", "1") != "0"     # (A/B: 0 renders them all here, as before)
NAV_LEAD_S, NAV_LEAD_MAX_M = 12.0, 500.0     # the nav graph's centre ahead of a moving car (navgraph.WINDOW_M 1100)
LAG_STALL_S = 2.0                     # s: a stall this long is logged without -v (its stack, then its length)
# The event loop's thread a step above the server's other threads (budget.raise_thread, Windows): a block built 17 s
# earlier held the car at the edge until the loop answered its request (A13 at 200 mph, 2 server cores, 2026-11-04;
# GitHub #725). BNGE_LOOP_RAISE=0: as the others, as before
LOOP_RAISE = os.environ.get("BNGE_LOOP_RAISE", "1") != "0"
# ms the interpreter lock is held before a waiting thread is let in (Python's own: 5): the event loop waits up to this
# long behind each busy thread. BNGE_SWITCH_MS=0: Python's own, as before
SWITCH_MS = float(os.environ.get("BNGE_SWITCH_MS", "2") or 0)
ROAD_PROCESSES = 2              # road-tile processes (--block-process on, --roads on)
# s a road tile may take in its worker before it is given up and its workers restarted (build_roads_tile): a first
# visit's slowest tiles take ~1 min on a busy 6-core laptop and several times that on a Steam Deck, where a tile given
# up at 2 min would be built and given up again for good; BNGE_ROAD_TILE_TIMEOUT=<s> to change it
ROAD_GEO_SLOW_S = 3.0           # a road geotile slower than this is logged (queue and build)
VEG_GEO_BATCH = 32              # trees geotiles one road-process job builds at most (_veg_geo_job): its nearest wait
                                # for the rest of it
VEG_GEO_PER_POSE = 64           # trees geotiles asked for at once when the road processes build them: two jobs' worth
SLOW_SELECT_S = 0.25            # a far-tile selection (update_far, on the event loop) slower than this is logged
WORLD_WARM_Z = 7                # the rough earth's tiles (Server.start_world_warm): 16,384, ~260 km at the equator
WORLD_WARM_CONC = 2
EYE_LOD_MIN_M = 3000.0          # an eye above this: the far tiles' detail by the 3D distance (eye_clearance)
MAP_HIGH_EYE_M = 40000.0        # an eye above this (the 3D map): the finest global tiles are leaves (farfield.select_global)
MAP_RAMP_START = 0.125           # the 3D map at a new anchor: K x this first (Conn.commit_anchor) ...
MAP_RAMP_STEP = 2.0              # ... times this each time that far field is done, up to 1
MAP_OUT_OF_VIEW_K = 3.0          # the 3D map: tiles off the screen chosen as if this many times as far (map_cone)
MAP_ZOOM_RAMP_K = 8.0            # the 3D map's eye this many times lower than before: coarse first again (on_pose)
MAP_FOCUS_MAX_M = 1200000.0      # a map_focus farther from the anchor than this is ignored (mod map3d REBASE_M)
FAR_TRACE = os.environ.get("BNGE_FARTRACE") == "1"     # (tools: every far tile sent, logged)
MAP_TEX_PX = 512                # the 3D map's far textures at least this (Server.tex_size) ...
MAP_TEX_MAX_LEVEL = 2           # ... up to this level (coarser ones: 4x the photo tiles each, and EOX answered 403 to
                                # ~135 requests/s with every level at 512 px and no cap, 2026-11-07)
MAP_TEX_OFF_PX = 128            # the 3D map: a far tile's texture off the screen (Server.tile_tex_px) ...
MAP_TEX_MIN_PX = 128            # ... never under this on it ...
MAP_TEX_OVERSAMPLE = 1.5        # ... and at most this many texels per screen pixel of its edge
MAP_SCREEN_H = 1440.0           # ... for a screen this many pixels high
FLY_TEX_EYE_M = 300.0           # flying higher than this over the ground: textures as fine as on screen too
MAP_TEX_UPGRADE_S = 1.0         # shown tiles checked this often for a larger texture as the camera turns
MAP_LOD_SCALE = 1.25            # ... and their detail reaching this much farther (Connection.kfun)
MAP_DETAIL_EYE_M = 3000.0       # the 3D map's view frame: roads, trees round its target only below this eye ...
MAP_BUILDINGS_EYE_M = 10000.0   # ... the buildings below this one (the user, 2026-11-07: "make buildings start rendering
MAP_BLD_FAR_EYE = 1.0           # further out"), the far ones to this many times the eye's height ...
MAP_BLD_FAR_MAX_M = 6000.0      # ... at most this far,
MAP_BLD_SKY_EYE = 3.0           # the skyline to this many times it ...
MAP_BLD_SKY_MAX_M = 20000.0     # ... at most this far (Connection.building_reach)
VIEW_ANCHOR_Q = 1 / 32          # the 3D map's view-only anchors' height quantum (m): no survey, no block driven on
ROAD_TILE_TIMEOUT_S = float(os.environ.get("BNGE_ROAD_TILE_TIMEOUT", "") or 300.0)
WORKERS = 3                     # far-tile workers (--workers): while driving (FAR_DRIVE)
FAR_DRIVE = 3
FAR_MAX_DESKTOP = 10            # ... up to this many with the 3D map open / the loading boost (desktops, budget.py)


class DynGate:
    """A semaphore whose limit is read at each acquire (limit(): an int >= 1)."""

    def __init__(self, limit):
        self.limit = limit
        self.busy = 0
        self.cv = threading.Condition()

    def __enter__(self):
        with self.cv:
            while self.busy >= max(1, int(self.limit())):
                self.cv.wait(0.25)          # (the limit can rise meanwhile: the map opened)
            self.busy += 1
        return self

    def __exit__(self, *exc):
        with self.cv:
            self.busy -= 1
            self.cv.notify_all()
        return False
TRACE_S = 60.0                  # BNGE_TRACEMALLOC: seconds between the memory reports ...
TRACE_TOP = 15                  # ... and the allocation sites in each
PROVISIONAL_ELSEWHERE = os.environ.get("BNGE_PROV_HERE") != "1"   # a teleport's first block in a block process
LANE_REQUEST, LANE_PREFETCH = 0, 1     # Server.block_pools: the blocks the mod asks for / the prefetch
HM_BUSY_TILE_WORKERS = 1    # far tiles built at once while a physics block is building in the server process (an
                            # anchor's first, or every one with --block-process off; the block processes' blocks do
                            # not share its interpreter: far tiles one at a time behind them were most of a Steam
                            # Deck's far field time, blocks building nearly all the time while driving, 2026-10-20)
WARM_STEPS_S = (0.0, 10.0, 20.0, 30.0)   # s ahead on the player's straight path whose road cells are planned early
WARM_AROUND = 2000.0        # m: and every cell this close to the player (any heading)
PREFETCH_START = 200.0      # m (Chebyshev) from the block centre before prefetching
# A fast player (FAST_BASE_V) gets the chain of its next PREFETCH_CHAIN blocks prefetched, PREFETCH_PARALLEL at once: at
# 500 mph the next block was asked for ~1.3 s before it was needed and took 0.5-3 s, so the game held its car for its
# ground half the time (Indianapolis 2026-10-30); one predicted block was not enough
PREFETCH_CHAIN = 4
READY_STEPS = 2             # grid steps a built block may be from the mod's target and still be taken (follow.lua)
PREFETCH_PARALLEL = 2
# The prefetch lane's block processes: two on a machine with the cores and memory for them (PREFETCH_PARALLEL's two
# blocks then build at once; with one process they took turns and a third of the blocks asked at 140-165 mph were not
# built yet, 4.4 s each, while the server used ~3 of 16 cores, 2026-11-08), one on a small one (a Steam Deck's memory).
# BNE_PREFETCH_PROCS sets it.
PREFETCH_PROCS_CPUS = 12
PREFETCH_PROCS_GB = 24.0


def prefetch_procs():
    """Block processes for the prefetch lane on this machine (PREFETCH_PROCS_*; BNE_PREFETCH_PROCS overrides)."""
    if os.environ.get("BNE_PREFETCH_PROCS"):
        try:
            return max(1, min(4, int(os.environ["BNE_PREFETCH_PROCS"])))
        except ValueError:
            pass
    import budget
    gb = budget.total_gb() or 0.0
    return 2 if (os.cpu_count() or 0) >= PREFETCH_PROCS_CPUS and gb >= PREFETCH_PROCS_GB and not budget.small() else 1
READY_SET_M = 3000.0        # m: the built blocks round the player the mod is told of (terrain_ready_set)
# The mod's terrain-follow rule (mod follow.lua): it asks once the player is more than
# FOLLOW_PREPARE (Chebyshev) from the block centre, for snap256(player + v * FOLLOW_LEAD),
# clamped so the player stays within FOLLOW_REACH of the new centre on both axes
# (1020 - 600 m margin, §1.2; with the 2046 m collidable width of §12 P1 the N/E edge is at
# +1022, so this keeps >= 602 m there). tests/test_protocol_agreement.py checks these
# against follow.lua.
FOLLOW_PREPARE = 300.0
FOLLOW_LEAD = 4.0
FOLLOW_REACH = 420.0
FRAME_S = 0.05              # the mod decides per frame, so it asks just past the threshold


@dataclass(frozen=True)
class FollowRule:
    """The mod's terrain-follow table for one block size (follow.lua, its flat *_1024 / *_512
    constants; tests/test_protocol_agreement.py checks both sides): centre grid, prepare distance
    (Chebyshev from the centre), target lead (s) and reach (max |centre - player| per axis), the
    prefetch start, and the fast mode (|v| > fast_v: lead fast_lead, reach fast_reach, the prepare
    test at player + v * fast_ahead). 1024² has no fast mode (fast_v infinite)."""
    grid: float
    prepare: float
    lead: float
    reach: float
    prefetch_start: float
    fast_v: float = math.inf
    fast_lead: float = 0.0
    fast_reach: float = 0.0
    fast_ahead: float = 0.0
    min_lead: float = 0.0           # the target at least this far ahead of a moving player (follow.lua MIN_LEAD)
    min_lead_v: float = 2.0         # (moving: faster than this, m/s; follow.lua MIN_LEAD_V)

    def mode(self, vx, vy):
        """(lead, reach, prepare-test ahead s) for this velocity."""
        if vx * vx + vy * vy > self.fast_v * self.fast_v:
            return self.fast_lead, self.fast_reach, self.fast_ahead
        return self.lead, self.reach, 0.0


# By the block's width in metres (heightmap.grid): docs/V1_4_SPEC.md §3.5 (M4): the 1 km block (collidable
# [c-512, c+510] at 2 m, c+511 at 1 m) on a 128 m grid: prepare at 128 m, lead 5 s (was 2: 2026-10-15, fewer moves),
# reach 510 - 250 = 260 m; fast mode above 150 m/s: lead 3 s, reach 510 - 90 = 420 m, prepare test 0.75 s ahead; the
# prefetch starts 28 m before the prepare distance. 2 km: the v1.3.1 block.
FOLLOW_RULES = {
    2048.0: FollowRule(heightmap.GRID_BY_M[2048.0], FOLLOW_PREPARE, FOLLOW_LEAD, FOLLOW_REACH, PREFETCH_START),
    1024.0: FollowRule(heightmap.GRID_BY_M[1024.0], 128.0, 5.0, 260.0, 100.0,
                       fast_v=150.0, fast_lead=3.0, fast_reach=420.0, fast_ahead=0.75, min_lead=256.0),
}
BLOCK_SIZES = tuple(int(m / heightmap.SQUARE) for m in sorted(FOLLOW_RULES))    # samples per edge (1024, 2048)
TERRAIN_BLOCK = farfield.TERRAIN_BLOCK     # terrain mode's frozen 2 km block
BLOCK_DEFAULT = 1024        # Server()'s block (test_v131_default.py pins the library default)
# The command line's block in mesh mode: 1 km (M4: in-game 2026-10-10, the 512² block at 2 m; since 2026-11-01 the
# same kilometre at 1 m, 1024² samples: the stock maps' resolution, a .ter swap of ~29 ms on the Stelvio)
CLI_BLOCK_MESH = 1024
# |cx|,|cy| of an anchor's first terrain. A rebase hint stays within ~512 m of the new origin, but
# a resync (same origin, after a server restart) asks for the block around the player, which can be
# up to the mod's maximum rebase distance (16 km) away, plus however far the player got past it.
# Clamping that to ±768 m made the mod reject every answer as stale and re-request forever.
ANCHOR_CENTRE_MAX = 21000.0
# A superseded anchor's files are deleted this long after the new anchor's far_done: the mod
# still shows (and deletes, time-sliced) old tiles after far_done, and the engine watches
# loaded files (ge/main.lua onFileChanged), so a shape must not vanish under a live TSStatic.
PRUNE_DELAY = 30.0
# A tile the server swapped out (retired) keeps its files this long: the mod shows it until the swap
# has run there, and with an import backlog (in-game 2026-10-10, ball drive at ~38 m/s: ~1000 swap
# groups waiting) that took minutes; evicting at 30 s deleted shapes under shown tiles, which then
# rendered as nothing (the world went invisible, collision stayed). v1.4 replaces this guess with
# the mod reporting its deletes.
RETIRED_KEEP = 600.0
PRUNE_RECHECK = 2.0          # s; a prune waiting for the mod to let go of an anchor's tiles re-checks this often
# The game watches /temp: deleting ~400 tile files in one burst stalled it for 200-340 ms
# (in-game 2026-10-10, ~30 s after every rebase). Old files are deleted gradually instead.
DELETE_INTERVAL = 0.2           # seconds between file deletions (5 files/s)
# Mesh mode with imagery writes ~5x as many files per km (64/128 m tiles and a texture per tile:
# ~400 tiles = 800 files per km; a 10 km anchor leaves ~11k), more than 5 files/s can ever delete.
# There the queue drains faster when it is long, still one file at a time and evenly spaced:
# rate = backlog / DELETE_DRAIN_S, between 5 and DELETE_MAX_RATE files/s.
DELETE_DRAIN_S = 60.0
DELETE_MAX_RATE = 40.0
# Horizon mode (M1h): the eye is quantised so a car on a slope does not re-select every pose: it moves
# when it changed by more than max(EYE_STEP, EYE_STEP_REL * eye). Without a camera in the pose the eye
# is EYE_ABOVE_PLAYER over the player's z. The radius is re-sent (status.radius) when it changed by
# more than RADIUS_REPORT_REL; a failed coarse DEM pass is retried after HORIZON_RETRY s.
EYE_STEP = 10.0
EYE_STEP_REL = 0.05
EYE_ABOVE_PLAYER = 1.5
RADIUS_REPORT_REL = 0.02
HORIZON_RETRY = 30.0
HORIZON_BLOCK_CACHE = 1024      # DEM tiles' block maxima kept between anchors (16 KB each)
# Anchor height quantum: relief survey of +-SURVEY_HALF around the anchor at DEM zoom SURVEY_ZOOM.
SURVEY_HALF = 15000.0
SURVEY_N = 61                       # 500 m apart
SURVEY_WINDOW = 1.0                 # blocks: the windows survey_q takes the relief of (+ one survey step: an unaligned block)
SURVEY_PERCENTILE = 99.0            # of the windows' relief the anchor's quantum must hold (the rest: their own, coarser)
BLOCK_SURVEY_M = 2048.0             # m: the largest block (1024² at 2 m); Server passes its own
SURVEY_ZOOM = 10
DELTA_MARGIN = 0.1          # m added to measured swap-group deltas (crossings between vertices)
NICE = 10                   # default niceness of the server process (§4.5)
NICE_ROADS = 3              # the road tile workers' niceness on top of it (blockworker.init: the terrain blocks first)
NICE_FAR = 6                # the far field's (the scenery far away last)
NICE_FAR_BUILDINGS = 9      # the far buildings': after the far field (on a Steam Deck's four cores a teleport's far
                            # buildings took ~23 s of CPU, next to the road tiles and the far field)
IMAGERY_PREFETCH = 40       # queued tiles whose imagery is fetched ahead of their build (only spare request tokens)
PREFETCH_AGAIN_S = 2.0      # s before a queued tile's imagery prefetch is asked for again
PREFETCHED_MAX = 8192       # tiles remembered as prefetched (a connection's record starts again past this)
# status.backlog (docs/V1_4_SPEC.md §3.0, §4): every status message carries the number of desired
# tiles of the current anchor not shown yet. Status messages only go out on events (v1.3.1); with
# --status-interval S > 0 the last status is also re-sent from on_pose at most every S seconds
# while the backlog changed, so tools/drive.py can sample it (M0 harness runs use 1).
STATUS_INTERVAL = 0.0
NET_TICK_S = 60.0           # the network summary line in the log and the netstats file
# Session folders of earlier runs (docs/ROADMAP.md "Robustness" 3). The game may still show a dead
# server's tiles (the engine watches loaded files: a deleted shape under a live TSStatic renders as
# nothing), so a restarted server deletes at start only folders whose marker is untouched for
# SESSION_STALE_S; younger ones wait for a game to claim them (hello / pose `sessions`). Unclaimed,
# one goes once SESSION_ORPHAN_S passed since the server found it AND since its marker was last
# touched (a live server touches its own, and its adopted ones, every SESSION_HEARTBEAT_S).
SESSION_STALE_S = 86400.0
SESSION_ORPHAN_S = 600.0
# A geotile every game dropped (roads_drop, bldfar_drop, ...): its session files deleted this long after, unless one
# is sent it again meanwhile (geo_dropped). Before (GitHub #599, 2026-10-31) they stayed until the server stopped: the
# geotiles are anchor-free, so prune never took them, and two days' driving left 9 GB in the session folder.
GEO_FILES_GRACE_S = 120.0
GEO_LIST_BATCH_S = 1.0          # geo_dropped: tiles whose grace ends within this share one listing of the folder
GEO_TRACKERS = {"roads": "roads", "bldfar": "bldfar", "bldsky": "bldsky", "brfar": "brfar"}  # kind -> Connection attr
SESSION_HEARTBEAT_S = 60.0
SESSION_SWEEP_S = 30.0      # main_async: how often sweep_sessions runs
LEASE_GRACE_S = 900.0       # a disconnected game's stream lease (PR #817) kept this long for it to reconnect
# Live retexture (--colormatch / --deshadow): when a region a shown tile was rendered without becomes
# ready, the tile's texture is rendered again under its next revision's name and the mod is told
# (far_retex) to point the tile's material at it. At most RETEX_INFLIGHT such renders on the tile pool
# at once (the other workers keep building), started at most RETEX_RATE per second, nearest the player
# first. A tile is re-rendered once per region of it that turns ready (a coarse L5 / L6 texture spans up
# to 3 x 3 / 4 x 4 z12 regions, as many de-shadow ones); the guard against a loop is RETEX_REPEAT re-renders
# in a row that came back still waiting for a region already ready: past that it waits to be re-sent.
# --retex-rate 0: no live swap (a stale tile is re-rendered when it is sent again, as before).
RETEX_RATE = 8.0
MAP_RETEX_BOOST = 4         # the 3D map open: RETEX_INFLIGHT and RETEX_RATE times this (retexture: in workers)
RETEX_INFLIGHT = 2         # (2: the 3D map's quick textures sharpen as their photos come, 2026-11-07)
RETEX_REPEAT = 3
# Shown in the mod's UI (V1_3_SPEC §C), after the imagery credits.
ELEVATION_CREDIT = "Elevation: Mapterhorn (see mapterhorn.com/attribution), AWS Terrain Tiles / Mapzen"

log = logging.getLogger("earthd")


@dataclass
class Anchor:
    id: int
    lat: float
    lon: float
    frame: geodesy.EnuFrame
    hm_zoom: int = 15
    zooms: dict = field(default_factory=dict)      # far level -> DEM zoom (L0 = hm_zoom)
    first_terrain: tuple = None                    # (cx, cy) of the anchor_ready terrain
    q: float = None                                # height quantum, from the relief survey
    q_lock: threading.Lock = field(default_factory=threading.Lock)
    grader: object = None                          # grading.Grader (--grading on), made on first use
    chunks: object = field(default_factory=heightmap.ChunkCache)   # the physics blocks' heights per 128 m
    bridges: object = None                         # roads.BridgeProfiles: deck heights per chain
    tunnels: object = None                         # roads.TunnelProfiles: floor heights per chain
    first_base: tuple = None                       # ((cx, cy), Future): a provisional first block's photo, reused
    zf: int = 14                                   # far level 3's Web-Mercator zoom here (farfield.global_zoom)


PIECE_PARTS = 24            # nodes whose geometry and texture the server keeps for their other pieces (Server.piece_parts)
# The far tiles' server key of a global tile (farfield.GTile, geotiles phase 3b): (GLOBAL_AID, key), the same under
# every anchor (anchor ids start at 1)
GLOBAL_AID = 0


@dataclass
class TileInfo:
    """A built far tile (server-wide cache entry)."""
    pos: list
    err: float
    sink: float | None
    skirt: float
    grid: np.ndarray            # (cells + 1)² absolute vertex heights (float32), for swap deltas
    bytes: int
    seconds: float
    material: str | None = None     # per-tile imagery material (None: the static palette material)
    texture: str | None = None      # file name of its texture (next to the shape)
    tex_bytes: int = 0
    tex_seconds: float = 0.0
    # --colormatch / --deshadow: regions whose correction was not ready when the texture was rendered
    # (left raw there; de-shadow ones as (imagery.DS_PENDING, rx, ry)), whether one of them has turned
    # out to change it (the texture is then rendered again under a new file name: live for a tile on
    # screen, far_retex, else the next time the tile is sent), and the texture's revision
    cm_pending: frozenset = frozenset()
    stale: bool = False
    rev: int = 0
    # live retexture: re-renders in a row that came back stale at once (still waiting for a region
    # whose status is ready: the loop guard, RETEX_REPEAT)
    retex_repeat: int = 0
    credits: frozenset = frozenset()    # data-credit source ids (credits.tile_sources; --credits on only)
    tex_px: int = 0                     # its texture's size (Server.tile_tex_px; 0: tex_size of its level)
    shape: str | None = None    # its shape file when not tile_name(): a copy under a new material name (reslot)
    tile: object = None         # the farfield tile it was built for (Tile, GTile, Piece)
    frame: tuple | None = None  # a global tile's frame (lat, lon): its geometry is in that frame's world


def predict_centre(px, py, vx, vy, lead, reach=FOLLOW_REACH, grid=heightmap.GRID_BY_M[2048.0], min_lead=0.0,
                   min_lead_v=2.0):
    """snap256(p + v*lead) (M4: the block's grid), the lead at least min_lead metres while moving (follow.lua's
    MIN_LEAD), clamped so p stays within `reach` of the centre on both axes."""
    lx, ly = vx * lead, vy * lead
    v2 = vx * vx + vy * vy
    if min_lead > 0 and v2 > min_lead_v * min_lead_v and lx * lx + ly * ly < min_lead * min_lead:
        k = min_lead / math.sqrt(v2)
        lx, ly = vx * k, vy * k
    out = []
    for p, d in ((px, lx), (py, ly)):
        c = heightmap.snap(p + d, grid)
        lo = math.ceil((p - reach) / grid) * grid
        hi = math.floor((p + reach) / grid) * grid
        out.append(float(min(max(c, lo), hi)) + 0.0)
    return tuple(out)


def predict_request(px, py, vx, vy, cx, cy, rule=FOLLOW_RULES[2048.0], at=None):
    """The centre the mod will ask for (V1_1_SPEC 4 prefetch): where a player moving
    straight at (vx, vy) leaves the FOLLOW_PREPARE box around the block centre, then the
    mod's own target rule there. (Spec 4 suggests snap256(p + v*6 s); predicting the mod's
    exact request instead avoids builds that are never asked for.) rule: the block size's
    follow table; in its fast mode the box test is on the point v * fast_ahead ahead."""
    lead, reach, ahead = rule.mode(vx, vy)
    v2 = vx * vx + vy * vy
    if (rule.min_lead > 0 and rule.min_lead_v ** 2 < v2 and v2 * lead * lead < rule.min_lead ** 2
            and (cx - px) * vx + (cy - py) * vy > 0
            and max(abs(px - cx), abs(py - cy)) <= rule.min_lead + rule.grid):
        # (follow.lua centreAhead: the centre just committed is ahead; the next request once the player is past it)
        t0 = ((cx - px) * vx + (cy - py) * vy) / v2 + FRAME_S
        px, py = px + vx * t0, py + vy * t0
    qx, qy = px + vx * ahead, py + vy * ahead
    t = 0.0
    if max(abs(qx - cx), abs(qy - cy)) <= rule.prepare:
        exits = [((c + math.copysign(rule.prepare, v)) - q) / v
                 for q, v, c in ((qx, vx, cx), (qy, vy, cy)) if abs(v) > 0.5]
        t = min(exits) + FRAME_S if exits else 0.0
    if at is not None:          # (the player where it asks: READY_STEPS, Connection.maybe_prefetch)
        at.append((px + vx * t, py + vy * t))
    return predict_centre(px + vx * t, py + vy * t, vx, vy, lead, reach, rule.grid, rule.min_lead, rule.min_lead_v)


def predict_chain(px, py, vx, vy, cx, cy, rule, n, at=None):
    """The next n centres the mod will ask for (predict_request, chained): after each, the player where its path passes
    nearest that centre, the next request from there. Fewer when the chain stops (no move, a repeat). at: a dict
    filled with each centre's request point (the player where it asks)."""
    out = []
    v2 = vx * vx + vy * vy
    if v2 < 1.0:
        return out
    for _ in range(n):
        where = []
        c = predict_request(px, py, vx, vy, cx, cy, rule, where)
        if c == (cx, cy) or c in out:
            break
        out.append(c)
        if at is not None:
            at[c] = where[0]
        t = max(0.0, ((c[0] - px) * vx + (c[1] - py) * vy) / v2)
        px, py, (cx, cy) = px + vx * t, py + vy * t, c
    return out


# A fast player on their GPS route: the blocks ahead predicted along it, not along a straight line (predict_chain): at
# each turn of a route through Manhattan at 150 mph the straight chain's blocks were never asked for, ~40 % of the
# blocks built, and the ones asked for after the turn queued behind them (nyc_speed.py 2026-11-02).
ROUTE_NEAR_M = 40.0         # m: the player this near the route's line ...
ROUTE_ALIGN = 0.7           # ... and heading along it (cos of the angle) is on it
ROUTE_STEP_M = 16.0         # m between the route points the mod's follow rule is tried at
ROUTE_AHEAD_S = 20.0        # s of driving along the route looked at, at most ...
ROUTE_PREDICT_MAX_M = 3000.0    # ... and this far


def predict_route_chain(xy, px, py, vx, vy, cx, cy, rule, n, at=None):
    """The next n centres the mod will ask for while the player at (px, py) moving (vx, vy) drives along the route
    polyline xy ((k, 2) in the anchor frame), or None when the player is not on it (ROUTE_NEAR_M, ROUTE_ALIGN). The
    mod's rule (predict_request) tried every ROUTE_STEP_M along the route with the route's own heading there."""
    sp = math.hypot(vx, vy)
    if sp < 1.0 or xy is None or len(xy) < 2:
        return None
    a, b = xy[:-1], xy[1:]
    d = b - a
    ln2 = np.maximum((d * d).sum(1), 1e-9)
    t = np.clip(((px - a[:, 0]) * d[:, 0] + (py - a[:, 1]) * d[:, 1]) / ln2, 0.0, 1.0)
    q = a + d * t[:, None]
    dist = np.hypot(q[:, 0] - px, q[:, 1] - py)
    j = int(np.argmin(dist))
    seg = float(np.sqrt(ln2[j]))
    if dist[j] > ROUTE_NEAR_M or ((d[j, 0] * vx + d[j, 1] * vy) / (seg * sp)) < ROUTE_ALIGN:
        return None
    reach = min(ROUTE_PREDICT_MAX_M, sp * ROUTE_AHEAD_S)
    out, gone, x, y, k = [], 0.0, float(q[j, 0]), float(q[j, 1]), j
    while gone < reach and k < len(d) and len(out) < n:
        ux, uy = d[k, 0] / math.sqrt(ln2[k]), d[k, 1] / math.sqrt(ln2[k])
        bx, by = float(b[k, 0]), float(b[k, 1])
        left = math.hypot(bx - x, by - y)
        step = min(ROUTE_STEP_M, left)
        if left <= 1e-6:
            k += 1
            continue
        x, y, gone = x + ux * step, y + uy * step, gone + step
        lvx, lvy = ux * sp, uy * sp
        _lead, _reach, ahead = rule.mode(lvx, lvy)
        if max(abs(x + lvx * ahead - cx), abs(y + lvy * ahead - cy)) > rule.prepare:
            where = []
            c = predict_request(x, y, lvx, lvy, cx, cy, rule, where)
            if c != (cx, cy) and c not in out:
                out.append(c)
                if at is not None:
                    at[c] = where[0]
                cx, cy = c
    return out


def _num(msg, name, default=None):
    v = msg.get(name, default)
    if v is None:
        return default
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{name} is not finite")
    return v


def survey_q(frame, dem, block_m=BLOCK_SURVEY_M):
    """Height quantum of an anchor (§1.3): the smallest ladder step whose block range covers the
    SURVEY_PERCENTILE relief of the windows of SURVEY_WINDOW blocks (block_m m) of a +-15 km survey at
    DEM z10, plus 10 %. A block only has to hold its own relief (its zBase is its own; one that does not
    fit takes the next coarser step, heightmap.block_params). The whole survey's relief made every
    mountain anchor 1/16 m: 6 cm steps on the runways and roads (Courchevel, in-game 2026-10-13); so did
    its one worst window of 1.5 blocks (the Stelvio's and Courchevel's, whose 2 km blocks within 6 km all
    hold 1/32 m): its 2 m samples rounded to 6 cm steps were bumps on every steep road, twice the time over
    0.2 g of 1/32 m in a quarter car (the user, 2026-11-01: "Fix the bumps")."""
    xs = np.linspace(-SURVEY_HALF, SURVEY_HALF, SURVEY_N)
    u = heightmap.surface_u(frame, lambda la, lo: dem.sample(la, lo, SURVEY_ZOOM), xs, xs)
    k = min(len(xs), int(math.ceil(SURVEY_WINDOW * block_m / (xs[1] - xs[0]))) + 1)
    win = np.lib.stride_tricks.sliding_window_view(u, (k, k))
    return heightmap.choose_q(float(np.percentile(win.max(axis=(2, 3)) - win.min(axis=(2, 3)), SURVEY_PERCENTILE)))


def game_running():
    """Was BeamNG.drive running when this server started? The launcher says (BNGEARTH_GAME_AT_START: it starts the game
    right after the server, a new game that holds none of the earlier sessions); else whether one runs now: its Linux
    build (cpupin.game_pids), its Windows build on Windows or under Wine / Proton (a process whose command line names
    BeamNG.drive.x64.exe). Unsure (an error): True, the careful answer (_scan_sessions then keeps earlier sessions for
    the game to claim)."""
    said = os.environ.get("BNGEARTH_GAME_AT_START")
    if said in ("0", "1"):
        return said == "1"
    try:
        import cpupin
        if cpupin.game_pids():
            return True
        if os.name == "nt" or not os.path.isdir("/proc"):
            return False
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/cmdline", "rb") as f:
                    if b"beamng.drive.x64.exe" in f.read().lower():
                        return True
            except OSError:
                continue
        return False
    except Exception:
        return True


def road_tracker(conn):
    """A connection's road tiles' tracker: geotiles (roads.GeoTracker) when its mod places them (Connection.road_geo),
    else the anchor's."""
    return roadsmod.GeoTracker() if getattr(conn, "road_geo", False) else roadsmod.Tracker()


OSM_FUEL_THREADS = 2     # OpenStreetMap fuel cells fetched at once (Server.osm_fuel_want)
OSM_FUEL_WARM_TRIES = 10    # the start's index read tried this often (Server._osm_fuel_warm) ...
OSM_FUEL_WARM_WAIT_S = 30.0  # ... this far apart


def osm_fuel_cells(frame, tile, geo=False):
    """The map cells whose OpenStreetMap fuel stations a road tile's layouts read (gasstations.station_cells): tile
    (tx, ty) of anchor frame, or geotile (gx, gy) with geo (its own frame; frame unused)."""
    if geo:
        import geotiles
        return gasstationsmod.station_cells(geotiles.frame(*tile), geotiles.geobox(*tile), None)
    return gasstationsmod.station_cells(frame, tile[0], tile[1])


def osm_fuel_tiles(frame, key, tiles, geo=False):
    """The road tiles among tiles whose stations reach map cell key (osm_fuel_cells)."""
    k = str(key)
    return {t for t in tiles if any(str(c) == k for c in osm_fuel_cells(frame, t, geo))}


TURBINE_GEO = 1
TURBINE_Z = 15             # about 1 km cells at mid-latitudes; existing far-content tracker
TURBINE_SEND_M = 4000.0
TURBINE_DROP_M = 5000.0
TURBINE_PER_POSE = 2       # bounded batches: only populated cells request terrain elevations


# Per-anchor tile files (besides far_ / hm_) that prune() deletes with their anchor
# A teleport's first block is built first without the road grading (make_provisional: the car is placed in ~5 s
# instead of 15-60 s on a fresh place) and replaced by the graded one when that is built (terrain_refresh)
PROVISIONAL_FIRST = os.environ.get("BNGE_PROVISIONAL", "1") != "0"
PROVISIONAL_KEEP_S = 60.0   # the provisional .ter is deleted this long after its replacement was sent
# A follow block the mod asked for whose road plans are not made yet (a first visit's cells, 15-120 s: requests queued
# 72-80 s in Chicago, the edge guard holding the car, 2026-11-01): sent provisional at once (make_provisional, its own
# threads) and the graded one as terrain_refresh when built (refine_follow); BNGE_PROVISIONAL_FOLLOW_S=0: never
PROVISIONAL_FOLLOW_S = float(os.environ.get("BNGE_PROVISIONAL_FOLLOW_S", "2.5"))
PROVISIONAL_RUNNING_S = 7.0     # s: a planned block already building in its process gets this long before a provisional

PADS_REFRESH_MS = 5.0       # the block in use built again with its gas station pads only below this speed (m/s) ...
PADS_SETTLE_S = 3.0         # ... and this long after the last new pad (Connection.pads_refresh)
NAV_WAIT_S = 30.0           # s an anchor's first AI graph waits at most for the road tiles round the car (nav_may_start)
TELEPORT_PAUSE_M = 20000.0  # an anchor asked this far from the player is a teleport: the old place's work pauses
TELEPORT_PAUSE_MAX_S = 120.0  # ... at most this long (the mod may never commit it)
BOOST_MOVE_MS = 3.0         # the loading boost ends when the player moves faster than this (m/s) ...
BOOST_MAX_S = 120.0         # ... or this long after the teleport
BOOST_GRACE_S = 5.0         # (the car dropped into place moves a little at first)
# While driving: the loading boost again when the world falls behind (the user's Manhattan drive 2026-11-01: the server
# saturated its 2 cores (4 CPUs) with the PC half idle, blocks queued 5-16 s and the edge guard held the car): the next
# block asked this long ago and not committed yet (still building, or its far tiles not live: the mod's gate), or the
# roads round the car not built; off once caught up for CATCHUP_CALM_S
CATCHUP_AFTER_S = 1.5
# ... the road tiles counted: those within this of the car (all of SEND_M's 640 m were never done while driving into
# new streets: the boost on all drive long at 20-60 mph, Fells Point 2026-11-07)
CATCHUP_ROADS_M = 300.0
CATCHUP_CALM_S = 3.0
BAKE_FETCH = os.environ.get("BNGE_BAKE_FETCH", "off") == "on"   # the data repo's packs (Server.bake_sync): off until
# there are any (the user, 2026-11-02: no cloud baking while world gen changes daily; local packs, BeamMP servers, later)
CATCHUP_MORE_S = 10.0       # s behind all along before the catch-up boost lends half the game's cores
HOLD_BOOST_CALM_S = 2.0     # s after an edge hold ends before its boost does (Connection.hold_boost)
CATCHUP_LEND_SHARE = float(os.environ.get("BNGE_CATCHUP_LEND", "0.25"))     # of the game's cores (cpupin.boosted)
CATCHUP_MIN_MS = 8.0        # m/s: only while driving (a parked player's work waits as before)
BOOST_FULL_MAX_S = 300.0    # full priority loading (Server.load_boost's teleportPriority) ends this long after at most

# (rs / rx: the road geotiles' skins and exact variants of the anchor; rail / railcol / tunnel: its road tiles' rail and
# tunnel shapes. Left out until 2026-11-02: kept until the server stopped, GitHub #599's 9 GB session folder)
ANCHOR_TILE_FILES = ("roads", "road", "walks", "walkcol", "deck", "bld", "bldcol", "bldfar", "wire", "veg", "navgraph",
                     "rs", "rx", "rail", "railcol", "tunnel")


class _BoundedLru:
    """The dict interface farfield.horizon_grid() uses for its block-maximum cache, over the server's
    OrderedDict: LRU order, at most `size` entries (only the one horizon worker reads and writes it)."""

    def __init__(self, od, size):
        self.od, self.size = od, size

    def get(self, key):
        v = self.od.get(key)
        if v is not None:
            self.od.move_to_end(key)
        return v

    def __contains__(self, key):
        return key in self.od

    def __setitem__(self, key, value):
        self.od[key] = value
        while len(self.od) > self.size:
            self.od.popitem(last=False)


def route_profile_smooth(z):
    """A route's elevation samples (even steps) as the road climbs them: a running median of 5 (the ground under a
    bridge, a dip or bump of one or two samples where the road crosses a valley or an embankment, drew spikes: the
    user, 2026-10-27, "way to spikey even for a short drive"), then a running mean of 3. Pure."""
    z = np.asarray(z, np.float64)
    n = len(z)
    if n < 5:
        return z
    p = np.r_[z[:1], z[:1], z, z[-1:], z[-1:]]
    med = np.median(np.stack([p[k:k + n] for k in range(5)]), axis=0)
    q = np.r_[med[:1], med, med[-1:]]
    return (q[:-2] + q[1:-1] + q[2:]) / 3.0


def parse_cache_cap(text):
    """--cache-cap: GB >= 0 (0 = unlimited)."""
    try:
        cachemod.cap_bytes(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return float(text)


def parse_tex_px(text):
    """--tex-near / --tex-far: a power of two from 16 to 2048 (render() maps the texture in
    MESH_CELLS equal cells, and the DXT1 mip chain halves it down to 1 x 1)."""
    px = int(text)
    if px < 16 or px > 2048 or px & (px - 1):
        raise argparse.ArgumentTypeError("texture size must be a power of two from 16 to 2048")
    return px


def parse_tex_list(text):
    """--tex: texture px per level from the finest, the last value repeating: '256,256,256,512'."""
    return tuple(parse_tex_px(v) for v in text.split(","))


def parse_radii(text):
    """'150,400' -> (150.0, 400.0): the radii of the exact level and the next one (32 cells: L-2,
    L-1; 64 cells: L-1, L0)."""
    r2, r1 = (float(v) for v in text.split(","))
    if not 0 <= r2 <= r1:
        raise argparse.ArgumentTypeError("need 0 <= exact-level radius <= next-level radius")
    return r2, r1


OPEN_FILES = 65536      # raise_open_files: the soft limit wanted


def raise_open_files():
    """Unix: the soft limit on open files raised to the hard one (at most OPEN_FILES), before any worker starts (they
    inherit it). Every decoded DEM tile a process keeps is a memory-mapped file (dem.py's mirror), and CPython's mmap
    keeps its own descriptor: a few hundred per worker soon after a start, and the default soft limit of 1024 ran out
    on a drive (GitHub #99: "Too many open files": far tiles and map cells failed)."""
    try:
        import resource
    except ImportError:             # (Windows: no such limit)
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = OPEN_FILES if hard == resource.RLIM_INFINITY else min(hard, OPEN_FILES)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            log.info("open files limit: %d -> %d", soft, want)
    except (ValueError, OSError) as e:
        log.warning("open files limit not raised: %s", e)


def prune_own_caches(cache_dir, started=None):
    """The caches outside the cap's MANAGED folders, each cut back to its own limit at start (cache.prune_dir): the map
    data and DEM mirrors, sea carving's raw rasters, compute sharing's kept replies, the road widths' walls. Each grew run after run past its
    limit (35 GB on a Steam Deck under a 20 GB cap, 2026-10-23: players' C: drives filled). Then the older versions'
    folders of the per-cell caches (cache.prune_versions: never read again, never deleted before: 8 GB of
    vegetation/v1..v17 here, 2026-10-28) and the revisit sidecar's compaction (revisit.compact: the files written
    before `started`, without the lines whose zip the prune removed)."""
    try:
        n = mapdatamod.prune_mirror(cache_dir)
        if n:
            log.info("cache: mapdata-mirror cut back: %d layer files' mirrors removed", n)
    except Exception as e:
        log.warning("cache: mapdata-mirror not cut back: %s", e)
    for sub, mb, match in (("seacarve", seacarvemod.RAW_MB, lambda n: n.endswith(".rawb.npy")),
                           ("seacarve", 0, lambda n: n.endswith(".raw.npy")),  # (whole rasters, before 2026-11-05)
                           (os.path.join("dem", "decoded"), demmod.MIRROR_MB,
                            lambda n: n.endswith((".npy", ".f32z"))),
                           ("peercache", peerworkmod.CACHE_BYTES / 1e6, None),
                           ("roadwidth", roadwidthmod.STORE_MB, lambda n: n.endswith(".npy")),
                           (revisitmod.SUBDIR, revisitmod.MAX_MB,       # (the zips and the files kept beside them)
                            lambda n, kept=areasmod.pinned_world(cache_dir): (n.endswith(".zip") or  # noqa: B008 (read once per pass, on purpose)
                                                                              n.endswith(revisitmod.SIDE_SUFFIXES))
                            and n.split(".", 1)[0] not in kept)):
        try:
            n, left = cachemod.prune_dir(os.path.join(cache_dir, sub), mb * 1e6, match=match)
            if n:
                log.info("cache: %s cut back: %d files removed, %.2f GB left", sub, n, left / 1e9)
        except Exception as e:      # a cache's trim must never stop the server
            log.warning("cache: %s not cut back: %s", sub, e)
    for sub, version in (("vegetation", vegetationmod.VERSION), ("seacarve", seacarvemod.VERSION),
                         ("roadwidth", roadwidthmod.WALLS_VERSION)):
        try:
            n, freed = cachemod.prune_versions(os.path.join(cache_dir, sub), f"v{version}")
            if n:
                log.info("cache: %s: %d older version folders removed (%.2f GB)", sub, n, freed / 1e9)
        except Exception as e:
            log.warning("cache: %s's older versions not removed: %s", sub, e)
    try:
        kept = revisitmod.compact(cache_dir, started)
        if kept:
            log.info("cache: revisit sidecar compacted: %d items", kept)
    except Exception as e:
        log.warning("cache: revisit sidecar not compacted: %s", e)
    finally:
        revisitmod.COMPACTED.set()


def _app_modules():
    """apppaths and bmplaunch: beside launcher.py, one folder up (the packaged app: in the bundle)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.append(root)
    import apppaths
    import bmplaunch
    return apppaths, bmplaunch


def saved_cache_dir():
    """The cache folder chosen on the settings page or with the app's --cache-dir (apppaths), or None."""
    try:
        return _app_modules()[0].saved_cache_dir()
    except Exception:
        return None


def app_paths_message(cache_dir, result):
    """app_paths for the settings page: the cache folder in use and the one kept for the next start, the BeamMP
    launcher (the player's choice, else the one found), whether a window is open (picking), and result: the answer to
    the last choice ({what, ok, text})."""
    try:
        ap, bm = _app_modules()
        dd = ap.data_dir()
        saved = ap.saved_cache_dir(dd) or ap.default_cache_dir()
        launcher = bm.remembered(dd) or bm.find(dd) or ""
    except Exception as e:          # (a source copy without them: the page shows the folder in use only)
        log.debug("app paths: %s: %s", type(e).__name__, e)
        saved, launcher = cache_dir, ""
    m = {"type": "app_paths", "cacheDir": os.path.abspath(cache_dir), "cacheNext": os.path.abspath(saved),
         "beammp": launcher}
    if result:
        m["result"] = result
    return m


def pick_app_path(what):
    """The system's window for the cache folder or the BeamMP launcher: apppaths.pick's (state, path)."""
    ap, bm = _app_modules()
    if what == "cache":
        return ap.pick("folder", "BeamNG.Earth: the folder for the download cache",
                       start=ap.saved_cache_dir() or ap.default_cache_dir())
    cur = bm.remembered(ap.data_dir()) or bm.find(ap.data_dir())
    return ap.pick("file", f"BeamNG.Earth: your BeamMP launcher ({bm.EXE})",
                   start=os.path.dirname(cur) if cur else None, name=bm.EXE)


def set_app_path(what, path):
    """Keep a typed or chosen path for the next start: {what, ok, text}."""
    ap, bm = _app_modules()
    if what == "cache":
        d, err = ap.set_cache_dir(path)
        if err:
            return {"what": what, "ok": False, "text": err}
        return {"what": what, "ok": True,
                "text": f"{d or ap.default_cache_dir()}: used from the next start of BeamNG.Earth"}
    exe = bm.remember(ap.data_dir(), path)
    if not exe:
        return {"what": what, "ok": False, "text": f"no {bm.EXE} in {ap.clean(path)}"}
    return {"what": what, "ok": True, "text": f"{exe}: used from the next start of BeamNG.Earth"}



def machine_profile(args, small):
    """The worker processes and analyses for the machine (budget.small): a small one (a Steam Deck, a 16 GB PC) fewer
    worker processes, each with smaller caches (budget.py), the imagery analyses off, where the player left the
    defaults; a desktop its road-tile processes by threads and memory. From 1.7.3 to 1.8.1 the small branch sat under
    the BNGE_FAST_BASE_V override (an A/B switch put in the middle of it), so small machines ran a desktop's processes
    and analyses: a Steam Deck's server at ~7 GB with 6 GB of swap (2026-11-04)."""
    import budget
    global FAST_BASE_V
    if small:
        # ... and "fast" (FAST_BASE_V: prefetched blocks without their photo until used, the chain ahead) from a lower
        # speed: at 87 mph a Steam Deck's server drew a full ground photo for every block (several CPU-s each) and the
        # road collision fell behind the car (2026-10-30)
        FAST_BASE_V = SMALL_FAST_BASE_V
        if args.road_processes == ROAD_PROCESSES:
            args.road_processes = budget.SMALL_ROAD_PROCESSES
        if args.workers == WORKERS:
            args.workers = budget.SMALL_FAR_WORKERS
        if budget.SMALL_ANALYSES_OFF:
            args.deshadow, args.colormatch = "off", "off"
        log.info("small machine (%.1f GB): %d far-tile and %d road-tile processes, smaller caches, imagery analyses "
                 "off (budget.py; BNGE_BUDGET=desktop for the desktop sizes)", budget.total_gb() or 0, args.workers,
                 args.road_processes)
    elif args.road_processes == ROAD_PROCESSES:
        args.road_processes = budget.road_processes()
        log.info("road-tile processes: %d (%d threads, %.0f GB)", args.road_processes, os.cpu_count() or 0,
                 budget.total_gb() or 0)
    # more far-tile processes on a big machine, used while the 3D map is open (Server.far_limit)
    args.far_max = args.workers
    if not small and (budget.total_gb() or 0) >= 24:
        args.far_max = max(args.workers, min((os.cpu_count() or 4) - 4, FAR_MAX_DESKTOP))
        log.info("far-tile processes: %d (%d at once while driving)", args.far_max, min(args.workers, FAR_DRIVE))

FAST_POOL_KEEP = 2          # the saved areas' fast mode: its road processes, one per CPU thread but these ...
FAST_POOL_GB = 2.0          # ... and at most one per this much memory (a road process holds ~1-1.5 GB at work) ...
FAST_POOL_MAX = 24          # ... and at most this many
FAST_POOL_FREE_GB = 1.5     # ... and one per this much of the memory free when it starts (the game, paused, keeps its
                            # own; sized from the total alone, a 32 GB PC with the game and the normal road pool running
                            # could be pushed into swap: PR #773's merge review, 2026-11-06)


def fast_pool_size(cpus=None, gb=None, free=None):
    """The road processes of the saved areas' fast mode on this computer (FAST_POOL_*): by its CPU threads, its memory
    and the memory free now. Pure given its arguments."""
    import budget
    cpus = (os.cpu_count() or 2) if cpus is None else cpus
    gb = (budget.total_gb() or 8.0) if gb is None else gb
    free = budget.free_gb() if free is None else free
    by_free = int(free / FAST_POOL_FREE_GB) if free is not None else FAST_POOL_MAX
    return max(2, min(FAST_POOL_MAX, cpus - FAST_POOL_KEEP, int(gb / FAST_POOL_GB), by_free))
