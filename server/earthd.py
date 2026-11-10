#!/usr/bin/env python3
"""BeamNG.Earth companion server (protocol 4). See docs/V1_3_SPEC.md, docs/V1_2_DESIGN.md
(and V1_1_SPEC.md).

    python3 server/earthd.py [--port 47613] [--user-dir DIR] [--cache-dir DIR] [--far-format cdae|dae]
                             [--imagery usgs,eox2023,eox2016|off] [--near-visual mesh|terrain]
                             [--cells 32|64] [--tex 256,256,256,512] [--block 1024|512] [--horizon]
                             [--deshadow off|lambert|full] [--colormatch off|s2]
                             [--weather on|off] [--gravity on|off] [--offline] [--credits on|off]
                             [--cache-cap GB]

Listens on 127.0.0.1 for the mod, builds terrain-block heightmaps (1024² on a world-locked
2 m lattice, 256 m-aligned centres; in mesh near-visual mode the block is physics only) and a
dynamic-LOD tile field (down to 64 m tiles on the physics lattice in mesh mode) with per-tile
satellite textures, streamed as swap groups, and writes them into
<user-dir>/temp/beamngEarth/<sessionId>/.

Tile layout (docs/V1_4_SPEC.md §3.1, M1): --cells 64 (the default in mesh mode since probe P8
passed) makes every tile 64 x 64 cells, FORMAT 4, with the exact level L-1, hello.cells and
far_tile.tex; --cells 32 (and terrain mode) is the v1.3.1 layout, FORMAT 3.

Terrain block (docs/V1_4_SPEC.md §3.5, M4): --block 512 (mesh mode only; the default there since the
in-game probes P2, P7 and P11 passed; Server() keeps 1024) makes the physics block 512² (collidable [c-512, c+510]) on
a 128 m centre grid, sends hello.blockSize and terrain.size 512, and predicts the mod's 512 follow
table (FOLLOW_RULES, with its fast mode above 150 m/s) for the prefetch.

Horizon (docs/V1_4_SPEC.md §3.6 "Radius", M1h): --horizon (mesh mode only; off by default until the
in-game probe P12 passes) replaces the 80 km square by a disc out to the line-of-sight horizon: the
eye's height above sea level from the pose (camera z in the anchor frame, whose h0 is sea level),
zmax from a coarse DEM pass per anchor (farfield.horizon_grid, in its own worker), RADIUS =
clamp(sqrt(2R'h_eye) + sqrt(2R'zmax_region), 80 km, 1500 km), the per-tile horizon cull with
refraction R' = R/(1-0.13), MAX_LEVEL 10, 256/128 px textures from L8, hello.horizon and
status.radius (the mod's visibleDistance follows it).

Network (docs/research/2026-10-10-rate-limits.md §5): every download goes through one
netfetch.Fetcher (per-host rate limits, priorities, retries, circuit breakers with failover along
the source chains, daily budgets in <cache-dir>/netstats/, one summary line a minute in the log).
--offline serves the disk caches only. While a host is unreachable, or tiles fell back to the
palette, every status text carries a note ("Offline - using cached data", "Degraded: ...").

Download cache (cache.py, docs/ROADMAP.md "Caching"): the raw downloads and region caches under
<cache-dir> are capped (--cache-cap GB, default the last cap the mod's settings page set, else 20; 0 =
unlimited); least recently used files go in a background thread, never what this session used within
the hour, never the gravity model or netstats/. The mod's page shows the usage ('cache' message) and
sets the cap or clears the cache ('cache' from the mod).

Sessions (docs/ROADMAP.md "Robustness" 3): a restarted server keeps the folders of earlier runs that
are less than a day old until the game says (hello / pose `sessions`) which ones its far tiles still
use: those are adopted and deleted (rate-limited) once no pose names them any more; the others go
SESSION_ORPHAN_S after nobody claimed them. The mod resyncs at its current origin meanwhile, so the
tiles on screen never lose their files.
"""

import os
# Heavy numpy work must not compete with the game's physics threads (§4.5): one BLAS/OpenMP
# thread per worker. Set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import errno  # noqa: E402
import glob  # noqa: E402
import functools  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import sys  # noqa: E402
import shutil  # noqa: E402
import collections  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict, deque  # noqa: E402
import concurrent.futures  # noqa: E402
import multiprocessing  # noqa: E402
from runtimeprofiler import ThreadPoolExecutor  # noqa: E402
from dataclasses import dataclass, field, replace  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import blockworker  # noqa: E402
import cache as cachemod  # noqa: E402
import cdae  # noqa: E402
import colormatch as colormatchmod  # noqa: E402
import credits as creditsmod  # noqa: E402
import secrossings  # noqa: E402
import covermap as covermapmod  # noqa: E402
import areas as areasmod  # noqa: E402
import surface as surfacemod  # noqa: E402
import navgraph as navgraphmod  # noqa: E402
import distanttraffic          # noqa: E402
import prewarm as prewarmmod  # noqa: E402
import parking as parkingmod  # noqa: E402
import gasstations as gasstationsmod  # noqa: E402
import roads as roadsmod  # noqa: E402
import privacyregion as privacyregionmod
import perfstats  # noqa: E402
import rails as railsmod  # noqa: E402
import randomplace as randomplacemod  # noqa: E402
import urgency as urgencymod  # noqa: E402
import placepreview as placepreviewmod  # noqa: E402
import placenames as placenamesmod  # noqa: E402
import maplabels as maplabelsmod  # noqa: E402
import runtimeprofiler
import streamleases
import pools as poolsmod  # noqa: E402
import plancache as plancachemod  # noqa: E402
import grading as gradingmod  # noqa: E402
import roadmesh as roadmeshmod  # noqa: E402
import roadwidth as roadwidthmod  # noqa: E402
import revisit as revisitmod  # noqa: E402
import signstream  # noqa: E402
import dirsigns as dirsignsmod  # noqa: E402
import mapdata as mapdatamod  # noqa: E402
import dem as demmod  # noqa: E402
import mapcells  # noqa: E402
import pacer  # noqa: E402
import peerwork as peerworkmod  # noqa: E402
import deshadow as deshadowmod  # noqa: E402
import farfield  # noqa: E402
import bc7  # noqa: E402
import terraincache  # noqa: E402
import terrainpatch as terrainpatchmod  # noqa: E402
import geodesy  # noqa: E402
import gravity as gravitymod  # noqa: E402
import airportgrade as airportgrademod  # noqa: E402
import airports as airportsmod  # noqa: E402
import seacarve as seacarvemod  # noqa: E402
import seatiles as seatilesmod  # noqa: E402
import vegetation as vegetationmod
import buildings as buildingsmod  # noqa: E402
import rockface as rockfacemod  # noqa: E402
import geotiles as geotilesmod  # noqa: E402
import riverlines as riverlinesmod  # noqa: E402
import groundahead as groundaheadmod  # noqa: E402
import routeahead as routeaheadmod  # noqa: E402
import furniture as furnituremod  # noqa: E402
import development as developmentmod  # noqa: E402
import townentry as townentrymod  # noqa: E402     (PR #597's town entry and exit signs)
import heightmap  # noqa: E402
import staging  # noqa: E402
import imagery as imagerymod  # noqa: E402
import matslots as matslotsmod  # noqa: E402
import maptiles as maptilesmod  # noqa: E402
import geocode as geocodemod  # noqa: E402
import routing as routingmod  # noqa: E402
import updates as updatesmod  # noqa: E402
import cpupin  # noqa: E402
import netstate as netstatemod  # noqa: E402
import netfetch  # noqa: E402
import protocol  # noqa: E402
import weather as weathermod  # noqa: E402
import earthbase
from earthbase import (  # (re-exported: callers use earthd.<name>)
    AHEAD_CELLS,  # noqa: F401
    AHEAD_LAYERS,  # noqa: F401
    AHEAD_MARGIN_M,  # noqa: F401
    AHEAD_MAX_M,  # noqa: F401
    AHEAD_MIN_V,  # noqa: F401
    AHEAD_MOVE_M,  # noqa: F401
    AHEAD_S,  # noqa: F401
    ANCHOR_CENTRE_MAX,  # noqa: F401
    ANCHOR_PROGRESS_S,  # noqa: F401
    ANCHOR_TILE_FILES,  # noqa: F401
    Anchor,  # noqa: F401
    BAKE_FETCH,  # noqa: F401
    BASE_AO,  # noqa: F401
    BASE_AO_VFS,  # noqa: F401
    BASE_FAST_WAIT_S,  # noqa: F401
    BASE_NO_STANDIN_MAX_S,  # noqa: F401
    BASE_PX,  # noqa: F401
    BASE_RENDER_PX,  # noqa: F401
    BASE_SHIFT_M,  # noqa: F401
    BASE_SIZES,  # noqa: F401
    BASE_WAIT_S,  # noqa: F401
    BLOCK_DEFAULT,  # noqa: F401
    BLOCK_DRAWN,  # noqa: F401
    BLOCK_SIZES,  # noqa: F401
    BLOCK_SURVEY_M,  # noqa: F401
    BOOST_FULL_MAX_S,  # noqa: F401
    BOOST_GRACE_S,  # noqa: F401
    BOOST_MAX_S,  # noqa: F401
    BOOST_MOVE_MS,  # noqa: F401
    CATCHUP_AFTER_S,  # noqa: F401
    CATCHUP_CALM_S,  # noqa: F401
    CATCHUP_LEND_SHARE,  # noqa: F401
    CATCHUP_MIN_MS,  # noqa: F401
    CATCHUP_MORE_S,  # noqa: F401
    CATCHUP_ROADS_M,  # noqa: F401
    CDAE_ROOT,  # noqa: F401
    CELLS_DEFAULT,  # noqa: F401
    CLI_BLOCK_MESH,  # noqa: F401
    CLI_CELLS_MESH,  # noqa: F401
    DELETE_DRAIN_S,  # noqa: F401
    DELETE_INTERVAL,  # noqa: F401
    DELETE_MAX_RATE,  # noqa: F401
    DELTA_MARGIN,  # noqa: F401
    DETAIL_ROAD_LIFT,  # noqa: F401
    DynGate,  # noqa: F401
    ELEVATION_CREDIT,  # noqa: F401
    EYE_ABOVE_PLAYER,  # noqa: F401
    EYE_LOD_MIN_M,  # noqa: F401
    EYE_STEP,  # noqa: F401
    EYE_STEP_REL,  # noqa: F401
    FAR_DRIVE,  # noqa: F401
    FAR_MAX_DESKTOP,  # noqa: F401
    FAR_TRACE,  # noqa: F401
    FAST_KEEP_S,  # noqa: F401
    FAST_POOL_FREE_GB,  # noqa: F401
    FAST_POOL_GB,  # noqa: F401
    FAST_POOL_KEEP,  # noqa: F401
    FAST_POOL_MAX,  # noqa: F401
    FIN_WORKER,  # noqa: F401
    FIN_WORKER_MIN_TRIES,  # noqa: F401
    FIN_WORKER_PROBE,  # noqa: F401
    FIN_WORKER_SHARE,  # noqa: F401
    FIN_WORKER_WINDOW,  # noqa: F401
    FLY_TEX_EYE_M,  # noqa: F401
    FOCI_MAX,  # noqa: F401
    FOCI_RANGE_M,  # noqa: F401
    FOCI_R_MAX_M,  # noqa: F401
    FOLLOW_LEAD,  # noqa: F401
    FOLLOW_PREPARE,  # noqa: F401
    FOLLOW_REACH,  # noqa: F401
    FOLLOW_RULES,  # noqa: F401
    FRAME_S,  # noqa: F401
    FollowRule,  # noqa: F401
    GEO_FILES_GRACE_S,  # noqa: F401
    GEO_LIST_BATCH_S,  # noqa: F401
    GEO_TRACKERS,  # noqa: F401
    GLOBAL_AID,  # noqa: F401
    GRAPHICS_SCALE,  # noqa: F401
    GROUND_RESEND_M,  # noqa: F401
    GROUND_RESEND_S,  # noqa: F401
    GROUND_WAIT_S,  # noqa: F401
    HM_BUSY_TILE_WORKERS,  # noqa: F401
    HM_CACHE,  # noqa: F401
    HM_KEEP_COMMITTED,  # noqa: F401
    HM_KEEP_SENT,  # noqa: F401
    HM_REQUEST_THREADS,  # noqa: F401
    HM_THREADS,  # noqa: F401
    HOLD_BOOST_CALM_S,  # noqa: F401
    HORIZON_BLOCK_CACHE,  # noqa: F401
    HORIZON_RETRY,  # noqa: F401
    IMAGERY_PREFETCH,  # noqa: F401
    IMG_WARM_MARGIN,  # noqa: F401
    LAG_REPORT_S,  # noqa: F401
    LAG_STALL_S,  # noqa: F401
    LAG_TICK_S,  # noqa: F401
    LANE_PREFETCH,  # noqa: F401
    LANE_REQUEST,  # noqa: F401
    LEASE_GRACE_S,  # noqa: F401
    LOOP_RAISE,  # noqa: F401
    MAP_BLD_FAR_EYE,  # noqa: F401
    MAP_BLD_FAR_MAX_M,  # noqa: F401
    MAP_BLD_SKY_EYE,  # noqa: F401
    MAP_BLD_SKY_MAX_M,  # noqa: F401
    MAP_BUILDINGS_EYE_M,  # noqa: F401
    MAP_DETAIL_EYE_M,  # noqa: F401
    MAP_FOCUS_MAX_M,  # noqa: F401
    MAP_HIGH_EYE_M,  # noqa: F401
    MAP_LOD_SCALE,  # noqa: F401
    MAP_OUT_OF_VIEW_K,  # noqa: F401
    MAP_RAMP_START,  # noqa: F401
    MAP_RAMP_STEP,  # noqa: F401
    MAP_RETEX_BOOST,  # noqa: F401
    MAP_SCREEN_H,  # noqa: F401
    MAP_TEX_MAX_LEVEL,  # noqa: F401
    MAP_TEX_MIN_PX,  # noqa: F401
    MAP_TEX_OFF_PX,  # noqa: F401
    MAP_TEX_OVERSAMPLE,  # noqa: F401
    MAP_TEX_PX,  # noqa: F401
    MAP_TEX_UPGRADE_S,  # noqa: F401
    MAP_ZOOM_RAMP_K,  # noqa: F401
    MESH_LAYOUT,  # noqa: F401
    NAV_LEAD_MAX_M,  # noqa: F401
    NAV_LEAD_S,  # noqa: F401
    NAV_WAIT_S,  # noqa: F401
    NEAR_VISUALS,  # noqa: F401
    NET_TICK_S,  # noqa: F401
    NICE,  # noqa: F401
    NICE_FAR,  # noqa: F401
    NICE_FAR_BUILDINGS,  # noqa: F401
    NICE_ROADS,  # noqa: F401
    OPEN_FILES,  # noqa: F401
    OSM_FUEL_THREADS,  # noqa: F401
    OSM_FUEL_WARM_TRIES,  # noqa: F401
    OSM_FUEL_WARM_WAIT_S,  # noqa: F401
    PADS_REFRESH_MS,  # noqa: F401
    PADS_SETTLE_S,  # noqa: F401
    PATCH_KEY,  # noqa: F401
    PERF_LOG_CHARS,  # noqa: F401
    PERF_LOG_LINES,  # noqa: F401
    PHOTO_PROCESS_MIN_CPUS,  # noqa: F401
    PHOTO_SKIP_V,  # noqa: F401
    PIECE_PARTS,  # noqa: F401
    PLACE_PREVIEWS_MAX,  # noqa: F401
    PREFETCHED_MAX,  # noqa: F401
    PREFETCH_AGAIN_S,  # noqa: F401
    PREFETCH_CHAIN,  # noqa: F401
    PREFETCH_PARALLEL,  # noqa: F401
    PREFETCH_PROCS_CPUS,  # noqa: F401
    PREFETCH_PROCS_GB,  # noqa: F401
    PREFETCH_START,  # noqa: F401
    PROGRESS_S,  # noqa: F401
    PROTOCOL,  # noqa: F401
    PROVISIONAL_ELSEWHERE,  # noqa: F401
    PROVISIONAL_FIRST,  # noqa: F401
    PROVISIONAL_FOLLOW_S,  # noqa: F401
    PROVISIONAL_KEEP_S,  # noqa: F401
    PROVISIONAL_RUNNING_S,  # noqa: F401
    PRUNE_DELAY,  # noqa: F401
    PRUNE_RECHECK,  # noqa: F401
    QUICK_BASE_PX,  # noqa: F401
    QUICK_MADE_KEPT,  # noqa: F401
    QUICK_MIN_COVER,  # noqa: F401
    QUICK_SHARE_WAIT_S,  # noqa: F401
    QUICK_SYNC_MAX_V,  # noqa: F401
    RADIUS_REPORT_REL,  # noqa: F401
    READY_SET_M,  # noqa: F401
    READY_STEPS,  # noqa: F401
    RECENT_FULL,  # noqa: F401
    RETEX_INFLIGHT,  # noqa: F401
    RETEX_RATE,  # noqa: F401
    RETEX_REPEAT,  # noqa: F401
    RETIRED_KEEP,  # noqa: F401
    ROAD_GEO_SLOW_S,  # noqa: F401
    ROAD_PROCESSES,  # noqa: F401
    ROAD_TILE_TIMEOUT_S,  # noqa: F401
    ROUTE_AHEAD_S,  # noqa: F401
    ROUTE_ALIGN,  # noqa: F401
    ROUTE_DOWNLOAD_MODES,  # noqa: F401
    ROUTE_IMG_TRIES,  # noqa: F401
    ROUTE_NEAR_M,  # noqa: F401
    ROUTE_PREDICT_MAX_M,  # noqa: F401
    ROUTE_RETRY_S,  # noqa: F401
    ROUTE_STEP_M,  # noqa: F401
    ROUTE_WHOLE_MAX_M,  # noqa: F401
    SEA_CHECK_S,  # noqa: F401
    SEND_BUFFER_MAX,  # noqa: F401
    SERVER_LIST_URL,  # noqa: F401
    SERVER_VERSION,  # noqa: F401
    SESSION_HEARTBEAT_S,  # noqa: F401
    SESSION_MARKER,  # noqa: F401
    SESSION_ORPHAN_S,  # noqa: F401
    SESSION_STALE_S,  # noqa: F401
    SESSION_SWEEP_S,  # noqa: F401
    SLOW_BASE_S,  # noqa: F401
    SLOW_SELECT_S,  # noqa: F401
    SMALL_FAST_BASE_V,  # noqa: F401
    STANDIN_WAIT_S,  # noqa: F401
    STATUS_INTERVAL,  # noqa: F401
    SURVEY_HALF,  # noqa: F401
    SURVEY_N,  # noqa: F401
    SURVEY_PERCENTILE,  # noqa: F401
    SURVEY_WINDOW,  # noqa: F401
    SURVEY_ZOOM,  # noqa: F401
    SWITCH_MS,  # noqa: F401
    TELEMETRY_CONSENT,  # noqa: F401
    TELEPORT_PAUSE_M,  # noqa: F401
    TELEPORT_PAUSE_MAX_S,  # noqa: F401
    TERRAIN_BLOCK,  # noqa: F401
    TERRAIN_STUCK_S,  # noqa: F401
    TEX_FAR,  # noqa: F401
    TEX_NEAR,  # noqa: F401
    TILE_CACHE,  # noqa: F401
    TRACE_S,  # noqa: F401
    TRACE_TOP,  # noqa: F401
    TRASH_DIR,  # noqa: F401
    TUNNEL_FLOOR_UNDER_M,  # noqa: F401
    TUNNEL_NEAR_M,  # noqa: F401
    TURBINE_DROP_M,  # noqa: F401
    TURBINE_GEO,  # noqa: F401
    TURBINE_PER_POSE,  # noqa: F401
    TURBINE_SEND_M,  # noqa: F401
    TURBINE_Z,  # noqa: F401
    TileInfo,  # noqa: F401
    URGENT_ELSEWHERE,  # noqa: F401
    VEG_GEO_BATCH,  # noqa: F401
    VEG_GEO_PER_POSE,  # noqa: F401
    VIEW_ANCHOR_Q,  # noqa: F401
    VIEW_AWAY_M,  # noqa: F401
    VIEW_CAP_MIN_M,  # noqa: F401
    WARM_AROUND,  # noqa: F401
    WARM_STEPS_S,  # noqa: F401
    WATERGUARD_VERSION,  # noqa: F401
    WHERE_PLACE_RETRY_S,  # noqa: F401
    WHERE_PLACE_TRIES,  # noqa: F401
    WORKERS,  # noqa: F401
    WORLD_WARM_CONC,  # noqa: F401
    WORLD_WARM_Z,  # noqa: F401
    _BoundedLru,  # noqa: F401
    _NO_PERF,  # noqa: F401
    _app_modules,  # noqa: F401
    _num,  # noqa: F401
    _perf,  # noqa: F401
    app_paths_message,  # noqa: F401
    base_ao_vfs,  # noqa: F401
    fast_pool_size,  # noqa: F401
    game_running,  # noqa: F401
    log,  # noqa: F401
    machine_profile,  # noqa: F401
    osm_fuel_cells,  # noqa: F401
    osm_fuel_tiles,  # noqa: F401
    parse_cache_cap,  # noqa: F401
    parse_foci,  # noqa: F401
    parse_radii,  # noqa: F401
    parse_tex_list,  # noqa: F401
    parse_tex_px,  # noqa: F401
    pick_app_path,  # noqa: F401
    predict_centre,  # noqa: F401
    predict_chain,  # noqa: F401
    predict_request,  # noqa: F401
    predict_route_chain,  # noqa: F401
    prefetch_procs,  # noqa: F401
    prune_own_caches,  # noqa: F401
    raise_open_files,  # noqa: F401
    resolve_block,  # noqa: F401
    resolve_cells,  # noqa: F401
    resolve_colormatch,  # noqa: F401
    resolve_horizon,  # noqa: F401
    road_tracker,  # noqa: F401
    route_profile_smooth,  # noqa: F401
    saved_cache_dir,  # noqa: F401
    set_app_path,  # noqa: F401
    survey_q,  # noqa: F401
)
from srvhouse import _HousekeepingMixin
from srvbuild import _BuildMixin
from srvterrain import _TerrainMixin
from srvfar import _FarMixin
from connmap import _MapMixin
from connterrain import _AnchorMixin
from connstream import _StreamMixin


class Server(_HousekeepingMixin, _BuildMixin, _TerrainMixin, _FarMixin):
    def __init__(self, user_dir, cache_dir, tile_workers=3, kfun=None,
                 hysteresis=farfield.HYSTERESIS, radius=farfield.RADIUS, far_format="cdae",
                 imagery=imagerymod.DEFAULT_CHAIN, near_visual="mesh", tex_near=TEX_NEAR, tex_far=TEX_FAR,
                 tone=1.0, near_radii=None, status_interval=STATUS_INTERVAL, cells=CELLS_DEFAULT, tex=None,
                 block=None, horizon=False, deshadow="off", weather=None, gravity=None, net=None,
                 colormatch="off", credits=None, cache=None, retex_rate=RETEX_RATE, surface=None,
                 mapdata=None, grading=False, road_widths=False, block_pools=(), sidewalks=True, buildings=False, traffic=False, far_pool=None,
                 road_pool=None, airports_db=None, sea=None, vegetation=None, leaf_on=True, season_mode="off",
                 far_bld_pool=None, gas_stations=False, regional_poles="off"):
        """kfun: K(L) for every level (default: the layout's K table); near_radii: the min-level
        radii (default: the layout's); cells: 32 (v1.3.1) or 64 (M1, mesh mode only); tex: texture
        px per level from the finest (default: tex_near, tex_near, tex_far with 32 cells,
        farfield.TEX_DEFAULT[64] with 64); block: terrain block samples per edge, 1024 (v1.3.1) or
        512 (M4, mesh mode only: a 128 m centre grid and the 512 follow table); horizon: M1h, the
        far field out to the line-of-sight horizon (mesh mode only; `radius` is then unused); net:
        the netfetch.Fetcher every download goes through (main_async passes the command line's; None =
        one of its own, no statistics file); colormatch: --colormatch (off, or s2: the colour-seam
        feather on the USGS tiles); cache: the download cache's cache.Cache (last use per file, the
        size cap, the page's 'cache' messages; None = no bookkeeping and no 'cache' message, Server()'s
        default); retex_rate: --retex-rate (live retextures per second, 0 = none)."""
        if os.environ.get("BNGE_WORLD_BUNDLE") and not (grading and mapdata is not None):
            from worldbundle import BundleError
            raise BundleError("Pinned road plans require roads and grading enabled")
        self.sid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        self.user_dir = user_dir           # the game's user folder (detail mode fills its terrain texture cache)
        self.cache_dir = cache_dir         # (here already: _scan_sessions moves old sessions into its trash)
        root = self.root = os.path.join(user_dir, "temp", "beamngEarth")
        self.cdae_root = os.path.join(user_dir, *CDAE_ROOT)
        # earlier runs' session folders a game may still show: sid -> monotonic time found / released
        # by a connection (docs/ROADMAP.md "Robustness" 3; claim_sessions, sweep_sessions)
        self.orphans = {}
        self.orphan_s = SESSION_ORPHAN_S
        self.heartbeat_at = time.monotonic()
        self._scan_sessions()
        self.out_dir = os.path.join(root, self.sid)
        self.cdae_dir = os.path.join(user_dir, *CDAE_ROOT, self.sid.lower())
        self.vfs_dir = f"/temp/beamngEarth/{self.sid}"
        self.prewarm_vfs = None             # the materials' prewarm shape (prewarm.py), written at the first hello
        self.matslots = matslotsmod.MatSlots(self.sid[-4:], held=self.slot_held)    # pooled names and shape files
        self.reslot_serial = itertools.count(1)
        self.slot_serial = itertools.count(1)      # far_tile slotSerial (matslots.py)
        self.base_px = BASE_PX              # detail mode: the ground photo's size (px), the mod's graphics.photoPx
        self.lod_px = farfield.SSE_TAU_PX   # screen-space LOD: the largest split pop allowed on screen (px; 0: off)
        self.update_check = False       # ask GitHub for a newer version after a hello (main_async turns it on; tests
                                        # never reach GitHub: its small daily budget ran out over a day of test runs)
        os.makedirs(self.out_dir, exist_ok=True)
        open(os.path.join(self.out_dir, SESSION_MARKER), "w").close()
        if far_format not in farfield.FORMATS:
            raise ValueError(f"unknown far-tile format {far_format!r}")
        self.far_format = far_format
        if near_visual not in NEAR_VISUALS:
            raise ValueError(f"unknown near-visual mode {near_visual!r}")
        # mesh: the TerrainBlock is physics only, the ground is drawn by the finest tiles (§B)
        self.near_visual = near_visual
        # the near tiles' asphalt layer (mesh mode: the tiles are the ground seen; surface.road_mask)
        self.road_masks = roadsmod.ROADS_IN_GROUND and near_visual not in BLOCK_DRAWN
        if tex is None and cells == 32:
            tex = (tex_near, tex_near, tex_far)         # the v1.3.1 rule: the two finest levels are near
        # Every layout-dependent rule reads self.layout (terrain mode: frozen at 32 cells)
        if block is None:               # BLOCK_DEFAULT; terrain mode: its frozen 2 km block
            block = BLOCK_DEFAULT if near_visual in MESH_LAYOUT else TERRAIN_BLOCK
        self.layout = farfield.Layout(cells, block, near_visual in MESH_LAYOUT, tex)
        self.follow = FOLLOW_RULES[self.layout.block * heightmap.SQUARE]     # the mod's terrain-follow table for its block
        self.min_level = self.layout.min_level
        # the finest level under a drawn block's forced box (detail: L-1, whose 128 m edges meet the 512²
        # block's 128 m-grid edges; terrain: L0 under the 256 m-grid 1024² block)
        self.forced_level = max(self.min_level, -1) if near_visual == "detail" else 0
        self.near_radii = dict(self.layout.near_radii if near_radii is None else near_radii)
        self.tile_cache = TILE_CACHE[cells]
        self.net = net if net is not None else netfetch.Fetcher()
        self.dem = demmod.Dem(os.path.join(cache_dir, "dem"), fetcher=self.net)
        import railways as _railways    # (coarse elevation: rail decks raised over the roads under them)
        _railways.COARSE_AT = self.dem.coarse_at
        # (and no separation in the ground under a rail bridge: railways.flat_rail_crossing)
        _railways.RAW_ELEV = lambda la, lo, _d=self.dem: _d.sample(la, lo, gradingmod.PROFILE_ZOOM)
        # runways graded in (--airports on: main() loads the OurAirports copy before the worker processes)
        self.airports = airportgrademod.AirportGrader(airports_db, self.dem) if airports_db is not None else None
        self.airports_db = airports_db  # (also the road rules' country: navgraph.country_near)
        self.regional_poles = regional_poles
        if regional_poles != "off":
            import countrybounds
            if isinstance(countrybounds.bundled(), countrybounds.Absent):    # (fetched from BeamNG.Earth-data once)
                log.warning("regional poles: no border data reachable yet: the stock scenery until it is")
            if regional_poles in ("fr-varied", "europe"):
                import townsigns
                import directionsigns
                townsigns.bundled()
                directionsigns.bundled()    # fail visibly before generating tiles if the bundle is missing/corrupt
            log.info("regional poles: %s (country polygons at each pole; data %s; other/unknown countries keep stock)",
                     regional_poles, countrybounds.signature())
        self.sea = sea              # seacarve.SeaCarver (--sea on): the sea floor carved, the water tiles
        self.veg = vegetation       # vegetation.Vegetation (--trees on): trees, bushes, rocks per cell
        if sea is not None and sea.dem is None:
            sea.dem = self.dem      # the lakes' levels
        if vegetation is not None and vegetation.dem is None:
            vegetation.dem = self.dem   # rocks on bare ground by its slope (vegetation.ROCK_SLOPE)
        # Satellite imagery (§A); None = --imagery off (v3-style palette tiles).
        # De-shadowing of the USGS tiles (--deshadow lambert|full; off = the imagery as served).
        self.deshadow = None
        if deshadow != "off" and imagery and deshadowmod.SOURCE in [getattr(s, "name", s) for s in imagery]:
            self.deshadow = deshadowmod.Deshadow(cache_dir, self.dem, deshadow)
            self.deshadow.add_listener(self._ds_listener)
        # Colour seams of the USGS mosaic (--colormatch s2; off = no object, textures as before).
        self.tex_modes = (deshadow, colormatch)     # (the revisit cache's key of a far texture: tex_key)
        self.colormatch = None
        if colormatch != "off" and imagery and colormatchmod.SOURCE in [getattr(s, "name", s) for s in imagery]:
            self.colormatch = colormatchmod.ColorMatch(cache_dir, colormatch)
            self.colormatch.add_listener(self._cm_listener)
        self.loop = None                    # the event loop (set by tile()), for the colormatch listener
        self.cm_waiting = {}                # region -> {(anchorId, key)} rendered without its correction
        self.tex_rev = {}                   # (anchorId, key) -> texture revision of its next build
        # live retexture (far_retex): stale tiles to re-render while shown, the renders running, the
        # earliest start of the next one, its timer, and the counts for the log / tests
        self.retex_rate = retex_rate
        self.retex_inflight = RETEX_INFLIGHT
        self.retex_todo = set()
        self.retex_busy = 0
        self._retex_next = 0.0
        self._retex_timer = None
        self.retex_stats = {"started": 0, "sent": 0, "dropped": 0}
        self.tone = tone
        self.imagery = (imagerymod.Imagery(os.path.join(cache_dir, "imagery"), imagery, tone=tone,
                                           deshadow=self.deshadow, colormatch=self.colormatch, fetcher=self.net,
                                           leaf_on=leaf_on, season_mode=season_mode)
                        if imagery else None)
        if self.imagery is not None:        # (a quick texture's photos downloaded: its live re-render, _cm_ready)
            self.imagery.add_img_listener(self._cm_listener)
        self.imagery_failures = 0           # tiles that fell back to the palette (network trouble)
        self.imagery_worker = 0             # far textures the far-tile processes rendered (blockworker.far_tile)
        self.fin_worker_hits = deque(maxlen=FIN_WORKER_WINDOW)     # (fin_worker_worth: recent tries found them?)
        self.fin_worker_tries = 0
        self.base_worker = 0                # base photos the block processes rendered (blockworker.base_photo)
        self.map_port = None                # the in-game map's tile endpoint (maptiles.py), hello.mapPort
        self.feed = None                    # the companion feed (companionfeed.py), when on
        self.map_tiles = None               # its MapTiles: also written as files on request (map_tiles message)
        self.map_files = None               # the maptiles.TileFiles writing them (made by the first request)
        # (anchorId, key) of cached tiles built with the palette because the imagery failed; shown in
        # the status note (re-texturing them once the host is back is a ROADMAP follow-up)
        self.degraded = set()
        # Heightmap workers: with the block processes they mostly wait on a process, and with two a
        # request queued behind a prefetch and an earlier request (3.5 s at 90 m/s, 2026-10-12)
        self.hm_pool = ThreadPoolExecutor(HM_THREADS, thread_name_prefix="heightmap")
        self.hm_proc = {}               # (aid, cx, cy) -> its build's future in a block process (provisional_follow)
        # the blocks the mod asks for (the request lane): their own threads, never behind the prefetch's and the
        # rebuilds' (a first visit's requested blocks waited 20-38 s for a heightmap thread, nyc_speed.py --cold
        # down Uptown at 100 mph, 2026-11-02)
        self.hm_req_pool = ThreadPoolExecutor(HM_REQUEST_THREADS, thread_name_prefix="heightmap-req")
        self.hm_here, self.hm_here_lock = 0, threading.Lock()  # blocks building in this process (far tiles wait)
        # a block's base photo rendered beside its heights (they do not depend on each other): after them it added
        # 2.3 s to a 12 s Baltimore block on a Steam Deck (2026-10-20)
        self.base_pool = ThreadPoolExecutor(HM_THREADS, thread_name_prefix="base-photo")
        # the photos a player waits for (a teleport's first block, the request lane's): never queued behind the other
        # blocks' (each holds a base-photo thread until its block process renders it; a teleport's photo waited
        # ~20 s behind the old place's on a Deck, 2026-10-23)
        self.urgent_base_pool = ThreadPoolExecutor(2, thread_name_prefix="base-urgent")
        self.geo_list_pool = None           # (geo_dropped: the session folder listed off the event loop)
        self.prov_pool = ThreadPoolExecutor(2, thread_name_prefix="provisional")    # follow blocks sent provisional
        self.prov_info = {}             # (aid, cx, cy) -> a provisional follow block's info (until the graded one)
        self.ground_pool = ThreadPoolExecutor(1, thread_name_prefix="ground-ahead")   # groundahead.py
        # the water plans (Connection.sea_tiles) apart: a plan takes seconds and holds while a teleport's block is
        # built (pacer.hold), and the ground ahead (a few ms) queued behind them: 2.2 s at 275 km/h (2026-11-07)
        self.sea_pool = ThreadPoolExecutor(1, thread_name_prefix="sea-plan")
        # the world map's requests (GPS route, address search, map tiles): their own threads, so a teleport's tile
        # builds on the default pool do not hold them back (a route took 30 s after one, 2026-10-14)
        self.ui_pool = ThreadPoolExecutor(2, thread_name_prefix="map-ui")
        # the search box's own: a search can take seconds, and must not hold up a route or the map's tiles
        self.search_pool = ThreadPoolExecutor(4, thread_name_prefix="map-search")
        # the status bar's place names (placenames.py): a thread of their own, a region's first fetch takes seconds
        self.place_pool = ThreadPoolExecutor(1, thread_name_prefix="place-names")
        self._place_names_lock = threading.Lock()
        # (a thread per helper slot too: a far thread waits for its tile's builder, here or on a helper)
        self.cache_dir = cache_dir
        self.picking = None          # the settings page's file window open (apppaths.pick): for which path
        revisitmod.CACHE_DIR = cache_dir
        self.refetched = 0              # far textures whose missing photo tiles were fetched here (make_tile)
        # self.share_lock, self.group_helper = threading.Lock(), None    # (share_start: shared world building, commented out)
        try:                            # (BeamMP sharing v2: a session's mark left by an earlier run; share_hub sets it)
            os.remove(os.path.join(cache_dir, "share", "active"))
        except (OSError, TypeError):
            pass
        self.tile_pool = ThreadPoolExecutor(tile_workers + (peerworkmod.MAX_SLOTS if hasattr(far_pool, "remote_slots")
                                                            else 0), thread_name_prefix="far")
        # (the 3D map's big leap, the user 2026-11-07: "big leaps in world gen speed": every far worker while the map
        # is open, FAR_DRIVE of them while driving: tile_workers is the pool's whole size, main() sizes it)
        self.far_max = tile_workers
        self.far_drive = min(tile_workers, FAR_DRIVE)
        self.far_gate = DynGate(lambda: self.far_limit())
        # Road plans (grading) of the map cells ahead of the player, made before a block needs them: a
        # cell's first plans (and street widths) take 2-10 s, more than a block move allows at speed
        # One thread takes the most urgent cell of the latest pose's list each time (not a FIFO: cells
        # queued while parked must not hold up the ones on the path once the player drives off)
        self.warm_want = []             # (grader, cell key), most urgent first (replaced every pose)
        # The follow / prefetch blocks build in processes of their own (blockworker.py: cores of their own,
        # not this interpreter's lock): (ProcessPoolExecutor of one, its queue of the road cells ahead) per
        # lane, LANE_REQUEST and LANE_PREFETCH, so a block the mod asks for never waits behind a prefetch
        # (one shared process: answers up to 4 s at 90 m/s, 2026-10-12). Empty: built in hm_pool's threads.
        self.block_pools = list(block_pools)
        # Far-tile geometry in worker processes too (blockworker.far_geometry; the texture stays here): a
        # ProcessPoolExecutor of tile_workers, or None (built in tile_pool's threads)
        self.far_pool = far_pool
        self.road_pool = road_pool          # road tiles in processes (blockworker.roads_tile), else threads
        self._veg_jobs, self._veg_job_lock = {}, threading.Lock()      # (_veg_geo_job)
        # the highway signs (signstream.py) in the same processes: no process of their own, their map cells warm there
        # (at most signstream.WORKERS a connection, only for road tiles sent)
        self.sign_pool = road_pool
        # far building tiles in a process of their own: in the road pool (first come, first served) a teleport's
        # hundreds of them queued ahead of the road tiles under the car, which drove on bare ground for minutes
        # (Sinaia 2026-10-15); falls back to the road pool
        self.far_bld_pool = far_bld_pool or road_pool
        self.warm_cv = threading.Condition()
        self.warm_thread = None
        self.tile_workers = tile_workers
        self.delta_pool = ThreadPoolExecutor(1, thread_name_prefix="deltas")   # swap group deltas (emit_groups)
        self.kfun, self.hysteresis, self.radius = kfun or self.layout.k, hysteresis, radius
        self.status_interval = status_interval
        # M1h: the line-of-sight far field. One worker builds the anchors' zmax grids (coarse DEM
        # passes) so neither a heightmap nor a tile ever waits behind one.
        if horizon and near_visual not in MESH_LAYOUT:
            raise ValueError("horizon mode needs the mesh (or detail) near-visual mode")
        self.horizon = bool(horizon)
        self.horizon_pool = ThreadPoolExecutor(1, thread_name_prefix="horizon") if self.horizon else None
        self.ahead_pool = ThreadPoolExecutor(1, thread_name_prefix="download-ahead")   # download_ahead
        self.ahead_busy = False
        self.ahead_last = None          # (anchor id, x, y) of the last round
        self.ahead_stats = {"dem": 0, "cells": 0, "deferred": 0}
        self.route_pool = ThreadPoolExecutor(1, thread_name_prefix="route-download")   # Connection.route_download
        self.horizon_grids = {}             # anchorId -> farfield.HorizonGrid
        self.horizon_inflight = {}          # anchorId -> (future, half-size)
        self.horizon_failed = {}            # anchorId -> monotonic time of the last failed build
        self.horizon_blocks = OrderedDict()     # (zoom, x, y, block) -> block maxima, LRU
        self.anchors = {}
        self._next_anchor = 1
        self.connections = set()
        self.stream_leases = streamleases.Registry()
        # Heightmaps: (anchorId, cx, cy) -> info, LRU order.
        self.hm = OrderedDict()
        self.hm_inflight = {}
        self.hm_stage = {}              # terrain key -> what its build waits on now (the stuck log, prepare_terrain)
        self.ter_serial = itertools.count()
        self.base_stale = set()         # blocks whose base photo lacks lazy imagery (NAIP) still to come
        self.base_skipped = set()      # blocks built without their photo (a fast player's prefetch: base_refresh)
        self.quick_pool = ThreadPoolExecutor(1, thread_name_prefix="quick-base")   # stand-in photos (_quick_later)
        self.quick_early = ThreadPoolExecutor(2, thread_name_prefix="quick-early")   # ... started with a block (fast)
        self.quick_lock = threading.Lock()  # (_quick_once: a block's stand-in made once)
        self.quick_made, self.quick_running = {}, {}
        self.quick_futs = {}            # (aid, cx, cy) -> a requested block's stand-in photo started with it (fast)
        self.photo_pool = None          # requested blocks' photos and stand-ins (main(): PHOTO_PROCESS_MIN_CPUS)
        self.recent_full = collections.deque(maxlen=RECENT_FULL)    # (anchorId, cx, cy, rgb) of the last full photos
        self.base_lock = threading.Lock()
        self.pads_rebuild = set()       # blocks built without gas station pads that lie on them (pads_stale)
        self.roads_rebuild = set()      # blocks built before every road cell under them was on disk (roads_refresh)
        self.pads_changed_at = 0.0      # (time.monotonic) the last new pad: Connection.pads_refresh waits a little
        self.fast = False               # the player faster than FAST_BASE_V (the latest pose): photos follow blocks
        self.base_pending = {}          # block -> the analysis regions its base photo was rendered without
        self.base_rev = itertools.count(1)
        self.hm_committed = deque(maxlen=HM_KEEP_COMMITTED)
        self.hm_sent = deque(maxlen=HM_KEEP_SENT)
        # Far tiles: (anchorId, key) -> TileInfo, LRU order; files are far_<anchorId>_<key>.<fmt>.
        self.tiles = OrderedDict()
        self.tile_inflight = {}
        # A node across the global tiles' region edge is a piece per global tile it overlaps (farfield.Piece), each
        # its node's geometry and texture cut: (anchorId, node key) -> {"built", "tex": (DDS bytes, bytes, sources,
        # pending regions)} of the last few, so the node's other pieces build neither again
        self.piece_parts = OrderedDict()
        self.build_stats = deque(maxlen=500)   # seconds per built tile, for the log
        self.level_stats = {}                  # level -> deque of (build s, texture s, shape B, texture B)
        self.prune_delay = PRUNE_DELAY
        self.prune_recheck = PRUNE_RECHECK
        self.retired_keep = RETIRED_KEEP
        self.delete_interval = DELETE_INTERVAL
        self.delete_max_rate = DELETE_MAX_RATE if near_visual in MESH_LAYOUT else 1.0 / DELETE_INTERVAL
        self.delete_queue = deque()     # (path, tile key or None) waiting for a rate-limited delete
        self._deleting = False
        self._prune_scheduled = set()
        # Real weather at the player (weather.py): a weathermod.Service, or None (--weather off)
        self.weather = weather
        # Real gravity at the player (gravity.py): a gravitymod.Provider, or None (--gravity off)
        self.gravity = gravity
        # On-screen data credits (credits.py): a creditsmod.Credits, or None (--credits off; Server()'s default)
        self.credits = credits
        self.surface = surface      # surface.Surface: ground models under the physics block (--surface on)
        if surface is not None and self.airports is not None:      # the runways painted in (GitHub #337)
            surface.extra_pads = self.airports.runway_pads
        self.mapdata = mapdata      # mapdata.MapData for the roads (--roads on), else None
        if mapdata is not None and getattr(mapdata, "dem", 0) is None:
            mapdata.dem = self.dem  # (gradesep's ground)
        self.grading = bool(grading) and mapdata is not None   # ground graded under roads (grading.py)
        # street widths measured between the buildings (roadwidth.py), decided once per road per run
        self.sidewalks = sidewalks  # raised sidewalk meshes with the road tiles (sidewalks.py)
        self.buildings = buildings  # building meshes with the road tiles (buildings.py)
        # gas stations with refuelling with the road tiles (gasstations.py; the user, 2026-10-23)
        # (the settings page's switch turns them off and on again: set_gas_stations; never on without the flag)
        self.gas_allowed = bool(gas_stations) and mapdata is not None
        self.gas_stations = self.gas_allowed
        # OpenStreetMap's fuel stations (osmfuel.py, the OSM test: the user, 2026-10-28, "lets test OSM if it is too laggy
        # we will revert"): fetched in the background per cell around the road tiles the car wants, never waited for; a
        # cell with stations that came after its tiles were sent sends them again (osm_fuel_want)
        self.osm_fuel = None
        if self.gas_allowed and mapdatamod.OSM_FUEL and hasattr(mapdata, "osm_fuel_fetch"):
            self.osm_fuel = ThreadPoolExecutor(OSM_FUEL_THREADS, thread_name_prefix="osm-fuel")
            self.osm_fuel_done, self.osm_fuel_wait = set(), {}
            self.osm_fuel.submit(self._osm_fuel_warm)
        self.dirs_on = False        # the settings page's Direction signs (dirsigns.enabled; BNE_DIRSIGNS=on forces it)
        if self.gas_stations and surface is not None and hasattr(surface, "gas_pads"):
            surface.gas_pads = True     # their forecourts painted asphalt (surface.Surface.fuel_pads)
        self.traffic = bool(traffic) and mapdata is not None   # the AI road graph around the player (navgraph.py)
        self.road_widths = (roadwidthmod.RoadWidths(
            lambda key: mapdata.cell(key, layers=("buildings",), kind="demand").layers.get("buildings"),
            roadwidthmod.store_dir(cache_dir, mapdata)).get
            if road_widths and mapdata is not None else None)
        if vegetation is not None:      # trees kept off the roads' drawn widths (vegetation._mask_roads)
            vegetation.surface, vegetation.widths = surface, self.road_widths
        self.grader_lock = threading.Lock()
        # The download cache's bookkeeping (cache.py): every producer reports its disk hits and writes
        self.cache = cache
        self.covermap = None            # the map's cache view (covermap.CoverMap; main_async, with a Cache)
        self.perf = perfstats.Collector()   # the anonymous performance data of the reports (telemetry.py, perfstats.py)
        self.areas = None               # saved areas downloaded ahead (areas.Areas; main_async, with a Cache)
        self.fast_pool = None           # the saved areas' fast mode: its own road processes (fast_precache)
        self.make_fast_pool = None      # n -> such a pool (main_async: the road pool's initializer), or None
        self._fast_lock = threading.Lock()
        if cache is not None:
            for producer in (self.dem, self.imagery, self.deshadow, self.colormatch):
                if producer is not None:
                    producer.disk = cache

    PACK_WIPE_MARK = "terrainpacks_wiped_v1"   # in the cache dir: the one wipe of the leaked packs done
    PACK_WIPE_MIN_BYTES = 5 * 1024 ** 3

    def snap(self, v):
        """Nearest block centre coordinate on the layout's grid (256 m; M4 512²: 128 m)."""
        return heightmap.snap(v, self.layout.grid)

    def new_anchor(self, lat, lon):
        a = Anchor(self._next_anchor, lat, lon, geodesy.EnuFrame(lat, lon))
        a.hm_zoom = demmod.zoom_for(lat, 1.25, demmod.FINE_ZOOM)    # ~1 m pixels where the data exists (dem.FINE_ZOOM)
        a.zooms = {lvl: self.layout.dem_zoom(lat, lvl) for lvl in range(1, farfield.HORIZON_MAX_LEVEL + 1)}
        a.zooms[0] = a.hm_zoom                      # L0 shares the terrain's DEM (§1.4)
        a.zf = farfield.global_zoom(lat)            # the global far tiles' finest zoom (geotiles phase 3b)
        if self.deshadow is not None:
            self.deshadow.prepare(lat, lon)         # the regions around it, in the background
        if self.colormatch is not None:
            self.colormatch.prepare(lat, lon)
        self._next_anchor += 1
        return a

    def anchor_q(self, anchor):
        """The anchor's height quantum, surveyed once (worker threads)."""
        with anchor.q_lock:
            if anchor.q is None:
                anchor.q = survey_q(anchor.frame, self.dem, self.layout.block * heightmap.SQUARE)
                log.info("anchor %d height quantum 1/%d m", anchor.id, round(1 / anchor.q))
            return anchor.q

    # --- heightmaps ------------------------------------------------------
    def hm_name(self, key):
        aid, cx, cy = key
        return f"hm_{aid}_{int(cx)}_{int(cy)}.png"

    def ter_name(self, key):
        """A new .ter name for every build of key: the engine skips a terrainFile it already has, and a
        rebuilt block (an evicted one, or one with more of its land cover) can differ."""
        return f"{self.hm_name(key)[:-4]}_{next(self.ter_serial)}.ter"

    def hm_vfs(self, key):
        return f"{self.vfs_dir}/{self.hm_name(key)}"

    global_far = True       # --global-far: far levels >= 3 as global tiles (geotiles phase 3b; the mod must place them)
    geo_tiles = True        # --geo-tiles: content in the global tiles' frames, kept by the mod across rebases (geotiles.py)

    def country_at(self, lat, lon):
        """The ISO country code at (lat, lon) (its nearest airport's: navgraph.country_near), or None."""
        try:
            return navgraphmod.country_near(self.airports_db, float(lat), float(lon))
        except (TypeError, ValueError):
            return None

    def left_hand(self, anchor):
        """Whether traffic drives on the left at the anchor (its country's road rules, navgraph.road_rules)."""
        lh = getattr(anchor, "_left_hand", None)
        if lh is None:
            lh = navgraphmod.road_rules(navgraphmod.country_near(self.airports_db, anchor.lat, anchor.lon))["rightHandDrive"]
            anchor._left_hand = lh
        return lh

    def buried(self, anchor):
        """The share of low-voltage lines underground in the anchor's country (furniture.buried_share: fewer utility
        poles)."""
        v = getattr(anchor, "_buried", None)
        if v is None:
            v = furnituremod.buried_share(navgraphmod.country_near(self.airports_db, anchor.lat, anchor.lon))
            anchor._buried = v
        return v

    def country(self, anchor):
        """The anchor's country (ISO alpha-2, its nearest airport's: navgraph.country_near), or None: the road
        markings' style (markings.style)."""
        v = getattr(anchor, "_country", "")
        if v == "":
            v = navgraphmod.country_near(self.airports_db, anchor.lat, anchor.lon) if self.airports_db else None
            anchor._country = v
        return v

    def lamp_region(self, anchor):
        """The anchor's street lamp rules (furniture.lamp_region: their spacing)."""
        v = getattr(anchor, "_lamp_region", "")
        if v == "":
            v = furnituremod.lamp_region(navgraphmod.country_near(self.airports_db, anchor.lat, anchor.lon),
                                         navgraphmod.region_near(self.airports_db, anchor.lat, anchor.lon))
            anchor._lamp_region = v
        return v

    def mph(self, anchor):
        """Whether speed limits are posted in mph at the anchor (its country, navgraph.road_rules)."""
        v = getattr(anchor, "_mph", None)
        if v is None:
            v = navgraphmod.road_rules(navgraphmod.country_near(self.airports_db, anchor.lat, anchor.lon))["mph"]
            anchor._mph = v
        return v

    boost_on = False
    boost_cpus = server_cpus = None

    boost_full = False

    WARM_ORDER = {"roads": 0, "infrastructure": 1, "buildings": 2, "building_parts": 3, "water": 4}
    WARM_PLACE_LAYERS = ("roads", "infrastructure")

    FAR_GEO_KEPT = 20000        # far geotiles of this session remembered (build_far_geo: sent again from the files)

    def set_dirs(self, on):
        """The settings page's Direction signs switch (message direction_signs): True when it changed."""
        on = bool(on)
        changed = on != self.dirs_on
        self.dirs_on = on
        return changed

    def set_gas_stations(self, on):
        """The settings page's "Gas stations" switch (the mod's gas_stations; the user, 2026-10-23: off saves their
        cost on a small machine): stations laid out, pads painted and the region's shapes named only while on, never
        on without --gas-stations on. Returns whether it changed (the road tiles are then sent again)."""
        on = bool(on) and self.gas_allowed
        if on == self.gas_stations:
            return False
        self.gas_stations = on
        if self.surface is not None and hasattr(self.surface, "gas_pads"):
            self.surface.gas_pads = on
            self.surface.clear_fuel_pads()
        self.pads_stale()               # (the blocks' photos again: with or without their forecourts)
        log.info("gas stations %s (settings)", "on" if on else "off")
        return True

    def live_busy(self):
        """The live world has work in flight: a worker pool's jobs (blocks, far tiles, road tiles, far buildings), or
        a player's loading line not full. The saved areas' downloads (areas.py) wait meanwhile."""
        pools = [self.road_pool, self.far_bld_pool, getattr(self.far_pool, "local", self.far_pool)]
        pools += [bp for bp, _ in self.block_pools]
        if any(getattr(p, "inflight", 0) for p in pools if p is not None):
            return True
        for c in list(self.connections):
            counts = getattr(c, "progress_sent", None)
            # (the (done, wanted) pairs only: send_progress's counts end with the downloads [in flight, MB, MB/s]
            # since the loading line shows them, and unpacking that as a pair raised ValueError here, every minute,
            # so the saved areas never downloaded)
            if counts and any(len(p) == 2 and p[0] < p[1] for p in counts):
                return True
        return False

    def driving(self):
        """A player is moving (faster than a walk): the saved areas' downloads run at areas.DUTY_DRIVE."""
        return any(math.hypot(*getattr(c, "vel", (0.0, 0.0))) > 2.0 for c in list(self.connections))

    def areas_message(self, result=None):
        """The 'areas' message (areas.Areas.message; result: the answer to an area_add)."""
        m = {"type": "areas", **self.areas.message()}
        if result is not None:
            m["result"] = result
        return m

    def world_signature(self):
        """The settings the revisit cache's items are built with ({name: value}; the parts of the revisit keys that
        are the player's settings or the data's versions, not the place): a sidecar row (revisit.sig, written where
        the item is kept: blockworker, plancache) is "current" when its code_key is this server's and each setting
        it names has the value given here (the map's cache view, covermap.py). One function: a new setting that
        changes what is built goes into its revisit key, its store site's revisit.sig and here, under the same name.
        Per-anchor parts (the height quantum, country, lamp rules, driving side) are the place's, not settings."""
        im = self.imagery
        md = self.mapdata.versions() if self.mapdata is not None and hasattr(self.mapdata, "versions") else None
        return {
            # road tiles (blockworker._roads_tile_cached)
            "sidewalks": bool(self.sidewalks),
            "lift": float(DETAIL_ROAD_LIFT if self.near_visual == "detail" else 0.0),
            "buildings": bool(self.buildings),
            "gas_stations": bool(self.gas_stations),
            "pole_profile": Server.pole_profile(self),
            # every kind: the map data's versions (Overture release, layer versions, the weekly rail / lane builds)
            "md": md,
            # far tiles' geometry and texture (blockworker._far_key, far_tile): the layout (quality: cells, texture px)
            "layout": repr(self.layout),
            "chain": tuple(s.name for s in im.chain) if im is not None else None,
            "tone": self.tone,
            "leaf_on": getattr(im, "leaf_on", None),
            "season_mode": getattr(im, "season_mode", None),
            # road plans (plancache.for_grader's signature)
            "plans": f"w{int(self.road_widths is not None)}s{int(self.sea is not None)}",
        }

    def set_base_px(self, px):
        """The ground photo's size (px) the mod's terrain texture set has (its graphics setting groundPhoto, applied at
        a level's start: the 'graphics' message's photoPx); one of BASE_SIZES, else nothing changes. The blocks built
        at another size get their photo again (base_stale: base_refresh sends it), the next ones are made at it.
        Returns whether it changed."""
        try:
            px = int(px)
        except (TypeError, ValueError):
            return False
        if px not in BASE_SIZES or px == self.base_px:
            return False
        self.base_px = px
        if self.near_visual == "detail":
            self.base_stale.update(k for k in self.hm if k[0] in self.anchors)
        log.info("ground photo %d px (settings)", px)
        return True

    # (shared world building v1, commented out 2026-10-27 (off since 2026-10-23); replaced by share_hub above)
    # def share_start(self, port=None):
    #     """Compute sharing in a BeamMP group: our GroupHelper listening (started once) on port (the BeamMP server's
    #     setting, its plugin's SHARE_PORT; BNGE_SHARE_PORT on this machine wins; else peerwork.SHARE_PORT). -> {port,
    #     code} (port None: not sharing)."""
    #     if not hasattr(self.far_pool, "set_group"):
    #         return {"port": None, "code": peerworkmod.code_id()}
    #     self.far_pool.sharing = True        # every tile built is kept from now on (a player arriving gets ours)
    #     with self.share_lock:
    #         if self.group_helper is None:
    #             self.group_helper = peerworkmod.GroupHelper(self.far_pool, self.anchor_id_for,
    #                                                         os.path.join(self.cache_dir, "peer-out"))
    #             self.group_helper.start(share_port(port))
    #     return {"port": self.group_helper.port, "code": peerworkmod.code_id(), "weight": peerworkmod.share_weight()}

    # def share_group(self, msg):
    #     """The mod's "peers": {me, members: [{id, host, port, code}]}: the players on our anchor with our code."""
    #     if not hasattr(self.far_pool, "set_group"):
    #         return
    #     code = peerworkmod.code_id()
    #     members, weights = [], {}
    #     for m in msg.get("members") or []:
    #         try:
    #             if m.get("code") == code and m.get("host") and 0 < int(m.get("port")) < 65536:
    #                 members.append((str(m["id"]), str(m["host"]), int(m["port"])))
    #                 w = float(m.get("weight") or 1.0)
    #                 weights[str(m["id"])] = w if 0.1 <= w <= 1000 else 1.0
    #         except (TypeError, ValueError, AttributeError):
    #             continue
    #     self.far_pool.set_group(str(msg.get("me")), members, weights)

    # def anchor_id_for(self, lat, lon):
    #     """Our anchor id at (lat, lon) when we have one there (its graders are made already), else the helper's."""
    #     for a in list(self.anchors.values()):
    #         if float(a.lat) == lat and float(a.lon) == lon:
    #             return a.id
    #     return peerworkmod._anchor_id(lat, lon)



class Connection(_MapMixin, _AnchorMixin, _StreamMixin):
    """One mod connection. Far-field state for the active anchor A:
        desired  D: the selected tiles (farfield.select)
        shown    C: tiles of executed swaps, (anchorId, key) -> bbox in A's frame; also holds
                    previous anchors' tiles after a rebase until they are covered
        sent     S: far_tile sent but not yet part of a far_swap (the mod keeps them hidden)
        groups:     farfield.swap_groups(C, D), recomputed whenever D changes
    A group is emitted (far_swap) once all its adds are sent; C then becomes C - remove + add.
    Sent tiles that no group needs any more are evicted with far_remove."""

    def __init__(self, server, reader, writer):
        self.s = server
        self.waterguard_ok = getattr(server, "sea", None) is None
        self.reader, self.writer = reader, writer
        self.loop = asyncio.get_running_loop()
        self.peer = writer.get_extra_info("peername")
        self.tasks = set()
        self.lease = None
        self.closing = False
        self.active = None              # committed anchor id
        self.block = None               # farfield.Block of the committed terrain
        self.pending_block = None       # the last prepare_terrain of the active anchor
        self.terrain_wait = 0           # prepare_terrain requests being built (roads.Tracker: the ground first)
        self.pending_at = None          # when the pending block was asked (on_pose: the catch-up boost)
        self.pending_answered = False   # ... and whether its terrain_ready went (then the wait is the mod's, not ours)
        self.catchup = False            # the loading boost is on to catch up while driving (CATCHUP_*)
        self.caught_up_at = None
        self.predicted_block = None     # the terrain the prefetch expects next
        self.predicted_chain = []       # ... and after it, a fast player's (PREFETCH_CHAIN; blocks)
        self.focus = (0.0, 0.0)         # player x, y in the active frame
        self.view_at = None             # the camera's x, y there when it is VIEW_AWAY_M or more from the car, else None
        self.foci = ()                  # other vehicles' collision foci there ((x, y, r, trees), parse_foci)
        self.vel = (0.0, 0.0)
        self.cam = None                 # pose.cam (protocol 3), kept for diagnostics
        self.desired = {}               # key -> Tile (active anchor)
        self.refined = set()            # nodes refined by the last selection (hysteresis)
        self.node_changed = {}          # node -> monotonic time of its last state change
        self.shown = {}
        self.sent = {}                  # key -> Tile (active anchor)
        self.retired = {}               # (anchorId, key) -> monotonic time it left shown / sent
        self.held = set()               # pose.held: previous anchors whose tile objects the mod still has
        self.slot_holds = {}            # pooled slot -> serial of its latest far_tile the mod has not reported gone
        # a teleport's anchor is being built (prepare_anchor > TELEPORT_PAUSE_M away): no new work for the old place
        # (its far tiles, road / tree / building tiles, prefetch) until it commits or fails (2026-10-16: the anchor's
        # first block waited 30 s for the CPU after a level load, 3.6 s alone)
        self.teleport_pending = False
        self.sessions = set()           # hello / pose `sessions`: earlier runs' folders the mod still uses
        self.groups = []                # [(adds, removes)] still to emit
        # their swap deltas, computed off the event loop: (adds, removes) as tuples -> Future of (dIn, dOut)
        # (group_deltas on the loop stalled it for seconds at 90 m/s, and every message waited, 2026-10-12)
        self.deltas = {}
        self.queue = []                 # tiles to build, by priority
        self.qi = 0
        self.next_group = 1
        self.inflight = {}              # (anchorId, key) -> (asyncio future, Tile)
        self.prefetched = {}            # (anchorId, key) -> when its imagery prefetch was last asked for
        self.failed = set()
        self.dirty = False              # far_done owed once every group is emitted
        self.batch = [0.0, 0, 0]        # [start, tiles sent, swaps sent] of the current change, for the log
        self.prefetching = False
        self.wake = asyncio.Event()
        self.streamer = None
        self.last_status = ""           # text of the last status message (re-sent with a new backlog)
        self.note_sent = ""             # the network note (Server.net_note) the last status carried
        self.status_sent = (0.0, None)  # (monotonic time, backlog) of the last status message
        # M1h (horizon mode): the eye in m above sea level (quantised, see EYE_STEP; frame
        # independent, so it survives a rebase; None before the first pose), the far-field radius of
        # the last selection, the radius the last status carried, the region's highest visible terrain
        self.eye = None
        self.radius = farfield.HORIZON_MIN if server.horizon else server.radius
        self.radius_sent = None
        self.zmax_region = None
        self.weather = weathermod.Tracker()     # real weather: fetch rule, last message sent
        self.road_geo = False                   # road tiles as geotiles (hello.geoTiles, Server.roads_geo_ok)
        self.roads = roadsmod.Tracker()         # road tiles sent to the mod (--roads on)
        self.signs = signstream.Tracker()       # their highway signs (key "s<tx>_<ty>")
        self.signs_on = True                    # the mod's highwaySigns setting (message highway_signs)
        self.refining = set()                   # graded first blocks on their way (refine_first; send_progress)
        self.ground_wait = {}                   # block key -> when its photo was first seen waiting (ground_progress)
        self.veg = vegetationmod.Tracker()      # vegetation tiles sent to the mod (--trees on)
        self.veg_flat = False                   # ... as one flat array each (the mod's hello.vegFlat; trees.lua)
        # far buildings (--buildings on); none under the road tiles (they carry the buildings themselves)
        # the far (lite) buildings under the road tiles too, nearest first: they come in seconds after a teleport and the
        # mod hides each as its road tile's full buildings are made (roads.lua updateFar), the town round the car
        # drawn simply first and in detail after (the user, 2026-10-25: "load simpler versions first then higher detail
        # version later"); without them the far ones came first and the car's surroundings last
        # past them the skyline (buildings.skyline_tile): the far buildings then come as whole skyline squares
        # (group), the skyline tiles where none of theirs are (inner)
        self.far_geo = False                    # ... as geotiles (hello.farGeo, Server.far_geo_ok): kept across rebases
        self.global_far = False                 # far levels >= 3 as global tiles (hello.globalFar, --global-far)
        self.prev_horizon = None                # (frame, HorizonGrid) of the anchor before the active one (horizon_select)
        self.gpolys = {}                        # (anchorId, key) -> a shown global tile's quad in that anchor's world
        self.ppolys = {}                        # piece key -> its polygon (pieces sent / shown under the active anchor)
        self.building_scale = 1.0
        self.bldfar = self.bldsky = self.brfar = None
        self.make_far_trackers(False)
        self.turbine_geo = False               # separate capability; no road/building radius increase
        self.turbines = vegetationmod.Tracker(TURBINE_SEND_M, TURBINE_DROP_M, TURBINE_PER_POSE, "turbines",
                                              geo=True, z=TURBINE_Z)
        self.nav = navgraphmod.Tracker()        # the AI road graph around the player (--traffic on)
        self.traffic_visuals = False            # negotiated visual routes, requested only at night with traffic on
        self.traffic_on = False
        self.distant_traffic = vegetationmod.Tracker(distanttraffic.SEND_M, distanttraffic.DROP_M, 1, "traffic",
                                                     geo=True, z=distanttraffic.Z, rebuild_history=False)
        self.distant_traffic_far = vegetationmod.Tracker(distanttraffic.FAR_SEND_M, distanttraffic.FAR_DROP_M, 1,
                                                         "traffic_far", inner=(distanttraffic.FAR_SKIP_M, 1),
                                                         geo=True, z=distanttraffic.Z, rebuild_history=False)
        self.nav_wait = {}                      # anchor id -> monotonic time its first AI graph was wanted
        self.gravity = gravitymod.Tracker()     # real gravity: recompute rule, last message sent
        self.credits = creditsmod.Tracker()     # data credits: throttle, last message sent (--credits on)
        self.ground_busy = False                # a groundahead job is running (one at a time)
        self.route_dl_version = 0               # bumped by every route(): supersedes a running route_download job
        self.route_points = []                  # the last planned route's points ([lat, lon]: a saved route's)
        self.sea_busy = False                   # a water-tile plan is running (sea_tiles)
        self.sea_checked = None                 # (anchor id, 512 m square, monotonic, complete) of the last plan
        self.sea_sent = None                    # (anchor id, tiles) of the last "sea" message
        self.base_busy = False                  # a base photo refresh is running (base_refresh)
        self.pads_busy = False                  # a block is being built again with its new pads (pads_refresh)
        self.roads_busy = False                 # ... with the roads of map cells that came after it (roads_refresh)
        self.ground_sent = None                 # (anchor id, z, monotonic) of the last ground_ahead sent
        self.following = True                   # the mod moves its block (pose "follow"; false high up)
        self.map_view = False                   # the 3D map's view frame (pose "view": on_pose)
        self.map_open = False                   # the 3D map open (pose "map")
        # the mod's graphics settings (message 'graphics', after every hello and on a change; set_graphics): the far
        # field's detail (lodScale times the layout's K: the distance a level splits at), the tree tiles' reach
        # (treeScale) and the far buildings', skyline's and far bridges' (buildingScale); 1 = the defaults
        self.lod_scale = self.tree_scale = self.building_scale = 1.0
        self.view_cap = None                    # the far field's radius at most (m): the mod's fixed view distance

    def send(self, msg):
        lease = getattr(self, "lease", None)
        if not getattr(self, "closing", False) and (lease is None or lease.connection is self) and not self.writer.is_closing():
            data = protocol.encode(msg)
            transport = getattr(self.writer, "transport", None)
            pending = transport.get_write_buffer_size() if transport is not None else 0
            if pending + len(data) > SEND_BUFFER_MAX:
                self.abort_stream("outgoing buffer limit exceeded")
                return False
            self.writer.write(data)
            return True
        return False

    def abort_stream(self, reason):
        """Close this transport, preserving ownership until the game's next complete snapshot."""
        if self.closing:
            return
        self.closing = True
        log.info("client %s: %s", self.peer, reason)
        self.writer.close()
        transport = getattr(self.writer, "transport", None)
        if transport is not None:
            transport.abort()            # a blocked peer must not keep a closing connection alive indefinitely
        if self.streamer is not None:
            self.streamer.cancel()
        for task in tuple(self.tasks):
            task.cancel()

    def backlog(self):
        """status.backlog: desired tiles of the current anchor that are not shown yet (V1_4_SPEC §4)."""
        aid = self.active
        if aid is None:
            return 0
        return sum(1 for k in self.desired if self.ck(k) not in self.shown)

    def status(self, text):
        n = self.backlog()
        self.last_status, self.status_sent = text, (time.monotonic(), n)
        note = self.note_sent = self.s.net_note()
        if note:                        # offline / degraded: the mod shows the status text as it is
            text = f"{text} | {note}" if text else note
        m = {"type": "status", "text": text, "backlog": n}
        if self.s.horizon:              # M1h: the far-field radius (the mod's visibleDistance follows it)
            m["radius"] = round(self.radius)
            self.radius_sent = self.radius
        self.send(m)

    def send_progress(self, aid):
        """The loading line's counts (the mod's loadprogress.lua): (done, wanted) of the road tiles, the tree tiles and
        the far tiles round the player, sent when they change, at most every PROGRESS_S."""
        roads = self.roads.progress() if self.s.mapdata is not None else (0, 0)
        trees = self.veg.progress() if self.s.veg is not None else (0, 0)
        want = len(self.desired)
        far = (want - self.backlog(), want)
        meter = self.__dict__.setdefault("net_meter", netstatemod.Meter())
        act, mb, rate = meter.read()
        # (the downloads the world waits on, across the server's processes: the loading line's and the edge hold's
        # "downloading" part; the user, 2026-11-04)
        net = [int(act), round(mb, 1), round(rate, 2)]
        counts = (self.ground_progress(aid), roads, trees, far, tuple(net))
        now = time.monotonic()
        if counts == getattr(self, "progress_sent", None) or now - getattr(self, "progress_at", 0.0) < PROGRESS_S:
            return
        self.progress_sent, self.progress_at = counts, now
        self.send({"type": "load_progress", "anchorId": aid, "ground": list(counts[0]), "roads": list(roads),
                   "trees": list(trees), "far": list(far), "net": net})

    def ground_progress(self, aid):
        """(1, 1) once the ground under the player is final, else (0, 1): a teleport's graded first block still on its
        way (refine_first), or the block in use waiting to be drawn again with its full photo (a quick one without the
        painted roads, or one without all its imagery: base_refresh), for GROUND_WAIT_S at most (an imagery source
        that does not come back must not hold the loading line)."""
        if any(k[0] == aid for k in self.refining):
            return 0, 1
        s = self.s
        key = next((k for k in reversed(s.hm_committed) if k[0] == aid), None)
        if key is None or s.near_visual != "detail" or not (key in s.base_stale or self.base_busy):
            return 1, 1
        since = self.ground_wait.setdefault(key, time.monotonic())
        return (1, 1) if time.monotonic() - since > GROUND_WAIT_S else (0, 1)

    def refresh_status(self):
        """Re-send the last status when the network note changed (a host went down or came back);
        --status-interval: also when the backlog changed, at most every status_interval seconds
        (off by default: v1.3.1 sends status on events only)."""
        if self.s.net_note() != self.note_sent:
            self.status(self.last_status)
            return
        iv = self.s.status_interval
        if iv <= 0 or time.monotonic() - self.status_sent[0] < iv:
            return
        if self.backlog() != self.status_sent[1]:
            self.status(self.last_status)

    async def run(self):
        log.info("client connected %s", self.peer)
        self.s.connections.add(self)
        self.streamer = asyncio.create_task(self.stream())
        try:
            while True:
                msg = await protocol.read_message(self.reader)
                if msg is None:
                    break
                try:
                    self.dispatch(msg)
                except (KeyError, TypeError, ValueError) as e:  # a malformed field must not drop the connection
                    log.warning("client %s: bad %r message: %s: %s", self.peer, msg.get("type"), type(e).__name__, e)
                except Exception:
                    log.exception("client %s: message %r failed", self.peer, msg.get("type"))
                await self.writer.drain()
        except (protocol.ProtocolError, ConnectionError) as e:
            log.warning("client %s: %s", self.peer, e)
        finally:
            self.closing = True
            self.route_dl_version += 1          # (a running route_download stops: no one to show its bar to)
            self.s.connections.discard(self)
            if not self.s.stream_leases.detach(self):
                self.s.slots_let_go(list(self.slot_holds))
            self.slot_holds = {}
            for tr in (self.roads, self.signs, self.veg, self.bldfar, self.bldsky, self.brfar, getattr(self, "turbines", None), self.nav,
                       self.distant_traffic, self.distant_traffic_far):
                if tr is not None:
                    tr.close()
            self.s.drop_claims(self)
            self.streamer.cancel()
            self.credits.close()
            pending_tasks = list(self.tasks)
            for t in pending_tasks:
                t.cancel()
            self.writer.close()
            # The streamer owns a separate Event.wait task; let its finally
            # finish before this connection (including a readiness probe) goes.
            await asyncio.gather(self.streamer, return_exceptions=True)
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            log.info("client disconnected %s", self.peer)

    def build_veg_tile(self, aid, tx, ty):
        """Server.build_veg_tile in this mod's format (veg_flat). The vegetation tracker's worker threads."""
        return self.s.build_veg_tile(aid, tx, ty, flat=self.veg_flat)

    def make_far_trackers(self, geo):
        """The far buildings', skyline's and far bridges' trackers (closing the old ones): the anchor's tiles, or
        geotiles (geo: far buildings in the road geotiles' grid sent as whole skyline squares, the skyline's and far
        bridges' squares at geotiles.SKY_Z / BRIDGE_Z), each at the building scale's reach (set_graphics)."""
        import geotiles
        for tr in (self.bldfar, self.bldsky, self.brfar):
            if tr is not None:
                tr.close()
        self.far_geo = bool(geo)
        bld = self.building_scale
        far_m = buildingsmod.FAR_SEND_M * bld
        sky_m = buildingsmod.SKY_SEND_M * bld
        br_m = buildingsmod.BRIDGE_FAR_SEND_M * bld
        far_drop = far_m + (buildingsmod.FAR_DROP_M - buildingsmod.FAR_SEND_M)
        sky_drop = sky_m + (buildingsmod.SKY_DROP_M - buildingsmod.SKY_SEND_M)
        br_drop = br_m + (buildingsmod.BRIDGE_FAR_DROP_M - buildingsmod.BRIDGE_FAR_SEND_M)
        # far buildings (--buildings on); none under the road tiles (they carry the buildings themselves)
        # the far (lite) buildings under the road tiles too, nearest first: they come in seconds after a teleport and
        # the mod hides each as its road tile's full buildings are made (roads.lua updateFar), the town round the car
        # drawn simply first and in detail after (the user, 2026-10-25: "load simpler versions first then higher
        # detail version later"); without them the far ones came first and the car's surroundings last
        # past them the skyline (buildings.skyline_tile): the far buildings then come as whole skyline squares
        # (group), the skyline tiles where none of theirs are (inner)
        sky = buildingsmod.SKY_SEND_M > buildingsmod.FAR_SEND_M
        bridges = buildingsmod.BRIDGE_FAR_SEND_M > roadsmod.SEND_M
        if geo:
            self.bldfar = vegetationmod.Tracker(far_m, far_drop, buildingsmod.FAR_PER_POSE, "bldfar", geo=True,
                                                z=geotiles.FAR_Z, group_z=geotiles.SKY_Z if sky else None)
            self.bldsky = (vegetationmod.Tracker(sky_m, sky_drop, buildingsmod.SKY_PER_POSE, "bldsky", geo=True,
                                                 z=geotiles.SKY_Z, inner=(far_m, buildingsmod.SKY_UNIT))
                           if sky else None)
            # (the far bridges' squares under the road tiles too: a square's pieces past their reach are wanted, the
            # mod hides each piece where its road geotile shows its own deck)
            self.brfar = (vegetationmod.Tracker(br_m, br_drop, buildingsmod.BRIDGE_FAR_PER_POSE, "brfar", geo=True,
                                                z=geotiles.BRIDGE_Z) if bridges else None)
            return
        self.bldfar = vegetationmod.Tracker(far_m, far_drop, buildingsmod.FAR_PER_POSE, "bldfar",
                                            group=buildingsmod.SKY_UNIT if sky else None)
        self.bldsky = (vegetationmod.Tracker(sky_m, sky_drop, buildingsmod.SKY_PER_POSE, "bldsky",
                                             unit=buildingsmod.SKY_UNIT, inner=(far_m, buildingsmod.SKY_UNIT))
                       if sky else None)
        # the bridges past the road tiles (buildings.far_bridge_tile): none built under the road tiles' own reach
        self.brfar = (vegetationmod.Tracker(br_m, br_drop, buildingsmod.BRIDGE_FAR_PER_POSE, "brfar",
                                            skip_m=roadsmod.SEND_M - roadsmod.TILE) if bridges else None)

    def build_far_geo(self, kind):
        """The far geotile tracker's build(gx, gy) of kind (Server.build_far_geo)."""
        return lambda gx, gy: self.s.build_far_geo(kind, gx, gy)

    def build_veg_geo(self, gx, gy):
        """Server.build_veg_geo in this mod's format (veg_flat): the geotile tracker's worker threads."""
        return self.s.build_veg_geo(gx, gy, flat=self.veg_flat, aid=self.active)

    SHARE_TICK_S = 2.0

    async def share_pump(self, hub):
        """The sharing hub's paced sending (mpshare.Hub.pump) and, every SHARE_TICK_S, its hello / uploads / prefetch
        around our player (Hub.tick, in the UI pool: it stats files), for the server's life."""
        tick_at = 0.0
        loop = asyncio.get_running_loop()
        while True:
            try:
                now = time.monotonic()
                if now >= tick_at:
                    tick_at = now + self.SHARE_TICK_S
                    ll = getattr(self, "pose_ll", None)
                    vx, vy = getattr(self, "vel", (0.0, 0.0))
                    await loop.run_in_executor(self.s.ui_pool, hub.tick, ll[0] if ll else None,
                                               ll[1] if ll else None, vx, vy)
                wait = hub.pump()
            except Exception as e:          # (never the connection's loop)
                log.warning("share: %s: %s", type(e).__name__, e)
                wait = 1.0
            await asyncio.sleep(min(wait if wait is not None else 0.05, max(0.005, tick_at - time.monotonic())))

    def spawn(self, coro):
        if self.closing:
            coro.close()
            return None
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    def version_note(self, msg):
        """Tell the player when the mod in their game is not this app's own (a BeamMP server sends its own copy of the
        mod and turns the player's off): an older one has none of the faster tile paths the hello negotiates (before
        1.7.2: no geotiles, every rebase rebuilt the world round the car: 'severe performance issues', players'
        reports 2026-11-07). hello.release (release.lua, packaged with the mod) names its release when it has one."""
        if not isinstance(msg.get("modVersion"), str) or "vegFlat" not in msg:
            return                      # (not a BeamNG.Earth mod's hello: every one since 1.7.0 names these)
        mine = updatesmod.local_version() or ""
        theirs = msg.get("release") if isinstance(msg.get("release"), str) else ""
        import geotiles
        old_paths = msg.get("geoTiles") != geotiles.Z or msg.get("roadGeo") != roadsmod.ROAD_GEO
        if theirs and mine and theirs != mine:
            log.warning("hello: the mod is BeamNG.Earth %s, this app %s", theirs, mine)
            self.status(f"The BeamNG.Earth mod in your game is {theirs}, your app is {mine}. On a BeamMP server its owner "
                        "should run the same version as the players (the server's mod replaces yours): mismatched "
                        "versions load the world slower.")
        elif old_paths:
            log.warning("hello: the mod is BeamNG.Earth 1.7.1 or older (no geotiles): the slow tile paths")
            self.status("The BeamNG.Earth mod in your game is much older than your app (1.7.1 or older), so the world "
                        "loads the slow way. On a BeamMP server: ask its owner to update the server's BeamNG.Earth mod.")
        elif not getattr(self, "waterguard_ok", True):
            self.status("The BeamNG.Earth mod in your game is older than your app: the world loads, but its underwater "
                        "tunnels may flood. On a BeamMP server: ask its owner to update the server's BeamNG.Earth mod.")

    def dispatch(self, msg):
        if getattr(self, "closing", False):
            return
        kind = msg["type"]
        if kind == "hello":
            # (an older mod without the water guard: its world still loads, only an underwater tunnel's water may come
            # in. Refusing every world job left a BeamMP server's players, whose mod is the server's own copy, in an
            # empty world whenever that copy was older than their app, 2026-11-07)
            self.waterguard_ok = self.s.sea is None or msg.get("waterguard") == WATERGUARD_VERSION
            if not self.waterguard_ok:
                log.warning("hello: the mod has no water guard (older than this server): underwater tunnels may flood")
        if getattr(self, "lease", None) is None and kind in ("prepare_anchor", "commit_anchor", "prepare_terrain", "commit_terrain", "pose"):
            # (a client that skipped hello: a fresh lease, not resumable, and the request goes on as before PR #817;
            # the mod always says hello first on its connection, so its reconnect still resumes before any request)
            self.s.attach_stream(self, {})
        if kind == "traffic_visuals":
            self.traffic_on = self.traffic_visuals and msg.get("enabled") is True
            if self.traffic_on and msg.get("reset") is True:
                # The connection outlives levels; a reloaded client discarded its route cache.
                self.distant_traffic.sent.clear()
                self.distant_traffic_far.sent.clear()
        elif kind == "hello":
            lease_id = self.s.attach_stream(self, msg)
            self.s.check_user_dir(self, msg.get("userPath"))
            lay = self.s.layout
            fmt = msg.get("format", lay.format)
            if msg.get("protocol") != PROTOCOL:
                self.status(f"protocol mismatch: mod {msg.get('protocol')}, server {PROTOCOL}")
            elif not isinstance(fmt, (int, float)) or fmt < lay.format:
                # the mod's format is the newest it reads (a FORMAT 4 mod reads 3 too)
                self.status(f"format mismatch: mod {msg.get('format')}, server {lay.format}")
            m = {"type": "hello", "protocol": PROTOCOL, "format": lay.format, "serverVersion": SERVER_VERSION,
                 # the welcome page's telemetry defaults for this computer's region (privacyregion.py: its own time
                 # zone, nothing asked of a server) and the consent version it asks for (telemetry.CONSENT_VERSION)
                 "privacyRegion": privacyregionmod.region()[0], "telemetryConsent": TELEMETRY_CONSENT,
                 "runtimeProfiler": 2, "appVersion": updatesmod.local_version() or "",     # the release (VERSION): the players' reports
                 "sessionId": self.s.sid, "streamLease": 1, "leaseId": lease_id,
                 "farFormat": self.s.far_format, "material": farfield.MATERIAL,
                 "nearVisual": self.s.near_visual, "attribution": self.s.attribution()}
            if self.s.imagery is not None:      # the world map credits the sources its view shows
                m["imageryCredits"] = self.s.imagery.credits_by_area()
                m["otherCredits"] = self.s.attribution_rest()
            if msg.get("waterguard") == WATERGUARD_VERSION:
                m["waterguard"] = WATERGUARD_VERSION
            if lay.format >= 4:         # M1: the tiles' cell count (crossover.defaultErr, the exact level)
                m["cells"] = lay.cells
            # the terrain block: samples per edge and their spacing (the mod's lattice / follow table, by its width in
            # metres: 1024 x 1 m since 2026-11-01)
            m["blockSize"], m["squareSize"] = lay.block, heightmap.SQUARE
            if self.s.horizon:                  # M1h: status.radius follows; the mod sets its sky and view distance
                m["horizon"] = True
            if self.s.deshadow is not None:     # --deshadow lambert|full: shown on the mod's settings page
                m["deshadow"] = self.s.deshadow.mode
            if self.s.colormatch is not None:   # --colormatch s2: likewise
                m["colormatch"] = self.s.colormatch.mode
            season_mode = getattr(self.s.imagery, "season_mode", "off") if self.s.imagery is not None else "off"
            if season_mode not in (None, "off"):    # --season: the mod tints the trees' leaves to match (seasonlook.lua)
                m["season"] = season_mode
            if self.s.map_port:                 # the world map's tiles (maptiles.py; --map-tiles on)
                m["mapPort"] = self.s.map_port
            if self.s.mapdata is not None:      # our shapes' materials, set up by the mod now (prewarm.py)
                if self.s.prewarm_vfs is None:
                    try:
                        self.s.prewarm_vfs = prewarmmod.write(self.s.out_dir, self.s.vfs_dir)
                    except Exception as e:      # (nothing to prewarm: the first tiles set them up)
                        log.info("prewarm shape: %s", e)
                        self.s.prewarm_vfs = ""
                if self.s.prewarm_vfs:
                    m["prewarm"] = self.s.prewarm_vfs
            # a game that outlived its server: keep the session folders its tiles still use
            self.s.claim_sessions(self, msg.get("sessions"))
            # a mod that reads a vegetation tile as one flat array of numbers (no table per item in its Lua: trees.lua);
            # an older one gets the rows
            self.veg_flat = msg.get("vegFlat") is True
            # a mod that places geotiles itself (hello.geoTiles: geotile.lua's zoom) gets the vegetation in them, kept
            # across rebases (--geo-tiles on); an older one, or --geo-tiles off, the anchor's tiles
            import geotiles
            geo = bool(self.s.geo_tiles) and msg.get("geoTiles") == geotiles.Z
            if geo != self.veg.geo:
                old = self.veg
                old.close()
                # (in the road processes the waiting tiles share a job, so more may wait: _veg_geo_job)
                per = VEG_GEO_PER_POSE if geo and self.s.road_pool is not None else vegetationmod.BUILD_PER_POSE
                self.veg = vegetationmod.Tracker(old.send_m, old.drop_m, per_pose=per, geo=geo)
            if geo:
                m["geoTiles"] = geotiles.Z
            self.traffic_visuals = (geo and msg.get("trafficVisuals") == distanttraffic.PROTO and self.s.traffic
                                    and self.s.mapdata is not None and self.s.far_bld_pool is not None)
            self.traffic_on = False
            if self.traffic_visuals:
                m["trafficVisuals"] = distanttraffic.PROTO
            # ... and the road tiles too when it places those (hello.roadGeo: roads.lua's ROAD_GEO; content kept across
            # rebases, a skin per anchor: roads.GeoTracker)
            road_geo = geo and msg.get("roadGeo") == roadsmod.ROAD_GEO and self.s.roads_geo_ok()
            if road_geo != self.road_geo:
                self.road_geo = road_geo
                self.roads.close()
                self.roads = road_tracker(self)
            if road_geo:
                m["roadGeo"] = roadsmod.ROAD_GEO
            # ... and the far buildings, skyline and far bridges when it places those too (hello.farGeo: roads.lua's
            # FAR_GEO; a far tile and the road geotile of the same key hold the same buildings: one shows them)
            far_geo = road_geo and msg.get("farGeo") == buildingsmod.FAR_GEO and self.s.far_geo_ok()
            if far_geo != self.far_geo:
                self.make_far_trackers(far_geo)
            if far_geo:
                m["farGeo"] = buildingsmod.FAR_GEO
            self.turbine_geo = msg.get("turbineGeo") == TURBINE_GEO
            if self.turbine_geo:
                m["turbineGeo"] = TURBINE_GEO
            # ... and the far field's levels from 3 up as global tiles when it places those (hello.globalFar:
            # farfield.lua's GLOBAL_FAR; geotiles phase 3b, --global-far): built once, kept across rebases
            self.ready_set = msg.get("readySet") == 1      # (the mod takes 'terrain_ready_set': follow.preferReady)
            self.global_far = (geo and msg.get("globalFar") == farfield.GLOBAL_PROTO and bool(self.s.global_far)
                               and self.s.layout.mesh)
            if self.global_far:
                m["globalFar"] = farfield.GLOBAL_PROTO
            self.send(m)
            self.version_note(msg)          # (after the hello and its protocol / format notes)
            if self.weather.msg is not None:        # a reconnected mod gets the weather again
                self.send(self.weather.msg)
            if self.gravity.msg is not None:        # ... and its gravity
                self.send(self.gravity.msg)
            if self.credits.msg is not None:        # ... and the data credits
                self.send(self.credits.msg)
            if self.s.net_note():                   # --offline / hosts down: say so at once
                self.status(self.last_status)
            if self.s.cache is not None:            # the download cache's usage (the settings page)
                self.send(self.s.cache.message())
            if self.s.areas is not None:            # the saved areas (the map's Places panel)
                self.send(self.s.areas_message())
            if self.s.update_check:
                self.spawn(self.check_update())     # a newer version on GitHub: the mod says so (updates.py)
            if getattr(self.s, "game_cpus", None):  # the game off the server's cores (cpupin.py)
                n = cpupin.pin_game(self.s.game_cpus)
                log.info("the game's %d %s on CPUs %s", n, "processes" if sys.platform == "win32" else "threads",
                         ",".join(map(str, sorted(self.s.game_cpus))))
        elif kind == "prepare_anchor":
            self.spawn(self.prepare_anchor(msg))
        elif kind == "server_list":             # the BeamNG.Earth servers for the tab in BeamMP's browser
            self.spawn(self.server_list())
        elif kind == "share_hello":             # a BeamMP session (mpsession): our share code and weight for the plugin
            import mpshare
            hub, new = self.s.share_hub(self.send)
            self.send({"type": "share_info", "code": mpshare.share_code(self.s.share_parts()),
                       "weight": peerworkmod.share_weight()})
            if new:
                self.spawn(self.share_pump(hub))
        elif kind == "share_group":             # the plugin's bngeGroup: the players near us that share our code
            hub = getattr(self.s, "_share_hub", None)
            if hub is not None:
                try:
                    hub.set_group(json.loads(msg.get("d") or "{}"))
                except ValueError:
                    pass
        elif kind == "share_in":                # a member's message through the plugin ("<from>|<json>")
            hub = getattr(self.s, "_share_hub", None)
            if hub is not None:
                hub.on_data(msg.get("d"))
        # (shared world building v1, commented out 2026-10-27: the mod sends neither any more)
        # elif kind == "share":                   # a BeamMP session: serve our group (peerwork.GroupHelper)
        #     self.send(dict(self.s.share_start(msg.get("port")), type="share"))
        # elif kind == "peers":                   # the players on our anchor (the BeamMP plugin, through mpsession)
        #     self.s.share_group(msg)
        elif kind == "prepare_terrain":
            self.spawn(self.prepare_terrain(msg))
        elif kind == "commit_anchor":
            self.commit_anchor(msg.get("anchorId"))
        elif kind == "commit_terrain":
            self.commit_terrain(msg)
        elif kind == "pose":
            self.on_pose(msg)
        elif kind == "map_tiles":               # the in-game map's tiles, as files (the UI cannot fetch http)
            self.spawn(self.map_tiles(msg))
        elif kind == "geocode":                 # the map's search box: addresses and places (geocode.py)
            self.spawn(self.geocode(msg))
        elif kind == "cache_map":               # the map's cache view: what is cached where (covermap.py)
            self.spawn(self.cache_map(msg))
        elif kind == "area_plan":               # the map's picked area: the estimate for its confirm box (areas.py)
            self.spawn(self.area_plan(msg))
        elif kind == "area_add":                # ... downloaded ahead
            self.spawn(self.area_add(msg))
        elif kind == "area_cmd":                # a saved area paused, resumed or deleted
            self.spawn(self.area_cmd(msg))
        elif kind == "route":                   # the GPS: the whole road route (routing.py)
            self.spawn(self.route(msg))
        elif kind == "where":                   # the status bar's street, town and country (geocode.py)
            self.spawn(self.where(msg))
        elif kind == "map_labels":              # the 3D map's labels, every kind by zoom (maplabels.py)
            self.spawn(self.map_labels(msg))
        elif kind == "map_focus":               # the 3D map flying: where the view will end (map3d.flyHint)
            self.on_map_focus(msg)
        elif kind == "random_place":            # the level select's "Random location" (randomplace.py)
            self.spawn(self.random_place(msg))
        elif kind == "place_previews":          # the level select's place pictures (placepreview.py)
            self.spawn(self.place_previews(msg))
        elif kind == "update_now":              # the update notice's "Update now" (selfupdate.py)
            self.spawn(self.update_now())
        elif kind == "open_page":               # the update notice's button: the releases page (updates.open_page)
            # (and the Add-ons page's links: an add-on's own text or repo, addons.Addons.openable)
            allow = self.s._addons.openable if getattr(self.s, "_addons", None) is not None else None
            if not updatesmod.open_page(msg.get("url"), allow=allow):
                log.info("open_page: refused %r", msg.get("url"))
        elif kind == "globe_req":               # the 3D map's whole-earth globe for an anchor (globe.py)
            self.spawn(self.globe(msg))
        elif kind == "space_clouds_req":        # the camera above the clouds: today's clouds round it (spaceclouds.py)
            self.spawn(self.space_clouds(msg))
        elif kind == "addons_list":             # the Add-ons page: the list (addons.py)
            self.spawn(self.addons_list(msg.get("refresh") is True))
        elif kind == "addon_detail":            # ... one add-on's README and changelog
            self.spawn(self.addon_detail(msg.get("id"), msg.get("refresh") is True))
        elif kind == "addon_install":           # ... Install / Update: downloaded and checked, then addon_apply
            self.spawn(self.addon_install(msg.get("id"), msg.get("consent") is True, msg.get("mp") is True))
        elif kind == "addon_add":               # ... Add from GitHub: a pasted repository put on the local list
            self.spawn(self.addon_add(msg.get("repo")))
        elif kind == "addon_options":           # ... its Options tab saved
            self.spawn(self.addon_options(msg.get("id"), msg.get("values")))
        elif kind == "addon_forget":            # ... one the player added taken off the list
            self.spawn(self.addon_forget(msg.get("id")))
        elif kind == "addon_run":               # ... a program beside the game: start / stop / remove / autostart
            self.spawn(self.addon_run(msg.get("id"), msg.get("op")))
        elif kind == "addon_applied":           # ... the game's mod manager did it (or not): logged, the list again
            self.spawn(self.addon_applied(msg.get("id"), msg.get("op"), msg.get("ok") is True, msg.get("error")))
        elif kind == "terrain_patch":           # ground under another vehicle's collision island (terrainpatch.lua)
            self.spawn(self.terrain_patch(msg))
        elif kind == "cache":
            self.on_cache(msg)
        elif kind == "app_path":                # the settings page's folders: the cache's, the BeamMP launcher's
            self.spawn(self.app_path(msg))
        elif kind == "runtime_profiler":
            mode = msg.get("mode", "full")
            if mode not in ("essential", "full"):
                raise ValueError("invalid profiler mode")
            self.profile_mode = mode
            self.profile_until = time.monotonic() + 3 if msg.get("enabled") is True else 0
            probe = msg.get("probe")
            if msg.get("enabled") is True and type(probe) is int and 0 <= probe <= 2147483647:
                self.send({"type": "runtime_profiler_pong", "probe": probe})
        elif kind == "perf_log":                # the mod's performance log (perflog.lua): its hitches, frame-rate drops,
            # minute lines and the machine, into this log beside the blocks' own timings (the file players send)
            lines = msg.get("lines")
            for line in (lines if isinstance(lines, list) else [])[:PERF_LOG_LINES]:
                if isinstance(line, str):
                    log.info("game: %s", line[:PERF_LOG_CHARS].replace("\n", " "))
        elif kind == "perf_setup":              # the anonymous performance data (perfsummary.lua; perfstats.py checks it):
            _perf(self.s).on_setup(msg.get("setup"))      # the machine and settings, once a connection
        elif kind == "perf_summary":            # ... and a 5-minute window of frame times, scene and world generation
            _perf(self.s).on_summary(msg.get("summary"))
        elif kind == "roads_reset":             # the mod dropped its roads (settings): send them all again
            self.roads.close()
            self.roads = road_tracker(self)
            self.signs.close()
            self.signs = signstream.Tracker()
        elif kind == "highway_signs":           # the settings page's switch: no sign tiles built while it is off
            self.signs_on = bool(msg.get("on", True))
        elif kind == "direction_signs":         # the settings page's switch: the sign tiles again, with or without
            if self.s.set_dirs(msg.get("on")):
                self.signs.refresh()
        elif kind == "graphics":                # the settings page's graphics knobs (set_graphics)
            self.set_graphics(msg)
        elif kind == "gas_stations":            # the settings page's switch: the road tiles again, with or without
            if self.s.set_gas_stations(msg.get("on")):
                self.roads.close()
                self.roads = road_tracker(self)
        else:
            log.debug("ignoring message type %r", kind)

    def set_graphics(self, msg):
        """The settings page's graphics (the user, 2026-10-28: "give player much more control in the graphics settings
        (view distance, LOD quality, ETC."), message 'graphics' {lodScale, treeScale, buildingScale, photoPx}, each
        optional (viewM: the fixed view distance in m, the far field's radius at most; 0 = the real horizon):
        lodScale times the K table (the far field's terrain detail: >1 splits each level farther out, finer
        ground to the horizon; Baltimore's first selection, 64 cells: 0.75 -> 0.78x the tiles, 1.3 -> ~1.3x, 2 ->
        2.1x), treeScale the tree tiles' reach (vegetation.SEND_M times it), buildingScale the far buildings', the
        skyline's and the far bridges' (buildings.FAR_SEND_M / SKY_SEND_M / BRIDGE_FAR_SEND_M times it), photoPx the
        ground photo's size (Server.set_base_px). Scales are clamped to GRAPHICS_SCALE; the far field is selected
        again at once, the trackers send and drop by their new reach at the next pose."""
        def scale(name, cur):
            v = msg.get(name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return cur
            return min(GRAPHICS_SCALE[1], max(GRAPHICS_SCALE[0], float(v)))
        lod, tree, bld = (scale("lodScale", self.lod_scale), scale("treeScale", self.tree_scale),
                          scale("buildingScale", self.building_scale))
        far = lod != self.lod_scale
        self.lod_scale = lod
        if tree != self.tree_scale:
            self.tree_scale = tree
            self.veg.send_m = vegetationmod.SEND_M * tree
            self.veg.drop_m = self.veg.send_m + (vegetationmod.DROP_M - vegetationmod.SEND_M)
        if bld != self.building_scale:
            self.building_scale = bld
            self.bldfar.send_m = buildingsmod.FAR_SEND_M * bld
            self.bldfar.drop_m = self.bldfar.send_m + (buildingsmod.FAR_DROP_M - buildingsmod.FAR_SEND_M)
            if self.bldsky is not None:
                self.bldsky.send_m = buildingsmod.SKY_SEND_M * bld
                self.bldsky.drop_m = self.bldsky.send_m + (buildingsmod.SKY_DROP_M - buildingsmod.SKY_SEND_M)
                self.bldsky.inner = (self.bldfar.send_m, buildingsmod.SKY_UNIT)
            if self.brfar is not None:
                self.brfar.send_m = buildingsmod.BRIDGE_FAR_SEND_M * bld
                self.brfar.drop_m = self.brfar.send_m + (buildingsmod.BRIDGE_FAR_DROP_M - buildingsmod.BRIDGE_FAR_SEND_M)
        if msg.get("photoPx") is not None:
            self.s.set_base_px(msg.get("photoPx"))
        v = msg.get("viewM")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            cap = max(VIEW_CAP_MIN_M, float(v)) if v > 0 else None
            far = far or cap != self.view_cap
            self.view_cap = cap
        log.info("graphics: terrain detail x%.2f, trees to %.0f m, far buildings to %.0f m, ground photo %d px, view %s",
                 self.lod_scale, self.veg.send_m, self.bldfar.send_m, self.s.base_px,
                 "to the horizon" if self.view_cap is None else f"{self.view_cap / 1000:.0f} km")
        if far and self.active is not None:
            self.update_far()

    def kfun(self):
        """K(L) of this connection's far field: the server's times the mod's terrain detail (lodScale), times
        MAP_LOD_SCALE with the 3D map's view frame (finer tiles farther out: high res at every zoom)."""
        m = self.map_view or self.map_open
        k, s = self.s.kfun, self.lod_scale * (MAP_LOD_SCALE * getattr(self, "lod_ramp", 1.0) if m else 1.0)
        return k if s == 1.0 else (lambda lvl: k(lvl) * s)

    async def globe(self, msg):
        """The mod's globe.lua (the 3D map far out): the whole earth as one sphere in anchorId's frame, its world texture
        made once and cached (globe.py); answered with its files, or nothing (no world texture could be had)."""
        aid = msg.get("anchorId")
        if aid not in self.s.anchors:
            return
        import globe as globemod
        frame = self.s.anchors[aid].frame
        try:
            res = await asyncio.get_running_loop().run_in_executor(
                None, globemod.build, self.s.out_dir, self.s.vfs_dir, frame, aid, self.s.cache_dir, self.s.net)
        except Exception as e:
            log.warning("globe: %s: %s", type(e).__name__, e)
            return
        if res is not None:
            self.send(dict(res, type="globe", anchorId=aid))

    async def space_clouds(self, msg):
        """The mod's spaceclouds.lua: the cloud shell round the point under a camera far above the clouds (lat, lon,
        alt m above sea level), in anchorId's frame; answered with its files (or nothing: no tile could be had)."""
        aid = msg.get("anchorId")
        try:
            lat, lon, alt = float(msg["lat"]), float(msg["lon"]), float(msg["alt"])
        except (KeyError, TypeError, ValueError):
            return
        if (aid not in self.s.anchors or not all(math.isfinite(v) for v in (lat, lon, alt))
                or abs(lat) > 90 or abs(lon) > 180 or not 0 <= alt <= 2_000_000):
            return
        import spaceclouds as spacecloudsmod
        frame = self.s.anchors[aid].frame
        sun = None                              # (the game's sun seen from there: its night side's lights follow it)
        try:
            az, el = float(msg["sunAz"]), float(msg["sunEl"])
            if math.isfinite(az) and math.isfinite(el) and -90 <= el <= 90:
                sun = spacecloudsmod.subsolar_from(lat, lon, az, el)
        except (KeyError, TypeError, ValueError):
            pass
        try:
            res = await asyncio.get_running_loop().run_in_executor(
                None, spacecloudsmod.build, self.s.out_dir, self.s.vfs_dir, frame, lat, lon, alt, self.s.net, sun)
        except Exception as e:
            log.warning("space clouds: %s: %s", type(e).__name__, e)
            return
        if res is not None:
            self.send(dict(res, type="space_clouds", anchorId=aid))



async def main_async(args):
    log.info("startup: initializing data services")
    if os.environ.get("BNGE_WORLD_BUNDLE") and (args.roads != "on" or args.grading != "on"):
        from worldbundle import BundleError
        raise BundleError("BNGE_WORLD_BUNDLE requires --roads on --grading on")
    if os.environ.get("BNGE_WORLD_BUNDLE") and getattr(args, "peers", ""):
        from worldbundle import BundleError
        raise BundleError("Legacy --peers helpers do not negotiate the pinned road-plan identity")
    if os.environ.get("BNGE_WORLD_BUNDLE"):
        from worldbundle import validate_bundle
        # Reject malformed/missing data before allocating services or starting cache maintenance.
        validate_bundle(os.environ["BNGE_WORLD_BUNDLE"])
    fmt = args.far_format
    if fmt == "cdae":
        v = cdae.engine_version(args.user_dir)
        if v is not None and v != cdae.VERSION_WORD:
            log.warning("the game writes shape version 0x%08x, this server 0x%08x: using DAE far tiles",
                        v, cdae.VERSION_WORD)
            fmt = "dae"
    kfun = None if args.lod_k is None else (lambda lvl: args.lod_k)
    radii = None
    if args.near_radii is not None:
        ml = farfield.MESH_MIN_LEVEL[args.cells]
        radii = {ml: args.near_radii[0], ml + 1: args.near_radii[1]}
    net = netfetch.Fetcher(stats_dir=os.path.join(args.cache_dir, "netstats"), offline=args.offline)
    # map data (mapdata.py): land cover for the surfaces, roads; one instance shares its cache locks
    mapdata = None
    try:
        mapdata = (mapdatamod.MapData(args.cache_dir, fetcher=net)
                   if "on" in (args.surface, args.roads, args.sea, args.trees) else None)
        if os.environ.get("BNGE_WORLD_BUNDLE"):
            from worldbundle import PinnedPlanCache
            PinnedPlanCache(os.environ["BNGE_WORLD_BUNDLE"],
                            f"w{int(args.road_widths == 'on')}s{int(args.sea == 'on')}", mapdata.bundle_versions())
    except BaseException:
        try:
            if mapdata is not None:
                mapdata.close()
        finally:
            net.close()
        raise
    log.info("startup: opening the download cache")
    disk = cachemod.Cache(args.cache_dir, args.cache_cap)
    threading.Thread(target=prune_own_caches, args=(args.cache_dir, time.time()), daemon=True,
                     name="cache-prune").start()
    pacer.DUTY = args.background_duty
    earthbase.DETAIL_RING_M = args.detail_ring
    earthbase.AHEAD_ON = args.download_ahead == "on"
    earthbase.ROUTE_AHEAD_ON = args.route_prefetch == "on" and args.route_prefetch_km > 0
    earthbase.ROUTE_AHEAD_MAX_M = args.route_prefetch_km * 1000.0
    # (before any worker process starts: they find the downloads' shared counters by name, netstate.ENV)
    if netstatemod.init():
        import atexit
        atexit.register(netstatemod.close)
    # (telemetry: every process's per-host download counts, summed by telemetry.Reporter; set before the workers start
    # so they inherit it)
    telemetry_on = args.telemetry == "on" and not args.offline and not os.environ.get("CI") \
        and os.environ.get("BNE_TELEMETRY", "on").lower() not in ("off", "0", "false", "no")
    if telemetry_on:
        proc_dir = os.path.join(args.cache_dir, "netstats", "proc")
        os.environ["BNE_NETSTATS_PROC"] = proc_dir
        netfetch._PROC_DIR = proc_dir
    log.info("startup: preparing worker processes")
    block_pools, far_pool, photo_pool = [], None, None
    plancachemod.start_run(args.cache_dir)      # (before the workers: they inherit the run's id)
    if args.block_process == "on":
        ctx = multiprocessing.get_context("spawn")
        for lane in (LANE_REQUEST, LANE_PREFETCH):
            warm_q = ctx.Queue(maxsize=64)
            # its feeder thread started now, not at the first warm_roads on the event loop: Thread.start waits for the
            # new thread to run, and a loaded machine stalled the loop 3.4 s there at a first visit's first pose
            # (2026-11-04; blockworker._read_warm skips the empty list)
            warm_q.put_nowait((None, 0.0, 0.0, []))
            block_pools.append((poolsmod.RespawningPool(
                "block", prefetch_procs() if lane == LANE_PREFETCH else 1, mp_context=ctx, initializer=blockworker.init,
                initargs=(args.cache_dir, args.offline, args.roads == "on",
                          args.grading == "on" and args.roads == "on",
                          args.road_widths == "on" and args.roads == "on", args.surface == "on", warm_q,
                          logging.getLogger().level, args.airports == "on", args.sea == "on", 0,
                          roadsmod.ROADS_IN_GROUND, args.gas_stations == "on" and args.roads == "on")), warm_q))
        if args.near_visual == "detail" and (os.cpu_count() or 0) >= PHOTO_PROCESS_MIN_CPUS:
            # requested blocks' photos and stand-ins in a process of their own (Server._base_process / _stand_in)
            photo_pool = poolsmod.RespawningPool(
                "photo", 1, mp_context=ctx, initializer=blockworker.init,
                initargs=(args.cache_dir, args.offline, args.roads == "on",
                          args.grading == "on" and args.roads == "on",
                          args.road_widths == "on" and args.roads == "on", args.surface == "on", None,
                          logging.getLogger().level, args.airports == "on", args.sea == "on", 0,
                          roadsmod.ROADS_IN_GROUND, args.gas_stations == "on" and args.roads == "on"))
        far_pool = poolsmod.RespawningPool(
            "far tile", max(args.workers, getattr(args, "far_max", args.workers)), mp_context=ctx, initializer=blockworker.init,
            initargs=(args.cache_dir, args.offline, args.roads == "on",
                      args.grading == "on" and args.roads == "on",
                      args.road_widths == "on" and args.roads == "on", False, None,
                      logging.getLogger().level, args.airports == "on", args.sea == "on", NICE_FAR,
                      roadsmod.ROADS_IN_GROUND))
    if far_pool is not None:
        # compute sharing: far tiles to helpers (--peers) and, in a BeamMP group, split with the players on our anchor,
        # built with the same settings as these workers (peerwork.py)
        far_pool = peerworkmod.SharedFarPool(
            far_pool, peerworkmod.parse_peers(args.peers),
            {"roads": args.roads == "on", "grading": args.grading == "on" and args.roads == "on",
             "road_widths": args.road_widths == "on" and args.roads == "on", "airports": args.airports == "on",
             "sea": args.sea == "on", "roads_in_ground": roadsmod.ROADS_IN_GROUND}, None,
            peerworkmod.ReplyCache(os.path.join(args.cache_dir, "peercache")))
    airports_db = None
    if args.airports == "on":       # before any worker process starts: they read this copy
        try:
            airports_db = airportsmod.Store(args.cache_dir, net).load()
            log.info("airports: %s", airports_db.summary())
        except airportsmod.AirportsUnavailable as e:
            log.warning("airports: no runway grading (%s)", e)
    road_pool = None
    if args.block_process == "on" and args.roads == "on" and args.road_processes > 0:
        road_pool = poolsmod.RespawningPool(      # road tiles (blockworker.roads_tile)
            "road tile", args.road_processes, mp_context=ctx, initializer=blockworker.init,
            initargs=(args.cache_dir, args.offline, True, args.grading == "on", args.road_widths == "on",
                      args.surface == "on", None, logging.getLogger().level,
                      args.airports == "on", args.sea == "on", NICE_ROADS, roadsmod.ROADS_IN_GROUND))
    make_fast_pool = None
    if road_pool is not None:
        def make_fast_pool(n):          # (the saved areas' fast mode: Server.fast_precache)
            return poolsmod.RespawningPool(
                "fast pre-cache", n, mp_context=ctx, initializer=blockworker.init,
                initargs=(args.cache_dir, args.offline, True, args.grading == "on", args.road_widths == "on",
                          args.surface == "on", None, logging.getLogger().level,
                          args.airports == "on", args.sea == "on", NICE_ROADS, roadsmod.ROADS_IN_GROUND))
    far_bld_pool = None
    if road_pool is not None and args.buildings == "on":
        far_bld_pool = poolsmod.RespawningPool(   # far building tiles (blockworker.far_buildings_tile)
            "far building", 1, mp_context=ctx, initializer=blockworker.init,
            initargs=(args.cache_dir, args.offline, True, args.grading == "on", args.road_widths == "on",
                      args.surface == "on", None, logging.getLogger().level,
                      args.airports == "on", args.sea == "on", NICE_FAR_BUILDINGS, roadsmod.ROADS_IN_GROUND))
    log.info("startup: preparing world folders and services")
    server = Server(args.user_dir, args.cache_dir, max(args.workers, getattr(args, "far_max", args.workers)), kfun, args.hysteresis,
                    args.far_radius, fmt,
                    imagery=args.imagery, near_visual=args.near_visual, tex_near=args.tex_near,
                    tex_far=args.tex_far, tone=args.tone, near_radii=radii,
                    leaf_on=args.leaf_on == "on", season_mode=args.season,
                    status_interval=args.status_interval, cells=args.cells, tex=args.tex, block=args.block,
                    horizon=args.horizon, deshadow=args.deshadow, colormatch=args.colormatch,
                    weather=(weathermod.Service(fetch=lambda lat, lon: weathermod.fetch(lat, lon, net=net))
                             if args.weather == "on" else None),
                    gravity=(gravitymod.Provider(os.path.join(args.cache_dir, "gravity"), net=net)
                             if args.gravity == "on" else None), net=net,
                    credits=creditsmod.Credits() if args.credits == "on" else None, cache=disk,
                    retex_rate=args.retex_rate,
                    surface=(surfacemod.Surface(mapdata) if args.surface == "on" else None),
                    mapdata=mapdata if args.roads == "on" else None,
                    grading=args.grading == "on" and args.roads == "on",
                    road_widths=args.road_widths == "on" and args.roads == "on",
                    sidewalks=args.sidewalks == "on",
                    buildings=args.buildings == "on" and args.roads == "on",
                    gas_stations=args.gas_stations == "on" and args.roads == "on",
                    regional_poles=args.regional_poles,
                    traffic=args.traffic == "on" and args.roads == "on",
                    block_pools=block_pools, far_pool=far_pool, road_pool=road_pool, airports_db=airports_db,
                    far_bld_pool=far_bld_pool,
                    sea=seacarvemod.SeaCarver(mapdata, args.cache_dir) if args.sea == "on" else None,
                    vegetation=(vegetationmod.Vegetation(mapdata, args.cache_dir) if args.trees == "on" else None))
    server.make_fast_pool = make_fast_pool
    server.update_check = args.update_check == "on"
    server.geo_tiles = args.geo_tiles == "on"
    server.global_far = args.global_far == "on"
    server.lod_px = args.lod_px
    loop = asyncio.get_running_loop()

    def areas_changed():            # (any thread) every connection gets the saved areas' progress
        def send():
            if server.areas is not None:
                m = server.areas_message()
                for c in list(server.connections):
                    c.send(m)
        loop.call_soon_threadsafe(send)
    try:            # saved areas downloaded ahead (areas.py): the list, its pins, the download thread
        server.areas = areasmod.Areas(args.cache_dir, server, notify=areas_changed).start()
    except Exception as e:      # (no saved areas; never the server)
        log.warning("saved areas: %s", e)
    try:            # the map's cache view: its index built in the background from the cache's keys (covermap.py)
        server.disk_cache = disk            # (net_note: a drive short of its free reserve, cache.disk_short)
        server.covermap = covermapmod.CoverMap(disk, args.cache_dir, mapdata if args.roads == "on" else None,
                                               server.world_signature).start()
    except Exception as e:      # (the map's view stays empty; never the server)
        log.warning("cache map: %s", e)

    server.game_cpus = getattr(args, "game_cpus", None)    # (cpupin.py: the game pinned at each hello)
    server.server_cpus = getattr(args, "server_cpus", None)
    server.photo_pool = photo_pool
    server.boost_cpus = (cpupin.boosted(server.server_cpus, server.game_cpus)
                         if getattr(args, "load_boost", "on") == "on" else None)
    # (a driving player's catch-up lends less: half the game's cores for the whole of a 150 mph drive took its frame
    # rate from ~52 to ~40 fps, nyc_speed.py 2026-11-02)
    server.catchup_cpus = (cpupin.boosted(server.server_cpus, server.game_cpus, CATCHUP_LEND_SHARE)
                           if server.boost_cpus else None) or server.boost_cpus

    async def net_log():
        while True:
            await asyncio.sleep(NET_TICK_S)
            line = await asyncio.get_running_loop().run_in_executor(None, net.tick)
            if line:
                rate = net.throughput()     # (a slow connection shows here: GitHub #241's stuck-looking roads)
                log.info("net: %s%s", line, "" if rate is None else f"; large downloads at {rate:.2f} MB/s")
            for c in list(server.connections):     # a host that came back while nobody posed
                c.refresh_status()
    ticker = asyncio.create_task(net_log())
    profile_task = asyncio.create_task(runtimeprofiler.stream(server))

    # Debug: event-loop stalls (every message the game sends waits for them). The loop beats; a thread
    # logs the main thread's stack when a beat is late, so the stall names its cause.
    beat = [time.perf_counter()]
    main_ident = threading.get_ident()
    if LOOP_RAISE:          # (this thread: the event loop's)
        import budget
        try:
            how = budget.raise_thread()
            if how:
                log.info("event loop priority: %s", how)
        except (OSError, AttributeError) as e:
            log.warning("could not raise the event loop's priority: %s", e)
    if SWITCH_MS:
        sys.setswitchinterval(SWITCH_MS / 1000.0)

    async def loop_beat():
        while True:
            beat[0] = time.perf_counter()
            await asyncio.sleep(LAG_TICK_S)
    lagger = asyncio.create_task(loop_beat())

    def main_stack():
        f = sys._current_frames().get(main_ident)
        stack = []
        while f is not None and len(stack) < 12:
            stack.append(f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}:{f.f_lineno}")
            f = f.f_back
        return " < ".join(stack)

    def watch_lag():
        # (long stalls always: GitHub #725's server.log had 19 s with no line right after a block was built, the
        # block's terrain_ready 25 s late and the car held at the edge; nothing said what the loop was doing)
        reported, long_at = None, None
        while True:
            time.sleep(LAG_TICK_S)
            b = beat[0]
            lag = time.perf_counter() - b - LAG_TICK_S
            if long_at is not None and long_at != b:
                log.warning("event loop: the stall above lasted %.1f s", beat[0] - long_at - LAG_TICK_S)
                long_at = None
            if lag > LAG_STALL_S and long_at is None:
                long_at = b
                log.warning("event loop stalled > %.1f s (every game message waits) in %s", lag, main_stack())
            elif lag > LAG_REPORT_S and reported != b and log.isEnabledFor(logging.DEBUG):
                reported = b
                log.debug("event loop stalled > %.2f s in %s", lag, main_stack())
    threading.Thread(target=watch_lag, daemon=True, name="lag-watch").start()

    # Debug: where this process's memory is (BNGE_TRACEMALLOC=1; numpy arrays included): the top allocation sites every
    # TRACE_S, for the small-machine budget (a Steam Deck: the main process at 2.3 GB, 2026-10-20)
    if os.environ.get("BNGE_TRACEMALLOC"):
        import tracemalloc

        def trace_mem():
            while True:
                time.sleep(TRACE_S)
                snap = tracemalloc.take_snapshot()
                top = snap.statistics("lineno")[:TRACE_TOP]
                total = sum(st.size for st in snap.statistics("filename"))
                log.info("memory: rss %.0f MB, %.0f MB traced; top: %s", blockworker._rss_mb(), total / 1e6, "; ".join(
                    f"{os.path.basename(st.traceback[0].filename)}:{st.traceback[0].lineno} {st.size / 1e6:.0f} MB"
                    for st in top))
        threading.Thread(target=trace_mem, daemon=True, name="trace-mem").start()

    async def sessions():
        while True:
            await asyncio.sleep(SESSION_SWEEP_S)
            try:
                await server.sweep_sessions_async()
            except Exception:           # a file-system error must not stop the sweeps (or the server)
                log.exception("session sweep failed")
    sweeper = asyncio.create_task(sessions())
    map_srv = None
    if getattr(args, "map_tiles", "on") == "on" and server.imagery is not None:
        try:                        # the world map's tiles, 127.0.0.1 only (a failure only loses the map)
            server.map_tiles = maptilesmod.MapTiles(server.imagery)
            map_srv, server.map_port = await server.map_tiles.start(getattr(args, "map_port", 0))
            log.info("map tiles on http://127.0.0.1:%d/tiles/{z}/{x}/{y}.jpg", server.map_port)
        except OSError as e:
            log.warning("map tiles off: %s", e)
    if telemetry_on:            # anonymous player count and data-source load (telemetry.py; BNE_TELEMETRY=off)
        import telemetry
        server.telemetry = telemetry.Reporter(
            args.cache_dir, updatesmod.local_version() or SERVER_VERSION,
            lambda: (bool(server.connections), getattr(server, "_share_hub", None) is not None),
            proc_dir=os.environ.get("BNE_NETSTATS_PROC"),
            # (the welcome page's choice and the settings page's switches: nothing before the choice; TELEMETRY.md)
            allowed=lambda: telemetry.choices(cpupin.player_settings(args.user_dir)),
            keep_id=not privacyregionmod.region()[1], perf=server.perf.take).start()
        logging.getLogger().addHandler(server.perf.errors)    # (error fingerprints: perfstats.ErrorPrints)
        log.info("telemetry: on, as the player chose on the welcome page (TELEMETRY.md; nothing before the choice; "
                 "BNE_TELEMETRY=off turns it off)")
    feed_srv = None
    if getattr(args, "companion_feed", "off") != "off":
        # the player's position for programs beside the game (companionfeed.py; docs/ADDONS.md), read-only
        import addons as addonsmod
        import companionfeed
        server.feed = companionfeed.Feed(updatesmod.local_version() or "", addonsmod.feed_origins(), args.companion_feed)
        try:
            feed_srv, fport = await server.feed.start(args.companion_port)
            log.info("companion feed on %s:%d/v1/ (read-only)",
                     "0.0.0.0" if args.companion_feed == "lan" else "127.0.0.1", fport)
        except OSError as e:
            server.feed = None
            # (its port taken: another BeamNG.Earth server runs, a test's beside the player's own, or a second app; the
            # feed is optional, the first one keeps it)
            if getattr(e, "errno", None) in (errno.EADDRINUSE, 10048):
                log.info("companion feed off: port %s is taken (another BeamNG.Earth server has it)", args.companion_port)
            else:
                log.warning("companion feed off: %s", e)
    # the programs beside the game set to start with BeamNG.Earth (addons.py, addonrun.py); ended with the server
    import atexit
    atexit.register(lambda: getattr(server, "_addons", None) is not None and server._addons.runner.stop_all())
    try:
        # (only a server the launcher started: a test's or a tool's own server never starts the player's add-ons; the
        # 2026-11-08 drive test started the real Live GPS add-on from the player's folder)
        if os.environ.get("BNGEARTH_PARENT"):
            await asyncio.get_running_loop().run_in_executor(None, server.addons().autostart)
    except Exception as e:      # (an add-on that does not start never stops the server)
        log.warning("add-ons: autostart: %s: %s", type(e).__name__, e)
    log.info("startup: opening the local connection")
    srv = await asyncio.start_server(
        lambda r, w: Connection(server, r, w).run(), "127.0.0.1", args.port)
    port = srv.sockets[0].getsockname()[1]
    print(f"earthd listening on 127.0.0.1:{port} session {server.sid} -> {server.out_dir} "
          f"(far tiles: {fmt}, near visual: {server.near_visual}, cells: {server.layout.cells} "
          f"(format {server.layout.format}), block: {server.layout.block}, "
          f"horizon: {'on' if server.horizon else 'off'}, "
          f"imagery: {','.join(args.imagery) if args.imagery else 'off'}"
          f"{'' if server.deshadow is None else ', deshadow: ' + server.deshadow.mode}"
          f"{'' if server.colormatch is None else ', colormatch: ' + server.colormatch.mode}, "
          f"weather: {args.weather}, gravity: {args.gravity}, credits: {args.credits}, "
          f"cache cap: {f'{disk.cap / cachemod.GB:g} GB' if disk.cap else 'unlimited'}"
          f"{', OFFLINE' if args.offline else ''})",
          flush=True)
    try:
        async with srv:
            await srv.serve_forever()
    finally:
        profile_task.cancel()
        try:
            await profile_task
        except asyncio.CancelledError:
            pass
        runtimeprofiler.collector.close()
        ticker.cancel()
        lagger.cancel()
        sweeper.cancel()
        if map_srv is not None:
            map_srv.close()
        if server.areas is not None:
            server.areas.close()
        net.close()                 # today's request counters
        disk.close()                # the cache's last uses


def main():
    # (the GPU kernels' Vulkan instance without gamescope's WSI layer: the server draws nothing, and in a Steam Deck's
    # Game Mode the layer failed an assert in vkCreateInstance and aborted the whole server, the game with it (Mesa
    # 26.2-devel, 2026-11-07). Its implicit layer's own off switch; this process and its workers only, not the game)
    if sys.platform.startswith("linux"):
        os.environ.setdefault("DISABLE_GAMESCOPE_WSI", "1")
    # (this run's id, inherited by its worker processes: roadz keeps one lidar decision per map cell and run in it)
    os.environ.setdefault("BNGE_ROADZ_SESSION", f"{os.getpid()}.{int(time.time())}")
    home = os.path.expanduser("~")
    ap = argparse.ArgumentParser(description="BeamNG.Earth companion server")
    ap.add_argument("--port", type=int, default=47613, help="TCP port on 127.0.0.1 (0 = any free port)")
    ap.add_argument("--user-dir", default=(os.path.join(os.environ.get("LOCALAPPDATA", home), "BeamNG", "BeamNG.drive",
                                                     "current") if sys.platform == "win32"
                                        else os.path.join(home, ".local/share/BeamNG/BeamNG.drive/current")),
                    help="BeamNG user folder; output goes to <user-dir>/temp/beamngEarth/<sessionId>/")
    ap.add_argument("--cache-dir", default=saved_cache_dir() or os.path.join(home, ".cache/beamng-earth"),
                    help="download cache root (DEM tiles go to <cache-dir>/dem/; default: the folder chosen on the "
                         "settings page or with the app's --cache-dir, else .cache/beamng-earth in the home folder)")
    ap.add_argument("--peers", default=os.environ.get("BNGE_PEERS", ""),
                    help="compute sharing: helpers (HOST[:PORT] ...; peerwork.py serve on them) that build far tiles "
                         "when this server's own workers are busy (default: BNGE_PEERS)")
    ap.add_argument("--workers", type=int, default=WORKERS, help="far-tile worker threads (keep <= 3: the game needs the cores)")
    ap.add_argument("--far-format", choices=farfield.FORMATS, default="cdae",
                    help="far-tile shape format: binary .cdae (default, ~5x faster to load) or Collada .dae")
    ap.add_argument("--lod-k", type=float, default=None,
                    help="far-field LOD: one K for every level (default: K(L1,L2)=3, K(L3+)=2.5)")
    ap.add_argument("--hysteresis", type=float, default=farfield.HYSTERESIS,
                    help="a refined tile stays refined until the player is (1 + h) x its switch distance away")
    ap.add_argument("--far-radius", type=float, default=farfield.RADIUS, help="far-field radius (m)")
    ap.add_argument("--nice", type=int, default=NICE, help="process niceness increment (0 = leave as is)")
    ap.add_argument("--pin", choices=("on", "off"), default="on",
                    help="Linux and Windows: the server on the last quarter of the physical cores and the game on "
                         "the rest (cpupin.py; default on; BNGE_SERVER_CORES=<n> for n cores)")
    ap.add_argument("--load-boost", choices=("on", "off"), default="on",
                    help="Linux or Windows with --pin on: after a teleport, while the player waits for the world, the server may "
                         "also use half of the game's cores (at idle priority) until the far field is done or the "
                         "player moves (cpupin.boosted; default on)")
    ap.add_argument("--sched", choices=("auto", "idle", "normal"), default="auto",
                    help="Linux: idle scheduling (the server runs only on CPU time the game leaves) or normal; auto "
                         "(default): idle unless the server has cores of its own (--pin)")
    ap.add_argument("--imagery", type=imagerymod.parse_chain, default=imagerymod.DEFAULT_CHAIN,
                    help="satellite imagery source chain, first with data wins: any of usgs,eox2023,eox2016, "
                         "or 'off' for the v1.2 palette look (default: usgs,eox2023,eox2016)")
    ap.add_argument("--near-visual", choices=NEAR_VISUALS, default="mesh",
                    help="mesh: the ground near the player is drawn by 64/128 m tiles and the terrain block is "
                         "physics only (default); terrain: the v1.2 look (the terrain block is drawn); detail: the "
                         "mesh layout with the physics block drawn as the near ground (aerial photo base colour, "
                         "surface detail textures)")
    ap.add_argument("--cells", type=int, choices=farfield.CELL_COUNTS, default=None,
                    help="tile cells per edge: 64 = docs/V1_4_SPEC.md M1 (FORMAT 4, exact level L-1; the "
                         "default in mesh mode), 32 = the v1.3.1 layout (FORMAT 3; terrain mode always)")
    ap.add_argument("--block", type=int, choices=BLOCK_SIZES, default=None,
                    help="terrain block samples per edge (heightmap.SQUARE apart, 1 m): 1024 = 1 km on a 128 m grid "
                         "(the default), 2048 = 2 km on a 256 m grid (mesh mode only)")
    ap.add_argument("--horizon", action=argparse.BooleanOptionalAction, default=None,
                    help="docs/V1_4_SPEC.md M1h: the far field out to the line-of-sight horizon (a disc up to "
                         "1500 km from the eye's height above sea level and the terrain in sight, L10 tiles, "
                         "per-tile horizon cull; hello.horizon, status.radius; mesh mode only; on by default in "
                         "mesh mode since 2026-10-11, --no-horizon for the fixed far radius)")
    ap.add_argument("--tex", type=parse_tex_list, default=None,
                    help="texture size (px) per level from the finest, the last value repeating "
                         "(default: 64 cells 256,256,256,512; 32 cells from --tex-near / --tex-far)")
    ap.add_argument("--tex-near", type=parse_tex_px, default=TEX_NEAR,
                    help="32 cells: texture size (px) of the two finest levels")
    ap.add_argument("--tex-far", type=parse_tex_px, default=TEX_FAR, help="32 cells: texture size (px) of the other levels")
    ap.add_argument("--leaf-on", choices=("on", "off"), default="on",
                    help="leaf-off imagery (Maryland's six-inch photos: bare trees, dormant grass) takes the summer "
                         "mosaic's colour in its vegetation (season.py; default on)")
    ap.add_argument("--season", choices=("auto", "spring", "fall", "winter", "off"), default="auto",
                    help="the season in the imagery and the trees' leaves: auto = today's date at the place (fall "
                         "colours from about October at 40 N, bare winter canopy from late November, the spring "
                         "flush in April), spring / fall / winter = that season in full, off = the summer look "
                         "(season.py, the mod's seasonlook.lua)")
    ap.add_argument("--tone", type=float, default=1.0,
                    help="strength of the imagery tone curve (0 = the imagery as downloaded)")
    ap.add_argument("--deshadow", choices=deshadowmod.MODES, default="off",
                    help="undo the capture sun's shading in the USGS imagery (docs/research/"
                         "2026-10-10-deshadow-prototype.md): lambert = slopes only, full = slopes and cast "
                         "shadows of the terrain; off = the imagery as served (default)")
    ap.add_argument("--colormatch", choices=colormatchmod.MODES, default=None,
                    help="remove the colour seams between NAIP flights in the USGS imagery (docs/research/"
                         "2026-10-10-imagery-colormatch.md): s2 = the seam feather against Sentinel-2 (EOX), "
                         "only near detected survey seams (the default when the imagery chain has USGS, since "
                         "the in-game checks of 2026-10-11); off = the imagery as served")
    ap.add_argument("--retex-rate", type=float, default=RETEX_RATE,
                    help="--colormatch / --deshadow: textures of tiles on screen rendered again (and swapped in the "
                         "game with far_retex) per second at most once a region they were rendered without is ready, "
                         "nearest first, one at a time on the tile pool (0 = none: such a tile keeps its texture until "
                         "it is sent again)")
    ap.add_argument("--near-radii", type=parse_radii, default=None,
                    help="mesh mode: radii in m around the player of the exact level and the next one "
                         "(L-2, L-1 with 32 cells; L-1, L0 with 64; default 150,400)")
    ap.add_argument("--status-interval", type=float, default=STATUS_INTERVAL,
                    help="re-send the last status with the current far-tile backlog at most every S seconds "
                         "while it changes (0 = status on events only, as v1.3.1; tools/drive.py runs use 1)")
    ap.add_argument("--lod-px", type=float, default=farfield.SSE_TAU_PX,
                    help="screen-space LOD: far tiles also split while their split would pop more than this many "
                         "pixels on the player's screen (camera, field of view, resolution; 0: distance rule only)")
    ap.add_argument("--update-check", choices=("on", "off"), default="on",
                    help="ask GitHub for a newer version after a game connects (updates.py; off: tests)")
    ap.add_argument("--weather", choices=("on", "off"), default="on",
                    help="real weather at the player from Open-Meteo.com (keyless; every 10 min or 25 km)")
    ap.add_argument("--gravity", choices=("on", "off"), default="on",
                    help="real gravity at the player (gravity.py: WGS84 normal + EGM2008 from <cache-dir>/gravity/, "
                         "every 2 km or 200 m of height; normal gravity while EGM2008 downloads, once, 74 MB)")
    ap.add_argument("--bridge-profile", choices=("curve", "legacy"), default="curve",
                    help="bridge decks: curve (grades of 5 %% up to what they must clear, joined by vertical curves, "
                         "a chain solved as one line) or legacy (deck_heights' line, tents, smoothing)")
    ap.add_argument("--telemetry", choices=("on", "off"),
                    default="off",      # (the launcher turns it on for players: tests and direct runs never report)
                    help="anonymous usage reports (telemetry.py: players online, data-source load; default on, "
                         "BNE_TELEMETRY=off; never in --offline)")
    ap.add_argument("--offline", action="store_true",
                    help="no network: serve elevation and imagery from the disk cache only (uncached tiles get "
                         "the palette, uncached terrain fails); the status says 'Offline - using cached data'")
    ap.add_argument("--roads", choices=("on", "off"), default="on",
                    help="Overture roads near the player (roads.py): DecalRoads in the mod (paved asphalt, "
                         "gravel / dirt tracks) within ~640 m, in 256 m tiles; off = none")
    ap.add_argument("--airports", choices=("on", "off"), default="off",
                    help="runways graded into the ground and the tiles (OurAirports, airportgrade.py); "
                         "the data is downloaded once (~5 MB) and refreshed weekly")
    ap.add_argument("--sea", choices=("on", "off"), default="off",
                    help="sea and lake water (seacarve.py, seatiles.py): the ground carved under Overture's ocean "
                         "and lake polygons and the mod's WaterBlock tiles that follow the Earth's curve")
    ap.add_argument("--geo-tiles", choices=("on", "off"), default="on",
                    help="content in the global tiles' own frames (geotiles.py: the vegetation, the road tiles' content, "
                         "the far buildings, skyline and far bridges), placed by the mod and kept across rebases "
                         "instead of built again for each anchor; off: per anchor (A/B)")
    ap.add_argument("--global-far", choices=("on", "off"), default="on",
                    help="far-field levels from 3 (2 km) up as global Web-Mercator tiles in their own frames "
                         "(farfield.select_global): built once, cached for any anchor, kept by the mod across rebases; "
                         "needs --geo-tiles on; off: the anchor's quadtree at every level (A/B)")
    ap.add_argument("--trees", choices=("on", "off"), default="off",
                    help="trees, bushes and rocks near the player (vegetation.py): from the canopy height model and "
                         "land cover, Forest items of stock shapes in the mod")
    ap.add_argument("--road-processes", type=int, default=ROAD_PROCESSES,
                    help="processes building the road tiles (--block-process on); 0 = threads in the server; left at "
                    "the default: 2-4 by the machine's threads and memory (budget.road_processes)")
    ap.add_argument("--download-ahead", choices=("on", "off"), default="on",
                    help="fetch the elevation tiles and map cells 10-60 s ahead of a driving player (AHEAD_S)")
    ap.add_argument("--route-prefetch", choices=("on", "off"), default="on",
                    help="prefetch a planned GPS route's corridor (elevation tiles and road/building map cells) up "
                         "to --route-prefetch-km ahead, at the same low priority as --download-ahead")
    ap.add_argument("--route-prefetch-km", type=float, default=earthbase.ROUTE_AHEAD_MAX_M / 1000.0,
                    help="how far along a planned GPS route to prefetch (15-25 km is sane; 0 disables it)")
    ap.add_argument("--detail-ring", type=float, default=earthbase.DETAIL_RING_M,
                    help="detail mode: metres of L-1 tiles around the drawn block (0 = none)")
    ap.add_argument("--background-duty", type=float, default=pacer.CLI_DUTY,
                    help="share of the time the background imagery analyses (colour seams, de-shadow) may run "
                         "(pacer.py): their memory traffic slowed the game's rendering up to 2x; 1 = unpaced")
    ap.add_argument("--sidewalks", choices=("on", "off"), default="on",
                    help="raised sidewalks (curbs and slabs, sidewalks.py) with the road tiles; off = the roads "
                         "without them")
    ap.add_argument("--buildings", choices=("on", "off"), default="off",
                    help="3D buildings from the Overture footprints with the road tiles (buildings.py; needs --roads "
                         "on): heights and roofs predicted where the data has none")
    ap.add_argument("--regional-poles", choices=("off", "fr-demo", "fr-all", "fr-varied", "europe"), default=os.environ.get("BNE_REGIONAL_POLES", "off"),
                    help="regional furniture: off=stock, fr-demo=50/50 poles, fr-all=concrete poles, "
                         "fr-varied=French kit; europe=country-specific European scenery and Quebec/Turkish STOP lettering; "
                         "worldwide country polygons at each object, French assets in FR only (default off)")
    ap.add_argument("--gas-stations", choices=("on", "off"), default="off",
                    help="gas stations with refuelling at real stations (gasstations.py; Overture places, needs "
                         "--roads on): West Coast USA's pumps and canopies, the stock refuel prompt")
    ap.add_argument("--traffic", choices=("on", "off"), default="on",
                    help="the AI road graph around the player for BeamNG's own traffic and AI (navgraph.py; needs "
                         "--roads on): the mod hands it to map.reset. On by default since 2026-10-14: whether cars "
                         "spawn is the player's choice in the game (level-start options, radial menu)")
    ap.add_argument("--road-widths", choices=("on", "off"), default="on",
                    help="street widths measured between the buildings that line them (roadwidth.py; Overture "
                         "buildings); off = class defaults")
    ap.add_argument("--grading", choices=("on", "off"), default="on",
                    help="grade the ground under roads (grading.py: smooth along, flat across, 1:2 slopes back "
                         "to the natural ground) in the physics block and the near mesh tiles; needs --roads on")
    ap.add_argument("--block-process", choices=("on", "off"), default="on",
                    help="build the follow / prefetch physics blocks in processes of their own (blockworker.py: one "
                         "for the mod's requests, one for the prefetch; their own cores, off this process's "
                         "interpreter lock); off = in this process's threads")
    ap.add_argument("--surface", choices=("on", "off"), default="on",
                    help="ground models under the physics block from ESA WorldCover land cover and the slope "
                         "(surface.py: asphalt, grass, dusty / loose dirt, rock, snow, mud, dirt), imported as 8 "
                         "terrain layers with each block (the mod's grass grows on them); off = dirt everywhere")
    ap.add_argument("--map-tiles", choices=("on", "off"), default="on",
                    help="serve the in-game world map's tiles (the imagery chain, z/x/y JPEG) over HTTP on "
                         "127.0.0.1; its port goes in hello.mapPort (needs imagery)")
    ap.add_argument("--companion-feed", choices=("off", "local", "lan"), default="local",
                    help="the player's position, read-only, for programs beside the game (companionfeed.py, "
                         "docs/ADDONS.md): local = this computer only (default), lan = also the local network")
    ap.add_argument("--companion-port", type=int, default=47615, help="the companion feed's port")
    ap.add_argument("--map-port", type=int, default=0,
                    help="port of the map tile endpoint (default 0: any free port, sent in hello.mapPort)")
    ap.add_argument("--credits", choices=("on", "off"), default="on",
                    help="'credits' messages: the imagery / elevation / weather sources of the tiles within 2 km "
                         "of the player and of the far field, for the mod's on-screen data credits (on change, "
                         "at most once a second)")
    ap.add_argument("--cache-cap", type=parse_cache_cap, default=None, metavar="GB",
                    help="size cap of the download cache in GB (0 = unlimited; default: the last cap set on the "
                         f"mod's settings page, else {cachemod.DEFAULT_CAP_GB:g}); least recently used files go, "
                         "never what this session used within the hour, the gravity model or netstats/")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    try:
        bad = _app_modules()[0].drive_relative_problem(args.cache_dir)
    except Exception:
        bad = None
    if bad:                     # (GitHub #895: c:\games\x with its backslashes dropped went to the current folder;
        ap.error(f"--cache-dir: {bad}")     # checked before anything writes there: cdn.chain's cdn.json)
    args.cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    os.environ["BNE_BRIDGE_PROFILE"] = args.bridge_profile     # read by roads.py here and in the block workers
    args.cells = resolve_cells(args.cells, args.near_visual)
    args.block = resolve_block(args.block, args.near_visual)
    args.horizon = resolve_horizon(args.horizon, args.near_visual)
    args.colormatch = resolve_colormatch(args.colormatch, args.imagery)
    if args.imagery:                # (the project's tile host where its switch turns it on: cdn.py)
        import cdn as cdnmod
        try:
            args.imagery = cdnmod.chain(args.imagery, netfetch.Fetcher(offline=args.offline), args.cache_dir)
        except Exception as e:
            log.warning("cdn: %s: %s", type(e).__name__, e)
    if args.cells != 32 and args.near_visual not in MESH_LAYOUT:
        ap.error("--cells 64 needs --near-visual mesh (terrain mode is frozen at the v1.3.1 layout)")
    if args.block != TERRAIN_BLOCK and args.near_visual not in MESH_LAYOUT:
        ap.error(f"--block {args.block} needs --near-visual mesh (terrain mode is frozen at its layout)")
    if args.horizon and args.near_visual not in MESH_LAYOUT:
        ap.error("--horizon needs --near-visual mesh (terrain mode is frozen at the v1.3.1 layout)")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("download cache: %s", args.cache_dir)
    raise_open_files()
    import fast
    try:
        fast.require(fast.MIN_VERSION, "the world server")     # at start, not in the middle of a drive
    except fast.NativeMissing as ex:
        log.error("%s", ex)
        raise SystemExit(str(ex))
    import sampler
    sampler.maybe_start()           # BNGE_SAMPLE=<path>: a sampling profile of this process's threads (debugging)
    import budget
    try:                            # the settings page's Performance choices (GitHub #529): before the priority and
        args.nice, chosen = cpupin.apply_player_settings(args.user_dir, args.nice, NICE)    # the cores' split below
        if chosen:
            log.info("the settings page: the server on %s", ", ".join(chosen))
    except Exception as e:          # (the defaults then)
        log.warning("could not read the settings page's server choices: %s", e)
    if args.nice:
        try:
            # all threads: below the game's main and physics threads (Windows: the "below normal" class)
            how = budget.lower_priority(args.nice)
            if how:
                log.info("server priority: %s", how)
        except (OSError, AttributeError) as e:
            log.warning("could not lower the process priority: %s", e)
    if args.sched == "auto":
        # Idle scheduling keeps the server off the game's CPU time when they share cores; with cores of its own (the
        # split below) it only made every server process equal: the workers' niceness (the ground under the car
        # first, NICE_ROADS / NICE_FAR) counts for nothing between idle-class tasks, and on a Steam Deck a teleport's
        # block took 105-170 s behind the far field and the road tiles (2026-10-20; 1.6 s on a desktop)
        split = cpupin.split(budget.SMALL_SERVER_SHARE if budget.small() else cpupin.SERVER_SHARE) \
            if args.pin == "on" and (sys.platform.startswith("linux") or sys.platform == "win32") else (None, None)
        args.sched = "normal" if split[0] else "idle"
    if args.sched == "idle" and hasattr(os, "sched_setscheduler") and hasattr(os, "SCHED_IDLE"):
        # Linux: CPU time only when a core would idle otherwise (the worker processes and threads made after this
        # inherit it). A recorded session (2026-10-15): with the server over 3 cores the game's physics fell behind
        # real time in 22 % of seconds (5 % under 1 core) and frame spikes nearly tripled; niceness alone still
        # gives it slices of the game's cores
        try:
            os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
        except OSError as e:
            log.warning("could not set idle scheduling: %s", e)
    if os.environ.get("BNGE_TRACEMALLOC"):      # (before the caches are made: main_async's trace_mem reports them)
        import tracemalloc
        tracemalloc.start(1)
    game_cpus = server_cpus = None
    small = budget.apply()
    machine_profile(args, small)
    if os.environ.get("BNGE_FAST_BASE_V"):     # (A/B: FAST_BASE_V in m/s)
        earthbase.FAST_BASE_V = float(os.environ["BNGE_FAST_BASE_V"])
    if args.pin == "on" and (sys.platform.startswith("linux") or sys.platform == "win32"):   # (Windows: KyzoFox, PR #112)
        try:
            server_cpus, game_cpus = cpupin.split(budget.SMALL_SERVER_SHARE if small else cpupin.SERVER_SHARE)
            cpupin.pin_self(server_cpus)
            if server_cpus:
                log.info("cores: the server on CPUs %s, the game on %s (cpupin.py; --pin off)",
                         ",".join(map(str, sorted(server_cpus))), ",".join(map(str, sorted(game_cpus))))
        except OSError as e:
            log.warning("could not split the cores: %s", e)
            game_cpus = None
    args.game_cpus = game_cpus
    args.server_cpus = server_cpus if game_cpus else None
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
