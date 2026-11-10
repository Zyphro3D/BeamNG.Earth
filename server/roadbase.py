"""Road builder, base layer (split out of roads.py): the shared constants, the map features' attribute readers (_osm,
_rule_value), the per-cell facts cache and the polyline helpers every other road module uses.
"""
import json
import logging
import math
import collections
import hashlib
import os
import threading
from collections import namedtuple
import numpy as np
import roadmesh
import staging


log = logging.getLogger("earthd.roads")

VERSION = 1
TILE = 256.0            # m, tile side in the anchor frame
SEND_M = float(os.environ.get("BNE_ROADS_SEND_M", "640"))     # tiles whose square comes this close to the player
                                                                # (Euclidean, tiles_near: a disc) are sent
DROP_M = float(os.environ.get("BNE_ROADS_DROP_M", str(SEND_M + 320)))   # ... and dropped beyond this
OVERLAP = 2.0           # m a road reaches past its tile's edge (no gap at tile seams)
SIMPLIFY_M = 0.25       # RDP tolerance (Road Spline's layerMgr)
SIMPLIFY_Z_M = 0.05     # ... and the vertical one of lines drawn at their own heights (decks, tunnels: simplify ztol)
MAX_NODES = 100         # nodes per DecalRoad (Road Spline's chunk size)
MAX_SPACING = 12.0      # m between nodes at most (official maps: median 11-12 m)
BUILD_PER_POSE = 4      # tiles started per pose at most
# At speed the tiles reach along the car's path (the user, 2026-10-19: "Going high speeds the road texture can struggle
# to keep up ... even if we have to sacrifice some things further away from the car"): above FAST_MS the tiles within
# CORRIDOR_M of the path ahead are wanted too, up to LEAD_S of driving (LEAD_MAX_M at most), and every wanted tile is
# built in the order the car reaches it (path_key): the corridor first, the ones beside and behind the car last.
FAST_MS = 25.0          # m/s (56 mph)
# (14 s / 2 km until 2026-11-03: the measured lead sat at that cap, ~800 m at 120 mph. 30 s on the Windows test PC
# (Ryzen 3600, I-10 120 mph): road lead p10 352 -> 832 m, median 800 -> 1632 m, no poll with the car past its roads
# (10 before), block times and holds the same; the user: "the world should generate far ahead")
LEAD_S = float(os.environ.get("BNE_ROADS_LEAD_S", "30"))         # s of driving ahead
LEAD_MAX_M = float(os.environ.get("BNE_ROADS_LEAD_M", "5000"))
CORRIDOR_M = 160.0      # m either side of the path
SIDE_PENALTY = 4.0      # a tile off the path counts this many times its sideways distance (path_key)

# class -> (width m, paved by default). Classes not here (footway, path, steps, cycleway, ...) are
# left out for now.
STYLE = {
    "motorway": (11.0, True), "trunk": (9.0, True), "primary": (8.0, True), "secondary": (7.5, True),
    "tertiary": (6.5, True), "residential": (6.0, True), "living_street": (5.0, True),
    "unclassified": (5.5, True), "service": (4.0, True), "track": (3.5, False), "unknown": (5.0, True),
    # racetracks (OpenStreetMap highway=raceway, osmtags.raceways: GitHub #143): a circuit's typical 12 m unless its
    # width is mapped; no markings, sidewalks, furniture or traffic (not in those classes' sets nor navgraph.DRIVE),
    # and graded by the terrain alone (grading.UNGRADED: an oval's banking stays)
    "raceway": (12.0, True),
    # OSM highway=busway (osmtags.busway_features): its own carriageway. Overture's transportation
    # segments omit it (the Dutch Zuidtangent is absent), so the class stays idle until that append.
    "busway": (7.0, True),
}
PAVED = {"paved", "asphalt", "concrete", "metal", "wood", "tar", "bitumen"}
COBBLE = {"paving_stones", "sett", "cobblestone", "unhewn_cobblestone", "bricks",   # Overture: "paving_stones"
          "interlocking", "paver", "paver_blocks"}      # (South Asia's names, PR #146)
# interlocking concrete pavers under the names the map gives them (surface_kind: a street paved with them is a
# cobble street, not a gravel track)
COBBLE_ALIASES = {"interlocking_pavers", "interlocking_stones", "pavers", "block_pavers", "paving_stone",
                  # OSM's surface=stone is mostly stone paving (flagstones, slabs), not a rock track (merge review of
                  # PR #119, 2026-10-22)
                  "stone", "stones"}
GRAVEL = {"gravel", "fine_gravel", "compacted", "pebblestone", "wbm", "laterite"}   # (water-bound macadam: PR #146)
# ... and the values that do mean "no hard surface": an unrecognised one is not one of these (the class decides)
UNPAVED = {"unpaved", "ground", "dirt", "earth", "mud", "grass", "grass_paver", "sand", "soil",
           # the rock family: a rough, unpaved surface, never a marked asphalt street (gridmap v2's Rock is a
           # TerrainMaterial — a heightfield shader, not a surface a decal can carry: 2026-10-22)
           "rock", "rocky", "bedrock",
           # more of OSM's recognised unpaved values, which an unrecognised value must not outrank
           "clay", "shells", "woodchips"}
# kinds with a hard, built surface: carriageway extras (parking, strips, sidewalks) apply
BUILT = {"paved", "cobble"}
# surface kind -> DecalRoad look (stock materials, loaded by the mod from the official levels)
LOOK = {
    "paved": {"m": "road_asphalt_2lane", "tl": 11.0, "p": 20},
    # Italy's cobblestone (a mesh material with groundType COBBLESTONE): a DecalRoad stretches its
    # texture across the whole width, so cobbled roads are drawn as parallel strips of about
    # "strip" m with a square texture repeat, which keeps the stones their size
    "cobble": {"m": "italy_cobblestone", "tl": 2.5, "p": 20, "strip": 2.5},
    "gravel": {"m": "m_dirt_road", "tl": 88.0, "p": 21},
    "dirt": {"m": "m_dirt_road", "tl": 88.0, "p": 21},
}
# a `surface=concrete` street is still a paved one (markings, widths, the tyres' asphalt feel) but is drawn with
# gridmap v2's plain concrete instead of asphalt (the user, 2026-10-22: "for concrete road surfaces")
CONCRETE = {"m": "m_gm_concrete_plain", "tl": 11.0, "p": 20}
# Dirt and gravel strips (from PR #455 by AnthonyGraceffa): with the carriageways in the ground (ROADS_IN_GROUND) a dirt
# or gravel road still gets a mesh of BeamNG's continuous rutted road strip over the ground photo's tint and its
# terrain layer (surface.PHOTO_ROAD_RGBA, paint_roads), fading into the ground at its sides and open ends (roadmesh
# EDGE_FADE_M, END_FADE_M). The strips of /assets/materials/decalroad/dirt, which Johnson Valley paints its dirt roads
# with, as our copies (dirtroad.materials.json: the stock opacity maps are patchy, ours is dirt_fade_o on the second
# uv; opacityFactor 0.85 lets the photo through). A variation's albedo is 16 times as long as wide (256 x 4096): its
# texture length is "aspect" times the road's width, so a repeat stays square; the gravel strip (1024 x 2048) twice
# the width. "vary": one road keeps one variation its whole length (look_of). Gravel light grey-beige, dirt brown.
# BNE_DIRT_STRIPS=off: no strip (the tint and the terrain layer alone), the old m_dirt_road look where one is drawn.
DIRT_STRIPS = os.environ.get("BNE_DIRT_STRIPS", "on") not in ("off", "0")
DIRT_VARIATIONS = ("bne_dirt_variation_01", "bne_dirt_variation_02", "bne_dirt_variation_03", "bne_dirt_variation_04")
DIRT_MATERIALS = DIRT_VARIATIONS + ("bne_dirt_gravel",)    # (prewarm.py)
DIRT_LOOK = {
    "gravel": {"m": "bne_dirt_gravel", "tl": 8.0, "p": 21, "aspect": 2.0},
    "dirt": {"m": DIRT_VARIATIONS[0], "tl": 56.0, "p": 21, "aspect": 16.0, "vary": DIRT_VARIATIONS},
}
DIRT_SPLIT = 4000       # nodes per strip chunk (no DecalRoad node cap: the chunks are meshed, roadmesh.build)
STRIP_JOIN_M = 1.0      # m: a strip's end this near another strip road's line joins it (no end fade there)


def terrain_surface(spec):
    """'rock', 'grass' or None: a spec whose surface belongs to the ground layer (surface.py) and must not be drawn
    as a road decal. A decal is a smooth ribbon; these surfaces are not, and the terrain materials that describe them
    (gridmap v2's Rock / Grass) cannot be carried by one."""
    if spec.surf in ROCKISH:
        return "rock"
    if spec.surf in GRASSY:
        return "grass"
    return None


def surface_material_kind(spec):
    """The material the ground under a spec is painted with (surface.py): 'rock' / 'grass' for those surfaces, else
    the spec's kind (paved, cobble, gravel, dirt)."""
    return terrain_surface(spec) or spec.kind


def look_of(spec, fid=None):
    """A spec's road look: gridmap v2's concrete for a `surface=concrete` street, a dirt or gravel strip (DIRT_LOOK:
    texture length its aspect times the width, one variation per road id fid), else its kind's (LOOK)."""
    if spec.kind == "paved" and spec.surf == "concrete":
        return CONCRETE
    look = DIRT_LOOK.get(spec.kind) if DIRT_STRIPS else None
    if look is None:
        return LOOK[spec.kind]
    vary = look.get("vary")
    if vary:
        n = 0 if fid is None else int.from_bytes(hashlib.blake2s(str(fid).encode(), digest_size=2).digest(), "little")
        look = {**look, "m": vary[n % len(vary)]}
    return {**look, "tl": round(float(look["aspect"]) * float(spec.width), 2)}
Spec = namedtuple("Spec", "width kind urban cls oneway walk dir kmh gap median lanes surf link sides parking mountain france state",
                  defaults=(False, 2.5, 0, None, None, None, None, None, False, None, True, False, False, None))   # median: m to the other carriageway's centre line
# (state: the US state / region whose marking profile the road takes, roadmarkings.profile; PR #843)
# (parking: whether a town street's carriageway holds parking lanes: _swedish_specs; mountain: a mountain pass road,
# _mountain_roads; france: PR #505's French road scenery, BNE_FR_ROADS)
# (sides: the sidewalk sides OpenStreetMap's sidewalk tag gives (_osm_sidewalks; walk_sides' +1 left / -1 right of the
# geometry, () none), None where it has none: the class and land-cover rule, has_sidewalks; from mpsteppe's PR #292)
# (lanes: resolved count, both directions on a two-way road, including highway/ramp defaults; None unknown elsewhere.
# osmtags.py / roadlanes.py retain the mapped evidence separately; surf: the surface
# rule value's base token, e.g. "concrete": the look's concrete, roads.py look_of; link: a ramp / slip road, Overture's
# subclass "link", its class the road's it leads to)
# road_specs' result per feature (walk: the sidewalk width, measured from the buildings where known: roadwidth.py;
# dir: a one-way road's direction along its geometry, +1 / -1, 0 two-way; kmh: its speed limit, None unknown;
# gap: m of road per junction along it (its connectors), None unknown)
# (width = carriageway). One-way driving widths (Overture's divided roads are two one-way segments, each
# drawn before as a full two-way street: 2026-10-11 orthophoto of Broadway, 14-15 m drawn for 8-9 m)
ONEWAY_STYLE = {"primary": 6.5, "secondary": 6.5, "tertiary": 4.0, "residential": 3.5, "living_street": 3.5,
                "unclassified": 3.5, "service": 3.5, "unknown": 3.5}

# Sidewalks and markings (user 2026-10-11, after the first street-view comparison of Fells Point:
# real streets have concrete sidewalks behind a curb and a double yellow centre line). Stock West
# Coast USA materials. renderPriority: lower draws on top (WCUSA: lines 2-6, the asphalt strip 20).
SIDEWALK = {"m": "sidewalk_concrete", "tl": 6.0, "p": 16, "w": 2.5}      # beyond the carriageway, city streets
# Storm drain inlets: a metal grate in the gutter (Driver Training's m_metal_grid_drain), DRAIN_IN m
# before the real end of each sidewalk run (one per corner, where real inlets sit) and every
# DRAIN_EVERY m along long runs (each is a DecalRoad: kept to ~40 per city tile)
DRAIN = {"m": "m_metal_grid_drain", "tl": 2.0, "p": 12, "w": 0.45, "len": 1.0}
DRAIN_IN, DRAIN_EVERY = 2.5, 90.0
GUTTER = {"m": "gutter1", "tl": 8.0, "p": 14, "w": 0.5}                  # at the curb line
CENTRE = {"tl": 12.0, "p": 6, "w": 0.12, "gap": 0.12}                    # a double line: two lines this far from the centre
EDGE_LINE = {"m": "line_white", "tl": 6.4, "p": 2, "w": 0.15, "inset": 0.3}
# The asphalt's edge where no curb ends it: East Coast USA's decal (a ragged opaque edge of gravel and dirt fading
# into the ground; its texture's u = 1 side lies on the asphalt, the opaque edge EDGE_DECAL_AT across it). The
# ground photo is ~2 m a pixel and the terrain's materials change on its 2 m grid: the edge was blurred and stepped
# (the user, 2026-10-28: "the edge of the road is blurry and slightly jagged").
EDGE_DECAL = {"m": "ECA_road_edge_dirt", "tl": 6.4, "p": 1, "w": 1.6, "layer": "edge"}
EDGE_DECAL_AT = 0.95
LADDER_LINE_W = 0.30            # a ladder crosswalk's side lines (MUTCD 3C.04: 6-24 in)
EDGE_DECAL_GAP_M = 1.5          # ... none where another carriageway's edge is this near past the decal (gores, medians)
MEASURE_SKIP = {"service"}      # classes not measured between buildings (lots, alleys: far walls)
MEASURE_MAX = 3.0               # a measured carriageway at most this times the class default
SIDEWALK_CLASSES = {"residential", "tertiary", "secondary", "primary", "unclassified", "living_street"}
CENTRE_CLASSES = {"primary", "secondary", "tertiary", "trunk"}
EDGE_LINE_CLASSES = {"primary", "secondary", "tertiary", "trunk", "motorway"}
LANE_CLASSES = {"primary", "secondary", "tertiary", "trunk", "motorway"}
# urban arterials not lined by buildings (roadwidth.py measures those): 4 lanes (Moab's Main Street, US-191,
# 4 lanes + a turn lane in the photos, was drawn 9 m)
# Ramps and slip roads (Overture subclass "link", any class): one lane and its shoulders, whatever the class
# (Columbia's US-29 ramps were drawn 13 m wide, a whole motorway, 2026-10-14 photos): m paved by lanes a way.
LINK_SHOULDER_M = 1.75
LINK_M = {1: 8.3, 2: 13.1}
LINK_LANE_M = 4.8       # a one-lane ramp's lane between its shoulders
URBAN_ARTERIAL = {"trunk": 13.2}     # primaries in towns are often 2 lanes (Lancaster County's PA-340)
LANE_M = 3.3            # m per lane when counting the lanes a carriageway holds


# m per driving lane by class when OpenStreetMap gives the lane count (osmtags.py; typical design widths)
LANE_W = {"motorway": 4.0, "trunk": 3.8, "primary": 3.3, "secondary": 3.3, "tertiary": 3.1}
LANE_W_DEFAULT = 3.0
OSM_LANES = (1, 12)     # validated lane counts (also used by regional surface classification)
OSM_WIDTH_M = (2.0, 40.0)


def _osm(feats, name, i, lo=None, hi=None):
    """OpenStreetMap's value `name` of feature i (osmtags.join's attrs column osm_<name>), or None (absent, not a number,
    outside lo..hi)."""
    col = feats.attrs.get("osm_" + name)
    if col is None:
        return None
    try:
        v = float(col[i])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return v
LANE_LINE = {"m": "line_dashed_long", "tl": 12.0, "p": 6, "w": 0.12}     # West Coast USA's 3 m dash, 9 m gap
CUT_MARGIN = 0.5                # m: a sidewalk / line stops this far outside a crossing road's carriageway
MIN_PIECE_M = 3.0               # shorter sidewalk / line pieces are dropped
DASH_MIN_M = 0.3                # ... but a broken line's dash is kept down to this
WALK_STEP_M = 1.0               # m the ground under a sidewalk may lie off the road's level: further, a cutting's wall or
                                # a drop is there and the sidewalk stops (it was draped up a wall, Munich's McGraw-Graben)
NEAR_M = 40.0                   # roads this far outside the tile still cut its sidewalks / lines
# Carriageway = driving width + parking lanes on city streets or paved edge strips on rural roads
# (docs/research/2026-10-11-official-road-study.md §2: rural paved surface = AI width + 2-2.5 m;
# West Coast USA's plain streets 12.5 m for 8 m of driving: ~2.3 m of parking each side). A width
# rule, where Overture has one, is the carriageway already.
PARKING_M = 2.3
# Rural motorways: two lanes a direction (most rural Interstates and European motorways; every US Interstate was drawn
# with three, GitHub #43) between paved shoulders, the edge line along each. Untagged urban highways also get two.
# Overture has no lane counts.
MOTORWAY_RURAL_M = 7.4
SHOULDER_M = 3.0
SHOULDER_WIDE_M = 3.0     # m: a motorway carriageway of SHOULDER_WIDE_LANES lanes or more (US interstates: 10 ft both
SHOULDER_WIDE_LANES = 3   # sides from three lanes; I-95 by the Middle Branch, the aerial vs ours, the user 2026-10-31)
PARKING_CLASSES = {"residential", "tertiary", "secondary", "unclassified", "living_street"}
EDGE_STRIP_M = 1.0
# ... by class where measured narrower: rural minor roads' paved width on the official aerials (PEMA 6-inch, Lancaster
# County PA, 20 tertiary 6.4 m and 20 secondary 6.7 m against 8.5 and 9.5 drawn; NAIP, El Paso County CO, tertiary 7.2;
# 2026-10-28, the user: "compare against high res satellite photos"): no strips on tertiary and unclassified roads,
# narrow ones on secondaries
EDGE_STRIP_CLASS_M = {"tertiary": 0.0, "unclassified": 0.0, "secondary": 0.25}
URBAN_CLASS = 50                # ESA WorldCover "built-up"
URBAN_SAMPLE_M = 20.0          # uniform arc-length samples; OSM vertex density must not decide road treatment
URBAN_SHARE = 0.5               # a road is urban when this share of its distance samples is built-up ...
URBAN_BESIDE_M = 30.0           # ... and at least URBAN_BESIDE_SHARE of them have built-up land this far to one side:


URBAN_BOTH_SHARE = 0.1          # a motorway or trunk needs built-up land on both
                                # sides of this share of its samples: a roadhouse beside the Eyre Highway (Eucla: one
                                # side 33 %, the other 0 %) is no town


def _beside(ll, counts, d):
    """The points d m to the left and to the right of each vertex of the lines (lon/lat (n, 2), counts: the vertices of
    each line in turn), square to the line there; a one-vertex line's are the vertex itself. Pure."""
    left, right = np.empty_like(ll), np.empty_like(ll)
    k = 0
    for c in counts:
        p = ll[k:k + c]
        if c < 2:
            left[k:k + c] = right[k:k + c] = p
            k += c
            continue
        m = np.array([111320.0 * math.cos(math.radians(float(p[:, 1].mean()))), 110540.0])   # m per degree
        t = np.gradient(p * m, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        off = np.stack([-t[:, 1], t[:, 0]], axis=1) * d / m
        left[k:k + c], right[k:k + c] = p + off, p - off
        k += c
    return left, right


def _rule_value(table, i):
    """The value of the rule of feature i covering the largest share of it, or None."""
    if table is None or len(table.feature) == 0:
        return None
    j = table.major(i)
    return None if j < 0 else table.value[j]


HOME_RANGES_ON = os.environ.get("BNE_HOME_RANGES", "on") != "off"


def _home_cell(feats, i):
    """str of feature i's home cell (the cell of its first point), kept per layer (level_ranges asks for it at every
    call: 95k cell_of in two Midtown cells' planning, 2026-11-02)."""
    every = feats.__dict__.get("_home_cells_all")
    if every is not None:
        return every[i]
    memo = feats.__dict__.get("_home_cells")
    if memo is None:
        memo = feats.__dict__.setdefault("_home_cells", {})
    got = memo.get(i)
    if got is None:
        import mapcells
        a = feats.coord_off[feats.ring_off[feats.part_off[i]]]
        got = memo[i] = str(mapcells.cell_of(float(feats.lonlat[a, 1]), float(feats.lonlat[a, 0])))
    return got


def _home_cells(feats):
    """[_home_cell(feats, i) for every feature], made at once (mapcells.cells_of: the same cells) and kept per layer
    (lane_flow asked every feature's one by one: 175k calls, 1.3 CPU-s of a Midtown road tile, 2026-11-08)."""
    every = feats.__dict__.get("_home_cells_all")
    if every is not None:
        return every
    n = len(feats)
    try:
        import mapcells
        a = np.asarray(feats.coord_off)[np.asarray(feats.ring_off)[np.asarray(feats.part_off)[:n]]]
        ll = np.asarray(feats.lonlat, np.float64)[a] if n else np.zeros((0, 2))
        every = [str(k) for k in mapcells.cells_of(ll[:, 1], ll[:, 0])]
    except (IndexError, ValueError):
        every = [_home_cell(feats, i) for i in range(n)]    # (one by one: the error, if any, of the feature asked)
    feats.__dict__["_home_cells_all"] = every
    return every


def home_copy(feats, i):
    """(feats', j) of feature i's copy in its home cell (the cell of its first point, among the eight round feats'
    cell: mapdata's _bnge_neighbours), None where that is this cell or not at hand. A road's copies in the cells it
    crosses each see other roads round them: what depends on those (its lanes, its bridge ranges) is taken from one."""
    me = feats.__dict__.get("_bnge_cell")
    hook = feats.__dict__.get("_bnge_neighbours")
    if me is None or hook is None:
        return None
    home = _home_cell(feats, i)
    if home == me[0]:
        return None
    try:
        near = {f.__dict__["_bnge_cell"][0]: f for f in hook() if f is not None and f.__dict__.get("_bnge_cell")}
    except Exception as e:              # (this cell's own)
        log.debug("home copy: neighbours: %s: %s", type(e).__name__, e)
        return None
    other = near.get(home)
    if other is None:
        return None
    pos = other.__dict__.get("_id_pos")
    if pos is None:
        pos = other.__dict__["_id_pos"] = {fid: j for j, fid in enumerate(other.ids.tolist())}
    j = pos.get(feats.ids[i])
    return None if j is None else (other, j)


# Per-cell facts shared by the server's processes and kept for the next run (the revisit cache: one file per cell and
# kind, keyed by the cell, its map data versions and the code): a cell's road specs and level ranges, which every
# process that draws or grades its roads made for itself (~0.2 s a city cell each; the terrain blocks', the road
# tiles' and the AI graph's processes, 2026-10-25). Kept only when complete (revisit.faults unchanged).
_FACTS_LOCK = threading.Lock()
# A cell's facts made in this process, by its id (_bnge_cell: the cell, the layer, the map data's versions), beyond the
# layer object they were made for: the map data's memory cache drops and reloads a city's big layers all the time,
# and each reload made them again. On a first visit down Manhattan 89 cells' facts were made 787 times, 685 of 1081
# CPU-s again (BNGE_FACTS_LOG, 2026-11-02): kept on disk only when complete, and almost nothing was, any fault of any
# thread counting against them (revisit.thread_faults now).
_FACTS_MEM = collections.OrderedDict()
FACTS_MEM_KEPT = 640         # entries (~8 a cell: ranges, specs, bridges ...; a city cell ~0.6 MB in all; budget: fewer when small)


def clear_facts_memory():
    """Drop the cell facts kept in memory (budget.maybe_shed)."""
    with _FACTS_LOCK:
        _FACTS_MEM.clear()


FACTS_SHORT_S = 10.0        # s a cell's facts made short of something (a fault) are used before they are made again
FACTS_LOG = bool(os.environ.get("BNGE_FACTS_LOG"))


def _facts_key(feats, kind, extra):
    cid = feats.__dict__.get("_bnge_cell")
    import revisit
    if cid is None or not revisit.ON or revisit.CACHE_DIR is None:
        return None
    # (by FACTS_FORMAT, not the code: kept across updates, the user, 2026-11-02; cid carries the map data's versions)
    return revisit.format_key("cellfacts", FACTS_FORMAT, kind, cid, extra)


def _facts_load(feats, kind, extra):
    k = _facts_key(feats, kind, extra)
    if k is None:
        return None
    import revisit
    got = revisit.load(revisit.CACHE_DIR, "cellfacts", k)
    data = got[0].get("facts.json") if got is not None else None
    if data is None:
        data = _facts_baked(feats, kind, extra)
    if data is None:
        return None
    try:
        return json.loads(data)
    except ValueError:
        return None


# The facts' output format: their key in the revisit cache (kept across updates: _facts_key) and in baked packs
# (bakepack.py), in place of the code hash; the plans' key holds it too (plancache.plans_format). Bump it whenever a
# change makes other facts (level ranges, specs, lane flow, grade separations or their inputs' rules), or players keep
# the old ones; tests/test_bake_formats.py fails until it is bumped.
# Its history:
#   14: one-way motorway / trunk carriageways of other names kept from overlapping too, a pair's share and gap read
#       from either side (#2374; 13's facts were written by the night's survey with an earlier state of it)
#   13: France: motorway carriageways of 3.5 m lanes and their shoulders (franceroads); nameless one-way motorway
#       carriageways capped against each other where they overlap (#2374)
#   12: France: no sidewalks on the short unnamed pieces of a junction (franceroads.py)
#   11: France: lanes given their room, one width per street (franceroads.py)
#   10: France: town street defaults and narrower sidewalks (franceroads.py)
#   9: France: BD TOPO widths floored on through roads (ign_largeur.py)
#   8: France: BD TOPO carriageway widths (ign_largeur.py)
#   7: PR #843 (restored highway shoulders, regional merge marking profiles: Spec.state)
#   6: busways (PR #836) and walkway spans (PR #842) in the cells
#   5: PR #505's Spec.france (French road scenery flag)
#   4: pass roads with a mapped or default two-lane count stay mountain roads (the Stelvio)
#   3: PR #471 lane counts (validated highway counts, two-lane highway default) and #686 (Swedish dual carriageways,
#       2.75 m lanes)
#   2: OSM's own surface first, "unpaved" on car roads gravel, regional gravel for untagged minor roads
FACTS_FORMAT = 14


def _facts_baked(feats, kind, extra):
    """A baked pack's facts of this layer's cell (bakepack.py), as JSON bytes, or None."""
    import revisit
    cid = feats.__dict__.get("_bnge_cell")
    if cid is None or revisit.CACHE_DIR is None:
        return None
    import bakepack
    try:
        versions = json.loads(cid[2])
    except (ValueError, IndexError, TypeError):
        return None
    return bakepack.find(revisit.CACHE_DIR, bakepack.facts_key(kind, extra, cid[0]),
                         {"bake_format": bakepack.BAKE_FORMAT, "facts_format": FACTS_FORMAT, "versions": versions})


def _facts_store(feats, kind, extra, data):
    k = _facts_key(feats, kind, extra)
    if k is None:
        return
    import revisit
    revisit.store(revisit.CACHE_DIR, "cellfacts", k, {"facts.json": json.dumps(data).encode("utf-8")})


FACTS_WAIT_S = 3.0          # s a process waits for another making a cell's facts this run before it makes them too
# (a 30 s wait on a cell's road specs made a Steam Deck's first road tile ~30 s later, 2026-10-26: holders busy)


def _run_facts_path(st):
    """This server run's file of a layer's level ranges (beside its road plans: plancache.RUN_ENV), shared by its
    processes, the lidar rule's too (a session's answer for the cell: roadz decides it once, on disk), or None."""
    import revisit
    cid = st.get("_bnge_cell")
    run = os.environ.get("BNGE_PLAN_RUN")
    if cid is None or not run or revisit.CACHE_DIR is None or not revisit.ON:
        return None
    import hashlib
    h = hashlib.sha1(repr((cid, bool(LIDAR_GAPS_ON), revisit.code_key())).encode()).hexdigest()[:20]
    return os.path.join(revisit.CACHE_DIR, "plans", run, "facts", h + ".json")


def _run_facts_load(path, n):
    try:
        with open(path, encoding="utf-8") as f:
            got = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(got, dict) or len(got.get("raw", ())) != n or len(got.get("lvl", ())) != n:
        return None
    return got


def _run_facts_store(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = staging.tmp_for(path)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        staging.replace(tmp, path)
    except OSError as e:
        log.debug("cell facts %s not kept for the run: %s", path, e)


def _ranges_mem_key(st):
    """_FACTS_MEM's key of a layer's level ranges: its cell id and the lidar's switch and provider (the same all along
    in a server; tests change them)."""
    cid = st.get("_bnge_cell")
    if cid is None:
        return None
    import roadz
    return (cid, "ranges", bool(LIDAR_GAPS_ON), id(getattr(roadz, "PROVIDER", None)))


# A bridge over a road spans more than its carriageway: the road's shoulders and the abutments' slopes stand under it.
# The map's bridge range often ends close to the lower road's edge (an 18 m deck of the Baltimore-Washington Parkway
# over an 11 m local road, 2026-10-25: the ground climbed 10 m from the road's edge to the Parkway within 3 m, wedges
# into its lanes), or, crossing at an angle, short of its edge under the upper road's outer lanes. A bridge range is
# lengthened until it covers, across the upper road's whole width, the lower road's carriageway, SKEW_MARGIN_M and an
# abutment slope of SKEW_SLOPE_M either side: (lower half width + margin + slope) / sin + upper half width / tan of
# the angle between them, at most SKEW_MAX_M more per end (onto the road joined there: _skew_decks).
SKEW_ON = os.environ.get("BNE_SKEW_DECKS", "on") != "off"
SKEW_MARGIN_M = 1.5         # m past the lower road's carriageway edge kept level (its shoulder)
SKEW_SLOPE_M = 6.0          # m for the abutment's slope up to the upper road, under the deck
SKEW_MAX_M = 30.0           # m a bridge range grows at most at each end
SKEW_MIN_SIN = math.sin(math.radians(20.0))    # flatter crossings are taken as this (the reach would run away)


# A tunnel's portal moved out past a ground road crossing over the tunnel's road at it (earthfast.tunnel_reach): its
# carriageway and TUNNEL_PORTAL_CLEAR_M past it (along the road: / the sine of the crossing, at least TUNNEL_ROAD_MIN_SIN),
# a crossing up to TUNNEL_PORTAL_REACH_M outside the portal, TUNNEL_PORTAL_GROW_M at most. OSM often ends an underpass's
# tunnel inside the carriageway over it: the portal's approach cut ran through the A-7's lanes near Malaga (GitHub #450).
TUNNEL_PORTAL_CLEAR_M = 3.0
TUNNEL_PORTAL_REACH_M = 10.0
TUNNEL_PORTAL_GROW_M = 30.0
TUNNEL_PORTAL_END_M = 2.0   # m: a crossing this close to the other road's end is a junction
# A portal inside the carriageway of a road running along the tunnel (the map's ground stretch of a service tunnel laid
# in a motorway's lanes: the Gotthard road tunnel's service gallery at Goeschenen, an arch and a mound in the A2's
# lanes, 2026-10-28) moves into the tunnel until clear, TUNNEL_PORTAL_SHRINK_M at most; a tunnel shorter than
# TUNNEL_PORTAL_MIN_M left that way is none.
TUNNEL_PORTAL_SHRINK_M = 60.0
TUNNEL_PORTAL_MIN_M = 8.0


GAP_DECK_M = 60.0
GAP_JUNCTION_M = 1.5
GAP_JOIN_M = 0.75           # (_skew_decks' tolerance for a road's end at another's)


# Ground stretches the lidar shows on a structure (roadz.py, US 3DEP; the user, 2026-10-29: "Sure add the check"): a
# drawn road's ground piece that touches a deck's end (rule_level_ranges) is measured along its first LIDAR_GAP_REACH_M
# with the decks' own lidar read, and is deck where its surface goes on from the deck's end well over the DTM (an
# untagged viaduct stretch: Springfield's, a measured deck eased 10-15 m down to it). roadz answers per map cell (the
# cell holding the deck's end), the same in every process (kept on disk; a cell whose lidar is not there in time keeps
# the rule for the session). BNE_LIDAR_GAPS=off: none.
LIDAR_GAPS_ON = os.environ.get("BNE_LIDAR_GAPS", "on") != "off"
LIDAR_GAP_TOUCH_M = 1.0     # m: a ground piece's end this near a deck piece's end goes on from it
LIDAR_GAP_REACH_M = 150.0   # m of a ground piece measured from that end at most


PASSAGE_M = 40.0        # m: a tunnel range this short between two bridge ranges is a passage through a tower
TUNNEL_GAP_M = 30.0     # m: tunnel ranges of a segment this close together are one tunnel


MIN_PART = 1e-3     # parts shorter than this share of a segment are dropped


PORTAL_CLEAR_M = 1.0        # m a tunnel's portal face keeps from the edge of a carriageway it passes under, its tube
PORTAL_EXTEND_MAX_M = 25.0  # going on straight up to this far past the mapped end for it (an underpass mapped short: its
                            # portal face stood across the E6's lanes in Norway, GitHub #2)
PORTAL_JOIN_M = 1.5         # a road this near a tunnel's end meets it there (a garage ramp ending in a street), not over it
UNDERPASS_MAX_M = 80.0      # only tunnel parts this short (an underpass) under roads of PORTAL_CLASSES are carried on
PORTAL_CLASSES = {"motorway", "trunk", "primary", "secondary"}


# surfaces that belong to the ground layer, not to a road decal: rock and grass are rough, mask-driven terrain
# materials (gridmap v2 draws them exactly that way), so the mod paints their ground (surface.py) instead of laying
# a smooth ribbon over it (2026-10-22)
ROCKISH = frozenset({"rock", "rocky", "bedrock"})
GRASSY = frozenset({"grass", "grass_paver"})


# Untagged minor roads in country where the mapped ones are gravel (2026-11-03: Estonia's unclassified roads and
# tracks 100 % without a surface, Finland's unclassified 71 %, Iowa's driveways all; drawn as asphalt): a road of
# REGIONAL_CLASSES without a surface value is gravel when, in its cell and the eight around it, the roads of its class
# that have one are REGIONAL_SHARE unpaved or more by length, over REGIONAL_MIN_M of them; never in town (the
# land-cover test), never with a mapped lane count or a good smoothness (a mapper who added those looked at a paved
# road: Ontario's untagged roads are often asphalt), never tertiary and up. BNE_REGIONAL_SURFACE=off: as before.
REGIONAL_CLASSES = {"unclassified", "service", "residential", "unknown"}
REGIONAL_SHARE = 0.75
REGIONAL_MIN_M = 3000.0
# a class with less of its own: the minor classes' surfaced roads together decide, over REGIONAL_POOL_MIN_M of them
# (Estonia's unclassified roads have no surface at all, its service roads are mapped gravel; Iowa's driveways none,
# its county roads gravel)
REGIONAL_POOL_MIN_M = 1500.0


GOOD_SMOOTHNESS = {"excellent", "good", "intermediate"}


RAMP_JOIN_DEG = 1e-5        # (~1 m) a ramp's end this close to a road's end: they meet (_ramp_junction)


DUAL_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary"}
DUAL_GAP_M = (4.0, 45.0)    # centreline to centreline of a divided road's two carriageways
DUAL_SHARE = 0.6            # this share of the shorter one runs beside the other ...
DUAL_ANGLE_DEG = 25.0       # ... parallel within this


URBAN_MEDIAN_M = 1.5    # m: a city's divided road (centre lines within URBAN_MEDIAN_GAP_M): a narrow concrete median,
URBAN_MEDIAN_GAP_M = 24.0   # its carriageways out to it (Boston Street, Mapillary 2026-10-17: two lanes and parking
URBAN_DUAL_MAX_W = 12.0     # each way either side of a ~1.5 m curb; we drew 6.5 m carriageways and a 4.5 m strip)


DUAL_MIN_MEDIAN_M = 0.6    # m between a divided road's carriageways at the least (a concrete barrier): the class's width
                            # overlapped them (Munich's McGraw-Graben: two 11 m carriageways 9.6 m apart, wider than its
                            # trench, their edges up its walls)
DUAL_MIN_W = 3.5            # ... but a carriageway keeps one lane


# Lane tapers (the user, 2026-10-31: "the bridge geometry as lanes merge and split is rough"; bridgewidth.py: 54 of
# central Baltimore's 163 bridges met a road of another width): where a road goes on as one of another width (the map
# splits its ways where the lane count changes) or onto its own deck (a city street's parking lanes end at its bridge),
# what is drawn of it (markings, curbs and sidewalks, the ground's paint) runs to the width at the joint over
# TAPER_M_PER_M per metre of change instead of stepping there: half way between two ground roads, the deck's own
# width at a deck (a deck keeps its width: its parapets and collision). Only where two roads alone meet: across a
# junction the width changes as real streets' do (its curbs and crosswalks are at each road's width), and a deck that
# splits into a divided road's carriageways widened each to its own width over the other. MUTCD 3B.09's lane-shift taper is about 50 m a
# 3.6 m lane at 45 mph (L = WS).
TAPER_M_PER_M = 12.0
TAPER_M = (15.0, 90.0)          # m: a taper's length at the least and most ...
TAPER_SHARE = 0.45              # ... and at most this share of the road's length (a taper at each end)
TAPER_MIN_DW = 0.4              # m of width change that tapers
TAPER_MAX_TURN = 35.0           # degrees between two roads' directions at a joint: one going on as the other


# Lane counts through merges and splits (the user, 2026-10-31, I-95's viaduct by the Middle Branch: "the generator
# gets very confused on what the width the bridges should be as they merge and split"): OpenStreetMap counts some
# pieces of a one-way road and not others, and a piece without a count took its class's width: a 4-lane deck (18.8 m)
# took a 1-lane ramp and went on 11.8 m wide. Where the map gives none, a one-way piece takes what flows into it at a
# merge or a road going on, or out of it at a split, or its branch's share (earthfast.lane_flow); ramps count as one,
# and where the map gives no count near it a road counts as its class's default lanes (a merge with a ramp still adds
# one: the northbound deck there, no counts at all, stepped where its ramp joined).
LANE_FLOW_CLASSES = {"motorway", "trunk", "primary", "secondary"}
LANE_FLOW_MAX = 8


# Lane lines through merges and splits (the user, 2026-10-31, I-95 by the Middle Branch): each piece painted its lines
# centred on its own line, so where a 4-lane road took a ramp and went on with 5 every line jumped half a lane at the
# joint. The wider piece's markings start shifted toward the ramp (earthfast.lane_shifts): its lines go on from the
# narrower one's and the added lane opens on the ramp's side, easing back to centred over SHIFT_M_PER_M per metre.
SHIFT_M_PER_M = 40.0
SHIFT_M = (40.0, 160.0)         # m of easing at the least and most ...
SHIFT_SHARE = 0.6               # ... and at most this share of the road's length
SHIFT_MIN_M = 0.5


# OpenStreetMap's sidewalk tag (osmtags.py) -> walk_sides' sides; "separate" (a footway mapped on its own beside the
# road) and "no" give none. Other values (or none) leave the class and land-cover rule.
SIDEWALK_TAG = {"both": (1, -1), "left": (1,), "right": (-1,), "no": (), "none": (), "separate": ()}


UNBUILT_SHARE = 0.5     # a feature Overture flags is_under_construction over this share of its length is not built


# Mountain pass roads (the user, 2026-10-31: the Stelvio Pass; its aerial: 6.1 m of asphalt between edge lines, no
# centre line, a dry-stone retaining wall right at the uphill edge and a stone parapet along the valley's; drawn 8 m
# wide with a centre line on a 20-24 m graded bench, the walls and the parapets set back off the road or missing). A
# rural road whose line has MOUNTAIN_SWITCHBACKS switchbacks or more (MOUNTAIN_TURN_DEG of turn within 2 x
# MOUNTAIN_SPAN_M: earthfast.tight_bends), no mapped width and no more than MOUNTAIN_MAX_LANES mapped lanes is
# MOUNTAIN_ROAD_M wide, without edge strips
# or a centre line, and the grading keeps no verge beside it (grading.shoulder). Its whole line counts: the map's road
# is one feature over the pass, the same in every cell.
MOUNTAIN_CLASSES = {"secondary", "tertiary", "unclassified"}
MOUNTAIN_SWITCHBACKS = 3
MOUNTAIN_SPAN_M = 20.0
MOUNTAIN_TURN_DEG = 150.0
MOUNTAIN_ROAD_M = 6.0
MOUNTAIN_MAX_LANES = 2     # a mapped lanes=2 is any two-way road's count, not a width: the Stelvio's own way got it
                           # with the per-way lane tags (2026-11-04), and the pass was 7.1 m again with a centre line,
                           # a 20 m bench and walls standing between its hairpin legs (the user, 2026-11-05)
MOUNTAIN_ON = os.environ.get("BNE_MOUNTAIN_ROADS", "on") != "off"


# A pass road's hairpins are widened (the Stelvio's aerial: ~6 m legs, 10-12 m round each hairpin, the turning cars'
# swept path; the AI cut the inside of a 6 m hairpin onto the island's parapet, 2026-11-01): HAIRPIN_WIDEN_M more at
# each hairpin's apex (the middle of its run of MOUNTAIN_TURN_DEG turns), fading out over HAIRPIN_SPAN_M from it, as a
# taper of the road (road_tapers: its paint, markings, the grading's pads and parapets follow).
HAIRPIN_WIDEN_M = 4.0
HAIRPIN_SPAN_M = 22.0


SE_TOWN_MAX_KMH = 70.0      # Trafikförordningen 3 kap. 17 §: 50 km/h in a built-up area, 70 outside; so a road of 70 or
                            # more is no town street, whatever stands beside it (a 100 km/h primary read as town)
# Rural roads with no surface in the map: gravel where the tagged ones mostly are. km paved / unpaved of the tagged ones
# in seven samples of 8 km round, Boras to Norrbotten, OpenStreetMap through Overture, 2026-11-02: unclassified 97 / 748,
# service 40 / 121; tertiary (264 / 81) and residential stay paved. "unknown" is TomTom's through Overture, in no
# OpenStreetMap at all: 100 km within 25 km of Boras, forest roads drawn as 7 m of asphalt
SE_GRAVEL_CLASSES = {"unclassified", "service", "unknown"}
SE_NO_STRIPS = {"residential", "living_street", "unclassified", "service", "unknown"}   # no paved edge strips


# Roundabouts (OpenStreetMap junction=roundabout / circular: osmtags osm_junction): the circulatory roadway as wide as
# the design guides make it, not a one-way street's lane (3.5-4 m: cars clipped the central island's curb). FHWA /
# NCHRP Report 672 (Roundabouts: An Informational Guide, 2nd ed., 6.4.5 / 6.5.5): single-lane 16-20 ft (4.9-6.1 m),
# two-lane 29-32 ft (8.8-9.8 m). No parking lanes or inferred sidewalks on the ring (urban False; mapped sidewalks
# stay). (cody196's PR #363 took 9 m a lane: twice the guides.)
RING_LANE_M = 5.5               # m: a single-lane ring
RING_EXTRA_LANE_M = 4.3         # m per further circulating lane
RING_JUNCTIONS = ("roundabout", "circular")


MERGE_TOUCH_M = 1.5     # m: decks side by side this close (edge to edge) are pulled together too (the Queensboro
                        # Bridge's upper roadway's two directions met at a 0.5-2 m step, a wall and stubs on it)


END_JOIN_M = 6.0
END_JOIN_COS = 0.8
END_JOIN_GROUND_M = 2.5     # (grading's APPROACH_TOUCH_M: a road ending this near leads onto the deck)
END_JOIN_MIN_M = 10.0       # m: shorter parts keep their ends


# ... and a free end standing on another deck part's carriageway of its level (within its half width and END_ON_MARGIN_M;
# no road going on on the ground there): a ramp's end on the cross street's bridge. Held at the ground, the ramp's
# short deck rose 8 m in 6 m to its measured joint, a wall (the Springfield Interchange, 2026-10-29). Solved again with
# that end at the other deck's height where they differ by more than END_ON_STEP_M (earthfast.deck_end_on).
END_ON_MARGIN_M = 0.5
END_ON_STEP_M = 0.3


# A ramp's deck beside its road's at one height while their slabs are joined (roadmesh MEDIAN_JOIN_M: one slab, no
# parapet between): solved apart, a ramp leaving at its own grade stood lower beside the main deck within metres of the
# fork, a lip down the middle of the joined slab a car hung on (the user, 2026-10-31, an I-95 exit by the Middle
# Branch: "see how the real bridge splits": one deck until the gore). Past the gore it eases into its own profile.
TIE_JOIN_M = 6.0                # m of edge gap the slabs are joined within (roadmesh.MEDIAN_JOIN_M)
TIE_EASE_GRADE = 0.025          # rise over run of the easing from the tied height into its own profile (with its own grade
                                # on top: 7.5 % at most past a 5 % solve) ...
TIE_EASE_MIN_M = 20.0           # ... over this at the least


TIE_MEASURED_M = 0.5        # m: a part whose heights a source measured (roadz: the lidar) is held at its road's
                            # only this near it (the two measured apart at a fork), never a level apart: the tie had
                            # pulled measured ramps by I-395 1.6 and 4.9 m off the lidar's


DUAL_DECK_M = 30.0      # m: parts of one level parallel within this are one deck's carriageways (_merge_overlaps)


STACK_STEP_M = 16.0     # m between the points where an upper deck must clear the one under it (_deck_stacks)


WATER_CLEAR_SHARE = 0.15
WATER_CLEAR_MIN = 2.5
WATER_CLEAR_MAX = 40.0
WATER_CAUSEWAY_M = 1500.0
WATER_CAUSEWAY_CLEAR = 5.0
WATER_STEP_M = 16.0         # m between a wet run's clearance points
NAV_RUN_M = 100.0           # a wet run this long takes the bridge's navigation clearance (NBI item 39)
GAP_DIP_M = 2.0             # BridgeProfiles._gap: the chain's lowest ground this far under its highest end, ...
GAP_LOW_M = 1.5             # ... the stretch within this of it is the water
GAP_SEA_M = 1.0             # ... and at most this high (a river at sea level)


DECK_CREST_R = 60.0     # m: the ground under a deck is its floor rounded to vertical curves this wide (deck_profile_once)
# ---- clearance-aware deck profile ----------------------------------------------------------------------
# 2026-10-18 (a user in Portland, OR): decks came out as "two flat pieces of road joined in the middle with slight
# up, then down slopes": deck_heights' straight line between abutments, a 5 % tent where a road crosses
# under, smoothed. deck_curve plans the deck as a road designer would: grades of DECK_DESIGN_GRADE up to
# whatever must be cleared beneath (never steeper than DECK_MAX_GRADE from an abutment), joined by parabolic vertical
# curves (one moving average of a piecewise-linear profile), the whole simple chain solved as one line.
# BNE_BRIDGE_PROFILE=legacy (earthd --bridge-profile legacy) brings the old profile back for comparison.
DECK_CURVE_K = 1500.0   # m of vertical curve per unit of grade change (15 m per %)
DECK_CURVE_MIN = 16.0   # m, shortest vertical curve
DECK_CURVE_KMIN = 260.0 # m per unit grade change at least: under 2 % of grade change between 4 m deck points
DECK_ROAD_HALF = 8.0    # m either side of a crossing road's centreline the clearance is held
DECK_CURVE_STEP = 1.0   # m, the profile's working grid
DECK_DESIGN_GRADE = 0.05  # a deck climbs to (and leaves) what it must clear at this grade (DECK_MAX_GRADE at most)
APPROACH_GRADE = 0.08  # m/m: the approach road's grade read from the ground 12 m past an abutment, at most


CHORD_KINK = 0.02       # curve_profile keeps the straight line between abutments meeting the approaches within this grade
CHORD_SHELF_M = 1.0     # m
CHORD_TOL = 0.35        # m a requirement may go short: 0.21 m of it margins, 6 m headroom to 5.86 m at worst


DECK_EASE_M = 24.0      # m, the moving average run over each part after the merges (deck_profile, curve mode)


LIFT_GRADE = 0.03       # _lift_smooth: a lift over the ground is run out along the deck at this grade ...
LIFT_EASE_M = 40.0      # ... and rounded over this many m
LIFT_END_GRADE = 0.01   # by an abutment the floor rises from the abutment no faster than this


DECK_SIGMA = 12.0       # m, deck profile smoothing (at least) ...
DECK_SIGMA_SHARE = 25.0  # ... a 25th of the part's length ...
DECK_SIGMA_MAX = 40.0   # ... at most


PIN_M = 1.0             # m: a ground road's end this near a joint of decks makes the joint an abutment (_pin) ...
PIN_ROAD_M = 20.0       # ... a road at least this long (a shorter piece between decks is a gap in the map's bridge tags)


TUNNEL_COVER = 8.0      # m from the floor up to the ground over it wanted inside a tunnel (the tube's 6.5 m and rock)
TUNNEL_GRADE = 0.05     # a tunnel floor falls no steeper than this from a portal
TUNNEL_STEP = 4.0       # m between tunnel profile points
TUNNEL_WET_SHARE = 0.25 # a tunnel part with more of its points under water is left out (TunnelProfiles.wet)
TUNNEL_WET_COVER_M = 60.0   # m: water this far over the chain's lower portal is no underwater tunnel's (_mark_drowned)
TUNNEL_WET_RUN_M = 20.0 # ... and a tunnel chain passing under water this far away from its ends (_mark_drowned)
TUNNEL_OPEN_COVER = 4.0 # m: a 'tunnel' whose ground nowhere rises this far over the straight line between its portals
                        # is an underpass in the open (under a bridge the map does not draw as a deck over it): a road
                        # on the ground, no tube, portals or holes (Central Park's 86th Street Transverse under its
                        # park bridges got portal frames standing in the open, the user's report 2026-10-17). Between it
                        # and TUNNEL_COVER: a low arch (a park's stone bridge over a sunken road, Central Park's West
                        # Drive over the 86th Street Transverse: 3.4 m clearance under ~1.5 m of arch and soil):
                        # straight between its portals, its tube as high as the ground over it allows (TUNNEL_SOIL_M
                        # under it, at least TUNNEL_LOW_H), of stone (roadmesh.tunnel)
PORTAL_LOOK_M = 12.0    # m inside a portal where the ground over it is read for its face's height ...
PORTAL_GROUND_M = 0.5   # ... which reaches this far over that ground ...
PORTAL_UP_MIN = 1.0     # ... and at least this far over the vault
SECTION_MIN_M = 8.0     # m: a tunnel's covered sections and the open gaps between them at least this long
FULL_MIN_M = 60.0       # a covered section reaching TUNNEL_COVER is a full tunnel (a rock tube) if it is this long or
FULL_DEEP_M = 16.0      # has this much ground over it somewhere; shorter and shallower, an underpass under an embankment:
                        # a low arch, the ground built up over it (a 31 m underpass under the E6 in Norway with 8.6 m of
                        # embankment over it stood as a bare 6.8 m concrete tube on the ground beside the road, GitHub #2)
TUNNEL_SOIL_M = 1.0     # m of arch and soil kept over a low tunnel's vault
ARCH_SIZE_SKIP_M = 5.0  # m at a low arch's mouths left out when its height is sized from the cover over it
ROAD_CAP_JOIN_M = 20.0  # a road with an end this near a low arch's mouth joins it (its own road going on), not one over it
ROAD_CAP_GAP_M = 0.5    # m a low arch's stone top stops short of a carriageway over its mouth ...
ROAD_CAP_OVER_M = 1.0   # ... and its portal face reaches at most this far over that road
TUNNEL_LOW_H = 3.6      # m: the lowest tube (floor to vault)
MAJOR_LOW_H = 5.0       # m: ... on a major road (CONCRETE_WALL_CLASSES: road tunnels clear ~4.5 m, a truck ~4 m)
# A covered section (a low arch or a tube) goes on along the tunnel until every ground road crossing over it within
# TUNNEL_ROAD_REACH_M of its ends is covered, its carriageway and TUNNEL_ROAD_CLEAR_M past it (along the tunnel: / the
# sine of the crossing angle, at least TUNNEL_ROAD_MIN_SIN). On lidar each carriageway of a divided motorway over an
# underpass is its own embankment: the arch under one was 8 m long under 13 m of lanes, its mouths and terrain holes
# in them (GitHub #450, the A-7 near Málaga).
TUNNEL_ROAD_CLEAR_M = 2.0
TUNNEL_ROAD_MIN_SIN = 0.35
TUNNEL_ROAD_REACH_M = 15.0


PORTAL_SHALLOW_COVER_M = 1.5    # m: a major road's open end run with at least this much ground over its floor is covered


NO_WALK_KMH = 80.0              # a one-way carriageway this fast has no sidewalks, whatever stands near it (Swedish 2+1
                                # roads past houses read as town, and got a raised sidewalk for a median: 2026-11-02)
ARTERIAL_KMH = 73.0             # a divided urban trunk this slow (45 mph) is a town arterial, not an expressway ...
ARTERIAL_GAP_M = 150.0          # ... or, with no limit, one with a junction this often (Coastal Highway ~50 m)


MEDIAN_SLAB_M = 6.0     # m: a divided road's median up to this wide is one raised island, curb to curb (each
                        # carriageway's half of it); wider, ...
MEDIAN_CURB_W = 0.6     # ... a curbed band this wide along each carriageway, ...
MEDIAN_MIN_M = 0.8      # ... where its ground between them is this wide or more (Boston Street's is ~1.5 m of concrete
                        # curb, Mapillary 2026-10-17), lawn on one wider than MEDIAN_SLAB_M (surface.paint_roads), ...
MEDIAN_MAX_M = 16.0     # ... and no wider (two one-way streets a block apart are not one road with a median)


JOIN_GAP_M = 14.0       # sidewalk ends this close (and facing each other) are joined: a sharp corner / road merge
JOIN_TURN_DEG = 70.0    # ... when the run turns by at most this to reach the other end


def densify(pts, spacing=MAX_SPACING):
    """Nodes added so no two are more than `spacing` m apart (xy). Pure."""
    import fast
    pts = np.asarray(pts, np.float64)
    if not len(pts):
        return pts
    fast.require(0, "densify")
    return fast._ef.densify(pts, float(spacing))     # the Rust loop


WALK_DRAPE_M = 2.0      # m between the ground samples under a sidewalk (roadmesh.DRAPE_STEP: the road mesh's edges)
WALK_GROUND_M = 0.5     # ... on the drawn ground (ground): its squares' kinks then cost the run under 0.5 cm over
                        # WALK_DRAPE_TOL_M (a 3 m dip on a hill, 1/32 m quantum: 1.9-2.0 cm; every 1 m: 2.7-2.8)
WALK_DRAPE_TOL_M = 0.02     # m a sidewalk point dropped again may lie off the line through its kept neighbours


WALK_ON_ROAD_M = 0.3    # m inside a carriageway's edge a sidewalk's slab may reach before it is cut (walks_off_roads)
WALK_TRIM_STEP_M = 1.0  # m between the points a run's slab is tested at
WALK_TRIM_KEEP_M = 3.0  # m: a shorter piece left between two roads is dropped


def split(pts, n=MAX_NODES):
    """Chunks of at most n nodes sharing their end nodes. Pure."""
    if len(pts) <= n:
        return [pts]
    out, i = [], 0
    while i < len(pts) - 1:
        out.append(pts[i:i + n])
        i += n - 1
    return out


TILE_BUILDING_LAYERS = ("buildings", "building_parts", "water")   # a road tile's own cells, with its buildings


COVER_MARGIN_M = 30.0   # m past the tile: the ground carriageways a deck's parapet is tested against (build_tile cover_out)


DECK = {"m": "road_asphalt_2lane", "tl": 11.0, "p": 20}      # bridge decks (roadmesh.deck)
DECK_STEP = 4.0         # m between deck profile points
DECK_CLEAR = 0.05       # m a deck stays above the ground under it (no z-fighting where it rests on it)
# Usable space below the lowest structural girder, not merely to the road surface.
# Six metres accommodates tall road vehicles such as semi-trucks; unlimited-height
# modded vehicles and raised equipment cannot have a finite universal guarantee.
MIN_BRIDGE_HEADROOM_M = 6.0
BRIDGE_HEADROOM_MARGIN_M = 0.15   # mesh simplification/float and vehicle clearance tolerance
DECK_ROAD_CLEAR = MIN_BRIDGE_HEADROOM_M + BRIDGE_HEADROOM_MARGIN_M + roadmesh.DECK_DEPTH + roadmesh.GIRDER_D
DECK_FULL_DEPTH = roadmesh.DECK_DEPTH + roadmesh.GIRDER_D    # m of slab and girders under a long span's road
DECK_MIN_DEPTH = 0.9    # m: the thinnest deck (a short slab bridge)
DECK_SPAN_RATIO = 22.0  # a deck is its main span / this deep (girder bridges run span/20-25), DECK_MIN..FULL_DEPTH


DECK_CLEAR_REACH_M = 20.0   # full headroom either side of a road crossing (matches approach fills' flat reach)
DECK_GRADE = 0.05       # that requirement falls 5 % along the deck from the crossing
DECK_MAX_GRADE = 0.06   # a deck climbs no steeper than this from an abutment
DECK_STEEP_GRADE = 0.15  # ... nor, whatever ground the caps were raised to, than this over lower ground (no walls)
CORNER_CONCRETE_M = 4.0  # m: a concrete street this close to a curb return fills its corner in concrete too
AT_GRADE_M = 3.0        # m: a road crossing a deck within AT_GRADE_REACH_M of an abutment that cannot get this
AT_GRADE_REACH_M = 12.0     # high over it is a junction at grade (the intersection a ramp starts from)
CHAIN_CELLS = 12        # map cells read at most to close one bridge chain


PIER_SPAN_M = 32.0      # m between a bridge's piers (a span), longer under a high deck: + PIER_SPAN_PER_M per m of height
PIER_SPAN_PER_M = 0.6
PIER_SPAN_MAX = 60.0
PIER_MIN_CLEAR = 1.5    # m from the ground up to a pier's top (under the girders): lower, the deck rests on the ground
PIER_ROAD_M = 7.0       # m a pier keeps from the centre line of a road under the deck (its half width + room)
PIER_CHANNEL_CLEAR = 15.0   # no pier in water whose run wants this much clearance (water_clear): a ship channel
PIER_IN_WATER_M = 8.0   # m a pier in water reaches under its surface


OBS_BLEND_DG = 0.04     # a measured deck eased onto its joints' heights changes its grade by at most this (BridgeProfiles)
OBS_BLEND_MIN_M = 10.0  # m, the shortest easing
OBS_MAX_GRADE = 0.14    # a measured deck steeper than this after the easing (2 m steps) keeps the inference
OBS_JOINT_SHORT_M = 30.0   # m: a part shorter than this, eased onto its joints too steeply, climbs evenly instead
OBS_JOINT_GRADE = 0.08  # a joint's measured height stays within this grade of the far end of a part nothing measured there
OBS_END_MAX_M = float(os.environ.get("BNE_OBS_END_MAX", "inf"))  # m a measured deck's end may lie off the graded road at
                        # its abutment, else the inference (inf: always eased onto it; 4 m made Springfield worse)


DECK_Z_HOOK = None      # grading sets it (BNE_INTERCHANGE + carriageways): hook(frame, lines) -> [z (k,) or None] per
                        # deck line, in the frame's up metres


TUNNEL = {"m": "road_asphalt_2lane", "tl": 11.0, "p": 20}     # the road surface in a tunnel (a deck: collides)
LOW_HOLE_M = 3.0            # m from a low arch's ends: its mouths, flat at its floor (arch_raise) and cut away ...
LOW_HOLE_STEP = 1.5         # ... the cut a terrain grid step (1 m) further (the squares from the floor up to the soil)
TUNNEL_HOLE_MARGIN = 0.5    # m around the tube a portal's terrain hole reaches (above, below and sideways)


ARCH_SIDE_M = 1.0           # m beside a low arch's tube its soil is raised too
ARCH_SOIL_MIN = 0.6         # m of ground kept over a low arch's vault (arch_raise)


HOLE_ROAD_MARGIN = 0.3      # m past another road's carriageway no tunnel hole is cut (spare_carriageways)
HOLE_JOIN_M = 3.0           # m: a road with an end this near a tunnel part's end is its approach (its mouth cut)


WALL_STEP = 2.0         # m between a retaining wall's points
BACK_TOP_OVER_M = 3.0   # m a wall's top may rise over its capped height (grading.wall_max_h) at its back edge
WALL_PROBE_M = 1.0      # m beyond the cut's foot (grading.WALL_T_M) where the natural ground's height is read
WALL_TOP_M = 0.15       # m the wall stands over the ground behind it
WALL_FOOT_M = 0.3       # m it reaches under the road
WALL_FULL = 0.9         # the graded ground this share of the way up to the natural ground behind a wall: its top ends
                        # there ...
WALL_THICK_MIN_M = 1.0  # ... at least this deep and at most WALL_THICK_MAX_M
WALL_THICK_MAX_M = 3.5
WALL_OTHER_GAP_M = 0.3  # m a wall's top keeps from another road's carriageway
WALL_TAPER_M = 8.0      # m at each end of a wall over which it comes down ...
WALL_TAPER_LOW_M = 0.6  # ... to this over the road
WALL_SLAB_DZ = 0.8      # m the ground behind a wall may fall under its top before the top ends there ...
WALL_COPING_MIN_M = 0.5  # ... at least its coping
WALL_DIAG_M = 1.45      # the terrain lattice's diagonal (heightmap.SQUARE 1 m)
# a major road's cutting is walled in concrete (Munich's McGraw-Graben, Mapillary 2026-10-21), a smaller road's in stone
# (Central Park's transverses)
CONCRETE_WALL_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}
WALL_EVEN_N = 2         # a wall's top as deep as its deepest within this many WALL_STEPs either side
WALL_STEP_UP_M = 0.5     # m the graded ground climbs over the road where a wall's face goes (short of the core's edge)


WALL_KEEP_M = 1.2       # m: a wall goes on while the cut stays this deep (it starts at grading.WALL_CUT_M) ...
WALL_GAP_N = 3          # ... over gaps of up to this many WALL_STEPs, and none shorter than WALL_MIN_N (the cut's depth
WALL_MIN_N = 3          # wavered round 2 m: stubs of wall with gaps, the 97th Street Transverse, 2026-10-17)
# A wall stands only where the graded ground behind its face climbs at least this steeply (m per m, its median over
# the wall; across WALL_STEEP_SPAN from 1 m in front of the face): the step the grading cut. A natural bank the
# terrain already draws as a slope is no cutting: a 1.5 m concrete wall stood in front of a low eroded bank beside
# US-163 in Utah (its ground rose 0.4-1.2 m per m; Munich's McGraw-Graben's cut walls 5-9, measured 2026-10-29)
WALL_STEEP = 2.0
WALL_STEEP_SPAN = (-1.0, 6.0)


WALL_DECK_MARGIN_M = 3.0    # m beyond a bridge deck's edge where a road under it gets no retaining wall ...
WALL_DECK_OVER_M = 2.0      # ... a deck this far over the road (an overpass, not the road's own approach)


WALL_TOP_EDGE_M = 0.3   # a road surface goes this far onto a wall's top at most (_off_wall_tops)


# Rock cliffs (the user, 2026-10-21, after the official maps: Italy's and West Coast USA's rural cuts are rock-face meshes
# sunk into the slope, overlapping): a rural road's retaining wall is drawn as Italy's rock-face pieces (trees.lua group
# "cliff") along its face instead of stone. (shape height m, its bottom under the pivot as a share of it, its depth in
# front of the pivot m per unit scale, its width m, its depth m)
CLIFF_SHAPES = ((2.9, 0.52, 1.5, 10.5, 2.9), (13.2, 0.17, 1.9, 12.9, 5.2))
CLIFF_TALL_M = 6.2      # a piece this tall or taller: the tall shape (trees.lua M.pick: nearer to it than the short one by log)
CLIFF_SINK_M = 1.0      # m the pieces reach under the road's level ...
CLIFF_OVER_M = 0.3      # ... and over the ground behind
CLIFF_SPACING = 0.55    # pieces along the face a share of their width apart (overlapping)
CLIFF_REACH_M = 8.0     # m behind the face a piece may reach at most (less where another road is)
CLIFF_EDGE_M = 0.5      # m off the carriageway's edge the pieces' fronts stand
CLIFF_ROAD_GAP_M = 0.3  # m a piece's footprint keeps from any carriageway


# Parapets (the Stelvio Pass, Mapillary 2026-10-21: a low dry-stone wall along the outside of every bend and drop): a
# rural road (not a motorway / trunk: guardrails there) whose ground falls PARAPET_DROP_M or more within PARAPET_PROBE_M of
# its edge gets a stone parapet just off its carriageway, not across a road joining it.
PARAPET_DROP_M = 2.0
# ... a mountain pass road's from a smaller drop: the Stelvio's parapet runs the whole valley side (_mountain_roads)
PARAPET_DROP_MOUNTAIN_M = 1.0
PARAPET_PROBE_MOUNTAIN_M = 5.0  # ... within this (the elevation data smooths its wall-held drop into a slope)
PARAPET_PROBE_M = 3.0
PARAPET_H = 0.6
PARAPET_W = 0.45
PARAPET_OFF_M = 0.2
PARAPET_ROAD_GAP_M = 2.0
NO_PARAPET_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "service", "track"}
CLIFF_MIN_CUT_M = 6.0   # a rural cut this deep: rock faces (cliff_rows), when CLIFF_WALLS_ON; else a dry-stone wall
CLIFF_WALLS_ON = os.environ.get("BNE_CLIFF_WALLS", "off") == "on"


TWIN_SHARE = 0.8        # a tunnel part beside another for this share of its length (their tubes overlapping) ...
TWIN_DZ_M = 2.0         # ... with its floor within this of the other's: twin tubes, one tube round both (GitHub #84:
                        # the Route des Tamarins' two carriageways 9.5 m apart on Réunion, each an 11 m road in its own
                        # tube, the tubes through each other's walls and portals)
TWIN_STEP_M = 10.0


DECK_CONTEXT_M = 24.0   # m around a tile its bridges' decks are passed as context (roadmesh.deck: whether an edge has
                        # another deck beside it; a parapet stood across the lanes at a tile seam, the deck beside it
                        # in the next tile, Queensboro Bridge 2026-10-17)


DECK_JOIN_M = 1.0       # m: deck parts' ends this close go on one from the other (no parapet end there)
ABUT_GROUND_M = 1.0     # a deck part's end within this of the ground under it is an abutment (roadmesh.deck: its
                        # parapets ramp down there to the approach)


# Junction pads on bridges: where deck parts meet (with each other, or with a ground road going on from the joint), the
# convex hull of their carriageways within PAD_REACH_M of the joint is paved at their heights (roadmesh.deck draws it,
# colliding): the corner of a turn, the wedge where a ramp leaves, a wider deck's edge past a narrower street's. The map
# leaves those open, over whatever lies under the bridge: at North Charles Street over I-83 in Baltimore the AI cut the
# corner onto the station's road and fell 6 m into the motorway's cut, 2 of 2 runs (2026-10-29). None where a deck simply
# goes on straight from another part (BNE_DECK_PADS=off: none at all).
PADS_ON = os.environ.get("BNE_DECK_PADS", "on") != "off"
PAD_REACH_M = 12.0
PAD_STEP_M = 2.0
PAD_JOIN_M = 0.75           # m: ends this near are one joint (_skew_decks' tolerance)
PAD_JOIN_Z_M = 0.75         # m: the full endpoint height spread of a physical joint, never a stacked crossing
PAD_GROUND_M = 1.0          # m: a ground road's end this near a joint goes on from it
PAD_STRAIGHT_COS = -0.97    # two parts leaving a joint this straight need none
# The pad reaches PAD_MARGIN_M past the carriageways (a turn cut across the corner stays that far from its rim); the
# decks' parapets stand nowhere inside it, and its rim has a railing where the ground under it lies PAD_WALL_DROP_M or
# more lower, except where a carriageway leaves it (within its half width + PAD_OPEN_M): the station road's parapet
# began at the joint, in the turn's path, and the AI clipped its end (Charles Street over I-83, 2026-10-29).
PAD_MARGIN_M = 2.5
PAD_OPEN_M = 0.75
PAD_WALL_DROP_M = 1.0
PAD_RIM_STEP_M = 1.0


PAD_FORK_MARGIN_M = 0.45    # m: a fork's pad past its carriageways (roadmesh.PARAPET_W: the decks' own slab edges)
PAD_FORK_OPEN_M = 0.1       # ... its rim open only where a carriageway crosses it (PAD_OPEN_M is past its sides)
PAD_FORK_COS = 0.85         # two parts leaving a joint this close in direction run on side by side (a fork) ...
PAD_FORK_STRAIGHT_COS = -0.9    # ... or this straight on from each other


# Sidewalks on bridges (the user, 2026-10-31, East Pratt Street over the Jones Falls: "check if this bridge has
# sidewalks"; its real deck has one each side, ours stopped the street's at both ends): a deck whose road has sidewalks
# on the ground (its own ground part, else a road going on from its end: deck_walks) carries them, the deck slab,
# parapets and collision that much wider, a concrete strip flush on the deck (a raised one would need its own collider
# across the car's path) beside the carriageway.
DECK_WALK = {"m": "sidewalk_concrete", "tl": 6.0, "p": 16}
DECK_WALK_CHAIN = 6              # roads all bridge a deck's sidewalks are looked for along (deck_walks)
DECK_WALK_LIFT_M = 0.03         # m the strip is drawn over the deck (no z fighting with its asphalt)
# A motorway's or trunk road's deck: concrete all across (the I-95 viaduct by the Middle Branch in the aerial: the whole
# deck pale concrete; the user, 2026-10-31: "the entire real bridge is concrete ... concrete shoulders [alone] don't
# look good or realistic")
DECK_CONCRETE_CLASSES = {"motorway", "trunk"}
# the stock highway concrete (West Coast USA's road_concrete: tined, jointed; its square tile two lanes wide), the
# texture once per DECK_CONCRETE_TILE_M across ("tu") and as long along; the plain slab concrete looked like paving slabs
DECK_CONCRETE_TILE_M = 7.3
DECK_CONCRETE_ON = False        # (off: the user, 2026-10-31, "the concrete texture does not look very good ... imperfections
                                # show up a lot more; switch back to asphalt for now")
DECK_CONCRETE = {"m": "bne_highway_concrete", "tl": DECK_CONCRETE_TILE_M, "p": 20}


GOING_ON_DEG = 20.0     # a road whose end meets another's end turning by under this (the straightest there) goes on in it


LINE_RANK = {"motorway": 7, "trunk": 6, "primary": 5, "secondary": 4, "tertiary": 3, "unclassified": 2, "residential": 2}


TILE_RETRY_S = 3.0              # s before a road tile whose build failed is built again (Tracker._built)
TILE_STUCK_S = 90.0             # s a road tile's build may take before the log says it may be stuck
TILE_FAILS_WARN = 5             # a tile's builds failed in a row before an expected kind of failure is logged as a warning


BLD_PARTS_MAX = 4               # building meshes per road tile ("bld", "bld2".."bld4": the mod's list)


# --- road geotiles (phase 1c, .claude/work/geotiles_plan.md) -------------------------------------------------------
# A road tile is a geotile (geotiles.Z: the global grid, key "g17_x_y"): its content (buildings, decks, tunnels,
# walls, wires, rails, props and guard rails, parking, stations, wells, sound) built once in its own frame and kept by
# the mod across rebases (placed into each anchor's world: geotile.lua), its skin (the draped road mesh, plazas,
# sidewalks) built per anchor on that anchor's lattice. After a rebase the server sends skins only (and the exact
# variants of the tall parts the rigid placement leaves off: GeoTracker.exact_need).

ROAD_GEO = 1                # the road geotiles' protocol (hello.roadGeo, the mod's roads.lua M.ROAD_GEO)
EXACT_TOL_COL_M = 0.03      # m a colliding part placed rigidly may be off (relief x the shear) before its exact variant
EXACT_TOL_M = 0.10          # ... a render-only one


# Sound spaces (the mod's ambience.lua makes them SFXSpaces: the stock maps' tunnel reverb in a tube, the one under a
# bridge below its deck): kinds, and how far a bridge's space reaches under the deck's surface
SOUND_TUNNEL, SOUND_BRIDGE, SOUND_TUNNEL_OPEN, SOUND_GARAGE, SOUND_PORTAL = 0, 1, 2, 3, 4
SOUND_REFINERY = (5, 6, 7)      # points: West Coast USA's refinery tower engines, large, small (industry.py)
SOUND_POWER_PLANT = 8           # points: West Coast USA's power plant sound (industry.py), played quieter by the mod
SOUND_UNDER_M = 25.0
# a closed tunnel's mouths: the stock portal snapshot event this far inside each (recorded 2026-10-26: the outside cut
# off within ~10 m of it, 7 dB down at 20 m, untouched from 40 m: a ~25 m fade outside the portal, both ways), its
# position SOUND_PORTAL_UP over the floor (a row [kind, x, y, z, 0, 0, 0, yaw (into the tube)])
SOUND_PORTAL_IN_M = 10.0
SOUND_PORTAL_UP = 2.0
# a low arch, or a tube shorter than this, has the open tunnel's (the outside heard, muffled ~5 dB) instead of the
# closed one's (the outside cut off): recorded 2026-10-26
SOUND_SHORT_TUNNEL_M = 120.0
SOUND_BOX_M, SOUND_TURN, SOUND_DZ_M, SOUND_EXTEND_M = 60.0, math.radians(10.0), 1.0, 1.0


COLLIDER_PIECE_M = 20.0     # collider_boxes: a run / deck cut into boxes this long
COLLIDER_MARGIN_M = 0.5
