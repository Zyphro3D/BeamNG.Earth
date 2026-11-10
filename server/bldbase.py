"""Procedural buildings, base layer (split out of buildings.py): the shared constants and materials (LOOKS and their
tints), colour parsing and the footprint geometry helpers (hash, area, rings, oriented boxes).
"""
import os
import hashlib
import math
import numpy as np
import fast
import francehouses


FLOOR_M = 3.0               # m per storey (residential)
GROUND_FLOOR_SHOP_M = 4.2   # m, a shop's ground floor
FOUNDATION_M = 0.6          # m the walls reach under the lowest ground of the footprint
# The walls reach this far under the lowest ground of the footprint (a skirt): beyond the physics block (+-512 m) the
# ground drawn is the far field's mesh, coarser with distance (tens of metres between vertices at 1-2 km), which on any
# slope runs metres under the heights the buildings stand on: they floated (the user, 2026-10-18: "houses and
# buildings in many parts of the world are floating"). Under a correct ground the skirt is hidden; the floors stay
# where they were (anchored at the eaves, the rest a plain plinth). The lite mesh (far tiles, a near tile's distant
# detail) stands on the coarsest ground: the deepest skirt.
SKIRT_M = 3.0
SKIRT_LITE_M = 10.0
MIN_AREA_M2 = 12.0          # footprints smaller than this are skipped (kiosks, sheds' slivers)
MAX_VERTS = 64              # footprints with more vertices are simplified until under this
SIMPLIFY_M = 0.35           # m: vertices closer than this to the line of their neighbours are dropped
TILE_PX_M = 12.0            # m of wall per 1024 px of the window sheets (5 sash bays of 2.4 m)
ROOF_UV_M = 6.0             # m per repeat of the roof textures
SIDING_UV_M = 3.0           # m per repeat of the plain wall textures (sheds, garages)
PITCH = {"gabled": 32.0, "hipped": 26.0, "half-hipped": 32.0, "pyramidal": 26.0}     # degrees
# roof:shape forms with pitches of their own, whatever the region's (roof_rise, _roof)
SKILLION_PITCH = 12.0           # a lean-to: one shallow slope from wall to wall
MANSARD_PITCHES = (68.0, 18.0)  # the steep lower slope, the shallow top
MANSARD_INSET_M = 1.2           # how far the lower slope comes in, at most
GAMBREL_PITCHES = (55.0, 25.0)  # a barn's: the same, on two sides only
GAMBREL_BREAK = 0.55            # the slope breaks this far out from the ridge (of the half width)
HALF_HIP = 0.35                 # the share of a half-hipped roof's height the small hips at its ends take
EAVE_M = 0.35               # m a pitched roof overhangs its walls
CORNICE_OUT_M = 0.28        # a flat roof's cornice: how far it stands out, its depth below and above the eave
CORNICE_DOWN_M = 0.45
CORNICE_UP_M = 0.2

# Materials (mod/.../art/shapes/beamngEarth/buildings.materials.json): the window sheets in a few tints, the
# shop / office sheet, roofs, plain walls.
FACADE_TINTS = ("bne_bld_fac_white", "bne_bld_fac_cream", "bne_bld_fac_sand", "bne_bld_fac_grey")
PAINT_TINTS = ("bne_bld_fac_blue", "bne_bld_fac_green", "bne_bld_fac_rose", "bne_bld_fac_yellow")   # painted houses
BRICK_TINTS = ("bne_bld_fac_white", "bne_bld_fac_brickdark", "bne_bld_fac_brickwarm")   # the brick band's
SHOP = "bne_bld_shop"
OFFICE = "bne_bld_office"
ROOF_FLAT = "bne_bld_roof_flat"
ROOF_SLATE = "bne_bld_roof_slate"
ROOF_SHINGLE = "bne_bld_roof_shingle"
SIDING = "bne_bld_siding"
FACADE = "bne_bld_facade"    # the plaster window sheet, tinted per building by vertex colour
STUCCO = "bne_bld_stucco"   # plain stucco walls (windows laid over them from the sheet)
ROOF_METAL = "bne_bld_roof_metal"   # standing-seam metal (barns, sheds, farm buildings)
PANEL = "bne_bld_panel"     # precast concrete panels (big-box stores, strip malls)
METALWALL = "bne_bld_metalwall"     # corrugated metal walls (warehouses, factories)
STONE = "bne_bld_stone"     # dressed stone (old villages and towns of southern and central Europe: Gordes came out as
                            # grey plaster, 2026-10-15)
STONE_FLOORS = 4            # stone walls on buildings this low at most
BIG_BOX_M2 = 1200.0         # m2: a low commercial building this big is a big box (blank walls, one storefront)
CHURCH_CLASSES = ("church", "chapel", "cathedral")
BARN_CLASSES = {"barn", "farm_auxiliary", "stable", "cowshed"}
# The mesh's materials (vertColor: each building's colours are its vertices'); the names above are the
# looks plan() picks, each one of these with a colour (LOOKS).
MARBLE = "bne_bld_marble"   # cut white marble (the stock m_marble_cut): Baltimore's row house steps
BRICK_WALL = "bne_bld_brick"    # plain red brick (the stock brick_plain): a garage's skin
CONCRETE = "bne_bridge_concrete"  # plain concrete (tunnel.materials.json): a precast garage's skin
SKY_WALL = "bne_bld_sky"    # the skyline's walls: plain (vertex colours), their windows lit at night (_sky_mesh)
LANDMARK_IRON = "bne_lm_iron"   # a landmark's painted iron (landmarks.py; vertex colours), lit at night (nightlights.lua)
LANDMARK_STONE = "bne_lm_stone"     # ... its stone (vertex colours), dark at night (lit by its lamps: the Colosseum)
LANDMARK_FLOODLIT = "bne_lm_floodlit"   # ... its floodlit stone, glowing at night (the Arc de Triomphe, Big Ben)
LANDMARK_LAMP = "bne_lm_lamp"   # ... its lamps (the Eiffel Tower's, the Colosseum's arches): points of light at night
LANDMARK_DIAL = "bne_lm_dial"   # ... a clock's dials (Big Ben's), lit from behind at night
MATERIALS = (FACADE, SHOP, OFFICE, ROOF_FLAT, ROOF_SLATE, ROOF_SHINGLE, SIDING, STUCCO, ROOF_METAL, PANEL, METALWALL,
             STONE, MARBLE, BRICK_WALL, CONCRETE, SKY_WALL, LANDMARK_IRON, LANDMARK_STONE, LANDMARK_LAMP,
             LANDMARK_FLOODLIT, LANDMARK_DIAL) + francehouses.MATERIALS
LOOKS = {
    "bne_bridge_steel": (METALWALL, (0.50, 0.52, 0.54)),
    "bne_marble": (MARBLE, (1.0, 1.0, 1.0)),
    "bne_brick_red": (BRICK_WALL, (1.0, 1.0, 1.0)), "bne_brick_dark": (BRICK_WALL, (0.78, 0.72, 0.70)),
    "bne_brick_tan": (BRICK_WALL, (1.0, 0.92, 0.80)),
    "bne_precast": (CONCRETE, (1.0, 1.0, 1.0)), "bne_precast_warm": (CONCRETE, (1.0, 0.95, 0.86)),
    "bne_bld_fac_white": (FACADE, (1.0, 1.0, 1.0)), "bne_bld_fac_cream": (FACADE, (1.0, 0.93, 0.80)),
    "bne_bld_fac_sand": (FACADE, (0.92, 0.82, 0.68)), "bne_bld_fac_grey": (FACADE, (0.80, 0.82, 0.84)),
    "bne_bld_fac_brickdark": (FACADE, (0.78, 0.72, 0.70)), "bne_bld_fac_brickwarm": (FACADE, (1.0, 0.88, 0.80)),
    "bne_bld_fac_blue": (FACADE, (0.80, 0.87, 0.96)), "bne_bld_fac_green": (FACADE, (0.84, 0.92, 0.82)),
    "bne_bld_fac_rose": (FACADE, (0.99, 0.86, 0.82)), "bne_bld_fac_yellow": (FACADE, (1.0, 0.95, 0.72)),
    SHOP: (SHOP, (1.0, 1.0, 1.0)), OFFICE: (OFFICE, (1.0, 1.0, 1.0)), SIDING: (SIDING, (0.93, 0.92, 0.88)),
}
ROOF_TINTS = {               # per roof material: the colours a building's roof takes (hashed)
    ROOF_FLAT: ((0.85, 0.85, 0.85), (0.62, 0.62, 0.64), (0.95, 0.93, 0.88), (0.45, 0.45, 0.46), (0.75, 0.70, 0.64)),
    # 4..: clay tiles (terracotta, red-brown, faded orange) and dark grey / anthracite / brown-black (regional)
    ROOF_SLATE: ((1.0, 1.0, 1.0), (0.80, 0.66, 0.56), (0.76, 0.80, 0.90), (0.62, 0.62, 0.62),
                 (0.95, 0.52, 0.36), (0.78, 0.40, 0.30), (1.0, 0.66, 0.46), (0.42, 0.43, 0.46), (0.30, 0.31, 0.33),
                 (0.48, 0.38, 0.32)),
    ROOF_SHINGLE: ((0.75, 0.72, 0.70), (0.55, 0.50, 0.48), (0.70, 0.56, 0.48), (0.62, 0.66, 0.72),
                   (0.95, 0.52, 0.36), (0.78, 0.40, 0.30), (0.42, 0.43, 0.46), (0.36, 0.42, 0.52)),
    ROOF_METAL: ((0.85, 0.86, 0.88), (0.55, 0.57, 0.60), (0.45, 0.58, 0.48), (0.62, 0.30, 0.26), (0.40, 0.42, 0.48)),
}
WALL_TINTS = {               # houses with plain walls and laid-over windows: their colours (hashed)
    SIDING: ((0.96, 0.95, 0.92), (0.90, 0.86, 0.76), (0.78, 0.84, 0.90), (0.95, 0.92, 0.74), (0.78, 0.84, 0.74),
             (0.82, 0.78, 0.72), (0.70, 0.72, 0.75), (0.62, 0.20, 0.17), (0.66, 0.24, 0.20)),   # the last: barn red
    STUCCO: ((1.0, 0.97, 0.90), (0.96, 0.88, 0.74), (0.93, 0.80, 0.68), (0.98, 0.95, 0.88), (0.88, 0.84, 0.78),
             # 5..: bright paints (Latin America, the Mediterranean, the tropics): ochre, terracotta, sky blue,
             # mint, pink, yellow
             (1.0, 0.78, 0.45), (0.92, 0.58, 0.44), (0.66, 0.82, 0.95), (0.70, 0.90, 0.78), (0.98, 0.72, 0.76),
             (1.0, 0.92, 0.52)),
    PANEL: ((0.96, 0.94, 0.89), (0.86, 0.86, 0.86), (0.92, 0.85, 0.74), (0.78, 0.80, 0.82), (0.88, 0.78, 0.68),
            (0.62, 0.62, 0.60), (0.56, 0.54, 0.50)),    # the last two: weathered concrete (parking decks)
    METALWALL: ((0.88, 0.89, 0.90), (0.72, 0.74, 0.77), (0.84, 0.80, 0.72), (0.66, 0.72, 0.78)),
    # limestone, honey, ochre and grey stone (the texture is a light ashlar: tinted down)
    STONE: ((1.0, 0.93, 0.80), (0.98, 0.86, 0.66), (0.95, 0.80, 0.60), (0.86, 0.85, 0.82), (0.92, 0.88, 0.80)),
    # curtain walls (the user, 2026-10-14: "the skyscrapers all use the same texture and look the same colour"):
    # silver, blue, green, bronze, charcoal and steel-blue glass, the common tints of real towers
    OFFICE: ((1.0, 1.0, 1.0), (0.74, 0.84, 0.98), (0.76, 0.92, 0.86), (0.98, 0.84, 0.66), (0.56, 0.58, 0.62),
             (0.66, 0.74, 0.86), (0.88, 0.90, 0.92)),
}
for _m, _cs in list(ROOF_TINTS.items()) + list(WALL_TINTS.items()):
    for _k, _c in enumerate(_cs):
        LOOKS[f"{_m}|{_k}"] = (_m, _c)
# stained wood boards (Alpine chalets; kept apart from the siding tints' indices): larch brown, dark brown, grey-brown
WOOD_TINTS = ((0.62, 0.44, 0.30), (0.46, 0.33, 0.24), (0.58, 0.52, 0.46))
for _k, _c in enumerate(WOOD_TINTS):
    LOOKS[f"{SIDING}|w{_k}"] = (SIDING, _c)
# Shop signs (Bishop's and Moab's main streets: a painted sign band over every shopfront, each shop its own
# colour): boxes over the ground floor of a shop's street wall, a colour per shop width (hashed).
SIGN_TINTS = ((0.62, 0.16, 0.14), (0.16, 0.22, 0.42), (0.18, 0.36, 0.24), (0.95, 0.90, 0.75), (0.42, 0.30, 0.22),
              (0.85, 0.65, 0.20), (0.13, 0.13, 0.14), (0.96, 0.96, 0.96), (0.15, 0.45, 0.48))
for _k, _c in enumerate(SIGN_TINTS):
    LOOKS[f"sign|{_k}"] = (FACADE, _c)
SIGN_H = 0.9                # m, the band's height (its top 0.15 m under the ground floor's top)
SIGN_OUT = 0.18             # m out from the wall
SIGN_SHOP_M = (6.0, 11.0)   # one shop's width (hashed in this range)
BARN_REDS = 2               # the last siding tints are barn red (barns only)
US_WALL_TINTS = {STUCCO: 5}  # the first tints of these kinds are the US's (later ones: regional paints)
JITTER = 0.06               # a building's colour varies this much (one factor, hashed)

COLOR_NAMES = {
    "white": (0.95, 0.95, 0.95), "black": (0.12, 0.12, 0.12),
    "grey": (0.50, 0.50, 0.50), "gray": (0.50, 0.50, 0.50),
    "lightgrey": (0.75, 0.75, 0.75), "lightgray": (0.75, 0.75, 0.75),
    "darkgrey": (0.30, 0.30, 0.30), "darkgray": (0.30, 0.30, 0.30),
    "silver": (0.75, 0.75, 0.78), "charcoal": (0.20, 0.20, 0.22), "anthracite": (0.22, 0.23, 0.25),
    "red": (0.75, 0.15, 0.15), "darkred": (0.50, 0.10, 0.10), "lightred": (0.85, 0.35, 0.35),
    "maroon": (0.45, 0.10, 0.15), "burgundy": (0.45, 0.12, 0.20), "crimson": (0.70, 0.10, 0.20),
    "pink": (0.90, 0.60, 0.65), "rose": (0.90, 0.55, 0.60), "coral": (0.90, 0.45, 0.35),
    "salmon": (0.90, 0.50, 0.45), "brick": (0.65, 0.25, 0.18),
    "green": (0.18, 0.55, 0.22), "darkgreen": (0.10, 0.35, 0.12), "lightgreen": (0.45, 0.75, 0.40),
    "olive": (0.45, 0.50, 0.20), "lime": (0.55, 0.75, 0.20), "forestgreen": (0.13, 0.40, 0.15),
    "teal": (0.12, 0.45, 0.45), "mint": (0.60, 0.85, 0.70), "sage": (0.55, 0.65, 0.55), "moss": (0.35, 0.45, 0.25),
    "blue": (0.18, 0.38, 0.72), "darkblue": (0.10, 0.20, 0.45), "lightblue": (0.50, 0.70, 0.88),
    "navy": (0.08, 0.15, 0.35), "skyblue": (0.50, 0.75, 0.90), "cyan": (0.20, 0.70, 0.75),
    "azure": (0.30, 0.60, 0.85), "cobalt": (0.15, 0.30, 0.65), "turquoise": (0.25, 0.70, 0.65),
    "steel": (0.45, 0.50, 0.55), "steel_blue": (0.28, 0.45, 0.60),
    "yellow": (0.92, 0.82, 0.20), "lightyellow": (0.95, 0.92, 0.60), "gold": (0.85, 0.70, 0.18),
    "orange": (0.90, 0.48, 0.12), "darkorange": (0.80, 0.35, 0.08), "amber": (0.88, 0.60, 0.12),
    "brown": (0.45, 0.30, 0.20), "darkbrown": (0.30, 0.20, 0.12), "lightbrown": (0.60, 0.45, 0.32),
    "rust": (0.60, 0.25, 0.15), "beige": (0.90, 0.85, 0.75), "cream": (0.95, 0.92, 0.82),
    "tan": (0.82, 0.70, 0.55), "terracotta": (0.78, 0.40, 0.26), "ochre": (0.80, 0.55, 0.20),
    "sand": (0.88, 0.80, 0.65), "khaki": (0.75, 0.70, 0.50), "copper": (0.55, 0.75, 0.65),
    "bronze": (0.55, 0.42, 0.25), "purple": (0.50, 0.20, 0.55), "violet": (0.55, 0.25, 0.60),
    "magenta": (0.75, 0.15, 0.55), "indigo": (0.25, 0.10, 0.45), "lavender": (0.75, 0.70, 0.85),
    "plum": (0.45, 0.20, 0.40),
}


def parse_color(val):
    """Parse an OSM/Overture color (hex or CSS name) into an (r, g, b) float tuple in [0, 1], or None."""
    if not val or not isinstance(val, str):
        return None
    s = val.strip().lower().replace(" ", "").replace("-", "_")
    if s in COLOR_NAMES:
        return COLOR_NAMES[s]
    if s.startswith("#"):
        hex_str = s[1:]
        if len(hex_str) in (3, 4):
            try:
                return tuple(int(hex_str[i] * 2, 16) / 255.0 for i in range(3))
            except ValueError:
                return None
        elif len(hex_str) in (6, 8):
            try:
                return tuple(int(hex_str[2 * i:2 * i + 2], 16) / 255.0 for i in range(3))
            except ValueError:
                return None
    return None


def custom_look(mat, rgb):
    """A look key in LOOKS for material mat with RGB tuple in [0, 1]."""
    key = f"{mat}|c_{rgb[0]:.3f}_{rgb[1]:.3f}_{rgb[2]:.3f}"
    if key not in LOOKS:
        LOOKS[key] = (mat, rgb)
    return key


class _Looks:
    """name -> (material index, RGBA bytes) for one building (its hashed brightness jitter)."""

    def __init__(self, seed):
        self.f = 1.0 + JITTER * (2.0 * _hash01(seed, "jit") - 1.0)

    def __getitem__(self, name):
        if name in LOOKS:
            mat, c = LOOKS[name]
        elif name.startswith("bne_fr_house_") and "|c_" in name:
            mat, rgb = name.split("|c_", 1)
            c = tuple(float(v) for v in rgb.split("_"))
        elif name.startswith("custom|") or ("|c_" in name):
            mat = name.split("|")[0]
            c = (1.0, 1.0, 1.0)
        else:
            mat, c = LOOKS.get(name, (FACADE, (1.0, 1.0, 1.0)))
        return MATERIALS.index(mat), tuple(int(max(0, min(255, round(255 * v * self.f)))) for v in c) + (255,)

# The window sheets' bands, top of the image first (v in Collada convention: v = 1 at the image top).
# (sheet, band index, bays per 1024 px)
BAND_TALL = (0, 3)          # m_highrise_plaster: tall windows on plaster (3 bays of uneven width)
BAND_PANEL = (1, 4)         # square windows on panels
BAND_SASH = (2, 5, (0, 2, 3))   # sash windows on plaster (bays 1 and 4 are black "open" windows: skipped)
BAND_BRICK = (3, 5)         # sash windows on red brick
BAND_OFFICE = (0, 5)        # m_highrise_metal: curtain wall
BAND_OFFICE2 = (1, 4)
BAND_SHOP = (2, 2)          # shopfront with awnings
BANDS = 4

# Overture "bridge_structure" (OSM man_made=bridge parts with a height: the Queensboro Bridge's towers, 110 m) was a
# glass tower the decks ran through (the user's report, 2026-10-17)
BRIDGE_CLASSES = {"bridge_structure"}
BRIDGE_PIER_M = 12.0        # a bridge structure's height when not mapped
STEEL_MATERIALS = ("", "steel", "metal", "iron")     # a part over a bridge support of these (or none mapped): steel
MASONRY_MATERIALS = ("concrete", "stone", "brick", "masonry", "granite")
STONE_FACADES = ("stone", "sandstone", "limestone", "granite", "marble", "masonry", "tuff", "slate")
STEEL = "bne_bridge_steel"  # (a LOOKS name: the corrugated-metal material, painted steel grey)
# a bridge structure in its mapped paint (bridgetypes.PAINTS: the Golden Gate Bridge's towers, building:colour #bf4e3b,
# were steel grey beside its orange cables, 2026-10-18)
for _k, _c in __import__("bridgetypes").PAINTS.items():
    LOOKS[f"{STEEL}|{_k}"] = (METALWALL, tuple(v / 255.0 for v in _c))
STEEL_LEG_MAX = 3.0         # m: a frame's legs at most this thick (a quarter of its short side)
STEEL_BAY_M = 14.0          # m between a frame's bracing levels
STEEL_TOP_SHARE = 0.85      # cross beams over the decks only from this share of the height up
SHED_CLASSES = {"shed", "garage", "garages", "carport", "roof", "hut", "cabin", "barn", "farm_auxiliary",
                "greenhouse", "service", "storage_tank", "toilets", "kiosk", "parking", "shelter"}
# A grain elevator (the plains' small towns: Strasburg, CO by I-70 had two 8-storey brick apartment blocks with windows
# in the farmland): a tall slender footprint of no class (or a silo's) in a US small town, plain concrete
# A pyramid (Giza's, GitHub #47: drawn as 66-140 m boxes with windows): a pyramidal roof mapped as (nearly) the whole
# height over a monument-sized footprint is a stone pyramid on a low plinth (pitched roofs are otherwise flat past 28 m)
PYRAMID_ROOF_SHARE = 0.8
PYRAMID_MIN_M2 = 500.0
PYRAMID_PLINTH_M = 1.0
ELEVATOR_CLASSES = {"", "industrial", "silo", "grain_silo", "storage_tank", "farm_auxiliary", "agricultural"}
ELEVATOR_MIN_H = 15.0       # m mapped (a height, no floors)
ELEVATOR_MAX_M2 = 1500.0
ELEVATOR_SLENDER = 0.7      # height / sqrt(area) at least (11 x 11 m, 25 m: 2.3; 28 x 31 m, 23 m: 0.78; a 15 m motel: 0.4)
GLASS_FLOORS = (20, 40)     # towers from 20 floors: glass, but for OLD_TOWER_GLASS of the old ones below 40 ...
OLD_TOWER_GLASS = 0.65      # ... (Midtown's pre-war skyscrapers are stone)
MIDRISE_GLASS_OFFICE = 0.45 # offices / hotels of 7+ floors: this share glass (modern), the rest masonry
TOWER_SHOP_FLOORS = 6       # a building this tall in a dense core: a shop / lobby ground floor whatever its class
SHOP_CLASSES = {"retail", "commercial", "supermarket", "kiosk", "office", "hotel", "civic", "restaurant"}
HOUSE_CLASSES = {"house", "detached", "semidetached_house", "bungalow", "residential", "farm", "static_caravan",
                 "cabin", "dormitory"}
ROW_CLASSES = {"terrace"}
GARAGE_MIN_M2 = 400.0       # a "parking" building this big is a parking structure
INDUSTRIAL_CLASSES = {"industrial", "warehouse", "manufacture", "factory", "hangar"}


def _hash01(*vals):
    """A number in [0, 1) from the values (stable across runs and machines)."""
    h = hashlib.blake2b(repr(vals).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


# ------------------------------------------------------------------------------------------- polygons

def _area2(p):
    """Twice the signed area of the ring p (n, 2), counter-clockwise positive."""
    x, y = p[:, 0], p[:, 1]
    # np.roll(v, -1) by hand: the same arrays without roll's axis handling (~28k calls a 9-tile set)
    return float(np.dot(x, np.concatenate((y[1:], y[:1]))) - np.dot(np.concatenate((x[1:], x[:1])), y))


def clean_ring(p):
    """The ring (n, 2) without its closing point, duplicate and nearly collinear vertices, counter-clockwise;
    None if fewer than 3 are left. The simplification in Rust (earthfast.clean_ring_core: numpy's arithmetic)."""
    fast.require(76, "clean_ring_core")
    p = fast._ef.clean_ring_core(np.ascontiguousarray(np.asarray(p, np.float64)[:, :2]), float(SIMPLIFY_M), int(MAX_VERTS))
    if p is None:
        return None
    p = np.asarray(p, np.float64)
    if _area2(p) < 0:
        p = p[::-1]
    return p


def triangulate(p):
    """Ear clipping of the counter-clockwise simple ring p (n, 2): (n-2, 3) indices into p (fewer if the
    ring is not simple: the rest is left out). The Rust loop (earthfast.triangulate)."""
    fast.require(76, "triangulate")
    return np.asarray(fast._ef.triangulate(np.ascontiguousarray(np.asarray(p, np.float64)[:, :2])), np.int64)


def oriented_box(p):
    """The minimum-area rectangle around ring p along one of its edges: (centre (2,), axis u (2,) along the
    long side, half length, half width). The edge search in Rust (earthfast.oriented_box_core: the same products and
    sums, math.hypot's own rounding), the rest here as before."""
    fast.require(76, "oriented_box_core")
    best = fast._ef.oriented_box_core(np.ascontiguousarray(np.asarray(p, np.float64)[:, :2]))
    if best is None:
        raise ValueError("oriented_box: a ring without an edge")
    ux, uy, vx, vy, amin, amax, bmin, bmax = best
    sa, sb = amax + amin, bmax + bmin
    c = np.array([ux * sa / 2 + vx * sb / 2, uy * sa / 2 + vy * sb / 2])
    hl, hw = np.float64((amax - amin) / 2), np.float64((bmax - bmin) / 2)     # (numpy scalars, as before)
    u = np.array([ux, uy])
    if hw > hl:
        u, hl, hw = np.array([vx, vy]), hw, hl
    return c, u, hl, hw


ALPINE_HUT_M2 = 120.0        # an isolated building this small above the tree line (a summit shelter, a mountain hut) ...
ALPINE_COVER = 0.02          # ... in a tile this empty gets dressed stone walls (the user, 2026-10-14: Mount Whitney's
                             # stone summit hut was a plain grey box)


LANDMARK_MATERIALS = {"iron": LANDMARK_IRON, "stone": LANDMARK_STONE, "floodlit": LANDMARK_FLOODLIT}


# A sash window cut from the plaster sheet (band 2, bay 0, its sill included): u, v ranges (Collada v up)
WIN_UV = (53 / 1024, 154 / 1024, 1.0 - (512 + 238) / 1024, 1.0 - (512 + 45) / 1024)
WIN_W, WIN_H, WIN_SILL = 1.05, 1.75, 0.85   # m
HOUSE_BAY_M = 3.4           # m between a house's windows
DOOR_GAP_M = 0.75           # m each side of a front door kept free of the ground floor's windows
WIN_OUT_M = 0.03            # m the window stands proud of its wall


PLAIN_STRIP = 0.3           # of a bay: the plain wall between two windows of a sheet's band (centred on a bay edge)


PORCH_DEPTH_M = 2.2
PORCH_H_M = 2.7             # the porch roof's underside above the floor
PORCH_DECK_M = 0.35         # the deck's top above the ground


PORTAL_NEAR_M = 25.0        # m round a building where the road at a tunnel's mouth is looked for


MARBLE_LOOK = "bne_marble"


CANOPY_POSTS_MAX_M = 8.0    # m: a canopy's underside up to this high stands on posts (higher: mapped on its own
CANOPY_POST_STEP_M = 8.0    # parts, a toll plaza's or a station's); posts this far apart along its length, ...
CANOPY_POST_HALF_M = 0.09   # ... this thick, set in this far from its edge
CANOPY_POST_INSET_M = 0.4


ROOF_CLUTTER_M2 = 250.0     # flat roofs this large get boxes: air-conditioning units, stair bulkheads


# Domes and onion domes from the map's roof:shape (GitHub #593's thread: "a dedicated landmark procedural generator"
# instead of hand-made models): a cathedral's, a mosque's or a capitol's dome is mapped as a building part with
# roof:shape=dome / onion and often its roof:height; drawn as a gable before. The footprint's own outline shrinks to
# a point up the profile (a round part: a dome; a square one: a cloister vault), (radius share, rise share) per ring.
DOME_ROOFS = ("dome", "onion")
DOME_RISE = {"dome": 1.0, "onion": 1.6}     # the rise over the footprint's half width when roof:height is not mapped
DOME_MAX_HW = 80.0          # m: the largest half width drawn as a dome (St. Peter's is ~21 m, the Pantheon's 22 m)
DOME_PROFILE = {
    "dome": tuple((math.cos(a), math.sin(a)) for a in np.linspace(0.0, math.pi / 2, 9)),
    "onion": ((1.0, 0.0), (1.18, 0.15), (1.22, 0.3), (1.1, 0.45), (0.8, 0.6), (0.45, 0.75), (0.2, 0.88),
              (0.06, 0.96), (0.0, 1.0)),
}


# The engine draws a shape's mesh with 16-bit indices: past 65,535 vertices they wrap and triangles join far-apart
# vertices (the user's report 2026-10-14: slabs across a Fells Point street; dense tiles had up to 81k vertices).
MAX_MESH_VERTS = 60000


# A parking structure you can drive into (the user, 2026-10-14: "garage buildings ... are solid"): open storeys of
# concrete decks on perimeter columns, a parapet round each upper deck, and a two-lane switchback ramp along the
# longest wall (one lane up per storey, a U-turn landing at each end). Only where the ramp fits inside the footprint
# and the ground under it is near level; otherwise the closed box as before. The tile's mesh collides as drawn.
DECK_LANE_M = 3.6           # a ramp lane
DECK_EDGE_M = 0.8           # the ramp lanes start this far in from the wall (the columns stand there)
DECK_LANDING_M = 7.0        # flat deck at each end of the ramps (the U-turn)
DECK_SLOPE = 0.17           # the steepest ramp
DECK_SLAB_M = 0.3
DECK_PARAPET_M = 1.0
DECK_COLUMN_M = 9.0         # perimeter column spacing
DECK_LEVEL_M = 3.5          # ground this uneven (max - min) under the footprint: no deck (up to it: the ground deck
                            # level at the highest ground on a plinth, driven into from the uphill side)
DECK_HEADROOM_M = 2.6       # the lowest storey height that still drives
DECK_STOREY_M = 3.0         # a deck's storey where the mapped floors are too many for the height


DECK_RAMP_EDGES = 4         # the longest edges tried for the ramp, longest first


# A garage's skin (the user's reports #51/#52, 2026-10-17: "Garages look ugly and basic", "all garages look the
# same"; Canton's on Mapillary: brick or precast concrete spandrels with long openings between, solid corners, a
# stair tower over the roof): per garage (hashed) one of GARAGE_SKINS
GARAGE_SKINS = ("bne_brick_red", "bne_brick_dark", "bne_brick_tan", "bne_precast", "bne_precast_warm")
SPANDREL_H = 1.1            # m of spandrel over each deck (the slab's edge behind it)
SPANDREL_T = 0.3            # m thick, its face just outside the outline
PIER_W = 0.9                # m: the solid corners and every PIER_EVERY_M along the outline
PIER_EVERY_M = 18.0
STAIR_TOWER_M = 4.5         # m square, at the outline's corner farthest from the ramps, ...
STAIR_TOWER_UP = 3.2        # ... this far over the top deck


GARAGE_DRIVE_W = 6.5        # m: a garage entrance's two lanes
GARAGE_DRIVE_MAX_M = 25.0   # m from the outline to a road's edge at most


OPEN_CANOPY_CLASSES = {"carport", "roof"}     # OSM building=carport / roof: a roof on posts, open under it
CANOPY_CLEAR_M = 2.6        # m: its underside (a car park canopy's clearance) when its height is not mapped ...
CANOPY_SLAB_M = 0.4         # ... and the roof slab's thickness
ROAD_THROUGH_M = 15.0       # a footprint that public roads run through this far is a structure over them
OVER_ROAD_CLEAR_M = 5.5     # ... lifted this high (an expressway's or a station's slab over the streets: Baltimore's
                            # 17,500 m2 one by I-83, the AI drove into its wall at 35 mph, 2026-10-14)
LEVEL_THROUGH_M = 5.0       # ... this far for one mapped over the ground (OSM layer >= 1, Overture's level): Noranda Station's
                            # concourse over Tonkin Highway in Perth (layer 1, no min_height), 80 x 118 m over the platforms
                            # in the median and 6 m of one carriageway, stood as a wall across its lanes (GitHub #914)
SITE_M2 = 100_000.0         # m2: a footprint this large that public roads run through (ROAD_THROUGH_M) is a mapped site
                            # (a planned development, a campus outline tagged building), not a structure over them: left
                            # out, not lifted (the largest decks over roads are a few hectares; OSM's 1.35 km2 "Blueberry
                            # Commons" by I-94 near Huntley MT, its streets inside, stood 5.5 m up as a 15 m wall over
                            # empty prairie, Tann2019 2026-11-05)
ACROSS_W = 8.0              # m: a footprint at most this wide that a public road crosses (its long middle line) is a
                            # walkway or canopy over it (GitHub #48: the 69th Street station's 3 m shelters stood across
                            # the Dan Ryan as brick walls; a road crossing one runs only its width inside it)
THROUGH_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified",
                   "living_street"}


GARAGE_TAGS = ("multi-storey", "multistorey", "multi_storey", "multi-level")
GARAGE_TAG_CLASSES = {"", "yes", "commercial", "garage", "garages", "parking", "parking_garage", "carport",
                      "service", "transportation", "industrial", "retail"}     # (not the apartments on a car park podium)


TUNNEL_THROUGH_M = 5.0      # a footprint a road tunnel runs into this far: its portal building, lifted over the tube
PORTAL_SKIP_CLASSES = ("service",)     # tunnels that lift no building: a driveway, loading dock or covered drop-off
                            # under a building, which stands on the ground (GitHub #60: a 15 m drive-through under an
                            # Oklahoma City church floated it 10 m up; in downtown Baltimore and Chicago 20 office and
                            # garage blocks of up to 7,156 m2 stood 7.4-10.5 m up over loading-dock tunnels, 2026-10-20)
OVER_TUNNEL_CLEAR_M = 7.5   # (the tube's vault is roadmesh.TUNNEL_H = 6.8 m over the floor; 2026-10-17: Eisenhower's
                            # east portal building closed the tunnel)


PASSAGE_W = 11.0           # m: the passage a bridge's road takes through its tower (Tower Bridge: 2 lanes + walks) ...
PASSAGE_H = 7.0            # ... this high over the tower's mapped bottom (its deck: the part starts at the road), ...
PASSAGE_CLEAR = 6.5        # m: the way over a deck through a building (with the decks' heights)
PASSAGE_BOTTOM_M = 20.0    # ... in a structure whose bottom is this low (not its high walkways or tops)


DECK_STAND_M = 2.0      # m: a deck this far over the ground under a footprint is what the building stands on ...
DECK_INSIDE_M = 1.0     # ... and it is left out where it reaches this far inside the deck's road (its width / 2) ...
DECK_REACH_M = 6.0      # m past the road's edge a footprint's centre still counts as on its deck
DECK_SEA_M = 0.5        # ... only over ground this low: water, a beach (a building on dry land under a viaduct stays)


ML_BRIDGE_M = 6.0          # m of a road bridge's line through a machine-learned footprint: the deck, not a building
ML_ROAD_M = 4.0            # m from a public road's centre line to a machine-learned footprint's centre: a vehicle on it
                           # (at least; where the road's drawn width is known, ML_ROAD_EDGE_M past its half-width)
ML_ROAD_EDGE_M = 1.0       # m past a road's drawn edge a machine-learned footprint's centre still counts as on its lanes
                           # (I-280 at Cupertino, 2026-11-08: Overture aed44c00-..., Microsoft's 8 x 30 m outline of a
                           # lorry, height 0.73 m, 7.0 m from the centre line of a 5-lane, 26 m wide carriageway stood as a
                           # house in the fast lane: ML_ROAD_M alone covered its middle two lanes)
ML_MEDIAN_M = 9.0          # m: a machine-learned footprint between two parallel roads this close on either side lies in a
                           # dual carriageway's median (GitHub #90: a house between Moscow's Bolshaya Tulskaya carriageways)


ML_STEEP_DEG = 25.0         # a machine-learned footprint on ground this steep ...
ML_FAR_FROM_WAY_M = 60.0    # ... this far from any mapped way (a road, track or path) is a rock or a shadow, not a hut
ML_CLIFF_DEG = 36.0         # ... and on ground this steep, wherever it is


CENTRE_PREFILTER_M = 1.0       # m of slack for Features.outer_centres (cumulative sums: within ~1e-5 m of the exact mean)


# A building is drawn as its parts (OSM's Simple 3D Buildings: the outline is not rendered where parts exist), but its
# parts must cover it: Ülemiste keskus, Tallinn (GitHub #608), a 51,000 m² mall mapped with height 10, has a single
# 452 m² part (height 5), and was drawn as that part alone, the rest of it bare ground. Of 409 buildings drawn as their
# parts in a player's cache (Tallinn, Baltimore, ...), 63 had parts over less than half their outline. Under
# PARTS_COVER_MIN the outline is drawn as well, the parts standing on it (one lower stays inside it), when it has a
# height or levels of its own (PARTS_COVER_MIN_H) and every part stands on the ground: a part off the ground (min_height)
# is a structure modelled in parts, its gaps open on purpose (a tower's legs, an arcade: its outline is its overall box).
PARTS_COVER_ON = os.environ.get("BNE_PARTS_COVER", "on") != "off"
PARTS_COVER_MIN = 0.5           # share of the outline the parts cover (grid samples inside it), below which it is drawn
PARTS_COVER_SAMPLES = 400       # about this many samples inside an outline
PARTS_COVER_MIN_H = 2.5         # m: an outline mapped lower than this is a base or an area, not drawn under its parts
PARTS_GROUND_M = 0.5            # m: a part whose bottom is higher than this is off the ground


# Toll plazas (the user's report at the Golden Gate Bridge, 2026-10-21: "Toll booths need work. west coast has one,
# reference that"): OSM maps a plaza's islands as low "buildings" (0.2 m there) with the booths as building parts
# standing on them; drawn as buildings they were bare boxes on slabs. An island with a booth on it, along a road, is
# West Coast USA's toll booth instead (trees.lua group "tollbooth": the island, its booth and kerbs in one shape, 22 m
# long along its local y), placed along the island (toll_booth_rows); the island and its booth parts are not built.
FLAT_FOOTPRINT_M = 0.1     # m: a footprint mapped lower than this is an area (a plaza), not drawn
G_TOLLBOOTH = "tollbooth"
TOLLBOOTH_H = 4.96          # m: the stock shape's height (s_toll_booth_center)
TOLL_ISLAND_MAX_H = 1.0     # m: an island is a "building" no taller than this ...
TOLL_ISLAND_MIN_L = 8.0     # ... at least this long ...
TOLL_ISLAND_MAX_W = 4.0     # ... and at most this wide
TOLL_BOOTH_H = (2.0, 8.0)   # m: a booth on it: a footprint this tall ...
TOLL_BOOTH_MAX_AREA = 40.0  # ... and this small (m2), its centre on the island
TOLL_ROAD_M = 25.0          # m: a road this near the island's centre ...
TOLL_ROAD_DEG = 25.0        # ... running along it within this


# The engine's LOD size of a near tile's shape below which its buildings are drawn as the lite mesh (the far tiles' one:
# no window bays, steps, porches, signs, cornices or roof clutter): the engine's own camera LOD (cdae lods), the tile's
# whole shape at once. Calibrated in game (2026-10-18, 1440p, the game's mesh detail at its default 2): a 21 m radius
# probe switched between 450 and 550 m, so a tile of ~190 m radius switches ~500 m from its centre (less with a lower
# mesh detail or screen resolution, as the stock shapes do).
BLD_LOD_PX = 2700.0


DOOR_OUT_M = 0.09            # m the door model's origin stands off the wall: its panel is 6.9 cm behind the origin
                             # (both Building Architect doors), so at 3 cm it was inside the wall, only the frame
                             # showing (Fells Point, 2026-10-17)
STOOP_SHARE = 0.8           # row houses with a front stoop (Baltimore's white marble steps)
# A row house's marble stoop (the user's report #47, 2026-10-17: "the steps model isnt good. Try and find one from the
# offical maps"): built as the official maps' stair blocks are (West Coast USA's stairblock1, Italy's old stairs:
# treads and risers up to a landing between stepped side walls); their own models carry race-track paint and logos
STOOP_H = 0.54              # m: three steps up to the door (Fells Point's on Mapillary, 2026-10-17: low white marble
STOOP_STEPS = 3             # steps, no side walls; ours were four between walls, big white blocks)
STOOP_RISE = 0.18           # m a step (more of them where the street falls away from the house)
STOOP_MAX_STEPS = 7
STOOP_RISE_MAX = 0.23       # m: the steepest outdoor step
STOOP_TREAD = 0.3           # m deep each step
STOOP_LANDING = 0.6         # m deep the landing at the door
STOOP_W = 1.3               # m wide the steps
DOOR_ROAD_M = 30.0          # a front door on the wall facing the nearest road within this
DOOR_CLASSES_SKIP = SHED_CLASSES | INDUSTRIAL_CLASSES
G_DOOR = "door"
FAR_SEND_M = float(os.environ.get("BNE_BLDFAR_SEND_M", "1600"))    # far building tiles (render-only, the lite mesh)
                                                                    # sent within this (Euclidean: roads.tiles_near)
FAR_DROP_M = float(os.environ.get("BNE_BLDFAR_DROP_M", str(FAR_SEND_M + 300)))     # ... dropped beyond
FAR_PER_POSE = 4
FAR_GEO = 1                 # the far geotiles' protocol (hello.farGeo, the mod's roads.lua M.FAR_GEO): far buildings,
                            # skyline and far bridges in the global tiles' frames (blockworker.far_geo)


MAIN_STREET_CLASSES = {"primary", "secondary", "tertiary"}
BACK_ROAD_CLASSES = {"service", "track", "path", "footway", "cycleway", "pedestrian", "steps", "alley"}
MAIN_STREET_SETBACK_M = 7.0  # m from a main street's edge within which a front wall has a shopfront
MAIN_STREET_SHOPS = 0.75     # the share of such buildings that do


DRIVEWAY_W = 3.2            # m
DRIVEWAY = {"m": "road_asphalt_2lane", "tl": 11.0, "p": 19}     # a road tile chunk's look (roads.LOOK)


# Company signs on walls (West Coast USA's, for its fictional firms): (prop group, width m, height m, panel thin
# along its local x (True) or y (False)). Warehouses: Logistics or Rentabox; big retail boxes: Lemon; one-storey
# restaurants: Turboburger.
WALL_SIGNS = {"logistics": ("wallsign_logistics", 10.8, 5.8, True), "rentabox": ("wallsign_rentabox", 8.2, 5.7, True),
              "lemon": ("wallsign_lemon", 11.5, 3.6, True), "burger": ("wallsign_turboburger", 5.8, 2.4, False)}
WALL_SIGN_SHARE = 0.6
RESTAURANT_CLASSES = {"restaurant", "fast_food", "cafe"}


G_FENCE = "fence"           # trees.lua: East Coast USA's low wooden fence (eca_bld_wood_fence_c, 5.4 m, along x)
FENCE_SEG_M = 5.4
FENCE_SHARE = 0.3           # detached houses with a fence along the front of their yard (hashed)
FENCE_BACK_M = 3.2          # m from the road's edge (the back of the sidewalk)
FENCE_MIN_YARD_M = 2.5      # the front yard at least this deep between the fence and the house
FENCE_CLEAR_M = 2.0         # m every piece keeps from any road's edge (a house on a bend or a corner: its fence ran
                            # along one road out across another, GitHub #105 Falcon Drive, Coquitlam)


TOWN_COVERAGE = 0.16  # a tile whose footprints cover less than this share is a town / suburb (heights)
# Small towns of the arid West (Bishop, Moab: one-storey main streets with false fronts, ranch houses), told
# from cities by the footprint share of the district around a tile (1 km: Bishop 0.14, Moab 0.11; Frederick
# 0.22, Fells Point 0.27): a tile's own share cannot (a main-street block is as dense as a city's).
DISTRICT_M = 1000.0
WEST_TOWN_COVERAGE = 0.18
WEST_TOWN_ONE_STOREY = 0.75  # share of predicted one-storey buildings there
FALSE_FRONT_M = 1.2  # m of parapet over a one-storey shop there
POLES_COVERAGE = 0.30  # ... and its streets get wired wooden poles (furniture; Ocean City's photos: poles


SUPPRESS_MARGIN_M = 5.0  # a baked 3dmr.eu model suppresses procedural buildings this far past its own bbox
SUPPRESS_MIN_M = 15.0  # ... but never less than this (a tiny model, id 41 ~0.5 m, must not blank a block)
SUPPRESS_MAX_M = 150.0  # ... nor more than this (a huge one, id 29 ~61 km, must not blank the whole tile)
MODEL_MAX_M = 1000.0  # a model wider than this is not placed: a district or stray geometry in the file (3dmr.eu ids 6
# and 7: 1.4-1.9 km of it around a building; 29: 61 km), not a building or a bridge (the Dragon Bridge: 842 m)
SUPPORT_DZ_WARN_M = 30.0  # threedmr.write_model's own support_dz past this is only logged, never dropped -
# an author's own legitimate elevated deck (a bridge, a viewing platform) is also a large support_dz
BRIDGE_ASPECT = 3.0  # a model's horizontal bbox this much longer one way than the other is a bridge deck
BRIDGE_MIN_SPAN_M = 100.0  # ... and only once it also reaches this far (a narrow wall or jetty is not a bridge
# deck's stand-in, whatever its own aspect ratio): only a model this shape gets its road-surface triangles
# carved out (_custom_tile_entries, threedmr.carve_bridge_deck_mask) so it stands as pure decoration over the
# procedural road stub it would otherwise fight for height - a monument or a building never is carved

FAR_LAYERS = ("buildings", "building_parts", "infrastructure", "roads", "water")


# ------------------------------------------------------------------------------------------- far street lamps
# The street lamps past the road tiles (the user, 2026-10-27: "extend the street and window lights as far as the
# buildings render"): the far and skyline tiles carry their streets' lamps as glow points (trees.lua's lampglow group,
# night only), placed along the mapped roads by furniture.py's spacing: the town streets in built-up squares (at least
# FAR_LAMP_MIN_BLD buildings in its 256 m square: the near rule's land cover is not at hand here), the highways there
# and every ramp. FAR_LAMP_Z over the ground, over the carriageway's edge.
FAR_LAMP_MIN_BLD = 8
FAR_LAMP_Z = 8.0
FAR_LAMP_MAX = 4000         # a tile's most (the skyline's 1 km squares: a dense city's ~3000)


# ------------------------------------------------------------------------------------------- the skyline
# Past the far buildings, a lighter ring out to SKY_SEND_M (the user, 2026-10-27: "see how far away we can render
# buildings"): one mesh per SKY_UNIT x SKY_UNIT road tiles (1 km), only the buildings big enough to see that far
# (SKY_MIN_H tall or SKY_MIN_AREA_M2 large), the lite mesh, the tallest first up to the 16-bit vertex budget. Far
# buildings to 4.8 km as 256 m tiles cost +37 % frame time at Federal Hill (1129 objects, ~0.7 GB of meshes); a
# skyline tile stands for 16 of them and holds a fraction of their buildings. The far buildings are sent for every
# tile of a skyline square that comes within FAR_SEND_M (vegetation.Tracker group), and the mod shows a skyline tile
# only while none of its squares' far buildings are there: no overlap, no hole.
SKY_UNIT = 4
SKY_SEND_M = float(os.environ.get("BNE_SKY_SEND_M", "4800"))
SKY_DROP_M = SKY_SEND_M + 800.0
SKY_PER_POSE = 2
SKY_MIN_AREA_M2 = 40.0      # (sheds left out)
SKY_BOX_AREA_M2 = 600.0     # a footprint smaller than this is drawn as its oriented box
SKY_MAX_CORNERS = 12        # a larger one with more corners: every k-th kept


# The mean colour of each building material's base texture (tools: a 64 x 64 resize of each material's
# baseColorMap from the game's content zips, 2026-10-27): a skyline building is drawn in its facade's and its roof's
# mean colour times their tints (MARBLE, nearly white, sampled at one texel: the vertex colour is the colour)
MAT_MEAN = {"bne_bld_facade": (0.679, 0.676, 0.683), "bne_bld_shop": (0.669, 0.691, 0.699),
            "bne_bld_office": (0.47, 0.493, 0.492), "bne_bld_roof_flat": (0.432, 0.431, 0.43),
            "bne_bld_roof_slate": (0.359, 0.36, 0.388), "bne_bld_roof_shingle": (0.585, 0.577, 0.571),
            "bne_bld_siding": (0.822, 0.822, 0.822), "bne_bld_stucco": (0.795, 0.795, 0.795),
            "bne_bld_roof_metal": (0.642, 0.642, 0.642), "bne_bld_panel": (0.815, 0.815, 0.814),
            "bne_bld_metalwall": (0.841, 0.841, 0.841), "bne_bld_stone": (0.83, 0.818, 0.803),
            "bne_bld_marble": (0.978, 0.978, 0.978), "bne_bld_brick": (0.717, 0.569, 0.513)}
SKY_UV = (0.5, 0.5)
# the skyline walls' windows (tools/build_sky_windows.py: SKY_WIN_BAYS x SKY_WIN_FLOORS of them in its map), lit at
# night (nightlights.lua; the user, 2026-10-27: "extend the street and window lights as far as the buildings render")
SKY_BAY_M = 3.5
SKY_WIN_BAYS = SKY_WIN_FLOORS = 16


# ------------------------------------------------------------------------------------------- far bridges
# The road bridges past the road tiles (their decks only came with the road tiles, ~0.8 km: a city's bridges were
# gone beyond that): one render-only slab per road tile that has bridge decks, out to BRIDGE_FAR_SEND_M, from the same
# deck solver as the road tiles (roads.BridgeProfiles), each part's segments in the tile of their midpoint (offsets
# from the whole part: tiles meet without a gap). A slab: the deck's top, its sides and its underside down to its
# girders, plain grey (MARBLE at one texel, vertex colours). The mod hides one while its road tile has a deck.
BRIDGE_FAR_SEND_M = float(os.environ.get("BNE_BRIDGE_FAR_SEND_M", "4800"))
BRIDGE_FAR_DROP_M = BRIDGE_FAR_SEND_M + 500.0
BRIDGE_FAR_PER_POSE = 2
BRIDGE_TOP_RGBA = (92, 92, 94, 255)        # the deck's asphalt from afar
BRIDGE_SIDE_RGBA = (176, 174, 168, 255)    # its concrete


FAR_NEED = 106             # earthfast.bridge_slabs


PHOTO_GROW_M = 1.5          # m a footprint is grown on the ground photo (edges, eaves) ...
PHOTO_LEAN = 0.12           # ... plus this share of its height (the photo's lean of tall buildings), at most
PHOTO_LEAN_MAX_M = 6.0      # ... this
PHOTO_FILL_M = 12.0         # m: the blur that fills them from the ground around


BRICK_WALK_COVERAGE = 0.35  # dense old cores (row houses) of brick regions ...
BRICK_WALK_SHARE = 0.55     # ... pave this share of their streets' sidewalks with brick (hashed per street)
