"""Procedural buildings: what is built for one footprint (split out of buildings.py): Spec, the regional Styles and
plan() (height, floors, roof, facade, shop band, landmark).
"""
from runtimeprofiler import step as profile_step
import math
import numpy as np
import landmarks
import ships
import francehouses
from bldbase import (
    ALPINE_HUT_M2,
    BAND_BRICK,
    BAND_OFFICE,
    BAND_OFFICE2,
    BAND_PANEL,
    BAND_SASH,
    BAND_TALL,
    BARN_CLASSES,
    BARN_REDS,
    BIG_BOX_M2,
    BRICK_TINTS,
    BRICK_WALL,
    BRIDGE_CLASSES,
    BRIDGE_PIER_M,
    CHURCH_CLASSES,
    COLOR_NAMES,
    CONCRETE,
    DOME_MAX_HW,
    DOME_RISE,
    DOME_ROOFS,
    ELEVATOR_CLASSES,
    ELEVATOR_MAX_M2,
    ELEVATOR_MIN_H,
    ELEVATOR_SLENDER,
    FACADE,
    FACADE_TINTS,
    FALSE_FRONT_M,
    FLOOR_M,
    GAMBREL_BREAK,
    GAMBREL_PITCHES,
    GARAGE_MIN_M2,
    GARAGE_TAG_CLASSES,
    GLASS_FLOORS,
    GROUND_FLOOR_SHOP_M,
    HOUSE_CLASSES,
    INDUSTRIAL_CLASSES,
    LANDMARK_DIAL,
    LANDMARK_FLOODLIT,
    LANDMARK_LAMP,
    LANDMARK_MATERIALS,
    MANSARD_INSET_M,
    MANSARD_PITCHES,
    MASONRY_MATERIALS,
    MATERIALS,
    METALWALL,
    MIDRISE_GLASS_OFFICE,
    OFFICE,
    OLD_TOWER_GLASS,
    PAINT_TINTS,
    PANEL,
    PITCH,
    PYRAMID_MIN_M2,
    PYRAMID_PLINTH_M,
    PYRAMID_ROOF_SHARE,
    ROOF_FLAT,
    ROOF_METAL,
    ROOF_SHINGLE,
    ROOF_SLATE,
    ROOF_TINTS,
    ROW_CLASSES,
    SHED_CLASSES,
    SHOP_CLASSES,
    SIDING,
    SKILLION_PITCH,
    STEEL,
    STONE,
    STONE_FACADES,
    STONE_FLOORS,
    STUCCO,
    TOWER_SHOP_FLOORS,
    US_WALL_TINTS,
    WALL_TINTS,
    WEST_TOWN_ONE_STOREY,
    WOOD_TINTS,
    _area2,
    _hash01,
    custom_look,
    oriented_box,
    parse_color,
)


class Spec:
    """What is built for one footprint (decided by plan())."""
    __slots__ = ("ring", "floors", "floor_h", "height", "roof", "roof_mat", "facade", "band", "shop", "plain", "min_h", "wall", "door", "windows", "front", "steeple", "stoop",
                 "sign", "porch", "cls", "seed", "deck", "pitch", "portal", "gring", "ship", "fr_house", "us_signs")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class Style:
    """A region's building habits (2026-10-15, the user: "the other parts of the world need more love": Swiss
    villages came out as clapboard houses with porches). Shares are of buildings whose data does not say:
    brick: of facades; walls: (kind, share) for detached houses' plain walls (SIDING, STUCCO; the rest the
    plaster window sheet), wall_tints: the tint indices the plain walls take (None: all but barn red);
    paint: plaster houses painted (PAINT_TINTS); porch; flat: detached houses with flat roofs; hipped: of the
    pitched; roofs: (roof material, (tint indices)) to pick from; pitch: the roof pitch in degrees (None: PITCH);
    stone: the share of low older buildings (houses and attached ones, up to STONE_FLOORS, no shops) with dressed stone
    walls; wood: siding walls are stained wood (WOOD_TINTS, barns too); barn_red: the share of barns painted barn red
    (the US's and Scandinavia's; elsewhere their siding colours: a Lauterbrunnen garage row came out a 150 m red
    wall, 2026-10-15); attached: low attached buildings (terraces, old town
    blocks) with a pitched roof (the US: flat; Amsterdam's canal houses, Europe's old towns: gables).
    Coarse boxes by region until a facade-material map; the US ones keep their earlier behaviour."""

    def __init__(self, name, brick, walls, porch, flat, hipped, roofs, pitch=None, paint=0.3, wall_tints=None,
                 wood=False, attached=0.0, barn_red=0.0, stone=0.0):
        self.name, self.brick, self.walls, self.porch, self.flat = name, brick, walls, porch, flat
        self.barn_red, self.stone = barn_red, stone
        self.hipped, self.roofs, self.pitch, self.paint, self.wall_tints = hipped, roofs, pitch, paint, wall_tints
        self.wood, self.attached = wood, attached


_US_ROOFS = ((ROOF_SLATE, (0, 1, 2, 3)), (ROOF_SHINGLE, (0, 1, 2, 3)))
_CLAY = (ROOF_SLATE, (4, 5, 6))
_DARK = (ROOF_SLATE, (7, 8, 9))
STYLES = {
    "us_east": Style("us_east", 0.75, ((SIDING, 0.75),), 0.55, 0.0, 0.4, _US_ROOFS, barn_red=0.5),
    "us_south": Style("us_south", 0.35, ((SIDING, 0.75),), 0.55, 0.0, 0.4, _US_ROOFS, barn_red=0.5),
    "us_plains": Style("us_plains", 0.15, ((SIDING, 0.75),), 0.55, 0.0, 0.4, _US_ROOFS, barn_red=0.5),
    "us_west": Style("us_west", 0.15, ((STUCCO, 0.65), (SIDING, 0.25)), 0.25, 0.0, 0.4, _US_ROOFS, barn_red=0.5),
    # Alpine villages: white plaster ground floors, dark wooden chalets, low wide gables of dark or brown roofs
    "alpine": Style("alpine", 0.05, ((SIDING, 0.45),), 0.0, 0.03, 0.15, (_DARK, (ROOF_SLATE, (9, 5))), pitch=24.0,
                    paint=0.05, wood=True, attached=0.85, stone=0.15),
    # north-west / central Europe: brick (Low Countries, northern Germany, Britain) or plaster, steep tiled or
    # slate roofs
    "europe_north": Style("europe_north", 0.5, (), 0.02, 0.05, 0.3, (_CLAY, _DARK, (ROOF_SLATE, (5,))), pitch=42.0,
                          paint=0.15, attached=0.75, stone=0.12),
    # Scandinavia and Finland: painted wooden houses (siding), little brick, dark or tiled roofs, falu-red barns
    "nordic": Style("nordic", 0.12, ((SIDING, 0.65),), 0.0, 0.08, 0.2, (_DARK, _CLAY), pitch=35.0, paint=0.2,
                    attached=0.3, barn_red=0.7),
    "mediterranean": Style("mediterranean", 0.08, ((STUCCO, 0.85),), 0.0, 0.35, 0.5, (_CLAY,), pitch=22.0,
                           paint=0.1, wall_tints=(0, 1, 2, 3, 4, 5), attached=0.8, stone=0.4),
    "latin_america": Style("latin_america", 0.25, ((STUCCO, 0.9),), 0.02, 0.7, 0.3, (_CLAY, (ROOF_METAL, (0, 1))),
                           pitch=22.0, paint=0.0, wall_tints=(0, 1, 2, 3, 5, 6, 7, 8, 9, 10)),
    "east_asia": Style("east_asia", 0.08, ((SIDING, 0.5), (STUCCO, 0.3)), 0.0, 0.35, 0.5,
                       (_DARK, (ROOF_SHINGLE, (6, 7)), (ROOF_METAL, (1, 4))), pitch=25.0, paint=0.0,
                       wall_tints=(0, 1, 6, 5), attached=0.3),
    "oceania": Style("oceania", 0.55, ((STUCCO, 0.5), (SIDING, 0.3)), 0.25, 0.05, 0.7, (_CLAY, (ROOF_METAL, (0, 1, 4))),
                     pitch=24.0, paint=0.1),
    # South Asia (India, Pakistan, Bangladesh, Sri Lanka, Nepal): a concrete frame and brick under plaster, flat terrace
    # roofs, pastel and whitewashed walls, metal roofs on sheds (OmdevShastri, PR #146)
    "south_asia": Style("south_asia", 0.08, ((STUCCO, 0.92),), 0.0, 0.94, 0.2, (_CLAY, (ROOF_METAL, (0, 1, 3))),
                        pitch=18.0, paint=0.15, wall_tints=(0, 1, 2, 3, 5, 6, 8, 10), attached=0.90),
    # elsewhere (Africa, the Middle East, ...): concrete and plaster, mostly flat roofs
    "warm": Style("warm", 0.15, ((STUCCO, 0.85),), 0.0, 0.75, 0.4, (_CLAY, (ROOF_METAL, (0, 1, 3))), pitch=20.0,
                  paint=0.0, wall_tints=(0, 1, 2, 3, 4, 5, 6, 8)),
    "cool": Style("cool", 0.3, ((SIDING, 0.4),), 0.05, 0.1, 0.3, (_DARK, (ROOF_METAL, (0, 1, 2, 3))), pitch=30.0,
                  paint=0.3, barn_red=0.5),
}


def region_style(lat, lon):
    """The Style of the region around (lat, lon) (None: the US East's)."""
    if lat is None or lon is None:
        return STYLES["us_east"]
    if 24.0 <= lat <= 72.0 and -170.0 <= lon <= -52.0:          # North America (and Canada)
        if lon < -100.0:
            return STYLES["us_west"]
        if lon <= -95.0:
            return STYLES["us_plains"]
        return STYLES["us_east"] if lat > 36.0 else STYLES["us_south"]
    if -56.0 <= lat < 24.0 and -120.0 <= lon <= -30.0:          # Mexico, the Caribbean, Central and South America
        return STYLES["latin_america"]
    if 45.4 <= lat <= 48.0 and 5.8 <= lon <= 16.5:              # the Alps
        return STYLES["alpine"]
    if 56.0 <= lat <= 72.0 and 4.5 <= lon <= 32.0:              # Norway, Sweden, Finland (brick Copenhagen, Malmo: not)
        return STYLES["nordic"]
    if 46.0 <= lat <= 72.0 and -11.0 <= lon <= 32.0:
        return STYLES["europe_north"]
    if 30.0 <= lat < 46.0 and -10.0 <= lon <= 40.0:             # southern Europe, North Africa's coast, Levant
        return STYLES["mediterranean"]
    if 20.0 <= lat <= 46.0 and 100.0 <= lon <= 146.0:           # China, Korea, Japan, Taiwan
        return STYLES["east_asia"]
    if -50.0 <= lat <= -10.0 and 110.0 <= lon <= 180.0:         # Australia, New Zealand
        return STYLES["oceania"]
    if 6.0 <= lat <= 37.5 and 68.0 <= lon <= 92.5:              # South Asia (not Myanmar, east of 92.5)
        return STYLES["south_asia"]
    if abs(lat) >= 50.0:
        return STYLES["cool"]
    return STYLES["warm"]


def brick_share(lat, lon):
    """The share of brick facades in a region (old East Coast / Midwest towns are mostly brick; the South and
    the West plaster, siding and stucco; region_style elsewhere). A coarse stand-in until a facade-material map."""
    if lat is None:
        return 0.4
    return region_style(lat, lon).brick


def roof_pitch(spec):
    """The roof pitch (degrees) of a Spec: its region's, else PITCH."""
    return spec.pitch or PITCH.get(spec.roof, PITCH["gabled"])


def roof_rise(roof, hw, pitch):
    """How far a roof rises (m) over a box of half width hw, pitch: its region's or PITCH's (a form with its own
    pitches keeps those). Pure."""
    tan = lambda deg: math.tan(math.radians(deg))
    if roof == "skillion":
        return 2.0 * hw * tan(SKILLION_PITCH)
    if roof == "mansard":
        inset = min(0.3 * hw, MANSARD_INSET_M)
        return inset * tan(MANSARD_PITCHES[0]) + (hw - inset) * tan(MANSARD_PITCHES[1])
    if roof == "gambrel":
        out = hw * (1.0 - GAMBREL_BREAK)
        return out * tan(GAMBREL_PITCHES[0]) + (hw - out) * tan(GAMBREL_PITCHES[1])
    return hw * tan(pitch)


def tree_line_m(lat):
    """A rough tree line (m) by latitude: ~3,600 m to 32 deg, falling ~110 m per degree (the Alps at 46 deg ~2,050 m,
    Scandinavia at 60 deg ~520 m), sea level towards 70 deg. Pure."""
    return max(0.0, 3600.0 - 110.0 * max(0.0, abs(float(lat)) - 32.0))


def landmark_spec(ring, name):
    """The Spec of landmark `name` (landmarks.WAYS) on its footprint ring: drawn by landmarks.build, no walls."""
    h = landmarks.height(name)
    return Spec(ring=np.asarray(ring, np.float64), floors=1, floor_h=h, height=h, roof="flat", roof_mat=ROOF_FLAT,
                facade=name, band=0, shop=False, plain=True, min_h=0.0, wall=None, door=None, windows=None, front=None,
                steeple=False, stoop=False, sign=None, porch=False, cls=landmarks.LANDMARK, seed=0.5, deck=False,
                pitch=0.0, portal=False)


def _landmark_tokens():
    """landmarks.build's tokens: its lamps and dials, in their own materials (lit at night)."""
    return {"lamp": (MATERIALS.index(LANDMARK_LAMP), landmarks.LAMP_COLOUR),
            "dial": (MATERIALS.index(LANDMARK_DIAL), landmarks.BB_DIAL),
            "floodlit": (MATERIALS.index(LANDMARK_FLOODLIT), (255, 255, 255, 255))}


def _landmark_token(spec):
    return (MATERIALS.index(LANDMARK_MATERIALS[landmarks.material(spec.facade)]), landmarks.colour(spec.facade))


@profile_step('Building plan')
def plan(ring, attrs, neighbours=0, lat=None, lon=None, look=None, town=False, dense=False, west_town=False,
         alpine=False):
    """The Spec of a footprint (local ring (n, 2) in metres, counter-clockwise) from its Overture attributes
    (dict: class, height, num_floors, min_height, roof_shape, roof_height; garage: a mapped multi-storey car park
    covers it, garage_rings) and the number of buildings
    sharing a wall with it. Height and roof are predicted when not given. west_town: a small town of the
    arid West (district_town): single-storey main streets and ranch houses."""
    if attrs.get("landmark"):           # (landmarks.py: its own model, no prediction)
        return landmark_spec(ring, attrs["landmark"])
    area = _area2(ring) / 2
    c = ring.mean(0)
    # (its random choices: keyed on the map feature's id, the same from every anchor and tile; a ring without one
    # by its local centroid, as before. Keyed on the centroid in the tile's frame alone, a house changed its look
    # whenever a rebase moved the anchor: geotiles, 2026-10-29)
    sid = attrs.get("seed_id")
    seed = look if look is not None else _hash01(sid, "plan") if sid is not None else \
        _hash01(round(float(c[0]) * 2), round(float(c[1]) * 2), round(area))
    if attrs.get("ship"):           # a moored ship (ships.is_ship): a hull, not a building
        return ships.spec(ring, attrs, seed)
    if attrs.get("pier"):           # a quay structure alongside one (ships.pier_alongside): a low deck
        return ships.pier_spec(ring, seed)
    style = region_style(lat, lon)
    _, _, hl, hw = oriented_box(ring)
    box_fill = area / max(4 * hl * hw, 1e-6)
    cls = (attrs.get("class") or "").lower()
    name = str(attrs.get("name") or "").lower()
    terminal = any(w in name for w in ("terminal", "concourse"))
    church = cls in CHURCH_CLASSES or any(w in name for w in ("church", "chapel", "cathedral"))
    if "hangar" in name:
        cls = "hangar"
    # a parking structure (a small "parking" building is a carport or a booth: a shed); "parking" is also a
    # shed class, which made big garages barns in barn-red siding (Columbia, 2026-10-14)
    garage = (cls in ("parking", "parking_garage") and area >= GARAGE_MIN_M2) or "parking garage" in name or (
        "garage" in name and area > 800) or (bool(attrs.get("garage")) and cls in GARAGE_TAG_CLASSES)
    narrow = hw < 4.2 and hl > 2.2 * hw                 # a row house: ~4-8 m wide, deep
    # in a dense core small buildings are row houses even where the footprints leave gaps (Fells Point)
    row = cls in ROW_CLASSES or (not west_town and ((narrow and (neighbours >= 1 or dense)) or (
        dense and area < 260 and cls not in SHOP_CLASSES)))
    shed = (cls in SHED_CLASSES and not garage) or (area < 40 and not row)
    shop = cls in SHOP_CLASSES
    industrial = cls in INDUSTRIAL_CLASSES or (area > 2500 and cls in ("", "commercial"))
    industrial = industrial and not (garage or terminal)

    h = attrs.get("height")
    nf = attrs.get("num_floors")
    if terminal and (h is None or not np.isfinite(h)) and (nf is None or not np.isfinite(nf)):
        h, nf = 13.0, 2                                 # an airport terminal: two tall glass storeys
    if garage and (h is None or not np.isfinite(h)) and (nf is None or not np.isfinite(nf)):
        nf = 4 + int(seed * 2)
    if cls == "hangar" and (h is None or not np.isfinite(h)):
        h, nf = 12.0, 1
    if church and (h is None or not np.isfinite(h)) and (nf is None or not np.isfinite(nf)) and area < 2500:
        nf = 1                                          # a nave: one tall storey under a steep gable
    h = None if h is None or not np.isfinite(h) or h <= 0 else float(h)
    nf = None if nf is None or not np.isfinite(nf) or nf <= 0 else int(round(nf))
    rh = attrs.get("roof_height")
    rh = float(rh) if rh is not None and np.isfinite(rh) else None
    if ((attrs.get("roof_shape") or "").lower() == "pyramidal" and h is not None and rh is not None
            and rh >= PYRAMID_ROOF_SHARE * h and area >= PYRAMID_MIN_M2 and hw > 1.0):
        look = f"{STONE}|0"
        return Spec(ring=ring, floors=1, floor_h=PYRAMID_PLINTH_M, height=PYRAMID_PLINTH_M, roof="hipped", roof_mat=look,
                    facade=look, band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None, windows=False,
                    steeple=False, porch=False, cls="pyramid", deck=False,
                    pitch=math.degrees(math.atan2(max(h - PYRAMID_PLINTH_M, 0.1), hw)))
    if (town and style.name.startswith("us_") and cls in ELEVATOR_CLASSES and h is not None and h >= ELEVATOR_MIN_H
            and nf is None and area <= ELEVATOR_MAX_M2 and h / math.sqrt(max(area, 1.0)) >= ELEVATOR_SLENDER
            and not (church or terminal or garage)):
        return Spec(ring=ring, floors=1, floor_h=h, height=h, roof="flat", roof_mat="bne_precast", facade="bne_precast",
                    band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None, windows=False, steeple=False,
                    porch=False, cls="elevator", deck=False, pitch=None)
    if attrs.get("bridge_part"):    # a mapped part of a bridge's tower (support_rings): a plain box of painted steel
        return Spec(ring=ring, floors=1, floor_h=h or BRIDGE_PIER_M, height=h or BRIDGE_PIER_M, roof="flat",
                    roof_mat=f"{STEEL}|{attrs['bridge_part']}", facade=f"{STEEL}|{attrs['bridge_part']}", band=None,
                    shop=False, plain=True,
                    seed=seed, min_h=0.0, wall=None, windows=False, steeple=False, porch=False, cls="bridge_part",
                    deck=False, pitch=None)
    mat_ = str(attrs.get("facade_material") or "").lower()
    if attrs.get("bridge_masonry") or (cls in BRIDGE_CLASSES and mat_ in MASONRY_MATERIALS):
        # a mapped masonry pier (Tower Bridge's 9 m concrete piers under its stone towers): a solid block, not a frame
        look = f"{STONE}|0" if mat_ in ("stone", "brick") else "bne_precast"
        return Spec(ring=ring, floors=1, floor_h=h or BRIDGE_PIER_M, height=h or BRIDGE_PIER_M, roof="flat",
                    roof_mat=look, facade=look, band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None,
                    windows=False, steeple=False, porch=False, cls="bridge_pier", deck=False, pitch=None)
    if cls in BRIDGE_CLASSES:       # a bridge's tower or pier: an open steel frame (_steel_frame), the decks through it
        import bridgetypes
        pt = bridgetypes.paint(attrs.get("facade_color"))
        return Spec(ring=ring, floors=1, floor_h=h or BRIDGE_PIER_M, height=h or BRIDGE_PIER_M, roof="flat",
                    roof_mat=ROOF_FLAT, facade=f"{STEEL}|{pt}" if pt else STEEL, band=None, shop=False, plain=True,
                    seed=seed, min_h=0.0,
                    wall=None, windows=False, steeple=False, porch=False, cls=cls, deck=False, pitch=None)
    floor_h = FLOOR_M
    if industrial:
        floor_h = 4.5
    if church:
        floor_h = 7.5
    # (a big shed is a barn; a row of garages, a carport or a canopy is not)
    barn = cls in BARN_CLASSES or (shed and area >= 150 and cls not in ("garage", "garages", "carport", "roof", "shelter"))
    if barn:
        floor_h = 5.5 if area >= 150 else 3.5          # one tall storey; the gable above it
    if h is None and nf is None:
        if shed:
            nf = 1
        elif row:
            nf = 2 if seed < 0.25 else (3 if seed < 0.85 else 4)
        elif west_town and not industrial and cls != "apartments":
            # Bishop's and Moab's photos (2026-10-14): one-storey shops with false fronts, ranch houses
            nf = 1 if seed < WEST_TOWN_ONE_STOREY else 2
        elif cls in HOUSE_CLASSES or (area < 220 and not shop):
            nf = 1 if seed < 0.3 else 2
        elif industrial:
            nf = 2 if area > 6000 else 1
        elif cls == "apartments":
            nf = 3 + int(seed * 3)
        elif town:
            nf = 1 if seed < 0.6 else 2                 # small towns: mostly single-storey shops and offices
        elif area < 900:
            nf = 2 + int(seed * 2)
        else:
            nf = 2 + int(seed * 3)
    roof_given = (attrs.get("roof_shape") or "").lower()
    if roof_given in ("gabled", "hipped", "flat", "pyramidal", "half-hipped", "mansard", "gambrel", "skillion"):
        roof = roof_given
    elif roof_given in DOME_ROOFS:
        roof = roof_given
    elif roof_given in ("round", "saltbox"):
        roof = "gabled"
    else:
        detached_house = (cls in HOUSE_CLASSES or (area < 260 and not shop)) and not industrial and not row
        if church and neighbours == 0:
            roof = "gabled"
        elif neighbours >= 1:
            # attached: flat in the US (a pitched roof would show its gable to the neighbour); old European towns
            # are gabled (the ridge on the long axis: a narrow deep house shows its gable to the street)
            low = (nf or (h or 0) / FLOOR_M or 3) <= 6
            roof = "gabled" if (style.attached and low and not industrial and not shop
                                and _hash01(seed, "attached roof") < style.attached) else "flat"
        elif barn:
            roof = "gabled"
        elif shed and area < 80:
            roof = "gabled" if seed > 0.4 else "flat"
        elif detached_house:
            if style.flat and _hash01(seed, "flat roof") < style.flat:
                roof = "flat"
            else:
                roof = "gabled" if seed < 1.0 - style.hipped else "hipped"
        else:
            roof = "flat"
    # pitched roofs only on near-rectangular footprints (a church's nave keeps its gable beside a tower or an apse)
    # (a mapped dome follows its own outline: round parts and the largest domes keep it; a sliver or a dome over
    # DOME_MAX_HW is more likely a mapping slip: flat)
    if roof in DOME_ROOFS:
        if hw < 1.0 or hw > DOME_MAX_HW:
            roof = "flat"
    elif roof != "flat" and (box_fill < (0.6 if church else 0.8) or hw < 2.0
                             or hw > (18.0 if barn else 20.0 if church else 14.0)):
        roof = "flat"
    # a city tower's ground floor is shops or a lobby (Midtown, 2026-10-14: most towers have no class, and their
    # curtain wall came down to the pavement)
    tall = (nf or (h or 0) / FLOOR_M) >= TOWER_SHOP_FLOORS
    if dense and tall and cls not in HOUSE_CLASSES and not (shed or barn or garage or terminal or church):
        shop = True
    if church:
        shop = False                                # (no shopfronts on a church, Cluj 2026-10-15)
    if h is None:
        h = nf * floor_h + (GROUND_FLOOR_SHOP_M - floor_h if shop and nf > 1 else 0.0) + (0.6 if roof == "flat" else 0.2)
        if west_town and roof == "flat" and nf == 1 and area >= 80 and not (shed or barn):
            h += FALSE_FRONT_M                      # a main-street shop's tall parapet
    else:
        if roof in DOME_ROOFS:
            h = max(2.5, h - dome_rise(roof, hw, attrs.get("roof_height")))
        elif attrs.get("roof_height") and np.isfinite(attrs["roof_height"]) and roof != "flat":
            h = max(2.5, h - float(attrs["roof_height"]))
        elif roof != "flat":
            h = max(2.5, h - roof_rise(roof, hw, style.pitch or PITCH.get(roof, PITCH["gabled"])))
        if nf is None:
            # a church: one tall storey (Cluj's St. Michael's, 25 m, came out as three 8 m floors)
            nf = 1 if church else max(1, int(round(h / floor_h)))
    nf = max(1, min(nf, 120))
    floor_h = h / nf
    if shed or barn:
        facade, band, plain = SIDING, None, True
    else:
        # glass curtain walls on towers; the mid-rises of old city cores (and a share of the older towers) are
        # masonry with punched windows (Midtown's street photos, 2026-10-14: the theatre district's 5-15 storey
        # brick and stone buildings had all come out as glass towers)
        u_glass = _hash01(seed, "glass")
        # a building drawn as its parts: the style of the whole building on every part (NYC 2026-10-14: a brick
        # part on a glass tower, each part decided by its own height)
        bh = attrs.get("building_height")
        snf = max(nf, int(bh / FLOOR_M)) if bh is not None and np.isfinite(bh) else nf
        modern = cls in ("office", "commercial", "hotel") and u_glass < MIDRISE_GLASS_OFFICE
        tall = snf >= GLASS_FLOORS[1] or (snf >= GLASS_FLOORS[0] and (modern or u_glass < OLD_TOWER_GLASS)) \
            or (snf >= 7 and modern)
        if snf >= 7 and not tall and not industrial:
            if _hash01(seed, "brick") < brick_share(lat, lon) * 1.2:
                facade, band = BRICK_TINTS[int(seed * 7919) % len(BRICK_TINTS)], BAND_BRICK
            else:
                facade = FACADE_TINTS[int(seed * 7919) % len(FACADE_TINTS)]
                band = BAND_TALL if seed < 0.6 else BAND_SASH
        elif tall or (industrial and not shed):
            facade = f"{OFFICE}|{int(_hash01(seed, 'glass tint') * len(WALL_TINTS[OFFICE]))}" if tall \
                else FACADE_TINTS[3]
            band = (BAND_OFFICE if seed < 0.5 else BAND_OFFICE2) if tall else BAND_PANEL
        else:
            u_brick = _hash01(seed, "brick")
            detached = (cls in HOUSE_CLASSES or (area < 260 and not shop)) and not row and neighbours == 0
            brick = u_brick < brick_share(lat, lon) * (1.15 if row else 0.35 if detached else 1.0)
            if brick:
                band = BAND_BRICK
                facade = BRICK_TINTS[int(seed * 7919) % len(BRICK_TINTS)]
            else:
                facade = FACADE_TINTS[int(seed * 7919) % len(FACADE_TINTS)]
                if (row or cls in HOUSE_CLASSES) and _hash01(seed, "paint") < style.paint:
                    facade = PAINT_TINTS[int(_hash01(seed, "pc") * len(PAINT_TINTS))]
                if row or cls in HOUSE_CLASSES or roof != "flat":
                    band = BAND_SASH if seed < 0.7 else BAND_PANEL
                else:
                    band = (BAND_SASH, BAND_TALL, BAND_PANEL)[int(seed * 104729) % 3]
        plain = False
    wall = None
    house = (cls in HOUSE_CLASSES or (area < 260 and not shop and not industrial)) and not row and not shed \
        and neighbours == 0 and area < 300         # a long "residential" block is not a house (Fells Point)
    if house and band is not BAND_BRICK and nf <= 3:
        # detached houses: siding in the East and South, stucco in the West; windows laid over the wall
        u = _hash01(seed, "wall")
        kind, acc = None, 0.0
        for k, share in style.walls:            # (the US East: siding 75 %; the West: stucco 65 %, siding 25 %)
            acc += share
            if u < acc:
                kind = k
                break
        if kind is not None:
            tints = style.wall_tints
            if style.wood and kind == SIDING:   # chalets: stained wood
                wall = f"{SIDING}|w{int(_hash01(seed, 'wc') * len(WOOD_TINTS))}"
            elif tints is not None:
                tints = [t for t in tints if t < len(WALL_TINTS[kind]) - (BARN_REDS if kind == SIDING else 0)] or [0]
                wall = f"{kind}|{tints[int(_hash01(seed, 'wc') * len(tints))]}"
            else:
                n = len(WALL_TINTS[kind]) - (BARN_REDS if kind == SIDING else 0)     # houses: not barn red
                n = min(n, US_WALL_TINTS.get(kind, n))
                wall = f"{kind}|{int(_hash01(seed, 'wc') * n)}"
            facade = wall
    # a hut above the tree line, alone: stone walls (alpine: the tile is nearly empty and high, tile_buildings)
    stone_hut = wall is None and alpine and area < ALPINE_HUT_M2 and neighbours == 0
    if stone_hut:                       # (sheds too: a summit shelter is shed-sized)
        wall = f"{STONE}|{int(_hash01(seed, 'stc') * len(WALL_TINTS[STONE]))}"
        facade = wall
    # dressed stone: a regional share of low older buildings, attached ones too (their plaster sheet otherwise)
    if wall is None and style.stone and nf <= STONE_FLOORS and band is not BAND_BRICK \
            and not (shop or industrial or terminal or garage or shed or barn or church) \
            and cls not in ("apartments", "office", "commercial", "hotel", "retail") \
            and _hash01(seed, "stone") < style.stone:
        wall = f"{STONE}|{int(_hash01(seed, 'stc') * len(WALL_TINTS[STONE]))}"
        facade = wall
    # the mapped facade material (OSM building:material, Overture facade_material) over the guesses (Tower Bridge's
    # stone towers came out as glass towers, 2026-10-18)
    fm = str(attrs.get("facade_material") or "").lower()
    if fm in STONE_FACADES and not (barn or shed):
        wall = f"{STONE}|{int(_hash01(seed, 'stc') * len(WALL_TINTS[STONE]))}"
        facade, plain = wall, False
    elif fm == "brick" and not (barn or shed):
        facade, band, plain, wall = BRICK_TINTS[int(seed * 7919) % len(BRICK_TINTS)], BAND_BRICK, False, None
    elif fm == "glass":
        facade = f"{OFFICE}|{int(_hash01(seed, 'glass tint') * len(WALL_TINTS[OFFICE]))}"
        band, plain, wall = BAND_OFFICE, False, None
    elif fm in ("wood", "timber", "log", "wood_boards") and not (barn or shed):
        wall = f"{SIDING}|w{int(_hash01(seed, 'wc') * len(WOOD_TINTS))}"
        facade, plain = wall, False
    elif fm in ("concrete", "cement", "precast") and not (barn or shed):
        wall = f"{CONCRETE}|0"
        facade, plain = wall, False
    elif fm in ("plaster", "stucco", "render") and not (barn or shed):
        wall = f"{STUCCO}|{int(_hash01(seed, 'wc') * len(WALL_TINTS[STUCCO]))}"
        facade = wall
    elif fm in ("metal", "steel", "corrugated_iron", "aluminium") and not (barn or shed):
        wall = f"{METALWALL}|{int(_hash01(seed, 'wc') * len(WALL_TINTS[METALWALL]))}"
        facade, plain = wall, False
    fc = parse_color(attrs.get("facade_color"))
    if fc is not None and not attrs.get("bridge_part"):
        if wall is not None:
            base = wall.split("|")[0]
            wall = custom_look(base, fc)
            facade = wall
        elif fm == "brick":
            facade = custom_look(BRICK_WALL, fc)
            wall = facade
        elif fm == "glass":
            facade = custom_look(OFFICE, fc)
        else:
            facade = custom_look(FACADE, fc)
    if terminal:
        facade, band, plain = OFFICE, BAND_OFFICE, False
    elif garage:
        facade, band, plain = FACADE_TINTS[3], BAND_PANEL, False
    windows = True
    if not (shed or barn or row or house or terminal or garage or church) and nf <= 2 and roof == "flat" \
            and (area >= BIG_BOX_M2 or industrial) \
            and cls not in HOUSE_CLASSES and cls != "apartments":
        kind = METALWALL if industrial else PANEL
        wall = f"{kind}|{int(_hash01(seed, 'wc') * len(WALL_TINTS[kind]))}"
        facade, windows = wall, False
    rmat = str(attrs.get("roof_material") or "").lower()
    if rmat in ("metal", "tin", "steel", "zinc", "corrugated_iron", "sheet"):
        roof_mat = f"{ROOF_METAL}|{int(_hash01(seed, 'roof') * len(ROOF_TINTS[ROOF_METAL]))}"
    elif rmat == "copper":
        roof_mat = custom_look(ROOF_METAL, COLOR_NAMES["copper"])
    elif rmat in ("tile", "tiles", "roof_tiles", "clay", "terracotta"):
        roof_mat = f"{ROOF_SLATE}|{4 + int(_hash01(seed, 'roof') * 3)}"
    elif rmat in ("slate", "stone"):
        roof_mat = f"{ROOF_SLATE}|{int(_hash01(seed, 'roof') * 4)}"
    elif rmat in ("shingle", "shingles", "wood"):
        roof_mat = f"{ROOF_SHINGLE}|{int(_hash01(seed, 'roof') * len(ROOF_TINTS[ROOF_SHINGLE]))}"
    elif rmat in ("concrete", "cement", "tar_paper", "bitumen", "asphalt", "felt", "gravel"):
        roof_mat = f"{ROOF_FLAT}|{int(_hash01(seed, 'roof') * len(ROOF_TINTS[ROOF_FLAT]))}"
    elif rmat == "glass":
        roof_mat = f"{OFFICE}|0"
    elif roof == "flat" or barn or shed:
        roof_mat = ROOF_FLAT if roof == "flat" else ROOF_METAL
        roof_mat = f"{roof_mat}|{int(_hash01(seed, 'roof') * len(ROOF_TINTS[roof_mat]))}"
    else:       # the region's pitched roofs (the US: slate or shingle by the seed, their first four tints)
        mat, tints = style.roofs[min(int(seed * len(style.roofs)), len(style.roofs) - 1)]
        roof_mat = f"{mat}|{tints[int(_hash01(seed, 'roof') * len(tints))]}"
    rc = parse_color(attrs.get("roof_color"))
    if rc is not None:
        base_rmat = roof_mat.split("|")[0]
        roof_mat = custom_look(base_rmat, rc)
    if (barn or shed) and not stone_hut:    # plain walls tinted: barns the region's barn red or wood, sheds the siding colours
        tints = WALL_TINTS[SIDING]
        if barn and _hash01(seed, "barn") < style.barn_red:
            facade = f"{SIDING}|{len(tints) - 1 - int(_hash01(seed, 'red') * 2)}"
        elif style.wood:
            facade = f"{SIDING}|w{int(_hash01(seed, 'wc') * len(WOOD_TINTS))}"
        else:
            facade = f"{SIDING}|{int(_hash01(seed, 'sc') * (len(tints) - 2))}"
    # a front porch (the photos of Lancaster, Ocean City, Frederick, Moab: most houses have one): pitched-roof
    # houses, a regional share; doors() puts it on the front door's wall
    porch = (house and roof != "flat" and not barn and not shed and area < 300
             and _hash01(seed, "porch") < style.porch)
    if porch:   # pitched roofs draw the walls on the oriented box: an L-shaped house's door wall can be inside it
        _, _, hl, hw = oriented_box(ring)
        porch = area >= 0.9 * 4 * hl * hw
    spec = Spec(ring=ring, floors=nf, floor_h=floor_h, height=h, roof=roof, roof_mat=roof_mat, facade=facade,
                band=band, shop=shop and not shed and nf >= 1, plain=plain, seed=seed, min_h=0.0, wall=wall,
                windows=windows, steeple=church and area < 2500 and roof not in DOME_ROOFS, porch=porch, cls=cls,
                deck=garage and roof == "flat", us_signs=style.name.startswith("us_"),
                pitch=math.degrees(math.atan2(dome_rise(roof, hw, rh), max(float(hw), 0.1))) if roof in DOME_ROOFS
                else style.pitch)
    return francehouses.adapt(spec, attrs, lat, lon)


def dome_rise(roof, hw, roof_height=None):
    """m a dome / onion roof rises over its eaves: the mapped roof:height, else DOME_RISE x the half width."""
    if roof_height is not None and np.isfinite(roof_height) and roof_height > 0:
        return float(roof_height)
    return DOME_RISE.get(roof, 1.0) * float(hw)
