"""Road grading, base layer (split out of grading.py): the settings, the road plans' data (Plan, PlanPack, road ids)
and the pure profile functions (deck caps, abutment fills, underpass dips, approach ramps, grade_points).
"""
import logging
import math
import os
import numpy as np
import roads as roadsmod
# A single-span bridge and the roads under it solved as one height problem (crossings.py; BNE_INTERCHANGE=1, off by
# default): its ends raised, or the roads under it lowered, whichever costs less, from the ground they can be trusted
# for (not lidar's fill under the deck, not an abutment on its slope). Replaces the dip, its lidar exception and the
# ramps for those bridges (a motorway's too); the rules above decide for every other.
import crossings as _xs


log = logging.getLogger("earthd.grading")

VERSION = 2
PROFILE_STEP = 2.0          # m between profile samples
SIGMA_M = 12.0              # m, profile smoothing of built roads
# ... and of motorways and trunks: real ones keep long, gentle grades, and 12 m kept every bump of the elevation data in
# them where it is a 30 m surface model (BR-381 in Brazil: a junction's pad, forest edges; 20 % grades between them, a
# flat bench halfway down a long descent; GitHub #266). Lidar areas change little: their roads are smooth already.
SIGMA_MAJOR_M = float(os.environ.get("BNE_GRADE_MAJOR", "40"))
# (and primary roads: one through Russian forest kept every 1-3 m bump of GLO-30 over 50-100 m at 12 m, a road of hills
# and walls on flat land, GitHub #335. Not secondary yet: Otto Road's junction with I-80's ramps came out 0.29 m apart
# once its dip under I-80 (underpass_offsets) was laid on the longer smoothing; the ramps' pins need a look first)
MAJOR_CLASSES = ("motorway", "trunk", "primary")      # (not their slip roads: Overture's subclass "link", Spec.link)
DIRT_SIGMA_M = 8.0          # m, profile smoothing of gravel / dirt roads (the main ones: they stay smooth)
# Toned down 2026-11-06 after 1.8.2 (GitHub #783, #784: "like a motocross track", "easier to drive beside the road
# than on it"): as merged, a forest track had dips to 1.5 m and 66 % grades between samples, a minor dirt road 65 cm
# dips, and every dirt road leaned up to 20 % sideways. Now a worn country road everyday cars drive: potholes of 5-12
# cm, a step of at most 15 cm per 2 m, a camber of a few percent, the elevation data's own noise smoothed over 6 m.
# a minor dirt road follows the elevation data closely instead, so it is bumpy and cannot be taken flat out
DIRT_SIGMA_MINOR_M = float(os.environ.get("BNE_DIRT_SIGMA", "6.0"))   # (5 followed the DEM's bumps: GitHub #414)
# ... where the elevation is only the coarse global model (dem.coarse_at: 30 m cells, 1-3 m of noise per cell), 6 m
# follows that noise, not the ground: a dirt road there is smoothed like an unclassified road (its potholes and
# patches, dirt_bumps, are added after and stay). GitHub #835: Fermanagh's gravel roads "undriveable at speed"
DIRT_SIGMA_COARSE_M = float(os.environ.get("BNE_DIRT_SIGMA_COARSE", "16.0"))
# longer smoothing for the mid classes too (mpsteppe's PR #289: real roads have long vertical curves, the larger the
# longer; 12 m left bumps the roads do not have); MAJOR_CLASSES keep SIGMA_MAJOR_M
CLASS_SIGMA_M = {"secondary": 25.0, "tertiary": 20.0, "unclassified": 16.0, "residential": 14.0}
# A running median before the smoothing (median_profile): a bridge deck left in the elevation data where it crosses
# over a road is a spike a few samples wide, and the Gaussian spread it into a hump in the road under (GitHub #271, an
# 8.5 m spike near Thetford: a 3.2 m hump). Slopes, crests and steps longer than the window stay.
MEDIAN_M = float(os.environ.get("BNE_GRADE_MEDIAN", "24"))
DIRT_MEDIAN_M = 10.0        # (tracks keep their small real bumps)
RAIL_GRADE_ON = os.environ.get("BNE_RAIL_GRADE", "1") != "0"    # the ground laid to the railways' bed (railways.py)
RAIL_TAG = "rail:"
SHOULDER_M = 1.0            # m of flat shoulder beyond the carriageway, each side
MOUNTAIN_SHOULDER_M = 0.5   # ... beside a mountain pass road (roads._mountain_roads)
# A mountain pass road's sides (Plan.steep): back to the natural ground at 1:STEEP_RUN over STEEP_T_MIN m at the least,
# not SLOPE_RUN / T_MIN's long fill and cut slopes (on the Stelvio they made a 20 m bench of a 6 m road between its wall
# and its parapet, the parapet's drop never within its reach: roads.PARAPET_*)
LINK_TURN_DEG = 20.0        # (PlanPack: two plans' ends at one point within this of straight on: one road going on)
EXTRAP_M = float(os.environ.get("BNE_GRADE_EXTRAP", "4"))             # m a segment's line is carried on past an end where its plan goes on (grade_points: no
                            # washboard of the vertex step on a grade)
STEEP_RUN = 0.5
STEEP_T_MIN = 1.5
SIDEWALK_MARGIN_M = 0.5     # m of flat ground beyond a sidewalk (roads.SIDEWALK_CLASSES)
LONG_PART_M = 3000.0        # a road part longer than this is planned per cell, only within CLIP_MARGIN_M of the cell
CLIP_MARGIN_M = 400.0
PLAN_THREADS = 4            # cells planned at once (_plan_missing)
PLAN_SETS_MAX = 16          # cell sets whose merged plans are kept (Grader._plan_set; ~1 MB each in a town)
SPREAD_CELLS = 16           # plans(): more map cells in the points' bbox: only the cells around the points
T_MIN, T_MAX = 5.0, 12.0    # m, transition back to the natural ground (5: roads close together meet gently)
SLOPE_RUN = 2.0             # transition = SLOPE_RUN x the height difference (1:2 slope)
VERTEX_STEP = 4.0           # m between the plan's vertices (segments for the grading pass; linear between: at 16 m a
                            # crest or a sag was a kink every 16 m, 'bumpy roads, worst uphill and at speed' on r/BeamNG,
                            # 2026-10-17: a San Francisco street's slope changed 4.4 % rms per 2 m, 1.2 % at 4 m;
                            # grading a 1 km block 0.06 -> 0.14 s)
BUCKET = 32.0               # m, query-point buckets
ROAD_LIFT = 0.04            # m the physics ground is raised under roads (see Grader.apply)
BLEND_R = 4.0               # m: roads near each other blend their targets by 1 / (1 + (d / BLEND_R)^2)
PROFILE_ZOOM = int(os.environ.get("BNE_DEM_ZOOM", "16"))   # DEM zoom of the profiles (one zoom for every caller)
WALL_CUT_M = 2.0            # a road cut deeper than this into the ground: a retaining wall at the core's edge
WALL_T_M = 0.8              # (roadmesh.retaining_walls) and the ground at its natural height right behind it, over
                            # this; on a 1:2 slope Central Park's sunken transverse was a grassy ditch between its
                            # stone walls (the user, 2026-10-17)
WALL_FLAT_M = 1.0           # ... after one terrain grid step (1 m) at the road's level behind the wall: rising at the
                            # wall itself, the grid's triangles stood as dark wedges in front of its face; the wall's
                            # top (roadmesh.WALL_THICK_M) covers the strip
TUNNEL_TAG = ":tun"          # in the fid of the tunnels' own plans (open sections, mouths): apply(roads_only) skips them
MOUTH_PLAN_M = 4.0          # m into a tunnel from a portal its floor is a grading plan too (Grader._mouth_plans)
TUNNEL_KEEP_IN_M = 3.0      # (roads.LOW_HOLE_M: a low arch's mouths, flat at its floor)
TUNNEL_KEEP_M = 5.0         # m beside a covered tunnel's road where the ground over it is left ungraded ...
OVER_TUNNEL_SMALL_M = 0.5   # ... but for a change smaller than this (a road crossing over it, graded flush)
WALL_AHEAD_M = 2.5          # ... only beside a segment or this little past its end (a joint between two: VERTEX_STEP)
YIELD_RAMP_M = 0.5          # m over which a road gives way to another's carriageway at another level (grade_points): in
                            # height difference up to WALL_CUT_M and in distance past its own carriageway's edge
# Roads at other heights beside a road's core (GitHub #92: a street in a cutting had its half under a service road on
# the bank 7.5 m above it pulled 2-4 m up its slope; hairpins blended into each other; an upper road's edge dragged into
# a pit by the road under it). Beside a point in a core, another road's segments outside their own core count less the
# further their height is from that core's: not at all from SEP_FULL_M, as before up to SEP_FROM_M, and their weight
# comes back over CORE_FADE_M past the core (no step at its edge).
SEP_ON = os.environ.get("BNE_GRADE_SEP", "on") != "off"
# Sharp cuttings (the user, 2026-10-21: the elevation data is smoothed, "sharper terrain features get lost": a street in
# a walled cutting read as a 4 m slope its asphalt climbed, McGraw-Graben, Munich). Beside a built road whose ground
# rises SHARP_MIN_M or more within SHARP_REACH_M of its core at SHARP_GRADE or steeper on average (a wall the data
# smoothed into a ramp), the ground is raised to that top from the core's edge out: the road's wall rule (WALL_CUT_M)
# then flattens a strip, steps up and roads.retaining_walls stands a wall there.
SHARP_ON = os.environ.get("BNE_SHARP", "on") != "off"
SHARP_REACH_M = 10.0
SHARP_STEP_M = 1.0
SHARP_MIN_M = 2.0
SHARP_GRADE = 0.6
SHARP_RUN_N = 3             # consecutive plan vertices (VERTEX_STEP apart) a cutting holds for at least
# A cutting under this deep is a wall only in a trench (a cutting on both sides: Munich's McGraw-Graben, Central Park's
# transverses); a lone bank of a metre or two beside a road is a graded slope, as real ones are (a 1.5 m concrete wall
# stood in the flat desert beside US-163 in Utah, where there is a low eroded rock bank: Mapillary, 2026-10-29).
# Hillside roads' deeper cuttings keep their walls (WALL_MAX_H_M high at most)
SHARP_LONE_MIN_M = 4.0
SEP_FROM_M = 2.0
SEP_FULL_M = 4.0
CORE_FADE_M = 2.0
SIDE_FADE_M = 2.0           # m of (side - ahead) over which a segment counts as beside the point (ahead: its own road)
RETRY_S = 60.0              # s before a cell whose road data failed is asked for again
PLAN_SLOW_S = 60.0          # s: a cell's plans still being made this long: its process's stacks to the log (stackdump)

# Junctions: a side road's end takes the height of the road it joins (snap_junctions; mpsteppe's PR #290)
JUNCTION_SNAP_ON = os.environ.get("BNE_JUNCTION_SNAP", "1") != "0"
JUNCTION_RANK = {"motorway": 9, "trunk": 8, "primary": 7, "secondary": 6, "tertiary": 5, "unclassified": 4,
                 "residential": 3, "living_street": 2, "service": 1}
JUNCTION_REACH_M = 3.0      # m beyond the joined road's carriageway edge a side road's end still counts as joining it
JUNCTION_END_M = 10.0       # m: a same-class road ending this close to the other's end continues it (no snap)
JUNCTION_TAPER_M = 40.0     # m along the side road the height change is eased in over (half the road when shorter)
JUNCTION_MAX_M = 6.0        # m: a bigger difference is left as it was (bad data; Perm, GitHub #414: 4-5 m steps where a
                            # 30 m DEM smeared an embankment into the junctions)
JUNCTION_HOLD_PAD_M = 2.0   # m beyond the joined road's carriageway a side road keeps its height (_hold_junction_corridors)
JUNCTION_HOLD_GRADE = 0.12  # ... and goes back to its own profile at most this steep
# A road's change at a junction eased back to its own profile no longer than it must be (junction_pads): the cosine adds
# at most JUNCTION_EASE_GRADE to the road's own grade (over 40 m whatever the step, a track leaving up a 30 % bank ran
# in a trench 1-2.4 m deep for 30 m, the user 2026-10-29 at -32.0951, 116.03866: "a closer and a bit more aggressive"
# curve), JUNCTION_EASE_MIN_M at least (a small step over a short ease: a kink). (The side road's landing on the joined
# road's surface, and equal-ranked T junctions, are the pads': snap_junctions, in Rust since 1.7.3, is left as it was.)
JUNCTION_EASE_GRADE = float(os.environ.get("BNE_JUNCTION_EASE_GRADE", "0.04"))
JUNCTION_EASE_MINOR_GRADE = 0.15    # ... on a track, a driveway, a service road (pad_class_w under PAD_LANDING_MIN_W)
JUNCTION_EASE_MIN_M = 30.0
JUNCTION_EASE_MINOR_MIN_M = 8.0


def ease_length(step, minor=False):
    """m a height step (m) is eased out over (cosine) along a road: its added grade at most JUNCTION_EASE_GRADE
    (JUNCTION_EASE_MINOR_GRADE on a minor road), from JUNCTION_EASE_MIN_M (_MINOR_) to JUNCTION_TAPER_M. Pure."""
    g = JUNCTION_EASE_MINOR_GRADE if minor else JUNCTION_EASE_GRADE
    lo = JUNCTION_EASE_MINOR_MIN_M if minor else JUNCTION_EASE_MIN_M
    return float(np.clip(abs(step) * math.pi / (2.0 * max(g, 1e-6)), lo, JUNCTION_TAPER_M))


# Roads side by side where the elevation is only the coarse global model (dem.coarse_at: no lidar; GitHub #414, Perm on
# Copernicus' 30 m: service roads 7 m beside the primary stood 4-5 m over it, where Mapillary shows them at grade): a
# lower-ranked road within COARSE_REACH_M of a higher-ranked one's edge eased to its height (equal ranks halfway), less a
# COARSE_MAX_SLOPE cross slope between their edges (a real hillside keeps its fall); over COARSE_MAX_STEP_M apart they
# are taken as truly separated. Rust: earthfast.ease_coarse.
# (off by default: at Perm the trench turned out to be the underpass dip, which this does not touch; an A/B across
# coarse-elevation places comes first, BNE_COARSE_EASE=1 turns it on)
COARSE_EASE_ON = os.environ.get("BNE_COARSE_EASE", "0") == "1"
COARSE_REACH_M = 20.0
COARSE_MAX_STEP_M = 6.0
COARSE_MAX_SLOPE = 0.25
COARSE_SIGMA_M = 8.0
COARSE_SIDE_M = 12.0        # m: plan_roads' side samples on coarse elevation (the median of three)
# Neighbouring roads coupled (the user, 2026-10-29: "the current model sinks or lifts roads too much, where perhaps the
# roads need to be on a similar level"): two at-grade roads side by side are each sampled and smoothed on their own, and
# the coarse model's error does not cancel between them: a frontage road 18 m beside a primary ended up 2-3x as far above
# or below it as the ground truly is, with steep connectors between. On coarse elevation (dem.coarse_at), where a road
# runs alongside a higher- (or equal-, half way) ranked one, within NEIGHBOUR_GAP_M of its edge (fully within
# NEIGHBOUR_FULL_M) and within NEIGHBOUR_PARALLEL_DEG of its direction, it takes that road's height plus the ground's own
# fall between them: the difference of a far smoother surface (NEIGHBOUR_TREND_SIGMA_M: a hillside's grade, not the
# data's noise). A difference over NEIGHBOUR_MAX_M is left: a real separation. Roads meeting at a node are left to
# snap_junctions. Before the junctions are snapped (connectors meet the coupled heights) and the approaches lifted and
# the underpasses dipped (bridges and their roads keep their separation). BNE_NEIGHBOURS=1: on. (Off by default, as the
# pads and the rail check: the three were ~20 % of a Midtown road tile set's build in Python, 2026-11-03, PR #482/#671.)
NEIGHBOUR_ON = os.environ.get("BNE_NEIGHBOURS", "0") == "1"
NEIGHBOUR_GAP_M = 20.0
NEIGHBOUR_FULL_M = 12.0
NEIGHBOUR_PARALLEL_DEG = 30.0
NEIGHBOUR_MAX_M = 4.0
NEIGHBOUR_TREND_SIGMA_M = 45.0
NEIGHBOUR_SMOOTH_M = 10.0       # m: the change smoothed along the road: no step where the neighbour ends


# Overpass approach ramps (from KyzoFox's 1.5.4, PR #64). A short bridge over a road had its ends on the ground on both
# sides (the DEM is too coarse for the real approach embankments), and roads.deck_profile_once caps how steeply a deck
# climbs from its ends: the deck over I-20 near Temple, GA reached ~2.5 m over the lanes (trucks hit it, 2026-10-19).
# The roads leading onto such a bridge are now graded up to the height the deck needs, over a smooth ramp, as real
# approach fills are: the bridge's ends (sampled from this graded ground) then sit high enough.
APPROACH_ON = os.environ.get("BNE_APPROACH_RAMPS", "on") != "off"
APPROACH_GRADE = float(os.environ.get("BNE_APPROACH_GRADE", "0.05"))   # steepest grade of a ramp (its middle)
APPROACH_MAX_LIFT = 10.0    # m an abutment is raised at most
APPROACH_MIN_LIFT = 0.5     # m: less is left as it is
APPROACH_MIN_RUN = 30.0     # m: the shortest ramp
APPROACH_MAX_RUN = 320.0    # m: the longest ramp
APPROACH_TOUCH_M = 2.5      # m: a road whose end is this close to an abutment leads onto the bridge
APPROACH_MIN_CROSS_M = 6.0  # m: a road crossing a deck closer than this to its end is a junction at grade (not
                            # under it)
APPROACH_FLAT_M = 20.0      # m from an abutment the deck needs its full clearance (beyond: it may fall at DECK_GRADE)
APPROACH_MARGIN_M = 0.3     # m over the clearance (the deck profile's smoothing takes a little off near its ends)
APPROACH_SKIP = ("motorway", "trunk")   # never raised (their links are: _never_raised); the roads over them rise
WATER_LIFT_ON = os.environ.get("BNE_WATER_FILL", "on") != "off"
WATER_CURVE_M = 16.0        # m from an abutment a deck over water spends on its vertical curve before climbing ...
WATER_MAX_LIFT = 2.0        # ... and its approach fill is raised at most this (any class: a motorway's too)
OBS_HALF_M = 4.0            # m: the half width a bridge line is matched with when its abutment's lift is measured
OBS_END_M = 12.0            # m from the end the measured deck's height is taken at most (its first measured station)
APPROACH_GAP_M = 1.0        # m of flat ground kept between a ramp's fill slope and the road beside it
ENU_CACHE_MAX = 512         # cells' road parts in the anchor frame kept (Grader._cell_enu; ~0.02-0.5 MB each)
APPROACH_CELL_M = 450.0     # m: a plan's end this far from a cell cannot raise what that cell's queries see
# A ramp the lidar measured (its deck end's height, roadz) rises that high beside a road at grade on a steep side (the
# lower road then cut into the raised ground, its retaining wall: WALL_CUT_M), as a real approach's walls stand: with
# the 1:2 fill slope it was capped to 0.7 m of a measured 3.3 m 27 m off the road beside it (Springfield's Old Keene
# Mill Road, approachlidar.py, 2026-10-31)
MEASURED_SIDE_RUN = 0.5
APPROACH_LIDAR_ON = os.environ.get("BNE_APPROACH_LIDAR", "on") != "off"   # (roadz APPROACH_*: Grader._obs)
TWIN_ON = os.environ.get("BNE_APPROACH_TWINS", "on") != "off"
TWIN_M = 60.0               # m: another abutment this near whose approach runs beside a ramp ...
TWIN_SHARE = 0.5            # ... and is raised at least this share as much rises with it (no cap between them)
APPROACH_CAP_GRADE = 0.10   # steepest grade a ramp falls at where a road beside it caps it (a cap alone dropped 3 m
                            # in 8 m by a side road north of the I-20 bridge near Temple, GA, 2026-10-20)

# Embankments at abutments (GitHub #38: a GPS drive near Presov, Slovakia flipped on the D1). A bridge's map line can end
# down its embankment, short of the abutment: the road's last metres followed the valley's side in the elevation data and
# the deck end, taken from that ground, sat 1.4 m under the road 2 m on (a 12 % deck and a step at 55 mph). A road ending
# at a bridge whose last ABUT_NEAR_M fall much more steeply toward it than its own grade further on keeps that grade to
# its end instead, as the fill under a real approach does (every class, motorways too).
ABUT_FILL_ON = os.environ.get("BNE_ABUT_FILL", "on") != "off"
# A road split at a junction (the map's segments end at every one) is smoothed as one: each part's profile also samples
# the ground along the straightest part that goes on from its end, so the parts meet at one height. Each was smoothed
# alone, and on a slope each end took the average of its own side: on BR-381's 20 % grades the next part began 2 m
# over where the last ended, a ramp at every junction (GitHub #266).
CHAIN_ON = os.environ.get("BNE_GRADE_CHAIN", "on") != "off"
CHAIN_JOIN_M = 1.0          # m: two parts' ends this close are one node
CHAIN_TURN_DEG = 35.0       # the part that goes on turns at most this at the node
CHAIN_LOOK_M = 10.0         # m along each part its direction at the node is taken over
# Underpasses (GitHub #138, #142, #126's rule: "the road under gives way"): a ground road crossing under a short
# overpass that is never raised (APPROACH_SKIP classes and links: a motorway's bridge over a local road) dips to the
# deck's height (its abutments' ground, interpolated) minus UNDER_CLEAR_M, falling back at UNDER_GRADE. Lidar terrain
# removes bridges and interpolates across the gap, at about the deck's height: the road under ran 0.7-2.5 m under the
# girders (A28 near Zwolle, I-84 over NE 181st Ave in Portland, I-395 in Springfield VA, 2026-10-22). Never raises.
UNDER_ON = os.environ.get("BNE_UNDERPASS", "on") != "off"
# Rounded junction corners (roadcorners.py, from PR #363): the fan of asphalt between a curbless corner's arc and its
# roads' old square corner graded at the roads' heights (Grader._make_cell)
CORNERS_ON = os.environ.get("BNE_GRADE_CORNERS", "on") != "off"
UNGRADED = {"raceway"}      # classes planned by nobody: the terrain is their surface (an oval's banking, GitHub #143) ...
UNGRADED_MARGIN_M = 2.0     # ... and nothing else grades their carriageway (+ this: kerbs, apron): a road crossing a
                            # track, or a tunnel's open approach under one, cut a flat trench through Daytona's banking
# A raceway's surface (GitHub #833: a GT3 flat out through a Nordschleife kink was launched by the elevation data's
# bumps): the terrain still, so the banking stays (Daytona's 31 degrees), but smoothed along the track: the ground's
# along-track Gaussian at each offset across, on a grid of RACEWAY_STEP_M along by RACEWAY_ACROSS_M across (the surface
# itself, not a correction to the point's ground: one of 2 m steps under-shot 6 m bumps by half); each map segment's own, the line continued RACEWAY_EXTEND sigmas past its ends (straight on,
# or round a closed loop) so its ends are smoothed too and every segment's is a pure function of its line and the DEM.
# It fades out over UNGRADED_MARGIN_M (kerbs, apron) to the plain ground. BNE_RACEWAY_SMOOTH=off: the ground as it is.
RACEWAY_SMOOTH_ON = os.environ.get("BNE_RACEWAY_SMOOTH", "on") != "off"
RACEWAY_SIGMA_M = float(os.environ.get("BNE_RACEWAY_SIGMA", "10.0"))
RACEWAY_STEP_M = 2.0
RACEWAY_ACROSS_M = 1.0
RACEWAY_EXTEND = 3.0
# The same structural headroom as the deck solver, including the lower road's
# physical surface lift. A stale 6.7 m target forced short motorway decks into
# clearance humps even after their underpass had been graded (Otto, Wyoming).
UNDER_CLEAR_M = roadsmod.DECK_ROAD_CLEAR + ROAD_LIFT
UNDER_GRADE = 0.06          # the dip's ramps
UNDER_MAX_BRIDGE_M = 120.0  # only single short spans (both ends on the ground): a chain's inner ends are no abutments
UNDER_MAX_DROP_M = 8.0      # never lowered more than this (bad data: rather a low bridge than a pit)
UNDER_FLAT_M = 12.0         # m either side of the crossing kept at the dip's full depth (under_dip)
UNDER_MAX_GRADE = 0.10      # a dip's ramp made steeper, up to this, to end inside its road part (under_dip)
UNDER_JOINT_FLAT_M = 12.0   # at a junction, every road takes the deepest dip within this of it ...
UNDER_JOINT_FADE_M = 20.0   # ... blended back to its own over this
# m: road ends this close are one node of the dip's network (underpass_offsets). At 0.5 m a trunk road whose end the map
# left 1.7 m short of its junction (Tonkin Highway at Morley Drive, Perth, GitHub #915) was not lowered with the
# junction's other roads, 6.1 m down for the underpass: more than snap_junctions takes (JUNCTION_MAX_M), so it stood
# as a 5 m step in 8 m (a 91 % grade) where it met them. Within a lane's width, as the junctions' own reach.
UNDER_NODE_M = float(os.environ.get("BNE_UNDER_NODE", "2.0"))
# A dip never takes a road under the sea's surface (+ UNDER_SEA_M): the sea's WaterBlock tiles flood their whole square
# (seatiles.py), so on low ground by the sea or a tidal river the dug road lay under water (Perth's Mitchell and Kwinana
# Freeways by the Swan River, GitHub #515, #518: ground 1-3 m up, dug up to UNDER_MAX_DROP_M; rebuilt offline
# 2026-10-30, 363 and 703 points of road on dry ground dug to -6.0 m and -5.4 m, none after). Held at every point of the road, not only at the crossing: a crossing on a bank under UNDER_SEA_M
# dug its flat and ramps 2-4 m under the sea on the dry ground beside (#515's interchange, rebuilt offline 2026-10-30).
# The deck keeps less headroom there instead (as UNDER_MAX_DROP_M: rather a low bridge than a flooded road); a road
# already under that is not lowered. A polder's (under UNDER_POLDER_M) dips as before: the tiles split away from it.
UNDER_SEA_ON = os.environ.get("BNE_UNDER_SEA", "on") != "off"
UNDER_SEA_M = 0.5           # m over the sea surface (seatiles.SEA_E) a dip's floor stays (the tiles' curve: <= 0.17 m)
UNDER_POLDER_M = -1.0       # m: a road this far under the sea is behind dikes (the Netherlands' polders: 2-6 m under)
# Where the elevation is lidar, a deck (its ends' ground, as raised) standing less than this over the road crossing
# under it is not at deck height in the data: the lidar has the road's own ground there, and the road keeps it (the deck
# rises instead). Sassafras Way runs on level ground under Pittsburgh's 28th Street Bridge (Street View), the bridge's
# mapped end 0.4 m under it in the lidar; dug 8 m to clear that, its dip ran on through 28th Street and lowered Liberty
# Avenue 3 m into a walled trench (the user's report, 2026-10-29). (Not on the coarse models: no cutting there, BR-381.)
UNDER_LIDAR_ON = os.environ.get("BNE_UNDER_LIDAR", "on") != "off"
UNDER_LIDAR_DECK_M = roadsmod.AT_GRADE_M
# A road's ground where a deck crosses over it is read from beside the deck (deck_capped): the elevation under a deck is
# the deck, not the road. Lidar keeps some short decks as ground (Tartu mnt's two 18-21 m decks over Järve tee at Assaku,
# Estonia, 59.3788 N 24.8441 E: a 6 m hump 40 m long in the road under them) and fills a removed bridge's gap at about
# the deck's height (the A28 / I-84 cases above). Järve tee's plan climbed the hump, and the underpass dip, measured from
# it, lowered the road and the streets joined to it 5.5 m under their ground for 560 m (GitHub #685). Under a deck the
# road's samples are capped at the straight line between its ground UNDER_DECK_SIDE_M past the deck's sides, in its plan
# (plan_roads) and its underpass depth (_under_ground) alike: never raised (a real dip under a bridge stays). Not on the
# coarse models (their smeared embankments are wider than that: UNDER_RUN_*).
UNDER_DECK_ON = os.environ.get("BNE_UNDER_DECK_GROUND", "on") != "off"
UNDER_DECK_SIDE_M = 15.0    # m past a deck's side the road's ground is read (Assaku's hump: 6-13 m past its decks)
# A walkway bridge is not a road, so it never becomes a deck. The elevation still has it: a footbridge across a
# motorway cut is a ridge of ground in the carriageway (a catwalk over I-87 at Lake George, GitHub #802). Its span
# caps the road under it the same way a road deck does, and nothing else (no ramp, no abutment).
WALK_SPAN_CLASSES = {"footway", "path", "pedestrian", "cycleway", "bridleway", "steps"}
WALK_SPAN_HALF_M = 4.0      # m: the walkway, wide enough that the side read covers the ridge the elevation leaves
UNDER_DECK_MAX_M = 30.0     # m along the road a deck covers at most either side of the crossing (an oblique one)
UNDER_DECK_PAD_M = 3.0 * SIGMA_MAJOR_M + PROFILE_STEP + UNDER_DECK_SIDE_M + UNDER_DECK_MAX_M    # m round a cell: the
                            # farthest its plans' samples reach past its box (their extensions), and a deck's window
# m around a cell's plans whose roads and spans its dips are solved on: the farthest a dip or a loose end reaches
# (underpass_offsets: flat + the longest ramp, a junction's share) and a span's length; every cell sees the same
UNDER_NEAR_M = 4 * VERTEX_STEP  # m past a cell's box its plans' dips are made (to their pieces' cut ends there)
UNDER_REACH_M = UNDER_FLAT_M + UNDER_MAX_DROP_M * math.pi / UNDER_GRADE + 2 * UNDER_JOINT_FLAT_M \
    + UNDER_JOINT_FADE_M + 10.0
# ... and under any other short overpass, by what its ramps could not raise it (BR-381 near Pouso Alegre: a local
# road's bridge whose ramps the frontage roads at both ends held down sat ~1 m over the motorway; the 30 m elevation
# data has no cutting under it, and the AI car drove into its abutment wall, 2026-10-25)
UNDER_ANY_ON = os.environ.get("BNE_UNDERPASS_ANY", "on") != "off"
# A single-span bridge and the roads under it solved as one height problem (crossings.py; BNE_INTERCHANGE=1, off by
# default): its ends raised, or the roads under it lowered, whichever costs less, from the ground they can be trusted
# for (not lidar's fill under the deck, not an abutment on its slope). Replaces the dip, its lidar exception and the
# ramps for those bridges (a motorway's too); the rules above decide for every other.
CW_ON = _xs.ON and os.environ.get("BNE_CARRIAGEWAYS", "on") != "off"          # carriageways.py: a motorway's roads and decks as one profile (with BNE_INTERCHANGE)
CW_MAX_M = 8000.0       # m: a carriageway chain is cut here (a branch point or a loose end comes long before)
CW_HEADROOM_M = roadsmod.MIN_BRIDGE_HEADROOM_M + roadsmod.BRIDGE_HEADROOM_MARGIN_M   # m of headroom the carriageways
                        # keep over one another, the roads and the tracks under them (+ the deck's depth): the decks' own
                        # (roads.deck_road_clear), so their girders clear what they span
CW_NONFAM_HI = False    # other roads' decks over a carriageway bound it: off (their decks rise over it as before; from their
                        # ends' ground a deck read low and dipped the Mitchell Freeway ~1 m under each street bridge)
CW_MIN_SIN = 0.3        # two carriageways crossing at a flatter angle are not a grade separation (lines along each other)
CW_STACK_SIN = 0.03      # decks at different levels crossing more steeply than this (2 degrees) are separated
CW_OVERLAP_MAX_M = 120.0  # m either side of a crossing two decks' clearance is held along at most
CW_OVERLAP_STEP_M = 12.0  # m between the points two overlapping decks' clearance is held at


def _merge_span_chains(parts):
    """parts [(line (k, 2+), fixed, rail)] with every simple chain of them joined into one line: parts of the same
    (fixed, rail) whose ends meet within UNDER_CHAIN_JOIN_M at a node no other part's end reaches. Pure."""
    f64 = np.float64
    n = len(parts)
    if n < 2:
        return list(parts)
    ends = np.array([[np.asarray(b, f64)[0, :2], np.asarray(b, f64)[-1, :2]] for b, _f, _r in parts]).reshape(-1, 2)
    d = np.hypot(ends[:, None, 0] - ends[None, :, 0], ends[:, None, 1] - ends[None, :, 1])
    near = d < UNDER_CHAIN_JOIN_M
    np.fill_diagonal(near, False)
    link = {}                               # end index (2 * part + 0 / 1) -> the other part's end joined there
    for e in range(2 * n):
        others = np.flatnonzero(near[e])
        if len(others) == 1 and others[0] // 2 != e // 2:
            o = int(others[0])
            if parts[e // 2][1:] == parts[o // 2][1:] and np.flatnonzero(near[o]).tolist() == [e]:
                link[e] = o
    out, used = [], set()
    for start in range(n):
        if start in used:
            continue
        # walk back to the chain's first part (or round a ring once)
        p, e_in = start, 0
        seen = {start}
        while 2 * p + e_in in link:
            o = link[2 * p + e_in]
            q = o // 2
            if q in seen:
                break
            seen.add(q)
            p, e_in = q, 1 - (o % 2)
        # and forward from it, joining
        line = np.asarray(parts[p][0], f64)
        if e_in == 1:
            line = line[::-1]
        used.add(p)
        e_out = 2 * p + (1 - e_in)
        while e_out in link:
            o = link[e_out]
            q = o // 2
            if q in used:
                break
            nxt = np.asarray(parts[q][0], f64)
            nxt = nxt if o % 2 == 0 else nxt[::-1]
            w = min(line.shape[1], nxt.shape[1])
            line = np.concatenate([line[:, :w], nxt[1:, :w]])
            used.add(q)
            e_out = 2 * q + (1 - o % 2)
        out.append((line, parts[p][1], parts[p][2]))
    return out


def _cw_level(q):
    """A carriageway part's level (its OSM layer; 1 for a bridge without one, 0 on the ground)."""
    return float(q[5]) if len(q) > 5 else (1.0 if q[2] else 0.0)


def _cw_half(q):
    return float(q[6]) if len(q) > 6 else 5.0
CW_MAX_DIG_M = 2.5      # m a carriageway is lowered at most under another road's deck (the deck is low for the rest)
CW_UNDER_M = 10.0       # m from a deck's line the ground under it is not trusted, rising to trusted at twice that
CW_MARGIN_M = 1200.0    # m past its home cell a cluster's chains reach (the farthest barely move the cell's)
CW_CLUSTER_M = 30000.0  # m of carriageways solved together at most (the far ones barely move the near)
XS_ON = _xs.ON

XS_FIXED_GRADE = 0.03   # a solved motorway's or trunk's ramps rise at this (_lift_approaches)
XS_FLOOR_M = 150.0      # m round a solved crossing no road is dug under its target (_lower_underpasses)
# ... but on coarse elevation (dem.coarse_at: the 30 m global model) a street's dip stays on the street: an overpass's
# embankments are smeared away there and its ramps, held down by the streets beside them, leave the whole clearance to
# the road under; spread along every street joined to it (underpass_offsets: by distance, at half grade), the 7.2 m dip
# under Ташкентская улица's bridge sank Демократическая улица and the streets round it 4-6 m for ~200 m while the ground
# between them stayed: walled trenches, humps of untouched ground between the streets and side roads standing as walls
# (Samara, 2026-11-02; the same trench Perm's rail bridge dug, GitHub #414, where railways.py now raises the track).
# There the dip runs along the street's own pieces (street_runs: carried on through every node where only its next
# piece joins it) and is back up by its junctions at up to UNDER_MAX_GRADE, as deep as fits: rather a lower bridge than
# a sunken neighbourhood (Samara: 3.2-3.8 m under the girders, 6-7 m before). A motorway's or trunk's dip (no
# junctions at grade: APPROACH_SKIP) spreads as before, and so does one whose street has room for less than
# UNDER_RUN_MIN_M (a junction right by the bridge is part of the underpass) or is a ring.
UNDER_RUN_ON = os.environ.get("BNE_UNDERPASS_RUN", "on") != "off"
# A short overpass mapped as several parts end to end (a 3 m stub split off at its end, two ways meeting mid-span) is one
# span for the underpass: each part had another joined to it, so none was a single span and the road under kept its
# ground: the M1's bridges near Dundalk 3.3-4 m over the motorway (GitHub #960), an overpass at the trunk's height by
# Orte (#967). Parts joined at a node where only they meet (no branch), the same road or rail kind, are one line here.
UNDER_CHAIN_ON = os.environ.get("BNE_UNDERPASS_CHAIN", "on") != "off"
UNDER_CHAIN_JOIN_M = 0.75
UNDER_CHAIN_MAX_M = UNDER_MAX_BRIDGE_M   # m: such a chain spans this at most (Orte's two parts, 194 m: their dip, 8 m
                            # deep, ran into the trunk's next structure as a 3 m step in 12 m; left as it was)
UNDER_STUB_M = 2 * APPROACH_MIN_CROSS_M     # m: a part this short (the carriageways' fork a deck ends on, 3 m stubs on the
                                            # M1's) joined to a span leaves it a single span
UNDER_RUN_MIN_M = 2.0
ABUT_UNDER_GRADE = 0.08     # past a road's abutment end, over the deck, its ground falls at this grade (grade_points)
ABUT_NEAR_M = 14.0          # m from the bridge end the road is checked ...
ABUT_FAR_M = 40.0           # ... against its grade from ABUT_NEAR_M to this far
ABUT_STEEP = 0.15           # a fall this much steeper than that grade, near the end: the valley's side, not the road
ABUT_MIN_M = 0.3            # m: a smaller fill at the end is left out
ABUT_MAX_M = 8.0            # m filled at most


def median_profile(z, step, width):
    """The running median of a profile sampled every `step` m over `width` m (the window shrunk evenly near the ends, so
    an end keeps its height). Pure."""
    import fast
    half = int(round(width / 2.0 / step))
    if half < 1 or len(z) < 3:
        return np.asarray(z, np.float64)
    fast.require(71, "median_profile")
    return fast._ef.median_profile(np.ascontiguousarray(z, np.float64), half)


def smooth_profile(z, step, sigma):
    """Gaussian smoothing of a profile sampled every `step` m (turned about its end points past the ends: the ends keep
    their heights, so the parts of a road split at a junction meet at one height). Pure."""
    import fast
    fast.require(69, "smooth_profile")      # the Rust loop (~7000 roads a 9-tile set)
    return fast._ef.smooth_profile(np.ascontiguousarray(z, np.float64), float(step), float(sigma))


def resample(xy, step):
    """Points every `step` m along a polyline (xy (k, 2)), the last point kept, and their arc lengths."""
    import fast
    if not len(xy):
        return xy[:1].copy(), np.zeros(1)
    fast.require(0, "resample")
    return fast._ef.resample(np.ascontiguousarray(np.asarray(xy, np.float64)[:, :2]), float(step))


def raceway_surface(frame, elev, xy, hw, sigma=None):
    """(line (k, 2) every RACEWAY_STEP_M, its stations (k,), offsets across (nt,), surface (k, nt) m) of one raceway
    segment xy (half width hw, its margin in): RACEWAY_* above. None for a line under 2 points."""
    sigma = RACEWAY_SIGMA_M if sigma is None else float(sigma)
    xy = np.asarray(xy, np.float64)[:, :2]
    line, st = resample(xy, RACEWAY_STEP_M)
    k = len(line)
    if k < 2 or st[-1] <= 0.0:
        return None
    step = float(st[-1]) / (k - 1) if k > 1 else RACEWAY_STEP_M
    m = max(1, int(math.ceil(RACEWAY_EXTEND * sigma / RACEWAY_STEP_M)))
    closed = k > 3 and math.hypot(*(line[0] - line[-1])) < 1.0
    if closed and k > 2 * m + 2:
        head, tail = line[-1 - m:-1], line[1:m + 1]
    else:
        def unit(v):
            return v / max(math.hypot(*v), 1e-9)
        t0, t1 = unit(line[0] - line[1]), unit(line[-1] - line[-2])
        d = step * np.arange(1, m + 1)[:, None]
        head, tail = (line[0] + t0 * d)[::-1], line[-1] + t1 * d
    full = np.concatenate([head, line, tail])
    tan = np.gradient(full, axis=0)
    tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
    nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)             # (left)
    nt = int(math.ceil(2.0 * hw / RACEWAY_ACROSS_M)) + 1
    offs = np.linspace(-hw, hw, nt)
    pts = full[:, None, :] + offs[None, :, None] * nor[:, None, :]
    la, lo, _ = frame.enu_to_geodetic(pts[..., 0].ravel(), pts[..., 1].ravel(), 0.0)
    z = np.asarray(elev(la, lo), np.float64).reshape(len(full), nt)
    zs = np.stack([smooth_profile(z[:, j], step, sigma) for j in range(nt)], axis=1)
    return line, np.arange(k) * step, offs, zs[len(head):len(head) + k]


def raceway_ground(e, n, base, zones, margin=None):
    """base (the ground at points e, n) as the raceways' smoothed surfaces (zones: [(line, stations, offsets, surface)]
    from raceway_surface): each point by its nearest zone it lies on (station on the segment, offset within), all of
    it on the carriageway, fading out over margin (UNGRADED_MARGIN_M) to base at its edge. A point on none keeps base."""
    margin = UNGRADED_MARGIN_M if margin is None else float(margin)
    e = np.asarray(e, np.float64)
    n = np.asarray(n, np.float64)
    out = np.array(base, np.float64, copy=True)
    if not len(e) or not zones:
        return out
    import fast
    fast.require(164, "polyline_project")
    pts = np.ascontiguousarray(np.stack([e, n], axis=1))
    best = np.full(len(e), np.inf)
    add = np.zeros(len(e))
    ex0, ex1, ny0, ny1 = float(e.min()), float(e.max()), float(n.min()), float(n.max())
    for line, st, offs, corr in zones:
        hw = float(offs[-1])
        if line[:, 0].min() - hw > ex1 or line[:, 0].max() + hw < ex0 or                 line[:, 1].min() - hw > ny1 or line[:, 1].max() + hw < ny0:
            continue
        station, _near, _tan, off, dist = fast._ef.polyline_project(np.ascontiguousarray(line, np.float64), pts)
        station, off, dist = np.asarray(station), np.asarray(off), np.asarray(dist)
        ok = (station >= 0.0) & (station <= st[-1]) & (np.abs(off) <= hw) & (dist < best)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        fs = np.clip(station[idx] / max(float(st[1] - st[0]), 1e-9), 0.0, len(st) - 1.0)
        i = np.minimum(np.floor(fs).astype(np.int64), len(st) - 2)
        a = fs - i
        ft = np.clip((off[idx] - offs[0]) / max(float(offs[1] - offs[0]), 1e-9), 0.0, len(offs) - 1.0)
        j = np.minimum(np.floor(ft).astype(np.int64), len(offs) - 2)
        b = ft - j
        c = (corr[i, j] * (1 - a) * (1 - b) + corr[i + 1, j] * a * (1 - b) + corr[i, j + 1] * (1 - a) * b
             + corr[i + 1, j + 1] * a * b)
        u = np.clip((hw - np.abs(off[idx])) / max(margin, 1e-9), 0.0, 1.0)
        add[idx] = (c - out[idx]) * u * u * (3 - 2 * u)
        best[idx] = dist[idx]
    return out + add


class Plan:
    """One road's grading: vertices (n, 2) in the frame's ENU metres, their target heights, core half width; abut: does
    it lead onto a bridge at its (first, last) vertex (grade_points)."""
    __slots__ = ("xy", "z", "half", "bbox", "abut", "cut", "road_half", "rid", "max_wall", "bank", "steep", "widen")

    def __init__(self, xy, z, half, abut=(False, False), cut=None, road_half=None):
        self.xy, self.z, self.half, self.abut = xy, z, float(half), (bool(abut[0]), bool(abut[1]))
        # (n - 1, 6) per segment, NaN where none: a cutting's wall offset from the centre line, its top's height and
        # the top's offset, on the left, then the right (_add_cuts)
        self.cut = cut
        self.road_half = float(road_half) if road_half is not None else float(half)    # the carriageway's half width
        self.rid = -1           # the road (map feature) it is part of: _plan_cell (each cell plans its own stretch of it)
        self.max_wall = WALL_MAX_H_M    # its cuttings' walls at most this tall (wall_max_h)
        self.bank = None                # (n,) its cross slope (m per m toward its left: a banked curve, bank) or None
        self.steep = False              # a mountain pass road's: steep sides (STEEP_RUN, roads._mountain_roads)
        self.widen = None               # (n,) m its core and carriageway are wider each side there (hairpin_cores) or None
        pad = half + T_MAX
        self.bbox = (xy[:, 0].min() - pad, xy[:, 1].min() - pad, xy[:, 0].max() + pad, xy[:, 1].max() + pad)


def clip_polyline(xy, box):
    """The runs (each (k, 2), k >= 2) of polyline xy (k, 2) inside box (x0, y0, x1, y1), segments cut at its edges
    (Liang-Barsky). Pure."""
    x0, y0, x1, y1 = box
    runs, cur = [], None
    for i in range(len(xy) - 1):
        px, py = float(xy[i, 0]), float(xy[i, 1])
        dx, dy = float(xy[i + 1, 0]) - px, float(xy[i + 1, 1]) - py
        ta, tb, ok = 0.0, 1.0, True
        for p, q in ((-dx, px - x0), (dx, x1 - px), (-dy, py - y0), (dy, y1 - py)):
            if p == 0.0:
                if q < 0.0:
                    ok = False
                    break
                continue
            r = q / p
            if p < 0.0:
                if r > tb:
                    ok = False
                    break
                ta = max(ta, r)
            else:
                if r < ta:
                    ok = False
                    break
                tb = min(tb, r)
        if not ok or tb <= ta:
            if cur is not None:
                runs.append(cur)
                cur = None
            continue
        a = (px + dx * ta, py + dy * ta) if ta > 0.0 else (px, py)
        b = (px + dx * tb, py + dy * tb) if tb < 1.0 else (float(xy[i + 1, 0]), float(xy[i + 1, 1]))
        if cur is not None and ta == 0.0:
            cur.append(b)
        else:
            if cur is not None:
                runs.append(cur)
            cur = [a, b]
        if tb < 1.0:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    return [np.array(r, np.float64) for r in runs if len(r) >= 2]


def plan_road(frame, lonlat, spec, elev):
    """The Plan of one road part (lonlat (k, 2)); elev(lat, lon) -> m at PROFILE_ZOOM. Pure given elev."""
    return plan_roads(frame, [(lonlat, spec)], elev)[0]


def _normals(xy):
    """Unit normals (k, 2) of a polyline's points (the direction from the neighbours). Pure."""
    xy = np.asarray(xy, np.float64)[:, :2]
    if len(xy) < 2:
        return np.zeros((len(xy), 2))
    d = np.gradient(xy, axis=0)
    ln = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    return np.stack([-d[:, 1] / ln, d[:, 0] / ln], axis=1)


_STENCILS = {}


def _stencil(r, sigma):
    """(offsets (19, 2) m, weights summing to 1): hexagonal rings out to r (the centre, 6 at r / 2, 12 at r), Gaussian
    weights of sigma."""
    got = _STENCILS.get((r, sigma))
    if got is None:
        pts = [(0.0, 0.0)]
        for ring, n in ((0.5, 6), (1.0, 12)):
            pts += [(r * ring * math.cos(2 * math.pi * i / n), r * ring * math.sin(2 * math.pi * i / n)) for i in range(n)]
        off = np.array(pts, np.float64)
        w = np.exp(-0.5 * (np.hypot(off[:, 0], off[:, 1]) / sigma) ** 2)
        got = _STENCILS[(r, sigma)] = (off, w / w.sum())
    return got


def field_heights(frame, elev, xy, r, sigma):
    """elev's Gaussian-weighted mean over the _stencil(r, sigma) around each point xy (n, 2) in the frame: one call."""
    off, wts = _stencil(r, sigma)
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    cloud = (xy[None, :, :] + off[:, None, :]).reshape(-1, 2)
    la, lo, _ = frame.enu_to_geodetic(cloud[:, 0], cloud[:, 1], 0.0)
    z = np.asarray(elev(la, lo), np.float64).reshape(len(off), -1)
    return (wts[:, None] * z).sum(axis=0)


def plan_roads(frame, parts, elev, abut=None, ext=None, pins=None, cuts=True, coarse=False, obs=None, decks=None,
               rings=None):
    """plan_road for many (lonlat, spec) parts with one elev call (its cost is per call); abut: per part (start, end)
    flags, the ends leading onto a bridge (fill_abutment); ext: per part (before, after), the lines (k, 2) in the frame
    from its start / end outward along the part the road goes on in (Grader._extensions), or None: their ground is in
    the part's profile smoothing; pins: per part (start, end) heights its ends are held at, or None (Grader._pins: a
    road ending on a more smoothed one meets it at its height); obs(xy (m, 2), spec) -> measured heights (m,) of a
    part's samples (NaN where none) or None: they stand in for the DEM's (roadz.py, on coarse elevation: Grader._obs);
    decks: [(line (k, 2) in the frame, half width)] crossing over the roads, their ground read beside them
    (deck_capped; not on coarse elevation); rings: per part, True for a mapped roundabout's ring (roads.is_ring: not
    banked, RING_LEVEL_ON). Pure given elev and obs."""
    if not parts:
        return []
    ll = np.concatenate([lonlat for lonlat, _ in parts])
    E, N, _ = frame.geodetic_to_enu(ll[:, 1], ll[:, 0], 0.0)
    onto_xy = np.zeros((0, 2))          # the part ends leading onto a bridge (fill_abutment's)
    if abut is not None and ABUT_FILL_ON:
        last = np.cumsum([len(lonlat) for lonlat, _ in parts]) - 1
        onto_xy = np.array([(E[i], N[i]) for j, (lonlat, _) in enumerate(parts) if len(lonlat)
                            for at_end, i in ((0, last[j] - len(lonlat) + 1), (1, last[j])) if abut[j][at_end]],
                           np.float64).reshape(-1, 2)
    prof, k = [], 0
    for j, (lonlat, spec) in enumerate(parts):
        m = len(lonlat)
        line = np.stack([E[k:k + m], N[k:k + m]], axis=1)
        k += m
        # (an even step, PROFILE_STEP or a little under, so the samples end on both ends: a short last step made the
        # smoothing lean at that end, and the parts of a road met 0.2 m apart on a 20 % grade)
        length = float(np.hypot(*np.diff(line, axis=0).T).sum()) if m >= 2 else 0.0
        step = length / max(1, math.ceil(length / PROFILE_STEP - 1e-9)) * (1.0 + 1e-9) if length > 0 else PROFILE_STEP
        xy, _ = resample(line, step)
        if len(xy) < 2:
            prof.append(None)
            continue
        side, bank_side, onto = [], [], []
        for out in (ext[j] if ext is not None else (None, None)):
            if out is None or len(out) < 2:
                side.append(np.zeros((0, 2)))
                bank_side.append(np.zeros((0, 2)))
                onto.append(False)
            else:                       # (from the node outward; the node is the part's own end sample)
                bank_side.append(resample(np.asarray(out, np.float64), step)[0][1:])
                # Extra bank geometry must not change longitudinal DEM sampling
                # or the bridge solver's established approach/underpass profiles.
                reach = 3.0 * profile_sigma(spec, coarse) + PROFILE_STEP
                arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))]
                if arc[-1] > reach + 1e-8:
                    out = np.vstack([np.asarray(out)[arc < reach], _point_along(out, reach)])
                    onto.append(False)
                else:                   # (does it end where another part leads onto a bridge?)
                    onto.append(bool(len(onto_xy)) and
                                float(np.hypot(*(onto_xy - np.asarray(out, np.float64)[-1]).T).min()) <= CHAIN_JOIN_M)
                side.append(resample(np.asarray(out, np.float64), step)[0][1:])
        prof.append((xy, side[0][::-1], side[1], step, bank_side[0][::-1], bank_side[1], onto))
    pts = [np.concatenate([b, xy, a]) for xy, b, a, *_ in (p for p in prof if p is not None)]
    if not pts:
        return [None] * len(parts)
    allxy = np.concatenate(pts)
    if coarse:
        # coarse elevation (dem.coarse_at, GitHub #414): the median of the centre line and COARSE_SIDE_M to either side,
        # so a pit or a spike narrower than the data's own cells does not drag the road into a walled trench (Perm's
        # primary sat 4-5 m under the ground 7 m beside it, at grade in the street photos)
        nrm = np.concatenate([_normals(q) for q in pts])
        three = np.concatenate([allxy, allxy + nrm * COARSE_SIDE_M, allxy - nrm * COARSE_SIDE_M])
        la, lo, _ = frame.enu_to_geodetic(three[:, 0], three[:, 1], 0.0)
        z3 = np.asarray(elev(la, lo), np.float64).reshape(3, -1)
        zall = np.median(z3, axis=0)
    else:
        la, lo, _ = frame.enu_to_geodetic(allxy[:, 0], allxy[:, 1], 0.0)
        zall = np.asarray(elev(la, lo), np.float64)
    out, k = [], 0
    for j, ((lonlat, spec), pr) in enumerate(zip(parts, prof)):
        if pr is None:
            out.append(None)
            continue
        xy, before, after, step, bank_before, bank_after, onto = pr
        nb, na = len(before), len(after)
        zx = zall[k:k + nb + len(xy) + na]
        k += nb + len(xy) + na
        if decks and not coarse:        # (the part's samples and its extensions', smoothed together)
            zx = deck_capped(np.concatenate([before, xy, after]), zx, decks)
        if obs is not None:
            o = obs(xy, spec)
            if o is not None and len(o) == len(xy) and np.isfinite(o).any():
                zx = zx.copy()
                zx[nb:nb + len(xy)] = np.where(np.isfinite(o), o, zx[nb:nb + len(xy)])
        ab = (False, False)
        if abut is not None and ABUT_FILL_ON:
            ab = (bool(abut[j][0]), bool(abut[j][1]))
            if ab[0] and nb:            # (no extension past an end onto a bridge)
                zx, nb = zx[nb:], 0
            if ab[1] and na:
                zx, na = zx[:len(zx) - na], 0
            # Filled with the road it goes on in past its other end: a part split at a junction by the bridge alone had
            # too few samples for the grade, and its end stayed at the slope's foot, 6 m under the road (Kurna - Tuhala
            # over Tallinna ringtee, 59.3364 N 24.8239 E, GitHub #684). An extension running onto a bridge's end (the
            # part on the other side of that junction) is filled the same, so the two parts meet.
            for at_end, n_ext in ((False, nb), (True, na)):
                if ab[int(at_end)] or (n_ext and onto[int(at_end)]):
                    zx = fill_abutment(zx, at_end, step)
        pin = pins[j] if pins is not None else (None, None)
        if pin[0] is not None and not nb and not ab[0]:     # (the end sample held: the smoothing keeps an end's height)
            zx = zx.copy()
            zx[0] = pin[0]
        if pin[1] is not None and not na and not ab[1]:
            zx = zx.copy()
            zx[-1] = pin[1]
        # Curvature and its runoff need the joined-road context too. Computing
        # banking on xy alone puts a partial curvature window at every map
        # segment end, even where the longitudinal profile is continuous.
        if ab[0]:
            bank_before = bank_before[:0]
        if ab[1]:
            bank_after = bank_after[:0]
        bank_xy = np.concatenate([bank_before, xy, bank_after])
        out.append(_plan(xy, zx, spec, ab, (nb, na), step, bank_xy=bank_xy,
                         bank_ext=(len(bank_before), len(bank_after)),
                         banked=not (RING_LEVEL_ON and rings is not None and rings[j]), coarse=coarse))
    if SHARP_ON and cuts:           # (cuts False: the profile alone, Grader._stronger_z)
        _add_cuts(frame, [p if p is not None and sp.kind in roadsmod.BUILT else None
                          for p, (_ll, sp) in zip(out, parts)], elev)
    return out


WALL_MAX_H_M = 4.0      # m a sharp cutting's wall stands over the road at most (the slope above it is graded) ...
WALL_MAX_H_CITY_M = 10.0  # ... a major road's in town (concrete): the full depth of its trench (Munich's McGraw-Graben:
                        # vertical concrete walls the whole way up, Mapillary 2026-10-21; with the 4 m cap the bank above
                        # stood as a bare cliff over the wall once the near ground was drawn sharp, 2026-10-22)


def wall_max_h(spec):
    """How tall a sharp cutting's wall beside road spec (roads.Spec) may stand: WALL_MAX_H_CITY_M for a major road in
    town (roads.CONCRETE_WALL_CLASSES, the concrete walls), else WALL_MAX_H_M (stone walls, mountain passes)."""
    return WALL_MAX_H_CITY_M if (spec.urban and spec.cls in roadsmod.CONCRETE_WALL_CLASSES) else WALL_MAX_H_M
HAIRPIN_SPAN_M = 20.0   # a road's direction this far back and on ...
HAIRPIN_TURN_DEG = 90.0  # ... turned this much: a tight bend (earthfast.tight_bends) ...
HAIRPIN_PAD_M = 15.0    # ... and this far either side of it: no sharp cutting, no retaining wall


def bend_zone(xy):
    """Which points of a polyline (k, 2) lie at or within HAIRPIN_PAD_M (along it) of a tight bend."""
    import fast
    fast.require(42, "tight_bends")
    xy = np.ascontiguousarray(np.asarray(xy, np.float64)[:, :2])
    m = np.asarray(fast._ef.tight_bends(xy, float(HAIRPIN_SPAN_M), float(HAIRPIN_TURN_DEG)))
    if not m.any():
        return m
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    sb = s[m]
    return np.abs(s[:, None] - sb[None, :]).min(axis=1) <= HAIRPIN_PAD_M


def _add_cuts(frame, plans, elev):
    """Sets Plan.cut on the plans (None entries skipped): per segment and side a cutting's wall (SHARP_*), from the raw
    ground elev(lat, lon) sampled outward from the carriageway's edge: the ground rising SHARP_MIN_M or more at
    SHARP_GRADE or steeper on average. The data spreads a vertical wall evenly round where it stands: its wall is put
    where the climb is half done (never on the carriageway; a fall: never inside the core, its sidewalk on the level),
    its top where it is done (the Rust kernels earthfast.sharp_probes / sharp_cuts; the ground lookup between them)."""
    import fast
    fast.require(42, "sharp cuttings")
    f64 = np.float64
    keep = [p for p in plans if p is not None and len(p.xy) >= 2]
    jobs = []
    for p in keep:
        q, o = fast._ef.sharp_probes(np.ascontiguousarray(p.xy[:, :2], f64), float(p.road_half), float(p.half),
                                     float(SHARP_REACH_M), float(SHARP_STEP_M))
        jobs.append((p, q, o))
    if not jobs:
        return
    allp = np.concatenate([q for _, q, _ in jobs])
    la, lo = enu_to_ll_dense(frame, allp[:, 0], allp[:, 1])
    g = np.asarray(elev(np.asarray(la), np.asarray(lo)), f64)       # (as plan_roads' heights)
    k = 0
    for p, q, o in jobs:
        cut = fast._ef.sharp_cuts(np.ascontiguousarray(g[k:k + len(q)]), np.ascontiguousarray(p.z, f64), o,
                                  float(p.road_half), float(p.half), float(SHARP_MIN_M), float(SHARP_GRADE),
                                  float(SHARP_STEP_M), int(SHARP_RUN_N))
        k += len(q)
        if cut is not None:
            # (none at a tight bend: a hairpin's apex and approaches are graded smoothly, no wall into the turn)
            bend = bend_zone(p.xy[:, :2])
            cut = cut.copy()
            if bend.any():
                cut[bend[:-1] | bend[1:]] = np.nan
            # (a cutting's wall at most WALL_MAX_H_M tall, the ground above it sloping back to the hill as real pass
            # walls do; a 12 m wall stood between the Stelvio's hairpin legs, 2026-10-21)
            zmin = np.minimum(p.z[:-1], p.z[1:])
            # (a shallow climb on one side only: no cutting there, SHARP_LONE_MIN_M)
            climb = np.isfinite(cut[:, [1, 4]])
            shallow = climb & ((cut[:, [1, 4]] - zmin[:, None]) < SHARP_LONE_MIN_M)
            lone = shallow & ~climb[:, ::-1]
            for side, c0 in ((0, 0), (1, 3)):
                cut[lone[:, side], c0:c0 + 3] = np.nan
            for c in (1, 4):
                cut[:, c] = np.minimum(cut[:, c], zmin + getattr(p, "max_wall", WALL_MAX_H_M))
            if np.isfinite(cut[:, 0]).any() or np.isfinite(cut[:, 3]).any():
                p.cut = cut


LL_GRID_M = 32.0       # m: enu_to_ll_dense's node spacing (the frame's conversion is smooth to < 0.1 mm over a cell)


def enu_to_ll_dense(frame, x, y, step=LL_GRID_M):
    """(lat, lon) of many points (x, y) in frame: the frame's exact conversion at the nodes of a step-m grid over
    them, interpolated between the nodes (earthfast grid_bilinear2); a scattered few (more nodes than a quarter of
    the points) converted one by one. _add_cuts' millions of probes: their conversion was most of its time."""
    import fast
    fast.require(158, "grid_bilinear2")
    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    if len(x) == 0:
        return np.zeros(0), np.zeros(0)
    x0 = math.floor(float(x.min()) / step) * step
    y0 = math.floor(float(y.min()) / step) * step
    nx = int(math.ceil((float(x.max()) - x0) / step)) + 2
    ny = int(math.ceil((float(y.max()) - y0) / step)) + 2
    if nx * ny * 4 > len(x):
        la, lo, _ = frame.enu_to_geodetic(x, y, 0.0)
        return np.asarray(la, np.float64), np.asarray(lo, np.float64)
    gx, gy = np.meshgrid(x0 + step * np.arange(nx), y0 + step * np.arange(ny))
    gla, glo, _ = frame.enu_to_geodetic(gx.ravel(), gy.ravel(), 0.0)
    return fast._ef.grid_bilinear2(np.asarray(gla, np.float64).reshape(ny, nx),
                                   np.asarray(glo, np.float64).reshape(ny, nx), x0, y0, step, x, y)


def _point_along(line, d):
    """The point d m along polyline line (k, 2) from its first point (its last when shorter). Walks only as far as it
    needs (a whole line's cumulative length per road end cost Grader._extensions most of its time, 2026-10-25)."""
    acc = 0.0
    for i in range(len(line) - 1):
        ax, ay = float(line[i][0]), float(line[i][1])
        bx, by = float(line[i + 1][0]), float(line[i + 1][1])
        seg = math.hypot(bx - ax, by - ay)
        if seg > 0 and acc + seg >= d:
            t = (d - acc) / seg
            return np.array([ax + (bx - ax) * t, ay + (by - ay) * t])
        acc += seg
    return np.array([float(line[-1][0]), float(line[-1][1])])


def profile_sigma(spec, coarse=False):
    """m: the profile smoothing of a road of this spec (coarse: on the coarse global elevation, plan_roads)."""
    if spec.kind not in roadsmod.BUILT:
        if coarse:
            return max(DIRT_SIGMA_COARSE_M, DIRT_SIGMA_M)
        if spec.cls in DIRT_SOFT_CLASSES:
            return DIRT_SIGMA_M          # a main dirt road stays as smooth as it was (dirt_bumps' "main" ones too)
        return DIRT_SIGMA_MINOR_M        # a minor one keeps the ground's own bumps
    if spec.cls in MAJOR_CLASSES and not getattr(spec, "link", False):
        return max(SIGMA_M, SIGMA_MAJOR_M)
    return max(SIGMA_M, CLASS_SIGMA_M.get(spec.cls, SIGMA_M)) if not getattr(spec, "link", False) else SIGMA_M


DIRT_ROUGH = float(os.environ.get("BNE_DIRT_BUMP", "1.0"))    # strength of a minor dirt road's damage (1: as below)
DIRT_MAJOR_SHARE = float(os.environ.get("BNE_DIRT_MAJOR", "0.15"))  # a MAIN dirt road is this much of it (softer)
# which dirt roads count as "main" (the ones kept soft): the classes a country road is signed as, not just the
# motorway ones (a mapped `secondary` gravel road is a main road where you drive it at 80 km/h)
DIRT_SOFT_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary"}
# a forest track (highway=track) is worse than any of them: "you cannot even get up to speed on it" (the user, 2026-11-02)
DIRT_TRACK_SHARE = float(os.environ.get("BNE_DIRT_TRACK", "1.3"))
DIRT_TRACK_WAVE_SHARE = float(os.environ.get("BNE_DIRT_TRACK_WAVE", "0.5"))   # ... but less of the soft undulation:
                                                                              # waves read as a pattern, holes do not
DIRT_VERTEX_STEP = 2.0          # m between a dirt road's plan vertices (4 m smoothed a pothole away)
DIRT_UNDULATION_M = 0.02        # m: centimetre-scale wear, not waves along the whole carriageway
DIRT_UNDULATION_WAVES = ((10.0, 1.0), (6.0, 0.8))
DIRT_POTHOLE_EVERY_M = 5.0      # m of road per pothole, on average
DIRT_POTHOLE_W_M = (3.0, 6.0)   # m: a pothole's width
DIRT_POTHOLE_D_M = (0.025, 0.045)  # m: its depth (before the track/main share and aggregate bound)
DIRT_PATCH_EVERY_M = 10.0       # m of road per broken patch (a step of a few cm with soft edges)
DIRT_PATCH_W_M = (3.0, 7.0)
DIRT_PATCH_D_M = (0.015, 0.030)
DIRT_PIT_EVERY_M = 150.0        # m of road per deep pit
DIRT_PIT_W_M = (2.0, 4.5)
DIRT_PIT_D_M = (0.04, 0.07)
DIRT_DEPTH_MAX_M = 0.08        # total synthetic damage, not per hole: overlapping/track-scaled pits must stay local
DIRT_RISE_MAX_M = 0.025        # removing the damage's mean must not turn the ground between holes into hills
DIRT_STEP_MAX_M = float(os.environ.get("BNE_DIRT_STEP", "0.35"))   # m: steepest artificial damage change per 2 m
DIRT_BASE_M = 25.0             # m: the bump pattern's own long waves are removed over this length, so a road keeps
                               # the terrain's course instead of sinking into it (the roads sat in a ditch, further in
                               # the worse it got: the damage's mean pulled them down along their length)
DIRT_TAPER_M = 22.0            # m at each end of a dirt road where its damage fades out: a junction meets the road it
                               # joins at that road's height, not on a berm of its own potholes


def dirt_taper(s_, length):
    """(n,) 0 at a dirt part's two ends (s_ from 0 to length) rising smoothly to 1 within DIRT_TAPER_M (half a short
    part's length): its damage and lean fade out where it meets another road, whatever its length."""
    s_ = np.asarray(s_, np.float64)
    if length <= 0.0 or DIRT_TAPER_M <= 0.0:
        return np.zeros(len(s_)) if length <= 0.0 else np.ones(len(s_))
    tap = np.clip(np.minimum(s_, length - s_) / min(DIRT_TAPER_M, 0.5 * length), 0.0, 1.0)
    return tap * tap * (3.0 - 2.0 * tap)


def dirt_bumps(xy, spec, n, ext=(0, 0), step=PROFILE_STEP):
    """What a minor dirt road is really like, one value per profile sample (n samples, ext: how many lie past the
    part's ends): a soft undulation plus discrete damage -- potholes, broken patches (a step with soft edges) and the
    odd deep pit -- placed from the road's own position, so every process builds the same ones. None for a built
    road. The sizes are metres; BNE_DIRT_BUMP scales the lot (1: as here, 0: off). DEM smoothing extensions get zero
    damage: neighbouring samples must not change this part's pattern or move its junction taper."""
    if spec.kind in roadsmod.BUILT or DIRT_ROUGH <= 0.0:
        return None
    if spec.cls == "track":
        scale, wave = DIRT_TRACK_SHARE, DIRT_TRACK_WAVE_SHARE
    elif spec.cls in DIRT_SOFT_CLASSES:
        scale = wave = DIRT_MAJOR_SHARE
    else:
        scale = wave = 1.0
    if scale <= 0.0 and wave <= 0.0:
        return None
    xy = np.asarray(xy, np.float64)[:, :2]
    n_core = n - ext[0] - ext[1]
    if len(xy) < 2 or n_core <= 0:
        return None
    n = n_core
    s_ = np.arange(n, dtype=np.float64) * float(step)
    seed = (int(round(float(xy[0, 0]) * 7.0)) * 7919 + int(round(float(xy[0, 1]) * 7.0)) * 104729) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    share2 = 0.0
    for knot_m, wshare in DIRT_UNDULATION_WAVES:           # the road's own soft rise and fall
        if s_[-1] < knot_m:
            continue
        knots = rng.normal(0.0, 1.0, int(s_[-1] / knot_m) + 3)
        t = s_ / knot_m
        i0 = np.clip(np.floor(t).astype(int), 0, len(knots) - 2)
        f = t - i0
        f = f * f * (3.0 - 2.0 * f)
        out += wshare * (knots[i0] * (1.0 - f) + knots[i0 + 1] * f)
        share2 += wshare * wshare
    if float(out.std()) > 1e-9:
        out *= DIRT_UNDULATION_M * wave / math.sqrt(share2)

    i_of = lambda pos: pos / float(step)    # (sample index of a distance along the part)

    def add(c, half, fn):                                  # a feature's few metres only (it is ~0 farther out):
        i0 = max(0, int(math.floor(i_of(c - half))))       # the whole road per pothole was O(length^2)
        i1 = min(n, int(math.ceil(i_of(c + half))) + 1)
        if i1 > i0:
            out[i0:i1] += fn(s_[i0:i1])

    def bowl(c, w, d):                                     # a rounded hole: full depth in the middle, soft rim
        add(c, 1.5 * w, lambda x: -d * np.exp(-((x - c) / (0.45 * w)) ** 2))

    length = float(s_[-1]) if len(s_) else 0.0
    if length > 10.0:
        for _ in range(int(rng.poisson(max(1.0, length / DIRT_POTHOLE_EVERY_M)))):
            bowl(float(rng.uniform(0.0, length)), float(rng.uniform(*DIRT_POTHOLE_W_M)),
                 float(rng.uniform(*DIRT_POTHOLE_D_M)) * DIRT_ROUGH * scale)
        for _ in range(int(rng.poisson(max(1.0, length / DIRT_PATCH_EVERY_M)))):   # broken patch: a step, soft edges
            c = float(rng.uniform(0.0, length))
            w = float(rng.uniform(*DIRT_PATCH_W_M))
            d = float(rng.uniform(*DIRT_PATCH_D_M)) * DIRT_ROUGH * scale * (1.0 if rng.random() < 0.5 else -1.0)
            edge = 0.35 * w
            add(c, 0.5 * w + 4.0 * edge, lambda x, c=c, w=w, d=d, edge=edge:
                d * 0.5 * (np.tanh((x - (c - 0.5 * w)) / edge) - np.tanh((x - (c + 0.5 * w)) / edge)))
        for _ in range(int(rng.poisson(max(0.3, length / DIRT_PIT_EVERY_M)))):      # the odd deep pit
            bowl(float(rng.uniform(0.0, length)), float(rng.uniform(*DIRT_PIT_W_M)),
                 float(rng.uniform(*DIRT_PIT_D_M)) * DIRT_ROUGH * scale)
        if DIRT_BASE_M > 0.0 and length > 1.5 * DIRT_BASE_M:   # keep the terrain's own course: no net rise or fall
            k = max(1, int(DIRT_BASE_M / max(step, 1e-6)))
            pad = k // 2
            padded = np.r_[np.full(pad, out[0]), out, np.full(k - pad - 1, out[-1])]
            out = out - np.convolve(padded, np.ones(k) / k, mode="valid")[:len(out)]
    # Tracks still get more damage, but not arbitrarily deep overlapping holes. Main roads keep their smaller
    # share; BNE_DIRT_BUMP still scales the bounds. Bound after removing the mean, which can itself raise berms.
    strength = DIRT_ROUGH * min(1.0, max(scale, wave))
    out = np.clip(out, -DIRT_DEPTH_MAX_M * strength, DIRT_RISE_MAX_M * strength)
    # The asymmetric bounds can leave a net depression. Balance by reducing the stronger side only, so neither
    # the depth nor the rise can exceed its bound again (and a zero sample never turns into a raised berm).
    if DIRT_BASE_M > 0.0 and length > 1.5 * DIRT_BASE_M:
        positive, negative = out > 0.0, out < 0.0
        up, down = float(out[positive].sum()), -float(out[negative].sum())
        if down > up:
            out[negative] *= up / down
        elif up > down:
            out[positive] *= down / up
    out *= dirt_taper(s_, length)       # (nothing right at a junction, short parts too: that road sets the height)
    return np.pad(out, ext)


def _plan(xy, z, spec, abut=(False, False), ext=(0, 0), step=PROFILE_STEP, bank_xy=None, bank_ext=None, banked=True,
          coarse=False):
    """ext: (before, after), samples of z past the part's ends (plan_roads): smoothed with it, then dropped. step: m
    between the samples. bank_xy: the matching extended geometry, trimmed after the curvature/runoff pass. banked
    False: level across (a roundabout's ring)."""
    built = spec.kind in roadsmod.BUILT
    zs = smooth_profile(median_profile(z, step, MEDIAN_M if built else DIRT_MEDIAN_M), step,
                        profile_sigma(spec, coarse))[ext[0]:len(z) - ext[1]]
    bumps = dirt_bumps(xy, spec, len(z), ext, step)
    if bumps is not None:                       # (not smoothed again: a pothole's edge is sharp, that is the point)
        damage = bumps[ext[0]:len(z) - ext[1]].copy()
        cap = DIRT_STEP_MAX_M
        if spec.cls == "track":
            cap *= 1.6
        cap *= step / PROFILE_STEP
        if cap > 0.0 and len(damage) > 1:
            # Limit the artificial damage alone, not the real climb. Integrating clipped TOTAL height differences
            # then smoothing the lost height back over 40 m made local pits into long hills/dips. Two local sweeps
            # bound neighbouring damage values without accumulating an offset down the road. Keep the ends pinned.
            # (native: dirtsteps.rs, the same two sweeps; Python loops per sample before)
            import fast
            fast.require(157, "dirt_step_clamp")
            damage = np.asarray(fast._ef.dirt_step_clamp(np.ascontiguousarray(damage, np.float64), float(cap)))
        zs = zs + damage
    # thin to VERTEX_STEP (the profile stays smooth between: linear in the grading pass); a minor dirt road keeps
    # every sample, or its potholes would be smoothed away by the linear stretches between the vertices
    minor_dirt = bumps is not None and spec.cls not in DIRT_SOFT_CLASSES
    vstep = DIRT_VERTEX_STEP if minor_dirt else VERTEX_STEP
    every = max(1, int(round(vstep / PROFILE_STEP)))
    keep = np.r_[np.arange(0, len(xy) - 1, every), len(xy) - 1]
    # the carriageway as a city street would have it (never from the land cover), or the measured one
    # (a mountain pass road its own: no town on a pass, roads._mountain_roads)
    width = max(spec.width, roadsmod.carriageway(spec.cls, spec.kind, True, None, spec.oneway)) \
        if built and not getattr(spec, "mountain", False) else spec.width
    if spec.median:             # a divided road's carriageway: never over the other one
        width = max(spec.width, roadsmod.dual_cap(spec.median, width))
    plan = Plan(xy[keep], zs[keep], width / 2 + shoulder(spec), abut, road_half=width / 2)
    plan.max_wall = wall_max_h(spec)
    plan.steep = bool(getattr(spec, "mountain", False))
    bext = ext if bank_ext is None else bank_ext
    b = bank(xy if bank_xy is None else bank_xy, spec, abut, bext) if banked else None
    if b is not None:
        if bank_xy is not None:
            b = b[bext[0]:len(bank_xy) - bext[1]]
        if not built:           # a dirt road's lean: level across where it meets another road (after the trim: the
            sp = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.asarray(xy, np.float64)[:, :2], axis=0).T))]  # ramp
            b = b * dirt_taper(sp, float(sp[-1]))                                      # was trimmed away with ext)
        plan.bank = b[keep]
    return plan


# Banked curves (the user, 2026-10-27: "improve how roads generate"; BeamNG's Hero Road editor banks its curves, our
# carriageways were graded level across): a motorway's or trunk's curve leans into its turn, the cross slope by the
# design-speed rule e = BANK_E_MAX x min(1, r_min / R), r_min = v² / (127 (e_max + f)) of the class's design speed v
# (km/h) and side friction f; a gentle bend (under 1 %) keeps level. It comes in over BANK_RUNOFF_M and fades out over
# BANK_FADE_M where the road meets a bridge (a deck is level across) or ends without a road going on (a tunnel's
# floor, a junction). The grading pass (grade_points) tilts the core across the carriageway by it.
BANK_ON = os.environ.get("BNE_BANK", "on") != "off"
BANK_E_MAX = 0.08
BANK_DESIGN = {"motorway": (110.0, 0.10), "trunk": (90.0, 0.12)}     # km/h, side friction
BANK_LINK = (60.0, 0.16)                                              # their slip roads
# Sparse motorway map lines are straight chords, often ~100 m between vertices: a 12 m stencil saw each chord as level
# and each corner as a tight bend, 8 % bank pulses (I-80, 41.41135, -105.83960; Discord 2026-10-28 and PR #410). The
# curvature is the road's mean turning over this far either side (earthfast.bank_profile), its context the joined
# roads' too (plan_roads bank_xy), then smoothed.
BANK_STENCIL_M = 100.0      # 200 m baseline
BANK_RUNOFF_M = 100.0       # full 8 % transition at <= 0.08 / 100 m away from ends
BANK_FADE_M = 100.0         # smoothstep to level at a bridge / loose end; no deck-height changes
BANK_SIGMA_M = 12.0         # remove discrete box-window steps in cross slope, not road elevation
# ... but not a mapped roundabout's ring (roads.is_ring): the Jüri roundabout of Tallinna ringtee (59.3535 N 24.8820 E,
# trunk, about 190 m across) was banked 8 % into the ring as a trunk's curve; the real one falls 2.4-2.9 % outward (the
# Estonian lidar). Rings are graded level across. BNE_RING_LEVEL=off: banked as before.
RING_LEVEL_ON = os.environ.get("BNE_RING_LEVEL", "on") != "off"


DIRT_BANK_M = float(os.environ.get("BNE_DIRT_BANK", "0.015"))   # cross slope a worn dirt road leans to (m per m)
DIRT_BANK_EDGE_M = 0.04         # maximum synthetic lift/drop of either edge, also on wide gravel roads
DIRT_BANK_WAVES = ((27.0, 1.0), (13.0, 0.6), (7.0, 0.35))      # how often the lean changes sides (m)


def dirt_bank(xy, spec, ext=(0, 0), step=PROFILE_STEP):
    """The cross slope a dirt road leans to along itself (m per m toward its left): now the left side is low, now the
    right, at a few scales -- a pothole on one side of the road rather than a wave across the whole of it. Seeded by the
    road's own position, so every process builds the same. None for a built road or BNE_DIRT_BANK 0."""
    if spec.kind in roadsmod.BUILT or DIRT_BANK_M <= 0.0:
        return None
    xy = np.asarray(xy, np.float64)[:, :2]
    if len(xy) < 3:
        return None
    s_ = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    seed = (int(round(float(xy[0, 0]) * 11.0)) * 2654435761 + int(round(float(xy[0, 1]) * 11.0)) * 40503) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    out = np.zeros(len(xy))
    norm = 0.0
    for size, share in DIRT_BANK_WAVES:
        if s_[-1] < size:
            continue
        knots = rng.uniform(-1.0, 1.0, int(s_[-1] / size) + 3)
        t = s_ / size
        i0 = np.clip(np.floor(t).astype(int), 0, len(knots) - 2)
        f = t - i0
        f = f * f * (3.0 - 2.0 * f)
        out += share * (knots[i0] * (1.0 - f) + knots[i0 + 1] * f)
        norm += share * share
    if float(out.std()) > 1e-9:
        out *= DIRT_BANK_M / math.sqrt(norm)
    # A slope-only limit grew with road width: a 3.5 m track at Mering kept ~83 cm edge-to-edge waves even after
    # bounding the centreline damage. Bound the edge displacement itself, before the junction fade.
    edge_slope = DIRT_BANK_EDGE_M / max(float(spec.width) / 2.0, 1e-6)
    out = np.clip(out, -edge_slope, edge_slope)
    if ext[0] or ext[1]:                      # a junction is level across: the lean fades out at the ends
        n = len(out)
        k = max(1, int(min(n, 12)))
        ramp = np.ones(n)
        ramp[:k] = np.linspace(0.0, 1.0, k)
        ramp[-k:] = np.linspace(1.0, 0.0, k)
        out *= ramp
    return out


def bank(xy, spec, abut=(False, False), ext=(0, 0)):
    """(n,) the cross slope of a road part's centreline xy (n, 2) (m per m toward its left), or None (not banked)."""
    if spec.kind not in roadsmod.BUILT:
        return dirt_bank(xy, spec, ext)
    if not BANK_ON or spec.cls not in BANK_DESIGN or len(xy) < 3:
        return None
    v, f = BANK_LINK if getattr(spec, "link", False) else BANK_DESIGN[spec.cls]
    r_min = v * v / (127.0 * (BANK_E_MAX + f))
    import fast
    fast.require(84, "bank_profile")
    fade = [BANK_FADE_M if (abut[k] or not ext[k]) else 0.0 for k in (0, 1)]
    b = np.asarray(fast._ef.bank_profile(np.ascontiguousarray(np.asarray(xy, np.float64)[:, :2]), float(BANK_E_MAX),
                                         float(r_min), float(BANK_STENCIL_M), float(BANK_RUNOFF_M),
                                         float(fade[0]), float(fade[1])), np.float64)
    step = float(np.median(np.linalg.norm(np.diff(np.asarray(xy)[:, :2], axis=0), axis=1)))
    if step > 0.0:
        b = smooth_profile(b, step, BANK_SIGMA_M)  # pinned ends keep bridge abutments level
    if fade[0]:
        b[0] = 0.0
    if fade[1]:
        b[-1] = 0.0
    return b if np.any(b != 0.0) else None


def road_rank(spec):
    """Higher: the road that keeps its profile at a junction (snap_junctions); a link ranks just under its class."""
    cls = spec.cls or ""
    base = cls.removesuffix("_link")
    r = JUNCTION_RANK.get(base, 0) * 2
    return r - 1 if cls != base else r


def _nearest_on(p, x, y):
    """(distance, height, arc position from p's nearer end) of the point of Plan p's centreline nearest (x, y)."""
    a, b = p.xy[:-1, :2], p.xy[1:, :2]
    d = b - a
    ll = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / ll, 0.0, 1.0)
    q = a + d * t[:, None]
    dist = np.hypot(q[:, 0] - x, q[:, 1] - y)
    i = int(dist.argmin())
    seg = np.sqrt(ll)
    s = float(seg[:i].sum() + seg[i] * t[i])
    return float(dist[i]), float(p.z[i] + (p.z[i + 1] - p.z[i]) * t[i]), min(s, float(seg.sum()) - s)


def couple_neighbours(plans, specs, trend):
    """NEIGHBOUR_* above: plans (in place) running alongside a higher- or equal-ranked one take its height plus the ground's
    fall between them. specs: their Specs; trend(xy (n, 2)) -> the far smoother surface's heights there. Plans leading
    onto a bridge (abut) are left as they are, and are no neighbour to follow either."""
    rank = np.array([road_rank(sp) for sp in specs])
    order = sorted(range(len(plans)), key=lambda j: (-rank[j], j))
    reach = NEIGHBOUR_GAP_M
    boxes = np.array([[p.xy[:, 0].min(), p.xy[:, 1].min(), p.xy[:, 0].max(), p.xy[:, 1].max()] for p in plans],
                     np.float64).reshape(-1, 4)
    cos_par = math.cos(math.radians(NEIGHBOUR_PARALLEL_DEG))
    ends = np.array([[p.xy[0, :2], p.xy[-1, :2]] for p in plans], np.float64).reshape(-1, 2, 2)
    for j in order:
        p = plans[j]
        if len(p.xy) < 2 or p.abut[0] or p.abut[1]:
            continue
        P = np.asarray(p.xy, np.float64)[:, :2]
        g = np.gradient(P, axis=0)
        tdir = g / np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-9)[:, None]
        pad = reach + p.road_half + 20.0
        cand = np.flatnonzero((rank >= rank[j]) & (boxes[:, 0] <= boxes[j, 2] + pad) & (boxes[:, 2] >= boxes[j, 0] - pad)
                              & (boxes[:, 1] <= boxes[j, 3] + pad) & (boxes[:, 3] >= boxes[j, 1] - pad))
        best_w = np.zeros(len(P))
        best_q = np.zeros((len(P), 2))
        best_zq = np.zeros(len(P))
        for k in cand:
            q = plans[k]
            if k == j or len(q.xy) < 2 or q.abut[0] or q.abut[1] or q.rid == p.rid:
                continue
            # (roads that meet at a node are a junction, snap_junctions': a road's two parts either side of a junction, a
            # slip road leaving its motorway)
            if np.hypot(*(ends[j][:, None, :] - ends[k][None, :, :]).reshape(-1, 2).T).min() <= CHAIN_JOIN_M:
                continue
            a, b = q.xy[:-1, :2], q.xy[1:, :2]
            d = b - a
            ll = np.maximum((d * d).sum(1), 1e-12)
            t = np.clip(((P[:, None, 0] - a[None, :, 0]) * d[None, :, 0] + (P[:, None, 1] - a[None, :, 1]) * d[None, :, 1])
                        / ll[None, :], 0.0, 1.0)
            qx = a[None, :, 0] + t * d[None, :, 0]
            qy = a[None, :, 1] + t * d[None, :, 1]
            dist = np.hypot(qx - P[:, None, 0], qy - P[:, None, 1])
            si = dist.argmin(1)
            r_ = np.arange(len(P))
            dmin = dist[r_, si]
            gap = dmin - q.road_half - p.road_half
            sdir = d[si] / np.sqrt(ll[si])[:, None]
            par = np.abs((tdir * sdir).sum(1)) >= cos_par
            u = np.clip((gap - NEIGHBOUR_FULL_M) / max(NEIGHBOUR_GAP_M - NEIGHBOUR_FULL_M, 1e-9), 0.0, 1.0)
            w = (1.0 - u * u * (3.0 - 2.0 * u)) * par * (gap <= NEIGHBOUR_GAP_M)
            w *= 1.0 if rank[k] > rank[j] else 0.5
            zq = q.z[si] + (q.z[si + 1] - q.z[si]) * t[r_, si]
            take = w > best_w
            best_w[take] = w[take]
            best_q[take] = np.c_[qx[r_, si], qy[r_, si]][take]
            best_zq[take] = zq[take]
        on = best_w > 0
        if not on.any():
            continue
        idx = np.flatnonzero(on)
        f = np.asarray(trend(np.vstack([P[idx], best_q[idx]])), np.float64)
        target = best_zq[idx] + (f[:len(idx)] - f[len(idx):])
        diff = target - p.z[idx]
        keep = np.abs(diff) <= NEIGHBOUR_MAX_M
        delta = np.zeros(len(P))
        wt = np.zeros(len(P))
        delta[idx[keep]] = best_w[idx[keep]] * diff[keep]
        wt[idx[keep]] = best_w[idx[keep]]
        if not wt.any():
            continue
        # smoothed along the road (a Gaussian over arc length), as a fraction of the coupled share: it fades out where the
        # neighbour ends instead of stepping
        s_ = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        kern = np.exp(-0.5 * ((s_[:, None] - s_[None, :]) / NEIGHBOUR_SMOOTH_M) ** 2)
        kern /= kern.sum(1, keepdims=True)
        p.z = np.asarray(p.z, np.float64) + kern @ delta


def snap_junctions(plans, specs):
    """A side road's end at a junction takes the height of the road it joins, eased in over
    JUNCTION_TAPER_M along the side road (each road is smoothed on its own: a side road met the main road up to a metre
    off, blended over BLEND_R only: a bump at every turn-off). Plans are changed in place, the higher ranks first, so
    a residential street joining a tertiary road meets it at its already adjusted height. With the coarse easing
    (BNE_COARSE_EASE=1) also the crossings (GitHub #414, Perm: service roads crossing a primary at a shared node stood
    as 4-5 m walls across it): where a road runs over a higher-ranked road's carriageway away from its own ends, it takes
    that road's height there. plans: [Plan], specs: their Specs. In Rust (earthfast.snap_junctions: 15 CPU-s of a
    teleport's road plans in Python, 2026-10-30)."""
    if not plans:
        return
    rank = np.array([road_rank(sp) for sp in specs], np.int64)
    js = _Junctions(plans)
    changed = js.snap(rank)
    # (a road joining one that the hold changed meets it at its new height: snapped again)
    if _hold_junction_corridors(js, rank, changed):
        _hold_junction_corridors(js, rank, js.snap(rank))
    js.store()


class _Junctions:
    """snap_junctions' plans as earthfast's flat arrays, made once for all its passes (each pass made them again:
    0.2-0.4 CPU-s per 16 road tiles), their heights kept here between the passes; store() puts them back on the plans
    changed."""

    def __init__(self, plans):
        import fast
        fast.require(144, "the junctions' heights")
        self.plans = plans
        n = [len(p.xy) for p in plans]
        self.off = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
        self.xy = np.ascontiguousarray(np.concatenate([np.asarray(p.xy, np.float64)[:, :2] for p in plans])
                                       .reshape(-1, 2))
        self.z = np.ascontiguousarray(np.concatenate([np.asarray(p.z, np.float64) for p in plans]))
        self.half = np.array([p.road_half for p in plans], np.float64)
        self.abut = np.array([[bool(p.abut[0]), bool(p.abut[1])] for p in plans], np.uint8).reshape(-1, 2)
        self.boxes = np.ascontiguousarray(np.array([q.bbox for q in plans], np.float64).reshape(-1, 4))
        self.touched = np.zeros(len(plans), bool)

    def snap(self, rank):
        """One pass of earthfast.snap_junctions; the plans it changed (bool array)."""
        import fast
        self.z, changed = fast._ef.snap_junctions(self.xy, self.z, self.off, rank, self.half, self.abut, self.boxes,
                                                  JUNCTION_REACH_M, JUNCTION_END_M, JUNCTION_TAPER_M, JUNCTION_MAX_M,
                                                  bool(COARSE_EASE_ON))
        changed = np.asarray(changed, bool)
        self.touched |= changed
        return changed

    def store(self):
        off = self.off
        for j in np.flatnonzero(self.touched):
            self.plans[j].z = self.z[off[j]:off[j + 1]].copy()


def _hold_junction_corridors(js, rank, changed):
    """A side road's end snapped to a wide road (snap_junctions) holds that road's height, at the nearest point of it,
    across its carriageway and shoulder (JUNCTION_HOLD_PAD_M), then returns to the profile the snap gave it at most
    JUNCTION_HOLD_GRADE steep. The snap eases from the end's first vertex, so on a 13 m trunk the next vertices lay on
    its lanes already 45-90 cm low, and the ground under the lane was blended down to them: a trench across the lane at
    a track's junction (48.83116, 29.33644, the user 2026-11-02; a hole the car took at speed). Only ends the snap
    changed and the road itself reaches (not a bridge abutment, not a gap the snap left as it was); a road shorter than
    three corridors, or one that does not meet its own profile again within its first half, is left as the snap made
    it. A second call changes nothing (the held heights and the limited grade stay). js: _Junctions (its heights changed
    in place); True when any was. In Rust (earthfast.hold_junction_corridors: 1.2 CPU-s per 16 road tiles in numpy)."""
    import fast
    if not changed.any():
        return False
    js.z, held = fast._ef.hold_junction_corridors(js.xy, js.z, js.off, rank, js.half, js.abut, js.boxes,
                                                  np.ascontiguousarray(changed, np.uint8), JUNCTION_REACH_M,
                                                  JUNCTION_HOLD_PAD_M, JUNCTION_HOLD_GRADE)
    held = np.asarray(held, bool)
    js.touched |= held
    return bool(held.any())


# Junction pads (the user, 2026-10-29: "lower ranked roads impacting higher ranked ones ... side streets coming in from a
# sloped hill create bumps and likewise dips on the lower side"; "roads facing each other having more influence ... especially
# when the facing roads are the same class"). grade_points blends every road's flat cross-section by distance alone, so
# inside a junction each road pulls the ground toward its own surface: where they disagree (equal ranks, which never
# snap; a road crossing another with no node; a crossroads on a slope, which no set of flat cross-sections can meet) the
# middle of the junction is a compromise, a bump on one side and a dip on the other. Weighting the blend by rank only
# moves the compromise (and needs the Rust loop changed); here every road is made to agree instead, so the blend's
# weights no longer matter there:
#   - each branch at a node (a road's end; or a road crossing another's centreline away from its ends: two branches)
#     gets an authority, its class's weight (PAD_CLASS_W) times its "through" factor (a branch with another one facing
#     it, PAD_FACE_DEG: a through road; a T's stem keeps PAD_STEM_F), normalised by the node's strongest, ** PAD_GAMMA;
#   - one plane is fitted to the branches' heights at the node and their grades leaving it, weighted by authority (and
#     pulled level by PAD_RIDGE): a T on a hillside is the main road's own grade and height, with next to no tilt
#     across it; a crossroads of equal roads a plane that matches both their grades;
#   - over the pad (each branch out to the other branches' cores + PAD_LANDING_M) every road lies on that plane: its
#     centreline height and its cross slope (Plan.bank: grade_points tilts its core by it), then back to its own
#     profile and cross slope over JUNCTION_TAPER_M / PAD_BANK_TAPER_M.
# The strongest road is all but unchanged (the plane is its own surface); the others warp to it, as real junctions are
# built. BNE_JUNCTION_PAD=1: on (off by default: NEIGHBOUR_ON).
PAD_ON = os.environ.get("BNE_JUNCTION_PAD", "0") == "1"
PAD_CLASS_W = {"motorway": 1.0, "trunk": 0.9, "primary": 0.8, "secondary": 0.65, "tertiary": 0.5,
               "residential": 0.35, "unclassified": 0.35, "living_street": 0.3, "service": 0.2, "track": 0.15}
PAD_OTHER_W = 0.25          # a class not listed
PAD_LINK_F = 0.75           # a slip road: its class's weight times this
PAD_UNBUILT_F = 0.6         # a gravel / dirt road: times this
PAD_FACE_DEG = (120.0, 160.0)   # angle to the most opposite other branch: no through road under the first, fully at it
PAD_STEM_F = 0.35           # a branch with none facing it keeps this share of its class's weight
PAD_GAMMA = 2.0             # authority = (weight / the node's strongest) ** this
PAD_MIN = 0.02
PAD_FIT_POW = 2.0           # the plane fitted with the authorities ** this (the strongest road all but decides it)
PAD_GRADE_M = 15.0          # m along a branch its grade leaving the node is taken over
PAD_FLAT_W = 1.0            # how much a road wants its own cross section across it, against its grade along it
PAD_RIDGE = 0.05            # the plane's tilt pulled to level with this weight (an authority is at most 1)
PAD_MAX_TILT = 0.15         # never steeper (bad data)
PAD_LANDING_M = 3.0         # m past the other branches' cores the pad reaches ...
PAD_LANDING_MINOR_M = 0.5   # ... on a branch weighing less than PAD_LANDING_MIN_W (a track, a driveway)
PAD_LANDING_MIN_W = 0.3
PAD_MIN_SIN = 0.35          # (a branch at a shallow angle: its core reaches along the other at most 1 / this)
PAD_REACH_F = 0.4           # a pad reaches at most this share of the branch (a short part keeps the rest to climb)
PAD_BANK_TAPER_M = 15.0     # m the cross slope returns to the road's own over
PAD_TAPERS_M = (8.0, 12.0, 16.0, 24.0, 32.0)    # m: the shorter eases back to a road's own profile (pad's edge on)
PAD_SINK_M = 0.25           # m a road is let sink under (or stand over) its own profile on the ease, past the pad's step
PAD_CROSS_MAX_M = 1.5       # m: two roads crossing with no node further apart than this are not at grade (left)
PAD_MIN_DELTA_M = 0.01      # m: a smaller change is not made
PAD_GRID_M = 32.0           # m: the crossing search's buckets
PAD_MIN_PART_M = 20.0       # m: a branch shorter than this (to its part's end) is left out of the pad


def pad_class_w(spec):
    """A road's own weight at a junction by its class (PAD_CLASS_W), less for a slip road or an unbuilt one. Pure."""
    cls = spec.cls or ""
    w = PAD_CLASS_W.get(cls.removesuffix("_link"), PAD_OTHER_W)
    if cls.endswith("_link") or getattr(spec, "link", False):
        w *= PAD_LINK_F
    if spec.kind not in roadsmod.BUILT:
        w *= PAD_UNBUILT_F
    return w


def pad_authority(weights, dirs):
    """The authority (0, 1] of each branch at one node: weights (n,) by class (pad_class_w), dirs (n, 2) the way each
    leaves the node. A branch with another facing it (PAD_FACE_DEG; fully only one of its own weight) is through, a stem
    keeps PAD_STEM_F of its weight;
    then each over the node's strongest, ** PAD_GAMMA. Equal through roads all 1. Pure."""
    w = np.asarray(weights, np.float64)
    n = len(w)
    if n < 2:
        return np.ones(n)
    d = np.asarray(dirs, np.float64).reshape(n, 2)
    d = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)[:, None]
    cos = np.clip(d @ d.T, -1.0, 1.0)
    np.fill_diagonal(cos, 1.0)
    lo, hi = PAD_FACE_DEG
    u = np.clip((np.degrees(np.arccos(cos)) - lo) / (hi - lo), 0.0, 1.0)
    # (facing: by angle, and fully only by a branch of its own weight: a street across a primary from a driveway is
    # hardly a through road)
    sim = np.minimum(w[:, None], w[None, :]) / np.maximum(np.maximum(w[:, None], w[None, :]), 1e-12)
    face = (u * u * (3.0 - 2.0 * u) * sim).max(axis=1)
    raw = w * (PAD_STEM_F + (1.0 - PAD_STEM_F) * face)
    top = raw.max()
    if top <= 0:
        return np.ones(n)
    return np.clip((raw / top) ** PAD_GAMMA, PAD_MIN, 1.0)


def pad_plane(z, grades, dirs, auth, cross=None):
    """(z0, g (2,)): the plane z0 + g . (p - node) fitted to the branches' heights z (n,) at the node, their grades (n,)
    along dirs (n, 2) leaving it and their own cross slopes (cross (n,), toward each branch's left; None: level),
    weighted by auth (n,) (a road wants its own grade along it and its own cross section across it, PAD_FLAT_W), the
    tilt pulled level by PAD_RIDGE and capped at PAD_MAX_TILT. Pure."""
    z, gr, a = (np.asarray(v, np.float64) for v in (z, grades, auth))
    cr = np.zeros(len(a)) if cross is None else np.asarray(cross, np.float64)
    d = np.asarray(dirs, np.float64).reshape(-1, 2)
    d = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)[:, None]
    nl = np.c_[-d[:, 1], d[:, 0]]
    z0 = float((a * z).sum() / max(a.sum(), 1e-12))
    # least squares: a_i (g . d_i - grade_i)^2 + PAD_FLAT_W a_i (g . n_i - cross_i)^2 + PAD_RIDGE |g|^2
    M = (a[:, None, None] * (d[:, :, None] * d[:, None, :] + PAD_FLAT_W * nl[:, :, None] * nl[:, None, :])).sum(0) \
        + PAD_RIDGE * np.eye(2)
    g = np.linalg.solve(M, (a[:, None] * (d * gr[:, None] + PAD_FLAT_W * nl * cr[:, None])).sum(0))
    t = float(np.hypot(*g))
    if t > PAD_MAX_TILT:
        g *= PAD_MAX_TILT / t
    return z0, g


def _arcs(p):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.asarray(p.xy, np.float64)[:, :2], axis=0).T))]


def _pad_nodes(plans, rank, boxes):
    """[(node xy (2,), [(plan j, arc at the node, direction +1 / -1)])]: the plans' ends joined within CHAIN_JOIN_M
    (three branches or more, or two of different roads at an angle), and the at-grade crossings of two plans away from
    their ends (two branches each)."""
    ends = []
    for j, p in enumerate(plans):
        if len(p.xy) < 2:
            continue
        s = _arcs(p)
        for e, (arc, sgn) in enumerate(((0.0, 1), (float(s[-1]), -1))):
            if p.abut[e]:
                continue
            ends.append((j, arc, sgn, float(p.xy[-1 if e else 0, 0]), float(p.xy[-1 if e else 0, 1])))
    nodes = []
    if ends:
        xy = np.array([[e[3], e[4]] for e in ends])
        used = np.zeros(len(ends), bool)
        order = np.argsort(xy[:, 0], kind="stable")
        sx = xy[order, 0]
        for ii, i in enumerate(order):
            if used[i]:
                continue
            lo = np.searchsorted(sx, xy[i, 0] - CHAIN_JOIN_M)
            hi = np.searchsorted(sx, xy[i, 0] + CHAIN_JOIN_M, side="right")
            grp = [int(k) for k in order[lo:hi] if not used[k] and math.hypot(*(xy[k] - xy[i])) <= CHAIN_JOIN_M]
            used[grp] = True
            br = [(ends[k][0], ends[k][1], ends[k][2]) for k in grp]
            # (and a road going on through the node in one part: two branches)
            x, y = float(xy[i, 0]), float(xy[i, 1])
            mine = {b[0] for b in br}
            for k in np.flatnonzero((boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3])):
                if int(k) in mine or len(plans[k].xy) < 2:
                    continue
                dist, _zq, from_end = _nearest_on(plans[k], x, y)
                if dist <= CHAIN_JOIN_M and from_end >= JUNCTION_END_M:
                    sk = _arcs(plans[k])
                    A = np.asarray(plans[k].xy, np.float64)[:, :2]
                    dd = A[1:] - A[:-1]
                    ll = np.maximum((dd * dd).sum(1), 1e-12)
                    t = np.clip(((x - A[:-1, 0]) * dd[:, 0] + (y - A[:-1, 1]) * dd[:, 1]) / ll, 0.0, 1.0)
                    m = int(np.argmin(np.hypot(A[:-1, 0] + t * dd[:, 0] - x, A[:-1, 1] + t * dd[:, 1] - y)))
                    at = float(sk[m] + t[m] * (sk[m + 1] - sk[m]))
                    br += [(int(k), at, 1), (int(k), at, -1)]
            if len({plans[b[0]].rid for b in br}) < 2:
                continue        # (one road's own parts: its chain)
            nodes.append((xy[i].copy(), br))
    # crossings with no node: proper intersections of two plans' centrelines, each away from both plans' ends (every
    # segment bucketed on a PAD_GRID_M grid, the pairs within a bucket tested at once)
    segs = [(j, np.asarray(p.xy, np.float64)[:, :2]) for j, p in enumerate(plans) if len(p.xy) >= 2]
    if not segs:
        return nodes
    P0 = np.concatenate([xy[:-1] for _j, xy in segs])
    P1 = np.concatenate([xy[1:] for _j, xy in segs])
    J = np.concatenate([np.full(len(xy) - 1, j) for j, xy in segs])
    K = np.concatenate([np.arange(len(xy) - 1) for _j, xy in segs])
    rid = np.array([p.rid for p in plans])
    lo = np.floor(np.minimum(P0, P1) / PAD_GRID_M).astype(np.int64)
    hi = np.floor(np.maximum(P0, P1) / PAD_GRID_M).astype(np.int64)
    nx_ = hi[:, 0] - lo[:, 0] + 1
    ny_ = hi[:, 1] - lo[:, 1] + 1
    cnt = nx_ * ny_
    sid = np.repeat(np.arange(len(P0)), cnt)
    loc = np.arange(len(sid)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cx = lo[sid, 0] + loc % np.repeat(nx_, cnt)
    cy = lo[sid, 1] + loc // np.repeat(nx_, cnt)
    key = cx * 2000003 + cy
    o = np.argsort(key, kind="stable")
    key, sid = key[o], sid[o]
    start = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    size = np.diff(np.r_[start, len(key)])
    gsize = np.repeat(size, size)
    lidx = np.arange(len(key)) - np.repeat(start, size)
    m = gsize - 1 - lidx                        # partners after each one in its bucket
    ia = np.repeat(np.arange(len(key)), m)
    ib = ia + (np.arange(len(ia)) - np.repeat(np.cumsum(m) - m, m)) + 1
    sa_, sb_ = sid[ia], sid[ib]
    keep = (J[sa_] != J[sb_]) & (rid[J[sa_]] != rid[J[sb_]])
    sa_, sb_ = sa_[keep], sb_[keep]
    if len(sa_):
        pair = np.unique(np.c_[np.minimum(sa_, sb_), np.maximum(sa_, sb_)], axis=0)
        sa_, sb_ = pair[:, 0], pair[:, 1]
        r = P1[sa_] - P0[sa_]
        t_ = P1[sb_] - P0[sb_]
        qp = P0[sb_] - P0[sa_]
        den = r[:, 0] * t_[:, 1] - r[:, 1] * t_[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (qp[:, 0] * t_[:, 1] - qp[:, 1] * t_[:, 0]) / den
            v = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / den
        hit = np.flatnonzero((np.abs(den) > 1e-12) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1))
        arcs = {}
        for h in hit:
            j, k = int(J[sa_[h]]), int(J[sb_[h]])
            for x in (j, k):
                if x not in arcs:
                    arcs[x] = _arcs(plans[x])
            sj, sk = arcs[j], arcs[k]
            i, mm = int(K[sa_[h]]), int(K[sb_[h]])
            a_arc = float(sj[i] + u[h] * (sj[i + 1] - sj[i]))
            b_arc = float(sk[mm] + v[h] * (sk[mm + 1] - sk[mm]))
            if min(a_arc, sj[-1] - a_arc) < JUNCTION_END_M or min(b_arc, sk[-1] - b_arc) < JUNCTION_END_M:
                continue    # (at an end: a node, found above, or a near miss)
            nodes.append((P0[sa_[h]] + u[h] * r[h], [(j, a_arc, 1), (j, a_arc, -1), (k, b_arc, 1), (k, b_arc, -1)]))
    return nodes


def junction_pads(plans, specs, nodes=None):
    """PAD_* above: the plans (in place) at every node lie on one plane over the junction's pad, its height and tilt
    weighted by each branch's authority (rank, and whether a branch faces it). plans: [Plan], specs: their Specs;
    nodes: a previous call's (the same plans' lines, other heights: the pass after the dips). Returns the nodes."""
    if not PAD_ON or len(plans) < 2:
        return nodes
    rank = np.array([road_rank(sp) for sp in specs])
    if nodes is None:
        boxes = np.array([q.bbox for q in plans], np.float64).reshape(-1, 4)
        nodes = _pad_nodes(plans, rank, boxes)
    # the strongest junctions first: a short part between two takes the stronger one's pad
    nodes.sort(key=lambda nd: (-max(rank[b[0]] for b in nd[1]), float(nd[0][0]), float(nd[0][1])))
    arcs = {}
    for node, br in nodes:
        for j, _a, _s in br:
            if j not in arcs:
                arcs[j] = _arcs(plans[j])
        crossing = all(0.0 < arc < float(arcs[j][-1]) for j, arc, _s in br)
        info = []
        for j, arc, sgn in br:
            p = plans[j]
            s = arcs.get(j)
            if s is None:
                s = arcs[j] = _arcs(p)
            avail = float(s[-1] - arc) if sgn > 0 else float(arc)
            if avail < PAD_MIN_PART_M:
                continue            # (a median's crossover, a short link: its two ends' landings, snap_junctions)
            look = min(CHAIN_LOOK_M, avail)
            px_ = np.interp(arc + sgn * look, s, p.xy[:, 0]) - np.interp(arc, s, p.xy[:, 0])
            py_ = np.interp(arc + sgn * look, s, p.xy[:, 1]) - np.interp(arc, s, p.xy[:, 1])
            gl = min(PAD_GRADE_M, avail)
            z_n = float(np.interp(arc, s, p.z))
            grade = (float(np.interp(arc + sgn * gl, s, p.z)) - z_n) / gl
            b_n = 0.0 if p.bank is None else float(np.interp(arc, s, p.bank)) * sgn     # (toward the branch's left)
            info.append((j, arc, sgn, avail, np.array([px_, py_]), z_n, grade, pad_class_w(specs[j]), b_n))
        if len(info) < 2 or len({plans[i[0]].rid for i in info}) < 2:
            continue
        if len(info) == 2 and float(info[0][4] @ info[1][4]) < -math.cos(math.radians(PAD_FACE_DEG[0])) * \
                math.hypot(*info[0][4]) * math.hypot(*info[1][4]):
            continue                # (a road going on into its next part: its chain's)
        zs = np.array([i[5] for i in info])
        if zs.max() - zs.min() > (PAD_CROSS_MAX_M if crossing else JUNCTION_MAX_M):
            continue                # (a ramp to an overpass, bad data, two roads crossing at two levels: as they were)
        dirs = np.array([i[4] for i in info])
        dirs /= np.maximum(np.hypot(dirs[:, 0], dirs[:, 1]), 1e-12)[:, None]
        auth = pad_authority([i[7] for i in info], dirs)
        # each branch's tangent at the node: through a road (a branch facing it, PAD_FACE_DEG[1]) the central difference
        # of the two (a crest's or a sag's one-sided grades both lean the same way), else its own grade leaving it
        gr = np.array([i[6] for i in info])
        cosd = dirs @ dirs.T
        np.fill_diagonal(cosd, 1.0)
        tang = gr.copy()
        lim = -math.cos(math.radians(PAD_FACE_DEG[1]))
        for n in range(len(info)):
            m = int(np.argmin(cosd[n]))
            if cosd[n, m] <= lim:
                tang[n] = 0.5 * (gr[n] - gr[m])
        z0, g = pad_plane(zs, tang, dirs, auth ** PAD_FIT_POW, [i[8] for i in info])
        nx, ny = float(node[0]), float(node[1])
        for n, (j, arc, sgn, avail, _d, z_n, _g, _w, _b) in enumerate(info):
            p = plans[j]
            s = arcs[j]
            # the pad along this branch: out to the other branches' cores (at their angle) and PAD_LANDING_M past
            # (a landing past them for a street's traffic to stop on; a track or a driveway climbs from the other's
            # carriageway edge, over its verge)
            street = pad_class_w(specs[j]) >= PAD_LANDING_MIN_W
            land = PAD_LANDING_M if street else PAD_LANDING_MINOR_M
            reach = 0.0
            for m, other in enumerate(info):
                if other[0] == j:
                    continue
                sin = abs(dirs[n, 0] * dirs[m, 1] - dirs[n, 1] * dirs[m, 0])
                if float(dirs[n] @ dirs[m]) > 0 or sin >= PAD_MIN_SIN:      # (not the branch facing it)
                    q = plans[other[0]]
                    reach = max(reach, (q.half if street else q.road_half) / max(sin, PAD_MIN_SIN))
            reach = min(reach + land, PAD_REACH_F * avail)
            d_all = (s - arc) * sgn
            win = np.flatnonzero((d_all >= -1e-9) & (d_all <= reach + JUNCTION_TAPER_M))    # (all it can change)
            if not len(win):
                continue
            d = d_all[win]
            xy = np.asarray(p.xy, np.float64)[:, :2]
            zw = np.asarray(p.z, np.float64)[win]
            # the plane, less the branch's own tangent: added to its profile, its height and tangent at the node become
            # the plane's and its own vertical curve stays (a road the plane is made from is all but unchanged)
            # (by its authority: a weak branch, a side road whose own end is the landing's step, lies on the plane itself)
            flat = z0 + g[0] * (xy[win, 0] - nx) + g[1] * (xy[win, 1] - ny)
            plane = flat + auth[n] * (zw - (z_n + tang[n] * d))
            tan = np.gradient(xy, axis=0)[win] if len(xy) > 1 else np.zeros((len(win), 2))
            tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-12)[:, None]
            bank_plane = g[0] * -tan[:, 1] + g[1] * tan[:, 0]      # (m per m toward the plan's left)
            own_bank = np.zeros(len(win)) if p.bank is None else np.asarray(p.bank, np.float64)[win]
            pad = d <= reach
            if not pad.any():
                continue
            z = zw.copy()
            bank = own_bank.copy()
            z[pad] = plane[pad]
            bank[pad] = bank_plane[pad]
            # back to its own: from the plane (carried on past the pad) to its own profile, the share eased out
            # (cosine): the grade runs on from the pad's without a break (easing the pad edge's height difference
            # alone kept the pad's level to its edge and its own grade right after: a 5-7 % kink there on a hillside)
            # (a track's or a driveway's as short as it must be: a side road leaving up a steep bank, held to the pad's level over a long ease,
            # ran in a trench 2.4 m deep, its own grade 30 %, the user 2026-10-29 at -32.0951, 116.03866: the longest
            # ease of PAD_TAPERS_M, up to JUNCTION_TAPER_M, that sinks it (or raises it) at most PAD_SINK_M more than the
            # pad edge's step faded out would, as a real one's ramp is: short and sharp on a bank, long on a gentle slope)
            de = float(d[pad].max())
            out = d > de
            k_e = int(np.argmax(np.where(pad, d, -np.inf)))
            step_ = abs(float(plane[k_e] - zw[k_e]))     # (the pad edge's own step: kept, whatever the ease)
            longest = max(min(JUNCTION_TAPER_M, 0.5 * avail - de, ease_length(step_, not street)), VERTEX_STEP)
            # (a street keeps its long ease: a junction's grade break is what its drivers feel)
            for taper in [longest] + ([] if street else sorted((t for t in PAD_TAPERS_M if t < longest), reverse=True)):
                w = 0.5 * (1.0 + np.cos(np.pi * np.clip((d - de) / taper, 0.0, 1.0)))
                if not out.any() or float((np.abs(w[out] * (plane[out] - zw[out])) - w[out] * step_).max()) <= PAD_SINK_M:
                    break
            btaper = max(min(PAD_BANK_TAPER_M, taper), VERTEX_STEP)
            wb = 0.5 * (1.0 + np.cos(np.pi * np.clip((d - de) / btaper, 0.0, 1.0)))
            z[out] += w[out] * (plane[out] - z[out])
            bank[out] += wb[out] * (bank_plane[out] - bank[out])
            if np.abs(z - zw).max() < PAD_MIN_DELTA_M and np.abs(bank - own_bank).max() < 1e-3:
                continue
            znew = np.asarray(p.z, np.float64).copy()
            znew[win] = z
            p.z = znew
            if p.bank is not None or np.abs(bank).max() > 1e-4:
                bnew = np.zeros(len(xy)) if p.bank is None else np.asarray(p.bank, np.float64).copy()
                bnew[win] = bank
                p.bank = bnew
    return nodes


LEG_RID_SHIFT = 44           # PlanPack: a pass road's leg (between hairpins) as rid + leg << this (rids < 2^41; earthfast's
                             # LEG_ROAD_MASK alike)
HAIRPIN_FOLD_RATIO = 0.5      # hairpin_cores: a point of the road's own line this much nearer than along it is another leg


def hairpin_cores(plans, specs, frame):
    """The pass roads' plans among plans ([(fid, Plan)], specs alike) widened, in place, round their widened hairpins
    (roads.hairpin_extra: the drawn road's own widening, on the plan's line): Plan.widen, their cuttings' walls and tops
    moved out by as much. A plan of its own over each hairpin (wider all along it, its ends held at their heights)
    reached the hairpin's other leg and stepped the lanes by up to 18 cm (the Stelvio; the user, 2026-11-01: "Fix the
    bumps"). Returns how many were widened."""
    n = 0
    for (_fid, p), sp in zip(plans, specs):
        if sp is None or not getattr(sp, "mountain", False) or len(p.xy) < 3:
            continue
        ex = roadsmod.hairpin_extra(frame, sp, np.asarray(p.xy, np.float64)[:, :2])
        if ex is None or float(np.max(ex)) < 0.05:
            continue
        ex = np.maximum(np.asarray(ex, np.float64), 0.0)
        # (never over half way to its own other leg: a point of the plan nearer than FOLD_RATIO of the way along to it,
        # across a hairpin's narrow middle; there the wider cores met and blended the legs' heights, a 22 cm dip)
        rows = np.nonzero(ex > 0.0)[0]
        import fast
        fast.require(126, "fold_clearance")
        room = np.asarray(fast._ef.fold_clearance(np.ascontiguousarray(np.asarray(p.xy, np.float64)[:, :2]),
                                                  rows.astype(np.int64), HAIRPIN_FOLD_RATIO)) / 2.0 - p.half
        ex[rows] = np.minimum(ex[rows], np.maximum(room, 0.0))
        if float(ex.max()) < 0.05:
            continue
        set_widen(p, ex)
        if p.cut is not None:
            seg = np.maximum(ex[:-1], ex[1:])
            c = np.asarray(p.cut, np.float64).copy()
            for col in (0, 2, 3, 5):        # (the walls' and the tops' offsets from the centre line)
                c[:, col] += seg
            p.cut = c
        n += 1
    return n


def set_widen(p, widen):
    """Plan p's widen (hairpin_cores; None: none), its box grown to reach as far."""
    p.widen = widen
    if widen is not None and len(widen):
        pad = p.half + float(np.max(widen)) + T_MAX
        xy = np.asarray(p.xy)
        p.bbox = (xy[:, 0].min() - pad, xy[:, 1].min() - pad, xy[:, 0].max() + pad, xy[:, 1].max() + pad)


def shoulder(spec):
    """m of flat ground beyond the carriageway edge, each side. Like the core width, never from the land
    cover: a road class that gets sidewalks in towns keeps a sidewalk-wide verge everywhere."""
    if spec.kind not in roadsmod.BUILT:
        return 0.0
    if getattr(spec, "mountain", False):        # (a pass road between its wall and its parapet: roads._mountain_roads)
        return MOUNTAIN_SHOULDER_M
    if spec.cls in roadsmod.SIDEWALK_CLASSES or (spec.cls == "trunk" and (not spec.oneway
                                                                          or roadsmod._arterial(spec))):
        return spec.walk + SIDEWALK_MARGIN_M
    return SHOULDER_M


def _twin_trimmed(secs):
    """secs ([(fid, spec, xy, z, kind)]) with the covered sections of twin tunnels (roads.twin_tunnels: side by side,
    one tube round both) cut to the stretch they share: one twin's portal a few metres deeper than the other's, the
    other's ground kept as it is over it (TUNNEL_KEEP_M wide) left the deeper one's approach on the hill there, its road
    mesh a wedge up into their shared mouth (Réunion's Route des Tamarins, GitHub #84). Pure."""
    cov = [n for n, sec in enumerate(secs) if sec[4] and len(sec[2]) >= 2]
    if len(cov) < 2:
        return secs
    parts = [(secs[n][0], secs[n][1], np.asarray(secs[n][2], np.float64), np.asarray(secs[n][3], np.float64), 0.0)
             for n in cov]
    out = list(secs)
    for i, j, _mid, _w in roadsmod.twin_tunnels(parts):
        for a, b in ((i, j), (j, i)):
            xa, xb = parts[a][2], parts[b][2]
            ub = xb[-1] - xb[0]
            ub = ub / max(float(np.hypot(*ub)), 1e-9)
            t = (xa - xb[0]) @ ub                   # along b, from its start
            keep = (t >= 0.0) & (t <= float((xb[-1] - xb[0]) @ ub))
            if keep.sum() >= 2 and not keep.all():
                fid, spec, xy, z, kind = secs[cov[a]]
                k = np.flatnonzero(keep)
                out[cov[a]] = (fid, spec, np.asarray(xy)[k[0]:k[-1] + 1], np.asarray(z)[k[0]:k[-1] + 1], kind)
    return out


_UNSET = object()


PLANNERS_KEPT = 6               # planning regions a PlanStore keeps in memory (~30 km each)
MAPPED_MAX = 256                # cells' plans mapped into an anchor's world kept (Grader._cell_plans)


def road_id(fid):
    """A stable integer id of a map feature (grade_points: a road never gives way to itself, though each map cell
    plans its own stretch of it: the Stelvio Pass's hairpins sank under their own other stretches, 2026-10-21)."""
    import zlib
    return zlib.crc32(str(fid).encode()) & 0x7FFFFFFF


DUPE_XY_M = 0.05             # consecutive plan vertices this close are one vertex (a repeated OSM node, a trim
                            # cut at a vertex, a junction snap moved an end centimetres to its junction): a centimetre
                            # segment with metres of height divides by its length downstream (seen as a 430 m spike
                            # and a pit beside it from 1.5 cm for 5.8 m)


def _dedupe_plan(p):
    """A plan without its zero-length segments (None: nothing left), its other arrays carried along (a run of
    duplicates keeps the last height: a junction snap's). The input plan is never changed."""
    xy = np.asarray(p.xy, np.float64)[:, :2]
    n = len(xy)
    if n < 2:
        return None
    if bool((np.hypot(*np.diff(xy, axis=0).T) > DUPE_XY_M).all()):
        return p
    keep = [0]
    for i in range(1, n):
        if float(np.hypot(*(xy[i] - xy[keep[-1]]))) > DUPE_XY_M:
            keep.append(i)
        else:
            keep[-1] = i
    keep = np.array(keep, np.int64)
    if len(keep) < 2:
        return None
    seg_ok = np.hypot(*np.diff(xy, axis=0).T) > DUPE_XY_M
    q = Plan(np.ascontiguousarray(xy[keep]), np.ascontiguousarray(np.asarray(p.z, np.float64)[keep]), p.half,
             p.abut, None if p.cut is None else np.ascontiguousarray(np.asarray(p.cut, np.float64)[seg_ok]),
             p.road_half)
    q.rid, q.max_wall = getattr(p, "rid", -1), getattr(p, "max_wall", q.max_wall)
    bk = getattr(p, "bank", None)
    q.bank = None if bk is None else np.ascontiguousarray(np.asarray(bk, np.float64)[keep])
    q.steep = bool(getattr(p, "steep", False))
    wd = getattr(p, "widen", None)
    q.widen = None if wd is None else np.ascontiguousarray(np.asarray(wd, np.float64)[keep])
    return q


class PlanPack:
    """Plans (a list, in grade_points' order) as grade_points takes them: every segment's arrays concatenated once, with
    each plan's box; a call keeps the segments of the plans whose boxes meet its points', in the same order (built per
    call, the plans' Python loop cost a Canton tile set ~1 s, 2026-10-25)."""

    def __init__(self, plans):
        rows = [(k, q) for k, p in enumerate(plans) if (q := _dedupe_plan(p)) is not None and len(q.xy) >= 2]
        self.n = len(rows)
        if not self.n:
            return
        f64 = np.float64
        npts = np.array([len(p.xy) for _k, p in rows], np.int64)
        nseg = npts - 1
        xy = np.concatenate([np.asarray(p.xy, f64)[:, :2] for _k, p in rows])
        z = np.concatenate([np.asarray(p.z, f64) for _k, p in rows])
        last = np.cumsum(npts) - 1
        ia = np.delete(np.arange(len(xy)), last)            # each segment's first point (not a plan's last point)
        self.a, self.b = np.ascontiguousarray(xy[ia]), np.ascontiguousarray(xy[ia + 1])
        self.za, self.zb = np.ascontiguousarray(z[ia]), np.ascontiguousarray(z[ia + 1])
        bk = np.concatenate([np.asarray(p.bank, f64) if getattr(p, "bank", None) is not None else np.zeros(len(p.xy))
                             for _k, p in rows])
        self.ba, self.bb = np.ascontiguousarray(bk[ia]), np.ascontiguousarray(bk[ia + 1])
        self.h = np.repeat(np.array([p.half for _k, p in rows], f64), nseg)
        self.rh = np.repeat(np.array([getattr(p, "road_half", p.half) for _k, p in rows], f64), nseg)
        # (a plan widened in places, Plan.widen: each segment by the more of its ends')
        wd = np.concatenate([np.asarray(p.widen, f64) if getattr(p, "widen", None) is not None else np.zeros(len(p.xy))
                             for _k, p in rows])
        if wd.any():
            ws = np.maximum(wd[ia], wd[ia + 1])
            self.h = self.h + ws
            self.rh = self.rh + ws
        rids = [getattr(p, "rid", -1) for _k, p in rows]
        self.rid = np.repeat(np.array([r if r is not None and r >= 0 else 2 ** 40 + k
                                       for (k, _p), r in zip(rows, rids)], np.int64), nseg)
        s0 = np.r_[0, np.cumsum(nseg)]
        # a pass road's legs between its hairpins' apexes (roads.hairpin_apex_rows) count as roads of their own: the
        # one leg's ground (its wall's strip, its sides) gives way over the other's carriageway at another level as
        # another road's does (grade_points), not the same road's own (a leg's strip behind its wall reached the leg
        # above's lane across a hairpin's narrow middle: a 30 cm step in it, the Stelvio, 2026-11-01)
        for n, (_k, p) in enumerate(rows):
            if getattr(p, "steep", False) and nseg[n] >= 2:
                apex = roadsmod.hairpin_apex_rows(p.xy)
                if apex:
                    leg = np.searchsorted(np.asarray(apex), np.arange(nseg[n]), side="right")
                    self.rid[s0[n]:s0[n + 1]] += leg.astype(np.int64) << LEG_RID_SHIFT
        ab = np.zeros(int(s0[-1]), bool)
        # per segment near a bridge abutment: the distance along its road from its start back to a start abutment
        # (column 0) and from its end on to an end abutment (column 1), NaN if none: the ground past the abutment,
        # over the deck, falls away at ABUT_UNDER_GRADE (grade_points), whichever of its road's segments reaches it
        abd = np.full((int(s0[-1]), 2), np.nan)
        cut = np.full((int(s0[-1]), 6), np.nan)
        for n, (_k, p) in enumerate(rows):
            abut = getattr(p, "abut", (False, False))
            if (abut[0] or abut[1]) and s0[n + 1] > s0[n]:
                arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p.xy[:, :2], axis=0).T))]
                if abut[0]:
                    near = arc[:-1] < ABUT_NEAR_M
                    ab[s0[n]:s0[n + 1]] |= near
                    abd[s0[n]:s0[n + 1], 0] = np.where(near, arc[:-1], np.nan)
                if abut[1]:
                    near = arc[-1] - arc[1:] < ABUT_NEAR_M
                    ab[s0[n]:s0[n + 1]] |= near
                    abd[s0[n]:s0[n + 1], 1] = np.where(near, arc[-1] - arc[1:], np.nan)
            c = getattr(p, "cut", None)
            if c is not None:
                cut[s0[n]:s0[n + 1]] = c
        self.ab = ab.astype(np.uint8)
        self.abd = abd
        self.cut = cut
        self.st = np.repeat(np.array([bool(getattr(p, "steep", False)) for _k, p in rows], np.uint8), nseg)
        # each segment's link: bit 0 its plan goes on before it, bit 1 after it (grade_points' EXTRAP_M); a plan's end
        # where another plan's end lies and the road carries straight on through it (outward directions within
        # LINK_TURN_DEG of opposite) goes on too: one road planned per map cell, a road renamed at a point
        first = np.zeros(int(nseg.sum()), bool)
        last = np.zeros(int(nseg.sum()), bool)
        ends = np.cumsum(nseg)
        first[(ends - nseg)[nseg > 0]] = True
        last[(ends - 1)[nseg > 0]] = True
        at = {}
        for n, (_k, p) in enumerate(rows):
            if nseg[n] <= 0:
                continue
            P = np.asarray(p.xy, f64)[:, :2]
            for e, (q, nb) in enumerate(((P[0], P[1]), (P[-1], P[-2]))):
                d = q - nb
                ln = float(np.hypot(*d))
                if ln > 1e-9:
                    at.setdefault((round(float(q[0]), 1), round(float(q[1]), 1)), []).append((n, e, d / ln))
        cos_min = -math.cos(math.radians(LINK_TURN_DEG))
        for group in at.values():
            if len(group) < 2:
                continue
            for n, e, d in group:
                if any(m != n and float(d @ d2) <= cos_min for m, _e2, d2 in group):
                    if e == 0:
                        first[ends[n] - nseg[n]] = False
                    else:
                        last[ends[n] - 1] = False
        self.lk = ((~first).astype(np.uint8) | ((~last).astype(np.uint8) << 1)).astype(np.uint8)
        self.box = np.array([p.bbox for _k, p in rows], f64).reshape(-1, 4)
        self.of = np.repeat(np.arange(self.n), nseg)

    def near(self, x0, y0, x1, y1):
        """The segment arrays of the plans whose boxes meet (x0, y0, x1, y1), or None."""
        if not self.n:
            return None
        bx = self.box
        keep = ~((bx[:, 2] < x0) | (bx[:, 0] > x1) | (bx[:, 3] < y0) | (bx[:, 1] > y1))
        if not keep.any():
            return None
        if keep.all():
            return self.a, self.b, self.za, self.zb, self.h, self.rh, self.rid, self.ab, self.cut, self.ba, self.bb, self.st, \
                self.lk, self.abd
        sk = keep[self.of]
        return tuple(np.ascontiguousarray(v[sk]) for v in (self.a, self.b, self.za, self.zb, self.h, self.rh, self.rid,
                                                           self.ab, self.cut, self.ba, self.bb, self.st, self.lk,
                                                           self.abd))


def grade_points(px, py, base, plans, lift=0.0):
    """base at the points (px, py) graded under the plans (see the module doc; a list or a PlanPack of them), by the
    Rust loop (native/earthfast grade_points: a Python loop per segment was the server's largest CPU cost while driving,
    2026-10-12). Pure."""
    px = np.asarray(px, np.float64)
    py = np.asarray(py, np.float64)
    base = np.asarray(base, np.float64)
    out = base.copy()
    if not len(px):
        return out
    pack = plans if isinstance(plans, PlanPack) else PlanPack(plans)
    # every segment of the plans near the points: ends, end heights, reach (core half width + T_MAX)
    got = pack.near(px.min(), py.min(), px.max(), py.max())
    if got is None:
        return out
    A, B, ZA, ZB, H, RH, RID, AB, CUT, BA, BB, ST, LK, ABD = got
    import fast
    fast.require(152, "grade_points")        # (the Rust loop: native/earthfast; its segments' links: 125, yield_ramp: 126,
                                             # legs blend over each other's bands: 133, a pass road's core
                                             # given way to whole: 134)
    f64 = np.float64
    return fast._ef.grade_points(np.ascontiguousarray(px, f64), np.ascontiguousarray(py, f64),
                                 np.ascontiguousarray(base, f64), A, B, ZA, ZB, H, RH, RID, AB, CUT, BA, BB, ST, LK,
                                 float(EXTRAP_M), float(lift), float(SLOPE_RUN), float(T_MIN), float(T_MAX), float(STEEP_RUN),
                                 float(STEEP_T_MIN), float(BLEND_R), float(BUCKET),
                                 float(WALL_CUT_M), float(WALL_T_M), float(WALL_FLAT_M), float(WALL_AHEAD_M),
                                 bool(SEP_ON), float(SEP_FROM_M), float(SEP_FULL_M), float(CORE_FADE_M),
                                 float(SIDE_FADE_M), float(SHARP_STEP_M), float(YIELD_RAMP_M), float(ABUT_UNDER_GRADE),
                                 np.ascontiguousarray(ABD, f64))


def bridge_chain(line, all_bridges):
    """The bridge parts joined to line through their ends (within 0.75 m), line itself excluded. Pure."""
    line = np.asarray(line, np.float64)[:, :2]
    parts = [np.asarray(b, np.float64)[:, :2] for b in all_bridges]
    parts = [b for b in parts if not (len(b) == len(line) and np.array_equal(b, line))]
    if not parts:
        return []
    first = np.array([b[0] for b in parts])
    last = np.array([b[-1] for b in parts])
    out, frontier, used = [], [line[0], line[-1]], np.zeros(len(parts), bool)
    while frontier:
        e = frontier.pop()
        d = np.minimum(np.hypot(first[:, 0] - e[0], first[:, 1] - e[1]), np.hypot(last[:, 0] - e[0], last[:, 1] - e[1]))
        for k in np.flatnonzero((d < 0.75) & ~used):    # (in the parts' order, as a loop over them would)
            used[k] = True
            out.append(parts[k])
            frontier += [parts[k][0], parts[k][-1]]
    return out


def bridge_chain_ends_free(line, all_bridges, xy):
    """Whether the end of bridge line nearest xy is an abutment: no other bridge part ends within 0.75 m. Pure."""
    line = np.asarray(line, np.float64)
    end = line[0, :2] if np.hypot(*(line[0, :2] - xy[:2])) <= np.hypot(*(line[-1, :2] - xy[:2])) else line[-1, :2]
    for o in all_bridges:
        o = np.asarray(o, np.float64)
        if len(o) == len(line) and np.array_equal(o, line):
            continue
        if min(np.hypot(*(o[0, :2] - end)), np.hypot(*(o[-1, :2] - end))) < 0.75:
            return False
    return True


def bridge_lifts(line, all_bridges, ground_lines, z_at, chain=None, rail_lines=()):
    """[(end xy (2,), lift m)] of a bridge part (line (k, 2) ENU m): at each of its ends that no other bridge part
    continues from (an abutment), how far the ground there must be raised for the deck to clear every road crossing
    under it (roads.DECK_ROAD_CLEAR + APPROACH_MARGIN_M over the road, full height within APPROACH_FLAT_M of the end,
    then falling at roads.DECK_GRADE), within [APPROACH_MIN_LIFT, APPROACH_MAX_LIFT]. Roads crossing within
    APPROACH_MIN_CROSS_M of either end meet it at grade and do not count; railways (rail_lines) always count (a track
    never meets a road deck at grade: the deck ended 3 m past one at Vitebsk, GitHub #324). z_at(xy (n, 2)) -> ground
    m. Pure given z_at."""
    line = np.asarray(line, np.float64)[:, :2]
    if len(line) < 2:
        return []
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
    # (a short part, or one no road crosses under, still raises its free end for the roads under the rest of its chain:
    # the deck carried a few metres onto the road joined at its end (roads._skew_decks) left that end unraised and the
    # whole deck ~4.5 m low over BR-343 at Teresina, GitHub #306)
    if s[-1] < 2 * APPROACH_MIN_CROSS_M and not chain:
        return []
    cr = roadsmod._crossings([line], ground_lines)
    nr = len(cr)
    if rail_lines:
        cr = cr + roadsmod._crossings([line], list(rail_lines))
    if not cr and not chain:
        return []
    zu = z_at(np.array([c[2] for c in cr], np.float64).reshape(-1, 2)) if cr else np.zeros(0)
    arcs = np.array([c[1] for c in cr], np.float64)

    def joined_at(end, own=None):
        own = line if own is None else own
        for other in all_bridges:
            if other is own or (len(other) == len(own) and np.array_equal(np.asarray(other)[:, :2], own[:, :2])):
                continue
            o = np.asarray(other, np.float64)
            if min(np.hypot(*(o[0, :2] - end)), np.hypot(*(o[-1, :2] - end))) < 0.75:
                return True
        return False
    joined = (joined_at(line[0]), joined_at(line[-1]))
    # (not a junction at an abutment; at a joint with the chain's next part the crossing is mid-bridge: Commerce
    # Street's bridge over I-95 in Springfield VA, split 2 m from where the motorway passes under, GitHub #138)
    rail = np.arange(len(cr)) >= nr
    under = ((arcs >= APPROACH_MIN_CROSS_M) | joined[0]) & ((arcs <= s[-1] - APPROACH_MIN_CROSS_M) | joined[1]) | rail
    zu, arcs, rail = zu[under], arcs[under], rail[under]
    pts = np.array([c[2] for c in cr], np.float64)[under].reshape(-1, 2)
    # (the crossings under the rest of its chain, by their distance from the abutment: Overture splits a bridge where
    # other ways meet it, and the road under it may pass under the next part, GitHub #138)
    # (a road meeting a part's free end is a junction at grade there, as for the line itself: the trunk's own next
    # carriageway at the far abutment of its chain counted as a road under it and raised a residential bridge joined
    # to the chain 9.7 m, the trunk deck climbing 10 m in 20 m to it, US-24 near Buena Vista CO, 2026-10-27)
    more = []
    for part in (chain or ()):
        part = np.asarray(part, np.float64)[:, :2]
        sp = float(np.hypot(*np.diff(part, axis=0).T).sum())
        pj = (joined_at(part[0], part), joined_at(part[-1], part))
        for c in roadsmod._crossings([part], list(ground_lines)):
            a = float(c[1])
            if (a >= APPROACH_MIN_CROSS_M or pj[0]) and (a <= sp - APPROACH_MIN_CROSS_M or pj[1]):
                more.append(np.asarray(c[2], np.float64))
        for c in roadsmod._crossings([part], list(rail_lines)) if rail_lines else ():
            more.append(np.asarray(c[2], np.float64))
    if more:
        mp = np.array(more, np.float64)
        zu = np.r_[zu, z_at(mp)]
        pts = np.r_[pts, mp]
        rail = np.r_[rail, np.zeros(len(mp), bool)]
    if not len(zu):
        return []
    out = []
    for (end, at), jn in zip(((line[0], 0.0), (line[-1], s[-1])), joined):
        if jn:
            continue
        d = np.r_[np.abs(arcs - at), np.hypot(*(pts[len(arcs):] - end).T)] if more else np.abs(arcs - at)
        far = (d >= APPROACH_MIN_CROSS_M) | rail     # (a road meeting the abutment itself: a junction at grade)
        if not far.any():
            continue
        d, zu_e = d[far], zu[far]
        need = zu_e + roadsmod.deck_road_clear(roadsmod.deck_depth(roadsmod.geom_span([line] + list(chain or ()))))             + APPROACH_MARGIN_M \
            - roadsmod.DECK_GRADE * np.maximum(d - APPROACH_FLAT_M, 0.0)
        lift = float(need.max() - z_at(end[None])[0])
        if lift >= APPROACH_MIN_LIFT:
            out.append((np.asarray(end, np.float64), min(lift, APPROACH_MAX_LIFT)))
    return out


class _Decks(list):
    """deck_capped's decks [(line (k, 2), half width)] with box (n, 4): each line's x0, y0, x1, y1; and flat(): every
    line's segments, made once (deck_capped tests every near deck's at once)."""

    def __init__(self, decks):
        super().__init__(decks)
        self.box = np.array([(*np.asarray(ln, np.float64)[:, :2].min(0), *np.asarray(ln, np.float64)[:, :2].max(0))
                             for ln, _h in self], np.float64).reshape(-1, 4)

    def flat(self):
        """Every line's segments in one: (off (n + 1,): line k's rows off[k]..off[k + 1] (none without a segment),
        starts (m, 2), vectors (m, 2), lengths (m,), each line's half width (n,)), made once (deck_capped's native
        test: each line's own differences and np.hypot, elementwise)."""
        got = getattr(self, "_flat", None)
        if got is None:
            lines = [np.asarray(ln, np.float64)[:, :2] for ln, _h in self]
            have = [b for b in lines if len(b) >= 2]
            off = np.r_[0, np.cumsum([max(len(b) - 1, 0) for b in lines])].astype(np.int64)
            if have:
                pts = np.concatenate(have)
                d = np.diff(pts, axis=0)
                keep = np.ones(len(d), bool)
                keep[np.cumsum([len(b) for b in have])[:-1] - 1] = False      # (from one line's end to the next)
                q, s = np.ascontiguousarray(pts[:-1][keep]), np.ascontiguousarray(d[keep])
                lb = np.hypot(*s.T)
            else:
                q, s, lb = np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
            got = self._flat = (off, q, s, lb, np.array([float(h) for _ln, h in self], np.float64))
        return got


def deck_capped(pts, z, decks):
    """z (n,), the ground at points pts (n, 2) along a road in order, capped where decks [(line (k, 2), half width)]
    cross over it (UNDER_DECK_*): from UNDER_DECK_SIDE_M past the deck's sides, at most the straight line between the
    ground there. A crossing within APPROACH_MIN_CROSS_M of a deck's end is a junction at grade and is left. Returns z
    itself when nothing is capped. Pure. (native underdeck.rs: every near deck's segments against the road's in one
    call; numpy per road part was ~half of plan_roads on a dense tile, 2026-11-08)"""
    if not decks or len(pts) < 2:
        return z
    pts = np.ascontiguousarray(np.asarray(pts, np.float64)[:, :2])
    test_box = isinstance(decks, _Decks)    # (every deck's box at once)
    if not test_box:                        # (a plain list: each deck's own test, the near ones' segments made here)
        lo_p, hi_p = pts.min(0) - UNDER_DECK_MAX_M, pts.max(0) + UNDER_DECK_MAX_M
        lines = [(np.asarray(ln, np.float64)[:, :2], h) for ln, h in decks]
        decks = _Decks([(ln, h) for ln, h in lines
                        if len(ln) >= 2 and not ((ln.max(0) < lo_p).any() or (ln.min(0) > hi_p).any())])
        if not len(decks):
            return z
    r = np.diff(pts, axis=0)
    la = np.hypot(*r.T)
    import fast
    code, out = fast.deck_capped(pts, z, la, decks.box, test_box, decks.flat(), UNDER_DECK_MAX_M,
                                 UNDER_DECK_SIDE_M, APPROACH_MIN_CROSS_M)
    return z if code == 0 else np.asarray(z, np.float64) if code == 1 else out


def fill_abutment(z, at_end, step=PROFILE_STEP):
    """The raw profile z (a sample every step m) of a road leading onto a bridge at its first (at_end False) or last
    sample, its last ABUT_NEAR_M filled up to the straight grade of the stretch from there to ABUT_FAR_M where they
    fall at least ABUT_STEEP more steeply than it toward the bridge (ABUT_FILL_ON). Returns z itself when nothing is
    filled. Pure."""
    z = np.asarray(z, np.float64)
    p = z[::-1] if at_end else z
    d = np.arange(len(p)) * step
    far = (d >= ABUT_NEAR_M) & (d <= ABUT_FAR_M)
    if far.sum() < 4:
        return z
    b, a = np.polyfit(d[far], p[far], 1)
    n = int((d < ABUT_NEAR_M).sum())
    e = a + b * d[:n]
    if e[0] - p[0] < ABUT_MIN_M or (np.diff(p[:n + 1]) / step - b).max() < ABUT_STEEP:
        return z
    q = p.copy()
    q[:n] = np.minimum(np.maximum(p[:n], e), p[:n] + ABUT_MAX_M)
    return q[::-1] if at_end else q


def underpass_offsets(lines, requirements, queries=None):
    """Smooth underpass dips along joined ground roads, never across space (earthfast.underpass_offsets; after
    wyo-backroads' PR #374): m (<= 0) to add to the roads' heights.

    lines: the ground roads [(k, 2)]; requirements [(line, arc m, depth m)] of the overpasses they cross under.
    queries: [(line, points (k, 2) on it)] where the offsets are asked (default: every line's own points); one
    array per query (per line by default). Roads join where their ends meet; a node shared by several roads takes
    one offset in each, the eighth norm combines nearby dips without summing their depths, loose ends stay pinned
    and only joined ground junctions carry a dip past the road under the bridge. Pure."""
    import fast
    fast.require(85, "underpass_offsets")
    f64 = np.float64
    pts = [np.asarray(ln, f64)[:, :2] for ln in lines]
    if queries is None:
        queries = list(enumerate(pts))
    if not queries:
        return []
    qs = [np.asarray(q, f64).reshape(-1, 2) for _l, q in queries]
    if not pts or not requirements:
        return [np.zeros(len(q)) for q in qs]
    starts = np.r_[0, np.cumsum([len(p) for p in pts])].astype(np.int64)
    xy = np.ascontiguousarray(np.vstack(pts)) if len(starts) and starts[-1] else np.zeros((0, 2))
    req = np.ascontiguousarray(np.asarray(requirements, f64).reshape(-1, 3))
    qxy = np.ascontiguousarray(np.vstack(qs)) if sum(len(q) for q in qs) else np.zeros((0, 2))
    qline = np.repeat(np.array([int(l) for l, _q in queries], np.int64), [len(q) for q in qs])
    prm = np.array([UNDER_FLAT_M, UNDER_GRADE, UNDER_MAX_GRADE, UNDER_MAX_DROP_M, UNDER_JOINT_FLAT_M,
                    UNDER_JOINT_FADE_M, UNDER_NODE_M], f64)
    out = fast._ef.underpass_offsets(xy, starts, req, qxy, qline, prm)
    return np.split(out, np.cumsum([len(q) for q in qs])[:-1])


def street_runs(lines, classes, requirements):
    """UNDER_RUN_* (coarse elevation): requirements [(line, arc m, depth m)] on the ground roads lines [(k, 2)] (classes:
    each one's road class) split into (those spread along the joined roads as before: [(line, arc, depth)], those kept
    on their street: [(run xy (k, 2), {its lines}, [(0, arc m along the run, depth m)])]). A run: a line carried on
    through every node where only its next piece joins it (the map splits a road at more than its junctions) to its
    junctions (3+ lines) and loose ends; underpass_offsets on a run alone lays each dip inside it (flat, then ramps up
    to UNDER_MAX_GRADE, as deep as fits). Kept on its run: a street's dip (not a motorway's or trunk's: APPROACH_SKIP)
    whose run has room for UNDER_RUN_MIN_M of it (or all of it, when shallower) and is no ring. Pure."""
    if not requirements:
        return list(requirements), []
    pts = [np.asarray(ln, np.float64)[:, :2] for ln in lines]
    nodes, edges, _adj = roadsmod._chain_graph(pts)
    inc = [[] for _ in nodes]
    for l, (a, b, _s) in enumerate(edges):
        inc[a].append(l)
        inc[b].append(l)
    memo = {}

    def run_of(i):
        """(xy, its arc lengths, {line: (arc of its start in the run, reversed, its own length)}, ring), the same for
        every line of the run."""
        got = memo.get(i)
        if got is not None:
            return got
        seq, ring = [(i, False)], False
        for ahead in (True, False):
            l, rev = i, False
            while True:
                a, b, _s = edges[l]
                end = (a if rev else b) if ahead else (b if rev else a)
                if len(inc[end]) != 2:
                    break
                m = inc[end][1] if inc[end][0] == l else inc[end][0]
                if any(m == k for k, _r in seq):
                    ring = True
                    break
                ma, mb, _s = edges[m]
                rev = (ma != end) if ahead else (mb != end)
                if ahead:
                    seq.append((m, rev))
                else:
                    seq.insert(0, (m, rev))
                l = m
        xy, first = [], []
        for m, rev in seq:
            p = pts[m][::-1] if rev else pts[m]
            first.append(sum(len(q) for q in xy) - (1 if xy else 0))
            xy.append(p if not xy else p[1:])
        xy = np.vstack(xy)
        sx = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        got = (xy, sx, {m: (float(sx[k]), rev, float(edges[m][2][-1])) for (m, rev), k in zip(seq, first)}, ring)
        for m, _rev in seq:
            memo[m] = got
        return got
    net, runs = [], {}
    for i, sc, depth in requirements:
        i = int(i)
        if classes[i] in APPROACH_SKIP:
            net.append((i, sc, depth))
            continue
        xy, sx, at, ring = run_of(i)
        s0, rev, length = at[i]
        arc = float(np.clip(s0 + (length - sc if rev else sc), 0.0, sx[-1]))
        fit = max(0.0, min(arc, float(sx[-1]) - arc) - UNDER_FLAT_M) * 2.0 * UNDER_MAX_GRADE / math.pi
        if ring or fit < min(depth, UNDER_RUN_MIN_M):
            net.append((i, sc, depth))
            continue
        runs.setdefault(id(xy), (xy, set(at), []))[2].append((0, arc, depth))
    return net, list(runs.values())


def under_dip(d, depth, room=(math.inf, math.inf)):
    """m added (<= 0) to a road's heights d m (along it, signed) from where it crosses under an overpass, for a dip
    depth m deep: flat under the deck (UNDER_FLAT_M either way), then back up on a cosine ramp whose steepest grade is
    UNDER_GRADE (a V of straight UNDER_GRADE ramps was a 6 % kink at each end and at the bottom: at motorway speed a
    jolt). room: m from the crossing to the road part's (start, end): a ramp that would run past it is made shorter,
    up to UNDER_MAX_GRADE, so the dip ends inside the part (its next part, past a junction, is not lowered: a step).
    None when that is too steep. Pure."""
    d = np.asarray(d, np.float64)
    run = depth * math.pi / (2 * UNDER_GRADE)
    runs = []
    for r in room:
        avail = r - UNDER_FLAT_M
        if avail >= run:
            runs.append(run)
        elif avail > 0 and depth * math.pi / (2 * avail) <= UNDER_MAX_GRADE:
            runs.append(avail)
        else:
            return None
    rr = np.where(d < 0, runs[0], runs[1])
    u = np.clip((np.abs(d) - UNDER_FLAT_M) / np.maximum(rr, 1e-9), 0.0, 1.0)
    return -depth * 0.5 * (1.0 + np.cos(math.pi * u))


def approach_add(xy, at_end, lift, half, others=(), side_run=None, grade=None):
    """m to add to a road plan's vertices (xy (n, 2)) rising to `lift` at its first (at_end False) or last vertex (a
    bridge's abutment), falling off along the road (arc length) on a cosine ramp whose steepest grade is APPROACH_GRADE;
    at most what keeps the fill slope (side_run x the height out, SLOPE_RUN by default) clear of the other roads
    [(line (k, 2), half width)] beside it, with APPROACH_GAP_M to spare. None when nothing is raised. Pure."""
    side_run = SLOPE_RUN if side_run is None else side_run
    xy = np.asarray(xy, np.float64)[:, :2]
    if len(xy) < 2 or lift < APPROACH_MIN_LIFT:
        return None
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    d = (s[-1] - s) if at_end else s
    run = float(np.clip(lift * math.pi / (2 * (grade or APPROACH_GRADE)), APPROACH_MIN_RUN,
                        APPROACH_MAX_RUN * (APPROACH_GRADE / grade if grade else 1.0)))
    u = np.clip(d / run, 0.0, 1.0)
    add = lift * 0.5 * (1.0 + np.cos(math.pi * u))
    on = add > 1e-3
    if not on.any():
        return None
    q0 = xy[on]
    lo_q, hi_q = q0.min(0), q0.max(0)
    for line, oh in others:
        line = np.asarray(line, np.float64)[:, :2]
        if len(line) < 2:
            continue
        reach = half + oh + APPROACH_GAP_M + side_run * lift
        if (line.max(0) < lo_q - reach).any() or (line.min(0) > hi_q + reach).any():
            continue                    # too far to matter
        q = xy[on]
        a, b = line[:-1], line[1:]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0.0, 1.0)
        dist = np.hypot(*(q[:, None, :] - (a[None] + t[..., None] * ab[None])).transpose(2, 0, 1)).min(1)
        cap = np.maximum(0.0, (dist - half - oh - APPROACH_GAP_M) / max(side_run, 1e-6))
        add[on] = np.minimum(add[on], cap)
    # a cap lowers the ramp before it at APPROACH_CAP_GRADE at most, instead of a step at the cap
    for i in range(1, len(add)):
        add[i] = min(add[i], add[i - 1] + APPROACH_CAP_GRADE * (s[i] - s[i - 1]))
    for i in range(len(add) - 2, -1, -1):
        add[i] = min(add[i], add[i + 1] + APPROACH_CAP_GRADE * (s[i + 1] - s[i]))
    if add.max() < 1e-3:
        return None
    return add


def _is_link(spec):
    return bool(getattr(spec, "link", False))


def _road_fids(i, lines, classes, gids, reach=XS_FLOOR_M, touch=3.0):
    """The feature ids of ground line i and of the lines continuing it within reach m (joined end to end, of its class:
    a road split at junctions, a dual carriageway's halves meeting it). Pure."""
    base = lines[i]
    c0 = np.asarray(base, np.float64)[len(base) // 2]
    road, frontier = {i}, [i]
    ends = np.array([[ln[0, :2], ln[-1, :2]] for ln in lines], np.float64)
    while frontier:
        k = frontier.pop()
        for e in (ends[k, 0], ends[k, 1]):
            if np.hypot(*(e - c0)) > reach:
                continue
            d = np.minimum(np.hypot(*(ends[:, 0] - e).T), np.hypot(*(ends[:, 1] - e).T))
            for m in np.flatnonzero(d <= touch):
                if m not in road and classes[m] == classes[i]:
                    road.add(int(m))
                    frontier.append(int(m))
    return {gids[m][0] for m in road}


def _seg_dist(line, p):
    """m from p to the polyline line (k, 2). Pure."""
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(a + t[:, None] * ab - p).T).min())


def _never_raised(spec):
    """A road whose approaches are never raised: a motorway or trunk (APPROACH_SKIP), not one of their links. A link
    leaving a motorway's bridge runs down a fill to the street: kept at the ground, its deck fell 11 m within 90 m and
    dropped beside its mainline while they still overlapped, and an AI car drove into the mainline's parapet trying to
    take the exit (US-101's exit 1E over the Los Angeles River, 2026-10-27)."""
    return spec.cls in APPROACH_SKIP and not _is_link(spec)


TERRAIN_ROUGH_M = float(os.environ.get("BNE_TERRAIN_BUMP", "0.35"))   # off-road roughness (std), about twice a minor
                                                                       # dirt road's: beside it is rougher than on it
TERRAIN_ROUGH_WAVES = ((9.0, 1.0), (4.5, 0.6), (2.4, 0.3))             # (feature size m, share)


def _hash01(i, j, salt=0):
    """A stable pseudo-random value in [0, 1) for lattice points (Python's hash() is randomised per process)."""
    i = np.asarray(i, np.int64)
    j = np.asarray(j, np.int64)
    h = (i * np.int64(73856093)) ^ (j * np.int64(19349663)) ^ np.int64(int(salt) * 83492791)
    h = h.astype(np.uint64)
    h ^= h >> np.uint64(13)
    h *= np.uint64(1274126177)
    h ^= h >> np.uint64(16)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


# The land off the roads roughened (PR #677): opt-in (BNE_TERRAIN_BUMPS=on) until it is one kernel applied the same way
# on every ground path (provisional blocks and far tiles too) and kept off towns, water and building sites; on, it is
# ~0.3-0.5 s of numpy per million samples on every block, road tile and far tile (2026-11-05 merge review).
TERRAIN_BUMPS_ON = os.environ.get("BNE_TERRAIN_BUMPS", "off") == "on"


def terrain_bumps(lat, lon):
    """World-locked roughness of the land off the roads (m), the same value whatever process or tile asks: value noise
    at a few sizes, about TERRAIN_ROUGH_M (std). 0.0 (nothing added) unless TERRAIN_BUMPS_ON, or when BNE_TERRAIN_BUMP
    is 0."""
    if not TERRAIN_BUMPS_ON or TERRAIN_ROUGH_M <= 0.0:
        return 0.0
    la = np.asarray(lat, np.float64)
    lo = np.asarray(lon, np.float64)
    y = la * 110574.0
    x = lo * 111320.0 * np.cos(np.radians(la))
    out = np.zeros(np.shape(la), np.float64)
    norm = 0.0
    for size, share in TERRAIN_ROUGH_WAVES:
        fx = x / size
        fy = y / size
        i0 = np.floor(fx)
        j0 = np.floor(fy)
        tx = fx - i0
        ty = fy - j0
        sx = tx * tx * (3.0 - 2.0 * tx)
        sy = ty * ty * (3.0 - 2.0 * ty)
        i0 = i0.astype(np.int64)
        j0 = j0.astype(np.int64)
        v00 = _hash01(i0, j0, size)
        v10 = _hash01(i0 + 1, j0, size)
        v01 = _hash01(i0, j0 + 1, size)
        v11 = _hash01(i0 + 1, j0 + 1, size)
        v = (v00 * (1.0 - sx) + v10 * sx) * (1.0 - sy) + (v01 * (1.0 - sx) + v11 * sx) * sy
        out += share * (v - 0.5)
        norm += share * share
    return out * (TERRAIN_ROUGH_M / (0.1875 * math.sqrt(norm)))
