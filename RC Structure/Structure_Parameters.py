# ==================================================
# Structure_Parameters.py
# ==================================================
# Global parameters for RC 3D frame model
#
# Unit system:
#   Force  = kip
#   Length = inch
#   Time   = second
#   Stress = ksi
# ==================================================

import math

# Geometry
NUM_BAY_X = 3
NUM_BAY_Y = 3
NUM_FLOOR = 8

BAY_X = 120.0
BAY_Y = 120.0
STORY_H = 120.0

# Material strengths
FC_COL_KSI = 5.0
FC_BEAM_KSI = 4.0

FY_KSI = 60.0
ES_KSI = 29000.0
STEEL_B = 0.01

# Section dimensions
B_COL = 18.0
H_COL = 18.0

B_BEAM = 12.0
H_BEAM = 18.0

# Legacy distance from concrete face to longitudinal bar centroid. New slab-
# aware designs instead use the explicit clear cover outside transverse steel.
COVER = 1.5
BEAM_CLEAR_COVER_IN = 1.5  # sheltered interior research frames, outside hoops
COL_CLEAR_COVER_IN = 1.5
AGGREGATE_MAX_SIZE_IN = 0.75
REINFORCEMENT_SPECIFICATION = "ASTM A706 Grade 60"
MATERIAL_EXPOSURE = "sheltered_interior"

# Rebar database
REBAR = {
    3: {"area": 0.11, "diameter": 0.375},
    4: {"area": 0.20, "diameter": 0.500},
    5: {"area": 0.31, "diameter": 0.625},
    6: {"area": 0.44, "diameter": 0.750},
    7: {"area": 0.60, "diameter": 0.875},
    8: {"area": 0.79, "diameter": 1.000},
    9: {"area": 1.00, "diameter": 1.128},
    10: {"area": 1.27, "diameter": 1.270},
    11: {"area": 1.56, "diameter": 1.410},
    14: {"area": 2.25, "diameter": 1.693},
    18: {"area": 4.00, "diameter": 2.257},
}


def rebar_area(bar_size):
    return REBAR[bar_size]["area"]


def rebar_diameter(bar_size):
    return REBAR[bar_size]["diameter"]


# Reinforcement layouts
COL_BAR_SIZE = 8
COL_TOP_BARS = 4
COL_BOT_BARS = 4
COL_SIDE_BARS = 2
COL_BAR_AREA = rebar_area(COL_BAR_SIZE)

BEAM_BAR_SIZE = 6
BEAM_TOP_BARS = 2
BEAM_BOT_BARS = 2
BEAM_SIDE_BARS = 0
BEAM_BAR_AREA = rebar_area(BEAM_BAR_SIZE)

# Transverse reinforcement used in shear design checks.
COL_STIRRUP_BAR_SIZE = 4
COL_STIRRUP_LEGS = 2
COL_STIRRUP_SPACING = 6.0
# Column hoop legs per direction as the capacity design selects them
# (ACI 318-19 18.7.5.4 defines Ash per direction): legs crossing the b faces
# run along h, legs crossing the h faces run along b. None means the same
# count both ways, COL_STIRRUP_LEGS. COL_STIRRUP_LEGS itself stays the
# scalar the hinge calibration and the legacy shear checks read; the design
# sets it to the lighter direction.
COL_STIRRUP_LEGS_BY_DIRECTION = None

BEAM_STIRRUP_BAR_SIZE = 4
BEAM_STIRRUP_LEGS = 2
BEAM_STIRRUP_SPACING = 6.0

STIRRUP_MIN_SPACING = 3.0
STIRRUP_SPACING_STEP = 1.0


def _reinforcement_member(member_type):
    member = str(member_type).lower()
    if member in ("col", "column"):
        return "column"
    if member == "beam":
        return "beam"
    raise ValueError("member_type must be 'beam' or 'column'.")


def core_cover_in(member_type):
    """Face-to-core boundary at outside of hoops, not longitudinal centroids.

    This geometry distinction does not validate a confined-concrete material
    law; calibration to the actual confinement layout remains a separate check.
    The old fiber partition is unchanged when no slab design is selected.
    """
    member = _reinforcement_member(member_type)
    if SLAB_THICKNESS_IN is None:
        return COVER
    cover = COL_CLEAR_COVER_IN if member == "column" else BEAM_CLEAR_COVER_IN
    if isinstance(cover, bool) or not math.isfinite(cover) or cover <= 0:
        raise ValueError("Clear cover must be positive and finite.")
    return cover


def longitudinal_cover_in(member_type, bar_size=None, stirrup_bar_size=None):
    """Concrete-face to longitudinal-bar centroid, in inches.

    Optional bar sizes price a candidate without changing the current section.
    New-mode bars touch the inside of the hoop: c_clear + d_hoop + d_bar/2.
    """
    member = _reinforcement_member(member_type)
    if SLAB_THICKNESS_IN is None:
        return COVER
    is_col = member == "column"
    if bar_size is None:
        bar_size = COL_BAR_SIZE if is_col else BEAM_BAR_SIZE
    if stirrup_bar_size is None:
        stirrup_bar_size = COL_STIRRUP_BAR_SIZE if is_col else BEAM_STIRRUP_BAR_SIZE
    return (core_cover_in(member) + rebar_diameter(stirrup_bar_size)
            + 0.5 * rebar_diameter(bar_size))


# ---- Orthogonal beam bar stacking at the joints (design-basis rule, 2026-09-27) --------------
# The x and y beam bars cannot share an elevation where they cross at a joint. One direction's
# top bars sit on top; that direction's bottom bars then sit nearer the bottom face (mirrored),
# so each direction gives up one layer of depth at one face. Bars run straight, so the lower
# position holds along the whole member and enters every strength that uses the bar depth.
#   "inner_over_outer"  the direction whose slab mats are the INNER mats sits on top, so the
#                       lower direction's bars cross only the inner mats and clear the outer
#                       bottom mat (Design/SMRF_Cage_Geometry.joint_assembly); resolved per design
#                       from the slab layout's outer_axis (x mats outer -> "y_over_x").
#   "x_over_y", "y_over_x"  fixed conventions.   "none"  legacy single elevation.
BEAM_BAR_STACKING = "inner_over_outer"
BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN = 1.0     # ACI 318-19 25.2.2, clear between layers
# Layers of beam bars per face (user decision 2026-09-27: "we can just add another layer"). The bars
# of a direction that do not pass the column bars in one layer take a second layer directly below
# (25.2.2), each layer one pitch (db + max(1 in, db)) apart; the lower direction's cage then starts
# below the upper direction's last layer. Beyond this many layers the beam widens (or the column
# grows where a lane-keeping cage exists), as before.
BEAM_BAR_MAX_LAYERS = 2
# How the two directions' layers order at a face (user decision 2026-09-27, "lets interleave the
# layers"). "interleaved": upper first layer, lower first layer, upper second, lower second, ...
# one pitch apart, so a direction's own layers sit two pitches apart with the crossing bars between
# them (25.2.2 alignment kept) and neither direction gives up a whole cage's depth. "blocked": one
# direction's whole cage below the other's (the convention until this decision).
BEAM_BAR_LAYER_ORDER = "interleaved"
_BEAM_LAYER_CACHE = {}


def beam_bar_stacking_convention():
    """The resolved stacking: 'x_over_y', 'y_over_x' or 'none'."""
    mode = BEAM_BAR_STACKING
    if mode in ("none", "x_over_y", "y_over_x"):
        return mode
    if mode != "inner_over_outer":
        raise ValueError(f"Unknown BEAM_BAR_STACKING {mode!r}")
    layout = (SLAB_REINFORCEMENT or {}).get("layout") if SLAB_THICKNESS_IN is not None else None
    outer = (layout or {}).get("outer_axis", "x")
    return "y_over_x" if outer == "x" else "x_over_y"


def beam_bar_layer_pitch_in(bar_size=None):
    """Centre-to-centre distance between two layers of beam bars: db + the 25.2.2 clear (>= db, 1 in)."""
    db = rebar_diameter(BEAM_BAR_SIZE if bar_size is None else bar_size)
    return db + max(BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN, db)


def beam_bars_per_layer(bar_size=None, bars_per_face=None, stirrup_bar_size=None):
    """{axis: n}: how many beam bars of this size pass the current column cage in one layer, per direction.

    The same geometry as the capacity check beam.bars_thread_column and the column cage rule
    (Design/SMRF_Cage_Geometry.column_cage_admits_beam_bars), on the live column cage and beam width.
    None without a slab (the legacy route has no joint assembly).
    """
    if SLAB_THICKNESS_IN is None:
        return None
    size = BEAM_BAR_SIZE if bar_size is None else bar_size
    n = max(BEAM_TOP_BARS, BEAM_BOT_BARS) if bars_per_face is None else bars_per_face
    stirrup = BEAM_STIRRUP_BAR_SIZE if stirrup_bar_size is None else stirrup_bar_size
    key = (B_COL, H_COL, COL_CLEAR_COVER_IN, COL_STIRRUP_BAR_SIZE, COL_BAR_SIZE, COL_TOP_BARS, COL_SIDE_BARS,
           B_BEAM, BEAM_CLEAR_COVER_IN, stirrup, size, n, AGGREGATE_MAX_SIZE_IN)
    cached = _BEAM_LAYER_CACHE.get(key)
    if cached is None:
        from Design.SMRF_Cage_Geometry import column_cage_admits_beam_bars
        if len(_BEAM_LAYER_CACHE) > 4096:
            _BEAM_LAYER_CACHE.clear()
        cached = column_cage_admits_beam_bars(
            B_COL, H_COL, COL_CLEAR_COVER_IN, rebar_diameter(COL_STIRRUP_BAR_SIZE), rebar_diameter(COL_BAR_SIZE),
            COL_TOP_BARS, COL_SIDE_BARS, B_BEAM, BEAM_CLEAR_COVER_IN, rebar_diameter(stirrup), rebar_diameter(size),
            n, AGGREGATE_MAX_SIZE_IN)["bars_per_layer"]
        _BEAM_LAYER_CACHE[key] = dict(cached)
    return dict(cached)


def _split_into_layers(n, fit, limit=None):
    """Bar counts per layer, filling each layer to what the lanes admit; one layer when everything fits.

    When no lane exists at all, or the bars would need more layers than ``limit`` allows, the bars
    are kept at one elevation: the arrangement is not one the rule accepts, the threading check
    fails on it, and the strengths are not priced on an arrangement that will not be built.
    """
    if fit is None or fit <= 0 or fit >= n:
        return [n]
    layers, remaining = [], n
    while remaining > 0:
        layers.append(min(fit, remaining))
        remaining -= layers[-1]
    if limit is not None and len(layers) > limit:
        return [n]
    return layers


def beam_bar_layer_slots(n_near, n_far, order=None):
    """Slot indices (0 nearest the face, one pitch apart) of the near and far directions' layers.

    "blocked": the near direction's layers first, then the far direction's. "interleaved": near
    first layer, far first layer, near second, far second, ... A single layer per direction gives the
    same slots either way.
    """
    order = BEAM_BAR_LAYER_ORDER if order is None else order
    if order == "blocked":
        return list(range(n_near)), list(range(n_near, n_near + n_far))
    if order != "interleaved":
        raise ValueError(f"Unknown BEAM_BAR_LAYER_ORDER {order!r}")
    near, far, slot = [], [], 0
    for i in range(max(n_near, n_far)):
        if i < n_near:
            near.append(slot)
            slot += 1
        if i < n_far:
            far.append(slot)
            slot += 1
    return near, far


def beam_bar_layers(bar_size=None, top_bars=None, bot_bars=None, stirrup_bar_size=None, max_layers=None):
    """Per direction and face: the layers the beam bars occupy and their centroid offset from that face.

    {axis: {face: {"layers", "per_layer", "offsets_in", "centroid_in"}}}. At the top face the upper
    direction's first layer comes first (offset base) and the layers then follow
    BEAM_BAR_LAYER_ORDER (interleaved: upper, lower, upper, lower ...; blocked: the upper cage, then
    the lower), one pitch apart; at the bottom face the mirror (the lower direction's bars nearest
    the face). A direction whose bars pass the column bars in one layer has one layer. Without a
    slab or with stacking "none": one layer at the nominal offset for both directions and faces.
    """
    base = (longitudinal_cover_in("beam", bar_size) if stirrup_bar_size is None
            else longitudinal_cover_in("beam", bar_size, stirrup_bar_size))
    counts = {"top": BEAM_TOP_BARS if top_bars is None else top_bars,
              "bottom": BEAM_BOT_BARS if bot_bars is None else bot_bars}
    convention = beam_bar_stacking_convention()
    if convention == "none" or SLAB_THICKNESS_IN is None:
        return {axis: {face: {"layers": 1, "per_layer": [counts[face]], "offsets_in": [base], "centroid_in": base}
                       for face in counts} for axis in ("x", "y")}
    pitch = beam_bar_layer_pitch_in(bar_size)
    # Under a one-layer limit the bars stay at one elevation (the threading check fails when they do
    # not fit); with more layers allowed, the lanes decide the split.
    limit = BEAM_BAR_MAX_LAYERS if max_layers is None else max_layers
    fit = (beam_bars_per_layer(bar_size, max(counts.values()), stirrup_bar_size) if limit > 1
           else {"x": None, "y": None})
    upper_axis, lower_axis = ("x", "y") if convention == "x_over_y" else ("y", "x")
    result = {"x": {}, "y": {}}
    for face in ("top", "bottom"):
        near, far = (upper_axis, lower_axis) if face == "top" else (lower_axis, upper_axis)
        near_layers = _split_into_layers(counts[face], fit[near], limit)
        far_layers = _split_into_layers(counts[face], fit[far], limit)
        near_slots, far_slots = beam_bar_layer_slots(len(near_layers), len(far_layers))
        near_offsets = [base + k * pitch for k in near_slots]
        far_offsets = [base + k * pitch for k in far_slots]
        for axis, layers, offsets in ((near, near_layers, near_offsets), (far, far_layers, far_offsets)):
            result[axis][face] = {"layers": len(layers), "per_layer": layers, "offsets_in": offsets,
                                  "centroid_in": sum(c * o for c, o in zip(layers, offsets)) / sum(layers)}
    return result


def beam_bar_stacking_offsets_in(bar_size=None, stirrup_bar_size=None, top_bars=None, bot_bars=None, max_layers=None):
    """{axis: {"top": offset, "bottom": offset}}: face-to-centroid offsets of the beam bars, in inches.

    The centroid of that direction's layers at that face (beam_bar_layers): the stacked orthogonal
    cages and, where the lanes force it, the second layer of bars. Without a slab (legacy route) or
    with stacking "none" both directions use the nominal offset at both faces.
    """
    layers = beam_bar_layers(bar_size, top_bars, bot_bars, stirrup_bar_size, max_layers)
    return {axis: {face: layers[axis][face]["centroid_in"] for face in ("top", "bottom")} for axis in ("x", "y")}


def beam_centroid_offset_in(axis, face, bar_size=None):
    """Face-to-centroid offset of the beam bars of one direction at one face ('top' or 'bottom')."""
    return beam_bar_stacking_offsets_in(bar_size)[axis][face]


def beam_worst_centroid_offset_in(bar_size=None, top_bars=None, bot_bars=None, max_layers=None):
    """The largest offset over directions and faces: the conservative single value for sizing."""
    offsets = beam_bar_stacking_offsets_in(bar_size, None, top_bars, bot_bars, max_layers)
    return max(value for faces in offsets.values() for value in faces.values())


def longitudinal_clear_spacing_in(member_type, bar_size=None):
    """ACI 318-19 25.2.1 / 25.2.3 unbundled single-layer minimum spacing.

    The aggregate term is active only for new slab-aware research designs.
    This does not cover hoop/crosstie or joint congestion and anchorage checks.
    """
    member = _reinforcement_member(member_type)
    if bar_size is None:
        bar_size = COL_BAR_SIZE if member == "column" else BEAM_BAR_SIZE
    db = rebar_diameter(bar_size)
    minimum = max(1.5, 1.5 * db) if member == "column" else max(1.0, db)
    if SLAB_THICKNESS_IN is not None:
        aggregate = AGGREGATE_MAX_SIZE_IN
        if isinstance(aggregate, bool) or not math.isfinite(aggregate) or aggregate <= 0:
            raise ValueError("Maximum aggregate size must be positive and finite.")
        minimum = max(minimum, 4.0 * aggregate / 3.0)
    return minimum

# Design DCR target band (iterative steel redesign)
DESIGN_DCR_MIN = 0.60   # lower bound — avoid over-design
DESIGN_DCR_MAX = 0.95   # upper bound — demand must be met
# 0.95 rather than 0.90: the discrete section and bar ladders cannot land
# every geometry inside a narrow window, and a design that misses by a
# percent is safe, not wrong. pilot30 case_0027 converged on a beam DCR of
# 0.910 with the column at 0.478 and SCWB at 2.32, then spent five
# iterations failing to do better because the ladder has no rung between.

# Loading / mass
G = 386.4

# Concrete unit weight (normal-weight concrete)
CONCRETE_UNIT_WEIGHT_KCF = 0.150              # kip/ft³
CONCRETE_UNIT_WEIGHT_KCI = CONCRETE_UNIT_WEIGHT_KCF / 1728.0  # kip/in³

# Legacy floor area loads are retained for existing analyses. The dead-load
# allowance already includes slab weight: never add computed slab weight to it.
# A selected SLAB_THICKNESS_IN opts into separately accounted slab + SDL and
# member self-weight. SDL is an explicit research assumption, not inferred from
# the legacy bundled allowance; effective seismic weight still needs project-
# specific occupancy/partition/roof review before code qualification.
FLOOR_DEAD_LOAD_KSF = 0.15   # kip/ft²  (slab self-weight + finishes + MEP)
FLOOR_LIVE_LOAD_KSF = 0.05   # kip/ft²  (office occupancy)
SLAB_THICKNESS_IN = None    # None preserves legacy loads/mass; design selects one h
FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF = 0.05  # explicit finishes/MEP/partition allowance
SEISMIC_LIVE_LOAD_FRACTION = 0.0         # research office-floor assumption

FX_FLOOR = 10.0

# Gravity load application.
# Options:
#   "nodal"        = tributary-area nodal loads at floor grid nodes
#   "beam_uniform" = tributary uniform loads on beams
#
# The nodal option is the baseline for the IMK hinge model because applying
# uniform element loads directly to the elastic spine can create a large
# gravity-only sidesway before pushover starts.
#
# When a design has produced a slab-to-frame transfer (Design/SMRF_Floor_Transfer),
# FLOOR_TRANSFER holds it and Gravity_Loads applies the slab's discrete beam
# node loads and column footprint loads instead of either option above. The
# effective mode is then "slab_transfer"; GRAVITY_LOAD_MODEL keeps naming the
# request's fallback so the design request identity stays stable.
GRAVITY_LOAD_MODEL = "nodal"
FLOOR_TRANSFER = None

# Selected slab reinforcement (Design/SMRF_Slab_Reinforcement record). When
# its layout exists, beam strengths for SCWB and for the IMK beam hinges
# include the developed slab mats within the ACI 6.3.2 effective flange
# (Design/SMRF_Beam_Slab_Strength); None keeps rectangular beam strengths.
SLAB_REINFORCEMENT = None
# The computed slab strip action evidence (Design/SMRF_Slab_Actions), kept
# whether or not it has been asserted as verified, so it can be reviewed.
SLAB_ACTIONS = None


def effective_gravity_load_model():
    return "slab_transfer" if FLOOR_TRANSFER is not None else GRAVITY_LOAD_MODEL

# Material tags
COVER_COL_TAG = 1
CORE_COL_TAG = 2
COVER_BEAM_TAG = 3
CORE_BEAM_TAG = 4
STEEL_TAG = 5

# Section tags
COL_SEC_TAG = 101
BEAM_SEC_TAG = 102

# Beam integration tags
COL_INTEG_TAG = 1
BEAM_INTEG_TAG = 2
NUM_INT_PTS = 5

# Geometric transformation tags
COL_TRANSF_TAG = 1
BEAM_X_TRANSF_TAG = 2
BEAM_Y_TRANSF_TAG = 3

# Element formulation
# Options:
#   "fiber" = distributed forceBeamColumn fiber sections
#   "imk"   = elastic interior members with IMKPeakOriented rotational hinges
ELEMENT_FORMULATION = "imk"

# Both member types use concentrated hinges. Concrete02 fiber models have
# envelope softening and unloading degradation, but do not represent every
# mechanism of repeated-cycle strength loss. That is a separate formulation.
# Per-member backbones come from Model/IMK_Calibration.py.
IMK_APPLY_TO_COLUMNS = True
IMK_APPLY_TO_BEAMS = True

# IMK lumped hinge tags
IMK_HINGE_NODE_TAG_BASE = 2000000
IMK_MATERIAL_TAG_BASE = 3000000
IMK_HINGE_ELEMENT_TAG_BASE = 4000000

# IMK hinge calibration defaults.
# These are intentionally centralized because IMK behavior should eventually be
# calibrated from design/code equations or experiments rather than hidden in the
# element builder.
# Backbone calibration source.
#   True  = per-member Haselton et al. (2008) rotation capacities and a yield
#           moment read off the nominal P-M surface at the member's axial load
#   False = the fixed IMK_THETA_* constants below, identical for every element
#
# The fixed constants cannot represent a column, whose plastic rotation
# capacity falls steeply with axial load. With them, no column in the model
# can degrade faster than any other and soft-story mechanisms cannot form.
IMK_USE_CALIBRATED_BACKBONE = True

# RC member flexure uses peak-oriented reloading (the intended split:
# IMKPeakOriented for flexure, IMKPinching for joint shear through its own
# adapter and topology, Model/Joint_Panel, which is a diagnostic prototype
# accepting shear-only inputs; the member/joint slip partition is unresolved
# and no pinching spring is installed in the frame). IMKBilin remains
# selectable for reproducing legacy runs, with its own argument signature.
IMK_MATERIAL_TYPE = "IMKPeakOriented"

# Cyclic deterioration capacities handed to the member materials.
#   "haselton_2008": Eq. 3.20 of PEER 2007/03, lambda = 170.7 (0.27)^nu
#           (0.10)^(s/d), times the nominal MEMBER yield rotation below, which
#           converts Haselton's Et = lambda My theta_y to the modern OpenSees
#           Et = Lambda My (Model/IMK_Calibration.deterioration_for_member).
#           The fitted capacity applies to the strength modes S and C only;
#           Haselton's calibration excluded A and K, which receive a
#           documented large-capacity approximation to zero deterioration.
#           Peak-oriented reloading is unchanged. The nominal 0.005/0.004 rad
#           yield rotations remain modeling assumptions, and extending a
#           column regression to beams is a research calibration, so the
#           calibration status stays provisional.
#   "direct": pass IMK_LAMBDA_* below unchanged (legacy energy convention).
IMK_DETERIORATION_MODE = "haselton_2008"
IMK_ENERGY_CONVENTION = "opensees_ref_energy_equals_lamda_times_fy"
IMK_CYCLIC_CALIBRATION_ID = "legacy_member_cyclic_defaults_v1"
IMK_CYCLIC_CALIBRATION_STATUS = "provisional_not_experimentally_calibrated"
# Corrected member mapping stays disabled until experimentally supported mode
# energies are supplied and reviewed (user decision, 2026-09-24). No hogging,
# average-strength or legacy-Lambda energy anchor is inferred.
IMK_ENERGY_MAPPING_MODE = "legacy_unmapped"
# Corrected mode: "explicit_reference_energy_v1". Profiles are keyed by string
# physical element tag, then i/j, then y/z; each contains its own evidence.
IMK_MEMBER_ENERGY_CALIBRATIONS = {}

# IMK rotational spring stiffness calibration.
#   "yield_rotation"          = Ke = My / theta_y_input
#   "member_stiffness_factor" = Ke = stiffness_factor * 6EI/L
#   "max"                     = legacy rule: max(My/theta_y_input, stiffness_factor*6EI/L)
#
# Use "member_stiffness_factor" for concentrated-plasticity springs so the
# elastic beam-column element carries member flexibility and the zeroLength
# hinge remains stiff until its backbone reaches yield strength.
IMK_HINGE_STIFFNESS_MODE = "member_stiffness_factor"
IMK_HINGE_STIFFNESS_FACTOR = 20.0
IMK_BEAM_THETA_Y = 0.005
IMK_COLUMN_THETA_Y = 0.004
IMK_LAMBDA_S = 10.0
IMK_LAMBDA_C = 10.0
IMK_LAMBDA_K = 10.0
# All LAMBDA values are direct modern OpenSees inputs: E_ref = Lamda * My.
# Their deformation coordinate is spring rotation (rad), not normalized drift.
# No additional theta_y multiplier is applied. A-mode values are approved only
# for the provisional diagnostic comparison (2026-09-22), not experimental fits.
IMK_LAMBDA_A = 10.0
IMK_C_S = 1.0
IMK_C_C = 1.0
IMK_C_K = 1.0
IMK_C_A = 1.0
IMK_THETA_P_POS = 0.020
IMK_THETA_P_NEG = 0.020
IMK_THETA_PC_POS = 0.060
IMK_THETA_PC_NEG = 0.060
IMK_RES_POS = 0.20
IMK_RES_NEG = 0.20

# Current OpenSees IMKBilin uses maximum/residual strength ratios, not
# post-yield hardening ratios, in the material command.
IMK_FMAXFY_POS = 1.10
IMK_FMAXFY_NEG = 1.10
IMK_FRESFY_POS = IMK_RES_POS
IMK_FRESFY_NEG = IMK_RES_NEG
IMK_THETA_U_POS = 0.120
IMK_THETA_U_NEG = 0.120
IMK_D_POS = 1.0
IMK_D_NEG = 1.0

# ---- Beam-column joints in the nonlinear frame (user decision 2026-09-27) ---------------------
# The intended split: IMKPeakOriented for member flexure, IMKPinching for joint shear and slip.
#   "rigid_centerline"       the legacy frame: beams and columns meet at one node, no joint spring.
#   "imk_pinching_scissors"  two coincident cores per elevated joint (the joint node stays the
#                            column core; a beam core shares its translations and vertical-axis
#                            rotation) joined by IMKPinching springs in the two vertical shear
#                            planes. Centreline geometry is kept. IMK formulation only; the
#                            elastic design model is unchanged. See Model/Joint_Springs.py.
JOINT_MODEL = "imk_pinching_scissors"
# What the joint spring's rotation stands for: panel shear only. Bar slip is owned per member end
# (Model/Deformation_Ownership): the member hinge keeps Haselton's a_sl = 1 unless a face slip
# interface is registered for that end. The 2026-09-27 value "joint_shear_and_slip" (which set
# a_sl = 0 in every hinge, column bases included, with nothing replacing it) is retired and refused
# at build time; outputs written under it carry that value in their identity and are not comparable.
JOINT_DEFORMATION_SCOPE = "joint_shear_only"
# Strength from ACI 318-19 Table 18.8.4.3 (gamma, Aj from the design record's joint-shear category
# when it has one), deformation from ASCE/SEI 41-17 Table 10-11 for conforming joints, scissors
# conjugacy Mn = Vn h_b and K = G Aj h_b. Transcribed, not experimentally calibrated.
JOINT_CALIBRATION_BASIS = "aci318_t18_8_4_3_strength__asce41_17_t10_11_deformation__scissors_v1"
JOINT_CALIBRATION_STATUS = "provisional_transcribed_not_experimentally_calibrated"
JOINT_STIFFNESS_MODIFIER = 1.0        # on G Aj h_b of the uncracked panel
JOINT_KAPPA_F = 0.25                  # IMKPinching break-point ratios: Ibarra, Medina & Krawinkler (2005)
JOINT_KAPPA_D = 0.25                  #   representative pinching level, not an RC joint fit
JOINT_LAMBDA_SUPPRESSION = 1.0e12     # no cyclic deterioration in the joint law (every mode suppressed)
JOINT_SHEAR_CATEGORIES = None         # the design record's capacity_design.joints.joints, set by apply_design
JOINT_BEAM_CORE_TAG_BASE = 1500000    # nodes; below the IMK hinge nodes (2000000+)
JOINT_ELEMENT_TAG_BASE = 1500000      # elements; above the members, below the IMK hinges (4000000+)
JOINT_MATERIAL_TAG_BASE = 1500000     # materials; below the IMK materials (3000000+)

# Effective (cracked) section stiffness for seismic analysis.
# ACI 318-19 Table 6.6.3.1.1(a): beams 0.35*Ig, columns 0.70*Ig.
#
# These apply only to members built as elastic elements with concentrated
# IMK springs, where cracking is not represented by the material model.
# Fiber-section members capture cracking through Concrete02 directly, so
# applying a modifier to them would double-count the stiffness reduction.
CRACKED_SECTION_ANALYSIS = True
BEAM_STIFFNESS_MODIFIER = 0.35
COLUMN_STIFFNESS_MODIFIER = 0.70


def section_stiffness_modifier(member_type):
    """Effective-stiffness multiplier on Ig for an elastic IMK member."""
    if not CRACKED_SECTION_ANALYSIS:
        return 1.0
    return COLUMN_STIFFNESS_MODIFIER if member_type == "column" else BEAM_STIFFNESS_MODIFIER


# Beam flexural stiffness in the frame models: the gross T/L section on the
# ACI 318-19 6.3.2 effective flange.
#
# R6.6.3.1.1 says the moment of inertia of a T-beam should be based on the
# effective flange width of 6.3.2 (and that 2 x the web Ig is the usual
# approximation); Table 6.6.3.1.1(a)'s 0.35 then applies to that Ig. The
# beam strengths already use the 6.3.2 flange (Design/SMRF_Beam_Slab_Strength),
# so this makes the analysis section the same section that is checked. The
# 150-case verification of 2026-09-16 (dv150_v9) found the rectangular web
# 5.0-5.4% off the monolithic slab/frame reference on column vertical
# reactions for every 2-bay plan; the 6.3.2 T/L section brings all of them
# to 1.0-1.3% and every control to <= 1.6%, and shortens periods 12-17%.
#
# Without a slab (legacy load mode) the beam is the rectangular web. The
# floor plate routine keeps its own 8.4.1.8 beam lines: that is the two-way
# slab convention for alpha_f, and its shells carry the slab there.
def effective_flange_width_in(bw, slab_h, clear_span, clear_to_adjacent_web, slab_sides):
    """ACI 318-19 Table 6.3.2.1 total flange width for a beam with slab on 1 or 2 sides."""
    if slab_sides == 2:
        overhang = min(8.0 * slab_h, clear_to_adjacent_web / 2.0, clear_span / 8.0)
    elif slab_sides == 1:
        overhang = min(6.0 * slab_h, clear_to_adjacent_web / 2.0, clear_span / 12.0)
    else:
        raise ValueError("slab_sides must be 1 or 2.")
    return bw + slab_sides * overhang, overhang


def t_section_inertia_in4(bw, h, slab_h, flange_width):
    """Gross I of a T/L section (web bw x h including the flange depth) about its own centroid."""
    web_h = h - slab_h
    if web_h <= 0.0 or flange_width < bw:
        raise ValueError("The slab must be thinner than the beam and the flange at least the web width.")
    flange_area, web_area = flange_width * slab_h, bw * web_h
    flange_y, web_y = slab_h / 2.0, slab_h + web_h / 2.0
    centroid = (flange_area * flange_y + web_area * web_y) / (flange_area + web_area)
    return (flange_width * slab_h ** 3 / 12.0 + flange_area * (flange_y - centroid) ** 2
            + bw * web_h ** 3 / 12.0 + web_area * (web_y - centroid) ** 2)


def beam_flange_geometry(axis, position):
    """Clear span, clear distance to the adjacent web and slab sides for a beam family.

    ``axis`` 'x' or 'y' (the beam's direction); ``position`` 'edge' (a
    perimeter line, slab on one side) or 'interior'. The clear span is the
    bay less the column dimension parallel to the beam (H_COL along x,
    B_COL along y, the joint adapter's convention).
    """
    if axis == "x":
        clear_span, clear_web = BAY_X - H_COL, BAY_Y - B_BEAM
    elif axis == "y":
        clear_span, clear_web = BAY_Y - B_COL, BAY_X - B_BEAM
    else:
        raise ValueError("axis must be 'x' or 'y'.")
    if position not in ("edge", "interior"):
        raise ValueError("position must be 'edge' or 'interior'.")
    return clear_span, clear_web, (1 if position == "edge" else 2)


def beam_flexural_section(axis, position):
    """Gross vertical-bending I and its basis for one beam family.

    Returns {"iy_in4", "flange_width_in", "overhang_in", "basis"}. With a
    slab: the T/L section on the 6.3.2 flange. Without one: the rectangular
    web, as before. This is what SMRF_Elastic, the IMK elastic elements and
    the SAP export use; the stiffness modifier is applied by the caller.
    """
    if SLAB_THICKNESS_IN is None:
        return {"iy_in4": rect_iy(B_BEAM, H_BEAM), "flange_width_in": B_BEAM, "overhang_in": 0.0,
                "basis": "rectangular web (no slab in the model)"}
    clear_span, clear_web, sides = beam_flange_geometry(axis, position)
    flange_width, overhang = effective_flange_width_in(B_BEAM, SLAB_THICKNESS_IN, clear_span, clear_web, sides)
    return {"iy_in4": t_section_inertia_in4(B_BEAM, H_BEAM, SLAB_THICKNESS_IN, flange_width),
            "flange_width_in": flange_width, "overhang_in": overhang,
            "basis": "gross T/L section on the ACI 318-19 6.3.2 effective flange (R6.6.3.1.1)"}


def beam_flexural_inertia_in4(axis, position):
    """Gross vertical-bending I of a beam family for the frame models (see beam_flexural_section)."""
    return beam_flexural_section(axis, position)["iy_in4"]


# Fiber section mesh settings
CORE_PATCH_NY = 12
CORE_PATCH_NZ = 12

COVER_PATCH_N_LONG = 12
COVER_PATCH_N_SHORT = 2

# Analysis settings
GRAVITY_TOL = 1.0e-5
GRAVITY_MAX_ITER = 100
GRAVITY_STEPS = 20

PUSHOVER_TOL = 1.0e-5
PUSHOVER_MAX_ITER = 50
PUSHOVER_FALLBACK_TOL = 5.0e-5
PUSHOVER_FALLBACK_MAX_ITER = 100

PUSHOVER_STEPS = 1200
PUSHOVER_DU = 0.05
PUSHOVER_DEBUG_EVERY = 10
PUSHOVER_TRACK_DESIGN_ENVELOPE = True
PUSHOVER_DESIGN_ENVELOPE_EVERY = 5
PUSHOVER_DESIGN_ENVELOPE_EVENT_MIN_STEP_GAP = 5
PUSHOVER_DESIGN_ENVELOPE_PEAK_SHEAR_REL_STEP = 0.05
PUSHOVER_DESIGN_ENVELOPE_DRIFT_JUMP_REL_STEP = 0.10
PUSHOVER_DCR_CHECK_ROOF_DRIFT_RATIO = 0.02
PUSHOVER_USE_TARGET_DCR_FOR_REDESIGN = True
PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO = 0.05
PUSHOVER_REDESIGN_MAX_ATTEMPTS = 3
PUSHOVER_ENVELOPE_REDESIGN_MAX_ATTEMPTS = 6
PUSHOVER_ENVELOPE_STOP_ON_STALLED_DCR = True
PUSHOVER_ENVELOPE_STALL_MIN_RERUNS = 2
PUSHOVER_ENVELOPE_STALL_DCR_TOL = 0.01
PUSHOVER_STOP_ON_LOAD_REVERSAL = True
PUSHOVER_LOAD_REVERSAL_LIMIT = 0.0
PUSHOVER_LOAD_REVERSAL_MIN_PEAK_FACTOR = 0.02

NUM_MODES = NUM_FLOOR + 2
EIGENVALUE_TOL = 1.0e-8

# ASCE 7 approximate fundamental period sanity check.
# US customary form: Ta = Ct * h_n^x, with h_n in feet.
# Default constants are for concrete moment-resisting frames.
ASCE_PERIOD_SYSTEM = "concrete_moment_frame"
ASCE_PERIOD_CT = 0.016
ASCE_PERIOD_EXPONENT = 0.90
ASCE_PERIOD_WARN_RATIO_LOW = 0.50
ASCE_PERIOD_WARN_RATIO_HIGH = 2.00

# ASCE 7-22 seismic design parameters for the equivalent lateral force
# procedure (Loads/Seismic_ELF.py), which drives the iterative design loop.
#
# Defaults describe a high-seismicity site with a special reinforced concrete
# moment frame. High seismicity is deliberate: the dataset needs structures
# whose design is governed by earthquake demand rather than gravity, otherwise
# member sizes barely respond to building height.
ASCE_SDS = 1.00     # design spectral response acceleration, short period
ASCE_SD1 = 0.60     # design spectral response acceleration, 1 second
ASCE_S1 = 0.60      # mapped MCE spectral acceleration at 1 second
ASCE_R = 8.0        # response modification factor, special RC moment frame
ASCE_IE = 1.0       # importance factor, risk category II
ASCE_CU = 1.4       # upper-limit coefficient on the calculated period
ASCE_TL = 8.0       # long-period transition period, seconds

# ACI 318-19 18.7.3 strong column / weak beam: sum(Mnc) >= 1.2 * sum(Mnb).
# Columns are capacity-protected rather than designed to the DCR band, so this
# ratio, not the band, is what sizes them.
SCWB_RATIO_MIN = 1.2

# Seismic hazard varied per generated case.
#
# Geometry alone barely moves design demand: a minimum-size column is adequate
# for most low-rise frames, so height and span variation alone produced nearly
# identical sections. Varying the design spectral accelerations changes the
# base shear by roughly 3x across this list, which is what actually
# differentiates one designed structure from another.
#
# Each entry is (label, SDS, SD1, S1). Values span Seismic Design Category C
# through E; lower categories are excluded because a special moment frame
# (R = 8) would not be the system of choice there, and gravity would govern
# the design and flatten the variety again.
SEISMIC_SITE_OPTIONS = (
    # SDC C per ASCE 7-22 Tables 11.6-1/11.6-2 needs SDS < 0.50 and SD1 < 0.20;
    # the earlier 0.50/0.25 pair sat on the D thresholds and derived as SDC D.
    ("sdc_c",      0.40, 0.19, 0.19),
    ("sdc_d_low",  0.75, 0.38, 0.38),
    ("sdc_d_high", 1.00, 0.60, 0.60),
    ("sdc_e",      1.25, 0.75, 0.75),
    ("sdc_e_near", 1.50, 0.90, 0.90),
)

SEISMIC_SITE_LABEL = "sdc_d_high"


def seismic_site_by_label(label):
    """Look up a (label, SDS, SD1, S1) hazard entry by its label."""
    for entry in SEISMIC_SITE_OPTIONS:
        if entry[0] == label:
            return entry
    raise ValueError(
        f"Unknown seismic site label {label!r}; expected one of "
        f"{[entry[0] for entry in SEISMIC_SITE_OPTIONS]}."
    )


def apply_seismic_site(label):
    """Set the ASCE design accelerations from a named hazard entry."""
    global SEISMIC_SITE_LABEL, ASCE_SDS, ASCE_SD1, ASCE_S1
    label, sds, sd1, s1 = seismic_site_by_label(label)
    SEISMIC_SITE_LABEL, ASCE_SDS, ASCE_SD1, ASCE_S1 = label, sds, sd1, s1
    return label


def building_height_ft():
    """Total structural height in feet."""
    return NUM_FLOOR * STORY_H / 12.0


def asce_approx_fundamental_period_sec():
    """ASCE approximate fundamental period Ta = Ct * h_n^x."""
    return ASCE_PERIOD_CT * building_height_ft() ** ASCE_PERIOD_EXPONENT


def asce_fundamental_period_check(model_period_sec=None, modal_source=None):
    """
    Return a compact period sanity check against ASCE approximate Ta.

    This is intended as an early flag, not a design acceptance criterion.
    """
    ta = asce_approx_fundamental_period_sec()
    check = {
        "equation": "Ta = Ct * h_n^x",
        "system": ASCE_PERIOD_SYSTEM,
        "ct": ASCE_PERIOD_CT,
        "exponent": ASCE_PERIOD_EXPONENT,
        "height_ft": building_height_ft(),
        "asce_ta_sec": ta,
        "modal_source": modal_source,
        "model_period_sec": model_period_sec,
        "period_ratio_to_asce_ta": None,
        "warn_ratio_low": ASCE_PERIOD_WARN_RATIO_LOW,
        "warn_ratio_high": ASCE_PERIOD_WARN_RATIO_HIGH,
        "passes_reasonableness_check": None,
        "warning": None,
    }

    if model_period_sec is None or ta <= 0.0:
        return check

    ratio = model_period_sec / ta
    check["period_ratio_to_asce_ta"] = ratio
    check["passes_reasonableness_check"] = (
        ASCE_PERIOD_WARN_RATIO_LOW <= ratio <= ASCE_PERIOD_WARN_RATIO_HIGH
    )

    if ratio > ASCE_PERIOD_WARN_RATIO_HIGH:
        check["warning"] = (
            "Model period is much longer than ASCE approximate Ta; "
            "check mass, stiffness, hinges, gravity state, and constraints."
        )
    elif ratio < ASCE_PERIOD_WARN_RATIO_LOW:
        check["warning"] = (
            "Model period is much shorter than ASCE approximate Ta; "
            "check mass, stiffness, rigid constraints, and units."
        )

    return check


# Primary post-gravity analysis mode.
# Options:
#   "ntha"     = nonlinear time-history analysis from Ground_Motions manifest
#   "pushover" = static nonlinear pushover with redesign/envelope checks
ANALYSIS_METHOD = "ntha"

# NTHA / ground-motion settings used by Main.py when ANALYSIS_METHOD = "ntha".
NTHA_SET_NAME = "peer_mle_all"
NTHA_SPLIT = None
NTHA_RESULT_ID = 1
NTHA_RECORD_ID_X = None
NTHA_RECORD_ID_Y = None
NTHA_X_ONLY = False
NTHA_SCALE_FACTOR = 2.25  # requested total amplitude multiplier; apply once
NTHA_DAMPING_RATIO = 0.05
NTHA_RAYLEIGH_MODE_I = 0
NTHA_RAYLEIGH_MODE_J = 2
NTHA_DT_FACTOR = 1.0
# Hinge rotation histories are the largest response array by far: one value
# per spring per direction per step. Peaks are tracked every step so no peak
# is missed; the history itself is decimated, since it exists to show the
# shape of the hysteresis rather than to resolve every cycle exactly.
NTHA_HINGE_HISTORY_STRIDE = 8

# Element end forces and assembled joint forces share the hinge stride by
# default. The analysis already queries every element every step for the
# envelope, so storing them costs storage rather than compute.
NTHA_ELEMENT_HISTORY_STRIDE = 8

NTHA_PROGRESS_EVERY = 100
NTHA_PRINT_CATALOG_SUMMARY = True

PENALTY_ALPHA_SP = 1.0e9
PENALTY_ALPHA_MP = 1.0e9


def concrete_ec_ksi(fc_ksi):
    fc_psi = fc_ksi * 1000.0
    return 57000.0 * math.sqrt(fc_psi) / 1000.0


def concrete_shear_modulus_ksi(ec_ksi):
    return 0.4 * ec_ksi


def rect_area(b, h):
    return b * h


def rect_iy(b, h):
    """Second moment of area about the local y-axis: Iy = integral(z^2) dA.

    Arguments follow the section's local frame, matching the OpenSees
    elasticBeamColumn signature (..., Iy, Iz, transfTag):
      b = width  measured along local y
      h = depth  measured along local z

    Bending in the local x-z plane -- vertical bending for a floor beam, whose
    geomTransf vecxz is (0, 0, 1) -- is governed by Iy, so this must be the
    strong-axis value for a beam deeper than it is wide.
    """
    return b * h**3 / 12.0


def rect_iz(b, h):
    """Second moment of area about the local z-axis: Iz = integral(y^2) dA."""
    return h * b**3 / 12.0


def approx_rect_j(b, h):
    return rect_iy(b, h) + rect_iz(b, h)


def rc_nominal_moment_ksi(fc_ksi, width_in, depth_in, steel_area_in2, member_type=None):
    # Generic legacy callers retain their former explicit COVER convention.
    cover = COVER if member_type is None else longitudinal_cover_in(member_type)
    d = depth_in - cover
    a = steel_area_in2 * FY_KSI / (0.85 * fc_ksi * width_in)
    a = min(a, 0.85 * depth_in)
    return steel_area_in2 * FY_KSI * (d - 0.5 * a)


def beam_nominal_moment_y():
    steel_area = max(BEAM_TOP_BARS, BEAM_BOT_BARS) * BEAM_BAR_AREA
    return rc_nominal_moment_ksi(FC_BEAM_KSI, B_BEAM, H_BEAM, steel_area, "beam")


def beam_nominal_moment_z():
    steel_area = max(BEAM_TOP_BARS, BEAM_BOT_BARS) * BEAM_BAR_AREA
    return rc_nominal_moment_ksi(FC_BEAM_KSI, H_BEAM, B_BEAM, steel_area, "beam")


def column_nominal_moment_y():
    steel_area = max(COL_TOP_BARS, COL_BOT_BARS) * COL_BAR_AREA
    return rc_nominal_moment_ksi(FC_COL_KSI, B_COL, H_COL, steel_area, "column")


def column_nominal_moment_z():
    steel_area = max(COL_TOP_BARS, COL_BOT_BARS) * COL_BAR_AREA
    return rc_nominal_moment_ksi(FC_COL_KSI, H_COL, B_COL, steel_area, "column")


# ASCE 7-22 2.3.6 load combination 6 for seismic design: 1.2D + Ev + Eh + 0.5L.
# The floor load lumps dead and live together, so the design loop applies a
# single equivalent factor to it and the full dead factor to self weight.
SEISMIC_COMBINATION_DEAD_FACTOR = 1.2
SEISMIC_COMBINATION_LIVE_FACTOR = 0.5


def seismic_combination_floor_factor():
    """Equivalent factor on the combined dead+live floor load for 1.2D + 0.5L."""
    dead = floor_dead_load_ksf()
    total = dead + FLOOR_LIVE_LOAD_KSF
    if total <= 0.0:
        return SEISMIC_COMBINATION_DEAD_FACTOR
    factored = (
        SEISMIC_COMBINATION_DEAD_FACTOR * dead
        + SEISMIC_COMBINATION_LIVE_FACTOR * FLOOR_LIVE_LOAD_KSF
    )
    return factored / total


def floor_load_ksi():
    """Total gravity floor load converted to kip/in²."""
    return (floor_dead_load_ksf() + FLOOR_LIVE_LOAD_KSF) / 144.0


def _validate_slab_load_state():
    """Reject invalid selected slab/load geometry instead of clipping weights."""
    if SLAB_THICKNESS_IN is None:
        return
    positive = {
        "SLAB_THICKNESS_IN": SLAB_THICKNESS_IN, "H_BEAM": H_BEAM,
        "STORY_H": STORY_H, "B_BEAM": B_BEAM, "B_COL": B_COL,
        "H_COL": H_COL, "BAY_X": BAY_X, "BAY_Y": BAY_Y,
        "CONCRETE_UNIT_WEIGHT_KCF": CONCRETE_UNIT_WEIGHT_KCF,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive in slab-aware mode.")
    if not SLAB_THICKNESS_IN < H_BEAM < STORY_H:
        raise ValueError("Slab-aware loads require SLAB_THICKNESS_IN < H_BEAM < STORY_H.")
    if H_COL >= BAY_X or B_COL >= BAY_Y:
        raise ValueError("Slab-aware beam self-weight requires positive clear spans in X and Y.")
    for name, value in (("FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF", FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF),
                        ("FLOOR_LIVE_LOAD_KSF", FLOOR_LIVE_LOAD_KSF)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not math.isfinite(SEISMIC_LIVE_LOAD_FRACTION) or not 0 <= SEISMIC_LIVE_LOAD_FRACTION <= 1:
        raise ValueError("SEISMIC_LIVE_LOAD_FRACTION must be finite and between 0 and 1.")


def slab_self_weight_ksf():
    """Selected slab weight; legacy bundled dead load has no known slab share."""
    if SLAB_THICKNESS_IN is None:
        return None
    _validate_slab_load_state()
    return CONCRETE_UNIT_WEIGHT_KCF * SLAB_THICKNESS_IN / 12.0


def floor_dead_load_ksf():
    """Area dead load, without separately applied beam/column self-weight."""
    if SLAB_THICKNESS_IN is None:
        return FLOOR_DEAD_LOAD_KSF
    return FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF + slab_self_weight_ksf()


def node_tributary_area_in2(i, j):
    """
    Tributary floor area (in²) for a structural node at grid position (i, j).
    Corner nodes receive a quarter bay, edge nodes a half bay, interior nodes
    a full bay in each direction.
    """
    tx = (BAY_X / 2.0) if (i == 0 or i == NUM_BAY_X) else BAY_X
    ty = (BAY_Y / 2.0) if (j == 0 or j == NUM_BAY_Y) else BAY_Y
    return tx * ty


def node_gravity_load_kip(i, j):
    """Tributary gravity load (kips, downward) at grid node (i, j)."""
    return floor_load_ksi() * node_tributary_area_in2(i, j)


def node_seismic_mass(i, j):
    """Translational seismic mass (kip·s²/in) at grid node (i, j)."""
    if SLAB_THICKNESS_IN is None:
        return node_gravity_load_kip(i, j) / G
    area_weight = ((floor_dead_load_ksf() + SEISMIC_LIVE_LOAD_FRACTION * FLOOR_LIVE_LOAD_KSF)
                   * node_tributary_area_in2(i, j) / 144.0)
    return (area_weight + node_structural_self_weight_kip(i, j)) / G


def total_floor_gravity_load():
    """Total gravity load per floor (kips)."""
    return floor_load_ksi() * (BAY_X * NUM_BAY_X) * (BAY_Y * NUM_BAY_Y)


def beam_tributary_width_in(beam_axis, perp_index):
    """
    Tributary width (in) for a beam.

    beam_axis   : 'x' or 'y'
    perp_index  : position index perpendicular to the beam span.
                  For X-beams this is the j-index (Y-direction).
                  For Y-beams this is the i-index (X-direction).
    """
    if beam_axis == "x":
        n_max  = NUM_BAY_Y
        bay_perp = BAY_Y
    else:
        n_max  = NUM_BAY_X
        bay_perp = BAY_X
    return (bay_perp / 2.0) if (perp_index == 0 or perp_index == n_max) else bay_perp


def beam_gravity_wz_kip_per_in(beam_axis, perp_index):
    """Uniform downward load (kip/in, negative) on a beam at perp_index."""
    trib = beam_tributary_width_in(beam_axis, perp_index)
    return -floor_load_ksi() * trib


def pushover_target_disp():
    return PUSHOVER_STEPS * PUSHOVER_DU


def col_self_weight_kip_per_in():
    """Equivalent column line weight; new mode excludes the slab-height slice."""
    if SLAB_THICKNESS_IN is not None:
        _validate_slab_load_state()
        return (CONCRETE_UNIT_WEIGHT_KCF / 1728.0 * B_COL * H_COL
                * (STORY_H - SLAB_THICKNESS_IN) / STORY_H)
    return CONCRETE_UNIT_WEIGHT_KCI * B_COL * H_COL


def beam_drop_weight_kip_per_in():
    """Physical weight per inch of the beam between the joint faces.

    Slab-aware mode: the drop below the slab, b (h - t); legacy mode the
    full section. beam_self_weight_kip_per_in smears this over the full
    centerline element (times the clear fraction); this is the unreduced
    value for a free body cut at the column faces.
    """
    if SLAB_THICKNESS_IN is not None:
        _validate_slab_load_state()
        return CONCRETE_UNIT_WEIGHT_KCF / 1728.0 * B_BEAM * (H_BEAM - SLAB_THICKNESS_IN)
    return CONCRETE_UNIT_WEIGHT_KCI * B_BEAM * H_BEAM


def beam_self_weight_kip_per_in(beam_axis=None):
    """Equivalent beam line weight, applied over the centerline element.

    Slab-aware mode counts only the drop below the slab, between column faces;
    beam section/stiffness dimensions are unchanged. An explicit axis is needed
    because X/Y column face dimensions and spans need not match.
    """
    if SLAB_THICKNESS_IN is not None:
        _validate_slab_load_state()
        if beam_axis not in {"x", "y"}:
            raise ValueError("Slab-aware beam self-weight requires beam_axis='x' or 'y'.")
        clear_fraction = (1.0 - H_COL / BAY_X) if beam_axis == "x" else (1.0 - B_COL / BAY_Y)
        return CONCRETE_UNIT_WEIGHT_KCF / 1728.0 * B_BEAM * (H_BEAM - SLAB_THICKNESS_IN) * clear_fraction
    return CONCRETE_UNIT_WEIGHT_KCI * B_BEAM * H_BEAM


def total_structural_self_weight_per_floor():
    """
    Total self-weight of all column and beam elements in one floor (kips).

    Columns  : NUM_FLOOR stories × (NUM_BAY_X+1) × (NUM_BAY_Y+1) columns,
               each STORY_H tall.
    Beams    : X-beams + Y-beams, each spanning one bay (BAY_X or BAY_Y).
    """
    n_col    = (NUM_BAY_X + 1) * (NUM_BAY_Y + 1)
    n_beam_x = NUM_BAY_X * (NUM_BAY_Y + 1)
    n_beam_y = (NUM_BAY_X + 1) * NUM_BAY_Y
    w_col_floor  = col_self_weight_kip_per_in()  * STORY_H * n_col
    w_beam_floor = (beam_self_weight_kip_per_in("x") * BAY_X * n_beam_x
                    + beam_self_weight_kip_per_in("y") * BAY_Y * n_beam_y)
    return w_col_floor + w_beam_floor


def node_structural_self_weight_kip(i, j):
    """One column story plus half of every incident beam's accounted weight.

    This is a mass/tributary-axial helper, NOT an extra nodal gravity load:
    Gravity_Loads applies the same member weights as distributed element loads.
    One whole column story is lumped at its upper floor (roof included).
    """
    if not (0 <= i <= NUM_BAY_X and 0 <= j <= NUM_BAY_Y):
        raise ValueError("Node indices must lie on the structural floor grid.")
    incident_x = int(i > 0) + int(i < NUM_BAY_X)
    incident_y = int(j > 0) + int(j < NUM_BAY_Y)
    return (col_self_weight_kip_per_in() * STORY_H
            + 0.5 * incident_x * BAY_X * beam_self_weight_kip_per_in("x")
            + 0.5 * incident_y * BAY_Y * beam_self_weight_kip_per_in("y"))


def total_floor_seismic_weight():
    """Floor weight matching exactly the mass assigned to structural grid nodes."""
    if SLAB_THICKNESS_IN is None:
        return total_floor_gravity_load()
    area = BAY_X * NUM_BAY_X * BAY_Y * NUM_BAY_Y
    return ((floor_dead_load_ksf() + SEISMIC_LIVE_LOAD_FRACTION * FLOOR_LIVE_LOAD_KSF) * area / 144.0
            + total_structural_self_weight_per_floor())


def floor_load_metadata():
    """Persist the load model without changing the existing numeric feature order."""
    selected = SLAB_THICKNESS_IN is not None
    dead = floor_dead_load_ksf()
    area_sqft = BAY_X * NUM_BAY_X * BAY_Y * NUM_BAY_Y / 144.0
    column_weight = (NUM_BAY_X + 1) * (NUM_BAY_Y + 1) * STORY_H * col_self_weight_kip_per_in()
    beam_weight_x = NUM_BAY_X * (NUM_BAY_Y + 1) * BAY_X * beam_self_weight_kip_per_in("x")
    beam_weight_y = NUM_BAY_Y * (NUM_BAY_X + 1) * BAY_Y * beam_self_weight_kip_per_in("y")
    return {
        "load_model": "slab_aware_v1" if selected else "legacy_bundled_v1",
        "slab_thickness_in": SLAB_THICKNESS_IN,
        "floor_superimposed_dead_load_ksf": FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF if selected else None,
        "slab_self_weight_ksf": slab_self_weight_ksf(),
        "floor_dead_load_ksf": dead,
        "floor_live_load_ksf": FLOOR_LIVE_LOAD_KSF,
        "seismic_live_load_fraction": SEISMIC_LIVE_LOAD_FRACTION if selected else 1.0,
        "floor_area_sqft": area_sqft,
        "floor_area_dead_weight_kip": dead * area_sqft,
        "total_floor_gravity_load_kip": total_floor_gravity_load(),
        "column_self_weight_per_floor_kip": column_weight,
        "beam_x_self_weight_per_floor_kip": beam_weight_x,
        "beam_y_self_weight_per_floor_kip": beam_weight_y,
        "member_self_weight_per_floor_kip": column_weight + beam_weight_x + beam_weight_y,
        "total_floor_seismic_weight_kip": total_floor_seismic_weight(),
        "load_accounting_basis": (
            "Uniform slab over centerline floor footprint; beam drops over clear spans between column faces; "
            "columns over story height minus slab thickness. Equivalent member weights distributed over "
            "centerline elements. One column story lumped at its upper floor plus half each incident beam. "
            "Perimeter surfaces and joints remain centerline-model idealizations, not a physical quantity takeoff."
            if selected else
            "Legacy bundled slab/finishes/MEP area dead load; full rectangular centerline member gravity weights; "
            "seismic mass uses floor D+100%L only and omits member self-weight."
        ),
    }
