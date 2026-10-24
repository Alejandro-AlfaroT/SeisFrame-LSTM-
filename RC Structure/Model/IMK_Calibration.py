"""
Model/IMK_Calibration.py
========================

Per-member IMK backbone calibration.

Before this module the hinge backbone was a set of global constants: every
element in every structure used theta_p = 0.020, theta_pc = 0.060 and a yield
moment computed with no axial load. That is defensible for beams, where axial
force is negligible, but it is wrong for columns, whose moment capacity and
plastic rotation capacity both depend strongly on how hard they are being
squeezed. With one constant backbone, a lightly loaded roof column and a
heavily loaded ground-floor column degrade identically, and no soft-story
mechanism can emerge from the model no matter what the ground motion does.

Two calibrations are provided:

  * yield moment from the nominal P-M interaction surface at the member's
    axial load, rather than the zero-axial flexural capacity
  * plastic and post-capping rotation capacity from Haselton et al. (2008),
    PEER 2007/03, which are the calibrations underlying the ASCE 41 and
    FEMA P-58 concrete hinge parameters

Beams are the zero-axial case of the same equations, so both member types go
through one path.

Unit system: kip, inch, ksi. The Haselton expressions are dimensional and
were fit in MPa, so concrete strength and steel yield are converted locally.
"""

from __future__ import annotations

import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model import Member_Groups as mg
from RC_Design_Check import _beta1, _col_steel_layers


KSI_TO_MPA = 6.894757

# Haselton et al. (2008) fit bounds. Outside these the expressions are
# extrapolation, so inputs are clamped and the clamp is reported.
NU_MIN, NU_MAX = 0.0, 0.70
RHO_SH_MIN, RHO_SH_MAX = 0.002, 0.020

# a_sl = 1 where bar slip from the joint is possible, which is the normal
# condition for a cast-in-place frame and the value Haselton's regression
# carries for the member. Since 2026-09-27 (evening) the term is owned per
# member end (Model/Deformation_Ownership): a member hinge keeps it unless a
# face slip interface is registered for that end. A declared joint scope
# never removes it, and a column base never loses it without a replacement.
BOND_SLIP_INDICATOR = 1.0


def bond_slip_indicator(end=None, *, member_scope=None):
    """Haselton's a_sl for one member end (Model/Deformation_Ownership.member_end descriptor).

    Without an end the policy's member default is returned (no interface can be registered without
    an end), which is what the output identity and legacy callers report.
    """
    from Model import Deformation_Ownership as ownership
    if end is None:
        return ownership.DEFAULT_MEMBER_INDICATOR
    return ownership.bond_slip_indicator_for_end(end, member_scope=member_scope)

THETA_P_FLOOR = 0.005
THETA_PC_FLOOR = 0.010
THETA_PC_CAP = 0.10

# Haselton et al. (2008), PEER 2007/03, notation (p. xvi): rho is the ratio of the TOTAL area of longitudinal
# reinforcement to b d (rho_t + rho', where rho' includes the web steel), and d is the effective depth, d = h - d'.
# Equation 3.10 keeps that b d denominator; the later NIST applicability summary's As/Ag belongs to another
# equation family and is not used here. (V6 IMK audit corrections, 2026-10-23: the numerator used to be the
# larger face only, max(top, bottom) bars, and Eq. 3.20 used s/h for s/d.)
RHO_DEFINITION = "haselton_2008_eq3_10_total_longitudinal_steel_over_b_d_v2"
EFFECTIVE_DEPTH_DEFINITION = "effective_depth_from_tension_steel_centroid_v1"
EQ_3_20_VERSION = "haselton_2008_eq3_20_spacing_over_effective_depth_v2"


def _clamp(value, low, high):
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Nominal P-M interaction
# ---------------------------------------------------------------------------

def column_pm_nominal(n_pts=120):
    """Nominal (unfactored) column P-M interaction surface.

    RC_Design_Check.build_column_PM_diagram returns phi-factored capacities
    because it exists to check designs. A hinge represents actual strength,
    not design strength, so the strength-reduction factor must not appear
    here. The sweep mirrors that function so the two stay comparable.

    Returns a list of (Pn_kip, Mn_kip_in), compression positive.
    """
    return column_pm_nominal_for(
        sp.B_COL, sp.H_COL, sp.FC_COL_KSI, _col_steel_layers(), n_pts=n_pts
    )


def column_pm_nominal_for(b, h, fc, layers, n_pts=120):
    """Nominal P-M surface for an explicitly given column section.

    Same sweep as column_pm_nominal, but the section is passed in rather than
    read from the Structure_Parameters globals. The design ladder needs the
    capacity of sections it is only considering, and mutating the globals to
    price a candidate would corrupt the model state mid-search.
    """
    fy = sp.FY_KSI
    es = sp.ES_KSI

    ecu = 0.003
    b1 = _beta1(fc)
    ag = b * h
    ast = sum(area for area, _ in layers)
    hc = h / 2.0

    p0 = 0.85 * fc * (ag - ast) + fy * ast
    diagram = [(0.80 * p0, 0.0)]

    for index in range(n_pts):
        c = 0.001 + (4.0 * h - 0.001) * index / (n_pts - 1)
        a = min(b1 * c, h)
        compression = 0.85 * fc * b * a
        axial = compression
        moment = compression * (hc - a / 2.0)

        for area, depth in layers:
            strain = ecu * (c - depth) / c
            stress = _clamp(es * strain, -fy, fy)
            net = stress - (0.85 * fc if depth <= a else 0.0)
            axial += net * area
            moment += net * area * (hc - depth)

        diagram.append((axial, abs(moment)))

    # Nominal axial strength stops at 0.80 P0 (ACI 318-19 22.4.2.1, tied columns).
    from Design.ACI_Checks import truncate_at_axial_cap
    return sorted(truncate_at_axial_cap(diagram, 0.80 * p0), key=lambda point: point[0])


def column_axial_domain(diagram=None):
    """(tension end, compression cap) of a nominal surface, kip."""
    diagram = diagram or column_pm_nominal()
    axials = [point[0] for point in diagram]
    return min(axials), max(axials)


def column_moment_at_axial(axial_kip, diagram=None):
    """Nominal moment capacity at a given axial load, on the upper envelope.

    The surface is the upper strength envelope M(P) between its tension end
    and the nominal axial cap 0.80 P0 (column_pm_nominal_for cuts it there).
    Between sweep points the moment is interpolated on that envelope: where
    several segments bracket the load the largest moment is taken, so a
    load just below the cap reads the moment at the cap, not zero. Outside
    the domain the section has no flexural strength to offer -- above the
    cap it cannot carry the axial load at all, below the tension end it is
    past its tensile strength -- and the lookup returns 0.0 explicitly
    rather than clamping to an end point. Callers that cannot represent a
    section without strength (the hinges) must refuse such a load.
    """
    diagram = diagram or column_pm_nominal()
    low, high = column_axial_domain(diagram)
    if axial_kip < low or axial_kip > high:
        return 0.0
    points = sorted(diagram, key=lambda point: point[0])
    best = None
    for (p1, m1), (p2, m2) in zip(points, points[1:]):
        if p1 <= axial_kip <= p2:
            weight = (axial_kip - p1) / (p2 - p1) if p2 > p1 else 1.0
            moment = m1 + weight * (m2 - m1)
            best = moment if best is None else max(best, moment)
    if best is None:                                            # a single-point surface
        best = max(m for p, m in points if p == axial_kip)
    return best


# ---------------------------------------------------------------------------
# Axial demand estimate
# ---------------------------------------------------------------------------

def column_gravity_axial(story_index, grid_i, grid_j):
    """Gravity axial load in a column, kips, compression positive.

    Hinge properties are fixed when the model is built, before any analysis
    has run, so the axial load is estimated from tributary gravity rather
    than read from a solved state. Every floor at or above this column's top
    contributes its tributary load, plus the self weight of the columns above.

    story_index is one-based: story 1 is the column between the base and the
    first elevated floor. Under a grouped design every floor above contributes
    the weight of its own members (Model.Member_Properties).
    """
    from Model import Roof_Extension as roof
    if mg.is_grouped():
        from Model import Member_Properties as mp
        return mp.column_gravity_axial(story_index, grid_i, grid_j)
    floors_above = max(0, sp.NUM_FLOOR - story_index + 1)
    floor_load = sp.node_gravity_load_kip(grid_i, grid_j) * floors_above
    if sp.SLAB_THICKNESS_IN is not None:
        # New slab-aware idealization includes tributary beam drops as well as
        # columns, matching the gravity-weight inventory. Still a tributary
        # estimate, NOT a solved factored joint axial-force envelope.
        self_weight = sp.node_structural_self_weight_kip(grid_i, grid_j) * floors_above
    else:
        self_weight = sp.col_self_weight_kip_per_in() * sp.STORY_H * floors_above
    return floor_load + self_weight + roof.weight_kip(grid_i, grid_j)


def column_grid_position(node_tag):
    """Recover (story_index, grid_i, grid_j) from a column's lower node."""
    x, y, z = ops.nodeCoord(node_tag)
    grid_i = int(round(x / sp.BAY_X)) if sp.BAY_X > 0 else 0
    grid_j = int(round(y / sp.BAY_Y)) if sp.BAY_Y > 0 else 0
    floor_below = int(round(z / sp.STORY_H)) if sp.STORY_H > 0 else 0
    return floor_below + 1, grid_i, grid_j


def _section_values(member_type, design=None):
    """(b, h, fc, bar_size, bar_area, top, bot, hoop_size, hoop_legs, hoop_spacing) of one member.

    ``design`` is the member's own (Model.Member_Groups.MemberDesign); without it the single column or
    beam design of Structure_Parameters is read (the uniform mode, unchanged).
    """
    if design is not None:
        return (design.b_in, design.h_in, design.fc_ksi, design.bar_size, design.bar_area_in2, design.top_bars,
                design.bot_bars, design.stirrup_bar_size, design.stirrup_legs, design.stirrup_spacing_in)
    if member_type == "column":
        return (sp.B_COL, sp.H_COL, sp.FC_COL_KSI, sp.COL_BAR_SIZE, sp.COL_BAR_AREA, sp.COL_TOP_BARS, sp.COL_BOT_BARS,
                sp.COL_STIRRUP_BAR_SIZE, sp.COL_STIRRUP_LEGS, sp.COL_STIRRUP_SPACING)
    return (sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI, sp.BEAM_BAR_SIZE, sp.BEAM_BAR_AREA, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS,
            sp.BEAM_STIRRUP_BAR_SIZE, sp.BEAM_STIRRUP_LEGS, sp.BEAM_STIRRUP_SPACING)


def _longitudinal_cover(member_type, design=None):
    kind = "column" if member_type == "column" else "beam"
    if design is None:
        return sp.longitudinal_cover_in(kind)
    return sp.longitudinal_cover_in(kind, design.bar_size, design.stirrup_bar_size)


def axial_load_ratio(axial_kip, member_type, design=None):
    """Normalized axial load nu = P / (Ag f'c)."""
    b, h, strength = _section_values(member_type, design)[:3]
    gross_area = b * h
    if gross_area <= 0.0 or strength <= 0.0:
        return 0.0
    return max(0.0, axial_kip / (gross_area * strength))


# ---------------------------------------------------------------------------
# Haselton et al. (2008) rotation capacities
# ---------------------------------------------------------------------------

def transverse_steel_ratio(member_type, design=None):
    """rho_sh = Ash / (b * s) for the member's confinement."""
    values = _section_values(member_type, design)
    width, bar_size, legs, spacing = values[0], values[7], values[8], values[9]
    if spacing <= 0.0 or width <= 0.0:
        return RHO_SH_MIN
    return legs * sp.rebar_area(bar_size) / (width * spacing)


def total_longitudinal_bars(member_type, design=None):
    """Every physical longitudinal bar once: the two bar faces plus the side bars of BOTH side faces
    (``side_bars`` is the count per side face; Model/Sections lays that many bars down each side face)."""
    if design is not None:
        top, bot, side = design.top_bars, design.bot_bars, design.side_bars
    elif member_type == "column":
        top, bot, side = sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS
    else:
        top, bot, side = sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS, sp.BEAM_SIDE_BARS
    return int(top) + int(bot) + 2 * int(side)


def effective_depths(member_type, design=None, rows=None):
    """Effective depth d of each spring axis, from the bar geometry the member's strengths use (inches).

    Spring y bends across the section depth h and spring z across the width b (Model/IMK_Hinges). d is that
    dimension less the distance from the tension face to the centroid of the tension steel:

      * a column has one bar layer per face, every bar at the grid offset of its first layer from its face
        (Structure_Parameters.longitudinal_cover_in), so d = h - offset for y and d = b - offset for z;
      * a beam's vertical bending (y) reads the centroids of its actual bar rows, stacked layers and slab-mat
        clearances included (``rows``: {"top": {"centroid_in"}, "bottom": {...}} from Model/Member_Bar_Layers or
        Structure_Parameters.beam_bar_layers). Hogging puts the top steel in tension and sagging the bottom; the
        two values are recorded, and the canonical direction (hogging, the positive input at the member's start
        end, the reference of the cyclic energy baseline in Model/IMK_Materials) is the one used, because the
        2008 column regressions take one d and a beam's directional use is an unsupported extension;
      * a beam's lateral bending (z) has its side-face bars at the first-layer offset: d = b - offset.

    Without ``rows`` a beam's vertical-bending d uses the first-layer offset and says so (``rows_source``).
    """
    b, h = _section_values(member_type, design)[:2]
    offset = _longitudinal_cover(member_type, design)
    result = {"definition": EFFECTIVE_DEPTH_DEFINITION, "rows_source": None,
              "z": {"depth_dimension": "b", "dimension_in": b, "tension_centroid_offset_in": offset, "d_in": b - offset,
                    "basis": "one bar layer on the tension side face, at the first-layer offset"}}
    if member_type == "column":
        result["y"] = {"depth_dimension": "h", "dimension_in": h, "tension_centroid_offset_in": offset, "d_in": h - offset,
                       "basis": "one bar layer on the tension face, at the first-layer offset (symmetric cage)"}
        result["rows_source"] = "column cage: one layer per face"
    else:
        if rows is not None:
            top, bottom = float(rows["top"]["centroid_in"]), float(rows["bottom"]["centroid_in"])
            result["rows_source"] = "the member's bar rows (stacked layers and slab-mat clearances included)"
        else:
            top = bottom = offset
            result["rows_source"] = "first-layer offset: no bar rows supplied (single-layer assumption)"
        result["beam_vertical_bending"] = {"hogging_d_in": h - top, "sagging_d_in": h - bottom,
                                           "used": "hogging (canonical direction)",
                                           "directional_use": "not supported: one d per spring (2008 column regression)"}
        result["y"] = {"depth_dimension": "h", "dimension_in": h, "tension_centroid_offset_in": top, "d_in": h - top,
                       "basis": "top steel in tension (hogging, the canonical direction)"}
    for axis in ("y", "z"):
        if not math.isfinite(result[axis]["d_in"]) or result[axis]["d_in"] <= 0.0:
            raise ValueError(f"{member_type} effective depth for spring {axis} is not positive: {result[axis]}")
    return result


def longitudinal_steel_record(member_type, design=None, depths=None):
    """Eq. 3.10's rho with its inputs: total longitudinal steel over b d (RHO_DEFINITION), d of the y spring."""
    b, _h, _fc, _size, bar_area = _section_values(member_type, design)[:5]
    depths = depths if depths is not None else effective_depths(member_type, design)
    bars = total_longitudinal_bars(member_type, design)
    d = depths["y"]["d_in"]
    rho = bars * bar_area / (b * d) if b > 0.0 and d > 0.0 else 0.0
    return {"definition": RHO_DEFINITION, "rho": rho, "total_bars": bars, "bar_area_in2": bar_area,
            "b_in": b, "d_in": d, "beam_extrapolation": member_type != "column",
            "scope": ("column regression applied to the beam's web steel only (no slab steel, no directional "
                      "calibration)" if member_type != "column" else "column")}


def longitudinal_steel_ratio(member_type, design=None, depths=None):
    """rho = A_s,total / (b d) of Haselton et al. (2008) Eq. 3.10 (longitudinal_steel_record)."""
    return longitudinal_steel_record(member_type, design, depths)["rho"]


def _stability_index(member_type, design=None):
    """s_n = (s / d_b) * sqrt(fy_MPa / 100), the rebar buckling index."""
    values = _section_values(member_type, design)
    spacing = values[9]
    bar_diameter = sp.rebar_diameter(values[3])
    if bar_diameter <= 0.0:
        return 1.0
    return (spacing / bar_diameter) * math.sqrt(sp.FY_KSI * KSI_TO_MPA / 100.0)


def haselton_theta_p(member_type, nu, end=None, *, member_scope=None, design=None, depths=None):
    """Plastic rotation capacity to the capping point.

    Haselton et al. (2008), PEER 2007/03, equation 3.10 (full form):

        theta_p = 0.12 (1 + 0.55 a_sl) (0.16)^nu (0.02 + 40 rho_sh)^0.43
                  (0.54)^(0.01 f'c_MPa) (0.66)^(0.1 s_n) (2.27)^(10 rho)

    rho is the total longitudinal steel over b d (RHO_DEFINITION). Beams are the nu = 0 case, where the
    (0.16)^nu term is unity: a column regression extended to beams.
    """
    nu = _clamp(nu, NU_MIN, NU_MAX)
    rho_sh = _clamp(transverse_steel_ratio(member_type, design), RHO_SH_MIN, RHO_SH_MAX)
    rho = longitudinal_steel_ratio(member_type, design, depths)
    fc_mpa = _section_values(member_type, design)[2] * KSI_TO_MPA
    s_n = _stability_index(member_type, design)

    theta_p = (
        0.12
        * (1.0 + 0.55 * bond_slip_indicator(end, member_scope=member_scope))
        * (0.16 ** nu)
        * ((0.02 + 40.0 * rho_sh) ** 0.43)
        * (0.54 ** (0.01 * fc_mpa))
        * (0.66 ** (0.1 * s_n))
        * (2.27 ** (10.0 * rho))
    )
    return max(THETA_P_FLOOR, theta_p)


def haselton_theta_pc(member_type, nu, design=None):
    """Post-capping rotation capacity.

    Haselton et al. (2008), equation 3.16:

        theta_pc = 0.76 (0.031)^nu (0.02 + 40 rho_sh)^1.02,  capped at 0.10

    The cap is part of the published expression: the fit is unreliable for
    very well confined members and would otherwise predict implausibly
    ductile post-peak behaviour.
    """
    nu = _clamp(nu, NU_MIN, NU_MAX)
    rho_sh = _clamp(transverse_steel_ratio(member_type, design), RHO_SH_MIN, RHO_SH_MAX)
    theta_pc = 0.76 * (0.031 ** nu) * ((0.02 + 40.0 * rho_sh) ** 1.02)
    return max(THETA_PC_FLOOR, min(THETA_PC_CAP, theta_pc))


def deterioration_for_member(member_type, nu, design=None, depths=None):
    """Explicit OpenSees energy convention; PEER 2007/03 Eq. 3.20, for each spring axis.

        lambda = 170.7 (0.27)^nu (0.10)^(s/d)

    with s the hoop spacing and d the effective depth of the spring's bending axis (effective_depths:
    d of spring y across h, of spring z across b; EQ_3_20_VERSION). Eq. 3.20 is a column regression,
    extended here to zero-axial beams and to a beam's lateral axis. The member yield rotation is a declared
    nominal assumption, NOT the artificially small My/Ke of a stiff numerical zeroLength spring.
    Conversion: Lambda_OpenSees = lambda_Haselton * theta_y_member. The top-level lambda keys are spring y's
    (the strong axis); ``deterioration_by_axis`` holds both springs'.
    """
    mode = getattr(sp, "IMK_DETERIORATION_MODE", "direct")
    if mode == "direct":
        return {"deterioration_source": "direct_opensees"}
    if mode != "haselton_2008":
        raise ValueError(f"Unknown IMK_DETERIORATION_MODE: {mode!r}")
    values = _section_values(member_type, design)
    spacing = values[9]
    depths = depths if depths is not None else effective_depths(member_type, design)
    theta_y = sp.IMK_COLUMN_THETA_Y if member_type == "column" else sp.IMK_BEAM_THETA_Y
    if not all(math.isfinite(v) and v > 0 for v in (spacing, depths["y"]["d_in"], depths["z"]["d_in"], theta_y)):
        raise ValueError("Deterioration calibration requires positive spacing, effective depths and member yield rotation")
    nu_used = _clamp(nu, NU_MIN, NU_MAX)
    by_axis = {}
    for axis in ("y", "z"):
        ratio = spacing / depths[axis]["d_in"]
        lam = 170.7 * 0.27**nu_used * 0.10**ratio
        by_axis[axis] = {"lambda_haselton_dimensionless": lam, "lambda_opensees_rad": lam*theta_y,
                         # PEER 2007/03, Sec. 2.1.2.1 (p. 9): the fitted lambda applies
                         # to S and C. A and K were excluded from that calibration.
                         # Current IMKPeakOriented requires positive parameters: a large
                         # finite capacity approximates disabled A/K, without 0 or inf.
                         "lambda_opensees_by_mode_rad": {"S": lam*theta_y, "C": lam*theta_y, "A": 1.0e12, "K": 1.0e12},
                         "spacing_in": spacing, "effective_depth_in": depths[axis]["d_in"],
                         "spacing_effective_depth_ratio": ratio}
    strong = by_axis["y"]
    return {"deterioration_source": "haselton_2008_eq3_20_nominal_member_theta_y",
            "deterioration_equation_version": EQ_3_20_VERSION,
            "lambda_haselton_dimensionless": strong["lambda_haselton_dimensionless"],
            "energy_reference_member_theta_y_rad": theta_y,
            "lambda_opensees_rad": strong["lambda_opensees_rad"],
            "lambda_opensees_by_mode_rad": dict(strong["lambda_opensees_by_mode_rad"]),
            "deterioration_by_axis": by_axis,
            "deterioration_effective_depths": depths,
            "deterioration_mode_basis": "haselton_2008_section_2_1_2_1_strength_only",
            "deterioration_suppressed_modes": ["A", "K"],
            "deterioration_suppression_lambda_rad": 1.0e12,
            "deterioration_spacing_effective_depth_ratio": strong["spacing_effective_depth_ratio"],
            "deterioration_axial_ratio_used": nu_used,
            "deterioration_axial_ratio_clamped": nu_used != nu,
            "deterioration_beam_extrapolation": member_type != "column",
            "deterioration_unsupported_extensions": (
                [] if member_type == "column" else
                ["column regression applied to a beam", "one canonical (hogging) depth for both signs of vertical bending",
                 "lateral-axis (z) spring of a beam"])}


def backbone_for_member(member_type, axial_kip=0.0, pm_diagram=None, end=None, *, member_scope=None, design=None,
                        rows=None):
    """Full per-member IMK backbone, for one member END when ``end`` is given.

    Returns yield moment, plastic and post-capping rotations, ultimate
    rotation, and the axial ratio that produced them. The axial ratio is
    carried through so it can be exported as a node/element feature: it is
    the single number that explains why two otherwise identical columns
    behave differently. ``end`` (Model/Deformation_Ownership.member_end)
    decides whether the plastic rotation keeps Haselton's bar-slip share
    (member hinge owns slip) or excludes it (a registered face interface
    owns it); the ownership record travels with the backbone. ``rows`` are a beam's bar rows (the ones its
    strengths use), from which the effective depths of Eq. 3.10 and Eq. 3.20 are taken.
    """
    from Model import Deformation_Ownership as ownership
    use_calibration = getattr(sp, "IMK_USE_CALIBRATED_BACKBONE", True)
    nu = axial_load_ratio(axial_kip, member_type, design)
    depths = effective_depths(member_type, design, rows)
    indicator = bond_slip_indicator(end, member_scope=member_scope)
    slip = {"bond_slip_indicator": indicator,
            "slip_owner": (ownership.ownership(end, member_scope=member_scope)["bar_slip"] if end is not None
                           else "member_hinge"),
            "end_location": None if end is None else end["location"]}

    if not use_calibration:
        return {
            "axial_kip": axial_kip,
            "axial_ratio": nu,
            "theta_p": sp.IMK_THETA_P_POS,
            "theta_pc": sp.IMK_THETA_PC_POS,
            "theta_u": sp.IMK_THETA_U_POS,
            "source": "fixed",
            "theta_p_neg": sp.IMK_THETA_P_NEG,
            "theta_pc_neg": sp.IMK_THETA_PC_NEG,
            "theta_u_neg": sp.IMK_THETA_U_NEG,
            **slip,
            **deterioration_for_member(member_type, nu, design, depths),
        }

    theta_p = haselton_theta_p(member_type, nu, end, member_scope=member_scope, design=design, depths=depths)
    theta_pc = haselton_theta_pc(member_type, nu, design)
    return {
        **slip,
        "axial_kip": axial_kip,
        "axial_ratio": nu,
        "theta_p": theta_p,
        "theta_pc": theta_pc,
        "theta_u": theta_p + theta_pc + (sp.IMK_COLUMN_THETA_Y if member_type == "column" else sp.IMK_BEAM_THETA_Y),
        "source": "haselton_2008",
        "theta_p_longitudinal_steel": longitudinal_steel_record(member_type, design, depths),
        **deterioration_for_member(member_type, nu, design, depths),
    }
