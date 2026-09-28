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
from RC_Design_Check import _beta1, _col_steel_layers


KSI_TO_MPA = 6.894757

# Haselton et al. (2008) fit bounds. Outside these the expressions are
# extrapolation, so inputs are clamped and the clamp is reported.
NU_MIN, NU_MAX = 0.0, 0.70
RHO_SH_MIN, RHO_SH_MAX = 0.002, 0.020

# a_sl = 1 where bar slip from the joint is possible, which is the normal
# condition for a cast-in-place frame and the value Haselton's regression
# carries for the member. Since 2026-09-27 the frame can carry slip in the
# joint spring instead (JOINT_DEFORMATION_SCOPE = "joint_shear_and_slip"),
# in which case the members drop the term so slip is counted once.
BOND_SLIP_INDICATOR = 1.0


def bond_slip_indicator():
    """Haselton's a_sl for the member hinges under the current joint model and deformation scope."""
    joint_model = getattr(sp, "JOINT_MODEL", "rigid_centerline")
    scope = getattr(sp, "JOINT_DEFORMATION_SCOPE", "joint_shear_only")
    if joint_model == "imk_pinching_scissors" and scope == "joint_shear_and_slip":
        return 0.0
    return BOND_SLIP_INDICATOR

THETA_P_FLOOR = 0.005
THETA_PC_FLOOR = 0.010
THETA_PC_CAP = 0.10


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
    first elevated floor.
    """
    floors_above = max(0, sp.NUM_FLOOR - story_index + 1)
    floor_load = sp.node_gravity_load_kip(grid_i, grid_j) * floors_above
    if sp.SLAB_THICKNESS_IN is not None:
        # New slab-aware idealization includes tributary beam drops as well as
        # columns, matching the gravity-weight inventory. Still a tributary
        # estimate, NOT a solved factored joint axial-force envelope.
        self_weight = sp.node_structural_self_weight_kip(grid_i, grid_j) * floors_above
    else:
        self_weight = sp.col_self_weight_kip_per_in() * sp.STORY_H * floors_above
    return floor_load + self_weight


def column_grid_position(node_tag):
    """Recover (story_index, grid_i, grid_j) from a column's lower node."""
    x, y, z = ops.nodeCoord(node_tag)
    grid_i = int(round(x / sp.BAY_X)) if sp.BAY_X > 0 else 0
    grid_j = int(round(y / sp.BAY_Y)) if sp.BAY_Y > 0 else 0
    floor_below = int(round(z / sp.STORY_H)) if sp.STORY_H > 0 else 0
    return floor_below + 1, grid_i, grid_j


def axial_load_ratio(axial_kip, member_type):
    """Normalized axial load nu = P / (Ag f'c)."""
    if member_type == "column":
        gross_area = sp.B_COL * sp.H_COL
        strength = sp.FC_COL_KSI
    else:
        gross_area = sp.B_BEAM * sp.H_BEAM
        strength = sp.FC_BEAM_KSI
    if gross_area <= 0.0 or strength <= 0.0:
        return 0.0
    return max(0.0, axial_kip / (gross_area * strength))


# ---------------------------------------------------------------------------
# Haselton et al. (2008) rotation capacities
# ---------------------------------------------------------------------------

def transverse_steel_ratio(member_type):
    """rho_sh = Ash / (b * s) for the member's confinement."""
    if member_type == "column":
        legs, bar_size, spacing, width = (
            sp.COL_STIRRUP_LEGS, sp.COL_STIRRUP_BAR_SIZE,
            sp.COL_STIRRUP_SPACING, sp.B_COL,
        )
    else:
        legs, bar_size, spacing, width = (
            sp.BEAM_STIRRUP_LEGS, sp.BEAM_STIRRUP_BAR_SIZE,
            sp.BEAM_STIRRUP_SPACING, sp.B_BEAM,
        )
    if spacing <= 0.0 or width <= 0.0:
        return RHO_SH_MIN
    return legs * sp.rebar_area(bar_size) / (width * spacing)


def longitudinal_steel_ratio(member_type):
    """rho = As / (b * d) using the tension-side longitudinal steel."""
    if member_type == "column":
        area = max(sp.COL_TOP_BARS, sp.COL_BOT_BARS) * sp.COL_BAR_AREA
        width, depth = sp.B_COL, sp.H_COL - sp.longitudinal_cover_in("column")
    else:
        area = max(sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS) * sp.BEAM_BAR_AREA
        width, depth = sp.B_BEAM, sp.H_BEAM - sp.longitudinal_cover_in("beam")
    if width <= 0.0 or depth <= 0.0:
        return 0.0
    return area / (width * depth)


def _stability_index(member_type):
    """s_n = (s / d_b) * sqrt(fy_MPa / 100), the rebar buckling index."""
    if member_type == "column":
        spacing = sp.COL_STIRRUP_SPACING
        bar_diameter = sp.rebar_diameter(sp.COL_BAR_SIZE)
    else:
        spacing = sp.BEAM_STIRRUP_SPACING
        bar_diameter = sp.rebar_diameter(sp.BEAM_BAR_SIZE)
    if bar_diameter <= 0.0:
        return 1.0
    return (spacing / bar_diameter) * math.sqrt(sp.FY_KSI * KSI_TO_MPA / 100.0)


def haselton_theta_p(member_type, nu):
    """Plastic rotation capacity to the capping point.

    Haselton et al. (2008), PEER 2007/03, equation 3.10 (full form):

        theta_p = 0.12 (1 + 0.55 a_sl) (0.16)^nu (0.02 + 40 rho_sh)^0.43
                  (0.54)^(0.01 f'c_MPa) (0.66)^(0.1 s_n) (2.27)^(10 rho)

    Beams are the nu = 0 case, where the (0.16)^nu term is unity.
    """
    nu = _clamp(nu, NU_MIN, NU_MAX)
    rho_sh = _clamp(transverse_steel_ratio(member_type), RHO_SH_MIN, RHO_SH_MAX)
    rho = longitudinal_steel_ratio(member_type)
    fc_mpa = (sp.FC_COL_KSI if member_type == "column" else sp.FC_BEAM_KSI) * KSI_TO_MPA
    s_n = _stability_index(member_type)

    theta_p = (
        0.12
        * (1.0 + 0.55 * bond_slip_indicator())
        * (0.16 ** nu)
        * ((0.02 + 40.0 * rho_sh) ** 0.43)
        * (0.54 ** (0.01 * fc_mpa))
        * (0.66 ** (0.1 * s_n))
        * (2.27 ** (10.0 * rho))
    )
    return max(THETA_P_FLOOR, theta_p)


def haselton_theta_pc(member_type, nu):
    """Post-capping rotation capacity.

    Haselton et al. (2008), equation 3.16:

        theta_pc = 0.76 (0.031)^nu (0.02 + 40 rho_sh)^1.02,  capped at 0.10

    The cap is part of the published expression: the fit is unreliable for
    very well confined members and would otherwise predict implausibly
    ductile post-peak behaviour.
    """
    nu = _clamp(nu, NU_MIN, NU_MAX)
    rho_sh = _clamp(transverse_steel_ratio(member_type), RHO_SH_MIN, RHO_SH_MAX)
    theta_pc = 0.76 * (0.031 ** nu) * ((0.02 + 40.0 * rho_sh) ** 1.02)
    return max(THETA_PC_FLOOR, min(THETA_PC_CAP, theta_pc))


def deterioration_for_member(member_type, nu):
    """Explicit OpenSees energy convention; PEER 2007/03 Eq. 3.20.

    Eq. 3.20 is a column regression, extended here to zero-axial beams.
    The member yield rotation is a declared nominal assumption, NOT the
    artificially small My/Ke of a stiff numerical zeroLength spring.
    Conversion: Lambda_OpenSees = lambda_Haselton * theta_y_member.
    """
    mode = getattr(sp, "IMK_DETERIORATION_MODE", "direct")
    if mode == "direct":
        return {"deterioration_source": "direct_opensees"}
    if mode != "haselton_2008":
        raise ValueError(f"Unknown IMK_DETERIORATION_MODE: {mode!r}")
    spacing = sp.COL_STIRRUP_SPACING if member_type == "column" else sp.BEAM_STIRRUP_SPACING
    depth = sp.H_COL if member_type == "column" else sp.H_BEAM
    theta_y = sp.IMK_COLUMN_THETA_Y if member_type == "column" else sp.IMK_BEAM_THETA_Y
    if not all(math.isfinite(v) and v > 0 for v in (spacing, depth, theta_y)):
        raise ValueError("Deterioration calibration requires positive spacing, depth and member yield rotation")
    nu_used = _clamp(nu, NU_MIN, NU_MAX)
    lam = 170.7 * 0.27**nu_used * 0.10**(spacing/depth)
    return {"deterioration_source": "haselton_2008_eq3_20_nominal_member_theta_y",
            "lambda_haselton_dimensionless": lam,
            "energy_reference_member_theta_y_rad": theta_y,
            "lambda_opensees_rad": lam*theta_y,
            # PEER 2007/03, Sec. 2.1.2.1 (p. 9): the fitted lambda applies
            # to S and C. A and K were excluded from that calibration.
            # Current IMKPeakOriented requires positive parameters: a large
            # finite capacity approximates disabled A/K, without 0 or inf.
            "lambda_opensees_by_mode_rad": {
                "S": lam*theta_y, "C": lam*theta_y, "A": 1.0e12, "K": 1.0e12,
            },
            "deterioration_mode_basis": "haselton_2008_section_2_1_2_1_strength_only",
            "deterioration_suppressed_modes": ["A", "K"],
            "deterioration_suppression_lambda_rad": 1.0e12,
            "deterioration_spacing_depth_ratio": spacing/depth,
            "deterioration_axial_ratio_used": nu_used,
            "deterioration_axial_ratio_clamped": nu_used != nu,
            "deterioration_beam_extrapolation": member_type != "column"}


def backbone_for_member(member_type, axial_kip=0.0, pm_diagram=None):
    """Full per-member IMK backbone.

    Returns yield moment, plastic and post-capping rotations, ultimate
    rotation, and the axial ratio that produced them. The axial ratio is
    carried through so it can be exported as a node/element feature: it is
    the single number that explains why two otherwise identical columns
    behave differently.
    """
    use_calibration = getattr(sp, "IMK_USE_CALIBRATED_BACKBONE", True)
    nu = axial_load_ratio(axial_kip, member_type)

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
            **deterioration_for_member(member_type, nu),
        }

    theta_p = haselton_theta_p(member_type, nu)
    theta_pc = haselton_theta_pc(member_type, nu)
    return {
        "axial_kip": axial_kip,
        "axial_ratio": nu,
        "theta_p": theta_p,
        "theta_pc": theta_pc,
        "theta_u": theta_p + theta_pc + (sp.IMK_COLUMN_THETA_Y if member_type == "column" else sp.IMK_BEAM_THETA_Y),
        "source": "haselton_2008",
        **deterioration_for_member(member_type, nu),
    }
