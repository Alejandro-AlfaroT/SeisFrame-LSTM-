"""
RC_Design_Check.py
ACI 318-19 demand-capacity checks for the RC frame.

All section geometry, material strengths, and rebar layout are read directly
from Structure_Parameters so this module stays in sync with the model.

Checks
------
  Columns : P-M interaction (uniaxial + biaxial via resultant moment), shear
  Beams   : flexure (positive + negative), shear

Usage
-----
    from RC_Design_Check import get_element_tags, run_checks, print_summary

    col_tags, beam_x_tags, beam_y_tags = get_element_tags()
    results = run_checks(col_tags, beam_x_tags + beam_y_tags)
    print_summary(results)
"""

import math
import openseespy.opensees as ops
import Structure_Parameters as sp


# ─────────────────────────────────────────────────────────────────────────────
# Stirrup area helper. Values come from Structure_Parameters so legacy
# pushover/final checks stay aligned with the redesign config.
# ─────────────────────────────────────────────────────────────────────────────
def _stirrup_area(bar_size, legs):
    return legs * sp.rebar_area(bar_size)


# ─────────────────────────────────────────────────────────────────────────────
# Derived section geometry  (consistent with Sections.py bar placement)
#
# In make_rc_rect_section, bars are placed at:
#   top layer:    z =  h/2 - cover  (from section centroid)
#   bottom layer: z = -h/2 + cover
# The member-specific centroid offset includes clear cover, hoop diameter,
# and half the longitudinal-bar diameter in slab-aware designs; legacy COVER
# remains the old centroid offset. Effective depth is d = h - that offset.
# ─────────────────────────────────────────────────────────────────────────────

def _col_steel_layers(h=None):
    """
    Column steel layers for P-M diagram sweep.
    Returns list of (area_in2, dist_from_compression_face_in).
    Compression face assumed at z = +h/2 (top face of cross-section).

    h defaults to the current section depth. The design ladder passes it
    explicitly to price a candidate section without mutating the globals.
    """
    cover = sp.longitudinal_cover_in("column")
    h     = sp.H_COL if h is None else h
    Ab    = sp.COL_BAR_AREA
    layers = [
        (sp.COL_TOP_BARS * Ab, cover),       # compression-side bars
        (sp.COL_BOT_BARS * Ab, h - cover),   # tension-side bars
    ]
    if sp.COL_SIDE_BARS > 0:
        # COL_SIDE_BARS is the count PER SIDE FACE: Model/Sections.py lays a
        # straight layer of that many bars down each of the two side faces, so
        # the section holds twice this count. Counting them once left the P-M
        # surface missing a quarter of the longitudinal steel, which understated
        # every column capacity derived from it -- hinge yield moments included.
        if sp.SLAB_THICKNESS_IN is None:
            layers.insert(1, (2 * sp.COL_SIDE_BARS * Ab, h / 2.0))
        else:
            # Match actual straight-layer fiber coordinates, not a lump at
            # middepth. There is one bar on each side at every interior level.
            side_layers = [(2 * Ab, cover + (h - 2 * cover) * k
                            / (sp.COL_SIDE_BARS + 1))
                           for k in range(1, sp.COL_SIDE_BARS + 1)]
            layers[1:1] = side_layers
    return layers


def _col_d():
    return sp.H_COL - sp.longitudinal_cover_in("column")


def _beam_d():
    return sp.H_BEAM - sp.longitudinal_cover_in("beam")


# ─────────────────────────────────────────────────────────────────────────────
# ACI 318-19 helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _beta1(fc_ksi):
    """ACI 318-19 §22.2.2.4.3  (0.65 ≤ β₁ ≤ 0.85)."""
    return max(0.65, min(0.85, 0.85 - 0.05 * (fc_ksi - 4.0)))


def _phi_PM(eps_t):
    """
    ACI 318-19 Table 21.2.2 – tied column φ factor.
    Transitions from 0.65 (compression-controlled) to 0.90 (tension-controlled).
    """
    if eps_t <= 0.002:
        return 0.65
    elif eps_t >= 0.005:
        return 0.90
    return 0.65 + (eps_t - 0.002) / 0.003 * 0.25


PHI_FLEX  = 0.90
PHI_SHEAR = 0.75


# ─────────────────────────────────────────────────────────────────────────────
# Element tag enumeration  (mirrors elements.py creation order exactly)
# ─────────────────────────────────────────────────────────────────────────────

def get_element_tags():
    """
    Return (col_tags, beam_x_tags, beam_y_tags) in the same order
    that create_elements() assigns them.

    For the IMK formulation the returned tags refer to the elasticBeamColumn
    spine element (not the zeroLength hinge elements).
    """
    col_tags, beam_x_tags, beam_y_tags = [], [], []
    ele_tag = 1

    for k in range(sp.NUM_FLOOR):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                col_tags.append(ele_tag)
                ele_tag += 1

    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X):
                beam_x_tags.append(ele_tag)
                ele_tag += 1

    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                beam_y_tags.append(ele_tag)
                ele_tag += 1

    return col_tags, beam_x_tags, beam_y_tags


# ─────────────────────────────────────────────────────────────────────────────
# Column P-M interaction diagram
# ─────────────────────────────────────────────────────────────────────────────

def build_column_PM_diagram(n_pts=120):
    """
    Generate the φPn–φMn interaction diagram by sweeping the neutral-axis
    depth c from near-zero (tension-controlled) to 4h (near pure compression).

    All inputs drawn from Structure_Parameters (FC_COL_KSI, B_COL, H_COL,
    FY_KSI, ES_KSI, COVER, COL_*_BARS, COL_BAR_AREA).

    Returns
    -------
    list of (phi_Pn_kips, phi_Mn_kip_in)
        Compression positive for φPn.  φMn is always ≥ 0.
    """
    fc     = sp.FC_COL_KSI
    fy     = sp.FY_KSI
    Es     = sp.ES_KSI
    b      = sp.B_COL
    h      = sp.H_COL
    layers = _col_steel_layers()

    ecu = 0.003
    b1  = _beta1(fc)
    Ag  = b * h
    Ast = sum(A for A, _ in layers)
    hc  = h / 2.0

    # Pure compression cap (zero eccentricity, tied column)
    P0     = 0.85 * fc * (Ag - Ast) + fy * Ast   # ACI 22.4.2.2
    Pn_max = 0.80 * P0                            # ACI 22.4.2.1
    diagram = [(0.65 * Pn_max, 0.0)]

    dt = max(d for _, d in layers)   # distance to extreme tension steel

    for i in range(n_pts):
        c  = 0.001 + (4.0 * h - 0.001) * i / (n_pts - 1)
        a  = min(b1 * c, h)
        Cc = 0.85 * fc * b * a
        Mc = Cc * (hc - a / 2.0)

        Pn = Cc
        Mn = Mc

        for As_i, d_i in layers:
            eps_i  = ecu * (c - d_i) / c
            fs_i   = max(-fy, min(fy, Es * eps_i))
            fs_net = fs_i - (0.85 * fc if d_i <= a else 0.0)
            F_i    = As_i * fs_net
            Pn    += F_i
            Mn    += F_i * (hc - d_i)

        eps_t = ecu * (dt - c) / c
        phi   = _phi_PM(eps_t)
        diagram.append((phi * Pn, phi * abs(Mn)))

    return diagram


# ─────────────────────────────────────────────────────────────────────────────
# Individual check functions
# ─────────────────────────────────────────────────────────────────────────────

def check_column_PM(Pu, Muz, Muy, diagram):
    """
    Check column P-M demand against the interaction diagram.
    Biaxial bending handled via resultant moment: Mu = √(Muz² + Muy²).

    Returns  (dcr, ok, Mu, phi_Mn_cap)
    """
    Mu = math.sqrt(Muz**2 + Muy**2)

    if Mu < 1e-4:
        phi_Pmax = diagram[0][0]
        dcr = Pu / phi_Pmax if phi_Pmax > 1e-6 else 999.0
        return dcr, dcr <= 1.0, Mu, phi_Pmax

    sorted_pts = sorted(diagram, key=lambda p: -p[0])
    phi_Mn_cap = None

    for i in range(len(sorted_pts) - 1):
        P1, M1 = sorted_pts[i]
        P2, M2 = sorted_pts[i + 1]
        if P2 <= Pu <= P1:
            dP = P1 - P2
            t  = (Pu - P2) / dP if dP > 1e-9 else 0.0
            M_interp = M2 + t * (M1 - M2)
            if phi_Mn_cap is None or M_interp > phi_Mn_cap:
                phi_Mn_cap = M_interp

    if phi_Mn_cap is None or phi_Mn_cap < 1e-6:
        return 999.0, False, Mu, 0.0

    dcr = Mu / phi_Mn_cap
    return dcr, dcr <= 1.0, Mu, phi_Mn_cap


def check_beam_flexure(Mu_pos, Mu_neg):
    """
    ACI 318-19 §22.3 – singly reinforced rectangular beam flexure.
    Uses BEAM_TOP_BARS, BEAM_BOT_BARS, FC_BEAM_KSI from Structure_Parameters.

    Returns  (dcr_pos, dcr_neg, phi_Mn_pos, phi_Mn_neg, ok)
    """
    fc = sp.FC_BEAM_KSI
    fy = sp.FY_KSI
    b  = sp.B_BEAM
    d  = _beam_d()

    def _phi_Mn(As):
        a = As * fy / (0.85 * fc * b)
        return PHI_FLEX * As * fy * (d - a / 2.0)

    phi_Mn_pos = _phi_Mn(sp.BEAM_BOT_BARS * sp.BEAM_BAR_AREA)
    phi_Mn_neg = _phi_Mn(sp.BEAM_TOP_BARS * sp.BEAM_BAR_AREA)

    dcr_pos = Mu_pos / phi_Mn_pos if phi_Mn_pos > 1e-6 else 0.0
    dcr_neg = Mu_neg / phi_Mn_neg if phi_Mn_neg > 1e-6 else 0.0

    return dcr_pos, dcr_neg, phi_Mn_pos, phi_Mn_neg, max(dcr_pos, dcr_neg) <= 1.0


def check_shear(Vu, Nu, bw, d, fc, Av, s, lambda_=1.0, h=None):
    """
    ACI 318-19 §22.5 – one-way shear strength of a member with Av ≥ Av,min.
    Vc  = [2λ√f'c + Nu/(6·Ag)] · bw · d   Table 22.5.5.1 (a), Nu/(6·Ag) in psi, ≤ 0.05 f'c (22.5.5.1.1)
          Vc ≤ 5λ√f'c · bw · d            22.5.5.1.2
    Vs  = Av · fy · d / s ≤ 8√f'c · bw · d  22.5.8.5.3 / 22.5.1.2
    φVn = 0.75(Vc + Vs)

    Nu : axial force, kips, positive = compression (tension reduces Vc, floored at 0); 0 for beams.
    h  : section depth for Ag = bw·h; without it the legacy bw·d/0.9 estimate is used.
    Returns  (phi_Vn, dcr, ok)

    Until 2026-09-26 the axial term was added as Nu/(6·Ag) in ksi straight to a force in kip.
    """
    fc_psi = fc * 1000.0
    Ag     = bw * h if h else bw * (d / 0.9)
    nu_psi = min(Nu * 1000.0 / (6.0 * Ag), 0.05 * fc_psi)
    Vc     = max(0.0, (2.0 * lambda_ * math.sqrt(fc_psi) + nu_psi) * bw * d / 1000.0)
    Vc     = min(Vc, 5.0 * lambda_ * math.sqrt(fc_psi) * bw * d / 1000.0)

    Vs     = min(Av * sp.FY_KSI * d / s, 8.0 * math.sqrt(fc_psi) * bw * d / 1000.0)
    phi_Vn = PHI_SHEAR * (Vc + Vs)

    dcr = Vu / phi_Vn if phi_Vn > 1e-6 else 999.0
    return phi_Vn, dcr, dcr <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Force extraction
# ─────────────────────────────────────────────────────────────────────────────

def _element_local_force(ele_tag):
    """
    Return the 12-component local end-force vector for a frame element.

    ops.eleForce() is not consistently local for every element formulation, so
    prefer the explicit localForce response and only fall back if OpenSees does
    not provide it.
    """
    try:
        forces = ops.eleResponse(ele_tag, "localForce")
    except Exception:
        forces = None

    if not forces:
        forces = ops.eleForce(ele_tag)

    if len(forces) != 12:
        raise ValueError(
            f"Expected 12 local force components for element {ele_tag}, "
            f"got {len(forces)}."
        )

    return list(forces)


def _extract_forces(ele_tag, local_force=None):
    """
    Read local end forces and return critical demands.
    P is compression-positive for the localForce response used here.

    ``local_force`` is an already-captured 12-component localForce vector
    (Design_Driver._capture_element_actions); when given, the live domain
    is not consulted, so the checks can be rerun on solved actions.
    """
    f    = _element_local_force(ele_tag) if local_force is None else list(local_force)
    if len(f) != 12:
        raise ValueError(f"Expected 12 local force components for element {ele_tag}, got {len(f)}.")
    P    = f[0]
    Vy   = max(abs(f[1]),  abs(f[7]))
    Vz   = max(abs(f[2]),  abs(f[8]))
    My   = max(abs(f[4]),  abs(f[10]))
    Mz   = max(abs(f[5]),  abs(f[11]))
    return P, Vy, Vz, My, Mz, f[4], f[10], f[5], f[11]


def beam_flexure_demands(My_i, My_j):
    """
    Map signed local end moments to (Mu_pos, Mu_neg) = (sagging, hogging).

    Both floor-beam transforms use vecxz = (0, 0, 1), so local z is global Z
    and vertical bending is the local-y moment. OpenSees localForce reports
    both end moments in the same rotational sense, which makes the physical
    sign flip between the ends: at end i a negative My is hogging (tension on
    top); at end j a positive My is hogging. Verified against a fixed-fixed
    span under uniform load, where both supports are hogging and localForce
    gives My_i = -wL^2/12, My_j = +wL^2/12.

    Reading the numerical sign directly (the previous behaviour) turned every
    gravity hogging pair into an equal sagging demand, so bottom steel was
    always designed to match top steel.
    """
    sagging_i, hogging_i = max(0.0, My_i), max(0.0, -My_i)
    sagging_j, hogging_j = max(0.0, -My_j), max(0.0, My_j)
    return max(sagging_i, sagging_j), max(hogging_i, hogging_j)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_checks(col_tags, beam_tags, col_diagram=None):
    """
    Run ACI 318-19 checks on all supplied element tags.

    Parameters
    ----------
    col_tags  : list of int – column element tags
    beam_tags : list of int – beam element tags (X and Y combined)

    Returns
    -------
    dict  {ele_tag: result_dict}
    """
    col_diagram = col_diagram or build_column_PM_diagram()
    results     = {}

    for tag in col_tags:
        P, Vy, Vz, My, Mz, _, _, _, _ = _extract_forces(tag)
        dcr_PM, ok_PM, Mu, phi_Mn = check_column_PM(P, Mz, My, col_diagram)
        Vu = math.sqrt(Vy**2 + Vz**2)
        phi_Vn, dcr_V, ok_V = check_shear(
            Vu,
            P,
            bw=sp.B_COL,
            d=_col_d(),
            fc=sp.FC_COL_KSI,
            Av=_stirrup_area(sp.COL_STIRRUP_BAR_SIZE, sp.COL_STIRRUP_LEGS),
            s=sp.COL_STIRRUP_SPACING,
            h=sp.H_COL,
        )
        results[tag] = {
            'type': 'column', 'Pu': P, 'Mu': Mu, 'phi_Mn_cap': phi_Mn,
            'Vu': Vu, 'phi_Vn': phi_Vn, 'dcr_PM': dcr_PM, 'dcr_V': dcr_V,
            'ok_PM': ok_PM, 'ok_V': ok_V, 'ok': ok_PM and ok_V,
            'stirrup_bar_size': sp.COL_STIRRUP_BAR_SIZE,
            'stirrup_legs': sp.COL_STIRRUP_LEGS,
            'stirrup_spacing_in': sp.COL_STIRRUP_SPACING,
            'stirrup_area_in2': _stirrup_area(sp.COL_STIRRUP_BAR_SIZE, sp.COL_STIRRUP_LEGS),
        }

    from Design.SMRF_Beam_Actions import current_beam_bending
    beam_tags = list(beam_tags)
    span_actions = current_beam_bending(beam_tags)
    for tag in beam_tags:
        P, Vy, Vz, My, Mz, My_i, My_j, _Mz_i, _Mz_j = _extract_forces(tag)
        # Vertical-plane flexure only: Mz is bending about the beam's vertical
        # axis and is not resisted by the top/bottom bars.
        envelope = span_actions[tag]["full_span"]
        Mu_pos, Mu_neg = envelope["mu_positive_kip_in"], envelope["mu_negative_kip_in"]
        dcr_pos, dcr_neg, phi_Mn_pos, phi_Mn_neg, ok_F = check_beam_flexure(
            Mu_pos, Mu_neg
        )
        Vu = math.sqrt(Vy**2 + Vz**2)
        phi_Vn, dcr_V, ok_V = check_shear(
            Vu,
            0.0,
            bw=sp.B_BEAM,
            d=_beam_d(),
            fc=sp.FC_BEAM_KSI,
            Av=_stirrup_area(sp.BEAM_STIRRUP_BAR_SIZE, sp.BEAM_STIRRUP_LEGS),
            s=sp.BEAM_STIRRUP_SPACING,
        )
        results[tag] = {
            'type': 'beam', 'Mu_pos': Mu_pos, 'Mu_neg': Mu_neg,
            'span_bending': span_actions[tag],
            'phi_Mn_pos': phi_Mn_pos, 'phi_Mn_neg': phi_Mn_neg,
            'Mu_y': My, 'Mu_z': Mz, 'Vu': Vu, 'phi_Vn': phi_Vn, 'dcr_pos': dcr_pos,
            'dcr_neg': dcr_neg, 'dcr_V': dcr_V,
            'ok_F': ok_F, 'ok_V': ok_V, 'ok': ok_F and ok_V,
            'stirrup_bar_size': sp.BEAM_STIRRUP_BAR_SIZE,
            'stirrup_legs': sp.BEAM_STIRRUP_LEGS,
            'stirrup_spacing_in': sp.BEAM_STIRRUP_SPACING,
            'stirrup_area_in2': _stirrup_area(sp.BEAM_STIRRUP_BAR_SIZE, sp.BEAM_STIRRUP_LEGS),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Results printer  (ASCII-safe for Windows cp1252 / cp850 terminals)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, verbose=False):
    col_fail, beam_fail = [], []
    col_dcr_PM_max = col_dcr_V_max = 0.0
    beam_dcr_F_max = beam_dcr_V_max = 0.0

    print("\n" + "=" * 70)
    print("  ACI 318-19 Design Check")
    print(
        f"  Column : {sp.B_COL}\" x {sp.H_COL}\"  fc={sp.FC_COL_KSI} ksi  "
        f"({sp.COL_TOP_BARS}T+{sp.COL_BOT_BARS}B+{sp.COL_SIDE_BARS}S) "
        f"#{sp.COL_BAR_SIZE}"
    )
    print(
        f"  Beam   : {sp.B_BEAM}\" x {sp.H_BEAM}\"  fc={sp.FC_BEAM_KSI} ksi  "
        f"{sp.BEAM_TOP_BARS}T+{sp.BEAM_BOT_BARS}B #{sp.BEAM_BAR_SIZE}"
    )
    print("=" * 70)

    print("\n  COLUMNS")
    print(
        f"  {'Tag':>5}  {'Pu(k)':>7}  {'Mu(k-in)':>9}  "
        f"{'φMn(k-in)':>11}  {'DCR P-M':>7}  {'Vu(k)':>6}  "
        f"{'DCR V':>6}  Status"
    )
    print("  " + "-" * 72)

    for tag, row in results.items():
        if row["type"] != "column":
            continue

        col_dcr_PM_max = max(col_dcr_PM_max, row["dcr_PM"])
        col_dcr_V_max = max(col_dcr_V_max, row["dcr_V"])
        if not row["ok"]:
            col_fail.append(tag)
        print(
            f"  {tag:>5}  {row['Pu']:>7.1f}  {row['Mu']:>9.1f}  "
            f"{row['phi_Mn_cap']:>11.1f}  {row['dcr_PM']:>7.3f}  "
            f"{row['Vu']:>6.2f}  {row['dcr_V']:>6.3f}  "
            f"{'OK' if row['ok'] else 'FAIL'}"
        )

    print(
        f"\n  Column P-M  max DCR : {col_dcr_PM_max:.3f}  "
        f"({'OK' if col_dcr_PM_max <= 1.0 else 'FAIL'})"
    )
    print(
        f"  Column shear max DCR: {col_dcr_V_max:.3f}  "
        f"({'OK' if col_dcr_V_max <= 1.0 else 'FAIL'})"
    )

    print("\n  BEAMS")
    print(
        f"  {'Tag':>5}  {'Mu(k-in)':>9}  {'φMn(k-in)':>10}  "
        f"{'DCR F':>6}  {'Vu(k)':>6}  {'DCR V':>6}  Status"
    )
    print("  " + "-" * 62)

    for tag, row in results.items():
        if row["type"] != "beam":
            continue

        beam_dcr_F_max = max(beam_dcr_F_max, row["dcr_pos"], row["dcr_neg"])
        beam_dcr_V_max = max(beam_dcr_V_max, row["dcr_V"])
        if not row["ok"]:
            beam_fail.append(tag)
        # Show the governing (highest-DCR) direction
        if row["dcr_pos"] >= row["dcr_neg"]:
            Mu_gov, phi_Mn_gov, dcr_gov = row["Mu_pos"], row["phi_Mn_pos"], row["dcr_pos"]
        else:
            Mu_gov, phi_Mn_gov, dcr_gov = row["Mu_neg"], row["phi_Mn_neg"], row["dcr_neg"]
        print(
            f"  {tag:>5}  {Mu_gov:>9.1f}  {phi_Mn_gov:>10.1f}  "
            f"{dcr_gov:>6.3f}  {row['Vu']:>6.2f}  "
            f"{row['dcr_V']:>6.3f}  {'OK' if row['ok'] else 'FAIL'}"
        )

    print(
        f"\n  Beam flexure max DCR: {beam_dcr_F_max:.3f}  "
        f"({'OK' if beam_dcr_F_max <= 1.0 else 'FAIL'})"
    )
    print(
        f"  Beam shear   max DCR: {beam_dcr_V_max:.3f}  "
        f"({'OK' if beam_dcr_V_max <= 1.0 else 'FAIL'})"
    )

    n_fail = len(col_fail) + len(beam_fail)
    print("\n" + "=" * 70)
    if n_fail == 0:
        print("  OVERALL: ALL ELEMENTS PASS")
    else:
        print(f"  OVERALL: {n_fail} ELEMENT(S) FAIL")
        if col_fail:
            print(f"  Failed columns : {col_fail}")
        if beam_fail:
            print(f"  Failed beams   : {beam_fail}")
    print("=" * 70 + "\n")

    return col_fail, beam_fail
