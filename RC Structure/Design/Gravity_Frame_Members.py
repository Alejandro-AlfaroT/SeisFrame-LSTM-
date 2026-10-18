"""Members not designated as part of the seismic-force-resisting system: ACI 318-19 18.14 (2026-10-16).

Under the perimeter-frame basis (user decision 2026-10-16) the special moment frames are the four perimeter
frames and every interior beam and column is a gravity member: designed for the gravity combinations of 5.3
and detailed for compatibility with the design displacement delta_u (18.14.2.1, 18.14.3.1). The branch of
18.14.3 a member falls in depends on the moments and shears induced in it at delta_u:

* 18.14.3.2, induced effects within the design strengths (Mu <= phi Mn and Vu <= phi Vn):
  (a) beams: 18.6.3.1 longitudinal limits; transverse reinforcement per 25.7 over the whole length at not
      more than d/2; hoops per 18.7.5.2 at not more than the lesser of 6 db and 6 in where Pu > Ag f'c / 10;
  (b) columns: 18.7.4.1 (rho 1 to 6 percent) and 18.7.6 (design shear); hoops per 25.7.4 over the full length
      at not more than the lesser of 6 db and 6 in; the 18.7.5.2(a) to (e) configuration over l0 (18.7.5.1);
  (c) columns with Pu > 0.35 Po also provide half the greater of Table 18.7.5.4 (a) and (b) over l0;
  (d) joints per Chapter 15.
* 18.14.3.3, induced effects above phi Mn or phi Vn, or not calculated:
  (a) materials and splices per 18.2.5 to 18.2.8; (b) beams per 18.14.3.2(a) and 18.6.5 (capacity shear);
  (c) columns per 18.7.4, 18.7.5 and 18.7.6, 18.7.4.3 excepted; (d) joints per the intermediate-frame joint
      shear rule (318-25 18.4.4.1).

The text was read from the ACI 318-25 book on 2026-10-16; the record's declared edition stays 318-19, whose
18.14.3 says the same in substance. 18.14.1.1 limits the section to SDC D, E and F; the research population
applies it to its SDC C frames as well, as a declared design-basis choice, so one rule covers the dataset.

This module is the rule layer only: pure functions of a member's induced actions, strengths, section and
detailing. Where the induced actions come from (the imposed design displacement on the continuous frame) and
how the detailing is selected belong to the design route. Units: kip, inch, ksi.
"""
from __future__ import annotations

import math

from Design.SMRF_Common import make_check, not_evaluated

BRANCH_WITHIN_STRENGTH = "18.14.3.2"
BRANCH_YIELDING = "18.14.3.3"
BRANCHES = (BRANCH_WITHIN_STRENGTH, BRANCH_YIELDING)
HOOP_SPACING_ABSOLUTE_MAX_IN = 6.0            # 18.14.3.2(a) and (b): the lesser of 6 db and 6 in
HOOP_SPACING_BAR_DIAMETERS = 6.0
BEAM_AXIAL_HOOP_FRACTION = 0.10               # 18.14.3.2(a): Pu > Ag f'c / 10 asks for hoops
HIGH_AXIAL_FRACTION_OF_PO = 0.35              # 18.14.3.2(c)
HIGH_AXIAL_ASH_FRACTION = 0.5                 # half of Table 18.7.5.4 (a) and (b)
JOINT_RULE = {BRANCH_WITHIN_STRENGTH: "ACI 318-19 Chapter 15 (15.3 ties, 15.4 joint shear)",
              BRANCH_YIELDING: "intermediate-frame joint shear (318-25 18.4.4.1; 318-19 18.4.4.1)"}


def _number(value, name, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}.")
    return float(value)


def nominal_axial_strength_kip(ag_in2, ast_in2, fc_ksi, fy_ksi):
    """Po of 22.4.2.2: 0.85 f'c (Ag - Ast) + fy Ast."""
    ag, ast = _number(ag_in2, "ag_in2", positive=True), _number(ast_in2, "ast_in2", minimum=0.0)
    if ast >= ag:
        raise ValueError("The longitudinal steel area must be less than the gross area.")
    return 0.85 * _number(fc_ksi, "fc_ksi", positive=True) * (ag - ast) + _number(fy_ksi, "fy_ksi", positive=True) * ast


def compatibility_branch(induced_mu_kip_in, induced_vu_kip, phi_mn_kip_in, phi_vn_kip, calculated=True):
    """Which branch of 18.14.3 a member falls in, with the reason.

    ``induced_*`` are the moment and shear induced by the gravity combination acting with the design
    displacement (18.14.2.1); ``phi_*`` the design strengths. ``calculated=False`` says the effects of
    delta_u were not computed, which 18.14.3.1 sends to 18.14.3.3 outright.
    """
    if not calculated:
        return branch_from_ratios(None, None, calculated=False)
    mu = abs(_number(induced_mu_kip_in, "induced_mu_kip_in"))
    vu = abs(_number(induced_vu_kip, "induced_vu_kip"))
    phi_mn = _number(phi_mn_kip_in, "phi_mn_kip_in", positive=True)
    phi_vn = _number(phi_vn_kip, "phi_vn_kip", positive=True)
    return branch_from_ratios(mu / phi_mn, vu / phi_vn)


def branch_from_ratios(moment_ratio, shear_ratio, calculated=True):
    """The 18.14.3 branch from the induced-to-design-strength ratios of a member or a group.

    A column's moment ratio is its P-M interaction ratio at the concurrent induced axial load (the induced
    moment against phi Mn at that load); a beam's is the larger of its sagging and hogging ratios; the shear
    ratio is Vu / phi Vn. The ratios are what Design.Group_Checks reports as DCRs for the compatibility
    cases, so the branch is decided by the same strength arithmetic as the strength checks.
    """
    if not calculated:
        return {"branch": BRANCH_YIELDING, "reason": "effects of the design displacement not calculated (18.14.3.1)",
                "moment_ratio": None, "shear_ratio": None}
    moment_ratio = _number(moment_ratio, "moment_ratio", minimum=0)
    shear_ratio = _number(shear_ratio, "shear_ratio", minimum=0)
    within = moment_ratio <= 1.0 + 1e-12 and shear_ratio <= 1.0 + 1e-12
    return {"branch": BRANCH_WITHIN_STRENGTH if within else BRANCH_YIELDING,
            "reason": ("induced moment and shear within phi Mn and phi Vn" if within else
                       "induced " + " and ".join(name for name, ratio in (("moment", moment_ratio), ("shear", shear_ratio))
                                                 if ratio > 1.0 + 1e-12) + " above the design strength"),
            "moment_ratio": moment_ratio, "shear_ratio": shear_ratio}


def induced_branch(induced, strengths=None):
    """The branch of an ``induced`` record: ratio form {"moment_ratio", "shear_ratio"} or action form
    {"mu_kip_in", "vu_kip"} against ``strengths`` {"phi_mn_kip_in", "phi_vn_kip"}; ``calculated`` False in either
    form sends the member to 18.14.3.3 (18.14.3.1)."""
    calculated = induced.get("calculated", True)
    if "moment_ratio" in induced or "shear_ratio" in induced:
        return branch_from_ratios(induced.get("moment_ratio"), induced.get("shear_ratio"), calculated=calculated)
    strengths = strengths or {}
    return compatibility_branch(induced["mu_kip_in"], induced["vu_kip"], strengths["phi_mn_kip_in"], strengths["phi_vn_kip"],
                                calculated=calculated)


def hoop_spacing_limit_in(smallest_bar_db_in):
    """The lesser of 6 db of the smallest enclosed longitudinal bar and 6 in (18.14.3.2(a), (b))."""
    return min(HOOP_SPACING_BAR_DIAMETERS * _number(smallest_bar_db_in, "smallest_bar_db_in", positive=True),
               HOOP_SPACING_ABSOLUTE_MAX_IN)


def beam_requirements(branch, d_in, smallest_bar_db_in, pu_kip=0.0, ag_in2=None, fc_ksi=None):
    """What 18.14.3 asks of a gravity beam in ``branch``: transverse kind and spacing, the provisions that apply."""
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}.")
    d = _number(d_in, "d_in", positive=True)
    pu = abs(_number(pu_kip, "pu_kip"))
    axial_threshold = None
    if ag_in2 is not None and fc_ksi is not None:
        axial_threshold = BEAM_AXIAL_HOOP_FRACTION * _number(ag_in2, "ag_in2", positive=True) * _number(fc_ksi, "fc_ksi", positive=True)
    axial_hoops = axial_threshold is not None and pu > axial_threshold
    if axial_hoops:
        transverse = {"kind": "hoops per 18.7.5.2", "spacing_max_in": hoop_spacing_limit_in(smallest_bar_db_in),
                      "basis": "18.14.3.2(a): Pu above Ag f'c / 10"}
    else:
        transverse = {"kind": "stirrups or hoops per 25.7", "spacing_max_in": d / 2.0,
                      "basis": "18.14.3.2(a): throughout the length at not more than d/2"}
    provisions = ["18.6.3.1 longitudinal limits", "transverse reinforcement throughout (18.14.3.2(a))"]
    if branch == BRANCH_YIELDING:
        provisions += ["18.6.5 design shear from the probable moments (18.14.3.3(b))",
                       "18.2.5 to 18.2.8 materials and splices (18.14.3.3(a))"]
    return {"branch": branch, "transverse": transverse, "axial_hoops": axial_hoops,
            "axial_threshold_kip": axial_threshold, "capacity_shear_required": branch == BRANCH_YIELDING,
            "provisions": provisions, "joints": JOINT_RULE[branch]}


def column_requirements(branch, smallest_bar_db_in, pu_kip, po_kip, ash_ratio_table_18_7_5_4=None):
    """What 18.14.3 asks of a gravity column in ``branch``.

    ``ash_ratio_table_18_7_5_4`` is the greater of Table 18.7.5.4 (a) and (b) as Ash / (s bc); the 18.14.3.2(c)
    column provides half of it over l0, the 18.14.3.3 column all of it (18.7.5).
    """
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}.")
    pu = abs(_number(pu_kip, "pu_kip"))
    po = _number(po_kip, "po_kip", positive=True)
    high_axial = pu > HIGH_AXIAL_FRACTION_OF_PO * po
    spacing = hoop_spacing_limit_in(smallest_bar_db_in)
    if branch == BRANCH_YIELDING:
        ash_fraction = 1.0
        confinement = "18.7.5 in full over l0 (18.14.3.3(c))"
    elif high_axial:
        ash_fraction = HIGH_AXIAL_ASH_FRACTION
        confinement = "half the greater of Table 18.7.5.4 (a) and (b) over l0 (18.14.3.2(c))"
    else:
        ash_fraction = 0.0
        confinement = "18.7.5.2(a) to (e) configuration over l0, no Ash quantity (18.14.3.2(b))"
    ash_required = None
    if ash_ratio_table_18_7_5_4 is not None:
        ash_required = ash_fraction * _number(ash_ratio_table_18_7_5_4, "ash_ratio_table_18_7_5_4", minimum=0.0)
    provisions = ["18.7.4.1 longitudinal ratio 1 to 6 percent", "18.7.6 design shear from the probable moments",
                  "hoops per 25.7.4 over the full length (18.14.3.2(b))", confinement]
    if branch == BRANCH_YIELDING:
        provisions += ["18.7.4 in full except 18.7.4.3 (18.14.3.3(c))", "18.2.5 to 18.2.8 materials and splices (18.14.3.3(a))"]
    return {"branch": branch, "high_axial": high_axial, "axial_ratio_to_po": pu / po,
            "hoop_spacing_full_length_max_in": spacing, "confinement_zone": "l0 from each joint face (18.7.5.1)",
            "ash_fraction_of_table_18_7_5_4": ash_fraction, "ash_ratio_required": ash_required,
            "bond_18_7_4_3_required": False, "provisions": provisions, "joints": JOINT_RULE[branch]}


# ---- checks -----------------------------------------------------------------------------------------------
def evaluate_gravity_beam(location, induced, strengths, section, detailing):
    """Checks of one gravity beam group.

    ``induced``: {"mu_kip_in", "vu_kip", "calculated": bool} or the ratio form {"moment_ratio", "shear_ratio"} of
    induced_branch; ``strengths``: {"phi_mn_kip_in", "phi_vn_kip",
    optional "ve_kip" (18.6.5 capacity shear demand) and "phi_vn_capacity_kip"}; ``section``: {"b_in", "d_in",
    "ag_in2", "fc_ksi", "fy_ksi", "as_top_in2", "as_bot_in2", "pu_kip"}; ``detailing``: {"stirrup_spacing_in",
    "smallest_bar_db_in", "hoops": bool}.
    """
    checks = []
    try:
        branch = induced_branch(induced, strengths)
        need = beam_requirements(branch["branch"], section["d_in"], detailing["smallest_bar_db_in"],
                                 pu_kip=section.get("pu_kip", 0.0), ag_in2=section.get("ag_in2"), fc_ksi=section.get("fc_ksi"))
    except (KeyError, TypeError, ValueError) as exc:
        return [not_evaluated("gravity.beam_compatibility_branch", "ACI 318-19 18.14.3.1", str(exc), location)]
    checks.append(make_check("gravity.beam_compatibility_branch", "ACI 318-19 18.14.3.1 / 18.14.2.1", 1, 1, "==", location=location,
                             details={**branch, "requirements": need}))
    spacing = detailing.get("stirrup_spacing_in")
    checks.append(make_check("gravity.beam_transverse_spacing", "ACI 318-19 18.14.3.2(a)", spacing, need["transverse"]["spacing_max_in"],
                             "<=", "in", location, details={"kind": need["transverse"]["kind"], "basis": need["transverse"]["basis"]}))
    if need["axial_hoops"]:
        checks.append(make_check("gravity.beam_hoops_for_axial_force", "ACI 318-19 18.14.3.2(a) with 18.7.5.2",
                                 int(bool(detailing.get("hoops"))), 1, "==", location=location,
                                 details={"pu_kip": section.get("pu_kip"), "threshold_kip": need["axial_threshold_kip"]}))
    # 18.6.3.1 longitudinal limits on each face
    try:
        b, d, fc, fy = (float(section[k]) for k in ("b_in", "d_in", "fc_ksi", "fy_ksi"))
        as_min = max(3.0 * math.sqrt(fc * 1000.0) / (fy * 1000.0), 200.0 / (fy * 1000.0)) * b * d
        for face in ("top", "bottom"):
            area = float(section[f"as_{face[:3]}_in2"])
            checks.append(make_check("gravity.beam_minimum_steel", "ACI 318-19 18.6.3.1 / 9.6.1.2 (18.14.3.2(a))", area, as_min, ">=", "in2",
                                     f"{location} {face}".strip()))
            checks.append(make_check("gravity.beam_maximum_steel_ratio", "ACI 318-19 18.6.3.1", area / (b * d),
                                     0.02 if fy >= 80.0 - 1e-9 else 0.025, "<=", "ratio",
                                     f"{location} {face}".strip()))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        checks.append(not_evaluated("gravity.beam_minimum_steel", "ACI 318-19 18.6.3.1", str(exc), location))
    if need["capacity_shear_required"]:
        ve, phi_vn = strengths.get("ve_kip"), strengths.get("phi_vn_capacity_kip")
        if ve is None or phi_vn is None:
            checks.append(not_evaluated("gravity.beam_capacity_shear", "ACI 318-19 18.6.5 (18.14.3.3(b))",
                                        "the probable-moment shear of the gravity beam was not supplied", location))
        else:
            checks.append(make_check("gravity.beam_capacity_shear", "ACI 318-19 18.6.5 (18.14.3.3(b))", ve, phi_vn, "<=", "kip", location))
    return checks


def evaluate_gravity_column(location, induced, strengths, section, detailing):
    """Checks of one gravity column group.

    ``induced``: {"mu_kip_in", "vu_kip", "calculated"} or the ratio form {"moment_ratio", "shear_ratio"};
    ``strengths``: {"phi_mn_kip_in", "phi_vn_kip", optional
    "ve_kip", "phi_vn_capacity_kip"}; ``section``: {"ag_in2", "ast_in2", "fc_ksi", "fy_ksi", "pu_kip"};
    ``detailing``: {"hoop_spacing_in", "smallest_bar_db_in", "ash_ratio_provided" (Ash / (s bc) over l0),
    "ash_ratio_table_18_7_5_4", "configuration_18_7_5_2_ok": bool}.
    """
    checks = []
    try:
        branch = induced_branch(induced, strengths)
        po = nominal_axial_strength_kip(section["ag_in2"], section["ast_in2"], section["fc_ksi"], section["fy_ksi"])
        need = column_requirements(branch["branch"], detailing["smallest_bar_db_in"], section.get("pu_kip", 0.0), po,
                                   ash_ratio_table_18_7_5_4=detailing.get("ash_ratio_table_18_7_5_4"))
    except (KeyError, TypeError, ValueError) as exc:
        return [not_evaluated("gravity.column_compatibility_branch", "ACI 318-19 18.14.3.1", str(exc), location)]
    checks.append(make_check("gravity.column_compatibility_branch", "ACI 318-19 18.14.3.1 / 18.14.2.1", 1, 1, "==", location=location,
                             details={**branch, "po_kip": po, "requirements": need}))
    rho = float(section["ast_in2"]) / float(section["ag_in2"])
    checks.append(make_check("gravity.column_minimum_rho", "ACI 318-19 18.7.4.1 (18.14.3.2(b))", rho, 0.01, ">=", "ratio", location))
    checks.append(make_check("gravity.column_maximum_rho", "ACI 318-19 18.7.4.1 (18.14.3.2(b))", rho, 0.06, "<=", "ratio", location))
    checks.append(make_check("gravity.column_hoop_spacing_full_length", "ACI 318-19 18.14.3.2(b)", detailing.get("hoop_spacing_in"),
                             need["hoop_spacing_full_length_max_in"], "<=", "in", location,
                             details={"basis": "hoops per 25.7.4 over the full length, the lesser of 6 db and 6 in"}))
    checks.append(make_check("gravity.column_hoop_configuration_lo", "ACI 318-19 18.7.5.2(a)-(e) over l0 (18.14.3.2(b))",
                             int(bool(detailing.get("configuration_18_7_5_2_ok"))), 1, "==", location=location))
    if need["ash_ratio_required"] is None:
        if need["ash_fraction_of_table_18_7_5_4"] > 0.0:
            checks.append(not_evaluated("gravity.column_confinement_lo", "ACI 318-19 Table 18.7.5.4 (18.14.3.2(c) / 18.14.3.3(c))",
                                        "the Table 18.7.5.4 ratio was not supplied", location))
    elif need["ash_fraction_of_table_18_7_5_4"] > 0.0:
        checks.append(make_check("gravity.column_confinement_lo", "ACI 318-19 Table 18.7.5.4 (18.14.3.2(c) / 18.14.3.3(c))",
                                 need["ash_ratio_required"], detailing.get("ash_ratio_provided"), "<=", "Ash/(s bc)", location,
                                 details={"fraction_of_table": need["ash_fraction_of_table_18_7_5_4"], "high_axial": need["high_axial"],
                                          "axial_ratio_to_po": need["axial_ratio_to_po"]}))
    ve, phi_vn = strengths.get("ve_kip"), strengths.get("phi_vn_capacity_kip")
    if ve is None or phi_vn is None:
        checks.append(not_evaluated("gravity.column_capacity_shear", "ACI 318-19 18.7.6 (18.14.3.2(b))",
                                    "the probable-moment shear of the gravity column was not supplied", location))
    else:
        checks.append(make_check("gravity.column_capacity_shear", "ACI 318-19 18.7.6 (18.14.3.2(b))", ve, phi_vn, "<=", "kip", location))
    return checks
