"""Capacity design at every joint: probable strengths, shear, joint shear, anchorage.

Pure computation on an explicit ``state`` dictionary (no OpenSees, no
Structure_Parameters); the driver assembles the state and applies the
selected transverse steel. Everything here is ACI 318-19 Chapter 18:

* Beam probable moment Mpr (18.6.5.1): the developed beam-plus-slab section
  of SMRF_Beam_Slab_Strength with steel at 1.25 fy and phi = 1.
* Beam design shear Ve: Mpr equilibrium over the clear span plus the
  factored gravity face reactions from the slab transfer (SMRF_Joints
  .beam_capacity_shear_envelope), for the high- and low-gravity seismic
  combinations. Vc = 0 in the hinge zone when the mechanism shear is at least
  half of Ve and the axial load is below Ag fc/20 (18.6.5.2); Vs limited to
  8 sqrt(fc) bw d (22.5.1.2). Hoop spacing/legs/bar are chosen from Ve, the
  18.6.4.4 spacing bounds and a 1-in grid from 3 in.
* Column design shear Ve (18.7.6.1.1), by a selectable, named method
  (``state["column_shear_method"]``, COLUMN_SHEAR_METHODS):
  - ``beam_joint_delivery_limited_v2`` (the default, the method every
    saved design was produced with): 2 Mpr,col / ln from the column's own
    probable P-M strength over its factored axial range, not exceeding the
    shear the beams' probable strengths can deliver through the joints
    (floor joints split the beam moments equally above/below; a roof joint
    sends all of it into the single column), never below the analysis
    shear.
  - ``column_own_probable_envelope_v3`` (2026-09-20, the engineering
    note's candidate): the column's own probable end moments over the
    factored axial range of each end, in each bending sense, summed for
    the governing sway and divided by the clear height, never below the
    analysis shear in that direction; no beam-sharing reduction. The
    probable moment over an axial range is the exact section solution at
    the maximizing load, located on the curve's vertices rather than on a
    sparse sample (probable_moment_over_axial_range).
  Both values are recorded for every story and direction whatever the
  method; the method only selects Ve. Vc per 18.7.6.2.1 / 22.5.5.1, hoops
  from Vs, the 18.7.5.4 confinement area and the 18.7.5.3 spacing limits
  (so from hx with every face bar tied).
* Joint shear (18.8.4): Vj = T + C - Vcol with T = 1.25 fy As including the
  slab bars in the effective flange, Vcol from the same mechanism, Aj per
  15.4.2.4 (SMRF_Joints.rectangular_joint_area), gamma per Table 18.8.4.3
  from its three inputs -- the column continuous or meeting 15.2.6, the beam
  in the direction of Vj continuous or meeting 15.2.7, confinement by two
  transverse beams per 15.2.8 (each at least 3/4 of the face it frames
  into, extending at least h beyond the joint, with two continuous top and
  bottom bars and No. 3 or larger stirrups) -- each resting on declared
  evidence, unknown evidence leaving the conservative row; phi = 0.85.
* Terminating beam bars (18.8.5.1): standard 90-degree hooks, ldh =
  fy db / (65 lambda sqrt(fc)) >= max(8 db, 6 in) inside the column depth
  less cover and hoop; through bars keep the 18.8.2.3 depth check.

Steel and concrete units: kip, inch, ksi (sqrt(fc) terms use psi).
"""
from __future__ import annotations

import math

import numpy as np

from Design.SMRF_Beam_Slab_Strength import composite_beam_strengths
from Design.SMRF_Common import make_check, not_evaluated
from Design.SMRF_Joints import (MPR_BASIS, beam_capacity_shear_envelope, rectangular_joint_area)

METHOD_VERSION = "aci318_19_capacity_design_v2_table_18_8_4_3_inputs"
_BAR = {3: (0.375, 0.11), 4: (0.5, 0.20), 5: (0.625, 0.31), 6: (0.75, 0.44), 7: (0.875, 0.60),
        8: (1.0, 0.79), 9: (1.128, 1.00), 10: (1.27, 1.27), 11: (1.41, 1.56), 14: (1.693, 2.25), 18: (2.257, 4.00)}
PHI_SHEAR = 0.75
PHI_JOINT = 0.85
SPACING_GRID_IN = 1.0
SPACING_MIN_IN = 3.0
STIRRUP_LADDER = ((4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (5, 4), (5, 6))   # (bar, legs): beam hoops
COLUMN_HOOP_BARS = (4, 5)   # column hoop bars; legs per direction come from the cage (SMRF_Cage_Layout)
# ACI 318-19 Table 18.8.4.3 (printed p. 313): Vn = gamma lambda sqrt(f'c) Aj,
# gamma by (column, beam in the direction of Vu, confinement by transverse
# beams according to 15.2.8). "continuous" = the table's "Continuous or meets
# 15.2.6" (column) / "Continuous or meets 15.2.7" (beam); "other" = its
# "Other" row. Read from the standard's text on 2026-09-18: 20 / 15 for a
# continuous column with a continuous beam (confined / not confined), 15 / 12
# with an Other beam, 15 / 12 for an Other column with a continuous beam,
# 12 / 8 for Other / Other. lambda = 1.0 (normalweight, 18.8.4.3).
JOINT_SHEAR_GAMMA = {("continuous", "continuous", True): 20.0, ("continuous", "continuous", False): 15.0,
                     ("continuous", "other", True): 15.0, ("continuous", "other", False): 12.0,
                     ("other", "continuous", True): 15.0, ("other", "continuous", False): 12.0,
                     ("other", "other", True): 12.0, ("other", "other", False): 8.0}
CONFINING_BEAM_WIDTH_RATIO = 0.75   # 15.2.8(a) and 18.8.3.2: beam width >= 3/4 of the column face it frames into
CONFINING_BEAM_MIN_BARS = 2         # 15.2.8(c): at least two continuous top and bottom bars
CONFINING_BEAM_MIN_STIRRUP = 3      # 15.2.8(c): No. 3 or larger stirrups
# Column design shear methods (18.7.6.1.1). Old evidence carries no method
# name and is read as the joint-limited method it was produced with.
COLUMN_SHEAR_METHOD_JOINT_LIMITED = "beam_joint_delivery_limited_v2"
COLUMN_SHEAR_METHOD_COLUMN_OWN = "column_own_probable_envelope_v3"
COLUMN_SHEAR_METHODS = (COLUMN_SHEAR_METHOD_JOINT_LIMITED, COLUMN_SHEAR_METHOD_COLUMN_OWN)
COLUMN_SHEAR_METHOD_DEFAULT = COLUMN_SHEAR_METHOD_JOINT_LIMITED
# Clear height of the column-own method: every story at the face-to-face
# height story_h - h_beam (the conservative convention of the engineering
# note, shorter than the base story's physical base-to-soffit height), or
# the physical height at the base story. The joint-limited method keeps the
# face-to-face height everywhere (its saved evidence was produced so).
CLEAR_HEIGHT_UNIFORM = "uniform_face_to_face"
CLEAR_HEIGHT_PHYSICAL = "physical_base"
CLEAR_HEIGHT_CONVENTIONS = (CLEAR_HEIGHT_UNIFORM, CLEAR_HEIGHT_PHYSICAL)
CLEAR_HEIGHT_CONVENTION_DEFAULT = CLEAR_HEIGHT_UNIFORM
PROBABLE_FY_FACTOR = 1.25           # 18.7.6.1.1 / 18.6.5.1: Mpr with steel at 1.25 fy, phi = 1
PROBABLE_CURVE_POINTS = 160         # the production P-M sweep (column_probable_pm)
PROBABLE_CURVE_REFINED_POINTS = 1600
EXACT_SECTION_TOLERANCE = 1e-14     # neutral-axis bisection tolerance as a fraction of h (axial residual ~1e-11 kip)
# Column local forces in the saved combination actions (Design_Driver
# _capture_element_actions): vecxz = (1, 0, 0) with local x up, so local y
# = -global Y and local z = +global X. Read from the 12-vector
# [N, Vy, Vz, T, My, Mz] at end i (bottom) then end j (top).
LOCAL_FORCE_CONVENTION = {
    "column_transformation": "geomTransf PDelta, vecxz = (1, 0, 0), local x from the bottom node upward",
    "local_axes": {"x": "+global Z", "y": "-global Y", "z": "+global X"},
    "shear_index": {"x": {"i": 2, "j": 8}, "y": {"i": 1, "j": 7}},
    "moment_index": {"x": {"i": 4, "j": 10}, "y": {"i": 5, "j": 11}},
    "direction_meaning": {"x": "shear of the x frames: local Vz (global X), bending My through the column depth h",
                          "y": "shear of the y frames: local Vy (global Y), bending Mz through the column width b"},
    "axial": "axial_i_kip / axial_j_kip are the joint-face axial loads at the bottom (i) and top (j) ends, "
             "compression positive (self-weight below the face removed)",
    "validated_by": "tests/test_smrf_column_shear_alternative.py: a cantilever column under a global X or Y tip force",
}


def column_confinement_ratio(b, h, fc, fy, clear_cover):
    """18.7.5.4 Ash/(s bc) without the high-axial form (Pu is not known before analysis)."""
    ag = b * h
    ach = (b - 2.0 * clear_cover) * (h - 2.0 * clear_cover)
    return max(0.3 * (ag / ach - 1.0) * fc / fy, 0.09 * fc / fy)


def column_layout_confinable(b, h, fc, fy, clear_cover, top_bars, side_bars):
    """Can the hoop bars offered confine this bar layout at the minimum spacing?

    Per direction (18.7.5.4), the most legs the face can engage -- one per
    bar, 18.7.5.2(b) -- with the largest hoop bar in COLUMN_HOOP_BARS must
    give Ash/s >= ratio * bc at SPACING_MIN_IN. A layout that fails this can
    never be detailed, whatever the shear, so the steel pick does not offer
    it; the high-axial form of the ratio is applied later by the capacity
    design and can still reject a layout that passes here.
    """
    ratio = column_confinement_ratio(b, h, fc, fy, clear_cover)
    ab = _BAR[COLUMN_HOOP_BARS[-1]][1]
    for legs_max, bc in ((top_bars, b - 2.0 * clear_cover), (side_bars + 2, h - 2.0 * clear_cover)):
        if legs_max * ab / (ratio * bc) < SPACING_MIN_IN - 1e-9:
            return False
    return True


def _grid_down(value):
    steps = math.floor((value - SPACING_MIN_IN) / SPACING_GRID_IN + 1e-9)
    return SPACING_MIN_IN + max(0, steps) * SPACING_GRID_IN if value >= SPACING_MIN_IN else None


def probable_beam_strengths(state, position, axis):
    """Composite beam Mn and Mpr (1.25 fy, phi = 1) for one family."""
    beam, slab = state["beam"], state["slab"]
    geometry = {"bay_x_in": state["geometry"]["bay_x_in"], "bay_y_in": state["geometry"]["bay_y_in"],
                "h_col_in": state["sections"]["h_col_in"], "b_col_in": state["sections"]["b_col_in"]}
    section = {"b_in": state["sections"]["b_beam_in"], "h_in": state["sections"]["h_beam_in"],
               "fc_ksi": state["sections"]["fc_beam_ksi"], "fy_ksi": state["materials"]["fy_ksi"],
               "bar_size": beam["bar_size"], "top_bars": beam["top_bars"], "bot_bars": beam["bot_bars"],
               # The direction's own top/bottom offsets when the orthogonal layers stack at the joints,
               # and its actual bar rows (2026-09-27) where the state carries them.
               "centroid_offset_in": (beam.get("centroid_offsets_by_axis_in") or {}).get(axis, beam["centroid_offset_in"]),
               "layers": (beam.get("layers") or {}).get(axis)}
    layout = slab.get("layout")
    slab_arg = {"thickness_in": slab["thickness_in"] if layout is not None else 0.0}
    nominal = composite_beam_strengths(section, slab_arg, layout, geometry, axis, position)
    probable = composite_beam_strengths({**section, "fy_ksi": 1.25 * section["fy_ksi"]},
                                        slab_arg, layout, geometry, axis, position)
    ab = _BAR[beam["bar_size"]][1]
    tension_hogging = beam["top_bars"] * ab + probable["slab_steel_in_flange_in2"]
    tension_sagging = beam["bot_bars"] * ab
    return {"axis": axis, "position": position,
            "mn_positive_kip_in": nominal["mn_positive_kip_in"], "mn_negative_kip_in": nominal["mn_negative_kip_in"],
            "mpr_positive_kip_in": probable["mn_positive_kip_in"], "mpr_negative_kip_in": probable["mn_negative_kip_in"],
            "effective_flange_width_in": probable["effective_flange_width_in"],
            "tension_steel_hogging_in2": tension_hogging, "tension_steel_sagging_in2": tension_sagging,
            "clear_span_in": nominal["clear_span_in"], "mpr_basis": MPR_BASIS}


def _span_face_reactions(beam, span_in, column_depth_in, factor):
    """Upward joint-face reactions of one span's transfer loads, zero end moments.

    The free body is the beam between the joint faces (18.6.5.1): a load at
    centerline fraction x sits at a' = x L - c/2 from the left face; loads
    inside a column footprint go straight to that column and are excluded.
    A point force P gives P (1 - a'/ln), P a'/ln; a local-y couple M (the
    slab's vertical-plane bending couple at that node, positive clockwise
    with local z up) gives -M/ln at the left face and +M/ln at the right.
    """
    clear = span_in - column_depth_in
    left = right = 0.0
    for x, p in beam["node_loads"]:
        a = x * span_in - 0.5 * column_depth_in
        if 0.0 < a < clear:
            left += factor * p * (1.0 - a / clear)
            right += factor * p * a / clear
    for x, _torsion, couple in beam.get("node_couples", []):
        a = x * span_in - 0.5 * column_depth_in
        if 0.0 < a < clear:
            left -= factor * couple / clear
            right += factor * couple / clear
    return left, right


def _drop_weight(beam, axis, geometry, sections):
    """Physical beam weight per inch between the faces; recovered from the smeared value if absent."""
    if beam.get("drop_weight_kip_per_in") is not None:
        return beam["drop_weight_kip_per_in"]
    span = geometry["bay_x_in"] if axis == "x" else geometry["bay_y_in"]
    depth = sections["h_col_in"] if axis == "x" else sections["b_col_in"]
    return beam["self_weight_kip_per_in"][axis] / (1.0 - depth / span)


def _face_reactions(transfer, axis, position, geometry, factor_dead, factor_live, beam_weight_per_in,
                    column_depth_in=None):
    """Envelope of the upward joint-face reactions of a family under factored gravity alone.

    Every span of the family and every unit case that can load it are
    evaluated end by end: dead once, live as the envelope over the all-panel
    case and each saved ACI 6.4.2 pattern (a pattern can load one span more
    than the full floor does). The left value is the largest left-face
    reaction over spans, the right value the largest right-face reaction,
    so neither end nor sway direction is judged on another end's span. With
    a zero live factor the live envelope is zero. Beam self-weight acts on
    the clear span.
    """
    nx, ny = geometry["num_bay_x"], geometry["num_bay_y"]
    if axis == "x":
        lines = [j for j in range(ny + 1) if (j in (0, ny)) == (position == "edge")]
    else:
        lines = [i for i in range(nx + 1) if (i in (0, nx)) == (position == "edge")]
    span = geometry["bay_x_in"] if axis == "x" else geometry["bay_y_in"]
    depth = column_depth_in if column_depth_in is not None else 0.0
    clear = span - depth
    # beam_weight_per_in is the PHYSICAL drop weight per inch between the
    # faces (state["beam"]["drop_weight_kip_per_in"]), not the smeared
    # centerline line weight, which already carries the clear fraction.
    self_weight = factor_dead * beam_weight_per_in * clear / 2.0
    cases = (transfer or {}).get("unit_cases", {}) if transfer else {}
    if not cases:
        return [self_weight, self_weight], "member self-weight only; no slab transfer supplied"
    dead, live = {}, {}
    for name, case in cases.items():
        factor = factor_dead if name == "dead" else factor_live
        if factor == 0.0 or not case:
            continue
        target = dead if name == "dead" else live
        for beam in case["beams"]:
            if beam["axis"] != axis or beam["line_index"] not in lines:
                continue
            key = (beam["line_index"], beam["span_index"])
            left, right = _span_face_reactions(beam, span, depth, factor)
            if name == "dead":
                target[key] = [left, right]
            else:
                # Live envelope per end over the all-panel case and every pattern.
                current = target.setdefault(key, [-math.inf, -math.inf])
                current[0] = max(current[0], left)
                current[1] = max(current[1], right)
    spans = set(dead) | set(live)
    if not spans:
        return [self_weight, self_weight], "member self-weight only; family has no transfer spans"
    left_by_span = {key: dead.get(key, [0.0, 0.0])[0] + (live[key][0] if key in live else 0.0) for key in spans}
    right_by_span = {key: dead.get(key, [0.0, 0.0])[1] + (live[key][1] if key in live else 0.0) for key in spans}
    left_key = max(left_by_span, key=left_by_span.get)
    right_key = max(right_by_span, key=right_by_span.get)
    return ([left_by_span[left_key] + self_weight, right_by_span[right_key] + self_weight],
            f"joint-face free body over the clear span; left governed by span {left_key}, right by span {right_key}; "
            f"live enveloped over {len([n for n in cases if n != 'dead'])} unit case(s); plus beam self-weight")


def design_beam_shear(state, strengths, transfer):
    """Ve at the faces for both seismic gravity families and the hinge-zone hoops."""
    sections, mats, beam, geometry = state["sections"], state["materials"], state["beam"], state["geometry"]
    fc, fy = sections["fc_beam_ksi"], mats["fy_ksi"]
    bw, h = sections["b_beam_in"], sections["h_beam_in"]
    # The smallest effective depth over directions and faces once the orthogonal layers stack.
    d = h - beam.get("worst_centroid_offset_in", beam["centroid_offset_in"])
    sds = state["sds"]
    families = {}
    worst_ve, worst_mechanism = 0.0, 0.0
    for key, s in strengths.items():
        entries = []
        for family_name, dead, live in (("seismic_high_gravity", 1.2 + 0.2 * sds, 1.0),
                                        ("seismic_low_gravity", 0.9 - 0.2 * sds, 0.0)):
            reactions, basis = _face_reactions(transfer, s["axis"], s["position"], geometry, dead, live,
                                               _drop_weight(beam, s["axis"], geometry, sections),
                                               column_depth_in=(sections["h_col_in"] if s["axis"] == "x"
                                                                else sections["b_col_in"]))
            data = {"id": f"{key}/{family_name}", "clear_span_in": s["clear_span_in"],
                    "mpr_left_positive_kip_in": s["mpr_positive_kip_in"], "mpr_left_negative_kip_in": s["mpr_negative_kip_in"],
                    "mpr_right_positive_kip_in": s["mpr_positive_kip_in"], "mpr_right_negative_kip_in": s["mpr_negative_kip_in"],
                    "gravity_reactions_kip": reactions, "mpr_basis": MPR_BASIS,
                    "gravity_basis": "factored_zero_end_moment_reactions",
                    "gravity_reaction_basis": basis, "gravity_family": family_name,
                    "dead_factor": dead, "live_factor": live}
            envelope = beam_capacity_shear_envelope(data)
            data["envelope"] = envelope
            entries.append(data)
            worst_ve = max(worst_ve, envelope["left_required_kip"], envelope["right_required_kip"])
            worst_mechanism = max(worst_mechanism, envelope["mechanism_shear_positive_kip"],
                                  envelope["mechanism_shear_negative_kip"])
        families[key] = entries
    # Hinge-zone concrete contribution (18.6.5.2): beams carry no axial load.
    vc_zero = worst_mechanism >= 0.5 * worst_ve
    vc = 0.0 if vc_zero else 2.0 * math.sqrt(fc * 1000.0) * bw * d / 1000.0
    vs_required = max(0.0, worst_ve / PHI_SHEAR - vc)
    vs_limit = 8.0 * math.sqrt(fc * 1000.0) * bw * d / 1000.0
    db = _BAR[beam["bar_size"]][0]
    bounds = {"d_over_4": d / 4.0, "six_db": 6.0 * db, "absolute": 6.0}
    # Hinge-zone hoops must also support the top and bottom bars (18.6.4.4 /
    # 25.7.2.3): the leg count is bounded by what the bars need and can engage.
    from Design.SMRF_Cage_Layout import beam_cage, cage_passes
    cover = beam.get("clear_cover_in")
    if cover is None:
        raise ValueError("Beam clear cover is required for the hoop arrangement.")
    selected, cage = None, None
    for bar, legs in STIRRUP_LADDER:
        trial = beam_cage(bw, cover, _BAR[bar][0], db, beam["top_bars"], beam["bot_bars"], legs=legs)
        if not cage_passes(trial):
            continue
        av = legs * _BAR[bar][1]
        s_shear = av * fy * d / vs_required if vs_required > 0 else float("inf")
        s_cap = min(s_shear, *bounds.values())
        spacing = _grid_down(s_cap)
        if spacing is not None:
            selected = {"bar_size": bar, "legs": legs, "spacing_in": spacing, "av_in2": av,
                        "spacing_from_shear_in": s_shear, "phi_vs_kip": PHI_SHEAR * av * fy * d / spacing,
                        "phi_vn_kip": PHI_SHEAR * (vc + av * fy * d / spacing), "crossties": legs - 2}
            cage = trial
            break
    if cage is None:
        first = beam_cage(bw, cover, _BAR[STIRRUP_LADDER[0][0]][0], db, beam["top_bars"], beam["bot_bars"])
        cage = {**first, "legs": None, "constructible": False, "arrangement": None, "hx_in": None,
                "checks": [{"rule": "18.6.4.4 / 25.7.2.3", "passes": False,
                            "detail": f"no hoop in the ladder is constructible: bars allow {first['legs_max']} legs, "
                                      f"support rules need {first['legs_min']}"}]}
    result = {"families": families, "ve_kip": worst_ve, "mechanism_shear_kip": worst_mechanism,
              "vc_zero_hinge_zone": vc_zero, "vc_kip": vc, "vs_required_kip": vs_required,
              "vs_limit_kip": vs_limit, "section_adequate": vs_required <= vs_limit, "cage": cage,
              "spacing_bounds_in": bounds, "hoops": selected,
              "basis": ("Vc = 0 in the hinge zone when the Mpr mechanism shear is at least half of Ve (18.6.5.2); "
                        "hoops at <= min(d/4, 6db, 6 in) and Av fyt d / Vs; uniform along the member")}
    for entries in families.values():
        for data in entries:
            data["phi_vn_left_kip"] = selected["phi_vn_kip"] if selected else 0.0
            data["phi_vn_right_kip"] = selected["phi_vn_kip"] if selected else 0.0
            data["shear_capacity_requirements_checked"] = bool(selected) and result["section_adequate"]
            data["hoops"] = selected
    return result


def column_probable_pm(state, fy_factor=1.25, n_pts=160):
    """Probable P-M surface of the column section (steel at fy_factor * fy)."""
    sections, col, mats = state["sections"], state["column"], state["materials"]
    b, h, fc = sections["b_col_in"], sections["h_col_in"], sections["fc_col_ksi"]
    fy, es = fy_factor * mats["fy_ksi"], mats["es_ksi"]
    layers = col["layers"]
    b1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 4.0)))
    ag, ast, hc = b * h, sum(a for a, _ in layers), h / 2.0
    p0 = 0.85 * fc * (ag - ast) + fy * ast
    diagram = [(0.80 * p0, 0.0)]
    for index in range(n_pts):
        c = 0.001 + (4.0 * h - 0.001) * index / (n_pts - 1)
        a = min(b1 * c, h)
        compression = 0.85 * fc * b * a
        axial, moment = compression, compression * (hc - a / 2.0)
        for area, depth in layers:
            strain = 0.003 * (c - depth) / c
            stress = max(-fy, min(fy, es * strain))
            net = stress - (0.85 * fc if depth <= a else 0.0)
            axial += net * area
            moment += net * area * (hc - depth)
        diagram.append((axial, abs(moment)))
    return sorted(diagram)


def _moment_at(diagram, axial):
    axials = [p for p, _ in diagram]
    moments = [m for _, m in diagram]
    if axial <= axials[0]:
        return moments[0]
    if axial >= axials[-1]:
        return moments[-1]
    for k in range(1, len(diagram)):
        if axial <= axials[k]:
            w = (axial - axials[k - 1]) / max(1e-12, axials[k] - axials[k - 1])
            return moments[k - 1] + w * (moments[k] - moments[k - 1])
    return moments[-1]


def column_action_envelopes(combinations, columns_per_story):
    """Per-story column axial and shear envelopes from saved combination actions, with their sources.

    ``combinations`` is the design_actions list: each carries ``id``,
    ``analysis_succeeded`` and ``members`` {tag: {member_type,
    local_force_kip_kipin (12), axial_i_kip, axial_j_kip}}. Column local
    forces follow LOCAL_FORCE_CONVENTION, so the shear of the x frames is
    local Vz (indices 2 / 8) and of the y frames local Vy (1 / 7). Returns
    the legacy per-story ``axial`` {story: (min, max)} and ``shear``
    {story: max |V| over both directions}, and ``detail`` per story: the
    axial range of each end with the combination and column tag of its
    extremes, the per-direction shear maxima with their sources, and the
    number of observations. A combination whose analysis did not succeed
    is skipped and counted; no combination at all is an error.
    """
    axial, shear, detail = {}, {}, {}
    used, skipped = 0, []
    for action in combinations or []:
        if action.get("analysis_succeeded") is False:
            skipped.append(action.get("id"))
            continue
        used += 1
        combination = action.get("id")
        for tag, member in (action.get("members") or {}).items():
            if member.get("member_type") != "column":
                continue
            story = (int(tag) - 1) // columns_per_story + 1
            forces = member["local_force_kip_kipin"]
            if len(forces) != 12 or not all(math.isfinite(float(v)) for v in forces):
                raise ValueError(f"column {tag} in {combination!r}: local force vector is not 12 finite values")
            entry = detail.setdefault(story, {
                "axial_by_end": {end: {"min_kip": math.inf, "max_kip": -math.inf, "min_source": None, "max_source": None}
                                 for end in ("i", "j")},
                "shear_by_direction_kip": {"x": 0.0, "y": 0.0},
                "shear_sources": {"x": None, "y": None},
                "shear_any_direction_kip": 0.0, "observations": 0})
            for end in ("i", "j"):
                value = float(member[f"axial_{end}_kip"])
                bucket = entry["axial_by_end"][end]
                if value < bucket["min_kip"]:
                    bucket["min_kip"], bucket["min_source"] = value, {"combination": combination, "column_tag": int(tag)}
                if value > bucket["max_kip"]:
                    bucket["max_kip"], bucket["max_source"] = value, {"combination": combination, "column_tag": int(tag)}
            for axis in ("x", "y"):
                for end, index in LOCAL_FORCE_CONVENTION["shear_index"][axis].items():
                    value = abs(float(forces[index]))
                    if value > entry["shear_by_direction_kip"][axis]:
                        entry["shear_by_direction_kip"][axis] = value
                        entry["shear_sources"][axis] = {"combination": combination, "column_tag": int(tag), "end": end,
                                                        "local_index": index}
            entry["shear_any_direction_kip"] = max(entry["shear_any_direction_kip"],
                                                   abs(float(forces[1])), abs(float(forces[2])),
                                                   abs(float(forces[7])), abs(float(forces[8])))
            entry["observations"] += 1
    if not used:
        raise ValueError("no successful combination actions: the column envelopes cannot be formed")
    for story, entry in detail.items():
        ends = entry["axial_by_end"]
        low = min(ends["i"]["min_kip"], ends["j"]["min_kip"])
        high = max(ends["i"]["max_kip"], ends["j"]["max_kip"])
        entry["axial_min_kip"], entry["axial_max_kip"] = low, high
        entry["axial_min_source"] = ends["i"]["min_source"] if ends["i"]["min_kip"] <= ends["j"]["min_kip"] else ends["j"]["min_source"]
        entry["axial_max_source"] = ends["i"]["max_source"] if ends["i"]["max_kip"] >= ends["j"]["max_kip"] else ends["j"]["max_source"]
        axial[story] = (low, high)
        shear[story] = entry["shear_any_direction_kip"]
    return {"axial": axial, "shear": shear, "detail": detail, "combinations_used": used,
            "combinations_skipped": skipped, "local_force_convention": LOCAL_FORCE_CONVENTION}


def _probable_section(b, h, fc, fy, es, layers):
    """Section constants shared by the sweep and the exact solver."""
    b1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 4.0)))
    ag, ast, hc = b * h, sum(a for a, _ in layers), h / 2.0
    p0 = 0.85 * fc * (ag - ast) + fy * ast
    return b1, ag, ast, hc, p0


def _section_forces_masked(b, h, fc, fy, es, layers, c, displaced):
    """(axial, signed moment) of the fibre model at depth c with an explicit displaced-concrete set.

    The declared section model subtracts the displaced concrete of a bar
    layer once the Whitney block reaches it (``_section_forces``). Fixing
    that set per branch lets a branch be evaluated as a continuous function
    up to and including its own end points, which is what the branch-aware
    solver needs; ``_section_forces`` itself is unchanged.
    """
    b1, ag, ast, hc, p0 = _probable_section(b, h, fc, fy, es, layers)
    a = min(b1 * c, h)
    compression = 0.85 * fc * b * a
    axial, moment = compression, compression * (hc - a / 2.0)
    for index, (area, depth) in enumerate(layers):
        strain = 0.003 * (c - depth) / c
        stress = max(-fy, min(fy, es * strain))
        net = stress - (0.85 * fc if displaced[index] else 0.0)
        axial += net * area
        moment += net * area * (hc - depth)
    return axial, moment


def _section_forces(b, h, fc, fy, es, layers, c):
    """(axial, |moment|) of the fibre model at neutral-axis depth c (compression face at depth 0)."""
    b1, ag, ast, hc, p0 = _probable_section(b, h, fc, fy, es, layers)
    a = min(b1 * c, h)
    compression = 0.85 * fc * b * a
    axial, moment = compression, compression * (hc - a / 2.0)
    for area, depth in layers:
        strain = 0.003 * (c - depth) / c
        stress = max(-fy, min(fy, es * strain))
        net = stress - (0.85 * fc if depth <= a else 0.0)
        axial += net * area
        moment += net * area * (hc - depth)
    return axial, abs(moment)


def probable_section_curve(b, h, fc, fy, es, layers, n_pts=PROBABLE_CURVE_POINTS):
    """The (P, M) sweep of column_probable_pm for an explicit section, steel at the given fy."""
    b1, ag, ast, hc, p0 = _probable_section(b, h, fc, fy, es, layers)
    diagram = [(0.80 * p0, 0.0)]
    for index in range(n_pts):
        c = 0.001 + (4.0 * h - 0.001) * index / (n_pts - 1)
        diagram.append(_section_forces(b, h, fc, fy, es, layers, c))
    return sorted(diagram)


SECTION_C_START_FRACTION = 1e-9     # the tension end of the sweep, as a fraction of h
SECTION_C_END_FACTOR = 50.0         # beyond the last block entry, the block saturation and every yield transition


def section_branches(b, h, fc, fy, es, layers):
    """The continuous branches of the declared section model, in increasing neutral-axis depth.

    The axial force of the model jumps down by 0.85 f'c A_k where the block
    reaches a bar layer (a = beta1 c = d_k), so it is monotone only between
    those depths. Each branch carries its closed c-interval and the set of
    layers whose displaced concrete is subtracted on it; a branch's function
    is continuous on the closed interval, and the values at a shared end
    point are the two one-sided limits of the model.
    """
    b1, ag, ast, hc, p0 = _probable_section(b, h, fc, fy, es, layers)
    entries = sorted({depth for _, depth in layers if 0.0 < depth < h})
    edges = [SECTION_C_START_FRACTION * h] + [depth / b1 for depth in entries] + [SECTION_C_END_FACTOR * h]
    branches = []
    for k in range(len(edges) - 1):
        threshold = entries[k - 1] if k >= 1 else -math.inf
        displaced = [depth <= threshold for _, depth in layers]
        branches.append({"index": k, "c_lo": edges[k], "c_hi": edges[k + 1], "displaced": displaced,
                         "block_entry_depth_in": entries[k - 1] if k >= 1 else None})
    return branches


def section_breakpoints(b, h, fc, fy, es, layers, lo, hi):
    """Slope changes of the branch functions strictly inside (lo, hi): steel yield transitions and block saturation."""
    b1, ag, ast, hc, p0 = _probable_section(b, h, fc, fy, es, layers)
    ey = fy / es
    points = set()
    for _, depth in layers:
        candidates = [0.003 * depth / (0.003 + ey)]                       # tension yield
        if ey < 0.003:
            candidates.append(0.003 * depth / (0.003 - ey))               # compression yield
        for c in candidates:
            if lo < c < hi:
                points.add(c)
    saturation = h / b1
    if lo < saturation < hi:
        points.add(saturation)
    return sorted(points)


def _bisect_monotone(f, target, lo, hi, tolerance):
    """Root of a monotone non-decreasing f on [lo, hi] with f(lo) <= target <= f(hi)."""
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance:
            break
    return 0.5 * (lo + hi)


def _section_domain(b, h, fc, fy, es, layers):
    b1, ag, ast, hc, p0 = _probable_section(b, h, fc, fy, es, layers)
    return -fy * ast, 0.80 * p0


def section_equilibrium_roots(b, h, fc, fy, es, layers, axial, tolerance=EXACT_SECTION_TOLERANCE):
    """Every equilibrium root of the declared section model at one axial load, branch by branch.

    Within a branch the axial force is monotone, so each branch whose
    one-sided range brackets the load holds exactly one root, found by
    bisection to ``tolerance`` x h in c. Where the load lies inside the jump
    at a block-entry depth, two adjacent branches bracket it and two roots
    are returned; each carries its neutral-axis depth, its own axial force,
    the residual against the requested load, its moment and its displaced
    layers. A load outside [-fy Ast, 0.80 P0] (22.4.2.1) raises. The
    caller decides which root to use; the design uses the largest moment
    (section_moment_at_axial).
    """
    tension, cap = _section_domain(b, h, fc, fy, es, layers)
    slack = 1e-9 * max(1.0, abs(axial))
    if not tension - slack <= axial <= cap + slack:
        raise ValueError(f"axial load {axial:.3f} kip is outside the section's [{tension:.3f}, {cap:.3f}] kip domain")
    roots = []
    for branch in section_branches(b, h, fc, fy, es, layers):
        displaced = branch["displaced"]

        def f(c):
            return _section_forces_masked(b, h, fc, fy, es, layers, c, displaced)[0]

        p_lo, p_hi = f(branch["c_lo"]), f(branch["c_hi"])
        if p_lo - slack <= axial <= p_hi + slack:
            c = _bisect_monotone(f, axial, branch["c_lo"], branch["c_hi"], tolerance * h)
            p, m = _section_forces_masked(b, h, fc, fy, es, layers, c, displaced)
            roots.append({"branch": branch["index"], "c_in": c, "axial_kip": p, "residual_kip": p - axial,
                          "moment_kip_in": abs(m), "signed_moment_kip_in": m,
                          "displaced_layers": [i for i, d in enumerate(displaced) if d],
                          "branch_axial_range_kip": [p_lo, p_hi],
                          "block_entry_depth_in": branch["block_entry_depth_in"]})
    if not roots:
        raise RuntimeError(f"no equilibrium root of the section model at {axial:.6f} kip (domain [{tension:.3f}, {cap:.3f}])")
    return roots


def section_moment_at_axial(b, h, fc, fy, es, layers, axial, tolerance=EXACT_SECTION_TOLERANCE, detail=False):
    """Probable moment of the section model at one axial load: the largest over its equilibrium roots (kip-in).

    The model's displaced-concrete jumps mean a load can sit in the gap
    between two branches and satisfy equilibrium at two neutral-axis
    depths; the larger moment is the capacity the design uses (a demand
    envelope), and every root is available through ``detail=True``.
    """
    roots = section_equilibrium_roots(b, h, fc, fy, es, layers, axial, tolerance)
    best = max(roots, key=lambda r: r["moment_kip_in"])
    if detail:
        return {**best, "roots": roots, "root_count": len(roots),
                "max_residual_kip": max(abs(r["residual_kip"]) for r in roots)}
    return best["moment_kip_in"]


def _golden_maximum(g, lo, hi, tolerance):
    """(value, c, final bracket width, evaluations) of the maximum of g on [lo, hi] by golden section."""
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    a, c = lo, hi
    x1, x2 = c - golden * (c - a), a + golden * (c - a)
    f1, f2 = g(x1), g(x2)
    evaluations = 2
    for _ in range(200):
        if c - a <= tolerance:
            break
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + golden * (c - a)
            f2 = g(x2)
        else:
            c, x2, f2 = x2, x1, f1
            x1 = c - golden * (c - a)
            f1 = g(x1)
        evaluations += 1
    value, point = max((f1, x1), (f2, x2))
    return value, point, c - a, evaluations


def _dense_golden_maximum(g, lo, hi, tolerance, samples=65):
    """(value, c, final bracket width, evaluations) of the maximum of g on [lo, hi]: a dense sample, then a
    golden-section search closed on the best sample's neighbours (the fallback where a segment's closed form is
    not trusted, probable_moment_over_axial_range)."""
    xs = [lo + (hi - lo) * k / (samples - 1) for k in range(samples)]
    values = [g(x) for x in xs]
    best = max(range(samples), key=lambda k: values[k])
    value, point, width, used = _golden_maximum(g, xs[max(0, best - 1)], xs[min(samples - 1, best + 1)], tolerance)
    if values[best] > value:
        value, point = values[best], xs[best]
    return value, point, width, samples + used


POLYNOMIAL_MISMATCH_TOLERANCE = 1e-8      # closed-form check of a smooth segment, relative to max(1, |M|) (kip-in)


def _segment_moment_polynomial(b, h, fc, fy, es, layers, displaced, c_mid):
    """(a, b, d0, e) of the signed masked moment M(c) = a c^2 + b c + d0 + e / c on one smooth segment.

    Between consecutive breakpoints (section_breakpoints) of a branch the
    steel state of every layer (elastic, yielded in tension, yielded in
    compression), the block saturation and the displaced-layer set are
    fixed, so the moment of ``_section_forces_masked`` is exactly this
    function of the neutral-axis depth. The states are read at ``c_mid``,
    the segment's midpoint; each layer's stress and the block depth are
    monotone in c, so a state shared by the two ends and the midpoint holds
    on the whole segment.

    Unsaturated block (beta1 c < h): a = -0.85 f'c b beta1^2 / 2 and
    b = 0.85 f'c b beta1 h / 2; saturated: a = b = 0 (the block's moment
    about mid-depth is zero). An elastic layer at depth d_k with area A_k
    has stress Es 0.003 (1 - d_k / c) and contributes Es 0.003 A_k (h/2 - d_k)
    to d0 and -Es 0.003 d_k A_k (h/2 - d_k) to e; a yielded layer contributes
    +-fy A_k (h/2 - d_k) to d0; a displaced layer subtracts 0.85 f'c A_k
    (h/2 - d_k) from d0.
    """
    b1, ag, ast, hc, p0 = _probable_section(b, h, fc, fy, es, layers)
    if b1 * c_mid < h:
        a2, a1 = -0.85 * fc * b * b1 ** 2 / 2.0, 0.85 * fc * b * b1 * h / 2.0
    else:
        a2, a1 = 0.0, 0.0
    d0, e = 0.0, 0.0
    for index, (area, depth) in enumerate(layers):
        lever = hc - depth
        stress = es * 0.003 * (c_mid - depth) / c_mid
        if stress <= -fy:
            d0 -= fy * area * lever
        elif stress >= fy:
            d0 += fy * area * lever
        else:
            d0 += es * 0.003 * area * lever
            e -= es * 0.003 * depth * area * lever
        if displaced[index]:
            d0 -= 0.85 * fc * area * lever
    return a2, a1, d0, e


def _stationary_depths(a, b, e, x0, x1, newton_steps=8):
    """Neutral-axis depths in [x0, x1] where a c^2 + b c + d0 + e / c is stationary, i.e. the real roots of
    2 a c^3 + b c^2 - e = 0 (a quadratic when a = 0, none when a = b = 0), each polished by Newton steps on the
    derivative and kept only when positive and inside the segment (a small relative tolerance at the ends)."""
    if a == 0.0 and b == 0.0:
        raw = []
    elif a == 0.0:
        q = e / b
        raw = [math.sqrt(q), -math.sqrt(q)] if q > 0.0 else ([0.0] if q == 0.0 else [])
    elif e == 0.0:
        raw = [0.0, -b / (2.0 * a)]
    else:
        # A near-double root can come back as a conjugate pair with a small imaginary part; keeping it is safe
        # because every candidate is evaluated through the model, never through the polynomial.
        raw = [float(r.real) for r in np.roots([2.0 * a, b, 0.0, -e]) if abs(r.imag) <= 1e-6 * max(1.0, abs(r.real))]
    slack = 1e-9 * max(x1 - x0, x1)
    depths = []
    for c in raw:
        if not (c > 0.0 and x0 - slack <= c <= x1 + slack):
            continue
        for _ in range(newton_steps):
            slope = 2.0 * a * c + b - e / (c * c)
            curvature = 2.0 * a + 2.0 * e / (c * c * c)
            if curvature == 0.0:
                break
            step = slope / curvature
            trial = c - step
            if not (trial > 0.0 and x0 - slack <= trial <= x1 + slack):
                break
            c = trial
            if abs(step) <= 1e-15 * c:
                break
        depths.append(min(x1, max(x0, c)))
    return sorted(depths)


def probable_moment_over_axial_range(section, p_min, p_max, n_pts=PROBABLE_CURVE_POINTS,
                                     refined_pts=PROBABLE_CURVE_REFINED_POINTS, dense_check_points=0,
                                     tolerance=EXACT_SECTION_TOLERANCE):
    """Largest probable moment of one section and bending sense over an axial range, branch-aware.

    ``section`` = {b_in, h_in, fc_ksi, fy_ksi (already factored), es_ksi,
    layers}. For every continuous branch of the section model the
    neutral-axis interval whose axial force lies in [p_min, p_max] is found
    by bisection (monotone within the branch); the branch's moment is then
    maximized on that closed interval, whose candidates are its end points
    (a range end, a block-entry limit, the tension end or the domain end),
    every steel-yield transition and the block saturation inside it, and
    the stationary points of the moment on each smooth sub-segment between
    them. On such a segment the steel states and the block saturation are
    fixed, so the signed moment is exactly M(c) = a c^2 + b c + d0 + e / c
    (``_segment_moment_polynomial``); the closed form is verified against
    the model at the segment's ends and midpoint, its stationary points are
    the real roots of 2 a c^3 + b c^2 - e = 0 (``_stationary_depths``), and
    each root is evaluated through the model itself. A segment whose closed
    form does not reproduce the model to rounding level (a coefficient bug
    or a missed breakpoint) falls back to a dense golden-section search and
    is counted. The largest candidate over all branches is the envelope; a
    load inside a block-entry jump is therefore covered by both adjacent
    branches. The result names the branch and the kind of point, and carries
    evidence: the stationary points found, the largest closed-form mismatch,
    the fallback count (and its bracket widths), the evaluation count, the
    round-trip residual (the winning depth's own axial force re-solved for
    its roots reproduces the moment) and, when ``dense_check_points`` > 0,
    a dense sweep of the raw model restricted to the range whose maximum
    must not exceed the envelope. The legacy 160-point and refined curve
    peaks are reported for comparison.
    """
    b, h, fc = section["b_in"], section["h_in"], section["fc_ksi"]
    fy, es, layers = section["fy_ksi"], section["es_ksi"], section["layers"]
    p_min, p_max = float(p_min), float(p_max)
    if p_min > p_max:
        raise ValueError("p_min must not exceed p_max")
    tension, cap = _section_domain(b, h, fc, fy, es, layers)
    slack = 1e-9 * max(1.0, abs(p_max), abs(p_min))
    if p_min < tension - slack or p_max > cap + slack:
        raise ValueError(f"axial range [{p_min:.3f}, {p_max:.3f}] kip leaves the section's [{tension:.3f}, {cap:.3f}] kip domain")
    c_tolerance = tolerance * h
    candidates, evaluations, bracket_widths, branches_checked = [], 0, [], []
    stationary_points, segments_checked, fallback_segments, max_mismatch = [], 0, 0, 0.0
    for branch in section_branches(b, h, fc, fy, es, layers):
        displaced = branch["displaced"]

        def f(c):
            return _section_forces_masked(b, h, fc, fy, es, layers, c, displaced)[0]

        def m(c):
            return _section_forces_masked(b, h, fc, fy, es, layers, c, displaced)[1]

        def g(c):
            return abs(m(c))

        p_lo, p_hi = f(branch["c_lo"]), f(branch["c_hi"])
        evaluations += 2
        if p_hi < p_min - slack or p_lo > p_max + slack:
            continue
        if p_lo >= p_min - slack:
            c_a, kind_a = branch["c_lo"], ("tension_end" if branch["index"] == 0 else "block_entry_limit")
        else:
            c_a, kind_a = _bisect_monotone(f, p_min, branch["c_lo"], branch["c_hi"], c_tolerance), "range_min"
        if p_hi <= p_max + slack:
            c_b, kind_b = branch["c_hi"], ("domain_end" if branch["block_entry_depth_in"] is None and branch["index"] > 0
                                           or branch["c_hi"] >= SECTION_C_END_FACTOR * h else "block_entry_limit")
        else:
            c_b, kind_b = _bisect_monotone(f, p_max, branch["c_lo"], branch["c_hi"], c_tolerance), "range_max"
        if c_b < c_a:
            continue
        branches_checked.append(branch["index"])
        interior = section_breakpoints(b, h, fc, fy, es, layers, c_a, c_b)
        points = [(c_a, kind_a)] + [(c, "yield_or_saturation_transition") for c in interior] + [(c_b, kind_b)]
        for c, kind in points:
            candidates.append((g(c), c, f(c), branch["index"], kind))
            evaluations += 2
        for (x0, _), (x1, _) in zip(points, points[1:]):
            if x1 - x0 <= c_tolerance:
                continue
            # On the smooth segment the signed moment is a c^2 + b c + d0 + e / c; the closed form is
            # checked against the model at both ends and the midpoint before its stationary points are used.
            segments_checked += 1
            c_mid = 0.5 * (x0 + x1)
            qa, qb, qd, qe = _segment_moment_polynomial(b, h, fc, fy, es, layers, displaced, c_mid)
            trusted = True
            for x in (x0, c_mid, x1):
                signed = m(x)
                evaluations += 1
                mismatch = abs(qa * x * x + qb * x + qd + qe / x - signed)
                max_mismatch = max(max_mismatch, mismatch)
                if mismatch > POLYNOMIAL_MISMATCH_TOLERANCE * max(1.0, abs(signed)):
                    trusted = False
            if not trusted:
                fallback_segments += 1
                value, c, width, used = _dense_golden_maximum(g, x0, x1, c_tolerance)
                evaluations += used
                bracket_widths.append(width)
                candidates.append((value, c, f(c), branch["index"], "fallback_search"))
                evaluations += 1
                continue
            for c in _stationary_depths(qa, qb, qe, x0, x1):
                value, p = g(c), f(c)
                evaluations += 2
                candidates.append((value, c, p, branch["index"], "stationary"))
                stationary_points.append({"c_in": c, "axial_kip": p, "moment_kip_in": value, "branch": branch["index"]})
    if not candidates:
        raise RuntimeError(f"no neutral-axis depth of the section model gives an axial force in [{p_min}, {p_max}] kip")
    best_m, best_c, best_p, best_branch, best_kind = max(candidates, key=lambda item: item[0])
    # An interior maximum (a stationary point of the closed form, or the fallback search's) is located
    # 'interior'; the raw kind of the winning candidate is reported beside it (candidate_kind).
    location = {"stationary": "interior", "fallback_search": "interior"}.get(best_kind, best_kind)
    # Legacy curve peaks for comparison (vertex scan of the raw sweep).
    coarse = probable_section_curve(b, h, fc, fy, es, layers, n_pts)
    refined = probable_section_curve(b, h, fc, fy, es, layers, refined_pts)

    def curve_peak(curve):
        points = [p_min, p_max] + [p for p, _ in curve if p_min < p < p_max]
        return max((_moment_at(curve, p), p) for p in points)

    m_coarse, p_coarse = curve_peak(coarse)
    m_refined, p_refined = curve_peak(refined)
    # Round trip: the winning depth's own axial force, re-solved, reproduces the moment.
    roots = section_equilibrium_roots(b, h, fc, fy, es, layers, best_p, tolerance)
    round_trip = min(abs(r["moment_kip_in"] - best_m) for r in roots)
    result = {"mpr_kip_in": best_m, "at_axial_kip": best_p, "at_neutral_axis_depth_in": best_c, "location": location,
              "candidate_kind": best_kind, "branch": best_branch, "axial_range_kip": [p_min, p_max],
              "curve_peak_kip_in": m_coarse, "curve_peak_at_axial_kip": p_coarse, "curve_points": n_pts,
              "refined_curve_peak_kip_in": m_refined, "refined_curve_peak_at_axial_kip": p_refined,
              "refined_curve_points": refined_pts,
              "relative_difference_curve_vs_exact": abs(m_coarse - best_m) / max(1e-12, best_m),
              "relative_difference_refined_vs_exact": abs(m_refined - best_m) / max(1e-12, best_m),
              "vertices_checked": len(candidates),
              "numerics": {"branches_checked": branches_checked, "candidates": len(candidates), "evaluations": evaluations,
                           "segments_checked": segments_checked,
                           "stationary_candidates": len(stationary_points),
                           "stationary_points": stationary_points,
                           "polynomial_max_mismatch_kip_in": max_mismatch,
                           "polynomial_mismatch_tolerance": POLYNOMIAL_MISMATCH_TOLERANCE,
                           "polynomial_fallback_segments": fallback_segments,
                           "golden_section_bracket_widths_in": bracket_widths,
                           "neutral_axis_tolerance_in": c_tolerance,
                           "round_trip_moment_residual_kip_in": round_trip,
                           "roots_at_peak_axial": [{k: r[k] for k in ("branch", "c_in", "axial_kip", "residual_kip", "moment_kip_in")}
                                                   for r in roots],
                           "max_root_residual_kip": max(abs(r["residual_kip"]) for r in roots)}}
    if dense_check_points:
        c_end = SECTION_C_END_FACTOR * h
        dense_best = None
        for k in range(1, dense_check_points + 1):
            c = SECTION_C_START_FRACTION * h + (c_end - SECTION_C_START_FRACTION * h) * (k / dense_check_points) ** 3
            p, m = _section_forces(b, h, fc, fy, es, layers, c)
            if p_min <= p <= p_max and (dense_best is None or m > dense_best[0]):
                dense_best = (m, c, p)
        result["numerics"]["dense_sweep"] = {"points": dense_check_points,
                                             "max_moment_kip_in": dense_best[0] if dense_best else None,
                                             "at_neutral_axis_depth_in": dense_best[1] if dense_best else None,
                                             "at_axial_kip": dense_best[2] if dense_best else None,
                                             "envelope_minus_dense_kip_in": (best_m - dense_best[0]) if dense_best else None,
                                             "envelope_covers_sweep": (dense_best is None) or (best_m >= dense_best[0] - 1e-9 * max(1.0, best_m))}
    return result


def _mirrored(layers, depth):
    """The same bars seen from the opposite compression face."""
    return sorted((area, depth - d) for area, d in layers)


def design_column_shear(state, strengths, method=None):
    """Column Ve per story and direction; hoops from shear, confinement and spacing limits.

    The two frame directions are designed separately (18.7.6.1.1 Ve, 18.7.5.4
    Ash and 22.5 Vs are all per direction): the x frame bends the column
    through h (compression on a b face; ``column.layers``) and its shear runs
    along h, resisted by the legs that cross the b faces; the y frame bends it
    through b (``column.layers_about_z``) with shear along b, resisted by the
    legs that cross the h faces. One hoop bar and one spacing serve both
    directions; the leg count is chosen per direction.

    ``method`` (else ``state["column_shear_method"]``, else the default)
    names the Ve rule, COLUMN_SHEAR_METHODS. Every story entry records the
    joint-limited value, the legacy nine-sample column-own value and the
    exact column-own envelope of each end and sense over the saved axial
    ranges (``state["column_action_envelopes"]``, else the story range for
    both ends); the method decides which becomes ``ve_kip``.
    """
    sections, col, mats, geometry = state["sections"], state["column"], state["materials"], state["geometry"]
    b, h, fc = sections["b_col_in"], sections["h_col_in"], sections["fc_col_ksi"]
    fy, es = mats["fy_ksi"], mats["es_ksi"]
    method = method or state.get("column_shear_method") or COLUMN_SHEAR_METHOD_DEFAULT
    if method not in COLUMN_SHEAR_METHODS:
        raise ValueError(f"unknown column shear method {method!r}; expected one of {COLUMN_SHEAR_METHODS}")
    convention = state.get("column_clear_height_convention") or CLEAR_HEIGHT_CONVENTION_DEFAULT
    if convention not in CLEAR_HEIGHT_CONVENTIONS:
        raise ValueError(f"unknown clear height convention {convention!r}; expected one of {CLEAR_HEIGHT_CONVENTIONS}")
    story_h, h_beam = geometry["story_h_in"], sections["h_beam_in"]
    ln_face = story_h - h_beam
    offset = col["centroid_offset_in"]
    cc = col["clear_cover_in"]
    ag = b * h
    envelopes = state.get("column_action_envelopes") or {}
    # Beam probable moments a joint can deliver to its columns, per direction.
    def beams_at(kind, axis):
        interior = strengths[f"{axis}_interior"]
        edge = strengths[f"{axis}_edge"]
        if kind == "interior":
            return interior["mpr_negative_kip_in"] + interior["mpr_positive_kip_in"]
        if kind == "edge":            # one interior beam perpendicular, two edge beams parallel
            return max(edge["mpr_negative_kip_in"] + edge["mpr_positive_kip_in"],
                       max(interior["mpr_negative_kip_in"], interior["mpr_positive_kip_in"]))
        return max(edge["mpr_negative_kip_in"], edge["mpr_positive_kip_in"])
    if b != h and not col.get("layers_about_z"):
        raise ValueError("Rectangular columns require layers_about_z for orthogonal probable strength.")
    layers_about_z = col.get("layers_about_z") or col["layers"]
    fy_pr = PROBABLE_FY_FACTOR * fy
    directions = {
        # depth along the shear, width across it, Ash core dimension perpendicular to the legs
        "x": {"depth_in": h, "width_in": b, "legs_key": "across_b_face", "bc_in": b - 2.0 * cc,
              "diagram": column_probable_pm({**state, "column": {**col, "layers": col["layers"]}}),
              "section": {"b_in": b, "h_in": h, "fc_ksi": fc, "fy_ksi": fy_pr, "es_ksi": es, "layers": list(col["layers"])},
              "layers_basis": "column.layers (bending through h, compression on a b face)"},
        "y": {"depth_in": b, "width_in": h, "legs_key": "across_h_face", "bc_in": h - 2.0 * cc,
              "diagram": column_probable_pm({**state,
                  "sections": {**sections, "b_col_in": h, "h_col_in": b},
                  "column": {**col, "layers": layers_about_z}}),
              "section": {"b_in": h, "h_in": b, "fc_ksi": fc, "fy_ksi": fy_pr, "es_ksi": es, "layers": list(layers_about_z)},
              "layers_basis": ("column.layers_about_z (bending through b, compression on an h face)"
                               if col.get("layers_about_z") else "column.layers reused; layers_about_z not supplied")},
    }
    joint_delivery = {axis: {kind: beams_at(kind, axis) for kind in ("interior", "edge", "corner")} for axis in directions}
    cage_identity = (f"{b:g}x{h:g} in f'c {fc:g} ksi; #{col['bar_size']} bars: {col['top_bars']} top, {col['bot_bars']} bottom, "
                     f"{col['side_bars']} per side face; centroid offset {offset:g} in")
    strength_basis = (f"probable flexural strength: fibre section with steel at {PROBABLE_FY_FACTOR:g} fy = {fy_pr:g} ksi, "
                      f"phi = 1, ecu 0.003, Whitney block (ACI 318-19 18.7.6.1.1, 18.6.5.1 / R18.7.6.1); exact neutral-axis "
                      f"solution at the maximizing axial load of each range")
    stories, worst = {}, {}
    for axis, spec in directions.items():
        d = spec["depth_in"] - offset
        width = spec["width_in"]
        section = spec["section"]
        mirrored = {**section, "layers": _mirrored(section["layers"], section["h_in"])}
        for story in range(1, geometry["num_floor"] + 1):
            p_min, p_max = state["column_axial_envelope"].get(story, (0.0, 0.0))
            detail = envelopes.get(story) or envelopes.get(str(story))
            top_is_roof = story == geometry["num_floor"]
            ln_physical = story_h - 0.5 * h_beam if story == 1 else ln_face
            ln_own = ln_physical if convention == CLEAR_HEIGHT_PHYSICAL else ln_face
            # --- joint-limited method numbers (unchanged from the saved evidence) ---
            samples = [p_min + (p_max - p_min) * k / 8.0 for k in range(9)]
            mpr_col = max(_moment_at(spec["diagram"], p) for p in samples)
            ve_own_legacy = 2.0 * mpr_col / ln_face
            # Interior column: floor joint below gives half the beam sum, joint above
            # gives half (or all of it at the roof).
            delivery = joint_delivery[axis]["interior"]
            m_bottom = delivery / 2.0 if story > 1 else mpr_col
            m_top = delivery if top_is_roof else delivery / 2.0
            ve_joint_limited = (m_top + m_bottom) / ln_face
            vu_any = state["column_shear_demand"].get(story, 0.0)
            vu_direction = detail["shear_by_direction_kip"][axis] if detail else None
            # --- column-own envelope: each end over its own axial range, each bending sense ---
            if detail:
                ends = {end: (detail["axial_by_end"][end]["min_kip"], detail["axial_by_end"][end]["max_kip"]) for end in ("i", "j")}
                end_sources = {end: {k: detail["axial_by_end"][end][k] for k in ("min_source", "max_source")} for end in ("i", "j")}
                end_basis = "each end over its own saved factored axial range (all combinations, every column of the story)"
            else:
                ends = {"i": (p_min, p_max), "j": (p_min, p_max)}
                end_sources = {"i": None, "j": None}
                end_basis = "per-end ranges not recorded: the story range is used for both ends"
            mpr = {end: {"positive": probable_moment_over_axial_range(section, *ends[end]),
                         "negative": probable_moment_over_axial_range(mirrored, *ends[end])} for end in ("i", "j")}
            sway = {"a": mpr["i"]["positive"]["mpr_kip_in"] + mpr["j"]["negative"]["mpr_kip_in"],
                    "b": mpr["i"]["negative"]["mpr_kip_in"] + mpr["j"]["positive"]["mpr_kip_in"]}
            governing_sway = max(sway, key=sway.get)
            ve_own_envelope = sway[governing_sway] / ln_own
            story_env = max(probable_moment_over_axial_range(section, p_min, p_max),
                            probable_moment_over_axial_range(mirrored, p_min, p_max), key=lambda r: r["mpr_kip_in"])
            ve_own_story = 2.0 * story_env["mpr_kip_in"] / ln_own
            if method == COLUMN_SHEAR_METHOD_JOINT_LIMITED:
                vu, vu_basis = vu_any, "largest |V| over both local shear axes of every combination (legacy scalar)"
                ve = max(min(ve_own_legacy, ve_joint_limited), vu)
                mechanism = min(ve_own_legacy, ve_joint_limited)
                ve_basis = ("max(min(2 Mpr,col / ln over the story's factored axial range on the nine-sample curve, "
                            "joint-limited beam Mpr delivery), Vu)")
                ln_used, ln_basis = ln_face, "face-to-face height story_h - h_beam at every story (the method's convention)"
            else:
                if vu_direction is not None:
                    vu = vu_direction
                    vu_basis = (f"largest |V| along the {axis}-frame shear axis (local index "
                                f"{LOCAL_FORCE_CONVENTION['shear_index'][axis]}) of every combination")
                else:
                    vu, vu_basis = vu_any, "per-direction shear not recorded: legacy any-direction maximum used (conservative)"
                ve = max(ve_own_envelope, vu)
                mechanism = ve_own_envelope
                ve_basis = ("max((Mpr,end i + Mpr,end j) / ln for the governing sway, each end at its own factored "
                            "axial range and bending sense, Vu in this direction); no beam-sharing reduction")
                ln_used, ln_basis = ln_own, (f"{convention}: " + ("physical base-to-soffit height at story 1, face-to-face elsewhere"
                                                                  if convention == CLEAR_HEIGHT_PHYSICAL
                                                                  else "face-to-face height story_h - h_beam at every story"))
            vc_zero = mechanism >= 0.5 * ve and p_min < ag * fc / 20.0
            nu_psi = max(0.0, p_min) * 1000.0 / ag
            vc = 0.0 if vc_zero else (2.0 * math.sqrt(fc * 1000.0) + min(nu_psi / 6.0, 0.05 * fc * 1000.0)) * width * d / 1000.0
            vs_required = max(0.0, ve / PHI_SHEAR - vc)
            entry = {"axis": axis, "story": story, "method": method,
                     "axial_min_kip": p_min, "axial_max_kip": p_max,
                     "mpr_column_kip_in": mpr_col, "ve_own_kip": ve_own_legacy, "ve_joint_limited_kip": ve_joint_limited,
                     "ve_own_envelope_kip": ve_own_envelope, "ve_own_story_envelope_kip": ve_own_story,
                     "vu_analysis_kip": vu, "vu_direction_kip": vu_direction, "vu_any_direction_kip": vu_any, "vu_basis": vu_basis,
                     "ve_kip": ve, "ve_basis": ve_basis,
                     "clear_height_in": ln_used, "clear_height_basis": ln_basis,
                     "face_to_face_clear_height_in": ln_face, "physical_clear_height_in": ln_physical,
                     "ve_own_envelope_at_physical_height_kip": sway[governing_sway] / ln_physical,
                     "vc_zero": vc_zero, "vc_kip": vc,
                     "vc_basis": ("Vc = 0: the earthquake-induced mechanism shear is at least half of Ve and Pu,min < Ag f'c / 20 "
                                  "(18.7.6.2.1 (a), (b))" if vc_zero else
                                  "Vc per 22.5.5.1 with axial load (18.7.6.2.1 conditions not both met)"),
                     "phi": PHI_SHEAR, "vs_required_kip": vs_required, "roof_story": top_is_roof,
                     "column_own": {
                         "axial_by_end_kip": {end: {"min_kip": ends[end][0], "max_kip": ends[end][1]} for end in ("i", "j")},
                         "axial_sources": end_sources, "axial_range_basis": end_basis,
                         "mpr_by_end_and_sense": mpr,
                         "sway_sums_kip_in": {"a": sway["a"], "b": sway["b"]},
                         "sway_basis": ("a: end i compression on the depth-0 face with end j on the opposite face; b: the reverse "
                                        "(double curvature); the larger sum governs"),
                         "governing_sway": governing_sway,
                         "mpr_end_i_kip_in": mpr["i"]["positive" if governing_sway == "a" else "negative"]["mpr_kip_in"],
                         "mpr_end_j_kip_in": mpr["j"]["negative" if governing_sway == "a" else "positive"]["mpr_kip_in"],
                         "story_envelope": {**story_env, "ve_kip": ve_own_story,
                                            "basis": "twice the largest Mpr of either sense over the story's merged axial range "
                                                     "(the engineering note's symmetric-section special case)"},
                         "form_difference_kip": ve_own_story - ve_own_envelope,
                         "strength_basis": strength_basis,
                         "section": {"b_in": section["b_in"], "h_in": section["h_in"], "fc_ksi": fc,
                                     "fy_probable_ksi": fy_pr, "es_ksi": es, "layers_area_in2_depth_in": section["layers"],
                                     "bar_count": col["top_bars"] + col["bot_bars"] + 2 * col["side_bars"],
                                     "bending_depth_in": spec["depth_in"], "shear_width_in": width, "d_in": d,
                                     "layers_basis": spec["layers_basis"], "cage_identity": cage_identity},
                         "numerics": {"curve_points": PROBABLE_CURVE_POINTS, "refined_curve_points": PROBABLE_CURVE_REFINED_POINTS,
                                      "max_relative_difference_curve_vs_exact": max(
                                          r["relative_difference_curve_vs_exact"] for e in mpr.values() for r in e.values()),
                                      "max_relative_difference_refined_vs_exact": max(
                                          r["relative_difference_refined_vs_exact"] for e in mpr.values() for r in e.values())}}}
            stories.setdefault(axis, {})[story] = entry
            if axis not in worst or vs_required > worst[axis]["vs_required_kip"]:
                worst[axis] = entry
    vs_limit = {axis: 8.0 * math.sqrt(fc * 1000.0) * spec["width_in"] * (spec["depth_in"] - offset) / 1000.0
                for axis, spec in directions.items()}
    governing = max(worst.values(), key=lambda w: w["vs_required_kip"]) if worst else None
    section_adequate = all(worst[axis]["vs_required_kip"] <= vs_limit[axis] for axis in worst)
    # Confinement (18.7.5.4) per direction, with every face bar tied by a leg or crosstie.
    ach = (b - 2.0 * cc) * (h - 2.0 * cc)
    p_max_all = max(v["axial_max_kip"] for axis in stories for v in stories[axis].values()) if stories else 0.0
    ratio = max(0.3 * (ag / ach - 1.0) * fc / fy, 0.09 * fc / fy)
    high_axial = p_max_all > 0.3 * ag * fc
    if high_axial:
        kf = max(1.0, fc / 25.0 + 0.6)
        perimeter_bars = col["top_bars"] + col["bot_bars"] + 2 * col["side_bars"]
        kn = perimeter_bars / max(1, perimeter_bars - 2)
        ratio = max(ratio, 0.2 * kf * kn * p_max_all / (fy * ach))
    # The cage: a perimeter hoop plus crossties, each engaging a bar. Which
    # bars need support (25.7.2.3 through 18.7.5.2(d), hx per 18.7.5.2(e)/(f))
    # and which can carry a crosstie (18.7.5.2(b)) bound the leg count in
    # each direction independently; Av and Ash below use each direction's
    # own realized legs, and hx is the arrangement's supported-bar spacing,
    # not an assumption (SMRF_Cage_Layout).
    from Design.SMRF_Cage_Layout import column_cage, cage_passes
    db = _BAR[col["bar_size"]][0]

    def bounds_for(hx):
        # The hx-dependent so (4-6 in) is recorded, but the methodology keeps a
        # conservative 4-in cap.
        return {"quarter_min_dimension": min(b, h) / 4.0, "six_db": 6.0 * db,
                "so": max(4.0, min(6.0, 4.0 + (14.0 - hx) / 3.0)), "conservative_so_cap": 4.0}

    selected, cage, bounds = None, None, None
    for bar in COLUMN_HOOP_BARS:
        base = column_cage(b, h, cc, _BAR[bar][0], db, col["top_bars"], col["side_bars"], high_axial=high_axial)
        options = base["constructible_legs_by_direction"]
        # Every constructible leg pair is priced; the pick for this bar is the
        # largest spacing the pair allows, then the fewest legs -- a crosstie
        # is cheaper to add than a hoop set.
        candidates = []
        for nb in options["across_b_face"]:
            for nh in options["across_h_face"]:
                legs = {"across_b_face": nb, "across_h_face": nh}
                trial = column_cage(b, h, cc, _BAR[bar][0], db, col["top_bars"], col["side_bars"],
                                    high_axial=high_axial, legs=legs)
                if cage_passes(trial):
                    candidates.append((legs, trial))
        priced = []
        for legs, trial in candidates:
            bounds = bounds_for(trial["hx_in"])
            per_direction = {}
            for axis, spec in directions.items():
                d = spec["depth_in"] - offset
                av = legs[spec["legs_key"]] * _BAR[bar][1]
                vs_req = worst[axis]["vs_required_kip"] if axis in worst else 0.0
                per_direction[axis] = {
                    "legs_key": spec["legs_key"], "legs": legs[spec["legs_key"]], "av_in2": av, "d_in": d,
                    "bc_in": spec["bc_in"], "vs_required_kip": vs_req,
                    "spacing_from_shear_in": av * fy * d / vs_req if vs_req > 0 else float("inf"),
                    "spacing_from_confinement_in": av / (ratio * spec["bc_in"]),
                    "ash_required_per_in": ratio * spec["bc_in"]}
            s_shear = min(v["spacing_from_shear_in"] for v in per_direction.values())
            s_conf = min(v["spacing_from_confinement_in"] for v in per_direction.values())
            spacing = _grid_down(min(s_shear, s_conf, *bounds.values()))
            if spacing is not None:
                priced.append((spacing, legs, trial, per_direction, bounds, s_shear, s_conf))
        if not priced:
            continue
        spacing, legs, trial, per_direction, bounds, s_shear, s_conf = max(
            priced, key=lambda item: (item[0], -sum(item[1].values()), -max(item[1].values())))
        for axis, v in per_direction.items():
            v["ash_provided_per_in"] = v["av_in2"] / spacing
            v["phi_vn_kip"] = PHI_SHEAR * ((worst[axis]["vc_kip"] if axis in worst else 0.0)
                                           + v["av_in2"] * fy * v["d_in"] / spacing)
        selected = {"bar_size": bar, "legs": legs, "spacing_in": spacing,
                    # Scalars for the single-value consumers (IMK rho_sh, legacy Av): the lighter direction.
                    "legs_model": min(legs.values()),
                    "av_in2": min(v["av_in2"] for v in per_direction.values()),
                    "spacing_from_shear_in": s_shear, "spacing_from_confinement_in": s_conf,
                    "ash_provided_per_in": min(v["ash_provided_per_in"] for v in per_direction.values()),
                    "ash_required_per_in": max(v["ash_required_per_in"] for v in per_direction.values()),
                    "phi_vn_kip": min(v["phi_vn_kip"] for v in per_direction.values()),
                    "crossties_per_direction": {k: n - 2 for k, n in legs.items()},
                    "by_direction": per_direction}
        cage = trial
        break
    if cage is None:
        # Nothing selected: report the arrangement bounds for the first hoop bar.
        first = column_cage(b, h, cc, _BAR[COLUMN_HOOP_BARS[0]][0], db, col["top_bars"], col["side_bars"],
                            high_axial=high_axial)
        hx = max(m["hx_in"] for m in first["minimal_support"].values())
        bounds = bounds_for(hx)
        cage = {**first, "legs": None, "constructible": False, "arrangement": None, "hx_in": hx,
                "checks": [{"rule": "18.7.5.2(b)/(d) / 18.7.5.3 / 18.7.5.4", "passes": False,
                            "detail": (f"no hoop of #{COLUMN_HOOP_BARS[0]}-#{COLUMN_HOOP_BARS[-1]} with legs in "
                                       f"{first['constructible_legs_by_direction']} reaches a spacing of at least "
                                       f"{SPACING_MIN_IN:g} in under the shear, confinement and spacing limits")}]}
    else:
        hx = cage["hx_in"]
    so = bounds["so"]
    hx_ok = (not high_axial) or hx <= 8.0
    # Provided-steel screen with the selected hoops: phi (Vc + min(Av fyt d / s,
    # Vs limit)) against the method's Ve, per story and direction.
    screen = column_shear_screen(stories, selected, vs_limit, fy)
    if method == COLUMN_SHEAR_METHOD_JOINT_LIMITED:
        method_basis = ("per direction: Ve = min(2 Mpr,col/ln over the factored axial range, joint-limited beam Mpr "
                        "delivery) >= Vu with Mpr about that direction's axis; Vc = 0 when mechanism shear >= Ve/2 "
                        "and Pu < Ag fc/20 (18.7.6.2.1); one hoop bar and spacing, legs per direction from Vs, "
                        "18.7.5.4 confinement (Ash perpendicular to each bc) and 18.7.5.3 spacing with every face bar tied")
    else:
        method_basis = ("per direction: Ve = max((Mpr,i + Mpr,j) / ln, Vu) with each end's probable moment the exact section "
                        "maximum over that end's saved factored axial range in the bending sense of the governing sway "
                        "(18.7.6.1.1; no beam-sharing reduction); Vc = 0 when Pu,min < Ag fc/20 (the mechanism shear is all "
                        "of Ve, 18.7.6.2.1); hoops as in the joint-limited method")
    return {"column_shear_method": method, "method_basis": method_basis,
            "clear_height_convention": convention if method == COLUMN_SHEAR_METHOD_COLUMN_OWN else CLEAR_HEIGHT_UNIFORM,
            "local_force_convention": LOCAL_FORCE_CONVENTION,
            "stories": stories, "governing": governing, "worst_by_direction": worst,
            "vs_limit_kip": vs_limit[governing["axis"]] if governing else min(vs_limit.values()),
            "vs_limit_by_direction_kip": vs_limit,
            "section_adequate": section_adequate,
            "cage": cage,
            "confinement": {"ach_in2": ach, "bc_in": max(b, h) - 2.0 * cc,
                            "bc_by_direction_in": {axis: spec["bc_in"] for axis, spec in directions.items()},
                            "ash_ratio_required": ratio, "high_axial": high_axial,
                            "hx_in": hx, "so_in": so, "hx_within_8in_when_required": hx_ok},
            "spacing_bounds_in": bounds, "hoops": selected, "joint_delivery_kip_in": joint_delivery,
            "screen": screen,
            "directions": {axis: {"depth_in": spec["depth_in"], "width_in": spec["width_in"], "legs_key": spec["legs_key"],
                                  "bc_in": spec["bc_in"], "layers_basis": spec["layers_basis"]}
                           for axis, spec in directions.items()},
            "basis": method_basis}


def column_shear_screen(stories, hoops, vs_limit, fy):
    """phi (Vc + min(Av fyt d / s, Vs limit)) with the given hoops against each story's Ve, per direction.

    ``hoops`` is a selected-hoop record ({bar_size, spacing_in, by_direction
    {axis: {av_in2, d_in}}}) or None (no screen). The ratio Ve / phi Vn is
    reported per story and direction; ``pass`` is ratio <= 1 for that
    comparison only.
    """
    if not hoops:
        return {"available": False, "reason": "no hoops selected"}
    rows = {}
    for axis, per_story in stories.items():
        direction = hoops["by_direction"][axis]
        vs_provided = direction["av_in2"] * fy * direction["d_in"] / hoops["spacing_in"]
        counted = min(vs_provided, vs_limit[axis])
        for story, entry in per_story.items():
            phi_vn = PHI_SHEAR * (entry["vc_kip"] + counted)
            rows.setdefault(axis, {})[story] = {
                "ve_kip": entry["ve_kip"], "vc_kip": entry["vc_kip"], "vs_provided_kip": vs_provided,
                "vs_limit_kip": vs_limit[axis], "vs_counted_kip": counted, "phi_vn_capped_kip": phi_vn,
                "ratio": entry["ve_kip"] / phi_vn if phi_vn > 0 else math.inf, "pass": entry["ve_kip"] <= phi_vn,
                "av_in2": direction["av_in2"], "d_in": direction["d_in"], "spacing_in": hoops["spacing_in"],
                "bar_size": hoops["bar_size"], "legs": direction["legs"]}
    return {"available": True, "rows": rows,
            "governing_ratio": max(r["ratio"] for per in rows.values() for r in per.values()),
            "all_pass": all(r["pass"] for per in rows.values() for r in per.values()),
            "basis": "phi (Vc + min(Av fyt d / s, 8 sqrt(f'c) b d)) with the selected hoops against the method's Ve"}


def joint_kinds(num_bay_x, num_bay_y):
    """The distinct joint connectivities of a uniform rectangular frame.

    Returns {kind: {axis: (family in that direction, beam count in that
    direction, perpendicular beam count)}}. A column on the y = 0 / y = Ly
    perimeter line (``edge_x``) carries two edge x-beams and one interior
    y-beam that terminates in it; a column on the x = 0 / x = Lx line
    (``edge_y``) the mirror. Both directions of both kinds are enumerated:
    the terminating interior-family beam and the pair of edge-family beams
    are different joints. Kinds that the plan does not contain are omitted.
    """
    kinds = {"corner": {"x": ("x_edge", 1, 1), "y": ("y_edge", 1, 1)}}
    if num_bay_x >= 2:
        kinds["edge_x"] = {"x": ("x_edge", 2, 1), "y": ("y_interior", 1, 2)}
    if num_bay_y >= 2:
        kinds["edge_y"] = {"x": ("x_interior", 1, 2), "y": ("y_edge", 2, 1)}
    if num_bay_x >= 2 and num_bay_y >= 2:
        kinds["interior"] = {"x": ("x_interior", 2, 2), "y": ("y_interior", 2, 2)}
    return kinds


def classify_joint_shear_inputs(level, beams_in_direction, transverse_beams, beam_width_in, transverse_face_width_in,
                                beam_depth_in, column_depth_in, clear_span_in, transverse_clear_span_in,
                                column_clear_height_in, beam_bars, beam_stirrup_bar_size, continuity):
    """The three inputs of ACI 318-19 Table 18.8.4.3, each with the evidence it rests on.

    ``continuity`` is the design's declaration of reinforcement continuity
    (Design_Driver._joint_continuity_declaration, saved under
    detailing.joint_continuity). None, or a missing or False item, leaves the
    corresponding input at the table's "Other"/"Not confined" value and lists
    what was not evaluated, so unknown detailing never reads as a compliant
    detail.

    Column (15.2.6): continuous at a floor joint when the column above
    extends at least the column depth h in the direction of joint shear
    ((a), clear height >= h) and the bars and hoops of the column below
    continue through the extension ((b), declared); "Other" at the roof,
    where no column extends above. Beam in the direction of Vu (15.2.7):
    continuous when beams frame into both faces of that direction, each
    extending at least the beam depth h beyond the face ((a), clear span
    >= h) with the bars and hoops of the beam on the opposite side
    continued through ((b), declared); "Other" where the beam terminates.
    Confinement (15.2.8): two transverse beams, each at least three-quarters
    of the width of the column face it frames into (a), extending at least
    h beyond the joint faces (b), with at least two continuous top and
    bottom bars and No. 3 or larger stirrups (c). The width test alone, or
    a single transverse beam, does not confine.
    """
    continuity = continuity if isinstance(continuity, dict) else {}
    unevaluated = []
    if level == "roof":
        column_state = "other"
        column_basis = "terminating column: no column extends above the roof joint (Table 18.8.4.3 Other)"
    else:
        extends = column_clear_height_in >= column_depth_in
        bars = continuity.get("column_reinforcement_continuous_through_floor_joints")
        if bars is None:
            unevaluated.append("column_reinforcement_continuous_through_floor_joints")
        column_state = "continuous" if extends and bars is True else "other"
        column_basis = (f"15.2.6: column above extends {column_clear_height_in:g} in >= h = {column_depth_in:g} in "
                        f"({'yes' if extends else 'no'}); bars and hoops of the column below continue through the joint: "
                        f"{'declared' if bars is True else 'not evaluated' if bars is None else 'no'}")
    if beams_in_direction >= 2:
        extends = clear_span_in >= beam_depth_in
        bars = continuity.get("beam_reinforcement_continuous_through_interior_joints")
        if bars is None:
            unevaluated.append("beam_reinforcement_continuous_through_interior_joints")
        beam_state = "continuous" if extends and bars is True else "other"
        beam_basis = (f"15.2.7: beams on both faces, each extending {clear_span_in:g} in >= h = {beam_depth_in:g} in "
                      f"({'yes' if extends else 'no'}); bars and hoops of the opposite beam continue through the joint: "
                      f"{'declared' if bars is True else 'not evaluated' if bars is None else 'no'}")
    else:
        beam_state = "other"
        beam_basis = "terminating beam: one beam in the direction of Vu (Table 18.8.4.3 Other)"
    width_ratio = beam_width_in / transverse_face_width_in
    width_ok = width_ratio >= CONFINING_BEAM_WIDTH_RATIO - 1e-12
    two_beams = transverse_beams >= 2
    transverse_extends = transverse_clear_span_in >= beam_depth_in
    bars_continuous = continuity.get("beam_reinforcement_continuous_through_interior_joints")
    bars_ok = (beam_bars is not None and min(beam_bars) >= CONFINING_BEAM_MIN_BARS and bars_continuous is True)
    stirrups_ok = beam_stirrup_bar_size is not None and beam_stirrup_bar_size >= CONFINING_BEAM_MIN_STIRRUP
    if two_beams and width_ok:
        if bars_continuous is None and "beam_reinforcement_continuous_through_interior_joints" not in unevaluated:
            unevaluated.append("beam_reinforcement_continuous_through_interior_joints")
        if beam_stirrup_bar_size is None:
            unevaluated.append("transverse_beam_stirrup_bar_size")
    confined = bool(two_beams and width_ok and transverse_extends and bars_ok and stirrups_ok)
    gamma = JOINT_SHEAR_GAMMA[(column_state, beam_state, confined)]
    return {"table": "ACI 318-19 Table 18.8.4.3",
            "column": {"state": column_state, "basis": column_basis},
            "beam": {"state": beam_state, "basis": beam_basis},
            "confinement": {"confined": confined, "transverse_beams": transverse_beams, "two_transverse_beams": two_beams,
                            "width_ratio": width_ratio, "width_ok": width_ok, "extends_beyond_joint": transverse_extends,
                            "continuous_top_and_bottom_bars": bars_ok, "stirrups_ok": stirrups_ok,
                            "basis": ("15.2.8: two transverse beams, each >= 3/4 of the column face it frames into, "
                                      "extending >= h beyond the joint, with >= 2 continuous top and bottom bars and "
                                      ">= No. 3 stirrups")},
            "gamma": gamma,
            "gamma_basis": (f"Table 18.8.4.3: column {column_state}, beam {beam_state}, "
                            f"{'confined' if confined else 'not confined'} -> gamma {gamma:g}"),
            "unevaluated_evidence": unevaluated, "evidence_complete": not unevaluated}


def design_joint_shear(state, strengths, column, beams=None):
    """Vj against phi Vn for every joint kind and direction, gamma from the Table 18.8.4.3 inputs.

    ``column`` is the column shear design (its hoops are the joint transverse
    reinforcement, 18.8.3.1); ``beams`` the beam shear design, whose hoops
    are the transverse beams' stirrups for 15.2.8(c). Continuity of the
    reinforcement through the joints comes from ``state['joint_continuity']``
    and is never inferred here.
    """
    sections, mats, geometry = state["sections"], state["materials"], state["geometry"]
    fy = mats["fy_ksi"]
    fc_joint = sections["fc_col_ksi"]
    bw, h_beam = sections["b_beam_in"], sections["h_beam_in"]
    story_h = geometry["story_h_in"]
    clear = {"x": geometry["bay_x_in"] - sections["h_col_in"], "y": geometry["bay_y_in"] - sections["b_col_in"]}
    beam_bars = (state["beam"]["top_bars"], state["beam"]["bot_bars"])
    beam_hoops = (beams or {}).get("hoops") if isinstance(beams, dict) else None
    stirrup = beam_hoops.get("bar_size") if isinstance(beam_hoops, dict) else None
    continuity = state.get("joint_continuity")
    results, evidence = {}, []
    kinds = joint_kinds(geometry["num_bay_x"], geometry["num_bay_y"])
    for level in ("floor", "roof"):
        topology = "terminating" if level == "roof" else "continuous"
        for kind, per_axis in kinds.items():
            for axis in ("x", "y"):
                # Column depth in the direction of joint shear and the width of
                # the face the beams in that direction frame into; the transverse
                # beams frame into the other faces, whose width is ``depth``.
                depth = sections["h_col_in"] if axis == "x" else sections["b_col_in"]
                width = sections["b_col_in"] if axis == "x" else sections["h_col_in"]
                family, count, perpendicular_count = per_axis[axis]
                fam = strengths[family]
                t_hog = 1.25 * fy * fam["tension_steel_hogging_in2"]
                t_sag = 1.25 * fy * fam["tension_steel_sagging_in2"]
                if count == 2:
                    forces = [t_hog, t_sag]
                    sum_mpr = fam["mpr_negative_kip_in"] + fam["mpr_positive_kip_in"]
                else:
                    forces = [max(t_hog, t_sag)]
                    sum_mpr = max(fam["mpr_negative_kip_in"], fam["mpr_positive_kip_in"])
                vcol = sum_mpr / story_h * (2.0 if level == "roof" else 1.0)
                vj = abs(sum(forces) - vcol)
                classification = classify_joint_shear_inputs(
                    level, count, perpendicular_count, bw, depth, h_beam, depth, clear[axis],
                    clear["y" if axis == "x" else "x"], story_h - h_beam, beam_bars, stirrup, continuity)
                gamma = classification["gamma"]
                # Faces with a beam at least 3/4 of their width: the 18.8.3.2
                # joint-hoop relaxation needs all four; not a Table 18.8.4.3 input.
                parallel = count if bw >= CONFINING_BEAM_WIDTH_RATIO * width else 0
                perpendicular = perpendicular_count if bw >= CONFINING_BEAM_WIDTH_RATIO * depth else 0
                aj = rectangular_joint_area(column_depth_in=depth, column_width_in=width,
                                            beam_width_in=bw, beam_center_offset_in=0.0)
                vn = gamma * math.sqrt(fc_joint * 1000.0) * aj / 1000.0
                entry = {"id": f"joint_shear/{level}/{kind}/{axis}", "level": level, "kind": kind, "axis": axis,
                         "beam_family": family, "beams_in_direction": count, "beams_perpendicular": perpendicular_count,
                         "beam_face_forces_kip": forces, "column_shear_kip": vcol,
                         "probable_face_forces_complete": True, "column_shear_consistent_with_mpr": True,
                         "nominal_vn_kip": vn, "capacity_basis": "ACI318-19_Table18.8.4.3",
                         "capacity_topology_and_confinement_checked": False,   # set once joint hoops are designed
                         "gamma": gamma, "gamma_basis": classification["gamma_basis"],
                         "classification": classification,
                         "column_continuity": topology,
                         "beam_continuity": "continuous" if count == 2 else "terminating",
                         "transverse_confinement": classification["confinement"]["confined"],
                         "evidence_complete": classification["evidence_complete"],
                         "unevaluated_evidence": classification["unevaluated_evidence"],
                         "confined_faces": parallel + perpendicular,
                         "aj_in2": aj, "vj_kip": vj, "phi_vn_kip": PHI_JOINT * vn,
                         "passes": vj <= PHI_JOINT * vn,
                         "slab_steel_in_tension_included": True}
                results[entry["id"]] = entry
                evidence.append(entry)
    # Joint transverse reinforcement (18.8.3.1): the column end-zone hoops
    # continue through the joint at their spacing and area (18.7.5.2-.4).
    # The 18.8.3.2 relaxation (Ash halved, spacing to 6 in within the
    # shallowest beam) needs beams on all four sides each at least 3/4 of
    # the column width; it is recorded per joint and applied nowhere, one
    # hoop specification serving every joint of the frame.
    hoops = column.get("hoops") if column else None
    relaxation_by_joint = {joint_id: v["confined_faces"] == 4 for joint_id, v in results.items()}
    joint_transverse = {
        "hoops": hoops, "basis": "18.8.3.1: column hoops per 18.7.5.2--18.7.5.4 continue through the joint depth",
        "relaxation_18_8_3_2_by_joint": relaxation_by_joint,
        "relaxation_18_8_3_2_applicable": all(relaxation_by_joint.values()),
        "relaxation_18_8_3_2_applied": False,
        "relaxation_basis": ("18.8.3.2: beams on all four sides, each >= 3/4 of the column width; distinct from the "
                             "Table 18.8.4.3 confinement input (15.2.8)"),
        "designed": hoops is not None,
        "column_shear_basis": ("Vcol = sum of beam Mpr at the joint divided by the story height, columns above and "
                               "below sharing equally (inflection at mid-height); at a terminating roof column the "
                               "single column takes the whole sum, so Vcol = 2 sum Mpr / H"),
    }
    for entry in evidence:
        # The downstream check (SMRF_Joints.joint_shear_check) accepts a
        # nominal strength only when its topology, continuity, confinement and
        # transverse steel were checked: joint hoops designed AND every
        # Table 18.8.4.3 input on supplied evidence. A conservative gamma on
        # missing evidence is arithmetic, not a verified classification.
        entry["capacity_topology_and_confinement_checked"] = bool(joint_transverse["designed"] and entry["evidence_complete"])
        entry["joint_transverse_reinforcement"] = joint_transverse["basis"]
        entry["column_shear_basis"] = joint_transverse["column_shear_basis"]
    return {"joints": results, "evidence": evidence, "all_pass": all(v["passes"] for v in results.values()),
            "evidence_complete": all(v["evidence_complete"] for v in results.values()),
            "joint_transverse": joint_transverse,
            "continuity_declaration": continuity,
            "basis": ("Vj = 1.25 fy (As,top + slab bars in flange) + 1.25 fy As,bot - Vcol; Vcol = sum Mpr / H "
                      "(2 sum Mpr / H at a terminating roof column); gamma from Table 18.8.4.3 by column continuity "
                      "(15.2.6), beam continuity in the direction of Vj (15.2.7) and confinement by two transverse "
                      "beams (15.2.8), each input on declared evidence; Aj per 15.4.2.4; phi = 0.85")}


def design_anchorage(state):
    """Terminating beam bars at exterior joints (18.8.5.1) and through bars (18.8.2.3).

    Through bars exist only in a direction with an interior joint (two or
    more bays); a one-bay direction terminates every bar with a hook.
    """
    sections, beam, mats, col = state["sections"], state["beam"], state["materials"], state["column"]
    geometry = state.get("geometry") or {}
    fy, fc = mats["fy_ksi"], sections["fc_col_ksi"]
    db = _BAR[beam["bar_size"]][0]
    ldh = max(fy * 1000.0 * db / (65.0 * math.sqrt(fc * 1000.0)), 8.0 * db, 6.0)
    hoop_db = _BAR[col["stirrup_bar_size"]][0]
    results = {}
    for axis in ("x", "y"):
        depth = sections["h_col_in"] if axis == "x" else sections["b_col_in"]
        bays = geometry.get("num_bay_x" if axis == "x" else "num_bay_y")
        through = True if bays is None else bays >= 2
        available = depth - col["clear_cover_in"] - hoop_db
        results[axis] = {"ldh_required_in": ldh, "embedment_available_in": available,
                         "hook": "standard 90-degree hook into the confined core, 18.8.5.1",
                         "passes": ldh <= available,
                         "through_bars_present": through,
                         "through_bar_depth_required_in": 20.0 * db, "through_bar_depth_available_in": depth,
                         "through_bar_passes": (20.0 * db <= depth) if through else True}
    return {"directions": results, "all_pass": all(v["passes"] and v["through_bar_passes"] for v in results.values()),
            "basis": ("ldh = fy db / (65 lambda sqrt(fc)) >= max(8db, 6 in); embedment = column depth - cover - hoop; "
                      "through bars need a joint depth of 20 db (normalweight, 18.8.2.3) where an interior joint exists")}


def straight_development_length(bar, fc_ksi, fy_ksi, top_bar=False):
    """ACI 318-19 25.4.2.4 straight development, uncoated Grade 60, normalweight.

    Clear spacing >= db with cover >= db and hoops throughout (or spacing >=
    2db and cover >= db) selects the /25 (#6 and smaller) or /20 (#7 and
    larger) form; psi_t = 1.3 for bars with more than 12 in of fresh
    concrete below them.
    """
    db = _BAR[bar][0]
    divisor = 25.0 if bar <= 6 else 20.0
    psi_t = 1.3 if top_bar else 1.0
    return max(12.0, fy_ksi * 1000.0 * psi_t / (divisor * math.sqrt(fc_ksi * 1000.0)) * db)


def design_splices(state):
    """Where and how longitudinal bars can be spliced (18.6.3.3, 18.7.4.3, 25.5, 18.2.7).

    Beams: Class B lap (1.3 ld) only outside the 2h hinge zones and enclosed
    by hoops at <= min(d/4, 4 in); when the middle of the clear span cannot
    hold 1.3 ld, Type 2 mechanical splices (18.2.7.1, permitted anywhere)
    are the design. Columns: Class B lap within the center half of the
    clear height, enclosed by the end-zone hoops; otherwise mechanical.
    """
    sections, beam, col, mats, geometry = state["sections"], state["beam"], state["column"], state["materials"], state["geometry"]
    fy = mats["fy_ksi"]
    result = {}
    # Beams: worst (shortest) clear span in either direction.
    clear = min(geometry["bay_x_in"] - sections["h_col_in"], geometry["bay_y_in"] - sections["b_col_in"])
    h = sections["h_beam_in"]
    lap_top = 1.3 * straight_development_length(beam["bar_size"], sections["fc_beam_ksi"], fy, top_bar=h - beam["centroid_offset_in"] > 12.0)
    lap_bot = 1.3 * straight_development_length(beam["bar_size"], sections["fc_beam_ksi"], fy, top_bar=False)
    available = clear - 2.0 * (2.0 * h)
    beam_lap_fits = max(lap_top, lap_bot) <= available
    result["beam"] = {"class_b_lap_top_in": lap_top, "class_b_lap_bottom_in": lap_bot,
                      "available_between_hinge_zones_in": available, "lap_splice_feasible": beam_lap_fits,
                      "splice_type": "class_B_lap_outside_hinge_zones" if beam_lap_fits else "type_2_mechanical_18.2.7",
                      "hoops_over_lap_max_spacing_in": min((h - beam["centroid_offset_in"]) / 4.0, 4.0),
                      "basis": "18.6.3.3 (no laps within 2h of the face or where yielding is expected); 25.5.2.1 Class B"}
    ln = geometry["story_h_in"] - h
    lap_col = 1.3 * straight_development_length(col["bar_size"], sections["fc_col_ksi"], fy, top_bar=False)
    center_half = ln / 2.0
    # 25.5.1.1: lap splices are not permitted for bars larger than No. 11 (No. 14 and No. 18 column bars,
    # offered since 2026-09-27 by the column cage rule), so those bars take Type 2 mechanical splices.
    lap_permitted = int(col["bar_size"]) <= 11
    col_lap_fits = lap_permitted and lap_col <= center_half
    result["column"] = {"class_b_lap_in": lap_col, "center_half_clear_height_in": center_half,
                        "lap_splice_permitted_25.5.1.1": lap_permitted,
                        "lap_splice_feasible": col_lap_fits,
                        "splice_type": "class_B_lap_center_half" if col_lap_fits else "type_2_mechanical_18.2.7",
                        "basis": ("18.7.4.3 (laps only in the center half, tension laps, hoops per 18.7.5.2/.3); 25.5.2.1 Class B; "
                                  "25.5.1.1 (no lap splices above No. 11: Type 2 mechanical, 18.2.7.1)")}
    result["all_designed"] = True
    return result


def design_bar_threading(state):
    """The beam bars pass between the column bars within the layer limit, and the stacked layers clear the slab mats.

    Design-basis rule of 2026-09-27 (Design/SMRF_Cage_Geometry.joint_assembly): a candidate whose
    beam bars cannot thread the column with the 25.2.1 clearance in at most ``beam["max_layers"]``
    layers (one at first; two since the same day's decision, the second layer directly below the
    first and priced in the strengths) is not accepted, and the search grows the column where a
    lane-keeping cage exists or widens the beam. The stacking convention is the one the state carries
    (Structure_Parameters.beam_bar_stacking_convention); a legacy single elevation is evaluated as
    x_over_y for the geometry only.
    """
    from Design.SMRF_Cage_Geometry import joint_assembly
    beam, column, sections = state["beam"], state["column"], state["sections"]
    slab = state.get("slab") or {}
    max_layers = int(beam.get("max_layers", 1))
    layer_order = beam.get("layer_order") or "blocked"
    record = {
        "sections": sections,
        "reinforcement": {
            "col_bar_size": column["bar_size"], "col_top_bars": column["top_bars"], "col_side_bars": column["side_bars"],
            "col_stirrup_bar_size": column["stirrup_bar_size"], "col_clear_cover_in": column["clear_cover_in"],
            "beam_bar_size": beam["bar_size"], "beam_top_bars": beam["top_bars"], "beam_bot_bars": beam["bot_bars"],
            "beam_stirrup_bar_size": beam.get("stirrup_bar_size", STIRRUP_LADDER[0][0]),
            "beam_clear_cover_in": beam["clear_cover_in"],
            "beam_longitudinal_centroid_offset_in": beam["centroid_offset_in"],
            "beam_bar_area_in2": _BAR[beam["bar_size"]][1],
            "beam_bar_stacking": {"max_layers": max_layers, "layer_order": layer_order,
                                  "convention": beam.get("stacking_convention")}},
        "materials": {"fy_ksi": state["materials"]["fy_ksi"],
                      "aggregate_size_in": state["materials"].get("aggregate_size_in", 0.75)},
        "slab": {"thickness_in": slab.get("thickness_in") or 0.0},
        "slab_reinforcement": {"layout": slab.get("layout")} if slab.get("layout") else {},
        "capacity_design": {},
    }
    convention = beam.get("stacking_convention") or "x_over_y"
    assembly = joint_assembly(record, stacking=convention if convention in ("x_over_y", "y_over_x") else "x_over_y",
                              max_layers=max_layers, layer_order=layer_order)
    directions = {axis: {key: value for key, value in d.items() if key != "nominal_positions_in_conflict"}
                  for axis, d in assembly["directions"].items()}
    return {"passes": all(d["fits_within_layer_limit"] for d in directions.values()),
            "max_layers": max_layers, "layer_order": layer_order, "layers": assembly["stacking"]["layers"],
            "by_direction": directions,
            "stacking": assembly["stacking"], "stacking_convention": convention,
            "slab_clashes": assembly["slab_clashes"], "slab_tight_crossings": assembly["slab_tight_crossings"],
            "slab_mats_placed": assembly["slab_mats_placed"],
            "stacking_clear_of_mats": not assembly["slab_clashes"],
            "basis": ("beam bars threaded between the column bars in plan with the 25.2.1 clearance, one layer "
                      "(Design/SMRF_Cage_Geometry); orthogonal top and bottom layers stacked at the joint, the "
                      "lower direction's offsets carried into its strengths")}


def build_capacity_design(state):
    """All capacity-design evidence for the current state, plus its checks."""
    strengths = {f"{axis}_{position}": probable_beam_strengths(state, position, axis)
                 for axis in ("x", "y") for position in ("edge", "interior")}
    beams = design_beam_shear(state, strengths, state.get("transfer"))
    columns = design_column_shear(state, strengths)
    joints = design_joint_shear(state, strengths, columns, beams)
    anchorage = design_anchorage(state)
    splices = design_splices(state)
    threading = design_bar_threading(state)
    beams["bars_thread_column"] = threading
    checks = []
    checks.append(make_check("detailing.splices_designed", "ACI 318-19 18.6.3.3 / 18.7.4.3 / 18.2.7 / 25.5",
                             1, 1, "==", details={"beam": splices["beam"], "column": splices["column"]}))
    checks.append(make_check("beam.bars_thread_column",
                             f"ACI 318-19 25.2.1 clearance; at most {threading['max_layers']} layer(s) between the column bars (25.2.2)",
                             int(threading["passes"]), 1, "==",
                             details={**{axis: {"bars_per_layer_that_fit": d["bars_per_layer_that_fit"],
                                                "layers_needed": d["layers_needed"],
                                                "layers_placed": threading["layers"].get(axis),
                                                "column_bar_lanes_blocked_in": d["column_bar_lanes_blocked_in"]}
                                         for axis, d in threading["by_direction"].items()},
                                      "max_layers": threading["max_layers"]}))
    checks.append(make_check("beam.bar_stacking_clear_of_slab_mats", "ACI 318-19 25.2.2; orthogonal layers at the joint",
                             int(threading["stacking_clear_of_mats"]), 1, "==",
                             details={"convention": threading["stacking_convention"],
                                      "lower_direction": threading["stacking"]["lower_direction"],
                                      "lower_layer_centroid_from_face_in": threading["stacking"]["lower_layer_centroid_from_face_in"],
                                      "overlaps": threading["slab_clashes"],
                                      "tight_crossings": len(threading["slab_tight_crossings"]),
                                      "slab_mats_placed": threading["slab_mats_placed"]}))
    checks.append(make_check("beam.capacity_shear_section", "ACI 318-19 22.5.1.2 with 18.6.5.1 Ve",
                             beams["vs_required_kip"], beams["vs_limit_kip"], "<=", "kip"))
    checks.append(make_check("beam.hoops_selected", "ACI 318-19 18.6.4.4 / 18.6.5", int(beams["hoops"] is not None), 1, "=="))
    checks.append(make_check("column.capacity_shear_section", "ACI 318-19 22.5.1.2 with 18.7.6.1.1 Ve",
                             columns["governing"]["vs_required_kip"] if columns["governing"] else 0.0,
                             columns["vs_limit_kip"], "<=", "kip",
                             details={"column_shear_method": columns["column_shear_method"],
                                      "clear_height_convention": columns["clear_height_convention"]}))
    checks.append(make_check("column.hoops_selected", "ACI 318-19 18.7.5.3 / 18.7.5.4 / 18.7.6",
                             int(columns["hoops"] is not None), 1, "=="))
    checks.append(make_check("column.supported_bar_spacing", "ACI 318-19 18.7.5.2",
                             int(columns["confinement"]["hx_within_8in_when_required"]), 1, "=="))
    from Design.SMRF_Cage_Layout import cage_passes
    for member, data in (("column", columns), ("beam", beams)):
        cage = data.get("cage") or {}
        failing = [c for c in cage.get("checks", []) if not c.get("passes")]
        checks.append(make_check(f"{member}.cage_layout",
                                 "ACI 318-19 18.7.5.2(b)-(f) / 25.7.2.3" if member == "column" else "ACI 318-19 18.6.4.4 / 25.7.2.3",
                                 int(cage_passes(cage)), 1, "==",
                                 details={"legs": cage.get("legs"), "constructible_legs": cage.get("constructible_legs"),
                                          "legs_min": cage.get("legs_min"), "legs_max": cage.get("legs_max"),
                                          "hx_in": cage.get("hx_in"), "failing": failing,
                                          "scope": "perimeter hoop plus crossties engaging bars; supported-bar spacing, "
                                                   "alternate-bar support and the 6-in clear rule from the arrangement; "
                                                   "hook geometry and placement not drawn"}))
    for entry in joints["evidence"]:
        # The numeric screen uses the conservative rows where evidence is
        # missing; the evidence item says whether the classification is
        # supported, and stays open (not failed) when it is not.
        checks.append(make_check("joint.shear_screen", "ACI 318-19 18.8.4", entry["vj_kip"], entry["phi_vn_kip"],
                                 "<=", "kip", entry["id"]))
        if entry["evidence_complete"]:
            checks.append(make_check("joint.classification_evidence", "ACI 318-19 Table 18.8.4.3; 15.2.6-15.2.8",
                                     1, 1, "==", location=entry["id"],
                                     details={"gamma_basis": entry["gamma_basis"],
                                              "continuity_declaration_present": isinstance(state.get("joint_continuity"), dict),
                                              "scope": "declared detailing intent; the drawn cage is independent verification"}))
        else:
            checks.append(not_evaluated("joint.classification_evidence", "ACI 318-19 Table 18.8.4.3; 15.2.6-15.2.8",
                                        f"Table 18.8.4.3 inputs rest on evidence not supplied "
                                        f"({', '.join(entry['unevaluated_evidence'])}); gamma {entry['gamma']:g} is the "
                                        f"conservative row, not a verified classification", entry["id"]))
    for axis, item in anchorage["directions"].items():
        checks.append(make_check("joint.terminating_bar_hook", "ACI 318-19 18.8.5.1", item["ldh_required_in"],
                                 item["embedment_available_in"], "<=", "in", f"exterior/{axis}"))
        if item["through_bars_present"]:
            checks.append(make_check("joint.through_bar_depth", "ACI 318-19 18.8.2.3",
                                     item["through_bar_depth_required_in"], item["through_bar_depth_available_in"],
                                     "<=", "in", f"interior/{axis}"))
    beam_evidence = [data for entries in beams["families"].values() for data in entries]
    return {"method_version": METHOD_VERSION, "column_shear_method": columns["column_shear_method"],
            "beam_strengths": strengths, "beams": beams, "bar_threading": threading,
            "columns": columns, "joints": joints, "anchorage": anchorage, "splices": splices,
            "transverse": {"beam": beams["hoops"], "column": columns["hoops"]},
            "joint_evidence": {"beam_capacity_shear": beam_evidence, "joint_shear": joints["evidence"]},
            "checks": checks,
            "accepted": all(c["status"] == "pass" for c in checks)}
