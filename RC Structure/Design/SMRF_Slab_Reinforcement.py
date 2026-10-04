"""Bounded slab strip strength and reinforcement design, with no invented loads.

This module accepts an independently verified OUT-OF-PLANE slab action envelope;
the project's rigid-diaphragm frame analysis is not such an envelope. It selects
one bar size and four uniform building-wide spacings (x/y, top/bottom), using
the worst per-unit-width demand in every supplied panel and floor. The strip
screen is not a complete slab design: punching, development, corner torsion,
integrity, fire resistance, and serviceability remain explicit open checks.

Basis: ACI 318-19 8.3.3.1, 8.6.1.1, 8.7.2.2, Table 20.5.1.3.1,
21.2.2, 22.2, Table 22.5.5.1(c), 22.5.5.1.1/.3, and 25.2.
Original primary text:
https://www.ocf.berkeley.edu/~chiep/wp-content/uploads/2024/01/CE-123-ACI-318-19.pdf

Input contract is documented on ``design_slab_reinforcement``. Verification
flags record the upstream analyst's assertions; they do not verify an analysis
by themselves. No code in this module creates those assertions for the caller.
"""
from __future__ import annotations

import hashlib
import json
import math

from .SMRF_Common import make_check, not_evaluated, summarize_checks


METHOD_VERSION = "aci318_19_verified_slab_strip_strength_v2_completion_checks"
_BARS = {4: (0.5, 0.20), 5: (0.625, 0.31), 6: (0.75, 0.44)}
_DEFAULT_POLICY = {"bar_sizes": [4, 5, 6], "spacing_options_in": list(range(3, 13)),
                   "clear_cover_in": 0.75, "outer_axis": "x"}
_INPUT_KEYS = {"thickness_in", "fc_ksi", "fy_ksi", "max_aggregate_size_in",
               "exposure", "steel_specification", "concrete_type", "panel_ids", "num_floor"}
_VERIFIED_FLAGS = ("verified", "analysis_applicability_verified", "all_floors_enveloped",
                   "load_pattern_envelope_verified", "spatial_envelope_per_unit_width",
                   "twisting_moment_resolution_verified", "zero_membrane_force_verified")


def _number(value, name, zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return value


def _inputs(inputs):
    if not isinstance(inputs, dict) or set(inputs) != _INPUT_KEYS:
        raise ValueError(f"Slab inputs must contain exactly {sorted(_INPUT_KEYS)}.")
    result = dict(inputs)
    for key in ("thickness_in", "fc_ksi", "fy_ksi", "max_aggregate_size_in", "num_floor"):
        result[key] = _number(inputs[key], key)
    if not result["num_floor"].is_integer():
        raise ValueError("num_floor must be an integer.")
    result["num_floor"] = int(result["num_floor"])
    if result["fy_ksi"] != 60 or not 2.5 <= result["fc_ksi"] <= 10:
        raise ValueError("Only Grade 60 and 2.5 <= fc <= 10 ksi are supported.")
    if (result["exposure"] != "sheltered_interior"
            or result["steel_specification"] != "ASTM A706"
            or result["concrete_type"] != "normalweight"):
        raise ValueError("Only sheltered interior, ASTM A706, normalweight slabs are supported.")
    panels = result["panel_ids"]
    if (not isinstance(panels, list) or not panels
            or any(not isinstance(p, str) or not p.strip() for p in panels)
            or len(set(panels)) != len(panels)):
        raise ValueError("panel_ids must be a nonempty list of unique nonblank strings.")
    result["panel_ids"] = sorted(panels)
    return result


def slab_input_signature(inputs):
    """Bind upstream action evidence to this thickness/material/panel inventory."""
    normalized = _inputs(inputs)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _policy(policy):
    if policy is None:
        policy = {}
    if not isinstance(policy, dict) or set(policy) - set(_DEFAULT_POLICY):
        raise ValueError("Unsupported slab reinforcement policy fields.")
    result = {**_DEFAULT_POLICY, **policy}
    bars, spacings = result["bar_sizes"], result["spacing_options_in"]
    if (not isinstance(bars, list) or not bars
            or any(type(b) is not int or b not in _BARS for b in bars)
            or len(set(bars)) != len(bars)):
        raise ValueError("bar_sizes must be a unique nonempty subset of [4, 5, 6].")
    if not isinstance(spacings, list) or not spacings or len(spacings) > 1000:
        raise ValueError("spacing_options_in must contain 1-1000 positive spacings.")
    result["spacing_options_in"] = sorted({_number(s, "spacing") for s in spacings}, reverse=True)
    result["bar_sizes"] = sorted(bars)
    result["clear_cover_in"] = _number(result["clear_cover_in"], "clear_cover_in")
    if result["clear_cover_in"] < 0.75:
        raise ValueError("Slab cover must be at least 0.75 in for this exposure/bar scope.")
    if result["outer_axis"] not in ("x", "y"):
        raise ValueError("outer_axis must be x or y.")
    return result


def _demands(evidence, inputs):
    if not isinstance(evidence, dict):
        raise ValueError("Verified slab action evidence is missing.")
    if any(evidence.get(flag) is not True for flag in _VERIFIED_FLAGS):
        missing = [flag for flag in _VERIFIED_FLAGS if evidence.get(flag) is not True]
        raise ValueError(f"Upstream slab analysis verification is incomplete: {missing}.")
    if "engineering_assertions" in evidence:
        from Design.SMRF_Common import assertion_provenance_valid
        assertions = evidence["engineering_assertions"]
        if (not assertion_provenance_valid(assertions)
                or any(assertions.get(flag) is not True for flag in _VERIFIED_FLAGS)):
            raise ValueError("Generated slab actions require explicit, named, dated engineering assertions.")
    for key in ("method", "source", "load_combination_basis", "analysis_model_sha256"):
        if not isinstance(evidence.get(key), str) or not evidence[key].strip():
            raise ValueError(f"Slab evidence requires {key} provenance.")
    if evidence["source"].startswith("Design/SMRF_Slab_Actions.py "):
        from Design.SMRF_Slab_Refinement import refinement_verified
        if not refinement_verified(evidence):
            raise ValueError("Generated slab actions require completed, identity-matched mesh refinement.")
    fingerprint = evidence["analysis_model_sha256"]
    if len(fingerprint) != 64 or any(c not in "0123456789abcdefABCDEF" for c in fingerprint):
        raise ValueError("analysis_model_sha256 must identify the upstream analysis model.")
    if evidence.get("method") not in ("plate_finite_element", "equivalent_frame", "direct_design"):
        raise ValueError("An applicable out-of-plane plate/equivalent-frame/direct-design analysis is required.")
    if evidence.get("slab_input_sha256") != slab_input_signature(inputs):
        raise ValueError("Slab demand evidence belongs to different slab inputs.")
    if evidence.get("shear_envelope_basis") != "support_face_maximum":
        raise ValueError("Use maximum support-face shears; candidate-dependent d-section reductions are not accepted.")
    if evidence.get("load_scope") != "uniform_area_gravity_no_concentrated_loads":
        raise ValueError("Only uniform area gravity actions without concentrated loads are supported.")
    records = evidence.get("strips")
    if not isinstance(records, list):
        raise ValueError("Slab strip action inventory is missing.")
    expected = {(panel, axis, face) for panel in inputs["panel_ids"]
                for axis in ("x", "y") for face in ("top", "bottom")}
    result = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Every slab strip must be a dictionary.")
        key = (row.get("panel_id"), row.get("axis"), row.get("face"))
        if any(not isinstance(item, str) for item in key) or key not in expected or key in result:
            raise ValueError("Unexpected or duplicate panel/axis/face in slab strip inventory.")
        if not isinstance(row.get("demand_location"), str) or not row["demand_location"].strip():
            raise ValueError("Each strip must identify the source demand location.")
        if row.get("tension_face_at_shear_verified") is not True:
            raise ValueError("Each shear envelope must identify its longitudinal tension face.")
        result[key] = {**row,
                       "mu_kip_in_per_ft": _number(row.get("mu_kip_in_per_ft"), "Mu", True),
                       "vu_kip_per_ft": _number(row.get("vu_kip_per_ft"), "Vu", True)}
    if set(result) != expected:
        raise ValueError("Provide every panel, both axes, and both faces; zero demand must be explicit.")
    return [result[key] for key in sorted(result)]


def rectangular_strip_strength(fc_ksi, fy_ksi, effective_depth_in, area_in2_per_ft):
    """Strain-compatible singly reinforced 12-in strip, no membrane axial force.

    Opposite-face reinforcement is ignored, and no compression-steel credit is
    taken. Returned phi varies with actual steel strain; slab acceptance still
    separately requires a tension-controlled section, not merely sufficient Mn.
    """
    fc, fy, d, area = [_number(v, n) for v, n in zip(
        (fc_ksi, fy_ksi, effective_depth_in, area_in2_per_ft), ("fc", "fy", "d", "As"))]
    if fy != 60 or not 2.5 <= fc <= 10:
        raise ValueError("This section routine supports Grade 60 and fc 2.5-10 ksi.")
    beta = max(0.65, 0.85 - 0.05 * max(0.0, fc - 4.0))
    low, high = 0.0, d
    for _ in range(80):
        c = (low + high) / 2.0
        strain = 0.003 * (d - c) / c
        steel_stress = min(fy, 29000.0 * strain)
        if 0.85 * fc * 12.0 * beta * c < area * steel_stress:
            low = c
        else:
            high = c
    c = (low + high) / 2.0
    a = beta * c
    strain = 0.003 * (d - c) / c
    yield_strain = fy / 29000.0
    phi = 0.65 + 0.25 * min(1.0, max(0.0, (strain - yield_strain) / 0.003))
    moment = 0.85 * fc * 12.0 * a * (d - a / 2.0)
    return {"beta1": beta, "neutral_axis_in": c, "stress_block_in": a,
            "steel_stress_ksi": min(fy, 29000.0 * strain), "tension_strain": strain,
            "minimum_tension_controlled_strain": yield_strain + 0.003,
            "tension_controlled": strain >= yield_strain + 0.003,
            "phi": phi, "mn_kip_in_per_ft": moment, "phi_mn_kip_in_per_ft": phi * moment}


def one_way_shear_strength(fc_ksi, effective_depth_in, area_in2_per_ft):
    """ACI 318-19 Table 22.5.5.1(c): normalweight, Nu=0, no shear steel.

    As must be longitudinal tension reinforcement at the checked section; the
    caller checks layer location and development separately. No 2sqrt(fc) floor
    is added. The result is capped by 22.5.5.1.1 and uses phi=0.75.
    """
    fc, d, area = [_number(v, n) for v, n in zip(
        (fc_ksi, effective_depth_in, area_in2_per_ft), ("fc", "d", "As"))]
    if not 2.5 <= fc <= 10:
        raise ValueError("This shear routine supports fc 2.5-10 ksi.")
    size_factor = min(1.0, math.sqrt(2.0 / (1.0 + d / 10.0)))
    rho = area / (12.0 * d)
    root_fc_psi = math.sqrt(fc * 1000.0)
    vc_psi = min(8.0 * size_factor * rho ** (1.0 / 3.0), 5.0) * root_fc_psi
    vc = vc_psi * 12.0 * d / 1000.0
    return {"rho_longitudinal": rho, "lambda_s": size_factor, "lambda": 1.0,
            "vc_psi": vc_psi, "vc_kip_per_ft": vc, "phi": 0.75,
            "phi_vc_kip_per_ft": 0.75 * vc}


def _layout(inputs, policy, bar, axis, spacing):
    diameter, area = _BARS[bar]
    offset = 0.5 * diameter + (0.0 if axis == policy["outer_axis"] else diameter)
    depth = inputs["thickness_in"] - policy["clear_cover_in"] - offset
    if depth <= 0:
        return None
    steel_area = area * 12.0 / spacing
    return {"bar_size": bar, "bar_diameter_in": diameter, "bar_area_in2": area,
            "spacing_in": spacing, "clear_cover_outer_mat_in": policy["clear_cover_in"],
            "layer": "outer" if axis == policy["outer_axis"] else "inner",
            "axis": axis, "effective_depth_in": depth, "area_in2_per_ft": steel_area,
            "minimum_area_in2_per_ft": 0.0018 * 12.0 * inputs["thickness_in"],
            "horizontal_clear_spacing_in": spacing - diameter,
            "minimum_clear_spacing_in": max(1.0, diameter, 4.0 / 3.0 * inputs["max_aggregate_size_in"]),
            "maximum_spacing_in": min(2.0 * inputs["thickness_in"], 18.0),
            "flexure": rectangular_strip_strength(inputs["fc_ksi"], inputs["fy_ksi"], depth, steel_area),
            "one_way_shear": one_way_shear_strength(inputs["fc_ksi"], depth, steel_area)}


def _layer_checks(layout, demand, inputs, location):
    return [make_check("slab_strip_minimum_flexural_steel", "ACI 318-19 8.6.1.1 (8.6.1.2 still open)",
                       layout["area_in2_per_ft"], layout["minimum_area_in2_per_ft"], ">=", "in2/ft", location),
            make_check("slab_strip_maximum_spacing", "ACI 318-19 8.7.2.2, critical sections",
                       layout["spacing_in"], layout["maximum_spacing_in"], units="in", location=location),
            make_check("slab_strip_horizontal_clear_spacing", "ACI 318-19 25.2.1",
                       layout["horizontal_clear_spacing_in"], layout["minimum_clear_spacing_in"], ">=", "in", location),
            make_check("slab_strip_tension_controlled", "ACI 318-19 8.3.3.1 / Table 21.2.2",
                       layout["flexure"]["tension_strain"], layout["flexure"]["minimum_tension_controlled_strain"],
                       ">=", "strain", location),
            make_check("slab_strip_flexure", "ACI 318-19 22.2 / 21.2.2",
                       demand["mu_kip_in_per_ft"], layout["flexure"]["phi_mn_kip_in_per_ft"],
                       units="kip-in/ft", location=location),
            make_check("slab_strip_shear_tension_layer", "ACI 318-19 R22.5.5.1, longitudinal As location",
                       layout["effective_depth_in"], 2.0 / 3.0 * inputs["thickness_in"], ">=", "in", location),
            make_check("slab_strip_one_way_shear", "ACI 318-19 Table 22.5.5.1(c), 22.5.5.1.1/.3",
                       demand["vu_kip_per_ft"], layout["one_way_shear"]["phi_vc_kip_per_ft"],
                       units="kip/ft", location=location)]


_CONTEXT_KEYS = {"clear_span_x_in", "clear_span_y_in", "beam_width_in", "alpha_f_min", "alpha_f_l2_l1_min",
                 "thickness_screen_passed", "column_core_width_in", "two_way_shear_path_assessed",
                 "columns_at_beam_intersections", "beam_clear_cover_in", "beam_hoop_diameter_in", "fc_beam_ksi"}


def _context(context):
    """Floor context for the completion checks; None keeps them not_evaluated."""
    if context is None:
        return None
    if not isinstance(context, dict) or set(context) != _CONTEXT_KEYS:
        raise ValueError(f"Slab context must contain exactly {sorted(_CONTEXT_KEYS)}.")
    result = dict(context)
    for key in ("clear_span_x_in", "clear_span_y_in", "beam_width_in", "column_core_width_in",
                "beam_clear_cover_in", "beam_hoop_diameter_in", "fc_beam_ksi"):
        result[key] = _number(context[key], key)
    result["alpha_f_min"] = _number(context["alpha_f_min"], "alpha_f_min", True)
    result["alpha_f_l2_l1_min"] = _number(context["alpha_f_l2_l1_min"], "alpha_f_l2_l1_min", True)
    for key in ("thickness_screen_passed", "two_way_shear_path_assessed", "columns_at_beam_intersections"):
        if context[key] is not True and context[key] is not False:
            raise ValueError(f"{key} must be a boolean.")
    return result


def development_length_in(bar, fc_ksi, fy_ksi):
    """ACI 318-19 25.4.2.4 straight development for #6 and smaller, slab mats.

    Clear spacing >= 2db and clear cover >= db hold for every allowed layout
    (spacing >= 3 in, cover >= 0.75 in), so the (fy psi_t psi_e psi_g)/(25 lambda
    sqrt(fc)) form applies. Slab bars have less than 12 in of fresh concrete
    below them, so psi_t = 1.0; uncoated A706 Grade 60: psi_e = psi_g = 1.0;
    normalweight: lambda = 1.0. 25.4.2.1 floor of 12 in.
    """
    diameter, _ = _BARS[bar]
    ld = fy_ksi * 1000.0 * 1.0 * 1.0 * 1.0 / (25.0 * 1.0 * math.sqrt(fc_ksi * 1000.0)) * diameter
    return max(12.0, ld)


def _completion_checks(inputs, layout, demands, context):
    """Checks beyond the strip section that a uniform continuous mat can settle.

    Continuous uniform top and bottom mats through every support satisfy the
    extension/continuity provisions of 8.7.4.1 by construction; what remains
    to verify numerically is listed here. Fire resistance and the slab-column
    local-steel provision are reported as open.
    """
    if context is None:
        return [not_evaluated("slab_completion_context", "ACI 318-19 8.3, 8.7, 22.6, 24.3, 25.4",
                              "Floor context (spans, beam width, alpha_f, thickness screen) was not supplied."),
                *_open_checks()]
    checks = []
    fc, fy, h = inputs["fc_ksi"], inputs["fy_ksi"], inputs["thickness_in"]
    bar = layout["bar_size"]
    ld = development_length_in(bar, fc, fy)
    shortest_clear = min(context["clear_span_x_in"], context["clear_span_y_in"])
    checks.append(make_check("slab_bar_development", "ACI 318-19 25.4.2.4 with 8.7.4.1 continuity",
                             ld, shortest_clear / 2.0, "<=", "in",
                             details={"development_length_in": ld, "basis": "continuous uniform mats; at interior "
                                      "supports the embedment beyond the critical section is at least half the "
                                      "shortest clear span; the perimeter is slab_perimeter_bar_anchorage"}))
    from Design.SMRF_Beam_Slab_Strength import perimeter_slab_bar_anchorage
    for axis in ("x", "y"):
        anchorage = perimeter_slab_bar_anchorage(layout, axis, context["beam_width_in"], context["beam_clear_cover_in"],
                                                 context["beam_hoop_diameter_in"], context["fc_beam_ksi"], fy)
        checks.append(make_check("slab_perimeter_bar_anchorage", "ACI 318-19 25.4.3.1 / 8.7.4.1.3 at a discontinuous edge",
                                 anchorage["ldh_required_in"], anchorage["embedment_available_in"], "<=", "in", axis,
                                 details={**{k: v for k, v in anchorage.items() if k not in ("axis",)},
                                          "requirement": "the mats parallel to this axis terminate at the building edge with a "
                                                         "standard hook into the perimeter beam; the same anchorage is what "
                                                         "credits the slab bars in the beam strength at exterior ends"}))
    checks.append(make_check("slab_continuity_and_extensions", "ACI 318-19 8.7.4.1.3 / Fig. 8.7.4.1.3",
                             1, 1, "==", details={"basis": "top and bottom mats continuous through all supports at "
                                                  "uniform spacing; no cutoffs, so every minimum extension is exceeded"}))
    s_max = min(15.0 * (40.0 / (2.0 / 3.0 * fy)) - 2.5 * layout["layers"]["x_top"]["clear_cover_outer_mat_in"],
                12.0 * (40.0 / (2.0 / 3.0 * fy)))
    for name, layer in sorted(layout["layers"].items()):
        checks.append(make_check("slab_crack_control_spacing", "ACI 318-19 24.3.2 with fs = 2/3 fy",
                                 layer["spacing_in"], s_max, "<=", "in", name))
    # Shear path. ACI 318-19 8.4.3 is the slab's one-way shear (8.4.3.1: Vu at
    # the face of the support; the strip check above, at the beam faces).
    # 8.4.4 is two-way shear: 8.4.4.1.1 requires it "in the vicinity of
    # columns, loads, and reaction areas" at the 22.6.4 critical sections.
    # This slab's reactions are line reactions on beams, and every column
    # stands at the intersection of beams in both directions, so the slab's
    # load reaches the columns through the beams; how much of it the beams
    # take is the beam-supported-slab rule of ACI 318-14 8.10.8 (the 2019
    # body text no longer carries it; R8.2.1 keeps the method available):
    # Table 8.10.8.1 assigns 100% of the panel shear on the 45-degree
    # tributaries to a beam with alpha_f1 l2/l1 >= 1.0 (8.10.8.2 interpolates
    # below that), 8.10.8.3 adds the loads applied to the beam directly
    # including its stem, and 8.10.8.4 requires resistance to the total shear
    # on the panel. Here the beams carry the plate's actual reactions (their
    # sum is checked against the floor load by the transfer ledger) plus the
    # drop weight (SMRF_Capacity_Design._drop_weight) in the beam shear
    # design, which is 8.10.8.3/8.10.8.4 by analysis rather than by
    # tributary; the criterion recorded here is the Table 8.10.8.1 one,
    # alpha_f1 l2/l1 on every edge of every panel, not alpha_f alone. Using a
    # direct-design-method clause as the load-path criterion for a slab
    # analysed as a plate is an interpretation, and it is the engineer's:
    # two_way_shear_path_assessed.
    shear_clause = ("ACI 318-19 8.4.3.1 / 8.4.4.1.1 / 22.6.4; ACI 318-14 Table 8.10.8.1, 8.10.8.3, 8.10.8.4 "
                    "via 318-19 R8.2.1")
    path_ok = (context["alpha_f_min"] >= 1.0 and context["alpha_f_l2_l1_min"] >= 1.0
               and context["columns_at_beam_intersections"])
    if path_ok and context["two_way_shear_path_assessed"]:
        checks.append(make_check("slab_two_way_shear_applicability", shear_clause,
                                 1.0, context["alpha_f_l2_l1_min"], "<=", "alpha_f1 l2/l1",
                                 details={"columns_at_beam_intersections": True,
                                          "alpha_f_min": context["alpha_f_min"],
                                          "alpha_f_l2_l1_min": context["alpha_f_l2_l1_min"],
                                          "beam_share_of_panel_shear": "100% (Table 8.10.8.1 at alpha_f1 l2/l1 >= 1.0)",
                                          "basis": "assessed: the slab bears on beams on every edge and every column is at "
                                                   "a beam intersection, so its reactions are line reactions and no "
                                                   "slab-column critical section (8.4.4.1.1 / 22.6.4) exists; with "
                                                   "alpha_f1 l2/l1 >= 1.0 on every edge the beams take the whole panel "
                                                   "shear (318-14 Table 8.10.8.1) and are designed for the plate's "
                                                   "reactions plus their stem weight (8.10.8.3, 8.10.8.4 by analysis: "
                                                   "the transfer ledger closes the floor load); the slab itself is "
                                                   "checked for one-way shear at the beam faces (8.4.3.1, the strip "
                                                   "check above)"}))
    elif path_ok:
        checks.append(not_evaluated("slab_two_way_shear_applicability", shear_clause,
                                    f"alpha_f >= 1.0 (min {context['alpha_f_min']:.2f}) and alpha_f1 l2/l1 >= 1.0 (min "
                                    f"{context['alpha_f_l2_l1_min']:.2f}) on every edge, and every column is at a beam "
                                    "intersection; whether the beam-supported shear path (318-14 8.10.8) replaces a "
                                    "slab-column two-way shear check for a plate-analysed slab is an engineering "
                                    "assessment to assert in Design.Config.SlabActionAssertions.two_way_shear_path_assessed."))
    else:
        checks.append(not_evaluated("slab_two_way_shear_applicability", "ACI 318-19 8.4.4.1.1 / 22.6",
                                    f"alpha_f1 l2/l1 < 1.0 on some edge (min {context['alpha_f_l2_l1_min']:.2f}; alpha_f min "
                                    f"{context['alpha_f_min']:.2f}) or a column bears on the slab directly: the beams do not "
                                    "take the whole panel shear and slab-column two-way shear is not implemented."))
    checks.append(make_check("slab_deflection_control", "ACI 318-19 8.3.2.1 via 8.3.1.2 minimum thickness",
                             int(context["thickness_screen_passed"]), 1, "==",
                             details={"basis": "thickness at or above the Table 8.3.1.2 minimum; deflections need not be calculated"}))
    checks.append(make_check("slab_integrity_bottom_bars_through_column", "ACI 318-19 8.7.4.2",
                             2.0 * layout["layers"]["x_bottom"]["spacing_in"], context["column_core_width_in"] + layout["layers"]["x_bottom"]["spacing_in"],
                             "<=", "in", details={"basis": "continuous bottom mat; two bars fit within the column core",
                                                  "placement_requirement": "centre the bottom mat on every column line"}))
    max_positive = max(row["mu_kip_in_per_ft"] for row in demands if row["face"] == "bottom")
    for name in ("x_top", "y_top", "x_bottom", "y_bottom"):
        checks.append(make_check("slab_corner_reinforcement", "ACI 318-19 8.7.3.1 (alpha_f > 1.0 discontinuous corners)",
                                 max_positive, layout["layers"][name]["flexure"]["phi_mn_kip_in_per_ft"],
                                 "<=", "kip-in/ft", name,
                                 details={"basis": "uniform mats extend over the full corner region in both directions"}))
    checks.extend(_open_checks())
    return checks


def _hook_rejected_spacings(bar, inputs, policy, context):
    """Spacings whose perimeter hook does not fit the perimeter beam (ACI 318-19 25.4.3.1), with the embedment
    offered: ({spacing: ldh}, embedment). Without floor context nothing is rejected."""
    if context is None:
        return {}, None
    from Design.SMRF_Beam_Slab_Strength import hook_development_length_in
    embedment = context["beam_width_in"] - context["beam_clear_cover_in"] - context["beam_hoop_diameter_in"]
    rejected = {}
    for spacing in policy["spacing_options_in"]:
        ldh, _factors = hook_development_length_in(bar, context["fc_beam_ksi"], inputs["fy_ksi"], spacing)
        if ldh > embedment:
            rejected[spacing] = ldh
    return rejected, embedment


def _envelopes(demands):
    """The governing Mu and Vu of each layer (axis_face) over every panel and floor."""
    envelopes = {}
    for axis in ("x", "y"):
        for face in ("top", "bottom"):
            rows = [row for row in demands if row["axis"] == axis and row["face"] == face]
            envelopes[f"{axis}_{face}"] = {"mu_kip_in_per_ft": max(row["mu_kip_in_per_ft"] for row in rows),
                                           "vu_kip_per_ft": max(row["vu_kip_per_ft"] for row in rows)}
    return envelopes


def explain_unsized_layers(record):
    """Why the ladder found no four-layer layout, read from the record's own inputs (the record is not changed).

    For each layer that no offered bar sized: its demand envelope and the offered candidate (bar, spacing whose
    hook fits) that fails the fewest layer checks, with those checks' demand and capacity. A candidate that fails
    only one-way shear says the thickness or the supporting beams govern, not the steel.
    """
    data = record.get("inputs") or {}
    try:
        inputs, policy, context = _inputs(data.get("slab")), _policy(data.get("policy")), _context(data.get("context"))
        demands = _demands(data.get("demand_evidence"), inputs)
    except (ValueError, TypeError, OverflowError) as exc:
        return {"reason": str(exc)}
    sized = None
    for trial in record.get("trial_history", []):
        layers = set(trial.get("layers_sized", []))
        sized = layers if sized is None else sized | layers
    offered = []
    for bar in policy["bar_sizes"]:
        rejected, _embedment = _hook_rejected_spacings(bar, inputs, policy, context)
        offered.extend((bar, spacing) for spacing in policy["spacing_options_in"] if spacing not in rejected)
    explanation = {}
    for name, envelope in _envelopes(demands).items():
        if name in (sized or set()):
            continue
        axis = name.split("_")[0]
        best = None
        for bar, spacing in offered:
            layer = _layout(inputs, policy, bar, axis, spacing)
            if layer is None:
                continue
            failed = [c for c in _layer_checks(layer, envelope, inputs, name) if c["status"] != "pass"]
            key = (len(failed), -bar, spacing)
            if best is None or key < best[0]:
                best = (key, layer, failed)
        entry = {"demand_envelope": envelope, "closest_offered_candidate": None, "failed_checks": [],
                 "offered_candidates": len(offered)}
        if best is not None:
            _key, layer, failed = best
            entry["closest_offered_candidate"] = {
                "bar_size": layer["bar_size"], "spacing_in": layer["spacing_in"],
                "effective_depth_in": layer["effective_depth_in"], "area_in2_per_ft": layer["area_in2_per_ft"],
                "phi_mn_kip_in_per_ft": layer["flexure"]["phi_mn_kip_in_per_ft"],
                "phi_vc_kip_per_ft": layer["one_way_shear"]["phi_vc_kip_per_ft"]}
            entry["failed_checks"] = [{"id": c["id"], "clause": c["clause"], "demand": c["demand"],
                                       "capacity": c["capacity"], "comparison": c["comparison"], "units": c["units"]}
                                      for c in failed]
        explanation[name] = entry
    return explanation


def _open_checks():
    return [not_evaluated("slab_column_local_minimum_steel", "ACI 318-19 8.6.1.2",
                          "Applicability of the slab-column local minimum steel provision to a beam-supported slab has not been assessed."),
            not_evaluated("slab_fire_resistance", "ACI 318-19 4.11 / applicable building code",
                          "Required fire-resistance rating and cover/thickness for it are outside this design scope.")]


def design_slab_reinforcement(slab_inputs, demand_evidence, policy=None, context=None):
    """Return a repeatable strength screen; never accepts the complete slab.

    ``slab_inputs``: thickness_in, fc_ksi, fy_ksi=60,
    max_aggregate_size_in, exposure='sheltered_interior',
    steel_specification='ASTM A706', concrete_type='normalweight',
    panel_ids (the entire unique panel inventory), num_floor.

    ``demand_evidence`` must provide all flags in _VERIFIED_FLAGS as True;
    method ('plate_finite_element', 'equivalent_frame', or 'direct_design'),
    source, load_combination_basis, analysis_model_sha256; slab_input_sha256
    from slab_input_signature; shear_envelope_basis='support_face_maximum';
    load_scope='uniform_area_gravity_no_concentrated_loads'. Its ``strips``
    list contains exactly one row for every (panel_id, axis x/y, face top/bottom),
    with nonnegative mu_kip_in_per_ft and vu_kip_per_ft, demand_location, and
    tension_face_at_shear_verified=True. Actions must envelope ALL floors,
    spatial locations within a panel (not an average strip moment), applicable
    factored combinations and live-load patterns, including twisting-moment
    resolution into orthogonal reinforcement demands. Support-face maximum
    shear is deliberately not reduced to d, so changing bar layout cannot
    invalidate its section location. Membrane axial force is outside scope.

    Optional policy: bar_sizes subset [4,5,6], spacing_options_in, clear_cover_in
    >=0.75, outer_axis x/y. Optional ``context`` (clear_span_x_in,
    clear_span_y_in, beam_width_in, alpha_f_min, alpha_f_l2_l1_min,
    thickness_screen_passed, column_core_width_in, two_way_shear_path_assessed,
    columns_at_beam_intersections) enables the development, crack-control, shear-path,
    deflection, integrity and corner checks; without it they stay open. Crossing orthogonal bars touch within each face mat;
    two face mats retain at least max(1 in, 4/3 aggregate) clear gap. One bar size
    is shared by all four layers; spacings may differ by axis/face but are
    uniform across the building. Failure exhausts the ladder without fallback.
    """
    record = {"method_version": METHOD_VERSION, "stage": "verified_strip_strength_only",
              "accepted": False, "screen_passed": False, "layout": None, "trial_history": []}
    try:
        inputs, resolved_policy = _inputs(slab_inputs), _policy(policy)
        resolved_context = _context(context)
        # Preserve immutable-by-copy, JSON-safe provenance in saved designs.
        # A caller changing its source dictionary must not silently change the
        # demand evidence underneath already-computed reinforcement capacities.
        # Retain valid inputs even when demands are unverified. Otherwise the
        # later audit replaces the actual verification gap with a misleading
        # "missing slab inputs" error and cannot reproduce the rejected attempt.
        saved_evidence = json.loads(json.dumps(demand_evidence, allow_nan=False))
        record["inputs"] = {"slab": inputs, "demand_evidence": saved_evidence, "policy": resolved_policy,
                            "context": resolved_context}
        demands = _demands(saved_evidence, inputs)
    except (ValueError, TypeError, OverflowError) as exc:
        checks = [not_evaluated("slab_verified_strip_action_inputs", "ACI 318-19 8.4 / Chapter 6",
                                str(exc)), *_open_checks()]
        return {**record, "checks": checks, "summary": summarize_checks(checks)}
    candidates = []
    for bar in resolved_policy["bar_sizes"]:
        diameter, _ = _BARS[bar]
        gap = inputs["thickness_in"] - 2.0 * resolved_policy["clear_cover_in"] - 4.0 * diameter
        minimum_gap = max(1.0, 4.0 / 3.0 * inputs["max_aggregate_size_in"])
        if gap < minimum_gap:
            record["trial_history"].append({"bar_size": bar, "passed": False,
                                            "reason": "Two orthogonal face mats do not fit with required clear gap."})
            continue
        # The mats end at the building perimeter, hooked into the perimeter
        # beam (ACI 318-19 25.4.3.1). The hook length depends on the hook
        # spacing (psi_r = 1.6 below 6 db), so each candidate spacing is
        # checked with its own hook: a spacing whose hook does not fit is
        # not offered for that bar, and a bar with no fitting spacing at
        # all is not offered. The check on the selected layout is
        # slab_perimeter_bar_anchorage.
        hook_rejected, embedment = _hook_rejected_spacings(bar, inputs, resolved_policy, resolved_context)
        if resolved_context is not None:
            if len(hook_rejected) == len(resolved_policy["spacing_options_in"]):
                ldh = min(hook_rejected.values())
                record["trial_history"].append({"bar_size": bar, "passed": False,
                                                "reason": f"Hook development {ldh:.2f} in exceeds the {embedment:.2f} in "
                                                          "the perimeter beam offers (25.4.3.1) at every spacing offered."})
                continue
        layers = {}
        envelopes = _envelopes(demands)
        for axis in ("x", "y"):
            for face in ("top", "bottom"):
                envelope = envelopes[f"{axis}_{face}"]
                for spacing in resolved_policy["spacing_options_in"]:
                    if spacing in hook_rejected:
                        continue
                    layer = _layout(inputs, resolved_policy, bar, axis, spacing)
                    if layer and all(c["status"] == "pass" for c in _layer_checks(layer, envelope, inputs, "trial")):
                        layers[f"{axis}_{face}"] = {**layer, "face": face, "demand_envelope": envelope}
                        break
        success = len(layers) == 4
        record["trial_history"].append({"bar_size": bar, "passed": success,
                                        "layers_sized": sorted(layers),
                                        "spacings_rejected_by_hook_in": sorted(hook_rejected)})
        if success:
            score = sum(layer["area_in2_per_ft"] for layer in layers.values())
            candidates.append((score, bar, {"bar_size": bar, "outer_axis": resolved_policy["outer_axis"],
                                           "mat_clear_gap_in": gap, "minimum_mat_clear_gap_in": minimum_gap,
                                           "uniform_all_panels_and_floors": True, "layers": layers}))
    if not candidates:
        checks = [make_check("slab_strip_reinforcement_ladder", "ACI 318-19 Chapters 8, 20, 21, 22, 25",
                             1, 0, details={"reason": "No allowed four-layer layout passes; increase slab thickness or revisit demands/policy."}),
                  *_open_checks()]
    else:
        _, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
        record["layout"] = selected
        checks = [make_check("slab_strip_cover", "ACI 318-19 Table 20.5.1.3.1, sheltered slab #11 and smaller",
                             resolved_policy["clear_cover_in"], .75, ">=", "in"),
                  make_check("slab_strip_face_mat_clearance", "ACI 318-19 25.2; conservative mat placement constraint",
                             selected["mat_clear_gap_in"], selected["minimum_mat_clear_gap_in"], ">=", "in")]
        for demand in demands:
            layer = selected["layers"][f"{demand['axis']}_{demand['face']}"]
            location = f"{demand['panel_id']}:{demand['axis']}:{demand['face']}"
            checks.extend(_layer_checks(layer, demand, inputs, location))
        record["screen_passed"] = all(check["status"] == "pass" for check in checks)
        checks.extend(_completion_checks(inputs, selected, demands, resolved_context))
    return {**record, "checks": checks, "summary": summarize_checks(checks)}


def evaluate_slab_reinforcement(record):
    """Recompute saved evidence, including every bar selection and capacity."""
    if not isinstance(record, dict) or not isinstance(record.get("inputs"), dict):
        return design_slab_reinforcement(None, None)["checks"]
    data = record["inputs"]
    expected = design_slab_reinforcement(data.get("slab"), data.get("demand_evidence"), data.get("policy"),
                                         data.get("context"))
    match = record == expected
    return [make_check("slab_strip_saved_evidence", "Deterministic slab strip strength evidence audit", int(match), 1, "=="),
            *expected["checks"]]
