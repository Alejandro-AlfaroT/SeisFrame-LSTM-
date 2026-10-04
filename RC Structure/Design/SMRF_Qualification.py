"""Connect design-candidate artifacts to an explicit, fail-closed checklist.

The current revision deliberately does not certify complete SMRF designs.
Open checks are retained until the relevant design algorithms and inputs
exist. Callers cannot turn an unevaluated check into a pass with a DCR flag.
"""
from __future__ import annotations

import math

from Design.SMRF_Common import make_check, not_evaluated, summarize_checks
from Design.SMRF_Demands import demand_scope_checks
from Design.SMRF_Detailing import evaluate_detailing
from Design.SMRF_Joints import evaluate_joints
from Design.SMRF_Slab import evaluate_slab
from Design.SMRF_Slab_Reinforcement import evaluate_slab_reinforcement


QUALIFICATION_VERSION = "smrf_qualification_v3_joint_actions_clear_cover"
# This gate is lifted only after required design algorithms and independent
# verification exist. A passing subset of unit tests is not release approval.
GENERATION_RELEASE_READY = False
SCOPE = {
    "system": "RC special moment frame archetype; every grid line in X and Y",
    "geometry": "regular rectangular grid; shared beam and column section families",
    "diaphragm": "rigid idealization; diaphragm/collector design not included",
    "foundation": "fixed supports; foundation design not included",
    "basis": ["ACI 318-19", "ASCE 7-22"],
    "claim": "research frame qualification, NOT full building-code certification",
}


def reinforcement_consistency_checks(record):
    """Check redundant saved values against bar numbers and model clear cover."""
    import Structure_Parameters as sp
    bars, material = record.get("reinforcement", {}), record.get("materials", {})
    checks = []
    for prefix in ("beam", "col"):
        try:
            size, hoop = bars[f"{prefix}_bar_size"], bars[f"{prefix}_stirrup_bar_size"]
            if type(size) is not int or type(hoop) is not int:
                raise ValueError("Bar numbers must be integers.")
            clear = bars[f"{prefix}_clear_cover_in"]
            if isinstance(clear, bool) or not math.isfinite(clear) or clear < 1.5:
                raise ValueError("Frame clear cover must be at least 1.5 in outside hoops.")
            expected = {f"{prefix}_bar_area_in2": sp.rebar_area(size),
                        f"{prefix}_bar_diameter_in": sp.rebar_diameter(size),
                        f"{prefix}_stirrup_diameter_in": sp.rebar_diameter(hoop),
                        f"{prefix}_longitudinal_centroid_offset_in": clear + sp.rebar_diameter(hoop) + sp.rebar_diameter(size)/2}
            matches = all(not isinstance(bars.get(key), bool) and
                          math.isclose(bars[key], value, rel_tol=0, abs_tol=1e-12)
                          for key, value in expected.items())
            checks.append(make_check(f"reinforcement.{prefix}_model_consistency",
                                     "Saved bar database / actual clear-cover geometry", int(matches), 1, "=="))
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            checks.append(not_evaluated(f"reinforcement.{prefix}_model_consistency",
                                        "Saved bar database / actual clear-cover geometry", str(exc)))
    aggregate = material.get("aggregate_size_in")
    supported = (material.get("reinforcement_specification") == "ASTM A706 Grade 60"
                 and material.get("exposure") == "sheltered_interior"
                 and material.get("fy_ksi") == material.get("fyt_ksi") == 60
                 and isinstance(aggregate, (float, int)) and not isinstance(aggregate, bool)
                 and math.isfinite(aggregate) and aggregate > 0)
    checks.append(make_check("reinforcement.declared_material_scope",
                             "Approved sheltered interior A706 Grade 60 research scope", int(supported), 1, "=="))
    return checks


def detailing_inputs(record):
    """Use saved, reproducible inputs only; do not invent a slab or hoop cage."""
    sections, bars = record.get("sections", {}), record.get("reinforcement", {})
    geometry, material = record.get("geometry", {}), record.get("materials", {})
    inputs = {"material": material, "geometry": {
        "span_x_in": geometry.get("bay_x_in"), "span_y_in": geometry.get("bay_y_in")}}
    for member, prefix in (("beam", "beam"), ("column", "col")):
        values = {"b_in": sections.get(f"b_{prefix}_in"),
                  "h_in": sections.get(f"h_{prefix}_in"),
                  "fc_ksi": sections.get(f"fc_{prefix}_ksi"),
                  "bar_db_in": bars.get(f"{prefix}_bar_diameter_in"),
                  "bar_area_in2": bars.get(f"{prefix}_bar_area_in2"),
                  "stirrup_db_in": bars.get(f"{prefix}_stirrup_diameter_in"),
                  "n_top": bars.get(f"{prefix}_top_bars"),
                  "n_bottom": bars.get(f"{prefix}_bot_bars"),
                  "n_side_per_face": bars.get(f"{prefix}_side_bars"),
                  "hoop_spacing_in": bars.get(f"{prefix}_stirrup_spacing_in"),
                  "aggregate_size_in": material.get("aggregate_size_in")}
        try:
            # Existing COVER means centroid offset, NOT clear concrete cover.
            # Show the actual consequence; never silently relabel it as 1.5 in clear.
            centroid = bars.get(f"{prefix}_longitudinal_centroid_offset_in", bars.get("longitudinal_centroid_offset_in"))
            values["clear_cover_in"] = (centroid -
                                        values["stirrup_db_in"] - values["bar_db_in"] / 2)
        except (KeyError, TypeError):
            pass
        if member == "column":
            try:
                values["clear_height_in"] = geometry["story_h_in"] - sections["h_beam_in"]
            except (KeyError, TypeError):
                pass
        # These must be outputs of an actual detailing procedure, not defaults.
        for key, value in record.get("detailing", {}).get(member, {}).items():
            if key not in values:
                values[key] = value
        inputs[member] = values
    return inputs


_VERIFICATION_ITEMS = {
    "floor.independent_hand_verification": "floor_hand_check_verified",
    "qualification.strength_model_verification": "strength_model_verified",
    "qualification.detailing_model_consistency": "detailing_model_consistency_verified",
    "slab_column_local_minimum_steel": "slab_column_local_steel_assessed",
    "slab_fire_resistance": "fire_resistance_scope_accepted",
    "detailing.congestion_and_placement": "congestion_and_placement_accepted",
    "floor.compatibility_idealization_reviewed": "floor_frame_compatibility_reviewed",
}


def _apply_verification_assertions(record, checks):
    """Replace human-verification items with asserted ones (Design.Config.IndependentVerification).

    The assertion is recorded with its author and basis; the code neither
    checks the assertion nor makes it. Without an author and basis an
    asserted flag is ignored.
    """
    policy = ((record.get("request_identity") or {}).get("policy") or {}).get("verification") \
        or (record.get("demand_basis") or {}).get("verification") or {}
    from Design.SMRF_Common import assertion_provenance_valid
    if not assertion_provenance_valid(policy):
        return checks
    result = []
    for check in checks:
        flag = _VERIFICATION_ITEMS.get(check.get("id"))
        if flag and check.get("status") == "not_evaluated" and policy.get(flag) is True:
            result.append(make_check(check["id"], check["clause"], 1, 1, "==", location=check.get("location", ""),
                                     details={"asserted_by": policy["asserted_by"], "assertion_date": policy.get("assertion_date"),
                                              "assertion_basis": policy["assertion_basis"],
                                              "original_reason": check.get("details", {}).get("reason")}))
        else:
            result.append(check)
    return result


def recomputed_evidence(record):
    """Rebuild the capacity design and slab strengths from the record; compare with the saved copies.

    Returns {"capacity": recomputed capacity or {} when unusable, "strengths":
    recomputed entries or {}, "families", "checks": the integrity checks}.
    Downstream qualification consumes only these recomputed objects: nothing
    nested in the saved copies is read once they have been compared. The
    saved evidence is unusable when the recomputation differs, the recorded
    hoops differ from the model's, or the slab strengths differ.
    """
    from Design.SMRF_Design_Evidence import capacity_design_recomputation, slab_strength_recomputation
    rebar = record.get("reinforcement", {})
    checks, capacity, strengths, families = [], {}, {}, {}
    if record.get("capacity_design"):
        recomputation = capacity_design_recomputation(record)
        checks.append(make_check(
            "qualification.capacity_evidence_recomputed", "Evidence integrity",
            int(recomputation["consistent"]), 1, "==",
            details={"basis": "capacity design rebuilt from the record's sections, bars, slab layout, transfer and "
                              "saved combination actions; every check, the hoops, joint shear, anchorage and "
                              "acceptance compared with the saved evidence; the recomputed object is what "
                              "qualification consumes",
                     "differences": recomputation["differences"]}))
        recomputed = recomputation["recomputed"] or {}
        transverse = recomputed.get("transverse") or {}
        # Column legs are selected per direction; the scalar the model reads is
        # the lighter direction (legs_model). Both must match the record.
        column_hoops = transverse.get("column") or {}
        hoops_match = bool(recomputed) and all(
            rebar.get(key) == (transverse.get(member) or {}).get(field)
            for member, prefix in (("beam", "beam"), ("column", "col"))
            for key, field in ((f"{prefix}_stirrup_bar_size", "bar_size"),
                               (f"{prefix}_stirrup_legs", "legs_model" if member == "column" else "legs"),
                               (f"{prefix}_stirrup_spacing_in", "spacing_in"))
        ) and rebar.get("col_stirrup_legs_by_direction") == column_hoops.get("legs")
        checks.append(make_check(
            "qualification.hoops_match_design", "Evidence integrity",
            int(hoops_match), 1, "==",
            details={"basis": "the transverse steel the model and IMK calibration read (reinforcement.*_stirrup_*) "
                              "must be the hoops the recomputed capacity design selects; column legs per direction, "
                              "with col_stirrup_legs the lighter direction",
                     "reinforcement": {k: rebar.get(k) for k in ("beam_stirrup_bar_size", "beam_stirrup_legs",
                                                                  "beam_stirrup_spacing_in", "col_stirrup_bar_size",
                                                                  "col_stirrup_legs", "col_stirrup_legs_by_direction",
                                                                  "col_stirrup_spacing_in")},
                     "recomputed_capacity_design": transverse}))
        if recomputation["consistent"] and hoops_match:
            capacity = recomputed
    if record.get("beam_slab_strengths") is not None or record.get("slab_reinforcement") is not None:
        strength = slab_strength_recomputation(record)
        checks.append(make_check(
            "qualification.slab_strength_evidence_recomputed", "Evidence integrity",
            int(strength["consistent"]), 1, "==",
            details={"basis": "beam-plus-slab strengths rebuilt from the record's sections, bars and slab layout and "
                              "compared entry by entry with the saved ones; the recomputed entries feed the joints",
                     "differences": strength["differences"]}))
        if strength["consistent"]:
            strengths, families = strength["entries"], strength["families"]
    return {"capacity": capacity, "strengths": strengths, "families": families, "checks": checks}


def _apply_capacity_design_evidence(record, checks, evidence):
    """Settle the checks the recomputed capacity design and slab strengths answer.

    The detailing module reports the cage-level provisions it cannot see as
    not_evaluated; where the capacity design carries the numbers (confinement
    area, supported-bar spacing, hoops from probable shear) those entries are
    replaced with evaluated ones, deterministic from the record. Only the
    recomputed objects in ``evidence`` are read.
    """
    capacity = evidence["capacity"]
    replaced = {}
    if capacity:
        checks = list(checks) + [c for c in capacity.get("checks", [])]
        columns, beams = capacity.get("columns", {}), capacity.get("beams", {})
        conf, hoops = columns.get("confinement", {}), columns.get("hoops")
        if hoops and conf:
            # Ash is per direction (18.7.5.4): the check reports the direction
            # with the smallest margin and carries both in its details.
            by_direction = hoops.get("by_direction") or {}
            if by_direction:
                tightest = min(by_direction.values(),
                               key=lambda v: v["ash_provided_per_in"] - v["ash_required_per_in"])
                demand, provided = tightest["ash_required_per_in"], tightest["ash_provided_per_in"]
            else:
                demand, provided = conf["ash_ratio_required"] * conf["bc_in"], hoops["ash_provided_per_in"]
            replaced["column.confinement_area_and_support"] = make_check(
                "column.confinement_area_and_support", "ACI 318-19 18.7.5.2--18.7.5.4",
                demand, provided, "<=", "in2/in",
                details={"legs": hoops["legs"], "bar_size": hoops["bar_size"], "spacing_in": hoops["spacing_in"],
                         "high_axial": conf["high_axial"],
                         "by_direction": {axis: {k: v[k] for k in ("legs_key", "legs", "bc_in", "ash_required_per_in",
                                                                  "ash_provided_per_in")}
                                          for axis, v in by_direction.items()},
                         "scope": "confinement steel quantity Ash/s against 18.7.5.4 in each direction with the leg "
                                  "count the generated arrangement realizes across that direction's faces "
                                  "(detailing.cage_layout); the tightest direction is the reported pair"})
            limit = 8.0 if conf["high_axial"] else 14.0
            replaced["column.supported_bar_distance_basic"] = make_check(
                "column.supported_bar_distance_basic", "ACI 318-19 18.7.5.2(e)/(f)",
                conf["hx_in"], limit, "<=", "in",
                details={"scope": "hx is the supported-bar spacing of the generated arrangement (detailing.cage_layout)"})
        beam_hoops = beams.get("hoops")
        if beam_hoops:
            replaced["beam.hoop_layout_and_axial_applicability"] = make_check(
                "beam.hoop_layout_and_axial_applicability", "ACI 318-19 18.6.2.1 / 18.6.4 / 18.6.5",
                int(bool(beams.get("section_adequate"))), 1, "==",
                details={"hoops": beam_hoops, "vc_zero_hinge_zone": beams.get("vc_zero_hinge_zone"),
                         "axial_basis": "frame beams carry no factored axial load above Ag fc/10"})
        splices = capacity.get("splices")
        if splices:
            replaced["detailing.anchorage_splices_and_cover"] = make_check(
                "detailing.anchorage_splices_and_cover", "ACI 318-19 18.6.3.3 / 18.7.4.4 / 18.8.5 / 20.5 / 25.4--25.5",
                int(capacity.get("anchorage", {}).get("all_pass", False)), 1, "==",
                details={"anchorage": capacity.get("anchorage", {}).get("directions"),
                         "beam_splice": splices["beam"], "column_splice": splices["column"],
                         "cover": "member clear covers audited by reinforcement consistency checks",
                         "scope": "hooked anchorage, through-bar depth, splice type/location and cover; "
                                  "congestion and placement are detailing.congestion_and_placement"})
            replaced["detailing.congestion_and_placement"] = not_evaluated(
                "detailing.congestion_and_placement", "ACI 318-19 25.2 / 26.6; constructability",
                "Bar-placement drawings, mechanical-splice staggering and congestion at joints are not designed.")
            replaced["beam.continuity_and_strength_balance"] = make_check(
                "beam.continuity_and_strength_balance", "ACI 318-19 18.6.3.1--18.6.3.3",
                1, 1, "==",
                details={"basis": "uniform top and bottom bars continuous through every support; end and along-span "
                                  "strength balance are the beam.span_strength checks; splices per detailing.splices_designed"})
        open_items = [c for c in capacity.get("checks", []) if c.get("status") == "not_evaluated"]
        failed_items = [c for c in capacity.get("checks", []) if c.get("status") == "fail"]
        if open_items and not failed_items:
            # Missing classification or detailing evidence stays open here; a
            # conservative numeric screen is not completion.
            replaced["qualification.joint_capacity_completion"] = not_evaluated(
                "qualification.joint_capacity_completion", "ACI 318-19 18.7.3; 18.8",
                "capacity-design items without supporting evidence: " + ", ".join(
                    f"{c.get('id')}@{c.get('location', '')}" if c.get("location") else str(c.get("id")) for c in open_items[:8]))
        else:
            replaced["qualification.joint_capacity_completion"] = make_check(
                "qualification.joint_capacity_completion", "ACI 318-19 18.7.3; 18.8",
                int(bool(capacity.get("accepted"))), 1, "==",
                details={"method_version": capacity.get("method_version"),
                         "joint_shear_all_pass": capacity.get("joints", {}).get("all_pass"),
                         "joint_evidence_complete": capacity.get("joints", {}).get("evidence_complete"),
                         "anchorage_all_pass": capacity.get("anchorage", {}).get("all_pass")})
        replaced["qualification.column_capacity_shear"] = make_check(
            "qualification.column_capacity_shear", "ACI 318-19 18.7.6",
            int(bool(hoops) and bool(columns.get("section_adequate"))), 1, "==",
            details={"governing": columns.get("governing")})
        replaced["qualification.detailing_model_consistency"] = {
            **not_evaluated("qualification.detailing_model_consistency", "Research model-to-design consistency",
                            "Independent validation (methodology item 7): confirm that confinement zones, bar "
                            "positions, cover and the selected hoops propagate consistently to section strength, "
                            "IMK calibration and exports. The mechanical identity of the saved hoops and the "
                            "designed hoops is qualification.hoops_match_design, evaluated here, not asserted."),
            "details": {"reason": "awaiting independent validation",
                        "rho_sh_source": "reinforcement.*_stirrup_* read by Model.IMK_Calibration.transverse_steel_ratio"}}
        cages = {member: (capacity.get(group) or {}).get("cage") or {} for member, group in (("column", "columns"), ("beam", "beams"))}
        from Design.SMRF_Cage_Layout import cage_passes
        replaced["detailing.cage_layout"] = make_check(
            "detailing.cage_layout", "ACI 318-19 18.6.4.4 / 18.7.5.2(b)-(f) / 25.7.2.3",
            int(all(cage_passes(c) for c in cages.values())), 1, "==",
            details={member: {"legs": c.get("legs"), "hx_in": c.get("hx_in"), "legs_min": c.get("legs_min"),
                              "legs_max": c.get("legs_max"),
                              "arrangement": {face: {"positions_in": a["positions_in"], "supported": a["supported"],
                                                     "crossties": a["crossties"]}
                                              for face, a in (c.get("arrangement") or {}).items()},
                              "failing": [x for x in c.get("checks", []) if not x.get("passes")]}
                     for member, c in cages.items()} | {
                "scope": "perimeter hoop plus crossties engaging bars, generated from the bar positions: alternate-bar "
                         "support, the 6-in clear rule and the supported-bar spacing are checked on the arrangement; "
                         "the leg count Av and Ash use is the arrangement's in both directions; 135-degree hooks, "
                         "crosstie end alternation and placement are fabrication requirements (detailing.congestion_and_placement)"})
    else:
        for name, clause, reason in (
            ("qualification.joint_capacity_completion", "ACI 318-19 18.7.3; 18.8",
             "No usable capacity design is saved with this record."),
            ("qualification.column_capacity_shear", "ACI 318-19 18.7.6",
             "No usable capacity design is saved with this record."),
            ("qualification.detailing_model_consistency", "Research model-to-design consistency",
             "No usable capacity design is saved with this record."),
            ("detailing.cage_layout", "ACI 318-19 18.6.4.4 / 18.7.5.2(b)-(f) / 25.7.2.3",
             "No usable capacity design is saved with this record."),
        ):
            replaced[name] = not_evaluated(name, clause, reason)
    strengths = evidence["strengths"] or {}
    layout = (record.get("slab_reinforcement") or {}).get("layout")
    established = {"developed_effective_width", "terminated_undeveloped", "flange_concrete_undeveloped_bars"}
    developed = bool(strengths) and layout is not None and all(
        v.get("slab_basis") in established for v in strengths.values())
    if developed:
        undeveloped_ends = sorted({key.rsplit("/", 1)[0] for key, v in strengths.items()
                                   if v.get("slab_basis") != "developed_effective_width"})
        replaced["qualification.slab_contribution"] = make_check(
            "qualification.slab_contribution", "ACI 318-19 18.7.3.2 / 6.3.2", 1, 1, "==",
            details={"basis": "slab mats within the effective flange, developed as continuous uniform mats at interior "
                              "ends and by the perimeter hook (25.4.3.1) at exterior ends, in SCWB, joint shear and the "
                              "beam hinges; where the hook does not fit, the exterior end carries no slab bars in either "
                              "sign (rectangular hogging, flange-concrete sagging)",
                     "undeveloped_exterior_ends": undeveloped_ends})
    else:
        replaced["qualification.slab_contribution"] = not_evaluated(
            "qualification.slab_contribution", "ACI 318-19 18.7.3.2 / 6.3.2",
            "Slab reinforcement is not established (slab action evidence not asserted as verified); beam strengths "
            "in the screen and hinges are rectangular proxies. Do not read this as zero slab strength.")
    result = []
    seen = set()
    for check in checks:
        name = check.get("id")
        if name in replaced and check.get("status") == "not_evaluated":
            if name not in seen:
                result.append(replaced[name])
                seen.add(name)
            continue
        result.append(check)
    for name, check in replaced.items():
        if name not in seen and not any(c.get("id") == name for c in result):
            result.append(check)
    return result


def qualify_design(record):
    inputs = detailing_inputs(record)
    checks = evaluate_detailing(inputs)
    checks.extend(reinforcement_consistency_checks(record))
    for member in ("beam", "column"):
        conflicts = [key for key, value in record.get("detailing", {}).get(member, {}).items()
                     if inputs[member].get(key) != value]
        checks.append(make_check(f"detailing.{member}_authoritative_inputs",
                                 "Detailing cannot replace the saved frame geometry or bars",
                                 int(not conflicts), 1, "==", details={"conflicting_fields": conflicts}))
    # Everything nested that the joints and the capacity items read is
    # recomputed from the record first; the saved copies are compared, then
    # ignored (a saved joint_evidence is never consumed at qualification).
    recomputed = recomputed_evidence(record)
    checks.extend(recomputed["checks"])
    evidence = {}
    if record.get("design_actions") is not None:
        from Design.SMRF_Design_Evidence import joint_evidence
        try:
            evidence = joint_evidence(record, capacity=recomputed["capacity"],
                                      beam_slab_strengths=recomputed["strengths"] or None)
        except (KeyError, ValueError, TypeError) as exc:
            evidence = {}
            checks.append(not_evaluated("joint.reproducible_actions", "ACI 318-19 18.7--18.8", str(exc)))
    checks.extend(evaluate_joints(evidence))
    checks.extend(evidence.get("completeness_checks", []))
    from Design.SMRF_Demands import evaluate_demand_basis
    # The strength-distribution evidence is checked against the beam families
    # rebuilt from the record's final cage (never against the saved copies).
    checks.extend(evaluate_demand_basis(record, strength_families=recomputed["families"] or None))
    checks.extend(evaluate_slab(record.get("slab")))
    if record.get("gravity_load_model") == "slab_transfer" or record.get("floor_transfer") is not None:
        from Design.SMRF_Floor_Transfer import validate_floor_transfer
        try:
            geometry, loads = record["geometry"], record["floor_loads"]
            area = geometry["num_bay_x"] * geometry["num_bay_y"] * geometry["bay_x_in"] * geometry["bay_y_in"] / 144.0
            validate_floor_transfer(record.get("floor_transfer"), geometry, record["sections"],
                                    record["slab"]["thickness_in"],
                                    loads["floor_dead_load_ksf"] * area, loads["floor_live_load_ksf"] * area)
            transfer_error = None
        except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
            transfer_error = str(exc)
        checks.append(make_check("floor.transfer_load_integrity", "Recomputed frame load inventory and force/moment balance",
                                 int(transfer_error is None), 1, "==", details={"error": transfer_error}))
        from Design.SMRF_Coupled_Comparison import evaluate_coupled_comparison
        checks.extend(evaluate_coupled_comparison(record))
    from Design.SMRF_Beam_Actions import evaluate_saved_beam_bending
    checks.append(evaluate_saved_beam_bending(record))
    from Design.SMRF_Design_Evidence import slab_strength_inputs, analysis_input_signature
    strength = record.get("slab_reinforcement") or {}
    strength_inputs = strength.get("inputs", {}).get("slab")
    if strength_inputs is not None or record.get("slab_reinforcement_inputs") is not None:
        try:
            from Design.SMRF_Slab_Reinforcement import slab_input_signature
            expected = slab_input_signature(slab_strength_inputs(record))
            saved = [value for value in (strength_inputs, record.get("slab_reinforcement_inputs"))
                     if value is not None]
            matches = all(slab_input_signature(value) == expected for value in saved)
        except (KeyError, ValueError, TypeError, OverflowError):
            matches = False
        checks.append(make_check("slab_strip.frame_input_consistency", "Slab strength / current frame inputs",
                                 int(matches), 1, "=="))
        if not matches:
            strength = None  # Never display capacities for a different slab as current checks.
    checks.extend(evaluate_slab_reinforcement(strength))
    from Design.SMRF_Slab_Actions import evaluate_slab_actions
    # Asserted evidence lives in the reinforcement inputs (audited by exact
    # recomputation); the computed evidence is always saved beside it so an
    # unasserted design still shows what there is to review.
    checks.extend(evaluate_slab_actions((strength or {}).get("inputs", {}).get("demand_evidence")
                                        or record.get("slab_actions")))
    slab_inputs = (record.get("slab") or {}).get("inputs", {})
    if slab_inputs:
        for group in ("geometry", "sections"):
            matches = all(record.get(group, {}).get(key) == value
                          for key, value in slab_inputs.get(group, {}).items())
            checks.append(make_check(f"slab.frame_{group}_consistency",
                                     "Slab/frame model consistency", int(matches), 1,
                                     comparison="=="))
    dcr = record.get("dcr", {})
    for member in ("column", "beam"):
        checks.append(make_check(f"candidate.{member}_strength_screen",
                                 "Preliminary current member-strength implementation",
                                 dcr.get(member), 1.0, units="DCR"))
    drift = record.get("drift_screen")
    try:
        drift_matches = isinstance(drift, dict) and drift.get("analysis_input_sha256") == analysis_input_signature(record)
    except (KeyError, ValueError, TypeError):
        drift_matches = False
    if drift_matches and drift.get("checks"):
        checks.extend(drift["checks"])
    else:
        checks.append(not_evaluated("demands.drift_and_stability_results", "ASCE 7-22 12.8.6--12.8.7",
                                    "Independent QEx/QEy drift/stability results are not available."))
    checks = _apply_capacity_design_evidence(record, checks, recomputed)
    checks.append(not_evaluated("qualification.strength_model_verification", "ACI 318-19 Chapters 6, 18, 21, 22",
                                "Verify biaxial P-M capacity, force signs and joint-face actions, beam axial effects, "
                                "strength reduction factors and cracked-stiffness assumptions independently."))
    checks = _apply_verification_assertions(record, checks)
    return {"schema_version": QUALIFICATION_VERSION, "scope": SCOPE.copy(),
            "implementation_stage": "partial_not_release_ready", "checks": checks,
            **summarize_checks(checks)}


def require_accepted_design(record):
    """Recompute the checklist; a cached/hand-edited accepted=True is insufficient."""
    qualification = qualify_design(record)
    if not qualification["accepted"]:
        counts = qualification["counts"]
        raise RuntimeError(
            f"SMRF design not qualified: {counts['fail']} failed and "
            f"{counts['not_evaluated']} unevaluated checks. No time-history analysis was launched. "
            "Review design.json qualification and SMRF_METHODOLOGY.md. "
            "This implementation is not yet release-ready for dataset generation."
        )
    return qualification


def ensure_generation_release_ready():
    if not GENERATION_RELEASE_READY:
        raise RuntimeError(
            "SMRF methodology overhaul is not release-ready. Generation is disabled before "
            "design/analysis work starts. Use --design-only to investigate a candidate in a NEW "
            "output root; no NTHA will run. Complete the open checks in SMRF_METHODOLOGY.md first."
        )
