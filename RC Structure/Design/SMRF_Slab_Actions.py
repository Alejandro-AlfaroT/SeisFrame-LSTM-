"""Slab strip action evidence from the flexible-beam floor model.

Produces the ``demand_evidence`` that ``SMRF_Slab_Reinforcement`` requires:
one row per (panel, bar axis, face) with the governing factored moment and
support-face shear per foot, resolved for twisting, enveloped over every
Gauss point outside the beam widths and recovered physical-face sample, every factored
combination and every required live-load pattern, on one common floor that
represents all floors.

Load cases (ACI 318-19 5.3.1): 1.4D and 1.2D + 1.6L. Live-load patterns
follow 6.4.3.3: when L <= 0.75 D, full factored live load on all panels
governs (6.4.3.3(a)); otherwise 3/4 of the factored live load on alternate
panels (checkerboards, for positive moments) and on the two panels adjacent
to each interior support line (for negative moments), never less than the
all-panel case (6.4.3.3(b) and (c)).

Twisting is resolved with the Wood-Armer rules for orthogonal x/y
reinforcement, using the floor model's sagging-positive mx, my and raw mxy.
Shear rows: the top-face (support) row carries the transverse shear
recovered AT the beam face in its owning element on the clear-span side.
The raw moment/shear tensor is recovered before Wood-Armer, and the tension
face is checked at that same location. Separate element-side results are
retained at transverse boundaries. The bottom-face row carries the
bounded maximum shear of the recovered cell fields in the pure sagging zone
(Wood-Armer bottom demand positive, top demand zero), which is where the bottom
mat is the longitudinal tension steel the one-way shear strength relies on.
The numerical upper bound must close against a same-location sagging witness;
unclosed recovery is an analysis failure, not verified demand evidence.

The verification flags the strip routine reads are engineering assertions
(``Design.Config.SlabActionAssertions``), made after reviewing this
analysis; this module records the numerical basis of each item
(``numerical_basis``) and combines it with the assertion. A flag is True
only when it is asserted AND its numerical precondition holds. Nothing here
certifies its own output: without assertions the evidence is complete but
unverified, and the strip routine selects no reinforcement.
"""
from __future__ import annotations

import hashlib
import json
import math

from Design import SMRF_Floor_Sections as floor_sections
from Design.SMRF_Floor_Analysis import analyze_floor, transfer_mesh_per_bay
from Design.SMRF_Slab_Recovery import (recover_panel_faces, recover_sagging_shear,
                                     SAGGING_RECOVERY_VERSION, METHOD_VERSION as FACE_RECOVERY_VERSION)
from Design.SMRF_Slab_Reinforcement import slab_input_signature
from Design.SMRF_Common import make_check, not_evaluated, assertion_provenance_valid

METHOD_VERSION = "smrf_slab_actions_physical_faces_sagging_bound_v5"
PATTERN_LIVE_FRACTION = 0.75
PATTERN_RULE_ALL = "ACI 318-19 6.4.3.3(a): L <= 0.75 D, full factored live load on all panels governs"
PATTERN_RULE_PARTIAL = ("ACI 318-19 6.4.3.3(b)/(c): 3/4 factored live load on alternate panels and on the "
                        "panels adjacent to each interior support line, enveloped with the all-panel case")
_FLAGS = ("verified", "analysis_applicability_verified", "all_floors_enveloped",
          "load_pattern_envelope_verified", "spatial_envelope_per_unit_width",
          "twisting_moment_resolution_verified", "zero_membrane_force_verified")


def wood_armer(mx, my, mxy):
    """Design moments per unit width for orthogonal x/y bars.

    Inputs are sagging-positive bending resultants and the twisting
    resultant, any consistent unit. Returns nonnegative demands
    ``(bottom_x, bottom_y, top_x, top_y)`` in the same unit.
    """
    t = abs(mxy)
    # Bottom (sagging) steel.
    bx, by = mx + t, my + t
    if bx < 0 and by < 0:
        bx = by = 0.0
    elif bx < 0:
        bx, by = 0.0, my + (t * t / abs(mx) if mx else 0.0)
    elif by < 0:
        bx, by = mx + (t * t / abs(my) if my else 0.0), 0.0
    # Top (hogging) steel: the design moments are negative; report magnitudes.
    tx, ty = mx - t, my - t
    if tx > 0 and ty > 0:
        tx = ty = 0.0
    elif tx > 0:
        tx, ty = 0.0, my - (t * t / abs(mx) if mx else 0.0)
    elif ty > 0:
        tx, ty = mx - (t * t / abs(my) if my else 0.0), 0.0
    return max(0.0, bx), max(0.0, by), max(0.0, -tx), max(0.0, -ty)


def slab_load_cases(dead_ksf, live_ksf, nx, ny):
    """Factored slab cases and the pattern rule that produced them."""
    cases = [{"id": "1.4D", "dead_factor": 1.4, "live_factor": 0.0, "live_pattern": "none"},
             {"id": "1.2D+1.6L_all", "dead_factor": 1.2, "live_factor": 1.6, "live_pattern": "all"}]
    if live_ksf <= PATTERN_LIVE_FRACTION * dead_ksf:
        return cases, PATTERN_RULE_ALL
    partial = 1.6 * PATTERN_LIVE_FRACTION
    even = [[i, j] for j in range(ny) for i in range(nx) if (i + j) % 2 == 0]
    odd = [[i, j] for j in range(ny) for i in range(nx) if (i + j) % 2 == 1]
    cases.append({"id": "1.2D+1.2L_even", "dead_factor": 1.2, "live_factor": partial, "live_pattern": even})
    cases.append({"id": "1.2D+1.2L_odd", "dead_factor": 1.2, "live_factor": partial, "live_pattern": odd})
    for line in range(1, ny):
        panels = [[i, j] for j in (line - 1, line) for i in range(nx)]
        cases.append({"id": f"1.2D+1.2L_adjacent_x_line_{line}", "dead_factor": 1.2,
                      "live_factor": partial, "live_pattern": panels})
    for line in range(1, nx):
        panels = [[i, j] for i in (line - 1, line) for j in range(ny)]
        cases.append({"id": f"1.2D+1.2L_adjacent_y_line_{line}", "dead_factor": 1.2,
                      "live_factor": partial, "live_pattern": panels})
    return cases, PATTERN_RULE_PARTIAL


def _model_signature(slab_record, geometry, sections, mesh, case_ids):
    payload = {"method": METHOD_VERSION, "slab": slab_record, "geometry": geometry,
               "sections": sections, "mesh_per_bay": mesh, "support_model": "flexible_beams",
               "cases": case_ids}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_slab_action_evidence(slab_record, geometry, sections, live_load_ksf, slab_inputs,
                               mesh_per_bay=None, uniform_all_floors=True, assertions=None,
                               *, mesh_spec=None, case_observer=None):
    """Slab strip demand evidence for ``design_slab_reinforcement``.

    ``sections`` needs the beam section and column footprint (as for the
    floor transfer) so beam widths can be excluded from the strip envelope.
    ``slab_inputs`` is the strip routine's input dictionary; its signature
    binds this evidence to that slab. ``uniform_all_floors`` must be True for
    the single-floor envelope to represent every floor. ``assertions`` is a
    mapping of the engineering assertions (see Design.Config
    .SlabActionAssertions); absent assertions leave every flag False.
    """
    by_line = floor_sections.is_by_line(sections)
    if not uniform_all_floors and not by_line:
        raise ValueError("Slab action evidence for non-uniform floors is not implemented.")
    if by_line and uniform_all_floors:
        raise ValueError("A floor described line by line is one floor of a grouped design: its evidence cannot claim to "
                         "represent every floor (pass uniform_all_floors=False; SMRF_Slab_Refinement combines the floors).")
    nx, ny = geometry["num_bay_x"], geometry["num_bay_y"]
    lx, ly = geometry["bay_x_in"], geometry["bay_y_in"]
    # Half width of the beam on every X line (at y = j ly) and every Y line (at x = i lx).
    half_x_lines = [floor_sections.beam_width(sections, "x", j) / 2.0 for j in range(ny + 1)]
    half_y_lines = [floor_sections.beam_width(sections, "y", i) / 2.0 for i in range(nx + 1)]
    half_beam = ({"x_lines": half_x_lines, "y_lines": half_y_lines} if by_line else sections["b_beam_in"] / 2.0)
    dead_ksf = (slab_record["concrete_unit_weight_kcf"] * slab_record["thickness_in"] / 12.0
                + slab_record["superimposed_dead_load_ksf"])
    mesh = (4 if mesh_spec is not None else transfer_mesh_per_bay(nx, ny)) if mesh_per_bay is None else mesh_per_bay
    cases, pattern_rule = slab_load_cases(dead_ksf, live_load_ksf, nx, ny)
    for case in cases:
        case["live_load_ksf"] = live_load_ksf
    envelope = {}   # (panel_id, axis, face) -> {"mu", "vu", "location"}
    max_membrane = 0.0
    equilibrium = []
    face_coverage = []
    solved_meshes = []
    for case in cases:
        kwargs = {"mesh_spec": mesh_spec} if mesh_spec is not None else {}
        try:
            result = analyze_floor(slab_record, geometry, sections, case, mesh_per_bay=mesh,
                                   support_model="flexible_beams", **kwargs)
        except Exception as exc:
            if case_observer is not None:
                case_observer(case, {"status": "exception", "error": f"{type(exc).__name__}: {exc}"})
            raise
        if case_observer is not None:
            case_observer(case, result)
        solved_meshes.append({"case_id": case["id"], "mesh": result.get("mesh")})
        if result["status"] != "transfer_complete":
            raise RuntimeError(f"Slab action case {case['id']} failed: status {result['status']}.")
        max_membrane = max(max_membrane, result["max_abs_membrane_kip_per_in"])
        equilibrium.append({"case_id": case["id"], **result["equilibrium"]})
        for panel in result["panels"]:
            points = panel["gauss_point_resultants"]
            # Moments: Wood-Armer envelope over the points outside the beam widths.
            for point in points:
                x, y = point["x_in"], point["y_in"]
                if (any(abs(y - j * ly) < half_x_lines[j] for j in range(ny + 1))
                        or any(abs(x - i * lx) < half_y_lines[i] for i in range(nx + 1))):
                    continue        # over a beam; not a slab strip section
                bx, by, tx, ty = wood_armer(point["mx"], point["my"], point["mxy_raw"])
                for (axis, face), moment in ((("x", "bottom"), bx), (("y", "bottom"), by),
                                             (("x", "top"), tx), (("y", "top"), ty)):
                    key = (panel["panel_id"], axis, face)
                    entry = envelope.setdefault(key, {"mu": 0.0, "vu": 0.0, "mu_location": None, "vu_location": None,
                                                      "shear_basis": None, "tension_face_verified": True})
                    mu = 12.0 * moment
                    if mu > entry["mu"]:
                        entry.update(mu=mu, mu_location={"case_id": case["id"], "x_in": x, "y_in": y,
                                                         "element": point["element"], "gauss_point": point["gauss_point"]})
            # Recover the raw tensor in the cell containing each physical face.
            # Preserve GP extrema and also sample the faces; do not average away
            # local peaks or use a nearest GP to infer the face's tension side.
            recovered = (recover_panel_faces(panel, geometry, None,
                                             face_widths=floor_sections.panel_face_widths(sections, panel["i"], panel["j"]))
                         if by_line else recover_panel_faces(panel, geometry, sections["b_beam_in"]))
            face_coverage.append({"case_id": case["id"], "panel_id": panel["panel_id"],
                                  "faces": recovered["coverage"], "complete": recovered["complete"]})
            for sample in recovered["samples"]:
                raw = sample["raw_resultants"]
                wa = wood_armer(raw["mx"], raw["my"], raw["mxy_raw"])
                location = {"case_id": case["id"], **sample, "recovery_method": FACE_RECOVERY_VERSION}
                for (moment_axis, face), moment in zip(
                        (("x", "bottom"), ("y", "bottom"), ("x", "top"), ("y", "top")), wa):
                    key = (panel["panel_id"], moment_axis, face)
                    entry = envelope.setdefault(key, {"mu": 0.0, "vu": 0.0, "mu_location": None, "vu_location": None,
                                                      "shear_basis": None, "tension_face_verified": True})
                    if 12.0 * moment > entry["mu"]:
                        entry.update(mu=12.0 * moment, mu_location=location)
                axis = sample["axis"]
                entry = envelope[(panel["panel_id"], axis, "top")]
                top_demand = wa[2 if axis == "x" else 3]
                vu = 12.0 * abs(raw["qx_raw" if axis == "x" else "qy_raw"])
                # Retain zero-shear coverage too; a missing sample and a true
                # zero are different. A governing non-top tension face remains
                # unresolved rather than being assigned bottom steel implicitly.
                valid_tension_face = top_demand > 0.0 or vu == 0.0
                tied = math.isclose(vu, entry["vu"], rel_tol=1e-12, abs_tol=0.0)
                if entry["shear_basis"] is None or (vu > entry["vu"] and not tied) or (tied and not valid_tension_face):
                    entry.update(vu=max(vu, entry["vu"]), shear_basis="raw tensor recovery in the physical face's clear-span cell",
                                 tension_face_verified=valid_tension_face,
                                 vu_location={**location, "top_demand_at_face_kip_in_per_in": top_demand})
                elif tied:
                    entry["vu"] = max(vu, entry["vu"])
            sagging_shear = recover_sagging_shear(
                panel, geometry, floor_sections.panel_face_widths(sections, panel["i"], panel["j"]))
            for axis in ("x", "y"):
                # Resolve the sign boundary in the recovered field; selecting
                # only sagging GPs makes the sampled region jump with the mesh.
                key = (panel["panel_id"], axis, "bottom")
                entry = envelope.setdefault(key, {"mu": 0.0, "vu": 0.0, "mu_location": None, "vu_location": None,
                                                  "shear_basis": None, "tension_face_verified": True})
                recovery = sagging_shear[axis]
                vu = 12.0 * recovery["shear_upper_kip_per_in"]
                if vu > entry["vu"]:
                    entry.update(vu=vu, shear_basis=SAGGING_RECOVERY_VERSION,
                                 vu_location={"case_id": case["id"], **(recovery["witness"] or {}),
                                              "recovery": recovery})
    if not envelope:
        raise RuntimeError("No slab Gauss points lie outside the beam widths; refine the mesh.")
    strips = []
    face_recovery_complete = True
    for (panel_id, axis, face), entry in sorted(envelope.items()):
        if face == "top" and entry.get("shear_basis") is None:
            face_recovery_complete = False
        if not entry.get("tension_face_verified", True):
            face_recovery_complete = False
        strips.append({"panel_id": panel_id, "axis": axis, "face": face,
                       "mu_kip_in_per_ft": entry["mu"], "vu_kip_per_ft": entry["vu"],
                       "demand_location": (f"{panel_id}: Wood-Armer {face} {axis} envelope over Gauss points outside beam "
                                           f"widths and recovered physical faces; moment at {entry['mu_location']}; "
                                           f"shear: {entry.get('shear_basis')} at {entry['vu_location']}"),
                       "moment_location": entry["mu_location"], "shear_location": entry["vu_location"],
                       "shear_basis": entry.get("shear_basis"),
                       "tension_face_at_shear_verified": bool(entry.get("tension_face_verified", True))})
    membrane_free = max_membrane <= 1e-9
    balanced = all(item["numerical_balance_passed"] for item in equilibrium)
    asserted = dict(assertions or {})
    numerical = {
        "analysis_applicability_verified": ("Linear elastic ShellMITC4 plate on elastic ACI 8.4.1.8 T-beams, uniform area "
                                            "gravity only, no openings/drops/concentrated loads (SMRF_Floor_Analysis)."),
        "all_floors_enveloped": ("Uniform slab thickness, superimposed dead load and live load on every floor including the "
                                 "roof, so one common floor represents all floors."),
        "load_pattern_envelope_verified": pattern_rule,
        "spatial_envelope_per_unit_width": ("Maximum over Gauss points outside beam widths and physical-face samples "
                                            "in their clear-span cells; raw tensor recovery before Wood-Armer; no strip averaging."),
        "twisting_moment_resolution_verified": "Wood-Armer resolution of mx, my, mxy into orthogonal x/y top and bottom demands.",
        "zero_membrane_force_verified": f"In-plane DOFs fixed; max |p11,p22,p12| = {max_membrane:.3e} kip/in.",
        "verified": (f"Numerical only: every case in equilibrium to 1e-8 ({balanced}), membrane-free response "
                     f"({membrane_free}), valid governing face recovery ({face_recovery_complete}). "
                     "Engineering verification is asserted in Design.Config.SlabActionAssertions "
                     "after an independent review (methodology item 7); it is not set by this module."),
    }
    if by_line:
        numerical["all_floors_enveloped"] = ("This record is ONE floor of a grouped design (its own beam lines and supports). "
                                             "The envelope over floors exists only in the by-floor combination of every "
                                             "mechanically distinct floor (SMRF_Slab_Refinement).")
    preconditions = {"zero_membrane_force_verified": membrane_free,
                     **({"all_floors_enveloped": False} if by_line else {}),
                     "mesh_refinement_verified": False,
                     "physical_recovery_valid": membrane_free and balanced and face_recovery_complete,
                     "verified": False}
    numerical["mesh_refinement_verified"] = "Single mesh only; a completed bounded refinement comparison is required."
    provenance_ok = assertion_provenance_valid(asserted)
    evidence = {flag: provenance_ok and asserted.get(flag) is True and preconditions.get(flag, True)
                for flag in _FLAGS}
    evidence.update({
        "method": "plate_finite_element",
        "source": f"Design/SMRF_Slab_Actions.py {METHOD_VERSION}; Design/SMRF_Floor_Analysis.py flexible_beams",
        "load_combination_basis": "ACI 318-19 5.3.1 (1.4D; 1.2D+1.6L) with " + pattern_rule,
        "analysis_model_sha256": _model_signature(slab_record, geometry, sections, solved_meshes, cases),
        "physical_model_sha256": _model_signature(slab_record, geometry, sections, None, cases),
        "slab_input_sha256": slab_input_signature(slab_inputs),
        "shear_envelope_basis": "support_face_maximum",
        "load_scope": "uniform_area_gravity_no_concentrated_loads",
        "strips": strips,
        "numerical_basis": numerical,
        "engineering_assertions": {key: asserted.get(key) for key in (*_FLAGS, "asserted_by", "assertion_date", "assertion_basis")},
        "numerical_preconditions": preconditions,
        "assertion_provenance_valid": provenance_ok,
        "cases": [{k: v for k, v in case.items()} for case in cases],
        "pattern_rule": pattern_rule,
        "mesh_per_bay": mesh if mesh_spec is None else None,
        "solved_meshes": solved_meshes,
        "refinement": {"required": True, "status": "not_requested", "all_within_tolerance": False},
        "beam_half_width_excluded_in": half_beam,
        "physical_face_coverage": face_coverage,
        "shear_recovery": {"method": FACE_RECOVERY_VERSION,
                           "top_rows_recovered_at_face": face_recovery_complete,
                           "bottom_rows": SAGGING_RECOVERY_VERSION},
        "equilibrium": equilibrium,
        "max_abs_membrane_kip_per_in": max_membrane,
        "independent_hand_check": False,
        "units": {"moment": "kip-in/ft", "shear": "kip/ft"},
    })
    return evidence


def compare_slab_action_refinement(coarse, fine, *, moment_tolerance, shear_tolerance):
    """Compare every strip at fixed physical inputs; never assert verification.

    Caller supplies investigation tolerances. A shared physical-model identity,
    exact load cases and strip inventory are required. Near-zero changes are not
    hidden by an arbitrary denominator floor: zero-to-nonzero remains unstable.
    This result describes agreement of sampled demands, not a rigorous error
    bound or proof that a physical floor idealization is applicable.
    """
    for value in (moment_tolerance, shear_tolerance):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("Provide explicit finite refinement tolerances between zero and one")
    if not isinstance(coarse, dict) or not isinstance(fine, dict):
        raise ValueError("Two saved slab action evidence records are required")
    identity = coarse.get("physical_model_sha256")
    if not isinstance(identity, str) or len(identity) != 64 or fine.get("physical_model_sha256") != identity:
        raise ValueError("Refinement requires identical physical-model and method identities")
    if coarse.get("cases") != fine.get("cases") or coarse.get("slab_input_sha256") != fine.get("slab_input_sha256"):
        raise ValueError("Refinement requires identical loads, patterns and slab inputs")
    if coarse.get("analysis_model_sha256") == fine.get("analysis_model_sha256"):
        raise ValueError("Refinement requires distinct analysis/mesh identities")
    def rows(evidence):
        result = {}
        for row in evidence.get("strips", []):
            key = (row["panel_id"], row["axis"], row["face"])
            if key in result:
                raise ValueError("Duplicate strip in refinement evidence")
            for metric in ("mu_kip_in_per_ft", "vu_kip_per_ft"):
                v = row[metric]
                if isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0:
                    raise ValueError("Refinement demands must be finite and nonnegative")
            result[key] = row
        if not result:
            raise ValueError("Refinement requires explicit nonempty strip inventories")
        return result
    a,b = rows(coarse),rows(fine)
    if set(a) != set(b):
        raise ValueError("Refinement strip inventories differ")
    comparisons = []
    for key in sorted(a):
        for metric,tolerance in (("mu_kip_in_per_ft",moment_tolerance),("vu_kip_per_ft",shear_tolerance)):
            before,after = a[key][metric],b[key][metric]
            change = abs(after-before)/before if before else 0.0 if after == 0 else None
            comparisons.append({"panel_id":key[0],"axis":key[1],"face":key[2],"metric":metric,
                "coarse":before,"fine":after,"relative_change":change,"tolerance":tolerance,
                "within_tolerance":change is not None and change<=tolerance})
    return {"method":"slab_strip_refinement_comparison_v1","physical_model_sha256":identity,
            "coarse_analysis_sha256":coarse.get("analysis_model_sha256"),
            "fine_analysis_sha256":fine.get("analysis_model_sha256"),
            "all_within_tolerance":all(row["within_tolerance"] for row in comparisons),
            "engineering_verified":False,"comparisons":comparisons}


def evaluate_slab_actions(evidence):
    """Qualification checks for saved slab action evidence."""
    if not isinstance(evidence, dict):
        return [not_evaluated("floor.qualified_slab_actions", "ACI 318-19 Chapters 6 and 8",
                              "No slab action evidence was generated.")]
    checks = []
    from Design.SMRF_Slab_Refinement import refinement_verified
    mesh_ok = refinement_verified(evidence)
    assertions = evidence.get("engineering_assertions") or {}
    flags_ok = (mesh_ok and assertion_provenance_valid(assertions)
                and all(assertions.get(flag) is True and evidence.get(flag) is True for flag in _FLAGS))
    details = {"numerical_basis": evidence.get("numerical_basis"), "pattern_rule": evidence.get("pattern_rule"),
               "cases": [c["id"] for c in evidence.get("cases", [])],
               "engineering_assertions": evidence.get("engineering_assertions")}
    if flags_ok:
        checks.append(make_check("floor.qualified_slab_actions", "ACI 318-19 5.3.1, 6.4.3.3, 8.4; Wood-Armer",
                                 1, 1, "==", details=details))
    else:
        missing = [flag for flag in _FLAGS if evidence.get(flag) is not True]
        if not mesh_ok:
            missing.append("completed_mesh_refinement")
        if not assertion_provenance_valid(assertions):
            missing.append("named_dated_assertion_basis")
        missing.extend("assertion." + flag for flag in _FLAGS if assertions.get(flag) is not True)
        checks.append({**not_evaluated("floor.qualified_slab_actions", "ACI 318-19 5.3.1, 6.4.3.3, 8.4; Wood-Armer",
                                       f"Slab action evidence is computed but not asserted as verified: {missing}. "
                                       "Assert in Design.Config.SlabActionAssertions after independent review."),
                       "details": {**details, "reason": f"unasserted or unmet: {missing}"}})
    balanced = all(item.get("numerical_balance_passed") for item in evidence.get("equilibrium", [])) \
        and bool(evidence.get("equilibrium"))
    checks.append(make_check("floor.slab_action_equilibrium", "Reaction/first-moment balance of every slab case",
                             int(balanced), 1, "=="))
    if mesh_ok:
        checks.append(make_check("floor.slab_action_mesh_refinement", "Declared numerical investigation tolerances",
                                 1, 1, "==", details={"basis": evidence["refinement"]["tolerance_basis"],
                                                      "engineering_verified": False}))
    else:
        checks.append(not_evaluated("floor.slab_action_mesh_refinement", "Declared numerical investigation tolerances",
                                    "A complete, identity-matched refinement comparison has not passed."))
    checks.append(not_evaluated("floor.independent_hand_verification", "Engineering review",
                                "Hand-check a representative floor's strip moments, shears and beam transfer against the model."))
    recovery = evidence.get("shear_recovery") or {}
    if recovery.get("top_rows_recovered_at_face") is True:
        checks.append(make_check("floor.support_face_shear_recovery", "Slab design section / shear recovery",
                                 1, 1, "==", details={"method": recovery.get("method"),
                                                      "bottom_rows": recovery.get("bottom_rows"),
                                                      "basis": "top rows: raw resultants recovered in the element containing "
                                                               "the physical face, on its clear-span side; all four faces covered; "
                                                               "tension face determined at the same location before enveloping"}))
    else:
        checks.append(not_evaluated("floor.support_face_shear_recovery", "Slab design section / shear recovery",
                                    "Face shear was not recovered for every support row, or a face's tension side "
                                    "was not the top; review the shear_location details."))
    return checks
