"""Explicit bounded slab-demand refinement, with retained unsuccessful attempts.

Agreement is a numerical screen. It does not resolve the support model,
floor/frame compatibility, membrane reinforcement, or engineering assertions.

A plan is either explicit (``meshes`` with offsets per bay) or a named
recipe (``recipe`` + ``levels`` + ``max_shells``) resolved for the current
bay dimensions and beam face at every call, so the policy in the request
identity is geometry-independent while the resolved coordinates and their
hashes travel with the evidence. A recipe whose affordable levels are fewer
than two yields an explicit ``unresolved_budget`` result: no solve, no
coarsening, no pass.
"""
from __future__ import annotations

import copy
import math
import time

from Design.SMRF_Floor_Mesh import floor_mesh, nested_refinement, resolve_recipe_plan

EXPLICIT_KEYS = {"meshes", "moment_tolerance", "shear_tolerance", "tolerance_basis"}
RECIPE_KEYS = {"recipe", "levels", "max_shells", "moment_tolerance", "shear_tolerance", "tolerance_basis"}


def _validate_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("Slab refinement policy must be a dictionary")
    keys = set(policy)
    if keys == EXPLICIT_KEYS:
        kind = "explicit"
    elif keys == RECIPE_KEYS:
        kind = "recipe"
    elif "meshes" in keys and "recipe" in keys:
        raise ValueError("Slab refinement takes an explicit mesh plan or a named recipe, not both")
    else:
        raise ValueError(f"Slab refinement requires exactly {sorted(EXPLICIT_KEYS)} or {sorted(RECIPE_KEYS)}")
    for key in ("moment_tolerance", "shear_tolerance"):
        v = policy[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 1:
            raise ValueError("Refinement tolerances must be explicit finite numbers between zero and one")
    if not isinstance(policy["tolerance_basis"], str) or not policy["tolerance_basis"].strip():
        raise ValueError("Refinement requires an explicit tolerance_basis")
    return kind


def _plan(geometry, policy, *, sections=None, recipe_inputs=None):
    """Grids of the plan, plus the recipe resolution (None for an explicit plan).

    A recipe resolves from the live ``sections`` (beam width) or, when
    checking a saved record, from the ``recipe_inputs`` it recorded.
    """
    kind = _validate_policy(policy)
    from Design.SMRF_Floor_Analysis import _integer, _number
    nx, ny = (_integer(geometry[k], k) for k in ("num_bay_x", "num_bay_y"))
    lx, ly = (_number(geometry[k], k) for k in ("bay_x_in", "bay_y_in"))
    resolution = None
    if kind == "explicit":
        specs = policy["meshes"]
        if not isinstance(specs, (list, tuple)) or not 2 <= len(specs) <= 8:
            raise ValueError("Bounded refinement requires between two and eight explicit meshes")
    else:
        if recipe_inputs is not None:
            source = ({"face_widths_in": recipe_inputs["face_widths_in"]} if "face_widths_in" in recipe_inputs
                      else {"b_beam_in": recipe_inputs["beam_width_in"]})
            if (recipe_inputs.get("num_bay_x"), recipe_inputs.get("num_bay_y"), recipe_inputs.get("bay_x_in"),
                    recipe_inputs.get("bay_y_in")) != (nx, ny, lx, ly):
                raise ValueError("Recorded recipe inputs disagree with the geometry")
        elif sections is not None:
            source = sections
        else:
            raise ValueError("A recipe plan needs the beam section to resolve")
        resolution = resolve_recipe_plan({"num_bay_x": nx, "num_bay_y": ny, "bay_x_in": lx, "bay_y_in": ly},
                                         source, policy)
        specs = resolution["meshes"]
    grids = [floor_mesh(nx, ny, lx, ly, 4, mesh_spec=spec) for spec in specs]
    for a, b in zip(grids, grids[1:]):
        nested_refinement(a, b)
    return grids, resolution


def refinement_verified(evidence):
    """Recompute the final screen and bind it to the actual returned demands."""
    from Design.SMRF_Slab_Actions import compare_slab_action_refinement
    if isinstance(evidence, dict) and evidence.get("by_floor") is not None:
        return by_floor_refinement_verified(evidence)
    try:
        report = evidence["refinement"]
        if report["status"] != "passed" or report["all_within_tolerance"] is not True:
            return False
        grids, resolution = _plan(report["geometry"], report["policy"], recipe_inputs=report.get("recipe_inputs"))
        if (resolution is None) != (report.get("resolution") is None):
            return False
        if resolution is not None and (resolution["status"] != "resolved"
                                       or resolution["meshes"] != report["resolved_meshes"]):
            return False
        levels = report["levels"]
        if len(levels) != len(grids) or any(level["status"] != "completed" for level in levels):
            return False
        for level, grid in zip(levels, grids):
            actions = level["actions"]
            meshes = actions["solved_meshes"]
            if [m["case_id"] for m in meshes] != [c["id"] for c in actions["cases"]]:
                return False
            if not meshes or any(m["mesh"] != grid for m in meshes):
                return False
            if not actions["equilibrium"] or not all(r["numerical_balance_passed"] for r in actions["equilibrium"]):
                return False
        coarse, fine = levels[-2]["actions"], levels[-1]["actions"]
        for key in ("analysis_model_sha256", "physical_model_sha256", "slab_input_sha256", "cases",
                    "strips", "solved_meshes", "equilibrium", "shear_recovery", "max_abs_membrane_kip_per_in"):
            if evidence[key] != fine[key]:
                return False
        comparison = compare_slab_action_refinement(
            coarse, fine, moment_tolerance=report["policy"]["moment_tolerance"],
            shear_tolerance=report["policy"]["shear_tolerance"])
        return comparison == report["comparisons"][-1] and comparison["all_within_tolerance"] is True
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


BY_FLOOR_METHOD = "slab_strip_envelope_over_distinct_floors_v1"


def _strip_envelope(floors):
    """Per (panel, axis, face): the largest moment and the largest shear over the floors, each with its own source."""
    rows = {}
    for entry in floors:
        tag = {"floor_sections_sha256": entry["floor_sections_sha256"], "floors": entry["floors"]}
        for row in entry["evidence"].get("strips", []):
            key = (row["panel_id"], row["axis"], row["face"])
            current = rows.get(key)
            if current is None:
                rows[key] = {**row, "moment_source_floor": tag, "shear_source_floor": tag}
                continue
            if row["mu_kip_in_per_ft"] > current["mu_kip_in_per_ft"]:
                current.update(mu_kip_in_per_ft=row["mu_kip_in_per_ft"], moment_location=row.get("moment_location"),
                               moment_source_floor=tag)
            if row["vu_kip_per_ft"] > current["vu_kip_per_ft"]:
                current.update(vu_kip_per_ft=row["vu_kip_per_ft"], shear_location=row.get("shear_location"),
                               shear_basis=row.get("shear_basis"), shear_source_floor=tag)
            current["tension_face_at_shear_verified"] = bool(current.get("tension_face_at_shear_verified")
                                                             and row.get("tension_face_at_shear_verified"))
    strips = []
    for key in sorted(rows):
        row = rows[key]
        row["demand_location"] = (f"{row['panel_id']}: envelope over {len(floors)} mechanically distinct floor(s); moment from floors "
                                  f"{row['moment_source_floor']['floors']} at {row.get('moment_location')}; shear from floors "
                                  f"{row['shear_source_floor']['floors']} ({row.get('shear_basis')}) at {row.get('shear_location')}")
        strips.append(row)
    return strips


def _by_floor_digest(floors):
    import hashlib
    import json
    payload = [[entry["floor_sections_sha256"], entry["floors"], entry["evidence"].get("analysis_model_sha256")] for entry in floors]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def combine_floor_evidence(floors, num_floor):
    """One demand record for the common slab layout from the evidence of every mechanically distinct floor.

    ``floors``: [{"floor_sections_sha256", "floors": [k, ...], "evidence": refined single-floor evidence}].
    A flag is True only when it is True on every floor; ``all_floors_enveloped`` (False on any single
    by-line floor) becomes True here exactly when the floors listed cover every elevated floor once and
    every other flag holds. Nothing is asserted by this function: the engineering assertions are those
    the single-floor records carry, and they must be the same on all of them.
    """
    from Design.SMRF_Slab_Actions import _FLAGS
    if not floors:
        raise ValueError("No floor evidence to combine.")
    served = sorted(k for entry in floors for k in entry["floors"])
    complete = served == list(range(1, num_floor + 1))
    first = floors[0]["evidence"]
    same_assertions = all(entry["evidence"].get("engineering_assertions") == first.get("engineering_assertions") for entry in floors)
    same_inputs = all(entry["evidence"].get("slab_input_sha256") == first.get("slab_input_sha256")
                      and entry["evidence"].get("cases") == first.get("cases") for entry in floors)
    statuses = [entry["evidence"].get("refinement", {}).get("status") for entry in floors]
    passed = all(status == "passed" for status in statuses)
    combined = {flag: all(entry["evidence"].get(flag) is True for entry in floors) for flag in _FLAGS
                if flag not in ("all_floors_enveloped", "verified")}
    others = all(combined.values()) and same_assertions and same_inputs
    asserted = (first.get("engineering_assertions") or {}).get("all_floors_enveloped") is True
    provenance = all(entry["evidence"].get("assertion_provenance_valid") is True for entry in floors)
    combined["all_floors_enveloped"] = bool(complete and same_inputs and same_assertions and asserted and provenance and passed)
    combined["verified"] = bool(others and combined["all_floors_enveloped"]
                                and all(entry["evidence"].get("numerical_preconditions", {}).get("physical_recovery_valid") is True
                                        and entry["evidence"].get("numerical_preconditions", {}).get("mesh_refinement_verified") is True
                                        for entry in floors)
                                and all((first.get("engineering_assertions") or {}).get(flag) is True for flag in _FLAGS))
    failing = next((entry for entry, status in zip(floors, statuses) if status != "passed"), None)
    numerical = dict(first.get("numerical_basis") or {})
    numerical["all_floors_enveloped"] = (
        f"Every mechanically distinct floor of the grouped design solved on its own beam lines and supports "
        f"({len(floors)} distinct floor(s) serving floors {served}); each strip row is the largest moment and the largest "
        "shear over them. One slab thickness, superimposed dead load and live load on every floor including the roof.")
    combined.update({
        "method": first.get("method"), "source": first.get("source"),
        "load_combination_basis": first.get("load_combination_basis"),
        "analysis_model_sha256": _by_floor_digest(floors),
        "physical_model_sha256": None,
        "slab_input_sha256": first.get("slab_input_sha256") if same_inputs else None,
        "shear_envelope_basis": first.get("shear_envelope_basis"), "load_scope": first.get("load_scope"),
        "strips": _strip_envelope(floors),
        "numerical_basis": numerical,
        "engineering_assertions": copy.deepcopy(first.get("engineering_assertions") or {}) if same_assertions else {},
        "numerical_preconditions": {"all_floors_covered_once": complete, "same_slab_inputs_and_cases": same_inputs,
                                    "same_assertions": same_assertions,
                                    "physical_recovery_valid": all(entry["evidence"].get("numerical_preconditions", {})
                                                                   .get("physical_recovery_valid") is True for entry in floors),
                                    "mesh_refinement_verified": passed, "verified": combined["verified"]},
        "assertion_provenance_valid": provenance,
        "cases": copy.deepcopy(first.get("cases", [])), "pattern_rule": first.get("pattern_rule"),
        "equilibrium": [{**row, "floor_sections_sha256": entry["floor_sections_sha256"]}
                        for entry in floors for row in entry["evidence"].get("equilibrium", [])],
        "max_abs_membrane_kip_per_in": max((entry["evidence"].get("max_abs_membrane_kip_per_in", 0.0) for entry in floors), default=0.0),
        "shear_recovery": {"method": (first.get("shear_recovery") or {}).get("method"),
                           "top_rows_recovered_at_face": all((entry["evidence"].get("shear_recovery") or {})
                                                             .get("top_rows_recovered_at_face") is True for entry in floors),
                           "bottom_rows": (first.get("shear_recovery") or {}).get("bottom_rows")},
        "independent_hand_check": False, "units": first.get("units"),
        "refinement": {"required": True, "method": BY_FLOOR_METHOD,
                       "status": "passed" if passed else (failing["evidence"].get("refinement", {}).get("status") or "not_completed"),
                       "status_detail": None if passed else (f"floors {failing['floors']}: "
                                                             f"{failing['evidence'].get('refinement', {}).get('status_detail')}"),
                       "all_within_tolerance": passed, "engineering_verified": False,
                       "tolerance_basis": (first.get("refinement") or {}).get("tolerance_basis"),
                       "by_floor": [{"floor_sections_sha256": entry["floor_sections_sha256"], "floors": entry["floors"],
                                     "status": status} for entry, status in zip(floors, statuses)]},
        "by_floor": floors, "distinct_floor_count": len(floors), "floors_served": served, "num_floor": num_floor,
    })
    return combined


def by_floor_refinement_verified(evidence):
    """Every floor's own refinement verified, the floors covering the building once, and the strips their envelope."""
    try:
        floors = evidence["by_floor"]
        if not floors or sorted(k for entry in floors for k in entry["floors"]) != list(range(1, evidence["num_floor"] + 1)):
            return False
        if len({entry["floor_sections_sha256"] for entry in floors}) != len(floors):
            return False
        for entry in floors:
            single = entry["evidence"]
            if single.get("by_floor") is not None or not refinement_verified(single):
                return False
            recorded = (single["refinement"].get("recipe_inputs") or {})
            if "face_widths_in" not in recorded and single["refinement"].get("resolution") is not None:
                return False                    # a by-line floor's recipe is resolved from its own beam faces
        return (evidence["strips"] == _strip_envelope(floors) and evidence["analysis_model_sha256"] == _by_floor_digest(floors)
                and evidence["refinement"]["status"] == "passed")
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def build_refined_slab_action_evidence_by_floor(slab_record, geometry, description, live_load_ksf, slab_inputs, policy,
                                                *, assertions=None, case_observer=None):
    """Refined strip demands of every mechanically distinct floor of a grouped design, and their envelope.

    ``description`` is Design.SMRF_Floor_Sections.by_floor(). Each distinct floor is solved through
    ``build_refined_slab_action_evidence`` on its own beam lines (mesh graded to its own beam faces); a
    floor whose refinement does not pass is kept in the result with its status, and the combination then
    does not pass either. ``case_observer(floor_signature, level_index, loadcase, result)``.
    """
    from Design import SMRF_Floor_Sections as floor_sections
    floor_sections.validate_by_floor(description, geometry["num_bay_x"], geometry["num_bay_y"], geometry["num_floor"])
    floors = []
    for sha, sections, served in floor_sections.distinct_floors(description):
        observer = None if case_observer is None else (lambda level, case, result, _sha=sha: case_observer(_sha, level, case, result))
        evidence = build_refined_slab_action_evidence(slab_record, geometry, sections, live_load_ksf, slab_inputs, policy,
                                                      assertions=assertions, case_observer=observer)
        floors.append({"floor_sections_sha256": sha, "floors": served, "evidence": evidence})
    return combine_floor_evidence(floors, geometry["num_floor"])


def build_refined_slab_action_evidence(slab_record, geometry, sections, live_load_ksf, slab_inputs,
                                       policy, *, assertions=None, case_observer=None):
    """Run all requested levels, retaining failed comparisons and solve errors.

    ``case_observer(level_index, loadcase, result)`` can persist full raw
    solutions immediately. The returned record always retains attempted
    case summaries, successful demand records and failure reasons. Failed
    fine solves cannot qualify a previous coarse result. Invalid plans are
    rejected before creating any OpenSees domain; a recipe that does not
    resolve to two affordable levels returns unverified evidence without
    solving anything.
    """
    from Design.SMRF_Slab_Actions import build_slab_action_evidence, compare_slab_action_refinement, _FLAGS
    from Design.SMRF_Floor_Sections import is_by_line
    sections_by_line = is_by_line(sections)
    grids, resolution = _plan(geometry, policy, sections=sections)
    report = {"required": True, "method": "bounded_explicit_slab_refinement_v2_recipes",
              "status": "running", "all_within_tolerance": False, "engineering_verified": False,
              "geometry": copy.deepcopy(geometry), "policy": copy.deepcopy(policy),
              "tolerance_basis": policy["tolerance_basis"], "levels": [], "comparisons": [],
              "resolution": copy.deepcopy(resolution),
              "recipe_inputs": None if resolution is None else copy.deepcopy(resolution["inputs"]),
              "resolved_meshes": None if resolution is None else copy.deepcopy(resolution["meshes"])}
    latest = {flag: False for flag in _FLAGS}
    latest.update(strips=[], cases=[], equilibrium=[], numerical_preconditions={},
                  numerical_basis={}, engineering_assertions=copy.deepcopy(assertions or {}))
    specs = policy["meshes"] if resolution is None else resolution["meshes"]
    if resolution is not None and resolution["status"] != "resolved":
        report["status"] = resolution["status"]
        report["status_detail"] = resolution["detail"]
        specs, grids = [], []
    for index, (spec, grid) in enumerate(zip(specs, grids)):
        level = {"index": index, "requested_mesh": grid, "status": "started", "attempted_cases": []}
        report["levels"].append(level)
        start = time.perf_counter()

        def observe(case, result):
            level["attempted_cases"].append({"loadcase": copy.deepcopy(case), **{
                key: copy.deepcopy(result[key]) for key in
                ("status", "error", "analysis_return_code", "mesh", "equilibrium", "transfer_equilibrium")
                if key in result}})
            if case_observer is not None:
                case_observer(index, case, result)

        try:
            actions = build_slab_action_evidence(
                slab_record, geometry, sections, live_load_ksf, slab_inputs,
                uniform_all_floors=not sections_by_line,
                assertions=assertions, mesh_spec=spec, case_observer=observe)
            if not actions["solved_meshes"] or any(m["mesh"] != grid for m in actions["solved_meshes"]):
                raise ValueError("Solved mesh provenance differs from the requested refinement mesh")
            level.update(status="completed", actions=actions)
            latest = actions
            if index:
                report["comparisons"].append(compare_slab_action_refinement(
                    report["levels"][index - 1]["actions"], actions,
                    moment_tolerance=policy["moment_tolerance"], shear_tolerance=policy["shear_tolerance"]))
        except Exception as exc:
            level.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["status"] = "analysis_failed"
            break
        finally:
            level["elapsed_seconds"] = time.perf_counter() - start
    if report["status"] == "running":
        report["all_within_tolerance"] = report["comparisons"][-1]["all_within_tolerance"]
        report["status"] = "passed" if report["all_within_tolerance"] else "comparison_failed"
    evidence = copy.deepcopy(latest)
    evidence["refinement"] = report
    passed = refinement_verified(evidence)
    evidence["numerical_preconditions"]["mesh_refinement_verified"] = passed
    evidence["numerical_preconditions"]["verified"] = (
        passed and evidence["numerical_preconditions"].get("physical_recovery_valid") is True)
    evidence["numerical_basis"]["mesh_refinement_verified"] = (
        f"Final pair of {len(grids)} declared meshes: {report['status']}; {policy['tolerance_basis']}. "
        "Local strip-demand agreement only; not support-model or engineering verification.")
    evidence["verified"] = (
        evidence.get("assertion_provenance_valid") is True
        and all(evidence["engineering_assertions"].get(flag) is True for flag in _FLAGS)
        and evidence["numerical_preconditions"]["verified"])
    return evidence
