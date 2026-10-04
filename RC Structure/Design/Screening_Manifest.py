"""Explicit screening manifests for Design/Verify_Designs: validated case lists, a per-case overview, the
four separately reported statuses, and the optional gravity/modal stage on the nonlinear model (2026-10-01).

A manifest (``seisframe_v2_screening_plan_v1``) names the cases to design instead of a
slice of the shuffled generation plan. It is validated against the running code before anything is
launched: its design basis must be the configured risk basis, its nonlinear profile must be a registered
profile whose settings it restates correctly, and every case must lie on the population's geometry grid.
A contradiction stops the launch; nothing is adjusted to make a manifest fit.

The runner keeps four things apart, because one is routinely mistaken for another:
  numerical_completion        did each stage run to its end (a design search that exhausts its budget
                              is a completed search with an unfavourable stop reason, not an error);
  design_checks               what the qualification checklist found (pass / fail / not evaluated);
  independent_verification    whether a person asserted the items code cannot check about itself; PROBE
                              values exercise the path and are never a verification;
  production_acceptance       False while GENERATION_RELEASE_READY is False, whatever the above say.

The gravity/modal stage builds the nonlinear model of the selected design under the manifest's profile,
runs the reference modal analysis, gravity and the post-gravity modal analysis exactly as the production
NTHA does, verifies the INSTALLED topology against the profile, checks the installed design against the
record, and reads every hinge's moment and rotation at the gravity state through the export path. It
runs no ground motion.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from pathlib import Path

MANIFEST_SCHEMA = "seisframe_v2_screening_plan_v1"
CASE_KEYS = ("case_id", "num_bay_x", "num_bay_y", "num_floor", "story_height_ft", "bay_x_width_ft", "bay_y_width_ft", "seismic_site")
STAGES = ("design", "gravity_modal", "figures")
_CASE_ID = re.compile(r"^case_\d{4}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _problem_list(manifest, ranges, sites, increment_ft):
    import Structure_Parameters as sp
    from Model.Analysis_Profile import PROFILES
    problems = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema is {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}")
    profile_id = manifest.get("profile_id")
    if profile_id not in PROFILES:
        problems.append(f"profile_id {profile_id!r} is not a registered analysis profile ({sorted(PROFILES)})")
    basis = manifest.get("design_basis") or {}
    if basis.get("risk_category") != sp.ASCE_RISK_CATEGORY:
        problems.append(f"design_basis.risk_category is {basis.get('risk_category')!r}; the configured basis is {sp.ASCE_RISK_CATEGORY!r}")
    try:
        sp.seismic_design_basis(risk_category=sp.ASCE_RISK_CATEGORY, importance_factor=basis.get("importance_factor"))
        if basis.get("importance_factor") is None:
            problems.append("design_basis.importance_factor is missing")
    except (ValueError, TypeError) as exc:
        problems.append(f"design_basis.importance_factor: {exc}")
    step = basis.get("bay_and_story_height_increment_ft")
    if step is not None and abs(float(step) - increment_ft) > 1e-12:
        problems.append(f"design_basis increment is {step} ft; the population grid is {increment_ft} ft")
    for key in ("num_bay_x", "num_bay_y", "num_floor", "story_height_ft", "bay_x_width_ft", "bay_y_width_ft"):
        declared = basis.get(key)
        code = [min(ranges[key]), max(ranges[key])]
        if declared is not None and [float(v) for v in declared] != [float(v) for v in code]:
            problems.append(f"design_basis.{key} is {declared}; the population range is {code}")
    declared_profile = manifest.get("nonlinear_profile") or {}
    if profile_id in PROFILES:
        settings = PROFILES[profile_id]["settings"]
        for manifest_key, setting in (("element_formulation", "ELEMENT_FORMULATION"), ("member_material", "IMK_MATERIAL_TYPE"),
                                      ("apply_to_beams", "IMK_APPLY_TO_BEAMS"), ("apply_to_columns", "IMK_APPLY_TO_COLUMNS"),
                                      ("joint_model", "JOINT_MODEL")):
            if manifest_key in declared_profile and declared_profile[manifest_key] != settings[setting]:
                problems.append(f"nonlinear_profile.{manifest_key} is {declared_profile[manifest_key]!r}; "
                                f"profile {profile_id} sets {settings[setting]!r}")
        if declared_profile.get("explicit_face_slip_interfaces", False) != PROFILES[profile_id]["explicit_face_slip_interfaces"]:
            problems.append("nonlinear_profile.explicit_face_slip_interfaces disagrees with the profile")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        return problems + ["cases is missing or empty"]
    seen_ids, seen_names = set(), set()
    for index, case in enumerate(cases):
        where = f"cases[{index}]"
        if not isinstance(case, dict):
            problems.append(f"{where} is not an object")
            continue
        cid = case.get("case_id")
        where = f"{where} ({cid})"
        if not isinstance(cid, str) or not _CASE_ID.match(cid):
            problems.append(f"{where}: case_id must look like case_0001")
        elif cid in seen_ids:
            problems.append(f"{where}: duplicate case_id")
        seen_ids.add(cid)
        name = case.get("geometry_name")
        if not isinstance(name, str) or not _SAFE_NAME.match(name):
            problems.append(f"{where}: geometry_name must be a nonempty name of letters, digits, '_', '.', '-'")
        elif name in seen_names:
            problems.append(f"{where}: duplicate geometry_name")
        seen_names.add(name)
        for key in ("num_bay_x", "num_bay_y", "num_floor"):
            value = case.get(key)
            if type(value) is not int:
                problems.append(f"{where}: {key} must be an integer, got {value!r}")
            elif value not in ranges[key]:
                problems.append(f"{where}: {key} = {value} is outside the population {min(ranges[key])}..{max(ranges[key])}")
        for key in ("story_height_ft", "bay_x_width_ft", "bay_y_width_ft"):
            value = case.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{where}: {key} must be a number, got {value!r}")
            elif float(value) not in [float(v) for v in ranges[key]]:
                problems.append(f"{where}: {key} = {value} ft is not a level of the population "
                                f"({min(ranges[key])}..{max(ranges[key])} ft in {increment_ft}-ft steps)")
        if case.get("seismic_site") not in sites:
            problems.append(f"{where}: seismic_site {case.get('seismic_site')!r} is not one of {list(sites)}")
    execution = manifest.get("execution") or {}
    if execution.get("initial_case_id") not in seen_ids:
        problems.append(f"execution.initial_case_id {execution.get('initial_case_id')!r} is not a manifest case")
    for key in ("initial_workers", "subsequent_max_workers", "max_section_iter"):
        if type(execution.get(key)) is not int or execution[key] < 1:
            problems.append(f"execution.{key} must be a positive integer")
    return problems


def load_manifest(path):
    """Read and validate a screening manifest; return the launch description. A contradiction raises."""
    import Structure_Parameters as sp
    from Generate_Parameterized_Dataset import GEOMETRY_INCREMENT_FT, POPULATION_BASIS, RANGES, SEISMIC_SITES
    from Model.Analysis_Profile import PROFILES
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    problems = _problem_list(manifest, RANGES, SEISMIC_SITES, GEOMETRY_INCREMENT_FT)
    if problems:
        raise ValueError(f"Screening manifest {path} is not valid for the running code:\n  - " + "\n  - ".join(problems))
    cases = []
    for case in manifest["cases"]:
        entry = {key: case[key] for key in CASE_KEYS}
        entry["geometry_name"] = case["geometry_name"]
        for key in ("label", "purpose"):
            if key in case:
                entry[key] = case[key]
        cases.append(entry)
    execution = manifest["execution"]
    return {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "schema": manifest["schema"],
        "profile_id": manifest["profile_id"], "profile_declarations": PROFILES[manifest["profile_id"]]["declarations"],
        "cases": cases, "initial_case_id": execution["initial_case_id"],
        "initial_workers": execution["initial_workers"], "subsequent_max_workers": execution["subsequent_max_workers"],
        "max_section_iter": execution["max_section_iter"], "output_root": execution.get("output_root"),
        "fresh_root_required": bool(execution.get("fresh_root_required")),
        "design_basis": sp.seismic_design_basis(), "population_basis": POPULATION_BASIS,
        "sampling": {"method": "explicit_manifest_cases_v1", "seed": None,
                     "selection": "the cases listed in the manifest, in manifest order; not sampled",
                     "bounds": {key: [min(values), max(values)] for key, values in RANGES.items()},
                     "discrete_mapping": f"every case lies on the population grid ({GEOMETRY_INCREMENT_FT}-ft length steps, integer counts)",
                     "manifest_sha256": hashlib.sha256(raw).hexdigest()},
        "status_in_manifest": manifest.get("status"), "scope": manifest.get("scope"),
        "acceptance_policy": manifest.get("acceptance_policy"),
    }


# ---- what a result row says about a design ------------------------------------------------------------
FAILURE_GROUPS = (("strong_column_weak_beam", ("scwb",)), ("joint", ("joint",)),
                  ("anchorage_and_bar_fit", ("anchorage", "thread", "development", "splice", "bar_clear", "stacking")),
                  ("drift_and_stability", ("demands.story_drift", "demands.stability", "demands.second_order")),
                  ("capacity_shear", ("capacity_shear",)), ("slab_and_floor", ("slab", "floor")))


def group_ids(ids):
    """Check ids grouped by subject for the summary; an id belongs to the first group it matches."""
    groups = {name: [] for name, _ in FAILURE_GROUPS}
    groups["other"] = []
    for check_id in sorted(set(ids)):
        for name, tokens in FAILURE_GROUPS:
            if any(token in check_id for token in tokens):
                groups[name].append(check_id)
                break
        else:
            groups["other"].append(check_id)
    return {name: values for name, values in groups.items() if values}


def design_overview(record):
    """Geometry, hazard, basis, selection, period and base shear, utilisation, drift and stability,
    refinement evidence of one design record. Values are read; a missing one is None, never assumed."""
    geometry, seismic, sections = record.get("geometry") or {}, record.get("seismic") or {}, record.get("sections") or {}
    reinforcement, search, demand = record.get("reinforcement") or {}, record.get("search") or {}, record.get("demand") or {}
    basis, loads, dcr = record.get("demand_basis") or {}, record.get("floor_loads") or {}, record.get("dcr") or {}
    drift = record.get("drift_screen") or {}
    stories = drift.get("stories") or []
    weight = loads.get("total_floor_seismic_weight_kip")
    total_weight = loads.get("total_seismic_weight_kip")            # per-floor weights plus the roof column extensions
    if total_weight is None:                                        # a record made before the extension was weighed
        total_weight = weight * geometry["num_floor"] if weight is not None and geometry.get("num_floor") else None
    base_shear = demand.get("base_shear_kip")

    def worst(rows, value, limit):
        rows = [r for r in rows if r.get(value) is not None and r.get(limit)]
        if not rows:
            return None
        row = max(rows, key=lambda r: r[value] / r[limit])
        return {"location": row.get("location"), value: row[value], limit: row[limit], "ratio_to_limit": row[value] / row[limit]}

    refinement = (record.get("slab_actions") or {}).get("refinement") or {}
    floor = record.get("floor_analysis") or {}
    slab_reinforcement = record.get("slab_reinforcement") or {}
    torsion = basis.get("torsion") or {}
    strength = ((basis.get("regularity") or {}).get("lateral_strength_distribution") or {}) if isinstance(basis.get("regularity"), dict) else {}
    return {
        "geometry": {**geometry, "bay_x_ft": None if geometry.get("bay_x_in") is None else geometry["bay_x_in"] / 12.0,
                     "bay_y_ft": None if geometry.get("bay_y_in") is None else geometry["bay_y_in"] / 12.0,
                     "story_h_ft": None if geometry.get("story_h_in") is None else geometry["story_h_in"] / 12.0},
        "hazard": {key: seismic.get(key) for key in ("site_label", "sds", "sd1", "s1", "r")},
        "design_basis": {"risk_category": seismic.get("risk_category"), "importance_factor": seismic.get("importance_factor"),
                         "drift": {key: (basis.get("drift") or {}).get(key) for key in (
                             "base_drift_limit_ratio", "effective_drift_limit_ratio", "rho_for_limit", "cd",
                             "seismic_design_category", "limit_seismic_design_category", "low_rise_allowance_asserted", "limit_scope")},
                         "occupancy": (basis.get("policy") or {}).get("occupancy"), "site_class": (basis.get("policy") or {}).get("site_class")},
        "sections": sections,
        "reinforcement": {key: reinforcement.get(key) for key in (
            "col_bar_size", "col_top_bars", "col_bot_bars", "col_side_bars", "col_stirrup_bar_size", "col_stirrup_legs",
            "col_stirrup_spacing_in", "beam_bar_size", "beam_top_bars", "beam_bot_bars", "beam_side_bars",
            "beam_stirrup_bar_size", "beam_stirrup_legs", "beam_stirrup_spacing_in")},
        "slab_thickness_in": (record.get("slab") or {}).get("thickness_in"),
        "search": {key: search.get(key) for key in ("stop_reason", "stop_detail", "selected_iteration", "last_iteration",
                                                    "iterations_evaluated", "max_section_iter")},
        "period_and_base_shear": {"model_period_sec": demand.get("model_period_sec"), "design_period_sec": basis.get("design_period_sec"),
                                  "period_basis": basis.get("period_basis"), "base_shear_kip": base_shear,
                                  "seismic_weight_kip": total_weight,
                                  "cs_base_shear_over_weight": base_shear / total_weight if base_shear is not None and total_weight else None},
        "member_utilisation": {key: dcr.get(key) for key in ("column", "beam", "governing", "governed_by", "band_lo", "band_hi",
                                                             "beam_in_band", "column_within_ceiling")},
        "scwb": {**{key: (record.get("scwb") or {}).get(key) for key in ("ratio_min", "ratio_provided", "screen_satisfied", "satisfied")},
                 "joint_min_ratio_provided": ((record.get("scwb") or {}).get("joint_check") or {}).get("min_ratio_provided"),
                 "ratio_provided_basis": "preliminary section proxy; use joint_min_ratio_provided for the evaluated joint rule"},
        "drift_and_stability": {"worst_story_drift": worst(stories, "drift_ratio", "allowable_drift_ratio"),
                                "worst_stability": worst(stories, "theta", "theta_limit"),
                                "drift_screen_accepted": drift.get("accepted"), "drift_screen_counts": drift.get("counts")},
        "torsion": {"tir": torsion.get("tir", torsion.get("max_drift_ratio")), "torsional_irregularity": torsion.get("torsional_irregularity")},
        # The story-strength model (review item M1) per case, so a screening run of several structures shows the
        # evidence an assertion would rest on (user decision 2026-10-02: assert after a multi-structure screen).
        "story_strength": {"one_side_fraction": strength.get("one_side_fraction"), "model": strength.get("model"),
                           "applicability_status": (strength.get("applicability") or {}).get("status"),
                           "type_1_strength_threshold": 0.75},
        "mesh_and_refinement": {
            "floor_transfer_mesh_per_bay": (record.get("floor_transfer") or {}).get("mesh_per_bay"),
            "floor_analysis": {key: floor.get(key) for key in ("status", "verified", "mesh_per_bay", "refinement_mesh_per_bay", "reason", "errors")},
            "slab_refinement": {key: refinement.get(key) for key in ("required", "method", "status", "all_within_tolerance",
                                                                     "engineering_verified", "policy", "unresolved", "reason", "levels_solved")},
            "slab_reinforcement": {key: slab_reinforcement.get(key) for key in ("stage", "accepted", "screen_passed")},
            "coupled_max_vertical_difference": (record.get("coupled_comparison") or {}).get("max_column_vertical_relative_difference"),
        },
        "gravity_failures_during_search": len(record.get("gravity_failures") or []),
    }


def status_fields(record, probe, probe_date, stages_run, stage_results=None):
    """The four statuses, kept apart. ``stage_results`` maps stage name to its result dict."""
    from Design.SMRF_Qualification import GENERATION_RELEASE_READY
    qualification = record.get("qualification") or {}
    failed = sorted({str(i).split(":")[0] for i in qualification.get("failed") or []})
    open_items = sorted({str(i).split(":")[0] for i in qualification.get("not_evaluated") or []})
    search = record.get("search") or {}
    verification = ((record.get("request_identity") or {}).get("policy") or {}).get("verification") or {}
    flags = {key: value for key, value in verification.items() if isinstance(value, bool)}
    stage_results = stage_results or {}
    refinement = ((record.get("slab_actions") or {}).get("refinement") or {})
    return {
        "numerical_completion": {
            "design": "completed",
            "design_search_stop_reason": search.get("stop_reason"),
            "design_search_budget_exhausted": search.get("stop_reason") == "iteration_budget_exhausted",
            "slab_refinement_status": refinement.get("status"),
            **{name: (stage_results.get(name) or {}).get("status", "not_run") for name in STAGES[1:]},
            "stages_requested": list(stages_run),
            "note": "a stage that ran to its end is completed whatever it found; a stop reason is evidence, not an error",
        },
        "design_checks": {
            "accepted_by_checklist": bool(qualification.get("accepted")), "counts": qualification.get("counts"),
            "failed": group_ids(failed), "not_evaluated": group_ids(open_items),
            "note": "the qualification checklist of the saved record under its own assertions; not a certification",
        },
        "independent_verification": {
            "mode": "PROBE assertions (pipeline exercise; NOT a verification)" if probe else "committed Design/Config.py",
            "probe_date": probe_date if probe else None,
            "asserted_by": verification.get("asserted_by"),
            "flags": flags,
            "verified_by_a_person": False if probe else bool(flags) and all(flags.values()),
            "story_strength_model_verified": bool(flags.get("story_strength_model_verified")),
            "unverified_items": sorted(key for key, value in flags.items() if not value) if not probe else sorted(flags),
            "note": ("PROBE values fill the assertion blocks so the assertion-dependent design path runs; every item they touch "
                     "remains unverified" if probe else "flags are the committed assertions of Design/Config.py"),
        },
        "production_acceptance": {
            "accepted": False if (probe or not GENERATION_RELEASE_READY) else bool(qualification.get("accepted")),
            "generation_release_ready": bool(GENERATION_RELEASE_READY),
            "reason": ("GENERATION_RELEASE_READY is False" if not GENERATION_RELEASE_READY else
                       "PROBE assertions cannot accept a design" if probe else "checklist result under the committed assertions"),
        },
    }


# ---- the gravity / modal stage on the nonlinear model --------------------------------------------------
def figures_stage(record, profile_id, out_dir):
    """The three figures of the selected design (Design/Design_Figures): the structure in 3D, the member cross
    sections with their reinforcement, the joint detailing with the joint shear table. Drawn from the record
    alone; never raises: the observed failure is the result."""
    started = time.perf_counter()
    result = {"status": "started", "profile_id": profile_id}
    try:
        from Design.Design_Figures import draw_design_figures
        out_dir = Path(out_dir)
        drawn = draw_design_figures(record, out_dir / "figures", label=out_dir.name)
        result.update({"status": "completed", "files": [str(Path(f).relative_to(out_dir)) for f in drawn["files"]],
                       "notes": drawn["notes"]})
    except Exception as exc:                                  # noqa: BLE001
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    result["elapsed_s"] = time.perf_counter() - started
    return result


def gravity_modal_stage(record, profile_id, out_dir):
    """Nonlinear build of the selected design under the profile; reference modal, gravity, post-gravity
    modal; installed-topology and installed-design verification; hinge moment-rotation at the gravity
    state through the export path. No ground motion. Never raises: the observed failure is the result."""
    started = time.perf_counter()
    result = {"status": "started", "profile_id": profile_id}
    try:
        import openseespy.opensees as ops
        import Structure_Parameters as sp
        import Ground_Motion_Main as gm
        from Analysis import Hinge_Hysteresis_Diagnostic as hd
        from Analysis import Hinge_Moment_Rotation as hmr
        from Analysis import NTHA as ntha
        from Design import Design_Driver as driver
        from Model import Analysis_Profile as ap

        from Design.Grouped_Record import apply_record
        apply_record(record)                                # a uniform or a grouped record, each by its own rules
        if profile_id:
            ap.apply_profile(profile_id)
        result["profile"] = ap.profile_identity()
        gravity, reference_modal, post_modal, reference_check, tangent_check = gm.build_gravity_modal_state()
        result["installed_topology"] = ap.verify_installed_domain(profile_id)
        result["installed_design_verification"] = hd.verify_installed_design(record)
        audit, inventory, hinges = hd.run_audit()
        measured = hd.measured_gravity_state(inventory)
        valid = [m for m in reference_modal if m.get("valid")]
        result["modal"] = {"periods_elastic_reference_sec": [m["period"] for m in valid[:6]],
                           "periods_post_gravity_tangent_sec": [m["period"] for m in post_modal if m.get("valid")][:3],
                           "design_model_period_sec": (record.get("demand") or {}).get("model_period_sec"),
                           "asce_reference_period_check": reference_check, "asce_post_gravity_period_check": tangent_check,
                           "note": ("the reference periods are of the nonlinear model's elastic state (hinges elastic, rigid centerline "
                                    "joints); the design period is of the elastic cracked design model")}
        result["gravity"] = {"analysis": gravity, **measured,
                             "basis": "apply_gravity_loads() at floor factor 1.0 (D + L), member self-weight; the production NTHA gravity state"}
        tags = ntha._imk_hinge_element_tags()
        rotation, moment, missing = ntha._hinge_history_rows(tags, ntha._query_hinge_states(tags))
        pseudo = {"hinge_tag_order": tags, "status": {"completed_steps": 0, "npts_requested": 0, "failed": False},
                  "hinge_rotation_history": [], "hinge_moment_history": [], "hinge_history_time": [], "hinge_rotation_steps": [],
                  "hinge_history_commit_count": [], "hinge_history_strategy": [],
                  "hinge_history_stride": int(getattr(sp, "NTHA_HINGE_HISTORY_STRIDE", 8)),
                  "hinge_gravity_state": {"time": float(ops.getTime()), "rotation": rotation, "moment": moment, "missing_values": missing}}
        export_dir = Path(out_dir) / "gravity_modal"
        counts = hmr.write_hinge_moment_rotation(export_dir, pseudo)
        arrays, rows, schema = hmr.read_hinge_moment_rotation(export_dir)
        import numpy as np
        finite = bool(arrays is not None and np.all(np.isfinite(arrays["gravity_moment_kip_in"])) and np.all(np.isfinite(arrays["gravity_rotation_rad"])))
        result["hinge_export_check"] = {
            "directory": str(export_dir), **counts, "hinges_in_domain": len(tags), "springs_in_table": len(rows),
            "hinges_in_inventory": len(hinges), "missing_values_at_gravity": missing, "all_gravity_values_finite": finite,
            "max_abs_gravity_moment_kip_in": None if arrays is None else float(np.nanmax(np.abs(arrays["gravity_moment_kip_in"]))),
            "max_abs_gravity_rotation_rad": None if arrays is None else float(np.nanmax(np.abs(arrays["gravity_rotation_rad"]))),
            "schema_version": schema.get("schema_version"),
            "note": "gravity state only: the schema and the spring table of the case with no transient rows (no ground motion was run)"}
        result["model_audit"] = audit
        checks = {"profile_installed": bool(result["installed_topology"]["consistent"]),
                  "installed_design_matches_record": bool(result["installed_design_verification"].get("consistent")),
                  "gravity_converged": bool(gravity) and not (isinstance(gravity, dict) and gravity.get("failed")),
                  "hinge_export_complete": finite and missing == 0 and len(rows) == 2 * len(tags) and len(tags) == len(hinges)}
        result["checks"] = checks
        result["status"] = "completed"
        result["all_checks_pass"] = all(checks.values())
        ops.wipe()
    except Exception as exc:                                # noqa: BLE001 -- the observed failure is the result
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    result["elapsed_s"] = time.perf_counter() - started
    return result


# The later stages by name, in the order STAGES lists them (Design/Verify_Designs runs them after the design).
STAGE_RUNNERS = {"gravity_modal": gravity_modal_stage, "figures": figures_stage}
