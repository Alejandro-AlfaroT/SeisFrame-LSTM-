"""Bare frame with the exported transfer against the monolithic coupled model.

The design frame is a centerline frame loaded by the floor transfer
(SMRF_Floor_Transfer). SMRF_Coupled_Analysis solves the same building with
shells on every floor, eccentric downstand webs and full-height columns in
one model, with no transfer step. Comparing the two under the same factored
gravity case (1.2D + 1.6L, all panels, member weight included) measures how
much of the coupled model's load path the transfer reproduces:

* column base vertical reactions, column by column (the load path);
* column base moments (eccentricity and interface-couple effects, which the
  bare frame carries only through the exported couples);
* the total, which must match to round-off because both models carry the
  same slab pressure and member weight ledgers.

Both models are run at gross stiffness so the comparison isolates the
transfer from the cracked-stiffness assumption of the design model. The
result is an evaluated check with a stated tolerance on the vertical load
path; the residual idealization (beam/slab eccentricity, joint flexibility,
membrane action) is a judgement recorded in
``Design.Config.IndependentVerification.floor_frame_compatibility_reviewed``.
"""
from __future__ import annotations

import contextlib
import io
import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Gravity import run_gravity_analysis
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity
from Design.SMRF_Elastic import build_design_model
from Design.SMRF_Floor_Analysis import MAX_SHELLS
from Loads.Gravity_Loads import apply_gravity_loads
from Model.Build_Model import node_tag

METHOD_VERSION = "smrf_transfer_vs_coupled_gravity_comparison_v2_pattern_and_mesh"
DEFAULT_TOLERANCE = 0.05          # per-column vertical reaction, relative to the coupled model
DEAD_FACTOR, LIVE_FACTOR = 1.2, 1.6


def comparison_mesh_per_bay(num_bay_x, num_bay_y, num_floor, preferred=8, minimum=4):
    for mesh in range(preferred, minimum - 1, -2):
        if num_floor * num_bay_x * num_bay_y * mesh * mesh <= MAX_SHELLS:
            return mesh
    raise ValueError(f"No coupled mesh of at least {minimum}/bay fits {MAX_SHELLS} shells for this building.")


def _frame_base_reactions(live_pattern="all"):
    """Bare design frame, gross stiffness, transfer loads at 1.2D + 1.6L; base reactions per column."""
    saved = (sp.BEAM_STIFFNESS_MODIFIER, sp.COLUMN_STIFFNESS_MODIFIER)
    try:
        sp.BEAM_STIFFNESS_MODIFIER, sp.COLUMN_STIFFNESS_MODIFIER = 1.0, 1.0
        ops.wipe()
        with contextlib.redirect_stdout(io.StringIO()):
            build_design_model()
            # The comparison is of the slab-to-frame load path: both models carry the slab pressure and the
            # framed members. The column extensions above the roof are a direct column load outside it.
            apply_gravity_loads(floor_factor=1.0, self_weight_factor=DEAD_FACTOR, dead_factor=DEAD_FACTOR,
                                live_factor=LIVE_FACTOR, live_pattern=live_pattern, roof_extension=False)
            run_gravity_analysis()
            ops.reactions()
        reactions = {}
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                node = node_tag(0, i, j)
                reactions[(i, j)] = [float(ops.nodeReaction(node, dof)) for dof in (3, 4, 5)]
        return reactions
    finally:
        sp.BEAM_STIFFNESS_MODIFIER, sp.COLUMN_STIFFNESS_MODIFIER = saved
        ops.wipe()


def moment_significance(rows, combination_actions):
    """How much the base-moment disagreement could matter to the column design.

    For each column kind: the coupled model's gravity base moment, the
    largest factored base moment the frame's own combinations put on those
    columns (seismic cases included), and the ratio. A moment gap of 20% on
    a gravity moment that is 5% of the governing design moment is a 1%
    matter; the number is recorded so the compatibility judgement can be
    made against the demand it affects, not in the abstract.
    """
    if not combination_actions:
        return None
    per_story = (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    governing = {}
    for action in combination_actions:
        for tag, member in action["members"].items():
            tag = int(tag)
            if member["member_type"] != "column" or tag > per_story:      # first-story columns only
                continue
            f = member["local_force_kip_kipin"]
            base = math.hypot(f[4], f[5])                                 # moment at end i (the base)
            governing[tag] = max(governing.get(tag, 0.0), base)
    by_kind = {}
    for row in rows:
        tag = row["grid_j"] * (sp.NUM_BAY_X + 1) + row["grid_i"] + 1
        entry = by_kind.setdefault(row["kind"], {"coupled_gravity_kip_in": 0.0, "frame_gravity_kip_in": 0.0,
                                                 "governing_design_kip_in": 0.0, "count": 0})
        entry["coupled_gravity_kip_in"] += row["coupled_base_moment_kip_in"]
        entry["frame_gravity_kip_in"] += row["frame_base_moment_kip_in"]
        entry["governing_design_kip_in"] += governing.get(tag, 0.0)
        entry["count"] += 1
    for entry in by_kind.values():
        g = entry["governing_design_kip_in"]
        entry["gravity_gap_over_governing"] = (abs(entry["coupled_gravity_kip_in"] - entry["frame_gravity_kip_in"]) / g
                                               if g > 1e-9 else None)
        entry["coupled_gravity_over_governing"] = entry["coupled_gravity_kip_in"] / g if g > 1e-9 else None
    return {"basis": ("first-story column base moments: coupled-model gravity moment, bare-frame gravity moment and "
                      "the largest factored base moment over every strength combination the frame was designed for; "
                      "the gap over the governing moment is what the compatibility judgement affects"),
            "by_kind": by_kind}


def coupled_sections_by_member():
    """By-member sections of the installed grouped design for SMRF_Coupled_Analysis, at gross stiffness."""
    from Design import SMRF_Floor_Sections as floor_sections
    from Design.SMRF_Coupled_Analysis import BY_MEMBER_SCHEMA
    from Model import Member_Groups as mg
    grades = {m.design.fc_ksi for m in mg.all_members("column")}
    if len(grades) != 1:
        raise mg.GroupedStateError(f"Column groups use different concrete grades {sorted(grades)}; this revision keeps one.")
    return {"schema": BY_MEMBER_SCHEMA, "floors": [floor_sections.for_floor(k) for k in range(1, sp.NUM_FLOOR + 1)],
            "fc_col_ksi": grades.pop(), "beam_stiffness_modifier": 1.0, "column_stiffness_modifier": 1.0}


def compare_transfer_to_coupled(slab_record, tolerance=DEFAULT_TOLERANCE, mesh_per_bay=None,
                                combination_actions=None):
    """Run both models from the current Structure_Parameters state and compare.

    Requires sp.FLOOR_TRANSFER for the current sections/slab. Returns a
    JSON-safe record with per-column differences and the evaluated outcome.
    ``combination_actions`` (the frame's solved strength combinations) adds
    the moment-significance evidence.
    """
    if sp.FLOOR_TRANSFER is None or sp.SLAB_THICKNESS_IN is None:
        raise ValueError("The coupled comparison needs the slab-aware state with a floor transfer.")
    geometry = {"num_bay_x": sp.NUM_BAY_X, "num_bay_y": sp.NUM_BAY_Y, "num_floor": sp.NUM_FLOOR,
                "bay_x_in": sp.BAY_X, "bay_y_in": sp.BAY_Y, "story_h_in": sp.STORY_H}
    from Model import Member_Groups as mg
    if mg.is_grouped():
        # The coupled reference resolves the installed members: every floor's own beam lines and the
        # columns of every story (below each floor; the column above a floor is the next floor's).
        sections = coupled_sections_by_member()
    else:
        sections = {"b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI,
                    "b_col_in": sp.B_COL, "h_col_in": sp.H_COL, "fc_col_ksi": sp.FC_COL_KSI,
                    "beam_stiffness_modifier": 1.0, "column_stiffness_modifier": 1.0}
    mesh = mesh_per_bay or comparison_mesh_per_bay(sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR)
    case = {"id": "gravity_strength_all", "dead_factor": DEAD_FACTOR, "live_factor": LIVE_FACTOR,
            "live_load_ksf": sp.FLOOR_LIVE_LOAD_KSF, "live_pattern": "all"}
    def coupled_rows(coupled_case, mesh_per_bay):
        ops.wipe()
        coupled = analyze_coupled_gravity(slab_record, geometry, sections, [coupled_case] * sp.NUM_FLOOR,
                                          mesh_per_bay=mesh_per_bay)
        ops.wipe()
        if coupled.get("status") != "diagnostic_complete":
            raise RuntimeError(f"Coupled diagnostic did not complete: {coupled.get('status')}.")
        return coupled, {(r["grid_i"], r["grid_j"]): r["force_moment"] for r in coupled["base_reactions"]}

    def compare(frame, coupled_base):
        rows = []
        for key in sorted(frame):
            fv, fmx, fmy = frame[key]
            cv, cmx, cmy = coupled_base[key][2], coupled_base[key][3], coupled_base[key][4]
            relative = (fv - cv) / cv if cv else float("inf")
            i, j = key
            kind = ("corner" if (i in (0, sp.NUM_BAY_X)) and (j in (0, sp.NUM_BAY_Y))
                    else "edge" if (i in (0, sp.NUM_BAY_X)) or (j in (0, sp.NUM_BAY_Y)) else "interior")
            rows.append({"grid_i": i, "grid_j": j, "kind": kind,
                         "frame_vertical_kip": fv, "coupled_vertical_kip": cv, "vertical_relative_difference": relative,
                         "frame_base_moment_kip_in": math.hypot(fmx, fmy),
                         "coupled_base_moment_kip_in": math.hypot(cmx, cmy)})
        return rows

    coupled, coupled_base = coupled_rows(case, mesh)
    rows = compare(_frame_base_reactions(), coupled_base)
    summary = summarize_columns(rows)

    # Asymmetric loading: the first saved ACI 6.4.2 pattern, both models.
    from Design.SMRF_Demands import live_load_patterns
    saved_cases = (sp.FLOOR_TRANSFER or {}).get("unit_cases", {})
    if mg.is_grouped():
        # Every floor's transfer carries the same pattern inventory (one live-pattern list per design).
        saved_cases = next(iter((sp.FLOOR_TRANSFER or {}).get("transfers", {}).values()), {}).get("unit_cases", {})
    pattern = next((p for p in live_load_patterns(sp.NUM_BAY_X, sp.NUM_BAY_Y)
                    if f"live_pattern_{p['id']}" in saved_cases), None)
    asymmetric = None
    if pattern is not None:
        pattern_case = {**case, "id": f"gravity_strength_{pattern['id']}", "live_pattern": pattern["panels"]}
        _coupled_p, base_p = coupled_rows(pattern_case, mesh)
        rows_p = compare(_frame_base_reactions(live_pattern=pattern["id"]), base_p)
        asymmetric = {"pattern": pattern, "case": pattern_case, "columns": rows_p, **summarize_columns(rows_p)}

    # Mesh sensitivity of the coupled reference itself: the next coarser even mesh.
    mesh_check = None
    if mesh - 2 >= 4:
        _coupled_c, base_c = coupled_rows(case, mesh - 2)
        change = max(abs((base_c[k][2] - coupled_base[k][2]) / coupled_base[k][2]) if coupled_base[k][2] else float("inf")
                     for k in coupled_base)
        moment_change = max(abs((math.hypot(base_c[k][3], base_c[k][4]) - math.hypot(coupled_base[k][3], coupled_base[k][4]))
                                / math.hypot(coupled_base[k][3], coupled_base[k][4]))
                            if math.hypot(coupled_base[k][3], coupled_base[k][4]) > 1e-9 else 0.0 for k in coupled_base)
        mesh_check = {"meshes_per_bay": [mesh - 2, mesh],
                      "max_column_vertical_relative_change": change,
                      "max_column_base_moment_relative_change": moment_change}

    return {
        "method_version": METHOD_VERSION, "coupled_method_version": coupled["method_version"],
        "case": case, "mesh_per_bay": mesh, "stiffness_basis": "gross in both models",
        "tolerance_vertical_relative": tolerance,
        **summary,
        "columns": rows,
        "asymmetric": asymmetric,
        "coupled_mesh_sensitivity": mesh_check,
        "moment_significance": moment_significance(rows, combination_actions),
        "coupled_equilibrium": coupled["equilibrium"],
        "coupled_weight_ledger": coupled["weight_ledger"],
        "vertical_path_within_tolerance": (summary["max_column_vertical_relative_difference"] <= tolerance
                                           and summary["total_relative_error"] < 1e-6
                                           and (asymmetric is None or
                                                (asymmetric["max_column_vertical_relative_difference"] <= tolerance
                                                 and asymmetric["total_relative_error"] < 1e-6))),
        "basis": ("bare centerline frame with the exported transfer (forces, bending couples as in-element force "
                  "pairs, torsion couples at the joints) versus the monolithic shell/web/column model, 1.2D + 1.6L "
                  "with member weight, both at gross stiffness; all panels and the first ACI 6.4.2 pattern; "
                  "vertical reactions column by column are the evaluated quantity, base moments and the coupled "
                  "model's own mesh sensitivity are recorded for the compatibility judgement"),
        "not_compared": ["beam end moments (the coupled model shares them between web and slab)",
                         "floor displacements", "cracked-stiffness state", "joint flexibility", "membrane action"],
    }


def validate_rows(rows, num_bay_x, num_bay_y):
    """Every column exactly once, finite values; raise otherwise."""
    if not isinstance(rows, list):
        raise ValueError("column rows must be a list")
    expected = {(i, j) for i in range(num_bay_x + 1) for j in range(num_bay_y + 1)}
    seen = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("column row must be a mapping")
        key = (row.get("grid_i"), row.get("grid_j"))
        seen.append(key)
        for field in ("frame_vertical_kip", "coupled_vertical_kip", "frame_base_moment_kip_in", "coupled_base_moment_kip_in"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"column {key} has a nonfinite {field}")
        if row.get("kind") not in ("corner", "edge", "interior"):
            raise ValueError(f"column {key} has an unknown kind")
    if len(seen) != len(set(seen)):
        raise ValueError("duplicate column identities")
    if set(seen) != expected:
        raise ValueError(f"column inventory {sorted(set(seen))} does not match the plan {sorted(expected)}")
    return rows


def summarize_columns(rows):
    """Recompute the comparison summary from its per-column rows."""
    frame_total = sum(r["frame_vertical_kip"] for r in rows)
    coupled_total = sum(r["coupled_vertical_kip"] for r in rows)
    worst = max((abs((r["frame_vertical_kip"] - r["coupled_vertical_kip"]) / r["coupled_vertical_kip"])
                 if r["coupled_vertical_kip"] else float("inf")) for r in rows) if rows else float("inf")
    total_error = abs(frame_total - coupled_total) / coupled_total if coupled_total else float("inf")
    moment_ratio = {}
    for kind in ("corner", "edge", "interior"):
        group = [r for r in rows if r["kind"] == kind]
        cm = sum(r["coupled_base_moment_kip_in"] for r in group)
        moment_ratio[kind] = (sum(r["frame_base_moment_kip_in"] for r in group) / cm) if cm > 1e-9 else None
    return {"frame_total_vertical_kip": frame_total, "coupled_total_vertical_kip": coupled_total,
            "total_relative_error": total_error, "max_column_vertical_relative_difference": worst,
            "base_moment_ratio_frame_over_coupled": moment_ratio}


def evaluate_coupled_comparison(record):
    """Qualification checks from a saved comparison; the judgement item stays separate.

    The summary figures are recomputed from the saved per-column rows, and
    the comparison must carry the signature of the frame inputs it was run
    on; a comparison from other inputs, or one whose rows no longer support
    its summary, is not evidence.
    """
    from Design.SMRF_Common import make_check, not_evaluated
    from Design.SMRF_Design_Evidence import analysis_input_signature
    comparison = record.get("coupled_comparison")
    review = not_evaluated("floor.compatibility_idealization_reviewed", "Engineering review",
                           "Beam/slab eccentricity, joint flexibility and membrane action are not represented by the "
                           "bare frame; assert floor_frame_compatibility_reviewed after review.")
    if not isinstance(comparison, dict) or comparison.get("method_version") != METHOD_VERSION:
        return [not_evaluated("floor.coupled_frame_compatibility", "Slab-to-frame load path vs the coupled model",
                              "No transfer-vs-coupled comparison of this method version was saved with this design."),
                review]
    if comparison.get("analysis_input_sha256") != analysis_input_signature(record):
        return [not_evaluated("floor.coupled_frame_compatibility", "Slab-to-frame load path vs the coupled model",
                              "The saved comparison does not carry this frame's input signature."), review]
    nx, ny = record["geometry"]["num_bay_x"], record["geometry"]["num_bay_y"]
    try:
        rows = validate_rows(comparison.get("columns"), nx, ny)
        summary = summarize_columns(rows)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return [not_evaluated("floor.coupled_frame_compatibility", "Slab-to-frame load path vs the coupled model",
                              f"The saved comparison rows are unusable: {exc}"), review]
    stale = [key for key in ("max_column_vertical_relative_difference", "total_relative_error")
             if not math.isclose(summary[key], comparison.get(key, float("nan")), rel_tol=1e-9, abs_tol=1e-12)]
    checks = [
        make_check("floor.coupled_frame_compatibility", "Slab-to-frame load path vs the coupled model",
                   summary["max_column_vertical_relative_difference"], comparison["tolerance_vertical_relative"],
                   "<=", "relative", details={"total_relative_error": summary["total_relative_error"],
                                               "base_moment_ratio_frame_over_coupled": summary["base_moment_ratio_frame_over_coupled"],
                                               "mesh_per_bay": comparison["mesh_per_bay"], "basis": comparison["basis"],
                                               "recomputed_from_rows": True,
                                               "saved_summary_stale": stale}),
        make_check("floor.coupled_total_load", "Both models carry the same slab pressure and member weight",
                   summary["total_relative_error"], 1e-6, "<=", "relative"),
        make_check("floor.coupled_summary_consistent", "Evidence integrity", len(stale), 0, "==",
                   details={"stale_fields": stale}),
    ]
    asymmetric = comparison.get("asymmetric")
    if isinstance(asymmetric, dict):
        try:
            asym_rows = validate_rows(asymmetric.get("columns"), nx, ny)
            asym = summarize_columns(asym_rows)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            checks.append(not_evaluated("floor.coupled_frame_compatibility_pattern",
                                        "Slab-to-frame load path vs the coupled model, asymmetric pattern",
                                        f"The saved pattern comparison rows are unusable: {exc}"))
            checks.append(not_evaluated("floor.coupled_total_load_pattern",
                                        "Both models carry the same pattern load", "see the pattern comparison rows"))
        else:
            pattern_id = (asymmetric.get("pattern") or {}).get("id")
            checks.append(make_check("floor.coupled_frame_compatibility_pattern",
                                     "Slab-to-frame load path vs the coupled model, asymmetric pattern",
                                     asym["max_column_vertical_relative_difference"], comparison["tolerance_vertical_relative"],
                                     "<=", "relative", details={"pattern": pattern_id,
                                                                 "base_moment_ratio_frame_over_coupled": asym["base_moment_ratio_frame_over_coupled"],
                                                                 "recomputed_from_rows": True}))
            checks.append(make_check("floor.coupled_total_load_pattern",
                                     "Both models carry the same pattern load",
                                     asym["total_relative_error"], 1e-6, "<=", "relative", details={"pattern": pattern_id}))
    else:
        checks.append(not_evaluated("floor.coupled_frame_compatibility_pattern",
                                    "Slab-to-frame load path vs the coupled model, asymmetric pattern",
                                    "No asymmetric-pattern comparison was saved (the transfer carries no live pattern)."))
        checks.append(not_evaluated("floor.coupled_total_load_pattern",
                                    "Both models carry the same pattern load",
                                    "No asymmetric-pattern comparison was saved (the transfer carries no live pattern)."))
    checks += [
        not_evaluated("floor.compatibility_idealization_reviewed", "Engineering review",
                      "Beam/slab eccentricity, joint flexibility and membrane action are not represented by the bare "
                      "frame (see base_moment_ratio_frame_over_coupled and the asymmetric/mesh evidence); "
                      "assert floor_frame_compatibility_reviewed after review."),
    ]
    return checks
