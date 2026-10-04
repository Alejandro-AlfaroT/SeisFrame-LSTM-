"""Attach actual joint evidence and a clearly unqualified shell-floor diagnostic.

Called after the chosen frame candidate and all of its load cases are saved.
The caller owns/clears its OpenSees frame domain before the separate floor
solver is entered. No time-history analysis or dataset generation occurs here.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from Design.SMRF_Common import not_evaluated
from Design.SMRF_Demands import strength_load_combinations, live_load_patterns
from Design.SMRF_Joint_Adapter import build_joint_evidence
from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement


def capacity_state_from_record(record):
    """The capacity-design input state, rebuilt from a saved record alone.

    Mirrors Design_Driver._capacity_state, which reads the live
    Structure_Parameters, so qualification can recompute the capacity
    design from what the artifact says was designed and compare it with
    what the artifact says was found. Column axial and shear envelopes come
    from the saved combination actions; beam self-weight from the saved
    sections, slab and unit weight (drop below the slab, clear span).
    """
    geometry, sections, rebar = record["geometry"], record["sections"], record["reinforcement"]
    materials, slab = record["materials"], record["slab"]
    unit_weight = slab["concrete_unit_weight_kcf"] / 1728.0
    b_beam, h_beam, thickness = sections["b_beam_in"], sections["h_beam_in"], slab["thickness_in"]
    drop_weight = unit_weight * b_beam * (h_beam - thickness)
    self_weight = {"x": drop_weight * (1.0 - sections["h_col_in"] / geometry["bay_x_in"]),
                   "y": drop_weight * (1.0 - sections["b_col_in"] / geometry["bay_y_in"])}
    cover = rebar["col_longitudinal_centroid_offset_in"]
    h_col, ab = sections["h_col_in"], rebar["col_bar_area_in2"]
    side = int(rebar["col_side_bars"])
    layers = [(rebar["col_top_bars"] * ab, cover)]
    layers += [(2 * ab, cover + (h_col - 2 * cover) * k / (side + 1)) for k in range(1, side + 1)]
    layers += [(rebar["col_bot_bars"] * ab, h_col - cover)]
    # Bending through b (the y frame): corners plus side-face bars in the outer
    # layers, one top and one bottom bar at each interior top/bottom position
    # (Design_Driver._col_steel_layers_about_z).
    b_col = sections["b_col_in"]
    outer = (2 + side) * ab
    layers_about_z = [(outer, cover)]
    for count in (max(2, int(rebar["col_top_bars"])), max(2, int(rebar["col_bot_bars"]))):
        layers_about_z += [(ab, cover + (b_col - 2 * cover) * k / (count - 1)) for k in range(1, count - 1)]
    layers_about_z.append((outer, b_col - cover))
    layers_about_z.sort(key=lambda layer: layer[1])
    per_story = (geometry["num_bay_x"] + 1) * (geometry["num_bay_y"] + 1)
    from Design.SMRF_Capacity_Design import (column_action_envelopes, COLUMN_SHEAR_METHOD_DEFAULT,
                                             CLEAR_HEIGHT_CONVENTION_DEFAULT)
    envelopes = column_action_envelopes((record.get("design_actions") or {}).get("combinations", []), per_story)
    axial, shear = envelopes["axial"], envelopes["shear"]
    # The Ve rule the saved evidence was produced with. Records written
    # before the method was named carry none and are read as the
    # joint-limited method they were produced with, never relabelled.
    saved_capacity = record.get("capacity_design") or {}
    saved_columns = saved_capacity.get("columns") or {}
    # The orthogonal beam bar stacking the record was designed with (2026-09-27): each direction's
    # top/bottom offsets and the convention. A record from before the rule carries none and is
    # read at its single elevation, never relabelled.
    nominal = rebar["beam_longitudinal_centroid_offset_in"]
    stacking = rebar.get("beam_bar_stacking") or {}
    offsets = stacking.get("offsets_in") or {axis: {"top": nominal, "bottom": nominal} for axis in ("x", "y")}
    return {
        "geometry": geometry,
        "sections": {key: sections[key] for key in ("b_col_in", "h_col_in", "fc_col_ksi", "b_beam_in", "h_beam_in", "fc_beam_ksi")},
        "materials": {"fy_ksi": materials["fy_ksi"], "es_ksi": materials["es_ksi"], "normalweight": materials["normalweight"],
                      "aggregate_size_in": materials.get("aggregate_size_in", 0.75)},
        "beam": {"bar_size": rebar["beam_bar_size"], "top_bars": rebar["beam_top_bars"], "bot_bars": rebar["beam_bot_bars"],
                 "centroid_offset_in": nominal, "clear_cover_in": rebar["beam_clear_cover_in"],
                 "stirrup_bar_size": rebar["beam_stirrup_bar_size"],
                 "centroid_offsets_by_axis_in": offsets,
                 "worst_centroid_offset_in": max(v for faces in offsets.values() for v in faces.values()),
                 "stacking_convention": stacking.get("convention", "none"),
                 "max_layers": int(stacking.get("max_layers", 1)), "layer_order": stacking.get("layer_order", "blocked"),
                 "layers": stacking.get("layers"),
                 "self_weight_kip_per_in": self_weight, "drop_weight_kip_per_in": drop_weight},
        "column": {"bar_size": rebar["col_bar_size"], "top_bars": rebar["col_top_bars"], "bot_bars": rebar["col_bot_bars"],
                   "side_bars": rebar["col_side_bars"], "centroid_offset_in": cover,
                   "clear_cover_in": rebar["col_clear_cover_in"], "stirrup_bar_size": rebar["col_stirrup_bar_size"],
                   "layers": layers, "layers_about_z": layers_about_z},
        "slab": {"thickness_in": thickness, "layout": (record.get("slab_reinforcement") or {}).get("layout")},
        "transfer": record.get("floor_transfer"), "sds": record["seismic"]["sds"],
        "column_axial_envelope": axial, "column_shear_demand": shear,
        "column_action_envelopes": envelopes["detail"],
        "combination_actions_used": envelopes["combinations_used"],
        "column_shear_method": saved_capacity.get("column_shear_method") or COLUMN_SHEAR_METHOD_DEFAULT,
        "column_clear_height_convention": saved_columns.get("clear_height_convention") or CLEAR_HEIGHT_CONVENTION_DEFAULT,
        # The design's declaration of reinforcement continuity through the
        # joints (Table 18.8.4.3 inputs); absent in older records, which then
        # classify every joint on the table's conservative rows.
        "joint_continuity": (record.get("detailing") or {}).get("joint_continuity"),
    }


def _close(a, b, tolerance=1e-9):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
    return a == b


def capacity_design_recomputation(record):
    """Recompute the capacity design from the record and compare with the saved one.

    Returns {"consistent": bool, "differences": [...], "recomputed": dict or
    None, "reason": str or None}. Compared: every saved check (id, status,
    demand, capacity, in order), the selected hoops, the joint-shear and
    anchorage outcomes and the overall acceptance. Any difference means the
    saved evidence does not describe the saved frame.
    """
    from Design.SMRF_Capacity_Design import build_capacity_design
    saved = record.get("capacity_design") or {}
    try:
        recomputed = build_capacity_design(capacity_state_from_record(record))
    except Exception as exc:                        # noqa: BLE001 -- any failure is a finding, not a pass
        return {"consistent": False, "differences": [f"recomputation failed: {type(exc).__name__}: {exc}"],
                "recomputed": None, "reason": str(exc)}
    differences = []
    if saved.get("method_version") != recomputed["method_version"]:
        differences.append(f"method_version {saved.get('method_version')!r} vs {recomputed['method_version']!r}")
    from Design.SMRF_Capacity_Design import COLUMN_SHEAR_METHOD_DEFAULT
    saved_method = saved.get("column_shear_method") or COLUMN_SHEAR_METHOD_DEFAULT
    if saved_method != recomputed.get("column_shear_method"):
        differences.append(f"column_shear_method {saved_method!r} vs {recomputed.get('column_shear_method')!r}")
    old, new = saved.get("checks") or [], recomputed["checks"]
    if len(old) != len(new):
        differences.append(f"{len(old)} saved checks vs {len(new)} recomputed")
    for a, b in zip(old, new):
        for key in ("id", "status", "location"):
            if a.get(key) != b.get(key):
                differences.append(f"check {a.get('id')}: {key} {a.get(key)!r} vs {b.get(key)!r}")
                break
        else:
            for key in ("demand", "capacity"):
                if not _close(a.get(key), b.get(key)):
                    differences.append(f"check {a.get('id')} {a.get('location', '')}: {key} {a.get(key)!r} vs {b.get(key)!r}")
    for member in ("beam", "column"):
        a, b = (saved.get("transverse") or {}).get(member), recomputed["transverse"][member]
        if (a is None) != (b is None) or (a and b and any(not _close(a.get(k), b.get(k)) for k in ("bar_size", "legs", "spacing_in"))):
            differences.append(f"{member} hoops {a!r} vs {b!r}")
    for path in (("joints", "all_pass"), ("anchorage", "all_pass"), ("accepted",)):
        a, b = saved, recomputed
        for key in path:
            a = (a or {}).get(key) if isinstance(a, dict) else None
            b = b.get(key) if isinstance(b, dict) else None
        if a != b:
            differences.append(f"{'.'.join(path)} {a!r} vs {b!r}")
    return {"consistent": not differences, "differences": differences[:20], "recomputed": recomputed, "reason": None}


def analysis_input_signature(record):
    """Bind solved actions to the selected frame, not merely the search request."""
    if record.get("member_groups") is not None:
        # A grouped record has no one section or cage: the frame is its assignments and group designs,
        # which the grouped block's own digest covers (Model.Member_Groups.GroupedDesign.identity).
        inputs = {key: record[key] for key in ("geometry", "materials", "floor_loads", "seismic", "demand")}
        inputs["member_groups_sha256"] = record["member_groups"]["sha256"]
        for key in ("gravity_load_model", "floor_transfer", "demand_basis"):
            inputs[key] = record.get(key)
        return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf-8")).hexdigest()
    fields = ("geometry", "sections", "reinforcement", "materials", "floor_loads", "seismic", "demand")
    inputs = {key: record[key] for key in fields}
    # Equal total weight is not equal loading: a changed transfer, panel
    # pattern or torsion policy invalidates the previously solved actions.
    for key in ("gravity_load_model", "floor_transfer", "demand_basis"):
        inputs[key] = record.get(key)
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validated_combinations(record):
    """The saved combination actions, after their IDs AND factors are checked against the canonical demand rule.

    The frame-input signature cannot detect changed action descriptors: the
    action rows are deliberately outside that signature. An unchanged case
    name alone is therefore insufficient evidence of its load combination.
    Returns (combinations, canonical ids in order); raises ValueError.
    """
    if record.get("design_actions", {}).get("analysis_input_sha256") != analysis_input_signature(record):
        raise ValueError("Solved member actions do not match the current selected frame inputs.")
    geometry = record["geometry"]
    patterns = (live_load_patterns(geometry["num_bay_x"], geometry["num_bay_y"])
                if (record.get("demand_basis") or {}).get("patterns_in_strength_envelope") else ())
    canonical = {item["id"]: item for item in strength_load_combinations(record["seismic"]["sds"], live_patterns=patterns)}
    combinations = record.get("design_actions", {}).get("combinations", [])
    if not isinstance(combinations, (list, tuple)):
        raise ValueError("Solved combination actions must be a list.")
    for item in combinations:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or item["id"] not in canonical:
            raise ValueError("Solved action combination has an unknown or missing canonical ID.")
        required = canonical[item["id"]]
        if item.get("family") != required["family"]:
            raise ValueError(f"Solved combination {item['id']} has a missing or inconsistent family.")
        if item.get("live_pattern") != required.get("live_pattern"):
            raise ValueError(f"Solved combination {item['id']} has a missing or inconsistent live_pattern.")
        for key in ("dead", "live", "ex", "ey"):
            value = item.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not math.isclose(value, required[key], rel_tol=0., abs_tol=1e-12)):
                raise ValueError(f"Solved combination {item['id']} has a missing or inconsistent {key} coefficient.")
    return combinations, list(canonical)


def joint_evidence(record, capacity=None, beam_slab_strengths=None):
    """Joint evidence from the validated combination actions (see ``validated_combinations``).

    ``capacity`` and ``beam_slab_strengths`` are the objects whose groups
    and slab contributions the joints use; qualification passes the ones it
    recomputed from the record so that no saved nested copy is consumed.
    Without them the record's own copies are used (design time).
    """
    if capacity is None:
        capacity = record.get("capacity_design") or {}
    if beam_slab_strengths is not None:
        record = {**record, "beam_slab_strengths": beam_slab_strengths}
    combinations, canonical_ids = validated_combinations(record)
    evidence = build_joint_evidence(record, combinations, expected_combination_ids=canonical_ids)
    # Probable-strength evidence (beam capacity shear, joint shear) comes from
    # the capacity design; the joint checks read it.
    groups = (capacity or {}).get("joint_evidence") or {}
    for name in ("beam_capacity_shear", "joint_shear"):
        if groups.get(name):
            evidence[name] = groups[name]
    # Exterior joints anchor terminating bars with hooks (joint.terminating_bar_hook
    # in the capacity design); the through-bar depth check applies where bars pass.
    if capacity:
        evidence["through_bar_anchorage"] = [entry for entry in evidence.get("through_bar_anchorage", [])
                                             if entry.get("bars_pass_through")]
    return evidence


def slab_strength_recomputation(record):
    """Recompute the beam-plus-slab strengths from the record and compare with the saved ones."""
    from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths
    try:
        entries, families = beam_slab_strengths(record)
    except Exception as exc:                        # noqa: BLE001
        return {"consistent": False, "differences": [f"recomputation failed: {type(exc).__name__}: {exc}"],
                "entries": None, "families": None}
    saved = record.get("beam_slab_strengths")
    differences = []
    if saved is None:
        differences.append("no beam_slab_strengths saved")
    else:
        if set(saved) != set(entries):
            differences.append(f"{len(saved)} saved entries vs {len(entries)} recomputed")
        for key in sorted(set(saved) & set(entries)):
            a, b = saved[key], entries[key]
            if set(a) != set(b) or any(not _close(a[k], b[k]) for k in a if not isinstance(a[k], dict)):
                differences.append(f"entry {key} differs")
                if len(differences) > 20:
                    break
    return {"consistent": not differences, "differences": differences[:20], "entries": entries, "families": families}


def slab_strength_inputs(record):
    slab, material = record["slab"], record["materials"]
    if material.get("reinforcement_specification") != "ASTM A706 Grade 60" or material.get("fy_ksi") != 60:
        raise ValueError("Slab steel must explicitly be ASTM A706 Grade 60; it cannot be inferred.")
    return {"thickness_in": slab["thickness_in"], "fc_ksi": slab["concrete_fc_ksi"],
            "fy_ksi": material["fy_ksi"], "max_aggregate_size_in": material["aggregate_size_in"],
            "exposure": material["exposure"], "steel_specification": "ASTM A706",
            "concrete_type": slab["concrete_type"],
            "panel_ids": sorted(panel["panel_id"] for panel in slab["panels"]),
            "num_floor": record["geometry"]["num_floor"]}


def floor_diagnostics(record, config):
    """Five explicit load patterns and one refined mesh, never verified=True.

    Checkerboards are diagnostics, not proof of the governing complete live
    load pattern envelope. Beam flexibility, twisting design actions and
    slab-to-frame transfer remain prerequisites for selecting reinforcement.
    """
    if not config.enabled:
        return {"status": "disabled", "verified": False, "cases": []}
    from Design.SMRF_Floor_Analysis import analyze_floor
    geometry = record["geometry"]
    even, odd = [], []
    for i in range(geometry["num_bay_x"]):
        for j in range(geometry["num_bay_y"]):
            (even if (i + j) % 2 == 0 else odd).append([i, j])
    cases = [
        {"id": "floor_1.4D", "dead_factor": 1.4, "live_factor": 0., "live_pattern": "none"},
        {"id": "floor_1.2D_1.6L_all", "dead_factor": 1.2, "live_factor": 1.6, "live_pattern": "all"},
        {"id": "floor_1.2D_1.6L_even", "dead_factor": 1.2, "live_factor": 1.6, "live_pattern": even},
        {"id": "floor_1.2D_1.6L_odd", "dead_factor": 1.2, "live_factor": 1.6, "live_pattern": odd},
        {"id": "floor_service_D_L_all", "dead_factor": 1., "live_factor": 1., "live_pattern": "all"},
    ]
    results, errors = [], []

    def check_result(result, case_id):
        if (result.get("status") != "diagnostic_complete"
                or result.get("equilibrium", {}).get("numerical_balance_passed") is not True):
            errors.append({"case_id": case_id, "error": "Floor solve or numerical equilibrium failed.",
                           "status": result.get("status"),
                           "analysis_return_code": result.get("analysis_return_code")})

    for case in cases:
        case["live_load_ksf"] = record["floor_loads"]["floor_live_load_ksf"]
        try:
            result = analyze_floor(record["slab"], geometry, record["sections"], case,
                                   mesh_per_bay=config.mesh_per_bay)
            results.append(result)
            check_result(result, case["id"])
        except Exception as exc:
            # Keep a failed diagnostic visible. It must never be an accepted
            # slab-demand source; do not suppress interrupts or exit signals.
            errors.append({"case_id": case["id"], "error": f"{type(exc).__name__}: {exc}"})
    refined = None
    try:
        refined = analyze_floor(record["slab"], geometry, record["sections"], cases[1],
                                 mesh_per_bay=config.refinement_mesh_per_bay)
        check_result(refined, cases[1]["id"] + "_refined")
    except Exception as exc:
        errors.append({"case_id": cases[1]["id"] + "_refined", "error": f"{type(exc).__name__}: {exc}"})
    transferred = record.get("gravity_load_model") == "slab_transfer" and record.get("floor_transfer") is not None
    return {"status": "diagnostic_only" if not errors else "diagnostic_errors",
            "verified": False, "cases": results, "errors": errors,
            "mesh_per_bay": config.mesh_per_bay,
            "refinement_mesh_per_bay": config.refinement_mesh_per_bay,
            "refined_full_live_case": refined,
            "transferred_to_frame_design": transferred,
            "reason": ("Rigid beam-line cases are diagnostics only. Frame gravity demands come from the separate "
                       "flexible-beam transfer in record['floor_transfer'] (all-panel live load, no pattern envelope); "
                       "twisting-to-design conversion and the complete load-pattern envelope remain open."
                       if transferred else
                       "Rigid beam-line support diagnostic; no validated elastic beam support, twisting-to-design conversion, complete load-pattern envelope or slab-to-frame force transfer.")}


def attach_design_evidence(record, cfg):
    record["slab_reinforcement_inputs"] = slab_strength_inputs(record)
    if record.get("slab_reinforcement") is None:
        # Legacy path (transfer disabled): no verified strip actions exist.
        record["slab_reinforcement"] = design_slab_reinforcement(record["slab_reinforcement_inputs"], None)
    from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths
    record["beam_slab_strengths"], record["beam_slab_families"] = beam_slab_strengths(record)
    record["joint_evidence"] = joint_evidence(record)
    record["floor_analysis"] = floor_diagnostics(record, cfg.floor_analysis)
    record["floor_analysis"]["checks"] = [not_evaluated(
        "floor.qualified_slab_actions", "ACI 318-19 Chapters 6 and 8",
        record["floor_analysis"].get("reason", "Floor analysis is disabled."))]
    return record
