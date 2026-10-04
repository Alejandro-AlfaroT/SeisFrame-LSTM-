"""Synchronized hinge moment-rotation histories: the export schema of the V2 learning target (2026-10-01).

Analysis.NTHA reads, at every scheduled snapshot, the relative rotation AND the conjugate moment of both
springs of every member hinge from the same committed state (zeroLength ``deformation`` and
``basicForce``; the pair the full-rate ``material i stressStrain`` recorder writes). This module turns
that result into files with an explicit schema, and compares it with recorder output on matched times.

Files written beside the other NTHA outputs:

  hinge_moment_rotation.npz          float64 arrays (see ARRAYS)
  hinge_moment_rotation_schema.json  what every array means, the sampling, the sign conventions, the
                                     reference state, the solver status, the identities, the limits
  hinge_springs.csv                  one row per spring (hinge element x local axis), in column order

What is stored and what is not:
  * rows are SCHEDULED SNAPSHOTS: one per stored scheduled step, after that step converged. When a step
    is recovered by subdivision OpenSees commits ten internal sub-steps; those are not snapshots and are
    not stored. ``commit_count`` says how many states the transient analysis had committed at each row,
    so a full-rate recorder (one row per commit) is matched by time or by commit count, never by a row
    number of this file;
  * values are ABSOLUTE and include the gravity-equilibrated state; that state is stored separately
    (``gravity_*``), so increments are a subtraction the reader makes knowingly;
  * a query that returned nothing is NaN and is counted; nothing is filled with zero;
  * a failed or truncated analysis writes what it has, with the status, and no fabricated rows.

The rotation is the TOTAL relative spring rotation of a member calibration that still contains
Haselton's bar-slip share wherever no face interface owns it (Model/Deformation_Ownership); it is not
experimentally isolated plastic flexure. Hinge histories alone do not establish a unique global
solution: floor motion, story drift and base shear are kept beside them as reference data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import Structure_Parameters as sp

SCHEMA_VERSION = "hinge_moment_rotation_v1"
NPZ_NAME = "hinge_moment_rotation.npz"
SCHEMA_NAME = "hinge_moment_rotation_schema.json"
SPRINGS_NAME = "hinge_springs.csv"
AXES = ("y", "z")                                   # local spring directions 5 and 6, in this order
AXIS_DIRECTION = {"y": 5, "z": 6}
STRATEGIES = ("Newton", "KrylovNewton", "KrylovNewton+subdivide")       # codes 0, 1, 2; anything else is -1
TIME_MATCH_TOLERANCE = 1e-9

ARRAYS = {
    "time_sec": "(n_rows,) analysis time of each stored snapshot, s",
    "scheduled_step": "(n_rows,) 0-based scheduled analysis step of each stored snapshot",
    "commit_count": "(n_rows,) states committed by the transient analysis up to and including the snapshot (1-based)",
    "internal_commits": "(n_rows,) states committed by that scheduled step: 1, or the subdivision count of a recovered step",
    "strategy_code": "(n_rows,) index into strategies (how the scheduled step converged); -1 if unknown",
    "rotation_rad": "(n_rows, n_springs) total relative spring rotation, rad",
    "moment_kip_in": "(n_rows, n_springs) conjugate spring moment, kip-in",
    "gravity_rotation_rad": "(n_springs,) rotation at the gravity-equilibrated state before the first transient step",
    "gravity_moment_kip_in": "(n_springs,) moment at that state",
    "gravity_time_sec": "() analysis time of that state",
    "spring_hinge_element_tag": "(n_springs,) zeroLength element tag of the spring's hinge",
    "spring_axis_direction": "(n_springs,) local spring direction: 5 (local y) or 6 (local z)",
    "spring_member_tag": "(n_springs,) physical member tag",
    "spring_end_id": "(n_springs,) 1 = member end i, 2 = member end j",
    "spring_material_tag": "(n_springs,) uniaxial material tag, -1 if not recorded",
}

SPRING_COLUMNS = (
    "spring_index", "hinge_element_tag", "physical_member_tag", "member_class", "member_type", "end", "end_id",
    "local_axis", "spring_local_direction", "material_tag", "material_type", "global_rotation_axis",
    "zero_length_orientation", "retained_joint_node", "member_hinge_node", "joint_node", "tied_global_dofs",
    "positive_moment_meaning", "negative_moment_meaning", "yield_moment_positive_kip_in", "yield_moment_negative_kip_in",
    "elastic_stiffness_kip_in_per_rad", "theta_p", "theta_pc", "theta_u", "calibration_id", "calibration_status",
    "deterioration_source", "energy_mapping_status", "deformation_scope", "bond_slip_indicator", "slip_owner",
    "end_location", "ownership_policy", "bar_slip_owner", "panel_shear_owner", "parameter_sha256",
    # Energy basis of the installed material (added 2026-10-04; older tables do not have these columns).
    "energy_anchor_policy", "energy_calibration_status", "energy_reference_moment_kip_in",
    "reference_energy_by_mode_kip_in_rad", "installed_lamda_by_mode_rad",
)


# Appended to the spring table only when the model was built from a grouped design: which design group each
# spring's member belongs to and where the member sits. A uniform model's table keeps exactly SPRING_COLUMNS.
GROUP_COLUMNS = ("group_id", "band", "location_class", "story_or_floor", "grid_i", "grid_j")


def spring_columns(rows):
    """The column list of a spring table: the group columns follow when any row belongs to a design group."""
    grouped = any(row.get("group_id") not in (None, "") for row in rows)
    return list(SPRING_COLUMNS) + (list(GROUP_COLUMNS) if grouped else [])


def _decode(hinge_tag):
    return divmod(int(hinge_tag) - sp.IMK_HINGE_ELEMENT_TAG_BASE, 10)          # (member tag, end id)


def spring_table(hinge_tag_order, registry=None):
    """One row per spring in history column order: identities, axes, sign meaning, calibration, ownership."""
    if registry is None:
        from Model.IMK_Hinges import hinge_registry
        registry = hinge_registry()
    rows = []
    for hinge_tag in hinge_tag_order:
        member_tag, end_id = _decode(hinge_tag)
        end = "i" if end_id == 1 else "j"
        entry = registry.get(int(member_tag)) or {}
        member_type = entry.get("member_type") or ""
        is_beam = member_type.startswith("beam")
        owned = (entry.get("deformation_ownership") or {}).get(end) or {}
        backbone = (entry.get("backbone_by_end") or {}).get(end) or {}
        for axis in AXES:
            material = ((entry.get("installed_materials") or {}).get(end) or {}).get(axis) or {}
            provenance = material.get("provenance") or {}
            context = provenance.get("spring_context") or {}
            if axis == "y" and is_beam:
                hogging = entry.get(f"yield_moment_y_hogging_{end}_kip_in")
                sagging = entry.get(f"yield_moment_y_sagging_{end}_kip_in")
                # Hogging is positive spring moment at end i and negative at end j (Model/IMK_Hinges._create_end_hinge).
                positive, negative = (hogging, sagging) if end_id == 1 else (sagging, hogging)
                meaning = (("hogging (top in tension)", "sagging") if end_id == 1 else ("sagging", "hogging (top in tension)"))
                stiffness = entry.get("ke_y_kip_in_per_rad")
            elif axis == "y":
                positive = negative = entry.get("yield_moment_y_kip_in")
                meaning = ("positive rotation about the spring's global axis", "negative rotation about the spring's global axis")
                stiffness = entry.get("ke_y_kip_in_per_rad")
            else:
                positive = negative = entry.get("yield_moment_z_kip_in")
                meaning = ("positive rotation about the spring's global axis", "negative rotation about the spring's global axis")
                stiffness = entry.get("ke_z_kip_in_per_rad")
            rows.append({
                "spring_index": len(rows), "hinge_element_tag": int(hinge_tag), "physical_member_tag": int(member_tag),
                "member_class": "beam" if is_beam else ("column" if member_type == "column" else ""),
                "member_type": member_type, "end": end, "end_id": int(end_id),
                "local_axis": axis, "spring_local_direction": AXIS_DIRECTION[axis],
                "material_tag": material.get("material_tag", ""), "material_type": material.get("material_type", entry.get("material_type", "")),
                "global_rotation_axis": json.dumps(context.get("global_rotation_axis")) if context else "",
                "zero_length_orientation": json.dumps(context.get("zero_length_orientation")) if context else "",
                "retained_joint_node": context.get("retained_joint_node", ""), "member_hinge_node": context.get("member_hinge_node", ""),
                "joint_node": entry.get("node_i" if end_id == 1 else "node_j", ""),
                "tied_global_dofs": json.dumps(context.get("tied_global_dofs")) if context else "",
                "positive_moment_meaning": meaning[0], "negative_moment_meaning": meaning[1],
                "yield_moment_positive_kip_in": "" if positive is None else positive,
                "yield_moment_negative_kip_in": "" if negative is None else negative,
                "elastic_stiffness_kip_in_per_rad": "" if stiffness is None else stiffness,
                "theta_p": backbone.get("theta_p", ""), "theta_pc": backbone.get("theta_pc", ""), "theta_u": backbone.get("theta_u", ""),
                "calibration_id": provenance.get("calibration_id", ""), "calibration_status": provenance.get("status", ""),
                "deterioration_source": provenance.get("deterioration_source", ""),
                "energy_mapping_status": provenance.get("energy_mapping_status", entry.get("energy_mapping_mode", "")),
                "deformation_scope": provenance.get("deformation_scope", ""),
                "bond_slip_indicator": provenance.get("bond_slip_indicator", owned.get("bond_slip_indicator", "")),
                "slip_owner": provenance.get("slip_owner", ""), "end_location": provenance.get("end_location", ""),
                "ownership_policy": owned.get("policy", ""), "bar_slip_owner": owned.get("bar_slip", ""),
                "panel_shear_owner": "" if owned.get("panel_shear") is None else owned.get("panel_shear"),
                "parameter_sha256": material.get("parameter_sha256", ""),
                **_energy_basis(material, provenance, entry),
            })
            if entry.get("group_id") is not None:
                rows[-1].update({key: entry.get(key, "") for key in GROUP_COLUMNS})
    return rows


def _energy_basis(material, provenance, entry):
    """What anchors the cyclic energy of one installed spring: read from the installed material's own record."""
    mapping = provenance.get("energy_mapping") or {}
    cyclic = material.get("cyclic") or {}
    energies = material.get("reference_energies_kip_in_rad") or {}
    lamda = {mode: cyclic.get(f"lamda_{mode.lower()}") for mode in energies}
    policy = mapping.get("version") or provenance.get("energy_mapping_status", entry.get("energy_mapping_mode", ""))
    reference = mapping.get("reference_moment_kip_in")
    if reference is None and policy == "legacy_unmapped":
        reference = (material.get("positive") or {}).get("fy")          # E_ref = Lamda * My of the positive input branch
    return {"energy_anchor_policy": policy,
            "energy_calibration_status": mapping.get("status") or (provenance.get("energy_calibration") or {}).get("status")
                                         or provenance.get("status", ""),
            "energy_reference_moment_kip_in": "" if reference is None else reference,
            "reference_energy_by_mode_kip_in_rad": json.dumps(energies) if energies else "",
            "installed_lamda_by_mode_rad": json.dumps(lamda) if energies else ""}


def _identity():
    from Model import Deformation_Ownership as own
    try:
        from Model.Analysis_Profile import profile_identity
        profile = profile_identity()
        profile = {"id": profile["id"], "sha256": profile["sha256"], "settings": profile["settings"]}
    except Exception as exc:                               # noqa: BLE001 -- reported, never required for an export
        profile = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "analysis_profile": profile,
        "member_material_type": getattr(sp, "IMK_MATERIAL_TYPE", None),
        "member_cyclic_calibration_id": getattr(sp, "IMK_CYCLIC_CALIBRATION_ID", None),
        "member_cyclic_calibration_status": getattr(sp, "IMK_CYCLIC_CALIBRATION_STATUS", None),
        "member_energy_mapping_mode": getattr(sp, "IMK_ENERGY_MAPPING_MODE", None),
        "member_deterioration_mode": getattr(sp, "IMK_DETERIORATION_MODE", None),
        "joint_model": getattr(sp, "JOINT_MODEL", None),
        "deformation_ownership_policy": own.POLICY_ID,
        "face_slip_interfaces_registered": len(own.slip_interfaces()),
        "risk_category": getattr(sp, "ASCE_RISK_CATEGORY", None), "importance_factor": getattr(sp, "ASCE_IE", None),
    }


def assemble(results, registry=None):
    """Arrays, spring rows and schema from a run_ntha result. Nothing is invented for a missing history."""
    tags = [int(t) for t in (results.get("hinge_tag_order") or [])]
    status = results.get("status") or {}
    base = {"schema_version": SCHEMA_VERSION, "available": bool(tags),
            "status": {"completed_steps": status.get("completed_steps"), "requested_steps": status.get("npts_requested"),
                       "failed": status.get("failed"), "failed_step": status.get("failed_step"),
                       "failed_time_sec": status.get("failed_time_sec"),
                       "domain_time_at_failure_sec": results.get("domain_time_at_failure"),
                       "truncated": bool(status.get("failed")) or (status.get("completed_steps") is not None
                                                                   and status.get("npts_requested") is not None
                                                                   and status["completed_steps"] < status["npts_requested"])}}
    if not tags:
        return None, [], {**base, "reason": "no member hinge springs are installed in this model (no IMK hinge elements in the domain)"}
    n_springs = 2 * len(tags)
    if "hinge_moment_history" not in results or "hinge_history_time" not in results:
        return None, [], {**base, "available": False,
                          "reason": "this result carries no synchronized moment history (produced before hinge_moment_rotation_v1)"}
    rotation = np.asarray(results.get("hinge_rotation_history") or [], dtype=np.float64).reshape(-1, n_springs)
    moment = np.asarray(results.get("hinge_moment_history") or [], dtype=np.float64).reshape(-1, n_springs)
    time = np.asarray(results.get("hinge_history_time") or [], dtype=np.float64)
    steps = np.asarray(results.get("hinge_rotation_steps") or [], dtype=np.int64)
    commits = np.asarray(results.get("hinge_history_commit_count") or [], dtype=np.int64)
    strategies = list(results.get("hinge_history_strategy") or [])
    if not (len(rotation) == len(moment) == len(time) == len(steps) == len(commits) == len(strategies)):
        raise ValueError("hinge history arrays have different lengths; refusing to pair moment and rotation by position: "
                         f"rotation {len(rotation)}, moment {len(moment)}, time {len(time)}, steps {len(steps)}, "
                         f"commits {len(commits)}, strategies {len(strategies)}")
    subdivision = int(results.get("hinge_history_substeps_per_subdivided_step") or 1)
    codes, internal = [], []
    for name in strategies:
        family = next((k for k, s in enumerate(STRATEGIES) if name == s or (k == 2 and str(name).startswith(s))), -1)
        codes.append(family)
        internal.append(subdivision if "subdivide" in str(name) else 1)
    gravity = results.get("hinge_gravity_state") or {}
    nan = np.full(n_springs, np.nan)
    rows = spring_table(tags, registry)
    arrays = {
        "time_sec": time, "scheduled_step": steps, "commit_count": commits,
        "internal_commits": np.asarray(internal, dtype=np.int64), "strategy_code": np.asarray(codes, dtype=np.int64),
        "rotation_rad": rotation, "moment_kip_in": moment,
        "gravity_rotation_rad": np.asarray(gravity.get("rotation"), dtype=np.float64) if gravity.get("rotation") is not None else nan,
        "gravity_moment_kip_in": np.asarray(gravity.get("moment"), dtype=np.float64) if gravity.get("moment") is not None else nan,
        "gravity_time_sec": np.asarray(gravity.get("time", np.nan), dtype=np.float64),
        "spring_hinge_element_tag": np.asarray([r["hinge_element_tag"] for r in rows], dtype=np.int64),
        "spring_axis_direction": np.asarray([r["spring_local_direction"] for r in rows], dtype=np.int64),
        "spring_member_tag": np.asarray([r["physical_member_tag"] for r in rows], dtype=np.int64),
        "spring_end_id": np.asarray([r["end_id"] for r in rows], dtype=np.int64),
        "spring_material_tag": np.asarray([int(r["material_tag"]) if r["material_tag"] != "" else -1 for r in rows], dtype=np.int64),
    }
    missing_rows = int(np.isnan(rotation).sum() + np.isnan(moment).sum())
    stride = int(results.get("hinge_history_stride") or 0)
    schema = {
        **base,
        "files": {"arrays": NPZ_NAME, "springs": SPRINGS_NAME},
        "arrays": ARRAYS, "precision": "float64",
        "counts": {"rows": int(len(time)), "hinges": len(tags), "springs": n_springs},
        "layout": ("columns are springs in the row order of hinge_springs.csv: for each hinge element in ascending element-tag "
                   "order, local y (direction 5) then local z (direction 6)"),
        "quantities": {
            "rotation_rad": ("total relative spring rotation: member hinge-node rotation minus retained joint-node rotation, "
                             "projected on the spring's local axis. It belongs to a member calibration that includes Haselton's "
                             "bar-slip share wherever no face interface owns it (bond_slip_indicator per spring); it is not "
                             "experimentally isolated plastic flexure"),
            "moment_kip_in": ("the spring's conjugate moment: zeroLength basicForce, equal to the material stress, in the same "
                              "local direction and sign as the rotation and read from the same committed state"),
        },
        "sign_convention": {
            "definition": "positive moment and positive rotation share the sign of the spring's local direction; nothing is negated",
            "beam_local_y": "hogging (top in tension) is positive at member end i and negative at member end j",
            "beam_local_z": "weak-axis (in-plan) bending; symmetric strength",
            "column": "both axes are symmetric; the global axis of each spring is in hinge_springs.csv (global_rotation_axis)",
            "asymmetry": "slab-driven hogging/sagging asymmetry of beam hinges is physical and kept; see yield_moment_positive/negative per spring",
            "cyclic_energy": ("per spring in hinge_springs.csv: energy_anchor_policy, the reference moment, the reference energy "
                              "of every deterioration mode and the Lamda installed on the positive input branch"),
            "relation_to_member_end_moment": ("the spring's own conjugate pair is stored; its relation to the elastic member's local "
                                              "end moments depends on the member type and axis (Analysis/Pushover_Diagnostic and "
                                              "Hinge_Hysteresis_Diagnostic.column_story_shears_from_hinges) and is not restated here"),
        },
        "reference_state": {
            "kind": "absolute",
            "note": ("values include the gravity-equilibrated state; gravity_rotation_rad and gravity_moment_kip_in hold that "
                     "state, read before the first transient step; increments are row minus gravity"),
            "gravity_state_recorded": bool(gravity), "gravity_missing_values": gravity.get("missing_values"),
        },
        "sampling": {
            "stride": stride,
            "rows": ("scheduled snapshots only: one row per stored scheduled step, taken after that step converged; the internal "
                     "sub-steps of a subdivided recovery are committed by OpenSees and never stored"),
            "time": "ops.getTime() at the snapshot; never inferred from a row index",
            "commit_count": ("states committed by the transient analysis up to the snapshot; a full-rate recorder writes one row per "
                             "commit, so its row for this snapshot is commit_count - 1"),
            "strategies": list(STRATEGIES), "subdivision_count": subdivision,
            "recovered_rows": int(sum(1 for c in codes if c != 0)), "subdivided_rows": int(sum(1 for v in internal if v > 1)),
            "alignment_rule": "pair this file with any other history by time_sec or commit_count; do not zip arrays by row number",
        },
        "missing": {"encoding": "NaN", "values_in_history": missing_rows,
                    "values_reported_by_analysis": results.get("hinge_history_missing_values"),
                    "note": "a query that returned no value; never zero-filled, so a missing label cannot read as a plausible zero"},
        "units": {"moment": "kip-in", "rotation": "rad", "time": "s"},
        "member_groups": {"present": len(spring_columns(rows)) > len(SPRING_COLUMNS),
                          "columns": list(GROUP_COLUMNS) if len(spring_columns(rows)) > len(SPRING_COLUMNS) else [],
                          "note": "under a grouped design every spring row names its member's design group and grid position"},
        "identity": _identity(),
        "global_reference_outputs": {
            "kept_in": "response_arrays.npz and time_history.csv beside this file",
            "floor_acceleration": "relative to the ground (UniformExcitation formulation); absolute = relative + ground acceleration",
            "story_drift_history": "unsigned |drift ratio| per story and direction; signed drift comes from the floor displacements",
            "base_shear": "sum of first-story column shears (global), including P-Delta and damping force",
        },
        "claims_not_made": [
            "hinge moment-rotation histories alone do not establish a unique global solution",
            "the rotation is not isolated plastic flexure",
            f"the member energy mapping ({getattr(sp, 'IMK_ENERGY_MAPPING_MODE', None)}, provisional) is not certified by this export",
            "rigid_centerline joints create no clear spans or rigid end offsets",
        ],
    }
    return arrays, rows, schema


def write_hinge_moment_rotation(output_dir, results, registry=None):
    """Write the three files; return the counts. A result without a moment history writes the schema only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, rows, schema = assemble(results, registry)
    if arrays is not None:
        np.savez_compressed(output_dir / NPZ_NAME, **arrays)
        with (output_dir / SPRINGS_NAME).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=spring_columns(rows))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / SCHEMA_NAME).write_text(json.dumps(schema, indent=1, default=str), encoding="utf-8")
    return {"hinge_moment_rotation_available": arrays is not None,
            "hinge_moment_rotation_rows": 0 if arrays is None else int(len(arrays["time_sec"])),
            "hinge_moment_rotation_springs": 0 if arrays is None else int(arrays["rotation_rad"].shape[1]),
            "hinge_moment_rotation_missing_values": None if arrays is None else schema["missing"]["values_in_history"]}


def read_hinge_moment_rotation(output_dir):
    """The saved arrays, spring rows and schema; arrays is None when the run had no moment history."""
    output_dir = Path(output_dir)
    schema = json.loads((output_dir / SCHEMA_NAME).read_text(encoding="utf-8"))
    if not (output_dir / NPZ_NAME).exists():
        return None, [], schema
    with np.load(output_dir / NPZ_NAME) as data:
        arrays = {key: data[key] for key in data.files}
    with (output_dir / SPRINGS_NAME).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return arrays, rows, schema


def compare_with_recorders(arrays, recorded, *, time_tolerance=TIME_MATCH_TOLERANCE):
    """Agreement of the stored snapshots with full-rate ``material stressStrain`` recorder output.

    ``recorded`` is Analysis.Hinge_Hysteresis_Diagnostic.read_hinge_recorders(...): per axis a time
    vector and (rows x hinges) moment and rotation in its ``hinge_tag_order``. Rows are matched BY TIME
    (the recorder has one row per commit, sub-steps included); a snapshot with no recorder row within
    the tolerance is counted as unmatched, never paired with a neighbour.
    """
    if arrays is None:
        return {"compared": False, "reason": "no stored moment history"}
    order = {int(tag): k for k, tag in enumerate(recorded["hinge_tag_order"])}
    hinge_tags = arrays["spring_hinge_element_tag"][0::2]
    out = {"compared": True, "matched_by": "time", "time_tolerance_sec": time_tolerance, "snapshots": int(len(arrays["time_sec"])), "axes": {}}
    for axis_index, axis in enumerate(AXES):
        block = recorded[axis]
        rec_time = np.asarray(block["time"], dtype=np.float64)
        columns = np.asarray([order.get(int(tag), -1) for tag in hinge_tags])
        if (columns < 0).any():
            out["axes"][axis] = {"compared": False, "reason": "recorder does not cover every stored hinge"}
            continue
        ours_m = arrays["moment_kip_in"][:, axis_index::2]
        ours_r = arrays["rotation_rad"][:, axis_index::2]
        if not len(rec_time) or not len(arrays["time_sec"]):
            out["axes"][axis] = {"compared": False, "reason": "no rows to compare", "recorder_rows": int(len(rec_time))}
            continue
        nearest = np.clip(np.searchsorted(rec_time, arrays["time_sec"]), 0, len(rec_time) - 1)
        before = np.clip(nearest - 1, 0, len(rec_time) - 1)
        nearest = np.where(np.abs(rec_time[before] - arrays["time_sec"]) < np.abs(rec_time[nearest] - arrays["time_sec"]), before, nearest)
        matched = np.abs(rec_time[nearest] - arrays["time_sec"]) <= time_tolerance
        rec_m = np.asarray(block["moment"])[nearest][:, columns][matched]
        rec_r = np.asarray(block["rotation"])[nearest][:, columns][matched]
        dm, dr = np.abs(rec_m - ours_m[matched]), np.abs(rec_r - ours_r[matched])
        by_commit = None
        if len(rec_time) >= int(arrays["commit_count"].max(initial=0)):
            by_commit = bool(np.all(np.abs(rec_time[arrays["commit_count"][matched] - 1] - arrays["time_sec"][matched]) <= time_tolerance))
        out["axes"][axis] = {
            "compared": True, "matched_snapshots": int(matched.sum()), "unmatched_snapshots": int((~matched).sum()),
            "recorder_rows": int(len(rec_time)), "recorder_rows_that_are_not_snapshots": int(len(rec_time) - matched.sum()),
            "max_abs_moment_difference_kip_in": float(dm.max()) if dm.size else None,
            "max_abs_rotation_difference_rad": float(dr.max()) if dr.size else None,
            "max_abs_moment_kip_in": float(np.abs(ours_m[matched]).max()) if dm.size else None,
            "max_abs_rotation_rad": float(np.abs(ours_r[matched]).max()) if dr.size else None,
            "commit_count_locates_the_same_recorder_row": by_commit,
        }
    return out
