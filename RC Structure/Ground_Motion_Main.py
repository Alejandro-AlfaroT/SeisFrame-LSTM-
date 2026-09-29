import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import openseespy.opensees as ops

import Structure_Parameters as sp
from Geometry_Overrides import (
    add_geometry_arguments,
    apply_geometry_from_args,
)
from Analysis.Gravity import run_gravity_analysis
from Analysis.Modal import run_modal_analysis
from Analysis.NTHA import envelope_to_end_force_rows, run_ntha
from Data_Generation.Graph_Exporter import (
    collect_element_rows,
    collect_global_parameters,
    collect_graph_edge_rows,
    collect_node_rows,
)
from Model.IMK_Hinges import hinge_registry
from Run_Naming import analysis_run_name, safe_name, variant_value
from Design.Design_Driver import DESIGN_ARTIFACT_NAME, load_or_create_design
from Design.SMRF_Qualification import require_accepted_design, ensure_generation_release_ready
from Loads.Gravity_Loads import apply_gravity_loads
from Loads.Ground_Motion import (
    ground_motion_catalog_summary,
    load_ground_motion_pair_by_result_id,
    load_ground_motion_pairs,
    load_ground_motion_record,
    summarize_record,
)
from Model.Build_Model import build_model


OUTPUT_IDENTITY_KEYS = (
    "num_bay_x",
    "num_bay_y",
    "num_floor",
    "bay_x_in",
    "bay_y_in",
    "story_h_in",
    "fc_col_ksi",
    "fc_beam_ksi",
    "fy_ksi",
    "b_col_in",
    "h_col_in",
    "b_beam_in",
    "h_beam_in",
    "col_bar_size",
    "col_top_bars",
    "col_bot_bars",
    "col_side_bars",
    "beam_bar_size",
    "beam_top_bars",
    "beam_bot_bars",
    "beam_side_bars",
    "element_formulation",
    "imk_apply_to_columns",
    "imk_apply_to_beams",
    "imk_material_type",
    "imk_energy_convention",
    "imk_energy_mapping_mode",
    "imk_member_energy_calibrations",
    "imk_cyclic_calibration_id",
    "imk_cyclic_calibration_status",
    "imk_cyclic_parameters",
    # A change of deterioration mode (direct constants vs Haselton Eq. 3.20 x theta_y) or of the
    # backbone source alters every spring while leaving imk_cyclic_parameters unchanged, so both
    # are identity keys, together with the installed policy and the member yield rotations it scales.
    "imk_deterioration_mode",
    "imk_use_calibrated_backbone",
    "imk_installed_deterioration",
    "imk_beam_theta_y",
    "imk_column_theta_y",
    "imk_hinge_stiffness_mode",
    "imk_hinge_stiffness_factor",
    # Joint springs (2026-09-27): their presence, what their rotation stands for, their calibration
    # basis and the member slip term they take over all change every response.
    "joint_model",
    "joint_deformation_scope",
    "joint_calibration_basis",
    "joint_stiffness_modifier",
    "joint_kappa",
    "imk_bond_slip_indicator",
    "slip_ownership_policy",
    "floor_loads",
    "reinforcement_geometry",
)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return path


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# Run-directory naming lives in Run_Naming so the dataset scheduler, which
# must not import OpenSees, can predict the same names this module creates.
_safe_name = safe_name
_variant_value = variant_value


def validate_ntha_output_compatibility(output_dir, overwrite_existing=False):
    output_dir = Path(output_dir)
    if overwrite_existing or not output_dir.exists():
        return

    generated_files = (
        output_dir / "status.json",
        output_dir / "summary.json",
        output_dir / "global_parameters.json",
    )
    if not any(path.exists() for path in generated_files):
        return

    parameters_path = output_dir / "global_parameters.json"
    if not parameters_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing NTHA output without configuration "
            f"metadata: {output_dir}. Use a different output directory or "
            "--overwrite-existing."
        )

    with parameters_path.open(encoding="utf-8") as file:
        previous = json.load(file)
    current = collect_global_parameters()
    differences = {
        key: {"existing": previous.get(key), "requested": current.get(key)}
        for key in OUTPUT_IDENTITY_KEYS
        if previous.get(key) != current.get(key)
    }
    if differences:
        preview = ", ".join(
            f"{key}={values['existing']!r}->{values['requested']!r}"
            for key, values in list(differences.items())[:8]
        )
        raise RuntimeError(
            "Refusing to mix different model configurations in "
            f"{output_dir}. Differences: {preview}. "
            "Use a different output directory or --overwrite-existing."
        )


def _story_drift_rows(results):
    rows = []
    for story, (dx, dy) in enumerate(
        zip(results["peak_story_drift_x"], results["peak_story_drift_y"]),
        start=1,
    ):
        rows.append(
            {
                "story": story,
                "floor_pair": f"{story - 1}-{story}",
                "peak_drift_x_ratio": dx,
                "peak_drift_y_ratio": dy,
                "peak_drift_x_percent": 100.0 * dx,
                "peak_drift_y_percent": 100.0 * dy,
                "peak_drift_resultant_ratio": (dx * dx + dy * dy) ** 0.5,
                "peak_drift_resultant_percent": 100.0 * (dx * dx + dy * dy) ** 0.5,
            }
        )
    return rows


def _time_history_rows(results):
    shear_x = results.get("base_shear_x") or []
    shear_y = results.get("base_shear_y") or []
    rows = []
    for i, t in enumerate(results["time_history"]):
        ux = results["roof_disp_x"][i]
        uy = results["roof_disp_y"][i]
        rows.append(
            {
                "step": i,
                "time_sec": t,
                "roof_disp_x_in": ux,
                "roof_disp_y_in": uy,
                "roof_disp_resultant_in": (ux * ux + uy * uy) ** 0.5,
                "base_shear_x_kip": shear_x[i] if i < len(shear_x) else 0.0,
                "base_shear_y_kip": shear_y[i] if i < len(shear_y) else 0.0,
            }
        )
    return rows


def _joint_spring_rows(results):
    """One row per joint spring plane: its calibration and the rotation it reached (empty without joints)."""
    from Model.Joint_Springs import joint_registry
    envelope = results.get("joint_rotation_envelope") or {}
    rows = []
    for joint, entry in sorted(joint_registry().items()):
        peaks = envelope.get(entry["element"]) or envelope.get(str(entry["element"]))
        for axis, plane in sorted(entry["planes"].items()):
            slot = 0 if plane["direction"] == 4 else 1
            rot_max = peaks["max"][slot] if peaks else 0.0
            rot_min = peaks["min"][slot] if peaks else 0.0
            rot_abs = peaks["abs_max"][slot] if peaks else 0.0
            theta_y = plane["theta_y_rad"]
            rows.append({
                "joint_node": joint, "beam_core": entry["beam_core"], "joint_ele_tag": entry["element"],
                "floor": entry["floor"], "grid_i": entry["grid_i"], "grid_j": entry["grid_j"],
                "kind": entry["kind"], "level": entry["level"], "axis": axis, "direction": plane["direction"],
                "joint_class": plane["joint_class"], "strength_source": plane["source"],
                "gamma": plane["gamma"], "aj_in2": plane["aj_in2"], "vn_kip": plane["vn_kip"],
                "mn_kip_in": plane["mn_kip_in"], "ke_kip_in_per_rad": plane["ke_kip_in_per_rad"],
                "theta_y": theta_y, "a_rad": plane["a_rad"], "b_rad": plane["b_rad"], "c_residual": plane["c_residual"],
                "axial_ratio": plane["axial_ratio"], "shear_ratio_design": plane["shear_ratio"],
                "kappa_f": plane["kappa_f"], "kappa_d": plane["kappa_d"],
                "rot_max": rot_max, "rot_min": rot_min, "rot_abs_max": rot_abs,
                "plastic_rotation": max(0.0, rot_abs - theta_y),
                "yielded": 1 if rot_abs > theta_y else 0,
                "past_capping": 1 if rot_abs - theta_y >= plane["a_rad"] else 0,
            })
    return rows


def _hinge_backbone_rows(results):
    """One row per IMK spring: its calibrated backbone and rotation envelope.

    Joins the per-build hinge registry, which records what each spring was
    actually given, against the rotation peaks it reached. This is the table
    every damage measure is computed from: without theta_p a rotation is just
    a number, and without the rotation the backbone says nothing about damage.
    """
    registry = hinge_registry()
    envelope = results.get("hinge_rotation_envelope") or {}
    base = sp.IMK_HINGE_ELEMENT_TAG_BASE

    rows = []
    for hinge_tag, peaks in sorted(envelope.items(), key=lambda item: int(item[0])):
        offset = int(hinge_tag) - base
        ele_tag, end_id = divmod(offset, 10)
        backbone = registry.get(ele_tag, {})
        theta_p = backbone.get("theta_p") or 0.0
        # Recorded rotation is the SPRING's rotation, so the elastic part to
        # remove is the spring's own yield rotation, not the member-level
        # nominal theta_y_target. Subtracting the nominal (a fixed 0.004 for
        # columns) took away 26x to 65x more than the spring actually stores
        # elastically, understating plastic rotation and damage_ratio -- by
        # 18% on the worst hinge of case_0008. theta_y_target is still
        # reported, because it is what the backbone was calibrated against.
        theta_y = backbone.get("theta_y_target") or 0.0
        rotation = max(peaks["abs_max"])
        # Beam ends can differ in strength (an exterior end without developed
        # slab bars); the spring's own yield rotation is the one at this end.
        end_theta_y = backbone.get("theta_y_spring_y_i" if end_id == 1 else "theta_y_spring_y_j")
        spring_theta_y = max(
            end_theta_y if end_theta_y is not None else (backbone.get("theta_y_spring_y") or 0.0),
            backbone.get("theta_y_spring_z") or 0.0,
        ) or theta_y
        plastic = max(0.0, rotation - spring_theta_y)
        # The strong-axis material this end was actually given: its type and the deterioration
        # capacities installed (under haselton_2008 these are lambda*theta_y for S and C and the
        # 1e12 suppression for A and K, not the IMK_LAMBDA_* constants).
        installed = ((backbone.get("installed_materials") or {}).get("i" if end_id == 1 else "j") or {}).get("y") or {}
        cyclic = installed.get("cyclic") or {}
        rows.append(
            {
                "hinge_ele_tag": int(hinge_tag),
                "ele_tag": ele_tag,
                "end_id": end_id,
                "member_type": backbone.get("member_type", ""),
                "material_type": backbone.get("material_type", ""),
                "deterioration_source": (installed.get("provenance") or {}).get("deterioration_source", ""),
                "lamda_s_installed": cyclic.get("lamda_s", ""),
                "lamda_c_installed": cyclic.get("lamda_c", ""),
                "lamda_a_installed": cyclic.get("lamda_a", ""),
                "lamda_k_installed": cyclic.get("lamda_k", ""),
                "axial_kip": backbone.get("axial_kip", 0.0),
                "axial_ratio": backbone.get("axial_ratio", 0.0),
                "yield_moment_kip_in": backbone.get("yield_moment_y_kip_in", 0.0),
                # Beams are asymmetric once slab mats join the negative
                # section; columns report the same value twice.
                "yield_moment_hogging_kip_in": backbone.get("yield_moment_y_hogging_kip_in",
                                                            backbone.get("yield_moment_y_kip_in", 0.0)),
                "yield_moment_sagging_kip_in": backbone.get("yield_moment_y_sagging_kip_in",
                                                            backbone.get("yield_moment_y_kip_in", 0.0)),
                "beam_family": backbone.get("beam_family") or "",
                "theta_y": theta_y,
                "theta_y_spring": spring_theta_y,
                "theta_p": theta_p,
                "theta_pc": backbone.get("theta_pc", 0.0),
                "theta_u": backbone.get("theta_u", 0.0),
                "rot_y_max": peaks["max"][0],
                "rot_y_min": peaks["min"][0],
                "rot_z_max": peaks["max"][1] if len(peaks["max"]) > 1 else 0.0,
                "rot_z_min": peaks["min"][1] if len(peaks["min"]) > 1 else 0.0,
                "rot_abs_max": rotation,
                "plastic_rotation": plastic,
                # Damage ratio: plastic rotation as a fraction of the capping
                # plastic rotation. 0 is elastic, 1 is the onset of strength
                # loss. Unlike the moment ratio this does not saturate.
                "damage_ratio": (plastic / theta_p) if theta_p > 0 else 0.0,
                "yielded": 1 if rotation > spring_theta_y else 0,
                "past_capping": 1 if theta_p > 0 and plastic >= theta_p else 0,
            }
        )
    return rows


def _write_response_arrays(output_dir, results):
    """Store the large response arrays compressed rather than as CSV.

    Per-floor kinematics and hinge rotation histories are far too large for
    the CSV interchange the smaller tables use: at full resolution they run
    to tens of megabytes per run, which would dominate dataset storage.
    """
    hinge_history = results.get("hinge_rotation_history") or []
    payload = {
        "floor_disp": np.asarray(results.get("floor_disp_history") or [], dtype=np.float32),
        "floor_vel": np.asarray(results.get("floor_vel_history") or [], dtype=np.float32),
        "floor_accel": np.asarray(results.get("floor_accel_history") or [], dtype=np.float32),
        "story_drift": np.asarray(results.get("story_drift_history") or [], dtype=np.float32),
        "base_shear": np.asarray(
            [results.get("base_shear_x") or [], results.get("base_shear_y") or []],
            dtype=np.float32,
        ).T,
        "hinge_rotation": np.asarray(hinge_history, dtype=np.float32),
        "hinge_rotation_steps": np.asarray(
            results.get("hinge_rotation_steps") or [], dtype=np.int32
        ),
        "hinge_tag_order": np.asarray(results.get("hinge_tag_order") or [], dtype=np.int64),
        # Element end forces in the member local frame, for element-level
        # targets. Joint forces are assembled separately in the global frame,
        # because local forces cannot be summed across differently oriented
        # members.
        "element_force": np.asarray(
            results.get("element_force_history") or [], dtype=np.float32
        ),
        "element_force_steps": np.asarray(
            results.get("element_force_steps") or [], dtype=np.int32
        ),
        "element_tag_order": np.asarray(
            results.get("element_tag_order") or [], dtype=np.int64
        ),
        "joint_force": np.asarray(
            results.get("joint_force_history") or [], dtype=np.float32
        ),
        "joint_node_order": np.asarray(
            results.get("joint_node_order") or [], dtype=np.int64
        ),
        "floor_master_nodes": np.asarray(
            results.get("floor_master_nodes") or [], dtype=np.int64
        ),
    }
    path = Path(output_dir) / "response_arrays.npz"
    np.savez_compressed(path, **payload)
    return path


def _node_envelope_rows(results):
    dof_names = ("ux", "uy", "uz", "rx", "ry", "rz")
    rows = []
    for node_tag, envelope in sorted(
        results["node_envelope"].items(),
        key=lambda item: int(item[0]),
    ):
        row = {"node_tag": int(node_tag)}
        for i, name in enumerate(dof_names):
            row[f"{name}_max"] = envelope["max"][i]
            row[f"{name}_min"] = envelope["min"][i]
            row[f"{name}_abs_max"] = max(
                abs(envelope["max"][i]),
                abs(envelope["min"][i]),
            )
        rows.append(row)
    return rows


def _result_summary(results, output_dir, elapsed_sec):
    drift_rows = _story_drift_rows(results)
    max_drift_x = max(drift_rows, key=lambda row: row["peak_drift_x_ratio"])
    max_drift_y = max(drift_rows, key=lambda row: row["peak_drift_y_ratio"])
    max_drift_resultant = max(
        drift_rows,
        key=lambda row: row["peak_drift_resultant_ratio"],
    )

    roof_x = results["roof_disp_x"]
    roof_y = results["roof_disp_y"]
    status = results["status"]

    return {
        "status": status,
        "record_summary_x": results["record_summary_x"],
        "record_summary_y": results["record_summary_y"],
        "damping_ratio": results["damping_ratio"],
        "rayleigh_a0": results["rayleigh_a0"],
        "rayleigh_a1": results["rayleigh_a1"],
        "elapsed_sec": elapsed_sec,
        "max_abs_roof_disp_x_in": max((abs(v) for v in roof_x), default=0.0),
        "max_abs_roof_disp_y_in": max((abs(v) for v in roof_y), default=0.0),
        "max_story_drift_x": max_drift_x,
        "max_story_drift_y": max_drift_y,
        "max_story_drift_resultant": max_drift_resultant,
        "output_dir": str(output_dir),
    }


def _deformation_ownership_rows():
    """One row per member end: who owns flexure, bar slip and panel shear there (Model/Deformation_Ownership)."""
    from Model.Deformation_Ownership import end_inventory
    return end_inventory(hinge_registry())


def save_ntha_outputs(output_dir, results, gravity_results, modal_results):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_rows = collect_node_rows()
    element_rows = collect_element_rows()
    edge_rows = collect_graph_edge_rows(element_rows)
    end_force_rows = envelope_to_end_force_rows(
        results["element_envelope"],
        element_rows,
    )
    story_rows = _story_drift_rows(results)
    time_rows = _time_history_rows(results)
    node_env_rows = _node_envelope_rows(results)
    hinge_rows = _hinge_backbone_rows(results)
    joint_rows = _joint_spring_rows(results)
    if joint_rows:
        _write_csv(output_dir / "joint_springs.csv", list(joint_rows[0].keys()), joint_rows)
    ownership_rows = _deformation_ownership_rows()
    if ownership_rows:
        _write_csv(output_dir / "deformation_ownership.csv", list(ownership_rows[0].keys()), ownership_rows)

    _write_json(output_dir / "status.json", results["status"])
    _write_json(output_dir / "record_summary_x.json", results["record_summary_x"])
    if results["record_summary_y"] is not None:
        _write_json(output_dir / "record_summary_y.json", results["record_summary_y"])
    _write_json(output_dir / "gravity_results.json", gravity_results)
    _write_json(output_dir / "modal_results.json", modal_results)
    _write_json(output_dir / "global_parameters.json", collect_global_parameters())
    _write_json(output_dir / "hinge_envelope.json", results["hinge_envelope"])
    _write_response_arrays(output_dir, results)
    if hinge_rows:
        _write_csv(output_dir / "hinge_backbone.csv", list(hinge_rows[0].keys()), hinge_rows)

    _write_csv(output_dir / "nodes.csv", list(node_rows[0].keys()), node_rows)
    _write_csv(output_dir / "elements.csv", list(element_rows[0].keys()), element_rows)
    _write_csv(output_dir / "edges.csv", list(edge_rows[0].keys()), edge_rows)
    _write_csv(output_dir / "time_history.csv", list(time_rows[0].keys()), time_rows)
    _write_csv(output_dir / "story_drift_peaks.csv", list(story_rows[0].keys()), story_rows)
    _write_csv(output_dir / "node_envelope.csv", list(node_env_rows[0].keys()), node_env_rows)
    if end_force_rows:
        _write_csv(
            output_dir / "element_end_force_envelope.csv",
            list(end_force_rows[0].keys()),
            end_force_rows,
        )

    return {
        "num_time_steps": len(time_rows),
        "num_hinges": len(hinge_rows),
        "num_nodes": len(node_rows),
        "num_elements": len(element_rows),
        "num_edges": len(edge_rows),
        "num_element_end_force_envelope_rows": len(end_force_rows),
    }



def first_valid_modal_period(modes):
    for item in modes:
        if item.get("valid") and item.get("period") is not None:
            return item["period"]
    return None


def asce_period_check_from_modal_results(modes, modal_source):
    return sp.asce_fundamental_period_check(
        model_period_sec=first_valid_modal_period(modes),
        modal_source=modal_source,
    )

def build_gravity_modal_state():
    ops.wipe()
    build_model()
    reference_modal_results = run_modal_analysis()
    reference_period_check = asce_period_check_from_modal_results(
        reference_modal_results,
        "elastic_reference_before_gravity",
    )
    apply_gravity_loads()
    gravity_results = run_gravity_analysis()
    post_gravity_modal_results = run_modal_analysis()
    tangent_period_check = asce_period_check_from_modal_results(
        post_gravity_modal_results,
        "post_gravity_tangent_diagnostic",
    )
    return gravity_results, reference_modal_results, post_gravity_modal_results, reference_period_check, tangent_period_check


def run_one(record_x, record_y, args, output_dir):
    gravity_results, reference_modal_results, modal_results, reference_period_check, tangent_period_check = build_gravity_modal_state()

    start = time.perf_counter()
    results = run_ntha(
        record_x,
        record_y=record_y,
        damping_ratio=args.damping_ratio,
        modal_results=reference_modal_results,
        rayleigh_mode_i=args.rayleigh_mode_i,
        rayleigh_mode_j=args.rayleigh_mode_j,
        dt_factor=args.dt_factor,
        log_path=output_dir / "opensees_ntha.log",
    )
    elapsed = time.perf_counter() - start

    file_counts = save_ntha_outputs(
        output_dir,
        results,
        gravity_results,
        reference_modal_results,
    )
    _write_json(output_dir / "modal_results_elastic_reference.json", reference_modal_results)
    _write_json(output_dir / "modal_results_post_gravity_tangent.json", modal_results)
    reference_valid_modes = [mode for mode in reference_modal_results if mode.get("valid")]
    tangent_valid_modes = [mode for mode in modal_results if mode.get("valid")]
    modal_diagnostics = {
        "rayleigh_modal_source": "elastic_reference_before_gravity",
        "elastic_reference_period_mode_1_sec": reference_valid_modes[0]["period"] if len(reference_valid_modes) > 0 else None,
        "elastic_reference_period_mode_2_sec": reference_valid_modes[1]["period"] if len(reference_valid_modes) > 1 else None,
        "post_gravity_tangent_period_mode_1_sec": tangent_valid_modes[0]["period"] if len(tangent_valid_modes) > 0 else None,
        "post_gravity_tangent_period_mode_2_sec": tangent_valid_modes[1]["period"] if len(tangent_valid_modes) > 1 else None,
        "gravity_roof_ux_in": gravity_results.get("ux"),
        "gravity_roof_uy_in": gravity_results.get("uy"),
        "asce_reference_period_check": reference_period_check,
        "asce_post_gravity_period_check": tangent_period_check,
    }
    _write_json(output_dir / "modal_diagnostics.json", modal_diagnostics)

    summary = _result_summary(results, output_dir, elapsed)
    summary.update(file_counts)
    summary.update(modal_diagnostics)
    _write_json(output_dir / "summary.json", summary)

    status = results["status"]
    drift = summary["max_story_drift_resultant"]
    print(
        f"NTHA {record_x.record_id}"
        f"{' + ' + record_y.record_id if record_y else ''}: "
        f"{status['completed_steps']}/{status['npts_requested']} steps, "
        f"failed={status['failed']}, "
        f"max resultant story drift={100.0 * drift['peak_drift_resultant_ratio']:.3f}% "
        f"at story {drift['story']}"
    )

    return summary


def select_records(args):
    if args.record_id_x:
        record_x = load_ground_motion_record(
            args.record_id_x,
            scale_factor=args.scale_factor,
        )
        record_y = None
        if args.record_id_y and not args.x_only:
            record_y = load_ground_motion_record(
                args.record_id_y,
                scale_factor=args.scale_factor,
            )
        return [(f"manual:{record_x.record_id}", record_x, record_y)]

    if args.result_id is not None:
        key, record_x, record_y = load_ground_motion_pair_by_result_id(
            args.result_id,
            set_name=args.set_name,
            split=args.split,
            scale_factor=args.scale_factor,
        )
        if args.x_only:
            record_y = None
        return [(key, record_x, record_y)]

    pairs = load_ground_motion_pairs(
        set_name=args.set_name,
        split=args.split,
        limit=args.limit,
        start_index=args.start_index,
        scale_factor=args.scale_factor,
    )
    if args.x_only:
        pairs = [(key, record_x, None) for key, record_x, _record_y in pairs]
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RC Structure nonlinear time-history analyses from the ground-motion manifest."
    )
    parser.add_argument("--record-id-x", help="Specific manifest record_id for X-direction excitation.")
    parser.add_argument("--record-id-y", help="Specific manifest record_id for Y-direction excitation.")
    parser.add_argument("--result-id", type=int, help="PEER search result_id pair to run.")
    parser.add_argument("--set-name", default="peer_mle_all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--x-only", action="store_true", help="Run only the X component.")
    parser.add_argument("--scale-factor", type=float, default=None)
    parser.add_argument("--damping-ratio", type=float, default=0.05)
    parser.add_argument("--rayleigh-mode-i", type=int, default=0)
    parser.add_argument("--rayleigh-mode-j", type=int, default=2)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "ntha"),
    )
    parser.add_argument("--catalog-summary", action="store_true")
    parser.add_argument("--design-only", action="store_true",
                        help="Create/review a design candidate only; never run ground-motion analysis.")
    parser.add_argument(
        "--design-file",
        default=None,
        help=(
            "ACI design artifact for this structure. Defaults to "
            f"<output-dir>/../{DESIGN_ARTIFACT_NAME}, so every record and "
            "intensity run of one case shares a single design. Created on "
            "first use and reused afterwards."
        ),
    )
    parser.add_argument(
        "--skip-design",
        action="store_true",
        help=(
            "Analyze with the Structure_Parameters sections instead of running "
            "the design loop. Reproduces pre-design behaviour; not for dataset "
            "generation, where it would give every structure identical members."
        ),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow replacing an existing run even when its model configuration differs.",
    )
    add_geometry_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    geometry_name = apply_geometry_from_args(args)
    if args.design_only and args.skip_design:
        raise ValueError("--design-only cannot be combined with --skip-design.")

    if args.catalog_summary:
        print(json.dumps(ground_motion_catalog_summary(), indent=2))
        if not args.record_id_x and args.result_id is None:
            return

    if not args.skip_design and not args.design_only:
        ensure_generation_release_ready()
    selected = [] if args.design_only else select_records(args)
    if not selected and not args.design_only:
        raise RuntimeError("No ground-motion records selected.")

    base_output_dir = Path(args.output_dir)
    default_output_dir = Path(__file__).resolve().parent / "outputs" / "ntha"
    if geometry_name and base_output_dir == default_output_dir:
        base_output_dir = base_output_dir / geometry_name
    design_record = None
    if args.skip_design:
        print()
        print(
            '[design] SKIPPED -- analyzing with the Structure_Parameters '
            'defaults. Every structure will share identical members.'
        )
    else:
        design_path = (
            Path(args.design_file)
            if args.design_file
            else base_output_dir.parent / DESIGN_ARTIFACT_NAME
        )
        design_record, created = load_or_create_design(design_path, verbose=True)
        sections = design_record['sections']
        dcr = design_record['dcr']
        print()
        print(f"[design] {'created' if created else 'reused'} {design_path}")
        print(
            f"  columns {sections['b_col_in']:g}x{sections['h_col_in']:g} in, "
            f"fc {sections['fc_col_ksi']:g} ksi"
        )
        print(
            f"  beams   {sections['b_beam_in']:g}x{sections['h_beam_in']:g} in, "
            f"fc {sections['fc_beam_ksi']:g} ksi"
        )
        print(
            f"  DCR beam {dcr['beam']:.3f} "
            f"(band {dcr['band_lo']:.2f}-{dcr['band_hi']:.2f}), "
            f"column {dcr['column']:.3f}; governed by {dcr['governed_by']}"
        )
        # DCR success and numerical convergence do not establish SMRF design.
        # Keep the candidate artifact for diagnosis, but never export this as
        # an accepted SMRF dataset case with missing/failed design checks.
        if args.design_only:
            report = design_record["qualification"]
            print(f"[design-only] {report['counts']}; SMRF accepted={report['accepted']}. No NTHA launched.")
            return
        require_accepted_design(design_record)

    run_summaries = []

    for key, record_x, record_y in selected:
        name = _safe_name(key.replace("peer_result_id:", "peer_"))
        if args.record_id_x:
            name = _safe_name(
                record_x.record_id + (f"__{record_y.record_id}" if record_y else "__x_only")
            )
        name = analysis_run_name(
            name,
            x_only=record_y is None,
            scale_factor=record_x.scale_factor,
            damping_ratio=args.damping_ratio,
            rayleigh_mode_i=args.rayleigh_mode_i,
            rayleigh_mode_j=args.rayleigh_mode_j,
            dt_factor=args.dt_factor,
        )

        output_dir = base_output_dir / name
        validate_ntha_output_compatibility(
            output_dir,
            overwrite_existing=args.overwrite_existing,
        )
        summary = run_one(record_x, record_y, args, output_dir)
        summary["pair_key"] = key
        summary["design"] = design_record
        run_summaries.append(summary)

    _write_json(base_output_dir / "ntha_batch_summary.json", run_summaries)
    print(f"Saved NTHA batch summary: {base_output_dir / 'ntha_batch_summary.json'}")


if __name__ == "__main__":
    main()
