import csv
import json
import math
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.Build_Model import floor_master_node, node_tag
from Model.IMK_Calibration import bond_slip_indicator


ELEMENT_TYPE_ID = {
    "column": 0,
    "beam_x": 1,
    "beam_y": 2,
}


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(data, file, indent=2)

    return path


def _node_grid_indices(tag):
    nodes_per_floor = (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    zero_based = tag - 1
    k = zero_based // nodes_per_floor
    floor_offset = zero_based % nodes_per_floor
    j = floor_offset // (sp.NUM_BAY_X + 1)
    i = floor_offset % (sp.NUM_BAY_X + 1)
    return k, i, j


def collect_node_rows():
    rows = []

    for k in range(sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                tag = node_tag(k, i, j)
                x, y, z = ops.nodeCoord(tag)
                rows.append(
                    {
                        "node_tag": tag,
                        "floor": k,
                        "grid_i": i,
                        "grid_j": j,
                        "x_in": x,
                        "y_in": y,
                        "z_in": z,
                        "is_base": int(k == 0),
                        "is_floor_master": int(
                            k > 0 and tag == floor_master_node(k)
                        ),
                    }
                )

    return rows


def collect_diaphragm_master_rows():
    rows = []

    for k in range(1, sp.NUM_FLOOR + 1):
        tag = floor_master_node(k)
        x, y, z = ops.nodeCoord(tag)
        rows.append(
            {
                "node_tag": tag,
                "floor": k,
                "x_in": x,
                "y_in": y,
                "z_in": z,
                "ux_in": ops.nodeDisp(tag, 1),
                "uy_in": ops.nodeDisp(tag, 2),
                "uz_in": ops.nodeDisp(tag, 3),
                "rx_rad": ops.nodeDisp(tag, 4),
                "ry_rad": ops.nodeDisp(tag, 5),
                "rz_rad": ops.nodeDisp(tag, 6),
            }
        )

    return rows


def _member_geometry(n_i, n_j):
    xi, yi, zi = ops.nodeCoord(n_i)
    xj, yj, zj = ops.nodeCoord(n_j)

    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    length = math.sqrt(dx * dx + dy * dy + dz * dz)

    if length <= 0.0:
        raise ValueError(f"Element has zero length between nodes {n_i} and {n_j}.")

    return {
        "length_in": length,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "dir_x": dx / length,
        "dir_y": dy / length,
        "dir_z": dz / length,
    }


def _element_property_row(ele_tag, n_i, n_j, element_type):
    geom = _member_geometry(n_i, n_j)

    if element_type == "column":
        sec_tag = sp.COL_SEC_TAG
        transf_tag = sp.COL_TRANSF_TAG
        integ_tag = sp.COL_INTEG_TAG
        b = sp.B_COL
        h = sp.H_COL
        fc = sp.FC_COL_KSI
        bar_size = sp.COL_BAR_SIZE
        bar_area = sp.COL_BAR_AREA
        top_bars = sp.COL_TOP_BARS
        bot_bars = sp.COL_BOT_BARS
        side_bars = sp.COL_SIDE_BARS
    else:
        sec_tag = sp.BEAM_SEC_TAG
        transf_tag = (
            sp.BEAM_X_TRANSF_TAG
            if element_type == "beam_x"
            else sp.BEAM_Y_TRANSF_TAG
        )
        integ_tag = sp.BEAM_INTEG_TAG
        b = sp.B_BEAM
        h = sp.H_BEAM
        fc = sp.FC_BEAM_KSI
        bar_size = sp.BEAM_BAR_SIZE
        bar_area = sp.BEAM_BAR_AREA
        top_bars = sp.BEAM_TOP_BARS
        bot_bars = sp.BEAM_BOT_BARS
        side_bars = sp.BEAM_SIDE_BARS

    row = {
        "ele_tag": ele_tag,
        "node_i": n_i,
        "node_j": n_j,
        "element_type": element_type,
        "element_type_id": ELEMENT_TYPE_ID[element_type],
        "section_tag": sec_tag,
        "transf_tag": transf_tag,
        "integration_tag": integ_tag,
        "b_in": b,
        "h_in": h,
        "area_in2": sp.rect_area(b, h),
        "fc_ksi": fc,
        "fy_ksi": sp.FY_KSI,
        "bar_size": bar_size,
        "bar_area_in2": bar_area,
        "top_bars": top_bars,
        "bot_bars": bot_bars,
        "side_bars": side_bars,
    }
    row.update(geom)
    return row


def collect_element_rows():
    rows = []
    ele_tag = 1

    for k in range(sp.NUM_FLOOR):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k + 1, i, j)
                rows.append(_element_property_row(ele_tag, n_i, n_j, "column"))
                ele_tag += 1

    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k, i + 1, j)
                rows.append(_element_property_row(ele_tag, n_i, n_j, "beam_x"))
                ele_tag += 1

    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k, i, j + 1)
                rows.append(_element_property_row(ele_tag, n_i, n_j, "beam_y"))
                ele_tag += 1

    return rows


def collect_graph_edge_rows(element_rows):
    rows = []

    for element in element_rows:
        for source, target in (
            (element["node_i"], element["node_j"]),
            (element["node_j"], element["node_i"]),
        ):
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "ele_tag": element["ele_tag"],
                    "element_type": element["element_type"],
                    "element_type_id": element["element_type_id"],
                    "length_in": element["length_in"],
                    "dir_x": element["dir_x"],
                    "dir_y": element["dir_y"],
                    "dir_z": element["dir_z"],
                    "b_in": element["b_in"],
                    "h_in": element["h_in"],
                    "area_in2": element["area_in2"],
                    "fc_ksi": element["fc_ksi"],
                    "fy_ksi": element["fy_ksi"],
                    "bar_area_in2": element["bar_area_in2"],
                    "top_bars": element["top_bars"],
                    "bot_bars": element["bot_bars"],
                    "side_bars": element["side_bars"],
                }
            )

    return rows


def collect_node_displacement_rows(node_rows):
    rows = []

    for node in node_rows:
        tag = node["node_tag"]
        rows.append(
            {
                "node_tag": tag,
                "ux_in": ops.nodeDisp(tag, 1),
                "uy_in": ops.nodeDisp(tag, 2),
                "uz_in": ops.nodeDisp(tag, 3),
                "rx_rad": ops.nodeDisp(tag, 4),
                "ry_rad": ops.nodeDisp(tag, 5),
                "rz_rad": ops.nodeDisp(tag, 6),
            }
        )

    return rows


def collect_node_reaction_rows(node_rows):
    ops.reactions()
    rows = []

    for node in node_rows:
        tag = node["node_tag"]
        rows.append(
            {
                "node_tag": tag,
                "rxn_x_kip": ops.nodeReaction(tag, 1),
                "rxn_y_kip": ops.nodeReaction(tag, 2),
                "rxn_z_kip": ops.nodeReaction(tag, 3),
                "rxn_mx_kip_in": ops.nodeReaction(tag, 4),
                "rxn_my_kip_in": ops.nodeReaction(tag, 5),
                "rxn_mz_kip_in": ops.nodeReaction(tag, 6),
            }
        )

    return rows


def _local_element_force(ele_tag):
    forces = ops.eleResponse(ele_tag, "localForce")
    if not forces:
        forces = ops.eleForce(ele_tag)
    if len(forces) != 12:
        raise ValueError(
            f"Expected 12 element end-force values for element {ele_tag}, "
            f"got {len(forces)}."
        )
    return forces


def collect_element_end_force_rows(element_rows):
    rows = []

    for element in element_rows:
        ele_tag = element["ele_tag"]
        forces = _local_element_force(ele_tag)

        for end_name, node_key, offset in (
            ("i", "node_i", 0),
            ("j", "node_j", 6),
        ):
            rows.append(
                {
                    "ele_tag": ele_tag,
                    "node_tag": element[node_key],
                    "end": end_name,
                    "element_type": element["element_type"],
                    "axial_kip": forces[offset + 0],
                    "shear_y_kip": forces[offset + 1],
                    "shear_z_kip": forces[offset + 2],
                    "torsion_kip_in": forces[offset + 3],
                    "moment_y_kip_in": forces[offset + 4],
                    "moment_z_kip_in": forces[offset + 5],
                }
            )

    return rows


def _node_feature_array(node_rows):
    return np.array(
        [
            [
                row["x_in"],
                row["y_in"],
                row["z_in"],
                row["floor"],
                row["grid_i"],
                row["grid_j"],
                row["is_base"],
                row["is_floor_master"],
            ]
            for row in node_rows
        ],
        dtype=np.float32,
    )


def _edge_arrays(edge_rows):
    edge_index = np.array(
        [[row["source"] - 1, row["target"] - 1] for row in edge_rows],
        dtype=np.int64,
    ).T

    edge_attr = np.array(
        [
            [
                row["element_type_id"],
                row["length_in"],
                row["dir_x"],
                row["dir_y"],
                row["dir_z"],
                row["b_in"],
                row["h_in"],
                row["area_in2"],
                row["fc_ksi"],
                row["fy_ksi"],
                row["bar_area_in2"],
                row["top_bars"],
                row["bot_bars"],
                row["side_bars"],
            ]
            for row in edge_rows
        ],
        dtype=np.float32,
    )

    return edge_index, edge_attr


def _pushover_array(pushover_results):
    return np.array(
        list(zip(pushover_results["roof_disp"], pushover_results["base_shear"])),
        dtype=np.float32,
    )


def _node_displacement_array(displacement_rows):
    return np.array(
        [
            [
                row["ux_in"],
                row["uy_in"],
                row["uz_in"],
                row["rx_rad"],
                row["ry_rad"],
                row["rz_rad"],
            ]
            for row in displacement_rows
        ],
        dtype=np.float32,
    )


def _node_reaction_array(reaction_rows):
    return np.array(
        [
            [
                row["rxn_x_kip"],
                row["rxn_y_kip"],
                row["rxn_z_kip"],
                row["rxn_mx_kip_in"],
                row["rxn_my_kip_in"],
                row["rxn_mz_kip_in"],
            ]
            for row in reaction_rows
        ],
        dtype=np.float32,
    )


def _element_end_force_array(force_rows):
    return np.array(
        [
            [
                row["ele_tag"],
                row["node_tag"],
                0 if row["end"] == "i" else 1,
                ELEMENT_TYPE_ID[row["element_type"]],
                row["axial_kip"],
                row["shear_y_kip"],
                row["shear_z_kip"],
                row["torsion_kip_in"],
                row["moment_y_kip_in"],
                row["moment_z_kip_in"],
            ]
            for row in force_rows
        ],
        dtype=np.float32,
    )


def collect_force_diagnostics(force_rows):
    diagnostics = {}

    for element_type in sorted({row["element_type"] for row in force_rows}):
        rows = [row for row in force_rows if row["element_type"] == element_type]
        diagnostics[element_type] = {
            "max_abs_axial_kip": max(abs(row["axial_kip"]) for row in rows),
            "max_abs_shear_y_kip": max(abs(row["shear_y_kip"]) for row in rows),
            "max_abs_shear_z_kip": max(abs(row["shear_z_kip"]) for row in rows),
            "max_abs_torsion_kip_in": max(abs(row["torsion_kip_in"]) for row in rows),
            "max_abs_moment_y_kip_in": max(
                abs(row["moment_y_kip_in"]) for row in rows
            ),
            "max_abs_moment_z_kip_in": max(
                abs(row["moment_z_kip_in"]) for row in rows
            ),
        }

    return diagnostics


def collect_reinforcement_geometry():
    """Selected SMRF reinforcement geometry, separate from the tensor schema.

    Legacy COVER combined core and bar-centroid offsets; its clear cover and
    exposure cannot be reconstructed safely. Return None for legacy mode.
    Core dimensions below describe the actual fiber partition at the outside
    of hoops, not a validation of its confinement constitutive properties.
    """
    if sp.SLAB_THICKNESS_IN is None:
        return None
    members = {}
    for member, prefix in (("column", "COL"), ("beam", "BEAM")):
        b, h = getattr(sp, f"B_{prefix}"), getattr(sp, f"H_{prefix}")
        core_cover = sp.core_cover_in(member)
        bar_cover = sp.longitudinal_cover_in(member)
        bar_size = getattr(sp, f"{prefix}_BAR_SIZE")
        hoop_size = getattr(sp, f"{prefix}_STIRRUP_BAR_SIZE")
        hoop_diameter = sp.rebar_diameter(hoop_size)
        if any(not math.isfinite(v) or v <= 0 for v in (b, h, core_cover, bar_cover)):
            raise ValueError("Reinforcement export requires finite positive geometry.")
        if 2 * bar_cover >= min(b, h):
            raise ValueError("Reinforcement bar centroids must fit inside the exported section.")
        members[member] = {
            "clear_cover_outside_hoops_in": core_cover,
            "longitudinal_centroid_offset_in": bar_cover,
            "longitudinal_bar_diameter_in": sp.rebar_diameter(bar_size),
            "stirrup_bar_diameter_in": hoop_diameter,
            "stirrup_bar_size": hoop_size,
            "stirrup_legs": getattr(sp, f"{prefix}_STIRRUP_LEGS"),
            "stirrup_spacing_in": getattr(sp, f"{prefix}_STIRRUP_SPACING"),
            "fiber_core_boundary_offset_in": core_cover,
            "fiber_core_width_in": b - 2 * core_cover,
            "fiber_core_depth_in": h - 2 * core_cover,
            "fiber_core_area_in2": (b - 2 * core_cover) * (h - 2 * core_cover),
            "hoop_centerline_width_in": b - 2 * core_cover - hoop_diameter,
            "hoop_centerline_depth_in": h - 2 * core_cover - hoop_diameter,
        }
    aggregate = sp.AGGREGATE_MAX_SIZE_IN
    if isinstance(aggregate, bool) or not math.isfinite(aggregate) or aggregate <= 0:
        raise ValueError("Reinforcement export requires a finite positive maximum aggregate size.")
    return {
        "schema_version": "smrf_reinforcement_geometry_v1",
        "cover_basis": "clear to outside of hoop; longitudinal centroid = clear + hoop diameter + half bar diameter",
        "fiber_core_basis": "boundary at outside of hoops; geometry only, confinement material calibration not implied",
        "aggregate_max_size_in": aggregate,
        "materials": {
            "reinforcement_specification": getattr(sp, "REINFORCEMENT_SPECIFICATION", None),
            "exposure": getattr(sp, "MATERIAL_EXPOSURE", None),
            "fy_ksi": sp.FY_KSI,
            "es_ksi": sp.ES_KSI,
            "concrete_unit_weight_kcf": sp.CONCRETE_UNIT_WEIGHT_KCF,
        },
        **members,
    }


def installed_deterioration_policy():
    """What the member springs are actually handed as deterioration capacities.

    The IMK_LAMBDA_* constants describe the direct convention only. Under
    IMK_DETERIORATION_MODE == "haselton_2008" each member receives
    Lamda_S = Lamda_C = lambda(nu, s/d) * theta_y,member with A and K
    suppressed, so the policy that fixes those values is recorded here (the
    per-member nu makes the column values member specific; the beam and the
    nu = 0 column values are quoted as anchors). Part of the output identity.
    """
    from Model.IMK_Calibration import deterioration_for_member   # lazy: avoids a cycle at import time
    mode = getattr(sp, "IMK_DETERIORATION_MODE", "direct")
    modes = ["S", "C", "K"] + (["A"] if sp.IMK_MATERIAL_TYPE == "IMKPeakOriented" else [])
    if mode == "direct":
        return {"mode": "direct", "basis": "IMK_LAMBDA_* constants passed unchanged (E_ref = Lamda * My)",
                "lamda": {m: getattr(sp, f"IMK_LAMBDA_{m}") for m in modes}}
    anchors = {}
    for member_type in ("beam_x", "column"):
        d = deterioration_for_member(member_type, 0.0)
        anchors["beam" if member_type == "beam_x" else "column_nu_0"] = {
            "spacing_depth_ratio": d["deterioration_spacing_depth_ratio"],
            "theta_y_member_rad": d["energy_reference_member_theta_y_rad"],
            "lambda_haselton": d["lambda_haselton_dimensionless"],
            "lamda_opensees_rad": d["lambda_opensees_rad"],
        }
    reference = deterioration_for_member("column", 0.0)
    return {"mode": mode, "basis": reference["deterioration_source"],
            "calibrated_modes": [m for m in modes if m not in reference["deterioration_suppressed_modes"]],
            "suppressed_modes": {m: reference["deterioration_suppression_lambda_rad"]
                                 for m in reference["deterioration_suppressed_modes"] if m in modes},
            "axial_ratio_clamp": [0.0, 0.70],
            "anchors": anchors}


def collect_global_parameters():
    return {
        "num_bay_x": sp.NUM_BAY_X,
        "num_bay_y": sp.NUM_BAY_Y,
        "num_floor": sp.NUM_FLOOR,
        "bay_x_in": sp.BAY_X,
        "bay_y_in": sp.BAY_Y,
        "story_h_in": sp.STORY_H,
        "fc_col_ksi": sp.FC_COL_KSI,
        "fc_beam_ksi": sp.FC_BEAM_KSI,
        "fy_ksi": sp.FY_KSI,
        "b_col_in": sp.B_COL,
        "h_col_in": sp.H_COL,
        "b_beam_in": sp.B_BEAM,
        "h_beam_in": sp.H_BEAM,
        "col_bar_size": sp.COL_BAR_SIZE,
        "col_top_bars": sp.COL_TOP_BARS,
        "col_bot_bars": sp.COL_BOT_BARS,
        "col_side_bars": sp.COL_SIDE_BARS,
        "col_bar_area_in2": sp.COL_BAR_AREA,
        "col_stirrup_bar_size": sp.COL_STIRRUP_BAR_SIZE,
        "col_stirrup_legs": sp.COL_STIRRUP_LEGS,
        "col_stirrup_spacing_in": sp.COL_STIRRUP_SPACING,
        "col_stirrup_area_in2": sp.COL_STIRRUP_LEGS * sp.rebar_area(sp.COL_STIRRUP_BAR_SIZE),
        "beam_bar_size": sp.BEAM_BAR_SIZE,
        "beam_top_bars": sp.BEAM_TOP_BARS,
        "beam_bot_bars": sp.BEAM_BOT_BARS,
        "beam_side_bars": sp.BEAM_SIDE_BARS,
        "beam_bar_area_in2": sp.BEAM_BAR_AREA,
        "beam_stirrup_bar_size": sp.BEAM_STIRRUP_BAR_SIZE,
        "beam_stirrup_legs": sp.BEAM_STIRRUP_LEGS,
        "beam_stirrup_spacing_in": sp.BEAM_STIRRUP_SPACING,
        "beam_stirrup_area_in2": sp.BEAM_STIRRUP_LEGS * sp.rebar_area(sp.BEAM_STIRRUP_BAR_SIZE),
        "floor_dead_load_ksf": sp.floor_dead_load_ksf(),
        "floor_loads": sp.floor_load_metadata() if sp.SLAB_THICKNESS_IN is not None else None,
        "reinforcement_geometry": collect_reinforcement_geometry(),
        "floor_live_load_ksf": sp.FLOOR_LIVE_LOAD_KSF,
        "floor_load_ksi": sp.floor_load_ksi(),
        "total_floor_gravity_load_kip": sp.total_floor_gravity_load(),
        "fx_floor_kip": sp.FX_FLOOR,
        "element_formulation": sp.ELEMENT_FORMULATION,
        "imk_apply_to_columns": sp.IMK_APPLY_TO_COLUMNS,
        "imk_apply_to_beams": sp.IMK_APPLY_TO_BEAMS,
        "imk_material_type": sp.IMK_MATERIAL_TYPE,
        "imk_energy_convention": sp.IMK_ENERGY_CONVENTION,
        "imk_energy_mapping_mode": sp.IMK_ENERGY_MAPPING_MODE,
        "imk_member_energy_calibrations": sp.IMK_MEMBER_ENERGY_CALIBRATIONS,
        "imk_cyclic_calibration_id": sp.IMK_CYCLIC_CALIBRATION_ID,
        "imk_cyclic_calibration_status": sp.IMK_CYCLIC_CALIBRATION_STATUS,
        "imk_cyclic_parameters": {
            "lamda_s": sp.IMK_LAMBDA_S, "lamda_c": sp.IMK_LAMBDA_C, "lamda_k": sp.IMK_LAMBDA_K,
            "lamda_a": sp.IMK_LAMBDA_A if sp.IMK_MATERIAL_TYPE == "IMKPeakOriented" else None,
            "c_s": sp.IMK_C_S, "c_c": sp.IMK_C_C, "c_k": sp.IMK_C_K,
            "c_a": sp.IMK_C_A if sp.IMK_MATERIAL_TYPE == "IMKPeakOriented" else None,
            "d_pos": sp.IMK_D_POS, "d_neg": sp.IMK_D_NEG,
        },
        # What the springs are actually given. Under haselton_2008 the IMK_LAMBDA_* constants above are
        # NOT installed: S and C receive lambda(nu, s/d) * theta_y,member per member and A/K are
        # suppressed, so the mode and its policy are part of the output identity (2026-09-26).
        "imk_deterioration_mode": getattr(sp, "IMK_DETERIORATION_MODE", "direct"),
        "imk_use_calibrated_backbone": sp.IMK_USE_CALIBRATED_BACKBONE,
        "imk_installed_deterioration": installed_deterioration_policy(),
        "imk_hinge_stiffness_mode": getattr(sp, "IMK_HINGE_STIFFNESS_MODE", "member_stiffness_factor"),
        "imk_hinge_stiffness_factor": sp.IMK_HINGE_STIFFNESS_FACTOR,
        # Joint springs and the member slip term they take over (2026-09-27).
        "joint_model": getattr(sp, "JOINT_MODEL", "rigid_centerline"),
        "joint_deformation_scope": getattr(sp, "JOINT_DEFORMATION_SCOPE", "joint_shear_only"),
        "joint_calibration_basis": getattr(sp, "JOINT_CALIBRATION_BASIS", None),
        "joint_calibration_status": getattr(sp, "JOINT_CALIBRATION_STATUS", None),
        "joint_stiffness_modifier": getattr(sp, "JOINT_STIFFNESS_MODIFIER", None),
        "joint_kappa": {"f": getattr(sp, "JOINT_KAPPA_F", None), "d": getattr(sp, "JOINT_KAPPA_D", None)},
        "joint_lambda_suppression": getattr(sp, "JOINT_LAMBDA_SUPPRESSION", None),
        "imk_bond_slip_indicator": bond_slip_indicator(),
        "imk_beam_theta_y": sp.IMK_BEAM_THETA_Y,
        "imk_column_theta_y": sp.IMK_COLUMN_THETA_Y,
        "imk_theta_p_pos": sp.IMK_THETA_P_POS,
        "imk_theta_pc_pos": sp.IMK_THETA_PC_POS,
        "imk_theta_u_pos": sp.IMK_THETA_U_POS,
        "imk_res_pos": sp.IMK_RES_POS,
        "imk_fmaxfy_pos": getattr(sp, "IMK_FMAXFY_POS", 1.10),
        "imk_fmaxfy_neg": getattr(sp, "IMK_FMAXFY_NEG", 1.10),
        "imk_fresfy_pos": getattr(sp, "IMK_FRESFY_POS", sp.IMK_RES_POS),
        "imk_fresfy_neg": getattr(sp, "IMK_FRESFY_NEG", sp.IMK_RES_NEG),
        "pushover_steps_requested": sp.PUSHOVER_STEPS,
        "pushover_du_in": sp.PUSHOVER_DU,
        "pushover_target_disp_in": sp.pushover_target_disp(),
        "roof_control_node": floor_master_node(sp.NUM_FLOOR),
    }


def save_analysis_sample(
    sample_dir,
    gravity_results,
    modal_results,
    pushover_results,
    design_status=None,
    design_check_results=None,
):
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    _write_json(sample_dir / "pushover_status.json", pushover_results["status"])
    _write_json(sample_dir / "pushover_convergence.json", pushover_results["convergence"])

    node_rows = collect_node_rows()
    diaphragm_master_rows = collect_diaphragm_master_rows()
    element_rows = collect_element_rows()
    edge_rows = collect_graph_edge_rows(element_rows)
    node_displacement_rows = collect_node_displacement_rows(node_rows)
    node_reaction_rows = collect_node_reaction_rows(node_rows)
    element_end_force_rows = collect_element_end_force_rows(element_rows)
    force_diagnostics = collect_force_diagnostics(element_end_force_rows)
    global_parameters = collect_global_parameters()

    _write_csv(sample_dir / "nodes.csv", list(node_rows[0].keys()), node_rows)
    _write_csv(
        sample_dir / "diaphragm_masters.csv",
        list(diaphragm_master_rows[0].keys()),
        diaphragm_master_rows,
    )
    _write_csv(sample_dir / "elements.csv", list(element_rows[0].keys()), element_rows)
    _write_csv(sample_dir / "edges.csv", list(edge_rows[0].keys()), edge_rows)
    _write_csv(
        sample_dir / "node_displacements.csv",
        list(node_displacement_rows[0].keys()),
        node_displacement_rows,
    )
    _write_csv(
        sample_dir / "node_reactions.csv",
        list(node_reaction_rows[0].keys()),
        node_reaction_rows,
    )
    _write_csv(
        sample_dir / "element_end_forces.csv",
        list(element_end_force_rows[0].keys()),
        element_end_force_rows,
    )

    _write_json(sample_dir / "global_parameters.json", global_parameters)
    _write_json(sample_dir / "gravity_results.json", gravity_results)
    _write_json(sample_dir / "modal_results.json", modal_results)
    _write_json(
        sample_dir / "failed_element_diagnostics.json",
        pushover_results["failed_element_diagnostics"],
    )
    _write_json(
        sample_dir / "mechanism_diagnostics.json",
        pushover_results["mechanism_diagnostics"],
    )
    _write_json(sample_dir / "force_diagnostics.json", force_diagnostics)
    if design_status is not None:
        _write_json(sample_dir / "design_status.json", design_status)
    design_check_paths = {}
    if design_check_results is not None:
        for name, results in design_check_results.items():
            path = sample_dir / f"design_check_{name}.json"
            _write_json(path, {str(k): v for k, v in results.items()})
            design_check_paths[f"design_check_{name}_json"] = str(path)

    pushover_rows = [
        {
            "step": step,
            "roof_disp_x_in": disp,
            "roof_drift_ratio": pushover_results["roof_drift"][step],
            "base_shear_x_kip": shear,
            "load_factor": pushover_results["load_factors"][step],
        }
        for step, (disp, shear) in enumerate(
            zip(pushover_results["roof_disp"], pushover_results["base_shear"])
        )
    ]
    _write_csv(
        sample_dir / "pushover_response.csv",
        [
            "step",
            "roof_disp_x_in",
            "roof_drift_ratio",
            "base_shear_x_kip",
            "load_factor",
        ],
        pushover_rows,
    )

    edge_index, edge_attr = _edge_arrays(edge_rows)
    np.savez_compressed(
        sample_dir / "graph_sample.npz",
        x=_node_feature_array(node_rows),
        edge_index=edge_index,
        edge_attr=edge_attr,
        pushover=_pushover_array(pushover_results),
        node_displacements=_node_displacement_array(node_displacement_rows),
        node_reactions=_node_reaction_array(node_reaction_rows),
        element_end_forces=_element_end_force_array(element_end_force_rows),
    )

    summary = {
        "num_nodes": len(node_rows),
        "num_diaphragm_master_nodes": len(diaphragm_master_rows),
        "num_elements": len(element_rows),
        "num_directed_edges": len(edge_rows),
        "num_element_end_force_rows": len(element_end_force_rows),
        "num_failed_element_diagnostics": len(
            pushover_results["failed_element_diagnostics"]
        ),
        "num_hinge_events": len(
            pushover_results["mechanism_diagnostics"]["hinge_events"]
        ),
        "num_column_yield_events": len(
            pushover_results["mechanism_diagnostics"]["column_yield_events"]
        ),
        "num_pushover_steps": len(pushover_results["roof_disp"]),
        "graph_sample_npz": str(sample_dir / "graph_sample.npz"),
    }
    if design_status is not None:
        summary["design_status_json"] = str(sample_dir / "design_status.json")
    summary.update(design_check_paths)
    _write_json(sample_dir / "sample_summary.json", summary)

    return summary
