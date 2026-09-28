"""Beam-column joint springs for the nonlinear frame: IMKPinching in the two vertical shear planes.

Topology (the centreline variant of the orthogonal scissors panel in Model/JOINT_PANEL.md): every
elevated joint node J stays the column core, keeping its tag, mass, diaphragm slaving and the column
ends. A coincident beam core B = JOINT_BEAM_CORE_TAG_BASE + J shares translations and the vertical-axis
rotation with J (equalDOF 1, 2, 3, 6). The beams frame into B, the columns into J, and one zeroLength
element joins the cores with IMKPinching in global Rx (direction 4, the y-z plane, shear from beams
spanning Y) and global Ry (direction 5, the x-z plane, shear from beams spanning X). Members keep the
centreline geometry, lengths, loads and masses of the rigid-joint frame; the joint has no finite size.

Calibration per joint and plane (Structure_Parameters.JOINT_CALIBRATION_BASIS):
  strength     Vn = gamma sqrt(f'c) Aj (ACI 318-19 Table 18.8.4.3, Aj per 15.4.2.4). gamma, Aj and Vn are
               taken from the design record's saved joint-shear category when the record carries one
               (sp.JOINT_SHEAR_CATEGORIES, set by Design_Driver.apply_design); otherwise they are
               classified here from the framing with reinforcement continuity assumed.
  conjugacy    The spring rotation is the panel shear strain (JOINT_PANEL.md), so M = tau Aj h_b:
               Mn = Vn h_b, K = JOINT_STIFFNESS_MODIFIER G Aj h_b with G = Ec / (2 (1 + nu)),
               Ec = 57 sqrt(f'c) ksi (ACI 19.2.2.1), nu = 0.2; theta_y = tau_n / G.
  deformation  ASCE/SEI 41-17 Table 10-11 (joints with conforming transverse reinforcement): a plateau to
               plastic rotation a, then a linear descent that reaches the residual ratio c at plastic
               rotation b, where strength is lost. In IMK terms dp = a, dpc = (b - a) / (1 - c),
               du = theta_y + b, FmaxFy = 1, FresFy = c. Rows by joint class ("interior" = beams on
               both faces in the plane's direction, else "other"), interpolated on P/(Ag f'c) between
               0.1 and 0.4 and on V/Vn between 1.2 and 1.5.
  cyclic       No cyclic deterioration (every Lamda = JOINT_LAMBDA_SUPPRESSION); pinching break-point
               ratios JOINT_KAPPA_F and JOINT_KAPPA_D. Measured OpenSees 3.8 rule: reloading first heads
               to a break point at rotation (1 - kappaD) times the permanent rotation left after
               unloading from the previous peak (its zero-moment crossing), at kappaF times the moment
               the peak-oriented straight line would have there, then continues to the previous peak.

JOINT_DEFORMATION_SCOPE says whether the member hinges keep Haselton's bond-slip term
(IMK_Calibration.bond_slip_indicator): with "joint_shear_and_slip" the members drop it and the joint
spring is read as carrying slip as well as panel shear. Nothing here is experimentally calibrated
(JOINT_CALIBRATION_STATUS); the ASCE 41 rows were verified on 2026-09-27 against the PEER report that
produced them (see ASCE41_JOINT_ROWS).
"""
from __future__ import annotations

import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Design.SMRF_Capacity_Design import CONFINING_BEAM_WIDTH_RATIO, JOINT_SHEAR_GAMMA
from Design.SMRF_Joints import rectangular_joint_area
from Model.IMK_Calibration import column_gravity_axial
from Model.IMK_Materials import CyclicParameters, RotationalBackbone, define_rotational_imk

POISSON_RATIO = 0.2                     # concrete, for G = Ec / (2 (1 + nu)); ACI gives no value
TOPOLOGY = "orthogonal_scissors_centerline_v1"

# ASCE/SEI 41 modeling parameters for reinforced concrete beam-column joints, conforming transverse
# reinforcement (C: hoops spaced at <= hc/2 within the joint; SMRF joint hoops per ACI 318-19 18.8.3 are
# far closer):
#   (joint class, P/(Ag f'c), V/Vn) -> (a, b, c)
# a = plastic rotation at the end of the strength plateau, b = plastic rotation at loss of strength,
# c = residual strength ratio. P is the design axial force on the column above the joint over the gross
# joint area; V/Vn the design joint shear over the nominal strength. "Interior joints" are joints with
# beams framing into both faces in the direction considered (the gamma table lists interior joints with
# and without transverse beams); "other joints" are exterior and knee joints.
# Verified on 2026-09-27 against Table 6-9 of Elwood, Matamoros, Wallace, Lehman, Heintz, Mitchell,
# Moore, Marshall, Comartin and Moehle, "Update to ASCE/SEI 41 Concrete Provisions" (PEER, 2007), the
# table adopted as ASCE 41-06 Supplement 1 and retained as Table 10-11 in ASCE 41-13 and 41-17; all
# eight conforming rows match. tests/test_joint_springs.py pins them so a change is a deliberate edit.
ASCE41_JOINT_ROWS = {
    ("interior", 0.1, 1.2): (0.015, 0.030, 0.2),
    ("interior", 0.1, 1.5): (0.015, 0.030, 0.2),
    ("interior", 0.4, 1.2): (0.015, 0.025, 0.2),
    ("interior", 0.4, 1.5): (0.015, 0.020, 0.2),
    ("other", 0.1, 1.2): (0.010, 0.020, 0.2),
    ("other", 0.1, 1.5): (0.010, 0.015, 0.2),
    ("other", 0.4, 1.2): (0.010, 0.020, 0.2),
    ("other", 0.4, 1.5): (0.010, 0.015, 0.2),
}
# The nonconforming rows of the same table, for reference only (never selected: SMRF hoops conform).
ASCE41_JOINT_ROWS_NONCONFORMING = {
    ("interior", 0.1, 1.2): (0.005, 0.020, 0.2),
    ("interior", 0.1, 1.5): (0.005, 0.015, 0.2),
    ("interior", 0.4, 1.2): (0.005, 0.015, 0.2),
    ("interior", 0.4, 1.5): (0.005, 0.015, 0.2),
    ("other", 0.1, 1.2): (0.005, 0.010, 0.2),
    ("other", 0.1, 1.5): (0.005, 0.010, 0.2),
    ("other", 0.4, 1.2): (0.000, 0.0075, 0.0),
    ("other", 0.4, 1.5): (0.000, 0.0075, 0.0),
}
SOURCE_REFS = (
    "ACI 318-19 Table 18.8.4.3 (gamma), 15.4.2.4 (Aj), 19.2.2.1 (Ec)",
    "ASCE/SEI 41-17 Table 10-11 (a, b, c), conforming rows, verified 2026-09-27 against Elwood et al. (2007) "
    "PEER 'Update to ASCE/SEI 41 Concrete Provisions' Table 6-9",
    "Ibarra, Medina & Krawinkler (2005) kappa = 0.25 as a representative pinching level, not an RC joint fit",
    "Model/JOINT_PANEL.md scissors conjugacy M = tau Aj h_b, K = G Aj h_b",
)

# Spring direction per plane: the zeroLength is oriented on the global axes (x local, y local).
PLANES = {"x": {"direction": 5, "plane": "xz", "global_rotation_axis": "Ry"},
          "y": {"direction": 4, "plane": "yz", "global_rotation_axis": "Rx"}}

_JOINT_REGISTRY = {}


# ---- registry and tags ------------------------------------------------------------------------------
def reset_joint_registry():
    _JOINT_REGISTRY.clear()


def joint_registry():
    return dict(_JOINT_REGISTRY)


def joints_enabled():
    return getattr(sp, "JOINT_MODEL", "rigid_centerline") == "imk_pinching_scissors"


def beam_core_tag(joint_node):
    return sp.JOINT_BEAM_CORE_TAG_BASE + int(joint_node)


def joint_element_tag(joint_node):
    return sp.JOINT_ELEMENT_TAG_BASE + int(joint_node)


def joint_material_tag(joint_node, direction):
    return sp.JOINT_MATERIAL_TAG_BASE + 10 * int(joint_node) + int(direction)


def joint_beam_node(joint_node):
    """The node a beam frames into at this joint: the beam core when the joint spring exists, else the joint."""
    entry = _JOINT_REGISTRY.get(int(joint_node))
    return entry["beam_core"] if entry else int(joint_node)


# ---- classification ---------------------------------------------------------------------------------
def joint_kind(i, j):
    x_edge, y_edge = i in (0, sp.NUM_BAY_X), j in (0, sp.NUM_BAY_Y)
    return "corner" if x_edge and y_edge else "edge_y" if x_edge else "edge_x" if y_edge else "interior"


def framing(i, j):
    """(beams spanning X, beams spanning Y) that meet the joint on grid (i, j)."""
    beams_x = int(i > 0) + int(i < sp.NUM_BAY_X)
    beams_y = int(j > 0) + int(j < sp.NUM_BAY_Y)
    return beams_x, beams_y


def classify(k, i, j, axis):
    """gamma, Aj and Vn for the shear plane of `axis` at joint (k, i, j), from the record or the framing."""
    level = "roof" if k == sp.NUM_FLOOR else "floor"
    kind = joint_kind(i, j)
    beams_x, beams_y = framing(i, j)
    in_direction, transverse = (beams_x, beams_y) if axis == "x" else (beams_y, beams_x)
    # Column depth in the direction of the joint shear, and the width of the face the beams in that
    # direction frame into (the same convention as Design/Joint_Design_Tracking).
    depth, width = (sp.H_COL, sp.B_COL) if axis == "x" else (sp.B_COL, sp.H_COL)
    aj = rectangular_joint_area(column_depth_in=depth, column_width_in=width, beam_width_in=sp.B_BEAM,
                                beam_center_offset_in=0.0)
    fc = sp.FC_COL_KSI
    category_id = f"joint_shear/{level}/{kind}/{axis}"
    saved = (getattr(sp, "JOINT_SHEAR_CATEGORIES", None) or {}).get(category_id)
    numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)  # noqa: E731
    notes = []
    if saved and all(numeric(saved.get(key)) for key in ("gamma", "nominal_vn_kip", "aj_in2")):
        classification = saved.get("classification") or {}
        gamma, vn, aj_saved = float(saved["gamma"]), float(saved["nominal_vn_kip"]), float(saved["aj_in2"])
        confined = bool((classification.get("confinement") or {}).get("confined"))
        beam_state = (classification.get("beam") or {}).get("state", "other")
        column_state = (classification.get("column") or {}).get("state", "other")
        vj = saved.get("vj_kip")
        shear_ratio = float(vj) / vn if numeric(vj) and vn > 0 else 1.0
        if not math.isclose(aj, aj_saved, rel_tol=1e-9):
            notes.append(f"record Aj {aj_saved:g} in2 differs from the framing Aj {aj:g} in2; the record's value is used")
        aj = aj_saved
        source = "design_record_joint_shear_category"
    else:
        column_state = "continuous" if level == "floor" else "other"
        beam_state = "continuous" if in_direction >= 2 else "other"
        # The transverse (confining) beams frame into the faces perpendicular to the transverse
        # direction, whose width is the column dimension along `axis`, i.e. `depth`.
        confined = transverse >= 2 and sp.B_BEAM / depth >= CONFINING_BEAM_WIDTH_RATIO - 1e-12
        gamma = JOINT_SHEAR_GAMMA[(column_state, beam_state, confined)]
        vn = gamma * math.sqrt(fc * 1000.0) * aj / 1000.0
        shear_ratio = 1.0
        source = "framing_with_assumed_reinforcement_continuity"
        notes.append("no saved joint-shear category: continuity of bars through the joint assumed, V/Vn taken as 1.0")
    return {"category_id": category_id, "level": level, "kind": kind, "axis": axis,
            "beams_in_direction": in_direction, "beams_transverse": transverse,
            "column_state": column_state, "beam_state": beam_state, "confined": confined,
            "gamma": gamma, "aj_in2": aj, "fc_ksi": fc, "vn_kip": vn, "tau_n_ksi": vn / aj,
            "shear_ratio": shear_ratio, "source": source, "notes": notes}


def asce41_deformation(joint_class, axial_ratio, shear_ratio):
    """(a, b, c) from ASCE41_JOINT_ROWS with linear interpolation on axial and shear ratios."""
    if joint_class not in ("interior", "other"):
        raise ValueError(f"Unknown joint class {joint_class!r}")
    t_axial = min(1.0, max(0.0, (axial_ratio - 0.1) / 0.3))
    t_shear = min(1.0, max(0.0, (shear_ratio - 1.2) / 0.3))

    def lerp(low, high, t):
        return low + (high - low) * t

    corner = {(p, v): ASCE41_JOINT_ROWS[(joint_class, p, v)] for p in (0.1, 0.4) for v in (1.2, 1.5)}
    values = []
    for index in range(3):
        at_low_axial = lerp(corner[(0.1, 1.2)][index], corner[(0.1, 1.5)][index], t_shear)
        at_high_axial = lerp(corner[(0.4, 1.2)][index], corner[(0.4, 1.5)][index], t_shear)
        values.append(lerp(at_low_axial, at_high_axial, t_axial))
    row = {"class": joint_class, "axial_ratio": axial_ratio, "shear_ratio": shear_ratio,
           "t_axial": t_axial, "t_shear": t_shear,
           "table": "ASCE/SEI 41-17 Table 10-11, conforming rows (verified against Elwood et al. 2007 Table 6-9)"}
    return values[0], values[1], values[2], row


# ---- calibration ------------------------------------------------------------------------------------
def joint_calibration(k, i, j, axis):
    """Everything one joint spring is given, with its numbers exposed for the registry and tests."""
    c = classify(k, i, j, axis)
    hb = sp.H_BEAM
    fc = c["fc_ksi"]
    ec = 57.0 * math.sqrt(fc * 1000.0)
    g = ec / (2.0 * (1.0 + POISSON_RATIO))
    modifier = float(getattr(sp, "JOINT_STIFFNESS_MODIFIER", 1.0))
    ke = modifier * g * c["aj_in2"] * hb
    mn = c["vn_kip"] * hb
    theta_y = mn / ke
    axial = column_gravity_axial(k + 1, i, j) if k < sp.NUM_FLOOR else 0.0
    axial_ratio = axial / (sp.B_COL * sp.H_COL * fc)
    # ASCE 41's "interior joint" is in-plane: beams frame into both faces in the direction considered.
    # Transverse beams (confinement) enter the strength gamma, not the deformation row.
    joint_class = "interior" if c["beams_in_direction"] >= 2 else "other"
    a, b, residual, row = asce41_deformation(joint_class, axial_ratio, c["shear_ratio"])
    dpc = (b - a) / (1.0 - residual)
    du = theta_y + b
    suppression = float(sp.JOINT_LAMBDA_SUPPRESSION)
    backbone = RotationalBackbone(a, dpc, du, mn, 1.0, residual)
    cyclic = CyclicParameters(lamda_s=suppression, lamda_c=suppression, lamda_k=suppression,
                              c_s=1.0, c_c=1.0, c_k=1.0, d_pos=1.0, d_neg=1.0,
                              lamda_a=suppression, c_a=1.0,
                              kappa_f=float(sp.JOINT_KAPPA_F), kappa_d=float(sp.JOINT_KAPPA_D))
    return {**c, "joint_class": joint_class, "axial_kip": axial, "axial_ratio": axial_ratio,
            "ec_ksi": ec, "g_ksi": g, "stiffness_modifier": modifier, "beam_depth_in": hb,
            "ke_kip_in_per_rad": ke, "mn_kip_in": mn, "theta_y_rad": theta_y,
            "a_rad": a, "b_rad": b, "c_residual": residual, "dpc_rad": dpc, "du_rad": du,
            "deformation_row": row, "kappa_f": float(sp.JOINT_KAPPA_F), "kappa_d": float(sp.JOINT_KAPPA_D),
            "lambda_suppression": suppression, "backbone": backbone, "cyclic": cyclic}


def _provenance(cal, joint_node):
    return {"calibration_id": sp.JOINT_CALIBRATION_BASIS, "status": sp.JOINT_CALIBRATION_STATUS,
            "deformation_scope": sp.JOINT_DEFORMATION_SCOPE, "source_refs": list(SOURCE_REFS),
            "topology": TOPOLOGY, "plane": PLANES[cal["axis"]]["plane"], "axis": cal["axis"],
            "joint_node": int(joint_node), "category_id": cal["category_id"], "strength_source": cal["source"],
            "coordinate": "theta_beam_core - theta_column_core (panel shear strain)",
            "deformation_row": cal["deformation_row"], "notes": cal["notes"]}


# ---- installation -----------------------------------------------------------------------------------
def install_joint_springs(node_tag):
    """Add a beam core and a two-plane pinching spring at every elevated joint. Returns the registry."""
    if not joints_enabled():
        return {}
    if sp.ELEMENT_FORMULATION != "imk":
        raise ValueError("JOINT_MODEL 'imk_pinching_scissors' requires ELEMENT_FORMULATION 'imk': the joint cores "
                         "add equalDOF chains that only the penalty handler of the IMK frame carries")
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joint = int(node_tag(k, i, j))
                core = beam_core_tag(joint)
                ops.node(core, *ops.nodeCoord(joint))
                ops.equalDOF(joint, core, 1, 2, 3, 6)
                planes = {}
                for axis in ("x", "y"):
                    cal = joint_calibration(k, i, j, axis)
                    direction = PLANES[axis]["direction"]
                    mat_tag = joint_material_tag(joint, direction)
                    installed = define_rotational_imk("IMKPinching", mat_tag, cal["ke_kip_in_per_rad"],
                                                      cal["backbone"], cal["backbone"], cal["cyclic"],
                                                      provenance=_provenance(cal, joint))
                    planes[axis] = {key: value for key, value in cal.items() if key not in ("backbone", "cyclic")}
                    planes[axis].update(material_tag=mat_tag, direction=direction,
                                        parameter_sha256=installed["parameter_sha256"], installed=installed)
                ops.element("zeroLength", joint_element_tag(joint), joint, core,
                            "-mat", joint_material_tag(joint, 4), joint_material_tag(joint, 5), "-dir", 4, 5,
                            "-orient", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, "-doRayleigh", 0)
                _JOINT_REGISTRY[joint] = {"node": joint, "beam_core": core, "element": joint_element_tag(joint),
                                          "floor": k, "grid_i": i, "grid_j": j, "kind": joint_kind(i, j),
                                          "level": "roof" if k == sp.NUM_FLOOR else "floor",
                                          "topology": TOPOLOGY, "planes": planes}
    return joint_registry()
