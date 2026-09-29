"""Isolated beam-column-joint subassembly with member-face slip interfaces (diagnostic, 2026-09-27).

Codex's joint-slip review asked for an isolated diagnostic before anything touches the frame: the
force/rotation map written first (Model/JOINT_SLIP_SUBASSEMBLY.md), flexural rotation, slip rotation
and panel shear recorded separately, total displacement and external/internal work checked, and the
provisional slip calibration kept out of the production model. Codex's slip-calibration resolution
(the same evening) added: the interface stiffness must be a property of the physical calibration,
invariant under node-order/axis reversal, with complete branches and reference energies mapped
together (Model/IMK_Materials.define_mapped_rotational_imk); bar slip is owned per member end
(Model/Deformation_Ownership); both spring orientations, both in-plane bending axes, elastic
flexibility, equilibrium, convergence and complete exports are reported. This module is that diagnostic.

Topology (interior joint, one vertical plane, one story between inflection points):

    top node (lateral displacement imposed, column axial load)
      |  elastic upper column, H/2
    column hinge (frame law)         <- optional column-face slip interface
    J = column core  ==panel spring (installed joint law)==  B = beam core
    beam hinge (frame law) -- optional beam-face slip interface -- B, both faces
      elastic half beams, L/2, far ends on vertical rollers
    column hinge (frame law)
      |  elastic lower column, H/2
    base pin

The members, hinges, backbones and the panel spring are the frame's own (Model/IMK_Hinges,
Model/IMK_Calibration, Model/Joint_Springs) read from an installed design record. The slip interfaces
are IMKPinching springs calibrated here, provisionally, from bond (Sezen, Lodhi, Setzler and
Chowdhury 2008, section 2.2): sy = ey fy db / (8 ub), theta_slip,y = sy / (d - cy), with ub = 12 sqrt(f'c)
psi in the concrete surrounding the anchored bar, and the post-yield slip over the yielded length at
ub' = 6 sqrt(f'c). Their reference energies are suppressed (no bond deterioration data) and declared
verification-only. Nothing here is installed by Model/Build_Model.

usage: python Analysis/Joint_Slip_Subassembly.py --root <design root> --case case_0002 --output-root <dir>
       [--variant installed|slip_interfaces|legacy_no_slip|all] [--axis x|y|both] [--column-slip] [--floor 2]
       [--reverse-springs] [--suppress-member-deterioration] [--step 0.005] [--orientation-regression]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import openseespy.opensees as ops  # noqa: E402
import Structure_Parameters as sp  # noqa: E402

VARIANTS = ("installed", "slip_interfaces", "legacy_no_slip")
AXES = ("x", "y")
DRIFT_PROTOCOL_PCT = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
CYCLES_PER_LEVEL = 2
STEP_DRIFT_PCT = 0.005
SOLVER_TOLERANCE = 1e-12                         # NormDispIncr; 1e-8 left the slip fixture's orientation check solver-limited (3.6e-7)
SOLVER_ITERATIONS = 300
ORIENTATION_TOLERANCE = 1e-9                     # relative discrepancy between equivalent spring orientations
PROVISIONAL_STEEL_STRAIN_AT_PROBABLE = 0.02      # A706 strain taken at 1.25 fy; declared, not measured
PROVISIONAL_POST_CAP_MULTIPLE = 10.0             # slip law post-capping slope: no data, kept far away
PROVISIONAL_INTERFACE_KE_RULE = "flexibility_mean_of_face_secants"   # one Ke for both directions, invariant
SLIP_CALIBRATION_ID = "sezen_setzler_2008_bond_slip_provisional_v1_invariant_ke"
SOURCES = {
    "bond_slip": "Sezen, Lodhi, Setzler and Chowdhury (2008), 14WCEE paper S15-019, section 2.2; Sezen and Setzler (2008), ACI SJ 105(3)",
    "haselton": "Haselton, Liel, Taylor Lange and Deierlein (2008), PEER 2007/03, eq. 3.10 and sections 2.1.2.2, 3.5.4",
    "mapping": "Model/IMK_Materials.define_mapped_rotational_imk: complete branches, directional D and reference energies mapped together",
    "ownership": "Model/Deformation_Ownership: one owner per mechanism per member end; a declared joint scope is not an owner",
    "scissors": "Alath and Kunnath (1995); Celik and Ellingwood (2008): scissors joint element conjugacy",
}

# Node and element tags of the subassembly (all far from the frame's bases).
N_BASE, N_J, N_B, N_TOP = 1, 10, 11, 20
N_BEAM_FAR = {"L": 30, "R": 31}
N_BEAM_SLIP = {"L": 40, "R": 41}       # between the beam core and the beam hinge node (slip interface node)
N_BEAM_HINGE = {"L": 50, "R": 51}      # member-end node of the beam hinge
N_COL_SLIP = {"T": 60, "B": 61}
N_COL_HINGE = {"T": 70, "B": 71}
E_PANEL = 100
E_BEAM_SLIP = {"L": 110, "R": 111}
E_BEAM_HINGE = {"L": 120, "R": 121}
E_COL_SLIP = {"T": 130, "B": 131}
E_COL_HINGE = {"T": 140, "B": 141}
E_BEAM = {"L": 150, "R": 151}
E_COL = {"T": 160, "B": 161}
MAT_BASE = 500
TRANSF_BEAM, TRANSF_COL = 1, 2
# The frame's sign rule: hogging is positive spring deformation where the beam leaves the joint
# toward +x (end i, the right beam here) and sagging is positive where it leaves toward -x (end j).
BEAM_END = {"L": "j", "R": "i"}
COLUMN_END = {"T": "j", "B": "i"}


# ---------------------------------------------------------------------------------------------------
# provisional slip calibration (diagnostic only)
# ---------------------------------------------------------------------------------------------------
def cracked_neutral_axis(b, d, d_prime, as_tension, as_compression, ec_ksi, es_ksi=sp.ES_KSI):
    """Elastic cracked-section neutral-axis depth kd (transformed section, compression steel counted)."""
    n = es_ksi / ec_ksi
    rho, rho_p = as_tension / (b * d), as_compression / (b * d)
    a = n * (rho + rho_p)
    k = math.sqrt(a * a + 2.0 * n * (rho + rho_p * d_prime / d)) - a
    return k * d


def slip_calibration(bar_size, fy_ksi, fc_anchorage_ksi, d_in, c_y_in, bars, ec_ksi, es_ksi=sp.ES_KSI,
                     strain_at_probable=PROVISIONAL_STEEL_STRAIN_AT_PROBABLE, ub_factor=12.0, ub_post_factor=6.0):
    """Bond-slip rotation at first bar yield and at the probable stress, Sezen et al. (2008) section 2.2.

    Elastic bond ub = ub_factor sqrt(f'c) (psi) over the development length ld = fy db / (4 ub); the
    strain profile is triangular, so the slip at yield is sy = ey fy db / (8 ub). Beyond yield a
    length ld' = (fs - fy) db / (4 ub') carries the average strain (es + ey) / 2 at ub' = ub_post_factor
    sqrt(f'c), adding (es + ey) / 2 * ld'. Rotation = slip / (d - c), tension-side slip about the
    neutral axis (the review's applicability caveats apply: not an interface law for compression-bar
    slip, through-bars or anchorage failure; first bar yield is not the nominal My, Codex resolution
    section 1).
    """
    db = sp.rebar_diameter(bar_size)
    fc_psi = fc_anchorage_ksi * 1000.0
    ub = ub_factor * math.sqrt(fc_psi) / 1000.0            # ksi
    ub_post = ub_post_factor * math.sqrt(fc_psi) / 1000.0
    ey = fy_ksi / es_ksi
    ld = fy_ksi * db / (4.0 * ub)
    s_y = ey * fy_ksi * db / (8.0 * ub)
    fs = 1.25 * fy_ksi
    ld_post = (fs - fy_ksi) * db / (4.0 * ub_post)
    s_u = s_y + 0.5 * (strain_at_probable + ey) * ld_post
    lever = d_in - c_y_in
    return {"bar_size": bar_size, "db_in": db, "fy_ksi": fy_ksi, "fc_anchorage_ksi": fc_anchorage_ksi,
            "ub_ksi": ub, "ub_post_ksi": ub_post, "ld_in": ld, "ld_post_in": ld_post, "ey": ey,
            "slip_y_in": s_y, "slip_u_in": s_u, "d_in": d_in, "c_y_in": c_y_in, "lever_in": lever,
            "theta_slip_y": s_y / lever, "theta_slip_u": s_u / lever,
            "strain_at_probable": strain_at_probable, "bars": bars,
            "source": SOURCES["bond_slip"], "status": "provisional; diagnostic only; not installed in the frame"}


def interface_stiffness(cal_hog, cal_sag, rule=PROVISIONAL_INTERFACE_KE_RULE):
    """One elastic stiffness for the interface, a property of the physical calibration and invariant
    under which direction the element calls positive (Codex resolution, section 2).

    The bond law gives a secant to first bar yield per face, My / theta_slip,y, which differ because
    the two faces have different lever arms; an IMK material has one Ke, so the interface takes the
    flexibility mean of the two secants and records how far each branch's own yield rotation My / Ke
    then sits from its target. A branch whose capping target falls inside that yield rotation is
    refused, not hidden.
    """
    if rule != PROVISIONAL_INTERFACE_KE_RULE:
        raise ValueError(f"unknown interface stiffness rule {rule!r}")
    secants = {"hogging": cal_hog["my"] / cal_hog["theta_slip_y"], "sagging": cal_sag["my"] / cal_sag["theta_slip_y"]}
    ke = 2.0 / (1.0 / secants["hogging"] + 1.0 / secants["sagging"])
    mismatch = {sign: (cal["my"] / ke - cal["theta_slip_y"]) / cal["theta_slip_y"]
                for sign, cal in (("hogging", cal_hog), ("sagging", cal_sag))}
    return ke, {"rule": rule, "face_secants_kip_in_per_rad": secants, "ke_kip_in_per_rad": ke,
                "branch_yield_rotation_mismatch": mismatch, "invariant_under_reversal": True,
                "status": "provisional; the rule is declared, not fitted"}


def _fix_out_of_plane(node):
    """In-plane (x-z) behaviour: uy, rx, rz are fixed on every node."""
    ops.fix(node, 0, 1, 0, 1, 0, 1)


def _rotational_spring(ele_tag, node_core, node_member, mat_tag, reverse_nodes):
    """A zero-length rotational spring about global Y between two coincident nodes; the in-plane
    translations are tied. ``reverse_nodes`` swaps the element's node order (its strain sign) for the
    orientation regression; the material must then be installed with its branches mapped accordingly."""
    ops.equalDOF(node_core, node_member, 1, 3)
    n_i, n_j = (node_member, node_core) if reverse_nodes else (node_core, node_member)
    ops.element("zeroLength", ele_tag, n_i, n_j, "-mat", mat_tag, "-dir", 5, "-orient", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _member_hinge_material(mat_tag, member_type, props, backbone, length, positive_my, negative_my, end_label):
    from Model.IMK_Hinges import _define_imk_peak_material, imk_hinge_stiffness
    ke = imk_hinge_stiffness(member_type, "rot_y", length, props=props)
    context = {"physical_member_tag": 0, "end": end_label, "member_type": member_type, "spring_local_direction": 5,
               "global_rotation_axis": (0, 1, 0), "zero_length_orientation": (1, 0, 0, 0, 1, 0),
               "retained_joint_node": 0, "member_hinge_node": 0,
               "rotation_definition": "hinge-node rotation minus retained-node rotation about global Y",
               "tied_global_dofs": (1, 2, 3, 4, 6)}
    installed = _define_imk_peak_material(mat_tag, ke, positive_my, backbone, negative_my, spring_context=context)
    return ke, installed


def _slip_material(mat_tag, cal_hog, cal_sag, kappa_f, kappa_d, suppression, *, reverse):
    """IMKPinching slip interface with the invariant Ke, both physical branches, directional D and
    (suppressed) reference energies mapped together; ``reverse`` says the element's positive strain
    is physically sagging (a beam leaving the joint toward -x, or a reversed node order)."""
    from Model.IMK_Materials import CyclicParameters, RotationalBackbone, define_mapped_rotational_imk
    ke, rule = interface_stiffness(cal_hog, cal_sag)
    branches, targets = {}, {}
    for sign, cal in (("hogging", cal_hog), ("sagging", cal_sag)):
        theta_y = cal["my"] / ke
        dp = cal["theta_slip_u"] - theta_y
        if dp <= 0.0:
            raise ValueError(f"incompatible slip target for {sign}: capping rotation {cal['theta_slip_u']:.5f} rad is not "
                             f"beyond the branch's own yield rotation My / Ke = {theta_y:.5f} rad")
        branches[sign] = RotationalBackbone(dp, PROVISIONAL_POST_CAP_MULTIPLE * dp,
                                            theta_y + (1.0 + PROVISIONAL_POST_CAP_MULTIPLE) * dp, cal["my"], 1.25, 0.2)
        targets[sign] = {"theta_y_branch": theta_y, "theta_slip_y_target": cal["theta_slip_y"],
                         "theta_cap_target": cal["theta_slip_u"], "dp": dp}
    cyclic = CyclicParameters(lamda_s=1.0, lamda_c=1.0, lamda_k=1.0, c_s=1.0, c_c=1.0, c_k=1.0,
                              d_pos=1.0, d_neg=1.0, lamda_a=1.0, c_a=1.0, kappa_f=kappa_f, kappa_d=kappa_d)
    energies = {mode: suppression * max(cal_hog["my"], cal_sag["my"]) for mode in ("S", "C", "A", "K")}
    profile = {"calibration_id": SLIP_CALIBRATION_ID, "status": "verification_only", "material_type": "IMKPinching",
               "units": "kip-in*rad", "deformation_scope": "member_face_bar_slip_rotation",
               "derivation": "cyclic bond deterioration suppressed: no data; every mode energy = suppression x max face My",
               "applicability_basis": "diagnostic subassembly only; provisional bond law; not a fit and not for structural runs",
               "source_refs": [SOURCES["bond_slip"]], "specimen_ids": ["none: provisional bond law, no specimen"],
               "energies_kip_in_rad": energies}
    installed = define_mapped_rotational_imk("IMKPinching", mat_tag, ke, branches["hogging"], branches["sagging"], cyclic,
                                             calibration=profile, reverse=reverse, physical_directions=("hogging", "sagging"),
                                             verification_only=True,
                                             provenance={"calibration_id": SLIP_CALIBRATION_ID, "status": "provisional_diagnostic_only",
                                                         "deformation_scope": "member_face_bar_slip_rotation",
                                                         "stiffness_rule": rule})
    return ke, installed, {"hogging": branches["hogging"].arguments(), "sagging": branches["sagging"].arguments()}, rule, targets


@contextlib.contextmanager
def _member_deterioration_suppressed(active):
    """Give the member hinges a suppressed (effectively infinite) energy capacity in-process, so the
    legacy E_ref = Lambda x Fy_positive anchor cannot tell one spring orientation from the other."""
    if not active:
        yield
        return
    saved = {k: getattr(sp, k) for k in ("IMK_DETERIORATION_MODE", "IMK_LAMBDA_S", "IMK_LAMBDA_C", "IMK_LAMBDA_K", "IMK_LAMBDA_A")}
    try:
        sp.IMK_DETERIORATION_MODE = "direct"
        for k in ("IMK_LAMBDA_S", "IMK_LAMBDA_C", "IMK_LAMBDA_K", "IMK_LAMBDA_A"):
            setattr(sp, k, float(sp.JOINT_LAMBDA_SUPPRESSION))
        yield
    finally:
        for k, v in saved.items():
            setattr(sp, k, v)


def build(record, case, variant, floor, column_slip, story_h, *, axis="x", reverse_springs=False,
          suppress_member_deterioration=False):
    """Build one variant; returns the description of every component for the recorders and the report."""
    from Model.IMK_Hinges import _member_properties
    from Model.IMK_Calibration import backbone_for_member, column_gravity_axial, haselton_theta_p, axial_load_ratio
    from Model.Joint_Springs import joint_calibration
    from Model.IMK_Materials import define_rotational_imk
    from Model import Deformation_Ownership as own
    from Design.Design_Driver import _beam_strength_families

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if axis not in AXES:
        raise ValueError(f"axis must be x or y, got {axis!r}")
    slip_interfaces = variant == "slip_interfaces"
    member_scope = own.LEGACY_UNOWNED_OVERRIDE if variant == "legacy_no_slip" else None
    with _member_deterioration_suppressed(suppress_member_deterioration):
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        own.begin_domain(f"Joint_Slip_Subassembly.build[{variant}]")     # registrations live with this domain only
        H = story_h
        L = sp.BAY_X if axis == "x" else sp.BAY_Y
        grid_i, grid_j = (1, max(1, sp.NUM_BAY_Y // 2)) if axis == "x" else (max(1, sp.NUM_BAY_X // 2), 1)
        axial = column_gravity_axial(floor, grid_i, grid_j)
        member_type = f"beam_{axis}"
        family = (axis, "interior")

        # nodes
        ops.node(N_BASE, 0.0, 0.0, -H / 2.0)
        for n in (N_J, N_B, N_COL_HINGE["T"], N_COL_HINGE["B"], N_BEAM_HINGE["L"], N_BEAM_HINGE["R"]):
            ops.node(n, 0.0, 0.0, 0.0)
        ops.node(N_TOP, 0.0, 0.0, H / 2.0)
        ops.node(N_BEAM_FAR["L"], -L / 2.0, 0.0, 0.0)
        ops.node(N_BEAM_FAR["R"], L / 2.0, 0.0, 0.0)
        for n in (N_J, N_B, N_TOP, *N_BEAM_HINGE.values(), *N_COL_HINGE.values()):
            _fix_out_of_plane(n)
        ops.fix(N_BASE, 1, 1, 1, 1, 0, 1)                       # pin (plus the out-of-plane fixities)
        for n in N_BEAM_FAR.values():
            ops.fix(n, 0, 1, 1, 1, 0, 1)                        # vertical roller, free rotation about Y
        ops.geomTransf("Linear", TRANSF_BEAM, 0.0, 0.0, 1.0)
        ops.geomTransf("Linear", TRANSF_COL, 0.0, 1.0, 0.0)

        # panel spring between the cores (the installed joint law, panel shear only)
        ops.equalDOF(N_J, N_B, 1, 3)
        cal = joint_calibration(floor, grid_i, grid_j, axis)
        panel_mat = MAT_BASE + 1
        panel_installed = define_rotational_imk("IMKPinching", panel_mat, cal["ke_kip_in_per_rad"], cal["backbone"], cal["backbone"],
                                                cal["cyclic"], provenance={"calibration_id": "installed_joint_law", "status": "as_installed",
                                                                           "deformation_scope": "panel shear strain (scissors)"})
        n_i, n_j = (N_B, N_J) if reverse_springs else (N_J, N_B)
        ops.element("zeroLength", E_PANEL, n_i, n_j, "-mat", panel_mat, "-dir", 5, "-orient", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        components = {"panel": {"element": E_PANEL, "kind": "panel", "material": panel_installed, "reversed": reverse_springs,
                                "ke_kip_in_per_rad": cal["ke_kip_in_per_rad"],
                                "calibration": {k: cal[k] for k in ("mn_kip_in", "ke_kip_in_per_rad", "theta_y_rad", "a_rad", "b_rad",
                                                                    "c_residual", "kappa_f", "kappa_d", "gamma", "vn_kip", "aj_in2", "joint_class")}}}

        # column: upper half J -> TOP, lower half J -> BASE, hinge at the joint ends (frame column law)
        col_props = _member_properties("column", axial_kip=axial)
        col_inertia = col_props["iy"] if axis == "x" else col_props["iz"]
        col_my = col_props["my"] if axis == "x" else col_props["mz"]
        ownership = {}
        for end, far, e_col, e_hinge, n_hinge, e_slip, n_slip in (("T", N_TOP, E_COL["T"], E_COL_HINGE["T"], N_COL_HINGE["T"], E_COL_SLIP["T"], N_COL_SLIP["T"]),
                                                                 ("B", N_BASE, E_COL["B"], E_COL_HINGE["B"], N_COL_HINGE["B"], E_COL_SLIP["B"], N_COL_SLIP["B"])):
            descriptor = own.member_end(e_col, "column", COLUMN_END[end], location="diagnostic_fixture", floor_index=floor)
            attach = N_J
            if column_slip:
                mat_s = MAT_BASE + 30 + (0 if end == "T" else 1)
                col_d = sp.H_COL - sp.longitudinal_cover_in("column")
                c_y = cracked_neutral_axis(sp.B_COL, col_d, sp.longitudinal_cover_in("column"), sp.COL_TOP_BARS * sp.COL_BAR_AREA,
                                           sp.COL_BOT_BARS * sp.COL_BAR_AREA, col_props["e"])
                cs = {**slip_calibration(sp.COL_BAR_SIZE, sp.FY_KSI, sp.FC_COL_KSI, col_d, c_y, sp.COL_TOP_BARS, col_props["e"]), "my": col_my}
                own.register_slip_interface(e_col, COLUMN_END[end], calibration_id=SLIP_CALIBRATION_ID, status="provisional_diagnostic_only",
                                            evidence="column bars are continuous through the joint; interface unverified", element_tag=e_slip)
                ke_s, inst_s, bb, rule, targets = _slip_material(mat_s, cs, cs, cal["kappa_f"], cal["kappa_d"], float(sp.JOINT_LAMBDA_SUPPRESSION),
                                                                 reverse=reverse_springs)
                ops.node(n_slip, 0.0, 0.0, 0.0); _fix_out_of_plane(n_slip)     # the interface node exists only when its interface does
                _rotational_spring(e_slip, N_J, n_slip, mat_s, reverse_springs)
                components[f"column_slip_{end}"] = {"element": e_slip, "kind": "slip", "material": inst_s, "calibration": cs, "backbone": bb,
                                                    "ke_kip_in_per_rad": ke_s, "stiffness_rule": rule, "branch_targets": targets,
                                                    "reversed": reverse_springs,
                                                    "note": "column bars are continuous through the joint; this interface is unverified"}
                attach = n_slip
            backbone = backbone_for_member("column", axial_kip=axial, end=descriptor, member_scope=member_scope)
            ownership[f"column_{end}"] = own.ownership(descriptor, joint_spring_present=True, member_scope=member_scope)
            mat = MAT_BASE + 10 + (0 if end == "T" else 1)
            positive, negative = col_my, col_my
            ke, installed = _member_hinge_material(mat, "column", col_props, backbone, H, positive, negative, f"column_{end}")
            _rotational_spring(e_hinge, attach, n_hinge, mat, reverse_springs)
            ops.element("elasticBeamColumn", e_col, n_hinge, far, col_props["area"], col_props["e"], col_props["g"],
                        col_props["j"], col_inertia, col_inertia, TRANSF_COL)
            components[f"column_hinge_{end}"] = {"element": e_hinge, "kind": "hinge", "material": installed, "ke_kip_in_per_rad": ke,
                                                 "my_kip_in": col_my, "backbone": backbone, "axial_kip": axial, "reversed": reverse_springs}

        # beams: left far -> core (joint end j: positive spring deformation = sagging), right core -> far (joint end i: positive = hogging)
        beam_props = _member_properties(member_type, family=family)
        rows = sp.beam_bar_layers()[axis]
        beam_d = {"top": sp.H_BEAM - rows["top"]["centroid_in"], "bottom": sp.H_BEAM - rows["bottom"]["centroid_in"]}
        as_top, as_bot = sp.BEAM_TOP_BARS * sp.BEAM_BAR_AREA, sp.BEAM_BOT_BARS * sp.BEAM_BAR_AREA
        c_hog = cracked_neutral_axis(sp.B_BEAM, beam_d["top"], rows["bottom"]["centroid_in"], as_top, as_bot, beam_props["e"])
        c_sag = cracked_neutral_axis(sp.B_BEAM, beam_d["bottom"], rows["top"]["centroid_in"], as_bot, as_top, beam_props["e"])
        families = _beam_strength_families()
        hogging, sagging = families[f"{axis}_interior"]["mn_negative_kip_in"], families[f"{axis}_interior"]["mn_positive_kip_in"]
        slip_hog = {**slip_calibration(sp.BEAM_BAR_SIZE, sp.FY_KSI, sp.FC_COL_KSI, beam_d["top"], c_hog, sp.BEAM_TOP_BARS, beam_props["e"]), "my": hogging, "sign": "hogging"}
        slip_sag = {**slip_calibration(sp.BEAM_BAR_SIZE, sp.FY_KSI, sp.FC_COL_KSI, beam_d["bottom"], c_sag, sp.BEAM_BOT_BARS, beam_props["e"]), "my": sagging, "sign": "sagging"}
        for side, far, e_beam, e_hinge, n_hinge, e_slip, n_slip in (("L", N_BEAM_FAR["L"], E_BEAM["L"], E_BEAM_HINGE["L"], N_BEAM_HINGE["L"], E_BEAM_SLIP["L"], N_BEAM_SLIP["L"]),
                                                                    ("R", N_BEAM_FAR["R"], E_BEAM["R"], E_BEAM_HINGE["R"], N_BEAM_HINGE["R"], E_BEAM_SLIP["R"], N_BEAM_SLIP["R"])):
            descriptor = own.member_end(e_beam, member_type, BEAM_END[side], location="diagnostic_fixture", floor_index=floor)
            physically_sagging_positive = side == "L"
            element_positive_is_sagging = physically_sagging_positive != reverse_springs
            attach = N_B
            if slip_interfaces:
                mat_s = MAT_BASE + 40 + (0 if side == "L" else 1)
                own.register_slip_interface(e_beam, BEAM_END[side], calibration_id=SLIP_CALIBRATION_ID, status="provisional_diagnostic_only",
                                            evidence=SOURCES["bond_slip"], element_tag=e_slip)
                ke_s, inst_s, bb, rule, targets = _slip_material(mat_s, slip_hog, slip_sag, cal["kappa_f"], cal["kappa_d"],
                                                                 float(sp.JOINT_LAMBDA_SUPPRESSION), reverse=element_positive_is_sagging)
                ops.node(n_slip, 0.0, 0.0, 0.0); _fix_out_of_plane(n_slip)
                _rotational_spring(e_slip, N_B, n_slip, mat_s, reverse_springs)
                components[f"beam_slip_{side}"] = {"element": e_slip, "kind": "slip", "material": inst_s, "ke_kip_in_per_rad": ke_s,
                                                   "calibration": {"hogging": slip_hog, "sagging": slip_sag}, "backbone": bb,
                                                   "stiffness_rule": rule, "branch_targets": targets, "reversed": reverse_springs,
                                                   "element_positive_is": "sagging" if element_positive_is_sagging else "hogging"}
                attach = n_slip
            backbone = backbone_for_member(member_type, end=descriptor, member_scope=member_scope)
            ownership[f"beam_{side}"] = own.ownership(descriptor, joint_spring_present=True, member_scope=member_scope)
            positive, negative = (sagging, hogging) if element_positive_is_sagging else (hogging, sagging)
            mat = MAT_BASE + 20 + (0 if side == "L" else 1)
            ke, installed = _member_hinge_material(mat, member_type, beam_props, backbone, L, positive, negative, f"beam_{side}")
            _rotational_spring(e_hinge, attach, n_hinge, mat, reverse_springs)
            b_i, b_j = (far, n_hinge) if side == "L" else (n_hinge, far)
            ops.element("elasticBeamColumn", e_beam, b_i, b_j, beam_props["area"], beam_props["e"], beam_props["g"],
                        beam_props["j"], beam_props["iy"], beam_props["iz"], TRANSF_BEAM)
            components[f"beam_hinge_{side}"] = {"element": e_hinge, "kind": "hinge", "material": installed, "ke_kip_in_per_rad": ke,
                                                "my_hogging_kip_in": hogging, "my_sagging_kip_in": sagging,
                                                "element_positive_is": "sagging" if element_positive_is_sagging else "hogging",
                                                "backbone": backbone, "reversed": reverse_springs}

        # gravity axial load on the column, then the lateral history
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(N_TOP, 0.0, 0.0, -axial, 0.0, 0.0, 0.0)
        ops.constraints("Penalty", sp.PENALTY_ALPHA_SP, sp.PENALTY_ALPHA_MP)   # the frame's handler: chained equalDOF ties
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-8, 50, 0)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 0.1)
        ops.analysis("Static")
        if ops.analyze(10) != 0:
            raise RuntimeError("gravity step failed")
        ops.loadConst("-time", 0.0)

        theta_p_with = haselton_theta_p(member_type, 0.0, None)      # a_sl = 1, the member default
        beam_backbone = components["beam_hinge_L"]["backbone"]
        meta = {"variant": variant, "axis": axis, "plane": f"{axis}-z", "floor": floor, "story_h_in": H, "bay_in": L,
                "reverse_springs": reverse_springs, "member_deterioration_suppressed": suppress_member_deterioration,
                "column_axial_kip": axial, "column_axial_ratio": axial_load_ratio(axial, "column"),
                "beam": {"member_type": member_type, "b_in": sp.B_BEAM, "h_in": sp.H_BEAM, "fc_ksi": sp.FC_BEAM_KSI, "bar_size": sp.BEAM_BAR_SIZE,
                         "top_bars": sp.BEAM_TOP_BARS, "bot_bars": sp.BEAM_BOT_BARS, "d_in": beam_d, "rows": rows, "family": f"{axis}_interior",
                         "my_hogging_kip_in": hogging, "my_sagging_kip_in": sagging, "iy_in4": beam_props["iy"], "e_ksi": beam_props["e"],
                         "stiffness_modifier": beam_props["stiffness_modifier"], "theta_y_member": beam_props["theta_y"]},
                "column": {"b_in": sp.B_COL, "h_in": sp.H_COL, "fc_ksi": sp.FC_COL_KSI, "bar_size": sp.COL_BAR_SIZE, "my_kip_in": col_my,
                           "i_in4": col_inertia, "e_ksi": col_props["e"], "stiffness_modifier": col_props["stiffness_modifier"]},
                "haselton": {"beam_theta_p_installed": beam_backbone["theta_p"], "beam_theta_pc": beam_backbone["theta_pc"],
                             "beam_theta_p_with_slip": theta_p_with, "beam_theta_p_without_slip": theta_p_with / 1.55,
                             "slip_share_difference_0_55": theta_p_with - theta_p_with / 1.55,
                             "bond_slip_indicator_beams": beam_backbone["bond_slip_indicator"],
                             "bond_slip_indicator_columns": components["column_hinge_T"]["backbone"]["bond_slip_indicator"]},
                "slip_calibration": {"hogging": slip_hog, "sagging": slip_sag, "column_slip_enabled": column_slip,
                                     "interface_stiffness_rule": PROVISIONAL_INTERFACE_KE_RULE},
                "deformation_ownership": ownership,
                "slip_domain": own.slip_domain(),
                "components": {name: {k: v for k, v in c.items() if k != "material"} for name, c in components.items()}}
        meta["slip_interfaces_installed"] = own.validate_installed()     # every registration names an element of this domain
        return components, meta


# ---------------------------------------------------------------------------------------------------
# loading, recording and checks
# ---------------------------------------------------------------------------------------------------
def protocol(story_h):
    targets = []
    for pct in DRIFT_PROTOCOL_PCT:
        for _ in range(CYCLES_PER_LEVEL):
            targets += [pct, -pct]
    targets.append(0.0)
    return [t / 100.0 * story_h for t in targets]


def run_protocol(components, meta, step_pct=STEP_DRIFT_PCT, tolerance=SOLVER_TOLERANCE, iterations=SOLVER_ITERATIONS, algorithm="Newton"):
    H = meta["story_h_in"]
    step = step_pct / 100.0 * H
    ops.wipeAnalysis()
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(N_TOP, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ops.constraints("Penalty", sp.PENALTY_ALPHA_SP, sp.PENALTY_ALPHA_MP)   # the frame's handler: chained equalDOF ties
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", tolerance, iterations, 0)
    ops.algorithm(algorithm)
    ops.analysis("Static")
    names = list(components)
    sign = {n: (-1.0 if components[n].get("reversed") else 1.0) for n in names}   # physical = sign x element pair
    hist = {"u_top": [], "u_J": [], "u_z_top": [], "u_z_J": [], "P": [], "theta_J": [], "theta_B": [], "target_index": [],
            "moment": {n: [] for n in names}, "rotation": {n: [] for n in names},
            "beam_end_moment": {"L": [], "R": []}, "column_end_moment": {"T": [], "B": []}, "column_axial": []}
    current, failures, retries = 0.0, 0, 0
    for k, target in enumerate(protocol(H)):
        n_steps = max(1, int(round(abs(target - current) / step)))
        du = (target - current) / n_steps
        ops.integrator("DisplacementControl", N_TOP, 1, du)
        for _ in range(n_steps):
            ok = ops.analyze(1)
            if ok != 0:
                retries += 1
                ops.algorithm("KrylovNewton" if algorithm != "KrylovNewton" else "Newton")
                ok = ops.analyze(1)
                ops.algorithm(algorithm)
                if ok != 0:
                    failures += 1
                    ops.integrator("DisplacementControl", N_TOP, 1, du / 10.0)
                    for _ in range(10):
                        if ops.analyze(1) != 0:
                            raise RuntimeError(f"subassembly step failed at u = {current:.4f} in")
                    ops.integrator("DisplacementControl", N_TOP, 1, du)
            current = ops.nodeDisp(N_TOP, 1)
            ops.reactions()
            hist["u_top"].append(current)
            hist["u_J"].append(ops.nodeDisp(N_J, 1))
            hist["u_z_top"].append(ops.nodeDisp(N_TOP, 3))
            hist["u_z_J"].append(ops.nodeDisp(N_J, 3))
            hist["P"].append(-ops.nodeReaction(N_BASE, 1))
            hist["theta_J"].append(ops.nodeDisp(N_J, 5))
            hist["theta_B"].append(ops.nodeDisp(N_B, 5))
            hist["target_index"].append(k)
            for n in names:
                e = components[n]["element"]
                # the spring's own conjugate pair: material 1 stress = moment, strain = rotation, in the
                # physical sign (core-side node first) whatever the element's node order
                hist["moment"][n].append(sign[n] * ops.eleResponse(e, "material", "1", "stress")[0])
                hist["rotation"][n].append(sign[n] * ops.eleResponse(e, "material", "1", "strain")[0])
            for side in ("L", "R"):
                f = ops.eleForce(E_BEAM[side])
                hist["beam_end_moment"][side].append(f[4] if side == "R" else f[10])     # moment about Y at the joint end
            for end in ("T", "B"):
                f = ops.eleForce(E_COL[end])
                hist["column_end_moment"][end].append(f[4])
            hist["column_axial"].append(ops.eleForce(E_COL["T"])[2])
    for key in ("u_top", "u_J", "u_z_top", "u_z_J", "P", "theta_J", "theta_B", "target_index", "column_axial"):
        hist[key] = np.array(hist[key])
    for group in ("moment", "rotation", "beam_end_moment", "column_end_moment"):
        hist[group] = {k: np.array(v) for k, v in hist[group].items()}
    hist["solver_failures"] = failures
    hist["solver_retries"] = retries
    hist["step_pct"] = step_pct
    hist["solver"] = {"test": "NormDispIncr", "tolerance": tolerance, "iterations": iterations, "algorithm": algorithm}
    return hist


def _work(force, disp):
    return float(np.sum(0.5 * (force[1:] + force[:-1]) * np.diff(disp))) if len(disp) > 1 else 0.0


def _cumulative_work(force, disp):
    out = np.zeros_like(disp)
    out[1:] = np.cumsum(0.5 * (force[1:] + force[:-1]) * np.diff(disp))
    return out


def elastic_member_energy(hist, meta):
    """Stored bending energy of the four elastic segments from their end moments (Euler-Bernoulli, the far end at
    zero moment): U = M^2 L' / (6 EI) for a linear moment diagram from M to 0 over L'."""
    H, L = meta["story_h_in"], meta["bay_in"]
    ei_c = meta["column"]["e_ksi"] * meta["column"]["i_in4"]
    ei_b = meta["beam"]["e_ksi"] * meta["beam"]["iy_in4"]
    u = np.zeros_like(hist["u_top"])
    for end in ("T", "B"):
        u += hist["column_end_moment"][end] ** 2 * (H / 2.0) / (6.0 * ei_c)
    for side in ("L", "R"):
        u += hist["beam_end_moment"][side] ** 2 * (L / 2.0) / (6.0 * ei_b)
    return u


def evaluate(hist, meta, components):
    H, L = meta["story_h_in"], meta["bay_in"]
    names = list(components)
    drift = hist["u_top"] / H
    work_ext = _cumulative_work(hist["P"], hist["u_top"]) + _cumulative_work(-meta["column_axial_kip"] * np.ones_like(hist["u_z_top"]), hist["u_z_top"])
    work_springs = {n: _cumulative_work(hist["moment"][n], hist["rotation"][n]) for n in names}
    # M^2 / (2 Ke) is the stored energy of a LINEAR ELASTIC spring on its installed stiffness. The signed
    # work less its change is exact dissipation only for a component that never left its elastic
    # branch; for a deteriorating IMK state (path-dependent unloading, internal variables) it is an
    # elastic-reference ESTIMATE and is labelled so (Codex next-calibration package, 2026-09-27). The
    # signed work is the reliable quantity; the global balance uses it.
    stored = {n: hist["moment"][n] ** 2 / (2.0 * components[n]["ke_kip_in_per_rad"]) for n in names}
    stayed_elastic = {n: bool(np.max(np.abs(hist["rotation"][n])) <= _elastic_rotation_limit(components[n])) for n in names}
    u_elastic = elastic_member_energy(hist, meta)
    work_int = sum(work_springs.values()) + u_elastic
    balance = work_ext - work_int
    cycles = []
    idx = hist["target_index"]
    for level in range(len(DRIFT_PROTOCOL_PCT)):
        for c in range(CYCLES_PER_LEVEL):
            k0 = (level * CYCLES_PER_LEVEL + c) * 2
            sel = (idx == k0) | (idx == k0 + 1)
            if not sel.any():
                continue
            start = int(np.argmax(sel))
            stop = int(len(sel) - np.argmax(sel[::-1]))
            seg = slice(max(start - 1, 0), stop)
            first, last = max(start - 1, 0), stop - 1
            ext = _work(hist["P"][seg], hist["u_top"][seg])
            per = {n: _work(hist["moment"][n][seg], hist["rotation"][n][seg]) for n in names}
            stored_change = {n: float(stored[n][last] - stored[n][first]) for n in names}
            cycles.append({"level_pct": DRIFT_PROTOCOL_PCT[level], "cycle": c + 1, "external_work_kip_in": ext,
                           "component_work_kip_in": per, "component_elastic_reference_stored_change_kip_in": stored_change,
                           "component_work_less_elastic_reference_kip_in": {n: per[n] - stored_change[n] for n in names},
                           "sum_component_work_kip_in": sum(per.values()),
                           "elastic_member_energy_change_kip_in": float(u_elastic[last] - u_elastic[first])})
    # Kinematic identities (small displacements, Linear transformation). Every spring pair is recorded in
    # the physical sign (core-side node first), so a member-end tangent is the core rotation plus the slip
    # rotation plus the hinge rotation. An elastic segment loaded by M at the hinge end and free at its far
    # end (roller or pin) has tangent - chord = -M L'/(3EI) with M the spring's own moment, which equals
    # the member-end moment by equilibrium at the hinge node. Hence
    #   column half: chord phi = theta_J + theta_slip + theta_hinge + M L'/(3EI),   drift = (phi_T + phi_B)/2
    #   beam half:   theta_B + theta_slip + theta_hinge + M L'/(3EI) = chord = -/+ u_z(J)/L' (left / right beam),
    # the beam chord being the core's vertical displacement (column axial shortening) over the half bay.
    ei_c = meta["column"]["e_ksi"] * meta["column"]["i_in4"]
    ei_b = meta["beam"]["e_ksi"] * meta["beam"]["iy_in4"]
    zeros = np.zeros_like(drift)
    phi, column_terms = {}, {}
    for end in ("T", "B"):
        th_h = hist["rotation"][f"column_hinge_{end}"]
        th_s = hist["rotation"].get(f"column_slip_{end}", zeros)
        m_h = hist["moment"][f"column_hinge_{end}"]
        elastic = m_h * (H / 2.0) / (3.0 * ei_c)
        phi[end] = (hist["theta_J"] + th_s + th_h) + elastic
        column_terms[end] = {"core": hist["theta_J"], "slip": th_s, "hinge": th_h, "elastic": elastic}
    drift_reconstructed = 0.5 * (phi["T"] + phi["B"])
    drift_err = float(np.max(np.abs(drift_reconstructed - drift)))
    beam_residual, beam_terms = {}, {}
    for side in ("L", "R"):
        th_h = hist["rotation"][f"beam_hinge_{side}"]
        th_s = hist["rotation"].get(f"beam_slip_{side}", zeros)
        m_h = hist["moment"][f"beam_hinge_{side}"]
        elastic = m_h * (L / 2.0) / (3.0 * ei_b)
        chord = (-1.0 if side == "L" else 1.0) * hist["u_z_J"] / (L / 2.0)
        beam_residual[side] = (hist["theta_B"] + th_s + th_h) + elastic - chord
        beam_terms[side] = {"slip": th_s, "hinge": th_h, "elastic": elastic, "chord": chord}
    gamma = hist["theta_B"] - hist["theta_J"]
    # Equilibrium at the beam core B. In the physical (core-side-first) signs the panel spring has B as
    # its second node and each beam hinge has B as its first node, so the resisting moments at B are
    # +M_panel, -M_beam_L, -M_beam_R and, with no external moment there, M_panel = M_beam_L + M_beam_R.
    beam_sum = hist["moment"]["beam_hinge_R"] + hist["moment"]["beam_hinge_L"]
    scale = max(float(np.max(np.abs(hist["moment"]["panel"]))), 1e-9)
    equilibrium = {"panel_minus_beam_sum_max_abs_kip_in": float(np.max(np.abs(hist["moment"]["panel"] - beam_sum))),
                   "panel_minus_beam_sum_relative": float(np.max(np.abs(hist["moment"]["panel"] - beam_sum)) / scale),
                   "basis": "M_panel = M_beam_L + M_beam_R in the recorded physical signs (moment equilibrium at the beam core)"}
    # elastic flexibility: at the first peak of the first two levels, the secant stiffness and where the drift comes from
    flexibility = {}
    for level in range(2):
        k0 = level * CYCLES_PER_LEVEL * 2
        sel = np.where(idx == k0)[0]
        if not len(sel):
            continue
        at = int(sel[-1])
        d = drift[at] if abs(drift[at]) > 0 else 1.0
        column_share = {term: float(0.5 * (column_terms["T"][term][at] + column_terms["B"][term][at]) / d) for term in ("core", "slip", "hinge", "elastic")}
        beam_share = {side: {term: float(beam_terms[side][term][at] / hist["theta_B"][at]) if abs(hist["theta_B"][at]) > 0 else None
                             for term in ("slip", "hinge", "elastic", "chord")} for side in ("L", "R")}
        flexibility[f"{DRIFT_PROTOCOL_PCT[level]:g}"] = {
            "drift": float(drift[at]), "shear_kip": float(hist["P"][at]),
            "secant_stiffness_kip_per_in": float(hist["P"][at] / hist["u_top"][at]) if hist["u_top"][at] else None,
            "drift_share_from_column_side": column_share,
            "core_rotation_theta_J": float(hist["theta_J"][at]), "beam_core_rotation_theta_B": float(hist["theta_B"][at]),
            "panel_shear_strain": float(gamma[at]),
            "beam_core_rotation_share": beam_share,
            "component_rotations": {n: float(hist["rotation"][n][at]) for n in names}}
    peaks = {n: {"rotation_max": float(hist["rotation"][n].max()), "rotation_min": float(hist["rotation"][n].min()),
                 "moment_max_kip_in": float(hist["moment"][n].max()), "moment_min_kip_in": float(hist["moment"][n].min()),
                 "work_kip_in": float(work_springs[n][-1]),
                 "work_less_elastic_reference_kip_in": float(work_springs[n][-1] - (stored[n][-1] - stored[n][0])),
                 "stayed_elastic": stayed_elastic[n],
                 "energy_basis": ("exact: the component never left its elastic branch, so work less the M^2/(2Ke) change is its dissipation (zero to solver precision)"
                                  if stayed_elastic[n] else
                                  "estimate: M^2/(2Ke) is the linear-elastic reference; not established for a deteriorating IMK state; do not fit energy capacities to it")}
             for n in names}
    first_yield = {}
    for side in ("L", "R"):
        c = components[f"beam_hinge_{side}"]
        m = hist["moment"][f"beam_hinge_{side}"]
        pos, neg = ((c["my_sagging_kip_in"], c["my_hogging_kip_in"]) if side == "L" else (c["my_hogging_kip_in"], c["my_sagging_kip_in"]))
        hit = np.where((m >= 0.99 * pos) | (m <= -0.99 * neg))[0]
        first_yield[side] = {"drift_pct": float(100.0 * abs(drift[hit[0]])) if len(hit) else None,
                             "step": int(hit[0]) if len(hit) else None}
    return {"drift": drift, "work_ext": work_ext, "work_springs": work_springs, "stored": stored, "u_elastic": u_elastic, "work_int": work_int,
            "balance": balance, "balance_max_abs_kip_in": float(np.max(np.abs(balance))),
            "balance_relative_to_external": float(np.max(np.abs(balance)) / max(np.max(np.abs(work_ext)), 1e-9)),
            "cycles": cycles, "drift_reconstructed": drift_reconstructed, "drift_reconstruction_max_error": drift_err,
            "gamma": gamma, "beam_kinematic_residual_max": {s: float(np.max(np.abs(r))) for s, r in beam_residual.items()},
            "equilibrium": equilibrium, "elastic_flexibility": flexibility,
            "energy_accounting": {"signed_work": "integral M dtheta per component from its own pair (reliable; used in the global balance)",
                                  "elastic_reference": "M^2 / (2 Ke) on the installed elastic stiffness; exact stored energy only for a component that stayed elastic",
                                  "exact_for": [n for n in names if stayed_elastic[n]],
                                  "estimate_for": [n for n in names if not stayed_elastic[n]]},
            "peaks": peaks, "first_beam_yield": first_yield,
            "peak_drift_pct": float(100.0 * np.max(np.abs(drift))), "peak_force_kip": float(np.max(np.abs(hist["P"])))}


def _elastic_rotation_limit(component):
    """The rotation beyond which the component's material leaves its elastic branch (its smaller yield
    rotation on the installed Ke)."""
    ke = component["ke_kip_in_per_rad"]
    if component["kind"] == "panel":
        return component["calibration"]["mn_kip_in"] / ke
    if "my_kip_in" in component:
        return component["my_kip_in"] / ke
    if "my_hogging_kip_in" in component:
        return min(component["my_hogging_kip_in"], component["my_sagging_kip_in"]) / ke
    cal = component["calibration"]
    strengths = [cal[s]["my"] for s in ("hogging", "sagging")] if "hogging" in cal else [cal["my"]]
    return min(strengths) / ke


def compare_histories(reference, other, names):
    """Maximum relative discrepancy between two runs of the same physical fixture (both in physical signs)."""
    out = {}
    scale_p = max(float(np.max(np.abs(reference["P"]))), 1e-12)
    out["shear"] = float(np.max(np.abs(reference["P"] - other["P"])) / scale_p)
    for n in names:
        sm = max(float(np.max(np.abs(reference["moment"][n]))), 1e-12)
        sr = max(float(np.max(np.abs(reference["rotation"][n]))), 1e-12)
        out[n] = {"moment": float(np.max(np.abs(reference["moment"][n] - other["moment"][n])) / sm),
                  "rotation": float(np.max(np.abs(reference["rotation"][n] - other["rotation"][n])) / sr)}
    out["max"] = max([out["shear"]] + [out[n][k] for n in names for k in ("moment", "rotation")])
    return out


# ---------------------------------------------------------------------------------------------------
# plots and output
# ---------------------------------------------------------------------------------------------------
def plot(label, hist, meta, ev, components, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    drift = 100.0 * ev["drift"]
    names = [n for n in components]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    ax = axes[0][0]
    ax.plot(drift, hist["P"], color="black", lw=0.8)
    ax.set_xlabel("story drift (%)"); ax.set_ylabel("column shear (kip)"); ax.set_title(f"{label}: force-drift loops", fontsize=10)
    ax.grid(True, lw=0.3, color="0.9")
    order = ["beam_hinge_L", "beam_slip_L", "panel", "column_hinge_T", "column_slip_T"]
    labels = {"beam_hinge_L": "beam hinge (left face)", "beam_slip_L": "beam slip interface (left face)", "panel": "panel shear spring",
              "column_hinge_T": "column hinge (top)", "column_slip_T": "column slip interface (top)"}
    for ax, name in zip((axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2]), order):
        if name not in components:
            ax.axis("off"); ax.set_title(f"{labels[name]}: not in this variant", fontsize=9); continue
        ax.plot(100.0 * hist["rotation"][name], hist["moment"][name] / 1000.0, color="black", lw=0.7)
        c = components[name]
        for key in ("my_hogging_kip_in", "my_kip_in"):
            if key in c:
                ax.axhline(c[key] / 1000.0, color="0.7", ls=":", lw=0.8)
        if "my_sagging_kip_in" in c:
            ax.axhline(-c["my_sagging_kip_in"] / 1000.0, color="0.7", ls=":", lw=0.8)
        if name == "panel":
            ax.axhline(c["calibration"]["mn_kip_in"] / 1000.0, color="0.7", ls=":", lw=0.8); ax.axhline(-c["calibration"]["mn_kip_in"] / 1000.0, color="0.7", ls=":", lw=0.8)
        ax.set_xlabel("rotation (%)"); ax.set_ylabel("moment (10³ kip-in)")
        p = ev["peaks"][name]
        ax.set_title(f"{labels[name]}: signed work {p['work_kip_in']:.0f} kip-in; less elastic reference {p['work_less_elastic_reference_kip_in']:.0f} "
                     f"({'exact, elastic' if p['stayed_elastic'] else 'estimate'})", fontsize=8.5)
        ax.grid(True, lw=0.3, color="0.9")
    fig.suptitle(f"{label} subassembly, floor {meta['floor']}, {meta['plane']} plane: {meta['column']['b_in']:g}x{meta['column']['h_in']:g} column / "
                 f"{meta['beam']['b_in']:g}x{meta['beam']['h_in']:g} beam; peak drift {ev['peak_drift_pct']:.1f} %, "
                 f"work balance error {100 * ev['balance_relative_to_external']:.3f} % of external work", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / f"{label}_loops.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    step = np.arange(len(drift))
    bottom = np.zeros_like(drift)
    greys = np.linspace(0.85, 0.1, len(names))
    for g, n in zip(greys, names):
        w = ev["work_springs"][n]
        axes[0].fill_between(step, bottom, bottom + w, color=str(g), label=n)
        bottom = bottom + w
    axes[0].plot(step, ev["work_ext"], color="red", lw=0.8, ls="--", label="external work")
    axes[0].plot(step, ev["work_int"], color="black", lw=0.6, label="internal (springs + elastic members)")
    axes[0].set_ylabel("cumulative work (kip-in)"); axes[0].legend(fontsize=7, ncol=3, loc="upper left"); axes[0].grid(True, lw=0.3, color="0.9")
    axes[0].set_title("work accumulated by component (stacked), against the external work", fontsize=10)
    for g, n in zip(greys, names):
        axes[1].plot(step, 100.0 * hist["rotation"][n], color=str(g), lw=0.7, label=n)
    axes[1].plot(step, drift, color="red", lw=0.8, ls="--", label="story drift")
    axes[1].plot(step, 100.0 * ev["drift_reconstructed"], color="black", lw=0.5, ls=":", label="drift reconstructed from rotations and elastic terms")
    axes[1].set_ylabel("rotation, drift (%)"); axes[1].set_xlabel("step"); axes[1].legend(fontsize=7, ncol=3, loc="upper left"); axes[1].grid(True, lw=0.3, color="0.9")
    fig.suptitle(f"{label}: component work and rotations over the protocol", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / f"{label}_work_and_rotations.png", dpi=140); plt.close(fig)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _jsonable(vars(value))
    return value


def export(label, hist, meta, ev, components, out):
    """Complete exports: JSON result, compressed histories, a plain CSV of every recorded pair, the plots."""
    names = list(components)
    plot(label, hist, meta, ev, components, out)
    np.savez_compressed(out / f"{label}_histories.npz", u_top=hist["u_top"], u_z_J=hist["u_z_J"], P=hist["P"], theta_J=hist["theta_J"],
                        theta_B=hist["theta_B"], drift_reconstructed=ev["drift_reconstructed"], work_ext=ev["work_ext"], u_elastic=ev["u_elastic"],
                        **{f"moment_{n}": hist["moment"][n] for n in names}, **{f"rotation_{n}": hist["rotation"][n] for n in names})
    with (out / f"{label}_histories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "u_top_in", "drift", "P_kip", "theta_J", "theta_B", "u_z_J_in"] +
                        [f"{kind}_{n}" for n in names for kind in ("moment_kip_in", "rotation_rad")])
        for k in range(len(hist["u_top"])):
            writer.writerow([k, hist["u_top"][k], ev["drift"][k], hist["P"][k], hist["theta_J"][k], hist["theta_B"][k], hist["u_z_J"][k]] +
                            [val for n in names for val in (hist["moment"][n][k], hist["rotation"][n][k])])
    result = {"meta": _jsonable(meta), "peaks": ev["peaks"], "first_beam_yield": ev["first_beam_yield"],
              "work_balance": {"max_abs_kip_in": ev["balance_max_abs_kip_in"], "relative_to_external": ev["balance_relative_to_external"]},
              "drift_reconstruction": {"max_error_drift_ratio": ev["drift_reconstruction_max_error"], "basis": "explicit identities; no sign search"},
              "beam_kinematic_residual_max": ev["beam_kinematic_residual_max"], "equilibrium": ev["equilibrium"],
              "elastic_flexibility": ev["elastic_flexibility"], "energy_accounting": ev["energy_accounting"],
              "peak_drift_pct": ev["peak_drift_pct"], "peak_force_kip": ev["peak_force_kip"],
              "panel_gamma_max": float(np.max(np.abs(ev["gamma"]))),
              "cycles": ev["cycles"], "solver_failures": hist["solver_failures"], "solver_retries": hist["solver_retries"],
              "step_pct": hist["step_pct"], "solver": hist["solver"]}
    (out / f"{label}_result.json").write_text(json.dumps(_jsonable(result), indent=1, allow_nan=False), encoding="utf-8")
    return result


def run_one(record, case, variant, floor, column_slip, out, *, axis="x", reverse_springs=False, suppress=False, step_pct=STEP_DRIFT_PCT,
            label=None, write=True, tolerance=SOLVER_TOLERANCE, iterations=SOLVER_ITERATIONS, algorithm="Newton"):
    label = label or f"{variant}_{axis}" + ("_reversed" if reverse_springs else "") + ("_nodet" if suppress else "")
    t0 = time.time()
    components, meta = build(record, case, variant, floor, column_slip, sp.STORY_H, axis=axis, reverse_springs=reverse_springs,
                             suppress_member_deterioration=suppress)
    hist = run_protocol(components, meta, step_pct=step_pct, tolerance=tolerance, iterations=iterations, algorithm=algorithm)
    ev = evaluate(hist, meta, components)
    result = export(label, hist, meta, ev, components, out) if write else None
    if result is not None:
        result["elapsed_sec"] = time.time() - t0
        result["label"] = label
    print(f"[{label}] peak drift {ev['peak_drift_pct']:.2f} %, peak shear {ev['peak_force_kip']:.1f} kip, work balance "
          f"{100 * ev['balance_relative_to_external']:.4f} % of external, drift reconstruction {ev['drift_reconstruction_max_error']:.1e}, "
          f"beam identity {max(ev['beam_kinematic_residual_max'].values()):.1e}, equilibrium {ev['equilibrium']['panel_minus_beam_sum_relative']:.1e}, "
          f"retries {hist['solver_retries']}, failures {hist['solver_failures']}", flush=True)
    for n in components:
        p = ev["peaks"][n]
        print(f"    {n:18s} rot {100 * p['rotation_min']:+.3f} .. {100 * p['rotation_max']:+.3f} %  moment {p['moment_min_kip_in']:8.0f} .. "
              f"{p['moment_max_kip_in']:8.0f}  work {p['work_kip_in']:8.1f}  less elastic ref {p['work_less_elastic_reference_kip_in']:8.1f} kip-in "
              f"({'exact' if p['stayed_elastic'] else 'estimate'})", flush=True)
    return components, meta, hist, ev, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True); parser.add_argument("--case", required=True); parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", default="all", choices=VARIANTS + ("all",))
    parser.add_argument("--axis", default="x", choices=AXES + ("both",), help="in-plane bending axis of the joint (x or y beams)")
    parser.add_argument("--column-slip", action="store_true", help="also place slip interfaces at the column faces (unverified for through-bars)")
    parser.add_argument("--floor", type=int, default=None, help="floor of the interior joint whose calibration is used (default: floor 2)")
    parser.add_argument("--reverse-springs", action="store_true", help="reverse every spring's node order (materials mapped accordingly)")
    parser.add_argument("--suppress-member-deterioration", action="store_true", help="give the member hinges suppressed energies in-process")
    parser.add_argument("--step", type=float, default=STEP_DRIFT_PCT, help="drift step in percent")
    parser.add_argument("--tolerance", type=float, default=SOLVER_TOLERANCE, help="NormDispIncr tolerance")
    parser.add_argument("--iterations", type=int, default=SOLVER_ITERATIONS, help="iterations before the KrylovNewton retry")
    parser.add_argument("--algorithm", default="Newton", choices=("Newton", "KrylovNewton", "ModifiedNewton"))
    parser.add_argument("--orientation-regression", action="store_true",
                        help="run installed and slip_interfaces in both spring orientations (member deterioration suppressed), require "
                             f"relative discrepancy < {ORIENTATION_TOLERANCE:g}; then the production law in both orientations, reported; "
                             "then a half-step convergence check")
    args = parser.parse_args(argv)
    from Analysis.Fixed_Design_Diagnostics import install_record, read_record
    from Model import Deformation_Ownership as own
    root, out = Path(args.root), Path(args.output_root) / args.case
    out.mkdir(parents=True, exist_ok=True)
    _, _, record = read_record(root, args.case)
    with contextlib.redirect_stdout(io.StringIO()):
        install_record(record, args.case)
    floor = args.floor or min(2, sp.NUM_FLOOR)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    axes = AXES if args.axis == "both" else (args.axis,)
    summary = {"case": args.case, "design_root": str(root), "record_sha256": record.get("request_identity", {}).get("sha256"),
               "protocol": {"drift_levels_pct": DRIFT_PROTOCOL_PCT, "cycles_per_level": CYCLES_PER_LEVEL, "step_drift_pct": args.step,
                            "solver": {"test": "NormDispIncr", "tolerance": args.tolerance, "iterations": args.iterations, "algorithm": args.algorithm}},
               "sources": SOURCES, "slip_ownership_policy": own.POLICY_ID, "runs": {}, "orientation_regression": None}
    for axis in axes:
        for variant in variants:
            *_, result = run_one(record, args.case, variant, floor, args.column_slip, out, axis=axis, reverse_springs=args.reverse_springs,
                                 suppress=args.suppress_member_deterioration, step_pct=args.step, tolerance=args.tolerance, iterations=args.iterations, algorithm=args.algorithm)
            summary["runs"][result["label"]] = result
    if args.orientation_regression:
        regression = {"tolerance": ORIENTATION_TOLERANCE, "suppressed_member_deterioration": {}, "production_member_law": {}, "convergence": {}}
        for axis in axes:
            for variant in ("installed", "slip_interfaces"):
                for suppress, bucket in ((True, "suppressed_member_deterioration"), (False, "production_member_law")):
                    runs = {}
                    for reverse in (False, True):
                        components, meta, hist, ev, result = run_one(record, args.case, variant, floor, args.column_slip, out, axis=axis,
                                                                     reverse_springs=reverse, suppress=suppress, step_pct=args.step, tolerance=args.tolerance, iterations=args.iterations, algorithm=args.algorithm)
                        runs[reverse] = (hist, list(components))
                        summary["runs"][result["label"]] = result
                    disc = compare_histories(runs[False][0], runs[True][0], runs[False][1])
                    disc["passed"] = bool(disc["max"] < ORIENTATION_TOLERANCE)
                    regression[bucket][f"{variant}_{axis}"] = disc
                    note = ("PASS" if disc["passed"] else "FAIL") if suppress else "reported; the legacy member energy anchor E_ref = Lambda x Fy_positive is orientation-dependent"
                    print(f"[orientation, {bucket}] {variant} {axis}: max relative discrepancy {disc['max']:.2e} ({note})", flush=True)
            coarse = summary["runs"][f"slip_interfaces_{axis}"]
            *_, fine = run_one(record, args.case, "slip_interfaces", floor, args.column_slip, out, axis=axis, step_pct=args.step / 2.0,
                               label=f"slip_interfaces_{axis}_halfstep", tolerance=args.tolerance, iterations=args.iterations, algorithm=args.algorithm)
            summary["runs"][fine["label"]] = fine
            regression["convergence"][f"slip_interfaces_{axis}"] = {
                "step_pct": [args.step, args.step / 2.0],
                "peak_shear_kip": [coarse["peak_force_kip"], fine["peak_force_kip"]],
                "peak_shear_relative_change": abs(fine["peak_force_kip"] - coarse["peak_force_kip"]) / coarse["peak_force_kip"],
                "beam_hinge_L_work_kip_in": [coarse["peaks"]["beam_hinge_L"]["work_kip_in"], fine["peaks"]["beam_hinge_L"]["work_kip_in"]],
                "beam_slip_L_work_kip_in": [coarse["peaks"]["beam_slip_L"]["work_kip_in"], fine["peaks"]["beam_slip_L"]["work_kip_in"]],
                "work_balance_relative": [coarse["work_balance"]["relative_to_external"], fine["work_balance"]["relative_to_external"]]}
        summary["orientation_regression"] = regression
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1, allow_nan=False), encoding="utf-8")
    ops.wipe()
    own.reset_slip_interfaces()
    return summary


if __name__ == "__main__":
    main()
