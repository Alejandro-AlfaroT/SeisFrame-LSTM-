"""Frame model builder: node numbering, base fixity, sections, members, rigid floors, mass.

Everything that turns Structure_Parameters into the OpenSees domain lives
here, together with the constraint handler the chosen element formulation
needs (penalty ties for the IMK springs' equalDOF chains, Transformation
otherwise). The numbering helpers (node_tag, diaphragm_master_tag,
floor_master_node, roof_master_node) are the contract every analysis, load
and export module relies on; Design.Export_SAP2000 mirrors them so SAP2000
joint labels match. Merged on 2026-09-25 from Model/nodes.py,
Model/diaphragms.py, Model/mass.py, Model/elements.py and
Analysis/Constraints.py; no numbering or ordering changed.
"""
import openseespy.opensees as ops

import Structure_Parameters as sp
from Model import Member_Groups as mg
from Model.IMK_Hinges import create_imk_member, reset_hinge_registry
from Model.Joint_Springs import install_joint_springs, joint_beam_node, reset_joint_registry
from Model.Sections import define_sections


# ---- node numbering and base fixity -----------------------------------------------------------------
def node_tag(k, i, j):
    return (
        k * ((sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1))
        + j * (sp.NUM_BAY_X + 1)
        + i
        + 1
    )


def diaphragm_master_tag(k):
    return 100000 + k


def floor_master_node(k):
    return diaphragm_master_tag(k)


def roof_master_node():
    return diaphragm_master_tag(sp.NUM_FLOOR)


def floor_nodes(k):
    nodes = []

    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            nodes.append(node_tag(k, i, j))

    return nodes


def create_nodes():
    for k in range(sp.NUM_FLOOR + 1):
        z = k * sp.STORY_H
        for j in range(sp.NUM_BAY_Y + 1):
            y = j * sp.BAY_Y
            for i in range(sp.NUM_BAY_X + 1):
                x = i * sp.BAY_X
                ops.node(node_tag(k, i, j), x, y, z)

    x_center = sp.NUM_BAY_X * sp.BAY_X / 2.0
    y_center = sp.NUM_BAY_Y * sp.BAY_Y / 2.0
    for k in range(1, sp.NUM_FLOOR + 1):
        ops.node(diaphragm_master_tag(k), x_center, y_center, k * sp.STORY_H)


def fix_base_nodes():
    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            ops.fix(node_tag(0, i, j), 1, 1, 1, 1, 1, 1)


# ---- members ----------------------------------------------------------------------------------------
def define_geometric_transformations():
    ops.geomTransf("PDelta", sp.COL_TRANSF_TAG, 1, 0, 0)
    ops.geomTransf("Linear", sp.BEAM_X_TRANSF_TAG, 0, 0, 1)
    ops.geomTransf("Linear", sp.BEAM_Y_TRANSF_TAG, 0, 0, 1)


def _create_frame_member(ele_tag, n_i, n_j, member_type, transf_tag, integ_tag):
    use_imk = (
        sp.ELEMENT_FORMULATION == "imk"
        and (
            (member_type == "column" and sp.IMK_APPLY_TO_COLUMNS)
            or (member_type in {"beam_x", "beam_y"} and sp.IMK_APPLY_TO_BEAMS)
        )
    )

    # Under the scissors joint model beams frame into the joint's beam core,
    # columns into the joint itself (Model/Joint_Springs); the joint nodes
    # remain the member's physical identity.
    joint_i, joint_j = n_i, n_j
    if member_type in {"beam_x", "beam_y"}:
        n_i, n_j = joint_beam_node(n_i), joint_beam_node(n_j)

    if use_imk:
        create_imk_member(ele_tag, n_i, n_j, member_type, transf_tag, joint_nodes=(joint_i, joint_j))
        return

    if sp.ELEMENT_FORMULATION in {"fiber", "imk"}:
        if mg.is_grouped():
            raise mg.GroupedStateError(
                f"Member {ele_tag} ({member_type}) would be a fiber element: the fiber sections are defined once per "
                "member type and are not wired for a grouped design. Grouped designs run with IMK hinges on beams and "
                "columns (Model/Analysis_Profile v2_nonlinear_flexure_screening_v1).")
        ops.element(
            "forceBeamColumn",
            ele_tag,
            n_i,
            n_j,
            transf_tag,
            integ_tag,
        )
        return

    raise ValueError(f"Unknown ELEMENT_FORMULATION: {sp.ELEMENT_FORMULATION}")


def create_column_elements(start_ele_tag=1):
    ele_tag = start_ele_tag

    for k in range(sp.NUM_FLOOR):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k + 1, i, j)

                _create_frame_member(
                    ele_tag,
                    n_i,
                    n_j,
                    "column",
                    sp.COL_TRANSF_TAG,
                    sp.COL_INTEG_TAG,
                )
                ele_tag += 1

    return ele_tag


def create_beam_x_elements(start_ele_tag):
    ele_tag = start_ele_tag

    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k, i + 1, j)

                _create_frame_member(
                    ele_tag,
                    n_i,
                    n_j,
                    "beam_x",
                    sp.BEAM_X_TRANSF_TAG,
                    sp.BEAM_INTEG_TAG,
                )
                ele_tag += 1

    return ele_tag


def create_beam_y_elements(start_ele_tag):
    ele_tag = start_ele_tag

    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                n_i = node_tag(k, i, j)
                n_j = node_tag(k, i, j + 1)

                _create_frame_member(
                    ele_tag,
                    n_i,
                    n_j,
                    "beam_y",
                    sp.BEAM_Y_TRANSF_TAG,
                    sp.BEAM_INTEG_TAG,
                )
                ele_tag += 1

    return ele_tag


def create_elements():
    define_geometric_transformations()

    next_ele_tag = create_column_elements(start_ele_tag=1)
    next_ele_tag = create_beam_x_elements(next_ele_tag)
    create_beam_y_elements(next_ele_tag)


# ---- rigid floors and seismic mass ------------------------------------------------------------------
def create_rigid_diaphragms():
    for k in range(1, sp.NUM_FLOOR + 1):
        master = floor_master_node(k)
        slaves = [n for n in floor_nodes(k) if n != master]

        ops.fix(master, 0, 0, 1, 1, 1, 0)
        ops.rigidDiaphragm(3, master, *slaves)


def assign_nodal_masses():
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                n = node_tag(k, i, j)
                m = installed_node_seismic_mass(k, i, j)
                ops.mass(n, m, m, 1.0e-8, 0.0, 0.0, 0.0)


def installed_node_seismic_mass(k, i, j):
    """Seismic mass at grid node (i, j) of floor k: one value per plan position in the uniform mode,
    the weights of that floor's own members under a grouped design (Model.Member_Properties)."""
    from Model import Roof_Extension as roof
    if mg.is_grouped():
        from Model import Member_Properties as mp
        return mp.node_seismic_mass(k, i, j)                   # the grouped ledger carries the roof extension
    return sp.node_seismic_mass(i, j) + roof.node_seismic_mass(k, i, j)


# ---- constraint handler -----------------------------------------------------------------------------
def apply_analysis_constraints():
    """The handler the element formulation needs: penalty ties for the IMK spring chains."""
    if sp.ELEMENT_FORMULATION == "imk":
        ops.constraints("Penalty", sp.PENALTY_ALPHA_SP, sp.PENALTY_ALPHA_MP)
        return

    ops.constraints("Transformation")


# ---- assembly ---------------------------------------------------------------------------------------
def _require_grouped_model_scope():
    """What the nonlinear frame supports under a grouped design; anything else is refused, not approximated.

    The fiber sections (one per member type) are not defined: every member must be an IMK member built
    from its own group's design. The scissors joint springs are calibrated from one beam and one column
    section and are not wired for groups; grouped designs use rigid centerline joints (the V2 profile).
    """
    if not (sp.ELEMENT_FORMULATION == "imk" and sp.IMK_APPLY_TO_COLUMNS and sp.IMK_APPLY_TO_BEAMS):
        raise mg.GroupedStateError("A grouped design needs ELEMENT_FORMULATION 'imk' with IMK hinges on beams and columns; "
                                   "the fiber sections are one per member type and are not wired for groups.")
    if getattr(sp, "JOINT_MODEL", "rigid_centerline") != "rigid_centerline":
        raise mg.GroupedStateError(f"JOINT_MODEL {sp.JOINT_MODEL!r} is not wired for a grouped design: the joint springs are "
                                   "calibrated from one beam and one column section. Use 'rigid_centerline'.")


def build_model():
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # Hinge backbones are per-build state: sections and axial loads change
    # between builds during the design search, so stale entries would
    # misreport what the current model actually contains.
    reset_hinge_registry()
    reset_joint_registry()
    # The slip-interface registry is scoped to this domain: a registration left from a previous build
    # would otherwise take bar slip away from a member hinge here while this build installs no
    # replacement interface (Unit 1 review, 2026-09-28, commit blocker 1).
    from Model.Deformation_Ownership import begin_domain, validate_installed
    begin_domain("Build_Model.build_model")

    create_nodes()
    fix_base_nodes()
    if mg.is_grouped():
        _require_grouped_model_scope()
    else:
        define_sections()
    # Joint springs (beam cores) must exist before the beams that frame into them.
    install_joint_springs(node_tag)
    create_elements()
    validate_installed(expect_none=True)      # this builder installs no face slip interfaces
    create_rigid_diaphragms()
    assign_nodal_masses()
