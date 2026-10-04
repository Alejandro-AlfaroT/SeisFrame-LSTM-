"""Elastic cracked-stiffness frame for preliminary design, not NTHA.

Same all-grid-line connectivity/tag order as Model.Build_Model. No IMK springs,
no material yielding, and no mutation of the chosen response-history model.
Columns retain P-Delta transformations. Centerline joints and rigid floors
are explicit idealizations; qualification still requires reviewing them.
"""
import openseespy.opensees as ops
import Structure_Parameters as sp
from Model import Member_Groups as mg
from Model import Member_Properties as mp
from Model.Build_Model import create_nodes, fix_base_nodes, node_tag, create_rigid_diaphragms, assign_nodal_masses
from Model.IMK_Hinges import reset_hinge_registry


def physical_members():
    tag = 1
    for k in range(sp.NUM_FLOOR):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                yield tag, node_tag(k, i, j), node_tag(k + 1, i, j), "column"
                tag += 1
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X):
                yield tag, node_tag(k, i, j), node_tag(k, i + 1, j), "beam_x"
                tag += 1
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                yield tag, node_tag(k, i, j), node_tag(k, i, j + 1), "beam_y"
                tag += 1


def beam_line_family(node_i, kind):
    """(axis, 'edge' | 'interior') of the beam that starts at ``node_i``, from the node grid."""
    per_story = (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    j, i = divmod((int(node_i) - 1) % per_story, sp.NUM_BAY_X + 1)
    if kind == "beam_x":
        return "x", "edge" if j in (0, sp.NUM_BAY_Y) else "interior"
    if kind == "beam_y":
        return "y", "edge" if i in (0, sp.NUM_BAY_X) else "interior"
    raise ValueError(f"{kind!r} is not a beam kind.")


def build_design_model():
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    reset_hinge_registry()
    create_nodes()
    fix_base_nodes()
    ops.geomTransf("PDelta", sp.COL_TRANSF_TAG, 1, 0, 0)
    ops.geomTransf("Linear", sp.BEAM_X_TRANSF_TAG, 0, 0, 1)
    ops.geomTransf("Linear", sp.BEAM_Y_TRANSF_TAG, 0, 0, 1)
    grouped = mg.is_grouped()
    for tag, ni, nj, kind in physical_members():
        column = kind == "column"
        if grouped:
            # Every member on its own group's section; its line's T/L flange from the actual neighbours.
            _create_resolved_member(tag, ni, nj, kind)
            continue
        b, h, fc = ((sp.B_COL, sp.H_COL, sp.FC_COL_KSI) if column else
                    (sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI))
        modifier = sp.section_stiffness_modifier("column" if column else "beam")
        e = sp.concrete_ec_ksi(fc)
        transform = (sp.COL_TRANSF_TAG if column else
                     sp.BEAM_X_TRANSF_TAG if kind == "beam_x" else sp.BEAM_Y_TRANSF_TAG)
        # Vertical bending: columns on the rectangular section, beams on the
        # T/L section of their line (ACI 318-19 R6.6.3.1.1, 6.3.2 flange).
        iy = sp.rect_iy(b, h) if column else sp.beam_flexural_inertia_in4(*beam_line_family(ni, kind))
        ops.element("elasticBeamColumn", tag, ni, nj, b * h, e,
                    sp.concrete_shear_modulus_ksi(e), modifier * sp.approx_rect_j(b, h),
                    modifier * iy, modifier * sp.rect_iz(b, h), transform)
    create_rigid_diaphragms()
    assign_nodal_masses()


def _create_resolved_member(tag, ni, nj, kind):
    member = mg.resolve(tag)
    if (member.member_type, member.node_i, member.node_j) != (kind, ni, nj):
        raise mg.GroupedStateError(f"Member {tag} is a {kind} between nodes {ni} and {nj} in the frame but the grouped "
                                   f"design holds a {member.member_type} between {member.node_i} and {member.node_j}.")
    section = mp.elastic_section(member)
    modifier = section["stiffness_modifier"]
    transform = (sp.COL_TRANSF_TAG if kind == "column" else
                 sp.BEAM_X_TRANSF_TAG if kind == "beam_x" else sp.BEAM_Y_TRANSF_TAG)
    ops.element("elasticBeamColumn", tag, ni, nj, section["area"], section["e"], section["g"],
                modifier * section["j"], modifier * section["iy"], modifier * section["iz"], transform)


def floor_xy_displacements(k):
    return {f"{i},{j}": {"x": ops.nodeDisp(node_tag(k, i, j), 1),
                         "y": ops.nodeDisp(node_tag(k, i, j), 2)}
            for j in range(sp.NUM_BAY_Y + 1) for i in range(sp.NUM_BAY_X + 1)}


def gravity_weight_per_story():
    """D+L and member weight, for the declared uniform-floor stability screen."""
    return sp.total_floor_gravity_load() + sp.total_structural_self_weight_per_floor()


def gravity_weight_above_story(story):
    """Gravity load carried at story ``story``: every floor from that story's top to the roof (kip).

    One floor weight repeated in the uniform mode (unchanged arithmetic); the installed members' own
    weights floor by floor under a grouped design.
    """
    from Model import Roof_Extension as roof
    if mg.is_grouped():
        return mp.gravity_above_kip(story)                     # the grouped ledger carries the roof extension
    return (sp.NUM_FLOOR - story + 1) * gravity_weight_per_story() + roof.total_weight_kip()
