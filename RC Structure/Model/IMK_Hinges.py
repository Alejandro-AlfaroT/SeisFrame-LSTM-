import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.IMK_Calibration import (
    bond_slip_indicator,
    backbone_for_member,
    column_axial_domain,
    column_gravity_axial,
    column_grid_position,
    column_moment_at_axial,
    column_pm_nominal,
)
from Model.IMK_Materials import (
    MAPPING_VERSION, CyclicParameters, RotationalBackbone, active_energy_modes, define_rotational_imk,
    define_mapped_rotational_imk, validate_energy_calibration,
)


# Per-hinge backbones recorded as the model is built, keyed by physical
# element tag.
_HINGE_REGISTRY = {}

# The nominal P-M sweep is identical for every column in a build, so it is
# computed once per model rather than once per member.
_PM_DIAGRAM_CACHE = {}


def reset_hinge_registry():
    """Clear per-build hinge state. Called when a new model is built."""
    _HINGE_REGISTRY.clear()
    _PM_DIAGRAM_CACHE.clear()


def hinge_registry():
    """All recorded hinge backbones, keyed by physical element tag."""
    return dict(_HINGE_REGISTRY)


def hinge_backbone(ele_tag):
    """Backbone recorded for one element, or None if it has no IMK hinges."""
    return _HINGE_REGISTRY.get(int(ele_tag))


def _cached_pm_diagram():
    key = (sp.B_COL, sp.H_COL, sp.FC_COL_KSI, sp.COL_BAR_SIZE,
           sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS)
    if key not in _PM_DIAGRAM_CACHE:
        _PM_DIAGRAM_CACHE.clear()
        _PM_DIAGRAM_CACHE[key] = column_pm_nominal()
    return _PM_DIAGRAM_CACHE[key]


def hinge_node_tag(ele_tag, end_id):
    return sp.IMK_HINGE_NODE_TAG_BASE + 10 * ele_tag + end_id


def hinge_element_tag(ele_tag, end_id):
    return sp.IMK_HINGE_ELEMENT_TAG_BASE + 10 * ele_tag + end_id


def imk_material_tag(ele_tag, end_id, local_rot_dir):
    return sp.IMK_MATERIAL_TAG_BASE + 100 * ele_tag + 10 * end_id + local_rot_dir


def _member_length(n_i, n_j):
    xi, yi, zi = ops.nodeCoord(n_i)
    xj, yj, zj = ops.nodeCoord(n_j)
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    length = math.sqrt(dx * dx + dy * dy + dz * dz)

    if length <= 0.0:
        raise ValueError(f"Zero-length physical member between nodes {n_i} and {n_j}.")

    return length


def beam_yield_moments(member_type, n_i, n_j):
    """(hogging, sagging) yield moments for a beam member, kip-in.

    Slab-aware designs use the developed beam-plus-slab strengths from
    Design.SMRF_Beam_Slab_Strength -- the same numbers the SCWB screen and
    the joint qualification compare columns against -- for the beam's
    family (x/y, perimeter/interior line). Without a slab the two signs
    come from the actual top and bottom bars. The legacy bundled-load model
    keeps its symmetric max(top, bottom) strength.
    """
    if sp.SLAB_THICKNESS_IN is None:
        symmetric = sp.beam_nominal_moment_y()
        return symmetric, symmetric, {"basis": "legacy symmetric max(top, bottom) bars", "family": None}
    from Design.SMRF_Beam_Slab_Strength import (composite_beam_strengths, beam_family,
                                                perimeter_slab_bar_anchorage, exterior_ends)
    xi, yi, _ = ops.nodeCoord(n_i)
    xj, yj, _ = ops.nodeCoord(n_j)
    if member_type == "beam_x":
        line = int(round(yi / sp.BAY_Y)) if sp.BAY_Y > 0 else 0
        span = int(round(min(xi, xj) / sp.BAY_X)) if sp.BAY_X > 0 else 0
        reversed_connectivity = xj < xi
    else:
        line = int(round(xi / sp.BAY_X)) if sp.BAY_X > 0 else 0
        span = int(round(min(yi, yj) / sp.BAY_Y)) if sp.BAY_Y > 0 else 0
        reversed_connectivity = yj < yi
    axis, position = beam_family(member_type, line, sp.NUM_BAY_X, sp.NUM_BAY_Y)
    exterior = exterior_ends(member_type, span, sp.NUM_BAY_X, sp.NUM_BAY_Y)
    if reversed_connectivity:
        exterior = {"i": exterior["j"], "j": exterior["i"]}
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout")
    beam = {"b_in": sp.B_BEAM, "h_in": sp.H_BEAM, "fc_ksi": sp.FC_BEAM_KSI, "fy_ksi": sp.FY_KSI,
            "bar_size": sp.BEAM_BAR_SIZE, "top_bars": sp.BEAM_TOP_BARS, "bot_bars": sp.BEAM_BOT_BARS,
            # This direction's top/bottom offsets and actual bar rows: the orthogonal cages stack at
            # the joints and the column lanes may force a second row (2026-09-27).
            "centroid_offset_in": sp.beam_bar_stacking_offsets_in()[axis],
            "layers": sp.beam_bar_layers()[axis]}
    # The chosen slab thickness enters whether or not a reinforcement layout
    # is established: with no layout composite_beam_strengths counts no slab
    # mats (hogging is the bare rectangle) but keeps the flange concrete in
    # compression under sagging, which needs no reinforcement to act. This is
    # the convention the design's family strengths, its SCWB pricing and the
    # F1 strength-distribution evidence use (SMRF_Beam_Slab_Strength
    # .beam_slab_strengths), and the elastic spine already carries the T/L
    # section; a hinge priced on thickness 0 was weaker in sagging than the
    # design assumed (2026-09-21).
    slab = {"thickness_in": sp.SLAB_THICKNESS_IN}
    geometry = {"bay_x_in": sp.BAY_X, "bay_y_in": sp.BAY_Y, "h_col_in": sp.H_COL, "b_col_in": sp.B_COL}
    family = composite_beam_strengths(beam, slab, layout, geometry, axis, position)
    anchorage = perimeter_slab_bar_anchorage(layout, axis, sp.B_BEAM, sp.BEAM_CLEAR_COVER_IN,
                                             sp.rebar_diameter(sp.BEAM_STIRRUP_BAR_SIZE), sp.FC_BEAM_KSI, sp.FY_KSI)
    hogging, sagging = family["mn_negative_kip_in"], family["mn_positive_kip_in"]
    undeveloped = layout is not None and anchorage is not None and not anchorage["developed"]
    # An exterior end whose slab bars are not developed at the perimeter
    # yields without them in either sign: rectangular hogging, and sagging
    # with the flange concrete but no bottom mat (SMRF_Beam_Slab_Strength).
    hogging_ends = {end: (family["mn_undeveloped_negative_kip_in"] if (undeveloped and exterior[end]) else hogging)
                    for end in ("i", "j")}
    sagging_ends = {end: (family["mn_undeveloped_positive_kip_in"] if (undeveloped and exterior[end]) else sagging)
                    for end in ("i", "j")}
    return hogging, sagging, {
        "basis": "beam plus developed slab mats in the ACI 6.3.2 flange" if layout is not None
                 else ("rectangular beam in hogging; the chosen slab's flange concrete in compression under sagging with no "
                       "slab mats counted (slab reinforcement not established)"),
        "family": f"{axis}_{position}",
        "effective_flange_width_in": family["effective_flange_width_in"],
        "slab_steel_in_flange_in2": family["slab_steel_in_flange_in2"],
        "hogging_i_kip_in": hogging_ends["i"], "hogging_j_kip_in": hogging_ends["j"],
        "sagging_i_kip_in": sagging_ends["i"], "sagging_j_kip_in": sagging_ends["j"],
        "exterior_ends": exterior, "exterior_anchorage": anchorage}


def beam_line_family(member_type, n_i):
    """(axis, 'edge' | 'interior') for the beam member starting at node n_i, from its coordinates."""
    from Design.SMRF_Beam_Slab_Strength import beam_family
    xi, yi, _ = ops.nodeCoord(n_i)
    if member_type == "beam_x":
        line = int(round(yi / sp.BAY_Y)) if sp.BAY_Y > 0 else 0
    else:
        line = int(round(xi / sp.BAY_X)) if sp.BAY_X > 0 else 0
    return beam_family(member_type, line, sp.NUM_BAY_X, sp.NUM_BAY_Y)


def _member_properties(member_type, axial_kip=0.0, family=None):
    """Elastic properties for an IMK member, on effective (cracked) stiffness.

    The stiffness modifier is applied once, here, so it reaches both the
    elastic element between the springs and the spring calibration itself
    (Ke = n * 6EI/L), keeping the two consistent. A beam's vertical-bending
    I is its line's T/L section (Structure_Parameters.beam_flexural_section),
    the same section the design frame analysed; ``family`` is (axis,
    position) from beam_line_family, and defaults to the interior line when
    the caller has no member in hand.
    """
    modifier = sp.section_stiffness_modifier(member_type)

    if member_type == "column":
        # Zero-axial flexural capacity understates a column badly: it ignores
        # the compression and side steel, and ignores axial load entirely.
        if getattr(sp, "IMK_USE_CALIBRATED_BACKBONE", True):
            diagram = _cached_pm_diagram()
            low, high = column_axial_domain(diagram)
            if not low <= axial_kip <= high:
                # Outside the nominal surface the section has no flexural
                # strength: it cannot carry the axial load.
                raise ValueError(f"Column gravity axial estimate {axial_kip:.1f} kip is outside the nominal "
                                 f"P-M surface [{low:.1f}, {high:.1f}] kip (0.80 P0 cap, ACI 318-19 22.4.2.1); "
                                 "the section cannot carry it and the hinge cannot be calibrated.")
            capacity = column_moment_at_axial(axial_kip, diagram)
            my = mz = capacity
        else:
            my, mz = sp.column_nominal_moment_y(), sp.column_nominal_moment_z()
        return {
            "area": sp.rect_area(sp.B_COL, sp.H_COL),
            "e": sp.concrete_ec_ksi(sp.FC_COL_KSI),
            "g": sp.concrete_shear_modulus_ksi(sp.concrete_ec_ksi(sp.FC_COL_KSI)),
            "j": modifier * sp.approx_rect_j(sp.B_COL, sp.H_COL),
            "iy": modifier * sp.rect_iy(sp.B_COL, sp.H_COL),
            "iz": modifier * sp.rect_iz(sp.B_COL, sp.H_COL),
            "my": my,
            "mz": mz,
            "theta_y": sp.IMK_COLUMN_THETA_Y,
            "stiffness_modifier": modifier,
        }

    axis, position = family if family is not None else (member_type[-1] if member_type in ("beam_x", "beam_y") else "x", "interior")
    section = sp.beam_flexural_section(axis, position)
    return {
        "area": sp.rect_area(sp.B_BEAM, sp.H_BEAM),
        "e": sp.concrete_ec_ksi(sp.FC_BEAM_KSI),
        "g": sp.concrete_shear_modulus_ksi(sp.concrete_ec_ksi(sp.FC_BEAM_KSI)),
        "j": modifier * sp.approx_rect_j(sp.B_BEAM, sp.H_BEAM),
        "iy": modifier * section["iy_in4"],
        "iz": modifier * sp.rect_iz(sp.B_BEAM, sp.H_BEAM),
        "iy_basis": section["basis"],
        "flange_width_in": section["flange_width_in"],
        "family": f"{axis}_{position}",
        "my": sp.beam_nominal_moment_y(),
        "mz": sp.beam_nominal_moment_z(),
        "theta_y": sp.IMK_BEAM_THETA_Y,
        "stiffness_modifier": modifier,
    }


def imk_member_properties(member_type):
    return _member_properties(member_type).copy()


def _elastic_rotational_stiffness(e_mod, inertia, length):
    return 6.0 * e_mod * inertia / length


def _hinge_yield_data(props, rot_dir):
    if rot_dir in (5, "rot_y"):
        return props["my"], props["iy"]

    if rot_dir in (6, "rot_z"):
        return props["mz"], props["iz"]

    raise ValueError(f"Unknown IMK hinge rotation direction: {rot_dir}")


def imk_hinge_stiffness_components(member_type, rot_dir, length, *, props=None, family=None, axial_kip=0.0):
    """Spring stiffness for a member; pass the member's own ``props`` (or its
    family/axial load) so an edge beam's spring uses its line's I and a
    column's its axial state, the same properties its elastic element gets."""
    props = props if props is not None else _member_properties(member_type, axial_kip=axial_kip, family=family)
    yield_moment, inertia = _hinge_yield_data(props, rot_dir)

    if props["theta_y"] <= 0.0:
        raise ValueError(f"IMK theta_y must be positive for {member_type} {rot_dir}.")

    yield_based_stiffness = yield_moment / props["theta_y"]
    member_based_stiffness = (
        sp.IMK_HINGE_STIFFNESS_FACTOR
        * _elastic_rotational_stiffness(props["e"], inertia, length)
    )

    mode = getattr(sp, "IMK_HINGE_STIFFNESS_MODE", "yield_rotation")

    if mode == "yield_rotation":
        selected_stiffness = yield_based_stiffness
    elif mode == "member_stiffness_factor":
        selected_stiffness = member_based_stiffness
    elif mode == "max":
        selected_stiffness = max(yield_based_stiffness, member_based_stiffness)
    else:
        raise ValueError(
            "Unknown IMK_HINGE_STIFFNESS_MODE "
            f"{mode!r}; use 'yield_rotation', 'member_stiffness_factor', or 'max'."
        )

    return {
        "mode": mode,
        "yield_moment": yield_moment,
        "target_theta_y": props["theta_y"],
        "yield_based_stiffness": yield_based_stiffness,
        "member_based_stiffness": member_based_stiffness,
        "selected_stiffness": selected_stiffness,
        "actual_theta_y": yield_moment / selected_stiffness,
    }


def imk_hinge_stiffness(member_type, rot_dir, length, *, props=None, family=None, axial_kip=0.0):
    return imk_hinge_stiffness_components(
        member_type, rot_dir, length, props=props, family=family, axial_kip=axial_kip,
    )["selected_stiffness"]


def imk_hinge_thresholds(member_type, rot_dir, length, *, props=None, family=None, axial_kip=0.0):
    components = imk_hinge_stiffness_components(member_type, rot_dir, length, props=props, family=family,
                                                axial_kip=axial_kip)
    theta_y = components["actual_theta_y"]
    theta_p = max(sp.IMK_THETA_P_POS, sp.IMK_THETA_P_NEG)

    return {
        "theta_y": theta_y,
        "target_theta_y": components["target_theta_y"],
        "theta_cap": theta_y + theta_p,
        "theta_u": max(sp.IMK_THETA_U_POS, sp.IMK_THETA_U_NEG),
        "stiffness_mode": components["mode"],
        "selected_stiffness": components["selected_stiffness"],
    }


def imk_elastic_inertia_factor():
    """(n+1)/n stiffness correction for the elastic segment between springs.

    With a rotational spring of stiffness Ke = n * (6EI/L) at each end, the
    elastic element between them must use I * (n+1)/n so the composite
    spring-element-spring assembly reproduces the real member stiffness
    (Ibarra & Krawinkler 2005; Zareian & Medina 2010). Without it the member
    is softer than intended by exactly that factor.

    Only the member-stiffness calibration carries this correction. The
    yield-rotation mode sets Ke from My/theta_y, which is not referenced to
    6EI/L, so no correction applies.
    """
    mode = getattr(sp, "IMK_HINGE_STIFFNESS_MODE", "yield_rotation")
    if mode != "member_stiffness_factor":
        return 1.0

    n = float(sp.IMK_HINGE_STIFFNESS_FACTOR)
    if n <= 0.0:
        raise ValueError("IMK_HINGE_STIFFNESS_FACTOR must be positive.")
    return (n + 1.0) / n


def _orientation(member_type):
    if member_type == "column":
        return (0.0, 0.0, 1.0, 1.0, 0.0, 0.0), (1, 2, 3, 6)

    if member_type == "beam_x":
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1, 2, 3, 4)

    if member_type == "beam_y":
        return (0.0, 1.0, 0.0, -1.0, 0.0, 0.0), (1, 2, 3, 5)

    raise ValueError(f"Unknown member_type: {member_type}")


def _define_imk_peak_material(mat_tag, elastic_stiffness, yield_moment, backbone=None,
                              yield_moment_negative=None, *, energy_calibration=None,
                              reverse_physical=False, physical_directions=("section_positive", "section_negative"),
                              verification_only=False, spring_context=None):
    """Install a member material with its explicit, modern OpenSees signature.

    The historical function name is retained for callers. Only Bilin and
    PeakOriented are member options; a joint Pinching spring requires its own
    topology and calibration. Legacy calls pass Lamda unchanged. The corrected
    path derives every Lamda from an explicit reviewed reference energy.
    """
    material_type = sp.IMK_MATERIAL_TYPE
    if material_type not in ("IMKBilin", "IMKPeakOriented"):
        raise ValueError("Member flexure requires IMKBilin or IMKPeakOriented; calibrate joint IMKPinching separately")
    if yield_moment_negative is None:
        yield_moment_negative = yield_moment

    def branch(sign, strength):
        def rotation(key):
            # Per-sign value, else the backbone's symmetric value, else the
            # global constant (a deterioration-only backbone carries no rotations).
            default = getattr(sp, f"IMK_{key.upper()}_{sign}")
            if backbone:
                return backbone.get(f"{key}_{sign.lower()}", backbone.get(key, default))
            return default
        return RotationalBackbone(rotation("theta_p"), rotation("theta_pc"), rotation("theta_u"), strength,
                                  getattr(sp, f"IMK_FMAXFY_{sign}"), getattr(sp, f"IMK_FRESFY_{sign}"))

    # Deterioration capacities: the member backbone carries them when
    # IMK_DETERIORATION_MODE translated Haselton's lambda into the OpenSees
    # convention (Lamda = lambda * theta_y,member, S and C only; A and K
    # suppressed by a large finite capacity); otherwise the IMK_LAMBDA_*
    # constants are passed unchanged. Either way Lamda is a command-level
    # value here (E_ref = Lamda * My), never multiplied by a rotation again.
    modes = active_energy_modes(material_type)
    by_mode = (backbone or {}).get("lambda_opensees_by_mode_rad")
    if by_mode is None and (backbone or {}).get("lambda_opensees_rad") is not None:
        # One translated value for every mode (the desktop's uniform form).
        by_mode = {mode: backbone["lambda_opensees_rad"] for mode in modes}
    if by_mode is not None:
        lamda = {mode: by_mode[mode] for mode in modes}
        deterioration_source = backbone.get("deterioration_source", "backbone_uniform_lambda")
        calibration_id = deterioration_source + "_v1"
    else:
        lamda = {mode: getattr(sp, f"IMK_LAMBDA_{mode}") for mode in modes}
        deterioration_source = (backbone or {}).get("deterioration_source", "direct_opensees")
        calibration_id = sp.IMK_CYCLIC_CALIBRATION_ID
    cyclic = CyclicParameters(
        lamda_s=lamda["S"], lamda_c=lamda["C"], lamda_k=lamda["K"],
        c_s=sp.IMK_C_S, c_c=sp.IMK_C_C, c_k=sp.IMK_C_K, d_pos=sp.IMK_D_POS, d_neg=sp.IMK_D_NEG,
        lamda_a=lamda.get("A"), c_a=sp.IMK_C_A if "A" in modes else None,
        energy_convention=sp.IMK_ENERGY_CONVENTION,
    )
    # Bar slip is owned per member end (Model/Deformation_Ownership): the backbone built for this
    # end carries the owner and Haselton's a_sl. No end has been given a calibrated interface, so
    # the decomposition is the legacy included scope unless a diagnostic registered one in-process.
    indicator = (backbone or {}).get("bond_slip_indicator", bond_slip_indicator())
    owner = (backbone or {}).get("slip_owner", "member_hinge")
    if indicator == 0.0 and owner == "face_interface":
        scope = ("member_end_spring; flexure only; bar slip owned by a registered face interface; "
                 "decomposition_not_certified")
    elif indicator == 0.0:
        scope = ("member_end_spring; flexure only; bar slip removed without an owner "
                 "(legacy 2026-09-27 state, diagnostic reproduction only)")
    else:
        scope = ("member_end_spring; flexure and bar slip inside the member hinge (legacy included scope); "
                 "slip_partition_not_validated")
    provenance = {"calibration_id": calibration_id,
                    "status": sp.IMK_CYCLIC_CALIBRATION_STATUS,
                    "deterioration_source": deterioration_source,
                    "backbone_source": (backbone or {}).get("source", "fixed"),
                    "deformation_scope": scope,
                    "bond_slip_indicator": indicator,
                    "slip_owner": owner,
                    "end_location": (backbone or {}).get("end_location")}
    if by_mode is not None:
        provenance["deterioration"] = {key: backbone[key] for key in backbone
                                       if key.startswith("deterioration_") or key.startswith("lambda_")
                                       or key == "energy_reference_member_theta_y_rad"}
    if spring_context is not None:
        provenance["spring_context"] = spring_context
    args = (material_type, mat_tag, elastic_stiffness, branch("POS", yield_moment),
            branch("NEG", yield_moment_negative), cyclic)
    if energy_calibration is not None:
        return define_mapped_rotational_imk(*args, calibration=energy_calibration,
            reverse=reverse_physical, physical_directions=physical_directions,
            provenance=provenance, verification_only=verification_only)
    provenance["energy_mapping_status"] = "legacy_unmapped"
    return define_rotational_imk(*args, provenance=provenance)


def _create_hinge_node(source_node, hinge_node):
    ops.node(hinge_node, *ops.nodeCoord(source_node))


def _create_end_hinge(
    ele_tag, end_id, retained_node, hinge_node, member_type, props, length, backbone=None,
    *, energy_profiles=None, reverse_physical=False, verification_only=False
):
    orient, tied_dofs = _orientation(member_type)
    ops.equalDOF(retained_node, hinge_node, *tied_dofs)

    mat_y = imk_material_tag(ele_tag, end_id, 5)
    mat_z = imk_material_tag(ele_tag, end_id, 6)
    ke_y = imk_hinge_stiffness(member_type, "rot_y", length, props=props)
    ke_z = imk_hinge_stiffness(member_type, "rot_z", length, props=props)
    global_axes = {"column": ((1, 0, 0), (0, 1, 0)),
                   "beam_x": ((0, 1, 0), (0, 0, 1)),
                   "beam_y": ((-1, 0, 0), (0, 0, 1))}[member_type]
    contexts = [{"physical_member_tag": int(ele_tag), "end": "i" if end_id == 1 else "j",
                 "member_type": member_type, "spring_local_direction": direction,
                 "global_rotation_axis": vector, "zero_length_orientation": orient,
                 "retained_joint_node": int(retained_node), "member_hinge_node": int(hinge_node),
                 "rotation_definition": "hinge-node rotation minus retained-joint rotation projected on spring axis",
                 "tied_global_dofs": tied_dofs}
                for direction, vector in zip((5, 6), global_axes)]

    # Beam hinges are asymmetric. Measured on the zeroLength springs; hogging is POSITIVE spring
    # deformation at end i and NEGATIVE at end j, for beam_x and beam_y alike.
    # Both strengths can differ between the ends. At an exterior end whose
    # slab bars are not developed at the perimeter the hinge yields without
    # them in either sign (beam_yield_moments).
    hogging = props.get("my_hogging_i" if end_id == 1 else "my_hogging_j", props.get("my_hogging", props["my"]))
    sagging = props.get("my_sagging_i" if end_id == 1 else "my_sagging_j", props.get("my_sagging", props["my"]))
    if end_id == 1:
        positive, negative = hogging, sagging
    else:
        positive, negative = sagging, hogging
    if energy_profiles is None:
        material_y = _define_imk_peak_material(mat_y, ke_y, positive, backbone, negative, spring_context=contexts[0])
        material_z = _define_imk_peak_material(mat_z, ke_z, props["mz"], backbone, spring_context=contexts[1])
    else:
        directions_y = (("hogging", "sagging") if member_type.startswith("beam")
                        else ("section_positive_y", "section_negative_y"))
        material_y = _define_imk_peak_material(mat_y, ke_y, hogging, backbone, sagging,
            energy_calibration=energy_profiles["y"], reverse_physical=reverse_physical,
            physical_directions=directions_y, verification_only=verification_only, spring_context=contexts[0])
        material_z = _define_imk_peak_material(mat_z, ke_z, props["mz"], backbone,
            energy_calibration=energy_profiles["z"], reverse_physical=reverse_physical,
            physical_directions=("section_positive_z", "section_negative_z"),
            verification_only=verification_only, spring_context=contexts[1])

    ops.element(
        "zeroLength",
        hinge_element_tag(ele_tag, end_id),
        retained_node,
        hinge_node,
        "-mat",
        mat_y,
        mat_z,
        "-dir",
        5,
        6,
        "-orient",
        *orient,
    )
    return {"y": material_y, "z": material_z}


def _member_energy_profiles(ele_tag, verification_profiles=None):
    mode = getattr(sp, "IMK_ENERGY_MAPPING_MODE", "legacy_unmapped")
    if verification_profiles is not None:
        if mode != MAPPING_VERSION:
            raise ValueError("Synthetic member fixtures require explicit corrected mapping mode")
        profiles = verification_profiles
    elif mode == "legacy_unmapped":
        return None
    elif mode == MAPPING_VERSION:
        profiles = sp.IMK_MEMBER_ENERGY_CALIBRATIONS.get(str(ele_tag))
    else:
        raise ValueError(f"Unknown IMK energy mapping mode: {mode}")
    if not isinstance(profiles, dict) or set(profiles) != {"i", "j"}:
        raise ValueError(f"Member {ele_tag} requires explicit energy profiles for ends i and j")
    validated = {}
    for end in ("i", "j"):
        if not isinstance(profiles[end], dict) or set(profiles[end]) != {"y", "z"}:
            raise ValueError(f"Member {ele_tag}/{end} requires energy profiles for axes y and z")
        validated[end] = {axis: validate_energy_calibration(profiles[end][axis], sp.IMK_MATERIAL_TYPE,
                           verification_only=verification_profiles is not None) for axis in ("y", "z")}
    return validated


def _canonical_member_reversed(n_i, n_j, member_type):
    axis = {"beam_x": 0, "beam_y": 1, "column": 2}[member_type]
    delta = [b - a for a, b in zip(ops.nodeCoord(n_i), ops.nodeCoord(n_j))]
    if abs(delta[axis]) <= 1e-9 or any(abs(d) > 1e-9 for k, d in enumerate(delta) if k != axis):
        raise ValueError("Corrected member mapping currently requires axis-aligned physical members")
    return delta[axis] < 0


def create_imk_member(ele_tag, n_i, n_j, member_type, transf_tag, *, joint_nodes=None,
                      _verification_calibrations=None, _member_scope=None):
    """One IMK member between end nodes n_i and n_j.

    ``joint_nodes`` names the physical joints the member belongs to when the
    end nodes are not the joints themselves (a beam framing into the joint's
    beam core under the scissors joint model, Model/Joint_Springs). Grid
    position, beam family and strengths are read at the joints; the hinges
    tie to the end nodes.
    """
    joint_i, joint_j = (int(joint_nodes[0]), int(joint_nodes[1])) if joint_nodes is not None else (n_i, n_j)
    # Validate all four profiles before mutating the OpenSees domain. Normal
    # builders never pass the private synthetic-fixture argument.
    energy_profiles = _member_energy_profiles(ele_tag, _verification_calibrations)
    reverse_connectivity = (_canonical_member_reversed(n_i, n_j, member_type)
                            if energy_profiles is not None else False)
    # A column's backbone depends on how hard it is being squeezed, so the
    # gravity axial load is estimated from tributary area before its hinge
    # properties are fixed. Beams carry no meaningful axial force.
    axial_kip = 0.0
    if member_type == "column":
        story_index, grid_i, grid_j = column_grid_position(joint_j if reverse_connectivity else joint_i)
        axial_kip = column_gravity_axial(story_index, grid_i, grid_j)

    family = beam_line_family(member_type, joint_i) if member_type in ("beam_x", "beam_y") else None
    props = _member_properties(member_type, axial_kip=axial_kip, family=family)
    strength_basis = None
    if member_type in ("beam_x", "beam_y"):
        hogging, sagging, strength_basis = beam_yield_moments(member_type, joint_i, joint_j)
        props.update(my_hogging=hogging, my_sagging=sagging, my=max(hogging, sagging),
                     my_hogging_i=strength_basis.get("hogging_i_kip_in", hogging),
                     my_hogging_j=strength_basis.get("hogging_j_kip_in", hogging),
                     my_sagging_i=strength_basis.get("sagging_i_kip_in", sagging),
                     my_sagging_j=strength_basis.get("sagging_j_kip_in", sagging))
    # One backbone per END: the bar-slip share of the plastic rotation is owned per end
    # (Model/Deformation_Ownership), so a first-story column's base end and joint end are
    # calibrated separately even though they share the axial load.
    from Model.Deformation_Ownership import member_end, ownership
    ends = {"i": member_end(ele_tag, member_type, "i", joint_i), "j": member_end(ele_tag, member_type, "j", joint_j)}
    backbones = {end: backbone_for_member(member_type, axial_kip=axial_kip, end=ends[end], member_scope=_member_scope)
                 for end in ("i", "j")}
    backbone = backbones["i"]
    length = _member_length(n_i, n_j)
    i_hinge_node = hinge_node_tag(ele_tag, 1)
    j_hinge_node = hinge_node_tag(ele_tag, 2)

    _create_hinge_node(n_i, i_hinge_node)
    _create_hinge_node(n_j, j_hinge_node)
    materials_i = _create_end_hinge(ele_tag, 1, n_i, i_hinge_node, member_type, props, length, backbones["i"],
        energy_profiles=None if energy_profiles is None else energy_profiles["i"],
        reverse_physical=reverse_connectivity, verification_only=_verification_calibrations is not None)
    materials_j = _create_end_hinge(ele_tag, 2, n_j, j_hinge_node, member_type, props, length, backbones["j"],
        energy_profiles=None if energy_profiles is None else energy_profiles["j"],
        reverse_physical=not reverse_connectivity, verification_only=_verification_calibrations is not None)

    ke_y = imk_hinge_stiffness(member_type, "rot_y", length, props=props)
    ke_z = imk_hinge_stiffness(member_type, "rot_z", length, props=props)
    _HINGE_REGISTRY[int(ele_tag)] = {
        "ele_tag": int(ele_tag),
        "member_type": member_type,
        "material_type": sp.IMK_MATERIAL_TYPE,
        "ke_y_kip_in_per_rad": ke_y,
        "ke_z_kip_in_per_rad": ke_z,
        "energy_mapping_mode": getattr(sp, "IMK_ENERGY_MAPPING_MODE", "legacy_unmapped"),
        "verification_only": _verification_calibrations is not None,
        "installed_materials": {"i": materials_i, "j": materials_j},
        # The element spans the two hinge nodes, so ops.eleNodes reports those
        # rather than the structural joints. Anything assembling forces at
        # joints needs the physical joints recorded here; the end nodes the
        # hinges tie to differ only under the scissors joint model.
        "node_i": joint_i,
        "node_j": joint_j,
        "end_node_i": int(n_i),
        "end_node_j": int(n_j),
        "length_in": length,
        "yield_moment_y_kip_in": props["my"],
        "yield_moment_y_hogging_kip_in": props.get("my_hogging", props["my"]),
        "yield_moment_y_hogging_i_kip_in": props.get("my_hogging_i", props.get("my_hogging", props["my"])),
        "yield_moment_y_hogging_j_kip_in": props.get("my_hogging_j", props.get("my_hogging", props["my"])),
        "yield_moment_y_sagging_kip_in": props.get("my_sagging", props["my"]),
        "yield_moment_y_sagging_i_kip_in": props.get("my_sagging_i", props.get("my_sagging", props["my"])),
        "yield_moment_y_sagging_j_kip_in": props.get("my_sagging_j", props.get("my_sagging", props["my"])),
        "exterior_slab_anchorage": (strength_basis or {}).get("exterior_anchorage"),
        "yield_moment_z_kip_in": props["mz"],
        "strength_basis": (strength_basis or {}).get("basis"),
        "beam_family": (strength_basis or {}).get("family") or props.get("family"),
        "stiffness_basis": props.get("iy_basis"),
        "flange_width_in": props.get("flange_width_in"),
        "iy_effective_in4": props["iy"],
        "theta_y_target": props["theta_y"],
        # The spring's OWN elastic limit. theta_y_target is a member-level
        # nominal (a fixed 0.004 for columns, 0.005 for beams), but the spring
        # is deliberately IMK_HINGE_STIFFNESS_FACTOR times the member
        # stiffness, so it yields at My/Ke -- 26x to 65x smaller across the
        # pilot. Recorded hinge rotation is spring rotation, so plastic
        # rotation must be measured from this, not from the member nominal.
        # With asymmetric beam strengths the spring yields first in the
        # weaker (sagging) direction; that is the rotation the yielded flag
        # must be measured against.
        "theta_y_spring_y": (
            min(props.get("my_hogging", props["my"]), props.get("my_sagging", props["my"])) / ke_y
            if ke_y > 0 else 0.0
        ),
        # Per end, where the ends differ (exterior ends without developed
        # slab bars): the spring at that end yields in its weaker direction.
        "theta_y_spring_y_i": (
            min(props.get("my_hogging_i", props.get("my_hogging", props["my"])),
                props.get("my_sagging_i", props.get("my_sagging", props["my"]))) / ke_y
            if ke_y > 0 else 0.0
        ),
        "theta_y_spring_y_j": (
            min(props.get("my_hogging_j", props.get("my_hogging", props["my"])),
                props.get("my_sagging_j", props.get("my_sagging", props["my"]))) / ke_y
            if ke_y > 0 else 0.0
        ),
        "theta_y_spring_y_hogging": (
            props.get("my_hogging", props["my"]) / ke_y if ke_y > 0 else 0.0
        ),
        "theta_y_spring_z": (
            props["mz"] / ke_z if ke_z > 0 else 0.0
        ),
        "stiffness_modifier": props["stiffness_modifier"],
        # Flat backbone keys are end i's; the two ends differ only where a face interface is
        # registered at one end (none in production), and backbone_by_end keeps both.
        **backbone,
        "backbone_by_end": backbones,
        "ends_differ": any(backbones["i"].get(k) != backbones["j"].get(k) for k in ("theta_p", "theta_pc", "theta_u")),
        "deformation_ownership": {end: ownership(ends[end], member_scope=_member_scope) for end in ("i", "j")},
    }

    inertia_factor = imk_elastic_inertia_factor()
    ops.element(
        "elasticBeamColumn",
        ele_tag,
        i_hinge_node,
        j_hinge_node,
        props["area"],
        props["e"],
        props["g"],
        props["j"],
        inertia_factor * props["iy"],
        inertia_factor * props["iz"],
        transf_tag,
    )
