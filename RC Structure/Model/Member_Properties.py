"""Member-resolved geometry, section properties, weights and masses (grouped iterative design, 2026-10-02).

Every quantity here is computed for an actual physical member, floor or joint through the canonical
resolver (Model.Member_Groups), so it is the same in the elastic design frame, the nonlinear frame, the
gravity loads, the seismic mass, the floor models, the checks and the exports. In the uniform mode each
function reduces exactly to the one-section helper of Structure_Parameters it replaces
(tests/test_member_properties.py holds the two equal); in the grouped mode there is no such helper to fall
back to.

Conventions carried over unchanged from the uniform code:
  * a column's ``h`` is its plan dimension along global X and ``b`` along global Y (the joint adapter's
    convention: the X clear span is the bay less h, the Y clear span the bay less b);
  * slab-aware weights: a column weighs its section over the story height less the slab thickness; a
    beam weighs its drop below the slab between the faces of its end joints, smeared over the centerline
    element; one whole column story is lumped at its upper floor with half of every incident beam;
  * without a selected slab (legacy bundled load mode) members weigh their full centerline rectangle and
    the seismic mass is the floor load only.

Declared for grouped designs (new, because one section used to make the question moot):
  * JOINT CORE. The joint at floor k takes the plan dimensions of the column BELOW it (story k): a
    column is cast to the floor and a smaller column above starts on top of it. Beam clear spans, face
    distances and flange clear spans at floor k use that column. Centerlines stay aligned. The
    constructible transition itself (bar paths, confinement, splice location) is a separate check
    (Design/SMRF_Transitions) and is not implied by this geometric convention.
  * FLANGE. The clear distance to the adjacent web is taken to the actual neighbouring parallel beam on
    each side, which may be of another group; the ACI 318-19 Table 6.3.2.1 overhang is evaluated per side.
"""
from __future__ import annotations

import Structure_Parameters as sp
from Model import Member_Groups as mg

JOINT_CORE_CONVENTION = "joint_core_takes_the_column_below_v1"


# ---- joints and faces ---------------------------------------------------------------------------------
def column_plan_dimensions(design):
    """(dimension along global X, dimension along global Y) of a column design: (h, b)."""
    return design.h_in, design.b_in


def joint_core_column(level, i, j):
    """The column whose section the joint at ``level`` takes: the one below it; at the base, the one above."""
    if level >= 1:
        return mg.column_at(level, i, j)
    return mg.column_at(1, i, j)


def joint_core_dimensions(level, i, j):
    """(along X, along Y) plan dimensions of the joint core at ``level``, grid (i, j)."""
    return column_plan_dimensions(joint_core_column(level, i, j).design)


def beam_axis(member):
    if member.member_type == "beam_x":
        return "x"
    if member.member_type == "beam_y":
        return "y"
    raise ValueError(f"Member {member.member_tag} is a {member.member_type}, not a beam.")


def beam_span_in(member):
    return sp.BAY_X if beam_axis(member) == "x" else sp.BAY_Y


def beam_end_joint_depths(member):
    """Joint dimension parallel to the beam at its ends i and j (inch)."""
    k, i, j = member.story_or_floor, member.grid_i, member.grid_j
    if beam_axis(member) == "x":
        return joint_core_dimensions(k, i, j)[0], joint_core_dimensions(k, i + 1, j)[0]
    return joint_core_dimensions(k, i, j)[1], joint_core_dimensions(k, i, j + 1)[1]


def beam_face_distances_in(member):
    """Distance from each end's centerline node to the joint face (half the joint depth there)."""
    depth_i, depth_j = beam_end_joint_depths(member)
    return 0.5 * depth_i, 0.5 * depth_j


def beam_clear_span_in(member):
    """Centerline span less half of each adjoining joint depth (unequal end columns allowed)."""
    face_i, face_j = beam_face_distances_in(member)
    clear = beam_span_in(member) - face_i - face_j
    if clear <= 0.0:
        raise ValueError(f"Beam {member.member_tag} has no clear span: span {beam_span_in(member)} in, faces {face_i} and {face_j} in.")
    return clear


def column_clear_height_in(member, convention="uniform_face_to_face"):
    """Story height less the depth of the deepest beam framing into the column's top joint.

    ``physical_base`` (base story only) measures from the base to the soffit of the beams at the first floor.
    Both conventions reduce to STORY_H - H_BEAM with one beam section.
    """
    if not member.is_column:
        raise ValueError(f"Member {member.member_tag} is not a column.")
    joint = mg.joint_members(member.story_or_floor, member.grid_i, member.grid_j)
    depths = [joint[key].design.h_in for key in ("beam_x_minus", "beam_x_plus", "beam_y_minus", "beam_y_plus") if joint[key] is not None]
    if convention not in ("uniform_face_to_face", "physical_base"):
        raise ValueError(f"Unknown clear-height convention {convention!r}.")
    return sp.STORY_H - max(depths)


# ---- beam flange and flexural section -----------------------------------------------------------------
def _parallel_neighbour(member, side):
    """The adjacent parallel beam on the minus or plus side at the same floor and span, or None at the perimeter."""
    k, i, j = member.story_or_floor, member.grid_i, member.grid_j
    step = -1 if side == "minus" else 1
    if beam_axis(member) == "x":
        line = j + step
        return mg.beam_at("beam_x", k, i, line) if 0 <= line <= sp.NUM_BAY_Y else None
    line = i + step
    return mg.beam_at("beam_y", k, line, j) if 0 <= line <= sp.NUM_BAY_X else None


def beam_flange_geometry(member):
    """Clear span and, per slab side, the clear distance to the adjacent web.

    Returns {"clear_span_in", "sides": [{"side", "clear_to_adjacent_web_in", "neighbour_tag"}], "slab_sides"}.
    A perimeter line has slab on one side only.
    """
    transverse = sp.BAY_Y if beam_axis(member) == "x" else sp.BAY_X
    sides = []
    for side in ("minus", "plus"):
        neighbour = _parallel_neighbour(member, side)
        if neighbour is not None:
            sides.append({"side": side, "neighbour_tag": neighbour.member_tag,
                          "clear_to_adjacent_web_in": transverse - 0.5 * member.design.b_in - 0.5 * neighbour.design.b_in})
    if not sides:
        raise ValueError(f"Beam {member.member_tag} has no slab on either side.")
    return {"clear_span_in": beam_clear_span_in(member), "sides": sides, "slab_sides": len(sides)}


def effective_flange(member, slab_thickness_in=None):
    """ACI 318-19 Table 6.3.2.1 flange of a beam: total width and the overhang on each slab side."""
    slab = sp.SLAB_THICKNESS_IN if slab_thickness_in is None else slab_thickness_in
    geometry = beam_flange_geometry(member)
    one_sided = geometry["slab_sides"] == 1
    overhangs = []
    for side in geometry["sides"]:
        limits = ((6.0 if one_sided else 8.0) * slab, side["clear_to_adjacent_web_in"] / 2.0,
                  geometry["clear_span_in"] / (12.0 if one_sided else 8.0))
        overhangs.append({"side": side["side"], "overhang_in": min(limits),
                          "clear_to_adjacent_web_in": side["clear_to_adjacent_web_in"]})
    return {"flange_width_in": member.design.b_in + sum(o["overhang_in"] for o in overhangs), "overhangs": overhangs,
            "slab_sides": geometry["slab_sides"], "clear_span_in": geometry["clear_span_in"]}


def beam_flexural_section(member):
    """Gross vertical-bending I of a beam and its basis (the member-resolved form of sp.beam_flexural_section)."""
    design = member.design
    if sp.SLAB_THICKNESS_IN is None:
        return {"iy_in4": sp.rect_iy(design.b_in, design.h_in), "flange_width_in": design.b_in, "overhang_in": 0.0,
                "overhangs": [], "basis": "rectangular web (no slab in the model)"}
    flange = effective_flange(member)
    return {"iy_in4": sp.t_section_inertia_in4(design.b_in, design.h_in, sp.SLAB_THICKNESS_IN, flange["flange_width_in"]),
            "flange_width_in": flange["flange_width_in"],
            "overhang_in": max(o["overhang_in"] for o in flange["overhangs"]),
            "overhangs": flange["overhangs"], "slab_sides": flange["slab_sides"], "clear_span_in": flange["clear_span_in"],
            "basis": "gross T/L section on the ACI 318-19 6.3.2 effective flange (R6.6.3.1.1)"}


def elastic_section(member):
    """Gross elastic properties of a member for the frame models: A, E, G, J, Iy (vertical bending), Iz.

    The stiffness modifier of ACI 318-19 Table 6.6.3.1.1(a) is returned separately; the caller applies it.
    """
    design = member.design
    e = sp.concrete_ec_ksi(design.fc_ksi)
    section = None if member.is_column else beam_flexural_section(member)
    return {"area": sp.rect_area(design.b_in, design.h_in), "e": e, "g": sp.concrete_shear_modulus_ksi(e),
            "j": sp.approx_rect_j(design.b_in, design.h_in),
            "iy": sp.rect_iy(design.b_in, design.h_in) if member.is_column else section["iy_in4"],
            "iz": sp.rect_iz(design.b_in, design.h_in),
            "stiffness_modifier": sp.section_stiffness_modifier("column" if member.is_column else "beam"),
            "flexural_section": section}


# ---- weights ------------------------------------------------------------------------------------------
def _unit_weight_kci():
    return sp.CONCRETE_UNIT_WEIGHT_KCF / 1728.0


def column_self_weight_kip_per_in(member):
    """Equivalent line weight of a column over its centerline element."""
    design = member.design
    if sp.SLAB_THICKNESS_IN is not None:
        _validate_slab_state(member)
        return _unit_weight_kci() * design.b_in * design.h_in * (sp.STORY_H - sp.SLAB_THICKNESS_IN) / sp.STORY_H
    return sp.CONCRETE_UNIT_WEIGHT_KCI * design.b_in * design.h_in


def beam_drop_weight_kip_per_in(member):
    """Physical weight per inch of the beam between its joint faces (the drop below the slab when one is selected)."""
    design = member.design
    if sp.SLAB_THICKNESS_IN is not None:
        _validate_slab_state(member)
        return _unit_weight_kci() * design.b_in * (design.h_in - sp.SLAB_THICKNESS_IN)
    return sp.CONCRETE_UNIT_WEIGHT_KCI * design.b_in * design.h_in


def beam_self_weight_kip_per_in(member):
    """Equivalent beam line weight over the centerline element (drop between faces, smeared)."""
    if sp.SLAB_THICKNESS_IN is not None:
        return beam_drop_weight_kip_per_in(member) * beam_clear_span_in(member) / beam_span_in(member)
    return beam_drop_weight_kip_per_in(member)


def member_self_weight_kip_per_in(member):
    return column_self_weight_kip_per_in(member) if member.is_column else beam_self_weight_kip_per_in(member)


def member_length_in(member):
    return sp.STORY_H if member.is_column else beam_span_in(member)


def member_self_weight_kip(member):
    return member_self_weight_kip_per_in(member) * member_length_in(member)


def _validate_slab_state(member):
    design = member.design
    if member.is_column:
        return
    if not sp.SLAB_THICKNESS_IN < design.h_in < sp.STORY_H:
        raise ValueError(f"Slab-aware loads require SLAB_THICKNESS_IN < beam depth < STORY_H (beam {member.member_tag}: "
                         f"{design.h_in} in, slab {sp.SLAB_THICKNESS_IN} in).")


def members_at_floor(floor):
    """The members whose weight belongs to floor ``floor``: the columns of story ``floor`` and the beams at it."""
    members = []
    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            members.append(mg.column_at(floor, i, j))
    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X):
            members.append(mg.beam_at("beam_x", floor, i, j))
    for j in range(sp.NUM_BAY_Y):
        for i in range(sp.NUM_BAY_X + 1):
            members.append(mg.beam_at("beam_y", floor, i, j))
    return members


def node_structural_self_weight_kip(floor, i, j):
    """One column story (the column below the node) plus half of every incident beam's accounted weight.

    A mass and tributary-axial helper, not an extra nodal gravity load: Loads.Gravity_Loads applies the
    same member weights as distributed element loads.
    """
    if not (0 <= i <= sp.NUM_BAY_X and 0 <= j <= sp.NUM_BAY_Y and 1 <= floor <= sp.NUM_FLOOR):
        raise ValueError("Node indices must lie on the structural floor grid.")
    joint = mg.joint_members(floor, i, j)
    weight = member_self_weight_kip(joint["column_below"]) + roof_extension_weight_kip(floor, i, j)
    for key in ("beam_x_minus", "beam_x_plus", "beam_y_minus", "beam_y_plus"):
        if joint[key] is not None:
            weight += 0.5 * member_self_weight_kip(joint[key])
    return weight


def roof_extension_weight_kip(floor, i, j):
    """Weight of the column extension above the roof joint of line (i, j), kip; 0 below the roof."""
    if floor != sp.NUM_FLOOR:
        return 0.0
    from Model import Roof_Extension as roof
    return roof.weight_kip(i, j)


def floor_roof_extension_weight_kip(floor):
    return sum(roof_extension_weight_kip(floor, i, j) for j in range(sp.NUM_BAY_Y + 1) for i in range(sp.NUM_BAY_X + 1))


def floor_structural_self_weight_kip(floor):
    """Self weight of the columns of story ``floor`` and of the beams at floor ``floor`` (kip); at the roof,
    also of the column extensions above the roof joints."""
    return sum(member_self_weight_kip(member) for member in members_at_floor(floor)) + floor_roof_extension_weight_kip(floor)


def floor_area_in2():
    return sp.BAY_X * sp.NUM_BAY_X * sp.BAY_Y * sp.NUM_BAY_Y


def floor_seismic_weight_kip(floor):
    """Floor weight matching exactly the mass assigned to that floor's structural grid nodes."""
    if sp.SLAB_THICKNESS_IN is None:
        return sp.total_floor_gravity_load()
    return ((sp.floor_dead_load_ksf() + sp.SEISMIC_LIVE_LOAD_FRACTION * sp.FLOOR_LIVE_LOAD_KSF) * floor_area_in2() / 144.0
            + floor_structural_self_weight_kip(floor))


def node_seismic_mass(floor, i, j):
    """Translational seismic mass (kip s2/in) at grid node (i, j) of floor ``floor``."""
    if sp.SLAB_THICKNESS_IN is None:
        return sp.node_gravity_load_kip(i, j) / sp.G
    area_weight = ((sp.floor_dead_load_ksf() + sp.SEISMIC_LIVE_LOAD_FRACTION * sp.FLOOR_LIVE_LOAD_KSF)
                   * sp.node_tributary_area_in2(i, j) / 144.0)
    return (area_weight + node_structural_self_weight_kip(floor, i, j)) / sp.G


def story_gravity_weight_kip(floor):
    """D + L and member weight of one floor, for the story stability screen."""
    return sp.total_floor_gravity_load() + floor_structural_self_weight_kip(floor)


def gravity_above_kip(story):
    """Gravity load at and above ``story`` (the floors story..roof), kip."""
    return sum(story_gravity_weight_kip(floor) for floor in range(story, sp.NUM_FLOOR + 1))


def column_gravity_axial(story, i, j):
    """Tributary gravity axial load in the column of ``story`` at (i, j), kip, compression positive.

    Every floor at or above the column's top contributes its tributary floor load and, with a selected
    slab, the tributary member weight of that floor (its own column story and half its incident beams).
    A tributary estimate for the hinge calibration, not a solved factored axial envelope.
    """
    floors = range(story, sp.NUM_FLOOR + 1)
    floor_load = sp.node_gravity_load_kip(i, j) * len(floors)
    if sp.SLAB_THICKNESS_IN is not None:
        return floor_load + sum(node_structural_self_weight_kip(floor, i, j) for floor in floors)
    return floor_load + sum(member_self_weight_kip(mg.column_at(floor, i, j)) + roof_extension_weight_kip(floor, i, j)
                            for floor in floors)


def weight_ledger():
    """Per-floor and total weights from the installed members: the one ledger gravity, mass and ELF share."""
    floors = []
    for floor in range(1, sp.NUM_FLOOR + 1):
        members = members_at_floor(floor)
        columns = sum(member_self_weight_kip(m) for m in members if m.is_column)
        beams_x = sum(member_self_weight_kip(m) for m in members if m.member_type == "beam_x")
        beams_y = sum(member_self_weight_kip(m) for m in members if m.member_type == "beam_y")
        floors.append({"floor": floor, "column_self_weight_kip": columns, "beam_x_self_weight_kip": beams_x,
                       "beam_y_self_weight_kip": beams_y, "member_self_weight_kip": columns + beams_x + beams_y,
                       "roof_column_extension_weight_kip": floor_roof_extension_weight_kip(floor),
                       "floor_gravity_load_kip": sp.total_floor_gravity_load(),
                       "seismic_weight_kip": floor_seismic_weight_kip(floor),
                       "mass_sum_kip": sum(node_seismic_mass(floor, i, j) for j in range(sp.NUM_BAY_Y + 1)
                                           for i in range(sp.NUM_BAY_X + 1)) * sp.G})
    from Model import Roof_Extension as roof
    return {"floors": floors,
            # member_self_weight is the framed members only (what the element loads apply); the column extensions
            # above the roof joints are their own item, inside the roof's seismic weight, mass and gravity total.
            "roof_column_extension": roof.declaration(),
            "total_roof_column_extension_weight_kip": sum(f["roof_column_extension_weight_kip"] for f in floors),
            "total_member_self_weight_kip": sum(f["member_self_weight_kip"] for f in floors),
            "total_seismic_weight_kip": sum(f["seismic_weight_kip"] for f in floors),
            "total_gravity_kip": sum(f["floor_gravity_load_kip"] + f["member_self_weight_kip"]
                                     + f["roof_column_extension_weight_kip"] for f in floors),
            "grouped": mg.is_grouped(), "joint_core_convention": JOINT_CORE_CONVENTION,
            "basis": ("slab over the centerline footprint; beam drops between the faces of their end joints; columns over the story "
                      "height less the slab; one column story lumped at its upper floor with half of each incident beam"
                      if sp.SLAB_THICKNESS_IN is not None else
                      "legacy bundled area dead load; full centerline member rectangles; seismic mass from the floor load only")}


def floor_load_metadata():
    """The load inventory of the installed grouped design (the member-resolved form of
    Structure_Parameters.floor_load_metadata): area loads as there, member weights from the member ledger."""
    area_sqft = floor_area_in2() / 144.0
    dead = sp.floor_dead_load_ksf()
    return {"load_model": "slab_aware_v1", "slab_thickness_in": sp.SLAB_THICKNESS_IN,
            "floor_superimposed_dead_load_ksf": sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF,
            "slab_self_weight_ksf": sp.slab_self_weight_ksf(), "floor_dead_load_ksf": dead,
            "floor_live_load_ksf": sp.FLOOR_LIVE_LOAD_KSF, "seismic_live_load_fraction": sp.SEISMIC_LIVE_LOAD_FRACTION,
            "floor_area_sqft": area_sqft, "floor_area_dead_weight_kip": dead * area_sqft,
            "total_floor_gravity_load_kip": sp.total_floor_gravity_load(),
            "member_weights": "by physical member (ledger): no one column or beam weight describes a grouped frame",
            "ledger": weight_ledger()}


# ---- quantities for the design comparison -------------------------------------------------------------
def concrete_volume_in3(member):
    """Modeled concrete volume of a member: a column over the story height, a beam's full section between its joint faces."""
    design = member.design
    if member.is_column:
        return design.b_in * design.h_in * sp.STORY_H
    return design.b_in * design.h_in * beam_clear_span_in(member)


# ---- reinforcement geometry and strengths of a resolved member ----------------------------------------
_PM_CACHE = {}


def longitudinal_cover_in(design):
    """Concrete face to the centroid of a first-layer longitudinal bar of this design."""
    return sp.longitudinal_cover_in("column" if design.is_column else "beam", design.bar_size, design.stirrup_bar_size)


def column_steel_layers(design, about="h"):
    """[(area, depth from the compression face)] of a column cage bending through its h or its b."""
    from RC_Design_Check import _design_col_steel_layers
    return _design_col_steel_layers(design, about=about)


def column_pm_diagram(design, about="h"):
    """Nominal P-M surface of a column design (Model.IMK_Calibration.column_pm_nominal_for), cached by its content."""
    from Model.IMK_Calibration import column_pm_nominal_for
    layers = column_steel_layers(design, about)
    width, depth = (design.b_in, design.h_in) if about == "h" else (design.h_in, design.b_in)
    key = (width, depth, design.fc_ksi, tuple(layers), sp.FY_KSI, sp.ES_KSI)
    if key not in _PM_CACHE:
        if len(_PM_CACHE) > 512:
            _PM_CACHE.clear()
        _PM_CACHE[key] = column_pm_nominal_for(width, depth, design.fc_ksi, layers)
    return _PM_CACHE[key]


def rc_nominal_moment_kip_in(design, about="y"):
    """Rectangular singly reinforced estimate with max(top, bottom) bars (Structure_Parameters.rc_nominal_moment_ksi).

    About y the section bends through its depth h; about z through its width b. The legacy symmetric
    beam strength without a slab and the weak-axis spring of every beam use it.
    """
    width, depth = (design.b_in, design.h_in) if about == "y" else (design.h_in, design.b_in)
    steel = max(design.top_bars, design.bot_bars) * design.bar_area_in2
    d = depth - longitudinal_cover_in(design)
    a = steel * sp.FY_KSI / (0.85 * design.fc_ksi * width)
    a = min(a, 0.85 * depth)
    return steel * sp.FY_KSI * (d - 0.5 * a)


def exterior_ends(member):
    """Which ends of a beam terminate at the building perimeter in the beam's own direction."""
    if beam_axis(member) == "x":
        return {"i": member.grid_i == 0, "j": member.grid_i == sp.NUM_BAY_X - 1}
    return {"i": member.grid_j == 0, "j": member.grid_j == sp.NUM_BAY_Y - 1}


def perimeter_beam(member):
    """The perimeter beam (perpendicular to this beam, at its floor) that the slab bars of this beam's flange hook into."""
    kind = "beam_y" if beam_axis(member) == "x" else "beam_x"
    return mg.beam_at(kind, member.story_or_floor, 0, 0)


def beam_strengths(member, fy_factor=1.0, slab_thickness_in=None):
    """Hogging and sagging yield strengths of one beam at each end, kip-in, with their basis.

    ``fy_factor`` 1.25 gives the probable strengths of the capacity design (18.6.5.1); ``slab_thickness_in``
    overrides the slab the flange is taken from (the capacity design passes 0 while no slab layout is
    established, as Design.SMRF_Capacity_Design.probable_beam_strengths does).

    The member-resolved form of Model.IMK_Hinges.beam_yield_moments and of one family of
    Design.SMRF_Beam_Slab_Strength.beam_slab_strengths: this beam's own section and bars, its rows from
    the group bar layering, its flange from its actual end columns and neighbouring webs, the slab mats
    of the common layout inside that flange, and the perimeter beam its exterior end hooks into.
    """
    from Design.SMRF_Beam_Slab_Strength import composite_beam_strengths, perimeter_slab_bar_anchorage
    from Model import Member_Bar_Layers as layers
    design, axis, position = member.design, beam_axis(member), member.location_class
    if sp.SLAB_THICKNESS_IN is None:
        symmetric = rc_nominal_moment_kip_in(design, "y")
        return {"hogging": symmetric, "sagging": symmetric, "family": None,
                "basis": {"basis": "legacy symmetric max(top, bottom) bars", "family": None}}
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout")
    rows = layers.member_rows(member)
    beam = {"b_in": design.b_in, "h_in": design.h_in, "fc_ksi": design.fc_ksi, "fy_ksi": fy_factor * sp.FY_KSI,
            "bar_size": design.bar_size, "top_bars": design.top_bars, "bot_bars": design.bot_bars,
            "centroid_offset_in": {face: rows[face]["centroid_in"] for face in ("top", "bottom")}, "layers": rows}
    thickness = sp.SLAB_THICKNESS_IN if slab_thickness_in is None else slab_thickness_in
    flange = effective_flange(member, thickness)
    family = composite_beam_strengths(beam, {"thickness_in": thickness}, layout, None, axis, position, flange=flange)
    edge = perimeter_beam(member)
    anchorage = perimeter_slab_bar_anchorage(layout, axis, edge.design.b_in, sp.BEAM_CLEAR_COVER_IN,
                                             sp.rebar_diameter(edge.design.stirrup_bar_size), edge.design.fc_ksi, sp.FY_KSI)
    if anchorage is not None:
        anchorage = {**anchorage, "perimeter_beam_group": edge.group_id}
    hogging, sagging = family["mn_negative_kip_in"], family["mn_positive_kip_in"]
    undeveloped = layout is not None and anchorage is not None and not anchorage["developed"]
    exterior = exterior_ends(member)
    hogging_ends = {end: (family["mn_undeveloped_negative_kip_in"] if (undeveloped and exterior[end]) else hogging)
                    for end in ("i", "j")}
    sagging_ends = {end: (family["mn_undeveloped_positive_kip_in"] if (undeveloped and exterior[end]) else sagging)
                    for end in ("i", "j")}
    return {"hogging": hogging, "sagging": sagging, "family": family, "flange": flange, "bar_rows": rows,
            "basis": {
                "basis": "beam plus developed slab mats in the ACI 6.3.2 flange" if layout is not None
                         else ("rectangular beam in hogging; the chosen slab's flange concrete in compression under sagging "
                               "with no slab mats counted (slab reinforcement not established)"),
                "family": f"{axis}_{position}", "group_id": member.group_id,
                "effective_flange_width_in": family["effective_flange_width_in"],
                "slab_steel_in_flange_in2": family["slab_steel_in_flange_in2"],
                "hogging_i_kip_in": hogging_ends["i"], "hogging_j_kip_in": hogging_ends["j"],
                "sagging_i_kip_in": sagging_ends["i"], "sagging_j_kip_in": sagging_ends["j"],
                "exterior_ends": exterior, "exterior_anchorage": anchorage}}
