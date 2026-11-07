"""Capacity design of a grouped design: every beam group, every column group, every joint (2026-10-02).

The uniform capacity design (Design.SMRF_Capacity_Design.build_capacity_design) prices one beam on four
line families, one column on every story and four joint kinds. Here the same provisions are evaluated on
what is actually installed:

  beams     each member's probable strengths on its own section, bar rows and flange, its own clear span
            between the faces of its own end columns, and the face reactions of its own floor's
            transfer; one hoop per beam group from the governing member (select_beam_hoops);
  columns   each column group through design_column_shear with its own section and cage, the stories of
            its band, the clear heights between the beams at its own joints and the factored axial and
            shear envelopes of its own members; the column-own method only (an equal split of beam
            moments between unlike columns is not equilibrium);
  joints    built from the real incident members (Model.Member_Groups.joint_members): joint shear with
            the tension of the beams on each face, Aj and f'c of the joint core (the column below), and
            the Table 18.8.4.3 inputs from the transition at that joint; strong-column / weak-beam with
            the column below and the column above each on its own section at its own factored axial
            loads, one column at the roof, where the 18.7.3.1 exception is decided per connection on its
            joint-face axial load (2026-11-01); hooked and through-bar anchorage against
            the core; beam bars against the core column's lanes; the column transition itself
            (Design.SMRF_Transitions).

Checks carry the group or joint they belong to in ``location``. Joints that share the same incident
groups share one joint-shear, anchorage and threading evaluation, listed with every joint it stands for;
strong-column checks are per physical joint because each column has its own axial envelope.

Evidence corrections of 2026-10-30 (code-check audit of the genetic search): the strong-column beam
strengths credit the slab mats as developed only where the slab record's own development check passes
(SMRF_Slab_Reinforcement.slab_strength_eligibility), and every unevaluated member names its upstream
cause; the solved actions are validated (unique, succeeded, joint-face) before either the columns or the
beams read them; each frame beam group's signed axial envelope is checked against ACI 318-19 18.6.4.7;
each group's splice records its permitted interval and proposed station, Grade 80 Type 2 mechanical
splices outside the 18.2.7.2 exclusion zones.
"""
from __future__ import annotations

import math

import Structure_Parameters as sp
from Design import SMRF_Capacity_Design as cd
from Design.SMRF_Beam_Slab_Strength import CageArrangementError
from Design.SMRF_Common import SectionAxialDomainError, action_inventory_problems, make_check, not_evaluated
from Design.SMRF_Joint_Adapter import nominal_rectangular_capacity
from Design.SMRF_Joints import (MPR_BASIS, JOINT_SHEAR_BASES, JOINT_SHEAR_PHI,
                                beam_capacity_shear_envelope, rectangular_joint_area)
from Design.SMRF_Slab_Reinforcement import slab_strength_eligibility
from Design.SMRF_Transitions import column_transition
from Loads import Perimeter_Strip as perimeter_strip
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg
from Model import Member_Properties as mp
from RC_Design_Check import _design_col_steel_layers

METHOD_VERSION = "aci318_19_group_capacity_design_v6_common_torsion_readback"
# The beam axial applicability and the splice placement are the shared ones of Design.SMRF_Capacity_Design.
splice_placement, splice_location_check, beam_axial_check = cd.splice_placement, cd.splice_location_check, cd.beam_axial_check
GRAVITY_FAMILIES = (("seismic_high_gravity", lambda sds: 1.2 + 0.2 * sds, 1.0),
                    ("seismic_low_gravity", lambda sds: 0.9 - 0.2 * sds, 0.0))


# Joint shear rules by joint direction (2026-10-16). A joint direction belongs to the SFRS when the column is part
# of a perimeter frame in that direction (every direction of every joint under the band policy); the other
# directions (interior joints; the gravity-beam direction of an edge column) are members not designated as part
# of the SFRS and their joints follow Chapter 15 (ACI 318-19 18.14.3.2 / 18.14.3.3, Chapter 15).
JOINT_RULE_SFRS = "18.8"
JOINT_RULE_GRAVITY = "chapter_15"
JOINT_RULE_CLAUSE = {JOINT_RULE_SFRS: "ACI 318-19 18.8.4",
                     JOINT_RULE_GRAVITY: "ACI 318-19 15.4.2 (15.4.1.1(b) nominal beam strengths; Table 15.4.2.3; Table 21.2.1(b))"}


def joint_rule(entry, axis, state=None):
    """Which joint shear rule a joint direction follows: 18.8 for the SFRS, Chapter 15 outside it."""
    state = state or mg.active()
    if state is None or not mg.is_system_policy(state.policy_id):
        return JOINT_RULE_SFRS
    directions = mg.column_sfrs_directions(entry["grid_i"], entry["grid_j"], sp.NUM_BAY_X, sp.NUM_BAY_Y,
                                           layout=mg.layout_of(state.policy_id))
    return JOINT_RULE_SFRS if axis in directions else JOINT_RULE_GRAVITY


def _layout():
    return (sp.SLAB_REINFORCEMENT or {}).get("layout") if sp.SLAB_THICKNESS_IN is not None else None


# ---- beams ------------------------------------------------------------------------------------------------
def beam_probable_strengths(member):
    """Nominal and probable (1.25 fy, phi = 1) strengths of one beam, with the steel that is in tension."""
    layout = _layout()
    thickness = sp.SLAB_THICKNESS_IN if layout is not None else 0.0
    nominal = mp.beam_strengths(member, 1.0, thickness)
    probable = mp.beam_strengths(member, cd.PROBABLE_FY_FACTOR, thickness)
    design = member.design
    ab = sp.rebar_area(design.bar_size)
    from Design.SMRF_Conventional_Floor import side_bar_strength_increment
    side_n, side_pr = side_bar_strength_increment(member, 1.0), side_bar_strength_increment(member, cd.PROBABLE_FY_FACTOR)
    return {"member_tag": member.member_tag, "group_id": member.group_id, "axis": mp.beam_axis(member),
            "position": member.location_class,
            "mn_positive_kip_in": nominal["sagging"] + side_n, "mn_negative_kip_in": nominal["hogging"] + side_n,
            "mpr_positive_kip_in": probable["sagging"] + side_pr, "mpr_negative_kip_in": probable["hogging"] + side_pr,
            "effective_flange_width_in": probable["family"]["effective_flange_width_in"],
            "tension_steel_hogging_in2": design.top_bars * ab + probable["family"]["slab_steel_in_flange_in2"],
            "tension_steel_sagging_in2": design.bot_bars * ab,
            "clear_span_in": mp.beam_clear_span_in(member), "mpr_basis": MPR_BASIS}


def _span_face_reactions(entry, span_in, face_i_in, face_j_in, factor):
    """Upward joint-face reactions of one span's transfer loads, zero end moments, unequal faces allowed."""
    clear = span_in - face_i_in - face_j_in
    left = right = 0.0
    for x, p in entry["node_loads"]:
        a = x * span_in - face_i_in
        if 0.0 < a < clear:
            left += factor * p * (1.0 - a / clear)
            right += factor * p * a / clear
    for x, _torsion, couple in entry.get("node_couples", []):
        a = x * span_in - face_i_in
        if 0.0 < a < clear:
            left -= factor * couple / clear
            right += factor * couple / clear
    return left, right


def member_face_reactions(transfer, member, factor_dead, factor_live, strips=None):
    """Factored gravity face reactions [left, right] of one beam from its own floor's transfer.

    Dead once, live as the envelope per end over the all-panel case and every saved pattern, plus the
    beam's own drop weight on its own clear span. A perimeter beam also carries, in every unit case, the
    perimeter strip of the physical floor on its clear span: the line load the frame receives
    (Loads.Perimeter_Strip, 2026-11-05); the transfer is the centreline plate's alone. ``strips`` is the
    caller's {floor: strip pieces}: a floor's footprint is built once and reused over the beams and families of
    one capacity build; without it the pieces are built once in this call (2026-11-06).
    """
    if transfer is not None and 'physical_gravity_member_actions' in transfer:
        from Design.SMRF_Physical_Member_Actions import physical_face_reactions
        return physical_face_reactions(transfer,member,factor_dead,factor_live)
    axis = mp.beam_axis(member)
    key = (axis, member.grid_j, member.grid_i) if axis == "x" else (axis, member.grid_i, member.grid_j)
    span = mp.beam_span_in(member)
    face_i, face_j = mp.beam_face_distances_in(member)
    self_weight = factor_dead * mp.beam_drop_weight_kip_per_in(member) * (span - face_i - face_j) / 2.0
    cases = (transfer or {}).get("unit_cases", {}) if transfer else {}
    if not cases:
        return [self_weight, self_weight], "member self-weight only; no slab transfer supplied"
    dead, live = [0.0, 0.0], None
    strip_loaded = False
    strip = perimeter_strip.beam_floor_pieces(member.story_or_floor, key, strips)
    for name, case in cases.items():
        factor = factor_dead if name == "dead" else factor_live
        if factor == 0.0 or not case:
            continue
        entry = next((b for b in case["beams"] if (b["axis"], b["line_index"], b["span_index"]) == key), None)
        if entry is None:
            raise ValueError(f"Floor transfer case {name!r} has no loads for beam {member.member_tag} {key}.")
        left, right = _span_face_reactions(entry, span, face_i, face_j, factor)
        strip_end = (factor * perimeter_strip.line_load_kip_per_in(strip, transfer, key, name)
                     * (span - face_i - face_j) / 2.0)
        if strip_end:
            strip_loaded = True
            left, right = left + strip_end, right + strip_end
        if name == "dead":
            dead = [left, right]
        else:
            live = [left, right] if live is None else [max(live[0], left), max(live[1], right)]
    live = live or [0.0, 0.0]
    return ([dead[0] + live[0] + self_weight, dead[1] + live[1] + self_weight],
            f"joint-face free body over this beam's own clear span (faces {face_i:g} and {face_j:g} in); live enveloped over "
            f"{len([n for n in cases if n != 'dead'])} unit case(s) of its own floor's transfer; plus its own drop weight"
            + (perimeter_strip.SHEAR_BASIS_CLAUSE if strip_loaded else ""))


def is_gravity_beam_group(state, gid):
    """A beam group outside the seismic force-resisting system: only under a system policy (the band policy keeps
    every member in the frame whatever its plan location says)."""
    return (mg.is_system_policy(state.policy_id) and not state.designs[gid].is_column
            and state.system_of(gid) == mg.SYSTEM_GRAVITY)


def beam_group_capacity(gid, state, transfers, sds, strips=None):
    """Capacity shear of every member of one beam group and the group's transverse steel: hinge-zone hoops for a
    perimeter-frame group, stirrups throughout at <= d/2 for a gravity-system group (18.14.3.2(a), 2026-10-18).
    ``strips`` is the {floor: perimeter strip pieces} store one capacity build shares between its groups."""
    strips = {} if strips is None else strips
    transverse_rule = cd.BEAM_TRANSVERSE_GRAVITY if is_gravity_beam_group(state, gid) else cd.BEAM_TRANSVERSE_SFRS
    design = state.designs[gid]
    rows = bar_layers.group_rows(gid)
    d = design.h_in - max(rows["top"]["centroid_in"], rows["bottom"]["centroid_in"])
    members, worst_ve, worst_mechanism = [], 0.0, 0.0
    for tag in state.groups[gid]["member_tags"]:
        member = state.member(tag)
        strengths = beam_probable_strengths(member)
        transfer = None if transfers is None else transfers if 'physical_gravity_member_actions' in transfers else transfers[member.story_or_floor]
        for family_name, dead_of, live in GRAVITY_FAMILIES:
            dead = dead_of(sds)
            reactions, basis = member_face_reactions(transfer, member, dead, live, strips)
            data = {"id": f"{gid}/member_{tag}/{family_name}", "member_tag": tag, "group_id": gid,
                    "clear_span_in": strengths["clear_span_in"],
                    "mpr_left_positive_kip_in": strengths["mpr_positive_kip_in"], "mpr_left_negative_kip_in": strengths["mpr_negative_kip_in"],
                    "mpr_right_positive_kip_in": strengths["mpr_positive_kip_in"], "mpr_right_negative_kip_in": strengths["mpr_negative_kip_in"],
                    "gravity_reactions_kip": reactions, "mpr_basis": MPR_BASIS,
                    "gravity_basis": "factored_zero_end_moment_reactions", "gravity_reaction_basis": basis,
                    "gravity_family": family_name, "dead_factor": dead, "live_factor": live}
            envelope = beam_capacity_shear_envelope(data)
            data["envelope"] = envelope
            data["governing_ve_kip"] = max(envelope["left_required_kip"], envelope["right_required_kip"])
            members.append(data)
            worst_ve = max(worst_ve, data["governing_ve_kip"])
            worst_mechanism = max(worst_mechanism, envelope["mechanism_shear_positive_kip"], envelope["mechanism_shear_negative_kip"])
    result = {"group_id": gid, "members": members, "effective_depth_in": d,
              **cd.select_beam_hoops(design.b_in, d, design.fc_ksi, sp.FYT_KSI, design.bar_size, sp.BEAM_CLEAR_COVER_IN,
                                     design.top_bars, design.bot_bars, worst_ve, worst_mechanism, transverse_rule=transverse_rule,
                                     torsion_at_per_in=float((((sp.BEAM_TORSION_DESIGN or {}).get(gid) or {}).get("at_per_in2")) or 0.0),
                                     torsion_spacing_limit_in=((sp.BEAM_TORSION_DESIGN or {}).get(gid) or {}).get("spacing_limit_in"))}
    from Design import SMRF_Conventional_Detail as conventional
    if not conventional.active():
        from Design.SMRF_Actual_Hoop_Selection import beam_selection
        from Design.SMRF_Splice_Physical import group_beam_lanes
        result=beam_selection(design,group_beam_lanes(state)[gid]["coordinates"],sp.BEAM_CLEAR_COVER_IN,result)
    selected = result["hoops"]
    for data in members:
        data["phi_vn_left_kip"] = selected["phi_vn_kip"] if selected else 0.0
        data["phi_vn_right_kip"] = selected["phi_vn_kip"] if selected else 0.0
        data["shear_capacity_requirements_checked"] = bool(selected) and result["section_adequate"]
        data["hoops"] = selected
    # The evidence rows the joint checks read: the governing member of each gravity family.
    result["governing_members"] = [max((m for m in members if m["gravity_family"] == name), key=lambda m: m["governing_ve_kip"])
                                   for name, _dead, _live in GRAVITY_FAMILIES]
    return result


def beam_axial_envelope(gid, state, cases, expected_ids=None):
    """The signed factored axial envelope of one beam group's members over ``cases`` (SMRF_Capacity_Design
    .beam_axial_envelope on the group's own section and member tags, against the expected strength combinations)."""
    design = state.designs[gid]
    return cd.beam_axial_envelope(cases, state.groups[gid]["member_tags"], design.b_in, design.h_in, design.fc_ksi, gid,
                                  expected_ids=expected_ids)


# ---- columns ----------------------------------------------------------------------------------------------
def column_clear_heights(member):
    """{"face", "physical"} clear height of one column between the beams at its own two joints.

    ``physical`` is base-to-soffit at the base story and soffit-to-top-of-beam elsewhere (half the deepest
    beam at each joint, measured from the centerline nodes); ``face`` keeps the uniform code's
    face-to-face convention, which at the base story also takes off half a beam depth at the base.
    """
    from Design.Group_Checks import joint_beam_depth_in
    k, i, j = member.story_or_floor, member.grid_i, member.grid_j
    top, bottom = joint_beam_depth_in(k, i, j), joint_beam_depth_in(k - 1, i, j)
    physical = sp.STORY_H - 0.5 * top - 0.5 * bottom
    face = physical if k > 1 else sp.STORY_H - top
    return {"face": face, "physical": physical}


def column_group_capacity(gid, state, actions, cfg):
    """Capacity shear, confinement and hoops of one column group on its own members' envelopes."""
    design, group = state.designs[gid], state.groups[gid]
    tags = group["member_tags"]
    filtered = [{"id": a["id"], "analysis_succeeded": a.get("analysis_succeeded"),
                 "members": {str(tag): a["members"][str(tag)] for tag in tags}} for a in actions]
    envelopes = cd.column_action_envelopes(filtered, (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1))
    heights = {}
    for tag in tags:
        member = state.member(tag)
        value = column_clear_heights(member)
        current = heights.setdefault(member.story_or_floor, value)
        heights[member.story_or_floor] = {key: min(current[key], value[key]) for key in value}
    policy = getattr(cfg, "capacity", None)
    local = {
        "geometry": {"story_h_in": sp.STORY_H, "num_floor": sp.NUM_FLOOR},
        "sections": {"b_col_in": design.b_in, "h_col_in": design.h_in, "fc_col_ksi": design.fc_ksi},
        "materials": {"fy_ksi": sp.FY_KSI, "fyt_ksi": sp.FYT_KSI, "es_ksi": sp.ES_KSI, "slab_fy_ksi": sp.SLAB_FY_KSI},
        "column": {"bar_size": design.bar_size, "top_bars": design.top_bars, "bot_bars": design.bot_bars,
                   "side_bars": design.side_bars, "centroid_offset_in": mp.longitudinal_cover_in(design),
                   "clear_cover_in": sp.COL_CLEAR_COVER_IN, "stirrup_bar_size": design.stirrup_bar_size,
                   "layers": _design_col_steel_layers(design, about="h"),
                   "layers_about_z": _design_col_steel_layers(design, about="b")},
        "column_axial_envelope": envelopes["axial"], "column_shear_demand": envelopes["shear"],
        "column_action_envelopes": envelopes["detail"],
        "column_shear_method": cd.COLUMN_SHEAR_METHOD_COLUMN_OWN,
        # 18.7.5.2(f): the high-axial Pu is the largest compression of the combinations including E (2026-11-07).
        "column_seismic_axial_max_kip": cd.seismic_column_compression_kip(actions, tags),
        "column_clear_height_convention": getattr(policy, "column_clear_height_convention", None),
        "stories": sorted(group["stories_or_floors"]), "clear_heights_in": heights,
    }
    declared = getattr(policy, "column_shear_method", None)
    if declared not in (None, cd.COLUMN_SHEAR_METHOD_COLUMN_OWN):
        raise mg.GroupedStateError(f"Column shear method {declared!r} is not defined for a grouped design: it splits the beam "
                                   "moments equally between the columns above and below a joint. Use "
                                   f"{cd.COLUMN_SHEAR_METHOD_COLUMN_OWN!r}.")
    try:
        result = cd.design_column_shear(local, None)
        from Design import SMRF_Conventional_Detail as conventional
        if not conventional.active():
            from Design.SMRF_Actual_Hoop_Selection import column_selection
            result=column_selection(design,sp.COL_CLEAR_COVER_IN,result,mp.longitudinal_cover_in(design))
    except SectionAxialDomainError as exc:
        # The factored axial range of this group's own members leaves its section's strength domain: the group
        # that owns the section is named so the search can reject the candidate for the right reason.
        exc.group_id = gid
        raise
    result["group_id"] = gid
    result["member_count"] = len(tags)
    result["envelope_basis"] = "factored joint-face axial loads and shears of this group's own members in every combination"
    return result


# ---- joints -----------------------------------------------------------------------------------------------
def _beams(joint, axis):
    return [joint[f"beam_{axis}_{side}"] for side in ("minus", "plus") if joint[f"beam_{axis}_{side}"] is not None]


def joint_inventory(state=None):
    """Every elevated joint with its real members, and the distinct joint types by incident groups.

    Returns {"joints": [...], "types": {type_id: {...}}}. A type is (joint core group, column above group
    or the roof, X beam group and count, Y beam group and count); its id is the first joint that has it.
    """
    state = state or mg.active()
    from Model.Floor_Footprint import physical_floor
    footprints={floor:physical_floor(floor,state) for floor in range(1,sp.NUM_FLOOR+1)}
    mat_layers=((sp.SLAB_REINFORCEMENT or {}).get("layout") or {}).get("layers",{})
    joints, types = [], {}
    for level in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joint = mg.joint_members(level, i, j)
                core, above = mp.joint_core_column(level, i, j), joint["column_above"]
                beams = {axis: _beams(joint, axis) for axis in ("x", "y")}
                mixed = {}
                for axis in ("x", "y"):
                    groups = sorted({b.group_id for b in beams[axis]})
                    mixed[axis] = len(groups) > 1
                    if mixed[axis] and mg.layout_of(state.policy_id) != mg.SFRS_LAYOUT_PERIMETER_CORE:
                        raise mg.GroupedStateError(f"Joint (level {level}, {i}, {j}): the two {axis.upper()} beams belong to "
                                                   "different groups; no along-line reinforcement transition is declared.")
                # A core corner of the perimeter-core layout is the one declared along-line transition: the beams of
                # both groups terminate in the joint with standard hooks (18.8.2.2); the type keys both groups.
                # Physical equivalence also needs hook direction and local
                # phase of every globally continuous mat. Incident groups
                # alone conflate opposite corners and noncommensurate bays.
                local_phases=tuple((name,round(((j*sp.BAY_Y if name.startswith("x_") else i*sp.BAY_X)
                                                %float(layer["spacing_in"])),9))
                                   for name,layer in sorted(mat_layers.items()) if layer.get("spacing_in",0)>0)
                footprint=footprints[level]
                physical_key=((i==0,i==sp.NUM_BAY_X,j==0,j==sp.NUM_BAY_Y),local_phases,
                              tuple(footprint[k] for k in ("x_min_in","x_max_in","y_min_in","y_max_in")))
                key = (core.group_id, None if above is None else above.group_id,
                       tuple(sorted({b.group_id for b in beams["x"]})) if beams["x"] else None, len(beams["x"]),
                       tuple(sorted({b.group_id for b in beams["y"]})) if beams["y"] else None, len(beams["y"]),physical_key)
                node = level * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1) + j * (sp.NUM_BAY_X + 1) + i + 1
                entry = {"id": f"joint_{node}", "node_tag": node, "level": level, "grid_i": i, "grid_j": j,
                         "is_roof": above is None, "core": core, "above": above, "below": joint["column_below"],
                         "beams": beams, "joint": joint, "mixed": mixed}
                kind = types.setdefault(key, {"type_id": f"joint_{node}", "key": key, "representative": entry, "joints": []})
                kind["joints"].append(entry["id"])
                entry["type_id"] = kind["type_id"]
                joints.append(entry)
    return {"joints": joints, "types": {kind["type_id"]: kind for kind in types.values()}}


def joint_transition(entry, column_hoops=None, column_axial=None):
    """The column transition at one joint (None at the roof), with the lower group's installed hoop spacing."""
    core, above = entry["core"], entry["above"]
    if above is None:
        return None
    upper_clear = column_clear_heights(above)["physical"]
    depths = [b.design.h_in for axis in ("x", "y") for b in entry["beams"][axis]]
    kwargs=dict(fy_ksi=sp.FY_KSI,clear_cover_in=sp.COL_CLEAR_COVER_IN,upper_clear_height_in=upper_clear,
                joint_depth_in=min(depths),lower_hoop_spacing_in=(column_hoops or {}).get(core.group_id))
    from Design import SMRF_Conventional_Detail as conventional
    if conventional.active():
        return column_transition(core.design,above.design,**kwargs,physical_context={
            "conventional":True,"aggregate_max_size_in":sp.AGGREGATE_MAX_SIZE_IN,
            "longitudinal_specification":sp.REINFORCEMENT_SPECIFICATION})
    preliminary=column_transition(core.design,above.design,**kwargs)
    from Design.SMRF_Transition_Context import beam_context
    context=beam_context(entry,preliminary,column_hoops)
    values=[(column_axial or {}).get(m.group_id) for m in (core,above)]
    context["axial_max_kip"]=max(values) if all(v is not None for v in values) else None
    return column_transition(core.design,above.design,**kwargs,physical_context=context)


def joint_shear(entry, axis, strengths, transition, beam_hoops, rule=JOINT_RULE_SFRS):
    """Vj against phi Vn for one joint and direction, from its real incident members.

    ``rule`` JOINT_RULE_SFRS: 18.8.4, beam steel at 1.25 fy and the column shear from the probable moments.
    JOINT_RULE_GRAVITY (a joint direction outside the seismic force-resisting system under the perimeter-frame
    basis): Chapter 15, 15.4.1.1(b), beam steel at fy and the column shear from the nominal moments; the
    strength row is the Table 18.8.4.3 classification of the same joint, at or below the Table 15.4.2.3 value
    in every row, so the Chapter 15 nominal strength is conservative. The strength reduction factor is
    0.75 for Chapter 15 (15.4.2.2 / Table 21.2.1(b)), 0.85 for SMRF joints (21.2.4.4).
    """
    phi = JOINT_SHEAR_PHI[rule]  # Reject unknown rules instead of treating them as gravity directions.
    core, beams = entry["core"].design, entry["beams"][axis]
    other = "y" if axis == "x" else "x"
    transverse = entry["beams"][other]
    depth, width = (core.h_in, core.b_in) if axis == "x" else (core.b_in, core.h_in)
    fy = sp.FY_KSI
    level = "roof" if entry["is_roof"] else "floor"
    steel_factor = cd.PROBABLE_FY_FACTOR if rule == JOINT_RULE_SFRS else 1.0
    moment_key = "mpr" if rule == JOINT_RULE_SFRS else "mn"

    def tension(beam, sign):
        s = strengths[beam.member_tag]
        return steel_factor * fy * (s["tension_steel_hogging_in2"] if sign == "hogging" else s["tension_steel_sagging_in2"])

    def mpr(beam, sign):
        s = strengths[beam.member_tag]
        return s[f"{moment_key}_negative_kip_in"] if sign == "hogging" else s[f"{moment_key}_positive_kip_in"]

    multiplier = 2.0 if level == "roof" else 1.0
    if len(beams) == 2:
        # Either sway: one beam hogging at its face, the other sagging.
        sways = []
        for hog, sag in ((beams[0], beams[1]), (beams[1], beams[0])):
            forces = [tension(hog, "hogging"), tension(sag, "sagging")]
            vcol = (mpr(hog, "hogging") + mpr(sag, "sagging")) / sp.STORY_H * multiplier
            sways.append((abs(sum(forces) - vcol), forces, vcol, hog.member_tag))
        vj, forces, vcol, hogging_beam = max(sways, key=lambda item: item[0])
    else:
        beam = beams[0]
        sign = "hogging" if tension(beam, "hogging") >= tension(beam, "sagging") else "sagging"
        forces = [max(tension(beam, "hogging"), tension(beam, "sagging"))]
        vcol = max(mpr(beam, "hogging"), mpr(beam, "sagging")) / sp.STORY_H * multiplier
        vj, hogging_beam = abs(sum(forces) - vcol), beam.member_tag if sign == "hogging" else None
    mixed = bool((entry.get("mixed") or {}).get(axis))
    bw = min(b.design.b_in for b in beams)                       # the lesser web at a mixed joint (conservative aj)
    beam_depth = max([b.design.h_in for b in beams] + [b.design.h_in for b in transverse])
    clear = min(mp.beam_clear_span_in(b) for b in beams)
    transverse_clear = min((mp.beam_clear_span_in(b) for b in transverse), default=0.0)
    above = entry["above"]
    column_clear = column_clear_heights(above)["physical"] if above is not None else 0.0
    continuity = {"beam_reinforcement_continuous_through_interior_joints": not mixed,
                  "column_reinforcement_continuous_through_floor_joints":
                      (None if transition is None else bool(transition["column_reinforcement_continuous"])),
                  "basis": (("a core corner of the perimeter-core layout: the beams of two groups terminate in the joint with "
                             "standard hooks from opposite faces (18.8.2.2), so the Table 18.8.4.3 beam input takes the 'Other' "
                             "row; " if mixed else
                             "both beams of a direction at a joint belong to one group (same bars, run through, spliced "
                             "outside the hinge zones); ")
                            + "column continuity from the declared transition at this joint "
                            f"({'roof' if transition is None else transition['kind']})")}
    transverse_design = transverse[0].design if transverse else None
    hoops = None if transverse_design is None else beam_hoops.get(transverse[0].group_id)
    classification = cd.classify_joint_shear_inputs(
        level, len(beams), len(transverse),
        transverse_design.b_in if transverse_design is not None else bw, depth, beam_depth, depth, clear, transverse_clear,
        column_clear, None if transverse_design is None else (transverse_design.top_bars, transverse_design.bot_bars),
        None if hoops is None else hoops.get("bar_size"), continuity)
    gamma = classification["gamma"]
    parallel = len(beams) if bw >= cd.CONFINING_BEAM_WIDTH_RATIO * width else 0
    perpendicular = (len(transverse) if transverse_design is not None
                     and transverse_design.b_in >= cd.CONFINING_BEAM_WIDTH_RATIO * depth else 0)
    # A reduced column's drawn confinement cannot justify using the entire
    # larger lower-column area. Use the lesser physical section dimensions;
    # retain the classification calculated above so shrinking the area cannot
    # manufacture a more favorable beam-confinement classification.
    area_depth, area_width = depth, width
    if above is not None:
        upper_depth, upper_width = ((above.design.h_in, above.design.b_in) if axis == 'x'
                                    else (above.design.b_in, above.design.h_in))
        area_depth, area_width = min(depth, upper_depth), min(width, upper_width)
    aj = rectangular_joint_area(column_depth_in=area_depth, column_width_in=area_width, beam_width_in=bw, beam_center_offset_in=0.0)
    vn = gamma * math.sqrt(core.fc_ksi * 1000.0) * aj / 1000.0
    return {"id": f"joint_shear/{entry['type_id']}/{axis}", "type_id": entry["type_id"], "level": level, "axis": axis,
            "core_group": entry["core"].group_id, "column_above_group": None if above is None else above.group_id,
            "beam_group": beams[0].group_id, "beam_groups": sorted({b.group_id for b in beams}), "mixed_groups": mixed,
            "transverse_beam_group": transverse[0].group_id if transverse else None,
            "beams_in_direction": len(beams), "beams_perpendicular": len(transverse),
            "beam_face_forces_kip": forces, "column_shear_kip": vcol, "hogging_beam_tag": hogging_beam,
            "probable_face_forces_complete": True, "column_shear_consistent_with_mpr": True,
            "nominal_vn_kip": vn, "capacity_basis": JOINT_SHEAR_BASES[rule],
            'joint_area_dimensions_in': {'parallel': area_depth, 'transverse': area_width},
            'joint_area_basis': 'lesser actual lower/upper section; original classification retained; physical confinement checked separately',
            "rule": rule, "clause": JOINT_RULE_CLAUSE[rule], "steel_stress_factor": steel_factor,
            "beam_moment_basis": "probable (1.25 fy)" if rule == JOINT_RULE_SFRS else "nominal (fy)",
            "capacity_topology_and_confinement_checked": False,
            "gamma": gamma, "gamma_basis": classification["gamma_basis"], "classification": classification,
            "column_continuity": "terminating" if level == "roof" else ("continuous" if classification["column"]["state"] == "continuous" else "other"),
            "beam_continuity": "continuous" if len(beams) == 2 else "terminating",
            "transverse_confinement": classification["confinement"]["confined"],
            "evidence_complete": classification["evidence_complete"],
            "unevaluated_evidence": classification["unevaluated_evidence"],
            "confined_faces": parallel + perpendicular, "joint_core_in": {"depth": depth, "width": width, "fc_ksi": core.fc_ksi},
            "aj_in2": aj, "vj_kip": vj, "phi": phi, "phi_vn_kip": phi * vn, "passes": vj <= phi * vn,
            "slab_steel_in_tension_included": True}


def joint_anchorage(entry, axis, column_hoop_bar):
    """Hooked anchorage of terminating beam bars (18.8.5.1) and through-bar depth (18.8.2.3) at one joint."""
    core, beams = entry["core"].design, entry["beams"][axis]
    mixed = bool((entry.get("mixed") or {}).get(axis))
    depth = core.h_in if axis == "x" else core.b_in
    # a mixed joint (core corner): the larger bar of the two groups governs the hook, and both terminate
    governing = max(beams, key=lambda b: sp.rebar_diameter(b.design.bar_size))
    db = sp.rebar_diameter(governing.design.bar_size)
    hoop_db = sp.rebar_diameter(column_hoop_bar if column_hoop_bar is not None else core.stirrup_bar_size)
    ldh = max(sp.FY_KSI * 1000.0 * db / (65.0 * math.sqrt(core.fc_ksi * 1000.0)), 8.0 * db, 6.0)
    through = len(beams) == 2 and not mixed
    # embedment to the inside of the far hoop as built (the column bars in the hoop corners on the grid)
    available = depth - (mp.longitudinal_cover_in(core) - 0.5 * sp.rebar_diameter(core.bar_size))
    required_through = max(sp.through_joint_bar_depth_factor() * db, max(b.design.h_in for b in beams) / 2.0)   # 18.8.2.3
    return {"type_id": entry["type_id"], "axis": axis, "through_bars_present": through,
            "ldh_required_in": ldh, "embedment_available_in": available, "passes": through or ldh <= available,
            "hook": ("standard 90-degree hooks of both beams into the confined core from opposite faces (18.8.2.2, 18.8.5.1): "
                     "the along-line transition at a core corner" if mixed else "standard 90-degree hook into the confined core, 18.8.5.1"),
            "through_bar_depth_required_in": required_through, "through_bar_depth_available_in": depth,
            "through_bar_passes": (required_through <= depth) if through else True,
            "beam_bar_diameter_in": db, "core_group": entry["core"].group_id, "beam_group": governing.group_id,
            "beam_groups": sorted({b.group_id for b in beams}), "mixed_groups": mixed}


def _lane_basis(inventory):
    """What the lanes of a threading row pass (Model.Member_Bar_Layers.joint_inventory of its joint)."""
    if inventory is None:
        return "the core cage"
    if not inventory["established"]:
        return ("not established: the transition to the column above declares no bar paths inside the joint, so no lane "
                f"is claimed ({bar_layers.JOINT_INVENTORY_RULE_ID})")
    return ("the joint's bar inventory: the transition moves bars inside the joint and each moved bar's whole bend "
            f"interval is kept clear ({bar_layers.JOINT_INVENTORY_RULE_ID})")


def joint_threading(entry, axis):
    """Do the beam bars of one direction pass the joint's column bars within the layer limit? The column bars are the
    joint core's cage, or, where the column above has another cage whose declared transition bends or offsets bars
    inside the joint, the joint's bar inventory with each moved bar's whole bend interval kept clear
    (Model.Member_Bar_Layers.bars_per_layer_at_joint, 2026-10-31); where that transition declares no bar paths, no lane
    is claimed and the row fails. The lanes the selection prices are the lanes this check reads; the row names both
    column groups of a transition joint (core_group, above_group)."""
    limit = sp.BEAM_BAR_MAX_LAYERS
    results = []
    inventory = bar_layers.joint_inventory(entry["level"], entry["grid_i"], entry["grid_j"])
    for beam in {b.group_id: b for b in entry["beams"][axis]}.values():        # one per group (two at a core corner)
        core_fit = bar_layers.bars_per_layer(entry["core"].design, beam.design, axis)
        fit = bar_layers.bars_per_layer_at_joint(entry["level"], entry["grid_i"], entry["grid_j"], beam.design, axis)
        n = max(beam.design.top_bars, beam.design.bot_bars)
        needed = math.ceil(n / fit) if fit and fit > 0 else None
        results.append({"type_id": entry["type_id"], "axis": axis, "bars_per_layer_that_fit": fit, "bars_per_face": n,
                        "layers_needed": needed, "max_layers": limit, "passes": needed is not None and needed <= limit,
                        "core_group": entry["core"].group_id, "beam_group": beam.group_id,
                        "above_group": None if inventory is None else inventory["above_group"],
                        "bars_per_layer_core_cage": core_fit, "lane_basis": _lane_basis(inventory)})
    worst = min(results, key=lambda r: (r["passes"], -(r["layers_needed"] or 10 ** 6)))
    return {**worst, "beam_groups": [r["beam_group"] for r in results]} if len(results) > 1 else worst


def slab_mat_clashes(entry):
    """Crossing slab mats against the top bar layers of the beams at one joint (Design.SMRF_Cage_Geometry rule)."""
    from Design.SMRF_Cage_Geometry import INTERLAYER_CLEAR_MIN_IN
    mats, placed = bar_layers.slab_mats()
    clashes, tight = [], []
    for axis in ("x", "y"):
        if not entry["beams"][axis]:
            continue
        for beam in {b.group_id: b for b in entry["beams"][axis]}.values():        # both groups at a core corner
            db = sp.rebar_diameter(beam.design.bar_size)
            for k, offset in enumerate(bar_layers.group_rows(beam.group_id)["top"]["offsets_in"]):
                for mat in mats:
                    if mat["runs_along"] == axis:
                        continue
                    gap = abs(mat["centroid_from_top_in"] - offset) - (mat["db_in"] + db) / 2.0
                    label = [mat["mat"], f"{axis} beam top bars layer {k + 1} ({beam.group_id})"]
                    if gap < 0.0:
                        clashes.append({"between": label, "overlap_in": -gap})
                    elif gap < INTERLAYER_CLEAR_MIN_IN:
                        tight.append({"between": label, "clear_in": gap})
    return {"type_id": entry["type_id"], "slab_mats_placed": placed, "clashes": clashes, "tight_crossings": tight,
            "passes": not clashes}


def _column_layers(design, axis, face):
    """[(area, depth from the compression face)] of a column bending in the plane of ``axis`` frames."""
    layers = _design_col_steel_layers(design, about="h" if axis == "x" else "b")
    if face == "positive":
        return layers
    depth = design.h_in if axis == "x" else design.b_in
    return sorted(((area, depth - d) for area, d in layers), key=lambda item: item[1])


def column_nominal_moment(design, axis, axial_kip):
    """Least nominal moment of a column over its two compression faces at one axial load, with the face."""
    width, depth = (design.b_in, design.h_in) if axis == "x" else (design.h_in, design.b_in)
    results = []
    for face in ("positive", "negative"):
        solved = nominal_rectangular_capacity(width_in=width, depth_in=depth, fc_ksi=design.fc_ksi, fy_ksi=sp.FY_KSI,
                                              es_ksi=sp.ES_KSI, layers=_column_layers(design, axis, face), axial_kip=axial_kip)
        results.append((solved["mn_kip_in"], face, solved))
        if design.top_bars == design.bot_bars:
            break                                   # a symmetric cage has the same strength on either face
    return min(results, key=lambda item: item[0])


def default_slab_eligibility():
    """The slab-credit eligibility of the slab record in force (no floor-context comparison; see
    SMRF_Slab_Reinforcement.slab_strength_eligibility), or the no-slab reason when the model has no slab."""
    if sp.SLAB_THICKNESS_IN is None:
        return {**slab_strength_eligibility(None), "reason_code": "no_slab_in_model",
                "reason": "no slab in the model: the developed beam-plus-slab strength is not defined"}
    return slab_strength_eligibility(sp.SLAB_REINFORCEMENT)


def scwb_evidence(inventory, actions, expected_ids, state=None, slab_eligibility=None):
    """Strong-column / weak-beam evidence at every joint, each column on its own section and axial loads.

    The structure SMRF_Joints.evaluate_joints reads: per joint, direction and sway sign, the capacities
    of the columns actually framing in (the column below at its top end, the column above at its bottom
    end; one column at the roof) and of the beams on each face. A column's Mn is the least over every
    solved combination at that column end's own joint-face axial load and over both compression faces;
    an axial load outside a section's domain leaves that column unevaluated with the reason (a section
    that cannot carry its axial load is rejected by the strength check, not hidden here). Beam strengths
    are the member's rectangular Mn plus the developed slab increment. At an exterior end the slab bars
    are credited where their perimeter hook develops at either datum, the support face or the joint face
    (D2, 2026-11-01): an envelope this rule alone reads. Beam sizing credits them only at both datums, and
    the hinges and the saved end strengths read the support face (Model.Member_Properties.beam_strengths);
    the datum is recorded on each beam capacity.

    At a connection where the column is discontinuous above (the roof: no column above in the framed
    inventory) each sway state carries ``connection`` evidence: the roof column's own gross area and f'c.
    SMRF_Joints.scwb_check then applies the ACI 318-19 18.7.3.1 exception there when that column's largest
    joint-face axial compression over every solved combination is below Ag f'c / 10 (2026-11-01); the
    largest over the combinations that include E is recorded beside it, never deciding. Every other check
    at the roof, and the beam and slab evidence the row needs, stay required.

    The solved actions are validated first (SMRF_Common.action_inventory_problems: unique ids, every
    expected case, succeeded, joint-face axial reference, finite end forces of every column); an invalid
    inventory leaves every column unevaluated with that reason, never a partial or overwritten envelope.
    The slab increment is credited as developed only when ``slab_eligibility`` (default: the record in
    force) establishes it; otherwise the beam carries the specific upstream reason. An exterior end whose
    perimeter hook does not fit keeps its known undeveloped strength (no slab bars, flange concrete in
    sagging), as before.
    """
    column_tags = sorted({column.member_tag for entry in inventory["joints"]
                          for column in (entry["below"], entry["above"]) if column is not None})
    by_id, action_problems = action_inventory_problems(actions, expected_ids, column_tags)
    eligibility = default_slab_eligibility() if slab_eligibility is None else slab_eligibility
    beam_cache, joints = {}, []

    def beam_strength(member):
        if member.member_tag not in beam_cache:
            beam_cache[member.member_tag] = mp.beam_strengths(member)
        return beam_cache[member.member_tag]

    for entry in inventory["joints"]:
        columns = [("below", entry["below"], "j")] + ([("above", entry["above"], "i")] if entry["above"] is not None else [])
        below = entry["below"].design
        connection = {"column_discontinuous_above": entry["above"] is None,
                      "column_gross_area_in2": below.b_in * below.h_in, "column_fc_ksi": below.fc_ksi,
                      "column_tag": entry["below"].member_tag, "column_group_id": entry["below"].group_id,
                      "basis": "framed inventory: no column above this joint (ACI 318-19 18.7.3.1)"
                               if entry["above"] is None else "a column continues above this joint"}
        record = {"id": entry["id"], "node_tag": entry["node_tag"], "floor": entry["level"], "grid_i": entry["grid_i"],
                  "grid_j": entry["grid_j"], "is_roof": entry["is_roof"], "type_id": entry["type_id"], "directions": {}}
        for axis in ("x", "y"):
            column_caps = []
            for position, column, end in columns:
                cap = {"tag": column.member_tag, "end": end, "position": position, "group_id": column.group_id,
                       "axial_envelope_checked": False}
                if action_problems:
                    cap.update(reason_code="action_inventory_invalid",
                               reason="the final factored-action inventory is not valid: " + "; ".join(action_problems[:4]))
                    column_caps.append(cap)
                    continue
                try:
                    observations = []
                    for cid in expected_ids:
                        p = float(by_id[cid]["members"][str(column.member_tag)][f"axial_{end}_kip"])
                        mn, face, solved = column_nominal_moment(column.design, axis, p)
                        observations.append({"combination_id": cid, "factored_axial_kip": p, "mn_kip_in": mn,
                                             "governing_compression_face": face})
                    governing = min(observations, key=lambda o: o["mn_kip_in"])
                    peak = max(observations, key=lambda o: o["factored_axial_kip"])
                    with_e = [_includes_earthquake(by_id[o["combination_id"]]) for o in observations]
                    seismic = [o for o, flag in zip(observations, with_e) if flag]
                    peak_e = (max(seismic, key=lambda o: o["factored_axial_kip"])
                              if seismic and None not in with_e else None)
                    cap.update(governing)
                    # The deciding Pu of 18.7.3.1 is axial_max_kip (every combination: never less than the clause's
                    # E-only value); the E-only maximum is recorded beside it, None when undeterminable.
                    cap.update(axial_max_combination_id=peak["combination_id"],
                               axial_max_e_kip=None if peak_e is None else peak_e["factored_axial_kip"],
                               axial_max_e_combination_id=None if peak_e is None else peak_e["combination_id"])
                    cap.update(axial_envelope_checked=True,
                               axial_min_kip=min(o["factored_axial_kip"] for o in observations),
                               axial_max_kip=max(o["factored_axial_kip"] for o in observations),
                               combination_count=len(observations),
                               compression_face_basis="minimum of both faces; conservative uniaxial sway envelope",
                               axial_reference="joint_faces", capacity_basis=solved["basis"])
                except (ValueError, TypeError, KeyError) as exc:
                    cap.update(reason=str(exc), reason_code=("column_axial_outside_section_domain"
                                                             if isinstance(exc, SectionAxialDomainError)
                                                             else "column_capacity_not_evaluated"))
                column_caps.append(cap)
            states = {}
            for sign in ("positive", "negative"):
                beam_caps = []
                for side, beam in (("minus", entry["joint"][f"beam_{axis}_minus"]), ("plus", entry["joint"][f"beam_{axis}_plus"])):
                    if beam is None:
                        continue
                    end = "j" if side == "minus" else "i"                     # the end of that beam at this joint
                    # Positive sway: the beam on the minus side sags at its right end, the one on the plus side hogs.
                    flexure = sign if end == "j" else ("negative" if sign == "positive" else "positive")
                    strengths = beam_strength(beam)
                    family, basis = strengths["family"], strengths["basis"]
                    cap = {"tag": beam.member_tag, "end": end, "position": f"{side}_side", "group_id": beam.group_id,
                           "flexure_sign": flexure}
                    if family is None:
                        cap.update(reason_code="no_slab_in_model",
                                   reason="no slab in the model: the developed beam-plus-slab strength is not defined")
                    else:
                        from Design.SMRF_Conventional_Floor import side_bar_strength_increment
                        rectangular = family["rectangular"][flexure]["mn_kip_in"] + side_bar_strength_increment(beam, 1.0)
                        exterior = basis["exterior_ends"][end]
                        hook = (basis.get("exterior_anchorage_by_end") or {}).get(end)
                        computed_end=((eligibility.get('member_development') or {}).get(str(beam.member_tag),{}).get('ends') or {}).get(end)
                        computed_route=eligibility.get('version')=='computed_physical_slab_detailing_and_nominal_development_v1'
                        undeveloped = (eligibility.get("layout_selected") and exterior and hook is not None
                                       and not hook["developed_either"])
                        if computed_route:
                            # Unknown nominal credit is never interpreted as zero
                            # slab steel or used to lower probable-strength demands.
                            undeveloped=False
                        cap.update(mn_kip_in=rectangular, capacity_basis=family["basis"])
                        if hook is not None:
                            cap.update(slab_hook={
                                "datum": ("support_face" if hook["developed_support_face"] else
                                          "joint_face" if hook["developed_joint_face"] else "neither"),
                                "ldh_in": hook["support_face"]["ldh_required_in"], "psi_o": hook["support_face"]["psi_o"],
                                "support_face_embedment_in": hook["support_face"]["embedment_available_in"],
                                "joint_face_embedment_in": hook["joint_face"]["embedment_available_in"],
                                "confinement_required_25_4_3_4": hook["support_face"]["confinement_required_25_4_3_4"]})
                        if not eligibility.get("layout_selected"):
                            cap.update(slab_reason_code=eligibility.get("reason_code") or "no_slab_layout",
                                       slab_reason="Developed beam-plus-slab section compatibility evidence is missing: "
                                                   + str(eligibility.get("reason")))
                        elif undeveloped:
                            # The perimeter hook does not fit: a known basis that needs no development (no slab bars in
                            # either sign; retain any flange concrete reached by the compression block).
                            total = family["undeveloped"][flexure]["mn_kip_in"]
                            increment = max(0.0, total - rectangular)
                            cap.update(slab_basis=("flange_concrete_undeveloped_bars" if increment > 1e-9
                                                   else "terminated_undeveloped"),
                                       slab_mn_kip_in=increment)
                        elif ((computed_route and not (computed_end or {}).get('developed_either')) or
                              (not computed_route and eligibility.get("credit") != "developed")):
                            cap.update(slab_reason_code=eligibility.get("reason_code") or "slab_credit_not_established",
                                       slab_reason="The slab mats are in the flange but their development is not "
                                                   "established: " + str(eligibility.get("reason")))
                        else:
                            cap.update(slab_basis="developed_effective_width",
                                       slab_mn_kip_in=max(0.0, family[f"slab_increment_{flexure}_kip_in"]))
                            if computed_route:
                                cap['computed_nominal_development']=computed_end
                                cap['slab_hook_datum_basis']='Actual full-fy per-bar evidence; either support or joint face for SCWB'
                    beam_caps.append(cap)
                states[sign] = {"nominal_strengths": True, "column_capacities": column_caps, "beam_capacities": beam_caps,
                                "connection": dict(connection)}
            # 18.7.3.2 is a special-moment-frame rule: required where the column is part of a perimeter frame in
            # this direction (every direction under the band policy); recorded, not required, elsewhere (18.14).
            states["required"] = joint_rule(entry, axis, state) == JOINT_RULE_SFRS
            states["requirement_basis"] = ("ACI 318-19 18.7.3.2: column of a special moment frame in this direction"
                                           if states["required"] else
                                           "not required: the column is not part of the seismic force-resisting system in this "
                                           "direction (ACI 318-19 18.14; perimeter-frame basis)")
            record["directions"][axis] = states
        joints.append(record)
    return {"joints": joints, "expected_combination_ids": list(expected_ids), "action_inventory_complete": not action_problems,
            "action_inventory_problems": action_problems[:12],
            # What the slab record establishes, as the qualification can recompute it from the saved record alone: the
            # floor-context comparison (context_current) belongs to the evaluation that made it, not to this evidence.
            "slab_eligibility": {key: eligibility.get(key) for key in (
                "version", "layout_selected", "slab_checks_complete", "slab_checks_passed", "interior_development",
                "perimeter_anchorage", "credit", "reason_code", "reason", "signature")},
            "axial_envelope_basis": ("minimum nominal strength of each column, on its own section, across every supplied "
                                     "factored case at its own joint-face axial load and both compression faces"),
            "uniaxial_only": True, "roof_exemption_applied": "per connection: ACI 318-19 18.7.3.1 (SMRF_Joints.scwb_check)"}


def _includes_earthquake(action):
    """Whether a solved combination includes E (its ex or ey factor is nonzero); None when the action does not carry
    the factors (Design.SMRF_Demands.strength_load_combinations writes both on every combination)."""
    if "ex" not in action and "ey" not in action:
        return None
    return bool(action.get("ex") or action.get("ey"))


# ---- the whole capacity design ------------------------------------------------------------------------------
def beam_torsion_evidence(state,expected_ids,beam_capacity,common_torsion=None,common_actions=None):
    """Required concurrent torsion row; legacy actions never imply Tu=0."""
    from Design.SMRF_Beam_Torsion_Adapter import evidence
    return evidence(state,expected_ids,beam_capacity,common_torsion,common_actions=common_actions)


def build_group_capacity_design(actions, expected_ids, cfg, transfers=None, state=None, compatibility=None,
                                slab_eligibility=None, common_torsion=None,common_actions=None):
    """Capacity-design evidence and checks of the installed grouped design.

    ``actions`` are the solved combinations (Group_Checks.capture_member_actions per combination);
    ``transfers`` {floor: validated transfer} or None; ``compatibility`` the compatibility run of
    Design.Perimeter_Systems under the system policy (the induced actions of the gravity members), None
    otherwise; ``slab_eligibility`` the slab-credit eligibility the strong-column beams read (default: the
    slab record in force, SMRF_Slab_Reinforcement.slab_strength_eligibility). Returns beams and columns by
    group, joints by type, transitions, strong-column evidence, the hoops each group needs, the gravity-frame
    evidence (system policy) and the checks.
    """
    state = state or mg.active()
    if state is None:
        raise mg.GroupedStateError("The group capacity design needs an installed grouped design.")
    # Never on known-invalid bar rows (2026-11-01): the complete shared arrangement is audited first and a state whose rows
    # do not fit is refused by name (the selection closes its terminal arrangement before this; this is the backstop).
    audit = bar_layers.row_audit(state)
    bar_layers.raise_arrangement_errors(audit)
    if not audit["valid"]:
        groups = sorted({item["group"] for item in audit["invalid"]})
        raise CageArrangementError(
            f"the capacity design needs valid bar rows; the complete arrangement leaves rows that do not fit in {groups[:6]}",
            {"outcome": "capacity_refused_invalid_rows", "group_ids": groups, "audit": audit}, {"rejected_state": state})
    sds = sp.ASCE_SDS
    from Design import SMRF_Conventional_Detail as conventional
    detail_conventional = conventional.active()
    torsion_designs = None
    if detail_conventional:
        from Design.SMRF_Conventional_Floor import group_torsion_designs, torsion_rows
        torsion_designs = group_torsion_designs(state, actions)
        sp.BEAM_TORSION_DESIGN = torsion_designs
    strips = {}                     # each floor's perimeter strip pieces, built once for this build (2026-11-06)
    beams = {gid: beam_group_capacity(gid, state, transfers, sds, strips)
             for gid, group in sorted(state.groups.items()) if group["member_type"] != "column"}
    for gid, data in beams.items():
        # Frame beams (18.6): the signed axial envelope of the strength combinations. Gravity beams keep their own
        # 18.14.3.2(a) axial branch in Design.Perimeter_Systems.gravity_frame_evidence.
        if not is_gravity_beam_group(state, gid):
            data["axial"] = beam_axial_envelope(gid, state, [("strength", a) for a in actions], expected_ids=expected_ids)
    columns = {gid: column_group_capacity(gid, state, actions, cfg)
               for gid, group in sorted(state.groups.items()) if group["member_type"] == "column"}
    beam_hoops = {gid: data["hoops"] or {} for gid, data in beams.items()}
    column_spacing = {gid: (data["hoops"] or {}).get("spacing_in") for gid, data in columns.items()}
    column_axial = {gid:max((row["axial_max_kip"] for per_story in data.get("stories",{}).values()
                            for row in per_story.values()),default=None) for gid,data in columns.items()}
    inventory = joint_inventory(state)
    strengths = {tag: beam_probable_strengths(state.member(tag))
                 for gid, group in state.groups.items() if group["member_type"] != "column" for tag in group["member_tags"]}
    joint_types, transitions = {}, {}
    for type_id, kind in sorted(inventory["types"].items()):
        entry = kind["representative"]
        transition = joint_transition(entry, column_spacing,column_axial)
        if detail_conventional:
            actual_beam_context=conventional.beam_joint_detail(entry)
        else:
            from Design.SMRF_Transition_Context import beam_context
            actual_beam_context=beam_context(entry,transition,column_spacing)
        roof_anchorage=None
        if entry["above"] is None and detail_conventional:
            core=entry["core"]
            roof_anchorage=conventional.roof_detail(core.design,extension_in=sp.roof_column_extension_in(core.design.b_in,core.design.h_in),
                joint_depth_in=min(b.design.h_in for members in entry["beams"].values() for b in members),
                fy_ksi=sp.FY_KSI,clear_cover_in=sp.COL_CLEAR_COVER_IN,
                spacing_in=column_spacing.get(core.group_id) or core.design.stirrup_spacing_in,
                normalweight=sp.CONCRETE_UNIT_WEIGHT_KCF==.150)
        elif entry["above"] is None:
            from Design.SMRF_Roof_Anchorage import verify_declared_roof
            core=entry["core"]
            roof_anchorage=verify_declared_roof(core.design,extension_in=sp.roof_column_extension_in(core.design.b_in,core.design.h_in),
                joint_depth_in=min(b.design.h_in for members in entry["beams"].values() for b in members),
                fy_ksi=sp.FY_KSI,clear_cover_in=sp.COL_CLEAR_COVER_IN,
                spacing_in=column_spacing.get(core.group_id) or core.design.stirrup_spacing_in,
                specification=sp.REINFORCEMENT_SPECIFICATION,physical_context=actual_beam_context,
                axial_max_kip=column_axial.get(core.group_id),normalweight=sp.CONCRETE_UNIT_WEIGHT_KCF==.150)
        if transition is not None and entry["above"].group_id != entry["core"].group_id:
            key = f"{entry['core'].group_id}>{entry['above'].group_id}"
            contextual = {**transition, 'type_id': type_id, 'joints': list(kind['joints'])}
            if key not in transitions:
                transitions[key] = {**contextual, 'contexts': {}}
            aggregate = transitions[key]
            aggregate['contexts'][type_id] = contextual
            aggregate['joints'] = sorted({j for item in aggregate['contexts'].values() for j in item['joints']})
            aggregate['supported'] = all(item['supported'] for item in aggregate['contexts'].values())
            aggregate['physically_verified'] = all(item.get('physically_verified', False) for item in aggregate['contexts'].values())
        shear = {axis: joint_shear(entry, axis, strengths, transition, beam_hoops, joint_rule(entry, axis, state))
                 for axis in ("x", "y") if entry["beams"][axis]}
        hoop_bar = (columns[entry["core"].group_id]["hoops"] or {}).get("bar_size")
        joint_types[type_id] = {
            "type_id": type_id, "joints": list(kind["joints"]), "level": entry["level"],
            "core_group": entry["core"].group_id, "column_above_group": None if entry["above"] is None else entry["above"].group_id,
            "transition": None if transition is None else {k: transition[k] for k in ("kind", "supported", "column_reinforcement_continuous")},
            'physical_transition': None if transition is None else transition.get('physical_detail'),
            'physical_beam_context': actual_beam_context,
            'physical_roof_anchorage': roof_anchorage,
            "shear": shear,
            "anchorage": {axis: joint_anchorage(entry, axis, hoop_bar) for axis in ("x", "y") if entry["beams"][axis]},
            "threading": {axis: joint_threading(entry, axis) for axis in ("x", "y") if entry["beams"][axis]},
            "slab_mats": slab_mat_clashes(entry),
            "joint_hoops": columns[entry["core"].group_id]["hoops"],
            "joint_hoop_basis": "18.8.3.1: the hoops of the column below (the joint core) continue through the joint depth"}
    # Joint transverse steel designed and every Table 18.8.4.3 input on evidence: only then is the nominal strength usable.
    for kind in joint_types.values():
        for item in kind["shear"].values():
            physical = kind.get('physical_transition')
            item["capacity_topology_and_confinement_checked"] = bool(kind["joint_hoops"] is not None and item["evidence_complete"]
                and (detail_conventional or (((kind.get('physical_roof_anchorage') or {}).get('status')=='pass') if kind['column_above_group'] is None
                     else (physical or {}).get('status') == 'pass')))
    scwb = scwb_evidence(inventory, actions, expected_ids, state=state, slab_eligibility=slab_eligibility)

    if detail_conventional:
        torsion={"method":"conventional_torsion_v1","designs":torsion_designs,"checks":torsion_rows(state,beams,torsion_designs)}
    else:
        torsion=beam_torsion_evidence(state,expected_ids,beams,common_torsion,common_actions)
    checks = list(torsion['checks'])
    for gid, data in beams.items():
        if detail_conventional:
            mat_inventory=[]
        else:
            from Design.SMRF_Joint_Mat_Physical import member_slab_inventory
            mat_inventory=[member_slab_inventory(state.member(tag),mp.effective_flange(state.member(tag)))
                           for tag in state.groups[gid]["member_tags"]] if _layout() is not None else []
        data["slab_physical_inventory"]=mat_inventory
        statuses=[r["status"] for r in mat_inventory]
        if detail_conventional and _layout() is not None:
            physical_row=make_check("beam.slab_physical_inventory","ACI 318-19 6.3.2;18.7.3.2;25.4",1,1,"==",location=gid)
        elif "fail" in statuses:
            physical_row=make_check("beam.slab_physical_inventory","ACI 318-19 6.3.2;18.7.3.2;25.4",0,1,"==",location=gid)
        elif not mat_inventory or any(s not in ("pass","fail") for s in statuses):
            physical_row=not_evaluated("beam.slab_physical_inventory","ACI 318-19 6.3.2;18.7.3.2;25.4",
                                      "Actual globally placed slab bars in every effective flange are not established",gid)
        else:
            physical_row=make_check("beam.slab_physical_inventory","ACI 318-19 6.3.2;18.7.3.2;25.4",
                                    int(all(s=="pass" for s in statuses)),1,"==",location=gid)
        physical_row["details"].update(member_inventories=mat_inventory)
        checks.append(physical_row)
        checks.append(make_check("beam.capacity_shear_section", "ACI 318-19 22.5.1.2 with 18.6.5.1 Ve",
                                 data["vs_required_kip"], data["vs_limit_kip"], "<=", "kip", gid))
        checks.append(make_check("beam.hoops_selected", "ACI 318-19 18.6.4.4 / 18.6.5", int(data["hoops"] is not None), 1, "==",
                                 location=gid))
        checks.append(_cage_check("beam", data["cage"], gid))
        if "axial" in data:
            checks.append(beam_axial_check(data["axial"]))
    for gid, data in columns.items():
        checks.append(make_check("column.capacity_shear_section", "ACI 318-19 22.5.1.2 with 18.7.6.1.1 Ve",
                                 data["governing"]["vs_required_kip"] if data["governing"] else 0.0, data["vs_limit_kip"],
                                 "<=", "kip", gid, details={"column_shear_method": data["column_shear_method"],
                                                           "clear_height_convention": data["clear_height_convention"]}))
        checks.append(make_check("column.hoops_selected", "ACI 318-19 18.7.5.3 / 18.7.5.4 / 18.7.6",
                                 int(data["hoops"] is not None), 1, "==", location=gid))
        checks.append(make_check("column.supported_bar_spacing", "ACI 318-19 18.7.5.2",
                                 int(data["confinement"]["hx_within_8in_when_required"]), 1, "==", location=gid))
        checks.append(_cage_check("column", data["cage"], gid))
        bond = column_group_bar_bond(gid, state)
        data["bar_bond"] = bond
        checks.append(make_check("column.bar_bond_development", bond["clause"], bond["factored_ld_in"], bond["half_clear_height_in"],
                                 "<=", "in", gid, details={key: bond[key] for key in (
                                     "bar_size", "cb", "ktr_in", "confinement_ratio", "ld_in", "clear_height_in", "clear_height_basis")}))
    for key, transition in sorted(transitions.items()):
        checks.append(make_check("column.transition", "ACI 318-19 10.7.4, 10.7.6.4, 15.2.6, 18.7.4.4, 25.5 (declared rules "
                                 f"{transition['rules']})", int(transition["supported"]), 1, "==", location=key,
                                 details={"kind": transition["kind"], "detail": transition["detail"], "items": transition["items"],
                                          "joints": transition["joints"]}))
    for type_id, kind in joint_types.items():
        beam_physical=kind.get("physical_beam_context") or {}
        if not beam_physical.get("beam_geometry_complete"):
            row=not_evaluated("beam.joint_physical_detail","ACI 318-19 18.8.5;25.2--25.3","Actual beam bars are incomplete",type_id)
        else:
            row=make_check("beam.joint_physical_detail","ACI 318-19 18.8.5;25.2--25.3",
                           int(beam_physical.get("status")=="pass"),1,"==",location=type_id,
                           details={k:v for k,v in beam_physical.items() if k!="beam_segments"})
        checks.append(row)
        if kind['column_above_group'] is None:
            roof=kind.get('physical_roof_anchorage') or {}
            clause='ACI 318-19 18.8.3;18.8.5.1--18.8.5.4;25.4.9;25.4.4 and ER-0188 where headed bars are selected'
            if roof.get('status') not in ('pass','fail'):
                row=not_evaluated('column.roof_anchorage_physical_detail',clause,'Roof product or physical premises are incomplete',type_id)
                row['details'].update(roof)
            else:
                row=make_check('column.roof_anchorage_physical_detail',clause,int(roof['status']=='pass'),1,'==',location=type_id,details=roof)
            checks.append(row)
        if kind['column_above_group'] is not None:
            physical = kind.get('physical_transition') or {}
            clause = 'ACI 318-19 10.7.4, 10.7.6.4, 18.7.5, 18.8.3, 20.5, 25.2--25.5; actual transition geometry'
            if physical.get('status') not in ('pass', 'fail'):
                row = not_evaluated('column.transition_physical_detail', clause,
                                    'Physical transition premises are incomplete', type_id)
                row['details'].update(physical)
            else:
                row = make_check('column.transition_physical_detail', clause,
                                 int(physical['status'] == 'pass'), 1, '==', location=type_id, details=physical)
            checks.append(row)
        for axis, item in kind["shear"].items():
            checks.append(make_check("joint.shear_screen", item["clause"], item["vj_kip"], item["phi_vn_kip"], "<=", "kip", item["id"],
                                     details={"rule": item["rule"], "beam_moment_basis": item["beam_moment_basis"]}))
            if item["evidence_complete"]:
                checks.append(make_check("joint.classification_evidence", "ACI 318-19 Table 18.8.4.3; 15.2.6-15.2.8", 1, 1, "==",
                                         location=item["id"], details={"gamma_basis": item["gamma_basis"],
                                                                       "scope": "declared detailing intent and the declared "
                                                                                "transition; the drawn cage is independent verification"}))
            else:
                checks.append(not_evaluated("joint.classification_evidence", "ACI 318-19 Table 18.8.4.3; 15.2.6-15.2.8",
                                            f"Table 18.8.4.3 inputs rest on evidence not supplied "
                                            f"({', '.join(item['unevaluated_evidence'])}); gamma {item['gamma']:g} is the "
                                            "conservative row, not a verified classification", item["id"]))
        for axis, item in kind["anchorage"].items():
            if item["through_bars_present"]:
                checks.append(make_check("joint.through_bar_depth", "ACI 318-19 18.8.2.3", item["through_bar_depth_required_in"],
                                         item["through_bar_depth_available_in"], "<=", "in", f"{type_id}/{axis}"))
            else:
                checks.append(make_check("joint.terminating_bar_hook", "ACI 318-19 18.8.5.1", item["ldh_required_in"],
                                         item["embedment_available_in"], "<=", "in", f"{type_id}/{axis}"))
        for axis, item in kind["threading"].items():
            checks.append(make_check("beam.bars_thread_column",
                                     f"ACI 318-19 25.2.1 clearance; at most {item['max_layers']} layer(s) between the column bars (25.2.2)",
                                     int(item["passes"]), 1, "==", location=f"{type_id}/{axis}",
                                     details={k: item[k] for k in ("bars_per_layer_that_fit", "layers_needed", "bars_per_face",
                                                                   "core_group", "beam_group", "bars_per_layer_core_cage",
                                                                   "lane_basis")}))
        mats = kind["slab_mats"]
        checks.append(make_check("beam.bar_stacking_clear_of_slab_mats", "ACI 318-19 25.2.2; orthogonal layers at the joint",
                                 int(mats["passes"]), 1, "==", location=type_id,
                                 details={"overlaps": mats["clashes"], "tight_crossings": len(mats["tight_crossings"]),
                                          "slab_mats_placed": mats["slab_mats_placed"]}))
    gravity = None
    if mg.is_system_policy(state.policy_id):
        # Members not designated as part of the SFRS (ACI 318-19 18.14): branch per gravity group from the induced
        # actions of the compatibility run, and the detailing that branch asks of the hoops selected above.
        from Design.Perimeter_Systems import gravity_frame_evidence
        gravity = gravity_frame_evidence(compatibility, cfg, state, beams, columns, actions, expected_ids=expected_ids)
        checks.extend(gravity["checks"])
    # Splices after the gravity branches: a gravity group under 18.14.3.3 takes the special splice rules (18.14.3.3(a)).
    branches = {gid: row.get("branch") for gid, row in ((gravity or {}).get("groups") or {}).items()}
    splices = group_splices(state, beam_hoops={gid: (data.get("hoops") or {}).get("spacing_in") for gid, data in beams.items()},
                            gravity_branches=branches, transitions=transitions,
                            column_hoops={gid: data.get("hoops") for gid, data in columns.items()})
    for gid, splice in splices.items():
        checks.append(splice_location_check(gid, splice))
        physical=splice.get("physical_detail") or {}
        clause="ACI 318-19 18.2.7;18.6.3.3;18.7.4.4;25.2--25.5; actual splice inventory"
        if physical.get("status") not in ("pass","fail"):
            row=not_evaluated("detailing.splice_physical_detail",clause,"Splice physical evidence is incomplete",gid)
            row["details"].update(physical)
        else:
            row=make_check("detailing.splice_physical_detail",clause,int(physical["status"]=="pass"),1,"==",location=gid,details=physical)
        checks.append(row)
    couplers = transition_splices(state, transitions, branches)
    for key, splice in couplers.items():
        check = splice_location_check(key, splice)
        checks.append({**check, "id": "detailing.transition_splice_location"})
    illegal = sorted([gid for gid, splice in splices.items() if not splice["placement"]["legal_placement"]]
                     + [key for key, splice in couplers.items() if not splice["placement"]["legal_placement"]])
    checks.append(make_check("detailing.splices_designed", "ACI 318-19 18.6.3.3 / 18.7.4.4 / 18.2.7 / 25.5",
                             int(not illegal), 1, "==",
                             details={"groups": splices, "transitions": couplers, "without_legal_placement": illegal,
                                      "basis": "a splice type and a legal proposed location for every group and every "
                                               "transition coupler (detailing.splice_location, "
                                               "detailing.transition_splice_location); coupler product qualification is "
                                               "a review"}))
    hoops = {gid: data["hoops"] for gid, data in {**beams, **columns}.items()}
    return {"method_version": METHOD_VERSION, "column_shear_method": cd.COLUMN_SHEAR_METHOD_COLUMN_OWN,
            "slab_eligibility_signature": scwb["slab_eligibility"].get("signature"),
            "slab_eligibility_reason_code": scwb["slab_eligibility"].get("reason_code"),
            "beams": beams, "columns": columns, "beam_strengths": strengths, "joint_types": joint_types,
            "transitions": transitions, "scwb": scwb, "splices": splices, "transition_splices": couplers, "transverse": hoops,
            "gravity_frame": gravity,"beam_torsion":torsion,"common_torsion_input":common_torsion,
            "joint_rules": {type_id: {axis: item["rule"] for axis, item in kind["shear"].items()} for type_id, kind in joint_types.items()},
            "joint_evidence": {"beam_capacity_shear": [m for data in beams.values() for m in data["governing_members"]],
                               "joint_shear": [item for kind in joint_types.values() for item in kind["shear"].values()]},
            "checks": checks, "accepted": all(c["status"] == "pass" for c in checks)}


def _cage_check(member, cage, gid):
    from Design.SMRF_Cage_Layout import cage_passes
    cage = cage or {}
    return make_check(f"{member}.cage_layout",
                      "ACI 318-19 18.7.5.2(b)-(f) / 25.7.2.3" if member == "column" else "ACI 318-19 18.6.4.4 / 25.7.2.3",
                      int(cage_passes(cage)), 1, "==", location=gid,
                      details={"legs": cage.get("legs"), "legs_min": cage.get("legs_min"), "legs_max": cage.get("legs_max"),
                               "hx_in": cage.get("hx_in"), "failing": [c for c in cage.get("checks", []) if not c.get("passes")]})


def column_group_bar_bond(gid, state):
    """ACI 318-19 18.7.4.3 for one column group: 1.25 ld (Eq. 25.4.2.4a, own cb, Ktr = 0) within half the least
    face-to-face clear height of its members."""
    from Design.ACI_Checks import column_bar_bond_18_7_4_3, column_bar_cb_in
    design, group = state.designs[gid], state.groups[gid]
    lu = min(column_clear_heights(state.member(tag))["face"] for tag in group["member_tags"])
    cb = column_bar_cb_in(design.b_in, design.h_in, sp.COL_CLEAR_COVER_IN, design.stirrup_bar_size, design.bar_size,
                          design.top_bars, design.side_bars)
    result = column_bar_bond_18_7_4_3(design.bar_size, design.fc_ksi, sp.FY_KSI, cb["cb_in"], lu)
    return {**result, "cb": cb,
            "clear_height_basis": "least face-to-face clear height of the group's own members (column_clear_heights)",
            "ktr_basis": "Ktr = 0: no transverse reinforcement credit (25.4.2.4 permits it)"}


def special_splice_rules(state, gid, gravity_branches=None):
    """Do the special-moment-frame splice rules (18.2.7, 18.6.3.3) govern this group? Every SFRS group; a gravity group
    under 18.14.3.3 (induced actions beyond the design strengths, or not calculated), which takes 18.2.5-18.2.8 by
    18.14.3.3(a); not a gravity group under 18.14.3.2. A gravity group whose branch is not known is treated as
    18.14.3.3 (conservatively)."""
    from Design.Gravity_Frame_Members import BRANCH_WITHIN_STRENGTH
    if not mg.is_system_policy(state.policy_id) or state.system_of(gid) == mg.SYSTEM_SFRS:
        return True
    return (gravity_branches or {}).get(gid) != BRANCH_WITHIN_STRENGTH


def group_splices(state, beam_hoops=None, gravity_branches=None, transitions=None, column_hoops=None):
    """Splice type, permitted interval and proposed location of every group's own bars (18.6.3.3, 18.7.4.4, 25.5, 18.2.7).

    A Type 2 mechanical splice is selected when the lap does not fit or the bar is too large to lap (25.5.1.1); the
    type alone does not establish a legal location, so each group carries ``placement`` (splice_placement). Under the
    special rules (special_splice_rules) a beam lap is usable only with hoops over it at no more than min(d/4, 4 in)
    (18.6.3.3); ``beam_hoops`` {gid: spacing} gives the spacing the capacity design selected (the installed spacing
    otherwise), and a lap that would need closer hoops than the uniform ones gives way to a Type 2 splice.
    ``gravity_branches`` {gid: 18.14 branch} decides which gravity groups take the special rules.
    """
    result = {}
    for gid, group in sorted(state.groups.items()):
        design = state.designs[gid]
        db = sp.rebar_diameter(design.bar_size)
        sfrs = special_splice_rules(state, gid, gravity_branches)
        if design.is_column:
            clear = min(column_clear_heights(state.member(tag))["physical"] for tag in group["member_tags"])
            lap = 1.3 * cd.straight_development_length(design.bar_size, design.fc_ksi, sp.FY_KSI, top_bar=False)
            permitted = int(design.bar_size) <= 11
            fits = permitted and lap <= clear / 2.0
            kind = "class_B_lap_center_half" if fits else "type_2_mechanical_18.2.7"
            result[gid] = {"class_b_lap_in": lap, "center_half_clear_height_in": clear / 2.0,
                           "lap_splice_permitted_25.5.1.1": permitted, "lap_splice_feasible": fits,
                           "splice_type": kind,
                           "longitudinal_grade_ksi": sp.FY_KSI,
                           "splice_rule": ("Class B lap in the centre half of the clear height (18.7.4.4)" if fits else
                                           "Type 2 mechanical splice (ACI 318-19 18.2.7; Class S in 318-25): "
                                           + ("lap splices are not permitted for bars larger than No. 11 (25.5.1.1)" if not permitted
                                              else "the Class B lap does not fit the centre half of the clear height")),
                           "placement": splice_placement("column", clear, max(design.b_in, design.h_in), sp.FY_KSI, kind,
                                                         lap, db, sfrs)}
        else:
            rows = bar_layers.group_rows(gid)
            clear = min(mp.beam_clear_span_in(state.member(tag)) for tag in group["member_tags"])
            lap_top = 1.3 * cd.straight_development_length(design.bar_size, design.fc_ksi, sp.FY_KSI,
                                                           top_bar=design.h_in - rows["top"]["offsets_in"][0] > 12.0)
            lap_bot = 1.3 * cd.straight_development_length(design.bar_size, design.fc_ksi, sp.FY_KSI, top_bar=False)
            available = clear - 4.0 * design.h_in
            permitted = int(design.bar_size) <= 11                                   # 25.5.1.1
            length_fits = permitted and max(lap_top, lap_bot) <= available
            d = design.h_in - max(rows["top"]["centroid_in"], rows["bottom"]["centroid_in"])
            enclosure_limit = min(d / 4.0, 4.0)
            spacing = (beam_hoops or {}).get(gid) or design.stirrup_spacing_in
            enclosed = spacing is not None and spacing <= enclosure_limit + 1e-9
            fits = length_fits and (enclosed or not sfrs)
            kind = "class_B_lap_outside_hinge_zones" if fits else "type_2_mechanical_18.2.7"
            reason = ("lap splices are not permitted for bars larger than No. 11 (25.5.1.1)" if not permitted else
                      "the Class B lap does not fit between the hinge zones" if not length_fits else
                      f"a Class B lap needs hoops over it at <= min(d/4, 4 in) = {enclosure_limit:.2f} in (18.6.3.3); the "
                      f"hoops are at {spacing:g} in along the member")
            result[gid] = {"class_b_lap_top_in": lap_top, "class_b_lap_bottom_in": lap_bot,
                           "available_between_hinge_zones_in": available,
                           "lap_splice_permitted_25.5.1.1": permitted, "lap_splice_feasible": fits,
                           "lap_enclosure": {"rule": "18.6.3.3: hoops over the lap at <= min(d/4, 4 in)", "applies": sfrs,
                                             "required_spacing_in": enclosure_limit, "hoop_spacing_in": spacing,
                                             "satisfied": bool(enclosed)},
                           "splice_type": kind,
                           "longitudinal_grade_ksi": sp.FY_KSI,
                           "splice_rule": ("Class B lap outside 2h of the joint faces with hoops over the lap (18.6.3.3)" if fits else
                                           "Type 2 mechanical splice (ACI 318-19 18.2.7; Class S in 318-25): " + reason),
                           "placement": splice_placement("beam", clear, design.h_in, sp.FY_KSI, kind,
                                                         max(lap_top, lap_bot), db, sfrs)}
    from Design import SMRF_Conventional_Detail as conventional
    detail_conventional=conventional.active()
    if not detail_conventional:
        from Design.SMRF_Splice_Physical import group_beam_lanes,verify_beam_mechanical_splice
        from Design.SMRF_Transition_Physical import verify_mechanical_splice
        lanes=group_beam_lanes(state)
    # A changing-band transition already declares the starts/continuations
    # of the upper bars. It owns that upper member's splice detail; a second
    # nominal group splice at the same elevation would duplicate the bars.
    managed={}
    by_joint={e["id"]:e for e in joint_inventory(state)["joints"]}
    for key,t in (transitions or {}).items():
        if t.get("kind")=="same_cage":
            continue
        for context in (t.get("contexts") or {key:t}).values():
            for joint in context.get("joints",[]):
                member=by_joint[joint]["above"]
                managed[member.member_tag]=(key,context)
    memo={}
    for gid,record in result.items():
        design=state.designs[gid]
        contexts=[]
        for tag in state.groups[gid]["member_tags"]:
            if tag in managed:
                key,transition=managed[tag]
                detail=transition.get("physical_detail") or {"status":"not_evaluated"}
                clear=column_clear_heights(state.member(tag))["physical"]
                selected=(transition.get("splice") or {}).get("splice_type","none")
                placement=splice_placement("column",clear,max(design.b_in,design.h_in),sp.FY_KSI,
                    "class_B_lap_center_half" if selected.startswith("class_B") else "type_2_mechanical_18.2.7",
                    (transition.get("splice") or {}).get("class_b_lap_in"),sp.rebar_diameter(design.bar_size),
                    special_splice_rules(state,gid,gravity_branches))
                if selected=="none":
                    placement.update(legal_placement=True,splice_length_in=0.,proposed_interval_in=[clear/2,clear/2],
                                     length_basis="No splice: the transition declares continuous or terminated bars.")
                else:
                    placement=_actual_device_placement(placement,detail.get("mechanical_splice_detail") or {})
                contexts.append({"member_tag":tag,"owner":"column_transition","transition_key":key,
                                 "transition_type_id":transition.get("type_id"),"physical_detail":detail,
                                 "clear_length_in":clear,"placement":placement,"splice_type":selected,"quantity_owner":"transition_context"})
                continue
            member=state.member(tag)
            clear=column_clear_heights(member)["physical"] if design.is_column else mp.beam_clear_span_in(member)
            spacing=(beam_hoops or {}).get(gid) or design.stirrup_spacing_in
            hoops_record=(column_hoops or {}).get(gid) or {}
            shear_spacing=(hoops_record.get("spacing_from_shear_beyond_l0_in",hoops_record.get("spacing_from_shear_in"))
                           if design.is_column else None)
            cache=(design,clear,spacing,shear_spacing)
            if not design.is_column:
                cache+=(gid,)
            if cache not in memo:
                if detail_conventional:
                    detail=_conventional_member_splice(design,gid,record,clear,spacing,shear_spacing=shear_spacing)
                elif design.is_column:
                    detail=verify_mechanical_splice(design,fy_ksi=sp.FY_KSI,clear_cover_in=sp.COL_CLEAR_COVER_IN,
                        clear_height_in=clear,spacing_in=spacing,specification=sp.REINFORCEMENT_SPECIFICATION,
                        aggregate_in=sp.AGGREGATE_MAX_SIZE_IN)
                else:
                    lane=lanes[gid]
                    if lane["status"]!="pass":
                        detail={"status":lane["status"],"reason":"No complete common straight beam-lane inventory.","lane_inventory":lane}
                    else:
                        detail=verify_beam_mechanical_splice(design,lane["coordinates"],clear_span_in=clear,
                            spacing_in=spacing,fy_ksi=sp.FY_KSI,specification=sp.REINFORCEMENT_SPECIFICATION,
                            clear_cover_in=sp.BEAM_CLEAR_COVER_IN,aggregate_in=sp.AGGREGATE_MAX_SIZE_IN)
                memo[cache]=detail
            detail=memo[cache]
            if detail_conventional and detail.get("splice")=="class_b_lap":
                lap_kind="class_B_lap_center_half" if design.is_column else "class_B_lap_outside_hinge_zones"
                lap_length=record["class_b_lap_in"] if design.is_column else max(record["class_b_lap_top_in"],record["class_b_lap_bottom_in"])
                contexts.append({"member_tag":tag,"owner":"ordinary_member_splice","clear_length_in":clear,"physical_detail":detail,
                                 "placement":splice_placement("column" if design.is_column else "beam",clear,
                                     max(design.b_in,design.h_in) if design.is_column else design.h_in,sp.FY_KSI,lap_kind,lap_length,
                                     sp.rebar_diameter(design.bar_size),special_splice_rules(state,gid,gravity_branches)),
                                 "splice_type":lap_kind,"quantity_owner":"ordinary_member_splice"})
                continue
            placement=_actual_device_placement(splice_placement("column" if design.is_column else "beam",clear,
                max(design.b_in,design.h_in) if design.is_column else design.h_in,sp.FY_KSI,"type_2_mechanical_18.2.7",None,
                sp.rebar_diameter(design.bar_size),special_splice_rules(state,gid,gravity_branches)),detail)
            contexts.append({"member_tag":tag,"owner":"ordinary_member_splice","clear_length_in":clear,
                             "physical_detail":detail,"placement":placement,"splice_type":"type_2_mechanical_18.2.7",
                             "quantity_owner":"ordinary_member_splice"})
        statuses=[v["physical_detail"].get("status","not_evaluated") for v in contexts]
        status="fail" if "fail" in statuses else "not_evaluated" if "not_evaluated" in statuses else "pass"
        record["lap_alternative"]=dict(record)
        kinds={v["splice_type"] for v in contexts}
        record["splice_type"]=next(iter(kinds)) if len(kinds)==1 else "mixed_actual_member_contexts"
        record["splice_rule"]=("Class B lap where the lap groups keep the 25.2.1 spacing, a Type 2 coupler otherwise; transition-managed members retain their declared transition paths." if detail_conventional else
                              "Explicit documented coaxial Type2 device with actual stagger and local transverse steel; transition-managed members retain their declared transition paths.")
        governing=min(contexts,key=lambda v:(v["placement"]["legal_placement"],v["clear_length_in"]))
        record["placement"]={**governing["placement"],"governing_member_tag":governing["member_tag"],
                              "all_member_placements_checked":len(contexts)}
        record["member_contexts"]=contexts
        record["physical_detail"]={"status":status,"method":"actual_member_splice_inventory_v1",
                                    "members":len(contexts),"failed_members":[v["member_tag"] for v in contexts if v["physical_detail"].get("status")=="fail"],
                                    "unknown_members":[v["member_tag"] for v in contexts if v["physical_detail"].get("status") not in ("pass","fail")],
                                    "context_owners":{str(v["member_tag"]):v["owner"] for v in contexts}}
    return result


def _conventional_member_splice(design,gid,record,clear,spacing,shear_spacing=None):
    """A member's own splice on the conventional basis: a lap where it fits and keeps its spacing, otherwise a coupler.
    A lap in a special-frame beam takes hoops at min(d/4, 4 in) over its length (18.6.3.3); where the member's hoops are wider,
    the extra sets over the lap are counted in the quantities."""
    from Design import SMRF_Conventional_Detail as conventional
    if design.is_column:
        lap,zone,rows=record["class_b_lap_in"],record["center_half_clear_height_in"],None
        cover,outside=sp.COL_CLEAR_COVER_IN,min(6.,(5. if sp.FY_KSI>=80. else 6.)*sp.rebar_diameter(design.bar_size))
        # Beyond l0 (18.7.5.1) the hoops need 18.7.5.5: s <= min(6 db, 6 in) unless shear (18.7.6) asks for less. The
        # coupler sits between the two l0 zones and may take a local gap within those limits (2026-11-07).
        l0=max(design.b_in,design.h_in,clear/6.,18.)
        local_gap=None if shear_spacing is None else {
            "limit_in":min(outside,float(shear_spacing)),"zone_in":clear-2.*l0,"l0_in":l0,"shear_spacing_in":shear_spacing,
            "basis":"18.7.5.5 beyond l0, no larger than the shear spacing of the selected hoops with Vc per 22.5.5.1 "
                    "(18.7.6.2.1 applies over l0 only)"}
    else:
        lap,zone,rows=max(record["class_b_lap_top_in"],record["class_b_lap_bottom_in"]),record["available_between_hinge_zones_in"],bar_layers.group_rows(gid)
        d=design.h_in-max(rows["top"]["centroid_in"],rows["bottom"]["centroid_in"])
        cover,outside=sp.BEAM_CLEAR_COVER_IN,d/2.
        local_gap=None
    detail=conventional.lap_or_device(design,lap_in=lap,zone_in=zone,rows=rows,clear_cover_in=cover,spacing_in=spacing,
        fy_ksi=sp.FY_KSI,specification=sp.REINFORCEMENT_SPECIFICATION,aggregate_in=sp.AGGREGATE_MAX_SIZE_IN,outside_limit_in=outside,
        local_gap=local_gap)
    enclosure=(record.get("lap_enclosure") or {})
    if detail.get("splice")=="class_b_lap" and enclosure.get("applies") and not enclosure.get("satisfied"):
        required=enclosure["required_spacing_in"]
        extra=max(0,math.ceil(lap/required)-math.ceil(lap/max(spacing,1e-9)))
        detail["additional_hoop_sets_over_lap"]=extra
        detail["additional_transverse_steel_volume_in3"]=extra*conventional.hoop_set_length_in(design)*sp.rebar_area(design.stirrup_bar_size)
    return detail


def _actual_device_placement(placement,detail):
    """Replace the preliminary 8db envelope by the actual staggered bodies."""
    bodies=detail.get("bodies") or []
    if not bodies:
        return placement
    interval=[min(b["z_interval_in"][0] for b in bodies),max(b["z_interval_in"][1] for b in bodies)]
    permitted=placement["permitted_interval_in"]
    return {**placement,"proposed_interval_in":interval,"splice_length_in":interval[1]-interval[0],
            "legal_placement":interval[0]>=permitted[0]-1e-9 and interval[1]<=permitted[1]+1e-9,
            "length_basis":"Actual published product bodies including the explicit axial stagger.",
            "product_qualification":detail.get("product")}


def transition_splices(state, transitions, gravity_branches=None):
    """Placement of the mechanical splices a column transition selects (Design.SMRF_Transitions): {key: placement}.

    Where the bars above and below a band boundary are spliced by Type 2 couplers (no lap: a bar above No. 11, or no
    room in the centre half), the couplers sit in the upper column's clear height; they get the same 18.2.7.2 location
    rule as a group's own splice, with the larger of the two bars for the coupler envelope.
    """
    result = {}
    for key, transition in sorted(transitions.items()):
        if not (transition.get("path_counts") or {}).get("mechanically_spliced"):
            continue
        lower_gid, upper_gid = key.split(">", 1)
        upper, lower = state.designs[upper_gid], state.designs[lower_gid]
        clear = 2.0 * (transition.get("splice") or {}).get("center_half_clear_height_in", 0.0)
        db = max(sp.rebar_diameter(upper.bar_size), sp.rebar_diameter(lower.bar_size))
        result[key] = {"splice_type": "type_2_mechanical_18.2.7", "joints": transition.get("joints"),
                       "placement": splice_placement("column", clear, max(upper.b_in, upper.h_in), sp.FY_KSI,
                                                     "type_2_mechanical_18.2.7", None, db,
                                                     special_splice_rules(state, upper_gid, gravity_branches))}
        result[key]["contexts"]={name:{"joints":item.get("joints",[]),"physical_detail":
                                      (item.get("physical_detail") or {}).get("mechanical_splice_detail",{"status":"not_evaluated"})}
                                 for name,item in (transition.get("contexts") or {key:transition}).items()}
        for context in result[key]["contexts"].values():
            context["placement"]=_actual_device_placement(result[key]["placement"],context["physical_detail"])
        if result[key]["contexts"]:
            result[key]["placement"]=min((v["placement"] for v in result[key]["contexts"].values()),key=lambda p:p["legal_placement"])
        statuses=[v["physical_detail"].get("status") for v in result[key]["contexts"].values()]
        result[key]["physical_detail"]={"status":"fail" if "fail" in statuses else "pass" if statuses and all(s=="pass" for s in statuses) else "not_evaluated"}
    return result


def hoop_updates(capacity, state=None):
    """{group id: MemberDesign with the hoops the capacity design selected} for groups whose hoops differ."""
    from dataclasses import replace
    state = state or mg.active()
    updates = {}
    for gid, hoops in capacity["transverse"].items():
        if hoops is None:
            continue
        design = state.designs[gid]
        if design.is_column:
            wanted = replace(design, stirrup_bar_size=hoops["bar_size"], stirrup_legs=hoops["legs_model"],
                             stirrup_spacing_in=hoops["spacing_in"],
                             stirrup_legs_by_direction=tuple(hoops["legs"][d] for d in mg.LEG_DIRECTIONS))
        else:
            wanted = replace(design, stirrup_bar_size=hoops["bar_size"], stirrup_legs=hoops["legs"],
                             stirrup_spacing_in=hoops["spacing_in"])
        if wanted != design:
            updates[gid] = wanted
    return _tied_hoop_updates(updates, state)


def hoop_selection_status(capacity, state=None):
    """{group id: "not_selected" | "installed_mismatch" | "installed_match"} of a capacity design against the installed
    hoops. A group with no selection is never a match with whatever it carries (its seed hoops); a selected group
    differs when hoop_updates (ties included) would change it."""
    state = state or mg.active()
    updates = hoop_updates(capacity, state)
    return {gid: ("not_selected" if hoops is None else "installed_mismatch" if gid in updates else "installed_match")
            for gid, hoops in sorted(capacity["transverse"].items())}


def _tied_hoop_updates(updates, state):
    """Under the system policy the groups of a design tie share one cage: the tie takes the governing hoops among
    its groups' selections (the largest Av / s, then the tightest spacing; adequate for every member of the tie
    in shear and confinement, the section and bars being the same) and changes only when that differs from what
    is installed, so a lighter group's own selection does not reopen a settled tie."""
    from dataclasses import replace
    ties = state.ties()
    if not ties:
        return updates
    tied = {gid for gids in ties.values() for gid in gids}
    result = {gid: design for gid, design in updates.items() if gid not in tied}
    for tie, gids in ties.items():
        wanted = [updates.get(gid, state.designs[gid]) for gid in gids]
        governing = max(wanted, key=lambda d: (d.stirrup_legs * sp.rebar_area(d.stirrup_bar_size) / d.stirrup_spacing_in,
                                              -d.stirrup_spacing_in, sp.rebar_area(d.stirrup_bar_size)))
        for gid in gids:
            design = replace(governing, member_type=state.groups[gid]["member_type"])
            if design != state.designs[gid]:
                result[gid] = design
    return result
