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
            loads, one column at the roof, no roof exemption; hooked and through-bar anchorage against
            the core; beam bars against the core column's lanes; the column transition itself
            (Design.SMRF_Transitions).

Checks carry the group or joint they belong to in ``location``. Joints that share the same incident
groups share one joint-shear, anchorage and threading evaluation, listed with every joint it stands for;
strong-column checks are per physical joint because each column has its own axial envelope.
"""
from __future__ import annotations

import math

import Structure_Parameters as sp
from Design import SMRF_Capacity_Design as cd
from Design.SMRF_Common import SectionAxialDomainError, make_check, not_evaluated
from Design.SMRF_Joint_Adapter import nominal_rectangular_capacity
from Design.SMRF_Joints import MPR_BASIS, beam_capacity_shear_envelope, rectangular_joint_area
from Design.SMRF_Transitions import column_transition
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg
from Model import Member_Properties as mp
from RC_Design_Check import _design_col_steel_layers

METHOD_VERSION = "aci318_19_group_capacity_design_v1"
GRAVITY_FAMILIES = (("seismic_high_gravity", lambda sds: 1.2 + 0.2 * sds, 1.0),
                    ("seismic_low_gravity", lambda sds: 0.9 - 0.2 * sds, 0.0))


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
    return {"member_tag": member.member_tag, "group_id": member.group_id, "axis": mp.beam_axis(member),
            "position": member.location_class,
            "mn_positive_kip_in": nominal["sagging"], "mn_negative_kip_in": nominal["hogging"],
            "mpr_positive_kip_in": probable["sagging"], "mpr_negative_kip_in": probable["hogging"],
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


def member_face_reactions(transfer, member, factor_dead, factor_live):
    """Factored gravity face reactions [left, right] of one beam from its own floor's transfer.

    Dead once, live as the envelope per end over the all-panel case and every saved pattern, plus the
    beam's own drop weight on its own clear span.
    """
    axis = mp.beam_axis(member)
    key = (axis, member.grid_j, member.grid_i) if axis == "x" else (axis, member.grid_i, member.grid_j)
    span = mp.beam_span_in(member)
    face_i, face_j = mp.beam_face_distances_in(member)
    self_weight = factor_dead * mp.beam_drop_weight_kip_per_in(member) * (span - face_i - face_j) / 2.0
    cases = (transfer or {}).get("unit_cases", {}) if transfer else {}
    if not cases:
        return [self_weight, self_weight], "member self-weight only; no slab transfer supplied"
    dead, live = [0.0, 0.0], None
    for name, case in cases.items():
        factor = factor_dead if name == "dead" else factor_live
        if factor == 0.0 or not case:
            continue
        entry = next((b for b in case["beams"] if (b["axis"], b["line_index"], b["span_index"]) == key), None)
        if entry is None:
            raise ValueError(f"Floor transfer case {name!r} has no loads for beam {member.member_tag} {key}.")
        left, right = _span_face_reactions(entry, span, face_i, face_j, factor)
        if name == "dead":
            dead = [left, right]
        else:
            live = [left, right] if live is None else [max(live[0], left), max(live[1], right)]
    live = live or [0.0, 0.0]
    return ([dead[0] + live[0] + self_weight, dead[1] + live[1] + self_weight],
            f"joint-face free body over this beam's own clear span (faces {face_i:g} and {face_j:g} in); live enveloped over "
            f"{len([n for n in cases if n != 'dead'])} unit case(s) of its own floor's transfer; plus its own drop weight")


def beam_group_capacity(gid, state, transfers, sds):
    """Capacity shear of every member of one beam group and the group's hinge-zone hoops."""
    design = state.designs[gid]
    rows = bar_layers.group_rows(gid)
    d = design.h_in - max(rows["top"]["centroid_in"], rows["bottom"]["centroid_in"])
    members, worst_ve, worst_mechanism = [], 0.0, 0.0
    for tag in state.groups[gid]["member_tags"]:
        member = state.member(tag)
        strengths = beam_probable_strengths(member)
        transfer = None if transfers is None else transfers[member.story_or_floor]
        for family_name, dead_of, live in GRAVITY_FAMILIES:
            dead = dead_of(sds)
            reactions, basis = member_face_reactions(transfer, member, dead, live)
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
              **cd.select_beam_hoops(design.b_in, d, design.fc_ksi, sp.FY_KSI, design.bar_size, sp.BEAM_CLEAR_COVER_IN,
                                     design.top_bars, design.bot_bars, worst_ve, worst_mechanism)}
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
        "materials": {"fy_ksi": sp.FY_KSI, "es_ksi": sp.ES_KSI},
        "column": {"bar_size": design.bar_size, "top_bars": design.top_bars, "bot_bars": design.bot_bars,
                   "side_bars": design.side_bars, "centroid_offset_in": mp.longitudinal_cover_in(design),
                   "clear_cover_in": sp.COL_CLEAR_COVER_IN, "stirrup_bar_size": design.stirrup_bar_size,
                   "layers": _design_col_steel_layers(design, about="h"),
                   "layers_about_z": _design_col_steel_layers(design, about="b")},
        "column_axial_envelope": envelopes["axial"], "column_shear_demand": envelopes["shear"],
        "column_action_envelopes": envelopes["detail"],
        "column_shear_method": cd.COLUMN_SHEAR_METHOD_COLUMN_OWN,
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
    joints, types = [], {}
    for level in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joint = mg.joint_members(level, i, j)
                core, above = mp.joint_core_column(level, i, j), joint["column_above"]
                beams = {axis: _beams(joint, axis) for axis in ("x", "y")}
                for axis in ("x", "y"):
                    if len({b.group_id for b in beams[axis]}) > 1:
                        raise mg.GroupedStateError(f"Joint (level {level}, {i}, {j}): the two {axis.upper()} beams belong to "
                                                   "different groups; no along-line reinforcement transition is declared.")
                key = (core.group_id, None if above is None else above.group_id,
                       beams["x"][0].group_id if beams["x"] else None, len(beams["x"]),
                       beams["y"][0].group_id if beams["y"] else None, len(beams["y"]))
                node = level * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1) + j * (sp.NUM_BAY_X + 1) + i + 1
                entry = {"id": f"joint_{node}", "node_tag": node, "level": level, "grid_i": i, "grid_j": j,
                         "is_roof": above is None, "core": core, "above": above, "below": joint["column_below"],
                         "beams": beams, "joint": joint}
                kind = types.setdefault(key, {"type_id": f"joint_{node}", "key": key, "representative": entry, "joints": []})
                kind["joints"].append(entry["id"])
                entry["type_id"] = kind["type_id"]
                joints.append(entry)
    return {"joints": joints, "types": {kind["type_id"]: kind for kind in types.values()}}


def joint_transition(entry, column_hoops=None):
    """The column transition at one joint (None at the roof), with the lower group's installed hoop spacing."""
    core, above = entry["core"], entry["above"]
    if above is None:
        return None
    upper_clear = column_clear_heights(above)["physical"]
    depths = [b.design.h_in for axis in ("x", "y") for b in entry["beams"][axis]]
    return column_transition(core.design, above.design, fy_ksi=sp.FY_KSI, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                             upper_clear_height_in=upper_clear, joint_depth_in=min(depths),
                             lower_hoop_spacing_in=(column_hoops or {}).get(core.group_id))


def joint_shear(entry, axis, strengths, transition, beam_hoops):
    """Vj against phi Vn for one joint and direction, from its real incident members (18.8.4)."""
    core, beams = entry["core"].design, entry["beams"][axis]
    other = "y" if axis == "x" else "x"
    transverse = entry["beams"][other]
    depth, width = (core.h_in, core.b_in) if axis == "x" else (core.b_in, core.h_in)
    fy = sp.FY_KSI
    level = "roof" if entry["is_roof"] else "floor"

    def tension(beam, sign):
        s = strengths[beam.member_tag]
        return 1.25 * fy * (s["tension_steel_hogging_in2"] if sign == "hogging" else s["tension_steel_sagging_in2"])

    def mpr(beam, sign):
        s = strengths[beam.member_tag]
        return s["mpr_negative_kip_in"] if sign == "hogging" else s["mpr_positive_kip_in"]

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
    bw = beams[0].design.b_in
    beam_depth = max([b.design.h_in for b in beams] + [b.design.h_in for b in transverse])
    clear = min(mp.beam_clear_span_in(b) for b in beams)
    transverse_clear = min((mp.beam_clear_span_in(b) for b in transverse), default=0.0)
    above = entry["above"]
    column_clear = column_clear_heights(above)["physical"] if above is not None else 0.0
    continuity = {"beam_reinforcement_continuous_through_interior_joints": True,
                  "column_reinforcement_continuous_through_floor_joints":
                      (None if transition is None else bool(transition["column_reinforcement_continuous"])),
                  "basis": ("both beams of a direction at a joint belong to one group (same bars, run through, spliced "
                            "outside the hinge zones); column continuity from the declared transition at this joint "
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
    aj = rectangular_joint_area(column_depth_in=depth, column_width_in=width, beam_width_in=bw, beam_center_offset_in=0.0)
    vn = gamma * math.sqrt(core.fc_ksi * 1000.0) * aj / 1000.0
    return {"id": f"joint_shear/{entry['type_id']}/{axis}", "type_id": entry["type_id"], "level": level, "axis": axis,
            "core_group": entry["core"].group_id, "column_above_group": None if above is None else above.group_id,
            "beam_group": beams[0].group_id, "transverse_beam_group": transverse[0].group_id if transverse else None,
            "beams_in_direction": len(beams), "beams_perpendicular": len(transverse),
            "beam_face_forces_kip": forces, "column_shear_kip": vcol, "hogging_beam_tag": hogging_beam,
            "probable_face_forces_complete": True, "column_shear_consistent_with_mpr": True,
            "nominal_vn_kip": vn, "capacity_basis": "ACI318-19_Table18.8.4.3",
            "capacity_topology_and_confinement_checked": False,
            "gamma": gamma, "gamma_basis": classification["gamma_basis"], "classification": classification,
            "column_continuity": "terminating" if level == "roof" else ("continuous" if classification["column"]["state"] == "continuous" else "other"),
            "beam_continuity": "continuous" if len(beams) == 2 else "terminating",
            "transverse_confinement": classification["confinement"]["confined"],
            "evidence_complete": classification["evidence_complete"],
            "unevaluated_evidence": classification["unevaluated_evidence"],
            "confined_faces": parallel + perpendicular, "joint_core_in": {"depth": depth, "width": width, "fc_ksi": core.fc_ksi},
            "aj_in2": aj, "vj_kip": vj, "phi_vn_kip": cd.PHI_JOINT * vn, "passes": vj <= cd.PHI_JOINT * vn,
            "slab_steel_in_tension_included": True}


def joint_anchorage(entry, axis, column_hoop_bar):
    """Hooked anchorage of terminating beam bars (18.8.5.1) and through-bar depth (18.8.2.3) at one joint."""
    core, beams = entry["core"].design, entry["beams"][axis]
    depth = core.h_in if axis == "x" else core.b_in
    db = sp.rebar_diameter(beams[0].design.bar_size)
    hoop_db = sp.rebar_diameter(column_hoop_bar if column_hoop_bar is not None else core.stirrup_bar_size)
    ldh = max(sp.FY_KSI * 1000.0 * db / (65.0 * math.sqrt(core.fc_ksi * 1000.0)), 8.0 * db, 6.0)
    through = len(beams) == 2
    available = depth - sp.COL_CLEAR_COVER_IN - hoop_db
    required_through = max(20.0 * db, max(b.design.h_in for b in beams) / 2.0)
    return {"type_id": entry["type_id"], "axis": axis, "through_bars_present": through,
            "ldh_required_in": ldh, "embedment_available_in": available, "passes": through or ldh <= available,
            "hook": "standard 90-degree hook into the confined core, 18.8.5.1",
            "through_bar_depth_required_in": required_through, "through_bar_depth_available_in": depth,
            "through_bar_passes": (required_through <= depth) if through else True,
            "beam_bar_diameter_in": db, "core_group": entry["core"].group_id, "beam_group": beams[0].group_id}


def joint_threading(entry, axis):
    """Do the beam bars of one direction pass the joint core's column bars within the layer limit?"""
    beam = entry["beams"][axis][0]
    fit = bar_layers.bars_per_layer(entry["core"].design, beam.design, axis)
    n = max(beam.design.top_bars, beam.design.bot_bars)
    needed = math.ceil(n / fit) if fit and fit > 0 else None
    limit = sp.BEAM_BAR_MAX_LAYERS
    return {"type_id": entry["type_id"], "axis": axis, "bars_per_layer_that_fit": fit, "bars_per_face": n,
            "layers_needed": needed, "max_layers": limit, "passes": needed is not None and needed <= limit,
            "core_group": entry["core"].group_id, "beam_group": beam.group_id}


def slab_mat_clashes(entry):
    """Crossing slab mats against the top bar layers of the beams at one joint (Design.SMRF_Cage_Geometry rule)."""
    from Design.SMRF_Cage_Geometry import INTERLAYER_CLEAR_MIN_IN, bar_diameter
    layout = (_layout() or {}).get("layers") or {}
    mats, placed = [], bool(layout)
    for name, layer in layout.items():
        if not name.endswith("top") and not name.endswith("bottom"):
            continue
        if any(key not in layer for key in ("bar_size", "clear_cover_outer_mat_in", "layer", "axis")):
            placed = False
            continue
        mat_db = bar_diameter(layer["bar_size"])
        if name.endswith("top"):
            depth = layer["clear_cover_outer_mat_in"] + mat_db / 2.0 + (0.0 if layer["layer"] == "outer" else mat_db)
        else:
            depth = (sp.SLAB_THICKNESS_IN - layer["clear_cover_outer_mat_in"] - mat_db / 2.0
                     - (0.0 if layer["layer"] == "outer" else mat_db))
        mats.append({"mat": name, "db_in": mat_db, "centroid_from_top_in": depth, "runs_along": layer["axis"]})
    clashes, tight = [], []
    for axis in ("x", "y"):
        if not entry["beams"][axis]:
            continue
        beam = entry["beams"][axis][0]
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


def scwb_evidence(inventory, actions, expected_ids, layout_established=None):
    """Strong-column / weak-beam evidence at every joint, each column on its own section and axial loads.

    The structure SMRF_Joints.evaluate_joints reads: per joint, direction and sway sign, the capacities
    of the columns actually framing in (the column below at its top end, the column above at its bottom
    end; one column at the roof) and of the beams on each face. A column's Mn is the least over every
    solved combination at that column end's own joint-face axial load and over both compression faces;
    an axial load outside a section's domain leaves that column unevaluated with the reason (a section
    that cannot carry its axial load is rejected by the strength check, not hidden here). Beam strengths
    are the member's rectangular Mn plus the developed slab increment, the numbers its hinge yields at.
    """
    by_id = {a["id"]: a for a in actions}
    missing = [cid for cid in expected_ids if cid not in by_id]
    established = _layout() is not None if layout_established is None else layout_established
    beam_cache, joints = {}, []

    def beam_strength(member):
        if member.member_tag not in beam_cache:
            beam_cache[member.member_tag] = mp.beam_strengths(member)
        return beam_cache[member.member_tag]

    for entry in inventory["joints"]:
        columns = [("below", entry["below"], "j")] + ([("above", entry["above"], "i")] if entry["above"] is not None else [])
        record = {"id": entry["id"], "node_tag": entry["node_tag"], "floor": entry["level"], "grid_i": entry["grid_i"],
                  "grid_j": entry["grid_j"], "is_roof": entry["is_roof"], "type_id": entry["type_id"], "directions": {}}
        for axis in ("x", "y"):
            column_caps = []
            for position, column, end in columns:
                cap = {"tag": column.member_tag, "end": end, "position": position, "group_id": column.group_id,
                       "axial_envelope_checked": False}
                try:
                    if missing:
                        raise ValueError(f"Final factored-action inventory is incomplete: missing {missing[:3]}.")
                    observations = []
                    for cid in expected_ids:
                        p = float(by_id[cid]["members"][str(column.member_tag)][f"axial_{end}_kip"])
                        mn, face, solved = column_nominal_moment(column.design, axis, p)
                        observations.append({"combination_id": cid, "factored_axial_kip": p, "mn_kip_in": mn,
                                             "governing_compression_face": face})
                    governing = min(observations, key=lambda o: o["mn_kip_in"])
                    cap.update(governing)
                    cap.update(axial_envelope_checked=True,
                               axial_min_kip=min(o["factored_axial_kip"] for o in observations),
                               axial_max_kip=max(o["factored_axial_kip"] for o in observations),
                               combination_count=len(observations),
                               compression_face_basis="minimum of both faces; conservative uniaxial sway envelope",
                               axial_reference="joint_faces", capacity_basis=solved["basis"])
                except (ValueError, TypeError, KeyError) as exc:
                    cap["reason"] = str(exc)
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
                        cap["reason"] = "no slab in the model: the developed beam-plus-slab strength is not defined"
                    else:
                        rectangular = family["rectangular"][flexure]["mn_kip_in"]
                        exterior = basis["exterior_ends"][end]
                        undeveloped = (exterior and basis["exterior_anchorage"] is not None
                                       and not basis["exterior_anchorage"]["developed"])
                        cap.update(mn_kip_in=rectangular, capacity_basis=family["basis"])
                        if not established:
                            cap["slab_reason"] = "Developed beam-plus-slab section compatibility evidence is missing."
                        elif undeveloped:
                            total = family["undeveloped"][flexure]["mn_kip_in"]
                            cap.update(slab_basis=("terminated_undeveloped" if flexure == "negative"
                                                   else "flange_concrete_undeveloped_bars"),
                                       slab_mn_kip_in=0.0 if flexure == "negative" else max(0.0, total - rectangular))
                        else:
                            cap.update(slab_basis="developed_effective_width",
                                       slab_mn_kip_in=max(0.0, family[f"slab_increment_{flexure}_kip_in"]))
                    beam_caps.append(cap)
                states[sign] = {"nominal_strengths": True, "column_capacities": column_caps, "beam_capacities": beam_caps}
            record["directions"][axis] = states
        joints.append(record)
    return {"joints": joints, "expected_combination_ids": list(expected_ids), "action_inventory_complete": not missing,
            "axial_envelope_basis": ("minimum nominal strength of each column, on its own section, across every supplied "
                                     "factored case at its own joint-face axial load and both compression faces"),
            "uniaxial_only": True, "roof_exemption_applied": False}


# ---- the whole capacity design ------------------------------------------------------------------------------
def build_group_capacity_design(actions, expected_ids, cfg, transfers=None, state=None):
    """Capacity-design evidence and checks of the installed grouped design.

    ``actions`` are the solved combinations (Group_Checks.capture_member_actions per combination);
    ``transfers`` {floor: validated transfer} or None. Returns beams and columns by group, joints by
    type, transitions, strong-column evidence, the hoops each group needs, and the checks.
    """
    state = state or mg.active()
    if state is None:
        raise mg.GroupedStateError("The group capacity design needs an installed grouped design.")
    sds = sp.ASCE_SDS
    beams = {gid: beam_group_capacity(gid, state, transfers, sds)
             for gid, group in sorted(state.groups.items()) if group["member_type"] != "column"}
    columns = {gid: column_group_capacity(gid, state, actions, cfg)
               for gid, group in sorted(state.groups.items()) if group["member_type"] == "column"}
    beam_hoops = {gid: data["hoops"] or {} for gid, data in beams.items()}
    column_spacing = {gid: (data["hoops"] or {}).get("spacing_in") for gid, data in columns.items()}
    inventory = joint_inventory(state)
    strengths = {tag: beam_probable_strengths(state.member(tag))
                 for gid, group in state.groups.items() if group["member_type"] != "column" for tag in group["member_tags"]}
    joint_types, transitions = {}, {}
    for type_id, kind in sorted(inventory["types"].items()):
        entry = kind["representative"]
        transition = joint_transition(entry, column_spacing)
        if transition is not None and entry["above"].group_id != entry["core"].group_id:
            transitions[f"{entry['core'].group_id}>{entry['above'].group_id}"] = {**transition, "type_id": type_id,
                                                                                 "joints": list(kind["joints"])}
        shear = {axis: joint_shear(entry, axis, strengths, transition, beam_hoops) for axis in ("x", "y") if entry["beams"][axis]}
        hoop_bar = (columns[entry["core"].group_id]["hoops"] or {}).get("bar_size")
        joint_types[type_id] = {
            "type_id": type_id, "joints": list(kind["joints"]), "level": entry["level"],
            "core_group": entry["core"].group_id, "column_above_group": None if entry["above"] is None else entry["above"].group_id,
            "transition": None if transition is None else {k: transition[k] for k in ("kind", "supported", "column_reinforcement_continuous")},
            "shear": shear,
            "anchorage": {axis: joint_anchorage(entry, axis, hoop_bar) for axis in ("x", "y") if entry["beams"][axis]},
            "threading": {axis: joint_threading(entry, axis) for axis in ("x", "y") if entry["beams"][axis]},
            "slab_mats": slab_mat_clashes(entry),
            "joint_hoops": columns[entry["core"].group_id]["hoops"],
            "joint_hoop_basis": "18.8.3.1: the hoops of the column below (the joint core) continue through the joint depth"}
    # Joint transverse steel designed and every Table 18.8.4.3 input on evidence: only then is the nominal strength usable.
    for kind in joint_types.values():
        for item in kind["shear"].values():
            item["capacity_topology_and_confinement_checked"] = bool(kind["joint_hoops"] is not None and item["evidence_complete"])
    scwb = scwb_evidence(inventory, actions, expected_ids)

    checks = []
    for gid, data in beams.items():
        checks.append(make_check("beam.capacity_shear_section", "ACI 318-19 22.5.1.2 with 18.6.5.1 Ve",
                                 data["vs_required_kip"], data["vs_limit_kip"], "<=", "kip", gid))
        checks.append(make_check("beam.hoops_selected", "ACI 318-19 18.6.4.4 / 18.6.5", int(data["hoops"] is not None), 1, "==",
                                 location=gid))
        checks.append(_cage_check("beam", data["cage"], gid))
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
        for axis, item in kind["shear"].items():
            checks.append(make_check("joint.shear_screen", "ACI 318-19 18.8.4", item["vj_kip"], item["phi_vn_kip"], "<=", "kip", item["id"]))
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
                                                                   "core_group", "beam_group")}))
        mats = kind["slab_mats"]
        checks.append(make_check("beam.bar_stacking_clear_of_slab_mats", "ACI 318-19 25.2.2; orthogonal layers at the joint",
                                 int(mats["passes"]), 1, "==", location=type_id,
                                 details={"overlaps": mats["clashes"], "tight_crossings": len(mats["tight_crossings"]),
                                          "slab_mats_placed": mats["slab_mats_placed"]}))
    splices = group_splices(state)
    checks.append(make_check("detailing.splices_designed", "ACI 318-19 18.6.3.3 / 18.7.4.4 / 18.2.7 / 25.5", 1, 1, "==",
                             details={"groups": splices}))
    hoops = {gid: data["hoops"] for gid, data in {**beams, **columns}.items()}
    return {"method_version": METHOD_VERSION, "column_shear_method": cd.COLUMN_SHEAR_METHOD_COLUMN_OWN,
            "beams": beams, "columns": columns, "beam_strengths": strengths, "joint_types": joint_types,
            "transitions": transitions, "scwb": scwb, "splices": splices, "transverse": hoops,
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


def group_splices(state):
    """Splice type and location of every group's own bars (18.6.3.3, 18.7.4.4, 25.5, 18.2.7)."""
    result = {}
    for gid, group in sorted(state.groups.items()):
        design = state.designs[gid]
        if design.is_column:
            clear = min(column_clear_heights(state.member(tag))["physical"] for tag in group["member_tags"])
            lap = 1.3 * cd.straight_development_length(design.bar_size, design.fc_ksi, sp.FY_KSI, top_bar=False)
            permitted = int(design.bar_size) <= 11
            fits = permitted and lap <= clear / 2.0
            result[gid] = {"class_b_lap_in": lap, "center_half_clear_height_in": clear / 2.0,
                           "lap_splice_permitted_25.5.1.1": permitted, "lap_splice_feasible": fits,
                           "splice_type": "class_B_lap_center_half" if fits else "type_2_mechanical_18.2.7"}
        else:
            rows = bar_layers.group_rows(gid)
            clear = min(mp.beam_clear_span_in(state.member(tag)) for tag in group["member_tags"])
            lap_top = 1.3 * cd.straight_development_length(design.bar_size, design.fc_ksi, sp.FY_KSI,
                                                           top_bar=design.h_in - rows["top"]["offsets_in"][0] > 12.0)
            lap_bot = 1.3 * cd.straight_development_length(design.bar_size, design.fc_ksi, sp.FY_KSI, top_bar=False)
            available = clear - 4.0 * design.h_in
            fits = max(lap_top, lap_bot) <= available
            result[gid] = {"class_b_lap_top_in": lap_top, "class_b_lap_bottom_in": lap_bot,
                           "available_between_hinge_zones_in": available, "lap_splice_feasible": fits,
                           "splice_type": "class_B_lap_outside_hinge_zones" if fits else "type_2_mechanical_18.2.7"}
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
    return updates
