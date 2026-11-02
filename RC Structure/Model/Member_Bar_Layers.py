"""Beam bar layers where different design groups meet (grouped iterative design, 2026-10-02).

The uniform code has one beam and one column cage, so one joint decides how the orthogonal beam bars
stack (Structure_Parameters.beam_bar_layers). With design groups the X and Y beams crossing at a joint
can have different widths, depths, bar sizes and counts, the column cage they thread differs between
plan locations, and one beam group runs through several kinds of joint. Beam bars are straight along a
line, so a group's rows have to work at every joint its members reach. This module solves that for the
installed design and returns, per beam group, the rows the strengths and the checks use.

Rules (the uniform rules, stated for unlike members; equal members give exactly the uniform result):
  * LANES. How many of a beam's bars pass a column cage in one layer is the geometry of
    Design.SMRF_Cage_Geometry.column_cage_admits_beam_bars for that beam and that column cage. A group
    takes the smallest number over the joints of its members and splits its bars into layers
    accordingly (Structure_Parameters._split_into_layers and BEAM_BAR_MAX_LAYERS).
  * COLUMN CAGE AT A JOINT. The cage of the column below the floor (the joint-core convention of
    Model.Member_Properties). Where the column above has another cage its bars or dowels also pass
    through the joint; that is the transition check's subject (Design.SMRF_Transitions), not this
    module's.
  * TOP FACE. Beam tops are flush with the slab. The direction the stacking convention puts on top
    takes the first layer; the layers then follow BEAM_BAR_LAYER_ORDER.
  * BOTTOM FACE. With equal beam depths the order is the mirror of the top (the convention's lower
    direction nearest the bottom). With unequal depths the deeper beam's bars are placed first: its
    soffit is lower, and the shallower beam's bars may clear them without giving up depth.
  * CLEARANCE. Each layer sits at the larger of its own nominal offset (clear cover + hoop + half a
    bar) and one pitch clear of every crossing layer already placed, the pitch between two layers
    being half of each bar plus max(1 in, the larger bar) (ACI 318-19 25.2.2).
  * ONE ARRANGEMENT PER GROUP. Offsets only ever increase while the joints are swept, until every
    joint is satisfied by the same rows; a sweep limit refuses an arrangement that does not settle.

  * SLAB MATS. The selected slab mats stay at their designed elevations. A beam row that crosses a
    mat moves inward to the next clear bar-grid elevation, then the crossing beam rows are coordinated
    again. The existing interlayer clearance is a declared construction allowance at these orthogonal
    crossings, not a claim that 25.2.2 prescribes that gap between orthogonal bars. No local bends or mat
    displacements are assumed. The resulting rows are used by strength, capacity and model consumers.

What this module does not decide: hook tails and whether the resulting rows fit the section
(Design.SMRF_Cage_Geometry / the capacity checks, and validate_bar_rows).

Row controls (2026-10-31): row_controls explains the arrangement without changing it: per layer, the last constraint
that raised its offset (the group's own layer pitch, a crossing beam layer, a slab mat) and the joint type where it
did. The lanes and the crossing gap are the declared detailing policy of Design.SMRF_Cage_Geometry.passage_policy.

Joint inventory (2026-10-31, JOINT_INVENTORY_RULE_ID): at a joint whose column above has another cage and whose declared
transition (Design.SMRF_Transitions.column_transition) bends or offsets bars inside the joint, the beam bars pass those
bars where they are: the lanes there are threaded through the joint's bar inventory, not through the core cage alone.
Every lower bar that stays put blocks its own lateral position; every moved bar blocks the whole lateral interval its bend
sweeps, from its lower position to its position at the top of the joint (v2, swept bend intervals). The two ends alone
(v1) were not conservative: a bar moved farther than twice the passage gap leaves a lane between its ends that the bar
crosses at some beam row elevation. Where the transition declares no bar paths (a refused transition), where the bars run
inside the joint is not established and no lane is claimed there (the threading check fails closed). Everywhere else the
inventory is the core cage and the lanes are the earlier ones exactly. The bar-row basis records the rule only where it
lowers a lane (``_basis["joint_inventory"]``), so the rows of every other design keep their earlier form.

Row audit (2026-11-01, ROW_AUDIT_ID): row_audit checks every beam group's rows on the complete shared arrangement of one
state, with the slab mats and settings in force. A column cage changes the lanes of the beams that thread it, so the
layers of those beams and of every beam they cross; a unit-by-unit admissibility found on another state is not evidence
for this one. The selection closes its terminal arrangement on it and the capacity design refuses a state it rejects.
"""
from __future__ import annotations

import Structure_Parameters as sp
from Model import Member_Groups as mg

RULE_ID = "group_bar_layers_v2__straight_bars_fixed_slab_mats__deeper_soffit_first"
MAX_SWEEPS = 64
_TOL = 1e-9
_CACHE = {}


def _db(size):
    return sp.rebar_diameter(size)


def nominal_offset_in(design):
    """Face to the centroid of a first-layer bar: clear cover + hoop + half the bar."""
    return sp.longitudinal_cover_in("beam", design.bar_size, design.stirrup_bar_size)


def pitch_in(design_a, design_b):
    """Centre-to-centre distance between two layers (of one beam or of two crossing beams), on the bar grid."""
    da, db = _db(design_a.bar_size), _db(design_b.bar_size)
    return sp.snap_up_to_grid(0.5 * (da + db) + max(sp.BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN, da, db))


def bars_per_layer(column_design, beam_design, axis):
    """How many bars of this beam pass this column cage in one layer, in the beam's own direction."""
    from Design.SMRF_Cage_Geometry import column_cage_admits_beam_bars
    n = max(beam_design.top_bars, beam_design.bot_bars)
    # The lanes lie on the bar grid from the layout hoop: both are part of the key (a changed grid is never served
    # another grid's lanes).
    key = ("fit", column_design.section_key(), beam_design.section_key(), axis, sp.COL_CLEAR_COVER_IN,
           sp.BEAM_CLEAR_COVER_IN, sp.AGGREGATE_MAX_SIZE_IN, sp.BAR_POSITION_GRID_IN, sp.BAR_LAYOUT_HOOP_DB_IN)
    if key not in _CACHE:
        if len(_CACHE) > 8192:
            _CACHE.clear()
        _CACHE[key] = column_cage_admits_beam_bars(
            column_design.b_in, column_design.h_in, sp.COL_CLEAR_COVER_IN, _db(column_design.stirrup_bar_size),
            _db(column_design.bar_size), column_design.top_bars, column_design.side_bars, beam_design.b_in,
            sp.BEAM_CLEAR_COVER_IN, _db(beam_design.stirrup_bar_size), _db(beam_design.bar_size), n,
            sp.AGGREGATE_MAX_SIZE_IN)["bars_per_layer"][axis]
    return _CACHE[key]


# ---- the joint's bar inventory at a column transition (2026-10-31) ---------------------------------------------
JOINT_INVENTORY_RULE_ID = "joint_inventory_lanes_v2__swept_bend_intervals"
INVENTORY_NOT_ESTABLISHED = "not_established"


def transition_inventory(below, above, transition):
    """The vertical bars inside a joint whose core column is ``below`` and whose column above is ``above``, from their
    declared transition (Design.SMRF_Transitions.column_transition result), in the core's plan coordinates
    (Design.SMRF_Cage_Geometry.column_bars_plan: x along h, y along b, from the corner):

      * None when no bar moves inside the joint (the core cage is the inventory);
      * a tuple of (x0, y0, x1, y1, db): every lower bar at its lower position (x0 = x1, y0 = y1), and every bar its
        declared path bends or offsets inside the joint as the plan segment from its lower position to its position at
        the top of the joint. The bend points bound a straight inclined run, so at every elevation of the joint the bar
        lies on that segment, and a lane clear of the segment's whole lateral interval is clear of the bar at every
        beam row (beam_bars_through_inventory);
      * INVENTORY_NOT_ESTABLISHED when the transition declares no bar paths (it was refused before any path was
        placed) or a moved bar lacks either position: where the bars run inside the joint is then not established and
        no lane is claimed through it."""
    from Design import SMRF_Transitions as transitions
    paths = transition.get("paths") or []
    if not paths:
        return INVENTORY_NOT_ESTABLISHED
    moved = [p for p in paths if p.get("bent")]
    if not moved:
        return None
    if any(p.get("lower_xy_in") is None or p.get("target_xy_in") is None for p in moved):
        return INVENTORY_NOT_ESTABLISHED
    db = transitions._BAR[int(below.bar_size)][0]
    segments = [(b["x_in"], b["y_in"], b["x_in"], b["y_in"]) for b in transitions.bar_coordinates(below, sp.COL_CLEAR_COVER_IN)]
    segments += [(p["lower_xy_in"][0], p["lower_xy_in"][1], p["target_xy_in"][0], p["target_xy_in"][1]) for p in moved]
    unique = set()
    for x0, y0, x1, y1 in segments:
        unique.add((round(y0 + 0.5 * below.h_in, 6), round(x0 + 0.5 * below.b_in, 6),
                    round(y1 + 0.5 * below.h_in, 6), round(x1 + 0.5 * below.b_in, 6)))
    return tuple((*entry, db) for entry in sorted(unique))


def joint_inventory(level, i, j):
    """The bar inventory of the joint at floor ``level``, grid (i, j) of the installed design: {"bars", "established",
    "core_group", "above_group", "transition", "bars_moved", "joint_depth_in", "upper_clear_height_in"} where the column
    above has another cage and its declared
    transition moves bars inside the joint, or declares no bar paths (``established`` False, ``bars`` is
    INVENTORY_NOT_ESTABLISHED); None otherwise (the core cage is the inventory). The cache holds the geometry only:
    equal designs at a corner, an edge and an interior joint share it, and the group ids are this joint's, attached on
    every call."""
    from Model import Member_Properties as mp
    joint = mg.joint_members(level, i, j)
    core, above = mp.joint_core_column(level, i, j), joint["column_above"]
    if above is None or above.design == core.design:
        return None
    from Design.Group_Capacity import column_clear_heights
    from Design.SMRF_Transitions import column_transition
    depth = min(joint[k].design.h_in for k in ("beam_x_minus", "beam_x_plus", "beam_y_minus", "beam_y_plus")
                if joint[k] is not None)
    clear = column_clear_heights(above)["physical"]
    key = ("inventory", core.design, above.design, depth, clear, sp.FY_KSI, sp.COL_CLEAR_COVER_IN, sp.BAR_POSITION_GRID_IN,
           sp.BAR_LAYOUT_HOOP_DB_IN)
    if key not in _CACHE:
        if len(_CACHE) > 8192:
            _CACHE.clear()
        transition = column_transition(core.design, above.design, fy_ksi=sp.FY_KSI, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                                       upper_clear_height_in=clear, joint_depth_in=depth)
        bars = transition_inventory(core.design, above.design, transition)
        _CACHE[key] = None if bars is None else {
            "bars": bars, "established": bars != INVENTORY_NOT_ESTABLISHED,
            "transition": {k: transition[k] for k in ("kind", "supported")},
            "bars_moved": sum(1 for p in transition.get("paths") or [] if p.get("bent")),
            "joint_depth_in": depth, "upper_clear_height_in": clear}
    geometry = _CACHE[key]
    return None if geometry is None else {**geometry, "core_group": core.group_id, "above_group": above.group_id}


def inventory_lanes(bars, column_design, beam_design, axis):
    """How many bars of ``beam_design`` pass, in one layer and its own direction, a joint whose core is
    ``column_design`` and whose vertical bars are ``bars`` (transition_inventory): the declared lane rule, each column
    bar with its own diameter and each moved bar's whole bend interval kept clear. 0 when the inventory is not
    established (INVENTORY_NOT_ESTABLISHED): no lane is claimed through bars whose paths are not declared."""
    from Design.SMRF_Cage_Geometry import beam_bars_through_inventory, clear_spacing_limit
    if isinstance(bars, str):
        return 0
    n = max(beam_design.top_bars, beam_design.bot_bars)
    key = ("inventory_fit", bars, column_design.b_in, column_design.h_in, beam_design.section_key(), axis,
           sp.BEAM_CLEAR_COVER_IN, sp.AGGREGATE_MAX_SIZE_IN, sp.BAR_POSITION_GRID_IN, sp.BAR_LAYOUT_HOOP_DB_IN)
    if key not in _CACHE:
        if len(_CACHE) > 8192:
            _CACHE.clear()
        db = _db(beam_design.bar_size)
        _CACHE[key] = beam_bars_through_inventory(
            column_design.b_in, column_design.h_in, beam_design.b_in, sp.BEAM_CLEAR_COVER_IN, _db(beam_design.stirrup_bar_size),
            db, n, bars, clear_spacing_limit("beam", db, sp.AGGREGATE_MAX_SIZE_IN))[axis]["bars_per_layer_that_fit"]
    return _CACHE[key]


def bars_per_layer_at_joint(level, i, j, beam_design, axis):
    """How many bars of this beam pass the joint at floor ``level``, grid (i, j) in one layer: through the joint's bar
    inventory where the column above's transition moves bars inside it (never more than the core cage admits; none
    where the transition declares no bar paths), else through the core cage (bars_per_layer)."""
    from Model import Member_Properties as mp
    core = mp.joint_core_column(level, i, j)
    own = bars_per_layer(core.design, beam_design, axis)
    inventory = joint_inventory(level, i, j)
    return own if inventory is None else min(own, inventory_lanes(inventory["bars"], core.design, beam_design, axis))


def joint_types():
    """The distinct joints of the installed design: (core column, X beam, Y beam) and where they occur.

    Each entry: {"column": ResolvedMember, "x": ResolvedMember | None, "y": ResolvedMember | None,
    "joints": [(level, i, j), ...]}. The two X beams (or Y beams) at an interior joint belong to one
    group by the grouping policy; a joint where they do not is refused, because bars of two cages would
    then meet inside the joint without a declared transition. The one declared transition is the core
    corner of the perimeter-core layout (2026-10-18): a core-ring beam and a gravity beam meet there, the
    bars of both terminate in the joint with standard hooks (18.8.2.2), and every cage pair of that joint
    is stacked, so each group's rows clear the crossing cages it actually meets.
    """
    from Model import Member_Properties as mp
    mixed_allowed = mg.active_layout() == mg.SFRS_LAYOUT_PERIMETER_CORE
    types = {}
    for level in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joint = mg.joint_members(level, i, j)
                candidates = {}
                for axis in ("x", "y"):
                    sides = [joint[f"beam_{axis}_{side}"] for side in ("minus", "plus") if joint[f"beam_{axis}_{side}"] is not None]
                    distinct = list({m.group_id: m for m in sides}.values())
                    if (len(distinct) > 1 or len({m.design for m in sides}) > 1) and not mixed_allowed:
                        raise mg.GroupedStateError(
                            f"Joint (level {level}, {i}, {j}): the two {axis.upper()} beams belong to different groups "
                            f"({sorted({m.group_id for m in sides})}); their bars cannot be taken as continuous through the "
                            "joint without a declared transition.")
                    candidates[axis] = distinct or [None]
                column = mp.joint_core_column(level, i, j)
                for bx in candidates["x"]:
                    for by in candidates["y"]:
                        key = (column.group_id, None if bx is None else bx.group_id, None if by is None else by.group_id)
                        entry = types.setdefault(key, {"column": column, "x": bx, "y": by, "joints": []})
                        entry["joints"].append((level, i, j))
    return list(types.values())


def _single_layer(design):
    base = nominal_offset_in(design)
    return {face: {"layers": 1, "per_layer": [count], "offsets_in": [base], "centroid_in": base}
            for face, count in (("top", design.top_bars), ("bottom", design.bot_bars))}


def slab_mats():
    """Selected mat centre planes shared by placement and the independent joint-clearance check.

    Returns (mats, complete). Cover is outside the outer mat, with touching orthogonal slab layers
    per the slab layout policy. Missing placement fields remain explicitly incomplete.
    """
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout") if sp.SLAB_THICKNESS_IN is not None else None
    layers = (layout or {}).get("layers") or {}
    mats, complete = [], bool(layers)
    for name, layer in sorted(layers.items()):
        if not name.endswith(("top", "bottom")):
            continue
        if any(k not in layer for k in ("bar_size", "clear_cover_outer_mat_in", "layer", "axis")):
            complete = False
            continue
        db = _db(layer["bar_size"])
        depth = layer["clear_cover_outer_mat_in"] + db / 2.0 + (0.0 if layer["layer"] == "outer" else db)
        if name.endswith("bottom"):
            depth = sp.SLAB_THICKNESS_IN - depth
        mats.append({"mat": name, "db_in": db, "centroid_from_top_in": depth, "runs_along": layer["axis"]})
    return mats, complete


def _place(sequence, offsets, depth_of, fixed=None, trace=None):
    """Sweep one joint's layers in order; raise each to its lowest admissible position.

    ``sequence``: [(group_id, layer_index, design)] from the face inward. ``offsets[(group, layer)]``
    are the current offsets from each beam's own face and only ever increase. ``depth_of[group]`` is
    how far that beam's face lies above the lowest face at this joint (0 at the top face, where all
    beams are flush). Returns True when something moved. ``trace`` = (controls, face, joint type): the
    constraint behind each raised offset is recorded there (row_controls); the placement is the same.
    """
    moved, placed = False, []
    fixed = fixed or {}
    for gid, layer, design in sequence:
        rise = depth_of[gid]
        position = rise + offsets[(gid, layer)]
        cause = None
        own = [(p, d) for g, _l, p, d in placed if g == gid]
        if own:
            lifted = max(p for p, _d in own) + pitch_in(design, design)
            if lifted > position:
                position, cause = lifted, {"by": "own_layer_pitch"}
        changed = True
        while changed:
            changed = False
            for g, other_layer, p, d in placed:
                if g != gid and abs(position - p) < pitch_in(design, d) - _TOL:
                    position, changed = p + pitch_in(design, d), True
                    cause = {"by": "crossing_layer", "group": g, "layer": other_layer}
            for p, db in fixed.get(gid, ()):
                clearance = 0.5 * (_db(design.bar_size) + db) + sp.BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN
                if abs(position - (rise + p)) < clearance - _TOL:
                    position = rise + sp.snap_up_to_grid(p + clearance)
                    changed = True
                    cause = {"by": "slab_mat", "mat_centroid_from_face_in": p, "mat_db_in": db}
        placed.append((gid, layer, position, design))
        if position - rise > offsets[(gid, layer)] + _TOL:
            offsets[(gid, layer)], moved = position - rise, True
            if trace is not None:
                controls, face, kind = trace
                controls[(face, gid, layer)] = {
                    **(cause or {"by": "unrecorded"}), "offset_in": position - rise,
                    "joint_type": {"column": kind["column"].group_id,
                                   "x": None if kind["x"] is None else kind["x"].group_id,
                                   "y": None if kind["y"] is None else kind["y"].group_id},
                    "joints": [list(j) for j in kind["joints"][:6]], "joint_count": len(kind["joints"])}
    return moved


def arrangement(controls=None):
    """{beam group id: {"top": rows, "bottom": rows}} for the installed design.

    Rows have the form of one face of Structure_Parameters.beam_bar_layers: layers, per_layer,
    offsets_in (from that face), centroid_in. Also returns, under "_basis", what was assumed.
    ``controls`` (a dict, row_controls) receives the constraint behind every raised offset; the
    arrangement is then solved afresh (never served from the cache) and is the same arrangement.
    """
    state = mg.active()
    if state is None:
        raise mg.GroupedStateError("Group bar layers describe an installed grouped design; none is installed.")
    convention = sp.beam_bar_stacking_convention()
    limit = sp.BEAM_BAR_MAX_LAYERS
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout") if sp.SLAB_THICKNESS_IN is not None else None
    mats, mats_complete = slab_mats()
    key = ("arrangement", state.identity(), convention, limit, sp.BEAM_BAR_LAYER_ORDER, sp.SLAB_THICKNESS_IN,
           sp.COL_CLEAR_COVER_IN, sp.BEAM_CLEAR_COVER_IN, sp.AGGREGATE_MAX_SIZE_IN, sp.COVER,
           sp.BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN, sp.BAR_POSITION_GRID_IN, sp.BAR_LAYOUT_HOOP_DB_IN, sp.FY_KSI,
           None if layout is None else layout.get("outer_axis"), mats_complete,
           tuple((m["mat"], m["db_in"], m["centroid_from_top_in"], m["runs_along"]) for m in mats))
    if controls is None and key in _CACHE:
        return _CACHE[key]
    beam_groups = {gid: state.designs[gid] for gid, group in state.groups.items() if group["member_type"] != "column"}
    if convention == "none" or sp.SLAB_THICKNESS_IN is None:
        result = {gid: _single_layer(design) for gid, design in beam_groups.items()}
        result["_basis"] = {"rule": RULE_ID, "convention": convention, "stacked": False, "sweeps": 0, "joint_types": 0}
        _CACHE[key] = result
        return result

    types = joint_types()
    # Lanes: the tightest cage each group meets.
    fit, inventory_effect = {}, []
    for kind in types:
        for axis in ("x", "y"):
            beam = kind[axis]
            if beam is None:
                continue
            # The lanes are measured whatever the layer limit (the threading check reads them); they
            # split the bars into layers only when more than one layer is allowed.
            core = n = bars_per_layer(kind["column"].design, beam.design, axis)
            # Where the column above has another cage whose transition moves bars inside the joint, the beam bars pass
            # the joint's inventory, not the core cage alone.
            for joint in kind["joints"]:
                at_joint = bars_per_layer_at_joint(*joint, beam.design, axis)
                if at_joint < core:
                    inventory = joint_inventory(*joint)
                    inventory_effect.append({"joint": list(joint), "axis": axis, "beam_group": beam.group_id,
                                             "core_cage_lanes": core, "joint_lanes": at_joint,
                                             "core_group": inventory["core_group"], "above_group": inventory["above_group"],
                                             "inventory_established": inventory["established"]})
                n = min(n, at_joint)
            fit[beam.group_id] = n if beam.group_id not in fit else min(fit[beam.group_id], n)
    per_layer = {gid: {face: sp._split_into_layers(count, fit.get(gid) if limit > 1 else None, limit)
                       for face, count in (("top", design.top_bars), ("bottom", design.bot_bars))}
                 for gid, design in beam_groups.items()}
    offsets = {face: {(gid, layer): nominal_offset_in(design) for gid, design in beam_groups.items()
                      for layer in range(len(per_layer[gid][face]))} for face in ("top", "bottom")}
    upper_axis, lower_axis = ("x", "y") if convention == "x_over_y" else ("y", "x")

    def sequence(kind, face):
        present = {axis: kind[axis] for axis in ("x", "y") if kind[axis] is not None}
        if len(present) == 1:
            (member,) = present.values()
            return ([(member.group_id, layer, member.design) for layer in range(len(per_layer[member.group_id][face]))],
                    {member.group_id: 0.0})
        if face == "top":
            near, far = present[upper_axis], present[lower_axis]
            depth_of = {near.group_id: 0.0, far.group_id: 0.0}
        else:
            hx, hy = present["x"].design.h_in, present["y"].design.h_in
            if abs(hx - hy) <= _TOL:
                near, far = present[lower_axis], present[upper_axis]
            else:
                near, far = (present["x"], present["y"]) if hx > hy else (present["y"], present["x"])
            deepest = max(hx, hy)
            depth_of = {m.group_id: deepest - m.design.h_in for m in (near, far)}
        near_slots, far_slots = sp.beam_bar_layer_slots(len(per_layer[near.group_id][face]), len(per_layer[far.group_id][face]))
        slots = sorted([(slot, near.group_id, layer, near.design) for layer, slot in enumerate(near_slots)]
                       + [(slot, far.group_id, layer, far.design) for layer, slot in enumerate(far_slots)])
        return [(gid, layer, design) for _slot, gid, layer, design in slots], depth_of

    sweeps = 0
    for face in ("top", "bottom"):
        plans = [sequence(kind, face) for kind in types]
        fixed = {gid: [(m["centroid_from_top_in"] if face == "top" else design.h_in - m["centroid_from_top_in"],
                        m["db_in"]) for m in mats if m["runs_along"] != state.groups[gid]["member_type"][-1]]
                 for gid, design in beam_groups.items()}
        for _ in range(MAX_SWEEPS):
            sweeps += 1
            if not any([_place(order, offsets[face], depth_of, fixed, None if controls is None else (controls, face, kind))
                        for (order, depth_of), kind in zip(plans, types)]):
                break
        else:
            raise mg.GroupedStateError(f"The {face} beam bar layers did not settle in {MAX_SWEEPS} sweeps over the joints; "
                                       "the groups' bars cannot be arranged by the declared rule.")
    result = {}
    for gid in beam_groups:
        result[gid] = {}
        for face in ("top", "bottom"):
            counts = per_layer[gid][face]
            rows = [offsets[face][(gid, layer)] for layer in range(len(counts))]
            result[gid][face] = {"layers": len(counts), "per_layer": counts, "offsets_in": rows,
                                 "centroid_in": sum(c * o for c, o in zip(counts, rows)) / sum(counts)}
    result["_basis"] = {"rule": RULE_ID, "convention": convention, "layer_order": sp.BEAM_BAR_LAYER_ORDER, "max_layers": limit,
                        "stacked": True, "sweeps": sweeps, "joint_types": len(types),
                        "bars_per_layer_governing": {gid: fit.get(gid) for gid in sorted(beam_groups)},
                        "interlayer_clear_min_in": sp.BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN,
                        "slab_mats": mats, "slab_mats_complete": mats_complete,
                        "slab_crossing_rule": "fixed mats; move straight beam rows inward on bar grid; "
                                              "interlayer gap is a declared construction allowance at orthogonal crossings"}
    if inventory_effect:
        # Recorded only where the joint inventory lowers a lane, so every other design's rows keep their earlier form.
        result["_basis"]["joint_inventory"] = {"rule": JOINT_INVENTORY_RULE_ID, "count": len(inventory_effect),
                                               "joints_with_fewer_lanes": inventory_effect[:24]}
    if len(_CACHE) > 8192:
        _CACHE.clear()
    _CACHE[key] = result
    return result


ROW_AUDIT_ID = "complete_arrangement_row_audit_v1"


def row_audit(state=None):
    """Every beam group's rows checked on the complete shared arrangement of ``state`` (default: the installed design):
    the arrangement of this module with the slab mats and settings in force, each face through
    Design.SMRF_Beam_Slab_Strength.validate_bar_rows (counts, increasing elevations inside the section, short of
    mid-depth). Returns {"rule", "state_identity", "valid", "invalid", "settle_error"}: each invalid face names its group,
    face, error ("past_mid_depth" or "rows_invalid"), reason, rows, section and bars and, from row_controls, what set each
    of its layers. An arrangement that does not settle is invalid with its reason (``settle_error``); it and a face whose
    row record is malformed ("rows_invalid") are errors a gate raises again (raise_arrangement_errors). Geometry only:
    nothing is changed."""
    from Design.SMRF_Beam_Slab_Strength import BeamBarRowsError, validate_bar_rows
    state = state or mg.active()
    if state is None:
        raise mg.GroupedStateError("A row audit describes a grouped design; none is given or installed.")
    result = {"rule": ROW_AUDIT_ID, "state_identity": state.identity(), "valid": True, "invalid": [], "settle_error": None}
    invalid = []
    with mg.installed(state):
        try:
            rows = arrangement()
        except mg.GroupedStateError as exc:
            result.update(valid=False, settle_error=str(exc))
            return result
        for gid in sorted(g for g, group in state.groups.items() if group["member_type"] != "column"):
            design = state.designs[gid]
            for face, count in (("top", design.top_bars), ("bottom", design.bot_bars)):
                try:
                    validate_bar_rows(rows[gid][face], count, f"{face} bars", design.h_in)
                    continue
                except BeamBarRowsError as exc:
                    error, reason = "past_mid_depth", str(exc)
                except ValueError as exc:
                    error, reason = "rows_invalid", str(exc)
                invalid.append({"group": gid, "face": face, "error": error, "reason": reason,
                                "per_layer": list(rows[gid][face]["per_layer"]),
                                "offsets_in": list(rows[gid][face]["offsets_in"]), "section_in": [design.b_in, design.h_in],
                                "bars": [design.bar_size, design.top_bars, design.bot_bars]})
        if invalid:
            controls = row_controls()
            for item in invalid:
                item["controls"] = controls[item["group"]][item["face"]]
    result.update(valid=not invalid, invalid=invalid)
    return result


def raise_arrangement_errors(audit):
    """Raise the errors a row audit caught that are not bar-row outcomes of the design: an arrangement that does not
    settle (layers that do not settle within MAX_SWEEPS, a joint whose beams differ without a declared transition) as
    GroupedStateError, and a malformed row record ("rows_invalid": counts, order or centroid that correct layering
    cannot produce) as ValueError, as when the strengths read the rows. Only rows past mid-depth are a design outcome."""
    if audit.get("settle_error"):
        raise mg.GroupedStateError(audit["settle_error"])
    malformed = [item for item in audit.get("invalid") or [] if item.get("error") == "rows_invalid"]
    if malformed:
        raise ValueError("malformed beam bar rows (an error in the layering or its inputs, not a design outcome): "
                         + "; ".join(f"{item['group']} {item['face']}: {item['reason']}" for item in malformed[:4]))


def row_controls():
    """{beam group id: {"top": [...], "bottom": [...]}} of the installed design: per layer, None where the layer kept
    its nominal offset, else the last constraint that raised it (``by``: own_layer_pitch, crossing_layer with the
    crossing group and layer, or slab_mat), its offset from the face and the joint type (core column, X and Y beam
    groups) and joints where it did. The arrangement is that of arrangement(); this explains it, it changes nothing."""
    controls = {}
    rows = arrangement(controls=controls)
    return {gid: {face: [controls.get((face, gid, layer)) for layer in range(value[face]["layers"])]
                  for face in ("top", "bottom")}
            for gid, value in rows.items() if gid != "_basis"}


def group_rows(group_id):
    """{"top": rows, "bottom": rows} of one beam group in the installed design."""
    rows = arrangement().get(group_id)
    if rows is None or group_id == "_basis":
        raise mg.GroupedStateError(f"{group_id!r} is not a beam group of the installed design.")
    return rows


def member_rows(member):
    return group_rows(member.group_id)


def centroid_offsets_in(member):
    rows = member_rows(member)
    return {face: rows[face]["centroid_in"] for face in ("top", "bottom")}
