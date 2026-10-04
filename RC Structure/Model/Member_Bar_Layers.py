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

What this module does not decide: slab mats against the top layers, hook tails, and whether the rows
fit the section (Design.SMRF_Cage_Geometry / the capacity checks, and validate_bar_rows).
"""
from __future__ import annotations

import Structure_Parameters as sp
from Model import Member_Groups as mg

RULE_ID = "group_bar_layers_v1__straight_bars_one_arrangement_per_group__deeper_soffit_first"
MAX_SWEEPS = 64
_TOL = 1e-9
_CACHE = {}


def _db(size):
    return sp.rebar_diameter(size)


def nominal_offset_in(design):
    """Face to the centroid of a first-layer bar: clear cover + hoop + half the bar."""
    return sp.longitudinal_cover_in("beam", design.bar_size, design.stirrup_bar_size)


def pitch_in(design_a, design_b):
    """Centre-to-centre distance between two layers (of one beam or of two crossing beams)."""
    da, db = _db(design_a.bar_size), _db(design_b.bar_size)
    return 0.5 * (da + db) + max(sp.BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN, da, db)


def bars_per_layer(column_design, beam_design, axis):
    """How many bars of this beam pass this column cage in one layer, in the beam's own direction."""
    from Design.SMRF_Cage_Geometry import column_cage_admits_beam_bars
    n = max(beam_design.top_bars, beam_design.bot_bars)
    key = ("fit", column_design.section_key(), beam_design.section_key(), axis, sp.COL_CLEAR_COVER_IN,
           sp.BEAM_CLEAR_COVER_IN, sp.AGGREGATE_MAX_SIZE_IN)
    if key not in _CACHE:
        if len(_CACHE) > 8192:
            _CACHE.clear()
        _CACHE[key] = column_cage_admits_beam_bars(
            column_design.b_in, column_design.h_in, sp.COL_CLEAR_COVER_IN, _db(column_design.stirrup_bar_size),
            _db(column_design.bar_size), column_design.top_bars, column_design.side_bars, beam_design.b_in,
            sp.BEAM_CLEAR_COVER_IN, _db(beam_design.stirrup_bar_size), _db(beam_design.bar_size), n,
            sp.AGGREGATE_MAX_SIZE_IN)["bars_per_layer"][axis]
    return _CACHE[key]


def joint_types():
    """The distinct joints of the installed design: (core column, X beam, Y beam) and where they occur.

    Each entry: {"column": ResolvedMember, "x": ResolvedMember | None, "y": ResolvedMember | None,
    "joints": [(level, i, j), ...]}. The two X beams (or Y beams) at an interior joint belong to one
    group by the grouping policy; a joint where they do not is refused, because bars of two cages would
    then meet inside the joint without a declared transition.
    """
    from Model import Member_Properties as mp
    types = {}
    for level in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joint = mg.joint_members(level, i, j)
                beams = {}
                for axis in ("x", "y"):
                    sides = [joint[f"beam_{axis}_{side}"] for side in ("minus", "plus") if joint[f"beam_{axis}_{side}"] is not None]
                    if len({m.group_id for m in sides}) > 1 or len({m.design for m in sides}) > 1:
                        raise mg.GroupedStateError(
                            f"Joint (level {level}, {i}, {j}): the two {axis.upper()} beams belong to different groups "
                            f"({sorted({m.group_id for m in sides})}); their bars cannot be taken as continuous through the "
                            "joint without a declared transition.")
                    beams[axis] = sides[0] if sides else None
                column = mp.joint_core_column(level, i, j)
                key = (column.group_id, None if beams["x"] is None else beams["x"].group_id,
                       None if beams["y"] is None else beams["y"].group_id)
                entry = types.setdefault(key, {"column": column, "x": beams["x"], "y": beams["y"], "joints": []})
                entry["joints"].append((level, i, j))
    return list(types.values())


def _single_layer(design):
    base = nominal_offset_in(design)
    return {face: {"layers": 1, "per_layer": [count], "offsets_in": [base], "centroid_in": base}
            for face, count in (("top", design.top_bars), ("bottom", design.bot_bars))}


def _place(sequence, offsets, depth_of):
    """Sweep one joint's layers in order; raise each to its lowest admissible position.

    ``sequence``: [(group_id, layer_index, design)] from the face inward. ``offsets[(group, layer)]``
    are the current offsets from each beam's own face and only ever increase. ``depth_of[group]`` is
    how far that beam's face lies above the lowest face at this joint (0 at the top face, where all
    beams are flush). Returns True when something moved.
    """
    moved, placed = False, []
    for gid, layer, design in sequence:
        rise = depth_of[gid]
        position = rise + offsets[(gid, layer)]
        own = [(p, d) for g, _l, p, d in placed if g == gid]
        if own:
            position = max(position, max(p for p, _d in own) + pitch_in(design, design))
        changed = True
        while changed:
            changed = False
            for g, _l, p, d in placed:
                if g != gid and abs(position - p) < pitch_in(design, d) - _TOL:
                    position, changed = p + pitch_in(design, d), True
        placed.append((gid, layer, position, design))
        if position - rise > offsets[(gid, layer)] + _TOL:
            offsets[(gid, layer)], moved = position - rise, True
    return moved


def arrangement():
    """{beam group id: {"top": rows, "bottom": rows}} for the installed design.

    Rows have the form of one face of Structure_Parameters.beam_bar_layers: layers, per_layer,
    offsets_in (from that face), centroid_in. Also returns, under "_basis", what was assumed.
    """
    state = mg.active()
    if state is None:
        raise mg.GroupedStateError("Group bar layers describe an installed grouped design; none is installed.")
    convention = sp.beam_bar_stacking_convention()
    limit = sp.BEAM_BAR_MAX_LAYERS
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout") if sp.SLAB_THICKNESS_IN is not None else None
    key = ("arrangement", state.identity(), convention, limit, sp.BEAM_BAR_LAYER_ORDER, sp.SLAB_THICKNESS_IN,
           sp.COL_CLEAR_COVER_IN, sp.BEAM_CLEAR_COVER_IN, sp.AGGREGATE_MAX_SIZE_IN, sp.COVER,
           sp.BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN, None if layout is None else layout.get("outer_axis"))
    if key in _CACHE:
        return _CACHE[key]
    beam_groups = {gid: state.designs[gid] for gid, group in state.groups.items() if group["member_type"] != "column"}
    if convention == "none" or sp.SLAB_THICKNESS_IN is None:
        result = {gid: _single_layer(design) for gid, design in beam_groups.items()}
        result["_basis"] = {"rule": RULE_ID, "convention": convention, "stacked": False, "sweeps": 0, "joint_types": 0}
        _CACHE[key] = result
        return result

    types = joint_types()
    # Lanes: the tightest cage each group meets.
    fit = {}
    for kind in types:
        for axis in ("x", "y"):
            beam = kind[axis]
            if beam is None:
                continue
            # The lanes are measured whatever the layer limit (the threading check reads them); they
            # split the bars into layers only when more than one layer is allowed.
            n = bars_per_layer(kind["column"].design, beam.design, axis)
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
        for _ in range(MAX_SWEEPS):
            sweeps += 1
            if not any([_place(order, offsets[face], depth_of) for order, depth_of in plans]):
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
                        "interlayer_clear_min_in": sp.BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN}
    if len(_CACHE) > 8192:
        _CACHE.clear()
    _CACHE[key] = result
    return result


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
