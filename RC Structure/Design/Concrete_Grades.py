"""Concrete grades by structural role in the grouped design engine (2026-10-23).

A grade family is a set of member groups that hold one specified concrete strength (f'c) over every story band. The
search moves a family one rung of its own ladder at a time (Design.Grouped_Search), and only for a failure of one of
its own groups:

  band policy (every frame is part of the seismic force-resisting system, SFRS)
      column            Section_Design.COLUMN_CONCRETE_STRENGTHS_KSI, from the seed's column grade (as before)
      beam              Section_Design.BEAM_CONCRETE_STRENGTHS_KSI, from the seed's beam grade (as before)
  two-system policies (perimeter frames, and the core ring or core lines the layout ladder adds)
      sfrs_column       the special moment frame column ladder (6-10 ksi), from the seed's column grade
      gravity_column    Section_Design.GRAVITY_COLUMN_CONCRETE_STRENGTHS_KSI (5, 6, 7 ksi), from 5 ksi, never above 7
      beam              every beam, special-frame and gravity, on the special moment frame beam ladder (5, 6, 8 ksi),
                        from the seed's beam grade

The gravity ladder is the user decision DEC-SFV6-GRAVITY-CONCRETE-20261023. The system of a group is the one its
grouping policy assigns from the active layout (Model.Member_Groups): a column the core ring or the core lines bring
into the SFRS is an sfrs_column under that policy and starts from the special-frame seed grade. A family is one grade
over the full height, so a column line keeps one grade (Design.SMRF_Transitions requires that at band boundaries).

Gravity beams keep the shared beam grade for now. The same decision gives them the 5-7 ksi ladder, but the floor model
casts the slab with the beams: the one beam grade of a floor is the slab's grade (Design.SMRF_Floor_Sections,
Design.SMRF_Slab), and each T or L flange takes its own web's concrete (Model.Member_Properties,
Design.SMRF_Beam_Slab_Strength). Every floor carries special-frame beams on its perimeter beside gravity beams inside,
so two beam grades on one floor need a declared slab grade, pour boundary and flange-concrete rule. That rule is not
decided; until it is, ``SEPARATE_GRAVITY_BEAMS`` stays False and a record states whether the shared beam grade lies
on the gravity ladder (``gravity_beams`` in ``describe``).

Joint concrete is unchanged: joint shear and hook development use the f'c of the column below the joint at every
joint (Model.Member_Properties.JOINT_CORE_CONVENTION), and column-strength concrete is declared placed in the floor at
the column where the column's f'c exceeds 1.4 times the floor's (ACI 318-19 15.5(a), user decision 2026-10-03). A
gravity column frames only gravity beams in every layout, and 7 ksi is 1.4 times a 5 ksi floor.

Records written before this module carry no ``concrete_grade_policy``; they are read under the earlier rule, one
grade per member type (``LEGACY_POLICY_ID``).
"""
from __future__ import annotations

from dataclasses import replace

import Structure_Parameters as sp
from Design.Section_Design import (BEAM_CONCRETE_STRENGTHS_KSI, COLUMN_CONCRETE_STRENGTHS_KSI,
                                   GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI, GRAVITY_COLUMN_CONCRETE_STRENGTHS_KSI)
from Model import Member_Groups as mg

LEGACY_POLICY_ID = "one_grade_per_member_type_v0"
DECISION_ID = "DEC-SFV6-GRAVITY-CONCRETE-20261023"
# Gravity beams on their own ladder need the slab, pour-boundary and flange-concrete rule described above.
SEPARATE_GRAVITY_BEAMS = False
# Every grade policy a record may name, with the definition it stands for. A record is read under the definition its
# id names, so changing a ladder or SEPARATE_GRAVITY_BEAMS needs a new entry and id: the module refuses to load when the
# current definition has none (see POLICY_ID below), instead of reading saved records under another definition.
KNOWN_POLICIES = {
    "role_concrete_grades_v1": {"separate_gravity_beams": False,
                                "special_frame_column_ladder_ksi": (6.0, 7.0, 8.0, 9.0, 10.0),
                                "special_frame_beam_ladder_ksi": (5.0, 6.0, 8.0),
                                "gravity_column_ladder_ksi": (5.0, 6.0, 7.0),
                                "gravity_beam_ladder_ksi": (5.0, 6.0, 7.0)},
}


def _current_definition():
    return {"separate_gravity_beams": SEPARATE_GRAVITY_BEAMS,
            "special_frame_column_ladder_ksi": tuple(float(v) for v in COLUMN_CONCRETE_STRENGTHS_KSI),
            "special_frame_beam_ladder_ksi": tuple(float(v) for v in BEAM_CONCRETE_STRENGTHS_KSI),
            "gravity_column_ladder_ksi": tuple(float(v) for v in GRAVITY_COLUMN_CONCRETE_STRENGTHS_KSI),
            "gravity_beam_ladder_ksi": tuple(float(v) for v in GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI)}


_matching = [pid for pid, definition in KNOWN_POLICIES.items() if definition == _current_definition()]
if len(_matching) != 1:
    raise RuntimeError("The concrete grade ladders or SEPARATE_GRAVITY_BEAMS match no entry of KNOWN_POLICIES: add a new "
                       "policy id for the new definition (saved records are read under the definition their id names).")
POLICY_ID = _matching[0]
# Fields of a saved concrete_grades block that are explanation, not state: a wording change keeps a record readable.
PROSE_KEYS = {"basis", "rule", "joint_concrete_basis", "slab_concrete_basis"}
GRAVITY_BEAM_RULE = ("gravity beams share the special-frame beam grade: the slab is cast with the beams and each T/L "
                     "flange takes its web's concrete, so two beam grades on one floor need a declared slab grade, pour "
                     "boundary and flange-concrete rule, which " + DECISION_ID + " leaves open")
JOINT_CONCRETE_BASIS = ("joint shear and hook development use the column below the joint (joint_core_takes_the_column_below_v1); "
                        "column-strength concrete is declared placed in the floor at the column where the column's f'c "
                        "exceeds 1.4 times the floor's (ACI 318-19 15.5(a), user decision 2026-10-03)")


def _family(member_type, systems, ladder, start, basis):
    return {"member_type": member_type, "systems": tuple(systems), "ladder": tuple(float(v) for v in ladder),
            "start_ksi": None if start is None else float(start), "basis": basis}


def families(policy_id):
    """{family: {member_type, systems, ladder, start_ksi, basis}} of a grouping policy. ``start_ksi`` None means the
    family starts from the seed's grade; a value means every new search starts that family there."""
    if not mg.is_system_policy(policy_id):
        return {"column": _family("column", (mg.SYSTEM_SFRS,), COLUMN_CONCRETE_STRENGTHS_KSI, None,
                                  "band policy: every column is part of the SFRS"),
                "beam": _family("beam", (mg.SYSTEM_SFRS,), BEAM_CONCRETE_STRENGTHS_KSI, None,
                                "band policy: every beam is part of the SFRS")}
    result = {"sfrs_column": _family("column", (mg.SYSTEM_SFRS,), COLUMN_CONCRETE_STRENGTHS_KSI, None,
                                     "special moment frame columns of the active layout"),
              "gravity_column": _family("column", (mg.SYSTEM_GRAVITY,), GRAVITY_COLUMN_CONCRETE_STRENGTHS_KSI,
                                        GRAVITY_COLUMN_CONCRETE_STRENGTHS_KSI[0],
                                        "columns outside the SFRS of the active layout (" + DECISION_ID + ")")}
    if SEPARATE_GRAVITY_BEAMS:
        result["sfrs_beam"] = _family("beam", (mg.SYSTEM_SFRS,), BEAM_CONCRETE_STRENGTHS_KSI, None,
                                      "special moment frame beams of the active layout")
        result["gravity_beam"] = _family("beam", (mg.SYSTEM_GRAVITY,), GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI,
                                         GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI[0],
                                         "beams outside the SFRS of the active layout (" + DECISION_ID + ")")
    else:
        result["beam"] = _family("beam", (mg.SYSTEM_SFRS, mg.SYSTEM_GRAVITY), BEAM_CONCRETE_STRENGTHS_KSI, None,
                                 GRAVITY_BEAM_RULE)
    return result


def family_of_group(policy_id, group):
    """The grade family of one group row of a policy's assignments."""
    kind = "column" if group["member_type"] == "column" else "beam"
    known = families(policy_id)
    if kind in known:
        return kind
    name = f"{group.get('system')}_{kind}"
    if name not in known:
        raise mg.GroupedStateError(f"Group {group.get('group_id')!r} has no concrete grade family under {policy_id!r} "
                                   f"(system {group.get('system')!r}).")
    return name


def family_of(state, gid):
    return family_of_group(state.policy_id, state.groups[gid])


def groups_of(state, family):
    """The group ids of one family (empty when the layout has none, e.g. a core ring that takes every interior column)."""
    if family not in families(state.policy_id):
        raise mg.GroupedStateError(f"{family!r} is not a concrete grade family under {state.policy_id!r} "
                                   f"(families {sorted(families(state.policy_id))}).")
    return [gid for gid in sorted(state.groups) if family_of(state, gid) == family]


def grade(state, family):
    """The one grade of a family, or None when the family has no group. Raises when its groups differ."""
    values = {state.designs[gid].fc_ksi for gid in groups_of(state, family)}
    if len(values) > 1:
        raise mg.GroupedStateError(f"The {family} groups use different concrete grades {sorted(values)}; a family keeps one "
                                   "grade over the full height.")
    return values.pop() if values else None


def next_grade(state, family):
    """The next rung of the family's ladder above its grade; None at the top or for a family without groups (a grade
    below the ladder, a legacy seed, steps to the first rung)."""
    current = grade(state, family)
    if current is None:
        return None
    return next((fc for fc in families(state.policy_id)[family]["ladder"] if fc > current + 1e-9), None)


def regraded(state, family, fc):
    """Every group of one family at grade ``fc``; the other families are untouched."""
    gids = groups_of(state, family)
    if not gids:
        return state
    return state.with_designs({gid: replace(state.designs[gid], fc_ksi=float(fc)) for gid in gids})


def seeded(state):
    """A new search's starting grades: each family with a declared start (the gravity ladder's 5 ksi) is set to it;
    the others keep the seed's grade."""
    for family, spec in families(state.policy_id).items():
        if spec["start_ksi"] is not None:
            state = regraded(state, family, spec["start_ksi"])
    return state


def problems(state, policy=POLICY_ID):
    """Why a grouped design's grades break a grade policy (empty when they hold).

    ``POLICY_ID``: one grade per family over the full height, and a family with a declared start (the gravity ladder)
    on its own ladder. ``LEGACY_POLICY_ID``: one column grade and one beam grade for the whole frame."""
    issues = []
    if policy == LEGACY_POLICY_ID:
        for kind in ("column", "beam"):
            values = {d.fc_ksi for gid, d in state.designs.items() if (state.groups[gid]["member_type"] == "column") == (kind == "column")}
            if len(values) > 1:
                issues.append(f"{kind} groups use different concrete grades {sorted(values)}; a record without a concrete "
                              f"grade policy keeps one {kind} grade for the whole frame")
        return issues
    if policy != POLICY_ID:
        return [f"unknown concrete grade policy {policy!r} (known: {POLICY_ID}, {LEGACY_POLICY_ID})"]
    for family, spec in families(state.policy_id).items():
        values = sorted({state.designs[gid].fc_ksi for gid in groups_of(state, family)})
        if len(values) > 1:
            issues.append(f"the {family} groups use different concrete grades {values}; a family keeps one grade over the "
                          "full height")
        elif values and spec["start_ksi"] is not None and not any(abs(values[0] - fc) <= 1e-9 for fc in spec["ladder"]):
            issues.append(f"the {family} grade {values[0]:g} ksi is outside its ladder {list(spec['ladder'])} ksi "
                          f"({spec['basis']})")
    return issues


def require(state, policy=POLICY_ID):
    issues = problems(state, policy)
    if issues:
        raise mg.GroupedStateError("Concrete grades are not admissible: " + "; ".join(issues) + ".")


def _gravity_beams(state, families_):
    """Whether the gravity beams hold a grade on the gravity ladder (the shared beam grade until a slab rule exists)."""
    gids = [gid for gid, g in state.groups.items()
            if mg.is_system_policy(state.policy_id) and g["member_type"] != "column" and state.system_of(gid) == mg.SYSTEM_GRAVITY]
    if not gids:
        return {"groups": 0, "fc_ksi": None, "within_gravity_ladder": None,
                "rule": "no gravity beams" if mg.is_system_policy(state.policy_id) else "band policy: no gravity system"}
    values = sorted({state.designs[gid].fc_ksi for gid in gids})
    within = all(any(abs(v - fc) <= 1e-9 for fc in GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI) for v in values)
    return {"groups": len(gids), "fc_ksi": values[0] if len(values) == 1 else values,
            "within_gravity_ladder": within, "gravity_ladder": list(GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI),
            "rule": "own family on the gravity ladder" if "gravity_beam" in families_ else GRAVITY_BEAM_RULE}


def describe(state):
    """The concrete grades of a grouped design by family, for its record and reports (no first-grade pick, no mean)."""
    known = families(state.policy_id)
    rows = {}
    for family, spec in known.items():
        gids = groups_of(state, family)
        fc = grade(state, family)
        rows[family] = {"member_type": spec["member_type"], "systems": list(spec["systems"]), "groups": len(gids),
                        "fc_ksi": fc, "ladder": list(spec["ladder"]), "start_ksi": spec["start_ksi"],
                        "at_top_rung": None if fc is None else fc >= spec["ladder"][-1] - 1e-9, "basis": spec["basis"]}
    return {"policy": POLICY_ID, "grouping_policy": state.policy_id, "families": rows,
            "gravity_beams": _gravity_beams(state, known), "joint_concrete_basis": JOINT_CONCRETE_BASIS,
            "slab_concrete_basis": "the slab is cast with the beams: its grade is the shared beam grade of the floor"}


def family_grades(policy_id, sections):
    """{family: sorted grades} of a section table {group id: {fc_ksi, ...}} written under ``policy_id`` (a layout
    attempt's last candidate), from that policy's assignments for the current geometry."""
    assignments = mg.build_assignments(sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR, policy_id)
    result = {}
    for group in assignments["groups"]:
        row = sections.get(group["group_id"])
        if row is not None:
            result.setdefault(family_of_group(policy_id, group), set()).add(float(row["fc_ksi"]))
    return {family: sorted(values) for family, values in sorted(result.items())}


def identity_block():
    """The declared grade policy, for the request identity: a record made under another policy is another request."""
    return {"policy": POLICY_ID, "decision": DECISION_ID,
            "special_frame_column_ladder_ksi": list(COLUMN_CONCRETE_STRENGTHS_KSI),
            "special_frame_beam_ladder_ksi": list(BEAM_CONCRETE_STRENGTHS_KSI),
            "gravity_column_ladder_ksi": list(GRAVITY_COLUMN_CONCRETE_STRENGTHS_KSI),
            "gravity_beam_ladder_ksi": list(GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI),
            "separate_gravity_beams": SEPARATE_GRAVITY_BEAMS,
            "gravity_beam_rule": None if SEPARATE_GRAVITY_BEAMS else GRAVITY_BEAM_RULE,
            "one_grade_per_family_over_the_full_height": True}


def policy_of_record(record):
    """The grade policy a saved grouped record was written under (the legacy rule when it names none)."""
    return ((record.get("materials") or {}).get("concrete_grade_policy")) or LEGACY_POLICY_ID


def structural(block):
    """A concrete_grades block without its explanatory text (PROSE_KEYS), at any depth: what install compares."""
    if isinstance(block, dict):
        return {key: structural(value) for key, value in block.items() if key not in PROSE_KEYS}
    if isinstance(block, (list, tuple)):
        return [structural(value) for value in block]
    return block


def readable_policy_problem(record):
    """Why this code cannot read a saved record's grades under the policy it names, or None (the current policy and
    the legacy rule are readable)."""
    policy = policy_of_record(record)
    if policy in (POLICY_ID, LEGACY_POLICY_ID):
        return None
    return (f"the record was written under concrete grade policy {policy!r}; this code reads {POLICY_ID!r} and records "
            f"without a policy ({LEGACY_POLICY_ID}); design the case again under the current policy")
