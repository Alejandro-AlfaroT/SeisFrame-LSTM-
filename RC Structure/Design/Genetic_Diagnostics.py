"""Normalized mutation diagnostics of one evaluated grouped design (genetic search, 2026-10-30).

The six mutation operators (Design.Genetic_Operators) read failures through this adapter only. It runs once per
evaluation, while the evaluator's in-memory record is still at hand, and keeps what routing needs:

  context      the parent the diagnostics belong to: its realized identity (cages and slab included), the run's
               evaluation basis, the declared genes and the parent's alleles, every group's physical description,
               and the joint topology the record's own strong-column and joint evidence was computed on.
  diagnostics  one entry per failed check row of a mapped source (and per failed strength component), with the
               original family / id / location, the declared comparison and units, the demand and capacity, the
               physical target (group, joint or joint type, axis, sway sign, story) and the dependencies its operator
               needs, each derived from the bound record (rows, completion flags, capacity and strength evidence).
  unknown      the rows that were not evaluated, grouped by check and reason code: evidence to obtain, never a
               reason to strengthen.

Nothing is trusted that the contract assessment (Design.Code_Check_Contract.assess_evaluation) did not validate:
an invalid record yields no diagnostics. A dependency is True only when the record shows it; a missing or malformed
item makes it False, and the router then abstains. No flag is set to obtain a proposal, nothing here is synthetic,
and a diagnostic grants no acceptance. Failures without a mapped source (and the analysis-shear component of a
strength row) are kept as unsupported diagnostics: they are reported, and the router abstains on them.

v2 (2026-10-30 review): a beam strength shortfall of a lane-limited group (the cages that reach the demand do not
thread the column, Group_Selection) is a bar-room deficit (beam_lane_limited_strength, H3), not a strength deficit;
the gravity column capacity-shear row is mapped; capacity-shear transverse selection needs the hoop selection to
have settled with a recorded outcome (an exhausted hoop ladder is an outcome); a column capacity-shear diagnostic
records whether the selected cage is the lightest the section admits (its ratio set by the minimum, not demand).

v3 (2026-10-30 second review): the gravity column capacity-shear row is authenticated against the producer's
per-direction record (a row that compares one direction's Ve with another direction's phi Vn is invalid evidence
for routing); a cage lighter than every admissible cage leaves the ratio-floor flag unknown; a lane-limited
shortfall's column levers come from the joint topology and follow its consistency (the record's topology_problems).

v4 (2026-10-30 fourth review round): the ratio-floor flag is read only on a settled longitudinal selection whose Ve
the probable-moment mechanism sets (not the analysis shear); an unevaluated gravity column capacity-shear row of a
column without selected hoops carries the reason the producer's own record shows (hoops_not_selected) instead of none.

v5 (2026-10-31 fifth review round): the mechanism test of the ratio-floor flag also requires the column-own Ve
method on the governing entry (the only method the grouped route builds; under another method the beams can set Ve
below the column's own envelope); hoops_not_selected is given only when the governing Ve is known, so the missing
hoops alone explain the unevaluated row.

v6 (2026-10-31 lower-perimeter cage audit): a beam flexural shortfall whose strength threading constrains (a stronger
admissible cage does not thread the column cages and the strongest threading cage still falls short; the selection's
threading evidence, Group_Selection) is beam_threading_constrained_strength (H3): the beam's own levers and the cores
that block the stronger cage, with the selection's geometry-only look-ahead of the core size. It is authenticated against
the record (constraints.beam_threading_evidence) and the group's final cage; without that the router abstains. A
lane-limited shortfall keeps its own meaning and route; an unlisted one stays a strength deficit (H2).

v7 (2026-10-31 cage-coordination review): the threading evidence keeps a core cage's own blockage apart from a column
transition's (a band-boundary joint whose bar inventory blocks the stronger cage, named by both column groups). A core
size target is read only from a look-ahead entry of scope core_cage; a transition's targets only from an entry of scope
transition_joint, for the band it resizes (the column above or the core below) at that joint's own inventory. An entry
without a scope gives no target. The cores whose own cage blocks are carried as blocking_core_cages.

v8 (2026-10-31 follow-up review): the same attribution reaches the other two triggers. A failed beam.bars_thread_column row
(beam_threading) states from its own lanes whether the core cage blocks (core_cage_blocks) and whether the transition
to the column above does (transition_blocks, with column_above_group checked against the joint topology); the
installed-cage lane evidence (constraints.beam_lane_evidence: this state's identity, the group's final cage) adds the
look-ahead targets of that joint, without a flexural shortfall. A lane-limited shortfall carries the selection's
threading fields when they authenticate. The router then grows a core only where its own cage blocks.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import Structure_Parameters as sp
from Design import Code_Check_Contract as contract
from Design.SMRF_Capacity_Design import COLUMN_SHEAR_METHOD_COLUMN_OWN
from Design.SMRF_Common import action_inventory_problems
from Design.SMRF_Joints import SLAB_BASES
from Model import Member_Groups as mg

VERSION = "seisframe_ga_diagnostic_adapter_v8"
# reason: (operator, required dependencies) -- the research operator contract (six_operator_contract_reference_v1).
RULES = {
    "scwb": ("H1", ("nominal_column_strength", "nominal_beam_slab_strength", "axial_case_coverage")),
    "beam_flexure": ("H2", ("member_actions", "section_strength")),
    "column_pm": ("H2", ("member_actions", "section_strength")),
    "beam_capacity_shear": ("H2", ("probable_equilibrium", "transverse_selection")),
    "column_capacity_shear": ("H2", ("probable_equilibrium", "transverse_selection")),
    "beam_width_room": ("H3", ("cage_geometry",)),
    "beam_row_depth": ("H3", ("cage_geometry",)),
    "beam_lane_limited_strength": ("H3", ("member_actions", "section_strength", "lane_limited_selection")),
    "beam_threading_constrained_strength": ("H3", ("member_actions", "section_strength", "threading_constrained_selection")),
    "beam_threading": ("H3", ("cage_geometry", "joint_incidence")),
    "slab_mat_overlap": ("H3", ("cage_geometry", "slab_layout", "diagnosed_dimension")),
    "joint_shear": ("H4", ("joint_classification", "joint_actions", "joint_capacity")),
    "through_bar_depth": ("H4", ("bar_geometry", "joint_incidence")),
    "hook_embedment": ("H4", ("bar_geometry", "joint_incidence")),
    "story_drift": ("H5", ("drift_analysis", "story_location", "limit_basis")),
    "story_stability": ("H5", ("stability_analysis", "story_location", "limit_basis")),
}
# (family, check id): (reason, comparison, raw units, declared units). The comparison and units are authenticated
# against the row; a row that differs is invalid evidence for routing. member.strength is split by component below.
SOURCES = {
    ("joints", "scwb"): ("scwb", "<=", "kip-in", "kip-in"),
    ("capacity", "beam.capacity_shear_section"): ("beam_capacity_shear", "<=", "kip", "kip"),
    ("joints", "beam_capacity_shear_left"): ("beam_capacity_shear", "<=", "kip", "kip"),
    ("joints", "beam_capacity_shear_right"): ("beam_capacity_shear", "<=", "kip", "kip"),
    ("capacity", "gravity.beam_capacity_shear"): ("beam_capacity_shear", "<=", "kip", "kip"),
    ("capacity", "column.capacity_shear_section"): ("column_capacity_shear", "<=", "kip", "kip"),
    ("capacity", "gravity.column_capacity_shear"): ("column_capacity_shear", "<=", "kip", "kip"),
    ("detailing", "beam.bar_clear_spacing"): ("beam_width_room", ">=", "in", "in"),
    ("capacity", "beam.bars_thread_column"): ("beam_threading", "==", "", "predicate"),
    ("capacity", "beam.bar_stacking_clear_of_slab_mats"): ("slab_mat_overlap", "==", "", "predicate"),
    ("joints", "joint_shear"): ("joint_shear", "<=", "kip", "kip"),
    ("capacity", "joint.shear_screen"): ("joint_shear", "<=", "kip", "kip"),
    ("joints", "joint_through_bar_depth"): ("through_bar_depth", "<=", "in", "in"),
    ("capacity", "joint.through_bar_depth"): ("through_bar_depth", "<=", "in", "in"),
    ("capacity", "joint.terminating_bar_hook"): ("hook_embedment", "<=", "in", "in"),
    ("drift", "demands.story_drift"): ("story_drift", "<=", "in/in", "in/in"),
    ("drift", "demands.stability"): ("story_stability", "<=", "dimensionless", "dimensionless"),
}
STRENGTH_SOURCE = ("strength", "member.strength")
# Reasons of the operator contract without a production source in this evaluator, with the reason.
UNMAPPED = {
    "beam_row_depth": ("bar rows that do not fit beside the crossing beams stop the evaluation at stage 'bar_rows' "
                       "(BeamBarRowsError); that record has no code check, so it is never a selection parent"),
}
RELATIVE_TOLERANCE = 1e-9
_JOINT_TYPE_AXIS = re.compile(r"^(?P<type>[^/]+)/(?P<axis>[xy])$")
_JOINT_SHEAR = re.compile(r"^joint_shear/(?P<type>[^/]+)/(?P<axis>[xy])$")
_SCWB = re.compile(r"^(?P<joint>[^/]+)/(?P<axis>[xy])/(?P<sign>positive|negative)$")
_STORY = re.compile(r"^story:(?P<story>\d+)/Q(?P<load>[xy])/(?P<axis>[xy])$")
_CLASH_GROUP = re.compile(r"\(([^()]+)\)$")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _close(a, b):
    return finite(a) and finite(b) and abs(a - b) <= RELATIVE_TOLERANCE * max(1.0, abs(a), abs(b))


# ---- context -----------------------------------------------------------------------------------------------------
def gene_rows(space):
    """The declared genes as plain rows (sizes as (b, h) tuples, grades as floats)."""
    return [{"name": g.name, "kind": g.kind, "groups": list(g.groups),
             "options": [tuple(float(x) for x in o) if g.kind == "size" else float(o) for o in g.options]}
            for g in space.genes]


def group_rows(state):
    """Every group's physical description. Under the band policy every member belongs to the SFRS."""
    system_policy = mg.is_system_policy(state.policy_id)
    result = {}
    for gid, group in sorted(state.groups.items()):
        result[gid] = {"member_type": group["member_type"],
                       "system": (state.system_of(gid) if system_policy else mg.SYSTEM_SFRS),
                       "stories_or_floors": [int(s) for s in group["stories_or_floors"]],
                       "band": group.get("band"), "location_class": group.get("location_class"),
                       "design_tie": group.get("design_tie"), "member_tags": sorted(int(t) for t in group["member_tags"])}
    return result


def joint_topology(capacity):
    """(joints, joint types, problems) from the capacity evidence of the evaluated design.

    joints {joint id: {floor, type_id, is_roof, core_group, columns, beams {axis: groups}, required {axis}}} from the
    strong-column records (the columns and beams actually framing into each joint, per direction); joint types
    {type id: {core_group, column_above_group, joints, beams {axis}, through_bars {axis}}} from the joint-type records.
    """
    problems, joints, types = [], {}, {}
    capacity = capacity if isinstance(capacity, dict) else {}
    kinds = capacity.get("joint_types") if isinstance(capacity.get("joint_types"), dict) else {}
    for type_id, kind in sorted(kinds.items()):
        if not isinstance(kind, dict) or not kind.get("core_group"):
            problems.append(f"joint type {type_id} has no core group")
            continue
        anchorage = kind.get("anchorage") if isinstance(kind.get("anchorage"), dict) else {}
        types[type_id] = {"core_group": kind["core_group"], "column_above_group": kind.get("column_above_group"),
                          "joints": sorted(kind.get("joints") or []), "beams": {"x": [], "y": []},
                          "through_bars": {axis: (item.get("through_bars_present") if isinstance(item, dict) else None)
                                           for axis, item in anchorage.items()}}
    scwb = capacity.get("scwb") if isinstance(capacity.get("scwb"), dict) else {}
    for record in scwb.get("joints") or []:
        if not isinstance(record, dict) or not record.get("id"):
            problems.append("a strong-column joint record has no id")
            continue
        directions = record.get("directions") if isinstance(record.get("directions"), dict) else {}
        entry = {"floor": record.get("floor"), "type_id": record.get("type_id"), "is_roof": record.get("is_roof"),
                 "core_group": (types.get(record.get("type_id")) or {}).get("core_group"),
                 "columns": [], "beams": {}, "required": {}}
        for axis in ("x", "y"):
            direction = directions.get(axis) if isinstance(directions.get(axis), dict) else {}
            state = direction.get("positive") if isinstance(direction.get("positive"), dict) else {}
            entry["beams"][axis] = sorted({b.get("group_id") for b in state.get("beam_capacities") or []
                                           if isinstance(b, dict) and b.get("group_id")})
            entry["required"][axis] = direction.get("required") is True
            if axis == "x" or not entry["columns"]:
                entry["columns"] = sorted({c.get("group_id") for c in state.get("column_capacities") or []
                                           if isinstance(c, dict) and c.get("group_id")})
        joints[record["id"]] = entry
        kind = types.get(record.get("type_id"))
        if kind is not None:
            for axis in ("x", "y"):
                kind["beams"][axis] = sorted(set(kind["beams"][axis]) | set(entry["beams"][axis]))
    return joints, types, problems


def parent_context(space, state, *, basis_identity, realized_identity, capacity=None):
    """The routing context of one evaluated parent, bound to its realized identity and the run's basis."""
    joints, types, problems = joint_topology(capacity)
    groups = group_rows(state)
    for jid, entry in joints.items():
        for gid in entry["columns"] + entry["beams"]["x"] + entry["beams"]["y"] + [entry["core_group"]]:
            if gid is not None and gid not in groups:
                problems.append(f"joint {jid} names group {gid}, which the parent does not have")
    for type_id, kind in types.items():
        for gid in [kind["core_group"], kind["column_above_group"]] + kind["beams"]["x"] + kind["beams"]["y"]:
            if gid is not None and gid not in groups:
                problems.append(f"joint type {type_id} names group {gid}, which the parent does not have")
    topology = {"joints": joints, "joint_types": types}
    return {"version": VERSION, "parent_identity": realized_identity, "state_identity": state.identity(),
            "basis_identity": basis_identity, "genes": gene_rows(space), "genome": list(space.encode(state)),
            "groups": groups, "joints": joints, "joint_types": types, "num_floor": int(state.counts["num_floor"]),
            "topology_sha256": digest(topology), "topology_problems": sorted(set(problems))[:40]}


# ---- dependencies --------------------------------------------------------------------------------------------------
def _settled(constraints):
    """The reinforcement passes settled: the selection converged to the installed cages. A verified cycle resolution
    (Code_Check_Contract v2) is not settled here: its cages meet the selection's rules but are not the selection's
    choice, so the dependencies that read that choice (the cage and bar geometry evidence, the ratio-floor premise)
    stay unmet and such a parent's geometry failures are not routed (2026-11-01 closure follow-up)."""
    return (constraints.get("longitudinal_selection_settled") is True and constraints.get("hoops_settled") is True
            and constraints.get("reinforcement_settled") is True)


def _row_passes(rows, key):
    row = rows.get(key)
    return isinstance(row, dict) and contract.classify_row(row)[0] == "pass"


def _row_known(rows, key):
    row = rows.get(key)
    return isinstance(row, dict) and contract.classify_row(row)[0] in ("pass", "fail")


def _cage_area(design):
    return (2 * design.top_bars + 2 * design.side_bars) * sp.rebar_area(design.bar_size)


def cage_at_ratio_floor(state, gid, cfg=None):
    """True when the group's selected column cage is the lightest one its section admits (its steel ratio is set by
    the minimum, not by demand), False when a lighter admissible cage exists, None when it cannot be established
    (no column, no admissible domain, or a cage lighter than every admissible one: it did not come from the
    selection). A sufficient condition only: under the min_deviation objective, the strong-column links and the
    transitions the selection can install a heavier cage that demand does not set either; the flag is then False
    and no lever is withheld on an unproven premise."""
    from Design import Group_Selection as selection
    from Design.Config import DesignConfig
    design = state.designs.get(gid) if state is not None else None
    if design is None or not design.is_column:
        return None
    try:
        candidates = selection.column_cage_candidates(design, cfg if cfg is not None else DesignConfig())
        areas = [_cage_area(c) for c in candidates]
        selected = _cage_area(design)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if not areas or not finite(selected):
        return None
    lightest = min(areas)
    if selected < lightest * (1 - RELATIVE_TOLERANCE):
        return None
    return selected <= lightest * (1 + RELATIVE_TOLERANCE)


def _gravity_column_shear_problems(capacity, gid, row):
    """The gravity column capacity-shear row against its producer's per-direction record (ACI 318-19 22.5.1.1 with
    18.7.6.1.1: demand and strength in the same direction). The producer (Perimeter_Systems.gravity_column_capacity_shear)
    compares each direction's worst-story Ve (worst_by_direction) with that direction's hoop phi Vn (hoops.by_direction)
    and reports the direction with the larger ratio, recorded as the gravity group's strengths.capacity_shear_axis;
    that is not always the governing (largest Vs,req) direction, so the row is read on the recorded axis. With Vs
    uncapped, the hoop spacing gives phi Vn >= Ve in each direction, so an authenticated row cannot fail under the
    present producer: genuine section deficits route through column.capacity_shear_section. A row without a recorded
    axis, or whose numbers are not that direction's, is not authenticated."""
    data = (capacity.get("columns") or {}).get(gid) or {}
    worst = data.get("worst_by_direction") if isinstance(data.get("worst_by_direction"), dict) else {}
    hoops = data.get("hoops") if isinstance(data.get("hoops"), dict) else {}
    own_by_axis = hoops.get("by_direction") if isinstance(hoops.get("by_direction"), dict) else {}
    gravity = capacity.get("gravity_frame") if isinstance(capacity.get("gravity_frame"), dict) else {}
    group = (gravity.get("groups") or {}).get(gid) if isinstance(gravity.get("groups"), dict) else None
    strengths = group.get("strengths") if isinstance(group, dict) and isinstance(group.get("strengths"), dict) else {}
    pairs = {}
    for a, entry in worst.items():
        own = own_by_axis.get(a)
        if isinstance(entry, dict) and isinstance(own, dict):
            pairs[a] = (entry.get("ve_kip"), own.get("phi_vn_kip"))
    axis = strengths.get("capacity_shear_axis")
    if not (isinstance(axis, str) and axis in pairs and all(finite(v) for v in pairs[axis])):
        return ["no per-direction capacity-shear record (the compared axis, its Ve and its hoop phi Vn) to authenticate "
                "the gravity column row"]
    ve, own = pairs[axis]
    problems = []
    if not _close(row.get("demand"), ve):
        problems.append(f"the gravity column row's demand is not the {axis}-direction Ve")
    if not _close(row.get("capacity"), own):
        problems.append(f"the gravity column row compares the {axis}-direction Ve with another direction's hoop phi Vn: "
                        "not a per-direction deficit")
    ratios = {a: v / p for a, (v, p) in pairs.items() if finite(v) and finite(p) and p > 0}
    if axis in ratios and ratios[axis] < max(ratios.values()) * (1 - RELATIVE_TOLERANCE):
        problems.append(f"the gravity column row compares the {axis} direction, not the direction with the larger "
                        "Ve / phi Vn")
    return problems


def threading_constraint_evidence(constraints, state, gid, ceiling=1.0):
    """(dependency, problems, fields) of a beam group's threading-constrained deficit: None when the record does not
    list the group (constraints.threading_constrained_beam_groups). Listed, the dependency is True only when the
    selection's record (constraints.beam_threading_evidence) shows an authenticated deficit whose installed cage is the
    group's final cage and the strongest that threads, still above the ceiling, while a stronger admissible cage does
    not thread the cores it names. ``fields`` carry the blocking cores and, per core, the look-ahead size (the first
    larger square size whose core-cage lanes carry the stronger cage with the same column cage: geometry only, from a
    look-ahead entry of scope core_cage), the cores whose own cage blocks (None when the record does not separate them),
    the blocking transitions by both column groups and, per transition, the band whose first size opens that joint's own
    inventory (from an entry of scope transition_joint). An entry without a scope gives no target."""
    listed = constraints.get("threading_constrained_beam_groups")
    if not (isinstance(listed, list) and gid in listed):
        return None, [], {}
    record = (constraints.get("beam_threading_evidence") or {}).get(gid)
    if not isinstance(record, dict):
        return False, ["no threading evidence for this group"], {}
    installed, best_t, best_n = (record.get(k) if isinstance(record.get(k), dict) else {}
                                 for k in ("installed", "best_threading", "best_nonthreading"))

    def cage(entry):
        return entry.get("bar_size"), entry.get("top_bars"), entry.get("bot_bars")

    design = state.designs.get(gid) if state is not None else None
    problems = []
    if not all(record.get(k) is True for k in ("authenticated", "threading_constrained_deficit", "constrains_strength",
                                               "deficit_remains")):
        problems.append("the selection record does not show an authenticated threading-constrained deficit")
    if design is None or cage(installed) != (design.bar_size, design.top_bars, design.bot_bars):
        problems.append("the recorded installed cage is not the group's final cage")
    if cage(installed) != cage(best_t) or best_t.get("threads") is not True:
        problems.append("the installed cage is not the strongest threading cage")
    if not (finite(best_t.get("dcr")) and best_t["dcr"] > ceiling):
        problems.append("the strongest threading cage does not fall short")
    if not (finite(best_n.get("dcr")) and finite(best_t.get("dcr")) and best_n.get("threads") is False
            and best_n["dcr"] < best_t["dcr"] * (1 - RELATIVE_TOLERANCE)):
        problems.append("no stronger nonthreading cage is recorded")
    fields = _blocking_fields(record, problems)
    if not fields["blocking_cores"]:
        problems.append("the record names no core that blocks the stronger cage")
    fields.update(best_threading_dcr=best_t.get("dcr"), best_nonthreading_dcr=best_n.get("dcr"),
                  best_nonthreading_cage=list(cage(best_n)))
    return not problems, problems, fields


def _blocking_fields(record, problems):
    """What blocks a cage and where the look-ahead aims, from a threading record (the selection's threading evidence
    or the installed-cage lane evidence): the blocking cores, the cores whose own cage blocks (None when the record does
    not separate them), the blocking transitions by both column groups, a size per core from a look-ahead entry of
    scope core_cage, and per transition the band whose first size opens that joint's own inventory from an entry of
    scope transition_joint. An entry without a scope gives no target; malformed parts are added to ``problems``."""
    cores = record.get("blocking_cores")
    if not (isinstance(cores, list) and all(isinstance(c, str) for c in cores)):
        problems.append("the record's blocking cores are not group ids")
        cores = []
    cage_cores = record.get("blocking_core_cages")
    if cage_cores is not None and not (isinstance(cage_cores, list) and all(isinstance(c, str) for c in cage_cores)):
        problems.append("the record's blocking core cages are not group ids")
        cage_cores = []
    transitions = record.get("blocking_transitions") or []
    if not (isinstance(transitions, list) and all(isinstance(t, dict) and isinstance(t.get("core_group"), str)
                                                  and isinstance(t.get("above_group"), str) for t in transitions)):
        problems.append("the record's blocking transitions are not pairs of column groups")
        transitions = []
    pairs = {(t["core_group"], t["above_group"]) for t in transitions}
    lookahead, targets = {}, {}
    entries = record.get("lookahead") if isinstance(record.get("lookahead"), dict) else {}
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("scope") == "core_cage" and entry.get("core_group") in cores:
            size, current = entry.get("first_size_same_cage"), entry.get("current_size_in")
            if finite(size) and finite(current) and size > current:
                lookahead[entry["core_group"]] = max(lookahead.get(entry["core_group"], size), size)
        elif entry.get("scope") == "transition_joint" and (entry.get("core_group"), entry.get("above_group")) in pairs:
            for band, group, size, current in (
                    ("upper_band", entry["above_group"], entry.get("first_upper_band_size"), entry.get("current_above_size_in")),
                    ("lower_band", entry["core_group"], entry.get("first_lower_band_size"), entry.get("current_size_in"))):
                if finite(size) and finite(current) and size > current:
                    token = (group, band, entry["core_group"], entry["above_group"])
                    targets[token] = max(targets.get(token, size), size)
    return {"blocking_cores": sorted(cores), "lane_lookahead": {c: [s, s] for c, s in sorted(lookahead.items())},
            "blocking_core_cages": None if cage_cores is None else sorted(cage_cores),
            "blocking_transitions": [{"core_group": c, "above_group": a} for c, a in sorted(pairs)],
            "transition_lookahead": [{"group": g, "band": b, "core_group": c, "above_group": a, "target": [s, s]}
                                     for (g, b, c, a), s in sorted(targets.items())]}


def installed_lane_evidence(constraints, ctx, state, gid):
    """(dependency, problems, fields) of the installed cage's lanes of a beam group whose threading row failed:
    None when the record carries no installed-cage lane evidence (constraints.beam_lane_evidence, written by
    Group_Selection.installed_lane_evidence on the evaluated state). True only when that evidence is of this evaluated
    state (its identity), lists this group with the group's final cage and is well formed; ``fields`` (_blocking_fields)
    then say what blocks the installed cage and where the look-ahead aims. No flexural shortfall and no
    strongest-threading-cage record is required."""
    from Design.Group_Selection import INSTALLED_LANES_ID
    block = constraints.get("beam_lane_evidence")
    if block is None:
        return None, [], {}
    if not isinstance(block, dict) or block.get("basis") != INSTALLED_LANES_ID:
        return False, ["the installed-cage lane evidence is not of the declared basis"], {}
    problems = []
    if block.get("state_identity") != ctx.get("state_identity"):
        problems.append("the installed-cage lane evidence is of another state")
    entry = (block.get("groups") or {}).get(gid) if isinstance(block.get("groups"), dict) else None
    if not isinstance(entry, dict):
        return False, problems + ["the installed-cage lane evidence does not list this group"], {}
    design = state.designs.get(gid) if state is not None else None
    if design is None or (entry.get("bar_size"), entry.get("top_bars"), entry.get("bot_bars")) != (
            design.bar_size, design.top_bars, design.bot_bars):
        problems.append("the recorded installed cage is not the group's final cage")
    fields = _blocking_fields(entry, problems)
    return not problems, problems, fields


def threading_row_attribution(ctx, kind, details, constraints, state, gid):
    """(fields, problems) of a failed beam.bars_thread_column row: from the row's own lanes, whether the core cage
    blocks (core_cage_blocks: its own lanes do not carry the bars within the layer cap) and whether the transition to
    the column above does (transition_blocks: the joint's inventory gives fewer lanes than the core cage and does not
    carry them); None for both when the row does not separate the two (the earlier row form). The row's core and column
    above must be the joint type's (a mismatch is a problem). Where either blocks, the installed-cage lane evidence
    (installed_lane_evidence) adds the look-ahead targets of that core and of that transition only, with its status in
    ``lane_evidence`` (authenticated, absent or not_authenticated); a target is never taken from evidence that does not
    authenticate. The row itself stays the failure; nothing here passes it."""
    problems = []
    core, above = details.get("core_group"), details.get("above_group")
    if kind is not None and core is not None and core != kind["core_group"]:
        problems.append(f"the threading row names core {core}, not the joint's core {kind['core_group']}")
    if kind is not None and above is not None and above != kind.get("column_above_group"):
        problems.append(f"the threading row names column above {above}, not the joint's {kind.get('column_above_group')}")
    n, cap = details.get("bars_per_face"), details.get("max_layers")
    joint_fit, core_fit = details.get("bars_per_layer_that_fit"), details.get("bars_per_layer_core_cage")
    fields = {"core_group": core, "column_above_group": above, "core_cage_blocks": None, "transition_blocks": None}
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (n, cap, joint_fit, core_fit)):
        return fields, problems

    def admits(fit):
        return fit > 0 and math.ceil(n / fit) <= cap

    fields["core_cage_blocks"] = not admits(core_fit)
    fields["transition_blocks"] = above is not None and joint_fit < core_fit and not admits(joint_fit)
    if not (fields["core_cage_blocks"] or fields["transition_blocks"]):
        return fields, problems
    ok, evidence_problems, lane_fields = installed_lane_evidence(constraints, ctx, state, gid)
    fields["lane_evidence"] = "absent" if ok is None else "authenticated" if ok else "not_authenticated"
    if evidence_problems:
        fields["lane_evidence_problems"] = list(evidence_problems)
    if ok:
        fields["lane_lookahead"] = ({c: s for c, s in lane_fields["lane_lookahead"].items() if c == core}
                                    if fields["core_cage_blocks"] else {})
        fields["transition_lookahead"] = ([r for r in lane_fields["transition_lookahead"]
                                           if (r["core_group"], r["above_group"]) == (core, above)]
                                          if fields["transition_blocks"] else [])
    return fields, problems


def _scwb_dependencies(direction_state, scwb, row):
    """H1 dependencies from the joint's own sway-state record, and the row's numbers recomputed from it."""
    problems = []
    state = direction_state if isinstance(direction_state, dict) else {}
    columns = [c for c in state.get("column_capacities") or [] if isinstance(c, dict)]
    beams = [b for b in state.get("beam_capacities") or [] if isinstance(b, dict)]
    expected = scwb.get("expected_combination_ids") if isinstance(scwb.get("expected_combination_ids"), list) else []
    column_ok = (state.get("nominal_strengths") is True and bool(columns)
                 and all(c.get("axial_envelope_checked") is True and finite(c.get("mn_kip_in")) and c["mn_kip_in"] >= 0
                         for c in columns))
    coverage_ok = (scwb.get("action_inventory_complete") is True and bool(expected) and bool(columns)
                   and all(c.get("combination_count") == len(expected) for c in columns))
    beam_ok = (bool(beams) and all(b.get("slab_basis") in SLAB_BASES and finite(b.get("mn_kip_in"))
                                   and finite(b.get("slab_mn_kip_in")) for b in beams))
    if column_ok and beam_ok:
        column_sum = sum(c["mn_kip_in"] for c in columns)
        beam_sum = sum(b["mn_kip_in"] + b["slab_mn_kip_in"] for b in beams)
        if not (_close(row.get("capacity"), column_sum) and _close(row.get("demand"), 1.2 * beam_sum)):
            problems.append("the row's sums do not follow from the joint's capacity records")
    return {"nominal_column_strength": column_ok, "nominal_beam_slab_strength": beam_ok,
            "axial_case_coverage": coverage_ok}, problems


def _member_actions(evidence):
    actions, expected = evidence.get("actions"), evidence.get("expected_ids")
    if not isinstance(actions, list) or not isinstance(expected, list):
        return False
    _by_id, problems = action_inventory_problems(actions, expected)
    return not problems


# ---- the adapter -------------------------------------------------------------------------------------------------
def _base(ctx, family, check_id, location, reason, row, *, component=None, units=None):
    suffix = f"#{component}" if component else ""
    return {"id": f"{family}|{check_id}|{location}{suffix}", "reason": reason, "status": row.get("status"),
            "applicable": True, "evidence_valid": True, "evidence_problems": [],
            "parent_identity": ctx["parent_identity"], "basis_identity": ctx["basis_identity"], "dependencies": {},
            "demand": row.get("demand"), "capacity": row.get("capacity"), "comparison": row.get("comparison"),
            "units": units if units is not None else row.get("units", ""), "synthetic": False,
            "source": {"family": family, "check_id": check_id, "location": location, "clause": row.get("clause"),
                       "component": component}}


def _strength_diagnostics(ctx, row, evidence, constraints, state=None):
    """member.strength split into its components (the compact row keeps only the governing DCR)."""
    gid = row.get("location", "")
    groups = (evidence.get("strengths") or {}).get("groups") if isinstance(evidence.get("strengths"), dict) else None
    result = (groups or {}).get(gid) if isinstance(groups, dict) else None
    group = ctx["groups"].get(gid)
    if not isinstance(result, dict) or group is None:
        item = _base(ctx, *STRENGTH_SOURCE, gid, None, row)
        item.update(evidence_valid=False, evidence_problems=["no strength evidence for the group of this row"])
        return [item]
    actions_ok = _member_actions(evidence)
    expected = evidence.get("expected_ids") if isinstance(evidence.get("expected_ids"), list) else []
    observations_ok = result.get("observations") == len(group["member_tags"]) * len(expected) and bool(expected)
    lanes = constraints.get("lane_limited_beam_groups")
    lane_limited = isinstance(lanes, list) and gid in lanes
    threading, threading_problems, threading_fields = threading_constraint_evidence(constraints, state, gid)
    if result.get("member_type") == "column":
        components = (("pm", "column_pm"), ("shear", "column_analysis_shear"))
    else:
        # A lane-limited group's installed cage is the strongest that threads: its shortfall is bar room, not strength.
        # A threading-constrained one is short of strength AND kept from a stronger cage by threading (the mixed deficit).
        flexure = ("beam_lane_limited_strength" if lane_limited else
                   "beam_threading_constrained_strength" if threading is not None else "beam_flexure")
        components = (("flexure_positive", flexure), ("flexure_negative", flexure), ("shear", "beam_analysis_shear"))
    dcrs = [result[name].get("dcr") for name, _ in components if isinstance(result.get(name), dict)]
    out = []
    for name, reason in components:
        part = result.get(name)
        if not isinstance(part, dict) or not finite(part.get("dcr")) or part["dcr"] <= 1.0:
            continue
        item = _base(ctx, *STRENGTH_SOURCE, gid, reason if reason in RULES else None, row, component=name)
        item["group_id"] = gid
        item["source"].update(member_tag=part.get("member_tag"), combination=part.get("combination"))
        if reason in ("beam_flexure", "beam_lane_limited_strength", "beam_threading_constrained_strength"):
            item.update(demand=part.get("mu_kip_in"), capacity=part.get("phi_mn_kip_in"), comparison="<=",
                        units="kip-in")
            section = finite(part.get("phi_mn_kip_in")) and part["phi_mn_kip_in"] > 0
            if part.get("tension_controlled") is False:
                item["evidence_problems"].append("the section is not tension-controlled: the DCR is a domain-failure "
                                                 "sentinel, not a moment ratio")
        elif reason == "column_pm":
            item.update(demand=part.get("dcr"), capacity=1.0, comparison="<=", units="DCR")
            section = part.get("basis") in ("biaxial_load_contour", "axial", "axial_compression_above_cap",
                                            "axial_tension_outside_surface")
            if part.get("domain_failure") == "moment_at_axial_end":
                item["evidence_problems"].append("moment at an axial load without flexural strength: a "
                                                 "domain-failure sentinel, not an interaction ratio")
        else:
            item.update(demand=part.get("vu_kip"), capacity=part.get("phi_vn_kip"), comparison="<=", units="kip")
            section = False
            item["evidence_problems"].append("analysis shear (phi Vn < Vu) is not a reason of the operator contract")
        item["dependencies"] = {"member_actions": bool(actions_ok and observations_ok), "section_strength": bool(section)}
        if reason == "beam_lane_limited_strength":
            item["dependencies"]["lane_limited_selection"] = lane_limited
            # The beam's own depth and width need no topology; the column levers (the cores it threads) do.
            item["joint_topology_consistent"] = not ctx["topology_problems"]
            # What blocks the cage that reaches the demand, where the selection's threading evidence authenticates:
            # the cores whose own cage blocks and the transitions by both groups, with the look-ahead targets.
            item["threading_evidence"] = ("absent" if threading is None else "authenticated" if threading
                                          else "not_authenticated")
            if threading is True:
                item.update(threading_fields)
        elif reason == "beam_threading_constrained_strength":
            item["dependencies"]["threading_constrained_selection"] = threading is True
            item["joint_topology_consistent"] = not ctx["topology_problems"]
            item.update(threading_fields)
            if threading_problems:
                item["dependency_problems"] = list(threading_problems)
        out.append(item)
    finite_dcrs = [d for d in dcrs if finite(d)]
    if not out or not (finite(row.get("demand")) and finite_dcrs and _close(row["demand"], max(finite_dcrs))):
        for item in out:
            item["evidence_problems"].append("the row's DCR does not follow from its components")
        item = _base(ctx, *STRENGTH_SOURCE, gid, None, row)
        item["evidence_problems"].append("the row's DCR does not follow from its components")
        out.append(item)
    for item in out:
        item["evidence_valid"] = not item["evidence_problems"]
    return out


def _source_diagnostic(ctx, row, rows, evidence, constraints, state=None, cfg=None):
    family, check_id, location = row.get("family"), row.get("id"), row.get("location", "")
    reason, comparison, raw_units, units = SOURCES[(family, check_id)]
    item = _base(ctx, family, check_id, location, reason, row, units=units)
    problems = item["evidence_problems"]
    if row.get("comparison") != comparison or (row.get("units") or "") != raw_units:
        problems.append(f"comparison/units {row.get('comparison')!r}/{row.get('units')!r} are not the declared "
                        f"{comparison!r}/{raw_units!r} of {check_id}")
    capacity = evidence.get("capacity") if isinstance(evidence.get("capacity"), dict) else {}
    settled = _settled(constraints)
    deps = {}
    if reason == "scwb":
        match = _SCWB.match(location)
        if not match:
            problems.append("unparsed strong-column location")
        else:
            item.update(joint_id=match["joint"], axis=match["axis"], sign=match["sign"])
            record = next((j for j in (capacity.get("scwb") or {}).get("joints") or []
                           if isinstance(j, dict) and j.get("id") == match["joint"]), None)
            direction = (((record or {}).get("directions") or {}).get(match["axis"]) or {}).get(match["sign"])
            deps, extra = _scwb_dependencies(direction, capacity.get("scwb") or {}, row)
            problems.extend(extra)
            if record is None:
                problems.append("no strong-column record for this joint")
    elif reason in ("beam_capacity_shear", "column_capacity_shear"):
        gid = location.split("/member_")[0] if check_id.startswith("beam_capacity_shear_") else location
        item["group_id"] = gid
        kind = "column" if reason == "column_capacity_shear" else "beam"
        # The inner hoop selection has finished with a recorded outcome: hoops chosen, or the ladder exhausted by the
        # same over-demand (the section levers are then exactly what remains).
        hoops = _row_known(rows, ("capacity", f"{kind}.hoops_selected", gid))
        if kind == "column":
            governing = ((capacity.get("columns") or {}).get(gid) or {}).get("governing")
            equilibrium = isinstance(governing, dict) and finite(governing.get("vs_required_kip"))
            # With the steel ratio at its minimum and Ve set by the probable-moment mechanism, a larger section grows
            # Ve faster than its shear limit. Read on a settled longitudinal selection under the column-own method
            # only; otherwise unknown (no lever is withheld on it).
            mechanism = (isinstance(governing, dict) and governing.get("method") == COLUMN_SHEAR_METHOD_COLUMN_OWN
                         and finite(governing.get("ve_own_envelope_kip")) and finite(governing.get("ve_kip"))
                         and governing["ve_own_envelope_kip"] >= governing["ve_kip"] * (1 - RELATIVE_TOLERANCE))
            item["cage_at_ratio_floor"] = (cage_at_ratio_floor(state, gid, cfg) if mechanism
                                           and constraints.get("longitudinal_selection_settled") is True else None)
            if check_id == "gravity.column_capacity_shear":
                problems.extend(_gravity_column_shear_problems(capacity, gid, row))
        elif check_id == "beam.capacity_shear_section":
            data = (capacity.get("beams") or {}).get(gid) or {}
            members = data.get("governing_members") or []
            equilibrium = (finite(data.get("vs_required_kip")) and bool(members)
                           and all(finite((m or {}).get("governing_ve_kip")) for m in members))
        else:
            equilibrium = finite(row.get("demand")) and row["demand"] > 0
        deps = {"probable_equilibrium": bool(equilibrium),
                "transverse_selection": bool(hoops and constraints.get("hoops_settled") is True)}
    elif reason == "beam_width_room":
        gid = location.split("/")[0]
        item["group_id"] = gid
        deps = {"cage_geometry": settled and gid in ctx["groups"]}
    elif reason in ("beam_threading", "through_bar_depth", "hook_embedment", "slab_mat_overlap", "joint_shear"):
        if reason == "slab_mat_overlap":
            type_id, axis = location, None
        else:
            match = (_JOINT_SHEAR if reason == "joint_shear" else _JOINT_TYPE_AXIS).match(location)
            type_id, axis = (match["type"], match["axis"]) if match else (None, None)
            if not match:
                problems.append("unparsed joint-type location")
        kind = ctx["joint_types"].get(type_id)
        record = ((capacity.get("joint_types") or {}).get(type_id) or {}) if type_id else {}
        item.update(joint_id=type_id, joint_type_id=type_id, axis=axis)
        incident = bool(kind and kind["core_group"] in ctx["groups"] and (axis is None or kind["beams"].get(axis)))
        if reason == "joint_shear":
            shear = (record.get("shear") or {}).get(axis) or {}
            deps = {"joint_classification": _row_passes(rows, ("capacity", "joint.classification_evidence",
                                                               f"joint_shear/{type_id}/{axis}")),
                    "joint_actions": finite(shear.get("vj_kip")),
                    "joint_capacity": finite(shear.get("phi_vn_kip"))
                    and shear.get("capacity_topology_and_confinement_checked") is True}
        elif reason in ("through_bar_depth", "hook_embedment"):
            through = (kind or {}).get("through_bars", {}).get(axis)
            deps = {"bar_geometry": settled,
                    "joint_incidence": incident and through is (reason == "through_bar_depth")}
        elif reason == "beam_threading":
            # The beam the threading record names, and only if it frames into this joint direction (no substitute).
            beam = ((record.get("threading") or {}).get(axis) or {}).get("beam_group")
            beams = (kind or {}).get("beams", {}).get(axis) or []
            item["group_id"] = beam if beam in beams else None
            if beam is not None and beam not in beams:
                problems.append(f"the threading record names beam group {beam}, which does not frame into this joint "
                                "direction")
            deps = {"cage_geometry": settled, "joint_incidence": incident and item["group_id"] is not None}
            # Whether the core cage or the transition to the column above blocks, from the row's own lanes, with the
            # installed cage's look-ahead targets where its lane evidence authenticates.
            fields, extra = threading_row_attribution(ctx, kind, (record.get("threading") or {}).get(axis) or {},
                                                      constraints, state, item["group_id"])
            item.update(fields)
            problems.extend(extra)
        else:
            clashes = (record.get("slab_mats") or {}).get("clashes") or []
            named = sorted({m.group(1) for c in clashes for label in (c.get("between") or [])[1:2]
                            for m in [_CLASH_GROUP.search(str(label))] if m})
            # A diagnosis gap, not invalid evidence: the router abstains on the missing dependency.
            item.update(group_id=named[0] if named else None, blocked_dimension=None, clashing_groups=named,
                        diagnosis_gap="the clash record names the crossing bar layers, not which section dimension "
                                      "is blocked")
            deps = {"cage_geometry": settled,
                    "slab_layout": (constraints.get("slab") or {}).get("layout_selected") is True,
                    "diagnosed_dimension": False}
    elif reason in ("story_drift", "story_stability"):
        match = _STORY.match(location)
        if not match or match["load"] != match["axis"]:
            problems.append("unparsed story location")
        else:
            story, axis = int(match["story"]), match["axis"]
            item.update(story=story, axis=axis)
            demand_basis = evidence.get("demand_basis") if isinstance(evidence.get("demand_basis"), dict) else {}
            coverage = _row_passes(rows, ("drift", "demands.node_coverage", f"story:{story}/Q{axis}"))
            deps = {("drift_analysis" if reason == "story_drift" else "stability_analysis"): coverage,
                    "story_location": 1 <= story <= ctx["num_floor"],
                    "limit_basis": isinstance(demand_basis.get("drift"), dict) and bool(demand_basis["drift"])
                    and finite(row.get("capacity")) and row["capacity"] > 0}
    if ctx["topology_problems"] and reason in ("scwb", "beam_threading", "slab_mat_overlap", "joint_shear",
                                               "through_bar_depth", "hook_embedment"):
        problems.append("the parent's joint topology is inconsistent with its groups")
    item["dependencies"] = {key: bool(value) for key, value in deps.items()}
    item["evidence_valid"] = not problems
    return item


def _record_reason(key, row, evidence):
    """The reason an unevaluated row that names none was not evaluated, when the producer's own record shows it
    (never assumed): a gravity column capacity-shear row of a column without selected hoops whose governing Ve is
    known (the missing hoops alone explain it)."""
    if key == ("capacity", "gravity.column_capacity_shear"):
        capacity = evidence.get("capacity") if isinstance(evidence.get("capacity"), dict) else {}
        data = (capacity.get("columns") or {}).get(row.get("location", ""))
        governing = data.get("governing") if isinstance(data, dict) else None
        if (isinstance(data, dict) and not data.get("hoops") and isinstance(governing, dict)
                and finite(governing.get("ve_kip"))):
            return ["hoops_not_selected"]
    return []


def diagnose(evaluation, assessed, ctx, cfg=None):
    """Normalized diagnostics of one evaluated record against its parent ``ctx`` (parent_context); ``cfg`` is the
    design configuration the record was evaluated under (the column cage domain).

    Returns {"version", "status", "parent_identity", "basis_identity", "diagnostics", "unknown", "unsupported",
    "counts"}. ``status`` is "invalid_record" (the contract assessment did not validate the record: no diagnostics),
    "stopped" (no code check was reached), "unbound_context" (``ctx`` is not this record's realized design and final
    state: no diagnostics) or "diagnosed". Rows the inventory declares not applicable are neither diagnostics nor
    evidence to obtain.
    """
    out = {"version": VERSION, "status": "diagnosed", "parent_identity": ctx.get("parent_identity"),
           "basis_identity": ctx.get("basis_identity"), "diagnostics": [], "unknown": [], "unsupported": {},
           "counts": {}}
    if not assessed.get("valid"):
        out["status"] = "invalid_record"
        return out
    if not assessed.get("code_check") or evaluation.get("status") != "evaluated":
        out["status"] = "stopped"
        return out
    # The context must be this record's own parent: its realized identity and final state.
    final = evaluation.get("state")
    if (ctx.get("parent_identity") != (evaluation.get("code_checks") or {}).get("realized_identity")
            or final is None or ctx.get("state_identity") != final.identity()):
        out["status"] = "unbound_context"
        return out
    rows_list = (evaluation.get("code_checks") or {}).get("rows") or []
    rows = {(r.get("family"), r.get("id"), r.get("location", "")): r for r in rows_list if isinstance(r, dict)}
    evidence = evaluation.get("evidence") if isinstance(evaluation.get("evidence"), dict) else {}
    constraints = evaluation.get("constraints") if isinstance(evaluation.get("constraints"), dict) else {}
    unknown = {}
    declared = {(n.get("family"), n.get("id"), n.get("location", ""))
                for n in (assessed["code_check"].get("not_applicable") or [])
                + (assessed["code_check"].get("excluded") or [])}
    for row in rows_list:
        state, _problems = contract.classify_row(row)
        key = (row.get("family"), row.get("id"))
        if state == "unknown" and (key[0], key[1], row.get("location", "")) in declared:
            continue                                  # declared not applicable, or outside the scope: nothing to obtain
        if state == "unknown":
            details = row.get("details") or {}
            codes = sorted(set(([details["reason_code"]] if details.get("reason_code") else [])
                               + list(details.get("reason_codes") or []))) or _record_reason(key, row, evidence) \
                or ["unclassified"]
            bucket = unknown.setdefault((key, tuple(codes)), {"family": key[0], "check_id": key[1], "reason_codes": codes,
                                                               "reason": SOURCES.get(key, (None,))[0], "count": 0,
                                                               "locations": []})
            bucket["count"] += 1
            if len(bucket["locations"]) < 6:
                bucket["locations"].append(row.get("location", ""))
            continue
        if state != "fail":
            continue
        if key == STRENGTH_SOURCE:
            out["diagnostics"].extend(_strength_diagnostics(ctx, row, evidence, constraints, final))
        elif key in SOURCES:
            out["diagnostics"].append(_source_diagnostic(ctx, row, rows, evidence, constraints, final, cfg))
        else:
            item = _base(ctx, key[0], key[1], row.get("location", ""), None, row)
            item["evidence_problems"].append("no mutation reason is declared for this check")
            item["evidence_valid"] = False
            out["diagnostics"].append(item)
            out["unsupported"][f"{key[0]}|{key[1]}"] = out["unsupported"].get(f"{key[0]}|{key[1]}", 0) + 1
    out["unknown"] = sorted(unknown.values(), key=lambda u: (u["family"], u["check_id"], u["reason_codes"]))
    reasons = {}
    for item in out["diagnostics"]:
        reasons[item["reason"] or "unsupported"] = reasons.get(item["reason"] or "unsupported", 0) + 1
    out["counts"] = {"diagnostics": len(out["diagnostics"]), "by_reason": reasons,
                     "unknown_rows": sum(u["count"] for u in out["unknown"])}
    return out
