"""Grouped design search: a feasibility phase and a bounded material-reduction phase (2026-10-02).

A candidate is a grouped design (Model.Member_Groups.GroupedDesign); Design.Grouped_Design.evaluate_candidate
evaluates one completely. This module proposes candidates, decides what happens to each, and keeps the
record of every one of them.

  seed          the uniform design expanded into its groups (the reproducibility reference), or a seed
                the caller supplies. It is where the search starts, not a lower bound on any section.
  feasibility   while the candidate is not feasible, the groups named by its failed constraints grow one
                ladder step (the column of a joint that fails joint shear, the beams of a story that fails
                drift, ...). Bounded by ``max_feasibility_trials``.
  reduction     from a feasible design, one group at a time is offered the next smaller ladder section.
                The whole frame is re-evaluated (slab, floor transfers of the floors that changed, every
                combination, every group's reinforcement re-selected at the new size, capacity design,
                joints, drift). The move is kept only under the declared policy; otherwise the previous
                design is restored exactly (designs are immutable: nothing has to be undone).
                Bounded by ``max_reduction_trials`` full evaluations.

Improvement policy ``dominance_concrete_and_total_reinforcement_v1`` (a bounded dominance rule, not a claim
of an optimum, and no cost data is implied): a feasible move is accepted when modeled concrete and total
reinforcement do not increase and at least one decreases. A feasible move that lowers one and raises the
other is kept in ``tradeoffs`` for review and not adopted. With both quantities tied, fewer distinct
form sizes wins. When a quantity is not available (no established slab layout) the comparison is
``unresolved`` and the move is not adopted: nothing is replaced by zero.

Column sizes along one column line may not grow upward and may not step in by 3 in or more per face
between adjacent bands (Design.SMRF_Transitions); moves that would break that are not evaluated and are
logged as skipped with the reason. Concrete grades stay the seed's one column grade and one beam grade.

Every candidate is logged with why it was proposed, what it changed, its constraints and quantities and
what was decided, and is appended to ``candidates.jsonl`` as soon as it is known when a log path is given,
so a candidate that fails (or an unexpected exception after it) leaves its evidence behind.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import Structure_Parameters as sp
from Design import Grouped_Design as gd
from Design import Group_Selection as selection
from Design.Section_Design import (BEAM_CONCRETE_STRENGTHS_KSI, COLUMN_CONCRETE_STRENGTHS_KSI, COLUMN_SIZES_IN,
                                   beam_ladder)
from Model import Member_Groups as mg
from Model import Member_Properties as mp

SEARCH_VERSION = "grouped_search_v2"
STEEL_GRADE_RETRY_RULE = "grade60_limits_then_grade80_full_search_v1"
SFRS_LAYOUT_LADDER_RULE = "perimeter_then_core_ring_then_core_lines_v1"
IMPROVEMENT_POLICY_ID = "dominance_concrete_and_total_reinforcement_v1"
# The lever order of the iterative design (user decision 2026-10-16): concrete strength first, then the reinforcement
# (which every evaluation exhausts through the cage selection), and the section size last. A growth move therefore
# steps the common column or beam grade one rung before any size; only a failure that is room for bars (threading,
# slab mats, bar rows, a cage layout) goes to the size directly, since a grade makes no room.
LEVER_ORDER_ID = "strength_then_reinforcement_then_size_v1"
GRADE_LADDERS = {"column": tuple(COLUMN_CONCRETE_STRENGTHS_KSI), "beam": tuple(BEAM_CONCRETE_STRENGTHS_KSI)}
GRADE_KEYS = {"column": "column_grade", "beam": "beam_grade"}
ROOM_FAILURES = ("beam.bars_thread_column", "beam.bar_stacking_clear_of_slab_mats", "beam.cage_layout", "column.cage_layout",
                 "joint.through_bar_depth")
MAX_COLUMN_SIZE_STEP_IN = 4.0          # adjacent bands: face step < 3 in on each face (ACI 318-19 10.7.4.2)
MAX_BEAM_REDUCTION_OPTIONS = 3         # lighter beam rungs looked at per group and sweep (the first that fits is tried)


@dataclass(frozen=True)
class SearchPolicy:
    """The bounds and the decision rule of one grouped search (part of the design identity)."""
    max_feasibility_trials: int = 10
    max_reduction_trials: int = 30
    improvement_policy: str = IMPROVEMENT_POLICY_ID
    quantity_tolerance: float = 1e-9
    reduction_order: str = "top_band_first__columns_then_beams__one_step_per_group_per_sweep"
    lever_order: str = LEVER_ORDER_ID
    grade80_retry_on_limits: bool = True
    # More frames around the core when a system-policy search finds no design (user direction 2026-10-18): the same
    # seed is searched again under the next layout of Model.Member_Groups.SFRS_LAYOUT_LADDER, each with its own budget.
    sfrs_layout_ladder_on_no_design: bool = True

    def problems(self):
        issues = []
        for name in ("max_feasibility_trials", "max_reduction_trials"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                issues.append(f"{name} must be a nonnegative integer")
        if self.improvement_policy != IMPROVEMENT_POLICY_ID:
            issues.append(f"unknown improvement policy {self.improvement_policy!r}")
        if self.lever_order != LEVER_ORDER_ID:
            issues.append(f"unknown lever order {self.lever_order!r}")
        if type(self.grade80_retry_on_limits) is not bool:
            issues.append("grade80_retry_on_limits must be a boolean")
        if type(self.sfrs_layout_ladder_on_no_design) is not bool:
            issues.append("sfrs_layout_ladder_on_no_design must be a boolean")
        return issues


def grade80_retry_reasons(evaluation):
    """Explicit strength/congestion evidence at the end of a failed Grade 60 search.

    Drift, solver errors, missing evidence, slab thickness and unsupported
    column transitions alone are not reasons to increase longitudinal fy.
    The resulting Grade 80 search must still satisfy every normal constraint.
    """
    if evaluation.get("feasible") or evaluation.get("status") not in ("evaluated", "infeasible"):
        return []
    reasons = []
    evidence = evaluation.get("failure_evidence") or {}
    if evaluation.get("status") == "infeasible":
        if evaluation.get("failed_stage") == "bar_rows" and evidence.get("group_ids"):
            reasons.append({"id": "beam.bar_rows", "groups": evidence["group_ids"]})
        if (evaluation.get("failed_stage") == "capacity_design"
                and evidence.get("exception") == "SectionAxialDomainError" and evidence.get("group_id")):
            reasons.append({"id": "column.axial_strength_domain", "groups": [evidence["group_id"]]})
        return reasons
    constraints = evaluation.get("constraints") or {}
    if constraints.get("strength_within_ceiling") is False:
        reasons.append({"id": "member.strength_ceiling", "dcr": evaluation.get("dcr")})
    if constraints.get("lane_limited_beam_groups"):
        reasons.append({"id": "beam.lane_congestion", "groups": constraints["lane_limited_beam_groups"]})
    # An exhausted beam domain means no admissible cage reaches its demand.
    beams = [gid for gid in constraints.get("exhausted_groups", []) if "__beam_" in gid]
    if beams:
        reasons.append({"id": "beam.reinforcement_domain_exhausted", "groups": beams})
    for line, item in (constraints.get("exhausted_column_lines") or {}).items():
        if str(item.get("blocked_by", "")).startswith("no admissible cage of this band meets its own strength"):
            reasons.append({"id": "column.reinforcement_domain_exhausted", "line": line, **item})
    eligible = {"beam.bars_thread_column", "beam.bar_stacking_clear_of_slab_mats",
                "beam.cage_layout", "column.cage_layout"}
    for item in constraints.get("capacity_design_failed_checks", []):
        if item.get("status") == "fail" and item.get("id") in eligible:
            reasons.append({"id": item["id"], "location": item.get("location", "")})
    return reasons


# ---- ladders ------------------------------------------------------------------------------------------------
def beam_rungs(member_type):
    """[(b, h)] offered to the beams of one direction, lightest first (flexural proxy b h^2)."""
    span = sp.BAY_X if member_type == "beam_x" else sp.BAY_Y
    rungs = {(b, h) for b, h, _fc in beam_ladder(span_in=span, story_height_in=sp.STORY_H)}
    return sorted(rungs, key=lambda r: (r[0] * r[1] ** 2, r[1], r[0]))


def _column_sizes(state=None):
    """The state's layout-specific column ladder, currently capped at 36 in for every layout."""
    from Design.Section_Design import column_sizes_in
    return list(column_sizes_in(None if state is None else mg.layout_of(state.policy_id)))


def _column_step(design, direction, state=None):
    sizes = _column_sizes(state)
    if design.b_in != design.h_in or design.b_in not in sizes:
        return None
    index = sizes.index(design.b_in) + direction
    return (sizes[index], sizes[index]) if 0 <= index < len(sizes) else None


def _beam_step(design, move):
    rungs = beam_rungs(design.member_type)
    current = (design.b_in, design.h_in)
    proxy = current[0] * current[1] ** 2
    if move == "smaller":
        # Less concrete, the least loss of flexural proxy: the candidates with a smaller area, strongest first.
        lighter = [r for r in rungs if r[0] * r[1] < current[0] * current[1] - 1e-9]
        return sorted(lighter, key=lambda r: (-(r[0] * r[1] ** 2), -r[1], r[0]))
    above = [r for r in rungs if (r[0] * r[1] ** 2, r[1], r[0]) > (proxy, current[1], current[0])]
    if move == "deeper":
        return next((r for r in above if r[1] > current[1]), None)
    if move == "wider":
        return next((r for r in above if r[1] == current[1] and r[0] > current[0]), None)
    if move == "larger":
        return above[0] if above else None
    raise ValueError(f"Unknown beam move {move!r}")


def resized(state, gid, size):
    """``gid``'s group at ``size``; under the system policy the whole design tie takes it."""
    design = replace(state.designs[gid], b_in=float(size[0]), h_in=float(size[1]))
    tie = state.tie_of(gid)
    return state.with_tied_designs({tie: design}) if tie else state.with_designs({gid: design})


def _tied_groups(state, gid):
    tie = state.tie_of(gid)
    return tuple(state.ties()[tie]) if tie else (gid,)


def _chains(state):
    """{unit label: [band tuples of group ids]} of the column lines selected together (Group_Selection._column_units)."""
    return {label: selection._column_chain(state, locations) for label, locations in selection._column_units(state)}


# ---- concrete grades (strength first) ---------------------------------------------------------------------------
def common_grade(state, kind):
    """The one column or beam grade of a grouped design (Grouped_Design._require_common_grades)."""
    grades = {d.fc_ksi for d in state.designs.values() if d.is_column == (kind == "column")}
    if len(grades) != 1:
        raise mg.GroupedStateError(f"{kind} groups use different concrete grades {sorted(grades)}.")
    return grades.pop()


def next_grade(state, kind):
    """The next rung of the common grade ladder above the current grade, or None at the top (a grade below the
    ladder, a legacy seed, steps to its first rung)."""
    current = common_grade(state, kind)
    return next((fc for fc in GRADE_LADDERS[kind] if fc > current + 1e-9), None)


def regraded(state, kind, fc):
    """Every column (or beam) group at grade ``fc``: the one grade per member type of the engine."""
    return state.with_designs({gid: replace(d, fc_ksi=float(fc)) for gid, d in state.designs.items()
                               if d.is_column == (kind == "column")})


def apply_moves(state, moves):
    """Apply growth moves {target: (to, reason)}: a grade key to every group of that kind, a design tie to its groups,
    a group id to that group (its tie under the system policy)."""
    for key, (to, _why) in moves.items():
        if key in GRADE_KEYS.values():
            state = regraded(state, key.split("_")[0], to)
        elif key in state.ties():
            state = state.with_tied_designs({key: replace(state.tie_designs()[key], b_in=float(to[0]), h_in=float(to[1]))})
        else:
            state = resized(state, key, to)
    return state


def _move_rows(moves):
    return [{"target": key, "kind": "grade" if key in GRADE_KEYS.values() else "size",
             "to": (float(to) if key in GRADE_KEYS.values() else list(to)), "reason": why}
            for key, (to, why) in sorted(moves.items())]


def _moves_summary(moves):
    grades = [f"{key.split('_')[0]} grade -> {to:g} ksi" for key, (to, _why) in sorted(moves.items()) if key in GRADE_KEYS.values()]
    sizes = sum(1 for key in moves if key not in GRADE_KEYS.values())
    parts = grades + ([f"grow {sizes} group(s)/tie(s)"] if sizes else [])
    return "; ".join(parts) if parts else "no move"


# ---- the two systems (decision D6: the gravity sections never exceed the perimeter-frame sections) ----------------
def _dominating_rung(member_type, b, h):
    """The lightest beam rung at least ``b`` wide and ``h`` deep ((b, h) itself when it is a rung), or None."""
    rungs = beam_rungs(member_type)
    if (b, h) in rungs:
        return (b, h)
    fits = [r for r in rungs if r[0] >= b - 1e-9 and r[1] >= h - 1e-9]
    return fits[0] if fits else None


def system_coupling_problem(state, gid, size):
    """Why an SFRS tie may not take ``size`` beside the gravity tie, or None; nothing under the band policy."""
    tie = state.tie_of(gid)
    if tie is None or mg.tie_kind(tie) not in ("sfrs_column", "sfrs_beam"):
        return None
    other = state.tie_designs().get(state.related_tie(tie, "gravity_column" if mg.tie_kind(tie) == "sfrs_column" else "gravity_beam"))
    if other is None:
        return None
    if size[0] < other.b_in - 1e-9 or size[1] < other.h_in - 1e-9:
        return (f"would be smaller ({size[0]:g} x {size[1]:g} in) than the gravity system's {other.b_in:g} x {other.h_in:g} in: "
                "the gravity sections never exceed the perimeter-frame sections (decision D6, 2026-10-16)")
    return None


def enforce_system_coupling(state):
    """Grow an SFRS tie up to the gravity tie where a gravity move overtook it. Returns (state, [(tie, from, to, reason)])."""
    changes = []
    designs = state.tie_designs()
    if not designs:
        return state, changes
    for tie in sorted(designs):
        family = mg.tie_kind(tie)
        if family not in ("sfrs_column", "sfrs_beam"):
            continue
        sfrs = state.tie_designs()[tie]
        gravity = state.tie_designs().get(state.related_tie(tie, family.replace('sfrs_', 'gravity_')))
        if gravity is None or (gravity.b_in <= sfrs.b_in + 1e-9 and gravity.h_in <= sfrs.h_in + 1e-9):
            continue
        b, h = max(sfrs.b_in, gravity.b_in), max(sfrs.h_in, gravity.h_in)
        target = (b,h) if sfrs.is_column else _dominating_rung(sfrs.member_type,b,h)
        if target is None:
            raise mg.GroupedStateError(f"No beam rung can enforce gravity/perimeter coupling for {tie}.")
        changes.append((tie,[sfrs.b_in,sfrs.h_in],list(target),"the perimeter sections are never smaller than the gravity sections in this band (decision D6)"))
        state = state.with_tied_designs({tie: replace(sfrs,b_in=target[0],h_in=target[1])})
    return state, changes


def column_taper_problem(state, gid, size):
    """Why a column group may not take ``size`` beside the bands below and above it, or None."""
    if mg.is_banded_policy(state.policy_id):
        problems = mg.section_order_problems(resized(state,gid,size))
        if problems:
            return problems[0]
    for chain in _chains(state).values():
        index = next((k for k, band in enumerate(chain) if gid in band), None)
        if index is None:
            continue
        if index > 0:
            lower = state.designs[chain[index - 1][0]].b_in
            if size[0] > lower:
                return f"would be larger ({size[0]:g} in) than the column below it ({lower:g} in)"
            if lower - size[0] > MAX_COLUMN_SIZE_STEP_IN:
                return (f"would step in {0.5 * (lower - size[0]):g} in per face from the column below ({lower:g} in): "
                        "3 in or more needs dowels (ACI 318-19 10.7.4.2), not a declared transition")
        if index < len(chain) - 1:
            upper = state.designs[chain[index + 1][0]].b_in
            if size[0] < upper:
                return f"would be smaller ({size[0]:g} in) than the column above it ({upper:g} in)"
            if size[0] - upper > MAX_COLUMN_SIZE_STEP_IN:
                return (f"would leave the column above ({upper:g} in) stepping in {0.5 * (size[0] - upper):g} in per face: "
                        "3 in or more needs dowels (ACI 318-19 10.7.4.2), not a declared transition")
    return None


def enforce_column_taper(state):
    """Grow columns until no line grows upward or steps in too far. Returns (state, [(gid, from, to, reason)])."""
    if mg.is_banded_policy(state.policy_id):
        return enforce_banded_section_order(state)
    changes = []
    for chain in _chains(state).values():
        for _ in range(4 * len(chain)):
            moved = False
            for index in range(len(chain) - 1):
                lower, upper = state.designs[chain[index][0]], state.designs[chain[index + 1][0]]
                if upper.b_in > lower.b_in:
                    target, gid, reason = (upper.b_in, upper.h_in), chain[index][0], "a column may not be smaller than the one above it"
                elif lower.b_in - upper.b_in > MAX_COLUMN_SIZE_STEP_IN:
                    grown = _column_step(upper, +1, state)
                    if grown is None:
                        continue
                    target, gid, reason = grown, chain[index + 1][0], "the face step from the column below must stay under 3 in"
                else:
                    continue
                changes.append((gid, [state.designs[gid].b_in, state.designs[gid].h_in], list(target), reason))
                state = resized(state, gid, target)
                moved = True
            if not moved:
                break
    return state, changes


def enforce_banded_section_order(state):
    """Normalize growth BEFORE any solve: both beam/column dimensions never increase upward.

    Lower bands grow to support enlarged upper bands. Column steps retain the existing <=4-in
    dimension-change rule for supported transitions; system coupling is rechecked after each sweep.
    Only enlargement is allowed here. If no ladder can satisfy the rule, refuse the state.
    """
    changes = []
    for _ in range(8 * len(state.groups)):
        previous = state.identity()
        state,coupled = enforce_system_coupling(state)
        changes.extend(coupled)
        lines = {}
        for gid,group in state.groups.items():
            lines.setdefault((group['member_type'],group['location_class']),[]).append(gid)
        for gids in lines.values():
            gids.sort(key=lambda g:min(state.groups[g]['stories_or_floors']))
            for low,up in zip(gids,gids[1:]):
                a,b = state.designs[low],state.designs[up]
                if b.b_in > a.b_in + 1e-9 or b.h_in > a.h_in + 1e-9:
                    width,depth=max(a.b_in,b.b_in),max(a.h_in,b.h_in)
                    if a.is_column:
                        size=next((v for v in _column_sizes(state) if v>=max(width,depth)-1e-9),None)
                        target=None if size is None else (size,size)
                    else:
                        # A beam tie spans X/Y: its rung must fit both directions' ladders.
                        common=None
                        for g in _tied_groups(state,low):
                            rungs=set(beam_rungs(state.designs[g].member_type))
                            common=rungs if common is None else common & rungs
                        eligible=[r for r in common if r[0]>=width-1e-9 and r[1]>=depth-1e-9]
                        target=min(eligible,key=lambda r:(r[0]*r[1],r)) if eligible else None
                    if target is None:
                        raise mg.GroupedStateError(f"No ladder section for {low} can support the section above within the hard size rule.")
                    changes.append((low,[a.b_in,a.h_in],list(target),"both section dimensions must not increase upward"))
                    state=resized(state,low,target)
                elif a.is_column and max(a.b_in-b.b_in,a.h_in-b.h_in)>MAX_COLUMN_SIZE_STEP_IN+1e-9:
                    target=_column_step(b,+1,state)
                    if target is None:
                        raise mg.GroupedStateError(f"No column rung can support the transition above {low}.")
                    changes.append((up,[b.b_in,b.h_in],list(target),"retain the supported column transition step"))
                    state=resized(state,up,target)
        if state.identity()==previous:
            mg.require_section_order(state)
            return state,changes
    raise mg.GroupedStateError("The hard section-size envelope and system coupling did not settle.")


def beam_size_problem(state, gid, size):
    """Why a beam group may not take ``size`` between its own end columns and over the slab, or None."""
    trial = resized(state, gid, size)
    if mg.is_banded_policy(state.policy_id):
        problems = mg.section_order_problems(trial)
        if problems:
            return problems[0]
    b, h = size
    if b < min(0.3 * h, 10.0):
        return f"width {b:g} in is below min(0.3 h, 10 in) (ACI 318-19 18.6.2.1(b))"
    with mg.installed(trial):
        for group in _tied_groups(trial, gid):                   # every group of the tie takes the size
            design = trial.designs[group]
            d = h - mp.longitudinal_cover_in(design)
            for tag in trial.groups[group]["member_tags"]:
                member = trial.member(tag)
                clear = mp.beam_clear_span_in(member)
                if clear < 4.0 * d:
                    return f"clear span {clear:g} in of member {tag} is below 4 d = {4.0 * d:g} in (ACI 318-19 18.6.2.1(a))"
    if sp.SLAB_THICKNESS_IN is not None and h <= sp.SLAB_THICKNESS_IN:
        return "no deeper than the slab"
    return None


# ---- proposals ------------------------------------------------------------------------------------------------
def _band_of(state, story):
    return next(g["band"] for g in state.groups.values() if story in g["stories_or_floors"])


def _core_groups_of_beam(gid, types):
    """The column groups forming the cores of the joints a beam group threads (from the capacity design's joint types)."""
    return sorted({kind["core_group"] for kind in types.values()
                   for item in (kind.get("threading") or {}).values() if item.get("beam_group") == gid})


def propose_growth(evaluation, state):
    """The moves a not-feasible candidate asks for: {target: (to, reason)}; empty when none is known.

    ``target`` is "column_grade" / "beam_grade" with the next grade (strength first), or a group id (its design
    tie under the system policy) with a ladder size (size last; see LEVER_ORDER_ID). Room-for-bars failures go to
    the size directly. Under the system policy drift moves the perimeter frames only, a lane-limited beam moves
    the columns it threads, and the gravity ties grow for their own checks; the D6 coupling is applied afterwards
    by enforce_system_coupling.
    """
    moves = {}
    columns = sorted(g for g, group in state.groups.items() if group["member_type"] == "column")
    beams = sorted(g for g, group in state.groups.items() if group["member_type"] != "column")
    # Under the band policy every frame is part of the SFRS; under the system policy only the perimeter groups are.
    sfrs = {g for g in state.groups if not state.ties() or state.system_of(g) == mg.SYSTEM_SFRS}

    def grade_move(kind, reason):
        fc = next_grade(state, kind)
        if fc is None:
            return False
        moves.setdefault(GRADE_KEYS[kind], (fc, reason))
        return True

    def size_key(gid):
        return state.tie_of(gid) or gid

    def grow_column(gid, reason, grade_first=True):
        if grade_first and grade_move("column", reason):
            return
        size = _column_step(state.designs[gid], +1, state)
        if size is not None:
            moves.setdefault(size_key(gid), (size, reason))
        elif not grade_first:
            grade_move("column", reason)                        # no larger size on the ladder: the grade is what is left

    def grow_beam(gid, reason, prefer="deeper", grade_first=True):
        if grade_first and grade_move("beam", reason):
            return
        design = state.designs[gid]
        order = ("wider", "deeper", "larger") if prefer == "wider" else ("deeper", "wider", "larger")
        size = next((s for s in (_beam_step(design, move) for move in order) if s is not None), None)
        if size is not None:
            moves.setdefault(size_key(gid), (size, reason))
        elif not grade_first:
            grade_move("beam", reason)

    if evaluation["status"] == "infeasible":
        stage, evidence = evaluation.get("failed_stage"), evaluation.get("failure_evidence") or {}
        if stage == "slab_thickness":
            for gid in beams:
                grow_beam(gid, "no slab thickness passes on these beams: a deeper beam raises the beam-to-slab stiffness",
                          grade_first=False)
        elif stage == "gravity_analysis":
            for gid in columns:
                grow_column(gid, "the gravity analysis did not converge on these columns", grade_first=False)
        elif stage == "capacity_design" and evidence.get("group_id"):
            grow_column(evidence["group_id"], "a factored axial load is outside this section's strength domain")
        elif stage == "bar_rows":
            for gid in evidence.get("group_ids") or []:
                if gid in state.groups:
                    grow_beam(gid, "the beam's bar rows reach past mid-depth beside the beams it crosses", grade_first=False)
        return moves
    constraints = evaluation["constraints"]
    capacity = (evaluation.get("evidence") or {}).get("capacity") or {}
    types = capacity.get("joint_types") or {}
    blocked = {}
    for item in (constraints.get("exhausted_column_lines") or {}).values():
        # A tied column band's key joins several location groups. Resolve all of them back to
        # physical groups and the band-local size tie, so a failed upper chain can actually grow.
        for key in (item.get("blocked_at") or "").split('+'):
            if key:
                blocked[key] = item.get('blocked_by')
                if key in state.groups:
                    blocked[size_key(key)] = item.get('blocked_by')
    lane_limited = set(constraints.get("lane_limited_beam_groups") or [])
    for gid in constraints.get("exhausted_groups", []):
        if state.groups[gid]["member_type"] != "column":
            if gid in lane_limited:
                # The cages that reach the demand do not thread the column bars: room, so the columns the beam threads.
                cores = _core_groups_of_beam(gid, types) or [g for g in columns if g in sfrs] or columns
                for core in cores:
                    grow_column(core, "the beam's bars do not thread this column's cage in enough lanes", grade_first=False)
            else:
                grow_beam(gid, "no admissible cage reaches this group's flexural demand")
        elif not blocked:
            grow_column(gid, "no admissible cage chain along this column line (strength, strong column or transition)")
        elif gid in blocked or size_key(gid) in blocked:
            # Only the band at which the chain breaks grows; the bands below follow through the taper rule if they must.
            grow_column(gid, f"the cage chain of this column line breaks at this band: {blocked.get(gid) or blocked.get(size_key(gid))}")
    for item in constraints.get("capacity_design_failed_checks", []):
        check, location = item["id"], item.get("location") or ""
        room = check in ROOM_FAILURES
        if check.startswith("beam.capacity_shear") or check in ("beam.hoops_selected", "beam.cage_layout"):
            grow_beam(location, f"{check} is not met", prefer="wider", grade_first=not room)
        elif check.startswith("column.") and location in state.groups:
            grow_column(location, f"{check} is not met", grade_first=not room)
        elif check == "column.transition":
            upper = location.split(">", 1)[-1]
            if upper in state.groups:
                grow_column(upper, "the transition from the column below is not a supported one", grade_first=False)
        elif check == "beam.bar_stacking_clear_of_slab_mats":
            # The stacked bars of the beams at this joint type meet the slab mats: room, so the beams there widen.
            kind = types.get(location)
            for item in (kind or {}).get("threading", {}).values():
                if item.get("beam_group") in state.groups:
                    grow_beam(item["beam_group"], f"{check} is not met at joint type {location}", prefer="wider", grade_first=False)
        elif check in ("joint.shear_screen", "joint.through_bar_depth", "joint.terminating_bar_hook", "beam.bars_thread_column"):
            type_id = location.split("/")[1] if check == "joint.shear_screen" else location.split("/")[0]
            kind = types.get(type_id)
            if kind is not None:
                grow_column(kind["core_group"], f"{check} is not met at the joints this column forms the core of",
                            grade_first=check in ("joint.shear_screen", "joint.terminating_bar_hook"))
        elif check.startswith("gravity.") and location in state.groups:
            # A gravity group's 18.14 check (its capacity shear is the one that can fail on a selected cage).
            grow = grow_column if state.groups[location]["member_type"] == "column" else grow_beam
            grow(location, f"{check} is not met")
    for item in constraints.get("joint_checks_failed", []):
        if item["id"] != "scwb":
            continue
        joint_id = (item.get("location") or "").split("/")[0]
        for kind in types.values():
            if joint_id in kind["joints"]:
                grow_column(kind["core_group"], "the strong-column rule is not met at a joint of this column")
                if kind.get("column_above_group"):
                    grow_column(kind["column_above_group"], "the strong-column rule is not met at a joint of this column")
    for label in constraints.get("drift_failed_checks", []):
        # "demands.story_drift@story:<k>/Q<axis>/<axis>"
        try:
            where = label.split("@story:", 1)[1]
            story, axis = int(where.split("/")[0]), where.rsplit("/", 1)[1]
        except (IndexError, ValueError):
            continue
        band = _band_of(state, story)
        targets = [g for g in beams if g.startswith(f"{band}__beam_{axis}__") and g in sfrs]
        before = len(moves)
        for gid in targets:
            grow_beam(gid, f"story {story} exceeds the drift limit in {axis}")
        if len(moves) == before:
            for gid in (g for g in columns if g.startswith(f"{band}__") and g in sfrs):
                grow_column(gid, f"story {story} exceeds the drift limit in {axis} and its beams have no deeper rung")
    return moves


def reduction_order(state):
    """Group ids in the order the reduction phase offers them a smaller section (one representative per design tie
    under the system policy: the perimeter columns, the interior columns, the perimeter beams, the interior beams)."""
    ties = state.ties()
    if ties:
        families = ("sfrs_column", "gravity_column", "sfrs_beam", "gravity_beam")
        names = sorted(ties,key=lambda t:(-min(state.groups[ties[t][0]]['stories_or_floors']),families.index(mg.tie_kind(t))))
        return [ties[name][0] for name in names]
    bands = sorted({g["band"] for g in state.groups.values()}, reverse=True)
    order = []
    for band in bands:
        order += sorted(g for g, group in state.groups.items() if group["band"] == band and group["member_type"] == "column")
        order += sorted(g for g, group in state.groups.items() if group["band"] == band and group["member_type"] != "column")
    return order


def compare_quantities(new, old, tolerance=1e-9):
    """(decision, detail) of a feasible candidate against the current design under the dominance policy."""
    concrete = (new["concrete_total_in3"], old["concrete_total_in3"])
    steel = (new["total_steel_in3"], old["total_steel_in3"])
    detail = {"concrete_total_in3": {"new": concrete[0], "current": concrete[1]},
              "total_steel_in3": {"new": steel[0], "current": steel[1]},
              "frame_steel_in3": {"new": new["frame_steel_in3"], "current": old["frame_steel_in3"]},
              "distinct_form_sizes": {"new": new["distinct_form_size_count"], "current": old["distinct_form_size_count"]}}
    if steel[0] is None or steel[1] is None:
        return "unresolved", {**detail, "reason": ("total reinforcement is not available: the slab reinforcement layout is not "
                                                   "established, and an unknown quantity is not replaced by zero")}

    def sign(a, b):
        return -1 if a < b * (1.0 - tolerance) else 1 if a > b * (1.0 + tolerance) else 0

    c, s = sign(*concrete), sign(*steel)
    if c <= 0 and s <= 0 and (c < 0 or s < 0):
        return "accepted", {**detail, "reason": "neither concrete nor total reinforcement increases and at least one decreases"}
    if c == 0 and s == 0:
        if new["distinct_form_size_count"] < old["distinct_form_size_count"]:
            return "accepted", {**detail, "reason": "quantities tied; fewer distinct form sizes"}
        return "rejected", {**detail, "reason": "quantities tied and no fewer form sizes"}
    if c >= 0 and s >= 0:
        return "rejected", {**detail, "reason": "neither quantity decreases"}
    return "tradeoff", {**detail, "reason": ("one quantity decreases and the other increases: retained for review, not adopted "
                                             "(no cost data is used)")}


# ---- the search -----------------------------------------------------------------------------------------------
class CandidateLog:
    """Every candidate in order, in memory and (when a path is given) appended to a JSON-lines file at once."""

    def __init__(self, path=None):
        self.entries, self.path = [], None if path is None else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise RuntimeError(f"Candidate log {self.path} already exists; a search writes a new log in a new directory.")

    def add(self, entry):
        entry = {"index": len(self.entries) + 1, **entry}
        self.entries.append(entry)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(gd._json_safe(entry), allow_nan=False) + "\n")
        return entry


def _summary_line(entry):
    evaluation = entry.get("evaluation") or {}
    quantities = evaluation.get("quantities") or {}
    return (f"[grouped] #{entry['index']:>3} {entry['phase']:<11} {entry['decision']:<10} "
            f"{(entry.get('proposal') or {}).get('summary', ''):<46} "
            + (f"concrete {quantities['concrete_total_yd3']:.1f} yd3, frame steel {quantities['frame_steel_lb'] / 2000.0:.2f} t"
               if quantities else f"{evaluation.get('failed_stage', '')}: {str(evaluation.get('reason', ''))[:70]}"))


def search(seed, cfg, policy=None, log_path=None, verbose=True, context=None):
    """Run the grouped search from ``seed`` (a GroupedDesign). Returns {"final", "log", "stop", ...}.

    ``final`` is the evaluation of the design the search ends on (with its in-memory evidence), or None
    when no feasible design was found within the feasibility budget; the candidate log says why.
    """
    policy = policy or SearchPolicy()
    problems = policy.problems()
    if problems:
        raise ValueError("Search policy is not valid: " + "; ".join(problems))
    log, context = CandidateLog(log_path), ({} if context is None else context)
    started = time.perf_counter()

    def evaluate(state, phase, proposal):
        try:
            mg.require_section_order(state)
            evaluation = gd.evaluate_candidate(state, cfg, context, label=f"{phase}_{len(log.entries) + 1}")
            mg.require_section_order(evaluation.get('state') or state)
        except Exception as exc:                                    # an error, not a design outcome: keep the evidence, re-raise
            log.add({"phase": phase, "proposal": proposal, "decision": "error",
                     "decision_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                     "evaluation": {"status": "error", "sections": gd._section_table(state)}})
            raise
        return evaluation

    def record(phase, proposal, evaluation, decision, reason, comparison=None):
        entry = log.add({"phase": phase, "proposal": proposal, "decision": decision, "decision_reason": reason,
                         "longitudinal_grade_ksi": sp.FY_KSI,
                         "comparison": comparison, "evaluation": gd.public_evaluation(evaluation)})
        if verbose:
            print(_summary_line(entry), flush=True)
        return entry

    # ---- feasibility ----
    state, changes = enforce_column_taper(seed)
    state, coupling = enforce_system_coupling(state)
    proposal = {"kind": "seed", "summary": "seed design", "reason": "the search starts here",
                "taper_adjustments": [{"group_id": g, "from": a, "to": b, "reason": r} for g, a, b, r in changes],
                "system_coupling": [{"tie": t, "from": a, "to": b, "reason": r} for t, a, b, r in coupling]}
    current, stop, trials = None, None, 0
    while True:
        evaluation = evaluate(state, "feasibility", proposal)
        trials += 1
        if evaluation["status"] == "evaluated" and evaluation["feasible"]:
            record("feasibility", proposal, evaluation, "feasible", "every acceptance constraint is met")
            current = evaluation
            break
        moves = propose_growth(evaluation, evaluation.get("state") or state)
        reason = (f"{evaluation.get('failed_stage')}: {evaluation.get('reason')}" if evaluation["status"] == "infeasible"
                  else "acceptance constraints not met")
        record("feasibility", proposal, evaluation, "not_feasible", reason)
        if not moves:
            stop = {"reason": "feasibility_no_growth_move",
                    "detail": "the failed constraints name no group with a larger ladder section to try"}
            break
        if trials >= policy.max_feasibility_trials:
            stop = {"reason": "feasibility_budget_exhausted",
                    "detail": f"{trials} candidates evaluated without a feasible one; budget exhaustion is not infeasibility"}
            break
        base = apply_moves(evaluation.get("state") or state, moves)
        state, changes = enforce_column_taper(base)
        state, coupling = enforce_system_coupling(state)
        proposal = {"kind": "growth", "summary": _moves_summary(moves),
                    "reason": "the previous candidate's failed constraints (strength first, then the section)",
                    "moves": _move_rows(moves),
                    "taper_adjustments": [{"group_id": g, "from": a, "to": b, "reason": r} for g, a, b, r in changes],
                    "system_coupling": [{"tie": t, "from": a, "to": b, "reason": r} for t, a, b, r in coupling]}
    feasibility_trials = trials

    # ---- reduction ----
    tradeoffs, reduction_trials, skipped, tried = [], 0, [], set()
    if current is not None:
        stop = None
        while stop is None:
            accepted_in_sweep = False
            for gid in reduction_order(current["state"]):
                if reduction_trials >= policy.max_reduction_trials:
                    stop = {"reason": "reduction_budget_exhausted",
                            "detail": f"{reduction_trials} reduction candidates evaluated; smaller sections may remain untried"}
                    break
                base = current["state"]
                design = base.designs[gid]
                # A column has one next smaller square; a beam has several lighter rungs, strongest first: the
                # first one that fits its own columns and the slab is the move, the others are logged as skipped.
                options = ([_column_step(design, -1, state)] if design.is_column else _beam_step(design, "smaller")[:MAX_BEAM_REDUCTION_OPTIONS])
                size = proposal = None
                for option in (o for o in options if o is not None):
                    problem = ((column_taper_problem(base, gid, option) if design.is_column
                                else beam_size_problem(base, gid, option)) or system_coupling_problem(base, gid, option))
                    proposal = {"kind": "reduction", "group_id": gid, "from": [design.b_in, design.h_in], "to": list(option),
                                "summary": f"{gid} {design.b_in:g}x{design.h_in:g} -> {option[0]:g}x{option[1]:g}",
                                "reason": ("next smaller ladder section for this group (less concrete, the least loss of "
                                           "b h^2 for a beam); every group's reinforcement is re-selected")}
                    if problem is None:
                        size = option
                        break
                    key = (gid, tuple(option), current["result_identity"])
                    if key not in skipped:
                        skipped.append(key)
                        log.add({"phase": "reduction", "proposal": proposal, "decision": "skipped", "decision_reason": problem,
                                 "evaluation": None})
                if size is None:
                    continue
                key = (gid, tuple(size), current["result_identity"])
                if key in tried:
                    continue                                        # already evaluated against this very design
                tried.add(key)
                evaluation = evaluate(resized(base, gid, size), "reduction", proposal)
                reduction_trials += 1
                if evaluation["status"] != "evaluated" or not evaluation["feasible"]:
                    reason = (f"{evaluation.get('failed_stage')}: {evaluation.get('reason')}" if evaluation["status"] == "infeasible"
                              else "acceptance constraints not met; the previous design is restored")
                    record("reduction", proposal, evaluation, "rejected", reason)
                    gd.install_evaluation(current)                  # rollback: members, slab, transfer and layout
                    continue
                decision, comparison = compare_quantities(evaluation["quantities"], current["quantities"], policy.quantity_tolerance)
                record("reduction", proposal, evaluation, decision, comparison["reason"], comparison)
                if decision == "accepted":
                    current, accepted_in_sweep = evaluation, True
                else:
                    gd.install_evaluation(current)                  # rollback: members, slab, transfer and layout
                if decision in ("tradeoff", "unresolved"):
                    tradeoffs.append({"candidate_index": log.entries[-1]["index"], "decision": decision, "proposal": proposal,
                                      "comparison": comparison, "sections": evaluation["sections"],
                                      "quantities": {k: v for k, v in evaluation["quantities"].items() if k != "groups"},
                                      "margins": evaluation["margins"]})
            if stop is None and not accepted_in_sweep:
                stop = {"reason": "reduction_sweep_without_accepted_move",
                        "detail": "one full sweep over the groups adopted no smaller section under the declared policy"}
        gd.install_evaluation(current)
    return {"version": SEARCH_VERSION, "policy": asdict(policy), "final": current, "stop": stop,
            "log": log.entries, "tradeoffs": tradeoffs,
            "counts": {"feasibility_candidates": feasibility_trials, "reduction_candidates": reduction_trials,
                       "skipped_moves": len(skipped), "accepted_reductions": sum(1 for e in log.entries
                                                                               if e["phase"] == "reduction" and e["decision"] == "accepted"),
                       "tradeoffs": len(tradeoffs)},
            "elapsed_seconds": time.perf_counter() - started,
            "interpretation": ("the stop reason says how this bounded search ended; none of them is evidence that no smaller "
                               "or no feasible frame exists, and the result is not a cost optimum")}
