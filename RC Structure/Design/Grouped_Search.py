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
from Design.Section_Design import COLUMN_SIZES_IN, beam_ladder
from Model import Member_Groups as mg
from Model import Member_Properties as mp

SEARCH_VERSION = "grouped_search_v1"
IMPROVEMENT_POLICY_ID = "dominance_concrete_and_total_reinforcement_v1"
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

    def problems(self):
        issues = []
        for name in ("max_feasibility_trials", "max_reduction_trials"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                issues.append(f"{name} must be a nonnegative integer")
        if self.improvement_policy != IMPROVEMENT_POLICY_ID:
            issues.append(f"unknown improvement policy {self.improvement_policy!r}")
        return issues


# ---- ladders ------------------------------------------------------------------------------------------------
def beam_rungs(member_type):
    """[(b, h)] offered to the beams of one direction, lightest first (flexural proxy b h^2)."""
    span = sp.BAY_X if member_type == "beam_x" else sp.BAY_Y
    rungs = {(b, h) for b, h, _fc in beam_ladder(span_in=span, story_height_in=sp.STORY_H)}
    return sorted(rungs, key=lambda r: (r[0] * r[1] ** 2, r[1], r[0]))


def _column_step(design, direction):
    sizes = list(COLUMN_SIZES_IN)
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
    design = state.designs[gid]
    return state.with_designs({gid: replace(design, b_in=float(size[0]), h_in=float(size[1]))})


def _chains(state):
    return {location: selection._column_chain(state, location) for location in mg.COLUMN_LOCATIONS}


def column_taper_problem(state, gid, size):
    """Why a column group may not take ``size`` beside the bands below and above it, or None."""
    for chain in _chains(state).values():
        if gid not in chain:
            continue
        index = chain.index(gid)
        if index > 0:
            lower = state.designs[chain[index - 1]].b_in
            if size[0] > lower:
                return f"would be larger ({size[0]:g} in) than the column below it ({lower:g} in)"
            if lower - size[0] > MAX_COLUMN_SIZE_STEP_IN:
                return (f"would step in {0.5 * (lower - size[0]):g} in per face from the column below ({lower:g} in): "
                        "3 in or more needs dowels (ACI 318-19 10.7.4.2), not a declared transition")
        if index < len(chain) - 1:
            upper = state.designs[chain[index + 1]].b_in
            if size[0] < upper:
                return f"would be smaller ({size[0]:g} in) than the column above it ({upper:g} in)"
            if size[0] - upper > MAX_COLUMN_SIZE_STEP_IN:
                return (f"would leave the column above ({upper:g} in) stepping in {0.5 * (size[0] - upper):g} in per face: "
                        "3 in or more needs dowels (ACI 318-19 10.7.4.2), not a declared transition")
    return None


def enforce_column_taper(state):
    """Grow columns until no line grows upward or steps in too far. Returns (state, [(gid, from, to, reason)])."""
    changes = []
    for chain in _chains(state).values():
        for _ in range(4 * len(chain)):
            moved = False
            for index in range(len(chain) - 1):
                lower, upper = state.designs[chain[index]], state.designs[chain[index + 1]]
                if upper.b_in > lower.b_in:
                    target, gid, reason = (upper.b_in, upper.h_in), chain[index], "a column may not be smaller than the one above it"
                elif lower.b_in - upper.b_in > MAX_COLUMN_SIZE_STEP_IN:
                    grown = _column_step(upper, +1)
                    if grown is None:
                        continue
                    target, gid, reason = grown, chain[index + 1], "the face step from the column below must stay under 3 in"
                else:
                    continue
                changes.append((gid, [state.designs[gid].b_in, state.designs[gid].h_in], list(target), reason))
                state = resized(state, gid, target)
                moved = True
            if not moved:
                break
    return state, changes


def beam_size_problem(state, gid, size):
    """Why a beam group may not take ``size`` between its own end columns and over the slab, or None."""
    trial = resized(state, gid, size)
    b, h = size
    if b < min(0.3 * h, 10.0):
        return f"width {b:g} in is below min(0.3 h, 10 in) (ACI 318-19 18.6.2.1(b))"
    with mg.installed(trial):
        design = trial.designs[gid]
        d = h - mp.longitudinal_cover_in(design)
        for tag in trial.groups[gid]["member_tags"]:
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


def propose_growth(evaluation, state):
    """The groups a not-feasible candidate asks to grow: {gid: (new size, reason)}. Empty when none is known."""
    moves = {}

    def grow_column(gid, reason):
        size = _column_step(state.designs[gid], +1)
        if size is not None and gid not in moves:
            moves[gid] = (size, reason)

    def grow_beam(gid, reason, prefer="deeper"):
        design = state.designs[gid]
        order = ("wider", "deeper", "larger") if prefer == "wider" else ("deeper", "wider", "larger")
        size = next((s for s in (_beam_step(design, move) for move in order) if s is not None), None)
        if size is not None and gid not in moves:
            moves[gid] = (size, reason)

    columns = sorted(g for g, group in state.groups.items() if group["member_type"] == "column")
    beams = sorted(g for g, group in state.groups.items() if group["member_type"] != "column")
    if evaluation["status"] == "infeasible":
        stage, evidence = evaluation.get("failed_stage"), evaluation.get("failure_evidence") or {}
        if stage == "slab_thickness":
            for gid in beams:
                grow_beam(gid, "no slab thickness passes on these beams: a deeper beam raises the beam-to-slab stiffness")
        elif stage == "gravity_analysis":
            for gid in columns:
                grow_column(gid, "the gravity analysis did not converge on these columns")
        elif stage == "capacity_design" and evidence.get("group_id"):
            grow_column(evidence["group_id"], "a factored axial load is outside this section's strength domain")
        return moves
    constraints = evaluation["constraints"]
    capacity = (evaluation.get("evidence") or {}).get("capacity") or {}
    types = capacity.get("joint_types") or {}
    blocked = {item["blocked_at"]: item.get("blocked_by") for item in (constraints.get("exhausted_column_lines") or {}).values()
               if item.get("blocked_at")}
    for gid in constraints.get("exhausted_groups", []):
        if state.groups[gid]["member_type"] != "column":
            grow_beam(gid, "no admissible cage reaches this group's flexural demand")
        elif not blocked:
            grow_column(gid, "no admissible cage chain along this column line (strength, strong column or transition)")
        elif gid in blocked:
            # Only the band at which the chain breaks grows; the bands below follow through the taper rule if they must.
            grow_column(gid, f"the cage chain of this column line breaks at this band: {blocked[gid]}")
    for item in constraints.get("capacity_design_failed_checks", []):
        check, location = item["id"], item.get("location") or ""
        if check.startswith("beam.capacity_shear") or check in ("beam.hoops_selected", "beam.cage_layout"):
            grow_beam(location, f"{check} is not met", prefer="wider")
        elif check.startswith("column.") and location in state.groups:
            grow_column(location, f"{check} is not met")
        elif check == "column.transition":
            upper = location.split(">", 1)[-1]
            if upper in state.groups:
                grow_column(upper, "the transition from the column below is not a supported one")
        elif check in ("joint.shear_screen", "joint.through_bar_depth", "joint.terminating_bar_hook", "beam.bars_thread_column"):
            type_id = location.split("/")[1] if check == "joint.shear_screen" else location.split("/")[0]
            kind = types.get(type_id)
            if kind is not None:
                grow_column(kind["core_group"], f"{check} is not met at the joints this column forms the core of")
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
        targets = [g for g in beams if g.startswith(f"{band}__beam_{axis}__")]
        before = len(moves)
        for gid in targets:
            grow_beam(gid, f"story {story} exceeds the drift limit in {axis}")
        if len(moves) == before:
            for gid in (g for g in columns if g.startswith(f"{band}__")):
                grow_column(gid, f"story {story} exceeds the drift limit in {axis} and its beams have no deeper rung")
    return moves


def reduction_order(state):
    """Group ids in the order the reduction phase offers them a smaller section."""
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
            evaluation = gd.evaluate_candidate(state, cfg, context, label=f"{phase}_{len(log.entries) + 1}")
        except Exception as exc:                                    # an error, not a design outcome: keep the evidence, re-raise
            log.add({"phase": phase, "proposal": proposal, "decision": "error",
                     "decision_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                     "evaluation": {"status": "error", "sections": gd._section_table(state)}})
            raise
        return evaluation

    def record(phase, proposal, evaluation, decision, reason, comparison=None):
        entry = log.add({"phase": phase, "proposal": proposal, "decision": decision, "decision_reason": reason,
                         "comparison": comparison, "evaluation": gd.public_evaluation(evaluation)})
        if verbose:
            print(_summary_line(entry), flush=True)
        return entry

    # ---- feasibility ----
    state, changes = enforce_column_taper(seed)
    proposal = {"kind": "seed", "summary": "seed design", "reason": "the search starts here",
                "taper_adjustments": [{"group_id": g, "from": a, "to": b, "reason": r} for g, a, b, r in changes]}
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
        base = evaluation.get("state") or state
        for gid, (size, _why) in moves.items():
            base = resized(base, gid, size)
        state, changes = enforce_column_taper(base)
        proposal = {"kind": "growth", "summary": f"grow {len(moves)} group(s)",
                    "reason": "the previous candidate's failed constraints",
                    "moves": [{"group_id": gid, "to": list(size), "reason": why} for gid, (size, why) in sorted(moves.items())],
                    "taper_adjustments": [{"group_id": g, "from": a, "to": b, "reason": r} for g, a, b, r in changes]}
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
                options = ([_column_step(design, -1)] if design.is_column else _beam_step(design, "smaller")[:MAX_BEAM_REDUCTION_OPTIONS])
                size = proposal = None
                for option in (o for o in options if o is not None):
                    problem = (column_taper_problem(base, gid, option) if design.is_column
                               else beam_size_problem(base, gid, option))
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
