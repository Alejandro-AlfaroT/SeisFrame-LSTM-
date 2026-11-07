"""Demand-sized beam-flexure jumps of the genetic search (2026-10-31): an analysis-free screen, on the evaluated parent's
frozen member demands, of the farther legal catalog options of the beam levers a flexural shortfall's route offers.

Trigger. An authenticated beam flexural shortfall: a failed member.strength component routed as beam_flexure,
beam_lane_limited_strength or beam_threading_constrained_strength (Design.Genetic_Diagnostics), bound to the parent and
the run's basis, its dependencies shown, and its pair finite and positive: the governing member moment Mu against that
member's own phi Mn, per sign. The route already offers the beam's depth and width (and, except lane-limited, the beam
concrete family's grade) one local catalog step away (Genetic_Operators.frontier, the next grade). The screen adds at
most one farther option sized on the deficit per size gene (proposed under whichever of depth and width contains it)
and one per concrete family (grade_up).

Requirement. With the parent's demand frozen, for each failed component

    required capacity       = Mu / target DCR         (kip-in; GenerationConfig.beam_jump_target_dcr)
    capacity ratio required = (Mu / phi Mn) / target DCR

so a component at Mu / phi Mn = 1.728 with a target of 0.85 needs about 2.03 times its phi Mn. It is a capacity
requirement, never a multiplier of the beam's dimensions or steel: whether an option reaches it is priced.

Pricing (price_unit). The beam unit (the groups of the size tie, one shared cage) is priced as the selection prices it
(Group_Selection.select_beam_groups: evaluate_beam_candidate per group on the member-paired frozen demands, merged over
the tie) in the parent's realized design, every other group's cage frozen: the bar rows and effective depths the layering
gives the cage beside the crossing beams and column cages, the composite T/L strength of the final check, the clear
spacing, layer cap, threading, row depth, strength balance and minimum steel, and the face ratio limit
min(18.6.3.1, RebarConfig.rho_beam_practical_max) per face on the actual rows' effective depth (the run's DesignConfig;
nothing here assumes a ratio). An option's estimate is the selection pool's best cage: the strongest cage that threads
the frozen column cages when any does, otherwise the strongest admissible one. Cages are tried from the most steel down;
while a threading cage is sought, a cage that its bar size's lanes (from the layering) cannot thread is deferred, and
once one is admissible the scan stops below STEEL_SCAN_FRACTION of its steel. On the 34 sections screened for o0005 of
the 10-minute run (2026-10-31) the estimate equalled the full selection's best threading cage exactly, at about a tenth
of its time.

Options. A farther option is a catalog option of the gene in a lever's direction (depth: deeper and no narrower; width:
wider and no shallower; grade: a higher grade of the family's ladder) whose requested genome the profile repair keeps
(Genetic_Search.DesignSpace.repair: lower bands grow under an upper-band option, an induced change the proposal records).
It is priced after that same repair (Grouped_Search.enforce_banded_section_order). The parent's own section is priced
first and must reproduce the recorded utilization within BASELINE_TOLERANCE, or the gene's screen gives no target
(calibration_mismatch). Then the farthest option. If it reaches the target, the search narrows toward the smallest option
that does: bisection on the catalog order between the parent and the smallest option known to reach it, then the
options below that one not yet priced, nearest first (the estimates need not rise with the catalog order); the result
states whether every smaller option was priced (smallest_confirmed). If the farthest option does not reach the target,
the others from the farthest down. Bounded by GenerationConfig.beam_jump_max_sections pricings per gene (a grade option
costs one per beam unit it prices) and beam_jump_max_seconds per observation (and the run's remaining wall time); an
option the budget left unpriced is listed as unscreened, never as infeasible.

Shortlist. Per gene: the smallest priced option estimated to reach the target (catalog order: the flexural proxy b h^2,
or the grade ladder); when none is, the first priced option in catalog order within beam_jump_partial_tolerance of the
best estimate that improves on the parent and on every other local step priced by at least beam_jump_min_gain (a useful
partial improvement, outcome target_unmet_estimated: the end of the catalog is never called adequate because the
catalog ends). The pool of the estimate (threading, or admissible when no cage threads the frozen column cages) is kept
with it. The local steps stay proposals; an option equal to one only adds its estimate to that step's attribution.

Units. Every shortlisted option carries the estimate of each beam unit it prices (``units``) and every gene screen the
parent's own estimate of each unit (``unit_baselines``; a concrete family's from its units' size screens), so the
improvement a diagnostic is credited with is always its own unit's estimate against the same unit's baseline: a family
grade option's gene-level estimate is its worst unit's, which may be another band than the diagnostic's (v2,
2026-11-01). For the same reason every gene screen gives, per unit, the smallest priced option whose estimate for that
unit reaches the target with a cage that threads the frozen column cages (``unit_targets``: the option, the unit's
estimate, the option's gene-level estimate and pool, and smallest_confirmed: that unit was priced at every smaller legal
option, in a row the screen kept; a pricing the deadline cut confirms nothing for any unit of its row, as for the gene):
a smaller grade may bring one band to the target while the family's worst band needs more.

Not authoritative. The demand is frozen: a larger beam is stiffer and heavier, and the reanalysis redistributes demand;
the column cages are frozen, so a coordination of the full evaluation may thread a cage the screen does not consider. An
estimate ranks a proposal; it certifies nothing and discards nothing. Every proposal is still repaired, finalized,
deduplicated, evaluated completely and assessed by the code-check contract.

Section_Design.beam_rung_carries_moment (the uniform route's moment-based rung helper) is not used: it assumes one
tension layer at d = h - 3.5 in on a rectangular web (no flange, compression steel or mats), a fixed ratio
(BEAM_GROWTH_STEEL_RATIO, not the configured ceiling), phi 0.9 without a tension-control check, one moment for both
signs and no bar, lane, layer, minimum-steel or balance rule. Stacked rows put the tension centroid 5.5 to 8.0 in from
the face on o0005 (d = 22.0 to 24.5 in at h = 30 in), so it overstates the sections the search screens.
"""
from __future__ import annotations

from Design.Config import validate_beam_practical_ceiling

import contextlib
import math
import time
from dataclasses import replace

import Structure_Parameters as sp
from Design import Group_Checks as checks
from Design import Group_Selection as selection
from Design import Grouped_Search as gs
from Design.Genetic_Diagnostics import RULES, finite
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg

VERSION = "beam_flexure_capacity_jump_screen_v2"   # v2: per-unit baselines, estimates and targets
LOCAL = "local_frontier_v1"
DEMAND_SIZED = "demand_sized_v1"
POLICIES = (LOCAL, DEMAND_SIZED)
# The flexural reasons and the levers of their routes the screen may extend (Genetic_Operators.route).
LEVERS = {"beam_flexure": ("depth", "width", "grade_up"),
          "beam_lane_limited_strength": ("depth", "width"),
          "beam_threading_constrained_strength": ("depth", "width", "grade_up")}
STEEL_SCAN_FRACTION = 0.75        # the scan stops below this share of the first admissible threading cage's steel
BASELINE_TOLERANCE = 0.02         # relative agreement of the re-priced parent section with its recorded utilization
# The slab and floor environment an evaluation leaves in Structure_Parameters (Genetic_Generation.ENVIRONMENT_KEYS).
ENVIRONMENT_KEYS = ("SLAB_THICKNESS_IN", "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF", "SEISMIC_LIVE_LOAD_FRACTION",
                    "FLOOR_TRANSFER", "SLAB_ACTIONS", "SLAB_REINFORCEMENT")
SUMMARY_KEYS = ("gene", "kind", "from", "levers", "status", "outcome", "calibration", "baseline_dcr",
                "recorded_utilization", "capacity_ratio_required", "target_dcr", "meets_target", "partial",
                "smallest_confirmed", "unit_baselines", "unit_targets")


def in_direction(lever, old, new):
    """Whether ``new`` is a move of ``lever`` from ``old`` (the directions Genetic_Operators.finalize enforces)."""
    if lever == "grade_up":
        return new > old
    if lever == "depth":
        return new[1] > old[1] and new[0] >= old[0]
    if lever == "width":
        return new[0] > old[0] and new[1] >= old[1]
    return False


def required_capacity(demand, capacity, target_dcr):
    """The capacity one failed flexural component needs at ``target_dcr`` with its demand frozen: {"utilization",
    "required_capacity_kip_in", "capacity_ratio_required", "target_dcr"}; None unless the demand and capacity are
    finite and positive and the target lies in (0, 1]."""
    if not (finite(demand) and finite(capacity) and demand > 0 and capacity > 0 and finite(target_dcr)
            and 0 < target_dcr <= 1):
        return None
    utilization = demand / capacity
    return {"utilization": utilization, "required_capacity_kip_in": demand / target_dcr,
            "capacity_ratio_required": utilization / target_dcr, "target_dcr": target_dcr}


def flexural_shortfalls(diagnostics, ctx, target_dcr):
    """[(diagnostic, requirement)] of the authenticated beam flexural shortfalls of ``ctx``'s parent: the diagnostics the
    router routes (bound, failed, valid, applicable, their dependencies shown, a kip-in "<=" pair above 1) of a beam
    group of the parent."""
    result = []
    for d in diagnostics:
        reason = d.get("reason")
        if reason not in LEVERS or d.get("status") != "fail" or d.get("applicable") is not True \
                or d.get("evidence_valid") is not True:
            continue
        if d.get("parent_identity") != ctx["parent_identity"] or d.get("basis_identity") != ctx["basis_identity"]:
            continue
        if any((d.get("dependencies") or {}).get(k) is not True for k in RULES[reason][1]):
            continue
        group = ctx["groups"].get(d.get("group_id"))
        if group is None or not group["member_type"].startswith("beam_") or d.get("comparison") != "<=" \
                or d.get("units") != "kip-in":
            continue
        need = required_capacity(d.get("demand"), d.get("capacity"), target_dcr)
        if need is not None and need["utilization"] > 1.0:
            result.append((d, need))
    return result


def _steel(cage):
    return (cage.top_bars + cage.bot_bars) * sp.rebar_area(cage.bar_size)


def price_unit(state, gids, rows_by, cfg, *, deadline=None, clock=time.perf_counter):
    """The selection pool's best cage of the beam unit ``gids`` in ``state`` on the frozen demand rows ``rows_by``
    ({gid: Group_Checks.beam_demand_rows}); the parent's slab and floor environment must be in force.

    Returns {"status": "priced" | "no_admissible_cage" | "no_demand" | "deadline", "estimated_dcr", "pool": "threading"
    | "admissible", "cage", "best_threading_dcr", "best_nonthreading_dcr" (of the cages priced; the deferred
    nonthreading cages are priced only when no cage threads), "cages_priced", "cages_deferred", "domain"}."""
    validate_beam_practical_ceiling(cfg)
    gids = tuple(gids)
    current = state.designs[gids[0]]
    base = {"estimated_dcr": None, "pool": None, "cage": None, "best_threading_dcr": None, "best_nonthreading_dcr": None,
            "cages_priced": 0, "cages_deferred": 0, "domain": 0}
    if any(not rows_by.get(gid) for gid in gids):
        return {**base, "status": "no_demand"}
    rows = [row for gid in gids for row in rows_by[gid]]
    demand = (max(r[2] for r in rows), max(r[3] for r in rows))
    with mg.installed(state):
        depths = [selection._beam_through_depth(state, gid) for gid in gids]
        paired = {gid: selection._paired_beam_demand(state, gid, rows_by[gid]) for gid in gids}
    through = min((d for d in depths if d is not None), default=None)
    candidates = selection.beam_cage_candidates(current, cfg, through)
    order = sorted(candidates, key=lambda c: (-_steel(c), -c.bar_size, c.top_bars, c.bot_bars))
    lanes = {}                     # bar size -> {gid: lanes per layer, None where the rows are not stacked}

    def learn(cage):
        if cage.bar_size in lanes:
            return
        trial = selection._trial_state(state, gids[0], replace(cage, member_type=state.groups[gids[0]]["member_type"]))
        with mg.installed(trial):
            try:
                basis = bar_layers.arrangement()["_basis"]
            except mg.GroupedStateError:
                lanes[cage.bar_size] = {}
                return
        governing = basis.get("bars_per_layer_governing") or {}
        lanes[cage.bar_size] = {gid: (governing.get(gid) if basis.get("stacked") else None) for gid in gids}

    def may_thread(cage):
        known = lanes.get(cage.bar_size) or {}
        if len(known) != len(gids):
            return True            # unknown lanes: priced, never presumed blocked
        n = max(cage.top_bars, cage.bot_bars)
        return all(fit is None or (fit > 0 and math.ceil(n / fit) <= sp.BEAM_BAR_MAX_LAYERS) for fit in known.values())

    best, priced = {True: None, False: None}, [0]

    def price(cage):
        results = [selection.evaluate_beam_candidate(state, gid, replace(cage, member_type=state.groups[gid]["member_type"]),
                                                      demand, cfg, paired[gid][0], paired_demand=paired[gid][1])
                   for gid in gids]
        priced[0] += 1
        if not all(r["admissible"] for r in results):
            return None
        merged = selection._merge_beam_results(results)
        if best[merged["threads"]] is None or merged["dcr"] < best[merged["threads"]][0]:
            best[merged["threads"]] = (merged["dcr"], cage, merged)
        return merged

    status, first, deferred = "priced", None, []
    for cage in order:
        if first is not None and _steel(cage) < STEEL_SCAN_FRACTION * first:
            break
        if deadline is not None and clock() > deadline:
            status = "deadline"
            break
        learn(cage)
        if not may_thread(cage):
            deferred.append(cage)
            continue
        merged = price(cage)
        if merged is not None and merged["threads"] and first is None:
            first = _steel(cage)
    if status == "priced" and best[True] is None:
        for cage in deferred:      # no threading cage: the pool is every admissible cage
            if deadline is not None and clock() > deadline:
                status = "deadline"
                break
            price(cage)
    result = {**base, "cages_priced": priced[0], "cages_deferred": len(deferred) if best[True] is not None else 0,
              "domain": len(candidates), "best_threading_dcr": None if best[True] is None else best[True][0],
              "best_nonthreading_dcr": None if best[False] is None else best[False][0]}
    chosen = best[True] or best[False]
    if status == "deadline" or chosen is None:
        return {**result, "status": "deadline" if status == "deadline" else "no_admissible_cage"}
    dcr, cage, merged = chosen
    return {**result, "status": "priced", "estimated_dcr": dcr, "pool": "threading" if best[True] else "admissible",
            "cage": {"bar_size": cage.bar_size, "top_bars": cage.top_bars, "bot_bars": cage.bot_bars,
                     "threads": merged["threads"], "layers": merged["layers"], "steel_in2": merged["steel_in2"]}}


@contextlib.contextmanager
def evaluated_environment(evaluation):
    """The slab and floor environment the evaluation was computed in, with its final state installed (what
    Grouped_Design.install_evaluation puts in force); Structure_Parameters and the installed design are restored on
    exit."""
    from Design import Design_Driver as driver
    saved = {key: getattr(sp, key) for key in ENVIRONMENT_KEYS}
    try:
        slab = evaluation["evidence"]["slab"]
        driver._apply_slab(slab["slab"])
        sp.FLOOR_TRANSFER, sp.SLAB_ACTIONS, sp.SLAB_REINFORCEMENT = slab["transfer"], slab["evidence"], slab["reinforcement"]
        with mg.installed(evaluation["state"]):
            yield
    finally:
        for key, value in saved.items():
            setattr(sp, key, value)


def _gene(ctx, gid, kind):
    matches = [(i, g) for i, g in enumerate(ctx["genes"]) if g["kind"] == kind and gid in g["groups"]]
    return matches[0] if len(matches) == 1 else (None, None)


def _value(value, kind):
    return tuple(float(x) for x in value) if kind == "size" else float(value)


def _plain(value):
    return list(value) if isinstance(value, tuple) else value


class _Budget:
    """Pricings per gene and the observation's deadline."""

    def __init__(self, max_sections, clock, deadline):
        self.max_sections, self.clock, self.deadline = int(max_sections), clock, deadline

    def expired(self):
        return self.deadline is not None and self.clock() > self.deadline


def _price_option(state, moves, units, rows_by, cfg, budget):
    """The estimate of ``units`` (beam units) after ``moves`` and the profile repair: their worst pool DCR. Returns
    (row, pricings)."""
    trial, changes = gs.enforce_banded_section_order(gs.apply_moves(state, moves))
    estimates, statuses, detail = [], [], []
    for unit in units:
        if budget.expired():
            statuses.append("deadline")
            break
        priced = price_unit(trial, unit, rows_by, cfg, deadline=budget.deadline, clock=budget.clock)
        statuses.append(priced["status"])
        estimates.append(priced["estimated_dcr"])
        detail.append({"groups": list(unit), "status": priced["status"], "estimated_dcr": priced["estimated_dcr"],
                       "pool": priced["pool"], "cage": priced["cage"], "cages_priced": priced["cages_priced"],
                       "best_threading_dcr": priced["best_threading_dcr"],
                       "best_nonthreading_dcr": priced["best_nonthreading_dcr"]})
    complete = len(statuses) == len(units) and all(s == "priced" for s in statuses)
    return {"status": "priced" if complete else next(s for s in statuses if s != "priced"),
            "estimated_dcr": max(estimates) if complete else None, "units": detail,
            "repair_changes": [[c[0], _plain(c[1]), _plain(c[2])] for c in changes]}, len(detail)


def _new_result(name, kind, old, levers, needs, params):
    recorded = max(need["utilization"] for _d, need in needs)
    return {"gene": name, "kind": kind, "from": _plain(old), "levers": sorted(levers), "status": "screened",
            "outcome": None, "calibration": None, "baseline_dcr": None, "recorded_utilization": recorded,
            "capacity_ratio_required": recorded / params["target_dcr"], "target_dcr": params["target_dcr"],
            "diagnostics": sorted(d["id"] for d, _need in needs), "meets_target": None, "partial": None,
            "smallest_confirmed": None, "unit_baselines": [], "unit_targets": [], "baseline": None, "priced": [],
            "rejected": [],
            "unscreened": [], "pricings": 0}


def _legal_options(ctx, space, index, gene, kind, old, levers, result):
    """[(catalog index, option, induced genes)] of the gene's farther options the profile repair keeps; the others are
    listed as rejected with the reason."""
    options = [_value(o, kind) for o in gene["options"]]
    legal = []
    for i, option in enumerate(options):
        if option == old or not any(in_direction(lever, old, option) for lever in levers):
            continue
        genome = list(ctx["genome"])
        genome[index] = i
        try:
            repaired, _trial, _changes = space.repair(tuple(genome))
        except (ValueError, mg.GroupedStateError) as exc:
            result["rejected"].append({"to": _plain(option), "reason": f"repair_refused: {exc}"[:300]})
            continue
        if repaired[index] != i:
            result["rejected"].append({"to": _plain(option), "reason": "repair_changed_the_requested_allele"})
            continue
        induced = sorted(ctx["genes"][j]["name"] for j, (a, b) in enumerate(zip(ctx["genome"], repaired))
                         if a != b and j != index)
        legal.append((i, option, induced))
    return legal


def _search(result, legal, local, price, budget, params, cost=1):
    """Price the farthest option. If it reaches the target, narrow toward the smallest option that does: bisection on
    the catalog order between the parent and the smallest option known to reach it, then the options below that one
    not yet priced, nearest first. Otherwise price the others from the farthest down. Each option costs ``cost``
    pricings of the gene's budget. Then shortlist; the options left unpriced are listed with the reason."""
    target, rows = params["target_dcr"], {}

    def visit(position):
        i, option, induced = legal[position]
        if result["pricings"] + cost > budget.max_sections or budget.expired():
            return None
        row, n = price(option)
        result["pricings"] += n
        if row["status"] == "deadline":
            return None                                    # no estimate: listed as unscreened (deadline) below
        row.update(to=_plain(option), order=i, induced_genes=induced, local_step=option in local,
                   meets_target=row["estimated_dcr"] is not None and row["estimated_dcr"] <= target)
        result["priced"].append(row)
        rows[position] = row
        return row

    farthest = visit(len(legal) - 1)
    adequate = None
    if farthest is not None and farthest["meets_target"]:
        low, adequate = -1, len(legal) - 1
        while adequate - low > 1:                          # exact when the estimates rise with the catalog order
            middle = (low + adequate) // 2
            row = visit(middle)
            if row is None:
                break
            if row["meets_target"]:
                adequate = middle
            else:
                low = middle
        for position in range(adequate - 1, -1, -1):        # the estimates need not rise with the order: the rest
            if position in rows:
                continue
            row = visit(position)
            if row is None:
                break
            if row["meets_target"]:
                adequate = position
    else:
        for position in range(len(legal) - 2, -1, -1):
            if visit(position) is None:
                break
    for position, (_i, option, _induced) in enumerate(legal):
        if position not in rows:
            reason = ("larger_than_an_adequate_option" if adequate is not None and position > adequate
                      else "deadline" if budget.expired() else "budget")
            result["unscreened"].append({"to": _plain(option), "reason": reason})
    _shortlist(result, params)
    if result["meets_target"] is not None:
        chosen = next(p for p, row in rows.items() if row["to"] == result["meets_target"]["to"])
        result["smallest_confirmed"] = all(p in rows for p in range(chosen))
    # Per unit: the smallest priced option (catalog order) whose estimate for that unit reaches the target with a cage
    # that threads the frozen column cages (a family's shortlist follows its worst unit). Confirmed when that unit was
    # priced at every smaller legal option in a kept row (a row the deadline cut is dropped whole, or, when an earlier
    # unit's status governs it, holds only the units priced before the cut): never a confirmation the pricing lacks.
    def priced_for(position, groups):
        return position in rows and any(tuple(u.get("groups") or ()) == groups and u.get("status") != "deadline"
                                        for u in rows[position].get("units") or [])

    targets = {}
    for position in sorted(rows):
        row = rows[position]
        for unit in row.get("units") or []:
            groups, dcr = tuple(unit.get("groups") or ()), unit.get("estimated_dcr")
            if (groups and groups not in targets and unit.get("pool") == "threading" and not isinstance(dcr, bool)
                    and isinstance(dcr, (int, float)) and math.isfinite(dcr) and dcr <= target):
                worst = (max(row["units"], key=lambda u: u["estimated_dcr"])
                         if row.get("estimated_dcr") is not None else None)
                targets[groups] = {"groups": list(groups), "to": row["to"], "estimated_dcr": dcr, "pool": "threading",
                                   "smallest_confirmed": all(priced_for(p, groups) for p in range(position)),
                                   "option_estimated_dcr": row.get("estimated_dcr"),
                                   "option_pool": None if worst is None else worst.get("pool")}
    result["unit_targets"] = [targets[groups] for groups in sorted(targets)]


def _shortlist(result, params):
    """The gene's target: the smallest priced option estimated to reach the target, else the first priced option in
    catalog order within the tolerance of the best estimate that beats the parent and every other local step priced by
    the declared gain (a useful partial improvement)."""
    target, gain, tolerance = params["target_dcr"], params["min_gain"], params["partial_tolerance"]
    priced = sorted((row for row in result["priced"] if row["estimated_dcr"] is not None), key=lambda r: r["order"])

    def entry(row):
        unit = max(row["units"], key=lambda u: u["estimated_dcr"])
        return {"to": row["to"], "estimated_dcr": row["estimated_dcr"], "pool": unit["pool"],
                "units": [{"groups": list(u["groups"]), "estimated_dcr": u["estimated_dcr"], "pool": u["pool"]}
                          for u in row["units"]]}
    adequate = [row for row in priced if row["estimated_dcr"] <= target]
    if adequate:
        result.update(outcome="target_met_estimated", meets_target=entry(adequate[0]))
        return
    if not priced:
        result["outcome"] = "no_option_priced"
        return
    best = min(row["estimated_dcr"] for row in priced)
    for row in (r for r in priced if r["estimated_dcr"] <= best + tolerance):
        bar = min([result["baseline_dcr"]] + [r["estimated_dcr"] for r in priced if r["local_step"] and r["to"] != row["to"]])
        if row["estimated_dcr"] <= bar - gain:
            result.update(outcome="target_unmet_estimated", partial=entry(row))
            return
    result["outcome"] = "target_unmet_no_useful_partial"


def _calibrate(result, baseline, recorded):
    result["baseline"] = baseline
    if baseline["status"] != "priced":
        result["outcome"] = f"baseline_{baseline['status']}"
        return False
    result["baseline_dcr"] = baseline["estimated_dcr"]
    agrees = abs(baseline["estimated_dcr"] - recorded) <= BASELINE_TOLERANCE * recorded
    result["calibration"] = "agrees" if agrees else "mismatch"
    if not agrees:
        result["outcome"] = "calibration_mismatch"
    return agrees


def _screen_size(state, name, entry, rows_by, ctx, cfg, space, params, budget):
    from Design.Genetic_Operators import frontier
    index, gene = entry["index"], entry["gene"]
    options = [_value(o, "size") for o in gene["options"]]
    old = options[ctx["genome"][index]]
    unit = tuple(sorted(gid for gid in gene["groups"] if not state.designs[gid].is_column))
    result = _new_result(name, "size", old, entry["levers"], entry["needs"], params)
    result["groups"] = list(unit)
    baseline, n = _price_option(state, {}, [unit], rows_by, cfg, budget)
    result["pricings"] += n
    if not _calibrate(result, baseline, result["recorded_utilization"]):
        return result
    result["unit_baselines"] = [{"groups": list(unit), "baseline_dcr": result["baseline_dcr"]}]
    legal = _legal_options(ctx, space, index, gene, "size", old, entry["levers"], result)
    if not legal:
        result["outcome"] = "no_legal_farther_option"
        return result
    local = {option for lever in entry["levers"] for option in frontier(options, old, lever)}
    _search(result, legal, local,
            lambda option: _price_option(state, {name: (option, "demand-sized jump screen")}, [unit], rows_by, cfg,
                                         budget), budget, params, cost=1)
    return result


def _screen_grade(state, name, entry, sizes, rows_by, ctx, cfg, space, params, budget):
    index, gene = entry["index"], entry["gene"]
    options = [_value(o, "grade") for o in gene["options"]]
    old = options[ctx["genome"][index]]
    result = _new_result(name, "grade", old, ("grade_up",), entry["needs"], params)
    units = []
    for size in sorted(entry["units"]):
        screened = sizes.get(size) or {}
        if screened.get("calibration") != "agrees":
            result["outcome"] = "unit_baseline_unavailable"
            result["status"] = "screened"
            return result
        units.append(tuple(screened["groups"]))
    result["groups"] = [list(u) for u in units]
    # The parent's own estimate, unit by unit, from the size screens (the same state and environment).
    result["baseline_dcr"] = max(sizes[s]["baseline_dcr"] for s in entry["units"])
    result["unit_baselines"] = [{"groups": list(sizes[s]["groups"]), "baseline_dcr": sizes[s]["baseline_dcr"]}
                                for s in sorted(entry["units"])]
    result["calibration"] = "agrees"
    legal = _legal_options(ctx, space, index, gene, "grade", old, ("grade_up",), result)
    if not legal:
        result["outcome"] = "no_legal_farther_option"
        return result
    higher = [o for o in options if o > old]
    local = {min(higher)} if higher else set()
    _search(result, legal, local,
            lambda option: _price_option(state, {name: (option, "demand-sized jump screen")}, units, rows_by, cfg,
                                         budget), budget, params, cost=len(units))
    return result


def screen_parent(state, actions, shortfalls, ctx, cfg, space, params, *, clock=time.perf_counter, deadline=None):
    """{gene name: screen} of the parent's flexural shortfalls (flexural_shortfalls): one per beam size tie that has
    one, and one per beam concrete family whose route offers grade_up. The parent's slab and floor environment and
    final state must be in force (evaluated_environment)."""
    validate_beam_practical_ceiling(cfg)
    sizes, grades = {}, {}
    for d, need in shortfalls:
        gid = d["group_id"]
        index, gene = _gene(ctx, gid, "size")
        if gene is None:
            continue
        entry = sizes.setdefault(gene["name"], {"index": index, "gene": gene, "needs": [], "levers": set()})
        entry["needs"].append((d, need))
        entry["levers"].update(lever for lever in LEVERS[d["reason"]] if lever != "grade_up")
        if "grade_up" in LEVERS[d["reason"]]:
            grade_index, grade = _gene(ctx, gid, "grade")
            if grade is not None:
                family = grades.setdefault(grade["name"], {"index": grade_index, "gene": grade, "needs": [], "units": set()})
                family["needs"].append((d, need))
                family["units"].add(gene["name"])
    rows_by = {gid: checks.beam_demand_rows(state.groups[gid]["member_tags"], actions)
               for entry in sizes.values() for gid in entry["gene"]["groups"] if not state.designs[gid].is_column}
    screens = {}
    worst = {name: max(need["utilization"] for _d, need in entry["needs"]) for name, entry in sizes.items()}
    for name in sorted(sizes, key=lambda n: (-worst[n], n)):
        screens[name] = _screen_size(state, name, sizes[name], rows_by, ctx, cfg, space, params,
                                     _Budget(params["max_sections"], clock, deadline))
    for name in sorted(grades):
        screens[name] = _screen_grade(state, name, grades[name], screens, rows_by, ctx, cfg, space, params,
                                      _Budget(params["max_sections"], clock, deadline))
    return screens


def ratio_ceiling(cfg):
    """The face steel ratio ceiling the pricing applies: the configured practical ceiling, the 18.6.3.1 code limit for
    the run's grade and the smaller of the two (Group_Selection.evaluate_beam_candidate)."""
    practical = None if cfg is None else validate_beam_practical_ceiling(cfg)
    code = sp.beam_rho_max_code()
    return {"practical_max": practical, "code_max": code, "effective_max": code if practical is None else min(code, practical)}


def summary(result):
    """The part of a gene's screen a diagnostic carries (the full screen stays on the observation)."""
    return {key: result.get(key) for key in SUMMARY_KEYS}


def screen_evaluation(evaluation, diagnosed, ctx, cfg, space, params, *, clock=time.perf_counter, deadline=None):
    """Screen an evaluated parent, while the evaluator's in-memory record is at hand (its final state, slab and floor
    environment and factored member actions), and attach the result to its flexural diagnostics as
    ``capacity_jump`` (bound to the parent and the basis). Returns the screen block, also kept as
    ``diagnosed["beam_jump_screen"]``. It stops at params["max_seconds"] from its start, or at ``deadline`` (the run's
    wall budget) when that comes first. A screen that fails is recorded and gives no target; it never raises (an
    interrupt does) and never changes the evaluation."""
    started = clock()
    limit = started + params["max_seconds"]
    deadline = limit if deadline is None else min(deadline, limit)
    block = {"version": VERSION, "policy": DEMAND_SIZED, "parent_identity": ctx.get("parent_identity"),
             "basis_identity": ctx.get("basis_identity"), "parameters": dict(params), "status": "screened",
             "ratio_ceiling": ratio_ceiling(cfg), "genes": {}, "pricings": 0, "seconds": 0.0, "deadline_reached": False,
             "error": None}
    shortfalls = []
    if diagnosed.get("status") != "diagnosed":
        block["status"] = "not_diagnosed"
    else:
        shortfalls = flexural_shortfalls(diagnosed.get("diagnostics") or [], ctx, params["target_dcr"])
        actions = (evaluation.get("evidence") or {}).get("actions") if isinstance(evaluation, dict) else None
        if not shortfalls:
            block["status"] = "no_flexural_shortfall"
        elif cfg is None:
            block["status"] = "no_design_configuration"
        elif evaluation.get("status") != "evaluated" or evaluation.get("state") is None or not isinstance(actions, list):
            block["status"] = "no_member_actions"
        else:
            try:
                with evaluated_environment(evaluation):
                    block["genes"] = screen_parent(evaluation["state"], actions, shortfalls, ctx, cfg, space, params,
                                                   clock=clock, deadline=deadline)
            except Exception as exc:                      # noqa: BLE001 -- a failed screen gives no target
                block.update(status="failed", genes={}, error=f"{type(exc).__name__}: {exc}"[:500])
    block["pricings"] = sum(g["pricings"] for g in block["genes"].values())
    block["deadline_reached"] = deadline is not None and clock() > deadline
    block["seconds"] = clock() - started
    for d, need in shortfalls:
        genes = {}
        for kind in ("size", "grade"):
            _index, gene = _gene(ctx, d["group_id"], kind)
            if gene is not None and gene["name"] in block["genes"]:
                genes[gene["name"]] = summary(block["genes"][gene["name"]])
        d["capacity_jump"] = {"version": VERSION, "parent_identity": block["parent_identity"],
                              "basis_identity": block["basis_identity"], "status": block["status"], **need,
                              "genes": genes}
    diagnosed["beam_jump_screen"] = block
    return block
