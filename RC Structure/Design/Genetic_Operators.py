"""The six mutation operators of the genetic search (research contract six_operator_contract_reference_v1, 2026-10-30).

  H1  strong-column / weak-beam: the next square size of each connected column tie; separately its family grade.
  H2  member strength: beam depth, beam width and beam-family grade alternatives; column size and column-family grade
      (for column capacity shear, no size step while the selected cage is at its minimum steel ratio, the
      longitudinal selection has settled and the probable-moment mechanism sets Ve under the column-own method: Ve
      then grows faster with the section than the 8 sqrt(f'c) b d limit. On the production route a cage that passes
      its P-M check is expected to have a mechanism-governed Ve; this is not established, so the record is read and
      the flag left unknown otherwise).
  H3  reinforcement space: bar room across the beam widens it, beam bars that do not thread grow the joint's core
      column, a slab-mat overlap needs its blocked dimension diagnosed first; a lane-limited beam strength shortfall
      (bar room, not strength) offers beam depth, beam width and the cores of the joints the beam frames into (only
      on a consistent joint topology, and only where it names such a joint), never a concrete grade; a
      threading-constrained strength shortfall (short of strength, with a stronger cage kept out by threading, v6)
      offers the beam's depth, width and grade, the blocking cores the record names among the joints the beam frames
      into (the next square size) and, where the selection's geometry-only look-ahead names one, the first larger core
      size whose lanes carry the stronger cage (lever column_lanes: a proposal for full evaluation, never a pass).
      Where the record separates the cores whose own cage blocks (blocking_core_cages), only those take column_up and
      column_lanes. A column transition that blocks (a band-boundary joint whose bar inventory closes the lanes) is
      named by both column groups; the first size of the column above, or of the core below, that opens that joint's
      own inventory is offered on that band (lever transition_lanes), only where the topology has a joint of the beam
      at which those two groups meet (v7, 2026-10-31 review). The same targeting (threading_levers) serves a failed
      threading row (beam_threading: the joint's core grows only where its own cage blocks; a transition-only blockage
      takes the band its joint look-ahead names, or is withheld when none is evidenced) and a lane-limited shortfall
      that carries the selection's threading fields (v8, 2026-10-31 follow-up review).
  H4  joint: joint shear compares core size and core-family grade; through-bar depth and hook embedment grow the core.
  H5  drift / stability: the SFRS columns of the implicated story and the beams (depth) of both floors that bound it,
      as separate alternatives.
  R1  random: a uniformly chosen mutable gene, then a uniformly chosen local neighbour (nearest lower/higher grade;
      the nondominated absolute dimension steps of a size).

A withheld lever (a rule that keeps a legal step back) is recorded with its rule, once per diagnostic and group,
only when the step exists (a larger square size); a lever without the evidence to aim it is recorded as withheld
for that reason: an inconsistent topology first, then a consistent one that names no joint the beam frames into.

Every guided proposal comes from a normalized diagnostic (Design.Genetic_Diagnostics) bound to the exact evaluated
parent and the run's evaluation basis; stale, unknown, invalid or dependency-incomplete evidence abstains with a
named reason, and a solver failure is never read as weakness. Each proposal requests one gene change. Size and
grade alternatives are separate proposals. The existing profile repair (Genetic_Search.DesignSpace.repair) then acts
on the whole profile, and ``finalize`` keeps the requested change apart from the changes repair induced: a repaired
genome equal to the parent is a no-op, a repair that removed or reversed the requested change is rejected. Nothing
here evaluates a design or grants acceptance. Crossover and adaptive operator weights are not part of this version.

v9 (2026-10-31, demand-sized jumps and severity-weighted selection; both are declared run policies, the earlier rules
remain as local_frontier_v1 and uniform_operator_v1):

  Demand-sized beam-flexure jumps (ctx["guided_policy"]["beam_jump"] == demand_sized_v1). A beam flexural shortfall
      (beam_flexure, beam_lane_limited_strength, beam_threading_constrained_strength) keeps its local steps and may add,
      per lever (depth, width, grade_up; the levers its route already offers), the one farther option the parent's
      screen shortlisted (Design.Genetic_Jumps: the smallest option estimated to reach the target DCR on the frozen
      parent demand, or a useful partial improvement when none is). The screen is read only when bound to this parent
      and basis and to the gene's current allele; otherwise, or when it shortlisted nothing, the route records why
      (route["jumps"]). A jump is one requested gene with the lever's own direction; its origin carries the estimate,
      which ranks the proposal and never certifies it.
  Severity-weighted guided selection (choose_guided with policy severity_weighted_v1). The routes that proposed are
      grouped by an actionable failure key (failure_key: the design tie, i.e. the size gene, of the member group of a
      strength, capacity-shear or bar-room failure; the beam's tie and the joint core of a threading failure; the joint
      core of a joint or strong-column failure; the column band of a story drift or stability failure). Each key takes
      the largest severity of its diagnostics (severity = max(0, u - 1), u = demand / capacity for "<=" checks and
      required / provided for ">=" checks, from the authenticated diagnostic; a check without a defensible ratio takes
      the declared categorical weight, and an unbounded one, no available clearance or capacity, the weight at the cap)
      and the weight selection_weight: floor + (1 - floor) min(severity, cap) / cap. Each distinct proposal is drawn
      with probability proportional to its share, the largest over the keys that offer it of the key's weight divided
      by the key's number of distinct proposals. So repeated rows, extra catalog alternatives and a proposal offered
      by two operators or two keys add no weight, a key that shares no proposal is drawn in proportion to its weight,
      and every eligible key keeps at least the floor. Only routes that proposed carry weight: unknown, stale, invalid
      or abstaining evidence acquires none. The weights follow the current violations, not learned operator
      performance; the random-exploration probability, parent selection and survival are unchanged.

v10 (2026-11-01, severity plus demand-sized preference; a third declared selection policy, the earlier two remain):

  severity_preferred_jumps_v1 (choose_guided). Under severity_weighted_v1 a failure's weight is spread evenly over its
      proposed fixes, so a screened demand-sized change and a small local step of the same failure are equally likely
      whatever their estimates. This policy changes the allocation between failure families and between fixes: it is
      not severity-only selection. Each guided draw is a mixture:
        with probability preferred_share (default 0.80) when the preferred pool is not empty, the preferred branch;
        otherwise, and with probability 1 - preferred_share, the severity_weighted_v1 draw over every eligible proposal.
      The guided probability itself (and so the R1 branch) is unchanged. The preferred pool holds the distinct
      proposals that the parent's demand-sized screen shortlisted (an origin with step demand_sized, authenticated by
      route: bound to this parent, basis and allele; a local origin of the same genome does not remove it) for a
      failure key whose utilization (its largest authenticated ratio) is at least severe_utilization (default 1.25).
      Each candidate is scored against its own beam unit: b, the unit's baseline estimate, and e, the unit's estimate
      under the change (``matched``; never another band's estimate), with target t and gain floor g (the jump policy's
      target DCR and min_gain):
        excluded when b <= t (no deficit in the screen); when e needs a cage that does not thread the frozen column
        cages (the unit's estimate pool is not "threading": a beam-only change does not reach anything there, the
        column clearance and coordination proposals stay in the normal branch); when b - e < g or b - e <= 0 (a
        negligible improvement); or when e <= t while a smaller option the route also proposes reaches the unit's
        target (larger_than_unit_target: the smaller one carries the preference);
        f = 1 when e <= t (estimated to reach the target), otherwise f = (b - e) / (b - t), the fraction of the unit's
        deficit closed (comparable across demand levels). The screen shortlists a gene's smallest option reaching the
        target, and a concrete family's follows its worst unit; so the screen also gives each unit's own smallest
        option reaching the target with a threading cage (unit_targets). Whatever the gene's own shortlist (also none,
        or another lever's move), when the route already proposes that option (a local step of a jump lever whose
        direction contains it) the route annotates it as that unit's demand-sized target. No proposal is added, so
        every policy draws over the same genomes; the annotation is evidence in the logs under every policy, counted
        apart (jump_routes.unit_target, jump_unit_target_steps_drawn). unit_smallest_confirmed records that the
        candidate is the unit's smallest target option and that unit was priced at every smaller legal option.
      A key K with weight w_K (selection_weight) and candidates C_K gives candidate p the score
        s(p, K) = w_K * f*_K * f(p, K) / sum over q in C_K of f(q, K),   f*_K = max over C_K of f,
      so a key's preferred mass is its weight times its best usefulness, split by usefulness (more alternatives add
      none). A candidate offered for several keys takes its largest score (never a sum; the first key in key order on
      a tie). P_preferred(p) = s(p) / sum of s. The final probability of a proposal is
        P(p) = a * P_preferred(p) + (1 - a) * P_normal(p),   a = preferred_share if the pool is not empty, else 0.
      Two rng.random() per guided draw (the branch, then the proposal over sorted genomes), so a seed reproduces the
      draws. A candidate already drawn for this parent in either branch (``exhausted``; the controller keeps them in its
      checkpoint) is excluded, so repair no-ops, duplicates and cache hits are not drawn again from it; an
      empty pool falls back to the normal branch with the reason (no_screened_candidate, no_eligible_severe_candidate,
      preferred_pool_exhausted). Only beam-flexure screens have an estimator: drift, column P-M, joint shear and the
      other checks keep their routes and their severity weighting in the normal branch, and so do the column clearance
      and coordination proposals of a threading-constrained beam (a beam-only change whose target stays unmet says so).
      The preferred_share and severe_utilization defaults are experimental search parameters, not code thresholds.
"""
from __future__ import annotations

import copy

from Design import Genetic_Jumps as jumps
from Design.Genetic_Diagnostics import RULES, digest, finite

VERSION = "seisframe_ga_operators_v10"
OPERATORS = ("H1", "H2", "H3", "H4", "H5", "R1")
UNIFORM = "uniform_operator_v1"
SEVERITY_WEIGHTED = "severity_weighted_v1"
PREFERRED = "severity_preferred_jumps_v1"
SELECTION_POLICIES = (UNIFORM, SEVERITY_WEIGHTED, PREFERRED)
SELECTION_VERSION = "severity_weighted_failure_selection_v1"
PREFERENCE_VERSION = "severe_demand_sized_preference_v1"
# The levers that move columns (clearance and coordination of a threading-constrained or lane-limited beam).
COLUMN_LEVERS = ("column_up", "column_lanes", "transition_lanes")
# Reasons whose failures share one key per actionable unit: the design tie (size gene) of the member group, whose
# section and grade genes every group of the tie shares.
MEMBER_KEYS = {"beam_flexure": "member_strength", "beam_lane_limited_strength": "member_strength",
               "beam_threading_constrained_strength": "member_strength", "column_pm": "member_strength",
               "beam_capacity_shear": "capacity_shear", "column_capacity_shear": "capacity_shear",
               "beam_width_room": "beam_width_room", "beam_row_depth": "beam_row_depth"}


def frontier(options, old, lever):
    """Minimal guided size steps: depth needs h_new > h_old and b_new >= b_old, width the converse; dominated
    alternatives (larger in both dimensions than another eligible one) are dropped, independent of catalog order."""
    old = tuple(old)
    options = [tuple(x) for x in options]
    dim = 1 if lever == "depth" else 0
    eligible = [x for x in options if x[dim] > old[dim] and x[1 - dim] >= old[1 - dim]]
    return [x for x in eligible if not any(y != x and all(a <= b for a, b in zip(y, x)) for y in eligible)]


def local_neighbors(options, old):
    """R1's support on one gene: nondominated absolute dimension steps of a size; nearest lower/higher grade."""
    if isinstance(old, (list, tuple)):
        rows = [(i, tuple(abs(a - b) for a, b in zip(x, old))) for i, x in enumerate(options) if tuple(x) != tuple(old)]
        return [i for i, d in rows if not any(e != d and all(a <= b for a, b in zip(e, d)) for _, e in rows)]
    lower = [i for i, x in enumerate(options) if x < old]
    upper = [i for i, x in enumerate(options) if x > old]
    return (([max(lower, key=lambda i: options[i])] if lower else [])
            + ([min(upper, key=lambda i: options[i])] if upper else []))


def validate_context(ctx):
    if not isinstance(ctx.get("parent_identity"), str) or not ctx["parent_identity"]:
        raise ValueError("missing parent identity")
    if not isinstance(ctx.get("basis_identity"), str) or not ctx["basis_identity"]:
        raise ValueError("missing evaluation basis identity")
    genes, genome = ctx["genes"], ctx["genome"]
    if len(genes) != len(genome) or len({g["name"] for g in genes}) != len(genes):
        raise ValueError("invalid gene inventory")
    for gene, allele in zip(genes, genome):
        if type(allele) is not int or not 0 <= allele < len(gene["options"]):
            raise ValueError("invalid allele")
        if gene["kind"] not in ("size", "grade") or not gene["groups"]:
            raise ValueError("invalid gene kind or scope")
        if any(g not in ctx["groups"] for g in gene["groups"]):
            raise ValueError("a gene names a group the parent does not have")
    for gid in ctx["groups"]:
        if sum(g["kind"] == "size" and gid in g["groups"] for g in genes) != 1:
            raise ValueError(f"group {gid} needs exactly one size gene")
        if sum(g["kind"] == "grade" and gid in g["groups"] for g in genes) != 1:
            raise ValueError(f"group {gid} needs exactly one concrete-family gene")


def route(ctx, diagnostic):
    """Eligible single-gene alternatives for one diagnostic, or a named abstention; never a pass."""
    validate_context(ctx)
    d = diagnostic
    base = {"version": VERSION, "diagnostic_id": d.get("id"), "operator": None, "proposals": [], "status": "abstain",
            "withheld": []}

    def stop(reason):
        return {**base, "reason": reason}

    if d.get("parent_identity") != ctx["parent_identity"] or d.get("basis_identity") != ctx["basis_identity"]:
        return stop("stale_or_unbound_evidence")
    if d.get("status") == "not_evaluated":
        return stop("evidence_acquisition_required")
    if d.get("reason") == "solver_noncompletion":
        return stop("analysis_diagnosis_required")
    if d.get("status") != "fail":
        return stop("no_valid_failed_trigger")
    if d.get("applicable") is not True or d.get("evidence_valid") is not True:
        return stop("invalid_or_inapplicable_evidence")
    if not d.get("id") or d.get("reason") not in RULES:
        return stop("unsupported_or_ambiguous_diagnosis")
    operator, required = RULES[d["reason"]]
    base["operator"] = operator
    if any((d.get("dependencies") or {}).get(k) is not True for k in required):
        return stop("trigger_dependencies_incomplete")
    a, b, comparison = d.get("demand"), d.get("capacity"), d.get("comparison")
    if not finite(a) or not finite(b) or comparison not in ("<=", ">=", "==") or not d.get("units"):
        return stop("invalid_numeric_check")
    if {"<=": a <= b, ">=": a >= b, "==": a == b}[comparison]:
        return stop("status_contradicts_numeric_check")
    reason, groups, candidates = d["reason"], ctx["groups"], []
    demand_sized = (ctx.get("guided_policy") or {}).get("beam_jump") == jumps.DEMAND_SIZED
    if demand_sized:
        base["jumps"] = []

    def steps(gid, lever, target=None, jump=False):
        """(gene index, gene, its options, the parent's value, the legal targets) of one lever on one group; with
        ``jump``, ``target`` only (a screened demand-sized option), when it is a move of this lever."""
        kind = "grade" if lever == "grade_up" else "size"
        matches = [(i, g) for i, g in enumerate(ctx["genes"]) if g["kind"] == kind and gid in g["groups"]]
        if len(matches) != 1:
            raise ValueError("ambiguous gene mapping")
        index, gene = matches[0]
        options = [tuple(o) if kind == "size" else o for o in gene["options"]]
        old = options[ctx["genome"][index]]
        if jump:
            wanted = None if target is None else (tuple(target) if kind == "size" else target)
            choices = [x for x in options if x == wanted and jumps.in_direction(lever, old, x)]
        elif lever == "grade_up":
            higher = [x for x in options if x > old]
            choices = [min(higher)] if higher else []
        elif lever == "column_up":
            larger = [x for x in options if x[0] == x[1] and x[0] > old[0] and x[1] > old[1]]
            choices = [min(larger)] if larger else []
        elif lever in ("column_lanes", "transition_lanes"):
            # The look-ahead size itself, only when it is a legal larger square option of this gene.
            choices = [x for x in options if target is not None and x == tuple(target) and x[0] == x[1]
                       and x[0] > old[0] and x[1] > old[1]]
        else:
            choices = frontier(options, old, lever)
        return index, gene, options, old, choices

    def withhold(gid, lever, rule, target=None):
        if steps(gid, lever, target)[4]:              # only a step that exists is withheld
            base["withheld"].append({"lever": lever, "group": gid, "reason": rule})

    def add(gid, lever, target=None, transition=None, jump=None):
        index, gene, options, old, choices = steps(gid, lever, target, jump is not None)
        for target in choices:
            raw = list(ctx["genome"])
            raw[index] = options.index(target)
            origin = {"diagnostic_id": d["id"], "reason": reason, "operator": operator, "target_group": gid, "lever": lever}
            if transition is not None:
                origin["transition"] = dict(transition)
            if jump is not None:
                origin["step"] = "demand_sized"
                origin["jump"] = dict(jump)
            candidates.append({"operator": operator, "parent_identity": ctx["parent_identity"],
                               "basis_identity": ctx["basis_identity"], "requested_genome": raw,
                               "origins": [origin],
                               "requested_gene": gene["name"], "from": old, "to": target,
                               "requested_scope_groups": list(gene["groups"]),
                               "synthetic_diagnostic": d.get("synthetic") is True})

    def threading_levers(gid, axis, named, cores, sized, transitions, consistent, missing):
        """The column levers of a threading blockage, shared by every trigger that carries one. ``named``: the cores
        the evidence names; ``cores``: those of joints the beam frames into along ``axis``; ``sized``: those whose own
        cage blocks, which take column_up (and column_lanes at the core look-ahead size); a core blocked only at a
        transition joint takes no core step (withheld as transition_only). ``transitions``: transition_lanes on the
        band a transition's look-ahead resizes, only at a joint of this beam where both groups meet. ``consistent``: the
        topology the levers aim through is the record's; ``missing``: the reason recorded when no core is incident.
        Returns the number of transition targets proposed."""
        inconsistent = ("joint_topology_inconsistent: the cores the beam threads come from a topology the record does "
                        "not support")
        lookahead = d.get("lane_lookahead") if isinstance(d.get("lane_lookahead"), dict) else {}
        if not consistent:
            if not named:
                base["withheld"].append({"lever": "column_up", "group": None, "reason": inconsistent})
            for core in named:
                withhold(core, "column_up", inconsistent)
        elif not cores:
            base["withheld"].append({"lever": "column_up", "group": None, "reason": missing})
        else:
            for core in cores:
                if core not in sized:
                    withhold(core, "column_up", "transition_only: the core cage admits the bars; its growth does not "
                                                "open the transition joint")
                    continue
                add(core, "column_up")
                if isinstance(lookahead.get(core), (list, tuple)) and len(lookahead[core]) == 2:
                    add(core, "column_lanes", tuple(lookahead[core]))
        proposed = 0
        for row in transitions if isinstance(transitions, list) else []:
            if not isinstance(row, dict):
                continue
            target_group, target = row.get("group"), row.get("target")
            pair = {"core_group": row.get("core_group"), "above_group": row.get("above_group"), "band": row.get("band")}
            if target_group not in groups or target_group not in (pair["core_group"], pair["above_group"]):
                continue
            if not (isinstance(target, (list, tuple)) and len(target) == 2):
                continue
            if not consistent:
                withhold(target_group, "transition_lanes", inconsistent, tuple(target))
            elif not any(t["core_group"] == pair["core_group"] and t.get("column_above_group") == pair["above_group"]
                         and gid in t["beams"].get(axis, []) for t in ctx.get("joint_types", {}).values()):
                base["withheld"].append({"lever": "transition_lanes", "group": target_group,
                                         "reason": "no_transition_incidence: the topology names no joint of the beam "
                                                   "where these two column groups meet"})
            else:
                before = len(candidates)
                add(target_group, "transition_lanes", tuple(target), transition=pair)
                proposed += len(candidates) > before
        return proposed

    def sized_cores(cores):
        own = d.get("blocking_core_cages")
        return cores if not isinstance(own, list) else [c for c in cores if c in own]

    def jump(gid, lever):
        """The farther option of ``lever`` the parent's demand-sized screen shortlisted (Design.Genetic_Jumps), as one
        more single-gene proposal, and the diagnostic's own unit's target option where the route already proposes it
        (annotated, never added); or the reason there is none, in route["jumps"]."""
        kind = "grade" if lever == "grade_up" else "size"
        _index, gene, options, old, _choices = steps(gid, lever)
        # The local steps of every lever of this reason on the same gene (a width step can be a depth jump's target).
        local = {x for other in jumps.LEVERS[reason] if ("grade" if other == "grade_up" else "size") == kind
                 for x in steps(gid, other)[4]}
        entry = {"gene": gene["name"], "lever": lever, "group": gid}

        def fallback(why):
            base["jumps"].append({**entry, "status": "fallback", "reason": why})

        screen = d.get("capacity_jump")
        if not isinstance(screen, dict):
            return fallback("no_screen")
        if (screen.get("version") != jumps.VERSION or screen.get("parent_identity") != ctx["parent_identity"]
                or screen.get("basis_identity") != ctx["basis_identity"]):
            return fallback("screen_not_bound_to_this_parent")
        if screen.get("status") != "screened":
            return fallback(f"screen_{screen.get('status')}")
        summary = (screen.get("genes") or {}).get(gene["name"])
        if not isinstance(summary, dict):
            return fallback("gene_not_screened")
        try:
            current = jumps._value(summary.get("from"), kind)
        except (TypeError, ValueError):
            current = None
        if current != old:
            return fallback("screen_of_another_allele")
        chosen = summary.get("meets_target") or summary.get("partial")
        to, unreadable = None, False
        if isinstance(chosen, dict):
            try:
                to = jumps._value(chosen.get("to"), kind)
            except (TypeError, ValueError):
                unreadable = True
        unit_baseline = next((u.get("baseline_dcr") for u in summary.get("unit_baselines") or []
                              if isinstance(u, dict) and gid in (u.get("groups") or [])), None)
        # The unit's own smallest option reaching the target with a threading cage, whatever the gene's shortlist (a
        # family's follows its worst unit). When the route already proposes it (a local step of a jump lever whose
        # direction contains it), its estimate is annotated on that proposal in this lever's pass when this lever's
        # direction contains it; no proposal is added, so the genomes every policy draws over are unchanged.
        unit_target = next((u for u in summary.get("unit_targets") or []
                            if isinstance(u, dict) and gid in (u.get("groups") or [])), None)
        try:
            unit_to = None if unit_target is None else jumps._value(unit_target.get("to"), kind)
        except (TypeError, ValueError):
            unit_to = None
        if unit_to is None or not finite(unit_target.get("estimated_dcr")) or not finite(unit_baseline):
            unit_target = unit_to = None
        proposed = (unit_to is not None and unit_to != to and unit_to in options
                    and any(c["requested_gene"] == gene["name"] and c["to"] == unit_to for c in candidates)
                    and any(jumps.in_direction(other, old, unit_to) for other in jumps.LEVERS[reason]
                            if ("grade" if other == "grade_up" else "size") == kind))
        target_record = None if unit_to is None else {
            "to": list(unit_to) if kind == "size" else unit_to, "estimated_dcr": unit_target["estimated_dcr"],
            "smallest_confirmed": unit_target.get("smallest_confirmed"), "this_option": unit_to == to,
            "annotated": unit_to == to or proposed}
        basis = "frozen parent demand; an estimate that ranks the proposal, not an evaluation"
        if proposed and jumps.in_direction(lever, old, unit_to):
            # The option's own gene-level estimate and pool (its worst unit's); the unit's under ``matched``.
            option_dcr, target_dcr = unit_target.get("option_estimated_dcr"), summary.get("target_dcr")
            add(gid, lever, unit_to, jump={
                "screen_version": jumps.VERSION, "outcome": summary.get("outcome"),
                "matched": {"group": gid, "baseline_dcr": unit_baseline, "estimated_dcr": unit_target["estimated_dcr"],
                            "pool": unit_target.get("pool"), "scope": "unit_target",
                            "unit_target": {**target_record, "this_option": True}},
                "meets_target_estimated": (option_dcr <= target_dcr if finite(option_dcr) and finite(target_dcr)
                                           else None),
                "estimated_dcr": option_dcr if finite(option_dcr) else None, "baseline_dcr": summary.get("baseline_dcr"),
                "target_dcr": target_dcr, "capacity_ratio_required": summary.get("capacity_ratio_required"),
                "diagnostic_utilization": screen.get("utilization"),
                "required_capacity_kip_in": screen.get("required_capacity_kip_in"), "local_step": unit_to in local,
                "pool": unit_target.get("option_pool"), "smallest_confirmed": None, "basis": basis})
            base["jumps"].append({**entry, "status": "unit_target", "to": list(unit_to) if kind == "size" else unit_to,
                                  "local_step": unit_to in local, "estimated_dcr": unit_target["estimated_dcr"]})
        if not isinstance(chosen, dict):
            return fallback(summary.get("outcome") or "no_target")
        if unreadable:
            return fallback("unreadable_target")
        if to not in options or not jumps.in_direction(lever, old, to):
            base["jumps"].append({**entry, "status": "other_lever", "to": list(to) if kind == "size" else to})
            return None
        # This diagnostic's own unit: its baseline, its estimate under the change and that estimate's cage pool
        # (threading, or admissible when no cage threads the frozen column cages); never another band's.
        unit = next((u for u in chosen.get("units") or [] if isinstance(u, dict) and gid in (u.get("groups") or [])), {})
        matched = ({"group": gid, "baseline_dcr": unit_baseline, "estimated_dcr": unit.get("estimated_dcr"),
                    "pool": unit.get("pool"), "scope": "shortlisted_option", "unit_target": target_record}
                   if finite(unit_baseline) and finite(unit.get("estimated_dcr")) else None)
        info = {"screen_version": jumps.VERSION, "outcome": summary.get("outcome"), "matched": matched,
                "meets_target_estimated": isinstance(summary.get("meets_target"), dict),
                "estimated_dcr": chosen.get("estimated_dcr"), "baseline_dcr": summary.get("baseline_dcr"),
                "target_dcr": summary.get("target_dcr"), "capacity_ratio_required": summary.get("capacity_ratio_required"),
                "diagnostic_utilization": screen.get("utilization"),
                "required_capacity_kip_in": screen.get("required_capacity_kip_in"), "local_step": to in local,
                "pool": chosen.get("pool"), "smallest_confirmed": summary.get("smallest_confirmed"), "basis": basis}
        add(gid, lever, to, jump=info)
        base["jumps"].append({**entry, "status": "proposed", "to": list(to) if kind == "size" else to,
                              "local_step": to in local, "outcome": info["outcome"],
                              "estimated_dcr": info["estimated_dcr"], "unit_target": target_record})
        return None

    joint = kind = None
    if operator == "H1":
        joint = ctx.get("joints", {}).get(d.get("joint_id"))
        if not joint or d.get("axis") not in ("x", "y"):
            return stop("missing_joint_or_axis")
        if d.get("sign") not in ("positive", "negative"):
            return stop("missing_sway_sign")
        if joint["required"].get(d["axis"]) is not True:
            return stop("not_a_required_scwb_direction")
        if not joint.get("core_group") or joint["core_group"] not in groups:
            return stop("missing_physical_core")
    elif operator == "H4" or reason == "beam_threading":
        # Joint-type rows: the joint core and incident beams of the type the row was computed on.
        kind = ctx.get("joint_types", {}).get(d.get("joint_type_id"))
        if not kind or d.get("axis") not in ("x", "y"):
            return stop("missing_joint_or_axis")
        if not kind.get("core_group") or kind["core_group"] not in groups:
            return stop("missing_physical_core")
    if operator == "H1":
        targets = joint.get("columns", [])
        if not targets or any(g not in groups or groups[g]["member_type"] != "column" for g in targets):
            return stop("invalid_connected_column_inventory")
        for gid in sorted(set(targets)):
            add(gid, "column_up")
            add(gid, "grade_up")
    elif operator in ("H2", "H3"):
        gid = d.get("group_id")
        if gid not in groups:
            return stop("missing_target_group")
        member_type = groups[gid]["member_type"]
        if reason.startswith("column_"):
            if member_type != "column":
                return stop("wrong_member_type")
            if reason == "column_capacity_shear" and d.get("cage_at_ratio_floor") is True:
                withhold(gid, "column_up", "cage_at_minimum_ratio: a larger section raises the probable-moment "
                                           "shear faster than the section limit")
            else:
                add(gid, "column_up")
            add(gid, "grade_up")
        else:
            if not member_type.startswith("beam_"):
                return stop("wrong_member_type")
            if reason in ("beam_flexure", "beam_capacity_shear"):
                for lever in ("depth", "width", "grade_up"):
                    add(gid, lever)
                if reason == "beam_flexure" and demand_sized:
                    for lever in jumps.LEVERS[reason]:
                        jump(gid, lever)
            elif reason == "beam_lane_limited_strength":
                add(gid, "depth")
                add(gid, "width")
                if demand_sized:
                    for lever in jumps.LEVERS[reason]:
                        jump(gid, lever)
                axis = member_type[len("beam_"):]
                incident = sorted({t["core_group"] for t in ctx.get("joint_types", {}).values()
                                   if gid in t["beams"].get(axis, []) and t["core_group"] in groups})
                # With the selection's threading fields, the cores they name and what blocks there; without, every core
                # the beam threads (more column lanes for its bars), as before.
                if isinstance(d.get("blocking_cores"), list):
                    named = sorted({c for c in d["blocking_cores"] if c in groups})
                    cores = [c for c in named if c in incident]
                else:
                    named = cores = incident
                threading_levers(gid, axis, named, cores, sized_cores(cores), d.get("transition_lookahead"),
                                 d.get("joint_topology_consistent") is True,
                                 "no_joint_incidence: the topology names no joint the beam frames into along its axis")
            elif reason == "beam_threading_constrained_strength":
                # Short of strength even unthreaded: the beam's own levers; and kept from a stronger cage by the cores the
                # record names (the next square size, and the look-ahead size where the record gives one) and by the
                # transitions it names (the band whose size opens that joint).
                for lever in ("depth", "width", "grade_up"):
                    add(gid, lever)
                if demand_sized:
                    for lever in jumps.LEVERS[reason]:
                        jump(gid, lever)
                axis = member_type[len("beam_"):]
                incident = {t["core_group"] for t in ctx.get("joint_types", {}).values() if gid in t["beams"].get(axis, [])}
                named = sorted({c for c in d.get("blocking_cores") or [] if c in groups})
                cores = [c for c in named if c in incident]
                threading_levers(gid, axis, named, cores, sized_cores(cores), d.get("transition_lookahead"),
                                 d.get("joint_topology_consistent") is True,
                                 "no_joint_incidence: the record names no blocking core of a joint the beam frames into "
                                 "along its axis")
            elif reason == "beam_width_room":
                add(gid, "width")
            elif reason == "beam_row_depth":
                add(gid, "depth")
            elif reason == "beam_threading":
                if gid not in kind["beams"].get(d["axis"], []):
                    return stop("beam_not_incident_to_joint_axis")
                # The joint's core grows where its own cage blocks (or where the row does not separate the two); a
                # blocking transition takes the band its joint look-ahead names, or is withheld when none is evidenced.
                core = kind["core_group"]
                proposed = threading_levers(gid, d["axis"], [core], [core],
                                            [] if d.get("core_cage_blocks") is False else [core],
                                            d.get("transition_lookahead"), True, "")
                if d.get("transition_blocks") is True and not proposed:
                    evidence = d.get("lane_evidence")
                    base["withheld"].append({
                        "lever": "transition_lanes", "group": d.get("column_above_group"),
                        "reason": ("no_transition_target: no size of either band opens this joint in the look-ahead"
                                   if evidence == "authenticated" else
                                   f"no_transition_target: the installed-cage lane evidence is {evidence or 'absent'}")})
            elif reason == "slab_mat_overlap":
                if d.get("blocked_dimension") not in ("width", "depth"):
                    return stop("ambiguous_geometric_bottleneck")
                add(gid, d["blocked_dimension"])
    elif operator == "H4":
        add(kind["core_group"], "column_up")
        if reason == "joint_shear":
            add(kind["core_group"], "grade_up")
    elif operator == "H5":
        story, axis = d.get("story"), d.get("axis")
        if type(story) is not int or axis not in ("x", "y"):
            return stop("missing_story_or_axis")
        # Story k spans floors k-1 and k: its columns, and the beams of both floors that bound it (floor 0 has none).
        targets = [gid for gid, g in sorted(groups.items()) if g.get("system") == "sfrs" and (
            (g["member_type"] == "column" and story in g["stories_or_floors"])
            or (g["member_type"] == "beam_" + axis and {story - 1, story} & set(g["stories_or_floors"])))]
        if not targets:
            return stop("no_eligible_sfrs_band")
        for gid in targets:
            add(gid, "column_up" if groups[gid]["member_type"] == "column" else "depth")
    unique = {}
    for p in candidates:
        key = tuple(p["requested_genome"])
        if key in unique:
            unique[key]["origins"].extend(p["origins"])
        else:
            unique[key] = p
    proposals = sorted(unique.values(), key=lambda p: (p["requested_gene"], p["requested_genome"]))
    return {**base, "status": "proposed" if proposals else "abstain",
            "reason": "eligible_alternatives" if proposals else "no_legal_move", "proposals": proposals}


def utilization(d):
    """The normalized utilization of a diagnostic's own check: demand / capacity for a "<=" check (capacity above
    zero, demand not negative), required / available for a ">=" check (the record's capacity / demand, available above
    zero), None for a predicate ("==") or for numbers without a defensible ratio. Never invented."""
    demand, capacity, comparison = d.get("demand"), d.get("capacity"), d.get("comparison")
    if not (finite(demand) and finite(capacity)):
        return None
    if comparison == "<=" and capacity > 0 and demand >= 0:
        return demand / capacity
    if comparison == ">=" and demand > 0 and capacity >= 0:
        return capacity / demand
    return None


def failure_key(ctx, d):
    """The actionable failure a diagnostic reports, as one key: member_strength / capacity_shear / bar-room
    failures by the design tie (size gene) of the member group; beam_threading by the beam's tie and the joint core;
    scwb by joint core and axis; joint shear, through-bar depth and hook embedment by joint core; slab-mat overlap by
    joint type; story drift and stability by the column band of the story (the rows of one failure, both signs, both
    faces and every group of a tie, share their key)."""
    reason = d.get("reason")

    def unit(gid):
        return next((g["name"] for g in ctx.get("genes") or [] if g["kind"] == "size" and gid in g["groups"]), gid)
    if reason in MEMBER_KEYS:
        return f"{MEMBER_KEYS[reason]}|{unit(d.get('group_id'))}"
    kind = (ctx.get("joint_types") or {}).get(d.get("joint_type_id")) or {}
    if reason == "beam_threading":
        return f"beam_threading|{unit(d.get('group_id'))}|{kind.get('core_group')}"
    if reason in ("joint_shear", "through_bar_depth", "hook_embedment"):
        return f"{reason}|{kind.get('core_group') or d.get('joint_type_id')}"
    if reason == "slab_mat_overlap":
        return f"slab_mat_overlap|{d.get('joint_type_id')}"
    if reason == "scwb":
        joint = (ctx.get("joints") or {}).get(d.get("joint_id")) or {}
        return f"scwb|{joint.get('core_group') or d.get('joint_id')}|{d.get('axis')}"
    if reason in ("story_drift", "story_stability"):
        story = d.get("story")
        bands = sorted({g.get("band") for g in (ctx.get("groups") or {}).values() if g.get("member_type") == "column"
                        and story in (g.get("stories_or_floors") or []) and g.get("band")})
        return f"{reason}|{bands[0] if len(bands) == 1 else f'story:{story}'}"
    return f"{reason}|{d.get('id')}"


def unbounded(d):
    """Whether a failed check's ratio is unbounded: a ">=" check with no available clearance (available <= 0 against
    a positive requirement) or a "<=" check with no capacity against a positive demand."""
    demand, capacity, comparison = d.get("demand"), d.get("capacity"), d.get("comparison")
    if not (finite(demand) and finite(capacity)):
        return False
    return (comparison == ">=" and demand <= 0 < capacity) or (comparison == "<=" and capacity <= 0 < demand)


def selection_weight(severity, floor, cap, categorical):
    """The weight of one failure key: floor + (1 - floor) min(severity, cap) / cap for a numeric severity (from the
    floor at zero to 1 at the cap and above), ``categorical`` for a check without a defensible ratio (severity None).
    Bounded in [min(floor, categorical), 1]: a more severe failure weighs more, none weighs nothing."""
    if severity is None:
        return categorical
    return floor + (1.0 - floor) * min(max(severity, 0.0), cap) / cap


def guided_alternatives(ctx, diagnostics):
    """Route every diagnostic of one parent. Returns {"by_operator": {operator: [proposal]}, "abstentions":
    {reason: count}, "withheld": {"lever:rule": count}, "routes": n, "failures": {key: failure}, "jumps": counts};
    a requested genome proposed for several diagnostics or operators appears once per operator, and its origins list
    every attribution (merged across operators by ``merge_origins``). A lever a rule withheld is counted once per
    diagnostic and withheld step (group), whether or not other levers were proposed, as abstentions are counted once
    per diagnostic. ``failures`` groups the routes that proposed by failure_key: {"reason", "operators",
    "diagnostic_ids", "utilization" and "severity" (the largest over the key's diagnostics with a ratio, else None),
    "numeric", "genomes" (the distinct requested genomes, sorted)}. ``jumps`` counts the demand-sized jump outcomes
    of the routes (proposed, of which local steps, other lever, a unit's own target option annotated on a proposal of
    the route, and fallbacks by reason)."""
    by_operator, abstentions, withheld, failures = {}, {}, {}, {}
    jump_counts = {"proposed": 0, "local_step": 0, "other_lever": 0, "unit_target": 0, "fallbacks": {}}
    for d in diagnostics:
        result = route(ctx, d)
        for item in result.get("withheld") or []:
            key = f"{item['lever']}:{item['reason'].split(':', 1)[0]}"
            withheld[key] = withheld.get(key, 0) + 1
        for item in result.get("jumps") or []:
            if item["status"] == "proposed":
                jump_counts["proposed"] += 1
                jump_counts["local_step"] += item.get("local_step") is True
            elif item["status"] == "fallback":
                jump_counts["fallbacks"][item["reason"]] = jump_counts["fallbacks"].get(item["reason"], 0) + 1
            elif item["status"] == "unit_target":
                jump_counts["unit_target"] += 1
            else:
                jump_counts["other_lever"] += 1
        if result["status"] != "proposed":
            abstentions[result["reason"]] = abstentions.get(result["reason"], 0) + 1
            continue
        bucket = by_operator.setdefault(result["operator"], {})
        for p in result["proposals"]:
            key = tuple(p["requested_genome"])
            if key in bucket:
                bucket[key]["origins"].extend(copy.deepcopy(p["origins"]))
            else:
                bucket[key] = copy.deepcopy(p)
        failure = failures.setdefault(failure_key(ctx, d), {"reason": d["reason"], "operators": set(),
                                                            "diagnostic_ids": [], "governing_diagnostic_id": d["id"],
                                                            "utilization": None, "severity": None, "numeric": False,
                                                            "unbounded": False, "genomes": set()})
        failure["operators"].add(result["operator"])
        failure["diagnostic_ids"].append(d["id"])
        # The key's utilization and severity are its largest finite ratio; an unbounded row governs (weight at cap).
        u = utilization(d)
        if u is not None and (failure["utilization"] is None or u > failure["utilization"]):
            failure.update(utilization=u, severity=max(0.0, u - 1.0), numeric=True)
            if not failure["unbounded"]:
                failure["governing_diagnostic_id"] = d["id"]
        if unbounded(d) and not failure["unbounded"]:
            failure.update(numeric=True, unbounded=True, governing_diagnostic_id=d["id"])
        failure["genomes"].update(tuple(p["requested_genome"]) for p in result["proposals"])
    for failure in failures.values():
        failure["operators"] = sorted(failure["operators"])
        failure["genomes"] = [list(g) for g in sorted(failure["genomes"])]
    jump_counts["fallbacks"] = dict(sorted(jump_counts["fallbacks"].items()))
    return {"by_operator": {op: sorted(b.values(), key=lambda p: (p["requested_gene"], p["requested_genome"]))
                            for op, b in sorted(by_operator.items())},
            "abstentions": dict(sorted(abstentions.items())), "withheld": dict(sorted(withheld.items())),
            "routes": len(diagnostics), "failures": dict(sorted(failures.items())), "jumps": jump_counts}


def merge_origins(alternatives, requested_genome):
    """Every origin, across operators, that requested ``requested_genome`` (attribution is never first-come)."""
    key = tuple(requested_genome)
    return [copy.deepcopy(o) for proposals in alternatives["by_operator"].values() for p in proposals
            if tuple(p["requested_genome"]) == key for o in p["origins"]]


def failure_table(alternatives, selection):
    """The weighted failure keys of one parent's alternatives under ``selection`` (severity_floor, severity_cap,
    categorical_weight), in key order, and the draw over distinct proposals. Returns (rows, shares): rows [{"key",
    "reason", "operators", "diagnostic_ids", "governing_diagnostic_id", "utilization", "severity", "numeric",
    "unbounded", "weight", "genomes", "governs_probability" (the probability that the drawn proposal is one this key
    governs), "addresses_probability" (that it is one this key offers)}]; shares {genome: (share, governing row)},
    the share being the largest over the offering keys of weight / number of distinct proposals (the first key in key
    order on a tie). Keys without a proposal are left out."""
    rows = []
    for key, failure in sorted((alternatives.get("failures") or {}).items()):
        if not failure["genomes"]:
            continue
        if failure.get("unbounded"):
            weight = 1.0                                   # no clearance or capacity: the weight at the cap
        else:
            weight = selection_weight(failure["severity"] if failure["numeric"] else None, selection["severity_floor"],
                                      selection["severity_cap"], selection["categorical_weight"])
        rows.append({**{k: failure.get(k) for k in ("reason", "operators", "diagnostic_ids", "governing_diagnostic_id",
                                                    "utilization", "severity", "numeric", "unbounded", "genomes")},
                     "key": key, "weight": weight})
    shares = {}
    for row in rows:
        share = row["weight"] / len(row["genomes"])
        for genome in row["genomes"]:
            if tuple(genome) not in shares or share > shares[tuple(genome)][0]:
                shares[tuple(genome)] = (share, row)
    total = sum(share for share, _row in shares.values())
    for row in rows:
        row["governs_probability"] = sum(s for s, r in shares.values() if r is row) / total
        row["addresses_probability"] = sum(shares[tuple(g)][0] for g in row["genomes"]) / total
    return rows, shares


def _origins_by_genome(alternatives):
    """{genome: (proposal, [origins])}: every distinct proposal once, with its origins across operators."""
    result = {}
    for _op, proposals in sorted(alternatives["by_operator"].items()):
        for p in proposals:
            genome = tuple(p["requested_genome"])
            if genome not in result:
                result[genome] = (p, [])
            result[genome][1].extend(p["origins"])
    return result


def preferred_pool(alternatives, selection, exhausted=()):
    """The preferred pool of severity_preferred_jumps_v1 (module docstring, v10). Returns {"eligible": [entry],
    "excluded": [entry], "shares": {genome: (score, entry)}, "fallback": None or the reason the pool is empty}; an
    eligible entry holds the genome, gene, change, governing key and diagnostic (and its reason), the key's
    utilization, severity and weight, the unit's baseline and estimate, its estimate pool (``pool``; the gene-level
    option's, its worst unit's, is ``gene_pool``), the improvement, the deficit fraction closed, whether the unit is
    estimated to reach the target and whether the candidate is its confirmed smallest target option, the scope of the
    estimate (the shortlisted option, or the unit's own target option), the gene's screen outcome, whether a local
    origin also proposes it, every diagnostic and lever that offers it under its key, its score and its probability in
    the pool. An excluded row is one per genome, key and reason, with every diagnostic and lever that gave it; a genome
    eligible under one key keeps its rows of the others (``eligible_under_keys``; ``already_drawn_under_keys`` when it
    is spent for this parent, its eligible entries then excluded as already_drawn_for_this_parent)."""
    rows, _shares = failure_table(alternatives, selection)
    by_key = {row["key"]: row for row in rows}
    key_of = {did: row["key"] for row in rows for did in row["diagnostic_ids"]}
    target, gain, severe = selection["target_dcr"], selection["min_gain"], selection["severe_utilization"]
    done = {tuple(g) for g in exhausted}
    per_key, excluded, screened = {}, [], 0
    for genome, (p, origins) in sorted(_origins_by_genome(alternatives).items()):
        local = any(o.get("step") != "demand_sized" for o in origins)
        best, reasons, support = {}, {}, {}

        def exclude(base, reason, **numbers):
            # One row per key and reason of this genome, with every diagnostic and lever that gave it.
            row = reasons.get((base["key"], reason))
            if row is None:
                reasons[(base["key"], reason)] = {**base, **numbers, "reason": reason,
                                                  "diagnostic_ids": [base["diagnostic_id"]], "levers": [base["lever"]]}
                return
            row["diagnostic_ids"] = sorted(set(row["diagnostic_ids"]) | {base["diagnostic_id"]}, key=str)
            row["levers"] = sorted(set(row["levers"]) | {base["lever"]}, key=str)

        for origin in origins:
            if origin.get("step") != "demand_sized" or not isinstance(origin.get("jump"), dict):
                continue
            jump, key = origin["jump"], key_of.get(origin.get("diagnostic_id"))
            row, matched = by_key.get(key), origin["jump"].get("matched")
            base = {"genome": list(genome), "gene": p["requested_gene"], "from": p["from"], "to": p["to"], "key": key,
                    "diagnostic_id": origin.get("diagnostic_id"), "diagnostic_reason": origin.get("reason"),
                    "lever": origin.get("lever"), "gene_outcome": jump.get("outcome"),
                    "scope": matched.get("scope") if isinstance(matched, dict) else None}
            if row is None:
                exclude(base, "no_failure_key")
                continue
            if row["utilization"] is None or row["utilization"] < severe:
                exclude(base, "not_severe", key_utilization=row["utilization"])
                continue
            if not isinstance(matched, dict) or not (finite(matched.get("baseline_dcr"))
                                                     and finite(matched.get("estimated_dcr"))):
                exclude(base, "no_unit_estimate")
                continue
            b, e = matched["baseline_dcr"], matched["estimated_dcr"]
            unit_target = matched.get("unit_target") if isinstance(matched.get("unit_target"), dict) else None
            numbers = {"baseline_utilization": b, "estimated_utilization": e, "improvement": b - e,
                       "pool": matched.get("pool"), "gene_pool": jump.get("pool"), "unit_target": unit_target}
            if b <= target:
                exclude(base, "parent_unit_within_target", **numbers)
                continue
            if matched.get("pool") != "threading":
                # The estimate needs a cage that does not thread the frozen column cages: no beam-only reach.
                exclude(base, "threading_blocked_estimate", **numbers)
                continue
            if b - e < gain or b - e <= 0:
                exclude(base, "negligible_improvement", **numbers)
                continue
            if (e <= target and unit_target is not None and unit_target.get("annotated") is True
                    and unit_target.get("this_option") is not True):
                # A smaller option the route also proposes reaches this unit's target: it carries the preference.
                exclude(base, "larger_than_unit_target", **numbers)
                continue
            f = 1.0 if e <= target else (b - e) / (b - target)
            entry = {**base, **numbers, "deficit_closed": f, "meets_target_estimated": e <= target,
                     "unit_smallest_confirmed": (None if e > target else unit_target is not None
                                                 and unit_target.get("this_option") is True
                                                 and unit_target.get("smallest_confirmed") is True),
                     "local_step": local, "beam_only_target_unmet": e > target,
                     "governing_diagnostic_id": row["governing_diagnostic_id"], "key_utilization": row["utilization"],
                     "severity": row["severity"], "weight": row["weight"]}
            ids, levers = support.setdefault(key, (set(), set()))
            ids.add(base["diagnostic_id"])
            levers.add(base["lever"])
            if key not in best or f > best[key]["deficit_closed"]:
                best[key] = entry
        for key, entry in best.items():
            entry.update(diagnostic_ids=sorted(support[key][0], key=str), levers=sorted(support[key][1], key=str))
        # The reasons of the keys this genome is not eligible under (all of them when it is eligible under none).
        others = [row for (key, _reason), row in reasons.items() if key not in best]
        if not best:
            excluded.extend(others)
            continue
        screened += 1
        for row in others:
            row["already_drawn_under_keys" if genome in done else "eligible_under_keys"] = sorted(best, key=str)
        if genome in done:
            excluded.extend({**e, "reason": "already_drawn_for_this_parent"} for e in best.values())
            excluded.extend(others)
            continue
        excluded.extend(others)
        for key, entry in best.items():
            per_key.setdefault(key, {})[genome] = entry
    shares = {}
    for key in sorted(per_key):
        candidates = per_key[key]
        total_f = sum(e["deficit_closed"] for e in candidates.values())
        top = max(e["deficit_closed"] for e in candidates.values())
        for genome, entry in sorted(candidates.items()):
            score = entry["weight"] * top * entry["deficit_closed"] / total_f
            if genome not in shares or score > shares[genome][0]:
                shares[genome] = (score, entry)
    total = sum(score for score, _e in shares.values())
    eligible = []
    for genome in sorted(shares):
        score, entry = shares[genome]
        eligible.append({**entry, "score": score, "probability": score / total})
    fallback = None
    if not eligible:
        fallback = ("preferred_pool_exhausted" if screened else
                    "no_eligible_severe_candidate" if excluded else "no_screened_candidate")
    return {"eligible": eligible, "excluded": excluded, "shares": shares, "total": total, "fallback": fallback}


def _share_draw(shares, rng):
    """One genome of {genome: (share, ...)} with probability share / total, one rng.random() over sorted genomes."""
    genomes = sorted(shares)
    total = sum(shares[g][0] for g in genomes)
    ticket, genome = rng.random() * total, genomes[-1]
    for candidate in genomes:
        ticket -= shares[candidate][0]
        if ticket < 0:
            return candidate, total
    return genome, total


def _proposal_for(alternatives, genome, operators):
    """(operator, proposal) of ``genome``, preferring ``operators`` (the governing key's), then operator order."""
    order = list(operators) + [op for op in sorted(alternatives["by_operator"]) if op not in operators]
    return next((op, p) for op in order for p in alternatives["by_operator"].get(op, [])
                if tuple(p["requested_genome"]) == tuple(genome))


def choose_guided(alternatives, rng, selection=None):
    """One guided proposal, or None when every diagnostic abstained.

    uniform_operator_v1 (the declared initial rule; also when ``selection`` is None): an operator uniformly among those
    with at least one proposal, then one of its proposals uniformly (fixed weights; no adaptation; two rng.choice).
    severity_weighted_v1: one distinct proposal, drawn with probability share / total over the sorted distinct
    genomes (failure_table: a genome's share is the largest over the keys that offer it of the key's weight divided by
    its number of distinct proposals; on a tie the first key in key order governs), with exactly one rng.random() per
    draw; the governing key is recorded, not drawn first. With ``selection`` the proposal carries a ``selection``
    record: the policy, the governing key, its governing diagnostic, utilization, severity, weight and probability, and
    the proposal's own probability.
    severity_preferred_jumps_v1 (module docstring, v10): the preferred branch with probability preferred_share when the
    pool is not empty, else the severity_weighted_v1 draw; ``selection["exhausted"]`` lists the pool candidates
    already drawn for this parent in either branch. Two rng.random() per draw. The record adds the branch, the branch
    probabilities, the fallback reason, the pool (eligible and excluded proposals with the reasons) and the
    proposal's probability in each branch and overall, and the whole distribution (every distinct proposal's final,
    preferred-branch and normal-branch probability, its governing key and the keys that offer it)."""
    policy = UNIFORM if selection is None else selection.get("policy", UNIFORM)
    if policy == UNIFORM:
        operators = [op for op, proposals in alternatives["by_operator"].items() if proposals]
        if not operators:
            return None
        operator = rng.choice(operators)
        proposal = copy.deepcopy(rng.choice(alternatives["by_operator"][operator]))
        proposal["origins"] = merge_origins(alternatives, proposal["requested_genome"])
        proposal["selected_operator"] = operator
        if selection is not None:
            key = tuple(proposal["requested_genome"])
            proposal["selection"] = {
                "policy": UNIFORM, "operator": operator, "operator_probability": 1.0 / len(operators),
                "proposal_probability_given_operator": 1.0 / len(alternatives["by_operator"][operator]),
                "proposal_probability": sum(
                    sum(tuple(p["requested_genome"]) == key for p in alternatives["by_operator"][op])
                    / (len(operators) * len(alternatives["by_operator"][op])) for op in operators)}
        return proposal
    if policy not in (SEVERITY_WEIGHTED, PREFERRED):
        raise ValueError(f"unknown guided selection policy {policy!r}")
    table, shares = failure_table(alternatives, selection)
    if not table:
        return None
    if policy == PREFERRED:
        return _preferred_choice(alternatives, rng, selection, table, shares)
    genome, total = _share_draw(shares, rng)
    share, governing = shares[genome]
    operator, found = _proposal_for(alternatives, genome, governing["operators"])
    proposal = copy.deepcopy(found)
    proposal["origins"] = merge_origins(alternatives, list(genome))
    proposal["selected_operator"] = operator
    proposal["selection"] = {
        "policy": SEVERITY_WEIGHTED, "version": SELECTION_VERSION,
        "parameters": {k: selection[k] for k in ("severity_floor", "severity_cap", "categorical_weight")},
        "key": governing["key"], "reason": governing["reason"],
        "governing_diagnostic_id": governing["governing_diagnostic_id"],
        "diagnostic_ids": list(governing["diagnostic_ids"]), "utilization": governing["utilization"],
        "severity": governing["severity"], "numeric": governing["numeric"], "unbounded": governing["unbounded"],
        "weight": governing["weight"], "share": share, "key_proposals": len(governing["genomes"]),
        "key_probability": governing["governs_probability"], "proposal_probability": share / total,
        "offering_keys": [row["key"] for row in table if list(genome) in row["genomes"]],
        "table": [{k: row[k] for k in ("key", "reason", "governing_diagnostic_id", "utilization", "severity", "numeric",
                                       "unbounded", "weight", "governs_probability", "addresses_probability")}
                  | {"proposals": len(row["genomes"])} for row in table]}
    return proposal


def guided_distribution(alternatives, selection=None):
    """{genome: {"probability", "preferred", "normal"}}: the exact probability that choose_guided draws each distinct
    proposal under ``selection`` (the overall probability, and its probability in the preferred and the normal branch;
    under uniform_operator_v1 both branch values are None). Empty when every diagnostic abstained."""
    policy = UNIFORM if selection is None else selection.get("policy", UNIFORM)
    genomes = _origins_by_genome(alternatives)
    if not genomes:
        return {}
    if policy == UNIFORM:
        operators = [op for op, proposals in alternatives["by_operator"].items() if proposals]
        return {g: {"probability": sum(sum(tuple(p["requested_genome"]) == g for p in alternatives["by_operator"][op])
                                       / (len(operators) * len(alternatives["by_operator"][op])) for op in operators),
                    "preferred": None, "normal": None} for g in sorted(genomes)}
    _table, shares = failure_table(alternatives, selection)
    total = sum(s for s, _r in shares.values())
    normal = {g: shares[g][0] / total for g in shares}
    if policy != PREFERRED:
        return {g: {"probability": normal[g], "preferred": None, "normal": normal[g]} for g in sorted(normal)}
    pool = preferred_pool(alternatives, selection, selection.get("exhausted") or ())
    alpha = selection["preferred_share"] if pool["eligible"] else 0.0
    preferred = {g: (pool["shares"][g][0] / pool["total"] if g in pool["shares"] else 0.0) for g in normal}
    return {g: {"probability": alpha * preferred[g] + (1.0 - alpha) * normal[g], "preferred": preferred[g],
                "normal": normal[g]} for g in sorted(normal)}


def _preferred_choice(alternatives, rng, selection, table, shares):
    """The severity_preferred_jumps_v1 draw (module docstring, v10)."""
    pool = preferred_pool(alternatives, selection, selection.get("exhausted") or ())
    alpha = selection["preferred_share"] if pool["eligible"] else 0.0
    branch_ticket = rng.random()
    preferred = bool(pool["eligible"]) and branch_ticket < alpha
    if preferred:
        genome, _total = _share_draw(pool["shares"], rng)
        entry = pool["shares"][genome][1]
        governing = next(row for row in table if row["key"] == entry["key"])
    else:
        genome, _total = _share_draw(shares, rng)
        entry, governing = None, shares[genome][1]
    normal_total = sum(share for share, _row in shares.values())
    p_normal = shares[genome][0] / normal_total
    p_preferred = pool["shares"][genome][0] / pool["total"] if genome in pool["shares"] else 0.0
    proposals = _origins_by_genome(alternatives)
    distribution = []
    for g in sorted(shares):
        g_normal = shares[g][0] / normal_total
        g_preferred = pool["shares"][g][0] / pool["total"] if g in pool["shares"] else 0.0
        p = proposals[g][0]
        distribution.append({"genome": list(g), "gene": p["requested_gene"], "from": p["from"], "to": p["to"],
                             "probability": alpha * g_preferred + (1.0 - alpha) * g_normal, "preferred": g_preferred,
                             "normal": g_normal, "governing_key": shares[g][1]["key"],
                             "offering_keys": [row["key"] for row in table if list(g) in row["genomes"]],
                             "preferred_candidate": g in pool["shares"]})
    operator, found = _proposal_for(alternatives, genome, governing["operators"])
    proposal = copy.deepcopy(found)
    proposal["origins"] = merge_origins(alternatives, list(genome))
    proposal["selected_operator"] = operator
    proposal["selection"] = {
        "policy": PREFERRED, "version": SELECTION_VERSION, "preference_version": PREFERENCE_VERSION,
        "parameters": {k: selection[k] for k in ("severity_floor", "severity_cap", "categorical_weight",
                                                 "severe_utilization", "preferred_share", "target_dcr", "min_gain")},
        "branch": "preferred" if preferred else "normal",
        "branch_probabilities": {"preferred": alpha, "normal": 1.0 - alpha},
        "fallback": pool["fallback"], "pool_existed": bool(pool["eligible"]),
        "preferred_candidate": genome in pool["shares"],
        "proposal_probability": alpha * p_preferred + (1.0 - alpha) * p_normal,
        "proposal_probability_preferred": p_preferred, "proposal_probability_normal": p_normal,
        "key": governing["key"], "reason": governing["reason"],
        "governing_diagnostic_id": governing["governing_diagnostic_id"], "utilization": governing["utilization"],
        "severity": governing["severity"], "weight": governing["weight"], "numeric": governing["numeric"],
        "unbounded": governing["unbounded"], "diagnostic_ids": list(governing["diagnostic_ids"]),
        "normal_share": shares[genome][0], "key_probability": governing["governs_probability"],
        "key_proposals": len(governing["genomes"]),
        "offering_keys": [row["key"] for row in table if list(genome) in row["genomes"]],
        "route_jump_fallbacks": dict((alternatives.get("jumps") or {}).get("fallbacks") or {}),
        "distribution": distribution,
        "estimate": None if entry is None else {k: entry[k] for k in (
            "baseline_utilization", "estimated_utilization", "improvement", "deficit_closed", "meets_target_estimated",
            "unit_smallest_confirmed", "unit_target", "beam_only_target_unmet", "scope", "gene_outcome", "pool",
            "gene_pool", "local_step", "diagnostic_reason")},
        "pool": {"eligible": [{k: v for k, v in e.items()} for e in pool["eligible"]], "excluded": pool["excluded"]},
        "table": [{k: row[k] for k in ("key", "reason", "governing_diagnostic_id", "utilization", "severity", "numeric",
                                       "unbounded", "weight", "governs_probability", "addresses_probability")}
                  | {"proposals": len(row["genomes"])} for row in table]}
    if entry is not None and entry["beam_only_target_unmet"]:
        note = "target unmet on the frozen parent: the beam-only change does not reach the target"
        # The column sentence only where this failure's own routes propose column clearance or coordination.
        dids = set(governing["diagnostic_ids"])
        if any(o.get("lever") in COLUMN_LEVERS and o.get("diagnostic_id") in dids
               for _p, origins in proposals.values() for o in origins):
            note += "; this failure's column clearance and coordination proposals stay in the normal branch"
        proposal["selection"]["note"] = note
    return proposal


def random_proposal(ctx, rng):
    """R1: a uniformly chosen gene with a nonempty local support, then a uniformly chosen neighbour."""
    validate_context(ctx)
    supports = []
    for i, g in enumerate(ctx["genes"]):
        options = [tuple(o) if g["kind"] == "size" else o for o in g["options"]]
        choices = local_neighbors(options, options[ctx["genome"][i]])
        if choices:
            supports.append((i, choices, options))
    if not supports:
        return None
    index, choices, options = rng.choice(supports)
    target = rng.choice(choices)
    gene = ctx["genes"][index]
    raw = list(ctx["genome"])
    raw[index] = target
    return {"operator": "R1", "selected_operator": "R1", "parent_identity": ctx["parent_identity"],
            "basis_identity": ctx["basis_identity"], "requested_genome": raw, "requested_gene": gene["name"],
            "from": options[ctx["genome"][index]], "to": options[target],
            "requested_scope_groups": list(gene["groups"]),
            "origins": [{"reason": "random_local", "operator": "R1", "lever": "random"}], "synthetic_diagnostic": False}


def _direction_valid(lever, old, new):
    if lever == "random":
        return True
    if lever == "grade_up":
        return new > old
    if lever in ("column_up", "column_lanes", "transition_lanes"):
        return new[0] == new[1] and new[0] > old[0] and new[1] > old[1]
    if lever == "depth":
        return new[1] > old[1] and new[0] >= old[0]
    if lever == "width":
        return new[0] > old[0] and new[1] >= old[1]
    return False


def finalize(ctx, proposal, repaired_genome, repair_log):
    """Validate the repaired genome against the parent and the requested intervention; expose its whole scope.

    Returns {"status": "ready_for_evaluation" | "noop" | "rejected", "reason", ...}. A ready child lists the requested
    change, every effective gene change after repair (the induced ones apart), the affected groups and member tags,
    the repair log and a genome identity; ``engineering_accepted`` is always False.
    """
    validate_context(ctx)
    if proposal["parent_identity"] != ctx["parent_identity"] or proposal["basis_identity"] != ctx["basis_identity"]:
        return {"status": "rejected", "reason": "stale_proposal", "proposal": proposal}
    try:
        validate_context({**ctx, "genome": list(repaired_genome)})
    except (ValueError, KeyError, TypeError):
        return {"status": "rejected", "reason": "invalid_repair_output", "proposal": proposal}
    effective = [i for i, (a, b) in enumerate(zip(ctx["genome"], repaired_genome)) if a != b]
    if not effective:
        return {"status": "noop", "reason": "repair_returned_parent", "proposal": proposal,
                "repair_log": copy.deepcopy(repair_log)}
    requested = next(i for i, g in enumerate(ctx["genes"]) if g["name"] == proposal["requested_gene"])
    if requested not in effective:
        return {"status": "rejected", "reason": "repair_removed_requested_intervention", "proposal": proposal,
                "repair_log": copy.deepcopy(repair_log)}
    gene = ctx["genes"][requested]
    options = [tuple(o) if gene["kind"] == "size" else o for o in gene["options"]]
    old, new = options[ctx["genome"][requested]], options[repaired_genome[requested]]
    for origin in proposal["origins"]:
        if not _direction_valid(origin["lever"], old, new):
            return {"status": "rejected", "reason": "repair_changed_intervention_direction", "proposal": proposal,
                    "repair_log": copy.deepcopy(repair_log)}

    def change(i):
        g = ctx["genes"][i]
        values = [tuple(o) if g["kind"] == "size" else o for o in g["options"]]
        return {"gene": g["name"], "kind": g["kind"], "from": values[ctx["genome"][i]], "to": values[repaired_genome[i]],
                "groups": list(g["groups"])}

    groups = sorted({gid for i in effective for gid in ctx["genes"][i]["groups"]})
    tags = sorted({tag for gid in groups for tag in ctx["groups"][gid]["member_tags"]})
    requested_change = change(requested)
    requested_change["requested_to"] = proposal["to"]
    return {"status": "ready_for_evaluation", "reason": "repaired_child_differs_from_parent", "engineering_accepted": False,
            "proposal": proposal, "repaired_genome": list(repaired_genome), "repair_log": copy.deepcopy(repair_log),
            "requested_change": requested_change,
            "induced_changes": [change(i) for i in effective if i != requested],
            "effective_genes": [ctx["genes"][i]["name"] for i in effective],
            "affected_groups": groups, "affected_member_tags": tags,
            "genome_identity": digest({"basis": ctx["basis_identity"], "genes": list(repaired_genome)}),
            "limits": "section/material and profile grammar only; cage and slab changes come from the evaluation"}


def deduplicate_ready(rows, key="genome_identity"):
    """(unique ready rows, duplicates) on ``key``; a duplicate's origins join the kept row's attribution."""
    unique, duplicates = {}, []
    for row in rows:
        if row["status"] != "ready_for_evaluation":
            continue
        identity = row[key]
        if identity in unique:
            kept = unique[identity]
            kept["proposal"]["origins"].extend(copy.deepcopy(row["proposal"]["origins"]))
            duplicates.append({"identity": identity, "kept_operator": kept["proposal"]["operator"],
                               "duplicate_operator": row["proposal"]["operator"], "origins": row["proposal"]["origins"]})
        else:
            unique[identity] = row
    return list(unique.values()), duplicates
