"""Experimental grouped genetic search with failure-guided mutation.

One fixed grouping/layout and longitudinal grade per run. Genes are declared
section rungs (one per tie) and concrete grades (one per family). Reinforcement
is reselected by the existing evaluator from a reproducible seed cage.

Every evaluation passes the code-check contract before it is ranked or archived
(Design.Code_Check_Contract.assess_evaluation, 2026-10-30): the record's schema,
explicit completion flags, the seed identity captured before evaluation, the
final state and its identity, and the code-check rows, recomputed against an
inventory derived from the final design itself. Only an "accepted" candidate
(the complete scoped code check passed, the evaluation completed, the
objectives available) enters the archive; "check_failed", "unresolved" and
"invalid" candidates keep their known failures as fields and as secondary
fitness keys, never as a pass. Review dependencies are listed, not asserted.
Nothing here is a design qualification: the saved record's qualification
(Design.Grouped_Record) remains the gate before any analysis.

No production runner imports this module. Evaluation remains serial because
OpenSees and Structure_Parameters have process-global state.
"""
from __future__ import annotations

from Design.Config import validate_beam_practical_ceiling

import math
import random
from dataclasses import asdict, dataclass

import Structure_Parameters as sp
from Design import Code_Check_Contract as contract
from Design import Concrete_Grades as cg, Grouped_Design as gd, Grouped_Search as gs
from Model import Member_Groups as mg

VERSION = "experimental_grouped_ga_heuristic_v2_code_check_contract"
CLASSIFICATIONS = ("feasible", "check_failed", "unresolved", "invalid")
CONTRACT_CLASSES = {"accepted": "feasible", "check_failed": "check_failed", "unresolved": "unresolved", "invalid": "invalid"}


@dataclass(frozen=True)
class GeneticPolicy:
    population_size: int = 8
    max_evaluations: int = 24
    max_generations: int = 10
    random_seed: int = 20261030
    heuristic_probability: float = 0.75
    crossover_probability: float = 0.8
    random_jump_probability: float = 0.2
    max_proposals_per_evaluation: int = 30

    def validate(self):
        for name in ("population_size", "max_evaluations", "max_generations", "max_proposals_per_evaluation"):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == "population_size" else 1):
                raise ValueError(f"{name} is outside its positive integer domain")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")
        for name in ("heuristic_probability", "crossover_probability", "random_jump_probability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class Gene:
    name: str
    kind: str
    groups: tuple
    options: tuple


class DesignSpace:
    """Finite domains; decoding starts from the same cages on every evaluation."""

    def __init__(self, seed):
        self.seed = seed
        genes, assigned = [], set()
        ties = seed.ties()
        units = [(name, tuple(sorted(gids))) for name, gids in sorted(ties.items())]
        assigned.update(g for _, gids in units for g in gids)
        units += [(g, (g,)) for g in sorted(seed.groups) if g not in assigned]
        for name, gids in units:
            design = seed.designs[gids[0]]
            if design.is_column:
                options = [(x, x) for x in gs._column_sizes(seed)]
            else:
                common = set(gs.beam_rungs(design.member_type))
                for gid in gids[1:]:
                    common &= set(gs.beam_rungs(seed.designs[gid].member_type))
                options = sorted(common, key=lambda x: (x[0] * x[1] ** 2, x))
            if not options:
                raise ValueError(f"No shared section ladder for {name}")
            genes.append(Gene(name, "size", gids, tuple(options)))
        for family, spec in sorted(cg.families(seed.policy_id).items()):
            gids = tuple(sorted(cg.groups_of(seed, family)))
            if gids:
                genes.append(Gene(gs.grade_key(family), "grade", gids, tuple(spec["ladder"])))
        self.genes = tuple(genes)
        self.initial = self.encode(seed)  # Refuse off-ladder seeds; never silently widen the domain.
        self.validate_state(seed)

    def encode(self, state):
        result = []
        for gene in self.genes:
            values = {(state.designs[g].b_in, state.designs[g].h_in) if gene.kind == "size"
                      else state.designs[g].fc_ksi for g in gene.groups}
            if len(values) != 1:
                raise ValueError(f"{gene.name} violates its tie/family")
            value = next(iter(values))
            if value not in gene.options:
                raise ValueError(f"{gene.name}: {value} is outside the declared ladder {gene.options}")
            result.append(gene.options.index(value))
        return tuple(result)

    def validate_state(self, state):
        self.encode(state)
        cg.require(state)
        mg.require_section_order(state)
        for gid, design in state.designs.items():
            problem = gs.system_coupling_problem(state, gid, (design.b_in, design.h_in))
            if problem:
                raise mg.GroupedStateError(problem)

    def decode(self, genome):
        if len(genome) != len(self.genes):
            raise ValueError("Genome length does not match the declared design space")
        moves = {}
        for gene, index in zip(self.genes, genome):
            if type(index) is not int or not 0 <= index < len(gene.options):
                raise ValueError(f"Invalid allele for {gene.name}: {index}")
            moves[gene.name] = (gene.options[index], "genetic allele")
        return gs.apply_moves(self.seed, moves)

    def repair(self, genome):
        state = self.decode(genome)
        # Existing repair enlarges lower bands and enforces system coupling;
        # every resulting dimension must still be a declared allele.
        state, changes = gs.enforce_banded_section_order(state)
        self.validate_state(state)
        return self.encode(state), state, changes


def heuristic_moves(evaluation, state, space):
    """Alternative move sets, not an instruction to accept any changed design."""
    moves = gs.propose_growth(evaluation, state)
    choices = [{key: value} for key, value in sorted(moves.items())]
    if len(moves) > 1:
        choices.append(moves)
    constraints = evaluation.get("constraints") or {}
    # The old growth routine only expands columns for this failure. Offer all
    # wider/deeper beam rungs too, including coordinated lower-band repair.
    for gid in sorted(set(constraints.get("lane_limited_beam_groups") or [])):
        if gid not in state.groups or state.designs[gid].is_column:
            continue
        gene = next(g for g in space.genes if g.kind == "size" and gid in g.groups)
        current = state.designs[gid]
        for size in gene.options:
            if size[0] >= current.b_in and size[1] >= current.h_in and size != (current.b_in, current.h_in):
                beam_move = {gene.name: (size, "lane congestion: test a wider/deeper beam and reselect all cages")}
                choices.append(beam_move)
                if moves:
                    choices.append({**moves, **beam_move})
    if evaluation.get("feasible"):
        for gene in space.genes:
            if gene.kind != "size":
                continue
            current = state.designs[gene.groups[0]]
            smaller = [r for r in gene.options if r[0] * r[1] < current.b_in * current.h_in]
            for size in sorted(smaller, key=lambda r: (-r[0] * r[1] ** 2, r))[:3]:
                choices.append({gene.name: (size, "feasible parent: test material reduction")})
    return choices


def mutate(genome, parent, space, rng, policy):
    """Guidance comes from the evaluated parent, then acts on the crossed child.

    Its relevance to the crossed child is only a hypothesis; full evaluation
    determines acceptance. Random mutation retains access to unanticipated moves.
    """
    if parent is not None and rng.random() < policy.heuristic_probability:
        evaluation, parent_state = parent["feedback"], parent["state"]
        choices = heuristic_moves(evaluation, parent_state, space)
        rng.shuffle(choices)
        for moves in choices:
            try:
                changed = gs.apply_moves(space.decode(genome), moves)
                encoded = space.encode(changed)
            except (ValueError, mg.GroupedStateError):
                continue
            if encoded != genome:
                return encoded, {"kind": "heuristic", "source_candidate": parent["index"], "moves": gs._move_rows(moves)}
    available = [i for i, gene in enumerate(space.genes) if len(gene.options) > 1]
    if not available:
        return genome, {"kind": "no_mutable_gene"}
    index = rng.choice(available)
    gene = space.genes[index]
    options = [x for x in range(len(gene.options)) if x != genome[index]]
    if rng.random() >= policy.random_jump_probability:
        distance = min(abs(x - genome[index]) for x in options)
        options = [x for x in options if abs(x - genome[index]) == distance]
    child = list(genome)
    child[index] = rng.choice(options)
    return tuple(child), {"kind": "random", "gene": gene.name, "from": genome[index], "to": child[index]}


def contract_assessment(evaluation, cfg, submitted_identity=None, submitted_state=None):
    """The authoritative assessment of one evaluation record (Design.Code_Check_Contract.assess_evaluation): given the
    submitted design, its final state must be that design with only the reinforcement reselected."""
    return contract.assess_evaluation(evaluation, submitted_identity=submitted_identity, cfg=cfg,
                                      submitted_state=submitted_state)


def unresolved_reasons(evaluation, cfg=None, submitted_identity=None, assessed=None):
    """Why an evaluation is not an accepted code design: record problems, numerical incompletion, and the code check's
    failures, unknown, missing and invalid rows, as the contract assessment states them."""
    assessed = assessed or contract_assessment(evaluation, cfg, submitted_identity)
    reasons = [f"invalid record: {p}" for p in assessed["problems"]]
    reasons += list(assessed["numerical_completion"]["problems"])
    check = assessed.get("code_check") or {}
    if check and check.get("status") != "passed":
        counts = check.get("counts") or {}
        reasons.append(f"code check {check.get('status')}: {counts.get('fail', 0)} failed, {counts.get('unknown', 0)} unknown, "
                       f"{counts.get('missing', 0)} missing, {counts.get('invalid', 0)} invalid, "
                       f"{counts.get('duplicates', 0)} duplicated, {counts.get('unexpected', 0)} unexpected row(s)")
        if check.get("unknown_reason_codes"):
            reasons.append("unknown evidence: " + ", ".join(check["unknown_reason_codes"][:8]))
        if any(c.get("id") == "scwb" for c in check.get("unknown", [])) or any(
                m.get("id") == "scwb" for m in check.get("missing", []) if isinstance(m, dict)):
            reasons.append("strong-column/weak-beam evidence is incomplete")
    for key, item in (assessed.get("objectives") or {}).items():
        if not item["available"] and check.get("status") == "passed":
            reasons.append(f"{key} is unavailable or invalid")
    return reasons


def assessment(evaluation, cfg, submitted_identity=None, assessed=None, submitted_state=None):
    """(classification, fitness) from the contract assessment; nothing the evaluator says about itself is trusted.

    Fitness is (tier, open evidence, max exceedance, sum of exceedances), smaller first:
      feasible (accepted)                                   (0, 0, 0, 0)
      check_failed (complete evidence, known failures)      (1, 0, max, sum)
      unresolved, evaluated (unknown or missing evidence)   (2, unknown + missing rows, max, sum)
      unresolved, stopped (no code check was reached)       (3, 0, 0, 0)
      invalid (malformed, unbound or self-contradictory)    (4, 0, 0, 0)
    Exceedances are the failed rows' dimensionless ones (Code_Check_Contract.violation). Within the unresolved tier
    the open evidence comes first, so turning a failing row into an unknown never improves a candidate (it adds one
    open row), and exceedances are compared between candidates with the same open evidence. A stopped evaluation
    never outranks one that was evaluated. No penalty buys a pass.
    """
    assessed = assessed or contract_assessment(evaluation, cfg, submitted_identity, submitted_state)
    classification = CONTRACT_CLASSES[assessed["classification"]]
    check = assessed.get("code_check") or {}
    violation = check.get("violation") or {}
    measures = (float(violation.get("max", 0.0)), float(violation.get("sum", 0.0)))
    if classification == "feasible":
        return classification, (0, 0, 0.0, 0.0)
    if classification == "invalid":
        return classification, (4, 0, 0.0, 0.0)
    if classification == "check_failed":
        return classification, (1, 0, *measures)
    if not check:                                       # stopped before any code check (an infeasible stage)
        return classification, (3, 0, 0.0, 0.0)
    counts = check.get("counts") or {}
    return classification, (2, int(counts.get("unknown", 0)) + int(counts.get("missing", 0)), *measures)


def objectives(candidate):
    q = candidate["evaluation"].get("quantities") or {}
    values = tuple(q.get(k) for k in ("concrete_total_in3", "total_steel_in3"))
    return tuple(v if type(v) in (int, float) and math.isfinite(v) and v >= 0 else math.inf for v in values)


def dominates(a, b):
    if a["fitness"] != b["fitness"]:
        return a["fitness"] < b["fitness"]
    if a["classification"] != "feasible":
        return False
    x, y = objectives(a), objectives(b)
    return all(i <= j for i, j in zip(x, y)) and any(i < j for i, j in zip(x, y))


def survive(candidates, count, rng):
    """Elitist nondominated sorting; crowding preserves material tradeoffs."""
    remaining, chosen = list(candidates), []
    while remaining and len(chosen) < count:
        front = [a for a in remaining if not any(dominates(b, a) for b in remaining if b is not a)]
        if len(chosen) + len(front) <= count:
            chosen.extend(front)
        else:
            rng.shuffle(front)
            distances = {a["index"]: 0.0 for a in front}
            if front[0]["classification"] == "feasible":
                for axis in range(2):
                    ordered = sorted(front, key=lambda a: objectives(a)[axis])
                    low, high = objectives(ordered[0])[axis], objectives(ordered[-1])[axis]
                    if not math.isfinite(high) or high <= low:
                        continue
                    distances[ordered[0]["index"]] = distances[ordered[-1]["index"]] = math.inf
                    for k in range(1, len(ordered) - 1):
                        distances[ordered[k]["index"]] += (objectives(ordered[k + 1])[axis] - objectives(ordered[k - 1])[axis]) / (high - low)
            front.sort(key=lambda a: distances[a["index"]], reverse=True)
            chosen.extend(front[:count - len(chosen)])
        ids = {a["index"] for a in front}
        remaining = [a for a in remaining if a["index"] not in ids]
    return chosen


def search(seed, cfg, policy=None, log_path=None, evaluator=None, verbose=True):
    """Return experimental public records; never write an accepted design artifact.

    A caller-provided evaluator supports controlled tests. Only the default
    evaluator performs the complete current grouped pipeline. Unknown exceptions
    abort and are logged instead of becoming structural failures.
    """
    validate_beam_practical_ceiling(cfg)
    policy = policy or GeneticPolicy()
    policy.validate()
    space, rng = DesignSpace(seed), random.Random(policy.random_seed)
    evaluate = evaluator or gd.evaluate_candidate
    log = gs.CandidateLog(log_path)
    context, seen, all_candidates, population = {}, set(), [], []
    counts = dict(evaluations=0, proposals=0, duplicates=0, invalid=0, heuristic=0, random=0, crossovers=0,
                  invalid_records=0)
    # The evaluator mutates these globals; every trial starts from the same
    # environment and restores it. Its own content-keyed floor caches are reused.
    output_keys = ("SLAB_THICKNESS_IN", "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF", "SEISMIC_LIVE_LOAD_FRACTION",
                   "FLOOR_TRANSFER", "SLAB_ACTIONS", "SLAB_REINFORCEMENT")
    environment = {key: getattr(sp, key) for key in output_keys}
    stop = "generation_budget_exhausted"
    try:
        for generation in range(policy.max_generations + 1):
            offspring = []
            attempts = 0
            while len(offspring) < policy.population_size and counts["evaluations"] < policy.max_evaluations:
                if attempts >= policy.population_size * policy.max_proposals_per_evaluation:
                    break
                attempts += 1
                counts["proposals"] += 1
                if not all_candidates:
                    genome, proposal = space.initial, {"kind": "seed"}
                else:
                    pool = population or all_candidates
                    a, b = (rng.choice(pool) for _ in range(2))
                    parent = a if dominates(a, b) else b if dominates(b, a) else rng.choice((a, b))
                    genome = parent["genome"]
                    parents = [parent["index"]]
                    if len(pool) > 1 and rng.random() < policy.crossover_probability:
                        mate = rng.choice([p for p in pool if p is not parent])
                        genome = tuple(x if rng.random() < .5 else y for x, y in zip(genome, mate["genome"]))
                        parents.append(mate["index"])
                        counts["crossovers"] += 1
                    genome, proposal = mutate(genome, parent, space, rng, policy)
                    counts[proposal["kind"]] = counts.get(proposal["kind"], 0) + 1
                    proposal["parents"] = parents
                try:
                    genome, trial, repairs = space.repair(genome)
                except (ValueError, mg.GroupedStateError) as exc:
                    counts["invalid"] += 1
                    log.add({"generation": generation, "proposal": proposal, "decision": "invalid_genome", "reason": str(exc)})
                    continue
                if genome in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(genome)
                counts["evaluations"] += 1
                index = counts["evaluations"]
                submitted = trial.identity()              # captured before evaluation; the record must name it
                try:
                    with mg.installed(trial):
                        raw = evaluate(trial, cfg, context, label=f"genetic_{index}")
                except Exception as exc:
                    log.add({"generation": generation, "proposal": proposal, "genome": genome,
                             "decision": "evaluation_error", "reason": f"{type(exc).__name__}: {exc}"})
                    raise
                finally:
                    for key, value in environment.items():
                        setattr(sp, key, value)
                if raw.get("status") not in ("evaluated", "infeasible"):
                    raise ValueError(f"Unexpected evaluator status: {raw.get('status')}")
                public = gd.public_evaluation(raw)
                assessed = contract_assessment(raw, cfg, submitted, submitted_state=trial)
                classification, fitness = assessment(raw, cfg, assessed=assessed)
                counts["invalid_records"] += classification == "invalid"
                # The evaluated final design when the record is valid and carries one; otherwise the submitted trial is
                # kept only to guide mutation and is labelled so (never presented as an evaluated design).
                final_state = raw.get("state") if assessed["valid"] and raw.get("state") is not None else None
                state = final_state if final_state is not None else trial
                # Keep only the topology needed for heuristic mutation, not all
                # frame actions and shell results for every population member.
                kinds = ((raw.get("evidence") or {}).get("capacity") or {}).get("joint_types") or {}
                topology = {key: {field: val for field, val in kind.items() if field in
                            ("core_group", "column_above_group", "joints", "threading")} for key, kind in kinds.items()}
                candidate = {"index": index, "genome": genome, "state": state, "evaluation": public,
                             "state_source": "evaluated_final_state" if final_state is not None else "submitted_trial",
                             "feedback": {**public, "feasible": classification == "feasible",
                                          "evidence": {"capacity": {"joint_types": topology}}},
                             "classification": classification, "fitness": fitness,
                             "contract": _contract_summary(assessed)}
                all_candidates.append(candidate)
                offspring.append(candidate)
                log.add({"generation": generation, "candidate_index": index, "genome": genome,
                         "proposal": proposal, "repairs": repairs, "classification": classification,
                         "unresolved_reasons": unresolved_reasons(raw, assessed=assessed),
                         "contract": candidate["contract"], "submitted_identity": submitted,
                         "state_source": candidate["state_source"],
                         "fitness": fitness, "evaluation": public, "evaluated_state": state.to_record()})
                if verbose:
                    print(f"Genetic candidate {index}/{policy.max_evaluations}: {classification}; {proposal['kind']}", flush=True)
                del raw
            population = survive(population + offspring, policy.population_size, rng)
            if counts["evaluations"] >= policy.max_evaluations:
                stop = "evaluation_budget_exhausted"
                break
            if not offspring:
                stop = "no_new_candidate_within_proposal_budget"
                break
    finally:
        for key, value in environment.items():
            setattr(sp, key, value)
    feasible = [a for a in all_candidates if a["classification"] == "feasible"]
    archive = [a for a in feasible if not any(dominates(b, a) for b in feasible if b is not a)]
    # Retain the existing representative-selection rule: never silently trade
    # additional steel for less concrete (or the reverse) in the chosen result.
    representative = feasible[0] if feasible else None
    for candidate in feasible[1:]:
        if gs.compare_quantities(candidate["evaluation"]["quantities"], representative["evaluation"]["quantities"])[0] == "accepted":
            representative = candidate
    def saved(candidate):
        return {"candidate_index": candidate["index"], "genome": candidate["genome"],
                "evaluation": candidate["evaluation"], "contract": candidate["contract"],
                "state": candidate["state"].to_record()}
    return {"version": VERSION, "experimental": True, "production_accepted": False,
            "policy": asdict(policy), "genes": [asdict(g) for g in space.genes], "stop": stop, "counts": counts,
            "outcomes": {key: sum(a["classification"] == key for a in all_candidates) for key in CLASSIFICATIONS},
            "representative": saved(representative) if representative else None,
            "pareto_candidates": [saved(a) for a in archive],
            "contract_version": contract.CONTRACT_VERSION,
            "scope": "Fixed layout and longitudinal grade; current cage selector and engineering checks. A feasible candidate "
                     "passed the scoped code check of Design.Code_Check_Contract (computed checks complete and passing, "
                     "review dependencies listed, not asserted); that is not a design qualification. No guarantee of "
                     "global optimality. Numerical incompletion and missing evidence are unresolved, never a pass."}


def _contract_summary(assessed):
    """What a candidate keeps of its contract assessment: classification, problems, the code-check counts and the
    known failures with their local witnesses, the unknown reasons and the review dependencies."""
    check = assessed.get("code_check") or {}
    return {"classification": assessed["classification"], "valid": assessed["valid"], "problems": assessed["problems"],
            "numerical_completion": assessed["numerical_completion"], "preliminary_screen": assessed["preliminary_screen"],
            "code_check_status": check.get("status"), "counts": check.get("counts"),
            "known_failures": (check.get("known_failures") or [])[:40],
            "unknown_reason_codes": check.get("unknown_reason_codes"),
            "missing": (check.get("missing") or [])[:40], "invalid": (check.get("invalid") or [])[:20],
            "review_dependencies": check.get("review_dependencies"), "violation": check.get("violation"),
            "objectives": assessed.get("objectives"), "nonlinear_performance": assessed.get("nonlinear_performance")}
