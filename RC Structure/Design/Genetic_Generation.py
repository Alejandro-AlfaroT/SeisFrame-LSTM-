"""Bounded serial generation controller of the genetic search (SeisFrame GA v3, 2026-10-30).

One case, one grouping, one longitudinal steel grade and one evaluation basis per run. Every distinct child is
evaluated completely by the existing grouped evaluator (Grouped_Design.evaluate_candidate) and assessed by the
code-check contract (Code_Check_Contract.assess_evaluation) before it is ranked or archived. Evaluation is serial.

  initialization   the seed, then the declared initialization policy (INITIALIZATION_POLICIES).
  generations      parents drawn uniformly from the working population; per child, with the declared probability a
                   guided operator (H1-H5, Genetic_Operators.choose_guided) on the parent's own diagnostics, otherwise
                   (or when every diagnostic abstains) R1. No crossover, no adaptive weights. Repair, no-op and
                   duplicate filtering come before any evaluation; an input already evaluated is a cache hit.
  selection        Genetic_Population.select over the population and the generation's offspring (mu + lambda), every
                   child included (a stopped or raised one is listed in the queues, never a parent), with the
                   quarantine computed over the whole history. A stop between steps first selects what was evaluated
                   since the last selection. Progress is a new realized design (one no earlier observation of the run
                   realized), evaluated in the generation, that enters the population and either dominates a member
                   of the previous population (a better tier, more coverage under the same inventory, or better
                   progress measures under the same comparison signature; each member compared on its own record, one
                   quarantined in this selection not at all) or enters while that population, less its members
                   quarantined in this selection, had a free slot; or a code-accepted identity added to the registry.
                   Any other entry is not progress: by the diversity tie-break (a noncomparable design; the seeded
                   anchor is redrawn each generation) or by rank (a nondominated design that dominates no member).
                   Progress is measured against the previous population, not against every earlier parent. A
                   selection that ends underfilled keeps every eligible candidate, and the population never shrinks
                   except by quarantine, so free-slot entries reset the stagnation counter at most population_size
                   times plus once per quarantined member. A generation without a new evaluation
                   stops the search; a generation drawn while no parent existed (exploration) fills free slots, so
                   its entrants reset the counter, bounded by the exploration budget.
  outputs          an append-only observation history, a proposal log, generation records, per-evaluation artifacts,
                   and views recomputed from the whole history: accepted registry, diversity view, evidence queue,
                   analysis-recovery queue, unroutable failures, quarantine.

Two identities are kept apart: ``input_identity`` (the evaluation basis plus the complete submitted design, seed
cages included: the cache key) and ``realized_identity`` (the final design with its selected cages and slab: the
registry key). Every step is checkpointed with the random-generator state; a run resumes only under the same
configuration fingerprint, and one process at a time owns the run directory: an OS lease under which every run file
is written, the request a new run claims the directory with and the error record included. A controller runs once;
it is spent when its run returns or raises, and a new one continues the run. Solve counts are work
accounting: the evaluator's floor caches live in the process, so a resumed run may count more solves than an
uninterrupted one. A budget, time or bound stop is never a statement of structural infeasibility, and a code-accepted
design is not a qualified design: qualification (Grouped_Record) and nonlinear analysis stay separate.

v3_6 (2026-10-31): two declared proposal policies, part of the evaluation basis (proposal_policy) and so of every
binding to it (diagnostics, the input cache, the registry, the fingerprint): the guided selection
(GenerationConfig.guided_selection: uniform_operator_v1, the earlier rule, or severity_weighted_v1,
Genetic_Operators.choose_guided) and the beam-flexure steps (beam_jump_policy: local_frontier_v1, the earlier rule, or
demand_sized_v1). Under demand_sized_v1 each valid evaluation is screened while the evaluator's record is at hand, for
demand-sized farther options of its beam flexural shortfalls (Design.Genetic_Jumps), within a per-gene and
per-observation budget and the run's remaining wall time; the screen's time is the run's. The screen is kept with the
observation's diagnostics, ranks proposals and never certifies one. The guided probability, parent selection,
survival and acceptance are unchanged. A run's configured practical beam steel ratio ceiling is never above
Design.Config.BEAM_PRACTICAL_RATIO_AUTHORIZED_MAX.

v3_7 (2026-11-01): the selection policy severity_preferred_jumps_v1 (Genetic_Operators v10) gives preferred_share of the
guided draws to the parent's severe screened demand-sized proposals (failures at preferred_severe_utilization and
above) when there are any; it needs the demand_sized_v1 jumps, a positive beam_jump_min_gain and a preferred_share
below 1. A pool candidate drawn for a parent, in either branch, is kept in the checkpoint (preferred_drawn) and not
drawn from the preferred branch again. The counters (preferred_selection) keep apart what was proposed (eligible pool
candidates when a parent is first routed), drawn (by branch), completely evaluated, evaluated incompletely or with an
error, and accepted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
import traceback
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import Structure_Parameters as sp
from Design import Code_Check_Contract as contract
from Design import Genetic_Diagnostics as diagnostics
from Design import Genetic_Jumps as jumps
from Design import Genetic_Operators as operators
from Design import Genetic_Population as population
from Design import Genetic_Search as ga
from Design import Grouped_Search as gs
from Design.Config import BEAM_PRACTICAL_RATIO_AUTHORIZED_MAX, validate_beam_practical_ceiling
from Design.Verification_Integrity import exclusive_lease
from Model import Member_Groups as mg

VERSION = "seisframe_ga_generation_v3_7"
INITIALIZATION_POLICIES = ("seed_only_v1", "seed_r1_neighbors_v1", "uniform_then_repair_v1", "constructive_profiles_v1")
ENVIRONMENT_KEYS = ("SLAB_THICKNESS_IN", "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF", "SEISMIC_LIVE_LOAD_FRACTION",
                    "FLOOR_TRANSFER", "SLAB_ACTIONS", "SLAB_REINFORCEMENT")
CAGE_FIELDS = ("bar_size", "top_bars", "bot_bars", "side_bars", "stirrup_bar_size", "stirrup_legs", "stirrup_spacing_in",
               "legs_by_direction")
LEASE = ".generation.lease"
ERROR_ASIDE = re.compile(r"error\.before_resume\.(\d+)\.json")
REQUEST = "request.json"
# The longest artifact name under the run directory: topology/<64 hex>.json.tmp, with its separators.
LONGEST_ARTIFACT = 1 + len("topology") + 1 + 64 + len(".json.tmp")
BOUND_NOTE = ("A budget, time, error, generation or stagnation bound ended the search. It proves neither structural "
              "infeasibility nor optimality, and an empty registry is not evidence that no code-accepted design exists.")
STOP_NOTES = {
    "accepted_target_reached": "The declared target was reached. Code acceptance is not qualification or nonlinear "
                               "validation, and the search established neither optimality nor coverage of the space.",
    "waiting_on_dependencies": "No eligible parent remains and the exploration budget is spent: the designs wait on "
                               "the evidence and analysis-recovery queues (and the quarantine, for contradictory "
                               "evidence). Unresolved evidence is not a failure, and this proves neither structural "
                               "infeasibility nor optimality.",
}
STOP_NOTES["profile_target_reached"] = STOP_NOTES["accepted_target_reached"]


@dataclass(frozen=True)
class GenerationConfig:
    random_seed: int = 20261030
    population_size: int = 4
    offspring_per_generation: int = 4
    max_generations: int = 6
    max_evaluations: int = 24
    max_proposals: int = 400
    max_wall_seconds: float | None = None
    initialization_policy: str = "seed_r1_neighbors_v1"
    initialization_evaluations: int = 3
    guided_probability: float = 0.75
    max_exploration_evaluations: int = 4
    max_stagnant_generations: int = 3
    max_attempts_per_child: int = 30
    target_accepted_designs: int | None = None
    target_distinct_profiles: int | None = None
    diversity_view_size: int = 8
    max_evaluation_errors: int = 2
    crossover: bool = False
    adaptive_operator_weights: bool = False
    # Guided selection (Genetic_Operators.choose_guided): uniform_operator_v1 (the earlier rule) or severity_weighted_v1,
    # whose key weight is severity_floor + (1 - severity_floor) min(severity, severity_cap) / severity_cap, or
    # categorical_weight for a check without a defensible ratio.
    guided_selection: str = operators.UNIFORM
    severity_floor: float = 0.2
    severity_cap: float = 1.0
    categorical_weight: float = 0.5
    # Beam-flexure steps (Design.Genetic_Jumps): local_frontier_v1 (the earlier rule) or demand_sized_v1, screened to the
    # target DCR within max_sections pricings per gene and max_seconds per observation; a partial improvement is the
    # smallest option within partial_tolerance of the best estimate that gains at least min_gain.
    beam_jump_policy: str = jumps.LOCAL
    beam_jump_target_dcr: float = 0.85
    beam_jump_max_sections: int = 8
    beam_jump_max_seconds: float = 15.0
    beam_jump_partial_tolerance: float = 0.02
    beam_jump_min_gain: float = 0.02
    # severity_preferred_jumps_v1 (Genetic_Operators v10; experimental search parameters, not code thresholds): the share
    # of guided draws given to the severe preferred pool when it is not empty, and the failure utilization from which a
    # failure's screened demand-sized proposals enter it.
    preferred_severe_utilization: float = 1.25
    preferred_share: float = 0.80

    def validate(self):
        for name in ("population_size", "offspring_per_generation", "max_generations", "max_evaluations",
                     "max_proposals", "max_attempts_per_child", "diversity_view_size", "max_evaluation_errors"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("initialization_evaluations", "max_exploration_evaluations", "max_stagnant_generations"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("target_accepted_designs", "target_distinct_profiles"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")
        p = self.guided_probability
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("guided_probability must be between zero and one")
        if self.max_wall_seconds is not None and (isinstance(self.max_wall_seconds, bool) or not isinstance(
                self.max_wall_seconds, (int, float)) or not self.max_wall_seconds > 0):
            raise ValueError("max_wall_seconds must be positive or None")
        if self.initialization_policy not in INITIALIZATION_POLICIES:
            raise ValueError(f"initialization_policy must be one of {INITIALIZATION_POLICIES}")
        if self.crossover or self.adaptive_operator_weights:
            raise ValueError("crossover and adaptive operator weights are not part of this version")
        if self.guided_selection not in operators.SELECTION_POLICIES:
            raise ValueError(f"guided_selection must be one of {operators.SELECTION_POLICIES}")
        if self.beam_jump_policy not in jumps.POLICIES:
            raise ValueError(f"beam_jump_policy must be one of {jumps.POLICIES}")
        # (name, lower bound, whether it may equal the bound, upper bound inclusive)
        for name, low, closed, high in (("severity_floor", 0.0, False, 1.0), ("severity_cap", 0.0, False, None),
                                        ("categorical_weight", 0.0, False, 1.0), ("beam_jump_target_dcr", 0.0, False, 1.0),
                                        ("beam_jump_max_seconds", 0.0, False, None),
                                        ("beam_jump_partial_tolerance", 0.0, True, None),
                                        ("beam_jump_min_gain", 0.0, True, None)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not (value >= low if closed else value > low) or (high is not None and value > high)):
                raise ValueError(f"{name} must be a finite number {'at least' if closed else 'above'} {low:g}"
                                 + (f" and at most {high:g}" if high is not None else ""))
        if self.categorical_weight < self.severity_floor:
            raise ValueError("categorical_weight must not be below severity_floor (every eligible key keeps the floor)")
        if type(self.beam_jump_max_sections) is not int or self.beam_jump_max_sections < 2:
            raise ValueError("beam_jump_max_sections must be an integer of at least 2 (the parent and one option)")
        for name, low, high in (("preferred_severe_utilization", 1.0, None), ("preferred_share", 0.0, 1.0)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < low or (high is not None and value > high)):
                raise ValueError(f"{name} must be a finite number of at least {low:g}"
                                 + (f" and at most {high:g}" if high is not None else ""))
        if self.guided_selection == operators.PREFERRED:
            if self.beam_jump_policy != jumps.DEMAND_SIZED:
                raise ValueError(f"guided_selection {operators.PREFERRED} needs beam_jump_policy {jumps.DEMAND_SIZED}: "
                                 "its preferred pool is the screened demand-sized proposals")
            if not self.beam_jump_min_gain > 0:
                raise ValueError("beam_jump_min_gain must be above zero under the preference (a candidate must reduce "
                                 "its unit's deficit)")
            if not self.preferred_share < 1:
                raise ValueError("preferred_share must be below 1: the normal branch keeps every eligible route")


def jump_draw_kind(origins):
    """How a drawn guided proposal relates to the demand-sized screen: "jump" (only a shortlisted jump proposes it),
    "annotated_local_step" (a local step the gene's screen also shortlisted), "unit_target_step" (a local step annotated
    only as a unit's own target option) or None."""
    def scope(origin):
        return ((origin.get("jump") or {}).get("matched") or {}).get("scope")
    sized = [o for o in origins if o.get("step") == "demand_sized"]
    if sized and len(sized) == len(origins) and any(scope(o) != "unit_target" for o in sized):
        return "jump"
    if any(scope(o) != "unit_target" for o in sized):
        return "annotated_local_step"
    return "unit_target_step" if sized else None


def proposal_policy(config, cfg=None):
    """The run's proposal policy, part of its evaluation basis: the guided selection rule and the beam-flexure step
    rule with the parameters each uses, and the beam face steel ratio ceiling the jump screen prices with
    (Genetic_Jumps.ratio_ceiling; None without a design configuration)."""
    selection = {"policy": config.guided_selection}
    if config.guided_selection in (operators.SEVERITY_WEIGHTED, operators.PREFERRED):
        selection.update(version=operators.SELECTION_VERSION, severity_floor=config.severity_floor,
                         severity_cap=config.severity_cap, categorical_weight=config.categorical_weight)
    if config.guided_selection == operators.PREFERRED:
        selection.update(preference_version=operators.PREFERENCE_VERSION,
                         severe_utilization=config.preferred_severe_utilization, preferred_share=config.preferred_share,
                         target_dcr=config.beam_jump_target_dcr, min_gain=config.beam_jump_min_gain)
    jump = {"policy": config.beam_jump_policy}
    if config.beam_jump_policy == jumps.DEMAND_SIZED:
        jump.update(version=jumps.VERSION, target_dcr=config.beam_jump_target_dcr,
                    max_sections=config.beam_jump_max_sections, max_seconds=config.beam_jump_max_seconds,
                    partial_tolerance=config.beam_jump_partial_tolerance, min_gain=config.beam_jump_min_gain,
                    steel_scan_fraction=jumps.STEEL_SCAN_FRACTION, baseline_tolerance=jumps.BASELINE_TOLERANCE)
    return {"guided_selection": selection, "beam_jump": jump,
            "beam_steel_ratio": None if cfg is None else jumps.ratio_ceiling(cfg),
            "authorized_beam_practical_max": BEAM_PRACTICAL_RATIO_AUTHORIZED_MAX}


# ---- constructive initializer (research initialization_distribution_survey_v1) -------------------------------------
class ProfileDP:
    """Band profiles (gravity, SFRS) from the bottom up, with exact suffix counts: every complete profile of the
    declared grammar (both dimensions nonincreasing upward, gravity no larger than SFRS, the column transition step)
    is drawn with equal probability."""

    def __init__(self, domains, column=False, step=4.0):
        self.domains, self.column, self.step = domains, column, step
        self.states = [[(i, j) for i, a in enumerate(g) for j, b in enumerate(s) if _ge(b, a)] for g, s in domains]
        self.edges = [{p: [q for q in self.states[k + 1] if self._transition(k, p, q)] for p in self.states[k]}
                      for k in range(len(domains) - 1)]
        self.counts = [{} for _ in domains]
        self.counts[-1] = {p: 1 for p in self.states[-1]}
        for k in range(len(domains) - 2, -1, -1):
            self.counts[k] = {p: sum(self.counts[k + 1][q] for q in self.edges[k][p]) for p in self.states[k]}
        self.total = sum(self.counts[0].values())
        if self.total <= 0:
            raise ValueError("no complete profile in the declared grammar")

    def _transition(self, k, p, q):
        for role in (0, 1):
            lower, upper = self.domains[k][role][p[role]], self.domains[k + 1][role][q[role]]
            if not _ge(lower, upper) or (self.column and max(a - b for a, b in zip(lower, upper)) > self.step + 1e-9):
                return False
        return True

    def sample(self, rng):
        path = []
        for k in range(len(self.domains)):
            eligible = self.states[k] if k == 0 else self.edges[k - 1][path[-1]]
            ticket = rng.randrange(sum(self.counts[k][p] for p in eligible))
            for p in eligible:
                ticket -= self.counts[k][p]
                if ticket < 0:
                    path.append(p)
                    break
        return path

    def probability(self, path):
        result = Fraction(1, 1)
        for k, p in enumerate(path):
            options = self.states[k] if k == 0 else self.edges[k - 1][path[k - 1]]
            result *= Fraction(self.counts[k][p], sum(self.counts[k][q] for q in options))
        return result


def _ge(a, b):
    return all(x >= y - 1e-9 for x, y in zip(a, b))


def constructive_blocks(space, seed):
    """The (beam, column) profile samplers of a three-story system policy, or a refusal: the grammar was audited for
    system_three_story_v1 only, and its genes must be one gravity and one SFRS tie per band and member kind."""
    if seed.policy_id != "system_three_story_v1":
        raise ValueError(f"constructive_profiles_v1 is audited for system_three_story_v1 only, not {seed.policy_id!r}")
    lookup = {g.name: i for i, g in enumerate(space.genes)}
    bands = sorted({g.name.split("__")[0] for g in space.genes if g.kind == "size"})
    blocks = {}
    for kind in ("beam", "column"):
        names = [(f"{b}__gravity_{kind}", f"{b}__sfrs_{kind}") for b in bands]
        if any(n not in lookup for pair in names for n in pair):
            raise ValueError("constructive_profiles_v1 needs one gravity and one SFRS size tie per band and member kind")
        indices = [(lookup[g], lookup[s]) for g, s in names]
        domains = [(space.genes[g].options, space.genes[s].options) for g, s in indices]
        blocks[kind] = (ProfileDP(domains, kind == "column", gs.MAX_COLUMN_SIZE_STEP_IN), indices)
    return blocks


# ---- small helpers -------------------------------------------------------------------------------------------------
def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_default)


def _default(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cages(state):
    return {gid: {"b_in": d.b_in, "h_in": d.h_in, "fc_ksi": d.fc_ksi,
                  **{f: (list(getattr(d, f)) if isinstance(getattr(d, f), tuple) else getattr(d, f)) for f in CAGE_FIELDS}}
            for gid, d in sorted(state.designs.items())}


def _cage_changes(before, after):
    """{group: {field: [before, after]}} of the reinforcement fields that differ (sections are reported elsewhere)."""
    changes = {}
    for gid in sorted(set(before) & set(after)):
        diff = {f: [before[gid][f], after[gid][f]] for f in CAGE_FIELDS if before[gid].get(f) != after[gid].get(f)}
        if diff:
            changes[gid] = diff
    return changes


def _windows_path_limit():
    """The longest usable path on Windows without long-path support (MAX_PATH less its terminator); None elsewhere."""
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            if winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1:
                return None
    except OSError:
        pass
    return 259


def _missing_counts(missing, patterns=()):
    """{family|id: required rows absent} over the complete missing list of a code-check result. A pattern miss (one
    per group) is keyed by its pattern's family and checks, from the expected inventory's ``patterns``, so the misses
    of one check across groups form one source."""
    named = {p["description"]: f"{p['family']}|{'+'.join(p['id_in']) if p.get('id_in') else 'rows_of_group'}"
             for p in patterns or () if isinstance(p, dict) and p.get("description")}
    counts = {}
    for item in missing or []:
        if item.get("pattern") is not None:
            source = named.get(item["pattern"]) or f"pattern|{item['pattern']}"
        else:
            source = f"{item.get('family')}|{item.get('id')}"
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _recordable(value, limit=2000):
    """A form of something the evaluator or an exception gave that a record can always hold: text as it is (cut to
    ``limit``), None as None, anything else by its repr."""
    if value is None:
        return None
    try:
        return (value if isinstance(value, str) else repr(value))[:limit]
    except Exception:                                     # noqa: BLE001 -- an object whose repr raises
        return f"<unrepresentable {type(value).__name__}>"


def exception_text(exc):
    """An exception's message (one whose __str__ raises is named by its type)."""
    try:
        return str(exc)
    except Exception:                                     # noqa: BLE001
        return f"<{type(exc).__name__} without a printable message>"


class RunDirectoryError(ValueError):
    """A directory a new run cannot claim, or whose saved request is not this run's. ``kind`` names the case so a
    caller can word its own advice: "has_request" or "foreign" (a new run), "no_request", "request_unreadable" or
    "request_differs" (a resume)."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def new_run_refusal(out):
    """Why ``out`` cannot take a new run, or None when it can: missing, empty, or holding only the lease marker and a
    cut request (request.json.tmp). "has_request" when it holds a saved request, otherwise "foreign"."""
    out = Path(out)
    if not out.exists():
        return None
    if any(p.name not in (LEASE, REQUEST + ".tmp") for p in out.iterdir()):
        return "has_request" if (out / REQUEST).exists() else "foreign"
    return None


def request_refusal(out, sha256):
    """Why ``out`` cannot continue the run of the request ``sha256``, or None when its saved request.json (a JSON
    object carrying its own sha256) is that request."""
    path = Path(out) / REQUEST
    if not path.exists():
        return "no_request"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "request_unreadable"
    if not isinstance(saved, dict):
        return "request_unreadable"
    return None if saved.get("sha256") == sha256 else "request_differs"


class GenerationRun:
    """One bounded serial run. ``basis`` is the result-affecting evaluation basis (evaluator identity, contract and
    evaluation versions, case, profile, grouping, seed design, steel grade); ``evaluator`` defaults to
    Grouped_Design.evaluate_candidate."""

    def __init__(self, seed_state, cfg, config, out_dir, *, case_id, basis, evaluator=None, verbose=True,
                 clock=time.perf_counter):
        config.validate()
        self._beam_practical_ceiling = validate_beam_practical_ceiling(cfg)
        self.seed_state, self.cfg, self.config, self.out = seed_state, cfg, config, Path(out_dir)
        # The proposal policy is part of the evaluation basis; a basis that names another one is refused.
        policy = proposal_policy(config, cfg)
        if "proposal_policy" in basis and basis["proposal_policy"] != policy:
            raise ValueError("the basis names another proposal policy (guided selection, beam jumps or beam steel ratio "
                             "ceiling) than this configuration's")
        basis = {**basis, "proposal_policy": policy}
        self.case_id, self.basis = case_id, basis
        self.basis_id = diagnostics.digest(basis)
        self.selection = {"policy": config.guided_selection, "severity_floor": config.severity_floor,
                          "severity_cap": config.severity_cap, "categorical_weight": config.categorical_weight,
                          "severe_utilization": config.preferred_severe_utilization,
                          "preferred_share": config.preferred_share, "target_dcr": config.beam_jump_target_dcr,
                          "min_gain": config.beam_jump_min_gain}
        self.jump_params = {"target_dcr": config.beam_jump_target_dcr, "max_sections": config.beam_jump_max_sections,
                            "max_seconds": config.beam_jump_max_seconds,
                            "partial_tolerance": config.beam_jump_partial_tolerance,
                            "min_gain": config.beam_jump_min_gain}
        if evaluator is None:
            from Design import Grouped_Design as gd
            evaluator = gd.evaluate_candidate
        self.evaluator, self.verbose, self.clock = evaluator, verbose, clock
        self.space = ga.DesignSpace(seed_state)
        self.genes = diagnostics.gene_rows(self.space)
        self.groups = diagnostics.group_rows(seed_state)
        weights = population.features(self.genes, self.space.initial)[1]
        self.run_ctx = {"case_id": case_id, "basis_id": self.basis_id, "feature_weights": weights,
                        "gene_kinds": [g["kind"] for g in self.genes]}
        self.blocks = (constructive_blocks(self.space, seed_state)
                       if config.initialization_policy == "constructive_profiles_v1" else None)
        self.steel_grade = sp.FY_KSI
        self.fingerprint = diagnostics.digest({
            "version": VERSION, "config": asdict(config), "case_id": case_id, "basis_id": self.basis_id,
            "genes": self.genes, "initial": list(self.space.initial), "seed_state": seed_state.identity(),
            "versions": [diagnostics.VERSION, operators.VERSION, population.SELECTION_POLICY,
                         population.PROGRESS_FAMILY_RULE, population.DESCRIPTOR, contract.CONTRACT_VERSION,
                         jumps.VERSION, operators.SELECTION_VERSION, operators.PREFERENCE_VERSION]})
        self.context = {}                 # the evaluator's content-keyed floor caches, reused within a process
        self.save_records = getattr(evaluator, "__name__", "") == "evaluate_candidate"
        self.history, self.by_id, self.by_input, self.origin_events = [], {}, {}, {}
        self.topologies, self.guided_cache = {}, {}
        self.state = None
        self._lease, self._clock_origin, self._spent = None, None, False
        self._check_path()                # refused before any caller creates the directory

    # ---- persistence -------------------------------------------------------------------------------------------
    def _validate_ceiling(self):
        ceiling = validate_beam_practical_ceiling(self.cfg)
        if ceiling != self._beam_practical_ceiling:
            raise ValueError("the practical beam steel ratio ceiling changed after the run basis was frozen; "
                             "create a new controller and evaluation basis")

    def _path(self, name):
        return self.out / name

    def _owned(self):
        """Every run file is written under the lease: a controller whose ``run`` returned or raised is spent (refused
        even if releasing its lease failed)."""
        if self._lease is None or self._spent:
            raise ValueError(f"this controller does not own {self.out}; call start() or resume() first, on a new "
                             "controller once this one has run (a controller is spent when its run returns or raises)")

    def _fresh(self):
        """A controller starts or resumes a run once, before it holds one in memory: a spent controller, one that
        already owns the directory (started or resumed, not yet run) or one that loaded a history is refused before
        anything is read or written, and keeps the lease it holds."""
        self._validate_ceiling()
        if self._spent or self._lease is not None or self.history or self.origin_events:
            raise ValueError(f"this controller already holds or ran the run in {self.out}; build a new GenerationRun "
                             "to resume it")

    def _append(self, name, record):
        self._owned()
        line = _json(record) + "\n"
        with open(self._path(name), "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return line

    def _durable_text(self, path, text):
        self._owned()
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())

    def _replace_file(self, path, text):
        """``text`` written whole: durable in a temporary file, then an atomic replacement, so an interruption leaves
        the old or the new file, never a truncated one. A write that fails removes its temporary file (ownership is
        checked first: a controller that does not own the directory never touches another owner's file)."""
        self._owned()
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            self._durable_text(tmp, text)
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def _write(self, name, value):
        self._owned()
        text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=_default) + "\n"
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._replace_file(path, text)

    def _replace_text(self, name, text):
        """Replace a log, the history or the saved request whole."""
        self._replace_file(self._path(name), text)

    # ---- ownership -------------------------------------------------------------------------------------------------
    def _check_path(self):
        limit = _windows_path_limit()
        if limit is not None and len(str(self.out.resolve())) + LONGEST_ARTIFACT > limit:
            raise ValueError(f"{self.out} is too long for this Windows installation (long paths are disabled): its "
                             f"artifacts would exceed {limit} characters; choose a shorter --output-dir")

    def _acquire(self):
        """Own the run directory until ``run`` returns or ``close`` is called: an OS lock on LEASE, released by the
        kernel if the process dies. A second process (or controller) is refused before it reads or writes a run file.
        The lease marker itself holds no run data (taking the lease creates it when it is missing)."""
        if self._lease is not None:
            return
        self._check_path()
        self.out.mkdir(parents=True, exist_ok=True)
        stack = ExitStack()
        try:
            stack.enter_context(exclusive_lease(self._path(LEASE)))
        except RuntimeError as exc:
            stack.close()
            raise ValueError(f"{self.out} is owned by another running generation process; once it has stopped, "
                             "that run continues with the command that started it (plus --resume), or choose another "
                             "output directory") from exc
        self._lease = stack

    def close(self):
        """Release the run directory (``run`` does this when it returns or raises). The controller lets go of the
        lease first: a release that fails never leaves it looking like an owner."""
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.close()

    def _abandon(self):
        """A start or resume that raised keeps no partial run in memory (summary() refuses it; a history it loaded
        still refuses another start or resume), then releases the directory."""
        self.state = None
        self.close()

    def _checkpoint(self):
        self.state["rng"] = [self.rng.getstate()[0], list(self.rng.getstate()[1]), self.rng.getstate()[2]]
        self._write("checkpoint.json", self.state)

    def _history_add(self, record, persist=True):
        if record["type"] == "observation" and record["id"] in self.by_id:
            raise ValueError(f"observation id {record['id']} is already in the history")      # checked before it is written
        if persist:
            line = self._append("observations.jsonl", record)
            self.state["history_count"] += 1
            self.state["history_digest"] = _sha(self.state["history_digest"] + _sha(line))
        if record["type"] == "observation":
            self.history.append(record)
            self.by_id[record["id"]] = record
            self.by_input.setdefault(record["input_identity"], record["id"])
        else:
            self.origin_events.setdefault(record["observation_id"], []).extend(record["origins"])

    def _log(self, name, record):
        self._append(name, {"step": self.state["step"], **record})

    # ---- start / resume ----------------------------------------------------------------------------------------
    def start(self, request_text=None):
        """Start a new run in ``out`` under its lease. With ``request_text`` (the caller's request, a JSON object
        carrying its own sha256), the directory is checked under the lease to be new (new_run_refusal) and the
        request is saved there, durably and atomically, before the first checkpoint: two processes cannot both
        claim one directory, and no request is written by a process that does not own it. Without it, a directory
        that holds a saved request (another run's claim) is refused too."""
        self._fresh()
        self._acquire()
        try:
            kind = new_run_refusal(self.out)
            if request_text is not None:
                if kind is not None:
                    raise RunDirectoryError(kind, f"{self.out} is not a new run directory ({kind})")
                self._replace_text(REQUEST, request_text)
            elif kind == "has_request":
                raise RunDirectoryError(kind, f"{self.out} holds a saved request; resume that run instead")
            return self._start()
        except BaseException:
            self._abandon()
            raise

    def _check_new(self):
        """Refuse a directory that already holds a run, before anything is written."""
        if self._path("checkpoint.json").exists():
            raise ValueError(f"{self.out} already holds a run; resume it instead")
        for name in ("observations.jsonl", "proposals.jsonl", "generations.jsonl"):
            if self._path(name).exists():
                raise ValueError(f"{self._path(name)} exists without a checkpoint; refusing to mix runs")

    def _start(self, resumes=()):
        self._check_new()
        self.rng = random.Random(self.config.random_seed)
        self.state = {"version": VERSION, "fingerprint": self.fingerprint, "basis_id": self.basis_id,
                      "phase": "initialization", "generation": 0, "population": [], "pool": [], "step": 0,
                      "attempts_generation": 0, "new_in_generation": 0, "init_attempts": 0, "exploration_used": 0,
                      "stagnant": 0, "accepted_identities_at_generation_start": [], "routed_parents": [],
                      "pending": None, "preferred_drawn": {},
                      "history_count": 0, "history_digest": "", "stop": None,
                      "elapsed_seconds": 0.0, "first_acceptance": None, "resumes": list(resumes),
                      "counters": {"steps": 0, "proposals": 0, "proposals_by_operator": {}, "proposals_by_phase": {},
                                   "guided_attempts": 0, "guided_fallback_to_r1": 0, "abstentions": {},
                                   "withheld_levers": {},
                                   "ready": 0, "noop": 0, "rejected": {}, "repair_refused": 0,
                                   "duplicates_within_generation": 0, "cache_hits": 0, "evaluations": 0,
                                   "evaluations_by_phase": {}, "outcomes": {}, "evaluation_errors": 0,
                                   "initialization_draws": 0, "initializer_repair_changed": 0,
                                   "exploration_evaluations": 0, "solves": {"frame": 0, "floor": 0},
                                   "frame_solves_unrecorded_evaluations": 0, "post_evaluation_errors": 0,
                                   "evaluation_seconds": 0.0, "affected_groups_total": 0,
                                   "beam_jump_screens": {"observations": 0, "seconds": 0.0, "pricings": 0, "statuses": {}},
                                   "jump_routes": {"proposed": 0, "local_step": 0, "other_lever": 0, "unit_target": 0,
                                                   "fallbacks": {}},
                                   "jump_proposals_drawn": 0, "jump_annotated_local_steps_drawn": 0,
                                   "jump_unit_target_steps_drawn": 0,
                                   "preferred_selection": {"proposed": 0, "guided_draws": 0, "draws_with_pool": 0,
                                                           "preferred_branch": 0, "normal_branch": 0, "fallbacks": {},
                                                           "not_evaluated": {},
                                                           "evaluated": {"preferred": 0, "normal": 0},
                                                           "incomplete": {"preferred": 0, "normal": 0},
                                                           "errors": {"preferred": 0, "normal": 0},
                                                           "accepted": {"preferred": 0, "normal": 0}}}}
        self._checkpoint()
        return self

    def resume(self, request_sha256=None):
        """Continue the run in ``out`` (after taking its lease). Everything is checked before anything is written:
        the fingerprint, the history's completeness and digest, and that every record after the checkpoint is one an
        interruption can leave (the pending evaluation that finished; one cache-hit origin of the next step, naming
        a known observation; log lines of the next step; one cut final line). A refused resume changes no run file
        (taking the lease may create the lease marker, which holds no run data). With ``request_sha256`` the saved
        request must be that one (request_refusal, under the lease), and a directory whose request was saved but
        which never reached its first checkpoint starts its run. An error record of the interrupted process
        (error.json) is set aside as error.before_resume.N.json."""
        self._fresh()
        self._acquire()
        try:
            if request_sha256 is not None:
                kind = request_refusal(self.out, request_sha256)
                if kind is not None:
                    raise RunDirectoryError(kind, f"{self.out} cannot continue this request ({kind})")
                if not self._path("checkpoint.json").exists():
                    # Started by this resume: checked first, then an earlier error record is set aside, then the
                    # first checkpoint records both (and any set aside by an earlier attempt that failed).
                    self._check_new()
                    earlier = self._unrecorded_error_records(())
                    aside = self._set_aside_error(self._free_number(lambda n: [f"error.before_resume.{n}.json"]))
                    self._start(resumes=[{"history_count": 0, "started_by_resume": True, "error_set_aside": aside,
                                          "unrecorded_error_records": earlier}])
                    return self
            if not self._path("checkpoint.json").exists():
                raise ValueError(f"{self.out} holds no checkpoint: there is no run to resume (a run whose request was "
                                 "saved before its first checkpoint starts on resume(request_sha256))")
            return self._resume()
        except BaseException:
            self._abandon()
            raise

    def _free_number(self, names, start=1):
        """The first N >= ``start`` for which no file ``names(N)`` exists: earlier sidecars are kept, never
        overwritten."""
        number = start
        while any(self._path(name).exists() for name in names(number)):
            number += 1
        return number

    def _unrecorded_error_records(self, resumes):
        """Error records set aside by an earlier resume that failed after the rename, so no checkpoint names them."""
        named = ({r.get("error_set_aside") for r in resumes}
                 | {name for r in resumes for name in r.get("unrecorded_error_records", ())})
        found = [p.name for p in self.out.glob("error.before_resume.*.json") if ERROR_ASIDE.fullmatch(p.name)]
        return sorted((name for name in found if name not in named),
                      key=lambda name: int(ERROR_ASIDE.fullmatch(name).group(1)))

    def _set_aside_error(self, number):
        """error.json of an earlier process, renamed error.before_resume.N.json (None when there was none)."""
        if not self._path("error.json").exists():
            return None
        self._owned()
        name = f"error.before_resume.{number}.json"
        os.replace(self._path("error.json"), self._path(name))
        return name

    def _resume(self):
        checkpoint = json.loads(self._path("checkpoint.json").read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != self.fingerprint:
            raise ValueError("the saved run has another configuration, case, basis, design space or version "
                             "fingerprint; it cannot be resumed under this one")
        lines, partial = self._read_history_lines()
        count = checkpoint["history_count"]
        digest = ""
        for line in lines[:count]:
            digest = _sha(digest + _sha(line))
        if len(lines) < count or digest != checkpoint["history_digest"]:
            raise ValueError("observations.jsonl does not match the checkpoint (truncated or edited history)")
        ids = [record.get("id") for record in map(json.loads, lines[:count]) if record.get("type") == "observation"]
        if len(ids) != len(set(ids)):
            raise ValueError("observations.jsonl names an observation id twice; inspect before resuming")
        known = set(ids)
        adopted, aside, kept = None, [], list(lines[:count])
        pending = checkpoint.get("pending")
        if pending is not None and pending.get("id") in known:
            raise ValueError("the checkpoint's pending evaluation names an observation already in the history; "
                             "inspect before resuming")
        for line in lines[count:]:
            record = json.loads(line)
            if (record.get("type") == "observation" and adopted is None and pending is not None
                    and record.get("input_identity") == pending["input_identity"] and record.get("id") == pending["id"]):
                adopted = record                          # the pending evaluation finished before the interruption
                kept.append(line)
            elif (record.get("type") == "origin" and pending is None and not aside
                  and record.get("step") == checkpoint["step"] + 1 and record.get("observation_id") in known):
                aside.append(line)                        # its step runs again and records it again
            else:
                raise ValueError("observations.jsonl holds records the checkpoint does not explain; inspect before resuming")
        logs = {name: self._plan_log(name, checkpoint["step"], pending is not None)
                for name in ("proposals.jsonl", "generations.jsonl")}
        # Everything is explained: restore the state, then set aside what was written after the checkpoint. The
        # sidecars are durable before a file is replaced, and every replacement is atomic.
        self.state = checkpoint
        rng_state = checkpoint["rng"]
        self.rng = random.Random()
        self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        # A resume cut before its own checkpoint left sidecars of its number: they are kept, never overwritten.
        number = self._free_number(lambda n: [f"observations.partial.{n}.txt", f"observations.after_checkpoint.{n}.txt",
                                              f"error.before_resume.{n}.json"]
                                   + [f"{log}.after_checkpoint.{n}.txt" for log in logs],
                                   start=len(self.state["resumes"]) + 1)
        # The error record goes aside first: a rename that fails leaves every other run file as it was. Records an
        # earlier resume set aside but could not record are named here.
        earlier = self._unrecorded_error_records(self.state["resumes"])
        error_aside = self._set_aside_error(number)
        if partial:
            self._durable_text(self._path(f"observations.partial.{number}.txt"), partial)
        if aside:
            self._durable_text(self._path(f"observations.after_checkpoint.{number}.txt"), "".join(aside))
        for name, (_kept_lines, moved_lines) in logs.items():
            if moved_lines:
                self._durable_text(self._path(f"{name}.after_checkpoint.{number}.txt"), "\n".join(moved_lines) + "\n")
        if partial or aside:
            self._replace_text("observations.jsonl", "".join(kept))
        moved = {}
        for name, (kept_lines, moved_lines) in logs.items():
            moved[name] = len(moved_lines)
            if moved_lines:
                self._replace_text(name, "".join(line + "\n" for line in kept_lines))
        for line in lines[:count]:
            self._history_add(json.loads(line), persist=False)
        if adopted is not None:
            self.state["history_count"] = count + 1
            self.state["history_digest"] = _sha(digest + _sha(kept[-1]))
            self._history_add(adopted, persist=False)
        self.state["resumes"].append({"history_count": count, "adopted_pending_observation": adopted is not None,
                                      "log_lines_moved_aside": moved, "history_lines_moved_aside": len(aside),
                                      "sidecar_number": number, "error_set_aside": error_aside,
                                      "unrecorded_error_records": earlier})
        if adopted is not None:
            # The pending checkpoint's time is the step's start: the adopted evaluation's own time is added, with its
            # jump screen's (the time between its end and the interruption is lost).
            screen = (adopted.get("diagnostics") or {}).get("beam_jump_screen") or {}
            self.state["elapsed_seconds"] += adopted["elapsed_seconds"] + (screen.get("seconds") or 0.0)
            logged = any(json.loads(line).get("observation") == adopted["id"] and json.loads(line).get("status") == "evaluated"
                         for line in (self._path("proposals.jsonl").read_text(encoding="utf-8").splitlines()
                                      if self._path("proposals.jsonl").exists() else []))
            self._absorb(adopted, pending, log=not logged)
        self._checkpoint()
        return self

    def _read_history_lines(self):
        """(complete lines, unterminated tail). Only a final line without its newline (a write the interruption
        cut) is a tail; a malformed complete line is corruption and refuses the resume, changing nothing."""
        path = self._path("observations.jsonl")
        if not path.exists():
            return [], ""
        text = path.read_text(encoding="utf-8")
        cut = text.rfind("\n") + 1
        lines, partial = text[:cut].splitlines(keepends=True), text[cut:]
        for number, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"observations.jsonl line {number} is not a complete record; inspect before resuming") from exc
            if not isinstance(record, dict):
                raise ValueError(f"observations.jsonl line {number} is not a record object; inspect before resuming")
        return lines, partial

    def _plan_log(self, name, step, pending):
        """(kept, moved) lines of a log; nothing is written here. Lines of the step after the checkpoint (it runs
        again) and a final line the interruption cut before its newline (complete or not) are moved to a sidecar by
        ``resume``. A malformed complete line, a line of a later step, or an earlier line after a later one cannot
        come from an interruption and refuse the resume."""
        path = self._path(name)
        if not path.exists():
            return [], []
        text = path.read_text(encoding="utf-8")
        cut = text.rfind("\n") + 1
        kept, moved = [], []
        # A pending checkpoint is written inside its step: only that step's lines can follow it. An end-of-step
        # checkpoint can be followed by lines of the next step only.
        last = step if pending else step + 1
        for number, line in enumerate(text[:cut].splitlines(), 1):
            try:
                at = json.loads(line).get("step")
            except (json.JSONDecodeError, AttributeError) as exc:
                raise ValueError(f"{name} line {number} is not a complete record; inspect before resuming") from exc
            if type(at) is not int or at > last or (moved and at <= step):
                raise ValueError(f"{name} line {number} (step {at!r}) does not follow the checkpoint at step {step}; "
                                 "inspect before resuming")
            (kept if at <= step else moved).append(line)
        tail = text[cut:]
        if tail:
            try:
                at = json.loads(tail).get("step")
            except (json.JSONDecodeError, AttributeError):
                at = None
            # Only the pending evaluation's own line (this step) or a line of the next step can be the cut one.
            if at is not None and (type(at) is not int or at != (step if pending else step + 1)):
                raise ValueError(f"{name} ends with a cut record of step {at!r} that the checkpoint at step {step} "
                                 "does not explain; inspect before resuming")
            moved.append(tail)                            # cut by the interruption: its write is repeated
        return kept, moved

    # ---- the loop ------------------------------------------------------------------------------------------------
    def run(self):
        """Step until a stop. The checkpoint is written only at consistent points (before an evaluation, after each
        step, at the stop): an interruption anywhere leaves the last consistent checkpoint on disk, and ``resume``
        adopts an evaluation that finished after it or evaluates the pending input again. An exception is recorded in
        error.json while the directory is still owned, then raised. The run directory is released when this returns
        or raises, and the controller is then spent."""
        self._owned()
        try:
            self._validate_ceiling()
            return self._run()
        except Exception as exc:
            self._record_error(exc)
            raise
        finally:
            self._clock_origin = None
            self._spent = True
            self.close()

    def _record_error(self, exc):
        """error.json for an exception that ended the run (a write that fails with an Exception never hides the run's
        exception; an interrupt during the write ends the process as any interrupt does)."""
        try:
            self._write("error.json", {"version": VERSION, "experimental": True, "production_accepted": False,
                                       "status": "error", "error": f"{type(exc).__name__}: {_recordable(exception_text(exc))}",
                                       "traceback": traceback.format_exc()[-8000:]})
        except Exception:                                 # noqa: BLE001 -- the run's own exception is raised instead
            pass

    def _run(self):
        started = self.clock()
        base_elapsed = self.state["elapsed_seconds"]
        self._clock_origin = (started, base_elapsed)
        if self.state.get("pending"):
            self._evaluate_pending()
            self.state["elapsed_seconds"] = self._elapsed()       # the evaluation's own time, as on adoption
            self._checkpoint()
        while self.state["stop"] is None:
            self.state["elapsed_seconds"] = self._elapsed()
            stop = self._budget_stop()
            if stop:
                self._final_selection()
                self._stop(*stop)
                break
            self.state["step"] += 1
            self.state["counters"]["steps"] += 1
            if self.state["phase"] == "initialization":
                self._initialization_step()
            else:
                self._generation_step()
            self.state["elapsed_seconds"] = self._elapsed()
            self._checkpoint()
        self.state["elapsed_seconds"] = self._elapsed()
        self._checkpoint()
        return self.result()

    def _elapsed(self):
        """Cumulative run time now (the clock while running; the saved total otherwise)."""
        if self._clock_origin is None:
            return self.state["elapsed_seconds"]
        started, base = self._clock_origin
        return base + (self.clock() - started)

    def _screen_deadline(self):
        """The clock reading a jump screen stops at: its per-observation budget, and no later than the run's wall
        budget (the screen's time is the run's)."""
        now = self.clock()
        deadline = now + self.config.beam_jump_max_seconds
        if self.config.max_wall_seconds is not None:
            deadline = min(deadline, now + max(0.0, self.config.max_wall_seconds - self._elapsed()))
        return deadline

    def _final_selection(self):
        """A stop between steps first selects what was evaluated since the last selection, so the population, its
        fronts and witnesses cover every evaluated child. It is a step of its own (its log lines carry its step)."""
        phase = self.state["phase"]
        if phase == "initialization":
            if not self.history:
                return
        elif not (phase == "generation" and (self.state["pool"] or self.state["new_in_generation"])):
            return
        self.state["step"] += 1
        self.state["counters"]["steps"] += 1
        if phase == "initialization":
            self.state["phase"] = "generation"
            self._select(initial=True, final=True)
        else:
            self._end_generation(final=True)

    def _stop(self, reason, detail=""):
        self.state["stop"] = {"reason": reason, "detail": detail, "structural_infeasibility_established": False,
                              "optimality_established": False, "note": STOP_NOTES.get(reason, BOUND_NOTE)}
        self.state["phase"] = "done"

    def _accepted_summary(self):
        registry = population.accepted_registry(self._records(), self.run_ctx)
        return registry["distinct_realized_designs"], registry["distinct_section_profiles"]

    def _accepted_identities(self):
        return {e["realized_identity"] for e in population.accepted_registry(self._records(), self.run_ctx)["entries"]}

    def _budget_stop(self):
        c, config = self.state["counters"], self.config
        accepted, profiles = self._accepted_summary()
        if config.target_accepted_designs is not None and accepted >= config.target_accepted_designs:
            return "accepted_target_reached", f"{accepted} distinct code-accepted realized designs"
        if config.target_distinct_profiles is not None and profiles >= config.target_distinct_profiles:
            return "profile_target_reached", f"{profiles} distinct section profiles among the accepted designs"
        if c["evaluations"] >= config.max_evaluations:
            return "evaluation_budget_exhausted", f"{c['evaluations']} evaluations"
        if c["proposals"] >= config.max_proposals:
            return "proposal_budget_exhausted", f"{c['proposals']} proposals"
        if config.max_wall_seconds is not None and self.state["elapsed_seconds"] >= config.max_wall_seconds:
            return "time_limit_reached", f"{self.state['elapsed_seconds']:.1f} s (checked between steps)"
        if c["evaluation_errors"] >= config.max_evaluation_errors:
            return "evaluation_error_bound_reached", f"{c['evaluation_errors']} evaluation exceptions"
        if self.state["phase"] == "generation" and self.state["generation"] >= config.max_generations:
            return "generation_limit_reached", f"{self.state['generation']} generations"
        if self.state["stagnant"] >= config.max_stagnant_generations > 0:
            return ("stagnation_bound_reached", f"{self.state['stagnant']} generations with parents in which no new "
                                                "realized design entered the population dominating a member of the "
                                                "previous one or into a free slot, and no accepted identity was added")
        return None

    # ---- contexts --------------------------------------------------------------------------------------------------
    def _seed_context(self):
        seed = self.by_id.get("o0001")
        identity = seed["input_identity"] if seed else diagnostics.digest(["seed", self.basis_id])
        return {"parent_identity": f"initializer:{identity}", "basis_identity": self.basis_id, "genes": self.genes,
                "genome": list(self.space.initial), "groups": self.groups, "joints": {}, "joint_types": {}}

    def _parent_context(self, record):
        topology = self.topologies.get(record["topology_sha256"]) or self._load_topology(record["topology_sha256"])
        return {"parent_identity": record["realized_identity"], "basis_identity": self.basis_id, "genes": self.genes,
                "genome": list(record["genome"]), "groups": self.groups, "joints": topology["joints"],
                "joint_types": topology["joint_types"], "num_floor": int(self.seed_state.counts["num_floor"]),
                "guided_policy": {"beam_jump": self.config.beam_jump_policy}}

    def _load_topology(self, sha):
        value = json.loads(self._path(f"topology/{sha}.json").read_text(encoding="utf-8"))
        self.topologies[sha] = value
        return value

    # ---- steps -----------------------------------------------------------------------------------------------------
    def _initialization_step(self):
        c = self.state["counters"]
        if not self.history and self.state["pending"] is None:
            self._propose({"operator": "SEED", "selected_operator": "SEED", "requested_genome": list(self.space.initial),
                           "origins": [{"operator": "SEED", "reason": "declared_seed", "lever": "seed"}]},
                          phase="initialization", ctx=None)
            return
        draws = 0 if self.config.initialization_policy == "seed_only_v1" else self.config.initialization_evaluations
        initial_evaluations = c["evaluations_by_phase"].get("initialization", 0) - 1
        if (initial_evaluations < draws
                and self.state["init_attempts"] < max(1, draws) * self.config.max_attempts_per_child):
            self.state["init_attempts"] += 1
            self._initializer_draw("initialization")
            return
        self.state["phase"] = "generation"
        self._select(initial=True)

    def _initializer_draw(self, phase):
        c = self.state["counters"]
        c["initialization_draws"] += 1
        ctx = self._seed_context()
        policy = self.config.initialization_policy
        if policy in ("seed_only_v1", "seed_r1_neighbors_v1"):
            proposal = operators.random_proposal(ctx, self.rng)
            if proposal is None:
                self._log("proposals.jsonl", {"phase": phase, "status": "abstain", "reason": "no_mutable_gene"})
                return
            proposal["operator"] = proposal["selected_operator"] = "INIT_R1"
            proposal["origins"] = [{"operator": "INIT_R1", "reason": "initializer_local_neighbor", "lever": "random"}]
        elif policy == "uniform_then_repair_v1":
            genome = [self.rng.randrange(len(g["options"])) for g in self.genes]
            proposal = {"operator": "INIT_UNIFORM", "selected_operator": "INIT_UNIFORM", "requested_genome": genome,
                        "origins": [{"operator": "INIT_UNIFORM", "reason": "uniform_alleles_then_repair", "lever": "random"}]}
        else:
            genome = list(self.space.initial)
            for kind in ("beam", "column"):
                sampler, indices = self.blocks[kind]
                for pair, columns in zip(sampler.sample(self.rng), indices):
                    for allele, j in zip(pair, columns):
                        genome[j] = allele
            for j, g in enumerate(self.genes):
                if g["kind"] == "grade":
                    genome[j] = self.rng.randrange(len(g["options"]))
            proposal = {"operator": "INIT_CONSTRUCTIVE", "selected_operator": "INIT_CONSTRUCTIVE",
                        "requested_genome": genome, "require_fixed_point": True,
                        "origins": [{"operator": "INIT_CONSTRUCTIVE", "reason": "uniform_grammar_profile", "lever": "random"}]}
        proposal.setdefault("parent_identity", ctx["parent_identity"])
        proposal.setdefault("basis_identity", self.basis_id)
        self._propose(proposal, phase=phase, ctx=ctx if proposal["operator"] == "INIT_R1" else None)

    def _generation_step(self):
        config = self.config
        if self.state["new_in_generation"] >= config.offspring_per_generation or \
                self.state["attempts_generation"] >= config.offspring_per_generation * config.max_attempts_per_child:
            self._end_generation()
            return
        self.state["attempts_generation"] += 1
        if not self.state["population"]:
            # No eligible parent: bounded replenishment from the initializer (never a strengthening move); the new
            # observations are selected at the end of the generation; with the budget spent, the case waits.
            if self.state["exploration_used"] < config.max_exploration_evaluations:
                self._initializer_draw("exploration")
                return
            if self.state["pool"]:
                self._end_generation()
                return
            self._stop("waiting_on_dependencies", self._waiting_detail())
            return
        c = self.state["counters"]
        parent = self.by_id[self.rng.choice(sorted(self.state["population"]))]
        ctx = self._parent_context(parent)
        proposal = None
        if self.rng.random() < config.guided_probability:
            c["guided_attempts"] += 1
            alternatives = self.guided_cache.get(parent["id"])
            if alternatives is None:
                alternatives = operators.guided_alternatives(ctx, parent["diagnostics"]["diagnostics"])
                self.guided_cache[parent["id"]] = alternatives
                if parent["id"] not in self.state["routed_parents"]:
                    # Abstentions per routed parent and diagnostic, not per draw (a parent is drawn many times).
                    self.state["routed_parents"].append(parent["id"])
                    for reason, n in alternatives["abstentions"].items():
                        c["abstentions"][reason] = c["abstentions"].get(reason, 0) + n
                    for lever, n in alternatives["withheld"].items():
                        c["withheld_levers"][lever] = c["withheld_levers"].get(lever, 0) + n
                    routes = c["jump_routes"]
                    for key in ("proposed", "local_step", "other_lever", "unit_target"):
                        routes[key] = routes.get(key, 0) + alternatives["jumps"].get(key, 0)
                    for reason, n in alternatives["jumps"]["fallbacks"].items():
                        routes["fallbacks"][reason] = routes["fallbacks"].get(reason, 0) + n
                    if self.config.guided_selection == operators.PREFERRED:
                        c["preferred_selection"]["proposed"] += len(
                            operators.preferred_pool(alternatives, self.selection)["eligible"])
            selection = self.selection
            if self.config.guided_selection == operators.PREFERRED:
                # The pool candidates already drawn for this parent (either branch) are not drawn from it again.
                selection = {**self.selection, "exhausted": self.state["preferred_drawn"].get(parent["id"], [])}
            proposal = operators.choose_guided(alternatives, self.rng, selection)
            chosen = (proposal or {}).get("selection") or {}
            if chosen.get("policy") == operators.PREFERRED:
                preferred = c["preferred_selection"]
                preferred["guided_draws"] += 1
                preferred["draws_with_pool"] += bool(chosen["pool_existed"])
                preferred[chosen["branch"] + "_branch"] += 1
                if chosen.get("fallback"):
                    preferred["fallbacks"][chosen["fallback"]] = preferred["fallbacks"].get(chosen["fallback"], 0) + 1
                if chosen["preferred_candidate"]:
                    # A pool candidate drawn in either branch is spent for this parent (evaluated, or a no-op,
                    # duplicate or cache hit): the preferred branch does not draw it again.
                    self.state["preferred_drawn"].setdefault(parent["id"], []).append(list(proposal["requested_genome"]))
            if proposal is None:
                c["guided_fallback_to_r1"] += 1
            else:
                drawn_kind = jump_draw_kind(proposal["origins"])
                if drawn_kind == "jump":
                    c["jump_proposals_drawn"] += 1              # a genome only a demand-sized jump proposes
                elif drawn_kind == "annotated_local_step":
                    c["jump_annotated_local_steps_drawn"] += 1  # a local step the gene's screen also shortlisted
                elif drawn_kind == "unit_target_step":
                    # a local step annotated only as a unit's own target option (counted apart)
                    c["jump_unit_target_steps_drawn"] = c.get("jump_unit_target_steps_drawn", 0) + 1
        if proposal is None:
            proposal = operators.random_proposal(ctx, self.rng)
        if proposal is None:
            self._log("proposals.jsonl", {"phase": "generation", "parent": parent["id"], "status": "abstain",
                                          "reason": "no_mutable_gene"})
            return
        proposal["parent_observation"] = parent["id"]
        before = {key: c[key] for key in ("noop", "repair_refused", "cache_hits", "duplicates_within_generation",
                                          "evaluations")}
        rejected = sum(c["rejected"].values())
        self._propose(proposal, phase="generation", ctx=ctx, parent=parent)
        if (proposal.get("selection") or {}).get("branch") == "preferred" and c["evaluations"] == before["evaluations"]:
            # Drawn from the preferred branch without a new evaluation: why (it is not drawn from that branch again).
            changed = next((key for key in ("noop", "repair_refused", "cache_hits", "duplicates_within_generation")
                            if c[key] != before[key]), "rejected" if sum(c["rejected"].values()) != rejected else "other")
            missed = c["preferred_selection"]["not_evaluated"]
            missed[changed] = missed.get(changed, 0) + 1

    def _waiting_detail(self):
        records = self._records()
        queue = population.evidence_queue(records, self.run_ctx)
        recovery = population.recovery_queue(records)
        quarantined = sorted({i for c in population.deduplicate(records)[2] for i in c["ids"]},
                             key=population.observation_order)
        top = "; ".join(f"{e['source']} ({len(e['blocked_observations'])} blocked)" for e in queue[:3])
        return (f"no eligible parent and the exploration budget is spent; evidence queue: {top or 'empty'}; "
                f"analysis-recovery queue: {len(recovery)} observation(s); quarantined for contradictory evidence "
                f"(an evaluator determinism problem): {len(quarantined)} observation(s)")

    # ---- one proposal --------------------------------------------------------------------------------------------
    def _propose(self, proposal, *, phase, ctx, parent=None):
        c = self.state["counters"]
        c["proposals"] += 1
        operator = proposal["selected_operator"]
        c["proposals_by_operator"][operator] = c["proposals_by_operator"].get(operator, 0) + 1
        c["proposals_by_phase"][phase] = c["proposals_by_phase"].get(phase, 0) + 1
        record = {"phase": phase, "generation": self.state["generation"], "operator": operator,
                  "parent": None if parent is None else parent["id"], "requested_gene": proposal.get("requested_gene"),
                  "from": proposal.get("from"), "to": proposal.get("to"), "origins": proposal["origins"],
                  "requested_genome": list(proposal["requested_genome"])}
        if proposal.get("selection") is not None:
            record["selection"] = proposal["selection"]           # the guided draw: key, severity, weight, probability
            record["parent_identity"] = proposal.get("parent_identity")
        try:
            repaired, trial, repair_log = self.space.repair(tuple(proposal["requested_genome"]))
        except (ValueError, mg.GroupedStateError) as exc:
            c["repair_refused"] += 1
            self._log("proposals.jsonl", {**record, "status": "rejected", "reason": "repair_refused", "detail": str(exc)})
            return
        repair_log = [list(x) if isinstance(x, tuple) else x for x in repair_log]
        if ctx is not None and operator not in ("SEED",):
            final = operators.finalize(ctx, proposal, list(repaired), repair_log)
        else:
            final = {"status": "ready_for_evaluation", "repaired_genome": list(repaired), "repair_log": repair_log,
                     "requested_change": None,
                     "induced_changes": [{"gene": self.genes[i]["name"], "from": a, "to": b}
                                         for i, (a, b) in enumerate(zip(proposal["requested_genome"], repaired)) if a != b],
                     "affected_groups": [], "affected_member_tags": []}
            if list(repaired) != list(proposal["requested_genome"]) or repair_log:
                c["initializer_repair_changed"] += operator.startswith("INIT")
                if proposal.get("require_fixed_point"):
                    raise ValueError("a constructive profile was changed by the profile repair: the declared grammar "
                                     "disagrees with the repair rules; the initializer is refused for this case")
        record.update(status=final["status"], reason=final.get("reason"), repaired_genome=list(repaired),
                      repair_log=repair_log)
        if final["status"] == "noop":
            c["noop"] += 1
            self._log("proposals.jsonl", record)
            return
        if final["status"] == "rejected":
            c["rejected"][final["reason"]] = c["rejected"].get(final["reason"], 0) + 1
            self._log("proposals.jsonl", record)
            return
        c["ready"] += 1
        input_identity = diagnostics.digest({"basis": self.basis_id, "submitted": trial.identity()})
        record["input_identity"] = input_identity
        existing = self.by_input.get(input_identity)
        if existing is not None:
            pool = self.state["pool"] if self.state["phase"] == "generation" else []
            if existing in pool:
                c["duplicates_within_generation"] += 1
                status = "duplicate_within_generation"
            else:
                c["cache_hits"] += 1
                status = "cache_hit"
                if self.state["phase"] == "generation":
                    self.state["pool"].append(existing)       # a child again: selected (or queued) with this generation
            event = {"type": "origin", "observation_id": existing, "origins": proposal["origins"], "status": status,
                     "step": self.state["step"]}
            self._history_add(event)
            self._log("proposals.jsonl", {**record, "status": status, "observation": existing})
            return
        index = len(self.history) + 1
        intervention = {"requested": final.get("requested_change"), "induced": final.get("induced_changes", []),
                        "repair_log": repair_log, "affected_groups": final.get("affected_groups", []),
                        "affected_member_tags": len(final.get("affected_member_tags", []))}
        c["affected_groups_total"] += len(intervention["affected_groups"])
        self.state["pending"] = {"id": f"o{index:04d}", "phase": phase, "generation": self.state["generation"],
                                 "input_identity": input_identity, "genome": list(repaired), "proposal": record,
                                 "parent": None if parent is None else parent["id"], "intervention": intervention}
        self._checkpoint()
        self._evaluate_pending()

    # ---- one evaluation --------------------------------------------------------------------------------------------
    def _evaluate_pending(self):
        self._validate_ceiling()
        pending = self.state["pending"]
        trial = self.space.decode(tuple(pending["genome"]))
        trial, _changes = gs.enforce_banded_section_order(trial)
        submitted = trial.identity()
        if diagnostics.digest({"basis": self.basis_id, "submitted": submitted}) != pending["input_identity"]:
            raise ValueError("the pending evaluation no longer decodes to its recorded input identity")
        environment = {key: getattr(sp, key) for key in ENVIRONMENT_KEYS}
        floor_before = self.context.get("floor_solves", 0)
        started = self.clock()
        raw, error = None, None
        try:
            with mg.installed(trial):
                raw = self.evaluator(trial, self.cfg, self.context, label=f"genetic_{pending['id']}")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:                          # noqa: BLE001 -- recorded, never read as a design failure
            error = {"stage": "evaluator", "type": type(exc).__name__, "message": _recordable(exception_text(exc)),
                     "traceback": traceback.format_exc()[-4000:]}
        finally:
            for key, value in environment.items():
                setattr(sp, key, value)
        self._validate_ceiling()
        if sp.FY_KSI != self.steel_grade:
            raise ValueError("the longitudinal steel grade changed during an evaluation; it is fixed per run")
        elapsed = self.clock() - started
        # The floor solves the evaluator spent, read from its shared cache (an exception carries no solve record).
        floor = self.context.get("floor_solves", 0) - floor_before
        written = {}                                      # the artifacts written for this observation, as they are
        try:
            record = self._observation(pending, trial, submitted, raw, error, elapsed, floor, artifacts=written)
            _json(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:                          # noqa: BLE001 -- the evaluation finished; its processing failed
            # Assessing, diagnosing or saving the result failed: a recorded error observation (counted against the
            # error bound, not replayed by a resume), never a design failure. It keeps every artifact already
            # written (the evaluation for analysis recovery), and holds only recordable values of the evaluator's
            # result, so it can always be saved.
            error = {"stage": "post_evaluation", "type": type(exc).__name__, "message": _recordable(exception_text(exc)),
                     "traceback": traceback.format_exc()[-4000:],
                     "evaluator_status": _recordable(raw.get("status"), 200) if isinstance(raw, dict) else None}
            solves = (raw or {}).get("solves") if isinstance(raw, dict) else None
            record = self._observation(pending, trial, submitted, None, error, elapsed, floor,
                                       solves=solves if isinstance(solves, dict) else None, artifacts=dict(written))
        self._history_add(record)
        self._absorb(record, pending)

    def _observation(self, pending, trial, submitted, raw, error, elapsed, floor_solves=0, solves=None, artifacts=None):
        base = {"type": "observation", "id": pending["id"], "step": self.state["step"], "phase": pending["phase"],
                "generation": pending["generation"], "case_id": self.case_id, "basis_id": self.basis_id,
                "input_identity": pending["input_identity"], "submitted_identity": submitted,
                "genome": list(pending["genome"]), "parent": pending["parent"], "proposal": pending["proposal"],
                "origins": pending["proposal"]["origins"], "intervention": pending["intervention"],
                "features": population.features(self.genes, pending["genome"])[0],
                "elapsed_seconds": elapsed, "qualification": "not_performed", "realized_identity": None,
                "artifacts": artifacts if artifacts is not None else {}}
        if error is not None:
            if solves is not None:
                recorded = {key: solves.get(key) if isinstance(solves.get(key), int) else None for key in ("frame", "floor")}
            else:
                recorded = {"frame": None, "floor": floor_solves}    # the frame solves of a raising evaluation are unknown
            return {**base, "outcome": "error", "valid": False, "classification": "invalid", "error": error,
                    "numerical_complete": False, "problems": [f"{error['type']}: {error['message'][:300]}"],
                    "solves": recorded, "nonlinear_performance": dict(contract.NONLINEAR_STATUS)}
        from Design.Grouped_Design import _json_safe, public_evaluation
        assessed = contract.assess_evaluation(raw, submitted_identity=submitted, cfg=self.cfg, submitted_state=trial)
        public = public_evaluation(raw)
        artifact = f"evaluations/{pending['id']}.json"
        self._write(artifact, public)
        base["artifacts"]["evaluation"] = artifact
        base["artifacts"]["evaluation_sha256"] = _sha((self._path(artifact)).read_text(encoding="utf-8"))
        check = assessed.get("code_check") or {}
        missing = check.get("missing") or []
        state = raw.get("state")
        # The contract's own inventory of the final design (it names each pattern's family and checks).
        expected = contract.evaluation_inventory(raw, self.cfg) if check and state is not None else None
        record = {**base, "outcome": raw.get("status"), "valid": assessed["valid"], "problems": assessed["problems"],
                  "classification": assessed["classification"], "accepted": assessed["accepted"],
                  "numerical_complete": assessed["numerical_completion"]["complete"],
                  "incomplete_flags": assessed["numerical_completion"]["problems"],
                  # A completion through a verified cycle resolution is named, never read as a convergence.
                  "selection_flags_met_by": assessed["numerical_completion"].get("selection_flags_met_by"),
                  "preliminary_screen": (assessed.get("preliminary_screen") or {}).get("passed"),
                  "code_check_status": check.get("status"), "code_check_counts": check.get("counts"),
                  "review_dependencies": assessed.get("review_dependencies"),
                  # Every missing source with its full count; the item list itself is a sample.
                  "missing": missing[:40], "missing_total": len(missing),
                  "missing_counts": _missing_counts(missing, (expected or {}).get("patterns", ())),
                  "nonlinear_performance": assessed["nonlinear_performance"],
                  "quantities": {k: v["value"] if v["available"] else None
                                 for k, v in (assessed.get("objectives") or {}).items()},
                  "solves": _json_safe(raw.get("solves") or {}), "evaluator_seconds": raw.get("elapsed_seconds"),
                  "failed_stage": raw.get("failed_stage"), "stop_reason": raw.get("reason")}
        if raw.get("status") == "evaluated" and assessed["valid"] and state is not None:
            realized = raw["code_checks"]["realized_identity"]
            evidence = raw.get("evidence") or {}
            ctx = diagnostics.parent_context(self.space, state, basis_identity=self.basis_id, realized_identity=realized,
                                             capacity=evidence.get("capacity"))
            topology = {"joints": ctx["joints"], "joint_types": ctx["joint_types"]}
            if ctx["topology_sha256"] not in self.topologies:
                if not self._path(f"topology/{ctx['topology_sha256']}.json").exists():
                    self._write(f"topology/{ctx['topology_sha256']}.json", topology)
                self.topologies[ctx["topology_sha256"]] = topology       # only once the file exists
            diag = diagnostics.diagnose(raw, assessed, ctx, cfg=self.cfg)
            if self.config.beam_jump_policy == jumps.DEMAND_SIZED:
                # While the evaluator's record is at hand: demand-sized options of the beam flexural shortfalls, kept
                # with the diagnostics (bounded; a failed screen gives no target; never a pass).
                jumps.screen_evaluation(raw, diag, ctx, self.cfg, self.space, self.jump_params, clock=self.clock,
                                        deadline=self._screen_deadline())
            if expected is None:
                expected = contract.evaluation_inventory(raw, self.cfg)
            checks, inventory = population.normalize_checks(raw["code_checks"]["rows"], expected)
            cages = _cages(state)
            parent = self.by_id.get(pending["parent"]) if pending["parent"] else None
            record.update(realized_identity=realized, state_identity=state.identity(), checks=checks,
                          inventory_id=inventory["inventory_id"], inventory_items=inventory["items"],
                          known_items=inventory["known_items"], diagnostics=diag, unknown=diag["unknown"],
                          topology_sha256=ctx["topology_sha256"], topology_problems=ctx["topology_problems"], cages=cages)
            record["intervention"] = {**record["intervention"],
                                      "reinforcement_vs_submitted": _cage_changes(_cages(trial), cages),
                                      "reinforcement_vs_parent": (_cage_changes(parent["cages"], cages)
                                                                  if parent and parent.get("cages") else None)}
            if parent and parent.get("checks") is not None:
                record["vs_parent"] = self._versus_parent(parent, record, evidence)
            if assessed["accepted"]:
                self._design_artifact(pending["id"], raw, state, realized, _json_safe, record["artifacts"])
                self._record_artifact(pending["id"], raw, record["artifacts"])
        return record

    def _versus_parent(self, parent, child, evidence):
        """The child against its parent, apart from the child's own governing maximum: each requesting diagnostic's
        check (component) before and after, the collateral regressions over every check, the evidence gained, and
        the material quantities. The parent's value is its diagnostic; the child's comes from its own record."""
        before = {d["id"]: d for d in (parent.get("diagnostics") or {}).get("diagnostics") or []}
        strengths = (evidence.get("strengths") or {}).get("groups") if isinstance(evidence.get("strengths"), dict) else {}
        targets, seen = [], set()
        for origin in child["origins"]:
            did = origin.get("diagnostic_id")
            if not did or did in seen:
                continue
            seen.add(did)
            key, _, component = did.partition("#")
            family, check_id, location = (key.split("|", 2) + ["", ""])[:3]
            old = before.get(did) or {}
            entry = {"diagnostic_id": did, "reason": origin.get("reason"), "operator": origin.get("operator"),
                     "parent": {"status": old.get("status"), "demand": old.get("demand"), "capacity": old.get("capacity"),
                                "comparison": old.get("comparison"), "units": old.get("units")}}
            if component:
                part = ((strengths or {}).get(location) or {}).get(component) or {}
                dcr = part.get("dcr")
                entry["child"] = {"status": ("fail" if dcr > 1.0 else "pass") if diagnostics.finite(dcr) else "unknown",
                                  "dcr": dcr if diagnostics.finite(dcr) else None}
                old_dcr = (old.get("demand") / old.get("capacity") if diagnostics.finite(old.get("demand"))
                           and diagnostics.finite(old.get("capacity")) and old.get("capacity") > 0 else None)
                entry["parent"]["dcr"] = old_dcr
                entry["improved"] = (entry["child"]["dcr"] < old_dcr if entry["child"]["dcr"] is not None
                                     and old_dcr is not None else None)
            else:
                canonical = population.canonical_key({"family": family, "id": check_id, "location": location})
                new, prior = child["checks"].get(canonical), (parent.get("checks") or {}).get(canonical)
                entry["child"] = ({"status": new["s"], "violation": new["v"], "utilization": new["u"]} if new
                                  else {"status": "not_required"})
                entry["parent"].update(violation=(prior or {}).get("v"), utilization=(prior or {}).get("u"))
                comparable = bool(new and prior and new["b"] == prior["b"] and new["s"] in population.KNOWN
                                  and prior["s"] in population.KNOWN)
                entry["improved"] = (new["s"] == "pass" or new["v"] < prior["v"]) if comparable else None
            estimate = next((o["jump"] for o in child["origins"]
                             if o.get("diagnostic_id") == did and isinstance(o.get("jump"), dict)), None)
            if estimate is not None:
                # The screen's estimate on the frozen parent demand, kept apart from the evaluated outcome.
                entry["estimated"] = {k: estimate.get(k) for k in ("estimated_dcr", "matched", "meets_target_estimated",
                                                                   "smallest_confirmed", "pool", "target_dcr", "outcome",
                                                                   "basis")}
            entry["resolved"] = entry["child"]["status"] == "pass"
            targets.append(entry)
        gained = sorted(k for k, e in child["checks"].items() if e["s"] in population.KNOWN
                        and ((parent.get("checks") or {}).get(k) or {}).get("s") not in population.KNOWN)
        return {"parent": parent["id"], "targets": targets, "collateral": population.regressions(parent, child),
                "evidence_gained": {"count": len(gained), "ids": gained[:20]},
                "quantities": {"parent": parent.get("quantities"), "child": child.get("quantities")}}

    def _design_artifact(self, observation_id, raw, state, realized, json_safe, artifacts):
        """designs/<id>.json: everything the realized identity hashes (member groups, slab thickness, slab layout with
        its inputs and floor context) and every code-check row, so the registry key can be verified and the design
        rebuilt or qualified without evaluating it again. Named in ``artifacts`` (the observation's) as soon as it
        exists, before it is read back for its hash."""
        slab = (raw.get("evidence") or {}).get("slab") or {}
        saved = json_safe({"slab": slab.get("slab"), "reinforcement": slab.get("reinforcement")})
        thickness = (saved.get("slab") or {}).get("thickness_in")
        value = {"observation": observation_id, "realized_identity": realized, "state_identity": state.identity(),
                 "basis_id": self.basis_id, "design_evidence": (self.basis or {}).get("design_evidence"),
                 "member_groups": state.to_record(), "slab": saved,
                 "code_check_rows": json_safe(raw["code_checks"]["rows"]),
                 "realized_identity_reproducible": contract.realized_identity(state.identity(), thickness,
                                                                              saved.get("reinforcement")) == realized,
                 "qualification": "not_performed"}
        name = f"designs/{observation_id}.json"
        self._write(name, value)
        artifacts["design"] = name
        artifacts["design_sha256"] = _sha(self._path(name).read_text(encoding="utf-8"))

    def _record_artifact(self, observation_id, raw, artifacts):
        """records/<id>.json.gz: the saved record of an accepted design and its replay summary. A failure here is recorded
        beside the design; it does not change the code-check classification of the candidate."""
        if not self.save_records:
            return
        from Design import Grouped_Record as gr
        name = f"records/{observation_id}.json.gz"
        try:
            saved = gr.save_evaluation_record(raw, self.cfg, self.basis.get("seed"), self._path(name))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:                          # noqa: BLE001 -- recorded with the observation
            artifacts["record_error"] = _recordable(f"{type(exc).__name__}: {exception_text(exc)}")
            return
        artifacts["record"] = name
        artifacts["record_sha256"] = saved["sha256"]
        artifacts["record_replay"] = saved["replay"]

    def _absorb(self, record, pending, log=True):
        """Bookkeeping after an observation is in the history (also on resume when a pending one is adopted)."""
        c = self.state["counters"]
        c["evaluations"] += 1
        phase = record["phase"]
        c["evaluations_by_phase"][phase] = c["evaluations_by_phase"].get(phase, 0) + 1
        outcome = "error" if record["outcome"] == "error" else \
            "infeasible" if record["outcome"] == "infeasible" else record["classification"]
        c["outcomes"][outcome] = c["outcomes"].get(outcome, 0) + 1
        c["evaluation_errors"] += record["outcome"] == "error"
        c["post_evaluation_errors"] += (record.get("error") or {}).get("stage") == "post_evaluation"
        c["exploration_evaluations"] += phase == "exploration"
        if phase == "exploration":
            self.state["exploration_used"] += 1
        for key in ("frame", "floor"):
            value = (record.get("solves") or {}).get(key)
            c["solves"][key] += value if isinstance(value, int) else 0
        c["frame_solves_unrecorded_evaluations"] += not isinstance((record.get("solves") or {}).get("frame"), int)
        c["evaluation_seconds"] += record["elapsed_seconds"]
        branch = ((record.get("proposal") or {}).get("selection") or {}).get("branch")
        if branch in ("preferred", "normal") and (record.get("proposal") or {}).get("selection", {}).get("policy") \
                == operators.PREFERRED:
            # Completely evaluated, evaluated incompletely, or an evaluator or processing error: kept apart.
            tier = ("errors" if record["outcome"] == "error" else
                    "evaluated" if record.get("numerical_complete") is True else "incomplete")
            c["preferred_selection"][tier][branch] += 1
            c["preferred_selection"]["accepted"][branch] += bool(record.get("accepted"))
        screen = record["diagnostics"].get("beam_jump_screen") if isinstance(record.get("diagnostics"), dict) else None
        if isinstance(screen, dict):
            screens = c["beam_jump_screens"]
            screens["observations"] += 1
            screens["seconds"] += screen["seconds"]
            screens["pricings"] += screen["pricings"]
            screens["statuses"][screen["status"]] = screens["statuses"].get(screen["status"], 0) + 1
        if record.get("accepted") and self.state["first_acceptance"] is None:
            # Elapsed time with the accepting evaluation included (an adopted one: resume has added its own time).
            self.state["first_acceptance"] = {"observation": record["id"], "evaluations": c["evaluations"],
                                              "elapsed_seconds": self._elapsed()}
        if self.state["phase"] == "generation" or phase == "exploration":
            # Every child joins the generation's selection, a stopped or raised one included (it is queued, never a
            # parent).
            self.state["pool"].append(record["id"])
            self.state["new_in_generation"] += phase in ("generation", "exploration")
        self.state["pending"] = None
        if log:
            self._log("proposals.jsonl", {**pending["proposal"], "status": "evaluated", "observation": record["id"],
                                          "classification": record["classification"], "outcome": record["outcome"]})
        if self.verbose:
            print(f"GA v3 {record['id']} [{phase}, gen {record['generation']}] {pending['proposal']['operator']}: "
                  f"{outcome}; {c['evaluations']}/{self.config.max_evaluations} evaluations", flush=True)

    # ---- selection -------------------------------------------------------------------------------------------------
    def _records(self):
        """Selection records with every origin, including those that reached an observation by a cache hit."""
        return [{**r, "origins": list(r.get("origins") or []) + self.origin_events.get(r["id"], [])} for r in self.history]

    def _select(self, initial=False, final=False):
        history = self._records()
        records = {r["id"]: r for r in history}
        before = sorted(self.state["population"])
        ids = sorted(set(self.state["population"]) | set(self.state["pool"])) if not initial else sorted(records)
        # The quarantine is computed over the whole history: a design contradicted by any observation is no parent.
        decision = population.select([records[i] for i in ids], self.config.population_size, self.run_ctx,
                                     self.config.random_seed, self.state["generation"], history=history)
        self.state["population"] = decision["selected_ids"]
        new_evaluations = self.state["new_in_generation"]
        changed = sorted(decision["selected_ids"]) != before
        # Progress is new information: a new realized design evaluated in this generation entered the population and
        # dominates a member of the previous population, or entered while that population (less its members
        # quarantined now) had a free slot, or a code-accepted identity was added to the registry (against the
        # identities when this generation began; its evaluations are already in the history). An entry by the
        # tie-break (its seed changes every generation) or by rank alone is not progress, nor is a population change
        # among earlier designs (a cache hit readmitted) or a design an earlier observation already realized
        # (reached again through another input, its evidence identical or quarantined).
        generation = self.state["generation"]
        fresh = {i for i in self.state["pool"] if records[i]["phase"] in ("generation", "exploration")
                 and records[i]["generation"] == generation}
        earlier = {population.identity_key(r) for r in history if r["id"] not in fresh and r.get("realized_identity")}
        rediscovered = sorted((i for i in fresh if population.identity_key(records[i]) in earlier),
                              key=population.observation_order)
        entered = sorted(fresh.difference(rediscovered) & set(decision["selected_ids"]), key=population.observation_order)
        audits = decision["audits"]
        # Each member is compared on its own record: one deduplicated under an earlier id that a cache hit readmitted
        # has no audit of its own in the decision. A member quarantined now is no comparison (its evidence is
        # contradicted); its slot is free.
        quarantined = {i for conflict in decision["conflicts"] for i in conflict["ids"]}
        members = [audits.get(m) or population.audit(records[m], self.run_ctx) for m in before if m not in quarantined]
        members = [a for a in members if a.get("eligible")]
        improving = [i for i in entered if i in audits and any(population.dominates(audits[i], a) for a in members)]
        free_slot = bool(entered) and len(set(before) - quarantined) < self.config.population_size
        accepted_now = self._accepted_identities()
        added = sorted(accepted_now - set(self.state["accepted_identities_at_generation_start"]))
        grew = bool(added)
        self.state["accepted_identities_at_generation_start"] = sorted(accepted_now)
        if not initial:
            if improving or free_slot or grew:
                self.state["stagnant"] = 0
            elif before:
                self.state["stagnant"] += 1
            # A generation that began without a parent drew exploration designs: their entries fill free slots
            # (bounded by the exploration budget, then the case waits).
        blocked = {}
        for item in decision["queues"]:
            blocked[item["reason"]] = blocked.get(item["reason"], 0) + 1
        status = ("full" if not decision["underfilled"] else "underfilled" if decision["selected_ids"] else "empty")
        self._log("generations.jsonl", {"generation": self.state["generation"], "initial": initial,
                                        "candidates": ids, "decision": decision, "population_status": status,
                                        "blocked_by": blocked, "new_evaluations": new_evaluations,
                                        "population_changed": changed, "new_entrants": entered,
                                        "improving_entrants": improving, "free_slot_entry": free_slot,
                                        "rediscovered_designs": rediscovered, "accepted_identities_added": len(added),
                                        "accepted_grew": grew, "stagnant_generations": self.state["stagnant"]})
        self.state["last_population_status"] = {"status": status, "selected": len(decision["selected_ids"]),
                                                "requested": self.config.population_size, "blocked_by": blocked}
        # No new evaluation in a whole generation: the neighbourhood gave nothing new. Readmitted cache hits do not
        # postpone the stop, except when they refill a population that had no parent.
        if not initial and not final and new_evaluations == 0 and (before or not decision["selected_ids"]):
            self._stop("no_new_candidate_within_proposal_budget",
                       "a whole generation produced no new evaluation (no-ops, duplicates, cache hits or rejections)")

    def _end_generation(self, final=False):
        self._select(final=final)
        self.state["generation"] += 1
        self.state["pool"], self.state["attempts_generation"], self.state["new_in_generation"] = [], 0, 0

    # ---- outputs ---------------------------------------------------------------------------------------------------
    def result(self):
        """The run's summary, saved as result.json (under the lease)."""
        summary = self.summary()
        self._write("result.json", summary)
        return summary

    def summary(self):
        """The run's summary from the state in memory, written nowhere (a spent controller can still give it). It is a
        copy throughout, views included: changing it leaves the controller's state and history as they were."""
        if self.state is None:
            raise ValueError(f"this controller holds no run of {self.out}; call start() or resume() first")
        records = self._records()
        views = population.views(records, self.run_ctx, self.config.random_seed, self.config.diversity_view_size)
        c = self.state["counters"]
        registry = views["accepted_registry"]
        evaluations = max(c["evaluations"], 1)
        summary = {
            "version": VERSION, "experimental": True, "production_accepted": False, "case_id": self.case_id,
            "basis_id": self.basis_id, "fingerprint": self.fingerprint, "config": asdict(self.config),
            "design_evidence": (self.basis or {}).get("design_evidence"),
            "stop": self.state["stop"], "phase": self.state["phase"],
            "generation": self.state["generation"], "counts": c, "population": self.state["population"],
            "population_status": self.state.get("last_population_status"),
            "metrics": {"distinct_code_accepted_realized_designs": registry["distinct_realized_designs"],
                        "distinct_accepted_section_profiles": registry["distinct_section_profiles"],
                        "distinct_accepted_genomes": registry["distinct_genomes"],
                        "qualified_designs": 0,
                        "first_acceptance": self.state["first_acceptance"],
                        "accepted_per_evaluation": registry["distinct_realized_designs"] / evaluations,
                        "noop_rate": c["noop"] / max(c["proposals"], 1),
                        "duplicate_or_cache_rate": (c["cache_hits"] + c["duplicates_within_generation"]) / max(c["proposals"], 1),
                        "mean_affected_groups_per_evaluated_child": c["affected_groups_total"] / evaluations,
                        "elapsed_seconds": self.state["elapsed_seconds"]},
            "views": views,
            "semantics": {"code_accepted": "the scoped code check of Design.Code_Check_Contract passed, the evaluation "
                                           "completed and the objectives are available; review dependencies listed",
                          "qualification": "not performed here (Grouped_Record.qualify_grouped_design is separate)",
                          "nonlinear_performance": "not requested",
                          "survivor": "a working-population member is a search parent, not an accepted design",
                          "evidence": "software behavior is tested; structural and search performance are not validated"},
        }
        return copy.deepcopy(summary)         # the views hold the history records' own genomes, artifacts and origins
