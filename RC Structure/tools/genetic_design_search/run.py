"""Separate experimental entry point; never emits a production design record.

Two engines: ``heuristic_v2`` (the existing iterative genetic search, Design.Genetic_Search.search, unchanged and the
default) and ``generation_v3`` (the bounded serial generation controller, Design.Genetic_Generation: six mutation
operators on normalized diagnostics, versioned population selection, accepted-design registry, checkpoint/resume).

``--beam-rho-practical-max`` sets the run's practical beam face steel ratio ceiling (RebarConfig.rho_beam_practical_max,
default 0.015) up to the authorized Design.Config.BEAM_PRACTICAL_RATIO_AUTHORIZED_MAX, never above; the request records
it (design_config_overrides) and it enters the evaluation identity, so the basis, the cache and a resume follow it.
generation_v3 declares its proposal policies (--guided-selection, --beam-jumps and their parameters) in the request's
generation_config and its basis (proposal_policy); the defaults are the earlier rules. --guided-selection
severity_preferred_jumps_v1 (with --beam-jumps demand_sized_v1) gives --preferred-share of the guided draws to the
severe screened demand-sized proposals (failures at --preferred-severe-utilization and above).

``--design-evidence PROFILE.json`` loads the run's design-evidence profile (Design.Design_Evidence_Profile, schema
design_evidence_profile_v1; tools/genetic_design_search/evidence_profiles/README.md): a slab refinement plan and the
engineering slab-action assertions, apart, with provenance, review and declared scope. It is validated and checked
against the run's case, profile, geometry, slab materials, loads and slab policy before anything is evaluated, and
its plan must resolve on the seed's floors; the configuration it implies enters the evaluation identity, its summary
(identity sha256 included, with the seed's plan resolution) the request (``design_evidence``), and its id, kind, review
status, asserted flag and identity sha256 the generation_v3 basis, so the cache, the registry and a resume follow it.
Without it the run keeps the unreviewed
defaults (single-mesh slab actions, nothing asserted). --plan-only prints the request with the profile's summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[2]
for path in (RC_DIR, RC_DIR / "Data_Generation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

ENGINES = ("heuristic_v2", "generation_v3")
# Options of one engine only: None means "not given", so the other engine can refuse it. The generation_v3 names
# are chosen so that no heuristic_v2 option, full or abbreviated as the earlier runner accepted it, changes meaning.
V2_ONLY = {"heuristic_probability": .75, "crossover_probability": .8, "random_jump_probability": .2,
           "max_proposals_per_evaluation": 30}
V3_ONLY = {"offspring_per_generation": None, "proposal_budget": 400, "max_wall_seconds": None,
           "initialization": "seed_r1_neighbors_v1", "initialization_evaluations": 3, "guided_probability": .75,
           "max_exploration_evaluations": 4, "max_stagnant_generations": 3, "max_attempts_per_child": 30,
           "target_accepted": None, "target_distinct_profiles": None, "diversity_view_size": 8,
           "evaluation_error_limit": 2, "guided_selection": "uniform_operator_v1", "severity_floor": .2,
           "severity_cap": 1.0, "categorical_weight": .5, "beam_jumps": "local_frontier_v1", "jump_target_dcr": .85,
           "jump_max_sections": 8, "jump_max_seconds": 15.0, "jump_partial_tolerance": .02, "jump_min_gain": .02,
           "preferred_severe_utilization": 1.25, "preferred_share": .80}
EXIT_ERROR_BOUND = 3                      # generation_v3 stopped at its evaluation-error bound (see the first error)


def parser():
    from Design.Grouped_Runner import GROUPINGS, DEFAULT_GROUPING
    from Design.Genetic_Generation import INITIALIZATION_POLICIES
    from Design.Genetic_Jumps import POLICIES as JUMP_POLICIES
    from Design.Genetic_Operators import SELECTION_POLICIES
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--engine", choices=ENGINES, default="heuristic_v2")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Existing screening manifest")
    source.add_argument("--case-file", type=Path, help="JSON object describing one case")
    result.add_argument("--case-id", help="Required with --manifest")
    result.add_argument("--profile", help="Optional explicit analysis profile with --case-file")
    result.add_argument("--grouping", choices=GROUPINGS, default=DEFAULT_GROUPING)
    result.add_argument("--seed-column", nargs=3, type=float, required=True, metavar=("B", "H", "FC"))
    result.add_argument("--seed-beam", nargs=3, type=float, required=True, metavar=("B", "H", "FC"))
    result.add_argument("--population-size", type=int, default=None, help="heuristic_v2: 8; generation_v3: 4")
    result.add_argument("--max-evaluations", type=int, default=24)
    result.add_argument("--max-generations", type=int, default=None,
                        help="heuristic_v2: offspring generations after the initial population (10); generation_v3: 6")
    result.add_argument("--random-seed", type=int, default=20261030)
    result.add_argument("--heuristic-probability", type=float, default=None, help="heuristic_v2 only (0.75)")
    result.add_argument("--crossover-probability", type=float, default=None, help="heuristic_v2 only (0.8)")
    result.add_argument("--random-jump-probability", type=float, default=None, help="heuristic_v2 only (0.2)")
    result.add_argument("--max-proposals-per-evaluation", type=int, default=None, help="heuristic_v2 only (30)")
    result.add_argument("--offspring-per-generation", type=int, default=None, help="generation_v3: default the population size")
    result.add_argument("--proposal-budget", type=int, default=None, help="generation_v3: every proposal attempt (400)")
    result.add_argument("--max-wall-seconds", type=float, default=None,
                        help="generation_v3: cumulative, checked between steps (an evaluation in progress finishes)")
    result.add_argument("--initialization", choices=INITIALIZATION_POLICIES, default=None,
                        help="generation_v3 (seed_r1_neighbors_v1)")
    result.add_argument("--initialization-evaluations", type=int, default=None, help="generation_v3: draws after the seed (3)")
    result.add_argument("--guided-probability", type=float, default=None, help="generation_v3 (0.75)")
    result.add_argument("--max-exploration-evaluations", type=int, default=None,
                        help="generation_v3: exploration evaluations over the whole run while no parent is eligible; "
                             "no-op, duplicate and cache-hit draws are not counted (4)")
    result.add_argument("--max-stagnant-generations", type=int, default=None, help="generation_v3 (3)")
    result.add_argument("--max-attempts-per-child", type=int, default=None, help="generation_v3 (30)")
    result.add_argument("--target-accepted", type=int, default=None,
                        help="generation_v3: stop at this many distinct code-accepted realized designs")
    result.add_argument("--target-distinct-profiles", type=int, default=None,
                        help="generation_v3: stop at this many distinct accepted section profiles")
    result.add_argument("--diversity-view-size", type=int, default=None, help="generation_v3 (8)")
    result.add_argument("--evaluation-error-limit", type=int, default=None,
                        help="generation_v3: stop at this many evaluation exceptions, raised by the evaluator or in "
                             "processing its result (2)")
    result.add_argument("--beam-rho-practical-max", type=float, default=None,
                        help="the run's practical beam face steel ratio ceiling (RebarConfig.rho_beam_practical_max, "
                             "default 0.015; at most the authorized 0.0175; the 18.6.3.1 limit governs when lower)")
    result.add_argument("--guided-selection", choices=SELECTION_POLICIES, default=None,
                        help="generation_v3: uniform_operator_v1 (the earlier rule, default) or severity_weighted_v1")
    result.add_argument("--severity-floor", type=float, default=None,
                        help="generation_v3, severity_weighted_v1: the least weight of an eligible failure (0.2)")
    result.add_argument("--severity-cap", type=float, default=None,
                        help="generation_v3, severity_weighted_v1: the severity at which a failure's weight reaches 1 (1.0)")
    result.add_argument("--categorical-weight", type=float, default=None,
                        help="generation_v3, severity_weighted_v1: the weight of a failure without a defensible ratio (0.5)")
    result.add_argument("--beam-jumps", choices=JUMP_POLICIES, default=None,
                        help="generation_v3: local_frontier_v1 (the earlier rule, default) or demand_sized_v1")
    result.add_argument("--jump-target-dcr", type=float, default=None,
                        help="generation_v3, demand_sized_v1: the DCR a jump is sized to on frozen demand (0.85)")
    result.add_argument("--jump-max-sections", type=int, default=None,
                        help="generation_v3, demand_sized_v1: sections priced per gene, the parent's included (8)")
    result.add_argument("--jump-max-seconds", type=float, default=None,
                        help="generation_v3, demand_sized_v1: screen time per observation, within the wall budget (15)")
    result.add_argument("--jump-partial-tolerance", type=float, default=None,
                        help="generation_v3, demand_sized_v1: a partial option within this DCR of the best (0.02)")
    result.add_argument("--jump-min-gain", type=float, default=None,
                        help="generation_v3, demand_sized_v1: the least DCR gain of a partial option (0.02)")
    result.add_argument("--preferred-severe-utilization", type=float, default=None,
                        help="generation_v3, severity_preferred_jumps_v1: the failure utilization from which its screened "
                             "demand-sized proposals enter the preferred pool (1.25; a search parameter, not a code limit)")
    result.add_argument("--preferred-share", type=float, default=None,
                        help="generation_v3, severity_preferred_jumps_v1: the share of guided draws given to the preferred "
                             "pool when it is not empty (0.80)")
    result.add_argument("--design-evidence", type=Path, default=None,
                        help="a design-evidence profile (design_evidence_profile_v1): slab refinement plan and slab-action "
                             "assertions with provenance, review and scope; default: single-mesh, nothing asserted")
    result.add_argument("--resume", action="store_true", help="generation_v3: continue the run in --output-dir")
    result.add_argument("--output-dir", type=Path, help="A new directory (existing ones are refused unless --resume)")
    result.add_argument("--plan-only", action="store_true", help="Validate inputs and print the search space; no solves or files")
    return result


def dump(value):
    from Design.Grouped_Design import _json_safe
    return json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"


def engine_options(args):
    """Fill the chosen engine's defaults and refuse the other engine's options."""
    own, other = (V2_ONLY, V3_ONLY) if args.engine == "heuristic_v2" else (V3_ONLY, V2_ONLY)
    given = sorted(key for key in other if getattr(args, key, None) is not None)
    if given:
        raise ValueError(f"--{given[0].replace('_', '-')} does not apply to --engine {args.engine}")
    if getattr(args, "resume", False) and args.engine != "generation_v3":
        raise ValueError("--resume applies to --engine generation_v3")
    for key, default in own.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)
    if args.population_size is None:
        args.population_size = 8 if args.engine == "heuristic_v2" else 4
    if args.max_generations is None:
        args.max_generations = 10 if args.engine == "heuristic_v2" else 6


def generation_config(args):
    from Design.Genetic_Generation import GenerationConfig

    def option(key):
        value = getattr(args, key, None)
        return V3_ONLY[key] if value is None else value
    config = GenerationConfig(
        random_seed=args.random_seed, population_size=args.population_size,
        offspring_per_generation=(args.population_size if args.offspring_per_generation is None
                                  else args.offspring_per_generation),
        max_generations=args.max_generations, max_evaluations=args.max_evaluations, max_proposals=args.proposal_budget,
        max_wall_seconds=args.max_wall_seconds, initialization_policy=args.initialization,
        initialization_evaluations=args.initialization_evaluations, guided_probability=args.guided_probability,
        max_exploration_evaluations=args.max_exploration_evaluations,
        max_stagnant_generations=args.max_stagnant_generations, max_attempts_per_child=args.max_attempts_per_child,
        target_accepted_designs=args.target_accepted, target_distinct_profiles=args.target_distinct_profiles,
        diversity_view_size=args.diversity_view_size, max_evaluation_errors=args.evaluation_error_limit,
        guided_selection=option("guided_selection"), severity_floor=option("severity_floor"),
        severity_cap=option("severity_cap"), categorical_weight=option("categorical_weight"),
        beam_jump_policy=option("beam_jumps"), beam_jump_target_dcr=option("jump_target_dcr"),
        beam_jump_max_sections=option("jump_max_sections"), beam_jump_max_seconds=option("jump_max_seconds"),
        beam_jump_partial_tolerance=option("jump_partial_tolerance"), beam_jump_min_gain=option("jump_min_gain"),
        preferred_severe_utilization=option("preferred_severe_utilization"), preferred_share=option("preferred_share"))
    config.validate()
    return config


def prepare(args):
    from Design import Design_Driver as driver, Genetic_Search as ga, Grouped_Record as gr
    from Design.Config import BEAM_PRACTICAL_RATIO_AUTHORIZED_MAX, DesignConfig, with_beam_practical_ratio
    from Design.Screening_Manifest import load_manifest
    from Design.Verify_Designs import configure_case
    from Design.Verification_Integrity import file_sha256
    from Model import Member_Groups as mg

    if not hasattr(args, "engine"):
        args.engine = "heuristic_v2"
    engine_options(args)
    if args.manifest:
        if not args.case_id or args.profile:
            raise ValueError("With --manifest, supply --case-id and use the manifest's profile")
        manifest = load_manifest(args.manifest)
        matching = [c for c in manifest["cases"] if c["case_id"] == args.case_id]
        if len(matching) != 1:
            raise ValueError(f"Expected one manifest case matching {args.case_id!r}")
        case, profile = matching[0], manifest["profile_id"]
        source_path = args.manifest
    else:
        if args.case_id:
            raise ValueError("--case-id is used only with --manifest")
        case = json.loads(args.case_file.read_text(encoding="utf-8-sig"))
        profile, source_path = args.profile, args.case_file
    configure_case(case, profile)
    cfg = DesignConfig.from_structure_parameters()
    ceiling, overrides = getattr(args, "beam_rho_practical_max", None), None
    if ceiling is not None:
        # Validated (at most the authorized ceiling) and recorded before the evaluation identity is taken.
        overrides = {"rebar.rho_beam_practical_max": {"value": float(ceiling), "default": cfg.rebar.rho_beam_practical_max,
                                                      "authorized_max": BEAM_PRACTICAL_RATIO_AUTHORIZED_MAX}}
        cfg = with_beam_practical_ratio(cfg, ceiling)
    # The design-evidence profile (2026-11-01): validated, checked against this run's scope and applied before the
    # evaluation identity is taken; without one, the unreviewed defaults are kept and said so.
    from Design import Design_Evidence_Profile as evidence_profile
    if getattr(args, "design_evidence", None) is not None:
        profile_file = evidence_profile.load(args.design_evidence)
        problems = profile_file.scope_problems(case, file_sha256(source_path), profile, cfg)
        if problems:
            raise ValueError("the design-evidence profile does not cover this run: " + "; ".join(problems[:6]))
        cfg = profile_file.apply(cfg)
        design_evidence = profile_file.summary()
    else:
        design_evidence = evidence_profile.default_summary()
    grouping = mg.default_system_policy(case.get("seismic_site")) if args.grouping == "auto" else args.grouping
    state, seed = gr.seed_from_sections(args.seed_column, args.seed_beam, policy_id=grouping)
    space = ga.DesignSpace(state)
    if cfg.floor_analysis.slab_refinement is not None:
        # The profile's plan resolved on the seed's own floors (no solve): a plan the slab stage would refuse is refused
        # here, before any evaluation.
        from Design import SMRF_Floor_Sections as floor_sections
        from Design.SMRF_Slab_Refinement import _plan
        seed_plan = []
        with mg.installed(state):
            for sha, sections, served in floor_sections.distinct_floors(floor_sections.by_floor()):
                try:
                    grids, resolution = _plan(driver._slab_geometry(), cfg.floor_analysis.slab_refinement,
                                              sections=sections)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"the design-evidence profile's slab refinement plan does not resolve on the seed's "
                                     f"floors: {exc}") from exc
                if resolution is not None and resolution.get("status") != "resolved":
                    # Fewer than two affordable levels, or fewer than all four recipe levels when a local extension is
                    # declared: the slab stage would record every such floor unresolved, with no refined evidence, so
                    # the plan is refused before any evaluation.
                    raise ValueError(f"the design-evidence profile's slab refinement plan does not resolve on the seed's "
                                     f"floor set {sha[:12]} (floors {served}): {resolution.get('status')}, "
                                     f"{resolution.get('detail')}")
                seed_plan.append({"floor_sections_sha256": sha, "floors": served, "meshes": len(grids),
                                  "status": "explicit" if resolution is None else resolution["status"],
                                  "levels": None if resolution is None else [
                                      {k: level[k] for k in ("level", "shell_count", "cells_per_bay")}
                                      for level in resolution["resolved_levels"]]})
        design_evidence["seed_plan_resolution"] = seed_plan
    request = {
        "version": ga.VERSION, "experimental": True, "production_accepted": False, "engine": args.engine,
        "case": case, "profile": profile,
        "case_source": {"path": str(source_path.resolve()), "sha256": file_sha256(source_path)},
        "evaluation_identity": driver.design_request_identity(cfg),
        "design_evidence": design_evidence,
        "runner_sha256": file_sha256(Path(__file__)),
        "grouping": mg.policy_of(grouping), "seed": seed, "seed_state": state.to_record(),
    }
    if overrides is not None:
        request["design_config_overrides"] = overrides
    if args.engine == "heuristic_v2":
        policy = ga.GeneticPolicy(**{key: getattr(args, key) for key in asdict(ga.GeneticPolicy())})
        policy.validate()
        request["policy"] = asdict(policy)            # the existing request layout: policy, genes, probe flag
    request.update(genes=[asdict(gene) for gene in space.genes], probe_assertions=False)
    if args.engine == "generation_v3":
        from Design import Genetic_Generation as gen, Grouped_Design  # noqa: F401 -- loaded before the sources are hashed
        import Structure_Parameters as sp
        config = generation_config(args)
        policy = config
        request.update(version=gen.VERSION, generation_config=asdict(config),
                       basis=generation_basis(request, state, sp, config, cfg), loaded_sources_sha256=loaded_sources())
    request["sha256"] = hashlib.sha256(dump(request).encode("utf-8")).hexdigest()
    return state, cfg, policy, request


def loaded_sources():
    """{repository path: sha256} of every repository module loaded at this point, beyond the folders the evaluation
    identity hashes (Geometry_Overrides, Run_Naming, Data_Generation ...): a resume refuses any edited source."""
    from Design.Verification_Integrity import file_sha256
    rows = {}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path or not str(path).endswith(".py"):
            continue
        resolved = Path(path).resolve()
        try:
            name = resolved.relative_to(RC_DIR).as_posix()
        except ValueError:
            continue
        if not name.startswith("tests/"):
            rows[name] = file_sha256(resolved)
    return dict(sorted(rows.items()))


def generation_basis(request, state, sp, config=None, cfg=None):
    """The result-affecting evaluation basis of a generation_v3 run (its id keys the cache, registry and diagnostics):
    the evaluator identity (design inputs, configuration and source hashes), evaluation and contract versions, the case
    and profile, the grouping, the seed design and the longitudinal steel grade, the design-evidence profile (its id,
    kind, review status, whether it asserts and its identity sha256; none without one); with ``config``, the proposal
    policy (Genetic_Generation.proposal_policy: selection, beam jumps and the beam steel ratio ceiling they price with)."""
    from Design import Code_Check_Contract as contract, Grouped_Design as gd
    evidence = request.get("design_evidence") or {}
    basis = {"evaluation_identity_sha256": request["evaluation_identity"]["sha256"],
             "evaluation_version": gd.EVALUATION_VERSION, "contract_version": contract.CONTRACT_VERSION,
             "case": request["case"], "profile": request["profile"], "grouping": request["grouping"]["id"],
             "seed_state_identity": state.identity(), "longitudinal_fy_ksi": sp.FY_KSI,
             "design_evidence": {k: evidence.get(k) for k in ("schema", "profile_id", "kind", "review_status", "asserted",
                                                              "identity_sha256", "procedure_status",
                                                              "procedure_identity_sha256")}}
    if config is not None:
        from Design import Genetic_Generation as gen
        basis["proposal_policy"] = gen.proposal_policy(config, cfg)
    return basis


def run_v2(args, state, cfg, policy, request):
    from Design import Genetic_Search as ga
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "request.json").write_text(dump(request), encoding="utf-8")
    started = time.perf_counter()
    try:
        result = ga.search(state, cfg, policy=policy, log_path=out / "candidates.jsonl")
    except Exception as exc:
        result = {"version": ga.VERSION, "experimental": True, "production_accepted": False,
                  "status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        code = 1
    else:
        result["status"] = "search_complete"
        code = 0
    result.update(request_sha256=request["sha256"], elapsed_seconds=time.perf_counter() - started)
    (out / "result.json").write_text(dump(result), encoding="utf-8")
    print(dump({k: result[k] for k in ("status", "stop", "counts", "outcomes", "error") if k in result}), end="")
    print(f"Experimental evidence saved to {out}")
    return code


def directory_refusal(kind, out):
    """The runner's advice for a directory the controller refuses (Genetic_Generation.RunDirectoryError kinds)."""
    return {
        "has_request": f"{out} already holds this request's run; continue it with --resume (identical command) or "
                       "choose a new --output-dir",
        "has_other_request": f"{out} already holds the run of another request; choose a new --output-dir (a run "
                             "resumes only with identical inputs)",
        "foreign": f"{out} already exists and holds no saved request of this tool; choose a new --output-dir or empty it",
        "no_request": f"{out} holds no saved request, so there is no run to resume; start one without --resume in a "
                      "new or empty --output-dir",
        "request_unreadable": f"{out / 'request.json'} is not a complete request; inspect it (a run is never started "
                              "without its whole request)",
        "request_differs": "the saved request differs from this one (case, profile, sources, seed or configuration); "
                           "a run resumes only with identical inputs",
    }[kind]


def run_v3(args, state, cfg, config, request):
    """Exit codes: 0 the search stopped (any stop reason; see result.json), 1 the controller raised (error.json) or
    releasing the lease failed after the run finished (result.json holds the run, no error.json is written),
    EXIT_ERROR_BOUND the search stopped at its evaluation-error bound (the first error is printed).

    The directory checks run twice: here first, so a directory this check refuses gets nothing (not even the lease
    marker), then by the controller under the lease, which decides: the request of a new run is saved only by the
    process that owns the directory, so two launches cannot both claim it."""
    from Design import Genetic_Generation as gen
    out = args.output_dir.resolve()
    # Built (and so validated) before any directory or file exists: a refused configuration leaves nothing behind.
    controller = gen.GenerationRun(state, cfg, config, out, case_id=request["case"].get("case_id"),
                                   basis=request["basis"])

    def advice(kind):
        # A saved request is this one (resume it), another (a new directory) or not a whole request (inspect it): a
        # launch that lost the claim to another request is not told to resume.
        if kind == "has_request":
            saved = gen.request_refusal(out, request["sha256"])
            kind = ("request_unreadable" if saved == "request_unreadable" else
                    "has_other_request" if saved is not None else kind)
        return directory_refusal(kind, out)
    kind = gen.request_refusal(out, request["sha256"]) if args.resume else gen.new_run_refusal(out)
    if kind is not None:
        raise ValueError(advice(kind))
    try:
        if args.resume:
            # A directory whose request was saved but which never reached its first checkpoint starts its run.
            controller.resume(request_sha256=request["sha256"])
        else:
            controller.start(request_text=dump(request))
    except gen.RunDirectoryError as exc:
        raise ValueError(advice(exc.kind)) from exc
    try:
        result = controller.run()
    except Exception as exc:
        # The controller recorded the exception in error.json under its lease (a failure to release the lease after
        # the run finished is reported here only; result.json then holds the finished run).
        print(dump({"status": "error", "error": f"{type(exc).__name__}: {gen.exception_text(exc)}"}), end="")
        return 1
    summary = {"status": "search_stopped", "stop": result["stop"], "counts": {
        k: result["counts"][k] for k in ("evaluations", "proposals", "cache_hits", "noop", "outcomes",
                                         "evaluation_errors")}, "metrics": result["metrics"]}
    code = 0
    if result["stop"]["reason"] == "evaluation_error_bound_reached":
        first = next((r for r in result["views"]["analysis_recovery_queue"] if r["kind"] == "evaluation_exception"), None)
        summary["first_evaluation_error"] = None if first is None else {"id": first["id"], "problems": first["problems"]}
        code = EXIT_ERROR_BOUND
    print(dump(summary), end="")
    print(f"Experimental evidence saved to {out}")
    return code


def run(args):
    state, cfg, policy, request = prepare(args)
    if args.plan_only:
        print(dump(request), end="")
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required for an experimental run")
    if args.engine == "heuristic_v2":
        return run_v2(args, state, cfg, policy, request)
    return run_v3(args, state, cfg, policy, request)


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        arguments.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
