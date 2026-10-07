"""Freeze an explicitly scoped, portable weighted-intensity diagnostic plan.

Plan construction and preflight never build a structural model or run NTHA.
Input design roots and output root must be inside RC Structure, copied with the
same relative layout on each device. One master plan assigns disjoint jobs to
1--4 devices. Each device runs Run_Weighted_Seismic_Calibration.py --shard N.

Scales are RAW common X/Y acceleration multipliers, not spectral alpha. A
passing weighted score does not close M1, qualify hinges, or release training.
The default refuses all unqualified designs. --allow-open-m1-diagnostic permits
only the single existing M1 check to remain open and preserves that fact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
from Data_Generation import Run_Weighted_Seismic_Calibration as runner
from Data_Generation.Seismic_Response_Score import default_policy


def collect_cases(roots, case_ids, profile, use_mode, expected_sources):
    roots = [runner.resolve(p) for p in roots]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate design roots")
    for root in roots:
        runner.portable_path(root)
        if not root.is_dir():
            raise ValueError(f"Design root does not exist: {root}")
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Select a nonempty unique case roster")
    cases = []
    for case_id in sorted(case_ids):
        if not case_id.startswith("case_") or Path(case_id).name != case_id or any(c in case_id for c in ("/", "\\", ":")):
            raise ValueError(f"Invalid case directory name: {case_id}")
        matches = [root for root in roots if (root / case_id).is_dir()]
        if len(matches) != 1:
            raise ValueError(f"{case_id}: need exactly one design directory; found {len(matches)}")
        root = matches[0]
        result = runner.read_json(root / case_id / "result.json")
        record = runner.read_json(root / case_id / "design.json")
        case = {"case_id": case_id, "design_root": runner.portable_path(root), "profile": profile,
                "design_sha256": runner.digest(root / case_id / "design.json"),
                "result_sha256": runner.digest(root / case_id / "result.json"),
                "original_qualification": {"result_accepted": result.get("accepted"),
                    "qualification_accepted": (record.get("qualification") or {}).get("accepted"),
                    "counts": result.get("counts") or {}, "fail_ids": result.get("fail_ids", []),
                    "not_evaluated_ids": result.get("not_evaluated_ids", [])},
                "original_probe_assertions": result.get("probe_assertions"), "training_release": False}
        runner.validate_case(case, use_mode=use_mode, expected_sources=expected_sources)
        cases.append(case)
    return cases


def assign_jobs(cases, pair_ids, *, seed, shards, assignment):
    if not pair_ids or len(set(pair_ids)) != len(pair_ids) or any(type(p) is not int or p <= 0 for p in pair_ids):
        raise ValueError("Explicit pair pool must contain unique positive integer IDs")
    if type(shards) is not int or not 1 <= shards <= 4 or shards > len(cases):
        raise ValueError("Use 1 to 4 nonempty device shards")
    if assignment not in ("one-per-case", "cross-product"):
        raise ValueError("Declare one-per-case or cross-product assignment")
    order = sorted(pair_ids)
    random.Random(f"{seed}:records").shuffle(order)
    jobs = []
    for index, case in enumerate(sorted(cases, key=lambda c: c["case_id"])):
        selected = [order[index % len(order)]] if assignment == "one-per-case" else order
        for result_id in selected:
            jobs.append({"job_id": f"{case['case_id']}_pair_{result_id}", "case_id": case["case_id"],
                         "result_id": result_id, "shard": index % shards + 1})
    return jobs


def build_plan(args):
    from Design import Design_Driver as driver
    roots = [runner.resolve(p) for p in args.design_root]
    if args.all_cases_in_roots:
        case_ids = sorted({p.name for root in roots for p in root.glob("case_*") if p.is_dir()})
    elif args.case_manifest:
        manifest = runner.read_json(runner.resolve(args.case_manifest))
        case_ids = [c["case_id"] for c in manifest["cases"]]
    else:
        case_ids = args.cases
    use_mode = "open-m1-diagnostic" if args.allow_open_m1_diagnostic else "qualified-only"
    cases = collect_cases(roots, case_ids, args.profile, use_mode, driver.source_sha256())
    if args.all_pairs_in_set:
        from Loads.Ground_Motion import ground_motion_pair_rows
        pairs = []
        for key, x, y in ground_motion_pair_rows(set_name=args.set_name):
            if y is not None and max(int(x["npts"]), int(y["npts"])) <= args.max_motion_points:
                pairs.append(int(key.split(":")[-1]))
    else:
        pairs = args.pairs
    jobs = assign_jobs(cases, pairs, seed=args.seed, shards=args.shards, assignment=args.assignment)
    plan = {"schema_version": runner.PLAN_SCHEMA, "created_utc": runner.now(),
            "scope": "full-record weighted intensity diagnostic; no M1 assertion or training release",
            "use_mode": use_mode, "training_release": False, "assignment": args.assignment,
            "seed": args.seed, "shards": args.shards, "cases": cases, "jobs": jobs,
            "result_ids": sorted(pairs), "set_name": args.set_name,
            "pair_pool_selection": "all paired usable set entries within declared point bound" if args.all_pairs_in_set else "explicit pair IDs",
            "maximum_motion_points": args.max_motion_points,
            "output_root": runner.portable_path(runner.resolve(args.output_root)),
            "workers": args.workers, "trial_timeout_seconds": args.timeout_seconds,
            "score_policy": default_policy(), "recorder_tolerances": runner.RECORDER_TOLERANCES,
            "scale_basis": "raw common X/Y acceleration multiplier on scale_factor=1 loaded records; not spectral alpha",
            "search": {"minimum_scale": args.minimum_scale, "initial_scale": args.initial_scale,
                "maximum_scale": args.maximum_scale, "maximum_trials_per_pair": args.max_trials,
                "relative_bracket_tolerance": args.relative_bracket_tolerance}}
    if args.case_manifest:
        roster_path = runner.resolve(args.case_manifest)
        plan["case_roster"] = {"path": runner.portable_path(roster_path), "sha256": runner.digest(roster_path)}
    motions = runner.motion_identity(plan)
    if any(c["npts"] > args.max_motion_points for m in motions.values() for c in m["components"].values()):
        raise ValueError("Selected pair exceeds declared maximum_motion_points; select another pool or declare a different bound")
    plan["frozen"] = {"sources": runner.source_identity(), "runtime": runner.runtime_identity(),
                      "motions": motions, "input_sha256": runner.capture_inputs(plan, motions)}
    plan["plan_sha256"] = runner.canonical_digest(plan)
    return runner.validate_plan(plan)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-root", action="append", required=True)
    roster = parser.add_mutually_exclusive_group(required=True)
    roster.add_argument("--cases", nargs="+")
    roster.add_argument("--all-cases-in-roots", action="store_true")
    roster.add_argument("--case-manifest", help="JSON manifest with an explicit cases list; missing/failed cases are never silently dropped")
    parser.add_argument("--profile", default=runner.RESEARCH_PROFILE)
    parser.add_argument("--allow-open-m1-diagnostic", action="store_true")
    parser.add_argument("--set-name", required=True)
    pool = parser.add_mutually_exclusive_group(required=True)
    pool.add_argument("--pairs", nargs="+", type=int)
    pool.add_argument("--all-pairs-in-set", action="store_true")
    parser.add_argument("--max-motion-points", type=int, default=15000)
    parser.add_argument("--assignment", choices=("one-per-case", "cross-product"), required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--minimum-scale", type=float, default=1.0)
    parser.add_argument("--initial-scale", type=float, default=1.0)
    parser.add_argument("--maximum-scale", type=float, default=3.5)
    parser.add_argument("--max-trials", type=int, default=7)
    parser.add_argument("--relative-bracket-tolerance", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = runner.resolve(args.output)
    if output.exists():
        raise FileExistsError("Choose a new plan filename; frozen plans are never overwritten")
    plan = build_plan(args)
    # Capture after all validation, then verify again against any concurrent edits.
    runner.verify_frozen_plan(plan)
    runner.write_json(output, plan)
    print(f"Plan only: {len(plan['cases'])} cases, {len(plan['jobs'])} explicit paired-record searches, "
          f"{plan['shards']} device shards, at most {len(plan['jobs']) * args.max_trials} full-record trials. "
          f"No NTHA launched. Plan SHA256 {plan['plan_sha256']}. Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
