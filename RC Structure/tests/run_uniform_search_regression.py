"""Run the pilot's feasibility regression, separate from the expensive reduction experiment.

Each case gets a fresh interpreter and the ordinary verification worker. The
explicit reduction budget is recorded in the design's configuration identity.
Original manifests and run folders are read only. The initial case runs alone.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reduction-trials", type=int, default=0)
    parser.add_argument("--worker-case")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    from Design import Verify_Designs as verify
    from Design.Screening_Manifest import load_manifest
    manifest = load_manifest(args.manifest)
    cases = manifest["cases"]
    profile = manifest["profile_id"]
    probe_date = "2026-10-02"
    metadata = {"mode": "manifest", "manifest_path": manifest["path"],
                "manifest_sha256": manifest["sha256"], "manifest_schema": manifest["schema"],
                "design_basis": manifest["design_basis"], "population_basis": manifest["population_basis"],
                "sampling": manifest["sampling"],
                "scope": f"Feasibility regression; reduction trial budget {args.reduction_trials}. " + manifest["scope"]}
    original_config = verify.probe_config

    def regression_config(date=None):
        cfg = original_config(date)
        cfg.iteration.uniform_reduction_trials = args.reduction_trials
        return cfg

    verify.probe_config = regression_config
    if args.summarize_only:
        plan = json.loads((args.root / "plan.json").read_text(encoding="utf-8"))
        if plan["uniform_reduction_trials"] != args.reduction_trials:
            raise ValueError("Summary reduction budget must match the saved plan")
        plan.update(metadata)
        (args.root / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        results = [json.loads((args.root / c["case_id"] / "result.json").read_text(encoding="utf-8")) for c in cases]
        verify.summarize(results, args.root, True, cases, probe_date, profile, plan)
        return
    if args.worker_case:
        case = next(c for c in cases if c["case_id"] == args.worker_case)
        verify.run_worker(case, args.root / case["case_id"], True, probe_date,
                          max_section_iter=manifest["max_section_iter"], profile_id=profile,
                          stages=("design", "gravity_modal"))
        return
    if args.root.exists():
        raise ValueError("Regression root must be new; prior results are preserved")
    args.root.mkdir(parents=True)
    plan = {**metadata, "cases": cases, "profile_id": profile, "probe_assertions": True, "probe_date": probe_date,
            "max_section_iter": manifest["max_section_iter"], "uniform_reduction_trials": args.reduction_trials,
            "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "basis": "feasibility regression; the separate case_0004 experiment validates 24 reduction trials"}
    (args.root / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def run(case):
        command = [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
                   "--manifest", str(args.manifest.resolve()), "--root", str(args.root.resolve()),
                   "--reduction-trials", str(args.reduction_trials), "--worker-case", case["case_id"]]
        with (args.root / (case["case_id"] + "_worker.log")).open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=RC, stdout=log, stderr=subprocess.STDOUT)
        path = args.root / case["case_id"] / "result.json"
        if not path.exists():
            raise RuntimeError(f"{case['case_id']} worker exited {completed.returncode} without evidence")
        result = json.loads(path.read_text(encoding="utf-8"))
        print(case["case_id"], result["status"], result.get("counts"), result.get("error"), flush=True)
        return result

    initial_id = manifest.get("initial_case_id", "case_0001")
    initial = next(c for c in cases if c["case_id"] == initial_id)
    results = [run(initial)]
    with ThreadPoolExecutor(max_workers=manifest.get("subsequent_max_workers", 4)) as pool:
        results.extend(pool.map(run, [c for c in cases if c != initial]))
    verify.summarize(results, args.root, True, cases, probe_date, profile, plan)


if __name__ == "__main__":
    main()
