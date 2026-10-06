"""Verify and summarize copied school-device intensity searches without rerunning them.

Keep complete device_01 ... device_04 folders. Paths in their artifact inventories
are relative to each device folder, so copies can be collected on another drive.
Incomplete collections are reported explicitly and exit nonzero unless
--allow-partial is requested for an interim review. Nothing here releases training.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path, PurePosixPath
import sys

RC = Path(__file__).resolve().parents[1]
if str(RC) not in sys.path:
    sys.path.insert(0, str(RC))

from Data_Generation.Run_Weighted_Seismic_Calibration import (
    assess_trial, canonical_digest, digest, next_scale, read_json, select_best, validate_plan, write_json,
)
from Data_Generation.Seismic_Response_Score import METRIC_NAMES, score_response


def artifact_path(root, label):
    """Inventories may refer only to ordinary files inside their device folder."""
    if not isinstance(label, str) or "\\" in label or ":" in label:
        raise ValueError(f"Invalid artifact path: {label!r}")
    relative = PurePosixPath(label)
    if relative.is_absolute() or not relative.parts or any(p in (".", "..") for p in relative.parts):
        raise ValueError(f"Invalid artifact path: {label!r}")
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes device directory: {label}")
    return target


def verify_search(root, job, progress, policy, search_settings):
    job_id = job["job_id"]
    search_path = root / job_id / "search.json"
    expected = progress.get("search_manifest_sha256", {}).get(f"{job_id}/search.json")
    if not expected or digest(search_path) != expected:
        raise ValueError(f"Missing or changed search manifest: {job_id}")
    search = read_json(search_path)
    for key in ("job_id", "case_id", "result_id", "shard"):
        if search.get(key) != job[key]:
            raise ValueError(f"Search identity mismatch {job_id}: {key}")
    if search.get("plan_sha256") != progress["plan_sha256"]:
        raise ValueError(f"Foreign plan in {job_id}")
    if search.get("status") != "completed":
        raise ValueError(f"Unfinished search supplied as complete: {job_id}")
    copies = [s for s in progress.get("searches", []) if s.get("job_id") == job_id]
    if len(copies) != 1 or canonical_digest(copies[0]) != canonical_digest(search):
        raise ValueError(f"Progress/search disagreement: {job_id}")
    inventory = search.get("artifact_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError(f"Missing artifact inventory: {job_id}")
    actual_files = {
        p.relative_to(root).as_posix() for p in (root / job_id).rglob("*")
        if p.is_file() and p != search_path
    }
    if set(inventory) != actual_files:
        raise ValueError(f"Missing or unexpected artifacts: {job_id}")
    for label, sha in inventory.items():
        if PurePosixPath(label).parts[0] != job_id:
            raise ValueError(f"Artifact belongs to another search: {label}")
        if digest(artifact_path(root, label)) != sha:
            raise ValueError(f"Artifact changed: {label}")
    trials = search.get("trials", [])
    if not trials:
        raise ValueError(f"Completed search has no trials: {job_id}")
    for i, trial in enumerate(trials, 1):
        if (type(trial.get("trial")) is not int or trial["trial"] != i
                or type(trial.get("scale")) not in (int, float)
                or not math.isfinite(trial["scale"]) or trial["scale"] <= 0
                or type(trial.get("usable")) is not bool):
            raise ValueError(f"Invalid trial ordering or scale: {job_id}")
        expected_scale = next_scale(trials[:i - 1], search_settings)
        if expected_scale is None or not math.isclose(trial["scale"], expected_scale, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Trial does not follow the planned search: {job_id} trial {i}")
        trial_prefix = f"{job_id}/trial_{i:02d}"
        score_path = artifact_path(root, f"{trial_prefix}/score.json")
        if canonical_digest(read_json(score_path)) != canonical_digest(trial):
            raise ValueError(f"Trial score file differs from search: {job_id} trial {i}")
        command = read_json(artifact_path(root, f"{trial_prefix}/command.json"))
        if (command.get("job_id") != job_id or command.get("plan_sha256") != progress["plan_sha256"]
                or command.get("scale") != trial["scale"]):
            raise ValueError(f"Trial command identity mismatch: {job_id} trial {i}")
        expected_manifest = f"{trial_prefix}/{job['case_id']}/peer_{job['result_id']}_scale_{trial['scale']:g}/manifest.json"
        if trial.get("manifest") != expected_manifest:
            raise ValueError(f"Trial manifest belongs to another run: {job_id} trial {i}")
        if trial.get("usable"):
            if trial.get("returncode") != 0 or trial.get("interruption") is not None:
                raise ValueError(f"Unsuccessful worker marked usable: {job_id} trial {i}")
            assessment = assess_trial(artifact_path(root, expected_manifest), policy)
            if assessment.get("usable") is not True or any(
                    canonical_digest(value) != canonical_digest(trial.get(key))
                    for key, value in assessment.items()):
                raise ValueError(f"Trial assessment does not reproduce from artifacts: {job_id} trial {i}")
            recomputed = score_response(trial["metrics"], policy)
            if canonical_digest(recomputed) != canonical_digest(trial.get("score")):
                raise ValueError(f"Score does not reproduce: {job_id} trial {i}")
        elif not isinstance(trial.get("rejection_reasons"), list) or not trial["rejection_reasons"] or trial.get("score") is not None:
            raise ValueError(f"Unusable trial lacks rejection evidence or carries a score: {job_id} trial {i}")
    if next_scale(trials, search_settings) is not None:
        raise ValueError(f"Search marked complete before its planned stopping condition: {job_id}")
    expected_stop = "unusable_trial" if not trials[-1]["usable"] else "budget_or_scale_bound_or_bracket_tolerance"
    if search.get("stop_reason") != expected_stop:
        raise ValueError(f"Search stopping reason differs from trials: {job_id}")
    best = select_best(trials)
    if canonical_digest(best) != canonical_digest(search.get("best_trial")):
        raise ValueError(f"Selected trial is not the lowest tested passing scale: {job_id}")
    if search.get("target_reached") is not (best is not None):
        raise ValueError(f"Target flag differs from trial evidence: {job_id}")
    if search.get("best_tested_passing_scale") != (best["scale"] if best else None):
        raise ValueError(f"Selected scale differs from trial evidence: {job_id}")
    return search


def collect(plan, roots):
    validate_plan(plan)
    jobs = {j["job_id"]: j for j in plan["jobs"]}
    seen_shards, verified = set(), {}
    devices, errors = [], []
    for raw_root in roots:
        root = Path(raw_root).resolve()
        try:
            identity = read_json(root / "identity.json")
            shard = identity["shard"]
            if shard in seen_shards:
                raise ValueError(f"Duplicate shard {shard}")
            if not 1 <= shard <= plan["shards"]:
                raise ValueError(f"Unexpected shard {shard}")
            seen_shards.add(shard)
            if identity.get("plan_sha256") != plan["plan_sha256"] or identity.get("frozen") != plan["frozen"]:
                raise ValueError(f"Foreign plan or source/input identity in device {shard}")
            progress = read_json(root / "progress.json")
            if progress.get("plan_sha256") != plan["plan_sha256"] or progress.get("shard") != shard:
                raise ValueError(f"Progress identity mismatch in device {shard}")
            assigned = {k: j for k, j in jobs.items() if j["shard"] == shard}
            planned_jobs = progress.get("planned_jobs", [])
            if len(planned_jobs) != len(set(planned_jobs)) or set(planned_jobs) != set(assigned):
                raise ValueError(f"Unexpected planned jobs in device {shard}")
            summaries = progress.get("searches", [])
            delivered = [s.get("job_id") for s in summaries]
            if len(delivered) != len(set(delivered)) or not set(delivered) <= set(assigned):
                raise ValueError(f"Duplicate or foreign searches in device {shard}")
            foreign_dirs = [p.name for p in root.iterdir() if p.is_dir() and p.name not in assigned]
            if foreign_dirs:
                raise ValueError(f"Unexpected search folders in device {shard}: {foreign_dirs}")
            device_verified = {}
            for job_id in delivered:
                device_verified[job_id] = verify_search(root, assigned[job_id], progress, plan["score_policy"], plan["search"])
            verified.update(device_verified)
            devices.append({"shard": shard, "path": str(root), "status": progress.get("status"),
                            "verified_searches": len(delivered), "worker_errors": progress.get("errors", [])})
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append({"device_root": str(root), "error": f"{type(exc).__name__}: {exc}"})
    rows = []
    for job_id, job in jobs.items():
        search = verified.get(job_id)
        best = search.get("best_trial") if search else None
        row = dict(job)
        row.update(status="missing" if search is None else "target_reached" if best else
                   "unusable" if any(not t.get("usable") for t in search["trials"]) else "target_not_reached",
                   trials=len(search["trials"]) if search else 0,
                   best_tested_passing_scale=best["scale"] if best else None,
                   score=best["score"]["score"] if best else None,
                   stop_reason=search.get("stop_reason") if search else None,
                   training_eligible=False)
        row.update({name: best["metrics"][name] if best else None for name in METRIC_NAMES})
        rows.append(row)
    missing = sorted(set(jobs) - set(verified))
    all_devices_finished = (len(devices) == plan["shards"] and
                            all(d["status"] in ("completed", "completed_with_errors") for d in devices))
    complete = not errors and not missing and all_devices_finished
    reaching_target = sum(r["status"] == "target_reached" for r in rows)
    return {"schema_version": "weighted_seismic_calibration_collection_v1",
            "plan_sha256": plan["plan_sha256"], "scope": "verified intensity-selection diagnostics; not training release",
            "training_eligible": False, "complete": complete,
            "completion_meaning": "all planned search evidence was collected and verified; this does not mean all searches reached target",
            "all_searches_reached_target": complete and reaching_target == len(jobs),
            "devices": devices, "errors": errors, "missing_jobs": missing, "planned_searches": len(jobs),
            "verified_searches": len(verified), "searches_reaching_target": reaching_target,
            "searches_with_unusable_trials": sum(any(not t["usable"] for t in s["trials"]) for s in verified.values()),
            "searches_without_passing_trial": sum(r["status"] in ("unusable", "target_not_reached") for r in rows),
            "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--device-root", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Fresh collection directory; repeated reviews require the same plan")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    plan = read_json(args.plan)
    summary = collect(plan, args.device_root)
    target = args.output / "calibration_summary.json"
    if target.exists() and read_json(target).get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Output contains a different plan's collection; use a fresh directory")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(target, summary)
    with (args.output / "calibration_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    print(f"Verified {summary['verified_searches']}/{summary['planned_searches']} searches; "
          f"{summary['searches_reaching_target']} reached the score target; complete={summary['complete']}")
    for error in summary["errors"]:
        print(error["error"])
    return 1 if summary["errors"] else 0 if summary["complete"] or args.allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
