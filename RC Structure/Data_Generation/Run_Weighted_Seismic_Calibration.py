"""Bounded, full-record intensity searches on explicitly saved designs.

This is a diagnostic controller, not a design-qualification or training release.
Each trial uses Fixed_Design_Diagnostics and the production damage definitions.
The weighted objective is separate from the existing is_inelastic label.

    python Data_Generation/Run_Weighted_Seismic_Calibration.py --plan plan.json --shard 1

The smallest passing scale *tested* is reported; nonlinear response need not
be monotonic. An unsuccessful solve never supplies a passing score. Resume
requires exactly the same plan, source files, designs, and motion files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

import numpy as np

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
from Data_Generation.Seismic_Response_Score import METRIC_NAMES, score_response, validate_policy

PLAN_SCHEMA = "weighted_seismic_calibration_plan_v2"
_PRINT_LOCK = threading.Lock()
_STOP = threading.Event()
# The producer writes full-rate stressStrain at 12 significant digits. These
# tolerances compare its float64 exports, not the compact float32 training copy.
RECORDER_TOLERANCES = {"relative": 1e-9, "moment_absolute_kip_in": 1e-7,
                       "rotation_absolute_rad": 1e-11, "time_absolute_sec": 1e-7}
RESEARCH_PROFILE = "v2_nonlinear_flexure_anchored_energy_research_v1"
M1_CHECK = "demands.torsional_irregularity"


def now():
    return datetime.now(timezone.utc).isoformat()


def announce(message):
    with _PRINT_LOCK:
        print(f"[{now()}] {message}", flush=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (RC / value).resolve()


def source_identity():
    files = list(RC.glob("*.py"))
    for folder in ("Model", "Design", "Analysis", "Loads", "Data_Generation", "tools"):
        files.extend((RC / folder).rglob("*.py"))
    return {p.relative_to(RC).as_posix(): hashlib.sha256(
        p.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest() for p in sorted(set(files))}


def portable_path(path):
    """Plans travel with an RC Structure checkout; do not freeze drive letters."""
    path = Path(path).resolve()
    try:
        return path.relative_to(RC).as_posix()
    except ValueError as exc:
        raise ValueError(f"Copy inputs inside RC Structure before freezing a portable plan: {path}") from exc


def runtime_identity():
    from tools.source_fingerprint import fingerprint
    value = fingerprint()  # imports/version queries only; no domain build or solve
    return {k: value[k] for k in ("python", "opensees_version", "packages")} | {
        "system": platform.system(), "machine": platform.machine()}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_case(case, *, use_mode, expected_sources):
    directory = resolve(case["design_root"]) / case["case_id"]
    design_path, result_path = directory / "design.json", directory / "result.json"
    if digest(design_path) != case["design_sha256"] or digest(result_path) != case["result_sha256"]:
        raise ValueError(f"Saved design/result changed: {case['case_id']}")
    record, result = read_json(design_path), read_json(result_path)
    if result.get("status") != "designed" or result.get("error") or result.get("fail_ids"):
        raise ValueError(f"Failed/error design cannot be calibrated: {case['case_id']}")
    gravity = (result.get("stage_results") or {}).get("gravity_modal") or {}
    if (gravity.get("status") != "completed" or gravity.get("all_checks_pass") is not True
            or gravity.get("profile_id") != case["profile"]):
        raise ValueError(f"Completed passing gravity/modal verification is required: {case['case_id']}")
    counts = result.get("counts") or {}
    if counts.get("fail") != 0:
        raise ValueError(f"Missing or failing design counts: {case['case_id']}")
    if result.get("design_sha256") != case["design_sha256"]:
        raise ValueError(f"Result does not identify this design: {case['case_id']}")
    request = record.get("request_identity") or {}
    if request.get("source_sha256") != expected_sources:
        raise ValueError(f"Stale design source identity: {case['case_id']}")
    profile = (request.get("policy") or {}).get("model_profile", {}).get("id")
    if case["profile"] != profile or result.get("profile_id") != profile:
        raise ValueError(f"Profile mismatch: {case['case_id']}")
    qualification = record.get("qualification") or {}
    checks = qualification.get("checks") or []
    if (not checks or qualification.get("duplicate_check_ids") or qualification.get("malformed_checks")
            or any(c.get("status") not in ("pass", "not_evaluated") for c in checks)):
        raise ValueError(f"Missing/failed qualification checks: {case['case_id']}")
    open_ids = [c["id"] for c in checks if c["status"] == "not_evaluated"]
    if sorted(open_ids) != sorted(result.get("not_evaluated_ids") or []):
        raise ValueError(f"Qualification/result open checks disagree: {case['case_id']}")
    if counts != qualification.get("counts") or counts.get("not_evaluated") != len(open_ids):
        raise ValueError(f"Qualification/result counts disagree: {case['case_id']}")
    if (counts.get("pass") != sum(c["status"] == "pass" for c in checks)
            or type(result.get("accepted")) is not bool or result.get("accepted") != qualification.get("accepted")):
        raise ValueError(f"Qualification acceptance/counts are inconsistent: {case['case_id']}")
    accepted = result.get("accepted") is True and qualification.get("accepted") is True
    if use_mode == "qualified-only" and not accepted:
        raise ValueError(f"Design is not qualified; do not infer acceptance from status=designed: {case['case_id']}")
    if not accepted and (use_mode != "open-m1-diagnostic" or open_ids != [M1_CHECK]):
        raise ValueError(f"Only an explicitly declared single open M1 diagnostic is allowed: {case['case_id']}")
    if accepted and open_ids:
        raise ValueError(f"Accepted design still has open checks: {case['case_id']}")
    if case["profile"] != RESEARCH_PROFILE:
        raise ValueError("This controller requires the anchored-energy research profile")
    original = {"result_accepted": result.get("accepted"), "qualification_accepted": qualification.get("accepted"),
                "counts": counts, "fail_ids": result.get("fail_ids", []),
                "not_evaluated_ids": result.get("not_evaluated_ids", [])}
    if case.get("original_qualification") != original:
        raise ValueError(f"Original qualification evidence mismatch: {case['case_id']}")
    return record, result


def validate_plan(plan):
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("Unsupported weighted calibration plan schema")
    validate_policy(plan["score_policy"])
    if plan.get("use_mode") not in ("qualified-only", "open-m1-diagnostic"):
        raise ValueError("Declare qualified-only or open-m1-diagnostic use")
    if plan.get("recorder_tolerances") != RECORDER_TOLERANCES:
        raise ValueError("Recorder tolerance policy must match this controller")
    search = plan["search"]
    lo, start, hi = (float(search[k]) for k in ("minimum_scale", "initial_scale", "maximum_scale"))
    if not all(math.isfinite(v) for v in (lo, start, hi)) or not 0 < lo <= start <= hi:
        raise ValueError("Require finite 0 < minimum <= initial <= maximum scale")
    if type(search["maximum_trials_per_pair"]) is not int or not 1 <= search["maximum_trials_per_pair"] <= 30:
        raise ValueError("maximum_trials_per_pair must be an integer from 1 to 30")
    if not 0 < float(search["relative_bracket_tolerance"]) < 1:
        raise ValueError("relative_bracket_tolerance must be in (0,1)")
    if not math.isfinite(float(plan["trial_timeout_seconds"])) or plan["trial_timeout_seconds"] <= 0:
        raise ValueError("A positive finite trial timeout is required")
    if type(plan["workers"]) is not int or not 1 <= plan["workers"] <= 4:
        raise ValueError("workers must be an integer from 1 to 4")
    cases = plan["cases"]
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Need unique nonempty cases")
    pairs = plan["result_ids"]
    if not pairs or len(set(pairs)) != len(pairs) or any(type(p) is not int or p < 1 for p in pairs):
        raise ValueError("Need unique positive integer result_ids")
    for case in cases:
        if Path(case["case_id"]).name != case["case_id"] or not case["case_id"].startswith("case_"):
            raise ValueError("case_id must be a case directory name")
        portable_path(resolve(case["design_root"]))
    portable_path(resolve(plan["output_root"]))
    shards = plan["shards"]
    if type(shards) is not int or not 1 <= shards <= 4:
        raise ValueError("shards must be 1 to 4")
    jobs = plan["jobs"]
    names = {c["case_id"] for c in cases}
    seen = set()
    for job in jobs:
        key = (job["case_id"], job["result_id"])
        if (key in seen or key[0] not in names or key[1] not in pairs or type(job["shard"]) is not int
                or not 1 <= job["shard"] <= shards or job.get("job_id") != f"{key[0]}_pair_{key[1]}"):
            raise ValueError("Invalid/duplicate job or shard")
        seen.add(key)
    if not jobs or {j["case_id"] for j in jobs} != names:
        raise ValueError("Every selected case needs an explicit job")
    if plan.get("assignment") == "one-per-case" and len(jobs) != len(cases):
        raise ValueError("one-per-case requires exactly one paired record per case")
    if plan.get("assignment") not in ("one-per-case", "cross-product"):
        raise ValueError("Unsupported job assignment")
    if plan.get("plan_sha256") != canonical_digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("Plan identity changed; rebuild the frozen plan")
    return plan


def motion_identity(plan):
    # Use the same manifest/set loader as the analysis; no import of search-export fixtures.
    from Loads.Ground_Motion import load_ground_motion_pairs, find_manifest_row
    identities = {}
    loaded = {key: (x, y) for key, x, y in load_ground_motion_pairs(set_name=plan["set_name"], scale_factor=1.0)}
    for result_id in plan["result_ids"]:
        key = f"peer_result_id:{result_id}"
        x, y = loaded[key]
        if y is None:
            raise ValueError(f"A two-component pair is required: {result_id}")
        components = {}
        for direction, record in (("x", x), ("y", y)):
            row = find_manifest_row(record.record_id)
            path = Path(record.source_path)
            if not np.isfinite(record.acceleration).all() or record.npts < 2 or record.dt_sec <= 0:
                raise ValueError(f"Invalid acceleration series: {record.record_id}")
            components[direction] = {"record_id": record.record_id, "path": portable_path(path),
                                     "sha256": digest(path), "dt_sec": record.dt_sec, "npts": record.npts,
                                     "duration_sec": record.duration_sec, "event": row.get("event_name"),
                                     "component": row.get("component"), "unscaled_pga_g": record.pga_g}
        identities[str(result_id)] = {"key": key, "components": components}
    return identities


def capture_inputs(plan, motions):
    paths = [RC / "Ground_Motions" / "metadata" / name for name in ("record_manifest.csv", "record_sets.csv")]
    paths.extend(resolve(c["design_root"]) / c["case_id"] / name
                 for c in plan["cases"] for name in ("design.json", "result.json"))
    paths.extend(resolve(c["path"]) for m in motions.values() for c in m["components"].values())
    if plan.get("case_roster"):
        paths.append(resolve(plan["case_roster"]["path"]))
    return {portable_path(p): digest(p) for p in paths}


def verify_frozen_plan(plan):
    from Design import Design_Driver as driver
    frozen = plan["frozen"]
    if source_identity() != frozen["sources"] or runtime_identity() != frozen["runtime"]:
        raise ValueError("Source/runtime differs from plan creation; no analysis launched")
    design_sources = driver.source_sha256()
    for case in plan["cases"]:
        validate_case(case, use_mode=plan["use_mode"], expected_sources=design_sources)
    motions = motion_identity(plan)
    if motions != frozen["motions"] or capture_inputs(plan, motions) != frozen["input_sha256"]:
        raise ValueError("Input/motion identity differs from plan creation")
    return frozen


def inputs_unchanged(sources, inputs):
    return source_identity() == sources and all(resolve(p).is_file() and digest(resolve(p)) == h for p, h in inputs.items())


@contextmanager
def execution_lock(root, plan_sha):
    """Exclusive creation also works on a shared folder; never steal stale locks."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".calibration.lock"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": platform.node(), "started_utc": now(), "plan_sha256": plan_sha}, handle)
    except FileExistsError as exc:
        raise ValueError(f"Calibration already locked: {path}; after a crash, verify its host/PID is stopped before removing the lock") from exc
    try:
        yield
    finally:
        path.unlink()


def next_scale(trials, search):
    """Bracket/refine observed usable scores; never assume a failure is a high score."""
    if len(trials) >= search["maximum_trials_per_pair"]:
        return None
    if not trials:
        return float(search["initial_scale"])
    # A failed/incomplete/invalid last trial ends this pair search conservatively.
    if not trials[-1].get("usable"):
        return None
    passed = [t["scale"] for t in trials if t.get("usable") and t["score"]["target_reached"]]
    below = [t["scale"] for t in trials if t.get("usable") and not t["score"]["target_reached"]]
    if passed:
        high = min(passed)
        lower = [x for x in below if x < high]
        if lower:
            low = max(lower)
            if high / low - 1 <= search["relative_bracket_tolerance"]:
                return None
            proposed = math.sqrt(low * high)
        elif high > search["minimum_scale"]:
            proposed = max(search["minimum_scale"], high / 2)
        else:
            return None
    else:
        high = max(below)
        if high >= search["maximum_scale"]:
            return None
        proposed = min(search["maximum_scale"], high * 2)
    proposed = round(float(proposed), 6)
    return None if any(math.isclose(proposed, t["scale"], rel_tol=1e-7) for t in trials) else proposed


def select_best(trials):
    passed = [t for t in trials if t.get("usable") and t["score"]["target_reached"]]
    return min(passed, key=lambda t: t["scale"]) if passed else None


def recorder_agreement_reasons(export, steps, tolerances=RECORDER_TOLERANCES):
    """Consume the real compare_with_recorders schema; no invented all_pass flag."""
    reasons = []
    agreement = export.get("agreement_with_full_rate_recorders") or {}
    if export.get("available") is not True or agreement.get("compared") is not True or agreement.get("snapshots") != steps:
        return ["moment_rotation_export_not_verified"]
    for axis in ("y", "z"):
        item = agreement.get("axes", {}).get(axis, {})
        if (item.get("compared") is not True or item.get("matched_snapshots") != steps
                or item.get("unmatched_snapshots") != 0
                or item.get("commit_count_locates_the_same_recorder_row") is not True):
            reasons.append(f"recorder_snapshot_matching:{axis}")
        for kind, unit, absolute in (("moment", "kip_in", "moment_absolute_kip_in"),
                                      ("rotation", "rad", "rotation_absolute_rad")):
            difference = item.get(f"max_abs_{kind}_difference_{unit}")
            magnitude = item.get(f"max_abs_{kind}_{unit}")
            if (not isinstance(difference, (int, float)) or not isinstance(magnitude, (int, float))
                    or not math.isfinite(difference) or not math.isfinite(magnitude)
                    or difference < 0 or magnitude < 0
                    or difference > tolerances[absolute] + tolerances["relative"] * magnitude):
                reasons.append(f"recorder_{kind}_difference:{axis}")
    return reasons


def assess_trial(manifest_path, policy):
    """Only finite, complete, consistently installed full records can be selected."""
    from Data_Generation.Hybrid_Exporter import _damage_metrics, COLLAPSE_DRIFT_RATIO
    m = read_json(manifest_path)
    solver = m.get("solver", {})
    reasons = []
    checks = {
        "completed_manifest": m.get("status") == "completed",
        "unchanged_saved_design": m.get("record_unchanged") is True,
        "complete_solver": solver.get("failed") is False and solver.get("truncated") is False
                           and solver.get("completed_steps", 0) == solver.get("requested_steps", -1)
                           and solver.get("completed_steps", 0) > 0,
        "installed_design": m.get("installed_design_verification", {}).get("consistent") is True,
        "installed_topology": m.get("installed_topology", {}).get("consistent") is True,
    }
    for key, passed in checks.items():
        if not passed:
            reasons.append(key)
    if reasons:
        return {"usable": False, "rejection_reasons": reasons, "checks": checks}
    ntha = Path(manifest_path).parent / "ntha"
    with np.load(ntha / "response_arrays.npz", allow_pickle=False) as archive:
        required = ("story_drift", "floor_disp", "hinge_rotation", "hinge_moment", "hinge_history_time", "hinge_history_commit_count", "hinge_tag_order", "base_shear")
        arrays = {key: archive[key] for key in required if key in archive.files}
    params = read_json(ntha / "global_parameters.json")
    status = read_json(ntha / "status.json")
    steps = int(solver["completed_steps"])
    if status.get("failed") is not False or status.get("completed_steps") != steps or status.get("npts_requested") != steps:
        reasons.append("production_status_disagrees")
    floors = params.get("num_floor")
    height = params.get("story_h_in")
    if not isinstance(floors, (int, float)) or int(floors) != floors or floors < 1 or not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
        reasons.append("invalid_building_geometry")
    for name in ("story_drift", "floor_disp", "hinge_rotation", "hinge_moment", "hinge_history_time", "hinge_history_commit_count"):
        a = arrays.get(name)
        if a is None or not a.size or a.ndim == 0 or a.shape[0] != steps or not np.isfinite(a).all():
            reasons.append(f"invalid_or_incomplete_array:{name}")
    if not reasons:
        for key in ("story_drift", "floor_disp"):
            if arrays[key].shape != (steps, 2 * int(floors)):
                reasons.append(f"invalid_array_shape:{key}")
        rotation = arrays["hinge_rotation"]
        if rotation.ndim != 2 or rotation.shape[1] % 2 or arrays["hinge_moment"].shape != rotation.shape:
            reasons.append("invalid_hinge_array_shape")
        time = arrays["hinge_history_time"]
        counts = arrays["hinge_history_commit_count"]
        if time.ndim != 1 or counts.ndim != 1 or np.any(np.diff(time) <= 0) or np.any(np.diff(counts) <= 0) or np.any(counts <= 0) or np.any(counts != np.floor(counts)):
            reasons.append("invalid_committed_history_order")
        duration = solver.get("scheduled_duration_sec")
        if (time.ndim != 1 or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
                or not np.allclose(time, np.arange(1, steps + 1) * duration / steps, rtol=1e-7, atol=1e-7)):
            reasons.append("incomplete_record_duration")
    export = m.get("hinges", {}).get("moment_rotation_export", {})
    reasons.extend(recorder_agreement_reasons(export, steps))
    alignment = m.get("hinges", {}).get("recorder_alignment", {})
    for axis in ("y", "z"):
        item = alignment.get(axis, {})
        if (item.get("time_strictly_increasing") is not True or item.get("rows_match_snapshots_plus_substeps") is not True
                or item.get("ntha_snapshots") != steps or item.get("rows_dropped_beyond_window") != 0):
            reasons.append(f"full_rate_alignment:{axis}")
    if reasons:
        return {"usable": False, "rejection_reasons": reasons, "checks": checks}
    with (ntha / "hinge_backbone.csv").open(newline="", encoding="utf-8-sig") as stream:
        hinge_rows = list(csv.DictReader(stream))
    if not hinge_rows or any(not math.isfinite(float(r[k])) for r in hinge_rows for k in ("damage_ratio", "yielded", "theta_p", "theta_y_spring", "plastic_rotation", "rot_abs_max")):
        return {"usable": False, "rejection_reasons": ["invalid_hinge_metrics"], "checks": checks}
    tags = arrays.get("hinge_tag_order", np.array([]))
    if (len(hinge_rows) * 2 != arrays["hinge_rotation"].shape[1] or tags.ndim != 1 or len(tags) != len(hinge_rows)
            or len({r["hinge_ele_tag"] for r in hinge_rows}) != len(hinge_rows)
            or {int(r["hinge_ele_tag"]) for r in hinge_rows} != set(tags.tolist())):
        reasons.append("hinge_metric_coverage_mismatch")
    for row in hinge_rows:
        theta_p, plastic, rotation, theta_y = (float(row[k]) for k in ("theta_p", "plastic_rotation", "rot_abs_max", "theta_y_spring"))
        if (row.get("material_type") != "IMKPeakOriented" or theta_p <= 0 or min(plastic, rotation, theta_y) < 0 or float(row["yielded"]) not in (0, 1)
                or not math.isclose(plastic, max(0.0, rotation - theta_y), abs_tol=1e-10, rel_tol=1e-8)
                or not math.isclose(float(row["damage_ratio"]), plastic / theta_p, abs_tol=1e-10, rel_tol=1e-8)
                or int(float(row["yielded"])) != int(rotation > theta_y)):
            reasons.append("inconsistent_rotation_damage_proxy")
            break
    damage = _damage_metrics(arrays, hinge_rows, params, status)
    if damage["truncated_steps"] or damage["damage_values_discarded"]:
        reasons.append("response_was_physically_filtered")
    if damage["collapse_flag"] or damage["peak_interstory_drift_ratio"] >= COLLAPSE_DRIFT_RATIO:
        reasons.append("existing_collapse_screen")
    metrics = {key: float(damage[key]) for key in METRIC_NAMES}
    result = {"usable": not reasons, "rejection_reasons": reasons, "checks": checks, "metrics": metrics,
              "hinge_denominator": len(hinge_rows), "legacy_is_inelastic": bool(damage["is_inelastic"]),
              "metric_basis": "Hybrid_Exporter._damage_metrics; simultaneous resultant drifts; hinge-row yielded fraction; max(max(0,rot_abs_max-theta_y_spring)/theta_p). Rotation-demand proxy, not fatigue damage or calibrated energy deterioration.",
              "recorder_tolerances": RECORDER_TOLERANCES,
              "source_files_changed_since_design": m["identity"]["source_files_changed_since_design"]}
    if not reasons:
        result["score"] = score_response(metrics, policy)
    return result


def verify_search_artifacts(history, root):
    artifacts = history.get("artifact_sha256")
    if not artifacts:
        raise ValueError("Saved search has no artifact inventory")
    for name, expected in artifacts.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError("Saved search artifacts are missing/changed; cannot resume")


def save_search(history, out, root):
    history["artifact_sha256"] = {p.relative_to(root).as_posix(): digest(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "search.json"}
    write_json(out / "search.json", history)


def run_pair(case, job, plan, root, sources, inputs):
    result_id, name = job["result_id"], job["job_id"]
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "search.json"
    history = read_json(history_path) if history_path.exists() else {"job_id": name, "plan_sha256": plan["plan_sha256"],
        "shard": job["shard"], "case_id": case["case_id"], "result_id": result_id,
        "status": "running", "trials": [], "scope": "saved-design/current-source diagnostic; not training acceptance"}
    if _STOP.is_set() and not history["trials"]:
        return dict(history, status="not_started", target_reached=False, stop_reason="interrupted")
    if any(history.get(k) != v for k, v in {"job_id": name, "plan_sha256": plan["plan_sha256"], "shard": job["shard"],
                                           "case_id": case["case_id"], "result_id": result_id}.items()):
        raise ValueError("Saved search identity does not match assigned job")
    if history["status"] != "running":
        verify_search_artifacts(history, root)
        return history
    if history["trials"]:
        verify_search_artifacts(history, root)
    while (scale := next_scale(history["trials"], plan["search"])) is not None:
        if _STOP.is_set():
            break
        if not inputs_unchanged(sources, inputs):
            raise RuntimeError("Source or input changed during calibration; no further trials launched")
        index = len(history["trials"]) + 1
        trial_root = out / f"trial_{index:02d}"
        if trial_root.exists():
            raise RuntimeError(f"Interrupted trial exists; inspect it before resuming: {trial_root}")
        trial_root.mkdir()
        args = [sys.executable, "-X", "utf8", "-B", str(RC / "Analysis" / "Fixed_Design_Diagnostics.py"), "ground-motion",
                "--root", str(resolve(case["design_root"])), "--case", case["case_id"], "--profile", case["profile"],
                "--result-id", str(result_id), "--set-name", plan["set_name"], "--scale", str(scale),
                "--output-root", str(trial_root), "--member-material", "IMKPeakOriented", "--hinge-history-stride", "1",
                "--dt-factor", "1", "--plot-limit", "4", "--label", "weighted seismic intensity diagnostic 45/35/15/5"]
        write_json(trial_root / "command.json", {"argv": args, "started_utc": now(), "scale": scale,
                   "plan_sha256": plan["plan_sha256"], "job_id": name,
                   "scale_basis": "raw common multiplicative factor on both unscaled catalog components; not spectral alpha"})
        announce(f"{name}: trial {index}, common X/Y scale {scale:g}")
        env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (trial_root / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(args, cwd=RC, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            deadline = time.monotonic() + plan["trial_timeout_seconds"]
            interruption = None
            try:
                while process.poll() is None:
                    if _STOP.is_set() or time.monotonic() >= deadline:
                        interruption = "interrupted" if _STOP.is_set() else "timeout"
                        process.kill()
                        break
                    try:
                        process.wait(timeout=0.25)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    process.kill()
                returncode = process.wait()
        manifest = trial_root / case["case_id"] / f"peer_{result_id}_scale_{scale:g}" / "manifest.json"
        result = {"scale": scale, "trial": index, "returncode": returncode,
                  "manifest": manifest.relative_to(root).as_posix(), "finished_utc": now(), "interruption": interruption}
        try:
            if interruption or returncode != 0 or not manifest.exists():
                result.update(usable=False, rejection_reasons=["timeout_or_worker_exit_or_missing_manifest"])
            else:
                result.update(assess_trial(manifest, plan["score_policy"]))
        except Exception as exc:
            result.update(usable=False, rejection_reasons=[f"assessment_error:{type(exc).__name__}:{exc}"])
        if not inputs_unchanged(sources, inputs):
            result.update(usable=False, rejection_reasons=["source_or_input_changed_during_trial"])
            result.pop("score", None)
        write_json(trial_root / "score.json", result)
        history["trials"].append(result)
        best = select_best(history["trials"])
        history["best_tested_passing_scale"] = best["scale"] if best else None
        save_search(history, out, root)
        label = f"score {result['score']['score']:.4f}" if result.get("usable") else f"unusable: {result['rejection_reasons']}"
        announce(f"{name}: scale {scale:g}, {label}")
    best = select_best(history["trials"])
    history.update(status="interrupted" if _STOP.is_set() else "completed", finished_utc=now(), best_trial=best,
                   target_reached=best is not None,
                   stop_reason="interrupted" if _STOP.is_set() else "unusable_trial" if history["trials"] and not history["trials"][-1].get("usable")
                   else "budget_or_scale_bound_or_bracket_tolerance")
    save_search(history, out, root)
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard", type=int, required=True, help="1-based device shard from the frozen master plan")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight-only", action="store_true", help="Read/validate only; never launches NTHA")
    action.add_argument("--summarize-only", action="store_true", help="Read saved progress only; never launches NTHA")
    action.add_argument("--first-job-only", action="store_true", help="One full-record search as the live producer canary; resume remaining jobs later")
    args = parser.parse_args(argv)
    plan = validate_plan(read_json(resolve(args.plan)))
    if not 1 <= args.shard <= plan["shards"]:
        raise ValueError("Shard is outside this plan")
    jobs = [j for j in plan["jobs"] if j["shard"] == args.shard]
    if not jobs:
        raise ValueError("This shard has no assigned jobs")
    root = resolve(plan["output_root"]) / f"device_{args.shard:02d}"
    identity = {"plan_sha256": plan["plan_sha256"], "shard": args.shard, "frozen": plan["frozen"]}
    if args.summarize_only:
        if read_json(root / "identity.json") != identity:
            raise ValueError("Saved device identity differs from plan")
        progress = read_json(root / "progress.json")
        print(json.dumps(progress, indent=2))
        return 0
    frozen = verify_frozen_plan(plan)
    sources, inputs = frozen["sources"], frozen["input_sha256"]
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "plan_sha256": plan["plan_sha256"], "shard": args.shard,
                          "assigned_jobs": jobs, "maximum_trials": len(jobs) * plan["search"]["maximum_trials_per_pair"],
                          "use_mode": plan["use_mode"], "training_release": False,
                          "score_policy": plan["score_policy"]}, indent=2))
        return 0
    with execution_lock(root, plan["plan_sha256"]):
        existing = [p for p in root.iterdir() if p.name != ".calibration.lock"]
        if existing:
            if not args.resume or read_json(root / "identity.json") != identity:
                raise ValueError("Output exists; resume needs identical plan, source, runtime and inputs")
            if (root / "progress.json").exists():
                prior = read_json(root / "progress.json")
                for path, expected in prior.get("search_manifest_sha256", {}).items():
                    target = (root / path).resolve()
                    if not target.is_relative_to(root.resolve()) or digest(target) != expected:
                        raise ValueError("Saved search manifest changed; cannot resume")
        else:
            write_json(root / "identity.json", identity)
        return execute_jobs(plan, args, root, jobs, sources, inputs)


def execute_jobs(plan, args, root, jobs, sources, inputs):
    _STOP.clear()
    progress = {"schema_version": "weighted_seismic_calibration_run_v1", "status": "running", "started_utc": now(),
                "plan_sha256": plan["plan_sha256"], "shard": args.shard, "training_release": False,
                "objective": "lowest full-record, noncollapse passing scale tested per case and pair",
                "scope": "intensity-selection diagnostic; hinge calibration remains provisional; not training release",
                "score_policy": plan["score_policy"], "searches": [], "errors": [],
                "planned_searches": len(jobs), "planned_jobs": [j["job_id"] for j in jobs], "search_manifest_sha256": {}}
    cases = {c["case_id"]: c for c in plan["cases"]}
    selected_jobs = jobs[:1] if args.first_job_only else jobs
    progress["executed_jobs"] = [j["job_id"] for j in selected_jobs]
    write_json(root / "progress.json", progress)
    announce(f"Starting {len(selected_jobs)} searches, {plan['workers']} workers; output {root}")
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda *_: _STOP.set())
    try:
        with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
            futures = {}
            for job in selected_jobs:
                case = cases[job["case_id"]]
                needed = {portable_path(resolve(case["design_root"]) / case["case_id"] / name) for name in ("design.json", "result.json")}
                needed.update(c["path"] for c in plan["frozen"]["motions"][str(job["result_id"])]["components"].values())
                needed.update(p for p in inputs if p.startswith("Ground_Motions/metadata/"))
                if plan.get("case_roster"):
                    needed.add(plan["case_roster"]["path"])
                job_inputs = {p: inputs[p] for p in needed}
                futures[pool.submit(run_pair, case, job, plan, root, sources, job_inputs)] = job["job_id"]
            for future in as_completed(futures):
                name = futures[future]
                try:
                    history = future.result()
                    progress["searches"].append(history)
                    if (root / name / "search.json").is_file():
                        progress["search_manifest_sha256"][f"{name}/search.json"] = digest(root / name / "search.json")
                except Exception as exc:
                    progress["errors"].append({"search": name, "error": f"{type(exc).__name__}: {exc}"})
                    announce(f"ERROR {name}: {exc}")
                write_json(root / "progress.json", progress)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    unusable = any(s.get("stop_reason") in ("unusable_trial", "interrupted") or not any(t.get("usable") for t in s["trials"]) for s in progress["searches"])
    progress.update(status="interrupted" if _STOP.is_set() else "completed_with_errors" if progress["errors"] or unusable
                    else "partial_canary" if args.first_job_only and len(jobs) > 1 else "completed", finished_utc=now())
    progress["searches_with_usable_trials"] = sum(any(t.get("usable") for t in s["trials"]) for s in progress["searches"])
    progress["searches_reaching_target"] = sum(s.get("target_reached", False) for s in progress["searches"])
    write_json(root / "progress.json", progress)
    write_json(root / "calibration_results.json", progress)
    announce(f"Finished: {progress['searches_reaching_target']}/{progress['planned_searches']} searches reached target")
    if _STOP.is_set():
        return 130
    if progress["errors"] or unusable:
        return 1
    return 0 if progress["searches_reaching_target"] == len(selected_jobs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
