"""Design-only verification run over the generation plan's own geometries.

Reproduces the geometry and hazard sampling of
Data_Generation/Generate_Parameterized_Dataset.build_plan (same seed, same
shuffle, same round-robin over sites), designs each case with the committed
Design/Config.py in a fresh interpreter, and tabulates what qualification
found. No ground motions are selected and no time-history analysis runs.

    python Design/Verify_Designs.py --count 150 --workers 4 --output-root outputs/design_verification_20260914

Split across machines with disjoint slices of the same plan (the plan and
its SHA are identical everywhere the code and RANGES are the same):

    python Design/Verify_Designs.py --count 150 --plan-only                       # print the plan SHA and stop
    python Design/Verify_Designs.py --count 150 --case-start 1 --case-end 25 --workers 4 --probe-assertions --probe-date 2026-09-16 --output-root <local>\\dv150
    ...
    python Design/Verify_Designs.py --count 150 --summarize-only --output-root <merged root>   # after copying case_* dirs together

Explicit screening manifests (V2, 2026-10-01). ``--manifest <screening_plan.json>`` replaces the
shuffled plan with the manifest's own cases, applies its analysis profile
(Model/Analysis_Profile) in the launcher's identity and in every worker, and
uses its section-iteration budget. The manifest is validated against the
running code (risk basis, profile, population grid) before anything is
written; a manifest run launches nothing by default:

    python Design/Verify_Designs.py --manifest <plan.json> --plan-only
    python Design/Verify_Designs.py --manifest <plan.json> --probe-assertions --probe-date 2026-10-01         --stages design gravity_modal --case-ids case_0001 --workers 1          # the initial case, alone
    python Design/Verify_Designs.py --manifest <plan.json> --probe-assertions --probe-date 2026-10-01         --stages design gravity_modal --remaining --workers 2                    # the others, once it has a result
    python Design/Verify_Designs.py --manifest <plan.json> --probe-assertions --summarize-only
    python Design/Verify_Designs.py --manifest <plan.json> --request-stop

``--stages design gravity_modal`` adds, after each design, the nonlinear build
of the selected design under the profile with gravity, modal analysis, the
installed-topology check and the hinge export check (no ground motion); a
stage requested later runs on the existing design and writes its own file.
Every result row keeps numerical completion, design checks, independent
verification and production acceptance apart (Design/Screening_Manifest).

Resumable: completed artifacts are checked against the current request and
requalified before reuse. Interrupted design locks are preserved for review.
``--request-stop`` lets a running launcher finish its in-flight cases and
start no more; relaunching resumes. Each case
keeps its full design.json. ``--probe-assertions`` fills the three assertion
blocks with PROBE values (labelled in every artifact's request identity) so
the pipeline can be exercised before the real assertions exist; it is not a
certification and the summary says so. Specify the same --probe-date on all
devices when creating their roots. The date is fixed in each plan.json;
geometry/hazard SHA alone does not establish matching code or assertions.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import random
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import date
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_DIR))
sys.path.insert(0, str(RC_DIR / "Data_Generation"))

from Generate_Parameterized_Dataset import (RANGES, SEED, SEISMIC_SITES, POPULATION_BASIS, SAMPLING_METHOD,  # noqa: E402
                                            candidate_geometries, feet_label, sampling_provenance)
from Design.Verification_Integrity import (checked_result, exclusive_lease, expected_identity,
                                           file_sha256)  # noqa: E402
from Design.Screening_Manifest import STAGES, STAGE_RUNNERS, design_overview, gravity_modal_stage, load_manifest, status_fields  # noqa: E402

PROBE = "PROBE -- design verification run, not a certification"
STOP_NAME = "STOP_VERIFICATION"
WORKER_TIMEOUT_S = 6 * 3600
PLAN_KEYS = ("case_id", "num_bay_x", "num_bay_y", "num_floor", "story_height_ft",
             "bay_x_width_ft", "bay_y_width_ft", "seismic_site")


def plan_cases(num_cases, seed=SEED, geometry_offset=0, seismic_sites=SEISMIC_SITES):
    """The first ``num_cases`` cases of the generation plan, geometry and site only."""
    if num_cases <= 0 or geometry_offset < 0 or not seismic_sites:
        raise ValueError("count must be positive, geometry-offset nonnegative, and sites nonempty")
    if any(site not in SEISMIC_SITES for site in seismic_sites):
        raise ValueError("Unknown seismic site in verification plan")
    # The generation plan's own named, seeded planner (the same candidate order build_plan slices).
    geometries = candidate_geometries(seed, SAMPLING_METHOD)
    if geometry_offset + num_cases > len(geometries):
        raise ValueError(f"{geometry_offset + num_cases} exceeds the {len(geometries)} plan geometries.")
    sites = list(seismic_sites)
    random.Random(seed + 2).shuffle(sites)
    cases = []
    for local_index, values in enumerate(geometries[geometry_offset:geometry_offset + num_cases], 1):
        bx, by, floors, story_ft, width_x_ft, width_y_ft = values
        case_index = geometry_offset + local_index
        cases.append({"case_id": f"case_{case_index:04d}", "num_bay_x": bx, "num_bay_y": by, "num_floor": floors,
                      "story_height_ft": story_ft, "bay_x_width_ft": width_x_ft, "bay_y_width_ft": width_y_ft,
                      "seismic_site": sites[(case_index - 1) % len(sites)],
                      "geometry_name": (f"case_{case_index:04d}_bx{bx}_by{by}_s{floors}_"
                                        f"sh{feet_label(story_ft)}ft_bwx{feet_label(width_x_ft)}ft_bwy{feet_label(width_y_ft)}ft_"
                                        f"{sites[(case_index - 1) % len(sites)]}")})
    return cases


def plan_sha256(cases):
    """SHA of the plan's geometry and hazard per case; the same on every machine that builds the same plan."""
    canonical = json.dumps([{k: c[k] for k in PLAN_KEYS} for c in cases], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def case_index(case):
    return int(case["case_id"].split("_")[1])


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(RC_DIR),
                              timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def methodology_sha256(identity):
    """The part of a design's request identity that must agree across machines.

    design_request_identity also hashes the geometry and hazard inputs, so
    every case has its own sha256. Schema, policy (the DesignConfig with its
    assertion stamps) and the source file hashes are what a multi-machine run
    has to hold constant.
    """
    if not identity or any(not identity.get(k) for k in ("schema", "policy", "source_sha256")):
        return None
    payload = {key: identity.get(key) for key in ("schema", "policy", "source_sha256")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tail_request_identity(design_path, tail_bytes=1 << 20):
    """request_identity is the last top-level key of design.json; read it without loading the file."""
    design_path = Path(design_path)
    size = design_path.stat().st_size
    with design_path.open("rb") as handle:
        handle.seek(max(0, size - tail_bytes))
        text = handle.read().decode("utf-8", "replace")
    marker = '"request_identity": '
    start = text.rfind(marker)
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start + len(marker))
    except ValueError:
        return None
    return value


def probe_config(probe_date=None):
    from Design.Config import (DesignConfig, SlabActionAssertions, DemandPolicy, IndependentVerification,
                               FloorAnalysisConfig, PROBE_SLAB_REFINEMENT)
    slab_flags = ("analysis_applicability_verified", "all_floors_enveloped", "load_pattern_envelope_verified",
                  "spatial_envelope_per_unit_width", "twisting_moment_resolution_verified",
                  "zero_membrane_force_verified", "verified", "two_way_shear_path_assessed")
    verify_flags = ("floor_hand_check_verified", "strength_model_verified", "detailing_model_consistency_verified",
                    "slab_column_local_steel_assessed", "fire_resistance_scope_accepted",
                    "congestion_and_placement_accepted", "floor_frame_compatibility_reviewed")
    if not probe_date or date.fromisoformat(probe_date).isoformat() != probe_date:
        raise ValueError("PROBE runs require a shared --probe-date YYYY-MM-DD")
    stamp = dict(asserted_by=PROBE, assertion_date=probe_date, assertion_basis=PROBE)
    return DesignConfig(
        slab_actions=SlabActionAssertions(**{k: True for k in slab_flags}, **stamp),
        demands=DemandPolicy(declared_by=PROBE, declaration_date=probe_date, declaration_basis=PROBE),
        verification=IndependentVerification(**{k: True for k in verify_flags}, **stamp),
        # Asserted slab actions need a refinement plan to select reinforcement;
        # the shared PROBE recipe keeps the methodology hash geometry-independent.
        floor_analysis=FloorAnalysisConfig(slab_refinement=dict(PROBE_SLAB_REFINEMENT)))


def search_summary(record):
    """How the section search ended and what the torsion assessment found, for the result row."""
    search = record.get("search") or {}
    torsion = (record.get("demand_basis") or {}).get("torsion") or {}
    return {"stop_reason": search.get("stop_reason"),
            "selected_iteration": search.get("selected_iteration"),
            "last_iteration": search.get("last_iteration"),
            "tir": torsion.get("tir", torsion.get("max_drift_ratio")),
            "torsional_irregularity": torsion.get("torsional_irregularity")}


def configure_case(case, profile_id=None, emit=False):
    """Apply one case's geometry, hazard and analysis profile to Structure_Parameters, in that order.

    The single place a worker (and anything reproducing a worker) configures itself, so the launcher's
    expected identity, the worker's design and a later nonlinear build of the record agree. Feet are
    converted with the exact factor 12 (half-foot values are exact binary fractions). Returns the
    geometry overrides and the profile identity (None without a profile).
    """
    import Structure_Parameters as sp
    import Geometry_Overrides as go
    overrides = {"NUM_BAY_X": case["num_bay_x"], "NUM_BAY_Y": case["num_bay_y"], "NUM_FLOOR": case["num_floor"],
                 "STORY_H": 12.0 * case["story_height_ft"], "BAY_X": 12.0 * case["bay_x_width_ft"],
                 "BAY_Y": 12.0 * case["bay_y_width_ft"]}
    go.apply_geometry_overrides(overrides, variant_name=case["geometry_name"], emit=emit)
    sp.apply_seismic_site(case["seismic_site"])
    profile = None
    if profile_id:
        from Model.Analysis_Profile import apply_profile
        profile = apply_profile(profile_id)
    # A stale importance factor or a category that is not the configured one stops here.
    sp.seismic_design_basis()
    return overrides, profile


def run_worker(case, out_dir, probe, probe_date=None, attempt_id=None, max_section_iter=None, profile_id=None,
               stages=("design",)):
    with exclusive_lease(Path(out_dir) / ".worker.lease"):
        return _run_worker(case, out_dir, probe, probe_date, attempt_id, max_section_iter, profile_id, stages)


def _run_worker(case, out_dir, probe, probe_date=None, attempt_id=None, max_section_iter=None, profile_id=None,
                stages=("design",)):
    """Design one case in this interpreter (and run its requested later stages); write result.json; never raise."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"case": case, "status": "started", "attempt_id": attempt_id,
              "probe_assertions": probe, "probe_date": probe_date if probe else None,
              "max_section_iter": max_section_iter, "profile_id": profile_id, "stages": list(stages),
              "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log = io.StringIO()
    t0 = time.perf_counter()
    try:
        import Structure_Parameters as sp
        from Design.Config import DesignConfig
        from Design.Design_Driver import load_or_create_design
        with contextlib.redirect_stdout(log):
            _overrides, profile = configure_case(case, profile_id, emit=True)
            result["profile_sha256"] = (profile or {}).get("sha256")
            result["configured_basis"] = {"risk_category": sp.ASCE_RISK_CATEGORY, "importance_factor": sp.ASCE_IE,
                                          "joint_model": sp.JOINT_MODEL, "member_material": sp.IMK_MATERIAL_TYPE,
                                          "analysis_profile_id": sp.ANALYSIS_PROFILE_ID}
            cfg = probe_config(probe_date) if probe else DesignConfig.from_structure_parameters()
            record, created = load_or_create_design(out_dir / "design.json", cfg=cfg, verbose=True,
                                                    max_section_iter=max_section_iter)
        q = record["qualification"]
        by_status = {}
        for c in q["checks"]:
            by_status.setdefault(c["status"], set()).add(c["id"].split(":")[0])
        s, reb = record["sections"], record["reinforcement"]
        result.update({
            "status": "designed", "created": created, "elapsed_s": time.perf_counter() - t0,
            "design_json_bytes": (out_dir / "design.json").stat().st_size,
            "design_sha256": file_sha256(out_dir / "design.json"),
            "request_sha256": (record.get("request_identity") or {}).get("sha256"),
            "methodology_sha256": methodology_sha256(record.get("request_identity")),
            "accepted": q["accepted"], "counts": q["counts"],
            "fail_ids": sorted(by_status.get("fail", [])), "not_evaluated_ids": sorted(by_status.get("not_evaluated", [])),
            "sections": s, "iterations": record.get("iterations"), "governed_by": record["dcr"]["governed_by"],
            "dcr": {"beam": record["dcr"]["beam"], "column": record["dcr"]["column"]},
            "model_period_sec": (record.get("demand") or {}).get("model_period_sec"),
            "column_bars": [reb["col_bar_size"], reb["col_top_bars"], reb["col_side_bars"]],
            "beam_bars": [reb["beam_bar_size"], reb["beam_top_bars"], reb["beam_bot_bars"]],
            "column_hoops": [reb["col_stirrup_bar_size"], reb["col_stirrup_legs"], reb["col_stirrup_spacing_in"]],
            "beam_hoops": [reb["beam_stirrup_bar_size"], reb["beam_stirrup_legs"], reb["beam_stirrup_spacing_in"]],
            "coupled_max_vertical_difference": (record.get("coupled_comparison") or {}).get("max_column_vertical_relative_difference"),
            "schema_version": record.get("schema_version"),
            **search_summary(record),
        })
        result["design_elapsed_s"] = result["elapsed_s"]
        result["overview"] = design_overview(record)
        # Later stages run on the selected design in this same interpreter, so they see exactly the
        # configuration the design was made under. A stage failure is recorded; the design stands.
        stage_results = {}
        for stage in STAGES[1:]:
            if stage not in stages:
                continue
            with contextlib.redirect_stdout(log):
                stage_results[stage] = STAGE_RUNNERS[stage](record, profile_id, out_dir)
            (out_dir / f"{stage}.json").write_text(json.dumps(stage_results[stage], indent=1, default=str), encoding="utf-8")
        result["stage_results"] = {name: {key: value for key, value in stage.items() if key not in ("model_audit", "traceback")}
                                   for name, stage in stage_results.items()}
        result.update(status_fields(record, probe, probe_date, stages, stage_results))
        result["elapsed_s"] = time.perf_counter() - t0
    except Exception as exc:                              # noqa: BLE001
        result.update({"status": "error", "elapsed_s": time.perf_counter() - t0,
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                       # Every failure keeps a complete row: what did not complete, and that nothing is accepted.
                       "numerical_completion": {"design": "error", "stages_requested": list(stages),
                                                "note": "the worker raised before a design record was written; see error and traceback"},
                       "design_checks": {"accepted_by_checklist": False, "note": "no design record to check"},
                       "production_acceptance": {"accepted": False, "reason": "no design record"}})
        # A refusal that carries its evidence (slab trials, failing checks, strip comparisons) keeps it
        # beside the result: the attempt has no design record to hold it.
        evidence = getattr(exc, "evidence", None)
        if evidence:
            (out_dir / "failure_evidence.json").write_text(json.dumps(evidence, indent=1, default=str), encoding="utf-8")
            result["failure_evidence"] = "failure_evidence.json"
    finally:
        result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (out_dir / "log.txt").write_text(log.getvalue(), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def saved_result(out_dir):
    """The case's result.json as a dict, or None."""
    existing = Path(out_dir) / "result.json"
    if not existing.exists():
        return None
    try:
        return json.loads(existing.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_interrupted_design(out_dir):
    """Historical API name; never remove another process's lock or evidence."""
    out_dir = Path(out_dir)
    if (out_dir / ".design.json.lock").exists():
        raise RuntimeError(f"Design lock exists in {out_dir}; establish worker ownership before manual recovery. Nothing removed.")
    return []


def validate_saved(case, out_dir, probe, probe_date, profile_id=None):
    from Design.Config import DesignConfig
    saved = saved_result(out_dir)
    if not saved or saved.get("status") != "designed":
        return saved
    if saved.get("profile_id") != profile_id:
        return {"case": case, "status": "error", "accepted": False,
                "error": f"Untrusted cached design: it was made under profile {saved.get('profile_id')!r}, this run uses "
                         f"{profile_id!r}. Existing files preserved; use a new output root."}
    cfg = probe_config(probe_date) if probe else DesignConfig.from_structure_parameters()
    return checked_result(case, out_dir, saved, expected_identity(case, cfg, profile_id), probe, probe_date)


def stage_file(out_dir, stage):
    return Path(out_dir) / f"{stage}.json"


def attach_stage_files(row, out_dir):
    """Merge the per-stage result files of a case into its row (a stage run after the design has its own file)."""
    results = dict(row.get("stage_results") or {})
    for stage in STAGES[1:]:
        path = stage_file(out_dir, stage)
        if not path.exists():
            continue
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            content = {"status": "error", "error": f"unreadable stage file: {exc}"}
        results[stage] = {key: value for key, value in content.items() if key not in ("model_audit", "traceback")}
    if results:
        row = {**row, "stage_results": results}
        if isinstance(row.get("numerical_completion"), dict):
            row["numerical_completion"] = {**row["numerical_completion"],
                                           **{stage: results[stage].get("status", "not_run") for stage in results}}
    return row


def _run_stage_worker(case, out_dir, probe, probe_date, profile_id, stages):
    """Run the requested later stages on a design that already exists; the design and its result are not rewritten."""
    out_dir = Path(out_dir).resolve()
    log = io.StringIO()
    outcome = {}
    with exclusive_lease(out_dir / ".worker.lease"):
        try:
            from Design.Config import DesignConfig
            from Design.Design_Driver import load_or_create_design
            if not (out_dir / "design.json").exists():
                raise RuntimeError("no design.json to run a stage on")
            with contextlib.redirect_stdout(log):
                configure_case(case, profile_id, emit=True)
                cfg = probe_config(probe_date) if probe else DesignConfig.from_structure_parameters()
                record, created = load_or_create_design(out_dir / "design.json", cfg=cfg, verbose=False)
                if created:                                 # cannot happen: the file exists and its identity was checked
                    raise RuntimeError("a design was created while running a stage")
                for stage in stages:
                    if stage == "design" or stage_file(out_dir, stage).exists():
                        continue                            # the first stage result stands; rerun in a new root
                    result = STAGE_RUNNERS[stage](record, profile_id, out_dir)
                    stage_file(out_dir, stage).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
                    outcome[stage] = result.get("status")
        except Exception as exc:                            # noqa: BLE001
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            (out_dir / "stage_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            with (out_dir / "stage_log.txt").open("a", encoding="utf-8") as handle:
                handle.write(log.getvalue())
    return outcome


def _worker_command(python_exe, case, out_dir, probe, probe_date, max_section_iter, profile_id, stages, extra):
    command = [python_exe, "-B", str(Path(__file__).resolve()), "--worker", json.dumps(case), str(out_dir), *extra]
    if probe:
        command.append("--probe-assertions")
        if probe_date:
            command += ["--probe-date", probe_date]
    if max_section_iter is not None:
        command += ["--max-section-iter", str(int(max_section_iter))]
    if profile_id:
        command += ["--profile", profile_id]
    command += ["--stages", *stages]
    return command


def _launch(python_exe, case, out_dir, probe, probe_date=None, log=print, max_section_iter=None, profile_id=None,
            stages=("design",)):
    out_dir = Path(out_dir)
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    with exclusive_lease(out_dir / ".worker.lease"):
        saved = saved_result(out_dir)
        cached = bool(saved and saved.get("status") == "designed")
        if cached:
            validated = validate_saved(case, out_dir, probe, probe_date, profile_id)
        else:
            clear_interrupted_design(out_dir)
    if cached:
        pending = [s for s in stages if s != "design" and not stage_file(out_dir, s).exists()]
        if pending and validated.get("status") == "designed":
            # A later stage on a design that already exists: its own fresh interpreter and its own file.
            command = _worker_command(python_exe, case, out_dir, probe, probe_date, max_section_iter, profile_id,
                                      ["design", *pending], ["--stage-only"])
            try:
                proc = subprocess.run(command, capture_output=True, text=True, cwd=str(RC_DIR), timeout=WORKER_TIMEOUT_S, env=env)
                (out_dir / "stage_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            except subprocess.TimeoutExpired:
                (out_dir / "stage_stderr.txt").write_text(f"stage worker killed after {WORKER_TIMEOUT_S / 3600:.0f} h", encoding="utf-8")
        return attach_stage_files(validated, out_dir), True
    attempt_id = uuid.uuid4().hex
    command = _worker_command(python_exe, case, out_dir, probe, probe_date, max_section_iter, profile_id, stages,
                              ["--attempt-id", attempt_id])
    # Same child environment the generation scheduler gives its workers (env built above).
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(RC_DIR), timeout=WORKER_TIMEOUT_S, env=env)
        stderr, outcome = proc.stderr or "", f"worker exited {proc.returncode} without a result"
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        outcome = f"worker killed after {WORKER_TIMEOUT_S / 3600:.0f} h"
    out_dir.mkdir(parents=True, exist_ok=True)   # a worker that never started left nothing behind
    (out_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    saved = saved_result(out_dir)
    if saved and saved.get("attempt_id") == attempt_id:
        return saved, False
    # A worker that died without writing (hard crash in the solver, timeout)
    # still gets a result.json so the merged summary shows the error rather
    # than "not yet run"; the next launch retries it because it is not "designed".
    result = {"case": case, "status": "error", "elapsed_s": time.perf_counter() - t0, "host": socket.gethostname(),
              "probe_assertions": probe, "probe_date": probe_date if probe else None, "profile_id": profile_id,
              "stages": list(stages), "attempt_id": attempt_id,
              "error": outcome, "stderr_tail": stderr[-2000:], "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
              "numerical_completion": {"design": "error", "stages_requested": list(stages),
                                       "note": "the worker process ended without a result (crash or timeout); see stderr.txt"},
              "design_checks": {"accepted_by_checklist": False, "note": "no design record to check"},
              "production_acceptance": {"accepted": False, "reason": "no design record"}}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result, False


def _fmt(value, spec="{:.3f}", blank=""):
    return blank if value is None else spec.format(value)


def status_lines(rows, plan):
    """The run's basis and the four statuses of every case, as Markdown lines (kept apart on purpose)."""
    plan = plan or {}
    lines = []
    basis, profile = plan.get("design_basis") or {}, plan.get("profile") or {}
    if plan.get("mode") == "manifest":
        lines.append(f"Manifest `{plan.get('manifest_path')}` (SHA-256 `{str(plan.get('manifest_sha256'))[:16]}`), "
                     f"schema `{plan.get('manifest_schema')}`; screening scope: {plan.get('scope') or 'not stated'}.")
    if basis:
        lines.append(f"Design basis: Risk Category {basis.get('risk_category')}, Ie = {basis.get('importance_factor')}, base story-drift "
                     f"limit {basis.get('allowable_story_drift_ratio')} h (Table 12.12-1, all other structures; divided by rho in SDC D-F; "
                     f"low-rise allowance asserted: {basis.get('low_rise_allowance_asserted')}).")
    if profile:
        settings = profile.get("settings") or {}
        lines.append(f"Analysis profile `{profile.get('id')}` (`{str(profile.get('sha256'))[:12]}`): member material "
                     f"{settings.get('IMK_MATERIAL_TYPE')}, joint model {settings.get('JOINT_MODEL')}, hinge history stride "
                     f"{settings.get('NTHA_HINGE_HISTORY_STRIDE')}. Joint idealisation: "
                     f"{(profile.get('declarations') or {}).get('joint_idealisation', 'not stated')}.")
    lines.append(f"Population basis `{plan.get('population_basis')}`; case selection: {(plan.get('sampling') or {}).get('method')}.\n")
    lines.append("## Status of each case (four separate statements)\n")
    lines.append("Numerical completion says a stage ran to its end. Design checks are the qualification checklist. Independent "
                 "verification says whether a person asserted the items code cannot check; PROBE values are never a verification. "
                 "Production acceptance is False while GENERATION_RELEASE_READY is False.\n")
    lines.append("| case | design search | stop reason | gravity/modal stage | checklist | fail | open | verification | production |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["case"]
        completion, checks = r.get("numerical_completion") or {}, r.get("design_checks") or {}
        counts = checks.get("counts") or {}
        verification, production = r.get("independent_verification") or {}, r.get("production_acceptance") or {}
        lines.append(f"| {c['case_id']} | {completion.get('design', str(r.get('status')))} | {completion.get('design_search_stop_reason') or ''} | "
                     f"{completion.get('gravity_modal', 'not_run')} | {'passes' if checks.get('accepted_by_checklist') else 'does not pass'} | "
                     f"{counts.get('fail', '')} | {counts.get('not_evaluated', '')} | "
                     f"{'PROBE (unverified)' if r.get('probe_assertions') else ('asserted' if verification.get('verified_by_a_person') else 'unverified')} | "
                     f"{'accepted' if production.get('accepted') else 'not accepted'} |")
    lines.append("")
    designed = [r for r in rows if r.get("overview")]
    if designed:
        lines.append("## Design overview\n")
        lines.append("| case | label | T model s | T design s | Cs = V/W | V kip | DCR col | DCR beam | drift / limit | theta / limit | Joint SCWB provided / min | slab refinement | design min | stage min |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in designed:
            o = r["overview"]
            pb, mu, ds = o["period_and_base_shear"], o["member_utilisation"], o["drift_and_stability"]
            drift, theta = ds.get("worst_story_drift") or {}, ds.get("worst_stability") or {}
            scwb, refinement = o.get("scwb") or {}, (o.get("mesh_and_refinement") or {}).get("slab_refinement") or {}
            stage = (r.get("stage_results") or {}).get("gravity_modal") or {}
            lines.append(f"| {r['case']['case_id']} | {r['case'].get('label', '')} | {_fmt(pb.get('model_period_sec'))} | "
                         f"{_fmt(pb.get('design_period_sec'))} | {_fmt(pb.get('cs_base_shear_over_weight'), '{:.4f}')} | "
                         f"{_fmt(pb.get('base_shear_kip'), '{:.0f}')} | {_fmt(mu.get('column'))} | {_fmt(mu.get('beam'))} | "
                         f"{_fmt(drift.get('drift_ratio'), '{:.4f}')} / {_fmt(drift.get('allowable_drift_ratio'), '{:.4f}')} | "
                         f"{_fmt(theta.get('theta'))} / {_fmt(theta.get('theta_limit'))} | "
                         f"{_fmt(scwb.get('joint_min_ratio_provided'), '{:.2f}')} / {_fmt(scwb.get('ratio_min'), '{:.2f}')} | "
                         f"{refinement.get('status') or ''} | {_fmt((r.get('design_elapsed_s') or r.get('elapsed_s') or 0) / 60, '{:.1f}')} | "
                         f"{_fmt(stage.get('elapsed_s') / 60 if stage.get('elapsed_s') is not None else None, '{:.1f}')} |")
        lines.append("")
        lines.append("## Failed and open checks by subject\n")
        for r in designed:
            checks = r.get("design_checks") or {}
            failed, open_items = checks.get("failed") or {}, checks.get("not_evaluated") or {}
            lines.append(f"- **{r['case']['case_id']}** failed: " + ("; ".join(f"{group}: {', '.join(ids)}" for group, ids in failed.items()) or "none")
                         + ". Not evaluated: " + ("; ".join(f"{group}: {', '.join(ids)}" for group, ids in open_items.items()) or "none") + ".")
        lines.append("")
    staged = [r for r in rows if (r.get("stage_results") or {}).get("gravity_modal")]
    if staged:
        lines.append("## Gravity and modal stage on the nonlinear model (no ground motion)\n")
        lines.append("| case | status | profile installed | design matches record | hinges | joint springs | face interfaces | hinge export complete | T1 nonlinear model s | gravity reaction / ledger kip |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in staged:
            s = r["stage_results"]["gravity_modal"]
            topology, checks = s.get("installed_topology") or {}, s.get("checks") or {}
            periods = (s.get("modal") or {}).get("periods_elastic_reference_sec") or []
            gravity = s.get("gravity") or {}
            lines.append(f"| {r['case']['case_id']} | {s.get('status')} | {checks.get('profile_installed')} | "
                         f"{checks.get('installed_design_matches_record')} | {topology.get('hinge_spring_elements_in_domain', '')} | "
                         f"{topology.get('joint_spring_elements_in_domain', '')} | {topology.get('face_slip_interfaces_registered', '')} | "
                         f"{checks.get('hinge_export_complete')} | {_fmt(periods[0] if periods else None)} | "
                         f"{_fmt(gravity.get('vertical_reaction_kip'), '{:.1f}')} / {_fmt(gravity.get('expected_vertical_load_kip'), '{:.1f}')} |"
                         + (f" {s.get('error')}" if s.get("status") == "error" else ""))
        lines.append("")
    return lines


def write_case_summary(row, case_dir):
    """A short human-readable file beside each case's artifacts."""
    case_dir = Path(case_dir)
    if not case_dir.exists():
        return
    c = row["case"]
    lines = [f"# {c['case_id']} {c.get('label', '')}\n", f"{c.get('purpose', '')}\n",
             f"Geometry {c['num_bay_x']} x {c['num_bay_y']} bays, {c['num_floor']} stories; bays {c['bay_x_width_ft']} x "
             f"{c['bay_y_width_ft']} ft; story {c['story_height_ft']} ft; site {c['seismic_site']}; name `{c.get('geometry_name')}`.",
             f"Status: {row.get('status')}; elapsed {_fmt(row.get('elapsed_s'), '{:.0f}')} s; host {row.get('host', '')}; "
             f"profile {row.get('profile_id')}; PROBE assertions {row.get('probe_assertions')}.\n"]
    if row.get("error"):
        lines.append(f"Error: {row['error']}\n")
    for key in ("numerical_completion", "design_checks", "independent_verification", "production_acceptance", "overview", "stage_results"):
        if row.get(key) is not None:
            lines.append(f"## {key}\n\n```json\n{json.dumps(row[key], indent=1, default=str)}\n```\n")
    lines.append("Artifacts in this folder: design.json (full record and iteration history), result.json, log.txt, stderr.txt, "
                 "and, when the stage ran, gravity_modal.json with gravity_modal/ (hinge spring table, schema, gravity-state arrays)."
                 + (" failure_evidence.json holds the trials and failing checks of the refused attempt."
                    if row.get("failure_evidence") else ""))
    (case_dir / "case_summary.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(results, root, probe, cases=None, probe_date=None, profile_id=None, plan=None):
    root = Path(root)
    cases = cases if cases is not None else [r["case"] for r in results]
    expected = {c["case_id"]: c for c in cases}
    if len(expected) != len(cases):
        raise ValueError("Duplicate cases in summary plan")
    supplied = {}
    for result in results:
        cid = result["case"]["case_id"]
        if cid not in expected or cid in supplied or result["case"] != expected[cid]:
            raise ValueError("Duplicate or mismatched result case in summary")
        supplied[cid] = result
    rows = []
    for cid, case in sorted(expected.items()):
        result = supplied.get(cid, {"case": case, "status": "missing"})
        if result.get("status") == "designed":
            with exclusive_lease(root / cid / ".worker.lease"):
                result = validate_saved(case, root / cid, probe, probe_date, profile_id) or {
                    "case": case, "status": "error", "error": "Saved result is missing"}
        result = attach_stage_files(result, root / cid)
        write_case_summary(result, root / cid)
        rows.append(result)
    designed = [r for r in rows if r.get("status") == "designed"]
    accepted = [r for r in designed if r.get("accepted")]
    errors = [r for r in rows if r.get("status") == "error"]
    missing = [r for r in rows if r.get("status") == "missing"]
    open_items = Counter(i for r in designed for i in r.get("not_evaluated_ids", []))
    fail_items = Counter(i for r in designed for i in r.get("fail_ids", []))
    identities = Counter(r.get("methodology_sha256") for r in designed)
    hosts = Counter(r.get("host") for r in designed)
    total_time = sum(r.get("elapsed_s", 0.0) for r in designed)
    total_bytes = sum(r.get("design_json_bytes", 0) for r in designed)
    lines = [f"# Design verification run -- {time.strftime('%Y-%m-%d %H:%M')}\n",
             f"{len(rows)} plan cases; {len(designed)} designed, {len(accepted)} accepted, "
             f"{len(designed) - len(accepted)} not accepted, {len(errors)} errors, {len(missing)} not yet run.",
             f"Assertions: {'PROBE values in all three blocks -- exercises the pipeline, certifies nothing' if probe else 'the committed Design/Config.py'}.",
             f"Design time {total_time / 60:.0f} min serial-equivalent ({total_time / max(1, len(designed)) / 60:.1f} min/case); "
             f"design.json total {total_bytes / 1e9:.2f} GB.",
             f"Methodology identities (schema + config + source hashes) among designed cases: {len(identities)}"
             + (" -- one code base and config for the whole run" if len(identities) == 1 else
                " -- no verified identity" if not identities else
                " -- ** more than one: not every case was designed with the same code or config **"),
             f"Hosts: " + ", ".join(f"{h} ({n})" for h, n in hosts.most_common()) + "\n"]
    lines += status_lines(rows, plan)
    if len(identities) > 1:
        lines.append("## Methodology identities\n")
        for sha, n in identities.most_common():
            members = [r for r in designed if r.get("methodology_sha256") == sha]
            by_host = Counter(r.get("host") for r in members)
            lines.append(f"- `{str(sha)[:12]}`: {n} cases on " + ", ".join(f"{h} ({k})" for h, k in by_host.most_common())
                         + "; e.g. " + ", ".join(r["case"]["case_id"] for r in members[:4]))
        lines.append("")
    if fail_items:
        lines.append("## Failed checks (cases with the item)\n")
        lines += [f"- `{k}`: {v}" for k, v in fail_items.most_common()]
        lines.append("")
    if open_items:
        lines.append("## Open (not evaluated) items (cases with the item)\n")
        lines += [f"- `{k}`: {v}" for k, v in open_items.most_common()]
        lines.append("")
    if errors:
        lines.append("## Errors\n")
        lines += [f"- {r['case']['case_id']} ({r.get('host', '?')}): {r.get('error')}" for r in errors]
        lines.append("")
    if missing:
        lines.append("## Not yet run\n")
        lines.append(", ".join(r["case"]["case_id"] for r in missing))
        lines.append("")
    lines.append("## Cases\n")
    lines.append("| case | plan | site | result | fail | open | min | MB | T1 s | sections | col bars | beam bars | col hoops | beam hoops | governed | stop (sel/last) | coupled max | host |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["case"]
        plan_text = f"{c['num_bay_x']}x{c['num_bay_y']}x{c['num_floor']} {c['bay_x_width_ft']:g}x{c['bay_y_width_ft']:g} ft / {c['story_height_ft']:g} ft"
        if r.get("status") != "designed":
            lines.append(f"| {c['case_id']} | {plan_text} | {c['seismic_site']} | {str(r.get('status', '')).upper()} | | | | | | "
                         f"{str(r.get('error', ''))[:80]} | | | | | | | | {r.get('host', '')} |")
            continue
        s = r["sections"]
        period = r.get("model_period_sec")
        lines.append(f"| {c['case_id']} | {plan_text} | {c['seismic_site']} | {'accepted' if r['accepted'] else 'open'} | "
                     f"{r['counts']['fail']} | {r['counts']['not_evaluated']} | {r['elapsed_s'] / 60:.1f} | {r['design_json_bytes'] / 1e6:.0f} | "
                     f"{period if period is None else '%.2f' % period} | "
                     f"{s['b_col_in']:g}x{s['h_col_in']:g} fc{s['fc_col_ksi']:g} / {s['b_beam_in']:g}x{s['h_beam_in']:g} fc{s['fc_beam_ksi']:g} | "
                     f"#{r['column_bars'][0]} {r['column_bars'][1]}T/{r['column_bars'][2]}S | #{r['beam_bars'][0]} {r['beam_bars'][1]}T/{r['beam_bars'][2]}B | "
                     f"#{r['column_hoops'][0]}-{r['column_hoops'][1]}L@{r['column_hoops'][2]:g} | #{r['beam_hoops'][0]}-{r['beam_hoops'][1]}L@{r['beam_hoops'][2]:g} | "
                     f"{r['governed_by']} | {r.get('stop_reason') or ''} ({r.get('selected_iteration') or ''}/{r.get('last_iteration') or ''}) | "
                     f"{100 * (r.get('coupled_max_vertical_difference') or 0):.1f}% | {r.get('host', '')} |")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "num_bay_x", "num_bay_y", "num_floor", "story_height_ft", "bay_x_width_ft", "bay_y_width_ft",
                         "seismic_site", "status", "accepted", "fail", "not_evaluated", "elapsed_s", "design_json_bytes",
                         "model_period_sec", "b_col_in", "h_col_in", "fc_col_ksi", "b_beam_in", "h_beam_in", "fc_beam_ksi",
                         "dcr_column", "dcr_beam", "governed_by", "iterations", "stop_reason", "selected_iteration",
                         "last_iteration", "tir", "torsional_irregularity", "host", "request_sha256", "methodology_sha256",
                         "fail_ids", "not_evaluated_ids"])
        for r in rows:
            c, s, d = r["case"], r.get("sections") or {}, r.get("dcr") or {}
            writer.writerow([c["case_id"], c["num_bay_x"], c["num_bay_y"], c["num_floor"], c["story_height_ft"],
                             c["bay_x_width_ft"], c["bay_y_width_ft"], c["seismic_site"], r.get("status"), r.get("accepted"),
                             (r.get("counts") or {}).get("fail"), (r.get("counts") or {}).get("not_evaluated"),
                             round(r.get("elapsed_s", 0.0), 1), r.get("design_json_bytes"), r.get("model_period_sec"),
                             s.get("b_col_in"), s.get("h_col_in"), s.get("fc_col_ksi"), s.get("b_beam_in"), s.get("h_beam_in"),
                             s.get("fc_beam_ksi"), d.get("column"), d.get("beam"), r.get("governed_by"), r.get("iterations"),
                             r.get("stop_reason"), r.get("selected_iteration"), r.get("last_iteration"), r.get("tir"),
                             r.get("torsional_irregularity"),
                             r.get("host"), r.get("request_sha256"), r.get("methodology_sha256"),
                             ";".join(r.get("fail_ids", [])), ";".join(r.get("not_evaluated_ids", []))])
    plan = plan or {}
    (root / "summary.json").write_text(json.dumps({
        "probe_assertions": probe, "probe_date": probe_date if probe else None, "profile_id": profile_id,
        "mode": plan.get("mode", "generation_plan_slice"), "manifest_sha256": plan.get("manifest_sha256"),
        "design_basis": plan.get("design_basis"), "profile": plan.get("profile"),
        "population_basis": plan.get("population_basis"), "sampling": plan.get("sampling"),
        "plan_sha256": plan.get("plan_sha256"), "max_section_iter": plan.get("max_section_iter"),
        "counts": {"cases": len(rows), "designed": len(designed), "checklist_passes": len(accepted),
                   "errors": len(errors), "not_yet_run": len(missing),
                   "production_accepted": sum(1 for r in rows if (r.get("production_acceptance") or {}).get("accepted"))},
        "status_fields": ["numerical_completion", "design_checks", "independent_verification", "production_acceptance"],
        "cases": rows}, indent=1, default=str), encoding="utf-8")
    return lines


def load_or_write_plan(root, cases, args):
    """plan.json is written once per root; later launches must build the same plan.

    A different SHA means this machine has different code, RANGES or
    arguments from whoever created the root -- its case_0007 would be a
    different building. The PROBE stamp date is fixed here so every case of
    the run carries one request identity.
    """
    import Structure_Parameters as sp
    from Model.Analysis_Profile import PROFILES, UNPROFILED
    sha = plan_sha256(cases)
    path = root / "plan.json"
    manifest = getattr(args, "manifest_data", None)
    profile_id = args.profile
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("plan_sha256") != sha:
            raise SystemExit(f"plan SHA mismatch: this launch builds {sha} but {path} holds {plan.get('plan_sha256')}. "
                             "Stop: git pull, check RANGES/SEISMIC_SITES and --count/--seed/--geometry-offset, or use a new root.")
        if plan.get("mode", "generation_plan_slice") != ("manifest" if manifest else "generation_plan_slice"):
            raise SystemExit(f"{path} was created as a {plan.get('mode', 'generation_plan_slice')} run; use one mode per root.")
        if manifest and plan.get("manifest_sha256") != manifest["sha256"]:
            raise SystemExit(f"The manifest changed since {path} was created (SHA-256 {plan.get('manifest_sha256')} then, "
                             f"{manifest['sha256']} now). A changed input needs a new output root; existing results are preserved.")
        if plan.get("profile_id") != profile_id:
            raise SystemExit(f"{path} was created with profile {plan.get('profile_id')!r}; this launch asks for {profile_id!r}. "
                             "Use one profile per root.")
        stored_basis, current_basis = plan.get("design_basis") or {}, sp.seismic_design_basis()
        for key in ("risk_category", "importance_factor", "allowable_story_drift_ratio"):
            if stored_basis.get(key) != current_basis.get(key):
                raise SystemExit(f"{path} was created under design basis {key} = {stored_basis.get(key)!r}; the running code has "
                                 f"{current_basis.get(key)!r}. Use a new output root; existing results are preserved.")
    else:
        if args.probe_assertions and not args.probe_date:
            raise SystemExit("New PROBE roots require --probe-date YYYY-MM-DD; use the same date on all devices.")
        if manifest and manifest["fresh_root_required"] and any(p.name not in (".launcher.lease", STOP_NAME) for p in root.iterdir()):
            raise SystemExit(f"The manifest requires a fresh output root, and {root} already holds files without a plan.json. "
                             "Choose a new root; nothing was changed.")
        # The profile identity under the default geometry: its settings and required state do not
        # depend on the case, so it is recorded once for the root.
        profile = None
        if profile_id:
            from Design.Verification_Integrity import _INPUT_LOCK
            from Model.Analysis_Profile import apply_profile
            with _INPUT_LOCK:
                snapshot = dict(vars(sp))
                try:
                    profile = apply_profile(profile_id)
                finally:
                    for key in set(vars(sp)) - set(snapshot):
                        delattr(sp, key)
                    vars(sp).update(snapshot)
        plan = {"mode": "manifest" if manifest else "generation_plan_slice",
                "seed": None if manifest else args.seed, "geometry_offset": None if manifest else args.geometry_offset,
                "count": len(cases), "plan_sha256": sha,
                "probe_assertions": args.probe_assertions,
                "probe_date": args.probe_date if args.probe_assertions else None,
                "max_section_iter": args.max_section_iter,
                "profile_id": profile_id, "profile": profile,
                "profile_note": None if profile_id else f"no profile applied: {UNPROFILED} (the repository defaults, joint model {sp.JOINT_MODEL})",
                "design_basis": sp.seismic_design_basis(),
                "population_basis": POPULATION_BASIS,
                "sampling": manifest["sampling"] if manifest else sampling_provenance(args.seed),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "created_on": socket.gethostname(),
                "cases": cases, "launches": []}
        if manifest:
            plan.update({"manifest_path": manifest["path"], "manifest_sha256": manifest["sha256"], "manifest_schema": manifest["schema"],
                         "scope": manifest["scope"], "initial_case_id": manifest["initial_case_id"],
                         "initial_workers": manifest["initial_workers"], "subsequent_max_workers": manifest["subsequent_max_workers"],
                         "acceptance_policy": manifest["acceptance_policy"],
                         "available_profiles": sorted(PROFILES)})
    if plan_sha256(plan["cases"]) != sha:
        raise SystemExit("Stored plan cases do not match its digest; files preserved.")
    if args.probe_date and args.probe_date != plan.get("probe_date"):
        raise SystemExit("PROBE date differs from the stored plan; use one shared date.")
    if args.max_section_iter is not None and args.max_section_iter != plan.get("max_section_iter"):
        raise SystemExit(f"{path} was created with max_section_iter={plan.get('max_section_iter')}; this launch asks for "
                         f"{args.max_section_iter}. Use one budget per root.")
    if args.probe_assertions:
        probe_config(plan.get("probe_date"))
    if bool(plan.get("probe_assertions")) != bool(args.probe_assertions):
        raise SystemExit(f"{path} was created with probe_assertions={plan.get('probe_assertions')}; "
                         f"this launch asks for {args.probe_assertions}. Use one setting per root.")
    plan.setdefault("launches", []).append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "git_head": git_head(),
        "python": args.python_exe, "case_start": args.case_start, "case_end": args.case_end,
        "case_ids": args.case_ids, "workers": args.workers, "summarize_only": args.summarize_only,
        "stages": list(args.stages), "remaining": bool(getattr(args, "remaining", False))})
    path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", nargs=2, metavar=("CASE_JSON", "OUT_DIR"), help=argparse.SUPPRESS)
    parser.add_argument("--probe-date", default=None, help="Shared YYYY-MM-DD assertion date; required for a new PROBE root.")
    parser.add_argument("--attempt-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-section-iter", type=int, default=None,
                        help="Section-iteration budget of the design search (default: the driver's 10); recorded in the plan, one value per root.")
    parser.add_argument("--count", type=int, default=150, help="Number of plan cases, from the first (default 150).")
    parser.add_argument("--geometry-offset", type=int, default=0, help="Skip this many plan geometries first.")
    parser.add_argument("--seed", type=int, default=SEED, help="Plan seed (default: the generation plan's).")
    parser.add_argument("--sites", nargs="*", default=None, help="Hazard labels to round-robin over (default: the plan's).")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Only these plan case ids.")
    parser.add_argument("--case-start", type=int, default=None, help="First plan case index to run on this machine (1-based, inclusive).")
    parser.add_argument("--case-end", type=int, default=None, help="Last plan case index to run on this machine (inclusive).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default 4; in a manifest run the manifest's limit for the launch).")
    parser.add_argument("--output-root", default=None, help="Default: outputs/design_verification_<date>.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--probe-assertions", action="store_true",
                        help="Fill the three assertion blocks with PROBE values (labelled); exercises the pipeline only.")
    parser.add_argument("--plan-only", action="store_true", help="Print the plan and its SHA, write nothing, and stop.")
    parser.add_argument("--request-stop", action="store_true",
                        help=f"Write <root>/{STOP_NAME} and exit; the running launcher finishes its in-flight cases and "
                             "starts no more. The next launch clears the file and resumes.")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Design nothing; rebuild summary.* from the result.json files already under the root "
                             "(after copying every machine's case_* directories into one root).")
    parser.add_argument("--manifest", default=None,
                        help="Explicit screening manifest (seisframe_v2_screening_plan_v1): its cases replace the shuffled plan, "
                             "its profile and section-iteration budget are used, and it is validated against the running code. "
                             "Cases must be selected with --case-ids or --remaining; nothing is launched by default.")
    parser.add_argument("--profile", default=None,
                        help="Analysis profile (Model/Analysis_Profile) applied in the launcher's identity and in every worker; "
                             "part of the design request identity. A manifest supplies its own.")
    parser.add_argument("--stages", nargs="+", default=["design"], choices=STAGES,
                        help="'design' (default), optionally followed by 'gravity_modal': the nonlinear build of the selected "
                             "design with gravity, modal analysis, installed-topology and hinge-export checks. No ground motion.")
    parser.add_argument("--remaining", action="store_true",
                        help="Manifest mode: run every manifest case that has no result yet (after the initial case has one).")
    parser.add_argument("--stage-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if "design" not in args.stages:
        parser.error("--stages must include 'design'; a later stage runs on a design that exists or is made in the same run")
    if args.worker:
        case = json.loads(args.worker[0])
        if args.stage_only:
            outcome = _run_stage_worker(case, args.worker[1], args.probe_assertions, args.probe_date, args.profile, args.stages)
            print(json.dumps(outcome, default=str))
            return 0
        result = run_worker(case, args.worker[1], args.probe_assertions, args.probe_date, args.attempt_id, args.max_section_iter,
                            args.profile, tuple(args.stages))
        print(json.dumps({k: result.get(k) for k in ("status", "elapsed_s", "accepted", "counts", "error")}, default=str))
        return 0
    if (args.workers is not None and args.workers <= 0) or args.sites == [] or args.case_ids == []:
        parser.error("workers must be positive; explicit sites/case-ids must not be empty")
    if any(v is not None and v < 1 for v in (args.case_start, args.case_end)):
        parser.error("case-start and case-end must be positive")
    if args.case_start is not None and args.case_end is not None and args.case_start > args.case_end:
        parser.error("case-start must not exceed case-end")
    args.manifest_data = None
    if args.manifest:
        try:
            manifest = load_manifest(args.manifest)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc))
        if args.profile not in (None, manifest["profile_id"]):
            raise SystemExit(f"--profile {args.profile!r} contradicts the manifest's profile {manifest['profile_id']!r}.")
        if args.max_section_iter not in (None, manifest["max_section_iter"]):
            raise SystemExit(f"--max-section-iter {args.max_section_iter} contradicts the manifest's budget {manifest['max_section_iter']}; "
                             "a budget change is a new manifest and a new root, not an override.")
        if args.sites is not None or args.case_start is not None or args.case_end is not None:
            raise SystemExit("A manifest names its cases and sites; select them with --case-ids or --remaining.")
        args.profile, args.max_section_iter, args.manifest_data = manifest["profile_id"], manifest["max_section_iter"], manifest
        args.output_root = args.output_root or manifest["output_root"]
        cases = manifest["cases"]
        print(f"Manifest: {len(cases)} cases from {manifest['path']} (SHA-256 {manifest['sha256'][:16]}); profile {manifest['profile_id']}; "
              f"budget {manifest['max_section_iter']} section iterations; initial case {manifest['initial_case_id']}")
    else:
        if args.remaining:
            parser.error("--remaining applies to a manifest run")
        cases = plan_cases(args.count, seed=args.seed, geometry_offset=args.geometry_offset,
                           seismic_sites=tuple(args.sites) if args.sites is not None else SEISMIC_SITES)
        print(f"Plan: {len(cases)} cases (seed {args.seed}, offset {args.geometry_offset}); "
              f"hazards {list(args.sites) if args.sites else list(SEISMIC_SITES)}")
    sha = plan_sha256(cases)
    print(f"Plan SHA256: {sha}")
    if args.plan_only:
        return 0
    # Absolute: workers run with cwd RC_DIR, so a root given relative to the
    # launcher's cwd would otherwise be resolved twice, to two different places.
    root = Path(args.output_root or (RC_DIR / "outputs" / f"design_verification_{time.strftime('%Y%m%d')}")).resolve()
    stop_file = root / STOP_NAME
    if args.request_stop:
        root.mkdir(parents=True, exist_ok=True)
        stop_file.write_text(f"{socket.gethostname()} {time.strftime('%Y-%m-%d %H:%M:%S')}{chr(10)}", encoding="utf-8")
        print(f"stop requested: {stop_file}")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lease(root / ".launcher.lease"):
        return _run_plan(args, cases, root)


def _run_plan(args, cases, root):
    sha = plan_sha256(cases)
    stop_file = root / STOP_NAME
    if args.case_ids and not set(args.case_ids).issubset({c["case_id"] for c in cases}):
        raise ValueError("Requested case IDs are not in the plan")
    # Selection rules of a manifest run are checked before anything is written to the root.
    manifest = args.manifest_data
    if manifest and not args.summarize_only:
        # A manifest run never launches its whole population by default. The initial case runs alone
        # with one worker; the others only once it has a result, with the manifest's worker cap.
        initial = manifest["initial_case_id"]
        initial_done = saved_result(root / initial) is not None
        if args.remaining:
            if args.case_ids:
                raise SystemExit("Give --case-ids or --remaining, not both.")
            if not initial_done:
                raise SystemExit(f"The initial case {initial} has no result yet; run it first: --case-ids {initial} --workers "
                                 f"{manifest['initial_workers']}.")
            args.case_ids = [c["case_id"] for c in cases if saved_result(root / c["case_id"]) is None]
            if not args.case_ids:
                raise SystemExit("Every manifest case already has a result; use --summarize-only, or a new root to rerun.")
        if not args.case_ids:
            raise SystemExit(f"A manifest run launches nothing by default. Start with --case-ids {initial} --workers "
                             f"{manifest['initial_workers']}; afterwards use --remaining (at most {manifest['subsequent_max_workers']} workers).")
        others = [cid for cid in args.case_ids if cid != initial]
        if others and not initial_done:
            raise SystemExit(f"The initial case {initial} runs first and alone; {others} wait until it has a result.")
        cap = manifest["subsequent_max_workers"] if initial_done else manifest["initial_workers"]
        if args.workers is None:
            args.workers = cap
        if args.workers > cap:
            raise SystemExit(f"--workers {args.workers} exceeds the manifest's limit of {cap} for this launch.")

    if args.workers is None:
        args.workers = 4
    plan = load_or_write_plan(root, cases, args)
    probe_date = plan.get("probe_date")

    run_log = root / "run_log.txt"

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    all_cases = list(cases)
    profile_id = plan.get("profile_id")

    def summarize_all(results_by_id):
        results = [results_by_id.get(c["case_id"]) or saved_result(root / c["case_id"]) or {"case": c, "status": "missing"}
                   for c in all_cases]
        return summarize(results, root, bool(plan.get("probe_assertions")), all_cases, probe_date, profile_id, plan)

    if args.summarize_only:
        lines = summarize_all({})
        print("\n".join(lines[:14]))
        print(f"summary: {root / 'summary.md'}")
        return 0

    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c["case_id"] in wanted]
    if args.case_start is not None or args.case_end is not None:
        lo = args.case_start or 1
        hi = args.case_end if args.case_end is not None else args.geometry_offset + args.count
        cases = [c for c in cases if lo <= case_index(c) <= hi]
    stages = tuple(args.stages)
    if not cases:
        raise ValueError("Requested case range selects no cases")
    if stop_file.exists():
        stop_file.unlink()
        log(f"cleared {STOP_NAME} left by an earlier stop request")
    log(f"launch on {socket.gethostname()} git {git_head() or '?'}: {len(cases)} cases "
        f"({cases[0]['case_id']}..{cases[-1]['case_id']}), {args.workers} workers, root {root}, "
        f"{'PROBE assertions dated ' + str(probe_date) if args.probe_assertions else 'committed Design/Config.py'}, "
        f"profile {profile_id or 'none (repository defaults)'}, stages {list(stages)}, "
        f"{'manifest ' + str(plan.get('manifest_sha256'))[:16] if manifest else 'generation plan slice'}, plan SHA {sha}")

    def run(case):
        if stop_file.exists():
            saved = saved_result(root / case["case_id"])
            if saved and saved.get("status") == "designed":
                return saved
            return {"case": case, "status": "missing", "host": socket.gethostname()}
        t0 = time.perf_counter()
        try:
            result, cached = _launch(args.python_exe, case, root / case["case_id"], args.probe_assertions, probe_date, log=log,
                                     max_section_iter=plan.get("max_section_iter"), profile_id=profile_id, stages=stages)
        except Exception as exc:                          # noqa: BLE001 -- one case must not take the launcher down
            result, cached = {"case": case, "status": "error", "host": socket.gethostname(),
                              "error": f"launcher: {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, False
        tag = "cached" if cached else f"{time.perf_counter() - t0:6.0f}s"
        counts = result.get("counts") or {}
        sections = result.get("sections") or {}
        section_text = (f"{sections['b_col_in']:g}x{sections['h_col_in']:g}/{sections['b_beam_in']:g}x{sections['h_beam_in']:g}"
                        if sections else "")
        log(f"{case['case_id']} {tag:>8}  {result.get('status')}  accepted={result.get('accepted')}  "
            f"fail={counts.get('fail')} open={counts.get('not_evaluated')}  {section_text}  {result.get('error', '') or ''}")
        return result

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, cases))
    if manifest:
        # The summary of a manifest run always covers the whole manifest: cases not launched yet are "missing".
        lines = summarize_all({r["case"]["case_id"]: r for r in results})
    else:
        lines = summarize(results, root, args.probe_assertions, cases, probe_date, profile_id, plan)
    status = Counter(r.get("status") for r in results)
    log(f"{'stopped on request' if stop_file.exists() else 'finished slice'}: {dict(status)}; summary {root / 'summary.md'}")
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
