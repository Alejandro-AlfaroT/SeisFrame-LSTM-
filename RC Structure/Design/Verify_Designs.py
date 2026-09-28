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

from Generate_Parameterized_Dataset import RANGES, SEED, SEISMIC_SITES  # noqa: E402
from Design.Verification_Integrity import (checked_result, exclusive_lease, expected_identity,
                                           file_sha256)  # noqa: E402

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
    geometries = list(product(*RANGES.values()))
    if geometry_offset + num_cases > len(geometries):
        raise ValueError(f"{geometry_offset + num_cases} exceeds the {len(geometries)} plan geometries.")
    random.Random(seed).shuffle(geometries)
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
                                        f"sh{story_ft}ft_bwx{width_x_ft}ft_bwy{width_y_ft}ft_{sites[(case_index - 1) % len(sites)]}")})
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


def run_worker(case, out_dir, probe, probe_date=None, attempt_id=None, max_section_iter=None):
    with exclusive_lease(Path(out_dir) / ".worker.lease"):
        return _run_worker(case, out_dir, probe, probe_date, attempt_id, max_section_iter)


def _run_worker(case, out_dir, probe, probe_date=None, attempt_id=None, max_section_iter=None):
    """Design one case in this interpreter; write result.json; never raise."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"case": case, "status": "started", "attempt_id": attempt_id,
              "probe_assertions": probe, "probe_date": probe_date if probe else None,
              "max_section_iter": max_section_iter,
              "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log = io.StringIO()
    t0 = time.perf_counter()
    try:
        import Structure_Parameters as sp
        import Geometry_Overrides as go
        from Design.Config import DesignConfig
        from Design.Design_Driver import load_or_create_design
        overrides = {"NUM_BAY_X": case["num_bay_x"], "NUM_BAY_Y": case["num_bay_y"], "NUM_FLOOR": case["num_floor"],
                     "STORY_H": 12.0 * case["story_height_ft"], "BAY_X": 12.0 * case["bay_x_width_ft"],
                     "BAY_Y": 12.0 * case["bay_y_width_ft"]}
        with contextlib.redirect_stdout(log):
            go.apply_geometry_overrides(overrides, variant_name=case["geometry_name"], emit=True)
            sp.apply_seismic_site(case["seismic_site"])
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
    except Exception as exc:                              # noqa: BLE001
        result.update({"status": "error", "elapsed_s": time.perf_counter() - t0,
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
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


def validate_saved(case, out_dir, probe, probe_date):
    from Design.Config import DesignConfig
    saved = saved_result(out_dir)
    if not saved or saved.get("status") != "designed":
        return saved
    cfg = probe_config(probe_date) if probe else DesignConfig.from_structure_parameters()
    return checked_result(case, out_dir, saved, expected_identity(case, cfg), probe, probe_date)


def _launch(python_exe, case, out_dir, probe, probe_date=None, log=print, max_section_iter=None):
    out_dir = Path(out_dir)
    with exclusive_lease(out_dir / ".worker.lease"):
        saved = saved_result(out_dir)
        if saved and saved.get("status") == "designed":
            return validate_saved(case, out_dir, probe, probe_date), True
        clear_interrupted_design(out_dir)
    attempt_id = uuid.uuid4().hex
    command = [python_exe, "-B", str(Path(__file__).resolve()), "--worker", json.dumps(case), str(out_dir),
               "--attempt-id", attempt_id]
    if probe:
        command.append("--probe-assertions")
        if probe_date:
            command += ["--probe-date", probe_date]
    if max_section_iter is not None:
        command += ["--max-section-iter", str(int(max_section_iter))]
    # Same child environment the generation scheduler gives its workers.
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
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
              "probe_assertions": probe, "probe_date": probe_date if probe else None,
              "error": outcome, "stderr_tail": stderr[-2000:], "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result, False


def summarize(results, root, probe, cases=None, probe_date=None):
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
                result = validate_saved(case, root / cid, probe, probe_date) or {
                    "case": case, "status": "error", "error": "Saved result is missing"}
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
        plan = f"{c['num_bay_x']}x{c['num_bay_y']}x{c['num_floor']} {c['bay_x_width_ft']}x{c['bay_y_width_ft']} ft / {c['story_height_ft']} ft"
        if r.get("status") != "designed":
            lines.append(f"| {c['case_id']} | {plan} | {c['seismic_site']} | {str(r.get('status', '')).upper()} | | | | | | "
                         f"{str(r.get('error', ''))[:80]} | | | | | | | | {r.get('host', '')} |")
            continue
        s = r["sections"]
        period = r.get("model_period_sec")
        lines.append(f"| {c['case_id']} | {plan} | {c['seismic_site']} | {'accepted' if r['accepted'] else 'open'} | "
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
    (root / "summary.json").write_text(json.dumps({"probe_assertions": probe, "cases": rows}, indent=1, default=str),
                                       encoding="utf-8")
    return lines


def load_or_write_plan(root, cases, args):
    """plan.json is written once per root; later launches must build the same plan.

    A different SHA means this machine has different code, RANGES or
    arguments from whoever created the root -- its case_0007 would be a
    different building. The PROBE stamp date is fixed here so every case of
    the run carries one request identity.
    """
    sha = plan_sha256(cases)
    path = root / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("plan_sha256") != sha:
            raise SystemExit(f"plan SHA mismatch: this launch builds {sha} but {path} holds {plan.get('plan_sha256')}. "
                             "Stop: git pull, check RANGES/SEISMIC_SITES and --count/--seed/--geometry-offset, or use a new root.")
    else:
        if args.probe_assertions and not args.probe_date:
            raise SystemExit("New PROBE roots require --probe-date YYYY-MM-DD; use the same date on all devices.")
        plan = {"seed": args.seed, "geometry_offset": args.geometry_offset, "count": args.count, "plan_sha256": sha,
                "probe_assertions": args.probe_assertions,
                "probe_date": args.probe_date if args.probe_assertions else None,
                "max_section_iter": args.max_section_iter,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "created_on": socket.gethostname(),
                "cases": cases, "launches": []}
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
        "case_ids": args.case_ids, "workers": args.workers, "summarize_only": args.summarize_only})
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
    parser.add_argument("--workers", type=int, default=4)
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
    args = parser.parse_args(argv)
    if args.worker:
        case = json.loads(args.worker[0])
        result = run_worker(case, args.worker[1], args.probe_assertions, args.probe_date, args.attempt_id, args.max_section_iter)
        print(json.dumps({k: result.get(k) for k in ("status", "elapsed_s", "accepted", "counts", "error")}, default=str))
        return 0
    if args.workers <= 0 or args.sites == [] or args.case_ids == []:
        parser.error("workers must be positive; explicit sites/case-ids must not be empty")
    if any(v is not None and v < 1 for v in (args.case_start, args.case_end)):
        parser.error("case-start and case-end must be positive")
    if args.case_start is not None and args.case_end is not None and args.case_start > args.case_end:
        parser.error("case-start must not exceed case-end")
    cases = plan_cases(args.count, seed=args.seed, geometry_offset=args.geometry_offset,
                       seismic_sites=tuple(args.sites) if args.sites is not None else SEISMIC_SITES)
    sha = plan_sha256(cases)
    print(f"Plan: {len(cases)} cases (seed {args.seed}, offset {args.geometry_offset}); "
          f"hazards {list(args.sites) if args.sites else list(SEISMIC_SITES)}")
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
    plan = load_or_write_plan(root, cases, args)
    probe_date = plan.get("probe_date")

    run_log = root / "run_log.txt"

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    if args.summarize_only:
        results = [saved_result(root / c["case_id"]) or {"case": c, "status": "missing"} for c in cases]
        lines = summarize(results, root, bool(plan.get("probe_assertions")), cases, probe_date)
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
    if not cases:
        raise ValueError("Requested case range selects no cases")
    if stop_file.exists():
        stop_file.unlink()
        log(f"cleared {STOP_NAME} left by an earlier stop request")
    log(f"launch on {socket.gethostname()} git {git_head() or '?'}: {len(cases)} cases "
        f"({cases[0]['case_id']}..{cases[-1]['case_id']}), {args.workers} workers, root {root}, "
        f"{'PROBE assertions dated ' + str(probe_date) if args.probe_assertions else 'committed Design/Config.py'}, "
        f"plan SHA {sha}")

    def run(case):
        if stop_file.exists():
            saved = saved_result(root / case["case_id"])
            if saved and saved.get("status") == "designed":
                return saved
            return {"case": case, "status": "missing", "host": socket.gethostname()}
        t0 = time.perf_counter()
        try:
            result, cached = _launch(args.python_exe, case, root / case["case_id"], args.probe_assertions, probe_date, log=log,
                                     max_section_iter=plan.get("max_section_iter"))
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
    lines = summarize(results, root, args.probe_assertions, cases, probe_date)
    status = Counter(r.get("status") for r in results)
    log(f"{'stopped on request' if stop_file.exists() else 'finished slice'}: {dict(status)}; summary {root / 'summary.md'}")
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
