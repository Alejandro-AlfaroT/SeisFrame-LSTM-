"""Declared diagnostic research batch: one nonlinear response-history run per designed structure.

This is NOT the production generation route. ``Design.SMRF_Qualification.GENERATION_RELEASE_READY`` is not
consulted or changed, no acceptance function is patched, and nothing here asserts a design check. The batch
runs the same model build, analysis and export code as production (``Ground_Motion_Main.run_one``) on saved
designs, and writes beside every run what a reader needs to decide what the run is worth: the design's
qualification state with every open and failed check id, the analysis profile, the source fingerprint, the
record pair and its file hashes, and the scaling. A surrogate trained on these runs emulates this specified
model; that is not a statement about physical structures.

Definition of the batch (pre-generation review, 2026-10-04):
  * one case = one structure with one ground-motion assignment and one nonlinear run;
  * both horizontal components of one recorded pair, X and Y together, relative timing kept, ONE common
    amplitude multiplier (the natural directional imbalance is preserved);
  * scale = alpha * Sa_design(T_ref) / sqrt(Sa_x_raw(T_ref) * Sa_y_raw(T_ref)), 5 %-damped pseudo-acceleration,
    the site's ASCE 7 design spectrum, T_ref = max(T_x, T_y) of the unscaled elastic model, each direction's
    period being the mode with the largest effective modal mass in that direction (never a mode picked by
    number, which could be torsional). T_ref is fixed before any shaking;
  * raw spectra are computed with multiplier 1; the multiplier written and applied is the TOTAL multiplier,
    and the catalog's own scale factor is not applied on top of it;
  * alpha in {0.5, 1.0, 1.5}, as near one third each as the count allows (the extra case goes to 1.0),
    balanced over hazard and height and assigned from a recorded seed. These are INPUT intensity groups, not
    promised response classes and not a code-compliant record-suite selection.

Execution: ``--plan-only`` writes and prints the plan; a run takes the cases of one shard (round robin over
the plan order) or named cases. Each case runs in its own process, claims its folder first (a second machine
or a second launch cannot run the same case), and is skipped when it already finished under the same plan.
A failed analysis keeps everything it wrote and is reported as failed; it is never retried into the same
folder and never counted as a completed run.

  python Data_Generation/Run_Research_Batch.py --manifest pilots/X/screening_plan.json --design-roots outputs/A outputs/B \\
         --output-root outputs/research_batch --plan-only
  python Data_Generation/Run_Research_Batch.py --output-root outputs/research_batch --design-roots outputs/A outputs/B \\
         --shard 1 --of 4 --workers 2
  python Data_Generation/Run_Research_Batch.py --output-root outputs/research_batch --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RC_DIR = Path(__file__).resolve().parents[1]
for _entry in (str(RC_DIR), str(Path(__file__).resolve().parent)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

PLAN_SCHEMA = "seisframe_v2_research_ntha_plan_v1"
PLAN_NAME = "ntha_plan.json"
IDENTITY_NAME = "run_identity.json"
STATUS_NAME = "run_status.json"
CLAIM_NAME = ".claim"
ALPHAS = (1.0, 0.5, 1.5)                       # cycle order: the extra case of an uneven count goes to 1.0, then 0.5
DEFAULT_SEED = 20261006
DEFAULT_RECORD_SET = "peer_mle_all"
DAMPING = 0.05
YIELD_TOLERANCE = 1e-6
FINAL_STATES = ("completed", "analysis_failed", "error", "not_run_design_not_usable")


# ---- small helpers ---------------------------------------------------------------------------------------
def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_identity():
    """Design-source aggregate (the same files a design record hashes) and the solver/library versions."""
    import importlib.metadata
    from Design import Design_Driver as driver
    import openseespy.opensees as ops
    files = driver.source_sha256()
    chain = {}
    for name in ("Ground_Motion_Main.py", "Data_Generation/Run_Research_Batch.py", "Data_Generation/Graph_Exporter.py",
                 "Loads/Ground_Motion.py"):
        chain[name] = hashlib.sha256((RC_DIR / name).read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()
    packages = {}
    for name in ("openseespy", "openseespywin", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"design_source_aggregate_sha256": _sha256_json(files), "design_source_files": len(files),
            "run_chain_sha256": chain, "opensees_version": str(ops.version()), "python": platform.python_version(),
            "packages": packages}


# ---- the scaling rule --------------------------------------------------------------------------------------
def design_spectrum_sa_g(period_sec, sds, sd1, tl):
    """ASCE 7 design response spectrum (11.4.6), g; no importance factor, no response modification."""
    if period_sec <= 0.0:
        raise ValueError("The reference period must be positive.")
    ts = sd1 / sds
    t0 = 0.2 * ts
    if period_sec < t0:
        return sds * (0.4 + 0.6 * period_sec / t0)
    if period_sec <= ts:
        return sds
    if period_sec <= tl:
        return sd1 / period_sec
    return sd1 * tl / period_sec ** 2


def pseudo_acceleration_g(acc_g, dt_sec, period_sec, damping=DAMPING):
    """5 %-damped pseudo-spectral acceleration of a linear oscillator, g (Newmark average acceleration)."""
    acc = np.asarray(acc_g, dtype=float)
    omega = 2.0 * np.pi / period_sec
    k, c = omega * omega, 2.0 * damping * omega
    k_eff = k + 2.0 * c / dt_sec + 4.0 / dt_sec ** 2
    u = v = 0.0
    a = -acc[0]
    peak = 0.0
    for i in range(1, len(acc)):
        dp = -(acc[i] - acc[i - 1]) + (4.0 / dt_sec + 2.0 * c) * v + 2.0 * a
        du = dp / k_eff
        dv = 2.0 * du / dt_sec - 2.0 * v
        da = 4.0 * (du - dt_sec * v) / dt_sec ** 2 - 2.0 * a
        u, v, a = u + du, v + dv, a + da
        peak = max(peak, abs(u))
    return peak * omega * omega


def directional_periods(modes):
    """T_x and T_y of the elastic model: in each direction, the valid mode with the largest effective modal mass.

    Returns the two periods, the mode numbers, their modal mass ratios and T_ref = max(T_x, T_y). A model whose
    modes carry no participation in a direction has no period there, and that is an error, not a default.
    """
    best = {}
    for mode in modes:
        participation = mode.get("participation") if mode.get("valid") else None
        if not participation:
            continue
        for axis in ("x", "y"):
            ratio = participation.get(f"modal_mass_ratio_{axis}")
            if ratio is not None and (axis not in best or ratio > best[axis]["modal_mass_ratio"]):
                best[axis] = {"mode": mode["mode"], "period_sec": mode["period"], "modal_mass_ratio": ratio}
    if set(best) != {"x", "y"} or min(best[a]["modal_mass_ratio"] for a in best) <= 0.0:
        raise RuntimeError("No translational mode found in both horizontal directions; the reference period is undefined.")
    reference = max(best["x"]["period_sec"], best["y"]["period_sec"])
    return {"x": best["x"], "y": best["y"], "reference_period_sec": reference,
            "reference_direction": "x" if best["x"]["period_sec"] >= best["y"]["period_sec"] else "y",
            "rule": "per direction, the valid mode with the largest effective modal mass; T_ref = max(T_x, T_y)",
            "same_mode_in_both_directions": best["x"]["mode"] == best["y"]["mode"]}


def scale_factor(alpha, design_sa_g, sa_x_raw_g, sa_y_raw_g):
    geometric_mean = float(np.sqrt(sa_x_raw_g * sa_y_raw_g))
    if not np.isfinite(geometric_mean) or geometric_mean <= 0.0:
        raise RuntimeError("The raw spectral accelerations are not positive; the pair cannot be scaled.")
    return alpha * design_sa_g / geometric_mean, geometric_mean


# ---- the plan ---------------------------------------------------------------------------------------------
def assign_alphas(cases, seed):
    """Intensity group of every case: cycled over the cases ordered by hazard, then height, ties by the seed."""
    rng = random.Random(f"{seed}:alpha")
    keyed = sorted(cases, key=lambda c: c["case_id"])
    tie = {c["case_id"]: rng.random() for c in keyed}
    ordered = sorted(keyed, key=lambda c: (c["seismic_site"], c["num_floor"], tie[c["case_id"]]))
    return {case["case_id"]: ALPHAS[index % len(ALPHAS)] for index, case in enumerate(ordered)}


def record_pairs(record_set):
    """The usable horizontal pairs of a record set with what identifies and groups them."""
    from Loads import Ground_Motion as gm
    pairs = []
    for key, row_x, row_y in gm.ground_motion_pair_rows(set_name=record_set):
        files = {}
        for axis, row in (("x", row_x), ("y", row_y)):
            path = gm._manifest_path(row, "processed_file")
            if path is None or not path.exists():
                path = gm._manifest_path(row, "raw_file")
            files[axis] = {"record_id": row["record_id"], "file": None if path is None else path.relative_to(gm.GROUND_MOTION_DIR).as_posix(),
                           "sha256": _sha256_file(path) if path is not None and path.exists() else None,
                           "dt_sec": float(row["dt_sec"]), "npts": int(row["npts"]),
                           "component_angle_deg": row.get("component_angle_deg"),
                           "catalog_scale_factor_not_applied": row.get("scale_factor")}
        pairs.append({"pair_key": key, "x": files["x"], "y": files["y"],
                      "event_name": row_x.get("event_name"), "event_year": row_x.get("event_year"),
                      "station_id": row_x.get("station_id"), "magnitude": row_x.get("magnitude"),
                      # Runs that share this group share an earthquake and station: keep them on one side of a train/test split.
                      "record_group": f"{row_x.get('event_name')}|{row_x.get('station_id')}"})
    return pairs


def assign_records(cases, pairs, seed):
    """One pair per case, drawn without replacement from a seeded shuffle; the shuffle repeats only when it runs out."""
    if not pairs:
        raise RuntimeError("The record set has no usable horizontal pair.")
    rng = random.Random(f"{seed}:records")
    order = list(range(len(pairs)))
    rng.shuffle(order)
    return {case["case_id"]: pairs[order[index % len(order)]] for index, case in enumerate(sorted(cases, key=lambda c: c["case_id"]))}


def build_plan(manifest_path, design_roots, record_set=DEFAULT_RECORD_SET, seed=DEFAULT_SEED):
    from Design.Screening_Manifest import load_manifest
    manifest = load_manifest(manifest_path)
    cases = manifest["cases"] if isinstance(manifest, dict) else manifest
    profile_id = manifest.get("profile_id") if isinstance(manifest, dict) else None
    alphas = assign_alphas(cases, seed)
    pairs = record_pairs(record_set)
    records = assign_records(cases, pairs, seed)
    rows = []
    for index, case in enumerate(sorted(cases, key=lambda c: c["case_id"])):
        # The plan pins the design each case will run: every machine must hold the same design.json.
        design_dir = find_design(case["case_id"], design_roots)
        pinned = None
        if design_dir is not None:
            state = design_state(_read_json(design_dir / "result.json"))
            pinned = {key: state[key] for key in ("status", "design_sha256", "profile_id", "profile_sha256")}
            pinned.update(failed_checks=len(state["failed_check_ids"]), open_checks=len(state["open_check_ids"]))
        rows.append({"index": index, "case": case, "alpha": alphas[case["case_id"]], "record": records[case["case_id"]],
                     "design": pinned})
    used = [row["record"]["pair_key"] for row in rows]
    body = {"schema": PLAN_SCHEMA,
            "status": "declared diagnostic research batch; not production generation, not a qualification",
            "case_definition": "one structure, one recorded horizontal pair (X and Y together), one common multiplier, one nonlinear run",
            "manifest": {"name": Path(manifest_path).name, "sha256": _sha256_file(manifest_path), "cases": len(cases)},
            "profile_id": profile_id,
            "designs_pinned": sum(1 for row in rows if row["design"] is not None),
            "seed": seed, "record_set": record_set, "record_pairs_available": len(pairs),
            "record_pairs_reused": len(used) - len(set(used)),
            "scaling": {"rule": "scale = alpha * Sa_design(T_ref) / sqrt(Sa_x_raw(T_ref) * Sa_y_raw(T_ref))",
                        "damping_ratio": DAMPING, "spectrum": "ASCE 7 design response spectrum of the case's site (SDS, SD1, TL)",
                        "reference_period": "max(T_x, T_y) of the unscaled elastic model; each from the mode with the largest "
                                            "effective modal mass in its direction; fixed before shaking",
                        "multiplier": "one common TOTAL multiplier on both components; the catalog scale factor is not applied",
                        "alphas": sorted(set(alphas.values())),
                        "alpha_counts": {str(a): sum(1 for v in alphas.values() if v == a) for a in sorted(set(alphas.values()))},
                        "alpha_assignment": "cycled over cases ordered by hazard then height, ties by the seed",
                        "limits": "input intensity groups; not response classes, not a code-compliant record-suite selection"},
            "analysis": {"damping_ratio": 0.05, "rayleigh_mode_i": 0, "rayleigh_mode_j": 2, "dt_factor": 1.0},
            "split_guidance": "runs sharing record.record_group share an earthquake and station; keep each group on one side of a split",
            "cases": rows}
    return {**body, "plan_sha256": _sha256_json(body)}


def load_or_write_plan(root, args):
    root = Path(root)
    path = root / PLAN_NAME
    if path.exists():
        plan = _read_json(path)
        body = {key: value for key, value in plan.items() if key != "plan_sha256"}
        if _sha256_json(body) != plan.get("plan_sha256"):
            raise RuntimeError(f"{path} does not match its own SHA-256; it was edited after it was written.")
        if args.manifest and args.design_roots:
            fresh = build_plan(args.manifest, args.design_roots, args.record_set, args.seed)
            if fresh["plan_sha256"] != plan["plan_sha256"]:
                raise RuntimeError("The requested manifest, designs, record files, record set or seed give a different plan from "
                                   f"the one saved in {path}. A changed input needs a new output root.")
        return plan
    if not args.manifest or not args.design_roots:
        raise RuntimeError("A new output root needs --manifest and --design-roots.")
    plan = build_plan(args.manifest, args.design_roots, args.record_set, args.seed)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(path, plan)
    return plan


# ---- one case --------------------------------------------------------------------------------------------
def find_design(case_id, design_roots):
    found = [Path(root) / case_id for root in design_roots
             if (Path(root) / case_id / "design.json").exists() and (Path(root) / case_id / "result.json").exists()]
    if len(found) > 1:
        raise RuntimeError(f"{case_id} has a design in more than one design root: {[str(p) for p in found]}")
    return found[0] if found else None


def design_state(result):
    """What the saved design says about itself; copied, never re-judged or asserted here."""
    counts = result.get("counts") or {}
    return {"status": result.get("status"), "accepted": result.get("accepted"), "counts": counts,
            "failed_check_ids": result.get("fail_ids") or [], "open_check_ids": result.get("not_evaluated_ids") or [],
            "probe_assertions": result.get("probe_assertions"), "probe_date": result.get("probe_date"),
            "design_sha256": result.get("design_sha256"), "request_sha256": result.get("request_sha256"),
            "profile_id": result.get("profile_id"), "profile_sha256": result.get("profile_sha256"),
            "sections": result.get("sections"), "stop_reason": result.get("stop_reason")}


def design_usable(state, include_failed=False):
    if state["status"] != "designed":
        return False, f"design status is {state['status']!r}"
    if state["failed_check_ids"] and not include_failed:
        return False, f"{len(state['failed_check_ids'])} failed design checks"
    return True, ""


def history_checks(run_dir):
    """Exported hinge histories: present, finite, aligned, with a signed absolute yield count by member class."""
    from Analysis.Hinge_Moment_Rotation import read_hinge_moment_rotation
    arrays, rows, schema = read_hinge_moment_rotation(run_dir)
    if arrays is None:
        return {"available": False, "reason": schema.get("reason")}
    rotation, moment = arrays["rotation_rad"], arrays["moment_kip_in"]
    n_rows = len(arrays["time_sec"])
    aligned = (rotation.shape == moment.shape == (n_rows, len(rows))
               and len(arrays["commit_count"]) == n_rows and len(arrays["scheduled_step"]) == n_rows)
    time_increasing = bool(n_rows < 2 or np.all(np.diff(arrays["time_sec"]) > 0))
    gravity_differs = None
    if n_rows:
        gravity_differs = float(np.nanmax(np.abs(rotation[0] - arrays["gravity_rotation_rad"])))
    yielded, counted = {}, {}
    for k, row in enumerate(rows):
        try:
            positive, negative = float(row["yield_moment_positive_kip_in"]), float(row["yield_moment_negative_kip_in"])
        except (TypeError, ValueError):
            continue
        strong = row["member_class"] == "column" or row["local_axis"] == "y"
        key = f"{row['member_class']}_{'strong' if strong else 'weak'}_axis"
        counted[key] = counted.get(key, 0) + 1
        if n_rows and (np.nanmax(moment[:, k]) >= positive * (1.0 - YIELD_TOLERANCE)
                       or np.nanmin(moment[:, k]) <= -negative * (1.0 - YIELD_TOLERANCE)):
            yielded[key] = yielded.get(key, 0) + 1
    return {"available": True, "rows": int(n_rows), "springs": int(len(rows)), "aligned": bool(aligned),
            "time_strictly_increasing": time_increasing,
            "non_finite_values": int((~np.isfinite(rotation)).sum() + (~np.isfinite(moment)).sum()),
            "first_row_minus_gravity_state_max_abs_rad": gravity_differs,
            "yield_definition": f"moment reaches +My(+) or -My(-) within {YIELD_TOLERANCE:g} relative at any stored step",
            "springs_by_class": counted, "yielded_by_class": yielded,
            "energy_anchor_policies": sorted({row.get("energy_anchor_policy", "") for row in rows}),
            "truncated": bool((schema.get("status") or {}).get("truncated"))}


def _folder_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def claim(case_dir):
    """Take the case folder for this process. A folder that is already claimed or already final is not taken."""
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(case_dir / CLAIM_NAME, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, stream)
    return True


def run_case(root, case_id, design_roots, include_failed=False, smoke_duration_sec=None):
    """Worker: configure, install the design, scale the pair, run, export, check. Returns the status dictionary."""
    root = Path(root)
    plan = _read_json(root / PLAN_NAME)
    row = next(item for item in plan["cases"] if item["case"]["case_id"] == case_id)
    case_dir = root / case_id
    started = time.perf_counter()
    status = {"case_id": case_id, "plan_sha256": plan["plan_sha256"], "alpha": row["alpha"],
              "record_pair": row["record"]["pair_key"], "host": socket.gethostname(),
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke_test": smoke_duration_sec is not None}
    try:
        import Structure_Parameters as sp
        import Ground_Motion_Main as gm_main
        import openseespy.opensees as ops
        from Analysis import Hinge_Hysteresis_Diagnostic as hd
        from Analysis.Modal import run_modal_analysis
        from Design import SMRF_Qualification as qualification
        from Design import Verify_Designs as verify
        from Design.Grouped_Record import apply_record
        from Loads.Ground_Motion import _to_g, load_ground_motion_record
        from Model import Analysis_Profile as ap
        from Model import Roof_Extension as roof
        from Model.Build_Model import build_model

        design_dir = find_design(case_id, design_roots)
        if design_dir is None or row.get("design") is None:
            status.update(state="not_run_design_not_usable",
                          reason="no design.json and result.json in the design roots" if design_dir is None
                          else "the plan pins no design for this case")
            return status
        result = _read_json(design_dir / "result.json")
        state = design_state(result)
        if state["design_sha256"] != row["design"]["design_sha256"]:
            raise RuntimeError("The design in the design roots is not the design the plan pins (different SHA-256).")
        usable, reason = design_usable(state, include_failed)
        identity = {"schema": "seisframe_v2_research_run_identity_v1", "diagnostic_research_batch": True,
                    "production_release": False,
                    "generation_release_ready_flag": bool(qualification.GENERATION_RELEASE_READY),
                    "gate": "the production gate was not consulted and no acceptance function was changed: this is a declared "
                            "diagnostic run of a saved design, with the design's own qualification state copied below",
                    "case": row["case"], "plan_sha256": plan["plan_sha256"], "design_dir": str(design_dir),
                    "design": state, "smoke_test": status["smoke_test"]}
        if not usable:
            _write_json(case_dir / IDENTITY_NAME, identity)
            status.update(state="not_run_design_not_usable", reason=reason, design=state)
            return status
        if plan.get("profile_id") and state["profile_id"] != plan["profile_id"]:
            raise RuntimeError(f"The design was made under profile {state['profile_id']!r}; the plan names {plan['profile_id']!r}.")
        if _sha256_file(design_dir / "design.json") != state["design_sha256"]:
            raise RuntimeError("design.json does not match the SHA-256 recorded in result.json.")
        record = _read_json(design_dir / "design.json")

        verify.configure_case(row["case"], state["profile_id"])
        apply_record(record)
        profile = ap.apply_profile(state["profile_id"])
        source = source_identity()
        recorded_source = (record.get("request_identity") or {}).get("source_sha256") or record.get("source_sha256")
        identity.update(profile={"id": profile["id"], "sha256": profile["sha256"], "state": profile["state"],
                                 "declarations": profile["declarations"]},
                        source=source,
                        design_source_matches_this_tree=(None if not isinstance(recorded_source, dict) else
                                                         _sha256_json(recorded_source) == source["design_source_aggregate_sha256"]),
                        roof_column_extension=roof.declaration())
        if state["profile_sha256"] and state["profile_sha256"] != profile["sha256"]:
            raise RuntimeError("The profile identity of this source tree differs from the one the design was made under.")

        # Reference period of the UNSCALED elastic model, fixed before any shaking.
        ops.wipe()
        build_model()
        periods = directional_periods(run_modal_analysis())
        ops.wipe()
        t_ref = periods["reference_period_sec"]
        design_sa = design_spectrum_sa_g(t_ref, sp.ASCE_SDS, sp.ASCE_SD1, sp.ASCE_TL)
        raw, files = {}, {}
        for axis in ("x", "y"):
            entry = row["record"][axis]
            unscaled = load_ground_motion_record(entry["record_id"], scale_factor=1.0)
            actual = _sha256_file(unscaled.source_path) if unscaled.source_path else None
            if entry["sha256"] and actual and actual != entry["sha256"]:
                raise RuntimeError(f"Record file of {entry['record_id']} differs from the plan (SHA-256 {actual[:12]} vs {entry['sha256'][:12]}).")
            files[axis] = {"record_id": entry["record_id"], "sha256": actual, "planned_sha256": entry["sha256"],
                           "dt_sec": unscaled.dt_sec, "npts": unscaled.npts}
            raw[axis] = pseudo_acceleration_g(_to_g(unscaled.acceleration, unscaled.units), unscaled.dt_sec, t_ref)
        factor, geometric_mean = scale_factor(row["alpha"], design_sa, raw["x"], raw["y"])
        record_x = load_ground_motion_record(row["record"]["x"]["record_id"], scale_factor=factor)
        record_y = load_ground_motion_record(row["record"]["y"]["record_id"], scale_factor=factor)
        scaled_pga = {"x": record_x.pga_g, "y": record_y.pga_g}        # of the whole record, before any smoke truncation
        if smoke_duration_sec is not None:                 # labelled truncation for a mechanics check; never a batch run
            for item in (record_x, record_y):
                item.acceleration = item.acceleration[:max(2, int(round(smoke_duration_sec / item.dt_sec)) + 1)]
        scaling = {"alpha": row["alpha"], "periods": periods, "reference_period_sec": t_ref,
                   "site": {"label": getattr(sp, "SEISMIC_SITE_LABEL", None), "sds": sp.ASCE_SDS, "sd1": sp.ASCE_SD1, "tl": sp.ASCE_TL},
                   "design_sa_at_reference_period_g": design_sa, "target_geometric_mean_sa_g": row["alpha"] * design_sa,
                   "raw_sa_at_reference_period_g": raw, "raw_geometric_mean_sa_g": geometric_mean,
                   "total_multiplier": factor, "catalog_scale_factor_applied": False,
                   "scaled_pga_g": scaled_pga,
                   "scaled_sa_at_reference_period_g": {axis: factor * raw[axis] for axis in raw},
                   "records": files, "record_group": row["record"]["record_group"], "damping_ratio": DAMPING,
                   "truncated_to_sec": smoke_duration_sec}
        identity["scaling"] = scaling
        _write_json(case_dir / IDENTITY_NAME, identity)

        run_args = SimpleNamespace(**plan["analysis"])
        summary = gm_main.run_one(record_x, record_y, run_args, case_dir)
        installed = hd.verify_installed_design(record)
        analysis = summary.get("status") or {}
        checks = history_checks(case_dir)
        failed = bool(analysis.get("failed")) or bool(checks.get("truncated"))
        problems = []
        if not installed.get("consistent"):
            problems.append("the installed model does not match the design record")
        if checks.get("available") and (not checks["aligned"] or checks["non_finite_values"] or not checks["time_strictly_increasing"]):
            problems.append("exported hinge histories are not aligned, finite and time-ordered")
        if not checks.get("available"):
            problems.append("no hinge moment-rotation history was exported")
        drift = summary.get("max_story_drift_resultant") or {}
        status.update(state="analysis_failed" if failed else "error" if problems else "completed", problems=problems,
                      completed_steps=analysis.get("completed_steps"), requested_steps=analysis.get("npts_requested"),
                      failed_step=analysis.get("failed_step"),
                      peak_story_drift_ratio=drift.get("peak_drift_resultant_ratio"), peak_drift_story=drift.get("story"),
                      total_multiplier=factor, reference_period_sec=t_ref, design=state,
                      installed_design_matches_record=installed.get("consistent"), histories=checks,
                      training_eligible=(not failed and not problems and smoke_duration_sec is None))
    except Exception as exc:                               # noqa: BLE001 -- everything is kept and reported
        status.update(state="error", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        status["elapsed_sec"] = time.perf_counter() - started
        status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if case_dir.exists():
            status["output_bytes"] = _folder_bytes(case_dir)
    return status


def worker_main(root, case_id, design_roots, include_failed, smoke_duration_sec):
    case_dir = Path(root) / case_id
    status = run_case(root, case_id, design_roots, include_failed, smoke_duration_sec)
    _write_json(case_dir / STATUS_NAME, status)
    print(json.dumps({key: status.get(key) for key in ("case_id", "state", "reason", "completed_steps", "requested_steps",
                                                        "peak_story_drift_ratio", "elapsed_sec")}, default=str), flush=True)
    return 0 if status["state"] == "completed" else 2


# ---- the launcher ------------------------------------------------------------------------------------------
def case_state(root, case_id):
    """'final:<state>', 'claimed' (running, or interrupted before it wrote a status) or 'open'."""
    case_dir = Path(root) / case_id
    if (case_dir / STATUS_NAME).exists():
        return "final:" + str(_read_json(case_dir / STATUS_NAME).get("state"))
    if (case_dir / CLAIM_NAME).exists():
        return "claimed"
    return "open"


def select_cases(plan, args):
    rows = plan["cases"]
    if args.case_ids:
        wanted = set(args.case_ids)
        missing = wanted - {row["case"]["case_id"] for row in rows}
        if missing:
            raise RuntimeError(f"Not in the plan: {sorted(missing)}")
        return [row["case"]["case_id"] for row in rows if row["case"]["case_id"] in wanted]
    if args.shard is not None:
        if not args.of or not 1 <= args.shard <= args.of:
            raise RuntimeError("--shard K needs --of N with 1 <= K <= N.")
        return [row["case"]["case_id"] for row in rows if row["index"] % args.of == args.shard - 1]
    return [row["case"]["case_id"] for row in rows]


def launch(root, case_id, args):
    case_dir = Path(root) / case_id
    if not claim(case_dir):
        return {"case_id": case_id, "state": "skipped_already_claimed"}
    command = [args.python_exe, "-X", "utf8", "-B", str(Path(__file__).resolve()), "--output-root", str(root), "--worker", case_id,
               "--design-roots", *[str(Path(p).resolve()) for p in args.design_roots]]
    if args.include_failed_designs:
        command.append("--include-failed-designs")
    if args.smoke_duration_sec is not None:
        command += ["--smoke-duration-sec", str(args.smoke_duration_sec)]
    with open(case_dir / "log.txt", "w", encoding="utf-8") as log, open(case_dir / "stderr.txt", "w", encoding="utf-8") as err:
        code = subprocess.run(command, cwd=str(RC_DIR), stdout=log, stderr=err).returncode
    if not (case_dir / STATUS_NAME).exists():              # the process died before it could report
        _write_json(case_dir / STATUS_NAME, {"case_id": case_id, "state": "error", "reason": f"worker exited with code {code} and no status",
                                             "host": socket.gethostname(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    return _read_json(case_dir / STATUS_NAME)


SUMMARY_COLUMNS = ("case_id", "state", "training_eligible", "alpha", "record_pair", "reference_period_sec", "total_multiplier",
                   "completed_steps", "requested_steps", "peak_story_drift_ratio", "peak_drift_story", "beam_strong_yielded",
                   "beam_strong_springs", "column_yielded", "column_springs", "open_checks", "failed_checks", "elapsed_min",
                   "output_mb", "host", "reason")


def summarize(root):
    """batch_status.csv / .json from the case folders: every case of the plan, run or not, with its state."""
    root = Path(root)
    plan = _read_json(root / PLAN_NAME)
    rows = []
    for item in plan["cases"]:
        case_id = item["case"]["case_id"]
        state = case_state(root, case_id)
        status = _read_json(root / case_id / STATUS_NAME) if state.startswith("final:") else {}
        histories = status.get("histories") or {}
        yielded, springs = histories.get("yielded_by_class") or {}, histories.get("springs_by_class") or {}
        design = status.get("design") or {}
        rows.append({"case_id": case_id, "state": status.get("state", state), "training_eligible": bool(status.get("training_eligible")),
                     "alpha": item["alpha"], "record_pair": item["record"]["pair_key"],
                     "reference_period_sec": status.get("reference_period_sec"), "total_multiplier": status.get("total_multiplier"),
                     "completed_steps": status.get("completed_steps"), "requested_steps": status.get("requested_steps"),
                     "peak_story_drift_ratio": status.get("peak_story_drift_ratio"), "peak_drift_story": status.get("peak_drift_story"),
                     "beam_strong_yielded": yielded.get("beam_strong_axis", 0) if histories else None,
                     "beam_strong_springs": springs.get("beam_strong_axis"),
                     "column_yielded": yielded.get("column_strong_axis", 0) if histories else None,
                     "column_springs": springs.get("column_strong_axis"),
                     "open_checks": ";".join(design.get("open_check_ids") or []), "failed_checks": ";".join(design.get("failed_check_ids") or []),
                     "elapsed_min": None if status.get("elapsed_sec") is None else status["elapsed_sec"] / 60.0,
                     "output_mb": None if status.get("output_bytes") is None else status["output_bytes"] / 1e6,
                     "host": status.get("host"), "reason": status.get("reason") or "; ".join(status.get("problems") or [])})
    counts = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    with (root / "batch_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"plan_sha256": plan["plan_sha256"], "cases": len(rows), "states": counts,
               "training_eligible": sum(1 for row in rows if row["training_eligible"]),
               "note": "training_eligible is a completed, untruncated, consistent run of this declared diagnostic batch; it is "
                       "not a production acceptance"}
    _write_json(root / "batch_status.json", {**summary, "rows": rows})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default=None, help="Screening plan of the designed cases (needed for a new output root).")
    parser.add_argument("--design-roots", nargs="+", default=None, help="Folders holding case_NNNN/design.json and result.json.")
    parser.add_argument("--record-set", default=DEFAULT_RECORD_SET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--of", type=int, default=None)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--include-failed-designs", action="store_true",
                        help="Also run designs that carry failed checks (recorded as such). Default: they are listed and not run.")
    parser.add_argument("--smoke-duration-sec", type=float, default=None,
                        help="Mechanics check only: run this many seconds of the record. The run is marked and never training-eligible.")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(args.output_root)

    if args.worker:
        return worker_main(root, args.worker, args.design_roots, args.include_failed_designs, args.smoke_duration_sec)
    if args.summarize_only:
        print(json.dumps(summarize(root), indent=1))
        return 0
    plan = load_or_write_plan(root, args)
    print(f"Plan {plan['plan_sha256'][:16]}: {len(plan['cases'])} cases, profile {plan['profile_id']}, record set {plan['record_set']} "
          f"({plan['record_pairs_available']} pairs, {plan['record_pairs_reused']} reused), alpha counts {plan['scaling']['alpha_counts']}")
    if args.plan_only:
        return 0
    if not args.design_roots:
        raise RuntimeError("A run needs --design-roots (where this machine keeps the pinned designs).")
    selected = select_cases(plan, args)
    todo = [case_id for case_id in selected if case_state(root, case_id) == "open"]
    held = [case_id for case_id in selected if case_state(root, case_id) == "claimed"]
    print(f"{len(selected)} cases selected: {len(todo)} to run, {len(selected) - len(todo) - len(held)} already final, "
          f"{len(held)} claimed by another launch or interrupted (not rerun: {held[:10]}{' ...' if len(held) > 10 else ''})")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for status in pool.map(lambda case_id: launch(root, case_id, args), todo):
            print(f"  {status.get('case_id')}: {status.get('state')} {status.get('reason') or ''}", flush=True)
    print(json.dumps(summarize(root), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
