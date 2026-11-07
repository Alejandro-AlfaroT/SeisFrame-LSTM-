"""Native response regression of the frozen IMK V2 calibration (verification C, 2026-10-29).

For each selected spring the saved design is rebuilt through the production path (no research adapter) and the
production-created material replays the archived fresh probes with the study's own protocol code (probes.py
``simulate`` and fema_probes ``monotonic``): monotonic in both signs, the FEMA 461 and 2015 NCREE protocols under
nominal half-member chord control (zero-length IMK spring in series with the elastic Kb, exact SP chord constraint)
and the legacy local-spring FEMA protocol. Each protocol keeps its archived amplitudes, cycles, control coordinate,
accepted (refined) step and the zero-history initial state; nothing is re-refined. The replay is compared with the
archived curves.npz and probe.json: every curve column, the zero-moment event (coordinates, level, cycle, leg), the
excursion peaks and endpoint moments, steps, recoveries and equilibrium error.

    python -X utf8 -B -I tools/imk_frozen_v2/probe_replay.py all --study <study root> --out <new dir>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as ev                                                   # noqa: E402

import numpy as np                                                      # noqa: E402

# beam 173 end j (reversed, asymmetric), column 74 end j (symmetric), beam 156 end i (not reversed), beam 161 end j,
# a symmetric column of another design, and an X-direction beam of a third design.
DEFAULT_SPRINGS = (("case_0064", 3017325), ("case_0064", 3007425), ("case_0064", 3015615), ("case_0064", 3016125),
                   ("case_0035", 3008525), ("case_0013", 3031915))
RESULT_KEYS = ("excursion_peaks", "endpoint_moments", "steps", "max_equilibrium_error_over_My", "recovery_steps",
               "final_spring_work", "final_control_work", "completed", "last_level", "final_control", "stopping")


def _load_study_probes(study, harness, out):
    # Matplotlib is configured here first, with its cache under the output directory, so the study's helper modules
    # (which point MPLCONFIGDIR into the evidence folder at import) cannot write into the evidence.
    os.environ["MPLCONFIGDIR"] = str(out / "mpl-config")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401
    sys.path.insert(1, str(harness))
    spec = importlib.util.spec_from_file_location("study_probes", study / "probes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import fema_probes
    return module, fema_probes


def _production_material(study, case, tag):
    from Model import IMK_Materials as mm
    from Analysis.Fixed_Design_Diagnostics import install_record
    from Model.Build_Model import build_model
    from Model.IMK_Hinges import hinge_registry
    import openseespy.opensees as ops
    if getattr(mm, "_research_half_member_installed", False):
        raise SystemExit("the research adapter is installed in this process")
    record = json.loads((study / "mce_average" / "inputs" / case / "grouped_design.json").read_bytes())
    install_record(record, case, "v2_nonlinear_flexure_screening_v1")
    build_model()
    found = [material for entry in hinge_registry().values() for mats in entry["installed_materials"].values()
             for material in mats.values() if material.get("material_tag") == tag]
    ops.wipe()
    if len(found) != 1:
        raise SystemExit(f"{case}: material {tag} found {len(found)} times")
    material = found[0]
    # The probe code reads Kb from the record it is given; the production record keeps it in the frozen calibration.
    kb = material["provenance"]["member_spring_calibration"]["reference"]["Kb_kip_in_per_rad"]
    return material, {**material, "provenance": {**material["provenance"], "research_mapping": {"Kb": kb}}}


def _curve_diff(new, old):
    """Column-wise differences; where the stored rows differ in number, the common leading rows are compared and the
    rows only one side holds are listed. Spring-rotation differences are located: in rows at exactly zero moment
    (after the material reached zero) the chord-controlled spring rotation is fixed only to the equilibrium tolerance
    (|chord - u| Kb < 1e-6 My), so the solved value can differ there without any material difference."""
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    report = {"shape_new": list(new.shape), "shape_archived": list(old.shape), "same_shape": new.shape == old.shape,
              "bitwise_identical": bool(new.shape == old.shape and np.array_equal(new, old))}
    rows = min(len(new), len(old))
    if rows == 0 or new.ndim != 2 or old.ndim != 2 or new.shape[1] != old.shape[1]:
        return report
    a, b = new[:rows], old[:rows]
    diff = np.abs(a - b)
    report["common_rows"] = rows
    report["max_abs_diff_by_column"] = diff.max(axis=0).tolist()
    differing = np.flatnonzero(diff.max(axis=1) > 0)
    report["differing_rows"] = int(len(differing))
    if len(differing):
        columns = sorted(set(np.flatnonzero(diff[differing].max(axis=0) > 0).tolist()))
        report["differing_columns"] = columns
        report["differing_rows_all_at_zero_moment"] = bool(np.all(a[differing, 1] == 0) and np.all(b[differing, 1] == 0)) \
            if new.shape[1] > 1 else None
    if len(new) != len(old):
        longer, label = (new, "new") if len(new) > len(old) else (old, "archived")
        report["rows_only_in"] = {"side": label, "rows": longer[rows:].tolist()}
    return report


def one(args):
    study = ev.study_root(args.study)
    harness = ev.harness_root(study)
    out = Path(args.out).resolve()
    dest = out / args.case / str(args.tag)
    dest.mkdir(parents=True, exist_ok=False)
    probes, fema = _load_study_probes(study, harness, out)
    archived_dir = study / "probes" / args.case / str(args.tag)
    archived = json.loads((archived_dir / "probe.json").read_text(encoding="utf-8"))
    material, m = _production_material(study, args.case, args.tag)
    report = {"verification": "native response regression (C)", "case": args.case, "material_tag": args.tag,
              "archived": str(archived_dir), "runtime": ev.runtime(),
              "study_probe_code_sha256": {"probes.py": ev.sha256_file(study / "probes.py"),
                                          "fema_probes.py": ev.sha256_file(harness / "fema_probes.py")},
              "command_equal_to_archived": material["command"] == archived["material"]["command"],
              "member_spring_calibration": material["provenance"]["member_spring_calibration"]["calibration_id"],
              "initial_state": "fresh zero-history material, as archived", "protocols": {}}
    with np.load(archived_dir / "curves.npz") as curves:
        archived_curves = {key: curves[key] for key in curves.files}
    # The same protocols are also run with the ARCHIVED command in this process and runtime (the control). It
    # separates what the production parameters do from what today's re-execution of the archived probe does.
    control = archived["material"]
    arrays, control_arrays = {}, {}
    for sign, key in ((1, "monotonic_positive"), (-1, "monotonic_negative")):
        actual, error = fema.monotonic(m, sign)
        arrays[key] = actual
        control_arrays[key], _ = fema.monotonic(control, sign)
        report["protocols"][key] = {"curve": _curve_diff(actual, archived_curves[key]), "backbone_error_over_My": error,
                                    "production_equals_control": bool(np.array_equal(actual, control_arrays[key]))}
    for name, entry in archived["protocols"].items():
        protocol = entry["protocol"]
        started = time.time()
        args_ = (np.asarray(protocol["amplitudes"], dtype=float), protocol["cycles"], protocol["step"],
                 protocol["control"] == "reference_chord")
        array, result = probes.simulate(m, *args_)
        control_array, control_result = probes.simulate(control, *args_)
        arrays[name], control_arrays[name] = array, control_array
        old = entry["result"]
        report["protocols"][name] = {
            "protocol": protocol, "elapsed_sec": time.time() - started,
            "curve": _curve_diff(array, archived_curves[name]),
            "production_equals_control": bool(np.array_equal(array, control_array)) and result == control_result,
            "control_curve_versus_archive": _curve_diff(control_array, archived_curves[name]),
            "zero_moment_event": {"new": result["failure"], "archived": old["failure"],
                                  "equal": result["failure"] == old["failure"]},
            "results_equal": {key: result.get(key) == old.get(key) for key in RESULT_KEYS},
            "excursion_peak_max_abs_diff": float(np.max(np.abs(np.asarray(result["excursion_peaks"])
                                                               - np.asarray(old["excursion_peaks"]))))
            if len(result["excursion_peaks"]) == len(old["excursion_peaks"]) else None,
            "archived_verification": entry["verification"]}
    np.savez_compressed(dest / "curves.npz", **arrays)
    protocols = report["protocols"].values()
    report["production_equals_control"] = all(p["production_equals_control"] for p in protocols)
    report["events_and_results_equal_archive"] = (
        all(p.get("zero_moment_event", {}).get("equal", True) for p in protocols)
        and all(all(v for k, v in p.get("results_equal", {}).items() if k != "max_equilibrium_error_over_My")
                for p in protocols))
    report["bitwise_reproduction"] = (report["command_equal_to_archived"]
                                      and all(p["curve"]["bitwise_identical"] for p in protocols)
                                      and report["events_and_results_equal_archive"])
    ev.write_json(dest / "replay.json", report)
    print(json.dumps({"case": args.case, "tag": args.tag, "bitwise_reproduction": report["bitwise_reproduction"],
                      "production_equals_control": report["production_equals_control"],
                      "events": {k: v.get("zero_moment_event", {}).get("new") for k, v in report["protocols"].items()}},
                     default=str), flush=True)
    return 0


def all_springs(args):
    out = ev.new_output_dir(args.out)
    study = ev.study_root(args.study)
    if ev.under_study(out, study):
        raise SystemExit("the output directory must not be inside the approved study")
    (out / "logs").mkdir()
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "FOR_DISABLE_CONSOLE_CTRL_HANDLER": "1"}

    def launch(spring):
        case, tag = spring
        log = out / "logs" / f"{case}_{tag}.log"
        started = time.time()
        with log.open("w", encoding="utf-8") as stream:
            code = subprocess.run([sys.executable, "-X", "utf8", "-B", "-I", __file__, "one", "--study", str(study),
                                   "--out", str(out), "--case", case, "--tag", str(tag)],
                                  stdout=stream, stderr=subprocess.STDOUT, env=env).returncode
        return {"case": case, "tag": tag, "exit_code": code, "elapsed_sec": time.time() - started}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(launch, DEFAULT_SPRINGS))
    summary = {"verification": "native response regression (C)", "runs": runs, "springs": {}}
    for case, tag in DEFAULT_SPRINGS:
        path = out / case / str(tag) / "replay.json"
        if path.exists():
            report = json.loads(path.read_text())
            summary["springs"][f"{case}/{tag}"] = {
                "bitwise_reproduction": report["bitwise_reproduction"],
                "command_equal_to_archived": report["command_equal_to_archived"],
                "production_equals_control": report["production_equals_control"],
                "events_and_results_equal_archive": report["events_and_results_equal_archive"],
                "protocols": {name: {"curve_bitwise": p["curve"].get("bitwise_identical"),
                                     "shapes": [p["curve"]["shape_new"], p["curve"]["shape_archived"]],
                                     "max_abs_diff_by_column": p["curve"].get("max_abs_diff_by_column"),
                                     "differing_rows": p["curve"].get("differing_rows"),
                                     "differing_columns": p["curve"].get("differing_columns"),
                                     "differing_rows_all_at_zero_moment": p["curve"].get("differing_rows_all_at_zero_moment"),
                                     "rows_only_in": (p["curve"].get("rows_only_in") or {}).get("side"),
                                     "control_reproduces_archive": (p.get("control_curve_versus_archive") or {}).get("bitwise_identical"),
                                     "production_equals_control": p.get("production_equals_control"),
                                     "event_equal": p.get("zero_moment_event", {}).get("equal"),
                                     "event": p.get("zero_moment_event", {}).get("new")}
                              for name, p in report["protocols"].items()}}
    complete = bool(summary["springs"]) and len(summary["springs"]) == len(DEFAULT_SPRINGS)
    summary["all_bitwise"] = complete and all(s["bitwise_reproduction"] for s in summary["springs"].values())
    summary["all_production_equal_control"] = complete and all(s["production_equals_control"] for s in summary["springs"].values())
    summary["all_events_and_results_equal_archive"] = complete and all(
        s["events_and_results_equal_archive"] and s["command_equal_to_archived"] for s in summary["springs"].values())
    ev.write_json(out / "probe_replay_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("all_bitwise", "all_production_equal_control",
                                              "all_events_and_results_equal_archive")} | {"runs": runs},
                     indent=1), flush=True)
    return 0 if summary["all_production_equal_control"] and summary["all_events_and_results_equal_archive"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("one", "all"))
    parser.add_argument("--study", required=True, help="root folder of the approved study (manifest.json, profile.py)")
    parser.add_argument("--out", default=str(ev.DEFAULT_OUTPUT / "probes"))
    parser.add_argument("--case", choices=ev.CASES)
    parser.add_argument("--tag", type=int)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.action == "one":
        if not args.case or not args.tag:
            raise SystemExit("one needs --case and --tag")
        return one(args)
    return all_springs(args)


if __name__ == "__main__":
    sys.exit(main())
