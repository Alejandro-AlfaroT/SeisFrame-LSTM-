"""Frame integration regression of the frozen IMK V2 calibration (verification D, 2026-10-29).

``run`` reruns one saved design at one intensity through the study's own saved-design earthquake harness
(v6-mcer-hinges preflight.prepare and run_earthquake.run: install_record under the V2 profile, gravity and modal
state, the same RSN864 pair, the saved average-intensity scale, dt factor 0.5, UmfPack, 5 % Rayleigh) WITHOUT the
research adapter, so every member spring comes from the production calibration. It writes only into a new output
directory. ``compare`` sets the result beside the approved batch: completion, drift and roof histories and peaks,
base shear, the full hinge histories, the committed material state vectors after gravity and after the record, the
installed commands and the selected hinges.

    python -X utf8 -B -I tools/imk_frozen_v2/frame_regression.py run --study <study root> --mode mce_average \
        --case case_0064 --out outputs/imk_frozen_v2_verification_20261029/frame
    python -X utf8 -B -I tools/imk_frozen_v2/frame_regression.py compare --study <study root> --mode mce_average \
        --case case_0064 --out outputs/imk_frozen_v2_verification_20261029/frame
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as ev                                                   # noqa: E402

import numpy as np                                                      # noqa: E402

STATE_DTYPE = np.dtype([("db_tag", "<i4"), ("v", "<f8", (70,))])
# IMKPeakOriented 3.8.0 sendSelf slots used by the harness (run_earthquake.py): engRefS, cPosFy, cPosFcap, cUi, cFi,
# cEngAcml, cNegFy, cNegFcap, cFailure_Flag.
NAMED_SLOTS = {34: "engRefS", 39: "cPosFy", 41: "cPosFcap", 50: "cUi", 51: "cFi", 54: "cEngAcml", 57: "cNegFy",
               59: "cNegFcap", 68: "cFailure_Flag"}


def _states(path):
    raw = np.fromfile(path, dtype=STATE_DTYPE, offset=4)["v"]
    return {int(v[0]): v for v in raw}


def run(args):
    study = ev.study_root(args.study)
    harness = ev.harness_root(study)
    out_mode = Path(args.out).resolve() / args.mode
    if ev.under_study(out_mode, study):
        raise SystemExit("the output directory must not be inside the approved study")
    case_dir = out_mode / "cases" / args.case
    if case_dir.exists() and any(case_dir.iterdir()):
        raise SystemExit(f"{case_dir} is not empty; the regression writes into a new directory only")
    case_dir.mkdir(parents=True, exist_ok=True)             # the harness creates only the earthquake folder (as run_case.py)
    (out_mode / "inputs" / args.case).mkdir(parents=True, exist_ok=True)
    # The harness reads its inputs relative to OUT: byte copies of the study's own inputs (hash checked).
    copies = {}
    for source, target in ((study / args.mode / "scaling.json", out_mode / "scaling.json"),
                           (study / args.mode / "inputs" / args.case / "grouped_design.json",
                            out_mode / "inputs" / args.case / "grouped_design.json")):
        if not target.exists():
            shutil.copyfile(source, target)
        if ev.sha256_file(source) != ev.sha256_file(target):
            raise SystemExit(f"{target} differs from {source}")
        copies[str(target)] = {"source": str(source), "sha256": ev.sha256_file(target)}

    sys.path.insert(1, str(harness))
    from Model import IMK_Materials as mm
    from Model import IMK_Spring_Mapping as mapping
    import preflight
    import run_earthquake as runner
    if getattr(mm, "_research_half_member_installed", False) or "profile" in sys.modules and \
            getattr(sys.modules["profile"], "PROFILE_ID", None) == ev.RESEARCH_PROFILE:
        raise SystemExit("the research adapter is installed in this process")
    if mapping.active_calibration() != mapping.FROZEN_V2:
        raise SystemExit(f"the configured calibration is {mapping.active_calibration()}, not {mapping.FROZEN_V2}")
    preflight.OUT = runner.OUT = out_mode
    manifest = {"verification": "frame integration regression (D)", "case": args.case, "mode": args.mode,
                "study": str(study), "harness": str(harness),
                "harness_sha256": {name: ev.sha256_file(harness / name) for name in ("preflight.py", "run_earthquake.py")},
                "harness_call": "run_earthquake.run(case, 0.5, '', 'UmfPack') as in the approved run_case.py",
                "adapter_installed": False, "member_spring_calibration": mapping.active_calibration(),
                "input_copies": copies, "git": ev.git_state(), "sources": ev.source_hashes(), "runtime": ev.runtime(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ev.write_json(out_mode / f"{args.case}_run_manifest.json", manifest)
    result = runner.run(args.case, 0.5, "", "UmfPack")
    manifest.update(status=result.get("status"), elapsed_sec=result.get("elapsed_sec"),
                    finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ev.write_json(out_mode / f"{args.case}_run_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("case", "mode", "status", "elapsed_sec")}), flush=True)
    return 0 if result.get("status") in ("completed", "truncated") else 1


def _array_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"shape_new": list(a.shape), "shape_approved": list(b.shape), "comparable": False}
    if a.size == 0:
        return {"shape": list(a.shape), "comparable": True, "bitwise_identical": True, "max_abs_diff": 0.0}
    diff = np.abs(a.astype(float) - b.astype(float))
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    return {"shape": list(a.shape), "comparable": True, "bitwise_identical": bool(np.array_equal(a, b)),
            "max_abs_diff": float(diff.max()), "max_abs_approved": scale,
            "max_abs_diff_over_max_abs": float(diff.max() / scale) if scale > 0 else None}


def compare(args):
    study = ev.study_root(args.study)
    new = Path(args.out).resolve() / args.mode / "cases" / args.case
    old = study / args.mode / "cases" / args.case
    report = {"verification": "frame integration regression (D)", "case": args.case, "mode": args.mode,
              "approved": str(old), "new": str(new)}
    s_new = json.loads((new / "earthquake" / "summary.json").read_text())
    s_old = json.loads((old / "earthquake" / "summary.json").read_text())
    keys = ("status", "committed_final_time_sec", "committed_rows", "peak_drift_x", "peak_drift_y", "failed_materials",
            "yielded_by_initial_envelope", "spring_count", "time_step", "dt_factor", "linear_solver")
    report["summary"] = {k: {"new": s_new.get(k), "approved": s_old.get(k), "equal": s_new.get(k) == s_old.get(k)}
                         for k in keys}
    report["summary"]["scale_factor"] = {"new": s_new["scaling"]["scale_factor"], "approved": s_old["scaling"]["scale_factor"],
                                         "equal": s_new["scaling"]["scale_factor"] == s_old["scaling"]["scale_factor"]}
    report["solver"] = {"new": s_new.get("solver"), "approved": s_old.get("solver")}
    report["selection"] = [{"criterion": a["criterion"], "new_material": a["spring"]["material_tag"],
                            "approved_material": b["spring"]["material_tag"],
                            "equal": a["spring"]["material_tag"] == b["spring"]["material_tag"]}
                           for a, b in zip(s_new["selection"], s_old["selection"])]
    building = {}
    with np.load(new / "earthquake" / "building_response.npz") as a, np.load(old / "earthquake" / "building_response.npz") as b:
        for key in sorted(set(a.files) | set(b.files)):
            building[key] = _array_diff(a[key], b[key]) if key in a.files and key in b.files else {"missing": True}
        drift_new, drift_old = a["story_drift_history"], b["story_drift_history"]
        building["peak_story_drift"] = {"new": np.max(np.abs(drift_new), axis=0).tolist(),
                                        "approved": np.max(np.abs(drift_old), axis=0).tolist()}
    report["building_response"] = building
    hinges = {}
    with np.load(new / "earthquake" / "full_hinge_histories.npz") as a, np.load(old / "earthquake" / "full_hinge_histories.npz") as b:
        hinges["material_tag_order_equal"] = bool(np.array_equal(a["material_tag"], b["material_tag"]))
        for key in ("time_sec", "rotation_rad", "moment_kip_in"):
            hinges[key] = _array_diff(a[key], b[key])
        if not hinges[key].get("bitwise_identical", False):
            diff = np.max(np.abs(a["moment_kip_in"] - b["moment_kip_in"]), axis=0)
            order = np.argsort(diff)[::-1][:10]
            hinges["largest_moment_differences"] = [{"material_tag": int(a["material_tag"][i]), "max_abs_diff": float(diff[i])}
                                                    for i in order]
    report["hinge_histories"] = hinges
    states = {}
    for commit, label in ((1, "after_gravity"), (2, "after_record")):
        a, b = _states(new / "earthquake" / f"domain.VECs.70.{commit}"), _states(old / "earthquake" / f"domain.VECs.70.{commit}")
        common = sorted(set(a) & set(b))
        diffs = np.array([np.abs(a[t][1:] - b[t][1:]) for t in common]) if common else np.zeros((0, 69))
        states[label] = {"materials_new": len(a), "materials_approved": len(b), "common": len(common),
                         "missing_in_new": sorted(set(b) - set(a))[:20], "extra_in_new": sorted(set(a) - set(b))[:20],
                         "bitwise_identical": bool(common) and all(np.array_equal(a[t], b[t]) for t in common),
                         "max_abs_diff_by_named_slot": {name: float(diffs[:, slot - 1].max()) if len(diffs) else 0.0
                                                        for slot, name in NAMED_SLOTS.items()},
                         "max_abs_diff_any_slot": float(diffs.max()) if len(diffs) else 0.0,
                         "failed_materials_new": int(sum(a[t][68] != 0 for t in a)),
                         "failed_materials_approved": int(sum(b[t][68] != 0 for t in b))}
    report["material_states"] = states
    r_new = json.loads((new / "registry.json").read_text())
    r_old = json.loads((old / "registry.json").read_text())
    commands = {"springs": 0, "mismatching": 0}
    for tag, entry in r_old.items():
        for end, mats in entry["installed_materials"].items():
            for axis, material in mats.items():
                commands["springs"] += 1
                if r_new[tag]["installed_materials"][end][axis]["command"] != material["command"]:
                    commands["mismatching"] += 1
    commands["members_new"], commands["members_approved"] = len(r_new), len(r_old)
    report["installed_commands"] = commands
    rank_new = json.loads((new / "earthquake" / "rankings.json").read_text())
    rank_old = json.loads((old / "earthquake" / "rankings.json").read_text())
    report["rankings_top10"] = {kind: {"new": [r["material_tag"] for r in rank_new[kind][:10]],
                                       "approved": [r["material_tag"] for r in rank_old[kind][:10]],
                                       "equal": [r["material_tag"] for r in rank_new[kind][:10]]
                                                == [r["material_tag"] for r in rank_old[kind][:10]],
                                       "rows_equal": rank_new[kind] == rank_old[kind]}
                                for kind in ("rotation", "deterioration")}
    identical = (all(v.get("equal", True) for v in report["summary"].values())
                 and all(v.get("bitwise_identical", False) for k, v in building.items() if k != "peak_story_drift")
                 and hinges["material_tag_order_equal"]
                 and all(hinges[k]["bitwise_identical"] for k in ("time_sec", "rotation_rad", "moment_kip_in"))
                 and all(s["bitwise_identical"] for s in states.values()) and commands["mismatching"] == 0
                 and all(r["equal"] for r in report["selection"]))
    report["bitwise_reproduction"] = identical
    report["runtime"] = ev.runtime()
    path = Path(args.out).resolve() / args.mode / f"{args.case}_comparison.json"
    ev.write_json(path, report)
    print(json.dumps({"case": args.case, "mode": args.mode, "bitwise_reproduction": identical,
                      "status": report["summary"]["status"], "peak_drift_x": report["summary"]["peak_drift_x"],
                      "peak_drift_y": report["summary"]["peak_drift_y"],
                      "hinge_moment_max_abs_diff": hinges["moment_kip_in"].get("max_abs_diff"),
                      "commands_mismatching": commands["mismatching"]}, indent=1), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("run", "compare"))
    parser.add_argument("--study", required=True, help="root folder of the approved study (manifest.json, profile.py)")
    parser.add_argument("--mode", required=True, choices=ev.MODES)
    parser.add_argument("--case", default="case_0064", choices=ev.CASES)
    parser.add_argument("--out", default=str(ev.DEFAULT_OUTPUT / "frame"))
    args = parser.parse_args()
    return run(args) if args.action == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
