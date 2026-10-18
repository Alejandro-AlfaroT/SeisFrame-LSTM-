"""Run the grouped (two-system) designer over every case of a screening manifest, several cases at a time, and summarize.

    python tools/run_grouped_batch.py --manifest pilots/<name>/screening_plan.json --output-root <new directory> \
        --workers 5 --max-feasibility-trials 16 --max-reduction-trials 6 --probe-date 2026-10-17 \
        [--stages design gravity_modal] [--grouping system_three_story_v1]

Each case runs Design/Grouped_Runner.py in its own process and directory (<output-root>/<case_id>); a case directory that
already exists is left alone and reported as such (a changed input or budget takes a new root). The seed of every case
follows the lever order of 2026-10-16: the first rung of the column and beam concrete ladders, a column of 18, 20 or
24 in by story count, and the lightest beam rung that the two bay spans and the story height admit at about span / 12
deep. The budgets are the search's; nothing here is a bound on any design.

Outputs in the root: batch_log.txt (one line per finished case), summary.csv and summary.md (status, acceptance,
checks, the four tie sections, the search outcome, the compatibility and gravity-branch results), and the manifest's
SHA-256 beside them.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))
if str(RC_DIR / "Data_Generation") not in sys.path:
    sys.path.insert(0, str(RC_DIR / "Data_Generation"))

TIES = ("sfrs_column", "gravity_column", "sfrs_beam", "gravity_beam")


def seed_for(case):
    """(column (b, h, fc), beam (b, h, fc)) where the search starts for one case, per the lever order."""
    from Design.Section_Design import BEAM_CONCRETE_STRENGTHS_KSI, COLUMN_CONCRETE_STRENGTHS_KSI, beam_ladder
    stories = int(case["num_floor"])
    column = 18.0 if stories <= 4 else 20.0 if stories <= 6 else 24.0
    story_in = 12.0 * float(case["story_height_ft"])
    spans = [12.0 * float(case["bay_x_width_ft"]), 12.0 * float(case["bay_y_width_ft"])]
    rungs = None
    for span in spans:
        own = {(b, h) for b, h, _fc in beam_ladder(span_in=span, story_height_in=story_in)}
        rungs = own if rungs is None else rungs & own
    if not rungs:
        raise ValueError(f"No beam rung serves both spans of {case['case_id']}.")
    target = max(spans) / 12.0
    depth = min((h for _b, h in rungs if h >= target - 1e-9), default=max(h for _b, h in rungs))
    width = min(b for b, h in rungs if h == depth)
    return (column, column, float(COLUMN_CONCRETE_STRENGTHS_KSI[0])), (width, depth, float(BEAM_CONCRETE_STRENGTHS_KSI[0]))


def run_case(case, args, root, manifest_path):
    out_dir = root / case["case_id"]
    started = time.perf_counter()
    if out_dir.exists():
        return {"case_id": case["case_id"], "status": "skipped_existing_directory", "elapsed_s": 0.0, "out_dir": str(out_dir)}
    column, beam = seed_for(case)
    command = [sys.executable, "-X", "utf8", "-B", str(RC_DIR / "Design" / "Grouped_Runner.py"),
               "--manifest", str(manifest_path), "--case-id", case["case_id"],
               "--probe-assertions", "--probe-date", args.probe_date,
               "--seed-column", *(f"{v:g}" for v in column), "--seed-beam", *(f"{v:g}" for v in beam),
               "--max-feasibility-trials", str(args.max_feasibility_trials), "--max-reduction-trials", str(args.max_reduction_trials),
               "--stages", *args.stages, "--grouping", args.grouping, "--output-dir", str(out_dir)]
    out_dir.mkdir(parents=True)
    with (out_dir / "batch_stdout.txt").open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=str(RC_DIR), stdout=handle, stderr=subprocess.STDOUT, text=True)
    row = {"case_id": case["case_id"], "return_code": completed.returncode, "elapsed_s": time.perf_counter() - started,
           "out_dir": str(out_dir), "seed_column": list(column), "seed_beam": list(beam)}
    result_path = out_dir / "result.json"
    if result_path.exists():
        row["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        row["status"] = row["result"].get("status")
    else:
        row["status"] = f"no_result_json (return code {completed.returncode})"
    return row


def _tie_text(systems, tie):
    items = (systems or {}).get("ties", {})
    matching = [(name,item) for name,item in sorted(items.items()) if name.rsplit('__',1)[-1] == tie]
    if not matching:
        return "-"
    return "; ".join((f"{name.rsplit('__',1)[0]}: " if name != tie else "") + f"{item[0]:g}x{item[1]:g}@{item[2]:g}"
                     for name,item in matching)


def summarize(rows, cases, root, manifest_sha):
    by_id = {c["case_id"]: c for c in cases}
    header = ["case_id", "geometry", "site", "status", "accepted", "pass", "fail", "open", "fail_ids", "search_stop", "feasibility",
              "reductions", "design_s", "sfrs_column", "gravity_column", "sfrs_beam", "gravity_beam", "period_s", "dcr_column",
              "dcr_beam", "roof_delta_u_in", "gravity_branches", "error"]
    table = []
    for row in sorted(rows, key=lambda r: r["case_id"]):
        case, result = by_id[row["case_id"]], row.get("result") or {}
        counts, search, systems = result.get("counts") or {}, result.get("search") or {}, result.get("systems") or {}
        compat = systems.get("roof_design_displacement_in") or {}
        table.append([
            row["case_id"],
            f"{case['num_bay_x']}x{case['num_bay_y']} bays {case['bay_x_width_ft']:g}x{case['bay_y_width_ft']:g} ft, {case['num_floor']} st of {case['story_height_ft']:g} ft",
            case["seismic_site"], row.get("status"), result.get("accepted"), counts.get("pass"), counts.get("fail"), counts.get("not_evaluated"),
            " ".join(result.get("fail_ids") or []), search.get("stop_reason"),
            (search.get("counts") or {}).get("feasibility_candidates"), (search.get("counts") or {}).get("reduction_candidates"),
            None if result.get("design_elapsed_s") is None else round(result["design_elapsed_s"]),
            _tie_text(systems, "sfrs_column"), _tie_text(systems, "gravity_column"), _tie_text(systems, "sfrs_beam"), _tie_text(systems, "gravity_beam"),
            None if result.get("model_period_sec") is None else round(result["model_period_sec"], 3),
            None if not result.get("dcr") else round(result["dcr"]["column"], 3), None if not result.get("dcr") else round(result["dcr"]["beam"], 3),
            "" if not compat else f"x {compat.get('x', 0):.2f} y {compat.get('y', 0):.2f}",
            " ".join(f"{gid.split('__')[-2][:4]}_{gid.split('__')[-1][:3]}:{b.split('.')[-1]}" for gid, b in (systems.get("gravity_frame_branches") or {}).items()),
            (result.get("error") or "")[:120] or ("" if (row.get("return_code") in (0, None)) else f"return code {row['return_code']}"),
        ])
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(table)
    designed = sum(1 for t in table if t[3] == "designed")
    accepted = sum(1 for t in table if t[4] is True)
    lines = [f"**Grouped two-system design batch, {time.strftime('%Y-%m-%d %H:%M')}**", "",
             f"{len(table)} cases; {designed} designed, {accepted} accepted by checklist, "
             f"{sum(1 for t in table if t[3] == 'no_feasible_design')} without a feasible design, "
             f"{sum(1 for t in table if t[3] not in ('designed', 'no_feasible_design'))} other. Manifest SHA-256 {manifest_sha[:16]}. "
             "PROBE assertions: exercises the pipeline, certifies nothing.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for t in table:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in t) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-feasibility-trials", type=int, default=16)
    parser.add_argument("--max-reduction-trials", type=int, default=6)
    parser.add_argument("--probe-date", required=True)
    parser.add_argument("--stages", nargs="+", default=["design", "gravity_modal"])
    parser.add_argument("--grouping", default="system_three_story_v1")
    parser.add_argument("--cases", nargs="*", default=None, help="Case ids to run (default: every case of the manifest).")
    args = parser.parse_args(argv)
    from Design.Screening_Manifest import load_manifest
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    cases = [c for c in manifest["cases"] if not args.cases or c["case_id"] in args.cases]
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_manifest.json").write_text(json.dumps({"manifest": str(manifest_path), "sha256": manifest["sha256"], "host": socket.gethostname(),
                                                        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args),
                                                        "cases": [c["case_id"] for c in cases]}, indent=1), encoding="utf-8")
    log = root / "batch_log.txt"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {len(cases)} cases, {args.workers} workers, grouping {args.grouping}, "
                     f"budget {args.max_feasibility_trials}/{args.max_reduction_trials}, stages {args.stages}\n")
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, args, root, manifest_path): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            result = row.get("result") or {}
            systems = result.get("systems") or {}
            line = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {row['case_id']} {row['elapsed_s']:7.0f}s  {row.get('status')}  "
                    f"accepted={result.get('accepted')} counts={result.get('counts')} "
                    f"ties={ {k: systems.get('ties', {}).get(k) for k in TIES} if systems else '-'} "
                    f"stop={(result.get('search') or {}).get('stop_reason')}")
            with log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)
    summarize(rows, cases, root, manifest["sha256"])
    print(f"summary: {root / 'summary.md'}")


if __name__ == "__main__":
    main()
