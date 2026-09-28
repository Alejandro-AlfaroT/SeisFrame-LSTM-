"""Run bounded coupled gravity diagnostics from a saved design in a NEW folder.

Does not resize, qualify, modify or resume the source design or any dataset.
Example (from project root):
  python 'RC Structure/tools/review_coupled_gravity.py' --design-file ... --output-dir ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity, METHOD_VERSION


def run_review(design_file, output_dir, meshes=(4, 8)):
    source, destination = Path(design_file).resolve(), Path(output_dir).resolve()
    data = source.read_bytes()
    record = json.loads(data)
    geometry, sections, slab = record["geometry"], record["sections"], record["slab"]
    live = record["floor_loads"]["floor_live_load_ksf"]
    if not meshes or any(type(m) is not int or m < 2 or m > 24 for m in meshes) or len(set(meshes)) != len(meshes):
        raise ValueError("Meshes must be distinct integers from 2 to 24.")
    # Independent from the saved verification assertions/accepted status.
    # This is a gross-stiffness diagnostic, not a design-artifact migration.
    sections = {**sections, "beam_stiffness_modifier": 1., "column_stiffness_modifier": 1.}
    zero = {"id": "unloaded", "dead_factor": 0., "live_factor": 0., "live_load_ksf": live, "live_pattern": "none"}
    all_case = {"id": "gravity_strength_all", "dead_factor": 1.2, "live_factor": 1.6,
                "live_load_ksf": live, "live_pattern": "all"}
    corner = {"id": "roof_corner_live", "dead_factor": 0., "live_factor": 1.,
              "live_load_ksf": live, "live_pattern": [[0, 0]]}
    cases = {"gravity_strength_all": [all_case]*geometry["num_floor"],
             "roof_corner_live": [zero]*(geometry["num_floor"]-1)+[corner]}
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, loadcases in cases.items():
        for mesh in meshes:
            print(f"Coupled diagnostic: {name}, mesh {mesh}/bay", flush=True)
            result = analyze_coupled_gravity(slab, geometry, sections, loadcases, mesh_per_bay=mesh)
            filename = f"{name}_m{mesh}.json"
            with (destination/filename).open("x", encoding="utf-8") as stream:
                json.dump(result, stream, allow_nan=False, separators=(",", ":"))
            row = {"case": name, "mesh": mesh, "status": result["status"], "file": filename}
            if result.get("floors"):
                membrane = max(abs(s["gauss_resultants_raw"][8*k+d]) for s in result["shell_resultants"]
                               for k in range(4) for d in range(3))
                row.update(roof_max_down_in=result["floors"][-1]["maximum_downward_displacement_in"],
                           force_relative_error=result["equilibrium"]["force_relative_error"],
                           moment_relative_error=result["equilibrium"]["moment_relative_error"],
                           interface_moment_relative_error=result["shell_to_frame_equilibrium"]["moment_relative_error"],
                           max_membrane_kip_per_in=membrane)
            rows.append(row)
            print(f"  {row['status']}", flush=True)
    summary = {"method_version": METHOD_VERSION, "source_design": str(source),
               "source_design_sha256": hashlib.sha256(data).hexdigest(),
               "verified": False, "generation_launched": False, "design_modified": False,
               "geometry": geometry, "sections": sections, "results": rows}
    with (destination/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    lines = ["# Coupled gravity diagnostic", "", "**Diagnostic only — not qualified for generation.**", "",
             f"Source: `{source}`", "", f"SHA256: `{summary['source_design_sha256']}`", "",
             "All shell/web/column stiffnesses are gross in this comparison. Existing design assertions are not used.", "",
             "| Case | Mesh/bay | Status | Roof maximum downward (in) | Force error | Moment error | Peak membrane (kip/in) |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['mesh']} | {row['status']} | "
                     f"{row.get('roof_max_down_in', float('nan')):.8g} | {row.get('force_relative_error', float('nan')):.3g} | "
                     f"{row.get('moment_relative_error', float('nan')):.3g} | {row.get('max_membrane_kip_per_in', float('nan')):.6g} |")
    lines.extend(["", "## What this does and does not establish", "",
                  "Slab, eccentric downstand webs and full-height columns now solve together. Six-component shell-to-frame "
                  "actions are recorded at the compatible displacements; they are not reusable fixed loads for a different frame.", "",
                  "The pressure and member-weight ledgers remain separate. Column flexibility and slab membrane actions "
                  "are retained. Web moments are not composite member capacities. No new bar design or time histories were run.", "",
                  "Force/moment balance is not mesh convergence: compare local resultants as well as displacements. "
                  "Growing peak membrane forces near idealized joint/beam connections must be investigated before reinforcement sizing.", "",
                  "Remaining: independent stiffness/mesh validation (including the inherited torsion proxy), composite "
                  "section force recovery, membrane-plus-bending slab design, support-face shear, seismic/rigid-diaphragm "
                  "consistency and integration with the design loop. The generation gate remains closed.", ""])
    with (destination/"REVIEW.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    if source.read_bytes() != data:
        raise RuntimeError("Source design changed during the diagnostic; review provenance before using these results.")
    print(f"Saved diagnostic review: {destination/'REVIEW.md'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing folders are refused.")
    parser.add_argument("--meshes", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()
    summary = run_review(args.design_file, args.output_dir, args.meshes)
    if any(row["status"] != "diagnostic_complete" for row in summary["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
