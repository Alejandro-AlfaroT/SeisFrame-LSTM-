"""Screen a generated dataset for ACI 318-19 demand-capacity ratios > 1.

The batch generation pipeline never runs a design check (see Redesign.py,
which is only wired into Main.py), so nothing in a generated dataset records
whether the fixed sections were actually adequate for the demand each
geometry/record pairing produced. This screens that after the fact.

Reuses RC_Design_Check.py's own ACI 318-19 formulas -- same P-M interaction
diagram, same shear and flexure equations -- applied to the per-element
force envelope (element_end_force_envelope.csv) that every completed NTHA run
already stores. No OpenSees model rebuild required.

Methodology note: the envelope stores max/min per force component
independently, not concurrent (P, M, V) triples at a single instant. The
column P-M check therefore brackets conservatively, evaluating both the max
and min axial force against the envelope moment and taking the worse case.
That makes this a screening tool, not a certified concurrent code check.

Usage
-----
    python Check_Dataset_DCR.py --dataset-root "<root>" [--dataset-root "<root2>"]
    python Check_Dataset_DCR.py --dataset-root "<root>" --csv dcr_by_case.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
import sys
import time


RC_DIR = Path(
    os.environ.get("RC_STRUCTURE_DIR", Path(__file__).resolve().parents[1])
).resolve()
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import Structure_Parameters as sp  # noqa: E402
from RC_Design_Check import (  # noqa: E402
    _beam_d,
    _col_d,
    _stirrup_area,
    build_column_PM_diagram,
    check_beam_flexure,
    check_column_PM,
    check_shear,
)


def _f(row, key):
    return float(row[key])


def evaluate_case(envelope_path: Path, column_diagram, column_av, beam_av):
    """Return the governing DCR for one case/run from its force envelope."""
    with envelope_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    by_element: dict[str, dict] = {}
    for row in rows:
        entry = by_element.setdefault(
            row["ele_tag"], {"type": row["element_type"], "ends": []}
        )
        entry["ends"].append(row)

    worst = {"dcr": 0.0, "kind": None, "ele_tag": None}

    def _consider(dcr, kind, ele_tag):
        nonlocal worst
        if dcr > worst["dcr"]:
            worst = {"dcr": dcr, "kind": kind, "ele_tag": ele_tag}

    for ele_tag, info in by_element.items():
        ends = info["ends"]
        shear_y = max(
            max(abs(_f(r, "shear_y_kip_max")), abs(_f(r, "shear_y_kip_min")))
            for r in ends
        )
        shear_z = max(
            max(abs(_f(r, "shear_z_kip_max")), abs(_f(r, "shear_z_kip_min")))
            for r in ends
        )
        shear = (shear_y**2 + shear_z**2) ** 0.5

        if info["type"] == "column":
            end_i = next((r for r in ends if r["end"] == "i"), ends[0])
            axial_candidates = [
                _f(end_i, "axial_kip_max"),
                _f(end_i, "axial_kip_min"),
            ]
            moment_y = max(
                max(
                    abs(_f(r, "moment_y_kip_in_max")),
                    abs(_f(r, "moment_y_kip_in_min")),
                )
                for r in ends
            )
            moment_z = max(
                max(
                    abs(_f(r, "moment_z_kip_in_max")),
                    abs(_f(r, "moment_z_kip_in_min")),
                )
                for r in ends
            )
            for axial in axial_candidates:
                dcr, _, _, _ = check_column_PM(axial, moment_z, moment_y, column_diagram)
                _consider(dcr, "column_PM", ele_tag)

            _, dcr_shear, _ = check_shear(
                shear,
                max(0.0, min(axial_candidates)),
                bw=sp.B_COL,
                d=_col_d(),
                fc=sp.FC_COL_KSI,
                Av=column_av,
                s=sp.COL_STIRRUP_SPACING,
                h=sp.H_COL,
            )
            _consider(dcr_shear, "column_shear", ele_tag)
        else:
            moments = []
            for row in ends:
                moments += [
                    _f(row, "moment_y_kip_in_max"),
                    _f(row, "moment_y_kip_in_min"),
                    _f(row, "moment_z_kip_in_max"),
                    _f(row, "moment_z_kip_in_min"),
                ]
            dcr_pos, dcr_neg, _, _, _ = check_beam_flexure(
                max(0.0, max(moments)), max(0.0, -min(moments))
            )
            _consider(max(dcr_pos, dcr_neg), "beam_flexure", ele_tag)

            _, dcr_shear, _ = check_shear(
                shear,
                0.0,
                bw=sp.B_BEAM,
                d=_beam_d(),
                fc=sp.FC_BEAM_KSI,
                Av=beam_av,
                s=sp.BEAM_STIRRUP_SPACING,
            )
            _consider(dcr_shear, "beam_shear", ele_tag)

    return worst


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="Generated dataset root containing cases/; repeat for multiple roots.",
    )
    parser.add_argument("--csv", default=None, help="Optional per-case CSV output path.")
    parser.add_argument("--top", type=int, default=15, help="How many worst cases to list.")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    column_diagram = build_column_PM_diagram()
    column_av = _stirrup_area(sp.COL_STIRRUP_BAR_SIZE, sp.COL_STIRRUP_LEGS)
    beam_av = _stirrup_area(sp.BEAM_STIRRUP_BAR_SIZE, sp.BEAM_STIRRUP_LEGS)

    envelope_files = []
    for root in args.dataset_root:
        pattern = str(
            Path(root) / "cases" / "*" / "ntha" / "*" / "element_end_force_envelope.csv"
        )
        envelope_files += sorted(glob.glob(pattern))
    print(f"Found {len(envelope_files)} case/run force envelopes.")
    if not envelope_files:
        raise SystemExit("No element_end_force_envelope.csv files found under the given roots.")

    results = []
    for index, path_str in enumerate(envelope_files, 1):
        path = Path(path_str)
        try:
            worst = evaluate_case(path, column_diagram, column_av, beam_av)
        except Exception as error:  # keep screening the rest of the dataset
            print(f"  ERROR on {path}: {error}")
            continue
        worst["case_id"] = path.parents[2].name
        worst["run_name"] = path.parents[0].name
        results.append(worst)
        if index % 500 == 0:
            print(f"  ...{index}/{len(envelope_files)} ({time.time() - start:.0f}s)")

    print(f"\nEvaluated {len(results)} cases in {time.time() - start:.0f}s")

    over_limit = [row for row in results if row["dcr"] > 1.0]
    print(f"Cases with at least one element DCR > 1.0: {len(over_limit)} / {len(results)}")
    by_kind: dict[str, int] = {}
    for row in over_limit:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    for kind, count in sorted(by_kind.items(), key=lambda item: -item[1]):
        print(f"  {kind}: {count} cases")

    ordered = sorted(results, key=lambda row: -row["dcr"])
    print(f"\nWorst {args.top} cases by governing DCR:")
    for row in ordered[: args.top]:
        print(
            f"  {row['case_id']}/{row['run_name']}: DCR={row['dcr']:.3f} "
            f"kind={row['kind']} ele_tag={row['ele_tag']}"
        )

    values = sorted(row["dcr"] for row in results)
    count = len(values)
    print(
        f"\nGoverning-DCR distribution: min={values[0]:.3f} "
        f"median={values[count // 2]:.3f} p90={values[int(count * 0.9)]:.3f} "
        f"p99={values[int(count * 0.99)]:.3f} max={values[-1]:.3f}"
    )

    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["case_id", "run_name", "dcr", "kind", "ele_tag"]
            )
            writer.writeheader()
            writer.writerows(ordered)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
