"""Read all requested design results before freezing a school calibration plan.

This produces an explicit eligible-case roster and an exclusion report. It does
not repair designs, assert M1, modify source identities, or launch any analyses.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

RC = Path(__file__).resolve().parents[1]
if str(RC) not in sys.path:
    sys.path.insert(0, str(RC))
from Data_Generation import Build_Weighted_Seismic_Calibration_Plan as builder
from Data_Generation import Run_Weighted_Seismic_Calibration as runner


def inspect_cases(roots, case_ids, profile, use_mode, source_map):
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("The requested roster must contain unique cases")
    eligible, excluded = [], []
    for case_id in case_ids:
        try:
            row = builder.collect_cases(roots, [case_id], profile, use_mode, source_map)[0]
            eligible.append(row)
        except (ValueError, FileNotFoundError, KeyError, TypeError, OSError) as exc:
            excluded.append({"case_id": case_id, "reason": f"{type(exc).__name__}: {exc}"})
    return {"schema_version": "weighted_calibration_design_inventory_v1", "profile": profile,
            "use_mode": use_mode, "requested_count": len(case_ids), "eligible_count": len(eligible),
            "excluded_count": len(excluded), "eligible": eligible, "excluded": excluded,
            "scope": "eligibility for an intensity diagnostic only; no training release or M1 assertion"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design-root", action="append", required=True)
    p.add_argument("--case-manifest", required=True)
    p.add_argument("--profile", default=runner.RESEARCH_PROFILE)
    p.add_argument("--allow-open-m1-diagnostic", action="store_true")
    p.add_argument("--output", required=True, help="Inventory directory; repeat inspection updates only these reports")
    args = p.parse_args(argv)
    from Design import Design_Driver as driver
    requested = runner.read_json(runner.resolve(args.case_manifest))
    mode = "open-m1-diagnostic" if args.allow_open_m1_diagnostic else "qualified-only"
    report = inspect_cases([runner.resolve(r) for r in args.design_root],
                           [r["case_id"] for r in requested["cases"]], args.profile, mode, driver.source_sha256())
    output = runner.resolve(args.output)
    runner.write_json(output / "design_inventory.json", report)
    runner.write_json(output / "eligible_cases.json", {
        "schema_version": "weighted_calibration_eligible_roster_v1", "profile_id": args.profile,
        "source_manifest_sha256": runner.digest(runner.resolve(args.case_manifest)),
        "inventory_sha256": runner.digest(output / "design_inventory.json"),
        "cases": [{"case_id": c["case_id"]} for c in report["eligible"]]})
    print(f"Read-only inventory: {report['eligible_count']}/{report['requested_count']} eligible; "
          f"{report['excluded_count']} excluded. No analyses launched. Report: {output / 'design_inventory.json'}")
    for row in report["excluded"]:
        print(f"  {row['case_id']}: {row['reason']}")
    return 0 if report["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
