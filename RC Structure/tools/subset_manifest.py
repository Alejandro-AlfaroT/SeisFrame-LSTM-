"""Write a screening manifest holding only the named cases of an existing one (for an end-to-end check).

  python tools/subset_manifest.py pilots/X/screening_plan.json pilots/X/check5_screening_plan.json --name v2ResearchCheck5 \\
         --case-ids case_0090 case_0099 case_0066 case_0001 case_0048

The first id named is the initial case. Everything else of the source manifest (profile, design basis, budget)
is kept, so the cases are designed exactly as they would be in the full run.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "Data_Generation")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    args = parser.parse_args()
    if args.target.exists():
        raise SystemExit(f"{args.target} exists; a manifest is not overwritten")
    manifest = json.loads(args.source.read_text(encoding="utf-8"))
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    missing = [cid for cid in args.case_ids if cid not in by_id]
    if missing or len(set(args.case_ids)) != len(args.case_ids):
        raise SystemExit(f"unknown or repeated case ids: {missing or args.case_ids}")
    manifest["cases"] = [by_id[cid] for cid in args.case_ids]
    manifest["status"] = f"prepared_for_{args.name}_not_executed"
    manifest["scope"] = f"End-to-end check on {len(args.case_ids)} cases of {args.source.name}. " + manifest.get("scope", "")
    manifest["execution"] = {**manifest["execution"], "initial_case_id": args.case_ids[0], "output_root": f"outputs/{args.name}_designs"}
    if isinstance(manifest.get("sampling"), dict):
        manifest["sampling"] = {**manifest["sampling"], "shard": args.name, "cases_in_this_manifest": len(args.case_ids),
                                "subset_of": args.source.name}
    args.target.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    from Design.Screening_Manifest import load_manifest
    load_manifest(args.target)
    print(f"{args.target}: {len(args.case_ids)} cases, initial {args.case_ids[0]}")


if __name__ == "__main__":
    main()
