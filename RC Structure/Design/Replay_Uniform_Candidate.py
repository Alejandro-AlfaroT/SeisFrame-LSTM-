"""Re-evaluate a saved uniform section pair without replacing the original run.

Example: python Design/Replay_Uniform_Candidate.py --case-dir <old case_0004>
         --output-dir <new folder> [--reduction-trials 24]
The saved dimensions seed a fresh design; the old cage is audited separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reduction-trials", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Replay output must be a new directory; existing evidence is never overwritten")
    from Design import Design_Driver as driver
    from Design.SMRF_Cage_Geometry import evaluate_cage_geometry
    from Design.Verify_Designs import configure_case, probe_config
    from Design.Config import DesignConfig

    path = args.case_dir / "design.json"
    data = path.read_bytes()
    original = json.loads(data)
    result = json.loads((args.case_dir / "result.json").read_text(encoding="utf-8-sig"))
    configure_case(result["case"], result.get("profile_id"))
    sections = original["sections"]
    for name, suffix in (("column", "col"), ("beam", "beam")):
        driver._apply_rung(tuple(sections[key] for key in (f"b_{suffix}_in", f"h_{suffix}_in", f"fc_{suffix}_ksi")), name)
    cfg = probe_config(result.get("probe_date")) if result.get("probe_assertions") else DesignConfig.from_structure_parameters()
    cfg.iteration.uniform_reduction_trials = args.reduction_trials
    args.output_dir.mkdir(parents=True)
    audit = evaluate_cage_geometry(original)
    report = {"source": str(path.resolve()), "source_sha256": hashlib.sha256(data).hexdigest(),
              "source_sections": sections, "saved_cage_audit": audit,
              "basis": "fresh reinforcement and complete analysis at the saved section seed; one feasibility "
                       "evaluation plus the declared reduction budget; no earthquake response evaluated",
              "request_identity": driver.design_request_identity(cfg)}
    (args.output_dir / "input.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        record = driver.design_structure(cfg, max_section_iter=1)
        (args.output_dir / "design.json").write_text(json.dumps(record), encoding="utf-8")
        report.update(status="evaluated", sections=record["sections"], dcr=record["dcr"],
                      search=record["search"], qualification_counts=record["qualification"]["counts"])
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}",
                      evidence=getattr(error, "evidence", None))
        raise
    finally:
        (args.output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
