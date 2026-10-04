"""Grouped design of one case in a new output directory (2026-10-02).

Designs one geometry with member design varying by story band and framing location
(Design.Grouped_Search), writes the grouped record with every candidate it evaluated, and optionally runs
the gravity / modal stage on the nonlinear model. No ground motion is selected and no time-history
analysis runs. One case per invocation; nothing is launched in parallel.

    python Design/Grouped_Runner.py --manifest <screening_plan.json> --case-id case_0001 \
        --probe-assertions --probe-date 2026-10-02 \
        --seed-column 42 42 5 --seed-beam 24 30 4 \
        --max-feasibility-trials 10 --max-reduction-trials 24 \
        --stages design gravity_modal --output-dir <new directory>

    python Design/Grouped_Runner.py ... --plan-only      # print the configured case, seed, budget and request identity; write nothing

The case comes from an explicit screening manifest (its geometry, hazard site and analysis profile are
applied exactly as Design/Verify_Designs applies them). The seed is where the search starts: one column
and one beam section for every group (``--seed-column b h fc --seed-beam b h fc``), or a saved uniform
design expanded into its groups (``--seed-design <design.json>``). It is not a bound on any group. The two
budgets are the number of complete candidate evaluations the feasibility phase and the reduction phase
may spend; both are part of the request identity.

Written to the output directory:
  grouped_design.json        the record (schema rc_smrf_grouped_candidate_v1) with its qualification
  grouped_candidates.jsonl   one line per candidate, appended as each is decided (kept when the search fails)
  grouped_search_failed.json the search result when no feasible design was found within the budget
  grouped_summary.md         group-by-group sections and cages, quantities, margins, the candidate table
  gravity_modal.json         the stage result, when that stage is requested
  result.json, log.txt       the result row and the console log

A directory holds one attempt. A changed input, source file, seed or budget is a different request: the
existing record is refused and an existing candidate log is never appended to. Use a new directory.

The result row keeps numerical completion, the evaluated design checks, independent verification and
production acceptance apart. ``--probe-assertions`` fills the assertion blocks with labelled PROBE values
so the assertion-dependent design path (slab reinforcement, joint evaluation) runs; it verifies nothing.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import traceback
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
for _path in (RC_DIR, RC_DIR / "Data_Generation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Design.Verification_Integrity import exclusive_lease, file_sha256             # noqa: E402

STAGES = ("design", "gravity_modal")
SUMMARY_NAME = "grouped_summary.md"
FAILED_NAME = "grouped_search_failed.json"


class _Tee:
    """Console output that is also kept for log.txt."""

    def __init__(self, stream):
        self.stream, self.parts = stream, []

    def write(self, text):
        self.parts.append(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def text(self):
        return "".join(self.parts)


def _find_case(manifest, case_id):
    for case in manifest["cases"]:
        if case["case_id"] == case_id:
            return case
    raise SystemExit(f"{case_id!r} is not a case of the manifest ({[c['case_id'] for c in manifest['cases']]}).")


def _seed(args):
    from Design import Grouped_Record as gr
    if args.seed_design:
        if args.seed_column or args.seed_beam:
            raise SystemExit("Give --seed-design or --seed-column/--seed-beam, not both.")
        path = Path(args.seed_design).resolve()
        state, seed = gr.seed_from_uniform_record(json.loads(path.read_text(encoding="utf-8")))
        return state, {**seed, "source_path": str(path), "source_sha256": file_sha256(path)}
    if not (args.seed_column and args.seed_beam):
        raise SystemExit("A seed is required: --seed-column b h fc --seed-beam b h fc, or --seed-design <design.json>.")
    return gr.seed_from_sections(args.seed_column, args.seed_beam)


def _status(record, probe, probe_date, stages, stage_results):
    """The separate statuses of a grouped result (Design.Screening_Manifest.status_fields, with the grouped search's own stop reasons)."""
    from Design.Screening_Manifest import status_fields
    status = status_fields(record, probe, probe_date, stages, stage_results)
    search = record.get("search") or {}
    status["numerical_completion"].update({
        "design_search_stop_reason": search.get("stop_reason"),
        "design_search_budget_exhausted": search.get("stop_reason") in ("feasibility_budget_exhausted", "reduction_budget_exhausted"),
        "design_search_counts": search.get("counts"),
        "note": ("a search that ran to its stop reason is completed whatever it found; reduction_budget_exhausted means smaller "
                 "sections may remain untried, not that the design is wrong; no stop reason is evidence of an optimum"),
    })
    status["implementation_tests"] = {"run_here": False,
                                      "note": "the unit tests are run separately (python -m unittest discover -s tests); a "
                                              "passing suite is not a design check and not a release decision"}
    return status


def write_summary(record, path):
    """Group-by-group sections and cages, quantities, margins and the candidate table, as plain Markdown."""
    lines = ["**Grouped design summary**", "",
             f"Schema {record['schema_version']}; request {(record.get('request_identity') or {}).get('sha256', '')[:16]}; "
             f"member groups {record['member_groups']['sha256'][:16]}.", ""]
    search = record["search"]
    lines += [f"Search stop: {search['stop_reason']} ({search['stop_detail']}).",
              f"Candidates: {search['counts']}. Seed: {json.dumps(search['seed'])}.", "",
              "**Groups**", "",
              "| group | members | b x h (in) | fc (ksi) | bars (size top/bot/side) | hoops (size legs @ spacing) | concrete (yd3) | long. steel (lb) | trans. steel (lb) |",
              "|---|---|---|---|---|---|---|---|---|"]
    density = 490.0 / 1728.0
    for gid, design in sorted(record["member_groups"]["designs"].items()):
        amounts = record["quantities"]["groups"][gid]
        lines.append(f"| {gid} | {amounts['members']} | {design['b_in']:g} x {design['h_in']:g} | {design['fc_ksi']:g} | "
                     f"No. {design['bar_size']} {design['top_bars']}/{design['bot_bars']}/{design['side_bars']} | "
                     f"No. {design['stirrup_bar_size']} {design['stirrup_legs']} @ {design['stirrup_spacing_in']:g} | "
                     f"{amounts['concrete_in3'] / 46656.0:.1f} | {amounts['longitudinal_steel_in3'] * density:.0f} | "
                     f"{amounts['transverse_steel_in3'] * density:.0f} |")
    q = record["quantities"]
    total_steel = "not available (no slab layout)" if q["total_steel_in3"] is None else f"{q['total_steel_in3'] * density:.0f} lb"
    lines += ["", "**Quantities**", "",
              f"Concrete {q['concrete_total_yd3']:.1f} yd3 (frame {q['concrete_frame_in3'] / 46656.0:.1f}, slab "
              f"{q['concrete_slab_in3'] / 46656.0:.1f}); frame steel {q['frame_steel_lb']:.0f} lb; total steel {total_steel}; "
              f"distinct form sizes {q['distinct_form_size_count']} ({json.dumps(q['distinct_form_sizes'])}).",
              f"Basis: {q['basis']}.", "", "**Least margins**", "",
              "| check family | demand / capacity | margin | at |", "|---|---|---|---|"]
    for name, item in record["margins"].items():
        lines.append(f"| {name} | not evaluated | | |" if item is None else
                     f"| {name} | {item['demand_over_capacity']:.3f} | {item['margin']:.3f} | {item['at']} |")
    lines += ["", "**Transitions at band boundaries**", "", "| lower > upper | kind | supported | joints |", "|---|---|---|---|"]
    for key, transition in sorted((record["capacity_design"].get("transitions") or {}).items()):
        lines.append(f"| {key} | {transition['kind']} | {transition['supported']} | {len(transition['joints'])} |")
    lines += ["", "**Candidates**", "",
              "| # | phase | decision | proposal | concrete (yd3) | frame steel (lb) | total steel (lb) | reason |",
              "|---|---|---|---|---|---|---|---|"]
    for entry in search["candidates"]:
        amounts = (entry.get("evaluation") or {}).get("quantities") or {}
        total = amounts.get("total_steel_in3")
        lines.append(f"| {entry['index']} | {entry['phase']} | {entry['decision']} | {(entry.get('proposal') or {}).get('summary', '')} | "
                     f"{amounts.get('concrete_total_yd3', float('nan')):.1f} | {amounts.get('frame_steel_lb', float('nan')):.0f} | "
                     f"{'' if total is None else format(total * density, '.0f')} | {str(entry.get('decision_reason') or '')[:160]} |")
    qualification = record["qualification"]
    lines += ["", "**Checklist**", "",
              f"Accepted by checklist: {qualification['accepted']}; counts {qualification['counts']}.",
              "Failed: " + (", ".join(sorted({str(i).split(':')[0] for i in qualification.get('failed') or []})) or "none") + ".",
              "Not evaluated: " + (", ".join(sorted({str(i).split(':')[0] for i in qualification.get('not_evaluated') or []})) or "none") + ".",
              "", "**Limitations**", ""] + [f"- {item}" for item in record.get("limitations", [])]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    from Design.Verify_Designs import configure_case, probe_config
    if args.case_json:
        # An explicit case outside any manifest (the unit tests' small frame); the profile is then named directly.
        case, profile_id = json.loads(args.case_json), args.profile
        manifest = {"path": None, "sha256": None}
    else:
        from Design.Screening_Manifest import load_manifest
        manifest = load_manifest(args.manifest)
        case = _find_case(manifest, args.case_id)
        profile_id = manifest["profile_id"]
    stages = tuple(args.stages)
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    started = time.perf_counter()
    result = {"mode": "grouped", "case": case, "status": "started", "probe_assertions": bool(args.probe_assertions),
              "probe_date": args.probe_date if args.probe_assertions else None, "profile_id": profile_id, "stages": list(stages),
              "manifest": {"path": manifest["path"], "sha256": manifest["sha256"]},
              "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = None
    try:
        import Structure_Parameters as sp
        from Design import Grouped_Record as gr
        from Design import Grouped_Search as gs
        from Design.Config import DesignConfig
        _overrides, profile = configure_case(case, profile_id, emit=True)
        result["profile_sha256"] = (profile or {}).get("sha256")
        result["configured_basis"] = {"risk_category": sp.ASCE_RISK_CATEGORY, "importance_factor": sp.ASCE_IE,
                                      "joint_model": sp.JOINT_MODEL, "member_material": sp.IMK_MATERIAL_TYPE,
                                      "analysis_profile_id": sp.ANALYSIS_PROFILE_ID}
        cfg = probe_config(args.probe_date) if args.probe_assertions else DesignConfig.from_structure_parameters()
        seed_state, seed = _seed(args)
        policy = gs.SearchPolicy(max_feasibility_trials=args.max_feasibility_trials, max_reduction_trials=args.max_reduction_trials)
        identity = gr.grouped_request_identity(cfg, seed, policy)
        result.update({"seed": seed, "search_policy": identity["search"], "request_sha256": identity["sha256"],
                       "group_count": identity["grouping"]["group_count"]})
        print(f"Grouped design of {case['case_id']} ({case.get('geometry_name')}); profile {profile_id}; "
              f"{identity['grouping']['group_count']} groups; request {identity['sha256'][:16]}")
        print(f"Seed {json.dumps(seed)}")
        print(f"Budget: {policy.max_feasibility_trials} feasibility and {policy.max_reduction_trials} reduction evaluations; "
              f"policy {policy.improvement_policy}")
        if args.plan_only:
            result["status"] = "plan_only"
            return result
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_lease(out_dir / ".worker.lease"):
            try:
                record, created = gr.load_or_create_grouped_design(out_dir / gr.ARTIFACT_NAME, cfg, seed_state, seed, policy)
            except gr.GroupedDesignNotFound as exc:
                search = exc.search
                (out_dir / FAILED_NAME).write_text(json.dumps(
                    {key: value for key, value in search.items() if key != "final"}, indent=1, default=str), encoding="utf-8")
                result.update({
                    "status": "no_feasible_design", "search_stop": search["stop"], "search_counts": search["counts"],
                    "numerical_completion": {"design": "completed_without_a_feasible_design",
                                             "design_search_stop_reason": search["stop"]["reason"],
                                             "design_search_budget_exhausted": search["stop"]["reason"] == "feasibility_budget_exhausted",
                                             "note": "every candidate is in grouped_candidates.jsonl with its failed constraints; a "
                                                     "budget stop is not evidence that no feasible frame exists"},
                    "design_checks": {"accepted_by_checklist": False, "note": "no design record to check"},
                    "production_acceptance": {"accepted": False, "reason": "no design record"}})
                return result
            qualification = record["qualification"]
            write_summary(record, out_dir / SUMMARY_NAME)
            result.update({
                "status": "designed", "created": created, "design_elapsed_s": time.perf_counter() - started,
                "design_json_bytes": (out_dir / gr.ARTIFACT_NAME).stat().st_size,
                "design_sha256": file_sha256(out_dir / gr.ARTIFACT_NAME),
                "schema_version": record["schema_version"], "member_groups_sha256": record["member_groups"]["sha256"],
                "accepted": qualification["accepted"], "counts": qualification["counts"],
                "fail_ids": sorted({str(i).split(":")[0] for i in qualification.get("failed") or []}),
                "not_evaluated_ids": sorted({str(i).split(":")[0] for i in qualification.get("not_evaluated") or []}),
                "sections_by_group": {gid: [d["b_in"], d["h_in"], d["fc_ksi"]] for gid, d in record["member_groups"]["designs"].items()},
                "distinct_form_sizes": record["quantities"]["distinct_form_sizes"],
                "quantities": {k: v for k, v in record["quantities"].items() if k not in ("groups", "basis")},
                "margins": record["margins"], "dcr": record["dcr"],
                "model_period_sec": record["demand"]["model_period_sec"],
                "search": {k: record["search"][k] for k in ("stop_reason", "stop_detail", "counts", "elapsed_seconds", "policy")},
                "tradeoffs_retained": len(record["search"]["tradeoffs"]),
                "coupled_max_vertical_difference": (record.get("coupled_comparison") or {}).get("max_column_vertical_relative_difference"),
            })
            stage_results = {}
            if "gravity_modal" in stages:
                from Design.Screening_Manifest import gravity_modal_stage
                stage_results["gravity_modal"] = gravity_modal_stage(record, profile_id, out_dir)
                (out_dir / "gravity_modal.json").write_text(json.dumps(stage_results["gravity_modal"], indent=1, default=str),
                                                           encoding="utf-8")
            result["stage_results"] = {name: {key: value for key, value in stage.items() if key not in ("model_audit", "traceback")}
                                       for name, stage in stage_results.items()}
            result.update(_status(record, bool(args.probe_assertions), args.probe_date, stages, stage_results))
    except SystemExit:
        raise
    except Exception as exc:                              # noqa: BLE001 -- the observed failure is the result row
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                       "numerical_completion": {"design": "error", "stages_requested": list(stages),
                                                "note": "the run raised; candidates evaluated before the error are in "
                                                        "grouped_candidates.jsonl"},
                       "design_checks": {"accepted_by_checklist": False, "note": "no design record to check"},
                       "production_acceptance": {"accepted": False, "reason": "no design record"}})
        print(result["traceback"])
    finally:
        sys.stdout = tee.stream
        result["elapsed_s"] = time.perf_counter() - started
        result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if out_dir is not None and not args.plan_only:
            (out_dir / "log.txt").write_text(tee.text(), encoding="utf-8")
            (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=None, help="Screening manifest (seisframe_v2_screening_plan_v1) that names the case.")
    parser.add_argument("--case-id", default=None, help="The manifest case to design.")
    parser.add_argument("--case-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--profile", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=None, help="A new directory for this attempt (required unless --plan-only).")
    parser.add_argument("--seed-column", nargs=3, type=float, metavar=("B", "H", "FC"), default=None,
                        help="Seed column section for every column group: b h fc (in, in, ksi); square.")
    parser.add_argument("--seed-beam", nargs=3, type=float, metavar=("B", "H", "FC"), default=None,
                        help="Seed beam section for every beam group: b h fc (in, in, ksi).")
    parser.add_argument("--seed-design", default=None, help="A saved uniform design.json of the same case, expanded into its groups.")
    parser.add_argument("--max-feasibility-trials", type=int, required=True,
                        help="Complete candidate evaluations the feasibility phase may spend (part of the request identity).")
    parser.add_argument("--max-reduction-trials", type=int, required=True,
                        help="Complete candidate evaluations the reduction phase may spend (part of the request identity).")
    parser.add_argument("--probe-assertions", action="store_true",
                        help="Fill the assertion blocks with labelled PROBE values; exercises the pipeline, verifies nothing.")
    parser.add_argument("--probe-date", default=None, help="YYYY-MM-DD assertion date; required with --probe-assertions.")
    parser.add_argument("--stages", nargs="+", default=["design"], choices=STAGES,
                        help="'design' (default), optionally followed by 'gravity_modal'. No ground motion.")
    parser.add_argument("--plan-only", action="store_true", help="Print the configured request and its identity; write nothing.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if "design" not in args.stages:
        raise SystemExit("--stages must include 'design'.")
    if not args.case_json and not (args.manifest and args.case_id):
        raise SystemExit("--manifest and --case-id are required.")
    if not args.plan_only and not args.output_dir:
        raise SystemExit("--output-dir is required (a new directory for this attempt).")
    if args.probe_assertions and not args.probe_date:
        raise SystemExit("--probe-assertions needs --probe-date YYYY-MM-DD.")
    result = run(args)
    print(json.dumps({key: result.get(key) for key in ("status", "elapsed_s", "accepted", "counts", "search", "error", "request_sha256")},
                     default=str))
    return 0 if result["status"] in ("designed", "plan_only", "no_feasible_design") else 1


if __name__ == "__main__":
    raise SystemExit(main())
