"""Design one building with the current methodology and check it on the nonlinear model (rewritten 2026-10-23).

Main.py designs a single structure on the grouped two-system route that the design batches use: perimeter special
moment frames are the seismic force-resisting system and every other beam and column is a gravity member. Members are
grouped by three-story band and plan location, and the groups of one system tie (special-frame or gravity, column or
beam) in a band share one design. The slab acts compositely with the beams (T and L sections, slab reinforcement
designed). The search starts from a seed and iterates candidate designs: a feasibility phase, then a reduction phase,
with the Grade 80 retry and the core-layout ladder (perimeter + core ring, then core lines) when a phase finds nothing.
The selected design is then built as the nonlinear IMK model and run through gravity and modal analysis.

Progress is printed live: the search prints one line per candidate as it is decided, and a heartbeat line every
``--heartbeat-seconds`` gives the elapsed time, the stage and the number of candidates decided so far.

    & "$env:USERPROFILE\\anaconda3\\envs\\OpPy\\python.exe" -X utf8 -B Main.py
    & "$env:USERPROFILE\\anaconda3\\envs\\OpPy\\python.exe" -X utf8 -B Main.py --bays-x 5 --bays-y 4 --floors 6 `
        --story-height-ft 13 --bay-x-ft 24 --bay-y-ft 22 --site sdc_d_low
    & "$env:USERPROFILE\\anaconda3\\envs\\OpPy\\python.exe" -X utf8 -B Main.py --plan-only

Run from the RC Structure folder. Without arguments it designs the building in CASE below. Every run writes a new
folder outputs/main/<geometry name>__<YYYYmmdd_HHMMSS>/ and never touches an earlier one:
  grouped_design.json        the design record (schema rc_smrf_grouped_candidate_v1) with its qualification checklist
  grouped_candidates*.jsonl  every candidate the search evaluated, appended as each is decided
  grouped_search_failed.json the search result when no feasible design was found within the budgets
  grouped_steel_grade_trial.json  the Grade 80 retry decision, when that retry was considered
  grouped_summary.md         sections and cages by group, quantities, margins, systems, the candidate table
  gravity_modal.json         gravity, modal and installed-model checks of the selected design; gravity_modal/ the hinge export
  result.json, log.txt       the result row and the console log

The design engine is Design/Grouped_Runner.run, called with the seed rule, budgets and grouping of
tools/run_grouped_batch.py and the analysis profile of the screening manifests (tools/build_lhs_pilot.py), so a run
here makes the same design request as a batch case of the same building. The assertion blocks are filled with PROBE
values dated today, as in every grouped batch: they let the slab-reinforcement and joint-evaluation path run and
verify nothing.

No ground motion is selected and no time-history analysis runs. Saving the time-history results of a grouped design
with a slab stops in Graph_Exporter.collect_reinforcement_geometry (GroupedStateError: it reads one stirrup size for
every member while writing global_parameters.json); the two time-history tests of tests/test_grouped_record.py error
on it. The hybrid training sample (Hybrid_Exporter) and the uniform driver Ground_Motion_Main.main also refuse grouped
designs.

The legacy single-building route is gone: Main.py no longer reads Structure_Parameters.ANALYSIS_METHOD, the NTHA_*
record settings or the RC_* environment geometry (Geometry_Overrides.apply_geometry_from_environment), and it no longer
runs the pushover redesign loop or writes the fixed-name files directly in outputs/. The building comes from CASE or
the command-line options below. The earlier Main.py is in the git history (commit f5734c5c and before).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path

# The Intel Fortran runtime built into openseespywin/opensees.pyd installs a console Ctrl+C handler that aborts the
# process ("forrtl: error (200)") before Python sees the interrupt, so no finally block runs (result.json, log.txt and
# the design lock). The runtime reads this variable when OpenSees is first imported (inside the design run); with the
# handler off, Ctrl+C raises KeyboardInterrupt.
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

# Render the design summaries on Windows, and print every line as it is written (live progress also when piped).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

RC_DIR = Path(__file__).resolve().parent
for _path in (RC_DIR, RC_DIR / "Data_Generation", RC_DIR / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# The building designed when Main.py runs without arguments: a mid-size member of the V2 population
# (Data_Generation/Generate_Parameterized_Dataset.RANGES). Lengths in feet; the site is a label of
# Structure_Parameters.SEISMIC_SITE_OPTIONS. Edit these values, or pass the matching command-line options.
CASE = {"num_bay_x": 4, "num_bay_y": 3, "num_floor": 5, "story_height_ft": 14.0,
        "bay_x_width_ft": 29.5, "bay_y_width_ft": 21.5, "seismic_site": "sdc_c"}
# Analysis profile the design is made under (the screening manifests' default): IMKPeakOriented hinges on beams and
# columns, rigid centerline joints, hinge histories at every step. A grouped design builds only under a V2 profile.
PROFILE_ID = "v2_nonlinear_flexure_screening_v1"
# Search budgets of tools/run_grouped_batch.py: complete candidate evaluations per phase (part of the request identity).
MAX_FEASIBILITY_TRIALS = 16
MAX_REDUCTION_TRIALS = 6
HEARTBEAT_SECONDS = 30.0
OUTPUT_ROOT = RC_DIR / "outputs" / "main"
FINISHED_STATUSES = ("designed", "plan_only", "no_feasible_design")


def _clock(seconds):
    seconds = int(max(0.0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _feet(value):
    return f"{float(value):g}"


def geometry_name(case):
    """Folder-safe name of the building, in the form of the screening manifests (tools/build_lhs_pilot.py)."""
    name = (f"main_bx{case['num_bay_x']}_by{case['num_bay_y']}_s{case['num_floor']}_sh{_feet(case['story_height_ft'])}"
            f"_bwx{_feet(case['bay_x_width_ft'])}_bwy{_feet(case['bay_y_width_ft'])}_{case['seismic_site']}")
    return name.replace(".", "p")


def build_case(args):
    """The case in the form Design/Verify_Designs.configure_case applies (feet, bay counts, site label)."""
    case = {"case_id": "main", "num_bay_x": args.bays_x, "num_bay_y": args.bays_y, "num_floor": args.floors,
            "story_height_ft": float(args.story_height_ft), "bay_x_width_ft": float(args.bay_x_ft),
            "bay_y_width_ft": float(args.bay_y_ft), "seismic_site": args.site}
    case["geometry_name"] = geometry_name(case)
    return case


def population_notes(case):
    """Why the case lies outside the V2 population, or an empty list when it is a member
    (Generate_Parameterized_Dataset: RANGES, SEISMIC_SITES and the height-to-width cap). A case outside it still runs."""
    saved_path, saved_dir = list(sys.path), os.environ.pop("RC_STRUCTURE_DIR", None)
    try:
        # The module puts RC_STRUCTURE_DIR first on the import path; read this checkout's population, then restore both.
        import Generate_Parameterized_Dataset as population
    finally:
        sys.path[:] = saved_path
        if saved_dir is not None:
            os.environ["RC_STRUCTURE_DIR"] = saved_dir
    notes = []
    for key, levels in population.RANGES.items():
        if float(case[key]) not in [float(level) for level in levels]:
            notes.append(f"{key} = {case[key]:g} is not a population level ({levels[0]:g} to {levels[-1]:g}"
                         + (")" if key.startswith("num_") else " in 0.5 ft steps)"))
    ratio = population.height_to_width_ratio(**{key: case[key] for key in population.RANGES})
    if ratio > population.MAX_HEIGHT_TO_WIDTH_RATIO + 1e-12:
        notes.append(f"height / least plan width = {ratio:.2f} exceeds {population.MAX_HEIGHT_TO_WIDTH_RATIO:g}")
    if case["seismic_site"] not in population.SEISMIC_SITES:
        notes.append(f"site {case['seismic_site']} is not one of {', '.join(population.SEISMIC_SITES)}")
    return notes


def batch_seed(case):
    """Where the search starts: the seed rule of tools/run_grouped_batch.py (lever order of 2026-10-16)."""
    from run_grouped_batch import seed_for
    return seed_for(case)


def runner_arguments(args, case, seed_column, seed_beam, out_dir):
    """The Design/Grouped_Runner arguments of this run, parsed by the runner's own parser."""
    from Design import Grouped_Runner as runner
    argv = ["--case-json", json.dumps(case), "--profile", args.profile,
            "--seed-column", *(f"{value:g}" for value in seed_column), "--seed-beam", *(f"{value:g}" for value in seed_beam),
            "--grouping", args.grouping,
            "--max-feasibility-trials", str(args.max_feasibility_trials),
            "--max-reduction-trials", str(args.max_reduction_trials),
            "--probe-assertions", "--probe-date", args.probe_date,
            "--stages", "design", *([] if args.design_only else ["gravity_modal"])]
    if args.no_grade80_retry:
        argv.append("--no-grade80-retry")
    argv += ["--plan-only"] if args.plan_only else ["--output-dir", str(out_dir)]
    return runner.build_parser().parse_args(argv)


def new_output_dir(args, case):
    """A new, empty folder for this run (the runner keeps one attempt per folder)."""
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(f"{out_dir} is not empty; give a new folder (or leave --output-dir out for a dated one).")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for attempt in range(100):
        out_dir = OUTPUT_ROOT / (f"{case['geometry_name']}__{stamp}" + (f"_{attempt}" if attempt else ""))
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            continue
    raise SystemExit(f"Could not create a new output folder under {OUTPUT_ROOT}.")


class Heartbeat:
    """Prints the elapsed time, the stage and the number of decided candidates every ``interval`` seconds.

    The count is read from the candidate logs the search appends to (grouped_candidates*.jsonl), so nothing inside
    the design engine is touched. Lines go to the console only (sys.__stdout__), not into the run's log.txt.
    """

    def __init__(self, out_dir, interval, started):
        self.out_dir, self.interval, self.started = Path(out_dir), float(interval), started
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="main-heartbeat", daemon=True)

    def __enter__(self):
        if self.interval > 0:
            self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        return False

    def candidates(self):
        total = 0
        for path in self.out_dir.glob("grouped_candidates*.jsonl"):
            try:
                with path.open("rb") as handle:
                    total += sum(1 for _line in handle)
            except OSError:
                pass
        return total

    def stage(self):
        if (self.out_dir / "grouped_design.json").exists():
            return "gravity and modal analysis of the selected design"
        if (self.out_dir / "grouped_search_failed.json").exists():
            return "writing the failed-search record"
        return "design search"

    def line(self):
        return (f"[main] {_clock(time.perf_counter() - self.started)} elapsed | {self.stage()} | "
                f"{self.candidates()} candidate(s) decided")

    def _run(self):
        stream = sys.__stdout__ or sys.stdout
        while not self._stop.wait(self.interval):
            try:
                stream.write(self.line() + "\n")
                stream.flush()
            except (OSError, ValueError):
                return


def method_text(grouping):
    """What the grouping policy designs, for the banner (Model/Member_Groups policies)."""
    from Model import Member_Groups as mg
    policy_id = mg.default_system_policy() if grouping == "auto" else grouping
    if not mg.is_system_policy(policy_id):
        return "grouped single-system design (legacy band policy: every frame special, no gravity system)"
    frames = {mg.SFRS_LAYOUT_PERIMETER: "perimeter special moment frames",
              mg.SFRS_LAYOUT_PERIMETER_CORE: "perimeter + core-ring special moment frames",
              mg.SFRS_LAYOUT_PERIMETER_CORE_LINES: "perimeter + core-line special moment frames"}[mg.layout_of(policy_id)]
    bands = "three-story bands" if mg.is_banded_policy(policy_id) else "one band over the height"
    return f"grouped two-system design: {frames} + interior gravity system; one design per system tie, {bands}"


def print_banner(args, case, seed_column, seed_beam, out_dir, notes):
    import Structure_Parameters as sp
    site = {label: (sds, sd1, s1) for label, sds, sd1, s1 in sp.SEISMIC_SITE_OPTIONS}[case["seismic_site"]]
    height = case["num_floor"] * case["story_height_ft"]
    plan_x, plan_y = case["num_bay_x"] * case["bay_x_width_ft"], case["num_bay_y"] * case["bay_y_width_ft"]
    slab = ("composite slab: T and L beams, slab reinforcement designed"
            if getattr(sp, "SLAB_STRUCTURAL_PARTICIPATION", True) else "no slab participation: rectangular beams")
    print("=" * 100)
    print(f"SeisFrame Main.py -- one building, current design methodology   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print(f"Building    {case['num_bay_x']} x {case['num_bay_y']} bays, {case['num_floor']} floors; story {_feet(case['story_height_ft'])} ft; "
          f"bays {_feet(case['bay_x_width_ft'])} ft (X) x {_feet(case['bay_y_width_ft'])} ft (Y)")
    print(f"            height {height:g} ft; plan {plan_x:g} x {plan_y:g} ft; height / least plan width {height / min(plan_x, plan_y):.2f}")
    print(f"Hazard      {case['seismic_site']} (SDS {site[0]:g} g, SD1 {site[1]:g} g, S1 {site[2]:g} g); "
          f"Risk Category {sp.ASCE_RISK_CATEGORY}")
    print(f"Method      {method_text(args.grouping)}")
    print(f"            {slab}; grouping {args.grouping}; analysis profile {args.profile}")
    print(f"Search      seed column {seed_column[0]:g} x {seed_column[1]:g} in, fc {seed_column[2]:g} ksi; "
          f"seed beam {seed_beam[0]:g} x {seed_beam[1]:g} in, fc {seed_beam[2]:g} ksi")
    print(f"            budgets {args.max_feasibility_trials} feasibility and {args.max_reduction_trials} reduction evaluations"
          + ("; Grade 80 retry off" if args.no_grade80_retry else ""))
    print(f"Assertions  PROBE values dated {args.probe_date} (let the slab and joint design path run; verify nothing)")
    print(f"Stages      design{'' if args.design_only else ' -> gravity and modal analysis on the nonlinear model'}")
    print(f"Population  " + ("member of the V2 population" if not notes else "outside the V2 population: " + "; ".join(notes)))
    print(f"Output      {out_dir if out_dir else '(plan only: nothing is written)'}")
    if not args.plan_only and args.heartbeat_seconds > 0:
        print(f"Progress    one [grouped] line per decided candidate; a [main] heartbeat every {args.heartbeat_seconds:g} s")
    print("=" * 100, flush=True)


def print_report(result, out_dir):
    """The end-of-run summary on the console; result.json and grouped_summary.md hold the full record."""
    status = result.get("status")
    print()
    print("=" * 100)
    print(f"Result      {status} after {_clock(result.get('elapsed_s') or 0.0)}")
    if status == "plan_only":
        print(f"Request     {result.get('request_sha256')} ({result.get('group_count')} groups, grouping {result.get('grouping_policy')})")
    elif status == "error":
        print(f"Error       {result.get('error')}")
        if out_dir and (Path(out_dir) / "log.txt").exists():
            print(f"            the traceback is in {Path(out_dir) / 'log.txt'}; candidates evaluated before the error are in "
                  "grouped_candidates*.jsonl")
        else:
            print("            the run stopped while configuring the case, before its folder was used; the traceback is above")
    elif status == "no_feasible_design":
        stop = result.get("search_stop") or {}
        print(f"Search      no feasible design: {stop.get('reason')} ({stop.get('detail', '')}); counts {result.get('search_counts')}")
        print("            every candidate and its failed constraints are in grouped_candidates*.jsonl and grouped_search_failed.json")
    elif status == "designed":
        search = result.get("search") or {}
        print(f"Search      stop {search.get('stop_reason')}: {search.get('stop_detail')}")
        grade = result.get("longitudinal_grade_ksi")
        print(f"            counts {search.get('counts')}; grouping selected {result.get('grouping_policy_selected')}; "
              f"longitudinal bars fy {'?' if grade is None else format(grade, 'g')} ksi")
        print(f"Checklist   accepted by checklist: {result.get('accepted')}; counts {result.get('counts')}")
        print(f"            failed: {', '.join(result.get('fail_ids') or []) or 'none'}; "
              f"not evaluated: {', '.join(result.get('not_evaluated_ids') or []) or 'none'}")
        production = result.get("production_acceptance") or {}
        print(f"Production  accepted: {production.get('accepted')} ({production.get('reason')})")
        ties = (result.get("systems") or {}).get("ties") or {}
        if ties:
            print("Ties        tie                             b x h (in)     fc (ksi)")
            for tie, (b, h, fc) in sorted(ties.items()):
                print(f"            {tie:<31} {b:g} x {h:g}{'':<8} {fc:g}")
        groups = result.get("sections_by_group") or {}
        print(f"Groups      {len(groups)} member groups (sections and cages by group in grouped_summary.md)")
        quantities = result.get("quantities") or {}
        if quantities:
            print(f"Quantities  concrete {quantities.get('concrete_total_yd3', float('nan')):.1f} yd3; "
                  f"frame steel {quantities.get('frame_steel_lb', float('nan')) / 2000.0:.1f} t; "
                  f"distinct form sizes {quantities.get('distinct_form_size_count')}")
        period = result.get("model_period_sec")
        print(f"Period      design model {'?' if period is None else format(period, '.3f')} s")
        stage = (result.get("stage_results") or {}).get("gravity_modal")
        if stage:
            print(f"Gravity     stage {stage.get('status')}; all checks pass: {stage.get('all_checks_pass')}; {stage.get('checks')}"
                  + (f"; error {stage.get('error')}" if stage.get("error") else ""))
            modal = stage.get("modal") or {}
            if modal:
                periods = ", ".join(f"{p:.3f}" for p in modal.get("periods_elastic_reference_sec") or [])
                tangent = ", ".join(f"{p:.3f}" for p in modal.get("periods_post_gravity_tangent_sec") or [])
                print(f"Modal       nonlinear model, elastic reference periods {periods} s; post-gravity tangent {tangent} s")
    if out_dir and status != "plan_only":
        print(f"Files       {out_dir}")
    print("=" * 100, flush=True)


def report_interrupt(out_dir, elapsed):
    """Ctrl+C: say what the folder holds and leave interrupted.json beside the runner's result.json.

    The runner's finally block still writes result.json and log.txt. Its status is 'started' when the interrupt came
    during the search, but 'designed' when it came during the gravity and modal stage (that stage then has no result),
    so the marker records which stage was cut short."""
    if out_dir is None:
        print(f"\n[main] interrupted after {_clock(elapsed)} (plan only: nothing was written).")
        return
    out_dir = Path(out_dir)
    stage = "gravity and modal analysis" if (out_dir / "grouped_design.json").exists() else "design search"
    try:
        (out_dir / "interrupted.json").write_text(json.dumps(
            {"status": "interrupted", "interrupted_during": stage, "elapsed_s": elapsed,
             "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "note": "the run did not finish; result.json in this folder describes the state when it stopped"},
            indent=1), encoding="utf-8")
    except OSError:
        pass
    print(f"\n[main] interrupted during the {stage} after {_clock(elapsed)}. The run did not finish (interrupted.json); "
          f"{out_dir} keeps the candidates decided so far (grouped_candidates*.jsonl), log.txt and result.json"
          + ("; grouped_design.json holds the selected design, its gravity and modal stage did not finish."
             if stage == "gravity and modal analysis" else ". Start a new run in a new folder."))


def build_parser():
    import Structure_Parameters as sp
    from Design import Grouped_Runner as runner
    from Model.Analysis_Profile import PROFILES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    building = parser.add_argument_group("building (defaults from CASE in Main.py)")
    building.add_argument("--bays-x", type=int, default=CASE["num_bay_x"], help="Bays in X.")
    building.add_argument("--bays-y", type=int, default=CASE["num_bay_y"], help="Bays in Y.")
    building.add_argument("--floors", type=int, default=CASE["num_floor"], help="Number of floors.")
    building.add_argument("--story-height-ft", type=float, default=CASE["story_height_ft"], help="Story height (ft).")
    building.add_argument("--bay-x-ft", type=float, default=CASE["bay_x_width_ft"], help="Bay width in X (ft).")
    building.add_argument("--bay-y-ft", type=float, default=CASE["bay_y_width_ft"], help="Bay width in Y (ft).")
    building.add_argument("--site", default=CASE["seismic_site"], choices=[option[0] for option in sp.SEISMIC_SITE_OPTIONS],
                          help="Design hazard site label (Structure_Parameters.SEISMIC_SITE_OPTIONS).")
    method = parser.add_argument_group("design method")
    method.add_argument("--profile", default=PROFILE_ID, choices=sorted(PROFILES),
                        help="Analysis profile the design is made under (Model/Analysis_Profile.PROFILES).")
    method.add_argument("--grouping", default=runner.DEFAULT_GROUPING, choices=runner.GROUPINGS,
                        help="Grouping policy (default: three-story system bands, as in the batches).")
    method.add_argument("--max-feasibility-trials", type=int, default=MAX_FEASIBILITY_TRIALS,
                        help="Candidate evaluations the feasibility phase may spend.")
    method.add_argument("--max-reduction-trials", type=int, default=MAX_REDUCTION_TRIALS,
                        help="Candidate evaluations the reduction phase may spend.")
    method.add_argument("--seed-column", nargs=3, type=float, metavar=("B", "H", "FC"), default=None,
                        help="Start the search from this square column section (in, in, ksi) instead of the batch seed rule.")
    method.add_argument("--seed-beam", nargs=3, type=float, metavar=("B", "H", "FC"), default=None,
                        help="Start the search from this beam section (in, in, ksi) instead of the batch seed rule.")
    method.add_argument("--no-grade80-retry", action="store_true", help="Disable the Grade 80 retry of the search.")
    method.add_argument("--probe-date", default=date.today().isoformat(),
                        help="YYYY-MM-DD date stamped on the PROBE assertions (default: today).")
    run = parser.add_argument_group("run")
    run.add_argument("--design-only", action="store_true", help="Stop after the design; skip the gravity and modal stage.")
    run.add_argument("--output-dir", default=None,
                     help="A new or empty folder for this run (default: outputs/main/<geometry name>__<date_time>).")
    run.add_argument("--heartbeat-seconds", type=float, default=HEARTBEAT_SECONDS,
                     help="Seconds between progress heartbeats (0 turns them off).")
    run.add_argument("--plan-only", action="store_true",
                     help="Print the configured request (case, seed, budgets, identity) and write nothing.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if date.fromisoformat(args.probe_date).isoformat() != args.probe_date:
            raise ValueError
    except ValueError:
        raise SystemExit(f"--probe-date must be YYYY-MM-DD, got {args.probe_date!r}.")
    if args.max_feasibility_trials < 1 or args.max_reduction_trials < 0:
        raise SystemExit("--max-feasibility-trials must be at least 1 and --max-reduction-trials at least 0.")
    if min(args.bays_x, args.bays_y, args.floors) < 1:
        raise SystemExit("--bays-x, --bays-y and --floors must each be at least 1.")
    if not all(math.isfinite(value) and value > 0 for value in (args.story_height_ft, args.bay_x_ft, args.bay_y_ft)):
        raise SystemExit("--story-height-ft, --bay-x-ft and --bay-y-ft must be positive numbers of feet.")
    if (args.seed_column is None) != (args.seed_beam is None):
        raise SystemExit("Give both --seed-column and --seed-beam, or neither (the batch seed rule).")
    if args.seed_column is not None:
        for name, values in (("--seed-column", args.seed_column), ("--seed-beam", args.seed_beam)):
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise SystemExit(f"{name} values must be positive numbers, got {values}.")
        if args.seed_column[0] != args.seed_column[1]:
            raise SystemExit(f"Columns are square in the grouped search; --seed-column is "
                             f"{args.seed_column[0]:g} x {args.seed_column[1]:g}.")
    case = build_case(args)
    notes = population_notes(case)
    if args.seed_column is not None:
        seed_column, seed_beam = tuple(args.seed_column), tuple(args.seed_beam)
    else:
        try:
            seed_column, seed_beam = batch_seed(case)
        except ValueError as exc:
            raise SystemExit(f"The batch seed rule has no seed for this building ({exc}); give --seed-column and --seed-beam.")
    out_dir = None if args.plan_only else new_output_dir(args, case)
    print_banner(args, case, seed_column, seed_beam, out_dir, notes)

    from Design import Grouped_Runner as runner
    runner_args = runner_arguments(args, case, seed_column, seed_beam, out_dir)
    started = time.perf_counter()
    try:
        if args.plan_only:
            result = runner.run(runner_args)
        else:
            with Heartbeat(out_dir, args.heartbeat_seconds, started):
                result = runner.run(runner_args)
    except KeyboardInterrupt:
        report_interrupt(out_dir, time.perf_counter() - started)
        return 130
    if out_dir is not None and result.get("status") == "error" and not any(out_dir.iterdir()):
        out_dir.rmdir()                     # the run stopped before using its folder; leave no empty folder behind
        out_dir = None
    print_report(result, out_dir)
    return 0 if result.get("status") in FINISHED_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
