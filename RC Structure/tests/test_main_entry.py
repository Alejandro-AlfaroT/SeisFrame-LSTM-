"""The single-building entry point Main.py on the grouped two-system route (rewritten 2026-10-23).

Fast checks of the plumbing only: the case it builds, the population check, the arguments it hands to
Design.Grouped_Runner (the batch seed rule, budgets, grouping, PROBE assertions and stages), the new-folder rule,
the heartbeat's candidate count and stage, the input checks, the interrupt and error paths, the end-of-run report,
and a plan-only run that writes nothing. The design search itself is covered by test_grouped_runner and the grouped
search tests. The runner is replaced by a stub wherever a test would otherwise design a building.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import Main  # noqa: E402

V2 = "v2_nonlinear_flexure_screening_v1"
# A population member given explicitly, so editing the CASE block in Main.py never changes these tests.
MEMBER = ("--bays-x", "4", "--bays-y", "3", "--floors", "5", "--story-height-ft", "14",
          "--bay-x-ft", "29.5", "--bay-y-ft", "21.5", "--site", "sdc_c")


def parse(*argv):
    return Main.build_parser().parse_args(list(argv))


def quiet_main(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = Main.main(argv)
    return code, out.getvalue()


class CaseAndPopulation(unittest.TestCase):
    def test_defaults_come_from_the_case_block(self):
        case = Main.build_case(parse())
        for key, value in Main.CASE.items():
            self.assertEqual(case[key], value)
        self.assertEqual(case["case_id"], "main")
        self.assertEqual(case["geometry_name"], Main.geometry_name(Main.CASE))

    def test_options_override_the_case_block(self):
        case = Main.build_case(parse("--bays-x", "6", "--bays-y", "5", "--floors", "8", "--story-height-ft", "12.5",
                                     "--bay-x-ft", "24", "--bay-y-ft", "26.5", "--site", "sdc_d_high"))
        self.assertEqual((case["num_bay_x"], case["num_bay_y"], case["num_floor"]), (6, 5, 8))
        self.assertEqual((case["story_height_ft"], case["bay_x_width_ft"], case["bay_y_width_ft"]), (12.5, 24.0, 26.5))
        self.assertEqual(case["geometry_name"], "main_bx6_by5_s8_sh12p5_bwx24_bwy26p5_sdc_d_high")

    def test_a_population_member_has_no_notes(self):
        self.assertEqual(Main.population_notes(Main.build_case(parse(*MEMBER))), [])

    def test_a_building_off_the_population_is_named_but_not_refused(self):
        notes = Main.population_notes(Main.build_case(parse(*MEMBER, "--bays-x", "2", "--bay-y-ft", "15.2", "--site", "sdc_e")))
        text = " ".join(notes)
        self.assertIn("num_bay_x", text)
        self.assertIn("bay_y_width_ft", text)
        self.assertIn("sdc_e", text)

    def test_the_slenderness_cap_is_reported(self):
        notes = Main.population_notes(Main.build_case(parse("--bays-x", "3", "--bays-y", "3", "--floors", "9",
                                                            "--story-height-ft", "18", "--bay-x-ft", "18", "--bay-y-ft", "18")))
        self.assertTrue(any("height / least plan width" in note for note in notes))

    def test_the_population_import_leaves_the_path_and_environment_as_they_were(self):
        elsewhere = str(Path(tempfile.gettempdir()) / "elsewhere")
        # Other tests have already imported the population module; drop it from the cache so its module-level path
        # inserts really run here, and put the cache back afterwards.
        with mock.patch.dict(sys.modules), mock.patch.dict(os.environ, {"RC_STRUCTURE_DIR": elsewhere}):
            sys.modules.pop("Generate_Parameterized_Dataset", None)
            before = list(sys.path)
            Main.population_notes(Main.build_case(parse(*MEMBER)))
            self.assertEqual(sys.path, before)
            self.assertEqual(os.environ["RC_STRUCTURE_DIR"], elsewhere)
            loaded = sys.modules["Generate_Parameterized_Dataset"]
            self.assertEqual(Path(loaded.RC_DIR).resolve(), RC_DIR.resolve())


class RunnerArguments(unittest.TestCase):
    def test_the_batch_request_is_handed_to_the_grouped_runner(self):
        from Design import Grouped_Runner as runner
        args = parse(*MEMBER, "--probe-date", "2026-10-23")
        case = Main.build_case(args)
        column, beam = Main.batch_seed(case)
        runner_args = Main.runner_arguments(args, case, column, beam, Path("unused"))
        self.assertEqual(json.loads(runner_args.case_json), case)
        self.assertEqual(runner_args.profile, V2)
        self.assertEqual(runner_args.grouping, runner.DEFAULT_GROUPING)
        self.assertEqual((runner_args.max_feasibility_trials, runner_args.max_reduction_trials),
                         (Main.MAX_FEASIBILITY_TRIALS, Main.MAX_REDUCTION_TRIALS))
        self.assertTrue(runner_args.probe_assertions)
        self.assertEqual(runner_args.probe_date, "2026-10-23")
        self.assertEqual(runner_args.stages, ["design", "gravity_modal"])
        self.assertEqual(tuple(runner_args.seed_column), tuple(column))
        self.assertEqual(tuple(runner_args.seed_beam), tuple(beam))
        self.assertFalse(runner_args.plan_only)
        self.assertEqual(runner_args.output_dir, "unused")

    def test_the_seed_follows_the_batch_rule(self):
        from run_grouped_batch import seed_for
        case = Main.build_case(parse(*MEMBER))
        self.assertEqual(Main.batch_seed(case), seed_for(case))

    def test_design_only_and_plan_only(self):
        args = parse("--design-only", "--plan-only", "--no-grade80-retry")
        case = Main.build_case(args)
        runner_args = Main.runner_arguments(args, case, (20.0, 20.0, 6.0), (16.0, 30.0, 5.0), None)
        self.assertEqual(runner_args.stages, ["design"])
        self.assertTrue(runner_args.plan_only)
        self.assertTrue(runner_args.no_grade80_retry)
        self.assertIsNone(runner_args.output_dir)

    def test_the_method_line_follows_the_grouping(self):
        self.assertIn("perimeter special moment frames", Main.method_text("system_three_story_v1"))
        self.assertIn("three-story bands", Main.method_text("system_three_story_v1"))
        self.assertIn("core-ring", Main.method_text("system_perimeter_core_v1"))
        self.assertIn("one band over the height", Main.method_text("system_uniform_v1"))
        self.assertIn("legacy band policy", Main.method_text("story_pairs_from_base_orphan_merged__location_class_v1"))


class OutputFolder(unittest.TestCase):
    def test_each_run_gets_a_new_dated_folder(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Main, "OUTPUT_ROOT", Path(tmp) / "main"):
            args, case = parse(), Main.build_case(parse(*MEMBER))
            first, second = Main.new_output_dir(args, case), Main.new_output_dir(args, case)
            self.assertNotEqual(first, second)
            for folder in (first, second):
                self.assertTrue(folder.is_dir())
                self.assertEqual(list(folder.iterdir()), [])
                self.assertTrue(folder.name.startswith(case["geometry_name"] + "__"))

    def test_a_given_folder_must_be_new_or_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            used = Path(tmp) / "used"
            used.mkdir()
            (used / "grouped_design.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                Main.new_output_dir(SimpleNamespace(output_dir=str(used)), Main.build_case(parse(*MEMBER)))
            fresh = Main.new_output_dir(SimpleNamespace(output_dir=str(Path(tmp) / "fresh")), Main.build_case(parse(*MEMBER)))
            self.assertTrue(fresh.is_dir())


class HeartbeatProgress(unittest.TestCase):
    def test_counts_every_candidate_log_and_names_the_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            beat = Main.Heartbeat(out, 0, time.perf_counter())
            self.assertEqual(beat.candidates(), 0)
            self.assertEqual(beat.stage(), "design search")
            (out / "grouped_candidates.jsonl").write_text('{"index": 1}\n{"index": 2}\n', encoding="utf-8")
            (out / "grouped_candidates_grade80.jsonl").write_text('{"index": 1}\n', encoding="utf-8")
            self.assertEqual(beat.candidates(), 3)
            self.assertIn("3 candidate(s) decided", beat.line())
            (out / "grouped_design.json").write_text("{}", encoding="utf-8")
            self.assertIn("gravity and modal", beat.stage())

    def test_prints_on_the_console_while_running_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = io.StringIO()
            with mock.patch.object(sys, "__stdout__", stream):
                with Main.Heartbeat(tmp, 0.05, time.perf_counter()):
                    deadline = time.monotonic() + 5.0
                    while stream.getvalue().count("[main]") < 2 and time.monotonic() < deadline:
                        time.sleep(0.02)
                self.assertGreaterEqual(stream.getvalue().count("[main]"), 2)
                count = stream.getvalue().count("[main]")
                time.sleep(0.2)
                self.assertEqual(stream.getvalue().count("[main]"), count)

    def test_zero_interval_prints_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = io.StringIO()
            with mock.patch.object(sys, "__stdout__", stream):
                with Main.Heartbeat(tmp, 0, time.perf_counter()):
                    time.sleep(0.1)
            self.assertEqual(stream.getvalue(), "")


class InputChecks(unittest.TestCase):
    def test_bad_inputs_stop_before_any_folder_or_run(self):
        cases = (["--probe-date", "23-10-2026"], ["--seed-column", "20", "20", "6"],
                 ["--seed-column", "20", "22", "6", "--seed-beam", "16", "30", "5"],
                 ["--seed-column", "20", "20", "-6", "--seed-beam", "16", "30", "5"],
                 ["--max-feasibility-trials", "0"], ["--floors", "0"], ["--bay-x-ft", "-24"], ["--story-height-ft", "nan"])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Main, "OUTPUT_ROOT", Path(tmp) / "main"), \
                mock.patch("Design.Grouped_Runner.run") as run:
            for argv in cases:
                with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        Main.main([*MEMBER, *argv])
            run.assert_not_called()
            self.assertFalse((Path(tmp) / "main").exists())


class InterruptAndError(unittest.TestCase):
    def test_ctrl_c_during_plan_only(self):
        with mock.patch("Design.Grouped_Runner.run", side_effect=KeyboardInterrupt):
            code, text = quiet_main([*MEMBER, "--plan-only"])
        self.assertEqual(code, 130)
        self.assertIn("plan only: nothing was written", text)

    def test_ctrl_c_during_the_search_leaves_a_marker(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Main, "OUTPUT_ROOT", Path(tmp) / "main"), \
                mock.patch("Design.Grouped_Runner.run", side_effect=KeyboardInterrupt):
            code, text = quiet_main([*MEMBER, "--heartbeat-seconds", "0"])
            (folder,) = (Path(tmp) / "main").iterdir()
            marker = json.loads((folder / "interrupted.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 130)
        self.assertEqual(marker["interrupted_during"], "design search")
        self.assertIn("interrupted during the design search", text)
        self.assertNotIn(".lock", text)

    def test_an_error_before_the_folder_is_used_leaves_no_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Main, "OUTPUT_ROOT", Path(tmp) / "main"), \
                mock.patch("Design.Grouped_Runner.run", return_value={"status": "error", "error": "ValueError: bad seed",
                                                                       "elapsed_s": 0.1}):
            code, text = quiet_main([*MEMBER, "--heartbeat-seconds", "0"])
            self.assertEqual(list((Path(tmp) / "main").iterdir()), [])
        self.assertEqual(code, 1)
        self.assertIn("ValueError: bad seed", text)
        self.assertIn("before its folder was used", text)


class Report(unittest.TestCase):
    DESIGNED = {
        "status": "designed", "elapsed_s": 896.0, "accepted": False, "counts": {"pass": 1222, "fail": 0, "not_evaluated": 3},
        "fail_ids": [], "not_evaluated_ids": ["demands.torsional_irregularity"], "grouping_policy_selected": "system_three_story_v1",
        "longitudinal_grade_ksi": 60.0, "model_period_sec": 1.512,
        "search": {"stop_reason": "reduction_budget_exhausted", "stop_detail": "6 reduction candidates evaluated",
                   "counts": {"feasibility_candidates": 5, "reduction_candidates": 6}},
        "production_acceptance": {"accepted": False, "reason": "GENERATION_RELEASE_READY is False"},
        "systems": {"ties": {"s01_03__sfrs_column": [24.0, 24.0, 10.0], "s01_03__gravity_beam": [16.0, 30.0, 8.0]}},
        "sections_by_group": {"s01_03__column__corner": [24.0, 24.0, 10.0]},
        "quantities": {"concrete_total_yd3": 1454.7, "frame_steel_lb": 299200.0, "distinct_form_size_count": 4},
        "stage_results": {"gravity_modal": {"status": "completed", "all_checks_pass": True, "checks": {"gravity_converged": True},
                                            "modal": {"periods_elastic_reference_sec": [1.076, 1.031],
                                                      "periods_post_gravity_tangent_sec": [1.089]}}},
    }

    def report(self, result, out_dir=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            Main.print_report(result, out_dir)
        return out.getvalue()

    def test_a_designed_run(self):
        text = self.report(self.DESIGNED, Path("C:/runs/example"))
        for expected in ("Result      designed after 00:14:56", "reduction_budget_exhausted", "fy 60 ksi",
                         "counts {'pass': 1222", "GENERATION_RELEASE_READY is False", "s01_03__sfrs_column", "24 x 24",
                         "concrete 1454.7 yd3", "frame steel 149.6 t", "design model 1.512 s", "all checks pass: True",
                         "1.076, 1.031", "Files"):
            self.assertIn(expected, text)

    def test_missing_values_do_not_break_the_report(self):
        sparse = {"status": "designed", "elapsed_s": 1.0, "search": {}, "stage_results": {}}
        text = self.report(sparse)
        self.assertIn("fy ? ksi", text)
        self.assertIn("design model ? s", text)

    def test_no_feasible_design(self):
        text = self.report({"status": "no_feasible_design", "elapsed_s": 60.0,
                            "search_stop": {"reason": "feasibility_budget_exhausted", "detail": "16 candidates"},
                            "search_counts": {"feasibility_candidates": 16}})
        self.assertIn("no feasible design: feasibility_budget_exhausted", text)
        self.assertIn("grouped_search_failed.json", text)


class PlanOnly(unittest.TestCase):
    def test_plan_only_prints_the_request_and_writes_nothing(self):
        # A fresh interpreter: configuring the case changes Structure_Parameters globals (geometry, profile).
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-X", "utf8", "-B", str(RC_DIR / "Main.py"), *MEMBER, "--plan-only",
                                   "--probe-date", "2026-10-23"],
                                  cwd=tmp, capture_output=True, text=True, encoding="utf-8", timeout=600)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("member of the V2 population", proc.stdout)
        self.assertIn("grouping system_three_story_v1", proc.stdout)
        self.assertIn("Result      plan_only", proc.stdout)
        self.assertIn("(plan only: nothing is written)", proc.stdout)


if __name__ == "__main__":
    unittest.main()
