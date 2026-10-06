"""Contract tests with producer-shaped files; no OpenSees model build or NTHA."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Data_Generation import Run_Weighted_Seismic_Calibration as run
from Data_Generation import Build_Weighted_Seismic_Calibration_Plan as build
from Data_Generation.Seismic_Response_Score import default_policy


class CalibrationContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rc_patch = patch.object(run, "RC", self.root)
        self.rc_patch.start()
        self.addCleanup(self.rc_patch.stop)
        run._STOP.clear()
        self.addCleanup(run._STOP.clear)

    def producer(self):
        folder = self.root / "trial"
        ntha = folder / "ntha"
        ntha.mkdir(parents=True)
        count = 3
        self.arrays = {"story_drift": np.array([[.001, 0], [.01, 0], [.003, 0]], dtype=np.float32),
                       "floor_disp": np.array([[.1, 0], [1.0, 0], [.3, 0]], dtype=np.float32),
                       "hinge_rotation": np.ones((count, 4), dtype=np.float32) * .00125,
                       "hinge_moment": np.ones((count, 4), dtype=np.float32),
                       "hinge_tag_order": np.array([101, 102]),
                       "hinge_history_time": np.array([.01, .02, .03]),
                       "hinge_history_commit_count": np.array([1, 2, 3])}
        np.savez(ntha / "response_arrays.npz", **self.arrays)
        run.write_json(ntha / "global_parameters.json", {"num_floor": 1, "story_h_in": 100})
        run.write_json(ntha / "status.json", {"failed": False, "completed_steps": count, "npts_requested": count})
        fields = ("hinge_ele_tag", "material_type", "damage_ratio", "yielded", "theta_p", "theta_y_spring", "plastic_rotation", "rot_abs_max", "past_capping")
        with (ntha / "hinge_backbone.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for tag, plastic in ((101, .00025), (102, 0)):
                writer.writerow(dict(hinge_ele_tag=tag, material_type="IMKPeakOriented", damage_ratio=plastic/.001,
                    yielded=int(plastic>0), theta_p=.001, theta_y_spring=.001, plastic_rotation=plastic,
                    rot_abs_max=.001+plastic, past_capping=0))
        axis = {"compared": True, "matched_snapshots": count, "unmatched_snapshots": 0,
                "recorder_rows": count, "recorder_rows_that_are_not_snapshots": 0,
                "max_abs_moment_difference_kip_in": 1e-9, "max_abs_rotation_difference_rad": 1e-13,
                "max_abs_moment_kip_in": 1.0, "max_abs_rotation_rad": .00125,
                "commit_count_locates_the_same_recorder_row": True}
        self.manifest = {"status": "completed", "record_unchanged": True,
            "solver": {"failed": False, "truncated": False, "completed_steps": count, "requested_steps": count,
                       "scheduled_duration_sec": .03},
            "installed_design_verification": {"consistent": True}, "installed_topology": {"consistent": True},
            "identity": {"source_files_changed_since_design": []},
            "hinges": {"moment_rotation_export": {"available": True, "agreement_with_full_rate_recorders": {
                "compared": True, "matched_by": "time", "snapshots": count, "axes": {"y": deepcopy(axis), "z": deepcopy(axis)}}},
                "recorder_alignment": {a: {"time_strictly_increasing": True, "rows_match_snapshots_plus_substeps": True,
                    "ntha_snapshots": count, "rows_dropped_beyond_window": 0} for a in ("y", "z")}}}
        path = folder / "manifest.json"
        run.write_json(path, self.manifest)
        return path

    def test_real_producer_schema_without_all_pass_is_accepted(self):
        value = run.assess_trial(self.producer(), default_policy())
        self.assertTrue(value["usable"], value)
        self.assertEqual(value["hinge_denominator"], 2)
        self.assertEqual(value["metrics"]["fraction_hinges_yielded"], .5)
        self.assertAlmostEqual(value["metrics"]["peak_hinge_damage_ratio"], .25)
        self.assertAlmostEqual(value["score"]["score"], .45*5+.35+.15*.01/.0125+.05*.01/.015)

    def test_recorder_mismatch_unmatched_and_nonfinite_reject(self):
        path = self.producer()
        for key, value in (("unmatched_snapshots", 1), ("max_abs_moment_difference_kip_in", .5),
                           ("max_abs_rotation_difference_rad", float("nan")),
                           ("commit_count_locates_the_same_recorder_row", False)):
            with self.subTest(key=key):
                manifest = deepcopy(self.manifest)
                manifest["hinges"]["moment_rotation_export"]["agreement_with_full_rate_recorders"]["axes"]["y"][key] = value
                path.write_text(json.dumps(manifest))
                self.assertFalse(run.assess_trial(path, default_policy())["usable"])

    def test_partial_solver_and_production_status_disagreement_reject(self):
        path = self.producer()
        manifest = deepcopy(self.manifest)
        manifest["solver"]["completed_steps"] = 2
        run.write_json(path, manifest)
        self.assertFalse(run.assess_trial(path, default_policy())["usable"])
        run.write_json(path, self.manifest)
        run.write_json(path.parent / "ntha/status.json", {"failed": True, "completed_steps": 3, "npts_requested": 3})
        self.assertFalse(run.assess_trial(path, default_policy())["usable"])

    def test_missing_nonfinite_and_truncated_arrays_reject(self):
        path = self.producer()
        originals = deepcopy(self.arrays)
        for kind in ("missing", "nonfinite", "truncated", "shape"):
            with self.subTest(kind=kind):
                arrays = deepcopy(originals)
                if kind == "missing":
                    del arrays["hinge_moment"]
                elif kind == "nonfinite":
                    arrays["hinge_rotation"][0, 0] = np.inf
                elif kind == "truncated":
                    arrays["floor_disp"] = arrays["floor_disp"][:-1]
                else:
                    arrays["story_drift"] = arrays["story_drift"][:, :1]
                np.savez(path.parent / "ntha/response_arrays.npz", **arrays)
                self.assertFalse(run.assess_trial(path, default_policy())["usable"])

    def test_missing_snapshot_time_and_hinge_coverage_reject(self):
        path = self.producer()
        self.arrays["hinge_history_time"] = np.array([.02, .025, .03])
        self.arrays["hinge_tag_order"] = np.array([101, 999])
        np.savez(path.parent / "ntha/response_arrays.npz", **self.arrays)
        self.assertFalse(run.assess_trial(path, default_policy())["usable"])

    def test_damage_is_rotation_proxy_not_arbitrary_scalar(self):
        path = self.producer()
        csv_path = path.parent / "ntha/hinge_backbone.csv"
        text = csv_path.read_text().replace("0.25,1", "0.95,1")
        csv_path.write_text(text)
        value = run.assess_trial(path, default_policy())
        self.assertFalse(value["usable"])
        self.assertIn("inconsistent_rotation_damage_proxy", value["rejection_reasons"])

    def case(self, *, open_ids=(), accepted=True, failed=0):
        folder = self.root / "designs/case_0001"
        folder.mkdir(parents=True, exist_ok=True)
        counts = {"pass": 1, "fail": failed, "not_evaluated": len(open_ids)}
        checks = [{"id": "member", "status": "pass"}] + [{"id": name, "status": "not_evaluated"} for name in open_ids]
        record = {"request_identity": {"source_sha256": {"design.py": "source"},
                  "policy": {"model_profile": {"id": run.RESEARCH_PROFILE}}},
                  "qualification": {"accepted": accepted, "counts": counts, "checks": checks}}
        run.write_json(folder / "design.json", record)
        result = {"status": "designed", "counts": counts, "accepted": accepted,
                  "design_sha256": run.digest(folder / "design.json"), "profile_id": run.RESEARCH_PROFILE,
                  "fail_ids": ["member"] if failed else [], "not_evaluated_ids": list(open_ids), "probe_assertions": True}
        result["stage_results"] = {"gravity_modal": {"status": "completed", "all_checks_pass": True, "profile_id": run.RESEARCH_PROFILE}}
        run.write_json(folder / "result.json", result)
        return record, result

    def test_collect_accepts_fresh_qualified_and_rejects_failed_stale_wrong_profile(self):
        self.case()
        cases = build.collect_cases(["designs"], ["case_0001"], run.RESEARCH_PROFILE, "qualified-only", {"design.py": "source"})
        self.assertEqual(cases[0]["design_root"], "designs")
        self.assertFalse(cases[0]["training_release"])
        for profile, sources in (("wrong", {"design.py": "source"}), (run.RESEARCH_PROFILE, {"design.py": "changed"})):
            with self.assertRaises(ValueError):
                build.collect_cases(["designs"], ["case_0001"], profile, "qualified-only", sources)
        self.case(failed=1)
        with self.assertRaises(ValueError):
            build.collect_cases(["designs"], ["case_0001"], run.RESEARCH_PROFILE, "qualified-only", {"design.py": "source"})

    def test_only_explicit_single_open_m1_diagnostic_permitted(self):
        self.case(open_ids=[run.M1_CHECK], accepted=False)
        args = (["designs"], ["case_0001"], run.RESEARCH_PROFILE)
        with self.assertRaises(ValueError):
            build.collect_cases(*args, "qualified-only", {"design.py": "source"})
        case = build.collect_cases(*args, "open-m1-diagnostic", {"design.py": "source"})[0]
        self.assertFalse(case["original_qualification"]["qualification_accepted"])
        self.assertTrue(case["original_probe_assertions"])
        self.case(open_ids=[run.M1_CHECK, "another_open_check"], accepted=False)
        with self.assertRaises(ValueError):
            build.collect_cases(*args, "open-m1-diagnostic", {"design.py": "source"})

    def test_duplicate_missing_cases_and_external_roots_reject(self):
        self.case()
        for roots, ids in ((["designs", "designs"], ["case_0001"]), (["designs"], ["case_0002"]),
                           (["designs"], ["case_0001", "case_0001"]), ([str(self.root.parent)], ["case_0001"])):
            with self.assertRaises(ValueError):
                build.collect_cases(roots, ids, run.RESEARCH_PROFILE, "qualified-only", {"design.py": "source"})

    def test_failed_gravity_stage_never_admitted_by_designed_status(self):
        _, result = self.case()
        result["stage_results"]["gravity_modal"]["all_checks_pass"] = False
        run.write_json(self.root / "designs/case_0001/result.json", result)
        with self.assertRaisesRegex(ValueError, "gravity/modal"):
            build.collect_cases(["designs"], ["case_0001"], run.RESEARCH_PROFILE, "qualified-only", {"design.py": "source"})

    def test_hundred_jobs_are_deterministic_disjoint_and_balanced(self):
        cases = [{"case_id": f"case_{i:04d}"} for i in range(1, 101)]
        args = dict(seed=20261006, shards=4, assignment="one-per-case")
        jobs = build.assign_jobs(cases, list(range(1, 55)), **args)
        self.assertEqual(jobs, build.assign_jobs(list(reversed(cases)), list(range(54, 0, -1)), **args))
        self.assertEqual(len(jobs), 100)
        self.assertEqual([sum(j["shard"] == n for j in jobs) for n in range(1, 5)], [25]*4)
        self.assertEqual(len({j["result_id"] for j in jobs[:54]}), 54)
        self.assertEqual(len({j["job_id"] for j in jobs}), 100)

    def test_search_uses_lowest_tested_passing_and_stops_on_unusable(self):
        search = dict(minimum_scale=.125, initial_scale=1., maximum_scale=8., maximum_trials_per_pair=7, relative_bracket_tolerance=.1)
        trials = []
        self.assertEqual(run.next_scale(trials, search), 1)
        trials.append({"scale": 1., "usable": True, "score": {"target_reached": False}})
        self.assertEqual(run.next_scale(trials, search), 2.)
        trials.append({"scale": 2., "usable": True, "score": {"target_reached": True}})
        self.assertAlmostEqual(run.next_scale(trials, search), 2**.5, places=6)
        trials.append({"scale": 1.4, "usable": True, "score": {"target_reached": True}})
        self.assertEqual(run.select_best(trials)["scale"], 1.4)
        trials.append({"scale": 1.18, "usable": False})
        self.assertIsNone(run.next_scale(trials, search))

    def test_exclusive_lock_and_changed_resume_artifacts(self):
        output = self.root / "output"
        with run.execution_lock(output, "plan"):
            with self.assertRaises(ValueError):
                with run.execution_lock(output, "plan"):
                    self.fail("second owner acquired lock")
        self.assertFalse((output / ".calibration.lock").exists())
        job_dir = output / "case_0001_pair_11"
        job_dir.mkdir()
        (job_dir / "worker.log").write_text("complete")
        history = {"trials": []}
        run.save_search(history, job_dir, output)
        run.verify_search_artifacts(history, output)
        (job_dir / "worker.log").write_text("changed")
        with self.assertRaises(ValueError):
            run.verify_search_artifacts(history, output)

    def test_completed_search_resume_does_not_launch_process(self):
        case = {"case_id": "case_0001"}
        job = dict(job_id="case_0001_pair_11", case_id="case_0001", result_id=11, shard=1)
        plan = {"plan_sha256": "plan"}
        out = self.root / job["job_id"]
        out.mkdir()
        (out / "worker.log").write_text("finished")
        history = dict(job, plan_sha256="plan", status="completed", trials=[], target_reached=False)
        run.save_search(history, out, self.root)
        with patch.object(run.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            returned = run.run_pair(case, job, plan, self.root, {}, {})
        self.assertEqual(returned["status"], "completed")

    def test_input_identity_change_and_plan_digest_are_detected(self):
        path = self.root / "motion.txt"
        path.write_text("original")
        with patch.object(run, "source_identity", return_value={"file": "sha"}):
            inputs = {"motion.txt": run.digest(path)}
            self.assertTrue(run.inputs_unchanged({"file": "sha"}, inputs))
            path.write_text("changed")
            self.assertFalse(run.inputs_unchanged({"file": "sha"}, inputs))
        self.assertEqual(run.canonical_digest({"b": 2, "a": 1}), run.canonical_digest({"a": 1, "b": 2}))
        self.assertNotEqual(run.canonical_digest({"a": 1}), run.canonical_digest({"a": 2}))

    def test_canary_submits_only_first_assigned_job(self):
        jobs = build.assign_jobs([{"case_id": f"case_{n:04d}"} for n in (1, 2)], [11], seed=1, shards=1, assignment="one-per-case")
        cases = [{"case_id": j["case_id"], "design_root": "designs"} for j in jobs]
        component = {"x": {"path": "x.txt"}, "y": {"path": "y.txt"}}
        plan = {"plan_sha256": "plan", "workers": 1, "cases": cases, "score_policy": default_policy(),
                "frozen": {"motions": {"11": {"components": component}}}}
        inputs = {f"designs/{c['case_id']}/{n}": "sha" for c in cases for n in ("design.json", "result.json")}
        inputs.update({"x.txt": "sha", "y.txt": "sha"})
        args = argparse.Namespace(shard=1, first_job_only=True)
        def fake(case, job, *unused):
            return dict(job, trials=[{"usable": True}], target_reached=True, stop_reason="bound")
        with patch.object(run, "run_pair", side_effect=fake) as called:
            self.assertEqual(run.execute_jobs(plan, args, self.root, jobs, {}, inputs), 0)
        self.assertEqual(called.call_count, 1)
        progress = run.read_json(self.root / "progress.json")
        self.assertEqual(progress["status"], "partial_canary")
        self.assertEqual(progress["planned_jobs"], [j["job_id"] for j in jobs])

    def test_preflight_never_starts_process_or_creates_output(self):
        plan = {"shards": 1, "jobs": [{"job_id": "case_0001_pair_11", "shard": 1}],
                "output_root": "outputs/new", "plan_sha256": "plan", "use_mode": "qualified-only",
                "frozen": {"sources": {}, "input_sha256": {}}, "search": {"maximum_trials_per_pair": 7},
                "score_policy": default_policy()}
        run.write_json(self.root / "plan.json", plan)
        with patch.object(run, "validate_plan", return_value=plan), patch.object(run, "verify_frozen_plan", return_value=plan["frozen"]), \
                patch.object(run.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            self.assertEqual(run.main(["--plan", "plan.json", "--shard", "1", "--preflight-only"]), 0)
        self.assertFalse((self.root / "outputs").exists())

    def test_timeout_keeps_failed_log_and_score(self):
        class Process:
            killed = False
            def poll(self):
                return -9 if self.killed else None
            def kill(self):
                self.killed = True
            def wait(self, timeout=None):
                return -9
        plan = {"plan_sha256": "plan", "search": {"minimum_scale": .125, "initial_scale": 1., "maximum_scale": 8.,
                "maximum_trials_per_pair": 7, "relative_bracket_tolerance": .1}, "set_name": "set",
                "trial_timeout_seconds": 0, "score_policy": default_policy()}
        case = {"case_id": "case_0001", "design_root": "designs", "profile": run.RESEARCH_PROFILE}
        job = {"case_id": "case_0001", "result_id": 11, "job_id": "case_0001_pair_11", "shard": 1}
        with patch.object(run, "inputs_unchanged", return_value=True), patch.object(run.subprocess, "Popen", return_value=Process()):
            history = run.run_pair(case, job, plan, self.root, {}, {})
        self.assertFalse(history["target_reached"])
        self.assertEqual(history["stop_reason"], "unusable_trial")
        self.assertEqual(history["trials"][0]["interruption"], "timeout")
        self.assertTrue((self.root / job["job_id"] / "trial_01/worker.log").exists())
        self.assertTrue((self.root / job["job_id"] / "trial_01/score.json").exists())

    def test_plan_builder_freezes_roster_design_result_motion_and_runtime(self):
        from Design import Design_Driver as driver
        self.case()
        metadata = self.root / "Ground_Motions/metadata"
        metadata.mkdir(parents=True)
        for name in ("record_manifest.csv", "record_sets.csv"):
            (metadata / name).write_text("catalog")
        for name in ("x.txt", "y.txt"):
            (self.root / name).write_text("0\n1\n0\n")
        run.write_json(self.root / "roster.json", {"cases": [{"case_id": "case_0001"}]})
        motions = {"11": {"components": {axis: {"path": f"{axis}.txt", "sha256": run.digest(self.root / f"{axis}.txt"), "npts": 3} for axis in ("x", "y")}}}
        args = build.build_parser().parse_args(["--design-root", "designs", "--case-manifest", "roster.json",
            "--set-name", "set", "--pairs", "11", "--assignment", "one-per-case", "--shards", "1",
            "--output-root", "outputs/new", "--output", "plan.json"])
        with patch.object(driver, "source_sha256", return_value={"design.py": "source"}), \
             patch.object(run, "source_identity", return_value={"controller.py": "sha"}), \
             patch.object(run, "runtime_identity", return_value={"python": "same"}), \
             patch.object(run, "motion_identity", return_value=motions):
            plan = build.build_plan(args)
            run.verify_frozen_plan(plan)
            self.assertIn("roster.json", plan["frozen"]["input_sha256"])
            self.assertIn("designs/case_0001/result.json", plan["frozen"]["input_sha256"])
            self.assertEqual(plan["frozen"]["runtime"], {"python": "same"})
            self.assertEqual(len(plan["jobs"]), 1)
            altered = deepcopy(plan)
            altered["search"]["maximum_scale"] = 9.
            with self.assertRaisesRegex(ValueError, "Plan identity"):
                run.validate_plan(altered)
            (self.root / "x.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Input/motion identity"):
                run.verify_frozen_plan(plan)
            with patch.object(run, "runtime_identity", return_value={"python": "changed"}):
                with self.assertRaisesRegex(ValueError, "Source/runtime"):
                    run.verify_frozen_plan(plan)

    def test_interruption_kills_worker_and_preserves_nonpassing_trial(self):
        class Process:
            killed = False
            def poll(self):
                run._STOP.set()
                return -9 if self.killed else None
            def kill(self):
                self.killed = True
            def wait(self, timeout=None):
                return -9
        process = Process()
        plan = {"plan_sha256": "plan", "search": {"minimum_scale": .125, "initial_scale": 1., "maximum_scale": 8.,
                "maximum_trials_per_pair": 7, "relative_bracket_tolerance": .1}, "set_name": "set",
                "trial_timeout_seconds": 3600, "score_policy": default_policy()}
        case = {"case_id": "case_0001", "design_root": "designs", "profile": run.RESEARCH_PROFILE}
        job = {"case_id": "case_0001", "result_id": 11, "job_id": "case_0001_pair_11", "shard": 1}
        with patch.object(run, "inputs_unchanged", return_value=True), patch.object(run.subprocess, "Popen", return_value=process):
            history = run.run_pair(case, job, plan, self.root, {}, {})
        self.assertTrue(process.killed)
        self.assertEqual(history["status"], "interrupted")
        self.assertEqual(history["trials"][0]["interruption"], "interrupted")
        self.assertNotIn("score", history["trials"][0])
        run.verify_search_artifacts(history, self.root)


if __name__ == "__main__":
    unittest.main()
