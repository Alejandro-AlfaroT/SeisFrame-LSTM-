"""Portable collection checks using tiny synthetic output files, never an NTHA.

Plan validation has its own runner tests. Here it is mocked so collection tests
remain independent of the evolving plan-building schema and frozen checkout.
Trial assessment and scoring use the real code and files.
"""
from copy import deepcopy
import csv
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
from Data_Generation import Collect_Weighted_Seismic_Calibration as collector
from Data_Generation import Run_Weighted_Seismic_Calibration as runner
from Data_Generation.Seismic_Response_Score import default_policy, score_response


class PortableCollection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.plan = {
            "plan_sha256": "frozen-test-plan", "shards": 2,
            "frozen": {"sources": {"model.py": "same-source"}, "runtime": {"python": "test"}},
            "score_policy": default_policy(),
            "search": {"minimum_scale": .25, "initial_scale": 1.0, "maximum_scale": 3.0,
                       "maximum_trials_per_pair": 1, "relative_bracket_tolerance": .05},
            "jobs": [{"job_id": f"case_{n:04d}__peer_11", "case_id": f"case_{n:04d}",
                      "result_id": 11, "shard": n} for n in (1, 2)],
        }
        self.validation = mock.patch.object(collector, "validate_plan", side_effect=lambda p: p)
        self.validation.start()
        self.addCleanup(self.validation.stop)
        self.roots = [self.base / f"device_{n:02d}" for n in (1, 2)]
        for root, job in zip(self.roots, self.plan["jobs"]):
            self.make_device(root, job)

    def make_device(self, root, job):
        trial_root = root / job["job_id"] / "trial_01"
        manifest = trial_root / job["case_id"] / "peer_11_scale_1" / "manifest.json"
        ntha = manifest.parent / "ntha"
        ntha.mkdir(parents=True)
        tags = np.arange(100, 110)
        rotations = np.full((2, 20), .0005)
        rotations[1, 0] = .013
        np.savez(ntha / "response_arrays.npz", story_drift=np.array([[.005, 0], [.020, 0]]),
                 floor_disp=np.array([[.6, 0], [2.4, 0]]), hinge_rotation=rotations,
                 hinge_moment=np.ones((2, 20)), hinge_history_time=np.array([.01, .02]),
                 hinge_history_commit_count=np.array([1, 2]), hinge_tag_order=tags)
        runner.write_json(ntha / "global_parameters.json", {"num_floor": 1, "story_h_in": 120})
        runner.write_json(ntha / "status.json", {"failed": False, "completed_steps": 2, "npts_requested": 2})
        rows = [{"hinge_ele_tag": int(tag), "material_type": "IMKPeakOriented", "theta_p": .04,
                 "theta_y_spring": .001, "plastic_rotation": .012 if i == 0 else 0,
                 "rot_abs_max": .013 if i == 0 else .0005, "damage_ratio": .3 if i == 0 else 0,
                 "yielded": 1 if i == 0 else 0, "past_capping": 0} for i, tag in enumerate(tags)]
        with (ntha / "hinge_backbone.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        agreement = {"compared": True, "snapshots": 2, "axes": {}}
        for axis in ("y", "z"):
            agreement["axes"][axis] = {
                "compared": True, "matched_snapshots": 2, "unmatched_snapshots": 0,
                "commit_count_locates_the_same_recorder_row": True,
                "max_abs_moment_difference_kip_in": 0, "max_abs_moment_kip_in": 1,
                "max_abs_rotation_difference_rad": 0, "max_abs_rotation_rad": .013,
            }
        runner.write_json(manifest, {
            "status": "completed", "record_unchanged": True, "case_id": job["case_id"],
            "solver": {"failed": False, "truncated": False, "completed_steps": 2,
                       "requested_steps": 2, "scheduled_duration_sec": .02},
            "installed_design_verification": {"consistent": True}, "installed_topology": {"consistent": True},
            "identity": {"source_files_changed_since_design": []},
            "hinges": {"moment_rotation_export": {"available": True, "agreement_with_full_rate_recorders": agreement},
                       "recorder_alignment": {axis: {"time_strictly_increasing": True,
                           "rows_match_snapshots_plus_substeps": True, "ntha_snapshots": 2,
                           "rows_dropped_beyond_window": 0} for axis in ("y", "z")}},
        })
        trial = {"trial": 1, "scale": 1.0, "returncode": 0, "interruption": None,
                 "manifest": manifest.relative_to(root).as_posix(),
                 **runner.assess_trial(manifest, self.plan["score_policy"])}
        self.assertTrue(trial["usable"], trial.get("rejection_reasons"))
        runner.write_json(trial_root / "score.json", trial)
        runner.write_json(trial_root / "command.json", {
            "job_id": job["job_id"], "plan_sha256": self.plan["plan_sha256"], "scale": 1.0})
        (trial_root / "worker.log").write_text("Synthetic fixture; no analysis executed.\n", encoding="utf-8")
        search = {**job, "plan_sha256": self.plan["plan_sha256"], "status": "completed", "trials": [trial],
                  "best_trial": deepcopy(trial), "best_tested_passing_scale": 1.0, "target_reached": True,
                  "stop_reason": "budget_or_scale_bound_or_bracket_tolerance"}
        runner.save_search(search, root / job["job_id"], root)
        runner.write_json(root / "identity.json", {"shard": job["shard"],
                          "plan_sha256": self.plan["plan_sha256"], "frozen": self.plan["frozen"]})
        runner.write_json(root / "progress.json", {
            "shard": job["shard"], "plan_sha256": self.plan["plan_sha256"], "status": "completed", "errors": [],
            "planned_jobs": [job["job_id"]], "searches": [search],
            "search_manifest_sha256": {f"{job['job_id']}/search.json": runner.digest(root / job["job_id"] / "search.json")},
        })

    def search(self, n=0):
        return runner.read_json(self.roots[n] / self.plan["jobs"][n]["job_id"] / "search.json")

    def save_search(self, search, n=0):
        root, job = self.roots[n], self.plan["jobs"][n]
        runner.save_search(search, root / job["job_id"], root)
        progress = runner.read_json(root / "progress.json")
        progress["searches"] = [search]
        progress["search_manifest_sha256"] = {f"{job['job_id']}/search.json": runner.digest(root / job["job_id"] / "search.json")}
        runner.write_json(root / "progress.json", progress)

    def save_trial(self, search, n=0):
        root, job = self.roots[n], self.plan["jobs"][n]
        trial = search["trials"][0]
        runner.write_json(root / job["job_id"] / "trial_01" / "score.json", trial)
        if search.get("best_trial") is not None:
            search["best_trial"] = deepcopy(trial)
        self.save_search(search, n)

    def change_progress(self, update, n=0):
        path = self.roots[n] / "progress.json"
        progress = runner.read_json(path)
        update(progress)
        runner.write_json(path, progress)

    def assertRejected(self, text):
        result = collector.collect(self.plan, self.roots)
        self.assertFalse(result["complete"])
        self.assertTrue(result["errors"], result)
        self.assertIn(text, str(result["errors"]))
        return result

    def test_copied_complete_devices_recompute_real_metrics(self):
        copied = []
        for root in self.roots:
            destination = self.base / "different_drive" / root.name
            shutil.copytree(root, destination)
            copied.append(destination)
        # The collection must not depend on the original machine's paths.
        for root in self.roots:
            shutil.rmtree(root)
        result = collector.collect(self.plan, copied)
        self.assertTrue(result["complete"], result["errors"])
        self.assertTrue(result["all_searches_reached_target"])
        self.assertEqual(result["verified_searches"], 2)
        self.assertEqual(result["searches_reaching_target"], 2)
        self.assertFalse(result["training_eligible"])
        row = result["rows"][0]
        self.assertAlmostEqual(row["fraction_hinges_yielded"], .1)
        self.assertAlmostEqual(row["peak_hinge_damage_ratio"], .3)
        self.assertAlmostEqual(row["peak_interstory_drift_ratio"], .02)
        self.assertAlmostEqual(row["roof_drift_ratio"], .02)
        self.assertAlmostEqual(row["score"], .45 + .35 * 1.2 + .15 * 1.6 + .05 * 4 / 3)
        self.assertTrue(all(not r["training_eligible"] for r in result["rows"]))

    def test_missing_device_is_incomplete(self):
        result = collector.collect(self.plan, self.roots[:1])
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_jobs"], [self.plan["jobs"][1]["job_id"]])
        self.assertFalse(result["errors"])

    def test_running_device_is_not_complete_even_with_all_results(self):
        self.change_progress(lambda p: p.update(status="running"))
        result = collector.collect(self.plan, self.roots)
        self.assertFalse(result["complete"])
        self.assertEqual(result["verified_searches"], 2)

    def test_missing_search_is_incomplete(self):
        self.change_progress(lambda p: p.update(searches=[], search_manifest_sha256={}))
        result = collector.collect(self.plan, self.roots)
        self.assertFalse(result["complete"])
        self.assertIn(self.plan["jobs"][0]["job_id"], result["missing_jobs"])

    def test_duplicate_device_shard_rejected(self):
        result = collector.collect(self.plan, [*self.roots, self.roots[0]])
        self.assertFalse(result["complete"])
        self.assertIn("Duplicate shard", str(result["errors"]))

    def test_foreign_source_identity_rejected(self):
        path = self.roots[0] / "identity.json"
        data = runner.read_json(path)
        data["frozen"]["sources"]["model.py"] = "different"
        runner.write_json(path, data)
        self.assertRejected("Foreign plan or source/input identity")

    def test_foreign_progress_plan_rejected(self):
        self.change_progress(lambda p: p.update(plan_sha256="different"))
        self.assertRejected("Progress identity mismatch")

    def test_duplicate_planned_job_rejected(self):
        self.change_progress(lambda p: p["planned_jobs"].append(p["planned_jobs"][0]))
        self.assertRejected("Unexpected planned jobs")

    def test_duplicate_search_rejected(self):
        self.change_progress(lambda p: p["searches"].append(p["searches"][0]))
        self.assertRejected("Duplicate or foreign searches")

    def test_foreign_search_rejected(self):
        self.change_progress(lambda p: p["searches"][0].update(job_id="foreign"))
        self.assertRejected("Duplicate or foreign searches")

    def test_foreign_folder_rejected(self):
        (self.roots[0] / "other_job").mkdir()
        self.assertRejected("Unexpected search folders")

    def test_modified_search_manifest_rejected(self):
        path = self.roots[0] / self.plan["jobs"][0]["job_id"] / "search.json"
        data = runner.read_json(path)
        data["status"] = "modified"
        runner.write_json(path, data)
        self.assertRejected("Missing or changed search manifest")

    def test_progress_search_disagreement_rejected(self):
        self.change_progress(lambda p: p["searches"][0].update(target_reached=False))
        self.assertRejected("Progress/search disagreement")

    def test_artifact_tampering_rejected(self):
        path = self.roots[0] / self.plan["jobs"][0]["job_id"] / "trial_01" / "worker.log"
        path.write_text("Modified after search completed", encoding="utf-8")
        self.assertRejected("Artifact changed")

    def test_missing_artifact_rejected(self):
        path = self.roots[0] / self.plan["jobs"][0]["job_id"] / "trial_01" / "worker.log"
        path.unlink()
        self.assertRejected("Missing or unexpected artifacts")

    def test_unexpected_artifact_rejected(self):
        path = self.roots[0] / self.plan["jobs"][0]["job_id"] / "foreign.txt"
        path.write_text("unplanned", encoding="utf-8")
        self.assertRejected("Missing or unexpected artifacts")

    def test_inventoried_score_disagrees_with_summary_rejected(self):
        search = self.search()
        path = self.roots[0] / self.plan["jobs"][0]["job_id"] / "trial_01" / "score.json"
        trial = runner.read_json(path)
        trial["metrics"]["peak_hinge_damage_ratio"] = .9
        runner.write_json(path, trial)
        self.save_search(search)
        self.assertRejected("Trial score file differs from search")

    def test_self_consistent_forged_score_does_not_replace_response_evidence(self):
        search = self.search()
        trial = search["trials"][0]
        trial["metrics"]["peak_hinge_damage_ratio"] = .9
        trial["score"] = score_response(trial["metrics"], self.plan["score_policy"])
        self.save_trial(search)
        self.assertRejected("Trial assessment does not reproduce")

    def test_raw_response_change_detected_even_after_rehashing_inventory(self):
        search = self.search()
        manifest = self.roots[0] / search["trials"][0]["manifest"]
        path = manifest.parent / "ntha" / "response_arrays.npz"
        with np.load(path) as data:
            arrays = dict(data)
        arrays["story_drift"][1, 0] = .03
        np.savez(path, **arrays)
        self.save_search(search)
        self.assertRejected("Trial assessment does not reproduce")

    def test_incomplete_solver_cannot_be_marked_usable(self):
        search = self.search()
        path = self.roots[0] / search["trials"][0]["manifest"]
        manifest = runner.read_json(path)
        manifest["solver"]["completed_steps"] = 1
        runner.write_json(path, manifest)
        self.save_search(search)
        self.assertRejected("Trial assessment does not reproduce")

    def test_failed_process_cannot_be_marked_usable(self):
        search = self.search()
        search["trials"][0]["returncode"] = 1
        self.save_trial(search)
        self.assertRejected("Unsuccessful worker marked usable")

    def test_foreign_manifest_path_rejected(self):
        search = self.search()
        search["trials"][0]["manifest"] = "../outside.json"
        self.save_trial(search)
        self.assertRejected("Trial manifest belongs to another run")

    def test_trial_scale_must_follow_plan(self):
        search = self.search()
        search["trials"][0]["scale"] = 2.0
        self.save_trial(search)
        self.assertRejected("Trial does not follow the planned search")

    def test_early_stopped_search_cannot_be_marked_completed(self):
        self.plan["search"]["maximum_trials_per_pair"] = 3
        self.assertRejected("before its planned stopping condition")

    def test_search_target_flag_must_match_best_trial(self):
        search = self.search()
        search["target_reached"] = False
        self.save_search(search)
        self.assertRejected("Target flag differs")

    def test_unusable_trial_is_reported_without_passing_score(self):
        search = self.search()
        trial = search["trials"][0]
        trial.update(usable=False, rejection_reasons=["timeout_or_worker_exit_or_missing_manifest"],
                     returncode=1, interruption="timeout")
        trial.pop("score")
        search.update(best_trial=None, best_tested_passing_scale=None, target_reached=False, stop_reason="unusable_trial")
        self.save_trial(search)
        result = collector.collect(self.plan, self.roots)
        self.assertTrue(result["complete"], result["errors"])
        self.assertFalse(result["all_searches_reached_target"])
        self.assertEqual(result["searches_with_unusable_trials"], 1)
        self.assertEqual(result["searches_without_passing_trial"], 1)
        self.assertEqual(result["rows"][0]["status"], "unusable")
        self.assertIsNone(result["rows"][0]["score"])
        self.assertFalse(result["rows"][0]["training_eligible"])

    def test_cli_partial_exit_status_and_saved_summary(self):
        plan_path = self.base / "plan.json"
        runner.write_json(plan_path, self.plan)
        output = self.base / "collection"
        argv = ["--plan", str(plan_path), "--device-root", str(self.roots[0]), "--output", str(output)]
        self.assertEqual(collector.main(argv), 2)
        self.assertFalse(runner.read_json(output / "calibration_summary.json")["complete"])
        self.assertTrue((output / "calibration_summary.csv").is_file())
        self.assertEqual(collector.main([*argv, "--allow-partial"]), 0)

    def test_cli_cannot_overwrite_another_plan_collection(self):
        plan_path, output = self.base / "plan.json", self.base / "collection"
        runner.write_json(plan_path, self.plan)
        runner.write_json(output / "calibration_summary.json", {"plan_sha256": "other"})
        with self.assertRaisesRegex(ValueError, "different plan"):
            collector.main(["--plan", str(plan_path), "--device-root", str(self.roots[0]), "--output", str(output)])


class ArtifactContainment(unittest.TestCase):
    def test_path_escape_forms_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ("../outside.txt", "/outside.txt", "C:/outside.txt", "job\\file.txt", ""):
                with self.subTest(label=label), self.assertRaises(ValueError):
                    collector.artifact_path(root, label)


class DesignInventory(unittest.TestCase):
    def test_duplicate_case_roster_rejected_before_reading(self):
        from Data_Generation import Inspect_Calibration_Designs as inventory
        with mock.patch.object(inventory.builder, "collect_cases") as read:
            with self.assertRaisesRegex(ValueError, "unique cases"):
                inventory.inspect_cases([], ["case_0001", "case_0001"], "profile", "qualified-only", {})
            read.assert_not_called()

    def test_invalid_case_name_is_an_explicit_exclusion(self):
        from Data_Generation import Inspect_Calibration_Designs as inventory
        result = inventory.inspect_cases([], ["../outside"], "profile", "qualified-only", {})
        self.assertEqual(result["eligible_count"], 0)
        self.assertEqual(result["excluded_count"], 1)
        self.assertIn("Invalid case directory name", result["excluded"][0]["reason"])

    def test_inventory_reports_every_eligible_failed_and_missing_case(self):
        from Data_Generation import Inspect_Calibration_Designs as inventory
        def inspect_one(roots, cases, profile, mode, sources):
            if cases == ["case_0002"]:
                raise ValueError("Failed design: column shear")
            if cases == ["case_0003"]:
                raise FileNotFoundError("result.json missing")
            return [{"case_id": cases[0], "training_release": False}]
        with mock.patch.object(inventory.builder, "collect_cases", side_effect=inspect_one) as read:
            result = inventory.inspect_cases(["root"], [f"case_{n:04d}" for n in range(1, 5)],
                                             "profile", "open-m1-diagnostic", {"source": "sha"})
        self.assertEqual(read.call_count, 4)
        self.assertEqual((result["requested_count"], result["eligible_count"], result["excluded_count"]), (4, 2, 2))
        self.assertEqual([c["case_id"] for c in result["eligible"]], ["case_0001", "case_0004"])
        self.assertEqual([c["case_id"] for c in result["excluded"]], ["case_0002", "case_0003"])
        self.assertIn("column shear", result["excluded"][0]["reason"])
        self.assertIn("result.json missing", result["excluded"][1]["reason"])


if __name__ == "__main__":
    unittest.main()
