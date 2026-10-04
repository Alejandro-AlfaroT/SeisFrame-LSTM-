"""The grouped design runner in a fresh interpreter on a tiny off-population frame (Design.Grouped_Runner, 2026-10-02).

The runner's real path: the V2 analysis profile and Risk Category III applied as the screening applies them,
PROBE assertions (so the slab layout and the joint evaluation run), the grouped search with explicit budgets,
the record, the candidate log, the summary and the gravity / modal stage. What is checked is the plumbing:
the files a run leaves, the separate statuses, the refusal of another request in the same directory, and the
outcome of a search that ends without a feasible design. The frame is a fixture, not a screening case.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
V2 = "v2_nonlinear_flexure_screening_v1"
CASE = {"case_id": "case_9002", "num_bay_x": 2, "num_bay_y": 1, "num_floor": 2, "story_height_ft": 12,
        "bay_x_width_ft": 15, "bay_y_width_ft": 15, "seismic_site": "sdc_d_low",
        "geometry_name": "fixture_bx2_by1_s2_sh12ft_bwx15ft_bwy15ft_sdc_d_low"}
PROBE_DATE = "2026-10-02"
SEED = ["--seed-column", "20", "20", "5", "--seed-beam", "14", "20", "4"]


def run(out_dir, *extra, feasibility=3, reduction=2, stages=("design", "gravity_modal")):
    command = [sys.executable, str(RC_DIR / "Design" / "Grouped_Runner.py"), "--case-json", json.dumps(CASE), "--profile", V2,
               "--probe-assertions", "--probe-date", PROBE_DATE, *SEED,
               "--max-feasibility-trials", str(feasibility), "--max-reduction-trials", str(reduction),
               "--stages", *stages, *(["--output-dir", str(out_dir)] if out_dir is not None else []), *extra]
    return subprocess.run(command, cwd=str(RC_DIR), capture_output=True, text=True, timeout=1800)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RunnerFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "attempt_1"
        cls.proc = run(cls.out)
        cls.result = load(cls.out / "result.json") if (cls.out / "result.json").exists() else None
        cls.record = load(cls.out / "grouped_design.json") if (cls.out / "grouped_design.json").exists() else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_run_leaves_the_record_the_candidate_log_the_summary_and_the_stage_result(self):
        self.assertIsNotNone(self.result, self.proc.stderr[-2000:])
        self.assertEqual(self.result["status"], "designed", self.result.get("error"))
        for name in ("grouped_design.json", "grouped_candidates.jsonl", "grouped_summary.md", "gravity_modal.json", "log.txt"):
            self.assertTrue((self.out / name).exists(), name)
        self.assertFalse((self.out / "grouped_search_failed.json").exists())
        self.assertFalse((self.out / "design.json").exists())                                   # no uniform record is written
        lines = [json.loads(line) for line in (self.out / "grouped_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([line["index"] for line in lines], [c["index"] for c in self.record["search"]["candidates"]])
        summary = (self.out / "grouped_summary.md").read_text(encoding="utf-8")
        for gid in self.record["member_groups"]["designs"]:
            self.assertIn(gid, summary)
        self.assertIn("[grouped] #", (self.out / "log.txt").read_text(encoding="utf-8"))

    def test_the_profile_the_risk_basis_and_the_request_reach_the_record(self):
        self.assertEqual(self.result["configured_basis"], {"risk_category": "III", "importance_factor": 1.25, "joint_model": "rigid_centerline",
                                                           "member_material": "IMKPeakOriented", "analysis_profile_id": V2})
        identity = self.record["request_identity"]
        self.assertEqual(identity["sha256"], self.result["request_sha256"])
        self.assertEqual(identity["policy"]["model_profile"]["id"], V2)
        self.assertEqual((identity["search"]["max_feasibility_trials"], identity["search"]["max_reduction_trials"]), (3, 2))
        self.assertEqual(identity["seed"]["column"], [20.0, 20.0, 5.0])
        self.assertEqual((self.record["seismic"]["risk_category"], self.record["seismic"]["importance_factor"]), ("III", 1.25))
        self.assertEqual(self.record["schema_version"], "rc_smrf_grouped_candidate_v1")
        self.assertNotIn("sections", self.record)

    def test_with_probe_assertions_the_slab_layout_is_selected_and_total_steel_is_known(self):
        self.assertTrue(self.record["stages"]["slab"]["layout_selected"])
        self.assertIsNotNone(self.record["quantities"]["total_steel_in3"])
        self.assertEqual(self.result["numerical_completion"]["slab_refinement_status"], "passed")
        scwb = [c for c in self.record["qualification"]["checks"] if c["id"] == "scwb"]
        self.assertTrue(scwb)
        self.assertEqual({c["status"] for c in scwb}, {"pass"})                                 # evaluated joint by joint, roof included
        decided = {c["decision"] for c in self.record["search"]["candidates"]}
        self.assertTrue(decided <= {"feasible", "not_feasible", "accepted", "rejected", "tradeoff", "unresolved", "skipped"})
        self.assertNotIn("unresolved", decided)                                                 # every quantity was available

    def test_the_statuses_are_separate_and_probe_accepts_nothing(self):
        completion = self.result["numerical_completion"]
        self.assertEqual((completion["design"], completion["gravity_modal"]), ("completed", "completed"))
        self.assertEqual(completion["design_search_stop_reason"], self.record["search"]["stop_reason"])
        self.assertEqual(completion["design_search_budget_exhausted"],
                         self.record["search"]["stop_reason"] in ("reduction_budget_exhausted", "feasibility_budget_exhausted"))
        self.assertEqual(self.result["design_checks"]["accepted_by_checklist"], self.record["qualification"]["accepted"])
        self.assertIn("PROBE", self.result["independent_verification"]["mode"])
        self.assertFalse(self.result["independent_verification"]["verified_by_a_person"])
        self.assertFalse(self.result["independent_verification"]["story_strength_model_verified"])
        self.assertFalse(self.result["production_acceptance"]["accepted"])
        self.assertFalse(self.result["production_acceptance"]["generation_release_ready"])
        self.assertFalse(self.result["implementation_tests"]["run_here"])
        open_items = self.result["not_evaluated_ids"]
        self.assertIn("demands.vertical_strength_regularity", open_items)                       # M1 stays open under PROBE
        self.assertFalse(self.record["qualification"]["accepted"])

    def test_the_stage_built_the_recorded_design(self):
        stage = self.result["stage_results"]["gravity_modal"]
        self.assertEqual(stage["status"], "completed")
        self.assertTrue(all(stage["checks"].values()), stage["checks"])
        self.assertTrue(stage["installed_design_verification"]["consistent"])
        full = load(self.out / "gravity_modal.json")
        self.assertEqual(full["model_audit"]["sections"]["design_mode"], "grouped")
        self.assertEqual(full["installed_topology"]["consistent"], True)

    def test_the_same_request_reuses_the_record_and_another_request_is_refused_in_the_same_directory(self):
        before = (self.out / "grouped_design.json").read_bytes()
        again = run(self.out, stages=("design",))
        result = load(self.out / "result.json")
        self.assertEqual((result["status"], result["created"]), ("designed", False), again.stderr[-1500:])
        self.assertEqual((self.out / "grouped_design.json").read_bytes(), before)
        other = run(self.out, reduction=3, stages=("design",))                                 # another budget: another request
        result = load(self.out / "result.json")
        self.assertEqual(result["status"], "error")
        self.assertIn("different inputs", result["error"])
        self.assertNotEqual(other.returncode, 0)
        self.assertEqual((self.out / "grouped_design.json").read_bytes(), before)
        self.assertFalse(result["production_acceptance"]["accepted"])

    def test_plan_only_prints_the_request_and_writes_nothing(self):
        target = Path(self.tmp.name) / "never_created"
        proc = run(target, "--plan-only")
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        self.assertFalse(target.exists())
        self.assertIn(self.result["request_sha256"][:16], proc.stdout)

    def test_a_search_without_a_feasible_design_keeps_its_candidates_and_says_how_it_stopped(self):
        target = Path(self.tmp.name) / "attempt_budget_1"
        proc = run(target, feasibility=1, reduction=0, stages=("design",))
        result = load(target / "result.json")
        self.assertEqual(result["status"], "no_feasible_design", proc.stderr[-1500:])
        self.assertEqual(proc.returncode, 0)
        self.assertFalse((target / "grouped_design.json").exists())
        failed = load(target / "grouped_search_failed.json")
        self.assertEqual(failed["stop"]["reason"], "feasibility_budget_exhausted")
        self.assertNotIn("final", failed)
        lines = (target / "grouped_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        candidate = json.loads(lines[0])
        self.assertEqual(candidate["decision"], "not_feasible")
        self.assertIn("constraints", candidate["evaluation"])
        self.assertTrue(result["numerical_completion"]["design_search_budget_exhausted"])
        self.assertEqual(result["numerical_completion"]["design"], "completed_without_a_feasible_design")
        self.assertFalse(result["design_checks"]["accepted_by_checklist"])
        self.assertFalse(result["production_acceptance"]["accepted"])
        # a second attempt in the same directory would have to append to that log: it is refused
        again = run(target, feasibility=1, reduction=0, stages=("design",))
        self.assertEqual(load(target / "result.json")["status"], "error")
        self.assertIn("already exists", load(target / "result.json")["error"])
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(len((target / "grouped_candidates.jsonl").read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
