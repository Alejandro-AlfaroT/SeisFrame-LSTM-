"""Reference specimen runner (Reference_Specimens/park_ruitong_1988/pr1_unit1.py): a failed run keeps its
history (Codex Unit 1 review, 2026-09-28, commit blocker 2), and mirrored histories are compared by phase.

The injected failures test preservation only; they are not physical-analysis failures.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

SPEC = RC_DIR / "Reference_Specimens" / "park_ruitong_1988" / "pr1_unit1.py"
spec = importlib.util.spec_from_file_location("pr1_unit1_under_test", SPEC)
pr1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr1)


class FailurePreservation(unittest.TestCase):
    def tearDown(self):
        pr1.ops.wipe()

    def test_failure_after_gravity_exports_completed_states_and_the_failure_record(self):
        def fail(rec, **kwargs):
            raise RuntimeError("review injected failure after gravity record")
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            with mock.patch.object(pr1, "run_example_protocol", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    pr1.run("injected", out)
            names = sorted(p.name for p in out.iterdir())
            self.assertEqual(names, ["injected_FAILED_history.csv", "injected_FAILED_history.npz", "injected_FAILED_result.json"])
            result = json.loads((out / "injected_FAILED_result.json").read_text(encoding="utf-8"))
        meta = result["meta"]
        self.assertEqual(meta["status"], "failed")
        self.assertIsNone(result["evaluation"])
        failure = meta["failure"]
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertIn("injected failure", failure["message"])
        self.assertEqual(failure["completed_rows"], 1)
        self.assertEqual(failure["last_completed_phase"], "gravity")
        self.assertEqual(failure["phase"]["phase"], "gravity")
        self.assertEqual(failure["solver_requested"], pr1.SOLVER)
        self.assertIn("Traceback", failure["traceback"])

    def test_failure_inside_the_displacement_cycles_records_phase_target_and_attempts(self):
        calls = {"n": 0}
        real = pr1._analyze_step

        def flaky(rec, integrator_args, allow_loose=True):
            calls["n"] += 1
            if calls["n"] > 1:                      # the first 0.05 mm step converges, then every attempt "fails"
                if allow_loose:
                    rec.failures += 1
                rec.last_attempt = None
                rec.failed_attempts = [{"algorithm": "Newton", "tolerance": pr1.SOLVER["tolerance"], "iterations": 150, "converged": False}]
                return False
            return real(rec, integrator_args, allow_loose=allow_loose)
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            with mock.patch.object(pr1, "_analyze_step", side_effect=flaky):
                with self.assertRaises(pr1.AnalysisFailure) as ctx:
                    pr1.run("cycles", out, protocol="example", step=0.05)
            record = ctx.exception.record
            self.assertEqual(record["phase"], "ex0")
            self.assertEqual(record["target"], 0.1)
            self.assertEqual(record["substep"], 1)
            result = json.loads((out / "cycles_FAILED_result.json").read_text(encoding="utf-8"))
            self.assertFalse((out / "cycles_result.json").exists())
        failure = result["meta"]["failure"]
        self.assertEqual(failure["exception_type"], "AnalysisFailure")
        self.assertEqual(failure["phase"], {"phase": "ex0", "target": 0.1, "control": "displacement"})
        self.assertEqual(failure["completed_rows"], 1 + 1)          # gravity + the one accepted 0.05 mm step
        self.assertEqual(failure["attempts_at_failing_step"][0]["algorithm"], "Newton")
        self.assertFalse(failure["attempts_at_failing_step"][0]["converged"])
        self.assertEqual(failure["recoveries_so_far"]["substep_x10"], 1)
        self.assertEqual(failure["solver_requested"]["equilibrium_limit"], 1e-6)
        self.assertIsNotNone(failure["u_tip_mm_at_failure"])

    def test_accepted_substeps_are_recorded_with_their_solver_settings(self):
        calls = {"n": 0}
        real = pr1._analyze_step

        def one_substep_round(rec, integrator_args, allow_loose=True):
            calls["n"] += 1
            if calls["n"] == 2:                     # the second full step fails once, then its ten sub-steps converge
                rec.last_attempt = None
                rec.failed_attempts = []
                return False
            return real(rec, integrator_args, allow_loose=allow_loose)
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            with mock.patch.object(pr1, "_analyze_step", side_effect=one_substep_round):
                rows, meta, ev = pr1.run("substeps", out, protocol="example", step=0.05)
        subs = [r for r in rows if r["substep"]]
        self.assertEqual(len(subs), 10)
        self.assertEqual([r["step_complete"] for r in subs], [0] * 9 + [1])
        self.assertTrue(all(r["solver_code"] == 0 and r["solver_tol"] == pr1.SOLVER["tolerance"] for r in subs))
        self.assertEqual(ev["substep_rows"], 10)
        self.assertEqual(meta["solver_recoveries_by_kind"]["substep_x10"], 1)
        self.assertEqual(meta["status"], "completed")
        # the sub-step rows advance the displacement by a tenth of the full step each
        u = [r["u_tip"] for r in subs]
        self.assertAlmostEqual(u[-1] - u[0], 0.9 * 0.05, places=9)


class PhaseMatchedComparison(unittest.TestCase):
    def rows(self, phases, values, complete=None):
        complete = complete or [1] * len(phases)
        return [{"phase": p, "u_tip": v, "V": 10.0 * v, "step_complete": c, "substep": 0, "solver_code": 0, "solver_tol": 1e-8}
                for p, v, c in zip(phases, values, complete)]

    def test_identical_phase_sequences_compare_every_row(self):
        a = self.rows(["gravity", "ex0", "ex0", "ex1"], [0.0, 1.0, 2.0, 1.0])
        b = self.rows(["gravity", "ex0", "ex0", "ex1"], [0.0, 1.0, 2.0, 1.0])
        out = pr1.compare(a, b, ["u_tip", "V", "solver_code"])
        self.assertTrue(out["matching_phases"])
        self.assertEqual(out["rows_compared"], 4)
        self.assertEqual(out["max"], 0.0)
        self.assertNotIn("solver_code", out["relative"])
        self.assertTrue(out["global_budget"]["passes"])

    def test_phase_mismatch_is_reported_not_truncated(self):
        a = self.rows(["gravity", "ex0", "ex0", "ex1", "ex1"], [0.0, 1.0, 2.0, 1.0, 0.0])
        b = self.rows(["gravity", "ex0", "ex1", "ex1"], [0.0, 1.0, 1.0, 0.0])
        out = pr1.compare(a, b, ["u_tip"])
        self.assertFalse(out["matching_phases"])
        self.assertEqual(out["first_phase_mismatch"], {"index": 2, "original": "ex0", "reflected": "ex1"})
        self.assertEqual(out["rows_compared"], 2)

    def test_substep_rows_are_excluded_from_the_row_match(self):
        a = self.rows(["gravity", "ex0", "ex0", "ex0"], [0.0, 0.5, 0.75, 1.0], complete=[1, 0, 0, 1])
        b = self.rows(["gravity", "ex0"], [0.0, 1.0])
        out = pr1.compare(a, b, ["u_tip"])
        self.assertTrue(out["matching_phases"])
        self.assertEqual(out["rows_compared"], 2)
        self.assertEqual(out["max"], 0.0)


if __name__ == "__main__":
    unittest.main()
