"""Durable diagnostics for interrupted verification workers; no structural solves."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design import Verify_Designs as verify
from Design import Design_Driver as driver


class ProgressFiles(unittest.TestCase):
    def test_malformed_snapshot_is_not_attached(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "progress.json")
            for value in ('[1,2]', 'null', '{truncated'):
                path.write_text(value, encoding="utf-8")
                self.assertIsNone(verify._saved_progress(folder, "attempt"))

    def test_nonfinite_trial_metric_is_explicit_and_does_not_abort_search(self):
        with tempfile.TemporaryDirectory() as folder:
            log = verify._ProgressLog(folder, "case_0036", "attempt")
            log({"event": "iteration_completed", "phase": "candidate.checks", "iteration": 1,
                 "beam_dcr": float("inf"), "other": float("nan"), "candidate_screen_passed": False})
            state = verify._saved_progress(folder, "attempt")
            self.assertEqual(state["last_completed_iteration"]["beam_dcr"], {"nonfinite": "Infinity"})
            self.assertEqual(state["last_completed_iteration"]["other"], {"nonfinite": "NaN"})
            self.assertFalse(state["last_completed_iteration"]["candidate_screen_passed"])

    def test_partial_phase_keeps_completed_iteration_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            log = verify._ProgressLog(folder, "case_0036", "attempt-a")
            log({"event": "candidate_started", "phase": "candidate.setup", "iteration": 1})
            completed = {"event": "iteration_completed", "phase": "candidate.checks", "iteration": 1,
                         "candidate_screen_passed": False, "beam_dcr": 0.9}
            log(completed)
            log({"event": "phase_started", "phase": "candidate.steel", "iteration": 2})
            snapshot = verify._saved_progress(folder, "attempt-a")
            self.assertEqual(snapshot["status"], "running")
            self.assertEqual(snapshot["active_phase"], "candidate.steel")
            self.assertEqual(snapshot["last_completed_iteration"]["iteration"], 1)
            self.assertFalse(snapshot["last_completed_iteration"]["candidate_screen_passed"])
            self.assertEqual(len(Path(folder, "progress.jsonl").read_text().splitlines()), 3)
            self.assertFalse(list(Path(folder).glob("*.tmp")))
            self.assertNotIn("design.json", [p.name for p in Path(folder).iterdir()])

    def test_retry_preserves_old_journal_but_stale_snapshot_is_never_attached(self):
        with tempfile.TemporaryDirectory() as folder:
            old = verify._ProgressLog(folder, "case_0036", "old")
            old({"event": "iteration_completed", "phase": "candidate.checks", "iteration": 16})
            self.assertIsNone(verify._saved_progress(folder, "new"))
            new = verify._ProgressLog(folder, "case_0036", "new")
            new({"event": "phase_started", "phase": "worker.configure"})
            state = verify._saved_progress(folder, "new")
            self.assertIsNone(state["last_completed_iteration"])
            self.assertEqual([json.loads(line)["attempt_id"] for line in
                              Path(folder, "progress.jsonl").read_text().splitlines()], ["old", "new"])
            self.assertIsNone(verify._saved_progress(folder, "old"))

    def test_failed_worker_keeps_current_phase_and_completed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            log = verify._ProgressLog(folder, "case_0036", "attempt")
            log({"event": "phase_completed", "phase": "candidate.slab"})
            log({"event": "phase_started", "phase": "candidate.steel"})
            log({"event": "worker_failed", "phase": "worker", "error": "test interruption"})
            state = verify._saved_progress(folder, "attempt")
            self.assertEqual(state["status"], "error")
            self.assertEqual(state["active_phase"], "candidate.steel")
            self.assertEqual(state["last_completed_phase"]["phase"], "candidate.slab")


class ProcessOutput(unittest.TestCase):
    def test_native_output_survives_hard_exit_and_existing_logs_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "stderr.txt").write_text("earlier attempt\n", encoding="utf-8")
            script = "import os; os.write(1,b'native stdout\\n'); os.write(2,b'native failure\\n'); os._exit(19)"
            code, timed_out = verify._run_logged_process([sys.executable, "-u", "-c", script], folder, dict(os.environ))
            self.assertEqual(code, 19)
            self.assertFalse(timed_out)
            self.assertIn("native stdout", Path(folder, "worker_stdout.txt").read_text())
            text = Path(folder, "stderr.txt").read_text()
            self.assertIn("earlier attempt", text)
            self.assertIn("native failure", text)

    def test_timeout_retains_already_written_process_output(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(verify, "WORKER_TIMEOUT_S", 1.0):
            script = "import os,time; os.write(1,b'before timeout\\n'); os.write(2,b'solver phase\\n'); time.sleep(30)"
            code, timed_out = verify._run_logged_process([sys.executable, "-u", "-c", script], folder, dict(os.environ))
            self.assertIsNone(code)
            self.assertTrue(timed_out)
            self.assertIn("before timeout", Path(folder, "worker_stdout.txt").read_text())
            self.assertIn("solver phase", Path(folder, "stderr.txt").read_text())
            self.assertIn("worker killed", Path(folder, "stderr.txt").read_text())

    def test_launcher_timeout_result_uses_only_its_attempt_progress(self):
        case = {"case_id": "case_0036"}
        with tempfile.TemporaryDirectory() as folder:
            def fake_run(command, out_dir, env, **kwargs):
                attempt = command[command.index("--attempt-id") + 1]
                progress = verify._ProgressLog(out_dir, case["case_id"], attempt)
                progress({"event": "iteration_completed", "phase": "candidate.checks", "iteration": 4,
                          "candidate_screen_passed": False})
                progress({"event": "phase_started", "phase": "candidate.steel", "iteration": 5})
                return None, True
            with patch.object(verify, "_run_logged_process", side_effect=fake_run):
                result, reused = verify._launch(sys.executable, case, folder, False)
            self.assertFalse(reused)
            self.assertEqual(result["status"], "error")
            self.assertFalse(result["production_acceptance"]["accepted"])
            self.assertFalse(result["design_checks"]["accepted_by_checklist"])
            self.assertEqual(result["progress"]["last_completed_iteration"]["iteration"], 4)
            self.assertEqual(result["progress"]["active_phase"], "candidate.steel")
            old = verify._ProgressLog(folder, case["case_id"], "stale")
            old({"event": "iteration_completed", "phase": "candidate.checks", "iteration": 999})
            with patch.object(verify, "_run_logged_process", return_value=(7, False)):
                retry, _ = verify._launch(sys.executable, case, folder, False)
            self.assertIsNone(retry["progress"])


class DriverObservations(unittest.TestCase):
    def test_observer_cannot_mutate_search_records(self):
        source = {"nested": [1, 2]}
        def observer(event):
            event["details"]["nested"].append(3)
        driver._notify_progress(observer, "iteration_completed", "candidate.checks", details=source)
        self.assertEqual(source, {"nested": [1, 2]})
        driver._notify_progress(None, "phase_started", "unused", details=source)

    def test_observer_failure_is_only_a_warning(self):
        def broken(event):
            raise RuntimeError("observer failure")
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            driver._notify_progress(broken, "iteration_completed", "candidate.checks", iteration=1)
        self.assertIn("observer failure", errors.getvalue())

    def test_bar_layout_phases_are_observational_and_leave_return_values_unchanged(self):
        cfg = SimpleNamespace(demands=SimpleNamespace(accidental_torsion_ratio=0))
        slab = {"thickness_in": 7.0}
        worst = {"beam": .7, "column": .6}
        joint, capacity = {"all_pass": True}, {"accepted": False}
        actions, elf, events = [{"id": "test"}], {"test": True}, []
        before = copy.deepcopy((slab, worst, joint, capacity, actions, elf))
        with patch.object(driver, "_sync_cfg_to_sp"), patch.object(driver, "_model_period", return_value=1.2), \
             patch.object(driver, "_steel_pass", return_value=(worst, elf, actions, joint)), \
             patch.object(driver, "_capacity_design", return_value=capacity), \
             patch.object(driver, "_repair_bar_stacking", return_value=(worst, joint, capacity, {"status": "not_needed"})):
            result = driver._analyze_bar_layout(cfg, slab, {}, 1, progress=events.append)
        self.assertEqual(result, (slab, 1.2, None, worst, elf, actions, joint, capacity))
        self.assertEqual(before, (slab, worst, joint, capacity, actions, elf))
        completed = [e["phase"] for e in events if e["event"] == "phase_completed"]
        self.assertEqual(completed, ["candidate.modal", "candidate.torsion", "candidate.steel", "candidate.capacity", "candidate.bar_stacking"])

    def test_load_wrapper_passes_observer_and_reports_completed_atomic_write(self):
        identity = {"sha256": "test"}
        events = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "design.json")
            def observe(event):
                events.append(event)
                if event["phase"] == "design.write" and event["event"] == "phase_completed":
                    self.assertTrue(path.exists())
            with patch.object(driver, "design_request_identity", return_value=identity), \
                 patch.object(driver, "design_structure", return_value={"schema_version": driver.DESIGN_SCHEMA_VERSION}) as search:
                record, created = driver.load_or_create_design(path, progress=observe)
            self.assertTrue(created)
            self.assertIs(search.call_args.kwargs["progress"], observe)
            self.assertEqual(json.loads(path.read_text())["request_identity"], identity)
            self.assertEqual([e["event"] for e in events], ["phase_started", "phase_completed"])

    def test_worker_failure_preserves_live_python_log_and_progress(self):
        case = {"case_id": "case_0036"}
        with tempfile.TemporaryDirectory() as folder:
            def interrupted_load(path, **kwargs):
                print("completed candidate before interruption")
                # This is visible before _run_worker reaches its finally block.
                self.assertIn("completed candidate", Path(folder, "log.txt").read_text())
                kwargs["progress"]({"event": "iteration_completed", "phase": "candidate.checks", "iteration": 3})
                kwargs["progress"]({"event": "phase_started", "phase": "candidate.steel", "iteration": 4})
                raise RuntimeError("injected interruption")
            with patch.object(verify, "configure_case", return_value=({}, None)), \
                 patch.object(driver, "load_or_create_design", side_effect=interrupted_load):
                result = verify._run_worker(case, folder, False, attempt_id="worker-test")
            self.assertEqual(result["status"], "error")
            self.assertIn("injected interruption", result["error"])
            self.assertEqual(result["progress"]["last_completed_iteration"]["iteration"], 3)
            self.assertEqual(result["progress"]["active_phase"], "candidate.steel")
            self.assertFalse(result["production_acceptance"]["accepted"])
            self.assertTrue(Path(folder, "result.json").exists())

    def test_progress_write_failure_cannot_discard_a_completed_design(self):
        case = {"case_id": "case_0036"}
        record = {"qualification": {"accepted": False, "checks": [], "counts": {"pass": 0, "fail": 0, "not_evaluated": 1}},
                  "sections": {}, "reinforcement": {
                      "col_bar_size": 8, "col_top_bars": 4, "col_side_bars": 2,
                      "beam_bar_size": 8, "beam_top_bars": 4, "beam_bot_bars": 4,
                      "col_stirrup_bar_size": 4, "col_stirrup_legs": 4, "col_stirrup_spacing_in": 3,
                      "beam_stirrup_bar_size": 4, "beam_stirrup_legs": 4, "beam_stirrup_spacing_in": 4},
                  "dcr": {"governed_by": "test", "beam": .5, "column": .4}, "iterations": 1}
        with tempfile.TemporaryDirectory() as folder:
            def completed_load(path, **kwargs):
                Path(path).write_text(json.dumps(record), encoding="utf-8")
                kwargs["progress"]({"event": "iteration_completed", "phase": "candidate.checks", "iteration": 1})
                return copy.deepcopy(record), True
            errors = io.StringIO()
            with patch.object(verify, "configure_case", return_value=({}, None)), \
                 patch.object(driver, "load_or_create_design", side_effect=completed_load), \
                 patch.object(verify, "design_overview", return_value={}), \
                 patch.object(verify, "status_fields", return_value={"production_acceptance": {"accepted": False}}), \
                 patch.object(verify._ProgressLog, "_write_event", side_effect=OSError("injected telemetry failure")), \
                 contextlib.redirect_stderr(errors):
                result = verify._run_worker(case, folder, False, attempt_id="completed-test")
            self.assertEqual(result["status"], "designed")
            self.assertTrue(Path(folder, "design.json").exists())
            self.assertEqual(json.loads(Path(folder, "result.json").read_text())["status"], "designed")
            self.assertNotIn("error", result)
            self.assertGreater(result["progress_warnings"]["count"], 0)
            self.assertIn("injected telemetry failure", errors.getvalue())
            self.assertFalse(result["production_acceptance"]["accepted"])


if __name__ == "__main__":
    unittest.main()
