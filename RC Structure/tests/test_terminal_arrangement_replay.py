"""The four recorded bar-row rejections of the 15-minute search run (o0005, o0009, o0011, o0014), replayed on the live
code without a solve (2026-11-01).

Opt-in and slow (minutes per case): set SEISFRAME_TERMINAL_REPLAY=1. The captured inputs (each case's state, solved
strength combinations, floor transfers and compatibility run) and the replay script are in the research evidence folder
research/ga_blocker_closure_impl_20261101 (inputs/ and cage_replay/replay_cases.py), which is not in the tracked tree;
without them the test is skipped.

Each case must end either with a complete arrangement whose rows fit and a capacity design of that final state, the
outer passes ending settled, on a proven cycle or on the pass budget, or with the structured incomplete outcome; the
replay refuses every OpenSees analyze and eigen call. Where the run's geometry audit (research/ga_blocker_closure_20261101
/GEOMETRY_AUDIT.json) is present, a closure must have received the recorded failed state, with the recorded rows among
its invalid ones, and returned rows that fit.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "research" / "ga_blocker_closure_impl_20261101"
SCRIPT = EVIDENCE / "cage_replay" / "replay_cases.py"
AUDIT = REPO / "research" / "ga_blocker_closure_20261101" / "GEOMETRY_AUDIT.json"
CASES = ("o0005", "o0009", "o0011", "o0014")
ENABLED = os.environ.get("SEISFRAME_TERMINAL_REPLAY") == "1" and SCRIPT.exists() and (EVIDENCE / "inputs").is_dir()


@unittest.skipUnless(ENABLED, "opt-in slow replay: set SEISFRAME_TERMINAL_REPLAY=1 with the research evidence folder present")
class RecordedRejections(unittest.TestCase):
    def test_each_recorded_rejection_ends_with_rows_that_fit_or_a_structured_outcome(self):
        recorded = ({r["id"]: r["saved_failure"] for r in json.loads(AUDIT.read_text(encoding="utf-8"))["results"]}
                    if AUDIT.exists() else {})
        for oid in CASES:
            with self.subTest(case=oid), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / f"{oid}.json"
                done = subprocess.run([sys.executable, "-X", "utf8", "-B", str(SCRIPT), "live", str(out), oid],
                                      timeout=3600, capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                report = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(report["analysis_calls"], [])
                case = report["cases"][0]
                if case["outcome"] == "completed":
                    self.assertTrue(case["final_row_audit"]["valid"])
                    self.assertTrue(case["capacity_is_of_final_state"])
                    self.assertIn(case["stop"], ("settled", "cycle", "pass_budget"))
                    if oid in recorded:
                        rows = {(i["group"], i["face"], tuple(i["rows"]["per_layer"]), tuple(i["rows"]["offsets_in"]))
                                for i in recorded[oid]["invalid"]}
                        received = [c for c in case["closure_trace"] if c["received_identity"] == recorded[oid]["identity"]]
                        self.assertTrue(received, "no closure received the recorded failed state")
                        self.assertTrue(rows <= {(i["group"], i["face"], tuple(i["per_layer"]), tuple(i["offsets_in"]))
                                                 for i in received[0]["received_invalid_rows"]})
                        self.assertEqual(received[0]["outcome"], "returned")
                        self.assertTrue(received[0]["returned_rows_fit"])
                else:
                    self.assertEqual(case["type"], "CageArrangementError")
                    self.assertTrue(case["evidence"]["classification"])
                    self.assertIn("not proof", case["evidence"]["reason"])


if __name__ == "__main__":
    unittest.main()
