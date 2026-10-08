"""Regression guards for the October 6 failed verification cases; no structural solves."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Slab_Refinement import refinement_diagnostics
from test_slab_thickness_search import tiny_frame, verify, driver, PROBE_DATE


def failed_fine_evidence():
    row = dict(panel_id="panel_x1_y1", axis="y", face="bottom", metric="vu_kip_per_ft",
               coarse=1., fine=1.050028428625, relative_change=.050028428625,
               tolerance=.05, within_tolerance=False)
    levels = []
    for index, cells in enumerate((12, 24, 48, 60)):
        level = dict(index=index, requested_mesh={"subdivisions_per_bay": cells},
                     status="completed", attempted_cases=[])
        if cells == 60:
            level.update(status="failed", error="RuntimeError: numeric solve failed",
                         attempted_cases=[dict(loadcase={"id": "1.2D+1.6L"}, status="analysis_failed",
                                               analysis_return_code=-3)])
        else:
            level["actions"] = dict(analysis_model_sha256=f"mesh{cells}", strips=[
                dict(panel_id="panel_x1_y1", axis="y", face="bottom",
                     shear_location={"loadcase": "1.2D+1.6L", "x_in": cells, "y_in": 100.},
                     shear_basis="synthetic_witness")])
        levels.append(level)
    return {"refinement": dict(status="analysis_failed", levels=levels, comparisons=[
        dict(coarse_analysis_sha256="mesh24", fine_analysis_sha256="mesh48", comparisons=[row])])}


class FailureDiagnostics(unittest.TestCase):
    def test_failed_fine_mesh_is_not_mislabelled_as_compared(self):
        evidence = failed_fine_evidence()
        before = copy.deepcopy(evidence)
        r = refinement_diagnostics(evidence)
        self.assertEqual(evidence, before)
        self.assertEqual(r["status"], "analysis_failed")
        self.assertIn("[24, 48]", r["detail"])
        self.assertNotIn("[48, 60]", r["detail"])
        self.assertIn("0.0500284286", r["detail"])
        self.assertIn("numeric solve failed", r["detail"])
        self.assertEqual(r["failed_levels"][0]["attempted_cases"][0]["analysis_return_code"], -3)
        last = r["last_completed_comparison"]
        self.assertEqual(last["coarse_level"]["index"], 1)
        self.assertEqual(last["fine_level"]["index"], 2)
        self.assertEqual(last["comparisons"][0]["fine_witness"]["shear_location"]["x_in"], 48)
        self.assertNotIn("actions", r["levels"][0])
        json.dumps(r, allow_nan=False)

    def test_no_identity_match_is_explicit_and_zero_to_nonzero_is_not_reported_zero(self):
        evidence = failed_fine_evidence()
        comparison = evidence["refinement"]["comparisons"][0]
        comparison["coarse_analysis_sha256"] = "not-saved"
        comparison["comparisons"][0]["relative_change"] = None
        r = refinement_diagnostics(evidence)
        self.assertIsNone(r["last_completed_comparison"]["coarse_level"])
        self.assertIn("identities unavailable", r["detail"])
        self.assertIn("undefined (zero-to-nonzero)", r["detail"])

    def test_driver_serializes_failed_solve_and_does_not_step_thickness(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            slab = driver._select_slab(cfg)
            record = {"layout": None, "checks": [dict(id="actions", status="not_evaluated", details={})]}
            with mock.patch("Design.SMRF_Slab_Refinement.build_refined_slab_action_evidence",
                            return_value=failed_fine_evidence()), \
                 mock.patch("Design.SMRF_Slab_Reinforcement.design_slab_reinforcement", return_value=record), \
                 mock.patch.object(driver, "_slab_strength_inputs_from_state", return_value={}), \
                 mock.patch.object(driver, "_slab_completion_context", return_value={}), \
                 mock.patch.object(driver, "_thicker_slab") as thicker:
                with self.assertRaises(driver.SlabDesignError) as caught:
                    driver._slab_thickness_search(cfg, slab)
            self.assertNotIsInstance(caught.exception, driver.SlabLayoutError)
            thicker.assert_not_called()
            r = caught.exception.evidence["refinement"]
            self.assertEqual(r["failed_levels"][0]["requested_mesh"]["subdivisions_per_bay"], 60)
            self.assertIn("[24, 48]", str(caught.exception))

    def test_sized_but_failed_completion_enters_layout_repair(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            slab = driver._select_slab(cfg)
            record = {"layout": {"bar_size": 4}, "screen_passed": False, "checks": [
                dict(id="slab_corner", status="fail", details={"reason": "corner flexure"})]}
            with mock.patch("Design.SMRF_Slab_Refinement.build_refined_slab_action_evidence",
                            return_value={"refinement": {"status": "passed"}}), \
                 mock.patch("Design.SMRF_Slab_Reinforcement.design_slab_reinforcement", return_value=record), \
                 mock.patch.object(driver, "_slab_strength_inputs_from_state", return_value={}), \
                 mock.patch.object(driver, "_slab_completion_context", return_value={}):
                with self.assertRaises(driver.SlabLayoutError) as caught:
                    driver._update_slab_reinforcement(cfg, slab)
            self.assertEqual(caught.exception.evidence["layout"], record["layout"])
            self.assertIn("corner flexure", str(caught.exception))

    def test_candidate_screen_rejects_evaluated_failure_but_keeps_open_scope_distinct(self):
        record = {"screen_passed": True, "checks": [dict(id="scope", status="not_evaluated")]}
        self.assertTrue(driver._slab_candidate_screen(record)["accepted"])
        record["checks"].append(dict(id="slab_corner", status="fail"))
        screen = driver._slab_candidate_screen(record)
        self.assertFalse(screen["accepted"])
        self.assertEqual(screen["scope_open_checks"], ["scope"])
        self.assertEqual(screen["failed_checks"][0]["id"], "slab_corner")

    def test_unasserted_slab_remains_a_provisional_screen_with_no_selected_steel(self):
        record = {"layout": None, "screen_passed": False,
                  "checks": [dict(id="slab_verified_strip_action_inputs", status="not_evaluated")]}
        screen = driver._slab_candidate_screen(record)
        self.assertTrue(screen["accepted"])
        self.assertFalse(screen["reinforcement_selected"])
        record["checks"].append(dict(id="slab_strip_reinforcement_ladder", status="fail"))
        self.assertFalse(driver._slab_candidate_screen(record)["accepted"])


if __name__ == "__main__":
    unittest.main()
