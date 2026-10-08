"""Corner demand must participate in mat selection, not only final audit.

All fixtures are section arithmetic with supplied action envelopes; no
OpenSees floor or frame analysis runs in this suite.
"""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_smrf_slab_reinforcement import inputs, evidence
from Design.SMRF_Slab_Reinforcement import (
    design_slab_reinforcement, evaluate_slab_reinforcement, explain_unsized_layers,
)


def context(**changes):
    return dict(clear_span_x_in=240.0, clear_span_y_in=240.0, beam_width_in=24.0,
                alpha_f_min=5.6, alpha_f_l2_l1_min=5.6, thickness_screen_passed=True,
                column_core_width_in=25.0, two_way_shear_path_assessed=True,
                columns_at_beam_intersections=True, beam_clear_cover_in=1.5,
                beam_hoop_diameter_in=0.5, fc_beam_ksi=5.0) | changes


def asymmetric_evidence(axis="x"):
    demand = evidence(mu=30.0, vu=0.5)
    for row in demand["strips"]:
        if row["axis"] == axis and row["face"] == "bottom":
            row["mu_kip_in_per_ft"] = 80.0
    return demand


class SlabCornerSelectionTests(unittest.TestCase):
    def test_asymmetric_positive_demand_sizes_all_mats_without_rewriting_strips(self):
        # Previously own-strip sizing selected weaker top/perpendicular mats:
        # screen_passed=True despite three failed corner checks for x outer.
        for governing_axis in ("x", "y"):
            for outer_axis in ("x", "y"):
                with self.subTest(governing_axis=governing_axis, outer_axis=outer_axis):
                    demand = asymmetric_evidence(governing_axis)
                    original = copy.deepcopy(demand)
                    result = design_slab_reinforcement(
                        inputs(), demand, {"outer_axis": outer_axis}, context())
                    self.assertTrue(result["strip_screen_passed"])
                    self.assertTrue(result["screen_passed"])
                    self.assertFalse(result["accepted"])
                    self.assertEqual(result["summary"]["counts"]["fail"], 0)
                    corners = [c for c in result["checks"] if c["id"] == "slab_corner_reinforcement"]
                    self.assertEqual(len(corners), 4)
                    self.assertTrue(all(c["status"] == "pass" and c["demand"] == 80.0 for c in corners))
                    self.assertEqual(demand, original)
                    self.assertEqual(result["inputs"]["demand_evidence"], original)
                    for name, layer in result["layout"]["layers"].items():
                        own_mu = 80.0 if name == governing_axis + "_bottom" else 30.0
                        self.assertEqual(layer["demand_envelope"]["mu_kip_in_per_ft"], own_mu)
                        self.assertEqual(layer["corner_moment_demand_kip_in_per_ft"], 80.0)
                        self.assertEqual(layer["selection_flexural_demand_kip_in_per_ft"], 80.0)
                        self.assertGreaterEqual(layer["flexure"]["phi_mn_kip_in_per_ft"], 80.0)
                    audit = evaluate_slab_reinforcement(json.loads(json.dumps(result)))
                    self.assertEqual(audit[0]["status"], "pass")

    def test_corner_requirement_does_not_reduce_larger_own_strip_demand(self):
        demand = asymmetric_evidence()
        for row in demand["strips"]:
            if row["axis"] == "y" and row["face"] == "top":
                row["mu_kip_in_per_ft"] = 180.0
        result = design_slab_reinforcement(inputs(), demand, context=context())
        layer = result["layout"]["layers"]["y_top"]
        self.assertTrue(result["screen_passed"])
        self.assertEqual(layer["demand_envelope"]["mu_kip_in_per_ft"], 180.0)
        self.assertEqual(layer["corner_moment_demand_kip_in_per_ft"], 80.0)
        self.assertEqual(layer["selection_flexural_demand_kip_in_per_ft"], 180.0)
        self.assertGreaterEqual(layer["flexure"]["phi_mn_kip_in_per_ft"], 180.0)

    def test_corner_exhaustion_has_no_fallback_and_explains_actual_demand(self):
        result = design_slab_reinforcement(
            inputs(), asymmetric_evidence(), {"bar_sizes": [4], "spacing_options_in": [11]}, context())
        self.assertIsNone(result["layout"])
        self.assertFalse(result["screen_passed"])
        explanation = explain_unsized_layers(result)
        self.assertEqual(set(explanation), {"y_top", "y_bottom"})
        for layer in explanation.values():
            self.assertEqual(layer["demand_envelope"]["mu_kip_in_per_ft"], 30.0)
            self.assertEqual(layer["corner_moment_demand_kip_in_per_ft"], 80.0)
            self.assertEqual([c["id"] for c in layer["failed_checks"]], ["slab_corner_reinforcement"])
            self.assertEqual(layer["failed_checks"][0]["demand"], 80.0)

    def test_failed_completion_cannot_claim_screen_pass_but_keeps_its_evidence(self):
        result = design_slab_reinforcement(
            inputs(), asymmetric_evidence(), context=context(thickness_screen_passed=False))
        self.assertIsNotNone(result["layout"])
        self.assertTrue(result["strip_screen_passed"])
        self.assertFalse(result["screen_passed"])
        self.assertEqual(result["summary"]["failed"], ["slab_deflection_control"])
        self.assertEqual(evaluate_slab_reinforcement(result)[0]["status"], "pass")

    def test_unevaluated_scope_stays_open_without_becoming_an_evaluated_failure(self):
        for supplied_context in (None, context(two_way_shear_path_assessed=False)):
            with self.subTest(context=supplied_context):
                result = design_slab_reinforcement(inputs(), asymmetric_evidence(), context=supplied_context)
                self.assertTrue(result["strip_screen_passed"])
                self.assertTrue(result["screen_passed"])
                self.assertFalse(result["accepted"])
                self.assertFalse(result["summary"]["accepted"])
                self.assertGreater(result["summary"]["counts"]["not_evaluated"], 0)
                self.assertEqual(result["summary"]["counts"]["fail"], 0)


if __name__ == "__main__":
    unittest.main()
