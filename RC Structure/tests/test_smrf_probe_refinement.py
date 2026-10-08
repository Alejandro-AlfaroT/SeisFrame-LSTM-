"""The named refinement recipe and the PROBE workflows that depend on it.

Software checks: the recipe reproduces the verified benchmark meshes,
resolves per geometry and beam face, stops explicitly at the shell budget,
both PROBE factories carry it, the design driver names a missing plan, and
a saved record cannot claim a plan its recorded inputs do not resolve to.
Passing here is not a validated mesh recipe for every geometry.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Design.Config import DesignConfig, FloorAnalysisConfig, PROBE_SLAB_REFINEMENT, SlabActionAssertions
from Design.SMRF_Floor_Mesh import graded_face_levels, resolve_recipe_plan, floor_mesh, MAX_EXPLICIT_SHELLS
from Design.SMRF_Slab_Actions import evaluate_slab_actions
from Design.SMRF_Slab_Refinement import build_refined_slab_action_evidence, refinement_verified, _plan
from test_smrf_slab_refinement import SLAB, GEOMETRY, SECTIONS, INPUTS, manufactured, signed, policy as explicit_policy

BENCHMARK_24 = [0., 2., 4., 7., 10., 15., 22., 32., 45., 60., 80., 100., 120., 140., 160., 180., 195., 208.,
                218., 225., 230., 233., 236., 238., 240.]
BENCHMARK_48 = [0., 1., 2., 3., 4., 5.5, 7., 8.5, 10., 12.5, 15., 18.5, 22., 27., 32., 38.5, 45., 52.5, 60., 70.,
                80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 187.5, 195., 201.5, 208., 213.,
                218., 221.5, 225., 227.5, 230., 231.5, 233., 234.5, 236., 237., 238., 239., 240.]
BENCHMARK_60 = [0., 1., 2., 3., 4., 4.75, 5.5, 6.25, 7., 7.75, 8.5, 9.25, 10., 11.25, 12.5, 13.75, 15., 18.5, 22.,
                27., 32., 38.5, 45., 52.5, 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170.,
                180., 187.5, 195., 201.5, 208., 213., 218., 221.5, 225., 226.25, 227.5, 228.75, 230., 230.75,
                231.5, 232.25, 233., 233.75, 234.5, 235.25, 236., 237., 238., 239., 240.]


def recipe_policy(**overrides):
    # These fixtures exercise the original base recipe, including two-level
    # plans. The optional four-base-level extension has separate tests.
    result = dict(PROBE_SLAB_REFINEMENT, **overrides)
    result.pop("local_extension", None)
    return result


class GradedFaceRecipeTests(unittest.TestCase):
    def assertOffsets(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=9)

    def test_recipe_reproduces_the_verified_benchmark_meshes_and_a_nested_coarse_level(self):
        coarse, base, fine, finest = graded_face_levels(240., 14., 4)
        self.assertOffsets(base, BENCHMARK_24)
        self.assertOffsets(fine, BENCHMARK_48)
        self.assertOffsets(finest, BENCHMARK_60)
        self.assertOffsets(coarse, [0., 7., 15., 32., 60., 100., 120., 140., 180., 208., 225., 233., 240.])
        for a, b in zip((coarse, base, fine), (base, fine, finest)):
            self.assertTrue(set(a) <= set(b))

    def test_recipe_places_the_beam_face_and_scales_with_bay_and_beam(self):
        for length, width in ((120., 12.), (180., 16.), (150., 10.)):
            with self.subTest(length=length, width=width):
                levels = graded_face_levels(length, width, 4)
                for offsets in levels:
                    self.assertEqual(offsets[0], 0.)
                    self.assertEqual(offsets[-1], length)
                    self.assertIn(width / 2., offsets)
                    self.assertIn(length - width / 2., offsets)
                    self.assertIn(length / 2., offsets)
                    self.assertTrue(all(b > a for a, b in zip(offsets, offsets[1:])))
                self.assertEqual([len(o) - 1 for o in levels], [12, 24, 48, 60])
        with self.assertRaises(ValueError):
            graded_face_levels(120., 70., 4)
        with self.assertRaises(ValueError):
            graded_face_levels(240., 14., 1)

    def test_rectangular_floor_resolves_each_direction_from_its_own_bay(self):
        r = resolve_recipe_plan({"num_bay_x": 2, "num_bay_y": 3, "bay_x_in": 240., "bay_y_in": 180.},
                                {"b_beam_in": 14.}, recipe_policy())
        self.assertEqual(r["status"], "resolved")
        self.assertEqual([m["shell_count"] for m in r["resolved_levels"]], [6 * 144, 6 * 576, 6 * 2304, 6 * 3600])
        for mesh in r["meshes"]:
            self.assertEqual(mesh["x_offsets_in"][-1], 240.)
            self.assertEqual(mesh["y_offsets_in"][-1], 180.)
            self.assertIn(7., mesh["y_offsets_in"])
            self.assertIn(173., mesh["y_offsets_in"])
        self.assertEqual(r["inputs"]["beam_width_in"], 14.)

    def test_budget_keeps_a_prefix_of_levels_and_reports_the_dropped_ones(self):
        cases = {(2, 6, 240.): [12, 24, 48, 60], (3, 3, 120.): [12, 24, 48, 60], (6, 6, 180.): [12, 24],
                 (4, 6, 150.): [12, 24], (3, 5, 150.): [12, 24, 48]}
        for (nx, ny, bay), expected in cases.items():
            with self.subTest(nx=nx, ny=ny):
                r = resolve_recipe_plan({"num_bay_x": nx, "num_bay_y": ny, "bay_x_in": bay, "bay_y_in": bay},
                                        {"b_beam_in": 14.}, recipe_policy(max_shells=45000))
                self.assertEqual(r["status"], "resolved")
                self.assertEqual([m["cells_per_bay"][0] for m in r["resolved_levels"]], expected)
                self.assertEqual(len(r["resolved_levels"]) + len(r["dropped_levels"]), 4)
                self.assertTrue(all(m["shell_count"] <= 45000 for m in r["resolved_levels"]))
                self.assertTrue(all(m["shell_count"] > 45000 for m in r["dropped_levels"]))
        r = resolve_recipe_plan({"num_bay_x": 6, "num_bay_y": 6, "bay_x_in": 180., "bay_y_in": 180.},
                                {"b_beam_in": 14.}, recipe_policy(max_shells=5000))
        self.assertEqual(r["status"], "unresolved_budget")
        self.assertEqual(len(r["meshes"]), 0)
        self.assertIn("needs two", r["detail"])

    def test_probe_budget_retains_all_levels_for_the_largest_pilot_floor(self):
        r = resolve_recipe_plan({"num_bay_x": 6, "num_bay_y": 6, "bay_x_in": 288., "bay_y_in": 288.},
                                {"b_beam_in": 10.}, recipe_policy())
        self.assertEqual([m["cells_per_bay"] for m in r["resolved_levels"]], [[12,12],[24,24],[48,48],[60,60]])
        self.assertEqual(r["resolved_levels"][-1]["shell_count"], 129600)
        self.assertEqual(r["dropped_levels"], [])
        self.assertEqual(PROBE_SLAB_REFINEMENT["moment_tolerance"], .05)
        self.assertEqual(PROBE_SLAB_REFINEMENT["shear_tolerance"], .05)

    def test_beam_size_change_rebuilds_the_resolved_coordinates(self):
        geometry = {"num_bay_x": 2, "num_bay_y": 2, "bay_x_in": 240., "bay_y_in": 240.}
        a = resolve_recipe_plan(geometry, {"b_beam_in": 14.}, recipe_policy())
        b = resolve_recipe_plan(geometry, {"b_beam_in": 18.}, recipe_policy())
        self.assertNotEqual(a["meshes"], b["meshes"])
        ha = floor_mesh(2, 2, 240., 240., 4, mesh_spec=a["meshes"][-1])["coordinate_sha256"]
        hb = floor_mesh(2, 2, 240., 240., 4, mesh_spec=b["meshes"][-1])["coordinate_sha256"]
        self.assertNotEqual(ha, hb)

    def test_policy_forms_are_validated_before_any_resolution(self):
        bad = [dict(recipe_policy(), meshes=[]), recipe_policy(recipe="unknown_v0"), recipe_policy(levels=1),
               recipe_policy(levels=5), recipe_policy(max_shells=MAX_EXPLICIT_SHELLS + 1), recipe_policy(tolerance_basis=" "),
               {k: v for k, v in recipe_policy().items() if k != "max_shells"}]
        geometry = {"num_bay_x": 1, "num_bay_y": 1, "bay_x_in": 200., "bay_y_in": 200.}
        for plan in bad:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                _plan(geometry, plan, sections={"b_beam_in": 14.})
        with self.assertRaisesRegex(ValueError, "beam section"):
            _plan(geometry, recipe_policy())
        grids, resolution = _plan(geometry, explicit_policy())
        self.assertIsNone(resolution)
        self.assertEqual(len(grids), 2)


class RecipeEvidenceTests(unittest.TestCase):
    def run_fixture(self, plan, flags=None, side_effect=manufactured):
        with mock.patch("Design.SMRF_Slab_Actions.analyze_floor", side_effect=side_effect) as solve:
            evidence = build_refined_slab_action_evidence(SLAB, GEOMETRY, SECTIONS, .05, INPUTS, plan, assertions=flags)
        return evidence, solve

    def test_recipe_evidence_records_its_resolution_and_verifies(self):
        e, solve = self.run_fixture(recipe_policy(levels=2), signed())
        report = e["refinement"]
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["resolution"]["status"], "resolved")
        self.assertEqual(report["recipe_inputs"]["beam_width_in"], SECTIONS["b_beam_in"])
        self.assertEqual(len(report["resolved_meshes"]), 2)
        self.assertEqual(solve.call_count, 4)          # two cases at two levels
        self.assertTrue(refinement_verified(e))
        self.assertTrue(e["verified"])
        self.assertEqual({c["id"]: c["status"] for c in evaluate_slab_actions(e)}["floor.slab_action_mesh_refinement"], "pass")

    def test_recorded_inputs_that_no_longer_resolve_to_the_meshes_fail_verification(self):
        e, _ = self.run_fixture(recipe_policy(levels=2), signed())
        tampered = copy.deepcopy(e)
        tampered["refinement"]["recipe_inputs"]["beam_width_in"] = 18.
        self.assertFalse(refinement_verified(tampered))
        tampered = copy.deepcopy(e)
        tampered["refinement"]["resolved_meshes"][0]["x_offsets_in"][1] += 1.
        self.assertFalse(refinement_verified(tampered))
        tampered = copy.deepcopy(e)
        tampered["refinement"]["geometry"]["bay_x_in"] = 220.
        self.assertFalse(refinement_verified(tampered))

    def test_unresolved_budget_solves_nothing_and_cannot_qualify(self):
        e, solve = self.run_fixture(recipe_policy(max_shells=100), signed())
        solve.assert_not_called()
        self.assertEqual(e["refinement"]["status"], "unresolved_budget")
        self.assertIn("needs two", e["refinement"]["status_detail"])
        self.assertEqual(e["refinement"]["levels"], [])
        self.assertFalse(e["verified"])
        self.assertFalse(refinement_verified(e))
        self.assertEqual(evaluate_slab_actions(e)[0]["status"], "not_evaluated")

    def test_conflicting_plan_rejected_before_solving(self):
        with mock.patch("Design.SMRF_Slab_Actions.analyze_floor") as solve:
            with self.assertRaisesRegex(ValueError, "not both"):
                build_refined_slab_action_evidence(SLAB, GEOMETRY, SECTIONS, .05, INPUTS,
                                                   dict(recipe_policy(), meshes=explicit_policy()["meshes"]))
            solve.assert_not_called()


class ProbeWorkflowTests(unittest.TestCase):
    def test_both_probe_factories_share_the_recipe_and_keep_their_assertion_scopes(self):
        from Design.Verify_Designs import probe_config as verify_probe
        from Design.Evidence_Summary import probe_config as evidence_probe
        a, b = verify_probe("2026-09-24"), evidence_probe("2026-09-24")
        self.assertEqual(a.floor_analysis.slab_refinement, PROBE_SLAB_REFINEMENT)
        self.assertEqual(b.floor_analysis.slab_refinement, PROBE_SLAB_REFINEMENT)
        self.assertTrue(a.slab_actions.all_asserted() and b.slab_actions.all_asserted())
        self.assertTrue(a.verification.floor_hand_check_verified)
        self.assertEqual(b.verification.asserted_by, "")
        self.assertEqual(PROBE_SLAB_REFINEMENT["recipe"], "graded_face_v1")
        self.assertEqual(PROBE_SLAB_REFINEMENT["local_extension"],
                         {"method": "failed_witness_face_band_v1", "max_extra_levels": 2})
        self.assertEqual(PROBE_SLAB_REFINEMENT["moment_tolerance"], .05)
        self.assertEqual(PROBE_SLAB_REFINEMENT["shear_tolerance"], .05)
        self.assertEqual(PROBE_SLAB_REFINEMENT["max_shells"], 130000)

    def test_methodology_identity_is_geometry_independent_with_a_recipe(self):
        from Design.Verify_Designs import probe_config, methodology_sha256
        import Structure_Parameters as sp
        from Design import Design_Driver as driver
        with mock.patch.object(sp, "BAY_X", 240.), mock.patch.object(sp, "NUM_BAY_X", 2):
            first = driver.design_request_identity(probe_config("2026-09-24"))
        with mock.patch.object(sp, "BAY_X", 180.), mock.patch.object(sp, "NUM_BAY_X", 5):
            second = driver.design_request_identity(probe_config("2026-09-24"))
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(methodology_sha256(first), methodology_sha256(second))

    def test_asserted_design_without_a_plan_is_named_before_any_solve(self):
        from Design import Design_Driver as driver
        cfg = DesignConfig(slab_actions=SlabActionAssertions(
            analysis_applicability_verified=True, all_floors_enveloped=True, load_pattern_envelope_verified=True,
            spatial_envelope_per_unit_width=True, twisting_moment_resolution_verified=True,
            zero_membrane_force_verified=True, verified=True, two_way_shear_path_assessed=True,
            asserted_by="test", assertion_date="2026-09-24", assertion_basis="unit test"))
        self.assertIsNone(cfg.floor_analysis.slab_refinement)
        with mock.patch("Design.SMRF_Slab_Actions.analyze_floor") as solve:
            with self.assertRaisesRegex(ValueError, "FloorAnalysisConfig.slab_refinement"):
                driver._update_slab_reinforcement(cfg, {})
            solve.assert_not_called()
        # Without assertions a missing plan is still the single-mesh path, unchanged.
        self.assertIsNone(DesignConfig().floor_analysis.slab_refinement)
        self.assertIsInstance(FloorAnalysisConfig(slab_refinement=dict(PROBE_SLAB_REFINEMENT)).slab_refinement, dict)


if __name__ == "__main__":
    unittest.main()
