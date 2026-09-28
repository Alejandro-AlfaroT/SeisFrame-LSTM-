"""Bounded coupled-model checks, not structural design qualification."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
import tempfile
import contextlib
import io
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity

SLAB = {"thickness_in": 5., "concrete_fc_ksi": 4., "concrete_unit_weight_kcf": .15,
        "superimposed_dead_load_ksf": .05}
GEOMETRY = {"num_bay_x": 1, "num_bay_y": 1, "num_floor": 2,
            "bay_x_in": 240., "bay_y_in": 200., "story_h_in": 144.}
SECTIONS = {"b_beam_in": 12., "h_beam_in": 18., "fc_beam_ksi": 4.,
            "b_col_in": 18., "h_col_in": 18., "fc_col_ksi": 5.}
DEAD = {"id": "D", "dead_factor": 1., "live_factor": 0., "live_load_ksf": .05, "live_pattern": "none"}
ZERO = {**DEAD, "id": "unloaded", "dead_factor": 0.}


class CoupledAnalysisTests(unittest.TestCase):
    def setUp(self):
        ops.wipe()
        self.addCleanup(ops.wipe)

    def analyze(self, geometry=None, sections=None, cases=None, **kwargs):
        g = geometry or GEOMETRY
        return analyze_coupled_gravity(SLAB, g, sections or SECTIONS, cases or [DEAD]*g["num_floor"], **kwargs)

    def test_symmetric_two_story_column_shortening_matches_axial_hand_solution(self):
        result = self.analyze(include_member_weight=False)
        p_floor = .1125*240*200/144
        ea = 18.*18.*57.*math.sqrt(5000.)
        expected = [-2.*p_floor/4*144/ea, -3.*p_floor/4*144/ea]
        self.assertEqual(result["status"], "diagnostic_complete")
        for floor, uz in zip(result["floors"], expected):
            for joint in floor["column_joint_displacements"]:
                self.assertAlmostEqual(joint["displacement_rotation"][2], uz, places=10)
        self.assertTrue(result["equilibrium"]["numerical_balance_passed"])
        self.assertTrue(result["shell_to_frame_equilibrium"]["numerical_balance_passed"])

    def test_loaded_roof_transmits_through_unloaded_lower_floor(self):
        result = self.analyze(cases=[ZERO, DEAD], include_member_weight=False)
        p = .1125*240*200/144
        ea = 18.*18.*57.*math.sqrt(5000.)
        for k, floor in enumerate(result["floors"], 1):
            for joint in floor["column_joint_displacements"]:
                self.assertAlmostEqual(joint["displacement_rotation"][2], -k*p/4*144/ea, places=10)
        self.assertAlmostEqual(result["equilibrium"]["base_force_kip"][2], p, places=8)

    def test_weight_ledger_excludes_slab_from_web_and_column_weight(self):
        result = self.analyze()
        gamma = .15/1728
        expected_slab = 2.*.1125*240*200/144
        expected_web = 2.*12*(18-5)*(2*(240-18)+2*(200-18))*gamma
        expected_column = 2.*4.*18*18*(144-5)*gamma
        ledger = result["weight_ledger"]
        self.assertAlmostEqual(ledger["slab_area_load_kip"], expected_slab)
        self.assertAlmostEqual(ledger["beam_drop_weight_kip"], expected_web)
        self.assertAlmostEqual(ledger["column_weight_kip"], expected_column)
        self.assertAlmostEqual(result["equilibrium"]["base_force_kip"][2], sum(ledger.values()), places=8)
        # Frame interface receives just slab pressure; web/column weights are
        # applied once, on their frame elements, not again through the shell.
        self.assertAlmostEqual(-result["shell_to_frame_equilibrium"]["interface_force_kip"][2], expected_slab, places=8)
        self.assertEqual(result["web_section"]["area_in2"], 12.*13.)
        self.assertEqual(result["web_centroid_offset_from_slab_midplane_in"], -9.)

    def test_asymmetric_patterns_balance_all_forces_moments_and_offset_compatibility(self):
        g = {**GEOMETRY, "num_bay_x": 2, "num_bay_y": 2}
        live = {**DEAD, "id": "corner_live", "dead_factor": 0., "live_factor": 1., "live_pattern": [[0, 0]]}
        for mesh in (2, 4, 6):
            with self.subTest(mesh=mesh):
                result = self.analyze(geometry=g, cases=[ZERO, live], mesh_per_bay=mesh)
                self.assertEqual(result["status"], "diagnostic_complete")
                self.assertLess(result["equilibrium"]["force_relative_error"], 1e-8)
                self.assertLess(result["equilibrium"]["moment_relative_error"], 1e-8)
                self.assertLess(result["rigid_offset_max_residual"], 1e-10)
                self.assertTrue(result["shell_to_frame_equilibrium"]["numerical_balance_passed"])
                p = .05*240*200/144
                self.assertAlmostEqual(result["equilibrium"]["applied_moment_kip_in"][0], -p*100)
                self.assertAlmostEqual(result["equilibrium"]["applied_moment_kip_in"][1], p*120)

    def test_real_support_flexibility_and_nonzero_membrane_are_retained(self):
        result = self.analyze(include_member_weight=False)
        rotations = [abs(v) for f in result["floors"] for j in f["column_joint_displacements"]
                     for v in j["displacement_rotation"][3:5]]
        self.assertGreater(max(rotations), 1e-6)
        membrane = [abs(row["gauss_resultants_raw"][8*k+d]) for row in result["shell_resultants"]
                    for k in range(4) for d in range(3)]
        self.assertGreater(max(membrane), 1e-6)
        stronger = self.analyze(sections={**SECTIONS, "fc_col_ksi": 20.}, include_member_weight=False)
        uz = result["floors"][0]["column_joint_displacements"][0]["displacement_rotation"][2]
        stronger_uz = stronger["floors"][0]["column_joint_displacements"][0]["displacement_rotation"][2]
        self.assertAlmostEqual(stronger_uz, uz/2, places=10)

    def test_json_safe_unqualified_and_no_global_parameter_mutation(self):
        import Structure_Parameters as sp
        before = (sp.NUM_BAY_X, sp.B_BEAM, sp.SLAB_THICKNESS_IN, sp.FLOOR_TRANSFER)
        inputs = copy.deepcopy((SLAB, GEOMETRY, SECTIONS))
        result = json.loads(json.dumps(self.analyze(), allow_nan=False))
        self.assertFalse(result["verified"])
        self.assertFalse(result["applied_to_design"])
        self.assertEqual(before, (sp.NUM_BAY_X, sp.B_BEAM, sp.SLAB_THICKNESS_IN, sp.FLOOR_TRANSFER))
        self.assertEqual(inputs, (SLAB, GEOMETRY, SECTIONS))
        self.assertEqual(ops.getNodeTags(), [])

    def test_existing_domain_is_preserved_and_failed_solve_cleans_scratch(self):
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(777, 0., 0., 0.)
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            self.analyze()
        self.assertEqual(ops.getNodeTags(), [777])
        ops.wipe()
        with mock.patch.object(ops, "analyze", return_value=-3):
            self.assertEqual(self.analyze()["status"], "analysis_failed")
        self.assertEqual(ops.getNodeTags(), [])

    def test_floor_inventory_and_total_mesh_limit_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "one explicit"):
            self.analyze(cases=[DEAD])
        with self.assertRaisesRegex(ValueError, "total shells"):
            self.analyze(geometry={**GEOMETRY, "num_floor": 100}, mesh_per_bay=16)
        with self.assertRaisesRegex(ValueError, "downstand"):
            self.analyze(sections={**SECTIONS, "h_beam_in": 5.})
        self.assertEqual(ops.getNodeTags(), [])

    def test_review_runner_preserves_source_and_refuses_existing_output(self):
        from tools.review_coupled_gravity import run_review
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp)/"design.json", Path(temp)/"review"
            payload = json.dumps({"slab": SLAB, "geometry": GEOMETRY, "sections": SECTIONS,
                                  "floor_loads": {"floor_live_load_ksf": .05}})
            source.write_text(payload, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_review(source, destination, meshes=[2])
            self.assertEqual(source.read_text(encoding="utf-8"), payload)
            self.assertEqual(len(summary["results"]), 2)
            self.assertTrue(all(r["status"] == "diagnostic_complete" for r in summary["results"]))
            self.assertFalse(summary["generation_launched"])
            self.assertTrue((destination/"REVIEW.md").is_file())
            with self.assertRaises(FileExistsError):
                run_review(source, destination, meshes=[2])


if __name__ == "__main__":
    unittest.main()
