"""Native response equivalence for pattern SPs versus the previous fix path."""
import numbers
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Design import SMRF_Floor_Analysis as floor
from Design.SMRF_Floor_Mesh import graded_face_levels


class FloorConstraintTests(unittest.TestCase):
    def setUp(self):
        ops.wipe()
        self.addCleanup(ops.wipe)

    def assert_response_equal(self, left, right, path="result"):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.assert_response_equal(left[key], right[key], path + "." + str(key))
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for i, (a, b) in enumerate(zip(left, right)):
                self.assert_response_equal(a, b, path + f"[{i}]")
        elif isinstance(left, numbers.Real) and not isinstance(left, bool):
            self.assertAlmostEqual(left, right, delta=2e-8 + 1e-9 * abs(left), msg=path)
        else:
            self.assertEqual(left, right, path)

    def test_native_full_response_matches_previous_fix_constraints(self):
        slab = dict(thickness_in=6., concrete_fc_ksi=8., concrete_unit_weight_kcf=.15,
                    superimposed_dead_load_ksf=.05)
        geometry = dict(num_bay_x=2, num_bay_y=2, bay_x_in=240., bay_y_in=300.)
        sections = dict(b_beam_in=14., h_beam_in=28., fc_beam_ksi=8.)
        for support, explicit, pattern in (
                ("rigid_lines", False, "all"), ("rigid_lines", True, [[0, 0], [1, 0]]),
                ("flexible_beams", False, [[0, 1]]), ("flexible_beams", True, "all")):
            with self.subTest(support=support, explicit=explicit, pattern=pattern):
                case = dict(id="SP_equivalence", dead_factor=1.2, live_factor=1.6,
                            live_load_ksf=.05, live_pattern=pattern)
                mesh_spec = None if not explicit else dict(
                    x_offsets_in=graded_face_levels(240., 14.)[1],
                    y_offsets_in=graded_face_levels(300., 14.)[1], max_shells=45000)
                current = floor.analyze_floor(slab, geometry, sections, case, 4, support,
                                              mesh_spec=mesh_spec)
                with mock.patch.object(floor, "_constrain_plate_node",
                                       side_effect=lambda n, held: ops.fix(n, 1, 1, int(held), 0, 0, 1)):
                    legacy = floor.analyze_floor(slab, geometry, sections, case, 4, support,
                                                 mesh_spec=mesh_spec)
                expected_status = "transfer_complete" if support == "flexible_beams" else "diagnostic_complete"
                self.assertEqual(current["status"], expected_status)
                self.assertTrue(current["equilibrium"]["numerical_balance_passed"])
                self.assertFalse(current["verified"])
                # Includes every Gauss-point moment/shear, deflection, reaction,
                # flexible-beam force/transfer, and the untouched qualification flags.
                self.assert_response_equal(current, legacy)
                self.assertEqual(ops.getNodeTags(), [])

    def test_actual_zero_restraints_remain_effective_at_nonunit_load_factors(self):
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.timeSeries("Constant", 2)
        ops.pattern("Plain", 2, 2)
        ops.node(1, 0., 0., 0.)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.uniaxialMaterial("Elastic", 1, 10.)
        for tag, held in ((2, False), (3, True)):
            ops.node(tag, 0., 0., 0.)
            floor._constrain_plate_node(tag, held)
            ops.element("zeroLength", tag, 1, tag, "-mat", *([1] * 6), "-dir", 1, 2, 3, 4, 5, 6)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for tag in (2, 3):
            ops.load(tag, 1., 1., 1., 1., 1., 1.)
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.algorithm("Linear")
        ops.integrator("LoadControl", .5)
        ops.analysis("Static")
        for factor in (.5, 1., 1.5):
            self.assertEqual(ops.analyze(1), 0)
            ops.reactions()
            for tag, held in ((2, False), (3, True)):
                for dof in range(1, 7):
                    fixed = dof in (1, 2, 6) or (held and dof == 3)
                    self.assertAlmostEqual(ops.nodeDisp(tag, dof), 0. if fixed else factor / 10., places=12)
                    self.assertAlmostEqual(ops.nodeReaction(tag, dof), -factor if fixed else 0., places=12)


if __name__ == "__main__":
    unittest.main()
