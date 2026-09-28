"""Scissors joint springs in the nonlinear frame (Model/Joint_Springs, 2026-09-27).

Covers the calibration arithmetic against a hand calculation, the transcribed ASCE 41 rows and their
interpolation, the slip partition switch on the member hinges, the frame topology under the joint
model (and its absence under the legacy model), static equilibrium and modal softening with joints,
and the output identity keys.
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openseespy.opensees as ops  # noqa: E402

import Structure_Parameters as sp  # noqa: E402
from Data_Generation.Graph_Exporter import collect_global_parameters  # noqa: E402
from Ground_Motion_Main import OUTPUT_IDENTITY_KEYS, validate_ntha_output_compatibility  # noqa: E402
from Model import Build_Model, IMK_Calibration, Joint_Springs  # noqa: E402

SMALL = {"NUM_BAY_X": 2, "NUM_BAY_Y": 1, "NUM_FLOOR": 2, "BAY_X": 240.0, "BAY_Y": 240.0, "STORY_H": 144.0}


def small_frame(**overrides):
    """Patch a 2 x 1 x 2 frame with the IMK formulation; JOINT_SHEAR_CATEGORIES cleared (framing fallback)."""
    values = {**SMALL, "ELEMENT_FORMULATION": "imk", "JOINT_SHEAR_CATEGORIES": None, **overrides}
    return contextlib.ExitStack(), values


class CalibrationArithmetic(unittest.TestCase):
    def test_interior_floor_joint_hand_check(self):
        # 24x24 column at 5 ksi, 20 in wide x 24 in deep beams on all four faces, floor joint, no record.
        with mock.patch.multiple(sp, NUM_BAY_X=2, NUM_BAY_Y=2, NUM_FLOOR=3, B_COL=24.0, H_COL=24.0, FC_COL_KSI=5.0,
                                 B_BEAM=20.0, H_BEAM=24.0, JOINT_SHEAR_CATEGORIES=None, JOINT_STIFFNESS_MODIFIER=1.0,
                                 JOINT_KAPPA_F=0.25, JOINT_KAPPA_D=0.25, JOINT_LAMBDA_SUPPRESSION=1.0e12), \
                mock.patch.object(IMK_Calibration, "column_gravity_axial", return_value=0.0), \
                mock.patch.object(Joint_Springs, "column_gravity_axial", return_value=0.0):
            cal = Joint_Springs.joint_calibration(1, 1, 1, "x")
        self.assertEqual(cal["kind"], "interior")
        self.assertEqual((cal["column_state"], cal["beam_state"], cal["confined"]), ("continuous", "continuous", True))
        self.assertEqual(cal["gamma"], 20.0)
        aj = 24.0 * 24.0                                # bj = min(24, 20 + 24, 20 + 2*2) = 24
        vn = 20.0 * math.sqrt(5000.0) * aj / 1000.0     # 814.6 kip
        self.assertAlmostEqual(cal["aj_in2"], aj)
        self.assertAlmostEqual(cal["vn_kip"], vn, places=9)
        self.assertAlmostEqual(cal["mn_kip_in"], vn * 24.0, places=9)
        g = 57.0 * math.sqrt(5000.0) / 2.4
        self.assertAlmostEqual(cal["ke_kip_in_per_rad"], g * aj * 24.0, places=6)
        self.assertAlmostEqual(cal["theta_y_rad"], (vn / aj) / g, places=12)
        self.assertEqual(cal["joint_class"], "interior")
        self.assertEqual((cal["a_rad"], cal["b_rad"], cal["c_residual"]), (0.015, 0.030, 0.2))
        self.assertAlmostEqual(cal["dpc_rad"], 0.015 / 0.8)
        self.assertAlmostEqual(cal["du_rad"], cal["theta_y_rad"] + 0.030)
        self.assertEqual(cal["backbone"].arguments(), (0.015, 0.015 / 0.8, cal["du_rad"], cal["mn_kip_in"], 1.0, 0.2))
        self.assertEqual(cal["cyclic"].arguments("IMKPinching")[-2:], (0.25, 0.25))

    def test_record_category_wins_over_framing(self):
        saved = {"joint_shear/floor/corner/x": {"gamma": 12.0, "nominal_vn_kip": 500.0, "aj_in2": 576.0, "vj_kip": 400.0,
                                                "classification": {"confinement": {"confined": False},
                                                                   "beam": {"state": "other"},
                                                                   "column": {"state": "continuous"}}}}
        with mock.patch.multiple(sp, NUM_BAY_X=2, NUM_BAY_Y=2, NUM_FLOOR=3, B_COL=24.0, H_COL=24.0, FC_COL_KSI=5.0,
                                 B_BEAM=20.0, H_BEAM=24.0, JOINT_SHEAR_CATEGORIES=saved), \
                mock.patch.object(Joint_Springs, "column_gravity_axial", return_value=0.0):
            cal = Joint_Springs.joint_calibration(1, 0, 0, "x")
        self.assertEqual(cal["source"], "design_record_joint_shear_category")
        self.assertEqual((cal["gamma"], cal["vn_kip"], cal["aj_in2"]), (12.0, 500.0, 576.0))
        self.assertAlmostEqual(cal["shear_ratio"], 0.8)
        self.assertEqual(cal["joint_class"], "other")
        self.assertEqual((cal["a_rad"], cal["b_rad"]), (0.010, 0.020))

    def test_interior_class_is_in_plane_not_confinement(self):
        # 16 in beams on a 32 in column do not confine the joint (gamma 15, not 20), but a joint with beams
        # on both faces in the plane's direction is still an ASCE 41 "interior joint" for the deformation row.
        with mock.patch.multiple(sp, NUM_BAY_X=2, NUM_BAY_Y=2, NUM_FLOOR=3, B_COL=32.0, H_COL=32.0, FC_COL_KSI=8.0,
                                 B_BEAM=16.0, H_BEAM=24.0, JOINT_SHEAR_CATEGORIES=None), \
                mock.patch.object(Joint_Springs, "column_gravity_axial", return_value=0.0):
            interior = Joint_Springs.joint_calibration(1, 1, 1, "x")
            edge = Joint_Springs.joint_calibration(1, 0, 1, "x")
        self.assertFalse(interior["confined"])
        self.assertEqual(interior["gamma"], 15.0)
        self.assertEqual(interior["joint_class"], "interior")
        self.assertEqual((interior["a_rad"], interior["b_rad"]), (0.015, 0.030))
        self.assertEqual(edge["beams_in_direction"], 1)
        self.assertEqual(edge["joint_class"], "other")

    def test_asce41_rows_pinned_and_interpolated(self):
        # Verified 2026-09-27 against Elwood et al. (2007) PEER "Update to ASCE/SEI 41 Concrete
        # Provisions" Table 6-9 (adopted as ASCE 41-06 S1, retained in 41-13/41-17 Table 10-11).
        expected = {
            ("interior", 0.1, 1.2): (0.015, 0.030, 0.2), ("interior", 0.1, 1.5): (0.015, 0.030, 0.2),
            ("interior", 0.4, 1.2): (0.015, 0.025, 0.2), ("interior", 0.4, 1.5): (0.015, 0.020, 0.2),
            ("other", 0.1, 1.2): (0.010, 0.020, 0.2), ("other", 0.1, 1.5): (0.010, 0.015, 0.2),
            ("other", 0.4, 1.2): (0.010, 0.020, 0.2), ("other", 0.4, 1.5): (0.010, 0.015, 0.2),
        }
        self.assertEqual(Joint_Springs.ASCE41_JOINT_ROWS, expected)
        self.assertEqual(Joint_Springs.ASCE41_JOINT_ROWS_NONCONFORMING[("other", 0.4, 1.5)], (0.0, 0.0075, 0.0))
        a, b, c, row = Joint_Springs.asce41_deformation("interior", 0.25, 1.0)
        self.assertEqual((a, c), (0.015, 0.2))
        self.assertAlmostEqual(b, 0.0275)              # halfway between 0.030 (P <= 0.1) and 0.025 (P >= 0.4)
        self.assertEqual((row["t_axial"], row["t_shear"]), (0.5, 0.0))
        a, b, c, _ = Joint_Springs.asce41_deformation("other", 0.0, 1.35)
        self.assertAlmostEqual(b, 0.0175)              # halfway between 0.020 (V/Vn <= 1.2) and 0.015 (>= 1.5)
        with self.assertRaises(ValueError):
            Joint_Springs.asce41_deformation("knee", 0.0, 1.0)


class SlipPartition(unittest.TestCase):
    def test_member_hinges_drop_slip_only_under_joint_springs_with_slip_scope(self):
        with mock.patch.multiple(sp, JOINT_MODEL="rigid_centerline", JOINT_DEFORMATION_SCOPE="joint_shear_and_slip"):
            self.assertEqual(IMK_Calibration.bond_slip_indicator(), 1.0)
        with mock.patch.multiple(sp, JOINT_MODEL="imk_pinching_scissors", JOINT_DEFORMATION_SCOPE="joint_shear_only"):
            self.assertEqual(IMK_Calibration.bond_slip_indicator(), 1.0)
        with mock.patch.multiple(sp, JOINT_MODEL="imk_pinching_scissors", JOINT_DEFORMATION_SCOPE="joint_shear_and_slip"):
            self.assertEqual(IMK_Calibration.bond_slip_indicator(), 0.0)
            without = IMK_Calibration.haselton_theta_p("beam_x", 0.0)
        with mock.patch.multiple(sp, JOINT_MODEL="rigid_centerline"):
            with_slip = IMK_Calibration.haselton_theta_p("beam_x", 0.0)
        if with_slip > IMK_Calibration.THETA_P_FLOOR and without > IMK_Calibration.THETA_P_FLOOR:
            self.assertAlmostEqual(without / with_slip, 1.0 / 1.55, places=9)


def build_small(joint_model):
    with mock.patch.multiple(sp, **SMALL, ELEMENT_FORMULATION="imk", JOINT_SHEAR_CATEGORIES=None,
                             JOINT_MODEL=joint_model), contextlib.redirect_stdout(io.StringIO()):
        Build_Model.build_model()
        Build_Model.apply_analysis_constraints()
        joints = Joint_Springs.joint_registry()
        nodes = set(ops.getNodeTags())
        elements = set(ops.getEleTags())
        period = 2.0 * math.pi / math.sqrt(ops.eigen("-genBandArpack", 1)[0])
        return joints, nodes, elements, period


class FrameTopology(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_joint_model_adds_one_core_and_one_spring_per_elevated_joint(self):
        joints, nodes, elements, period = build_small("imk_pinching_scissors")
        expected = SMALL["NUM_FLOOR"] * (SMALL["NUM_BAY_X"] + 1) * (SMALL["NUM_BAY_Y"] + 1)
        self.assertEqual(len(joints), expected)
        for joint, entry in joints.items():
            self.assertIn(entry["beam_core"], nodes)
            self.assertIn(entry["element"], elements)
            self.assertEqual(ops.nodeCoord(entry["beam_core"]), ops.nodeCoord(joint))
            self.assertEqual(set(entry["planes"]), {"x", "y"})
            self.assertEqual(entry["planes"]["x"]["direction"], 5)
            self.assertEqual(entry["planes"]["y"]["direction"], 4)
            self.assertEqual(entry["planes"]["x"]["installed"]["material_type"], "IMKPinching")
        # Beams tie to the beam cores, columns to the joints; the registry keeps the joints as identity.
        from Model.IMK_Hinges import hinge_registry
        for tag, member in hinge_registry().items():
            if member["member_type"] == "column":
                self.assertEqual((member["end_node_i"], member["end_node_j"]), (member["node_i"], member["node_j"]))
            else:
                self.assertEqual(member["end_node_i"], joints[member["node_i"]]["beam_core"])
                self.assertEqual(member["end_node_j"], joints[member["node_j"]]["beam_core"])
                self.assertEqual(member["installed_materials"]["i"]["y"]["provenance"]["bond_slip_indicator"], 0.0)
        self.assertGreater(period, 0.0)

    def test_legacy_model_has_no_cores_and_is_stiffer(self):
        joints_on, nodes_on, elements_on, period_on = build_small("imk_pinching_scissors")
        ops.wipe()
        joints_off, nodes_off, elements_off, period_off = build_small("rigid_centerline")
        self.assertEqual(joints_off, {})
        self.assertEqual(len(nodes_on) - len(nodes_off), len(joints_on))
        self.assertEqual(len(elements_on) - len(elements_off), len(joints_on))
        self.assertFalse(any(t >= sp.JOINT_BEAM_CORE_TAG_BASE and t < sp.IMK_HINGE_NODE_TAG_BASE for t in nodes_off))
        self.assertGreater(period_on, period_off, "joint flexibility lengthens the fundamental period")
        self.assertLess(period_on / period_off, 1.5, "but not absurdly: the joints are stiff panels")

    def test_lateral_push_is_carried_to_the_base_and_rotates_the_joints(self):
        joints, _, _, _ = build_small("imk_pinching_scissors")
        with mock.patch.multiple(sp, **SMALL, ELEMENT_FORMULATION="imk"):
            top = Build_Model.floor_master_node(SMALL["NUM_FLOOR"])
            push = 50.0
            ops.timeSeries("Linear", 901)
            ops.pattern("Plain", 901, 901)
            ops.load(top, push, 0.0, 0.0, 0.0, 0.0, 0.0)
            ops.system("BandGeneral")
            ops.numberer("RCM")
            ops.test("NormDispIncr", 1e-8, 60)
            ops.algorithm("Newton")
            ops.integrator("LoadControl", 0.1)
            ops.analysis("Static")
            for _ in range(10):
                self.assertEqual(ops.analyze(1), 0)
            base_columns = range(1, (SMALL["NUM_BAY_X"] + 1) * (SMALL["NUM_BAY_Y"] + 1) + 1)
            base_shear = sum(ops.eleForce(tag)[0] for tag in base_columns)
            self.assertAlmostEqual(abs(base_shear), push, delta=1e-6 * push)
            rotations = [abs(ops.eleResponse(entry["element"], "deformation")[1]) for entry in joints.values()]
            self.assertGreater(max(rotations), 0.0, "an X push shears the x-z joint planes (direction 5)")
            self.assertLess(max(rotations), 1e-3, "elastic at this load")

    def test_joint_model_refuses_the_elastic_formulation(self):
        with mock.patch.multiple(sp, **SMALL, ELEMENT_FORMULATION="fiber", JOINT_MODEL="imk_pinching_scissors"):
            with self.assertRaisesRegex(ValueError, "requires ELEMENT_FORMULATION 'imk'"):
                Build_Model.build_model()


class Identity(unittest.TestCase):
    def test_joint_keys_are_part_of_the_output_identity(self):
        for key in ("joint_model", "joint_deformation_scope", "joint_calibration_basis", "joint_kappa",
                    "imk_bond_slip_indicator"):
            self.assertIn(key, OUTPUT_IDENTITY_KEYS)
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "global_parameters.json").write_text(json.dumps(collect_global_parameters()),
                                                                   encoding="utf-8")
            other = "rigid_centerline" if sp.JOINT_MODEL == "imk_pinching_scissors" else "imk_pinching_scissors"
            with mock.patch.object(sp, "JOINT_MODEL", other):
                with self.assertRaisesRegex(RuntimeError, "joint_model"):
                    validate_ntha_output_compatibility(Path(temp_dir))
            with mock.patch.object(sp, "JOINT_KAPPA_F", 0.5):
                with self.assertRaisesRegex(RuntimeError, "joint_kappa"):
                    validate_ntha_output_compatibility(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
