"""Verification of the peak-oriented and pinching reloading rules as OpenSees 3.8 implements them, and
of the output identity's response to a deterioration-mode change.

Measured rules (this file pins them; see Model/IMK_MATERIALS.md):
  * IMKPeakOriented: after a full reversal the reloading path is one straight line from the zero-force
    crossing to the previous peak (with whatever strength that peak has deteriorated to), so its tangent
    is far below Ke and constant. IMKBilin reloads at Ke until it meets the backbone.
  * IMKPinching: the reloading path first heads to a break point at rotation (1 - kappaD) x the permanent
    rotation left after unloading from the previous peak (its zero-moment crossing), at kappaF x the force
    the peak-oriented straight line would have there, then to the peak. The test uses a spring whose yield
    rotation is a fifth of the peak so that this is distinguishable from (1 - kappaD) x the peak rotation.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openseespy.opensees as ops  # noqa: E402

import Structure_Parameters as sp  # noqa: E402
from Data_Generation.Graph_Exporter import collect_global_parameters, installed_deterioration_policy  # noqa: E402
from Ground_Motion_Main import OUTPUT_IDENTITY_KEYS, validate_ntha_output_compatibility  # noqa: E402
from Model import IMK_Calibration, IMK_Hinges, IMK_Materials  # noqa: E402

KE, FY = 1.0e6, 100.0
PEAK = 0.02


def walk(mat_tag, path, steps=400):
    """Drive a material through straight-line rotation targets; return (rotation, moment, tangent)."""
    ops.testUniaxialMaterial(mat_tag)
    history, current = [], 0.0
    for target in path:
        for j in range(1, steps + 1):
            rotation = current + (target - current) * j / steps
            ops.setStrain(rotation)
            history.append((rotation, ops.getStress(), ops.getTangent()))
        current = target
    return history


def reload_segment(history, steps=400):
    """The third leg (-PEAK -> +PEAK) restricted to positive moment: the reloading path."""
    return [(r, m, t) for r, m, t in history[2 * steps:3 * steps] if m > 0.0]


class PeakOrientedReloading(unittest.TestCase):
    def setUp(self):
        ops.wipe()

    def tearDown(self):
        ops.wipe()

    def install_member_material(self, kind):
        with mock.patch.object(sp, "IMK_DETERIORATION_MODE", "haselton_2008"), \
                mock.patch.object(sp, "IMK_MATERIAL_TYPE", kind):
            backbone = IMK_Calibration.deterioration_for_member("beam_x", 0.0)
            IMK_Hinges._define_imk_peak_material(1, KE, FY, backbone)

    def test_peak_oriented_reloads_on_one_line_to_the_previous_peak(self):
        self.install_member_material("IMKPeakOriented")
        history = walk(1, [PEAK, -PEAK, PEAK])
        first_peak = max(m for _, m, _ in history[:400])
        path = reload_segment(history)
        zero_rotation = path[0][0]
        # Skip the first point (the elastic-to-reload corner) and the arrival at the peak.
        body = [(r, m, t) for r, m, t in path[1:] if r < PEAK - 1e-6]
        tangents = [t for _, _, t in body]
        self.assertGreater(len(body), 100)
        self.assertLess(max(tangents), 0.01 * KE, "peak-oriented reloading is far softer than Ke")
        self.assertLess((max(tangents) - min(tangents)) / max(tangents), 0.02, "one straight line")
        # The line aims at the previous peak: extrapolating it to PEAK gives the moment actually reached there.
        r1, m1, _ = body[-1]
        slope = m1 / (r1 - zero_rotation)
        reached = history[3 * 400 - 1][1]
        self.assertAlmostEqual(slope * (PEAK - zero_rotation), reached, delta=0.01 * reached)
        self.assertLess(reached, first_peak, "strength deteriorates between the two positive peaks")

    def test_bilin_reloads_at_ke_until_the_backbone(self):
        self.install_member_material("IMKBilin")
        history = walk(1, [PEAK, -PEAK, PEAK])
        path = reload_segment(history)
        # Reloading at Ke from zero moment reaches the (deteriorated) yield strength within one step of
        # 1e-4 rad (Ke * 1e-4 = My), so the elastic branch is a single point: its tangent is Ke and the
        # moment sits on the backbone right after.
        self.assertGreater(path[0][2], 0.5 * KE, "the reloading branch starts at Ke")
        self.assertGreater(path[1][1], 0.5 * FY, "and meets the backbone within one step")
        body = [(r, m, t) for r, m, t in path[2:] if r < PEAK - 1e-6]
        soft = [t for _, _, t in body if t < 0.01 * KE]
        self.assertGreater(len(soft), 100, "then the hardening backbone")
        self.assertEqual(len(soft), len(body), "no peak-oriented straight line: backbone only after yield")


class PinchingBreakPoint(unittest.TestCase):
    def setUp(self):
        ops.wipe()

    def tearDown(self):
        ops.wipe()

    def test_break_point_follows_kappa_d_and_kappa_f(self):
        branch = IMK_Materials.RotationalBackbone(0.02, 0.06, 0.12, FY, 1.10, 0.20)
        ke = FY / (0.2 * PEAK)                     # yield rotation a fifth of the peak: theta_y = 0.004 rad
        for kappa_f, kappa_d in ((0.5, 0.5), (0.3, 0.7), (0.7, 0.3)):
            ops.wipe()
            cyclic = IMK_Materials.CyclicParameters(
                lamda_s=1e12, lamda_c=1e12, lamda_k=1e12, c_s=1.0, c_c=1.0, c_k=1.0, d_pos=1.0, d_neg=1.0,
                lamda_a=1e12, c_a=1.0, kappa_f=kappa_f, kappa_d=kappa_d)
            IMK_Materials.define_rotational_imk(
                "IMKPinching", 1, ke, branch, branch, cyclic,
                provenance={"calibration_id": "verification_fixture", "status": "verification_only",
                            "deformation_scope": "material_test"})
            history = walk(1, [PEAK, -PEAK, PEAK])
            first_peak = max(m for _, m, _ in history[:400])
            # permanent rotation: the zero-moment crossing while unloading from the positive peak
            permanent = next(r for r, m, _ in history[400:800] if m <= 0.0)
            path = reload_segment(history)
            zero_rotation = path[0][0]
            body = [(r, m, t) for r, m, t in path[1:] if r < PEAK - 1e-6]
            # Two straight branches: the tangent holds its first value up to the break point, changes
            # over at most two points, then holds a second value to the peak.
            first_tangent = body[0][2]
            departure = next(i for i, (_, _, t) in enumerate(body) if abs(t / first_tangent - 1.0) > 0.01)
            break_rotation, break_moment, _ = body[departure]
            second_tangent = body[departure + 2][2]
            self.assertGreater(second_tangent, first_tangent, "stiffer second branch toward the peak")
            for _, _, t in body[departure + 2:]:
                self.assertAlmostEqual(t / second_tangent, 1.0, delta=0.01)
            expected_rotation = (1.0 - kappa_d) * permanent
            line_moment_there = first_peak * (expected_rotation - zero_rotation) / (PEAK - zero_rotation)
            # the permanent rotation and the break are each read off the 1e-4 rad grid: allow 2.5 steps
            self.assertAlmostEqual(break_rotation, expected_rotation, delta=PEAK / 400 * 2.5,
                                   msg=f"break rotation for kappa ({kappa_f}, {kappa_d})")
            # ...and clearly NOT (1 - kappaD) x the peak rotation with this yield rotation
            self.assertGreater(abs(break_rotation - (1.0 - kappa_d) * PEAK), 10 * PEAK / 400)
            self.assertAlmostEqual(break_moment / line_moment_there, kappa_f, delta=0.02,
                                   msg=f"break moment for kappa ({kappa_f}, {kappa_d})")


class DiagnosticRunnerDefaults(unittest.TestCase):
    def test_ground_motion_runner_defaults_to_the_production_material(self):
        from Analysis.Fixed_Design_Diagnostics import build_parser
        args = build_parser().parse_args(["ground-motion", "--root", "r", "--case", "c", "--output-root", "o",
                                          "--result-id", "1"])
        self.assertEqual(args.member_material, sp.IMK_MATERIAL_TYPE)
        self.assertEqual(args.set_name, "peer_strong_63")


class DeteriorationModeIdentity(unittest.TestCase):
    def test_installed_policy_describes_haselton_anchors(self):
        with mock.patch.object(sp, "IMK_DETERIORATION_MODE", "haselton_2008"), \
                mock.patch.object(sp, "IMK_MATERIAL_TYPE", "IMKPeakOriented"):
            policy = installed_deterioration_policy()
            beam = IMK_Calibration.deterioration_for_member("beam_x", 0.0)
            column = IMK_Calibration.deterioration_for_member("column", 0.0)
        self.assertEqual(policy["mode"], "haselton_2008")
        self.assertEqual(policy["calibrated_modes"], ["S", "C"])
        self.assertEqual(policy["suppressed_modes"], {"A": 1.0e12, "K": 1.0e12})
        self.assertEqual(policy["anchors"]["beam"]["lamda_opensees_rad"], beam["lambda_opensees_rad"])
        self.assertEqual(policy["anchors"]["column_nu_0"]["lamda_opensees_rad"], column["lambda_opensees_rad"])
        with mock.patch.object(sp, "IMK_DETERIORATION_MODE", "direct"), \
                mock.patch.object(sp, "IMK_MATERIAL_TYPE", "IMKPeakOriented"):
            direct = installed_deterioration_policy()
        self.assertEqual(direct["mode"], "direct")
        self.assertEqual(direct["lamda"], {"S": sp.IMK_LAMBDA_S, "C": sp.IMK_LAMBDA_C,
                                           "K": sp.IMK_LAMBDA_K, "A": sp.IMK_LAMBDA_A})

    def test_identity_keys_cover_the_deterioration_mode(self):
        for key in ("imk_deterioration_mode", "imk_use_calibrated_backbone", "imk_installed_deterioration",
                    "imk_beam_theta_y", "imk_column_theta_y"):
            self.assertIn(key, OUTPUT_IDENTITY_KEYS)

    def test_output_identity_rejects_a_deterioration_mode_switch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "global_parameters.json").write_text(json.dumps(collect_global_parameters()),
                                                               encoding="utf-8")
            other = "direct" if sp.IMK_DETERIORATION_MODE == "haselton_2008" else "haselton_2008"
            with mock.patch.object(sp, "IMK_DETERIORATION_MODE", other):
                with self.assertRaisesRegex(RuntimeError, "imk_deterioration_mode"):
                    validate_ntha_output_compatibility(output_dir)
            with mock.patch.object(sp, "IMK_USE_CALIBRATED_BACKBONE", not sp.IMK_USE_CALIBRATED_BACKBONE):
                with self.assertRaisesRegex(RuntimeError, "imk_use_calibrated_backbone"):
                    validate_ntha_output_compatibility(output_dir)
            with mock.patch.object(sp, "IMK_BEAM_THETA_Y", sp.IMK_BEAM_THETA_Y * 2):
                with self.assertRaisesRegex(RuntimeError, "imk_installed_deterioration|imk_beam_theta_y"):
                    validate_ntha_output_compatibility(output_dir)


if __name__ == "__main__":
    unittest.main()
