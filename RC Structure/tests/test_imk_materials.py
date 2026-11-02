"""Modern IMK adapters and real 3D zeroLength fixtures; synthetic inputs only."""
import contextlib
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
import Structure_Parameters as sp
from Model import IMK_Hinges
from Model.IMK_Materials import CyclicParameters, RotationalBackbone, define_rotational_imk, ENERGY_CONVENTION
from Model.IMK_Spring_Mapping import FROZEN_V2, LEGACY_REPLAY

PROVENANCE = {"calibration_id": "synthetic_material_fixture_v1", "status": "verification_only",
              "deformation_scope": "isolated_rotational_spring; no_joint_physics_claim"}
POSITIVE = RotationalBackbone(0.01, 0.05, 0.09, 100.0, 1.10, 0.20)
NEGATIVE = RotationalBackbone(0.01, 0.05, 0.09, 70.0, 1.10, 0.20)


def fixture_cyclic(material_type):
    return CyclicParameters(10., 10., 10., 1., 1., 1., 1., 1.,
                            lamda_a=10. if material_type != "IMKBilin" else None,
                            c_a=1. if material_type != "IMKBilin" else None,
                            kappa_f=0.5 if material_type == "IMKPinching" else None,
                            kappa_d=0.5 if material_type == "IMKPinching" else None)


def rotation_history(peaks, step=0.0002):
    history = [0.0]
    for peak in peaks:
        n = max(1, int(np.ceil(abs(peak - history[-1]) / step)))
        history.extend(np.linspace(history[-1], peak, n + 1)[1:])
    return np.array(history)


def run_fixture(material_type, history, *, moment_scale=1.0, cyclic=None,
                orient=(1., 0., 0., 0., 1., 0.), local_dir=5, global_dof=5):
    """Displacement-controlled spring; return conjugate local response at commits."""
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0., 0., 0.)
        ops.node(2, 0., 0., 0.)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.equalDOF(1, 2, *(d for d in range(1, 7) if d != global_dof))
        metadata = define_rotational_imk(
            material_type, 11, 1e5 * moment_scale,
            replace(POSITIVE, fy=POSITIVE.fy * moment_scale),
            replace(NEGATIVE, fy=NEGATIVE.fy * moment_scale),
            cyclic or fixture_cyclic(material_type), provenance=PROVENANCE)
        ops.element("zeroLength", 21, 1, 2, "-mat", 11, "-dir", local_dir, "-orient", *orient)
        assert ops.eleNodes(21) == [1, 2]
        assert ops.nodeCoord(1) == ops.nodeCoord(2)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        load = [0.] * 6
        load[global_dof - 1] = 1.
        ops.load(2, *load)
        # Identity equalDOF constraints and homogeneous supports can be enforced
        # exactly here; avoid adding penalty-spring compliance to a material test.
        ops.constraints("Plain")
        ops.numberer("Plain")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-10, 50)
        ops.algorithm("Newton")
        ops.integrator("DisplacementControl", 2, global_dof, 1e-6)
        ops.analysis("Static")
        response = [[0., 0.]]
        for target in history[1:]:
            increment = float(target - ops.nodeDisp(2, global_dof))
            if abs(increment) < 1e-15:
                continue
            ops.integrator("DisplacementControl", 2, global_dof, increment)
            if ops.analyze(1) != 0:
                raise AssertionError(f"{material_type}: failed at rotation {target}")
            pair = ops.eleResponse(21, "material", 1, "stressStrain")
            # Check the actual element recordable pair, not only the constructor.
            np.testing.assert_allclose(pair, [ops.eleResponse(21, "basicForce")[0],
                                              ops.eleResponse(21, "deformation")[0]], rtol=1e-10, atol=1e-12)
            response.append(pair)
        return np.array(response), metadata
    finally:
        ops.wipe()


class ModernIMKTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_command_signatures_preserve_modern_energy_parameters(self):
        for material_type, argument_count in (("IMKBilin", 23), ("IMKPeakOriented", 25), ("IMKPinching", 27)):
            cyclic = fixture_cyclic(material_type)
            cyclic = replace(cyclic, lamda_s=2., lamda_c=3., lamda_k=5., c_s=0.8, c_c=0.9, c_k=1.2)
            if material_type != "IMKBilin":
                cyclic = replace(cyclic, lamda_a=4., c_a=1.1)
            with self.subTest(material_type=material_type), mock.patch.object(ops, "uniaxialMaterial") as call:
                metadata = define_rotational_imk(material_type, 11, 1e5, POSITIVE, NEGATIVE, cyclic, provenance=PROVENANCE)
                args = call.call_args.args
                self.assertEqual(len(args), argument_count)
                expected = (2., 3., 5., 0.8, 0.9, 1.2, 1., 1.) if material_type == "IMKBilin" else (
                    2., 3., 4., 5., 0.8, 0.9, 1.1, 1.2, 1., 1.)
                if material_type == "IMKPinching":
                    expected += (0.5, 0.5)
                self.assertEqual(args[15:], expected)  # no multiplication by My/Ke
                self.assertEqual(metadata["cyclic"]["energy_convention"], ENERGY_CONVENTION)
                json.dumps(metadata, allow_nan=False)

    def test_invalid_or_incomplete_calibration_is_rejected_before_installation(self):
        valid = fixture_cyclic("IMKPinching")
        invalid = [replace(valid, lamda_a=None), replace(valid, c_a=None), replace(valid, kappa_f=None),
                   replace(valid, kappa_d=1.), replace(valid, lamda_s=0.), replace(valid, lamda_k=float("nan")),
                   replace(valid, lamda_c=float("inf")), replace(valid, energy_convention="gamma_Fy_theta_y")]
        with mock.patch.object(ops, "uniaxialMaterial") as call:
            for cyclic in invalid:
                with self.subTest(cyclic=cyclic), self.assertRaises(ValueError):
                    define_rotational_imk("IMKPinching", 1, 1e5, POSITIVE, NEGATIVE, cyclic, provenance=PROVENANCE)
            call.assert_not_called()

    def test_parameter_identity_is_tag_independent_but_tracks_calibration(self):
        with mock.patch.object(ops, "uniaxialMaterial"):
            kwargs = dict(provenance=dict(PROVENANCE))
            first = define_rotational_imk("IMKPeakOriented", 1, 1e5, POSITIVE, NEGATIVE,
                                         fixture_cyclic("IMKPeakOriented"), **kwargs)
            second = define_rotational_imk("IMKPeakOriented", 2, 1e5, POSITIVE, NEGATIVE,
                                          fixture_cyclic("IMKPeakOriented"), **kwargs)
            changed = define_rotational_imk("IMKPeakOriented", 3, 1e5, POSITIVE, NEGATIVE,
                                           replace(fixture_cyclic("IMKPeakOriented"), lamda_a=9.), **kwargs)
            kwargs["provenance"]["calibration_id"] = "changed_after_installation"
        self.assertEqual(first["parameter_sha256"], second["parameter_sha256"])
        self.assertNotEqual(first["parameter_sha256"], changed["parameter_sha256"])
        self.assertEqual(first["provenance"]["calibration_id"], PROVENANCE["calibration_id"])

    def test_real_monotonic_response_matches_both_backbone_signs(self):
        for material_type in ("IMKPeakOriented", "IMKPinching"):
            for sign, branch in ((1., POSITIVE), (-1., NEGATIVE)):
                dy = branch.fy / 1e5
                cap = dy + branch.dp
                for deformation, expected in ((dy / 2, branch.fy / 2), (cap, branch.fy * branch.fmax_fy),
                                              (cap + branch.dpc / 2, branch.fy * branch.fmax_fy / 2)):
                    with self.subTest(material=material_type, sign=sign, deformation=deformation):
                        response, _ = run_fixture(material_type, rotation_history([sign * deformation]))
                        self.assertAlmostEqual(response[-1, 0], sign * expected, delta=1e-6)

    def test_real_cyclic_response_is_finite_and_force_unit_invariant(self):
        history = rotation_history([.02, -.02, .03, -.03, .02, -.02, 0.])
        for material_type in ("IMKPeakOriented", "IMKPinching"):
            with self.subTest(material=material_type):
                response, _ = run_fixture(material_type, history)
                scaled, _ = run_fixture(material_type, history, moment_scale=7.3)
                self.assertTrue(np.isfinite(response).all())
                self.assertGreater(response[:, 0].max(), POSITIVE.fy)
                self.assertLess(response[:, 0].min(), -NEGATIVE.fy)
                np.testing.assert_allclose(scaled[:, 0] / 7.3, response[:, 0], rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(scaled[:, 1], response[:, 1], atol=1e-12)
                work = np.sum(np.diff(response[:, 1]) * (response[1:, 0] + response[:-1, 0]) / 2)
                self.assertGreater(work, 0.)

    def test_three_dimensional_axes_match_member_orientations(self):
        for kind, axes in (("column", ((5, 4, 1), (6, 5, 1))),
                           ("beam_x", ((5, 5, 1), (6, 6, 1))),
                           ("beam_y", ((5, 4, -1), (6, 6, 1)))):
            orient, _ = IMK_Hinges._orientation(kind)
            for local_dir, global_dof, direction_sign in axes:
                for material_type in ("IMKPeakOriented", "IMKPinching"):
                    with self.subTest(kind=kind, local_dir=local_dir, material=material_type):
                        response, _ = run_fixture(material_type, rotation_history([.0002]),
                                                  orient=orient, local_dir=local_dir, global_dof=global_dof)
                        self.assertAlmostEqual(response[-1, 1], direction_sign * .0002, places=12)
                        self.assertAlmostEqual(response[-1, 0], direction_sign * 20., places=8)

    def test_member_adapter_preserves_asymmetric_limits_and_rejects_joint_material(self):
        with contextlib.ExitStack() as stack:
            # A direct low-level call with the global IMK_THETA_* limits installed as native values: the pre-freeze
            # rule, kept as the explicit legacy replay (frozen IMK V2 calibration, 2026-10-29).
            for name, value in {"IMK_MATERIAL_TYPE": "IMKPeakOriented", "IMK_LAMBDA_A": 10., "IMK_C_A": 1.,
                                "IMK_THETA_P_NEG": .031, "IMK_THETA_PC_NEG": .071, "IMK_THETA_U_NEG": .151,
                                "IMK_MEMBER_SPRING_CALIBRATION": LEGACY_REPLAY}.items():
                stack.enter_context(mock.patch.object(sp, name, value))
            call = stack.enter_context(mock.patch.object(ops, "uniaxialMaterial"))
            metadata = IMK_Hinges._define_imk_peak_material(1, 1e5, 100., yield_moment_negative=70.)
            self.assertEqual(call.call_args.args[9:15], (.031, .071, .151, 70., 1.1, .2))
            self.assertIn("not_validated", metadata["provenance"]["deformation_scope"])
            sp.IMK_MATERIAL_TYPE = "IMKPinching"
            with self.assertRaisesRegex(ValueError, "joint"):
                IMK_Hinges._define_imk_peak_material(2, 1e5, 100.)

    def test_frozen_calibration_refuses_the_direct_low_level_member_material(self):
        # Frozen IMK V2 calibration (2026-10-29): only the canonical path with a Haselton backbone built under it.
        with mock.patch.object(sp, "IMK_MEMBER_SPRING_CALIBRATION", FROZEN_V2), \
                mock.patch.object(ops, "uniaxialMaterial") as call:
            with self.assertRaisesRegex(ValueError, FROZEN_V2):
                IMK_Hinges._define_imk_peak_material(1, 1e5, 100., yield_moment_negative=70.)
            call.assert_not_called()

    def test_modern_diagnostics_do_not_claim_bilin_plastic_rotation(self):
        from Analysis import Hinge_Hysteresis_Diagnostic as hd
        for material_type in ("IMKPeakOriented", "IMKPinching"):
            response, metadata = run_fixture(material_type, rotation_history([.02, -.02, 0.]))
            hinge = {"hinge_tag": 21, "member_tag": 1, "member_type": "beam_x", "end": "i", "joint_node": 1,
                     "material_type": material_type, "theta_p": .01, "theta_pc": .05, "theta_u": .09,
                     "installed_materials": {"y": metadata}, "springs": {"y": {"ke": 1e5,
                         "fy_positive_kip_in": 100., "fy_negative_kip_in": 70.,
                         "theta_y_positive": .001, "theta_y_negative": .0007}}}
            recorded = {"hinge_tag_order": [21], "y": {"time": np.arange(len(response), dtype=float),
                        "moment": response[:, :1], "rotation": response[:, 1:], "rows": len(response)}}
            row = hd.evaluate_hinge_histories(recorded, [hinge])[0]
            self.assertTrue(row["yielded_positive"] and row["yielded_negative"])
            self.assertIsNone(row["plastic_rotation_accumulated"])
            self.assertIsNone(row["criteria_disagree"])
            summary = hd.yield_summary([row])
            self.assertIsNone(summary["max_plastic_rotation_accumulated"])
            self.assertEqual(summary["springs_with_accumulated_plastic_rotation"], 0)
            self.assertIn("unavailable", hd._loop_title(row))
            json.dumps({"row": row, "summary": summary}, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
