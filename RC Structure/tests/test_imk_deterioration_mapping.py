"""Haselton (2008) Eq. 3.20 deterioration translated into OpenSees Lamda.

Pins the arithmetic (Lamda = lambda_Haselton x theta_y,member for S and C,
a large finite capacity for A and K), the command slots the values land in
for both member materials, and the provenance recorded. Not an experimental
calibration: the status the material records stays provisional.
"""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Model import IMK_Hinges
from Model.IMK_Calibration import deterioration_for_member

BACKBONE = {"theta_p": .02, "theta_pc": .06, "theta_u": .12, "source": "haselton_2008",
            "deterioration_source": "haselton_2008_eq3_20_nominal_member_theta_y",
            "lambda_haselton_dimensionless": 120., "energy_reference_member_theta_y_rad": .005,
            "lambda_opensees_rad": .6, "lambda_opensees_by_mode_rad": {"S": .6, "C": .6, "A": 1e12, "K": 1e12},
            "deterioration_suppressed_modes": ["A", "K"]}


class HaseltonTranslationTests(unittest.TestCase):
    def test_column_lambda_is_eq_3_20_times_the_member_yield_rotation(self):
        with mock.patch.multiple(sp, IMK_DETERIORATION_MODE="haselton_2008", COL_STIRRUP_SPACING=4., H_COL=24.,
                                 IMK_COLUMN_THETA_Y=.004, BEAM_STIRRUP_SPACING=3., H_BEAM=20., IMK_BEAM_THETA_Y=.005):
            d = deterioration_for_member("column", .2)
            # Eq. 3.20 takes s/d with the effective depth d = h - d' (2026-10-23; s/h before)
            lam = 170.7 * .27 ** .2 * .10 ** (4. / (24. - sp.longitudinal_cover_in("column")))
            self.assertAlmostEqual(d["lambda_haselton_dimensionless"], lam, places=9)
            self.assertAlmostEqual(d["lambda_opensees_rad"], lam * .004, places=9)
            self.assertEqual(d["energy_reference_member_theta_y_rad"], .004)
            self.assertAlmostEqual(d["lambda_opensees_by_mode_rad"]["S"], lam * .004, places=9)
            self.assertAlmostEqual(d["lambda_opensees_by_mode_rad"]["C"], lam * .004, places=9)
            self.assertEqual(d["lambda_opensees_by_mode_rad"]["A"], 1e12)
            self.assertEqual(d["lambda_opensees_by_mode_rad"]["K"], 1e12)
            self.assertEqual(d["deterioration_suppressed_modes"], ["A", "K"])
            self.assertFalse(d["deterioration_beam_extrapolation"])
            self.assertFalse(d["deterioration_axial_ratio_clamped"])
            beam = deterioration_for_member("beam_x", 0.)
            self.assertAlmostEqual(beam["lambda_opensees_rad"],
                                   170.7 * .10 ** (3. / (20. - sp.longitudinal_cover_in("beam"))) * .005, places=9)
            self.assertTrue(beam["deterioration_beam_extrapolation"])
            self.assertTrue(deterioration_for_member("column", 2.)["deterioration_axial_ratio_clamped"])
        with mock.patch.object(sp, "IMK_DETERIORATION_MODE", "direct"):
            self.assertEqual(deterioration_for_member("column", .2), {"deterioration_source": "direct_opensees"})
        with mock.patch.object(sp, "IMK_DETERIORATION_MODE", "haselton_2009"), self.assertRaises(ValueError):
            deterioration_for_member("column", .2)

    def test_translated_lambdas_land_in_the_command_slots_of_both_materials(self):
        cases = {"IMKPeakOriented": (.6, .6, 1e12, 1e12, sp.IMK_C_S, sp.IMK_C_C, sp.IMK_C_A, sp.IMK_C_K),
                 "IMKBilin": (.6, .6, 1e12, sp.IMK_C_S, sp.IMK_C_C, sp.IMK_C_K)}
        for material, cyclic in cases.items():
            with self.subTest(material=material), mock.patch.object(sp, "IMK_MATERIAL_TYPE", material), \
                    mock.patch.object(IMK_Hinges.ops, "uniaxialMaterial") as install:
                meta = IMK_Hinges._define_imk_peak_material(7, 5000., 100., dict(BACKBONE), 80.)
                args = install.call_args.args
                self.assertEqual(args[:3], (material, 7, 5000.))
                self.assertEqual(args[3:9], (.02, .06, .12, 100., sp.IMK_FMAXFY_POS, sp.IMK_FRESFY_POS))
                self.assertEqual(args[9:15], (.02, .06, .12, 80., sp.IMK_FMAXFY_NEG, sp.IMK_FRESFY_NEG))
                self.assertEqual(args[15:15 + len(cyclic)], cyclic)
                self.assertEqual(args[15 + len(cyclic):], (sp.IMK_D_POS, sp.IMK_D_NEG))
                self.assertEqual(meta["cyclic"]["lamda_s"], .6)
                self.assertEqual(meta["provenance"]["deterioration_source"], BACKBONE["deterioration_source"])
                self.assertEqual(meta["provenance"]["calibration_id"], BACKBONE["deterioration_source"] + "_v1")
                self.assertEqual(meta["provenance"]["status"], sp.IMK_CYCLIC_CALIBRATION_STATUS)
                self.assertEqual(meta["provenance"]["deterioration"]["lambda_opensees_rad"], .6)
                self.assertEqual(meta["reference_energies_kip_in_rad"]["S"], 60.)

    def test_without_translated_lambdas_the_constants_pass_unchanged(self):
        with mock.patch.object(sp, "IMK_MATERIAL_TYPE", "IMKPeakOriented"), \
                mock.patch.object(IMK_Hinges.ops, "uniaxialMaterial") as install:
            meta = IMK_Hinges._define_imk_peak_material(8, 5000., 100.,
                                                        {"theta_p": .02, "theta_pc": .06, "theta_u": .12, "source": "fixed"})
            self.assertEqual(install.call_args.args[15:19], (sp.IMK_LAMBDA_S, sp.IMK_LAMBDA_C, sp.IMK_LAMBDA_A, sp.IMK_LAMBDA_K))
            self.assertEqual(meta["provenance"]["deterioration_source"], "direct_opensees")
            self.assertEqual(meta["provenance"]["calibration_id"], sp.IMK_CYCLIC_CALIBRATION_ID)

    def test_ultimate_rotation_must_clear_the_capping_rotation(self):
        with mock.patch.object(sp, "IMK_MATERIAL_TYPE", "IMKPeakOriented"), \
                mock.patch.object(IMK_Hinges.ops, "uniaxialMaterial"), self.assertRaisesRegex(ValueError, "capping"):
            IMK_Hinges._define_imk_peak_material(9, 5000., 100., {"theta_p": .10, "theta_pc": .01, "theta_u": .105, "source": "fixed"})


if __name__ == "__main__":
    unittest.main()
