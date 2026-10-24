"""The design period basis of ASCE 7-22 12.8.2 (2026-10-24).

Cu, the coefficient for the upper limit on the calculated period, is read from Table 12.8-1 at the site's SD1 (a
fixed 1.4 was used before), and the calculated period is the entire structure's, every beam and column of the
continuous frame, under every grouping policy: it is not separated by system (before, the two-system route took it
from the released perimeter-SFRS model). Torsion, strength and drift solves keep their own lateral design model.
"""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Design import Design_Driver as driver                                       # noqa: E402
from Design import Perimeter_Systems as systems                                  # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Loads import Seismic_ELF as elf                                             # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

TABLE = ((0.4, 1.4), (0.3, 1.4), (0.2, 1.5), (0.15, 1.6), (0.1, 1.7))          # ASCE 7-22 Table 12.8-1, by hand


class Table12_8_1(unittest.TestCase):
    def test_the_table_is_the_code_table(self):
        self.assertEqual(sp.ASCE_CU_TABLE_12_8_1, TABLE)
        self.assertFalse(hasattr(sp, "ASCE_CU"))                               # the fixed value is gone, not shadowed

    def test_rows_bounds_and_linear_interpolation_between_rows(self):
        for sd1, cu in ((2.0, 1.4), (0.4, 1.4), (0.38, 1.4), (0.3, 1.4), (0.2, 1.5), (0.15, 1.6), (0.1, 1.7), (0.0, 1.7)):
            with self.subTest(sd1=sd1):
                self.assertAlmostEqual(sp.asce_cu(sd1), cu, places=12)
        self.assertAlmostEqual(sp.asce_cu(0.19), 1.5 + (1.6 - 1.5) * (0.2 - 0.19) / (0.2 - 0.15), places=12)  # 1.52
        self.assertAlmostEqual(sp.asce_cu(0.25), 1.45, places=12)
        self.assertAlmostEqual(sp.asce_cu(0.12), 1.66, places=12)
        self.assertAlmostEqual(sp.asce_cu(0.1 + 0.05), 1.6, places=12)                                         # float row
        self.assertIn("interpolated", sp.asce_cu_table_12_8_1(0.19)[1])
        self.assertNotIn("interpolated", sp.asce_cu_table_12_8_1(0.35)[1])                                     # 1.4 both rows
        with self.assertRaises(ValueError):
            sp.asce_cu(math.nan)

    def test_cu_follows_the_configured_site(self):
        expected = {"sdc_c": 1.52, "sdc_d_low": 1.4, "sdc_d_high": 1.4, "sdc_e": 1.4, "sdc_e_near": 1.4}
        saved = sp.SEISMIC_SITE_LABEL
        try:
            for label, _sds, sd1, _s1 in sp.SEISMIC_SITE_OPTIONS:
                with self.subTest(site=label):
                    sp.apply_seismic_site(label)
                    self.assertAlmostEqual(sp.asce_cu(), expected[label], places=12)
                    self.assertEqual(sp.asce_cu(), sp.asce_cu(sd1))
        finally:
            sp.apply_seismic_site(saved)


class PeriodCap(unittest.TestCase):
    GEOMETRY = {"NUM_FLOOR": 5, "STORY_H": 168.0}                               # hn = 70 ft, Ta = 0.016 x 70^0.9

    def test_the_cap_is_table_cu_times_ta(self):
        ta = 0.016 * 70.0 ** 0.9
        for sd1, cu in ((0.19, 1.52), (0.6, 1.4)):
            with self.subTest(sd1=sd1), mock.patch.multiple(sp, ASCE_SD1=sd1, **self.GEOMETRY):
                self.assertEqual(elf.design_period_sec(5.0), (cu * ta, ta, True))               # capped at Cu Ta
                period, _ta, capped = elf.design_period_sec(0.9 * cu * ta)
                self.assertAlmostEqual(period, 0.9 * cu * ta, places=12)
                self.assertFalse(capped)
                self.assertEqual(elf.design_period_sec(None), (ta, ta, False))                  # Ta used directly

    def test_the_default_building_cap_moves_from_1_4_to_1_52(self):
        with mock.patch.multiple(sp, ASCE_SD1=0.19, **self.GEOMETRY):
            period, ta, capped = elf.design_period_sec(1.519)                                   # the 2026-10-23 record
        self.assertTrue(capped)
        self.assertAlmostEqual(period, 1.52 * ta, places=12)                                    # 1.113 s
        self.assertGreater(period, 1.4 * ta)                                                    # 1.025 s before


class PeriodOfTheEntireStructure(unittest.TestCase):
    def test_the_period_solve_builds_the_continuous_frame_and_restores_the_design_model(self):
        seen = []

        def build():
            seen.append(sp.LATERAL_DESIGN_MODEL)

        modes = [{"valid": True, "period": 0.8}]
        with mock.patch.object(driver, "build_model", side_effect=build), \
                mock.patch.object(driver, "run_modal_analysis", return_value=modes), \
                mock.patch.object(driver.ops, "wipe"), mock.patch.object(sp, "LATERAL_DESIGN_MODEL", "perimeter_sfrs"):
            self.assertEqual(driver._model_period(), 0.8)
            self.assertEqual(seen, ["all_frames"])
            self.assertEqual(sp.LATERAL_DESIGN_MODEL, "perimeter_sfrs")                         # restored
            with mock.patch.object(driver, "run_modal_analysis", side_effect=RuntimeError("modal failed")):
                with self.assertRaises(RuntimeError):
                    driver._model_period()
            self.assertEqual(sp.LATERAL_DESIGN_MODEL, "perimeter_sfrs")                         # restored on failure

    def test_the_two_system_period_is_that_of_every_frame(self):
        with gf.frame():
            state = mg.GroupedDesign.uniform(gf.NX, gf.NY, gf.NZ, gf.COLUMN, gf.BEAM, policy_id=mg.SYSTEM_POLICY_ID)
            mg.install(state)
            self.assertEqual(systems.design_model_for(state), "perimeter_sfrs")                 # torsion, strength, drift
            period = systems.model_period(state)
            with systems.lateral_model("all_frames"):
                continuous = driver._model_period()
            with mock.patch.object(driver, "PERIOD_MODEL", "perimeter_sfrs"):
                released = driver._model_period()                                               # the earlier basis
            self.assertEqual(sp.LATERAL_DESIGN_MODEL, "all_frames")
        self.assertAlmostEqual(period, continuous, places=6)                 # two eigen solves of one model
        self.assertLess(period, released)                    # the gravity frames stiffen the entire structure


class RecordedBasis(unittest.TestCase):
    def test_the_request_identity_names_the_table_its_value_and_the_period_model(self):
        with mock.patch.multiple(sp, ASCE_SD1=0.19):                           # SDC C: the table's Cu is not 1.4
            inputs = driver.design_request_identity()["inputs"]
        self.assertAlmostEqual(inputs["ASCE_CU"], 1.52, places=12)
        self.assertEqual([list(row) for row in sp.ASCE_CU_TABLE_12_8_1], inputs["ASCE_CU_TABLE_12_8_1"])
        self.assertEqual(inputs["ASCE_CU_RULE"], sp.ASCE_CU_RULE)
        self.assertEqual(inputs["PERIOD_MODEL"], "all_frames")

    def test_the_demand_basis_records_cu_and_the_period_model(self):
        with mock.patch.multiple(sp, ASCE_SD1=0.19, NUM_FLOOR=5, STORY_H=168.0):
            forces = elf.elf_story_forces(1.519)                                # the producer of cu and cu_rule
            self.assertTrue(forces["period_capped_at_cu_ta"])
            self.assertAlmostEqual(forces["design_period_sec"], 1.52 * forces["asce_ta_sec"], places=12)
            basis = driver._demand_basis(DesignConfig(), None, forces, {"assumptions": {}},
                                         strength={"one_side_fraction": {}}, regularity={})
        period = basis["period_basis"]
        self.assertAlmostEqual(period["cu"], 1.52, places=12)
        self.assertEqual(period["cu_rule"], sp.ASCE_CU_RULE)
        self.assertEqual(period["period_model"], "all_frames")
        self.assertIn("entire structure", period["period_model_basis"])


if __name__ == "__main__":
    unittest.main()
