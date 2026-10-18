"""Perimeter-frame basis (2026-10-16): which members are SFRS and which are gravity, and the ACI 318-19 18.14 rules."""
import math
import unittest

from Design import Gravity_Frame_Members as gm
from Model import Member_Groups as mg


class MemberSystem(unittest.TestCase):
    def test_perimeter_columns_and_beams_are_sfrs_and_the_interior_is_gravity(self):
        nx, ny = 3, 4
        self.assertEqual(mg.member_system("column", 0, 0, nx, ny), mg.SYSTEM_SFRS)          # corner
        self.assertEqual(mg.member_system("column", 0, 2, nx, ny), mg.SYSTEM_SFRS)          # x-boundary edge
        self.assertEqual(mg.member_system("column", 1, 4, nx, ny), mg.SYSTEM_SFRS)          # y-boundary edge
        self.assertEqual(mg.member_system("column", 1, 2, nx, ny), mg.SYSTEM_GRAVITY)       # interior
        self.assertEqual(mg.member_system("beam_x", 1, 0, nx, ny), mg.SYSTEM_SFRS)          # perimeter x-frame
        self.assertEqual(mg.member_system("beam_x", 1, 2, nx, ny), mg.SYSTEM_GRAVITY)       # interior x line
        self.assertEqual(mg.member_system("beam_y", 3, 1, nx, ny), mg.SYSTEM_SFRS)          # perimeter y-frame
        self.assertEqual(mg.member_system("beam_y", 2, 1, nx, ny), mg.SYSTEM_GRAVITY)

    def test_sfrs_directions_of_a_column(self):
        self.assertEqual(mg.column_sfrs_directions(0, 0, 3, 4), frozenset({"x", "y"}))      # corner: both frames
        self.assertEqual(mg.column_sfrs_directions(0, 2, 3, 4), frozenset({"y"}))           # on x = 0: a frame along y
        self.assertEqual(mg.column_sfrs_directions(2, 4, 3, 4), frozenset({"x"}))           # on y = Ly: a frame along x
        self.assertEqual(mg.column_sfrs_directions(1, 2, 3, 4), frozenset())

    def test_inventory_counts_every_member_once(self):
        counts = mg.system_inventory(3, 4)
        total = {kind: counts[mg.SYSTEM_SFRS][kind] + counts[mg.SYSTEM_GRAVITY][kind] for kind in ("column", "beam_x", "beam_y")}
        self.assertEqual(total, {"column": 20, "beam_x": 15, "beam_y": 16})
        self.assertEqual(counts[mg.SYSTEM_SFRS], {"column": 14, "beam_x": 6, "beam_y": 8})
        self.assertEqual(counts[mg.SYSTEM_GRAVITY], {"column": 6, "beam_x": 9, "beam_y": 8})


class Branch(unittest.TestCase):
    def test_branch_by_induced_effects(self):
        within = gm.compatibility_branch(900.0, 20.0, 1000.0, 30.0)
        self.assertEqual(within["branch"], gm.BRANCH_WITHIN_STRENGTH)
        self.assertAlmostEqual(within["moment_ratio"], 0.9)
        over = gm.compatibility_branch(1100.0, 20.0, 1000.0, 30.0)
        self.assertEqual(over["branch"], gm.BRANCH_YIELDING)
        self.assertIn("moment", over["reason"])
        shear = gm.compatibility_branch(-500.0, -31.0, 1000.0, 30.0)
        self.assertEqual(shear["branch"], gm.BRANCH_YIELDING)
        self.assertIn("shear", shear["reason"])
        self.assertEqual(gm.compatibility_branch(0.0, 0.0, 1000.0, 30.0, calculated=False)["branch"], gm.BRANCH_YIELDING)
        with self.assertRaises(ValueError):
            gm.compatibility_branch(1.0, 1.0, 0.0, 30.0)

    def test_hoop_spacing_limit_and_po(self):
        self.assertEqual(gm.hoop_spacing_limit_in(0.75), 4.5)                             # 6 db governs for #6
        self.assertEqual(gm.hoop_spacing_limit_in(1.128), 6.0)                            # 6 in governs for #9
        po = gm.nominal_axial_strength_kip(400.0, 8.0, 6.0, 60.0)
        self.assertAlmostEqual(po, 0.85 * 6.0 * 392.0 + 60.0 * 8.0)


class BeamRules(unittest.TestCase):
    def test_within_strength_beam_needs_d_over_2_stirrups_and_no_capacity_shear(self):
        need = gm.beam_requirements(gm.BRANCH_WITHIN_STRENGTH, d_in=21.0, smallest_bar_db_in=0.875, pu_kip=10.0, ag_in2=280.0, fc_ksi=5.0)
        self.assertEqual(need["transverse"]["spacing_max_in"], 10.5)
        self.assertFalse(need["axial_hoops"])
        self.assertFalse(need["capacity_shear_required"])
        self.assertIn("Chapter 15", need["joints"])

    def test_axial_force_above_ag_fc_over_10_asks_for_hoops(self):
        need = gm.beam_requirements(gm.BRANCH_WITHIN_STRENGTH, d_in=21.0, smallest_bar_db_in=0.875, pu_kip=150.0, ag_in2=280.0, fc_ksi=5.0)
        self.assertTrue(need["axial_hoops"])
        self.assertEqual(need["transverse"]["spacing_max_in"], 5.25)
        self.assertEqual(need["transverse"]["kind"], "hoops per 18.7.5.2")

    def test_yielding_beam_adds_capacity_shear_and_the_intermediate_joint_rule(self):
        need = gm.beam_requirements(gm.BRANCH_YIELDING, d_in=21.0, smallest_bar_db_in=1.0)
        self.assertTrue(need["capacity_shear_required"])
        self.assertIn("18.4.4.1", need["joints"])
        self.assertEqual(need["transverse"]["spacing_max_in"], 10.5)

    def test_evaluate_gravity_beam_checks(self):
        section = {"b_in": 14.0, "d_in": 21.0, "ag_in2": 14.0 * 24.0, "fc_ksi": 5.0, "fy_ksi": 60.0,
                   "as_top_in2": 1.8, "as_bot_in2": 1.2, "pu_kip": 0.0}
        checks = gm.evaluate_gravity_beam("x interior", {"mu_kip_in": 900.0, "vu_kip": 20.0}, {"phi_mn_kip_in": 1000.0, "phi_vn_kip": 30.0},
                                          section, {"stirrup_spacing_in": 10.0, "smallest_bar_db_in": 0.875, "hoops": False})
        by_id = {(c["id"], c["location"]): c for c in checks}
        self.assertEqual(by_id[("gravity.beam_compatibility_branch", "x interior")]["details"]["branch"], gm.BRANCH_WITHIN_STRENGTH)
        self.assertEqual(by_id[("gravity.beam_transverse_spacing", "x interior")]["status"], "pass")
        self.assertEqual(by_id[("gravity.beam_minimum_steel", "x interior top")]["status"], "pass")
        self.assertNotIn(("gravity.beam_capacity_shear", "x interior"), by_id)
        yielding = gm.evaluate_gravity_beam("x interior", {"mu_kip_in": 1200.0, "vu_kip": 20.0}, {"phi_mn_kip_in": 1000.0, "phi_vn_kip": 30.0},
                                            section, {"stirrup_spacing_in": 12.0, "smallest_bar_db_in": 0.875, "hoops": False})
        by_id = {c["id"]: c for c in yielding}
        self.assertEqual(by_id["gravity.beam_transverse_spacing"]["status"], "fail")           # 12 > d/2 = 10.5
        self.assertEqual(by_id["gravity.beam_capacity_shear"]["status"], "not_evaluated")       # Ve not supplied


class ColumnRules(unittest.TestCase):
    def test_low_axial_within_strength_column(self):
        need = gm.column_requirements(gm.BRANCH_WITHIN_STRENGTH, 1.0, pu_kip=200.0, po_kip=2000.0, ash_ratio_table_18_7_5_4=0.012)
        self.assertFalse(need["high_axial"])
        self.assertEqual(need["hoop_spacing_full_length_max_in"], 6.0)
        self.assertEqual(need["ash_fraction_of_table_18_7_5_4"], 0.0)
        self.assertEqual(need["ash_ratio_required"], 0.0)
        self.assertFalse(need["bond_18_7_4_3_required"])

    def test_high_axial_column_provides_half_the_table(self):
        need = gm.column_requirements(gm.BRANCH_WITHIN_STRENGTH, 0.75, pu_kip=800.0, po_kip=2000.0, ash_ratio_table_18_7_5_4=0.012)
        self.assertTrue(need["high_axial"])
        self.assertEqual(need["hoop_spacing_full_length_max_in"], 4.5)
        self.assertAlmostEqual(need["ash_ratio_required"], 0.006)

    def test_yielding_column_provides_the_whole_table(self):
        need = gm.column_requirements(gm.BRANCH_YIELDING, 1.0, pu_kip=100.0, po_kip=2000.0, ash_ratio_table_18_7_5_4=0.012)
        self.assertAlmostEqual(need["ash_ratio_required"], 0.012)
        self.assertIn("18.4.4.1", need["joints"])

    def test_evaluate_gravity_column_checks(self):
        section = {"ag_in2": 400.0, "ast_in2": 6.0, "fc_ksi": 6.0, "fy_ksi": 60.0, "pu_kip": 900.0}
        detailing = {"hoop_spacing_in": 4.0, "smallest_bar_db_in": 0.875, "ash_ratio_provided": 0.007,
                     "ash_ratio_table_18_7_5_4": 0.012, "configuration_18_7_5_2_ok": True}
        checks = gm.evaluate_gravity_column("interior", {"mu_kip_in": 500.0, "vu_kip": 10.0},
                                            {"phi_mn_kip_in": 1500.0, "phi_vn_kip": 40.0, "ve_kip": 25.0, "phi_vn_capacity_kip": 40.0},
                                            section, detailing)
        by_id = {c["id"]: c for c in checks}
        self.assertEqual(by_id["gravity.column_compatibility_branch"]["details"]["branch"], gm.BRANCH_WITHIN_STRENGTH)
        self.assertTrue(by_id["gravity.column_compatibility_branch"]["details"]["requirements"]["high_axial"])   # 900 > 0.35 Po
        self.assertEqual(by_id["gravity.column_hoop_spacing_full_length"]["status"], "pass")                     # 4.0 <= 5.25
        self.assertEqual(by_id["gravity.column_confinement_lo"]["status"], "pass")                               # 0.006 <= 0.007
        self.assertEqual(by_id["gravity.column_capacity_shear"]["status"], "pass")
        self.assertEqual(by_id["gravity.column_minimum_rho"]["status"], "pass")
        light = gm.evaluate_gravity_column("interior", {"mu_kip_in": 500.0, "vu_kip": 10.0}, {"phi_mn_kip_in": 1500.0, "phi_vn_kip": 40.0},
                                           {**section, "pu_kip": 200.0}, {**detailing, "hoop_spacing_in": 6.0})
        by_id = {c["id"]: c for c in light}
        self.assertNotIn("gravity.column_confinement_lo", by_id)                                                  # configuration only
        self.assertEqual(by_id["gravity.column_hoop_spacing_full_length"]["status"], "fail")                     # 6.0 > 5.25
        self.assertEqual(by_id["gravity.column_capacity_shear"]["status"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
