"""Transverse steel of beams outside the seismic force-resisting system (ACI 318-19 18.14.3.2(a), design decision of
2026-10-18): closed stirrups throughout the length at not more than d/2 and 24 in, tightened to d/4 and 12 in where Vs
exceeds 4 sqrt(f'c) bw d (Table 9.7.6.2.2); the 18.6.5 shear is kept. Perimeter-frame beams keep the 18.6.4.4
hinge-zone hoops."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Design import Gravity_Frame_Members as gm
from Design import SMRF_Capacity_Design as cd


class GravityBeamTransverseTests(unittest.TestCase):
    def setUp(self):
        self.saved = sp.FY_KSI, sp.REINFORCEMENT_SPECIFICATION
        sp.apply_reinforcement_grade(fy_ksi=60.0)

    def tearDown(self):
        sp.FY_KSI, sp.REINFORCEMENT_SPECIFICATION = self.saved

    def test_gravity_rule_bounds_the_spacing_at_half_the_depth(self):
        """14 x 26 beam, No. 7 bars, light Ve: the SFRS rule gives min(d/4, 6 db, 6 in), the gravity rule d/2."""
        args = (14.0, 23.5, 5.0, 60.0, 7, 1.5, 2, 2, 40.0, 10.0)
        sfrs = cd.select_beam_hoops(*args)
        gravity = cd.select_beam_hoops(*args, transverse_rule=cd.BEAM_TRANSVERSE_GRAVITY)
        self.assertEqual(sfrs["transverse_rule"], cd.BEAM_TRANSVERSE_SFRS)
        self.assertEqual(set(sfrs["spacing_bounds_in"]), {"d_over_4", "bar_diameters", "absolute"})
        self.assertEqual(gravity["spacing_bounds_in"], {"d_over_2": 11.75, "absolute": 24.0})
        self.assertGreater(gravity["hoops"]["spacing_in"], sfrs["hoops"]["spacing_in"])
        self.assertLessEqual(gravity["hoops"]["spacing_in"], 11.75)
        self.assertIn("18.14.3.2(a)", gravity["basis"])
        self.assertIn("stirrups", gravity["transverse_kind"])
        # the 18.6.5 shear arithmetic is the same under both rules
        for key in ("ve_kip", "vc_kip", "vs_required_kip", "vs_limit_kip"):
            self.assertEqual(sfrs[key], gravity[key])
        self.assertAlmostEqual(gravity["vs_four_root_fc_kip"], 0.5 * gravity["vs_limit_kip"])

    def test_high_shear_tightens_the_gravity_spacing_to_a_quarter_of_the_depth(self):
        """Vs above 4 sqrt(f'c) bw d takes the Table 9.7.6.2.2 row d/4 and 12 in."""
        heavy = cd.select_beam_hoops(14.0, 23.5, 5.0, 60.0, 7, 1.5, 2, 2, 220.0, 150.0, transverse_rule=cd.BEAM_TRANSVERSE_GRAVITY)
        self.assertGreater(heavy["vs_required_kip"], heavy["vs_four_root_fc_kip"])
        self.assertEqual(heavy["spacing_bounds_in"], {"d_over_4": 5.875, "absolute": 12.0})
        if heavy["hoops"]:
            self.assertLessEqual(heavy["hoops"]["spacing_in"], 5.875)

    def test_an_unknown_rule_is_refused(self):
        with self.assertRaises(ValueError):
            cd.select_beam_hoops(14.0, 23.5, 5.0, 60.0, 7, 1.5, 2, 2, 40.0, 10.0, transverse_rule="hoops_everywhere")

    def test_beam_requirements_follow_the_shear_row(self):
        low = gm.beam_requirements(gm.BRANCH_YIELDING, d_in=23.5, smallest_bar_db_in=0.875, vs_required_kip=40.0, vs_four_root_fc_kip=93.0)
        high = gm.beam_requirements(gm.BRANCH_YIELDING, d_in=23.5, smallest_bar_db_in=0.875, vs_required_kip=120.0, vs_four_root_fc_kip=93.0)
        deep = gm.beam_requirements(gm.BRANCH_WITHIN_STRENGTH, d_in=60.0, smallest_bar_db_in=1.0)
        self.assertEqual(low["transverse"]["spacing_max_in"], 11.75)
        self.assertEqual(high["transverse"]["spacing_max_in"], 5.875)
        self.assertEqual(deep["transverse"]["spacing_max_in"], 24.0)                   # Table 9.7.6.2.2 cap
        self.assertIn("stirrups", low["transverse"]["kind"])
        self.assertTrue(low["capacity_shear_required"])
        self.assertFalse(deep["capacity_shear_required"])
        # an axial force above Ag f'c / 10 still asks for hoops at the lesser of 6 db and 6 in
        axial = gm.beam_requirements(gm.BRANCH_YIELDING, d_in=23.5, smallest_bar_db_in=0.875, pu_kip=200.0, ag_in2=364.0, fc_ksi=5.0)
        self.assertTrue(axial["axial_hoops"])
        self.assertEqual(axial["transverse"]["spacing_max_in"], 5.25)


if __name__ == "__main__":
    unittest.main()
