"""ACI 318-19 Table 22.5.5.1 one-way shear: the axial term is a stress in psi multiplied by bw·d.

Until 2026-09-26 both ``check_shear`` implementations added Nu/(6·Ag) in ksi straight to a force in
kip, so a column received essentially no axial credit. These tests pin the corrected form against
hand arithmetic and keep the two implementations (Design/ACI_Checks and RC_Design_Check) identical.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Structure_Parameters as sp  # noqa: E402
from Design import ACI_Checks  # noqa: E402
import RC_Design_Check  # noqa: E402

FC, FY = 5.0, sp.FY_KSI            # ksi
BW, H, D = 24.0, 24.0, 21.5        # in
AV, S = 0.40, 4.0                  # two #4 legs at 4 in


def hand_phi_vn(nu_kip, fc=FC, bw=BW, h=H, d=D, av=AV, s=S, fy=FY):
    fc_psi = fc * 1000.0
    nu_psi = min(nu_kip * 1000.0 / (6.0 * bw * h), 0.05 * fc_psi)
    vc = max(0.0, (2.0 * math.sqrt(fc_psi) + nu_psi) * bw * d / 1000.0)
    vc = min(vc, 5.0 * math.sqrt(fc_psi) * bw * d / 1000.0)
    vs = min(av * fy * d / s, 8.0 * math.sqrt(fc_psi) * bw * d / 1000.0)
    return 0.75 * (vc + vs)


class ShearCheckUnits(unittest.TestCase):
    def phi_vn_both(self, nu_kip):
        a = ACI_Checks.check_shear(100.0, nu_kip, bw=BW, d=D, fc_ksi=FC, Av=AV, s=S, fy_ksi=FY, h=H).capacity
        b, _, _ = RC_Design_Check.check_shear(100.0, nu_kip, bw=BW, d=D, fc=FC, Av=AV, s=S, h=H)
        self.assertAlmostEqual(a, b, places=9, msg="the two implementations must agree")
        return a

    def test_axial_compression_credit_matches_table_22_5_5_1(self):
        # 24x24 column, Nu = 500 kip: Nu/(6Ag) = 144.7 psi, below the 0.05 f'c = 250 psi cap.
        phi_vn = self.phi_vn_both(500.0)
        self.assertAlmostEqual(phi_vn, hand_phi_vn(500.0), places=9)
        vc_credit_kip = (500.0 * 1000.0 / (6.0 * BW * H)) * BW * D / 1000.0
        self.assertAlmostEqual(vc_credit_kip, 74.65, places=1)
        self.assertGreater(phi_vn - self.phi_vn_both(0.0), 0.75 * 74.0,
                           "the axial credit is tens of kip, not a fraction of a kip")

    def test_zero_axial_is_the_plain_2_root_fc_form(self):
        phi_vn = self.phi_vn_both(0.0)
        vc = 2.0 * math.sqrt(FC * 1000.0) * BW * D / 1000.0
        vs = AV * FY * D / S
        self.assertAlmostEqual(phi_vn, 0.75 * (vc + vs), places=9)

    def test_axial_term_capped_at_0_05_fc_and_vc_at_5_root_fc(self):
        # Nu/(6Ag) for 2000 kip is 579 psi -> capped at 250 psi; Vc then sits at the 5*sqrt(f'c) ceiling.
        phi_vn = self.phi_vn_both(2000.0)
        self.assertAlmostEqual(phi_vn, hand_phi_vn(2000.0), places=9)
        fc_psi = FC * 1000.0
        vc_cap = 5.0 * math.sqrt(fc_psi) * BW * D / 1000.0
        vs = AV * FY * D / S
        self.assertAlmostEqual(phi_vn, 0.75 * (vc_cap + vs), places=9)

    def test_tension_reduces_vc_and_floors_at_zero(self):
        self.assertLess(self.phi_vn_both(-200.0), self.phi_vn_both(0.0))
        # Enough tension to drive the bracket negative: Vc = 0, only Vs remains.
        phi_vn = self.phi_vn_both(-3000.0)
        vs = min(AV * FY * D / S, 8.0 * math.sqrt(FC * 1000.0) * BW * D / 1000.0)
        self.assertAlmostEqual(phi_vn, 0.75 * vs, places=9)

    def test_vs_is_limited_by_the_cross_section(self):
        # Very heavy stirrups: Vs is held at 8*sqrt(f'c)*bw*d (22.5.1.2).
        a = ACI_Checks.check_shear(100.0, 0.0, bw=BW, d=D, fc_ksi=FC, Av=4.0, s=2.0, fy_ksi=FY, h=H).capacity
        vc = 2.0 * math.sqrt(FC * 1000.0) * BW * D / 1000.0
        vs_cap = 8.0 * math.sqrt(FC * 1000.0) * BW * D / 1000.0
        self.assertAlmostEqual(a, 0.75 * (vc + vs_cap), places=9)

    def test_legacy_gross_area_estimate_when_depth_not_given(self):
        a = ACI_Checks.check_shear(100.0, 500.0, bw=BW, d=D, fc_ksi=FC, Av=AV, s=S, fy_ksi=FY).capacity
        self.assertAlmostEqual(a, hand_phi_vn(500.0, h=D / 0.9), places=9)

    def test_units_of_the_axial_term_are_a_stress(self):
        # Doubling bw at fixed Nu halves the stress Nu/(6Ag) and doubles bw*d: the axial credit in
        # kip is unchanged. The old code's credit scaled inversely with the section instead.
        def credit(bw):
            with_n = ACI_Checks.check_shear(1.0, 300.0, bw=bw, d=D, fc_ksi=FC, Av=AV, s=S, fy_ksi=FY, h=H).capacity
            without = ACI_Checks.check_shear(1.0, 0.0, bw=bw, d=D, fc_ksi=FC, Av=AV, s=S, fy_ksi=FY, h=H).capacity
            return with_n - without
        self.assertAlmostEqual(credit(BW), credit(2 * BW), places=9)


if __name__ == "__main__":
    unittest.main()
