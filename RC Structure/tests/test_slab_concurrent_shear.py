"""Independent Table22.5.5.1(c) axial/shear boundary calculations."""
import math
import unittest
from Design.SMRF_Slab_Local_Shear import check_one_way_strength
from Design.SMRF_Slab_Reinforcement import one_way_shear_strength


class ConcurrentShearTests(unittest.TestCase):
    def run_check(self,**kw):
        return check_one_way_strength(**dict(fc_ksi=4.,slab_thickness_in=8.,cut_width_in=12.,
            effective_depth_in=6.5,developed_tension_area_in2=.4,n_tension_kip=0.,q_vertical_kip=1.,**kw))

    def test_zero_N_independent_legacy_expression(self):
        r=self.run_check();old=one_way_shear_strength(4.,6.5,.4)
        self.assertAlmostEqual(r['phi_vc_kip'],old['phi_vc_kip_per_ft'],places=12)

    def test_signed_tension_reduces_capacity_without_axial_envelope(self):
        base=dict(fc_ksi=4.,slab_thickness_in=8.,cut_width_in=12.,effective_depth_in=6.5,
                  developed_tension_area_in2=.4,q_vertical_kip=-1.)
        for n in (12.,-12.):
            r=check_one_way_strength(**base,n_tension_kip=n)
            vc=8*(.4/(12*6.5))**(1/3)*math.sqrt(4000)/1000-n/(6*12*8)
            self.assertAlmostEqual(r['phi_vc_kip'],.75*vc*12*6.5,places=12)
        self.assertLess(check_one_way_strength(**base,n_tension_kip=12.)['phi_vc_kip'],
                        check_one_way_strength(**base,n_tension_kip=-12.)['phi_vc_kip'])

    def test_zero_capacity_is_finite_serializable_failure(self):
        r=check_one_way_strength(fc_ksi=4.,slab_thickness_in=8.,cut_width_in=12.,effective_depth_in=6.5,
            developed_tension_area_in2=.4,n_tension_kip=1000.,q_vertical_kip=1.)
        self.assertEqual(r['phi_vc_kip'],0.);self.assertEqual(r['status'],'fail');self.assertIsNone(r['utilization'])

    def test_axial_and_total_caps_apply_independently(self):
        r=check_one_way_strength(fc_ksi=4.,slab_thickness_in=8.,cut_width_in=12.,effective_depth_in=6.5,
            developed_tension_area_in2=30.,n_tension_kip=-10000.,q_vertical_kip=1.)
        self.assertEqual(r['axial_stress_term_ksi'],.2)
        self.assertAlmostEqual(r['vc_ksi'],5*math.sqrt(4000)/1000,places=12)

    def test_size_effect_and_zero_developed_area(self):
        r=check_one_way_strength(fc_ksi=5.,slab_thickness_in=30.,cut_width_in=20.,effective_depth_in=27.,
            developed_tension_area_in2=0.,n_tension_kip=0.,q_vertical_kip=1.)
        self.assertAlmostEqual(r['lambda_s'],math.sqrt(2/3.7),places=12)
        self.assertEqual(r['phi_vc_kip'],0.);self.assertEqual(r['status'],'fail')
