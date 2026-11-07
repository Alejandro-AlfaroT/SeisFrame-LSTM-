"""Independent analytical extrema and discontinuity witnesses, no native solves."""
import math
import unittest
import numpy as np
from Design.SMRF_Torsion_Station_Coverage import polynomial_points,interval_points,detail_boundaries
from Design.SMRF_Beam_Torsion import solid_section

SECTION=solid_section([[-5,5,-20,0]],5.,section_id='explicit_polynomial_fixture')
SCALE=4*SECTION['Ag_in2']*SECTION['lambda']*math.sqrt(SECTION['fc_ksi']*1000)/1000


class StationCoverageTests(unittest.TestCase):
    def samples(self,n,t):return [[n(x),t(x),0.,0.] for x in (0.,.5,1.)]
    def test_quadratic_axial_peak_controls_with_constant_torque(self):
        # r=.2+(x-.3)^2; |T|/sqrt(r) peaks exactly at .3, not at
        # endpoints/midpoint or any torque stationary point.
        c,p=polynomial_points(self.samples(lambda x:SCALE*(.8-(x-.3)**2),lambda x:2.),SECTION)
        roots=[x for x,why in p if 'guard_concurrent_threshold_ratio_stationary_point' in why]
        self.assertTrue(any(abs(x-.3)<1e-12 for x in roots))
        for x in np.linspace(0,1,101):self.assertLessEqual(2/math.sqrt(.2+(x-.3)**2),2/math.sqrt(.2)+1e-12)
    def test_affine_ratio_has_interior_stationary_point(self):
        # T=1+x, r=.8+x; derivative changes sign at x=-.6 (outside).
        # T=1+x, r=.4+x gives x=.2 (interior minimum, must retain too).
        _,p=polynomial_points(self.samples(lambda x:SCALE*(.6-x),lambda x:1+x),SECTION)
        self.assertTrue(any(abs(x-.2)<1e-12 and 'guard_concurrent_threshold_ratio_stationary_point' in why for x,why in p))
    def test_two_axial_formula_boundaries_and_invalid_interior_not_omitted(self):
        _,p=polynomial_points(self.samples(lambda x:SCALE*(1-(x-.2)*(x-.8)),lambda x:1.),SECTION)
        roots=[x for x,why in p if 'guard_threshold_axial_formula_domain_boundary' in why]
        self.assertTrue(any(abs(x-.2)<1e-12 for x in roots));self.assertTrue(any(abs(x-.8)<1e-12 for x in roots))
        self.assertTrue(any(abs(x-.5)<1e-12 for x,why in p))
    def provider(self,x,trace,*,cubic=False,missing=False):
        row=dict(guard_cut_id=f'case/member/{x}|exact_threshold|guard:P5/design',kind='physical',
            N_tension_kip=SCALE*(.8-(x-.3)**2)+(x**3 if cubic else 0),T_about_positive_axis_kipin=2.,
            Q_vertical_kip=3.-2*x,V_inplane_kip=1.+x)
        return dict(demands=[] if missing and 0<x<.5 else [row])
    def test_independent_fraction_and_station_free_guard_matching(self):
        points,audits=interval_points(self.provider,SECTION)
        self.assertTrue(any(abs(x-.3)<1e-12 for x,_ in points))
        self.assertLess(max(abs(x) for x in audits[0]['independent_residual']),1e-10)
        self.assertAlmostEqual(audits[0]['independent_fraction'],1/math.sqrt(7))
    def test_nonquadratic_field_and_missing_guard_are_refused(self):
        with self.assertRaisesRegex(ValueError,'polynomial interval witness'):
            interval_points(lambda x,t:self.provider(x,t,cubic=True),SECTION)
        with self.assertRaises(ValueError):interval_points(lambda x,t:self.provider(x,t,missing=True),SECTION)
    def test_all_installed_transverse_variant_switches_and_zone_edges(self):
        detail=dict(splice_detail=dict(hoop_stations_in=[45.,55.],transverse_detail=dict(zone_in=[44.5,56.5])))
        self.assertEqual(detail_boundaries(detail,100.,3.),[42.,44.5,45.,55.,56.5,58.])
        self.assertEqual(detail_boundaries(dict(splice_detail=dict(status='not_evaluated')),100.,3.),[])


if __name__=='__main__':unittest.main()
