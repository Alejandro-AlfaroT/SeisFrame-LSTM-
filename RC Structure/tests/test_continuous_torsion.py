import copy
import math
import unittest
import numpy as np
from Design.SMRF_Beam_Torsion import solid_section
from Design.SMRF_Continuous_Torsion import threshold_interval,continuous_member_coverage

SECTION=solid_section([[-5,5,-20,0]],10.,section_id='analytic_200in2_60in_perimeter')
# Exact: phi*Tth = .75*.1*200^2/60 = 50 kip-in at zero axial force.

class ContinuousThresholdTests(unittest.TestCase):
    def test_constant_and_exact_touch_are_distinct(self):
        self.assertTrue(threshold_interval(SECTION,[[0,49.],[0,0],[0,0]])['below_threshold_everywhere'])
        result=threshold_interval(SECTION,[[0,50.],[0,0],[0,0]])
        self.assertFalse(result['below_threshold_everywhere']);self.assertEqual(result['nonexempt_witness']['margin_upper_kip2in2'],0.)
        self.assertFalse(result['nonlinear_material_reserve_certified'])

    def test_hidden_peak_missed_by_endpoints_and_midpoint(self):
        # T/50=1.02-(x-.3)^2, all three fitting stations below one.
        c=np.array([[0,46.5],[0,30.],[0,-50.]])
        for x in (0,.5,1):self.assertLess(abs(np.array([1,x,x*x])@c[:,1]),50.)
        result=threshold_interval(SECTION,c)
        self.assertFalse(result['below_threshold_everywhere']);self.assertIsNotNone(result['nonexempt_witness'])

    def test_positive_quartic_requires_subdivision(self):
        # Peak .999 at .3, so strict positivity is independently known.
        c=[[0,45.45],[0,30.],[0,-50.]]
        r=threshold_interval(SECTION,c)
        self.assertTrue(r['below_threshold_everywhere']);self.assertGreater(r['processed_subintervals'],1)
        self.assertAlmostEqual(sum(b['interval'][1]-b['interval'][0] for b in r['certified_intervals']),1.)
        bounded=threshold_interval(SECTION,c,maximum_depth=0)
        self.assertFalse(bounded['below_threshold_everywhere']);self.assertTrue(bounded['unresolved_intervals'])

    def test_negative_axial_radicand_is_not_a_failure_or_exemption(self):
        # k=4*200*.1=80 kip. N=81 exceeds the threshold formula domain.
        r=threshold_interval(SECTION,[[81,0],[0,0],[0,0]])
        self.assertEqual(r['status'],'not_evaluated');self.assertFalse(r['below_threshold_everywhere'])

    def test_nonrational_root_and_reflection_preserve_result(self):
        s=solid_section([[-5,5,-20,0]],5.,section_id='irrational_root')
        c=np.array([[10.,4.],[3.,7.],[-2.,-5.]])
        r=threshold_interval(s,c);reflected=np.vstack((c.sum(axis=0),-c[1]-2*c[2],c[2]))
        self.assertTrue(r['below_threshold_everywhere']);self.assertTrue(threshold_interval(s,reflected)['below_threshold_everywhere'])

    def raw(self,*,elf=False):
        coeff=[[0,20,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
        rows=[dict(member_tag=1,case_id='c',interval_in=[a,b],coefficients=coeff) for a,b in ((0,4),(4,10))]
        return dict(selected_member_tags=[1],sections={'1':SECTION},
            analysis_cases=[dict(analysis_case_id='c',branch='actual_physical_ELF' if elf else 'actual_gravity')],
            records=[dict(member_tag=1,case_id='c',station_in=x) for x in (0,4,10)],polynomial_audits=rows,
            guard_polynomial_audits=[dict(member_tag=1,case_id='c',interval_in=r['interval_in'],
                guards=[dict(guard_suffix='|P5/design',kind='physical',coefficients=[v[:4] for v in coeff])]) for r in rows] if elf else [])

    def test_all_intervals_required_and_nonexempt_does_not_become_failure(self):
        raw=self.raw();self.assertEqual(continuous_member_coverage(raw)['members']['1']['status'],'pass')
        bad=copy.deepcopy(raw);bad['polynomial_audits'].pop()
        self.assertEqual(continuous_member_coverage(bad)['members']['1']['status'],'not_evaluated')
        raw['polynomial_audits'][0]['coefficients'][0][1]=60.
        self.assertEqual(continuous_member_coverage(raw)['members']['1']['status'],'not_evaluated')

    def test_ELF_needs_primary_P5_not_only_physical_samples(self):
        raw=self.raw(elf=True);self.assertEqual(continuous_member_coverage(raw)['members']['1']['status'],'pass')
        raw['guard_polynomial_audits'].pop()
        result=continuous_member_coverage(raw)['members']['1']
        self.assertEqual(result['status'],'not_evaluated');self.assertTrue(result['inventory_issues'])

    def test_malformed_polynomial_and_limits_refused(self):
        for coefficients in ([[0,1],[0,0]],[[0,float('nan')],[0,0],[0,0]]):
            with self.assertRaises(ValueError):threshold_interval(SECTION,coefficients)
        with self.assertRaises(ValueError):threshold_interval(SECTION,[[0,1],[0,0],[0,0]],maximum_leaves=0)

if __name__=='__main__':unittest.main()
