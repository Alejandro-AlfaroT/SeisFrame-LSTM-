"""Independent transformed-section integrals and code formula witnesses."""
import math,unittest
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from Design.SMRF_Slab_Stiffness import SlabStripStiffness,_circle_integrals

def rows(area=.31/8,bar=.31):
    return [dict(id=f'{a}{s}',axis=a,z_in=s*z,area_in2_per_in=area,
                 bar_area_in2=bar,fy_ksi=60,es_ksi=29000)
            for a,z in [('x',3),('y',2.25)] for s in [-1,1]]

class StiffnessTests(unittest.TestCase):
    def setUp(self):self.s=SlabStripStiffness(8,5,rows(),ec_ksi=57*math.sqrt(5000))

    def test_partial_circle_closed_form_independent_quadrature(self):
        r=.37
        for cut in [-r-1,-r*.6,0,r*.7,r+1]:
            values=_circle_integrals(r,cut)
            upper=math.asin(np.clip(cut/r,-1,1))
            for p in range(3):
                expected=quad(lambda t:(r*math.sin(t))**p*2*r*r*math.cos(t)**2,-math.pi/2,upper,epsabs=1e-13)[0]
                self.assertAlmostEqual(values[p],expected,12)

    def test_independent_finite_void_cracked_neutral_axis_and_I(self):
        material=rows();ec=57*math.sqrt(5000)
        # Different integration parameter and adaptive quadrature, no production
        # circle formulas or property recovery in this independent calculation.
        def integral(c,p):
            answer=c**(p+1)/(p+1)
            for row in material:
                center=4+row['z_in'];r=math.sqrt(row['bar_area_in2']/math.pi)
                count=row['area_in2_per_in']/row['bar_area_in2']
                if c>center-r:
                    upper=math.asin(np.clip((c-center)/r,-1,1))
                    answer-=quad(lambda t:(c-center-r*math.sin(t))**p*2*r*r*math.cos(t)**2*count,
                                 -math.pi/2,upper,epsabs=1e-12)[0]
                if row['axis']=='x':answer+=29000/ec*row['area_in2_per_in']*(c-center)**p
            return answer
        c=brentq(lambda c:integral(c,1),0,8,xtol=1e-12)
        p=self.s.properties('x',1)
        self.assertAlmostEqual(p['neutral_depth_from_compression_face_in'],c,10)
        self.assertAlmostEqual(p['Icr_in4_per_in'],integral(c,2),10)
        self.assertAlmostEqual(p['Icr_over_Ig'],.22447737232883652,10)

    def test_minimum_mat_counterexample_to_quarter_gross_bound(self):
        s=SlabStripStiffness(8,5,rows(.0018*8,.2),ec_ksi=57*math.sqrt(5000))
        for axis,expected in [('x',.09494533560973495),('y',.07582205139419154)]:
            p=s.properties(axis,1)
            self.assertAlmostEqual(p['Icr_over_Ig'],expected,10)
            self.assertLess(p['Icr_over_Ig'],.25)

    def test_code_Ie_hand_and_limits(self):
        p=self.s.properties('x',1);mcr=p['Mcr_kip_in_per_in'];r=p['Icr_over_Ig']
        for ratio in [0,.5,2/3,1,2,4,20]:
            out=self.s.evaluate('x',ratio*mcr,purpose='service_deflection')
            expected=1 if ratio<=2/3 else r/(1-(2/3/ratio)**2*(1-r))
            self.assertAlmostEqual(out['alpha_Ie_over_Ig'],expected,12)
            self.assertGreaterEqual(out['alpha_Ie_over_Ig'],r-1e-12)
            self.assertLessEqual(out['alpha_Ie_over_Ig'],1+1e-12)
        self.assertAlmostEqual(self.s.evaluate('x',2*mcr,purpose='service_deflection')['alpha_Ie_over_Ig'],.24564434425423717,10)

    def test_asymmetric_mat_reflection_and_axis_covariance(self):
        inventory=rows();inventory[0]['area_in2_per_in']=.31/5
        s=SlabStripStiffness(8,5,inventory,ec_ksi=4030)
        plus=s.properties('x',1);minus=s.properties('x',-1)
        self.assertGreater(abs(plus['Icr_over_Ig']-minus['Icr_over_Ig']),.01)
        reflected=[dict(r,z_in=-r['z_in']) for r in inventory]
        sr=SlabStripStiffness(8,5,reflected,ec_ksi=4030)
        self.assertAlmostEqual(plus['Icr_in4_per_in'],sr.properties('x',-1)['Icr_in4_per_in'],12)
        swapped=[dict(r,axis='y' if r['axis']=='x' else 'x') for r in inventory]
        sy=SlabStripStiffness(8,5,swapped,ec_ksi=4030)
        self.assertAlmostEqual(plus['Icr_in4_per_in'],sy.properties('y',1)['Icr_in4_per_in'],12)

    def test_geometric_scaling_and_explicit_elastic_domain(self):
        scale=2
        inventory=[dict(r,z_in=r['z_in']*scale,bar_area_in2=r['bar_area_in2']*scale**2,
                        area_in2_per_in=r['area_in2_per_in']*scale) for r in rows()]
        big=SlabStripStiffness(16,5,inventory,ec_ksi=self.s.ec)
        self.assertAlmostEqual(big.properties('x',1)['Icr_in4_per_in'],self.s.properties('x',1)['Icr_in4_per_in']*scale**3,9)
        small=self.s.evaluate('x',5,purpose='factored_action_stiffness')
        large=big.evaluate('x',5*scale**2,purpose='factored_action_stiffness')
        self.assertAlmostEqual(small['max_steel_yield_utilization'],large['max_steel_yield_utilization'],12)
        self.assertTrue(small['elastic_reference_applicable'])
        failed=self.s.evaluate('x',100,purpose='factored_action_stiffness')
        self.assertFalse(failed['elastic_reference_applicable'])
        self.assertTrue(failed['needs_secant_or_nonlinear_analysis'])
        self.assertIn('local_elastic_steel_estimate_exceeds_actual_yield',failed['applicability_reasons'])
        self.assertGreater(failed['Icr_in4_per_in'],0)
        self.assertFalse(failed['complete_floor_qualified'])

    def test_input_domain_and_sign_contract(self):
        self.assertEqual(self.s.properties('x',1)['compression_face'],'bottom')
        self.assertEqual(self.s.properties('x',-1)['compression_face'],'top')
        with self.assertRaises(ValueError):self.s.evaluate('x',1,purpose='unidentified')
        with self.assertRaises(ValueError):self.s.properties('z',1)
        with self.assertRaises(ValueError):SlabStripStiffness(8,5,rows(),ec_ksi=float('nan'))
        both=self.s.both_axes({'x':1,'y':-2},purpose='service_deflection')
        self.assertEqual(both['y']['moment_sign'],-1)

if __name__=='__main__':unittest.main()
