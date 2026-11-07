from decimal import Decimal,localcontext
from fractions import Fraction
import json
import unittest
from Design.SMRF_Interval_Material import (Interval,Gradient,UnsupportedBox,
    concrete_stress_bounds,concrete_stress_gradient)
from Design.SMRF_Interval_Quadrature import integrate_rectangle,replay_rectangle_integral


class RectangleIntegralTests(unittest.TestCase):
    def test_affine_integrals_are_exact_in_one_cell(self):
        fn=lambda x,y:[2*x-3*y+5,x+y]
        result=integrate_rectangle(fn,fn,[(0,2),(-1,3)],provider_identity='analytic_affine',
                                   absolute_error=['1e-50']*2,maximum_cells=1)
        self.assertTrue(result['requested_error_met']);self.assertEqual(result['leaf_cells'],1)
        self.assertEqual([Interval(*v) for v in result['integral']],[Interval(32),Interval(16)])

    def test_nonlinear_integral_against_exact_rational(self):
        fn=lambda x,y:[x.square()+y.square()]
        result=integrate_rectangle(fn,fn,[(0,1),(0,1)],provider_identity='analytic_quadratic',
                                   absolute_error=['.001'],maximum_cells=1024)
        self.assertTrue(result['requested_error_met'])
        low,high=map(Fraction,result['integral'][0]);self.assertLessEqual(low,Fraction(2,3));self.assertGreaterEqual(high,Fraction(2,3))
        self.assertGreater(result['leaf_cells'],1)

    def test_branch_cells_keep_value_bound_until_split(self):
        def values(x,y):
            if x.lo>=0:return [x]
            if x.hi<=0:return [-x]
            return [Interval(0,max(x.lo.copy_abs(),x.hi.copy_abs()))]
        def gradient(x,y):
            if x.value.lo>=0:return [x]
            if x.value.hi<=0:return [-x]
            raise UnsupportedBox('absolute_value_branch')
        result=integrate_rectangle(values,gradient,[(-1,1),(0,2)],provider_identity='analytic_abs',
                                   absolute_error=['1e-40'],maximum_cells=8)
        self.assertTrue(result['requested_error_met']);self.assertEqual(Interval(*result['integral'][0]),Interval(2))
        self.assertGreaterEqual(result['calls']['fallback_cells'],1)

    def test_exhausted_budget_is_unknown_but_retains_valid_enclosure(self):
        fn=lambda x,y:[x.square()+y.square()]
        result=integrate_rectangle(fn,None,[(0,1),(0,1)],provider_identity='analytic_quadratic_no_derivative',
                                   absolute_error=['1e-10'],maximum_cells=1)
        self.assertEqual(result['status'],'unknown');self.assertFalse(result['physical_failure_claimed'])
        self.assertEqual(result['integral'],[['0','2']])

    def test_uniform_actual_concrete_stress_has_independent_area_oracle(self):
        def values(x,y):return concrete_stress_bounds(-.0004,.0002,0.,fc_ksi=4.,peak_strain=.002)
        def gradient(x,y):
            return concrete_stress_gradient(*[Gradient.constant(v,2) for v in (-.0004,.0002,0.)],
                                            fc_ksi=4.,peak_strain=.002)
        result=integrate_rectangle(values,gradient,[(-3,2),(-1,3)],provider_identity='actual_uniform_slab',
                                   absolute_error=['1e-50']*3,maximum_cells=1)
        self.assertTrue(result['requested_error_met'])
        with localcontext() as ctx:
            ctx.prec=110
            fc,peak,e=map(Decimal,(4.,.002,-.0004));n=Decimal(.8)+fc*Decimal(6.894757293168)/17
            r=-e/peak;expected=-20*fc*n*r/(n-1+(n*r.ln()).exp())
        self.assertTrue(Interval(*result['integral'][0]).contains(expected))
        self.assertFalse(result['continuous_capacity_verified'])

    def test_replay_and_forged_acceptance(self):
        fn=lambda x,y:[x+y]
        result=integrate_rectangle(fn,fn,[(0,1),(0,1)],provider_identity='linear',absolute_error=['1e-20'],maximum_cells=2)
        self.assertEqual(replay_rectangle_integral(json.loads(json.dumps(result)),fn,fn,provider_identity='linear'),result)
        result['continuous_capacity_verified']=True
        with self.assertRaises(ValueError):replay_rectangle_integral(result,fn,fn,provider_identity='linear')

    def test_invalid_geometry_provider_shape_and_limits_refused(self):
        fn=lambda x,y:[x]
        for rectangle in [[(0,0),(0,1)],[(1,0),(0,1)]]:
            with self.assertRaises(ValueError):
                integrate_rectangle(fn,fn,rectangle,provider_identity='bad',absolute_error=['.1'])
        with self.assertRaises(ValueError):
            integrate_rectangle(fn,fn,[(0,1),(0,1)],provider_identity='bad',absolute_error=['.1','.1'])
        with self.assertRaises(ValueError):
            integrate_rectangle(fn,fn,[(0,1),(0,1)],provider_identity='bad',absolute_error=['0'])
        with self.assertRaises(ValueError):
            integrate_rectangle(fn,fn,[(0,1),(0,1)],provider_identity='bad',absolute_error=['.1'],maximum_cells=True)


if __name__=='__main__':unittest.main()
