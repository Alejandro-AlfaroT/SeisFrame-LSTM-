from decimal import Decimal,localcontext
from fractions import Fraction
import json
import unittest
from Design.SMRF_Interval_Material import Interval,UnsupportedBox
from Design.SMRF_Interval_Quadrature import integrate_rectangle,replay_rectangle_integral


class GaussianEnclosureTests(unittest.TestCase):
    def test_constant_fourth_derivative_corrects_gauss_error_exactly(self):
        fn=lambda x,y:[x.square().square()+y.square().square()]
        result=integrate_rectangle(fn,fn,[(-1,1),(-1,1)],provider_identity='quartic',
            absolute_error=['1e-50'],maximum_cells=1,fourth_provider=fn)
        self.assertTrue(result['requested_error_met']);self.assertEqual(result['leaf_cells'],1)
        lo,hi=map(Fraction,result['integral'][0])
        self.assertLessEqual(lo,Fraction(8,5));self.assertGreaterEqual(hi,Fraction(8,5))
        self.assertTrue(result['leaves'][0]['fourth_derivative_used'])

    def test_cubic_gauss_exactness_and_replay(self):
        fn=lambda x,y:[x*x*x+2*y.square()]
        result=integrate_rectangle(fn,fn,[(-1,1),(-1,1)],provider_identity='cubic',
            absolute_error=['1e-50'],maximum_cells=1,fourth_provider=fn)
        self.assertTrue(result['requested_error_met'])
        lo,hi=map(Fraction,result['integral'][0]);self.assertLessEqual(lo,Fraction(8,3));self.assertGreaterEqual(hi,Fraction(8,3))
        self.assertEqual(replay_rectangle_integral(json.loads(json.dumps(result)),fn,fn,provider_identity='cubic',fourth_provider=fn),result)
        with self.assertRaises(ValueError):replay_rectangle_integral(result,fn,fn,provider_identity='cubic')

    def test_exponential_integral_against_independent_antiderivative(self):
        fn=lambda x,y:[x.exp()]
        result=integrate_rectangle(fn,fn,[('-.1','.2'),(0,1)],provider_identity='exp',
            absolute_error=['1e-8'],maximum_cells=128,fourth_provider=fn)
        self.assertTrue(result['requested_error_met'])
        with localcontext() as ctx:
            ctx.prec=110;expected=Decimal('.2').exp()-Decimal('-.1').exp()
        self.assertTrue(Interval(*result['integral'][0]).contains(expected))

    def test_nonsmooth_cell_retains_value_bound_and_subdivides(self):
        def values(x,y):
            if x.lo>=0:return [x]
            if x.hi<=0:return [-x]
            return [Interval(0,max(x.lo.copy_abs(),x.hi.copy_abs()))]
        def derivatives(x,y):
            if x.value.lo>=0:return [x]
            if x.value.hi<=0:return [-x]
            raise UnsupportedBox('absolute_value_branch')
        result=integrate_rectangle(values,derivatives,[(-1,1),(0,1)],provider_identity='abs',
            absolute_error=['1e-40'],maximum_cells=8,fourth_provider=derivatives)
        self.assertTrue(result['requested_error_met']);self.assertTrue(Interval(*result['integral'][0]).contains(1))
        self.assertGreater(result['calls']['fallback_cells'],0)

    def test_invalid_fourth_provider_is_refused(self):
        fn=lambda x,y:[x+y]
        with self.assertRaises(ValueError):
            integrate_rectangle(fn,fn,[(0,1),(0,1)],provider_identity='bad',
                absolute_error=['.1'],fourth_provider=lambda x,y:[Interval(0)])


if __name__=='__main__':unittest.main()
