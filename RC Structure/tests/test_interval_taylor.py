from decimal import Decimal,localcontext
from fractions import Fraction
import unittest
from Design.SMRF_Interval_Material import Interval,UnsupportedBox
from Design.SMRF_Interval_Taylor import Taylor


class TaylorCoefficientTests(unittest.TestCase):
    def assert_enclosed(self,bound,value):
        self.assertLessEqual(Fraction(bound.lo),Fraction(value));self.assertGreaterEqual(Fraction(bound.hi),Fraction(value))

    def test_polynomial_coefficients_and_whole_interval_fourth_derivative(self):
        x=Taylor.variable((0,0));y=(1+x)*(1+x)*(1+x)*(1+x)
        for bound,expected in zip(y.coefficients,(1,4,6,4,1)):self.assert_enclosed(bound,expected)
        x=Taylor.variable((-1,2));fourth=(x*x*x*x).coefficients[4]
        self.assertEqual(fourth,Interval(1))

    def test_reciprocal_geometric_series(self):
        x=Taylor.variable((0,0));y=1/(1+x)
        for i,bound in enumerate(y.coefficients):self.assert_enclosed(bound,(-1)**i)

    def test_log_exp_and_square_root_independent_series(self):
        x=Taylor.variable((0,0))
        for i,bound in enumerate((1+x).log().coefficients[1:],1):self.assert_enclosed(bound,Fraction((-1)**(i+1),i))
        for bound,expected in zip((1+x).sqrt().coefficients,(1,Fraction(1,2),Fraction(-1,8),Fraction(1,16),Fraction(-5,128))):
            self.assert_enclosed(bound,expected)
        for bound,expected in zip(x.exp().coefficients,(1,1,Fraction(1,2),Fraction(1,6),Fraction(1,24))):self.assert_enclosed(bound,expected)

    def test_variable_exponent_has_independent_hand_series(self):
        # log((1+x)^(2+x))=2x+x^3/6-x^4/6+O(x^5).
        x=Taylor.variable((0,0));y=((2+x)*(1+x).log()).exp()
        for bound,expected in zip(y.coefficients,(1,2,2,Fraction(3,2),Fraction(5,6))):self.assert_enclosed(bound,expected)

    def test_range_coefficients_contain_analytic_derivatives(self):
        y=Taylor.variable(('-.1','.2')).exp()
        with localcontext() as ctx:
            ctx.prec=110
            for value in ('-.1','0','.2'):
                derivative=Decimal(value).exp()
                self.assertTrue(y.coefficients[4].contains(derivative/24))

    def test_non_smooth_and_invalid_orders_refused(self):
        with self.assertRaises(UnsupportedBox):Taylor.variable((0,1)).sqrt()
        with self.assertRaises(ValueError):Taylor.variable((1,2),order=0)
        with self.assertRaises(ValueError):Taylor.variable((1,2),order=9)
        with self.assertRaises(ValueError):Taylor.variable((1,2),order=2)+Taylor.variable((1,2),order=3)


if __name__=='__main__':unittest.main()
