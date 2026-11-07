from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace
import unittest
from Design.SMRF_Interval_Material import Interval,Gradient,UnsupportedBox
from Design.SMRF_Interval_Taylor import Taylor
from Design.SMRF_Interval_Bands import clipped_boundaries,band_provider,material_branch_curves
from Design.SMRF_Interval_Quadrature import integrate_rectangle,replay_rectangle_integral


class CurvedStripCoordinatesTests(unittest.TestCase):
    def test_polynomial_moments_cover_original_rectangle(self):
        curves=[lambda z:z/2+Interval('.25'),lambda z:z.square()/4+1]
        density=lambda s,z:[s*0+1,s,z,s.square(),z.square(),s*z]
        total=[Interval(0)]*6
        for band in range(3):
            fn=band_provider(density,curves,band,(0,2))
            result=integrate_rectangle(fn,fn,[(0,1),(0,1)],provider_identity='polynomial_band/'+str(band),
                absolute_error=['1e-8']*6,maximum_cells=256,fourth_provider=fn)
            self.assertTrue(result['requested_error_met'])
            total=[a+Interval(*b) for a,b in zip(total,result['integral'])]
        for bounds,expected in zip(total,[2,2,1,Fraction(8,3),Fraction(2,3),1]):
            self.assertLessEqual(Fraction(bounds.lo),expected);self.assertGreaterEqual(Fraction(bounds.hi),expected)

    def test_crossing_curves_have_order_statistic_enclosures_and_exact_coverage(self):
        curves=[lambda z:z,lambda z:1-z,lambda z:4*z-2]
        whole=clipped_boundaries(curves,Interval(0,1),(0,1))
        for numerator in range(21):
            z=Fraction(numerator,20);exact=sorted([Fraction(0),z,1-z,max(0,min(1,4*z-2)),Fraction(1)])
            for value,bounds in zip(exact,whole):
                self.assertLessEqual(Fraction(bounds.lo),value);self.assertGreaterEqual(Fraction(bounds.hi),value)
            p=Decimal(numerator)/20
            point=clipped_boundaries(curves,Interval(p),(0,1))
            total=sum((b-a for a,b in zip(point[:-1],point[1:])),Interval(0))
            self.assertTrue(total.contains(1));self.assertLess(total.hi-total.lo,Decimal('1e-60'))
        with self.assertRaises(UnsupportedBox):clipped_boundaries(curves,Gradient.variable((0,1),0,1),(0,1))
        with self.assertRaises(UnsupportedBox):clipped_boundaries(curves,Taylor.variable((0,1)),(0,1))

    def test_clipped_linear_curve_integral_and_replay(self):
        curves=[lambda z:4*z-2];total=Interval(0)
        for band in range(2):
            fn=band_provider(lambda s,z:[s*0+1],curves,band,(0,1))
            result=integrate_rectangle(fn,fn,[(0,1),(0,1)],provider_identity='clipped/'+str(band),
                absolute_error=['1e-40'],maximum_cells=32,fourth_provider=fn)
            self.assertTrue(result['requested_error_met'])
            total+=Interval(*result['integral'][0])
            self.assertEqual(replay_rectangle_integral(result,fn,fn,provider_identity='clipped/'+str(band),fourth_provider=fn),result)
        self.assertTrue(total.contains(1))

    def test_actual_branch_curve_matches_independent_determinant_root(self):
        section=SimpleNamespace(j=0,other=1,sref=0.,peak=.002)
        q=list(map(Interval,['.001','.002','.0004','.0001','.0002','.0003','.00001']))
        curve=material_branch_curves(section,q,dict(id='slab',s=(-2,2)))[0]
        z=Fraction(1,4);a=Fraction(1,1000)+z*Fraction(1,10000)
        b=Fraction(2,1000)+z*Fraction(2,10000);g=Fraction(4,10000)+z*Fraction(3,10000)
        expected=(g*g/(4*b)-a)/Fraction(1,100000)
        actual=curve(Interval('.25'))
        self.assertLessEqual(Fraction(actual.lo),expected);self.assertGreaterEqual(Fraction(actual.hi),expected)
        web=material_branch_curves(section,q,dict(id='web',s=(-2,2)))[0](Interval('.25'))
        self.assertTrue(web.contains('-102.5'))

    def test_pole_and_zero_weak_curvature_are_not_smoothness_claims(self):
        section=SimpleNamespace(j=0,other=1,sref=0.,peak=.002)
        q=list(map(Interval,['0','.001','.0004','0','.001','0','.00001']))
        curve=material_branch_curves(section,q,dict(id='slab',s=(-2,2)))[0]
        self.assertEqual(curve(Interval(-2,0)),Interval(-2,2))
        with self.assertRaises(UnsupportedBox):curve(Gradient.variable((-2,0),0,1))
        q[6]=Interval(0)
        with self.assertRaises(UnsupportedBox):material_branch_curves(section,q,dict(id='slab',s=(-2,2)))

    def test_near_rank_one_curve_cancellation_has_exact_constant_oracle(self):
        section=SimpleNamespace(j=0,other=1,sref=0.,peak=.002)
        # a=1+z-5e-5,b=1+z,g=2*(1+z),ks=1e-5.
        # det(E)=ks*(s-5)*(1+z), so the zero-eigenvalue curve is exactly s=5.
        q=list(map(Interval,['.99995','1','2','1','1','2','.00001']))
        curve=material_branch_curves(section,q,dict(id='slab',s=(0,10)))[0]
        values=[curve(Interval('-.25','.25')),curve(Gradient.variable(('-.25','.25'),0,1)),
                curve(Taylor.variable(('-.25','.25')))]
        for value in values:
            bounds=value if isinstance(value,Interval) else value.value
            self.assertTrue(bounds.contains(5));self.assertLess(bounds.hi-bounds.lo,Decimal('1e-60'))
            if isinstance(value,(Gradient,Taylor)):
                for derivative in value.derivatives:
                    self.assertTrue(derivative.contains(0));self.assertLess(derivative.hi-derivative.lo,Decimal('1e-60'))

    def test_invalid_band_coordinate_is_refused(self):
        with self.assertRaises(ValueError):band_provider(lambda s,z:[s],[],1,(0,1))
        fn=band_provider(lambda s,z:[s],[],0,(0,1))
        with self.assertRaises(ValueError):fn(Interval('-.1','.1'),Interval(0,1))


if __name__=='__main__':unittest.main()
