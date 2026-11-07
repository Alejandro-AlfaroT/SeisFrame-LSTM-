import decimal
import unittest
import numpy as np
from scipy import sparse
from Design.SMRF_Paired_Arithmetic import two_sum,add_pair,scale_pair,paired_dense_action,paired_sparse_action

class PairedArithmetic(unittest.TestCase):
    def test_lost_displacement_update_survives_physical_cancellation(self):
        h,l=add_pair(np.array([1.,1.]),np.zeros(2),[2.**-54,0.])
        self.assertEqual(h[0],1.)
        self.assertEqual(l[0],2.**-54)
        matrix=np.array([[2.**54,-2.**54]])
        ah,al=paired_dense_action(matrix,h,l)
        self.assertEqual(float(ah[0]+al[0]),1.)
        self.assertEqual(float((matrix@h)[0]),0.)

    def test_rigid_motion_subtraction_preserves_small_strain(self):
        h,l=two_sum(np.array([1.,1.]),[2.**-54,0.])
        h,l=add_pair(h,l,[-1.,-1.])
        np.testing.assert_array_equal(h,[2.**-54,0.]);np.testing.assert_array_equal(l,[0.,0.])

    def test_decimal_oracle_dense_sparse_products_and_scaling(self):
        with decimal.localcontext() as ctx:
            ctx.prec=800;D=decimal.Decimal.from_float;rng=np.random.default_rng(243)
            a=rng.normal(size=(17,21))*2.**rng.integers(-80,80,(17,21))
            h=rng.normal(size=21)*2.**rng.integers(-80,80,21);l=np.spacing(h)*rng.normal(size=21)*.25
            dh,dl=paired_dense_action(a,h,l);sh,sl=paired_sparse_action(sparse.csr_matrix(a),h,l)
            np.testing.assert_array_equal(dh,sh);np.testing.assert_array_equal(dl,sl)
            for row in range(len(a)):
                exact=sum((D(float(v))*(D(float(x))+D(float(y))) for v,x,y in zip(a[row],h,l)),decimal.Decimal(0))
                error=abs(D(float(dh[row]))+D(float(dl[row]))-exact)
                self.assertLessEqual(error,abs(exact)*D(2.**-100)+D(2.**-1000))
            for factor in (.5,-1.,1.+2.**-27):
                ph,pl=scale_pair(h,l,factor)
                for x,y,p,q in zip(h,l,ph,pl):
                    exact=(D(float(x))+D(float(y)))*D(factor)
                    self.assertLessEqual(abs(D(float(p))+D(float(q))-exact),abs(exact)*D(2.**-100))

    def test_duplicate_csr_entries_empty_rows_and_chunk_boundary(self):
        a=sparse.csr_matrix(([1e16,1.,-1e16],[0,0,0],[0,3,3]),shape=(2,1))
        h,l=paired_sparse_action(a,[1.],[2.**-54])
        np.testing.assert_array_equal(h,[1.,0.]);np.testing.assert_array_equal(l,[2.**-54,0.])
        h,l=paired_sparse_action(sparse.csr_matrix((16386,1)),[0.],[0.])
        self.assertFalse(np.any(h));self.assertFalse(np.any(l))

    def test_invalid_or_unverified_arithmetic_is_refused(self):
        for matrix,h,l in (([[np.nan]],[1.],[0.]),([[1.]],[1.],[]),([[1j]],[1.],[0.]),([[1.]],[np.inf],[0.])):
            with self.assertRaises(ValueError):paired_dense_action(matrix,h,l)
        with self.assertRaises(ValueError):paired_sparse_action(sparse.eye(1,format='coo'),[1.],[0.])
        with self.assertRaises(FloatingPointError):scale_pair([1.],[0.],2.**451)
        with self.assertRaises(FloatingPointError):two_sum([1e308],[1e308])
        with self.assertRaises(ValueError):two_sum([1j],[0.])

if __name__=='__main__':unittest.main()
