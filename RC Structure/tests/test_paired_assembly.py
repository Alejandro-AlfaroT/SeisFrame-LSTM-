import decimal
import unittest
import numpy as np
from Design.SMRF_Paired_Assembly import paired_congruence,paired_triplet_matrix
from Design.SMRF_Paired_Arithmetic import paired_dense_action,add_pair

class PairedAssembly(unittest.TestCase):
    def test_congruence_preserves_small_physical_energy(self):
        k=np.diag([2.**54,1.]);t=np.array([[1.,1.],[0.,1.]])
        h,l=paired_congruence(k,t);u=np.array([-1.,1.])
        np.testing.assert_array_equal((t.T@k@t)@u,[0.,0.])
        ah,al=add_pair(*paired_dense_action(h,u,np.zeros(2)),*paired_dense_action(l,u,np.zeros(2)))
        np.testing.assert_array_equal(ah+al,[0.,1.])
        self.assertEqual(float(u@(ah+al)),1.)

    def test_duplicate_sum_preserves_nonzero_low_coefficient(self):
        h,l=paired_triplet_matrix([2.**54,1.,-2.**54],[0.,0.,0.],[0,0,0],[0,0,1],shape=(2,2))
        self.assertEqual(l[0,0],1.)
        np.testing.assert_array_equal(h.toarray()@np.ones(2),[0.,0.])
        np.testing.assert_array_equal(h.toarray()@np.ones(2)+l.toarray()@np.ones(2),[1.,0.])

    def test_independent_decimal_rectangular_congruence(self):
        with decimal.localcontext() as ctx:
            ctx.prec=800;D=decimal.Decimal.from_float;rng=np.random.default_rng(246)
            k=rng.normal(size=(5,5))*2.**rng.integers(-15,15,(5,5));t=rng.normal(size=(5,3))
            h,l=paired_congruence(k,t)
            for i in range(3):
                for j in range(3):
                    exact=sum((D(float(t[a,i]))*D(float(k[a,b]))*D(float(t[b,j])) for a in range(5) for b in range(5)),decimal.Decimal(0))
                    self.assertLessEqual(abs(D(float(h[i,j]))+D(float(l[i,j]))-exact),max(D(1.),abs(exact))*D(2.**-98))

    def test_permutation_and_group_order_are_preserved(self):
        k=np.array([[2.,3.,5.],[7.,11.,13.],[17.,19.,23.]])
        t=np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])
        h,l=paired_congruence(k,t);np.testing.assert_array_equal(h,t.T@k@t);self.assertFalse(np.any(l))
        for order in ([0,1,2],[2,0,1],[1,2,0]):
            values=np.array([1e16,1.,-1e16])[order]
            h,l=paired_triplet_matrix(values,np.zeros(3),[0,0,0],[0,0,0],shape=(1,1))
            self.assertEqual(h[0,0],1.);self.assertEqual(l[0,0],0.)

    def test_invalid_geometry_indices_and_overflow_refused(self):
        with self.assertRaises(ValueError):paired_congruence(np.eye(2),np.eye(3))
        with self.assertRaises(ValueError):paired_congruence([[1j]],[[1.]])
        for values,low,rows,columns in (([1.],[],[0],[0]),([np.nan],[0.],[0],[0]),([1.],[0.],[-1],[0]),([1.],[0.],[.5],[0])):
            with self.assertRaises(ValueError):paired_triplet_matrix(values,low,rows,columns,shape=(1,1))
        with self.assertRaises(FloatingPointError):paired_triplet_matrix([1e308,1e308],[0.,0.],[0,0],[0,0],shape=(1,1))

if __name__=='__main__':unittest.main()
