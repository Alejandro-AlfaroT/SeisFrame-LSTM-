"""Independent linear identities and complete-equation refinement safeguards."""
import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from Design.SMRF_Linear_Refinement import refine_gauged_rhs


class RefinementTests(unittest.TestCase):
    def test_singular_spring_chain_has_exact_balanced_full_equations(self):
        # Three actual springs: one gauge, two independent self-balanced loads.
        a=sparse.csc_matrix([[2.,-2.,0.],[-2.,5.,-3.],[0.,-3.,3.]])
        free=np.array([1,2]);exact=np.array([[3.,-2.],[5.,4.]])
        b=np.array([[-6.,4.],[0.,-22.],[6.,18.]])
        factor=splu(a[free,:][:,free]);initial=exact+np.array([[.1,-.2],[.3,.1]])
        out,audit=refine_gauged_rhs(a[:,free],b,free,initial,factor.solve)
        np.testing.assert_allclose(out,exact,atol=2e-15,rtol=0)
        np.testing.assert_allclose(a[:,free]@out,b,atol=4e-15,rtol=0)
        self.assertTrue(audit['all_equation_rows_included'])
        self.assertFalse(audit['forward_error_bound_claimed'])

    def test_bad_correction_is_discarded_per_column(self):
        a=sparse.eye(2,format='csr');b=np.array([[1.,2.],[3.,4.]])
        x=b+.1
        def mixed(r):
            result=r.copy();result[:,1]*=-1;return result
        out,audit=refine_gauged_rhs(a,b,np.array([0,1]),x,mixed)
        np.testing.assert_array_equal(out[:,0],b[:,0]);np.testing.assert_array_equal(out[:,1],x[:,1])
        self.assertLessEqual(audit['final_absolute_residuals'][1],audit['initial_absolute_residuals'][1])
        self.assertEqual(audit['correction_solves'],1)

    def test_gauge_equation_cannot_disappear_from_residual(self):
        a=sparse.csr_matrix([[1.],[1.]]);b=np.array([[100.],[1.]])
        out,audit=refine_gauged_rhs(a,b,np.array([1]),np.array([[50.]]),lambda r:r)
        # The free-row exact answer worsens the omitted row; it must not replace50.
        np.testing.assert_array_equal(out,[[50.]])
        self.assertEqual(audit['final_absolute_residuals'],[50.])

    def test_zero_state_and_explicit_no_iteration_do_not_solve(self):
        def forbidden(r):raise AssertionError('No solve needed')
        a=sparse.eye(2,format='csr');free=np.array([0,1]);zero=np.zeros((2,1))
        out,audit=refine_gauged_rhs(a,zero,free,zero,forbidden)
        self.assertEqual(audit['correction_solves'],0)
        out,audit=refine_gauged_rhs(a,zero,free,np.ones((2,1)),forbidden,maximum_steps=0)
        np.testing.assert_array_equal(out,1.)

    def test_nonfinite_and_unbounded_inputs_refused(self):
        a=sparse.eye(2,format='csr');b=np.ones((2,1));x=b+.1;free=np.array([0,1])
        with self.assertRaises(ValueError):refine_gauged_rhs(a,b,free,x,lambda r:r*float('nan'))
        with self.assertRaises(ValueError):refine_gauged_rhs(a,b,free,x,lambda r:r,maximum_steps=4)
        with self.assertRaises(ValueError):refine_gauged_rhs(a,b,np.array([0,0]),x,lambda r:r)


if __name__=='__main__':unittest.main()
