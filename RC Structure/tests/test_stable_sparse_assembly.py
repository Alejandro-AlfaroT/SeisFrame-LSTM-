"""Independent exact-decimal and kinematic matrix assembly oracles."""
from decimal import Decimal,localcontext
import unittest
import numpy as np
from scipy import sparse
from Design.SMRF_Stable_Sparse_Assembly import accurate_triplet_matrix


class StableAssembly(unittest.TestCase):
    def test_cancellation_and_permutation_against_decimal(self):
        rng=np.random.default_rng(206);values=np.r_[1e16,np.ones(1001),-1e16,2.**-30,-2.**-31]
        with localcontext() as ctx:
            ctx.prec=90;expected=float(sum((Decimal.from_float(float(v)) for v in values),Decimal(0)))
        for order in (np.arange(len(values)),rng.permutation(len(values)),np.arange(len(values))[::-1]):
            actual=accurate_triplet_matrix(values[order],np.zeros(len(values),int),np.zeros(len(values),int),shape=(1,1))
            self.assertEqual(actual[0,0],expected)
        ordinary=sparse.coo_matrix((values,(np.zeros(len(values),int),np.zeros(len(values),int))),shape=(1,1)).tocsr()
        self.assertNotEqual(ordinary[0,0],expected)

    def test_multiple_rectangular_groups_and_exact_zeros(self):
        data=[1e16,7.,-1e16,3.,-3.,-5.,2.];rows=[2,2,2,0,0,1,1];cols=[1,1,1,4,4,0,2]
        result=accurate_triplet_matrix(data,rows,cols,shape=(4,5))
        expected=np.zeros((4,5));expected[2,1]=7.;expected[1,0]=-5.;expected[1,2]=2.
        np.testing.assert_array_equal(result.toarray(),expected);self.assertEqual(result.nnz,3)
        self.assertTrue(result.has_canonical_format)

    def test_repeated_rigid_support_incidence_preserves_virtual_work(self):
        # Local slave nodes both inherit one actual support translation, while
        # a third node remains free. No physical DOF or constraint is discarded.
        stiffness=np.array([[1e16,-1e16,0.],[-1e16,1e16+4.,-4.],[0.,-4.,4.]])
        ids=np.array([0,0,1]);r,c=np.indices(stiffness.shape)
        actual=accurate_triplet_matrix(stiffness.ravel(),ids[r.ravel()],ids[c.ravel()],shape=(2,2))
        expected=np.array([[4.,-4.],[-4.,4.]])
        np.testing.assert_array_equal(actual.toarray(),expected)
        u=np.array([.25,.75]);local=u[ids]
        with localcontext() as ctx:
            ctx.prec=90
            energy=sum((Decimal.from_float(float(local[i]))*Decimal.from_float(float(stiffness[i,j]))*Decimal.from_float(float(local[j]))
                for i in range(3) for j in range(3)),Decimal(0))
        self.assertEqual(float(u@actual@u),float(energy))
        np.testing.assert_array_equal(actual@np.ones(2),0.)

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(accurate_triplet_matrix([],[],[],shape=(0,3)).shape,(0,3))
        for data,r,c,shape in (([1.],[0.],[0],(1,1)),([1.],[1],[0],(1,1)),([np.nan],[0],[0],(1,1)),
            ([1.],[0,1],[0],(2,2)),([1.],[0],[0],(-1,1)),([1.],[0],[0],(True,1))):
            with self.assertRaises(ValueError):accurate_triplet_matrix(data,r,c,shape=shape)
        with self.assertRaises((ValueError,OverflowError)):
            accurate_triplet_matrix([1e308,1e308],[0,0],[0,0],shape=(1,1))


if __name__=='__main__':unittest.main()
