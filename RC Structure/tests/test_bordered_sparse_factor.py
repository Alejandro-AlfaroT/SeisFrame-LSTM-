"""Independent full dense oracles for retained small-border sparse equations."""
import unittest
import numpy as np
from Design.SMRF_Bordered_Sparse_Factor import BorderedSparseFactor
from Design.SMRF_Mixed_Membrane_Energy import MixedMembraneEnergy


class BorderedFactor(unittest.TestCase):
    def test_noncontiguous_unsymmetric_border_vector_and_batch(self):
        rng=np.random.default_rng(193);a=rng.normal(size=(31,31));a+=np.diag(np.arange(31)+15.)
        rhs=rng.normal(size=(31,4));solver=BorderedSparseFactor(a,np.array([17,2,29,11]))
        expected=np.linalg.solve(a,rhs);actual=solver.solve(rhs)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(solver.solve(rhs[:,2]),expected[:,2],rtol=1e-12,atol=1e-13)
        self.assertLess(np.max(abs(a@actual-rhs)),1e-12);self.assertTrue(solver.metadata['all_equation_rows_retained'])

    def test_indefinite_mixed_three_gauge_equations_remain_in_solution(self):
        rng=np.random.default_rng(194);q=np.linalg.qr(rng.normal(size=(7,7)))[0];r=q[:,:3];active=q[:,3:]
        a=active@np.diag([3.,5.,11.,13.])@active.T;b=active@rng.normal(size=(4,2));x=active@np.diag([1/3,1/5,1/11,1/13])@active.T@b
        m=np.block([[a,b],[b.T,b.T@x+np.diag([2.,4.])]])
        model=MixedMembraneEnergy(m,np.eye(9),[(i,1) for i in range(9)],np.arange(7),beam_bending_factor=.5,rigid_modes=r,input_identity='mixed_border_oracle')
        border=np.array([9,10,11,16,17,18]);solver=BorderedSparseFactor(model.stiffness,border)
        rhs=rng.normal(size=(19,3));actual=solver.solve(rhs);expected=np.linalg.solve(model.stiffness.toarray(),rhs)
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)
        self.assertLess(np.max(abs(model.stiffness@actual-rhs)),1e-11)
        # Nonzero loads on the constraint rows are deliberately retained.
        self.assertGreater(np.max(abs(rhs[16:])),.1)

    def test_invalid_border_nonfinite_rhs_and_singular_system_refused(self):
        for border in (np.array([],int),np.array([1,1]),np.array([-1]),np.array([4]),np.array([1.])):
            with self.assertRaises(ValueError):BorderedSparseFactor(np.eye(4),border)
        solver=BorderedSparseFactor(np.eye(4),np.array([1]))
        for rhs in (np.zeros(3),np.array([0.,float('nan'),0.,0.])):
            with self.assertRaises(ValueError):solver.solve(rhs)
        with self.assertRaises(ValueError):BorderedSparseFactor(np.zeros((2,2)),np.array([0,1]))
        with self.assertRaises(RuntimeError):BorderedSparseFactor(np.zeros((3,3)),np.array([0]))


if __name__=='__main__':unittest.main()
