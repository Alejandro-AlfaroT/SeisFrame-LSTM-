"""Independent spring/work/affine examples for exact static condensation."""
import unittest
import numpy as np
from scipy import sparse
from Design.SMRF_Static_Substructure import condense,AffineKinematics


class StaticCondensationTests(unittest.TestCase):
    def test_explicit_interior_border_retains_equations_loads_and_identity(self):
        p=np.array([[2.,1.,0.,0.,1.],[0.,3.,1.,1.,0.],[1.,0.,2.,0.,1.],[0.,1.,0.,4.,1.],[1.,0.,1.,0.,3.]])
        k=p.T@p;ids=['a','b','c','d','e'];loads=np.arange(10.).reshape(5,2)
        plain=condense(k,ids,['a','e'],loads,input_identity='same_physical_system')
        bounded=condense(k,ids,['a','e'],loads,input_identity='same_physical_system',interior_border_dofs=['c'])
        np.testing.assert_allclose(bounded.stiffness,plain.stiffness,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(bounded.loads,plain.loads,rtol=1e-13,atol=1e-13)
        ub=np.array([[.1,.2],[-.3,.4]])
        np.testing.assert_allclose(bounded.recover(ub),plain.recover(ub),rtol=1e-13,atol=1e-13)
        self.assertLess(bounded.residuals(ub)['maximum_interior_force_residual'],1e-12)
        record=bounded.record();self.assertEqual(record['interior_border_dofs'],['c'])
        self.assertGreater(record['factor_storage_bytes'],bounded._factor.influence.nbytes)
        self.assertNotEqual(record['identity_sha256'],plain.record()['identity_sha256'])
        self.assertNotIn('interior_border_dofs',plain.record())
        for border in (['a'],['absent'],['c','c']):
            with self.assertRaises(ValueError):condense(k,ids,['a','e'],loads,input_identity='invalid',interior_border_dofs=border)

    def test_residual_refinement_corrects_original_equation_without_relaxing_guard(self):
        sub=condense([[4.,-1.],[-1.,3.]],['a','b'],['a'],[0.,1.],input_identity='injected roundoff')
        exact=sub._factor
        class PerturbedFactor:
            L=exact.L;U=exact.U
            def solve(self,rhs):return exact.solve(rhs)*(1+1e-4)
        sub._factor=PerturbedFactor()
        result=sub.recover([.2])
        np.testing.assert_allclose(result,[.2,.4],atol=1e-12)
        self.assertGreater(sub.record()['same_factor_residual_corrections'],0)
        self.assertLessEqual(sub.record()['maximum_factor_backward_error'],1e-9)
        class WrongFactor:
            def solve(self,rhs):return np.zeros_like(rhs)
        sub._factor=WrongFactor()
        with self.assertRaisesRegex(ValueError,'equation residual check'):
            sub.recover([.2])

    def test_two_series_springs_and_internal_point_load(self):
        # k1=2, k2=3: kseries=6/5. Interior force10 splits6/4.
        k=np.array([[2.,-2.,0.],[-2.,5.,-3.],[0.,-3.,3.]])
        sub=condense(k,['left','inside','right'],['right','left'],[0.,10.,0.],input_identity={'source':'hand spring chain'})
        np.testing.assert_allclose(sub.stiffness,1.2*np.array([[1.,-1.],[-1.,1.]]))
        np.testing.assert_allclose(sub.loads,[6.,4.])
        np.testing.assert_allclose(sub.recover([2.,1.]),[1.,3.6,2.])
        check=sub.residuals([2.,1.])
        self.assertLess(check['maximum_interior_force_residual'],1e-12)
        self.assertLess(check['maximum_boundary_reaction_residual'],1e-12)
        self.assertLess(check['maximum_stationary_potential_energy_residual'],1e-12)
        np.testing.assert_allclose(sub.stiffness@np.ones(2),0.,atol=1e-14)
        self.assertEqual(sub.record()['physical_dof_ids'],['left','inside','right'])
        self.assertEqual(sub.record()['boundary_dof_ids'],['right','left'])

    def test_multiple_load_rhs_cached_factor_matches_independent_dense_solve(self):
        # Deterministic nonsingular coupled system. Dense full solve is an
        # independent oracle, not the same block elimination routine.
        a=np.array([[4.,1.,2.,0.],[1.,3.,0.,1.],[2.,0.,5.,1.],[0.,1.,1.,3.]])
        loads=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
        ids=[(7,1),(7,2),(8,1),(8,2)]
        sub=condense(sparse.csr_matrix(a),ids,[ids[3],ids[0]],loads,input_identity='independent dense oracle',block_size=1)
        ub=np.linalg.solve(sub.stiffness,sub.loads)
        np.testing.assert_allclose(sub.recover(ub),np.linalg.solve(a,loads),rtol=1e-13,atol=1e-13)
        other=2*loads[:,0]-loads[:,1]
        other_ub=np.linalg.solve(sub.stiffness,sub.condense_loads(other))
        np.testing.assert_allclose(sub.recover(other_ub,other),np.linalg.solve(a,other),atol=1e-13)
        self.assertEqual(sub.record()['factorizations'],1)
        self.assertGreater(sub.record()['factor_solve_batches'],3)

    def test_no_interior_dofs_keeps_exact_order_and_loads(self):
        k=np.array([[3.,1.],[1.,4.]])
        sub=condense(k,['a','b'],['b','a'],[2.,3.],input_identity='all retained')
        np.testing.assert_array_equal(sub.stiffness,[[4.,1.],[1.,3.]])
        np.testing.assert_array_equal(sub.loads,[3.,2.])
        np.testing.assert_array_equal(sub.recover([.2,.1]),[.1,.2])
        self.assertEqual(sub.record()['factorizations'],0)

    def test_unsupported_or_asymmetric_system_is_refused(self):
        with self.assertRaisesRegex(ValueError,'singular'):condense(np.diag([1.,0.]),['a','b'],['a'],[0.,0.],input_identity='singular')
        with self.assertRaisesRegex(ValueError,'symmetric'):condense([[1.,2.],[0.,1.]],['a','b'],['a'],[0.,0.],input_identity='asymmetric')
        with self.assertRaisesRegex(ValueError,'absent'):condense(np.eye(2),['a','b'],['c'],[0.,0.],input_identity='missing')
        with self.assertRaisesRegex(ValueError,'unique'):condense(np.eye(2),['a','a'],['a'],[0.,0.],input_identity='duplicate')

    def test_explicit_rigid_motion_transform_preserves_virtual_work(self):
        # Two spring attachment points at y=+-2 move ux=U-y*theta.
        # Physical forces [3,5] therefore give [8,4] generalized load.
        t=np.array([[1.,-2.],[1.,2.]])
        mapped=AffineKinematics(np.diag([2.,3.]),[3.,5.],['left.ux','right.ux'],['Ux','Rz'],t,input_identity='explicit rigid arm')
        np.testing.assert_allclose(mapped.stiffness.toarray(),[[5.,2.],[2.,20.]])
        np.testing.assert_allclose(mapped.loads,[8.,4.])
        q=np.array([.1,.2]);dq=np.array([-.3,.4])
        forces=mapped.forces(q)
        reduced=mapped.stiffness@q-mapped.loads
        self.assertAlmostEqual(float((t@dq)@forces),float(dq@reduced),places=13)
        np.testing.assert_allclose(mapped.recover(q),[-.3,.5])

    def test_nonzero_support_lift_has_exact_load_correction(self):
        k=np.array([[2.,-2.],[-2.,2.]])
        mapped=AffineKinematics(k,[0.,6.],['fixed','free'],['free'],[[0.],[1.]],lift=[1.,0.],input_identity='support settlement')
        np.testing.assert_allclose(mapped.loads,[8.])
        q=np.linalg.solve(mapped.stiffness.toarray(),mapped.loads)
        np.testing.assert_allclose(mapped.recover(q),[1.,4.])
        np.testing.assert_allclose(mapped.forces(q),[-6.,0.])

    def test_identity_changes_for_load_and_boundary_changes(self):
        k=np.array([[2.,-1.],[-1.,2.]])
        a=condense(k,['a','b'],['a'],[0.,1.],input_identity='same assembly')
        b=condense(k,['a','b'],['a'],[0.,2.],input_identity='same assembly')
        c=condense(k,['a','b'],['b'],[0.,1.],input_identity='same assembly')
        self.assertNotEqual(a.record()['identity_sha256'],b.record()['identity_sha256'])
        self.assertNotEqual(a.record()['identity_sha256'],c.record()['identity_sha256'])


if __name__=='__main__':unittest.main()
