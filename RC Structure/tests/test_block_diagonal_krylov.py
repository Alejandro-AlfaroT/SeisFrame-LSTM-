"""Full dense and physical-floor checks for bounded two-block Krylov solves."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import block_diag
from Design.SMRF_Block_Diagonal_Krylov import BlockDiagonalKrylovFactor,KrylovConvergenceError
from Design.SMRF_Mixed_Membrane_Energy import MixedMembraneEnergy
from Design.SMRF_Static_Substructure import condense
from Design.SMRF_Beam_Shell_Rotation_Dofs import retain_independent_beam_rotations
from test_floor_generalized_stiffness import floor_fixture


def synthetic():
    rng=np.random.default_rng(199);p=rng.normal(size=(12,12));p=p.T@p+np.eye(12)
    q=rng.normal(size=(5,5));q=q.T@q+np.eye(5)
    ep,up=np.linalg.eigh(p);eq,uq=np.linalg.eigh(q);ph=(up*np.sqrt(ep))@up.T;qh=(uq*np.sqrt(eq))@uq.T
    s=rng.normal(size=(12,5));s/=np.linalg.norm(s,2);c=ph@s@qh
    return np.block([[p,c],[c.T,-q]]),block_diag((up/np.sqrt(ep))@up.T,(uq/np.sqrt(eq))@uq.T),rng.normal(size=(17,4))


class BlockKrylov(unittest.TestCase):
    def test_independent_dense_solution_and_preconditioned_spectrum(self):
        a,scale,rhs=synthetic();eigen=np.linalg.eigvalsh(scale@a@scale)
        self.assertGreaterEqual(min(abs(eigen)),1-1e-12);self.assertLessEqual(max(abs(eigen)),np.sqrt(2)+1e-12)
        solver=BlockDiagonalKrylovFactor(a,12);actual=solver.solve(rhs);expected=np.linalg.solve(a,rhs)
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11)
        self.assertTrue(all(r['accepted'] and r['iterations']<=80 for r in solver.solve_audits))
        self.assertLess(np.max(abs(a@actual-rhs)),1e-11)
        np.testing.assert_array_equal(solver.solve(np.zeros(17)),0.)

    def test_physical_stationary_floor_with_retained_mean_border(self):
        floor=floor_fixture();retain_independent_beam_rotations(floor);m=floor.membrane_dofs
        coords=np.asarray([floor.node_positions[n] for n in floor.node_ids]);r=np.zeros((len(m),3));n=3*len(coords)
        r[:n:3,0]=1.;r[1:n:3,1]=1.;r[:n:3,2]=-coords[:,1];r[1:n:3,2]=coords[:,0];r[2:n:3,2]=1.;r[n:,2]=1.
        other=.35*floor.parts['web_bending']+.2*floor.parts['slab_bending']+floor.parts['shear']+.1*floor.parts['web_torsion']
        model=MixedMembraneEnergy(floor.parts['membrane'],other,floor.dof_ids,m,beam_bending_factor=.35,rigid_modes=r,input_identity='actual_floor_krylov')
        first=int(np.lexsort((coords[:,1],coords[:,0]))[0]);far=int(np.argmax(coords[:,0]));indices=[model.n+3*first,model.n+3*first+1,model.n+3*far+1,*range(model.n+len(m),len(model.dof_ids))]
        border=[model.dof_ids[i] for i in indices];loads=model.loads(floor.loads(-.002)['loads']);positive=model.n-len(floor.boundary_dofs)
        reference=condense(model.stiffness,model.dof_ids,floor.boundary_dofs,loads,input_identity='same',interior_border_dofs=border)
        made=[]
        def factory(matrix):
            result=BlockDiagonalKrylovFactor(matrix,positive);made.append(result);return result
        with patch('Design.SMRF_Bordered_Sparse_Factor.splu',side_effect=factory):
            actual=condense(model.stiffness,model.dof_ids,floor.boundary_dofs,loads,input_identity='same',interior_border_dofs=border)
        ub=np.zeros(len(floor.boundary_dofs));a=actual.recover(ub);b=reference.recover(ub)
        np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-11)
        np.testing.assert_allclose(actual.stiffness,reference.stiffness,rtol=1e-8,atol=1e-6)
        self.assertLess(actual.residuals(ub)['factor_backward_error'],1e-9)
        self.assertGreater(actual.record()['factor_storage_bytes'],made[0].storage_bytes)
        self.assertNotIn('interior_core_solver',reference.record())
        self.assertEqual(actual.record()['interior_core_solver']['method'],made[0].metadata['method'])
        self.assertEqual(len(actual.record()['interior_core_solver']['source_sha256']),64)
        self.assertEqual(actual.record()['interior_core_solver']['relative_tolerance'],1e-13)
        self.assertEqual(len(made),1);self.assertTrue(all(r['accepted'] for r in made[0].solve_audits))

    def test_exhausted_iteration_cannot_be_accepted(self):
        a,_,rhs=synthetic();solver=BlockDiagonalKrylovFactor(a,12,maximum_iterations=1,restart=1)
        with self.assertRaisesRegex(KrylovConvergenceError,'did not converge') as caught:solver.solve(rhs[:,0])
        self.assertFalse(solver.solve_audits[-1]['accepted']);self.assertEqual(solver.solve_audits[-1]['iterations'],1)
        self.assertEqual(caught.exception.audit,solver.solve_audits[-1])

    def test_border_setup_exhaustion_is_not_structural_singularity(self):
        a,_,rhs=synthetic()
        # Two extra explicit equations: a retained boundary and a border row.
        # The 17-equation core is nonsingular, but one iteration is insufficient.
        full=np.zeros((19,19));full[:17,:17]=a
        full[:17,17]=rhs[:,0];full[17,:17]=rhs[:,0]
        full[17,17]=2.;full[18,18]=1.
        self.assertEqual(np.linalg.matrix_rank(full),19)
        def factory(matrix):return BlockDiagonalKrylovFactor(matrix,12,maximum_iterations=1,restart=1)
        with patch('Design.SMRF_Bordered_Sparse_Factor.splu',side_effect=factory):
            with self.assertRaises(KrylovConvergenceError) as caught:
                condense(full,list(range(19)),[18],np.zeros(19),input_identity='bounded_nonsingular',interior_border_dofs=[17])
        self.assertEqual(caught.exception.audit['iterations'],1)
        self.assertFalse(caught.exception.audit['accepted'])
        self.assertNotIn('singular',str(caught.exception))

    def test_invalid_bounds_blocks_and_rhs_refused(self):
        a,_,_=synthetic()
        for kwargs in (dict(positive_size=0),dict(relative_tolerance=1e-5),dict(maximum_iterations=0),dict(restart=81)):
            with self.assertRaises(ValueError):BlockDiagonalKrylovFactor(a,**{'positive_size':12,**kwargs})
        with self.assertRaises(ValueError):BlockDiagonalKrylovFactor(np.eye(5),2)
        solver=BlockDiagonalKrylovFactor(a,12)
        with self.assertRaises(ValueError):solver.solve(np.zeros(16))


if __name__=='__main__':unittest.main()
