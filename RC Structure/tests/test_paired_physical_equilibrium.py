import unittest
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from Design.SMRF_Paired_Physical_Equilibrium import PairedMembraneEnergy,remove_rigid_pair,solve_paired_dirichlet,PairedPhysicalConvergenceError
from Design.SMRF_Paired_Arithmetic import paired_sparse_action
from test_implicit_membrane_energy import physical_fixture

class PairedPhysicalEquilibrium(unittest.TestCase):
    def test_boundary_coupling_needs_low_displacement_correction(self):
        a=sparse.csr_matrix([[1e16,-1e16,0.],[-1e16,1e16,0.],[0.,0.,1.]])
        result=solve_paired_dirichlet(lambda h,l:paired_sparse_action(a,h,l),a,[0],[1.],[0.,1.,2.],input_identity='independent_lost_update')
        self.assertEqual(result['displacements_hi'][1],1.)
        self.assertNotEqual(result['displacements_lo'][1],0.)
        self.assertTrue(result['audit']['passed'])
        np.testing.assert_allclose(result['physical_action_hi'][1:]+result['physical_action_lo'][1:],[1.,2.],rtol=0,atol=1e-12)
        self.assertEqual(result['displacements_lo'][0],0.)

    def test_dense_schur_oracle_and_paired_field_energy(self):
        for beta in (.35,.5):
            floor,old=physical_fixture(beta);mh=old.M;ml=mh*1e-14;oh=old.O;ol=oh*1e-14
            coords=np.asarray([floor.node_positions[n] for n in floor.node_ids]);n=3*len(coords);r=np.zeros((len(old.m),3))
            r[:n:3,0]=1;r[1:n:3,1]=1;r[:n:3,2]=-coords[:,1];r[1:n:3,2]=coords[:,0];r[2:n:3,2]=1;r[n:,2]=1
            model=PairedMembraneEnergy(mh,ml,oh,ol,old.m,beam_bending_factor=beta,rigid_modes=r,
                gauge_indices=old.metadata['gauge_indices'],input_identity='independent_dense_pair')
            m,b=model.m,model.b;M=(mh+ml).toarray();A=M[np.ix_(m,m)];B=M[np.ix_(m,b)];x=np.zeros_like(B)
            x[model.free]=np.linalg.solve(A[np.ix_(model.free,model.free)],B[model.free])
            dense=M+(oh+ol).toarray();dense[np.ix_(b,b)]+=(beta-1)*(M[np.ix_(b,b)]-B.T@x)
            hi=np.random.default_rng(247).normal(size=len(dense))*1e-4;lo=np.spacing(hi)/4
            fh,fl=model.apply_pair(hi,lo);np.testing.assert_allclose(fh+fl,dense@(hi+lo),rtol=1e-8,atol=1e-8)
            indices=np.array([floor.dof_ids.index(d) for d in floor.boundary_dofs]);inside=np.setdiff1d(np.arange(len(dense)),indices)
            loads=floor.loads(-.002)['loads'];ub=np.zeros(len(indices))
            result=solve_paired_dirichlet(model.apply_pair,dense,indices,ub,loads,input_identity='same_dense_boundary')
            expected=np.linalg.solve(dense[np.ix_(inside,inside)],loads[inside])
            np.testing.assert_allclose((result['displacements_hi']+result['displacements_lo'])[inside],expected,rtol=1e-7,atol=1e-10)
            self.assertTrue(result['audit']['passed']);self.assertLess(model.last_relaxation['relative_full_equation_residual'],1e-9)
            with self.assertRaises(ValueError):model.apply(hi)
            with self.assertRaises(ValueError):model.fields(hi)

    def test_rigid_motion_preserves_small_low_field(self):
        points=np.array([[0.,0.,0.],[8.,0.,0.],[8.,6.,0.],[0.,6.,0.]])
        floor=SimpleNamespace(node_ids=list(range(4)),node_positions=dict(enumerate(points)),dof_ids=[(i,j) for i in range(4) for j in range(1,7)])
        rotation=np.array([.25,-.125,.5]);translation=np.array([4.,2.,8.]);motion=np.empty((4,6))
        motion[:,:3]=translation+np.cross(rotation,points);motion[:,3:]=rotation
        low=np.zeros(24);low[0]=2.**-60
        h,l,_=remove_rigid_pair(floor,motion.ravel(),low)
        np.testing.assert_array_equal(h+l,low)

    def test_incomplete_solve_keeps_pairs_and_boundary_is_checked(self):
        with self.assertRaises(PairedPhysicalConvergenceError) as caught:
            solve_paired_dirichlet(lambda h,l:(h,l),2*np.eye(3),[0],[0.],[0.,1.,1.],input_identity='inadequate')
        self.assertFalse(caught.exception.result['audit']['passed'])
        self.assertLessEqual(len(caught.exception.result['audit']['history']),3)
        self.assertIn('displacements_lo',caught.exception.result)
        with self.assertRaises(ValueError):solve_paired_dirichlet(lambda h,l:(h,l),np.eye(3),[0],[1.],np.zeros(3),initial_pair=(np.zeros(3),np.zeros(3)),input_identity='wrong_boundary')
        for kwargs in (dict(tolerance=1e-8),dict(maximum_steps=4)):
            with self.assertRaises(ValueError):solve_paired_dirichlet(lambda h,l:(h,l),np.eye(3),[0],[0.],np.zeros(3),input_identity='strict',**kwargs)

if __name__=='__main__':unittest.main()
