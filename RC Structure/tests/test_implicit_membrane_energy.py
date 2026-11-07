"""Independent dense energies and full physical Dirichlet residual oracles."""
import unittest
import numpy as np
from Design.SMRF_Implicit_Membrane_Energy import ImplicitMembraneEnergy
from Design.SMRF_Physical_Residual_Solve import solve_physical_dirichlet,PhysicalResidualConvergenceError
from Design.SMRF_Floor_Substructure import GeneralizedFloorStiffness
from Design.SMRF_Beam_Shell_Rotation_Dofs import retain_independent_beam_rotations
from test_floor_generalized_stiffness import floor_fixture


def physical_fixture(beta=.35):
    floor=floor_fixture();retain_independent_beam_rotations(floor);m=floor.membrane_dofs
    coords=np.asarray([floor.node_positions[n] for n in floor.node_ids]);r=np.zeros((len(m),3));n=3*len(coords)
    r[:n:3,0]=1;r[1:n:3,1]=1;r[:n:3,2]=-coords[:,1];r[1:n:3,2]=coords[:,0];r[2:n:3,2]=1;r[n:,2]=1
    first=int(np.lexsort((coords[:,1],coords[:,0]))[0]);far=int(np.argmax(coords[:,0]));gauges=[3*first,3*first+1,3*far+1]
    other=beta*floor.parts['web_bending']+.2*floor.parts['slab_bending']+floor.parts['shear']+.1*floor.parts['web_torsion']
    model=ImplicitMembraneEnergy(floor.parts['membrane'],other,m,beam_bending_factor=beta,rigid_modes=r,gauge_indices=gauges,input_identity='independent_floor')
    return floor,model


class ImplicitEnergy(unittest.TestCase):
    def test_dense_elimination_and_local_virtual_work(self):
        for beta in (.35,.5,1.):
            floor,model=physical_fixture(beta);m,b=model.m,model.b;M=model.M.toarray();A=M[np.ix_(m,m)];B=M[np.ix_(m,b)]
            x=np.zeros_like(B);x[model.free]=np.linalg.solve(A[np.ix_(model.free,model.free)],B[model.free])
            dense=M+model.O.toarray();dense[np.ix_(b,b)]+=(beta-1)*(M[np.ix_(b,b)]-B.T@x)
            u=np.random.default_rng(211).normal(size=len(floor.dof_ids))*1e-4;actual=model.apply(u)
            np.testing.assert_allclose(actual,dense@u,rtol=1e-9,atol=1e-8)
            um,ub=model.fields(u);local=np.zeros_like(u)
            for row in floor.elements:
                t=row['transform'];ind=row['indices'];ua=t@u[ind]
                f=row['parts']['membrane']@(t@um[ind]+beta*(t@ub[ind]))
                if row['kind']=='slab':f+=.2*row['parts']['slab_bending']@ua+row['parts']['shear']@ua
                else:f+=beta*row['parts']['web_bending']@ua+.1*row['parts']['web_torsion']@ua
                np.add.at(local,ind,t.T@f)
            np.testing.assert_allclose(actual,local,rtol=1e-9,atol=1e-8)
            self.assertLess(model.last_relaxation['relative_full_equation_residual'],1e-9)
            self.assertFalse(hasattr(model,'stiffness'))

    def test_actual_floor_dirichlet_against_full_dense_operator(self):
        floor,model=physical_fixture();legacy=GeneralizedFloorStiffness(floor,beam_bending_factor=.35,slab_bending_factors=.2,
            web_torsion_factor=.1,stiffness_basis='same independent physical model',input_identity='same')
        n=len(floor.dof_ids);indices=[floor.dof_ids.index(d) for d in floor.boundary_dofs];interior=np.setdiff1d(np.arange(n),indices)
        ub=np.random.default_rng(212).normal(size=len(indices))*1e-5;loads=floor.loads(-.002)['loads']
        actual=solve_physical_dirichlet(model.apply,legacy.stiffness,indices,ub,loads,input_identity='actual_fixture')
        # Independent dense analytical Schur elimination, not basis calls to apply.
        m,b=model.m,model.b;M=model.M.toarray();A=M[np.ix_(m,m)];B=M[np.ix_(m,b)];x=np.zeros_like(B)
        x[model.free]=np.linalg.solve(A[np.ix_(model.free,model.free)],B[model.free]);dense=M+model.O.toarray()
        dense[np.ix_(b,b)]+=(model.beta-1)*(M[np.ix_(b,b)]-B.T@x)
        expected=np.zeros(n);expected[indices]=ub;expected[interior]=np.linalg.solve(dense[np.ix_(interior,interior)],loads[interior]-dense[np.ix_(interior,indices)]@ub)
        np.testing.assert_allclose(actual['displacements'],expected,rtol=1e-7,atol=1e-10)
        self.assertTrue(actual['audit']['passed']);self.assertLessEqual(actual['audit']['relative_interior_force_residual'],1e-9)
        self.assertFalse(actual['audit']['preconditioner_is_authoritative'])

    def test_incomplete_inner_equilibrium_and_false_gauges_refused(self):
        floor,model=physical_fixture()
        from types import SimpleNamespace
        model.factor=SimpleNamespace(solve=lambda rhs:np.zeros_like(rhs))
        with self.assertRaisesRegex(ValueError,'equilibrium'):
            model.apply(np.random.default_rng(213).normal(size=len(floor.dof_ids)))
        with self.assertRaises(ValueError):
            ImplicitMembraneEnergy(np.eye(4),np.eye(4),[0,1],beam_bending_factor=.5,rigid_modes=np.ones((2,1)),gauge_indices=[0],input_identity='false')


class PhysicalRefinement(unittest.TestCase):
    def test_corrected_preconditioner_is_not_equation_authority(self):
        rng=np.random.default_rng(214);x=rng.normal(size=(12,12));a=x.T@x+np.eye(12)
        pre=a+np.diag(np.linspace(.0001,.001,12));f=rng.normal(size=12);b=np.array([2,9]);ub=np.array([.2,-.1]);i=np.setdiff1d(np.arange(12),b)
        result=solve_physical_dirichlet(lambda u:a@u,pre,b,ub,f,input_identity='known_full_matrix')
        expected=np.zeros(12);expected[b]=ub;expected[i]=np.linalg.solve(a[np.ix_(i,i)],f[i]-a[np.ix_(i,b)]@ub)
        np.testing.assert_allclose(result['displacements'],expected,rtol=1e-10,atol=1e-11)
        self.assertLess(result['audit']['absolute_interior_force_residual'],result['audit']['initial_absolute_interior_force_residual'])
        self.assertGreater(result['audit']['correction_solves'],0)

    def test_exhaustion_and_large_coefficients_do_not_create_acceptance(self):
        for pre in (2*np.eye(3),1e20*np.eye(3)):
            with self.assertRaises(PhysicalResidualConvergenceError) as caught:
                solve_physical_dirichlet(lambda u:u,pre,[0],[0.],[0.,1.,1.],input_identity='deliberately_inadequate')
            self.assertFalse(caught.exception.audit['passed']);self.assertLessEqual(caught.exception.audit['correction_solves'],3)
            self.assertEqual(caught.exception.audit['force_scale'],1.)
            self.assertEqual(caught.exception.displacements.shape,(3,))
            np.testing.assert_array_equal(caught.exception.physical_action,caught.exception.displacements)

    def test_bad_shape_limits_and_nonfinite_action_refused(self):
        for kwargs in (dict(tolerance=1e-8),dict(maximum_steps=4),dict(maximum_steps=-1),dict(input_identity='')):
            with self.assertRaises(ValueError):solve_physical_dirichlet(lambda u:u,np.eye(3),[0],[0.],np.zeros(3),**{'input_identity':'test',**kwargs})
        with self.assertRaises(ValueError):solve_physical_dirichlet(lambda u:np.full(3,np.nan),np.eye(3),[0],[0.],np.zeros(3),input_identity='test')


if __name__=='__main__':unittest.main()
