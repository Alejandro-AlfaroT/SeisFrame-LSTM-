"""Independent column mechanics and supplied-interface assembly checks."""
import unittest
import copy
from unittest.mock import patch
import numpy as np
from Design.SMRF_Substructure_Frame import ElasticColumn,SubstructureFrame


def definition(**updates):
    return dict(id='c',i=1,j=2,A=10.,E=200.,G=80.,J=15.,Iy=20.,Iz=30.,vecxz=[1.,0.,0.],**updates)


class ElasticColumnTests(unittest.TestCase):
    def test_rotated_three_dimensional_stiffness_and_rigid_body_modes(self):
        column=ElasticColumn(definition(),[0.,0.,0.],[0.,0.,10.])
        np.testing.assert_allclose(column.rotation,[[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])
        self.assertAlmostEqual(column.stiffness[0,0],12*200*20/10**3)
        self.assertAlmostEqual(column.stiffness[1,1],12*200*30/10**3)
        for translation in np.eye(3):
            u=np.zeros(12);u[:3]=u[6:9]=translation
            np.testing.assert_allclose(column.stiffness@u,0.,atol=1e-12)
        for omega in np.eye(3):
            u=np.zeros(12);u[3:6]=u[9:12]=omega;u[6:9]=np.cross(omega,[0.,0.,10.])
            np.testing.assert_allclose(column.stiffness@u,0.,atol=1e-12)

    def test_native_end_force_and_distributed_weight_are_not_average_axial(self):
        column=ElasticColumn(definition(),[0.,0.,0.],[0.,0.,10.])
        u=np.zeros(12);u[6]=.1;u[8]=-.025
        result=column.response(u,-.4,pdelta=True)
        # EA/L*du=-5, while basic axial includes -wxL/2=+2.
        self.assertAlmostEqual(result['pdelta_basic_axial_tension_positive'],-3.)
        np.testing.assert_allclose(result['local_force'][[0,6]],[7.,-3.])
        self.assertAlmostEqual(result['geometric_global_force'][6],-.03)
        self.assertAlmostEqual(result['geometric_global_force'][0],.03)

    def test_native_release_removes_only_named_end_moment(self):
        column=ElasticColumn(definition(releasey=2),[0.,0.,0.],[0.,0.,10.])
        u=np.linspace(.001,.012,12)
        result=column.response(u,pdelta=False)
        self.assertAlmostEqual(result['local_force'][10],0.)
        self.assertNotAlmostEqual(result['local_force'][4],0.)
        with self.assertRaisesRegex(ValueError,'codes'):ElasticColumn(definition(releasez=4),[0.,0.,0.],[0.,0.,10.])


class GlobalFrameTests(unittest.TestCase):
    def test_saved_force_replay_rejects_changed_raw_force_and_displacement(self):
        loads=np.zeros(12);loads[6]=2.;loads[8]=-3.;model=self.model(loads);record=model.solve(pdelta=True)
        with patch.object(model,'solve',side_effect=AssertionError('Saved replay cannot solve')):
            result=model.verify_saved(record)
        self.assertTrue(result['equilibrium_verified']);self.assertEqual(result['frame_solves'],0)
        for kind in ('displacement','column_force','reaction','operator'):
            bad=copy.deepcopy(record)
            if kind=='displacement':bad['cases'][0]['physical_displacements'][6]+=.01
            elif kind=='column_force':bad['cases'][0]['columns']['c']['local_force'][6]+=.01
            elif kind=='reaction':bad['cases'][0]['physical_residual_or_reaction'][0]+=.01
            else:bad['identity_sha256']='changed'
            with self.subTest(kind=kind),self.assertRaises(ValueError):model.verify_saved(bad)

    def model(self,loads,**options):
        return SubstructureFrame([{'id':1,'xyz':[0.,0.,0.]},{'id':2,'xyz':[0.,0.,10.]}],[],[definition(axial_load_kip_per_in=-.4)],
                                 supports=[(1,d) for d in range(1,7)],loads=loads,input_identity='independent cantilever',**options)

    def test_cantilever_native_pdelta_closed_form_and_local_global_force_difference(self):
        loads=np.zeros(12);loads[6]=2.;loads[8]=-3.
        model=self.model(loads)
        linear=model.solve(pdelta=False)['cases'][0];second=model.solve(pdelta=True)['cases'][0]
        self.assertEqual(second['status'],'equilibrated')
        self.assertAlmostEqual(linear['physical_displacements'][6],2/(3*200*20/10**3),places=12)
        self.assertAlmostEqual(second['physical_displacements'][6],2/(3*200*20/10**3-3/10),places=12)
        self.assertAlmostEqual(second['physical_displacements'][8],-.025,places=12)
        forces=second['columns']['c']
        self.assertAlmostEqual(forces['global_resisting_force'][6],2.,places=11)
        self.assertGreater(forces['local_force'][8],2.) # native local shear excludes geometric addition
        self.assertLess(second['maximum_free_equilibrium_residual'],1e-10)

    def test_multiple_rhs_loads_and_single_linear_factor(self):
        loads=np.zeros((12,2));loads[6]=[2.,-4.];loads[8]=[-3.,-6.]
        result=self.model(loads).solve(pdelta=True)
        self.assertEqual(result['status'],'equilibrated')
        for case,h,p in zip(result['cases'],[2.,-4.],[3.,6.]):
            self.assertAlmostEqual(case['physical_displacements'][6],h/(12-p/10),places=11)

    def test_supplied_floor_interface_is_added_with_its_off_diagonal_coupling(self):
        nodes=[{'id':1,'xyz':[0.,0.,0.]},{'id':2,'xyz':[0.,0.,10.]}]
        base=[(1,d) for d in range(1,7)]
        interface={'id':'coupled boundary','dof_ids':[(2,1),(2,3)],'stiffness':[[2.,.7],[.7,3.]],'loads':[2.,-5.],'input_identity':'dense oracle floor'}
        frame=SubstructureFrame(nodes,[interface],[definition()],supports=base,input_identity='exact finite floor interface')
        result=frame.solve(pdelta=False)['cases'][0]
        k=frame.stiffness.toarray();direct=np.linalg.solve(k[6:,6:],frame.loads[6:,0])
        np.testing.assert_allclose(result['physical_displacements'][6:],direct,rtol=1e-13)
        self.assertNotEqual(direct[0],2/14) # the floor vertical/horizontal coupling matters

    def test_exact_nonlinear_residual_jacobian_matches_centered_difference(self):
        loads=np.zeros(12);loads[6]=2.;loads[8]=-3.;frame=self.model(loads)
        u=np.linspace(-.001,.002,12);force,jac=frame._geometric(u,0,tangent=True)
        direction=np.linspace(-.3,.4,12);eps=1e-7
        derivative=(frame._geometric(u+eps*direction,0,tangent=False)[0]-frame._geometric(u-eps*direction,0,tangent=False)[0])/(2*eps)
        np.testing.assert_allclose(jac@direction,derivative,rtol=1e-9,atol=1e-9)

    def test_missing_constraints_are_refused(self):
        with self.assertRaisesRegex(ValueError,'singular'):
            SubstructureFrame([{'id':1,'xyz':[0.,0.,0.]},{'id':2,'xyz':[0.,0.,10.]}],[],[definition()],input_identity='unsupported')


if __name__=='__main__':unittest.main()
