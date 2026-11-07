"""Independent reduced-energy oracles for the optional mixed numerical form."""
import unittest
import numpy as np
from Design.SMRF_Mixed_Membrane_Energy import MixedMembraneEnergy
from Design.SMRF_Floor_Substructure import GeneralizedFloorStiffness
from Design.SMRF_Beam_Shell_Rotation_Dofs import retain_independent_beam_rotations
from Design.SMRF_Static_Substructure import condense
from test_floor_generalized_stiffness import floor_fixture


def eliminate(matrix,n):
    a=matrix.toarray()
    return a[:n,:n] if n==len(a) else a[:n,:n]-a[:n,n:]@np.linalg.solve(a[n:,n:],a[n:,:n])


class MixedEnergy(unittest.TestCase):
    def test_shift_coordinates_eliminate_dense_membrane_cross_layer_coupling(self):
        floor=floor_fixture();retain_independent_beam_rotations(floor);m=floor.membrane_dofs
        coords=np.asarray([floor.node_positions[n] for n in floor.node_ids]);r=np.zeros((len(m),3));n=3*len(coords)
        r[:n:3,0]=1.;r[1:n:3,1]=1.;r[:n:3,2]=-coords[:,1];r[1:n:3,2]=coords[:,0];r[2:n:3,2]=1.;r[n:,2]=1.
        other=.35*floor.parts['web_bending']+.2*floor.parts['slab_bending']+floor.parts['shear']+.1*floor.parts['web_torsion']
        kwargs=dict(beam_bending_factor=.35,rigid_modes=r,input_identity='coordinate_change_only')
        original=MixedMembraneEnergy(floor.parts['membrane'],other,floor.dof_ids,m,**kwargs)
        shift=MixedMembraneEnergy(floor.parts['membrane'],other,floor.dof_ids,m,**kwargs,auxiliary_coordinates='relaxation_shift')
        expected=eliminate(original.stiffness,original.n);actual=eliminate(shift.stiffness,shift.n)
        self.assertLess(np.max(abs(actual-expected))/np.max(abs(expected)),2e-12)
        self.assertLess(shift.C.nnz,original.C.nnz);self.assertEqual(shift.C[m].nnz,0)
        loads=floor.loads(-.002)['loads'];boundary=floor.boundary_dofs
        subs=[condense(v.stiffness,v.dof_ids,boundary,v.loads(loads),input_identity='same_physics') for v in (original,shift)]
        solutions=[s.recover(np.zeros(len(boundary))) for s in subs]
        np.testing.assert_allclose(solutions[0][:original.n],solutions[1][:shift.n],rtol=1e-8,atol=1e-12)
        fields=[v.fields(u) for v,u in zip((original,shift),solutions)]
        for row in floor.elements:
            t=row['transform'];ii=row['indices'];part=row['parts']['membrane']
            actions=[part@(t@um[ii]+.35*(t@bu[ii])) for um,bu in fields]
            np.testing.assert_allclose(*actions,rtol=1e-7,atol=1e-8)
        self.assertLess(shift.audit(solutions[1])['auxiliary_stationarity_relative_error'],1e-9)
        self.assertLess(shift.audit(solutions[1])['mean_constraint_relative_error'],1e-9)
        gross=MixedMembraneEnergy(floor.parts['membrane'],other,floor.dof_ids,m,**{**kwargs,'beam_bending_factor':1.},auxiliary_coordinates='relaxation_shift')
        u=np.random.default_rng(196).normal(size=gross.n);um,bu=gross.fields(u)
        np.testing.assert_array_equal(um+bu,u)
        np.testing.assert_allclose(gross.physical_action(u),(floor.parts['membrane']+other)@u,rtol=1e-12,atol=1e-8)

    def test_independent_dense_energy_and_stationary_equations(self):
        rng=np.random.default_rng(191);p=rng.normal(size=(8,5));a=p.T@p+np.eye(5);b=rng.normal(size=(5,3))
        c=np.diag([2.,3.,7.]);d=c+b.T@np.linalg.solve(a,b);m=np.block([[a,b],[b.T,d]]);o=np.eye(8)*.13
        for beta in (.35,.5,1.):
            model=MixedMembraneEnergy(m,o,[(i,1) for i in range(8)],np.arange(5),beam_bending_factor=beta,
                rigid_modes=np.zeros((5,0)),input_identity='independent_dense_fixture')
            expected=m+o;expected[5:,5:]+=(beta-1)*c
            np.testing.assert_allclose(eliminate(model.stiffness,8),expected,rtol=1e-13,atol=1e-13)
            rhs=rng.normal(size=8);v=np.linalg.solve(model.stiffness.toarray(),model.loads(rhs));u=v[:8]
            np.testing.assert_allclose(expected@u,rhs,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(model.physical_action(v),rhs,rtol=1e-12,atol=1e-12)
            self.assertLess(model.audit(v)['auxiliary_stationarity_relative_error'],1e-12)
            self.assertFalse(model.metadata['qualification_asserted'])

    def test_declared_nullspace_gauges_do_not_add_physical_restraints(self):
        rng=np.random.default_rng(192);q=np.linalg.qr(rng.normal(size=(7,7)))[0];r=q[:,:3];active=q[:,3:]
        a=active@np.diag([3.,5.,11.,13.])@active.T;b=active@rng.normal(size=(4,2));x=active@np.diag([1/3,1/5,1/11,1/13])@active.T@b
        c=np.diag([2.,4.]);m=np.block([[a,b],[b.T,b.T@x+c]]);o=np.eye(9)
        model=MixedMembraneEnergy(m,o,[(i,1) for i in range(9)],np.arange(7),beam_bending_factor=.35,rigid_modes=r,input_identity='three_null_modes')
        expected=m+o;expected[7:,7:]-=.65*c
        np.testing.assert_allclose(eliminate(model.stiffness,9),expected,rtol=2e-13,atol=2e-13)
        v=np.linalg.solve(model.stiffness.toarray(),model.loads(rng.normal(size=9)))
        self.assertLess(model.audit(v)['mean_constraint_relative_error'],1e-12)
        self.assertLess(model.audit(v)['unsubtracted_gauge_reaction_max'],1e-12)

    def test_physical_floor_condensation_and_element_energy_match_old_form(self):
        floor=floor_fixture();retain_independent_beam_rotations(floor);m=floor.membrane_dofs
        coords=np.asarray([floor.node_positions[n] for n in floor.node_ids]);r=np.zeros((len(m),3));n=3*len(coords)
        r[:n:3,0]=1.;r[1:n:3,1]=1.;r[:n:3,2]=-coords[:,1];r[1:n:3,2]=coords[:,0];r[2:n:3,2]=1.;r[n:,2]=1.
        other=.35*floor.parts['web_bending']+.2*floor.parts['slab_bending']+floor.parts['shear']+.1*floor.parts['web_torsion']
        model=MixedMembraneEnergy(floor.parts['membrane'],other,floor.dof_ids,m,beam_bending_factor=.35,rigid_modes=r,input_identity='same_physical_floor')
        old=GeneralizedFloorStiffness(floor,beam_bending_factor=.35,slab_bending_factors=.2,web_torsion_factor=.1,
            stiffness_basis='test only',input_identity='same_physical_floor')
        expected=old.stiffness.toarray();error=np.max(abs(eliminate(model.stiffness,len(floor.dof_ids))-expected))/np.max(abs(expected))
        self.assertLess(error,2e-12)
        loads=floor.loads(-.002);a=condense(model.stiffness,model.dof_ids,floor.boundary_dofs,model.loads(loads['loads']),input_identity='mixed_test')
        b=condense(old.stiffness,floor.dof_ids,floor.boundary_dofs,loads['loads'],input_identity='old_test')
        np.testing.assert_allclose(a.stiffness,b.stiffness,rtol=2e-10,atol=1e-7)
        ub=np.zeros(len(floor.boundary_dofs));v=a.recover(ub);u=b.recover(ub)
        np.testing.assert_allclose(v[:model.n],u,rtol=1e-8,atol=1e-12)
        um,bu=model.fields(v);original_membrane,original_bending=old.fields(u)
        for row in floor.elements:
            t=row['transform'];ii=row['indices'];part=row['parts']['membrane']
            np.testing.assert_allclose(part@(t@um[ii]+.35*(t@bu[ii])),
                part@(t@original_membrane[ii]+.35*(t@original_bending[ii])),rtol=1e-7,atol=1e-8)
        self.assertLess(model.audit(v)['auxiliary_stationarity_relative_error'],1e-9)
        self.assertLess(model.audit(v)['mean_constraint_relative_error'],1e-9)

    def test_invented_gauges_and_nonphysical_inputs_refused(self):
        kwargs=dict(membrane=np.eye(4),other=np.eye(4),physical_dof_ids=[(i,1) for i in range(4)],
            membrane_indices=np.arange(2),beam_bending_factor=.5,rigid_modes=np.zeros((2,0)),input_identity='refusal')
        for update in (dict(beam_bending_factor=0.),dict(rigid_modes=np.ones((2,1))),dict(membrane_indices=np.array([0,0])),dict(tolerance=1e-8),dict(auxiliary_coordinates='silent_other')):
            with self.assertRaises(ValueError):MixedMembraneEnergy(**{**kwargs,**update})
        model=MixedMembraneEnergy(**kwargs)
        with self.assertRaises(ValueError):model.loads(np.zeros(3))
        with self.assertRaises(ValueError):model.coordinates(np.zeros(4))


if __name__=='__main__':unittest.main()
