"""Energy, physical-force and load-ownership checks on an actual floor mesh."""
import unittest
import numpy as np
from Design.SMRF_Floor_Substructure import FloorAssembly,GeneralizedFloorStiffness,rigid_arm_matrix
from Design.SMRF_Static_Substructure import condense


def floor_fixture():
    supports=[{'id':f'j{i}{j}','x':x,'y':y,'b_in':8.,'h_in':8.}
              for j,y in enumerate((0.,40.)) for i,x in enumerate((0.,60.))]
    beams=[]
    for axis,lines,end in (('x',(0.,40.),60.),('y',(0.,60.),40.)):
        for line in lines:
            beams.append({'id':f'{axis}{line}','axis':axis,'line':line,'start':0.,'end':end,
                          'b_in':4.,'h_in':12.,'E_ksi':4000.})
    return FloorAssembly([0.,4.,30.,56.,60.],[0.,4.,20.,36.,40.],floor_id='f1',
                         thickness_in=3.,elastic_modulus_ksi=4000.,supports=supports,beams=beams)


class GeneralizedFloorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.floor=floor_fixture()

    def stiffness(self,beta=.35,alpha=.2,torsion=.1):
        return GeneralizedFloorStiffness(self.floor,beam_bending_factor=beta,slab_bending_factors=alpha,
            web_torsion_factor=torsion,stiffness_basis='test prescribed sensitivity, no engineering assertion',
            input_identity='independent_algebra_fixture')

    def test_blocked_storage_and_physical_shift_match_independent_dense_solve(self):
        floor=self.floor;model=self.stiffness();membrane=floor.parts['membrane'].toarray()
        m,b=floor.membrane_dofs,floor.bending_dofs;free=model._relaxation_free
        a=membrane[np.ix_(m,m)];coupling=membrane[np.ix_(m,b)]
        # Dense LAPACK solve of all columns is independent of the bounded
        # sparse block construction and later single-physical-RHS application.
        x=np.zeros_like(coupling);x[free]=np.linalg.solve(a[np.ix_(free,free)],coupling[free])
        expected=membrane[np.ix_(b,b)]-coupling.T@x
        np.testing.assert_allclose(model.couple_stiffness.toarray(),expected,rtol=1e-9,atol=1e-8)
        u=np.random.default_rng(108).normal(size=len(floor.dof_ids))*.001
        um,ub=model.fields(u)
        np.testing.assert_allclose(um[m]-u[m],x@u[b],rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(um+ub,u,rtol=0,atol=1e-17)
        self.assertFalse(hasattr(model,'_x'))
        self.assertGreater(model.metadata['dense_influence_storage_avoided_bytes'],0)
        # A bad factor cannot manufacture a zero residual or borrow the
        # construction-time refinement audit for a changed physical motion.
        from types import SimpleNamespace
        model._relaxation_factor=SimpleNamespace(solve=lambda rhs:np.zeros_like(rhs))
        with self.assertRaisesRegex(ValueError,'Physical membrane relaxation equilibrium'):
            model.fields(u)

    def test_exact_decomposed_energy_and_physical_forces(self):
        floor=self.floor;model=self.stiffness();u=np.random.default_rng(20261103).normal(size=len(floor.dof_ids))*.001
        um,ub=model.fields(u)
        energy=um@(floor.parts['membrane']@um)+.35*(ub@(floor.parts['membrane']@ub))
        energy+=.35*(u@(floor.parts['web_bending']@u))+.2*(u@(floor.parts['slab_bending']@u))
        energy+=u@(floor.parts['shear']@u)+.1*(u@(floor.parts['web_torsion']@u))
        self.assertAlmostEqual((u@(model.stiffness@u))/energy,1.,places=11)
        spectrum=np.linalg.eigvalsh(model.couple_stiffness.toarray())
        self.assertGreaterEqual(spectrum.min(),-1e-9*max(1.,spectrum.max()))
        result=model.recover(u,floor.loads(-.002))
        self.assertLess(result['force_partition_relative_error'],1e-10)

    def test_recovery_ignores_rigid_translation_and_preserves_supplied_motion(self):
        floor=self.floor;model=self.stiffness()
        u=np.random.default_rng(63).normal(size=len(floor.dof_ids))*.0001
        moved=u.copy();moved.reshape(-1,6)[:,:3]+=[12.,-8.,3.]
        load=floor.loads(-.002)
        a=model.recover(u,load);b=model.recover(moved,load)
        self.assertLess(b['force_partition_relative_error'],1e-10)
        for row,left,right in zip(floor.elements,a['elements'],b['elements']):
            np.testing.assert_allclose(left['global_resisting_force'],right['global_resisting_force'],rtol=1e-8,atol=1e-7)
            np.testing.assert_allclose(right['displacements'],row['transform']@moved[row['indices']],rtol=0,atol=1e-13)
            if row['kind']=='slab':
                np.testing.assert_allclose(left['raw_resultants'],right['raw_resultants'],rtol=1e-8,atol=1e-9)
        rigid=np.zeros_like(u);rigid.reshape(-1,6)[:,:3]=[12.,-8.,3.]
        zero=model.recover(rigid,floor.loads(0.,beam_weight_factor=0.))
        self.assertEqual(zero['force_partition_relative_error'],0.)
        self.assertTrue(all(not np.any(row['global_elastic_force']) for row in zero['elements']))

    def test_all_rigid_rotations_are_null_and_do_not_change_deformed_forces(self):
        floor=self.floor;model=self.stiffness();load=floor.loads(-.002)
        u=np.random.default_rng(75).normal(size=len(floor.dof_ids))*.0001
        reference=model.recover(u,load)
        for axis in range(3):
            q=np.array([12.,-8.,3.,0.,0.,0.]);q[3+axis]=.01
            rigid=np.concatenate([rigid_arm_matrix(floor.node_positions[n])@q for n in floor.node_ids])
            null=model.recover(rigid,floor.loads(0.,beam_weight_factor=0.))
            self.assertLess(max(abs(v) for row in null['elements'] for v in row['global_elastic_force']),1e-8)
            actual=model.recover(u+rigid,load)
            self.assertLess(actual['force_partition_relative_error'],1e-10)
            for row,a,b in zip(floor.elements,reference['elements'],actual['elements']):
                np.testing.assert_allclose(a['global_resisting_force'],b['global_resisting_force'],rtol=1e-8,atol=1e-7)
                np.testing.assert_allclose(b['displacements'],row['transform']@(u+rigid)[row['indices']],rtol=0,atol=1e-13)
                if row['kind']=='slab':np.testing.assert_allclose(a['raw_resultants'],b['raw_resultants'],rtol=1e-8,atol=1e-9)

    def test_gross_limit_and_membrane_preservation(self):
        floor=self.floor;gross=self.stiffness(1.,1.,1.);cracked=self.stiffness()
        np.testing.assert_allclose(gross.stiffness.toarray(),floor.stiffness.toarray(),rtol=1e-11,atol=1e-8)
        u=np.zeros(len(floor.dof_ids));u[floor.membrane_dofs]=np.random.default_rng(4).normal(size=len(floor.membrane_dofs))*.001
        np.testing.assert_allclose(cracked.stiffness@u,floor.stiffness@u,rtol=1e-10,atol=1e-8)
        loads=floor.loads(-.002)
        old=floor.recover(u,loads);new=gross.recover(u,loads)
        for a,b in zip(old['elements'],new['elements']):
            np.testing.assert_allclose(a['global_resisting_force'],b['global_resisting_force'],rtol=1e-10,atol=1e-8)

    def test_physical_weight_and_full_equilibrium(self):
        floor=self.floor;model=self.stiffness();load=floor.loads(-3.*floor.gamma)
        # Explicit disjoint volume: slab plus four clear-span downstands.
        expected=-(60*40*3+4*9*(2*(60-8)+2*(40-8)))*floor.gamma
        self.assertAlmostEqual(load['loads'][2::6].sum(),expected,places=11)
        sub=condense(model.stiffness,floor.dof_ids,floor.boundary_dofs,load['loads'],input_identity='test')
        u=sub.recover(np.zeros(len(floor.boundary_dofs)))
        result=model.recover(u,load)
        residual=np.array(result['assembled_element_resistance'])
        ids=set(floor.boundary_dofs);free=[i for i,d in enumerate(floor.dof_ids) if d not in ids]
        np.testing.assert_allclose(residual[free],np.zeros(len(free)),atol=1e-8)
        reaction=np.zeros(6);external=np.zeros(6)
        for i,node in enumerate(floor.node_ids):
            arm=rigid_arm_matrix(floor.node_positions[node])
            reaction+=arm.T@residual[6*i:6*i+6]
            external+=arm.T@load['loads'][6*i:6*i+6]
        np.testing.assert_allclose(reaction+external,np.zeros(6),atol=1e-7)

    def test_full_actual_bending_matrix_matches_scalar_limit_and_keeps_twist(self):
        floor=self.floor;nu=floor.nu
        gross=floor.ec*floor.thickness**3/12/(1-nu**2)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
        ds={r['id']:.2*gross for r in floor.elements if r['kind']=='slab'}
        model=GeneralizedFloorStiffness(floor,beam_bending_factor=.35,slab_bending_factors=None,
            slab_bending_matrices=ds,web_torsion_factor=.1,stiffness_basis='independent fullD scalar-limit oracle',input_identity='fullD')
        reference=self.stiffness()
        np.testing.assert_allclose(model.stiffness.toarray(),reference.stiffness.toarray(),rtol=1e-10,atol=1e-8)
        u=np.random.default_rng(76).normal(size=len(floor.dof_ids))*.0001
        a=model.recover(u,floor.loads(-.002));b=reference.recover(u,floor.loads(-.002))
        for actual,expected in zip(a['elements'],b['elements']):
            if actual['kind']=='slab':np.testing.assert_allclose(actual['raw_resultants'],expected['raw_resultants'],rtol=1e-10,atol=1e-8)
        # Unequal directions with actual bending/twist couplings are supplied
        # as one PSD matrix, never inferred from two independent scalar I's.
        factor=np.array([[1.,.1,.05],[0,.7,.1],[0,0,.4]])
        d=factor.T@factor*5000.;ds={key:d for key in ds}
        anisotropic=GeneralizedFloorStiffness(floor,beam_bending_factor=.35,slab_bending_factors=None,
            slab_bending_matrices=ds,web_torsion_factor=.1,stiffness_basis='explicit anisotropic energy witness',input_identity='fullD2')
        self.assertLess(anisotropic.recover(u,floor.loads(-.002))['force_partition_relative_error'],1e-10)
        change=float(u@((anisotropic.stiffness-model.stiffness)@u));expected=0.
        for row in floor.elements:
            if row['kind']!='slab':continue
            q=-row['native']['strain'].reshape(4,8,24)[:,3:6,:]@(row['transform']@u[row['indices']])
            expected+=row['area_in2']/4*sum(k@(d-.2*gross)@k for k in q)
        self.assertAlmostEqual(change,expected,places=10)

    def test_declared_per_member_torsional_stiffness_has_exact_physical_recovery(self):
        floor=self.floor;factors={e['member_id']:(0. if e['axis']=='x' else 1.) for e in floor.elements if e['kind']=='web'}
        model=GeneralizedFloorStiffness(floor,beam_bending_factor=.35,slab_bending_factors=.2,
            web_torsion_factor=factors,stiffness_basis='declared zero/gross comparison, not applicability assertion',input_identity='mixedT')
        u=np.random.default_rng(33).normal(size=len(floor.dof_ids))*.0001
        r=model.recover(u,floor.loads(-.002));self.assertLess(r['force_partition_relative_error'],1e-10)
        # Zero torsion removes only axial-axis rotational stiffness; it cannot
        # delete beam axial, strong/weak bending or gravity self-weight.
        for row in r['elements']:
            if row['kind']=='web' and row['axis']=='x':
                self.assertAlmostEqual(row['global_resisting_force'][3],0.,places=10)
                self.assertAlmostEqual(row['global_resisting_force'][9],0.,places=10)


if __name__=='__main__':unittest.main()
