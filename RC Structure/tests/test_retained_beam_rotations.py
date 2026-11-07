"""Equivalence of two algebraic representations of the same physical connection."""
import unittest
import numpy as np
from Design.SMRF_Beam_Shell_Rotation_Dofs import retain_independent_beam_rotations
from Design.SMRF_Beam_Shell_Connection import connect_independent_beam_rotations
from Design.SMRF_Floor_Substructure import GeneralizedFloorStiffness,rigid_arm_matrix,FloorAssembly
from test_floor_generalized_stiffness import floor_fixture


def refined_fixture():
    base=floor_fixture()
    return FloorAssembly([0.,4.,4.5,5.,10.,30.,50.,55.,55.5,56.,60.],
        [0.,4.,4.5,5.,10.,20.,30.,35.,35.5,36.,40.],floor_id='graded',thickness_in=3.,elastic_modulus_ksi=4000.,
        supports=base.supports,beams=base.beams)


class RetainedRotationTests(unittest.TestCase):
    def test_independent_dense_condensation_preserves_complete_cracked_energy(self):
        for fixture in (floor_fixture,refined_fixture):
            retained=fixture();early=fixture();n=len(early.dof_ids)
            meta=retain_independent_beam_rotations(retained);connect_independent_beam_rotations(early)
            self.assertGreater(meta['internal_rotation_dof_count'],0)
            self.assertEqual(retained.dof_ids[:n],early.dof_ids)
            self.assertEqual(retained.boundary_dofs,early.boundary_dofs)
            for beta in (.35,.5,1.):
                args=dict(beam_bending_factor=beta,slab_bending_factors=.2,web_torsion_factor=.1,
                    stiffness_basis='explicit equivalence test only',input_identity='retained_rotation_test')
                full=GeneralizedFloorStiffness(retained,**args);prior=GeneralizedFloorStiffness(early,**args)
                k=full.stiffness.toarray();internal=np.arange(n,len(retained.dof_ids))
                # Independent dense elimination of all internal rotations from
                # the completed cracked floor, not the earlier beam-only map.
                transform=np.vstack((np.eye(n),-np.linalg.solve(k[np.ix_(internal,internal)],k[internal,:n])))
                reduced=transform.T@k@transform;expected=prior.stiffness.toarray()
                error=float(np.max(abs(reduced-expected)))/max(1.,float(np.max(abs(expected))))
                self.assertLess(error,2e-12)
                u=np.random.default_rng(182).normal(size=n)*1e-4;uf=transform@u
                self.assertLess(float(np.max(abs((k@uf)[internal]))),1e-7)
                for row in retained.elements:
                    if row['kind']=='web':
                        original=next(r for r in early.elements if r['id']==row['id'])
                        np.testing.assert_allclose(row['transform']@uf[row['indices']],original['transform']@u[original['indices']],rtol=2e-10,atol=1e-13)
                response=full.recover(uf,retained.loads(-.002))
                self.assertLess(response['force_partition_relative_error'],1e-9)
                self.assertFalse(response['qualification_asserted'])

    def test_all_six_rigid_modes_and_shell_drilling_independence(self):
        floor=floor_fixture();meta=retain_independent_beam_rotations(floor);n=meta['original_physical_dof_count']
        model=GeneralizedFloorStiffness(floor,beam_bending_factor=.5,slab_bending_factors=.5,web_torsion_factor=1.,
            stiffness_basis='rigid mode test only',input_identity='retained_rotation_rigid')
        q=np.array([1.,-2.,3.,.001,-.002,.003])
        rigid=np.r_[np.concatenate([rigid_arm_matrix(floor.node_positions[node])@q for node in floor.node_ids]),
                    np.full(len(floor.dof_ids)-n,q[5])]
        for row in floor.elements:
            expected=np.concatenate([rigid_arm_matrix(p)@q for p in row['points']])
            np.testing.assert_allclose(row['transform']@rigid[row['indices']],expected,atol=1e-14,rtol=0)
        response=model.recover(rigid,floor.loads(0.,beam_weight_factor=0.))
        self.assertLess(np.max(np.abs(response['assembled_element_resistance'])),1e-8)
        drill=np.zeros(len(floor.dof_ids));supports={s['id'] for s in floor.supports}
        for i,node in enumerate(floor.node_ids):
            if node not in supports:drill[6*i+5]=.01
        for row in floor.elements:
            if row['kind']=='web':np.testing.assert_allclose(row['transform']@drill[row['indices']],0.,atol=0)
        with self.assertRaisesRegex(ValueError,'already'):retain_independent_beam_rotations(floor)
        with self.assertRaisesRegex(ValueError,'already'):connect_independent_beam_rotations(floor)

    def test_unlabelled_extra_rotations_cannot_enter_generalized_model(self):
        floor=floor_fixture();meta=retain_independent_beam_rotations(floor);meta['internal_rotation_dof_indices']=[]
        with self.assertRaisesRegex(ValueError,'Declared internal'):
            GeneralizedFloorStiffness(floor,beam_bending_factor=.5,slab_bending_factors=.5,web_torsion_factor=1.,
                stiffness_basis='bad unlabelled fixture',input_identity='bad')

    def test_single_segment_has_no_invented_internal_dof(self):
        base=floor_fixture();floor=FloorAssembly([0.,4.,56.,60.],[0.,4.,36.,40.],floor_id='single',thickness_in=3.,
            elastic_modulus_ksi=4000.,supports=base.supports,beams=base.beams)
        old=floor.stiffness.copy();dofs=list(floor.dof_ids);metadata=retain_independent_beam_rotations(floor)
        self.assertEqual(metadata['internal_rotation_dof_count'],0);self.assertEqual(floor.dof_ids,dofs)
        np.testing.assert_allclose(floor.stiffness.toarray(),old.toarray(),rtol=1e-14,atol=1e-8)


if __name__=='__main__':unittest.main()
