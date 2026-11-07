"""Independent reused-operator and exact declared-width force witnesses."""
import copy
import unittest
from unittest import mock
import numpy as np
from Design import SMRF_Diaphragm_Actions as actions
from Design.SMRF_Physical_Diaphragm import PhysicalDiaphragm,physical_membrane_strip_cut
from tests.test_diaphragm_numerical_refinement import manufactured_input


class DiaphragmBatchTests(unittest.TestCase):
    def test_one_LU_across_independent_signed_batches_reproduces_every_old_response(self):
        mesh,components,cases=manufactured_input();cuts=[dict(axis='x',coordinate_in=3.5)]
        signed=[{**cases[0],'id':str(f),'factors':{'quadratic':f}} for f in (1.,-2.,.3)]
        reference=[actions.solve_diaphragm_actions(mesh,components,[c],cuts=cuts) for c in signed]
        real=actions.splu
        with mock.patch.object(actions,'splu',wraps=real) as factor:
            operator=actions.PreparedDiaphragmOperator(mesh)
            actual=[operator.solve(components,[c],cuts=cuts) for c in signed]
            self.assertEqual(factor.call_count,1)
        self.assertEqual(operator.record()['solve_calls'],3);self.assertEqual(operator.record()['case_rhs_solved'],3)
        for old,new in zip(reference,actual):
            a,b=old['cases'][0],new['cases'][0]
            np.testing.assert_allclose(a['displacements_in'],b['displacements_in'],rtol=0,atol=0)
            self.assertEqual(a['beam_axial'],b['beam_axial']);self.assertEqual(a['cuts'],b['cuts'])
            for x,y in zip(a['slab_samples'],b['slab_samples']):
                np.testing.assert_allclose([x[k] for k in ('Nxx_kip_per_in','Nyy_kip_per_in','Nxy_kip_per_in')],
                                           [y[k] for k in ('Nxx_kip_per_in','Nyy_kip_per_in','Nxy_kip_per_in')],atol=1e-13)

    def test_cache_rejects_changed_geometry_material_and_its_own_mutated_snapshot(self):
        mesh,components,cases=manufactured_input();op=actions.PreparedDiaphragmOperator(mesh)
        changed=copy.deepcopy(mesh);changed['E_ksi']*=.5
        with self.assertRaisesRegex(ValueError,'stale'):actions.solve_diaphragm_actions(changed,components,cases,operator=op)
        op.mesh['nodes'][0][0]+=.1
        with self.assertRaisesRegex(ValueError,'stale'):op.solve(components,cases)

    def test_physical_witness_batch_keeps_original_case_and_component_identities(self):
        from tests.test_physical_diaphragm import inputs,conventional
        d,mass,source,inf=inputs();adapter=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        requests=[dict(id=f'load{i}',influences=inf,witness_id='signed_X_30Y',conventional_forces={**conventional(),'omega0':omega},
                       cuts=[dict(axis='x',coordinate_in=11.7)]) for i,omega in enumerate((3.,4.))]
        references=[adapter.solve(r['influences'],r['witness_id'],conventional_forces=r['conventional_forces'],cuts=r['cuts']) for r in requests]
        real=actions.splu
        with mock.patch.object(actions,'splu',wraps=real) as factor:batch=adapter.solve_batch(requests)
        self.assertEqual(factor.call_count,1);self.assertEqual(batch['prepared_operator']['case_rhs_solved'],14)
        for ref,request in zip(references,requests):
            result=batch['results'][request['id']]
            self.assertEqual(result['force_case_definitions'],ref['force_case_definitions'])
            self.assertEqual(result['component_equilibrium'],ref['component_equilibrium'])
            for a,b in zip(ref['cases'],result['cases']):
                np.testing.assert_allclose(a['displacements_in'],b['displacements_in'],atol=1e-13)
                self.assertEqual(a['member_external_axial'],b['member_external_axial'])
                self.assertEqual(a['continuum_cut_equilibrium'],b['continuum_cut_equilibrium'])
                self.assertEqual(b['slab_samples'],[]);self.assertTrue(b['slab_samples_omitted'])

    def test_declared_width_slab_N_and_first_moment_match_independent_polynomial_integrals(self):
        mesh=actions.rectangular_mesh([0.,3.,10.],[0.,2.,5.,8.],thickness_in=2.,elastic_modulus_ksi=100.,poisson=0.)
        xy=np.asarray(mesh['nodes']);x,y=xy.T;u=np.c_[.01*x+.002*x*y,-.003*y+.004*x*y]
        result=dict(mesh=mesh,cases=[dict(id='exact',displacements_in=u.tolist(),beam_axial=[])])
        cut=physical_membrane_strip_cut(result,'exact',axis='x',station_in=3.,interval_in=[1.,7.],trace='plus',
            transverse_reference_in=2.,web_widths_in={})
        np.testing.assert_allclose(cut['slab_integrated_N_kip'],[21.6,10.8,13.2],atol=1e-12)
        self.assertAlmostEqual(cut['Ms_transverse_first_moment_kipin'],50.4)
        self.assertEqual(len(cut['slab_fields']),3)
        self.assertFalse(cut['physical_out_of_plane_compatibility_claimed'])
        for p in cut['slab_fields']:
            self.assertTrue(np.max(abs(np.asarray(p['sample_tensors'])[:,3:]))==0.)
            self.assertLess(max(abs(v) for v in p['affine_transfer_error']),1e-12)

    def test_strip_web_N_uses_whole_actual_width_and_never_invents_strong_moment(self):
        mesh=actions.rectangular_mesh([0.,10.],[-5.,0.,5.],thickness_in=2.,elastic_modulus_ksi=100.,beams=[
            dict(id='segment',member_id='beam',axis='x',i=2,j=3,area_web_in2=8.)])
        result=dict(mesh=mesh,cases=[dict(id='force',displacements_in=np.zeros((6,2)).tolist(),beam_axial=[dict(id='segment',N_kip=7.)])])
        cut=physical_membrane_strip_cut(result,'force',axis='x',station_in=5.,interval_in=[-2.,2.],trace='plus',
            transverse_reference_in=1.,web_widths_in={'beam':4.})
        np.testing.assert_array_equal(cut['total_integrated_N_kip'],[7.,0.,0.]);self.assertEqual(cut['Ms_transverse_first_moment_kipin'],-7.)
        self.assertEqual(cut['webs'][0]['m_z_kip_in'],0.);self.assertEqual(cut['webs'][0]['reference_z_in'],0.)
        with self.assertRaisesRegex(ValueError,'cuts a web'):
            physical_membrane_strip_cut(result,'force',axis='x',station_in=5.,interval_in=[-1.,2.],trace='plus',transverse_reference_in=0.,web_widths_in={'beam':4.})


if __name__=='__main__':unittest.main()
