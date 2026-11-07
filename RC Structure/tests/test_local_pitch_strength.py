import copy
import unittest
from unittest.mock import patch
import numpy as np
from Design.SMRF_Local_Pitch_Strength import bind_local_pitch_action,assess_local_pitch_cut_family,verify_local_pitch_cut_assessment
from Design.SMRF_Pitch_Spatial_Section import pitch_spatial_section
from Design.SMRF_Spatial_Floor_Section import SpatialFloorSection
from Design.SMRF_Common_Floor_Actions import _hash
from test_pitch_spatial_section import fixture


def example():
    inv,args,ev=fixture();inv.update(physical_detail_status='pass',unqualified_geometry_diagnostic=False)
    args['transverse_reference']=14.;built=pitch_spatial_section(inv,**args,development_evidence=ev)
    q=np.array([.0001,.0001,0.,0.,0.,0.,0.]);values=np.array(built['section'].state(q)['design_resultants'])*24
    integral=np.r_[values[:6],0.,0.];first=14*integral;first[0]+=values[6]
    raw=dict(identity=dict(state_identity=inv['state_identity'],combination_id='G1',location_id='cut1',recovery_id='physics1'),
        slab=dict(axis='x',station_in=10.,trace='plus',interval_in=[2.,26.],integral=integral.tolist(),transverse_first_moment=first.tolist(),
            pieces=[dict(interval_in=[2.,26.],integral=integral.tolist(),transverse_first_moment=first.tolist())]),web=None,
        reference_z_in=0.,transverse_reference_in=14.,resultants=values[:6].tolist(),Ms_transverse_first_moment_kipin=float(values[6]),
        accounting=dict(equilibrium_residual=[0.],equilibrium_tolerance=[1e-9]))
    spec=dict(cut_id='cut1',floor=1,axis='x',station_in=10.,interval_in=[2.,26.],trace='plus',material_scope='slab_only')
    return inv,raw,spec,ev,q


def bind(inv,raw):
    return bind_local_pitch_action(raw,floor=1,inventory=inv,analysis_input_identity='analysis',source_response_sha256='sealed_response',definition_sha256='definition')


def assess(inv,raw,spec,ev,q):
    return assess_local_pitch_cut_family(None,inv,[spec],{'cut1':bind(inv,raw)},slab_fc_ksi=5.,ec_ksi=4000.,phi_requirements={'NM':.65,'shear':.6},
        required_cut_ids=['cut1'],required_case_ids=['G1'],assessment_identity='fixture_only',max_evaluations=20,
        development_by_cut={'cut1':ev},initial_strains={'cut1':q})


class LocalPitchStrengthTests(unittest.TestCase):
    def test_actual_local_material_and_optimizer_free_witness_replay(self):
        inv,raw,spec,ev,q=example();out=assess(inv,raw,spec,ev,q)
        self.assertEqual(out['status'],'pass',out['records'][0]['pending'])
        self.assertEqual(out['records'][0]['physical_bar_ids'],['x0','x1','x2','x3','y1'])
        self.assertFalse(out['continuous_coverage_verified']);self.assertFalse(out['engineering_qualified'])
        self.assertFalse(out['axial_window_equivalence_claimed']);self.assertFalse(out['nominal_probable_capacity_inventory_modified'])
        with patch.object(SpatialFloorSection,'solve',side_effect=AssertionError('No inverse in saved verification')),patch.object(SpatialFloorSection,'capacity_separation_certificate',side_effect=AssertionError('No dual search')):
            replay=verify_local_pitch_cut_assessment(out)
        self.assertEqual(replay['status'],'pass');self.assertTrue(replay['saved_claims_verified'])

    def test_changed_station_trace_or_action_cannot_borrow_local_material(self):
        inv,raw,spec,ev,q=example()
        for key,value in (('station_in',11.),('trace','minus'),('interval_in',[2.,25.]),('floor',2)):
            changed=dict(spec,**{key:value})
            with self.assertRaises(ValueError):assess(inv,raw,changed,ev,q)
        action=bind(inv,raw);action['resultants'][0]+=1
        with self.assertRaisesRegex(ValueError,'raw cut reconstruction'):
            assess_local_pitch_cut_family(None,inv,[spec],{'cut1':action},slab_fc_ksi=5,ec_ksi=4000,phi_requirements={'shear':.6},
                required_cut_ids=['cut1'],required_case_ids=['G1'],assessment_identity='bad',development_by_cut={'cut1':ev})

    def test_bound_failure_and_missing_development_remain_distinct(self):
        inv,raw,spec,ev,q=example();raw['resultants'][0]=1e5;raw['slab']['integral'][0]=1e5
        raw['slab']['pieces'][0]['integral'][0]=1e5;raw['slab']['transverse_first_moment'][0]=14e5;raw['slab']['pieces'][0]['transverse_first_moment'][0]=14e5
        out=assess(inv,raw,spec,ev,q);self.assertEqual(out['status'],'fail')
        replay=verify_local_pitch_cut_assessment(out);self.assertEqual(replay['status'],'fail');self.assertTrue(replay['saved_claims_verified'])
        unresolved=assess(inv,raw,spec,[],q);self.assertEqual(unresolved['status'],'unknown')
        self.assertEqual(verify_local_pitch_cut_assessment(unresolved)['status'],'unknown')

    def test_actual_upstream_equilibrium_and_detail_failures_stay_unknown_on_replay(self):
        inv,raw,spec,ev,q=example();raw['accounting']['equilibrium_residual']=[1e-7]
        out=assess(inv,raw,spec,ev,q);self.assertEqual(out['status'],'unknown')
        self.assertEqual(verify_local_pitch_cut_assessment(out)['status'],'unknown')
        inv,raw,spec,ev,q=example();inv['physical_detail_status']='unknown'
        out=assess(inv,raw,spec,ev,q);self.assertEqual(out['status'],'unknown')
        self.assertEqual(verify_local_pitch_cut_assessment(out)['status'],'unknown')

    def test_tampered_capacity_strain_quadrature_and_inventory_are_refused(self):
        inv,raw,spec,ev,q=example();out=assess(inv,raw,spec,ev,q)
        bad=copy.deepcopy(out);bad['records'][0]['section_assessment']['numerical_state']['generalized_strain'][0]+=.001
        self.assertFalse(verify_local_pitch_cut_assessment(bad)['saved_claims_verified'])
        bad=copy.deepcopy(out);bad['records'][0]['section_assessment']['numerical_state']['quadrature']['orders']=[12,12]
        with self.assertRaisesRegex(ValueError,'quadrature pair'):verify_local_pitch_cut_assessment(bad)
        bad=copy.deepcopy(out);bad['replay_payload']['inventory']['floors']['1']['bars'][0]['area_in2']*=1.1;bad['payload_sha256']=_hash(bad['replay_payload'])
        with self.assertRaises(ValueError):verify_local_pitch_cut_assessment(bad)
        bad=copy.deepcopy(out);bad['records'][0]['status']='unknown';bad['status']='fail'
        self.assertFalse(verify_local_pitch_cut_assessment(bad)['saved_claims_verified'])

    def test_required_case_cut_and_development_inventories_not_truncated(self):
        inv,raw,spec,ev,q=example();out=assess(inv,raw,spec,ev,q)
        for key,value in (('required_cut_ids',['cut1','missing']),('required_case_ids',['G1','missing']),('development_by_cut',{})):
            bad=copy.deepcopy(out);bad['replay_payload'][key]=value;bad['payload_sha256']=_hash(bad['replay_payload'])
            with self.assertRaises(ValueError):verify_local_pitch_cut_assessment(bad)

if __name__=='__main__':unittest.main()
