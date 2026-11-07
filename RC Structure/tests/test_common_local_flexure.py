import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from Design.SMRF_Common_Local_Flexure import inward_bar_clear_band,transfer_shortfall,plan_local_transfer,one_way_from_development
from Design.SMRF_Physical_Floor_Materials import spatial_section_from_inventory
from Design.SMRF_Coupled_Floor_Design import _hash
from test_physical_floor_cuts import shell
from test_physical_floor_materials import bar


class LocalFlexureTests(unittest.TestCase):
    def test_two_faces_signed_shortfall_preserves_mean_and_original_guard(self):
        result=transfer_shortfall([-80.,-40.],[-1.,1.],100.)
        self.assertEqual(result['face_increments_kipin'],[-30.,30.])
        self.assertEqual(result['guarded_face_moments_kipin'],[-110.,-10.])
        self.assertEqual(sum(result['guarded_face_moments_kipin']),-120.)
        self.assertEqual(result['guarded_transfer_kipin'],100.)
        reverse=transfer_shortfall([80.,40.],[-1.,1.],-100.)
        np.testing.assert_array_equal(reverse['guarded_face_moments_kipin'],[110.,10.])
        enough=transfer_shortfall([-100.,40.],[-1.,1.],100.)
        self.assertEqual(enough['added_transfer_kipin'],0.)

    def test_edge_one_actual_face_takes_whole_shortfall(self):
        self.assertEqual(transfer_shortfall([10.],[1.],30.)['guarded_face_moments_kipin'],[30.])
        self.assertEqual(transfer_shortfall([-10.],[-1.],30.)['guarded_face_moments_kipin'],[-30.])
        with self.assertRaises(ValueError):transfer_shortfall([1.,2.],[1.,1.],3.)

    def test_bar_clear_band_shrinks_within_code_width(self):
        bars=[dict(s_in=0.,db_in=1.),dict(s_in=10.,db_in=2.)]
        a,b=inward_bar_clear_band([0.,10.],bars)
        self.assertGreater(a,.5);self.assertLess(b,9.);self.assertGreater(b,a)
        with self.assertRaises(ValueError):inward_bar_clear_band([0.,1.],[dict(s_in=.5,db_in=4.)])

    def test_slab_only_preserves_both_beam_void_directions_without_credit(self):
        bars=[bar('matx',s=1.,z=2.8,area=.2,run=(0,10)),bar('maty','y',s=4.,z=2.,area=.2,run=(0,4)),
              bar('beamx',s=3.,z=0.,area=.31,run=(0,10),owner='web_line'),
              bar('beamy','y',s=6.,z=-2.,area=.31,run=(0,4),owner='web_line'),
              bar('below',s=2.,z=-8.,area=.31,run=(0,10),owner='web_line')]
        inv=dict(state_identity='s',identity_sha256='inv',slab_thickness_in=8.,floors={'1':dict(bars=bars)})
        args=dict(floor=1,axis='x',station_in=5.,interval_in=[0,4],axial_accounting_interval_in=[3,7],fc_ksi=5.,ec_ksi=4000.,phi_requirements={'NM':.65})
        out=spatial_section_from_inventory(inv,**args,material_scope='slab_only',apply_development_domains=True)
        self.assertIsNotNone(out['section'],out['issues']);self.assertEqual(out['physical_bar_ids'],['matx','maty'])
        longitudinal={r['id']:r for r in out['constructor']['longitudinal_bars']}
        self.assertFalse(longitudinal['beamx']['strength_credited']);self.assertNotIn('below',longitudinal)
        transverse=out['constructor']['transverse_rows'];self.assertIsNone(next(r for r in transverse if 'beamy' in r['physical_bar_ids'])['strength_run_interval_in'])
        # Independent uniform compressive concrete-volume difference. Steel
        # credit is disabled in BOTH inventories; only finite voids differ.
        bare=copy.deepcopy(inv);bare['floors']['1']['bars']=bars[:2]
        first=out['section'];second=spatial_section_from_inventory(bare,**args,material_scope='slab_only')['section']
        q=np.array([-.0002,-.0002,0,0,0,0,0.])
        self.assertGreater(first.state(q)['nominal_resultants'][0],second.state(q)['nominal_resultants'][0])

    def fixture(self):
        fp=dict(x_min_in=-50.,x_max_in=50.,y_min_in=-50.,y_max_in=50.)
        support=dict(id=1,x=0.,y=0.,h_in=20.,b_in=20.)
        definition=dict(state_identity='s',identity_sha256='d',floors=[dict(level=1,supports=[support],beams=[],footprint=fp)])
        bars=[]
        for axis in ('x','y'):
            for face,z in (('top',2.8 if axis=='x' else 2.),('bottom',-2.8 if axis=='x' else -2.)):
                for at in (-15.,-5.,5.,15.):bars.append(dict(bar(f'{axis}/{face}/{at}',axis,s=at,z=z,area=.2,run=(-45,45)),mat_name=axis+'_'+face))
        inventory=dict(state_identity='s',identity_sha256='i',slab_thickness_in=8.,floors={'1':dict(bars=bars,footprint=fp)})
        coef=np.zeros((4,8));coef[0]=[2,3,.4,-5,-7,.2,1,2]
        response=dict(definition_sha256='d',cases=[dict(id='G',dead=1.,live=0.)],floors=[dict(level=1,responses=[dict(case_id='G',elements=[shell('s',-50,50,-50,50,coef)],force_partition_relative_error=0.)])])
        bundle=dict(source_id='actual_frozen_gravity',response=response,material_inventory_id='i',analysis_input_identity='input',response_payload_sha256=_hash(response))
        common=dict(state_identity='s',input_signature='input',definition_sha256='d',definition=definition,inventory=inventory,
            combinations=[dict(id='G',ex=0.,ey=0.)],local_checks=dict(records=[dict(id='local/'+t,source_id='actual_frozen_gravity',floor=1,support_id=1,case_id='G',trace=t,
            flexural_transfer_required_kipin=dict(x=100.,y=150.)) for t in ('inside','outside')]))
        return common,bundle

    def test_complete_local_plan_keeps_same_case_seven_tensors_and_y_sign(self):
        common,bundle=self.fixture()
        with patch('Design.SMRF_Common_Action_Analysis.physical_response_bundles',return_value=[bundle]):
            plan=plan_local_transfer(common,SimpleNamespace(identity=lambda:'s'),selected_floor_case_pair=['actual_frozen_gravity',1,'G'],maximum_cuts=100)
        self.assertEqual(plan['required_cut_count'],16);self.assertEqual(len(plan['transfer_proofs']),4)
        for proof in plan['transfer_proofs']:
            self.assertAlmostEqual(proof['guarded_transfer_kipin'],100. if proof['axis']=='x' else -150.)
            self.assertEqual(proof['actual_bar_clear_band_in'],[-22.,22.])
        for spec in plan['cut_specs']:
            self.assertEqual(spec['material_scope'],'slab_only');self.assertIsNone(spec['member_id'])
            if spec['local_branch']=='physical':
                from Design.SMRF_Axial_Action_Accounting import verify_point_accounting
                row=plan['cuts'][spec['cut_id']]
                self.assertEqual(row['axial_action_window']['status'],'computed')
                self.assertEqual(verify_point_accounting(row,spec)['status'],'pass')
                self.assertGreaterEqual(spec['axial_window_selection']['actual_length_in'],24.)
                continue
            base=plan['cuts'][spec['cut_id'].removesuffix('/gamma_f')];guard=plan['cuts'][spec['cut_id']]
            self.assertNotIn('axial_action_window',guard)
            j=3 if spec['axis']=='x' else 4
            np.testing.assert_array_equal(np.delete(base['resultants'],j),np.delete(guard['resultants'],j))
        with patch('Design.SMRF_Common_Action_Analysis.physical_response_bundles',return_value=[bundle]):
            with self.assertRaisesRegex(ValueError,'exceeds declared cut budget'):
                plan_local_transfer(common,SimpleNamespace(identity=lambda:'s'),selected_floor_case_pair=['actual_frozen_gravity',1,'G'],maximum_cuts=15)


if __name__=='__main__':unittest.main()
