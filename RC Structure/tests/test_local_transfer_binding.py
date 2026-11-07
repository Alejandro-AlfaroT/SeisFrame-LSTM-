import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from Design.SMRF_Common_Local_Flexure import plan_local_transfer
from Design.SMRF_Local_Pitch_Strength import _check_point
from Design.SMRF_Local_Transfer_Binding import reconstruct_local_face_branch,projector_parameters
from Design.SMRF_Common_Guard_Plan import _PhysicalSource,primary_guard_increment
from Design.SMRF_Common_Action_Analysis import physical_response_bundles
from Design.SMRF_Primary_Diaphragm_Batch import analyze_primary_diaphragms,PrimaryDiaphragmReadout
from Design.SMRF_Primary_Moment_Guards import physical_cut_states
from tests.test_common_local_flexure import LocalFlexureTests
from tests.test_common_guard_plan import fixture,settings


class LocalTransferBindingTests(unittest.TestCase):
    def gravity_plans(self):
        common,bundle=LocalFlexureTests().fixture();state=SimpleNamespace(identity=lambda:'s')
        with patch('Design.SMRF_Common_Action_Analysis.physical_response_bundles',return_value=[bundle]):
            old=plan_local_transfer(common,state,selected_floor_case_pair=['actual_frozen_gravity',1,'G'],maximum_cuts=100)
            new=plan_local_transfer(common,state,selected_floor_case_pair=['actual_frozen_gravity',1,'G'],maximum_cuts=100,local_pitch_actions=True)
        return common,old,new

    def test_all_gravity_faces_traces_axes_and_shortfalls_reproduce_exactly(self):
        common,old,new=self.gravity_plans();self.assertEqual(new['required_cut_ids'],old['required_cut_ids'])
        self.assertEqual(new['required_cut_count'],16);self.assertEqual(new['transfer_proofs'],old['transfer_proofs'])
        for spec in new['cut_specs']:
            row=new['cuts'][spec['cut_id']]
            self.assertEqual({k:v for k,v in row.items() if k!='local_point_binding'},old['cuts'][spec['cut_id']])
            _check_point(json.loads(json.dumps(row)),spec,common['inventory'])
        self.assertFalse(new['continuous_capacity_verified'])
        self.assertFalse(new['nominal_probable_capacity_inventory_modified'])

    def test_force_pair_transfer_selector_and_geometry_tamper_refused(self):
        common,old,new=self.gravity_plans();spec=next(s for s in new['cut_specs'] if s['local_branch']=='physical/gamma_f')
        original=new['cuts'][spec['cut_id']]
        def force(r):r['resultants'][0]+=.1
        def selector(r):r['local_point_binding']['include_shortfall']=False
        def face(r):r['local_point_binding']['face_index']=1-r['local_point_binding']['face_index']
        def transfer(r):r['local_point_binding']['recipe']['support_record']['flexural_transfer_required_kipin']['x']+=1
        for change in (force,selector,face,transfer):
            bad=copy.deepcopy(original);change(bad)
            with self.subTest(change=change.__name__),self.assertRaises(ValueError):_check_point(bad,spec,common['inventory'])
        for change in ({'station_in':0.},{'trace':'invalid'},{'material_scope':'common_slab_web'},{'member_id':'beam'}):
            with self.assertRaises(ValueError):_check_point(original,{**spec,**change},common['inventory'])

    def test_primary_and_all_seven_P5_branches_preserve_concurrent_components(self):
        common,plan,p,state=fixture();batch=analyze_primary_diaphragms(common,**settings());reader=PrimaryDiaphragmReadout(batch)
        sources={b['source_id']:_PhysicalSource(b,common) for b in physical_response_bundles(common)}
        spec=dict(plan['cut_specs'][0],member_id=None,material_scope='slab_only');cid=plan['cuts'][spec['cut_id']]['identity']['combination_id']
        source=sources[spec['source_id']];gravity=sources['actual_frozen_gravity'];gid=common['gravity_case_mapping'][cid]
        raw=source.cut(spec,cid,member=None);g=gravity.cut(spec,gid,member=None)
        exact=dict(spec,member_id='1:x:0:0',interval_in=list(p.flange))
        pe=source.cut(exact,cid);ge=gravity.cut(exact,gid);inc=primary_guard_increment(common,cid,'1:x:0:0',spec['station_in'])
        inputs=dict(name=spec['cut_id'],raw=raw,primary=dict(increment=inc,physical_effective=pe,gravity_effective=ge,
            gravity_local=g,projector_parameters=projector_parameters(p)),p5_readouts={})
        bids=[r['id'] for r in reader.floor(1)['witnesses'][cid]['cases']]
        for bid in bids:inputs['p5_readouts'][bid]=reader.cut(1,cid,bid,axis=spec['axis'],station_in=spec['station_in'],
            interval_in=spec['interval_in'],trace=spec['trace'],transverse_reference_in=raw['transverse_reference_in'])
        base=physical_cut_states([raw],outer_interval_in=spec['interval_in'])[0][0]
        self.assertEqual(reconstruct_local_face_branch(inputs,'physical'),base)
        for mode in ('strong','weak','both'):
            row=reconstruct_local_face_branch(inputs,'primary/'+mode)
            self.assertEqual(row['resultants'][2],base['resultants'][2]);self.assertEqual(row['resultants'][5],base['resultants'][5])
            self.assertTrue(row['strength_guard']['slab_only']);self.assertFalse(row['accounting']['physical_equilibrium_solution_claimed'])
        self.assertEqual(len(bids),7)
        for bid in bids:
            row=reconstruct_local_face_branch(inputs,'P5/'+bid);readout=inputs['p5_readouts'][bid]
            np.testing.assert_array_equal(row['resultants'][:3],np.array(g['slab']['integral'][:3])+readout['slab_integrated_N_kip'])
            np.testing.assert_array_equal(row['resultants'][3:6],base['resultants'][3:6])
            for mode in ('strong','both'):
                simultaneous=reconstruct_local_face_branch(inputs,'P5/'+bid+'/'+mode)
                self.assertEqual(simultaneous['resultants'][2],row['resultants'][2])
                self.assertEqual(simultaneous['resultants'][5],row['resultants'][5])
                self.assertTrue(simultaneous['strength_guard']['web_design_force_omitted_from_slab_resistance'])
        bad=copy.deepcopy(inputs);bad['primary']['increment']['station_in']+=1
        with self.assertRaises(ValueError):reconstruct_local_face_branch(bad,'primary/strong')
        bad=copy.deepcopy(inputs);bad['p5_readouts'][bids[0]]['station_in']+=1
        with self.assertRaises(ValueError):reconstruct_local_face_branch(bad,'P5/'+bids[0])


if __name__=='__main__':unittest.main()
