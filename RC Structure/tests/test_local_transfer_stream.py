import copy
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Common_Local_Stream import assess_local_strength,verify_local_strength
from Design.SMRF_Common_Strength_Stream import LOCAL_PITCH_BASIS
from Design.SMRF_Spatial_Floor_Section import SpatialFloorSection
import Design.SMRF_Common_Strength_Stream as common_stream
import tests.test_local_strength_stream as source_fixture


class LocalTransferStreamTests(unittest.TestCase):
    def fixture(self,stack):
        common,state,limits=source_fixture.LocalStrengthStreamTests().fixture(stack)
        plan=copy.deepcopy(common_stream.plan_physical_cut_family(local_pitch_actions=True,local_material_events_by_floor={'1':{}}))
        plan['one_way_raw_cuts']=[dict(specification=plan['cut_specs'][0])]
        plan['payload_sha256']=_hash({k:v for k,v in plan.items() if k!='payload_sha256'})
        def planner(*a,**kw):
            self.assertTrue(kw['local_pitch_actions'])
            if kw['maximum_cuts']<2:raise ValueError('Complete local transfer inventory exceeds declared cut budget')
            return plan
        stack.enter_context(patch('Design.SMRF_Common_Local_Stream.plan_local_transfer',side_effect=planner))
        stack.enter_context(patch('Design.SMRF_Common_Local_Stream.actual_bar_development',side_effect=lambda state,inv,specs,**kw:dict(cuts=[dict(evidence=[]) for s in specs])))
        stack.enter_context(patch('Design.SMRF_Common_Local_Stream.one_way_from_development',return_value=dict(status='pass',synthetic_controller_only=True)))
        return common,state,limits

    def test_full_chunks_and_separate_one_way_inventory_replay(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack)
            result=assess_local_strength(state,common,limits=limits,artifact_directory=temporary,material_action_basis=LOCAL_PITCH_BASIS)
            self.assertEqual(result['status'],'pass');self.assertEqual(result['processed_cut_count'],2)
            with patch.object(SpatialFloorSection,'solve',side_effect=AssertionError('No saved inverse')):
                audit=verify_local_strength(result,common,state=state)
            self.assertEqual(audit['status'],'pass');self.assertEqual(audit['one_way_cut_count'],1)
            self.assertFalse(audit['continuous_capacity_verified'])
            for missing in ('chunks','one_way_checks'):
                bad=copy.deepcopy(result);bad[missing].pop();bad['payload_sha256']=_hash({k:v for k,v in bad.items() if k!='payload_sha256'})
                with self.assertRaisesRegex(ValueError,'completion claim|inventory claim'):verify_local_strength(bad,common,state=state)
            with patch('Design.SMRF_Local_Pitch_Strength.actual_local_cut_development',return_value=[]):
                with self.assertRaisesRegex(ValueError,'development changed'):verify_local_strength(result,common,state=state)
            changed=copy.deepcopy(common);changed['analysis_source_identity']='other'
            with self.assertRaisesRegex(ValueError,'source differs'):verify_local_strength(result,changed,state=state)

    def test_cut_bound_cannot_drop_required_transfer_or_shear(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack);limits['max_required_cuts']=1
            result=assess_local_strength(state,common,limits=limits,artifact_directory=temporary,material_action_basis=LOCAL_PITCH_BASIS)
            self.assertEqual(result['status'],'unknown');self.assertFalse(result['complete_required_inventory'])
            self.assertEqual(result['remaining_pairs'],result['full_universe'])
            self.assertEqual(verify_local_strength(result,common,state=state)['status'],'unknown')


if __name__=='__main__':unittest.main()
