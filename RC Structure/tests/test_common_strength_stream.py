"""Stream coverage/tamper tests with real zero-demand constitutive witnesses.

The planner fixture is explicitly synthetic; these tests qualify controller
accounting, never physical floor applicability or installed development.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Common_Strength_Stream import assess_common_strength,verify_strength_stream
from tests.test_physical_floor_strength import PhysicalStrengthTests


class StrengthStreamTests(unittest.TestCase):
    def fixture(self,stack):
        inv,spec,action,dev=PhysicalStrengthTests().fixture();cid=action['identity']['combination_id']
        common=dict(input_signature='unit_stream',state_identity='actual_state',analysis_source_identity='unit_source',inventory=inv,
            definition=dict(floors=[dict(level=1,fc_ksi=5.)]),p5={})
        pairs=[['unit',1,cid]];specs=[];cuts={}
        for name in ('a','b'):
            row=copy.deepcopy(action);row['identity']['location_id']=name
            for source in row['source_identities']:source['location_id']=name
            specs.append(dict(spec,cut_id=name));cuts[name]=row
        plan=dict(cut_specs=specs,cuts=cuts,required_cut_ids=['a','b'],required_cut_count=2,required_case_ids=[cid],
            assigned_joint_or_other_axis_regions=[],scope=dict(full_floor_case_universe=pairs,full_universe_sha256=_hash(pairs)))
        plan['payload_sha256']=_hash(plan)
        bundle=dict(source_id='unit',response=dict(cases=[dict(id=cid)]))
        stack.enter_context(patch('Design.SMRF_Common_Action_Analysis.physical_response_bundles',return_value=[bundle]))
        stack.enter_context(patch('Design.SMRF_Common_Strength_Stream.plan_physical_cut_family',return_value=plan))
        def guards(common,partial,p5,**kw):
            if len(partial['cut_specs'])>kw['maximum_cuts']:raise ValueError('Complete required primary guards exceed declared cut budget')
            return partial
        stack.enter_context(patch('Design.SMRF_Common_Strength_Stream.build_common_guard_plan',side_effect=guards))
        def development(state,inventory,chosen,**kw):return dict(cuts=[copy.deepcopy(dev['cuts'][0]) for _ in chosen])
        stack.enter_context(patch('Design.SMRF_Actual_Bar_Development.actual_bar_development',side_effect=development))
        state=SimpleNamespace(identity=lambda:'actual_state')
        limits=dict(strength_wall_seconds=60,max_required_cuts=10,max_chunk_cuts=1,max_strength_evaluations=100)
        return common,state,limits

    def test_complete_two_chunk_readback_and_missing_chunk_cannot_pass(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack)
            record=assess_common_strength(state,common,limits=limits,artifact_directory=temporary)
            self.assertEqual(record['status'],'pass');self.assertEqual(len(record['chunks']),2)
            replay=verify_strength_stream(record,common,state=state)
            self.assertEqual(replay['status'],'pass');self.assertEqual(replay['inverse_solves'],0)
            bad=copy.deepcopy(record);bad['chunks'].pop();bad['payload_sha256']=_hash({k:v for k,v in bad.items() if k!='payload_sha256'})
            with self.assertRaisesRegex(ValueError,'scope claim|universe claim'):verify_strength_stream(bad,common,state=state)
            path=Path(record['chunks'][0]['artifact_file']);path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'bytes changed'):verify_strength_stream(record,common,state=state)

    def test_cut_allowance_does_not_remove_unprocessed_original_scope(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack);limits['max_required_cuts']=1
            record=assess_common_strength(state,common,limits=limits,artifact_directory=temporary)
            self.assertEqual(record['status'],'unknown');self.assertFalse(record['complete_required_inventory'])
            self.assertEqual(record['remaining_pairs'],record['full_floor_case_universe'])
            self.assertEqual(verify_strength_stream(record,common,state=state)['status'],'unknown')

if __name__=='__main__':unittest.main()
