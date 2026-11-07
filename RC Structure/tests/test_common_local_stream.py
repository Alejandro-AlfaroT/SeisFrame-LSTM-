"""Local streaming controller, real constitutive witnesses; synthetic planner."""
import copy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Common_Local_Stream import assess_local_strength,verify_local_strength
from tests.test_physical_floor_strength import PhysicalStrengthTests


class LocalStreamTests(unittest.TestCase):
    def fixture(self,stack):
        inventory,spec,action,dev=PhysicalStrengthTests().fixture();cid=action['identity']['combination_id'];pair=['unit',1,cid]
        common=dict(state_identity='actual_state',input_signature='input',inventory=inventory,definition=dict(floors=[dict(level=1,fc_ksi=5.)]))
        specs=[dict(spec,cut_id='a'),dict(spec,cut_id='b')];cuts={s['cut_id']:copy.deepcopy(action) for s in specs}
        plan=dict(cut_specs=specs,cuts=cuts,required_cut_ids=['a','b'],required_case_ids=[cid],one_way_raw_cuts=[dict(specification=spec)])
        plan['payload_sha256']=_hash(plan)
        stack.enter_context(patch('Design.SMRF_Common_Local_Stream.local_universe',return_value=[pair]))
        stack.enter_context(patch('Design.SMRF_Common_Local_Stream.plan_local_transfer',return_value=plan))
        stack.enter_context(patch('Design.SMRF_Common_Local_Stream.actual_bar_development',side_effect=lambda state,inventory,specs,**kw:dict(cuts=[copy.deepcopy(dev['cuts'][0]) for _ in specs])))
        stack.enter_context(patch('Design.SMRF_Common_Local_Stream.one_way_from_development',return_value=dict(status='pass',synthetic_controller_only=True)))
        return common,SimpleNamespace(identity=lambda:'actual_state'),dict(strength_wall_seconds=60,max_required_cuts=10,max_chunk_cuts=1,max_strength_evaluations=100)

    def test_complete_local_chunk_and_shear_scope_reconstructed(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack)
            result=assess_local_strength(state,common,limits=limits,artifact_directory=temporary)
            self.assertEqual(result['status'],'pass');self.assertEqual(result['processed_cut_count'],2)
            replay=verify_local_strength(result,common,state=state)
            self.assertEqual(replay['status'],'pass');self.assertEqual(replay['one_way_cut_count'],1)
            for missing in ('chunks','one_way_checks'):
                corrupt=copy.deepcopy(result);corrupt[missing].pop();corrupt['payload_sha256']=_hash({k:v for k,v in corrupt.items() if k!='payload_sha256'})
                with self.assertRaisesRegex(ValueError,'completion claim|inventory claim'):verify_local_strength(corrupt,common,state=state)

    def test_unprocessed_scope_never_passes(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack);limits['strength_wall_seconds']=0
            result=assess_local_strength(state,common,limits=limits,artifact_directory=temporary)
            self.assertEqual(result['status'],'unknown');self.assertTrue(result['remaining_pairs'])
            self.assertEqual(verify_local_strength(result,common,state=state)['status'],'unknown')
