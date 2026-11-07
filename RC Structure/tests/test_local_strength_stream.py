"""Named local stream controller with actual constitutive zero-state checks.

Source plans/development are explicit doubles; physical producer coverage is
tested independently. This fixture does not qualify an actual floor.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Bar_Pitch_Layers import METHOD as PITCH_METHOD
from Design.SMRF_Common_Guard_Plan import LOCAL_PLAN_CONTRACT
from Design.SMRF_Local_Pitch_Strength import bind_local_pitch_action
from Design.SMRF_Common_Strength_Stream import assess_common_strength,verify_strength_stream,LOCAL_PITCH_BASIS
from Design.SMRF_Spatial_Floor_Section import SpatialFloorSection
from tests.test_local_pitch_strength import example


class LocalStrengthStreamTests(unittest.TestCase):
    def fixture(self,stack):
        inv,raw,spec,ev,q=example()
        raw['resultants']=[0.]*6;raw['Ms_transverse_first_moment_kipin']=0.
        raw['slab']['integral']=[0.]*8;raw['slab']['transverse_first_moment']=[0.]*8
        raw['slab']['pieces'][0]['integral']=[0.]*8;raw['slab']['pieces'][0]['transverse_first_moment']=[0.]*8
        specs=[];cuts={};pairs=[['unit',1,'G1']]
        for name in ('a','b'):
            original=copy.deepcopy(raw);original['identity']['location_id']=name
            specs.append(dict(spec,cut_id=name))
            cuts[name]=bind_local_pitch_action(original,floor=1,inventory=inv,analysis_input_identity='input',source_response_sha256='response',definition_sha256='definition')
        plan=dict(cut_specs=specs,cuts=cuts,required_cut_ids=['a','b'],required_cut_count=2,required_case_ids=['G1'],
            assigned_joint_or_other_axis_regions=[],action_contract=LOCAL_PLAN_CONTRACT,
            scope=dict(full_floor_case_universe=pairs,full_universe_sha256=_hash(pairs)))
        plan['payload_sha256']=_hash(plan)
        common=dict(input_signature='input',state_identity=inv['state_identity'],analysis_source_identity='source',inventory=inv,
            definition=dict(floors=[dict(level=1,fc_ksi=5.)]),p5={},floor_material_mapping=PITCH_METHOD)
        stack.enter_context(patch('Design.SMRF_Common_Action_Analysis.physical_response_bundles',return_value=[dict(source_id='unit',response=dict(cases=[dict(id='G1')]))]))
        def planner(*a,**kw):
            self.assertTrue(kw['local_pitch_actions']);self.assertEqual(set(kw['local_material_events_by_floor']),{'1'})
            return plan
        stack.enter_context(patch('Design.SMRF_Common_Strength_Stream.plan_physical_cut_family',side_effect=planner))
        def guards(common,partial,p5,**kw):
            self.assertTrue(kw['local_pitch_actions'])
            if len(partial['cut_specs'])>kw['maximum_cuts']:raise ValueError('Complete primary guards exceed budget')
            return partial
        stack.enter_context(patch('Design.SMRF_Common_Strength_Stream.build_common_guard_plan',side_effect=guards))
        stack.enter_context(patch('Design.SMRF_Local_Material_Events.installed_local_material_events',return_value=dict(method='explicit_event_controller_double',floor=1)))
        stack.enter_context(patch('Design.SMRF_Local_Pitch_Strength.actual_local_cut_development',side_effect=lambda *a,**k:copy.deepcopy(ev)))
        state=SimpleNamespace(identity=lambda:inv['state_identity'])
        limits=dict(strength_wall_seconds=60,max_required_cuts=10,max_chunk_cuts=1,max_strength_evaluations=30)
        return common,state,limits

    def test_complete_saved_witnesses_replay_without_inverse_and_missing_chunk_refused(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack)
            result=assess_common_strength(state,common,limits=limits,artifact_directory=temporary,material_action_basis=LOCAL_PITCH_BASIS)
            self.assertEqual(result['status'],'pass');self.assertEqual(len(result['chunks']),2)
            self.assertFalse(result['continuous_capacity_verified']);self.assertFalse(result['complete_design_qualified'])
            with (patch.object(SpatialFloorSection,'solve',side_effect=AssertionError('No inverse on saved readback')),
                  patch.object(SpatialFloorSection,'capacity_separation_certificate',side_effect=AssertionError('No dual search'))):
                audit=verify_strength_stream(result,common,state=state)
            self.assertEqual(audit['status'],'pass');self.assertEqual(audit['inverse_solves'],0)
            bad=copy.deepcopy(result);bad['chunks'].pop();bad['payload_sha256']=_hash({k:v for k,v in bad.items() if k!='payload_sha256'})
            with self.assertRaisesRegex(ValueError,'scope claim|universe claim'):verify_strength_stream(bad,common,state=state)

    def test_changed_source_material_events_and_actual_development_refused(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack)
            result=assess_common_strength(state,common,limits=limits,artifact_directory=temporary,material_action_basis=LOCAL_PITCH_BASIS)
            other=copy.deepcopy(common);other['analysis_source_identity']='different'
            with self.assertRaisesRegex(ValueError,'source changed'):verify_strength_stream(result,other,state=state)
            with patch('Design.SMRF_Local_Material_Events.installed_local_material_events',return_value={'changed':True}):
                with self.assertRaisesRegex(ValueError,'material events'):verify_strength_stream(result,common,state=state)
            with patch('Design.SMRF_Local_Pitch_Strength.actual_local_cut_development',return_value=[]):
                with self.assertRaisesRegex(ValueError,'development differs'):verify_strength_stream(result,common,state=state)
            other['floor_material_mapping']='clipped_cylinders_v2'
            with self.assertRaisesRegex(ValueError,'fixed-pitch'):verify_strength_stream(result,other,state=state)

    def test_bound_remains_unknown_and_incompatible_mapping_cannot_start(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            common,state,limits=self.fixture(stack);limits['max_required_cuts']=1
            bad=copy.deepcopy(common);bad['floor_material_mapping']='clipped_cylinders_v2'
            with self.assertRaisesRegex(ValueError,'fixed-pitch'):
                assess_common_strength(state,bad,limits=limits,artifact_directory=temporary,material_action_basis=LOCAL_PITCH_BASIS)
            record=assess_common_strength(state,common,limits=limits,artifact_directory=temporary,material_action_basis=LOCAL_PITCH_BASIS)
            self.assertEqual(record['status'],'unknown');self.assertFalse(record['complete_required_inventory'])
            self.assertEqual(record['remaining_pairs'],record['full_floor_case_universe'])
            self.assertEqual(verify_strength_stream(record,common,state=state)['status'],'unknown')


if __name__=='__main__':unittest.main()
