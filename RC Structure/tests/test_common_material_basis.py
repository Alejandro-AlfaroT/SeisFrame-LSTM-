"""Action/strength controller identity tests; no mechanical analysis is mocked as proof."""
import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from Design.SMRF_Coupled_Floor_Design import _hash
from Design import SMRF_Common_Action_Analysis as actions
from Design.SMRF_Bar_Pitch_Layers import METHOD as PITCH
from Design.SMRF_Common_Strength_Stream import LEGACY_BASIS,LOCAL_PITCH_BASIS,assess_common_strength
from Design.SMRF_Common_Local_Stream import assess_local_strength


class CommonMaterialBasisTests(unittest.TestCase):
    def common(self,basis=LOCAL_PITCH_BASIS):
        source={'fixture.py':'explicit_controller_source_double'}
        common=dict(method=actions.METHOD,state_identity='state',definition={'identity_sha256':'definition','floors':[{'level':1,'fc_ksi':5.}]},
            inventory={'identity_sha256':'inventory'},source_identity=source,analysis_source_identity=_hash(source),
            limits={},combinations=[],model_period_sec=1.,demand_parameters={},elf={},grid_masses={},
            floor_material_mapping=PITCH,strength_material_action_basis=basis)
        common['input_signature']=actions.common_input_identity(common['state_identity'],common['definition'],common['inventory'],
            common['analysis_source_identity'],{},[],1.,{},{},{},material_mapping=PITCH,strength_material_action_basis=basis)
        common['payload_sha256']=_hash(common)
        return common

    def test_basis_changes_signature_and_resealed_payload_cannot_change_choice(self):
        local=self.common();legacy=self.common(LEGACY_BASIS)
        self.assertNotEqual(local['input_signature'],legacy['input_signature'])
        self.assertEqual(actions.verify_common_input_identity(local),LOCAL_PITCH_BASIS)
        self.assertEqual(actions.verify_common_input_identity(legacy),LEGACY_BASIS)
        for field,value in [('strength_material_action_basis',LEGACY_BASIS),('floor_material_mapping','clipped_cylinders_v2'),
                            ('method','old_method'),('analysis_source_identity','arbitrary')]:
            bad=copy.deepcopy(local);bad[field]=value;bad['payload_sha256']=_hash({k:v for k,v in bad.items() if k!='payload_sha256'})
            with self.assertRaises(ValueError):actions.verify_common_input_identity(bad)
        bad=copy.deepcopy(local);bad.pop('strength_material_action_basis')
        with self.assertRaises(KeyError):actions.verify_common_input_identity(bad)

    def test_incompatible_choice_stops_before_any_installed_or_native_work(self):
        with patch.object(actions.mg,'active',side_effect=AssertionError('Must fail before installed model access')):
            with self.assertRaisesRegex(ValueError,'fixed-pitch'):
                actions.analyze_installed(None,None,source_identity={},model_period_sec=1.,limits={},strength_material_action_basis=LOCAL_PITCH_BASIS)

    def test_explicit_choice_reaches_frozen_family_and_unknown_stays_unknown(self):
        state=SimpleNamespace(identity=lambda:'state');cfg=SimpleNamespace(demands=SimpleNamespace(live_load_patterning=False,accidental_torsion_ratio=.05,risk_category='II'))
        common=self.common();limits={k:1. for k in actions._LIMITS}
        with ExitStack() as stack:
            for target,value in [('mg.active',state),('current_source_identity',common['source_identity']),('installed_definition',common['definition']),
                                 ('installed_inventory',common['inventory']),('strength_load_combinations',[]),('installed_grid_masses',{}),('elf_story_forces',{})]:
                stack.enter_context(patch('Design.SMRF_Common_Action_Analysis.'+target,return_value=value))
            model=stack.enter_context(patch.object(actions,'CoupledFloorSystem'))
            freeze=stack.enter_context(patch.object(actions,'frozen_gravity_family',return_value=(dict(status='unknown',reason='explicit_fixture_domain_failure'),None)))
            out=actions.analyze_installed(state,cfg,source_identity=common['source_identity'],model_period_sec=1.,limits=limits,
                material_mapping=PITCH,strength_material_action_basis=LOCAL_PITCH_BASIS)
        self.assertEqual(freeze.call_args.kwargs['material_mapping'],PITCH)
        self.assertEqual(out['strength_material_action_basis'],LOCAL_PITCH_BASIS)
        self.assertEqual(out['status'],'unknown');self.assertFalse(out['qualification_asserted'])
        self.assertFalse(out['official_pilot_started']);self.assertEqual(actions.verify_common_input_identity(out),LOCAL_PITCH_BASIS)
        self.assertEqual(model.call_count,1)

    def test_both_final_strength_streams_receive_identical_bound_basis(self):
        for basis in (LEGACY_BASIS,LOCAL_PITCH_BASIS):
            common=self.common(basis);state=SimpleNamespace(identity=lambda:'state')
            with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                for target,value in [('installed_definition',common['definition']),('installed_inventory',common['inventory']),('current_source_identity',common['source_identity'])]:
                    stack.enter_context(patch('Design.SMRF_Common_Action_Analysis.'+target,return_value=value))
                stack.enter_context(patch('Design.SMRF_Common_Guard_Plan.PreparedCommonGuardSources',return_value=SimpleNamespace(bundles=[])))
                for target in ('SMRF_Common_Local_Checks.assess_local_checks','SMRF_Common_Corner_Demands.corner_demands',
                               'SMRF_Common_Corner_Demands.primary_corner_envelopes','SMRF_Common_Torsion_Stream.assess_torsion_stream'):
                    stack.enter_context(patch('Design.'+target,return_value={'explicit_test_double':True}))
                primary=stack.enter_context(patch('Design.SMRF_Common_Strength_Stream.assess_common_strength',return_value={'explicit_test_double':True}))
                local=stack.enter_context(patch('Design.SMRF_Common_Local_Stream.assess_local_strength',return_value={'explicit_test_double':True}))
                out=actions.assess_installed(state,common,artifact_directory=directory)
                for call in (primary,local):self.assertEqual(call.call_args.kwargs['material_action_basis'],basis)
                self.assertTrue(out['source_unchanged']);self.assertEqual(out['strength_material_action_basis'],basis)

    def test_stream_cannot_silently_switch_basis_under_same_common_identity(self):
        common=self.common()
        for assess in (assess_common_strength,assess_local_strength):
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                with self.assertRaisesRegex(ValueError,'bound common input'):
                    assess(None,common,limits={},artifact_directory=directory)
                self.assertEqual(list(Path(directory).iterdir()),[])

    def test_actual_choice_must_equal_method_proof_protocol_choice(self):
        common=self.common()
        common['method_procedure']=dict(procedure=dict(protocol=dict(material_action_basis=dict(
            floor_material_mapping=PITCH,strength_material_action_basis=LOCAL_PITCH_BASIS))))
        self.assertEqual(actions.verify_common_input_identity(common),LOCAL_PITCH_BASIS)
        common['method_procedure']['procedure']['protocol']['material_action_basis']['strength_material_action_basis']=LEGACY_BASIS
        with self.assertRaisesRegex(ValueError,'method-proof protocol'):actions.verify_common_input_identity(common)

    def test_saved_physics_rejects_resealed_choice_before_native_reconstruction(self):
        from Design.SMRF_Common_Physics_Replay import replay_physics
        common=self.common();common['strength_material_action_basis']=LEGACY_BASIS
        common['payload_sha256']=_hash({k:v for k,v in common.items() if k!='payload_sha256'})
        with (patch.object(actions,'current_source_identity',return_value=common['source_identity']),
              patch('Design.SMRF_Common_Physics_Replay.installed_definition',side_effect=AssertionError('No reconstruction after identity mismatch'))):
            with self.assertRaisesRegex(ValueError,'input identity'):replay_physics(None,common)


if __name__=='__main__':unittest.main()
