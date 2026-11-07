"""Consumer safety with explicit orchestration doubles, never engineering proof."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from Design import SMRF_Common_Action_Analysis as actions
from Design.SMRF_Common_Action_Contract import ROW_IDS


class ContinuousAcceptanceTests(unittest.TestCase):
    def replay(self,common_status='pass',local_status='pass',continuous=False,minimum_status='pass'):
        # All unrelated validation branches are explicit test doubles. The
        # real row-producing function must distinguish finite-cut success
        # from the computed reader's absent continuous proof.
        state=SimpleNamespace(identity=lambda:'fixture')
        package=dict(state_identity='fixture',source_identity={'fixture':'fixture'},
            definition_sha256='fixture',payload_sha256='fixture',inventory={},
            p5_parameters=dict(demand_factors_by_floor={}),p5={},
            common_strength=dict(chunks=[],continuous_capacity_verified=True),
            local_strength=dict(continuous_capacity_verified=True),
            slab_corner_demands=dict(records=[]),method_procedure=dict(procedure={},evidence_base_dir='fixture'),
            common_torsion={})
        local=dict(replay_payload=dict(response_bundles=[],inventory={}),
            records=[dict(shear=dict(status='pass'),local_minimum=dict(status=minimum_status))])
        package['local_checks']=local
        common_audit=dict(status=common_status,complete_required_inventory=True,continuous_capacity_verified=continuous,
                          saved_claims_verified=True)
        local_audit=dict(status=local_status,complete_required_inventory=True,continuous_capacity_verified=continuous,
                         saved_claims_verified=True)
        method=dict(proof_checks={name:[dict(status='pass')] for name in ROW_IDS if name.startswith('common.disposition.')},
                    method_problems=[],pending_tolerance_protocol=False)
        patches={
            'Design.SMRF_Common_Action_Contract.validate_package':None,
            'Design.SMRF_Common_Physics_Replay.replay_physics':dict(fixture=True),
            'Design.SMRF_Primary_Diaphragm_Batch.replay_primary_diaphragms':dict(force_cases_checked=1,maximum_normalized_nodal_residual=0.,factorizations=0),
            'Design.SMRF_Common_Guard_Plan.PreparedCommonGuardSources':SimpleNamespace(),
            'Design.SMRF_Common_Strength_Stream.verify_strength_stream':common_audit,
            'Design.SMRF_Common_Local_Checks.replay_local_checks':local,
            'Design.SMRF_Common_Local_Stream.verify_local_strength':local_audit,
            'Design.SMRF_Common_Corner_Demands.primary_corner_envelopes':{},
            'Design.SMRF_Common_Corner_Demands.corner_demands':dict(records=[]),
            'Design.SMRF_Common_Action_Procedure.assessment':method,
            'Design.SMRF_Common_Torsion_Stream.verify_torsion_stream':dict(status='pass',complete_required_inventory=True),
            'Design.SMRF_Common_Action_Analysis.physical_response_bundles':[],
        }
        with ExitStack() as stack:
            for name,value in patches.items():stack.enter_context(patch(name,return_value=value))
            result=actions.replay_assessment(state,package)
        self.assertEqual(result['audits']['strength'],common_audit)
        self.assertEqual(result['audits']['local_strength'],local_audit)
        rows={r['id']:r for r in result['checks']}
        self.assertEqual(rows['common.p5_signed_load_path']['status'],'pass')
        self.assertTrue(result['audits']['corner']['verified'])
        return rows

    def test_finite_cut_pass_cannot_become_shared_or_diaphragm_acceptance(self):
        rows=self.replay()
        for name in ('common.shared_steel_concurrent_strength','common.diaphragm_collectors_chords'):
            with self.subTest(name=name):
                self.assertEqual(rows[name]['status'],'not_evaluated')
                self.assertIn('continuous',str(rows[name]).lower())

    def test_method_examples_cannot_replace_actual_continuous_local_capacity(self):
        rows=self.replay()
        name='common.disposition.local_slab_column_steel'
        self.assertEqual(rows[name]['status'],'not_evaluated')
        self.assertIn('continuous',str(rows[name]).lower())

    def test_proved_capacity_failure_remains_a_failure_without_continuous_pass(self):
        rows=self.replay(common_status='fail',local_status='fail')
        for name in ('common.shared_steel_concurrent_strength','common.diaphragm_collectors_chords',
                     'common.disposition.local_slab_column_steel'):
            self.assertEqual(rows[name]['status'],'fail')

    def test_known_local_minimum_failure_is_not_hidden_by_missing_continuous_capacity(self):
        rows=self.replay(minimum_status='fail')
        self.assertEqual(rows['common.disposition.local_slab_column_steel']['status'],'fail')
        self.assertEqual(rows['common.shared_steel_concurrent_strength']['status'],'not_evaluated')


if __name__=='__main__':unittest.main()
