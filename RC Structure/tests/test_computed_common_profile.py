"""Synthetic proof-reader tests. No engineering method is qualified by these data."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

RC=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RC))
from Design import Design_Evidence_Profile as ep
from Design import SMRF_Common_Action_Procedure as cp
from Design import SMRF_Common_Action_Contract as ca
from Design.Config import DesignConfig


class ComputedProfile(unittest.TestCase):
    def setUp(self):
        scratch=RC/'tests'/'_computed_profile_scratch';scratch.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch);self.root=Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        self.source='Design/SMRF_Common_Action_Procedure.py'
        self.sources={self.source:hashlib.sha256((RC/self.source).read_bytes()).hexdigest()}
        for target,name,value in ((ca,'REQUIRED_SOURCE_FILES',(self.source,)),(cp,'SOURCE_FILES',(self.source,))):
            guard=patch.object(target,name,value);guard.start();self.addCleanup(guard.stop)
        original=RC.parent/'research/ga_overnight_closure_20261102/launch/profiles/case_0007_inactive.json'
        self.profile=json.loads(original.read_text())
        domain=self.profile.pop('procedure')['domain']
        self.profile.update(schema=ep.SCHEMA_V3,kind=ep.COMPUTED_KIND,review_status='not_applicable_computed',
            slab_action_assertions=None,review=None,common_action_procedure=cp.pending_procedure(domain,self.sources))
        self.profile['provenance']['purpose']='TEST FIXTURE: numerical parser only; not engineering evidence.'
        self.path=self.root/'profile.json'

    def write(self,path,value):
        path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'description':'TEST FIXTURE numerical evidence'}

    def load(self):
        self.write(self.path,self.profile)
        return ep.load(self.path,allow_test_fixture=True)

    def proofs(self,actual=12.,reference=12.005):
        procedure=self.profile['common_action_procedure']
        a=self.write(self.root/'actual.json',{'actual':actual})
        b=self.write(self.root/'reference.json',{'expected':reference})
        basis=self.write(self.root/'criterion.json',{'purpose':'TEST FIXTURE arithmetic tolerance, not engineering approval'})
        procedure['protocol']['mesh_refinement'].update(absolute_tolerances={'moment_kip_in':.01,'shear_kip':.01},tolerance_basis=basis)
        for i,name in enumerate(ca.ROW_IDS):
            proof={'schema':cp.PROOF_SCHEMA,'row_id':name,'source_identity_sha256':ca.sha256(self.sources),
                'domain_sha256':ca.sha256(procedure['domain']),'protocol_sha256':ca.sha256(procedure['protocol']),
                'independent_method':'TEST FIXTURE separate scalar oracle','coverage_basis':basis,
                'comparisons':[{'name':'TEST FIXTURE only','actual':{'file':a,'json_path':['actual']},
                    'reference':{'file':b,'json_path':['expected']},'comparison':'close','absolute_tolerance':.01,
                    'relative_tolerance':0.,'criterion_source':basis}]}
            procedure['proofs'][name]=[self.write(self.root/f'proof_{i}.json',proof)]

    def test_pending_load_preserves_all_legacy_flags_false(self):
        loaded=self.load();cfg=loaded.apply(DesignConfig())
        self.assertEqual(loaded.procedure_status,'pending')
        self.assertFalse(loaded.asserted)
        self.assertFalse(cfg.slab_actions.all_asserted())
        self.assertTrue(all(asdict(cfg.slab_actions)[k] is False for k in ep.ASSERTION_FLAGS))
        self.assertEqual(asdict(cfg.slab_actions)['asserted_by'],'')
        self.assertIsNone(cfg.floor_analysis.slab_procedure)
        self.assertEqual(cfg.floor_analysis.common_action_procedure['kind'],ep.COMPUTED_KIND)
        self.assertEqual(cfg.floor_analysis.common_action_procedure['procedure']['schema'],cp.SCHEMA)

    def test_explicit_local_basis_is_pending_and_reaches_candidate_analysis(self):
        from Design.SMRF_Bar_Pitch_Layers import METHOD as PITCH
        from Design.SMRF_Common_Strength_Stream import LOCAL_PITCH_BASIS
        from Design.SMRF_Computed_Coordination import analyze
        procedure=self.profile['common_action_procedure']
        self.profile['common_action_procedure']=cp.pending_procedure(procedure['domain'],self.sources,
            material_mapping=PITCH,strength_material_action_basis=LOCAL_PITCH_BASIS)
        loaded=self.load();cfg=loaded.apply(DesignConfig());self.assertEqual(loaded.procedure_status,'pending')
        with patch('Design.SMRF_Common_Action_Analysis.analyze_installed',return_value={'explicit_controller_double':True}) as producer:
            analyze(None,cfg,{},1.)
        self.assertEqual(producer.call_args.kwargs['material_mapping'],PITCH)
        self.assertEqual(producer.call_args.kwargs['strength_material_action_basis'],LOCAL_PITCH_BASIS)
        self.assertFalse(producer.call_args.kwargs['final_assessment']);self.assertFalse(loaded.asserted)

    def test_missing_old_or_incompatible_material_protocol_cannot_load(self):
        from Design.SMRF_Common_Strength_Stream import LOCAL_PITCH_BASIS
        original=copy.deepcopy(self.profile['common_action_procedure'])
        for change in ('missing','old','incompatible'):
            procedure=copy.deepcopy(original);self.profile['common_action_procedure']=procedure
            if change=='missing':procedure['protocol'].pop('material_action_basis')
            elif change=='old':procedure['schema']='computed_common_actions_v1'
            else:procedure['protocol']['material_action_basis']['strength_material_action_basis']=LOCAL_PITCH_BASIS
            with self.assertRaises(ep.EvidenceProfileError):self.load()

    def test_changing_material_choice_invalidates_existing_method_proofs(self):
        from Design.SMRF_Bar_Pitch_Layers import METHOD as PITCH
        from Design.SMRF_Common_Strength_Stream import LOCAL_PITCH_BASIS
        self.proofs()
        self.profile['common_action_procedure']['protocol']['material_action_basis']=dict(
            floor_material_mapping=PITCH,strength_material_action_basis=LOCAL_PITCH_BASIS)
        with self.assertRaisesRegex(ep.EvidenceProfileError,'exact method/domain/protocol'):self.load()

    def test_no_synthesized_reviewer_or_legacy_assertions(self):
        for key,value in [('review',{'reviewed_by':'invented'}),('slab_action_assertions',{'verified':True})]:
            with self.subTest(key=key):
                old=self.profile[key];self.profile[key]=value
                with self.assertRaises(ep.EvidenceProfileError):self.load()
                self.profile[key]=old

    def test_ready_flag_not_an_input(self):
        self.profile['common_action_procedure']['status']='computed_ready'
        with self.assertRaises(ep.EvidenceProfileError):self.load()

    def test_complete_numeric_proofs_derive_status_only(self):
        self.proofs();loaded=self.load();summary=loaded.summary()
        self.assertEqual(summary['procedure_status'],'computed_ready')
        self.assertFalse(summary['asserted']);self.assertIsNone(summary['review'])
        self.assertEqual(len(summary['computed_proof_assessment']['proof_checks']),13)

    def test_comparison_failure_stays_computed_failure(self):
        self.proofs(actual=15.)
        self.assertEqual(self.load().procedure_status,'computed_failed')

    def test_each_missing_proof_remains_pending(self):
        self.proofs();procedure=self.profile['common_action_procedure']
        for name in ca.ROW_IDS:
            with self.subTest(row=name):
                refs=procedure['proofs'][name];procedure['proofs'][name]=[]
                self.assertEqual(self.load().procedure_status,'pending')
                procedure['proofs'][name]=refs

    def test_source_change_cannot_be_ready(self):
        self.profile['common_action_procedure']['method']['source_sha256'][self.source]='0'*64
        self.assertEqual(self.load().procedure_status,'pending_method_changed')

    def test_changed_domain_refuses_old_proof(self):
        self.proofs();self.profile['common_action_procedure']['domain']['beam_fc_ksi'].append(15.)
        with self.assertRaisesRegex(ep.EvidenceProfileError,'exact method/domain/protocol'):self.load()

    def test_changed_protocol_refuses_old_proof(self):
        self.proofs();self.profile['common_action_procedure']['protocol']['analysis_limits']['max_case_count']=65
        with self.assertRaisesRegex(ep.EvidenceProfileError,'exact method/domain/protocol'):self.load()

    def test_changed_raw_file_refuses_profile(self):
        self.proofs();(self.root/'actual.json').write_text('{"actual": 99}')
        with self.assertRaisesRegex(ep.EvidenceProfileError,'absent or changed'):self.load()

    def test_boolean_observation_is_not_numeric_evidence(self):
        self.proofs(actual=True)
        with self.assertRaisesRegex(ep.EvidenceProfileError,'non-boolean'):self.load()

    def test_same_file_is_not_independent_reference(self):
        self.proofs();procedure=self.profile['common_action_procedure'];name=ca.ROW_IDS[0]
        path=self.root/'proof_0.json';proof=json.loads(path.read_text())
        proof['comparisons'][0]['reference']=proof['comparisons'][0]['actual']
        procedure['proofs'][name]=[self.write(path,proof)]
        with self.assertRaisesRegex(ep.EvidenceProfileError,'same file'):self.load()

    def test_named_review_discriminator_not_accepted_on_computed_schema(self):
        self.profile['kind']='reviewed_procedure'
        with self.assertRaises(ep.EvidenceProfileError):self.load()

    def test_fixture_text_rejected_by_real_runner_loader(self):
        self.write(self.path,self.profile)
        with self.assertRaisesRegex(ep.EvidenceProfileError,'not a computed run'):ep.load(self.path)

    def test_configuration_and_profile_identity_change_with_actual_limits(self):
        first=self.load();first_identity=first.identity();first_cfg=asdict(first.apply(DesignConfig()))
        self.profile['common_action_procedure']['protocol']['analysis_limits']['max_case_count']=65
        second=self.load()
        self.assertNotEqual(first_identity,second.identity())
        self.assertNotEqual(first_cfg,asdict(second.apply(DesignConfig())))

    def test_legacy_apply_clears_computed_route(self):
        cfg=self.load().apply(DesignConfig())
        original=RC.parent/'research/ga_overnight_closure_20261102/launch/profiles/case_0007_inactive.json'
        self.assertIsNone(ep.load(original).apply(cfg).floor_analysis.common_action_procedure)


if __name__=='__main__':unittest.main()
