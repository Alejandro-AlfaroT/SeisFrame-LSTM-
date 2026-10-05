"""Failure injection for batch identities and portable M1 qualification consumption."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
from Data_Generation import Run_Research_Batch as batch
from Data_Generation import Research_Batch_Evidence as evidence


class SourceAndCollection(unittest.TestCase):
    def setUp(self):
        self.source = {'design_source_aggregate_sha256': batch._sha256_json({'a.py':'aaa'}),
                       'run_chain_sha256': {'runner.py': 'bbb'}, 'python': '3.12'}
        self.effective = {'accepted': True, 'failed_check_ids': [], 'open_check_ids': []}
        self.item = {'index': 0, 'case': {'case_id':'case_0001'}, 'design': {
            'design_sha256':'design', 'result_sha256':'result', 'm1_addendum_sha256':'m1',
            'effective_design_sha256':batch._sha256_json(self.effective)}}
        self.plan = {'schema':batch.PLAN_SCHEMA, 'source':self.source, 'cases':[self.item]}
        self.plan['plan_sha256'] = batch._sha256_json(self.plan)
        self.status = {'case_id':'case_0001', 'plan_sha256':self.plan['plan_sha256'],
                       'state':'completed', 'training_eligible':True, 'smoke_test':False,
                       'effective_design':self.effective}
        self.identity = {'plan_sha256':self.plan['plan_sha256'], 'case':self.item['case'],
                         'source':self.source, 'design_source_matches_this_tree':True,
                         'result_sha256':'result', 'design':{'design_sha256':'design'},
                         'm1_addendum':{'sha256':'m1'}, 'effective_design':self.effective}

    def test_source_and_environment_must_match_and_design_identity_is_required(self):
        record = {'request_identity':{'source_sha256':{'a.py':'aaa'}}}
        batch.require_source_match(self.plan, self.source, record)
        for change in ({'python':'3.13'}, {'run_chain_sha256':{'runner.py':'changed'}}):
            with self.assertRaisesRegex(RuntimeError, 'Source/environment'):
                batch.require_source_match(self.plan, {**self.source, **change}, record)
        for bad in ({}, {'source_sha256':{}}, {'source_sha256':{'a.py':'changed'}}):
            with self.assertRaisesRegex(RuntimeError, 'Design source'):
                batch.require_source_match(self.plan, self.source, bad)

    def test_plan_rejects_tampering_old_schema_and_duplicate_cases(self):
        batch.validate_plan(self.plan)
        edited = copy.deepcopy(self.plan); edited['source']['python']='changed'
        with self.assertRaisesRegex(RuntimeError, 'edited'):
            batch.validate_plan(edited)
        for mutation in ('legacy','duplicate'):
            bad = copy.deepcopy(self.plan); bad.pop('plan_sha256')
            if mutation == 'legacy': bad['schema']='seisframe_v2_research_ntha_plan_v1'
            else: bad['cases'].append(copy.deepcopy(bad['cases'][0]))
            bad['plan_sha256']=batch._sha256_json(bad)
            with self.assertRaises(RuntimeError): batch.validate_plan(bad)

    def test_collection_refuses_foreign_plan_case_source_design_or_addendum(self):
        batch.validate_collected_case(self.plan,self.item,self.status,self.identity)
        for field, value in [('plan_sha256','foreign'),('case_id','case_9999')]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                batch.validate_collected_case(self.plan,self.item,{**self.status,field:value},self.identity)
        for field, value in [('source',{}),('case',{}),('design',{}),('result_sha256','foreign'),
                             ('m1_addendum',None),('design_source_matches_this_tree',False)]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                batch.validate_collected_case(self.plan,self.item,self.status,{**self.identity,field:value})

    def test_tampered_effective_qualification_cannot_be_laundered_through_status(self):
        changed = {**self.effective, 'extra':'forged'}
        with self.assertRaises(RuntimeError):
            batch.validate_collected_case(self.plan,self.item,{**self.status,'effective_design':changed},
                                          {**self.identity,'effective_design':changed})
        for bad in ({'state':'analysis_failed'}, {'smoke_test':True}):
            with self.assertRaises(RuntimeError):
                batch.validate_collected_case(self.plan,self.item,{**self.status,**bad},self.identity)

    def test_collection_rejects_extra_case_without_writing_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/batch.PLAN_NAME).write_text(json.dumps(self.plan))
            (root/'case_9999').mkdir(); (root/'case_9999/run_status.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Unplanned'):
                batch.summarize(root)
            self.assertFalse((root/'batch_status.json').exists())

    def test_worker_rejects_stale_source_before_model_build_or_shaking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); designs=root/'designs'; case=designs/'case_0001'; case.mkdir(parents=True)
            (case/'design.json').write_text(json.dumps({'source_sha256':{'a.py':'stale'}}))
            state={'status':'designed','fail_ids':[],'not_evaluated_ids':[],
                   'design_sha256':batch._sha256_file(case/'design.json'),'profile_id':'test'}
            (case/'result.json').write_text(json.dumps(state))
            plan=copy.deepcopy(self.plan); plan.pop('plan_sha256')
            plan['cases'][0].update(alpha=1.0,record={'pair_key':'pair'})
            plan['cases'][0]['design'].update(design_sha256=state['design_sha256'],
                                             result_sha256=batch._sha256_file(case/'result.json'))
            plan['plan_sha256']=batch._sha256_json(plan)
            (root/batch.PLAN_NAME).write_text(json.dumps(plan)); (root/'case_0001').mkdir()
            with patch.object(batch,'source_identity',return_value=self.source), \
                 patch('Model.Build_Model.build_model') as build, patch('Ground_Motion_Main.run_one') as run:
                result=batch.run_case(root,'case_0001',[designs])
            self.assertEqual(result['state'],'error')
            self.assertIn('Design source',result['reason'])
            build.assert_not_called(); run.assert_not_called()


class AddendumConsumption(unittest.TestCase):
    def test_portable_evidence_uses_roles_not_original_machine_paths(self):
        old={'design_path':r'C:\old\case_0001\design.json','review_path':r'C:\old\review\case_0001\review.json',
             'assertion_sha256':'a','evidence_sha256':{r'C:\old\case_0001\design.json':'d',
             r'C:\old\review\case_0001\pm65\x_high_0.json':'r',
             r'C:\old\torsional_strength_assertion.json':'a'}}
        expected={'design':'d','review/pm65/x_high_0.json':'r','assertion':'a'}
        self.assertEqual(evidence.portable_evidence(old), expected)
        bad=copy.deepcopy(old); bad['evidence_sha256'][r'C:\old\review\case_0001\..\escape.json']='x'
        with self.assertRaises(ValueError): evidence.portable_evidence(bad)

    def test_missing_review_and_duplicate_addenda_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'case_0001').mkdir()
            (root/'case_0001/qualification_addendum.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'review evidence'):
                evidence.load_case_addendum('case_0001','unused',[root],[])
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                evidence.load_case_addendum('case_0001','unused',[root,root],[])

    def test_validator_reconstructs_and_rejects_altered_qualification_or_evidence(self):
        saved={'design_path':'/old/design.json','review_path':'/old/review/review.json',
               'assertion_sha256':'a','evidence_sha256':{'/old/design.json':'d',
                 '/old/review/review.json':'r','/old/torsional_strength_assertion.json':'a'},
               'method':'method','qualification':{'accepted':True,'checks':[],'tuple_field':(1,2)},'assertion':{},
               'production_acceptance':False}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'addendum.json'
            with patch('tools.assert_torsional_strength.apply_assertion', return_value=saved) as replay:
                p.write_text(json.dumps(saved))
                self.assertEqual(evidence.validated_addendum('design',p,'review')['method'],'method')
                self.assertEqual(replay.call_count,1)
                for key,value in [('qualification',{'accepted':False}),('evaluator_sha256','changed')]:
                    p.write_text(json.dumps({**saved,key:value}))
                    with self.assertRaisesRegex(ValueError,'differs'):
                        evidence.validated_addendum('design',p,'review')
                changed=copy.deepcopy(saved); changed['evidence_sha256']['/old/review/review.json']='tampered'
                p.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'differs'):
                    evidence.validated_addendum('design',p,'review')

    def test_effective_state_preserves_original_and_other_checks(self):
        original={'accepted':False,'counts':{},'open_check_ids':['m1'],'failed_check_ids':[], 'probe_assertions':True}
        q={'accepted':False,'counts':{'pass':1,'fail':1,'not_evaluated':0},'checks':[
            {'id':'m1','status':'pass'},{'id':'other','status':'fail'}]}
        result=evidence.effective_state(original,{'qualification':q})
        self.assertEqual(result['failed_check_ids'],['other'])
        self.assertEqual(original['open_check_ids'],['m1'])
        self.assertTrue(result['probe_assertions'])


class HistoryCoverage(unittest.TestCase):
    def test_empty_short_or_invalid_index_histories_cannot_complete(self):
        analysis={'npts_requested':3,'completed_steps':3}
        checks={'rows':3,'springs':2,'scheduled_steps_complete':True,'commit_counts_valid':True}
        self.assertTrue(batch.complete_history(analysis,checks))
        for change in ({'rows':0},{'rows':2},{'springs':0},{'scheduled_steps_complete':False},{'commit_counts_valid':False}):
            self.assertFalse(batch.complete_history(analysis,{**checks,**change}))
        self.assertFalse(batch.complete_history({'npts_requested':0,'completed_steps':0},checks))

    def test_history_reader_rejects_nonfinite_time_and_missing_step(self):
        import numpy as np
        arrays={'rotation_rad':np.zeros((3,1)),'moment_kip_in':np.zeros((3,1)),
                'time_sec':np.array([.1,.2,.3]),'commit_count':np.array([1,3,4]),
                'scheduled_step':np.arange(3),'gravity_rotation_rad':np.zeros(1)}
        rows=[{'yield_moment_positive_kip_in':'1','yield_moment_negative_kip_in':'1',
               'member_class':'beam','local_axis':'y'}]
        with patch('Analysis.Hinge_Moment_Rotation.read_hinge_moment_rotation',return_value=(arrays,rows,{})):
            checks=batch.history_checks('unused')
            self.assertTrue(checks['scheduled_steps_complete'])
            self.assertTrue(checks['commit_counts_valid'])  # recovered substeps may increase commit count by >1
            arrays['time_sec'][1]=np.nan
            arrays['scheduled_step'][1]=2
            arrays['commit_count'][1]=1
            checks=batch.history_checks('unused')
            self.assertFalse(checks['time_strictly_increasing'])
            self.assertFalse(checks['scheduled_steps_complete'])
            self.assertFalse(checks['commit_counts_valid'])


if __name__ == '__main__': unittest.main()
