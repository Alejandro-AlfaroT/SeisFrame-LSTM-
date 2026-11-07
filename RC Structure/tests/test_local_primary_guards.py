"""Actual source recipe replay for the existing analytical guard-plan fixture."""
import copy
import json
import unittest
from unittest.mock import patch
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Common_Guard_Plan import build_common_guard_plan,_PhysicalSource,LOCAL_PLAN_CONTRACT
from Design.SMRF_Common_Action_Analysis import physical_response_bundles
from Design.SMRF_Local_Pitch_Strength import bind_local_pitch_action,_check_point
from Design.SMRF_Local_Primary_Guards import reconstruct_primary_guard_recipe
from Design.SMRF_Primary_Diaphragm_Batch import analyze_primary_diaphragms
from tests.test_common_guard_plan import fixture,settings


def local_fixture():
    common,plan,projector,state=fixture()
    sources={b['source_id']:_PhysicalSource(b,common) for b in physical_response_bundles(common)}
    local=copy.deepcopy(plan)
    for spec in local['cut_specs']:
        spec['material_scope']='slab_only' if spec.get('member_id') is None else 'common_slab_web'
        source=sources[spec['source_id']];cid=plan['cuts'][spec['cut_id']]['identity']['combination_id']
        raw=source.cut(spec,cid)
        local['cuts'][spec['cut_id']]=bind_local_pitch_action(raw,floor=spec['floor'],inventory=common['inventory'],
            analysis_input_identity=common['input_signature'],source_response_sha256=source.bundle['response_payload_sha256'],
            definition_sha256=common['definition_sha256'])
    local['action_contract']=LOCAL_PLAN_CONTRACT
    local['payload_sha256']=_hash({k:v for k,v in local.items() if k!='payload_sha256'})
    return common,plan,local,projector,state


class LocalPrimaryGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common,old,local,p,cls.state=local_fixture();cls.local_plan=local
        cls.p5=analyze_primary_diaphragms(cls.common,**settings());cls.projector=p
        with patch('Design.SMRF_Common_Guard_Plan.installed_full_t_projector',return_value=p):
            cls.legacy=build_common_guard_plan(cls.common,old,cls.p5,state=cls.state,maximum_cuts=500)
            cls.result=build_common_guard_plan(cls.common,local,cls.p5,state=cls.state,maximum_cuts=500,local_pitch_actions=True)

    def test_all_132_original_signed_seven_forces_and_124_guards_reproduce(self):
        self.assertEqual(self.result['required_cut_ids'],self.legacy['required_cut_ids'])
        self.assertEqual(len(self.result['cuts']),132)
        guards=0;moments=0;simultaneous=0
        for spec in self.result['cut_specs']:
            row=self.result['cuts'][spec['cut_id']];old=self.legacy['cuts'][spec['cut_id']]
            self.assertEqual(row['resultants'],old['resultants'])
            self.assertEqual(row['accounting'],old['accounting'])
            self.assertEqual(row['identity'],old['identity'])
            _check_point(row,spec,self.common['inventory'])
            if row.get('strength_guard'):
                guards+=1
                self.assertFalse(row['accounting']['physical_equilibrium_solution_claimed'])
                moments+=row['strength_guard']['guard_identity'].startswith('primary_M/')
                simultaneous+=bool(row['strength_guard'].get('simultaneous_primary_membrane_and_moment'))
            self.assertEqual(spec['material_scope'],'slab_only' if spec.get('member_id') is None else 'common_slab_web')
        self.assertEqual(guards,124);self.assertEqual(moments,12);self.assertGreater(simultaneous,0)
        for group in self.result['primary_guard_inventory']:self.assertEqual(len(group['complete_P5_branch_ids']),7)
        self.assertFalse(self.result['mesh_qualification_inferred'])

    def test_explicit_local_contract_and_complete_budget_required(self):
        with self.assertRaisesRegex(ValueError,'contract'):
            build_common_guard_plan(self.common,self.local_plan,self.p5,state=self.state,maximum_cuts=500)
        old=copy.deepcopy(self.local_plan);old.pop('action_contract')
        with self.assertRaisesRegex(ValueError,'contract'):
            build_common_guard_plan(self.common,old,self.p5,state=self.state,maximum_cuts=500,local_pitch_actions=True)
        with patch('Design.SMRF_Common_Guard_Plan.installed_full_t_projector',return_value=self.projector):
            with self.assertRaisesRegex(ValueError,'budget'):
                build_common_guard_plan(self.common,self.local_plan,self.p5,state=self.state,maximum_cuts=131,local_pitch_actions=True)

    def test_saved_json_roundtrip_retains_exact_recipes(self):
        restored=json.loads(json.dumps(self.result,allow_nan=False))
        for spec in restored['cut_specs']:
            _check_point(restored['cuts'][spec['cut_id']],spec,self.common['inventory'])

    def test_saved_guard_geometry_action_recipe_and_selector_tamper_refused(self):
        spec=next(s for s in self.result['cut_specs'] if 'primary_M/both' in s.get('guard_branch_id',''))
        original=self.result['cuts'][spec['cut_id']]
        def bad_force(row):row['resultants'][0]+=.1
        def bad_point(row):row['local_point_binding']['point']['station_in']+=.1
        def bad_branch(row):row['local_point_binding']['branch_id']='primary_M/strong'
        def bad_location(row):row['local_point_binding']['location_id']='other'
        def bad_projector(row):row['local_point_binding']['recipe']['arguments']['projector_parameters']['web_E_ksi']*=2
        def bad_raw(row):row['local_point_binding']['recipe']['arguments']['physical_cuts'][0]['slab']['station_in']+=1
        for change in (bad_force,bad_point,bad_branch,bad_location,bad_projector,bad_raw):
            with self.subTest(change=change.__name__):
                row=copy.deepcopy(original);change(row)
                with self.assertRaises((ValueError,KeyError)):_check_point(row,spec,self.common['inventory'])
        changed=copy.deepcopy(spec);changed['trace']='minus'
        with self.assertRaisesRegex(ValueError,'geometry'):_check_point(original,changed,self.common['inventory'])

    def test_P5_summary_recomputed_even_with_refreshed_recipe_digest(self):
        row=next(r for r in self.result['cuts'].values() if r.get('local_point_binding',{}).get('kind')=='primary_guard'
            and r['local_point_binding']['recipe']['kind']=='membrane')
        recipe=copy.deepcopy(row['local_point_binding']['recipe'])
        recipe['arguments']['primary_membrane_sources'][0]['slab_membrane_integral_kip'][0]+=.1
        recipe['payload_sha256']=_hash({k:v for k,v in recipe.items() if k!='payload_sha256'})
        with self.assertRaisesRegex(ValueError,'summary differs'):reconstruct_primary_guard_recipe(recipe)


if __name__=='__main__':unittest.main()
