"""Catalog retention and actual-failure feedback, no candidate acceptance."""
import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import Structure_Parameters as sp
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Computed_Coordination import mat_feedback,ordered_mat_trials


class MatCoordinationTests(unittest.TestCase):
    def test_final_capacity_recomputes_scwb_and_hoop_settlement(self):
        from Design.SMRF_Computed_Coordination import _coordinate_one_layout
        from Design import Grouped_Design as gd,Group_Capacity as capacities,Group_Selection as selection
        for final_hoops in ({},{'beam':object()}):
            with self.subTest(final_hoops=bool(final_hoops)),ExitStack() as stack:
                state=SimpleNamespace(identity=lambda:'same_cages')
                common=dict(state_identity='same_cages',inventory=dict(layout_sha256=_hash({})),
                    definition=dict(slab_thickness_in=8.),input_signature='same_input',
                    combinations=[dict(id='C')],actions=[],capacity_gravity_transfer={},compatibility_actions={})
                slab=dict(reinforcement={});old=dict(name='before_final_slab');final=dict(name='final_slab')
                stack.enter_context(patch.object(sp,'SLAB_REINFORCEMENT',dict(layout={})))
                stack.enter_context(patch.object(sp,'SLAB_THICKNESS_IN',8.))
                stack.enter_context(patch('Model.Member_Groups.install'))
                stack.enter_context(patch('Design.SMRF_Computed_Coordination.refresh_slab'))
                stack.enter_context(patch('Design.SMRF_Slab_Reinforcement.slab_strength_eligibility',return_value={}))
                stack.enter_context(patch.object(selection,'select_reinforcement',return_value=(state,dict(settled=True,exhausted_groups=[]))))
                stack.enter_context(patch.object(capacities,'build_group_capacity_design',side_effect=[old,old,final]))
                stack.enter_context(patch.object(capacities,'hoop_updates',side_effect=lambda cap,s:final_hoops if cap is final else {}))
                stack.enter_context(patch.object(gd,'joint_check_inputs',side_effect=lambda cap,s:cap))
                stack.enter_context(patch('Design.SMRF_Joints.evaluate_joints',side_effect=lambda cap:[dict(id='scwb',status='pass',capacity=cap['name'])]))
                stack.enter_context(patch.object(gd,'scwb_constraint',side_effect=lambda checks,cap:dict(capacity=cap['name'],checks=checks)))
                stack.enter_context(patch('Design.SMRF_Common_Action_Analysis.assess_installed',return_value=common))
                result=_coordinate_one_layout(state,SimpleNamespace(),{},slab,1.,common,maximum_passes=1)
                settled=result['settled'];self.assertIs(settled['capacity'],final)
                self.assertEqual(settled['scwb']['capacity'],'final_slab')
                self.assertEqual(settled['scwb']['checks'],settled['joint_checks'])
                self.assertEqual(settled['hoops_settled'],not bool(final_hoops))
                self.assertEqual(result['layout_settled'],not bool(final_hoops))

    def layout(self,value):return dict(layers={n:dict(area_in2_per_ft=value) for n in ('x_top','x_bottom','y_top','y_bottom')})
    def test_unknown_does_not_trigger_invented_capacity_failure(self):
        value=mat_feedback(dict(common_strength=dict(status='unknown'),local_strength=None),dict(checks=[dict(id='pending',status='not_evaluated')]))
        self.assertEqual(value['known_failures'],[]);self.assertFalse(value['unknown_converted_to_capacity_failure'])
    def test_actual_local_inventory_deficit_drives_named_layer_only(self):
        common=dict(local_checks=dict(records=[dict(id='support',local_minimum=dict(checks={
            'x_top':dict(status='fail',required_in2=3.,credit=dict(installed_area_in2=2.)),
            'y_top':dict(status='unknown',required_in2=5.,credit=dict(installed_area_in2=2.))}))]))
        value=mat_feedback(common,dict(checks=[]))
        self.assertEqual(value['mat_density_target_ratios']['x_top'],1.5)
        self.assertEqual(value['mat_density_target_ratios']['y_top'],1.)
        self.assertEqual(len(value['known_failures']),1)
    def test_priority_retains_every_untried_original_catalog_proposal(self):
        candidates=[dict(candidate_id=str(v),layout=self.layout(v),layout_sha256=str(v),nominal_steel_area_sum_in2_per_ft=4*v) for v in (.5,1.,1.5,2.,3.)]
        catalog=dict(candidates=candidates);original=copy.deepcopy(catalog)
        feedback=dict(mat_density_target_ratios={n:1.8 for n in self.layout(1.)['layers']})
        result=ordered_mat_trials(catalog,self.layout(1.),{'1.0'},feedback)
        self.assertEqual([r['candidate_id'] for r in result],['2.0','3.0','1.5','0.5'])
        self.assertEqual({r['candidate_id'] for r in result},{r['candidate_id'] for r in candidates}-{'1.0'})
        self.assertEqual(catalog,original)

    def test_actual_trial_change_requires_new_input_and_counts_every_period(self):
        from Design.SMRF_Computed_Coordination import coordinate
        layouts=[self.layout(v) for v in (1.,2.,3.)]
        candidates=[dict(candidate_id=str(i),layout=x,layout_sha256=_hash(x),nominal_steel_area_sum_in2_per_ft=4*(i+1)) for i,x in enumerate(layouts)]
        catalog=dict(payload_sha256='complete_catalog',enumerated_count=3,candidates=candidates)
        context=dict(computed_slab_catalog=catalog);state=SimpleNamespace(identity=lambda:'unchanged_cages');calls=[]
        common=dict(inventory=dict(layout_sha256=_hash(layouts[0])),input_signature='initial')
        slab=dict(original_catalog_sha256=catalog['payload_sha256'],reinforcement=dict(layout=layouts[0],checks=[]))
        def one(current,cfg,ctx,slab,period,fields,**kw):
            actual=_hash(sp.SLAB_REINFORCEMENT['layout'])
            self.assertEqual(fields['inventory']['layout_sha256'],actual)
            calls.append(('coordinate',actual,fields['input_signature']))
            slab['reinforcement']=dict(layout=copy.deepcopy(sp.SLAB_REINFORCEMENT['layout']),checks=[])
            sp.SLAB_REINFORCEMENT=slab['reinforcement']
            fields=dict(fields,common_strength=dict(status='fail' if len(calls)==1 else 'pass',payload_sha256='explicit_test_strength'))
            fields['payload_sha256']=_hash(fields)
            return dict(state=current,slab=slab,period=period,common=fields,layout_settled=True,
                settled=dict(common_coordination=dict(additional_period_solves=2)))
        def analyze(current,cfg,ctx,period):
            actual=_hash(sp.SLAB_REINFORCEMENT['layout']);calls.append(('new_actions',actual))
            return dict(status='responses_computed',inventory=dict(layout_sha256=actual),input_signature='new_actual_mat')
        with ExitStack() as stack:
            stack.enter_context(patch.object(sp,'SLAB_REINFORCEMENT',dict(layout=copy.deepcopy(layouts[0]))))
            stack.enter_context(patch('Design.SMRF_Computed_Coordination._coordinate_one_layout',side_effect=one))
            stack.enter_context(patch('Design.SMRF_Computed_Coordination.analyze',side_effect=analyze))
            stack.enter_context(patch('Model.Member_Bar_Layers.row_audit',return_value=dict(valid=True)))
            stack.enter_context(patch('Design.SMRF_Joint_Mat_Physical.global_mat_placement',return_value=dict(status='pass')))
            stack.enter_context(patch('Design.SMRF_Physical_Floor_Materials.installed_inventory',side_effect=lambda current:dict(layout_sha256=_hash(sp.SLAB_REINFORCEMENT['layout']))))
            period=stack.enter_context(patch('Design.Perimeter_Systems.model_period',return_value=1.2))
            result=coordinate(state,SimpleNamespace(),context,slab,1.,common,maximum_mat_trials=3)
            self.assertEqual([c[0] for c in calls],['coordinate','new_actions','coordinate'])
            self.assertNotEqual(calls[0][1],calls[1][1]);self.assertEqual(calls[1][1],calls[2][1])
            self.assertEqual(result['common']['input_signature'],'new_actual_mat')
            self.assertEqual(period.call_count,1)
            details=result['settled']['common_coordination'];self.assertEqual(details['additional_period_solves'],5)
            self.assertEqual(details['mat_coordination']['original_catalog_count'],3)
            self.assertEqual(details['mat_coordination']['untried_geometry_candidate_count'],1)
