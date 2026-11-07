"""Independent mechanics and conservative-approximation checks; no OpenSees."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

from Design.SMRF_Story_Strength import (
    analyze_frame, adjacent_strength_bounds, lipschitz_interaction,
    rectangular_interaction, layered_interaction, transport_inner, analyze_gravity_family, verify_solution,
)
from Design.SMRF_Biaxial_Section import BiaxialSection


def box(moment, *, axial=(-10000.,10000.)):
    planes=[[0.,1.,float(moment)],[0.,-1.,float(moment)]]
    return {'nominal':True,'phi':1.,'axial_domain_kip':list(axial),
            'inner_halfplanes':planes,'outer_halfplanes':copy.deepcopy(planes)}


def portal(stories=1, *, column=100., beam=1000., shear=1000., base=None, joint=100000.):
    nodes=[{'id':f'{floor}/{j}','floor':floor,'x_in':20.*j,'z_in':10.*floor,'fixed_base':floor==0}
           for floor in range(stories+1) for j in range(2)]
    members=[];joints=[]
    for floor in range(1,stories+1):
        cap=column[floor-1] if isinstance(column,list) else column
        for j in range(2):
            moment=cap[j] if isinstance(cap,list) else cap
            members.append({'id':f'c{floor}/{j}','kind':'column','i':f'{floor-1}/{j}','j':f'{floor}/{j}',
                            'capacity_reference':'centerlines',
                            'interaction_i':box(base if floor==1 and base is not None else moment),
                            'interaction_j':box(moment),'nominal_shear_kip':shear})
        name=f'b{floor}'
        members.append({'id':name,'kind':'beam','i':f'{floor}/0','j':f'{floor}/1',
                        'capacity_reference':'centerlines',
                        'interaction_i':box(beam),'interaction_j':box(beam),
                        'span_interaction':box(beam),'nominal_shear_kip':10000.})
        for j,end in [(0,'Mi'),(1,'Mj')]:
            for sign in (-1.,1.):
                joints.append({'id':f'{floor}/{j}/{sign:g}','node':f'{floor}/{j}',
                               'sense':int(sign),
                               'terms':{name:{end:sign}},'limit':joint,'nominal':True,
                               'basis':'synthetic connection moment limit, independent portal closed-form test'})
    return {'state_identity':'synthetic-explicit-portal','source_identity':'algebra-test-not-candidate',
            'combination_id':'zero-gravity-static-strength','nodes':nodes,'members':members,'joint_constraints':joints}


class StoryResistance(unittest.TestCase):
    def test_physical_polynomial_gravity_matches_independent_uniform_load_model(self):
        from Design.SMRF_Story_Load_Field import METHOD
        original=portal(beam=50.);beam=next(m for m in original['members'] if m['kind']=='beam')
        beam['transverse_load_kip_per_in']=.5
        physical=copy.deepcopy(original);pbeam=next(m for m in physical['members'] if m['kind']=='beam')
        pbeam.pop('transverse_load_kip_per_in');pbeam['fixed_section_field']=dict(method=METHOD,source_identity='uniform_closed_form',pieces=[
            dict(start_fraction=0.,end_fraction=1.,P_offset_kip=[0.,0.],M_offset_kipin=[0.,100.,-100.],V_offset_kip=[5.,-10.])])
        reference=analyze_frame(original);actual=analyze_frame(physical)
        self.assertEqual(actual['status'],'bounded')
        for old,new in zip(reference['stories'][0]['senses'],actual['stories'][0]['senses']):
            for bound in ('inner','outer'):
                self.assertAlmostEqual(old[bound]['strength_kip'],new[bound]['strength_kip'],places=7)
                self.assertLess(verify_solution(physical,new[bound])['maximum_strength_constraint_violation'],1e-7)

    def test_piecewise_axial_transfer_cannot_hide_between_member_ends(self):
        from Design.SMRF_Story_Load_Field import METHOD
        model=portal();beam=next(m for m in model['members'] if m['kind']=='beam')
        beam['span_interaction']=box(1000.,axial=(-1.,1.))
        beam['fixed_section_field']=dict(method=METHOD,source_identity='balanced_internal_axial_transfer',pieces=[
            dict(start_fraction=0.,end_fraction=.37,P_offset_kip=[0.,10/.37],M_offset_kipin=[0.,0.,0.],V_offset_kip=[0.,0.]),
            dict(start_fraction=.37,end_fraction=1.,P_offset_kip=[10/.63,-10/.63],M_offset_kipin=[0.,0.,0.],V_offset_kip=[0.,0.])])
        result=analyze_frame(model)
        self.assertEqual(result['stories'][0]['senses'][0]['outer']['status'],'infeasible')

    def test_combined_axial_moment_facet_extremum_differs_from_moment_extremum(self):
        from Design.SMRF_Story_Load_Field import METHOD,validate,maximum_violation,evaluate
        field=dict(method=METHOD,source_identity='independent_completed_square',pieces=[dict(start_fraction=0.,end_fraction=1.,
            P_offset_kip=[-20.,40.],M_offset_kipin=[0.,100.,-100.],V_offset_kip=[5.,-10.])])
        interaction=box(25.);interaction['inner_halfplanes']=[[.5,1.,25.]]
        validate(field);worst,at=maximum_violation(field,0.,1.,0.,0.,0.,interaction,'inner')
        # .5*(40s-20)+100s(1-s)-25 = 1-100(s-.6)^2.
        self.assertAlmostEqual(worst,1.);self.assertAlmostEqual(at[0],.6)
        self.assertAlmostEqual(.5*evaluate(field,'P',.5)+evaluate(field,'M',.5)-25.,0.)

    def test_physical_field_transport_replays_every_continuous_strength_facet(self):
        from Design.SMRF_Story_Load_Field import METHOD
        reference=portal();reference['baseline_member_actions']={m['id']:{'P':0.,'Mi':0.,'Mj':0.} for m in reference['members']}
        target=copy.deepcopy(reference);target['combination_id']='physical_uniform_gravity'
        for m in target['members']:
            if m['kind']=='column':target['baseline_member_actions'][m['id']]['P']=.1
            else:m['fixed_section_field']=dict(method=METHOD,source_identity='uniform_w_point01',pieces=[
                dict(start_fraction=0.,end_fraction=1.,P_offset_kip=[0.,0.],M_offset_kipin=[0.,2.,-2.],V_offset_kip=[.1,-.2])])
        original=analyze_frame(reference)
        transported=transport_inner(reference,original['stories'][0]['senses'][0]['inner'],target,1,1)
        self.assertEqual(transported['status'],'feasible_lower_bound')
        self.assertLess(verify_solution(target,transported)['maximum_equilibrium_residual'],1e-8)

    def test_fixed_field_preserves_one_sided_jumps_and_rejects_double_loaded_field(self):
        from Design.SMRF_Story_Load_Field import METHOD,validate,evaluate,maximum_violation
        field=dict(method=METHOD,source_identity='internal_balanced_moment_couples',pieces=[
            dict(start_fraction=0.,end_fraction=.4,P_offset_kip=[0.,0.],M_offset_kipin=[0.,0.,0.],V_offset_kip=[0.,0.]),
            dict(start_fraction=.4,end_fraction=1.,P_offset_kip=[0.,0.],M_offset_kipin=[50.,-50.,0.],V_offset_kip=[-2.5,0.])])
        validate(field);self.assertEqual(evaluate(field,'M',.4,side='left'),0.)
        self.assertEqual(evaluate(field,'M',.4,side='right'),30.)
        self.assertAlmostEqual(maximum_violation(field,0.,1.,0.,0.,0.,box(20.),'inner')[0],10.)
        model=portal();beam=next(m for m in model['members'] if m['kind']=='beam')
        beam.update(fixed_section_field=field,transverse_load_kip_per_in=1.)
        with self.assertRaisesRegex(ValueError,'exactly once'):analyze_frame(model)

    def strength(self,model):
        result=analyze_frame(model,time_limit_seconds=10.)
        self.assertEqual(result['status'],'bounded')
        for story in result['stories']:
            self.assertAlmostEqual(story['lower_kip'],story['upper_kip'],places=7)
            for sense in story['senses']:
                for bound in ('inner','outer'):
                    c=sense[bound]['certificate']
                    self.assertLess(c['maximum_equilibrium_residual'],1e-7)
                    self.assertLess(c['primal_dual_gap'],1e-7)
        return [r['lower_kip'] for r in result['stories']]

    def test_column_mechanism_matches_four_hinges_over_height(self):
        self.assertAlmostEqual(self.strength(portal())[0],4*100/10)

    def test_beam_mechanism_includes_actual_base_hinges(self):
        self.assertAlmostEqual(self.strength(portal(beam=20))[0],(2*100+2*20)/10)

    def test_unequal_column_strengths_are_not_averaged_before_limiting(self):
        self.assertAlmostEqual(self.strength(portal(column=[[10,100]]))[0],2*(10+100)/10)

    def test_column_shear_controls_before_flexure(self):
        self.assertAlmostEqual(self.strength(portal(shear=5))[0],2*5)

    def test_base_capacity_is_explicit_not_doubled_from_roof(self):
        self.assertAlmostEqual(self.strength(portal(base=10))[0],(2*10+2*100)/10)

    def test_joint_transfer_controls_top_column_moments(self):
        self.assertAlmostEqual(self.strength(portal(joint=70))[0],(2*100+2*70)/10)

    def test_existing_lateral_load_is_not_mislabelled_incremental_strength(self):
        model=portal()
        for node in model['nodes']:
            if node['floor']==1:node['load']={'x_kip':5.}
        result=analyze_frame(model,time_limit_seconds=10.)
        self.assertAlmostEqual(result['stories'][0]['lower_kip'],40.)
        positive,negative=result['stories'][0]['senses']
        self.assertAlmostEqual(positive['inner']['additional_story_force_kip'],30.)
        self.assertAlmostEqual(negative['inner']['additional_story_force_kip'],50.)

    def test_joint_face_moments_use_the_clear_column_height(self):
        model=portal()
        for member in model['members']:
            if member['kind']=='column':
                member.update(capacity_reference='joint_faces',face_offset_i_in=1.,face_offset_j_in=1.)
        self.assertAlmostEqual(self.strength(model)[0],4*100/(10-2))

    def test_actual_axial_equilibrium_reduces_column_strength(self):
        model=portal()
        for node in model['nodes']:
            if node['floor']==1:node['load']={'z_kip':-100.}
        for member in model['members']:
            if member['kind']=='column':
                interaction={'nominal':True,'phi':1.,'axial_domain_kip':[0.,1000.],
                             'inner_halfplanes':[[.1,1.,100.],[.1,-1.,100.]],
                             'outer_halfplanes':[[.1,1.,100.],[.1,-1.,100.]]}
                member['interaction_i']=member['interaction_j']=interaction
        # Sum(P_left+P_right)=200; sum of four end limits=400-.2*200.
        self.assertAlmostEqual(self.strength(model)[0],(400-.2*200)/10)

    def test_two_story_column_transition_and_roof_roster(self):
        self.assertEqual(self.strength(portal(stories=2,column=[100.,20.])),[40.,8.])

    def test_two_story_beam_mechanism_enforces_adjacent_joint_equilibrium(self):
        # First-story force: both floors' beam hinge work and two base hinges.
        # Upper-story force pair: lower joints can mobilize their columns;
        # only roof beams restrict the upper column top moment sum.
        actual=self.strength(portal(stories=2,beam=20.))
        self.assertAlmostEqual(actual[0],(2*100+4*20)/10)
        self.assertAlmostEqual(actual[1],(2*100+2*20)/10)

    def test_uniform_beam_gravity_has_interior_moment_limit(self):
        model=portal(beam=20.)
        beam=next(m for m in model['members'] if m['kind']=='beam')
        beam['transverse_load_kip_per_in']=1.
        # Fixed-end and positive-span capacities cannot sustain wL^2/8=50
        # with only 20+20=40 kip-in of positive/negative plastic capacity.
        result=analyze_frame(model,time_limit_seconds=10.)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['stories'][0]['senses'][0]['inner']['status'],'infeasible')

    def test_exact_point_load_interior_hinge_is_not_missed(self):
        model=portal(beam=20.)
        next(m for m in model['members'] if m['kind']=='beam')['transverse_point_loads']=[[.37,10.]]
        # Simple support moment at load = 10*20*.37*.63=46.62;
        # two plastic span/end moment limits can carry at most 40.
        result=analyze_frame(model,time_limit_seconds=10.)
        self.assertEqual(result['stories'][0]['senses'][0]['outer']['status'],'infeasible')

    def test_parallel_lines_share_one_diaphragm_without_force_allocation(self):
        model=portal();other=portal(column=50.)
        for n in other['nodes']:n['id']='second:'+n['id']
        for m in other['members']:
            m['id']='second:'+m['id'];m['i']='second:'+m['i'];m['j']='second:'+m['j'];m['line_id']='second'
        for j in other['joint_constraints']:
            j['id']='second:'+j['id'];j['node']='second:'+j['node'];j['terms']={'second:'+mid:v for mid,v in j['terms'].items()}
        for key in ('nodes','members','joint_constraints'):model[key]+=other[key]
        self.assertAlmostEqual(self.strength(model)[0],40.+20.)

    def test_joint_traction_inner_outer_interval_is_derived_without_lever_arm(self):
        from Design.SMRF_Story_Adapter import joint_traction_bounds
        rows=joint_traction_bounds(node='n',joint_id='j',column_below_id='c',column_length_in=10.,
            left={'id':'l','top_tension_kip':30.,'bottom_tension_kip':20.},
            right={'id':'r','top_tension_kip':40.,'bottom_tension_kip':10.},nominal_vn_kip=100.)
        actions={'c':{'Mi':20.,'Mj':30.},'l':{'P':7.},'r':{'P':9.}}
        evaluate=lambda row:row['constant']+sum(v*actions[m][name] for m,terms in row['terms'].items() for name,v in terms.items())
        inner=[evaluate(r) for r in rows if r['bound']=='inner']
        outer=[evaluate(r) for r in rows if r['bound']=='outer']
        q=[5.+l-r for l in (-20.,7.+30.) for r in (-10.,9.+40.)]
        self.assertEqual(inner,[max(q),-min(q)])
        self.assertEqual(outer,[min(q),-max(q)])

    def test_transport_is_a_verified_lower_bound_for_every_gravity_case(self):
        reference=portal();reference['baseline_member_actions']={m['id']:{'P':0.,'Mi':0.,'Mj':0.} for m in reference['members']}
        for member in reference['members']:
            if member['kind']=='column':
                curve=box(100.,axial=(-1000.,999.));curve['inner_halfplanes']=[[.1,1.,100.],[.1,-1.,100.]];curve['outer_halfplanes']=copy.deepcopy(curve['inner_halfplanes'])
                member['interaction_i']=member['interaction_j']=curve
        target=copy.deepcopy(reference);target['combination_id']='gravity_100_per_column'
        for node in target['nodes']:
            if node['floor']==1:node['load']={'z_kip':-100.}
        for member in target['members']:
            if member['kind']=='column':target['baseline_member_actions'][member['id']]['P']=100.
        optimized=analyze_frame(reference);direct=analyze_frame(target)
        lower=transport_inner(reference,optimized['stories'][0]['senses'][0]['inner'],target,1,1)
        self.assertEqual(lower['status'],'feasible_lower_bound')
        self.assertLessEqual(lower['strength_kip'],direct['stories'][0]['lower_kip']+1e-5)
        self.assertGreater(lower['strength_kip'],35.9)
        self.assertLess(lower['certificate']['maximum_equilibrium_residual'],1e-8)
        self.assertLess(verify_solution(target,lower)['maximum_equilibrium_residual'],1e-8)
        family=analyze_gravity_family([reference,target])
        self.assertEqual(family['transported_inner_certificates'],2)
        self.assertLessEqual(family['stories'][0]['lower_kip'],36.+1e-5)
        self.assertGreaterEqual(family['stories'][0]['upper_kip'],36.)

    def test_saved_full_primal_dual_certificate_replays_and_refuses_tampering(self):
        model=portal(stories=2,beam=20.)
        result=analyze_frame(model)
        for row in result['stories']:
            for sense in row['senses']:
                for bound in ('inner','outer'):
                    report=verify_solution(model,sense[bound])
                    self.assertLess(report['primal_dual_gap'],1e-8)
                    self.assertLess(report['dual_stationarity_residual'],1e-8)
        saved=result['stories'][0]['senses'][0]['outer']
        changes=[lambda r:r['primal_variables'].__setitem__(-1,r['primal_variables'][-1]+1.),
                 lambda r:r.__setitem__('strength_kip',r['strength_kip']+1.),
                 lambda r:r['certificate']['dual_proof']['equations'].__setitem__(0,123.),
                 lambda r:r.__setitem__('model_sha256','different'),
                 lambda r:r['member_end_actions']['b1'].__setitem__('Mi',123.)]
        for change in changes:
            tampered=copy.deepcopy(saved);change(tampered)
            with self.assertRaises(ValueError):verify_solution(model,tampered)

    def test_certificate_checks_exact_span_extremum_not_only_saved_stations(self):
        model=portal(beam=1000.)
        beam=next(m for m in model['members'] if m['kind']=='beam');beam['transverse_load_kip_per_in']=.2
        result=analyze_frame(model)
        saved=result['stories'][0]['senses'][0]['inner']
        self.assertTrue(verify_solution(model,saved)['exact_span_extrema_checked'])

    def test_column_transverse_load_outside_end_controlled_domain_is_refused(self):
        model=portal();model['members'][0]['transverse_load_kip_per_in']=.01
        with self.assertRaisesRegex(ValueError,'column|Column'):analyze_frame(model)

    def test_joint_traction_uncertainty_cannot_be_relabelled_an_exact_strength(self):
        from Design.SMRF_Story_Adapter import joint_traction_bounds
        model=portal();model['joint_constraints']=[]
        beam=next(m for m in model['members'] if m['kind']=='beam')
        for name in ('interaction_i','interaction_j','span_interaction'):beam[name]=box(1000.,axial=(-1e-9,1e-9))
        traction={'id':'b1','top_tension_kip':10.,'bottom_tension_kip':10.}
        for j in (0,1):
            model['joint_constraints']+=joint_traction_bounds(node=f'1/{j}',joint_id=f'j{j}',column_below_id=f'c1/{j}',column_length_in=10.,
                nominal_vn_kip=12.,**({'right':traction} if j==0 else {'left':traction}))
        result=analyze_frame(model)['stories'][0]
        self.assertAlmostEqual(result['lower_kip'],4.,places=6)
        self.assertAlmostEqual(result['upper_kip'],40.,places=6)

    def test_exact_gravity_case_collection_refuses_seismic_averaging(self):
        from types import SimpleNamespace
        from Design.SMRF_Story_Adapter import collect_gravity_cases
        from Design.Perimeter_Systems import CONTINUOUS_MODEL
        actions=[dict(id='gravity',dead=1.4,live=0.,ex=0.,ey=0.,members={'actual':'g'}),
                 dict(id='high_x+',dead=1.5,live=1.,ex=1.3,ey=.39,members={'actual':'seismic'})]
        with self.assertRaisesRegex(ValueError,'No exact solved gravity'):collect_gravity_cases(actions,SimpleNamespace(_cache={}))
        result=collect_gravity_cases(actions,SimpleNamespace(_cache={(CONTINUOUS_MODEL,1.5,1.,'all'):{'members':{'actual':'cached gravity'}}}))
        self.assertEqual(result[1]['members'],{'actual':'cached gravity'})
        self.assertEqual(result[1]['represented_design_combination_ids'],['high_x+'])

    def test_missing_joint_or_state_identity_is_not_a_computed_failure(self):
        model=portal();model['joint_constraints']=[]
        with self.assertRaisesRegex(ValueError,'Joint coverage'):analyze_frame(model)
        model=portal();model.pop('state_identity')
        with self.assertRaisesRegex(ValueError,'identity'):analyze_frame(model)

    def test_interval_crossing_is_preserved(self):
        rows=adjacent_strength_bounds([{'story':1,'status':'bounded','lower_kip':99.,'upper_kip':101.},
                                      {'story':2,'status':'bounded','lower_kip':100.,'upper_kip':100.}])
        self.assertEqual(rows[0]['threshold_crossings'],[1.])
        self.assertFalse(rows[0]['definitely_at_least_1'])
        self.assertFalse(rows[0]['definitely_below_1'])

    def test_weaker_sway_strength_needs_both_lower_but_any_certified_upper(self):
        from Design import SMRF_Story_Strength as module
        original=module._solve
        def one_timeout(model,story,sway,bound,**kwargs):
            if sway==1 and bound=='outer':return {'status':'unresolved','reason':'synthetic time limit'}
            return original(model,story,sway,bound,**kwargs)
        with patch.object(module,'_solve',one_timeout):result=analyze_frame(portal())
        self.assertEqual(result['status'],'bounded')
        self.assertAlmostEqual(result['stories'][0]['lower_kip'],40.)
        self.assertAlmostEqual(result['stories'][0]['upper_kip'],40.)
        self.assertIsNone(result['stories'][0]['senses'][0]['upper_kip'])
        self.assertEqual(result['stories'][0]['upper_bound_sway_inventory'],[-1])

    def test_cached_physical_matrices_equal_fresh_build_for_all_force_pairs(self):
        from Design.SMRF_Story_Strength import _build,_cached_build
        model=portal(stories=3,beam=20.);cache={}
        for bound in ('inner','outer'):
            for story in (1,2,3):
                for sway in (-1,1):
                    direct=_build(model,story,sway,bound,{})
                    cached=_cached_build(model,story,sway,bound,{},cache)
                    for key in ('A_eq','A_ub'):self.assertEqual((direct[key]!=cached[key]).nnz,0)
                    for key in ('b_eq','b_ub','c'):np.testing.assert_array_equal(direct[key],cached[key])
                    self.assertEqual(direct['bounds'],cached['bounds'])
        self.assertEqual(len(cache),2)


class VerticalClassification(unittest.TestCase):
    def evidence(self,lower,upper,full_ev=True):
        import Structure_Parameters as sp
        from Design import Grouped_Design as gd
        strength={'stories':[{'story':s,'by_direction':{a:{'story_strength_kip':lo,'story_strength_lower_kip':lo,'story_strength_upper_kip':hi} for a in ('x','y')}} for s,lo,hi in [(1,lower,upper),(2,100.,100.)]],
                  'unreduced_sds_for_vertical_ground_motion':full_ev,'applicability':{'status':'verified'}}
        drift={'assumptions':{'rho_for_drift_load':1.},'stories':[{'location':f'story:{s}/Q{a}/{a}','elastic_drift_in':3.-s,'drift_ratio':.001*(3.-s)} for s in (1,2) for a in ('x','y')]}
        with patch.object(sp,'NUM_FLOOR',2):
            return gd.vertical_regularity(drift,{'story_forces_kip':[10.,10.],'base_shear_kip':20.,'cs_basic':sp.ASCE_SDS/(sp.ASCE_R/sp.ASCE_IE)},strength,{'floors':[{'floor':s,'seismic_weight_kip':100.} for s in (1,2)]})

    def test_uncertain_one_is_conservatively_classified_with_all_consequences(self):
        from Design import Grouped_Design as gd
        evidence=self.evidence(99.,101.)
        self.assertTrue(evidence['evidence']['strength_complete'])
        self.assertTrue(all(r['conservative_type_4a_for_uncertain_1'] for r in evidence['strength']))
        row=gd.regularity_checks(evidence,'E',strength_model_verified=True)[1]
        self.assertEqual(row['status'],'pass')
        evidence=self.evidence(99.,101.,full_ev=False)
        self.assertEqual(gd.regularity_checks(evidence,'E',strength_model_verified=True)[1]['status'],'not_evaluated')

    def test_uncertainty_at_actual_prohibition_threshold_stays_incomplete(self):
        from Design import Grouped_Design as gd
        for low,high in [(79.,81.),(64.,66.)]:
            evidence=self.evidence(low,high)
            self.assertFalse(evidence['evidence']['strength_complete'])
            self.assertEqual(gd.regularity_checks(evidence,'E',strength_model_verified=True)[1]['status'],'not_evaluated')
        self.assertEqual(gd.regularity_checks(self.evidence(60.,64.),'D',strength_model_verified=True)[1]['status'],'fail')


class AdapterCertificateReadback(unittest.TestCase):
    def test_every_gravity_and_both_senses_survive_json_record_replay(self):
        import json
        from types import SimpleNamespace
        import Structure_Parameters as sp
        from Design import SMRF_Story_Adapter as adapter
        state=SimpleNamespace(identity=lambda:'synthetic-explicit-portal',members=lambda:[SimpleNamespace(member_tag=f'c1/{j}',is_column=True,story_or_floor=1) for j in (0,1)])
        actions=[dict(id=f'load{n}',dead=d,live=0.,ex=0.,ey=0.,live_pattern='all',analysis_succeeded=True,axial_reference='joint_faces',
                      members={f'c1/{j}':{'axial_i_kip':100*d,'axial_j_kip':100*d,'centerline_axial_i_kip':100*d} for j in (0,1)}) for n,d in [(1,1.),(2,.5)]]
        gravity=adapter.collect_gravity_cases(actions,SimpleNamespace(_cache={}))
        def build(state,case,capacity,axis,**kwargs):
            model=portal();model.update(combination_id=case['id'],source_identity=kwargs['source'])
            model['baseline_member_actions']={m['id']:{'P':0.,'Mi':0.,'Mj':0.} for m in model['members']}
            for node in model['nodes']:
                if node['floor']==1:node['load']={'z_kip':-case['dead']*100.}
            for member in model['members']:
                if member['kind']=='column':model['baseline_member_actions'][member['id']]['P']=case['dead']*100.
            model.update(load_factors={k:case[k] for k in ('dead','live','ex','ey','live_pattern')},actual_saved_actions_sha256=adapter._hash(case),
                         orthogonal_column_actions=[],joint_bound_inputs=[],scope=['independent synthetic portal fixture'])
            return model
        with patch.object(sp,'NUM_FLOOR',1),patch.object(adapter,'build_frame',build):
            result=adapter.calculate(state,actions,[a['id'] for a in actions],{},gravity_actions=gravity)
            restored=json.loads(json.dumps(result))
            proof=adapter.verify_saved_result(state,actions,[a['id'] for a in actions],{},gravity_actions=gravity,result=restored)
            self.assertEqual(proof['certificates_checked'],12)
            self.assertTrue(adapter.computed_verification_valid(restored))
            for mutation in ('case','family','ratio','gravity','signed','source'):
                changed=copy.deepcopy(restored)
                if mutation=='case':changed['cases'].pop()
                elif mutation=='family':changed['stories'][0]['by_direction']['x']['story_strength_lower_kip']+=1.
                elif mutation=='ratio':changed['adjacent_ratio_bounds']['x'].append({'invented':True})
                elif mutation=='gravity':changed['gravity_actions_sha256']='bad'
                elif mutation=='signed':changed['signed_actions_sha256']='bad'
                else:changed['source_identity']['sha256']='bad'
                with self.assertRaises(ValueError):adapter.verify_saved_result(state,actions,[a['id'] for a in actions],{},gravity_actions=gravity,result=changed)


class InteractionBounds(unittest.TestCase):
    def test_layered_t_beam_nominal_curve_has_certified_bounds_both_signs(self):
        from Design.SMRF_Uniaxial_Section import LayeredSection
        section=LayeredSection(24.,[(0.,6.,70.,5.),(6.,24.,20.,5.)],
            [(2.,2.37,.79,60.),(22.,3.16,.79,60.),(1.5,1.2,.2,60.),(4.5,1.2,.2,60.)])
        bounds=layered_interaction(section,divisions=128)
        for face,sign in [('top',1.),('bottom',-1.)]:
            state=section.state(np.geomspace(.01,10000.,2001),face,phi_mode='nominal')
            p=state['pn'];m=sign*state['mn'];lo,hi=bounds['axial_domain_kip'];keep=(p>=lo)&(p<=hi);p=p[keep];m=m[keep]
            for name,below in [('inner_halfplanes',True),('outer_halfplanes',False)]:
                limit=np.min([c-a*p for a,b,c in bounds[name] if b==sign],axis=0)
                self.assertTrue((limit<=m+1e-6).all() if below else (limit>=m-1e-6).all())
    def test_nonconcave_lipschitz_curve_does_not_use_unsafe_convex_hull(self):
        p=np.linspace(-1.,1.,17);f=1.+.1*np.cos(5*p)
        bound=lipschitz_interaction(p,f,f,.5)
        dense=np.linspace(-1.,1.,2001);actual=1.+.1*np.cos(5*dense)
        for key,relation in [('inner_halfplanes','below'),('outer_halfplanes','above')]:
            planes=[(a,b,c) for a,b,c in bound[key] if b>0]
            limit=np.min([c-a*dense for a,b,c in planes],axis=0)
            self.assertTrue((limit<=actual+1e-12).all() if relation=='below' else (limit>=actual-1e-12).all())
        self.assertGreater(bound['maximum_moment_gap_kip_in'],.1)

    def test_finite_bar_nominal_curve_and_continuous_material_bound(self):
        bars=[(x,y,.79) for x in (-8.,0.,8.) for y in (-8.,8.)]
        section=BiaxialSection(22.,22.,5.,60.,bars)
        coarse=rectangular_interaction(section,divisions=32)
        fine=rectangular_interaction(section,divisions=128)
        self.assertLess(fine['maximum_moment_gap_kip_in'],coarse['maximum_moment_gap_kip_in'])
        # Independent c-grid states must lie between both envelopes wherever
        # their nominal axial force is inside the declared tied-column domain.
        c=np.geomspace(.01,10000.,5001);state=section.state(c,math.pi/2)
        p=state['p']/state['phi'];m=state['mh']/state['phi']
        keep=(p>=fine['axial_domain_kip'][0])&(p<=fine['axial_domain_kip'][1]);p=p[keep];m=m[keep]
        np.testing.assert_array_less(np.abs(np.diff(m)),11*np.diff(p)+1e-8)
        for name,below in [('inner_halfplanes',True),('outer_halfplanes',False)]:
            lim=np.min([c-a*p for a,b,c in fine[name] if b>0],axis=0)
            self.assertTrue((lim<=m+1e-6).all() if below else (lim>=m-1e-6).all())

    def test_overlapping_finite_bars_cannot_claim_the_proof(self):
        section=BiaxialSection(10.,10.,5.,60.,[(-.1,-.1,1.),(-.1,.1,1.),(.1,-.1,1.),(.1,.1,1.)])
        with self.assertRaisesRegex(ValueError,'Overlapping'):rectangular_interaction(section)


if __name__=='__main__':unittest.main()
