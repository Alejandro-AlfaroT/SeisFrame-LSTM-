import copy
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_common_floor_actions import gravity,projector
from tests.test_physical_floor_materials import bar
from Design.SMRF_Primary_Moment_Guards import primary_moment_guards,primary_membrane_guards,physical_cut_states
from Design.SMRF_Physical_Floor_Strength import assess_physical_cut_family,replay_physical_cut_assessment,verify_physical_cut_assessment,plan_physical_cut_family
from Design.SMRF_Physical_Floor_Materials import spatial_section_from_inventory
from Design.SMRF_Physical_Slab_Selection import select_physical_layout,catalog_proposals
from Design.SMRF_Common_Floor_Actions import _hash


class PrimaryGuardTests(unittest.TestCase):
    def setup(self):
        p=projector();cuts=[gravity((-30,-5),'left',False),gravity(),gravity((15,50),'right',False)]
        g=gravity((-30,50),'effective',True);g['identity']['combination_id']='Gexact'
        inc=dict(source_operation='released_loaded_minus_identical_released_gravity',state_identity='actual_state',
            gravity_load_id='Gexact',physical_combination_id=cuts[0]['identity']['combination_id'],axis='x',station_in=100.,
            n_tension_kip=0.,m_z_kip_in=-300.,m_s_kip_in=55.,reference_z_in=0.,reference_s_in=5.)
        return p,cuts,g,inc

    def test_signed_guards_preserve_physical_N_other_tensors_and_weak_action(self):
        p,cuts,g,inc=self.setup()
        out=primary_moment_guards(cuts,g,projector=p,primary_increment=inc,guard_identity='guard',components=('strong','weak','both'))
        self.assertEqual(len(out['branches']),4)
        def total(states):
            return sum((np.array([r['resultants'][0],r['resultants'][3],r['resultants'][6]+(r['reference_s_in']-5)*r['resultants'][0]]) for r in states),np.zeros(3))
        original=total(out['branches'][0]['states'])
        for branch in out['branches'][1:]:
            got=total(branch['states']);proof=branch['proof'];target=np.array(proof['target_normal'])
            self.assertAlmostEqual(got[0],original[0],9)
            if branch['branch_id'].endswith('strong'):
                self.assertAlmostEqual(got[1],target[1],9);self.assertAlmostEqual(got[2],original[2],9)
            elif branch['branch_id'].endswith('weak'):
                self.assertAlmostEqual(got[1],original[1],9);self.assertAlmostEqual(got[2],target[2],9)
            else:np.testing.assert_allclose(got[1:],target[1:],atol=1e-9)
            for originalrow,guard in zip(out['branches'][0]['states'],branch['states']):
                np.testing.assert_allclose(np.array(guard['resultants'])[[1,2,4,5]],np.array(originalrow['resultants'])[[1,2,4,5]])
                self.assertFalse(guard['accounting']['physical_equilibrium_solution_claimed'])
        self.assertFalse(out['force_bound_claimed'])

    def test_missing_gravity_case_and_wrong_width_refused(self):
        p,cuts,g,inc=self.setup();bad=dict(inc,gravity_load_id='wrong')
        with self.assertRaises(ValueError):primary_moment_guards(cuts,g,projector=p,primary_increment=bad,guard_identity='bad')
        g['slab']['interval_in']=[-29,50]
        with self.assertRaises(ValueError):primary_moment_guards(cuts,g,projector=p,primary_increment=inc,guard_identity='bad')

    def test_bar_clear_enclosing_partition_keeps_exact_flange_target_and_zero_extension(self):
        p,cuts,g,inc=self.setup();cuts=[gravity((-40,-5),'left',False),gravity(),gravity((15,60),'right',False)]
        exact=gravity((-30,50),'effective',True)
        with self.assertRaises(ValueError):primary_moment_guards(cuts,g,projector=p,primary_increment=inc,guard_identity='missing',outer_interval_in=[-40,60])
        out=primary_moment_guards(cuts,g,projector=p,primary_increment=inc,guard_identity='enclosing',outer_interval_in=[-40,60],physical_effective_cut=exact)
        original=out['branches'][0]['states'];strong=out['branches'][1]['states']
        self.assertAlmostEqual(sum(r['resultants'][3] for r in strong)-sum(r['resultants'][3] for r in original),-300,10)
        self.assertAlmostEqual(sum(r['resultants'][0] for r in strong),sum(r['resultants'][0] for r in original),10)

    def test_complete_prescribed_primary_membrane_and_simultaneous_moment(self):
        p,cuts,g,inc=self.setup();gravitycuts=copy.deepcopy(cuts);sources=[]
        for cut,baseline in zip(cuts,gravitycuts):
            baseline['identity']['combination_id']='Gexact';a,b=cut['slab']['interval_in'];w=b-a
            source=dict(identity=copy.deepcopy(cut['identity']),axis='x',station_in=100.,interval_in=[a,b],
                reference_z_in=0.,source_evidence_id='actual_primary_floor_freebody',branch_id='collector',gravity_case_id='Gexact',
                floor_freebody_residual=[0.]*6,floor_freebody_tolerance=[1e-9]*6,
                slab_membrane_integral_kip=[2*w,-w,.3*w],
                slab_membrane_global_first_moment_kipin=[b*b-a*a,-(b*b-a*a)/2,.15*(b*b-a*a)])
            if cut.get('web'):source['web']=dict(member_id='7',N_tension_kip=30.,transverse_reference_in=5.)
            sources.append(source)
        out=primary_membrane_guards(cuts,gravitycuts,sources,outer_interval_in=p.flange,guard_identity='P5',projector=p,primary_increment=inc)
        self.assertEqual(len(out['branches']),4)
        def total(states):return sum((np.array([r['resultants'][0],r['resultants'][3],r['resultants'][6]+(r['reference_s_in']-5)*r['resultants'][0]]) for r in states),np.zeros(3))
        base=total(out['branches'][0]['states']);mem=total(out['branches'][1]['states']);strong=total(out['branches'][2]['states']);both=total(out['branches'][3]['states'])
        self.assertAlmostEqual(mem[0],base[0]+190,10)
        self.assertAlmostEqual(strong[0],mem[0],10);self.assertAlmostEqual(both[0],mem[0],10)
        self.assertAlmostEqual(strong[1],base[1]-300,10)
        self.assertAlmostEqual(strong[2],mem[2],10);self.assertAlmostEqual(both[2],mem[2]+55,10)
        sources[0]['floor_freebody_residual'][0]=1
        with self.assertRaises(ValueError):primary_membrane_guards(cuts,gravitycuts,sources,outer_interval_in=p.flange,guard_identity='bad')


class PhysicalStrengthTests(unittest.TestCase):
    def fixture(self):
        bars=[bar('xtop',s=1,z=2.8,area=.31,run=(0,10)),bar('xbottom',s=2,z=-2.8,area=.31,run=(0,10)),
              bar('ytop1','y',s=4,z=2,area=.31,run=(0,8)),bar('ytop2','y',s=6,z=2,area=.31,run=(0,8))]
        inv=dict(state_identity='actual_state',identity_sha256='inventory',slab_thickness_in=8.,physical_detail_status='pass',
                 floors={'1':{'bars':bars}})
        spec=dict(cut_id='c',floor=1,axis='x',station_in=5.,interval_in=[0,4],axial_accounting_interval_in=[3,7])
        raw=gravity((0,4),'c',False);raw['slab']['station_in']=5.
        raw['slab']['integral']=[0.]*8;raw['slab']['transverse_first_moment']=[0.]*8
        raw['slab']['pieces']=[dict(interval_in=[0,4],integral=[0.]*8,transverse_first_moment=[0.]*8)]
        raw['resultants']=[0.]*6;raw['Ms_transverse_first_moment_kipin']=0.
        raw.update(material_inventory_id='inventory',analysis_input_identity='actual_fixture_input')
        states,_=physical_cut_states([raw],outer_interval_in=[0,4]);action=states[0]
        from Design.SMRF_Axial_Action_Accounting import make_window_evidence
        action['source_response_sha256']='analytical_uniform_zero_response'
        action['axial_action_window']=make_window_evidence(action,[3,7],[dict(interval_in=[3,7],
            coefficients=[[0.]*7]*3,sample_values=[[0.]*7]*3,quarter_witness=[0.]*7)],
            source_response_sha256=action['source_response_sha256'])
        evidence=[dict(status='pass',evidence_id='actual_provider',station_interval_in=[5,5],state_identity='actual_state',
            material_inventory_id='inventory',floor=1,axis='x',interval_in=[0,4],axial_accounting_interval_in=[3,7],
            physical_bar_ids=[b['id'] for b in bars],full_fy_only=True)]
        evidence=[dict(evidence[0],physical_bar_ids=[b['id']],full_fy_development_domain=dict(status='computed',
            interval_in=list(b['straight_run_interval_in']),physical_straight_run_interval_in=list(b['straight_run_interval_in']),
            basis='full_fy_both_signs_both_sides_no_25_4_10_reduction')) for b in bars]
        dev=dict(cuts=[dict(evidence=evidence)])
        return inv,spec,action,dev

    def test_atomic_zero_state_replay_exact_identity_and_tamper(self):
        inv,spec,action,dev=self.fixture()
        out=assess_physical_cut_family(None,inv,[spec],{'c':action},slab_fc_ksi=5,ec_ksi=4000,phi_requirements={'NM':.65},
            required_cut_ids=['c'],required_case_ids=[action['identity']['combination_id']],assessment_identity='actual',development_result=dev)
        self.assertEqual(out['status'],'pass',out['records'])
        replay=replay_physical_cut_assessment(out);self.assertEqual(replay['payload_sha256'],out['payload_sha256'])
        with patch('Design.SMRF_Spatial_Floor_Section.SpatialFloorSection.solve',side_effect=AssertionError('No inverse on replay')),patch(
                'Design.SMRF_Spatial_Floor_Section.SpatialFloorSection.capacity_separation_certificate',side_effect=AssertionError('No dual search on replay')):
            verified=verify_physical_cut_assessment(out)
        self.assertEqual(verified['status'],'pass');self.assertTrue(verified['saved_claims_verified'])
        self.assertEqual(verified['inverse_solves'],0);self.assertEqual(verified['dual_searches'],0)
        bad=copy.deepcopy(out);bad['replay_payload']['cuts']['c']['resultants'][0]=1
        with self.assertRaises(ValueError):replay_physical_cut_assessment(bad)
        self.assertFalse(out['complete_design_qualified'])

    def test_development_domains_preserve_all_void_but_only_developed_steel(self):
        inv,spec,action,dev=self.fixture();e=dev['cuts'][0]['evidence'][0];evidence=[]
        for b in inv['floors']['1']['bars']:
            domain=[2,3] if b['axis']=='y' else [0,4]
            evidence.append(dict(e,physical_bar_ids=[b['id']],status='fail',full_fy_development_domain=dict(status='computed',
                interval_in=domain,physical_straight_run_interval_in=b['straight_run_interval_in'],
                basis='full_fy_both_signs_both_sides_no_25_4_10_reduction')))
        args={k:spec[k] for k in ('floor','axis','station_in','interval_in','axial_accounting_interval_in')}
        plain=spatial_section_from_inventory(inv,**args,fc_ksi=5,ec_ksi=4000,phi_requirements={'NM':.65})
        limited=spatial_section_from_inventory(inv,**args,fc_ksi=5,ec_ksi=4000,phi_requirements={'NM':.65},
            development_evidence=evidence,apply_development_domains=True)
        self.assertEqual(limited['development']['status'],'pass')
        self.assertEqual(limited['development']['zero_strength_credit_ids'],['xbottom','xtop'])
        q=np.array([-.0001,-.00008,0,0,0,0,0]);a=plain['section'].state(q);b=limited['section'].state(q)
        np.testing.assert_allclose(a['concrete_nominal_resultants'],b['concrete_nominal_resultants'],atol=1e-12)
        self.assertEqual(b['steel_nominal_resultants'][0],0.)
        self.assertAlmostEqual(b['steel_nominal_resultants'][1],a['steel_nominal_resultants'][1]/4,12)
        self.assertTrue(limited['development']['installed_concrete_voids_preserved'])

    def test_selection_requires_exact_assessment_and_never_mutates_candidate(self):
        inv,spec,action,dev=self.fixture()
        assessed=assess_physical_cut_family(None,inv,[spec],{'c':action},slab_fc_ksi=5,ec_ksi=4000,phi_requirements={'NM':.65},
            required_cut_ids=['c'],required_case_ids=[action['identity']['combination_id']],assessment_identity='actual',development_result=dev)
        candidates=[dict(candidate_id='A',layout={'bar':4}),dict(candidate_id='B',layout={'bar':5})]
        original=copy.deepcopy(candidates)
        def callback(c,identity):
            return dict(installed_layout=c['layout'],layout_sha256=_hash(c['layout']),state_identity='actual_state',
                assessment=assessed,qualification={'status':'pass','evidence_id':'computed_witness'},quantities={'steel_volume_in3':10 if c['candidate_id']=='A' else 20})
        result=select_physical_layout(candidates,callback,selection_identity='catalog',maximum_candidates=2,
            required_cut_ids=['c'],required_case_ids=[action['identity']['combination_id']])
        self.assertEqual(result['selected_candidate_id'],'A');self.assertEqual(candidates,original)
        def stale(c,identity):
            out=callback(c,identity);out['state_identity']='other';return out
        result=select_physical_layout(candidates,stale,selection_identity='catalog',maximum_candidates=2,
            required_cut_ids=['c'],required_case_ids=[action['identity']['combination_id']])
        self.assertEqual(result['status'],'unknown')

    def test_actual_mesh_planner_computes_complete_two_axis_trace_inventory(self):
        inv,_,_,_=self.fixture();fp=dict(x_min_in=0.,x_max_in=4.,y_min_in=0.,y_max_in=4.)
        inv['floors']['1']['bars'][2]['s_in']=1.5;inv['floors']['1']['bars'][3]['s_in']=2.5
        desc=dict(level=1,footprint=fp,supports=[],beams=[])
        definition=dict(state_identity='actual_state',identity_sha256='definition',floors=[desc],slab_thickness_in=8.,gamma_kip_in3=.0001)
        row=dict(id='cell',kind='slab',positions_in=[[0,0,0],[4,0,0],[4,4,0],[0,4,0]],
                 raw_resultants=[[1.,2.,.1,3.,4.,.2,.3,.4]]*4)
        response=dict(definition_sha256='definition',cases=[dict(id='G',dead=1.,live=1.)],
            floors=[dict(level=1,grid={'x_in':[0,4],'y_in':[0,4]},responses=[dict(case_id='G',elements=[row],force_partition_relative_error=0.)])])
        bundle=dict(source_id='actual',response=response,material_inventory_id='inventory',response_payload_sha256=_hash(response),analysis_input_identity='actual_fixture')
        result=plan_physical_cut_family(definition,inv,[bundle],planner_identity='full',maximum_cuts=6)
        self.assertEqual(result['required_cut_count'],6)
        self.assertEqual({s['axis'] for s in result['cut_specs']},{'x','y'})
        self.assertEqual({s['station_in'] for s in result['cut_specs']},{0.,2.,4.})
        self.assertTrue(result['all_floor_trace_partitions_verified'])
        self.assertFalse(result['mesh_qualification_inferred'])
        other=copy.deepcopy(bundle);other['source_id']='other'
        complete=plan_physical_cut_family(definition,inv,[bundle,other],planner_identity='both',maximum_cuts=12)
        first=plan_physical_cut_family(definition,inv,[bundle,other],planner_identity='first',maximum_cuts=6,
            selected_floor_case_pairs=[['actual',1,'G']])
        second=plan_physical_cut_family(definition,inv,[bundle,other],planner_identity='second',maximum_cuts=6,
            selected_floor_case_pairs=[['other',1,'G']])
        self.assertFalse(first['scope']['complete_required_universe'])
        self.assertEqual(first['scope']['full_universe_sha256'],complete['scope']['full_universe_sha256'])
        self.assertEqual({**first['cuts'],**second['cuts']},complete['cuts'])
        self.assertEqual(first['cut_specs']+second['cut_specs'],complete['cut_specs'])
        with self.assertRaisesRegex(ValueError,'subset'):plan_physical_cut_family(definition,inv,[bundle],planner_identity='bad',selected_floor_case_pairs=[['actual',2,'G']])
        with self.assertRaises(ValueError):plan_physical_cut_family(definition,inv,[bundle],planner_identity='budget',maximum_cuts=5)
        rectangular=copy.deepcopy(definition);rectangular['floors'][0]['supports']=[dict(id='rect',x=2.,y=2.,h_in=2.,b_in=1.)]
        result=plan_physical_cut_family(rectangular,inv,[bundle],planner_identity='rectangle',maximum_cuts=100)
        joint=[r for r in result['assigned_joint_or_other_axis_regions'] if r['kind']=='local_joint']
        x=[r for r in joint if r['axis']=='x'];y=[r for r in joint if r['axis']=='y']
        self.assertEqual({tuple(r['axial_interval_in']) for r in x},{(1.,3.)})
        self.assertEqual({tuple(r['interval_in']) for r in x},{(1.5,2.5)})
        self.assertEqual({tuple(r['axial_interval_in']) for r in y},{(1.5,2.5)})
        self.assertEqual({tuple(r['interval_in']) for r in y},{(1.,3.)})

    def test_catalog_preserves_four_independent_spacing_axes(self):
        inputs=dict(thickness_in=8.,fc_ksi=5.,fy_ksi=60.,max_aggregate_size_in=.75,exposure='sheltered_interior',
            steel_specification='ASTM A706',concrete_type='normalweight',panel_ids=['p'],num_floor=1)
        out=catalog_proposals(inputs,dict(bar_sizes=[4,5,6],spacing_options_in=[6.,8.]),maximum_candidates=48)
        self.assertEqual(out['enumerated_count'],48)
        self.assertEqual(len(out['candidates'])+len(out['rejected']),48)
        self.assertTrue(any(len({r['spacing_in'] for r in c['layout']['layers'].values()})==2 for c in out['candidates']))
        self.assertFalse(out['capacity_or_zero_N_assertion_used'])
        with self.assertRaises(ValueError):catalog_proposals(inputs,maximum_candidates=10)


if __name__=='__main__':unittest.main()
