"""Synthetic stream scope; real threshold/hoop kernel and independent readback."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Beam_Torsion import threshold
from Design.SMRF_Common_Torsion_Stream import assess_torsion_stream,verify_torsion_stream,assess_record
from tests.test_beam_torsion_physical import D,SECTION,POINTS,DEV


class TorsionStream(unittest.TestCase):
    def fixture(self,stack):
        state=SimpleNamespace(identity=lambda:'synthetic_state',designs={'beam':D},groups={'beam':dict(member_tags=[1])},
            member=lambda tag:SimpleNamespace(design=D,group_id='beam'))
        common=dict(input_signature='synthetic_input',state_identity=state.identity(),inventory=dict(identity_sha256='synthetic_mat'))
        context=SimpleNamespace(validate=lambda c:context)
        rows=[dict(member_tag=1,case_id='G',cut_id=name,state_identity=state.identity(),recovery_id='explicit_fixture',station_in=x)
              for name,x in (('left',0.),('right',100.))]
        raw=dict(source_id='explicit_controller_fixture',records=rows,sections={'1':SECTION},cut_inventory_complete=True,slab_fc_ksi=5.,selected_member_tags=[1],
            analysis_cases=[dict(analysis_case_id='G',branch='actual_gravity')],guard_polynomial_audits=[],
            polynomial_audits=[dict(member_tag=1,case_id='G',interval_in=[0.,100.],
                coefficients=[[0.,20.,0.,0.,0.,0.],[0.]*6,[0.]*6])])
        state.raw_fixture=raw
        raw['payload_sha256']=_hash(raw)
        ident={k:rows[0][k] for k in ('case_id','cut_id','state_identity','recovery_id')}
        plan=dict(record_identity=ident,section_id=SECTION['section_id'],all_demands_below_threshold=True,
            demands=[dict(N_tension_kip=0.,T_about_positive_axis_kipin=20.,threshold=threshold(SECTION,20.,0.))])
        plan['payload_sha256']=_hash(plan)
        stack.enter_context(patch('Design.SMRF_Common_Torsion_Stream.recover_common_torsion',return_value=raw))
        def planned(state,common,record,*args,**kwargs):
            value=copy.deepcopy(plan)
            value['record_identity']={k:record[k] for k in ('case_id','cut_id','state_identity','recovery_id')}
            value['payload_sha256']=_hash({k:v for k,v in value.items() if k!='payload_sha256'})
            return value
        stack.enter_context(patch('Design.SMRF_Common_Torsion_Stream.plan_record',side_effect=planned))
        limits=dict(strength_wall_seconds=30,max_required_cuts=100,max_strength_evaluations=100)
        common['limits']=limits
        return state,common,context,limits,plan

    def test_complete_threshold_inventory_and_replay(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            state,common,context,limits,plan=self.fixture(stack)
            saved=assess_torsion_stream(state,common,limits=limits,artifact_directory=directory,prepared_sources=context)
            self.assertEqual(saved['status'],'pass');self.assertEqual(saved['material_cut_count'],0)
            replay=verify_torsion_stream(saved,common,state=state,prepared_sources=context)
            self.assertEqual(replay['status'],'pass');self.assertEqual(replay['inverse_solves'],0)
            corrupt=copy.deepcopy(saved);corrupt['chunks']=[];corrupt['payload_sha256']=_hash({k:v for k,v in corrupt.items() if k!='payload_sha256'})
            with self.assertRaisesRegex(ValueError,'complete/group'):verify_torsion_stream(corrupt,common,state=state,prepared_sources=context)

    def test_zero_wall_budget_retains_unknown(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            state,common,context,limits,plan=self.fixture(stack);limits['strength_wall_seconds']=0
            saved=assess_torsion_stream(state,common,limits=limits,artifact_directory=directory,prepared_sources=context)
            self.assertEqual(saved['status'],'not_evaluated');self.assertFalse(saved['complete_required_inventory'])
            self.assertEqual(verify_torsion_stream(saved,common,state=state,prepared_sources=context)['status'],'not_evaluated')

    def test_passing_samples_without_continuous_reserve_cannot_pass_stream(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            state,common,context,limits,plan=self.fixture(stack)
            # Explicit controller counterexample: sampled material checks pass,
            # but the actual interval exceeds its exemption at an interior point.
            c=threshold(SECTION,0.,0.)['phi_Tth_kip_in']
            state.raw_fixture['polynomial_audits'][0]['coefficients']=[[0.,.93*c,0,0,0,0],[0.,.6*c,0,0,0,0],[0.,-c,0,0,0,0]]
            state.raw_fixture['payload_sha256']=_hash({k:v for k,v in state.raw_fixture.items() if k!='payload_sha256'})
            saved=assess_torsion_stream(state,common,limits=limits,artifact_directory=directory,prepared_sources=context)
            self.assertEqual(saved['status'],'not_evaluated')
            self.assertEqual(verify_torsion_stream(saved,common,state=state,prepared_sources=context)['status'],'not_evaluated')

    def test_passing_threshold_flag_with_tensile_guard_is_refused(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            state,common,context,limits,plan=self.fixture(stack)
            plan['demands'][0]['N_tension_kip']=1000.;plan['payload_sha256']=_hash({k:v for k,v in plan.items() if k!='payload_sha256'})
            with self.assertRaisesRegex(ValueError,'threshold claim'):assess_torsion_stream(state,common,limits=limits,artifact_directory=directory,prepared_sources=context)

    def test_first_proved_failure_leaves_unvisited_member_explicitly_pending(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            state,common,context,limits,plan=self.fixture(stack)
            state.groups['beam']['member_tags']=[1,2]
            assessed=stack.enter_context(patch('Design.SMRF_Common_Torsion_Stream.assess_record',return_value=dict(status='fail',basis='explicit_controller_failure')))
            saved=assess_torsion_stream(state,common,limits=limits,artifact_directory=directory,prepared_sources=context)
            self.assertEqual(assessed.call_count,1);self.assertEqual(len(saved['chunks']),1)
            self.assertEqual(saved['status'],'fail');self.assertFalse(saved['complete_required_inventory'])
            self.assertEqual(saved['groups']['beam']['pending_member_tags'],[2])
            replay=verify_torsion_stream(saved,common,state=state,prepared_sources=context)
            self.assertEqual(replay['status'],'fail');self.assertEqual(replay['enumerated_members'],1)
            self.assertEqual(replay['required_members'],2)

    def test_saved_roster_and_resource_allocation_cannot_change(self):
        with ExitStack() as stack,tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            state,common,context,limits,plan=self.fixture(stack)
            saved=assess_torsion_stream(state,common,limits=limits,artifact_directory=directory,prepared_sources=context)
            for key,value in [('member_inventory',[2]),('limits',dict(limits,max_required_cuts=101))]:
                corrupt=copy.deepcopy(saved);corrupt[key]=value;corrupt['payload_sha256']=_hash({k:v for k,v in corrupt.items() if k!='payload_sha256'})
                with self.assertRaisesRegex(ValueError,'inventory changed'):verify_torsion_stream(corrupt,common,state=state,prepared_sources=context)


class CompleteReinforcementBridge(unittest.TestCase):
    def test_real_compatible_beam_state_reserved_with_actual_hoop_kernel(self):
        from tests.test_physical_floor_strength import PhysicalStrengthTests
        from tests.test_physical_floor_materials import bar
        from Design.SMRF_Physical_Floor_Strength import assess_physical_cut_family
        from Design.SMRF_Primary_Moment_Guards import physical_cut_states
        from tests.test_common_floor_actions import gravity
        ident='explicit_unloaded_beam';coordinates=[{**p,'id':'member_1/beam/'+p['id']} for p in POINTS]
        bars=[dict(bar(p['id'],s=p['x_in'],z=p['y_in']-4.,area=.79,run=(0,100),owner='web_line'),member_id='1') for p in coordinates]
        bars +=[bar('y'+str(i),'y',s=x,z=3.,area=.31,run=(-9,9)) for i,x in enumerate((46.,54.))]
        inv=dict(state_identity=ident,identity_sha256='explicit_physical_steel_fixture',slab_thickness_in=8.,physical_detail_status='pass',
            floors={'1':dict(bars=bars)},beam_geometry={'1':dict(axis='x',floor=1,line_in=0.,b_in=18.,h_in=16.,fc_ksi=5.,ec_ksi=4000.)})
        specs=[];cuts={};dev=[]
        for name in ('zero_physical','zero_primary_guard'):
            spec=dict(cut_id=name,floor=1,axis='x',station_in=50.,interval_in=[-9.,9.],axial_accounting_interval_in=[40.,60.],member_id='1')
            raw=gravity((-9,9),name,False);raw['identity'].update(state_identity=ident,combination_id='synthetic_C06')
            raw['slab']['station_in']=50.;raw['slab']['integral']=[0.]*8;raw['slab']['transverse_first_moment']=[0.]*8
            raw['slab']['pieces']=[dict(interval_in=[-9,9],integral=[0.]*8,transverse_first_moment=[0.]*8)]
            raw['resultants']=[0.]*6;raw['Ms_transverse_first_moment_kipin']=0.
            raw.update(material_inventory_id=inv['identity_sha256'],analysis_input_identity='synthetic_zero_load')
            states,_=physical_cut_states([raw],outer_interval_in=[-9,9]);cuts[name]=states[0];specs.append(spec)
            from Design.SMRF_Axial_Action_Accounting import make_window_evidence
            cuts[name]['source_response_sha256']='analytical_zero_load'
            cuts[name]['axial_action_window']=make_window_evidence(cuts[name],[40,60],[dict(interval_in=[40,60],
                coefficients=[[0.]*7]*3,sample_values=[[0.]*7]*3,quarter_witness=[0.]*7)],source_response_sha256='analytical_zero_load')
            dev.append(dict(evidence=[dict(status='pass',evidence_id='explicit_anchored_fixture',station_interval_in=[50.,50.],
                state_identity=ident,material_inventory_id=inv['identity_sha256'],floor=1,axis='x',interval_in=[-9.,9.],
                axial_accounting_interval_in=[40.,60.],physical_bar_ids=[b['id']],full_fy_only=True,
                full_fy_development_domain=dict(status='computed',interval_in=list(b['straight_run_interval_in']),
                    physical_straight_run_interval_in=list(b['straight_run_interval_in']),
                    basis='full_fy_both_signs_both_sides_no_25_4_10_reduction')) for b in bars]))
        material=assess_physical_cut_family(None,inv,specs,cuts,slab_fc_ksi=5.,ec_ksi=4000.,phi_requirements=dict(NM=.65,diaphragm_shear=.6),
            required_cut_ids=[s['cut_id'] for s in specs],required_case_ids=['synthetic_C06'],assessment_identity='two_explicit_zero_states',development_result=dict(cuts=dev))
        self.assertEqual(material['status'],'pass')
        record=dict(member_tag=1,case_id='synthetic_C06',cut_id='middle',state_identity=ident,recovery_id='independent_algebra',station_in=50.)
        plan=dict(record_identity={k:record[k] for k in ('case_id','cut_id','state_identity','recovery_id')},section_id=SECTION['section_id'],
            all_demands_below_threshold=False,material_plan=dict(required_cut_ids=[s['cut_id'] for s in specs],cut_specs=specs,cuts=cuts),
            demands=[dict(guard_cut_id='zero_primary_guard',kind='explicit_test_torsion',N_tension_kip=0.,T_about_positive_axis_kipin=100.,Q_vertical_kip=0.,V_inplane_kip=0.)])
        plan['payload_sha256']=_hash(plan)
        detail=dict(coordinates=coordinates,development=dict(DEV,longitudinal_bar_ids=[p['id'] for p in coordinates],support_faces_in=[0.,100.],clear_span_in=100.),
            splice_detail=dict(status='pass',hoop_stations_in=[47.,50.,53.],transverse_detail=dict(bar_size=5,legs=3,spacing_in=3.,zone_in=[46.,54.])))
        capacity=dict(members=[dict(member_tag=1,governing_ve_kip=10.)],vc_kip=0.,effective_depth_in=13.)
        state=SimpleNamespace(member=lambda t:SimpleNamespace(design=D))
        result=assess_record(state,record,SECTION,plan,detail,capacity,material)
        self.assertEqual(result['status'],'pass',result)
        self.assertEqual(len(result['longitudinal_reserve']['source_guard_cut_ids']),2)
        self.assertTrue(all(abs(b['nominal_stress_ksi'])<1e-8 for b in result['longitudinal_reserve']['bars']))
        # Full capacity-design shear still takes precedence over zero Vu.
        capacity['members'][0]['governing_ve_kip']=200.
        self.assertEqual(assess_record(state,record,SECTION,plan,detail,capacity,material)['status'],'fail')
        detail['splice_detail']['status']='not_evaluated'
        self.assertEqual(assess_record(state,record,SECTION,plan,detail,capacity,material)['status'],'not_evaluated')
        detail['splice_detail'].update(status='fail',checks=[dict(rule='actual_device_collision',status='fail',passes=False)])
        failed=assess_record(state,record,SECTION,plan,detail,capacity,material)
        self.assertEqual(failed['status'],'fail');self.assertEqual(failed['physical_detail_failure_checks'][0]['rule'],'actual_device_collision')

if __name__=='__main__':unittest.main()
