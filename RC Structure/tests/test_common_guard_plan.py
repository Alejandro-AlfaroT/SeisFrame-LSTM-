"""Explicit analytical cut fixtures exercise full production guard assembly.

These fields test the design-force adapter, not 3D structural qualification.
The prescribed P5 fields are independently equilibrated small Q4 fixtures.
"""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Primary_Diaphragm_Batch import analyze_primary_diaphragms
from Design.SMRF_Common_Guard_Plan import build_common_guard_plan,primary_guard_increment,_PhysicalSource
from Design.SMRF_Common_Action_Analysis import physical_response_bundles
from Design.SMRF_Primary_Moment_Guards import physical_cut_states
from Design.SMRF_Full_T_Projection import FullTIncrementProjector
from Model.Floor_Footprint import from_perimeter_beams
from tests.test_primary_diaphragm_batch import common_fixture,settings
from tests.test_physical_floor_cuts import shell,beam


def fixture():
    common=common_fixture();d=common['definition'];stateid=d['state_identity'];member='1:x:0:0'
    # The complete physical footprint is supplied by the installed adapter in
    # production; this old pure geometry fixture derives it independently.
    for floor in d['floors']:floor['footprint']=from_perimeter_beams(d['geometry'],floor['beams'])
    d['identity_sha256']=_hash({k:v for k,v in d.items() if k!='identity_sha256'});common['definition_sha256']=d['identity_sha256']
    for record in common['p5_influences'].values():record['physical_definition_sha256']=d['identity_sha256']
    inventory=dict(state_identity=stateid,identity_sha256='actual_analytical_inventory',floors={str(k):dict(bars=[]) for k in (1,2)})
    inventory['floors']['1']['bars']=[dict(axis='x',s_in=5.,db_in=.5)]
    common['inventory']=inventory
    gcases=[dict(id='high',dead=1.2,live=1.6),dict(id='low',dead=.9,live=0.)]
    def response(cases,physical):
        floors=[]
        for desc in d['floors']:
            k=desc['level'];fp=desc['footprint'];results=[]
            for case in cases:
                coefficients=np.zeros((4,8));coefficients[0]=np.array([1.,2.,.1,3.,4.,.2,.5,.6])*(2 if physical else 1)
                e=shell('actual_shell',fp['x_min_in'],fp['x_max_in'],fp['y_min_in'],fp['y_max_in'],coefficients)
                web=beam(n=2.,q=-case['dead']*d['gamma_kip_in3']*2*4,length=20.,m0=3.,v0=1.,z=-2.5)
                web.update(id='actual_web',member_id=member,section=dict(b_in=2.,h_in=5.))
                results.append(dict(case_id=case['id'],elements=[e,web],force_partition_relative_error=0.))
            floors.append(dict(level=k,responses=results))
        return dict(definition_sha256=d['identity_sha256'],source_identity=common['analysis_source_identity'],cases=cases,floors=floors)
    common['gravity']=dict(design_response=response(gcases,False));common['physical_branches']=[];common['primary']={}
    for cid,gid in (('Ehigh','high'),('Elow','low')):
        original=next(c for c in gcases if c['id']==gid);case={**original,'id':cid}
        common['physical_branches'].append(dict(groups={gid:dict(total_response=response([case],True))}))
        definition=dict(state_identity=stateid,physical_definition_sha256=d['identity_sha256'],nodes=d['nodes'],
            beams=[dict(id=member,i=100,j=101)],gravity_case=original)
        definition['identity_sha256']=_hash(definition)
        web=beam(n=0.,q=0.,length=20.,m0=-3.,v0=.2,z=0.)
        inc=dict(status='equilibrated',primary_definition_sha256=definition['identity_sha256'],source_identity=common['analysis_source_identity'],
            floor_loads=copy.deepcopy(common['p5_influences'][gid]['witnesses'][0]['loads']),
            physical_gravity_case_id=gid,frame=dict(identity_sha256='explicit_unloaded_beam_equilibrium'),
            beams={member:dict(global_force_increment=web['global_resisting_force'])})
        common['primary'][gid]=dict(definition=definition,increments={cid:inc},gravity_transfer=dict(physical_case_id=gid))
    common['compatibility']=dict(response=dict(definition_sha256=d['identity_sha256'],cases=[],floors=[]))
    specs=[];cuts={}
    for bundle in physical_response_bundles(common):
        if not bundle['source_id'].startswith('physical_ELF'):continue
        source=_PhysicalSource(bundle,common);cid=next(iter(source.cases))
        for label,interval,owner in (('core',[-1.,1.],member),('remote',[8.,10.],None)):
            key=cid+'/'+label;spec=dict(cut_id=key,floor=1,axis='x',station_in=10.,interval_in=interval,
                axial_accounting_interval_in=[8.,12.],member_id=owner,trace='plus',source_id=bundle['source_id'])
            raw=source.cut(spec,cid);states,_=physical_cut_states([raw],outer_interval_in=interval);specs.append(spec);cuts[key]=states[0]
    plan=dict(state_identity=stateid,material_inventory_id=inventory['identity_sha256'],definition_sha256=d['identity_sha256'],
        cut_specs=specs,cuts=cuts,required_cut_ids=[s['cut_id'] for s in specs],required_case_ids=['Ehigh','Elow'],required_cut_count=4)
    plan['payload_sha256']=_hash(plan)
    projector=FullTIncrementProjector(slab_thickness_in=1.,beam_depth_in=5.,flange_interval_in=[-1.,5.],web_interval_in=[-1.,1.],
        slab_E_ksi=200.,web_E_ksi=200.,axis='x',transverse_reference_in=0.)
    state=SimpleNamespace(identity=lambda:stateid,member=lambda tag:tag)
    return common,plan,projector,state


class GuardPlanTests(unittest.TestCase):
    def test_P5_window_cache_reproduces_changed_points_and_refuses_bad_selection(self):
        from Design.SMRF_Axial_Action_Accounting import primary_membrane_axial_action,select_axial_window
        from Design.SMRF_Primary_Diaphragm_Batch import PrimaryDiaphragmReadout
        from Design.SMRF_Primary_Moment_Guards import primary_membrane_source_from_readout
        common,plan,_,_=fixture();batch=analyze_primary_diaphragms(common,**settings());reader=PrimaryDiaphragmReadout(batch)
        spec=copy.deepcopy(plan['cut_specs'][0]);cid=plan['cuts'][spec['cut_id']]['identity']['combination_id']
        bundle=next(b for b in physical_response_bundles(common) if b['source_id']==spec['source_id'])
        source=_PhysicalSource(bundle,common);branch='diaphragm_fpx'
        spec['axial_window_selection']=select_axial_window(10.,4.,[0.,20.],[])
        def produce(template,mutate=False):
            raw=source.cut(template,cid)
            point=reader.cut(template['floor'],cid,branch,axis=template['axis'],station_in=template['station_in'],
                interval_in=template['interval_in'],trace=template['trace'],transverse_reference_in=raw['transverse_reference_in'])
            verified=primary_membrane_source_from_readout(point,identity=raw['identity'])
            if mutate:verified['slab_membrane_integral_kip'][0]+=.1
            return primary_membrane_axial_action(reader,template,raw,verified)
        first=produce(spec);self.assertEqual(first['axial_action_window']['status'],'computed')
        self.assertEqual(len(reader._axial_action_windows),1)
        # Same physical counting interval but another point/trace; the point
        # must be recomputed while the complete field remains identical.
        moved=copy.deepcopy(spec);moved['station_in']=11.;moved.pop('axial_window_selection')
        second=produce(moved);third=produce(moved)
        self.assertEqual(second,third);self.assertEqual(len(reader._axial_action_windows),2)
        with self.assertRaisesRegex(ValueError,'actual original point'):produce(moved,mutate=True)
        failed=copy.deepcopy(spec);failed['axial_window_selection']=select_axial_window(10.,24.,[0.,20.],[])
        self.assertEqual(produce(failed)['axial_action_window']['status'],'not_evaluated')
        self.assertEqual(len(reader._axial_action_windows),2)
        reader.floor(2);self.assertEqual(reader._axial_action_windows,{})

    def test_all_original_seven_P5_primary_and_simultaneous_states_retained(self):
        common,plan,p,state=fixture();p5=analyze_primary_diaphragms(common,**settings())
        with patch('Design.SMRF_Common_Guard_Plan.installed_full_t_projector',return_value=p):
            result=build_common_guard_plan(common,plan,p5,state=state,maximum_cuts=500)
        self.assertEqual(result['required_cut_count'],132)
        self.assertEqual(len(result['primary_guard_inventory']),2)
        for key in plan['cuts']:self.assertEqual(result['cuts'][key],plan['cuts'][key])
        for row in result['primary_guard_inventory']:
            self.assertEqual(len(row['complete_P5_branch_ids']),7)
            self.assertGreater(row['enclosing_bar_clear_interval_in'][1],5.25)
            self.assertEqual(row['effective_flange_interval_in'],[-1.,5.])
            self.assertAlmostEqual(row['primary_increment']['m_z_kip_in'],-5.)
        self.assertFalse(result['physical_displacement_superposition_claimed'])
        self.assertEqual(result['new_structural_or_inverse_solves'],0)
        extras=[r for key,r in result['cuts'].items() if key not in plan['cuts']]
        moments=[r for r in extras if r.get('strength_guard',{}).get('guard_identity','').startswith('primary_M/')]
        self.assertEqual(len(moments),12) # Two cases x two material strips x three components.
        self.assertTrue(all(r['axial_action_window']['status']=='computed' for r in moments))
        self.assertTrue(all(r['axial_action_window']['binding']['recovery_id']==r['identity']['recovery_id'] for r in moments))
        guarded=[r for r in extras if 'strength_guard' in r]
        self.assertEqual(len(guarded),124)
        self.assertTrue(all(r['axial_action_window']['status']=='computed' for r in guarded))
        self.assertTrue(all(r['axial_action_window']['binding']['recovery_id']==r['identity']['recovery_id'] for r in guarded))
        self.assertTrue(all('axial_action_window' in r for r in extras if 'strength_guard' not in r))
        self.assertTrue(all(r['material_inventory_id']=='actual_analytical_inventory' for r in extras))
        self.assertTrue(any(r.get('strength_guard',{}).get('simultaneous_primary_membrane_and_moment') for r in extras))
        with patch('Design.SMRF_Common_Guard_Plan.installed_full_t_projector',return_value=p):
            with self.assertRaisesRegex(ValueError,'budget'):
                build_common_guard_plan(common,plan,p5,state=state,maximum_cuts=100)

    def test_source_cut_and_primary_axial_premise_tamper_are_rejected(self):
        common,plan,p,state=fixture();p5=analyze_primary_diaphragms(common,**settings())
        changed=copy.deepcopy(plan);changed['cuts']['Ehigh/core']['resultants'][0]+=1.
        changed['payload_sha256']=_hash({k:v for k,v in changed.items() if k!='payload_sha256'})
        with patch('Design.SMRF_Common_Guard_Plan.installed_full_t_projector',return_value=p):
            with self.assertRaisesRegex(ValueError,'reproduce'):
                build_common_guard_plan(common,changed,p5,state=state,maximum_cuts=500)
        inc=common['primary']['high']['increments']['Ehigh'];f=inc['beams']['1:x:0:0']['global_force_increment']
        f[0]=-1.;f[6]=1.
        with self.assertRaisesRegex(ValueError,'nonzero external axial'):primary_guard_increment(common,'Ehigh','1:x:0:0',10.)


if __name__=='__main__':unittest.main()
