"""Independent 25.4 arithmetic and actual-cut identity/edge regressions."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import Structure_Parameters as sp
from Design import SMRF_Actual_Bar_Development as dev

class Arithmetic(unittest.TestCase):
    def straight(self,**changes):
        args=dict(db_in=.5,bar_size=4,fy_ksi=60.,fc_ksi=4.,cb_in=1.5,concrete_below_in=6,
            minimum_center_spacing_in=6.)
        return dev.straight_lengths(**{**args,**changes})
    def test_independent_straight_length_and_ratio_cap(self):
        out=self.straight()
        self.assertAlmostEqual(out['ld_tension_in'],max(12.,.075*60000*.8/(math.sqrt(4000)*2.5)*.5))
        self.assertEqual(out['confinement_ratio'],2.5)
        self.assertAlmostEqual(out['ld_compression_in'],max(.02*60000/math.sqrt(4000)*.5,.0003*60000*.5,8.))
    def test_top_bar_factor_uses_actual_cast_depth(self):
        bottom=self.straight(concrete_below_in=12,cb_in=.75);top=self.straight(concrete_below_in=12.001,cb_in=.75)
        self.assertAlmostEqual(top['ld_tension_in']/bottom['ld_tension_in'],1.3)
    def test_grade80_mandatory_ktr_is_not_erased_by_zero_length_credit(self):
        out=self.straight(fy_ksi=80.,minimum_center_spacing_in=5.)
        self.assertEqual(out['status'],'fail');self.assertEqual(out['ktr_required_25_4_2_2_in'],.25)
        good=self.straight(fy_ksi=80.,minimum_center_spacing_in=5.,ktr_actual_lower_bound_in=.25)
        self.assertEqual(good['status'],'pass');self.assertEqual(out['ld_tension_in'],good['ld_tension_in'])
    def test_hook_algebra_has_no_strength_phi(self):
        out=dev.hook_tension(db_in=.5,bar_size=4,fy_ksi=60.,fc_ksi=4.,minimum_hook_spacing_in=6,
            side_cover_in=4,top_bottom_cover_in=1.5,radius_in=1.75,extension_in=6,tail_fits=True,available_in=20)
        self.assertAlmostEqual(out['ldh_required_in'],max(60000*(4000/15000+.6)/(55*math.sqrt(4000))*.5**1.5,6))
        self.assertEqual(out['status'],'pass')
    def test_short_bend_or_tail_is_known_failure(self):
        for radius,tail in ((1.,6),(1.75,5.99)):
            out=dev.hook_tension(db_in=.5,bar_size=4,fy_ksi=60.,fc_ksi=5.,minimum_hook_spacing_in=6,
                side_cover_in=4,top_bottom_cover_in=2,radius_in=radius,extension_in=tail,tail_fits=True,available_in=40)
            self.assertEqual(out['status'],'fail')
    def test_unproved_mandatory_hook_ties_are_unknown(self):
        out=dev.hook_tension(db_in=.5,bar_size=4,fy_ksi=60.,fc_ksi=5.,minimum_hook_spacing_in=6,
            side_cover_in=1.5,top_bottom_cover_in=1.5,radius_in=1.75,extension_in=6,tail_fits=True,available_in=40)
        self.assertEqual(out['status'],'unknown')
    def test_hook_never_develops_compression(self):
        out=dev.side_development(3,[0,100],self.straight(),[{'status':'pass'},{'status':'pass'}])
        self.assertEqual(out['status'],'fail');self.assertEqual(out['sides'][0]['tension_status'],'pass')
        self.assertEqual(out['sides'][0]['compression_status'],'fail')
    def test_both_sides_required(self):
        out=dev.side_development(50,[0,52],self.straight(),[{'status':'fail'},{'status':'fail'}])
        self.assertEqual(out['status'],'fail');self.assertEqual(out['sides'][0]['status'],'pass')

class PhysicalCut(unittest.TestCase):
    def setUp(self):
        self.state=SimpleNamespace(identity=lambda:'state1',members=lambda:[])
        footprint={f'{a}_{side}_in':v for a in ('x','y') for side,v in (('min',-10.),('max',110.))}
        self.bars=[dict(id=f'floor_1/slab/{a}_top/0',axis=a,s_in=50.,mat_name=f'{a}_top',z_in=2.5,
            area_in2=.2,db_in=.5,fy_ksi=60.,es_ksi=29000.,straight_run_interval_in=[-3.25,103.25],operator_owner='slab') for a in ('x','y')]
        mat=dict(bar_size=4,spacing_in=6.,straight_run_interval_in=[-3.25,103.25],hook_vertical_axes_in=[-5.,105.],
            hook_radius_in=1.75,hook_extension_in=6.,hook_tail_depth_in=9.25,hook_tail_depth_limit_in=20.,hook_tail_fits=True)
        self.placement={'status':'pass','layout_sha256':'layout1','floors':{'1':{'footprint':footprint,
            'layers':{'x_top':mat,'y_top':mat}}}}
        self.inv={'state_identity':'state1','layout_sha256':'layout1','slab_thickness_in':8.,
            'floors':{'1':{'bars':self.bars,'footprint':footprint}},'beam_geometry':{},'beam_bars':{}}
        self.inv['identity_sha256']=dev._hash(self.inv)
        self.cut={'floor':1,'axis':'x','station_in':50.,'interval_in':[30.,70.],
            'axial_accounting_interval_in':[49.,51.],'physical_bar_ids':[b['id'] for b in self.bars]}
    def evaluate(self,**changes):
        from Model import Member_Groups as mg
        from Design import SMRF_Joint_Mat_Physical as mats
        with patch.object(mg,'active',return_value=self.state),patch.object(mats,'global_mat_placement',return_value=self.placement),patch.object(sp,'SLAB_THICKNESS_IN',8.):
            return dev.actual_bar_development(self.state,self.inv,[{**self.cut,**changes}],slab_fc_ksi=5.)
    def test_exact_ids_material_and_transverse_edges(self):
        result=self.evaluate();self.assertEqual(result['status'],'pass')
        cross=next(e for e in result['cuts'][0]['evidence'] if e['bar_axis']=='y')
        self.assertEqual(cross['critical_stations_along_bar_in'],[30.,70.])
        self.assertEqual(cross['material_inventory_id'],self.inv['identity_sha256'])
        self.assertEqual(cross['station_interval_in'],[50.,50.])
        self.assertTrue(cross['full_fy_only']);self.assertFalse(cross['actual_demand_failure_established'])
    def test_strip_extended_into_hook_end_region_is_computed_failure(self):
        result=self.evaluate(interval_in=[-3.25,70.]);self.assertEqual(result['status'],'fail')
        cross=next(e for e in result['cuts'][0]['evidence'] if e['bar_axis']=='y')
        self.assertEqual(cross['checks'][0]['sides'][0]['straight_available_in'],0.)
    def test_changed_inventory_rejected(self):
        self.inv['floors']['1']['bars'][0]['s_in']+=1
        with self.assertRaisesRegex(ValueError,'content'):self.evaluate()
    def test_missing_physical_id_stays_unknown(self):
        result=self.evaluate(physical_bar_ids=['missing'])
        self.assertEqual(result['status'],'unknown');self.assertEqual(result['cuts'][0]['missing_physical_bar_ids'],['missing'])
    def test_explicit_local_ids_preserve_actual_development_outside_old_window(self):
        self.bars[1]['s_in']=53.;self.inv['identity_sha256']=dev._hash({k:v for k,v in self.inv.items() if k!='identity_sha256'})
        old=self.evaluate();self.assertEqual(old['cuts'][0]['missing_physical_bar_ids'],[self.bars[1]['id']])
        local=self.evaluate(bar_selection_basis='explicit_local_physical_ids',axial_accounting_interval_in=None)
        self.assertEqual(local['status'],'pass')
        cross=next(e for e in local['cuts'][0]['evidence'] if e['bar_axis']=='y')
        self.assertEqual(cross['critical_stations_along_bar_in'],[30.,70.]);self.assertIsNone(cross['axial_accounting_interval_in'])
        self.assertEqual(cross['full_fy_development_domain']['physical_straight_run_interval_in'],[-3.25,103.25])
        self.assertFalse(cross['actual_demand_failure_established'])
        for changes in (dict(physical_bar_ids=[]),dict(physical_bar_ids=['same','same']),dict(axial_accounting_interval_in=[49.,51.])):
            with self.assertRaises(ValueError):self.evaluate(**{'bar_selection_basis':'explicit_local_physical_ids','axial_accounting_interval_in':None,**changes})

    def test_local_pitch_adapter_computes_evidence_for_exact_one_sided_bar(self):
        from Design.SMRF_Pitch_Spatial_Section import developed_pitch_spatial_section
        from Model import Member_Groups as mg
        from Design import SMRF_Joint_Mat_Physical as mats
        self.bars[1].update(s_in=47.,z_in=-2.5)
        for bar in list(self.bars):self.bars.append(dict(bar,id=bar['id']+'_next',s_in=bar['s_in']+6))
        self.inv['identity_sha256']=dev._hash({k:v for k,v in self.inv.items() if k!='identity_sha256'})
        with patch.object(mg,'active',return_value=self.state),patch.object(mats,'global_mat_placement',return_value=self.placement),patch.object(sp,'SLAB_THICKNESS_IN',8.):
            result=developed_pitch_spatial_section(self.state,self.inv,floor=1,axis='x',station_in=50.,interval_in=[30.,70.],
                fc_ksi=5.,ec_ksi=4000.,phi_requirements={'NM':.65},material_scope='slab_only',trace='plus')
        self.assertEqual(result['status'],'inventory_constructed',result['issues']);self.assertEqual(result['development_status'],'pass')
        self.assertEqual(result['constructor']['transverse_rows'][0]['physical_bar_ids'],[self.bars[3]['id']])
        selected={e['physical_bar_ids'][0] for e in result['actual_development_evidence']['cuts'][0]['evidence']}
        self.assertEqual(selected,set(result['physical_bar_ids']));self.assertNotIn(self.bars[1]['id'],selected)
        self.assertFalse(result['continuous_action_coverage_verified']);self.assertFalse(result['engineering_qualified'])

class BeamOwnership(unittest.TestCase):
    def fixture(self):
        design=SimpleNamespace(b_in=30.,h_in=28.,fc_ksi=5.,bar_size=8,stirrup_bar_size=5,stirrup_spacing_in=3.)
        members={str(i):SimpleNamespace(group_id='same',design=design) for i in (1,2,3)}
        bars=[];geometry={};family={'face':'top','row':0,'bend_radius_in':3.5,'required_ldh_in':15.,'standard_extension_in':12.}
        for i in (1,2,3):
            run=[(i-1)*100.,i*100.];physical=[run[0]-3 if i==1 else run[0],run[1]+3 if i==3 else run[1]]
            bar=dict(id=f'member_{i}/beam/top_0_0',source_lane_id='top_0_0',member_id=str(i),operator_owner='web_line',
                axis='x',s_in=0.,z_in=.5,db_in=1.,fy_ksi=60.,straight_run_interval_in=run)
            bars.append(bar)
            ends=[{'joint_id':f'j{i-1}','beam_context_status':'pass','hook_families':[family] if i==1 else []},
                  {'joint_id':f'j{i}','beam_context_status':'pass','hook_families':[family] if i==3 else []}]
            geometry[str(i)]={'line_in':0.,'straight_run_evidence':{'geometry_complete':True,'physical_detail_status':'pass',
                'evidence_id':f'e{i}','end_evidence':ends,'bars':[{'source_lane_id':'top_0_0','member_grid_interval_in':run,'straight_run_interval_in':physical}]}}
        return members,bars,{'beam_geometry':geometry}
    def test_continuous_grid_planes_are_not_false_bar_ends(self):
        from Design import SMRF_Splice_Physical as splice
        members,bars,inventory=self.fixture()
        with patch.object(splice,'group_beam_lanes',return_value={'same':{'coordinates':[{}]*8}}),patch.object(splice,'beam_cage_geometry',return_value={'passes':True,'segments':[],'hoop_leg_horizontal_projection_factor':1.}):
            out=dev._beam_station(bars[1],150.,inventory,members,bars,8.,5.,True,{})
        self.assertEqual(out['status'],'pass');self.assertEqual(out['straight_run_interval_in'],[-3.,303.])
        self.assertEqual(len(out['continuous_physical_bar_owners']),3)
    def test_different_group_does_not_invent_continuity(self):
        members,bars,inventory=self.fixture();members['1'].group_id='different'
        out=dev._beam_station(bars[1],150.,inventory,members,bars,8.,5.,True,{})
        self.assertEqual(out['status'],'unknown')

if __name__=='__main__':unittest.main()
