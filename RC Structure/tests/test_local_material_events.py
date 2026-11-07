"""Known physical pitch/end intervals and actual Q4 force reconstruction."""
import copy
import json
import math
import unittest
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Actual_Bar_Development import METHOD as DEVELOPMENT_METHOD,side_development,straight_lengths,full_fy_developed_domain
from Design.SMRF_Local_Material_Events import local_material_events,verify_local_material_events
from Design.SMRF_Physical_Floor_Strength import plan_physical_cut_family
from Design.SMRF_Local_Pitch_Strength import _check_point
from tests.test_bar_pitch_layers import fixture as mat_fixture


def seal(row,key='evidence_id'):
    row[key]=_hash({k:v for k,v in row.items() if k!=key})
    return row


def fixture(crossing=False):
    inv=mat_fixture();inv['floors']['1']['bars']+=mat_fixture('y')['floors']['1']['bars']
    bars=inv['floors']['1']['bars']
    for bar in bars:bar['id']=bar['axis']+'/'+bar['id']
    if crossing:bars.append(dict(bars[-1],id='actual_y_beam',operator_owner='web_line',s_in=12.,z_in=2.))
    seal(inv,'identity_sha256');fp=inv['floors']['1']['footprint'];cuts=[]
    # Independently known No.4, fc=4, fy=60 domain: 12-in tension governs
    # over 9-in compression, leaving [14,16] within the [2,28] run.
    lengths=straight_lengths(db_in=.5,bar_size=4,fy_ksi=60,fc_ksi=4,cb_in=1.5,concrete_below_in=6,minimum_center_spacing_in=6)
    for bar in bars:
        axis=bar['axis'];cross='y' if axis=='x' else 'x';run=bar['straight_run_interval_in'];station=sum(run)/2
        spec=dict(floor=1,axis=axis,station_in=station,interval_in=[fp[cross+'_min_in'],fp[cross+'_max_in']],
            physical_bar_ids=[bar['id']],bar_selection_basis='explicit_local_physical_ids')
        check=side_development(station,run,copy.deepcopy(lengths),[{'status':'fail'},{'status':'fail'}])
        evidence=seal(dict(state_identity=inv['state_identity'],material_inventory_id=inv['identity_sha256'],physical_bar_ids=[bar['id']],
            full_fy_only=True,status=check['status'],checks=[check],full_fy_development_domain=full_fy_developed_domain(check)))
        cuts.append(dict(specification=spec,evidence=[evidence],missing_physical_bar_ids=[],status=evidence['status']))
    development=seal(dict(method=DEVELOPMENT_METHOD,state_identity=inv['state_identity'],material_inventory_id=inv['identity_sha256'],
        full_fy_only=True,cuts=cuts,status='pass'))
    return inv,development


def physical_fixture(inv):
    fp=inv['floors']['1']['footprint'];desc=dict(level=1,footprint=fp,supports=[],beams=[])
    definition=dict(state_identity=inv['state_identity'],identity_sha256='analytical_geometry',floors=[desc],slab_thickness_in=8.,gamma_kip_in3=.0001)
    row=dict(id='cell',kind='slab',positions_in=[[0,0,0],[30,0,0],[30,30,0],[0,30,0]],
        raw_resultants=[[1.,2.,.1,3.,4.,.2,.3,.4]]*4)
    response=dict(definition_sha256=definition['identity_sha256'],cases=[dict(id='G',dead=1.,live=1.)],
        floors=[dict(level=1,grid={'x_in':[0.,30.],'y_in':[0.,30.]},responses=[dict(case_id='G',elements=[row],force_partition_relative_error=0.)])])
    bundle=dict(source_id='analytical_actual',response=response,material_inventory_id=inv['identity_sha256'],
        response_payload_sha256=_hash(response),analysis_input_identity='explicit_analytical_physical_input')
    return definition,[bundle]


class LocalMaterialEventsTests(unittest.TestCase):
    def test_known_run_pitch_development_and_crossing_cylinder_boundaries(self):
        inv,dev=fixture(True);out=local_material_events(inv,1,dev)
        for axis in ('x','y'):
            got=set(out['events'][axis]['coordinates_in'])
            self.assertTrue({0.,2.,8.,14.,16.,20.,26.,28.,30.}<=got)
        radius=math.sqrt(.2/math.pi)
        self.assertTrue({12-radius,12.,12+radius}<=set(out['events']['x']['coordinates_in']))
        for row in out['domains'].values():self.assertEqual(row['interval_in'],[14.,16.])
        self.assertEqual(out['unknown_domains'],[])
        self.assertFalse(out['continuous_capacity_verified'])
        self.assertTrue(out['curved_crossing_voids_still_vary_inside_intervals'])
        self.assertEqual(verify_local_material_events(json.loads(json.dumps(out)),inv),out)

    def test_unknown_end_development_stays_unknown(self):
        inv,dev=fixture();e=dev['cuts'][0]['evidence'][0];e['checks'][0]['straight_required']['status']='unknown'
        e['full_fy_development_domain']=dict(status='unknown',station_domains=[full_fy_developed_domain(e['checks'][0])])
        seal(e);seal(dev);out=local_material_events(inv,1,dev)
        self.assertEqual(len(out['unknown_domains']),1)
        self.assertNotIn(e['physical_bar_ids'][0],out['domains'])
        self.assertFalse(out['continuous_capacity_verified'])

    def test_missing_duplicate_and_resealed_domain_or_event_changes_refused(self):
        inv,dev=fixture()
        for mode in ('missing','duplicate','domain','check'):
            with self.subTest(mode=mode):
                bad=copy.deepcopy(dev)
                if mode=='missing':bad['cuts'].pop()
                elif mode=='duplicate':bad['cuts'].append(copy.deepcopy(bad['cuts'][0]))
                else:
                    e=bad['cuts'][0]['evidence'][0]
                    if mode=='domain':e['full_fy_development_domain']['interval_in'][0]-=1
                    else:e['checks'][0]['straight_required']['ld_tension_in']+=1
                    seal(e)
                seal(bad)
                with self.assertRaises(ValueError):local_material_events(inv,1,bad)
        out=local_material_events(inv,1,dev);out['events']['x']['coordinates_in'].remove(16.);seal(out,'payload_sha256')
        with self.assertRaisesRegex(ValueError,'differ from actual'):verify_local_material_events(out,inv)

    def test_local_plan_retains_both_event_traces_and_exact_actual_raw_actions(self):
        inv,dev=fixture();events=local_material_events(inv,1,dev);definition,bundles=physical_fixture(inv)
        local=plan_physical_cut_family(definition,inv,bundles,planner_identity='local_events',maximum_cuts=100,
            local_pitch_actions=True,local_material_events_by_floor={'1':events})
        self.assertEqual(local['required_cut_count'],48)
        self.assertFalse(local['continuous_capacity_verified'])
        self.assertFalse(local['nonlinear_interaction_between_samples_bound_claimed'])
        for axis in ('x','y'):
            for where in events['events'][axis]['coordinates_in'][1:-1]:
                self.assertEqual({s['trace'] for s in local['cut_specs'] if s['axis']==axis and s['station_in']==where},{'minus','plus'})
        for spec in local['cut_specs']:
            row=local['cuts'][spec['cut_id']];_check_point(row,spec,inv)
            for got,expected in zip(row['resultants'][:6],[30.,60.,3.,90.,120.,6.]):self.assertAlmostEqual(got,expected,10)
            self.assertAlmostEqual(row['resultants'][6],0.,10)
            self.assertEqual(spec['material_scope'],'slab_only')
        with self.assertRaisesRegex(ValueError,'budget'):
            plan_physical_cut_family(definition,inv,bundles,planner_identity='bounded',maximum_cuts=47,
                local_pitch_actions=True,local_material_events_by_floor={'1':events})
        with self.assertRaisesRegex(ValueError,'every selected floor'):
            plan_physical_cut_family(definition,inv,bundles,planner_identity='missing',local_pitch_actions=True)
        legacy=plan_physical_cut_family(definition,inv,bundles,planner_identity='legacy')
        self.assertEqual(legacy['required_cut_count'],6)
        self.assertNotIn('action_contract',legacy)


if __name__=='__main__':unittest.main()
