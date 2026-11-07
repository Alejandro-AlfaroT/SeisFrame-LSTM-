"""Final computed proof dispatch, retained Ax and source-state rejection."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from Design.Config import DesignConfig
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Final_Regularity import refresh_torsion_classification,_validate_common,refresh_actual_pattern_basis
from Design import SMRF_Story_Adapter as adapter
from Design import Design_Driver as driver
from Design.SMRF_Plan_Symmetry import METHOD as REFLECTION_METHOD
from Design.SMRF_Common_Compatibility import displacement_audit,adapt_compatibility


def torsion_fixture(a=.11,b=.09,applied=2.):
    ratio=2*max(a,b)/(a+b);ax=max(1.,min(3.,(ratio/1.2)**2));rows=[]
    for axis in ('x','y'):
        for sign in (1,-1):
            rows.append(dict(story=1,direction=axis,case=axis+('+' if sign>0 else '-'),eccentricity_sign=sign,
                delta_end_a_in=a,delta_end_b_in=b,delta_max_over_avg=ratio,
                delta_level_a_in=sign*a,delta_level_b_in=sign*b,level_max_over_avg=ratio,ax_level=ax))
    return dict(stories=rows,cases=[r['case'] for r in rows],tir=ratio,max_drift_ratio=ratio,
        tir_by_case={r['case']:ratio for r in rows},amplification_by_level={1:ax},ratio=.05,
        assessment_amplification=1.,amplification=applied,amplification_envelope_12_8_4_3=ax)


class FinalRegularityTests(unittest.TestCase):
    def test_existing_continuous_unloaded_period_does_not_call_gravity_loads(self):
        import Structure_Parameters as sp
        # This is a call-path test only. Modal/stiffness equivalence has its
        # own native evidence; no native model or eigen is requested here.
        old=sp.LATERAL_DESIGN_MODEL;sentinel={'physical_gravity_member_actions':['not_a_legacy_transfer']}
        def build():
            self.assertEqual(sp.LATERAL_DESIGN_MODEL,'all_frames')
            self.assertIs(sp.FLOOR_TRANSFER,sentinel)
        with patch.object(sp,'FLOOR_TRANSFER',sentinel),patch.object(driver.ops,'wipe') as wipe,\
             patch.object(driver,'build_model',side_effect=build) as builder,\
             patch.object(driver,'apply_gravity_loads',side_effect=AssertionError('Period must not invent gravity cache use')),\
             patch.object(driver,'run_gravity_analysis',side_effect=AssertionError('Existing period is unloaded')),\
             patch.object(driver,'run_modal_analysis',return_value=[dict(valid=True,period=1.25)]) as modal:
            self.assertEqual(driver._model_period(),1.25);builder.assert_called_once();wipe.assert_called_once()
            modal.assert_called_once_with(constraint_handler='Transformation')
        self.assertEqual(sp.LATERAL_DESIGN_MODEL,old)

    def test_actual_pattern_roster_is_complete_without_any_legacy_transfer_cache(self):
        from Design.SMRF_Demands import live_load_patterns,strength_load_combinations
        patterns=live_load_patterns(3,2);cases=strength_load_combinations(1.25,live_patterns=patterns)
        result=refresh_actual_pattern_basis({'patterns_in_strength_envelope':False},cases,sds=1.25,
            num_bay_x=3,num_bay_y=2,patterning_enabled=True)
        self.assertEqual(result['live_load_patterns'],patterns);self.assertTrue(result['patterns_in_strength_envelope'])
        self.assertEqual(len(result['live_pattern_source']['actual_pattern_combination_ids']),len(patterns))
        with self.assertRaisesRegex(ValueError,'complete inventory'):
            refresh_actual_pattern_basis({},cases[:-1],sds=1.25,num_bay_x=3,num_bay_y=2,patterning_enabled=True)

    def test_compatibility_targets_are_finite_unique_and_roof_is_selected_by_level(self):
        nodes=[dict(id=f'{k}/{i}/{j}',xyz=[i*10.,j*20.,k*12.]) for k in (1,2) for i in (0,1) for j in (0,1)]
        definition=dict(geometry=dict(nf=2,nx=1,ny=1,story_h=12.),nodes=nodes)
        levels=[dict(floor=k,design_in=dict(ux=.01*k,uy=-.02*k,rz_rad=.001*k)) for k in (2,1)]
        ids=[];values=[]
        for node in nodes:
            x,y,z=node['xyz'];k=z/12.;ids.extend([[node['id'],d] for d in (1,2,6)])
            values.extend([-.01*k+(y-7.)*.001*k,.02*k-(x-3.)*.001*k,-.001*k])
        imposed=dict(axis='x',sign=-1.,reference_xy_in=[3.,7.],levels=levels)
        case=dict(id='compat_x',gravity_case_id='G',dead=1.,live=1.,imposed=imposed)
        response=dict(status='equilibrated',cases=[case],frame=dict(physical_dof_ids=ids,cases=[dict(physical_displacements=values)]))
        self.assertEqual(len(displacement_audit(definition,response,0)),2)
        for kind in ('nan_target','nan_origin','duplicate','extra_key','bad_floor'):
            changed=copy.deepcopy(response);target=changed['cases'][0]['imposed']
            if kind=='nan_target':target['levels'][0]['design_in']['ux']=float('nan')
            elif kind=='nan_origin':target['reference_xy_in'][0]=float('nan')
            elif kind=='duplicate':target['levels'].append(copy.deepcopy(target['levels'][0]))
            elif kind=='extra_key':target['levels'][0]['design_in']['uz']=0.
            else:target['levels'][0]['floor']=2.
            with self.subTest(kind=kind),self.assertRaises(ValueError):displacement_audit(definition,changed,0)
        with self.assertRaises(ValueError):displacement_audit(definition,response,0,absolute_tolerance=float('nan'))
        second=copy.deepcopy(case);second['id']='compat_y';second['imposed']['axis']='y'
        response['cases'].append(second);response['frame']['cases'].append(copy.deepcopy(response['frame']['cases'][0]))
        with patch('Design.SMRF_Common_Compatibility.build_member_actions',return_value={'action':{}}):
            result=adapt_compatibility(None,definition,{'response':response},cd=5.5,importance_factor=1.)
        self.assertAlmostEqual(result['design_displacements']['axes']['x']['roof_design_displacement_in'],.02)
        self.assertAlmostEqual(result['design_displacements']['axes']['y']['roof_design_displacement_in'],.04)
        with self.assertRaises(ValueError):adapt_compatibility(None,definition,{'response':response},cd=float('nan'),importance_factor=1.)

    def test_reflected_distribution_resolves_torsion_without_inventing_total_strength(self):
        strength=dict(one_side_fraction=.5,independent_distribution_bound=dict(method=REFLECTION_METHOD,
            verified=True,one_side_fraction=.5,problems=[]))
        old=torsion_fixture();original=copy.deepcopy(old)
        result=refresh_torsion_classification(old,strength,num_floors=1,accidental_ratio=.05)
        self.assertEqual(result['classification']['label'],'none')
        self.assertEqual(result['amplification_required'],1.)
        self.assertEqual(result['amplification'],2.)
        self.assertEqual(result['stories'],original['stories']);self.assertEqual(old,original)
        self.assertFalse(result['final_regularity_binding']['computed_story_certificate_valid'])

    def test_missing_proof_stays_unknown_and_unapplied_Ax_cannot_be_refreshed_away(self):
        strength=dict(one_side_fraction=.5,applicability=dict(status='verified'))
        result=refresh_torsion_classification(torsion_fixture(),strength,num_floors=1,accidental_ratio=.05)
        self.assertEqual(result['classification']['label'],'unresolved')
        with self.assertRaisesRegex(ValueError,'greater Ax'):
            refresh_torsion_classification(torsion_fixture(.15,.05,1.),strength,num_floors=1,accidental_ratio=.05)
        broken=torsion_fixture();broken['stories'][0]['delta_level_a_in']+=.02
        with self.assertRaisesRegex(ValueError,'failed replay'):
            refresh_torsion_classification(broken,strength,num_floors=1,accidental_ratio=.05)

    def test_computed_dispatch_requires_certificate_payload_and_current_source(self):
        # This tests dispatch only; primal/dual mathematical replay is covered
        # independently by test_story_strength_equilibrium and saved readback.
        result=dict(method=adapter.METHOD,adapter_method=adapter.ADAPTER_METHOD,applicability=dict(status='verified'))
        source=adapter.source_identity()
        result['verification']=dict(method=adapter.VERIFICATION_METHOD,passed=True,source_identity=source,
            certificate_payload_sha256=_hash(adapter._certificate_payload(result)))
        cfg=DesignConfig();self.assertFalse(cfg.verification.story_strength_model_verified)
        self.assertTrue(driver._strength_model_verified(cfg,result))
        result['stories']=[{'changed':True}]
        self.assertFalse(driver._strength_model_verified(cfg,result))
        result.pop('stories');result['verification']['source_identity']={'old':'source'}
        cfg.verification.story_strength_model_verified=True
        cfg.verification.asserted_by='legacy fixture';cfg.verification.assertion_date='2026-11-03';cfg.verification.assertion_basis='Explicit legacy-only fixture'
        self.assertFalse(driver._strength_model_verified(cfg,result))

    def test_common_rejects_mutated_payload_before_any_strength_calculation(self):
        common=dict(method='coordinated_actual_common_floor_member_actions_v1',status='responses_computed',source_unchanged=True)
        common['payload_sha256']=_hash(common);common['actions']=[{'silently_changed':True}]
        with self.assertRaisesRegex(ValueError,'immutable'):_validate_common(SimpleNamespace(identity=lambda:'same'),common)


if __name__=='__main__':unittest.main()
