"""Independent wrench transport, exact-width and same-steel reservation tests."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Common_Guard_Plan import PreparedCommonGuardSources
from Design.SMRF_Common_Torsion_Guards import plan_record,longitudinal_reserve
from Design.SMRF_Common_Torsion_Recovery import cut_wrench
from Design.SMRF_Beam_Torsion import solid_section
from Design.SMRF_Primary_Diaphragm_Batch import analyze_primary_diaphragms
from Design.SMRF_Beam_Torsion_Detail import transverse_variants,variant_design
from tests.test_common_guard_plan import fixture,settings
from tests.test_beam_torsion_physical import D
from tests.test_physical_floor_strength import PhysicalStrengthTests
from Design.SMRF_Physical_Floor_Strength import assess_physical_cut_family


class ExactThresholdGuards(unittest.TestCase):
    def run_fixture(self):
        common,old,p,state=fixture();common['p5']=analyze_primary_diaphragms(common,**settings())
        # At z=0 positive X torque is -Fi_rx. Supply balanced, nonzero
        # primary torque so accidentally discarding it is observable.
        inc=common['primary']['high']['increments']['Ehigh']['beams']['1:x:0:0']['global_force_increment']
        inc[3]=-7.;inc[9]=7.
        context=PreparedCommonGuardSources(common);spec=dict(old['cut_specs'][0],interval_in=[-1.,5.])
        member=SimpleNamespace(story_or_floor=1);state.member=lambda tag:member
        raw=context.sources[spec['source_id']].cut(spec,'Ehigh')
        section=solid_section([[-1,1,-5,0],[-1,5,-1,0]],5.,section_id='exact_geometry')
        section.update(axis='x')
        row=dict(state_identity=state.identity(),section_id=section['section_id'],analysis_input_identity=common['input_signature'],
            material_inventory_id=common['inventory']['identity_sha256'],member_tag='1:x:0:0',station_in=10.,trace='plus',
            interval_in=[-1.,5.],reference_z_in=-2.5,reference_s_in=0.,case_id=spec['source_id']+'/Ehigh',raw_case_id='Ehigh',
            cut_id='member_mid',source_id=spec['source_id'],recovery_id=raw['identity']['recovery_id'])
        row.update(cut_wrench(raw,reference_s_in=0.,reference_z_in=-2.5))
        with patch('Design.SMRF_Common_Guard_Plan.installed_full_t_projector',return_value=p):
            plan=plan_record(state,common,row,section,maximum_cuts=500,prepared_sources=context)
        return common,plan,row,raw

    def test_exact_threshold_axial_never_uses_enlarged_material_width(self):
        common,plan,row,raw=self.run_fixture()
        self.assertEqual(plan['exact_torsion_interval_in'],[-1.,5.])
        self.assertGreater(plan['material_interval_in'][1],5.25)
        first=plan['demands'][0]
        self.assertAlmostEqual(first['N_tension_kip'],14.) # 2 kip/in * 6 in + 2 kip web
        self.assertAlmostEqual(first['T_about_positive_axis_kipin'],row['T_about_positive_axis_kipin'])
        self.assertEqual(len(plan['demands']),66) # physical, duplicated effective core, 3 M, 7 + 21 P5, times 2 T states
        self.assertFalse(plan['physical_displacement_superposition_claimed'])
        self.assertTrue(any('/collector_eq_omega' in r['guard_cut_id'] for r in plan['demands']))

    def test_primary_torque_and_P5_plane_transport_are_present_once(self):
        common,plan,row,raw=self.run_fixture();zr=row['reference_z_in'];context=PreparedCommonGuardSources(common)
        spec=plan['exact_threshold_plan']['cut_specs'][0]
        graw=context.sources['actual_frozen_gravity'].cut(spec,'high');g=cut_wrench(graw,reference_s_in=0.,reference_z_in=zr)
        # The fixture has zero primary in-plane shear; its primary torque is 7.
        target=next(d for d in plan['demands'] if d['kind']=='gravity_plus_primary_T')
        self.assertAlmostEqual(target['T_about_positive_axis_kipin'],g['T_about_positive_axis_kipin']+7.)
        for d in plan['demands']:
            a=plan['exact_threshold_plan']['cuts'][d['guard_cut_id']]
            if '|guard:P5' not in d['guard_cut_id']:continue
            if d['kind']=='physical_T_with_same_guard_membrane':
                expected=row['T_about_positive_axis_kipin']+zr*(a['resultants'][2]-raw['resultants'][2])
            else:expected=g['T_about_positive_axis_kipin']+7.+zr*(a['resultants'][2]-graw['resultants'][2])
            self.assertAlmostEqual(d['T_about_positive_axis_kipin'],expected,11)


class ActualTorsionDetail(unittest.TestCase):
    def test_local_splice_spacing_cannot_be_replaced_by_baseline(self):
        detail=dict(status='pass',hoop_stations_in=[45.,55.],transverse_detail=dict(bar_size=5,legs=4,spacing_in=10.,zone_in=[44.5,55.5]))
        left=transverse_variants(D,detail,station_from_left_face_in=10,clear_span_in=100)
        middle=transverse_variants(D,detail,station_from_left_face_in=50,clear_span_in=100)
        edge=transverse_variants(D,detail,station_from_left_face_in=45,clear_span_in=100)
        self.assertEqual(left['variants'][0]['spacing_in'],3.)
        self.assertEqual(middle['variants'][0]['spacing_in'],10.)
        self.assertEqual(len(edge['variants']),2)
        self.assertEqual(variant_design(D,middle['variants'][0]).stirrup_spacing_in,10.)
        detail['status']='not_evaluated'
        self.assertEqual(transverse_variants(D,detail,station_from_left_face_in=50,clear_span_in=100)['variants'],[])

    def test_unbalanced_native_torque_is_rejected(self):
        from tests.test_physical_floor_cuts import beam
        from Design.SMRF_Physical_Floor_Cuts import web_section
        b=beam(q=0.);b['global_resisting_force'][3]=1.
        with self.assertRaisesRegex(ValueError,'equilibrium'):web_section(b,5.,uniform_vertical_load_kip_in=0.)


class CompatibleReserve(unittest.TestCase):
    def test_real_compatible_zero_state_and_missing_guard_refusal(self):
        inv,spec,action,dev=PhysicalStrengthTests().fixture();spec=dict(spec,member_id=None)
        assessed=assess_physical_cut_family(None,inv,[spec],{'c':action},slab_fc_ksi=5.,ec_ksi=4000.,
            phi_requirements=dict(NM=.65,diaphragm_shear=.6),required_cut_ids=['c'],required_case_ids=[action['identity']['combination_id']],
            assessment_identity='explicit_constitutive_fixture',development_result=dev)
        # The passing slab-only witness does not own a beam cage and cannot
        # be relabelled into a complete torsion reinforcement reserve.
        record=dict(case_id='case',cut_id='c',state_identity='actual_state',recovery_id='r',member_tag='beam')
        plan=dict(required_cut_ids=['c'],cut_specs=[spec],cuts={'c':action})
        self.assertEqual(longitudinal_reserve(assessed,plan,record,['xtop'])['status'],'not_evaluated')
        plan=dict(required_cut_ids=['missing'],cut_specs=[],cuts={})
        with self.assertRaises(ValueError):longitudinal_reserve(assessed,plan,record,['xtop'])


if __name__=='__main__':unittest.main()
