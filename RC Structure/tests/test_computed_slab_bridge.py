"""No-solve physical-slab bridge oracles; synthetic data do not qualify a candidate."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests import grouped_fixture as gf
import Structure_Parameters as sp
from Model import Member_Groups as mg,Member_Properties as mp
from Design import SMRF_Slab_Reinforcement as sr,SMRF_Computed_Slab as cs,Code_Check_Contract as cc
from Design.Config import DesignConfig


def inputs():
    return dict(thickness_in=7.5,fc_ksi=5.,fy_ksi=60.,max_aggregate_size_in=.75,
        exposure='sheltered_interior',steel_specification='ASTM A706',concrete_type='normalweight',
        panel_ids=['actual_panel'],num_floor=4)


class ComputedSlabBridge(unittest.TestCase):
    def test_computed_physical_transfer_is_bound_without_legacy_solver(self):
        from Design import Grouped_Record as gr
        source={'actual_method.py':'a'*64};actions=[{'id':'actual_G'}]
        transfer=dict(physical_gravity_member_actions=actions,expected_physical_gravity_case_ids=['actual_G'],
            state_identity='same_state',source_identity=gr._sha256(source),physical_definition_sha256='definition')
        common=dict(capacity_gravity_transfer=transfer,story_strength_gravity_actions=actions,
            source_identity=source,definition_sha256='definition')
        record=dict(member_groups={'sha256':'same_state'},floor_transfer=transfer,
            request_identity={'policy':{'floor_analysis':{'common_action_procedure':{'kind':'computed_common_actions_v1'}}}})
        with patch.object(gr,'_bound_common_actions',return_value=common),patch(
                'Design.SMRF_Floor_Transfer.validate_floor_transfers_by_floor',side_effect=AssertionError('legacy transfer')):
            self.assertEqual(gr._physical_gravity_transfer(record),transfer)
            changed=copy.deepcopy(record);changed['floor_transfer']['source_identity']='stale'
            with self.assertRaises(ValueError):gr._physical_gravity_transfer(changed)
            changed=copy.deepcopy(record);changed['request_identity']['policy']['floor_analysis']={}
            with self.assertRaises(ValueError):gr._physical_gravity_transfer(changed)

    def test_common_physical_transfer_rejects_duplicate_gravity_cases(self):
        from Design import Grouped_Record as gr
        source={'actual.py':'b'*64};actions=[{'id':'G'},{'id':'G'}]
        transfer=dict(physical_gravity_member_actions=actions,expected_physical_gravity_case_ids=['G','G'],
            state_identity='state',source_identity=gr._sha256(source),physical_definition_sha256='def')
        record=dict(member_groups={'sha256':'state'},floor_transfer=transfer,
            request_identity={'policy':{'floor_analysis':{'common_action_procedure':{'kind':'computed_common_actions_v1'}}}})
        common=dict(capacity_gravity_transfer=transfer,story_strength_gravity_actions=actions,
            source_identity=source,definition_sha256='def')
        with patch.object(gr,'_bound_common_actions',return_value=common):
            with self.assertRaisesRegex(ValueError,'duplicated'):gr._physical_gravity_transfer(record)

    def test_layout_constructor_keeps_all_four_catalog_choices(self):
        spacing={'x_top':3,'x_bottom':8,'y_top':10,'y_bottom':12}
        layout=sr.computed_layout(inputs(),4,spacing)
        self.assertEqual({k:v['spacing_in'] for k,v in layout['layers'].items()},spacing)
        self.assertEqual(layout['layers']['x_top']['effective_depth_in'],6.5)
        self.assertEqual(layout['layers']['y_bottom']['effective_depth_in'],6.)
        self.assertNotIn('verified',layout);self.assertNotIn('asserted_by',layout)

    def test_layout_does_not_silently_add_catalog_options(self):
        with self.assertRaises(ValueError):sr.computed_layout(inputs(),8,{k:6 for k in ('x_top','x_bottom','y_top','y_bottom')})
        with self.assertRaises(ValueError):sr.computed_layout(inputs(),4,{k:14 for k in ('x_top','x_bottom','y_top','y_bottom')})

    def test_unknown_material_inventory_is_not_developed(self):
        with gf.frame():
            state=gf.uniform_state();mg.install(state)
            proof=cs.nominal_development(state,None,inputs(),{})
            self.assertEqual(proof['status'],'unknown');self.assertEqual(proof['members'],{})

    def test_nominal_and_probable_ignore_shared_failure_flags(self):
        from Design.Group_Capacity import beam_probable_strengths
        with gf.frame(slab=7.5):
            state=gf.uniform_state();mg.install(state)
            layout=sr.computed_layout(inputs(),4,{k:6 for k in ('x_top','x_bottom','y_top','y_bottom')})
            member=next(m for m in state.members() if not m.is_column)
            sp.SLAB_REINFORCEMENT={'layout':layout,'method_version':cs.METHOD,'shared_status':'pass'}
            before=beam_probable_strengths(member)
            sp.SLAB_REINFORCEMENT['shared_status']='fail'
            sp.SLAB_REINFORCEMENT['reserved_steel_fraction']=.99
            self.assertEqual(before,beam_probable_strengths(member))
            self.assertGreater(before['mpr_negative_kip_in'],before['mn_negative_kip_in'])

    def test_integrity_counts_actual_phase_in_longitudinal_bar_envelope(self):
        with gf.frame():
            state=gf.uniform_state();mg.install(state)
            placement={'floors':{str(f):{'layers':{
                'x_bottom':{'db_in':.5,'phase_in':0.,'global_transverse_coordinates_in':[j*sp.BAY_Y+d for j in range(sp.NUM_BAY_Y+1) for d in (-1.,1.)]},
                'y_bottom':{'db_in':.5,'phase_in':0.,'global_transverse_coordinates_in':[i*sp.BAY_X+d for i in range(sp.NUM_BAY_X+1) for d in (-1.,1.)]}}}
                for f in range(1,sp.NUM_FLOOR+1)}}
            result=cs.physical_integrity(state,placement)
            self.assertEqual(result['status'],'pass')
            self.assertTrue(all(r['count']==2 for r in result['joints']))
            placement['floors']['1']['layers']['x_bottom']['global_transverse_coordinates_in'].remove(1.)
            failed=cs.physical_integrity(state,placement)
            self.assertEqual(failed['status'],'fail')
            self.assertTrue(any(r.get('count')==1 for r in failed['joints']))

    def test_hoop_core_bar_outside_longitudinal_envelope_is_not_counted(self):
        from Design.SMRF_Transitions import bar_coordinates
        with gf.frame():
            state=gf.uniform_state();mg.install(state)
            core=next(m for m in state.members() if m.is_column).design
            hi=max(b['x_in'] for b in bar_coordinates(core,sp.COL_CLEAR_COVER_IN))-sp.rebar_diameter(core.bar_size)/2-.25
            placement={'floors':{str(f):{'layers':{axis+'_bottom':{'db_in':.5,'phase_in':0.,
                'global_transverse_coordinates_in':[j*(sp.BAY_Y if axis=='x' else sp.BAY_X)+d
                 for j in range((sp.NUM_BAY_Y if axis=='x' else sp.NUM_BAY_X)+1) for d in (0.,hi+.01)]}
                for axis in ('x','y')}} for f in range(1,sp.NUM_FLOOR+1)}}
            result=cs.physical_integrity(state,placement)
            self.assertEqual(result['status'],'fail')
            self.assertTrue(all(r['count']==1 for r in result['joints']))

    def test_computed_contract_replaces_only_old_assertion_structural_rows(self):
        with gf.frame():
            state=gf.uniform_state();cfg=DesignConfig()
            cfg.floor_analysis.common_action_procedure={'kind':'computed_common_actions_v1','status':'pending'}
            inventory=cc.expected_inventory(state,cfg)
            required={tuple(k) for row in inventory['conditional'] for k in row['requires']}
            self.assertNotIn(('slab','slab_two_way_shear_applicability',''),required)
            self.assertNotIn(('slab','slab_column_local_minimum_steel',''),required)
            self.assertIn(('common_actions','common.disposition.two_way_shear_path',''),inventory['exact'])
            self.assertIn(('common_actions','common.disposition.local_slab_column_steel',''),inventory['exact'])
            self.assertIn(('slab','slab_integrity_bottom_bars_through_column',''),required)
            self.assertIn(('slab','slab_computed_corner_reinforcement','x_top'),required)

    def test_computed_replay_dispatch_never_calls_legacy_selector(self):
        record={'method_version':cs.METHOD,'checks':[]}
        with patch.object(cs,'replay',return_value=record),patch.object(sr,'design_slab_reinforcement',side_effect=AssertionError('legacy gate')):
            result=sr.evaluate_slab_reinforcement(record)
            self.assertEqual(result[0]['status'],'pass')

    def test_computed_eligibility_dispatch_never_fakes_legacy_flags(self):
        record={'method_version':cs.METHOD}
        with patch.object(cs,'eligibility',return_value={'credit':'not_established'}) as called:
            self.assertEqual(sr.slab_strength_eligibility(record),{'credit':'not_established'})
            called.assert_called_once_with(record,None)


if __name__=='__main__':unittest.main()
