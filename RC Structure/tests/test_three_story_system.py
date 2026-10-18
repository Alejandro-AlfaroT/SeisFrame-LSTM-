"""Three-story systems preserve lateral/gravity classification and enforce both dimensions at every boundary."""
import sys,unittest,copy
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import Structure_Parameters as sp
from Model import Member_Groups as mg
from Design import Grouped_Search as gs, Group_Selection as sel, Perimeter_Systems as systems
from Design.Section_Design import COLUMN_SIZE_CAP_IN,COLUMN_SIZES_IN
from Design.Grouped_Runner import DEFAULT_GROUPING

POLICY=mg.SYSTEM_BANDED_POLICY_ID

class ThreeStorySystems(unittest.TestCase):
    def setUp(self):
        self.frame=gf.frame(NUM_FLOOR=9)
        self.frame.__enter__();self.addCleanup(self.frame.__exit__,None,None,None)
        self.state=mg.GroupedDesign.uniform(gf.NX,gf.NY,9,gf.COLUMN,gf.BEAM,policy_id=POLICY)

    def tied(self,state,band,family,**kwargs):
        tie=f'{band}__{family}';d=state.tie_designs()[tie]
        return state.with_tied_designs({tie:replace(d,**kwargs)})

    def test_three_story_boundaries_and_short_final_band(self):
        self.assertEqual(mg.bands_of(POLICY,9),[[1,2,3],[4,5,6],[7,8,9]])
        self.assertEqual(mg.bands_of(POLICY,4),[[1,2,3],[4]])
        self.assertEqual(mg.bands_of(POLICY,8),[[1,2,3],[4,5,6],[7,8]])
        self.assertEqual(len(self.state.groups),24)
        self.assertEqual(len(self.state.ties()),12)
        self.assertEqual(DEFAULT_GROUPING,POLICY)

    def test_cap_ladder_reaches_40_but_not_42(self):
        self.assertEqual(COLUMN_SIZE_CAP_IN,40.)
        self.assertEqual(COLUMN_SIZES_IN[-3:],(36.,38.,40.))
        good=self.state
        for band in ('s01_03','s04_06','s07_09'):
            good=self.tied(good,band,'sfrs_column',b_in=40.,h_in=40.)
        mg.install(good)
        bad=self.tied(self.state,'s01_03','sfrs_column',b_in=42.,h_in=42.)
        with self.assertRaisesRegex(mg.GroupedStateError,'cap'):mg.install(bad)

    def test_perimeter_physics_and_gravity_classification_stay_active(self):
        self.assertTrue(systems.two_system(self.state))
        self.assertEqual(systems.design_model_for(self.state),systems.RELEASED_MODEL)
        with mg.installed(self.state):
            edge=mg.beam_at('beam_x',1,0,0)
            inward=mg.beam_at('beam_x',1,0,1)
            self.assertEqual(self.state.system_of(edge.group_id),mg.SYSTEM_SFRS)
            self.assertEqual(self.state.system_of(inward.group_id),mg.SYSTEM_GRAVITY)
        self.assertEqual(len(sel._beam_units(self.state)),6)
        self.assertEqual(len(sel._column_units(self.state)),2)
        self.assertEqual(len(sel._column_chain(self.state,mg.COLUMN_LOCATIONS[:3])),3)

    def test_valid_taper_and_independent_cages_round_trip(self):
        s=self.tied(self.state,'s01_03','sfrs_column',b_in=30.,h_in=30.)
        s=self.tied(s,'s04_06','sfrs_column',b_in=28.,h_in=28.,top_bars=5,bot_bars=5)
        mg.install(s)
        self.assertEqual(mg.GroupedDesign.from_record(s.to_record()).identity(),s.identity())
        self.assertNotEqual(s.tie_designs()['s01_03__sfrs_column'],s.tie_designs()['s04_06__sfrs_column'])

    def test_upper_column_growth_is_refused_before_install_and_read(self):
        mg.install(self.state)
        bad=self.tied(self.state,'s04_06','sfrs_column',b_in=28.,h_in=28.)
        with self.assertRaisesRegex(mg.GroupedStateError,'below'):mg.install(bad)
        self.assertIs(mg.active(),self.state)
        # This is a canonically hashed, untampered record; the geometry rule itself refuses it.
        with self.assertRaisesRegex(mg.GroupedStateError,'below'):mg.GroupedDesign.from_record(bad.to_record())

    def test_beam_width_and_depth_are_checked_separately_not_by_area(self):
        for kwargs in ({'b_in':20.,'h_in':24.},{'b_in':16.,'h_in':30.}):
            s=self.tied(self.state,'s04_06','sfrs_beam',**kwargs)
            self.assertLess(kwargs['b_in']*kwargs['h_in'],gf.BEAM.b_in*gf.BEAM.h_in)
            with self.assertRaisesRegex(mg.GroupedStateError,'upward'):mg.install(s)
            with self.assertRaisesRegex(mg.GroupedStateError,'upward'):mg.GroupedDesign.from_record(s.to_record())

    def test_growth_propagates_down_and_retains_transition_steps(self):
        s=self.tied(self.state,'s07_09','sfrs_column',b_in=34.,h_in=34.)
        fixed,changes=gs.enforce_column_taper(s)
        self.assertTrue(changes);self.assertEqual(mg.section_order_problems(fixed),[])
        self.assertEqual([fixed.tie_designs()[f'{b}__sfrs_column'].b_in for b in ('s01_03','s04_06','s07_09')],[34.,34.,34.])
        s=self.tied(self.state,'s01_03','sfrs_column',b_in=36.,h_in=36.)
        fixed,_=gs.enforce_column_taper(s)
        self.assertEqual([fixed.tie_designs()[f'{b}__sfrs_column'].b_in for b in ('s01_03','s04_06','s07_09')],[36.,32.,28.])
        mg.install(fixed)

    def test_beam_growth_and_system_coupling_settle_together(self):
        s=self.tied(self.state,'s07_09','gravity_beam',b_in=22.,h_in=32.)
        fixed,changes=gs.enforce_column_taper(s)
        self.assertTrue(changes);mg.install(fixed)
        for band in ('s01_03','s04_06','s07_09'):
            for family in ('sfrs_beam','gravity_beam'):
                d=fixed.tie_designs()[f'{band}__{family}']
                self.assertGreaterEqual(d.b_in,22.);self.assertGreaterEqual(d.h_in,32.)

    def test_reduction_cannot_undercut_the_section_above(self):
        gid='s01_03__beam_x__edge'
        self.assertIn('upward',gs.beam_size_problem(self.state,gid,(16.,24.)))
        self.assertIn('upward',gs.column_taper_problem(self.state,'s01_03__column__corner',(24.,24.)))
        order=gs.reduction_order(self.state)
        self.assertEqual(len(order),12)
        self.assertTrue(all(g.startswith('s07_09') for g in order[:4]))

    def test_search_checks_returned_states_independently_of_evaluator(self):
        from Design.Config import DesignConfig
        bad=self.tied(self.state,'s04_06','sfrs_column',b_in=28.,h_in=28.)
        with patch.object(gs.gd,'evaluate_candidate',return_value={'state':bad,'status':'evaluated','feasible':True}):
            with self.assertRaisesRegex(mg.GroupedStateError,'upward'):
                gs.search(self.state,DesignConfig.from_structure_parameters(),verbose=False)

    def test_failed_tied_upper_column_chain_targets_the_right_band(self):
        s=gs.regraded(self.state,'column',10.)
        gids=s.ties()['s04_06__sfrs_column']
        constraints={'exhausted_groups':gids,'exhausted_column_lines':{
            'sfrs_column':{'blocked_at':'+'.join(gids),'blocked_by':'no admissible cage'}},
            'capacity_design_failed_checks':[],'joint_checks_failed':[],'drift_failed_checks':[],
            'joint_scwb':{'all_pass':True}}
        moves=gs.propose_growth({'status':'evaluated','constraints':constraints},s)
        self.assertIn('s04_06__sfrs_column',moves)
        self.assertNotIn('s01_03__sfrs_column',moves)


if __name__=='__main__':unittest.main()
