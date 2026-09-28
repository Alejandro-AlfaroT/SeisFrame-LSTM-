"""Regression: selected hoops must be reflected in returned strength checks."""
import contextlib, copy, sys, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Design import Design_Driver as driver
from Design.Config import DesignConfig
from Design.ACI_Checks import check_beam_flexure_pos

class CageClosureTests(unittest.TestCase):
    def setUp(self):
        self.stack=contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for key in driver._STATE_KEYS:
            self.stack.enter_context(mock.patch.object(sp,key,copy.deepcopy(getattr(sp,key))))
        sp.SLAB_THICKNESS_IN=6.; sp.FLOOR_TRANSFER=None
        sp.B_BEAM=14.; sp.H_BEAM=28.; sp.FC_BEAM_KSI=4.
        sp.BEAM_BAR_SIZE=8; sp.BEAM_TOP_BARS=sp.BEAM_BOT_BARS=6
        sp.BEAM_STIRRUP_BAR_SIZE=4
        self.cfg=DesignConfig.from_structure_parameters(); driver._sync_cfg_to_sp(self.cfg)
        self.actions=[dict(id='D',analysis_succeeded=True,members={'1':dict(member_type='column',
            local_force_kip_kipin=[100.,0.,0.,0.,0.,0.,-100.,0.,0.,0.,0.,0.],axial_i_kip=100.,axial_j_kip=100.)})]

    def run_pass(self,budget):
        # Choose a demand that passes the #4-hoop cage and fails after #5
        # hoops move the longitudinal bars inward. No fake capacity formula.
        old=check_beam_flexure_pos(1.,self.cfg).capacity
        demand=old*.999
        def checks(*args,**kwargs):
            ls=check_beam_flexure_pos(demand,self.cfg)
            return .4,ls.dcr,{}
        def hoops(*args):
            sp.BEAM_STIRRUP_BAR_SIZE=5; driver._sync_cfg_to_sp(self.cfg)
        joint=dict(evaluated=True,all_pass=True,steel_raised=False,steel_exhausted=False)
        with mock.patch.object(driver,'_analyze_combination',return_value=None), \
             mock.patch.object(driver,'_capture_element_actions',return_value={}), \
             mock.patch.object(driver,'_apply_transverse_geometry'), \
             mock.patch.object(driver,'_governing_dcrs',side_effect=checks), \
             mock.patch.object(driver,'redesign_steel',return_value=(None,None,True,[])), \
             mock.patch.object(driver,'_capacity_design',side_effect=hoops), \
             mock.patch.object(driver,'_scwb_steel_floor',return_value=joint), \
             mock.patch.object(driver,'_joint_scwb_state',return_value=joint):
            worst,*_=driver._steel_pass(self.cfg,1.,max_steel_iter=budget)
        actual=check_beam_flexure_pos(demand,self.cfg)
        self.assertGreater(actual.dcr,1.)
        self.assertAlmostEqual(worst['beam'],actual.dcr,places=12)

    def test_hoop_only_change_cannot_return_previous_passing_dcr(self):
        self.run_pass(3)

    def test_final_budget_hoop_selection_is_rechecked(self):
        self.run_pass(1)

    def test_signature_tracks_hoops_and_directional_legs(self):
        before=driver._cage_signature()
        sp.COL_STIRRUP_LEGS_BY_DIRECTION={'across_b_face':8,'across_h_face':6}
        self.assertNotEqual(driver._cage_signature(),before)

    def test_capacity_evidence_is_rebuilt_on_installed_bar_positions(self):
        from Design import SMRF_Capacity_Design as cd
        covers=[]
        def build(state):
            covers.append(state['beam']['centroid_offset_in'])
            return dict(transverse=dict(beam=dict(bar_size=5,spacing_in=4.,legs=2),column=None),
                        evaluated_cover_in=state['beam']['centroid_offset_in'])
        # Use the real capacity-state geometry and the real installation code;
        # only the hoop selector is controlled to exercise a diameter change.
        with mock.patch.object(cd,'build_capacity_design',side_effect=build):
            result=driver._capacity_design(self.cfg,self.actions)
        self.assertEqual(len(covers),2)
        self.assertNotEqual(covers[0],covers[-1])
        self.assertEqual(result['evaluated_cover_in'],sp.longitudinal_cover_in('beam'))

    def test_capacity_selection_cycle_is_preserved_as_failure(self):
        from Design import SMRF_Capacity_Design as cd
        def build(state):
            next_bar=5 if sp.BEAM_STIRRUP_BAR_SIZE==4 else 4
            return dict(transverse=dict(beam=dict(bar_size=next_bar,spacing_in=4.,legs=2),column=None))
        with mock.patch.object(cd,'build_capacity_design',side_effect=build):
            with self.assertRaisesRegex(RuntimeError,'cycled'):
                driver._capacity_design(self.cfg,self.actions)

if __name__=='__main__': unittest.main()
