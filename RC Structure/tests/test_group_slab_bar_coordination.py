"""Slab mats are physical obstacles, not a plotting correction or a waived clash check."""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import Structure_Parameters as sp
from Model import Member_Groups as mg, Member_Bar_Layers as bl, Member_Properties as mp
from Design import Group_Capacity as gc, Grouped_Design as gd


def layout(t=7.5, size=4, cover=.75):
    db=sp.rebar_diameter(size)
    return {"outer_axis":"x", "layers":{
        f"{axis}_{face}": {"axis":axis, "layer":"outer" if axis=="x" else "inner",
            "bar_size":size, "clear_cover_outer_mat_in":cover, "bar_area_in2":sp.rebar_area(size),
            "spacing_in":12., "effective_depth_in":t-cover-db/2-(0 if axis=="x" else db)}
        for axis in ("x","y") for face in ("top","bottom")}}


class FixedMats(unittest.TestCase):
    def setUp(self):
        self.column=replace(gf.COLUMN,b_in=30.,h_in=30.,fc_ksi=10.,bar_size=11,
                            top_bars=6,bot_bars=6,side_bars=4)
        self.beam=replace(gf.BEAM,b_in=26.,h_in=36.,fc_ksi=8.,bar_size=9,top_bars=4,bot_bars=4)
        self.frame=gf.frame(slab=7.5,column=self.column,beam=self.beam,BEAM_BAR_STACKING="y_over_x")
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__,None,None,None)
        self.state=mg.install(gf.uniform_state(self.column,self.beam))
        self.bare=copy.deepcopy(bl.arrangement())
        sp.SLAB_REINFORCEMENT={"layout":layout()}

    def test_saved_collision_reproduction_and_real_clearance_after_placement(self):
        gid="s01_02__beam_x__edge"
        old=self.bare[gid]['top']['offsets_in'][0]
        self.assertEqual(old,5.5)
        # Independent cylinder geometry: y bottom mat at 6 in, No.4 x No.9.
        self.assertAlmostEqual((.5+1.128)/2-abs(6.-old),.314)
        rows=bl.arrangement()
        self.assertEqual(rows[gid]['top']['offsets_in'],[8.,13.])
        for g,group in self.state.groups.items():
            if group['member_type']=='column':continue
            axis=group['member_type'][-1]; db=sp.rebar_diameter(self.state.designs[g].bar_size)
            # Independently derived mat centres; does not call bl.slab_mats or the clash checker.
            planes=[(1.,'x'),(6.5,'x'),(1.5,'y'),(6.,'y')]
            for offset in rows[g]['top']['offsets_in']:
                for depth,direction in planes:
                    if direction!=axis:
                        self.assertGreaterEqual(abs(depth-offset)-(.5+db)/2,1.-1e-9)
        for kind in gc.joint_inventory()['types'].values():
            check=gc.slab_mat_clashes(kind['representative'])
            self.assertTrue(check['passes'])
            self.assertEqual(check['tight_crossings'],[])

    def test_mats_sections_and_bar_counts_are_not_changed(self):
        saved=copy.deepcopy(sp.SLAB_REINFORCEMENT); identity=self.state.identity()
        rows=bl.arrangement()
        self.assertEqual(saved,sp.SLAB_REINFORCEMENT)
        self.assertEqual(identity,self.state.identity())
        for gid in self.bare:
            if gid=='_basis':continue
            self.assertEqual(self.bare[gid]['bottom'],rows[gid]['bottom'])
            for face in ('top','bottom'):
                self.assertEqual(self.bare[gid][face]['per_layer'],rows[gid][face]['per_layer'])

    def test_crossing_beams_still_clear_and_group_rows_are_continuous(self):
        rows=bl.arrangement()
        for kind in bl.joint_types():
            x,y=kind['x'],kind['y']
            if not x or not y:continue
            for a in rows[x.group_id]['top']['offsets_in']:
                for b in rows[y.group_id]['top']['offsets_in']:
                    self.assertGreaterEqual(abs(a-b),bl.pitch_in(x.design,y.design)-1e-9)
            for member in (x,y):self.assertEqual(bl.member_rows(member),rows[member.group_id])

    def test_cache_tracks_cover_bar_size_and_slab_thickness(self):
        first=bl.arrangement()
        sp.SLAB_REINFORCEMENT['layout']=layout(cover=1.25)
        self.assertNotEqual(first,bl.arrangement())
        sp.SLAB_REINFORCEMENT['layout']=layout(size=5)
        self.assertNotEqual(first,bl.arrangement())
        sp.SLAB_THICKNESS_IN=8.5
        sp.SLAB_REINFORCEMENT['layout']=layout(t=8.5)
        self.assertNotEqual(first,bl.arrangement())

    def test_strength_reads_the_placed_rows_and_loses_web_lever_arm(self):
        member=mg.beam_at('beam_x',1,0,0)
        rows=bl.member_rows(member)
        actual=mp.beam_strengths(member)
        self.assertEqual(actual['bar_rows'],rows)
        from Design.Group_Checks import beam_flexural_strengths
        old=beam_flexural_strengths(member.design,self.bare[member.group_id])
        new=beam_flexural_strengths(member.design,rows)
        self.assertLess(new['negative']['phi_mn_kip_in'],old['negative']['phi_mn_kip_in'])

    def test_too_shallow_section_is_still_refused(self):
        changes={gid:replace(d,h_in=20.) for gid,d in self.state.designs.items() if not d.is_column}
        shallow=self.state.with_designs(changes)
        self.assertTrue(gd.invalid_beam_row_groups(shallow))

    def test_no_layout_retains_original_rows(self):
        sp.SLAB_REINFORCEMENT=None
        self.assertEqual(self.bare,bl.arrangement())

    def test_fixed_obstacle_can_force_recheck_of_an_earlier_crossing(self):
        d=self.beam
        offsets={('x',0):3.,('y',0):3.}
        bl._place([('x',0,d),('y',0,d)],offsets,{'x':0.,'y':0.}, {'x':[(3.,.5)]})
        self.assertGreaterEqual(offsets[('x',0)],5.)
        self.assertGreaterEqual(abs(offsets[('x',0)]-offsets[('y',0)]),bl.pitch_in(d,d))


if __name__=='__main__':unittest.main()
