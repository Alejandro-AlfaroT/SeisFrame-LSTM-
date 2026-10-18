"""Composite credit must enter selection and final checks with the same physical anchorage limits."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import Structure_Parameters as sp
from Model import Member_Groups as mg, Member_Properties as mp, Member_Bar_Layers as bl
from Design import Group_Checks as checks, Group_Selection as selection
from Design.Config import DesignConfig


def slab_layout(size=3, spacing=12.):
    return {"bar_size":size,"outer_axis":"x", "layers":{
        f"{axis}_{face}":{"bar_size":size,"bar_area_in2":sp.rebar_area(size),"spacing_in":spacing,
            "effective_depth_in":gf.SLAB-1.25 if face=="top" else 1.25,
            "axis":axis,"layer":"outer" if axis=="x" else "inner","clear_cover_outer_mat_in":1.}
        for axis in ("x","y") for face in ("top","bottom")}}


class CompositeGroupedStrength(unittest.TestCase):
    def setUp(self):
        self.frame=gf.frame(BEAM_BAR_STACKING="x_over_y")
        self.frame.__enter__()
        self.state=mg.install(gf.uniform_state())

    def tearDown(self):
        self.frame.__exit__(None,None,None)

    def beam(self, span=0, line=0):
        return mg.beam_at("beam_x",1,span,line)

    def strength(self, member):
        return checks.resolved_beam_flexural_strengths(member,bl.member_rows(member))

    def test_interior_T_and_perimeter_L_use_different_flange_widths(self):
        edge,interior=self.strength(self.beam()),self.strength(self.beam(line=1))
        self.assertEqual(edge['positive']['slab_sides'],1)
        self.assertEqual(interior['positive']['slab_sides'],2)
        self.assertGreater(interior['positive']['effective_flange_width_in'],edge['positive']['effective_flange_width_in'])

    def test_unreinforced_flange_concrete_adds_sagging_capacity(self):
        member=self.beam(); rows=bl.member_rows(member)
        old=checks.beam_flexural_strengths(member.design,rows)
        new=self.strength(member)
        self.assertGreater(new['positive']['phi_mn_kip_in'],old['positive']['phi_mn_kip_in'])
        self.assertEqual(new['basis']['slab_steel_in_flange_in2'],0.)

    def test_section_force_and_moment_balance_independently(self):
        # Reassemble the section result from the Whitney block and every returned steel force.
        member=self.beam(line=1)
        raw=mp.beam_strengths(member)['family']; sec=raw['composite']['positive']
        a=sec['stress_block_in']; bf=raw['effective_flange_width_in']; t=gf.SLAB; bw=member.design.b_in
        flange_depth=min(a,t); web_depth=max(0,a-t)
        cc=.85*member.design.fc_ksi*(bf*flange_depth+bw*web_depth)
        cc_m=.85*member.design.fc_ksi*(bf*flange_depth**2/2+bw*web_depth*(t+web_depth/2))
        self.assertAlmostEqual(cc+sum(force for _,force in sec['steel_forces_kip']),0.,places=8)
        self.assertAlmostEqual(-(cc_m+sum(y*force for y,force in sec['steel_forces_kip'])),sec['mn_kip_in'],places=8)

    def test_developed_mats_add_hogging_capacity(self):
        before=self.strength(self.beam())
        sp.SLAB_REINFORCEMENT={'layout':slab_layout()}
        after=self.strength(self.beam())
        self.assertTrue(after['basis']['exterior_anchorage']['developed'])
        self.assertGreater(after['negative']['phi_mn_kip_in'],before['negative']['phi_mn_kip_in'])

    def test_undeveloped_exterior_mats_are_excluded_but_flange_concrete_remains(self):
        sp.SLAB_REINFORCEMENT={'layout':slab_layout(8,5.)}
        member=self.beam(); result=self.strength(member); raw=mp.beam_strengths(member)['family']
        self.assertFalse(result['basis']['exterior_anchorage']['developed'])
        self.assertAlmostEqual(result['negative']['mn_kip_in'],raw['rectangular']['negative']['mn_kip_in'])
        self.assertAlmostEqual(result['positive']['mn_kip_in'],raw['undeveloped']['positive']['mn_kip_in'])
        self.assertGreater(raw['composite']['negative']['mn_kip_in'],result['negative']['mn_kip_in'])

    def test_no_slab_keeps_the_explicit_web_only_reference(self):
        sp.SLAB_THICKNESS_IN=None
        member=self.beam(); rows=bl.member_rows(member)
        self.assertEqual(self.strength(member),checks.beam_flexural_strengths(member.design,rows))

    def test_candidate_and_final_group_check_agree_with_same_cage_and_demands(self):
        sp.SLAB_REINFORCEMENT={'layout':slab_layout()}
        member=self.beam(); gid=member.group_id; rows=bl.member_rows(member); mu=(1000.,1100.)
        candidate=selection.evaluate_beam_candidate(self.state,gid,member.design,mu,DesignConfig())
        self.assertTrue(candidate['admissible'],candidate)
        demand_rows=[(tag,'synthetic',*mu,0.) for tag in self.state.groups[gid]['member_tags']]
        final=checks.beam_group_strength(member.design,demand_rows,rows,self.state)
        self.assertAlmostEqual(candidate['dcr'],max(final['flexure_positive']['dcr'],final['flexure_negative']['dcr']),places=12)
        self.assertEqual(final['strength_basis'],checks.BEAM_STRENGTH_BASIS)

    def test_final_check_uses_the_demanding_members_own_capacity(self):
        edge='s01_02__beam_y__edge'
        self.state=mg.install(self.state.with_designs({edge:replace(self.state.designs[edge],b_in=14.)}))
        sp.SLAB_REINFORCEMENT={'layout':slab_layout(8,12.)}
        member=self.beam(); group=self.state.groups[member.group_id]; rows=bl.member_rows(member)
        tags=group['member_tags'][:2]
        demand_rows=[(tags[0],'A',800.,1200.,0.),(tags[1],'B',1600.,700.,0.)]
        result=checks.beam_group_strength(member.design,demand_rows,rows,self.state)
        expected=max((mu/self.strength(self.state.member(tag))['negative']['phi_mn_kip_in'],tag)
                     for tag,_,_,mu,_ in demand_rows)
        self.assertAlmostEqual(result['flexure_negative']['dcr'],expected[0],places=12)
        self.assertEqual(result['flexure_negative']['member_tag'],expected[1])

    def test_slab_layout_change_is_not_hidden_by_strength_cache(self):
        member=self.beam()
        first=self.strength(member)
        sp.SLAB_REINFORCEMENT={'layout':slab_layout()}
        second=self.strength(member)
        self.assertNotEqual(first['negative']['phi_mn_kip_in'],second['negative']['phi_mn_kip_in'])

    def test_detailing_uses_nominal_composite_end_strengths_not_web_or_phi_values(self):
        sp.SLAB_REINFORCEMENT={'layout':slab_layout(8,5.)}
        member=self.beam(); raw=mp.beam_strengths(member)['family']; rows=bl.member_rows(member)
        detail=checks.beam_detailing_strengths(member,rows)
        self.assertAlmostEqual(detail['mn_negative_left_kipin'],raw['rectangular']['negative']['mn_kip_in'])
        self.assertAlmostEqual(detail['mn_negative_right_kipin'],raw['composite']['negative']['mn_kip_in'])
        self.assertAlmostEqual(detail['mn_positive_min_along_kipin'],min(raw['undeveloped']['positive']['mn_kip_in'],
                                                                     raw['composite']['positive']['mn_kip_in']))

    def test_excess_composite_tension_steel_is_rejected_even_at_small_demand(self):
        sp.SLAB_REINFORCEMENT={'layout':slab_layout(8,3.)}
        member=self.beam(); rows=bl.member_rows(member)
        result=selection.evaluate_beam_candidate(self.state,member.group_id,member.design,(1.,1.),DesignConfig())
        self.assertFalse(result['admissible'])
        self.assertIn('tension controlled',result['reason'])
        final=checks.beam_group_strength(member.design,[(member.member_tag,'tiny',1.,1.,0.)],rows,self.state)
        self.assertGreater(final['flexure_negative']['dcr'],1.)
        self.assertFalse(final['flexure_negative']['tension_controlled'])


if __name__=='__main__':unittest.main()
