"""Independent N-M witnesses and production adapters, without structural solves."""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import numpy as np
from Design.SMRF_Beam_Axial import section_family, capacity_at_axial, installed_member_sections
from Design import Group_Checks as checks, Group_Selection as selection
from Design.Config import DesignConfig
from Model import Member_Groups as mg, Member_Bar_Layers as bar_layers, Member_Properties as mp
from test_shared_section_actions import independent_state


class BeamExternalAxial(unittest.TestCase):
    def family(self, **overrides):
        args = dict(depth_in=30., web_width_in=16., fc_ksi=5., fy_ksi=60., beam_bar_area_in2=.79,
                    bar_rows={f: [{'bars': 4, 'elevation_from_face_in': 2.5}] for f in ('top', 'bottom')},
                    slab_thickness_in=8., flange_width_in=48.,
                    slab_layers=[{'face': 'top', 'depth_in': 1., 'area_in2': 1.2},
                                 {'face': 'bottom', 'depth_in': 6.5, 'area_in2': .8}],
                    slab_bar_areas={'top': .2, 'bottom': .2})
        args.update(overrides)
        return section_family(**args)

    def test_axial_capacity_material_integral_and_end_development(self):
        sections = self.family(undeveloped_ends=('j',))
        ps = [-125., 0., 100.]
        result = capacity_at_axial(sections, ps)
        for i, p in enumerate(ps):
            for sign, face in [('positive', 'top'), ('negative', 'bottom')]:
                section = sections[result['section'][sign][i]]
                solution = section.at_axial(p, face)
                pn, mn = independent_state(section, float(solution['c']), face, 'auto')
                self.assertAlmostEqual(pn, p, places=5)
                self.assertAlmostEqual(abs(mn), result[f'{sign}_kip_in'][i], places=4)
        self.assertLess(result['negative_kip_in'][0], result['negative_kip_in'][1])
        self.assertLess(sections['j'].ast, sections['span'].ast)
        self.assertTrue(capacity_at_axial(sections, [-10000.])['domain_failure'][0])

    def test_both_axial_ends_survive_demand_capture(self):
        force = [0.]*12; force[0] = 17.; force[6] = 42.; force[1] = 9.
        actions = [{'id': 'C1', 'members': {'7': {'local_force_kip_kipin': force,
            'span_bending': {'full_span': {'mu_positive_kip_in': 2000., 'mu_negative_kip_in': 1200.}}}}}]
        self.assertEqual(checks.beam_demand_rows([7], actions), [(7, 'C1', 2000., 1200., 9., -42., 17.)])

    def test_installed_final_check_prices_axial_tension_and_preserves_nominal(self):
        beam = replace(gf.BEAM, top_bars=4, bot_bars=4)
        with gf.frame(beam=beam):
            state = gf.uniform_state(beam=beam)
            with mg.installed(state):
                gid = next(g for g, data in state.groups.items() if data['member_type'] == 'beam_x')
                tag = state.groups[gid]['member_tags'][0]
                rows = bar_layers.arrangement()[gid]
                before = copy.deepcopy(mp.beam_strengths(state.member(tag)))
                no_n = checks.beam_group_strength(state.designs[gid], [(tag, 'N0', 2000., 2000., 50., 0., 0.)], rows, state)
                tension = checks.beam_group_strength(state.designs[gid], [(tag, 'NT', 2000., 2000., 50., -100., -20.)], rows, state)
                self.assertGreater(tension['flexure_positive']['dcr'], no_n['flexure_positive']['dcr'])
                self.assertGreater(tension['shear']['dcr'], no_n['shear']['dcr'])
                self.assertEqual(tension['flexure_positive']['axial_kip'], -100.)
                self.assertEqual(mp.beam_strengths(state.member(tag)), before)
                outside = checks.beam_group_strength(state.designs[gid], [(tag, 'OUT', 1., 1., 1., -10000., 0.)], rows, state)
                self.assertTrue(outside['flexure_positive']['axial_domain_failure'])
                self.assertGreater(outside['flexure_positive']['dcr'], 1.)

    def test_selection_does_not_accept_zero_n_probe_for_axial_failure(self):
        beam = replace(gf.BEAM, top_bars=4, bot_bars=4)
        with gf.frame(slab=None, beam=beam):
            state = gf.uniform_state(beam=beam)
            with mg.installed(state):
                gid = next(g for g, data in state.groups.items() if data['member_type'] == 'beam_x')
                tags = checks.beam_group_representatives(state, gid)
                cfg = DesignConfig(); cfg.rebar.rho_beam_practical_max = .0175
                paired = {tag: (1., 1., -10000., 0.) for tag in tags}
                result = selection.evaluate_beam_candidate(state, gid, state.designs[gid], (1., 1.), cfg, tags, paired)
                self.assertFalse(result['admissible'])
                self.assertEqual(result['reason_code'], 'axial_domain')


if __name__ == '__main__':
    unittest.main()
