import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Generated_Sections import beam_section_from_design, column_section_from_design
from Design.SMRF_Cracked_Section import cracked_flexural_rigidity
from Design.SMRF_Section_Interaction import solve_section_at_axial_force, section_state


def record():
    r = dict(cover_basis='clear_cover_outside_hoops')
    for prefix, n, side in (('beam', 4, 1), ('col', 5, 3)):
        r.update({f'{prefix}_bar_size': 8, f'{prefix}_stirrup_bar_size': 4,
                  f'{prefix}_clear_cover_in': 1.5, f'{prefix}_longitudinal_centroid_offset_in': 2.5,
                  f'{prefix}_top_bars': n, f'{prefix}_bot_bars': n, f'{prefix}_side_bars': side,
                  f'{prefix}_bar_diameter_in': 1., f'{prefix}_bar_area_in2': .79,
                  f'{prefix}_stirrup_diameter_in': .5})
    return dict(geometry=dict(bay_x_in=240., bay_y_in=300.),
                sections=dict(b_beam_in=14., h_beam_in=28., fc_beam_ksi=4.,
                              b_col_in=24., h_col_in=30., fc_col_ksi=5.),
                reinforcement=r, materials=dict(fy_ksi=60.),
                slab=dict(thickness_in=6., concrete_fc_ksi=4.), slab_reinforcement=dict(layout=None))


def rectangle():
    return dict(id='analytic_singly_reinforced_rectangle', depth_in=24., fc_ksi=4., fy_ksi=60.,
                concrete_bands=[dict(top_in=0., bottom_in=24., width_in=12.)],
                steel_layers=[dict(id='bottom', depth_in=22., area_in2=2.)])


class GeneratedSectionTests(unittest.TestCase):
    def test_every_beam_and_column_bar_is_preserved(self):
        d = record(); before = copy.deepcopy(d)
        b = beam_section_from_design(d, axis='x', position='interior', scope='beam_rectangle')
        self.assertEqual(len(b['steel_layers']), 10)
        self.assertAlmostEqual(sum(v['area_in2'] for v in b['steel_layers']), 10*.79)
        self.assertEqual([v['depth_in'] for v in b['steel_layers'][-2:]], [14., 14.])
        for axis in ('h', 'b'):
            c = column_section_from_design(d, bending_depth=axis)
            self.assertEqual(len(c['steel_layers']), 16)
            self.assertAlmostEqual(sum(v['area_in2'] for v in c['steel_layers']), 16*.79)
        self.assertEqual(d, before)

    def test_column_other_axis_projects_actual_cage_instead_of_rotating_a_generic_layer(self):
        d = record()
        c = column_section_from_design(d, bending_depth='b')
        self.assertEqual(c['depth_in'], 24.)
        self.assertEqual(c['concrete_bands'][0]['width_in'], 30.)
        at_left = sum(v['area_in2'] for v in c['steel_layers'] if v['depth_in'] == 2.5)
        self.assertAlmostEqual(at_left, (2+3)*.79)

    def test_changed_hoop_changes_centroid_and_section_identity(self):
        d = record(); first = column_section_from_design(d, bending_depth='h')
        d['reinforcement'].update(col_stirrup_bar_size=5, col_stirrup_diameter_in=.625,
                                  col_longitudinal_centroid_offset_in=2.625)
        second = column_section_from_design(d, bending_depth='h')
        self.assertNotEqual(first['source_sha256'], second['source_sha256'])
        self.assertEqual(second['steel_layers'][0]['depth_in'], 2.625)

    def test_stale_cover_and_bar_area_are_refused(self):
        for key, value in (('beam_longitudinal_centroid_offset_in', 1.5), ('beam_bar_area_in2', .8),
                           ('cover_basis', 'legacy'), ('beam_side_bars', True)):
            d = record(); d['reinforcement'][key] = value
            with self.assertRaises(ValueError):
                beam_section_from_design(d, axis='x', position='interior', scope='beam_rectangle')

    def test_unknown_slab_is_not_silently_converted_to_no_slab(self):
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            beam_section_from_design(record(), axis='x', position='interior')
        s = beam_section_from_design(record(), axis='x', position='interior', slab_steel='omit_for_sensitivity')
        self.assertFalse(s['provenance']['composite_reinforcement_complete'])
        self.assertFalse(s['production_enabled'])
        self.assertFalse(s['development_assessed'])

    def test_actual_slab_mat_depths_areas_and_flange_not_counted_twice(self):
        d = record(); d['slab_reinforcement']['layout'] = dict(uniform_all_panels_and_floors=True,
            layers={f'x_{face}': dict(bar_area_in2=.20, bar_diameter_in=.5, spacing_in=8., effective_depth_in=5.) for face in ('top','bottom')})
        s = beam_section_from_design(d, axis='x', position='interior')
        bf = 14.+2*(240.-30.)/8
        self.assertEqual(s['provenance']['effective_flange_width_in'], bf)
        self.assertEqual([v['depth_in'] for v in s['steel_layers'][-2:]], [1., 5.])
        self.assertAlmostEqual(s['provenance']['slab_steel_area_in2'], 2*.20*bf/8)
        self.assertAlmostEqual(sum((v['bottom_in']-v['top_in'])*v['width_in'] for v in s['concrete_bands']), bf*6+14*22)
        self.assertTrue(s['provenance']['composite_reinforcement_complete'])
        self.assertFalse(s['development_assessed'])

    def test_mixed_concrete_strengths_not_silently_homogenized(self):
        d = record(); d['slab']['concrete_fc_ksi'] = 5.
        with self.assertRaisesRegex(ValueError, 'multi-material'):
            beam_section_from_design(d, axis='x', position='edge', slab_steel='omit_for_sensitivity')

    def test_structurepoint_published_t_beam_crossing_flange(self):
        # StructurePoint Aug 10 2021 Case Two (4.4.2), pp1-3, Table1 p12.
        # Published centroid d=36, As=12.48, bf=30, bw=14, tf=7, h=40.
        # Lump all yielded tension bars at their stated centroid for Mn only;
        # this fixture does not infer extreme-layer strain or independently phi.
        s = dict(id='SP_T_case2', depth_in=40., fc_ksi=4., fy_ksi=60.,
                 concrete_bands=[dict(top_in=0., bottom_in=7., width_in=30.),
                                 dict(top_in=7., bottom_in=40., width_in=14.)],
                 steel_layers=[dict(id='tension_centroid', depth_in=36., area_in2=12.48)])
        r = solve_section_at_axial_force(s, 0., compression_face='top', reference_y_in=20.)
        self.assertAlmostEqual(r['stress_block_depth_in'], 7.73, delta=.01)
        self.assertAlmostEqual(r['moment_sagging_kip_in']/12, 2017., delta=.5)
        self.assertAlmostEqual(.9*r['moment_sagging_kip_in']/12, 1815.1, delta=.3)


class CrackedSectionTests(unittest.TestCase):
    def test_singly_reinforced_closed_form_independent_reference(self):
        ec=3600.; es=29000.; n=es/ec; As=2.; b=12.; d=22.
        c=(-n*As+math.sqrt((n*As)**2+2*b*n*As*d))/b
        I=b*c**3/3+n*As*(d-c)**2
        r=cracked_flexural_rigidity(rectangle(),compression_face='top',reference_y_in=12.,ec_ksi=ec)
        self.assertAlmostEqual(r['neutral_axis_from_top_in'],c,places=10)
        self.assertAlmostEqual(r['i_cracked_transformed_in4'],I,places=8)

    def test_reference_shift_preserves_ei_and_transforms_coupling(self):
        a=cracked_flexural_rigidity(rectangle(),compression_face='top',reference_y_in=0.)
        b=cracked_flexural_rigidity(rectangle(),compression_face='top',reference_y_in=12.)
        A,B=a['tangent_matrix'][0]; D=a['tangent_matrix'][1][1]
        self.assertAlmostEqual(b['tangent_matrix'][0][1],B-12*A)
        self.assertAlmostEqual(b['tangent_matrix'][1][1],D-24*B+144*A)
        self.assertAlmostEqual(a['ei_constant_axial_force_kip_in2'],b['ei_constant_axial_force_kip_in2'])
        self.assertGreater(b['tangent_matrix'][1][1],b['ei_constant_axial_force_kip_in2'])

    def test_symmetric_rectangle_and_physical_t_asymmetry(self):
        d=record(); r=beam_section_from_design(d,axis='x',position='interior',scope='beam_rectangle')
        t=beam_section_from_design(d,axis='x',position='interior',slab_steel='omit_for_sensitivity')
        values=lambda s: [cracked_flexural_rigidity(s,compression_face=f,reference_y_in=14.)['ei_constant_axial_force_kip_in2'] for f in ('top','bottom')]
        ra,rb=values(r); ta,tb=values(t)
        self.assertAlmostEqual(ra,rb)
        self.assertGreater(ta,tb)
        self.assertAlmostEqual(rb,tb)  # Flange is wholly cracked in this negative-bending fixture.

    def test_compression_steel_displaces_concrete_at_its_own_strain(self):
        s=rectangle(); s['steel_layers'].append(dict(id='compression',depth_in=2.,area_in2=1.))
        ec=3600.;es=29000.;n=es/ec
        c=(-(n*2+(n-1))+math.sqrt((n*2+(n-1))**2+24*(n*2*22+(n-1)*2)))/12
        expected=12*c**3/3+n*2*(22-c)**2+(n-1)*(c-2)**2
        r=cracked_flexural_rigidity(s,compression_face='top',reference_y_in=12.,ec_ksi=ec)
        self.assertAlmostEqual(r['neutral_axis_from_top_in'],c,places=10)
        self.assertAlmostEqual(r['i_cracked_transformed_in4'],expected,places=8)

    def test_matrix_at_neutral_axis_has_no_axial_bending_coupling(self):
        a=cracked_flexural_rigidity(rectangle(),compression_face='bottom',reference_y_in=12.)
        b=cracked_flexural_rigidity(rectangle(),compression_face='bottom',reference_y_in=a['neutral_axis_from_top_in'])
        self.assertAlmostEqual(b['tangent_matrix'][0][1],0.,delta=1e-7)
        self.assertGreater(b['ei_constant_axial_force_kip_in2'],0.)

    def test_bad_modulus_and_face_are_rejected(self):
        for kwargs in (dict(ec_ksi=-1),dict(es_ksi=1),dict(compression_face='global_y')):
            args=dict(compression_face='top',reference_y_in=12.);args.update(kwargs)
            with self.assertRaises(ValueError):cracked_flexural_rigidity(rectangle(),**args)


class CircularDisplacementTests(unittest.TestCase):
    def test_half_circle_has_correct_area_and_centroid(self):
        s=rectangle();s['steel_layers']=[dict(id='bar',depth_in=3.,area_in2=math.pi*.5**2,bar_diameter_in=1.)]
        r=section_state(s,3./.85,compression_face='top',reference_y_in=12.,displaced_concrete_mode='circular')
        v=r['steel_components'][0]
        self.assertAlmostEqual(v['displaced_concrete_force_kip'],.85*4*math.pi*.5**2/2)
        self.assertAlmostEqual(v['displaced_concrete_centroid_in'],3.-4*.5/(3*math.pi))
        expected=v['steel_force_compression_kip']*(12-3)-v['displaced_concrete_force_kip']*(12-v['displaced_concrete_centroid_in'])
        self.assertAlmostEqual(v['moment_sagging_kip_in'],expected)

    def test_force_is_continuous_when_block_crosses_bar_centroid(self):
        s=rectangle();s['steel_layers'].append(dict(id='upper',depth_in=2.,area_in2=1.,bar_diameter_in=1.128))
        s['steel_layers'][0]['bar_diameter_in']=1.128
        values=[section_state(s,(2+delta)/.85,compression_face='top',reference_y_in=12.,
                              displaced_concrete_mode='circular')['axial_tension_kip'] for delta in (-1e-9,1e-9)]
        self.assertLess(abs(values[1]-values[0]),1e-5)
        self.assertLess(values[1],values[0])

    def test_old_ambiguous_rectangle_now_has_one_balanced_root(self):
        s=dict(id='SP_rectangle',depth_in=16.,fc_ksi=5.,fy_ksi=60.,
               concrete_bands=[dict(top_in=0.,bottom_in=16.,width_in=16.)],
               steel_layers=[dict(id=f'layer_{i}',depth_in=y,area_in2=4.,bar_diameter_in=1.128) for i,y in enumerate((2.5,13.5))])
        with self.assertRaisesRegex(ValueError,'2 admissible'):
            solve_section_at_axial_force(s,10.,compression_face='top',reference_y_in=8.)
        r=solve_section_at_axial_force(s,10.,compression_face='top',reference_y_in=8.,displaced_concrete_mode='circular')
        self.assertLess(abs(r['axial_equilibrium_error_kip']),1e-7)

    def test_round_mode_refuses_missing_diameter_and_excess_displaced_width(self):
        for diameter in (None,.01):
            s=rectangle()
            if diameter is not None:s['steel_layers'][0]['bar_diameter_in']=diameter
            with self.assertRaises(ValueError):
                section_state(s,3.,compression_face='top',reference_y_in=12.,displaced_concrete_mode='circular')

    def test_reference_shift_and_reflection_preserve_round_bar_moments(self):
        s=rectangle();s['steel_layers'][0]['bar_diameter_in']=1.128
        for ref in (0.,12.):
            a=solve_section_at_axial_force(s,20.,compression_face='top',reference_y_in=ref,displaced_concrete_mode='circular')
            reflected=copy.deepcopy(s)
            for bar in reflected['steel_layers']:bar['depth_in']=24.-bar['depth_in']
            b=solve_section_at_axial_force(reflected,20.,compression_face='bottom',reference_y_in=24-ref,displaced_concrete_mode='circular')
            self.assertAlmostEqual(a['moment_sagging_kip_in'],-b['moment_sagging_kip_in'],places=6)


if __name__=='__main__': unittest.main()



class StaggeredBeamRowsTests(unittest.TestCase):
    """The generated beam section places each direction's rows at the record's staggered elevations."""

    def test_rows_follow_the_record_layers_and_keep_every_bar(self):
        d = record()
        d['reinforcement']['beam_bar_stacking'] = {
            'layers': {'x': {'top': {'per_layer': [3, 1], 'offsets_in': [2.5, 4.5]},
                             'bottom': {'per_layer': [4], 'offsets_in': [2.5]}},
                       'y': {'top': {'per_layer': [4], 'offsets_in': [2.5]},
                             'bottom': {'per_layer': [3, 1], 'offsets_in': [2.5, 4.5]}}}}
        x = beam_section_from_design(d, axis='x', position='interior', scope='beam_rectangle')
        depths = sorted(round(l['depth_in'], 2) for l in x['steel_layers'] if l['id'].startswith('top'))
        self.assertEqual(depths, [2.5, 2.5, 2.5, 4.5])
        bottoms = sorted(round(l['depth_in'], 2) for l in x['steel_layers'] if l['id'].startswith('bottom'))
        self.assertEqual(bottoms, [25.5] * 4)
        self.assertAlmostEqual(sum(l['area_in2'] for l in x['steel_layers']), 8 * .79 + 2 * .79)
        y = beam_section_from_design(d, axis='y', position='interior', scope='beam_rectangle')
        self.assertEqual(sorted(round(l['depth_in'], 2) for l in y['steel_layers'] if l['id'].startswith('bottom')), [23.5, 25.5, 25.5, 25.5])
        d['reinforcement']['beam_bar_stacking']['layers']['x']['top']['per_layer'] = [2, 1]
        with self.assertRaisesRegex(ValueError, 'not the 4'):
            beam_section_from_design(d, axis='x', position='interior', scope='beam_rectangle')
        # Codex item 3: unequal arrays are rejected, never zipped down to fewer bars
        d['reinforcement']['beam_bar_stacking']['layers']['x']['top'] = {'per_layer': [3, 1], 'offsets_in': [2.5]}
        with self.assertRaisesRegex(ValueError, 'must match'):
            beam_section_from_design(d, axis='x', position='interior', scope='beam_rectangle')

