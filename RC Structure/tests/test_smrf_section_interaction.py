import copy,math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Section_Interaction import section_state,solve_section_at_axial_force,demand_from_global_cut
from Design.SMRF_Beam_Slab_Strength import section_moment


def rectangle(b=16.,h=16.,fc=5.,layers=((2.5,4.),(13.5,4.))):
    return dict(id='fixture_rectangle',fc_ksi=fc,fy_ksi=60.,depth_in=h,
        concrete_bands=[dict(top_in=0.,bottom_in=h,width_in=b)],
        steel_layers=[dict(id=f'layer_{i}',depth_in=y,area_in2=a) for i,(y,a) in enumerate(layers)])


class SectionInteractionTests(unittest.TestCase):
    def test_published_structurepoint_control_points(self):
        # Primary reference: StructurePoint May 2022, ACI 318-19 example,
        # printed Table 1 p26 (PDF p29). Column: 16x16, two layers of 4 #9,
        # y=2.5/13.5 in, fc=5 ksi, fy=60 ksi. Check all five flexural points.
        ey=60./29000.
        points=[(0.,.65,622.3,169.86),(.5*ey,.65,421.9,220.05),
                (ey,.65,270.9,250.77),(ey+.003,.9,171.6,286.75)]
        for strain,phi,P,M in points:
            with self.subTest(strain=strain):
                r=section_state(rectangle(),13.5*.003/(.003+strain),compression_face='top',reference_y_in=8.)
                self.assertAlmostEqual(-phi*r['axial_tension_kip'],P,delta=.15)
                self.assertAlmostEqual(phi*r['moment_sagging_kip_in']/12.,M,delta=.10)
        r=solve_section_at_axial_force(rectangle(),0.,compression_face='top',reference_y_in=8.)
        self.assertAlmostEqual(.9*r['moment_sagging_kip_in']/12.,213.91,delta=.08)

    def test_singly_reinforced_hand_equilibrium_under_compression_and_tension(self):
        sec=rectangle(b=12.,h=24.,fc=4.,layers=((22.,2.),))
        for N in (-60.,0.,60.):
            r=solve_section_at_axial_force(sec,N,compression_face='top',reference_y_in=12.)
            C=120.-N;a=C/(.85*4.*12.)
            expected=C*(12.-a/2)+120.*(22.-12.)
            self.assertAlmostEqual(r['moment_sagging_kip_in'],expected,places=6)
            self.assertAlmostEqual(r['neutral_axis_in'],a/.85,places=7)

    def test_symmetric_section_has_opposite_moments_at_same_axial_force(self):
        for N in (-400.,0.,100.):
            top=solve_section_at_axial_force(rectangle(),N,compression_face='top',reference_y_in=8.)
            bot=solve_section_at_axial_force(rectangle(),N,compression_face='bottom',reference_y_in=8.)
            self.assertAlmostEqual(top['moment_sagging_kip_in'],-bot['moment_sagging_kip_in'],places=6)

    def test_reference_translation_keeps_the_axial_moment_term(self):
        N=-200.;a=solve_section_at_axial_force(rectangle(),N,compression_face='top',reference_y_in=0.)
        b=solve_section_at_axial_force(rectangle(),N,compression_face='top',reference_y_in=8.)
        self.assertAlmostEqual(b['moment_sagging_kip_in'],a['moment_sagging_kip_in']-N*8.,places=6)

    def test_zero_axial_legacy_mode_matches_existing_rectangle_and_t_section(self):
        for bf in (14.,65.):
            sec=rectangle(b=14.,h=28.,fc=8.,layers=((2.5,2.4),(25.5,1.8)))
            sec['concrete_bands']=[dict(top_in=0.,bottom_in=6.,width_in=bf),dict(top_in=6.,bottom_in=28.,width_in=14.)]
            old=section_moment([(2.5,2.4),(25.5,1.8)],8.,60.,28.,14.,flange_width=bf,flange_depth=6.)
            new=solve_section_at_axial_force(sec,0.,compression_face='top',reference_y_in=3.,subtract_displaced_concrete=False)
            self.assertAlmostEqual(new['moment_sagging_kip_in'],old['mn_kip_in'],places=5)

    def test_concrete_band_integration_crosses_the_flange_boundary(self):
        sec=rectangle(b=14.,h=28.,fc=4.,layers=((25.,1.),))
        sec['concrete_bands']=[dict(top_in=0.,bottom_in=6.,width_in=65.),dict(top_in=6.,bottom_in=28.,width_in=14.)]
        r=section_state(sec,10.,compression_face='top',reference_y_in=3.)
        parts=r['concrete_components'];self.assertEqual(len(parts),2)
        self.assertAlmostEqual(sum(p['force_compression_kip'] for p in parts),.85*4*(65*6+14*2.5))
        self.assertAlmostEqual(sum(p['moment_sagging_kip_in'] for p in parts),.85*4*14*2.5*(3.-7.25))
        deep=section_state(sec,30.,compression_face='bottom',reference_y_in=3.)
        self.assertEqual(len(deep['concrete_components']),2)

    def test_steel_displacement_is_subtracted_once(self):
        with_subtraction=section_state(rectangle(),13.5,compression_face='top',reference_y_in=8.)
        gross=section_state(rectangle(),13.5,compression_face='top',reference_y_in=8.,subtract_displaced_concrete=False)
        self.assertAlmostEqual(with_subtraction['axial_tension_kip']-gross['axial_tension_kip'],.85*5*4)
        self.assertAlmostEqual(gross['moment_sagging_kip_in']-with_subtraction['moment_sagging_kip_in'],.85*5*4*(8-2.5))

    def test_ambiguous_lumped_steel_root_is_refused(self):
        with self.assertRaisesRegex(ValueError,'2 admissible'):
            solve_section_at_axial_force(rectangle(),10.,compression_face='top',reference_y_in=8.)

    def test_force_outside_branch_is_not_silently_clamped(self):
        for N in (1000.,-2000.):
            with self.assertRaisesRegex(ValueError,'outside'):
                solve_section_at_axial_force(rectangle(),N,compression_face='top',reference_y_in=8.)

    def test_pure_axial_plateaus_are_not_reported_as_unique_flexural_roots(self):
        for N in (480.,-1534.):
            with self.assertRaisesRegex(ValueError,'endpoints'):
                solve_section_at_axial_force(rectangle(),N,compression_face='top',reference_y_in=8.)

    def test_invalid_geometry_and_duplicate_steel_ids_are_refused(self):
        for field,value in (('fc_ksi',float('nan')),('fy_ksi',True),('depth_in',-1.)):
            sec=rectangle();sec[field]=value
            with self.assertRaises(ValueError):section_state(sec,5.,compression_face='top',reference_y_in=8.)
        sec=rectangle();sec['steel_layers'][1]['id']=sec['steel_layers'][0]['id']
        with self.assertRaisesRegex(ValueError,'unique'):section_state(sec,5.,compression_face='top',reference_y_in=8.)
        sec=rectangle();sec['concrete_bands'][0]['top_in']=1.
        with self.assertRaisesRegex(ValueError,'cover'):section_state(sec,5.,compression_face='top',reference_y_in=8.)

    def test_both_global_axes_and_opposite_faces_map_to_identical_section_actions(self):
        for axis,wrench in (('x',[25.,0.,0.,0.,-400.,0.]),('y',[0.,25.,0.,400.,0.,0.])):
            for sign in (-1,1):
                d=demand_from_global_cut([sign*v for v in wrench],axis=axis,outward_normal_sign=sign,
                    reference_xyz_in=[0.,0.,0.],section_top_z_in=3.,section_id='owned_section')
                self.assertEqual(d['axial_tension_kip'],25.);self.assertEqual(d['moment_sagging_kip_in'],400.)
                self.assertEqual(d['reference_y_in'],3.)

    def test_physical_t_section_asymmetry_survives_face_reversal(self):
        sec=rectangle(b=14.,h=28.,fc=4.,layers=((2.5,4.),(25.5,4.)))
        sec['concrete_bands']=[dict(top_in=0.,bottom_in=6.,width_in=65.),dict(top_in=6.,bottom_in=28.,width_in=14.)]
        a=solve_section_at_axial_force(sec,0.,compression_face='top',reference_y_in=3.)
        b=solve_section_at_axial_force(sec,0.,compression_face='bottom',reference_y_in=3.)
        self.assertGreater(a['moment_sagging_kip_in'],abs(b['moment_sagging_kip_in']))
        self.assertFalse(a['production_enabled']);self.assertFalse(a['engineering_verified'])


if __name__=='__main__':unittest.main(verbosity=2)
