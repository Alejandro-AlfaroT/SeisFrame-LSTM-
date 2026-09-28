"""Geometry/load hand checks and native-versus-condensed perimeter evidence."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import openseespy.opensees as ops
from Design.SMRF_Floor_Mesh import coupled_floor_mesh
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity
from Design.SMRF_Floor_Compatibility import CompatibleFloor
from Design.SMRF_Composite_Sections import recover_floor_cut
from Design.SMRF_Cut_Regions import recover_floor_regions
from tests.test_smrf_coupled_analysis import SLAB,GEOMETRY,SECTIONS,DEAD,ZERO

FACE='beam_outer_faces'


class PerimeterTests(unittest.TestCase):
    def setUp(self):
        ops.wipe()
        self.addCleanup(ops.wipe)

    def test_geometry_hand_area_preserves_every_original_coordinate(self):
        for mesh in (2,4,8):
            before=coupled_floor_mesh(2,6,240.,240.,mesh,beam_width_in=16.)
            after=coupled_floor_mesh(2,6,240.,240.,mesh,beam_width_in=16.,slab_perimeter=FACE)
            self.assertEqual(after['x_coordinates_in'][1:-1],before['x_coordinates_in'])
            self.assertEqual(after['y_coordinates_in'][1:-1],before['y_coordinates_in'])
            self.assertEqual(after['represented_area_in2'],496.*1456.)
            self.assertEqual(after['added_area_in2'],16*(480+1440)+16**2)
            self.assertEqual(after['shell_count'],(2*mesh+2)*(6*mesh+2))
            self.assertNotEqual(before['coordinate_sha256'],after['coordinate_sha256'])

    def test_extension_counts_against_budgets_and_preserves_existing_model(self):
        with self.assertRaisesRegex(ValueError,'budget'):
            coupled_floor_mesh(1,1,240.,200.,2,beam_width_in=12.,slab_perimeter=FACE,uniform_shell_limit=8)
        ops.model('basic','-ndm',3,'-ndf',6);ops.node(999,0.,0.,0.)
        with self.assertRaisesRegex(RuntimeError,'preserved'):
            analyze_coupled_gravity(SLAB,GEOMETRY,SECTIONS,[DEAD]*2,slab_perimeter=FACE)
        self.assertEqual(ops.getNodeTags(),[999])
        ops.wipe()
        g=dict(GEOMETRY,num_bay_x=1,num_bay_y=12,num_floor=6)
        with self.assertRaisesRegex(ValueError,'total shells'):
            analyze_coupled_gravity(SLAB,g,SECTIONS,[DEAD]*6,mesh_per_bay=10,slab_perimeter=FACE)

    def test_expanded_uniform_load_matches_axial_shortening_hand_solution(self):
        g=dict(GEOMETRY,bay_y_in=240.)
        for mesh in (2,4):
            r=analyze_coupled_gravity(SLAB,g,SECTIONS,[DEAD]*2,mesh_per_bay=mesh,
                slab_perimeter=FACE,include_member_weight=False)
            p=(.15*5/12+.05)*252**2/144
            ea=18**2*57*math.sqrt(5000.)
            self.assertEqual(r['status'],'diagnostic_complete')
            self.assertAlmostEqual(r['weight_ledger']['slab_area_load_kip'],2*p,places=8)
            for floor,factor in zip(r['floors'],(2.,3.)):
                for j in floor['column_joint_displacements']:
                    self.assertAlmostEqual(j['displacement_rotation'][2],-factor*p/4*144/ea,places=10)
            self.assertTrue(r['equilibrium']['numerical_balance_passed'])

    def test_corner_pattern_has_correct_area_centroid_and_no_extra_webs(self):
        g=dict(GEOMETRY,num_bay_x=2,num_bay_y=2)
        case=dict(DEAD,dead_factor=0.,live_factor=1.,live_pattern=[[0,0]])
        r=analyze_coupled_gravity(SLAB,g,SECTIONS,[ZERO,case],mesh_per_bay=4,slab_perimeter=FACE)
        p=.05*246*206/144
        applied=r['equilibrium']
        self.assertAlmostEqual(applied['applied_force_kip'][2],-p,places=9)
        self.assertAlmostEqual(applied['applied_moment_kip_in'][0],-p*97,places=8)
        self.assertAlmostEqual(applied['applied_moment_kip_in'][1],p*117,places=8)
        self.assertEqual(len(r['web_segment_actions']),2*4*(3*2+3*2))
        self.assertTrue(r['shell_to_frame_equilibrium']['numerical_balance_passed'])
        self.assertLess(r['rigid_offset_max_residual'],1e-10)

    def compare_reduction(self,restraint):
        g=dict(GEOMETRY,num_bay_x=2)
        case=dict(DEAD,dead_factor=1.2,live_factor=1.6,live_pattern=[[0,0]])
        model=CompatibleFloor(SLAB,g,SECTIONS,2,slab_perimeter=FACE)
        reduced=model.solve([ZERO,case],inplane_restraint=restraint)
        native=analyze_coupled_gravity(SLAB,g,SECTIONS,[ZERO,case],mesh_per_bay=2,slab_perimeter=FACE,
            inplane_restraint=restraint,constraint_handler='Transformation' if restraint=='finite_membrane' else 'Lagrange')
        self.assertEqual(native['status'],'diagnostic_complete')
        u=[[j['displacement_rotation'] for j in f['column_joint_displacements']] for f in native['floors']]
        np.testing.assert_allclose(reduced['joint_displacements'],u,rtol=1e-8,atol=1e-10)
        for a,b in zip(reduced['column_actions'],native['column_actions']):
            np.testing.assert_allclose(a['global_force_kip_kip_in'],b['global_force_kip_kip_in'],rtol=1e-8,atol=1e-6)
        self.assertAlmostEqual(reduced['weight_ledger']['slab_pressure_kip'],native['weight_ledger']['slab_area_load_kip'],places=8)
        for f in reduced['floors']:
            shells=[s for s in native['shell_resultants'] if s['floor']==f['story']]
            self.assertEqual(len(f['shells']),len(shells))
            for a,b in zip(f['shells'],shells):
                np.testing.assert_allclose(a['gauss_resultants_raw'],b['gauss_resultants_raw'],rtol=1e-7,atol=1e-8)
        self.assertFalse(reduced['applied_to_design'])

    def test_compatible_finite_membrane_solution_matches_native(self):
        self.compare_reduction('finite_membrane')

    def test_compatible_rigid_joints_solution_matches_native(self):
        self.compare_reduction('rigid_joints')

    def test_native_cuts_cover_full_edge_strength_width_and_refuse_missing_strip(self):
        g=dict(GEOMETRY,num_bay_x=2,num_bay_y=2)
        r=analyze_coupled_gravity(SLAB,g,SECTIONS,[DEAD]*2,mesh_per_bay=2,slab_perimeter=FACE)
        for axis,cut in (('x',120.),('y',100.)):
            result=recover_floor_regions(r,2,axis,cut)
            self.assertTrue(result['native_whole_cut_balanced'])
            self.assertTrue(result['numerical_partition_passed'])
            self.assertTrue(result['partition']['geometry_matches_all_strength_sections'])
            for row in result['rows']:
                if row['kind']=='beam_flange':self.assertAlmostEqual(row['missing_flange_width_in'],0.)
            self.assertFalse(result['capacity_pairing_authorized'])
        broken=copy.deepcopy(r)
        broken['shell_resultants']=[s for s in broken['shell_resultants'] if s['cell_i']!=-1]
        with self.assertRaisesRegex(ValueError,'inventory'):recover_floor_cut(broken,2,'x',120.)
        broken=copy.deepcopy(r)
        broken['inputs']['slab_perimeter']='centerlines'
        with self.assertRaisesRegex(ValueError,'inventory'):recover_floor_cut(broken,2,'x',120.)
        broken=copy.deepcopy(r)
        shell=next(s for s in broken['shell_resultants'] if s['floor']==2)
        shell['node_positions_in'][0]=list(shell['node_positions_in'][0]);shell['node_positions_in'][0][0]-=1
        with self.assertRaisesRegex(ValueError,'coordinates'):recover_floor_cut(broken,2,'x',120.)

    def test_rigid_floor_includes_outer_nodes_and_recovered_constraint_forces(self):
        r=analyze_coupled_gravity(SLAB,GEOMETRY,SECTIONS,[ZERO,DEAD],mesh_per_bay=2,slab_perimeter=FACE,
            inplane_restraint='rigid_floor',constraint_handler='Lagrange')
        self.assertEqual(r['assembly']['constrained_node_count_per_floor'],25)
        self.assertEqual(len(r['diaphragm_constraint_actions']),50)
        self.assertTrue(r['equilibrium']['numerical_balance_passed'])
        self.assertTrue(r['shell_to_frame_equilibrium']['numerical_balance_passed'])

    def test_perimeter_identity_cannot_reuse_old_condensed_stiffness(self):
        old=CompatibleFloor(SLAB,GEOMETRY,SECTIONS,2)
        new=CompatibleFloor(SLAB,GEOMETRY,SECTIONS,2,slab_perimeter=FACE)
        self.assertNotEqual(old._identity,new._identity)
        old.slab_perimeter=FACE
        with self.assertRaisesRegex(ValueError,'Inputs changed'):old.solve([DEAD]*2)
        with self.assertRaises(ValueError):CompatibleFloor(SLAB,GEOMETRY,SECTIONS,2,slab_perimeter='guessed')


if __name__=='__main__':unittest.main()
