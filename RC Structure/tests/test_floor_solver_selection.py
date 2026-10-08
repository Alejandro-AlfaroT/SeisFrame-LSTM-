"""Bounded solver equivalence and actual-solver provenance; no design qualification."""
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Design import SMRF_Floor_Analysis as floor_analysis
from Design.SMRF_Floor_Mesh import floor_mesh
from Design.SMRF_Slab_Actions import build_slab_action_evidence
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity
from Design.SMRF_Floor_Compatibility import CompatibleFloor

SLAB=dict(thickness_in=6.,concrete_fc_ksi=4.,concrete_unit_weight_kcf=.15,
          superimposed_dead_load_ksf=.05)
GEOMETRY=dict(num_bay_x=2,num_bay_y=2,bay_x_in=200.,bay_y_in=240.)
SECTIONS=dict(b_beam_in=14.,h_beam_in=28.,fc_beam_ksi=8.,b_col_in=36.,h_col_in=36.,fc_col_ksi=8.)
CASE=dict(id='asymmetric',dead_factor=1.2,live_factor=1.6,live_load_ksf=.05,live_pattern=[[0,0],[1,0]])
SPEC=dict(x_offsets_in=[0.,7.,45.,100.,193.,200.],
          y_offsets_in=[0.,7.,24.,60.,120.,216.,233.,240.],max_shells=8192)
SMALL_GEOMETRY=dict(num_bay_x=1,num_bay_y=1,bay_x_in=200.,bay_y_in=240.,num_floor=1,story_h_in=144.)
SMALL_SPEC=dict(x_offsets_in=[0.,100.,200.],y_offsets_in=[0.,120.,240.],max_shells=8192)
DEAD=dict(id='D',dead_factor=1.,live_factor=0.,live_load_ksf=.05,live_pattern='none')


class FloorSolverSelectionTests(unittest.TestCase):
    def setUp(self):
        ops.wipe()
        self.addCleanup(ops.wipe)

    def assert_same_numeric_tree(self, left, right, path='root'):
        """Compare inventories exactly and physical values with a tight mixed tolerance."""
        if isinstance(left,dict):
            self.assertIsInstance(right,dict,path)
            self.assertEqual(set(left),set(right),path)
            for k in left:self.assert_same_numeric_tree(left[k],right[k],path+'.'+str(k))
        elif isinstance(left,list):
            self.assertIsInstance(right,list,path)
            self.assertEqual(len(left),len(right),path)
            for k,(a,b) in enumerate(zip(left,right)):self.assert_same_numeric_tree(a,b,path+f'[{k}]')
        elif isinstance(left,float):
            self.assertTrue(math.isfinite(left) and math.isfinite(right),path)
            self.assertTrue(math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-9),f'{path}: {left} != {right}')
        else:
            self.assertEqual(left,right,path)

    def test_explicit_solver_selection_preserves_uniform_geometry_and_budget(self):
        native=floor_mesh(1,1,200.,240.,2)
        explicit=floor_mesh(1,1,200.,240.,2,mesh_spec=SMALL_SPEC)
        self.assertEqual(native['solver'],'BandGeneral')
        self.assertEqual(explicit['solver'],'SuperLU')
        for key in ('coordinate_sha256','shell_count','node_count','x_coordinates_in','y_coordinates_in'):
            self.assertEqual(native[key],explicit[key],key)
        with self.assertRaisesRegex(ValueError,'exceeding explicit budget'):
            floor_mesh(2,2,200.,240.,2,mesh_spec=dict(SMALL_SPEC,max_shells=4))

    def test_nonuniform_floor_force_displacement_and_strip_demands_match_umfpack(self):
        original_grid=floor_analysis.floor_mesh
        results={}
        evidence={}
        slab_inputs=dict(thickness_in=6.,fc_ksi=4.,fy_ksi=60.,max_aggregate_size_in=.75,
                         exposure='sheltered_interior',steel_specification='ASTM A706',
                         concrete_type='normalweight',num_floor=1,
                         panel_ids=[f'panel_x{i}_y{j}' for i in (1,2) for j in (1,2)])
        for solver in ('UmfPack','SuperLU'):
            def selected_grid(*args,**kwargs):
                grid=original_grid(*args,**kwargs)
                grid['solver']=solver
                return grid
            with mock.patch.object(floor_analysis,'floor_mesh',side_effect=selected_grid), mock.patch.object(ops,'system',wraps=ops.system) as system:
                result=floor_analysis.analyze_floor(SLAB,GEOMETRY,SECTIONS,CASE,
                                                   support_model='flexible_beams',mesh_spec=SPEC)
                system.assert_called_once_with(solver)
                self.assertEqual(result['mesh']['solver'],solver)
                self.assertEqual(result['status'],'transfer_complete')
                self.assertTrue(result['equilibrium']['numerical_balance_passed'])
                self.assertTrue(result['transfer_equilibrium']['numerical_balance_passed'])
                results[solver]=result
                # Exercise production recovery/envelope generation too, not only raw shell response.
                evidence[solver]=build_slab_action_evidence(SLAB,GEOMETRY,SECTIONS,.05,slab_inputs,mesh_spec=SPEC)
                self.assertFalse(evidence[solver]['verified'])
        for key in ('vertical_displacements','support_node_reactions','beam_transfer','column_direct_loads'):
            self.assert_same_numeric_tree(results['UmfPack'][key],results['SuperLU'][key],key)
        panels={s:{p['panel_id']:p for p in r['panels']} for s,r in results.items()}
        self.assertEqual(set(panels['UmfPack']),set(panels['SuperLU']))
        for panel_id in panels['UmfPack']:
            self.assert_same_numeric_tree(panels['UmfPack'][panel_id]['gauss_point_resultants'],
                                          panels['SuperLU'][panel_id]['gauss_point_resultants'],panel_id)
        strips={s:{(r['panel_id'],r['axis'],r['face']):r for r in e['strips']} for s,e in evidence.items()}
        self.assertEqual(set(strips['UmfPack']),set(strips['SuperLU']))
        for key in strips['UmfPack']:
            for metric in ('mu_kip_in_per_ft','vu_kip_per_ft'):
                self.assert_same_numeric_tree(strips['UmfPack'][key][metric],strips['SuperLU'][key][metric],str(key)+metric)
        self.assertEqual(evidence['UmfPack']['physical_model_sha256'],evidence['SuperLU']['physical_model_sha256'])
        self.assertNotEqual(evidence['UmfPack']['analysis_model_sha256'],evidence['SuperLU']['analysis_model_sha256'])

    def test_failed_explicit_solve_remains_unverified_without_solver_fallback(self):
        with mock.patch.object(ops,'analyze',return_value=-3), mock.patch.object(ops,'system',wraps=ops.system) as system:
            result=floor_analysis.analyze_floor(SLAB,GEOMETRY,SECTIONS,CASE,
                                               support_model='flexible_beams',mesh_spec=SPEC)
        system.assert_called_once_with('SuperLU')
        self.assertEqual(result['status'],'analysis_failed')
        self.assertEqual(result['mesh']['solver'],'SuperLU')
        self.assertFalse(result['verified'])
        self.assertNotIn('panels',result)
        self.assertEqual(ops.getNodeTags(),[])

    def test_coupled_metadata_records_retained_umfpack_solver(self):
        for mesh_spec in (None,SMALL_SPEC):
            with self.subTest(explicit=mesh_spec is not None), mock.patch.object(ops,'system',wraps=ops.system) as system:
                result=analyze_coupled_gravity(SLAB,SMALL_GEOMETRY,SECTIONS,[DEAD],
                                               mesh_per_bay=2,mesh_spec=mesh_spec)
            self.assertEqual(result['status'],'diagnostic_complete')
            self.assertEqual(result['mesh_metadata']['solver'],'UmfPack')
            system.assert_called_once_with('UmfPack')
            self.assertTrue(result['equilibrium']['numerical_balance_passed'])

    def test_compatible_response_metadata_records_reference_and_default_solvers(self):
        for mesh_spec in (None,SMALL_SPEC):
            model=CompatibleFloor(SLAB,SMALL_GEOMETRY,SECTIONS,mesh_per_bay=2,mesh_spec=mesh_spec)
            for reference,solver in ((False,'SuperLU'),(True,'UmfPack')):
                with self.subTest(explicit=mesh_spec is not None,reference=reference), mock.patch.object(ops,'system',wraps=ops.system) as system:
                    result=model.response(DEAD,_reference_solver=reference)
                self.assertEqual(result['mesh_metadata']['solver'],solver)
                self.assertEqual(result['response_system'],solver)
                system.assert_called_once_with(solver)

    def test_compatible_stiffness_metadata_records_superlu(self):
        model=CompatibleFloor(SLAB,SMALL_GEOMETRY,SECTIONS,mesh_per_bay=2)
        zero=dict(DEAD,dead_factor=0.)
        with mock.patch.object(ops,'system',wraps=ops.system) as system:
            result=model.response(zero,_extract_stiffness=True)
        self.assertEqual(result['mesh_metadata']['solver'],'SuperLU')
        system.assert_called_once_with('SuperLU')


if __name__=='__main__':
    unittest.main()
