"""Bounded witness-based mesh extension, replay integrity and refusal guards."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Floor_Mesh import (floor_mesh, resolve_recipe_plan, nested_refinement,
                                    witness_face_band_extension, LOCAL_EXTENSION_METHOD)
from Design.SMRF_Slab_Actions import compare_slab_action_refinement
from Design.SMRF_Slab_Refinement import build_refined_slab_action_evidence, refinement_verified, refinement_diagnostics


def policy(**changes):
    return dict(recipe='graded_face_v1', levels=4, max_shells=130000,
                moment_tolerance=.05, shear_tolerance=.05, tolerance_basis='Manufactured test only',
                local_extension=dict(method=LOCAL_EXTENSION_METHOD, max_extra_levels=2), **changes)


G=dict(num_bay_x=1, num_bay_y=1, bay_x_in=240., bay_y_in=300.)
S=dict(b_beam_in=20.)
CASE=dict(id='manufactured')


def action(grid, shear, location=None):
    return dict(analysis_model_sha256=hashlib.sha256(json.dumps(grid,sort_keys=True).encode()).hexdigest(),
                physical_model_sha256='p'*64, slab_input_sha256='s'*64, cases=[CASE],
                solved_meshes=[dict(case_id=CASE['id'],mesh=grid)],
                equilibrium=[dict(numerical_balance_passed=True)], shear_recovery={}, max_abs_membrane_kip_per_in=0.,
                numerical_preconditions=dict(physical_recovery_valid=True), numerical_basis={},
                engineering_assertions={}, assertion_provenance_valid=False,
                strips=[dict(panel_id='panel_x1_y1',axis='x',face='bottom',
                             mu_kip_in_per_ft=1.,vu_kip_per_ft=shear,
                             shear_location=dict(x_in=10.,y_in=150.) if location is None else location)])


def resolved(g=G, s=S):
    resolution=resolve_recipe_plan(g,s,policy())
    grid=floor_mesh(g['num_bay_x'],g['num_bay_y'],g['bay_x_in'],g['bay_y_in'],4,
                    mesh_spec=resolution['meshes'][-1])
    return resolution,grid


def proposal(g=G,s=S,location=None,used=()):
    resolution,grid=resolved(g,s)
    a,b=action(dict(grid,solver='coarse'),1.,location),action(grid,1.1,location)
    comparison=compare_slab_action_refinement(a,b,moment_tolerance=.05,shear_tolerance=.05)
    return witness_face_band_extension(g,resolution['inputs'],a,b,comparison,grid,used),grid


class LocalFaceMeshTests(unittest.TestCase):
    def test_transpose_and_reflection_select_the_same_physical_face_direction(self):
        original,grid=proposal()
        self.assertEqual(original['selected_axis'],'x')
        self.assertEqual(original['cells_per_bay'],[84,60])
        reflected,_=proposal(location=dict(x_in=230.,y_in=150.))
        self.assertEqual(reflected['selected_axis'],'x')
        self.assertEqual(reflected['mesh_spec'],original['mesh_spec'])
        transposed,_=proposal(dict(G,bay_x_in=300.,bay_y_in=240.),location=dict(x_in=150.,y_in=10.))
        self.assertEqual(transposed['selected_axis'],'y')
        self.assertEqual(transposed['mesh_spec']['y_offsets_in'],original['mesh_spec']['x_offsets_in'])
        nested_refinement(grid,floor_mesh(1,1,240.,300.,4,mesh_spec=original['mesh_spec']))

    def test_tied_faces_choose_x_before_y_and_an_axis_is_never_extended_twice(self):
        chosen,_=proposal(dict(G,bay_y_in=240.),location=dict(x_in=10.,y_in=10.))
        self.assertEqual(chosen['selected_axis'],'x')
        other,_=proposal(dict(G,bay_y_in=240.),location=dict(x_in=10.,y_in=10.),used=('x',))
        self.assertEqual(other['selected_axis'],'y')
        exhausted,_=proposal(dict(G,bay_y_in=240.),location=dict(x_in=10.,y_in=10.),used=('x','y'))
        self.assertEqual(exhausted['status'],'no_supported_witness')

    def test_nonfinite_missing_or_outside_witness_refuses_selection(self):
        for location in ({'x_in':float('nan'),'y_in':150.}, {'x_in':10.},
                         {'x_in':True,'y_in':150.}, {'x_in':-1.,'y_in':150.}):
            with self.subTest(location=location):
                chosen,_=proposal(location=location)
                self.assertEqual(chosen['status'],'unsupported_evidence')
        chosen,_=proposal(location=dict(x_in=120.,y_in=150.))
        self.assertEqual(chosen['status'],'no_supported_witness')

    def test_shell_cap_blocks_extension_without_coarsening(self):
        chosen,grid=proposal(dict(G,num_bay_x=6,num_bay_y=6))
        self.assertEqual(grid['shell_count'],129600)
        self.assertEqual(chosen['shell_count'],181440)
        self.assertEqual(chosen['status'],'budget_exceeded')
        self.assertEqual(chosen['mesh_spec']['max_shells'],130000)

    def test_by_line_mesh_uses_each_bays_faces_and_retains_every_coordinate(self):
        g=dict(G,num_bay_x=2,num_bay_y=2)
        s=dict(face_widths_in={'x':[20.,24.,28.],'y':[16.,22.,30.]})
        chosen,grid=proposal(g,s,dict(x_in=10.,y_in=150.))
        self.assertEqual(chosen['status'],'proposed')
        self.assertEqual(chosen['cells_per_bay'],[84,60])
        spec=chosen['mesh_spec']
        self.assertNotEqual(spec['x_offsets_in'][0],spec['x_offsets_in'][1])
        self.assertEqual({len(v) for v in spec['x_offsets_in']},{85})
        nested_refinement(grid,floor_mesh(2,2,240.,300.,4,mesh_spec=spec))


class LocalExtensionWorkflowTests(unittest.TestCase):
    def run_fixture(self, values=(1.,1.,1.,1.1,1.11), locations=None, fail=None, p=None, geometry=None, sections=None):
        g=G if geometry is None else geometry
        s=S if sections is None else sections
        calls=[]
        def solve(*args,mesh_spec,**kwargs):
            index=len(calls)
            calls.append(copy.deepcopy(mesh_spec))
            if index==fail:raise RuntimeError('retained synthetic solver failure')
            grid=floor_mesh(g['num_bay_x'],g['num_bay_y'],g['bay_x_in'],g['bay_y_in'],4,mesh_spec=mesh_spec)
            return action(grid,values[index],None if locations is None else locations[index])
        with mock.patch('Design.SMRF_Slab_Actions.build_slab_action_evidence',side_effect=solve):
            evidence=build_refined_slab_action_evidence({},g,s,0.,{},policy() if p is None else p)
        return evidence,calls

    def test_full_base_failure_is_retained_then_one_extension_passes_and_replays(self):
        e,calls=self.run_fixture()
        report=e['refinement']
        self.assertEqual(len(calls),5)
        self.assertEqual([l['requested_mesh']['subdivisions_x_per_bay'] for l in report['levels']],[12,24,48,60,84])
        self.assertFalse(report['comparisons'][2]['all_within_tolerance'])
        self.assertTrue(report['comparisons'][3]['all_within_tolerance'])
        self.assertEqual(report['extension_attempts'][0]['selected_axis'],'x')
        self.assertTrue(refinement_verified(e))
        self.assertFalse(e['verified'])
        self.assertEqual(refinement_diagnostics(e)['extension_attempts'],report['extension_attempts'])

    def test_passed_base_and_disabled_extension_do_not_add_levels(self):
        e,calls=self.run_fixture(values=(1.,1.,1.,1.))
        self.assertEqual(len(calls),4)
        self.assertEqual(e['refinement']['extension_attempts'],[])
        self.assertTrue(refinement_verified(e))
        p=policy();del p['local_extension']
        e,calls=self.run_fixture(p=p)
        self.assertEqual(len(calls),4)
        self.assertEqual(e['refinement']['status'],'comparison_failed')
        self.assertFalse(refinement_verified(e))

    def test_by_line_workflow_reconstructs_its_own_face_widths(self):
        from Design.SMRF_Floor_Sections import SCHEMA
        beam=lambda width:dict(b_in=width,h_in=24.,fc_ksi=4.)
        sections=dict(schema=SCHEMA,fc_beam_ksi=4.,
                      beam_lines=dict(x=[beam(16.),beam(22.)],y=[beam(20.),beam(28.)]),
                      supports=[[dict(b_in=24.,h_in=24.) for _ in range(2)] for _ in range(2)])
        e,calls=self.run_fixture(sections=sections)
        self.assertEqual(len(calls),5)
        self.assertEqual(e['refinement']['recipe_inputs']['face_widths_in'],
                         {'x':[20.,28.],'y':[16.,22.]})
        self.assertTrue(refinement_verified(e))
        changed=copy.deepcopy(e)
        changed['refinement']['recipe_inputs']['face_widths_in']['x'][0]+=2.
        self.assertFalse(refinement_verified(changed))

    def test_second_extension_uses_other_axis_and_stops_at_declared_limit(self):
        locations=[dict(x_in=10.,y_in=10.)]*6
        e,calls=self.run_fixture(values=(1.,1.,1.,1.1,1.2,1.21),locations=locations)
        self.assertEqual(len(calls),6)
        self.assertEqual([d['selected_axis'] for d in e['refinement']['extension_attempts']],['x','y'])
        self.assertTrue(refinement_verified(e))
        p=policy();p['local_extension']['max_extra_levels']=1
        e,calls=self.run_fixture(values=(1.,1.,1.,1.1,1.2),locations=locations,p=p)
        self.assertEqual(len(calls),5)
        self.assertEqual(e['refinement']['extension_attempts'][-1]['status'],'limit_reached')
        self.assertFalse(refinement_verified(e))

    def test_solver_failure_at_base_or_extension_never_accepts_previous_coarse(self):
        for failed_index in (3,4):
            with self.subTest(failed_index=failed_index):
                e,calls=self.run_fixture(fail=failed_index)
                self.assertEqual(len(calls),failed_index+1)
                self.assertEqual(e['refinement']['status'],'analysis_failed')
                self.assertEqual(e['refinement']['levels'][-1]['status'],'failed')
                self.assertFalse(refinement_verified(e))
                if failed_index==3:self.assertEqual(e['refinement']['extension_attempts'],[])

    def test_missing_witness_and_insufficient_extension_budget_remain_failed(self):
        locations=[dict(x_in=10.,y_in=150.)]*3+[{}]
        e,calls=self.run_fixture(locations=locations)
        self.assertEqual(len(calls),4)
        self.assertEqual(e['refinement']['status'],'comparison_failed')
        self.assertEqual(e['refinement']['extension_attempts'][0]['status'],'unsupported_evidence')
        e,calls=self.run_fixture(geometry=dict(G,num_bay_x=6,num_bay_y=6))
        self.assertEqual(len(calls),4)
        self.assertEqual(e['refinement']['extension_attempts'][0]['status'],'budget_exceeded')
        self.assertFalse(refinement_verified(e))

    def test_insufficient_base_budget_solves_nothing_and_invalid_policy_is_rejected(self):
        p=policy();p['max_shells']=3000
        e,calls=self.run_fixture(p=p)
        self.assertEqual(calls,[])
        self.assertEqual(e['refinement']['status'],'unresolved_budget')
        for key,value in (('max_extra_levels',3),('max_extra_levels',True),('method','unknown')):
            p=policy();p['local_extension'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.run_fixture(p=p)
        p=policy();p['levels']=3
        with self.assertRaises(ValueError):self.run_fixture(p=p)

    def test_replay_rejects_cached_early_pass_truncation_plan_and_witness_tampering(self):
        original,_=self.run_fixture()
        def bad_early(e):e['refinement']['comparisons'][0]['comparisons'][0]['relative_change']=.9
        def forged_base(e):e['refinement']['comparisons'][2]['all_within_tolerance']=True
        def truncate(e):e['refinement']['levels'].pop()
        def wrong_axis(e):e['refinement']['extension_attempts'][0]['selected_axis']='y'
        def wrong_mesh(e):e['refinement']['levels'][-1]['requested_mesh']['shell_count']+=1
        def wrong_witness(e):e['refinement']['levels'][3]['actions']['strips'][0]['shear_location']['x_in']=120.
        for change in (bad_early,forged_base,truncate,wrong_axis,wrong_mesh,wrong_witness):
            with self.subTest(change=change.__name__):
                altered=copy.deepcopy(original);change(altered)
                self.assertFalse(refinement_verified(altered))


if __name__=='__main__':
    unittest.main()
