"""Native equivalence, nonuniform load placement and refinement failure guards."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops
from Design.SMRF_Floor_Analysis import analyze_floor
from Design.SMRF_Floor_Mesh import floor_mesh, MAX_EXPLICIT_SHELLS
from Design.SMRF_Slab_Actions import evaluate_slab_actions
from Design.SMRF_Slab_Refinement import build_refined_slab_action_evidence, refinement_verified
from test_smrf_slab_recovery import panel

SLAB = dict(thickness_in=6., concrete_fc_ksi=4., concrete_unit_weight_kcf=.15,
            superimposed_dead_load_ksf=.05)
GEOMETRY = dict(num_bay_x=1, num_bay_y=1, bay_x_in=200., bay_y_in=200.)
SECTIONS = dict(b_beam_in=14., h_beam_in=28., fc_beam_ksi=8., b_col_in=36., h_col_in=36.)
CASE = dict(id='pattern', dead_factor=1.2, live_factor=1.6, live_load_ksf=.05, live_pattern='all')
INPUTS = dict(thickness_in=6., fc_ksi=4., fy_ksi=60., max_aggregate_size_in=.75,
              exposure='sheltered_interior', steel_specification='ASTM A706', concrete_type='normalweight',
              num_floor=1, panel_ids=['panel_x1_y1'])


def spec(xs, ys=None, budget=8192):
    return dict(x_offsets_in=list(xs), y_offsets_in=list(xs if ys is None else ys), max_shells=budget)


def policy():
    return dict(meshes=[spec([0., 7., 50., 100., 150., 193., 200.]),
                        spec([0., 3.5, 7., 28.5, 50., 75., 100., 125., 150., 171.5, 193., 196.5, 200.])],
                moment_tolerance=.05, shear_tolerance=.05, tolerance_basis='Manufactured-field test only')


def signed():
    flags = dict.fromkeys(('verified', 'analysis_applicability_verified', 'all_floors_enveloped',
                          'load_pattern_envelope_verified', 'spatial_envelope_per_unit_width',
                          'twisting_moment_resolution_verified', 'zero_membrane_force_verified'), True)
    return dict(flags, asserted_by='test fixture', assertion_date='2026-09-24',
                assertion_basis='Manufactured constant field, not engineering approval')


def manufactured(slab, geometry, sections, case, *, mesh_spec, **kwargs):
    grid = floor_mesh(1, 1, 200., 200., 4, mesh_spec=mesh_spec)
    return dict(status='transfer_complete', mesh=grid, max_abs_membrane_kip_per_in=0.,
                equilibrium=dict(numerical_balance_passed=True),
                panels=[panel(mesh_spec['x_offsets_in'], mesh_spec['y_offsets_in'],
                              lambda x, y, i, j: dict(mx=-1., my=-2., mxy_raw=0., qx_raw=1., qy_raw=2.))])


class ExplicitFloorMeshTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_explicit_uniform_grid_reproduces_native_uniform_result_and_transfer(self):
        for support in ('rigid_lines', 'flexible_beams'):
            with self.subTest(support=support):
                old = analyze_floor(SLAB, GEOMETRY, SECTIONS, CASE, 8, support)
                new = analyze_floor(SLAB, GEOMETRY, SECTIONS, CASE, support_model=support,
                                    mesh_spec=spec([i * 25. for i in range(9)]))
                self.assertEqual(old['status'], new['status'])
                self.assertEqual(old['mesh']['coordinate_sha256'], new['mesh']['coordinate_sha256'])
                self.assertAlmostEqual(old['maximum_downward_displacement_in'], new['maximum_downward_displacement_in'], places=11)
                for a, b in zip(old['panels'][0]['gauss_point_resultants'], new['panels'][0]['gauss_point_resultants']):
                    for key in ('x_in', 'y_in', 'mx', 'my', 'mxy_raw', 'qx_raw', 'qy_raw'):
                        self.assertAlmostEqual(a[key], b[key], places=10)
                if support == 'flexible_beams':
                    for a, b in zip(old['beam_transfer'], new['beam_transfer']):
                        self.assertAlmostEqual(a['total_kip'], b['total_kip'], places=10)

    def test_nonuniform_anisotropic_multibay_pattern_balances_and_uses_physical_fractions(self):
        g = dict(GEOMETRY, num_bay_x=2, num_bay_y=2, bay_y_in=240.)
        s = spec([0., 7., 45., 100., 193., 200.], [0., 7., 24., 60., 120., 216., 233., 240.])
        case = dict(CASE, live_pattern=[[0, 0], [1, 0]])
        before = copy.deepcopy((g, s, case))
        r = analyze_floor(SLAB, g, SECTIONS, case, support_model='flexible_beams', mesh_spec=s)
        self.assertEqual((g, s, case), before)
        self.assertEqual(r['status'], 'transfer_complete')
        self.assertEqual(r['mesh']['shell_count'], 140)
        self.assertTrue(r['equilibrium']['numerical_balance_passed'])
        self.assertTrue(r['transfer_equilibrium']['numerical_balance_passed'])
        self.assertEqual({(c['grid_i'], c['grid_j']) for c in r['column_direct_loads']},
                         {(i, j) for i in range(3) for j in range(3)})
        for beam in r['beam_transfer']:
            coords = s['x_offsets_in'] if beam['axis'] == 'x' else s['y_offsets_in']
            self.assertEqual([n['x_fraction'] for n in beam['node_loads']], [v / coords[-1] for v in coords[1:-1]])
        self.assertEqual(ops.getNodeTags(), [])
        json.dumps(r, allow_nan=False)

    def test_invalid_mesh_and_budget_rejected_without_wiping_existing_domain(self):
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        ops.node(900, 1., 2., 3.)
        bad = [spec([0, 0, 200]), spec([0, 10, 199]), spec([0, math.nan, 200]),
               spec([0, True, 200]), spec([0, 7, 100, 193, 200], budget=4),
               spec([0, 100, 200], budget=MAX_EXPLICIT_SHELLS + 1)]
        for s in bad:
            with self.subTest(mesh=s), self.assertRaises(ValueError):
                analyze_floor(SLAB, GEOMETRY, SECTIONS, CASE, mesh_spec=s)
            self.assertEqual(ops.getNodeTags(), [900])


class SlabRefinementTests(unittest.TestCase):
    def run_fixture(self, side_effect=manufactured, flags=None, plan=None):
        with mock.patch('Design.SMRF_Slab_Actions.analyze_floor', side_effect=side_effect):
            return build_refined_slab_action_evidence(SLAB, GEOMETRY, SECTIONS, .05, INPUTS,
                                                      policy() if plan is None else plan, assertions=flags)

    def test_complete_refinement_keeps_engineering_verification_separate(self):
        e = self.run_fixture()
        self.assertTrue(refinement_verified(e))
        self.assertFalse(e['verified'])
        self.assertFalse(e['refinement']['engineering_verified'])
        self.assertEqual(len(e['refinement']['levels']), 2)
        self.assertEqual([len(l['attempted_cases']) for l in e['refinement']['levels']], [2, 2])
        signed_e = self.run_fixture(flags=signed())
        self.assertTrue(signed_e['verified'])
        checks = {c['id']: c['status'] for c in evaluate_slab_actions(signed_e)}
        self.assertEqual(checks['floor.independent_hand_verification'], 'not_evaluated')
        self.assertEqual(checks['floor.slab_action_mesh_refinement'], 'pass')
        json.dumps(signed_e, allow_nan=False)

    def test_failed_fine_solve_retains_coarse_evidence_but_cannot_qualify_it(self):
        def fail(*args, **kwargs):
            if len(kwargs['mesh_spec']['x_offsets_in']) > 7:
                raise RuntimeError('retained synthetic solver failure')
            return manufactured(*args, **kwargs)
        e = self.run_fixture(fail, signed())
        self.assertFalse(e['verified'])
        self.assertFalse(refinement_verified(e))
        self.assertEqual(e['refinement']['status'], 'analysis_failed')
        self.assertIn('actions', e['refinement']['levels'][0])
        self.assertIn('retained synthetic solver failure', e['refinement']['levels'][1]['error'])
        self.assertEqual(e['refinement']['levels'][1]['attempted_cases'][0]['status'], 'exception')

    def test_first_failure_remains_serializable_unverified_evidence(self):
        e = self.run_fixture(lambda *a, **k: {'status': 'analysis_failed', 'analysis_return_code': -3})
        self.assertEqual(e['strips'], [])
        self.assertFalse(e['verified'])
        self.assertFalse(refinement_verified(e))
        json.dumps(e, allow_nan=False)
        self.assertEqual(evaluate_slab_actions(e)[0]['status'], 'not_evaluated')

    def test_unstable_fine_demands_remain_failed_even_with_assertions(self):
        def changed(*args, **kwargs):
            r = manufactured(*args, **kwargs)
            if len(kwargs['mesh_spec']['x_offsets_in']) > 7:
                for p in r['panels'][0]['gauss_point_resultants']:
                    p['qx_raw'] *= 2
            return r
        e = self.run_fixture(changed, signed())
        self.assertEqual(e['refinement']['status'], 'comparison_failed')
        self.assertFalse(e['verified'])
        self.assertTrue(any(not r['within_tolerance'] for r in e['refinement']['comparisons'][0]['comparisons']))

    def test_cached_pass_cannot_hide_altered_demands_mesh_or_partial_coverage(self):
        original = self.run_fixture(flags=signed())
        variants = []
        e = copy.deepcopy(original); e['strips'][0]['mu_kip_in_per_ft'] += 1; variants.append(e)
        e = copy.deepcopy(original); e['solved_meshes'][0]['mesh']['x_offsets_in'][1] += 1; variants.append(e)
        e = copy.deepcopy(original); e['refinement']['levels'].pop(0); variants.append(e)
        e = copy.deepcopy(original); e['refinement']['levels'][-1]['actions']['solved_meshes'].pop(0); variants.append(e)
        for e in variants:
            self.assertFalse(refinement_verified(e))
            self.assertEqual(evaluate_slab_actions(e)[0]['status'], 'not_evaluated')

    def test_reinforcement_consumer_rechecks_refinement_instead_of_trusting_cached_flags(self):
        from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
        e = self.run_fixture(flags=signed())
        e['strips'][0]['mu_kip_in_per_ft'] += 1.
        self.assertTrue(e['verified'])   # stale cached flag cannot authorize selection
        result = design_slab_reinforcement(INPUTS, e)
        self.assertIsNone(result['layout'])
        self.assertIn('identity-matched mesh refinement', result['checks'][0]['details']['reason'])

    def test_load_magnitude_changes_physical_identity_even_with_same_case_names(self):
        with mock.patch('Design.SMRF_Slab_Actions.analyze_floor', side_effect=manufactured):
            a = build_refined_slab_action_evidence(SLAB, GEOMETRY, SECTIONS, .05, INPUTS, policy())
            b = build_refined_slab_action_evidence(SLAB, GEOMETRY, SECTIONS, .06, INPUTS, policy())
        self.assertEqual([c['id'] for c in a['cases']], [c['id'] for c in b['cases']])
        self.assertNotEqual(a['physical_model_sha256'], b['physical_model_sha256'])

    def test_nonnested_duplicate_overbudget_and_unbounded_plans_fail_before_solving(self):
        variants = []
        p = policy(); p['meshes'][1]['x_offsets_in'][2] = 8.; variants.append(p)
        p = policy(); p['meshes'][1] = copy.deepcopy(p['meshes'][0]); variants.append(p)
        p = policy(); p['meshes'][1]['max_shells'] = 4; variants.append(p)
        p = policy(); p['moment_tolerance'] = None; variants.append(p)
        p = policy(); p['tolerance_basis'] = ''; variants.append(p)
        with mock.patch('Design.SMRF_Slab_Actions.analyze_floor') as solve:
            for p in variants:
                with self.subTest(plan=p), self.assertRaises(ValueError):
                    build_refined_slab_action_evidence(SLAB, GEOMETRY, SECTIONS, .05, INPUTS, p)
            solve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
