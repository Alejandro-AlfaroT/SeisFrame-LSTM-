"""No-solve regressions for the November 1 safeguard review.

All passing numeric rows below are synthetic software evidence, never engineering approval.
The detailing checklist is independent of both expected_inventory and the row producer.
"""
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import contract_fixture as cf
from Design import Code_Check_Contract as contract
from Design import Design_Evidence_Profile as dep
from Design import Genetic_Search as ga
from Design import Genetic_Generation as gen
from Design import Group_Selection as selection
from Design import Grouped_Design as gd
from Design import Grouped_Search as search
from Design import Grouped_Record as gr
from Design import Group_Checks as checks
from Design import Genetic_Jumps as jumps
from Design import Design_Driver as driver
from Design.Config import DesignConfig, with_beam_practical_ratio, validate_beam_practical_ceiling
import Redesign as redesign
import Structure_Parameters as sp
from Model import Member_Groups as mg


# Deliberately separate from the contract and producer lists. Empty suffix = group-level row.
BEAM_COMMON = {
    'beam.minimum_width': ('',), 'beam.width_to_depth_design_basis': ('',),
    'beam.depth_to_width_design_basis': ('',), 'beam.bar_clear_spacing': ('top', 'bottom'),
    'beam.minimum_steel': ('top', 'bottom'), 'beam.maximum_rho': ('top', 'bottom'),
    'beam.continuous_bars': ('top', 'bottom'),
}
BEAM_SFRS = {
    'beam.clear_span': ('',), 'beam.column_projection': ('',),
    'beam.reversal_balance': ('left', 'right'), 'beam.span_strength': ('positive', 'negative'),
    'beam.end_hoop_spacing': ('',), 'beam.end_zone_length': ('',), 'beam.first_hoop': ('',),
}
COLUMN = {
    'column.minimum_dimension': ('',), 'column.aspect_ratio': ('',),
    'column.bar_clear_spacing': ('top', 'bottom'), 'column.side_bar_clear_spacing': ('',),
    'column.minimum_rho': ('',), 'column.maximum_rho': ('',),
    'column.end_hoop_spacing_dimension_bar': ('',), 'column.end_hoop_spacing': ('',),
    'column.end_zone_length': ('',), 'column.supported_bar_distance_basic': ('',),
    'column.confinement_area_and_support': ('',),
}


def detailing_keys(state):
    keys = {('detailing', 'detailing.material_scope', '')}
    with mg.installed(state):
        for gid, design in state.designs.items():
            gravity = mg.is_system_policy(state.policy_id) and state.system_of(gid) == mg.SYSTEM_GRAVITY
            rules = COLUMN if design.is_column else {**BEAM_COMMON, **({} if gravity else BEAM_SFRS)}
            for name, suffixes in rules.items():
                keys.update(('detailing', name, f'{gid}/{s}' if s else gid) for s in suffixes)
            if not design.is_column and not gravity:
                # Geometry contexts, never emitted rows or the contract's inventory.
                contexts = {}
                for tag in state.groups[gid]['member_tags']:
                    contexts.setdefault(checks.beam_section_context(state.member(tag)), tag)
                for tag in list(contexts.values())[1:]:
                    for name in ('beam.reversal_balance', 'beam.span_strength'):
                        keys.update(('detailing', name, f'{gid}/member:{tag}/{s}') for s in BEAM_SFRS[name])
    return keys


def complete_rows(state):
    # Other families use their existing fixture; no detailing row is taken from the inventory under test.
    return [r for r in cf.rows_for(state) if r['family'] != 'detailing'] + [cf._row(*key) for key in sorted(detailing_keys(state))]


class DetailingInventory(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)
        self.state = gf.uniform_state()

    def test_failing_minimum_steel_cannot_be_hidden_by_omission(self):
        rows = complete_rows(self.state)
        key = ('detailing', 'beam.minimum_steel', cf.any_gid(self.state, 'beam') + '/top')
        remaining = [r for r in rows if (r['family'], r['id'], r['location']) != key]
        failed = contract.assess_evaluation(cf.evaluation_record(self.state, rows=remaining + [cf._row(*key, status='fail')]))
        omitted = contract.assess_evaluation(cf.evaluation_record(self.state, rows=remaining))
        self.assertEqual(failed['classification'], 'check_failed')
        self.assertFalse(omitted['accepted'], omitted['classification'])

    def test_every_independently_listed_detailing_identity_is_required(self):
        for policy in ('band', 'system_three_story_v1'):
            state = self.state if policy == 'band' else gr.seed_from_sections((26, 26, 6), (16, 30, 5), policy_id=policy)[0]
            rows, expected = complete_rows(state), contract.expected_inventory(state)
            self.assertTrue(contract.assess_evaluation(cf.evaluation_record(state, rows=rows))['accepted'])
            for key in sorted(detailing_keys(state)):
                with self.subTest(policy=policy, key=key):
                    remaining = [r for r in rows if (r['family'], r['id'], r['location']) != key]
                    result = contract.code_check_result(remaining, expected)
                    self.assertFalse(result['accepted'])
                    self.assertIn(dict(zip(('family', 'id', 'location'), key)), result['missing'])

    def test_producer_identities_match_the_independent_candidate_checklist(self):
        for state in (self.state, gr.seed_from_sections((26, 26, 6), (16, 30, 5), policy_id='system_three_story_v1')[0]):
            cfg = DesignConfig()
            with mg.installed(state):
                view = {'materials': gr._materials(cfg, state), 'detailing': gr.group_detailing(state, {'columns': {}}),
                        'geometry': driver._slab_geometry()}
                # Synthetic complete strengths isolate row production; no slab assertions or solves.
                strengths = {f'mn_{sign}_{end}_kipin': 1000 for sign in ('positive', 'negative')
                             for end in ('left', 'right', 'min_along')}
                with patch.object(checks, 'beam_detailing_strengths', return_value=strengths):
                    rows = gr.group_detailing_checks(view, state, {})
            found = {('detailing', row['id'], row.get('location', '')) for row in rows}
            # Confinement comes from the separate capacity completion producer.
            required = {key for key in detailing_keys(state) if key[1] != 'column.confinement_area_and_support'}
            # Missing declarations generate unknown rows with the same IDs, not invented passing evidence.
            self.assertEqual(found, required)

    def test_contexts_faces_and_gravity_scope_are_distinct(self):
        state = gr.seed_from_sections((26, 26, 6), (16, 30, 5), policy_id='system_three_story_v1')[0]
        expected = contract.expected_inventory(state)['exact']
        self.assertTrue(any('/member:' in loc for fam, name, loc in expected if fam == 'detailing'))
        for gid, design in state.designs.items():
            if design.is_column:
                self.assertIn(('detailing', 'column.end_hoop_spacing_dimension_bar', gid), expected)
                continue
            for face in ('top', 'bottom'):
                self.assertIn(('detailing', 'beam.minimum_steel', f'{gid}/{face}'), expected)
                self.assertNotIn(('detailing', 'beam.minimum_steel', f'{gid}/x/{face}'), expected)
            gravity = state.system_of(gid) == mg.SYSTEM_GRAVITY
            self.assertEqual(('detailing', 'beam.first_hoop', gid) in expected, not gravity)
            self.assertEqual(('detailing', 'beam.reversal_balance', f'{gid}/left') in expected, not gravity)
        # With no slab the member contexts legitimately collapse to one representative per group.
        with patch.object(sp, 'SLAB_THICKNESS_IN', None):
            exact = contract.expected_inventory(state)['exact']
            self.assertFalse(any('/member:' in loc for fam, name, loc in exact if fam == 'detailing'))

    def test_duplicate_malformed_unexpected_unknown_and_review_rows_cannot_replace_required_rows(self):
        expected, rows = contract.expected_inventory(self.state), complete_rows(self.state)
        key = ('detailing', 'beam.minimum_steel', cf.any_gid(self.state, 'beam') + '/top')
        required = next(r for r in rows if (r['family'], r['id'], r['location']) == key)
        self.assertEqual(contract.code_check_result(rows + [required], expected)['status'], 'invalid')
        remaining = [r for r in rows if r is not required]
        for replacement, status in ((dict(required, demand=2), 'invalid'),
                                    (dict(required, location=key[2] + '/bogus'), 'invalid'),
                                    (cf._row(*key, status='not_evaluated'), 'incomplete')):
            self.assertEqual(contract.code_check_result(remaining + [replacement], expected)['status'], status)
        from Design.SMRF_Common import awaiting_review
        reviewed = contract.compact_row('detailing', awaiting_review(key[1], 'fixture', 'fixture', 'invented', key[2]))
        self.assertEqual(contract.code_check_result(remaining + [reviewed], expected)['status'], 'invalid')
        self.assertTrue(contract.code_check_result(rows + [cf._row('detailing', 'fixture.note', '')], expected)['accepted'])

    def test_old_contract_and_evaluation_versions_are_not_current_evidence(self):
        for old in ('contract', 'evaluation'):
            record = cf.evaluation_record(self.state, rows=complete_rows(self.state))
            if old == 'contract':
                record['code_checks']['contract_version'] = 'grouped_code_check_contract_v3'
            else:
                record['version'] = 'grouped_candidate_evaluation_v4_cycle_resolution'
            self.assertEqual(contract.assess_evaluation(record)['classification'], 'invalid')

    def test_assessment_and_normalization_use_the_final_slab_after_global_restoration(self):
        record = cf.evaluation_record(self.state, rows=complete_rows(self.state))
        expected = contract.expected_inventory(self.state)
        with patch.object(sp, 'SLAB_THICKNESS_IN', None):
            self.assertEqual(contract.evaluation_inventory(record)['exact'], expected['exact'])
            self.assertEqual(contract.assess_evaluation(record)['classification'], 'accepted')
            self.assertIsNone(sp.SLAB_THICKNESS_IN)
            # Deleting a row from an additional context still fails with a different caller environment.
            key = next(key for key in expected['exact'] if key[0] == 'detailing' and '/member:' in key[2])
            rows = [r for r in record['code_checks']['rows'] if (r['family'], r['id'], r['location']) != key]
            changed = copy.deepcopy(record)
            changed['code_checks']['rows'] = rows
            assessed = contract.assess_evaluation(changed)
            self.assertFalse(assessed['accepted'])
            self.assertIn(dict(zip(('family', 'id', 'location'), key)), assessed['code_check']['missing'])
            del changed['evidence']['slab']['slab']['thickness_in']
            self.assertFalse(contract.assess_evaluation(changed)['accepted'])
            self.assertIsNone(sp.SLAB_THICKNESS_IN)


class EvidenceLoading(unittest.TestCase):
    def test_relabelled_v1_fixture_is_rejected_by_public_loader(self):
        fixture = Path(__file__).parent / 'fixtures/design_evidence_test_fixture.json'
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copyfile(fixture.with_name('design_evidence_fixture_review.txt'), tmp / 'design_evidence_fixture_review.txt')
            value = json.loads(fixture.read_text(encoding='utf-8'))
            value['kind'] = 'reviewed'
            path = tmp / 'profile.json'
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(dep.EvidenceProfileError, 'TEST FIXTURE'):
                dep.load(path)

    def test_both_formats_keep_fixture_warnings_and_require_explicit_test_mode(self):
        from test_slab_procedure import procedure_block
        fixture = Path(__file__).parent / 'fixtures/design_evidence_test_fixture.json'
        original = fixture.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copyfile(fixture.with_name('design_evidence_fixture_review.txt'), tmp / 'design_evidence_fixture_review.txt')
            for schema, production_kind in ((dep.SCHEMA, 'reviewed'), (dep.SCHEMA_V2, 'reviewed_procedure')):
                value = json.loads(original)
                value['schema'] = schema
                if schema == dep.SCHEMA_V2:
                    value['scope'].update(floor_sections_sha256=None, slab_thickness_in=None)
                    evidence = [{k: value['review']['evidence'][0][k] for k in ('path', 'sha256')}]
                    value['procedure'] = procedure_block(evidence=evidence, accepted=True, plan=value['slab_refinement'])
                for kind in ('test_fixture', production_kind):
                    value['kind'] = kind
                    path = tmp / 'profile.json'
                    path.write_text(json.dumps(value), encoding='utf-8')
                    for allow in (False, True):
                        with self.subTest(schema=schema, kind=kind, allow=allow):
                            if kind == 'test_fixture' and allow:
                                profile = dep.load(path, allow_test_fixture=True)
                                self.assertTrue(profile.asserted)
                                self.assertIn('TEST FIXTURE', profile.data['review']['reviewed_by'])
                            else:
                                with self.assertRaises(dep.EvidenceProfileError):
                                    dep.load(path, allow_test_fixture=allow)
        self.assertEqual(fixture.read_bytes(), original)


class ReinforcementCeiling(unittest.TestCase):
    def test_direct_selector_refuses_mutated_configuration_before_pricing(self):
        cfg = DesignConfig()
        cfg.rebar.rho_beam_practical_max = 0.020
        with patch.object(selection, '_trial_state', side_effect=AssertionError('pricing reached')):
            with self.assertRaisesRegex(ValueError, 'authorized'):
                selection.evaluate_beam_candidate(None, None, None, (1, 1), cfg)

    def test_all_shared_entry_points_reject_invalid_mutable_configuration(self):
        bad_values = (0.020, 0.0175 + 1e-9, math.nan, math.inf, -math.inf, 0, -0.01, True, '0.0175', None)
        for value in bad_values:
            cfg = DesignConfig()
            cfg.rebar.rho_beam_practical_max = value
            calls = {
                'legacy_ga': lambda: ga.search(None, cfg),
                'grouped_search': lambda: search.search(None, cfg),
                'evaluator': lambda: gd.evaluate_candidate(None, cfg),
                'selector': lambda: selection.evaluate_beam_candidate(None, None, None, (1, 1), cfg),
                'selection': lambda: selection.select_reinforcement(None, [], cfg),
                'beam_domain': lambda: selection.beam_cage_candidates(None, cfg),
                'jump_price': lambda: jumps.price_unit(None, (), {}, cfg),
                'jump_basis': lambda: jumps.ratio_ceiling(cfg),
                'v3': lambda: gen.GenerationRun(None, cfg, gen.GenerationConfig(), 'never_created', case_id='fixture', basis={}),
                'uniform_selector': lambda: redesign._beam_candidates(0, math.inf, 0, math.inf, cfg),
                'uniform_design': lambda: driver.design_structure(cfg, verbose=False),
                'saved_request': lambda: driver.design_request_identity(cfg),
            }
            for name, call in calls.items():
                with self.subTest(entry=name, value=value), self.assertRaisesRegex(ValueError, 'practical beam steel ratio ceiling'):
                    call()
        cfg = DesignConfig()
        del cfg.rebar.rho_beam_practical_max
        # Dataclass defaults remain visible after instance deletion; a genuinely absent declaration is rejected.
        from types import SimpleNamespace
        with self.assertRaises(ValueError):
            validate_beam_practical_ceiling(SimpleNamespace(rebar=SimpleNamespace()))

    def test_valid_ceiling_prices_cages_and_preserves_conservative_face_ratio(self):
        from Model import Member_Bar_Layers as layers
        with gf.frame():
            state = gf.uniform_state()
            gid = cf.any_gid(state, 'beam')
            design = state.designs[gid]
            with mg.installed(state):
                rows = layers.arrangement()[gid]
            strengths = checks.beam_flexural_strengths(design, rows)
            actual_ratio = max(design.top_bars, design.bot_bars) * sp.rebar_area(design.bar_size) / (
                design.b_in * min(strengths[s]['d_in'] for s in ('positive', 'negative')))
            for ceiling in (0.0175, 0.015, 0.0125):
                cfg = with_beam_practical_ratio(DesignConfig(), ceiling)
                priced = selection.evaluate_beam_candidate(state, gid, design, (1, 1), cfg)
                self.assertTrue(priced['admissible'], priced)
                self.assertEqual(jumps.ratio_ceiling(cfg)['practical_max'], ceiling)
                self.assertEqual(driver.design_request_identity(cfg)['policy']['rebar']['rho_beam_practical_max'], ceiling)
            cfg = with_beam_practical_ratio(DesignConfig(), actual_ratio * 0.99)
            self.assertEqual(selection.evaluate_beam_candidate(state, gid, design, (1, 1), cfg)['reason_code'], 'steel_ratio_practical')
            cfg.rebar.rho_beam_practical_max = actual_ratio * 1.01
            self.assertTrue(selection.evaluate_beam_candidate(state, gid, design, (1, 1), cfg)['admissible'])

    def _seven_no9_cage_is_rejected_at_both_requested_ceilings(self, basis, expected_ratio):
        import genetic_fixture as gfx
        from Model import Member_Bar_Layers as layers
        previous = sp.DETAIL_BASIS
        sp.DETAIL_BASIS = basis
        try:
            with gf.frame():
                state = gfx.seed_state()
                gid = 's01_03__beam_y__edge'
                cage = replace(state.designs[gid], bar_size=9, top_bars=7, bot_bars=7)
                trial = selection._trial_state(state, gid, cage)
                with mg.installed(trial):
                    strengths = checks.beam_flexural_strengths(cage, layers.arrangement()[gid])
                ratio = 7 * sp.rebar_area(9) / (cage.b_in * min(strengths[s]['d_in'] for s in ('positive', 'negative')))
                self.assertAlmostEqual(ratio, expected_ratio)
                cfg = with_beam_practical_ratio(DesignConfig(), 0.0175)
                self.assertEqual(selection.evaluate_beam_candidate(state, gid, cage, (1, 1), cfg)['reason_code'],
                                 'steel_ratio_practical')
                cfg.rebar.rho_beam_practical_max = 0.020
                with self.assertRaisesRegex(ValueError, 'authorized'):
                    selection.evaluate_beam_candidate(state, gid, cage, (1, 1), cfg)
        finally:
            sp.DETAIL_BASIS = previous

    def test_reviewed_seven_no9_cage_is_rejected_at_both_requested_ceilings(self):
        # The reviewed value is of the 3D physical route: crossing layers keep a clear spacing (2026-11-07).
        self._seven_no9_cage_is_rejected_at_both_requested_ceilings("physical_3d_v2", 0.019568690095846646)

    def test_conventional_seven_no9_cage_is_rejected_at_both_requested_ceilings(self):
        # Conventional detailing: the crossing layers rest in contact, so d is deeper and rho smaller, still above
        # the 0.0175 ceiling (2026-11-07).
        self._seven_no9_cage_is_rejected_at_both_requested_ceilings("conventional_detailing_v1", 0.018067846607669615)

    def test_controller_refuses_mutation_after_basis_freeze_before_start_or_resume(self):
        import genetic_fixture as gfx
        with gf.frame(), tempfile.TemporaryDirectory() as tmp:
            for ceiling in (0.0175, 0.0125):
                for mutated in (0.020, math.nan, 0.010):
                    cfg = with_beam_practical_ratio(DesignConfig(), ceiling)
                    out = Path(tmp) / 'unused'
                    controller = gen.GenerationRun(gfx.seed_state(), cfg, gen.GenerationConfig(), out,
                                                   case_id='fixture', basis={'fixture': True})
                    cfg.rebar.rho_beam_practical_max = mutated
                    for boundary in (controller.start, controller.resume, controller._evaluate_pending):
                        with self.assertRaisesRegex(ValueError, 'practical beam steel ratio ceiling'):
                            boundary()
                    self.assertFalse(out.exists())

    def test_old_contract_changes_controller_fingerprint_and_checkpoint_is_refused(self):
        import genetic_fixture as gfx
        with gf.frame(), tempfile.TemporaryDirectory() as tmp:
            seed, cfg = gfx.seed_state(), DesignConfig()
            kwargs = dict(case_id='fixture', basis={'fixture': True})
            out = Path(tmp) / 'unused'
            current = gen.GenerationRun(seed, cfg, gen.GenerationConfig(), out, **kwargs)
            with patch.object(contract, 'CONTRACT_VERSION', 'grouped_code_check_contract_v3'):
                old = gen.GenerationRun(seed, cfg, gen.GenerationConfig(), out, **kwargs)
            self.assertNotEqual(current.fingerprint, old.fingerprint)
            # Exercise the real resume fingerprint check without taking a lease or running a search.
            with patch.object(Path, 'read_text', return_value=json.dumps({'fingerprint': old.fingerprint})):
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    current._resume()


if __name__ == '__main__':
    unittest.main()
