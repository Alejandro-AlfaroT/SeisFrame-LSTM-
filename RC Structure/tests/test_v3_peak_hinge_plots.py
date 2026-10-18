"""Selection, axis mapping and artifact integrity; synthetic responses, no NTHA."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

RC = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('peak_plots', RC/'tools/multi_device_calibration/plot_peak_hinges.py')
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)
import test_weighted_seismic_collection as fixtures


def write_rows(path, records):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


class PeakHingePlots(unittest.TestCase):
    def test_damage_selection_and_deterministic_ties(self):
        records = [{'hinge_ele_tag': n, 'damage_ratio': d, 'member_type': 'beam_x'}
                   for n, d in ((4, .2), (9, .4), (7, .4))]
        selected, ties = plots.peak_row(records)
        self.assertEqual(selected['hinge_ele_tag'], 7)
        self.assertEqual(ties, [7, 9])
        for row in records: row['damage_ratio'] = 0
        self.assertEqual(plots.peak_row(records)[0]['hinge_ele_tag'], 4)

    def test_larger_column_damage_is_excluded_and_zero_beams_stay_beams(self):
        records = [{'hinge_ele_tag': 1, 'damage_ratio': .9, 'member_type': 'column'},
                   {'hinge_ele_tag': 3, 'damage_ratio': .2, 'member_type': 'beam_y'},
                   {'hinge_ele_tag': 2, 'damage_ratio': .2, 'member_type': 'beam_x'}]
        selected, ties = plots.peak_row(records)
        self.assertEqual(selected['hinge_ele_tag'], 2)
        self.assertEqual(ties, [2, 3])
        self.assertEqual(plots.peak_row(records, member_scope='all')[0]['hinge_ele_tag'], 1)
        for row in records[1:]: row['damage_ratio'] = 0
        selected, ties = plots.peak_row(records)
        self.assertEqual(selected['hinge_ele_tag'], 2)
        self.assertEqual(ties, [2, 3])

    def test_missing_beams_and_ambiguous_types_fail_without_column_fallback(self):
        with self.assertRaisesRegex(ValueError, 'No beam hinges'):
            plots.peak_row([{'hinge_ele_tag': 1, 'damage_ratio': .5, 'member_type': 'column'}])
        with self.assertRaisesRegex(ValueError, 'member type'):
            plots.peak_row([{'hinge_ele_tag': 1, 'damage_ratio': .5}])

    def test_invalid_damage_rows_rejected(self):
        for records in ([], [{'hinge_ele_tag': 1, 'damage_ratio': float('nan')}],
                        [{'hinge_ele_tag': 1, 'damage_ratio': -.1}],
                        [{'hinge_ele_tag': 1, 'damage_ratio': .1}]*2):
            with self.assertRaises(ValueError): plots.peak_row(records)

    def test_permuted_axis_mapping_and_mismatch_detection(self):
        arrays = {'time_sec': np.array([.01, .02]), 'rotation_rad': np.array([[.01, -.02], [.02, -.04]]),
                  'moment_kip_in': np.array([[99., -12.], [199., -24.]]),
                  'spring_hinge_element_tag': np.array([77, 77]), 'spring_axis_direction': np.array([6, 5]),
                  'spring_member_tag': np.array([8, 8]), 'spring_end_id': np.array([2, 2])}
        springs = [{'spring_index': i, 'hinge_element_tag': 77, 'local_axis': a, 'spring_local_direction': d,
                    'physical_member_tag': 8, 'end_id': 2} for i, a, d in ((1, 'y', 5), (0, 'z', 6))]
        backbone = {'hinge_ele_tag': 77, 'ele_tag': 8, 'end_id': 2, 'rot_abs_max': .04}
        selected = plots.select_histories(arrays, springs, backbone)
        np.testing.assert_array_equal(selected['y']['moment'], [-12, -24])
        np.testing.assert_array_equal(selected['z']['rotation'], [.01, .02])
        backbone['rot_abs_max'] = .05
        with self.assertRaisesRegex(ValueError, 'envelope'): plots.select_histories(arrays, springs, backbone)
        backbone['rot_abs_max'] = .04
        springs[0]['physical_member_tag'] = 9
        with self.assertRaisesRegex(ValueError, 'identities'): plots.select_histories(arrays, springs, backbone)

    def test_output_cannot_modify_analysis_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'device'
            with self.assertRaises(ValueError): plots.require_separate_output(root/'job'/'plots', [root])
            self.assertEqual(plots.require_separate_output(Path(tmp)/'plots', [root]), Path(tmp)/'plots')

    def test_verified_batch_exports_and_rejects_tampered_history(self):
        # Reuse collection's real scoring/artifact fixture, adding explicit spring histories.
        fixture = fixtures.PortableCollection()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root, job = fixture.roots[0], fixture.plan['jobs'][0]
        manifest = root/fixture.search()['trials'][0]['manifest']
        ntha = manifest.parent/'ntha'
        backbone = plots.rows(ntha/'hinge_backbone.csv')
        # The structure-wide maximum is a column; the plot must select a beam
        # while preserving that larger structure-wide calibration metric.
        for row in backbone:
            row.update(ele_tag='8', end_id='1', member_type='column' if row['hinge_ele_tag'] == '100' else 'beam_x')
            if row['hinge_ele_tag'] == '101':
                row.update(rot_abs_max='.005', plastic_rotation='.004', damage_ratio='.1', yielded='1')
        write_rows(ntha/'hinge_backbone.csv', backbone)
        with np.load(ntha/'response_arrays.npz') as compact:
            saved = {key: compact[key] for key in compact.files}
        saved['hinge_rotation'][1, 2] = .005
        np.savez(ntha/'response_arrays.npz', **saved)
        with np.load(ntha/'response_arrays.npz') as compact:
            tags = compact['hinge_tag_order']
            np.savez(ntha/'hinge_moment_rotation.npz', time_sec=compact['hinge_history_time'],
                     commit_count=compact['hinge_history_commit_count'], rotation_rad=compact['hinge_rotation'],
                     moment_kip_in=compact['hinge_moment'], spring_hinge_element_tag=np.repeat(tags, 2),
                     spring_axis_direction=np.tile([5, 6], len(tags)), spring_member_tag=np.full(20, 8),
                     spring_end_id=np.ones(20, dtype=int))
        springs = [{'spring_index': i, 'hinge_element_tag': int(tag), 'local_axis': 'y' if i % 2 == 0 else 'z',
                    'spring_local_direction': 5 + i % 2, 'physical_member_tag': 8, 'end_id': 1,
                    'end': 'i', 'member_type': 'column' if tag == 100 else 'beam_x', 'joint_node': 22}
                   for i, tag in enumerate(np.repeat(tags, 2))]
        write_rows(ntha/'hinge_springs.csv', springs)
        plots.runner.write_json(ntha/'hinge_moment_rotation_schema.json', {
            'schema_version': 'hinge_moment_rotation_v1', 'available': True,
            'status': {'failed': False, 'truncated': False, 'completed_steps': 2}})
        search = fixture.search()
        search['trials'][0].update(plots.runner.assess_trial(manifest, fixture.plan['score_policy']))
        search['best_trial'] = dict(search['trials'][0])
        plots.runner.write_json(manifest.parents[2]/'score.json', search['trials'][0])
        fixture.save_search(search)
        plan_path = fixture.base/'plan.json'
        plots.runner.write_json(plan_path, fixture.plan)
        output = fixture.base/'plots'
        before = {p.relative_to(root).as_posix(): plots.runner.digest(p) for p in root.rglob('*') if p.is_file()}
        with patch.object(plots.runner, 'validate_plan', side_effect=lambda p: p):
            self.assertEqual(plots.batch(plan_path, [root], output), 0)
        report = plots.runner.read_json(output/'device_01'/job['job_id']/'trial_01'/'peak_hinge.json')
        self.assertEqual(report['hinge_element_tag'], 101)
        self.assertAlmostEqual(report['damage_ratio'], .1)
        self.assertEqual(report['selection_scope'], 'beam')
        self.assertEqual(report['structure_peak_hinge_element_tag'], 100)
        self.assertAlmostEqual(report['structure_peak_hinge_damage_ratio'], .3)
        self.assertEqual(report['score'], search['trials'][0]['score'])
        self.assertTrue(report['identity']['context']['selected_trial'])
        self.assertFalse(report['training_release'])
        self.assertEqual(report['samples'], 2)
        after = {p.relative_to(root).as_posix(): plots.runner.digest(p) for p in root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        with patch.object(plots.runner, 'validate_plan', side_effect=lambda p: p):
            with self.assertRaisesRegex(ValueError, 'selection scope'):
                plots.batch(plan_path, [root], output, member_scope='all')
        (ntha/'hinge_moment_rotation.npz').write_bytes(b'tampered')
        with patch.object(plots.runner, 'validate_plan', side_effect=lambda p: p):
            self.assertEqual(plots.batch(plan_path, [root], output), 1)
        self.assertFalse(plots.rows(output/'peak_hinge_summary.csv'))
        errors = plots.runner.read_json(output/'peak_hinge_plots.json')['errors']
        self.assertIn('Artifact changed', str(errors))


if __name__ == '__main__':
    unittest.main()
