"""School entrypoint failure boundaries; no structural analysis."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

RC = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('v3_' + name, RC / 'pilots/v2WeightedCalibrationV3' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


school = load('school')
setup = load('verify_setup')


class LaunchBoundaries(unittest.TestCase):
    def test_frozen_inventory_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(school, 'RC', Path(tmp)), patch.object(school, 'call') as call:
            path = Path(tmp) / school.PILOT / 'prepared' / school.DEFAULT_RUN / 'plan.json'
            path.parent.mkdir(parents=True)
            path.write_text('frozen')
            for action in ('inventory', 'prepare'):
                with self.assertRaises(FileExistsError):
                    school.main([action])
            call.assert_not_called()
            self.assertEqual(path.read_text(), 'frozen')

    def test_exclusions_block_builder_until_explicitly_selected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(school, 'RC', Path(tmp)):
            path = Path(tmp) / school.PILOT / 'prepared' / school.DEFAULT_RUN / 'design_inventory.json'
            path.parent.mkdir(parents=True)
            path.write_text('{"eligible_count": 99, "excluded_count": 1}')
            with patch.object(school, 'call') as call:
                with self.assertRaisesRegex(ValueError, 'Excluded cases'):
                    school.main(['prepare'])
                self.assertFalse(any('Build_Weighted' in c.args[0][0] for c in call.call_args_list))
            with patch.object(school, 'call') as call:
                school.main(['prepare', '--allow-excluded'])
                build = call.call_args_list[-1].args[0]
                self.assertIn('Build_Weighted', build[0])
                self.assertTrue(build[build.index('--case-manifest') + 1].endswith('/eligible_cases.json'))

    def test_failed_canary_blocks_remainder_on_fresh_and_resume(self):
        for extra in ([], ['--resume']):
            commands = []
            def execute(command, **kwargs):
                commands.append(command)
                if '--first-job-only' in command:
                    raise RuntimeError('unmet target')
            with patch.object(school, 'call', side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, 'unmet target'):
                    school.main(['run', '--device', '1', *extra])
            self.assertEqual(len(commands), 4)
            self.assertIn('--first-job-only', commands[2])
            self.assertEqual('--resume' in commands[2], bool(extra))
            self.assertEqual(commands[-1][0], school.PLOTTER)

    def test_preflight_never_invokes_search(self):
        with patch.object(school, 'call') as call:
            school.main(['run', '--device', '4', '--preflight-only'])
        commands = [c.args[0] for c in call.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn('--preflight-only', commands[-1])
        self.assertFalse(any('--first-job-only' in c or '--resume' in c for c in commands))

    def test_run_name_path_escape_rejected_before_setup(self):
        with patch.object(school, 'call') as call:
            with self.assertRaises(ValueError):
                school.main(['prepare', '--run-name', '../outside'])
            call.assert_not_called()

    def test_first_search_only_exports_without_running_remainder(self):
        with patch.object(school, 'call') as call:
            school.main(['run', '--device', '2', '--first-job-only'])
        commands = [c.args[0] for c in call.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertIn('--first-job-only', commands[2])
        self.assertEqual(commands[-1], [school.PLOTTER, '--plan',
                         f'{school.PILOT}/prepared/wcalV3_r1/plan.json', '--device', '2'])

    def test_plot_failure_does_not_hide_original_analysis_failure(self):
        def execute(command, **kwargs):
            if '--first-job-only' in command:
                raise RuntimeError('analysis failure')
            if command[0] == school.PLOTTER:
                raise RuntimeError('plot failure')
        with patch.object(school, 'call', side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, 'analysis failure'):
                school.main(['run', '--device', '1'])

    def test_successful_analysis_with_failed_plots_reports_failure(self):
        def execute(command, **kwargs):
            if command[0] == school.PLOTTER:
                raise RuntimeError('plot failure')
        with patch.object(school, 'call', side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, 'plot failure'):
                school.main(['run', '--device', '1'])

    def test_plot_regeneration_never_starts_analysis(self):
        with patch.object(school, 'call') as call:
            school.main(['plots', '--device', '3'])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[0][0], school.PLOTTER)

    def test_launcher_hash_tolerates_git_line_endings_but_detects_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'launcher.py'
            path.write_bytes(b'one\ntwo\n')
            expected = setup.launcher_digest(path)
            path.write_bytes(b'\xef\xbb\xbfone\r\ntwo\r\n')
            self.assertEqual(setup.launcher_digest(path), expected)
            path.write_bytes(b'one\r\nchanged\r\n')
            self.assertNotEqual(setup.launcher_digest(path), expected)


if __name__ == '__main__':
    unittest.main()
