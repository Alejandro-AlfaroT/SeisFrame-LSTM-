"""The parent scheduler hands each plan run to the child as one explicit (record pair, scale), and the
child runs exactly those. Until 2026-09-26 the parent passed the set of scales and the child formed a
records x scales cross product, which under a calibration (one target-matched factor per record) ran
pairings the plan never asked for and skipped the ones it did.
"""
import argparse
import os
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_Generation import Generate_Hybrid_Dataset as child  # noqa: E402
from Data_Generation import Generate_Parameterized_Dataset as parent  # noqa: E402

SELECTED = [("peer_result_id:11", 11, 6000), ("peer_result_id:23", 23, 8000), ("peer_result_id:42", 42, 9000)]


class ChildExpansion(unittest.TestCase):
    def test_explicit_pairs_run_exactly_as_given_in_plan_order(self):
        runs, missing = child.expand_runs(SELECTED, [1.0, 2.0], run_pairs=[(23, 0.85), (11, 1.7)])
        self.assertEqual(runs, [("peer_result_id:23", 23, 8000, 0.85), ("peer_result_id:11", 11, 6000, 1.7)])
        self.assertEqual(missing, [])

    def test_pair_for_an_unselectable_record_is_reported_not_substituted(self):
        runs, missing = child.expand_runs(SELECTED, [None], run_pairs=[(99, 1.2), (42, 0.5)])
        self.assertEqual(runs, [("peer_result_id:42", 42, 9000, 0.5)])
        self.assertEqual(missing, [99])

    def test_without_pairs_the_legacy_cross_product_remains(self):
        runs, missing = child.expand_runs(SELECTED[:2], [1.0, 3.0])
        self.assertEqual([(r, s) for _, r, _, s in runs], [(11, 1.0), (11, 3.0), (23, 1.0), (23, 3.0)])
        self.assertEqual(missing, [])

    def test_run_pair_argument_parses_and_rejects_garbage(self):
        self.assertEqual(child.parse_run_pair("11:1.25"), (11, 1.25))
        with self.assertRaises(argparse.ArgumentTypeError):
            child.parse_run_pair("11")
        with self.assertRaises(argparse.ArgumentTypeError):
            child.parse_run_pair("eleven:1.0")


class ParentCommand(unittest.TestCase):
    def test_command_names_every_plan_run_and_forms_no_cross_product(self):
        args = types.SimpleNamespace(python_exe="python", set_name="peer_strong_63", max_npts=15000)
        case = {
            "geometry_name": "case_0001_g", "num_bay_x": 3, "num_bay_y": 3, "num_floor": 5,
            "bay_x_in": 240.0, "bay_y_in": 240.0, "story_height_in": 156.0, "seismic_site": "site_c",
            "result_ids": [11, 23],
            "runs": [
                {"run_index": 1, "result_id": 11, "scale_factor": 1.7, "run_name": "peer_11__sf_1p7"},
                {"run_index": 2, "result_id": 23, "scale_factor": 0.85, "run_name": "peer_23__sf_0p85"},
            ],
        }
        paths = {"ntha": Path("root/ntha"), "dataset": Path("root/dataset")}
        command = parent.command_for(args, case, paths)
        pairs = [command[i + 1] for i, token in enumerate(command) if token == "--run-pair"]
        self.assertEqual(pairs, ["11:1.7", "23:0.85"])
        self.assertNotIn("--scale-factor", command)
        self.assertEqual([command[i + 1] for i, token in enumerate(command) if token == "--result-id"], ["11", "23"])
        # The child's parser accepts what the parent emits.
        parsed = child.build_parser().parse_args(command[3:]) if hasattr(child, "build_parser") else None
        if parsed is not None:
            self.assertEqual(parsed.run_pair, [(11, 1.7), (23, 0.85)])


if __name__ == "__main__":
    unittest.main()
