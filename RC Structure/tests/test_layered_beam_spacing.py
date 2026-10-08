"""Regressions for layered cages rejected by the legacy single-row check."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design import SMRF_Beam_Spacing as spacing
from Design.SMRF_Common import make_check, summarize_checks
from Design.SMRF_Detailing import evaluate_detailing
from Design.SMRF_Qualification import detailing_inputs

FIXTURES = json.loads((Path(__file__).parent / "fixtures/beam_spacing_saved_cases.json").read_text())


def inputs(case="case_0069"):
    return detailing_inputs(copy.deepcopy(FIXTURES[case]))


def evaluate(data):
    return spacing.evaluate_layered_beam_spacing(data["beam"], data["column"])


def horizontal(checks):
    return [c for c in checks if c["id"] == "beam.bar_clear_spacing"]


def recenter(row):
    row["layers"] = len(row["per_layer"])
    row["centroid_in"] = sum(n * o for n, o in zip(row["per_layer"], row["offsets_in"])) / sum(row["per_layer"])


class LayeredBeamSpacingTests(unittest.TestCase):
    def test_saved_regressions_pass_all_rows_and_do_not_mutate_inputs(self):
        for cid in FIXTURES:
            with self.subTest(case=cid):
                data = inputs(cid)
                before = copy.deepcopy(data)
                checks = evaluate(data)
                self.assertTrue(summarize_checks(checks)["accepted"], checks)
                self.assertEqual(data, before)
                self.assertEqual(len(horizontal(checks)), 2)
                integrated = horizontal(evaluate_detailing(data))
                self.assertEqual([c["status"] for c in integrated], ["pass", "pass"])

    def test_different_axis_splits_are_retained(self):
        check = horizontal(evaluate(inputs("case_0027")))[0]
        rows = check["details"]["rows"]
        self.assertEqual([r["bar_count"] for r in rows["x"]], [4, 2])
        self.assertEqual([r["bar_count"] for r in rows["y"]], [3, 3])

    def test_legacy_single_row_failures_are_not_relaxed(self):
        expected = {"case_0007": .475, "case_0027": .8, "case_0069": .9342, "case_0085": 1.2277777777777779}
        for cid, old_clear in expected.items():
            with self.subTest(case=cid):
                data = inputs(cid)
                data["beam"].pop("bar_stacking")
                checks = horizontal(evaluate_detailing(data))
                self.assertEqual([c["status"] for c in checks], ["fail", "fail"])
                self.assertAlmostEqual(checks[0]["demand"], old_clear)

    def test_explicit_none_keeps_true_single_row_failure(self):
        data = inputs("case_0007")
        stack = data["beam"]["bar_stacking"]
        stack["convention"] = "none"
        for faces in stack["layers"].values():
            for row in faces.values():
                row.update(per_layer=[11], offsets_in=[2.625])
                recenter(row)
        self.assertEqual([c["status"] for c in horizontal(evaluate(data))], ["fail", "fail"])

    def test_bottom_counts_are_independent(self):
        data = inputs()
        data["beam"]["n_bottom"] = 6
        for faces in data["beam"]["bar_stacking"]["layers"].values():
            row = faces["bottom"]
            row.update(per_layer=[6], offsets_in=row["offsets_in"][:1])
            recenter(row)
        checks = evaluate(data)
        self.assertTrue(summarize_checks(checks)["accepted"], checks)
        bottom = next(c for c in horizontal(checks) if c["location"] == "bottom")
        self.assertTrue(all(len(rows) == 1 for rows in bottom["details"]["rows"].values()))

    def test_too_few_lanes_is_a_failure_not_a_partial_pass(self):
        data = inputs()
        data["beam"]["b_in"] = 12
        checks = evaluate(data)
        self.assertTrue(any(c["id"] == "beam.bar_layer_capacity" and c["status"] == "fail" for c in checks))
        self.assertTrue(all(c["status"] == "not_evaluated" for c in horizontal(checks)))

    def test_aggregate_can_make_declared_rows_impossible(self):
        data = inputs()
        data["beam"]["aggregate_size_in"] = 3
        self.assertFalse(summarize_checks(evaluate(data))["accepted"])

    def test_raw_equality_roundoff_passes_only_locally(self):
        checks = horizontal(evaluate(inputs("case_0085")))
        self.assertLess(checks[0]["demand"], 1.27)
        self.assertEqual(checks[0]["status"], "pass")
        self.assertEqual(checks[0]["details"]["numerical_tolerance_in"], 1e-9)
        self.assertEqual(make_check("strict", "unchanged", 1.27 - 1e-12, 1.27, ">=")["status"], "fail")

    def test_actual_underspacing_is_rejected(self):
        original = spacing.beam_bars_threading
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            points = result["x"]["threaded_positions_in"]
            points[1] = points[0] + 2 * 1.128 - 1e-5
            return result
        with mock.patch.object(spacing, "beam_bars_threading", side_effect=corrupt):
            self.assertTrue(any(c["status"] == "fail" for c in horizontal(evaluate(inputs()))))

    def test_positions_are_independently_checked_against_beam_cover(self):
        original = spacing.beam_bars_threading
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            result["x"]["threaded_positions_in"][0] = 0.0
            return result
        with mock.patch.object(spacing, "beam_bars_threading", side_effect=corrupt):
            checks = evaluate(inputs())
        self.assertTrue(any(c["id"] == "beam.bar_row_cover" and c["status"] == "fail" for c in checks))

    def test_malformed_metadata_fails_closed_in_public_evaluator(self):
        modifications = []
        for replacement in (None, {}, [], {"max_layers": 2}):
            modifications.append(lambda d, value=replacement: d["beam"].__setitem__("bar_stacking", value))
        def change(key, value):
            return lambda d: d["beam"]["bar_stacking"].__setitem__(key, value)
        modifications.extend([change("max_layers", 1), change("layer_order", "unknown"), change("convention", "unknown")])
        modifications.append(lambda d: d["beam"]["bar_stacking"]["layers"].pop("y"))
        modifications.append(lambda d: d["beam"]["bar_stacking"]["layers"]["x"].pop("bottom"))
        def row_change(key, value):
            return lambda d: d["beam"]["bar_stacking"]["layers"]["x"]["top"].__setitem__(key, value)
        modifications.extend([row_change("per_layer", [6, 4]), row_change("layers", 2.5),
                              row_change("centroid_in", float("nan")), row_change("centroid_in", True),
                              row_change("offsets_in", [True, 7.201]), row_change("offsets_in", [float("nan"), 7.201])])
        for index, mutate in enumerate(modifications):
            with self.subTest(mutation=index):
                data = inputs()
                mutate(data)
                checks = evaluate_detailing(data)
                self.assertTrue(any(c["id"] == "beam.bar_fit_complete" and c["status"] == "not_evaluated" for c in checks))
                self.assertEqual(horizontal(checks), [])

    def test_consistent_centroid_does_not_hide_wrong_layer_elevation(self):
        data = inputs()
        row = data["beam"]["bar_stacking"]["layers"]["x"]["top"]
        row["offsets_in"][1] = row["offsets_in"][0] + .5
        recenter(row)
        with self.assertRaisesRegex(ValueError, "stacking geometry"):
            evaluate(data)

    def test_too_shallow_section_does_not_accept_opposing_rows(self):
        data = inputs()
        data["beam"]["h_in"] = 18
        with self.assertRaises(ValueError):
            evaluate(data)


if __name__ == "__main__":
    unittest.main()
