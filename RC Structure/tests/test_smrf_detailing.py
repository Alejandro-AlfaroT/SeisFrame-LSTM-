import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Detailing import evaluate_detailing, select_transverse_geometry
from Design.SMRF_Common import make_check, summarize_checks
from Design.Section_Design import beam_ladder, validate_rung


def example():
    common = {"b_in": 24, "h_in": 24, "fc_ksi": 5, "clear_cover_in": 1.5,
              "stirrup_db_in": 0.5, "bar_db_in": 1, "bar_area_in2": 0.79,
              "n_top": 4, "n_bottom": 4, "aggregate_size_in": 0.75,
              "hoop_spacing_in": 3, "end_zone_length_in": 48}
    beam = dict(common, b_in=12, n_top=3, n_bottom=3, continuous_top_bars=3,
                continuous_bottom_bars=3, first_hoop_distance_in=2)
    for key in ("positive_left", "negative_left", "positive_right", "negative_right", "positive_min_along", "negative_min_along"):
        beam[f"mn_{key}_kipin"] = 1000
    return {"material": {"fy_ksi": 60, "normalweight": True},
            "geometry": {"span_x_in": 150, "span_y_in": 180}, "beam": beam,
            "column": dict(common, n_side_per_face=2, clear_height_in=120, hx_in=12)}


def find(checks, name, location=None):
    return next(c for c in checks if c["id"] == name and (location is None or c["location"] == location))


class DetailingTests(unittest.TestCase):
    def test_transverse_selector_rounds_down_and_does_not_invent_cage(self):
        data = example()
        data["beam"]["hoop_spacing_in"] = 6
        data["column"]["hoop_spacing_in"] = 6
        before = copy.deepcopy(data)
        selected = select_transverse_geometry(data)
        self.assertEqual(data, before)
        self.assertEqual(selected["stage"], "scalar_geometry_only")
        self.assertFalse(selected["full_cage_verified"])
        # Beam d=21.5, d/4=5.375: next permitted integer spacing is 5.
        self.assertEqual(selected["beam"]["hoop_spacing_in"], 5)
        self.assertEqual(selected["beam"]["end_zone_length_in"], 48)
        self.assertEqual(selected["beam"]["first_hoop_distance_in"], 2)
        self.assertEqual(selected["column"]["hoop_spacing_in"], 4)
        self.assertEqual(selected["column"]["end_zone_length_in"], 24)
        for name in ("beam", "column"):
            self.assertFalse(selected[name]["full_cage_verified"])
            for key in ("hx_in", "continuous_top_bars", "confinement_area_in2", "joint_shear"):
                self.assertNotIn(key, selected[name])

    def test_transverse_selector_preserves_tighter_requested_spacing(self):
        data = example()
        data["beam"]["hoop_spacing_in"] = 3.8
        data["beam"]["first_hoop_distance_in"] = 1.25
        result = select_transverse_geometry(data)
        self.assertEqual(result["beam"]["hoop_spacing_in"], 3)
        self.assertEqual(result["beam"]["first_hoop_distance_in"], 1.25)
        self.assertEqual(result["column"]["hoop_spacing_in"], 3)

    def test_transverse_selector_rounds_on_minimum_anchored_grid(self):
        data = example()
        for name in ("beam", "column"):
            data[name]["hoop_spacing_in"] = 6
        result = select_transverse_geometry(data, minimum_spacing_in=2.75, spacing_step_in=.5)
        self.assertEqual(result["beam"]["hoop_spacing_in"], 5.25)
        self.assertEqual(result["column"]["hoop_spacing_in"], 3.75)
        self.assertLessEqual(result["beam"]["hoop_spacing_in"], result["beam"]["code_spacing_bound_in"])

    def test_transverse_selector_smaller_longitudinal_bars_reduce_spacing(self):
        data = example()
        data["beam"].update(hoop_spacing_in=6, bar_db_in=.5)
        data["column"].update(hoop_spacing_in=6, bar_db_in=.5)
        result = select_transverse_geometry(data)
        self.assertEqual(result["beam"]["hoop_spacing_in"], 3)
        self.assertEqual(result["column"]["hoop_spacing_in"], 3)

    def test_column_end_zone_uses_both_dimensions_clear_height_and_18_floor(self):
        for b, h, clear_height, expected in ((20, 30, 120, 30), (12, 12, 60, 18), (24, 24, 240, 40)):
            data = example()
            data["column"].update(b_in=b, h_in=h, clear_height_in=clear_height)
            self.assertEqual(select_transverse_geometry(data)["column"]["end_zone_length_in"], expected)

    def test_transverse_selector_rejects_exhausted_grid_without_fallback(self):
        cases = []
        data = example()
        data["beam"]["h_in"] = 12  # d/4 below minimum spacing 3.
        cases.append(data)
        data = example()
        data["column"]["b_in"] = 10  # dimension/4 below 3.
        cases.append(data)
        data = example()
        data["beam"]["hoop_spacing_in"] = 2.9  # Never increase existing spacing.
        cases.append(data)
        for data in cases:
            with self.assertRaisesRegex(ValueError, "No allowed"):
                select_transverse_geometry(data)

    def test_transverse_selector_rejects_invalid_scope_policy_and_geometry(self):
        for kwargs in ({"minimum_spacing_in": 0}, {"spacing_step_in": 0},
                       {"spacing_step_in": True}, {"minimum_spacing_in": math.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_transverse_geometry(example(), **kwargs)
        for field, value in (("fy_ksi", 80), ("normalweight", False)):
            data = example()
            data["material"][field] = value
            with self.assertRaises(ValueError):
                select_transverse_geometry(data)
        for name, key, value in (("beam", "bar_db_in", math.nan), ("column", "clear_height_in", None),
                                 ("beam", "clear_cover_in", 7), ("column", "hoop_spacing_in", True)):
            data = example()
            data[name][key] = value
            with self.assertRaises(ValueError):
                select_transverse_geometry(data)

    def test_known_spacing_violation_is_visible_without_end_zone_evidence(self):
        data = example()
        data["beam"].pop("end_zone_length_in")
        data["column"].pop("end_zone_length_in")
        data["column"].pop("hx_in")
        data["beam"]["hoop_spacing_in"] = 6
        data["column"].update(b_in=20, hoop_spacing_in=6)
        result = evaluate_detailing(data)
        self.assertEqual(find(result, "beam.end_hoop_spacing")["status"], "fail")
        self.assertEqual(find(result, "column.end_hoop_spacing_dimension_bar")["status"], "fail")
        self.assertEqual(find(result, "column.end_hoop_spacing")["status"], "fail")
        self.assertEqual(find(result, "beam.end_zone_length")["status"], "not_evaluated")
        self.assertEqual(find(result, "column.end_zone_length")["status"], "not_evaluated")

    def test_first_hoop_failure_is_independent_of_missing_spacing_and_zone(self):
        data = example()
        data["beam"].pop("hoop_spacing_in")
        data["beam"].pop("end_zone_length_in")
        data["beam"]["first_hoop_distance_in"] = 2.5
        self.assertEqual(find(evaluate_detailing(data), "beam.first_hoop")["status"], "fail")

    def test_conservative_column_spacing_pass_is_not_supported_bar_verification(self):
        data = example()
        data["column"].pop("hx_in")
        data["column"]["hoop_spacing_in"] = 4
        result = evaluate_detailing(data)
        self.assertEqual(find(result, "column.end_hoop_spacing")["status"], "pass")
        self.assertEqual(find(result, "column.supported_bar_distance_basic")["status"], "not_evaluated")
        self.assertEqual(find(result, "column.confinement_area_and_support")["status"], "not_evaluated")

    def test_column_spacing_between_conservative_and_possible_limits_is_unknown_without_hx(self):
        data = example()
        data["column"].pop("hx_in")
        data["column"]["hoop_spacing_in"] = 5
        result = evaluate_detailing(data)
        self.assertEqual(find(result, "column.end_hoop_spacing_dimension_bar")["status"], "pass")
        self.assertEqual(find(result, "column.end_hoop_spacing")["status"], "not_evaluated")
        data["column"]["hx_in"] = 8
        self.assertEqual(find(evaluate_detailing(data), "column.end_hoop_spacing")["status"], "pass")

    def test_invalid_supported_bar_distance_is_not_hidden_by_small_spacing(self):
        data = example()
        data["column"]["hx_in"] = 15
        result = evaluate_detailing(data)
        self.assertEqual(find(result, "column.end_hoop_spacing")["status"], "pass")
        self.assertEqual(find(result, "column.supported_bar_distance_basic")["status"], "fail")

    def test_known_dimensions_pass_but_missing_cage_never_certifies(self):
        checks = evaluate_detailing(example())
        self.assertEqual(find(checks, "beam.clear_span", "x")["status"], "pass")
        self.assertEqual(find(checks, "beam.end_hoop_spacing")["status"], "pass")
        self.assertFalse(summarize_checks(checks)["accepted"])
        self.assertEqual(find(checks, "column.confinement_area_and_support")["status"], "not_evaluated")

    def test_shorter_axis_governs_clear_span(self):
        data = example()
        data["geometry"]["span_x_in"] = 100
        checks = evaluate_detailing(data)
        self.assertEqual(find(checks, "beam.clear_span", "x")["status"], "fail")
        self.assertEqual(find(checks, "beam.clear_span", "y")["status"], "pass")

    def test_minimum_beam_width_is_min_not_max(self):
        # ACI 318-19 18.6.2.1(b) asks for the lesser of 0.3h and 10 in, so a 7 in web on a 20 in beam satisfies
        # the Code check; the design basis of 2026-10-14 (web at least half the depth, beam deeper than wide)
        # is a separate check and a separate rung rule.
        data = example()
        data["beam"].update(b_in=7, h_in=20)
        checks = evaluate_detailing(data)
        self.assertEqual(find(checks, "beam.minimum_width")["status"], "pass")
        self.assertEqual(find(checks, "beam.width_to_depth_design_basis")["status"], "fail")
        self.assertEqual(find(checks, "beam.depth_to_width_design_basis")["status"], "pass")
        with self.assertRaises(ValueError):
            validate_rung((7, 20, 5), "beam")
        with self.assertRaises(ValueError):
            validate_rung((5, 20, 5), "beam")
        with self.assertRaises(ValueError):
            validate_rung((18, 20, 5), "beam")                 # 0.9: not deep enough for the design basis
        validate_rung((10, 20, 5), "beam")
        validate_rung((14, 20, 5), "beam")

    def test_empty_ladder_does_not_fabricate_a_legal_rung(self):
        with self.assertRaises(ValueError):
            beam_ladder(span_in=20, story_height_in=120)

    def test_column_ratio_maximum_is_six_percent(self):
        data = example()
        data["column"].update(b_in=12, h_in=12, n_top=6, n_bottom=6, n_side_per_face=4)
        checks = evaluate_detailing(data)
        self.assertEqual(find(checks, "column.maximum_rho")["status"], "fail")

    def test_effective_depth_uses_clear_cover_plus_hoop_and_half_bar(self):
        checks = evaluate_detailing(example())
        result = find(checks, "beam.clear_span", "x")
        # a 1.0 in bar with a 0.5 in hoop: 1.5 + the #5 layout hoop + 0.5 = 2.625 in, taken up to the 3.0 in grid step
        self.assertEqual(result["capacity"], 4 * (24 - 3.0))

    def test_bar_fit_uses_actual_diameter_and_aggregate(self):
        data = example()
        data["beam"]["n_top"] = 8
        self.assertEqual(find(evaluate_detailing(data), "beam.bar_clear_spacing", "top")["status"], "fail")
        data = example()
        data["beam"].pop("aggregate_size_in")
        self.assertEqual(find(evaluate_detailing(data), "beam.bar_fit_complete")["status"], "not_evaluated")

    def test_longitudinal_moment_balance_and_first_hoop(self):
        data = example()
        data["beam"]["mn_positive_left_kipin"] = 499
        data["beam"]["first_hoop_distance_in"] = 2.1
        checks = evaluate_detailing(data)
        self.assertEqual(find(checks, "beam.reversal_balance", "left")["status"], "fail")
        self.assertEqual(find(checks, "beam.first_hoop")["status"], "fail")

    def test_missing_and_nonfinite_data_fail_closed(self):
        for data in ({}, {"geometry": {}}, dict(example(), material={"fy_ksi": 80, "normalweight": True})):
            self.assertFalse(summarize_checks(evaluate_detailing(data))["accepted"])
        for number in (None, math.nan, math.inf, True):
            result = make_check("invalid", "test", number, 1)
            self.assertEqual(result["status"], "not_evaluated")
        self.assertFalse(summarize_checks([])["accepted"])
        c = make_check("same", "test", 1, 1)
        self.assertFalse(summarize_checks([c, copy.deepcopy(c)])["accepted"])


if __name__ == "__main__":
    unittest.main()
