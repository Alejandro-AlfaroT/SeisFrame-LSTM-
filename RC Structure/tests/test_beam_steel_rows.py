"""Regressions for layered face totals, actual depths and Grade 60/80 qualification.

Small hand-calculable sections reproduce the false failures in smoke cases
0001 and 0003. These are implementation tests, not structural acceptance.
"""
import copy
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import Structure_Parameters as sp
from Design import Group_Checks as gc, Grouped_Record as gr
from Design.SMRF_Detailing import evaluate_detailing
from Model import Member_Groups as mg
from test_smrf_detailing import example, find


def rows(counts, offsets):
    return {"per_layer": counts, "offsets_in": offsets, "layers": len(counts),
            "centroid_in": sum(n * offset for n, offset in zip(counts, offsets)) / sum(counts)}


def layered_example():
    data = example()
    data["beam"].update(b_in=14., h_in=28., fc_ksi=8., bar_db_in=.875, bar_area_in2=.60,
                        n_top=4, n_bottom=4, bar_position_grid_in=.5,
                        bar_rows={"top": rows([2, 2], [5., 11.5]), "bottom": rows([2, 2], [3., 7.])})
    return data


class LayeredSteelTests(unittest.TestCase):
    def test_case0001_counts_all_bars_and_keeps_row_width_fit_separate(self):
        data = layered_example()
        original = copy.deepcopy(data)
        result = evaluate_detailing(data)
        for face, depth in (("top", 19.75), ("bottom", 23.)):
            item = find(result, "beam.minimum_steel", face)
            self.assertEqual(item["status"], "pass")
            self.assertAlmostEqual(item["demand"], 2.4)  # four No. 7, not two in the widest row
            self.assertAlmostEqual(item["capacity"], 3 * math.sqrt(8000) / 60000 * 14 * depth)
            self.assertEqual(item["details"]["sections"]["group"]["d_in"], depth)
            spacing = find(result, "beam.bar_clear_spacing", face)
            self.assertEqual(spacing["status"], "pass")
            self.assertEqual(spacing["details"]["bars_per_layer"], [2, 2])
        self.assertEqual(data, original)

    def test_case0003_count_fix_alone_is_insufficient(self):
        data = layered_example()
        data["beam"].update(b_in=16., h_in=30., bar_db_in=.625, bar_area_in2=.31, n_top=6, n_bottom=6,
                            bar_rows={"top": rows([3, 3], [9., 12.5]), "bottom": rows([3, 3], [6., 9.])})
        result = evaluate_detailing(data)
        for face, d in (("top", 19.25), ("bottom", 22.5)):
            item = find(result, "beam.minimum_steel", face)
            self.assertEqual(item["status"], "pass")
            self.assertAlmostEqual(item["demand"], 1.86)
            self.assertAlmostEqual(item["capacity"], 3 * math.sqrt(8000) / 60000 * 16 * d)
        legacy = copy.deepcopy(data)
        del legacy["beam"]["bar_rows"]
        self.assertEqual(find(evaluate_detailing(legacy), "beam.minimum_steel", "top")["status"], "fail")

    def test_grade_is_explicit_even_when_current_model_has_other_grade(self):
        for grade, cap in ((60., .025), (80., .02)):
            with self.subTest(grade=grade), patch.object(sp, "FY_KSI", 140. - grade):
                data = layered_example()
                data["material"]["fy_ksi"] = grade
                result = evaluate_detailing(data)
                item = find(result, "beam.minimum_steel", "top")
                self.assertAlmostEqual(item["capacity"], 3 * math.sqrt(8000) / (grade * 1000) * 14 * 19.75)
                self.assertEqual(item["details"]["sections"]["group"]["fy_ksi"], grade)
                self.assertEqual(find(result, "beam.maximum_rho", "top")["capacity"], cap)

    def test_total_steel_also_controls_maximum_ratio(self):
        data = layered_example()
        data["beam"].update(b_in=12., h_in=24., bar_db_in=1.41, bar_area_in2=1.56,
                            bar_rows={face: rows([2, 2], [3., 9.]) for face in ("top", "bottom")})
        result = evaluate_detailing(data)
        item = find(result, "beam.maximum_rho", "top")
        self.assertEqual(item["status"], "fail")
        self.assertAlmostEqual(item["demand"], 6.24 / (12 * 18))

    def test_real_minimum_failure_is_not_bypassed(self):
        data = layered_example()
        data["beam"].update(n_top=2, n_bottom=2, bar_area_in2=.31, bar_db_in=.625,
                            bar_rows={face: rows([2], [3.]) for face in ("top", "bottom")})
        self.assertEqual(find(evaluate_detailing(data), "beam.minimum_steel", "top")["status"], "fail")

    def test_one_bar_per_row_still_has_two_total_bars(self):
        data = layered_example()
        data["beam"].update(n_top=2, n_bottom=2,
                            bar_rows={face: rows([1, 1], [3., 6.]) for face in ("top", "bottom")})
        result = evaluate_detailing(data)
        self.assertEqual(find(result, "beam.bar_clear_spacing", "top")["status"], "pass")
        self.assertAlmostEqual(find(result, "beam.minimum_steel", "top")["demand"], 1.2)

    def test_malformed_declared_rows_never_fall_back_to_nominal_depth(self):
        mutations = [lambda b: b["bar_rows"]["top"].update(per_layer=[2, 1]),
                     lambda b: b["bar_rows"]["top"].update(centroid_in=3.),
                     lambda b: b["bar_rows"]["top"].update(offsets_in=[5., 15.]),
                     lambda b: b["bar_rows"].pop("bottom"),
                     lambda b: b.update(bar_rows=None)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = layered_example()
                mutate(data["beam"])
                result = evaluate_detailing(data)
                self.assertEqual(find(result, "beam.longitudinal_steel_complete")["status"], "not_evaluated")
                self.assertFalse(any(c["id"] == "beam.minimum_steel" for c in result))

    def test_legacy_single_row_and_uniform_directional_rows(self):
        data = layered_example()
        del data["beam"]["bar_rows"]
        legacy = find(evaluate_detailing(data), "beam.minimum_steel", "top")
        self.assertAlmostEqual(legacy["capacity"], 3 * math.sqrt(8000) / 60000 * 14 * 25.)
        # Minimum As is governed by the greater d; maximum rho by the smaller d.
        data["beam"]["bar_stacking"] = {"layers": {
            axis: {face: rows([4], [offset]) for face in ("top", "bottom")}
            for axis, offset in (("x", 3.), ("y", 5.))}}
        result = evaluate_detailing(data)
        self.assertAlmostEqual(find(result, "beam.minimum_steel", "top")["capacity"], legacy["capacity"])
        self.assertAlmostEqual(find(result, "beam.maximum_rho", "top")["demand"], 2.4 / (14 * 23))

    def test_selection_and_qualification_use_identical_face_limits_at_both_grades(self):
        data = layered_example()
        design = replace(gf.BEAM, b_in=14., h_in=28., fc_ksi=8., bar_size=7, top_bars=4, bot_bars=4)
        for grade in (60., 80.):
            with self.subTest(grade=grade), patch.object(sp, "FY_KSI", grade):
                data["material"]["fy_ksi"] = grade
                strength = gc.beam_flexural_strengths(design, data["beam"]["bar_rows"])
                qualified = evaluate_detailing(data)
                for face, sign in (("top", "negative"), ("bottom", "positive")):
                    item = find(qualified, "beam.minimum_steel", face)
                    self.assertEqual(item["demand"], strength[sign]["as_in2"])
                    self.assertEqual(item["capacity"], strength[sign]["as_min_in2"])

    def test_real_grouped_adapter_retains_face_totals_and_passes_actual_rows(self):
        design = replace(gf.BEAM, b_in=14., h_in=28., fc_ksi=8., bar_size=7, top_bars=4, bot_bars=4)
        for grade in (60., 80.):
            with self.subTest(grade=grade), gf.frame(slab=None, beam=design, FY_KSI=grade):
                state = mg.install(gf.uniform_state(beam=design))
                arrangement = {gid: copy.deepcopy(layered_example()["beam"]["bar_rows"])
                               for gid, d in state.designs.items() if not d.is_column}
                record = {"materials": {"fy_ksi": grade, "normalweight": True, "aggregate_size_in": .75,
                                        "beam_clear_cover_in": 1.5, "col_clear_cover_in": 1.5},
                          "geometry": {"bay_x_in": gf.BAY_X, "bay_y_in": gf.BAY_Y}}
                with patch.object(gr.bar_layers, "arrangement", return_value=arrangement), \
                     patch("openseespy.opensees.analyze", side_effect=AssertionError("No analysis in this test")):
                    result = gr.group_detailing_checks(record, state, {})
                steel = [c for c in result if c["id"] == "beam.minimum_steel"]
                self.assertEqual(len(steel), 2 * len(arrangement))
                for item in steel:
                    self.assertEqual(item["status"], "pass")
                    self.assertAlmostEqual(item["demand"], 2.4)
                    self.assertEqual(item["details"]["sections"]["group"]["bar_count"], 4)


if __name__ == "__main__":
    unittest.main()
