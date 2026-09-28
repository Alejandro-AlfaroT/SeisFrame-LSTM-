"""Cage geometry beyond the tie arrangement (Design/SMRF_Cage_Geometry, 2026-09-27 detailing review items 1-2)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Design import SMRF_Cage_Geometry as cg  # noqa: E402


def record(**overrides):
    base = {
        "sections": {"b_col_in": 32.0, "h_col_in": 32.0, "fc_col_ksi": 8.0, "b_beam_in": 16.0, "h_beam_in": 24.0, "fc_beam_ksi": 8.0},
        "reinforcement": {"col_bar_size": 11, "col_top_bars": 4, "col_bot_bars": 4, "col_side_bars": 3,
                          "beam_bar_size": 8, "beam_top_bars": 6, "beam_bot_bars": 6, "beam_side_bars": 0,
                          "col_stirrup_bar_size": 5, "col_stirrup_legs": 4,
                          "col_stirrup_legs_by_direction": {"across_b_face": 4, "across_h_face": 4}, "col_stirrup_spacing_in": 3.0,
                          "beam_stirrup_bar_size": 4, "beam_stirrup_legs": 4, "beam_stirrup_spacing_in": 4.0,
                          "beam_clear_cover_in": 1.5, "col_clear_cover_in": 1.5,
                          "beam_longitudinal_centroid_offset_in": 2.5, "col_longitudinal_centroid_offset_in": 2.83,
                          "col_bar_area_in2": 1.56, "beam_bar_area_in2": 0.79},
        "materials": {"fy_ksi": 60.0, "aggregate_size_in": 0.75},
        "slab": {"thickness_in": 5.0},
        "slab_reinforcement": {"layout": {"layers": {
            "x_top": {"bar_size": 4, "layer": "outer", "axis": "x", "clear_cover_outer_mat_in": 0.75},
            "x_bottom": {"bar_size": 4, "layer": "outer", "axis": "x", "clear_cover_outer_mat_in": 0.75},
            "y_top": {"bar_size": 4, "layer": "inner", "axis": "y", "clear_cover_outer_mat_in": 0.75},
            "y_bottom": {"bar_size": 4, "layer": "inner", "axis": "y", "clear_cover_outer_mat_in": 0.75}}}},
        "capacity_design": {"anchorage": {"directions": {"x": {"ldh_required_in": 10.3}, "y": {"ldh_required_in": 10.3}}}},
    }
    for key, value in overrides.items():
        base[key].update(value)
    return base


class Hooks(unittest.TestCase):
    def test_table_25_3_2_tie_hooks(self):
        h = cg.hook_geometry(5, "seismic_135")
        self.assertEqual(h["inside_bend_diameter_in"], 4 * 0.625)
        self.assertEqual(h["extension_in"], max(6 * 0.625, 3.0))          # 3.75 in
        self.assertEqual(cg.hook_geometry(3, "seismic_135")["extension_in"], 3.0)   # 6db = 2.25 < 3 in floor
        self.assertEqual(cg.hook_geometry(4, "crosstie_90")["extension_in"], 6 * 0.5)
        self.assertEqual(cg.hook_geometry(6, "crosstie_90")["extension_in"], 12 * 0.75)
        self.assertEqual(cg.hook_geometry(6, "crosstie_90")["inside_bend_diameter_in"], 6 * 0.75)
        with self.assertRaises(ValueError):
            cg.hook_geometry(9, "seismic_135")

    def test_table_25_3_1_development_hooks(self):
        h8, h11 = cg.hook_geometry(8, "development_90"), cg.hook_geometry(11, "development_90")
        self.assertEqual((h8["inside_bend_diameter_in"], h8["extension_in"]), (6.0, 12.0))
        self.assertEqual((h11["inside_bend_diameter_in"], h11["extension_in"]), (8 * 1.41, 12 * 1.41))
        self.assertEqual(cg.hook_geometry(8, "development_180")["extension_in"], 4.0)

    def test_tie_set_alternates_the_90_degree_end(self):
        spans = [("x", 11.6, 26.3), ("x", 20.4, 26.3)]
        set0, set1 = cg.tie_set(32.0, 32.0, 1.5, 5, spans, 0), cg.tie_set(32.0, 32.0, 1.5, 5, spans, 1)
        self.assertEqual(set0["hoop"]["out_to_out_in"], [29.0, 29.0])
        self.assertEqual(set0["hoop"]["hooks"], ["seismic_135", "seismic_135"])
        self.assertEqual([t["ninety_degree_end"] for t in set0["crossties"]], ["far", "near"])
        self.assertEqual([t["ninety_degree_end"] for t in set1["crossties"]], ["near", "far"])
        self.assertGreater(set0["crossties"][0]["cut_length_in"], 26.3 + 3.75 + 3.75)


class Clearances(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(cg.clear_spacing_limit("beam", 1.0, 0.75), 1.0)
        self.assertEqual(cg.clear_spacing_limit("beam", 1.0, 1.0), 4.0 / 3.0)
        self.assertEqual(cg.clear_spacing_limit("column", 1.41, 0.75), 1.5 * 1.41)

    def test_threading_between_column_bars(self):
        # six 1 in bars between 10.5 and 21.5 with 1 in clear: 6 fit with no obstacles ...
        self.assertEqual(len(cg.thread_bars(6, 1.0, 10.5, 21.5, [], 1.41, 1.0)), 6)
        # ... but only 3 once column bars sit at 11.6 and 20.4 (the 32 in column's b-face lanes)
        placed = cg.thread_bars(6, 1.0, 10.5, 21.5, [11.61, 20.39], 1.41, 1.0)
        self.assertEqual(len(placed), 3)
        for p in placed:
            for o in (11.61, 20.39):
                self.assertGreaterEqual(abs(p - o), (1.41 + 1.0) / 2 + 1.0 - 1e-9)


class JointAssembly(unittest.TestCase):
    def test_case_0002_like_joint(self):
        res = cg.evaluate_cage_geometry(record())
        self.assertTrue(res["column"]["passes"])
        self.assertTrue(res["beam"]["passes"])
        joint = res["joint"]
        self.assertEqual(joint["directions"]["x"]["bars_per_layer_that_fit"], 3)
        self.assertEqual(joint["directions"]["y"]["bars_per_layer_that_fit"], 4)
        self.assertEqual(joint["directions"]["x"]["layers_needed"], 2)
        self.assertFalse(joint["passes"])
        self.assertAlmostEqual(joint["stacking"]["lower_layer_centroid_from_face_in"], 4.5)
        self.assertAlmostEqual(joint["stacking"]["effective_depth_lower_in"], 19.5)
        self.assertLess(joint["stacking"]["lower_direction_lever_arm_ratio"], 0.92)
        # y over x keeps the crossing mats clear of the lower layer; x over y overlaps the x bottom mat
        self.assertEqual(res["stacking_selected"], "y_over_x")
        self.assertEqual(res["joint_by_stacking"]["y_over_x"]["slab_clashes"], [])
        self.assertTrue(any(c.get("overlap_in") for c in res["joint_by_stacking"]["x_over_y"]["slab_clashes"]))
        self.assertTrue(joint["exterior_hooks"]["tail_within_joint_depth"])
        # the deepest top layer governs the tail: the lower direction's bars sit 4.5 in from the face
        self.assertAlmostEqual(joint["exterior_hooks"]["tail_reaches_from_top_face_in"], 4.5 + 3.0 + 1.0 + 12.0)

    def test_narrow_beam_with_few_bars_threads_cleanly(self):
        res = cg.evaluate_cage_geometry(record(reinforcement={"beam_top_bars": 3, "beam_bot_bars": 3, "beam_stirrup_legs": 2}))
        self.assertTrue(res["joint"]["directions"]["x"]["fits_in_one_layer"])
        self.assertTrue(res["joint"]["directions"]["y"]["fits_in_one_layer"])

    def test_stacking_convention_is_validated(self):
        with self.assertRaises(ValueError):
            cg.joint_assembly(record(), stacking="sideways")


if __name__ == "__main__":
    unittest.main()
