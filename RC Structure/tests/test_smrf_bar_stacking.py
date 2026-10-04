"""Design-basis rules of 2026-09-27: beam bars thread the column in one layer, and the orthogonal
top/bottom layers stack at the joints with the lower direction's depth carried into its strengths."""
import copy
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Structure_Parameters as sp  # noqa: E402
from Design import Section_Design  # noqa: E402
from Design.SMRF_Beam_Slab_Strength import composite_beam_strengths  # noqa: E402
from Design.SMRF_Capacity_Design import design_bar_threading  # noqa: E402
from Design import Design_Driver  # noqa: E402

LAYOUT = {"outer_axis": "x", "layers": {
    "x_top": {"bar_size": 4, "layer": "outer", "axis": "x", "clear_cover_outer_mat_in": 0.75, "effective_depth_in": 4.0,
              "spacing_in": 10.0, "bar_area_in2": 0.2},
    "x_bottom": {"bar_size": 4, "layer": "outer", "axis": "x", "clear_cover_outer_mat_in": 0.75, "effective_depth_in": 4.0,
                 "spacing_in": 10.0, "bar_area_in2": 0.2},
    "y_top": {"bar_size": 4, "layer": "inner", "axis": "y", "clear_cover_outer_mat_in": 0.75, "effective_depth_in": 3.5,
              "spacing_in": 10.0, "bar_area_in2": 0.2},
    "y_bottom": {"bar_size": 4, "layer": "inner", "axis": "y", "clear_cover_outer_mat_in": 0.75, "effective_depth_in": 3.5,
                 "spacing_in": 10.0, "bar_area_in2": 0.2}}}


def slab_state(**overrides):
    """A 32-in column with four #11 per face and two interior side bars; a 24-in beam with six #8 per face
    (seven pass per layer, so one layer), two layers allowed."""
    values = {"SLAB_THICKNESS_IN": 5.0, "SLAB_REINFORCEMENT": {"layout": LAYOUT}, "BEAM_BAR_SIZE": 8,
              "BEAM_STIRRUP_BAR_SIZE": 4, "BEAM_CLEAR_COVER_IN": 1.5, "BEAM_BAR_STACKING": "inner_over_outer",
              "B_COL": 32.0, "H_COL": 32.0, "COL_CLEAR_COVER_IN": 1.5, "COL_STIRRUP_BAR_SIZE": 5, "COL_BAR_SIZE": 11,
              "COL_TOP_BARS": 4, "COL_SIDE_BARS": 2, "B_BEAM": 24.0, "BEAM_TOP_BARS": 6, "BEAM_BOT_BARS": 6,
              "AGGREGATE_MAX_SIZE_IN": 0.75, "BEAM_BAR_MAX_LAYERS": 2}
    values.update(overrides)
    return mock.patch.multiple(sp, **values)


class StackingOffsets(unittest.TestCase):
    def test_inner_over_outer_resolves_from_the_layout(self):
        with slab_state():
            self.assertEqual(sp.beam_bar_stacking_convention(), "y_over_x")
            base = sp.longitudinal_cover_in("beam")
            offsets = sp.beam_bar_stacking_offsets_in()
        self.assertAlmostEqual(base, 1.5 + 0.5 + 0.5)                 # 2.5 in
        self.assertEqual(offsets["y"], {"top": base, "bottom": base + 2.0})
        self.assertEqual(offsets["x"], {"top": base + 2.0, "bottom": base})
        with slab_state(SLAB_REINFORCEMENT={"layout": {**LAYOUT, "outer_axis": "y"}}):
            self.assertEqual(sp.beam_bar_stacking_convention(), "x_over_y")

    def test_legacy_and_none_keep_one_elevation(self):
        with slab_state(BEAM_BAR_STACKING="none"):
            offsets = sp.beam_bar_stacking_offsets_in()
            self.assertEqual(offsets["x"], offsets["y"])
            self.assertEqual(offsets["x"]["top"], offsets["x"]["bottom"])
        with slab_state(SLAB_THICKNESS_IN=None):
            offsets = sp.beam_bar_stacking_offsets_in()
            self.assertEqual(offsets["x"]["top"], sp.longitudinal_cover_in("beam"))
        with slab_state(BEAM_BAR_STACKING="sideways"):
            with self.assertRaises(ValueError):
                sp.beam_bar_stacking_convention()

    def test_worst_offset_and_face_lookup(self):
        with slab_state():
            self.assertAlmostEqual(sp.beam_worst_centroid_offset_in(), 4.5)
            self.assertAlmostEqual(sp.beam_centroid_offset_in("x", "top"), 4.5)
            self.assertAlmostEqual(sp.beam_centroid_offset_in("x", "bottom"), 2.5)
            self.assertAlmostEqual(sp.beam_centroid_offset_in("y", "top"), 2.5)
            self.assertEqual(sp.beam_bar_layers()["x"]["top"]["per_layer"], [6])

    def test_second_layer_where_the_lanes_force_it(self):
        """A 16-in beam passes three #8 per layer between the column bars: six bars take two layers of
        three, one pitch (db + max(1 in, db) = 2 in) apart; under the blocked order the lower
        direction's cage starts below the upper direction's second layer."""
        with slab_state(B_BEAM=16.0, BEAM_BAR_LAYER_ORDER="blocked"):
            self.assertEqual(sp.beam_bars_per_layer(), {"x": 3, "y": 3})
            layers = sp.beam_bar_layers()
            self.assertEqual(layers["y"]["top"]["per_layer"], [3, 3])
            self.assertEqual(layers["y"]["top"]["offsets_in"], [2.5, 4.5])
            self.assertEqual(layers["x"]["top"]["offsets_in"], [6.5, 8.5])
            self.assertEqual(layers["x"]["bottom"]["offsets_in"], [2.5, 4.5])          # mirrored at the bottom
            self.assertEqual(layers["y"]["bottom"]["offsets_in"], [6.5, 8.5])
            offsets = sp.beam_bar_stacking_offsets_in()
            self.assertEqual(offsets, {"y": {"top": 3.5, "bottom": 7.5}, "x": {"top": 7.5, "bottom": 3.5}})
            self.assertAlmostEqual(sp.beam_worst_centroid_offset_in(), 7.5)
            # a candidate with fewer bars fits one layer and keeps the nominal offsets
            self.assertEqual(sp.beam_bar_stacking_offsets_in(top_bars=3, bot_bars=3),
                             {"y": {"top": 2.5, "bottom": 4.5}, "x": {"top": 4.5, "bottom": 2.5}})
        with slab_state(B_BEAM=16.0, BEAM_BAR_MAX_LAYERS=1):
            # under a one-layer limit the bars stay at one elevation and the threading check fails instead
            self.assertEqual(sp.beam_bar_layers()["x"]["top"]["per_layer"], [6])
            self.assertEqual(sp.beam_bar_stacking_offsets_in(), {"y": {"top": 2.5, "bottom": 4.5}, "x": {"top": 4.5, "bottom": 2.5}})

    def test_interleaved_layers_alternate_the_directions(self):
        """The interleaved order (user decision 2026-09-27): upper first layer, lower first layer, upper
        second, lower second, one pitch apart, so neither direction gives up a whole cage's depth."""
        self.assertEqual(sp.beam_bar_layer_slots(2, 2, "interleaved"), ([0, 2], [1, 3]))
        self.assertEqual(sp.beam_bar_layer_slots(1, 2, "interleaved"), ([0], [1, 2]))
        self.assertEqual(sp.beam_bar_layer_slots(2, 1, "interleaved"), ([0, 2], [1]))
        self.assertEqual(sp.beam_bar_layer_slots(1, 1, "interleaved"), sp.beam_bar_layer_slots(1, 1, "blocked"))
        self.assertEqual(sp.beam_bar_layer_slots(2, 2, "blocked"), ([0, 1], [2, 3]))
        with self.assertRaises(ValueError):
            sp.beam_bar_layer_slots(1, 1, "sideways")
        with slab_state(B_BEAM=16.0, BEAM_BAR_LAYER_ORDER="interleaved"):
            layers = sp.beam_bar_layers()
            self.assertEqual(layers["y"]["top"]["offsets_in"], [2.5, 6.5])          # y over x: y first, x, y, x
            self.assertEqual(layers["x"]["top"]["offsets_in"], [4.5, 8.5])
            self.assertEqual(layers["x"]["bottom"]["offsets_in"], [2.5, 6.5])       # mirrored: x nearest the bottom
            self.assertEqual(layers["y"]["bottom"]["offsets_in"], [4.5, 8.5])
            self.assertEqual(sp.beam_bar_stacking_offsets_in(), {"y": {"top": 4.5, "bottom": 6.5}, "x": {"top": 6.5, "bottom": 4.5}})
            self.assertAlmostEqual(sp.beam_worst_centroid_offset_in(), 6.5)         # blocked would be 7.5


class StrengthsWithStackedLayers(unittest.TestCase):
    beam = {"b_in": 16.0, "h_in": 24.0, "fc_ksi": 8.0, "fy_ksi": 60.0, "bar_size": 8, "top_bars": 6, "bot_bars": 6}
    geometry = {"bay_x_in": 144.0, "bay_y_in": 180.0, "h_col_in": 32.0, "b_col_in": 32.0}

    def test_lower_top_layer_reduces_hogging_only(self):
        nominal = composite_beam_strengths({**self.beam, "centroid_offset_in": 2.5}, {"thickness_in": 0.0}, None, self.geometry, "x", "interior")
        stacked = composite_beam_strengths({**self.beam, "centroid_offset_in": {"top": 4.5, "bottom": 2.5}},
                                           {"thickness_in": 0.0}, None, self.geometry, "x", "interior")
        self.assertLess(stacked["mn_negative_kip_in"], nominal["mn_negative_kip_in"])
        ratio = stacked["mn_negative_kip_in"] / nominal["mn_negative_kip_in"]
        self.assertGreater(ratio, 0.85); self.assertLess(ratio, 0.95)
        # sagging: the bottom bars are at the nominal offset; the top (compression-side) bars moved but
        # the section is under-reinforced, so the change is small
        self.assertAlmostEqual(stacked["mn_positive_kip_in"] / nominal["mn_positive_kip_in"], 1.0, delta=0.03)
        self.assertEqual(stacked["centroid_offsets_in"], {"top": 4.5, "bottom": 2.5})

    def test_record_families_use_each_direction_offsets(self):
        from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths
        record = {"sections": {"b_col_in": 32.0, "h_col_in": 32.0, "fc_col_ksi": 8.0, "b_beam_in": 16.0, "h_beam_in": 24.0, "fc_beam_ksi": 8.0},
                  "reinforcement": {"beam_bar_size": 8, "beam_top_bars": 6, "beam_bot_bars": 6, "beam_longitudinal_centroid_offset_in": 2.5,
                                    "beam_clear_cover_in": 1.5, "beam_stirrup_diameter_in": 0.5,
                                    "beam_bar_stacking": {"convention": "y_over_x", "offsets_in": {"x": {"top": 4.5, "bottom": 2.5},
                                                                                                    "y": {"top": 2.5, "bottom": 4.5}}}},
                  "geometry": {"num_bay_x": 2, "num_bay_y": 5, "num_floor": 8, "bay_x_in": 144.0, "bay_y_in": 180.0},
                  "materials": {"fy_ksi": 60.0}, "slab": {"thickness_in": 5.0},
                  "slab_reinforcement": {"layout": LAYOUT}}
        _entries, families = beam_slab_strengths(record)
        self.assertEqual(families["x_interior"]["centroid_offsets_in"], {"top": 4.5, "bottom": 2.5})
        self.assertEqual(families["y_interior"]["centroid_offsets_in"], {"top": 2.5, "bottom": 4.5})
        self.assertLess(families["x_interior"]["rectangular"]["negative"]["mn_kip_in"],
                        families["y_interior"]["rectangular"]["negative"]["mn_kip_in"])


class ThreadingScreen(unittest.TestCase):
    def state(self, beam_width, top_bars=6, max_layers=1, layer_order="blocked"):
        return {"sections": {"b_col_in": 32.0, "h_col_in": 32.0, "fc_col_ksi": 8.0, "b_beam_in": beam_width, "h_beam_in": 24.0, "fc_beam_ksi": 8.0},
                "materials": {"fy_ksi": 60.0, "es_ksi": 29000.0, "normalweight": True, "aggregate_size_in": 0.75},
                "beam": {"bar_size": 8, "top_bars": top_bars, "bot_bars": top_bars, "centroid_offset_in": 2.5, "clear_cover_in": 1.5,
                         "stirrup_bar_size": 4, "stacking_convention": "y_over_x", "max_layers": max_layers, "layer_order": layer_order,
                         "centroid_offsets_by_axis_in": {"x": {"top": 4.5, "bottom": 2.5}, "y": {"top": 2.5, "bottom": 4.5}},
                         "worst_centroid_offset_in": 4.5},
                "column": {"bar_size": 11, "top_bars": 4, "bot_bars": 4, "side_bars": 3, "centroid_offset_in": 2.83,
                           "clear_cover_in": 1.5, "stirrup_bar_size": 5},
                "slab": {"thickness_in": 5.0, "layout": LAYOUT}}

    def test_sixteen_inch_beam_fails_and_twenty_four_passes(self):
        narrow = design_bar_threading(self.state(16.0))
        self.assertFalse(narrow["passes"])
        self.assertEqual(narrow["by_direction"]["x"]["bars_per_layer_that_fit"], 3)
        self.assertTrue(narrow["stacking_clear_of_mats"])
        wide = design_bar_threading(self.state(24.0))
        self.assertTrue(wide["passes"])
        self.assertTrue(all(d["fits_in_one_layer"] for d in wide["by_direction"].values()))

    def test_two_layers_pass_the_sixteen_inch_beam_and_price_the_depth(self):
        two = design_bar_threading(self.state(16.0, max_layers=2))
        self.assertTrue(two["passes"])
        self.assertEqual(two["max_layers"], 2)
        # three side bars per h face leave four lanes for the y direction, four top bars three for x
        self.assertEqual(two["layers"], {"y": [4, 2], "x": [3, 3]})
        self.assertFalse(any(d["fits_in_one_layer"] for d in two["by_direction"].values()))
        self.assertTrue(all(d["fits_within_layer_limit"] for d in two["by_direction"].values()))
        stacking = two["stacking"]
        self.assertEqual(stacking["layer_offsets_from_face_in"], {"y": [2.5, 4.5], "x": [6.5, 8.5]})
        self.assertAlmostEqual(stacking["upper_layer_centroid_from_face_in"], (4 * 2.5 + 2 * 4.5) / 6.0)
        self.assertAlmostEqual(stacking["lower_layer_centroid_from_face_in"], 7.5)
        self.assertLess(stacking["lower_direction_lever_arm_ratio"], 0.8)
        three = design_bar_threading(self.state(16.0, top_bars=7, max_layers=2))
        self.assertFalse(three["passes"], "seven bars at three per layer need a third layer")
        # interleaved: y first layer, x first layer, y second, x second
        inter = design_bar_threading(self.state(16.0, max_layers=2, layer_order="interleaved"))
        self.assertTrue(inter["passes"])
        self.assertEqual(inter["layer_order"], "interleaved")
        self.assertEqual(inter["stacking"]["layer_offsets_from_face_in"], {"y": [2.5, 6.5], "x": [4.5, 8.5]})
        self.assertAlmostEqual(inter["stacking"]["lower_layer_centroid_from_face_in"], 6.5)
        self.assertGreater(inter["stacking"]["lower_direction_lever_arm_ratio"], stacking["lower_direction_lever_arm_ratio"])

    def test_planner_widens_the_beam_when_bars_do_not_thread(self):
        ladder = Section_Design.beam_ladder(span_in=180.0, story_height_in=168.0)
        index = Section_Design.nearest_rung_index(ladder, 16.0, 24.0, 8.0)
        self.assertEqual(ladder[index], (16.0, 24.0, 8.0))
        self.assertIn((24.0, 24.0, 8.0), ladder, "the ladder now offers the wider variants")
        flags = {"drift_ok": True, "scwb_ok": True, "joint_scwb_failed": False, "capacity_accepted": False,
                 "beam_section_adequate": True, "beam_hoops_selected": True, "beam_bars_thread": False,
                 "column_section_adequate": True, "column_hoops_selected": True, "joints_all_pass": True,
                 "anchorage_all_pass": True, "joint_shear_ratio": 0.5}
        columns = Section_Design.column_ladder()
        col = Section_Design.nearest_rung_index(columns, 32.0, 32.0, 8.0)
        next_col, next_beam, reasons = Design_Driver._plan_next_rungs(columns, ladder, col, index, {"beam": 0.8, "column": 0.4},
                                                                      0.85, 1.0, col, flags)
        self.assertIn("beam_bar_threading", reasons)
        self.assertIn("beam_bar_threading", Design_Driver.BEAM_STEP_REASONS)
        self.assertEqual(next_col, col)
        self.assertGreater(next_beam, index)
        self.assertEqual(ladder[next_beam][1], 24.0, "same depth")
        self.assertGreater(ladder[next_beam][0], 16.0, "wider")


class RecordRoundTrip(unittest.TestCase):
    def test_apply_design_restores_the_stacking_and_rejects_a_mismatch(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "outputs", "dv_local_gm_20260926", "case_0002", "design.json")
        if not os.path.exists(path):
            self.skipTest("local case_0002 record not present")
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        g = record["geometry"]
        geometry = ("NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "STORY_H", "BAY_X", "BAY_Y", "NUM_MODES")
        self.assertIn("BEAM_BAR_STACKING", Design_Driver._STATE_KEYS, "the search restores the stacking with the rest of the state")
        with mock.patch.multiple(sp, **{name: getattr(sp, name) for name in set(Design_Driver._STATE_KEYS) | set(geometry)}):
            sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = g["num_bay_x"], g["num_bay_y"], g["num_floor"]
            sp.STORY_H, sp.BAY_X, sp.BAY_Y = g["story_h_in"], g["bay_x_in"], g["bay_y_in"]
            sp.NUM_MODES = 3 * sp.NUM_FLOOR
            Design_Driver.apply_design(copy.deepcopy(record))
            # a record from before the rule: single elevation, stacking "none"
            self.assertEqual(sp.BEAM_BAR_STACKING, "none")
            self.assertEqual(sp.beam_bar_stacking_offsets_in()["x"]["top"], sp.longitudinal_cover_in("beam"))
            # a record carrying the rule must reproduce
            stacked = copy.deepcopy(record)
            stacked["reinforcement"]["beam_bar_stacking"] = {"mode": "inner_over_outer", "convention": "y_over_x",
                                                            "offsets_in": {"x": {"top": 4.5, "bottom": 2.5}, "y": {"top": 2.5, "bottom": 4.5}}}
            Design_Driver.apply_design(stacked)
            self.assertEqual(sp.beam_bar_stacking_convention(), "y_over_x")
            wrong = copy.deepcopy(stacked)
            wrong["reinforcement"]["beam_bar_stacking"]["offsets_in"]["x"]["top"] = 3.0
            with self.assertRaisesRegex(ValueError, "stacking does not reproduce"):
                Design_Driver.apply_design(wrong)


class ActualRows(unittest.TestCase):
    """Review items 1 and 3 (2026-09-27): every bar row is its own steel layer in the strength module, and
    malformed row metadata is rejected rather than silently truncated."""
    beam = {"b_in": 26.0, "h_in": 28.0, "fc_ksi": 5.0, "fy_ksi": 60.0, "bar_size": 11, "top_bars": 5, "bot_bars": 5}
    geometry = {"bay_x_in": 144.0, "bay_y_in": 180.0, "h_col_in": 40.0, "b_col_in": 40.0}
    rows = {"top": {"layers": 2, "per_layer": [3, 2], "offsets_in": [5.65, 11.29], "centroid_in": (3 * 5.65 + 2 * 11.29) / 5},
            "bottom": {"layers": 2, "per_layer": [3, 2], "offsets_in": [2.83, 8.47], "centroid_in": (3 * 2.83 + 2 * 8.47) / 5}}

    def test_rows_beat_the_collapsed_centroid_when_the_deep_row_is_in_tension(self):
        collapsed = composite_beam_strengths({**self.beam, "centroid_offset_in": {"top": self.rows["top"]["centroid_in"], "bottom": self.rows["bottom"]["centroid_in"]}},
                                             {"thickness_in": 0.0}, None, self.geometry, "x", "interior")
        actual = composite_beam_strengths({**self.beam, "centroid_offset_in": None, "layers": self.rows},
                                          {"thickness_in": 0.0}, None, self.geometry, "x", "interior")
        for sign in ("negative", "positive"):
            ratio = actual["rectangular"][sign]["mn_kip_in"] / collapsed["rectangular"][sign]["mn_kip_in"]
            self.assertGreater(ratio, 1.03, f"{sign}: the deep row of the compression face lies below the neutral axis")
            self.assertLess(ratio, 1.15)
        self.assertEqual(actual["bar_rows"]["top"], [{"bars": 3, "elevation_from_face_in": 5.65}, {"bars": 2, "elevation_from_face_in": 11.29}])
        self.assertAlmostEqual(actual["centroid_offsets_in"]["top"], self.rows["top"]["centroid_in"])
        self.assertAlmostEqual(actual["steel_area_in2"]["top"], 5 * 1.56)
        # a single row per face reproduces the collapsed result exactly
        one = {"top": {"per_layer": [5], "offsets_in": [2.83]}, "bottom": {"per_layer": [5], "offsets_in": [2.83]}}
        single_rows = composite_beam_strengths({**self.beam, "centroid_offset_in": None, "layers": one}, {"thickness_in": 0.0}, None, self.geometry, "x", "interior")
        single = composite_beam_strengths({**self.beam, "centroid_offset_in": 2.83}, {"thickness_in": 0.0}, None, self.geometry, "x", "interior")
        self.assertAlmostEqual(single_rows["mn_negative_kip_in"], single["mn_negative_kip_in"], places=6)

    def test_malformed_rows_are_rejected(self):
        from Design.SMRF_Beam_Slab_Strength import validate_bar_rows
        good = {"layers": 2, "per_layer": [3, 1], "offsets_in": [2.5, 4.5]}
        self.assertEqual(validate_bar_rows(good, 4, "top bars", 24.0), [(3, 2.5), (1, 4.5)])
        cases = [({"per_layer": [3, 1], "offsets_in": [2.5]}, 4, "must match"),               # a zip would keep 3 of 4 bars
                 ({"per_layer": [3, 1], "offsets_in": [2.5, 4.5]}, 5, "not the 5"),
                 ({"layers": 1, "per_layer": [3, 1], "offsets_in": [2.5, 4.5]}, 4, "declared 1 layers"),
                 ({"per_layer": [3, 1], "offsets_in": [2.5, 4.5], "centroid_in": 2.5}, 4, "centroid"),
                 ({"per_layer": [3, 1], "offsets_in": [4.5, 2.5]}, 4, "strictly increasing"),
                 ({"per_layer": [3, 1], "offsets_in": [2.5, 12.5]}, 4, "past mid-depth"),
                 ({"per_layer": [3, 0], "offsets_in": [2.5, 4.5]}, 3, "positive integer"),
                 ({"per_layer": [], "offsets_in": []}, 0, "must match")]
        for rows, count, message in cases:
            with self.assertRaisesRegex(ValueError, message, msg=str(rows)):
                validate_bar_rows(rows, count, "top bars", 24.0)
        with self.assertRaisesRegex(ValueError, "must match"):
            composite_beam_strengths({**self.beam, "centroid_offset_in": None,
                                      "layers": {"top": {"per_layer": [3, 2], "offsets_in": [5.65]}, "bottom": self.rows["bottom"]}},
                                     {"thickness_in": 0.0}, None, self.geometry, "x", "interior")

    def test_record_families_carry_the_rows(self):
        from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths
        record = {"sections": {"b_col_in": 40.0, "h_col_in": 40.0, "fc_col_ksi": 5.0, "b_beam_in": 26.0, "h_beam_in": 28.0, "fc_beam_ksi": 5.0},
                  "reinforcement": {"beam_bar_size": 11, "beam_top_bars": 5, "beam_bot_bars": 5, "beam_longitudinal_centroid_offset_in": 2.83,
                                    "beam_clear_cover_in": 1.5, "beam_stirrup_diameter_in": 0.625,
                                    "beam_bar_stacking": {"convention": "y_over_x", "layer_order": "interleaved", "max_layers": 2,
                                                          "offsets_in": {"x": {"top": self.rows["top"]["centroid_in"], "bottom": self.rows["bottom"]["centroid_in"]},
                                                                         "y": {"top": self.rows["bottom"]["centroid_in"], "bottom": self.rows["top"]["centroid_in"]}},
                                                          "layers": {"x": self.rows, "y": {"top": self.rows["bottom"], "bottom": self.rows["top"]}}}},
                  "geometry": {"num_bay_x": 2, "num_bay_y": 5, "num_floor": 8, "bay_x_in": 144.0, "bay_y_in": 180.0},
                  "materials": {"fy_ksi": 60.0}, "slab": {"thickness_in": 5.0}, "slab_reinforcement": {"layout": LAYOUT}}
        _entries, families = beam_slab_strengths(record)
        self.assertEqual([r["bars"] for r in families["x_interior"]["bar_rows"]["top"]], [3, 2])
        self.assertEqual([r["bars"] for r in families["y_interior"]["bar_rows"]["top"]], [3, 2])
        self.assertLess(families["x_interior"]["rectangular"]["negative"]["mn_kip_in"], families["y_interior"]["rectangular"]["negative"]["mn_kip_in"])


if __name__ == "__main__":
    unittest.main()
