"""The bar position grid (design-basis rule 2026-10-14): first bars in the hoop corners on the 0.5 in grid, grid
pitches along a face, grid layer pitches, hoops that follow the bars, and the beam proportion rules."""
import contextlib
import math
import unittest
from unittest import mock

import Structure_Parameters as sp
from Design import Section_Design as sd
from Design.ACI_Checks import column_bar_cb_in
from Design.SMRF_Cage_Geometry import beam_cage_geometry, column_bars_plan, thread_bars
from Design.SMRF_Cage_Layout import beam_cage, column_cage, face_bar_positions


class GridHelpers(unittest.TestCase):
    def test_snap_and_first_offset(self):
        self.assertEqual(sp.snap_up_to_grid(2.375), 2.5)
        self.assertEqual(sp.snap_up_to_grid(2.5), 2.5)
        self.assertEqual(sp.snap_up_to_grid(2.5000000001), 2.5)
        self.assertEqual(sp.snap_up_to_grid(2.564), 3.0)
        # the layout is set for the #5 hoop of the ladders whatever hoop is selected (BAR_LAYOUT_HOOP_DB_IN),
        # and the capacity design measures its confinement core to the same hoop
        from Design.SMRF_Capacity_Design import LAYOUT_HOOP_DB_IN
        self.assertEqual(sp.BAR_LAYOUT_HOOP_DB_IN, 0.625)
        self.assertEqual(LAYOUT_HOOP_DB_IN, sp.BAR_LAYOUT_HOOP_DB_IN)
        self.assertEqual(sp.first_bar_offset_in(1.5, 0.5, 0.75), 2.5)          # #6 bar: 2.5 exactly with the #5 hoop
        self.assertEqual(sp.first_bar_offset_in(1.5, 0.5, 1.0), 3.0)           # #8 bar: 2.625 -> 3.0
        self.assertEqual(sp.first_bar_offset_in(1.5, 0.625, 1.0), 3.0)         # the same with a #5 hoop selected
        self.assertEqual(sp.first_bar_offset_in(1.5, 0.625, 1.128), 3.0)       # #9 bar, #5 hoop
        self.assertEqual(sp.first_bar_offset_in(1.5, 0.5, 1.41), 3.0)          # #11 bar: 2.83 -> 3.0
        self.assertEqual(sp.first_bar_offset_in(1.5, 0.5, 1.0, grid_in=None), 2.5)    # pre-rule records: exact
        with self.assertRaises(ValueError):
            sp.snap_up_to_grid(math.nan)

    def test_face_positions_equal_pitch_and_symmetric_unequal(self):
        self.assertEqual(sp.face_bar_positions_in(20.0, 2.5, 4), [2.5, 7.5, 12.5, 17.5])
        self.assertEqual(sp.face_bar_positions_in(20.0, 2.5, 5), [2.5, 6.0, 10.0, 14.0, 17.5])
        self.assertEqual(sp.face_bar_positions_in(18.0, 2.5, 4), [2.5, 7.0, 11.0, 15.5])
        self.assertEqual(sp.face_bar_positions_in(12.0, 2.5, 3), [2.5, 6.0, 9.5])
        self.assertEqual(sp.face_bar_positions_in(14.0, 2.5, 1), [7.0])
        for width, count in ((20.0, 5), (18.0, 4), (22.0, 6), (24.0, 7), (14.0, 6)):
            positions = sp.face_bar_positions_in(width, 2.5, count)
            for p in positions:
                self.assertAlmostEqual(p / 0.5, round(p / 0.5), places=9, msg=f"{p} off the grid")
            mirrored = [width - p for p in reversed(positions)]
            self.assertEqual([round(p, 9) for p in positions], [round(p, 9) for p in mirrored])
            pitches = sp.face_bar_pitches_in(width, 2.5, count)
            self.assertLessEqual(max(pitches) - min(pitches), 0.5 + 1e-9)
        self.assertEqual(sp.face_bar_min_pitch_in(20.0, 2.5, 5), 3.5)
        self.assertEqual(sp.face_bar_max_pitch_in(20.0, 2.5, 5), 4.0)
        self.assertEqual(sp.face_bar_min_pitch_in(20.0, 2.5, 1), math.inf)
        # more bars than grid steps: no grid arrangement; the even pitch below one step is returned so a
        # clear-spacing screen fails on it (a candidate screen asked for 12 bars across a 10 in web)
        self.assertLess(sp.face_bar_min_pitch_in(10.0, 2.5, 12), 0.5)
        self.assertEqual(len(sp.face_bar_positions_in(10.0, 2.5, 12)), 12)
        with self.assertRaises(ValueError):
            sp.face_bar_positions_in(10.25, 2.5, 3)                         # span off the grid

    def test_cage_layout_uses_the_grid(self):
        self.assertEqual(face_bar_positions(20.0, 1.5, 0.5, 0.75, 5), [2.5, 6.0, 10.0, 14.0, 17.5])
        plan = column_bars_plan(20.0, 20.0, 1.5, 0.5, 0.75, 4, 2)
        self.assertIn((2.5, 2.5), plan)
        self.assertIn((17.5, 17.5), plan)
        self.assertTrue(all(abs(v / 0.5 - round(v / 0.5)) < 1e-9 for xy in plan for v in xy))
        self.assertEqual(beam_cage(12.0, 1.5, 0.5, 0.875, 3, 2)["faces"], {"top": [3.0, 6.0, 9.0], "bottom": [3.0, 9.0]})
        cage = column_cage(20.0, 20.0, 1.5, 0.5, 0.75, 4, 2, legs=4)
        self.assertTrue(cage["constructible"])

    def test_threaded_lanes_land_on_the_grid(self):
        placed = thread_bars(3, 0.875, 6.4375, 13.5625, [7.5, 12.5], 0.75, 1.0)
        self.assertTrue(all(abs(p / 0.5 - round(p / 0.5)) < 1e-9 for p in placed))
        for p in placed:
            for o in (7.5, 12.5):
                self.assertGreaterEqual(abs(p - o) - (0.75 + 0.875) / 2.0, 1.0 - 1e-9)

    def test_cb_uses_the_least_grid_pitch(self):
        cb = column_bar_cb_in(20.0, 20.0, 1.5, 4, 6, 5, 2)
        self.assertEqual(cb["edge_distance_in"], 2.5)
        self.assertEqual(cb["min_center_spacing_in"], 3.5)
        self.assertEqual(cb["cb_in"], 1.75)


class CoverAsBuilt(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name, value in {"SLAB_THICKNESS_IN": 6.0, "BEAM_CLEAR_COVER_IN": 1.5, "COL_CLEAR_COVER_IN": 1.5,
                            "COL_BAR_SIZE": 6, "COL_STIRRUP_BAR_SIZE": 4, "BEAM_BAR_SIZE": 7, "BEAM_STIRRUP_BAR_SIZE": 4,
                            "BEAM_TOP_BARS": 2, "BEAM_BOT_BARS": 2, "BEAM_BAR_STACKING": "x_over_y",
                            "BEAM_BAR_MAX_LAYERS": 1}.items():
            self.stack.enter_context(mock.patch.object(sp, name, value))
        self.addCleanup(self.stack.close)

    def test_offsets_and_covers(self):
        self.assertEqual(sp.longitudinal_cover_in("column"), 2.5)
        self.assertEqual(sp.clear_cover_provided_in("column"), 2.5 - 0.5 - 0.375)
        self.assertEqual(sp.core_cover_in("column"), 2.5 - 0.5 - 0.375)
        self.assertEqual(sp.specified_clear_cover_in("column"), 1.5)
        self.assertEqual(sp.longitudinal_cover_in("beam"), 3.0)                  # #7 bar: 2.5625 -> 3.0 (layout hoop #5)
        self.assertEqual(sp.clear_cover_provided_in("beam"), 3.0 - 0.5 - 0.4375)
        self.assertGreaterEqual(sp.clear_cover_provided_in("beam"), 1.5)
        self.assertEqual(sp.longitudinal_cover_in("beam", 9, 5), 3.0)
        self.assertEqual(sp.beam_bar_layer_pitch_in(7), 2.0)
        self.assertEqual(sp.beam_bar_layer_pitch_in(9), 2.5)
        self.assertEqual(sp.beam_bar_layer_pitch_in(10), 3.0)

    def test_beam_hoops_follow_the_first_layer_of_each_direction(self):
        covers = sp.beam_hoop_cover_in()
        side = sp.clear_cover_provided_in("beam")
        # x over y: the x bars nearest the top face, the y bars one pitch down; mirrored at the bottom
        self.assertAlmostEqual(covers["x"]["top"], side)
        self.assertAlmostEqual(covers["y"]["top"], side + 2.0)
        self.assertAlmostEqual(covers["y"]["bottom"], side)
        self.assertAlmostEqual(covers["x"]["bottom"], side + 2.0)
        offsets = sp.beam_bar_layers()
        for axis in ("x", "y"):
            for face in ("top", "bottom"):
                self.assertAlmostEqual(offsets[axis][face]["offsets_in"][0] - 0.5 - 0.4375, covers[axis][face])
        self.assertAlmostEqual(side, 3.0 - 0.5 - 0.4375)

    def test_legacy_route_untouched(self):
        sp.SLAB_THICKNESS_IN = None
        self.assertEqual(sp.longitudinal_cover_in("beam", 11, 5), sp.COVER)
        self.assertEqual(sp.core_cover_in("column"), sp.COVER)
        self.assertEqual(sp.clear_cover_provided_in("beam"), sp.COVER)


class BeamCageGeometryHoops(unittest.TestCase):
    def test_tie_sets_per_direction_from_the_record(self):
        record = {"sections": {"b_beam_in": 14.0, "h_beam_in": 28.0},
                  "reinforcement": {"beam_bar_size": 8, "beam_stirrup_bar_size": 4, "beam_clear_cover_in": 1.5,
                                    "beam_top_bars": 3, "beam_bot_bars": 2, "beam_stirrup_legs": 2,
                                    "beam_stirrup_spacing_in": 4.0,
                                    "beam_hoop_cover_in": {"x": {"top": 1.5, "bottom": 3.5}, "y": {"top": 3.5, "bottom": 1.5}}},
                  "materials": {"aggregate_size_in": 0.75}}
        geometry = beam_cage_geometry(record)
        # #8 bars with a #4 hoop: first bar 3.0 in (layout hoop #5), so the lateral cover as built is 2.0 in
        self.assertEqual(geometry["clear_cover_provided_in"], 2.0)
        self.assertEqual(geometry["tie_sets"][0]["hoop"]["out_to_out_in"], [10.0, 24.0])
        self.assertEqual(geometry["tie_sets_by_direction"]["x"][0]["hoop"]["out_to_out_in"], [10.0, 23.0])
        self.assertEqual(geometry["tie_sets_by_direction"]["x"][0]["hoop"]["clear_cover_in"]["depth_faces"], [1.5, 3.5])
        self.assertEqual(geometry["tie_sets_by_direction"]["y"][0]["hoop"]["out_to_out_in"], [10.0, 23.0])


class BeamProportions(unittest.TestCase):
    def test_paired_widths_and_variants_keep_the_beam_deeper_than_wide(self):
        self.assertEqual(sd.BEAM_MIN_WIDTH_DEPTH_RATIO, 0.5)
        self.assertAlmostEqual(sd.BEAM_MIN_DEPTH_WIDTH_RATIO, 4.0 / 3.0)
        for depth in sd.BEAM_DEPTHS_IN:
            widths = sd.beam_widths_for_depth(depth)
            self.assertTrue(widths)
            for width in widths:
                self.assertGreaterEqual(width / depth, 0.5 - 1e-9)
                self.assertGreaterEqual(depth / width, 4.0 / 3.0 - 1e-9)
        self.assertEqual(sd.beam_widths_for_depth(24.0), [12.0, 16.0])
        self.assertEqual(sd.beam_widths_for_depth(36.0), [18.0, 22.0, 26.0])
        for rung in sd.beam_ladder():
            sd.validate_rung(rung, "beam")
        with self.assertRaises(ValueError):
            sd.validate_rung((24.0, 24.0, 5.0), "beam")                      # square
        with self.assertRaises(ValueError):
            sd.validate_rung((20.0, 22.0, 5.0), "beam")                      # 0.91: not deep enough
        with self.assertRaises(ValueError):
            sd.validate_rung((10.0, 24.0, 5.0), "beam")                      # 0.42: web too narrow for the basis


if __name__ == "__main__":
    unittest.main()
