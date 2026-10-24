"""Cover/steel coordinates for new SMRF candidates, with legacy isolation."""

import contextlib
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
import RC_Design_Check as checks
import Redesign as redesign
from Design.Config import DesignConfig
from Model import Sections as sections
from Model import IMK_Calibration as imk


class CoverTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name, value in {
            "SLAB_THICKNESS_IN": 6.0, "COVER": 1.5,
            "BEAM_CLEAR_COVER_IN": 1.5, "COL_CLEAR_COVER_IN": 1.5,
            "AGGREGATE_MAX_SIZE_IN": 0.75,
            "B_COL": 18.0, "H_COL": 18.0, "B_BEAM": 12.0, "H_BEAM": 18.0,
            "COL_BAR_SIZE": 8, "COL_TOP_BARS": 4, "COL_BOT_BARS": 4,
            "COL_SIDE_BARS": 2, "COL_BAR_AREA": sp.rebar_area(8),
            "BEAM_BAR_SIZE": 6, "BEAM_TOP_BARS": 2, "BEAM_BOT_BARS": 2,
            "BEAM_SIDE_BARS": 0, "BEAM_BAR_AREA": sp.rebar_area(6),
            "COL_STIRRUP_BAR_SIZE": 4, "BEAM_STIRRUP_BAR_SIZE": 4,
        }.items():
            self.stack.enter_context(mock.patch.object(sp, name, value))
        self.addCleanup(self.stack.close)

    def test_explicit_member_and_candidate_centroid_offsets(self):
        # Bar centres on the 0.5 in grid (design-basis rule 2026-10-14): #6 beam bars with #4 hoops sit at
        # 2.375 -> 2.5 in, the hoop follows them, so the cover as built is 1.625 in (1.5 specified).
        # the layout is set for the #5 hoop of the ladders: #8 column bars at 2.625 -> 3.0 in, #6 beam bars at 2.5 in
        self.assertEqual(sp.longitudinal_cover_in("column"), 3.0)
        self.assertEqual(sp.longitudinal_cover_in("beam"), 2.5)
        self.assertEqual(sp.longitudinal_cover_in("beam", 8, 3), 3.0)
        self.assertEqual(sp.core_cover_in("beam"), 1.625)
        self.assertEqual(sp.core_cover_in("column"), 2.0)
        self.assertEqual(sp.specified_clear_cover_in("beam"), 1.5)
        self.assertEqual(sp.COL_BAR_SIZE, 8)
        self.assertEqual(sp.BEAM_BAR_SIZE, 6)

    def test_legacy_offsets_ignore_new_defaults_and_candidate_diameters(self):
        sp.SLAB_THICKNESS_IN = None
        sp.COVER = 1.625
        sp.BEAM_CLEAR_COVER_IN = 2.0
        self.assertEqual(sp.longitudinal_cover_in("beam", 11, 5), 1.625)
        self.assertEqual(sp.longitudinal_cover_in("column", 11), 1.625)
        self.assertEqual(sp.core_cover_in("beam"), 1.625)
        self.assertEqual(checks._beam_d(), 16.375)
        self.assertEqual(checks._col_d(), 16.375)

    def test_aggregate_spacing_and_invalid_inputs(self):
        self.assertEqual(sp.longitudinal_clear_spacing_in("beam", 6), 1.0)
        self.assertEqual(sp.longitudinal_clear_spacing_in("column", 8), 1.5)
        sp.AGGREGATE_MAX_SIZE_IN = 1.5
        self.assertEqual(sp.longitudinal_clear_spacing_in("beam", 6), 2.0)
        self.assertEqual(sp.longitudinal_clear_spacing_in("column", 8), 2.0)
        sp.AGGREGATE_MAX_SIZE_IN = math.nan
        with self.assertRaises(ValueError):
            sp.longitudinal_clear_spacing_in("beam")
        with self.assertRaises(ValueError):
            sp.longitudinal_cover_in("slab")
        sp.COL_CLEAR_COVER_IN = 0.0
        with self.assertRaises(ValueError):
            sp.longitudinal_cover_in("column")

    def test_design_depths_and_nominal_moments_use_member_cover(self):
        self.assertEqual(checks._beam_d(), 15.5)                      # #6 bars on the grid: 2.5 in, not 2.375
        self.assertEqual(checks._col_d(), 15.0)                       # #8 bars: 3.0 in (layout hoop #5)
        area = 2 * sp.rebar_area(6)
        expected = area * sp.FY_KSI * (15.5 - area * sp.FY_KSI
                                       / (1.7 * sp.FC_BEAM_KSI * 12))
        self.assertAlmostEqual(sp.beam_nominal_moment_y(), expected)
        # Haselton (2008) Eq. 3.10: TOTAL longitudinal steel over b d (2026-10-23): the beam's 2 top and 2 bottom
        # bars; the column's 4 + 4 face bars and 2 side bars on EACH side face (4 + 4 + 2 x 2 = 12)
        self.assertAlmostEqual(imk.longitudinal_steel_ratio("beam"),
                               4 * sp.rebar_area(6) / (12 * 15.5))
        self.assertAlmostEqual(imk.longitudinal_steel_ratio("column"),
                               12 * sp.COL_BAR_AREA / (18 * 15.0))

    def test_column_steel_layers_match_actual_side_coordinates(self):
        layers = checks._col_steel_layers()
        self.assertEqual(len(layers), 4)
        self.assertAlmostEqual(sum(a for a, _ in layers), 12 * sp.COL_BAR_AREA)
        self.assertEqual(layers[0][1], 3.0)
        # the two side bars on the 0.5 in grid: 4.0 in pitches across the 18 in face from 3.0 in
        self.assertAlmostEqual(layers[1][1], 7.0)
        self.assertAlmostEqual(layers[2][1], 11.0)
        self.assertEqual(layers[-1][1], 15.0)
        proportional = redesign._layers_for_Ast(12 * sp.COL_BAR_AREA)
        for actual, expected in zip(proportional, layers):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])

    def test_legacy_column_side_steel_lump_unchanged(self):
        sp.SLAB_THICKNESS_IN = None
        expected = [(4 * sp.COL_BAR_AREA, 1.5), (4 * sp.COL_BAR_AREA, 9.0),
                    (4 * sp.COL_BAR_AREA, 16.5)]
        self.assertEqual(checks._col_steel_layers(), expected)
        self.assertEqual(redesign._layers_for_Ast(12 * sp.COL_BAR_AREA), expected)

    def test_column_candidate_uses_its_own_bar_layout_and_offset(self):
        candidate = (11, 3, 3, 1, 8 * sp.rebar_area(11))
        layers = redesign._layers_for_Ast(candidate[-1], candidate=candidate)
        self.assertEqual(len(layers), 3)
        self.assertAlmostEqual(layers[0][1], 3.0)                       # #11 bars: 2.705 in taken up to the grid
        self.assertAlmostEqual(layers[0][0], 3 * sp.rebar_area(11))
        self.assertAlmostEqual(layers[1][0], 2 * sp.rebar_area(11))
        self.assertEqual(layers[1][1], 9.0)
        self.assertEqual(sp.COL_BAR_SIZE, 8)

    def test_new_fiber_core_and_longitudinal_offsets_are_separate(self):
        with mock.patch.object(sections, "ops") as ops:
            sections.define_column_section()
        core = ops.patch.call_args_list[0].args
        # the core boundary is the hoop as built: #8 bars at 3.0 in less a #4 hoop and half a bar = 2.0 in cover
        self.assertEqual(core[-4:], (-7.0, -7.0, 7.0, 7.0))
        # one fiber per bar where the grid cage puts it: 4 + 4 face bars at 3.0, 7.0, 11.0, 15.0 in,
        # two side bars at 7.0 and 11.0 in on each side (18 in column, #8 bars, #4 hoops)
        self.assertFalse(ops.layer.called)
        fibers = sorted((round(call.args[0], 6), round(call.args[1], 6)) for call in ops.fiber.call_args_list)
        self.assertEqual(len(fibers), 12)
        self.assertEqual({z for _, z in fibers if abs(z) == 6.0}, {-6.0, 6.0})
        self.assertEqual(sorted(y for y, z in fibers if z == 6.0), [-6.0, -2.0, 2.0, 6.0])
        self.assertEqual(sorted(y for y, z in fibers if z == -6.0), [-6.0, -2.0, 2.0, 6.0])
        self.assertEqual(sorted((y, z) for y, z in fibers if abs(z) != 6.0), [(-6.0, -2.0), (-6.0, 2.0), (6.0, -2.0), (6.0, 2.0)])

    def test_legacy_fiber_core_and_bar_coordinates_unchanged(self):
        sp.SLAB_THICKNESS_IN = None
        with mock.patch.object(sections, "ops") as ops:
            sections.define_column_section()
        self.assertEqual(ops.patch.call_args_list[0].args[-4:], (-7.5, -7.5, 7.5, 7.5))
        self.assertEqual(ops.layer.call_args_list[0].args[-4:], (-7.5, 7.5, 7.5, 7.5))

    def test_longitudinal_offset_cannot_be_outside_core(self):
        with self.assertRaises(ValueError):
            sections.make_rc_rect_section(1, 12, 18, 1.5, 2, 3, 4,
                                          2, 2, .44, longitudinal_cover=1.0)

    def test_beam_candidates_check_candidate_diameter_spacing(self):
        cfg = DesignConfig()
        cfg.rebar.bar_sizes_beam = (8,)
        # Four #8 bars fit the old centroid assumption in a 10-inch beam, but
        # not 1.5-inch clear cover + #4 hoops + #8 centroid offsets.
        sp.B_BEAM = 10.0
        with mock.patch.object(cfg.rebar, "beam_n_iter", return_value=range(4, 5)):
            self.assertEqual(redesign._beam_candidates(0, 20, 0, 20, cfg), [])
            sp.SLAB_THICKNESS_IN = None
            self.assertIn((8, 4, 4), redesign._beam_candidates(0, 20, 0, 20, cfg))

    def test_aggregate_limit_removes_crowded_beam_candidates(self):
        cfg = DesignConfig()
        cfg.rebar.bar_sizes_beam = (6,)
        with mock.patch.object(cfg.rebar, "beam_n_iter", return_value=range(4, 5)):
            sp.B_BEAM = 12.0
            self.assertIn((6, 4, 4), redesign._beam_candidates(0, 20, 0, 20, cfg))
            sp.AGGREGATE_MAX_SIZE_IN = 1.5
            self.assertEqual(redesign._beam_candidates(0, 20, 0, 20, cfg), [])

    def test_column_candidates_check_actual_clear_cover_and_spacing(self):
        cfg = DesignConfig()
        cfg.rebar.bar_sizes_col = (8,)
        cfg.rebar.col_n_side_options = (0,)
        sp.B_COL = 12.0
        sp.H_COL = 18.0
        with mock.patch.object(cfg.rebar, "col_n_top_iter", return_value=range(4, 5)):
            self.assertEqual(redesign._col_candidates(0, 20, cfg), [])
            sp.SLAB_THICKNESS_IN = None
            self.assertIn((8, 4, 4, 0, 8 * sp.rebar_area(8)),
                          redesign._col_candidates(0, 20, cfg))

    def test_beam_picker_prices_each_candidate_at_its_own_depth(self):
        cfg = DesignConfig()
        cfg.dcr.objective = "min_deviation"
        with mock.patch.object(sp, "longitudinal_cover_in", wraps=sp.longitudinal_cover_in) as cover:
            redesign._pick_beam([(4, 2, 2), (10, 2, 2)], Mu_pos=250, Mu_neg=250, cfg=cfg)
        self.assertIn(mock.call("beam", 4), cover.call_args_list)
        self.assertIn(mock.call("beam", 10), cover.call_args_list)

    def test_column_picker_uses_candidate_layout_not_current_layout(self):
        cfg = DesignConfig()
        cfg.dcr.objective = "min_deviation"
        candidates = [(5, 3, 3, 1, 8 * sp.rebar_area(5)),
                      (8, 2, 2, 0, 4 * sp.rebar_area(8))]
        with mock.patch.object(redesign, "_phi_Mn_at_Pu", return_value=1000) as capacity:
            redesign._pick_col(candidates, Pu=100, Mu=800, cfg=cfg)
        self.assertEqual(capacity.call_args_list,
                         [mock.call(100, c[-1], candidate=c) for c in candidates])


if __name__ == "__main__":
    unittest.main()
