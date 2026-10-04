"""Research check5: enumerate multi-row beam cages without leaving seed bars."""
import math
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Structure_Parameters as sp
import Redesign as rd
from Design.Config import DesignConfig
from Design.Section_Design import beam_ladder, beam_widths_for_depth


class BeamCandidateCoverage(unittest.TestCase):
    def test_wide_rung_remains_inside_existing_proportioning_limits(self):
        self.assertIn((34., 36., 8.), beam_ladder(span_in=318., story_height_in=174.))
        for depth in range(18, 37, 2):
            self.assertTrue(all(width <= depth for width in beam_widths_for_depth(depth)))

    def frame(self, **overrides):
        state = dict(B_COL=36., H_COL=36., FC_COL_KSI=10.,
                     COL_BAR_SIZE=11, COL_TOP_BARS=5, COL_BOT_BARS=5, COL_SIDE_BARS=3,
                     COL_STIRRUP_BAR_SIZE=5, COL_CLEAR_COVER_IN=1.5,
                     B_BEAM=30., H_BEAM=36., FC_BEAM_KSI=8., FY_KSI=60.,
                     BEAM_BAR_SIZE=6, BEAM_TOP_BARS=2, BEAM_BOT_BARS=2,
                     BEAM_STIRRUP_BAR_SIZE=4, BEAM_CLEAR_COVER_IN=1.5,
                     BEAM_BAR_MAX_LAYERS=2, BEAM_BAR_STACKING='y_over_x',
                     BEAM_BAR_LAYER_ORDER='interleaved', AGGREGATE_MAX_SIZE_IN=.75,
                     NUM_BAY_X=2, NUM_BAY_Y=2, SLAB_THICKNESS_IN=7.5)
        state.update(overrides)
        return patch.multiple(sp, **state)

    def test_case0048_iteration8_has_a_strength_candidate_beyond_seven_bars(self):
        with self.frame():
            cfg = DesignConfig()
            mu = 18149.554457353763  # saved iteration 8, kip-in
            d = sp.H_BEAM - sp.beam_worst_centroid_offset_in(max_layers=1)
            lo = rd._beam_As_for_Mu(mu, 8., 60., 30., d)
            self.assertGreater(lo, 7 * sp.rebar_area(11))
            candidates = rd._beam_candidates(lo, math.inf, lo, math.inf, cfg, threading=True)
            self.assertTrue(candidates)
            best = rd._pick_beam(candidates, mu, mu, cfg)
            self.assertGreater(best[1], 7)
            self.assertLessEqual(rd._beam_candidate_layers(*best), 2)
            depth = sp.H_BEAM - sp.beam_worst_centroid_offset_in(*best)
            area = best[1] * sp.rebar_area(best[0])
            phi_mn = .9 * area * 60. * (depth - area * 60. / (1.7 * 8. * 30.))
            # The quick capacity estimate remains slightly short here; the
            # full frame must re-evaluate it, rather than treating it as a pass.
            self.assertLess(mu / phi_mn, 1.1)
            self.assertEqual(best, (11, 8, 8))
            cfg.rebar.beam_n_range = (2, 7)
            self.assertEqual(rd._beam_candidates(lo, math.inf, lo, math.inf, cfg), [])

    def test_two_rows_on_both_faces_are_not_rejected_by_one_row_width(self):
        with self.frame(B_BEAM=14.):
            cfg = DesignConfig()
            cfg.rebar.bar_sizes_beam = [8]
            cfg.rebar.beam_n_range = (6, 6)
            # Six No. 8 bars do not fit across this web in one row, but two
            # rows pass the actual column lanes in both directions.
            cover = sp.longitudinal_cover_in('beam', 8)
            self.assertLess((14. - 2 * cover) / 5 - 1., sp.longitudinal_clear_spacing_in('beam', 8))
            self.assertIn((8, 6, 6), rd._beam_candidates(0., math.inf, 0., math.inf, cfg, threading=True))
            with patch.object(sp, 'BEAM_BAR_MAX_LAYERS', 1):
                self.assertEqual(rd._beam_candidates(0., math.inf, 0., math.inf, cfg, threading=True), [])
            self.assertEqual(rd._beam_candidates(0., math.inf, 0., math.inf, cfg, threading=False), [])

    def test_larger_count_does_not_relax_joint_depth_or_layer_limit(self):
        with self.frame(B_COL=24., H_COL=24.):
            cfg = DesignConfig()
            cfg.rebar.bar_sizes_beam = [11]
            self.assertEqual(rd._beam_candidates(0., math.inf, 0., math.inf, cfg, threading=True), [])

    def test_dense_cage_outside_hoop_ladder_does_not_displace_supported_alternative(self):
        with self.frame():
            cfg = DesignConfig()
            offered = rd._beam_candidates(11.12, math.inf, 11.12, math.inf, cfg, threading=True)
            self.assertIn((11, 8, 8), offered)
            self.assertFalse(any(c[0] == 9 and c[1] >= 12 for c in offered))
        with self.frame(B_BEAM=16., B_COL=32., H_COL=32., COL_TOP_BARS=4,
                        COL_BOT_BARS=4, COL_SIDE_BARS=2):
            cfg = DesignConfig()
            cfg.rebar.bar_sizes_beam = [8]
            cfg.rebar.beam_n_range = (7, 14)
            self.assertEqual(rd._beam_candidates(0., math.inf, 0., math.inf, cfg, threading=True), [])


if __name__ == '__main__':
    unittest.main()
