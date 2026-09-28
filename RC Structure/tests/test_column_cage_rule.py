"""The column cage rule (design basis 2026-09-27): column cages are offered only where their lanes admit the
beam bars of both directions in one layer; when none exist the beam widens, when they exist but strength
needed more steel the column grows."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Structure_Parameters as sp  # noqa: E402
import Redesign as rd  # noqa: E402
from Design import Design_Driver as driver  # noqa: E402
from Design import Section_Design  # noqa: E402
from Design.Config import DesignConfig  # noqa: E402
from Design.SMRF_Cage_Geometry import column_cage_admits_beam_bars  # noqa: E402


def frame(beam_width):
    """A 32-in fc-5 column, #5 hoops, and a beam of the given width carrying six #8 bars in #4 stirrups."""
    return mock.patch.multiple(
        sp, B_COL=32.0, H_COL=32.0, FC_COL_KSI=5.0, FY_KSI=60.0, COL_CLEAR_COVER_IN=1.5, COL_STIRRUP_BAR_SIZE=5,
        B_BEAM=beam_width, H_BEAM=24.0, BEAM_CLEAR_COVER_IN=1.5, BEAM_STIRRUP_BAR_SIZE=4,
        BEAM_BAR_SIZE=8, BEAM_TOP_BARS=6, BEAM_BOT_BARS=6, AGGREGATE_MAX_SIZE_IN=0.75, SLAB_THICKNESS_IN=5.0,
        BEAM_BAR_MAX_LAYERS=1)


class CageGeometry(unittest.TestCase):
    def admits(self, beam_width, top_bars, side_bars, col_bar=11):
        return column_cage_admits_beam_bars(32.0, 32.0, 1.5, 0.625, sp.rebar_diameter(col_bar), top_bars, side_bars,
                                            beam_width, 1.5, 0.5, 1.0, 6, 0.75)

    def test_four_bars_per_face_block_a_sixteen_inch_beam_and_three_open_a_twenty(self):
        four = self.admits(16.0, 4, 2)
        self.assertFalse(four["passes"])
        self.assertEqual(four["bars_per_layer"]["x"], 3)
        three_narrow = self.admits(16.0, 3, 1)
        self.assertFalse(three_narrow["passes"], "the middle bar splits a 16-in band into two short lanes")
        three = self.admits(20.0, 3, 1)
        self.assertTrue(three["passes"])
        self.assertGreaterEqual(min(three["bars_per_layer"].values()), 6)
        four_wide = self.admits(20.0, 4, 2)
        self.assertFalse(four_wide["passes"], "two interior lanes leave five bars at 20 in")
        self.assertTrue(self.admits(24.0, 4, 2)["passes"])


class CandidateGenerator(unittest.TestCase):
    def setUp(self):
        self.cfg = DesignConfig()

    def test_threading_filter_keeps_only_cages_whose_lanes_admit_the_beam_bars(self):
        ag = 32.0 * 32.0
        with frame(20.0):
            every = rd._col_candidates(0.01 * ag, 0.06 * ag, self.cfg)
            threading = rd._col_candidates(0.01 * ag, 0.06 * ag, self.cfg, threading=True)
        self.assertTrue(threading)
        self.assertLess(len(threading), len(every))
        self.assertEqual({(c[1], c[3]) for c in threading}, {(3, 1)}, "three bars per face, one interior side bar")
        self.assertTrue({(c[1], c[3]) for c in every} - {(3, 1)})

    def test_pool_prefers_threading_cages_then_stronger_threading_cages_then_falls_back(self):
        ag = 32.0 * 32.0
        with frame(20.0):
            pool, threaded = rd.col_candidates_for_beam_bars(0.01 * ag, 0.02 * ag, self.cfg)
            self.assertTrue(threaded)
            self.assertTrue(all((c[1], c[3]) == (3, 1) for c in pool))
            # a band no threading cage reaches: the stronger threading cages above it are offered
            lo = max(c[4] for c in pool) + 0.01
            pool_hi, threaded_hi = rd.col_candidates_for_beam_bars(0.010 * ag, lo, self.cfg)
            self.assertTrue(threaded_hi)
            self.assertTrue(all((c[1], c[3]) == (3, 1) for c in pool_hi))
            self.assertTrue(rd.column_cages_that_thread_exist(self.cfg))
        with frame(16.0):
            pool, threaded = rd.col_candidates_for_beam_bars(0.01 * ag, 0.03 * ag, self.cfg)
            self.assertFalse(threaded, "no cage at a 32-in column threads six #8 through a 16-in beam")
            self.assertTrue(pool, "strength is never left short")
            self.assertFalse(rd.column_cages_that_thread_exist(self.cfg))
        with frame(20.0), mock.patch.object(sp, "SLAB_THICKNESS_IN", None):
            pool, threaded = rd.col_candidates_for_beam_bars(0.01 * ag, 0.03 * ag, self.cfg)
            self.assertTrue(threaded, "the legacy path is untouched")
            self.assertEqual(pool, rd._col_candidates(0.01 * ag, 0.03 * ag, self.cfg))

    def test_two_layers_let_the_narrow_beam_through_most_cages(self):
        """The two-layer decision: six #8 in a 16-in beam take two layers of three, so cages with three
        bars per layer pass; the rule still excludes a cage that would need a third layer."""
        ag = 32.0 * 32.0
        with frame(16.0), mock.patch.object(sp, "BEAM_BAR_MAX_LAYERS", 2):
            threading = rd._col_candidates(0.01 * ag, 0.06 * ag, self.cfg, threading=True)
            self.assertTrue(threading)
            self.assertTrue(any(c[1] == 4 for c in threading), "four per face: three lanes of one, two layers")
            self.assertTrue(rd.column_cages_that_thread_exist(self.cfg))
        with frame(16.0), mock.patch.object(sp, "BEAM_BAR_MAX_LAYERS", 2), mock.patch.object(sp, "BEAM_TOP_BARS", 7), \
                mock.patch.object(sp, "BEAM_BOT_BARS", 7):
            self.assertFalse(rd._col_candidates(0.01 * ag, 0.06 * ag, self.cfg, threading=True) and
                             all(sp.beam_bars_per_layer(8, 7)[a] * 2 >= 7 for a in ("x", "y")),
                             "seven bars at three per layer need a third layer")


class PlannerLever(unittest.TestCase):
    def setUp(self):
        self.columns = Section_Design.column_ladder()
        self.beams = Section_Design.beam_ladder(span_in=180.0, story_height_in=168.0)
        self.ci = next(i for i, r in enumerate(self.columns) if r == (32.0, 32.0, 8.0))
        self.bi = next(i for i, r in enumerate(self.beams) if r == (16.0, 24.0, 8.0))
        self.flags = {"drift_ok": True, "scwb_ok": True, "joint_scwb_failed": False, "capacity_accepted": False,
                      "beam_section_adequate": True, "beam_hoops_selected": True, "beam_bars_thread": False,
                      "column_section_adequate": True, "column_hoops_selected": True, "joints_all_pass": True,
                      "anchorage_all_pass": True, "joint_shear_ratio": 0.5}

    def plan(self, **overrides):
        flags = {**self.flags, **overrides}
        return driver._plan_next_rungs(self.columns, self.beams, self.ci, self.bi, {"beam": 0.8, "column": 0.4},
                                       0.85, 1.0, self.ci, flags)

    def test_column_grows_when_threading_cages_exist_and_beam_widens_when_none_do(self):
        next_column, next_beam, reasons = self.plan(column_cage_can_thread=True)
        self.assertIn("column_cage_threading", reasons)
        self.assertIn("column_cage_threading", driver.COLUMN_STEP_REASONS)
        self.assertGreater(next_column, self.ci)
        self.assertEqual(next_beam, self.bi, "the beam holds: the lanes, not the band, are the lever")
        next_column, next_beam, reasons = self.plan(column_cage_can_thread=False)
        self.assertNotIn("column_cage_threading", reasons)
        self.assertEqual(next_column, self.ci)
        self.assertGreater(self.beams[next_beam][0], 16.0)
        self.assertEqual(self.beams[next_beam][1], 24.0)


class FewestLayersBeamPick(unittest.TestCase):
    """The larger-bars-per-layer option (user decision 2026-09-27): among beam cages within the DCR
    ceiling, the pick takes the fewest layers first, so four #10 in one layer beat six #8 in two."""

    def test_pick_prefers_the_one_layer_cage_when_the_objective_alone_would_not(self):
        self.assertIn(11, DesignConfig().rebar.bar_sizes_beam)
        cfg = DesignConfig()
        with frame(20.0), mock.patch.object(sp, "BEAM_BAR_MAX_LAYERS", 2):
            self.assertEqual(rd._beam_candidate_layers(8, 6, 6), 2)      # five per layer at 20 in
            self.assertEqual(rd._beam_candidate_layers(10, 4, 4), 1)     # four per layer
            two, one = (8, 6, 6), (10, 4, 4)
            cfg.rebar.beam_prefer_fewest_layers = False
            # a demand at which the objective alone prefers the two-layer cage (its DCR nearer the target)
            mu = next((m for m in range(500, 8000, 50) if rd._pick_beam([two, one], m, m, cfg) == two), None)
            self.assertIsNotNone(mu, "some demand makes the two-layer cage the objective's pick")
            cfg.rebar.beam_prefer_fewest_layers = True
            self.assertEqual(rd._pick_beam([two, one], mu, mu, cfg), one)
            # a demand no cage carries: every cage is kept, fewest layers first, so the section can grow
            self.assertEqual(rd._pick_beam([two, one], 20000.0, 20000.0, cfg), one)
            cfg.rebar.beam_prefer_fewest_layers = False
            self.assertEqual(rd._pick_beam([two, one], 20000.0, 20000.0, cfg), one, "the stronger cage is nearer the target")
        with frame(20.0), mock.patch.object(sp, "SLAB_THICKNESS_IN", None):
            self.assertEqual(rd._beam_candidate_layers(8, 6, 6), 1, "legacy path: no joint assembly")


class ColumnSideOneLayerPreference(unittest.TestCase):
    """The column-side one-layer preference (user decision 2026-09-27): among cages within the ceiling, the
    strength pick and the strong-column escalation take the cages the beam bars pass in fewer layers first."""

    def test_layers_per_cage_and_the_escalation_order(self):
        cfg = DesignConfig()
        self.assertTrue(cfg.rebar.col_prefer_fewest_beam_layers)
        with frame(20.0), mock.patch.object(sp, "BEAM_BAR_MAX_LAYERS", 2):
            self.assertEqual(rd._column_cage_beam_layers(11, 4, 2), 2)       # five per layer for six #8
            self.assertEqual(rd._column_cage_beam_layers(11, 3, 1), 1)       # six per layer
            two = (11, 4, 4, 2, 12 * 1.56)          # 18.7 in2, two beam layers
            lighter_two = (10, 4, 4, 2, 12 * 1.27)  # 15.2 in2, two beam layers
            one = (14, 3, 3, 1, 8 * 2.25)           # 18.0 in2, one beam layer
            self.assertEqual(rd.scwb_cage_order(two, cfg)[0], 2)
            self.assertEqual(rd.scwb_cage_order(one, cfg)[0], 1)
            # with the preference the one-layer cage leads although it is not the lightest
            self.assertEqual(sorted([two, lighter_two, one], key=lambda c: rd.scwb_cage_order(c, cfg)), [one, lighter_two, two])
            cfg.rebar.col_prefer_fewest_beam_layers = False
            self.assertEqual(rd.scwb_cage_order(two, cfg)[0], 0)
            self.assertEqual(sorted([two, lighter_two, one], key=lambda c: rd.scwb_cage_order(c, cfg)), [lighter_two, one, two])
        with frame(20.0), mock.patch.object(sp, "SLAB_THICKNESS_IN", None):
            self.assertEqual(rd._column_cage_beam_layers(11, 4, 2), 1, "legacy path: no joint assembly")

    def test_strength_pick_prefers_the_one_layer_cage_within_the_ceiling(self):
        cfg = DesignConfig()
        cfg.dcr.dcr_target = 0.6
        with frame(20.0), mock.patch.object(sp, "BEAM_BAR_MAX_LAYERS", 2), mock.patch.object(sp, "FC_COL_KSI", 5.0):
            two = (11, 4, 4, 2, 12 * 1.56)
            one = (11, 3, 3, 1, 8 * 1.56)

            def pick(prefer, mu):
                cfg.rebar.col_prefer_fewest_beam_layers = prefer
                return rd._pick_col([two, one], Pu=200.0, Mu=float(mu), cfg=cfg)

            # some demand makes the objective alone take the heavier two-layer cage (nearer the low
            # target) while the preference takes the one-layer cage, which is still within the ceiling
            flipped = [m for m in range(2000, 120000, 500) if pick(False, m) == two and pick(True, m) == one]
            self.assertTrue(flipped, "the preference should flip the pick somewhere within the ceiling")
            # between the one-layer cage's ceiling and the two-layer cage's, the preference does not
            # apply and the objective keeps the two-layer cage; beyond both ceilings every cage is kept
            # and the fewest layers lead again (the section grows next)
            over = [m for m in range(2000, 120000, 500) if pick(False, m) == two and pick(True, m) == two]
            self.assertTrue(over, "between the ceilings the two-layer cage stays the pick")
            self.assertLess(min(flipped), min(over))
            self.assertLess(min(over), max(flipped))
            # no moment: the least steel, the one-layer cage either way
            self.assertEqual(pick(False, 1.0), one)
            self.assertEqual(pick(True, 1.0), one)


class LargerBars(unittest.TestCase):
    """Fewer, larger bars per face: No. 14 joined the column ladder with the rule; it cannot be lap spliced."""

    def test_number_14_is_offered_and_threads_where_number_11_runs_out_of_steel(self):
        self.assertIn(14, DesignConfig().rebar.bar_sizes_col)
        self.assertNotIn(18, DesignConfig().rebar.bar_sizes_col)
        ag = 32.0 * 32.0
        with frame(20.0):
            threading = rd._col_candidates(0.01 * ag, 0.06 * ag, DesignConfig(), threading=True)
        by_size = {}
        for c in threading:
            by_size.setdefault(c[0], []).append(c[4])
        self.assertIn(14, by_size)
        self.assertGreater(max(by_size[14]), max(by_size[11]), "the same lanes hold more steel with No. 14")

    def test_number_14_column_bars_take_mechanical_splices(self):
        from Design.SMRF_Capacity_Design import design_splices
        state = {"sections": {"b_col_in": 36.0, "h_col_in": 36.0, "fc_col_ksi": 5.0, "b_beam_in": 24.0, "h_beam_in": 24.0, "fc_beam_ksi": 5.0},
                 "beam": {"bar_size": 8, "centroid_offset_in": 2.5}, "column": {"bar_size": 14},
                 "materials": {"fy_ksi": 60.0}, "geometry": {"bay_x_in": 180.0, "bay_y_in": 180.0, "story_h_in": 168.0}}
        big = design_splices(state)["column"]
        self.assertFalse(big["lap_splice_permitted_25.5.1.1"])
        self.assertFalse(big["lap_splice_feasible"])
        self.assertEqual(big["splice_type"], "type_2_mechanical_18.2.7")
        state["column"]["bar_size"] = 11
        eleven = design_splices(state)["column"]
        self.assertTrue(eleven["lap_splice_permitted_25.5.1.1"])
        self.assertEqual(eleven["splice_type"], "class_B_lap_center_half" if eleven["lap_splice_feasible"] else "type_2_mechanical_18.2.7")


if __name__ == "__main__":
    unittest.main()
