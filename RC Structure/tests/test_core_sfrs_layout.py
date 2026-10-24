"""Perimeter + centered core SMRF layout (user proposal of 2026-10-18): the ring of frames around the middle bay (odd
bay count) or the two middle bays (even bay count) joins the perimeter frames in the seismic force-resisting system.
The policies system_perimeter_core_v1 and system_perimeter_core_three_story_v1 carry the layout; every other policy keeps
the perimeter alone, and the classification of the perimeter policies is unchanged."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
from Design.Config import DesignConfig
from Model import Member_Groups as mg
from test_core_layout_regressions import complete_evaluation

CORE = mg.SFRS_LAYOUT_PERIMETER_CORE


class CoreLinesTests(unittest.TestCase):
    def test_the_core_follows_the_figure(self):
        """Odd bay counts take the middle bay, even counts the two middle bays (the 25-combination figure)."""
        self.assertEqual({n: mg.core_lines(n) for n in range(3, 8)}, {3: (1, 2), 4: (1, 3), 5: (2, 3), 6: (2, 4), 7: (3, 4)})
        for bad in (1, 2, 2.0, "3"):
            with self.assertRaises(ValueError):
                mg.core_lines(bad)

    def test_policies_and_layouts(self):
        self.assertEqual(mg.layout_of(mg.SYSTEM_CORE_POLICY_ID), CORE)
        self.assertEqual(mg.layout_of(mg.SYSTEM_CORE_BANDED_POLICY_ID), CORE)
        for pid in (mg.SYSTEM_POLICY_ID, mg.SYSTEM_BANDED_POLICY_ID, mg.POLICY_ID):
            self.assertEqual(mg.layout_of(pid), mg.SFRS_LAYOUT_PERIMETER)
        self.assertTrue(mg.is_system_policy(mg.SYSTEM_CORE_POLICY_ID) and mg.is_system_policy(mg.SYSTEM_CORE_BANDED_POLICY_ID))
        self.assertTrue(mg.is_banded_policy(mg.SYSTEM_CORE_BANDED_POLICY_ID) and mg.is_banded_policy(mg.SYSTEM_BANDED_POLICY_ID))
        self.assertFalse(mg.is_banded_policy(mg.SYSTEM_CORE_POLICY_ID) or mg.is_banded_policy(mg.SYSTEM_POLICY_ID))
        self.assertEqual(mg.stories_per_band_of(mg.SYSTEM_CORE_BANDED_POLICY_ID, 7), 3)
        self.assertEqual(mg.stories_per_band_of(mg.SYSTEM_CORE_POLICY_ID, 7), 7)
        self.assertEqual(mg.policy_of(mg.SYSTEM_CORE_BANDED_POLICY_ID)["sfrs_layout"], CORE)
        self.assertEqual(mg.policy_of(mg.SYSTEM_BANDED_POLICY_ID)["sfrs_layout"], mg.SFRS_LAYOUT_PERIMETER)
        # every site starts on the perimeter; the core comes in through the ladder only when the perimeter fails
        for site in ("sdc_d_high", "sdc_d_low", "sdc_c", "sdc_b", None):
            self.assertEqual(mg.default_system_policy(site), mg.SYSTEM_BANDED_POLICY_ID)


class FourByThreeTests(unittest.TestCase):
    """4 x 3 bays: a 2 x 1 core between x-lines 1 and 3 and y-lines 1 and 2 (the figure's "4 x 3 bays" cell)."""
    NX, NY = 4, 3

    def test_core_beams_and_columns(self):
        beams = [(t, i, j) for t, ri, rj in (("beam_x", range(4), range(4)), ("beam_y", range(5), range(3)))
                 for j in rj for i in ri if mg.beam_location(t, i, j, self.NX, self.NY, CORE) == "core"]
        self.assertEqual(beams, [("beam_x", 1, 1), ("beam_x", 2, 1), ("beam_x", 1, 2), ("beam_x", 2, 2), ("beam_y", 1, 1), ("beam_y", 3, 1)])
        columns = {(i, j): mg.column_location(i, j, self.NX, self.NY, CORE) for j in range(4) for i in range(5)}
        self.assertEqual({k for k, v in columns.items() if v == "core_corner"}, {(1, 1), (3, 1), (1, 2), (3, 2)})
        self.assertEqual({k for k, v in columns.items() if v == "core_edge"}, {(2, 1), (2, 2)})
        self.assertEqual(sorted(mg.column_sfrs_directions(2, 1, self.NX, self.NY, CORE)), ["x"])    # on the x-running side only
        self.assertEqual(sorted(mg.column_sfrs_directions(1, 1, self.NX, self.NY, CORE)), ["x", "y"])
        self.assertEqual(sorted(mg.column_sfrs_directions(0, 1, self.NX, self.NY, CORE)), ["y"])     # a perimeter column as before
        self.assertEqual(mg.member_system("beam_y", 2, 1, self.NX, self.NY, CORE), mg.SYSTEM_GRAVITY)  # the line through the core
        self.assertEqual(mg.member_system("beam_x", 1, 1, self.NX, self.NY, CORE), mg.SYSTEM_SFRS)
        # the perimeter layout is untouched by the new argument
        self.assertEqual(mg.column_location(1, 1, self.NX, self.NY), "interior")
        self.assertEqual(mg.beam_location("beam_x", 1, 1, self.NX, self.NY), "interior")
        self.assertEqual(sorted(mg.column_sfrs_directions(1, 1, self.NX, self.NY)), [])

    def test_inventory_and_assignments(self):
        core = mg.system_inventory(self.NX, self.NY, CORE)
        perimeter = mg.system_inventory(self.NX, self.NY)
        self.assertEqual(core[mg.SYSTEM_SFRS], {"column": 20, "beam_x": 12, "beam_y": 8})     # all 20 columns sit on a frame here
        self.assertEqual(core[mg.SYSTEM_GRAVITY], {"column": 0, "beam_x": 4, "beam_y": 7})
        self.assertEqual(perimeter[mg.SYSTEM_SFRS], {"column": 14, "beam_x": 8, "beam_y": 6})
        assignments = mg.build_assignments(self.NX, self.NY, 4, mg.SYSTEM_CORE_BANDED_POLICY_ID)
        ties = {(g["band"], g["location_class"], g["member_type"]): g["design_tie"] for g in assignments["groups"]}
        self.assertEqual(ties[("s01_03", "core", "beam_x")], "s01_03__sfrs_beam")
        self.assertEqual(ties[("s04_04", "core_edge", "column")], "s04_04__sfrs_column")
        self.assertEqual(ties[("s01_03", "interior", "beam_y")], "s01_03__gravity_beam")
        self.assertFalse(any(key[1:] == ("interior", "column") for key in ties))
        self.assertEqual(assignments["story_bands"], [[1, 2, 3], [4]])
        uniform = mg.build_assignments(self.NX, self.NY, 4, mg.SYSTEM_CORE_POLICY_ID)
        self.assertEqual(uniform["story_bands"], [[1, 2, 3, 4]])
        self.assertEqual({g["design_tie"] for g in uniform["groups"] if g["location_class"].startswith("core")},
                         {"sfrs_column", "sfrs_beam"})
        self.assertNotEqual(assignments["sha256"], mg.build_assignments(self.NX, self.NY, 4, mg.SYSTEM_BANDED_POLICY_ID)["sha256"])

    def test_small_frames_are_refused_under_the_core_layout(self):
        with self.assertRaises(ValueError):
            mg.build_assignments(2, 3, 4, mg.SYSTEM_CORE_POLICY_ID)
        mg.build_assignments(2, 3, 4, mg.SYSTEM_POLICY_ID)                       # the perimeter policies do not care


class CoreLineFramesTests(unittest.TestCase):
    """The third layout: full-length frames on the core's bounding lines (4 x 3: x-lines 1 and 3, y-lines 1 and 2)."""
    LINES = mg.SFRS_LAYOUT_PERIMETER_CORE_LINES

    def test_core_lines_run_the_full_width(self):
        beams = [(t, i, j) for t, ri, rj in (("beam_x", range(4), range(4)), ("beam_y", range(5), range(3)))
                 for j in rj for i in ri if mg.beam_location(t, i, j, 4, 3, self.LINES) == "core"]
        self.assertEqual(len([b for b in beams if b[0] == "beam_x"]), 8)          # two full x-lines of 4 spans
        self.assertEqual(len([b for b in beams if b[0] == "beam_y"]), 6)          # two full y-lines of 3 spans
        self.assertEqual(mg.column_location(2, 1, 4, 3, self.LINES), "core_edge")
        self.assertEqual(mg.column_location(1, 1, 4, 3, self.LINES), "core_corner")
        self.assertEqual(mg.column_location(2, 0, 4, 3, self.LINES), "y_boundary_edge")
        # a perimeter column on a core line resists both directions (the core frame ends on it)
        self.assertEqual(sorted(mg.column_sfrs_directions(0, 1, 4, 3, self.LINES)), ["x", "y"])
        self.assertEqual(sorted(mg.column_sfrs_directions(2, 1, 4, 3, self.LINES)), ["x"])
        inventory = mg.system_inventory(4, 3, self.LINES)
        self.assertEqual(inventory[mg.SYSTEM_SFRS], {"column": 20, "beam_x": 16, "beam_y": 12})
        # no beam of one group meets a beam of another in line: every interior joint's x beams share a location
        for j in range(1, 3):
            for i in range(1, 4):
                left, right = mg.beam_location("beam_x", i - 1, j, 4, 3, self.LINES), mg.beam_location("beam_x", i, j, 4, 3, self.LINES)
                self.assertEqual(left, right, (i, j))

    def test_the_ladder_and_the_policies(self):
        self.assertEqual(mg.layout_of(mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID), self.LINES)
        self.assertTrue(mg.is_banded_policy(mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID))
        self.assertEqual(mg.layout_ladder_next(mg.SYSTEM_BANDED_POLICY_ID), mg.SYSTEM_CORE_BANDED_POLICY_ID)
        self.assertEqual(mg.layout_ladder_next(mg.SYSTEM_CORE_BANDED_POLICY_ID), mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID)
        self.assertIsNone(mg.layout_ladder_next(mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID))
        self.assertEqual(mg.layout_ladder_next(mg.SYSTEM_POLICY_ID), mg.SYSTEM_CORE_POLICY_ID)
        self.assertIsNone(mg.layout_ladder_next(mg.POLICY_ID))
        assignments = mg.build_assignments(4, 3, 3, mg.SYSTEM_CORE_LINES_POLICY_ID)
        self.assertEqual({g["design_tie"] for g in assignments["groups"]}, {"sfrs_column", "sfrs_beam", "gravity_beam"})


class LayoutLadderSearchTests(unittest.TestCase):
    """The grouped search climbs the ladder with an injected per-layout search."""

    def setUp(self):
        import Structure_Parameters as sp
        self.sp = sp
        self.frame = gf.frame(NUM_BAY_X=4, NUM_BAY_Y=3, NUM_FLOOR=3)
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)

    def _fake(self, feasible_at):
        calls = []

        def search(cfg, state, policy, log_path, verbose):
            calls.append(state.policy_id)
            # every attempt starts from the seed sections, never from what an earlier layout grew to; the grades follow the
            # attempt's own roles (Design.Concrete_Grades): its special-frame columns at the seed's 6 ksi, the columns it
            # leaves outside the SFRS at the gravity ladder's 5 ksi (2026-10-23)
            for gid, d in state.designs.items():
                if d.is_column:
                    grade = 5.0 if state.system_of(gid) == mg.SYSTEM_GRAVITY else 6.0
                    self.assertEqual((d.b_in, d.h_in, d.fc_ksi), (24.0, 24.0, grade), (state.policy_id, gid))
            found = mg.layout_of(state.policy_id) in feasible_at
            result = {"version": "test", "final": complete_evaluation(state) if found else None, "log": [{"index": 1}, {"index": 2}],
                      "stop": {"reason": "found" if found else "feasibility_no_growth_move"}, "counts": {"feasibility_candidates": 2},
                      "elapsed_seconds": 1.0, "tradeoffs": []}
            return result, cfg or DesignConfig.from_structure_parameters()
        return search, calls

    def test_climbs_until_a_design_is_found(self):
        from Design import Grouped_Record as gr
        from Design import Grouped_Search as gs
        state, seed = gr.seed_from_sections((24.0, 24.0, 6.0), (14.0, 24.0, 5.0), policy_id=mg.SYSTEM_BANDED_POLICY_ID)
        search, calls = self._fake({mg.SFRS_LAYOUT_PERIMETER_CORE_LINES})
        result, _cfg = gr._search_with_layout_ladder(None, state, seed, gs.SearchPolicy(), None, False, search=search)
        self.assertEqual(calls, [mg.SYSTEM_BANDED_POLICY_ID, mg.SYSTEM_CORE_BANDED_POLICY_ID, mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID])
        trial = result["sfrs_layout_trial"]
        self.assertTrue(trial["attempted"])
        self.assertIn("original seed sections", trial["restart_basis"])
        self.assertEqual(trial["selected_policy"], mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID)
        self.assertEqual([a["sfrs_layout"] for a in trial["attempts"]], list(mg.SFRS_LAYOUT_LADDER))
        self.assertEqual([a["column_size_cap_in"] for a in trial["attempts"]], [36., 36., 36.])
        self.assertEqual([e["index"] for e in result["log"]], [1, 2, 3, 4, 5, 6])
        self.assertEqual(result["counts"]["feasibility_candidates"], 6)
        self.assertEqual(result["elapsed_seconds"], 3.0)
        self.assertEqual(result["final"]["state"].policy_id, mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID)

    def test_feasible_perimeter_at_36_is_retained(self):
        from Design import Grouped_Record as gr
        from Design import Grouped_Search as gs
        calls = []

        def search(cfg, state, policy, log_path, verbose):
            calls.append(state.policy_id)
            self.assertEqual({d.b_in for d in state.designs.values() if d.is_column}, {36.})
            return ({"final": complete_evaluation(state), "log": [], "stop": {"reason": "reduction_budget_exhausted"},
                     "counts": {}, "elapsed_seconds": 0.}, cfg)

        state, seed = gr.seed_from_sections((36., 36., 10.), (14., 24., 5.), policy_id=mg.SYSTEM_BANDED_POLICY_ID)
        result, _ = gr._search_with_layout_ladder(None, state, seed, gs.SearchPolicy(), None, False, search=search)
        self.assertEqual(calls, [mg.SYSTEM_BANDED_POLICY_ID])
        self.assertFalse(result["sfrs_layout_trial"]["attempted"])
        self.assertIsNotNone(result["final"])

    def test_stops_at_the_top_and_respects_the_switch(self):
        from Design import Grouped_Record as gr
        from Design import Grouped_Search as gs
        state, seed = gr.seed_from_sections((24.0, 24.0, 6.0), (14.0, 24.0, 5.0), policy_id=mg.SYSTEM_BANDED_POLICY_ID)
        search, calls = self._fake(set())
        result, _ = gr._search_with_layout_ladder(None, state, seed, gs.SearchPolicy(), None, False, search=search)
        self.assertIsNone(result["final"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["sfrs_layout_trial"]["selected_policy"], mg.SYSTEM_BANDED_POLICY_ID)
        self.assertIn("top of the layout ladder", result["sfrs_layout_trial"]["stop"])
        search, calls = self._fake(set())
        off = gs.SearchPolicy(sfrs_layout_ladder_on_no_design=False)
        result, _ = gr._search_with_layout_ladder(None, state, seed, off, None, False, search=search)
        self.assertEqual(calls, [mg.SYSTEM_BANDED_POLICY_ID])
        self.assertFalse(result["sfrs_layout_trial"]["attempted"])
        # a frame with 2 bays cannot climb
        self.sp.NUM_BAY_X = 2
        state, seed = gr.seed_from_sections((24.0, 24.0, 6.0), (14.0, 24.0, 5.0), policy_id=mg.SYSTEM_BANDED_POLICY_ID)
        search, calls = self._fake(set())
        result, _ = gr._search_with_layout_ladder(None, state, seed, gs.SearchPolicy(), None, False, search=search)
        self.assertEqual(calls, [mg.SYSTEM_BANDED_POLICY_ID])
        self.assertIn("3 bays", result["sfrs_layout_trial"]["stop"])


class LayoutLadderErrorTests(unittest.TestCase):
    """An attempt that raises is recorded and the ladder goes on (the overnight batch of 2026-10-18 lost five cases to
    a crash inside the core-ring attempt)."""

    def setUp(self):
        import Structure_Parameters as sp
        self.sp = sp
        self.saved = {k: getattr(sp, k) for k in ("NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR")}
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 4, 3, 3

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(self.sp, k, v)
        mg.clear()

    def test_a_crashing_layout_is_recorded_and_skipped(self):
        from Design import Grouped_Record as gr
        from Design import Grouped_Search as gs
        calls = []

        def search(cfg, state, policy, log_path, verbose):
            calls.append(state.policy_id)
            layout = mg.layout_of(state.policy_id)
            if layout == mg.SFRS_LAYOUT_PERIMETER_CORE:
                raise mg.GroupedStateError("Floor 1: the spans of beam_x line 2 have different sections")
            found = layout == mg.SFRS_LAYOUT_PERIMETER_CORE_LINES
            return ({"version": "test", "final": {"state": state} if found else None, "log": [{"index": 1}],
                     "stop": {"reason": "found" if found else "feasibility_no_growth_move"}, "counts": {"feasibility_candidates": 1},
                     "elapsed_seconds": 1.0, "tradeoffs": []}, cfg)
        state, seed = gr.seed_from_sections((24.0, 24.0, 6.0), (14.0, 24.0, 5.0), policy_id=mg.SYSTEM_BANDED_POLICY_ID)
        result, _ = gr._search_with_layout_ladder(None, state, seed, gs.SearchPolicy(), None, False, search=search)
        self.assertEqual(len(calls), 3)
        trial = result["sfrs_layout_trial"]
        self.assertEqual(trial["selected_policy"], mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID)
        self.assertEqual([a.get("error") is not None for a in trial["attempts"]], [False, True, False])
        self.assertIn("different sections", trial["attempt_errors"][0]["error"])
        self.assertEqual(trial["attempts"][1]["stop"]["reason"], "attempt_error")
        self.assertIsNotNone(result["final"])

    def test_every_attempt_raising_is_an_error(self):
        from Design import Grouped_Record as gr
        from Design import Grouped_Search as gs

        def search(cfg, state, policy, log_path, verbose):
            raise RuntimeError("solver down")
        state, seed = gr.seed_from_sections((24.0, 24.0, 6.0), (14.0, 24.0, 5.0), policy_id=mg.SYSTEM_BANDED_POLICY_ID)
        with self.assertRaises(RuntimeError):
            gr._search_with_layout_ladder(None, state, seed, gs.SearchPolicy(), None, False, search=search)


class FiveByFourTests(unittest.TestCase):
    def test_a_one_by_two_core_leaves_gravity_columns(self):
        core = mg.system_inventory(5, 4, CORE)
        self.assertEqual(core[mg.SYSTEM_SFRS]["column"], 24)      # 16 perimeter + 6 ring nodes... see below
        self.assertEqual(core[mg.SYSTEM_GRAVITY]["column"], 6)
        locations = {mg.column_location(i, j, 5, 4, CORE) for j in range(5) for i in range(6)}
        self.assertEqual(locations, {"corner", "x_boundary_edge", "y_boundary_edge", "core_corner", "core_edge", "interior"})


if __name__ == "__main__":
    unittest.main()
