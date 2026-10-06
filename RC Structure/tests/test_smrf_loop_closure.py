"""The section search closes on what qualification evaluates.

Covers the planner that moves the ladder rungs, the depth step for beams,
checks re-run on captured actions, and the strong-column steel selection.
"""
import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Design import Design_Driver as driver
from Design.ACI_Checks import run_checks_phase1
from Design.Config import DesignConfig
from Design.Section_Design import beam_ladder, column_ladder
from RC_Design_Check import get_element_tags


# The column ladder's largest size (36 in until 2026-09-27, 42 in since): the scenarios below that need
# "no larger column left" use it rather than a fixed number.
TOP = column_ladder()[-1][0]


def flags(**overrides):
    base = {"drift_ok": True, "scwb_ok": True, "joint_scwb_failed": False, "capacity_accepted": True,
            "beam_section_adequate": True, "beam_hoops_selected": True, "column_section_adequate": True,
            "column_hoops_selected": True, "joints_all_pass": True, "anchorage_all_pass": True}
    base.update(overrides)
    return base


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.columns = column_ladder()
        self.beams = beam_ladder(span_in=240., story_height_in=144.)

    def test_beam_ladder_is_ordered_by_capacity_and_offers_a_wider_variant(self):
        from Design.Section_Design import (flexural_capacity_proxy, beam_width_for_depth, beam_widths_for_depth,
                                           BEAM_WIDTH_STEP_IN, BEAM_WIDTH_VARIANTS)
        proxies = [flexural_capacity_proxy(r) for r in self.beams]
        self.assertEqual(proxies, sorted(proxies))
        depths = {r[1] for r in self.beams}
        for h in depths:
            widths = sorted({r[0] for r in self.beams if r[1] == h})
            self.assertEqual(widths, beam_widths_for_depth(h))
            # the paired width, 4-in wider variants (up to four widths since 2026-09-27), none wider than deep
            self.assertEqual(widths[:2], [beam_width_for_depth(h), beam_width_for_depth(h) + BEAM_WIDTH_STEP_IN])
            self.assertLessEqual(len(widths), BEAM_WIDTH_VARIANTS)
            self.assertTrue(all(w <= h for w in widths))
        self.assertEqual(beam_widths_for_depth(24.), [12., 16., 20., 24.])          # reaches 3/4 of a 32-in column

    def test_next_deeper_beam_skips_concrete_strength_and_width_rungs(self):
        index = next(i for i, r in enumerate(self.beams) if r[:3] == (10., 18., 5.))
        deeper = driver._next_deeper_beam_index(self.beams, index)
        self.assertGreater(self.beams[deeper][1], 18.)
        self.assertTrue(all(r[1] <= 18. for r in self.beams[index:deeper]))
        self.assertIsNone(driver._next_deeper_beam_index(self.beams, len(self.beams) - 1))

    def test_next_wider_beam_keeps_the_depth(self):
        index = next(i for i, r in enumerate(self.beams) if r[:3] == (10., 18., 5.))
        wider = driver._next_wider_beam_index(self.beams, index)
        self.assertEqual(self.beams[wider][1], 18.)
        self.assertEqual(self.beams[wider][0], 14.)
        widest = max(i for i, r in enumerate(self.beams) if r[1] == 18.)
        self.assertIsNone(driver._next_wider_beam_index(self.beams, widest))

    def test_drift_failure_takes_the_next_beam_depth_and_holds_the_column(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (30., 30., 6.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 18., 6.))
        worst = {"beam": 0.87, "column": 0.75}       # in band; column well under its ceiling
        next_column, next_beam, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=0, flags=flags(drift_ok=False))
        self.assertIn("drift", reasons)
        self.assertGreater(self.beams[next_beam][1], 18.)
        self.assertGreaterEqual(next_column, ci)      # never steps down while a requirement is unmet

    def test_compatible_beam_index_prefers_the_next_feasible_rung_above(self):
        sp_saved = {k: getattr(sp, k) for k in ("BAY_X", "BAY_Y", "STORY_H", "B_COL", "H_COL", "SLAB_THICKNESS_IN",
                                                 "BEAM_BAR_SIZE", "BEAM_STIRRUP_BAR_SIZE")}
        try:
            sp.BAY_X = sp.BAY_Y = 120.; sp.STORY_H = 168.; sp.B_COL = sp.H_COL = 36.; sp.SLAB_THICKNESS_IN = None
            beams = beam_ladder(span_in=120., story_height_in=168.)
            target = next(i for i, r in enumerate(beams) if r[:2] == (12., 24.))   # ln = 84 < 4d: forbidden
            chosen = driver._compatible_beam_index(beams, target)
            self.assertGreaterEqual(chosen, target)
            self.assertLess(beams[chosen][1], 24.)                # a wider, shallower rung above it
            self.assertGreaterEqual(beams[chosen][0], 12.)
        finally:
            for k, v in sp_saved.items():
                setattr(sp, k, v)

    def test_beam_capacity_shear_takes_the_wider_variant_first(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (30., 30., 6.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 20., 5.))
        worst = {"beam": 0.86, "column": 0.75}
        _c, next_beam, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=0,
            flags=flags(beam_section_adequate=False))
        self.assertIn("beam_capacity_shear", reasons)
        # width first: the shear section answers to bw d and short spans cap d
        self.assertEqual(self.beams[next_beam][:2], (14., 20.))

    def test_joint_scwb_failure_takes_the_next_column_size(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (26., 26., 8.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 16., 8.))
        worst = {"beam": 0.83, "column": 0.87}
        next_column, next_beam, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=ci,
            flags=flags(scwb_ok=False, joint_scwb_failed=True))
        self.assertIn("joint_scwb", reasons)
        # strength before geometry (2026-10-03): the same size at the top grade first ...
        self.assertEqual(self.columns[next_column], (26., 26., 10.))
        self.assertGreaterEqual(next_beam, bi)        # strength may still move the beam; never down
        # ... and only a column already at the top grade takes the next size, at the top grade
        again, _b, _r = driver._plan_next_rungs(
            self.columns, self.beams, next_column, bi, worst, 0.75, 1.0, scwb_index=next_column,
            flags=flags(scwb_ok=False, joint_scwb_failed=True))
        self.assertEqual(self.columns[again], (28., 28., 10.))

    def test_step_down_allowed_only_when_every_requirement_is_met(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (TOP, TOP, 6.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 18., 6.))
        worst = {"beam": 0.85, "column": 0.6}
        met, _b, reasons = driver._plan_next_rungs(self.columns, self.beams, ci, bi, worst, 0.75, 1.0,
                                                   scwb_index=0, flags=flags())
        self.assertEqual(reasons, [])
        self.assertLess(met, ci)                      # strength alone may shrink an oversize column
        held, _b, _r = driver._plan_next_rungs(self.columns, self.beams, ci, bi, worst, 0.75, 1.0,
                                               scwb_index=0, flags=flags(drift_ok=False))
        self.assertEqual(held, ci)                    # top of the ladder: drift grows the beam instead

    def test_joint_shear_jumps_to_the_rung_whose_joint_area_covers_the_shortfall(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (24., 24., 6.))
        bi = next(i for i, r in enumerate(self.beams) if r == (10., 20., 5.))
        worst = {"beam": 0.89, "column": 0.5}
        next_column, _b, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=0,
            flags=flags(joints_all_pass=False, joint_shear_ratio=1.6))
        self.assertIn("joint_shear_or_anchorage", reasons)
        b, h, fc = self.columns[next_column]
        self.assertGreaterEqual(b * h * fc ** .5, 24. * 24. * 6. ** .5 * 1.6 - 1e-9)
        # and not further than the first rung that covers it
        b0, h0, fc0 = self.columns[next_column - 1]
        self.assertLess(b0 * h0 * fc0 ** .5, 24. * 24. * 6. ** .5 * 1.6)

    def test_top_of_beam_ladder_falls_back_to_the_column(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (30., 30., 6.))
        bi = len(self.beams) - 1
        worst = {"beam": 0.5, "column": 0.5}
        next_column, next_beam, _r = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, worst, 0.75, 1.0, scwb_index=0, flags=flags(drift_ok=False))
        self.assertEqual(next_beam, bi)
        self.assertGreater(next_column, ci)

    def test_next_stronger_index_keeps_the_dimensions(self):
        ci = next(i for i, r in enumerate(self.columns) if r == (36., 36., 6.))
        for offset, strength in enumerate((7., 8., 9., 10.)):
            self.assertEqual(self.columns[driver._next_stronger_index(self.columns, ci + offset)], (36., 36., strength))
        self.assertIsNone(driver._next_stronger_index(self.columns, ci + 4))          # 36x36 fc-10: top column grade
        beams = beam_ladder(span_in=168., story_height_in=156.)                       # case_0013 ladder
        bi = next(i for i, r in enumerate(beams) if r == (20., 32., 5.))
        self.assertEqual(beams[driver._next_stronger_index(beams, bi)], (20., 32., 6.))
        self.assertIsNone(driver._next_stronger_index(beams, next(i for i, r in enumerate(beams) if r == (20., 32., 8.))))

    def test_beam_shear_at_the_top_dimensions_takes_the_next_concrete_strength(self):
        """case_0013: the widest, deepest rung its span allows (20x32 fc-4 on the two-width ladder of the
        run, 28x32 since 2026-09-27) leaves fc 5/6/8 as candidates."""
        beams = beam_ladder(span_in=168., story_height_in=156.)
        ci = next(i for i, r in enumerate(self.columns) if r == (36., 36., 6.))
        top_width = max(r[0] for r in beams if r[1] == 32.)
        bi = next(i for i, r in enumerate(beams) if r == (top_width, 32., 5.))
        self.assertIsNone(driver._next_deeper_beam_index(beams, bi))
        self.assertIsNone(driver._next_wider_beam_index(beams, bi))
        next_column, next_beam, reasons = driver._plan_next_rungs(
            self.columns, beams, ci, bi, {"beam": 0.84, "column": 0.44}, 0.85, 1.0, scwb_index=0,
            flags=flags(beam_section_adequate=False, capacity_accepted=False))
        self.assertEqual(reasons, ["beam_capacity_shear"])
        self.assertEqual(beams[next_beam], (top_width, 32., 6.))
        self.assertEqual(next_column, ci)                                             # the column is not the lever

    def test_column_shear_at_the_largest_size_takes_the_next_concrete_strength(self):
        """case_0073: column shear fails at the largest size at fc-6 (36x36 when it ran) with fc-8 unvisited."""
        ci = next(i for i, r in enumerate(self.columns) if r == (TOP, TOP, 6.))
        bi = next(i for i, r in enumerate(self.beams) if r == (14., 28., 8.))
        self.assertIsNone(driver._next_larger_column_index(self.columns, ci))
        next_column, next_beam, reasons = driver._plan_next_rungs(
            self.columns, self.beams, ci, bi, {"beam": 0.85, "column": 0.55}, 0.85, 1.0, scwb_index=0,
            flags=flags(column_section_adequate=False, capacity_accepted=False))
        self.assertEqual(reasons, ["column_capacity_shear"])
        self.assertEqual(self.columns[next_column], (TOP, TOP, 10.))                   # the top grade (2026-10-03)
        self.assertEqual(next_beam, bi)


class CandidatePlanTests(unittest.TestCase):
    """_plan_next_candidate: feasibility under the proposed columns, visited pairs, explicit stop reasons."""

    def setUp(self):
        self.columns = column_ladder()
        self.beams = beam_ladder(span_in=168., story_height_in=168.)                   # case_0017: 120 x 168 ft bays
        # the largest column at fc-6 (36x36 when case_0017 ran; the ladder top has moved since)
        self.c36 = next(i for i, r in enumerate(self.columns) if r == (TOP, TOP, 6.))
        # The widest 22-in rung: 16x22 fc-8 when case_0017 ran (two widths per depth), 20x22 fc-8 since
        # the 2026-09-27 ladder offers wider variants; the scenarios need the rung with no wider variant.
        self.b22 = next(i for i, r in enumerate(self.beams) if r == (20., 22., 8.))
        self.b24 = next(i for i, r in enumerate(self.beams) if r == (16., 24., 6.))

    def feasible(self, column_index):
        """Under 36-in columns on a 120-in bay (ln = 84 in) nothing deeper than 22 in fits 4d; smaller columns allow 24."""
        cap = 22. if self.columns[column_index][0] >= 36. else 24.
        return [i for i, r in enumerate(self.beams) if r[1] <= cap]

    def plan(self, ci, bi, visited, **overrides):
        return driver._plan_next_candidate(self.columns, self.beams, ci, bi, {"beam": 0.86, "column": 0.70}, 0.85, 1.0,
                                           0, flags(**overrides), visited, self.feasible)

    def test_infeasible_proposal_is_substituted_and_a_visited_pair_ends_with_an_explicit_reason(self):
        """case_0017 iteration 7: the planner asks for a 24-in beam under 36-in columns; the clear span forbids it."""
        visited = {(self.c36, self.b22)}
        plan = self.plan(self.c36, self.b22, visited, beam_section_adequate=False, capacity_accepted=False)
        self.assertEqual(self.beams[plan["proposed"]["beam"]][1], 24.)                 # what the strategy asked for
        self.assertEqual(plan["substitutions"][0]["stage"], "clear_span_compatibility")
        self.assertIsNone(plan["candidate"])                                           # 20x22 fc-8 is the top feasible rung
        self.assertEqual(plan["stop_reason"], "no_candidate_under_strategy")           # unvisited pairs exist: larger columns
        self.assertIn("beam_capacity_shear", plan["stop_detail"])

    def test_column_reasons_advance_the_column_past_a_visited_pair(self):
        visited = {(self.c36, self.b22)}
        plan = self.plan(self.c36, self.b22, visited, joints_all_pass=False, capacity_accepted=False)
        self.assertIsNotNone(plan["candidate"])
        self.assertEqual(self.columns[plan["candidate"]["column"]], (TOP, TOP, 10.))  # same size, top grade
        self.assertEqual(plan["candidate"]["beam"], self.b22)                           # the deeper proposal cannot fit
        self.assertTrue(plan["substitutions"])
        self.assertLessEqual({s["stage"] for s in plan["substitutions"]}, {"clear_span_compatibility", "visited_pair"})
        # With the strength rung already visited too, the next unvisited pair is one column rung further.
        visited.update((i, self.b22) for i in range(self.c36, len(self.columns)))
        plan = self.plan(self.c36, self.b22, visited, joints_all_pass=False, capacity_accepted=False)
        self.assertIsNone(plan["candidate"])                                            # all grades at the largest size have been visited
        self.assertEqual(plan["stop_reason"], "candidate_set_exhausted")

    def test_domain_exhaustion_is_named_when_no_unvisited_pair_remains(self):
        top = len(self.columns) - 1
        visited = {(top, j) for j in self.feasible(top)}
        plan = self.plan(top, self.b22, visited, joints_all_pass=False, capacity_accepted=False)
        self.assertIsNone(plan["candidate"])
        self.assertEqual(plan["stop_reason"], "candidate_set_exhausted")

    def test_unvisited_feasible_proposal_passes_through_unchanged(self):
        c32 = next(i for i, r in enumerate(self.columns) if r == (32., 32., 8.))
        plan = self.plan(c32, self.b22, {(c32, self.b22)}, beam_section_adequate=False, capacity_accepted=False)
        self.assertEqual(plan["substitutions"], [])
        self.assertEqual(self.beams[plan["candidate"]["beam"]][1], 24.)                 # deeper fits under 32-in columns
        self.assertEqual(plan["candidate"]["column"], c32)
        self.assertIsNone(plan["stop_reason"])

    def test_feasible_beam_indices_follow_both_clear_spans(self):
        saved = {k: getattr(sp, k) for k in ("BAY_X", "BAY_Y", "STORY_H", "SLAB_THICKNESS_IN")}
        try:
            # Explicit clear cover (slab set): #6 bars inside #4 hoops sit 2.375 in from the face, so a
            # 24-in beam has d = 21.625 in and needs 86.5 in of clear span: 88 under 32-in columns, 84 under 36.
            sp.BAY_X, sp.BAY_Y, sp.STORY_H, sp.SLAB_THICKNESS_IN = 120., 168., 168., 5.0
            beams = beam_ladder(span_in=168., story_height_in=168.)
            under_36 = driver._feasible_beam_indices(beams, 36., 36.)
            under_32 = driver._feasible_beam_indices(beams, 32., 32.)
            self.assertTrue(all(beams[i][1] <= 22. for i in under_36))
            self.assertTrue(any(beams[i][1] == 24. for i in under_32))
            self.assertLess(set(under_36), set(under_32))
        finally:
            for k, v in saved.items():
                setattr(sp, k, v)


class CapturedActionCheckTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "NUM_MODES", "ASCE_SDS"}:
            self.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 2, 1, 2
        sp.NUM_MODES = sp.NUM_FLOOR + 2
        sp.SLAB_THICKNESS_IN = None
        sp.FLOOR_TRANSFER = None
        sp.SLAB_REINFORCEMENT = None
        sp.SLAB_ACTIONS = None

    def tearDown(self):
        driver.ops.wipe()
        self.stack.close()

    def test_checks_on_captured_actions_match_the_live_domain(self):
        cfg = DesignConfig.from_structure_parameters()
        driver._sync_cfg_to_sp(cfg)
        driver._analyze_combination({"dead": 1.2, "live": 1.6, "ex": 1.0, "ey": 0.3, "live_pattern": "all"}, 0.5)
        cols, bx, by = get_element_tags()
        live = run_checks_phase1(cols, bx + by, cfg)
        captured = driver._capture_element_actions(1.2)
        driver.ops.wipe()                              # the live domain is gone
        offline = run_checks_phase1(cols, bx + by, cfg, member_actions=captured)
        self.assertEqual(set(live), set(offline))
        for tag, result in live.items():
            for name, state in result.limit_states.items():
                self.assertAlmostEqual(state.dcr, offline[tag].limit_states[name].dcr, places=12, msg=(tag, name))
            self.assertEqual(result.section_info.get("Pu_kip"), offline[tag].section_info.get("Pu_kip"))

    def test_missing_captured_member_is_an_error_not_a_zero(self):
        cfg = DesignConfig.from_structure_parameters()
        cols, bx, by = get_element_tags()
        with self.assertRaises(KeyError):
            run_checks_phase1(cols, bx + by, cfg, member_actions={})


class StrongColumnSteelTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR"}:
            self.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.B_COL = sp.H_COL = 24.
        sp.FC_COL_KSI = 5.
        sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS = 8, 3, 3, 2
        sp.COL_BAR_AREA = sp.rebar_area(8)
        sp.SLAB_THICKNESS_IN = 6.
        sp.FLOOR_TRANSFER = None
        sp.SLAB_REINFORCEMENT = None

    def tearDown(self):
        self.stack.close()

    def _joint_state(self, factor):
        """A failed check whose required sum is ``factor`` times the current column Mn."""
        from Design.SMRF_Joint_Adapter import record_section_capacity
        state = driver._state_record_core()
        p = 40.
        mn = min(record_section_capacity(state, "column", "x", face, p)["mn_kip_in"] for face in ("positive", "negative"))
        demand = factor * mn
        sway = {"nominal_strengths": True,
                "column_capacities": [{"tag": 1, "end": "j", "factored_axial_kip": p, "axial_min_kip": p * .5,
                                       "axial_max_kip": p * 1.5, "mn_kip_in": mn, "axial_envelope_checked": True}],
                "beam_capacities": []}
        check = {"id": "scwb", "status": "fail", "demand": demand, "capacity": mn,
                 "details": {"ratio_provided": mn / demand * 1.2, "sum_mnc_kip_in": mn}}
        return {"_failing": [(check, sway, "x")], "_state": state, "all_pass": False}, mn

    def test_least_cage_meeting_the_joint_rule_is_selected(self):
        cfg = DesignConfig()
        joint, current = self._joint_state(factor=1.35)
        update, exhausted = driver._scwb_column_steel(cfg, joint)
        self.assertFalse(exhausted)
        self.assertIsNotNone(update)
        from Design.SMRF_Joint_Adapter import record_section_capacity
        state = driver._state_record_core()
        state["reinforcement"].update(col_bar_size=update["bar_size"], col_top_bars=update["n_top"],
                                      col_bot_bars=update["n_bot"], col_side_bars=update["n_side"],
                                      col_bar_area_in2=sp.rebar_area(update["bar_size"]),
                                      col_bar_diameter_in=sp.rebar_diameter(update["bar_size"]),
                                      col_longitudinal_centroid_offset_in=sp.longitudinal_cover_in("column", update["bar_size"]))
        provided = min(record_section_capacity(state, "column", "x", face, p)["mn_kip_in"]
                       for p in (20., 40., 60.) for face in ("positive", "negative"))
        self.assertGreaterEqual(provided, current * 1.35)
        ast = (2 * update["n_top"] + 2 * update["n_side"]) * sp.rebar_area(update["bar_size"])
        self.assertLessEqual(ast / (sp.B_COL * sp.H_COL), cfg.rebar.rho_col_practical_max + 1e-12)
        # Least steel: no lighter admissible cage satisfies the rule.
        from Redesign import _col_candidates
        ag = sp.B_COL * sp.H_COL
        for candidate in _col_candidates(cfg.rebar.rho_col_min * ag, ast - 1e-9, cfg):
            trial = driver._state_record_core()
            trial["reinforcement"].update(col_bar_size=candidate[0], col_top_bars=candidate[1], col_bot_bars=candidate[2],
                                          col_side_bars=candidate[3], col_bar_area_in2=sp.rebar_area(candidate[0]),
                                          col_bar_diameter_in=sp.rebar_diameter(candidate[0]),
                                          col_longitudinal_centroid_offset_in=sp.longitudinal_cover_in("column", candidate[0]))
            lighter = min(record_section_capacity(trial, "column", "x", face, p)["mn_kip_in"]
                          for p in (20., 40., 60.) for face in ("positive", "negative"))
            self.assertLess(lighter, current * 1.35, candidate)

    def test_unreachable_requirement_is_reported_exhausted_not_faked(self):
        cfg = DesignConfig()
        joint, _current = self._joint_state(factor=50.)
        update, exhausted = driver._scwb_column_steel(cfg, joint)
        self.assertTrue(exhausted)
        self.assertIsNotNone(update)                  # the closest cage is installed so the shortfall is measured

    def test_no_failure_means_no_change(self):
        self.assertEqual(driver._scwb_column_steel(DesignConfig(), {"_failing": [], "_state": {}}), (None, False))


if __name__ == "__main__":
    unittest.main()
