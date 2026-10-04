"""Non-monotone feasibility, tradeoffs, budgets, and candidate isolation."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Design import Uniform_Search as search
from Design import Design_Driver as driver
import Structure_Parameters as sp


def snapshot(column, beam=(16., 22., 4.), steel=100.):
    return {"entry": {"column_section": list(column), "beam_section": list(beam),
                      "quantities": {"concrete_total_in3": column[0] ** 2 + beam[0] * beam[1],
                                     "total_steel_in3": steel, "column_fc_ksi": column[2],
                                     "beam_fc_ksi": beam[2], "distinct_form_size_count": 2}}}


class BoundedUniformSearch(unittest.TestCase):
    def test_promising_path_is_not_starved_by_old_joint_alternatives(self):
        old_parent = search.feasibility_priority({"beam": 5.15, "column": .84}, 1., False,
                                                [{"id": "joint"}] * 5, True)
        improved_parent = search.feasibility_priority({"beam": 1.11, "column": .74}, 1., False,
                                                     [{"id": "joint"}] * 17, False)
        self.assertLess(improved_parent, old_parent)
        feasible_strength = search.feasibility_priority({"beam": .9, "column": .5}, 1., True,
                                                       [{"id": "joint"}], True)
        self.assertLess(feasible_strength, improved_parent)

    def test_failure_does_not_prune_smaller_passing_island(self):
        columns = [(v, v, 4.) for v in (20., 22., 24., 26., 28., 30.)]
        beams = [(16., 22., 4.)]
        state = search.start(columns, beams, 5, 0, snapshot(columns[5]), 20)
        ci = search.next_pair(state)
        while ci is not None:
            accepted = columns[ci][0] in (22., 28.)  # failures at 26 and 24 hide an island from bisection
            ci = search.advance(state, columns, beams, ci, accepted,
                                snapshot=snapshot(columns[ci]) if accepted else None,
                                constraints={"joint": "pass" if accepted else "fail"})
        self.assertEqual(state["accepted"]["entry"]["column_section"][0], 22.)
        self.assertEqual({t["column"][0] for t in state["trials"]}, {20., 22., 24., 26., 28.})
        self.assertTrue(all(t["constraints"] for t in state["trials"]))

    def test_stronger_smaller_column_replaces_the_larger_one(self):
        """Strength before geometry (user decision 2026-10-03): a smaller column at a higher grade is the
        design; under the material-dominance rule it was kept only as a tradeoff and the larger column stood."""
        columns = [(28., 28., 4.), (28., 28., 8.), (30., 30., 4.)]
        beams = [(16., 22., 4.)]
        incumbent = snapshot(columns[2])
        state = search.start(columns, beams, 2, 0, incumbent, 8)
        ci = search.next_pair(state)
        tried = []
        while ci is not None:
            tried.append(columns[ci])
            ok = columns[ci][2] == 8.
            ci = search.advance(state, columns, beams, ci, ok, snapshot=snapshot(columns[ci]) if ok else None)
        self.assertEqual(tried, [(28., 28., 8.), (28., 28., 4.)])                 # top grade first, then the grade descent
        self.assertEqual(state["accepted"]["entry"]["column_section"], [28., 28., 8.])
        self.assertEqual(state["feasible"][-1]["decision"], "adopted")
        self.assertFalse(state["trials"][0]["material_dominance"])                # recorded: the grade went up
        self.assertEqual([t["phase"] for t in state["trials"]], ["size", "grade"])
        report = search.summary(state, columns, beams)
        self.assertEqual((report["policy"], report["stop_reason"]), ("uniform_size_first_v2", "declared_descent_complete"))

    def test_the_grade_descent_keeps_the_lowest_passing_grade_and_three_failures_end_the_size_descent(self):
        columns = [(s, s, g) for s in (20., 22., 24., 26., 28., 30.) for g in (5., 6., 8., 10.)]
        beams = [(16., 22., 5.)]
        start = columns.index((30., 30., 10.))
        state = search.start(columns, beams, start, 0, snapshot(columns[start], beams[0]), 24)
        ci = search.next_pair(state)
        tried = []
        while ci is not None:
            tried.append(columns[ci])
            ok = columns[ci][0] >= 28. and columns[ci][2] >= 6.
            ci = search.advance(state, columns, beams, ci, ok, snapshot=snapshot(columns[ci], beams[0]) if ok else None)
        self.assertEqual(tried, [(28., 28., 10.), (26., 26., 10.), (24., 24., 10.), (22., 22., 10.),   # three failures stop it
                                 (28., 28., 8.), (28., 28., 6.), (28., 28., 5.)])                  # 5 ksi fails: stop
        self.assertEqual(state["accepted"]["entry"]["column_section"], [28., 28., 6.])
        report = search.summary(state, columns, beams)
        self.assertEqual(report["untested_count"], 0)

    def test_a_size_that_fails_joint_shear_alone_is_retried_with_a_confining_beam(self):
        """r5 screen: 34 in at 10 ksi missed joint shear by 1 to 5 percent under a 24-in beam (0.71 of the column);
        the retry widens the beam to three quarters of the column, and the wider beam is then held."""
        columns = [(s, s, 10.) for s in (30., 32., 34., 36.)]
        beams = [(24., 30., 5.), (28., 30., 5.), (28., 30., 6.), (24., 32., 5.)]
        joint_only = {"strength_within_ceiling": True, "drift_accepted": True, "joint_scwb": {"all_pass": True},
                      "capacity_design_failed_checks": [{"id": "joint.shear_screen"}, {"id": "joint.shear_screen"}]}
        other = {**joint_only, "capacity_design_failed_checks": [{"id": "joint.shear_screen"}, {"id": "column.capacity_shear_section"}]}
        state = search.start(columns, beams, 3, 0, snapshot(columns[3], beams[0]), 24)
        log = []
        ci = search.next_pair(state)
        while ci is not None:
            bi = state["beam"]
            log.append((columns[ci][0], beams[bi][0]))
            ok = columns[ci][0] == 34. and beams[bi][0] == 28.            # only the confined 34-in joint passes
            constraints = None if ok else (joint_only if columns[ci][0] >= 32. else other)
            ci = search.advance(state, columns, beams, ci, ok, snapshot=snapshot(columns[ci], beams[bi]) if ok else None,
                                constraints=constraints, actual_beam_index=bi)
        # 34 with 24 fails -> retried with 28 (>= 25.5) and passes; 32 with the held 28-in beam fails (28 >= 24: no
        # retry); 30 fails on another check: no retry
        self.assertEqual(log, [(34., 24.), (34., 28.), (32., 28.), (30., 28.)])
        self.assertEqual(state["accepted"]["entry"]["column_section"][0], 34.)
        self.assertEqual(state["accepted"]["entry"]["beam_section"][0], 28.)
        self.assertEqual(state["trials"][0]["retry_with_wider_beam"], [28., 30., 5.])
        self.assertIsNone(search.confining_beam_retry(columns, beams, 2, 0, other))
        with_layering = {**joint_only, "capacity_design_failed_checks": [{"id": "joint.shear_screen"},
                                                                         {"id": "beam.bar_stacking_clear_of_slab_mats"}]}
        self.assertEqual(search.confining_beam_retry(columns, beams, 2, 0, with_layering), 1)   # layering beside joint shear: retried
        layering_only = {**joint_only, "capacity_design_failed_checks": [{"id": "beam.bar_stacking_clear_of_slab_mats"}]}
        self.assertIsNone(search.confining_beam_retry(columns, beams, 2, 0, layering_only))
        self.assertIsNone(search.confining_beam_retry(columns, beams, 2, 1, joint_only))       # already 3/4 of the column
        self.assertIsNone(search.confining_beam_retry(columns, beams, 2, 3, joint_only))       # no wider 32-in rung

    def test_budget_and_unresolved_analysis_preserve_incumbent_and_untested(self):
        columns = [(v, v, 4.) for v in (26., 28., 30.)]
        beams = [(16., 22., 4.), (20., 22., 4.)]
        incumbent = snapshot(columns[2], beams[1])
        state = search.start(columns, beams, 2, 1, incumbent, 1)
        ci = search.next_pair(state)
        self.assertIsNone(search.advance(state, columns, beams, ci, False,
                                        note="slab refinement unresolved", outcome="analysis_unresolved"))
        report = search.summary(state, columns, beams)
        self.assertIs(state["accepted"], incumbent)
        self.assertGreater(report["untested_count"], 0)
        self.assertEqual(report["stop_reason"], "trial_budget_exhausted")
        self.assertEqual(report["trials"][0]["outcome"], "analysis_unresolved")
        self.assertIsNone(search.next_pair(search.start(columns, beams, 2, 1, incumbent, 0)))

    def test_more_steel_or_missing_quantities_cannot_silently_win(self):
        incumbent = snapshot((30., 30., 4.))["entry"]["quantities"]
        candidate = snapshot((28., 28., 4.), steel=120.)["entry"]["quantities"]
        self.assertFalse(search.dominates(candidate, incumbent))
        candidate["total_steel_in3"] = None
        self.assertIsNone(search.dominates(candidate, incumbent))

    def test_joint_alternatives_are_siblings_of_the_failed_candidate(self):
        columns = [(30., 30., 4.), (30., 30., 6.), (32., 32., 4.)]
        beams = [(16., 22., 4.), (20., 22., 4.)]
        moves = search.targeted_alternatives(columns, beams, 0, 0, ["joint_shear_or_anchorage"])
        self.assertEqual([m[:2] for m in moves], [(1, 0), (0, 1)])

    def test_snapshots_and_restores_do_not_alias_nested_state(self):
        original = driver._capture_state()
        try:
            sp.SLAB_REINFORCEMENT = {"layout": {"example": [1]}}
            baseline = driver._capture_state()
            sp.SLAB_REINFORCEMENT["layout"]["example"].append(2)
            self.assertEqual(baseline["SLAB_REINFORCEMENT"]["layout"]["example"], [1])
            driver._restore_state(baseline)
            sp.SLAB_REINFORCEMENT["layout"]["example"].append(3)
            self.assertEqual(baseline["SLAB_REINFORCEMENT"]["layout"]["example"], [1])
        finally:
            driver._restore_state(original)


if __name__ == "__main__":
    unittest.main()
