"""Core-layout size guards and layout/steel retry state, without a building search.

Search outcomes are injected; section validation, evaluation installation, retry
coordination and saved grade-evidence validation use the production functions.
"""
import copy
import json
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import Structure_Parameters as sp
from Design import Grouped_Design as gd, Grouped_Record as gr, Grouped_Search as gs
from Design.Config import DesignConfig
from Model import Member_Groups as mg

CORE_POLICIES = (mg.SYSTEM_CORE_BANDED_POLICY_ID, mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID)
BANDED_POLICIES = (mg.SYSTEM_BANDED_POLICY_ID, *CORE_POLICIES)


def complete_evaluation(state, thickness=9.0):
    """Minimal retained evaluation for the real install_evaluation contract."""
    marker = {"fixture_policy": state.policy_id, "fixture_grade": sp.FY_KSI}
    return {"status": "evaluated", "feasible": True, "state": state,
            "evidence": {"slab": {
                "slab": {"thickness_in": thickness, "superimposed_dead_load_ksf": 0.025,
                         "live_load_mass_fraction": 0.2},
                "transfer": {"fixture_transfer": marker},
                "reinforcement": {"fixture_reinforcement": marker},
                "evidence": {"fixture_actions": marker}}}}


class SectionGuards(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(NUM_BAY_X=4, NUM_BAY_Y=4, NUM_FLOOR=6)
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)

    def state(self, policy):
        return mg.GroupedDesign.uniform(4, 4, 6, gf.COLUMN, gf.BEAM, policy_id=policy)

    def reject_install_and_reload(self, state, pattern):
        previous = mg.active()
        with self.assertRaisesRegex(mg.GroupedStateError, pattern):
            mg.install(state)
        self.assertIs(mg.active(), previous)
        with self.assertRaisesRegex(mg.GroupedStateError, pattern):
            mg.GroupedDesign.from_record(json.loads(json.dumps(state.to_record())))

    def test_upward_column_growth_is_refused_in_every_banded_layout(self):
        for policy in BANDED_POLICIES:
            for family in ("sfrs_column", "gravity_column"):
                with self.subTest(policy=policy, family=family):
                    state = self.state(policy)
                    mg.install(state)
                    bad = state.with_tied_designs({f"s04_06__{family}": replace(gf.COLUMN, b_in=28., h_in=28.)})
                    self.reject_install_and_reload(bad, "upward")

    def test_beam_dimensions_are_checked_individually_in_core_layouts(self):
        for policy in CORE_POLICIES:
            for family in ("sfrs_beam", "gravity_beam"):
                # Both examples have less area than 18 x 28, but one dimension grows.
                for b, h in ((20., 24.), (16., 30.)):
                    with self.subTest(policy=policy, family=family, section=(b, h)):
                        state = self.state(policy)
                        mg.install(state)
                        bad = state.with_tied_designs({f"s04_06__{family}": replace(gf.BEAM, b_in=b, h_in=h)})
                        self.reject_install_and_reload(bad, "upward")

    def test_layout_caps_apply_on_install_and_reload_including_uniform_policies(self):
        for banded in (False, True):
            for layout, cap in ((mg.SFRS_LAYOUT_PERIMETER, 36.), (mg.SFRS_LAYOUT_PERIMETER_CORE, 36.),
                                (mg.SFRS_LAYOUT_PERIMETER_CORE_LINES, 36.)):
                with self.subTest(layout=layout, banded=banded):
                    state = self.state(mg.layout_policy(layout, banded))
                    good = state.with_tied_designs({tie: replace(d, b_in=cap, h_in=cap)
                                                   for tie, d in state.tie_designs().items() if d.is_column})
                    mg.install(good)
                    self.assertEqual(mg.GroupedDesign.from_record(good.to_record()).identity(), good.identity())
                    for family in ("sfrs_column", "gravity_column"):
                        tie = ("s01_03__" if banded else "") + family
                        bad = good.with_tied_designs({tie: replace(gf.COLUMN, b_in=cap + 2., h_in=cap + 2.)})
                        self.reject_install_and_reload(bad, "cap")

    def test_valid_core_taper_still_round_trips(self):
        for policy in CORE_POLICIES:
            with self.subTest(policy=policy):
                state = self.state(policy).with_tied_designs({
                    "s01_03__sfrs_column": replace(gf.COLUMN, b_in=32., h_in=32.),
                    "s04_06__sfrs_column": replace(gf.COLUMN, b_in=28., h_in=28.),
                    "s01_03__sfrs_beam": replace(gf.BEAM, b_in=20., h_in=30.)})
                mg.install(state)
                self.assertEqual(mg.section_order_problems(state), [])
                self.assertEqual(mg.GroupedDesign.from_record(state.to_record()).identity(), state.identity())

    def test_core_beam_reduction_cannot_undercut_the_band_above(self):
        for policy in CORE_POLICIES:
            with self.subTest(policy=policy):
                self.assertIn("upward", gs.beam_size_problem(self.state(policy), "s01_03__beam_x__core", (16., 24.)))

    def test_search_independently_rejects_invalid_evaluator_output(self):
        for policy in CORE_POLICIES:
            with self.subTest(policy=policy):
                state = self.state(policy)
                bad = state.with_tied_designs({"s04_06__sfrs_beam": replace(gf.BEAM, b_in=20., h_in=30.)})
                with patch.object(gd, "evaluate_candidate", return_value={"state": bad, "status": "evaluated", "feasible": True}):
                    with self.assertRaisesRegex(mg.GroupedStateError, "upward"):
                        gs.search(state, DesignConfig.from_structure_parameters(), verbose=False)


class LayoutRetryState(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(NUM_BAY_X=4, NUM_BAY_Y=3, NUM_FLOOR=3, FY_KSI=60.,
                              REINFORCEMENT_SPECIFICATION=sp.REINFORCEMENT_SPECIFICATION_BY_GRADE[60.])
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)
        self.cfg = DesignConfig.from_structure_parameters()
        self.seed_state, self.seed = gr.seed_from_sections((24., 24., 6.), (14., 24., 5.),
                                                          policy_id=mg.SYSTEM_BANDED_POLICY_ID)

    def search_outcomes(self, selected_policy, selected_grade, error=False):
        calls, results = [], []

        def search(state, cfg, policy, **kwargs):
            calls.append((state.policy_id, sp.FY_KSI))
            self.assertEqual(cfg.materials.fy_ksi, sp.FY_KSI)
            found = state.policy_id == selected_policy and sp.FY_KSI == selected_grade
            if state.policy_id == mg.SYSTEM_BANDED_POLICY_ID:
                constraints = {"strength_within_ceiling": True, "drift_accepted": False}
            else:
                # Both core layouts try Grade 80, with distinct reasons, so a stale
                # candidate reference cannot accidentally validate the wrong layout.
                constraints = {"lane_limited_beam_groups": [state.policy_id + "/beam"]}
            e = complete_evaluation(state, 7. if state.policy_id == mg.SYSTEM_BANDED_POLICY_ID else 9.)
            gd.install_evaluation(e)
            if error and state.policy_id == selected_policy and sp.FY_KSI == selected_grade:
                raise RuntimeError("fixture core search failed")
            final = e if found else None
            public = {"status": "evaluated", "feasible": found, "constraints": {} if found else constraints}
            result = {"version": gs.SEARCH_VERSION, "policy": asdict(policy), "final": final,
                      "stop": {"reason": "reduction_budget_exhausted" if found else "feasibility_no_growth_move"},
                      "log": [{"index": 1, "phase": "feasibility", "longitudinal_grade_ksi": sp.FY_KSI,
                               "decision": "feasible" if found else "not_feasible", "evaluation": public}],
                      "counts": {"feasibility_candidates": 1, "reduction_candidates": 0, "skipped_moves": 0,
                                 "accepted_reductions": 0, "tradeoffs": 0},
                      "tradeoffs": [], "elapsed_seconds": 1.}
            results.append(result)
            return result
        return search, calls, results

    def run_ladder(self, search):
        with patch.object(gs, "search", side_effect=search):
            return gr._search_with_layout_ladder(self.cfg, self.seed_state, self.seed, gs.SearchPolicy(), None, False)

    def assert_selected_state(self, result, cfg):
        final = result["final"]
        slab = final["evidence"]["slab"]
        self.assertIs(mg.active(), final["state"])
        self.assertEqual(sp.FY_KSI, cfg.materials.fy_ksi)
        self.assertEqual(sp.REINFORCEMENT_SPECIFICATION, cfg.materials.reinforcement_specification)
        self.assertEqual(sp.SLAB_THICKNESS_IN, slab["slab"]["thickness_in"])
        self.assertEqual(sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF, slab["slab"]["superimposed_dead_load_ksf"])
        self.assertEqual(sp.SEISMIC_LIVE_LOAD_FRACTION, slab["slab"]["live_load_mass_fraction"])
        self.assertIs(sp.FLOOR_TRANSFER, slab["transfer"])
        self.assertIs(sp.SLAB_ACTIONS, slab["evidence"])
        self.assertIs(sp.SLAB_REINFORCEMENT, slab["reinforcement"])

    def test_selected_core_layout_and_slab_are_installed_for_both_grades(self):
        for selected_policy in CORE_POLICIES:
            for grade in (60., 80.):
                with self.subTest(policy=selected_policy, grade=grade):
                    mg.clear()
                    sp.apply_reinforcement_grade(60.)
                    search, calls, _ = self.search_outcomes(selected_policy, grade)
                    result, cfg = self.run_ladder(search)
                    self.assertEqual(calls[-1], (selected_policy, grade))
                    self.assert_selected_state(result, cfg)

    def test_grade_retry_reference_survives_two_or_three_layouts_and_json_round_trip(self):
        for selected_policy in CORE_POLICIES:
            with self.subTest(policy=selected_policy):
                mg.clear()
                sp.apply_reinforcement_grade(60.)
                search, calls, attempts = self.search_outcomes(selected_policy, 80.)
                result, cfg = self.run_ladder(search)
                saved = json.loads(json.dumps({"materials": {"fy_ksi": 80.}, "search": {
                    "policy": result["policy"], "steel_grade_trial": result["steel_grade_trial"],
                    "candidates": result["log"]}}))
                trial = saved["search"]["steel_grade_trial"]
                trigger = saved["search"]["candidates"][trial["trigger_candidate_index"] - 1]
                self.assertEqual(trigger["grouping_policy"], selected_policy)
                self.assertEqual(trigger["longitudinal_grade_ksi"], 60.)
                self.assertEqual(gs.grade80_retry_reasons(trigger["evaluation"]), trial["triggers"])
                self.assertEqual(trial["trigger_candidate_index"], len(calls) - 1)
                # Per-grade logs retain their local indices.
                self.assertTrue(all(r["log"][0]["index"] == 1 for r in attempts))
                sp.apply_reinforcement_grade(60.)
                self.assertEqual(gr._validated_record_grade(saved), 80.)
                self.assertEqual(sp.FY_KSI, 60.)
                wrong = copy.deepcopy(saved)
                wrong["search"]["steel_grade_trial"]["trigger_candidate_index"] = 1
                with self.assertRaisesRegex(ValueError, "retry lacks"):
                    gr._validated_record_grade(wrong)

    def test_failed_and_error_core_attempts_restore_previous_state(self):
        for error in (False, True):
            with self.subTest(error=error):
                mg.clear()
                sp.apply_reinforcement_grade(60.)
                search, _, _ = self.search_outcomes(mg.SYSTEM_CORE_BANDED_POLICY_ID if error else None, 80., error=error)
                result, cfg = self.run_ladder(search)
                self.assertIsNone(result["final"])
                self.assertEqual(cfg.materials.fy_ksi, 60.)
                if error:
                    # An attempt that raises is recorded and the ladder goes on to the next layout (2026-10-19:
                    # the overnight batch lost five cases to a crash inside the core-ring attempt).
                    errors = result["sfrs_layout_trial"]["attempt_errors"]
                    self.assertEqual([e["grouping_policy"] for e in errors], [mg.SYSTEM_CORE_BANDED_POLICY_ID])
                    self.assertIn("fixture core search failed", errors[0]["error"])
                    self.assertEqual([a["sfrs_layout"] for a in result["sfrs_layout_trial"]["attempts"]], list(mg.SFRS_LAYOUT_LADDER))
                self.assertIs(mg.active(), self.seed_state)
                self.assertEqual(sp.FY_KSI, 60.)
                self.assertEqual(sp.SLAB_THICKNESS_IN, 7.)
                self.assertEqual(sp.FLOOR_TRANSFER["fixture_transfer"]["fixture_policy"], mg.SYSTEM_BANDED_POLICY_ID)


if __name__ == "__main__":
    unittest.main()
