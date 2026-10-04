"""The grouped search's own logic: ladders, taper rules, growth proposals, the dominance policy, rollback,
budgets and the candidate log (Design.Grouped_Search, 2026-10-02).

The candidate evaluation is replaced by a stub with stated rules, so each decision of the search is
checked against an outcome known in advance and the tests run in a fraction of a second. The real
evaluation, record and qualification are exercised end to end in test_grouped_record.
"""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

from Design import Grouped_Design as gd                                          # noqa: E402
from Design import Grouped_Search as gs                                          # noqa: E402
from Design.Section_Design import COLUMN_SIZES_IN                                # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

TOP_INTERIOR, BASE_INTERIOR = "s03_04__column__interior", "s01_02__column__interior"


def quantities(state, steel):
    concrete = sum(d.b_in * d.h_in * len(state.groups[gid]["member_tags"]) for gid, d in state.designs.items())
    forms = state.distinct_form_sizes()
    return {"concrete_total_in3": concrete, "concrete_total_yd3": concrete / 46656.0, "total_steel_in3": steel,
            "frame_steel_in3": 0.0 if steel is None else steel, "frame_steel_lb": 0.0,
            "distinct_form_sizes": forms, "distinct_form_size_count": sum(len(v) for v in forms.values())}


class Stub:
    """A candidate evaluation with declared rules: feasible(state), steel(state), constraints(state)."""

    def __init__(self, feasible, steel=lambda state: 100.0, constraints=lambda state: {}, infeasible=lambda state: None):
        self.feasible, self.steel, self.constraints, self.infeasible = feasible, steel, constraints, infeasible
        self.evaluated, self.installed = [], []

    def evaluate(self, state, cfg, context=None, label="candidate", keep_actions=True):
        self.evaluated.append(state)
        mg.install(state)
        record = {"version": "stub", "label": label, "sections": gd._section_table(state), "stages": {}, "solves": {},
                  "elapsed_seconds": 0.0, "state": state}
        reason = self.infeasible(state)
        if reason is not None:
            return {**record, "status": "infeasible", "feasible": False, "failed_stage": reason[0], "reason": reason[1],
                    "failure_evidence": reason[2]}
        return {**record, "status": "evaluated", "feasible": bool(self.feasible(state)), "constraints": self.constraints(state),
                "quantities": quantities(state, self.steel(state)), "margins": {}, "dcr": {"column": 0.5, "beam": 0.5},
                "result_identity": state.identity(), "evidence": {"capacity": {"joint_types": {}}}}

    def install(self, evaluation):
        self.installed.append(evaluation["state"])
        mg.install(evaluation["state"])
        return evaluation

    def run(self, seed, policy=None, log_path=None):
        with mock.patch.object(gd, "evaluate_candidate", self.evaluate), mock.patch.object(gd, "install_evaluation", self.install):
            return gs.search(seed, cfg=None, policy=policy, log_path=log_path, verbose=False)


def columns_at_least(size):
    return lambda state: all(d.b_in >= size for d in state.designs.values() if d.is_column)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.seed = gf.uniform_state()                       # 26 x 26 columns, 18 x 28 beams

    def tearDown(self):
        self.stack.__exit__(None, None, None)


class LaddersAndRules(Fixture):
    def test_a_column_steps_one_square_size_and_stops_at_the_ends_of_the_ladder(self):
        self.assertEqual(gs._column_step(gf.COLUMN, -1), (24.0, 24.0))
        self.assertEqual(gs._column_step(gf.COLUMN, +1), (28.0, 28.0))
        self.assertIsNone(gs._column_step(replace(gf.COLUMN, b_in=COLUMN_SIZES_IN[0], h_in=COLUMN_SIZES_IN[0]), -1))
        self.assertIsNone(gs._column_step(replace(gf.COLUMN, b_in=COLUMN_SIZES_IN[-1], h_in=COLUMN_SIZES_IN[-1]), +1))
        self.assertIsNone(gs._column_step(replace(gf.COLUMN, b_in=26.0, h_in=30.0), -1))      # not a square of the ladder

    def test_lighter_beam_rungs_have_less_concrete_and_come_strongest_first(self):
        beam = replace(gf.BEAM, member_type="beam_x")
        lighter = gs._beam_step(beam, "smaller")
        self.assertTrue(lighter)
        self.assertTrue(all(b * h < 18.0 * 28.0 for b, h in lighter))
        proxies = [b * h * h for b, h in lighter]
        self.assertEqual(proxies, sorted(proxies, reverse=True))
        self.assertTrue(set(lighter) <= set(gs.beam_rungs("beam_x")))

    def test_growth_moves_of_a_beam_go_deeper_or_wider_within_the_ladder(self):
        beam = replace(gf.BEAM, member_type="beam_y")
        deeper, wider = gs._beam_step(beam, "deeper"), gs._beam_step(beam, "wider")
        self.assertGreater(deeper[1], 28.0)
        self.assertEqual((wider[1], wider[0] > 18.0), (28.0, True))
        for rung in (deeper, wider):
            self.assertIn(rung, gs.beam_rungs("beam_y"))

    def test_a_column_may_not_be_smaller_than_the_one_above_nor_step_in_three_inches_per_face(self):
        self.assertIn("smaller", gs.column_taper_problem(self.seed, BASE_INTERIOR, (24.0, 24.0)))
        self.assertIsNone(gs.column_taper_problem(self.seed, TOP_INTERIOR, (24.0, 24.0)))        # 1 in per face
        self.assertIsNone(gs.column_taper_problem(self.seed, TOP_INTERIOR, (22.0, 22.0)))        # 2 in per face
        self.assertIn("3 in or more", gs.column_taper_problem(self.seed, TOP_INTERIOR, (20.0, 20.0)))
        wide_below = gs.resized(self.seed, BASE_INTERIOR, (32.0, 32.0))
        self.assertIn("3 in or more", gs.column_taper_problem(wide_below, BASE_INTERIOR, (36.0, 36.0)))  # 36 over 26 above

    def test_taper_enforcement_only_grows_and_says_why(self):
        state = gs.resized(self.seed, TOP_INTERIOR, (30.0, 30.0))                                # larger above than below
        fixed, changes = gs.enforce_column_taper(state)
        self.assertEqual(fixed.designs[BASE_INTERIOR].b_in, 30.0)
        self.assertEqual([c[0] for c in changes], [BASE_INTERIOR])
        state = gs.resized(self.seed, BASE_INTERIOR, (36.0, 36.0))                               # 5 in per face over 26
        fixed, changes = gs.enforce_column_taper(state)
        self.assertEqual((fixed.designs[BASE_INTERIOR].b_in, fixed.designs[TOP_INTERIOR].b_in), (36.0, 32.0))
        for gid, before, after, reason in changes:
            self.assertGreater(after[0], before[0])
            self.assertTrue(reason)
        untouched, none = gs.enforce_column_taper(self.seed)
        self.assertEqual((untouched.identity(), none), (self.seed.identity(), []))

    def test_a_beam_rung_must_fit_its_own_columns_and_the_slab(self):
        gid = "s01_02__beam_x__edge"
        self.assertIsNone(gs.beam_size_problem(self.seed, gid, (16.0, 26.0)))
        self.assertIn("18.6.2.1(b)", gs.beam_size_problem(self.seed, gid, (6.0, 28.0)))
        self.assertIn("slab", gs.beam_size_problem(self.seed, gid, (18.0, gf.SLAB)))
        self.assertIs(mg.active(), None)                                                          # the trial was not left installed

    def test_reduction_order_is_top_band_first_columns_before_beams(self):
        order = gs.reduction_order(self.seed)
        self.assertEqual(len(order), len(self.seed.groups))
        self.assertTrue(all(gid.startswith("s03_04") for gid in order[:8]))
        self.assertTrue(all("column" in gid for gid in order[:4]) and all("beam" in gid for gid in order[4:8]))

    def test_policy_rejects_negative_budgets_and_unknown_rules(self):
        self.assertEqual(gs.SearchPolicy().problems(), [])
        self.assertTrue(gs.SearchPolicy(max_reduction_trials=-1).problems())
        self.assertTrue(gs.SearchPolicy(improvement_policy="cheapest").problems())
        with self.assertRaises(ValueError):
            Stub(lambda s: True).run(self.seed, gs.SearchPolicy(max_feasibility_trials=1.5))


class DominancePolicy(unittest.TestCase):
    @staticmethod
    def q(concrete, steel, forms=2):
        return {"concrete_total_in3": concrete, "total_steel_in3": steel, "frame_steel_in3": steel, "distinct_form_size_count": forms}

    def decide(self, new, old):
        return gs.compare_quantities(self.q(*new), self.q(*old))[0]

    def test_a_move_is_accepted_when_neither_quantity_rises_and_one_falls(self):
        self.assertEqual(self.decide((90, 100), (100, 100)), "accepted")
        self.assertEqual(self.decide((100, 90), (100, 100)), "accepted")
        self.assertEqual(self.decide((90, 90), (100, 100)), "accepted")

    def test_a_move_that_lowers_one_and_raises_the_other_is_a_tradeoff_not_adopted(self):
        self.assertEqual(self.decide((90, 110), (100, 100)), "tradeoff")
        self.assertEqual(self.decide((110, 90), (100, 100)), "tradeoff")

    def test_a_move_that_lowers_nothing_is_rejected(self):
        self.assertEqual(self.decide((110, 100), (100, 100)), "rejected")
        self.assertEqual(self.decide((100, 100), (100, 100)), "rejected")

    def test_fewer_form_sizes_decide_only_a_tie(self):
        self.assertEqual(self.decide((100, 100, 1), (100, 100, 2)), "accepted")
        self.assertEqual(self.decide((100, 110, 1), (100, 100, 2)), "rejected")                  # more steel: forms do not rescue it

    def test_an_unknown_quantity_is_unresolved_never_zero(self):
        decision, detail = gs.compare_quantities(self.q(90, None), self.q(100, 100))
        self.assertEqual(decision, "unresolved")
        self.assertIn("not replaced by zero", detail["reason"])
        self.assertEqual(gs.compare_quantities(self.q(90, 50), self.q(100, None))[0], "unresolved")


class GrowthProposals(Fixture):
    def evaluation(self, **constraints):
        base = {"exhausted_groups": [], "capacity_design_failed_checks": [], "joint_checks_failed": [], "drift_failed_checks": []}
        return {"status": "evaluated", "constraints": {**base, **constraints},
                "evidence": {"capacity": {"joint_types": {
                    "joint_17": {"core_group": BASE_INTERIOR, "column_above_group": BASE_INTERIOR, "joints": ["joint_17", "joint_20"]},
                    "joint_29": {"core_group": BASE_INTERIOR, "column_above_group": TOP_INTERIOR, "joints": ["joint_29"]}}}}}

    def test_joint_shear_grows_the_column_that_forms_the_joint_core(self):
        moves = gs.propose_growth(self.evaluation(capacity_design_failed_checks=[
            {"id": "joint.shear_screen", "location": "joint_shear/joint_17/x"}]), self.seed)
        self.assertEqual({gid: size for gid, (size, _why) in moves.items()}, {BASE_INTERIOR: (28.0, 28.0)})

    def test_a_strong_column_failure_grows_both_columns_of_the_joint(self):
        moves = gs.propose_growth(self.evaluation(joint_checks_failed=[{"id": "scwb", "location": "joint_29/x/positive"}]), self.seed)
        self.assertEqual(set(moves), {BASE_INTERIOR, TOP_INTERIOR})

    def test_drift_grows_the_beams_of_the_story_band_in_that_direction(self):
        moves = gs.propose_growth(self.evaluation(drift_failed_checks=["demands.story_drift@story:3/Qy/y"]), self.seed)
        self.assertEqual(set(moves), {"s03_04__beam_y__edge", "s03_04__beam_y__interior"})
        self.assertTrue(all(size[1] > 28.0 for size, _why in moves.values()))                    # deeper, not wider

    def test_beam_capacity_shear_prefers_a_wider_beam_and_exhausted_groups_grow(self):
        moves = gs.propose_growth(self.evaluation(
            capacity_design_failed_checks=[{"id": "beam.capacity_shear_section", "location": "s01_02__beam_x__edge"}],
            exhausted_groups=[TOP_INTERIOR]), self.seed)
        self.assertEqual(moves["s01_02__beam_x__edge"][0][1], 28.0)
        self.assertGreater(moves["s01_02__beam_x__edge"][0][0], 18.0)
        self.assertEqual(moves[TOP_INTERIOR][0], (28.0, 28.0))

    def test_an_axial_domain_rejection_grows_the_group_that_owns_the_section(self):
        infeasible = {"status": "infeasible", "failed_stage": "capacity_design", "failure_evidence": {"group_id": TOP_INTERIOR}}
        self.assertEqual(set(gs.propose_growth(infeasible, self.seed)), {TOP_INTERIOR})
        no_thickness = {"status": "infeasible", "failed_stage": "slab_thickness", "failure_evidence": {}}
        self.assertEqual(set(gs.propose_growth(no_thickness, self.seed)),
                         {gid for gid, g in self.seed.groups.items() if g["member_type"] != "column"})

    def test_a_failure_that_names_no_group_proposes_nothing(self):
        self.assertEqual(gs.propose_growth({"status": "infeasible", "failed_stage": "slab_reinforcement", "failure_evidence": {}},
                                           self.seed), {})


class SearchRuns(Fixture):
    def test_reduction_adopts_smaller_columns_until_the_rule_stops_it_and_restores_after_every_rejected_trial(self):
        stub = Stub(columns_at_least(22.0))
        result = stub.run(self.seed, gs.SearchPolicy(max_reduction_trials=400))
        final = result["final"]["state"]
        self.assertEqual({d.b_in for d in final.designs.values() if d.is_column}, {22.0})
        # the stub lets every lighter beam rung pass too: no beam group ends with more concrete than the seed's 18 x 28
        self.assertTrue(all(d.b_in * d.h_in <= 18.0 * 28.0 for d in final.designs.values() if not d.is_column))
        self.assertEqual(result["stop"]["reason"], "reduction_sweep_without_accepted_move")
        self.assertIs(mg.active(), final)                                                       # the adopted design is in force
        self.assertIs(stub.installed[-1], final)
        log = result["log"]
        self.assertEqual([entry["index"] for entry in log], list(range(1, len(log) + 1)))
        accepted = [e for e in log if e["decision"] == "accepted"]
        rejected = [e for e in log if e["decision"] == "rejected"]
        self.assertEqual(result["counts"]["accepted_reductions"], len(accepted))
        self.assertTrue(rejected)
        for entry in rejected:                                                                    # 21-in columns are not feasible
            self.assertEqual(entry["proposal"]["to"], [21.0, 21.0])                             # one ladder step below 22 (1-in steps to 24)
        # every accepted move lowered concrete without raising steel, against the design it replaced
        for entry in accepted:
            comparison = entry["comparison"]
            self.assertLess(comparison["concrete_total_in3"]["new"], comparison["concrete_total_in3"]["current"])
            self.assertLessEqual(comparison["total_steel_in3"]["new"], comparison["total_steel_in3"]["current"])
        # a base column was never tried below the column above it: those moves are logged as skipped, with the reason
        skipped = [e for e in log if e["decision"] == "skipped"]
        self.assertTrue(any("column above" in e["decision_reason"] for e in skipped))
        self.assertTrue(all(e["evaluation"] is None for e in skipped))

    def test_a_move_that_needs_more_steel_is_kept_as_a_tradeoff_and_not_adopted(self):
        # steel rises as soon as any column is below 26 in
        stub = Stub(lambda state: True, steel=lambda state: 100.0 if columns_at_least(26.0)(state) else 120.0)
        result = stub.run(self.seed, gs.SearchPolicy(max_reduction_trials=8))
        self.assertEqual({d.b_in for d in result["final"]["state"].designs.values() if d.is_column}, {26.0})
        column_moves = [t for t in result["tradeoffs"] if "column" in t["proposal"]["group_id"]]
        self.assertTrue(column_moves)
        for tradeoff in column_moves:
            self.assertEqual(tradeoff["decision"], "tradeoff")
            self.assertIn("quantities", tradeoff)
            self.assertNotIn("groups", tradeoff["quantities"])
        self.assertIs(mg.active(), result["final"]["state"])

    def test_without_total_steel_nothing_is_adopted_and_the_moves_are_unresolved(self):
        stub = Stub(lambda state: True, steel=lambda state: None)
        result = stub.run(self.seed, gs.SearchPolicy(max_reduction_trials=5))
        self.assertEqual(result["final"]["state"].identity(), self.seed.identity())
        self.assertEqual({t["decision"] for t in result["tradeoffs"]}, {"unresolved"})
        self.assertEqual(result["counts"]["accepted_reductions"], 0)

    def test_the_reduction_budget_is_a_stop_reason_of_its_own(self):
        stub = Stub(columns_at_least(14.0))
        result = stub.run(self.seed, gs.SearchPolicy(max_reduction_trials=3))
        self.assertEqual(result["stop"]["reason"], "reduction_budget_exhausted")
        self.assertEqual(result["counts"]["reduction_candidates"], 3)
        self.assertEqual(len(stub.evaluated), 1 + 3)
        self.assertIsNotNone(result["final"])

    def test_a_zero_reduction_budget_returns_the_first_feasible_design(self):
        result = Stub(lambda state: True).run(self.seed, gs.SearchPolicy(max_reduction_trials=0))
        self.assertEqual(result["final"]["state"].identity(), self.seed.identity())
        self.assertEqual(result["counts"]["reduction_candidates"], 0)

    def test_feasibility_grows_the_groups_the_failures_name_until_the_design_is_feasible(self):
        def constraints(state):
            failing = state.designs[BASE_INTERIOR].b_in < 30.0
            return {"exhausted_groups": [BASE_INTERIOR] if failing else [], "capacity_design_failed_checks": [],
                    "joint_checks_failed": [], "drift_failed_checks": []}
        stub = Stub(lambda state: state.designs[BASE_INTERIOR].b_in >= 30.0, constraints=constraints)
        result = stub.run(self.seed, gs.SearchPolicy(max_reduction_trials=0))
        self.assertEqual(result["counts"]["feasibility_candidates"], 3)                          # 26, 28, 30
        self.assertEqual(result["final"]["state"].designs[BASE_INTERIOR].b_in, 30.0)
        self.assertEqual(result["final"]["state"].designs[TOP_INTERIOR].b_in, 26.0)              # only the named group grew
        growth = [e for e in result["log"] if e["proposal"]["kind"] == "growth"]
        self.assertEqual([m["group_id"] for e in growth for m in e["proposal"]["moves"]], [BASE_INTERIOR, BASE_INTERIOR])

    def test_feasibility_budget_exhaustion_is_not_reported_as_infeasibility(self):
        constraints = lambda state: {"exhausted_groups": [BASE_INTERIOR], "capacity_design_failed_checks": [],   # noqa: E731
                                     "joint_checks_failed": [], "drift_failed_checks": []}
        result = Stub(lambda state: False, constraints=constraints).run(self.seed, gs.SearchPolicy(max_feasibility_trials=2))
        self.assertIsNone(result["final"])
        self.assertEqual(result["stop"]["reason"], "feasibility_budget_exhausted")
        self.assertIn("not infeasibility", result["stop"]["detail"])
        self.assertEqual(len(result["log"]), 2)
        self.assertEqual({e["decision"] for e in result["log"]}, {"not_feasible"})

    def test_a_failure_with_no_growth_move_stops_with_its_own_reason_and_keeps_the_evidence(self):
        stub = Stub(lambda state: False, infeasible=lambda state: ("slab_reinforcement", "slab refinement comparison_failed",
                                                                 {"refinement": [{"floors": [1, 2]}]}))
        result = stub.run(self.seed)
        self.assertIsNone(result["final"])
        self.assertEqual(result["stop"]["reason"], "feasibility_no_growth_move")
        entry = result["log"][0]
        self.assertEqual(entry["evaluation"]["failed_stage"], "slab_reinforcement")
        self.assertEqual(entry["evaluation"]["failure_evidence"], {"refinement": [{"floors": [1, 2]}]})

    def test_the_seed_is_made_to_taper_before_it_is_evaluated_and_the_adjustment_is_logged(self):
        seed = gs.resized(self.seed, TOP_INTERIOR, (30.0, 30.0))
        result = Stub(lambda state: True).run(seed, gs.SearchPolicy(max_reduction_trials=0))
        self.assertEqual(result["final"]["state"].designs[BASE_INTERIOR].b_in, 30.0)
        adjustments = result["log"][0]["proposal"]["taper_adjustments"]
        self.assertEqual([a["group_id"] for a in adjustments], [BASE_INTERIOR])

    def test_every_candidate_is_appended_to_the_log_file_as_it_is_decided(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidates.jsonl"
            result = Stub(columns_at_least(24.0)).run(self.seed, gs.SearchPolicy(max_reduction_trials=6), log_path=path)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["index"] for line in lines], [entry["index"] for entry in result["log"]])
            self.assertEqual([line["decision"] for line in lines], [entry["decision"] for entry in result["log"]])
            for line in lines:
                if line["evaluation"] is not None:
                    self.assertNotIn("evidence", line["evaluation"])
                    self.assertNotIn("state", line["evaluation"])
            # a second search never appends to or replaces an existing log
            with self.assertRaises(RuntimeError):
                Stub(lambda state: True).run(self.seed, log_path=path)

    def test_an_unexpected_error_is_logged_with_its_traceback_and_raised(self):
        calls = {"n": 0}

        def broken(state, cfg, context=None, label="", keep_actions=True):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyError("not a design outcome")
            return good.evaluate(state, cfg, context, label)

        good = Stub(lambda state: True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidates.jsonl"
            with mock.patch.object(gd, "evaluate_candidate", broken), mock.patch.object(gd, "install_evaluation", good.install):
                with self.assertRaises(KeyError):
                    gs.search(self.seed, cfg=None, log_path=path, verbose=False)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([line["decision"] for line in lines], ["feasible", "error"])
        self.assertIn("KeyError", lines[1]["decision_reason"])
        self.assertIn("Traceback", lines[1]["traceback"])
        self.assertEqual(lines[1]["evaluation"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
