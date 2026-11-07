"""Controlled search behavior, not structural validation or a performance benchmark."""
import json
import math
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_fixture as cf
import grouped_fixture as gf
import Structure_Parameters as sp
from Design import Concrete_Grades as cg, Genetic_Search as ga, Grouped_Record as gr, Grouped_Search as gs
from Model import Member_Groups as mg
from test_grouped_search import Stub


class GeneticSearch(unittest.TestCase):
    def setUp(self):
        self.fixture = gf.frame()
        self.fixture.__enter__()
        common = set(gs.beam_rungs("beam_x")) & set(gs.beam_rungs("beam_y"))
        self.beam = min(common, key=lambda r: abs(r[0] - 16) + abs(r[1] - 30))
        self.seed, _ = gr.seed_from_sections((26, 26, 6), (*self.beam, 5), policy_id="system_three_story_v1")
        self.space = ga.DesignSpace(self.seed)

    def tearDown(self):
        self.fixture.__exit__(None, None, None)

    def run_search(self, stub=None, **kwargs):
        stub = stub or Stub(lambda s: True)
        return ga.search(self.seed, None, evaluator=stub.evaluate, verbose=False, **kwargs)

    def test_budget_reproducibility_unique_evaluations_and_crossover(self):
        policy = ga.GeneticPolicy(population_size=4, max_evaluations=12, max_generations=8,
                                  heuristic_probability=0, crossover_probability=1)
        stub_a, stub_b = Stub(lambda s: True), Stub(lambda s: True)
        a, b = (self.run_search(stub, policy=policy) for stub in (stub_a, stub_b))
        self.assertEqual(a, b)
        self.assertEqual(a["counts"]["evaluations"], 12)
        self.assertEqual(a["stop"], "evaluation_budget_exhausted")
        self.assertGreater(a["counts"]["crossovers"], 0)
        self.assertEqual(len({self.space.encode(s) for s in stub_a.evaluated}), 12)
        for state in stub_a.evaluated:
            self.space.validate_state(state)
            self.assertLessEqual(max(d.b_in for d in state.designs.values() if d.is_column), 36)
        self.assertIsNone(mg.active())
        self.assertFalse(a["production_accepted"])

    def test_one_evaluation_budget_does_not_expand_to_population_size(self):
        result = self.run_search(policy=ga.GeneticPolicy(max_evaluations=1))
        self.assertEqual(result["counts"]["evaluations"], 1)

    def test_generation_budget_includes_one_initial_population(self):
        result = self.run_search(policy=ga.GeneticPolicy(population_size=2, max_evaluations=50,
                                                       max_generations=1, heuristic_probability=0))
        self.assertEqual(result["stop"], "generation_budget_exhausted")
        self.assertEqual(result["counts"]["evaluations"], 4)

    def test_duplicate_repaired_candidates_do_not_spend_evaluation_budget(self):
        with mock.patch.object(ga, "mutate", return_value=(self.space.initial, {"kind": "random"})):
            result = self.run_search(policy=ga.GeneticPolicy(population_size=2, max_evaluations=10,
                                                           max_proposals_per_evaluation=2))
        self.assertEqual(result["counts"]["evaluations"], 1)
        self.assertGreater(result["counts"]["duplicates"], 0)
        self.assertEqual(result["stop"], "no_new_candidate_within_proposal_budget")

    def test_repair_restores_taper_ties_and_grade_families(self):
        genome = list(self.space.initial)
        columns = [i for i, g in enumerate(self.space.genes) if g.kind == "size" and "sfrs_column" in g.name]
        self.assertGreaterEqual(len(columns), 2)
        upper = max(columns, key=lambda i: min(self.seed.groups[self.space.genes[i].groups[0]]["stories_or_floors"]))
        genome[upper] = self.space.genes[upper].options.index((34, 34))
        repaired, state, changes = self.space.repair(tuple(genome))
        self.assertTrue(changes)
        self.space.validate_state(state)
        for gids in state.ties().values():
            self.assertEqual(len({(state.designs[g].b_in, state.designs[g].h_in) for g in gids}), 1)
        cg.require(state)
        self.assertEqual(self.space.encode(state), repaired)

    def test_capped_columns_still_offer_beam_congestion_alternatives(self):
        state = self.seed
        for gene in self.space.genes:
            if gene.kind == "size" and state.designs[gene.groups[0]].is_column:
                state = gs.apply_moves(state, {gene.name: ((36, 36), "test cap")})
        state = gs.regraded(state, "sfrs_column", 10)
        gid = next(g for g in state.groups if not state.designs[g].is_column and state.system_of(g) == mg.SYSTEM_SFRS)
        feedback = {"status": "evaluated", "feasible": False,
                    "constraints": {"lane_limited_beam_groups": [gid], "exhausted_groups": [gid]}}
        options = ga.heuristic_moves(feedback, state, self.space)
        beam_gene = next(g for g in self.space.genes if g.kind == "size" and gid in g.groups)
        alternatives = [m for m in options if beam_gene.name in m]
        self.assertTrue(alternatives)
        for moves in alternatives:
            _, repaired, _ = self.space.repair(self.space.encode(gs.apply_moves(state, moves)))
            self.assertTrue(all(d.b_in <= 36 for d in repaired.designs.values() if d.is_column))
        parent = {"index": 7, "feedback": feedback, "state": state}
        _, proposal = ga.mutate(self.space.encode(state), parent, self.space, random.Random(5),
                                ga.GeneticPolicy(heuristic_probability=1))
        self.assertEqual(proposal["kind"], "heuristic")
        self.assertEqual(proposal["source_candidate"], 7)

    def test_random_mutation_remains_available(self):
        genome, proposal = ga.mutate(self.space.initial, None, self.space, random.Random(1), ga.GeneticPolicy())
        self.assertNotEqual(genome, self.space.initial)
        self.assertEqual(proposal["kind"], "random")

    def test_unresolved_analysis_cannot_be_reported_as_feasible(self):
        stub = Stub(lambda s: False, constraints=lambda s: {"reinforcement_settled": False})
        result = self.run_search(stub, policy=ga.GeneticPolicy(max_evaluations=3))
        self.assertEqual(result["outcomes"]["unresolved"], 3)
        self.assertIsNone(result["representative"])
        self.assertFalse(result["pareto_candidates"])
        # A stopped evaluation is unresolved; sparse records without the contract are invalid, and neither is feasible.
        self.assertEqual(ga.assessment({"status": "infeasible", "failed_stage": "slab_refinement"}, None)[0], "unresolved")
        for evaluation in ({"status": "evaluated", "constraints": {"capacity_design_failed_checks": [{"status": "not_evaluated"}]}},
                           {"status": "evaluated", "evidence": {"joint_checks": [{"id": "joint_shear", "status": "not_evaluated"}]}}):
            self.assertEqual(ga.assessment(evaluation, None)[0], "invalid")

    def test_completed_failure_and_contradictory_pass(self):
        gid = sorted(self.seed.groups)[0]
        with mg.installed(self.seed):
            failed = cf.evaluation_record(self.seed, rows=cf.rows_for(self.seed, None, {("strength", "member.strength", gid): "fail"}))
        classification, fitness = ga.assessment(failed, None)
        self.assertEqual(classification, "check_failed")
        self.assertEqual(fitness[0], 1)
        self.assertGreater(fitness[2], 0)
        # The same record claiming feasibility contradicts its own rows: invalid, never feasible, never an exception path.
        with mg.installed(self.seed):
            self.assertEqual(ga.assessment({**failed, "feasible": True}, None)[0], "invalid")

    def test_screen_pass_with_missing_steel_or_joint_evidence_is_unresolved(self):
        unknown_scwb = lambda s: cf.rows_for(s, None, {k: ("not_evaluated", "slab_action_review_missing")  # noqa: E731
                                                       for k in cf.scwb_keys(s)})
        for rows, steel, reason in [(None, None, "total_steel_in3"), (unknown_scwb, 100, "strong-column")]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                stub = Stub(lambda s: True, steel=lambda s: steel, rows=rows)
                log = Path(tmp) / "candidates.jsonl"
                result = self.run_search(stub, policy=ga.GeneticPolicy(max_evaluations=1), log_path=log)
                self.assertEqual(result["outcomes"]["unresolved"], 1)
                self.assertIsNone(result["representative"])
                entry = json.loads(log.read_text())
                self.assertTrue(any(reason in item for item in entry["unresolved_reasons"]))
                self.assertTrue(entry["evaluation"]["candidate_screen_passed"])  # The raw screen result is preserved,
                self.assertFalse(entry["evaluation"]["feasible"])               # and is not code-design feasibility.

    def test_environment_restored_after_success_and_exception_and_error_logged(self):
        keys = ("SLAB_THICKNESS_IN", "FLOOR_TRANSFER", "SLAB_ACTIONS", "SLAB_REINFORCEMENT")
        original = {k: getattr(sp, k) for k in keys}
        stub = Stub(lambda s: True)
        starts = []
        def evaluate(state, *args, **kwargs):
            starts.append({k: getattr(sp, k) for k in keys})
            for k in keys:
                setattr(sp, k, 99)
            return stub.evaluate(state, *args, **kwargs)
        ga.search(self.seed, None, evaluator=evaluate, policy=ga.GeneticPolicy(max_evaluations=3), verbose=False)
        self.assertEqual(starts, [original] * 3)
        self.assertEqual({k: getattr(sp, k) for k in keys}, original)
        def broken(*args, **kwargs):
            sp.SLAB_THICKNESS_IN = 99
            raise RuntimeError("controlled solver error")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "candidates.jsonl"
            with mg.installed(self.seed):
                with self.assertRaisesRegex(RuntimeError, "controlled solver error"):
                    ga.search(self.seed, None, evaluator=broken, log_path=log, verbose=False)
                self.assertEqual(mg.active().identity(), self.seed.identity())
            self.assertEqual(json.loads(log.read_text())["decision"], "evaluation_error")
        self.assertEqual({k: getattr(sp, k) for k in keys}, original)

    def test_log_is_reviewable_and_existing_log_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "candidates.jsonl"
            result = self.run_search(policy=ga.GeneticPolicy(max_evaluations=3), log_path=log)
            rows = [json.loads(line) for line in log.read_text().splitlines() if '"candidate_index"' in line]
            self.assertEqual(len(rows), result["counts"]["evaluations"])
            self.assertTrue(all("evaluated_state" in row and "proposal" in row for row in rows))
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                self.run_search(log_path=log)

    def test_off_ladder_seed_refused(self):
        gid = next(g for g in self.seed.groups if self.seed.designs[g].is_column)
        with self.assertRaisesRegex(ValueError, "outside the declared ladder"):
            ga.DesignSpace(gs.resized(self.seed, gid, (37, 37)))

    def test_feasible_tradeoff_saved_without_replacing_representative(self):
        stub = Stub(lambda s: True)
        values = iter([(100, 100), (90, 110), (110, 90), (95, 95)])
        def evaluate(*args, **kwargs):
            raw = stub.evaluate(*args, **kwargs)
            concrete, steel = next(values)
            raw["quantities"].update(concrete_total_in3=concrete, total_steel_in3=steel, frame_steel_in3=steel)
            return raw
        result = ga.search(self.seed, None, evaluator=evaluate,
                           policy=ga.GeneticPolicy(max_evaluations=4, heuristic_probability=0), verbose=False)
        self.assertEqual(result["representative"]["candidate_index"], 4)
        self.assertEqual({c["candidate_index"] for c in result["pareto_candidates"]}, {2, 3, 4})


class Selection(unittest.TestCase):
    def candidate(self, index, concrete, steel, classification="feasible"):
        return {"index": index, "classification": classification,
                "fitness": (0, 0, 0) if classification == "feasible" else (1, 1, 1),
                "evaluation": {"quantities": {"concrete_total_in3": concrete, "total_steel_in3": steel}}}

    def test_feasible_precedes_cheaper_failed_and_crowding_preserves_extremes(self):
        candidates = [self.candidate(i, 10 + i, 20 - i) for i in range(5)]
        candidates.append(self.candidate(5, 0, 0, "check_failed"))
        kept = ga.survive(candidates, 2, random.Random(0))
        self.assertEqual({c["index"] for c in kept}, {0, 4})

    def test_missing_quantities_never_zero(self):
        self.assertEqual(ga.objectives(self.candidate(1, None, math.nan)), (math.inf, math.inf))

    def test_policy_rejects_invalid_domains(self):
        for changes in ({"population_size": 1}, {"max_evaluations": True}, {"max_generations": 0},
                        {"random_seed": 1.2}, {"heuristic_probability": math.nan}, {"crossover_probability": 1.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(ga.GeneticPolicy(), **changes).validate()


if __name__ == "__main__":
    unittest.main()
