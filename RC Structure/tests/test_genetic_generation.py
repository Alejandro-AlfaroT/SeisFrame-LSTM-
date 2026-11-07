"""The bounded serial generation controller (Design.Genetic_Generation) end to end with a mocked evaluator.

The evaluator returns synthetic, contract-valid records from declared rules of the submitted design (genetic_fixture).
These tests establish controller semantics -- accounting, reproducibility, waiting and recovery states, identities,
retention and checkpoint/resume -- not structural credibility or search performance.
"""
import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genetic_fixture as gfx
import grouped_fixture as gf
import Structure_Parameters as sp
from Design import Code_Check_Contract as contract
from Design import Genetic_Generation as gen
from Model import Member_Groups as mg

DRIFT = ("drift", "demands.story_drift", "story:1/Qx/x")
CORNER = "s01_03__column__corner"


class RuleEvaluator:
    """Synthetic responses: story-1 X drift fails while the lower SFRS columns are smaller than ``drift_column``;
    ``unknown_scwb`` leaves every strong-column row unevaluated (slab actions not asserted); ``fail_on`` raises;
    ``drop`` leaves rows out; ``extra`` adds rows; each call spends ``floor_solves`` in the shared floor cache and
    ``seconds`` on ``clock`` (a FakeClock)."""

    def __init__(self, drift_column=28.0, unknown_scwb=False, fail_on=None, interrupt_at=None, drop=None, extra=None,
                 floor_solves=0, clock=None, seconds=0.0):
        self.drift_column, self.unknown_scwb, self.fail_on, self.interrupt_at = drift_column, unknown_scwb, fail_on, interrupt_at
        self.drop, self.extra, self.floor_solves, self.clock, self.seconds = drop, extra or {}, floor_solves, clock, seconds
        self.calls = []

    def __call__(self, state, cfg, context=None, label="candidate", keep_actions=True):
        self.calls.append(state.identity())
        if self.clock is not None:
            self.clock.now += self.seconds
        if context is not None and self.floor_solves:
            context["floor_solves"] = context.get("floor_solves", 0) + self.floor_solves
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt
        if self.fail_on is not None and self.fail_on(state):
            raise RuntimeError("synthetic solver failure")
        replace_rows = dict(self.extra)
        if state.designs[CORNER].b_in < self.drift_column:
            replace_rows[DRIFT] = gfx.row(*DRIFT, 0.03, 0.02, "<=", "in/in")
        if self.unknown_scwb:
            for key in contract.expected_inventory(state, cfg)["exact"]:
                if key[1] == "scwb":
                    replace_rows[key] = gfx.unknown_row(*key, "slab_action_review_missing")
        return gfx.record(state, cfg, replace=replace_rows, drop=self.drop)


class GradedDrift(RuleEvaluator):
    """Story-1 X drift fails by less as the corner column grows, and never passes within the catalog."""

    def __call__(self, state, cfg, context=None, label="candidate", keep_actions=True):
        demand = 0.02 + 0.0005 * max(0.0, 60.0 - state.designs[CORNER].b_in)
        self.extra = {DRIFT: gfx.row(*DRIFT, demand, 0.02, "<=", "in/in")}
        return super().__call__(state, cfg, context, label, keep_actions)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def lines(out, name):
    return (Path(out) / name).read_text(encoding="utf-8").splitlines()


def observations(out):
    return [json.loads(line) for line in (Path(out) / "observations.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["type"] == "observation"]


def summary(out):
    return [(o["id"], o["input_identity"], o["genome"], o["classification"], o["proposal"]["operator"],
             o.get("realized_identity")) for o in observations(out)]


def files(out, skip=()):
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in Path(out).rglob("*") if p.is_file() and p.name not in skip}


class Unprintable(RuntimeError):
    """An exception whose message cannot be formed."""

    def __str__(self):
        raise RuntimeError("this exception cannot be printed")


class Controller(unittest.TestCase):
    CONFIG = gen.GenerationConfig(random_seed=11, population_size=2, offspring_per_generation=2, max_generations=6,
                                  max_evaluations=14, initialization_policy="seed_r1_neighbors_v1",
                                  initialization_evaluations=1, guided_probability=1.0, max_exploration_evaluations=2,
                                  max_stagnant_generations=3)

    def setUp(self):
        self.fixture = gf.frame()
        self.fixture.__enter__()
        self.seed = gfx.seed_state()
        self.tmp = Path(tempfile.mkdtemp(prefix="ga_v3_"))
        self.environment = {key: getattr(sp, key) for key in gen.ENVIRONMENT_KEYS + ("FY_KSI",)}

    def tearDown(self):
        self.fixture.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def controller(self, out, evaluator, config=None, clock=None, case_id="fixture_case"):
        kwargs = {"clock": clock} if clock else {}
        return gen.GenerationRun(self.seed, None, config or self.CONFIG, self.tmp / out, case_id=case_id,
                                 basis={"fixture": "synthetic evaluator", "seed": self.seed.identity()},
                                 evaluator=evaluator, verbose=False, **kwargs)

    def run_new(self, out, evaluator=None, config=None, clock=None):
        return self.controller(out, evaluator or RuleEvaluator(), config, clock).start().run()

    def test_guided_search_reaches_a_code_accepted_design_and_accounts_for_every_proposal(self):
        # Two column steps above the seed: one random initializer step cannot reach it, a guided H5 path can.
        result = self.run_new("a", RuleEvaluator(drift_column=30.0), replace(self.CONFIG, target_accepted_designs=1))
        self.assertEqual(result["stop"]["reason"], "accepted_target_reached")
        self.assertEqual(result["stop"]["note"], gen.STOP_NOTES["accepted_target_reached"])
        self.assertFalse(result["stop"]["structural_infeasibility_established"])
        registry = result["views"]["accepted_registry"]
        self.assertEqual(registry["distinct_realized_designs"], 1)
        entry = registry["entries"][0]
        self.assertEqual((entry["qualification"], entry["nonlinear_performance"]["status"]), ("not_performed", "not_requested"))
        accepted = next(o for o in observations(self.tmp / "a") if o["id"] == entry["id"])
        self.assertGreaterEqual(accepted["cages"][CORNER]["b_in"], 30.0)
        self.assertIn("design", accepted["artifacts"])
        self.assertTrue((self.tmp / "a" / accepted["artifacts"]["design"]).exists())
        c = result["counts"]
        self.assertEqual(c["evaluations"], len(observations(self.tmp / "a")))
        self.assertEqual(c["ready"], c["evaluations"] + c["cache_hits"] + c["duplicates_within_generation"])
        self.assertEqual(c["proposals"], c["ready"] + c["noop"] + sum(c["rejected"].values()) + c["repair_refused"])
        lines = (self.tmp / "a" / "proposals.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), c["proposals"])
        self.assertIn("H5", c["proposals_by_operator"])
        self.assertEqual(result["metrics"]["first_acceptance"]["observation"], entry["id"])
        self.assertFalse(result["production_accepted"])
        self.assertIsNone(mg.active())
        self.assertEqual({key: getattr(sp, key) for key in self.environment}, self.environment)

    def test_runs_are_reproducible(self):
        self.run_new("a")
        self.run_new("b")
        self.assertEqual(summary(self.tmp / "a"), summary(self.tmp / "b"))
        a = json.loads((self.tmp / "a" / "result.json").read_text(encoding="utf-8"))
        b = json.loads((self.tmp / "b" / "result.json").read_text(encoding="utf-8"))
        for key in ("evaluation_seconds",):
            a["counts"].pop(key), b["counts"].pop(key)
        self.assertEqual(a["counts"], b["counts"])
        self.assertEqual(a["population"], b["population"])

    def test_identities_are_separate_and_bound(self):
        self.run_new("a")
        for o in observations(self.tmp / "a"):
            self.assertNotEqual(o["input_identity"], o["realized_identity"])
            # The declared basis with the run's proposal policy (the selection and beam-jump rules) joined to it.
            self.assertEqual(o["basis_id"], gen.diagnostics.digest({"fixture": "synthetic evaluator",
                                                                     "seed": self.seed.identity(),
                                                                     "proposal_policy": gen.proposal_policy(self.CONFIG)}))
            for d in o["diagnostics"]["diagnostics"]:
                self.assertEqual(d["parent_identity"], o["realized_identity"])
            if o["parent"]:
                self.assertIsNotNone(o["intervention"]["requested"])
                self.assertIn("reinforcement_vs_parent", o["intervention"])

    def test_missing_evidence_without_failures_waits_instead_of_strengthening(self):
        result = self.run_new("w", RuleEvaluator(drift_column=0.0, unknown_scwb=True))
        self.assertEqual(result["stop"]["reason"], "waiting_on_dependencies")
        self.assertEqual(result["stop"]["note"], gen.STOP_NOTES["waiting_on_dependencies"])
        self.assertIn("quarantined for contradictory evidence", result["stop"]["detail"])
        self.assertFalse(result["stop"]["structural_infeasibility_established"])
        c = result["counts"]
        self.assertEqual(c["guided_attempts"], 0)
        self.assertEqual(c["exploration_evaluations"], self.CONFIG.max_exploration_evaluations)
        queue = result["views"]["evidence_queue"]
        self.assertEqual(queue[0]["source"], "slab_action_engineering_assertion")
        self.assertEqual(len(queue[0]["blocked_observations"]), c["evaluations"])
        self.assertEqual(result["population_status"]["status"], "empty")
        self.assertEqual(result["metrics"]["distinct_code_accepted_realized_designs"], 0)

    def test_unknown_strong_column_rows_never_produce_hierarchy_moves(self):
        result = self.run_new("u", RuleEvaluator(unknown_scwb=True))
        operators = set(result["counts"]["proposals_by_operator"])
        self.assertNotIn("H1", operators)
        self.assertIn("H5", operators)
        for o in observations(self.tmp / "u"):
            self.assertNotIn("scwb", o["diagnostics"]["counts"]["by_reason"])
        self.assertEqual(result["metrics"]["distinct_code_accepted_realized_designs"], 0)

    def test_evaluation_exceptions_are_recorded_bounded_and_never_parents(self):
        seed_identity = self.seed.identity()
        evaluator = RuleEvaluator(fail_on=lambda state: state.identity() != seed_identity)
        result = self.run_new("e", evaluator, replace(self.CONFIG, max_evaluation_errors=2))
        self.assertEqual(result["stop"]["reason"], "evaluation_error_bound_reached")
        errors = [o for o in observations(self.tmp / "e") if o["outcome"] == "error"]
        self.assertEqual(len(errors), 2)
        recovery = {r["id"]: r["kind"] for r in result["views"]["analysis_recovery_queue"]}
        self.assertEqual({recovery[o["id"]] for o in errors}, {"evaluation_exception"})
        self.assertTrue(set(result["population"]).isdisjoint(o["id"] for o in errors))

    def test_budget_exhaustion_is_not_infeasibility(self):
        result = self.run_new("b", config=replace(self.CONFIG, max_evaluations=2))
        self.assertEqual(result["stop"]["reason"], "evaluation_budget_exhausted")
        self.assertEqual(result["stop"]["note"], gen.BOUND_NOTE)
        self.assertIn("neither structural infeasibility", result["stop"]["note"])
        self.assertTrue((self.tmp / "b" / "result.json").exists())

    def test_time_limit_is_checked_between_steps(self):
        ticks = iter(range(0, 10 ** 6, 50))
        result = self.run_new("t", config=replace(self.CONFIG, max_wall_seconds=60.0), clock=lambda: float(next(ticks)))
        self.assertEqual(result["stop"]["reason"], "time_limit_reached")

    def test_resume_after_interruption_reproduces_the_uninterrupted_run(self):
        self.run_new("full")
        interrupted = self.controller("part", RuleEvaluator(interrupt_at=5)).start()
        with self.assertRaises(KeyboardInterrupt):
            interrupted.run()
        checkpoint = json.loads((self.tmp / "part" / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(checkpoint["pending"])
        resumed = self.controller("part", RuleEvaluator()).resume()
        resumed.run()
        self.assertEqual(summary(self.tmp / "part"), summary(self.tmp / "full"))
        state = json.loads((self.tmp / "part" / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(len(state["resumes"]), 1)
        self.assertFalse(state["resumes"][0]["adopted_pending_observation"])

    def test_resume_adopts_an_evaluation_that_finished_before_the_interruption(self):
        self.run_new("full")
        original = gen.GenerationRun._absorb
        calls = {"n": 0}

        def interrupt(self_, record, pending, log=True):
            calls["n"] += 1
            if calls["n"] == 4:
                raise KeyboardInterrupt
            return original(self_, record, pending, log)
        with mock.patch.object(gen.GenerationRun, "_absorb", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.controller("part", RuleEvaluator()).start().run()
        evaluator = RuleEvaluator()
        resumed = self.controller("part", evaluator).resume()
        state = resumed.state["resumes"][0]
        self.assertTrue(state["adopted_pending_observation"])
        resumed.run()
        self.assertEqual(summary(self.tmp / "part"), summary(self.tmp / "full"))
        self.assertEqual(len(evaluator.calls) + 4, len(observations(self.tmp / "full")))

    def test_resume_sets_aside_a_cut_write_and_refuses_corruption_unchanged(self):
        self.run_new("full")
        with self.assertRaises(KeyboardInterrupt):
            self.controller("part", RuleEvaluator(interrupt_at=5)).start().run()
        path = self.tmp / "part" / "observations.jsonl"
        with open(path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write('{"type": "observation", "id": "o00')       # a write the interruption cut
        self.controller("part", RuleEvaluator()).resume().run()
        self.assertEqual(summary(self.tmp / "part"), summary(self.tmp / "full"))
        self.assertTrue(list((self.tmp / "part").glob("observations.partial.*.txt")))
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[1] = "{not a record\n"
        path.write_text("".join(lines), encoding="utf-8", newline="\n")
        before = path.read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            self.controller("part", RuleEvaluator()).resume()
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_new_accepted_designs_reset_stagnation(self):
        """Every design is accepted and the population is held at its oldest member, so the population never changes;
        each generation still adds accepted designs, which is progress, not stagnation."""
        original = gen.population.select

        def sticky(records, count, run, seed, generation=0, history=None):
            decision = original(records, count, run, seed, generation, history)
            decision["selected_ids"] = sorted(r["id"] for r in records)[:1]
            return decision
        config = replace(self.CONFIG, population_size=1, offspring_per_generation=2, max_generations=30,
                         max_evaluations=14, max_stagnant_generations=2)
        with mock.patch.object(gen.population, "select", sticky):
            result = self.run_new("s", RuleEvaluator(drift_column=0.0), config)
        self.assertEqual(result["stop"]["reason"], "evaluation_budget_exhausted", result["stop"])
        self.assertEqual(result["population"], ["o0001"])
        self.assertEqual(result["metrics"]["distinct_code_accepted_realized_designs"], result["counts"]["evaluations"])

    def test_a_refused_resume_changes_no_file(self):
        with self.assertRaises(KeyboardInterrupt):
            self.controller("part", RuleEvaluator(interrupt_at=5)).start().run()
        out = self.tmp / "part"
        first = observations(out)[0]
        with open(out / "observations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({**first, "id": "o9999", "input_identity": "unexplained"}) + "\n")
            stream.write('{"type": "observation", "id": "o99')
        with open(out / "proposals.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"step": 10 ** 6, "status": "noop"}) + "\n")
        before = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}
        with self.assertRaises(ValueError):
            self.controller("part", RuleEvaluator()).resume()
        self.assertEqual({p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}, before)

    def test_resume_refuses_another_configuration_and_start_refuses_an_existing_run(self):
        self.run_new("a", config=replace(self.CONFIG, max_evaluations=3))
        with self.assertRaises(ValueError):
            self.controller("a", RuleEvaluator(), replace(self.CONFIG, max_evaluations=3, population_size=3)).resume()
        with self.assertRaises(ValueError):
            self.controller("a", RuleEvaluator(), replace(self.CONFIG, max_evaluations=3), case_id="another_case").resume()
        with self.assertRaises(ValueError):
            self.controller("a", RuleEvaluator(), replace(self.CONFIG, max_evaluations=3)).start()

    def test_resume_refuses_records_no_interruption_can_leave(self):
        config = replace(self.CONFIG, max_evaluations=4)

        def finished(name):
            self.run_new(name, config=config)
            return self.tmp / name, json.loads((self.tmp / name / "checkpoint.json").read_text(encoding="utf-8"))
        out, checkpoint = finished("origin")
        with open(out / "observations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"type": "origin", "observation_id": "o9999", "origins": [], "status": "cache_hit",
                                     "step": checkpoint["step"] + 1}) + "\n")
        out2, _ = finished("garbled")
        rows = lines(out2, "proposals.jsonl")
        rows[1] = "{not a record"
        (out2 / "proposals.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
        out3, checkpoint3 = finished("ahead")
        with open(out3 / "generations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"step": checkpoint3["step"] + 2}) + "\n")
        refused = [out, out2, out3]

        def appended(name, target, *texts):
            path, checkpoint = finished(name)
            with open(path / target, "a", encoding="utf-8", newline="\n") as stream:
                for text in texts:
                    stream.write(text(checkpoint["step"]) if callable(text) else text)
            refused.append(path)

        def origin(step, observation="o0001"):
            return json.dumps({"type": "origin", "observation_id": observation, "origins": [], "status": "cache_hit",
                               "step": step}) + "\n"
        # An earlier step after a later one; a cut final line of a step the checkpoint does not explain.
        appended("regress", "proposals.jsonl", lambda s: json.dumps({"step": s + 1}) + "\n",
                 lambda s: json.dumps({"step": s}) + "\n")
        appended("cut_ahead", "proposals.jsonl", lambda s: json.dumps({"step": s + 5}))
        # Origins: two of the next step, one of the checkpoint's own step; a history line that is not an object.
        appended("two_origins", "observations.jsonl", lambda s: origin(s + 1), lambda s: origin(s + 1))
        appended("old_origin", "observations.jsonl", lambda s: origin(s))
        appended("not_object", "observations.jsonl", "[1, 2]\n")
        # An origin after a pending checkpoint (its step runs again inside the pending evaluation: none can follow).
        with self.assertRaises(KeyboardInterrupt):
            self.controller("pending", RuleEvaluator(interrupt_at=3), config).start().run()
        pending = json.loads((self.tmp / "pending" / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(pending["pending"])
        with open(self.tmp / "pending" / "observations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(origin(pending["step"] + 1))
        refused.append(self.tmp / "pending")
        for path in refused:
            before = {p.name: p.read_bytes() for p in path.rglob("*") if p.is_file()}
            with self.assertRaises(ValueError, msg=path.name):
                self.controller(path.name, RuleEvaluator(), config).resume()
            self.assertEqual({p.name: p.read_bytes() for p in path.rglob("*") if p.is_file()}, before, path.name)
        # The one origin an interruption can leave (a cache hit of the next step naming a known observation) resumes.
        path, checkpoint = finished("one_origin")
        with open(path / "observations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(origin(checkpoint["step"] + 1))
        self.controller("one_origin", RuleEvaluator(), config).resume().close()

    def test_errored_children_are_queued_in_their_generations_decision(self):
        evaluator = RuleEvaluator()
        evaluator.fail_on = lambda state: len(evaluator.calls) in (3, 5)
        self.run_new("q", evaluator, replace(self.CONFIG, max_evaluation_errors=5, max_evaluations=8))
        errored = {o["id"] for o in observations(self.tmp / "q") if o["outcome"] == "error"}
        self.assertEqual(errored, {"o0003", "o0005"})
        decisions = [json.loads(line) for line in lines(self.tmp / "q", "generations.jsonl")]
        queued = {item["id"]: item["reason"] for g in decisions if not g["initial"] for item in g["decision"]["queues"]}
        self.assertEqual({i: queued.get(i) for i in errored}, {i: "analysis_recovery_required" for i in errored})
        self.assertTrue(all(g["blocked_by"].get("analysis_recovery_required") for g in decisions
                            if not g["initial"] and errored & set(g["candidates"])))
        self.assertFalse(errored & {i for g in decisions for i in g["decision"]["selected_ids"]})
        # Absorbed into their own generation (not readmitted later as a cache hit): each generation's decision
        # counts every child evaluated in it, the raised ones included, and lists them among its candidates.
        records = observations(self.tmp / "q")
        for g in (g for g in decisions if not g["initial"]):
            own = {o["id"] for o in records if o["phase"] in ("generation", "exploration") and o["generation"] == g["generation"]}
            self.assertEqual(g["new_evaluations"], len(own), g["generation"])
            self.assertTrue(own <= set(g["candidates"]), g["generation"])
        self.assertTrue(all(any(o["id"] in g["candidates"] for g in decisions if g["generation"] == o["generation"]
                                and not g["initial"]) for o in records if o["id"] in errored))

    def test_exploration_generations_are_not_stagnation(self):
        """Every record waits on evidence: one exploration draw per generation spends the whole exploration budget,
        then the case waits (an exploration generation is replenishment, not search stagnation)."""
        config = replace(self.CONFIG, offspring_per_generation=1, max_exploration_evaluations=4,
                         max_stagnant_generations=2, max_generations=30, max_evaluations=30)
        result = self.run_new("x", RuleEvaluator(drift_column=0.0, unknown_scwb=True), config)
        self.assertEqual(result["stop"]["reason"], "waiting_on_dependencies", result["stop"])
        self.assertEqual(result["counts"]["exploration_evaluations"], 4)
        self.assertIn("evidence", result["stop"]["note"])
        self.assertFalse(result["stop"]["structural_infeasibility_established"])

    def test_a_budget_stop_selects_the_last_offspring(self):
        """The pilot limits: 16 evaluations end exactly after the third offspring generation. The stop selects it,
        so the run equals one whose generation limit ends it. The synthetic landscape is flat (every design fails the
        same drift row by the same amount, so none dominates another): under the pilot's stagnation bound it ends
        after two generations whose entries came only from the tie-break; the budget paths are compared without it."""
        flat = gen.GenerationConfig(random_seed=20261030, population_size=4, offspring_per_generation=4,
                                    max_generations=3, max_evaluations=16, initialization_evaluations=3,
                                    guided_probability=0.75, max_exploration_evaluations=4, max_stagnant_generations=2)
        stagnant = self.run_new("b2", RuleEvaluator(drift_column=99.0, unknown_scwb=True), flat)
        self.assertEqual((stagnant["stop"]["reason"], stagnant["generation"]), ("stagnation_bound_reached", 2))
        self.assertTrue(any(json.loads(line)["new_entrants"] for line in lines(self.tmp / "b2", "generations.jsonl")
                            if not json.loads(line)["initial"]))          # entries there were, none of them progress
        pilot = replace(flat, max_stagnant_generations=0)
        budget = self.run_new("b16", RuleEvaluator(drift_column=99.0, unknown_scwb=True), pilot)
        limit = self.run_new("b17", RuleEvaluator(drift_column=99.0, unknown_scwb=True),
                             replace(pilot, max_evaluations=17))
        self.assertEqual((budget["stop"]["reason"], limit["stop"]["reason"]),
                         ("evaluation_budget_exhausted", "generation_limit_reached"))
        self.assertEqual(summary(self.tmp / "b16"), summary(self.tmp / "b17"))
        self.assertEqual((budget["population"], budget["generation"]), (limit["population"], 3))
        selected_from = {i for line in lines(self.tmp / "b16", "generations.jsonl") for i in json.loads(line)["candidates"]}
        self.assertEqual({o["id"] for o in observations(self.tmp / "b16")} - selected_from, set())
        # The final selection is a step of its own: after every proposal, and no other generation record shares it.
        steps = [json.loads(line)["step"] for line in lines(self.tmp / "b16", "generations.jsonl")]
        self.assertGreater(steps[-1], max(json.loads(line)["step"] for line in lines(self.tmp / "b16", "proposals.jsonl")))
        self.assertEqual(len(steps), len(set(steps)))

    def test_a_budget_stop_during_initialization_selects_the_initial_population(self):
        result = self.run_new("i", RuleEvaluator(drift_column=0.0),
                              replace(self.CONFIG, initialization_evaluations=3, max_evaluations=2))
        self.assertEqual(result["stop"]["reason"], "evaluation_budget_exhausted")
        decisions = [json.loads(line) for line in lines(self.tmp / "i", "generations.jsonl")]
        self.assertEqual([(g["initial"], g["candidates"]) for g in decisions], [(True, ["o0001", "o0002"])])
        self.assertEqual(sorted(result["population"]), ["o0001", "o0002"])

    def test_missing_rows_beyond_the_sample_reach_the_evidence_queue(self):
        def drop(row):
            return row["family"] == "joints" and row["id"] == "scwb" or (
                (row["family"], row["id"], row["location"]) == ("strength", "member.strength", CORNER))
        result = self.run_new("m", RuleEvaluator(drift_column=0.0, drop=drop))
        records = observations(self.tmp / "m")
        self.assertTrue(all(o["missing_total"] > 40 and len(o["missing"]) == 40 for o in records))
        queue = {e["source"]: e for e in result["views"]["evidence_queue"]}
        self.assertEqual(queue["missing_row:joints|scwb"]["affected_rows"],
                         sum(o["missing_counts"]["joints|scwb"] for o in records))
        self.assertEqual(queue["missing_row:strength|member.strength"]["affected_rows"], len(records))

    def test_pattern_misses_of_one_check_form_one_source(self):
        """The contract reports a pattern miss per group by its description; it is keyed by the pattern's family
        and checks (from the contract's own inventory), so every group's miss of one check is one source."""
        result = self.run_new("pm", RuleEvaluator(drift_column=0.0, drop=lambda row: row["family"] == "joints"
                                                  and row["id"].startswith("beam_capacity_shear_")),
                              replace(self.CONFIG, max_evaluations=2))
        key = "joints|beam_capacity_shear_left+beam_capacity_shear_right"
        for o in observations(self.tmp / "pm"):
            self.assertIn(key, o["missing_counts"])
            self.assertFalse([k for k in o["missing_counts"] if k.startswith(("None|", "pattern|"))], o["missing_counts"])
            patterns = [m for m in o["missing"] if "pattern" in m]
            self.assertTrue(patterns)
            self.assertEqual(o["missing_counts"][key], o["missing_total"] - sum(
                n for k, n in o["missing_counts"].items() if k != key))
        sources = [e["source"] for e in result["views"]["evidence_queue"] if "beam_capacity_shear" in e["source"]]
        self.assertEqual(sources, ["missing_row:" + key])

    def test_resume_rewrites_are_atomic(self):
        self.run_new("full")
        with self.assertRaises(KeyboardInterrupt):
            self.controller("part", RuleEvaluator(interrupt_at=5)).start().run()
        out = self.tmp / "part"
        step = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))["step"]
        with open(out / "observations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write('{"type": "observation", "id": "o00')
        with open(out / "proposals.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"step": step, "status": "evaluated"}))    # cut before its newline
        before = {name: (out / name).read_bytes() for name in ("observations.jsonl", "proposals.jsonl")}
        with mock.patch.object(gen.os, "replace", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.controller("part", RuleEvaluator()).resume()
        self.assertEqual({name: (out / name).read_bytes() for name in before}, before)
        result = self.controller("part", RuleEvaluator()).resume().run()
        self.assertEqual(summary(out), summary(self.tmp / "full"))
        self.assertEqual(len(lines(out, "proposals.jsonl")), result["counts"]["proposals"])

    def test_a_cut_final_log_record_is_set_aside_not_glued(self):
        self.run_new("full")
        original = gen.GenerationRun._absorb
        calls = {"n": 0}

        def interrupt(self_, record, pending, log=True):
            calls["n"] += 1
            if calls["n"] == 4:
                raise KeyboardInterrupt
            return original(self_, record, pending, log)
        with mock.patch.object(gen.GenerationRun, "_absorb", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.controller("part", RuleEvaluator()).start().run()
        out = self.tmp / "part"
        checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
        with open(out / "proposals.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"step": checkpoint["step"], "status": "evaluated",
                                     "observation": checkpoint["pending"]["id"]}))
        result = self.controller("part", RuleEvaluator()).resume().run()
        self.assertEqual(summary(out), summary(self.tmp / "full"))
        records = [json.loads(line) for line in lines(out, "proposals.jsonl")]
        self.assertEqual(len(records), result["counts"]["proposals"])
        self.assertTrue((out / "proposals.jsonl").read_text(encoding="utf-8").endswith("\n"))
        self.assertTrue((out / "proposals.jsonl.after_checkpoint.1.txt").exists())

    def test_one_process_at_a_time_owns_the_run_directory(self):
        first = self.controller("l", RuleEvaluator()).start()
        with self.assertRaises(ValueError) as refused:
            self.controller("l", RuleEvaluator()).resume()
        self.assertIn("owned by another", str(refused.exception))
        seen = []
        evaluator = RuleEvaluator()

        def intrude(state):
            if len(evaluator.calls) == 2:
                try:
                    self.controller("l", RuleEvaluator()).resume()
                except ValueError as exc:
                    seen.append(str(exc))
            return False
        evaluator.fail_on = intrude
        first.evaluator = evaluator
        first.run()
        self.assertEqual(len(seen), 1)
        self.assertEqual(len({o["id"] for o in observations(self.tmp / "l")}), len(observations(self.tmp / "l")))
        self.controller("l", RuleEvaluator()).resume().run()          # released when the run ended

    def test_a_failure_after_the_evaluator_returns_is_a_recorded_error(self):
        original = gen.diagnostics.diagnose
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("synthetic processing failure")
            return original(*args, **kwargs)
        with mock.patch.object(gen.diagnostics, "diagnose", flaky):
            result = self.run_new("p", RuleEvaluator(), replace(self.CONFIG, max_evaluation_errors=3))
        second = observations(self.tmp / "p")[1]
        self.assertEqual((second["outcome"], second["error"]["stage"]), ("error", "post_evaluation"))
        self.assertEqual((result["counts"]["post_evaluation_errors"], result["counts"]["evaluation_errors"]), (1, 1))
        self.assertEqual(result["stop"]["reason"], "evaluation_budget_exhausted")
        self.assertIsNone(json.loads((self.tmp / "p" / "checkpoint.json").read_text(encoding="utf-8"))["pending"])
        # The evaluation it already wrote stays bound to the record (for analysis recovery), never orphaned.
        self.assertEqual(second["error"]["evaluator_status"], "evaluated")
        saved = self.tmp / "p" / second["artifacts"]["evaluation"]
        self.assertEqual(gen._sha(saved.read_text(encoding="utf-8")), second["artifacts"]["evaluation_sha256"])
        recovery = {r["id"]: r for r in result["views"]["analysis_recovery_queue"]}
        self.assertEqual((recovery[second["id"]]["stage"], recovery[second["id"]]["artifacts"]["evaluation"]),
                         ("post_evaluation", second["artifacts"]["evaluation"]))

    def test_no_artifact_is_left_without_its_record(self):
        """An accepted design whose record cannot be saved (a nonfinite value) is an error observation that still
        names the evaluation and design files written for it; the registry refuses nothing it never listed."""
        class NonfiniteOnce(RuleEvaluator):
            def __call__(self, *args, **kwargs):
                record = super().__call__(*args, **kwargs)
                if len(self.calls) == 1:
                    record["elapsed_seconds"] = float("nan")
                return record
        result = self.run_new("n", NonfiniteOnce(drift_column=0.0), replace(self.CONFIG, max_evaluations=3))
        out, records = self.tmp / "n", observations(self.tmp / "n")
        self.assertEqual((records[0]["outcome"], records[0]["error"]["stage"]), ("error", "post_evaluation"))
        self.assertTrue({"evaluation", "design"} <= set(records[0]["artifacts"]), records[0]["artifacts"])
        named = {o["artifacts"][k] for o in records for k in ("evaluation", "design") if k in o["artifacts"]}
        on_disk = {p.relative_to(out).as_posix() for folder in ("evaluations", "designs") for p in (out / folder).glob("*.json")}
        self.assertEqual(on_disk, named)
        self.assertNotIn(records[0]["id"], [e["id"] for e in result["views"]["accepted_registry"]["entries"]])

    def test_output_paths_beyond_the_windows_limit_are_refused_before_writing(self):
        out = self.tmp / "deep"
        limit = len(str(out.resolve())) + gen.LONGEST_ARTIFACT - 1
        with mock.patch.object(gen, "_windows_path_limit", return_value=limit):
            with self.assertRaises(ValueError):
                self.controller("deep", RuleEvaluator()).start()
        self.assertFalse(out.exists())

    def test_solver_work_of_raising_evaluations_is_accounted(self):
        seed_identity = self.seed.identity()
        evaluator = RuleEvaluator(fail_on=lambda state: state.identity() != seed_identity, floor_solves=3)
        result = self.run_new("s", evaluator, replace(self.CONFIG, max_evaluation_errors=2))
        errors = [o for o in observations(self.tmp / "s") if o["outcome"] == "error"]
        self.assertEqual([o["solves"] for o in errors], [{"frame": None, "floor": 3}] * 2)
        self.assertEqual(result["counts"]["solves"]["floor"], 3 * len(errors))
        self.assertEqual(result["counts"]["frame_solves_unrecorded_evaluations"], result["counts"]["evaluations"])

    def test_time_to_first_acceptance_includes_the_accepting_evaluation(self):
        clock = FakeClock()
        evaluator = RuleEvaluator(drift_column=0.0, clock=clock, seconds=100.0)
        result = self.run_new("t1", evaluator, replace(self.CONFIG, target_accepted_designs=1), clock=clock)
        self.assertEqual(result["metrics"]["first_acceptance"],
                         {"observation": "o0001", "evaluations": 1, "elapsed_seconds": 100.0})

    def test_children_record_their_target_and_accepted_designs_can_be_rebuilt(self):
        result = self.run_new("v", RuleEvaluator(drift_column=30.0), replace(self.CONFIG, target_accepted_designs=1))
        out = self.tmp / "v"
        guided = [o for o in observations(out) if o["proposal"]["operator"] == "H5" and o.get("vs_parent")]
        self.assertTrue(guided)
        for o in guided:
            versus = o["vs_parent"]
            self.assertEqual(versus["parent"], o["parent"])
            target = next(t for t in versus["targets"] if t["diagnostic_id"] == "drift|" + "|".join(DRIFT[1:]))
            self.assertEqual(target["parent"]["status"], "fail")
            self.assertEqual(target["resolved"], target["child"]["status"] == "pass")
            # The child's side is the child's own check, not the parent's.
            own = o["checks"].get(gen.population.canonical_key({"family": DRIFT[0], "id": DRIFT[1], "location": DRIFT[2]}))
            self.assertEqual(target["child"]["status"], own["s"] if own else "not_required")
            self.assertIn("counts", versus["collateral"])
        entry = result["views"]["accepted_registry"]["entries"][0]
        saved = json.loads((out / entry["artifacts"]["design"]).read_text(encoding="utf-8"))
        self.assertTrue(saved["realized_identity_reproducible"])
        self.assertEqual(contract.realized_identity(saved["state_identity"], saved["slab"]["slab"]["thickness_in"],
                                                    saved["slab"]["reinforcement"]), entry["realized_identity"])
        public = json.loads((out / f"evaluations/{entry['id']}.json").read_text(encoding="utf-8"))["code_checks"]
        self.assertEqual(len(saved["code_check_rows"]),
                         public["passing_rows_omitted_from_log"] + len(public["rows_not_passing"]))

    def test_abstentions_are_counted_once_per_routed_parent(self):
        key = ("detailing", "beam.minimum_width", "s01_03__beam_x__edge")
        result = self.run_new("ab", RuleEvaluator(extra={key: gfx.row(*key, 8.0, 10.0, ">=", "in")}))
        routed = json.loads((self.tmp / "ab" / "checkpoint.json").read_text(encoding="utf-8"))["routed_parents"]
        c = result["counts"]
        self.assertEqual(c["abstentions"].get("invalid_or_inapplicable_evidence"), len(routed))
        self.assertGreater(c["guided_attempts"], len(routed))

    def test_cache_hits_keep_their_origins_and_noops_spend_no_evaluation(self):
        config = replace(self.CONFIG, max_evaluations=1)
        self.controller("c", RuleEvaluator(), config).start().run()
        controller = self.controller("c", RuleEvaluator(), config).resume()
        self.addCleanup(controller.close)
        seed = controller.by_id["o0001"]
        controller.state["phase"] = "generation"
        controller._propose({"operator": "INIT_UNIFORM", "selected_operator": "INIT_UNIFORM",
                             "requested_genome": list(seed["genome"]),
                             "origins": [{"operator": "INIT_UNIFORM", "reason": "test repeat", "lever": "random"}]},
                            phase="generation", ctx=None)
        self.assertEqual(controller.state["counters"]["cache_hits"], 1)
        self.assertEqual(controller.state["pool"], ["o0001"])         # a child again: in this generation's selection
        origins = next(r for r in controller._records() if r["id"] == "o0001")["origins"]
        self.assertEqual([o["operator"] for o in origins], ["SEED", "INIT_UNIFORM"])
        ctx = controller._parent_context(seed)
        gene = next(g for g in ctx["genes"] if g["kind"] == "grade")
        controller._propose({"operator": "R1", "selected_operator": "R1", "parent_identity": ctx["parent_identity"],
                             "basis_identity": ctx["basis_identity"], "requested_genome": list(seed["genome"]),
                             "requested_gene": gene["name"], "from": None, "to": None,
                             "origins": [{"operator": "R1", "lever": "random"}]}, phase="generation", ctx=ctx, parent=seed)
        self.assertEqual(controller.state["counters"]["noop"], 1)
        self.assertEqual(controller.state["counters"]["evaluations"], 1)

    def test_a_spent_controller_writes_nothing(self):
        first = self.controller("sp", RuleEvaluator(interrupt_at=5))
        first.start()
        with self.assertRaises(KeyboardInterrupt):
            first.run()
        out = self.tmp / "sp"
        before = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}
        first.evaluator = RuleEvaluator()
        with self.assertRaises(ValueError) as refused:
            first.run()                                     # released when its run raised: spent, not an owner
        self.assertIn("does not own", str(refused.exception))
        self.assertEqual(first.evaluator.calls, [])         # refused before any evaluation
        for again in (first.resume, first.start):
            with self.assertRaises(ValueError) as reused:   # a spent controller is never reused
                again()
            self.assertIn("build a new GenerationRun", str(reused.exception))
        self.assertEqual({p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}, before)
        second = self.controller("sp", RuleEvaluator()).resume()
        with self.assertRaises(ValueError):                 # a known id is refused before anything is written
            second._history_add(dict(second.by_id["o0001"]))
        self.assertEqual((out / "observations.jsonl").read_bytes(), before["observations.jsonl"])
        final = second.run()
        self.assertEqual(len({o["id"] for o in observations(out)}), len(observations(out)))
        written = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}
        again = second.summary()                            # a spent controller still summarizes, as it saved
        self.assertEqual(again["counts"], json.loads((out / "result.json").read_text(encoding="utf-8"))["counts"])
        self.assertEqual(again["counts"], final["counts"])
        again["counts"]["evaluations"] = -1                 # a copy: changing it leaves the state as it was
        self.assertEqual(second.summary()["counts"],
                         json.loads((out / "result.json").read_text(encoding="utf-8"))["counts"])
        self.assertEqual({p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}, written)
        with self.assertRaises(ValueError) as unstarted:
            self.controller("never", RuleEvaluator()).summary()
        self.assertIn("holds no run", str(unstarted.exception))

    def test_a_failed_topology_write_is_written_by_the_next_observation(self):
        original = gen.GenerationRun._write
        failed = []

        def flaky(self_, name, value):
            if name.startswith("topology/") and not failed:
                failed.append(name)
                raise PermissionError("synthetic: the file is held by another process")
            return original(self_, name, value)
        checkpoints = {"n": 0}
        original_checkpoint = gen.GenerationRun._checkpoint

        def interrupt(self_):
            original_checkpoint(self_)
            checkpoints["n"] += 1
            if checkpoints["n"] >= 9 and self_.state["pending"] is None and not checkpoints.get("done"):
                checkpoints["done"] = True
                raise KeyboardInterrupt                     # between steps: the resume reads topologies from disk
        with mock.patch.object(gen.GenerationRun, "_write", flaky), \
                mock.patch.object(gen.GenerationRun, "_checkpoint", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_new("tw", RuleEvaluator(), replace(self.CONFIG, max_evaluation_errors=3))
        out = self.tmp / "tw"
        self.assertEqual(observations(out)[0]["error"]["stage"], "post_evaluation")
        cited = {o["topology_sha256"] for o in observations(out) if o.get("topology_sha256")}
        self.assertTrue(cited)
        self.assertTrue(all((out / f"topology/{sha}.json").exists() for sha in cited))
        result = self.controller("tw", RuleEvaluator(), replace(self.CONFIG, max_evaluation_errors=3)).resume().run()
        self.assertTrue(result["stop"]["reason"])

    def test_an_adopted_evaluation_keeps_its_time_and_error_stage(self):
        original_absorb, original_diagnose = gen.GenerationRun._absorb, gen.diagnostics.diagnose
        for name, diagnose_fails in (("adopt_time", False), ("adopt_error", True)):
            clock, calls = FakeClock(), {"absorb": 0, "diagnose": 0}

            def interrupt(self_, record, pending, log=True):
                calls["absorb"] += 1
                if calls["absorb"] == 1:
                    raise KeyboardInterrupt
                return original_absorb(self_, record, pending, log)

            def diagnose(*args, **kwargs):
                calls["diagnose"] += 1
                if diagnose_fails and calls["diagnose"] == 1:
                    raise RuntimeError("synthetic processing failure")
                return original_diagnose(*args, **kwargs)
            config = replace(self.CONFIG, target_accepted_designs=None if diagnose_fails else 1, max_evaluations=2,
                             max_evaluation_errors=3)
            with mock.patch.object(gen.GenerationRun, "_absorb", interrupt), \
                    mock.patch.object(gen.diagnostics, "diagnose", diagnose):
                with self.assertRaises(KeyboardInterrupt):
                    self.controller(name, RuleEvaluator(drift_column=0.0, clock=clock, seconds=100.0), config,
                                    clock=clock).start().run()
                resumed = self.controller(name, RuleEvaluator(drift_column=0.0, clock=clock, seconds=100.0), config,
                                          clock=clock).resume()
            self.assertTrue(resumed.state["resumes"][0]["adopted_pending_observation"])
            result = resumed.run()
            if diagnose_fails:
                self.assertEqual(result["counts"]["post_evaluation_errors"], 1)
            else:
                self.assertEqual(result["metrics"]["first_acceptance"]["elapsed_seconds"], 100.0)
                self.assertGreaterEqual(result["metrics"]["elapsed_seconds"], 100.0)

    def test_progress_is_a_new_design_entering_or_a_new_accepted_design(self):
        """A population change among earlier designs (a readmitted cache hit, a tie broken another way) is not
        progress, nor is a new design that entered by the tie-break alone; a generation with no new evaluation stops
        the search."""
        self.run_new("pg", config=replace(self.CONFIG, max_evaluations=6))
        controller = self.controller("pg", RuleEvaluator(), replace(self.CONFIG, max_evaluations=6)).resume()
        self.addCleanup(controller.close)
        fresh = next(o for o in controller.history if o["phase"] == "generation")
        original = gen.population.select
        chosen = {}

        def pick(records, count, run, seed, generation=0, history=None):
            decision = original(records, count, run, seed, generation, history=history)
            decision["selected_ids"] = list(chosen["ids"])
            decision["audits"].update(chosen.get("audits") or {})
            return decision

        def generation(population_, pool, new, selected, start=None, audits=None):
            identities = sorted(controller._accepted_identities()) if start is None else start
            controller.state.update(phase="generation", stop=None, generation=fresh["generation"],
                                    population=list(population_), pool=list(pool), new_in_generation=new,
                                    stagnant=0, accepted_identities_at_generation_start=identities)
            chosen.update(ids=selected, audits=audits)
            with mock.patch.object(gen.population, "select", pick):
                controller._end_generation()
            return controller.state["stagnant"], (controller.state["stop"] or {}).get("reason")

        def audit(tier, signature="same"):
            return {"eligible": True, "tier": tier, "inventory_id": "inventory", "known_items": 10,
                    "signature": signature, "progress": (0.5, 0.5)}

        def last():
            return json.loads(lines(self.tmp / "pg", "generations.jsonl")[-1])
        self.assertEqual(generation(["o0002"], ["o0001"], 0, ["o0001"]),
                         (1, "no_new_candidate_within_proposal_budget"))     # a readmitted cache hit: no progress
        self.assertEqual(generation(["o0002"], [fresh["id"], "o0001"], 1, ["o0001"]), (1, None))   # churn only
        # In a full population, a new entrant that dominates a member it replaced (a better tier) is progress; one
        # noncomparable with every member (another signature) entered by the tie-break alone, which is not. Into a
        # population with a free slot it is.
        full = ["o0001", "o0002"]
        better = {fresh["id"]: audit(1), "o0002": audit(2), "o0001": audit(1, "other")}
        apart = {fresh["id"]: audit(1, "one"), "o0002": audit(1, "two"), "o0001": audit(1, "three")}
        self.assertEqual(generation(full, [fresh["id"]], 1, ["o0001", fresh["id"]], audits=better), (0, None))
        self.assertEqual((last()["improving_entrants"], last()["free_slot_entry"]), ([fresh["id"]], False))
        self.assertEqual(generation(full, [fresh["id"]], 1, ["o0001", fresh["id"]], audits=apart), (1, None))
        logged = last()
        self.assertEqual((logged["new_entrants"], logged["improving_entrants"], logged["free_slot_entry"]),
                         ([fresh["id"]], [], False))
        self.assertEqual(generation(["o0002"], [fresh["id"]], 1, ["o0002", fresh["id"]], audits=apart), (0, None))
        self.assertTrue(last()["free_slot_entry"])
        self.assertEqual(generation([], ["o0001"], 0, ["o0001"]), (0, None))     # an empty population refilled
        self.assertEqual(generation([], [], 0, []), (0, "no_new_candidate_within_proposal_budget"))  # still empty
        # A design an earlier observation already realized, reached again through another input: no progress.
        twin = {**copy.deepcopy(controller.by_id["o0001"]), "id": "o9001", "phase": "generation",
                "generation": fresh["generation"], "input_identity": "another input of the same realized design"}
        controller.history.append(twin)
        controller.by_id[twin["id"]] = twin
        self.assertEqual(generation(["o0002"], ["o9001"], 1, ["o9001"]), (1, None))
        logged = last()
        self.assertEqual((logged["new_entrants"], logged["rediscovered_designs"]), ([], ["o9001"]))
        # A free slot taken without a new entrant is no progress: a readmitted cache hit or a rediscovered design.
        self.assertEqual(generation(["o0002"], [fresh["id"], "o0001"], 1, ["o0002", "o0001"]), (1, None))
        self.assertEqual(generation(["o0002"], ["o9001"], 1, ["o0002", "o9001"]), (1, None))
        self.assertFalse(last()["free_slot_entry"])
        # A member deduplicated under an earlier id that a cache hit readmitted (o9001 under o0001) has no audit of
        # its own in the decision: it is compared on its own record.
        own = gen.population.audit(controller.by_id["o0001"], controller.run_ctx)
        self.assertTrue(own["eligible"])
        ahead = {fresh["id"]: audit(own["tier"] - 1, "one"), "o0002": audit(own["tier"] - 1, "two")}
        self.assertEqual(generation(["o9001", "o0002"], [fresh["id"], "o0001"], 1, [fresh["id"], "o0002"],
                                    audits=ahead), (0, None))
        self.assertEqual((last()["improving_entrants"], last()["free_slot_entry"]), ([fresh["id"]], False))
        # A code-accepted identity added counts even when another left the registry (the net size unchanged).
        identities = sorted(controller._accepted_identities())
        self.assertTrue(identities)
        self.assertEqual(generation(["o0002"], [], 1, ["o0002"], start=identities[1:] + ["quarantined since"]),
                         (0, None))
        self.assertEqual(generation(["o0002"], [], 1, ["o0002"]), (1, None))
        # A member quarantined in this selection (another observation contradicts it) is no comparison, and its slot
        # is free: a new design entering beside the other member is a free-slot entry, not an improving one.
        self.assertTrue(controller.by_id["o0002"].get("realized_identity"))
        contradiction = {**copy.deepcopy(controller.by_id["o0002"]), "id": "o9002",
                         "quantities": {"contradicted": True}}
        controller.history.append(contradiction)
        controller.by_id[contradiction["id"]] = contradiction
        beaten = {fresh["id"]: audit(1, "one"), "o0002": audit(2, "one"), "o0001": audit(1, "two")}
        self.assertEqual(generation(full, [fresh["id"]], 1, ["o0001", fresh["id"]], audits=beaten), (0, None))
        logged = last()
        self.assertEqual((logged["improving_entrants"], logged["free_slot_entry"]), ([], True))
        self.assertIn({"id": "o0002", "reason": "quarantined_conflicting_evidence"}, logged["decision"]["queues"])

    def test_a_resumed_run_equals_the_uninterrupted_one_in_counters_and_logs(self):
        # The pilot limits without the stagnation bound: this landscape ends on it before the budget, and the budget
        # stop is the one with a final selection to interrupt.
        pilot = gen.GenerationConfig(random_seed=20261030, population_size=4, offspring_per_generation=4,
                                     max_generations=3, max_evaluations=16, initialization_evaluations=3,
                                     guided_probability=0.75, max_exploration_evaluations=4, max_stagnant_generations=0)
        evaluator = dict(drift_column=30.0, extra={("detailing", "beam.minimum_width", "s01_03__beam_x__edge"): gfx.row(
            "detailing", "beam.minimum_width", "s01_03__beam_x__edge", 8.0, 10.0, ">=", "in")})
        full = self.run_new("eq_full", RuleEvaluator(**evaluator), pilot)
        self.assertEqual(full["stop"]["reason"], "evaluation_budget_exhausted")      # a stop with a final selection

        def counts(result):
            return {k: v for k, v in result["counts"].items() if k != "evaluation_seconds"}

        def compare(name):
            out = self.tmp / name
            result = self.controller(name, RuleEvaluator(**evaluator), pilot).resume().run()
            self.assertEqual(counts(result), counts(full), name)
            for log in ("generations.jsonl", "proposals.jsonl"):
                self.assertEqual(lines(out, log), lines(self.tmp / "eq_full", log), (name, log))
            self.assertEqual(summary(out), summary(self.tmp / "eq_full"), name)
        # Inside an evaluation (a pending checkpoint), and inside the final selection after it logged its record.
        with self.assertRaises(KeyboardInterrupt):
            self.controller("eq_eval", RuleEvaluator(interrupt_at=9, **evaluator), pilot).start().run()
        compare("eq_eval")
        original = gen.GenerationRun._stop

        def interrupt(self_, reason, detail=""):
            raise KeyboardInterrupt
        with mock.patch.object(gen.GenerationRun, "_stop", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.controller("eq_final", RuleEvaluator(**evaluator), pilot).start().run()
        self.assertIs(gen.GenerationRun._stop, original)
        compare("eq_final")

    def test_sidecars_of_an_interrupted_resume_are_kept(self):
        original = gen.GenerationRun._absorb
        calls = {"n": 0}

        def interrupt(self_, record, pending, log=True):
            calls["n"] += 1
            if calls["n"] == 4:
                raise KeyboardInterrupt
            return original(self_, record, pending, log)
        with mock.patch.object(gen.GenerationRun, "_absorb", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.controller("sc", RuleEvaluator()).start().run()
        out = self.tmp / "sc"
        checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
        with open(out / "proposals.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"step": checkpoint["step"], "status": "evaluated",
                                     "observation": checkpoint["pending"]["id"]}))
        earlier = out / "proposals.jsonl.after_checkpoint.1.txt"     # left by a resume cut before its checkpoint
        earlier.write_text("an earlier resume's record\n", encoding="utf-8", newline="\n")
        resumed = self.controller("sc", RuleEvaluator()).resume()
        self.assertEqual(earlier.read_text(encoding="utf-8"), "an earlier resume's record\n")
        self.assertEqual(resumed.state["resumes"][0]["sidecar_number"], 2)
        self.assertTrue((out / "proposals.jsonl.after_checkpoint.2.txt").exists())
        resumed.close()

    def test_selection_is_given_the_whole_history(self):
        original = gen.population.select
        seen = []

        def record_history(records, count, run, seed, generation=0, history=None):
            seen.append((len(records), None if history is None else [r["id"] for r in history]))
            return original(records, count, run, seed, generation, history=history)
        with mock.patch.object(gen.population, "select", record_history):
            self.run_new("hs", config=replace(self.CONFIG, max_evaluations=8))
        ids = [o["id"] for o in observations(self.tmp / "hs")]
        self.assertTrue(seen)
        self.assertTrue(all(history is not None and history == ids[:len(history)] for _n, history in seen))
        self.assertEqual(seen[-1][1], ids)

    def test_a_reevaluated_pending_evaluation_keeps_its_time(self):
        """Interrupted inside an evaluation, the resume evaluates the input again; the checkpoint written after it
        carries that evaluation's time, so a second interruption before the next step loses none of it."""
        config = replace(self.CONFIG, max_evaluations=40, max_wall_seconds=550.0)

        def evaluator(clock, **kwargs):
            return RuleEvaluator(clock=clock, seconds=100.0, **kwargs)
        clock = FakeClock()
        full = self.controller("rt_full", evaluator(clock), config, clock=clock).start().run()
        self.assertEqual(full["stop"]["reason"], "time_limit_reached")
        clock = FakeClock()
        with self.assertRaises(KeyboardInterrupt):
            self.controller("rt", evaluator(clock, interrupt_at=3), config, clock=clock).start().run()
        resumed = self.controller("rt", evaluator(clock), config, clock=clock).resume()
        original = gen.GenerationRun._checkpoint

        def interrupt(self_):
            original(self_)
            raise KeyboardInterrupt                         # right after the checkpoint that follows the evaluation
        with mock.patch.object(gen.GenerationRun, "_checkpoint", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                resumed.run()
        saved = json.loads((self.tmp / "rt" / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertIsNone(saved["pending"])
        self.assertEqual(saved["elapsed_seconds"], saved["counters"]["evaluation_seconds"])
        result = self.controller("rt", evaluator(clock), config, clock=clock).resume().run()
        self.assertEqual((result["counts"]["evaluations"], result["metrics"]["elapsed_seconds"]),
                         (full["counts"]["evaluations"], full["metrics"]["elapsed_seconds"]))

    def test_an_unrecordable_evaluator_status_is_recorded_as_text(self):
        class Token:
            def __repr__(self):
                return "Token()"
        for name, status in (("nan", float("nan")), ("token", Token())):
            base = RuleEvaluator()

            def evaluator(state, cfg, context=None, label="candidate", keep_actions=True, _status=status, _base=base):
                record = _base(state, cfg, context, label, keep_actions)
                if len(_base.calls) == 2:
                    record["status"] = _status
                return record
            result = self.run_new(name, evaluator, replace(self.CONFIG, max_evaluations=3, max_evaluation_errors=3))
            error = observations(self.tmp / name)[1]["error"]
            self.assertEqual((error["stage"], error["evaluator_status"]), ("post_evaluation", repr(status)), name)
            self.assertEqual(result["counts"]["post_evaluation_errors"], 1, name)
            self.assertEqual(result["stop"]["reason"], "evaluation_budget_exhausted", name)

    def test_a_design_file_is_named_by_its_record_even_when_reading_it_back_fails(self):
        original = Path.read_text
        failed = []

        def flaky(self_, *args, **kwargs):
            if self_.parent.name == "designs" and not failed:
                failed.append(self_.name)
                raise PermissionError("synthetic: the file is held by another process")
            return original(self_, *args, **kwargs)
        config = replace(self.CONFIG, target_accepted_designs=1, max_evaluation_errors=3)
        with mock.patch.object(Path, "read_text", flaky):
            self.run_new("rb", RuleEvaluator(drift_column=0.0), config)
        out = self.tmp / "rb"
        self.assertEqual(failed, ["o0001.json"])
        first = observations(out)[0]
        self.assertEqual((first["error"]["stage"], first["artifacts"].get("design")), ("post_evaluation", "designs/o0001.json"))
        named = {v for o in observations(out) for k, v in o["artifacts"].items() if not k.endswith("sha256")}
        on_disk = {p.relative_to(out).as_posix() for folder in ("designs", "evaluations") for p in (out / folder).iterdir()}
        self.assertEqual(on_disk - named, set())

    def test_a_failed_replacement_leaves_no_temporary_file(self):
        original = os.replace
        failed = []

        def flaky(src, dst):
            if Path(dst).parent.name == "designs" and not failed:
                failed.append(Path(dst).name)
                raise PermissionError("synthetic: the target is held by another process")
            return original(src, dst)
        config = replace(self.CONFIG, target_accepted_designs=1, max_evaluation_errors=3)
        with mock.patch.object(gen.os, "replace", flaky):
            self.run_new("rp", RuleEvaluator(drift_column=0.0), config)
        out = self.tmp / "rp"
        self.assertEqual(failed, ["o0001.json"])
        self.assertEqual(list(out.rglob("*.tmp")), [])
        first = observations(out)[0]
        self.assertEqual(first["error"]["stage"], "post_evaluation")
        self.assertNotIn("design", first["artifacts"])                     # never named before it exists

    def test_an_exception_is_recorded_under_the_lease_and_set_aside_on_resume(self):
        original_absorb, original_write = gen.GenerationRun._absorb, gen.GenerationRun._write
        calls, writes = {"n": 0}, []

        def failing(self_, record, pending, log=True):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("synthetic controller failure")
            return original_absorb(self_, record, pending, log)

        def watched(self_, name, value):
            writes.append((name, self_._lease is not None))
            return original_write(self_, name, value)
        with mock.patch.object(gen.GenerationRun, "_absorb", failing), \
                mock.patch.object(gen.GenerationRun, "_write", watched):
            with self.assertRaises(RuntimeError):
                self.run_new("ej")
        out = self.tmp / "ej"
        self.assertIn(("error.json", True), writes)                         # written while the lease is held
        self.assertIn("RuntimeError: synthetic controller failure",
                      json.loads((out / "error.json").read_text(encoding="utf-8"))["error"])
        resumed = self.controller("ej", RuleEvaluator()).resume()
        self.assertFalse((out / "error.json").exists())
        self.assertEqual(resumed.state["resumes"][0]["error_set_aside"], "error.before_resume.1.json")
        self.assertTrue((out / "error.before_resume.1.json").exists())
        self.assertTrue(resumed.run()["stop"]["reason"])
        self.assertFalse((out / "error.json").exists())

    def test_an_earlier_observation_sidecar_or_error_record_is_never_overwritten(self):
        for kept in ("observations.partial.1.txt", "error.before_resume.1.json"):
            name = "osc_" + kept.split(".")[0]
            with self.assertRaises(KeyboardInterrupt):
                self.controller(name, RuleEvaluator(interrupt_at=3)).start().run()
            out = self.tmp / name
            (out / kept).write_text("an earlier resume's file\n", encoding="utf-8", newline="\n")
            with open(out / "observations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
                stream.write('{"type": "observation", "id": "cut')         # a write the interruption cut
            (out / "error.json").write_text("{}\n", encoding="utf-8", newline="\n")
            resumed = self.controller(name, RuleEvaluator()).resume()
            self.addCleanup(resumed.close)
            self.assertEqual((out / kept).read_text(encoding="utf-8"), "an earlier resume's file\n", kept)
            self.assertEqual(resumed.state["resumes"][0]["sidecar_number"], 2, kept)
            self.assertTrue((out / "observations.partial.2.txt").exists(), kept)
            self.assertTrue((out / "error.before_resume.2.json").exists(), kept)

    def test_the_waiting_detail_counts_the_quarantined_observations(self):
        self.run_new("qd", config=replace(self.CONFIG, max_evaluations=2))
        controller = self.controller("qd", RuleEvaluator(), replace(self.CONFIG, max_evaluations=2)).resume()
        self.addCleanup(controller.close)
        seed = controller.by_id["o0001"]
        for number, classification in ((9001, "unresolved"), (9002, "accepted")):
            twin = {**copy.deepcopy(seed), "id": f"o{number}", "classification": classification}
            controller.history.append(twin)
            controller.by_id[twin["id"]] = twin
        self.assertIn("quarantined for contradictory evidence (an evaluator determinism problem): 3 observation(s)",
                      controller._waiting_detail())

    def test_the_run_configuration_reaches_the_diagnostics(self):
        from Design.Config import DesignConfig
        cfg, seen = DesignConfig(), []
        original = gen.diagnostics.diagnose

        def spy(*args, **kwargs):
            seen.append(kwargs.get("cfg"))
            return original(*args, **kwargs)
        run = gen.GenerationRun(self.seed, cfg, replace(self.CONFIG, max_evaluations=2), self.tmp / "cfg",
                                case_id="fixture_case", basis={"fixture": "synthetic evaluator"},
                                evaluator=RuleEvaluator(), verbose=False)
        with mock.patch.object(gen.diagnostics, "diagnose", spy):
            run.start().run()
        self.assertTrue(seen)
        self.assertTrue(all(c is cfg for c in seen), seen)

    def test_withheld_levers_are_counted_once_per_routed_parent(self):
        original = gen.operators.guided_alternatives

        def with_withheld(ctx, diagnostics):
            return {**original(ctx, diagnostics), "withheld": {"column_up:synthetic_rule": 2}}
        with mock.patch.object(gen.operators, "guided_alternatives", with_withheld):
            result = self.run_new("wl")
        routed = json.loads((self.tmp / "wl" / "checkpoint.json").read_text(encoding="utf-8"))["routed_parents"]
        self.assertTrue(routed)
        self.assertEqual(result["counts"]["withheld_levers"], {"column_up:synthetic_rule": 2 * len(routed)})
        self.assertGreater(result["counts"]["guided_attempts"], len(routed))

    def test_stagnation_counts_against_the_identities_accepted_when_each_generation_began(self):
        """Only the seed is accepted: it is added once, at the initial selection, and never again, so generations that
        bring no new design end the run on the bound."""
        class SeedOnly(RuleEvaluator):
            def __call__(self, state, cfg, context=None, label="candidate", keep_actions=True):
                self.drift_column = 0.0 if not self.calls else 1000.0
                return super().__call__(state, cfg, context, label, keep_actions)
        config = replace(self.CONFIG, population_size=1, max_generations=30, max_evaluations=40,
                         max_stagnant_generations=2)
        result = self.run_new("sa", SeedOnly(), config)
        self.assertEqual(result["stop"]["reason"], "stagnation_bound_reached", result["stop"])
        self.assertEqual(result["metrics"]["distinct_code_accepted_realized_designs"], 1)
        added = [json.loads(line)["accepted_identities_added"] for line in lines(self.tmp / "sa", "generations.jsonl")]
        self.assertEqual((added[0], set(added[1:])), (1, {0}))

    def test_an_exception_without_a_printable_message_is_still_recorded(self):
        evaluator, original = RuleEvaluator(), gen.diagnostics.diagnose
        calls = {"diagnose": 0}

        def raising(state, cfg, context=None, label="candidate", keep_actions=True):
            record = evaluator(state, cfg, context, label, keep_actions)
            if len(evaluator.calls) == 2:
                raise Unprintable()                         # at the evaluator
            return record

        def diagnose(*args, **kwargs):
            calls["diagnose"] += 1
            if calls["diagnose"] == 2:
                raise Unprintable()                         # after the evaluator returned
            return original(*args, **kwargs)
        with mock.patch.object(gen.diagnostics, "diagnose", diagnose):
            result = self.run_new("up", raising, replace(self.CONFIG, max_evaluations=4, max_evaluation_errors=3))
        errors = [o["error"] for o in observations(self.tmp / "up") if o.get("error")]
        self.assertEqual(sorted(e["stage"] for e in errors), ["evaluator", "post_evaluation"])
        self.assertEqual({e["message"] for e in errors}, {"<Unprintable without a printable message>"})
        self.assertEqual(result["stop"]["reason"], "evaluation_budget_exhausted")
        # One that ends the run is recorded in error.json the same way.
        with mock.patch.object(gen.GenerationRun, "_absorb", side_effect=Unprintable()):
            with self.assertRaises(Unprintable):
                self.run_new("ux")
        self.assertEqual(json.loads((self.tmp / "ux" / "error.json").read_text(encoding="utf-8"))["error"],
                         "Unprintable: <Unprintable without a printable message>")

    def test_the_controller_checks_the_saved_request_under_its_lease(self):
        config = replace(self.CONFIG, max_evaluations=2)
        self.controller("rq", RuleEvaluator(), config).start(request_text=json.dumps({"sha256": "A"})).close()
        out = self.tmp / "rq"
        before = files(out)
        with self.assertRaises(gen.RunDirectoryError) as refused:
            self.controller("rq", RuleEvaluator(), config).resume(request_sha256="B")
        self.assertEqual(refused.exception.kind, "request_differs")
        self.assertEqual(files(out), before)
        (self.tmp / "nr").mkdir()
        with self.assertRaises(gen.RunDirectoryError) as refused:
            self.controller("nr", RuleEvaluator()).resume(request_sha256="A")
        self.assertEqual(refused.exception.kind, "no_request")
        self.assertEqual(sorted(files(self.tmp / "nr")), [gen.LEASE])        # the lease marker only
        # A start without a request does not take a directory another run claimed with its saved request.
        claimed = self.tmp / "claimed"
        claimed.mkdir()
        (claimed / "request.json").write_text(json.dumps({"sha256": "A"}), encoding="utf-8")
        with self.assertRaises(gen.RunDirectoryError) as refused:
            self.controller("claimed", RuleEvaluator()).start()
        self.assertEqual(refused.exception.kind, "has_request")
        self.assertEqual(sorted(files(claimed)), [gen.LEASE, "request.json"])
        self.assertEqual(self.controller("rq", RuleEvaluator(), config).resume(request_sha256="A").run()["stop"]["reason"],
                         "evaluation_budget_exhausted")

    def test_a_spent_controller_touches_no_file_even_through_its_writers(self):
        controller = self.controller("sw", RuleEvaluator(), replace(self.CONFIG, max_evaluations=2))
        controller.start().run()
        out = self.tmp / "sw"
        (out / "proposals.jsonl.tmp").write_text("another owner's file\n", encoding="utf-8")
        before = files(out)
        for write in (lambda: controller._write("newdir/x.json", {}),
                      lambda: controller._replace_text("proposals.jsonl", "x\n")):
            with self.assertRaises(ValueError):
                write()
        self.assertEqual(files(out), before)
        self.assertFalse((out / "newdir").exists())

    def test_a_run_interrupted_in_its_first_evaluation_leaves_a_spent_controller(self):
        first = self.controller("s0", RuleEvaluator(interrupt_at=1)).start()
        with self.assertRaises(KeyboardInterrupt):
            first.run()
        self.assertEqual(first.history, [])
        out = self.tmp / "s0"
        before = files(out)
        for again in (first.resume, first.start):
            with self.assertRaises(ValueError) as reused:
                again()
            self.assertIn("build a new GenerationRun", str(reused.exception))
        self.assertEqual(files(out), before)
        self.assertTrue(self.controller("s0", RuleEvaluator()).resume().run()["stop"]["reason"])

    def test_starting_or_resuming_an_owning_controller_again_keeps_its_lease_and_run(self):
        config = replace(self.CONFIG, max_evaluations=2)
        started = self.controller("ow", RuleEvaluator(), config).start()
        resumed = None
        for owner in ("started", "resumed"):
            controller = started if owner == "started" else resumed
            for again in (controller.start, controller.resume):
                with self.assertRaises(ValueError) as refused:
                    again()
                self.assertIn("build a new GenerationRun", str(refused.exception), owner)
            with self.assertRaises(ValueError) as taken:     # still owned: no other controller takes the directory
                self.controller("ow", RuleEvaluator(), config).resume()
            self.assertIn("owned by another", str(taken.exception), owner)
            self.assertTrue(controller.run()["stop"]["reason"], owner)
            resumed = self.controller("ow", RuleEvaluator(), config).resume()
        resumed.close()
        # A controller that loaded a history and released it without running is not reused either.
        with self.assertRaises(ValueError) as reused:
            resumed.resume()
        self.assertIn("build a new GenerationRun", str(reused.exception))

    def test_a_history_naming_an_id_twice_is_refused_before_anything_is_written(self):
        config = replace(self.CONFIG, max_evaluations=3)
        self.run_new("dup", config=config)
        out = self.tmp / "dup"
        rows = [line + "\n" for line in lines(out, "observations.jsonl")]
        rows.append(rows[0])                                # an edited history: the first observation twice
        checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
        digest = ""
        for line in rows:
            digest = gen._sha(digest + gen._sha(line))
        checkpoint.update(history_count=len(rows), history_digest=digest)
        (out / "observations.jsonl").write_text("".join(rows), encoding="utf-8", newline="\n")
        (out / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8", newline="\n")
        with open(out / "proposals.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"step": checkpoint["step"] + 1}))         # a cut log line a resume would move
        before = files(out)
        with self.assertRaises(ValueError) as refused:
            self.controller("dup", RuleEvaluator(), config).resume()
        self.assertIn("names an observation id twice", str(refused.exception))
        self.assertEqual(files(out), before)

    def test_a_pending_evaluation_named_like_a_known_observation_is_refused_unchanged(self):
        config = replace(self.CONFIG, max_evaluations=4)
        with self.assertRaises(KeyboardInterrupt):
            self.controller("pk", RuleEvaluator(interrupt_at=3), config).start().run()
        out = self.tmp / "pk"
        checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(checkpoint["pending"])
        checkpoint["pending"]["id"] = "o0001"               # an edited checkpoint
        (out / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8", newline="\n")
        with open(out / "observations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write('{"type": "observation", "id": "o00')                  # a write the interruption cut
        (out / "error.json").write_text("{}\n", encoding="utf-8", newline="\n")
        before = files(out)
        with self.assertRaises(ValueError) as refused:
            self.controller("pk", RuleEvaluator(), config).resume()
        self.assertIn("already in the history", str(refused.exception))
        self.assertEqual(files(out), before)

    def test_a_failed_release_leaves_a_spent_controller_that_writes_nothing(self):
        config = replace(self.CONFIG, max_evaluations=2)
        controller = self.controller("cf", RuleEvaluator(), config).start()
        real = controller._lease

        class FailingRelease:
            def close(self_):
                real.close()
                raise OSError("synthetic unlock failure")
        controller._lease = FailingRelease()
        with self.assertRaises(OSError):
            controller.run()
        out = self.tmp / "cf"
        self.assertTrue((out / "result.json").exists())
        self.assertIsNone(controller._lease)                # let go of before the release was tried
        before = files(out)
        for again in (controller.run, controller.result):
            with self.assertRaises(ValueError) as refused:
                again()
            self.assertIn("does not own", str(refused.exception))
        self.assertEqual(files(out), before)
        self.controller("cf", RuleEvaluator(), config).resume().close()          # the lease is free again

    def test_a_failed_error_set_aside_changes_no_other_file(self):
        with self.assertRaises(KeyboardInterrupt):
            self.controller("ea", RuleEvaluator(interrupt_at=3)).start().run()
        out = self.tmp / "ea"
        with open(out / "observations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write('{"type": "observation", "id": "cut')                  # a resume would set it aside
        (out / "error.json").write_text("{}\n", encoding="utf-8", newline="\n")
        before, original = files(out), os.replace

        def held(src, dst):
            if Path(src).name == "error.json":
                raise PermissionError("synthetic: held open by another process")
            return original(src, dst)
        with mock.patch.object(gen.os, "replace", held):
            with self.assertRaises(PermissionError):
                self.controller("ea", RuleEvaluator()).resume()
        self.assertEqual(files(out), before)
        resumed = self.controller("ea", RuleEvaluator()).resume()             # once the file is free
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.state["resumes"][0]["error_set_aside"], "error.before_resume.1.json")
        self.assertTrue((out / "observations.partial.1.txt").exists())

    def test_a_summary_is_a_copy_throughout(self):
        controller = self.controller("sc", RuleEvaluator(drift_column=30.0), replace(self.CONFIG, target_accepted_designs=1))
        controller.start().run()
        out = self.tmp / "sc"
        written = files(out)
        snapshot = json.dumps(controller.summary(), sort_keys=True, default=str)
        changed = controller.summary()
        entry = changed["views"]["accepted_registry"]["entries"][0]
        self.assertTrue(entry["origins"] and isinstance(entry["artifacts"], dict))
        entry["genome"][0] = -1
        entry["artifacts"]["changed"] = True
        entry["origins"][0]["changed"] = True
        changed["counts"]["evaluations"] = -1
        changed["counts"]["outcomes"]["changed"] = 1
        changed["counts"]["solves"]["frame"] = -1
        changed["stop"]["reason"] = "changed"
        changed["population"].append("changed")
        changed["population_status"]["blocked_by"]["changed"] = 1
        changed["metrics"]["first_acceptance"]["changed"] = True
        self.assertEqual(json.dumps(controller.summary(), sort_keys=True, default=str), snapshot)
        self.assertNotIn("changed", controller.by_id[entry["id"]]["artifacts"])
        self.assertNotEqual(controller.by_id[entry["id"]]["genome"][0], -1)
        self.assertEqual(files(out), written)

    def test_a_resume_refused_before_its_first_checkpoint_sets_no_error_record_aside(self):
        out = self.tmp / "nc"
        out.mkdir()
        (out / "request.json").write_text(json.dumps({"sha256": "A"}), encoding="utf-8")
        (out / "observations.jsonl").write_text("", encoding="utf-8")
        (out / "error.json").write_text("{}\n", encoding="utf-8")
        before = files(out)
        with self.assertRaises(ValueError) as refused:
            self.controller("nc", RuleEvaluator()).resume(request_sha256="A")
        self.assertIn("exists without a checkpoint", str(refused.exception))
        after = files(out)
        self.assertEqual(after.pop(gen.LEASE), b"0")        # the lease marker, which holds no run data
        self.assertEqual(after, before)
        # Without its request's sha256 a directory with no checkpoint holds no run to resume.
        with self.assertRaises(ValueError) as refused:
            self.controller("nc", RuleEvaluator()).resume()
        self.assertIn("holds no checkpoint", str(refused.exception))
        self.assertEqual({k: v for k, v in files(out).items() if k != gen.LEASE}, before)

    def test_an_error_record_set_aside_by_a_failed_resume_is_named_by_the_next(self):
        def failing(self_):
            raise OSError("synthetic: the disk is full")
        # Start on resume: the first checkpoint write fails after error.json was set aside.
        out = self.tmp / "orph"
        out.mkdir()
        (out / "request.json").write_text(json.dumps({"sha256": "A"}), encoding="utf-8")
        (out / "error.json").write_text("{}\n", encoding="utf-8")
        failed = self.controller("orph", RuleEvaluator())
        with mock.patch.object(gen.GenerationRun, "_checkpoint", failing):
            with self.assertRaises(OSError):
                failed.resume(request_sha256="A")
        self.assertEqual(sorted(files(out)), sorted([gen.LEASE, "request.json", "error.before_resume.1.json"]))
        with self.assertRaises(ValueError) as refused:
            failed.summary()                                # no partial run is kept in memory
        self.assertIn("holds no run", str(refused.exception))
        started = self.controller("orph", RuleEvaluator()).resume(request_sha256="A")
        started.close()
        self.assertEqual(started.state["resumes"], [{"history_count": 0, "started_by_resume": True,
                                                     "error_set_aside": None,
                                                     "unrecorded_error_records": ["error.before_resume.1.json"]}])
        # A resume of a checkpointed run: its final checkpoint fails after error.json was set aside.
        config = replace(self.CONFIG, max_evaluations=4)
        with self.assertRaises(KeyboardInterrupt):
            self.controller("orph2", RuleEvaluator(interrupt_at=3), config).start().run()
        out = self.tmp / "orph2"
        (out / "error.json").write_text("{}\n", encoding="utf-8")
        with mock.patch.object(gen.GenerationRun, "_checkpoint", failing):
            with self.assertRaises(OSError):
                self.controller("orph2", RuleEvaluator(), config).resume()
        self.assertTrue((out / "error.before_resume.1.json").exists())
        resumed = self.controller("orph2", RuleEvaluator(), config).resume()
        entry = resumed.state["resumes"][-1]
        self.assertEqual((entry["sidecar_number"], entry["error_set_aside"], entry["unrecorded_error_records"]),
                         (2, None, ["error.before_resume.1.json"]))
        self.assertTrue(resumed.run()["stop"]["reason"])
        again = self.controller("orph2", RuleEvaluator(), config).resume()    # named once: not again
        again.close()
        self.assertEqual(again.state["resumes"][-1]["unrecorded_error_records"], [])

    def test_an_interrupt_before_the_release_leaves_a_spent_controller_that_writes_nothing(self):
        controller = self.controller("ci", RuleEvaluator(), replace(self.CONFIG, max_evaluations=2)).start()
        self.addCleanup(gen.GenerationRun.close, controller)

        def interrupted(self_):
            raise KeyboardInterrupt                         # before close() lets go of the lease
        with mock.patch.object(gen.GenerationRun, "close", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                controller.run()
        self.assertIsNotNone(controller._lease)             # still held, but the controller is spent
        out = self.tmp / "ci"
        before = files(out, skip=(gen.LEASE,))              # the held marker is locked against reading on Windows
        for again in (controller.run, controller.result):
            with self.assertRaises(ValueError) as refused:
                again()
            self.assertIn("does not own", str(refused.exception))
        self.assertEqual(files(out, skip=(gen.LEASE,)), before)
        controller.close()

    def test_a_design_that_dominates_a_member_on_its_real_audit_resets_the_stagnation_counter(self):
        """A graded synthetic landscape under the pilot limits: children with a larger corner column fail the drift
        row by less, so entries dominate members on the selection's own audits and the bound does not end the run."""
        pilot = gen.GenerationConfig(random_seed=2026103101, population_size=4, offspring_per_generation=4,
                                     max_generations=3, max_evaluations=16, initialization_evaluations=3,
                                     guided_probability=0.75, max_exploration_evaluations=4, max_stagnant_generations=2)
        result = self.run_new("gd", GradedDrift(drift_column=0.0, unknown_scwb=True), pilot)
        logged = [json.loads(line) for line in lines(self.tmp / "gd", "generations.jsonl")]
        self.assertTrue(any(g["improving_entrants"] for g in logged if not g["initial"]), logged)
        self.assertEqual(result["stop"]["reason"], "evaluation_budget_exhausted", result["stop"])

    def test_constructive_initializer_draws_repair_fixed_profiles(self):
        config = replace(self.CONFIG, initialization_policy="constructive_profiles_v1", initialization_evaluations=3,
                         max_evaluations=4)
        result = self.run_new("k", config=config)
        self.assertEqual(result["counts"]["initializer_repair_changed"], 0)
        self.assertEqual(result["counts"]["evaluations_by_phase"]["initialization"], 4)
        blocks = gen.constructive_blocks(gen.ga.DesignSpace(self.seed), self.seed)
        sampler = blocks["column"][0]
        from fractions import Fraction
        paths = set()
        import random as _random
        rng = _random.Random(2)
        for _ in range(200):
            paths.add(tuple(map(tuple, sampler.sample(rng))))
        self.assertTrue(all(sampler.probability([tuple(p) for p in path]) == Fraction(1, sampler.total) for path in paths))

    def test_configuration_refuses_crossover_and_adaptive_weights(self):
        for field in ("crossover", "adaptive_operator_weights"):
            with self.assertRaises(ValueError):
                replace(self.CONFIG, **{field: True}).validate()
        with self.assertRaises(ValueError):
            replace(self.CONFIG, initialization_policy="latin_hypercube").validate()


class Runner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "tools" / "genetic_design_search" / "run.py"
        spec = importlib.util.spec_from_file_location("genetic_run_under_test", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def args(self, *extra):
        return self.runner.parser().parse_args(["--case-file", "case.json", "--seed-column", "30", "30", "6",
                                             "--seed-beam", "16", "30", "5", *extra])

    def test_existing_engine_remains_the_default_and_engines_refuse_each_others_options(self):
        args = self.args()
        self.runner.engine_options(args)
        self.assertEqual((args.engine, args.population_size, args.crossover_probability), ("heuristic_v2", 8, 0.8))
        with self.assertRaises(ValueError):
            self.runner.engine_options(self.args("--engine", "generation_v3", "--crossover-probability", "0.5"))
        with self.assertRaises(ValueError):
            self.runner.engine_options(self.args("--proposal-budget", "10"))
        with self.assertRaises(ValueError):
            self.runner.engine_options(self.args("--resume"))

    def test_v3_option_names_leave_heuristic_v2_abbreviations_unchanged(self):
        for option, field in (("--max-proposals", "max_proposals_per_evaluation"), ("--max-evaluation", "max_evaluations")):
            args = self.args(option, "10")
            self.runner.engine_options(args)
            self.assertEqual(getattr(args, field), 10)
        args = self.args("--engine", "generation_v3", "--proposal-budget", "50", "--evaluation-error-limit", "4")
        self.runner.engine_options(args)
        config = self.runner.generation_config(args)
        self.assertEqual((config.max_proposals, config.max_evaluation_errors), (50, 4))

    def test_option_guard_tolerates_an_older_namespace(self):
        import argparse
        older = argparse.Namespace(**{k: v for k, v in vars(self.args()).items() if k not in ("engine", "resume")})
        older.engine = "heuristic_v2"
        self.runner.engine_options(older)
        self.assertEqual(older.population_size, 8)

    def test_generation_config_from_arguments(self):
        args = self.args("--engine", "generation_v3", "--max-wall-seconds", "900", "--target-accepted", "2")
        self.runner.engine_options(args)
        config = self.runner.generation_config(args)
        self.assertEqual((config.population_size, config.offspring_per_generation, config.max_generations),
                         (4, 4, 6))
        self.assertEqual((config.max_wall_seconds, config.target_accepted_designs, config.crossover), (900.0, 2, False))
        zero = self.args("--engine", "generation_v3", "--offspring-per-generation", "0")
        self.runner.engine_options(zero)
        with self.assertRaises(ValueError):
            self.runner.generation_config(zero)


class RunnerControllerBoundary(unittest.TestCase):
    """run.py's generation_v3 driver around the controller (the evaluator is the synthetic one)."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "tools" / "genetic_design_search" / "run.py"
        spec = importlib.util.spec_from_file_location("genetic_run_boundary", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def setUp(self):
        self.fixture = gf.frame()
        self.fixture.__enter__()
        self.seed = gfx.seed_state()
        self.tmp = Path(tempfile.mkdtemp(prefix="ga_v3_run_"))
        self.request = {"case": {"case_id": "fixture_case"}, "sha256": "fixture-request",
                        "basis": {"fixture": "synthetic evaluator", "seed": self.seed.identity()}}

    def tearDown(self):
        self.fixture.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_v3(self, out, evaluator, resume=False, config=None):
        import argparse
        original = gen.GenerationRun

        def build(*args, **kwargs):
            return original(*args, evaluator=evaluator, verbose=False, **kwargs)
        args = argparse.Namespace(resume=resume, output_dir=out)
        with mock.patch.object(gen, "GenerationRun", build), mock.patch("builtins.print"):
            return self.runner.run_v3(args, self.seed, None, config or Controller.CONFIG, self.request)

    def test_a_refused_configuration_leaves_no_directory(self):
        out = self.tmp / "refused"
        with mock.patch.object(gen, "GenerationRun", side_effect=ValueError("refused configuration")):
            with self.assertRaises(ValueError):
                self.runner.run_v3(__import__("argparse").Namespace(resume=False, output_dir=out), self.seed, None,
                                   Controller.CONFIG, self.request)
        self.assertFalse(out.exists())

    def test_existing_directory_points_to_resume_and_an_unstarted_run_starts_on_resume(self):
        out = self.tmp / "run"
        out.mkdir()
        (out / "request.json").write_text(self.runner.dump(self.request), encoding="utf-8")
        (out / "error.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ValueError) as refused:
            self.run_v3(out, RuleEvaluator())
        self.assertIn("--resume", str(refused.exception))
        self.assertEqual(self.run_v3(out, RuleEvaluator(), resume=True), 0)
        self.assertTrue((out / "checkpoint.json").exists())
        self.assertFalse((out / "error.json").exists())
        self.assertTrue((out / "error.before_resume.1.json").exists())
        # The run this resume started records it, and the error record it set aside, from its first checkpoint on.
        resumes = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))["resumes"]
        self.assertEqual(resumes[0], {"history_count": 0, "started_by_resume": True,
                                      "error_set_aside": "error.before_resume.1.json", "unrecorded_error_records": []})
        # A directory that holds another request's run is not offered --resume.
        other = self.tmp / "other"
        other.mkdir()
        (other / "request.json").write_text(self.runner.dump({**self.request, "sha256": "another request"}),
                                            encoding="utf-8")
        with self.assertRaises(ValueError) as refused:
            self.run_v3(other, RuleEvaluator())
        self.assertIn("another request", str(refused.exception))
        self.assertNotIn("--resume", str(refused.exception))

    def test_the_evaluation_error_bound_has_its_own_exit_code(self):
        seed_identity = self.seed.identity()
        evaluator = RuleEvaluator(fail_on=lambda state: state.identity() != seed_identity)
        self.assertEqual(self.run_v3(self.tmp / "errors", evaluator), self.runner.EXIT_ERROR_BOUND)

    def test_a_path_the_controller_refuses_leaves_no_directory(self):
        out = self.tmp / "deep" / "run"
        limit = len(str(out.resolve())) + gen.LONGEST_ARTIFACT - 1
        with mock.patch.object(gen, "_windows_path_limit", return_value=limit):
            for resume in (False, True):
                with self.assertRaises(ValueError) as refused:
                    self.run_v3(out, RuleEvaluator(), resume=resume)
                self.assertIn("too long", str(refused.exception))
        self.assertFalse((self.tmp / "deep").exists())

    def test_an_empty_directory_starts_and_a_foreign_one_is_refused_consistently(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        self.assertEqual(self.run_v3(empty, RuleEvaluator()), 0)
        self.assertTrue((empty / "checkpoint.json").exists())
        self.assertFalse((empty / "request.json.tmp").exists())
        self.assertEqual(json.loads((empty / "request.json").read_text(encoding="utf-8"))["sha256"], self.request["sha256"])
        foreign = self.tmp / "foreign"
        foreign.mkdir()
        (foreign / "notes.txt").write_text("not a run\n", encoding="utf-8")
        messages = []
        for resume in (False, True):
            with self.assertRaises(ValueError) as refused:
                self.run_v3(foreign, RuleEvaluator(), resume=resume)
            messages.append(str(refused.exception))
        # Neither answer sends the user to the other mode: both lead to a new or emptied directory.
        self.assertTrue(all("new" in m and "empty" in m for m in messages), messages)
        self.assertEqual(sorted(p.name for p in foreign.iterdir()), ["notes.txt"])

    def test_the_request_hashes_every_loaded_repository_source(self):
        rows = self.runner.loaded_sources()
        rc = Path(__file__).resolve().parents[1]
        self.assertIn("Design/Genetic_Generation.py", rows)
        self.assertFalse([name for name in rows if name.startswith("tests/")])
        from Design.Verification_Integrity import file_sha256
        self.assertEqual(rows["Design/Genetic_Generation.py"], file_sha256(rc / "Design" / "Genetic_Generation.py"))
        self.assertEqual(list(rows), sorted(rows))          # the order feeds the request hash (dump keeps it)
        self.assertIn("Structure_Parameters.py", rows)      # beyond Design/: the root and the model modules
        self.assertIn("Model/Member_Groups.py", rows)

    def test_only_the_owner_of_a_directory_saves_its_request(self):
        from Design.Verification_Integrity import exclusive_lease
        owned = self.tmp / "owned"
        with exclusive_lease(owned / gen.LEASE):            # another process holds the new directory
            with self.assertRaises(ValueError) as refused:
                self.run_v3(owned, RuleEvaluator())
            self.assertIn("owned by another", str(refused.exception))
            self.assertIn("the command that started it", str(refused.exception))
            self.assertEqual(sorted(p.name for p in owned.iterdir()), [gen.LEASE])
        # The check under the lease decides: another launch that claimed the directory after this runner's own check
        # keeps its request, and this launch is refused.
        raced, real, controller_class = self.tmp / "raced", gen.new_run_refusal, gen.GenerationRun
        other = self.runner.dump({**self.request, "sha256": "the other launch's request"})
        state = {"claimed": False}

        def race(path):
            kind = real(path)
            if not state["claimed"]:
                state["claimed"] = True
                winner = controller_class(self.seed, None, Controller.CONFIG, raced, case_id="fixture_case",
                                          basis=self.request["basis"], evaluator=RuleEvaluator(), verbose=False)
                winner.start(request_text=other)
                winner.close()
            return kind
        with mock.patch.object(gen, "new_run_refusal", race):
            with self.assertRaises(ValueError) as refused:
                self.run_v3(raced, RuleEvaluator())
        # The loser's request is not the saved one: it is sent to a new directory, not told to resume.
        self.assertIn("another request", str(refused.exception))
        self.assertNotIn("--resume", str(refused.exception))
        self.assertEqual((raced / "request.json").read_text(encoding="utf-8"), other)

    def test_a_marker_or_a_cut_request_alone_is_a_new_directory(self):
        for name in (gen.LEASE, "request.json.tmp"):
            out = self.tmp / ("only" + name.replace(".", "_"))
            out.mkdir()
            (out / name).write_text("0", encoding="utf-8")
            self.assertEqual(self.run_v3(out, RuleEvaluator()), 0, name)
            self.assertFalse((out / "request.json.tmp").exists(), name)
        # A request write cut at its replacement leaves no request, and the same command then starts the run.
        out, original = self.tmp / "cut_request", os.replace

        def cut(src, dst):
            if Path(dst).name == "request.json":
                raise OSError("synthetic: interrupted while replacing")
            return original(src, dst)
        with mock.patch.object(gen.os, "replace", cut):
            with self.assertRaises(OSError):
                self.run_v3(out, RuleEvaluator())
        self.assertEqual(sorted(p.name for p in out.iterdir()), [gen.LEASE])
        self.assertEqual(self.run_v3(out, RuleEvaluator()), 0)
        # A saved request that is not a whole JSON object refuses a resume, changing nothing.
        for name, text in (("broken", "{"), ("not_object", "[1, 2]")):
            broken = self.tmp / name
            broken.mkdir()
            (broken / "request.json").write_text(text, encoding="utf-8")
            for resume in (True, False):                    # resumed or started: the same advice
                with self.assertRaises(ValueError) as refused:
                    self.run_v3(broken, RuleEvaluator(), resume=resume)
                self.assertIn("not a complete request", str(refused.exception), (name, resume))
            self.assertEqual(sorted(p.name for p in broken.iterdir()), ["request.json"], name)

    def test_an_exception_without_a_printable_message_ends_the_runner_with_exit_code_1(self):
        out = self.tmp / "unprintable"
        with mock.patch.object(gen.GenerationRun, "_absorb", side_effect=Unprintable()):
            self.assertEqual(self.run_v3(out, RuleEvaluator()), 1)
        self.assertIn("Unprintable", json.loads((out / "error.json").read_text(encoding="utf-8"))["error"])


if __name__ == "__main__":
    unittest.main()
