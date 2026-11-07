"""Severity plus demand-sized preference of the genetic search (selection policy severity_preferred_jumps_v1,
2026-11-01).

  P1  a severe failure with useful screened proposals gets the configured preferred allocation of the guided draws:
      the branch frequency is the declared share, and the probabilities normalize;
  P2  on the numbers of o0002 of the 10-minute jumps run, the strongly improving 22 x 36 in beam is far more likely than
      the small 5 -> 6 ksi step, and than the 8 ksi partial; an improvement is always the diagnostic's own unit's
      estimate against the same unit's baseline (a family grade option is governed by another band);
  P3  a marginal failure does not trigger the preference; drift and column checks get no beam estimate;
  P4  a screened proposal keeps its preference when a local step also proposes it; repeated evidence, both groups of a
      tie and two operators add no priority;
  P5  no useful jump, invalid or stale evidence, a negligible or no improvement and an exhausted pool fall back
      explicitly to the normal branch; an estimate that needs a cage that does not thread the frozen column cages is no
      beam-only reach of the target; a threading-constrained failure keeps its column alternatives and says the
      beam-only target is unmet;
  P6  seeded draws repeat and every logged probability is the exact distribution's;
  P8  a unit's own smallest target option: the screen gives it per unit (a family's shortlist follows its worst unit;
      confirmed only when that unit was priced at every smaller option), the route annotates its existing proposal of
      it without adding one (every policy draws over the same genomes), also when the gene has no shortlist or the
      shortlist is another lever's move, with the option's own gene-level fields, and the pool prefers it over the
      family's larger option; the drawn annotations and their routes are counted apart; the entry logs the unit's
      estimate pool beside the gene's; an excluded proposal is one row per key and reason, a partly eligible genome
      keeps the rows of its other keys, and eligible entries and already-drawn rows carry every diagnostic and lever;
      the note names column proposals only where the failure has them;
  P7  the controller: the preferred branch's children are evaluated and assessed like any other and never accepted by
      estimate, a pool candidate drawn for a parent (either branch) is not drawn from the preferred branch again, the
      logged distribution is complete and normalized, the counters keep proposed, drawn, completed, errored and
      accepted apart, the policy and its parameters bind the basis (a change refuses the resume), and the configuration
      and runner declare it.

Synthetic diagnostics and screen summaries on the fixture frame, no analysis: nothing here is a designed or accepted
structure.
"""
import copy
import importlib.util
import json
import random
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_fixture as cf                                                    # noqa: E402
import genetic_fixture as gfx                                                    # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402
import test_genetic_jumps as jt                                                  # noqa: E402

from Design import Genetic_Diagnostics as dg                                     # noqa: E402
from Design import Genetic_Generation as gen                                     # noqa: E402
from Design import Genetic_Jumps as jumps                                        # noqa: E402
from Design import Genetic_Operators as ops                                      # noqa: E402
from Design import Genetic_Search as ga                                          # noqa: E402
from Design.Config import DesignConfig, with_beam_practical_ratio                # noqa: E402

X_EDGE, Y_EDGE, COLUMN = jt.X_EDGE, jt.Y_EDGE, jt.COLUMN
S01 = [X_EDGE, Y_EDGE]
S04 = ["s04_04__beam_x__edge", "s04_04__beam_y__edge"]
TIE01, TIE04 = "s01_03__sfrs_beam", "s04_04__sfrs_beam"
SELECTION = {"policy": ops.PREFERRED, "severity_floor": 0.2, "severity_cap": 1.0, "categorical_weight": 0.5,
             "severe_utilization": 1.25, "preferred_share": 0.80, "target_dcr": 0.85, "min_gain": 0.02}
WEIGHTED = {**SELECTION, "policy": ops.SEVERITY_WEIGHTED}


def size_screen(gene, old, unit, b, to, e, outcome=None, pool="threading"):
    """A size gene's screen summary as Genetic_Jumps.summary writes it (one unit; ``pool`` the estimate's cage pool)."""
    meets = e <= 0.85
    chosen = {"to": list(to), "estimated_dcr": e, "pool": pool,
              "units": [{"groups": list(unit), "estimated_dcr": e, "pool": pool}]}
    return {"gene": gene, "kind": "size", "from": list(old), "levers": ["depth", "width"], "status": "screened",
            "outcome": outcome or ("target_met_estimated" if meets else "target_unmet_estimated"),
            "calibration": "agrees", "baseline_dcr": b, "recorded_utilization": b, "capacity_ratio_required": b / 0.85,
            "target_dcr": 0.85, "meets_target": chosen if meets and not outcome else None,
            "partial": None if meets or outcome else chosen, "smallest_confirmed": True if meets else None,
            "unit_baselines": [{"groups": list(unit), "baseline_dcr": b}]}


def grade_screen(old, to, per_unit, outcome=None):
    """The beam family's grade screen summary: per_unit [(unit groups, baseline, estimate)]."""
    meets = all(e <= 0.85 for _u, _b, e in per_unit)
    chosen = {"to": to, "estimated_dcr": max(e for _u, _b, e in per_unit), "pool": "threading",
              "units": [{"groups": list(u), "estimated_dcr": e, "pool": "threading"} for u, _b, e in per_unit]}
    return {"gene": "beam_grade", "kind": "grade", "from": old, "levers": ["grade_up"], "status": "screened",
            "outcome": outcome or ("target_met_estimated" if meets else "target_unmet_estimated"), "calibration": "agrees",
            "baseline_dcr": max(b for _u, b, _e in per_unit), "recorded_utilization": max(b for _u, b, _e in per_unit),
            "capacity_ratio_required": max(b for _u, b, _e in per_unit) / 0.85, "target_dcr": 0.85,
            "meets_target": chosen if meets and not outcome else None, "partial": None if meets or outcome else chosen,
            "smallest_confirmed": None, "unit_baselines": [{"groups": list(u), "baseline_dcr": b} for u, b, _e in per_unit]}


def flexural(ctx, gid, u, genes, reason="beam_flexure", component="flexure_negative", **fields):
    """A failed beam flexural component with its demand-sized screen (capacity_jump), bound to ``ctx``."""
    item = {"id": f"strength|member.strength|{gid}#{component}", "reason": reason, "status": "fail", "applicable": True,
            "evidence_valid": True, "evidence_problems": [], "parent_identity": ctx["parent_identity"],
            "basis_identity": ctx["basis_identity"], "dependencies": {k: True for k in dg.RULES[reason][1]},
            "demand": 1000.0 * u, "capacity": 1000.0, "comparison": "<=", "units": "kip-in", "synthetic": False,
            "group_id": gid, "source": {"component": component}}
    item["capacity_jump"] = {"version": jumps.VERSION, "parent_identity": ctx["parent_identity"],
                             "basis_identity": ctx["basis_identity"], "status": "screened", "utilization": u,
                             "required_capacity_kip_in": 1000.0 * u / 0.85, "capacity_ratio_required": u / 0.85,
                             "target_dcr": 0.85, "genes": genes}
    item.update(fields)
    return item


def o0002_like(ctx, u=1.749, size_to=(22.0, 36.0), size_e=1.203):
    """The beam shortfall of o0002 of the 10-minute jumps run on the fixture's lower tie: a 22 x 36 in beam estimated
    at 1.203 and the 8 ksi family partial at 1.638 (its upper band at 1.413 -> 1.329)."""
    return flexural(ctx, X_EDGE, u, {
        TIE01: size_screen(TIE01, (16.0, 30.0), S01, u, size_to, size_e),
        "beam_grade": grade_screen(5.0, 8.0, [(S01, u, 1.638), (S04, 1.413, 1.329)])})


class Fixture(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()
        self.seed = gfx.seed_state()
        self.ctx = {**jt.selection_ctx(), "guided_policy": {"beam_jump": jumps.DEMAND_SIZED}}

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def drift(self, u=2.1, story=2):
        return jt.diag(self.ctx, "story_drift", 0.02 * u, 0.02, units="in/in", story=story, axis="x")

    def shear(self, u=1.05):
        return jt.diag(self.ctx, "column_capacity_shear", 100.0 * u, 100.0, group_id=COLUMN)

    def genome_of(self, gene, to):
        index = next(i for i, g in enumerate(self.ctx["genes"]) if g["name"] == gene)
        options = [tuple(o) if isinstance(o, (list, tuple)) else o for o in self.ctx["genes"][index]["options"]]
        genome = list(self.ctx["genome"])
        genome[index] = options.index(tuple(to) if isinstance(to, (list, tuple)) else to)
        return tuple(genome)


class Allocation(Fixture):
    def test_a_severe_failure_with_useful_screened_proposals_gets_the_preferred_allocation(self):
        alternatives = ops.guided_alternatives(self.ctx, [o0002_like(self.ctx), self.drift(), self.shear()])
        pool = ops.preferred_pool(alternatives, SELECTION)
        self.assertTrue(pool["eligible"])
        self.assertIsNone(pool["fallback"])
        distribution = ops.guided_distribution(alternatives, SELECTION)
        self.assertAlmostEqual(sum(v["probability"] for v in distribution.values()), 1.0)
        self.assertAlmostEqual(sum(v["preferred"] for v in distribution.values()), 1.0)
        candidates = {tuple(e["genome"]) for e in pool["eligible"]}
        expected = 0.8 + 0.2 * sum(distribution[g]["normal"] for g in candidates)
        self.assertAlmostEqual(sum(distribution[g]["probability"] for g in candidates), expected)
        rng, drawn = random.Random(20261101), []
        for _ in range(4000):
            proposal = ops.choose_guided(alternatives, rng, SELECTION)
            drawn.append((proposal["selection"]["branch"], tuple(proposal["requested_genome"])))
        self.assertAlmostEqual(sum(b == "preferred" for b, _g in drawn) / len(drawn), 0.8, delta=0.02)
        self.assertAlmostEqual(sum(g in candidates for _b, g in drawn) / len(drawn), expected, delta=0.02)
        # Every eligible route stays available in the normal branch: drift and column shear are still drawn.
        keys = Counter(g for b, g in drawn if b == "normal")
        self.assertTrue(any(g not in candidates for g in keys))

    def test_the_strongly_improving_beam_is_far_more_likely_than_the_small_grade_step(self):
        alternatives = ops.guided_alternatives(self.ctx, [o0002_like(self.ctx), self.drift(), self.shear()])
        distribution = ops.guided_distribution(alternatives, SELECTION)
        before = ops.guided_distribution(alternatives, WEIGHTED)
        beam, small, partial = (self.genome_of(TIE01, (22.0, 36.0)), self.genome_of("beam_grade", 6.0),
                                self.genome_of("beam_grade", 8.0))
        # Under severity_weighted_v1 the three fixes of the failure were equally likely.
        self.assertAlmostEqual(before[beam]["probability"], before[small]["probability"])
        self.assertAlmostEqual(before[beam]["probability"], before[partial]["probability"])
        self.assertGreater(distribution[beam]["probability"], 10 * distribution[small]["probability"])
        self.assertGreater(distribution[beam]["probability"], 3 * distribution[partial]["probability"])
        self.assertEqual(distribution[small]["preferred"], 0.0)          # a local step the screen did not shortlist
        pool = {tuple(e["genome"]): e for e in ops.preferred_pool(alternatives, SELECTION)["eligible"]}
        self.assertAlmostEqual(pool[beam]["deficit_closed"], (1.749 - 1.203) / (1.749 - 0.85))
        self.assertAlmostEqual(pool[partial]["deficit_closed"], (1.749 - 1.638) / (1.749 - 0.85))

    def test_an_improvement_is_the_diagnostics_own_unit_against_the_same_units_baseline(self):
        # An upper-band diagnostic: the family grade option's estimate for that band (1.329 against 1.413), never the
        # family's worst unit (1.638, the lower band).
        upper = flexural(self.ctx, S04[0], 1.413, {
            TIE04: size_screen(TIE04, (16.0, 30.0), S04, 1.413, (26.0, 36.0), 0.927),
            "beam_grade": grade_screen(5.0, 8.0, [(S01, 1.749, 1.638), (S04, 1.413, 1.329)])})
        alternatives = ops.guided_alternatives(self.ctx, [upper])
        pool = {(e["gene"], e["to"] if not isinstance(e["to"], list) else tuple(e["to"])): e
                for e in ops.preferred_pool(alternatives, SELECTION)["eligible"]}
        grade = pool[("beam_grade", 8.0)]
        self.assertEqual((grade["baseline_utilization"], grade["estimated_utilization"]), (1.413, 1.329))
        self.assertAlmostEqual(grade["deficit_closed"], (1.413 - 1.329) / (1.413 - 0.85))
        route = ops.route(self.ctx, upper)
        origin = next(o for p in route["proposals"] for o in p["origins"] if o.get("step") and o["lever"] == "grade_up")
        self.assertEqual(origin["jump"]["matched"], {"group": S04[0], "baseline_dcr": 1.413, "estimated_dcr": 1.329,
                                                     "pool": "threading", "scope": "shortlisted_option",
                                                     "unit_target": None})
        self.assertEqual(origin["jump"]["estimated_dcr"], 1.638)          # the family's gene-level estimate, kept apart

    def test_a_marginal_failure_does_not_trigger_the_preference_and_other_checks_get_no_beam_estimate(self):
        marginal = o0002_like(self.ctx, u=1.15, size_e=0.84)
        alternatives = ops.guided_alternatives(self.ctx, [marginal, self.drift(), self.shear(u=1.6)])
        pool = ops.preferred_pool(alternatives, SELECTION)
        self.assertEqual((pool["eligible"], pool["fallback"]), ([], "no_eligible_severe_candidate"))
        self.assertEqual({e["reason"] for e in pool["excluded"]}, {"not_severe"})
        self.assertTrue(all(e["key"] == ops.failure_key(self.ctx, marginal) for e in pool["excluded"]))
        distribution = ops.guided_distribution(alternatives, SELECTION)
        weighted = ops.guided_distribution(alternatives, WEIGHTED)
        for genome in distribution:
            self.assertAlmostEqual(distribution[genome]["probability"], weighted[genome]["probability"])
        proposal = ops.choose_guided(alternatives, random.Random(1), SELECTION)
        self.assertEqual((proposal["selection"]["branch"], proposal["selection"]["branch_probabilities"]["preferred"],
                          proposal["selection"]["fallback"]), ("normal", 0.0, "no_eligible_severe_candidate"))


class Deduplication(Fixture):
    def test_a_screened_local_step_keeps_its_preference(self):
        # 18 x 36 in is the width frontier's own step from 16 x 30 in; the screen shortlisted it too.
        d = o0002_like(self.ctx, size_to=(18.0, 36.0), size_e=1.10)
        alternatives = ops.guided_alternatives(self.ctx, [d])
        genome = self.genome_of(TIE01, (18.0, 36.0))
        entry = next(e for e in ops.preferred_pool(alternatives, SELECTION)["eligible"] if tuple(e["genome"]) == genome)
        self.assertTrue(entry["local_step"])
        self.assertGreater(ops.guided_distribution(alternatives, SELECTION)[genome]["preferred"], 0.5)

    def test_repeated_evidence_groups_of_a_tie_and_operators_add_no_priority(self):
        once = [o0002_like(self.ctx), self.drift(), self.shear()]
        positive = o0002_like(self.ctx)
        positive.update(id=f"strength|member.strength|{X_EDGE}#flexure_positive", source={"component": "flexure_positive"})
        other_group = flexural(self.ctx, Y_EDGE, 1.556, copy.deepcopy(o0002_like(self.ctx)["capacity_jump"]["genes"]))
        h3 = flexural(self.ctx, X_EDGE, 1.749, copy.deepcopy(o0002_like(self.ctx)["capacity_jump"]["genes"]),
                      reason="beam_threading_constrained_strength", component="flexure_h3",
                      joint_topology_consistent=True, blocking_cores=[], blocking_core_cages=[], blocking_transitions=[],
                      lane_lookahead={}, transition_lookahead=[])
        a = ops.guided_distribution(ops.guided_alternatives(self.ctx, once), SELECTION)
        b = ops.guided_distribution(ops.guided_alternatives(self.ctx, once + [positive, other_group, h3]), SELECTION)
        self.assertEqual(set(a), set(b))
        for genome in a:
            self.assertAlmostEqual(a[genome]["probability"], b[genome]["probability"])
        pool = ops.preferred_pool(ops.guided_alternatives(self.ctx, once + [positive, other_group, h3]), SELECTION)
        genomes = [tuple(e["genome"]) for e in pool["eligible"]]
        self.assertEqual(len(genomes), len(set(genomes)))


class Fallbacks(Fixture):
    def test_no_useful_jump_invalid_or_stale_evidence_and_negligible_gains_fall_back_explicitly(self):
        none = flexural(self.ctx, X_EDGE, 1.749, {
            TIE01: size_screen(TIE01, (16.0, 30.0), S01, 1.749, (22.0, 36.0), 1.2,
                               outcome="target_unmet_no_useful_partial")})
        pool = ops.preferred_pool(ops.guided_alternatives(self.ctx, [none, self.drift()]), SELECTION)
        self.assertEqual((pool["eligible"], pool["fallback"]), ([], "no_screened_candidate"))
        for bad in ({**o0002_like(self.ctx), "evidence_valid": False}, {**o0002_like(self.ctx), "parent_identity": "x"},
                    {**o0002_like(self.ctx), "demand": float("nan")}):
            pool = ops.preferred_pool(ops.guided_alternatives(self.ctx, [bad, self.drift()]), SELECTION)
            self.assertEqual(pool["eligible"], [])
        negligible = flexural(self.ctx, X_EDGE, 1.749, {
            TIE01: size_screen(TIE01, (16.0, 30.0), S01, 1.749, (26.0, 36.0), 1.74)})
        pool = ops.preferred_pool(ops.guided_alternatives(self.ctx, [negligible]), SELECTION)
        self.assertEqual(({e["reason"] for e in pool["excluded"]}, pool["fallback"]),
                         ({"negligible_improvement"}, "no_eligible_severe_candidate"))

    def test_an_estimate_that_needs_a_nonthreading_cage_is_no_beam_only_reach(self):
        # No cage of 22 x 36 in threads the frozen column cages: the screen priced the admissible pool (0.80).
        cores = sorted({t["core_group"] for t in self.ctx["joint_types"].values() if X_EDGE in t["beams"]["x"]})
        blocked = flexural(self.ctx, X_EDGE, 1.749, {
            TIE01: size_screen(TIE01, (16.0, 30.0), S01, 1.749, (22.0, 36.0), 0.80, pool="admissible"),
            "beam_grade": grade_screen(5.0, 8.0, [(S01, 1.749, 1.638), (S04, 1.413, 1.329)])},
            reason="beam_threading_constrained_strength", joint_topology_consistent=True, blocking_cores=cores,
            blocking_core_cages=cores, blocking_transitions=[], lane_lookahead={cores[0]: [30.0, 30.0]},
            transition_lookahead=[])
        alternatives = ops.guided_alternatives(self.ctx, [blocked, self.drift()])
        pool = ops.preferred_pool(alternatives, SELECTION)
        beam = self.genome_of(TIE01, (22.0, 36.0))
        self.assertNotIn(beam, [tuple(e["genome"]) for e in pool["eligible"]])
        self.assertIn(("threading_blocked_estimate", "admissible"),
                      {(e["reason"], e.get("pool")) for e in pool["excluded"] if tuple(e["genome"]) == beam})
        distribution = ops.guided_distribution(alternatives, SELECTION)
        self.assertEqual(distribution[beam]["preferred"], 0.0)
        self.assertGreater(distribution[beam]["normal"], 0.0)            # still offered, in the normal branch
        columns = [g for g, levers in operators_by_lever(alternatives).items() if levers & {"column_up", "column_lanes"}]
        self.assertTrue(columns and all(distribution[g]["probability"] > 0 for g in columns))
        # The grade partial is a threading estimate: it stays, and says the beam-only target is unmet.
        self.assertTrue(all(e["beam_only_target_unmet"] for e in pool["eligible"]))

    def test_no_improvement_never_enters_the_pool(self):
        flat = flexural(self.ctx, X_EDGE, 1.749, {
            "beam_grade": grade_screen(5.0, 8.0, [(S01, 1.749, 1.749), (S04, 1.413, 1.413)])})
        pool = ops.preferred_pool(ops.guided_alternatives(self.ctx, [flat]), {**SELECTION, "min_gain": 0.0})
        self.assertEqual((pool["eligible"], {e["reason"] for e in pool["excluded"]}), ([], {"negligible_improvement"}))

    def test_an_exhausted_pool_falls_back_and_is_never_drawn_from_again(self):
        alternatives = ops.guided_alternatives(self.ctx, [o0002_like(self.ctx), self.drift()])
        pool = ops.preferred_pool(alternatives, SELECTION)
        drawn, rng = [], random.Random(5)
        for _ in range(len(pool["eligible"]) + 20):
            proposal = ops.choose_guided(alternatives, rng, {**SELECTION, "exhausted": drawn})
            if proposal["selection"]["branch"] == "preferred":
                self.assertNotIn(list(proposal["requested_genome"]), drawn)
                drawn.append(list(proposal["requested_genome"]))
        self.assertEqual(sorted(map(tuple, drawn)), sorted(tuple(e["genome"]) for e in pool["eligible"]))
        last = ops.choose_guided(alternatives, rng, {**SELECTION, "exhausted": drawn})
        self.assertEqual((last["selection"]["branch"], last["selection"]["fallback"]), ("normal", "preferred_pool_exhausted"))
        self.assertEqual({e["reason"] for e in last["selection"]["pool"]["excluded"]
                          if tuple(e["genome"]) in map(tuple, drawn)}, {"already_drawn_for_this_parent"})

    def test_a_threading_constrained_failure_keeps_its_column_alternatives(self):
        cores = sorted({t["core_group"] for t in self.ctx["joint_types"].values() if X_EDGE in t["beams"]["x"]})
        constrained = flexural(self.ctx, X_EDGE, 1.749, o0002_like(self.ctx)["capacity_jump"]["genes"],
                               reason="beam_threading_constrained_strength", joint_topology_consistent=True,
                               blocking_cores=cores, blocking_core_cages=cores, blocking_transitions=[],
                               lane_lookahead={cores[0]: [30.0, 30.0]}, transition_lookahead=[])
        alternatives = ops.guided_alternatives(self.ctx, [constrained])
        distribution = ops.guided_distribution(alternatives, SELECTION)
        columns = [g for g, p in operators_by_lever(alternatives).items() if p & {"column_up", "column_lanes"}]
        self.assertTrue(columns)
        self.assertTrue(all(distribution[g]["probability"] > 0 for g in columns))
        pool = ops.preferred_pool(alternatives, SELECTION)
        self.assertTrue(all(e["beam_only_target_unmet"] for e in pool["eligible"]))
        rng = random.Random(3)
        notes = {ops.choose_guided(alternatives, rng, SELECTION)["selection"].get("note") for _ in range(30)}
        self.assertTrue(any(note and "target unmet" in note for note in notes))
        self.assertTrue(any(note and "column clearance and coordination" in note for note in notes))

    def test_the_column_sentence_of_the_note_only_where_the_failure_proposes_column_moves(self):
        alternatives = ops.guided_alternatives(self.ctx, [o0002_like(self.ctx)])     # plain beam flexure: no column lever
        rng = random.Random(4)
        notes = {ops.choose_guided(alternatives, rng, SELECTION)["selection"].get("note") for _ in range(40)} - {None}
        self.assertTrue(notes)
        self.assertTrue(all("target unmet" in note and "column" not in note for note in notes))


class UnitTargets(Fixture):
    def unit_reach(self, unit_targets=True):
        """An upper-band diagnostic whose own unit reaches the target with 6 ksi (the local step) while the family's
        shortlist is the 8 ksi partial of its worst (lower) unit."""
        grade = grade_screen(5.0, 8.0, [(S01, 1.749, 1.638), (S04, 1.413, 0.80)])
        if unit_targets:
            # At 6 ksi the family's worst unit (the lower band) has no threading cage: the option's own pool.
            grade["unit_targets"] = [{"groups": list(S04), "to": 6.0, "estimated_dcr": 0.84, "pool": "threading",
                                      "smallest_confirmed": True, "option_estimated_dcr": 1.60,
                                      "option_pool": "admissible"}]
        return flexural(self.ctx, S04[0], 1.413, {
            TIE04: size_screen(TIE04, (16.0, 30.0), S04, 1.413, (26.0, 36.0), 0.927), "beam_grade": grade})

    def test_the_screen_gives_each_units_smallest_threading_target_option(self):
        result = {"pricings": 0, "priced": [], "unscreened": [], "baseline_dcr": 1.5, "outcome": None,
                  "meets_target": None, "partial": None, "smallest_confirmed": None}
        estimates = {6.0: {"g0": (0.84, "threading"), "g1": (1.30, "threading"), "g2": (0.70, "admissible")},
                     8.0: {"g0": (0.70, "threading"), "g1": (1.20, "threading"), "g2": (0.60, "threading")}}

        def price(option):
            units = [{"groups": [g], "estimated_dcr": dcr, "pool": pool} for g, (dcr, pool) in sorted(estimates[option].items())]
            return {"status": "priced", "estimated_dcr": max(u["estimated_dcr"] for u in units), "units": units,
                    "repair_changes": []}, 3
        jumps._search(result, [(1, 6.0, []), (2, 8.0, [])], {6.0}, price, jumps._Budget(8, lambda: 0.0, None),
                      jt.PARAMS, cost=3)
        # The family follows its worst unit (g1): an 8 ksi partial.
        self.assertEqual((result["outcome"], result["partial"]["to"]), ("target_unmet_estimated", 8.0))
        # g0 reaches the target with 6 ksi already; g2 only with a threading cage at 8 ksi; g1 never.
        self.assertEqual(result["unit_targets"], [
            {"groups": ["g0"], "to": 6.0, "estimated_dcr": 0.84, "pool": "threading", "smallest_confirmed": True,
             "option_estimated_dcr": 1.30, "option_pool": "threading"},
            {"groups": ["g2"], "to": 8.0, "estimated_dcr": 0.60, "pool": "threading", "smallest_confirmed": True,
             "option_estimated_dcr": 1.20, "option_pool": "threading"}])

    def test_a_unit_the_deadline_cut_from_a_smaller_row_is_not_confirmed(self):
        # 7 ksi: unit gA has no admissible cage and the deadline passes before gB is priced (the row keeps gA only).
        result = {"pricings": 0, "priced": [], "unscreened": [], "baseline_dcr": 1.5, "outcome": None,
                  "meets_target": None, "partial": None, "smallest_confirmed": None}
        rows = {6.0: [("gA", "priced", 1.5, "threading"), ("gB", "priced", 1.3, "threading")],
                7.0: [("gA", "no_admissible_cage", None, None)],
                8.0: [("gA", "priced", 1.2, "threading"), ("gB", "priced", 0.70, "threading")]}

        def price(option):
            units = [{"groups": [g], "status": s, "estimated_dcr": e, "pool": p} for g, s, e, p in rows[option]]
            complete = len(units) == 2 and all(u["status"] == "priced" for u in units)
            status = "priced" if complete else next(u["status"] for u in units if u["status"] != "priced")
            return {"status": status, "estimated_dcr": max(u["estimated_dcr"] for u in units) if complete else None,
                    "units": units, "repair_changes": []}, len(units)
        jumps._search(result, [(1, 6.0, []), (2, 7.0, []), (3, 8.0, [])], {6.0}, price,
                      jumps._Budget(12, lambda: 0.0, None), jt.PARAMS, cost=2)
        self.assertEqual([(t["groups"], t["to"], t["smallest_confirmed"]) for t in result["unit_targets"]],
                         [(["gB"], 8.0, False)])
        self.assertEqual(jumps.summary(result)["unit_targets"], result["unit_targets"])

    def test_a_units_smaller_target_option_carries_the_preference_over_the_familys_larger_one(self):
        alternatives = ops.guided_alternatives(self.ctx, [self.unit_reach(), self.drift()])
        six, eight = self.genome_of("beam_grade", 6.0), self.genome_of("beam_grade", 8.0)
        origins = ops._origins_by_genome(alternatives)
        annotated = [o for o in origins[six][1] if o.get("step") == "demand_sized"]
        self.assertEqual([o["jump"]["matched"]["scope"] for o in annotated], ["unit_target"])
        self.assertTrue(any(o.get("step") != "demand_sized" for o in origins[six][1]))   # the local step it already was
        pool = ops.preferred_pool(alternatives, SELECTION)
        eligible = {tuple(e["genome"]): e for e in pool["eligible"]}
        self.assertEqual((eligible[six]["deficit_closed"], eligible[six]["meets_target_estimated"],
                          eligible[six]["unit_smallest_confirmed"], eligible[six]["estimated_utilization"],
                          eligible[six]["scope"]), (1.0, True, True, 0.84, "unit_target"))
        # The gene-level fields are the option's own (its worst unit's), the unit's under pool and the estimate.
        self.assertEqual((eligible[six]["pool"], eligible[six]["gene_pool"], eligible[six]["gene_outcome"]),
                         ("threading", "admissible", "target_unmet_estimated"))
        annotation = annotated[0]["jump"]
        self.assertEqual((annotation["estimated_dcr"], annotation["pool"], annotation["meets_target_estimated"],
                          annotation["smallest_confirmed"]), (1.60, "admissible", False, None))
        self.assertEqual(alternatives["jumps"]["unit_target"], 1)
        self.assertNotIn(eight, eligible)
        self.assertIn("larger_than_unit_target",
                      {e["reason"] for e in pool["excluded"] if tuple(e["genome"]) == eight})
        # No proposal was added: the genomes, keys and severity-weighted draw are those without the unit target.
        plain = ops.guided_alternatives(self.ctx, [self.unit_reach(unit_targets=False), self.drift()])
        self.assertEqual(set(ops._origins_by_genome(alternatives)), set(ops._origins_by_genome(plain)))
        self.assertEqual({k: f["genomes"] for k, f in alternatives["failures"].items()},
                         {k: f["genomes"] for k, f in plain["failures"].items()})
        self.assertEqual(ops.guided_distribution(alternatives, WEIGHTED), ops.guided_distribution(plain, WEIGHTED))
        # Without it, the family option reaches the unit's target but is not its confirmed smallest.
        before = {tuple(e["genome"]): e for e in ops.preferred_pool(plain, SELECTION)["eligible"]}
        self.assertEqual((before[eight]["deficit_closed"], before[eight]["unit_smallest_confirmed"],
                          before[eight]["gene_outcome"]), (1.0, False, "target_unmet_estimated"))

    def test_a_units_target_is_annotated_when_the_gene_has_no_shortlist(self):
        # The worst unit has no useful partial, so the family has no shortlisted option at all.
        grade = grade_screen(5.0, 8.0, [(S01, 1.749, 1.735), (S04, 1.413, 0.80)], outcome="target_unmet_no_useful_partial")
        grade["unit_targets"] = [{"groups": list(S04), "to": 6.0, "estimated_dcr": 0.84, "pool": "threading",
                                  "smallest_confirmed": True, "option_estimated_dcr": 1.74, "option_pool": "threading"}]
        d = flexural(self.ctx, S04[0], 1.413, {"beam_grade": grade})
        route = ops.route(self.ctx, d)
        grade_rows = [(j["status"], j.get("reason")) for j in route["jumps"] if j["gene"] == "beam_grade"]
        self.assertEqual(grade_rows, [("unit_target", None), ("fallback", "target_unmet_no_useful_partial")])
        alternatives = ops.guided_alternatives(self.ctx, [d, self.drift()])
        pool = ops.preferred_pool(alternatives, SELECTION)
        six = self.genome_of("beam_grade", 6.0)
        self.assertEqual([(tuple(e["genome"]), e["scope"], e["deficit_closed"], e["gene_outcome"]) for e in pool["eligible"]],
                         [(six, "unit_target", 1.0, "target_unmet_no_useful_partial")])
        grade.pop("unit_targets")
        plain = ops.guided_alternatives(self.ctx, [flexural(self.ctx, S04[0], 1.413, {"beam_grade": grade}), self.drift()])
        self.assertEqual(ops.guided_distribution(alternatives, WEIGHTED), ops.guided_distribution(plain, WEIGHTED))
        self.assertEqual(ops.preferred_pool(plain, SELECTION)["eligible"], [])

    def test_a_units_target_is_annotated_when_the_shortlist_is_another_levers_move(self):
        # The size shortlist (16 x 32, depth) needs a cage that does not thread; the unit's threading target is the
        # width step 20 x 30, which the route proposes.
        size = size_screen(TIE01, (16.0, 30.0), S01, 1.749, (16.0, 32.0), 0.80, pool="admissible")
        size["unit_targets"] = [{"groups": list(S01), "to": [20.0, 30.0], "estimated_dcr": 0.82, "pool": "threading",
                                 "smallest_confirmed": True, "option_estimated_dcr": 0.82, "option_pool": "threading"}]
        d = flexural(self.ctx, X_EDGE, 1.749, {TIE01: size})
        alternatives = ops.guided_alternatives(self.ctx, [d, self.drift()])
        pool = ops.preferred_pool(alternatives, SELECTION)
        width, depth = self.genome_of(TIE01, (20.0, 30.0)), self.genome_of(TIE01, (16.0, 32.0))
        self.assertEqual([(tuple(e["genome"]), e["scope"], e["levers"]) for e in pool["eligible"]],
                         [(width, "unit_target", ["width"])])
        self.assertIn("threading_blocked_estimate", {e["reason"] for e in pool["excluded"] if tuple(e["genome"]) == depth})
        size.pop("unit_targets")
        plain = ops.guided_alternatives(self.ctx, [flexural(self.ctx, X_EDGE, 1.749, {TIE01: size}), self.drift()])
        self.assertEqual(ops.guided_distribution(alternatives, WEIGHTED), ops.guided_distribution(plain, WEIGHTED))

    def test_jump_draw_kind_separates_unit_target_steps(self):
        local = {"lever": "grade_up", "diagnostic_id": "d"}
        shortlisted = {**local, "step": "demand_sized", "jump": {"matched": {"scope": "shortlisted_option"}}}
        unit = {**local, "step": "demand_sized", "jump": {"matched": {"scope": "unit_target"}}}
        unmatched = {**local, "step": "demand_sized", "jump": {"matched": None}}
        self.assertEqual([gen.jump_draw_kind(o) for o in ([shortlisted], [unmatched], [local, shortlisted],
                                                          [local, unit], [local, unit, shortlisted], [local], [])],
                         ["jump", "jump", "annotated_local_step", "unit_target_step", "annotated_local_step", None, None])

    def test_the_entry_logs_the_units_own_estimate_pool_beside_the_genes(self):
        grade = grade_screen(5.0, 8.0, [(S01, 1.749, 1.638), (S04, 1.413, 1.329)])
        grade["partial"]["pool"] = "admissible"                           # the worst unit's (the lower band's)
        grade["partial"]["units"][0]["pool"] = "admissible"
        pool = ops.preferred_pool(ops.guided_alternatives(self.ctx, [flexural(self.ctx, S04[0], 1.413,
                                                                              {"beam_grade": grade})]), SELECTION)
        entry = next(e for e in pool["eligible"] if e["gene"] == "beam_grade")
        self.assertEqual((entry["pool"], entry["gene_pool"]), ("threading", "admissible"))

    def test_an_excluded_proposal_is_one_row_per_key_and_reason_and_partly_eligible_keys_are_kept(self):
        genes = {TIE01: size_screen(TIE01, (16.0, 30.0), S01, 1.749, (26.0, 36.0), 1.74)}
        both = [flexural(self.ctx, gid, 1.749, copy.deepcopy(genes)) for gid in S01]
        pool = ops.preferred_pool(ops.guided_alternatives(self.ctx, both), SELECTION)
        rows = [e for e in pool["excluded"] if tuple(e["genome"]) == self.genome_of(TIE01, (26.0, 36.0))]
        self.assertEqual([(e["reason"], e["levers"], e["diagnostic_ids"]) for e in rows],
                         [("negligible_improvement", ["depth", "width"], sorted(d["id"] for d in both))])
        # A genome eligible under one key keeps its exclusion under another (a marginal upper band).
        marginal = flexural(self.ctx, S04[0], 1.15, {
            "beam_grade": grade_screen(5.0, 8.0, [(S01, 1.749, 1.638), (S04, 1.413, 1.329)])})
        lower = o0002_like(self.ctx)
        pool = ops.preferred_pool(ops.guided_alternatives(self.ctx, [lower, marginal]), SELECTION)
        eight = self.genome_of("beam_grade", 8.0)
        self.assertIn(eight, [tuple(e["genome"]) for e in pool["eligible"]])
        self.assertEqual([(e["reason"], e["key"], e["eligible_under_keys"]) for e in pool["excluded"]
                          if tuple(e["genome"]) == eight],
                         [("not_severe", ops.failure_key(self.ctx, marginal), [ops.failure_key(self.ctx, lower)])])
        # Spent for this parent: the eligible entry is excluded as already drawn with every diagnostic and lever, and
        # the other key's row names the keys it was drawn under.
        both_lower = [lower, flexural(self.ctx, Y_EDGE, 1.749, copy.deepcopy(lower["capacity_jump"]["genes"]))]
        spent = ops.preferred_pool(ops.guided_alternatives(self.ctx, both_lower + [marginal]), SELECTION, exhausted=[eight])
        rows = {e["reason"]: e for e in spent["excluded"] if tuple(e["genome"]) == eight}
        self.assertEqual((rows["already_drawn_for_this_parent"]["diagnostic_ids"],
                          rows["already_drawn_for_this_parent"]["levers"]),
                         (sorted(d["id"] for d in both_lower), ["grade_up"]))
        self.assertEqual(rows["not_severe"]["already_drawn_under_keys"], [ops.failure_key(self.ctx, lower)])
        self.assertNotIn("eligible_under_keys", rows["not_severe"])


def operators_by_lever(alternatives):
    levers = {}
    for genome, (_p, origins) in ops._origins_by_genome(alternatives).items():
        levers[genome] = {o["lever"] for o in origins}
    return levers


class Reproducibility(Fixture):
    def test_seeded_draws_repeat_and_logged_probabilities_are_the_exact_distribution(self):
        alternatives = ops.guided_alternatives(self.ctx, [o0002_like(self.ctx), self.drift(), self.shear()])
        distribution = ops.guided_distribution(alternatives, SELECTION)

        def sequence(seed):
            rng = random.Random(seed)
            return [ops.choose_guided(alternatives, rng, SELECTION) for _ in range(40)]
        a, b = sequence(11), sequence(11)
        self.assertEqual([p["requested_genome"] for p in a], [p["requested_genome"] for p in b])
        self.assertNotEqual([p["requested_genome"] for p in a], [p["requested_genome"] for p in sequence(12)])
        for proposal in a:
            exact = distribution[tuple(proposal["requested_genome"])]
            record = proposal["selection"]
            self.assertAlmostEqual(record["proposal_probability"], exact["probability"])
            self.assertAlmostEqual(record["proposal_probability_preferred"], exact["preferred"])
            self.assertAlmostEqual(record["proposal_probability_normal"], exact["normal"])
            self.assertIn(record["branch"], ("preferred", "normal"))
        # The earlier policies are unchanged by the new one.
        uniform = ops.guided_distribution(alternatives, {"policy": ops.UNIFORM})
        self.assertAlmostEqual(sum(v["probability"] for v in uniform.values()), 1.0)
        rng_a, rng_b = random.Random(9), random.Random(9)
        for _ in range(20):
            self.assertEqual(ops.choose_guided(alternatives, rng_a, WEIGHTED)["requested_genome"],
                             ops.choose_guided(alternatives, rng_b, WEIGHTED)["requested_genome"])


class ControllerPreference(unittest.TestCase):
    CONFIG = replace(jt.ControllerPolicies.CONFIG, guided_selection=ops.PREFERRED, max_evaluations=7,
                     offspring_per_generation=6)

    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()
        self.layout = mock.patch.object(cf, "SLAB_RECORD", {"layout": None, "inputs": {"slab": {"thickness_in": 7.0},
                                                                                       "context": {"fixture": "priced"}}})
        self.layout.start()
        self.seed = gfx.seed_state()
        self.cfg = with_beam_practical_ratio(DesignConfig(), 0.0175)
        self.tmp = Path(tempfile.mkdtemp(prefix="ga_pref_"))

    def tearDown(self):
        self.layout.stop()
        self.frame.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def controller(self, out, config, basis=None):
        return gen.GenerationRun(self.seed, self.cfg, config, self.tmp / out, case_id="fixture_case",
                                 basis=basis or {"fixture": "preference", "seed": self.seed.identity()},
                                 evaluator=jt.PricedEvaluator(self.cfg), verbose=False)

    def test_the_preferred_branch_is_drawn_evaluated_and_never_accepted_by_estimate(self):
        result = self.controller("a", self.CONFIG).start().run()
        rows = [json.loads(line) for line in (self.tmp / "a" / "proposals.jsonl").read_text(encoding="utf-8").splitlines()]
        guided = [r for r in rows if (r.get("selection") or {}).get("policy") == ops.PREFERRED]
        self.assertTrue(guided)
        for row in guided:
            record = row["selection"]
            for key in ("branch", "branch_probabilities", "fallback", "pool_existed", "pool", "proposal_probability",
                        "proposal_probability_preferred", "proposal_probability_normal", "key",
                        "governing_diagnostic_id"):
                self.assertIn(key, record)
            self.assertIsNotNone(row["parent_identity"])
        preferred = [r for r in guided if r["selection"]["branch"] == "preferred"]
        self.assertTrue(preferred, "no preferred draw under this seed")
        drawn = Counter((r["parent"], tuple(r["requested_genome"])) for r in preferred)
        self.assertEqual(max(drawn.values()), 1)                          # never drawn twice from the branch per parent
        for row in preferred:
            self.assertIsNotNone(row["selection"]["estimate"])
        observations = [json.loads(line) for line in (self.tmp / "a" / "observations.jsonl").read_text(
            encoding="utf-8").splitlines() if json.loads(line)["type"] == "observation"]
        evaluated = [o for o in observations if (o["proposal"].get("selection") or {}).get("branch") == "preferred"]
        self.assertTrue(evaluated)
        for o in evaluated:
            self.assertIs(o["accepted"], False)
            self.assertEqual(o["classification"], "check_failed")
        counters = result["counts"]["preferred_selection"]
        self.assertEqual(counters["preferred_branch"], len(preferred))
        self.assertEqual(counters["evaluated"]["preferred"], len(evaluated))
        self.assertEqual(counters["accepted"], {"preferred": 0, "normal": 0})
        self.assertEqual(counters["guided_draws"], len(guided))
        checkpoint = json.loads((self.tmp / "a" / "checkpoint.json").read_text(encoding="utf-8"))
        for parent, genomes in checkpoint["preferred_drawn"].items():
            self.assertEqual(len(genomes), len({tuple(g) for g in genomes}))
        spent = {(p, tuple(g)) for p, genomes in checkpoint["preferred_drawn"].items() for g in genomes}
        for row in guided:
            record = row["selection"]
            # The whole distribution is logged: normalized, and the drawn proposal's entry is its probability.
            self.assertAlmostEqual(sum(e["probability"] for e in record["distribution"]), 1.0)
            entry = next(e for e in record["distribution"] if e["genome"] == row["requested_genome"])
            self.assertAlmostEqual(entry["probability"], record["proposal_probability"])
            if record["preferred_candidate"]:                            # spent for its parent, whichever branch drew it
                self.assertIn((row["parent"], tuple(row["requested_genome"])), spent)
        self.assertGreaterEqual(counters["proposed"], 1)
        self.assertEqual((counters["errors"], counters["incomplete"]),
                         ({"preferred": 0, "normal": 0}, {"preferred": 0, "normal": 0}))
        self.assertEqual(result["metrics"]["distinct_code_accepted_realized_designs"], 0)

    def test_the_controller_counts_each_drawn_jump_kind_apart(self):
        original = ops.choose_guided

        def with_unit_targets(alternatives, rng, selection=None):
            # A unit-target annotation on every drawn local step, as the router writes one (the counters' wiring only).
            proposal = original(alternatives, rng, selection)
            if proposal is not None and not any(o.get("step") == "demand_sized" for o in proposal["origins"]):
                proposal["origins"] = proposal["origins"] + [{**proposal["origins"][0], "step": "demand_sized",
                                                              "jump": {"matched": {"scope": "unit_target"}}}]
            return proposal
        with mock.patch.object(gen.operators, "choose_guided", with_unit_targets):
            result = self.controller("k", self.CONFIG).start().run()
        rows = [json.loads(line) for line in (self.tmp / "k" / "proposals.jsonl").read_text(encoding="utf-8").splitlines()]
        kinds = Counter(gen.jump_draw_kind(r["origins"]) for r in rows if r.get("selection") is not None)
        counts = result["counts"]
        self.assertGreater(kinds["unit_target_step"], 0)
        self.assertEqual((counts["jump_proposals_drawn"], counts["jump_annotated_local_steps_drawn"],
                          counts["jump_unit_target_steps_drawn"]),
                         (kinds["jump"], kinds["annotated_local_step"], kinds["unit_target_step"]))

    def test_an_evaluator_error_is_not_counted_as_a_completed_evaluation(self):
        class FailingChildren(jt.PricedEvaluator):
            def __call__(self, state, cfg, context=None, label="candidate", keep_actions=True):
                if self.parent is not None and state.identity() != self.parent:
                    raise RuntimeError("synthetic evaluator failure of a child")
                return super().__call__(state, cfg, context, label, keep_actions)
        controller = gen.GenerationRun(self.seed, self.cfg, self.CONFIG, self.tmp / "e", case_id="fixture_case",
                                       basis={"fixture": "preference", "seed": self.seed.identity()},
                                       evaluator=FailingChildren(self.cfg), verbose=False)
        result = controller.start().run()
        counters = result["counts"]["preferred_selection"]
        errored = [json.loads(line) for line in (self.tmp / "e" / "observations.jsonl").read_text(
            encoding="utf-8").splitlines() if json.loads(line).get("outcome") == "error"]
        self.assertTrue(errored)
        self.assertEqual(sum(counters["errors"].values()), len(errored))
        self.assertEqual(counters["evaluated"], {"preferred": 0, "normal": 0})
        self.assertEqual(counters["accepted"], {"preferred": 0, "normal": 0})

    def test_the_policy_and_its_parameters_bind_the_basis_and_refuse_a_changed_resume(self):
        basis = {"fixture": "preference", "seed": self.seed.identity()}
        config = replace(self.CONFIG, max_evaluations=1)
        first = self.controller("r", config, basis)
        policy = gen.proposal_policy(config, self.cfg)["guided_selection"]
        self.assertEqual({k: policy[k] for k in ("policy", "preference_version", "severe_utilization", "preferred_share",
                                                 "target_dcr", "min_gain")},
                         {"policy": ops.PREFERRED, "preference_version": ops.PREFERENCE_VERSION,
                          "severe_utilization": 1.25, "preferred_share": 0.8, "target_dcr": 0.85, "min_gain": 0.02})
        self.assertEqual(gen.proposal_policy(config, self.cfg)["beam_steel_ratio"]["practical_max"], 0.0175)
        first.start().run()
        for change in (replace(config, preferred_share=0.7), replace(config, preferred_severe_utilization=1.3),
                       replace(config, guided_selection=ops.SEVERITY_WEIGHTED)):
            other = self.controller("r", change, basis)
            self.assertNotEqual(other.basis_id, first.basis_id)
            with self.assertRaises(ValueError):
                other.resume()
        self.controller("r", config, basis).resume().close()

    def test_the_configuration_and_runner_declare_the_policy(self):
        for change in ({"preferred_share": 1.2}, {"preferred_share": -0.1}, {"preferred_severe_utilization": 0.9},
                       {"preferred_severe_utilization": float("inf")}, {"preferred_share": True},
                       {"beam_jump_policy": jumps.LOCAL}, {"preferred_share": 1.0}, {"beam_jump_min_gain": 0.0}):
            with self.assertRaises(ValueError, msg=repr(change)):
                replace(self.CONFIG, **change).validate()
        defaults = gen.GenerationConfig()
        self.assertEqual((defaults.guided_selection, defaults.preferred_severe_utilization, defaults.preferred_share),
                         (ops.UNIFORM, 1.25, 0.80))
        path = Path(gen.__file__).resolve().parents[1] / "tools" / "genetic_design_search" / "run.py"
        spec = importlib.util.spec_from_file_location("genetic_run_preference", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        args = runner.parser().parse_args(["--case-file", "case.json", "--seed-column", "30", "30", "6", "--seed-beam",
                                           "16", "30", "5", "--engine", "generation_v3", "--guided-selection",
                                           ops.PREFERRED, "--beam-jumps", jumps.DEMAND_SIZED, "--preferred-share", "0.8",
                                           "--preferred-severe-utilization", "1.25"])
        runner.engine_options(args)
        config = runner.generation_config(args)
        self.assertEqual((config.guided_selection, config.preferred_share, config.preferred_severe_utilization),
                         (ops.PREFERRED, 0.8, 1.25))
        self.assertEqual({k: runner.V3_ONLY[k] for k in ("preferred_severe_utilization", "preferred_share")},
                         {"preferred_severe_utilization": defaults.preferred_severe_utilization,
                          "preferred_share": defaults.preferred_share})
        with self.assertRaises(ValueError):
            runner.engine_options(self.runner_args(runner, "--preferred-share", "0.8"))

    @staticmethod
    def runner_args(runner, *extra):
        return runner.parser().parse_args(["--case-file", "case.json", "--seed-column", "30", "30", "6", "--seed-beam",
                                           "16", "30", "5", *extra])


if __name__ == "__main__":
    unittest.main()
