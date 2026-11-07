"""Demand-sized beam-flexure jumps and severity-weighted guided selection of the genetic search (2026-10-31).

  J1  the capacity requirement on frozen demand: Mu / target DCR, and the utilization over the target as the capacity
      ratio, never a multiplier of dimensions or steel; pairs without a defensible ratio give none;
  J2  real cage pricing of the fixture's perimeter beam tie (Genetic_Jumps.price_unit and screen_evaluation): a severe
      deficit shortlists a farther legal option than a marginal one; beyond the catalog the target is reported unmet
      and the end of the catalog is never called adequate; a recorded utilization the parent's pricing does not
      reproduce gives no target; the budget lists what it did not price;
  J3  the practical beam steel ratio ceiling: 0.0175 accepted, a stricter value kept, anything above refused by the
      configuration helper, the runner and the controller; the runner records it in the request, the evaluation
      identity and the basis; the screen prices with the configured ceiling;
  J4  the router: a screened jump is one more single-gene proposal in its lever's direction, finalized for evaluation and
      never accepted; an unbound, stale or another allele's screen gives none and says why; under the earlier rule the
      routes are unchanged; the column capacity-shear safeguard and the threading levers are untouched;
  J5  severity-weighted selection: severity raises a failure's probability without starving the others; repeated rows,
      every group of a design tie, extra catalog alternatives and a proposal offered under two operators or two keys
      add no weight; seeded draws repeat; invalid or stale evidence acquires no weight; a predicate check keeps the
      declared categorical weight and an unbounded ratio the weight at the cap; the earlier rule draws exactly as
      before;
  J6  the controller: a jump child is evaluated and assessed like any other and its estimate cannot mark it accepted;
      the proposal policy joins the basis, and a changed policy, parameter or ceiling refuses the resume.

Synthetic demands on the fixture frame, no analysis: nothing here is a designed or accepted structure.
"""
import copy
import importlib.util
import json
import math
import random
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_fixture as cf                                                    # noqa: E402
import genetic_fixture as gfx                                                    # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402
import test_beam_threading_evidence as tb                                        # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Genetic_Diagnostics as dg                                     # noqa: E402
from Design import Genetic_Generation as gen                                     # noqa: E402
from Design import Genetic_Jumps as jumps                                        # noqa: E402
from Design import Genetic_Operators as ops                                      # noqa: E402
from Design import Genetic_Search as ga                                          # noqa: E402
from Design import Group_Checks as checks                                        # noqa: E402
from Design import Grouped_Search as gs                                          # noqa: E402
from Design.Config import BEAM_PRACTICAL_RATIO_AUTHORIZED_MAX, DesignConfig, with_beam_practical_ratio  # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

TIE = "s01_03__sfrs_beam"
X_EDGE, Y_EDGE = "s01_03__beam_x__edge", "s01_03__beam_y__edge"
COLUMN = "s01_03__column__corner"
PARAMS = {"target_dcr": 0.85, "max_sections": 8, "max_seconds": 120.0, "partial_tolerance": 0.02, "min_gain": 0.02}
DEMAND_SIZED = {"beam_jump": jumps.DEMAND_SIZED}
SELECTION = {"policy": ops.SEVERITY_WEIGHTED, "severity_floor": 0.2, "severity_cap": 1.0, "categorical_weight": 0.5}
# The 10-minute run's case (research/ga_v3_10min_20261031/case_0007.json): the runner's request is built on it.
CASE = {"case_id": "case_0007", "label": "lhs_007", "num_bay_x": 3, "num_bay_y": 3, "num_floor": 6, "bay_x_width_ft": 23.5,
        "bay_y_width_ft": 28.0, "story_height_ft": 15.5, "seismic_site": "sdc_d_high",
        "geometry_name": "lhs_bx3_by3_s6_sh15p5_bwx23p5_bwy28_sdc_d_high",
        "purpose": "Latin hypercube point 7 of 10, seed 20261017."}


def demand_actions(state, mu, ratio=0.6):
    """The fixture's two combinations (their ids, as the record's action inventory expects) with every perimeter edge
    beam of the lower band at (ratio * mu sagging, mu hogging) and every other member as tb.synthetic_actions has it."""
    members = tb.synthetic_actions(state, {X_EDGE: (ratio * mu, mu), Y_EDGE: (ratio * mu, mu)})[0]["members"]
    return [{**action, "members": copy.deepcopy(members)} for action in gfx.actions()]


def slab_evidence():
    return {"slab": {"thickness_in": sp.SLAB_THICKNESS_IN, "superimposed_dead_load_ksf": sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF,
                     "live_load_mass_fraction": sp.SEISMIC_LIVE_LOAD_FRACTION},
            "transfer": None, "evidence": None, "reinforcement": None}


def rows_of(state, mu):
    actions = demand_actions(state, mu)
    return {gid: checks.beam_demand_rows(state.groups[gid]["member_tags"], actions) for gid in (X_EDGE, Y_EDGE)}


def size_index(ctx, gene, value):
    g = next(g for g in ctx["genes"] if g["name"] == gene)
    return [tuple(o) for o in g["options"]].index(tuple(value))


class Requirement(unittest.TestCase):
    def test_the_requirement_is_capacity_over_the_target_not_a_dimension_multiplier(self):
        # o0005 of the 10-minute run: s01_03 x edge, hogging, Mu 12084.6 against phi Mn 6993.1 kip-in.
        need = jumps.required_capacity(12084.647, 6993.129, 0.85)
        self.assertAlmostEqual(need["utilization"], 1.72808, places=4)
        self.assertAlmostEqual(need["capacity_ratio_required"], 1.72808 / 0.85, places=4)
        self.assertAlmostEqual(need["required_capacity_kip_in"], 12084.647 / 0.85, places=6)
        self.assertGreater(need["capacity_ratio_required"], 2.0)
        self.assertEqual(set(need), {"utilization", "required_capacity_kip_in", "capacity_ratio_required", "target_dcr"})

    def test_pairs_without_a_defensible_ratio_give_no_requirement(self):
        for demand, capacity, target in ((math.nan, 1.0, 0.85), (1.0, 0.0, 0.85), (1.0, -2.0, 0.85), (-1.0, 1.0, 0.85),
                                         (1.0, 1.0, 0.0), (1.0, 1.0, 1.2), (True, 1.0, 0.85), (1.0, math.inf, 0.85)):
            self.assertIsNone(jumps.required_capacity(demand, capacity, target), (demand, capacity, target))

    def test_lever_directions(self):
        self.assertTrue(jumps.in_direction("depth", (16.0, 30.0), (16.0, 32.0)))
        self.assertFalse(jumps.in_direction("depth", (16.0, 30.0), (20.0, 30.0)))
        self.assertTrue(jumps.in_direction("width", (16.0, 30.0), (20.0, 30.0)))
        self.assertFalse(jumps.in_direction("width", (20.0, 30.0), (18.0, 36.0)))
        self.assertTrue(jumps.in_direction("grade_up", 5.0, 8.0))
        self.assertFalse(jumps.in_direction("column_up", (26.0, 26.0), (28.0, 28.0)))


class Fixture(unittest.TestCase):
    """The fixture frame's GA seed (perimeter beams 16 x 30 in, columns 26 in), its routing context and real pricing."""

    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()
        self.seed = gfx.seed_state()
        self.space = ga.DesignSpace(self.seed)
        self.cfg = with_beam_practical_ratio(DesignConfig(), 0.0175)
        self.ctx = dg.parent_context(self.space, self.seed, basis_identity="jump-fixture", realized_identity="parent-J",
                                     capacity=gfx.capacity_evidence(self.seed))

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def baseline(self, mu, cfg=None):
        return jumps.price_unit(self.seed, (X_EDGE, Y_EDGE), rows_of(self.seed, mu), cfg or self.cfg)

    def shortfall(self, mu, u, reason="beam_flexure", component="flexure_negative", gid=X_EDGE):
        """A failed component diagnostic as the adapter writes it, at utilization ``u`` = Mu / phi Mn."""
        return {"id": f"strength|member.strength|{gid}#{component}", "reason": reason, "status": "fail",
                "applicable": True, "evidence_valid": True, "evidence_problems": [],
                "parent_identity": self.ctx["parent_identity"], "basis_identity": self.ctx["basis_identity"],
                "dependencies": {k: True for k in dg.RULES[reason][1]}, "demand": mu, "capacity": mu / u,
                "comparison": "<=", "units": "kip-in", "synthetic": False, "group_id": gid,
                "source": {"family": "strength", "check_id": "member.strength", "location": gid, "component": component}}

    def screen(self, mu, u=None, cfg=None, params=None, reason="beam_flexure"):
        cfg = cfg or self.cfg
        u = self.baseline(mu, cfg)["estimated_dcr"] if u is None else u
        d = self.shortfall(mu, u, reason)
        diagnosed = {"status": "diagnosed", "diagnostics": [d]}
        evaluation = {"status": "evaluated", "state": self.seed,
                      "evidence": {"actions": demand_actions(self.seed, mu), "slab": slab_evidence()}}
        block = jumps.screen_evaluation(evaluation, diagnosed, self.ctx, cfg, self.space, params or PARAMS)
        return block, d


class Screen(Fixture):
    def test_a_severe_deficit_shortlists_a_farther_option_than_a_marginal_one(self):
        marginal, d_marginal = self.screen(8000.0)
        severe, d_severe = self.screen(12000.0)
        self.assertLess(d_marginal["capacity_jump"]["utilization"], d_severe["capacity_jump"]["utilization"])
        chosen = {}
        for name, block, d in (("marginal", marginal, d_marginal), ("severe", severe, d_severe)):
            gene = block["genes"][TIE]
            self.assertEqual((block["status"], gene["calibration"], gene["outcome"]),
                             ("screened", "agrees", "target_met_estimated"), name)
            self.assertLessEqual(gene["meets_target"]["estimated_dcr"], 0.85)
            # The smallest adequate option: every priced option before it in catalog order is above the target.
            index = size_index(self.ctx, TIE, gene["meets_target"]["to"])
            self.assertTrue(all(r["estimated_dcr"] > 0.85 for r in gene["priced"] if r["order"] < index))
            self.assertAlmostEqual(d["capacity_jump"]["capacity_ratio_required"], d["capacity_jump"]["utilization"] / 0.85)
            chosen[name] = index
        self.assertGreater(chosen["severe"], chosen["marginal"])
        # Both are legal catalog options of the gene that the profile repair keeps.
        for index in chosen.values():
            genome = list(self.ctx["genome"])
            genome[next(i for i, g in enumerate(self.ctx["genes"]) if g["name"] == TIE)] = index
            self.assertEqual(self.space.repair(tuple(genome))[0][genome.index(index)], index)

    def test_beyond_the_catalog_the_target_is_unmet_and_the_end_is_not_called_adequate(self):
        block, d = self.screen(30000.0)
        gene = block["genes"][TIE]
        self.assertIsNone(gene["meets_target"])
        self.assertEqual(gene["outcome"], "target_unmet_estimated")
        farthest = [r for r in gene["priced"] if r["to"] == [26.0, 36.0]]
        self.assertEqual(len(farthest), 1)
        self.assertFalse(farthest[0]["meets_target"])
        self.assertGreater(farthest[0]["estimated_dcr"], 0.85)
        self.assertGreater(gene["partial"]["estimated_dcr"], 0.85)
        self.assertIsNone(d["capacity_jump"]["genes"][TIE]["meets_target"])
        # The partial option is the smallest within the tolerance of the best estimate.
        best = min(r["estimated_dcr"] for r in gene["priced"])
        self.assertLessEqual(gene["partial"]["estimated_dcr"], best + PARAMS["partial_tolerance"])

    def test_a_recorded_utilization_the_parent_pricing_does_not_reproduce_gives_no_target(self):
        u = self.baseline(12000.0)["estimated_dcr"]
        block, d = self.screen(12000.0, u=1.2 * u)
        gene = block["genes"][TIE]
        self.assertEqual((gene["calibration"], gene["outcome"], gene["meets_target"], gene["partial"]),
                         ("mismatch", "calibration_mismatch", None, None))
        self.assertEqual(gene["priced"], [])
        route = ops.route({**self.ctx, "guided_policy": DEMAND_SIZED}, d)
        self.assertIn(("fallback", "calibration_mismatch"),
                      {(j["status"], j.get("reason")) for j in route["jumps"] if j["gene"] == TIE})
        self.assertFalse([p for p in route["proposals"] if any(o.get("step") for o in p["origins"])])

    def test_a_cut_search_narrows_toward_the_smallest_adequate_option_instead_of_the_catalog_end(self):
        # A small parent (every beam tie 12 x 22 in) with a severe deficit: fifteen farther legal options.
        parent = gs.enforce_banded_section_order(gs.apply_moves(self.seed, {
            name: ((12.0, 22.0), "test parent") for name in ("s01_03__sfrs_beam", "s01_03__gravity_beam",
                                                            "s04_04__sfrs_beam", "s04_04__gravity_beam")}))[0]
        ctx = dg.parent_context(self.space, parent, basis_identity="jump-fixture", realized_identity="parent-J",
                                capacity=gfx.capacity_evidence(parent))
        mu = 4500.0
        u = jumps.price_unit(parent, (X_EDGE, Y_EDGE), rows_of(parent, mu), self.cfg)["estimated_dcr"]
        self.assertGreater(u / 0.85, 2.0)
        results = {}
        for label, sections in (("default", 8), ("all", 40)):
            d = self.shortfall(mu, u)
            evaluation = {"status": "evaluated", "state": parent,
                          "evidence": {"actions": demand_actions(parent, mu), "slab": slab_evidence()}}
            block = jumps.screen_evaluation(evaluation, {"status": "diagnosed", "diagnostics": [d]}, ctx, self.cfg,
                                            self.space, {**PARAMS, "max_sections": sections})
            results[label] = block["genes"][TIE]
        everything = results["all"]
        self.assertTrue(everything["smallest_confirmed"])
        smallest = min((r for r in everything["priced"] if r["meets_target"]), key=lambda r: r["order"])
        self.assertEqual(everything["meets_target"]["to"], smallest["to"])
        cut = results["default"]
        self.assertEqual(cut["outcome"], "target_met_estimated")
        self.assertNotEqual(cut["meets_target"]["to"], [26.0, 36.0])           # not the catalog end
        self.assertLessEqual(cut["pricings"], 8)
        self.assertIn(cut["smallest_confirmed"], (True, False))
        if cut["smallest_confirmed"]:
            self.assertEqual(cut["meets_target"]["to"], smallest["to"])
        # Every option the cut search left unpriced is named with its reason; none is called infeasible.
        self.assertEqual({u["reason"] for u in cut["unscreened"]} - {"budget", "larger_than_an_adequate_option"}, set())

    def test_the_partial_rule_keeps_the_first_option_within_the_tolerance_that_is_useful(self):
        # o0005 of the 10-minute run, s04_06__sfrs_beam from 16 x 30 in: the screened estimates (REPLAY.json).
        rows = [((16.0, 32.0), 13, 1.2802, True), ((20.0, 30.0), 14, 1.3542, True), ((20.0, 32.0), 15, 1.2394, False),
                ((18.0, 36.0), 16, 0.9526, True), ((24.0, 32.0), 17, 1.2039, False), ((22.0, 36.0), 18, 0.9388, False),
                ((26.0, 36.0), 19, 0.9282, False)]
        result = {"baseline_dcr": 1.4031, "outcome": None, "meets_target": None, "partial": None,
                  "priced": [{"to": list(to), "order": order, "estimated_dcr": dcr, "local_step": local,
                              "units": [{"groups": [X_EDGE, Y_EDGE], "estimated_dcr": dcr, "pool": "threading"}]}
                             for to, order, dcr, local in rows]}
        jumps._shortlist(result, PARAMS)
        # 22 x 36 is the smallest within the tolerance but gains less than 0.02 on the local 18 x 36; 26 x 36 does.
        self.assertEqual((result["outcome"], result["partial"]["to"]), ("target_unmet_estimated", [26.0, 36.0]))
        self.assertEqual(result["partial"]["pool"], "threading")
        self.assertIsNone(result["meets_target"])

    def test_a_pricing_the_deadline_cuts_is_unscreened_and_never_confirms_the_smallest(self):
        # Three options; the farthest reaches the target, the middle one's pricing is cut by the deadline.
        result = {"pricings": 1, "priced": [], "unscreened": [], "baseline_dcr": 1.5, "outcome": None,
                  "meets_target": None, "partial": None, "smallest_confirmed": None}
        clock = {"now": 0.0}

        def price(option):
            if option == (18.0, 32.0):
                clock["now"] = 10.0                             # the deadline passes during this pricing
                return {"status": "deadline", "estimated_dcr": None, "units": [], "repair_changes": []}, 1
            dcr = {(16.0, 32.0): 0.95, (20.0, 36.0): 0.70}[option]
            return {"status": "priced", "estimated_dcr": dcr,
                    "units": [{"groups": [X_EDGE, Y_EDGE], "estimated_dcr": dcr, "pool": "threading"}],
                    "repair_changes": []}, 1
        budget = jumps._Budget(8, lambda: clock["now"], 5.0)
        legal = [(13, (16.0, 32.0), []), (14, (18.0, 32.0), []), (15, (20.0, 36.0), [])]
        jumps._search(result, legal, set(), price, budget, PARAMS)
        self.assertEqual(result["meets_target"]["to"], [20.0, 36.0])
        self.assertIs(result["smallest_confirmed"], False)
        self.assertIn({"to": [18.0, 32.0], "reason": "deadline"}, result["unscreened"])
        self.assertNotIn([18.0, 32.0], [r["to"] for r in result["priced"]])

    def test_a_grade_option_costs_one_pricing_per_unit_it_prices(self):
        result = {"pricings": 1, "priced": [], "unscreened": [], "baseline_dcr": 2.0, "outcome": None,
                  "meets_target": None, "partial": None, "smallest_confirmed": None}
        calls = []

        def price(option):
            calls.append(option)
            return {"status": "priced", "estimated_dcr": 1.5,
                    "units": [{"groups": [f"g{k}"], "estimated_dcr": 1.5, "pool": "threading"} for k in range(3)],
                    "repair_changes": []}, 3
        budget = jumps._Budget(5, lambda: 0.0, None)
        jumps._search(result, [(1, 6.0, []), (2, 8.0, [])], {6.0}, price, budget, PARAMS, cost=3)
        self.assertEqual((calls, result["pricings"]), ([8.0], 4))              # 1 + 3, never above the cap of 5
        self.assertEqual(result["unscreened"], [{"to": 6.0, "reason": "budget"}])

    def test_the_budget_lists_what_it_did_not_price(self):
        block, _d = self.screen(30000.0, params={**PARAMS, "max_sections": 3})
        gene = block["genes"][TIE]
        self.assertEqual(gene["pricings"], 3)                    # the parent and two options
        self.assertEqual(len(gene["priced"]), 2)
        self.assertTrue(gene["unscreened"])
        self.assertEqual({u["reason"] for u in gene["unscreened"]}, {"budget"})
        # Every farther option of either lever is priced, left unscreened or rejected by the repair: none disappears.
        options = [tuple(o) for o in next(g for g in self.ctx["genes"] if g["name"] == TIE)["options"]]
        cone = [o for o in options if any(jumps.in_direction(lv, (16.0, 30.0), o) for lv in ("depth", "width"))]
        self.assertEqual(len(gene["priced"]) + len(gene["unscreened"]) + len(gene["rejected"]), len(cone))
        expired, _d = self.screen(30000.0, params={**PARAMS, "max_seconds": 1e-9})
        # A deadline already reached stops the screen: no target, and nothing is called infeasible.
        self.assertTrue(expired["deadline_reached"])
        self.assertTrue(all(g["meets_target"] is None and g["partial"] is None for g in expired["genes"].values()))
        self.assertFalse([r for g in expired["genes"].values() for r in g["priced"]])

    def test_a_failed_screen_is_recorded_and_gives_no_target(self):
        d = self.shortfall(12000.0, 1.5)
        diagnosed = {"status": "diagnosed", "diagnostics": [d]}
        evaluation = {"status": "evaluated", "state": self.seed, "evidence": {"actions": [{"id": "x", "members": {}}],
                                                                              "slab": slab_evidence()}}
        before = {key: getattr(sp, key) for key in jumps.ENVIRONMENT_KEYS}
        block = jumps.screen_evaluation(evaluation, diagnosed, self.ctx, self.cfg, self.space, PARAMS)
        self.assertEqual((block["status"], block["genes"]), ("failed", {}))
        self.assertIn("KeyError", block["error"])
        self.assertEqual(d["capacity_jump"]["status"], "failed")
        self.assertEqual({key: getattr(sp, key) for key in jumps.ENVIRONMENT_KEYS}, before)
        self.assertIsNone(mg.active())
        none = jumps.screen_evaluation(evaluation, {"status": "diagnosed", "diagnostics": [d]}, self.ctx, None,
                                       self.space, PARAMS)
        self.assertEqual(none["status"], "no_design_configuration")


class Ceiling(Fixture):
    def test_the_authorized_ceiling_is_the_most_a_run_may_configure(self):
        self.assertEqual(BEAM_PRACTICAL_RATIO_AUTHORIZED_MAX, 0.0175)
        base = DesignConfig()
        self.assertEqual(base.rebar.rho_beam_practical_max, 0.0175)         # the default is the authorized ceiling (2026-11-07)
        self.assertEqual(with_beam_practical_ratio(base, 0.0175).rebar.rho_beam_practical_max, 0.0175)
        self.assertEqual(with_beam_practical_ratio(base, 0.0125).rebar.rho_beam_practical_max, 0.0125)  # stricter kept
        self.assertEqual(base.rebar.rho_beam_practical_max, 0.0175)         # the input is not changed
        for value in (0.018, 0.0175 + 1e-9, 0.025, 0.0, -0.01, math.nan, math.inf, True, "0.0175"):
            with self.assertRaises(ValueError, msg=repr(value)):
                with_beam_practical_ratio(base, value)

    def test_the_controller_refuses_a_ceiling_above_the_authorized_one(self):
        cfg = DesignConfig()
        cfg.rebar.rho_beam_practical_max = 0.02
        with self.assertRaises(ValueError):
            gen.GenerationRun(self.seed, cfg, gen.GenerationConfig(), Path(tempfile.gettempdir()) / "never_created",
                              case_id="fixture", basis={"fixture": True})

    def test_the_screen_prices_with_the_configured_ceiling(self):
        at_015 = self.baseline(8000.0, with_beam_practical_ratio(DesignConfig(), 0.015))
        at_0175 = self.baseline(8000.0)
        self.assertLess(at_0175["estimated_dcr"], at_015["estimated_dcr"] - 0.05)
        block, _d = self.screen(8000.0)
        self.assertEqual(block["ratio_ceiling"], {"practical_max": 0.0175, "code_max": sp.beam_rho_max_code(),
                                                  "effective_max": min(0.0175, sp.beam_rho_max_code())})
        self.assertNotIn("0.015", Path(jumps.__file__).read_text(encoding="utf-8"))


class RunnerCeiling(unittest.TestCase):
    """The runner on the 10-minute run's case (its own analysis profile, not the fixture frame); the parameters are
    restored afterwards."""

    def setUp(self):
        mg.clear()
        self.snapshot = dict(vars(sp))

    def tearDown(self):
        mg.clear()
        for key in set(vars(sp)) - set(self.snapshot):
            delattr(sp, key)
        vars(sp).update(self.snapshot)

    def test_the_runner_records_the_ceiling_in_the_request_identity_and_basis(self):
        path = Path(gen.__file__).resolve().parents[1] / "tools" / "genetic_design_search" / "run.py"
        spec = importlib.util.spec_from_file_location("genetic_run_jumps", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        tmp = Path(tempfile.mkdtemp(prefix="ga_jumps_"))
        try:
            (tmp / "case.json").write_text(json.dumps(CASE), encoding="utf-8")
            common = ["--engine", "generation_v3", "--case-file", str(tmp / "case.json"), "--profile",
                      "v2_nonlinear_flexure_screening_v1", "--grouping", "system_three_story_v1", "--seed-column", "30", "30",
                      "6", "--seed-beam", "16", "30", "5", "--guided-selection", "severity_weighted_v1", "--beam-jumps",
                      "demand_sized_v1"]
            requests = {}
            for label, extra in (("authorized", ["--beam-rho-practical-max", "0.015"]), ("default", [])):
                state, cfg, config, request = runner.prepare(runner.parser().parse_args(common + extra))
                requests[label] = (state, cfg, config, request)
            # A stricter run ceiling than the 0.0175 default is recorded the same way (2026-11-07).
            state, cfg, config, request = requests["authorized"]
            self.assertEqual(cfg.rebar.rho_beam_practical_max, 0.015)
            self.assertEqual(request["evaluation_identity"]["policy"]["rebar"]["rho_beam_practical_max"], 0.015)
            self.assertEqual(request["design_config_overrides"], {"rebar.rho_beam_practical_max": {
                "value": 0.015, "default": 0.0175, "authorized_max": 0.0175}})
            policy = request["basis"]["proposal_policy"]
            self.assertEqual(policy["beam_steel_ratio"]["practical_max"], 0.015)
            self.assertEqual((policy["guided_selection"]["policy"], policy["beam_jump"]["policy"]),
                             ("severity_weighted_v1", "demand_sized_v1"))
            self.assertEqual(request["generation_config"]["beam_jump_policy"], "demand_sized_v1")
            controller = gen.GenerationRun(state, cfg, config, tmp / "run", case_id="case_0007", basis=request["basis"])
            self.assertEqual(controller.basis_id, dg.digest(request["basis"]))
            _s, default_cfg, _c, default = requests["default"]
            self.assertEqual(default_cfg.rebar.rho_beam_practical_max, 0.0175)
            self.assertNotIn("design_config_overrides", default)
            self.assertNotEqual(default["evaluation_identity"]["sha256"], request["evaluation_identity"]["sha256"])
            self.assertNotEqual(dg.digest(default["basis"]), dg.digest(request["basis"]))
            self.assertNotEqual(default["sha256"], request["sha256"])
            with self.assertRaises(ValueError):
                runner.prepare(runner.parser().parse_args(common + ["--beam-rho-practical-max", "0.018"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Router(Fixture):
    def jump_route(self, mu=12000.0):
        _block, d = self.screen(mu)
        return d, ops.route({**self.ctx, "guided_policy": DEMAND_SIZED}, d)

    def test_a_screened_jump_is_one_single_gene_proposal_in_its_lever_direction(self):
        d, route = self.jump_route()
        target = tuple(d["capacity_jump"]["genes"][TIE]["meets_target"]["to"])
        jumped = [p for p in route["proposals"] if any(o.get("step") == "demand_sized" for o in p["origins"])]
        self.assertIn(target, {tuple(p["to"]) for p in jumped if p["requested_gene"] == TIE})
        for p in route["proposals"]:
            self.assertEqual(sum(a != b for a, b in zip(p["requested_genome"], self.ctx["genome"])), 1)
        proposal = next(p for p in jumped if p["requested_gene"] == TIE)
        for origin in proposal["origins"]:
            self.assertTrue(jumps.in_direction(origin["lever"], tuple(proposal["from"]), tuple(proposal["to"])))
        info = next(o["jump"] for o in proposal["origins"] if o.get("jump"))
        self.assertTrue(info["meets_target_estimated"])
        self.assertLessEqual(info["estimated_dcr"], 0.85)
        repaired, _trial, log = self.space.repair(tuple(proposal["requested_genome"]))
        final = ops.finalize(self.ctx, proposal, list(repaired), log)
        self.assertEqual(final["status"], "ready_for_evaluation")
        self.assertIs(final["engineering_accepted"], False)
        self.assertEqual(tuple(final["requested_change"]["to"]), target)

    def test_the_earlier_rule_routes_exactly_as_before(self):
        d, route = self.jump_route()
        earlier = ops.route(self.ctx, d)
        self.assertNotIn("jumps", earlier)
        self.assertFalse([p for p in earlier["proposals"] if any(o.get("step") for o in p["origins"])])
        local = sorted(tuple(p["requested_genome"]) for p in earlier["proposals"])
        kept = sorted(tuple(p["requested_genome"]) for p in route["proposals"]
                      if any(not o.get("step") for o in p["origins"]))
        self.assertEqual(local, kept)
        # Under the earlier rule the screen is ignored even where one exists: the same as with no screen at all.
        bare = {k: v for k, v in d.items() if k != "capacity_jump"}
        self.assertEqual(ops.route(self.ctx, bare)["proposals"], earlier["proposals"])

    def test_an_unbound_stale_or_failed_screen_gives_no_jump_and_says_why(self):
        d, _route = self.jump_route()
        ctx = {**self.ctx, "guided_policy": DEMAND_SIZED}
        cases = (("no_screen", lambda x: x.pop("capacity_jump")),
                 ("screen_not_bound_to_this_parent", lambda x: x["capacity_jump"].update(parent_identity="another")),
                 ("screen_not_bound_to_this_parent", lambda x: x["capacity_jump"].update(version="another")),
                 ("screen_failed", lambda x: x["capacity_jump"].update(status="failed")),
                 ("screen_of_another_allele", lambda x: x["capacity_jump"]["genes"][TIE].update({"from": [20.0, 30.0]})))
        for reason, change in cases:
            changed = copy.deepcopy(d)
            change(changed)
            route = ops.route(ctx, changed)
            self.assertIn(reason, {j.get("reason") for j in route["jumps"] if j["gene"] == TIE}, reason)
            self.assertFalse([p for p in route["proposals"] if p["requested_gene"] == TIE
                              and any(o.get("step") for o in p["origins"])], reason)

    def test_the_column_shear_safeguard_and_the_threading_levers_are_unchanged(self):
        ctx = {**self.ctx, "guided_policy": DEMAND_SIZED}
        shear = {**self.shortfall(500.0, 1.05), "id": f"capacity|column.capacity_shear_section|{COLUMN}",
                 "reason": "column_capacity_shear", "group_id": COLUMN, "units": "kip", "cage_at_ratio_floor": True,
                 "dependencies": {"probable_equilibrium": True, "transverse_selection": True}}
        route = ops.route(ctx, shear)
        self.assertEqual({o["lever"] for p in route["proposals"] for o in p["origins"]}, {"grade_up"})
        self.assertIn(("column_up", COLUMN), {(w["lever"], w["group"]) for w in route["withheld"]})
        self.assertEqual(route["jumps"], [])                             # no jump lever on a column shear failure
        cores = sorted({t["core_group"] for t in self.ctx["joint_types"].values() if X_EDGE in t["beams"]["x"]})
        constrained = {**self.shortfall(12000.0, 1.6, reason="beam_threading_constrained_strength"),
                       "joint_topology_consistent": True, "blocking_cores": cores, "blocking_core_cages": cores,
                       "blocking_transitions": [], "lane_lookahead": {cores[0]: [30.0, 30.0]}, "transition_lookahead": []}
        before, after = ops.route(self.ctx, constrained), ops.route(ctx, constrained)
        self.assertEqual([(p["requested_gene"], p["to"]) for p in before["proposals"]],
                         [(p["requested_gene"], p["to"]) for p in after["proposals"]])
        self.assertEqual(before["withheld"], after["withheld"])
        self.assertEqual({j["reason"] for j in after["jumps"]}, {"no_screen"})


def selection_ctx():
    """The operators' fixture context (gf.frame must be active)."""
    seed = gfx.seed_state()
    return dg.parent_context(ga.DesignSpace(seed), seed, basis_identity="fixture-basis", realized_identity="parent-A",
                             capacity=gfx.capacity_evidence(seed))


def diag(ctx, reason, demand, capacity, comparison="<=", units="kip", **fields):
    item = {"id": f"fixture|{reason}|{len(fields)}|{demand}|{fields.get('group_id')}|{fields.get('story')}"
                  f"|{fields.get('component', '')}",
            "reason": reason, "status": "fail", "applicable": True, "evidence_valid": True,
            "parent_identity": ctx["parent_identity"], "basis_identity": ctx["basis_identity"],
            "dependencies": {k: True for k in dg.RULES[reason][1]}, "demand": demand, "capacity": capacity,
            "comparison": comparison, "units": units, "synthetic": True}
    item.update(fields)
    return item


class Selection(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()
        self.ctx = selection_ctx()

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def table(self, diagnostics, selection=SELECTION):
        rows, _shares = ops.failure_table(ops.guided_alternatives(self.ctx, diagnostics), selection)
        return {row["key"]: row for row in rows}

    def key(self, reason, gid):
        return ops.failure_key(self.ctx, {"reason": reason, "group_id": gid})

    def test_severity_raises_a_failures_probability_without_starving_the_others(self):
        severe = diag(self.ctx, "beam_flexure", 1700.0, 1000.0, units="kip-in", group_id=X_EDGE)
        mild = diag(self.ctx, "column_capacity_shear", 105.0, 100.0, group_id=COLUMN)
        table = self.table([severe, mild])
        strength, shear = table[self.key("beam_flexure", X_EDGE)], table[self.key("column_capacity_shear", COLUMN)]
        self.assertEqual((strength["key"], shear["key"]), ("member_strength|s01_03__sfrs_beam",
                                                           "capacity_shear|s01_03__sfrs_column"))
        self.assertAlmostEqual(strength["severity"], 0.7)
        self.assertAlmostEqual(shear["severity"], 0.05)
        self.assertAlmostEqual(strength["weight"], 0.2 + 0.8 * 0.7)
        self.assertAlmostEqual(shear["weight"], 0.2 + 0.8 * 0.05)
        # Disjoint proposals: each failure is drawn in proportion to its weight.
        self.assertAlmostEqual(shear["governs_probability"], shear["weight"] / (strength["weight"] + shear["weight"]))
        self.assertGreater(strength["governs_probability"], shear["governs_probability"])
        self.assertGreaterEqual(shear["governs_probability"], 0.2 / (strength["weight"] + shear["weight"]) - 1e-12)
        alternatives = ops.guided_alternatives(self.ctx, [severe, mild])
        rng = random.Random(20261031)
        drawn = [ops.choose_guided(alternatives, rng, SELECTION)["selection"]["key"] for _ in range(3000)]
        frequency = drawn.count(shear["key"]) / len(drawn)
        self.assertAlmostEqual(frequency, shear["governs_probability"], delta=0.03)
        self.assertGreater(frequency, 0.0)
        # A severity beyond the cap weighs 1, no more: the ratio of weights is bounded by 1 / floor.
        huge = self.table([diag(self.ctx, "beam_flexure", 99000.0, 1000.0, units="kip-in", group_id=X_EDGE), mild])
        self.assertEqual(huge[strength["key"]]["weight"], 1.0)

    def test_every_group_of_a_design_tie_is_one_failure(self):
        x = diag(self.ctx, "beam_flexure", 1700.0, 1000.0, units="kip-in", group_id=X_EDGE)
        y = diag(self.ctx, "beam_flexure", 1300.0, 1000.0, units="kip-in", group_id=Y_EDGE)
        mild = diag(self.ctx, "column_capacity_shear", 105.0, 100.0, group_id=COLUMN)
        both, one = self.table([x, y, mild]), self.table([x, mild])
        self.assertEqual(set(both), set(one))
        strength = both[self.key("beam_flexure", X_EDGE)]
        self.assertEqual(strength["governing_diagnostic_id"], x["id"])
        self.assertAlmostEqual(strength["utilization"], 1.7)
        for key in both:
            self.assertAlmostEqual(both[key]["governs_probability"], one[key]["governs_probability"])

    def test_a_proposal_offered_by_two_keys_is_drawn_once_at_its_largest_share(self):
        shear = diag(self.ctx, "beam_capacity_shear", 130.0, 100.0, group_id=X_EDGE)
        room = diag(self.ctx, "beam_width_room", 0.5, 1.0, comparison=">=", units="in", group_id=X_EDGE)
        alternatives = ops.guided_alternatives(self.ctx, [shear, room])
        rows, shares = ops.failure_table(alternatives, SELECTION)
        by_key = {row["key"]: row for row in rows}
        a, b = by_key[self.key("beam_capacity_shear", X_EDGE)], by_key[self.key("beam_width_room", X_EDGE)]
        shared = {tuple(g) for g in a["genomes"]} & {tuple(g) for g in b["genomes"]}
        self.assertTrue(shared)                                      # the width steps
        total = sum(s for s, _r in shares.values())
        for genome in shared:
            share = max(a["weight"] / len(a["genomes"]), b["weight"] / len(b["genomes"]))
            self.assertAlmostEqual(shares[genome][0], share)         # the larger share, not the sum
        self.assertAlmostEqual(sum(s for s, _r in shares.values()) / total, 1.0)
        proposal = ops.choose_guided(alternatives, random.Random(9), SELECTION)
        self.assertAlmostEqual(proposal["selection"]["proposal_probability"],
                               shares[tuple(proposal["requested_genome"])][0] / total)

    def test_an_unbounded_ratio_takes_the_weight_at_the_cap(self):
        overlap = diag(self.ctx, "beam_width_room", -0.2, 1.2, comparison=">=", units="in", group_id=X_EDGE)
        mild = diag(self.ctx, "beam_width_room", 1.0, 1.1, comparison=">=", units="in", group_id=X_EDGE, component="b")
        (row,), _shares = ops.failure_table(ops.guided_alternatives(self.ctx, [mild, overlap]), SELECTION)
        self.assertEqual((row["unbounded"], row["weight"], row["governing_diagnostic_id"]), (True, 1.0, overlap["id"]))
        self.assertTrue(ops.unbounded(overlap))
        self.assertFalse(ops.unbounded(mild))
        self.assertIsNone(ops.utilization(overlap))

    def test_repeated_rows_and_extra_alternatives_add_no_weight(self):
        once = [diag(self.ctx, "beam_flexure", 1700.0, 1000.0, units="kip-in", group_id=X_EDGE, component="n"),
                diag(self.ctx, "column_capacity_shear", 105.0, 100.0, group_id=COLUMN)]
        repeated = once + [diag(self.ctx, "beam_flexure", 1200.0, 1000.0, units="kip-in", group_id=X_EDGE, component="p"),
                           diag(self.ctx, "column_capacity_shear", 102.0, 100.0, group_id=COLUMN, component="row2")]
        a, b = self.table(once), self.table(repeated)
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertAlmostEqual(a[key]["governs_probability"], b[key]["governs_probability"])
            self.assertEqual(a[key]["genomes"], b[key]["genomes"])
        # Two failures of equal severity are equally likely whatever their number of catalog alternatives.
        equal = self.table([diag(self.ctx, "beam_flexure", 1500.0, 1000.0, units="kip-in", group_id=X_EDGE),
                            diag(self.ctx, "column_capacity_shear", 150.0, 100.0, group_id=COLUMN)])
        strength, shear = equal[self.key("beam_flexure", X_EDGE)], equal[self.key("column_capacity_shear", COLUMN)]
        self.assertNotEqual(len(strength["genomes"]), len(shear["genomes"]))
        self.assertAlmostEqual(strength["governs_probability"], shear["governs_probability"])

    def test_a_proposal_under_two_operators_counts_once(self):
        flexure = diag(self.ctx, "beam_flexure", 1700.0, 1000.0, units="kip-in", group_id=X_EDGE, component="p")
        constrained = diag(self.ctx, "beam_threading_constrained_strength", 1500.0, 1000.0, units="kip-in",
                           group_id=X_EDGE, component="n", joint_topology_consistent=True, blocking_cores=[],
                           blocking_core_cages=[], blocking_transitions=[], lane_lookahead={}, transition_lookahead=[])
        alternatives = ops.guided_alternatives(self.ctx, [flexure, constrained])
        shared = {tuple(p["requested_genome"]) for p in alternatives["by_operator"]["H2"]} & \
            {tuple(p["requested_genome"]) for p in alternatives["by_operator"]["H3"]}
        self.assertTrue(shared)                                          # depth, width and grade under H2 and H3
        (row,), _shares = ops.failure_table(alternatives, SELECTION)
        self.assertEqual(row["operators"], ["H2", "H3"])
        self.assertEqual(len(row["genomes"]), len({tuple(g) for g in row["genomes"]}))
        rng = random.Random(3)
        proposal = ops.choose_guided(alternatives, rng, SELECTION)
        self.assertAlmostEqual(proposal["selection"]["proposal_probability"], 1.0 / len(row["genomes"]))
        # Under the earlier rule the same genome under two operators was twice as likely as the others.
        uniform = ops.choose_guided(alternatives, random.Random(3), {"policy": ops.UNIFORM})
        self.assertIn("proposal_probability", uniform["selection"])

    def test_seeded_draws_repeat_and_the_earlier_rule_draws_as_before(self):
        diagnostics = [diag(self.ctx, "beam_flexure", 1700.0, 1000.0, units="kip-in", group_id=X_EDGE),
                       diag(self.ctx, "column_capacity_shear", 105.0, 100.0, group_id=COLUMN),
                       diag(self.ctx, "story_drift", 0.03, 0.02, units="in/in", story=2, axis="x")]
        alternatives = ops.guided_alternatives(self.ctx, diagnostics)

        def sequence(seed, selection):
            rng = random.Random(seed)
            return [tuple(ops.choose_guided(alternatives, rng, selection)["requested_genome"]) for _ in range(50)]
        self.assertEqual(sequence(7, SELECTION), sequence(7, SELECTION))
        self.assertNotEqual(sequence(7, SELECTION), sequence(8, SELECTION))
        rng_a, rng_b = random.Random(5), random.Random(5)
        for _ in range(50):
            a = ops.choose_guided(alternatives, rng_a)
            b = ops.choose_guided(alternatives, rng_b, {"policy": ops.UNIFORM, **{k: v for k, v in SELECTION.items()
                                                                                 if k != "policy"}})
            self.assertEqual((a["requested_genome"], a["selected_operator"]), (b["requested_genome"], b["selected_operator"]))
        self.assertEqual(rng_a.getstate(), rng_b.getstate())
        with self.assertRaises(ValueError):
            ops.choose_guided(alternatives, random.Random(1), {**SELECTION, "policy": "adaptive"})

    def test_invalid_or_stale_evidence_acquires_no_weight(self):
        valid = diag(self.ctx, "column_capacity_shear", 105.0, 100.0, group_id=COLUMN)
        bad = [{**diag(self.ctx, "beam_flexure", 9000.0, 1000.0, units="kip-in", group_id=X_EDGE), "evidence_valid": False},
               {**diag(self.ctx, "beam_flexure", 9000.0, 1000.0, units="kip-in", group_id=Y_EDGE),
                "parent_identity": "another parent"},
               diag(self.ctx, "beam_flexure", math.nan, 1000.0, units="kip-in", group_id=X_EDGE, component="nan"),
               {**diag(self.ctx, "beam_flexure", 9000.0, 1000.0, units="kip-in", group_id=X_EDGE, component="u"),
                "status": "unknown"},
               {**diag(self.ctx, "beam_flexure", 9000.0, 1000.0, units="kip-in", group_id=X_EDGE, component="d"),
                "dependencies": {"member_actions": False, "section_strength": True}}]
        self.assertEqual(set(self.table([valid] + bad)), {self.key("column_capacity_shear", COLUMN)})
        self.assertIsNone(ops.choose_guided(ops.guided_alternatives(self.ctx, bad), random.Random(1), SELECTION))
        for d in bad[2:3]:
            self.assertIsNone(ops.utilization(d))

    def test_a_predicate_check_keeps_the_categorical_weight(self):
        kind = next(t for t in self.ctx["joint_types"].values() if X_EDGE in t["beams"]["x"])
        type_id = next(k for k, v in self.ctx["joint_types"].items() if v is kind)
        threading = diag(self.ctx, "beam_threading", 0, 1, comparison="==", units="predicate", group_id=X_EDGE,
                         joint_id=type_id, joint_type_id=type_id, axis="x", core_group=kind["core_group"])
        (row,), _shares = ops.failure_table(ops.guided_alternatives(self.ctx, [threading]), SELECTION)
        self.assertEqual((row["numeric"], row["utilization"], row["severity"], row["weight"]), (False, None, None, 0.5))
        self.assertIsNone(ops.utilization(threading))

    def test_utilization_follows_the_check_direction(self):
        self.assertEqual(ops.utilization({"demand": 3.0, "capacity": 2.0, "comparison": "<="}), 1.5)
        self.assertEqual(ops.utilization({"demand": 0.5, "capacity": 1.0, "comparison": ">="}), 2.0)  # required/provided
        for d in ({"demand": 1.0, "capacity": 0.0, "comparison": "<="}, {"demand": 0.0, "capacity": 1.0, "comparison": ">="},
                  {"demand": 0, "capacity": 1, "comparison": "=="}, {"demand": None, "capacity": 1.0, "comparison": "<="}):
            self.assertIsNone(ops.utilization(d))

    def test_failure_keys_group_repeated_evidence_by_actionable_unit(self):
        a = diag(self.ctx, "story_drift", 0.03, 0.02, units="in/in", story=2, axis="x")
        b = diag(self.ctx, "story_drift", 0.025, 0.02, units="in/in", story=3, axis="y")
        self.assertEqual(ops.failure_key(self.ctx, a), ops.failure_key(self.ctx, b))      # one band
        c = diag(self.ctx, "story_drift", 0.025, 0.02, units="in/in", story=4, axis="x")
        self.assertNotEqual(ops.failure_key(self.ctx, a), ops.failure_key(self.ctx, c))   # the band above
        self.assertEqual(ops.failure_key(self.ctx, {"reason": "beam_flexure", "group_id": X_EDGE}),
                         ops.failure_key(self.ctx, {"reason": "beam_threading_constrained_strength", "group_id": Y_EDGE}))


class PricedEvaluator:
    """Synthetic evaluations whose perimeter beam strength comes from the screen's own pricing of the submitted design
    (Mu = ``mu`` hogging on the lower-band edge beams), so a parent's recorded utilization is the one its screen
    reproduces; a child (any design other than ``parent``) carries ``redistribution`` times that demand, as a
    reanalysis of a stiffer frame may. Everything else passes."""

    def __init__(self, cfg, mu=12000.0, redistribution=1.5):
        self.cfg, self.mu, self.redistribution, self.parent = cfg, mu, redistribution, None

    def __call__(self, state, cfg, context=None, label="candidate", keep_actions=True):
        if self.parent is None:
            self.parent = state.identity()
        mu = self.mu * (1.0 if state.identity() == self.parent else self.redistribution)
        actions = demand_actions(state, mu)
        rows = {gid: checks.beam_demand_rows(state.groups[gid]["member_tags"], actions) for gid in (X_EDGE, Y_EDGE)}
        # The record's own slab (contract_fixture: the thickness and the layout its realized identity binds) with the
        # floor inputs an installation needs; priced in that environment, as the screen will price it.
        slab = {"slab": {**slab_evidence()["slab"], "thickness_in": cf.SLAB_THICKNESS_IN}, "transfer": None,
                "evidence": None, "reinforcement": copy.deepcopy(cf.SLAB_RECORD)}
        with jumps.evaluated_environment({"evidence": {"slab": slab}, "state": state}):
            u = jumps.price_unit(state, (X_EDGE, Y_EDGE), rows, self.cfg)["estimated_dcr"]
        overrides = {gid: {"flexure_negative": {"dcr": u, "mu_kip_in": mu, "phi_mn_kip_in": mu / u}} for gid in (X_EDGE, Y_EDGE)}
        record = gfx.record(state, cfg, strength_overrides=overrides)
        record["evidence"]["actions"] = actions
        record["evidence"]["slab"].update(slab={**record["evidence"]["slab"]["slab"], **slab["slab"]}, transfer=None,
                                          evidence=None)
        return record


class ControllerPolicies(unittest.TestCase):
    CONFIG = gen.GenerationConfig(random_seed=4, population_size=1, offspring_per_generation=4, max_generations=2,
                                  max_evaluations=6, initialization_policy="seed_only_v1", initialization_evaluations=0,
                                  guided_probability=1.0, max_exploration_evaluations=0, max_stagnant_generations=3,
                                  guided_selection=ops.SEVERITY_WEIGHTED, beam_jump_policy=jumps.DEMAND_SIZED,
                                  beam_jump_max_seconds=120.0)

    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()
        # The contract fixture's slab layout is a stub without layers; these records carry none (no mats), so the
        # composite strengths can be priced in the record's own environment.
        self.layout = mock.patch.object(cf, "SLAB_RECORD", {"layout": None, "inputs": {"slab": {"thickness_in": 7.0},
                                                                                       "context": {"fixture": "priced"}}})
        self.layout.start()
        self.seed = gfx.seed_state()
        self.cfg = with_beam_practical_ratio(DesignConfig(), 0.0175)
        self.tmp = Path(tempfile.mkdtemp(prefix="ga_jump_"))

    def tearDown(self):
        self.layout.stop()
        self.frame.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def controller(self, out, config, evaluator, cfg="default", basis=None):
        return gen.GenerationRun(self.seed, self.cfg if cfg == "default" else cfg, config, self.tmp / out,
                                 case_id="fixture_case", basis=basis or {"fixture": "priced", "seed": self.seed.identity()},
                                 evaluator=evaluator, verbose=False)

    def test_a_jump_child_is_evaluated_and_its_estimate_cannot_mark_it_accepted(self):
        result = self.controller("a", self.CONFIG, PricedEvaluator(self.cfg)).start().run()
        records = [json.loads(line) for line in (self.tmp / "a" / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
        records = [r for r in records if r["type"] == "observation"]
        seed = records[0]
        screen = seed["diagnostics"]["beam_jump_screen"]
        self.assertEqual((screen["status"], screen["genes"][TIE]["calibration"]), ("screened", "agrees"))
        jumped = [r for r in records[1:] if any(o.get("step") == "demand_sized" for o in r["origins"])]
        self.assertTrue(jumped, "no demand-sized child was drawn under this seed")
        for child in jumped:
            self.assertEqual(child["proposal"]["selection"]["policy"], ops.SEVERITY_WEIGHTED)
            self.assertIs(child["accepted"], False)
            self.assertEqual(child["classification"], "check_failed")
            estimated = [t for t in child["vs_parent"]["targets"] if "estimated" in t]
            self.assertTrue(estimated)
            for target in estimated:
                # The estimate is kept apart from the evaluated outcome, which decides.
                self.assertEqual(target["child"]["status"], "fail")
                self.assertIs(target["resolved"], False)
            self.assertEqual(child["intervention"]["requested"]["gene"], child["proposal"]["requested_gene"])
        counts = result["counts"]
        self.assertGreaterEqual(counts["jump_proposals_drawn"] + counts["jump_annotated_local_steps_drawn"]
                                + counts["jump_unit_target_steps_drawn"], len(jumped))
        by_id = {r["id"]: r for r in records}
        for child in records[1:]:
            self.assertEqual(child["proposal"]["parent_identity"], by_id[child["parent"]]["realized_identity"])
        self.assertEqual(counts["beam_jump_screens"]["observations"], len(records))
        self.assertEqual(result["metrics"]["distinct_code_accepted_realized_designs"], 0)

    def test_the_proposal_policy_joins_the_basis_and_a_change_refuses_the_resume(self):
        basis = {"fixture": "priced", "seed": self.seed.identity()}
        legacy = replace(self.CONFIG, guided_selection=ops.UNIFORM, beam_jump_policy=jumps.LOCAL, max_evaluations=1)
        first = self.controller("r", legacy, PricedEvaluator(self.cfg), basis=basis)
        self.assertEqual(first.basis_id, dg.digest({**basis, "proposal_policy": gen.proposal_policy(legacy, self.cfg)}))
        first.start().run()
        changes = (replace(legacy, guided_selection=ops.SEVERITY_WEIGHTED), replace(legacy, beam_jump_policy=jumps.DEMAND_SIZED),
                   replace(legacy, guided_selection=ops.SEVERITY_WEIGHTED, severity_floor=0.3))
        ids = {first.basis_id}
        for config in changes:
            other = self.controller("r", config, PricedEvaluator(self.cfg), basis=basis)
            ids.add(other.basis_id)
            with self.assertRaises(ValueError):
                other.resume()
        self.assertEqual(len(ids), 1 + len(changes))
        stricter = self.controller("r", legacy, PricedEvaluator(self.cfg), cfg=with_beam_practical_ratio(DesignConfig(), 0.015),
                                   basis=basis)
        self.assertNotEqual(stricter.basis_id, first.basis_id)
        with self.assertRaises(ValueError):
            stricter.resume()
        # The same configuration resumes; a basis that names another proposal policy is refused at once.
        self.controller("r", legacy, PricedEvaluator(self.cfg), basis=basis).resume().close()
        with self.assertRaises(ValueError):
            self.controller("s", legacy, PricedEvaluator(self.cfg),
                            basis={**basis, "proposal_policy": gen.proposal_policy(self.CONFIG, self.cfg)})

    def test_the_new_configuration_fields_are_validated(self):
        for change in ({"guided_selection": "adaptive"}, {"beam_jump_policy": "largest"}, {"severity_floor": 0.0},
                       {"severity_floor": 1.5}, {"severity_cap": 0.0}, {"severity_cap": math.inf},
                       {"categorical_weight": 0.1}, {"beam_jump_target_dcr": 1.2}, {"beam_jump_target_dcr": 0.0},
                       {"beam_jump_max_sections": 1}, {"beam_jump_max_sections": 2.0}, {"beam_jump_max_seconds": 0.0},
                       {"beam_jump_partial_tolerance": -0.1}, {"beam_jump_min_gain": math.nan}, {"severity_floor": True}):
            with self.assertRaises(ValueError, msg=repr(change)):
                replace(self.CONFIG, **change).validate()
        gen.GenerationConfig().validate()
        self.assertEqual((gen.GenerationConfig().guided_selection, gen.GenerationConfig().beam_jump_policy),
                         (ops.UNIFORM, jumps.LOCAL))                     # the earlier rules stay the default


if __name__ == "__main__":
    unittest.main()
