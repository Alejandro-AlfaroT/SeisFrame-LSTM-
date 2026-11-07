"""Corrections from the review of the cage coordination (2026-10-31): swept bend intervals, transition attribution, the
content-keyed coordination memo and the joint inventory's group ids.

  F1  a moved column bar blocks the whole lateral interval of its bend (joint_inventory_lanes_v2__swept_bend_intervals):
      the review's counterexample (32 in #11 columns going from twelve to eight bars, a 22 x 36 in beam with twelve #4
      per face) has five lanes, not the six the two ends alone gave, and the authoritative threading row fails; every
      lane returned clears the moving bar at every elevation of its bend, while one lane of the two-end rule does not;
      a cage with no moved bar keeps the core cage's lanes and a plain point inventory reproduces them; ordinary
      supported steps keep their lanes; a transition that declares no bar paths claims no lane (fails closed);
  F2  a transition joint's blockage is named by both column groups and kept apart from a core cage's own; the
      look-ahead resizes the column above at the joint's own inventory (28 to 30 in opens three lanes, growing the
      core leaves none), a core-only look-ahead says it is core-only, and the search offers the upper band's size
      (lever transition_lanes) only where the topology has that joint;
  F3  the failed-coordination memo is keyed by content: equal copies hit, and a nested setting, an action, the
      aggregate, a cover, the crossing gap or fyt changed in place is another key and runs the coordination again;
  F4  equal designs at a corner, an edge and an interior joint share one cached geometry and each reports its own groups;
  R   (follow-up review) the real failed beam.bars_thread_column row of the 32-below/28-above transition, through the
      contract assessment, the diagnostic adapter and the router, offers the upper band 28 -> 30 in and withholds the
      lower core's 32 -> 34 in (its own cage admits the bars); with no or unauthenticated installed-cage lane evidence it
      proposes nothing for the transition and says why; a row whose core cage blocks keeps the core's growth; a real
      lane-limited shortfall (the selector's own evidence) grows only the cores whose cage blocks, at the look-ahead size.

Synthetic geometry and selection on the fixture frames, no analysis: nothing here is a designed or accepted structure.
"""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genetic_fixture as gfx                                                    # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402
import test_beam_threading_evidence as tb                                        # noqa: E402
import test_joint_inventory_lanes as tj                                          # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Code_Check_Contract as contract                               # noqa: E402
from Design import Genetic_Diagnostics as dg                                     # noqa: E402
from Design import Genetic_Operators as ops                                      # noqa: E402
from Design import Genetic_Search as ga                                          # noqa: E402
from Design import Group_Capacity as gc                                          # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design import SMRF_Cage_Geometry as geometry                                # noqa: E402
from Design import SMRF_Transitions as transitions                               # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Model import Member_Bar_Layers as bar_layers                                # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model import Member_Properties as mp                                        # noqa: E402

# The review's counterexample (SWEPT_LANE_REPRO.json): both cages and the beam are in the selector's domains.
CX_LOWER = replace(gf.COLUMN, b_in=32.0, h_in=32.0, fc_ksi=5.0, bar_size=11, top_bars=4, bot_bars=4, side_bars=2)
CX_UPPER = replace(CX_LOWER, top_bars=3, bot_bars=3, side_bars=1)
CX_BEAM = replace(tj.BEAM, fc_ksi=4.0, bar_size=4, top_bars=12, bot_bars=12)
UPPER_CORNER = "s04_06__column__corner"
LOWER_CORNER = "s01_03__column__corner"
TIE_X, TIE_Y = "s01_03__beam_x__edge", "s01_03__beam_y__edge"


def ga_two_bands(upper_size=28.0):
    """tj.two_bands with the interior (gravity) columns at 6 ksi, inside the search's gravity grade ladder; the
    perimeter transition is unchanged."""
    state = tj.two_bands(upper_size=upper_size)
    return state.with_designs({g: replace(d, fc_ksi=6.0) for g, d in state.designs.items() if "__column__interior" in g})


def through_the_router(state, core_group, level, axis, constraints):
    """The real failed threading row (Group_Capacity.joint_threading) of the joint type at ``level`` whose core is
    ``core_group``, put into a contract-valid record (its other rows synthetic and passing), then assessed, diagnosed
    and routed. Returns (row, assessed, diagnostic, route, ctx)."""
    kinds = [kind["representative"] for kind in gc.joint_inventory(state)["types"].values()]
    entry = next(e for e in kinds if e["level"] == level and e["core"].group_id == core_group)
    row = gc.joint_threading(entry, axis)
    evidence = gfx.capacity_evidence(state)
    evidence["joint_types"][entry["type_id"]]["threading"][axis] = row
    key = ("capacity", "beam.bars_thread_column", f"{entry['type_id']}/{axis}")
    record = gfx.record(state, evidence=evidence, replace={key: gfx.row(*key, int(row["passes"]), 1, "==", details=row)},
                        constraints=constraints)
    assessed = contract.assess_evaluation(record, submitted_identity=state.identity(), submitted_state=state)
    ctx = dg.parent_context(ga.DesignSpace(state), state, basis_identity="routing-regression",
                            realized_identity=record["code_checks"]["realized_identity"], capacity=evidence)
    (diagnostic,) = [d for d in dg.diagnose(record, assessed, ctx)["diagnostics"] if d["reason"] == "beam_threading"]
    return row, assessed, diagnostic, ops.route(ctx, diagnostic), ctx


def moves(route):
    return sorted((p["requested_gene"], tuple(p["from"]), tuple(p["to"]),
                   tuple(sorted((o["lever"], o["target_group"]) for o in p["origins"]))) for p in route["proposals"])


def counterexample_state():
    state = mg.GroupedDesign.uniform(gf.NX, gf.NY, 6, CX_LOWER, CX_BEAM, policy_id=tj.POLICY)
    return state.with_designs({g: CX_UPPER for g in state.designs if g.startswith("s04_06__column__")})


def lanes_of(inventory, column=CX_LOWER, beam=CX_BEAM, axis="x"):
    db = sp.rebar_diameter(beam.bar_size)
    return geometry.beam_bars_through_inventory(
        column.b_in, column.h_in, beam.b_in, sp.BEAM_CLEAR_COVER_IN, sp.rebar_diameter(beam.stirrup_bar_size), db,
        beam.top_bars, inventory, geometry.clear_spacing_limit("beam", db, sp.AGGREGATE_MAX_SIZE_IN))[axis]


def nearest_approach(lanes, path, column, column_db, beam_db, clear, samples=200):
    """The smallest (centre distance - required separation) between any lane and the bar on ``path`` over its bend
    (X beams: the lateral coordinate is the transition's x plus half of b)."""
    a, b = path["lower_xy_in"][0] + 0.5 * column.b_in, path["target_xy_in"][0] + 0.5 * column.b_in
    gap = 0.5 * (column_db + beam_db) + clear
    return min(abs(p - (a + (b - a) * k / samples)) - gap for p in lanes for k in range(samples + 1))


class SweptBendIntervals(unittest.TestCase):
    def test_the_review_counterexample_has_five_lanes_and_the_threading_row_fails(self):
        with gf.frame(**tj.SIX):
            transition = transitions.column_transition(CX_LOWER, CX_UPPER, fy_ksi=60.0, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                                                       upper_clear_height_in=132.0, joint_depth_in=36.0)
            self.assertTrue(transition["supported"])
            moved = [p for p in transition["paths"] if p["bent"]]
            self.assertTrue(moved)
            col_db = sp.rebar_diameter(CX_LOWER.bar_size)
            # The two-end rule of v1, rebuilt here: every lower bar and each moved bar's top position as points.
            ends = [(b["y_in"] + 16.0, b["x_in"] + 16.0, col_db)
                    for b in transitions.bar_coordinates(CX_LOWER, sp.COL_CLEAR_COVER_IN)]
            ends += [(p["target_xy_in"][1] + 16.0, p["target_xy_in"][0] + 16.0, col_db) for p in moved]
            two_ends = lanes_of(ends)
            swept = lanes_of(bar_layers.transition_inventory(CX_LOWER, CX_UPPER, transition))
            self.assertEqual((two_ends["bars_per_layer_that_fit"], swept["bars_per_layer_that_fit"]), (6, 5))
            self.assertIn(18.0, two_ends["threaded_positions_in"])
            self.assertGreater(swept["swept_intervals"], 0)
            beam_db = sp.rebar_diameter(CX_BEAM.bar_size)
            clear = swept["clearance_used_in"]
            # A two-end lane meets the moving bar somewhere on its bend; no swept lane does, at any elevation.
            self.assertLess(min(nearest_approach(two_ends["threaded_positions_in"], p, CX_LOWER, col_db, beam_db, clear)
                                for p in moved), -0.5)
            self.assertGreaterEqual(min(nearest_approach(swept["threaded_positions_in"], p, CX_LOWER, col_db, beam_db, clear)
                                        for p in moved), -1e-9)
            state = mg.install(counterexample_state())
            kinds = [kind["representative"] for kind in gc.joint_inventory(state)["types"].values()]
            boundary = next(e for e in kinds if e["level"] == 3 and e["core"].group_id == "s01_03__column__corner")
            row = gc.joint_threading(boundary, "x")
            self.assertEqual((row["bars_per_layer_that_fit"], row["bars_per_layer_core_cage"], row["layers_needed"],
                              row["passes"]), (5, 8, 3, False))
            self.assertEqual((row["core_group"], row["above_group"]), ("s01_03__column__corner", UPPER_CORNER))
            self.assertIn(bar_layers.JOINT_INVENTORY_RULE_ID, row["lane_basis"])
            recorded = bar_layers.arrangement()["_basis"]["joint_inventory"]
            self.assertEqual(recorded["rule"], "joint_inventory_lanes_v2__swept_bend_intervals")
            self.assertTrue(all(e["inventory_established"] and e["above_group"].startswith("s04_06__")
                                for e in recorded["joints_with_fewer_lanes"]))

    def test_a_point_inventory_reproduces_the_core_cage_and_unmoved_cages_keep_it(self):
        with gf.frame(**tj.SIX):
            plan = geometry.column_bars_plan(tj.LOWER.b_in, tj.LOWER.h_in, sp.COL_CLEAR_COVER_IN,
                                             sp.rebar_diameter(tj.LOWER.stirrup_bar_size), sp.rebar_diameter(tj.LOWER.bar_size),
                                             tj.LOWER.top_bars, tj.LOWER.side_bars)
            db = sp.rebar_diameter(tj.LOWER.bar_size)
            for axis in ("x", "y"):
                core = bar_layers.bars_per_layer(tj.LOWER, tj.BEAM, axis)
                self.assertEqual(lanes_of([(x, y, db) for x, y in plan], tj.LOWER, tj.BEAM, axis)["bars_per_layer_that_fit"], core)
                self.assertEqual(lanes_of([(x, y, x, y, db) for x, y in plan], tj.LOWER, tj.BEAM, axis)["bars_per_layer_that_fit"],
                                 core)
            same = transitions.column_transition(tj.LOWER, tj.LOWER, fy_ksi=60.0, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                                                 upper_clear_height_in=153.0, joint_depth_in=36.0)
            self.assertIsNone(bar_layers.transition_inventory(tj.LOWER, tj.LOWER, same))
            mg.install(mg.GroupedDesign.uniform(gf.NX, gf.NY, 6, tj.LOWER, tj.BEAM, policy_id=tj.POLICY))
            self.assertNotIn("joint_inventory", bar_layers.arrangement()["_basis"])

    def test_ordinary_supported_steps_keep_their_lanes(self):
        with gf.frame():
            for size, expected in ((31.0, 3), (30.0, 3), (28.0, 0)):
                upper = replace(tj.LOWER, b_in=size, h_in=size)
                result = transitions.column_transition(tj.LOWER, upper, fy_ksi=60.0, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                                                       upper_clear_height_in=153.0, joint_depth_in=36.0)
                self.assertTrue(result["supported"], size)
                bars = bar_layers.transition_inventory(tj.LOWER, upper, result)
                for axis in ("x", "y"):
                    direction = lanes_of(bars, tj.LOWER, tj.BEAM, axis)
                    self.assertEqual(direction["bars_per_layer_that_fit"], expected, (size, axis))
                    self.assertGreater(direction["swept_intervals"], 0)

    def test_a_transition_without_declared_bar_paths_claims_no_lane(self):
        with gf.frame(**tj.SIX):
            larger_bars = replace(tj.LOWER, b_in=28.0, h_in=28.0, bar_size=11)
            refused = transitions.column_transition(tj.LOWER, larger_bars, fy_ksi=60.0, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                                                    upper_clear_height_in=153.0, joint_depth_in=36.0)
            self.assertEqual((refused["supported"], refused["paths"]), (False, []))
            bars = bar_layers.transition_inventory(tj.LOWER, larger_bars, refused)
            self.assertEqual(bars, bar_layers.INVENTORY_NOT_ESTABLISHED)
            self.assertEqual(bar_layers.inventory_lanes(bars, tj.LOWER, tj.BEAM, "x"), 0)
            base = mg.GroupedDesign.uniform(gf.NX, gf.NY, 6, tj.LOWER, tj.BEAM, policy_id=tj.POLICY)
            state = mg.install(base.with_designs({g: replace(larger_bars, member_type="column") for g in base.groups
                                                  if g.startswith("s04_06__column__")}))
            inventory = bar_layers.joint_inventory(3, 0, 0)
            self.assertFalse(inventory["established"])
            self.assertEqual(bar_layers.bars_per_layer_at_joint(3, 0, 0, tj.BEAM, "x"), 0)
            kinds = [kind["representative"] for kind in gc.joint_inventory(state)["types"].values()]
            boundary = next(e for e in kinds if e["level"] == 3 and e["core"].group_id == "s01_03__column__corner")
            row = gc.joint_threading(boundary, "x")
            self.assertFalse(row["passes"])
            self.assertTrue(row["lane_basis"].startswith("not established"))
            recorded = bar_layers.arrangement()["_basis"]["joint_inventory"]["joints_with_fewer_lanes"]
            self.assertTrue(recorded and not any(e["inventory_established"] for e in recorded))


class InventoryGroupIds(unittest.TestCase):
    def test_equal_designs_share_the_geometry_and_each_joint_reports_its_own_groups(self):
        with gf.frame(**tj.SIX):
            mg.install(tj.two_bands())
            bar_layers._CACHE.clear()
            for joint in ((3, 0, 0), (3, 1, 0), (3, 1, 1)):
                inventory = bar_layers.joint_inventory(*joint)
                self.assertEqual((inventory["core_group"], inventory["above_group"]),
                                 (mp.joint_core_column(*joint).group_id, mg.joint_members(*joint)["column_above"].group_id))
            self.assertEqual(sum(1 for key in bar_layers._CACHE if key[0] == "inventory"), 1)
            cached = next(value for key, value in bar_layers._CACHE.items() if key[0] == "inventory")
            self.assertFalse({"core_group", "above_group"} & set(cached))


class TransitionAttribution(unittest.TestCase):
    GID = "s01_03__beam_x__edge"

    def test_the_lane_evidence_names_both_column_groups_and_keeps_the_core_cage_apart(self):
        with gf.frame(**tj.SIX):
            state = mg.install(tj.two_bands())
            lanes = selection._lane_governors(state, (self.GID,), state.designs[self.GID])[self.GID]
            self.assertEqual(lanes["bars_per_layer"], 0)
            self.assertTrue(all(v == 3 for v in lanes["core_cage_lanes"].values()))
            name = "s01_03__column__corner>s04_06__column__corner"
            self.assertIn(name, lanes["governing_transitions"])
            entry = lanes["transitions"][name]
            self.assertEqual((entry["joint_lanes"], entry["core_cage_lanes"], entry["established"]), (0, 3, True))
            cores, cages, pairs = selection._blocking({self.GID: lanes}, (self.GID,), state.designs[self.GID])
            self.assertEqual(cages, [])                                              # the core cages admit the cage
            self.assertIn({"core_group": "s01_03__column__corner", "above_group": "s04_06__column__corner"}, pairs)
            self.assertTrue(set(cores) == {p["core_group"] for p in pairs})

    def test_the_lookahead_resizes_the_band_that_opens_the_joint(self):
        with gf.frame(**tj.SIX):
            state = mg.install(tj.two_bands())
            look = selection._lane_lookahead(state, (self.GID,), state.designs[self.GID], DesignConfig())
            self.assertFalse([e for e in look.values() if e["scope"] == "core_cage"])     # no core-only claim at 32 in
            entry = look["s01_03__column__corner>s04_06__column__corner/x"]
            self.assertEqual((entry["scope"], entry["current_size_in"], entry["current_above_size_in"]),
                             ("transition_joint", 32.0, 28.0))
            self.assertEqual((entry["first_upper_band_size"], entry["first_lower_band_size"]), (30.0, None))
            first = entry["upper_band"][0]
            self.assertEqual((first["size_in"], first["transition_supported"], first["joint_lanes"]), (30.0, True, 3))
            self.assertTrue(all(not r["transition_supported"] and r["joint_lanes"] == 0 for r in entry["lower_band"]))
            self.assertTrue(entry["coupled"])                                      # evidence only: two bands change

    def test_a_core_only_lookahead_says_so_and_names_its_transition_partner(self):
        with gf.frame(**tj.SIX):
            state = mg.install(tj.two_bands(beam=replace(tj.BEAM, top_bars=8, bot_bars=8)))
            look = selection._lane_lookahead(state, (self.GID,), state.designs[self.GID], DesignConfig())
            core = look["s01_03__column__corner/x"]
            self.assertEqual(core["scope"], "core_cage")
            self.assertEqual(core["transition_partners"], ["s04_06__column__corner"])
            self.assertIn("not a transition joint", core["basis"])
            self.assertIn("s01_03__column__corner>s04_06__column__corner/x", look)

    def evidence(self, core, upper, size, **changes):
        record = {"authenticated": True, "threading_constrained_deficit": True, "constrains_strength": True,
                  "deficit_remains": True, "blocking_cores": [core], "blocking_core_cages": [],
                  "blocking_transitions": [{"core_group": core, "above_group": upper}],
                  "lookahead": {f"{core}>{upper}/x": {"scope": "transition_joint", "core_group": core, "above_group": upper,
                                                      "current_size_in": 26.0, "current_above_size_in": 24.0,
                                                      "first_upper_band_size": size, "first_lower_band_size": None}}}
        record.update(changes)
        return {"threading_constrained_beam_groups": [self.GID], "beam_threading_evidence": {self.GID: record}}

    def test_the_diagnostic_reads_targets_by_scope(self):
        core, upper = "s01_03__column__corner", "s04_04__column__corner"
        _ok, _problems, fields = dg.threading_constraint_evidence(self.evidence(core, upper, 26.0), None, self.GID)
        self.assertEqual(fields["transition_lookahead"], [{"group": upper, "band": "upper_band", "core_group": core,
                                                           "above_group": upper, "target": [26.0, 26.0]}])
        self.assertEqual((fields["lane_lookahead"], fields["blocking_core_cages"]), ({}, []))
        unscoped = {f"{core}/x": {"core_group": core, "current_size_in": 26.0, "first_size_same_cage": 30.0}}
        _ok, _problems, fields = dg.threading_constraint_evidence(self.evidence(core, upper, 26.0, lookahead=unscoped),
                                                                  None, self.GID)
        self.assertEqual((fields["lane_lookahead"], fields["transition_lookahead"]), ({}, []))

    def test_the_search_offers_the_upper_band_only_where_the_two_groups_meet(self):
        with gf.frame():
            seed = gfx.seed_state()
            ctx = dg.parent_context(ga.DesignSpace(seed), seed, basis_identity="fixture-basis", realized_identity="parent-A",
                                    capacity=gfx.capacity_evidence(seed))
            core, upper = "s01_03__column__corner", "s04_04__column__corner"
            index = next(i for i, g in enumerate(ctx["genes"]) if g["kind"] == "size" and upper in g["groups"])
            old = tuple(ctx["genes"][index]["options"][ctx["genome"][index]])
            target = min(tuple(o) for o in ctx["genes"][index]["options"] if o[0] == o[1] and o[0] > old[0])
            row = {"group": upper, "band": "upper_band", "core_group": core, "above_group": upper, "target": list(target)}
            deps = dg.RULES["beam_threading_constrained_strength"][1]
            d = {"id": "synthetic:transition", "reason": "beam_threading_constrained_strength", "status": "fail",
                 "applicable": True, "evidence_valid": True, "parent_identity": ctx["parent_identity"],
                 "basis_identity": ctx["basis_identity"], "dependencies": {k: True for k in deps}, "demand": 2.0,
                 "capacity": 1.0, "comparison": "<=", "units": "synthetic_test_units", "synthetic": True,
                 "group_id": self.GID, "joint_topology_consistent": True, "blocking_cores": [core],
                 "blocking_core_cages": [], "lane_lookahead": {}, "transition_lookahead": [row]}
            routed = ops.route(ctx, d)
            levers = {(o["lever"], o["target_group"]) for p in routed["proposals"] for o in p["origins"]}
            self.assertIn(("transition_lanes", upper), levers)
            self.assertNotIn(("column_up", core), levers)            # the core cage itself does not block
            (jump,) = [p for p in routed["proposals"] if any(o["lever"] == "transition_lanes" for o in p["origins"])]
            self.assertEqual((tuple(jump["to"]), jump["origins"][0]["transition"]["above_group"]), (target, upper))
            self.assertEqual(sum(a != b for a, b in zip(ctx["genome"], jump["requested_genome"])), 1)
            self.assertTrue(ops._direction_valid("transition_lanes", old, target))
            # No joint of this beam where the core meets that column: withheld, not proposed.
            foreign = {**copy.deepcopy(d), "transition_lookahead": [{**row, "core_group": "s01_03__column__interior"}]}
            routed = ops.route(ctx, foreign)
            self.assertNotIn("transition_lanes", {o["lever"] for p in routed["proposals"] for o in p["origins"]})
            self.assertIn("no_transition_incidence", [w["reason"].split(":")[0] for w in routed["withheld"]])
            # An inconsistent topology withholds it too; a record that does not separate the core cages keeps column_up.
            routed = ops.route(ctx, {**copy.deepcopy(d), "joint_topology_consistent": False})
            self.assertIn(("transition_lanes", upper), {(w["lever"], w["group"]) for w in routed["withheld"]})
            legacy = {k: v for k, v in copy.deepcopy(d).items() if k != "blocking_core_cages"}
            self.assertIn(("column_up", core), {(o["lever"], o["target_group"]) for p in ops.route(ctx, legacy)["proposals"]
                                                for o in p["origins"]})


class RoutingFromTheFailedRow(unittest.TestCase):
    def test_the_real_transition_row_offers_the_upper_band_not_the_lower_core(self):
        with gf.frame(**tj.SIX):
            state = mg.install(ga_two_bands())
            lanes = selection.installed_lane_evidence(state, DesignConfig())
            self.assertIn(TIE_X, lanes["groups"])                      # the selector reports this tie "selected"
            row, assessed, d, route, ctx = through_the_router(state, LOWER_CORNER, 3, "x", {"beam_lane_evidence": lanes})
            self.assertEqual((row["bars_per_layer_that_fit"], row["bars_per_layer_core_cage"], row["passes"],
                              row["above_group"]), (0, 3, False, UPPER_CORNER))
            self.assertEqual((assessed["valid"], assessed["classification"]), (True, "check_failed"))
            self.assertTrue(d["evidence_valid"] and all(d["dependencies"].values()), d)
            self.assertEqual((d["core_cage_blocks"], d["transition_blocks"], d["lane_evidence"]), (False, True, "authenticated"))
            self.assertEqual(moves(route), [("s04_06__sfrs_column", (28.0, 28.0), (30.0, 30.0),
                                             (("transition_lanes", UPPER_CORNER),))])
            (proposal,) = route["proposals"]
            self.assertEqual(proposal["origins"][0]["transition"],
                             {"core_group": LOWER_CORNER, "above_group": UPPER_CORNER, "band": "upper_band"})
            self.assertEqual(sum(a != b for a, b in zip(ctx["genome"], proposal["requested_genome"])), 1)
            self.assertIn(("column_up", LOWER_CORNER, "transition_only"),
                          [(w["lever"], w["group"], w["reason"].split(":")[0]) for w in route["withheld"]])
            # Without the installed-cage lane evidence, or with evidence of another state: no target, and why.
            for constraints, status in ((None, "absent"),
                                        ({"beam_lane_evidence": {**lanes, "state_identity": "another"}}, "not_authenticated")):
                _row, _assessed, d, route, _ctx = through_the_router(state, LOWER_CORNER, 3, "x", constraints)
                self.assertEqual((d["lane_evidence"], route["proposals"]), (status, []))
                self.assertIn(("transition_lanes", UPPER_CORNER, "no_transition_target"),
                              [(w["lever"], w["group"], w["reason"].split(":")[0]) for w in route["withheld"]])

    def test_a_row_whose_core_cage_blocks_keeps_the_core_growth(self):
        with gf.frame(**tj.SIX):
            state = mg.install(ga_two_bands())
            lanes = selection.installed_lane_evidence(state, DesignConfig())
            row, _assessed, d, route, _ctx = through_the_router(state, UPPER_CORNER, 4, "x", {"beam_lane_evidence": lanes})
            self.assertEqual((row["bars_per_layer_that_fit"], row["bars_per_layer_core_cage"], row["above_group"]), (0, 0, None))
            self.assertEqual((d["core_cage_blocks"], d["transition_blocks"]), (True, False))
            levers = {(lever, group) for move in moves(route) for lever, group in move[3]}
            self.assertIn(("column_up", UPPER_CORNER), levers)
            self.assertFalse({lever for lever, _group in levers} & {"transition_lanes"})
            self.assertTrue(all(move[0] == "s04_06__sfrs_column" for move in moves(route)))
            self.assertFalse([w for w in route["withheld"] if w["reason"].startswith("transition_only")])

    def test_a_real_lane_limited_shortfall_grows_only_the_cores_whose_cage_blocks(self):
        with gf.frame(**tj.SIX):
            state = mg.install(ga_two_bands(upper_size=30.0))
            actions = tb.synthetic_actions(state, {TIE_X: (12000.0, 12000.0), TIE_Y: (12000.0, 12000.0)})
            design, unit = selection.select_beam_groups(state, (TIE_X, TIE_Y), actions, DesignConfig())
            self.assertEqual((unit["status"], unit["lane_limited"]), ("strength_short", True))
            chosen = mg.install(selection._trial_state(state, TIE_X, replace(design, member_type="beam_x")))
            summary = selection.threading_constraint_summary({"passes": [{"beams": {"s01_03__sfrs_beam": unit}}]})
            overrides = {TIE_X: {"flexure_negative": {"dcr": unit["dcr"], "mu_kip_in": 1000.0 * unit["dcr"]}}}
            listed = {"lane_limited_beam_groups": [TIE_X, TIE_Y], "threading_constrained_beam_groups": [TIE_X, TIE_Y]}
            routes = {}
            for name, constraints in (("evidence", {**listed, "beam_threading_evidence": summary}),
                                      ("earlier record", {"lane_limited_beam_groups": [TIE_X, TIE_Y]})):
                record = gfx.record(chosen, strength_overrides=overrides, constraints=constraints)
                assessed = contract.assess_evaluation(record, submitted_identity=chosen.identity(), submitted_state=chosen)
                ctx = dg.parent_context(ga.DesignSpace(chosen), chosen, basis_identity="routing-regression",
                                        realized_identity=record["code_checks"]["realized_identity"],
                                        capacity=gfx.capacity_evidence(chosen))
                (d,) = [x for x in dg.diagnose(record, assessed, ctx)["diagnostics"]
                        if x["reason"] == "beam_lane_limited_strength"]
                routes[name] = (d, ops.route(ctx, d), ctx)
            d, route, ctx = routes["evidence"]
            self.assertEqual(d["threading_evidence"], "authenticated")
            incident = sorted({t["core_group"] for t in ctx["joint_types"].values() if TIE_X in t["beams"]["x"]})
            sized = [c for c in incident if c in d["blocking_core_cages"]]
            self.assertTrue(sized)
            levers = {(lever, group) for move in moves(route) for lever, group in move[3]}
            self.assertEqual({g for lever, g in levers if lever == "column_up"}, set(sized))
            self.assertEqual({g for lever, g in levers if lever == "column_lanes"}, set(d["lane_lookahead"]) & set(sized))
            self.assertFalse({lever for lever, _g in levers} & {"grade_up", "transition_lanes"})
            self.assertIn(("width", TIE_X), levers)            # 22 x 36 in is the deepest section of its ladder here
            # An earlier record without the selection's evidence keeps the earlier routing: every core it threads.
            d, route, _ctx = routes["earlier record"]
            self.assertEqual(d["threading_evidence"], "absent")
            self.assertEqual({g for move in moves(route) for lever, g in move[3] if lever == "column_up"}, set(incident))
            # The same branch takes a transition target through the shared targeting where the evidence names one.
            d, _route, ctx = routes["evidence"]
            target = {"group": UPPER_CORNER, "band": "upper_band", "core_group": LOWER_CORNER, "above_group": UPPER_CORNER,
                      "target": [32.0, 32.0]}
            route = ops.route(ctx, {**copy.deepcopy(d), "blocking_core_cages": [], "transition_lookahead": [target]})
            levers = {(lever, group) for move in moves(route) for lever, group in move[3]}
            self.assertIn(("transition_lanes", UPPER_CORNER), levers)
            self.assertFalse({g for lever, g in levers if lever in ("column_up", "column_lanes")})


class CoordinationMemo(unittest.TestCase):
    def key(self, state, actions, cfg):
        report = {"settled": True, "passes": [{"beams": {}, "columns": {}}]}
        return selection._coordination_key(state, report, actions, cfg, 1.0, 4, None)

    def test_the_key_is_the_content_of_every_input(self):
        with gf.frame(**tb.RULES_SEISMIC):
            state = mg.install(tb.system_state())
            actions, cfg = tb.synthetic_actions(state), DesignConfig()
            base = self.key(state, actions, cfg)
            self.assertIsNotNone(base)
            self.assertEqual(self.key(state, copy.deepcopy(actions), copy.deepcopy(cfg)), base)      # equal copies hit
            cfg.rebar.rho_col_practical_max = 0.06
            self.assertNotEqual(self.key(state, actions, cfg), base)
            cfg.rebar.rho_col_practical_max = DesignConfig().rebar.rho_col_practical_max
            member = next(iter(actions[0]["members"].values()))
            member["local_force_kip_kipin"][4] += 1.0
            self.assertNotEqual(self.key(state, actions, cfg), base)
            member["local_force_kip_kipin"][4] -= 1.0
            self.assertEqual(self.key(state, actions, cfg), base)
            for name, value in (("AGGREGATE_MAX_SIZE_IN", 1.5), ("BEAM_CLEAR_COVER_IN", 2.0), ("COL_CLEAR_COVER_IN", 2.0),
                                ("BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN", 1.5), ("FYT_KSI", 80.0)):
                before = getattr(sp, name)
                setattr(sp, name, value)
                self.assertNotEqual(self.key(state, actions, cfg), base, name)
                setattr(sp, name, before)
            self.assertEqual(self.key(state, actions, cfg), base)

            class Opaque:
                pass
            self.assertIsNone(self.key(state, [*actions, Opaque()], cfg))          # no content form: no memo

    def test_a_failed_coordination_is_reused_only_for_the_same_inputs(self):
        with gf.frame(**tb.RULES_SEISMIC):
            state = mg.install(tb.system_state())
            actions = tb.synthetic_actions(state, {tb.X: (6000.0, 6000.0), tb.Y: (6000.0, 6000.0)})
            cfg = DesignConfig()
            selected, base = selection.select_reinforcement(state, actions, cfg, scwb_margin=4.0, coordinate=False)
            selection._COORDINATION_MEMO.clear()
            first = copy.deepcopy(base)
            selection.coordinate_threading(selected, actions, cfg, 4.0, 4, first)
            self.assertEqual(first["coordination"]["status"], "no_coordinated_change")
            self.assertFalse(first["coordination"].get("reused"))
            again = copy.deepcopy(base)
            selection.coordinate_threading(selected, copy.deepcopy(actions), copy.deepcopy(cfg), 4.0, 4, again)
            self.assertTrue(again["coordination"]["reused"])
            self.assertEqual(again["coordination"]["trials"], first["coordination"]["trials"])
            cfg.rebar.rho_col_practical_max = 0.06                       # the same object, changed in place
            changed = copy.deepcopy(base)
            selection.coordinate_threading(selected, actions, cfg, 4.0, 4, changed)
            self.assertFalse(changed["coordination"].get("reused"))
            self.assertTrue(changed["coordination"]["trials"])
            self.assertIs(mg.active(), state)


if __name__ == "__main__":
    unittest.main()
