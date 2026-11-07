"""The six mutation operators (Design.Genetic_Operators) on the fixture frame's real catalogs, ties and topology.

The diagnostics are explicitly synthetic (``synthetic: True``): they exercise routing, targeting, repair
validation and attribution, never a structural check or a search benefit.
"""
import copy
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genetic_fixture as gfx
import grouped_fixture as gf
from Design import Genetic_Diagnostics as dg
from Design import Genetic_Operators as ops
from Design import Genetic_Search as ga

BASIS = "fixture-basis"
BEAM, COLUMN = "s01_03__beam_x__edge", "s01_03__column__corner"


def diagnostic(ctx, reason, **fields):
    deps = dg.RULES.get(reason, (None, ()))[1]
    return {"id": "synthetic:" + reason, "reason": reason, "status": "fail", "applicable": True, "evidence_valid": True,
            "parent_identity": ctx["parent_identity"], "basis_identity": ctx["basis_identity"],
            "dependencies": {k: True for k in deps}, "demand": 2.0, "capacity": 1.0, "comparison": "<=",
            "units": "synthetic_test_units", "synthetic": True, **fields}


def all_diagnostics(ctx):
    rows = []
    for reason in dg.RULES:
        if reason == "scwb":
            fields = {"joint_id": "joint_13", "axis": "x", "sign": "positive"}
        elif reason in ("joint_shear", "hook_embedment"):
            fields = {"joint_type_id": "joint_13", "joint_id": "joint_13", "axis": "x"}
        elif reason == "through_bar_depth":
            fields = {"joint_type_id": "joint_14", "joint_id": "joint_14", "axis": "x"}
        elif reason in ("story_drift", "story_stability"):
            fields = {"story": 1, "axis": "x"}
        elif reason.startswith("column_"):
            fields = {"group_id": COLUMN}
        else:
            fields = {"group_id": BEAM}
            if reason == "beam_threading":
                fields.update(joint_type_id="joint_14", joint_id="joint_14", axis="x")
            if reason == "slab_mat_overlap":
                fields["blocked_dimension"] = "width"
            if reason == "beam_lane_limited_strength":
                fields["joint_topology_consistent"] = True
            if reason == "beam_threading_constrained_strength":
                # the corner core blocks the stronger cage; the look-ahead names its largest square size
                index = next(i for i, g in enumerate(ctx["genes"]) if g["kind"] == "size" and COLUMN in g["groups"])
                options = [tuple(o) for o in ctx["genes"][index]["options"]]
                old = options[ctx["genome"][index]]
                larger = sorted(o for o in options if o[0] == o[1] and o[0] > old[0])
                fields.update(joint_topology_consistent=True, blocking_cores=[COLUMN],
                              lane_lookahead={COLUMN: list(larger[-1])} if larger else {})
        rows.append(diagnostic(ctx, reason, **fields))
    return rows


class Operators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gf.frame():
            cls.seed = gfx.seed_state()
            cls.space = ga.DesignSpace(cls.seed)
            cls.base = dg.parent_context(cls.space, cls.seed, basis_identity=BASIS, realized_identity="parent-A",
                                         capacity=gfx.capacity_evidence(cls.seed))

    def setUp(self):
        self.fixture = gf.frame()
        self.fixture.__enter__()
        self.ctx = copy.deepcopy(self.base)
        self.rows = {d["reason"]: d for d in all_diagnostics(self.ctx)}

    def tearDown(self):
        self.fixture.__exit__(None, None, None)

    def proposals(self, reason):
        return ops.route(self.ctx, self.rows[reason])["proposals"]

    def gene(self, name):
        return next(i for i, g in enumerate(self.ctx["genes"]) if g["name"] == name)

    def test_every_reason_routes_to_its_operator(self):
        for reason, (operator, _deps) in dg.RULES.items():
            with self.subTest(reason=reason):
                result = ops.route(self.ctx, self.rows[reason])
                self.assertEqual((result["status"], result["operator"]), ("proposed", operator))

    def test_each_requested_child_changes_one_gene(self):
        for d in self.rows.values():
            for p in ops.route(self.ctx, d)["proposals"]:
                self.assertEqual(sum(a != b for a, b in zip(self.ctx["genome"], p["requested_genome"])), 1)

    def test_size_and_grade_alternatives_are_separate_proposals(self):
        for reason in ("scwb", "column_pm", "beam_flexure", "joint_shear"):
            kinds = [self.ctx["genes"][self.gene(p["requested_gene"])]["kind"] for p in self.proposals(reason)]
            self.assertIn("size", kinds)
            self.assertIn("grade", kinds)

    def test_inputs_are_unchanged(self):
        before = copy.deepcopy((self.ctx, self.rows))
        for d in self.rows.values():
            ops.route(self.ctx, d)
        self.assertEqual((self.ctx, self.rows), before)

    def test_stale_parent_or_basis_abstains(self):
        for field in ("parent_identity", "basis_identity"):
            d = copy.deepcopy(self.rows["scwb"])
            d[field] = "other"
            self.assertEqual(ops.route(self.ctx, d)["reason"], "stale_or_unbound_evidence")

    def test_unknown_evidence_and_solver_failure_never_strengthen(self):
        d = copy.deepcopy(self.rows["scwb"])
        d["status"] = "not_evaluated"
        self.assertEqual(ops.route(self.ctx, d)["reason"], "evidence_acquisition_required")
        self.assertEqual(ops.route(self.ctx, diagnostic(self.ctx, "solver_noncompletion"))["reason"],
                         "analysis_diagnosis_required")
        self.assertEqual(ops.route(self.ctx, diagnostic(self.ctx, "cage_exhausted"))["reason"],
                         "unsupported_or_ambiguous_diagnosis")

    def test_every_dependency_is_required_and_none_defaults_true(self):
        for reason, (_op, deps) in dg.RULES.items():
            for dep in deps:
                d = copy.deepcopy(self.rows[reason])
                d["dependencies"][dep] = False
                self.assertEqual(ops.route(self.ctx, d)["reason"], "trigger_dependencies_incomplete")
        d = copy.deepcopy(self.rows["scwb"])
        d["dependencies"] = {}
        self.assertEqual(ops.route(self.ctx, d)["reason"], "trigger_dependencies_incomplete")

    def test_numeric_inputs_are_validated_not_zeroed(self):
        for value in (None, True, float("nan"), float("inf"), "2"):
            d = copy.deepcopy(self.rows["beam_flexure"])
            d["demand"] = value
            self.assertEqual(ops.route(self.ctx, d)["reason"], "invalid_numeric_check")
        d = copy.deepcopy(self.rows["scwb"])
        d["demand"] = 0.5
        self.assertEqual(ops.route(self.ctx, d)["reason"], "status_contradicts_numeric_check")
        d = copy.deepcopy(self.rows["through_bar_depth"])
        d.update(demand=1.0, capacity=2.0, comparison=">=")
        self.assertEqual(ops.route(self.ctx, d)["status"], "proposed")

    def test_joint_targeting(self):
        d = copy.deepcopy(self.rows["scwb"])
        d["axis"] = "z"
        self.assertEqual(ops.route(self.ctx, d)["reason"], "missing_joint_or_axis")
        d = copy.deepcopy(self.rows["scwb"])
        del d["sign"]
        self.assertEqual(ops.route(self.ctx, d)["reason"], "missing_sway_sign")
        d = copy.deepcopy(self.rows["scwb"])
        d.update(joint_id="joint_14", axis="y")       # a gravity-system direction at a y-boundary column
        self.assertEqual(ops.route(self.ctx, d)["reason"], "not_a_required_scwb_direction")

    def test_same_tie_is_proposed_once_and_band_boundary_offers_two_ties(self):
        levers = [p["origins"][0]["lever"] for p in self.proposals("scwb")]
        self.assertEqual((levers.count("column_up"), levers.count("grade_up")), (1, 1))
        d = copy.deepcopy(self.rows["scwb"])
        d["joint_id"] = "joint_37"                    # level 3: band s01_03 below, s04_04 above
        genes = sorted(p["requested_gene"] for p in ops.route(self.ctx, d)["proposals"]
                       if p["origins"][0]["lever"] == "column_up")
        self.assertEqual(genes, ["s01_03__sfrs_column", "s04_04__sfrs_column"])

    def test_roof_joint_has_one_real_column(self):
        self.assertEqual(self.ctx["joints"]["joint_49"]["columns"], ["s04_04__column__corner"])
        d = copy.deepcopy(self.rows["scwb"])
        d["joint_id"] = "joint_49"
        self.assertEqual([p["requested_gene"] for p in ops.route(self.ctx, d)["proposals"]
                          if p["origins"][0]["lever"] == "column_up"], ["s04_04__sfrs_column"])

    def test_threading_grows_only_the_actual_core(self):
        proposals = self.proposals("beam_threading")
        self.assertEqual([p["requested_gene"] for p in proposals], ["s01_03__sfrs_column"])
        self.assertEqual(proposals[0]["origins"][0]["target_group"], "s01_03__column__y_boundary_edge")
        d = copy.deepcopy(self.rows["beam_threading"])
        d["group_id"] = "s01_03__beam_x__interior"
        self.assertEqual(ops.route(self.ctx, d)["reason"], "beam_not_incident_to_joint_axis")
        self.ctx["joint_types"]["joint_14"]["core_group"] = None
        self.assertEqual(ops.route(self.ctx, self.rows["beam_threading"])["reason"], "missing_physical_core")

    def test_width_and_depth_levers_never_shrink_the_other_dimension(self):
        for p in self.proposals("beam_width_room"):
            self.assertEqual(p["origins"][0]["lever"], "width")
            self.assertGreater(p["to"][0], p["from"][0])
            self.assertGreaterEqual(p["to"][1], p["from"][1])
        d = copy.deepcopy(self.rows["beam_width_room"])
        d["reason"] = "beam_row_depth"
        for p in ops.route(self.ctx, d)["proposals"]:
            self.assertGreater(p["to"][1], p["from"][1])
            self.assertGreaterEqual(p["to"][0], p["from"][0])

    def test_frontier_keeps_every_minimal_alternative(self):
        options = [tuple(o) for o in self.ctx["genes"][self.gene("s01_03__sfrs_beam")]["options"]]
        old = options[self.ctx["genome"][self.gene("s01_03__sfrs_beam")]]
        eligible = [x for x in options if x[1] > old[1] and x[0] >= old[0]]
        minimal = sorted(x for x in eligible if not any(y != x and y[0] <= x[0] and y[1] <= x[1] for y in eligible))
        self.assertEqual(sorted(ops.frontier(options, old, "depth")), minimal)
        self.assertEqual(ops.frontier([(16, 24), (16, 30), (18, 26), (20, 30)], (16, 24), "depth"), [(16, 30), (18, 26)])

    def test_slab_overlap_requires_a_diagnosed_dimension(self):
        d = copy.deepcopy(self.rows["slab_mat_overlap"])
        del d["blocked_dimension"]
        self.assertEqual(ops.route(self.ctx, d)["reason"], "ambiguous_geometric_bottleneck")

    def test_wrong_member_type_abstains(self):
        d = copy.deepcopy(self.rows["beam_flexure"])
        d["group_id"] = COLUMN
        self.assertEqual(ops.route(self.ctx, d)["reason"], "wrong_member_type")

    def test_drift_targets_the_sfrs_band_and_keeps_the_xy_beam_tie(self):
        proposals = self.proposals("story_drift")
        self.assertTrue(proposals)
        for p in proposals:
            self.assertNotIn("gravity", p["requested_gene"])
            for gid in p["requested_scope_groups"]:
                self.assertIn(1, self.ctx["groups"][gid]["stories_or_floors"])
        beams = [p for p in proposals if "__sfrs_beam" in p["requested_gene"]]
        self.assertEqual({self.ctx["groups"][g]["member_type"] for g in beams[0]["requested_scope_groups"]},
                         {"beam_x", "beam_y"})
        d = copy.deepcopy(self.rows["story_drift"])
        d["story"] = 99
        self.assertEqual(ops.route(self.ctx, d)["reason"], "no_eligible_sfrs_band")

    def test_band_bottom_story_drift_includes_the_beams_below_it(self):
        # Story 4 spans floors 3 and 4: the floor-3 beams belong to the lower band (s01_03), the rest to s04_04.
        d = copy.deepcopy(self.rows["story_drift"])
        d["story"] = 4
        genes = {p["requested_gene"] for p in ops.route(self.ctx, d)["proposals"]}
        self.assertEqual(genes, {"s01_03__sfrs_beam", "s04_04__sfrs_beam", "s04_04__sfrs_column"})
        d["story"] = 1                                        # floor 0 has no beams
        genes = {p["requested_gene"] for p in ops.route(self.ctx, d)["proposals"]}
        self.assertEqual(genes, {"s01_03__sfrs_beam", "s01_03__sfrs_column"})
        d["story"] = 3                                        # the band's top story: floors 2 and 3, never floor 4
        targets = {o["target_group"] for p in ops.route(self.ctx, d)["proposals"] for o in p["origins"]}
        self.assertEqual({t.split("__")[0] for t in targets}, {"s01_03"})
        self.assertTrue(any("__beam_x" in t for t in targets) and any("__column" in t for t in targets), targets)

    def test_lane_limited_strength_offers_bar_room_levers_never_a_grade(self):
        routed = ops.route(self.ctx, self.rows["beam_lane_limited_strength"])
        self.assertEqual(routed["operator"], "H3")
        kinds = {self.ctx["genes"][self.gene(p["requested_gene"])]["kind"] for p in routed["proposals"]}
        self.assertEqual(kinds, {"size"})
        cores = {o["target_group"] for p in routed["proposals"] for o in p["origins"] if o["lever"] == "column_up"}
        self.assertEqual(cores, {t["core_group"] for t in self.ctx["joint_types"].values() if BEAM in t["beams"]["x"]})
        self.assertTrue(cores)
        # The cores come from the joint topology: without a consistent one only the beam's own levers remain, and
        # each core's withheld step is counted once.
        for consistent in (False, None):
            d = {**copy.deepcopy(self.rows["beam_lane_limited_strength"]), "joint_topology_consistent": consistent}
            routed = ops.route(self.ctx, d)
            self.assertEqual({o["lever"] for p in routed["proposals"] for o in p["origins"]}, {"depth", "width"})
            self.assertEqual(sorted((w["lever"], w["group"], w["reason"].split(":")[0]) for w in routed["withheld"]),
                             sorted(("column_up", core, "joint_topology_inconsistent") for core in cores))
            counted = ops.guided_alternatives(self.ctx, [d])
            self.assertEqual(counted["withheld"], {"column_up:joint_topology_inconsistent": len(cores)})
            self.assertEqual(counted["abstentions"], {})
        # A core already at the largest square size has no step to withhold: nothing is counted for it.
        d = {**copy.deepcopy(self.rows["beam_lane_limited_strength"]), "joint_topology_consistent": False}
        capped = copy.deepcopy(self.ctx)
        for core in cores:
            index = next(i for i, g in enumerate(capped["genes"]) if g["kind"] == "size" and core in g["groups"])
            options = capped["genes"][index]["options"]
            capped["genome"][index] = max((i for i, o in enumerate(options) if o[0] == o[1]), key=lambda i: options[i][0])
        self.assertEqual(ops.route(capped, d)["withheld"], [])
        # No joint the beam frames into: the column lever has no target, recorded as withheld for that reason.
        bare = copy.deepcopy(self.ctx)
        for kind in bare["joint_types"].values():
            kind["beams"]["x"] = [b for b in kind["beams"]["x"] if b != BEAM]
        d = {**copy.deepcopy(self.rows["beam_lane_limited_strength"]), "joint_topology_consistent": True}
        routed = ops.route(bare, d)
        self.assertEqual({o["lever"] for p in routed["proposals"] for o in p["origins"]}, {"depth", "width"})
        self.assertEqual(ops.guided_alternatives(bare, [d])["withheld"], {"column_up:no_joint_incidence": 1})
        # An inconsistent topology whose named cores are not groups of the parent: the inconsistency is the reason,
        # not a missing incidence.
        foreign = copy.deepcopy(self.ctx)
        for kind in foreign["joint_types"].values():
            kind["core_group"] = "not_a_group"
        d = {**copy.deepcopy(self.rows["beam_lane_limited_strength"]), "joint_topology_consistent": False}
        self.assertEqual([(w["group"], w["reason"].split(":")[0]) for w in ops.route(foreign, d)["withheld"]],
                         [(None, "joint_topology_inconsistent")])
        self.assertEqual(ops.guided_alternatives(foreign, [d])["withheld"], {"column_up:joint_topology_inconsistent": 1})

    def test_threading_constrained_strength_offers_beam_levers_blocking_cores_and_the_lookahead_size(self):
        d = self.rows["beam_threading_constrained_strength"]
        routed = ops.route(self.ctx, d)
        self.assertEqual((routed["status"], routed["operator"]), ("proposed", "H3"))
        levers = {(o["lever"], o["target_group"]) for p in routed["proposals"] for o in p["origins"]}
        self.assertTrue({("depth", BEAM), ("width", BEAM), ("grade_up", BEAM), ("column_up", COLUMN),
                         ("column_lanes", COLUMN)} <= levers, levers)
        jumps = [tuple(p["to"]) for p in routed["proposals"] if any(o["lever"] == "column_lanes" for o in p["origins"])]
        self.assertEqual(jumps, [tuple(d["lane_lookahead"][COLUMN])])
        # A look-ahead size that is not a legal larger square option of the core's gene proposes nothing.
        for size in ([999.0, 999.0], [1.0, 1.0], [30.0, 32.0]):
            bad = {**copy.deepcopy(d), "lane_lookahead": {COLUMN: size}}
            self.assertNotIn("column_lanes", {o["lever"] for p in ops.route(self.ctx, bad)["proposals"] for o in p["origins"]})
        # An inconsistent topology withholds the core levers (one withheld step per core); the beam's own levers stay.
        bad = {**copy.deepcopy(d), "joint_topology_consistent": False}
        routed = ops.route(self.ctx, bad)
        self.assertEqual({o["lever"] for p in routed["proposals"] for o in p["origins"]}, {"depth", "width", "grade_up"})
        self.assertEqual([(w["lever"], w["group"]) for w in routed["withheld"]], [("column_up", COLUMN)])
        # A named core that is not the core of a joint the beam frames into is no target.
        incident = {t["core_group"] for t in self.ctx["joint_types"].values() if BEAM in t["beams"]["x"]}
        foreign = next(g for g, row in sorted(self.ctx["groups"].items()) if row["member_type"] == "column" and g not in incident)
        routed = ops.route(self.ctx, {**copy.deepcopy(d), "blocking_cores": [foreign]})
        self.assertFalse({o["lever"] for p in routed["proposals"] for o in p["origins"]} & {"column_up", "column_lanes"})
        self.assertEqual([w["reason"].split(":")[0] for w in routed["withheld"]], ["no_joint_incidence"])
        # The repair keeps a column_lanes request only in its direction (a larger square size).
        self.assertTrue(ops._direction_valid("column_lanes", (26.0, 26.0), (30.0, 30.0)))
        self.assertFalse(ops._direction_valid("column_lanes", (26.0, 26.0), (24.0, 24.0)))
        self.assertFalse(ops._direction_valid("column_lanes", (26.0, 26.0), (28.0, 30.0)))

    def test_column_capacity_shear_withholds_the_size_step_at_the_ratio_floor(self):
        for floor, levers in ((True, {"grade_up"}), (False, {"column_up", "grade_up"}), (None, {"column_up", "grade_up"})):
            d = {**copy.deepcopy(self.rows["column_capacity_shear"]), "cage_at_ratio_floor": floor}
            routed = ops.route(self.ctx, d)
            self.assertEqual({o["lever"] for p in routed["proposals"] for o in p["origins"]}, levers)
            self.assertEqual([w["lever"] for w in routed["withheld"]], ["column_up"] if floor else [])
            self.assertEqual(ops.guided_alternatives(self.ctx, [d])["withheld"],
                             {"column_up:cage_at_minimum_ratio": 1} if floor else {})
        # The withholding is specific to capacity shear: an interaction failure keeps its size step.
        d = {**copy.deepcopy(self.rows["column_pm"]), "cage_at_ratio_floor": True}
        self.assertIn("column_up", {o["lever"] for p in ops.route(self.ctx, d)["proposals"] for o in p["origins"]})
        # At the top grade the floor leaves no legal move: the route abstains and the withheld step is still counted.
        d = {**copy.deepcopy(self.rows["column_capacity_shear"]), "cage_at_ratio_floor": True}
        top = copy.deepcopy(self.ctx)
        grade = next(i for i, g in enumerate(top["genes"]) if g["kind"] == "grade" and d["group_id"] in g["groups"])
        top["genome"][grade] = max(range(len(top["genes"][grade]["options"])), key=lambda i: top["genes"][grade]["options"][i])
        self.assertEqual(ops.route(top, d)["reason"], "no_legal_move")
        counted = ops.guided_alternatives(top, [d])
        self.assertEqual((counted["withheld"], counted["abstentions"]),
                         ({"column_up:cage_at_minimum_ratio": 1}, {"no_legal_move": 1}))
        # A column already at the largest square size: no step exists, so none is withheld.
        size = next(i for i, g in enumerate(top["genes"]) if g["kind"] == "size" and d["group_id"] in g["groups"])
        options = top["genes"][size]["options"]
        top["genome"][size] = max((i for i, o in enumerate(options) if o[0] == o[1]), key=lambda i: options[i][0])
        self.assertEqual(ops.guided_alternatives(top, [d])["withheld"], {})

    def test_capped_catalog_abstains(self):
        self.ctx["genome"] = [len(g["options"]) - 1 for g in self.ctx["genes"]]
        self.assertEqual(ops.route(self.ctx, diagnostic(self.ctx, "scwb", joint_id="joint_13", axis="x",
                                                        sign="positive"))["reason"], "no_legal_move")

    def test_no_reserve_trimming_or_crossover_operator(self):
        self.assertNotIn("reserve_trim", dg.RULES)
        self.assertEqual(ops.OPERATORS, ("H1", "H2", "H3", "H4", "H5", "R1"))

    def test_random_proposal_is_reproducible_local_and_on_catalog(self):
        a = ops.random_proposal(self.ctx, random.Random(7))
        self.assertEqual(a, ops.random_proposal(self.ctx, random.Random(7)))
        for seed in range(30):
            p = ops.random_proposal(self.ctx, random.Random(seed))
            self.assertEqual(sum(x != y for x, y in zip(self.ctx["genome"], p["requested_genome"])), 1)
            i = self.gene(p["requested_gene"])
            self.assertIn(p["requested_genome"][i], ops.local_neighbors(
                [tuple(o) if self.ctx["genes"][i]["kind"] == "size" else o for o in self.ctx["genes"][i]["options"]],
                p["from"]))
        self.assertEqual(ops.local_neighbors([8, 5, 6], 6), [1, 0])

    def test_guided_choice_merges_attribution_across_operators(self):
        diagnostics = [self.rows["beam_threading"], self.rows["through_bar_depth"], self.rows["column_pm"]]
        alternatives = ops.guided_alternatives(self.ctx, diagnostics)
        self.assertEqual(sorted(alternatives["by_operator"]), ["H2", "H3", "H4"])
        core_up = [p for p in alternatives["by_operator"]["H3"]][0]["requested_genome"]
        origins = ops.merge_origins(alternatives, core_up)
        self.assertEqual({o["operator"] for o in origins}, {"H2", "H3", "H4"})
        picks = {ops.choose_guided(alternatives, random.Random(s))["selected_operator"] for s in range(40)}
        self.assertEqual(picks, {"H2", "H3", "H4"})
        self.assertIsNone(ops.choose_guided(ops.guided_alternatives(self.ctx, []), random.Random(1)))

    def test_finalize_noop_rejection_and_stale(self):
        p = self.proposals("scwb")[0]
        self.assertEqual(ops.finalize(self.ctx, p, self.ctx["genome"], [])["status"], "noop")
        self.assertEqual(ops.finalize(self.ctx, p, [True] * len(self.ctx["genes"]), [])["reason"], "invalid_repair_output")
        up = next(x for x in self.proposals("scwb") if x["origins"][0]["lever"] == "column_up")
        i = self.gene(up["requested_gene"])
        child = list(self.ctx["genome"])
        child[i] -= 1
        self.assertEqual(ops.finalize(self.ctx, up, child, [])["reason"], "repair_changed_intervention_direction")
        stale = copy.deepcopy(p)
        stale["basis_identity"] = "other"
        self.assertEqual(ops.finalize(self.ctx, stale, stale["requested_genome"], [])["reason"], "stale_proposal")

    def test_real_repair_induced_changes_are_explicit(self):
        """An upper-band SFRS column larger than the band below: the existing repair grows the lower band too."""
        up = next(p for p in self.proposals("story_drift") if "s01_03__sfrs_column" == p["requested_gene"])
        upper = copy.deepcopy(up)
        i = self.gene("s04_04__sfrs_column")
        lower = self.gene("s01_03__sfrs_column")
        upper["requested_genome"] = list(self.ctx["genome"])
        upper["requested_genome"][i] = self.ctx["genome"][lower] + 2
        upper["requested_gene"] = "s04_04__sfrs_column"
        upper["from"] = tuple(self.ctx["genes"][i]["options"][self.ctx["genome"][i]])
        upper["to"] = tuple(self.ctx["genes"][i]["options"][upper["requested_genome"][i]])
        repaired, _state, log = self.space.repair(tuple(upper["requested_genome"]))
        out = ops.finalize(self.ctx, upper, list(repaired), [list(x) for x in log])
        self.assertEqual(out["status"], "ready_for_evaluation")
        self.assertTrue(out["repair_log"])
        self.assertIn("s01_03__sfrs_column", [c["gene"] for c in out["induced_changes"]])
        self.assertEqual(out["requested_change"]["gene"], "s04_04__sfrs_column")
        self.assertGreater(len(out["affected_groups"]), 3)
        self.assertFalse(out["engineering_accepted"])

    def test_family_grade_move_reports_every_group(self):
        p = next(x for x in self.proposals("beam_flexure") if x["origins"][0]["lever"] == "grade_up")
        out = ops.finalize(self.ctx, p, p["requested_genome"], [])
        self.assertEqual(len(out["affected_groups"]), 8)

    def test_deduplication_after_repair_and_basis_separation(self):
        p = self.proposals("scwb")[0]
        a = ops.finalize(self.ctx, p, p["requested_genome"], [])
        unique, duplicates = ops.deduplicate_ready([a, copy.deepcopy(a)])
        self.assertEqual((len(unique), len(duplicates)), (1, 1))
        self.assertEqual(len(unique[0]["proposal"]["origins"]), 2)
        other = copy.deepcopy(self.ctx)
        other["basis_identity"] = "another"
        q = copy.deepcopy(p)
        q["basis_identity"] = "another"
        self.assertNotEqual(a["genome_identity"], ops.finalize(other, q, q["requested_genome"], [])["genome_identity"])


if __name__ == "__main__":
    unittest.main()
