"""The normalized diagnostic adapter (Design.Genetic_Diagnostics) on contract-valid synthetic records.

Software semantics only: the numbers are synthetic and nothing here validates a structural check.
"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genetic_fixture as gfx
import grouped_fixture as gf
from Design import Code_Check_Contract as contract
from Design import Genetic_Diagnostics as dg
from Design import Genetic_Operators as ops
from Design import Genetic_Search as ga

BASIS = "fixture-basis"


class DiagnosticAdapter(unittest.TestCase):
    def setUp(self):
        self.fixture = gf.frame()
        self.fixture.__enter__()
        self.seed = gfx.seed_state()
        self.space = ga.DesignSpace(self.seed)

    def tearDown(self):
        self.fixture.__exit__(None, None, None)

    def diagnose(self, record):
        assessed = contract.assess_evaluation(record, submitted_identity=self.seed.identity(), submitted_state=self.seed)
        realized = (record.get("code_checks") or {}).get("realized_identity") or "none"
        ctx = dg.parent_context(self.space, record["state"], basis_identity=BASIS, realized_identity=realized,
                                capacity=(record.get("evidence") or {}).get("capacity"))
        return assessed, ctx, dg.diagnose(record, assessed, ctx)

    def only(self, result, reason):
        items = [d for d in result["diagnostics"] if d["reason"] == reason]
        self.assertEqual(len(items), 1, result["diagnostics"])
        return items[0]

    def test_passing_record_has_no_diagnostics(self):
        assessed, ctx, result = self.diagnose(gfx.record(self.seed))
        self.assertEqual(assessed["classification"], "accepted")
        self.assertEqual((result["status"], result["diagnostics"], result["unknown"]), ("diagnosed", [], []))
        self.assertFalse(ctx["topology_problems"])
        self.assertEqual(ctx["genome"], list(self.space.initial))

    def test_scwb_failure_with_complete_evidence_routes_h1(self):
        evidence = gfx.capacity_evidence(self.seed)
        row = gfx.scwb_failure(evidence, "joint_13", "x", "positive")
        record = gfx.record(self.seed, replace={("joints", "scwb", "joint_13/x/positive"): row}, evidence=evidence)
        assessed, ctx, result = self.diagnose(record)
        self.assertEqual(assessed["classification"], "check_failed")
        d = self.only(result, "scwb")
        self.assertTrue(d["evidence_valid"], d["evidence_problems"])
        self.assertEqual(d["dependencies"], {"nominal_column_strength": True, "nominal_beam_slab_strength": True,
                                             "axial_case_coverage": True})
        self.assertEqual((d["joint_id"], d["axis"], d["sign"]), ("joint_13", "x", "positive"))
        self.assertEqual((d["parent_identity"], d["basis_identity"]), (ctx["parent_identity"], BASIS))
        routed = ops.route(ctx, d)
        self.assertEqual((routed["status"], routed["operator"]), ("proposed", "H1"))
        self.assertEqual(sorted(p["requested_gene"] for p in routed["proposals"]),
                         ["s01_03__sfrs_column", "sfrs_column_grade"])

    def test_scwb_row_must_follow_from_its_joint_record(self):
        evidence = gfx.capacity_evidence(self.seed)
        row = gfx.scwb_failure(evidence, "joint_13", "x", "positive")
        evidence = copy.deepcopy(evidence)
        record_ = next(j for j in evidence["scwb"]["joints"] if j["id"] == "joint_13")
        record_["directions"]["x"]["positive"]["column_capacities"][0]["mn_kip_in"] = 1.0
        record = gfx.record(self.seed, replace={("joints", "scwb", "joint_13/x/positive"): row}, evidence=evidence)
        d = self.only(self.diagnose(record)[2], "scwb")
        self.assertFalse(d["evidence_valid"])
        self.assertEqual(ops.route(self.diagnose(record)[1], d)["reason"], "invalid_or_inapplicable_evidence")

    def test_scwb_axial_coverage_is_derived_not_assumed(self):
        evidence = gfx.capacity_evidence(self.seed)
        row = gfx.scwb_failure(evidence, "joint_13", "x", "positive")
        next(j for j in evidence["scwb"]["joints"] if j["id"] == "joint_13")["directions"]["x"]["positive"][
            "column_capacities"][0]["combination_count"] = 1
        record = gfx.record(self.seed, replace={("joints", "scwb", "joint_13/x/positive"): row}, evidence=evidence)
        _assessed, ctx, result = self.diagnose(record)
        d = self.only(result, "scwb")
        self.assertFalse(d["dependencies"]["axial_case_coverage"])
        self.assertEqual(ops.route(ctx, d)["reason"], "trigger_dependencies_incomplete")

    def test_unknown_scwb_is_evidence_to_obtain_not_a_diagnostic(self):
        replace = {key: gfx.unknown_row(*key, "slab_action_review_missing") for key in
                   [k for k in contract.expected_inventory(self.seed)["exact"] if k[1] == "scwb"]}
        assessed, _ctx, result = self.diagnose(gfx.record(self.seed, replace=replace))
        self.assertEqual(assessed["classification"], "unresolved")
        self.assertEqual(result["diagnostics"], [])
        self.assertEqual(len(result["unknown"]), 1)
        self.assertEqual(result["unknown"][0]["reason_codes"], ["slab_action_review_missing"])
        self.assertEqual(result["unknown"][0]["count"], len(replace))

    def test_strength_row_is_split_into_its_failed_components(self):
        gid = "s01_03__beam_x__edge"
        record = gfx.record(self.seed, strength_overrides={gid: {"flexure_positive": {"dcr": 1.3, "mu_kip_in": 1300.0},
                                                                 "shear": {"dcr": 1.2, "vu_kip": 120.0}}})
        _assessed, ctx, result = self.diagnose(record)
        flexure = self.only(result, "beam_flexure")
        self.assertEqual((flexure["demand"], flexure["capacity"], flexure["units"]), (1300.0, 1000.0, "kip-in"))
        self.assertEqual(flexure["dependencies"], {"member_actions": True, "section_strength": True})
        self.assertEqual(flexure["source"]["component"], "flexure_positive")
        levers = sorted(o["lever"] for p in ops.route(ctx, flexure)["proposals"] for o in p["origins"])
        self.assertIn("depth", levers)
        self.assertIn("grade_up", levers)
        shear = [d for d in result["diagnostics"] if d["source"]["component"] == "shear"]
        self.assertEqual(len(shear), 1)
        self.assertIsNone(shear[0]["reason"])
        self.assertEqual(ops.route(ctx, shear[0])["proposals"], [])

    def test_column_interaction_routes_size_and_grade(self):
        gid = "s01_03__column__corner"
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, strength_overrides={gid: {"pm": {"dcr": 1.2}}}))
        d = self.only(result, "column_pm")
        routed = ops.route(ctx, d)
        self.assertEqual(sorted(p["requested_gene"] for p in routed["proposals"]),
                         ["s01_03__sfrs_column", "sfrs_column_grade"])

    def test_domain_failure_sentinels_are_not_routed(self):
        gid = "s01_03__beam_x__edge"
        record = gfx.record(self.seed, strength_overrides={
            gid: {"flexure_negative": {"dcr": 999.0, "mu_kip_in": 500.0, "tension_controlled": False}}})
        _assessed, ctx, result = self.diagnose(record)
        d = self.only(result, "beam_flexure")
        self.assertFalse(d["evidence_valid"])
        self.assertEqual(ops.route(ctx, d)["reason"], "invalid_or_inapplicable_evidence")

    def test_capacity_shear_needs_a_settled_hoop_selection_outcome(self):
        gid = "s01_03__beam_x__edge"
        shear = gfx.row("capacity", "beam.capacity_shear_section", gid, 120.0, 100.0, "<=", "kip")
        key = ("capacity", "beam.hoops_selected", gid)
        exhausted = {("capacity", "beam.capacity_shear_section", gid): shear, key: gfx.row(*key, 0, 1, "==")}
        # A hoop ladder exhausted by the same over-demand is a finished selection: the section levers remain.
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace=exhausted))
        d = self.only(result, "beam_capacity_shear")
        self.assertTrue(d["dependencies"]["transverse_selection"])
        self.assertEqual(ops.route(ctx, d)["operator"], "H2")
        self.assertEqual(ops.route(ctx, d)["status"], "proposed")
        # Hoops not settled, or no recorded selection outcome: the inner selection must finish first.
        for record in (gfx.record(self.seed, replace=exhausted, constraints={"hoops_settled": False}),
                       gfx.record(self.seed, replace={**exhausted, key: gfx.unknown_row(*key, "fixture_pending")})):
            _assessed, ctx, result = self.diagnose(record)
            d = self.only(result, "beam_capacity_shear")
            self.assertFalse(d["dependencies"]["transverse_selection"])
            self.assertEqual(ops.route(ctx, d)["reason"], "trigger_dependencies_incomplete")
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace={
            ("capacity", "beam.capacity_shear_section", gid): shear}))
        self.assertEqual(ops.route(ctx, self.only(result, "beam_capacity_shear"))["status"], "proposed")

    def test_lane_limited_beam_shortfall_is_bar_room_not_strength(self):
        gid = "s01_03__beam_x__edge"
        overrides = {gid: {"flexure_negative": {"dcr": 1.3, "mu_kip_in": 1300.0}}}
        record = gfx.record(self.seed, strength_overrides=overrides, constraints={"lane_limited_beam_groups": [gid]})
        _assessed, ctx, result = self.diagnose(record)
        self.assertFalse([x for x in result["diagnostics"] if x["reason"] == "beam_flexure"])
        d = self.only(result, "beam_lane_limited_strength")
        self.assertTrue(all(d["dependencies"].values()), d["dependencies"])
        routed = ops.route(ctx, d)
        self.assertEqual((routed["status"], routed["operator"]), ("proposed", "H3"))
        levers = {o["lever"] for p in routed["proposals"] for o in p["origins"]}
        self.assertNotIn("grade_up", levers)
        self.assertTrue({"depth", "column_up"} <= levers, levers)
        cores = {o["target_group"] for p in routed["proposals"] for o in p["origins"] if o["lever"] == "column_up"}
        expected = {t["core_group"] for t in ctx["joint_types"].values() if gid in t["beams"]["x"]}
        self.assertEqual(cores, expected)
        # Without the lane-limited selection record the same shortfall is a strength deficit (H2, grade included).
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, strength_overrides=overrides))
        routed = ops.route(ctx, self.only(result, "beam_flexure"))
        self.assertIn("grade_up", {o["lever"] for p in routed["proposals"] for o in p["origins"]})

    def gravity_shear_evidence(self, gid, axis, ve, phi_vn, governing="y"):
        """The producer's per-direction records of a gravity column: each direction's worst-story Ve, each direction's
        hoop phi Vn with the lighter-direction scalar, and the direction the row compares (strengths.capacity_shear_axis
        of Perimeter_Systems.gravity_column_capacity_shear); ``axis`` None leaves the gravity-frame record out."""
        evidence = gfx.capacity_evidence(self.seed)
        evidence["columns"][gid].update(
            governing={"axis": governing, "ve_kip": ve[governing], "vs_required_kip": 40.0},
            worst_by_direction={a: {"axis": a, "ve_kip": ve[a]} for a in ("x", "y")},
            hoops={"phi_vn_kip": min(phi_vn.values()), "by_direction": {a: {"phi_vn_kip": phi_vn[a]} for a in ("x", "y")}})
        if axis is not None:
            evidence["gravity_frame"] = {"groups": {gid: {"strengths": {"capacity_shear_axis": axis}}}}
        return evidence

    def test_gravity_column_capacity_shear_is_mapped_per_direction(self):
        gid = "s01_03__column__interior"
        key = ("capacity", "gravity.column_capacity_shear", gid)
        # The recorded direction's Ve against that direction's phi Vn: a deficit the section levers address. x has the
        # larger ratio although y governs Vs,req: the row is read on the recorded axis, not the governing one.
        record = gfx.record(self.seed, replace={key: gfx.row(*key, 450.0, 420.0, "<=", "kip")},
                            evidence=self.gravity_shear_evidence(gid, "x", {"x": 450.0, "y": 460.0}, {"x": 420.0, "y": 500.0}))
        _assessed, ctx, result = self.diagnose(record)
        self.assertEqual(result["unsupported"], {})
        d = self.only(result, "column_capacity_shear")
        self.assertTrue(d["evidence_valid"], d["evidence_problems"])
        self.assertTrue(all(d["dependencies"].values()), d["dependencies"])
        routed = ops.route(ctx, d)
        self.assertEqual((routed["status"], routed["operator"]), ("proposed", "H2"))
        genes = {p["requested_gene"] for p in routed["proposals"]}
        self.assertTrue(genes and all("gravity" in g for g in genes), genes)
        # A row that compares y's Ve with x's (lighter) phi Vn while each direction passes: not routed.
        asymmetric = ({"x": 202.81, "y": 241.63}, {"x": 208.53, "y": 278.04})
        mixed = gfx.record(self.seed, replace={key: gfx.row(*key, 241.63, 208.53, "<=", "kip")},
                           evidence=self.gravity_shear_evidence(gid, "y", *asymmetric))
        _assessed, ctx, result = self.diagnose(mixed)
        d = self.only(result, "column_capacity_shear")
        self.assertFalse(d["evidence_valid"])
        self.assertTrue(any("another direction" in p for p in d["evidence_problems"]), d["evidence_problems"])
        self.assertEqual(ops.route(ctx, d)["reason"], "invalid_or_inapplicable_evidence")
        # The same row without a recorded direction (as the cross-direction producer wrote it): not authenticated.
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace={key: gfx.row(*key, 241.63, 208.53, "<=", "kip")},
                                                          evidence=self.gravity_shear_evidence(gid, None, *asymmetric)))
        d = self.only(result, "column_capacity_shear")
        self.assertFalse(d["evidence_valid"])
        self.assertTrue(any("no per-direction" in p for p in d["evidence_problems"]), d["evidence_problems"])
        # A recorded direction that is not the one with the larger ratio: not authenticated.
        _assessed, ctx, result = self.diagnose(gfx.record(
            self.seed, replace={key: gfx.row(*key, 450.0, 420.0, "<=", "kip")},
            evidence=self.gravity_shear_evidence(gid, "y", {"x": 500.0, "y": 450.0}, {"x": 430.0, "y": 420.0})))
        d = self.only(result, "column_capacity_shear")
        self.assertEqual(d["evidence_problems"], ["the gravity column row compares the y direction, not the direction "
                                                  "with the larger Ve / phi Vn"])
        # No per-direction hoop record to authenticate the row against: not routed either.
        bare = self.gravity_shear_evidence(gid, "x", {"x": 450.0, "y": 460.0}, {"x": 420.0, "y": 500.0})
        bare["columns"][gid].pop("hoops")
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace={key: gfx.row(*key, 450.0, 420.0, "<=", "kip")},
                                                          evidence=bare))
        self.assertFalse(self.only(result, "column_capacity_shear")["evidence_valid"])

    def test_gravity_column_row_demand_must_be_the_recorded_directions_ve(self):
        gid = "s01_03__column__interior"
        key = ("capacity", "gravity.column_capacity_shear", gid)
        evidence = self.gravity_shear_evidence(gid, "x", {"x": 450.0, "y": 300.0}, {"x": 420.0, "y": 500.0}, governing="x")
        record = gfx.record(self.seed, replace={key: gfx.row(*key, 470.0, 420.0, "<=", "kip")}, evidence=evidence)
        _assessed, ctx, result = self.diagnose(record)
        d = self.only(result, "column_capacity_shear")
        self.assertFalse(d["evidence_valid"])
        self.assertEqual(d["evidence_problems"], ["the gravity column row's demand is not the x-direction Ve"])
        self.assertEqual(ops.route(ctx, d)["reason"], "invalid_or_inapplicable_evidence")

    def test_an_unevaluated_gravity_shear_row_names_the_missing_hoops(self):
        from Design.SMRF_Common import not_evaluated
        gid = "s01_03__column__interior"
        key = ("capacity", "gravity.column_capacity_shear", gid)
        bare = contract.compact_row("capacity", not_evaluated(key[1], "fixture clause", "not supplied", gid))
        self.assertFalse((bare.get("details") or {}).get("reason_codes"))
        # The reason comes from the producer's own record, never assumed: no hoops there, or none named. Without a
        # known governing Ve the missing hoops are not the whole explanation: no reason is given.
        unknown_ve = {"axis": "x", "ve_kip": float("nan"), "vs_required_kip": 40.0}
        for hoops, governing, codes in ((None, "kept", ["hoops_not_selected"]), ({"phi_vn_kip": 90.0}, "kept", ["unclassified"]),
                                        (None, None, ["unclassified"]), (None, unknown_ve, ["unclassified"])):
            evidence = gfx.capacity_evidence(self.seed)
            evidence["columns"][gid]["hoops"] = hoops
            if governing != "kept":
                evidence["columns"][gid]["governing"] = governing
            _assessed, _ctx, result = self.diagnose(gfx.record(self.seed, replace={key: bare}, evidence=evidence))
            item = next(u for u in result["unknown"] if u["check_id"] == key[1])
            self.assertEqual(item["reason_codes"], codes, (hoops, governing))

    def test_the_ratio_floor_flag_needs_a_settled_selection_and_a_mechanism_ve(self):
        from dataclasses import replace
        from Design import Group_Selection as selection
        from Design.Config import DesignConfig
        gid = "s01_03__column__corner"
        design = self.seed.designs[gid]
        light = min(selection.column_cage_candidates(design, DesignConfig()), key=dg._cage_area)
        tie = next(t for t, groups in self.seed.ties().items() if gid in groups)
        state = self.seed.with_tied_designs({tie: replace(design, bar_size=light.bar_size, top_bars=light.top_bars,
                                                          bot_bars=light.top_bars, side_bars=light.side_bars)})
        self.assertIs(dg.cage_at_ratio_floor(state, gid), True)
        key = ("capacity", "column.capacity_shear_section", gid)
        failing = {key: gfx.row(*key, 120.0, 100.0, "<=", "kip")}

        def flag(**kwargs):
            record = gfx.record(state, replace=failing, **kwargs)
            assessed = contract.assess_evaluation(record, submitted_identity=state.identity(), submitted_state=state)
            ctx = dg.parent_context(self.space, record["state"], basis_identity=BASIS,
                                    realized_identity=record["code_checks"]["realized_identity"],
                                    capacity=record["evidence"]["capacity"])
            return self.only(dg.diagnose(record, assessed, ctx), "column_capacity_shear")["cage_at_ratio_floor"]
        self.assertIs(flag(), True)
        self.assertIsNone(flag(constraints={"longitudinal_selection_settled": False}))
        # The analysis shear sets Ve, the record does not say what sets it, or another Ve method (the beams can set Ve
        # below the column's own envelope): unknown, no lever withheld.
        from Design import SMRF_Capacity_Design as scd
        own = scd.COLUMN_SHEAR_METHOD_COLUMN_OWN
        for governing in ({"axis": "x", "ve_kip": 60.0, "vs_required_kip": 40.0, "ve_own_envelope_kip": 45.0,
                           "vu_analysis_kip": 60.0, "method": own}, {"axis": "x", "ve_kip": 60.0, "vs_required_kip": 40.0},
                          {"axis": "x", "ve_kip": 60.0, "vs_required_kip": 40.0, "ve_own_envelope_kip": 60.0},
                          *({"axis": "x", "ve_kip": 68.5, "vs_required_kip": 40.0, "ve_own_envelope_kip": 129.7,
                             "vu_analysis_kip": 40.0, "method": method}
                            for method in scd.COLUMN_SHEAR_METHODS if method != own)):
            evidence = gfx.capacity_evidence(state)
            evidence["columns"][gid]["governing"] = governing
            self.assertIsNone(flag(evidence=evidence), governing)

    def test_column_cage_at_ratio_floor_is_derived_from_the_cage_domain(self):
        import types
        from dataclasses import replace
        from Design import Group_Selection as selection
        from Design.Config import DesignConfig
        gid = "s01_03__column__corner"
        design = self.seed.designs[gid]
        candidates = selection.column_cage_candidates(design, DesignConfig())
        light = min(candidates, key=dg._cage_area)
        heavy = max(candidates, key=dg._cage_area)
        def with_cage(cage, bars=None):
            top, side = bars or (cage.top_bars, cage.side_bars)
            return types.SimpleNamespace(designs={gid: replace(design, bar_size=cage.bar_size, top_bars=top, bot_bars=top,
                                                               side_bars=side)})
        for cage, expected in ((light, True), (heavy, False)):
            self.assertIs(dg.cage_at_ratio_floor(with_cage(cage), gid), expected)
        self.assertIsNone(dg.cage_at_ratio_floor(self.seed, "s01_03__beam_x__edge"))
        # A cage lighter than every admissible one did not come from the selection: unknown, not "at the floor".
        below = with_cage(light, (2, 0))
        self.assertLess(dg._cage_area(below.designs[gid]), min(map(dg._cage_area, candidates)))
        self.assertIsNone(dg.cage_at_ratio_floor(below, gid))
        # The run's configuration sets the domain, and diagnose passes it on.
        strict = replace(DesignConfig(), rebar=replace(DesignConfig().rebar, rho_col_min=0.02))
        floor = min(selection.column_cage_candidates(design, strict), key=dg._cage_area)
        self.assertGreater(dg._cage_area(floor), dg._cage_area(light))
        self.assertIs(dg.cage_at_ratio_floor(with_cage(floor), gid, strict), True)
        self.assertIs(dg.cage_at_ratio_floor(with_cage(floor), gid), False)
        self.assertIsNone(dg.cage_at_ratio_floor(with_cage(light), gid, strict))
        tie = next(t for t, groups in self.seed.ties().items() if gid in groups)
        state = self.seed.with_tied_designs({tie: with_cage(floor).designs[gid]})
        key = ("capacity", "column.capacity_shear_section", gid)
        record = gfx.record(state, replace={key: gfx.row(*key, 120.0, 100.0, "<=", "kip")})
        assessed = contract.assess_evaluation(record, submitted_identity=state.identity(), submitted_state=state)
        ctx = dg.parent_context(self.space, record["state"], basis_identity=BASIS,
                                realized_identity=record["code_checks"]["realized_identity"],
                                capacity=record["evidence"]["capacity"])
        for cfg, expected in ((strict, True), (None, False)):
            result = dg.diagnose(record, assessed, ctx, cfg=cfg)
            self.assertIs(self.only(result, "column_capacity_shear")["cage_at_ratio_floor"], expected)

    def test_drift_targets_the_story_and_axis(self):
        key = ("drift", "demands.story_drift", "story:2/Qx/x")
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace={key: gfx.row(*key, 0.03, 0.02, "<=", "in/in")}))
        d = self.only(result, "story_drift")
        self.assertEqual((d["story"], d["axis"]), (2, "x"))
        self.assertTrue(all(d["dependencies"].values()), d["dependencies"])
        genes = sorted(p["requested_gene"] for p in ops.route(ctx, d)["proposals"])
        self.assertTrue(genes)
        self.assertTrue(all(g.startswith("s01_03__sfrs_") for g in genes), genes)

    def test_declared_units_are_authenticated(self):
        key = ("drift", "demands.story_drift", "story:2/Qx/x")
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace={key: gfx.row(*key, 0.03, 0.02, "<=", "in")}))
        d = self.only(result, "story_drift")
        self.assertFalse(d["evidence_valid"])
        self.assertEqual(ops.route(ctx, d)["reason"], "invalid_or_inapplicable_evidence")

    def test_drift_needs_its_analysis_coverage(self):
        key = ("drift", "demands.story_drift", "story:2/Qx/x")
        coverage = ("drift", "demands.node_coverage", "story:2/Qx")
        record = gfx.record(self.seed, replace={key: gfx.row(*key, 0.03, 0.02, "<=", "in/in"),
                                                coverage: gfx.unknown_row(*coverage, "fixture_missing_nodes")})
        _assessed, ctx, result = self.diagnose(record)
        d = self.only(result, "story_drift")
        self.assertFalse(d["dependencies"]["drift_analysis"])
        self.assertEqual(ops.route(ctx, d)["reason"], "trigger_dependencies_incomplete")

    def test_threading_targets_the_joint_core(self):
        key = ("capacity", "beam.bars_thread_column", "joint_14/x")
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace={key: gfx.row(*key, 0, 1, "==")}))
        d = self.only(result, "beam_threading")
        self.assertEqual((d["joint_type_id"], d["group_id"], d["units"]), ("joint_14", "s01_03__beam_x__edge", "predicate"))
        routed = ops.route(ctx, d)
        self.assertEqual([p["requested_gene"] for p in routed["proposals"]], ["s01_03__sfrs_column"])
        self.assertEqual(routed["proposals"][0]["origins"][0]["target_group"], "s01_03__column__y_boundary_edge")

    def test_joint_shear_compares_core_size_and_grade(self):
        location = "joint_shear/joint_13/x"
        record = gfx.record(self.seed, replace={("joints", "joint_shear", location):
                                                gfx.row("joints", "joint_shear", location, 250.0, 200.0, "<=", "kip")})
        _assessed, ctx, result = self.diagnose(record)
        d = self.only(result, "joint_shear")
        self.assertTrue(all(d["dependencies"].values()), d["dependencies"])
        self.assertEqual(sorted(p["requested_gene"] for p in ops.route(ctx, d)["proposals"]),
                         ["s01_03__sfrs_column", "sfrs_column_grade"])

    def test_slab_overlap_without_blocked_dimension_abstains(self):
        key = ("capacity", "beam.bar_stacking_clear_of_slab_mats", "joint_13")
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace={key: gfx.row(*key, 0, 1, "==")}))
        d = self.only(result, "slab_mat_overlap")
        self.assertFalse(d["dependencies"]["diagnosed_dimension"])
        self.assertIsNone(d["blocked_dimension"])
        self.assertEqual(ops.route(ctx, d)["proposals"], [])

    def test_failure_without_a_declared_reason_is_reported_unsupported(self):
        key = ("detailing", "beam.minimum_width", "s01_03__beam_x__edge")
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, replace={key: gfx.row(*key, 8.0, 10.0, ">=", "in")}))
        self.assertEqual(result["unsupported"], {"detailing|beam.minimum_width": 1})
        d = next(x for x in result["diagnostics"] if x["source"]["check_id"] == "beam.minimum_width")
        self.assertEqual(ops.route(ctx, d)["reason"], "invalid_or_inapplicable_evidence")

    def test_invalid_record_yields_no_diagnostics(self):
        record = gfx.record(self.seed, strength_overrides={"s01_03__column__corner": {"pm": {"dcr": 1.2}}})
        record["feasible"] = True                         # contradicts its own failed row
        assessed, _ctx, result = self.diagnose(record)
        self.assertFalse(assessed["valid"])
        self.assertEqual((result["status"], result["diagnostics"]), ("invalid_record", []))

    def test_topology_naming_unknown_groups_invalidates_joint_diagnostics(self):
        evidence = gfx.capacity_evidence(self.seed)
        row = gfx.scwb_failure(evidence, "joint_13", "x", "positive")
        evidence["joint_types"]["joint_13"]["core_group"] = "not_a_group"
        record = gfx.record(self.seed, replace={("joints", "scwb", "joint_13/x/positive"): row}, evidence=evidence)
        _assessed, ctx, result = self.diagnose(record)
        self.assertTrue(ctx["topology_problems"])
        self.assertFalse(self.only(result, "scwb")["evidence_valid"])

    def test_lane_limited_column_levers_follow_the_joint_topology(self):
        gid = "s01_03__beam_x__edge"
        overrides = {gid: {"flexure_negative": {"dcr": 1.3, "mu_kip_in": 1300.0}}}
        lanes = {"lane_limited_beam_groups": [gid]}
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, strength_overrides=overrides, constraints=lanes))
        self.assertIs(self.only(result, "beam_lane_limited_strength")["joint_topology_consistent"], True)
        evidence = gfx.capacity_evidence(self.seed)
        evidence["joint_types"]["joint_13"]["core_group"] = "not_a_group"
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, strength_overrides=overrides, constraints=lanes,
                                                          evidence=evidence))
        self.assertTrue(ctx["topology_problems"])
        d = self.only(result, "beam_lane_limited_strength")
        self.assertIs(d["joint_topology_consistent"], False)
        routed = ops.route(ctx, d)
        self.assertEqual({o["lever"] for p in routed["proposals"] for o in p["origins"]}, {"depth", "width"})
        cores = {t["core_group"] for t in ctx["joint_types"].values()
                 if gid in t["beams"]["x"] and t["core_group"] in ctx["groups"]}
        self.assertTrue(cores)
        self.assertEqual(sorted((w["lever"], w["group"]) for w in routed["withheld"]),
                         sorted(("column_up", core) for core in cores))      # one withheld step per core

    def test_a_context_of_another_design_yields_no_diagnostics(self):
        record = gfx.record(self.seed, strength_overrides={"s01_03__column__corner": {"pm": {"dcr": 1.2}}})
        assessed, ctx, _result = self.diagnose(record)
        for field, value in (("parent_identity", "another realized design"), ("state_identity", "another state")):
            other = {**ctx, field: value}
            self.assertEqual(dg.diagnose(record, assessed, other)["status"], "unbound_context")

    def test_threading_never_substitutes_a_beam(self):
        key = ("capacity", "beam.bars_thread_column", "joint_14/x")
        for beam in (None, "s01_03__beam_x__interior"):
            evidence = gfx.capacity_evidence(self.seed)
            evidence["joint_types"]["joint_14"]["threading"]["x"]["beam_group"] = beam
            record = gfx.record(self.seed, replace={key: gfx.row(*key, 0, 1, "==")}, evidence=evidence)
            _assessed, ctx, result = self.diagnose(record)
            d = self.only(result, "beam_threading")
            self.assertIsNone(d["group_id"])
            self.assertEqual(ops.route(ctx, d)["proposals"], [])

    def test_declared_not_applicable_rows_are_not_evidence_to_obtain(self):
        record = gfx.record(self.seed)
        assessed, ctx, _result = self.diagnose(record)
        record["code_checks"]["rows"].append(gfx.unknown_row("joints", "through_bar_anchorage", "", "fixture_code"))
        self.assertEqual(len(dg.diagnose(record, assessed, ctx)["unknown"]), 1)
        assessed["code_check"]["not_applicable"] = [{"family": "joints", "id": "through_bar_anchorage", "location": ""}]
        self.assertEqual(dg.diagnose(record, assessed, ctx)["unknown"], [])

    def test_adapter_does_not_mutate_the_record(self):
        evidence = gfx.capacity_evidence(self.seed)
        row = gfx.scwb_failure(evidence, "joint_13", "x", "positive")
        record = gfx.record(self.seed, replace={("joints", "scwb", "joint_13/x/positive"): row}, evidence=evidence)
        before = copy.deepcopy({k: v for k, v in record.items() if k != "state"})
        self.diagnose(record)
        self.assertEqual({k: v for k, v in record.items() if k != "state"}, before)

    def threading_constraints(self, gid, lookahead_size, **changes):
        """A selection record of a threading-constrained deficit of ``gid`` on the seed's own cage (Group_Selection
        threading_constraint_summary form): the strongest threading cage short, a stronger nonthreading one blocked by
        the cores of the joints the beam threads, and a geometry-only look-ahead size for each core."""
        design = self.seed.designs[gid]
        cage = {"bar_size": design.bar_size, "top_bars": design.top_bars, "bot_bars": design.bot_bars}
        cores = sorted({kind["core_group"] for kind in gfx.capacity_evidence(self.seed)["joint_types"].values()
                        if any(item["beam_group"] == gid for item in kind["threading"].values())})
        record = {"unit": "fixture", "basis": "beam_threading_evidence_v2", "authenticated": True, "lane_limited": False,
                  "constrains_strength": True, "deficit_remains": True, "threading_constrained_deficit": True,
                  "installed": {**cage, "dcr": 1.3, "threads": True}, "best_threading": {**cage, "dcr": 1.3, "threads": True},
                  "best_nonthreading": {**cage, "bar_size": cage["bar_size"] + 2, "dcr": 1.05, "threads": False},
                  "blocking_cores": cores,
                  "lookahead": {f"{core}/x": {"scope": "core_cage", "core_group": core, "axis": "x",
                                              "current_size_in": self.seed.designs[core].b_in,
                                              "lanes_needed_per_layer": 3, "first_size_same_cage": lookahead_size,
                                              "first_size_any_cage": self.seed.designs[core].b_in} for core in cores}}
        record.update(changes)
        return {"threading_constrained_beam_groups": [gid], "beam_threading_evidence": {gid: record}}, cores

    def test_threading_constrained_shortfall_routes_beam_levers_its_blocking_cores_and_the_lookahead_size(self):
        gid = "s01_03__beam_x__edge"
        overrides = {gid: {"flexure_negative": {"dcr": 1.3, "mu_kip_in": 1300.0}}}
        core = "s01_03__column__corner"
        gene = next(g for g in self.space.genes if g.kind == "size" and core in g.groups)
        current = self.seed.designs[core].b_in
        larger = sorted(o[0] for o in gene.options if o[0] == o[1] and o[0] > current)
        self.assertGreaterEqual(len(larger), 2)                       # a look-ahead beyond the next step exists
        constraints, cores = self.threading_constraints(gid, larger[1])
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, strength_overrides=overrides, constraints=constraints))
        self.assertFalse([x for x in result["diagnostics"] if x["reason"] in ("beam_flexure", "beam_lane_limited_strength")])
        d = self.only(result, "beam_threading_constrained_strength")
        self.assertTrue(all(d["dependencies"].values()), (d["dependencies"], d.get("dependency_problems")))
        self.assertEqual(d["blocking_cores"], cores)
        routed = ops.route(ctx, d)
        self.assertEqual((routed["status"], routed["operator"]), ("proposed", "H3"))
        levers = {o["lever"] for p in routed["proposals"] for o in p["origins"]}
        self.assertTrue({"depth", "width", "grade_up", "column_up", "column_lanes"} <= levers, levers)
        targets = {o["target_group"] for p in routed["proposals"] for o in p["origins"] if o["lever"].startswith("column_")}
        self.assertEqual(targets, set(cores))
        jumps = {tuple(p["to"]) for p in routed["proposals"] if any(o["lever"] == "column_lanes" for o in p["origins"])}
        self.assertEqual(jumps, {(larger[1], larger[1])})
        # Evidence that does not authenticate (another cage than the final one, no stronger nonthreading cage, or a
        # deficit that is not recorded): the dependency is False and the router abstains, never a pass.
        for changes in ({"installed": {"bar_size": 4, "top_bars": 2, "bot_bars": 2, "dcr": 1.3, "threads": True}},
                        {"best_nonthreading": {"bar_size": 11, "top_bars": 3, "bot_bars": 3, "dcr": 1.4, "threads": False}},
                        {"deficit_remains": False}, {"blocking_cores": []}):
            constraints, _cores = self.threading_constraints(gid, larger[1], **changes)
            _assessed, ctx, result = self.diagnose(gfx.record(self.seed, strength_overrides=overrides, constraints=constraints))
            d = self.only(result, "beam_threading_constrained_strength")
            self.assertFalse(d["dependencies"]["threading_constrained_selection"], changes)
            self.assertTrue(d["dependency_problems"])
            self.assertEqual(ops.route(ctx, d)["reason"], "trigger_dependencies_incomplete")
        # lane_limited keeps its own meaning and route; an unlisted shortfall is a strength deficit (H2).
        constraints, _cores = self.threading_constraints(gid, larger[1])
        both = {**constraints, "lane_limited_beam_groups": [gid]}
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, strength_overrides=overrides, constraints=both))
        self.only(result, "beam_lane_limited_strength")
        _assessed, ctx, result = self.diagnose(gfx.record(self.seed, strength_overrides=overrides))
        self.assertEqual(ops.route(ctx, self.only(result, "beam_flexure"))["operator"], "H2")

    def test_every_declared_source_and_reason_is_routable(self):
        strength = {"beam_flexure", "beam_lane_limited_strength", "beam_threading_constrained_strength",
                    "column_pm"}                                                      # member.strength components
        reasons = {spec[0] for spec in dg.SOURCES.values()} | strength
        self.assertEqual(reasons | set(dg.UNMAPPED), set(dg.RULES))


if __name__ == "__main__":
    unittest.main()
