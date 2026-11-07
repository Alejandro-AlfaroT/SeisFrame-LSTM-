"""Population selection and accepted-design retention (Design.Genetic_Population) on synthetic selection records.

Selection semantics only: tiers, comparability, fronts, diversity, witnesses, deduplication and the archive views.
None of it is evidence that the policy improves a search or that a design is structurally credible.
"""
import copy
import random
import unittest

from Design import Genetic_Population as gp

RUN = {"case_id": "case", "basis_id": "basis", "feature_weights": [0.5, 0.5, 1.0], "gene_kinds": ["size", "grade"]}


def check(state="pass", v=0.0, u=0.5, b="B"):
    return {"s": state, "v": v, "u": u, "b": b}


def rec(rid, classification="unresolved", checks=None, *, inventory="inv-A", known_items=None, complete=True,
        realized=None, genome=(0, 0), features=(0.0, 0.0, 0.0), quantities=None, origins=None, unknown=None, valid=True):
    checks = checks if checks is not None else {"strength|member.strength|g1": check("fail", 0.3, 1.3)}
    known = sum(1 for c in checks.values() if c["s"] in ("pass", "fail"))
    return {"id": rid, "case_id": "case", "basis_id": "basis", "realized_identity": realized or f"realized-{rid}",
            "input_identity": f"input-{rid}", "classification": classification, "valid": valid,
            "numerical_complete": complete, "code_check_status": "failed", "checks": checks, "inventory_id": inventory,
            "known_items": known if known_items is None else known_items, "genome": list(genome), "features": list(features),
            "quantities": quantities or {"concrete_total_in3": 1.0, "total_steel_in3": 1.0},
            "qualification": "not_performed", "nonlinear_performance": {"status": "not_requested"},
            "origins": origins or [{"operator": "R1"}], "unknown": unknown or [], "missing": [], "outcome": "evaluated"}


def accepted(rid, **kw):
    return rec(rid, "accepted", {"strength|member.strength|g1": check("pass", 0.0, 0.8)}, **kw)


class Selection(unittest.TestCase):
    def test_tiers_and_eligibility(self):
        records = [accepted("a"), rec("f", "check_failed"), rec("u", "unresolved"),
                   rec("e", "unresolved", {"k|x|": check("unknown", 0.0, None)}),
                   rec("n", "unresolved", complete=False), rec("i", "invalid", valid=False)]
        decision = gp.select(records, 6, RUN, 1)
        self.assertEqual(decision["selected_ids"], ["a", "f", "u"])
        reasons = {q["id"]: q["reason"] for q in decision["queues"]}
        self.assertEqual(reasons, {"e": "evidence_acquisition_required", "n": "analysis_recovery_required",
                                   "i": "invalid_record"})
        self.assertTrue(decision["underfilled"])
        self.assertTrue(decision["survivor_is_not_acceptance"])
        self.assertEqual([f["tiers"] for f in decision["fronts"]], [[0], [1], [2]])

    def test_coverage_is_compared_only_within_one_inventory(self):
        more, less = rec("more", known_items=10), rec("less", known_items=8)
        self.assertTrue(gp.dominates(gp.audit(more, RUN), gp.audit(less, RUN)))
        other = rec("other", known_items=8, inventory="inv-B")
        self.assertFalse(gp.dominates(gp.audit(more, RUN), gp.audit(other, RUN)))
        self.assertFalse(gp.dominates(gp.audit(other, RUN), gp.audit(more, RUN)))

    def test_progress_is_compared_only_between_equal_signatures(self):
        a = rec("a", checks={"x|a|": check("fail", 0.1, 1.1), "x|b|": check("pass")})
        b = rec("b", checks={"x|a|": check("fail", 0.5, 1.5), "x|b|": check("pass")})
        self.assertTrue(gp.dominates(gp.audit(a, RUN), gp.audit(b, RUN)))
        basis = copy.deepcopy(b)
        basis["checks"]["x|b|"]["b"] = "other-branch"
        self.assertFalse(gp.dominates(gp.audit(a, RUN), gp.audit(basis, RUN)))
        mask = rec("m", checks={"x|a|": check("fail", 0.5, 1.5), "x|b|": check("unknown", 0.0, None)}, known_items=2)
        self.assertFalse(gp.dominates(gp.audit(a, RUN), gp.audit(mask, RUN)))

    def test_family_maximum_keeps_row_count_from_dominating_the_mean(self):
        many = {f"joints|scwb|j{i}": check("fail", 0.1, 1.1) for i in range(20)}
        many["drift|demands.story_drift|s1"] = check("pass")
        audit = gp.audit(rec("a", checks=many), RUN)
        self.assertEqual(audit["progress"], (0.1, 0.05))

    def test_accepted_designs_never_dominate_each_other(self):
        a, b = accepted("a", quantities={"concrete_total_in3": 1.0}), accepted("b", quantities={"concrete_total_in3": 9.0})
        self.assertFalse(gp.dominates(gp.audit(a, RUN), gp.audit(b, RUN)))
        self.assertFalse(gp.dominates(gp.audit(b, RUN), gp.audit(a, RUN)))

    def test_contradictory_acceptance_is_invalid(self):
        bad = rec("x", "accepted", {"k|a|": check("fail", 0.2, 1.2)})
        self.assertEqual(gp.audit(bad, RUN)["reason"], "contradictory_acceptance")

    def test_fixed_domain_diversity_and_order_invariance(self):
        records = [rec(f"r{i}", features=(i / 9, i / 9, (i % 3) / 2)) for i in range(10)]
        first = gp.select(records, 4, RUN, 7)
        shuffled = list(records)
        random.Random(3).shuffle(shuffled)
        self.assertEqual(gp.select(shuffled, 4, RUN, 7)["selected_ids"], first["selected_ids"])
        chosen = [r for r in records if r["id"] in first["selected_ids"]]
        self.assertTrue(any(r["features"][0] <= 0.12 for r in chosen) and any(r["features"][0] >= 0.88 for r in chosen))

    def test_local_regressions_survive_aggregate_progress(self):
        parent = rec("p", checks={"x|a|": check("fail", 0.5, 1.5), "x|b|": check("fail", 0.1, 1.1),
                                  "x|c|": check("pass", 0.0, 0.5), "x|d|": check("pass", 0.0, 0.5)})
        child = rec("c", checks={"x|a|": check("fail", 0.2, 1.2), "x|b|": check("fail", 0.15, 1.15),
                                 "x|c|": check("pass", 0.0, 0.9), "x|d|": check("unknown", 0.0, None),
                                 "x|e|": check("pass", 0.0, 0.1)}, known_items=4)
        decision = gp.select([parent, child], 1, RUN, 1)
        witness = next(w for w in decision["regression_witnesses"] if w["reference"] == "p")
        self.assertEqual([w["id"] for w in witness["violation_worse"]], ["x|b|"])
        self.assertEqual([w["id"] for w in witness["utilization_worse"]], ["x|b|", "x|c|"])
        self.assertEqual([w["id"] for w in witness["evidence_lost"]], ["x|d|"])
        self.assertEqual([w["id"] for w in witness["applicability_changed"]], ["x|e|"])
        self.assertFalse(decision["structural_benefit_established"])

    def test_duplicates_merge_and_conflicts_are_quarantined(self):
        a = rec("o1", realized="same", origins=[{"operator": "H1"}])
        b = rec("o2", realized="same", origins=[{"operator": "R1"}])
        b["checks"] = copy.deepcopy(a["checks"])
        clean, duplicates, conflicts = gp.deduplicate([a, b])
        self.assertEqual((len(clean), len(duplicates), conflicts), (1, 1, []))
        self.assertEqual([o["operator"] for o in clean[0]["origins"]], ["H1", "R1"])
        c = copy.deepcopy(b)
        c["id"] = "o3"
        c["checks"]["strength|member.strength|g1"]["v"] = 0.9
        clean, _dup, conflicts = gp.deduplicate([a, b, c])
        self.assertEqual((clean, len(conflicts)), ([], 1))
        self.assertEqual(gp.select([a, b, c], 2, RUN, 1)["selected_ids"], [])
        reused = copy.deepcopy(a)
        reused["realized_identity"] = "another design"
        with self.assertRaises(ValueError):
            gp.deduplicate([a, reused])


class Archive(unittest.TestCase):
    def test_registry_keeps_every_distinct_accepted_design_despite_material_dominance(self):
        records = [accepted(f"a{i}", genome=(i, 0), features=(i / 7, i / 7, 0.0),
                            quantities={"concrete_total_in3": 100.0 + i, "total_steel_in3": 10.0 + i}) for i in range(8)]
        records.append(rec("f", "check_failed"))
        views = gp.views(records, RUN, 5, 3)
        registry = views["accepted_registry"]
        self.assertEqual(registry["distinct_realized_designs"], 8)
        self.assertFalse(registry["material_dominance_removes_entries"])
        self.assertEqual(views["diversity_view"]["size"], 3)
        self.assertEqual(gp.views(records, RUN, 5, 3)["accepted_registry"]["distinct_realized_designs"], 8)
        for entry in registry["entries"]:
            self.assertEqual(entry["qualification"], "not_performed")
            self.assertEqual(entry["nonlinear_performance"], {"status": "not_requested"})
        self.assertTrue(views["diversity_view"]["not_training_ready_by_code_acceptance_alone"])

    def test_contradiction_across_classifications_is_quarantined_not_accepted(self):
        good = accepted("o1", realized="same")
        other = rec("o2", "check_failed", realized="same")
        views = gp.views([good, other], RUN, 1, 3)
        self.assertEqual(views["accepted_registry"]["distinct_realized_designs"], 0)
        self.assertEqual(views["quarantine"][0]["ids"], ["o1", "o2"])
        self.assertEqual(gp.select([good, other], 2, RUN, 1)["selected_ids"], [])
        bad = rec("x", "accepted", {"k|a|": check("fail", 0.2, 1.2)})
        self.assertEqual(gp.accepted_registry([bad], RUN)["refused"],
                         [{"id": "x", "duplicate_ids": ["x"], "reason": "contradictory_acceptance"}])

    def test_selection_quarantines_against_the_whole_history(self):
        # o3 (no longer a candidate) and o9 observe one realized design with different evidence.
        old, child = rec("o3", realized="same"), rec("o9", realized="same")
        child["checks"] = {"strength|member.strength|g1": check("fail", 0.7, 1.7)}
        population = rec("o1")
        decision = gp.select([population, child], 2, RUN, 1, history=[population, old, child])
        self.assertEqual(decision["selected_ids"], ["o1"])
        self.assertIn({"id": "o9", "reason": "quarantined_conflicting_evidence"}, decision["queues"])
        self.assertEqual(decision["conflicts"][0]["ids"], ["o3", "o9"])
        self.assertEqual(gp.select([population, child], 2, RUN, 1)["selected_ids"], ["o1", "o9"])

    def test_registry_lists_every_accepted_observation_it_excludes(self):
        twin_a, twin_b = rec("o3", "accepted", {"k|a|": check("fail", 0.2, 1.2)}, realized="twin"), None
        twin_b = copy.deepcopy(twin_a)
        twin_b.update(id="o4", input_identity="input-o4")
        stopped = accepted("o5")
        stopped["outcome"] = "error"
        halted = accepted("o6")
        halted["outcome"] = "infeasible"
        unrealized = accepted("o7")
        unrealized["realized_identity"] = None
        registry = gp.accepted_registry([twin_a, twin_b, stopped, halted, unrealized, accepted("o8")], RUN)
        self.assertEqual([e["id"] for e in registry["entries"]], ["o8"])
        self.assertEqual(registry["duplicates"], [])
        refused = {r["id"]: r for r in registry["refused"]}
        self.assertEqual(refused["o3"]["duplicate_ids"], ["o3", "o4"])
        self.assertEqual({k: v["reason"] for k, v in refused.items()},
                         {"o3": "contradictory_acceptance", "o5": "contradictory_acceptance",
                          "o6": "contradictory_acceptance", "o7": "contradictory_acceptance"})

    def test_unbound_records_are_refused_not_crashed_and_ids_keep_observation_order(self):
        loose = rec("o2")
        del loose["case_id"]
        self.assertEqual(gp.audit(loose, RUN)["reason"], "unbound_record")
        self.assertEqual(gp.select([rec("o1"), loose], 2, RUN, 1)["selected_ids"], ["o1"])
        gp.views([rec("o1"), loose], RUN, 1, 3)
        first, later = rec("o9999", realized="same", origins=[{"operator": "H1"}]), rec("o10000", realized="same")
        later["checks"] = copy.deepcopy(first["checks"])
        clean, duplicates, _ = gp.deduplicate([later, first])
        self.assertEqual((clean[0]["id"], duplicates[0]["ids"]), ("o9999", ["o9999", "o10000"]))
        self.assertEqual(clean[0]["origins"][0]["operator"], "H1")

    def test_every_blocked_record_is_visible_in_a_queue(self):
        error = rec("x", "invalid", valid=False)
        error.update(outcome="error", realized_identity=None)
        reasons = {q["id"]: q["reason"] for q in gp.select([rec("ok"), error], 2, RUN, 1)["queues"]}
        self.assertEqual(reasons, {"x": "analysis_recovery_required"})
        passed = rec("q", "unresolved", {"k|a|": check("pass")})
        passed.update(code_check_status="passed", quantities={"concrete_total_in3": 1.0, "total_steel_in3": None})
        odd = rec("u", "unresolved", {"k|a|": check("pass")})
        failed = rec("f", "check_failed")
        queue = {e["source"]: e for e in gp.evidence_queue([passed, odd, failed], RUN)}
        self.assertEqual(queue["material_quantities_unavailable"]["blocked_observations"], ["q"])
        self.assertEqual(queue["unexplained_unresolved_record"]["observations"], ["u"])
        self.assertNotIn("f", [i for e in queue.values() for i in e["observations"]])
        # A quantity gap is listed before it is the last blocker: on a failed record it is shown, not blocking.
        short = rec("g", "check_failed", quantities={"concrete_total_in3": 1.0, "total_steel_in3": None})
        entry = {e["source"]: e for e in gp.evidence_queue([short], RUN)}["material_quantities_unavailable"]
        self.assertEqual((entry["observations"], entry["blocked_observations"]), (["g"], []))
        # Whenever it is present, unknown rows or not.
        unknown = [{"family": "joints", "check_id": "scwb", "reason_codes": ["slab_action_review_missing"], "count": 4,
                    "locations": []}]
        both = rec("b", "unresolved", {"k|a|": check("pass")}, unknown=unknown,
                   quantities={"concrete_total_in3": 1.0, "total_steel_in3": None})
        queue = {e["source"]: e for e in gp.evidence_queue([both], RUN)}
        self.assertEqual(queue["material_quantities_unavailable"]["observations"], ["b"])
        self.assertEqual(queue["slab_action_engineering_assertion"]["observations"], ["b"])

    def test_evidence_queue_counts_every_missing_source_and_each_design_once(self):
        counts = {"joints|scwb": 112, "strength|member.strength": 1}
        sample = [{"family": "joints", "id": "scwb", "location": f"j{i}"} for i in range(40)]
        one = rec("o1", "unresolved", {"k|a|": check("pass")})
        one.update(missing=sample, missing_counts=counts)
        twin = copy.deepcopy(one)
        twin.update(id="o2", input_identity="input-o2")
        one["realized_identity"] = twin["realized_identity"] = "same"
        clash_a = rec("o3", "unresolved", {"k|a|": check("pass")}, realized="clash")
        clash_a.update(missing=sample, missing_counts=counts)
        clash_b = copy.deepcopy(clash_a)
        clash_b.update(id="o4", input_identity="input-o4", checks={"k|a|": check("unknown", 0.0, None)})
        queue = {e["source"]: e for e in gp.evidence_queue([one, twin, clash_a, clash_b], RUN)}
        self.assertEqual(set(queue), {"missing_row:joints|scwb", "missing_row:strength|member.strength"})
        self.assertEqual(queue["missing_row:joints|scwb"]["affected_rows"], 112)
        self.assertEqual(queue["missing_row:joints|scwb"]["observations"], ["o1", "o2"])
        self.assertEqual(queue["missing_row:strength|member.strength"]["affected_rows"], 1)
        # A record without the full counts falls back to its item list; a pattern miss is keyed as a pattern.
        older = rec("o5", "unresolved", {"k|a|": check("pass")})
        older["missing"] = [{"family": "joints", "id": "scwb", "location": "j1"},
                            {"pattern": "beam capacity shear at both faces of g1", "found": 0, "expected": 2}]
        sources = {e["source"] for e in gp.evidence_queue([older], RUN)}
        self.assertEqual(sources, {"missing_row:joints|scwb", "missing_row:pattern|beam capacity shear at both faces of g1"})

    def test_identical_descriptors_are_kept_as_distinct_designs(self):
        records = [accepted("a", realized="cage-1", genome=(3, 1)), accepted("b", realized="cage-2", genome=(3, 1))]
        registry = gp.accepted_registry(records, RUN)
        self.assertEqual(registry["distinct_realized_designs"], 2)
        self.assertEqual(registry["descriptor_collisions"], [{"genome": [3, 1], "ids": ["a", "b"]}])
        self.assertEqual(registry["distinct_section_profiles"], 1)

    def test_evidence_queue_groups_by_missing_source(self):
        unknown = [{"family": "joints", "check_id": "scwb", "reason_codes": ["slab_action_review_missing"], "count": 192,
                    "locations": []}]
        records = [rec("e1", "unresolved", {"k|a|": check("pass")}, unknown=unknown),
                   rec("e2", "unresolved", {"k|a|": check("pass")}, unknown=unknown),
                   rec("p", "unresolved", unknown=unknown)]
        queue = gp.evidence_queue(records, RUN)
        self.assertEqual(len(queue), 1)
        entry = queue[0]
        self.assertEqual(entry["source"], "slab_action_engineering_assertion")
        self.assertEqual(entry["affected_rows"], 576)
        self.assertEqual(entry["blocked_observations"], ["e1", "e2"])
        self.assertEqual(entry["observations"], ["e1", "e2", "p"])
        self.assertIn("never sets it", entry["action"])

    def test_recovery_queue_never_diagnoses_weakness(self):
        error = rec("x", "invalid", valid=False)
        error.update(outcome="error", realized_identity=None, error={"stage": "post_evaluation"},
                     artifacts={"evaluation": "evaluations/x.json"})
        stopped = rec("s", "unresolved", complete=False)
        stopped.update(outcome="infeasible", realized_identity=None, failed_stage="gravity_analysis")
        rows = {r["id"]: r for r in gp.recovery_queue([error, stopped, rec("ok")])}
        self.assertEqual((rows["x"]["kind"], rows["s"]["kind"], rows["s"]["stage"]),
                         ("evaluation_exception", "evaluator_stage_stopped", "gravity_analysis"))
        # An exception's stage, and what it already wrote, are where analysis recovery starts.
        self.assertEqual((rows["x"]["stage"], rows["x"]["artifacts"]), ("post_evaluation", {"evaluation": "evaluations/x.json"}))
        self.assertNotIn("ok", rows)
        self.assertEqual(gp.audit(stopped, RUN)["reason"], "analysis_recovery_required")

    def test_unroutable_failures_name_every_failure_no_operator_can_act_on(self):
        def diagnostic(check_id, reason, valid=True, problems=(), deps=None, status="fail"):
            return {"reason": reason, "status": status, "evidence_valid": valid, "evidence_problems": list(problems),
                    "dependencies": deps if deps is not None else {k: True for k in gp.RULES.get(reason, (None, ()))[1]},
                    "source": {"family": "capacity", "check_id": check_id}}
        mixed = diagnostic("gravity.column_capacity_shear", "column_capacity_shear", False,
                           ["the gravity column row compares the x-direction Ve with another direction's hoop phi Vn"])
        orphan = diagnostic("column.hoops_selected", None, False, ["no mutation reason is declared for this check"])
        waiting = diagnostic("column.capacity_shear_section", "column_capacity_shear",
                             deps={"probable_equilibrium": True, "transverse_selection": False})
        routable = diagnostic("column.capacity_shear_section", "column_capacity_shear")
        passing = diagnostic("beam.capacity_shear_section", "beam_capacity_shear", status="pass")
        passing_unproven = diagnostic("beam.capacity_shear_section", "beam_capacity_shear", False, ["synthetic"],
                                      status="pass")                       # passes: no failure to list, valid or not
        records = []
        for rid, items, realized in (("a", [mixed, orphan, routable, passing, passing_unproven], "r1"), ("b", [mixed], "r2"),
                                     ("c", [mixed], "r2"), ("d", [waiting], "r3")):
            r = rec(rid, "check_failed", realized=realized)
            r["diagnostics"] = {"diagnostics": copy.deepcopy(items)}
            records.append(r)
        done = accepted("e")
        done["diagnostics"] = {"diagnostics": [copy.deepcopy(mixed)]}       # an accepted record is never listed
        foreign = rec("f", "check_failed", realized="r4")
        foreign.update(basis_id="another basis", diagnostics={"diagnostics": [copy.deepcopy(mixed)]})   # another run
        rows = {(e["kind"], e["check"]): e for e in gp.unroutable_failures(records + [done, foreign], RUN)}
        self.assertEqual(set(rows), {("invalid_evidence", "capacity|gravity.column_capacity_shear"),
                                     ("no_operator", "capacity|column.hoops_selected"),
                                     ("dependencies_incomplete", "capacity|column.capacity_shear_section")})
        shear = rows[("invalid_evidence", "capacity|gravity.column_capacity_shear")]
        # Each realized design once (b and c are one design with identical evidence: one row, both observations).
        self.assertEqual((shear["observations"], shear["rows"]), (["a", "b", "c"], 2))
        self.assertTrue(any("another direction" in p for p in shear["problems"]))
        waiting_entry = rows[("dependencies_incomplete", "capacity|column.capacity_shear_section")]
        # Only the waiting design: a routable failure of the same check (in a) is never listed.
        self.assertEqual((waiting_entry["problems"], waiting_entry["observations"]),
                         (["dependency not shown: transverse_selection"], ["d"]))
        self.assertIn("unroutable_failures", gp.views(records, RUN, 1, 4))

    def test_unroutable_strength_rows_are_filed_by_what_blocks_them(self):
        """A strength row that did not authenticate is invalid evidence whatever its items' reasons; the analysis-shear
        component has no operator and names its component; a row and its components count as one row."""
        def item(component, reason, problems, location="g1"):
            return {"reason": reason, "status": "fail", "evidence_valid": not problems, "evidence_problems": list(problems),
                    "dependencies": {"member_actions": True, "section_strength": True},
                    "source": {"family": "strength", "check_id": "member.strength", "location": location,
                               "component": component}}
        mismatch = "the row's DCR does not follow from its components"
        diagnostics = [item("flexure_positive", "beam_flexure", [mismatch]), item(None, None, [mismatch]),
                       item("shear", None, ["analysis shear (phi Vn < Vu) is not a reason of the operator contract"]),
                       item(None, None, ["no strength evidence for the group of this row"], location="g2")]
        r = rec("s", "check_failed")
        r["diagnostics"] = {"diagnostics": diagnostics}
        rows = {(e["kind"], e["check"]): e for e in gp.unroutable_failures([r], RUN)}
        self.assertEqual(set(rows), {("invalid_evidence", "strength|member.strength"),
                                     ("no_operator", "strength|member.strength#shear")})
        invalid = rows[("invalid_evidence", "strength|member.strength")]
        self.assertEqual(invalid["rows"], 2)                                 # g1 (row and component) and g2
        self.assertEqual(invalid["problems"], sorted([mismatch, "no strength evidence for the group of this row"]))
        self.assertEqual(rows[("no_operator", "strength|member.strength#shear")]["rows"], 1)

    def test_a_whole_row_strength_item_is_invalid_evidence_however_it_is_marked(self):
        r = rec("w", "check_failed")
        r["diagnostics"] = {"diagnostics": [
            {"reason": reason, "status": "fail", "evidence_valid": True, "evidence_problems": [],
             "dependencies": {"member_actions": True, "section_strength": True},
             "source": {"family": "strength", "check_id": "member.strength", "location": location, "component": None}}
            for reason, location in ((None, "g1"), ("beam_flexure", "g2"))]}
        self.assertEqual([(e["kind"], e["check"], e["rows"]) for e in gp.unroutable_failures([r], RUN)],
                         [("invalid_evidence", "strength|member.strength", 2)])

    def test_missing_hoops_point_to_the_upstream_hoop_selection(self):
        unknown = [{"family": "capacity", "check_id": "gravity.column_capacity_shear", "reason_codes": ["hoops_not_selected"],
                    "count": 1, "locations": ["g1"]}]
        queue = gp.evidence_queue([rec("h", "unresolved", unknown=unknown)], RUN)
        self.assertEqual([(e["source"], e["reason_codes"]) for e in queue],
                         [("upstream_hoop_selection", ["hoops_not_selected"])])
        self.assertIn("not evidence to obtain separately", queue[0]["action"])


class NormalizedChecks(unittest.TestCase):
    EXPECTED = {"exact": {("strength", "member.strength", "g1"), ("slab", "slab.reinforcement_designed", "")},
                "conditional": [{"trigger": ("slab", "slab.reinforcement_designed", ""),
                                 "requires": [("slab", "slab_bar_development", "")]}],
                "patterns": [{"family": "detailing", "group": "g1", "minimum": 1},
                             {"family": "joints", "id_in": ("beam_capacity_shear_left", "beam_capacity_shear_right"),
                              "prefix": "g1/member_", "suffix": "/family_a", "exactly": 2}],
                "not_applicable": []}

    def rows(self, slab_status="pass", member=16):
        def row(family, check_id, location, status="pass"):
            base = {"family": family, "id": check_id, "location": location, "clause": "c", "status": status,
                    "units": "", "details": {}}
            if status == "pass":
                base.update(demand=1.0, capacity=2.0, comparison="<=")
            elif status == "fail":
                base.update(demand=3.0, capacity=2.0, comparison="<=")
            else:
                base.update(demand=None, capacity=None)
            return base
        return [row("strength", "member.strength", "g1", "fail"), row("slab", "slab.reinforcement_designed", "", slab_status),
                row("detailing", "beam.minimum_width", "g1"),
                row("joints", "beam_capacity_shear_left", f"g1/member_{member}/family_a"),
                row("joints", "beam_capacity_shear_right", f"g1/member_{member}/family_a")]

    def test_canonical_keys_missing_rows_and_inventory(self):
        checks, inventory = gp.normalize_checks(self.rows(), self.EXPECTED)
        self.assertIn("joints|beam_capacity_shear_left|g1/family_a", checks)
        self.assertEqual(checks["slab|slab_bar_development|"]["s"], "missing")
        self.assertAlmostEqual(checks["strength|member.strength|g1"]["v"], 0.5)
        self.assertAlmostEqual(checks["strength|member.strength|g1"]["u"], 1.5)
        # another governing member gives the same keys; a triggered conditional row changes the inventory
        moved, same = gp.normalize_checks(self.rows(member=22), self.EXPECTED)
        self.assertEqual(set(moved), set(checks))
        self.assertEqual(same["inventory_id"], inventory["inventory_id"])
        untriggered, other = gp.normalize_checks(self.rows(slab_status="not_evaluated"), self.EXPECTED)
        self.assertNotIn("slab|slab_bar_development|", untriggered)
        self.assertNotEqual(other["inventory_id"], inventory["inventory_id"])
        self.assertEqual(inventory["known_items"], 5)


if __name__ == "__main__":
    unittest.main()
