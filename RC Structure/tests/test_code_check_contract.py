"""The code-check contract of a grouped candidate (Design.Code_Check_Contract; code-check audit of the genetic search,
2026-10-30).

Regressions of findings F01 (an all-unknown strong-column inventory reported as passing) and F02 (records that lack
or forge evidence passing the search boundary). Rows are synthetic and the inventory is derived from the fixture
frame's own topology: these are checks of the acceptance logic, not of any structural calculation.
"""
import copy
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_fixture as cf                                                    # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402

from Design import Code_Check_Contract as contract                               # noqa: E402
from Design import Genetic_Search as ga                                          # noqa: E402
from Design import Grouped_Record as gr                                          # noqa: E402
from Design.SMRF_Common import (action_inventory_problems, check_row_problems, make_check,  # noqa: E402
                                not_evaluated, summarize_checks)
from Model import Member_Groups as mg                                            # noqa: E402

CASE_0009 = Path(__file__).resolve().parents[2] / "research" / "ga_heuristics_20261030" / "experiments" / "E6_data" \
    / "case_0009" / "baseline" / "raw.json"
CASE_0009_SHA256 = "30e26587e2fce3cd5fbe115244e875d7365b716df3649219006247501a08c723"


def scwb_row(location, status):
    if status == "pass":
        return make_check("scwb", "ACI 318-19 18.7.3.2", 120.0, 200.0, "<=", "kip-in", location,
                          details={"ratio_provided": 1.6})
    if status == "fail":
        return make_check("scwb", "ACI 318-19 18.7.3.2", 300.0, 200.0, "<=", "kip-in", location,
                          details={"ratio_provided": 0.8})
    row = not_evaluated("scwb", "ACI 318-19 18.7.3.2", "Every beam needs an explicit developed slab contribution or "
                                                       "absence basis.", location)
    row["details"]["reason_codes"] = ["slab_action_review_missing"]
    return row


class StrongColumnSummary(unittest.TestCase):
    """F01: all_pass means every required joint, direction and sway sign present once and passing."""
    REQUIRED = [f"joint_{n}/{axis}/{sign}" for n in (13, 14) for axis in ("x", "y") for sign in ("positive", "negative")]

    def summary(self, statuses, required=REQUIRED):
        return contract.scwb_summary([scwb_row(loc, s) for loc, s in statuses.items()], required)

    def test_unknown_only_is_not_a_pass(self):
        result = self.summary({loc: "not_evaluated" for loc in self.REQUIRED})
        self.assertFalse(result["all_pass"])
        self.assertTrue(result["no_known_failure"])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["min_ratio_provided"])                     # no ratio is made up for unknown rows
        self.assertEqual(result["unknown_reason_codes"], ["slab_action_review_missing"])

    def test_mixed_pass_and_unknown_is_not_a_pass(self):
        statuses = {loc: "pass" for loc in self.REQUIRED}
        statuses[self.REQUIRED[0]] = "not_evaluated"
        self.assertFalse(self.summary(statuses)["all_pass"])

    def test_a_failure_stays_visible_beside_unknown_rows(self):
        statuses = {loc: "not_evaluated" for loc in self.REQUIRED}
        statuses[self.REQUIRED[1]] = "fail"
        result = self.summary(statuses)
        self.assertFalse(result["all_pass"])
        self.assertFalse(result["no_known_failure"])
        self.assertEqual(result["counts"]["fail"], 1)
        self.assertEqual(result["min_ratio_provided"], 0.8)

    def test_an_empty_list_against_a_required_inventory_is_missing_not_vacuous(self):
        result = contract.scwb_summary([], self.REQUIRED)
        self.assertFalse(result["all_pass"])
        self.assertEqual(result["missing_count"], len(self.REQUIRED))

    def test_a_complete_passing_inventory_passes(self):
        result = self.summary({loc: "pass" for loc in self.REQUIRED})
        self.assertTrue(result["all_pass"])
        self.assertTrue(result["complete"])

    def test_a_legitimately_inapplicable_direction_is_not_required(self):
        # The y direction of joint 14 is not part of the SFRS: it is not in the required inventory and not asked for.
        required = [loc for loc in self.REQUIRED if not loc.startswith("joint_14/y")]
        result = self.summary({loc: "pass" for loc in required}, required)
        self.assertTrue(result["all_pass"])
        result = self.summary({loc: "pass" for loc in self.REQUIRED}, required)       # a row the design does not have
        self.assertFalse(result["all_pass"])
        self.assertTrue(result["unexpected"])


class RowValidation(unittest.TestCase):
    """F02 and the gate audit item 3: one row read as the evidence it claims."""

    def test_a_pass_that_contradicts_its_numbers_is_flagged(self):
        forged = {"id": "x", "clause": "test", "status": "pass", "demand": 2.0, "capacity": 1.0, "comparison": "<="}
        self.assertIn("status_contradicts_numbers", check_row_problems(forged))
        self.assertFalse(summarize_checks([forged])["accepted"])
        self.assertEqual(summarize_checks([forged])["contradictory_checks"], ["x"])

    def test_a_pass_without_numbers_is_malformed(self):
        self.assertIn("status_without_finite_numbers", check_row_problems({"id": "x", "clause": "test", "status": "pass"}))
        self.assertFalse(summarize_checks([{"id": "x", "clause": "test", "status": "pass"}])["accepted"])

    def test_a_declared_producer_tolerance_is_honoured(self):
        row = {"id": "x", "clause": "c", "status": "pass", "demand": 0.9999999, "capacity": 1.0, "comparison": ">=",
               "details": {"numerical_tolerance_in": 1e-6}}
        self.assertEqual(check_row_problems(row), [])

    def test_unknown_input_kinds_are_kept_distinguishable(self):
        kinds = {name: make_check("x", "c", value, 1.0)["details"].get("input_problems", {}).get("demand")
                 for name, value in (("none", None), ("nan", math.nan), ("inf", math.inf), ("bool", True), ("text", "a"))}
        self.assertEqual(kinds, {"none": "missing", "nan": "nan", "inf": "infinite", "bool": "boolean", "text": "non_numeric"})
        zero = make_check("x", "c", 5.0, 0.0)
        self.assertEqual(zero["status"], "fail")                     # a known zero capacity with demand is a failure
        self.assertTrue(json.dumps(make_check("x", "c", math.nan, 1.0), allow_nan=False))

    def test_a_valid_numeric_zero_is_not_missing(self):
        self.assertEqual(make_check("x", "c", 0.0, 1.0)["status"], "pass")


class ActionInventory(unittest.TestCase):
    """SCWB audit item 5: duplicate, missing, failed and wrongly referenced cases are reported, never overwritten."""

    def action(self, cid, **changes):
        return {"id": cid, "analysis_succeeded": True, "axial_reference": "joint_faces",
                "members": {"1": {"axial_i_kip": 10.0, "axial_j_kip": 9.0}}, **changes}

    def test_a_valid_inventory(self):
        by_id, problems = action_inventory_problems([self.action("a"), self.action("b")], ["a", "b"], [1])
        self.assertEqual((sorted(by_id), problems), (["a", "b"], []))

    def test_each_defect_is_reported(self):
        cases = {"duplicate": ([self.action("a"), self.action("a"), self.action("b")], "duplicate combination a"),
                 "missing": ([self.action("a")], "missing combination b"),
                 "failed": ([self.action("a"), self.action("b", analysis_succeeded=None)], "did not confirm"),
                 "reference": ([self.action("a"), self.action("b", axial_reference="centerline")], "axial_reference"),
                 "nonfinite": ([self.action("a"), self.action("b", members={"1": {"axial_i_kip": math.nan,
                                                                                  "axial_j_kip": 1.0}})], "nonfinite")}
        for name, (actions, text) in cases.items():
            with self.subTest(name):
                _by_id, problems = action_inventory_problems(actions, ["a", "b"], [1])
                self.assertTrue(any(text in p for p in problems), problems)


class Inventory(unittest.TestCase):
    """F02: the required inventory comes from the final design's own topology and system assignment."""

    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def test_band_policy_requires_every_joint_direction_and_sign(self):
        state = gf.uniform_state()
        expected = contract.expected_inventory(state)
        joints = gf.NZ * (gf.NX + 1) * (gf.NY + 1)
        self.assertEqual(len(expected["scwb_required"]), joints * 2 * 2)
        self.assertEqual(expected["scwb_not_required"], {})

    def test_system_policy_requires_only_the_sfrs_directions_with_a_basis_for_the_rest(self):
        state, _ = gr.seed_from_sections((26, 26, 6), (16, 30, 5), policy_id="system_three_story_v1")
        expected = contract.expected_inventory(state)
        for location in expected["scwb_required"]:
            joint, axis, _sign = location.split("/")
            self.assertNotIn(f"{joint}/{axis}", expected["scwb_not_required"])
        self.assertTrue(expected["scwb_not_required"])
        self.assertTrue(all("not part of the seismic force-resisting system" in basis
                            for basis in expected["scwb_not_required"].values()))
        gravity = [gid for gid in state.groups if state.system_of(gid) == mg.SYSTEM_GRAVITY and not state.designs[gid].is_column]
        axial = {loc for fam, name, loc in expected["exact"] if name == "beam.axial_force_applicability"}
        self.assertTrue(gravity)
        self.assertFalse(set(gravity) & axial)                     # gravity beams keep their own 18.14 axial branch
        self.assertTrue(axial)

    def test_a_geometry_that_is_not_the_designs_is_refused(self):
        state = gf.uniform_state()
        import Structure_Parameters as sp
        sp.NUM_FLOOR = gf.NZ + 1
        with self.assertRaisesRegex(ValueError, "geometry"):
            contract.expected_inventory(state)


class ContractResult(unittest.TestCase):
    """F01 / F02: the authoritative result and the search boundary."""

    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.state = gf.uniform_state()
        self.expected = contract.expected_inventory(self.state)
        self.scwb = cf.scwb_keys(self.state)

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def result(self, rows):
        return contract.code_check_result(rows, self.expected)

    def test_a_complete_passing_inventory_is_accepted(self):
        result = self.result(cf.rows_for(self.state))
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["accepted"])
        record = cf.evaluation_record(self.state)
        assessed = contract.assess_evaluation(record, submitted_identity=self.state.identity())
        self.assertEqual((assessed["classification"], assessed["valid"], record["feasible"]), ("accepted", True, True))

    def test_unknown_strong_column_evidence_is_incomplete_with_its_reason(self):
        rows = cf.rows_for(self.state, statuses={k: ("not_evaluated", "slab_action_review_missing") for k in self.scwb})
        result = self.result(rows)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["unknown_reason_codes"], ["slab_action_review_missing"])
        self.assertEqual(result["counts"]["unknown"], len(self.scwb))

    def test_a_failure_with_unknown_rows_is_failed_and_both_stay_listed(self):
        statuses = {k: "not_evaluated" for k in self.scwb[1:]}
        statuses[self.scwb[0]] = "fail"
        result = self.result(cf.rows_for(self.state, statuses=statuses))
        self.assertEqual(result["status"], "failed")
        self.assertEqual([(f["id"], f["location"]) for f in result["known_failures"]], [(self.scwb[0][1], self.scwb[0][2])])
        self.assertEqual(result["counts"]["unknown"], len(self.scwb) - 1)
        record = cf.evaluation_record(self.state, rows=cf.rows_for(self.state, statuses=statuses))
        self.assertEqual(ga.assessment(record, None)[0], "unresolved")       # unknown evidence: not a complete failure

    def test_an_omitted_joint_or_direction_is_missing(self):
        joint, axis = self.scwb[0][2].split("/")[:2]
        dropped = [k for k in self.scwb if k[2].startswith(f"{joint}/{axis}/")]
        result = self.result(cf.rows_for(self.state, drop=set(dropped)))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len([m for m in result["missing"] if m.get("id") == "scwb"]), 2)

    def test_every_required_row_dropped_alone_is_missing(self):
        for key in sorted(self.expected["exact"])[::37]:
            with self.subTest(key=key):
                result = self.result(cf.rows_for(self.state, drop={key}))
                self.assertNotEqual(result["status"], "passed")
                self.assertIn({"family": key[0], "id": key[1], "location": key[2]}, result["missing"])

    def test_a_duplicated_row_is_invalid(self):
        rows = cf.rows_for(self.state)
        result = self.result(rows + [dict(rows[0])])
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(result["duplicates"])

    def test_a_contradictory_numeric_pass_is_invalid(self):
        rows = cf.rows_for(self.state)
        forged = dict(rows[0], status="pass", demand=2.0, capacity=1.0, comparison="<=")
        result = self.result([forged] + rows[1:])
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(result["invalid"])

    def test_a_row_the_design_does_not_have_is_invalid_but_an_unrelated_extra_row_is_not_demanded(self):
        rows = cf.rows_for(self.state)
        self.assertEqual(self.result(rows + [cf._row("joints", "scwb", "joint_9999/x/positive")])["status"], "invalid")
        extra = cf._row("detailing", "detailing.unrelated_note", "s01_02__column__edge/extra")
        self.assertEqual(self.result(rows + [extra])["status"], "passed")

    def test_a_required_row_cannot_hide_behind_a_review_label(self):
        # The case_0009 situation relabelled as "awaiting review": every required strong-column row claims the slab
        # assertion review. Only the declared review-dependent checks may wait on review, and only on their own item.
        from Design.SMRF_Common import awaiting_review
        rows = cf.rows_for(self.state, drop=set(self.scwb))
        masked = [contract.compact_row("joints", awaiting_review("scwb", "ACI 318-19 18.7.3.2", "slab not asserted",
                                                                 "slab_action_assertions", key[2])) for key in self.scwb]
        result = self.result(rows + masked)
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["accepted"])
        record = cf.evaluation_record(self.state, rows=rows + masked)
        self.assertFalse(record["feasible"])
        self.assertEqual(contract.assess_evaluation(record)["classification"], "invalid")
        # A declared review dependency on its own item is accepted as a review item.
        key = ("regularity", "demands.vertical_strength_regularity", "")
        permitted = [r for r in cf.rows_for(self.state) if (r["family"], r["id"], r["location"]) != key]
        permitted.append(contract.compact_row("regularity", awaiting_review(
            "demands.vertical_strength_regularity", "ASCE 7-22 Table 12.3-2", "M1", "story_strength_model_verified")))
        self.assertEqual(self.result(permitted)["status"], "passed")

    def test_a_conditionally_required_slab_row_cannot_hide_behind_a_review_label(self):
        from Design.SMRF_Common import awaiting_review
        key = ("slab", "slab_bar_development", "")
        rows = [r for r in cf.rows_for(self.state) if (r["family"], r["id"], r["location"]) != key]
        rows.append(contract.compact_row("slab", awaiting_review("slab_bar_development", "ACI 318-19 25.4.2.4", "x",
                                                                 "anything_at_all")))
        self.assertEqual(self.result(rows)["status"], "invalid")

    def test_a_failing_duplicate_stays_a_known_failure(self):
        rows = cf.rows_for(self.state)
        failing = cf._row(*self.scwb[0], status="fail")
        result = self.result(rows + [failing])
        self.assertEqual(result["status"], "invalid")
        self.assertIn((self.scwb[0][1], self.scwb[0][2]), [(f["id"], f["location"]) for f in result["known_failures"]])

    def test_review_items_are_listed_and_do_not_fail_or_pass_the_computed_checks(self):
        rows = cf.rows_for(self.state)
        # A neutral, non-required review item: the fire row is a declared scope exclusion since 2026-11-01
        # (tests/test_slab_structural_gates.py), never a review.
        review = not_evaluated("slab_review_fixture", "fixture clause", "fixture review")
        review["details"]["review_dependency"] = {"item": "fixture_review", "kind": "engineering_review"}
        result = self.result(rows + [contract.compact_row("slab", review)])
        self.assertEqual(result["status"], "passed")
        self.assertEqual([r["item"] for r in result["review_dependencies"]], ["fixture_review"])


class SearchBoundary(unittest.TestCase):
    """F02: what the genetic search's boundary refuses before ranking or archiving a candidate."""

    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.state = gf.uniform_state()
        self.record = cf.evaluation_record(self.state)

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def assess(self, record, submitted=None):
        return contract.assess_evaluation(record, submitted_identity=submitted or self.state.identity())

    def test_the_reference_record_is_accepted(self):
        self.assertEqual(self.assess(self.record)["classification"], "accepted")

    def test_each_required_field_deleted_is_invalid(self):
        for key in ("state", "result_identity", "seed_identity", "code_checks", "code_check", "constraints",
                    "candidate_screen_passed", "evidence"):
            with self.subTest(key=key):
                record = {k: v for k, v in self.record.items() if k != key}
                assessed = self.assess(record)
                self.assertEqual(assessed["classification"], "invalid")
                self.assertFalse(assessed["accepted"])

    def test_completion_flags_must_be_explicit_booleans(self):
        for flag in contract.COMPLETION_FLAGS:
            for value in (None, 1, "true"):
                with self.subTest(flag=flag, value=value):
                    record = copy.copy(self.record)
                    record["constraints"] = {**self.record["constraints"], flag: value}
                    self.assertEqual(self.assess(record)["classification"], "invalid")
            record = copy.copy(self.record)
            record["constraints"] = {k: v for k, v in self.record["constraints"].items() if k != flag}
            self.assertEqual(self.assess(record)["classification"], "invalid")

    def test_a_false_completion_flag_is_unresolved_and_not_feasible(self):
        record = cf.evaluation_record(self.state, constraints={"reinforcement_settled": False})
        self.assertFalse(record["feasible"])
        self.assertEqual(self.assess(record)["classification"], "unresolved")

    def test_another_candidates_evidence_or_identity_is_invalid(self):
        other = gf.changed_state()
        other_record = cf.evaluation_record(other)
        swapped = {**self.record, "code_checks": other_record["code_checks"], "code_check": other_record["code_check"]}
        self.assertEqual(self.assess(swapped)["classification"], "invalid")
        self.assertEqual(self.assess({**self.record, "result_identity": other.identity()})["classification"], "invalid")
        self.assertEqual(self.assess(self.record, submitted=other.identity())["classification"], "invalid")
        realized = copy.deepcopy(self.record)
        realized["code_checks"]["realized_identity"] = "0" * 64
        self.assertEqual(self.assess(realized)["classification"], "invalid")

    def test_a_reported_result_that_does_not_follow_from_the_rows_is_invalid(self):
        rows = cf.rows_for(self.state, statuses={cf.scwb_keys(self.state)[0]: "not_evaluated"})
        record = cf.evaluation_record(self.state, rows=rows)
        forged = {**record, "code_check": self.record["code_check"], "feasible": True}
        self.assertEqual(self.assess(forged)["classification"], "invalid")
        self.assertEqual(self.assess(record)["classification"], "unresolved")

    def test_a_final_state_with_reselected_cages_is_normal_but_other_sections_are_not(self):
        from dataclasses import replace
        gid = sorted(self.state.groups)[0]
        design = self.state.designs[gid]
        recaged = self.state.with_designs({gid: replace(design, top_bars=design.top_bars + 1,
                                                        stirrup_spacing_in=design.stirrup_spacing_in - 1.0)})
        record = cf.evaluation_record(recaged, seed_identity=self.state.identity())
        assessed = contract.assess_evaluation(record, submitted_identity=self.state.identity(), submitted_state=self.state)
        self.assertEqual(assessed["classification"], "accepted")
        # Another candidate's complete, self-consistent record under the submitted identity: other sections.
        other = cf.evaluation_record(gf.changed_state(), seed_identity=self.state.identity())
        assessed = contract.assess_evaluation(other, submitted_identity=self.state.identity(), submitted_state=self.state)
        self.assertEqual(assessed["classification"], "invalid")
        self.assertTrue(any("differ from the submitted design" in p for p in assessed["problems"]))
        regraded = self.state.with_designs({gid: replace(design, fc_ksi=design.fc_ksi + 1.0)})
        assessed = contract.assess_evaluation(cf.evaluation_record(regraded, seed_identity=self.state.identity()),
                                              submitted_identity=self.state.identity(), submitted_state=self.state)
        self.assertEqual(assessed["classification"], "invalid")

    def test_the_ga_ranks_evidence_before_exceedance_and_stops_last(self):
        statuses = {k: "not_evaluated" for k in cf.scwb_keys(self.state)[1:]}
        failing = {**statuses, cf.scwb_keys(self.state)[0]: "fail"}
        hidden = {**statuses, cf.scwb_keys(self.state)[0]: "not_evaluated"}
        _, with_failure = ga.assessment(cf.evaluation_record(self.state, rows=cf.rows_for(self.state, statuses=failing)), None)
        _, failure_hidden = ga.assessment(cf.evaluation_record(self.state, rows=cf.rows_for(self.state, statuses=hidden)), None)
        _, stopped = ga.assessment({"status": "infeasible", "failed_stage": "capacity_design",
                                    "seed_identity": self.state.identity()}, None)
        self.assertLess(with_failure, failure_hidden)          # turning a failure into an unknown never improves it
        self.assertLess(failure_hidden, stopped)               # a stopped evaluation never outranks an evaluated one
        forged = cf.evaluation_record(self.state, rows=cf.rows_for(self.state) + [cf.rows_for(self.state)[0]])
        classification, fitness = ga.assessment(forged, None)
        self.assertEqual(classification, "invalid")
        self.assertLess(stopped, fitness)


class Case0009Replay(unittest.TestCase):
    """F01 replay: the saved case_0009 candidate (288 unknown strong-column rows, screen passed) stays unaccepted.

    The research artifact is read, never written; its SHA-256 is the one the audit recorded.
    """

    @classmethod
    def setUpClass(cls):
        if not CASE_0009.exists():
            raise unittest.SkipTest(f"research artifact not present: {CASE_0009}")
        data = CASE_0009.read_bytes()
        cls.sha = hashlib.sha256(data).hexdigest()
        cls.raw = json.loads(data)

    def test_the_artifact_is_unchanged(self):
        self.assertEqual(self.sha, CASE_0009_SHA256)

    def test_its_screen_result_is_preserved_and_its_strong_column_evidence_is_incomplete(self):
        self.assertTrue(self.raw["feasible"])                          # the historical v1 screen, as saved
        self.assertTrue(self.raw["constraints"]["candidate_screen_passed"])
        rows = self.raw["evidence"]["joint_checks"]
        summary = contract.scwb_summary(rows)
        self.assertEqual(summary["counts"], {"pass": 0, "fail": 0, "not_evaluated": 288})
        self.assertFalse(summary["all_pass"])
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["no_known_failure"])

    def test_the_corrected_admission_does_not_accept_it(self):
        assessed = contract.assess_evaluation(self.raw)
        self.assertFalse(assessed["accepted"])
        self.assertEqual(assessed["classification"], "invalid")        # a v1 record: no final design object, no contract rows
        self.assertNotEqual(ga.assessment(self.raw, None)[0], "feasible")


if __name__ == "__main__":
    unittest.main()
