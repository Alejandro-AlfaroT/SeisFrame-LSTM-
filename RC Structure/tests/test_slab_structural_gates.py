"""Structural slab gates and the declared fire-resistance exclusion (2026-11-01).

  P  producer: the two-way shear path and the slab-column local steel (ACI 318-19 8.6.1.2) are each resolved on
     their own: a pass only with the beam-supported premise and that item's own assertion, otherwise an unknown row
     with its own reason code (never a non-blocking review); the fire row is a declared scope exclusion;
  E  slab-strength eligibility: an unresolved gate is an unknown check, the exclusion is listed apart;
  C  contract: both gates are required completion rows, each blocks acceptance on its own, a known failure beside
     an unresolved gate classifies as unresolved, not as a check failure; the fire row is skipped only as the
     declared exclusion and is invalid as a pass, fail or review;
  G  the search's normalized checks skip the excluded row;
  Q  qualification: no verification flag passes the excluded item or the local-steel gate, the exclusion is listed
     apart, and a passing fire row blocks;
  A  assertions: the local-steel flag lives on SlabActionAssertions, outside all_asserted; a profile that predates
     it loads with its identity unchanged, and a non-boolean value is refused.

No solve.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_fixture as cf                                                    # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402
from test_smrf_slab_reinforcement import CORNERS, evidence, inputs               # noqa: E402

from Design import Code_Check_Contract as contract                               # noqa: E402
from Design import Design_Evidence_Profile as dep                                # noqa: E402
from Design import Genetic_Population as population                              # noqa: E402
from Design.Config import DesignConfig, IndependentVerification, SlabActionAssertions  # noqa: E402
from Design.SMRF_Common import (SCOPE_EXCLUSIONS, awaiting_review, make_check, not_evaluated,  # noqa: E402
                                scope_exclusion, split_scope_exclusions)
from Design.SMRF_Qualification import _apply_verification_assertions, qualification_scope  # noqa: E402
from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement, slab_strength_eligibility  # noqa: E402

RC = Path(__file__).resolve().parents[1]
EXAMPLE = RC / "tools/genetic_design_search/evidence_profiles/case_0007_slab_refinement_unreviewed.json"
EXAMPLE_IDENTITY = "59d60994a8d2ca71d61774c55e7c87089c550f5cc898311aef0e3454a3e3bdb5"   # recorded before the flag
CONTEXT = {"clear_span_x_in": 102.0, "clear_span_y_in": 102.0, "beam_width_in": 10.0, "alpha_f_min": 5.6,
           "alpha_f_l2_l1_min": 5.6, "thickness_screen_passed": True, "column_core_width_in": 13.0,
           "two_way_shear_path_assessed": True, "slab_column_local_steel_assessed": True,
           "columns_at_beam_intersections": True, "beam_clear_cover_in": 1.5, "beam_hoop_diameter_in": 0.5,
           "fc_beam_ksi": 4.0, "perimeter_hook_corners": CORNERS}
TWO_WAY, LOCAL, FIRE = "slab_two_way_shear_applicability", "slab_column_local_minimum_steel", "slab_fire_resistance"


def design(**changes):
    result = design_slab_reinforcement(inputs(), evidence(), context={**CONTEXT, **changes})
    return result, {c["id"]: c for c in result["checks"] if c["location"] == ""}


def fire_exclusion():
    return scope_exclusion(FIRE, "ACI 318-19 4.11", "excluded", "fire_resistance")


class Producer(unittest.TestCase):
    def test_both_assessed_on_a_beam_supported_floor_pass_and_fire_stays_excluded(self):
        result, rows = design()
        self.assertEqual((rows[TWO_WAY]["status"], rows[LOCAL]["status"]), ("pass", "pass"))
        self.assertIn("not applicable by the beam-supported premise", rows[LOCAL]["details"]["basis"])
        self.assertEqual(result["summary"]["not_evaluated"], [FIRE])
        self.assertEqual(rows[FIRE]["details"]["scope_exclusion"]["item"], "fire_resistance")
        self.assertNotIn("review_dependency", rows[FIRE]["details"])

    def test_each_gate_is_resolved_on_its_own_assertion(self):
        _result, rows = design(two_way_shear_path_assessed=False)
        self.assertEqual(rows[LOCAL]["status"], "pass")
        self.assertEqual((rows[TWO_WAY]["status"], rows[TWO_WAY]["details"]["reason_code"]),
                         ("not_evaluated", "two_way_shear_path_assessment_required"))
        _result, rows = design(slab_column_local_steel_assessed=False)
        self.assertEqual(rows[TWO_WAY]["status"], "pass")
        self.assertEqual((rows[LOCAL]["status"], rows[LOCAL]["details"]["reason_code"]),
                         ("not_evaluated", "slab_column_local_steel_assessment_required"))
        for row in (design(two_way_shear_path_assessed=False)[1][TWO_WAY], rows[LOCAL]):
            self.assertNotIn("review_dependency", row["details"])     # a structural item, never a non-blocking review

    def test_without_the_beam_supported_premise_no_assertion_settles_either_gate(self):
        for changes in ({"alpha_f_l2_l1_min": 0.8}, {"alpha_f_min": 0.5}, {"columns_at_beam_intersections": False}):
            with self.subTest(changes=changes):
                _result, rows = design(**changes)
                self.assertEqual(rows[TWO_WAY]["details"]["reason_code"], "slab_two_way_shear_not_implemented")
                self.assertEqual(rows[LOCAL]["details"]["reason_code"], "slab_column_local_steel_not_implemented")

    def test_without_a_floor_context_the_local_steel_gate_is_unresolved(self):
        rows = {c["id"]: c for c in design_slab_reinforcement(inputs(), evidence())["checks"]}
        self.assertEqual(rows[LOCAL]["details"]["reason_code"], "slab_column_local_steel_assessment_required")
        self.assertIn("scope_exclusion", rows[FIRE]["details"])

    def test_without_a_layout_the_local_steel_gate_still_reads_the_floor(self):
        # Demands not accepted (no strip evidence): the saved floor context still decides the gate's reason.
        weak = design_slab_reinforcement(inputs(), {"verified": False}, context={**CONTEXT, "alpha_f_l2_l1_min": 0.4})
        self.assertIsNone(weak["layout"])
        rows = {c["id"]: c for c in weak["checks"]}
        self.assertEqual(rows[LOCAL]["details"]["reason_code"], "slab_column_local_steel_not_implemented")
        supported = design_slab_reinforcement(inputs(), {"verified": False}, context=CONTEXT)
        local = {c["id"]: c for c in supported["checks"]}[LOCAL]
        self.assertEqual(local["status"], "pass")                                  # assessed, on the beam-supported floor
        # The ladder exhausted (verified demands no layout carries): the same reading of the floor.
        heavy = design_slab_reinforcement(inputs(), evidence(mu=5000.0), context={**CONTEXT, "alpha_f_l2_l1_min": 0.4})
        self.assertIsNone(heavy["layout"])
        self.assertIn("slab_strip_reinforcement_ladder", {c["id"] for c in heavy["checks"]})
        self.assertEqual({c["id"]: c for c in heavy["checks"]}[LOCAL]["details"]["reason_code"],
                         "slab_column_local_steel_not_implemented")
        exhausted = design_slab_reinforcement(inputs(), evidence(mu=5000.0), context=CONTEXT)
        self.assertEqual({c["id"]: c for c in exhausted["checks"]}[LOCAL]["status"], "pass")

    def test_the_context_key_is_required_and_boolean(self):
        missing = {k: v for k, v in CONTEXT.items() if k != "slab_column_local_steel_assessed"}
        self.assertIsNone(design_slab_reinforcement(inputs(), evidence(), context=missing)["layout"])
        self.assertIsNone(design(slab_column_local_steel_assessed=1)[0]["layout"])


class Eligibility(unittest.TestCase):
    def test_an_unresolved_gate_is_unknown_and_the_exclusion_is_listed_apart(self):
        eligibility = slab_strength_eligibility(design(slab_column_local_steel_assessed=False)[0])
        self.assertEqual([c["id"] for c in eligibility["unknown_checks"]], [LOCAL])
        self.assertEqual([(c["id"], c["item"]) for c in eligibility["scope_exclusions"]], [(FIRE, "fire_resistance")])
        self.assertFalse(eligibility["slab_checks_complete"])
        self.assertEqual(eligibility["credit"], "developed")            # the credit reads development alone
        resolved = slab_strength_eligibility(design()[0])
        self.assertEqual((resolved["unknown_checks"], resolved["slab_checks_complete"]), ([], True))


class Contract(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.state = gf.uniform_state()
        self.expected = contract.expected_inventory(self.state)

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def result(self, rows):
        return contract.code_check_result(rows, self.expected)

    def with_fire_row(self, row):
        return self.result(cf.rows_for(self.state) + [contract.compact_row("slab", row)])

    def assess(self, rows):
        return contract.assess_evaluation(cf.evaluation_record(self.state, rows=rows),
                                          submitted_identity=self.state.identity())

    def test_both_gates_are_required_completion_rows(self):
        required = {tuple(key) for rule in self.expected["conditional"] for key in rule["requires"]}
        self.assertTrue({("slab", TWO_WAY, ""), ("slab", LOCAL, "")} <= required)
        self.assertEqual(self.result(cf.rows_for(self.state))["status"], "passed")

    def test_each_gate_blocks_acceptance_on_its_own(self):
        for name, code in ((TWO_WAY, "two_way_shear_path_assessment_required"),
                           (LOCAL, "slab_column_local_steel_assessment_required")):
            key = ("slab", name, "")
            with self.subTest(gate=name):
                unresolved = self.result(cf.rows_for(self.state, statuses={key: ("not_evaluated", code)}))
                self.assertEqual((unresolved["status"], unresolved["accepted"]), ("incomplete", False))
                self.assertEqual(unresolved["unknown_reason_codes"], [code])
                missing = self.result(cf.rows_for(self.state, drop=(key,)))
                self.assertEqual(missing["status"], "incomplete")
                self.assertIn({"family": "slab", "id": name, "location": ""}, missing["missing"])

    def test_a_known_failure_beside_an_unresolved_gate_is_unresolved_not_a_check_failure(self):
        failing = cf.scwb_keys(self.state)[0]
        rows = cf.rows_for(self.state, statuses={("slab", TWO_WAY, ""): ("not_evaluated",
                                                                         "two_way_shear_path_assessment_required"),
                                                 failing: "fail"})
        self.assertEqual((self.result(rows)["status"], self.result(rows)["counts"]["unknown"]), ("failed", 1))
        assessed = self.assess(rows)
        self.assertEqual((assessed["valid"], assessed["classification"]), (True, "unresolved"))
        self.assertEqual(self.assess(cf.rows_for(self.state, statuses={failing: "fail"}))["classification"],
                         "check_failed")                                  # the control: no open gate

    def test_the_fire_row_is_skipped_only_as_the_declared_exclusion(self):
        result = self.with_fire_row(fire_exclusion())
        self.assertEqual((result["status"], result["counts"]["unknown"]), ("passed", 0))
        self.assertEqual([(e["id"], e["item"]) for e in result["excluded"]], [(FIRE, "fire_resistance")])
        self.assertEqual(result["excluded"][0]["basis"], SCOPE_EXCLUSIONS["fire_resistance"])
        self.assertEqual(contract.SCOPE["excluded"]["fire_resistance"], SCOPE_EXCLUSIONS["fire_resistance"])
        self.assertIn("not checked and not passed", SCOPE_EXCLUSIONS["fire_resistance"])

    def test_a_fire_row_that_passes_fails_or_waits_on_a_review_is_invalid(self):
        for row in (make_check(FIRE, "ACI 318-19 4.11", 1, 1, "=="), make_check(FIRE, "ACI 318-19 4.11", 2.0, 1.0, "<="),
                    awaiting_review(FIRE, "ACI 318-19 4.11", "outside scope", "fire_resistance_scope_accepted")):
            with self.subTest(status=row["status"]):
                result = self.with_fire_row(row)
                self.assertEqual((result["status"], result["accepted"]), ("invalid", False))

    def test_the_label_alone_excludes_nothing(self):
        # The declared row without its label contradicts the scope: invalid, never skipped.
        self.assertEqual(self.with_fire_row(not_evaluated(FIRE, "ACI 318-19 4.11", "unlabelled"))["status"], "invalid")
        # A labelled row the scope does not declare is an ordinary unknown.
        other = scope_exclusion("slab_other_item", "fixture clause", "excluded", "fire_resistance")
        self.assertEqual(self.with_fire_row(other)["status"], "incomplete")
        with self.assertRaises(ValueError):
            scope_exclusion(FIRE, "ACI 318-19 4.11", "excluded", "not_a_declared_exclusion")

    def test_the_search_s_normalized_checks_skip_the_excluded_row(self):
        fire = contract.compact_row("slab", fire_exclusion())
        checks, _inventory = population.normalize_checks(cf.rows_for(self.state) + [fire], self.expected)
        self.assertNotIn(f"slab|{FIRE}|", checks)
        self.assertEqual(checks[f"slab|{LOCAL}|"]["s"], "pass")


class Qualification(unittest.TestCase):
    def test_no_verification_flag_passes_the_excluded_item_or_the_local_steel_gate(self):
        policy = {"floor_hand_check_verified": True, "fire_resistance_scope_accepted": True,
                  "slab_column_local_steel_assessed": True, "asserted_by": "an engineer",
                  "assertion_date": "2026-11-01", "assertion_basis": "a hand check"}
        checks = [fire_exclusion(), not_evaluated(LOCAL, "ACI 318-19 8.6.1.2", "open"),
                  not_evaluated("floor.independent_hand_verification", "review", "open")]
        result = {c["id"]: c for c in _apply_verification_assertions({"demand_basis": {"verification": policy}}, checks)}
        self.assertEqual((result[FIRE]["status"], result[LOCAL]["status"]), ("not_evaluated", "not_evaluated"))
        self.assertEqual(result["floor.independent_hand_verification"]["status"], "pass")      # the control
        for retired in ("slab_column_local_steel_assessed", "fire_resistance_scope_accepted"):
            self.assertNotIn(retired, vars(IndependentVerification()))

    def test_the_exclusion_is_listed_apart_and_any_other_fire_row_blocks(self):
        kept, excluded = split_scope_exclusions([make_check("fixture.row", "fixture clause", 1, 1, "=="), fire_exclusion()])
        self.assertEqual([c["id"] for c in kept], ["fixture.row"])
        self.assertEqual([(e["id"], e["status"]) for e in excluded], [(FIRE, "excluded: not checked, not passed")])
        for row in (make_check(FIRE, "ACI 318-19 4.11", 1, 1, "=="), not_evaluated(FIRE, "ACI 318-19 4.11", "unlabelled"),
                    awaiting_review(FIRE, "ACI 318-19 4.11", "outside scope", "fire_resistance_scope_accepted")):
            with self.subTest(status=row["status"]):
                kept, excluded = split_scope_exclusions([row])
                self.assertEqual((excluded, [c["status"] for c in kept]), ([], ["not_evaluated"]))
        self.assertEqual(qualification_scope()["excluded"], SCOPE_EXCLUSIONS)


class Assertions(unittest.TestCase):
    def test_the_local_steel_flag_is_a_slab_assertion_outside_all_asserted(self):
        stamp = dict(asserted_by="an engineer", assertion_date="2026-11-01", assertion_basis="a review")
        seven = {flag: True for flag in ("analysis_applicability_verified", "all_floors_enveloped",
                                         "load_pattern_envelope_verified", "spatial_envelope_per_unit_width",
                                         "twisting_moment_resolution_verified", "zero_membrane_force_verified",
                                         "verified")}
        self.assertTrue(SlabActionAssertions(**seven, **stamp).all_asserted())
        self.assertFalse(SlabActionAssertions().slab_column_local_steel_assessed)
        self.assertIn("slab_column_local_steel_assessed", dep.ASSERTION_FLAGS)

    def test_a_profile_that_predates_the_flag_loads_with_its_identity_unchanged(self):
        profile = dep.load(EXAMPLE)
        self.assertNotIn("slab_column_local_steel_assessed", profile.data["slab_action_assertions"])
        self.assertEqual(profile.identity(), EXAMPLE_IDENTITY)
        self.assertFalse(profile.apply(DesignConfig()).slab_actions.slab_column_local_steel_assessed)

    def test_a_non_boolean_local_steel_flag_is_refused(self):
        value = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        value["slab_action_assertions"]["slab_column_local_steel_assessed"] = "yes"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(dep.EvidenceProfileError, "slab_column_local_steel_assessed"):
                dep.load(path)


if __name__ == "__main__":
    unittest.main()
