"""Incomplete cages retain proven failures without accepting seed reinforcement.

No structural solver runs: these tests isolate the qualification consumer's
contract with independently recomputed capacity evidence.
"""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Design.SMRF_Common import make_check, summarize_checks
from Design.SMRF_Qualification import (
    _apply_capacity_design_evidence, detailing_inputs, recomputed_evidence,
)


def fixture():
    beam = {"bar_size": 5, "legs": 6, "spacing_in": 6.0}
    column = {"bar_size": 4, "legs_model": 4, "spacing_in": 3.0,
              "legs": {"across_b_face": 4, "across_h_face": 6}}
    rebar = {"beam_stirrup_bar_size": 5, "beam_stirrup_legs": 6,
             "beam_stirrup_spacing_in": 6.0, "col_stirrup_bar_size": 4,
             "col_stirrup_legs": 4, "col_stirrup_spacing_in": 3.0,
             "col_stirrup_legs_by_direction": copy.deepcopy(column["legs"])}
    capacity = {"transverse": {"beam": beam, "column": column}, "accepted": True,
                "checks": [], "columns": {"hoops": column, "section_adequate": True},
                "beams": {"hoops": beam, "section_adequate": True}}
    return {"reinforcement": rebar, "capacity_design": copy.deepcopy(capacity)}, capacity


def recompute(record, capacity, consistent=True):
    result = {"consistent": consistent, "differences": [] if consistent else ["saved check differs"],
              "recomputed": capacity, "reason": None}
    with patch("Design.SMRF_Design_Evidence.capacity_design_recomputation", return_value=result):
        return recomputed_evidence(record)


def checks_by_id(evidence):
    return {c["id"]: c for c in evidence["checks"]}


class IncompleteCageTests(unittest.TestCase):
    def test_missing_column_selection_retains_section_failure_and_rejects_completion(self):
        record, capacity = fixture()
        capacity["transverse"]["column"] = None
        capacity["columns"].update(hoops=None, section_adequate=False,
                                   governing={"vs_required_kip": 1655.542, "ve_kip": 1241.657})
        capacity["accepted"] = False
        capacity["checks"] = [
            make_check("column.capacity_shear_section", "ACI 318-19 22.5", 1655.542, 1320.312),
            make_check("column.hoops_selected", "Capacity-design completion", 0, 1, "==")]
        record["reinforcement"].update(col_stirrup_legs=2, col_stirrup_spacing_in=4.0,
                                      col_stirrup_legs_by_direction={"across_b_face": 2, "across_h_face": 2})
        before = copy.deepcopy(record)
        evidence = recompute(record, capacity)
        checks = checks_by_id(evidence)
        self.assertIs(evidence["capacity"], capacity)
        self.assertEqual(checks["qualification.capacity_evidence_recomputed"]["status"], "pass")
        identity = checks["qualification.hoops_match_design"]
        self.assertEqual(identity["status"], "not_evaluated")
        self.assertEqual(identity["details"]["members"], {"beam": "installed_match", "column": "not_selected"})
        self.assertEqual(checks["qualification.hoop_selection_complete"]["status"], "fail")
        final = _apply_capacity_design_evidence(record, evidence["checks"], evidence)
        final_by_id = {c["id"]: c for c in final}
        for name in ("column.capacity_shear_section", "column.hoops_selected",
                     "qualification.column_capacity_shear", "qualification.joint_capacity_completion"):
            self.assertEqual(final_by_id[name]["status"], "fail")
        self.assertFalse(summarize_checks(final)["accepted"])
        self.assertEqual(record, before)

    def test_missing_beam_or_both_cages_are_explicitly_incomplete(self):
        for members in (("beam",), ("beam", "column")):
            with self.subTest(members=members):
                record, capacity = fixture()
                for member in members:
                    capacity["transverse"][member] = None
                evidence = recompute(record, capacity)
                checks = checks_by_id(evidence)
                self.assertIs(evidence["capacity"], capacity)
                self.assertEqual(checks["qualification.hoop_selection_complete"]["status"], "fail")
                self.assertEqual(checks["qualification.hoops_match_design"]["status"], "not_evaluated")
                self.assertFalse(summarize_checks(evidence["checks"])["accepted"])

    def test_selected_mismatch_still_rejects_capacity_when_other_member_is_missing(self):
        record, capacity = fixture()
        capacity["transverse"]["column"] = None
        record["reinforcement"]["beam_stirrup_spacing_in"] = 600
        evidence = recompute(record, capacity)
        identity = checks_by_id(evidence)["qualification.hoops_match_design"]
        self.assertEqual(evidence["capacity"], {})
        self.assertEqual(identity["status"], "fail")
        self.assertEqual(identity["details"]["mismatched_members"], ["beam"])
        self.assertEqual(identity["details"]["missing_selections"], ["column"])

    def test_selected_column_directional_legs_must_match_even_when_scalar_matches(self):
        record, capacity = fixture()
        record["reinforcement"]["col_stirrup_legs_by_direction"]["across_h_face"] = 4
        evidence = recompute(record, capacity)
        self.assertEqual(evidence["capacity"], {})
        self.assertEqual(checks_by_id(evidence)["qualification.hoops_match_design"]["status"], "fail")

    def test_consistent_complete_selections_keep_existing_identity_pass(self):
        record, capacity = fixture()
        evidence = recompute(record, capacity)
        self.assertIs(evidence["capacity"], capacity)
        self.assertEqual(checks_by_id(evidence)["qualification.hoops_match_design"]["status"], "pass")
        self.assertEqual(checks_by_id(evidence)["qualification.hoop_selection_complete"]["status"], "pass")

    def test_inconsistent_saved_capacity_is_never_consumed(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                record, capacity = fixture()
                if missing:
                    capacity["transverse"]["column"] = None
                evidence = recompute(record, capacity, consistent=False)
                self.assertEqual(evidence["capacity"], {})
                self.assertEqual(checks_by_id(evidence)["qualification.capacity_evidence_recomputed"]["status"], "fail")

    def test_failed_recomputation_does_not_invent_a_hoop_selection_failure(self):
        record, _ = fixture()
        evidence = recompute(record, None, consistent=False)
        checks = checks_by_id(evidence)
        self.assertEqual(evidence["capacity"], {})
        self.assertEqual(checks["qualification.hoops_match_design"]["status"], "not_evaluated")
        self.assertNotIn("qualification.hoop_selection_complete", checks)


class LayeredDetailingAdapterTests(unittest.TestCase):
    def test_saved_declaration_forwarded_including_malformed_values(self):
        for value in ({"max_layers": 3, "layers": {"x": {}}}, None, {}, "bad"):
            with self.subTest(value=value):
                record = {"reinforcement": {"beam_bar_stacking": value}}
                inputs = detailing_inputs(record)
                self.assertIn("bar_stacking", inputs["beam"])
                self.assertEqual(inputs["beam"]["bar_stacking"], value)
                self.assertNotIn("bar_stacking", inputs["column"])

    def test_absent_declaration_preserves_legacy_and_cannot_be_injected(self):
        record = {"reinforcement": {}, "detailing": {"beam": {"bar_stacking": {"passes": True}}}}
        self.assertNotIn("bar_stacking", detailing_inputs(record)["beam"])

    def test_detailing_cannot_override_saved_layered_declaration(self):
        actual = {"max_layers": 3, "layer_order": "alternating"}
        record = {"reinforcement": {"beam_bar_stacking": actual},
                  "detailing": {"beam": {"bar_stacking": {"passes": True}, "b_in": 99}},
                  "sections": {"b_beam_in": 30}}
        inputs = detailing_inputs(record)
        self.assertEqual(inputs["beam"]["bar_stacking"], actual)
        self.assertEqual(inputs["beam"]["b_in"], 30)


if __name__ == "__main__":
    unittest.main()
