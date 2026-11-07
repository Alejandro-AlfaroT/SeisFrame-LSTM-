"""A missing physical premise of the story-strength model leaves the story strength not calculated and the vertical
strength regularity row not evaluated; it is not a software error, and binding errors still raise (2026-11-05)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp


class StoryPremiseRefusal(unittest.TestCase):
    def strength(self, error):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import Grouped_Design as gd, SMRF_Story_Adapter as adapter
        from Design.Config import DesignConfig
        with gf.frame():
            state = mg.install(gf.uniform_state())

            def refuse(*a, **k):
                raise error
            with mock.patch.object(adapter, "calculate", refuse):
                result = gd.story_strength_distribution(DesignConfig(), state=state, actions=[], expected_ids=[],
                                                        capacity={}, gravity_actions=[])
            ledger = {"floors": [{"floor": k, "seismic_weight_kip": 100.0} for k in range(1, sp.NUM_FLOOR + 1)]}
            vertical = gd.vertical_regularity(None, None, result, ledger)
            self.rows = {c["id"]: c for c in gd.regularity_checks(vertical, "D", strength_model_verified=False)}
            return result

    def test_premise_refusal_is_an_incomplete_record_with_its_reason(self):
        from Design.SMRF_Story_Adapter import StoryStrengthPremiseUnavailable
        result = self.strength(StoryStrengthPremiseUnavailable(
            "Joint nominal strength lacks topology/confinement evidence", joint_id="joint_7", joint_type="T1", axis="x"))
        self.assertEqual(result["stories"], [])
        self.assertEqual(result["applicability"]["status"], "incomplete")
        self.assertFalse(result["applicability"]["verified"])
        self.assertIn("topology/confinement", result["applicability"]["technical_gaps"][0])
        self.assertEqual(result["premise_refusal"]["joint_id"], "joint_7")
        self.assertIn("one_side_fraction", result)

    def test_the_row_is_not_evaluated_and_not_a_review_state(self):
        from Design import Grouped_Design as gd, Code_Check_Contract as contract
        from Design.SMRF_Story_Adapter import StoryStrengthPremiseUnavailable, computed_verification_valid
        result = self.strength(StoryStrengthPremiseUnavailable("No physical positive nominal transverse-steel strength"))
        self.assertFalse(computed_verification_valid(result))
        row = self.rows["demands.vertical_strength_regularity"]
        self.assertEqual(row["status"], "not_evaluated")
        self.assertNotIn("review_dependency", row.get("details") or {})

    def test_binding_errors_still_raise(self):
        with self.assertRaises(ValueError) as caught:
            self.strength(ValueError("Story strength action inventory: differs"))
        from Design.SMRF_Story_Adapter import StoryStrengthPremiseUnavailable
        self.assertNotIsInstance(caught.exception, StoryStrengthPremiseUnavailable)


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
