"""The conventional story-strength route binds its inputs before the mechanism runs (2026-11-07).

A binding error (the signed combinations, members or gravity states do not match the design inventory) is a software error
and raises; a premise the story mechanism lacks leaves the story strength not calculated, the record incomplete with its
reason, and the vertical strength regularity row not evaluated.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp  # noqa: E402


class ConventionalStoryBinding(unittest.TestCase):
    def setUp(self):
        self.previous = sp.DETAIL_BASIS
        sp.DETAIL_BASIS = "conventional_detailing_v1"

    def tearDown(self):
        sp.DETAIL_BASIS = self.previous

    def strength(self, **patches):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import Grouped_Design as gd
        from Design.Config import DesignConfig
        with gf.frame():
            state = mg.install(gf.uniform_state())
            with mock.patch.multiple("Design.SMRF_Story_Adapter", **patches) if patches else _nothing():
                result = gd.story_strength_distribution(DesignConfig(), state=state, actions=[], expected_ids=[],
                                                        capacity={}, gravity_actions=[])
            ledger = {"floors": [{"floor": k, "seismic_weight_kip": 100.0} for k in range(1, sp.NUM_FLOOR + 1)]}
            vertical = gd.vertical_regularity(None, None, result, ledger)
            self.rows = {c["id"]: c for c in gd.regularity_checks(vertical, "D", strength_model_verified=False)}
            return result

    def test_an_unbound_action_inventory_raises(self):
        with self.assertRaisesRegex(ValueError, "Story strength action inventory"):
            self.strength()

    def test_a_missing_premise_is_an_incomplete_record_with_its_reason(self):
        from Design import SMRF_Conventional_Story as conventional_story

        def lacking(*a, **k):
            raise ValueError("no gravity state of the seismic combinations is available")
        with mock.patch.object(conventional_story, "calculate", lacking):
            result = self.strength(_validate_inventory=lambda *a, **k: None)
        self.assertEqual(result["stories"], [])
        self.assertEqual(result["applicability"]["status"], "incomplete")
        self.assertFalse(result["applicability"]["verified"])
        self.assertIn("no gravity state", result["applicability"]["technical_gaps"][0])
        self.assertEqual(self.rows["demands.vertical_strength_regularity"]["status"], "not_evaluated")


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    unittest.main()
