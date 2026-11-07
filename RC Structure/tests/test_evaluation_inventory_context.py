"""The search re-derives a candidate's inventory in the candidate's realized context (2026-11-07).

The joint types depend on the realized slab layout and the beam contexts on the detail basis, so
Code_Check_Contract.evaluation_inventory installs the record's slab thickness, slab layout and detail basis while it
derives the inventory and puts the caller's values back, also when the derivation raises.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import Structure_Parameters as sp  # noqa: E402
from Design import Code_Check_Contract as contract  # noqa: E402

LAYOUT = {"layout": {"marker": "realized"}}


def _evaluation(basis="physical_3d_v2", thickness=7.5):
    return {"state": object(), "detail_basis": basis,
            "evidence": {"slab": {"slab": {"thickness_in": thickness}, "reinforcement": LAYOUT}}}


class EvaluationInventoryContext(unittest.TestCase):
    def setUp(self):
        self.before = (sp.SLAB_THICKNESS_IN, sp.SLAB_REINFORCEMENT, sp.DETAIL_BASIS)
        self.seen = {}

    def tearDown(self):
        sp.SLAB_THICKNESS_IN, sp.SLAB_REINFORCEMENT, sp.DETAIL_BASIS = self.before

    def _capture(self, state, cfg=None):
        self.seen.update(thickness=sp.SLAB_THICKNESS_IN, reinforcement=sp.SLAB_REINFORCEMENT, basis=sp.DETAIL_BASIS)
        return {"exact": set()}

    def test_the_realized_slab_layout_and_basis_are_installed(self):
        with mock.patch.object(contract, "expected_inventory", side_effect=self._capture):
            contract.evaluation_inventory(_evaluation())
        self.assertEqual(self.seen, {"thickness": 7.5, "reinforcement": LAYOUT, "basis": "physical_3d_v2"})
        self.assertEqual((sp.SLAB_THICKNESS_IN, sp.SLAB_REINFORCEMENT, sp.DETAIL_BASIS), self.before)

    def test_a_record_without_a_basis_keeps_the_callers_basis(self):
        evaluation = _evaluation()
        del evaluation["detail_basis"]
        with mock.patch.object(contract, "expected_inventory", side_effect=self._capture):
            contract.evaluation_inventory(evaluation)
        self.assertEqual(self.seen["basis"], self.before[2])

    def test_the_callers_values_come_back_when_the_derivation_raises(self):
        with mock.patch.object(contract, "expected_inventory", side_effect=RuntimeError("underivable")):
            with self.assertRaises(RuntimeError):
                contract.evaluation_inventory(_evaluation())
        self.assertEqual((sp.SLAB_THICKNESS_IN, sp.SLAB_REINFORCEMENT, sp.DETAIL_BASIS), self.before)

    def test_a_missing_thickness_declaration_is_refused_before_anything_is_installed(self):
        evaluation = _evaluation()
        del evaluation["evidence"]["slab"]["slab"]["thickness_in"]
        with self.assertRaises(ValueError):
            contract.evaluation_inventory(evaluation)
        self.assertEqual((sp.SLAB_THICKNESS_IN, sp.SLAB_REINFORCEMENT, sp.DETAIL_BASIS), self.before)


if __name__ == "__main__":
    unittest.main()
