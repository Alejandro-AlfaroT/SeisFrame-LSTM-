"""The required inventory of the frame-plus-transfer production route (2026-11-07). No solve."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

from Design import Code_Check_Contract as cc                                    # noqa: E402
from Design import SMRF_Common_Action_Contract as ca                            # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Design.SMRF_Conventional_Floor import floor_row_inventory                   # noqa: E402


class FrameRouteInventory(unittest.TestCase):
    def test_the_floor_rows_replace_the_common_action_rows(self):
        with gf.frame():
            expected = cc.expected_inventory(gf.uniform_state())
        self.assertLessEqual(floor_row_inventory(), expected["exact"])
        self.assertFalse(ca.required_inventory() & expected["exact"])
        self.assertEqual(expected["scope"], "aci318_19_smrf_candidate_code_checks_v4_frame_transfer")

    def test_the_computed_route_keeps_its_common_action_rows(self):
        cfg = DesignConfig()
        cfg.floor_analysis.common_action_procedure = {"kind": "computed_common_actions_v1"}
        with gf.frame():
            expected = cc.expected_inventory(gf.uniform_state(), cfg)
        self.assertLessEqual(ca.required_inventory(), expected["exact"])
        self.assertFalse(floor_row_inventory() & expected["exact"])

    def test_torsion_and_physical_rows_stay_required(self):
        with gf.frame():
            state = gf.uniform_state()
            expected = cc.expected_inventory(state)
        names = {name for _family, name, _location in expected["exact"]}
        for name in ("beam.torsion_concurrent", "beam.joint_physical_detail", "detailing.splice_physical_detail",
                     "beam.slab_physical_inventory", "column.roof_anchorage_physical_detail"):
            self.assertIn(name, names)

    def test_missing_floor_rows_block_acceptance(self):
        with gf.frame():
            state = gf.uniform_state()
            expected = cc.expected_inventory(state)
        result = cc.code_check_result([], expected)
        self.assertFalse(result["accepted"])
        missing = {(r.get("family"), r.get("id"), r.get("location", "")) for r in result["missing"]}
        self.assertLessEqual(floor_row_inventory(), missing)

    def test_an_unknown_detail_basis_is_refused(self):
        from Design import Grouped_Design as gd
        cfg = DesignConfig()
        cfg.floor_analysis.detail_basis = "something_else"
        with gf.frame():
            with self.assertRaises(ValueError):
                gd.evaluate_candidate(gf.uniform_state(), cfg, {})


if __name__ == "__main__":
    unittest.main()
