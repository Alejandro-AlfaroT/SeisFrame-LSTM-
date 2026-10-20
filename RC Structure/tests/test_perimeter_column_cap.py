"""Perimeter and core layouts stop at 36 in columns (design decision of 2026-10-19).
At each layout's cap, the existing reinforcement search can use its code-limit ceiling."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Design import Grouped_Search as gs
from Design import Group_Selection as selection
from Design import Section_Design as sd
from Design.Config import DesignConfig
from Model import Member_Groups as mg


def _rho(design):
    return (design.top_bars + design.bot_bars + 2 * design.side_bars) * sp.rebar_area(design.bar_size) / (design.b_in * design.h_in)


class PerimeterColumnCapTests(unittest.TestCase):
    def test_caps_by_layout(self):
        self.assertEqual(sd.PERIMETER_COLUMN_SIZE_CAP_IN, 36.0)
        self.assertEqual(sd.COLUMN_SIZE_CAP_IN, 36.0)
        self.assertEqual(sd.column_size_cap_in(mg.SFRS_LAYOUT_PERIMETER), 36.0)
        self.assertEqual(sd.column_size_cap_in(mg.SFRS_LAYOUT_PERIMETER_CORE), sd.COLUMN_SIZE_CAP_IN)
        self.assertEqual(sd.column_size_cap_in(mg.SFRS_LAYOUT_PERIMETER_CORE_LINES), sd.COLUMN_SIZE_CAP_IN)
        self.assertEqual(sd.column_size_cap_in(None), sd.COLUMN_SIZE_CAP_IN)              # the band policy, the uniform route
        self.assertEqual(sd.column_sizes_in(mg.SFRS_LAYOUT_PERIMETER)[-3:], (32.0, 34.0, 36.0))
        self.assertEqual(sd.column_sizes_in(mg.SFRS_LAYOUT_PERIMETER_CORE)[-1], sd.COLUMN_SIZE_CAP_IN)

    def test_no_layout_can_grow_above_36_and_reduction_remains_available(self):
        saved = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 4, 3, 3
        try:
            column = mg.MemberDesign("column", 36.0, 36.0, 10.0, 10, 5, 5, 3, 5, 4, 3.0, (4, 4))
            beam = mg.MemberDesign("beam_x", 16.0, 28.0, 8.0, 8, 4, 4, 0, 4, 2, 5.0)
            for policy in (mg.SYSTEM_BANDED_POLICY_ID, mg.SYSTEM_CORE_BANDED_POLICY_ID,
                           mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID):
                with self.subTest(policy=policy):
                    state = mg.GroupedDesign.uniform(4, 3, 3, column, beam, policy_id=policy)
                    self.assertIsNone(gs._column_step(column, +1, state))
                    self.assertEqual(gs._column_step(column, -1, state), (34.0, 34.0))
        finally:
            sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = saved

    def test_reinforcement_opens_to_the_code_limit_at_the_cap(self):
        cfg = DesignConfig()
        # Use a lower practical ceiling so the real 36-in cage geometry offers
        # candidates on both sides of it. Geometry can govern below the default
        # 4% ceiling even when the size-cap rule permits a higher steel ratio.
        cfg = replace(cfg, rebar=replace(cfg.rebar, rho_col_practical_max=0.03))
        at_cap = mg.MemberDesign("column", 36.0, 36.0, 10.0, 10, 5, 5, 3, 5, 4, 3.0, (4, 4))
        below = mg.MemberDesign("column", 34.0, 34.0, 10.0, 10, 5, 5, 3, 5, 4, 3.0, (4, 4))
        for layout in mg.SFRS_LAYOUT_LADDER:
            with self.subTest(layout=layout):
                heavy_at_cap = max(_rho(c) for c in selection.column_cage_candidates(at_cap, cfg, layout=layout))
                heavy_below = max(_rho(c) for c in selection.column_cage_candidates(below, cfg, layout=layout))
                self.assertGreater(heavy_at_cap, cfg.rebar.rho_col_practical_max)
                self.assertLessEqual(heavy_at_cap, cfg.rebar.rho_col_max + 1e-9)
                self.assertLessEqual(heavy_below, cfg.rebar.rho_col_practical_max + 1e-9)


if __name__ == "__main__":
    unittest.main()
