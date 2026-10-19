"""Perimeter-only designs stop at 32 in columns (design decision of 2026-10-18): the core frames come in when the
perimeter columns would pass 32 in, and at the cap the column reinforcement may go to the ACI 318-19 18.7.4.1 limit
before the layout ladder adds frames. The core layouts keep the general cap."""
import sys
import unittest
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
        self.assertEqual(sd.PERIMETER_COLUMN_SIZE_CAP_IN, 32.0)
        self.assertEqual(sd.column_size_cap_in(mg.SFRS_LAYOUT_PERIMETER), 32.0)
        self.assertEqual(sd.column_size_cap_in(mg.SFRS_LAYOUT_PERIMETER_CORE), sd.COLUMN_SIZE_CAP_IN)
        self.assertEqual(sd.column_size_cap_in(mg.SFRS_LAYOUT_PERIMETER_CORE_LINES), sd.COLUMN_SIZE_CAP_IN)
        self.assertEqual(sd.column_size_cap_in(None), sd.COLUMN_SIZE_CAP_IN)              # the band policy, the uniform route
        self.assertEqual(sd.column_sizes_in(mg.SFRS_LAYOUT_PERIMETER)[-1], 32.0)
        self.assertEqual(sd.column_sizes_in(mg.SFRS_LAYOUT_PERIMETER_CORE)[-1], sd.COLUMN_SIZE_CAP_IN)

    def test_the_perimeter_search_has_no_size_above_32(self):
        saved = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 4, 3, 3
        try:
            column = mg.MemberDesign("column", 32.0, 32.0, 10.0, 10, 5, 5, 3, 5, 4, 3.0, (4, 4))
            beam = mg.MemberDesign("beam_x", 16.0, 28.0, 8.0, 8, 4, 4, 0, 4, 2, 5.0)
            perimeter = mg.GroupedDesign.uniform(4, 3, 3, column, beam, policy_id=mg.SYSTEM_BANDED_POLICY_ID)
            core = mg.GroupedDesign.uniform(4, 3, 3, column, beam, policy_id=mg.SYSTEM_CORE_BANDED_POLICY_ID)
            self.assertIsNone(gs._column_step(column, +1, perimeter))
            self.assertEqual(gs._column_step(column, +1, core), (34.0, 34.0))
            self.assertEqual(gs._column_step(column, -1, perimeter), (30.0, 30.0))
        finally:
            sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = saved

    def test_reinforcement_opens_to_the_code_limit_at_the_cap(self):
        cfg = DesignConfig()
        at_cap = mg.MemberDesign("column", 32.0, 32.0, 10.0, 10, 5, 5, 3, 5, 4, 3.0, (4, 4))
        below = mg.MemberDesign("column", 30.0, 30.0, 10.0, 10, 5, 5, 3, 5, 4, 3.0, (4, 4))
        heavy_at_cap = max(_rho(c) for c in selection.column_cage_candidates(at_cap, cfg, layout=mg.SFRS_LAYOUT_PERIMETER))
        heavy_below = max(_rho(c) for c in selection.column_cage_candidates(below, cfg, layout=mg.SFRS_LAYOUT_PERIMETER))
        heavy_core = max(_rho(c) for c in selection.column_cage_candidates(at_cap, cfg, layout=mg.SFRS_LAYOUT_PERIMETER_CORE))
        self.assertGreater(heavy_at_cap, cfg.rebar.rho_col_practical_max)                 # the last lever at 32 in
        self.assertLessEqual(heavy_at_cap, cfg.rebar.rho_col_max + 1e-9)
        self.assertLessEqual(heavy_below, cfg.rebar.rho_col_practical_max + 1e-9)
        self.assertLessEqual(heavy_core, cfg.rebar.rho_col_practical_max + 1e-9)          # 32 in is not the core layout's cap


if __name__ == "__main__":
    unittest.main()
