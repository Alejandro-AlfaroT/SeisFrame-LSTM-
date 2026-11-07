"""Column cages are selected only where catalog hoops can confine them (2026-11-07).

The case 9 seed cage (30 x 30 in, f'c 5 ksi, 12 No. 8 with 3 bars on each b face) under Pu > 0.3 Ag f'c needs every bar
tied with hx <= 8 in (ACI 318-19 18.7.5.2(f)); its b-face bars are 12.5 in apart, so no hoop exists. Pu is the largest
compression of the combinations including E.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp  # noqa: E402
from Design import SMRF_Capacity_Design as cd  # noqa: E402


def _column(**changes):
    from tests import grouped_fixture as gf
    from Model import Member_Groups as mg
    with gf.frame():
        state = gf.uniform_state()
        column = next(d for d in state.designs.values() if d.is_column)
    base = replace(column, b_in=30.0, h_in=30.0, fc_ksi=5.0, bar_size=8, top_bars=3, bot_bars=3, side_bars=3)
    return replace(base, **changes)


class ColumnHoopFeasibility(unittest.TestCase):
    def setUp(self):
        from tests import grouped_fixture as gf
        self.frame = gf.frame()
        self.frame.__enter__()

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def test_three_bars_on_a_30_in_face_cannot_be_confined_at_high_axial(self):
        feasible, detail = cd.column_hoop_feasibility(_column(), 1430.0)
        self.assertGreater(1430.0, 0.3 * 30.0 * 30.0 * 5.0)
        self.assertFalse(feasible)
        self.assertTrue(detail["high_axial"])

    def test_the_same_cage_is_confined_below_the_high_axial_threshold(self):
        feasible, detail = cd.column_hoop_feasibility(_column(), 1000.0)
        self.assertTrue(feasible)
        self.assertFalse(detail["high_axial"])
        self.assertGreaterEqual(detail["spacing_in"], cd.SPACING_MIN_IN)

    def test_five_bars_on_each_b_face_are_confined_at_high_axial(self):
        feasible, detail = cd.column_hoop_feasibility(_column(top_bars=5, bot_bars=5), 1430.0)
        self.assertTrue(feasible)
        self.assertTrue(detail["high_axial"])
        self.assertIn(detail["hoop_bar"], cd.COLUMN_HOOP_BARS)


class SeismicCompression(unittest.TestCase):
    ACTIONS = [
        {"id": "1.2D+1.6L", "ex": 0.0, "ey": 0.0, "analysis_succeeded": True,
         "members": {"1": {"axial_i_kip": 1680.0, "axial_j_kip": 1650.0}}},
        {"id": "1.2D+0.5L+Ex", "ex": 1.0, "ey": 0.0, "analysis_succeeded": True,
         "members": {"1": {"axial_i_kip": 1430.0, "axial_j_kip": 1400.0}}},
        {"id": "0.9D-Ey", "ex": 0.0, "ey": -1.0, "analysis_succeeded": True,
         "members": {"1": {"axial_i_kip": 900.0, "axial_j_kip": 880.0}}},
        {"id": "failed+Ex", "ex": 1.0, "ey": 0.0, "analysis_succeeded": False,
         "members": {"1": {"axial_i_kip": 9999.0, "axial_j_kip": 9999.0}}},
    ]

    def test_only_successful_combinations_including_e_count(self):
        self.assertEqual(cd.seismic_column_compression_kip(self.ACTIONS, [1]), 1430.0)

    def test_none_without_a_combination_including_e(self):
        self.assertIsNone(cd.seismic_column_compression_kip(self.ACTIONS[:1], [1]))


if __name__ == "__main__":
    unittest.main()
