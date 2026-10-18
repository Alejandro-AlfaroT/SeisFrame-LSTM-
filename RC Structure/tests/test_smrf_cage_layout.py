"""Hoop/crosstie arrangement: which bars need support and how many legs a cage can hold."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Cage_Layout import (beam_cage, column_cage, face_bar_positions, support_pattern,
                                     arrangement_with_crossties, cage_passes)


class SupportPatternTests(unittest.TestCase):
    def test_alternate_bars_and_the_six_inch_rule(self):
        # Six #6 bars across a 26-in face: 4.2 in on centres -> corners and alternate bars.
        pos = face_bar_positions(26.0, 1.5, 0.5, 0.75, 6)
        pattern = support_pattern(pos, 0.75)
        self.assertEqual(pattern["supported"], [True, False, True, False, True, True])
        self.assertTrue(pattern["alternate_bars_supported"])
        self.assertLessEqual(pattern["max_unsupported_clear_in"], 6.0)
        self.assertAlmostEqual(pattern["hx_in"], max(pos[2] - pos[0], pos[4] - pos[2]))
        # Four bars 7 in apart: an unsupported bar would be 6.3 in clear -> every bar tied.
        pos = face_bar_positions(26.0, 1.5, 0.5, 0.75, 4)
        self.assertEqual(support_pattern(pos, 0.75)["supported"], [True] * 4)
        # hx limit: three bars 11.5 in apart could skip the middle by the 6-in rule? no (10.75 clear) -> tied;
        # a 14-in hx cap also binds when spacing allows skipping.
        pos = [2.5, 6.5, 10.5, 14.5, 18.5]                    # 4-in spacing, skipping gives hx = 8
        self.assertEqual(support_pattern(pos, 0.75)["supported"], [True, False, True, False, True])
        self.assertEqual(support_pattern(pos, 0.75, hx_max_in=6.0)["supported"], [True] * 5)

    def test_high_axial_ties_every_bar(self):
        pos = face_bar_positions(26.0, 1.5, 0.5, 0.75, 6)
        self.assertEqual(support_pattern(pos, 0.75, hx_max_in=8.0, every_bar=True)["supported"], [True] * 6)

    def test_extra_crossties_grow_from_the_minimal_pattern(self):
        pos = face_bar_positions(26.0, 1.5, 0.5, 0.75, 6)
        three = arrangement_with_crossties(pos, 0.75, 3)
        self.assertEqual(three["crossties"], 3)
        self.assertTrue(three["alternate_bars_supported"])
        with self.assertRaises(ValueError):
            arrangement_with_crossties(pos, 0.75, 1)          # below the minimum the rules need
        with self.assertRaises(ValueError):
            arrangement_with_crossties(pos, 0.75, 5)          # more than the interior bars


class ColumnCageTests(unittest.TestCase):
    def test_three_top_bars_allow_three_legs_not_six(self):
        """The c09 case: 28x28 with 3 top bars and 3 side bars per face."""
        cage = column_cage(28.0, 28.0, 1.5, 0.625, 1.27, top_bars=3, side_bars=3)
        self.assertEqual(cage["legs_max"], {"across_b_face": 3, "across_h_face": 5})
        self.assertNotIn(6, cage["constructible_legs"])
        self.assertLessEqual(max(cage["constructible_legs"]), 3)
        realized = column_cage(28.0, 28.0, 1.5, 0.625, 1.27, 3, 3, legs=3)
        self.assertTrue(cage_passes(realized))
        self.assertEqual(realized["arrangement"]["b_face"]["supported"], [True, True, True])
        six = column_cage(28.0, 28.0, 1.5, 0.625, 1.27, 3, 3, legs=6)
        self.assertFalse(six["constructible"])
        self.assertFalse(cage_passes(six))

    def test_legs_per_direction_for_an_asymmetric_layout(self):
        """28x28 with #10 bars, 3 top and 4 side per face: 3 legs across b, 4 to 6 across h."""
        cage = column_cage(28.0, 28.0, 1.5, 0.625, 1.27, top_bars=3, side_bars=4)
        self.assertEqual(cage["legs_min"], {"across_b_face": 3, "across_h_face": 4})
        self.assertEqual(cage["legs_max"], {"across_b_face": 3, "across_h_face": 6})
        self.assertEqual(cage["constructible_legs"], [])
        self.assertEqual(cage["constructible_legs_by_direction"], {"across_b_face": [3], "across_h_face": [4, 5, 6]})
        realized = column_cage(28.0, 28.0, 1.5, 0.625, 1.27, 3, 4, legs={"across_b_face": 3, "across_h_face": 4})
        self.assertTrue(cage_passes(realized))
        self.assertEqual(realized["legs"], {"across_b_face": 3, "across_h_face": 4})
        self.assertEqual(realized["arrangement"]["b_face"]["crossties"], 1)
        self.assertEqual(realized["arrangement"]["h_face"]["crossties"], 2)
        self.assertEqual(realized["arrangement"]["h_face"]["supported"], [True, False, True, False, True, True])
        same_both_ways = column_cage(28.0, 28.0, 1.5, 0.625, 1.27, 3, 4, legs=4)
        self.assertFalse(same_both_ways["constructible"])
        with self.assertRaises(ValueError):
            column_cage(28.0, 28.0, 1.5, 0.625, 1.27, 3, 4, legs={"across_b_face": 3})

    def test_high_axial_requires_every_bar_and_hx_within_8(self):
        cage = column_cage(24.0, 24.0, 1.5, 0.5, 1.0, 4, 3, high_axial=True)
        self.assertEqual(cage["legs_min"], {"across_b_face": 4, "across_h_face": 5})
        self.assertEqual(cage["hx_max_in"], 8.0)
        realized = column_cage(24.0, 24.0, 1.5, 0.5, 1.0, 4, 3, high_axial=True, legs=4)
        self.assertFalse(realized["constructible"])               # the 5-bar side faces need 5 legs

    def test_narrow_face_needs_no_crosstie(self):
        cage = column_cage(14.0, 14.0, 1.5, 0.5, 0.75, 2, 0)
        self.assertEqual(cage["legs_min"], {"across_b_face": 2, "across_h_face": 2})
        self.assertEqual(cage["constructible_legs"], [2])


class BeamCageTests(unittest.TestCase):
    def test_seven_top_bars_need_crossties(self):
        cage = beam_cage(16.0, 1.5, 0.5, 0.75, top_bars=7, bot_bars=7)
        self.assertEqual(cage["legs_min"], 4)                      # S U S U S U S: two crossties
        self.assertEqual(cage["legs_max"], 7)
        two = beam_cage(16.0, 1.5, 0.5, 0.75, 7, 7, legs=2)
        self.assertFalse(two["constructible"])
        four = beam_cage(16.0, 1.5, 0.5, 0.75, 7, 7, legs=4)
        self.assertTrue(cage_passes(four))
        self.assertEqual(four["arrangement"]["top"]["supported"], [True, False, True, False, True, False, True])

    def test_three_bars_in_a_ten_inch_beam_fit_a_closed_hoop(self):
        cage = beam_cage(10.0, 1.5, 0.5, 0.75, 3, 3, legs=2)
        self.assertTrue(cage_passes(cage))
        self.assertEqual(cage["legs_max"], 3)
        # Crossties engage a bar on each face: with 3 top and 2 bottom bars only the hoop is possible.
        self.assertEqual(beam_cage(10.0, 1.5, 0.5, 0.75, 3, 2)["legs_max"], 2)


if __name__ == "__main__":
    unittest.main()
