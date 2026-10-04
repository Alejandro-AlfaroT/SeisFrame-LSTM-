"""Beam growth rule (user decision 2026-10-02): the beam strength jump lands on the lightest rung that carries the
governing factored moment at the declared steel ratio, and never above the capacity-proxy jump. A beam the steel pass
could not reinforce reports its DCR on the seed bars; that number no longer sizes the section."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Structure_Parameters as sp                                              # noqa: E402
from Design import Design_Driver as driver                                     # noqa: E402
from Design.Section_Design import (                                            # noqa: E402
    BEAM_GROWTH_DEPTH_OFFSET_IN, BEAM_GROWTH_STEEL_RATIO, beam_ladder, beam_rung_carries_moment, column_ladder,
    suggest_beam_rung_for_moment, suggest_rung_index)

FLAGS = {"drift_ok": True, "scwb_ok": True, "joint_scwb_failed": False, "capacity_accepted": True,
         "beam_section_adequate": True, "beam_hoops_selected": True, "column_section_adequate": True,
         "column_hoops_selected": True, "joints_all_pass": True, "anchorage_all_pass": True, "joint_shear_ratio": None}


class MomentSizedJump(unittest.TestCase):
    def setUp(self):
        self.beams = beam_ladder(span_in=360.0, story_height_in=168.0)
        self.columns = column_ladder()
        self.start = self.beams.index((12.0, 24.0, 5.0))

    def test_rung_capacity_matches_the_hand_calculation(self):
        b, h, fc = 16.0, 30.0, 4.0
        d = h - BEAM_GROWTH_DEPTH_OFFSET_IN
        area = BEAM_GROWTH_STEEL_RATIO * b * d
        a = area * 60.0 / (0.85 * fc * b)
        phi_mn = 0.9 * area * 60.0 * (d - a / 2.0)
        self.assertTrue(beam_rung_carries_moment((b, h, fc), phi_mn - 1.0, 60.0))
        self.assertFalse(beam_rung_carries_moment((b, h, fc), phi_mn + 1.0, 60.0))

    def test_the_jump_lands_on_the_first_rung_that_carries_the_moment(self):
        moment = 9200.0                                                       # kip-in; the long-span case's order
        index = suggest_beam_rung_for_moment(self.beams, self.start, moment, 60.0)
        self.assertTrue(beam_rung_carries_moment(self.beams[index], moment, 60.0))
        self.assertTrue(all(not beam_rung_carries_moment(self.beams[i], moment, 60.0) for i in range(self.start, index)))
        self.assertEqual(suggest_beam_rung_for_moment(self.beams, self.start, 1.0e9, 60.0), len(self.beams) - 1)
        self.assertEqual(suggest_beam_rung_for_moment(self.beams, self.start, 10.0, 60.0), self.start)

    def test_a_seed_bar_dcr_no_longer_sends_the_beam_to_the_top(self):
        """DCR 9.2 on two No. 6 bars: the proxy jump is the top rung, the moment names a far lighter one."""
        worst = {"beam": 9.22, "column": 0.5}
        proxy = suggest_rung_index(self.beams, self.start, 9.22, 0.85)
        self.assertEqual(proxy, len(self.beams) - 1)
        _c, old_beam, _r = driver._plan_next_rungs(self.columns, self.beams, 40, self.start, worst, 0.85, 1.0, 40, FLAGS)
        self.assertEqual(old_beam, proxy)                                     # without the moment the proxy stands
        _c, beam, reasons = driver._plan_next_rungs(self.columns, self.beams, 40, self.start,
                                                    {**worst, "beam_flexure_demand_kip_in": 9200.0}, 0.85, 1.0, 40, FLAGS)
        self.assertIn("beam_strength", reasons)
        self.assertLess(beam, proxy)
        self.assertEqual(beam, suggest_beam_rung_for_moment(self.beams, self.start, 9200.0, sp.FY_KSI))
        b, h, fc = self.beams[beam]
        self.assertLess(b * h * h * fc ** 0.5, 0.5 * 30.0 * 36.0 * 36.0 * 8.0 ** 0.5)

    def test_the_proxy_jump_stays_the_upper_bound(self):
        """A modest overstress: the proxy asks for less than the moment rule would, and is kept."""
        worst = {"beam": 1.05, "column": 0.5, "beam_flexure_demand_kip_in": 1.0e9}
        proxy = suggest_rung_index(self.beams, self.start, 1.05, 0.85)
        _c, beam, _r = driver._plan_next_rungs(self.columns, self.beams, 40, self.start, worst, 0.85, 1.0, 40, FLAGS)
        self.assertEqual(beam, proxy)

    def test_a_passing_beam_is_not_moved_by_the_rule(self):
        worst = {"beam": 0.85, "column": 0.5, "beam_flexure_demand_kip_in": 9200.0}
        _c, beam, reasons = driver._plan_next_rungs(self.columns, self.beams, 40, self.start, worst, 0.85, 1.0, 40, FLAGS)
        self.assertEqual(beam, self.start)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
